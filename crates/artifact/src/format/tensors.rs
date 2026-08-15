//! The safetensors container, both directions: the writer that emits the resident
//! artifact, and the mmap'd name-indexed reader over one file or a whole shard set.
//!
//! **One module rather than two, and the tests are the argument.** A writer's real output is
//! an OFFSET TABLE, and the only way to assert one is to read it back — every test below
//! writes with [`SafeWriter`] and reads with [`Safetensors`], including the two that exist
//! solely to pin the borrow paths' bytes against the owning path's. Splitting the halves
//! would have put each test in a file that cannot build its own fixture. They also share the
//! one type this format's correctness turns on, [`super::Dtype`]: the reader narrows to it
//! and refuses anything else, the writer names it per tensor, and an fp8 encoding slipping
//! past either end is silent, plausible, wrong output.

use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use std::borrow::Cow;
use std::collections::HashMap;

use super::Dtype;

// ── safetensors reader (fp8 source shards + the resident artifact) ──────────────

/// One tensor as the writer holds it: name, dtype, shape, and its bytes.
///
/// The bytes are a `Cow` so a verbatim copy can **borrow the source mmap** rather than be
/// `to_vec()`'d into host RAM — see [`SafeWriter`] for why that decides whether a K3
/// conversion is possible at all.
type Tensor<'a> = (String, Dtype, Vec<usize>, Cow<'a, [u8]>);

/// Minimal safetensors writer for the resident artifact — collects tensors, then serializes
/// `u64 header_len ‖ JSON header ‖ concatenated data`.
///
/// Each tensor's bytes are a `Cow`, so a verbatim copy **borrows the source mmap and costs no
/// host RAM**; only tensors a converter actually computes (a widened bf16→f32 norm, an
/// e8m0→f32 scale grid) are owned. That matters because the resident set is not always small:
/// GLM's is ~10 GiB and fitted in RAM, but **Kimi-K3's is ~113.5 GB** (108.81 of bf16 trunk
/// plus 4.70 of embed and lm_head) and does not, on a 128 GB box. Every large K3 tensor is a
/// verbatim copy, so host-RAM peak becomes the sum of the *converted* tensors.
///
/// The lifetime is the source [`Safetensors`]'s, so open the source before the writer — every
/// converter already does.
#[derive(Default)]
pub struct SafeWriter<'a> {
    tensors: Vec<Tensor<'a>>,
}

impl<'a> SafeWriter<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(
        &mut self,
        name: impl Into<String>,
        dtype: Dtype,
        shape: Vec<usize>,
        bytes: impl Into<Cow<'a, [u8]>>,
    ) {
        self.tensors.push((name.into(), dtype, shape, bytes.into()));
    }

    /// Add an fp8 weight and its f32 block scale under this engine's two resident names,
    /// `<name>.weight` and `<name>.weight_scale_inv`. Both `copy_fp8` (GLM: the scale is
    /// already f32) and `copy_fp8_e8m0` (V4: it is an e8m0 byte) end here, so the pair can
    /// never be written under one convention by one converter and another by the other.
    fn add_fp8_pair(
        &mut self,
        name: &str,
        weight: Cow<'a, [u8]>,
        shape: &[usize],
        scale: Cow<'a, [u8]>,
        sshape: &[usize],
    ) -> Result<()> {
        // Both payloads are now the same TYPE, where they used to be `&[u8]` and `Vec<u8>` —
        // so transposing them stopped being a compile error. These two checks put the barrier
        // back at add time: an fp8 weight is one byte per element and an f32 scale grid is
        // four, so a swap fails here rather than at load, a whole convert later.
        for (suffix, dtype, bpe, shape, payload) in [
            ("weight", Dtype::F8E4M3, 1usize, shape, weight),
            ("weight_scale_inv", Dtype::F32, 4, sshape, scale),
        ] {
            let want = shape.iter().product::<usize>() * bpe;
            ensure!(
                payload.len() == want,
                "{name}.{suffix}: {} bytes for shape {shape:?} at {bpe} B/elem — want {want}. \
                 A weight/scale transposition looks exactly like this.",
                payload.len()
            );
            self.tensors
                .push((format!("{name}.{suffix}"), dtype, shape.to_vec(), payload));
        }
        Ok(())
    }

    /// Copy an fp8 tensor (`<name>.weight` F8E4M3 + `.weight_scale_inv` F32) from a
    /// reader verbatim — the resident attn/dense/indexer projections.
    pub fn copy_fp8(&mut self, src: &'a Safetensors, name: &str) -> Result<()> {
        let (w, shape) = src.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
        let (sc, ssh) = src.typed(&format!("{name}.weight_scale_inv"), Dtype::F32)?;
        // Both halves are already what the loader wants, so both borrow the mmap.
        self.add_fp8_pair(name, w.into(), shape, sc.into(), ssh)
    }

    /// Copy a V4 fp8 tensor (`<name>.weight` F8E4M3 + `<name>.scale` **F8_E8M0**) into
    /// the resident set, widening only the SCALE: weight bytes verbatim, and each e8m0
    /// exponent byte to the f32 `<name>.weight_scale_inv` the resident path already reads.
    ///
    /// **Lossless, and provably so.** e8m0 is a bare exponent with value `2^(b-127)`; every
    /// b in 0..=254 is exactly representable in f32 (b=0 is 2^-127, an f32 subnormal but an
    /// exact one — 2^-149 is the smallest), and 0xFF is the format's NaN, which
    /// [`crate::quant::e8m0`] refuses rather than propagate into a whole block.
    /// So this is a re-encoding of the same numbers, not a requantization — which is the
    /// point: V4's attention is already fp8 at the 128×128 block size rivoli uses.
    ///
    /// The two conventions also AGREE on direction, which is the part that would be silent
    /// if it did not: `inference/kernel.py`'s `fp8_gemm` accumulates
    /// `C += C_local * scale_a * scale_b` (line 46) and `fp4_gemm` the same at line 509 —
    /// a multiplier, exactly as `quant::dequant_fp8_block` applies GLM's. Were V4's the
    /// reciprocal, every resident attention tensor would be off by `s²` with no error.
    ///
    /// The output is named `weight_scale_inv` rather than keeping the source's `.scale`
    /// because that is this engine's name for "the f32 block multiplier of an fp8 weight",
    /// and it is what `dequant_fp8` and `pin.rs` read. A tensor of a different dtype under
    /// the source's name would be the more confusing of the two.
    pub fn copy_fp8_e8m0(&mut self, src: &'a Safetensors, name: &str) -> Result<()> {
        let (w, shape) = src.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
        let (sc, ssh) = src.typed(&format!("{name}.scale"), Dtype::F8E8M0)?;
        let block = crate::quant::FP8_BLOCK;
        ensure!(
            shape.len() == 2,
            "{name}.weight: shape {shape:?} is not 2-D"
        );
        let want = [shape[0].div_ceil(block), shape[1].div_ceil(block)];
        ensure!(
            ssh == want,
            "{name}.scale: shape {ssh:?} != {want:?} for a {shape:?} weight at block {block}"
        );
        let f32b: Vec<u8> = sc
            .iter()
            .map(|&b| crate::quant::e8m0(b).map(f32::to_le_bytes))
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("{name}.scale"))?
            .concat();
        // The weight borrows; only the widened scale grid materializes.
        self.add_fp8_pair(name, w.into(), shape, f32b.into(), ssh)
    }

    /// Quantize a BF16 tensor to fp8-e4m3 at `block` and add it as the `<name>.weight` +
    /// `<name>.weight_scale_inv` pair the resident fp8 path reads. `name` is the BASE, as
    /// [`Self::copy_fp8_e8m0`] takes it.
    ///
    /// **The only converter path in this tree that quantizes rather than copies**, and
    /// `quant::quantize_fp8_block`'s doc carries the whole argument for why: GLM, DeepSeek-V4 and
    /// Kimi-K3 all ship a publisher's quantization to copy, Muse Glimmer ships bf16 and nothing
    /// else, so the scale choice here is rivoli's and answers to a dNLL measurement rather than to
    /// an upstream.
    ///
    /// **Owned output, unlike [`Self::copy_verbatim`]'s borrow.** One byte per weight plus the
    /// grid, so quantizing the shipped Glimmer's 416 projections holds ~25.2 GB before [`Self::write`]
    /// runs. `ponytail:` accepted rather than engineered around — this is a one-off offline
    /// conversion on a 128 GB host, and it is the only reason a bf16 convert of the same model
    /// costs no host memory at all. Upgrade path if it ever binds: `write` gains a streaming mode
    /// that emits each payload as it is produced instead of collecting them first.
    pub fn add_quantized_fp8(
        &mut self,
        src: &'a Safetensors,
        name: &str,
        block: usize,
    ) -> Result<()> {
        let wname = format!("{name}.weight");
        let (raw, shape) = src.typed(&wname, Dtype::Bf16)?;
        ensure!(shape.len() == 2, "{wname}: shape {shape:?} is not 2-D");
        let w: Vec<f32> = raw
            .chunks_exact(2)
            .map(|c| rivoli_core::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        let (packed, scale) = crate::quant::quantize_fp8_block(&w, [shape[0], shape[1]], block)
            .with_context(|| wname.clone())?;
        let sshape: Vec<usize> = shape.iter().map(|d| d.div_ceil(block)).collect();
        let sbytes: Vec<u8> = scale.iter().flat_map(|s| s.to_le_bytes()).collect();
        self.add_fp8_pair(name, packed.into(), shape, sbytes.into(), &sshape)
    }

    /// Copy a tensor verbatim — same dtype, same shape, same bytes. For the ones that are
    /// already what the loader wants (`attn_sink` and the `hc_*` tables are F32; `embed`
    /// and `head` stay BF16 because whether to requantize them is a QUALITY decision with
    /// a measurement attached, not a conversion detail to settle here).
    pub fn copy_verbatim(&mut self, src: &'a Safetensors, name: &str, dtype: Dtype) -> Result<()> {
        let (b, shape) = src.typed(name, dtype)?;
        // Borrowed, not copied: this is the path K3's 108.81 GB of bf16 trunk rides, and
        // `.to_vec()` here was what made a K3 conversion impossible on a 128 GB host.
        self.add(name, dtype, shape.to_vec(), b);
        Ok(())
    }

    /// Add a bf16 tensor from a reader, widened to f32 (norms, router gate,
    /// weights_proj, k_norm — everything the loader reads as f32).
    pub fn add_widened(&mut self, src: &'a Safetensors, name: &str) -> Result<()> {
        let (bytes, shape) = src.typed(name, Dtype::Bf16)?;
        let f32b: Vec<u8> = bytes
            .chunks_exact(2)
            .flat_map(|c| {
                rivoli_core::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes()
            })
            .collect();
        self.add(name, Dtype::F32, shape.to_vec(), f32b);
        Ok(())
    }

    pub fn write(&self, path: &str) -> Result<()> {
        use std::io::Write;
        let mut hdr = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, dtype, shape, payload) in &self.tensors {
            let begin = offset;
            offset += payload.len();
            hdr.insert(
                name.clone(),
                serde_json::json!({ "dtype": dtype.name(), "shape": shape, "data_offsets": [begin, offset] }),
            );
        }
        let hjson = serde_json::to_vec(&serde_json::Value::Object(hdr))?;
        // Write to a sibling and rename, for a reason the owning version did not have: a
        // borrowed payload is read HERE, not at `add` time, so `File::create(path)` on a path
        // that is also one of the mapped sources would truncate the mapping out from under
        // the very bytes about to be written — SIGBUS on the pages past the new EOF, a fatal
        // signal rather than an error, with the output left half-formed. `add_indexer --stash`
        // pointed at its own previous output is a plausible way to get there. The rename
        // keeps the old inode alive for any existing mmap and publishes all-or-nothing.
        let part = format!("{path}.{}.part", std::process::id());
        let file = std::fs::File::create(&part).with_context(|| format!("create {part}"))?;
        let mut f = std::io::BufWriter::new(file);
        f.write_all(&(hjson.len() as u64).to_le_bytes())?;
        f.write_all(&hjson)?;
        for (_, _, _, payload) in &self.tensors {
            f.write_all(payload)?;
        }
        f.flush()?;
        drop(f);
        std::fs::rename(&part, path).with_context(|| format!("rename {part} -> {path}"))
    }
}

/// One tensor's entry in the index: where its bytes are, and how to read them.
struct TensorDesc {
    shard: usize,
    begin: usize,
    len: usize,
    dtype: Dtype,
    shape: Vec<usize>,
}

/// Read-only mmap of one or more safetensors files, tensors indexed by name. Serves
/// the multi-shard fp8 source (converter) and the single resident file (loader).
pub struct Safetensors {
    mmaps: Vec<Mmap>,
    index: HashMap<String, TensorDesc>,
}

impl Safetensors {
    /// mmap + index every `*.safetensors` under `dir` (sorted).
    pub fn open_dir(dir: &str) -> Result<Self> {
        let mut paths: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("read dir {dir}"))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        ensure!(!paths.is_empty(), "no *.safetensors in {dir}");
        Self::open_files(&paths)
    }

    /// mmap + index one file.
    pub fn open_file(path: &str) -> Result<Self> {
        Self::open_files(std::slice::from_ref(&std::path::PathBuf::from(path)))
    }

    /// mmap + index a NAMED subset of a checkpoint's shards, resolved through its
    /// `model.safetensors.index.json`: every shard holding a tensor whose name `want`
    /// accepts, and no others.
    ///
    /// [`Self::open_dir`] takes the whole directory, which is wrong for a partial convert
    /// of a very large checkpoint for two reasons. It reads shards it will never touch —
    /// V4-Flash is 48 of them, one per layer — and, more sharply, it fails on ANY truncated
    /// shard. A checkpoint being fetched has truncated shards by definition, and refusing
    /// to convert layer 0 because layer 31 is still downloading is a self-inflicted stall.
    /// Selecting by name keeps the truncation check (a shard we DO need, still short, fails
    /// loudly and by name) while ignoring the ones we do not.
    pub fn open_indexed(dir: &str, want: impl Fn(&str) -> bool) -> Result<Self> {
        let ipath = format!("{dir}/model.safetensors.index.json");
        let idx: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&ipath).with_context(|| format!("read {ipath}"))?,
        )
        .with_context(|| format!("parse {ipath}"))?;
        let map = idx["weight_map"]
            .as_object()
            .with_context(|| format!("{ipath}: no weight_map"))?;
        let mut shards: Vec<String> = map
            .iter()
            .filter(|(name, _)| want(name))
            .filter_map(|(_, sh)| sh.as_str().map(str::to_string))
            .collect();
        shards.sort();
        shards.dedup();
        ensure!(
            !shards.is_empty(),
            "{ipath}: no tensor matched the requested selection"
        );
        let paths: Vec<std::path::PathBuf> =
            shards.iter().map(|s| format!("{dir}/{s}").into()).collect();
        Self::open_files(&paths)
    }

    fn open_files(paths: &[std::path::PathBuf]) -> Result<Self> {
        let mut mmaps = Vec::with_capacity(paths.len());
        let mut index = HashMap::new();
        for path in paths {
            let file = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
            // SAFETY: read-only for the reader's lifetime.
            let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {path:?}"))?;
            // `read_metadata`, not `SafeTensors::deserialize`: the latter hands back
            // `TensorView`s BORROWING the buffer, and `Self` owns the mmaps — that is the
            // self-referential struct this offset index exists to avoid. `read_metadata`
            // returns the header length and owned offsets, which is exactly what `TensorDesc` wants.
            //
            // It subsumes both truncation checks this used to make by hand, and is stricter
            // than either: it requires `8 + header + data == file length` EXACTLY, where the
            // hand-rolled pair only asked that the last tensor end at or before EOF. It also
            // checks contiguity and that each tensor's byte extent matches its own
            // shape × dtype, per tensor, which nothing here did.
            //
            // Why that matters, kept from the hand-rolled version: a TRUNCATED shard — a
            // download still in flight, an interrupted copy — carries a complete header
            // describing data that is not there yet. Unchecked, the file opens, indexes
            // cleanly, and every read past the end panics on a slice range deep inside a
            // converter. Measured 2026-08-04: the V4-Flash checkpoint was being fetched while
            // this code was written, and 21 of its 48 shards were absent or partial at any
            // given moment. The crate's error names neither the file nor that diagnosis, so
            // both are added back here — a correct check with an unreadable message gets
            // misread as corruption.
            //
            // The context offers truncation as ONE cause and not the only one, because
            // `read_metadata` also rejects a malformed header — bad offsets, a shape that
            // disagrees with its own extent. Naming only truncation would send a reader
            // hunting a download that finished fine. The crate's own reason is appended by
            // anyhow after this line, and it is the half that says which.
            let (hlen, meta) = safetensors::SafeTensors::read_metadata(&mmap)
                .map_err(|e| anyhow::anyhow!("{e}"))
                .with_context(|| {
                    format!(
                        "{path:?}: not a readable safetensors ({} bytes) — truncated, still \
                         downloading, or a malformed header",
                        mmap.len()
                    )
                })?;
            let base = 8 + hlen;
            let shard = mmaps.len();
            for (name, t) in meta.tensors() {
                let dtype = Dtype::narrow(t.dtype)
                    .with_context(|| format!("unsupported dtype for {name}"))?;
                let (b, e) = t.data_offsets;
                index.insert(
                    name,
                    TensorDesc {
                        shard,
                        begin: base + b,
                        len: e - b,
                        dtype,
                        shape: t.shape.clone(),
                    },
                );
            }
            mmaps.push(mmap);
        }
        Ok(Self { mmaps, index })
    }

    fn loc(&self, name: &str) -> Result<&TensorDesc> {
        self.index
            .get(name)
            .with_context(|| format!("tensor {name} not found"))
    }

    /// Bytes + dtype-check + shape of a tensor (fails loud on a dtype mismatch).
    pub fn typed(&self, name: &str, want: Dtype) -> Result<(&[u8], &[usize])> {
        let l = self.loc(name)?;
        ensure!(
            l.dtype == want,
            "{name}: dtype {:?}, expected {want:?}",
            l.dtype
        );
        Ok((&self.mmaps[l.shard][l.begin..l.begin + l.len], &l.shape))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        Ok(&self.loc(name)?.shape)
    }

    pub fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Every indexed tensor name, **sorted**.
    ///
    /// The sort is not cosmetic. `index` is a `HashMap`, so iteration order varies between runs;
    /// `SafeWriter` writes tensors in the order they were added, and a converter that enumerated
    /// in hash order would emit a byte-different `resident.safetensors` on every run from the
    /// same input — same tensors, same values, different layout. That defeats the one cheap check
    /// a repack has (`sha256` two runs and compare) and it would make a re-conversion look like a
    /// change. Sorted here rather than at each caller so there is no way to get it wrong.
    ///
    /// Added for `convert_k3`, whose resident pass is "copy everything that is not routed and not
    /// vision" — driven off what the checkpoint HAS rather than off a list of names this port
    /// believes in, because K3's trunk is entirely BF16 and later stages need tensors S1a does
    /// not read.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.index.keys().map(String::as_str).collect();
        v.sort_unstable();
        v
    }

    /// Bytes, dtype and shape, with **no dtype expectation** — for a verbatim copy.
    ///
    /// The only accessor here that does not check a dtype, and the only correct use is passing the
    /// bytes through unexamined. Anything that INTERPRETS them must use [`Self::typed`] and name
    /// the dtype it is about to assume; that check is what stopped an FNUZ fp8 tensor from being
    /// decoded as OCP e4m3 (see [`Dtype::narrow`]), and it is not one to route around.
    pub fn raw(&self, name: &str) -> Result<(&[u8], Dtype, &[usize])> {
        let l = self.loc(name)?;
        Ok((
            &self.mmaps[l.shard][l.begin..l.begin + l.len],
            l.dtype,
            &l.shape,
        ))
    }

    /// Dequantize a block-scaled fp8 projection (`<name>.weight` F8E4M3 +
    /// `<name>.weight_scale_inv` F32) to row-major f32 `[o_dim, i_dim]`. The one
    /// fp8-read used by both converters (`bin/convert` → `.vq3`, `bin/fp8_to_i4` →
    /// `.i4`), so the two cannot drift on the decode convention.
    ///
    /// Both shapes are checked here rather than trusted: `weight_scale_inv` is
    /// `[ceil(o/block), ceil(i/block)]` row-major, and a scale tensor of the wrong
    /// extent would otherwise mis-tile silently — a wrong-but-plausible dequant,
    /// which is far worse than a hard failure.
    pub fn dequant_fp8(&self, name: &str, shape: [usize; 2], block: usize) -> Result<Vec<f32>> {
        let [o_dim, i_dim] = shape;
        let (w, shape) = self.typed(&format!("{name}.weight"), Dtype::F8E4M3)?;
        ensure!(
            shape == [o_dim, i_dim],
            "{name}.weight: shape {shape:?} != [{o_dim},{i_dim}]"
        );
        let (sb, ssh) = self.typed(&format!("{name}.weight_scale_inv"), Dtype::F32)?;
        let want = [o_dim.div_ceil(block), i_dim.div_ceil(block)];
        ensure!(
            ssh == want,
            "{name}.weight_scale_inv: shape {ssh:?} != {want:?} (block {block})"
        );
        let scale = crate::quant::read_f32(sb);
        Ok(crate::quant::dequant_fp8_block(
            crate::quant::Fp8W {
                packed: w,
                scale: &scale,
                block,
            },
            [o_dim, i_dim],
        ))
    }
}

#[cfg(test)]
mod tests {
    // Every test here round-trips: it writes a file with one half and reads it with the
    // other, or hand-builds a header the reader must refuse. Crate-wide `unwrap`/`expect`
    // are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::super::fixtures::tmpdir;
    use super::*;

    #[test]
    fn safewriter_roundtrips_through_reader() {
        let dir = tmpdir("fmt_test");
        let path = format!("{dir}/r.safetensors");
        let mut w = SafeWriter::new();
        let a: Vec<u8> = (0..48u8).collect(); // F32 [2,6]
        let b: Vec<u8> = (100..108u8).collect(); // I8 [2,4]
        w.add("a", Dtype::F32, vec![2, 6], a.clone());
        w.add("b", Dtype::I8, vec![2, 4], b.clone());
        w.write(&path).unwrap();

        let st = Safetensors::open_file(&path).unwrap();
        assert_eq!(st.shape("a").unwrap(), &[2, 6]);
        let (bb, sh) = st.typed("b", Dtype::I8).unwrap();
        assert_eq!(bb, &b[..]);
        assert_eq!(sh, &[2, 4]);
        assert!(st.typed("a", Dtype::I8).is_err()); // dtype mismatch fails loud
        assert!(!st.has("missing"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The borrowing copy paths must write byte-identical output to the owning one.
    ///
    /// `SafeWriter` stopped `to_vec()`-ing verbatim tensors so a ~113.5 GB K3 resident set
    /// could be written on a 128 GB host. That is pure plumbing and must be invisible in the
    /// bytes: GLM's 675 GiB and V4's ~146 GiB of published artifacts are read back by offset,
    /// so a shifted `data_offsets` or a reordered pair invalidates both with no error.
    ///
    /// **What this catches, precisely** — worth stating because the obvious reading is wrong.
    /// Both arms share one `write()`, so a *symmetric* offset bug shifts them equally and the
    /// comparison alone would pass. It still reddens, but by a different route: the source is
    /// re-opened through `Safetensors`, whose `read_metadata` demands
    /// `8 + header + data == file length` and per-tensor extent agreement, so a malformed
    /// writer fails at `open_file` before any comparison. What the *comparison* uniquely
    /// catches is asymmetry between the borrowed and owned paths — a borrowed slice that
    /// under-copies, a length read from the wrong arm, or the fp8 pair emitted in the other
    /// order (the header is a map, but `data_offsets` follow insertion order, so a swap moves
    /// bytes while leaving every name and shape intact).
    ///
    /// Verified red on all three of those, and on a symmetric offset bug via the reader.
    #[test]
    fn borrowed_and_owned_payloads_write_the_same_bytes() {
        let dir = tmpdir("borrow_test");
        // One tensor per borrow path: `copy_verbatim` (K3's trunk rides this), `copy_fp8`
        // (both halves borrow), and `copy_fp8_e8m0` (MIXED — borrowed weight, owned widened
        // scale), which is what every V4 resident attention tensor rides.
        let src_path = format!("{dir}/src.safetensors");
        let bf16: Vec<u8> = (0..24u8).collect();
        let w8: Vec<u8> = (0..8u8).map(|b| b | 0x38).collect();
        let sc: Vec<u8> = (0..4u32)
            .flat_map(|i| (i as f32 + 1.0).to_le_bytes())
            .collect();
        // One e8m0 byte: `copy_fp8_e8m0` requires the grid to be ceil(shape/128) per dim,
        // so a [2,4] weight at block 128 takes exactly [1,1]. 129 = 2^2, and not the 0xFF NaN.
        let e8: Vec<u8> = vec![129];
        let mut s = SafeWriter::new();
        s.add("t", Dtype::Bf16, vec![3, 4], bf16.clone());
        s.add("p.weight", Dtype::F8E4M3, vec![2, 4], w8.clone());
        s.add("p.weight_scale_inv", Dtype::F32, vec![1, 4], sc);
        s.add("q.weight", Dtype::F8E4M3, vec![2, 4], w8.clone());
        s.add("q.scale", Dtype::F8E8M0, vec![1, 1], e8.clone());
        s.write(&src_path).unwrap();
        let src = Safetensors::open_file(&src_path).unwrap();

        // Arm 1: the borrow paths.
        let borrowed = format!("{dir}/borrowed.safetensors");
        let mut b = SafeWriter::new();
        b.copy_verbatim(&src, "t", Dtype::Bf16).unwrap();
        b.copy_fp8(&src, "p").unwrap();
        b.copy_fp8_e8m0(&src, "q").unwrap();
        b.write(&borrowed).unwrap();

        // Arm 2: the same tensors, same order, through the owning entry point — including
        // the e8m0 grid widened by hand, so the mixed pair is compared and not just written.
        let owned = format!("{dir}/owned.safetensors");
        let widened: Vec<u8> = e8
            .iter()
            .flat_map(|&x| crate::quant::e8m0(x).unwrap().to_le_bytes())
            .collect();
        let mut o = SafeWriter::new();
        o.copy_verbatim(&src, "t", Dtype::Bf16).unwrap();
        o.copy_fp8(&src, "p").unwrap();
        o.add("q.weight", Dtype::F8E4M3, vec![2, 4], w8);
        o.add("q.weight_scale_inv", Dtype::F32, vec![1, 1], widened);
        o.write(&owned).unwrap();

        assert_eq!(
            std::fs::read(&borrowed).unwrap(),
            std::fs::read(&owned).unwrap(),
            "borrowing changed the bytes on disk"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `dequant_fp8` must apply the right block scale to the right tile, and must
    /// REJECT a `weight_scale_inv` of the wrong extent — a silently mis-tiled scale
    /// grid is the failure this guard exists to prevent, and it produces plausible
    /// weights rather than an obvious error.
    #[test]
    fn dequant_fp8_tiles_scales_and_rejects_bad_shapes() {
        let dir = tmpdir("deqfp8_test");
        let path = format!("{dir}/w.safetensors");
        // A [2,4] fp8 matrix with block=2 → a [1,2] scale grid: the left 2 columns
        // scale by 10, the right 2 by 100. A row/column-swapped tiling gives
        // different answers here, so this pins the orientation.
        let mut w = SafeWriter::new();
        w.add(
            "t.weight",
            Dtype::F8E4M3,
            vec![2, 4],
            vec![0x38, 0x40, 0x38, 0xB8, 0xB8, 0x00, 0x40, 0x38], // 1,2,1,-1 / -1,0,2,1
        );
        w.add(
            "t.weight_scale_inv",
            Dtype::F32,
            vec![1, 2],
            [10.0f32, 100.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        w.write(&path).unwrap();
        let st = Safetensors::open_file(&path).unwrap();

        let got = st.dequant_fp8("t", [2, 4], 2).unwrap();
        assert_eq!(
            got,
            vec![10.0, 20.0, 100.0, -100.0, -10.0, 0.0, 200.0, 100.0]
        );

        // Wrong declared dims, and a scale grid of the wrong extent, both fail loud.
        assert!(st.dequant_fp8("t", [4, 2], 2).is_err());
        assert!(st.dequant_fp8("t", [2, 4], 1).is_err()); // would want a [2,4] grid
    }

    /// A truncated shard — a download in flight, an interrupted copy — must be refused by
    /// name, not indexed and then panicked on mid-read. Both the header's own extent and
    /// the last tensor's are checked, because a file can be long enough for one and not
    /// the other.
    #[test]
    fn a_truncated_shard_is_refused_rather_than_indexed() {
        let dir = tmpdir("trunc");
        let path = format!("{dir}/t.safetensors");
        let mut w = SafeWriter::new();
        w.add("t", Dtype::F32, vec![4], vec![0u8; 16]);
        w.write(&path).unwrap();
        let whole = std::fs::read(&path).unwrap();
        Safetensors::open_file(&path).expect("the intact file must open");

        for cut in [whole.len() - 8, 4] {
            std::fs::write(&path, &whole[..cut]).unwrap();
            let err = match Safetensors::open_file(&path) {
                Ok(_) => panic!("cut to {cut} still opened"),
                Err(e) => format!("{e:#}"),
            };
            assert!(
                err.contains("truncated") || err.contains("not a safetensors"),
                "cut to {cut}: {err}"
            );
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The reader stopped doing its own offset arithmetic on 2026-08-06 and the comment
    /// there claims the `safetensors` crate checks strictly MORE. That is the kind of claim
    /// this repo keeps finding to be false, so it is a test.
    ///
    /// The three rejected headers are each the plausible-but-wrong case rather than garbage:
    /// every one parses as JSON, names a real dtype, and is the right total length, so the
    /// old hand-rolled reader indexed all three and would have handed out a mis-shaped or
    /// mis-placed tensor. None of them was reachable by `--verify`, because a converter
    /// compares bytes it laid out itself.
    ///
    /// **The control arm is the point.** Without a well-formed file that must OPEN, three
    /// `is_err()`s prove only that the reader rejects things — a reader that rejected
    /// everything would pass. It also pins which property each arm violates: the control and
    /// the first case differ in exactly one integer.
    ///
    /// Each arm asserts the error it should get, not merely that there was one. Bare
    /// `is_err()` on all three would stay green if a single coarse check started catching
    /// everything — the arms would stop being three tests and nothing would say so. `shape`
    /// must fail on the extent-vs-dtype rule and the other two on placement, and the two
    /// placement arms name the tensor at fault, which is the only thing separating them.
    #[test]
    fn a_header_that_disagrees_with_its_own_shape_is_refused() {
        let dir = tmpdir("hdrmath");
        let path = format!("{dir}/t.safetensors");
        let write = |hdr: &str, data: usize| {
            let mut b = (hdr.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(hdr.as_bytes());
            b.resize(b.len() + data, 0u8);
            std::fs::write(&path, &b).unwrap();
        };

        write(
            r#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#,
            16,
        );
        Safetensors::open_file(&path).expect("the well-formed control must open");

        for (why, hdr, data, want) in [
            (
                "shape says 8 f32 = 32 bytes but the extent is 16",
                r#"{"t":{"dtype":"F32","shape":[8],"data_offsets":[0,16]}}"#,
                16,
                "invalid shape, data type, or offset for tensor",
            ),
            (
                "the only tensor does not start at offset 0",
                r#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[8,24]}}"#,
                24,
                "invalid offset for tensor `t`",
            ),
            (
                "two tensors leave an 8-byte hole between them",
                r#"{"a":{"dtype":"F32","shape":[4],"data_offsets":[0,16]},"b":{"dtype":"F32","shape":[4],"data_offsets":[24,40]}}"#,
                40,
                "invalid offset for tensor `b`",
            ),
        ] {
            write(hdr, data);
            let err = match Safetensors::open_file(&path) {
                Ok(_) => panic!("accepted: {why}"),
                Err(e) => format!("{e:#}"),
            };
            assert!(err.contains(want), "{why}: wanted {want:?}, got {err:?}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
