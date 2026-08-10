//! The on-disk artifact: the single source of truth for the layout the converter
//! writes and the loader reads. An artifact directory is:
//!
//! ```text
//! manifest.json          # HF config fields + a `format` section (FormatMeta)
//! codebooks.f32          # 3 per-projection codebooks (gate, up, down): VQ_K·VQ_DIM f32 each
//! resident.safetensors   # every resident weight (fp8 attn/dense, int8 embed, f32 norms, bf16 indexer)
//! L{03..NN}.vq3          # per MoE layer: header + (n_experts + 1) expert blocks,
//!                        #   block = gate‖up‖down; block n_experts = the shared expert
//! L{03..NN}.i4           # optional int4 twin of the .vq3 (bin/fp8_to_i4): headerless,
//!                        #   (n_experts + 1) blocks from offset 0. Streamed under --i4.
//! L{00..NN}.f4           # DeepSeek-V4-Flash (bin/convert_v4): header + exactly n_experts
//!                        #   FP4 blocks and NO shared block — V4's shared expert is fp8
//!                        #   e4m3 and rides `resident.safetensors`. See [`RoutedFmt`].
//! ```

use anyhow::{Context, Result, bail, ensure};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::HashMap;
use std::os::fd::{AsRawFd, RawFd};

use crate::artifact::quant::{
    VQ_ALIGN, VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VQ_K, f4_expert_bytes, f4_expert_stride,
    i4_expert_bytes, i4_expert_stride, vq_expert_bytes, vq_expert_stride,
};

/// The `format` section of `manifest.json` — everything the loader needs beyond
/// the HF config fields (which `ModelConfig` reads from the same file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatMeta {
    pub version: u32,
    pub vq_dim: usize,
    pub vq_k: usize,
    pub vq_index_bits: usize,
    pub vq_group: usize,
    /// fp8 `weight_scale_inv` tile size (128 for the GLM-5.2 checkpoint).
    pub fp8_block: usize,
}

impl FormatMeta {
    pub const VERSION: u32 = 1;

    /// The current build's parameters — what the converter stamps into the manifest.
    pub fn current(fp8_block: usize) -> Self {
        Self {
            version: Self::VERSION,
            vq_dim: VQ_DIM,
            vq_k: VQ_K,
            vq_index_bits: VQ_INDEX_BITS,
            vq_group: VQ_GROUP,
            fp8_block,
        }
    }

    /// Read `<dir>/manifest.json`'s `format` section and check it matches this build
    /// (VQ params are compiled into the kernels, so a mismatch is unrunnable).
    pub fn load(dir: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{dir}/manifest.json"))?)
                .with_context(|| format!("parse {dir}/manifest.json"))?;
        let m: FormatMeta = serde_json::from_value(v["format"].clone())
            .context("manifest.json missing/invalid `format` section")?;
        ensure!(
            m.version == Self::VERSION,
            "artifact format v{} != build v{}",
            m.version,
            Self::VERSION
        );
        ensure!(
            m.vq_dim == VQ_DIM
                && m.vq_k == VQ_K
                && m.vq_index_bits == VQ_INDEX_BITS
                && m.vq_group == VQ_GROUP,
            "artifact VQ params differ from the compiled-in kernel params"
        );
        // The fp8 GEMV kernels index the block scale with a SHIFT (`blk_shift`), so a
        // non-power-of-two tile is unrunnable. The kernel launchers reject it too (arg
        // guard 1003); catching it here turns a mid-decode HIP error into a startup
        // message that names the offending value.
        ensure!(
            m.fp8_block > 0 && m.fp8_block.is_power_of_two(),
            "artifact fp8_block ({}) must be a power of two",
            m.fp8_block
        );
        Ok(m)
    }
}

// ── safetensors reader (fp8 source shards + the resident artifact) ──────────────

/// A tensor's dtype in a safetensors header (only the ones this engine uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    U8,
    I8,
    I64,
    Bf16,
    F8E4M3,
    /// A bare 8-bit exponent, `2^(b-127)`. DeepSeek-V4-Flash's block scales — 128×128 on
    /// the fp8 attention tensors, one per 32 weights along the input dim on the FP4
    /// experts. Decoded by [`crate::artifact::quant::e8m0`].
    F8E8M0,
}

impl Dtype {
    /// Narrow the `safetensors` crate's dtype to the ones this engine decodes.
    ///
    /// This enum stays rather than becoming a re-export because the crate's is
    /// `#[non_exhaustive]` with ~20 variants: every `match` on it would need a `_` arm, and
    /// the point of a closed seven-variant enum is that adding a dtype is a COMPILE error at
    /// each decode site instead of a runtime fallthrough. The narrowing happens once, here.
    ///
    /// **That is not hypothetical, and 0.8.0 is why.** Its two additions over 0.7.0 are
    /// `F8_E4M3FNUZ` and `F8_E5M2FNUZ` — the AMD/Graphcore fp8 encodings, a different
    /// exponent bias and no signed zero, and this engine runs on AMD. The `_` arm below
    /// REFUSES them. A re-export would instead have let an FNUZ tensor reach
    /// `quant::dequant_fp8_block`, which decodes OCP e4m3 unconditionally: every weight off by
    /// a power of two, no error anywhere, fluent wrong output. The bytes do not say which
    /// encoding they are — only the dtype string does — so this match is the whole check.
    fn narrow(d: safetensors::Dtype) -> Option<Dtype> {
        Some(match d {
            safetensors::Dtype::F32 => Dtype::F32,
            safetensors::Dtype::U8 => Dtype::U8,
            safetensors::Dtype::I8 => Dtype::I8,
            safetensors::Dtype::I64 => Dtype::I64,
            safetensors::Dtype::BF16 => Dtype::Bf16,
            safetensors::Dtype::F8_E4M3 => Dtype::F8E4M3,
            safetensors::Dtype::F8_E8M0 => Dtype::F8E8M0,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Dtype::F32 => "F32",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I64 => "I64",
            Dtype::Bf16 => "BF16",
            Dtype::F8E4M3 => "F8_E4M3",
            Dtype::F8E8M0 => "F8_E8M0",
        }
    }
}

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
    /// [`crate::artifact::quant::e8m0`] refuses rather than propagate into a whole block.
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
        let block = crate::artifact::quant::FP8_BLOCK;
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
            .map(|&b| crate::artifact::quant::e8m0(b).map(f32::to_le_bytes))
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("{name}.scale"))?
            .concat();
        // The weight borrows; only the widened scale grid materializes.
        self.add_fp8_pair(name, w.into(), shape, f32b.into(), ssh)
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
            .flat_map(|c| crate::math::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])).to_le_bytes())
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

    /// Dequantize a block-scaled fp8 projection (`<name>.weight` F8E4M3 +
    /// `<name>.weight_scale_inv` F32) to row-major f32 `[o_dim, i_dim]`. The one
    /// fp8-read used by both converters (`bin/convert` → `.vq3`, `bin/fp8_to_i4` →
    /// `.i4`), so the two cannot drift on the decode convention.
    ///
    /// Both shapes are checked here rather than trusted: `weight_scale_inv` is
    /// `[ceil(o/block), ceil(i/block)]` row-major, and a scale tensor of the wrong
    /// extent would otherwise mis-tile silently — a wrong-but-plausible dequant,
    /// which is far worse than a hard failure.
    pub fn dequant_fp8(
        &self,
        name: &str,
        o_dim: usize,
        i_dim: usize,
        block: usize,
    ) -> Result<Vec<f32>> {
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
        let scale = crate::artifact::quant::read_f32(sb);
        Ok(crate::artifact::quant::dequant_fp8_block(
            w, &scale, o_dim, i_dim, block,
        ))
    }
}

// ── `.f4` repack (DeepSeek-V4-Flash's native FP4 routed experts) ────────────────
//
// V4 ships its routed experts ALREADY at 4 bits: `<proj>.weight` is `I8[o, i/2]` holding
// e2m1 nibble pairs, `<proj>.scale` is `F8_E8M0[o, i/32]`. rivoli's `.f4` block is the
// same nibbles and the same exponent bytes at O_DIRECT-aligned offsets — a REPACK, with
// nothing fit, nothing re-rounded, and no error introduced.
//
// Two facts make the copy a plain `memcpy` rather than a transcode. Both are read from the
// reference; neither is CHECKED here, and saying so matters because a copy cannot detect
// either being wrong:
//   * NIBBLE ORDER. torch's `float4_e2m1fn_x2` and `inference/convert.py:31-33`
//     (`low = x & 0x0F; high = (x >> 4) & 0x0F; stack([TABLE[low], TABLE[high]])`) put
//     logical element 2j in the LOW nibble of byte j — the convention `quant::matvec_i4`
//     already reads. A repack under the opposite convention is the same byte copy, so this
//     becomes checkable only when a `matvec_f4` exists to decode one. That is S3's, and it
//     is the check to add there.
//   * The SCALE GRID. `inference/kernel.py:468` declares
//     `scales_b: T.Tensor[(N, ceildiv(K, 32))]` indexed `[n, k]` — `[o_dim, f4_groups]`
//     row-major, which IS checked, by the shape guard in `F4Expert::spans`.

/// One V4 routed expert, located in a source checkpoint: which tensors it is made of and
/// what shape they must have.
///
/// A struct rather than four positional arguments repeated at every entry point, for the
/// reason `SetDims` gives: nothing about a transposed `(expert_in, moe_inter)` fails a check —
/// an expert's three projections are `(moe_inter, expert_in)·2 + (expert_in, moe_inter)`, so the
/// swap produces exactly the same byte count and streams the wrong weights. Named fields
/// make the swap visible at the call site instead of invisible inside an argument list.
pub struct F4Expert<'a> {
    pub src: &'a Safetensors,
    /// `layers.{l}.ffn.experts.{e}` — see `quant::v4_expert_base`.
    pub base: String,
    pub expert_in: usize,
    pub moe_inter: usize,
}

impl F4Expert<'_> {
    /// This expert's six source spans, paired with their byte offset inside an `.f4`
    /// block: `(w1, w1.scale, w3, w3.scale, w2, w2.scale)` — see
    /// [`crate::artifact::quant::V4_PROJ`] for why that is gate/up/down order.
    ///
    /// ONE definition of the layout, used by both [`Self::pack`] and [`Self::diff`]. That
    /// is deliberate: what the verifier must prove is that the VALUES pass through
    /// untouched, and re-deriving the offsets in both places would only test that a copy
    /// of the arithmetic agrees with itself. The layout is pinned separately, by
    /// `f4_expert_bytes` (size) and by [`ExpertHeader`] (dims), and a wrong layout changes
    /// the file's length.
    fn spans(&self) -> Result<Vec<(usize, &[u8])>> {
        let (expert_in, moe_inter, base) = (self.expert_in, self.moe_inter, &self.base);
        // The offsets come from `f4_slot_offsets` — the SAME function the streaming pool
        // points its `ExpertDescF4` at (`memory::routed::TierFmt`). This used to walk the
        // spans and accumulate `off` itself, which made the writer and the reader two
        // implementations of one layout: a shifted scale span would have been written and
        // read consistently by rivoli and disagreed with nothing until the kernel decoded a
        // projection against another one's exponents. Now a change to the layout moves both
        // ends or neither.
        let off = crate::artifact::quant::f4_slot_offsets(expert_in, moe_inter);
        let mut out = Vec::with_capacity(6);
        for (p, (proj, (o_dim, i_dim))) in
            crate::artifact::quant::v4_expert_projs(expert_in, moe_inter)
                .into_iter()
                .enumerate()
        {
            // Shapes are checked, not trusted: `[o, i/2]` and `[o, i/32]` are the only pair
            // the byte counts below are correct for, and a transposed or mis-blocked source
            // would otherwise copy the right NUMBER of bytes in the wrong order — a file
            // that passes every length check and decodes to noise.
            let (w, wsh) = self
                .src
                .typed(&format!("{base}.{proj}.weight"), Dtype::I8)?;
            ensure!(
                wsh == [o_dim, i_dim / 2],
                "{base}.{proj}.weight: shape {wsh:?} != [{o_dim},{}] (FP4 nibble pairs)",
                i_dim / 2
            );
            let (sc, ssh) = self
                .src
                .typed(&format!("{base}.{proj}.scale"), Dtype::F8E8M0)?;
            let groups = crate::artifact::quant::f4_groups(i_dim);
            ensure!(
                ssh == [o_dim, groups],
                "{base}.{proj}.scale: shape {ssh:?} != [{o_dim},{groups}] (one e8m0 per {} \
                 weights along the input dim)",
                crate::artifact::quant::F4_GROUP
            );
            let wb = o_dim * crate::artifact::quant::f4_row_bytes(i_dim);
            ensure!(
                w.len() == wb && sc.len() == o_dim * groups,
                "{base}.{proj}: source spans are shorter than their shapes"
            );
            // **The e8m0 NaN check, and this is the one place in the engine that can make
            // it.** `0xff` is the format's NaN. The kernel decodes it correctly
            // (`common.hpp::e8m0f` returns a quiet NaN rather than `2^128`) but cannot
            // REFUSE it, and `moe_fixed`'s saturating clamp then launders the NaN into a
            // finite ±2^14 — so one bad byte is 32 weights of plausible garbage with no
            // error anywhere. `docs/investigations/v4-flash-port.md` §S3 requirement 10.
            //
            // It runs HERE, at repack, because this is the only path that reads every
            // routed scale byte: at decode the bytes DMA from NVMe straight into the pool
            // slot and the host never sees them. Measured on the shipped 43-layer set
            // (9,261,023,232 scale bytes): 9 distinct codes, all in `0x76..=0x7e`
            // (2^-9..2^-1), zero `0x00` and zero `0xff` — so this guard is green on every
            // artifact that exists, and the only thing that has made it speak is the
            // injection in this file's own
            // `an_e8m0_nan_scale_byte_is_refused_at_repack_and_a_subnormal_one_is_not`.
            //
            // `0x00` is deliberately NOT refused. It is `2^-127`, a legal encoding that
            // `e8m0f` and `quant::e8m0` both decode exactly (f32 carries it as a
            // subnormal); refusing it would be inventing a rule the format does not have.
            // The reason it is worth a sentence is that `b << 23` WOULD hand back +0, which
            // is why both decoders special-case it.
            if let Some(k) = sc.iter().position(|&b| b == 0xff) {
                bail!(
                    "{base}.{proj}.scale[{}][{}] is 0xff — the e8m0 NaN. The FP4 kernels \
                     cannot reject it and `moe_fixed`'s clamp turns it into a finite \
                     ±2^14, so a whole {}-weight group would decode to plausible garbage.",
                    k / groups,
                    k % groups,
                    crate::artifact::quant::F4_GROUP,
                );
            }
            out.push((off[p * 2], w));
            out.push((off[p * 2 + 1], sc));
        }
        // No `off[5] + …== f4_expert_bytes(…)` assertion here: `slot_offsets` derives both
        // from the same per-projection byte counts, so it can never fire. The check that
        // CAN is `pack`'s, on the buffer it was handed.
        Ok(out)
    }

    /// Repack into `dst` (`f4_expert_bytes` long). A byte copy — nothing is fit, nothing
    /// is re-rounded, and no error is introduced.
    pub fn pack(&self, dst: &mut [u8]) -> Result<()> {
        let want = crate::artifact::quant::f4_expert_bytes(self.expert_in, self.moe_inter);
        ensure!(
            dst.len() == want,
            "{}: destination is {} bytes, an expert block is {want}",
            self.base,
            dst.len()
        );
        for (off, bytes) in self.spans()? {
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        }
        Ok(())
    }

    /// Byte offsets within `block` that disagree with the source tensors. **Empty means
    /// the repack was bit-exact**; anything else names where it was not.
    ///
    /// Returns the offsets rather than a bool so a caller can say WHICH bytes moved —
    /// which is what lets `convert_v4 --verify` name where a 3.4 GB layer file went wrong.
    /// Note that `diff` shares `spans()` with [`Self::pack`], so against a block derived
    /// from `pack` it is a tautology; its value is against a block read back from DISK,
    /// which is the only way `--verify` uses it.
    ///
    /// **Sharing `spans()` also means `diff` inherits its e8m0 `0xff` refusal, so on a source
    /// carrying one it returns `Err` instead of a byte list.** Deliberate: a source with an
    /// e8m0 NaN is unusable whether or not the bytes round-tripped, and "your source has a
    /// NaN scale at w3[7][2]" is the more actionable of the two reports. `convert_v4
    /// --verify` propagates it and never reaches its "N bytes differ" summary. Nothing
    /// observable changes on the shipped artifact — measured zero `0xff` across all
    /// 9,261,023,232 of its scale bytes.
    pub fn diff(&self, block: &[u8]) -> Result<Vec<usize>> {
        let mut bad = Vec::new();
        for (off, want) in self.spans()? {
            let got = block
                .get(off..off + want.len())
                .with_context(|| format!("{}: block shorter than the source spans", self.base))?;
            if got != want {
                bad.extend(
                    (0..want.len())
                        .filter(|&k| got[k] != want[k])
                        .map(|k| off + k),
                );
            }
        }
        Ok(bad)
    }
}

/// Host memory one call to [`write_expert_layer`] may hold, regardless of the layer's size.
///
/// 1 GiB, chosen against the two things it trades off. It must stay large RELATIVE TO one
/// expert block (not a multiple of one — it is a multiple of none of the four strides) so
/// [`crate::artifact::quant::fill_expert_blocks`] still has more blocks per window than it has
/// threads. Blocks per 1 GiB window, against ~32 threads:
///
/// | format | expert bytes = stride | per window |
/// |---|---:|---:|
/// | GLM `.vq3` 6144/2048 | 15,335,424 | 70 |
/// | GLM `.i4` 6144/2048 | 20,054,016 | 53 |
/// | V4 `.f4` 4096/2048 | 13,369,344 | 80 |
/// | K3 `.f4` 3584/3072 | 17,547,264 | 61 |
///
/// And it must stay small next to the machine: this box is 128 GB of LPDDR5 *shared with the
/// GPU*, `/tmp` is a 63 GB tmpfs living in that same RAM, and a convert runs alongside whatever
/// else holds the arena.
pub const LAYER_WINDOW: usize = 1 << 30;

/// The header has to fit the block-0 pad it is written into. Both are compile-time constants,
/// so this is a compile-time check — it was briefly an `ensure!` inside
/// [`write_expert_layer`], which is a runtime test of `40 <= 4096` on every layer.
const _: () = assert!(EXPERT_HEADER_BYTES <= VQ_ALIGN);

/// Write one layer's expert file — header block, then `blocks` expert blocks — in bounded host
/// memory, published to `path` by a rename from `<path>.<pid>.part`.
///
/// **Bounded, because the buffered form does not scale past GLM.** Both converters used to
/// allocate the whole file (`vec![0u8; VQ_ALIGN + blocks * stride]`), fill it, and write it in
/// one call: 3.42 GB per layer for V4 (256 blocks) and 3.94 GB for GLM (257 — its shared expert
/// rides the same slab), survivable, but **15.72 GB for Kimi-K3** (896 x 17,547,264) on a box
/// whose entire LPDDR5 is 128 GB and shared with the GPU. The buffer was pure waste at any
/// size — every byte was written out once and never re-read.
/// Parallelism survives because the window is the unit, not the expert:
/// `fill_expert_blocks` still packs each window across all threads over disjoint slices, so
/// serialising the pack to save memory was never on the table.
///
/// **Atomic, because both converters resume by SKIPPING an output path that already exists,
/// without reading it** (`bin/convert` `continue`s, `convert_v4` sets `reused`). A non-atomic
/// write plus a kill would leave a short multi-GB `L{ll}.{ext}` that re-running the tool can
/// never repair: the artifact fails at load on `open_routed`'s `ensure!(len == want)`, and the
/// fix is a manual `rm` of a file nobody would suspect. Found 2026-08-06 because `convert_v4`
/// had tmp+rename from the day it was written and `bin/convert` did not — two hand-written
/// copies of one loop, and the single step where they diverged was the one carrying the defect.
///
/// **The temp name carries the pid, and that is not decoration** — self-review 2026-08-07, the
/// same lesson `bin/ppl`, `f4_loading`, `v4_encoding` and `v4_oracle` each learned on their own
/// scratch paths. Agents share this machine and a convert takes no lock (it is CPU only; the
/// GPU flock does not serialise it), so two runs into one `out_dir` are reachable. On a FIXED
/// `<path>.part` both would `File::create` + truncate + write concurrently, and the rename
/// would publish interleaved bytes. Interleaving two writes **of equal length** yields a file
/// of exactly the right length, so `open_routed`'s length check passes it — the one corruption
/// shape that gets past the loader. The cost is that a killed run leaves
/// `L{ll}.{ext}.<pid>.part` behind rather than a name the next run overwrites, which is the
/// better failure: multi-GB debris under an obviously non-artifact name is visible.
///
/// **No `fsync` before the rename**, deliberately: the guarantee is against process death, not
/// power loss, and one fsync per 3.5 GB layer buys a property no converter has ever claimed.
/// `I4Source::stamp` DOES fsync its manifest, and that asymmetry is right — a torn manifest is
/// unrecoverable, a torn layer file is regenerable.
///
/// **`fill` must write all `bytes` of the slot it is handed.** This is a real obligation, not a
/// formality, and it is stronger than what the buffered form required. The whole-layer `vec!`
/// gave every block fresh zeros, so a closure that wrote only part of its slot left the rest
/// `0x00`; the reused window hands it **the previous expert's payload**. Both current closures
/// are total — `encode_expert` advances `off` by `vq_proj_bytes` across exactly three
/// projections, and `F4Expert::pack` copies six spans that tile `[0, f4_expert_bytes)` — but
/// the tree already contains a helper built to tolerate a short write:
/// [`crate::artifact::quant::write_le_scales`] "stop[s] at whichever of the two runs out".
///
/// Concretely, the shape to avoid: a scale iterator one group short (`f4_groups` is
/// `div_ceil`, or a format that reads scales from the source instead of computing them) leaves
/// the tail of that projection's scale span holding **another expert's e8m0 exponents**. Right
/// file length, every length check passes, and `--verify` compares through the same `spans()`
/// so it never looks there. Under the buffered writer the identical bug wrote `0x00` = a dead
/// group, which is visible. `bin/fp8_to_i4` states this same requirement correctly for its own
/// reused buffer; the debug-only clear below keeps dev builds behaving like the old writer.
///
/// The `bytes..stride` padding is a weaker and separate matter: nothing writes it at all, so it
/// stays zero from the single allocation — see the comment there.
///
/// `window` is the host-memory ceiling and both converters pass [`LAYER_WINDOW`]. It is a
/// parameter rather than a constant read from inside so a test can reach the window BOUNDARY
/// without allocating a gigabyte; a thin wrapper supplying the constant would be a second copy
/// of this parameter list, which `jscpd` refuses (measured: 6 lines, 45 tokens).
pub fn write_expert_layer(
    path: &str,
    header: &[u8; EXPERT_HEADER_BYTES],
    stride: usize,
    bytes: usize,
    blocks: usize,
    window: usize,
    fill: impl Fn(usize, &mut [u8]) -> Result<()> + Sync,
) -> Result<u64> {
    use std::io::Write;
    // A `stride` of 0 would reach `chunks_exact_mut(0)` and PANIC rather than return, and
    // `window / stride` would divide by zero one line further down. Refused, not clamped: the
    // `.max(1)` this replaced turned a caller's bad dimension into a panic in someone else's
    // function. `bytes <= stride` is `fill_expert_blocks`'s own check and is left to it.
    ensure!(
        stride > 0,
        "expert stride is 0 — no block geometry to write"
    );
    let part = format!("{path}.{}.part", std::process::id());
    // Not a `BufWriter`. Every write below is either the one `VQ_ALIGN` header block or a whole
    // window, and `BufWriter` passes any write at or above its capacity straight through, so it
    // would buffer nothing and its flush would guard nothing.
    let mut f = std::fs::File::create(&part).with_context(|| format!("create {part}"))?;
    // Block 0 is the header, padded to `VQ_ALIGN` so expert 0 starts block-aligned for the
    // loader's O_DIRECT reads. Same layout the buffered writer produced.
    let mut pad = [0u8; VQ_ALIGN];
    pad[..EXPERT_HEADER_BYTES].copy_from_slice(header);
    f.write_all(&pad).with_context(|| format!("write {part}"))?;

    let per = (window / stride).clamp(1, blocks.max(1));
    // Zeroed ONCE, not per window, and the reuse is safe for a specific reason:
    // `fill_expert_blocks` hands each closure `&mut slot[..bytes]`, so the `bytes..stride`
    // padding is never written by anybody and stays zero for the buffer's whole life — the
    // same way it did in the whole-layer `vec!` this replaced. A per-window `fill(0)` was here
    // first, justified as stopping one window's tail leaking into the next window's padding;
    // the red-proof for that showed the test stayed GREEN without it, because no path dirties
    // padding at all. It was a memset of up to 1 GiB per window (~16 GiB per K3 layer)
    // defending against nothing.
    let mut win = vec![0u8; per * stride];
    for start in (0..blocks).step_by(per) {
        let span = &mut win[..per.min(blocks - start) * stride];
        // Dev-profile only, and NOT for the padding — for the DATA region. It costs nothing in
        // release and makes a `fill` that writes less than `bytes` degrade the way it did under
        // the whole-layer buffer: zeros, a visibly dead group, instead of the previous expert's
        // payload read as this one's. See the `fill` obligation in the doc comment. This is
        // insurance, not a check — it cannot report the short write, only defuse it.
        #[cfg(debug_assertions)]
        span.fill(0);
        crate::artifact::quant::fill_expert_blocks(
            span,
            stride,
            bytes,
            span.len() / stride,
            |j, slot| fill(start + j, slot),
        )?;
        f.write_all(span).with_context(|| format!("write {part}"))?;
    }
    // `File` has no user-space buffer to flush; the bytes are in the kernel by here, which is
    // all the rename needs (see the fsync paragraph above).
    drop(f);
    std::fs::rename(&part, path).with_context(|| format!("rename {part} -> {path}"))?;
    Ok((VQ_ALIGN + blocks * stride) as u64)
}

/// Write `<out_dir>/manifest.json` and copy `aux` (tokenizer and friends) beside it, so
/// the artifact is self-contained. The last step of every converter.
///
/// A missing aux file is a WARNING rather than an error: the artifact is still loadable
/// without `generation_config.json`, and failing a multi-hour convert on its absence at
/// the very end would be the worse trade. A missing manifest is not survivable, so that
/// one propagates.
pub fn finish_artifact(
    tool: &str,
    out_dir: &str,
    src_dir: &str,
    manifest: &serde_json::Value,
    aux: &[&str],
) -> Result<()> {
    let path = format!("{out_dir}/manifest.json");
    std::fs::write(&path, serde_json::to_vec_pretty(manifest)?)
        .with_context(|| format!("write {path}"))?;
    for name in aux {
        match std::fs::copy(format!("{src_dir}/{name}"), format!("{out_dir}/{name}")) {
            Ok(_) => eprintln!("{tool}: copied {name}"),
            Err(e) => eprintln!("{tool}: WARNING: {name} not copied ({e})"),
        }
    }
    Ok(())
}

/// Provenance of the artifact's `.i4` expert set: which tool produced it, from what,
/// and over which layers. Absent on artifacts built before this field existed — and
/// that absence is itself the signal, since a `vq3_to_i4` set and an `fp8_to_i4` set are
/// otherwise byte-indistinguishable on disk, which is exactly how a bad `.i4` set
/// stayed invisible.
///
/// EVERY writer of `L{l}.i4` must call [`I4Source::stamp`]. A stale stamp is worse
/// than no stamp: the engine reports it as fact, so a tool that rewrites the set
/// without restamping turns an honest ambiguity into a confident lie.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct I4Source {
    /// Binary that wrote the set, e.g. `"fp8_to_i4"`.
    pub tool: String,
    /// Derivation chain, e.g. `"fp8->int4"` vs the older `"fp8->vq3->int4"`.
    pub chain: String,
    /// Path of the source the weights were derived from.
    pub src: String,
    /// Half-open layer range this provenance covers, `[from, to)`. A run that
    /// rebuilds only part of the set records only that part, so a mixed artifact
    /// never claims to be uniform.
    pub layers: [usize; 2],
    /// [`crate::artifact::quant::I4_GROUP`] the set was quantized at — weights per f32 scale
    /// along the input dim. Without it a G=128 set and a G=64 set are the same
    /// `tool`/`chain`/`src` triple, and the quality difference between them would be
    /// unattributable. `None` is an artifact predating group scales (per-row); such a
    /// set has a different `.i4` file size and is rejected by `ExpertSet::open`, so
    /// this is a diagnosis, not a load-time guard.
    #[serde(default)]
    pub group: Option<usize>,
}

impl I4Source {
    /// Read `<dir>/manifest.json`'s `i4_source` section. `Ok(None)` means "no
    /// manifest, or no such field" — an unstamped artifact, which is a reportable
    /// fact rather than an error. A field that is PRESENT but unparseable is an
    /// error: silently reporting it as "unstamped" would hide a real corruption.
    pub fn load(dir: &str) -> Result<Option<Self>> {
        let Ok(text) = std::fs::read(format!("{dir}/manifest.json")) else {
            return Ok(None);
        };
        let v: serde_json::Value =
            serde_json::from_slice(&text).with_context(|| format!("parse {dir}/manifest.json"))?;
        let Some(f) = v.get("i4_source") else {
            return Ok(None);
        };
        Ok(Some(
            serde_json::from_value(f.clone()).context("manifest i4_source is malformed")?,
        ))
    }

    /// Record this provenance in `<dir>/manifest.json`, merging with an existing
    /// stamp when the two describe the same derivation over adjoining layers — so a
    /// run resumed with `--from` still ends up claiming the whole set it rebuilt,
    /// rather than only its own final leg.
    ///
    /// Written tmp→fsync→rename: a torn `manifest.json` bricks an artifact whose
    /// `.i4` set alone is ~365 GB.
    pub fn stamp(&self, dir: &str) -> Result<()> {
        let path = format!("{dir}/manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("read {path}"))?)
                .with_context(|| format!("parse {path}"))?;
        // An unreadable prior stamp is not an error HERE — we are replacing it. Only
        // the reader is strict, so a corrupt field can never be misreported as fact.
        let merged = match Self::load(dir).ok().flatten() {
            // Same derivation and the ranges touch → one contiguous claim. `group` is
            // part of "same derivation": merging a G=64 run into a G=128 run would
            // claim one uniform set for two incompatible formats.
            Some(p)
                if (p.tool.as_str(), p.chain.as_str(), p.src.as_str(), p.group)
                    == (&self.tool, &self.chain, &self.src, self.group)
                    && p.layers[0] <= self.layers[1]
                    && self.layers[0] <= p.layers[1] =>
            {
                Self {
                    layers: [
                        p.layers[0].min(self.layers[0]),
                        p.layers[1].max(self.layers[1]),
                    ],
                    ..self.clone()
                }
            }
            _ => self.clone(),
        };
        m["i4_source"] = serde_json::to_value(&merged)?;
        let tmp = format!("{path}.tmp");
        let mut f = std::fs::File::create(&tmp).with_context(|| format!("create {tmp}"))?;
        {
            use std::io::Write;
            f.write_all(&serde_json::to_vec_pretty(&m)?)?;
        }
        f.sync_all().with_context(|| format!("fsync {tmp}"))?;
        drop(f);
        std::fs::rename(&tmp, &path).with_context(|| format!("rename {tmp} -> {path}"))?;
        Ok(())
    }
}

/// The layer range an `.f4` artifact actually HOLDS, from `manifest.json`'s `f4_source`,
/// confronted with the model's own layer count.
///
/// **Read, never inferred from `num_hidden_layers`.** `convert_v4` deliberately does not
/// rewrite that field — its comment says why: every per-layer table in a V4 config
/// (`compress_ratios`, `num_hash_layers`) is indexed by the REAL layer id, so a 3-layer
/// artifact that renumbered itself as a 3-layer MODEL would mis-key all of them. A partial
/// artifact of a 43-layer model is therefore normal, and `/var/db/rivoli/v4-f4-l0-2` is
/// exactly that. Without this the loader walks to layer 3 and dies with
/// "tensor layers.3.attn_norm.weight not found", which reads like a corrupt checkpoint
/// rather than a partial artifact — the failure `convert_v4`'s own resident-set comment
/// predicts.
///
/// Absent or malformed is an ERROR, unlike [`I4Source::load`]'s `Ok(None)`: `.i4`
/// provenance diagnoses an artifact that loads either way, while this is the loader's only
/// source for **which layers exist**, and `0..n_layers` is precisely the wrong guess for
/// the partial artifact the field exists to describe. `to > n_layers` is a manifest whose
/// two halves were written by different runs; `from >= to` is an artifact holding nothing.
///
/// One function rather than a type with a `load` and a `range`: the two were only ever
/// called together, in that order, and the pair's only other caller was a test.
pub fn f4_layer_range(dir: &str, n_layers: usize) -> Result<std::ops::Range<usize>> {
    /// Just the field the loader needs. `tool`/`chain`/`src` are provenance for a human;
    /// a field with no reader is how [`I4Source::group`] nearly became decoration.
    #[derive(Deserialize)]
    struct F4Source {
        layers: [usize; 2],
    }

    let path = format!("{dir}/manifest.json");
    let mut v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("read {path}"))?)
            .with_context(|| format!("parse {path}"))?;
    let f = v
        .get_mut("f4_source")
        .map(serde_json::Value::take)
        .with_context(|| format!("{path} has no `f4_source` — not a convert_v4 artifact"))?;
    let [from, to] = serde_json::from_value::<F4Source>(f)
        .with_context(|| format!("{path}: f4_source malformed"))?
        .layers;
    ensure!(
        from < to && to <= n_layers,
        "{path}: f4_source.layers [{from}, {to}) is not a non-empty range inside \
         [0, num_hidden_layers={n_layers})"
    );
    Ok(from..to)
}

// ── .vq3 / .i4 / .f4 expert files (streamed routed experts + resident shared) ────

/// Per-file header for a routed-expert layer file. Little-endian, 40 bytes, at the start
/// of the file. Self-describing: a dim/version mismatch or truncation fails loud on open.
///
/// **The dims are here because nothing else catches a transposition.** An expert's three
/// projections are `(moe_inter, expert_in) · 2 + (expert_in, moe_inter)`, so swapping the two
/// widths gives a file of EXACTLY the same size that passes every length check and streams
/// the wrong bytes. Which formats carry one, and why `.i4` does not, is
/// [`RoutedFmt::magic`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ExpertHeader {
    /// [`VQ3_MAGIC`] or [`F4_MAGIC`] — which format's blocks follow.
    pub magic: [u8; 4],
    pub version: u32,
    pub layer: u32,
    /// ROUTED experts. `.vq3` holds `n_experts + 1` blocks (the last is the shared
    /// expert); `.f4` holds exactly `n_experts`, because V4's shared expert is fp8 e4m3
    /// and cannot share a file with FP4 blocks (see `quant::v4_expert_base`).
    pub n_experts: u32,
    pub expert_in: u32,
    pub moe_inter: u32,
    pub stride: u64, // per-expert O_DIRECT-aligned block stride
    pub reserved: u64,
}

pub const VQ3_MAGIC: [u8; 4] = *b"VQ3\0";
pub const F4_MAGIC: [u8; 4] = *b"FP4\0";
pub const EXPERT_HEADER_BYTES: usize = 40;

impl ExpertHeader {
    /// A header for one layer file. `stride` is the value the WRITER indexes blocks with,
    /// passed in rather than re-derived here: re-deriving would let the header disagree
    /// with the file it describes and still look self-consistent.
    pub fn new(
        magic: [u8; 4],
        layer: usize,
        n_experts: usize,
        expert_in: usize,
        moe_inter: usize,
        stride: usize,
    ) -> Self {
        Self {
            magic,
            version: FormatMeta::VERSION,
            layer: layer as u32,
            n_experts: n_experts as u32,
            expert_in: expert_in as u32,
            moe_inter: moe_inter as u32,
            stride: stride as u64,
            reserved: 0,
        }
    }

    /// Serialize to the 40-byte on-disk header.
    pub fn to_bytes(&self) -> [u8; EXPERT_HEADER_BYTES] {
        let mut b = [0u8; EXPERT_HEADER_BYTES];
        b[0..4].copy_from_slice(&self.magic);
        b[4..8].copy_from_slice(&self.version.to_le_bytes());
        b[8..12].copy_from_slice(&self.layer.to_le_bytes());
        b[12..16].copy_from_slice(&self.n_experts.to_le_bytes());
        b[16..20].copy_from_slice(&self.expert_in.to_le_bytes());
        b[20..24].copy_from_slice(&self.moe_inter.to_le_bytes());
        b[24..32].copy_from_slice(&self.stride.to_le_bytes());
        b[32..40].copy_from_slice(&self.reserved.to_le_bytes());
        b
    }

    /// Parse and validate a layer file's header against the format the CALLER is reading
    /// for.
    ///
    /// The magic is a parameter rather than a constant because it is the only thing that
    /// separates a `.vq3` from an `.f4` up front, and the two differ in BLOCK COUNT as well
    /// as in content: reading one as the other addresses a whole expert stride past the end,
    /// or stops one short. It is the discriminant the length check alone cannot be.
    ///
    /// `fmt` rather than a `(magic, extension)` pair, so the two cannot be mismatched by a
    /// caller. A headerless format simply has no magic to check ([`RoutedFmt::magic`]).
    /// Takes the array, not a slice: the length check this used to carry became a
    /// compile-time truth once both callers switched to `[u8; EXPERT_HEADER_BYTES]` (the
    /// 40-byte read below, and `to_bytes()` in the tests). A guard that cannot fire is
    /// worse than none, so the type carries the length instead.
    fn from_bytes(b: &[u8; EXPERT_HEADER_BYTES], fmt: RoutedFmt) -> Result<Self> {
        let h = Self {
            magic: b[0..4].try_into()?,
            version: u32::from_le_bytes(b[4..8].try_into()?),
            layer: u32::from_le_bytes(b[8..12].try_into()?),
            n_experts: u32::from_le_bytes(b[12..16].try_into()?),
            expert_in: u32::from_le_bytes(b[16..20].try_into()?),
            moe_inter: u32::from_le_bytes(b[20..24].try_into()?),
            stride: u64::from_le_bytes(b[24..32].try_into()?),
            reserved: u64::from_le_bytes(b[32..40].try_into()?),
        };
        if let Some(want) = fmt.magic() {
            ensure!(
                h.magic == want,
                "expected .{} magic {want:?}, file has {:?}",
                fmt.ext(),
                h.magic
            );
        }
        ensure!(
            h.version == FormatMeta::VERSION,
            ".{} version {} != {}",
            fmt.ext(),
            h.version,
            FormatMeta::VERSION
        );
        Ok(h)
    }
}

/// Which container a routed-expert set is opened as.
///
/// A `bool` (`i4: true/false`) while there were two, and the bool could not have carried
/// the property that actually separates the third: **`.f4` holds `n_experts` blocks and NO
/// shared one.** V4's shared expert is `F8_E4M3` at 128×128, not FP4 (see
/// [`crate::artifact::quant::v4_expert_base`]), so it cannot share a file with FP4 blocks —
/// it rides `resident.safetensors` instead. Verified against the shipped artifact:
/// `L00.f4` is `4096 + 256 × 13369344 = 3422556160` bytes exactly, so the `n_experts + 1`
/// this replaced demanded 13.37 MB that is not in the file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoutedFmt {
    Vq3,
    I4,
    F4,
}

impl RoutedFmt {
    /// The per-layer file's extension. Public because it is also how a routed FORMAT
    /// names itself in a log line or an error — `RoutedPool` reports its tier with it.
    pub fn ext(self) -> &'static str {
        match self {
            RoutedFmt::Vq3 => "vq3",
            RoutedFmt::I4 => "i4",
            RoutedFmt::F4 => "f4",
        }
    }

    /// The magic the file's header must carry, or `None` for the headerless `.i4`.
    /// `.i4` gets away without one only because it is a twin of a `.vq3` that was already
    /// validated against the same dims; `.vq3` and `.f4` are standalone and carry one.
    fn magic(self) -> Option<[u8; 4]> {
        match self {
            RoutedFmt::Vq3 => Some(VQ3_MAGIC),
            RoutedFmt::I4 => None,
            RoutedFmt::F4 => Some(F4_MAGIC),
        }
    }

    /// Byte offset of block 0. Derived from [`Self::magic`] rather than tabulated beside
    /// it: "has a header" and "reserves an aligned block for it" are the same fact.
    fn hbytes(self) -> usize {
        match self.magic() {
            Some(_) => crate::artifact::quant::VQ_ALIGN,
            None => 0,
        }
    }

    /// Whether a shared-expert block follows the `n_experts` routed ones.
    ///
    /// **The one definition of that fact.** [`ExpertSet::open_routed`] sizes the file with it and
    /// [`ExpertSet::shared_block`] refuses without it, so a format cannot be sized for a
    /// shared block it does not have, or refuse to hand back one it does. Before `.f4` this
    /// lived only as a hard-coded `n_experts + 1` inside the length check — which is
    /// exactly the pair the old `from_bytes` error told S3 to relax together.
    fn has_shared(self) -> bool {
        match self {
            RoutedFmt::Vq3 | RoutedFmt::I4 => true,
            RoutedFmt::F4 => false,
        }
    }

    /// The six byte offsets of this format's projections inside one expert block, in
    /// descriptor field order. Beside [`Self::geometry`] because the block SIZE and the
    /// layout INSIDE it are one fact about a format — and because **no downstream check can
    /// catch them being paired wrongly.** `.f4` and `.i4` tile a block identically for 25% of
    /// all `i_dim` (`ceil(i/32) == 4·ceil(i/128)`, i.e. `i mod 128 ∈ {0} ∪ {97..127}`), both
    /// models' dimensions included; `quant::f4_slot_offsets` has the arithmetic. Derived here,
    /// so the pairing is unrepresentable rather than merely warned about.
    fn slot_offsets(self, expert_in: usize, moe_inter: usize) -> [usize; 6] {
        use crate::artifact::quant::{f4_slot_offsets, i4_slot_offsets, vq_slot_offsets};
        match self {
            RoutedFmt::Vq3 => vq_slot_offsets(expert_in, moe_inter),
            RoutedFmt::I4 => i4_slot_offsets(expert_in, moe_inter),
            RoutedFmt::F4 => f4_slot_offsets(expert_in, moe_inter),
        }
    }

    /// `(per-expert O_DIRECT-aligned stride, useful bytes in one block)`.
    fn geometry(self, expert_in: usize, moe_inter: usize) -> (usize, usize) {
        match self {
            RoutedFmt::Vq3 => (
                vq_expert_stride(expert_in, moe_inter),
                vq_expert_bytes(expert_in, moe_inter),
            ),
            RoutedFmt::I4 => (
                i4_expert_stride(expert_in, moe_inter),
                i4_expert_bytes(expert_in, moe_inter),
            ),
            RoutedFmt::F4 => (
                f4_expert_stride(expert_in, moe_inter),
                f4_expert_bytes(expert_in, moe_inter),
            ),
        }
    }
}

/// The dimensions an expert set is opened against. `n_layers` is EXCLUSIVE and is the
/// CALLER's bound, not the artifact's — the pin opens one layer past `cfg.n_layers` when
/// the MTP head has an expert file, and the V4 pin bounds it by the artifact's own
/// `f4_source` range ([`f4_layer_range`]) rather than by `num_hidden_layers`.
///
/// A struct rather than five positional `usize`s, because nothing about a transposed
/// `(expert_in, moe_inter)` fails a check: the set opens, every length matches, and it streams
/// the wrong bytes. Five bare `usize`s in a row is the argument list that gets transposed.
#[derive(Clone, Copy)]
pub struct SetDims {
    /// First layer with a file, and one past the last — [`SetDims::new`] takes them as the
    /// `Range` they are and splits them here only to keep `SetDims` `Copy`. GLM's starts
    /// after the dense prefix and may run one past `cfg.n_layers` for the MTP head; V4's is
    /// the artifact's own `f4_source` range ([`f4_layer_range`]), and V4 has no dense
    /// layers at all.
    pub first_layer: usize,
    pub n_layers: usize,
    pub n_experts: usize,
    pub expert_in: usize,
    pub moe_inter: usize,
}

impl SetDims {
    /// Build from the layer range and the expert-block dims.
    ///
    /// Positional, against the "named fields make a transposition visible" argument above: the
    /// struct literal was character-identical at both engine call sites bar the range, which
    /// `jscpd` refuses.
    ///
    /// **That trade got worse on 2026-08-10 and this is the note saying so.** It used to be
    /// covered by every call site reading `…hidden, …moe_inter` — argument text matching
    /// parameter name, so a swap was visible anyway. After `hidden` became `expert_in` the
    /// callers still pass `cfg.hidden` (`pin.rs`, `tests/f4_loading.rs`), which is correct for
    /// GLM and V4 because for them the two widths are one number, but the textual match is
    /// gone. A K3 site written `SetDims::new(range, cfg.n_experts, cfg.hidden, cfg.moe_inter)`
    /// is exactly the substitution this struct exists to prevent and nothing at the call site
    /// flags it. The K3 pin is the third call site; once it exists the literals are no longer
    /// character-identical, so the `jscpd` objection lapses and `new` should go back to taking
    /// the struct.
    pub fn new(
        layers: std::ops::Range<usize>,
        n_experts: usize,
        expert_in: usize,
        moe_inter: usize,
    ) -> Self {
        Self {
            first_layer: layers.start,
            n_layers: layers.end,
            n_experts,
            expert_in,
            moe_inter,
        }
    }
}

/// The per-layer expert files, opened O_DIRECT — one type for every routed format
/// ([`RoutedFmt`]). Routed experts (`0..n_experts`) stream via [`ExpertSet::read_spec`];
/// where the format carries a shared block (`.vq3`/`.i4`, block `n_experts`) it is read
/// once for resident placement via [`ExpertSet::shared_block`]. An `.i4` block is
/// gate‖gate_scale‖up‖up_scale‖down‖down_scale; an `.f4` block is w1‖w1_scale‖w3‖w3_scale‖
/// w2‖w2_scale, the same gate/up/down slot order (see [`F4Expert::spans`]).
pub struct ExpertSet {
    files: Vec<std::fs::File>, // O_DIRECT, index = layer - first_layer
    first_layer: usize,
    n_layers: usize,
    n_experts: usize,
    stride: usize,
    expert_bytes: usize,
    hbytes: usize, // block 0 starts at hbytes (aligned): VQ_ALIGN for .vq3/.f4, 0 for .i4
    /// Carried from [`RoutedFmt::has_shared`] so [`Self::shared_block`] and the length
    /// check in [`Self::open_routed`] can never disagree about whether block `n_experts`
    /// exists.
    has_shared: bool,
    /// Which format these blocks are, and where its six projections sit inside one — both
    /// resolved at open, from the same [`RoutedFmt`] that sized the file, so a tier cannot
    /// be built against another format's layout ([`RoutedFmt::slot_offsets`]).
    fmt: RoutedFmt,
    slot_offsets: [usize; 6],
}

impl ExpertSet {
    /// Open the routed expert set in one format. See [`RoutedFmt`] — extension,
    /// block-start offset, header magic, block count and block geometry are all read
    /// from it.
    ///
    /// ONE opener rather than one per format, because those are the entire difference and
    /// the pin chooses between them at RUNTIME — separate entry points meant writing the
    /// identical dimension list once each, and nothing about a transposed
    /// `(expert_in, moe_inter)` fails a length check: every set opens and streams the wrong
    /// bytes.
    pub fn open_routed(dir: &str, fmt: RoutedFmt, d: SetDims) -> Result<Self> {
        let SetDims {
            first_layer,
            n_layers,
            n_experts,
            expert_in,
            moe_inter,
        } = d;
        // Refused rather than left to underflow: `n_layers - first_layer` below is a
        // `usize` subtraction, so an inverted range panicked inside `Vec::with_capacity`
        // instead of naming the range. `open_vq3` takes loose arguments and cannot be
        // relied on to have checked.
        ensure!(
            first_layer <= n_layers,
            "layer range [{first_layer}, {n_layers}) is inverted"
        );
        let (stride, expert_bytes) = fmt.geometry(expert_in, moe_inter);
        let (hbytes, has_shared) = (fmt.hbytes(), fmt.has_shared());
        // Routed blocks, plus the shared one only where the format has one. Both terms come
        // from `fmt`, so the block count and `shared_block`'s refusal cannot disagree.
        let want = hbytes + (n_experts + usize::from(has_shared)) * stride;
        let mut files = Vec::with_capacity(n_layers - first_layer);
        for l in first_layer..n_layers {
            let path = format!("{dir}/L{l:02}.{}", fmt.ext());
            let f = open_direct(&path).with_context(|| format!("open {path}"))?;
            let len = f.metadata()?.len() as usize;
            ensure!(len == want, "{path}: {len} bytes, expected {want}");
            if fmt.magic().is_some() {
                // Header via a separate BUFFERED fd — the O_DIRECT one belongs to the
                // streamer, and 40 bytes at offset 0 is neither length- nor
                // buffer-aligned.
                //
                // Read EXACTLY the header. This was `std::fs::read(path)`, which pulls the
                // WHOLE layer file through the page cache to look at 40 bytes of it — and
                // the cost was not small. **Measured on the GLM arm 2026-08-05**, both
                // binaries built before either ran and the two alternated three times with
                // no `cargo build` between them (`tests/artifact.rs::artifact_reads_back`
                // over `/var/db/rivoli/glm52-vq3-full`, 76 `.vq3` layers, medians):
                //
                //     std::fs::read   180.1 s   298.49 GB read from the block device
                //     40-byte pread     0.038 s   0.48 MB
                //
                // ~4700x, and it is startup on EVERY run — a one-time cost amortizes
                // against nothing. V4 pays the same shape at 43 x 3,422,556,160 = 147 GB.
                // A pre-existing defect in shipped GLM code that the `.f4` reader walked
                // into, not one `.f4` introduced.
                let mut raw = [0u8; EXPERT_HEADER_BYTES];
                let mut hf = std::fs::File::open(&path)
                    .with_context(|| format!("open {path} for its header"))?;
                std::io::Read::read_exact(&mut hf, &mut raw)
                    .with_context(|| format!("read {path} header"))?;
                let h = ExpertHeader::from_bytes(&raw, fmt)?;
                // `stride` is checked too, and it was not before. The converter writes the
                // value it INDEXED BLOCKS WITH (`ExpertHeader::new`'s doc says why it is
                // passed rather than re-derived) while `RoutedFmt::geometry` re-derives it
                // here — so without this conjunct the header's one non-redundant field had
                // no reader, and a writer whose stride disagreed with this build's would
                // pass every check on a file of the right total length.
                ensure!(
                    h.layer as usize == l
                        && h.n_experts as usize == n_experts
                        && h.expert_in as usize == expert_in
                        && h.moe_inter as usize == moe_inter
                        && h.stride as usize == stride,
                    // `expert_in (hidden_size)` rather than bare `expert_in`: whoever reads this
                    // is holding a config.json and an artifact, and `expert_in` is a name that
                    // appears in neither. For GLM and V4 the value IS `hidden_size`; on K3 it
                    // is `routed_expert_hidden_size`, which is why both are named.
                    "{path}: header (layer {} experts {} expert_in {} moe_inter {} stride {}) \
                     disagrees with config (layer {l} experts {n_experts} \
                     expert_in [hidden_size / routed_expert_hidden_size] {expert_in} \
                     moe_inter [moe_intermediate_size] {moe_inter} stride {stride})",
                    h.layer,
                    h.n_experts,
                    h.expert_in,
                    h.moe_inter,
                    h.stride
                );
            }
            files.push(f);
        }
        Ok(Self {
            files,
            first_layer,
            n_layers,
            n_experts,
            stride,
            expert_bytes,
            hbytes,
            has_shared,
            fmt,
            slot_offsets: fmt.slot_offsets(expert_in, moe_inter),
        })
    }

    /// Which routed format this set holds.
    pub fn fmt(&self) -> RoutedFmt {
        self.fmt
    }

    /// The six projection offsets inside one of this set's expert blocks
    /// ([`RoutedFmt::slot_offsets`]).
    pub fn slot_offsets(&self) -> [usize; 6] {
        self.slot_offsets
    }

    /// The layer range this set holds, absolute ids, end-exclusive.
    pub fn layers(&self) -> std::ops::Range<usize> {
        self.first_layer..self.n_layers
    }

    /// Routed experts per layer. Excludes the shared block where the format has one.
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }

    /// The int3-VQ set by loose dimensions — `tests/artifact.rs` opens the shipped `.vq3`
    /// this way, naming each dimension at the call site. The engine's own format is a
    /// runtime choice, so it goes through [`ExpertSet::open_routed`] directly.
    pub fn open_vq3(
        dir: &str,
        first_layer: usize,
        n_layers: usize,
        n_experts: usize,
        expert_in: usize,
        moe_inter: usize,
    ) -> Result<Self> {
        let d = SetDims::new(first_layer..n_layers, n_experts, expert_in, moe_inter);
        Self::open_routed(dir, RoutedFmt::Vq3, d)
    }

    /// Cold-read spec for a routed expert: `(fd, begin, useful_len)`, `begin` aligned.
    ///
    /// **Known-thin coverage on the GLM side, recorded 2026-08-05 and deliberately not
    /// fixed here.** `begin` is a function of the EXPERT only, so the layer→file mapping
    /// `files[layer - first_layer]` is observable solely through which fd comes back — and
    /// `tests/artifact.rs` calls `read_spec(dense_layers, 0)` and nothing else, so a wrong
    /// mapping survives it. `tests/f4_loading.rs`'s non-zero-start case is the one test that
    /// pins this, by resolving each fd through `/proc/self/fd` and asserting the FILENAME.
    /// That instrument was arrived at the hard way: two injected wrong mappings passed a
    /// distinct-fds check and an offset check first, because `layer % files.len()` is
    /// *identical* to `layer - first_layer` for a 3-layer artifact. Widening it to the GLM
    /// set is a GLM-side change and belongs to whoever owns `tests/artifact.rs`.
    pub fn read_spec(&self, layer: usize, expert: usize) -> Result<(RawFd, usize, usize)> {
        ensure!(
            (self.first_layer..self.n_layers).contains(&layer),
            "layer {layer} out of MoE range"
        );
        ensure!(
            expert < self.n_experts,
            "expert {expert} >= {}",
            self.n_experts
        );
        let fd = self.files[layer - self.first_layer].as_raw_fd();
        Ok((fd, self.hbytes + expert * self.stride, self.expert_bytes))
    }

    /// Read the shared expert's block (index `n_experts`) for resident placement, via a
    /// one-time buffered aligned pread.
    ///
    /// Refuses on a format that has no such block. That refusal is **not** redundant with
    /// the EOF the read would otherwise hit: an `.f4` ends at exactly
    /// `hbytes + n_experts·stride`, so `read_exact_at` there fails only by the accident of
    /// the file stopping where it does — an `.f4` with any trailing bytes would return
    /// garbage and call it a shared expert. The check is on the FORMAT, which is the fact.
    pub fn shared_block(&self, layer: usize) -> Result<Vec<u8>> {
        use std::os::unix::fs::FileExt;
        ensure!(
            self.has_shared,
            "this expert set has no shared block: it holds {} routed experts and stops \
             there. V4's shared expert is fp8 e4m3 at 128x128, not FP4 — it is in \
             `resident.safetensors`, not in the `.f4`",
            self.n_experts
        );
        ensure!(
            (self.first_layer..self.n_layers).contains(&layer),
            "layer {layer} out of MoE range"
        );
        let off = self.hbytes + self.n_experts * self.stride;
        // O_DIRECT needs aligned buffer+offset+len; read the full aligned stride.
        let f = &self.files[layer - self.first_layer];
        let mut aligned = vec![0u8; self.stride];
        f.read_exact_at(&mut aligned, off as u64)
            .with_context(|| format!("read shared block layer {layer}"))?;
        aligned.truncate(self.expert_bytes);
        Ok(aligned)
    }

    pub fn expert_slot(&self) -> usize {
        self.stride
    }
}

/// Open a file O_DIRECT (page-cache-bypassing NVMe DMA).
fn open_direct(path: &str) -> Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    Ok(std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)?)
}

/// Load the 3 per-projection codebooks from `<dir>/codebooks.f32` (gate, up, down),
/// each `VQ_K·VQ_DIM` f32, concatenated in that order.
pub fn load_codebooks(dir: &str) -> Result<[Vec<f32>; 3]> {
    let raw = crate::artifact::quant::read_f32(&std::fs::read(format!("{dir}/codebooks.f32"))?);
    ensure!(
        raw.len() == 3 * VQ_K * VQ_DIM,
        "codebooks.f32: {} f32, expected {}",
        raw.len(),
        3 * VQ_K * VQ_DIM
    );
    let n = VQ_K * VQ_DIM;
    Ok([
        raw[..n].to_vec(),
        raw[n..2 * n].to_vec(),
        raw[2 * n..].to_vec(),
    ])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

    use super::*;

    /// A fresh, empty scratch directory for one test. Every test here needs one and each
    /// had spelled out the same four lines; a shared helper also guarantees the
    /// remove-then-create, which a test that only created would inherit stale files from.
    fn tmpdir(tag: &str) -> String {
        let dir = std::env::temp_dir()
            .join(format!("rivoli_{tag}"))
            .to_string_lossy()
            .to_string();
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn vq3_header_roundtrips() {
        let h = ExpertHeader::new(VQ3_MAGIC, 7, 256, 6144, 2048, vq_expert_stride(6144, 2048));
        let back = ExpertHeader::from_bytes(&h.to_bytes(), RoutedFmt::Vq3).unwrap();
        assert_eq!(back.magic, VQ3_MAGIC);
        assert_eq!(back.layer, 7);
        assert_eq!(back.n_experts, 256);
        assert_eq!(back.expert_in, 6144);
        assert_eq!(back.moe_inter, 2048);
        assert_eq!(back.stride, vq_expert_stride(6144, 2048) as u64);
        // a corrupt magic must fail
        let mut bad = h.to_bytes();
        bad[0] = b'X';
        assert!(ExpertHeader::from_bytes(&bad, RoutedFmt::Vq3).is_err());
        // …and so must a WELL-FORMED header of the other format. This is the check the
        // length test cannot make: a `.vq3` and an `.f4` of the same nominal dims differ
        // in block count, so accepting either magic addresses past the last expert.
        assert!(ExpertHeader::from_bytes(&h.to_bytes(), RoutedFmt::F4).is_err());
        let f4 = ExpertHeader::new(F4_MAGIC, 7, 256, 6144, 2048, f4_expert_stride(6144, 2048));
        assert!(ExpertHeader::from_bytes(&f4.to_bytes(), RoutedFmt::Vq3).is_err());
        // Bidirectional: the correct pairing still parses, so the two refusals above are
        // the magic and not a header that stopped round-tripping.
        assert_eq!(
            ExpertHeader::from_bytes(&f4.to_bytes(), RoutedFmt::F4)
                .unwrap()
                .magic,
            F4_MAGIC
        );
    }

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
            .flat_map(|&x| crate::artifact::quant::e8m0(x).unwrap().to_le_bytes())
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

        let got = st.dequant_fp8("t", 2, 4, 2).unwrap();
        assert_eq!(
            got,
            vec![10.0, 20.0, 100.0, -100.0, -10.0, 0.0, 200.0, 100.0]
        );

        // Wrong declared dims, and a scale grid of the wrong extent, both fail loud.
        assert!(st.dequant_fp8("t", 4, 2, 2).is_err());
        assert!(st.dequant_fp8("t", 2, 4, 1).is_err()); // would want a [2,4] grid
    }

    /// Provenance round-trips, a missing field reads as unstamped, a malformed one is
    /// an error (not silently "unstamped"), and a resumed run merges into one range.
    #[test]
    fn i4_source_round_trips_and_merges_adjoining_runs() {
        let dir = tmpdir("i4src_test");
        let mf = format!("{dir}/manifest.json");
        std::fs::write(&mf, br#"{"hidden_size":6144}"#).unwrap();
        assert!(I4Source::load(&dir).unwrap().is_none()); // no field yet

        let a = I4Source {
            tool: "fp8_to_i4".into(),
            chain: "fp8->int4".into(),
            src: "/src".into(),
            layers: [3, 40],
            group: Some(crate::artifact::quant::I4_GROUP),
        };
        a.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().as_ref(), Some(&a));
        // The stamp must not clobber the rest of the manifest.
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&mf).unwrap()).unwrap();
        assert_eq!(v["hidden_size"], 6144);

        // A resume picking up where the first run stopped claims the whole range.
        I4Source {
            layers: [40, 78],
            ..a.clone()
        }
        .stamp(&dir)
        .unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().unwrap().layers, [3, 78]);

        // A different derivation replaces rather than merges — no stale claims.
        let b = I4Source {
            chain: "fp8->vq3->int4".into(),
            layers: [3, 78],
            ..a.clone()
        };
        b.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap(), Some(b.clone()));

        // A different GROUP is a different derivation too: rebuilding the tail at
        // G=64 must not merge into a G=128 claim over the head, or the manifest would
        // describe one uniform set where two incompatible formats sit side by side.
        let c = I4Source {
            group: Some(64),
            layers: [40, 78],
            ..b.clone()
        };
        c.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap(), Some(c));

        // An artifact stamped before group scales existed reads as group: None.
        std::fs::write(
            &mf,
            br#"{"i4_source":{"tool":"t","chain":"c","src":"/s","layers":[0,1]}}"#,
        )
        .unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().unwrap().group, None);

        // Present-but-malformed is an error, never a silent "unstamped".
        std::fs::write(&mf, br#"{"i4_source":{"tool":"x"}}"#).unwrap();
        assert!(I4Source::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── the `.f4` repack ───────────────────────────────────────────────────────────

    /// A tiny V4-shaped FP4 expert on disk. Dims are the smallest multiples of `F4_GROUP`
    /// that give a projection more than one group, so a group-index error has somewhere to
    /// show; `w1` and `w3` are given different content so a slot swap is not invisible.
    struct F4Fixture {
        dir: String,
        expert_in: usize,
        moe_inter: usize,
    }

    impl F4Fixture {
        fn new(tag: &str) -> Self {
            Self::with_scale_byte(tag, None)
        }

        /// `poison = Some((slot, k, b))` overwrites projection `slot`'s scale byte `k` with
        /// `b` — the one-field perturbation the e8m0 cases below need, so the control and
        /// the break differ in exactly one byte and nothing else.
        fn with_scale_byte(tag: &str, poison: Option<(usize, usize, u8)>) -> Self {
            use crate::artifact::quant::{F4_GROUP, f4_groups, f4_row_bytes, v4_expert_projs};
            let dir = tmpdir(&format!("f4_{tag}"));
            let (expert_in, moe_inter) = (64, 32);
            let mut w = SafeWriter::new();
            for (slot, (proj, (o_dim, i_dim))) in v4_expert_projs(expert_in, moe_inter)
                .into_iter()
                .enumerate()
            {
                // `tag` is '1' | '3' | '2', which keeps the three projections distinct —
                // in particular w1 != w3, which have identical shapes.
                let t = usize::from(proj.as_bytes()[1]);
                let weight: Vec<u8> = (0..o_dim * f4_row_bytes(i_dim))
                    .map(|k| ((k * 7 + t) % 251) as u8)
                    .collect();
                // 100..=149 — inside the band the SHIPPED artifact actually uses
                // (measured 2026-08-05 over all 9,261,023,232 of its scale bytes: 9 distinct
                // codes, 0x76..=0x7e), and in particular never 0xff. So the clean fixture
                // exercises the accept path rather than the reject one.
                let mut scale: Vec<u8> = (0..o_dim * f4_groups(i_dim))
                    .map(|k| (100 + (k + t) % 50) as u8)
                    .collect();
                if let Some((s, k, b)) = poison
                    && s == slot
                {
                    scale[k] = b;
                }
                w.add(
                    format!("e.{proj}.weight"),
                    Dtype::I8,
                    vec![o_dim, i_dim / 2],
                    weight,
                );
                w.add(
                    format!("e.{proj}.scale"),
                    Dtype::F8E8M0,
                    vec![o_dim, i_dim / F4_GROUP],
                    scale,
                );
            }
            w.write(&format!("{dir}/e.safetensors")).unwrap();
            Self {
                dir,
                expert_in,
                moe_inter,
            }
        }

        fn open(&self) -> Safetensors {
            Safetensors::open_file(&format!("{}/e.safetensors", self.dir)).unwrap()
        }

        fn expert<'a>(&self, src: &'a Safetensors) -> F4Expert<'a> {
            F4Expert {
                src,
                base: "e".into(),
                expert_in: self.expert_in,
                moe_inter: self.moe_inter,
            }
        }
    }

    impl Drop for F4Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// **An e8m0 `0xff` scale byte is refused at repack, and `0x00` is not.**
    ///
    /// `0xff` is the format's NaN. `common.hpp::e8m0f` decodes it correctly — to a quiet
    /// NaN, which is the right answer — and then `moe_fixed`'s saturating clamp launders it
    /// into a finite ±2^14, so a single bad byte becomes 32 weights of plausible garbage
    /// with no error anywhere downstream. The kernels cannot refuse it; this is the only
    /// path in the engine that reads every routed scale byte, because at decode they DMA
    /// from NVMe straight into the pool slot and the host never sees them.
    /// `docs/investigations/v4-flash-port.md` §S3 requirement 10.
    ///
    /// **Both directions, and the second is the one that needed measuring.** The guard must
    /// fire on `0xff` in any of the three projections — proved by injecting one, per
    /// projection, and requiring the message to name it. And it must leave everything else
    /// bit-identical: the requirement was handed over as "reject `0x00`/`0xff`", and `0x00`
    /// is a LEGAL encoding (`2^-127`, which f32 carries exactly as a subnormal and which
    /// both `quant::e8m0` and `e8m0f` special-case for that reason). Refusing it would be
    /// inventing a rule the format does not have, so a `0x00` fixture must pack unchanged —
    /// and it must pack to the SAME bytes the clean control does apart from that one, which
    /// is what stops this from being a test that any accept-everything packer passes.
    ///
    /// Measured before writing either half: over the shipped 43-layer set, 9,261,023,232
    /// scale bytes, 9 distinct codes, all in `0x76..=0x7e`, **zero `0x00` and zero `0xff`**.
    /// So this guard is green on every artifact that exists and the injection above is the
    /// only thing that has ever made it speak.
    #[test]
    fn an_e8m0_nan_scale_byte_is_refused_at_repack_and_a_subnormal_one_is_not() {
        use crate::artifact::quant::f4_expert_bytes;
        let clean = F4Fixture::new("e8m0_ok");
        let st = clean.open();
        let n = f4_expert_bytes(clean.expert_in, clean.moe_inter);
        let mut base = vec![0u8; n];
        clean
            .expert(&st)
            .pack(&mut base)
            .expect("a fixture with no 0xff must pack");

        // One `0xff` per projection, at a byte that is not the first — a guard that only
        // looked at scale[0] would pass a first-byte-only test.
        for slot in 0..3 {
            let fx = F4Fixture::with_scale_byte(&format!("e8m0_nan{slot}"), Some((slot, 5, 0xff)));
            let st = fx.open();
            let e = format!(
                "{:#}",
                fx.expert(&st)
                    .pack(&mut vec![0u8; n])
                    .err()
                    .unwrap_or_else(|| panic!("slot {slot}: a 0xff scale byte must be refused"))
            );
            let proj = crate::artifact::quant::V4_PROJ[slot];
            assert!(
                e.contains(&format!("{proj}.scale[")) && e.contains("0xff"),
                "slot {slot}: the refusal must name the projection and the byte, got: {e}"
            );
        }

        // `0x00` passes, and changes exactly the byte it was written into. Two assertions,
        // because "it packed" alone would also hold for a packer that dropped the scales.
        let fx = F4Fixture::with_scale_byte("e8m0_zero", Some((1, 5, 0x00)));
        let st = fx.open();
        let mut got = vec![0u8; n];
        fx.expert(&st)
            .pack(&mut got)
            .expect("0x00 is 2^-127, a legal e8m0 encoding — it must NOT be refused");
        let diff: Vec<usize> = (0..n).filter(|&k| got[k] != base[k]).collect();
        let off = crate::artifact::quant::f4_slot_offsets(fx.expert_in, fx.moe_inter);
        assert_eq!(
            diff,
            vec![off[3] + 5],
            "a 0x00 in w3's scales must move exactly that byte of the block"
        );
    }

    /// **A packed `.f4` block is the six source tensors concatenated, in this order.**
    ///
    /// The expected block is built here from the tensor NAMES spelled out literally, not
    /// from `F4Expert::spans` — that independence is the whole point. `pack` and `diff`
    /// share `spans`, so `diff` can only ever report `block != pack's output`; asking it
    /// about a block derived from `pack` is `A == A` and cannot fail. Comparing against a
    /// literal order and a literal concatenation CAN fail, and does: verified by mutation
    /// (2026-08-05) against a packer that swapped nibbles, a `V4_PROJ` with w1/w3
    /// transposed, and `F4_GROUP` changed 32 → 64.
    ///
    /// What this does NOT establish is that the order is the RIGHT one — `w1` really being
    /// gate and `w3` really being up is pinned separately, against the reference source, by
    /// `quant::tests::v4_proj_order_matches_the_reference_expert_forward`. And nibble
    /// ORDER within a byte is unchecked here by construction: a repack with the opposite
    /// convention is the same byte copy. That becomes checkable only when a `matvec_f4`
    /// exists to decode one, which is S3.
    #[test]
    fn f4_pack_concatenates_the_source_tensors_in_w1_w3_w2_order() {
        use crate::artifact::quant::f4_expert_bytes;
        let fx = F4Fixture::new("pack");
        let st = fx.open();

        let mut want = Vec::new();
        for name in [
            "e.w1.weight",
            "e.w1.scale",
            "e.w3.weight",
            "e.w3.scale",
            "e.w2.weight",
            "e.w2.scale",
        ] {
            let dt = if name.ends_with(".scale") {
                Dtype::F8E8M0
            } else {
                Dtype::I8
            };
            want.extend_from_slice(st.typed(name, dt).unwrap().0);
        }

        let mut got = vec![0u8; f4_expert_bytes(fx.expert_in, fx.moe_inter)];
        fx.expert(&st).pack(&mut got).unwrap();
        assert_eq!(
            got.len(),
            want.len(),
            "block size disagrees with the source spans"
        );
        assert_eq!(got, want, "the repack is not a straight concatenation");

        // `diff` agrees with the same independently-built block, and reports the exact
        // offset of a single changed byte — which is what makes `convert_v4 --verify`
        // able to name where a 3.4 GB layer file went wrong.
        let e = fx.expert(&st);
        assert_eq!(e.diff(&want).unwrap(), Vec::<usize>::new());
        for k in [0, want.len() / 3, want.len() - 1] {
            let mut bad = want.clone();
            bad[k] ^= 0xFF;
            assert_eq!(e.diff(&bad).unwrap(), vec![k], "diff missed a flip at {k}");
        }
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

    /// Every `*.part` currently in `dir`, as bare file names.
    fn walk_parts(dir: &str) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".part"))
            .collect();
        v.sort();
        v
    }

    /// Streaming a layer in windows produces the same file the whole-layer buffer did, and
    /// keeps doing so when the block count is not a multiple of the window.
    ///
    /// The reference arm here is the code both converters used to run inline — allocate
    /// `VQ_ALIGN + blocks * stride`, stamp the header, `fill_expert_blocks` over the rest,
    /// write once. That is what G1a's byte-identity claim is against, so it is spelled out
    /// rather than derived from the thing under test.
    ///
    /// `stride > bytes` and every payload non-zero, so a block landing in the wrong slot moves
    /// bytes rather than merely repeating them, and any padding that got written shows up.
    ///
    /// `blocks = 7` against `per = 2` deliberately leaves a short final window; an off-by-one
    /// there writes 8 blocks or 6, and the length assertion catches it.
    #[test]
    fn a_windowed_expert_layer_is_byte_identical_to_the_buffered_one() {
        let dir = tmpdir("expert_layer");
        let (stride, bytes, blocks) = (64usize, 40usize, 7usize);
        let header = ExpertHeader::new(F4_MAGIC, 3, blocks, 128, 64, stride).to_bytes();
        // Distinct non-zero payload per expert, and the LAST byte of each differs, so a
        // block written into the wrong slot moves bytes rather than merely repeating them.
        let fill = |e: usize, slot: &mut [u8]| -> Result<()> {
            slot.fill(0xA0 | (e as u8 & 0x0f));
            slot[bytes - 1] = 0xE0 | (e as u8 & 0x0f);
            Ok(())
        };

        // Reference: the whole-layer buffer, as both converters wrote it before 2026-08-10.
        let mut buf = vec![0u8; VQ_ALIGN + blocks * stride];
        buf[..EXPERT_HEADER_BYTES].copy_from_slice(&header);
        crate::artifact::quant::fill_expert_blocks(
            &mut buf[VQ_ALIGN..],
            stride,
            bytes,
            blocks,
            fill,
        )
        .unwrap();

        // One window per two blocks: four windows, the last holding one block.
        let path = format!("{dir}/L03.f4");
        let n =
            write_expert_layer(&path, &header, stride, bytes, blocks, 2 * stride, fill).unwrap();

        // A zero stride is REFUSED, not clamped. It used to be `stride.max(1)`, which turned a
        // caller's bad geometry into a panic inside `chunks_exact_mut(0)` two functions away.
        let zero = format!("{dir}/L04.f4");
        assert!(
            write_expert_layer(&zero, &header, 0, 0, blocks, 2 * stride, fill).is_err(),
            "stride 0 accepted"
        );
        assert!(!std::path::Path::new(&zero).exists(), "refused but created");

        // Another process's debris is NOT adopted. This arm moved here from `write_atomic`'s
        // test when that function was deleted: the pid-suffixed `.part` is the defence against
        // two concurrent converts into one `out_dir`, whose interleaved writes OF EQUAL LENGTH
        // yield a file of exactly the right length — the one corruption shape `open_routed`'s
        // length check cannot see. Seeded longer than the payload so adopting it would fail on
        // length rather than on content, and left in place afterwards to prove it was untouched.
        let foreign = format!("{path}.999999.part");
        std::fs::write(&foreign, vec![0xFDu8; buf.len() + 64]).unwrap();
        let again = format!("{dir}/L05.f4");
        write_expert_layer(&again, &header, stride, bytes, blocks, 2 * stride, fill).unwrap();
        assert_eq!(
            std::fs::read(&again).unwrap().len(),
            buf.len(),
            "adopted it"
        );
        assert_eq!(
            std::fs::read(&foreign).unwrap().len(),
            buf.len() + 64,
            "consumed another process's .part"
        );
        std::fs::remove_file(&foreign).unwrap();

        let got = std::fs::read(&path).unwrap();

        assert_eq!(n as usize, buf.len(), "reported length");
        assert_eq!(got.len(), buf.len(), "file length");
        assert_eq!(
            got,
            buf,
            "windowed output differs from the buffered form at byte {:?}",
            got.iter().zip(&buf).position(|(a, b)| a != b)
        );
        // Independent of the reference arm: every block's padding is zero. Stated separately
        // because if BOTH arms grew the same padding bug the comparison above would pass.
        for e in 0..blocks {
            let pad = &got[VQ_ALIGN + e * stride + bytes..VQ_ALIGN + (e + 1) * stride];
            assert!(pad.iter().all(|&b| b == 0), "block {e} padding not zero");
        }
        assert_eq!(walk_parts(&dir), Vec::<String>::new(), "left a temp file");
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
