//! Opening a routed-expert set for DECODE: one O_DIRECT fd per layer, confronted with the
//! dims the caller believes in, and the read specs the streamer turns into NVMe DMA.
//!
//! The read half of what [`super::layer`] writes. Everything here is a check made once at
//! open, because at decode the bytes go from NVMe straight into a pool slot and the host
//! never looks at them — a set opened against a transposed `(expert_in, moe_inter)` streams
//! the wrong bytes for the rest of the run and nothing downstream can tell.
//!
//! [`f4_source`] and [`f4_layer_range`] are here rather than with the rest of
//! `manifest.json` ([`super::meta`]) because the range is not provenance: it is the input to
//! the [`SetDims`] two lines below, the loader's ONLY source for which layers exist, and a
//! `0..num_hidden_layers` guess is precisely wrong for the partial artifact it describes.
//! The producer travels with it for the reason its own doc gives — the two shapes must
//! agree, and one function between them is the cheapest way to guarantee that.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::os::fd::{AsRawFd, RawFd};

use super::header::{EXPERT_HEADER_BYTES, ExpertHeader, LayerDims, RoutedFmt};
use crate::quant::{VQ_DIM, VQ_K};

/// The dimensions an expert set is opened against. `n_layers` is EXCLUSIVE and is the
/// CALLER's bound, not the artifact's — the pin opens one layer past `cfg.n_layers` when
/// the MTP head has an expert file, and the V4 pin bounds it by the artifact's own
/// `f4_source` range ([`super::meta::f4_layer_range`]) rather than by `num_hidden_layers`.
///
/// A struct rather than five positional `usize`s, because nothing about a transposed
/// `(expert_in, moe_inter)` fails a check: the set opens, every length matches, and it streams
/// the wrong bytes. Five bare `usize`s in a row is the argument list that gets transposed.
#[derive(Clone, Copy)]
pub struct SetDims {
    /// First layer with a file, and one past the last — [`SetDims::new`] takes them as the
    /// `Range` they are and splits them here only to keep `SetDims` `Copy`. GLM's starts
    /// after the dense prefix and may run one past `cfg.n_layers` for the MTP head; V4's is
    /// the artifact's own `f4_source` range ([`super::meta::f4_layer_range`]), and V4 has no dense
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
/// w2‖w2_scale, the same gate/up/down slot order (see [`super::expert::F4Expert::spans`]).
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
        // instead of naming the range. `SetDims` splits the `Range` into two fields to stay
        // `Copy`, and nothing on the way in rejects an inverted one.
        ensure!(
            first_layer <= n_layers,
            "layer range [{first_layer}, {n_layers}) is inverted"
        );
        let (stride, expert_bytes) = fmt.geometry(expert_in, moe_inter);
        let mut files = Vec::with_capacity(n_layers - first_layer);
        for layer in first_layer..n_layers {
            files.push(open_layer(
                dir,
                fmt,
                LayerDims {
                    layer,
                    n_experts,
                    expert_in,
                    moe_inter,
                    stride,
                },
            )?);
        }
        Ok(Self {
            files,
            first_layer,
            n_layers,
            n_experts,
            stride,
            expert_bytes,
            hbytes: fmt.hbytes(),
            has_shared: fmt.has_shared(),
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

    /// The int3-VQ set, for a caller whose format is not a runtime choice —
    /// `tests/artifact.rs` opens the shipped `.vq3` this way. The engine's own format IS a
    /// runtime choice, so it goes through [`ExpertSet::open_routed`] directly.
    ///
    /// Took the six dimensions loose until 2026-08-15, when the list was replaced by the
    /// [`SetDims`] it was building anyway. Naming each dimension at the call site is what
    /// [`SetDims::new`] is for and it still reads the same there; restating the list a second
    /// time here bought nothing and made this the longest argument list in the file.
    pub fn open_vq3(dir: &str, d: SetDims) -> Result<Self> {
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

/// Open one layer file O_DIRECT and confront it with `want`: its LENGTH first, then — where
/// the format carries one ([`RoutedFmt::magic`]) — its 40-byte header.
fn open_layer(dir: &str, fmt: RoutedFmt, want: LayerDims) -> Result<std::fs::File> {
    let (layer, ext) = (want.layer, fmt.ext());
    let path = format!("{dir}/L{layer:02}.{ext}");
    let f = open_direct(&path).with_context(|| format!("open {path}"))?;
    let len = f.metadata()?.len() as usize;
    // Routed blocks, plus the shared one only where the format has one. Both terms come
    // from `fmt`, so the block count and `shared_block`'s refusal cannot disagree.
    let bytes = fmt.hbytes() + (want.n_experts + usize::from(fmt.has_shared())) * want.stride;
    ensure!(len == bytes, "{path}: {len} bytes, expected {bytes}");
    if fmt.magic().is_some() {
        check_layer_header(&path, fmt, want)?;
    }
    Ok(f)
}

/// Parse a layer file's 40-byte header and confront it with the dims the set is being opened
/// against. Headerless `.i4` never reaches here — see [`open_layer`].
fn check_layer_header(path: &str, fmt: RoutedFmt, want: LayerDims) -> Result<()> {
    // Header via a separate BUFFERED fd — the O_DIRECT one belongs to the streamer, and 40
    // bytes at offset 0 is neither length- nor buffer-aligned.
    //
    // Read EXACTLY the header. This was `std::fs::read(path)`, which pulls the WHOLE layer
    // file through the page cache to look at 40 bytes of it — and the cost was not small.
    // **Measured on the GLM arm 2026-08-05**, both binaries built before either ran and the
    // two alternated three times with no `cargo build` between them
    // (`tests/artifact.rs::artifact_reads_back` over `/var/db/rivoli/glm52-vq3-full`, 76
    // `.vq3` layers, medians):
    //
    //     std::fs::read   180.1 s   298.49 GB read from the block device
    //     40-byte pread     0.038 s   0.48 MB
    //
    // ~4700x, and it is startup on EVERY run — a one-time cost amortizes against nothing. V4
    // pays the same shape at 43 x 3,422,556,160 = 147 GB. A pre-existing defect in shipped
    // GLM code that the `.f4` reader walked into, not one `.f4` introduced.
    let mut raw = [0u8; EXPERT_HEADER_BYTES];
    let mut hf =
        std::fs::File::open(path).with_context(|| format!("open {path} for its header"))?;
    std::io::Read::read_exact(&mut hf, &mut raw).with_context(|| format!("read {path} header"))?;
    let h = ExpertHeader::from_bytes(&raw, fmt)?;
    let LayerDims {
        layer,
        n_experts,
        expert_in,
        moe_inter,
        stride,
    } = want;
    // `stride` is checked too, and it was not before. The converter writes the value it
    // INDEXED BLOCKS WITH (`ExpertHeader::new`'s doc says why it is passed rather than
    // re-derived) while `RoutedFmt::geometry` re-derives it here — so without this conjunct
    // the header's one non-redundant field had no reader, and a writer whose stride disagreed
    // with this build's would pass every check on a file of the right total length.
    ensure!(
        h.layer as usize == layer
            && h.n_experts as usize == n_experts
            && h.expert_in as usize == expert_in
            && h.moe_inter as usize == moe_inter
            && h.stride as usize == stride,
        // `expert_in (hidden_size)` rather than bare `expert_in`: whoever reads this
        // is holding a config.json and an artifact, and `expert_in` is a name that
        // appears in neither. For GLM and V4 the value IS `hidden_size`; on K3 it
        // is `routed_expert_hidden_size`, which is why both are named.
        "{path}: header (layer {} experts {} expert_in {} moe_inter {} stride {}) \
         disagrees with config (layer {layer} experts {n_experts} \
         expert_in [hidden_size / routed_expert_hidden_size] {expert_in} \
         moe_inter [moe_intermediate_size] {moe_inter} stride {stride})",
        h.layer,
        h.n_experts,
        h.expert_in,
        h.moe_inter,
        h.stride
    );
    Ok(())
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
    let raw = crate::quant::read_f32(&std::fs::read(format!("{dir}/codebooks.f32"))?);
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

/// The `f4_source` provenance block, written by both `.f4` converters and read by
/// [`f4_layer_range`].
///
/// **Deliberately adjacent to its only reader.** `layers` is not decoration — `f4_layer_range` is
/// the loader's sole source for which layers an artifact holds, and treats the block's absence as
/// a hard error — so the producer and the consumer of this shape must agree, and the cheapest way
/// to guarantee that is one function between them. Factored 2026-08-11, when `convert_k3` became
/// the second producer and the duplication gate refused the copy; before that `convert_k3` omitted
/// the block entirely and every artifact it wrote was unopenable.
///
/// `tool` and `chain` are for a human reading a manifest six months later — two `.f4` sets built
/// from different checkpoints are byte-indistinguishable on disk. `src` is the checkpoint path.
///
/// `chain` is a literal rather than a parameter because both producers' sources are the SAME
/// encoding — OCP MX e2m1 nibbles with e8m0 group scales — so both really are `fp4 -> fp4`, and
/// `tool` already says which converter ran.
pub fn f4_source(tool: &str, src_dir: &str, layers: std::ops::Range<usize>) -> serde_json::Value {
    serde_json::json!({
        "tool": tool,
        "chain": "fp4->fp4 (repack)",
        "src": src_dir,
        "layers": [layers.start, layers.end],
    })
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
        // Both converters that produce `.f4` are named, because the message is the first thing a
        // reader sees and "not a convert_v4 artifact" in front of a Kimi-K3 directory sends them
        // looking for the wrong bug. `convert_k3` shipped without this stamp for exactly one
        // commit, and this is the error it produced.
        .with_context(|| {
            format!("{path} has no `f4_source` — not a convert_v4 or convert_k3 artifact")
        })?;
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
