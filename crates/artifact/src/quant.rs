//! Weight decode + the VQ-int3 quantizer. CPU oracles the HIP kernels are
//! validated against, plus the offline `quant_vq` encode. The formats: VQ-int3 and
//! group-scaled int4 (routed experts), fp8-e4m3 block (attention/dense), int8 per-row
//! (embed/lm_head), bf16 (indexer), f32 (norms/gate). Functions take raw byte
//! slices + dims — no snapshot-struct coupling.
//!
//! **This file is now the SHARED VOCABULARY; one format per submodule.** Split 2026-08-15,
//! when CodeScene scored the single 2247-line file 8.54: past ~500 lines it prices a
//! module's LCOM4, and four formats, a k-means learner and three checkpoints' naming tables
//! had accumulated here with no call edge between them. What stayed is exactly what more
//! than one format needs — the byte readers, the GEMV entry contract, the three weight
//! views, the shared row loop, and the routed-expert BLOCK geometry that all three routed
//! formats walk. What left went by format: [`vq`], [`int4`], [`f4`], [`fp8`], plus
//! [`kmeans`] (the codebook learner) and [`naming`] (what the checkpoints call things).
//!
//! **The split was a move, not a rewrite**: every body and every comment travelled verbatim,
//! because in this repo the comments carry the measurement that chose the constant. Two
//! private functions widened to `pub(super)` and a handful of intra-doc links gained a
//! `super::`; nothing else changed, and every public name is re-exported below at its
//! original path, so no caller outside this file moved.
//!
//! `matvec_i8` is the one format that stayed: int8 per-row has no geometry, no block layout
//! and no converter — it is a single oracle over `matvec_bytes`, so a submodule holding it
//! would hold nothing else.

pub mod f4;
pub mod fp8;
pub mod int4;
pub mod kmeans;
pub mod naming;
pub mod vq;

/// Every name the submodules own, re-exported at the path it has always had.
///
/// The address of a public function is itself an interface: `bin/convert`, `bin/fp8_to_i4`,
/// `format.rs` and the engine's kernel tests all spell these `rivoli_artifact::quant::…`,
/// and nothing was learned in the 2026-08-15 split that justifies making them move. Adding
/// a submodule is a file-layout decision; it is not an API change, and this block is what
/// keeps the two separate.
pub use f4::{
    F4_GROUP, F4_LUT, e8m0, f4_expert_bytes, f4_expert_stride, f4_groups, f4_proj_bytes,
    f4_row_bytes, f4_slot_offsets,
};
pub use fp8::{FP8_BLOCK, dequant_fp8_block, matvec_fp8, quantize_fp8_block};
pub use int4::{
    I4_GROUP, dequant_i4, i4_expert_bytes, i4_expert_stride, i4_groups, i4_proj_bytes,
    i4_row_bytes, i4_slot_offsets, matvec_i4, quant_i4, write_i4_proj,
};
pub use kmeans::{codebook_norms, learn_codebook, learn_codebook_k, sample_subvectors};
pub use naming::{
    ExpertProjs, K3_PACKED, K3_PROJ, K3_SCALE, K3_TEXT_PREFIX, PROJ, V4_PROJ, expert_base,
    expert_projs, k3_expert_base, v4_expert_base, v4_expert_projs,
};
pub use vq::{
    VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VQ_K, VqProj, matvec_vq, quant_vq, set_idx, vq_decode_proj,
    vq_expert, vq_expert_bytes, vq_expert_stride, vq_groups, vq_proj_bytes, vq_refit, vq_row_bytes,
    vq_slot_offsets,
};

/// Raw bytes as little-endian `N`-byte words — the READ side of [`write_le_scales`], and
/// the one place the byte→word convention lives. `chunks_exact` truncates a ragged tail,
/// exactly as the writer does, so the two halves agree about a short buffer too.
fn read_le<const N: usize>(bytes: &[u8]) -> impl Iterator<Item = [u8; N]> + '_ {
    bytes.chunks_exact(N).filter_map(|c| c.try_into().ok())
}

/// Read an F32 tensor's raw little-endian bytes into a `Vec<f32>`. For O-length
/// tensors only (norm weights, per-projection codebooks) — loaded once at startup.
pub fn read_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0, "F32 tensor length not a multiple of 4");
    read_le(bytes).map(f32::from_le_bytes).collect()
}

/// The entry contract every `matvec_*` oracle below shares: one `y` per output row, one
/// `x` per input column. Each oracle then asserts its OWN packed array against the
/// geometry it just derived, which is why those checks stay at the call sites.
///
/// Debug-only, because these are oracles swept per expert per layer by the kernel tests —
/// a release-mode length check would be pure cost on a path that only ever runs against
/// arrays this module also built. `#[track_caller]` so a failure names the oracle that was
/// mis-called rather than this line.
#[track_caller]
fn debug_check_gemv(y: &[f32], x: &[f32], o_dim: usize, i_dim: usize) {
    debug_assert_eq!(y.len(), o_dim);
    debug_assert_eq!(x.len(), i_dim);
}

/// The row loop the two BYTE-per-weight oracles share: `packed` is `y.len()` rows of
/// `i_dim` bytes, row-major, and `row_dot` turns row `o` into its output element.
///
/// Only the decode differs — an e4m3 byte against a block scale, a signed byte against a
/// row scale — and both spelled out the same `zip(chunks_exact(i_dim))` accumulate. The
/// int4 and VQ oracles do NOT use it: their rows are packed sub-byte, so `i_dim` is not
/// their row stride.
/// A packed matrix and its scales, viewed as ONE value — the `Weights` fix the old
/// parameter-list jscpd exemption named as the honest one, paid for by CodeScene's
/// arg-count rule (2026-08-15) exactly as that note predicted ("if that hop is ever paid
/// for another reason"). Per-format structs rather than an enum: every oracle is
/// format-specific, and the pairing makes "i4 scales beside an i8 array" unrepresentable.
pub struct Fp8W<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [f32],
    pub block: usize,
}

pub struct VqW<'a> {
    pub indices: &'a [u8],
    pub scales: &'a [u16],
    pub codebook: &'a [f32],
}

/// The row/group-scaled byte view `matvec_i4` and `matvec_i8` share: same bytes-and-
/// scales SHAPE, different interpretation — which lives in each oracle, not here. One
/// struct, because two twin field lists were themselves the clone jscpd flagged.
pub struct RowScaledW<'a> {
    pub packed: &'a [u8],
    pub scale: &'a [f32],
}

/// A flat `[n][VQ_DIM]` run of subvectors — a learned codebook, a k-means sample, or a
/// centroid table. The three are the SAME layout and none of them is the others: a sample
/// passed where a codebook belongs indexes legally, fits, and writes a legal WRONG file.
/// That is the failure `vq::subvec`'s doc is written to prevent one call site at a time,
/// and this is the same rule said once, in the signature, where the reader is.
///
/// **A NAME, not a newtype, and the difference is exactly what it buys and does not.**
/// `&Subvectors` IS `&[f32]`: nothing is enforced, no conversion runs, and no caller outside
/// this crate changed when it appeared. A real newtype WOULD enforce it, and costs a `.0` hop
/// in the k-means inner loop plus a signature change at `bin/convert` — the same trade the
/// old tree's `WMat` note priced and declined, and it has not been re-priced here. Read this
/// as documentation with a type's reach, never as a check.
///
/// Introduced 2026-08-15 with the split, when the learner and the VQ encoder came out of
/// `quant.rs` as modules of bare `&[f32]` and CodeScene scored them 51% and 33% on Primitive
/// Obsession — "lacks a domain language that encapsulates the semantics of function
/// arguments". The domain language was real and was living in prose; this is where it goes.
pub type Subvectors = [f32];

/// A row-major `[o_dim][i_dim]` f32 weight block, or one row or one `VQ_GROUP` of one — the
/// input side of every quantizer in this module and the output of every dense decode.
/// Distinct from [`Subvectors`] because the two are indexed by different rules and the
/// quantizers take both: `quant_vq(w, o_dim, i_dim, codebook)` is weights, then a codebook,
/// and swapping them is a length error only when the lengths happen to differ. Same standing
/// as [`Subvectors`] — a name, not a newtype.
pub type Weights = [f32];

impl<'a> Fp8W<'a> {
    pub fn new(packed: &'a [u8], scale: &'a [f32], block: usize) -> Self {
        Fp8W {
            packed,
            scale,
            block,
        }
    }

    /// The scale governing element `[o, i]`. One spelling of the tile-grid index, shared by
    /// the GEMV oracle and the dense dequant: a grid the two read differently is invisible
    /// to every length and shape check downstream.
    #[inline]
    fn tile_scale(&self, sc_cols: usize, o: usize, i: usize) -> f32 {
        self.scale[(o / self.block) * sc_cols + i / self.block]
    }
}

impl<'a> VqW<'a> {
    pub fn new(indices: &'a [u8], scales: &'a [u16], codebook: &'a [f32]) -> Self {
        VqW {
            indices,
            scales,
            codebook,
        }
    }
}

impl<'a> RowScaledW<'a> {
    pub fn new(packed: &'a [u8], scale: &'a [f32]) -> Self {
        RowScaledW { packed, scale }
    }
}

fn matvec_bytes(y: &mut [f32], packed: &[u8], i_dim: usize, row_dot: impl Fn(usize, &[u8]) -> f32) {
    for (o, (yo, row)) in y.iter_mut().zip(packed.chunks_exact(i_dim)).enumerate() {
        *yo = row_dot(o, row);
    }
}

/// Reference int8 GEMV `y[o] = scale[o] · Σ_i x[i]·(i8)packed[o·i_dim+i]` — the CPU
/// oracle for the `gemv_i8` kernel (lm_head → logits). `packed` is raw bytes
/// reinterpreted as signed, matching the kernel's `signed char`.
pub fn matvec_i8(y: &mut [f32], x: &[f32], w: RowScaledW<'_>, shape: [usize; 2]) {
    let RowScaledW { packed, scale } = w;
    let [o_dim, i_dim] = shape;
    debug_check_gemv(y, x, o_dim, i_dim);
    debug_assert_eq!(packed.len(), o_dim * i_dim);
    matvec_bytes(y, packed, i_dim, |o, row| {
        let mut acc = 0.0f32;
        for (&b, &xi) in row.iter().zip(x) {
            acc += xi * (b as i8) as f32;
        }
        // Row scale applied to the finished dot, exactly as before: `scale[o] · Σ …`.
        acc * scale[o]
    });
}

// ── the routed-expert block: what all three routed formats share ─────────────
//
// `.vq3`, `.i4` and `.f4` differ only in how many bytes a projection's packed rows and its
// scales take. Everything else about a block — three projections in `vq_expert_layout`
// order, each `packed ‖ scales`, tightly packed, the whole padded up to `VQ_ALIGN` so one
// expert is one O_DIRECT read — is format-independent and is written down exactly once
// here. Three hand-written copies of that walk is what `slot_offsets` exists to prevent,
// and the `.i4` copy had already drifted into a different SHAPE from the `.vq3` one while
// computing the same thing before it was folded in.

/// O_DIRECT alignment the converter pads each expert's on-disk block up to, so one
/// expert fetch is a single block-aligned read (mirrors int4's slot alignment).
pub const VQ_ALIGN: usize = 4096;

/// The three projections of one MoE expert in on-disk order, as `(o_dim, i_dim)`:
/// gate and up map `expert_in`→`moe_inter`; down maps `moe_inter`→`expert_in`.
///
/// `expert_in` is the width the ROUTED EXPERT BLOCK is entered at, which is not always
/// the model's `hidden_size`. GLM-5.2 and V4 route on the residual stream, so for them
/// the two are the same number and every call site passes `cfg.hidden`. Kimi-K3 does
/// not: it down-projects 7168→3584 and runs its experts in that latent space
/// (`routed_expert_hidden_size` 3584 against `hidden_size` 7168 —
/// `docs/reference/k3-architecture.md` §2), so its experts are `[3072,3584]`·2 +
/// `[3584,3072]` and every geometry function here takes 3584.
///
/// Hence the parameter is named for the ROLE, not for `hidden`. Passing 7168 where 3584
/// belongs computes a self-consistent layout of the wrong size, which the length checks
/// catch — but the reverse, feeding the latent to K3's SHARED expert (a trunk-side
/// `[7168,6144]` that stays full width), is self-consistent AND the same shape class, so
/// it streams the wrong weights through all 92 MoE layers and fails nothing. A field
/// called `hidden` holding 3584 is how that substitution gets made without anyone seeing
/// it; this name is the whole defence.
pub fn vq_expert_layout(expert_in: usize, moe_inter: usize) -> [(usize, usize); 3] {
    [
        (moe_inter, expert_in),
        (moe_inter, expert_in),
        (expert_in, moe_inter),
    ]
}

/// Sum one format's `*_proj_bytes` over an expert's three projections. The `(o, i)` dims
/// are format-independent, so this is the one place the "an expert is gate‖up‖down"
/// structure is written; the three formats differ only in `proj`.
fn expert_bytes(expert_in: usize, moe_inter: usize, proj: fn(usize, usize) -> usize) -> usize {
    vq_expert_layout(expert_in, moe_inter)
        .iter()
        .map(|&(o, i)| proj(o, i))
        .sum()
}

/// Round an expert's unpadded size up to [`VQ_ALIGN`], so one expert is a single
/// block-aligned O_DIRECT read. Shared by all three routed formats.
fn expert_stride(bytes: usize) -> usize {
    bytes.div_ceil(VQ_ALIGN) * VQ_ALIGN
}

/// The six byte offsets within one expert block, in expert-descriptor field order:
/// `[gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale]`.
///
/// **ONE definition for all three routed formats.** The STRUCTURE — three projections in
/// [`vq_expert_layout`] order, each `packed rows ‖ group scales`, tightly packed — is
/// format-independent; the formats differ only in the two byte counts, which is what
/// `span` supplies. Three hand-written copies of this walk is what
/// `f4_slot_offsets` would otherwise have been, and the `.i4` copy had already drifted
/// into a different SHAPE (a flat let-chain) from the `.vq3` one while computing the same
/// thing — two spellings of one rule, which is the setup where a third gets it wrong.
///
/// `span(o, i)` returns `(packed bytes, scale bytes)` for one projection, and the sum of
/// the pair is by construction that format's `*_proj_bytes` — so these offsets and
/// [`expert_bytes`] cannot disagree about where a block ends.
fn slot_offsets(
    expert_in: usize,
    moe_inter: usize,
    span: impl Fn(usize, usize) -> (usize, usize),
) -> [usize; 6] {
    let mut off = [0usize; 6];
    let mut base = 0usize;
    for (p, &(o, i)) in vq_expert_layout(expert_in, moe_inter).iter().enumerate() {
        let (packed, scales) = span(o, i);
        off[p * 2] = base;
        off[p * 2 + 1] = base + packed;
        base += packed + scales;
    }
    off
}

/// Write a projection's group scales little-endian into the front of `dst`, `N` bytes
/// each, stopping at whichever of the two runs out (`chunks_exact_mut` truncates a ragged
/// tail rather than panicking).
///
/// `N` is the whole difference between the two producers: `.vq3` stores bf16 scales
/// (2 bytes, `bin/convert`) and `.i4` stores f32 ones (4, [`write_i4_proj`]). Takes the
/// encoded bytes rather than the numbers, so one function covers both without a numeric
/// trait bound for two call sites.
pub fn write_le_scales<const N: usize>(dst: &mut [u8], scales: impl Iterator<Item = [u8; N]>) {
    for (s, out) in scales.zip(dst.chunks_exact_mut(N)) {
        out.copy_from_slice(&s);
    }
}

/// Worker count for the two fan-outs here (expert packing below, k-means assignment in
/// [`kmeans`]), with a fixed fallback. One owner so a box that cannot report its parallelism
/// gets the same fan-out in both — and, for the k-means half, so the fallback cannot silently
/// become a second number, which would change how the merge rounds and with it the shipped
/// bytes. That argument is why it stayed in the parent when the learner moved out on
/// 2026-08-15: a per-module copy is exactly the second number it exists to prevent.
fn worker_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |t| t.get())
}

/// Fill `n` consecutive expert blocks of `stride` bytes in `buf`, in parallel, calling
/// `fill(e, &mut block[..bytes])` for each. The padding between `bytes` and `stride` is
/// left as the caller found it.
///
/// **`fill` must write all `bytes` it is given.** Nothing here clears a slot first, and the one
/// caller reuses its buffer across windows, so a partial write leaves whatever the previous
/// occupant of those bytes was — see `format::write_expert_layer`, which spells out the failure
/// this produces and why it is worse than the zeros the old whole-layer buffer would have left.
///
/// Called only by `format::write_expert_layer`, which windows it — both converters reach it
/// through there rather than filling a whole-layer buffer. The split is by DISJOINT
/// `split_at_mut` slices rather than by index, so the borrow checker witnesses that no two
/// workers can touch the same block — with indices, an off-by-one in the chunking would be an
/// aliasing bug the compiler could not see, and the whole point of this loop is that expert
/// `e` lands at exactly `e · stride`.
pub(crate) fn fill_expert_blocks(
    buf: &mut [u8],
    stride: usize,
    bytes: usize,
    n: usize,
    fill: impl Fn(usize, &mut [u8]) -> anyhow::Result<()> + Sync,
) -> anyhow::Result<()> {
    anyhow::ensure!(bytes <= stride, "expert bytes {bytes} > stride {stride}");
    anyhow::ensure!(
        buf.len() >= n * stride,
        "buffer {} < {n} blocks of {stride}",
        buf.len()
    );
    let per = n.div_ceil(worker_threads().max(1)).max(1);
    let fill = &fill;
    std::thread::scope(|s| -> anyhow::Result<()> {
        let mut rest = &mut buf[..n * stride];
        let (mut e0, mut handles) = (0usize, Vec::new());
        while e0 < n {
            let take = per.min(n - e0);
            let (mine, tail) = rest.split_at_mut(take * stride);
            rest = tail;
            let base = e0;
            e0 += take;
            handles.push(s.spawn(move || -> anyhow::Result<()> {
                for (j, slot) in mine.chunks_exact_mut(stride).enumerate() {
                    fill(base + j, &mut slot[..bytes])?;
                }
                Ok(())
            }));
        }
        for h in handles {
            h.join()
                .map_err(|_| anyhow::anyhow!("expert worker panicked"))??;
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    // Both of these confront the SHARED block walk with numbers nothing in it reads, so a
    // failure names a disagreement rather than a typo. Crate-wide `unwrap`/`expect` are
    // `deny`; here a firing one IS the failure report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// All three layouts come off ONE walk (`slot_offsets`), so this confronts that walk
    /// with the byte counts nothing in it reads: `*_proj_bytes`, which `expert_bytes` sums
    /// independently. Projection `p` must begin at the sum of the `p` before it.
    ///
    /// Deliberately NOT "each scale span abuts the next offset" — that is the walk's own
    /// `base += packed + scales` restated, and a test that respells the formula it checks
    /// can only fail when the copy drifts. These two comparisons can fail on the code.
    #[test]
    fn every_routed_format_places_each_projection_at_the_sum_of_the_ones_before_it() {
        let (expert_in, inter) = (6144usize, 2048usize);
        let [(go, gi), (uo, ui), _] = vq_expert_layout(expert_in, inter);
        for (name, off, proj) in [
            (
                "vq3",
                vq_slot_offsets(expert_in, inter),
                vq_proj_bytes as fn(usize, usize) -> usize,
            ),
            ("i4", i4_slot_offsets(expert_in, inter), i4_proj_bytes),
            ("f4", f4_slot_offsets(expert_in, inter), f4_proj_bytes),
        ] {
            assert_eq!(
                off[0], 0,
                "{name}: the block starts at the first projection"
            );
            assert_eq!(off[2], proj(go, gi), "{name}: up_packed");
            assert_eq!(off[4], proj(go, gi) + proj(uo, ui), "{name}: down_packed");
        }
    }

    #[test]
    fn read_f32_roundtrips() {
        let vals = [1.5f32, -2.25, 0.0, 3.75];
        let mut bytes = Vec::new();
        for v in vals {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(read_f32(&bytes), vals);
    }
}
