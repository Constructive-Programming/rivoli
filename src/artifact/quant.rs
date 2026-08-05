//! Weight decode + the VQ-int3 quantizer. CPU oracles the HIP kernels are
//! validated against, plus the offline `quant_vq` encode. The formats: VQ-int3 and
//! group-scaled int4 (routed experts), fp8-e4m3 block (attention/dense), int8 per-row
//! (embed/lm_head), bf16 (indexer), f32 (norms/gate). Functions take raw byte
//! slices + dims — no snapshot-struct coupling.

use crate::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16};

/// Read an F32 tensor's raw little-endian bytes into a `Vec<f32>`. For O-length
/// tensors only (norm weights, per-projection codebooks) — loaded once at startup.
pub fn read_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert_eq!(bytes.len() % 4, 0, "F32 tensor length not a multiple of 4");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The entry contract every `matvec_*` oracle below shares: one `y` per output row, one
/// `x` per input column. Each oracle then asserts its OWN packed array against the
/// geometry it just derived, which is why those checks stay at the call sites.
///
/// Debug-only, because these are oracles and `bin/i4_audit` sweeps them per expert per
/// layer — a release-mode length check would be pure cost on a path that only ever runs
/// against arrays this module also built. `#[track_caller]` so a failure names the oracle
/// that was mis-called rather than this line.
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
fn matvec_bytes(y: &mut [f32], packed: &[u8], i_dim: usize, row_dot: impl Fn(usize, &[u8]) -> f32) {
    for (o, (yo, row)) in y.iter_mut().zip(packed.chunks_exact(i_dim)).enumerate() {
        *yo = row_dot(o, row);
    }
}

/// GEMV against an **fp8-e4m3** block-scaled matrix `W[o_dim, i_dim]` — the CPU
/// oracle the HIP `gemv_fp8` kernel (attention/dense projections) is validated
/// against. `scale` is the F32 `weight_scale_inv`, one value per `block × block`
/// tile: `w[o,i] = e4m3(packed[o·i_dim+i]) · scale[(o/block)·sc_cols + i/block]`.
pub fn matvec_fp8(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scale: &[f32],
    i_dim: usize,
    block: usize,
) {
    debug_assert_eq!(x.len(), i_dim);
    let sc_cols = i_dim.div_ceil(block);
    matvec_bytes(y, packed, i_dim, |o, row| {
        let mut acc = 0.0f32;
        for (i, (&b, &xi)) in row.iter().zip(x).enumerate() {
            acc += e4m3_to_f32(b) * scale[(o / block) * sc_cols + i / block] * xi;
        }
        acc
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Vector-quantized int3 experts (M1 scalar oracle).
//
// We requantize the fp8 checkpoint's experts into a uniform container the streaming
// kernel reads. Each row's weights are split into `VQ_DIM`-wide subvectors, and each
// subvector is quantized to the nearest entry of a learned `VQ_K`-entry codebook
// (vector quantization — the E8/QuIP# idea, but a data-learned codebook, which on
// GLM-5.2 weights ties per-row int4 where a fixed E8 lattice fell short). A bf16
// scale per `VQ_GROUP` weights along the input dim carries the outer magnitude.
//
// On disk per row: `i_dim/VQ_DIM` codebook indices (VQ_INDEX_BITS each, packed) +
// `i_dim/VQ_GROUP` bf16 group scales. `w[subvec] ≈ scale_group · codebook[index]`.
// This module DEFINES the bytes the converter (M2) writes and the HIP kernel (M3)
// reads. The codebook is learned once by the converter, stored in the file header,
// and embedded in the kernel. Hadamard rotation (QuIP incoherence) is CLOSED, not merely
// deferred: this comment used to cite `docs/int3.md` for "no gain on these
// already-well-conditioned weights", and that file was never committed. The claim was
// re-derived from evidence that does exist and it holds — see
// docs/investigations/codebook-rotation.md, which also measures the OTHER argument (one
// codebook shared by 75 layers) and finds 0.09% to recover against a 2% bar.

/// Weights per learned codebook entry (subvector dimension).
pub const VQ_DIM: usize = 4;
/// Codebook entries. 4096 → 12-bit index = 3.0 bpw for the indices (before scales),
/// the rate at which learned VQ ties int4 on these weights.
pub const VQ_K: usize = 4096;
/// Bits per packed codebook index (`log2(VQ_K)`).
pub const VQ_INDEX_BITS: usize = 12;
/// Weights per bf16 group scale (along the input dim).
pub const VQ_GROUP: usize = 64;

/// ‖c‖² per codebook entry — the argmin-VQ precompute, so nearest is
/// `argmin(‖c‖² − 2·x·c)` (half the flops of the full squared distance, same
/// argmin/tie-break). Shared by the CPU encoder and the GPU converter.
pub fn codebook_norms(codebook: &[f32]) -> Vec<f32> {
    (0..VQ_K)
        .map(|k| {
            codebook[k * VQ_DIM..(k + 1) * VQ_DIM]
                .iter()
                .map(|&c| c * c)
                .sum()
        })
        .collect()
}

/// Packed-index byte stride of one VQ row: `i_dim/VQ_DIM` indices × `VQ_INDEX_BITS`,
/// rounded to bytes. `i_dim` is a multiple of `VQ_DIM` and 8 for all GLM-5.2 dims,
/// so this is exact (`i_dim·VQ_INDEX_BITS / (VQ_DIM·8)`).
pub fn vq_row_bytes(i_dim: usize) -> usize {
    debug_assert_eq!(
        (i_dim / VQ_DIM * VQ_INDEX_BITS) % 8,
        0,
        "VQ row not byte-aligned"
    );
    i_dim / VQ_DIM * VQ_INDEX_BITS / 8
}

/// Number of bf16 group scales in one VQ row (`i_dim / VQ_GROUP`).
pub fn vq_groups(i_dim: usize) -> usize {
    debug_assert_eq!(
        i_dim % VQ_GROUP,
        0,
        "i_dim {i_dim} not a multiple of VQ_GROUP"
    );
    i_dim / VQ_GROUP
}

/// On-disk bytes of one VQ-quantized projection `W[o_dim, i_dim]`: `o_dim` rows of
/// packed 12-bit indices, then `o_dim · vq_groups` bf16 group scales (2 bytes each).
/// The single source of truth the converter (M2) writes and the loader (M3) reads;
/// the two arrays are stored back-to-back so one projection is one contiguous span.
pub fn vq_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * vq_row_bytes(i_dim) + o_dim * vq_groups(i_dim) * 2
}

/// O_DIRECT alignment the converter pads each expert's on-disk block up to, so one
/// expert fetch is a single block-aligned read (mirrors int4's slot alignment).
pub const VQ_ALIGN: usize = 4096;

/// The three projections of one MoE expert in on-disk order, as `(o_dim, i_dim)`:
/// gate and up map hidden→moe_inter (`i_dim = hidden`); down maps moe_inter→hidden.
pub fn vq_expert_layout(hidden: usize, moe_inter: usize) -> [(usize, usize); 3] {
    [
        (moe_inter, hidden),
        (moe_inter, hidden),
        (hidden, moe_inter),
    ]
}

/// The checkpoint tensor-name suffixes of those same three projections, in the SAME
/// order — every offline tool that walks an expert zips this against
/// [`vq_expert_layout`], so one index means one projection in both. It lives beside the
/// layout rather than in each `src/bin` (where it had been declared, identically, four
/// times) because a second copy of an ORDER-BEARING list is exactly the kind that goes
/// wrong silently: a reordered copy still compiles and still runs, and scores gate's
/// weights against up's.
pub const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// One expert's three projections: the checkpoint tensor suffix paired with the
/// `(o_dim, i_dim)` it has, in slot order. See [`expert_projs`].
pub type ExpertProjs = [(&'static str, (usize, usize)); 3];

/// [`PROJ`] already zipped against [`vq_expert_layout`] — the name/shape pairs an expert
/// encoder walks, in slot order. Here for the same reason `PROJ` is: the ZIP is as
/// order-bearing as the list, and `bin/convert` and `bin/fp8_to_i4` had spelled it out
/// identically (down to recomputing the layout once per expert inside the worker loop).
pub fn expert_projs(hidden: usize, moe_inter: usize) -> ExpertProjs {
    let [g, u, d] = vq_expert_layout(hidden, moe_inter);
    [(PROJ[0], g), (PROJ[1], u), (PROJ[2], d)]
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

/// The checkpoint tensor prefix of expert `e` in `layer`. Routed experts are numbered;
/// `e == n_experts` is the SHARED expert, which lives under an entirely different name.
///
/// Beside `PROJ` for the same reason: three tools (`convert`, `fp8_to_i4`, `i4_audit`)
/// walk a layer's `n_experts + 1` blocks in this order, and a copy that got the boundary
/// wrong would quantize the shared expert's weights into a routed slot — producing a file
/// of exactly the right size that every length check passes.
pub fn expert_base(layer: usize, e: usize, n_experts: usize) -> String {
    if e < n_experts {
        format!("model.layers.{layer}.mlp.experts.{e}")
    } else {
        format!("model.layers.{layer}.mlp.shared_experts")
    }
}

/// Sum one format's `*_proj_bytes` over an expert's three projections. The `(o, i)` dims
/// are format-independent, so this is the one place the "an expert is gate‖up‖down"
/// structure is written; the three formats differ only in `proj`.
fn expert_bytes(hidden: usize, moe_inter: usize, proj: fn(usize, usize) -> usize) -> usize {
    vq_expert_layout(hidden, moe_inter)
        .iter()
        .map(|&(o, i)| proj(o, i))
        .sum()
}

/// Round an expert's unpadded size up to [`VQ_ALIGN`], so one expert is a single
/// block-aligned O_DIRECT read. Shared by all three routed formats.
fn expert_stride(bytes: usize) -> usize {
    bytes.div_ceil(VQ_ALIGN) * VQ_ALIGN
}

/// Fill `n` consecutive expert blocks of `stride` bytes in `buf`, in parallel, calling
/// `fill(e, &mut block[..bytes])` for each. The padding between `bytes` and `stride` is
/// left as the caller found it.
///
/// One implementation for both converters. The split is by DISJOINT `split_at_mut` slices
/// rather than by index, so the borrow checker witnesses that no two workers can touch the
/// same block — with indices, an off-by-one in the chunking would be an aliasing bug the
/// compiler could not see, and the whole point of this loop is that expert `e` lands at
/// exactly `e · stride`.
pub fn fill_expert_blocks(
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
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let per = n.div_ceil(threads.max(1)).max(1);
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

/// Unpadded on-disk bytes of one expert (gate‖up‖down concatenated).
pub fn vq_expert_bytes(hidden: usize, moe_inter: usize) -> usize {
    expert_bytes(hidden, moe_inter, vq_proj_bytes)
}

/// Per-expert on-disk stride: [`vq_expert_bytes`] padded up to [`VQ_ALIGN`]. The
/// fixed stride at which experts sit in a per-layer `.vq3` file.
pub fn vq_expert_stride(hidden: usize, moe_inter: usize) -> usize {
    expert_stride(vq_expert_bytes(hidden, moe_inter))
}

/// One VQ projection borrowed from an on-disk expert block: packed 12-bit indices
/// then little-endian bf16 group scales. Run it through [`VqProj::gemv`].
pub struct VqProj<'a> {
    pub indices: &'a [u8],
    pub scales: &'a [u8],
    pub o_dim: usize,
    pub i_dim: usize,
}

impl VqProj<'_> {
    /// The borrowed group scales as the little-endian bf16 WORDS [`matvec_vq`] takes.
    ///
    /// Allocates, so it is for offline and test use only — the decode path never calls it
    /// (`matvec_vq` is handed words the loader already has, and [`vq_decode_proj`] reads
    /// the two bytes in place). It exists because the byte→word convention is THIS
    /// module's: every caller that re-spelled `u16::from_le_bytes` over `chunks_exact(2)`
    /// was a copy of a layout rule that belongs here, and one of them lives in a binary
    /// that never sees a change to this format.
    pub fn scales_u16(&self) -> Vec<u16> {
        self.scales
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

/// Slice expert `e`'s three [`VqProj`] out of a per-layer `.vq3` buffer (`n_experts`
/// blocks at [`vq_expert_stride`]). The layout the converter writes: each expert is
/// `gate ‖ up ‖ down`, each projection indices-then-scales.
pub fn vq_expert(layer: &[u8], e: usize, hidden: usize, moe_inter: usize) -> [VqProj<'_>; 3] {
    let stride = vq_expert_stride(hidden, moe_inter);
    let blk = &layer[e * stride..e * stride + vq_expert_bytes(hidden, moe_inter)];
    let dims = vq_expert_layout(hidden, moe_inter);
    let off = vq_slot_offsets(hidden, moe_inter); // the single source of the layout
    core::array::from_fn(|k| {
        let (o_dim, i_dim) = dims[k];
        let (io, so) = (off[k * 2], off[k * 2 + 1]); // indices start, scales start
        let sc_end = so + o_dim * vq_groups(i_dim) * 2;
        VqProj {
            indices: &blk[io..so], // indices span [io, so): scales begin at so
            scales: &blk[so..sc_end],
            o_dim,
            i_dim,
        }
    })
}

/// Write a 12-bit `idx` for the `k`-th subvector of a packed row. Two 12-bit indices
/// occupy 3 bytes; `12k % 8 ∈ {0,4}`, so the value never spans more than 2 bytes.
/// Public so the GPU converter (`bin/fp82vq`) packs indices identically to `quant_vq`.
#[inline]
pub fn set_idx(row: &mut [u8], k: usize, idx: u16) {
    let (base, shift) = (k * VQ_INDEX_BITS / 8, (k * VQ_INDEX_BITS) % 8);
    let v = (u32::from(idx) & 0xFFF) << shift;
    row[base] |= v as u8;
    row[base + 1] |= (v >> 8) as u8;
}

/// Read the 12-bit codebook index of the `k`-th subvector of a packed row. Public so
/// the offline readers (`bin/i4_audit`, [`vq_decode_proj`]) decode indices identically
/// to `matvec_vq`.
#[inline]
pub fn get_idx(row: &[u8], k: usize) -> usize {
    let (base, shift) = (k * VQ_INDEX_BITS / 8, (k * VQ_INDEX_BITS) % 8);
    (((row[base] as u16 | (row[base + 1] as u16) << 8) >> shift) & 0xFFF) as usize
}

/// Dequantize an fp8 (e4m3) block-scaled weight matrix `W[o_dim, i_dim]` to f32
/// (row-major). `scale` is the F32 `weight_scale_inv` tensor, one value per
/// `block × block` tile: `w[o,i] = e4m3(packed[o,i]) · scale[(o/block)·sc_cols +
/// i/block]`. The DeepSeek/GLM fp8 convention (mirrors colibri's converter); the
/// `_inv` name is historical — it is the dequant multiplier. Offline (converter)
/// use only: this materializes `o_dim·i_dim` f32.
pub fn dequant_fp8_block(
    packed: &[u8],
    scale: &[f32],
    o_dim: usize,
    i_dim: usize,
    block: usize,
) -> Vec<f32> {
    debug_assert_eq!(packed.len(), o_dim * i_dim);
    let sc_cols = i_dim.div_ceil(block);
    debug_assert_eq!(scale.len(), o_dim.div_ceil(block) * sc_cols);
    let mut out = vec![0.0f32; o_dim * i_dim];
    for o in 0..o_dim {
        for i in 0..i_dim {
            let s = scale[(o / block) * sc_cols + i / block];
            out[o * i_dim + i] = e4m3_to_f32(packed[o * i_dim + i]) * s;
        }
    }
    out
}

/// The closed-form group refit: least-squares `s = Σ w·c / Σ c·c` over the codebook
/// entries `idxs` already chose for `seg`, returned as bf16. `None` when the fit is
/// degenerate (all-zero entries, or `s ≤ 0`) — the caller keeps its amax scale.
///
/// Shared with `bin/convert`'s `--gpu` encoder, which is documented (and asserted by
/// `--validate`) to produce output BIT-IDENTICAL to [`quant_vq`]. Two copies of this
/// accumulation is two places for one to pick up a different summation order and turn
/// that guarantee into a mismatch report against the shipped bytes.
pub fn vq_refit(seg: &[f32], idxs: &[u16], codebook: &[f32]) -> Option<u16> {
    let (mut num, mut den) = (0.0f32, 0.0f32);
    for (t, chunk) in seg.chunks_exact(VQ_DIM).enumerate() {
        let c = &codebook[idxs[t] as usize * VQ_DIM..][..VQ_DIM];
        for (d, &v) in chunk.iter().enumerate() {
            num += v * c[d];
            den += c[d] * c[d];
        }
    }
    (den > 0.0 && num > 0.0).then(|| f32_to_bf16(num / den))
}

/// Quantize a row-major f32 weight matrix `W[o_dim, i_dim]` to VQ-int3 against a
/// learned `codebook` (`VQ_K · VQ_DIM` f32). Returns `(indices, scales)`: `indices`
/// is `o_dim · vq_row_bytes(i_dim)` packed 12-bit codebook indices; `scales` is
/// `o_dim · (i_dim/VQ_GROUP)` bf16 group scales.
///
/// Per group the encode alternates once: assign subvectors to nearest entries under
/// the amax-derived scale, refit the scale in closed form against the chosen entries
/// (least-squares: `s = Σ w·c / Σ c·c`), then re-assign under the refitted scale.
/// The stored bf16 scale is the one the final assignment used, so the oracle
/// reproduces the on-disk values exactly.
pub fn quant_vq(w: &[f32], o_dim: usize, i_dim: usize, codebook: &[f32]) -> (Vec<u8>, Vec<u16>) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    debug_assert_eq!(
        i_dim % VQ_GROUP,
        0,
        "i_dim {i_dim} not a multiple of VQ_GROUP"
    );
    const SUBS: usize = VQ_GROUP / VQ_DIM; // subvectors per group
    let ngroups = i_dim / VQ_GROUP;
    let rb = vq_row_bytes(i_dim);
    let mut indices = vec![0u8; o_dim * rb];
    let mut scales = vec![0u16; o_dim * ngroups];
    let norms = codebook_norms(codebook);
    let nearest = |sub: &[f32; VQ_DIM]| -> u16 {
        let mut best = (f32::INFINITY, 0u16);
        for k in 0..VQ_K {
            let c = &codebook[k * VQ_DIM..(k + 1) * VQ_DIM];
            let dot: f32 = sub.iter().zip(c).map(|(&a, &b)| a * b).sum();
            let d = norms[k] - 2.0 * dot;
            if d < best.0 {
                best = (d, k as u16);
            }
        }
        best.1
    };
    let assign = |seg: &[f32], inv: f32, idxs: &mut [u16; SUBS]| {
        let mut sub = [0.0f32; VQ_DIM];
        for (t, chunk) in seg.chunks_exact(VQ_DIM).enumerate() {
            for (d, &v) in chunk.iter().enumerate() {
                sub[d] = v * inv; // normalize into the codebook's scale
            }
            idxs[t] = nearest(&sub);
        }
    };
    for o in 0..o_dim {
        let wr = &w[o * i_dim..(o + 1) * i_dim];
        let ir = &mut indices[o * rb..(o + 1) * rb];
        for grp in 0..ngroups {
            let seg = &wr[grp * VQ_GROUP..(grp + 1) * VQ_GROUP];
            let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let mut sb = f32_to_bf16(if amax > 0.0 { amax } else { 1.0 });
            let mut idxs = [0u16; SUBS];
            assign(seg, 1.0 / bf16_to_f32(sb), &mut idxs);
            // Refit against the chosen entries, then re-assign under the new scale.
            // A degenerate fit keeps the amax scale and the first assignment.
            if let Some(refit) = vq_refit(seg, &idxs, codebook)
                && refit != sb
            {
                sb = refit;
                assign(seg, 1.0 / bf16_to_f32(sb), &mut idxs);
            }
            scales[o * ngroups + grp] = sb;
            for (t, &ix) in idxs.iter().enumerate() {
                set_idx(ir, grp * SUBS + t, ix);
            }
        }
    }
    (indices, scales)
}

/// GEMV oracle against a VQ-int3 matrix: `y[o] = Σ_subvec scale[o, group(subvec)] ·
/// codebook[idx(o,subvec)] · x[subvec]`. The scalar reference the HIP VQ kernel (M3)
/// is validated against. Codebook entries and group scales decoded inline; no
/// per-expert f32 materialization.
pub fn matvec_vq(
    y: &mut [f32],
    x: &[f32],
    indices: &[u8],
    scales: &[u16],
    codebook: &[f32],
    o_dim: usize,
    i_dim: usize,
) {
    let ngroups = i_dim / VQ_GROUP;
    let nsub = i_dim / VQ_DIM;
    let rb = vq_row_bytes(i_dim);
    debug_check_gemv(y, x, o_dim, i_dim);
    for (o, yo) in y.iter_mut().enumerate() {
        let ir = &indices[o * rb..(o + 1) * rb];
        let mut acc = 0.0f32;
        for k in 0..nsub {
            let i0 = k * VQ_DIM;
            let s = bf16_to_f32(scales[o * ngroups + i0 / VQ_GROUP]);
            let c = &codebook[get_idx(ir, k) * VQ_DIM..][..VQ_DIM];
            let mut dot = 0.0f32;
            for (d, &cw) in c.iter().enumerate() {
                dot += x[i0 + d] * cw;
            }
            acc += s * dot;
        }
        *yo = acc;
    }
}

/// Decode one VQ projection to a DENSE row-major `W[o_dim, i_dim]` — the inverse of
/// [`quant_vq`], and the SINGLE VQ reader: `bin/i4_audit` compares what this returns
/// against the fp8 ground truth, and any audit that reproduces the retired
/// `vq3 → int4` chain must decode identically or its "old set" baseline describes a
/// converter that never existed. (The same rule [`write_i4_proj`] states for the `.i4`
/// writer.)
///
/// Materializes `o_dim·i_dim` f32 — offline use only; the decode path is
/// [`matvec_vq`], which never builds the dense matrix.
pub fn vq_decode_proj(p: &VqProj, codebook: &[f32]) -> Vec<f32> {
    let (o_dim, i_dim) = (p.o_dim, p.i_dim);
    let (rb, ng, nsub) = (vq_row_bytes(i_dim), vq_groups(i_dim), i_dim / VQ_DIM);
    let mut w = vec![0f32; o_dim * i_dim];
    for o in 0..o_dim {
        let ir = &p.indices[o * rb..(o + 1) * rb];
        for k in 0..nsub {
            let g = (o * ng + (k * VQ_DIM) / VQ_GROUP) * 2;
            let s = bf16_to_f32(u16::from_le_bytes([p.scales[g], p.scales[g + 1]]));
            let c = &codebook[get_idx(ir, k) * VQ_DIM..][..VQ_DIM];
            for (d, &cw) in c.iter().enumerate() {
                w[o * i_dim + k * VQ_DIM + d] = s * cw;
            }
        }
    }
    w
}

// ── codebook learning ───────────────────────────────────────────────────────
//
// Lives here rather than in the converter because it once had a second consumer, the
// per-layer-codebook study, which had to fit codebooks the SAME way the shipped one was
// fitted or the comparison would measure the fitting procedure instead of the thing it
// was pricing. That study closed negative (docs/investigations/codebook-rotation.md,
// 2026-08-01: 0.09% recovered against a 2% bar) and its binary is gone — recover it from
// tag `archive/vq-study`. `convert` is the only caller now.

/// Append every `stride`-th group-normalized subvector of `w` to the codebook sample.
pub fn sample_subvectors(w: &[f32], i_dim: usize, stride: usize, out: &mut Vec<f32>) {
    let mut n = 0usize;
    for row in w.chunks_exact(i_dim) {
        for grp in row.chunks_exact(VQ_GROUP) {
            let amax = grp.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let inv = 1.0 / bf16_to_f32(f32_to_bf16(if amax > 0.0 { amax } else { 1.0 }));
            for sub in grp.chunks_exact(VQ_DIM) {
                if n.is_multiple_of(stride) {
                    out.extend(sub.iter().map(|&v| v * inv));
                }
                n += 1;
            }
        }
    }
}

/// k-means (k-means++ seed, threaded Lloyd, convergence-stopped) → VQ_K·VQ_DIM.
pub fn learn_codebook(sample: &[f32], max_iters: usize) -> Vec<f32> {
    let n = sample.len() / VQ_DIM;
    assert!(n >= VQ_K, "sample {n} < VQ_K {VQ_K}");
    let threads = std::thread::available_parallelism().map_or(4, |t| t.get());
    let mut c = vec![0.0f32; VQ_K * VQ_DIM];
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut mind = vec![1.0f32; n];
    for j in 0..VQ_K {
        let total: f64 = mind.iter().map(|&d| d as f64).sum();
        let mut t = next() * total;
        let mut pick = n - 1;
        for (i, &d) in mind.iter().enumerate() {
            t -= d as f64;
            if t <= 0.0 {
                pick = i;
                break;
            }
        }
        let seed = &sample[pick * VQ_DIM..(pick + 1) * VQ_DIM];
        c[j * VQ_DIM..(j + 1) * VQ_DIM].copy_from_slice(seed);
        for (i, md) in mind.iter_mut().enumerate() {
            let v = &sample[i * VQ_DIM..(i + 1) * VQ_DIM];
            let d: f32 = v.iter().zip(seed).map(|(&x, &y)| (x - y) * (x - y)).sum();
            *md = if j == 0 { d } else { md.min(d) };
        }
    }
    let mut prev = f64::INFINITY;
    for _ in 0..max_iters {
        let chunk = n.div_ceil(threads);
        let parts: Vec<(Vec<f32>, Vec<u32>, f64)> = std::thread::scope(|s| {
            let mut hs = Vec::new();
            for t in 0..threads {
                let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(n));
                if lo >= hi {
                    break;
                }
                let c = &c;
                hs.push(s.spawn(move || {
                    let mut sum = vec![0.0f32; VQ_K * VQ_DIM];
                    let mut cnt = vec![0u32; VQ_K];
                    let mut dist = 0.0f64;
                    for i in lo..hi {
                        let v = &sample[i * VQ_DIM..(i + 1) * VQ_DIM];
                        let mut best = (f32::INFINITY, 0usize);
                        for j in 0..VQ_K {
                            let cc = &c[j * VQ_DIM..(j + 1) * VQ_DIM];
                            let d: f32 = v.iter().zip(cc).map(|(&x, &y)| (x - y) * (x - y)).sum();
                            if d < best.0 {
                                best = (d, j);
                            }
                        }
                        dist += best.0 as f64;
                        cnt[best.1] += 1;
                        for d in 0..VQ_DIM {
                            sum[best.1 * VQ_DIM + d] += v[d];
                        }
                    }
                    (sum, cnt, dist)
                }));
            }
            hs.into_iter().filter_map(|h| h.join().ok()).collect()
        });
        let mut sum = vec![0.0f32; VQ_K * VQ_DIM];
        let mut cnt = vec![0u32; VQ_K];
        let mut dist = 0.0f64;
        for (ps, pc, pd) in parts {
            for (a, b) in sum.iter_mut().zip(ps) {
                *a += b;
            }
            for (a, b) in cnt.iter_mut().zip(pc) {
                *a += b;
            }
            dist += pd;
        }
        for j in 0..VQ_K {
            if cnt[j] > 0 {
                for d in 0..VQ_DIM {
                    c[j * VQ_DIM + d] = sum[j * VQ_DIM + d] / cnt[j] as f32;
                }
            }
        }
        if ((prev - dist) / dist.max(f64::MIN_POSITIVE)).abs() < 1e-4 {
            break;
        }
        prev = dist;
    }
    c
}

// ── int4 (group-scaled): the "warm expert" format ───────────────────────────
// Symmetric int4 with one f32 scale per [`I4_GROUP`] weights ALONG THE INPUT DIM:
// `W[o,i] = (nibble(o,i) − 8) · scale[o·ngroups + i/I4_GROUP]`. `packed` = o_dim rows
// of `i4_row_bytes(i_dim)` bytes (LOW nibble = col 2j, HIGH = col 2j+1).
//
// This replaced a PER-ROW scale (`scale[o]`, i.e. one step for all 6144 weights of a
// gate/up row), which is a known-bad design point and measured like one: a single
// outlier set the step for the whole row and rounded the bulk toward zero — 603 of
// 6144 rows past 50% zeros on L03 e0 down_proj, and `--mode int4` PPL 73.4 against
// int3-vq's 5.28. `.vq3` already carries one scale per `VQ_GROUP` = 64 weights, which
// is why it does not suffer this. Group scales are what the int4 literature and every
// published int4 GLM checkpoint (AWQ/GPTQ, `group_size=128`) actually use.
//
// The scale therefore lives INSIDE the dot (each group's partial is scaled before it
// is accumulated), not outside it — see `dot_i4_wave` in kernels/common.hpp.

/// Weights per f32 int4 group scale, along the input dimension. 128 is the
/// AWQ/GPTQ/Marlin default; 64 (what `.vq3` uses) is the other point worth sweeping.
/// A multiple of 8, so the GPU dot's 8-nibble dword fast path never straddles a group.
pub const I4_GROUP: usize = 128;

/// Number of f32 group scales in one int4 row (`ceil(i_dim / I4_GROUP)`).
pub fn i4_groups(i_dim: usize) -> usize {
    i_dim.div_ceil(I4_GROUP)
}

/// Row stride in bytes for an int4 matrix (2 nibbles/byte).
pub fn i4_row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

// --- FP4 (`.f4`) — DeepSeek-V4-Flash's native routed-expert format --------------------
//
// Chosen 2026-08-03 to keep source fidelity: V4-Flash ships its 296.35B routed-expert
// params ALREADY at 4 bits (OCP MX e2m1 nibbles with e8m0 block scales), so producing a
// `.f4` artifact is a REPACK into rivoli's O_DIRECT-aligned block layout, not a
// quantization. There is no fit, no codebook, and no error introduced — which is the whole
// point: re-quantizing a 4-bit source into 3.25-bit int3-vq is the lossy-on-lossy chain
// docs/investigations/int4-scales.md records at PPL 73.43. See other-models.md §5.
//
// Unlike int4 (uniform levels, f32 group scale), e2m1 is a fixed NON-uniform 16-value
// codebook and the scale is a bare power of two, so dequant is a 16-entry lookup times a
// shift. That makes it cheaper on the GPU than either neighbour, not more expensive.

/// Weights per e8m0 scale along the input dim. 32 is the OCP MX block size, which is what
/// V4-Flash's `.scale` tensors carry (1×32 sub-blocks nested inside its 128×128 fp8
/// blocks). Distinct from `I4_GROUP`(128) and `VQ_GROUP`(64) on purpose — this one is
/// dictated by the source, not chosen by us.
pub const F4_GROUP: usize = 32;

/// The 16 e2m1 values, indexed by nibble. Bit 3 is sign, bits 2-1 exponent, bit 0
/// mantissa; there are no infinities and no NaN — every code is a finite value, which is
/// why a bare lookup is a complete decoder.
pub const F4_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one e8m0 scale byte: a bare exponent, value `2^(b - 127)`. `0xFF` is the
/// format's NaN and must not appear in weights — a scale that is NaN poisons a whole
/// 32-weight block, so it is worth failing on rather than propagating.
pub fn e8m0(b: u8) -> anyhow::Result<f32> {
    if b == 0xFF {
        anyhow::bail!("e8m0 scale byte 0xFF is NaN");
    }
    // exp2 rather than `bits << 23`: the bit form is exact only for b >= 1, and b = 0 is
    // 2^-127, an f32 SUBNORMAL that the shift encodes as +0.
    Ok(f32::exp2(b as f32 - 127.0))
}

/// Number of e8m0 group scales in one FP4 row (`ceil(i_dim / F4_GROUP)`).
pub fn f4_groups(i_dim: usize) -> usize {
    i_dim.div_ceil(F4_GROUP)
}

/// Row stride in bytes for an FP4 matrix (2 nibbles/byte) — same packing as int4, so the
/// `.i4` container's nibble addressing carries over unchanged.
pub fn f4_row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

/// On-disk bytes of one FP4-packed projection `W[o_dim, i_dim]`: packed nibbles, then one
/// e8m0 byte per group. Mirrors `vq_proj_bytes`/`i4_proj_bytes` so the block layout and
/// `VQ_ALIGN` stride math are shared across all three formats.
pub fn f4_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * f4_row_bytes(i_dim) + o_dim * f4_groups(i_dim)
}

/// Unpadded on-disk bytes of one FP4 expert (w1‖w1_scale‖w3‖w3_scale‖w2‖w2_scale — see
/// [`V4_PROJ`] for why that is gate/up/down order).
pub fn f4_expert_bytes(hidden: usize, moe_inter: usize) -> usize {
    expert_bytes(hidden, moe_inter, f4_proj_bytes)
}

/// Per-expert on-disk stride: [`f4_expert_bytes`] padded up to [`VQ_ALIGN`], so one FP4
/// expert is a single block-aligned read (mirrors the `.vq3`/`.i4` stride).
pub fn f4_expert_stride(hidden: usize, moe_inter: usize) -> usize {
    expert_stride(f4_expert_bytes(hidden, moe_inter))
}

/// fp8 `weight_scale_inv` tile size: 128×128 for both the GLM-5.2 checkpoint and
/// DeepSeek-V4-Flash's `quantization_config.weight_block_size`. Distinct from
/// [`F4_GROUP`] (32, along the input dim only), which is the FP4 experts' scheme.
pub const FP8_BLOCK: usize = 128;

// --- DeepSeek-V4-Flash tensor naming ------------------------------------------------
//
// V4's checkpoint uses its reference implementation's names, NOT HuggingFace's: no
// `model.` prefix, `attn`/`ffn` rather than `self_attn`/`mlp`, `.scale` rather than
// `.weight_scale_inv`, and `w1`/`w3`/`w2` rather than `gate_proj`/`up_proj`/`down_proj`.
// Verified against the shipped `model.safetensors.index.json` (72,317 entries), 2026-08-04.

/// V4's three expert projections in the SAME slot order as [`PROJ`] — i.e. gate, up, down.
///
/// **The order is `w1, w3, w2` and that is not a typo.** `inference/model.py`'s
/// `Expert.forward` is `gate = self.w1(x)`, `up = self.w3(x)`, then `return self.w2(…)`,
/// so w3 is the UP projection and w2 is down. Storing them in gate/up/down order keeps one
/// slot index meaning one projection across all three of this engine's formats.
///
/// A `w2` in the wrong slot is caught by its shape (`[hidden, moe_inter]`, transposed from
/// the other two). A `w1`/`w3` SWAP is not: they are the same shape, and a repack that
/// swapped them would be internally consistent and byte-clean. Only a numerical oracle
/// against the reference can see that, which is what S1b exists for.
pub const V4_PROJ: [&str; 3] = ["w1", "w3", "w2"];

/// [`V4_PROJ`] zipped against [`vq_expert_layout`] — the V4 analogue of [`expert_projs`].
pub fn v4_expert_projs(hidden: usize, moe_inter: usize) -> ExpertProjs {
    let [g, u, d] = vq_expert_layout(hidden, moe_inter);
    [(V4_PROJ[0], g), (V4_PROJ[1], u), (V4_PROJ[2], d)]
}

/// V4's tensor prefix for expert `e` in `layer`; `e == n_experts` is the SHARED expert.
/// The V4 analogue of [`expert_base`], and the boundary matters for the same reason —
/// except that in V4 the two are not even the same *format*: routed experts are FP4
/// (`I8` nibble pairs + `F8_E8M0` scales) and the shared expert is `F8_E4M3` at 128×128,
/// so a block written past the boundary is not merely the wrong weights, it is the wrong
/// arithmetic. `.f4` therefore holds routed experts ONLY; the shared expert rides the
/// resident fp8 path.
pub fn v4_expert_base(layer: usize, e: usize, n_experts: usize) -> String {
    if e < n_experts {
        format!("layers.{layer}.ffn.experts.{e}")
    } else {
        format!("layers.{layer}.ffn.shared_experts")
    }
}

// jscpd:ignore-start — the four `matvec_*` oracles' PARAMETER LISTS. `matvec_i4` and
// `matvec_i8` take character-identical arguments (`y, x, packed, scale, o_dim, i_dim`),
// which rustfmt expands past 100 columns into eight lines — enough to clone against each
// other and against a caller's. No BODY duplication remains: the shared row loop is
// `matvec_bytes` and the shared entry contract is `debug_check_gemv`.
//
// The honest fix is a `Weights<'_>` sum type (`Fp8{packed,scale,block} | I4{..} | I8{..} |
// Vq{..}`), which would also make "int4 scales with an int8 packed array" unrepresentable
// — worth doing, but it rewrites call sites in i4_audit, convert, tests/kernel.rs and
// tests/common. Renaming a parameter to break the hash was the alternative and is exactly
// the masking this gate exists to undo.
/// Reference int4 GEMV `y[o] = Σ_i x[i]·(nibble(o,i) − 8)·scale[o, i/I4_GROUP]` — the
/// CPU oracle the `moe_gateup_i4`/`moe_down_i4` kernels validate against. `scale` is
/// `o_dim · i4_groups(i_dim)` f32, row-major.
pub fn matvec_i4(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scale: &[f32],
    o_dim: usize,
    i_dim: usize,
) {
    let (rb, ng) = (i4_row_bytes(i_dim), i4_groups(i_dim));
    debug_check_gemv(y, x, o_dim, i_dim);
    debug_assert_eq!(scale.len(), o_dim * ng);
    for (o, yo) in y.iter_mut().enumerate() {
        let row = &packed[o * rb..(o + 1) * rb];
        let srow = &scale[o * ng..(o + 1) * ng];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            let byte = row[i >> 1];
            let n = (if i & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as i32 - 8;
            acc += xi * n as f32 * srow[i / I4_GROUP];
        }
        *yo = acc;
    }
}

/// Reference int8 GEMV `y[o] = scale[o] · Σ_i x[i]·(i8)packed[o·i_dim+i]` — the CPU
/// oracle for the `gemv_i8` kernel (lm_head → logits). `packed` is raw bytes
/// reinterpreted as signed, matching the kernel's `signed char`.
pub fn matvec_i8(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scale: &[f32],
    o_dim: usize,
    i_dim: usize,
) {
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
// jscpd:ignore-end

/// Quantize `w[o_dim·i_dim]` (row-major) → group-scaled symmetric int4 (packed bytes
/// plus `o_dim · i4_groups(i_dim)` f32 scales). Per group of [`I4_GROUP`] weights along
/// the input dim, `s = max|group|/7` so the group's extreme maps to nibble 15 (value +7);
/// nibbles clamp to `[0,15]`. Round-trips through [`matvec_i4`].
//  (Rewrapped so no line STARTS with `+`: rustdoc reads a leading `+ ` as a list bullet,
//  which is what `clippy::doc_lazy_continuation` was flagging.)
///
/// The scale is per GROUP and not per ROW because an outlier only ever coarsens its
/// own 128 weights instead of the whole 6144-wide row — see the module comment above
/// for the measurement that forced the change.
pub fn quant_i4(w: &[f32], o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    let (rb, ng) = (i4_row_bytes(i_dim), i4_groups(i_dim));
    let mut packed = vec![0u8; o_dim * rb];
    let mut scale = vec![0.0f32; o_dim * ng];
    for o in 0..o_dim {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        for (g, seg) in row.chunks(I4_GROUP).enumerate() {
            let amax = seg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
            scale[o * ng + g] = s;
            for (t, &wi) in seg.iter().enumerate() {
                let i = g * I4_GROUP + t;
                let q = ((wi / s).round() as i32 + 8).clamp(0, 15) as u8;
                let bi = o * rb + (i >> 1);
                if i & 1 == 0 {
                    packed[bi] = (packed[bi] & 0xF0) | q;
                } else {
                    packed[bi] = (packed[bi] & 0x0F) | (q << 4);
                }
            }
        }
    }
    (packed, scale)
}

/// Reconstruct the dense `W[o_dim, i_dim]` a `(packed, scale)` int4 pair represents —
/// the inverse of [`quant_i4`], and the SINGLE int4 reader for offline use.
///
/// Deliberately spells out the nibble convention rather than calling [`matvec_i4`]
/// with basis vectors: an audit that reconstructs weights through the very routine it
/// is auditing cannot detect a decode bug. The round trip `dequant_i4(quant_i4(w))` is
/// unit-tested below, which is what keeps the two spellings honest.
pub fn dequant_i4(packed: &[u8], scale: &[f32], o_dim: usize, i_dim: usize) -> Vec<f32> {
    let (rb, ng) = (i4_row_bytes(i_dim), i4_groups(i_dim));
    debug_assert_eq!(scale.len(), o_dim * ng);
    let mut w = vec![0f32; o_dim * i_dim];
    for o in 0..o_dim {
        let row = &packed[o * rb..(o + 1) * rb];
        let srow = &scale[o * ng..(o + 1) * ng];
        for i in 0..i_dim {
            let b = row[i >> 1];
            let n = (if i & 1 == 0 { b & 0x0F } else { b >> 4 }) as i32 - 8;
            w[o * i_dim + i] = n as f32 * srow[i / I4_GROUP];
        }
    }
    w
}

/// On-disk bytes of one int4 projection `W[o_dim, i_dim]`: `o_dim` packed rows then
/// `o_dim · i4_groups(i_dim)` f32 group scales, back-to-back (one projection = one
/// contiguous span).
pub fn i4_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * i4_row_bytes(i_dim) + o_dim * i4_groups(i_dim) * 4
}

/// Unpadded on-disk bytes of one int4 expert (gate‖gate_scale‖up‖up_scale‖down‖
/// down_scale). Reuses [`vq_expert_layout`] — the (o,i) dims are format-independent.
pub fn i4_expert_bytes(hidden: usize, moe_inter: usize) -> usize {
    expert_bytes(hidden, moe_inter, i4_proj_bytes)
}

/// Per-expert on-disk stride: [`i4_expert_bytes`] padded up to [`VQ_ALIGN`], so one
/// int4 expert is a single block-aligned read (mirrors the `.vq3` stride).
pub fn i4_expert_stride(hidden: usize, moe_inter: usize) -> usize {
    expert_stride(i4_expert_bytes(hidden, moe_inter))
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
    hidden: usize,
    moe_inter: usize,
    span: impl Fn(usize, usize) -> (usize, usize),
) -> [usize; 6] {
    let mut off = [0usize; 6];
    let mut base = 0usize;
    for (p, &(o, i)) in vq_expert_layout(hidden, moe_inter).iter().enumerate() {
        let (packed, scales) = span(o, i);
        off[p * 2] = base;
        off[p * 2 + 1] = base + packed;
        base += packed + scales;
    }
    off
}

/// The six byte offsets within one int3-VQ expert block:
/// `[gate.indices, gate.scales, up.indices, up.scales, down.indices, down.scales]`.
/// The SINGLE source of the `.vq3` slot layout — [`vq_expert`] slices by it and the pin
/// points its descriptors at it, so the two cannot disagree.
pub fn vq_slot_offsets(hidden: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(hidden, moe_inter, |o, i| {
        (o * vq_row_bytes(i), o * vq_groups(i) * 2)
    })
}

/// The six byte offsets within one int4 expert block, in expert-descriptor field
/// order (moe.hip's int4 interpretation of the shared `ExpertDesc` six-pointer layout).
/// Every packed span starts 4-byte aligned (rows are `i_dim/2`, i_dim a multiple of
/// 8; scale spans are whole f32), so `dot_i4_wave`'s dword fast path stays valid.
pub fn i4_slot_offsets(hidden: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(hidden, moe_inter, |o, i| {
        (o * i4_row_bytes(i), o * i4_groups(i) * 4)
    })
}

/// The six byte offsets within one FP4 expert block, in `ExpertDescF4` field order:
/// `[w1, w1_scale, w3, w3_scale, w2, w2_scale]` — see [`V4_PROJ`] for why that is
/// gate/up/down order.
///
/// The scale span is ONE BYTE per group, not two (`.vq3` bf16) or four (`.i4` f32): e8m0 is
/// a bare exponent. That is the whole difference from its two siblings, and it is why
/// `backend::ExpertDescF4` carries `*const u8` scale pointers where `ExpertDesc` carries
/// `*const u16` — a `.f4` block resolved through the int4 offsets would put every
/// projection but the first at the wrong address, and the ones it did find it would decode
/// at group 128 against a uniform codebook.
pub fn f4_slot_offsets(hidden: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(hidden, moe_inter, |o, i| {
        (o * f4_row_bytes(i), o * f4_groups(i))
    })
}

/// Write projection `k`'s packed nibbles + f32 group scales into an expert block at
/// the offsets [`i4_slot_offsets`] defines. The SINGLE writer of the `.i4` slot layout
/// — `bin/fp8_to_i4` goes through it, as did the retired `vq3_to_i4`, so no two
/// producers can disagree on where a projection's bytes land (the same rule
/// `vq_slot_offsets` states for `.vq3`). `packed`/`scale` are exactly what [`quant_i4`] returned for this
/// projection; a short `scale` would leave the tail of the span holding the PREVIOUS
/// projection's bytes, so the lengths are checked rather than trusted.
pub fn write_i4_proj(slot: &mut [u8], off: &[usize; 6], k: usize, packed: &[u8], scale: &[f32]) {
    debug_assert!(
        off[k * 2] + packed.len() <= off[k * 2 + 1],
        "packed overruns its span"
    );
    // The scale span must END where the next projection begins (and the last one
    // inside the slot): a `scale` sized for a different `I4_GROUP` would otherwise
    // run into — or leave stale bytes in — the neighbouring projection.
    let scale_end = off[k * 2 + 1] + scale.len() * 4;
    match off.get(k * 2 + 2) {
        Some(&next) => debug_assert_eq!(scale_end, next, "scale span != i4_slot_offsets"),
        None => debug_assert!(scale_end <= slot.len(), "scale span overruns the slot"),
    }
    let po = off[k * 2];
    slot[po..po + packed.len()].copy_from_slice(packed);
    write_le_scales(
        &mut slot[off[k * 2 + 1]..],
        scale.iter().map(|s| s.to_le_bytes()),
    );
}

#[cfg(test)]
mod tests {
    // Quantization tests compare against hand-computed constants; an `unwrap` that fires
    // IS the failure report. Crate-wide these are `deny`.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Deterministic uniform `[-1, 1)` stream. One spelling of the LCG the tests below
    /// draw weights from, seeded per test so no two share a state and a failure
    /// reproduces from its seed alone.
    fn uniform(seed: u64) -> impl FnMut() -> f32 {
        let mut st = seed;
        move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((st >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// `dequant_i4` is a SECOND spelling of the nibble convention `matvec_i4` and
    /// `quant_i4` carry (deliberately — an audit that reconstructs weights through the
    /// routine it audits cannot see a decode bug). This is what keeps the spellings
    /// honest: the round trip must land inside one quantiser step, and `matvec_i4` on
    /// the packed bytes must equal a plain dot against the reconstruction.
    ///
    /// The step bound is the assertion that matters, and it is now PER GROUP:
    /// `amax(group)/7` with round-to-nearest puts every weight within `s_g/2` of a grid
    /// point, so `max|w - ŵ| ≤ s_g/2` holds inside each group. An implementation that
    /// dropped the −8 zero point, swapped the nibble halves, or — the regression this
    /// test exists to catch — kept ONE scale for the whole row would blow past it,
    /// because the row's groups differ in magnitude by 10^g here. An error of exactly
    /// zero would mean no rounding happened at all and the test is measuring nothing.
    #[test]
    fn dequant_i4_inverts_quant_i4_within_one_step_per_group() {
        // 3 full groups + a partial 4th, so the div_ceil group count is exercised too.
        let (o_dim, i_dim) = (5usize, I4_GROUP * 3 + 16);
        let ng = i4_groups(i_dim);
        assert_eq!(ng, 4);
        let mut rnd = uniform(0x1234_5678);
        // Group `g` of every row scaled by 10^g: a single per-row scale could not stay
        // within half a step of the SMALL groups, so this separates the two formats.
        let mut w: Vec<f32> = (0..o_dim * i_dim)
            .map(|n| rnd() * 10f32.powi(((n % i_dim) / I4_GROUP) as i32))
            .collect();
        for o in 0..o_dim {
            for g in 0..ng {
                w[o * i_dim + g * I4_GROUP] = 9.0 * 10f32.powi(g as i32); // the amax setter
            }
        }
        let (packed, scale) = quant_i4(&w, o_dim, i_dim);
        assert_eq!(scale.len(), o_dim * ng);
        let back = dequant_i4(&packed, &scale, o_dim, i_dim);
        for o in 0..o_dim {
            for g in 0..ng {
                let s = scale[o * ng + g];
                let cols = g * I4_GROUP..((g + 1) * I4_GROUP).min(i_dim);
                let err = cols
                    .map(|i| (w[o * i_dim + i] - back[o * i_dim + i]).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    err <= s * 0.5 + 1e-6,
                    "row {o} group {g}: max round-trip error {err:.6e} exceeds half a step ({:.6e})",
                    s * 0.5
                );
                assert!(
                    err > 0.0,
                    "row {o} group {g}: zero error means no rounding happened"
                );
            }
        }
        // The GEMV oracle and the dense reconstruction must agree on the same bytes.
        let x: Vec<f32> = (0..i_dim).map(|_| rnd()).collect();
        let mut y = vec![0f32; o_dim];
        matvec_i4(&mut y, &x, &packed, &scale, o_dim, i_dim);
        for (o, &yo) in y.iter().enumerate() {
            let want: f32 = (0..i_dim).map(|i| back[o * i_dim + i] * x[i]).sum();
            assert!(
                (yo - want).abs() <= 1e-4 * want.abs().max(1.0),
                "row {o}: {yo} != {want}"
            );
        }
    }

    /// Group scales must beat a per-row scale on weights whose magnitude varies ALONG
    /// the row — the whole reason for the format change. Reconstruction error is
    /// measured against a per-row `amax/7` quantiser spelled out here, so the
    /// comparison does not depend on the old implementation still existing.
    ///
    /// Scored on the BULK — every column outside the row's one outlier group — and not
    /// on the whole row. Whole-row rel-L2 is dominated by the outlier group, which both
    /// quantisers represent about equally well; it comes out ~0.071 either way and
    /// hides the entire effect. The bulk is where decode quality lives and where the
    /// per-row scale rounds weights to zero, so that is what the assertion prices.
    #[test]
    fn group_scales_beat_a_per_row_scale_on_the_bulk() {
        let (o_dim, i_dim) = (4usize, I4_GROUP * 8);
        let mut rnd = uniform(0xACE1);
        // One outlier group per row, 1000× the rest — the pathology a per-row scale
        // cannot absorb (it rounds the other 7/8 of the row toward zero).
        let outlier = |o: usize| o % 8;
        let w: Vec<f32> = (0..o_dim * i_dim)
            .map(|n| {
                let big = (n % i_dim) / I4_GROUP == outlier(n / i_dim);
                rnd() * if big { 1000.0 } else { 1.0 }
            })
            .collect();
        // rel-L2 and zero-fraction over the non-outlier columns only.
        let bulk = |rec: &[f32]| -> (f64, f64) {
            let (mut n, mut d, mut z, mut c) = (0f64, 0f64, 0usize, 0usize);
            for o in 0..o_dim {
                for i in 0..i_dim {
                    if i / I4_GROUP == outlier(o) {
                        continue;
                    }
                    let (a, b) = (w[o * i_dim + i] as f64, rec[o * i_dim + i] as f64);
                    (n, d, c) = (n + (b - a) * (b - a), d + a * a, c + 1);
                    z += usize::from(b == 0.0);
                }
            }
            ((n / d).sqrt(), z as f64 / c as f64)
        };
        let (packed, scale) = quant_i4(&w, o_dim, i_dim);
        let (g_rel, g_zero) = bulk(&dequant_i4(&packed, &scale, o_dim, i_dim));
        // Per-row reference: s = max|row|/7, round-to-nearest, clamp to [-8, 7].
        let per_row: Vec<f32> = w
            .chunks_exact(i_dim)
            .flat_map(|row| {
                let s = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 7.0;
                row.iter()
                    .map(move |&v| ((v / s).round() as i32).clamp(-8, 7) as f32 * s)
            })
            .collect();
        let (r_rel, r_zero) = bulk(&per_row);
        assert!(
            g_rel * 4.0 < r_rel,
            "bulk relL2: group-{I4_GROUP} {g_rel:.4} is not decisively better than per-row {r_rel:.4}"
        );
        // The mechanism, not just the score: the per-row scale rounds nearly the whole
        // bulk to zero, the group scale rounds almost none of it.
        assert!(
            r_zero > 0.9,
            "per-row bulk should be almost all zeros, got {r_zero:.3}"
        );
        assert!(
            g_zero < 0.1,
            "group bulk should barely round to zero, got {g_zero:.3}"
        );
    }

    /// **The `.f4` slot layout, pinned against the SHIPPED artifact's own geometry — and
    /// it is byte-identical to `.i4`'s, which the name says because the first draft of this
    /// test asserted the opposite and failed.**
    ///
    /// Not a restatement of `slot_offsets`: the six numbers are written out, and they are
    /// checked against a file length nothing here computes — `L00.f4` is
    /// `4096 + 256 × 13369344 = 3422556160` bytes, verified on disk. A layout that agreed
    /// with itself but not with the converter would move at least one of them.
    ///
    /// **And it asserts that `.f4` and `.i4` are byte-identical here, which is not what was
    /// expected and is the more useful half.** Both pack nibbles at `i_dim/2` bytes a row;
    /// `.f4` spends `ceil(i/32) × 1` byte on scales and `.i4` spends `ceil(i/128) × 4`, and
    /// those are the SAME NUMBER whenever `i_dim` is a multiple of 128 — which it is for
    /// every dimension either model ships. So at V4's dims the two formats agree on all six
    /// offsets, on `*_expert_bytes`, and therefore on the whole file length.
    ///
    /// The consequence is worth stating where someone will read it: **nothing structural can
    /// separate an `.f4` from an `.i4`.** Not the length, not the slot offsets, not a
    /// descriptor's six addresses — an `.f4` block resolved through `i4_slot_offsets` finds
    /// every projection at exactly the right address and then decodes e2m1 nibbles as
    /// `n − 8` against a group-128 f32 scale read out of e8m0 bytes. The separation is the
    /// header magic (`ExpertHeader::from_bytes`, and `format.rs`'s
    /// `magic_separates_the_formats_when_the_length_cannot` is named for this) and the
    /// descriptor TYPE (`backend::ExpertDescF4` vs `ExpertDesc`). That is why
    /// `memory::routed::TierFmt` carries a `RoutedFmt` and not the `int4: bool` it replaced.
    ///
    /// They do diverge at dims where `i_dim % 128 != 0` — the toy `(64, 32)` fixtures in
    /// `tests/v4_loading.rs` are such a case — so a check that compares them is not vacuous
    /// everywhere, only where it matters.
    #[test]
    fn f4_slot_offsets_match_the_shipped_block_and_are_indistinguishable_from_i4() {
        // DeepSeek-V4-Flash: hidden 4096, moe_intermediate_size 2048.
        let (hidden, inter) = (4096usize, 2048usize);
        let off = f4_slot_offsets(hidden, inter);
        assert_eq!(
            off,
            [0, 4_194_304, 4_456_448, 8_650_752, 8_912_896, 13_107_200],
            "the .f4 slot layout moved"
        );
        // The last span (w2's scales) ends exactly at the block size, and the block size is
        // the shipped file's own stride.
        assert_eq!(off[5] + hidden * f4_groups(inter), 13_369_344);
        assert_eq!(f4_expert_bytes(hidden, inter), 13_369_344);
        assert_eq!(f4_expert_stride(hidden, inter), 13_369_344, "no padding at these dims");
        assert_eq!(4096 + 256 * f4_expert_stride(hidden, inter), 3_422_556_160);

        // Every packed span 4-byte aligned, so `dot_f4_wave_r`'s dword fast path is taken
        // rather than falling back to its scalar tail. A PERFORMANCE property, not a
        // correctness one — the kernel predicates on the alignment and handles both.
        for k in [0, 2, 4] {
            assert_eq!(off[k] % 4, 0, "packed span {} is not 4-byte aligned", off[k]);
        }

        // The coincidence, pinned so it is a known fact rather than a surprise at a call
        // site: at V4's dims `.f4` and `.i4` are the same layout AND the same size.
        assert_eq!(
            off,
            i4_slot_offsets(hidden, inter),
            "at i_dim % 128 == 0 the two nibble formats tile identically — if this ever \
             stops being true, the claim in this test's doc has to change with it"
        );
        assert_eq!(f4_expert_bytes(hidden, inter), i4_expert_bytes(hidden, inter));
        // `.vq3` is genuinely a different size (12-bit indices over VQ_DIM=4), so it is the
        // one format a length check does separate.
        assert_ne!(off, vq_slot_offsets(hidden, inter));
        assert_ne!(f4_expert_bytes(hidden, inter), vq_expert_bytes(hidden, inter));

        // Where they DO differ: `i_dim` not a multiple of 128. `tests/v4_loading.rs`'s toy
        // fixtures live here, which is the only reason a test can SEE an `.f4` set resolved
        // through `.i4`'s layout at all. There is no runtime check for it — that is the
        // finding above, not an omission.
        assert_ne!(f4_slot_offsets(64, 32), i4_slot_offsets(64, 32));
    }

    /// All three layouts come off ONE walk (`slot_offsets`), so this confronts that walk
    /// with the byte counts nothing in it reads: `*_proj_bytes`, which `expert_bytes` sums
    /// independently. Projection `p` must begin at the sum of the `p` before it.
    ///
    /// Deliberately NOT "each scale span abuts the next offset" — that is the walk's own
    /// `base += packed + scales` restated, and a test that respells the formula it checks
    /// can only fail when the copy drifts. These two comparisons can fail on the code.
    #[test]
    fn every_routed_format_places_each_projection_at_the_sum_of_the_ones_before_it() {
        let (hidden, inter) = (6144usize, 2048usize);
        let [(go, gi), (uo, ui), _] = vq_expert_layout(hidden, inter);
        for (name, off, proj) in [
            ("vq3", vq_slot_offsets(hidden, inter), vq_proj_bytes as fn(usize, usize) -> usize),
            ("i4", i4_slot_offsets(hidden, inter), i4_proj_bytes),
            ("f4", f4_slot_offsets(hidden, inter), f4_proj_bytes),
        ] {
            assert_eq!(off[0], 0, "{name}: the block starts at the first projection");
            assert_eq!(off[2], proj(go, gi), "{name}: up_packed");
            assert_eq!(off[4], proj(go, gi) + proj(uo, ui), "{name}: down_packed");
        }
    }

    #[test]
    fn i4_slot_offsets_are_contiguous_and_aligned() {
        let (hidden, inter) = (6144usize, 2048usize);
        let off = i4_slot_offsets(hidden, inter);
        assert_eq!(off[0], 0);
        for &o in &off {
            assert_eq!(o % 4, 0, "packed/scale span {o} not 4-byte aligned");
        }
        // last span (down_scale) ends exactly at i4_expert_bytes.
        assert_eq!(
            off[5] + hidden * i4_groups(inter) * 4,
            i4_expert_bytes(hidden, inter)
        );
    }

    #[test]
    fn i4_quant_matvec_roundtrip() {
        // quant_i4 → matvec_i4 approximates the true GEMV within int4 error.
        // i spans several groups so the group indexing is part of what is checked.
        let (o, i) = (16usize, I4_GROUP * 3);
        let mut w = vec![0.0f32; o * i];
        let mut s = 0x2468u64;
        let mut rf = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for v in w.iter_mut() {
            *v = rf();
        }
        let x: Vec<f32> = (0..i).map(|_| rf()).collect();
        let mut want = vec![0.0f32; o];
        for oo in 0..o {
            want[oo] = (0..i).map(|ii| w[oo * i + ii] * x[ii]).sum();
        }
        let (packed, scale) = quant_i4(&w, o, i);
        let mut got = vec![0.0f32; o];
        matvec_i4(&mut got, &x, &packed, &scale, o, i);
        // int4 group quant: err bounded by ~scale·Σ|x| worst case; check it tracks.
        let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let err = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err < 0.15 * mx + 0.1,
            "i4 roundtrip err={err:.3} max={mx:.3}"
        );
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

    // ── VQ-int3 oracle (M1) ─────────────────────────────────────────────────

    /// A tiny deterministic codebook padded to VQ_K entries (unused rows pushed far
    /// away so they never win a nearest-lookup). Entries 0..n hold `pts`.
    fn tiny_codebook(pts: &[[f32; VQ_DIM]]) -> Vec<f32> {
        let mut cb = vec![1e30f32; VQ_K * VQ_DIM]; // far-away filler
        for (k, p) in pts.iter().enumerate() {
            cb[k * VQ_DIM..(k + 1) * VQ_DIM].copy_from_slice(p);
        }
        cb
    }

    #[test]
    fn vq_expert_slicing_roundtrips_gemv() {
        // Build a 1-expert `.vq3` block from quant_vq of synthetic gate/up/down, slice
        // it back with vq_expert, and check each projection's gemv equals matvec_vq on
        // the original arrays — the converter→loader byte contract for a whole expert.
        let (hidden, moe_inter) = (VQ_GROUP, VQ_GROUP); // tiny but valid (one group each)
        let cb = tiny_codebook(&[[1.0, -1.0, 0.5, -0.5], [0.25, 0.5, 0.75, 1.0]]);
        let stride = vq_expert_stride(hidden, moe_inter);
        let mut layer = vec![0u8; stride]; // one expert
        let dims = vq_expert_layout(hidden, moe_inter);
        // Deterministic weights per projection; keep originals to GEMV against.
        let mut originals = Vec::new();
        let mut off = 0;
        for (p, &(o_dim, i_dim)) in dims.iter().enumerate() {
            let mut w = vec![0.0f32; o_dim * i_dim];
            for (n, wv) in w.iter_mut().enumerate() {
                let e = (n + p) % 2;
                *wv = cb[e * VQ_DIM + (n % VQ_DIM)] * 0.5;
            }
            let (indices, scales) = quant_vq(&w, o_dim, i_dim, &cb);
            let ib = o_dim * vq_row_bytes(i_dim);
            layer[off..off + ib].copy_from_slice(&indices);
            for (s, dst) in scales.iter().zip(layer[off + ib..].chunks_exact_mut(2)) {
                dst.copy_from_slice(&s.to_le_bytes());
            }
            off += vq_proj_bytes(o_dim, i_dim);
            originals.push((indices, scales, o_dim, i_dim));
        }
        let projs = vq_expert(&layer, 0, hidden, moe_inter);
        for (proj, (indices, scales, o_dim, i_dim)) in projs.iter().zip(&originals) {
            let x: Vec<f32> = (0..*i_dim).map(|k| (k + 1) as f32).collect();
            let mut y_load = vec![0.0f32; *o_dim];
            let mut y_ref = vec![0.0f32; *o_dim];
            // GEMV the SLICED projection (decoding its borrowed bf16 scale bytes) and
            // the ORIGINAL arrays — the converter→loader byte contract for an expert.
            let ps = proj.scales_u16();
            matvec_vq(
                &mut y_load,
                &x,
                proj.indices,
                &ps,
                &cb,
                proj.o_dim,
                proj.i_dim,
            );
            matvec_vq(&mut y_ref, &x, indices, scales, &cb, *o_dim, *i_dim);
            assert_eq!(y_load, y_ref);
        }
    }

    #[test]
    fn vq_index_pack_unpack_all_positions() {
        // 12-bit indices, 2 per 3 bytes; every subvector slot must round-trip across
        // the nibble straddle (even k at shift 0, odd k at shift 4).
        let n = 6; // 6 subvectors → i_dim = 24
        let mut row = vec![0u8; vq_row_bytes(n * VQ_DIM)];
        let vals = [0u16, 4095, 1, 2048, 4094, 7];
        for (k, &v) in vals.iter().enumerate() {
            set_idx(&mut row, k, v);
        }
        for (k, &v) in vals.iter().enumerate() {
            assert_eq!(get_idx(&row, k), v as usize, "slot {k}");
        }
    }

    #[test]
    fn vq_gemv_matches_codebook_dot() {
        // One full group (i_dim = VQ_GROUP), one row. Each subvector is one of two
        // codebook entries; amax over the group is 1.0 ⇒ scale 1, exact quant.
        let cb = tiny_codebook(&[[1.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, -1.0]]);
        let i_dim = VQ_GROUP;
        let nsub = i_dim / VQ_DIM;
        let entry = |t: usize| t % 2; // alternate entries 0,1,0,1,…
        let mut w = vec![0.0f32; i_dim];
        for t in 0..nsub {
            let e = entry(t);
            w[t * VQ_DIM..(t + 1) * VQ_DIM].copy_from_slice(&cb[e * VQ_DIM..(e + 1) * VQ_DIM]);
        }
        let (idx, scales) = quant_vq(&w, 1, i_dim, &cb);
        let x: Vec<f32> = (0..i_dim).map(|i| (i + 1) as f32).collect();
        let mut y = [0.0f32];
        matvec_vq(&mut y, &x, &idx, &scales, &cb, 1, i_dim);
        // Expected = Σ_t codebook[entry(t)] · x_subvec(t), with scale 1.
        let mut exp = 0.0f32;
        for t in 0..nsub {
            let e = entry(t);
            for d in 0..VQ_DIM {
                exp += cb[e * VQ_DIM + d] * x[t * VQ_DIM + d];
            }
        }
        assert!((y[0] - exp).abs() < 1e-4, "y0={} exp={}", y[0], exp);
    }

    #[test]
    fn vq_roundtrip_reconstructs_within_codebook() {
        // Unit-peak codebook entries and a bf16-exact group scale ⇒ normalized
        // subvectors land exactly on entries, so quant→dequant is exact — the
        // faithfulness the converter relies on. (Both entries peak at 1 so the group
        // amax = the scale S; S=0.5 is bf16-exact.)
        let cb = tiny_codebook(&[[1.0, -1.0, 0.5, -0.5], [0.25, 0.5, 0.75, 1.0]]);
        let i_dim = VQ_GROUP;
        let nsub = i_dim / VQ_DIM;
        let entry = |t: usize| t % 2;
        let scale = 0.5f32;
        let mut w = vec![0.0f32; i_dim];
        for t in 0..nsub {
            let e = entry(t);
            for d in 0..VQ_DIM {
                w[t * VQ_DIM + d] = cb[e * VQ_DIM + d] * scale;
            }
        }
        let (idx, scales) = quant_vq(&w, 1, i_dim, &cb);
        let s = bf16_to_f32(scales[0]);
        assert!((s - scale).abs() < 1e-6, "group scale {s} != {scale}");
        for t in 0..nsub {
            let e = &cb[get_idx(&idx, t) * VQ_DIM..][..VQ_DIM];
            for d in 0..VQ_DIM {
                assert!((e[d] * s - w[t * VQ_DIM + d]).abs() < 1e-4, "t{t} d{d}");
            }
        }
    }

    #[test]
    fn fp8_block_dequant_applies_tile_scale() {
        // 2×2 fp8 matrix, block=1 (per-element scale for the test) — check the tile
        // indexing and that e4m3 decode is wired. 0x38=1.0, 0x40=2.0, 0xB8=-1.0.
        let packed = [0x38u8, 0x40, 0xB8, 0x00]; // [[1, 2],[-1, 0]]
        let scale = [10.0f32, 100.0, 1000.0, 1.0]; // per element (block=1)
        let out = dequant_fp8_block(&packed, &scale, 2, 2, 1);
        assert_eq!(out, [10.0, 200.0, -1000.0, 0.0]);
    }

    #[test]
    fn matvec_fp8_matches_dequant_dot() {
        // Same 2×2 fp8 as above; GEMV must equal the dequant-then-dot reference.
        let packed = [0x38u8, 0x40, 0xB8, 0x00]; // [[1,2],[-1,0]] × per-elem scale
        let scale = [10.0f32, 100.0, 1000.0, 1.0];
        let x = [3.0f32, 5.0];
        let mut y = [0.0f32; 2];
        matvec_fp8(&mut y, &x, &packed, &scale, 2, 1);
        // row0: 10·3 + 200·5 = 1030 ; row1: -1000·3 + 0·5 = -3000
        assert_eq!(y, [1030.0, -3000.0]);
    }

    /// The e2m1 table decoded from its bit fields rather than restated, so this is a
    /// check and not a copy of the constant. Bit 3 sign, bits 2-1 exponent, bit 0
    /// mantissa; exponent 0 is the subnormal case (value = mantissa/2), every other
    /// exponent e is `2^(e-1) * (1 + mantissa/2)`. There is no Inf and no NaN, which is
    /// what makes a bare 16-entry lookup a COMPLETE decoder.
    #[test]
    fn f4_lut_matches_the_e2m1_bit_fields() {
        for code in 0u8..16 {
            let (sign, exp, man) = (code >> 3, (code >> 1) & 3, code & 1);
            let mag = if exp == 0 {
                man as f32 * 0.5
            } else {
                f32::exp2(exp as f32 - 1.0) * (1.0 + man as f32 * 0.5)
            };
            let want = if sign == 1 { -mag } else { mag };
            assert_eq!(
                F4_LUT[code as usize], want,
                "e2m1 code {code:#06b} decodes to {want}, table says {}",
                F4_LUT[code as usize]
            );
        }
        // The largest magnitude is 6.0, so a block's dynamic range is entirely in its
        // e8m0 scale — a fact the repack relies on to carry values through untouched.
        assert_eq!(F4_LUT.iter().cloned().fold(0.0f32, f32::max), 6.0);
    }

    /// e8m0 is a bare exponent, `2^(b-127)`, and both ends matter: b=0 lands on an f32
    /// subnormal (the `bits << 23` form silently returns +0 there) and 0xFF is the
    /// format's NaN, which would poison a whole 32-weight block.
    #[test]
    fn e8m0_covers_both_ends_and_rejects_nan() {
        assert_eq!(e8m0(127).unwrap(), 1.0);
        assert_eq!(e8m0(128).unwrap(), 2.0);
        assert_eq!(e8m0(126).unwrap(), 0.5);
        assert_eq!(e8m0(254).unwrap(), f32::exp2(127.0));
        let lo = e8m0(0).unwrap();
        assert!(lo > 0.0 && lo.is_finite(), "b=0 must be 2^-127, got {lo}");
        assert_eq!(lo, f32::exp2(-127.0));
        assert!(e8m0(0xFF).is_err(), "0xFF is NaN and must be refused");
    }

    /// **`V4_PROJ`'s ORDER, derived from the reference rather than restated.**
    ///
    /// This exists because a mutation test found the hole: swapping `w1` and `w3` in the
    /// constant is invisible to everything else in S1a. The `.f4` repack maps source name →
    /// block slot through this one constant, so the writer and the byte-exactness verifier
    /// both move — the artifact is self-consistently wrong, byte-clean, and only a
    /// numerical oracle against the reference could see it. The two tensors even have
    /// identical shapes, so no dimension check helps.
    ///
    /// So the order is read back out of `inference/model.py`'s `Expert.forward`
    /// (`gate = self.w1(x)`, `up = self.w3(x)`, `return self.w2(…)`) and compared. That
    /// turns the doc comment's citation from decoration into a check. Skipped when the
    /// checkpoint is absent — and S1b's oracle remains the real gate, since this pins only
    /// what the reference SAYS, not what rivoli then computes.
    #[test]
    fn v4_proj_order_matches_the_reference_expert_forward() {
        const REF: &str = "/var/db/rivoli/deepseek-v4-flash-0731/inference/model.py";
        let Ok(src) = std::fs::read_to_string(REF) else {
            eprintln!("SKIP v4_proj_order: no reference at {REF} — V4_PROJ is UNPINNED");
            return;
        };
        // `Expert.forward` only — `MoE.forward` and the mtp blocks mention `w1`/`w2` too.
        let body = src
            .split_once("class Expert(nn.Module)")
            .map(|(_, rest)| rest)
            .and_then(|rest| rest.split_once("class MoE(nn.Module)"))
            .map(|(body, _)| body)
            .expect("Expert class not found — the reference has been restructured");
        let pick = |lhs: &str| -> String {
            let at = body
                .find(lhs)
                .unwrap_or_else(|| panic!("{lhs:?} not in Expert.forward"));
            let rest = &body[at + lhs.len()..];
            let w = rest
                .split_once('(')
                .map(|(w, _)| w.trim())
                .expect("no call after the projection");
            assert!(w.starts_with('w') && w.len() == 2, "unexpected projection {w:?}");
            w.to_string()
        };
        let got = [
            pick("gate = self."),
            pick("up = self."),
            pick("return self."),
        ];
        assert_eq!(
            got,
            V4_PROJ.map(String::from),
            "V4_PROJ is [gate, up, down]; the reference says {got:?}"
        );
    }

    /// `.f4` shares `.i4`'s nibble packing but carries a one-byte scale per 32 weights
    /// instead of an f32 per 128, so its rows are wider. GLM's widths stand in for the
    /// arithmetic; V4-Flash's own are hidden 4096 / moe_inter 2048.
    #[test]
    fn f4_layout_sizes() {
        assert_eq!(f4_row_bytes(2048), 1024);
        assert_eq!(f4_groups(2048), 64);
        assert_eq!(f4_groups(2049), 65, "a ragged tail gets its own scale");
        // 4.25 bits/weight: 4 for the nibble, 8/32 for the scale.
        let (o, i) = (4096usize, 2048usize);
        assert_eq!(f4_proj_bytes(o, i), o * 1024 + o * 64);
        assert_eq!(f4_proj_bytes(o, i) as f64 * 8.0 / (o * i) as f64, 4.25);
        // Strictly larger than int3-vq's 3.25, and that is the trade we chose.
        assert!(f4_proj_bytes(o, i) > vq_proj_bytes(o, i));
    }
}
