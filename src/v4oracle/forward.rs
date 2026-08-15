//! A CPU transliteration of `inference/model.py`'s `Transformer.forward` main path, and
//! the deliberate-breakage set the gate is proved against.
//!
//! # What this is for
//!
//! S2 and S3 build the V4 attention frontend and MoE on the GPU. Every defect available to
//! them is *silent-wrong*: a missing QK-norm, RoPE on the wrong dims, an unclamped SwiGLU,
//! a mis-grouped output projection — none crash, all produce fluent wrong text, and
//! `distinct`/`longest repeated block` cannot see any of them (CLAUDE.md; they have misled
//! three investigations in this repo). So the gate is built first and proved before the
//! thing it gates exists.
//!
//! # Fidelity, and what the tolerance therefore has to be
//!
//! Reproduced exactly, because each changes a VALUE:
//! - every `act_quant` the reference performs, at its own block size and with ue8m0 scale
//!   rounding — including the fp8 activation quantization inside every quantized `Linear`,
//!   which is why the goldens are not "f32 reference values" but "what fp8 arithmetic
//!   produces";
//! - bf16 rounding at every point the reference stores into a bf16 tensor: each GEMM
//!   output, each `RMSNorm` return, `apply_rotary_emb`'s in-place `copy_` back into a bf16
//!   view, and the residual stream itself;
//! - the fp4 expert weights and their group-32 e8m0 scales, dequantized as
//!   `fp4_gemm` applies them.
//!
//! NOT reproduced, because each changes only a summation ORDER: `fp8_gemm`'s two-level fp32
//! accumulator, `sparse_attn`'s block-64 online softmax, and `hc_split_sinkhorn`'s warp
//! reductions. Pinning those would pin the oracle to one kernel's tiling, and the whole
//! point is that S2 may tile differently. The residual disagreement from re-association is
//! the floor on any tolerance built on these goldens.
//!
//! # Out of scope, deliberately
//!
//! - **DSpark/MTP** (`forward_spec`, `mtp.*`, `markov_head`, `confidence_head`) — see
//!   `docs/investigations/v4-flash-port.md` §"Scope cut".
//! - **Batch > 1.** Everything here runs `bsz = 1`. The reference's batch dimension is
//!   elementwise-independent everywhere on this path *except* the compressor's shared
//!   `kv_state`/`score_state` buffers, which are sliced `[:bsz]` and so are also
//!   independent. A batching bug is therefore NOT covered by these goldens, and S2 must not
//!   read this as evidence about batched decode.
//! - **Tensor parallelism** (`world_size == 1`), so no `all_reduce` ordering is modelled.

use crate::v4oracle::numerics::{
    act_quant_inplace, bf16_decode, bf16_encode, fp4_act_quant_inplace, hadamard_rotate, sigmoid,
    silu, softplus,
};
use crate::v4oracle::weights::{V4Config, WMat};

// ---------------------------------------------------------------------------------------
// deliberate breakages
// ---------------------------------------------------------------------------------------

/// Declares [`Defect`] and [`Defect::ALL`] from ONE list.
///
/// The array used to be hand-maintained beside the enum, and its own doc said so: *"Adding a
/// variant to the enum does NOT add it here — this array is hand-maintained and a variant
/// missing from it is silently untested."* That is a real escape, not a hypothetical one: a
/// variant absent from the list runs in NO matrix while the suite stays green and the count
/// of defects goes UP. `tests/v4_oracle.rs::expect` forces the author to *classify* a new
/// variant (its match is exhaustive and wildcard-free), but classifying is not listing.
///
/// A source-scanning test was written to catch that on 2026-08-05 and is now deleted with the
/// array: generating both from one declaration removes the escape rather than detecting it.
/// A variant not in this invocation does not exist; one that is in it reaches `ALL` by
/// construction; a duplicate is `E0428` at compile time rather than a runtime assertion. The
/// parser also could not see a non-bare variant — `Foo(usize)`, `Bar = 1` — which is exactly
/// what harness consolidation would introduce.
///
/// Declaration order is `ALL` order, and `None` must stay first: `breakages()` filters it out
/// by value, but `tests/f4_attn.rs`'s grid indexes off the order.
macro_rules! defects {
    ($( $(#[$m:meta])* $v:ident ),+ $(,)?) => {
        /// A single, deliberately-wrong variant of the transliteration.
        ///
        /// These are not hypotheticals. Every one is a transcription slip a competent implementer
        /// makes when porting `model.py` to a kernel, and each is silent: it changes the text and
        /// nothing else. `tests/v4_oracle.rs` runs the whole set against the case grid and asserts,
        /// for each, both halves of the claim — the goldens it MUST perturb, and the goldens it
        /// MUST leave bit-identical. A defect that moves everything is evidence of nothing.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Defect { $( $(#[$m])* $v ),+ }

        impl Defect {
            /// Every defect, `None` first. Generated from the same declaration as the enum,
            /// so the two cannot drift.
            pub const ALL: &'static [Defect] = &[ $( Defect::$v ),+ ];
        }
    };
}

defects! {
    /// The real transliteration.
    None,

    // -- q path -------------------------------------------------------------------------
    /// Drop the weightless per-head RMS after `wq_b` (model.py:504).
    SkipQkNorm,
    /// Apply `q_norm`'s learned weight to the per-head RMS. `q_norm` is `RMSNorm(1024)` and
    /// the QK-norm is over `head_dim = 512` with NO weight; reusing the weight is the
    /// natural mistake because both are called "the q norm".
    QkNormUsesQNormWeight,
    /// QK-norm after RoPE instead of before (model.py:504 then :505).
    QkNormAfterRope,

    // -- rope ---------------------------------------------------------------------------
    /// Rotate the whole `head_dim`, not `x[..., -rope_head_dim:]`.
    RopeAllDims,
    /// Rotate `x[..., :rope_head_dim]` — the same count of dims, the wrong end.
    RopeFirstDims,
    /// Pair `(i, i + rd/2)` (GPT-NeoX / GLM style) instead of `view_as_complex`'s adjacent
    /// `(2i, 2i+1)`. rivoli's existing RoPE is half-split, so this is the *default* mistake.
    RopeHalfSplit,
    /// Keep `compress_rope_theta` on compressed layers but drop the YaRN interpolation
    /// (`original_seq_len = 0`). The insidious half of the per-layer selection: the theta is
    /// right, so the frequencies look plausible at every scale.
    RopeNoYarn,
    /// Use the YaRN table on every layer, including `compress_ratio == 0`.
    RopeYarnEverywhere,
    /// Keep YaRN on compressed layers but use the base `rope_theta` there instead of
    /// `compress_rope_theta`. The other half of the same mistake -- and DISTINCT from
    /// `RopeNoYarn`: if both selected the same table they would be one defect wearing two
    /// names, and the matrix would count one piece of evidence twice.
    RopeBaseThetaEverywhere,

    // -- kv path ------------------------------------------------------------------------
    /// Omit the partial fp8 `act_quant` of the KV entry entirely.
    SkipKvActQuant,
    /// Quantize all `head_dim` dims instead of `[0, head_dim - rope_head_dim)`. This is the
    /// llama.cpp failure mode named in v4-flash-port.md §0.2: it corrupts the positional
    /// dims and produces noise without a crash.
    KvActQuantWholeTensor,
    /// Right operation, block 128 instead of 64.
    KvActQuantBlock128,
    /// Right operation and block, but no ue8m0 power-of-two scale rounding.
    KvActQuantNoRoundScale,

    // -- attention core -----------------------------------------------------------------
    /// Drop `attn_sink` from the softmax denominator.
    SkipAttnSink,
    /// Add `exp(attn_sink[h])` to the denominator WITHOUT the running-max shift the rest of
    /// the online softmax uses (`kernel.py:346` subtracts `scores_max[i]`). The sink then
    /// carries the wrong weight by a factor of `exp(max)`.
    ///
    /// Deliberately NOT "sink in the numerator too": a sink treated as a real key with a
    /// zero value vector is algebraically the reference, so that would be a variant that
    /// cannot fail. Getting the max-shift wrong is the mistake this kernel actually invites.
    AttnSinkNotMaxShifted,
    /// At prefill, seed the ring with the FIRST `window_size` positions instead of the last
    /// (model.py:526-528 keeps `kv[:, -win:]` and rotates it so slot `t % win` holds
    /// position `t`). "Just copy the prefill KV into the cache" is the obvious
    /// implementation and it is right exactly when the prompt fits the window — which is
    /// why a short-prompt fixture cannot see it.
    PrefillRingWritesFirstWindow,

    // -- output -------------------------------------------------------------------------
    /// Omit `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)` (model.py:539).
    SkipOutputDerotation,
    /// De-rotate with the forward rotation instead of the conjugate.
    OutputDerotationForward,
    /// Split `o` into `o_groups` along the HEAD_DIM axis instead of the head axis:
    /// `o.view(b, s, n_groups, -1)` is over a head-major flattening, so group `g` is the
    /// contiguous run of `n_heads / o_groups` heads, not a slice of every head's `head_dim`.
    /// Both are permutations of the same 4096 numbers into the same number of dot products,
    /// so magnitudes, norms and every summary statistic survive it — the "right summation,
    /// wrong grouping" hazard named in the S1b brief.
    WoGroupsSplitHeadDim,
    /// Keep the grouping but assign heads to groups round-robin instead of in contiguous
    /// runs of `n_heads / o_groups`.
    WoGroupsInterleaved,

    // -- compressor / indexer -----------------------------------------------------------
    /// Ignore `Compressor.overlap` (`compress_ratio == 4`) and pool non-overlapping blocks.
    CompressorNoOverlap,
    /// Drop the learned intra-block position embedding `ape` from the pooling scores.
    CompressorNoApe,
    /// RoPE the compressed entry at the block's LAST position instead of its first
    /// (`freqs_cis[:cutoff:ratio]`, model.py:370).
    CompressorRopeAtBlockEnd,
    /// Drop the `relu_()` in the indexer score (model.py:427).
    IndexerNoRelu,
    /// Skip the indexer's fp4 simulation of `q` and of its compressed kv.
    IndexerNoFp4Quant,
    /// Skip `rotate_activation` (the Hadamard spread) in the indexer.
    IndexerNoHadamard,
    /// Drop the `weights_proj` per-head factor, i.e. sum the head scores unweighted.
    IndexerNoWeights,
    /// Round the head-score accumulator to bf16 after EVERY term instead of accumulating in
    /// f32 and rounding once (model.py:427's `.sum(dim=2)`).
    ///
    /// **This was the oracle's own behaviour until 2026-08-05** — see `Oracle::bf16_sum` for
    /// the measurement that settled it. It is kept as a defect precisely because it was
    /// believed correct: every test in this file is self-relative, so an error the oracle
    /// shared with its own defect matrix cancelled everywhere and nothing could see it. The
    /// absolute check is `bf16_reduction_matches_torch_and_not_a_running_fold`;
    /// this variant is its other half.
    IndexerBf16RunningSum,

    // -- MoE ----------------------------------------------------------------------------
    /// `swiglu_limit = 0` — the unclamped SwiGLU rivoli ships.
    SwigluUnclamped,
    /// Clamp the gate on both sides. The reference clamps `up` symmetrically but `gate`
    /// only from above (model.py:606-607).
    SwigluClampGateBothSides,
    /// Score with `softmax` instead of `sqrt(softplus(·))`.
    RouterSoftmax,
    /// Compute `softplus` as `ln(1+e^x)` with no `threshold=20` identity branch.
    RouterNoSoftplusThreshold,
    /// Use the bias-shifted scores as the routing WEIGHTS. The reference uses the bias for
    /// selection only (model.py:577-585).
    RouterBiasedWeights,
    /// Skip the `weights /= weights.sum()` renormalization.
    RouterNoRenorm,
    /// Skip the `route_scale` multiply.
    RouterNoScale,
    /// Route the hash layers by top-k score instead of by `tid2eid[input_id]`.
    HashRoutingIgnored,
    /// Apply the routing weight to the expert's OUTPUT rather than to its SwiGLU
    /// intermediate. Mathematically identical in exact arithmetic; the reference rounds the
    /// weighted intermediate to bf16 before `w2`, so it is not identical here.
    RouteWeightAfterW2,
    /// Scale the shared expert by a routing weight. It has none (model.py:648).
    SharedExpertWeighted,
    /// Read the packed fp4 nibbles high-first. Settled by `convert.py`, but kept so the
    /// decision stays A/B-able rather than merely asserted.
    Fp4NibbleSwap,

    // -- mHC ----------------------------------------------------------------------------
    /// Run `hc_sinkhorn_iters - 1` iterations — the off-by-one from counting the leading
    /// column pass as an iteration when the config already does.
    ///
    /// **Its detectability is weight-dependent, which is why it carries no tolerance and
    /// sits in `targeted_defects()` rather than the matrix.** On the toy fixture the 4x4
    /// matrix reaches a BITWISE fixed point well before iteration 20, so 19 and 20 agree
    /// exactly and `sinkhorn_has_converged_long_before_iteration_20` asserts that. **On the
    /// checkpoint they do not agree**: measured 2026-08-07 with `v4-oracle defects --layer 0
    /// --decode-steps 1`, this moves 39,893/53,248 of `L0.pre.ffn_norm_out`, all 78 router
    /// weights, and 143,026/212,992 of `L0.pre.out`. Convergence is to within f32 rounding;
    /// whether the last ulp settles depends on the mixes, and `hc_post` plus the MoE spread
    /// the difference from there. Counts, not magnitudes — the sweep does not measure size.
    ///
    /// **The name is provisional (2026-08-07)** and probably wrong: it replaced
    /// `SinkhornOneFewerIter` on the toy reading alone, and the checkpoint number above
    /// landed afterwards. This is a defect that one fixture cannot see, not a probe.
    SinkhornIterCountProbe,
    /// Index the combination matrix `[dest, source]` instead of `[source, dest]`.
    SinkhornCombTransposed,
    /// Drop the `comb @ residual` term from `hc_post` — a plain residual add.
    HcPostNoComb,
    /// Drop the `* rsqrt(mean(x^2) + eps)` factor on the mHC mixes.
    HcPreNoRsqrt,

    // -- head tail ----------------------------------------------------------------------
    /// Drop the `* rsqrt(mean(x^2) + eps)` factor on `hc_head`'s mixes (model.py:712-713).
    /// The head's own copy of the mistake `HcPreNoRsqrt` models inside a block: two separate
    /// sites, and a port can get one right and the other wrong.
    HeadHcNoRsqrt,
    /// Take `hc_head`'s RMS per hyper-connection COPY (`mean` over `dim`) and apply copy
    /// `j`'s to mix `j`, instead of one statistic over the full `hc_mult * dim` flattened
    /// row. `hc_head_fn` yields exactly as many mixes as there are copies, so the
    /// wrong version type-checks and produces the same shapes — it is only wrong by value.
    /// It would NOT line up in `hc_pre`, which has 24 mixes against 4 copies.
    HeadHcRsqrtPerCopy,
    /// Skip the final `RMSNorm` entirely (`self.head(self.norm(h))`, model.py:923). The
    /// natural slip for anyone who reads `hc_head` as already normalising.
    HeadNormSkipped,
    /// Run the final `RMSNorm` but return f32, skipping the bf16 store `RMSNorm.forward`
    /// performs on the way out. Measured at 7.5e-3 on a 3.1 max (0.24%) elsewhere in this
    /// port, and it is a live choice on the device: `kernels/mla.hip::rmsnorm_batch` bf16-rounds
    /// and `kernels/linalg.hip::rmsnorm_rows` does not.
    HeadNormNotBf16,
    /// Take the final `RMSNorm`'s statistic JOINTLY over all `s * dim` values instead of per
    /// token: what handing a single-row norm kernel an `s x dim` buffer does. Invisible at
    /// decode (`s == 1`), which is the only shape most smoke tests run.
    ///
    /// Only the STATISTIC is modelled. `kernels/linalg.hip::rmsnorm_rows` would also read its
    /// `dim`-long gain past the end of the tensor, which is undefined rather than wrapped;
    /// this tiles the gain instead, which is the charitable realization. The joint statistic
    /// is the load-bearing half, and an out-of-bounds read is not a thing an oracle can
    /// stand in for anyway.
    HeadNormOverAllTokens,
    /// `ParallelHead.forward` slices `x[:, -1]`; take `x[:, 0]` instead. Also invisible at
    /// decode, and at prefill it produces perfectly well-formed logits for the wrong token.
    HeadLogitsFromFirstRow,

    // -- candidate designs, modelled before they are built -------------------------------
    /// The split-k fp8 GEMV's fold order (`gemv_fp8_bf16_splitk` — **NOT BUILT**; it lives on
/// branch `wt/v4-splitk`, was measured and REJECTED in §M9, and must not be confused with
/// `kernels/linalg.hip::gemv_fp8_splitk`, which is GLM's and ships), applied to
    /// exactly the GEMVs the device dispatch predicate selects — see [`Oracle::splitk_selects`]
    /// for the predicate and [`splitk_fold`] for the partial ordering, both derived from the
    /// ONE spec in `docs/investigations/v4-decode-decomposition.md` §M9.
    ///
    /// **Not a transcription slip: a candidate design's arithmetic, priced on the real
    /// checkpoint before the kernel was built** (the `SinkhornIterCountProbe` precedent —
    /// a variant whose detectability is weight-dependent and which the toy CANNOT see:
    /// the predicate needs `k >= 4096` and the toy's largest K is `dim = 256`, so the toy
    /// matrix is structurally blind to it and it sits in `targeted_defects()` instead;
    /// `the_splitk_fold_is_toy_blind_partition_exact_and_nonzero_at_real_dims` holds the
    /// three structural claims). §M9's stretch is authorized to abandon byte-identity for
    /// this ONE kernel and measure the quality drop instead; an emit under this defect vs
    /// an emit under `None`, `v4-oracle cmp`'d, IS that measurement's host half.
    SplitKFoldOrder,

    // -- precision ----------------------------------------------------------------------
    /// Skip the bf16 stores that go through `round_bf16`. Not a bug anyone would ship — it
    /// is here to MEASURE how much of the golden's value is bf16 fidelity, which sets the
    /// floor for any tolerance derived from these goldens.
    ///
    /// **It does NOT reach every bf16 store, so the floor it reports is optimistic.** Four
    /// sites on the indexer path round through `bf16_decode(bf16_encode(..))` directly and
    /// ignore this flag: the `weights_proj * scale` store, the einsum store, the per-term
    /// `dot * wt` store, and `bf16_sum`'s final round. Three predate this variant; the fourth
    /// arrived with the 2026-08-05 reduction fix. They are bf16-by-reference on purpose --
    /// that chain's precision selects WHICH positions are attended -- so routing them through
    /// here would make the flag change the attended SET, not just the precision.
    /// `qk_norm`'s `b()` closure is the same pattern for the same reason.
    NoBf16Rounding,
}

impl Defect {
    /// Every variant except [`Defect::None`] -- the breakages proper.
    pub fn breakages() -> impl Iterator<Item = Defect> {
        Self::ALL.iter().copied().filter(|d| *d != Defect::None)
    }

    /// The variant whose `Debug` name is exactly `name` -- the parser behind
    /// `v4-oracle emit --defect`.
    ///
    /// The `Err` carries EVERY variant name, because the caller's one job is to refuse
    /// loudly: a typo that silently fell back to `None` would emit two identical goldens
    /// and an A/B that cannot fail -- this repo's most-repeated failure shape, and the
    /// reason this is `Result` rather than `Option`. Exact match only; forgiving case
    /// would put every name one typo away from a different one.
    pub fn from_flag(name: &str) -> Result<Defect, String> {
        Self::ALL
            .iter()
            .copied()
            .find(|d| format!("{d:?}") == name)
            .ok_or_else(|| {
                let all: Vec<String> = Self::ALL.iter().map(|d| format!("{d:?}")).collect();
                format!(
                    "unknown defect {name:?}. The variants are: {}",
                    all.join(", ")
                )
            })
    }
}

/// The split-k GEMV's fold, on the oracle's own per-element arithmetic — one half of the
/// partial-ordering spec in `docs/investigations/v4-decode-decomposition.md` §M9; the other
/// half is `gemv_fp8_bf16_splitk` (unmerged — see [`Oracle::splitk_fold`]'s note), and
/// `tests/f4_kernel.rs::the_splitk_kernel_folds_in_the_registered_partial_order` pins the
/// kernel to a transliteration of the SAME spec bit-for-bit, which is what closes the
/// "oracle models a different reassociation than the kernel executes" failure mode.
///
/// The spec, restated (the doc is normative):
/// 1. **Partition**: 256 partials, fixed at dispatch. Partial `t` owns dword-quads
///    `q ≡ t (mod 256)` — columns `4q..4q+4` — folded in ascending column order within
///    the partial. (`k % 4 == 0` is the predicate's own guard; every captured shape has
///    `k = 4096`, i.e. 4 quads per partial.)
/// 2. **Wave combine**: partials `32w..32w+32` reduce through `wave_sum`'s shfl-down
///    ladder (offsets 16, 8, 4, 2, 1). Out-of-range lanes are not modelled because lane
///    0's dependency cone never contains one — the same argument
///    `the_fp8_dot_sums_in_source_order_through_both_loops` records for the serial
///    kernel's ladder.
/// 3. **Final fold**: the 8 wave sums added in ascending wave order by one thread.
///
/// What is deliberately NOT modelled, because it is IDENTICAL in both arms of the A/B this
/// fold exists for and therefore cancels: the kernel's per-quad grouping
/// `s * (x0*l0 + x1*l1 + x2*l2 + x3*l3)` and its FMA contraction. The oracle's serial fold
/// already differs from the serial kernel by exactly those (the module doc's "NOT
/// reproduced" list — the goldens' tolerance floor); `SplitKFoldOrder` vs `None` isolates
/// the partition + combine-tree change, which is the only thing the split-k kernel changes
/// relative to the serial kernel.
///
/// No `mul_add`, matching `linear`'s serial fold: both arms use plain `x*w` products in
/// plain `+` chains, so the A/B's delta is reassociation and nothing else.
pub fn splitk_fold(x: &[f32], w: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    debug_assert!(x.len().is_multiple_of(4), "the predicate guards k % 4 == 0");
    let quads = x.len() / 4;
    splitk_combine(|start, stride| {
        let mut p = 0.0f32;
        for q in (start..quads).step_by(stride) {
            for i in q * 4..q * 4 + 4 {
                p += x[i] * w[i];
            }
        }
        p
    })
}

/// §M9's partition and combine tree, generic over the per-thread chain — ONE definition,
/// shared by [`splitk_fold`] (the oracle's per-element chain) and the device test's
/// transliteration of the kernel's fma chain (`tests/f4_kernel.rs::FoldRow::splitk`).
/// Review 2026-08-08 found the earlier arrangement held two private copies of the
/// partition and ladder whose agreement was inspection, not code — exactly failure mode
/// #8's residual surface. Hoisted so the fold's SHAPE exists once: `chain(start, stride)`
/// is called at `start = 32w + l`, `stride = 256`, wave sums fold ascending; the GPU test
/// then pins the kernel to this same shape bit-for-bit, closing the loop executably.
pub fn splitk_combine(chain: impl Fn(usize, usize) -> f32) -> f32 {
    const THREADS: usize = 256; // ROWS_PER_BLOCK * WAVE — the split count, fixed at dispatch
    const WAVE: usize = 32;
    let mut acc = 0.0f32;
    for wv in 0..THREADS / WAVE {
        let mut lanes = [0.0f32; WAVE];
        for (l, p) in lanes.iter_mut().enumerate() {
            *p = chain(wv * WAVE + l, THREADS);
        }
        acc += wave_ladder(lanes);
    }
    acc
}

/// `wave_sum`'s shfl-down ladder over one wave's 32 partials, lane 0's result.
/// Out-of-range lanes are not modelled: on the device they self-double, but lane 0's
/// dependency cone reads a lane only at a step where all of that lane's prior updates
/// were in-range, so the cone never contains a doubled value —
/// `tests/f4_kernel.rs::the_fp8_dot_sums_in_source_order_through_both_loops` records the
/// same argument for the serial kernel's ladder, and both fold tests replay this exact
/// function.
pub fn wave_ladder(mut lanes: [f32; 32]) -> f32 {
    for o in [16usize, 8, 4, 2, 1] {
        let prev = lanes;
        for l in 0..(32 - o) {
            lanes[l] = prev[l] + prev[l + o];
        }
    }
    lanes[0]
}

// ---------------------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------------------

/// One `Compressor`'s parameters. `wkv`/`wgate` are `Linear(..., dtype=torch.float32)` in
/// the reference, so they take the un-quantized `F.linear` path — the checkpoint stores
/// them in bf16 and the module holds them in f32.
#[derive(Clone)]
pub struct CompressorW {
    pub ratio: usize,
    pub overlap: bool,
    /// `head_dim` of the *compressor*: `args.head_dim` for the attention one,
    /// `args.index_head_dim` for the indexer's.
    pub d: usize,
    /// `rotate=True` only for the indexer's compressor: Hadamard + fp4 instead of
    /// partial fp8.
    pub rotate: bool,
    /// `[ratio, coff * d]`.
    pub ape: Vec<f32>,
    pub wkv: WMat,
    pub wgate: WMat,
    pub norm: Vec<f32>,
}

impl CompressorW {
    pub fn coff(&self) -> usize {
        1 + usize::from(self.overlap)
    }
}

#[derive(Clone)]
pub struct IndexerW {
    pub wq_b: WMat,
    pub weights_proj: WMat,
    pub compressor: CompressorW,
}

#[derive(Clone)]
pub struct ExpertW {
    pub w1: WMat,
    pub w2: WMat,
    pub w3: WMat,
}

#[derive(Clone)]
pub struct LayerW {
    pub attn_sink: Vec<f32>,
    pub wq_a: WMat,
    pub q_norm: Vec<f32>,
    pub wq_b: WMat,
    pub wkv: WMat,
    pub kv_norm: Vec<f32>,
    /// fp8 on disk, dequantized to bf16 at load exactly as `convert.py`'s `wo_a` branch
    /// does, because `Attention.forward` consumes it raw in an einsum rather than through
    /// `Linear.forward` — there is no activation quantization on this one.
    pub wo_a: WMat,
    pub wo_b: WMat,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub gate_w: WMat,
    /// `Some` iff the layer routes by score (`layer_id >= n_hash_layers`).
    pub gate_bias: Option<Vec<f32>>,
    /// `Some` iff the layer routes by hash. `[vocab_size, n_activated_experts]`.
    pub tid2eid: Option<Vec<i64>>,
    pub compressor: Option<CompressorW>,
    /// Present only where `compress_ratio == 4` — 21 of the 43 layers.
    pub indexer: Option<IndexerW>,
    /// Routed experts, indexed by expert id. Sparse: only the ones a run actually reaches
    /// are loaded, since one is 13.37 MB.
    pub experts: std::collections::HashMap<usize, ExpertW>,
    pub shared: ExpertW,
}

/// Everything one `Block.forward` call needs that does not vary within it.
///
/// A struct rather than eight more parameters: `attention`, `moe`, `gate` and `run_layer`
/// all took the same tail, and four copies of a parameter list is four places for `s` and
/// `start_pos` to get swapped. `s` is the number of query rows — the prompt length at
/// prefill, and 1 at decode, which is also `start_pos`'s discriminant (`start_pos == 0`
/// means prefill throughout the reference).
pub struct LayerCtx<'a> {
    pub lw: &'a LayerW,
    pub layer: usize,
    pub s: usize,
    pub start_pos: usize,
    pub input_ids: &'a [u32],
    /// Which call this is: `"pre"` for the prefill, `"dec0"`, `"dec1"`, ... for the decode
    /// steps. NOT the golden prefix -- see [`LayerCtx::tag`].
    pub step_tag: &'a str,
}

impl LayerCtx<'_> {
    /// The prefix every recorded golden carries: `L{layer}.{step_tag}`.
    ///
    /// A method, not a field, because it must be impossible to apply inconsistently. When
    /// the layer id was prepended in `run_layer` alone, the goldens pushed inside
    /// `attention` and `moe` kept the bare step tag, a four-layer run wrote `pre.q` four
    /// times, and `Capture::float` -- which returns the FIRST match -- silently hid three of
    /// them. Every push in this file goes through here.
    pub fn tag(&self) -> String {
        format!("L{}.{}", self.layer, self.step_tag)
    }
}

// ---------------------------------------------------------------------------------------
// mutable state
// ---------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct CompState {
    /// `[coff * ratio, coff * d]`, f32, zero-initialised.
    pub kv_state: Vec<f32>,
    /// `[coff * ratio, coff * d]`, f32, `-inf`-initialised.
    pub score_state: Vec<f32>,
    /// `[max_seq_len / ratio, d]` — the compressed region this compressor writes into.
    /// For the attention compressor this is a VIEW of `kv_cache[window_size..]` in the
    /// reference; here it is a separate buffer that `Attention` concatenates, which is the
    /// same values in the same order.
    pub cache: Vec<f32>,
}

/// One layer's host-side caches, carried across decode steps: the sliding-window ring
/// plus the two compressors' pooling state. Named for the ring because that is the part
/// `Oracle::attention` indexes modulo the window; the two `CompState` halves are
/// append-only.
pub struct LayerRings {
    /// `[window_size, head_dim]` — the sliding-window ring only. The compressed region
    /// lives in `comp.cache`.
    pub win_cache: Vec<f32>,
    pub comp: Option<CompState>,
    pub idx_comp: Option<CompState>,
}

// ---------------------------------------------------------------------------------------
// captured goldens
// ---------------------------------------------------------------------------------------

/// What the oracle records. Float tensors are the goldens proper; the integer tensors are
/// SELECTION goldens (indexer top-k, router choices), which no numeric tolerance can stand
/// in for — a wrong Hadamard basis or a wrong router tie-break changes *which* values are
/// combined while leaving every magnitude plausible.
#[derive(Default, Clone)]
pub struct Capture {
    pub floats: Vec<Named<f32>>,
    pub ints: Vec<Named<i64>>,
    pub counters: Counters,
}

/// Reachability counters. These exist so the defect matrix can assert magnitude-gated
/// defects BIDIRECTIONALLY without fitting the expectation to the observation: e.g.
/// `SwigluUnclamped` must perturb a case iff `swiglu_clamped > 0` in that case, which is
/// measured independently of whether the defect fired.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counters {
    /// Clamp EVENTS the `swiglu_limit = 10.0` bound caused -- not elements: one element can
    /// contribute twice when both its `up` and its `gate` are out of range. Only ever read
    /// as zero-vs-nonzero, so the distinction costs nothing, but the name should not lie.
    pub swiglu_clamp_events: usize,
    /// Router logits at which `ln(1 + e^x)` OVERFLOWS f32 -- i.e. where dropping
    /// `softplus`'s `threshold = 20` identity branch is observable at all.
    ///
    /// NOT "logits above 20", which is what this counted first and is the wrong instrument:
    /// for `20 < x < ~88` the two forms are bit-identical in f32 (`ln(1+e^x) = x +
    /// ln(1+e^-x)`, and at x = 21 the correction is 7.6e-10 against an ulp of 1.9e-6). The
    /// threshold only becomes load-bearing where `e^x` reaches infinity, near 88. A counter
    /// that fired at 20 would make `RouterNoSoftplusThreshold` look reachable in a range
    /// where it provably is not.
    pub softplus_overflows: usize,
    /// Compressed blocks emitted by ANY compressor in this call -- the attention one and,
    /// on a ratio-4 layer, the indexer's own. They always fire together, which is why one
    /// counter suffices; `reachable()` reads it as "compression happened at all".
    pub compressed_blocks: usize,
    /// Prefill positions that did NOT survive into the sliding-window ring —
    /// `seqlen.saturating_sub(window_size)`. Zero means the ring was never rotated, so
    /// `PrefillRingWritesFirstWindow` is inert by construction.
    pub prefill_evicted: usize,
    /// Blocks the INDEXER's own compressor emitted. Separate from `compressed_blocks`,
    /// which counted both and so read 2x on any ratio-4 layer -- harmless while only
    /// `> 0` was read, and silently wrong for the first predicate written on the count.
    pub indexer_compressed_blocks: usize,
    /// Query rows where `index_topk` actually CUT (`k < n_compressed`).
    ///
    /// Zero means the indexer selected every compressed block, so `.compress_idxs` records
    /// an invariant set and cannot distinguish a right ranking from a wrong one. Without
    /// this counter that vacuity is invisible -- the goldens still exist, still compare
    /// equal, and still look like coverage.
    pub indexer_truncated: usize,
    /// The indexer ran (this layer has `compress_ratio == 4`).
    pub indexer_ran: bool,
}

impl Capture {
    /// Record a float tensor. Public so a driver can add goldens the layer body does not
    /// produce -- the embedding, a head output -- under the same naming.
    ///
    /// A duplicate name is a hard error, not a second entry. `float()` returns the FIRST
    /// match, so a collision makes every later tensor of that name invisible to both the
    /// comparator and the golden file -- and the four-layer emit produced exactly that
    /// before `run_layer` started prefixing the layer id. Silent shadowing is the failure
    /// mode this whole oracle exists to not have.
    pub fn push(&mut self, name: &str, shape: &[usize], v: Vec<f32>) {
        push_unique(&mut self.floats, name, shape, v);
    }
    /// Record an integer (selection) tensor. Same uniqueness rule as [`Capture::push`].
    pub fn push_i(&mut self, name: &str, shape: &[usize], v: Vec<i64>) {
        push_unique(&mut self.ints, name, shape, v);
    }
    pub fn float(&self, name: &str) -> Option<&[f32]> {
        find_tensor(&self.floats, name)
    }
    pub fn int(&self, name: &str) -> Option<&[i64]> {
        find_tensor(&self.ints, name)
    }
}

/// One recorded tensor: `(name, shape, values)`.
type Named<T> = (String, Vec<usize>, Vec<T>);

/// The FIRST tensor of this name, which is what makes [`push_unique`]'s duplicate assertion
/// load-bearing rather than decorative. Generic so `floats` and `ints` cannot drift apart.
fn find_tensor<'a, T>(from: &'a [Named<T>], name: &str) -> Option<&'a [T]> {
    from.iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, _, v)| v.as_slice())
}

/// Append one named tensor, refusing a duplicate name and a shape that does not describe it.
///
/// Takes the destination list rather than `&mut Capture`: `floats` and `ints` are SEPARATE
/// namespaces, and a helper that searched both would refuse a legal `foo` recorded once as
/// each. So each call still fails for its own caller's reason, and names its own tensor.
fn push_unique<T>(into: &mut Vec<Named<T>>, name: &str, shape: &[usize], v: Vec<T>) {
    assert_eq!(
        shape.iter().product::<usize>(),
        v.len(),
        "{name}: shape/len mismatch"
    );
    assert!(
        find_tensor(into, name).is_none(),
        "duplicate golden name {name}"
    );
    into.push((name.to_string(), shape.to_vec(), v));
}

// ---------------------------------------------------------------------------------------
// the oracle
// ---------------------------------------------------------------------------------------

pub struct Oracle {
    pub cfg: V4Config,
    pub defect: Defect,
    /// `precompute_freqs_cis(rope_head_dim, max_seq_len, 0, rope_theta, …)` — no YaRN.
    freqs_base: Vec<(f32, f32)>,
    /// `precompute_freqs_cis(rope_head_dim, max_seq_len, original_seq_len,
    /// compress_rope_theta, …)` — with YaRN.
    freqs_yarn: Vec<(f32, f32)>,
    /// YaRN dropped, `compress_rope_theta` kept — only `Defect::RopeNoYarn` uses this.
    freqs_no_yarn_compress_theta: Vec<(f32, f32)>,
    /// YaRN kept, base `rope_theta` — only `Defect::RopeBaseThetaEverywhere` uses this.
    freqs_yarn_base_theta: Vec<(f32, f32)>,
}

impl Oracle {
    pub fn new(cfg: V4Config, defect: Defect) -> Self {
        // The two tables `Attention.__init__` builds. They differ ONLY in the pair below:
        // a ratio-0 layer passes `original_seq_len = 0`, which disables the YaRN branch, and
        // the base `rope_theta`; a compressed layer passes the real length and
        // `compress_rope_theta`. Everything else is shared, so it comes from `cfg`.
        let freqs_base = precompute_freqs_cis(&cfg, 0, cfg.rope_theta);
        let freqs_yarn = precompute_freqs_cis(&cfg, cfg.original_seq_len, cfg.compress_rope_theta);
        // The other two corners of the (YaRN on/off) x (which theta) square. Built here
        // rather than inside the defect so the two rope defects cannot collapse into one
        // table and be counted as two independent pieces of evidence.
        let freqs_no_yarn_compress_theta = precompute_freqs_cis(&cfg, 0, cfg.compress_rope_theta);
        let freqs_yarn_base_theta =
            precompute_freqs_cis(&cfg, cfg.original_seq_len, cfg.rope_theta);
        Self {
            cfg,
            defect,
            freqs_base,
            freqs_yarn,
            freqs_no_yarn_compress_theta,
            freqs_yarn_base_theta,
        }
    }

    /// `Attention.__init__`'s per-layer table selection (model.py:481-488): compressed
    /// layers get YaRN and `compress_rope_theta`; `compress_ratio == 0` disables the
    /// interpolation branch (`original_seq_len = 0`) and uses base `rope_theta`.
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn freqs(&self, layer: usize) -> &[(f32, f32)] {
        let compressed = self.cfg.compress_ratio(layer) != 0;
        match self.defect {
            Defect::RopeNoYarn if compressed => &self.freqs_no_yarn_compress_theta,
            Defect::RopeBaseThetaEverywhere if compressed => &self.freqs_yarn_base_theta,
            Defect::RopeYarnEverywhere if !compressed => &self.freqs_yarn,
            _ if compressed => &self.freqs_yarn,
            _ => &self.freqs_base,
        }
    }

    pub fn fresh_state(&self, layer: usize) -> LayerRings {
        let c = &self.cfg;
        let ratio = c.compress_ratio(layer);
        let comp =
            (ratio != 0).then(|| new_comp_state(ratio, c.head_dim, ratio == 4, c.max_seq_len));
        let idx_comp =
            (ratio == 4).then(|| new_comp_state(ratio, c.index_head_dim, true, c.max_seq_len));
        LayerRings {
            win_cache: vec![0.0; c.window_size * c.head_dim],
            comp,
            idx_comp,
        }
    }

    fn round_bf16(&self, v: &mut [f32]) {
        if self.defect == Defect::NoBf16Rounding {
            return;
        }
        for x in v.iter_mut() {
            *x = bf16_decode(bf16_encode(*x));
        }
    }

    /// A bf16 REDUCTION as PyTorch performs one: accumulate in **f32**, round to bf16 **once**
    /// at the end. Its only call site is the indexer's per-head score sum (model.py:427).
    ///
    /// **Corrected 2026-08-05.** It previously rounded the accumulator after every term — a
    /// running bf16 fold — under a comment reading "`.sum(dim=2)` -> bf16". That is true of
    /// the output DTYPE and false of the ACCUMULATOR: torch reduces reduced-precision floats
    /// through `acc_type`, i.e. f32, and rounds to the output dtype once. **The justification
    /// is that fidelity argument, and only that** — `acc_type` is a property of the reduction
    /// and the oracle was modelling the output dtype in its place.
    ///
    /// Measured on CPU torch, 2026-08-05, at this call site's real summand
    /// `relu(einsum) * weights_proj(x)` — note the `relu_` applies to the einsum ONLY and
    /// `weights_proj` is a bare `Linear` with no activation (model.py:400, :424), so the
    /// terms are **signed and can cancel**. 64 heads, 4000 trials: the running fold disagreed
    /// with `x.sum()` **72.6%** of the time, max |delta| **0.25**, mean signed delta
    /// **-1.7e-4**. An independent run reported 73.0% / 0.125 / +1.4e-4.
    ///
    /// So it is **noise, not drift** — there is no systematic direction, and any claim of one
    /// comes from assuming non-negative summands, which this site does not have. A 72.6%
    /// disagreement rate at up to 0.25 absolute is sufficient on its own: this chain decides
    /// WHICH positions are attended, and `.compress_idxs` is recorded score-ORDERED, so a
    /// perturbation changes the arithmetic even when the selected set survives it.
    ///
    /// **What this does NOT close.** The old fold pinned the summation ORDER as a side
    /// effect; an f32 accumulator does not, and torch's reduction is vectorized and
    /// tree-shaped while this one is a sequential fold. "Accumulate in f32, round once" does
    /// not by itself pick a unique answer. Measured rather than argued, 2026-08-05: a
    /// sequential f32 fold rounded once agreed with `torch.sum()` bit for bit in
    /// **20000 of 20000** trials — 5000 each at n = 4, n = 64, n = 512, and n = 64 with
    /// magnitudes spread 32x. Not a proof: re-association noise is ~2^-21 relative against a
    /// 2^-8 half-ulp rounding margin (the bf16 ulp at 1.0 is 2^-7 -- bf16 keeps 7 explicit
    /// mantissa bits), so a disagreement needs the f32 result to land within re-association
    /// distance of a rounding boundary, which is rare rather than impossible. Treat the
    /// ordering as unpinned-but-unobserved, and if a device kernel ever disagrees here by
    /// exactly one bf16 ulp, this is the first place to look.
    pub fn bf16_sum(&self, terms: impl Iterator<Item = f32>) -> f32 {
        let b = |x: f32| bf16_decode(bf16_encode(x));
        match self.defect {
            Defect::IndexerBf16RunningSum => terms.fold(0.0f32, |a, t| b(a + t)),
            _ => b(terms.sum::<f32>()),
        }
    }

    /// One dequantized weight row, honouring `Fp4NibbleSwap`.
    fn wrow(&self, w: &WMat, r: usize, buf: &mut Vec<f32>) {
        w.row(r, buf);
        if self.defect == Defect::Fp4NibbleSwap && matches!(w, WMat::Fp4 { .. }) {
            for pair in buf.chunks_exact_mut(2) {
                pair.swap(0, 1);
            }
        }
    }

    /// Whether the device would dispatch this GEMV to `gemv_fp8_bf16_splitk` — the oracle
    /// half of the ONE dispatch predicate, spec'd in
    /// `docs/investigations/v4-decode-decomposition.md` §M9 and mirrored by
    /// `kernels/mla.hip::rivoli_gemv_fp8_bf16`.
    ///
    /// `WMat::Fp8` here stands for "goes through `gemv_fp8_bf16`" — every fp8-quantized
    /// linear on the oracle's path does, and the one fp8-on-disk tensor consumed as Dense
    /// (`wo_a`, whose device GEMV reads the fp8 bytes but is measured bit-equal to the
    /// bf16 dequant) is excluded by shape anyway: `n_out = 8192 > 2048`, and it is also
    /// the launcher's only `groups > 1` caller. The mirror is three of the launcher's
    /// five terms; the two unmirrored ones discharge structurally — `groups == 1` by
    /// wo_a's double exclusion, `block >= 4` because the oracle's `Fp8` is the 128x128
    /// grid only. The shape terms are the launcher's own, same constants: at the seven
    /// decode shapes this selects exactly `wq_a` [1024x4096], `wkv` [512x4096] and the
    /// shared expert's gate/up [2048x4096] at `m = 1`. The bound is INCLUSIVE, so tiny
    /// prefills can select too — `wq_a` at exactly `m = 2`, `wkv` through `m = 4` —
    /// consistently on both sides; at the recorded prompts (goldens 13 tokens, bench
    /// 218) prefill selects nothing. `k % 4 == 0` mirrors the launcher's guard that
    /// keeps the fold on the dword path the spec orders (every captured `k` is 4096).
    fn splitk_selects(&self, m: usize, n: usize, k: usize, w: &WMat) -> bool {
        self.defect == Defect::SplitKFoldOrder
            && matches!(w, WMat::Fp8 { .. })
            && m * n <= 2048
            && k >= 4096
            && k.is_multiple_of(4)
    }

    /// `model.py::linear` — dispatches on the weight's storage format. `x` is `[m, k]`
    /// row-major; the result is `[m, n]` and is NOT bf16-rounded here (callers round where
    /// the reference stores).
    ///
    /// `pub` since 2026-08-08 (the `qk_norm` precedent) so
    /// `tests/v4_oracle.rs::the_splitk_fold_is_toy_blind_partition_exact_and_nonzero_at_real_dims`
    /// can drive the `splitk_selects` WIRING at real dims: review found that no committed
    /// test ever reached `linear` with the predicate true — the toy drive selects nothing
    /// by design and the real-dims probes called `splitk_fold` directly — so a predicate
    /// typo that made it never-true would reproduce §M9's "all tensors bit-identical"
    /// host result VACUOUSLY. The unrounded return is what makes the wiring observable
    /// (the raw fold delta is ~59% of sums; the bf16 stores downstream absorb it).
    pub fn linear(&self, x: &[f32], m: usize, k: usize, w: &WMat) -> Vec<f32> {
        // `assert`, not `debug_assert`: `[profile.release]` does not set `debug-assertions`
        // and CLAUDE.md prescribes `cargo test --release`, so a debug assertion here would
        // never run in the only configuration anyone uses. These are per-GEMM, not
        // per-element -- the cost is nil against a 4000-line CPU oracle.
        assert_eq!(
            w.cols(),
            k,
            "linear: weight expects k={}, got {k}",
            w.cols()
        );
        assert_eq!(x.len(), m * k);
        let n = w.rows();
        // For quantized weights the activation is first quantized to fp8 at block 128 with
        // a ue8m0 scale. `linear()` does this out-of-place, so the caller's x is untouched.
        let xq = match w {
            WMat::Dense { .. } => None,
            WMat::Fp8 { .. } | WMat::Fp4 { .. } => {
                // `kernel.py::act_quant` line 112: `assert N % block_size == 0`. Without
                // this, a future config with a non-multiple K would quantize a short tail
                // block and produce values the reference cannot produce, silently.
                assert!(
                    k.is_multiple_of(128),
                    "act_quant needs K % 128 == 0, got {k}"
                );
                let mut q = x.to_vec();
                for row in q.chunks_mut(k) {
                    act_quant_inplace(row, 128, true);
                }
                Some(q)
            }
        };
        let a = xq.as_deref().unwrap_or(x);
        let splitk = self.splitk_selects(m, n, k, w);

        let mut out = vec![0.0f32; m * n];
        let mut wr = Vec::with_capacity(k);
        for j in 0..n {
            self.wrow(w, j, &mut wr);
            for i in 0..m {
                let xi = &a[i * k..i * k + k];
                out[i * n + j] = if splitk {
                    splitk_fold(xi, &wr)
                } else {
                    let mut acc = 0.0f32;
                    for t in 0..k {
                        acc += xi[t] * wr[t];
                    }
                    acc
                };
            }
        }
        out
    }

    /// `RMSNorm.forward` — fp32 internals, learned weight, then the bf16 store the reference
    /// performs on the way out (`(self.weight * x).to(dtype)`, model.py:202).
    fn rmsnorm(&self, x: &mut [f32], d: usize, w: &[f32]) {
        self.rmsnorm_raw(x, d, w);
        self.round_bf16(x);
    }

    /// `RMSNorm.forward` without that store.
    ///
    /// Split out for the head tail alone, which needs both halves independently: the store is
    /// what `Defect::HeadNormNotBf16` suppresses, and `d` is what `HeadNormOverAllTokens`
    /// widens to `s * dim`. Both are choices a device implementation actually faces —
    /// `kernels/mla.hip::rmsnorm_batch` bf16-rounds and is one block per row, while
    /// `kernels/linalg.hip::rmsnorm_rows` neither rounds nor spans rows.
    fn rmsnorm_raw(&self, x: &mut [f32], d: usize, w: &[f32]) {
        // `zip` below stops at the shorter side, so a short `w` would leave the tail of every
        // row not merely un-gained but UN-SCALED by `rs`, and say nothing. That is reachable:
        // the checkpoint carries `norm.weight` [4096] beside `q_norm.weight` [1024] and
        // `kv_norm.weight` [512], and `load_head_tail` flattens whatever tensor it is handed.
        // Same reasoning as `hc_head`'s `hc_head_scale` assert -- read the shape rather than
        // trust the caller -- and this was the one norm parameter whose mis-load was silent.
        assert_eq!(
            w.len(),
            d,
            "RMSNorm weight must be [d]; a short one truncates in silence"
        );
        for row in x.chunks_mut(d) {
            let var = row.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let rs = (var + self.cfg.norm_eps).sqrt().recip();
            for (v, &g) in row.iter_mut().zip(w) {
                *v = g * (*v * rs);
            }
        }
    }

    /// The QK-norm: `q *= rsqrt(q.square().mean(-1) + eps)` over `head_dim`, applied after
    /// the unflatten to (heads, head_dim), with **no learnable weight** (model.py:504).
    ///
    /// `pub` since 2026-08-06 so `tests/headtail.rs` can score
    /// `kernels/mla.hip::qk_norm` against it. The alternative was a host transliteration
    /// of the seven lines below, which `build.rs`'s duplication gate would have rejected —
    /// and rightly, because a reference carrying the same bf16 placement as the thing it
    /// scores is wrong in the same places.
    pub fn qk_norm(&self, q: &mut [f32], head_dim: usize, q_norm_w: &[f32]) {
        if self.defect == Defect::SkipQkNorm {
            return;
        }
        // Computed in BF16, unlike `rmsnorm`. `RMSNorm.forward` (model.py:197-202) opens
        // with an explicit `x = x.float()`; model.py:504 does not — `q` is bf16 out of
        // `fp8_gemm`, so `q.square()`, the `+ eps` and `torch.rsqrt` are all bf16-valued.
        // Keeping the statistic in f32 leaves `rs` up to 2^-9 (~0.2%) off, and it is the
        // SAME factor for every dim of a head, so it scales that head's whole logit row —
        // two orders of magnitude above the re-association floor this file claims.
        //
        // INFERRED: `.mean(-1)` on a bf16 tensor reduces through torch's `acc_type`, i.e. an
        // f32 accumulator with a bf16 result. The per-element squares are bf16 either way.
        let b = |x: f32| bf16_decode(bf16_encode(x));
        for row in q.chunks_mut(head_dim) {
            let var = b(row.iter().map(|v| b(v * v)).sum::<f32>() / head_dim as f32);
            let rs = b(b(var + self.cfg.norm_eps).sqrt().recip());
            for (i, v) in row.iter_mut().enumerate() {
                *v = b(*v * rs);
                if self.defect == Defect::QkNormUsesQNormWeight {
                    *v = b(*v * q_norm_w[i % q_norm_w.len()]);
                }
            }
        }
    }

    /// `apply_rotary_emb` over one row of `len` values: rotates the LAST `rd` of them,
    /// pairing adjacent dims as `view_as_complex` does, and rounds the in-place `copy_`
    /// back into the bf16 view.
    ///
    /// `RopeAllDims` covers more dims than the table has frequencies, so it cycles the
    /// table. There is no "right" way to be wrong here; cycling is what a kernel written
    /// with the wrong slice bound and the reference's own `freqs_cis` would do.
    fn rope_row(&self, row: &mut [f32], rd: usize, f: (usize, &[(f32, f32)]), inverse: bool) {
        let (pos, table) = f;
        let len = row.len();
        let (start, n) = match self.defect {
            Defect::RopeAllDims => (0, len),
            Defect::RopeFirstDims => (0, rd),
            _ => (len - rd, rd),
        };
        let half = n / 2;
        let seg = &mut row[start..start + n];
        for i in 0..half {
            let (c, s) = table[pos * (rd / 2) + (i % (rd / 2))];
            let s = if inverse { -s } else { s };
            let (ia, ib) = if self.defect == Defect::RopeHalfSplit {
                (i, i + half)
            } else {
                (2 * i, 2 * i + 1)
            };
            let (a, b) = (seg[ia], seg[ib]);
            seg[ia] = a * c - b * s;
            seg[ib] = a * s + b * c;
        }
        self.round_bf16(row);
    }
}

fn new_comp_state(ratio: usize, d: usize, overlap: bool, max_seq_len: usize) -> CompState {
    let coff = 1 + usize::from(overlap);
    CompState {
        kv_state: vec![0.0; coff * ratio * coff * d],
        score_state: vec![f32::NEG_INFINITY; coff * ratio * coff * d],
        cache: vec![0.0; (max_seq_len / ratio) * d],
    }
}

// ---------------------------------------------------------------------------------------
// RoPE tables
// ---------------------------------------------------------------------------------------

/// `model.py::precompute_freqs_cis`, verbatim including the quirk that `find_correction_range`
/// works in `dim` units while `linear_ramp_factor` indexes `dim // 2` entries.
///
/// Returns `[seqlen * (dim/2)]` pairs `(cos, sin)`.
fn precompute_freqs_cis(cfg: &V4Config, original_seq_len: usize, base: f32) -> Vec<(f32, f32)> {
    let (dim, seqlen) = (cfg.rope_head_dim, cfg.max_seq_len);
    let (factor, beta_fast, beta_slow) = (cfg.rope_factor, cfg.beta_fast, cfg.beta_slow);
    let half = dim / 2;
    // `freqs`, the YaRN blend, the outer product and `polar` are all FLOAT32 in the
    // reference; only `find_correction_dim`/`find_correction_range` are doubles (they go
    // through Python's `math.log`/`floor`). Computing the table in f64 throughout leaves a
    // sub-bf16-ulp angle error that still shows up as sporadic one-ulp disagreement against
    // a faithful implementation.
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / dim as f32))
        .collect();
    if original_seq_len > 0 {
        let base = f64::from(base);
        let fcd = |rot: f64| {
            dim as f64 * (original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = fcd(beta_fast as f64).floor().max(0.0);
        let high = fcd(beta_slow as f64).ceil().min(dim as f64 - 1.0);
        let (min, max) = if low == high {
            (low, high + 0.001)
        } else {
            (low, high)
        };
        let (min, max) = (min as f32, max as f32);
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f32 - min) / (max - min)).clamp(0.0, 1.0);
            let smooth = 1.0 - ramp;
            *f = *f / factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut out = Vec::with_capacity(seqlen * half);
    for t in 0..seqlen {
        for f in &freqs {
            let a = t as f32 * f;
            out.push((a.cos(), a.sin()));
        }
    }
    out
}

// ---------------------------------------------------------------------------------------
// helpers shared by the block body
// ---------------------------------------------------------------------------------------

/// Softmax over `n` elements strided by `stride`, in fp32, `-inf`-safe.
fn softmax_strided(v: &mut [f32], n: usize, stride: usize, offset: usize) {
    let mut m = f32::NEG_INFINITY;
    for i in 0..n {
        m = m.max(v[offset + i * stride]);
    }
    if !m.is_finite() {
        // Every entry masked: the reference's `exp(-inf - -inf)` is NaN, but this only
        // arises for a fully-masked pooling window, which the callers never construct.
        for i in 0..n {
            v[offset + i * stride] = 0.0;
        }
        return;
    }
    let mut s = 0.0f32;
    for i in 0..n {
        let e = (v[offset + i * stride] - m).exp();
        v[offset + i * stride] = e;
        s += e;
    }
    for i in 0..n {
        v[offset + i * stride] /= s;
    }
}

/// `torch.topk(v, k)[1]` — descending by value, ties to the LOWER index.
///
/// **`.compress_idxs` is therefore score-ORDERED, and a consumer must compare it as a SET.**
/// A scoring difference too small to matter still permutes the list, and PyTorch does not
/// guarantee a tie-break on CUDA at all, so a positional comparison is stricter than the
/// arithmetic supports. `.indexer_scores` records the full pre-top-k matrix alongside, so a
/// tie-break disagreement can be told from a real scoring disagreement.
fn topk_idx(v: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    // `total_cmp`, not `partial_cmp`: mapping NaN to `Equal` is not a total order and
    // Rust >= 1.81's `sort_by` may panic on one. Scores cannot be NaN today, but a sort
    // comparator is not the place to rely on that.
    order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

// ---------------------------------------------------------------------------------------
// Block: hyper-connections
// ---------------------------------------------------------------------------------------

/// The Sinkhorn mixture one `hc_pre` produced, for the `hc_post` that closes the SAME
/// sublayer — `post` is `[s, hc]` and `comb` is `[s, hc, hc]`.
///
/// A block runs the pair twice, around attention and around the FFN, at identical types and
/// shapes: crossing the two halves was a well-typed call before this type existed.
struct HcMix {
    post: Vec<f32>,
    comb: Vec<f32>,
}

impl Oracle {
    /// `kernel.py::hc_split_sinkhorn` for one token.
    ///
    /// `mixes` is `[(2 + hc) * hc]`: `hc` pre-weights, `hc` post-weights, then `hc * hc`
    /// combination logits. Note the FIRST normalisation pair is a row *softmax* followed by
    /// a column divide, and only the remaining `iters - 1` passes are plain row/column
    /// divides — that asymmetry is easy to lose in a port.
    fn hc_split_sinkhorn(
        &self,
        mixes: &[f32],
        scale: &[f32],
        base: &[f32],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hc = self.cfg.hc_mult;
        let eps = self.cfg.hc_eps;
        let mut pre = vec![0.0f32; hc];
        let mut post = vec![0.0f32; hc];
        let mut comb = vec![0.0f32; hc * hc];
        for j in 0..hc {
            pre[j] = sigmoid(mixes[j] * scale[0] + base[j]) + eps;
            post[j] = 2.0 * sigmoid(mixes[j + hc] * scale[1] + base[j + hc]);
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] =
                    mixes[j * hc + k + 2 * hc] * scale[2] + base[j * hc + k + 2 * hc];
            }
        }
        // comb = comb.softmax(-1) + eps
        for j in 0..hc {
            softmax_strided(&mut comb, hc, 1, j * hc);
            for k in 0..hc {
                comb[j * hc + k] += eps;
            }
        }
        // The Sinkhorn passes: `comb / (comb.sum(-1) + eps)` and `comb / (comb.sum(-2) + eps)`
        // differ only in which index they hold fixed, so one normaliser takes that as an
        // index function. Two copies would be two places to get the eps or the axis wrong.
        let norm = |c: &mut [f32], by_row: bool| {
            let at = |fixed: usize, run: usize| {
                if by_row {
                    fixed * hc + run
                } else {
                    run * hc + fixed
                }
            };
            for fixed in 0..hc {
                let s: f32 = (0..hc).map(|r| c[at(fixed, r)]).sum();
                for r in 0..hc {
                    c[at(fixed, r)] /= s + eps;
                }
            }
        };
        norm(&mut comb, false);
        let iters = if self.defect == Defect::SinkhornIterCountProbe {
            self.cfg.hc_sinkhorn_iters - 1
        } else {
            self.cfg.hc_sinkhorn_iters
        };
        for _ in 0..iters.saturating_sub(1) {
            norm(&mut comb, true);
            norm(&mut comb, false);
        }
        (pre, post, comb)
    }

    /// `torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)` — collapse one token's `hc_mult`
    /// residual copies by the gate vector `pre`, writing `dim` values.
    ///
    /// Extracted because `hc_pre` and `hc_head` share it VERBATIM in the reference too
    /// (model.py:687 and :715 are the same expression), so this is one behaviour with two
    /// call sites rather than two behaviours that happen to look alike — the case where
    /// factoring cannot hide a divergence. (It was forced by `jscpd`, which flagged the two
    /// copies at 98 tokens; the reference-shares-it argument is why the answer was to factor
    /// rather than to add an ignore marker.)
    ///
    /// Each output element accumulates across the copies in copy order, the order `hc_pre`
    /// had before the extraction -- which the diff shows directly, the body being the same
    /// fold with `out.iter_mut().enumerate()` in place of `for d in 0..dim`. So no golden
    /// moved, by inspection rather than by measurement. A 200-trial bitwise run at
    /// `hc = 4, dim = 257`, with gate weights and residual copies spread across 2^-20..2^10
    /// so that cancellation was available, agreed at **0 bitwise differences**, and reported 184
    /// differences under a one-ulp control -- but that only confirmed that an identical fold
    /// is identical.
    fn hc_blend(&self, pre: &[f32], flat: &[f32], out: &mut [f32]) {
        let dim = self.cfg.dim;
        for (d, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (c, &p) in pre.iter().enumerate() {
                acc += p * flat[c * dim + d];
            }
            *o = acc;
        }
    }

    /// `Block.hc_pre`. `h` is `[s, hc, dim]`; the rsqrt is over the FULL `hc * dim`
    /// flattened row, not per copy.
    fn hc_pre(
        &self,
        h: &[f32],
        s: usize,
        fnw: &[f32],
        scale: &[f32],
        base: &[f32],
    ) -> (Vec<f32>, HcMix) {
        let (hc, dim, mix) = (self.cfg.hc_mult, self.cfg.dim, self.cfg.mix_hc());
        let hcd = hc * dim;
        let mut y = vec![0.0f32; s * dim];
        let mut post = vec![0.0f32; s * hc];
        let mut comb = vec![0.0f32; s * hc * hc];
        for t in 0..s {
            let flat = &h[t * hcd..(t + 1) * hcd];
            let var = flat.iter().map(|v| v * v).sum::<f32>() / hcd as f32;
            let rs = if self.defect == Defect::HcPreNoRsqrt {
                1.0
            } else {
                (var + self.cfg.norm_eps).sqrt().recip()
            };
            let mut mixes = vec![0.0f32; mix];
            for (j, m) in mixes.iter_mut().enumerate() {
                let w = &fnw[j * hcd..(j + 1) * hcd];
                *m = flat.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() * rs;
            }
            let (pre, row_post, row_comb) = self.hc_split_sinkhorn(&mixes, scale, base);
            self.hc_blend(&pre, flat, &mut y[t * dim..(t + 1) * dim]);
            post[t * hc..(t + 1) * hc].copy_from_slice(&row_post);
            comb[t * hc * hc..(t + 1) * hc * hc].copy_from_slice(&row_comb);
        }
        // `y.to(dtype)` — back to bf16.
        self.round_bf16(&mut y);
        (y, HcMix { post, comb })
    }

    /// `Block.hc_post`: `y[k] = post[k] * x + sum_j comb[j, k] * residual[j]`.
    ///
    /// `mix.comb` is indexed `[source, dest]` — the Sinkhorn row-softmax runs over the DEST
    /// index and the column normalisation over the SOURCE index. Transposing it keeps every
    /// row of the result a convex-ish combination of the same vectors and is therefore
    /// invisible to any magnitude check.
    fn hc_post(&self, x: &[f32], residual: &[f32], mix: &HcMix, s: usize) -> Vec<f32> {
        let HcMix { post, comb } = mix;
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let mut y = vec![0.0f32; s * hc * dim];
        for t in 0..s {
            for k in 0..hc {
                for d in 0..dim {
                    let mut acc = post[t * hc + k] * x[t * dim + d];
                    for j in 0..hc {
                        let c = if self.defect == Defect::SinkhornCombTransposed {
                            comb[t * hc * hc + k * hc + j]
                        } else {
                            comb[t * hc * hc + j * hc + k]
                        };
                        if self.defect != Defect::HcPostNoComb {
                            acc += c * residual[(t * hc + j) * dim + d];
                        }
                    }
                    if self.defect == Defect::HcPostNoComb {
                        acc += residual[(t * hc + k) * dim + d];
                    }
                    y[(t * hc + k) * dim + d] = acc;
                }
            }
        }
        // `.type_as(x)` — the residual stream is bf16.
        self.round_bf16(&mut y);
        y
    }
}

// ---------------------------------------------------------------------------------------
// Compressor
// ---------------------------------------------------------------------------------------

impl Oracle {
    /// `Compressor.forward`. `x` is `[s, dim]` in the block's activation dtype; the
    /// reference immediately casts it to f32 and stays there until the final `.to(dtype)`.
    ///
    /// Returns the compressed rows `[n_blocks, d]` — post-norm, post-RoPE, post-quantization,
    /// exactly what lands in the cache — or `None` when `should_compress` is false. In the
    /// `None` case the state updates have still happened, exactly as the reference does
    /// (model.py:331-367).
    #[allow(clippy::too_many_arguments)]
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn compressor(
        &self,
        cw: &CompressorW,
        cs: &mut CompState,
        x: &[f32],
        s: usize,
        start_pos: usize,
        freqs: &[(f32, f32)],
        counters: &mut Counters,
    ) -> Option<Vec<f32>> {
        let (ratio, d, coff) = (cw.ratio, cw.d, cw.coff());
        let overlap = cw.overlap && self.defect != Defect::CompressorNoOverlap;
        let rd = self.cfg.rope_head_dim;
        let cd = coff * d;
        let use_ape = self.defect != Defect::CompressorNoApe;

        let mut kv = self.linear(x, s, self.cfg.dim, &cw.wkv);
        let mut score = self.linear(x, s, self.cfg.dim, &cw.wgate);

        let (mut pooled, first_block) = if start_pos == 0 {
            let should = s >= ratio;
            let remainder = s % ratio;
            let cutoff = s - remainder;
            let state_off = if overlap { ratio } else { 0 };
            if overlap && cutoff >= ratio {
                for j in 0..ratio {
                    let src = (cutoff - ratio + j) * cd;
                    cs.kv_state[j * cd..(j + 1) * cd].copy_from_slice(&kv[src..src + cd]);
                    for e in 0..cd {
                        cs.score_state[j * cd + e] =
                            score[src + e] + if use_ape { cw.ape[j * cd + e] } else { 0.0 };
                    }
                }
            }
            if remainder > 0 {
                for j in 0..remainder {
                    let src = (cutoff + j) * cd;
                    let dst = (state_off + j) * cd;
                    cs.kv_state[dst..dst + cd].copy_from_slice(&kv[src..src + cd]);
                    for e in 0..cd {
                        cs.score_state[dst + e] =
                            score[src + e] + if use_ape { cw.ape[j * cd + e] } else { 0.0 };
                    }
                }
                kv.truncate(cutoff * cd);
                score.truncate(cutoff * cd);
            }
            if !should {
                return None;
            }
            let nblk = cutoff / ratio;
            // score += ape, per position within the block
            if use_ape {
                for b in 0..nblk {
                    for j in 0..ratio {
                        for e in 0..cd {
                            score[(b * ratio + j) * cd + e] += cw.ape[j * cd + e];
                        }
                    }
                }
            }
            // `overlap_transform`: 2*ratio entries of width d per block — the current
            // block's "normal" half in slots [ratio, 2*ratio) and the PREVIOUS block's
            // "overlap" half in slots [0, ratio), zero / -inf for block 0.
            let ents = if overlap { 2 * ratio } else { ratio };
            let mut kb = vec![0.0f32; nblk * ents * d];
            let mut sb = vec![f32::NEG_INFINITY; nblk * ents * d];
            for b in 0..nblk {
                for j in 0..ents {
                    let (src_blk, src_j, half) = if !overlap {
                        (Some(b), j, 0)
                    } else if j >= ratio {
                        (Some(b), j - ratio, d)
                    } else if b > 0 {
                        (Some(b - 1), j, 0)
                    } else {
                        (None, 0, 0)
                    };
                    let Some(sblk) = src_blk else { continue };
                    let src = (sblk * ratio + src_j) * cd + half;
                    let dst = (b * ents + j) * d;
                    kb[dst..dst + d].copy_from_slice(&kv[src..src + d]);
                    sb[dst..dst + d].copy_from_slice(&score[src..src + d]);
                }
            }
            // softmax over the ENTRY axis, independently per feature.
            let mut out = vec![0.0f32; nblk * d];
            for b in 0..nblk {
                for e in 0..d {
                    softmax_strided(&mut sb, ents, d, b * ents * d + e);
                    for j in 0..ents {
                        out[b * d + e] += kb[(b * ents + j) * d + e] * sb[(b * ents + j) * d + e];
                    }
                }
            }
            (out, 0)
        } else {
            let should = (start_pos + 1).is_multiple_of(ratio);
            let slot_in_block = start_pos % ratio;
            if use_ape {
                let ape = &cw.ape[slot_in_block * cd..(slot_in_block + 1) * cd];
                for (v, a) in score.iter_mut().zip(ape) {
                    *v += a;
                }
            }
            let state_off = if overlap { ratio } else { 0 };
            let dst = (state_off + slot_in_block) * cd;
            cs.kv_state[dst..dst + cd].copy_from_slice(&kv[..cd]);
            cs.score_state[dst..dst + cd].copy_from_slice(&score[..cd]);
            if !should {
                return None;
            }
            let ents = if overlap { 2 * ratio } else { ratio };
            // Gather: slots [0, ratio) contribute their first d dims (the overlap half),
            // slots [ratio, 2*ratio) their last d dims. Without overlap, cd == d and every
            // slot contributes all of itself.
            let mut kb = vec![0.0f32; ents * d];
            let mut sb = vec![0.0f32; ents * d];
            for j in 0..ents {
                let half = if overlap && j >= ratio { d } else { 0 };
                kb[j * d..(j + 1) * d]
                    .copy_from_slice(&cs.kv_state[j * cd + half..j * cd + half + d]);
                sb[j * d..(j + 1) * d]
                    .copy_from_slice(&cs.score_state[j * cd + half..j * cd + half + d]);
            }
            let mut out = vec![0.0f32; d];
            for e in 0..d {
                softmax_strided(&mut sb, ents, d, e);
                for j in 0..ents {
                    out[e] += kb[j * d + e] * sb[j * d + e];
                }
            }
            if overlap {
                let (lo, hi) = cs.kv_state.split_at_mut(ratio * cd);
                lo.copy_from_slice(&hi[..ratio * cd]);
                let (lo, hi) = cs.score_state.split_at_mut(ratio * cd);
                lo.copy_from_slice(&hi[..ratio * cd]);
            }
            (out, start_pos / ratio)
        };

        let nblk = pooled.len() / d;
        // `self.norm(kv.to(dtype))` — bf16 store, then RMSNorm back to bf16.
        self.round_bf16(&mut pooled);
        self.rmsnorm(&mut pooled, d, &cw.norm);
        for b in 0..nblk {
            // model.py:370/372 — the block is rotated at its FIRST position.
            let block = first_block + b;
            let pos = if self.defect == Defect::CompressorRopeAtBlockEnd {
                block * ratio + ratio - 1
            } else {
                block * ratio
            };
            let row = &mut pooled[b * d..(b + 1) * d];
            self.rope_row(row, rd, (pos, freqs), false);
            if cw.rotate {
                self.indexer_spread(row);
            } else {
                self.kv_act_quant(row, d, rd);
            }
            let dst = block * d;
            // Fail CLOSED. Silently dropping the row would leave the indexer scoring
            // queries against a zero slot -- fluent wrong text, no crash, which is the
            // exact failure mode this oracle exists to make impossible.
            assert!(
                dst + d <= cs.cache.len(),
                "compressed block {block} exceeds max_seq_len/{ratio}; raise cfg.max_seq_len"
            );
            cs.cache[dst..dst + d].copy_from_slice(row);
        }
        counters.compressed_blocks += nblk;
        Some(pooled)
    }

    /// `rotate_activation` then `fp4_act_quant(·, 32, inplace=True)` — what the indexer does
    /// to BOTH its query rows and its compressed kv rows (`Indexer.forward` lines 420-422,
    /// `Compressor.forward` lines 374-376). One helper because the pair must stay together:
    /// the Hadamard spread exists to make the fp4 grouping well-conditioned, so applying
    /// one without the other is a different algorithm, not a partial one.
    fn indexer_spread(&self, row: &mut [f32]) {
        if self.defect != Defect::IndexerNoHadamard {
            hadamard_rotate(row);
            self.round_bf16(row);
        }
        if self.defect != Defect::IndexerNoFp4Quant {
            fp4_act_quant_inplace(row, 32);
            self.round_bf16(row);
        }
    }

    /// The PARTIAL fp8 simulation of a KV entry: `act_quant(kv[..., :-rope_head_dim], 64,
    /// scale_fmt, scale_dtype, inplace=True)` — dims `[0, d - rd)` at block 64, dims
    /// `[d - rd, d)` left alone so the positional information keeps bf16 precision.
    fn kv_act_quant(&self, row: &mut [f32], d: usize, rd: usize) {
        let (n, block, round) = match self.defect {
            Defect::SkipKvActQuant => return,
            Defect::KvActQuantWholeTensor => (d, 64, true),
            Defect::KvActQuantBlock128 => (d - rd, 128, true),
            Defect::KvActQuantNoRoundScale => (d - rd, 64, false),
            _ => (d - rd, 64, true),
        };
        act_quant_inplace(&mut row[..n], block, round);
        self.round_bf16(row);
    }
}

// ---------------------------------------------------------------------------------------
// Indexer
// ---------------------------------------------------------------------------------------

impl Oracle {
    /// `Indexer.forward` — selects which compressed positions each query may attend to.
    ///
    /// Returns one index list per query row, already offset into the attention's kv space,
    /// with `-1` for masked slots. `scores_out` receives the FULL `[s, n_compressed]` score
    /// matrix — not just the selected entries — so that its length does not depend on the
    /// selection under test, and so a consumer can tell a `topk` tie-break disagreement from
    /// a real scoring disagreement.
    #[allow(clippy::too_many_arguments)]
    /// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
    /// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
    /// Driving these in isolation is what closes three measured coverage holes -- ratio-128
    /// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
    /// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
    /// Visibility only; no behaviour change.
    pub fn indexer(
        &self,
        step: &LayerCtx,
        iw: &IndexerW,
        cs: &mut CompState,
        x: &[f32],
        qr: &[f32],
        offset: usize,
        freqs: &[(f32, f32)],
        counters: &mut Counters,
        scores_out: &mut Vec<f32>,
    ) -> Vec<Vec<i64>> {
        let LayerCtx { s, start_pos, .. } = *step;
        let c = &self.cfg;
        let (h, hd, ratio, rd) = (
            c.index_n_heads,
            c.index_head_dim,
            iw.compressor.ratio,
            c.rope_head_dim,
        );
        let end_pos = start_pos + s;

        let mut q = self.linear(qr, s, c.q_lora_rank, &iw.wq_b);
        self.round_bf16(&mut q);
        for t in 0..s {
            for hh in 0..h {
                let row = &mut q[(t * h + hh) * hd..(t * h + hh + 1) * hd];
                self.rope_row(row, rd, (start_pos + t, freqs), false);
                self.indexer_spread(row);
            }
        }

        // Its own scratch counter: the indexer's compressor is not the attention's.
        let mut own = Counters::default();
        self.compressor(&iw.compressor, cs, x, s, start_pos, freqs, &mut own);
        counters.indexer_compressed_blocks += own.compressed_blocks;

        // `weights_proj(x) * (softmax_scale * n_heads ** -0.5)`.
        let mut w = self.linear(x, s, c.dim, &iw.weights_proj);
        self.round_bf16(&mut w);
        // bf16 all the way: `weights_proj` is a bf16 `Linear`, so the scale multiply lands
        // in bf16 too (model.py:424).
        let wscale = (hd as f32).powf(-0.5) * (h as f32).powf(-0.5);
        for v in w.iter_mut() {
            *v = bf16_decode(bf16_encode(*v * wscale));
        }

        let n_comp = end_pos / ratio;
        let mut out = Vec::with_capacity(s);
        counters.indexer_ran = true;
        for t in 0..s {
            let mut score = vec![0.0f32; n_comp];
            for (ci, sc) in score.iter_mut().enumerate() {
                // `einsum` -> bf16, `relu_()` in place -> bf16, `* weights` -> bf16 are all
                // ELEMENTWISE and genuinely land in bf16 (model.py:426-427); the final
                // `.sum(dim=2)` is a REDUCTION and accumulates in f32, rounding once. Those
                // two halves were conflated here until 2026-08-05 -- see `bf16_sum`, which
                // carries the measurement. This chain decides WHICH blocks are attended, so
                // a faithful kernel's bf16 and an f32 oracle can disagree on the SET near a
                // tie, which no numeric tolerance would show.
                *sc = self.bf16_sum((0..h).map(|hh| {
                    let qh = &q[(t * h + hh) * hd..(t * h + hh + 1) * hd];
                    let kvc = &cs.cache[ci * hd..(ci + 1) * hd];
                    // The einsum itself: f32 accumulation, one bf16 store, which is torch's
                    // bf16 matmul and was already right.
                    let mut dot =
                        bf16_decode(bf16_encode((0..hd).map(|i| qh[i] * kvc[i]).sum::<f32>()));
                    if self.defect != Defect::IndexerNoRelu {
                        dot = dot.max(0.0);
                    }
                    let wt = if self.defect == Defect::IndexerNoWeights {
                        1.0
                    } else {
                        w[t * h + hh]
                    };
                    bf16_decode(bf16_encode(dot * wt))
                }));
            }
            // Causal mask over compressed blocks, applied before topk (model.py:430-432)
            // and again to the SELECTED indices afterwards (:434-436) — the second pass is
            // what turns a fully-masked row's arbitrary topk into -1s.
            let limit = if start_pos == 0 {
                (t + 1) / ratio
            } else {
                n_comp
            };
            if start_pos == 0 {
                for (ci, sc) in score.iter_mut().enumerate() {
                    if ci >= limit {
                        *sc = f32::NEG_INFINITY;
                    }
                }
            }
            scores_out.extend_from_slice(&score);
            let k = c.index_topk.min(n_comp);
            counters.indexer_truncated += usize::from(k < n_comp);
            let sel = topk_idx(&score, k);
            out.push(
                sel.iter()
                    .map(|&i| {
                        if start_pos == 0 && i >= limit {
                            -1
                        } else {
                            (i + offset) as i64
                        }
                    })
                    .collect(),
            );
        }
        out
    }
}

// ---------------------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------------------

/// `[[f(t, j) for j in range(cols)] for t in range(seqlen)]` — the shape BOTH prefill
/// comprehensions in `model.py` have, and nothing else.
///
/// It owns the RECTANGULARITY that `attn::gather_slot_idxs` refuses a ragged buffer for, and no
/// selection logic: every index and every `-1` below stays with its own function, so a wrong
/// one cannot travel between [`window_topk`] and [`compress_topk`].
fn prefill_rows(seqlen: usize, cols: usize, f: impl Fn(usize, usize) -> i64) -> Vec<Vec<i64>> {
    (0..seqlen)
        .map(|t| (0..cols).map(|j| f(t, j)).collect())
        .collect()
}

/// `model.py::get_window_topk_idxs` — the sliding-window index list for each query row.
pub fn window_topk(win: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i64>> {
    if start_pos >= win.saturating_sub(1) && start_pos > 0 {
        let sp = start_pos % win;
        let mut v: Vec<i64> = ((sp + 1)..win).map(|i| i as i64).collect();
        v.extend((0..=sp).map(|i| i as i64));
        vec![v]
    } else if start_pos > 0 {
        let mut v: Vec<i64> = (0..=start_pos).map(|i| i as i64).collect();
        v.resize(win, -1);
        vec![v]
    } else {
        // `win == 0` gives `cols == 0`, so `win - 1` is never reached and this is `seqlen`
        // empty rows rather than the overflow panic a dev-profile build used to take. Not
        // reachable: `attn::gather_slot_idxs` returns early on `win == 0` and `sliding_window` is
        // 128 in the shipped config. Recorded because it is the one input on which this
        // spelling and the nested one it replaced differ.
        prefill_rows(seqlen, win.min(seqlen), |t, j| {
            let v = t.saturating_sub(win - 1) + j;
            if v > t { -1 } else { v as i64 }
        })
    }
}

/// `model.py::get_compress_topk_idxs` — the arithmetic (indexer-free) compressed selection
/// used where `compress_ratio != 4`.
/// `pub` for S2c: the compressor and indexer goldens are otherwise reachable only
/// through `run_layer`, which drags in the full MoE and 3.4 GB of experts per layer.
/// Driving these in isolation is what closes three measured coverage holes -- ratio-128
/// pooling (no golden at all at a 13-token prompt), its empty `[13,0]` selection
/// tensor, and the ranking, which `index_topk` never truncates below 2052 tokens.
/// Visibility only; no behaviour change.
pub fn compress_topk(
    ratio: usize,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
) -> Vec<Vec<i64>> {
    if start_pos > 0 {
        vec![
            (0..(start_pos + 1) / ratio)
                .map(|i| (i + offset) as i64)
                .collect(),
        ]
    } else {
        // `seqlen / ratio` is evaluated here rather than inside the row, so `ratio == 0`
        // divides by zero even at `seqlen == 0`, where the nested spelling returned `vec![]`.
        // `ratio == 0` already panicked for every other `seqlen` — on `(t + 1) / ratio` — so
        // it was never a supported input; only the empty case moved.
        prefill_rows(seqlen, seqlen / ratio, |t, c| {
            if c >= (t + 1) / ratio {
                -1
            } else {
                (c + offset) as i64
            }
        })
    }
}

impl Oracle {
    /// `kernel.py::sparse_attn`, as mathematics rather than as a tiling.
    ///
    /// `attn_sink` enters the softmax DENOMINATOR only: the kernel adds
    /// `exp(attn_sink[h] - running_max)` to `sum_exp` after the last block and never adds a
    /// matching term to `acc_o`. It is therefore a learned per-head leak of probability
    /// mass, not an extra key. Note that "sink as a real key with a zero value vector" is
    /// exactly this and NOT a defect, which is why no variant models it.
    fn sparse_attn(
        &self,
        q: &[f32],
        kv: &[f32],
        sink: &[f32],
        topk: &[Vec<i64>],
        m: usize,
        scale: f32,
    ) -> Vec<f32> {
        let (h, d) = (self.cfg.n_heads, self.cfg.head_dim);
        let mut o = vec![0.0f32; m * h * d];
        for t in 0..m {
            let idxs = &topk[t];
            for hh in 0..h {
                let qh = &q[(t * h + hh) * d..(t * h + hh + 1) * d];
                let mut logits = Vec::with_capacity(idxs.len());
                let mut mx = f32::NEG_INFINITY;
                for &ix in idxs {
                    if ix < 0 {
                        logits.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let k = &kv[ix as usize * d..(ix as usize + 1) * d];
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += qh[i] * k[i];
                    }
                    let l = acc * scale;
                    mx = mx.max(l);
                    logits.push(l);
                }
                // A fully-masked query would divide by a sum of `exp(-inf)` and write NaN
                // goldens. Loud, and `assert` because release builds skip debug assertions.
                assert!(mx.is_finite(), "query {t} head {hh} attends to nothing");
                let mut sum = 0.0f32;
                let mut acc = vec![0.0f32; d];
                for (&ix, &l) in idxs.iter().zip(&logits) {
                    if ix < 0 {
                        continue;
                    }
                    let e = (l - mx).exp();
                    sum += e;
                    let k = &kv[ix as usize * d..(ix as usize + 1) * d];
                    for i in 0..d {
                        acc[i] += e * k[i];
                    }
                }
                match self.defect {
                    Defect::SkipAttnSink => {}
                    Defect::AttnSinkNotMaxShifted => sum += sink[hh].exp(),
                    _ => sum += (sink[hh] - mx).exp(),
                }
                let dst = (t * h + hh) * d;
                for i in 0..d {
                    o[dst + i] = acc[i] / sum;
                }
            }
        }
        // `sparse_attn` writes a bf16 tensor.
        self.round_bf16(&mut o);
        o
    }

    /// `Attention.forward` (model.py:490-548).
    fn attention(
        &self,
        step: &LayerCtx,
        st: &mut LayerRings,
        x: &[f32],
        cap: &mut Capture,
    ) -> Vec<f32> {
        let LayerCtx {
            lw,
            layer,
            s,
            start_pos,
            ..
        } = *step;
        let tag = step.tag();
        let c = &self.cfg;
        let (win, ratio, rd, d, nh) = (
            c.window_size,
            c.compress_ratio(layer),
            c.rope_head_dim,
            c.head_dim,
            c.n_heads,
        );
        let freqs = self.freqs(layer);

        // -- q ---------------------------------------------------------------------------
        let mut qr = self.linear(x, s, c.dim, &lw.wq_a);
        self.round_bf16(&mut qr);
        self.rmsnorm(&mut qr, c.q_lora_rank, &lw.q_norm);
        let mut q = self.linear(&qr, s, c.q_lora_rank, &lw.wq_b);
        self.round_bf16(&mut q);
        // `Defect::QkNormAfterRope` is a SWAP and nothing else, so the rope loop is written
        // once and only the `qk_norm` call moves across it. Two transcriptions of the loop
        // are two places for the defect arm to stop being the clean arm's mirror image,
        // which would make the A/B measure something other than the swap.
        let after = self.defect == Defect::QkNormAfterRope;
        if !after {
            self.qk_norm(&mut q, d, &lw.q_norm);
        }
        for i in 0..s * nh {
            self.rope_row(
                &mut q[i * d..(i + 1) * d],
                rd,
                (start_pos + i / nh, freqs),
                false,
            );
        }
        if after {
            self.qk_norm(&mut q, d, &lw.q_norm);
        }
        cap.push(&format!("{tag}.q"), &[s, nh, d], q.clone());

        // -- kv --------------------------------------------------------------------------
        let mut kv = self.linear(x, s, c.dim, &lw.wkv);
        self.round_bf16(&mut kv);
        self.rmsnorm(&mut kv, d, &lw.kv_norm);
        for t in 0..s {
            self.rope_row(
                &mut kv[t * d..(t + 1) * d],
                rd,
                (start_pos + t, freqs),
                false,
            );
            self.kv_act_quant(&mut kv[t * d..(t + 1) * d], d, rd);
        }
        cap.push(&format!("{tag}.kv_entry"), &[s, d], kv.clone());

        // -- selection -------------------------------------------------------------------
        let mut topk = window_topk(win, s, start_pos);
        // `window_topk` yields one row per query: `s` at prefill, and exactly 1 at decode,
        // where `s` is also 1. So the query count is `s` and there is no second extent --
        // and this is what ENFORCES the bsz=1 scope cut rather than merely stating it.
        assert_eq!(topk.len(), s, "query count and row count disagree");
        if ratio != 0 {
            let offset = if start_pos == 0 { s } else { win };
            let extra = if let (Some(iw), Some(ics)) = (&lw.indexer, st.idx_comp.as_mut()) {
                let mut sel_scores = Vec::new();
                let e = self.indexer(
                    step,
                    iw,
                    ics,
                    x,
                    &qr,
                    offset,
                    freqs,
                    &mut cap.counters,
                    &mut sel_scores,
                );
                let rows = if sel_scores.is_empty() { 0 } else { s };
                let cols = sel_scores.len().checked_div(rows).unwrap_or(0);
                cap.push(&format!("{tag}.indexer_scores"), &[rows, cols], sel_scores);
                e
            } else {
                compress_topk(ratio, s, start_pos, offset)
            };
            let flat: Vec<i64> = extra.iter().flatten().copied().collect();
            let cols = extra.first().map_or(0, Vec::len);
            cap.push_i(&format!("{tag}.compress_idxs"), &[extra.len(), cols], flat);
            for (row, e) in topk.iter_mut().zip(extra) {
                row.extend(e);
            }
        }

        // -- cache, compression, attention -----------------------------------------------
        // The ring write differs by phase; the compressor call does not, and its ORDER does:
        // the reference writes the window entry first in both branches (model.py:523-537).
        if start_pos == 0 {
            if s <= win {
                st.win_cache[..s * d].copy_from_slice(&kv[..s * d]);
            } else if self.defect == Defect::PrefillRingWritesFirstWindow {
                st.win_cache[..win * d].copy_from_slice(&kv[..win * d]);
            } else {
                // slot (t % win) holds position t, for the last `win` positions.
                for t in (s - win)..s {
                    let slot = t % win;
                    st.win_cache[slot * d..(slot + 1) * d].copy_from_slice(&kv[t * d..(t + 1) * d]);
                }
            }
            cap.counters.prefill_evicted = s.saturating_sub(win);
        } else {
            let slot = start_pos % win;
            st.win_cache[slot * d..(slot + 1) * d].copy_from_slice(&kv[..d]);
        }

        let compressed = match (ratio, &lw.compressor, st.comp.as_mut()) {
            (0, _, _) => None,
            (_, Some(cw), Some(cs)) => {
                self.compressor(cw, cs, x, s, start_pos, freqs, &mut cap.counters)
            }
            _ => None,
        };
        if let Some(z) = &compressed {
            cap.push(&format!("{tag}.compressed"), &[z.len() / d, d], z.clone());
        }

        // What the attention reads. At prefill it is the whole prompt's KV with THIS call's
        // compressed rows appended at index `s` (which is why the offset above was `s`); at
        // decode it is the ring followed by the whole compressed region, which is the
        // reference's single `kv_cache` buffer split in two here.
        let full = if start_pos == 0 {
            let mut f = kv.clone();
            if let Some(z) = compressed {
                f.extend(z);
            }
            f
        } else {
            let mut f = st.win_cache.clone();
            if let Some(cs) = st.comp.as_ref() {
                f.extend_from_slice(&cs.cache);
            }
            f
        };
        let mut o = self.sparse_attn(&q, &full, &lw.attn_sink, &topk, s, (d as f32).powf(-0.5));

        // -- output ----------------------------------------------------------------------
        // Captured on BOTH sides of the de-rotation on purpose: without the pre-image, a
        // de-rotation defect has no golden that must stay identical and the check loses its
        // silent half.
        cap.push(&format!("{tag}.attn_core_out"), &[s, nh, d], o.clone());
        if self.defect != Defect::SkipOutputDerotation {
            let inverse = self.defect != Defect::OutputDerotationForward;
            for i in 0..s * nh {
                self.rope_row(
                    &mut o[i * d..(i + 1) * d],
                    rd,
                    (start_pos + i / nh, freqs),
                    inverse,
                );
            }
        }
        cap.push(&format!("{tag}.attn_derot"), &[s, nh, d], o.clone());

        let g = c.o_groups;
        let hpg = nh / g;
        let gd = nh * d / g;
        // `o.view(b, s, n_groups, -1)` over a head-major flattening: group `gi` is heads
        // [gi*hpg, (gi+1)*hpg), each contributing its whole head_dim.
        let mut og = vec![0.0f32; s * g * gd];
        for t in 0..s {
            for gi in 0..g {
                for e in 0..gd {
                    let (head, dim_i) = match self.defect {
                        Defect::WoGroupsSplitHeadDim => (e / (d / g), gi * (d / g) + e % (d / g)),
                        Defect::WoGroupsInterleaved => (gi + (e / d) * g, e % d),
                        _ => (gi * hpg + e / d, e % d),
                    };
                    og[(t * g + gi) * gd + e] = o[(t * nh + head) * d + dim_i];
                }
            }
        }
        let r = c.o_lora_rank;
        let mut y = vec![0.0f32; s * g * r];
        let mut wr = Vec::with_capacity(gd);
        for gi in 0..g {
            for ri in 0..r {
                self.wrow(&lw.wo_a, gi * r + ri, &mut wr);
                for t in 0..s {
                    let ov = &og[(t * g + gi) * gd..(t * g + gi + 1) * gd];
                    y[(t * g + gi) * r + ri] = ov.iter().zip(&wr).map(|(a, b)| a * b).sum::<f32>();
                }
            }
        }
        self.round_bf16(&mut y);
        let mut out = self.linear(&y, s, g * r, &lw.wo_b);
        self.round_bf16(&mut out);
        out
    }
}

// ---------------------------------------------------------------------------------------
// MoE
// ---------------------------------------------------------------------------------------

impl Oracle {
    /// `Gate.forward`. Returns `(weights, indices)`, both `[m, n_activated_experts]`.
    ///
    /// Two things here are easy to lose and impossible to see afterwards:
    /// - the load-balancing `bias` shifts the scores used for SELECTION and is absent from
    ///   the scores used as WEIGHTS (`original_scores`, model.py:577-585);
    /// - hash layers (`layer_id < n_hash_layers`) take their indices from
    ///   `tid2eid[input_id]` and bypass the scores entirely — but the gate still runs, and
    ///   its scores still become the weights.
    pub fn gate(
        &self,
        step: &LayerCtx,
        x: &[f32],
        counters: &mut Counters,
    ) -> (Vec<f32>, Vec<usize>) {
        let LayerCtx {
            lw,
            s: m,
            input_ids,
            ..
        } = *step;
        let c = &self.cfg;
        let k = c.n_activated_experts;
        let logits = self.linear(x, m, c.dim, &lw.gate_w);
        let n = c.n_routed_experts;
        let is_softmax = self.defect == Defect::RouterSoftmax;
        let mut original = vec![0.0f32; m * n];
        for t in 0..m {
            let row = &logits[t * n..(t + 1) * n];
            if is_softmax {
                let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut s = 0.0f32;
                for (i, &v) in row.iter().enumerate() {
                    let e = (v - mx).exp();
                    original[t * n + i] = e;
                    s += e;
                }
                for v in &mut original[t * n..(t + 1) * n] {
                    *v /= s;
                }
            } else {
                for (i, &v) in row.iter().enumerate() {
                    if v.exp().is_infinite() {
                        counters.softplus_overflows += 1;
                    }
                    let sp = if self.defect == Defect::RouterNoSoftplusThreshold {
                        (1.0 + v.exp()).ln()
                    } else {
                        softplus(v)
                    };
                    original[t * n + i] = sp.sqrt();
                }
            }
        }
        let mut selection = original.clone();
        if let Some(b) = &lw.gate_bias {
            for t in 0..m {
                for i in 0..n {
                    selection[t * n + i] += b[i];
                }
            }
        }

        let mut idx = vec![0usize; m * k];
        let mut wts = vec![0.0f32; m * k];
        for t in 0..m {
            let sel: Vec<usize> = match &lw.tid2eid {
                Some(map) if self.defect != Defect::HashRoutingIgnored => {
                    let base = input_ids[t] as usize * k;
                    map[base..base + k].iter().map(|&e| e as usize).collect()
                }
                _ => topk_idx(&selection[t * n..(t + 1) * n], k),
            };
            let src = if self.defect == Defect::RouterBiasedWeights {
                &selection
            } else {
                &original
            };
            for (j, &e) in sel.iter().enumerate() {
                idx[t * k + j] = e;
                wts[t * k + j] = src[t * n + e];
            }
            if !is_softmax && self.defect != Defect::RouterNoRenorm {
                let s: f32 = wts[t * k..(t + 1) * k].iter().sum();
                for v in &mut wts[t * k..(t + 1) * k] {
                    *v /= s;
                }
            }
            if self.defect != Defect::RouterNoScale {
                for v in &mut wts[t * k..(t + 1) * k] {
                    *v *= c.route_scale;
                }
            }
        }
        (wts, idx)
    }

    /// `Expert.forward` — SwiGLU with `swiglu_limit = 10.0`.
    ///
    /// The clamp is ASYMMETRIC: `up` is clamped to `[-limit, +limit]` and `gate` only from
    /// above (model.py:606-607). And the routing weight multiplies the SwiGLU intermediate,
    /// before the `.to(bf16)` that precedes `w2` — not the expert's output.
    pub fn expert(
        &self,
        e: &ExpertW,
        x: &[f32],
        m: usize,
        weight: Option<&[f32]>,
        counters: &mut Counters,
    ) -> Vec<f32> {
        let c = &self.cfg;
        let inter = c.moe_inter_dim;
        let mut g = self.linear(x, m, c.dim, &e.w1);
        let mut u = self.linear(x, m, c.dim, &e.w3);
        self.round_bf16(&mut g);
        self.round_bf16(&mut u);
        let limit = match self.defect {
            Defect::SwigluUnclamped => 0.0,
            _ => c.swiglu_limit,
        };
        let mut h = vec![0.0f32; m * inter];
        for i in 0..m * inter {
            let (mut gi, mut ui) = (g[i], u[i]);
            if limit > 0.0 {
                if ui < -limit || ui > limit {
                    counters.swiglu_clamp_events += 1;
                }
                if gi > limit || (self.defect == Defect::SwigluClampGateBothSides && gi < -limit) {
                    counters.swiglu_clamp_events += 1;
                }
                ui = ui.clamp(-limit, limit);
                gi = if self.defect == Defect::SwigluClampGateBothSides {
                    gi.clamp(-limit, limit)
                } else {
                    gi.min(limit)
                };
            }
            h[i] = silu(gi) * ui;
        }
        let apply_before = self.defect != Defect::RouteWeightAfterW2;
        if let Some(w) = weight
            && apply_before
        {
            for t in 0..m {
                for i in 0..inter {
                    h[t * inter + i] *= w[t];
                }
            }
        }
        self.round_bf16(&mut h);
        let mut out = self.linear(&h, m, inter, &e.w2);
        self.round_bf16(&mut out);
        if let Some(w) = weight
            && !apply_before
        {
            for t in 0..m {
                for i in 0..c.dim {
                    out[t * c.dim + i] *= w[t];
                }
            }
            self.round_bf16(&mut out);
        }
        out
    }

    /// `MoE.forward`. Accumulates in f32 in ASCENDING EXPERT ID, then adds the shared
    /// expert last — the reference's order, kept because it is free to keep and re-ordering
    /// a 7-term f32 sum is one more thing a consumer would have to allow for.
    fn moe(&self, step: &LayerCtx, x: &[f32], cap: &mut Capture) -> Vec<f32> {
        let LayerCtx { lw, s: m, .. } = *step;
        let tag = step.tag();
        let c = &self.cfg;
        let k = c.n_activated_experts;
        let (wts, idx) = self.gate(step, x, &mut cap.counters);
        cap.push(&format!("{tag}.router_weights"), &[m, k], wts.clone());
        cap.push_i(
            &format!("{tag}.router_indices"),
            &[m, k],
            idx.iter().map(|&i| i as i64).collect(),
        );

        let mut y = vec![0.0f32; m * c.dim];
        let mut by_expert: std::collections::BTreeMap<usize, Vec<(usize, f32)>> =
            Default::default();
        for t in 0..m {
            for j in 0..k {
                by_expert
                    .entry(idx[t * k + j])
                    .or_default()
                    .push((t, wts[t * k + j]));
            }
        }
        for (e, rows) in &by_expert {
            let Some(ew) = lw.experts.get(e) else {
                // The driver loads exactly the experts a run reaches; a miss means the
                // caller and the router disagree, which must not be papered over.
                panic!("expert {e} was routed to but not loaded");
            };
            let mut xs = Vec::with_capacity(rows.len() * c.dim);
            let mut ws = Vec::with_capacity(rows.len());
            for &(t, w) in rows {
                xs.extend_from_slice(&x[t * c.dim..(t + 1) * c.dim]);
                ws.push(w);
            }
            let o = self.expert(ew, &xs, rows.len(), Some(&ws), &mut cap.counters);
            for (r, &(t, _)) in rows.iter().enumerate() {
                for i in 0..c.dim {
                    y[t * c.dim + i] += o[r * c.dim + i];
                }
            }
        }
        let sw = if self.defect == Defect::SharedExpertWeighted {
            Some(vec![c.route_scale; m])
        } else {
            None
        };
        let sh = self.expert(&lw.shared, x, m, sw.as_deref(), &mut cap.counters);
        for i in 0..m * c.dim {
            y[i] += sh[i];
        }
        // `y.type_as(x)`.
        self.round_bf16(&mut y);
        y
    }
}

// ---------------------------------------------------------------------------------------
// drivers
// ---------------------------------------------------------------------------------------

/// `Block.hc_head`, the final `RMSNorm`, and `ParallelHead` — the head tail's weights.
///
/// **Transliterated 2026-08-05, but never driven from the layer chain, and that restriction
/// is the point.** The note this replaces refused to transliterate the head tail at all, on
/// the grounds that the goldens stop at layer 4 of 43 and so a logits vector taken there is
/// not any quantity the model computes. That argument is still correct and is preserved by
/// construction: `bin/v4-oracle` never composes [`Oracle::head_tail`] with `run_layer`. It
/// drives it from a declared synthetic probe instead, so the emitted golden cannot be
/// mistaken for the model's logits — its input is visibly not a residual stream — while still
/// exercising the real weights at the real `dim` and `vocab_size`, which is all the device
/// side needs to be scored against.
///
/// What that buys is the whole point of the exercise: before it, the first decode's logits
/// were **ungated by construction**. Every per-layer golden could be perfect and the sampled
/// token still wrong, with nothing in the tree able to say so.
///
/// What it still does NOT cover, stated so nobody has to infer it:
/// - **The composition.** No golden anywhere asserts that 43 layers followed by this head
///   tail produce a particular logits vector. Only S4, with a full-depth run, can.
/// - **Weight SELECTION.** These are arithmetic goldens over whatever this struct is handed.
///   A port that fed `layers.42.hc_ffn_fn` where `hc_head_fn` was due would reproduce every
///   golden here exactly. The loader is what has to get that right.
/// - **Sampling.** `sample(logits, temperature)` (model.py:924) is out of scope, as is
///   `forward_spec` and everything MTP.
///
/// A struct of its own rather than fields hung off the embedding, because the two are
/// separately loadable: `bin/v4-oracle defects` drives the head tail and never embeds
/// anything, and `embed.weight` is 2.1 GB once widened to f32 on a machine whose memory is
/// shared with a live decode.
pub struct HeadTailW {
    /// `hc_head_fn`, `[hc_mult, hc_mult * dim]`. F32 on disk — `Transformer.__init__` builds
    /// it under `with set_dtype(torch.float32)`, so unlike the block weights there is no
    /// quantization and no bf16 store anywhere in its use.
    pub hc_head_fn: Vec<f32>,
    /// `hc_head_base`, `[hc_mult]` — one bias per hyper-connection copy.
    pub hc_head_base: Vec<f32>,
    /// `hc_head_scale`, `[1]`. A single scalar broadcast over every mix, where a `Block`'s
    /// `hc_*_scale` is `[3]` (pre, post, comb). Reusing the block's layout here would index
    /// past the tensor rather than merely computing the wrong thing, which is the one mistake
    /// on this path that fails loudly.
    pub hc_head_scale: Vec<f32>,
    /// `norm.weight`, `[dim]` — the final `RMSNorm`'s learned gain.
    pub norm: Vec<f32>,
    /// `head.weight`, `[vocab_size, dim]`. bf16 in the checkpoint and held as f32 by
    /// `ParallelHead`, so it takes `linear()`'s dense branch: **no activation quantization**,
    /// and the logits come out f32 and are never rounded.
    pub lm_head: WMat,
}

impl Oracle {
    /// `Transformer.forward` lines 914-916: embed, then expand to `hc_mult` copies.
    /// Returns `[s, hc_mult, dim]`.
    pub fn embed(&self, embed: &WMat, ids: &[u32]) -> Vec<f32> {
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let mut row = Vec::with_capacity(dim);
        let mut out = Vec::with_capacity(ids.len() * hc * dim);
        for &t in ids {
            embed.row(t as usize, &mut row);
            for _ in 0..hc {
                out.extend_from_slice(&row);
            }
        }
        out
    }

    /// `Block.hc_head` (model.py:709-716). `h` is `[s, hc_mult, dim]`; the result is
    /// `[s, dim]`, bf16 as `y.to(dtype)` leaves it.
    ///
    /// This is `hc_pre`'s *pre* branch and nothing else: no Sinkhorn, no `post`, no
    /// combination matrix. It cannot be reached by reusing `hc_split_sinkhorn` even by
    /// accident — that wants `(2 + hc) * hc = 24` mixes and `hc_head_fn` yields `hc = 4` —
    /// which is why there is no defect for "ran the Sinkhorn here".
    ///
    /// Called as `layer.hc_head(...)` on the LAST block, so `norm_eps`/`hc_eps` are that
    /// block's. They are `args.norm_eps`/`args.hc_eps`, identical to the Transformer's, so
    /// reading them from `cfg` is exact rather than approximate.
    fn hc_head(&self, hw: &HeadTailW, h: &[f32], s: usize) -> Vec<f32> {
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let hcd = hc * dim;
        assert_eq!(h.len(), s * hcd);
        assert_eq!(hw.hc_head_fn.len(), hc * hcd);
        assert_eq!(hw.hc_head_base.len(), hc);
        // `[1]`, not `[3]`: read the shape rather than trusting the caller to have loaded the
        // right tensor. `hc_attn_scale` has the same name shape and three entries, and
        // indexing [0] of it would be silently plausible.
        assert_eq!(
            hw.hc_head_scale.len(),
            1,
            "hc_head_scale is a scalar, not a Block's [3]"
        );
        let mut y = vec![0.0f32; s * dim];
        for t in 0..s {
            let flat = &h[t * hcd..(t + 1) * hcd];
            // `torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)` over the FULL
            // flattened row. One statistic for every mix; the per-copy variant below is the
            // wrong version, kept so the gate can be shown to reject it.
            let rs: Vec<f32> = match self.defect {
                Defect::HeadHcNoRsqrt => vec![1.0; hc],
                Defect::HeadHcRsqrtPerCopy => (0..hc)
                    .map(|c| {
                        let seg = &flat[c * dim..(c + 1) * dim];
                        let var = seg.iter().map(|v| v * v).sum::<f32>() / dim as f32;
                        (var + self.cfg.norm_eps).sqrt().recip()
                    })
                    .collect(),
                _ => {
                    let var = flat.iter().map(|v| v * v).sum::<f32>() / hcd as f32;
                    vec![(var + self.cfg.norm_eps).sqrt().recip(); hc]
                }
            };
            let mut pre = vec![0.0f32; hc];
            for (j, p) in pre.iter_mut().enumerate() {
                let w = &hw.hc_head_fn[j * hcd..(j + 1) * hcd];
                let m = flat.iter().zip(w).map(|(a, b)| a * b).sum::<f32>() * rs[j];
                *p = sigmoid(m * hw.hc_head_scale[0] + hw.hc_head_base[j]) + self.cfg.hc_eps;
            }
            self.hc_blend(&pre, flat, &mut y[t * dim..(t + 1) * dim]);
        }
        // `y.to(dtype)` — the residual stream this came from is bf16.
        self.round_bf16(&mut y);
        y
    }

    /// The whole head tail: `hc_head`, the final `RMSNorm`, then `ParallelHead`
    /// (model.py:922-923). Returns the `[vocab_size]` logits and records three goldens under
    /// `head.{step_tag}.`.
    ///
    /// It does NOT record its input. The caller owns that: in `bin/v4-oracle` the input is a
    /// declared probe and is pushed there as `head.probe.in`, and in the defect matrix it is
    /// already recorded as the layer's `.out`. Recording it here as well would put a second
    /// copy under a name ending in `.in`, which the matrix treats as fixed-by-construction —
    /// a silent claim no implementation could violate would then be attached to a tensor that
    /// every upstream defect moves.
    pub fn head_tail(
        &self,
        hw: &HeadTailW,
        h: &[f32],
        s: usize,
        step_tag: &str,
        cap: &mut Capture,
    ) {
        let (dim, vocab) = (self.cfg.dim, self.cfg.vocab_size);
        // Both `[s, dim]` goldens below go through here. Two spelled-out `cap.push` calls is
        // what they were until the `step_tag` rename pushed each past `max_width` and rustfmt
        // reflowed them into a literal 27-token clone — the manufactured-duplication case
        // CLAUDE.md warns about. One closure states the prefix and the shape once.
        let mut record =
            |name: &str, v: Vec<f32>| cap.push(&format!("head.{step_tag}.{name}"), &[s, dim], v);
        let mut x = self.hc_head(hw, h, s);
        record("hc_head_out", x.clone());

        if self.defect != Defect::HeadNormSkipped {
            match self.defect {
                // A single-row norm kernel handed `s x dim`: one statistic over everything,
                // the learned gain still landing per dim because it repeats every `dim`.
                // Tiling the weight is how that is expressed with one norm routine rather
                // than a second copy of the loop.
                Defect::HeadNormOverAllTokens => {
                    let tiled: Vec<f32> = (0..s).flat_map(|_| hw.norm.iter().copied()).collect();
                    self.rmsnorm_raw(&mut x, s * dim, &tiled);
                }
                _ => self.rmsnorm_raw(&mut x, dim, &hw.norm),
            }
            if self.defect != Defect::HeadNormNotBf16 {
                self.round_bf16(&mut x);
            }
        }
        // `final_norm_out`, not `norm_out`: the matrix selects goldens by NAME SUFFIX, and
        // `.norm_out` would sit one character away from matching `.attn_norm_out` and
        // `.ffn_norm_out` for anyone who later writes the suffix without the leading dot.
        record("final_norm_out", x.clone());

        // `ParallelHead.forward` with `full_logits=False`: `x[:, -1]` — the LAST row only.
        let row = if self.defect == Defect::HeadLogitsFromFirstRow {
            0
        } else {
            s - 1
        };
        // `F.linear(x.float(), self.weight)` on an f32 parameter: the dense branch, so no
        // activation quantization, and the result is f32 and stays f32. Rounding it to bf16
        // here would be a defect all of its own; there is no store in the reference to model.
        let logits = self.linear(&x[row * dim..(row + 1) * dim], 1, dim, &hw.lm_head);
        // Recorded, not returned. Both callers read their logits back out of `cap`, and a
        // return value would offer a second path to the number the goldens are the record of.
        cap.push(&format!("head.{step_tag}.logits"), &[1, vocab], logits);
    }

    /// `Block.forward` (model.py:695-707) for one layer, in place on the residual stream.
    pub fn run_layer(
        &self,
        step: &LayerCtx,
        st: &mut LayerRings,
        h: &mut Vec<f32>,
        cap: &mut Capture,
    ) {
        let LayerCtx { lw, s, .. } = *step;
        let tag = step.tag();
        cap.push(
            &format!("{tag}.in"),
            &[s, self.cfg.hc_mult, self.cfg.dim],
            h.clone(),
        );

        let residual = h.clone();
        let (mut x, mix) = self.hc_pre(h, s, &lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base);
        self.rmsnorm(&mut x, self.cfg.dim, &lw.attn_norm);
        cap.push(
            &format!("{tag}.attn_norm_out"),
            &[s, self.cfg.dim],
            x.clone(),
        );
        let a = self.attention(step, st, &x, cap);
        cap.push(&format!("{tag}.attn_out"), &[s, self.cfg.dim], a.clone());
        *h = self.hc_post(&a, &residual, &mix, s);

        let residual = h.clone();
        let (mut x, mix) = self.hc_pre(h, s, &lw.hc_ffn_fn, &lw.hc_ffn_scale, &lw.hc_ffn_base);
        self.rmsnorm(&mut x, self.cfg.dim, &lw.ffn_norm);
        cap.push(
            &format!("{tag}.ffn_norm_out"),
            &[s, self.cfg.dim],
            x.clone(),
        );
        let f = self.moe(step, &x, cap);
        cap.push(&format!("{tag}.ffn_out"), &[s, self.cfg.dim], f.clone());
        *h = self.hc_post(&f, &residual, &mix, s);

        cap.push(
            &format!("{tag}.out"),
            &[s, self.cfg.hc_mult, self.cfg.dim],
            h.clone(),
        );
    }
}
