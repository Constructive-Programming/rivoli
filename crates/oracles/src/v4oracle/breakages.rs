//! The deliberate-breakage set the V4 oracle is proved against: [`Defect`], the one
//! declaration `Defect::ALL` is generated from, and the split-k fold that
//! `Defect::SplitKFoldOrder` selects.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut is by COHESION: nothing here reads a weight, a
//! cache or a config — the enum is a closed vocabulary that every other file in
//! `v4oracle/` matches on, and the three fold functions are `SplitKFoldOrder`'s own
//! arithmetic, spec'd in one place and shared with the device test. `forward.rs` re-exports
//! all four public items at their original paths, so `v4oracle::forward::{Defect,
//! splitk_fold, splitk_combine, wave_ladder}` still resolves.
//!
//! Every body moved unchanged. This is a frozen transliteration — see `forward.rs`'s module
//! doc for what is reproduced exactly, what is reproduced only up to summation order, and
//! what is out of scope; all of it still governs this file.

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
    /// and `kernels/linalg.hip::rmsnorm_single` does not.
    HeadNormNotBf16,
    /// Take the final `RMSNorm`'s statistic JOINTLY over all `s * dim` values instead of per
    /// token: what handing a single-row norm kernel an `s x dim` buffer does. Invisible at
    /// decode (`s == 1`), which is the only shape most smoke tests run.
    ///
    /// Only the STATISTIC is modelled. `kernels/linalg.hip::rmsnorm_single` would also read its
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
