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
//!
//! # What lives next door
//!
//! What is left here is the SHARED core — the [`Oracle`] itself, the primitives more than one
//! stage calls ([`Oracle::linear`], the two norms, the QK-norm, `rope_row`, the bf16 stores,
//! the mHC blend, the two softmaxes and the top-k), the two RoPE tables, and
//! [`Oracle::run_layer`], which is `Block.forward` and therefore the thing that calls all of
//! it. The stages themselves are siblings:
//!
//! | module | what moved |
//! |---|---|
//! | [`crate::v4oracle::breakages`] | [`Defect`], its one-declaration `ALL`, and the split-k fold |
//! | [`crate::v4oracle::layer`] | [`LayerW`] and friends, [`LayerCtx`], [`LayerRings`], [`CompState`] |
//! | [`crate::v4oracle::capture`] | [`Capture`], [`Counters`], the duplicate-refusing append |
//! | [`crate::v4oracle::hc`] | `hc_pre`, `hc_post`, `hc_split_sinkhorn` |
//! | [`crate::v4oracle::attention`] | `Attention.forward`, its two index-list constructors, the online softmax |
//! | [`crate::v4oracle::compress`] | `Compressor.forward` and `Indexer.forward` |
//! | [`crate::v4oracle::router`] | `Gate.forward` — scoring, the selection bias, the hash bypass |
//! | [`crate::v4oracle::moe`] | `Expert.forward` and `MoE.forward` |
//! | [`crate::v4oracle::head`] | the embedding, `hc_head`, the final norm, `ParallelHead` |
//!
//! **The reason is mechanical, not conceptual, and it happened in two passes.** On
//! 2026-08-15 the attention block went first: this file had 1068 code lines against
//! CodeScene's file-size cliff measured at ~880 and scored 9.09, and
//! `crates/cli/tests/codescene.rs` gates the whole tree at 10/10. The rest followed the same
//! day under the 800-line hard gate (`crates/cli/tests/line_limit.rs`). Both cuts ran along
//! seams that already existed — `run_layer` calls each stage exactly once and threads
//! `Sinks`, `HcW`/`HcMix` and `LayerCtx` — so what widened was visibility, not API:
//! `Oracle::{wrow, round_bf16, rmsnorm, rmsnorm_raw, rope_row, hc_blend}`, `softmax_strided`
//! and `topk_idx` are `pub(super)`.
//!
//! **What the CodeScene gate still reports here, and why it is not fixable in this module.**
//! `Oracle::{linear, run_layer}` (5 arguments each), `Oracle::head_tail` and
//! `Oracle::expert` (6 each) sit above the Rust threshold of 4, so `forward.rs`, `head.rs`
//! and `moe.rs` score 9.68 rather than 10.0. Each has an obvious bundling — `linear` already
//! asserts `w.cols() == k`, so `k` is redundant; `run_layer`'s `st`/`cap` pair is exactly
//! [`Sinks`], which `attention` already takes — and none of them can be applied from inside
//! this crate's `src/`: the arity is pinned by call sites in `crates/oracles/tests/{v4_oracle,
//! common/oracle_probe}.rs` and `crates/engine/tests/headtail.rs`. Whoever moves those four
//! call sites gets the last 0.32 back; splitting the file cannot.
//!
//! **Every public item is re-exported below at its original path**, so
//! `v4oracle::forward::{Capture, Defect, HeadTailW, LayerCtx, LayerW, Oracle, splitk_fold,
//! window_topk, …}` still resolves and no caller outside this module changed. Every body
//! moved VERBATIM: this is a frozen transliteration, the arithmetic is `model.py`'s, and
//! tidying anything on the way across would be a value change wearing a refactor's clothes.

use crate::v4oracle::attention::Sinks;
use crate::v4oracle::hc::HcW;
use crate::v4oracle::numerics::{act_quant_inplace, bf16_decode, bf16_encode};
use crate::v4oracle::weights::{V4Config, WMat};

/// The oracle's public surface, kept at the paths it has always had.
///
/// `tests/v4_oracle.rs` names these by full path (`v4oracle::forward::window_topk`), and so
/// do `tests/common/oracle_probe.rs`, `crates/engine/tests/headtail.rs`, `golden.rs` and
/// `toy.rs`. The re-export is what makes the 2026-08-15 file split invisible to every caller
/// rather than a rename with extra steps.
pub use crate::v4oracle::attention::{compress_topk, window_topk};
pub use crate::v4oracle::breakages::{Defect, splitk_combine, splitk_fold, wave_ladder};
pub use crate::v4oracle::capture::{Capture, Counters};
pub use crate::v4oracle::head::HeadTailW;
pub use crate::v4oracle::layer::{
    CompState, CompressorW, ExpertW, IndexerW, LayerCtx, LayerRings, LayerW,
};

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

/// One GEMV's extents: `[m, k]` activations against an `[n, k]` weight.
///
/// One parameter rather than three because the split-k dispatch predicate reads all three
/// together and a swapped pair does not fail — it silently selects a DIFFERENT set of
/// kernels, which is the whole hazard [`Oracle::splitk_selects`] exists to pin.
#[derive(Clone, Copy)]
struct GemvShape {
    m: usize,
    n: usize,
    k: usize,
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

    pub(super) fn round_bf16(&self, v: &mut [f32]) {
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
    pub(super) fn wrow(&self, w: &WMat, r: usize, buf: &mut Vec<f32>) {
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
    fn splitk_selects(&self, sh: GemvShape, w: &WMat) -> bool {
        self.defect == Defect::SplitKFoldOrder
            && matches!(w, WMat::Fp8 { .. })
            && sh.m * sh.n <= 2048
            && sh.k >= 4096
            && sh.k.is_multiple_of(4)
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
        let splitk = self.splitk_selects(GemvShape { m, n, k }, w);

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
    pub(super) fn rmsnorm(&self, x: &mut [f32], d: usize, w: &[f32]) {
        self.rmsnorm_raw(x, d, w);
        self.round_bf16(x);
    }

    /// `RMSNorm.forward` without that store.
    ///
    /// Split out for the head tail alone, which needs both halves independently: the store is
    /// what `Defect::HeadNormNotBf16` suppresses, and `d` is what `HeadNormOverAllTokens`
    /// widens to `s * dim`. Both are choices a device implementation actually faces —
    /// `kernels/mla.hip::rmsnorm_batch` bf16-rounds and is one block per row, while
    /// `kernels/linalg.hip::rmsnorm_single` neither rounds nor spans rows.
    pub(super) fn rmsnorm_raw(&self, x: &mut [f32], d: usize, w: &[f32]) {
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
    pub(super) fn rope_row(
        &self,
        row: &mut [f32],
        rd: usize,
        f: (usize, &[(f32, f32)]),
        inverse: bool,
    ) {
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
    /// It lives HERE, with `linear` and the two norms, for the same reason `softmax_strided`
    /// does: a primitive whose callers are in two different modules (`hc::hc_pre` and
    /// `head::hc_head` since the 2026-08-15 split) belongs to the shared root, not to
    /// whichever of the two happens to be read first.
    ///
    /// Each output element accumulates across the copies in copy order, the order `hc_pre`
    /// had before the extraction -- which the diff shows directly, the body being the same
    /// fold with `out.iter_mut().enumerate()` in place of `for d in 0..dim`. So no golden
    /// moved, by inspection rather than by measurement. A 200-trial bitwise run at
    /// `hc = 4, dim = 257`, with gate weights and residual copies spread across 2^-20..2^10
    /// so that cancellation was available, agreed at **0 bitwise differences**, and reported 184
    /// differences under a one-ulp control -- but that only confirmed that an identical fold
    /// is identical.
    pub(super) fn hc_blend(&self, pre: &[f32], flat: &[f32], out: &mut [f32]) {
        let dim = self.cfg.dim;
        for (d, o) in out.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (c, &p) in pre.iter().enumerate() {
                acc += p * flat[c * dim + d];
            }
            *o = acc;
        }
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
// reductions shared by the block body
// ---------------------------------------------------------------------------------------

/// Softmax over `n` elements strided by `stride`, in fp32, `-inf`-safe.
pub(super) fn softmax_strided(v: &mut [f32], n: usize, stride: usize, offset: usize) {
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
pub(super) fn topk_idx(v: &[f32], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    // `total_cmp`, not `partial_cmp`: mapping NaN to `Equal` is not a total order and
    // Rust >= 1.81's `sort_by` may panic on one. Scores cannot be NaN today, but a sort
    // comparator is not the place to rely on that.
    order.sort_by(|&a, &b| v[b].total_cmp(&v[a]).then(a.cmp(&b)));
    order.truncate(k);
    order
}

// ---------------------------------------------------------------------------------------
// Block.forward
// ---------------------------------------------------------------------------------------

impl Oracle {
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
        let (hc, dim) = (self.cfg.hc_mult, self.cfg.dim);
        let mut sk = Sinks { st, cap };
        sk.cap.push(&format!("{tag}.in"), &[s, hc, dim], h.clone());

        let residual = h.clone();
        let attn_hc = HcW {
            fnw: &lw.hc_attn_fn,
            scale: &lw.hc_attn_scale,
            base: &lw.hc_attn_base,
        };
        let (mut x, mix) = self.hc_pre(h, s, attn_hc);
        self.rmsnorm(&mut x, dim, &lw.attn_norm);
        sk.cap
            .push(&format!("{tag}.attn_norm_out"), &[s, dim], x.clone());
        let a = self.attention(step, &mut sk, &x);
        sk.cap
            .push(&format!("{tag}.attn_out"), &[s, dim], a.clone());
        *h = self.hc_post(&a, &residual, &mix);

        let residual = h.clone();
        let ffn_hc = HcW {
            fnw: &lw.hc_ffn_fn,
            scale: &lw.hc_ffn_scale,
            base: &lw.hc_ffn_base,
        };
        let (mut x, mix) = self.hc_pre(h, s, ffn_hc);
        self.rmsnorm(&mut x, dim, &lw.ffn_norm);
        sk.cap
            .push(&format!("{tag}.ffn_norm_out"), &[s, dim], x.clone());
        let f = self.moe(step, &x, sk.cap);
        sk.cap.push(&format!("{tag}.ffn_out"), &[s, dim], f.clone());
        *h = self.hc_post(&f, &residual, &mix);

        sk.cap.push(&format!("{tag}.out"), &[s, hc, dim], h.clone());
    }
}
