//! The V4 layer loop, scored against `bin/v4-oracle`'s **real-weight** per-layer goldens.
//!
//! This is the gate. `src/v4gpu.rs`'s own `mod tests` covers the placement and fill rules the
//! port has measured to be invisible to any numeric comparison; this file covers the arithmetic,
//! at real dims, against the reference — and it is the first thing in this port to score the
//! whole of `Block.forward` (hyper-connections, attention, router, MoE) rather than attention
//! alone.
//!
//! # What it needs, and why it skips rather than fails
//!
//! * `/var/db/rivoli/v4-f4-l0-2` (12 GB) — layers 0-2, both ratio-0 layers plus one ratio-4.
//! * `/var/db/rivoli/v4-f4-l3-5` — the only fixture whose range does not start at 0, which is
//!   what makes the layer-0 refusal testable at all.
//! * `/var/db/rivoli/v4-goldens-l2.bin` — `v4-oracle emit --layers 2 --decode-steps 1` over
//!   `/var/db/rivoli/deepseek-v4-flash-0731`. **Regenerate it with that exact command**; the
//!   binary needs no feature and touches no GPU, so it can run beside a held device.
//!
//! An explicitly-set env var that does not resolve is a FAILURE, never a skip, for the reason
//! `tests/common/v4_artifact_dir.rs` states: libtest captures stderr on passing tests, so an
//! `eprintln!` skip is invisible in a green run.
//!
//! # Why layers 0 and 1 and not layer 2
//!
//! Layer 2 is `compress_ratio == 4`, which is the one class this loop CANNOT be scored on.
//! `Oracle::attention` selects `lw.indexer` there and `topk_idx` returns **score-ordered** rows,
//! while the engine selects blocks positionally. Below 2052 positions the SET is identical —
//! fixed by the causal mask, not by any score — but `sparse_attn` folds an online softmax over
//! the rows IN THE ORDER GIVEN, so every disagreement on a ratio-4 layer mixes a real defect with
//! a deliberate fold-order difference and is uninterpretable. That is
//! `docs/investigations/v4-flash-port.md` §"The pre-indexer shortcut is narrower than it sounds"
//! arriving at its consequence, and it is why the goldens are emitted at `--layers 2`.
//!
//! Layers 0 and 1 are also both **hash-routed** (`n_hash_layers == 3`), so they exercise the
//! `tid2eid` path and not the scored one. The scored router is uncovered here and says so below.
//!
//! # PREDICTED BEFORE MEASURING — stated here so it cannot be fitted afterwards
//!
//! **PARTLY SUPERSEDED 2026-08-05. Read `# CORRECTED 2026-08-05` below before relying on any row
//! of the table in this section.** The prediction is kept because it was made before the
//! measurement and must stay legible as what was believed; three of its cells are now known false
//! and each is marked. This section is the belief; that one is the measurement.
//!
//! The port's own prediction for one ratio-0 layer at real dims is "≥99% of elements
//! bit-identical; the remainder at exactly 1 bf16 ULP; none above 2", from tree-vs-sequential
//! re-association surviving a bf16 re-truncation at every stage boundary. That prediction was
//! about ATTENTION. This loop adds three things it did not cover, so the prediction splits:
//!
//! **Three comparisons exist, and the table says which.** The engine's two sublayer outputs both
//! land in one scratch buffer that `hc_post` consumes, so `attn_out` and `ffn_out` are not
//! separately readable — the BLOCK output is. An earlier draft of this table listed bounds for all
//! seven goldens the oracle emits, four of which this file never reads: a gate claimed, not built.
//!
//! | golden | compared? | predicted |
//! |---|---|---|
//! | `L{l}.{tag}.out` | **yes**, bound 5e-2 | most elements bit-identical in bf16; `max_rel` well under the bound |
//! | `L{l}.{tag}.router_{indices,weights}` | **yes**, set-equal + bound 1e-2 | **near-exact** — `sqrt(softplus(·))` on a dense f32 GEMV, renormalised, so `max_rel <= 1e-3` |
//! | `head.probe.logits` | **yes**, bound 5e-2 | tightest of the three: three ops on a declared probe, no MoE anywhere |
//! | `attn_norm_out`, `attn_out`, `ffn_norm_out`, `ffn_out` | ~~**no** — emitted, not readable per-sublayer~~ **WRONG WHEN WRITTEN: `attn_norm_out` and `ffn_norm_out` were already compared. CORRECTED 2026-08-05: `attn_out` is read too; only `ffn_out` is not** | `ffn_out` would be worst, by a named cause: the shared expert is unclamped, one contribution in seven |
//! | `head.probe.{hc_head_out,final_norm_out}` | **no** — a gap, not a decision | a `hc_head`-vs-final-norm swap inside `head_tail` is visible only through the logits |
//!
//! And the sharpest prediction, which one probe settles: **the missing clamp is INERT at this
//! prompt.** `swiglu_limit` is 10.0, `ffn_norm_out` ranges within ±1.1, and the two fp8
//! projections are 4096-wide reductions of that — so `max|gate|` and `max|up|` are predicted
//! **under 10**, which would mean the whole deviation reduces to `F.silu`'s multiply form plus
//! one missing bf16 round of the product, and `ffn_out` should land at the tight end of its band.
//! If either operand exceeds 10 the clamp binds, the deviation is unbounded, and `ffn_out`'s
//! number says nothing about the rest of the loop. Those are very different findings and no
//! golden distinguishes them, because a golden only ever sees the sum of seven contributions.
//!
//! **The 5e-2 bound is a CHOSEN separation, not a measured one, and the numbers behind it were taken
//! elsewhere.** The port's seen-red record — `rel 4.2e-1` and `6.7e-1` on `attn_derot`, `1.06` on
//! `q` — is `tests/v4_attn.rs`'s attention cell, on tensors this file did not read when this was
//! written (CORRECTED 2026-08-05: it reads both now). No break has
//! ever been measured through `check` on `L*.out`. Those figures are the right order of magnitude to
//! calibrate against and they do not transfer directly, for a reason stated below: `hc_post` mixes
//! the sublayer output with four residual copies, so a sublayer error is DILUTED in `.out`. So 5e-2
//! is chosen to sit far above re-association and far below a wiring error, and the tight numbers are
//! reported rather than asserted. Injecting a break through THIS gate and recording what it measures
//! is the work that would make the bound measured.
//!
//! # CORRECTED 2026-08-05 — the attention half is bisected, and `attn_out` cannot carry the reading put on it
//!
//! `V4Engine::probe_attn_stages` reads `q`, `kv_entry`, `attn_derot` and `attn_out` out of the
//! scratch buffers ONE `attention` call leaves them in, so the attention half is bisected at four
//! points instead of bounded at one. Three of those four are new here; `attn_out` replaces the
//! single-tensor probe this file used before.
//!
//! ## The mechanism: three fp8 activation requantizations, each a ~16x amplifier
//!
//! The block performs **three** fp8 ACTIVATION requantizations — `act_quant(xq)`,
//! `act_quant(qrq)`, `act_quant(y)` — each a 4-significant-bit step function sitting immediately
//! downstream of a bf16 store. A bf16 ULP is 2^-8..2^-7 relative and an e4m3 step is 2^-4..2^-3,
//! depending on where in the binade the value sits — **16x larger at the same point**. So the
//! ordinary output of a re-associated reduction flips a quantization bin on a few percent of
//! elements, each flip moves that element 16x further than the difference that caused it, and
//! every downstream tensor is a dense reduction over the quantized vector — so ONE flip perturbs
//! ALL of them.
//!
//! Measured by transcribing this file's own pipeline to numpy and perturbing one stage. The
//! transcription IS the experiment — which ops, in which order, with the perturbation injected
//! where — so it is committed rather than described: **`docs/measurement/probes/v4_attn_amplification.py`**
//! prints every number in this section, in ~8 s, touching no GPU.
//!
//! | perturbation | effect |
//! |---|---|
//! | **1** of 13,312 `qrq` elements moved ONE e4m3 step | `q` differs on **4.6%** of 425,984 elements |
//! | **1** of 32,768 `attn_derot` elements moved 1 bf16 ULP (`L1.dec0`) | `attn_out` differs on **21%**, `max_rel` **0.71** |
//! | f32 vs **f64** accumulation, identical semantics, same weights | `attn_out` `max_rel` **1.35** on `L0.pre` |
//!
//! That last row is the floor on any bound here: two implementations differing ONLY in accumulator
//! precision — the mildest legitimate difference there is — land 1.35 apart, 27x outside the 5e-2
//! this file asserts. Isolating the ops shows `act_quant(y)` is the dominant amplifier of the
//! three (46% vs 20% of `attn_out` differing, same perturbation, with and without it).
//!
//! ## So the clean/dirty ORDERING is set by act_quant placement, not by proximity to a defect
//!
//! | tensor | act_quants upstream | host sim vs oracle, 4 cells |
//! |---|---|---|
//! | `attn_norm_out` | 0 | engine measures 0.05% — the ONLY tensor in the block with none |
//! | `kv_entry` | 1 (`xq`) + its own partial block-64 | **bit-identical on all four** |
//! | `q`, `attn_derot` | 2 (`xq`, `qrq`) | 0% – 4.2%, p99.9 <= 25 ULP |
//! | `attn_out` | 3 (`xq`, `qrq`, `y`) | 0% – **21%** |
//!
//! **What this retracts.** `docs/investigations/v4-flash-port.md` reads its bisection table as
//! "`attn_out` is the first bad tensor … between the clean tensor and the wrong one there are
//! exactly two ops", and concludes `hc_pre` and both `RMSNorm`s are therefore confirmed. That
//! inference does not hold: `attn_norm_out` is clean because it is the one tensor upstream of
//! every amplifier, not because everything before it is right. `router_weights` and
//! `head.probe.logits` are clean for the same structural reason and not because the head is
//! better ported — the gate is `linear(x.float(), w.float())` with no activation quantization
//! anywhere, and the head tail runs on a declared probe through an int8 per-row `lm_head`.
//! The ordering in that table is what the block's arithmetic produces for ANY implementation.
//!
//! That doc also reads `attn_out`'s `max_abs 7.81e-2` as "~20 ULP … which is not re-association".
//! **That is a unit error** — it divides `max_abs` by the bf16 ULP at an element other than the one
//! it occurred on. `Gap::abs_ulp` now MEASURES it: **10.0**, at an element with |x| in [1, 2). Do
//! not trust the earlier corrections either, including the `2.5` this header asserted for one
//! revision and the `1.25` implied by the tensor's true maximum — all three were derived by
//! assuming a magnitude, and all three were wrong. 10 ULP is squarely re-association here: a bare
//! numpy-vs-oracle fold difference on this same tensor reaches p99.9 = 17.8.
//!
//! ## MEASURED ON THE DEVICE 2026-08-06 — and it retracted the residual AND my own discriminator
//!
//! An earlier version of this section claimed a RESIDUAL: that no single perturbation of
//! `attn_derot` reproduced the engine's three `attn_out` statistics at once, so its error tail was
//! heavier than amplification explained. **That was a straw man and it is withdrawn.** The sweep
//! behind it varied ONE parameter — the fraction of perturbed elements — at a fixed ±1 ULP
//! magnitude, while this same header records real fold-order noise as heavy-tailed (median 1 ULP,
//! p99 21, max 326). Rejecting a null model already known to be the wrong shape produces no
//! residual. Review caught it; a 2-D sweep brackets all three statistics at once.
//!
//! It also said: "**any movement at all in `kv_entry` … is a real defect**". **That was wrong, and
//! the device would have been convicted by it.** It was written from a host sim driven by the
//! oracle's EXACT `attn_norm_out`, where `kv_entry` is bit-identical on all four cells. The engine
//! does not have an exact input: its `attn_norm_out` differs on 26/53,248 elements (all at exactly
//! 1 ULP — clean re-association in `hc_pre` + `RMSNorm`), and `act_quant(xq)` amplifies that.
//! A criterion that assumes a perfect input cannot judge an implementation that has a real one.
//!
//! What the device says, layer 0 prefill, against the goldens:
//!
//! | tensor | differ | max_abs | max_rel | reading |
//! |---|---:|---:|---:|---|
//! | `attn_norm_out` | 0.05% | 4.9e-4 | 7.1e-3 | every difference exactly 1 ULP, `>1ULP` = **0** |
//! | `kv_entry` | 0.69% | 6.3e-2 | 9.5e-1 | |
//! | `q` | 6.69% | 9.4e-2 | 2.7e1 | |
//! | `attn_derot` | 14.48% | 6.3e-2 | 6.9e0 | |
//! | `attn_out` | 57.92% | 7.8e-2 | 3.5e1 | |
//!
//! **Feeding the device's own measured input deviation into the transcription reproduces its
//! output deviation, jointly, on both layers.** Perturbing the golden `attn_norm_out` by exactly
//! the deviation the device has — 26 elements at 1 ULP on L0, 132 (8 of them 2 ULP) on L1 — and
//! running the block, over 40 seeds:
//!
//! | cell | device `kv_entry` | sim p10/median/p90 | device `q` | sim p10/median/p90 | device percentile |
//! |---|---:|---|---:|---|---|
//! | L0.pre | 0.69% | 0.00 / 0.62 / 1.49 | 6.69% | 0.00 / 6.17 / 13.68 | **55th, 62nd** |
//! | L1.pre | 1.58% | 1.38 / 2.62 / 3.99 | 23.25% | 26.02 / 32.86 / 46.00 | **15th, 5th** |
//!
//! The perturbation SIZE is the device's own measurement, not a fitted parameter; only the seed
//! varies. The device lands mid-distribution on L0 and in the LOW tail on L1 — i.e. **closer to
//! the oracle than a typical correct implementation with its input deviation**. There is no
//! coordinate in which it is worse than the model predicts, which is the opposite of a defect
//! signature.
//!
//! The decode cells corroborate: at `L1.dec0` the engine's `attn_norm_out`, `kv_entry` and `q` are
//! all **bit-identical** to the oracle, and `attn_derot` still differs on 14% — because the ring
//! it attends was written by the PREFILL, whose `kv_entry` differed. Inherited, not independent.
//!
//! **Conclusion, and it is narrower than the first draft of this line claimed.** The 5e-2 bound
//! was unmeetable and is gone. The engine's deviation is CONSISTENT with one ULP of
//! `attn_norm_out` re-association amplified by three fp8 requantizations, and adversarial review
//! could not construct a defect from `Defect::ALL` consistent with the device's differing-fraction
//! vector — every one it tried lands at percentile 0 or 100 against the device. That is real
//! evidence and it is stronger than what the assertions below encode.
//!
//! **It is not "no defect is visible", which this said for one revision.** Two gaps stop it:
//!
//! * The envelope draws its perturbed element POSITIONS uniformly at random. The device's 26 are
//!   wherever `hc_pre`'s fold order crossed a bf16 boundary, and their indices were never
//!   recorded. Clustering them into one token row instead inverts the §9 percentile result — the
//!   device goes from mid-distribution to worse than every sample on 4 of 4 coordinates. Which
//!   model is right is UNMEASURED, and it is cheap to settle: `probe_pre_norm` already returns
//!   `attn_norm_out`, so dumping the differing INDICES and driving the envelope from them would
//!   replace a distribution-over-an-assumption with one deterministic number.
//! * The input deviation is ATTRIBUTED to re-association from its magnitude. A systematic ~1e-6
//!   relative error in the pre-norm path — a hardware `rsqrtf` where `kernels/mla.hip:331` argues
//!   for `1.0f/sqrtf`, or eps outside the sqrt — has the same 26-elements-at-1-ULP signature. The
//!   discriminator is sign-correlation within a row (systematic) versus sign-random
//!   (re-association), and it was not run.
//!
//! ## The bounds below are DERIVED, and the derivation is runnable
//!
//! `AttnStages::scored` carries the four bounds, the table they come from, and — read this before
//! trusting them — how little two of them buy. Each is the geometric mean of an ENVELOPE and the
//! weakest DEFECT above it, measured over the in-scope subset of `Defect::ALL`. The separations
//! are **45x (`kv_entry`), 30x (`q`), 1.3x (`attn_derot`), 1.6x (`attn_out`)**: the first two are
//! real gates, the last two barely gate at all, and **seven of eighteen in-scope defects cleared
//! the looser first-draft bounds simultaneously**. `QkNormAfterRope` moves `attn_out` LESS than
//! the device does, so no bound on that tensor can separate it at any value.
//!
//! **`max_rel` is the wrong statistic and these bounds are the best it admits.** It is
//! floor-dominated and near-blind to SCALING defects — `SkipQkNorm` roughly doubles every element
//! of `q` and reads 1.07 against a 275 bound. The statistic that separated every defect review
//! could construct is the differing-element FRACTION, and nothing here asserts it. That is the
//! single highest-value thing owed on this file: `Gap` already computes `bf16_differing` and
//! `check` already prints it.
//!
//! Two things that derivation is NOT. It is calibrated through the HOST TRANSCRIPTION, not through
//! `check` — so it inherits the transcription's fidelity, and the probe says plainly that it
//! models `src/attn.rs` rather than the kernels. And the envelope depends on today's
//! `attn_norm_out` deviation; if `hc_pre` or `v4_rmsnorm` changes, re-run §8 rather than trusting
//! these four numbers.
//!
//! ## What stays RED, and why that is the honest state
//!
//! `ffn_norm_out` and `.out` keep the **underived 5e-2** and this file stays red on them. They are
//! downstream of the same amplifier — `.out` is `attn_out` diluted through `hc_post` with four
//! residual copies, then the whole MoE — so the bound is wrong for them too, in the same
//! direction and for the same reason. It is not moved, because measuring their envelope needs
//! `hc_pre`/`hc_post` and the MoE transcribed, and neither is in this track's file set. **Red on
//! an underived bound is honest; green on a widened one is not**, and that applies as much to the
//! two tensors I could not calibrate as to the four I could.
//!
//! **Owed, tracked nowhere else:** `src/bin/v4-oracle.rs`'s `emit()` hardcodes `Defect::None`. A
//! `--defect` flag would put a perturbed golden one command away and let the four bounds above be
//! re-derived through THIS gate rather than through a host transcription — which is the one
//! weakness the derivation still has. `Defect::ALL` already enumerates the breakages.

// `rocm`: `v4gpu` is `rocm`-gated because every launcher it drives is `backend::hip`'s, and
// since 2026-08-06 that is the only backend. The rule this used to cite by name
// (`tests/kernel_coverage.rs`, deleted with the Vulkan backend) is worth stating directly,
// because it outlived its enforcer: do not add stubs that claim a parity nothing measured.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::artifact::model::{V4Config, load_config};
use rivoli::math::f32_to_bf16;
use rivoli::memory::pin::V4Pin;
use rivoli::v4gpu::V4Engine;
use rivoli::v4oracle::golden::GoldenSet;

#[path = "common/v4_artifact_dir.rs"]
mod v4_artifact_dir;
use v4_artifact_dir::{v4_artifact, v4_artifact_l3_5};

/// Device budget for the fixture pins.
///
/// **The same 5 GiB `tests/v4_pool.rs` uses, and the first value here was wrong in the direction
/// that made the whole gate red.** Measured on the shipped fixture: `resident.safetensors` is
/// 2,607,031,354 B = **2.428 GiB**, so `pool_budget` leaves `CAPACITY - 2.428 GiB - 16 MiB` against
/// a `3 x 256 x 13,369,344 B` = **9.56 GiB** routed set. At the 8 GiB this first said, the pool is
/// 5.56 GiB and `budget * 2 = 11.11 GiB > 9.56` — so the oversubscription assertion below FIRED and
/// nothing after it ran. At 5 GiB the pool is 2.56 GiB (205 slots) and it passes.
///
/// The comment this replaces said "resident set 2.5 GB ... under 60% of the set" while asserting
/// `budget * 2 < routed`, i.e. under 50% — a GB/GiB mix beside a bound that did not match the prose.
/// That is precisely the failure `tests/v4_pool.rs` records ("a 0.06% margin read as 8%"),
/// reproduced in the file that cites it. Numbers restated in one unit.
///
/// Oversubscription is the point: a budget large enough to hold everything would make every lookup a
/// HIT and the streaming path would be untested by the only test that drives the real router. It is
/// asserted rather than assumed, because a fixture that grew would silently turn this into an
/// all-resident run. Note what 205 slots against ~156 distinct lookups does NOT buy: no eviction.
/// That is `tests/v4_pool.rs`'s case, not this one; here the prefill misses and the decode hits.
const CAPACITY: usize = 5 << 30;

/// Where `v4-oracle emit --layers 2 --decode-steps 1` put its output.
const GOLDENS_ENV: &str = "RIVOLI_V4_GOLDENS";
const GOLDENS_DEFAULT: &str = "/var/db/rivoli/v4-goldens-l2.bin";

/// How far apart two tensors are, in the two units that mean different things.
///
/// `max_rel` is what a wiring error moves (the port's seen-red record is 4.2e-1 to 1.06).
/// `bf16_differing` is what re-association moves: every stage boundary in this pipeline
/// re-truncates to bf16, so an f32 delta only survives the store when the value sits within a
/// bf16 rounding boundary — which is why a count of differing bf16 CODES is the sensitive
/// instrument and the relative error is the decisive one.
#[derive(Debug)]
struct Gap {
    max_rel: f32,
    max_abs: f32,
    bf16_differing: usize,
    /// The largest difference expressed in ULP **of the reference element it is on**.
    ///
    /// **Added 2026-08-05 because the other three units all mislead here, in opposite
    /// directions.** `max_abs` is taken wherever the tensor is largest, so dividing it by the ULP
    /// at a TYPICAL element inflates it — on `attn_out`, whose values run 1.2 median to 13.1 max,
    /// `attn_out`, whose values run 1.2 median to 13.1 max, `max_abs 7.8e-2` reads as "~10 ULP"
    /// against the median. **CORRECTED 2026-08-06: this said "and is 1.25 ULP at the element it
    /// actually occurred on". That was another assumed magnitude — see [`Gap::abs_ulp`], which
    /// measures it at 10.0.** And `max_rel` carries a 1e-3 floor, so a one-ULP move on a near-zero
    /// element manufactures a large ratio: the host sim's worst `attn_out` `max_rel` of 2.4e-1 sits
    /// on an element of magnitude 6.0e-4. This is the unit that is the same size everywhere,
    /// which is what makes "1" mean the same thing across every tensor this file compares.
    max_ulp: f32,
    /// Differences strictly larger than one ULP. Re-association plus a bf16 store produces ones;
    /// a wiring error produces thousands. The COUNT is what separates them, not the max.
    above_1ulp: usize,
    /// `max_abs` expressed in ULP **of the element `max_abs` occurred on**.
    ///
    /// **This is the exact quantity `docs/investigations/v4-flash-port.md` got wrong**, and it is
    /// printed because "print what you cite" is the rule that failure broke. That doc divided
    /// `max_abs 7.81e-2` by a bf16 ULP taken at some other element, read 20.0, and concluded
    /// "which is not re-association".
    ///
    /// **Three people then produced three different corrections, all by ASSUMING which element
    /// `max_abs` sat on, and all wrong.** 20.0 is the ULP at |x| = 0.6; 2.5 is the ULP at |x| = 7.7
    /// (the tensor's quoted range, and the figure this comment itself asserted for one revision);
    /// 1.25 is the ULP at |x| = 13.1 (its true max). **Measured, it is 10.0** — `max_abs` occurs at
    /// an element with |x| in [1, 2), and 10 ULP is squarely inside re-association here (a bare
    /// numpy-vs-oracle fold difference on this tensor reaches p99.9 = 17.8). Four derivations,
    /// one measurement, and the measurement agreed with none of them. That is the entire argument
    /// for computing this in the instrument instead of in a comment.
    ///
    /// Distinct from [`Gap::max_ulp`], which maximises `d / ulp(b)` over ALL elements and is
    /// therefore dominated by near-zero references sitting on `REL_FLOOR` — it reads ~5,000 on the
    /// same tensor. Three different numbers, all defensible, none interchangeable: this is the one
    /// that answers "is the biggest absolute difference large for where it is?".
    abs_ulp: f32,
    total: usize,
    nonfinite: usize,
}

/// One bf16 ULP at `v`'s magnitude. bf16 keeps 7 explicit mantissa bits, so it is
/// `2^(floor(log2|v|) - 7)`.
///
/// **Floored at the SAME `REL_FLOOR` [`gap`] uses on `max_rel`, and for the same reason.** This
/// value is a DIVISOR, so an unfloored near-zero reference does not merely produce a large ratio,
/// it produces a meaningless one: at `v = 0` the natural guard (`f32::MIN_POSITIVE`, 2^-126) turns
/// any ordinary difference into ~1e37 ULP, which saturates `max_ulp` and increments `above_1ulp`.
/// A first version did exactly that. It is reachable — `Oracle::kv_act_quant` sends any element
/// under ~2^-9 of its block amax to exactly 0 — and it would have made the one metric introduced
/// here to be comparable across tensors the one metric that is not.
fn bf16_ulp(v: f32) -> f32 {
    (v.abs().max(REL_FLOOR).log2().floor() - 7.0).exp2()
}

/// The magnitude below which a reference element is treated as being AT this magnitude, by both
/// scale-relative metrics. One constant because the two must agree: a `max_rel` floored at 1e-3
/// beside a `max_ulp` floored somewhere else would disagree about which elements are near zero.
const REL_FLOOR: f32 = 1e-3;

fn gap(got: &[f32], want: &[f32]) -> Gap {
    assert_eq!(got.len(), want.len(), "comparing tensors of different lengths");
    let mut g = Gap {
        max_rel: 0.0,
        max_abs: 0.0,
        bf16_differing: 0,
        max_ulp: 0.0,
        above_1ulp: 0,
        abs_ulp: 0.0,
        total: got.len(),
        nonfinite: 0,
    };
    for (&a, &b) in got.iter().zip(want) {
        if !a.is_finite() {
            g.nonfinite += 1;
            continue;
        }
        if f32_to_bf16(a) != f32_to_bf16(b) {
            g.bf16_differing += 1;
        }
        let d = (a - b).abs();
        if d > g.max_abs {
            g.max_abs = d;
            g.abs_ulp = d / bf16_ulp(b);
        }
        let u = d / bf16_ulp(b);
        g.max_ulp = g.max_ulp.max(u);
        if u > 1.0 {
            g.above_1ulp += 1;
        }
        // Relative to the REFERENCE's magnitude, with a floor so a near-zero element cannot
        // manufacture a huge ratio out of a one-ulp absolute difference.
        let scale = b.abs().max(REL_FLOOR);
        g.max_rel = g.max_rel.max(d / scale);
    }
    g
}

/// Every comparison this run made, so ONE window produces the whole picture.
///
/// The first gate run panicked on the FIRST tensor and told me nothing about the other five — on a
/// device held exclusively and queued for. Collecting and failing once at the end costs nothing and
/// is the difference between one datum and a bisection. That the run needed a second window to
/// learn what a single one could have is the process finding, not the numeric one.
static FAILURES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Print one comparison, and record a failure if `bound` is `Some` and breached. See [`FAILURES`].
///
/// `bound: Option<f32>` rather than two functions, and `None` is not a weaker gate — it is the
/// honest state of the three attention-stage tensors, whose separation nothing in this tree has
/// measured. See the header's "What is deliberately NOT done here". Non-finiteness is a failure
/// either way: an fp8 GEMV over a 4096-wide reduction that overflows is a defect, not a magnitude,
/// and that judgement needs no calibration.
fn check(what: &str, got: &[f32], want: &[f32], bound: Option<f32>) {
    let g = gap(got, want);
    // Marked on the LINE and not only in this file's header: a reader scanning the run's output
    // for what went red must be able to see which rows cannot. Stated as a property of the row
    // and not as a count, because `check` does not know how many rows a run prints and a comment
    // that says "three of five" here rots the moment a caller adds a fourth unbounded one.
    let unbounded = match bound {
        Some(_) => "",
        None => "  [reported, no bound]",
    };
    eprintln!(
        "  {what:<32} ULP max {:>9.1}  >1ULP {:>7}  bf16 differ {:>7}/{:<7} ({:5.2}%)  \
         max_abs {:.3e} ({:.1} ULP there)  max_rel {:.3e}{unbounded}",
        g.max_ulp,
        g.above_1ulp,
        g.bf16_differing,
        g.total,
        100.0 * g.bf16_differing as f64 / g.total as f64,
        g.max_abs,
        g.abs_ulp,
        g.max_rel,
    );
    let mut f = FAILURES.lock().expect("poisoned");
    if g.nonfinite > 0 {
        f.push(format!("{what}: {} non-finite — a defect, not a tolerance", g.nonfinite));
    }
    // The breached bound is named in the message: two different ones are live in a run (5e-2 on
    // five tensors, 1e-2 on `router_weights`), and `FAILURES` is the whole record of a run on a
    // device that was queued for. `filter` and not `is_some_and`, so `b` survives into the format.
    if let Some(b) = bound.filter(|b| g.max_rel >= *b) {
        // The bound is CHOSEN, not measured through this gate — see the header.
        f.push(format!(
            "{what}: max_rel {:.3e} >= the {b:.0e} separation ({g:?})",
            g.max_rel
        ));
    }
}

/// Fail once, with everything.
fn report() {
    let f = FAILURES.lock().expect("poisoned");
    assert!(f.is_empty(), "{} comparison(s) outside bound:\n  {}", f.len(), f.join("\n  "));
}

/// One golden tensor by name. Fails loudly on a miss: a typo'd name silently comparing nothing
/// is how a gate passes vacuously.
fn golden_i64<'g>(gs: &'g GoldenSet, name: &str) -> &'g [i64] {
    gs.ints
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, _, v)| v.as_slice())
        .unwrap_or_else(|| panic!("no int golden named {name:?}"))
}

fn golden<'g>(gs: &'g GoldenSet, name: &str) -> &'g [f32] {
    gs.floats
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, _, v)| v.as_slice())
        .unwrap_or_else(|| {
            let mut have: Vec<&str> = gs.floats.iter().map(|(n, _, _)| n.as_str()).collect();
            have.sort_unstable();
            panic!("no golden named {name:?}. The file holds: {have:?}")
        })
}

fn meta<'g>(gs: &'g GoldenSet, key: &str) -> &'g str {
    gs.meta
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("the golden file carries no {key:?} in its metadata"))
}

/// The prompt ids the goldens were taken at, out of the metadata rather than re-tokenized.
///
/// Re-tokenizing would be a second implementation of the thing that has to match exactly: the
/// hash-routed layers index `tid2eid` BY TOKEN ID, so one different id routes to different
/// experts and every FFN golden moves for a reason that is not the engine's.
fn prompt_ids(gs: &GoldenSet) -> Vec<u32> {
    let raw = meta(gs, "prompt_ids");
    let inner = raw
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .unwrap_or_else(|| panic!("prompt_ids metadata is not a debug-formatted list: {raw:?}"));
    inner
        .split(',')
        .map(|s| s.trim().parse::<u32>().expect("a token id"))
        .collect()
}

fn open_goldens() -> Option<GoldenSet> {
    let named = std::env::var(GOLDENS_ENV).ok();
    let path = named.clone().unwrap_or_else(|| GOLDENS_DEFAULT.to_string());
    match std::fs::File::open(&path) {
        Ok(mut f) => Some(GoldenSet::read(&mut f).expect("reading the golden file")),
        Err(e) => {
            // An explicitly-set path that does not resolve is a failure, not a skip.
            assert!(named.is_none(), "{GOLDENS_ENV}={path} does not open ({e}) — refusing to pass by skipping");
            eprintln!(
                "SKIP: no goldens at {path}. Generate with:\n  \
                 cargo run --release --bin v4-oracle -- emit --model \
                 /var/db/rivoli/deepseek-v4-flash-0731 --out {path} --layers 2 --decode-steps 1"
            );
            None
        }
    }
}

/// **Criterion 5, and it has no other test anywhere.**
///
/// `V4Pin::build` deliberately does NOT refuse an artifact that starts past layer 0 — which
/// layers a file holds is a property of the loader, and refusing it there made every partial
/// artifact but the first unloadable. A DECODE is the thing that cannot start at layer 3, because
/// there is no residual stream to enter the model with. `V4Pin::layer` takes ABSOLUTE ids, so a
/// pin over 3..6 answers every lookup correctly and the arithmetic is a different model's, with
/// nothing anywhere to notice.
///
/// Runs FIRST and drops its pin, so it does not hold the device while the real work runs.
fn a_pin_that_does_not_start_at_layer_zero_cannot_decode() -> bool {
    let Some(dir) = v4_artifact_l3_5("resident.safetensors") else {
        return false;
    };
    let cfg: V4Config = load_config(&dir).expect("l3-5 config");
    let pin = V4Pin::build(&dir, &cfg, CAPACITY, "2q", Default::default(), None)
        .expect("the LOADER must accept a partial artifact — that is the whole point of the split");
    assert_eq!(pin.range(), 3..6, "the fixture whose range does not start at 0");
    // `map(|_| ())` because `V4Engine` is not `Debug` and `expect_err` wants it to be. Turning
    // the Ok side into `()` is the narrow fix; deriving `Debug` on an engine holding 30 raw
    // device pointers would print addresses and imply they were inspectable.
    let err = V4Engine::new(pin, cfg, 8, 4)
        .map(|_| ())
        .expect_err("a decode must refuse a pin that starts at layer 3")
        .to_string();
    assert!(
        err.contains("must start at layer 0"),
        "refused for the wrong reason: {err}"
    );
    true
}

/// The context ceiling, refused at startup rather than 41 layers into some later token.
fn a_context_past_the_indexers_reach_is_refused_at_startup(dir: &str, cfg: &V4Config) {
    let pin = V4Pin::build(dir, cfg, CAPACITY, "2q", Default::default(), None).expect("pin");
    // 2052 at the shipped `index_topk = 512`. `prompt + ngen + 1` is the context, so this asks
    // for exactly one position too many.
    let over = 4 * (cfg.index_topk + 1);
    let err = V4Engine::new(pin, cfg.clone(), 8, over)
        .map(|_| ())
        .expect_err("a context past the positional selection's reach must be refused")
        .to_string();
    assert!(
        err.contains("lightning indexer") && err.contains("POSITIONALLY"),
        "refused for the wrong reason: {err}"
    );
}

/// Every layer of the goldens, both phases, on the reference's own input.
///
/// The phase order is the ORACLE's — all layers of `pre`, then all layers of `dec0` — and it is
/// not cosmetic: each layer's KV cache and pooling state carry across phases, so running layer 0's
/// decode before layer 1's prefill would attend a ring the prefill had not written.
fn every_layer_matches_the_oracle_at_real_weights(
    dir: &str,
    cfg: &V4Config,
    gs: &GoldenSet,
    ids: &[u32],
) {
    let layers: usize = meta(gs, "layers").parse().expect("layers metadata");
    let steps: usize = meta(gs, "decode_steps").parse().expect("decode_steps metadata");
    assert!(layers >= 2, "the goldens must cover at least the two ratio-0 layers, not {layers}");
    eprintln!(
        "goldens: {} tokens, {layers} layers, {steps} decode step(s); model {}",
        ids.len(),
        meta(gs, "model")
    );

    let pin = V4Pin::build(dir, cfg, CAPACITY, "2q", Default::default(), None).expect("pin");
    // The oversubscription this test's value depends on, asserted rather than assumed.
    let routed = (pin.range().len() * cfg.n_experts) as u64
        * rivoli::artifact::quant::f4_expert_stride(cfg.hidden, cfg.moe_inter) as u64;
    let budget = pin.routed.budget() as u64;
    assert!(
        budget * 2 < routed,
        "the pool ({budget} B) is not meaningfully oversubscribed against the routed set \
         ({routed} B), so this test would never take a miss and the streaming path is uncovered"
    );
    // Read off the PIN, before it is moved into the engine — the engine needs no accessor for
    // something its caller already had.
    let held = pin.range();
    let mut e = V4Engine::new(pin, cfg.clone(), ids.len(), steps + 1).expect("engine");
    // Only the layers the goldens cover AND the pin holds, and only the ratio-0 ones — see the
    // header on why layer 2 is uninterpretable.
    let scored: Vec<usize> = (0..layers)
        .filter(|&l| held.contains(&l))
        .filter(|&l| cfg.compress_ratio(l).expect("ratio") == 0)
        .collect();
    assert!(
        !scored.is_empty(),
        "no ratio-0 layer is both in the goldens and in the pin — nothing would be compared"
    );
    eprintln!("scoring layers {scored:?} (ratio-0 only, hash-routed)");

    let last = *ids.last().expect("a prompt");
    for phase in 0..=steps {
        let (tag, here, start_pos): (String, Vec<u32>, usize) = if phase == 0 {
            ("pre".into(), ids.to_vec(), 0)
        } else {
            // Each decode step re-feeds the prompt's LAST token, exactly as `drive` does. Not a
            // claim about what the model would generate: it makes the capture a well-defined
            // function of the prompt and the cached state.
            (format!("dec{}", phase - 1), vec![last], ids.len() + phase - 1)
        };
        for &l in &scored {
            let m = here.len();
            eprintln!("L{l}.{tag}  ({m} row(s) at start_pos {start_pos})");
            // The reference's own residual for this layer and phase — NOT the engine's
            // accumulated one, so each layer's number is that layer's.
            e.set_residual(golden(gs, &format!("L{l}.{tag}.in"))).expect("set residual");
            // The EARLIEST comparable tensor: `hc_pre` + `attn_norm`, before attention runs.
            // Idempotent, so `probe_layer` below re-runs it harmlessly. This is what says whether
            // the hyper-connection prologue or the attention block owns the error.
            let anp = e.probe_pre_norm(l, false, &here, start_pos).expect("pre_norm probe");
            check(
                &format!("L{l}.{tag}.attn_norm_out"),
                &anp,
                golden(gs, &format!("L{l}.{tag}.attn_norm_out")),
                Some(5e-2),
            );
            // The attention half, bisected at every point ONE call leaves readable. Safe to
            // re-run here because these are ratio-0 layers — the probe refuses a compressing one
            // rather than trusting the caller. Sharpest-first, NOT pipeline order: see the
            // header's ladder, and believe the first line that moves. `attn_core_out` is not
            // readable at all; `AttnStages` says why.
            let st = e
                .probe_attn_stages(l, &here, start_pos)
                .expect("attn stages probe");
            for (name, got, bound) in st.scored() {
                let w = format!("L{l}.{tag}.{name}");
                check(&w, got, golden(gs, &w), Some(bound));
            }
            e.probe_layer(l, &here, start_pos).expect("probe_layer");
            // **The bisection.** `xw` still holds this layer's `ffn_norm_out` — `moe` quantizes a
            // COPY into `xq`. Everything up to and including attention, `hc_post`, the second
            // `hc_pre` and its norm is upstream of it; the whole MoE is downstream. Compared before
            // `.out` so the report reads in pipeline order.
            let fno = e.probe_working(m).expect("working readback");
            check(
                &format!("L{l}.{tag}.ffn_norm_out"),
                &fno,
                golden(gs, &format!("L{l}.{tag}.ffn_norm_out")),
                Some(5e-2),
            );
            let got = e.residual(m).expect("residual readback");
            // CORRECTED 2026-08-05: the two stages are not separately readable in ONE pass —
            // `attention`'s output and the MoE's land in the same scratch buffer that the next
            // `hc_post` consumes — so the block's OUTPUT is what this line compares. The ATTENTION
            // half is now read above, by re-running it alone (`probe_attn_stages`); the MoE half
            // still is not, so `.out` remains the only comparison that sees it, and remains unable
            // to attribute WHICH half moved. The 4.2e-1 the port measured for a dropped persist
            // copy was measured on `attn_derot` in `tests/v4_attn.rs` — a tensor this file now
            // reads too, though at real weights rather than toy dims.
            let w = format!("L{l}.{tag}.out");
            check(&w, &got, golden(gs, &w), Some(5e-2));

            // **The router, which the block output dilutes past any useful bound.** `hc_post`
            // mixes the FFN output with four residual copies, so a wrong routing weight — the
            // `Defect::RouterBiasedWeights` case, taking weights from the bias-shifted scores —
            // sits under `check`'s 5e-2 on `.out`. Compared here directly, and as a SET plus a
            // per-expert weight, because `route_into` sorts its picks by score while `tid2eid`
            // yields them in table order: the multiset is what both agree on.
            let (ids_got, w_got) = e.probe_route();
            let ri = golden_i64(gs, &format!("L{l}.{tag}.router_indices"));
            let rw = golden(gs, &format!("L{l}.{tag}.router_weights"));
            let k = ids_got.len();
            // The LAST row, which is what `probe_route` holds — stated rather than assumed.
            let (want_i, want_w) = (&ri[(m - 1) * k..m * k], &rw[(m - 1) * k..m * k]);
            let mut pairs: Vec<(usize, f32)> = ids_got.iter().copied().zip(w_got).collect();
            let mut want: Vec<(usize, f32)> =
                want_i.iter().map(|&i| i as usize).zip(want_w.iter().copied()).collect();
            pairs.sort_by_key(|&(e, _)| e);
            want.sort_by_key(|&(e, _)| e);
            let got_ids: Vec<usize> = pairs.iter().map(|&(e, _)| e).collect();
            let want_ids: Vec<usize> = want.iter().map(|&(e, _)| e).collect();
            assert_eq!(
                got_ids, want_ids,
                "L{l}.{tag}: the router selected different experts than the reference. On a hash \
                 layer that is a tid2eid indexing bug; on a scored one it is the selection scores."
            );
            let wg: Vec<f32> = pairs.iter().map(|&(_, w)| w).collect();
            let ww: Vec<f32> = want.iter().map(|&(_, w)| w).collect();
            check(&format!("L{l}.{tag}.router_weights"), &wg, &ww, Some(1e-2));

            // The clamp question, on this layer's own FFN input. Must come immediately after the
            // layer: it reads `xq`, which still holds this layer's quantized `ffn_norm` output.
            let (g, u) = e.probe_shared_operands(l, m).expect("shared operands");
            let mx = |v: &[f32]| v.iter().fold(0.0f32, |a, &x| a.max(x.abs()));
            let limit = cfg.swiglu_limit as f32;
            let (mg, mu) = (mx(&g), mx(&u));
            eprintln!(
                "    shared SwiGLU operands: max|gate| {mg:.3}  max|up| {mu:.3}  vs swiglu_limit \
                 {limit} -> clamp {}",
                if mg.max(mu) > limit { "BINDS" } else { "inert" }
            );
            // Finiteness IS a gate — an fp8 GEMV over a 4096-wide reduction that overflows is a
            // defect, not a magnitude. Whether the clamp BINDS is deliberately NOT asserted: the
            // header predicts inert, and turning a prediction into an assertion would make a
            // correct engine go red for telling me my prediction was wrong. The answer is
            // reported and recorded in `docs/investigations/v4-flash-port.md`.
            assert!(
                mg.is_finite() && mu.is_finite(),
                "L{l}.{tag}: the shared expert's SwiGLU operands are not finite \
                 (max|gate| {mg}, max|up| {mu})"
            );
        }
    }

    // A miss actually happened. `budget * 2 < routed` proves the pool is OVERSUBSCRIBED, not
    // that any lookup missed — and a run that happened to hit throughout would leave
    // `miss_stream`, `wait_on` and the second accumulator row untested while reporting green.
    let (hits, misses) = (e.pool_hits(), e.pool_misses());
    eprintln!("pool: {hits} hit / {misses} miss");
    assert!(
        misses > 0,
        "no expert missed, so the streaming path (miss_stream, wait_on, acc row 1) ran zero times \
         and this test's green says nothing about it"
    );
    assert!(hits > 0, "no expert hit, so the resident path ran zero times");

    // --- the head tail, on the oracle's DECLARED probe ---------------------------------
    //
    // This is the hole `docs/investigations/v4-flash-port.md` names as structural: "the last
    // three ops of `Transformer.forward` have neither an implementation nor a golden … the first
    // decode's logits are ungated by construction". Both exist now. The probe is deliberately NOT
    // the layer chain's residual — composing 2 layers of 43 would produce a logits vector that is
    // not any quantity the model computes, and `fixed_probe`'s own doc says a tensor named
    // `logits` sitting beside real per-layer goldens is the most misusable thing that file could
    // write. The head tail is a pure function of its input, so the golden gates the same
    // arithmetic either way.
    let probe = golden(gs, "head.probe.in");
    let row = cfg.hc_mult * cfg.hidden;
    let s = probe.len() / row;
    eprintln!("head.probe ({s} rows)");
    e.set_residual(probe).expect("set probe residual");
    let logits = e.probe_head_tail(s).expect("head tail");
    // `ParallelHead` slices `x[:, -1]` after the norm, so only row `s - 1`'s logits exist.
    check(
        "head.probe.logits",
        &logits,
        golden(gs, "head.probe.logits"),
        Some(5e-2),
    );
}

/// **ONE `#[test]`, and that is not tidiness.**
///
/// libtest runs `#[test]` fns on parallel threads, and each case here builds a `V4Pin` — a
/// `DeviceTier` allocation, a pool VMM and an io_uring ring. Run in parallel on 2026-08-05 that
/// **wedged the device**: 19 threads, four `io_sq_thread`s (the tell — four rings means four
/// pools), zero output in 12 minutes, killed by PID. `--test-threads=1` fixes it and is the wrong
/// fix, because it lives in whoever remembers to type it and the cost of forgetting is a wedged
/// sole-tenant GPU. `tests/v4_pin.rs` and `tests/v4_pool.rs` both made this call; this follows
/// them.
///
/// **Order is load-bearing.** The layer-0 refusal runs first and drops its pin, so it does not
/// hold a second tier while the real work runs.
#[test]
fn the_v4_layer_loop() {
    let refused = a_pin_that_does_not_start_at_layer_zero_cannot_decode();
    let Some(dir) = v4_artifact("resident.safetensors") else {
        // Say what did and did not run. A green result on a machine with neither fixture is
        // vacuous, and the only thing worse than that is not knowing it was.
        eprintln!(
            "SKIP: no l0-2 V4 artifact on this machine. layer-0 refusal: {}",
            if refused { "CHECKED (l3-5 present)" } else { "not checked (no l3-5 either)" }
        );
        return;
    };
    let cfg: V4Config = load_config(&dir).expect("l0-2 config");
    a_context_past_the_indexers_reach_is_refused_at_startup(&dir, &cfg);
    let Some(gs) = open_goldens() else { return };
    let ids = prompt_ids(&gs);
    every_layer_matches_the_oracle_at_real_weights(&dir, &cfg, &gs, &ids);
    report();
}
