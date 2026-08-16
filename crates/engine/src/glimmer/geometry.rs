//! Muse Glimmer's shapes and footprints: the per-layer attention window, the QK scale pair,
//! and every byte count the pin's [`Floor`] is built from. **Arithmetic over a config,
//! touching no device.**
//!
//! # Why this is a module of its own and not the top of `pin.rs`
//!
//! `pin.rs`, `engine.rs` and `forward.rs` are all `#[cfg(feature = "rocm")]` — they name
//! `DeviceTier` and `DeviceBuf`. **`.github/workflows/ci.yml` has no `rocm` job and no GPU
//! job at all**, so anything behind that gate is compiled exactly as often as someone runs it
//! here. `old:` learned this the direct way: its `GlimmerPin::partition` was unreachable from
//! a featureless build, and the fix there was to move the arithmetic onto the config type.
//! Here it moves into an ungated sibling instead, which keeps it beside the pin that consumes
//! it while still being built and TESTED by `cargo test --workspace`.
//!
//! The two functions that carry a trap are free for a second reason as well: [`window_of`]'s
//! rule and [`qk_scales`]' pairing are the two things a loop is most likely to get fluently
//! wrong, and a rule whose test needs 55 GB of weights to run is a rule nothing runs.

use anyhow::{Context, Result, ensure};
use rivoli_artifact::glimmer::GLIMMER_LAYER_TENSORS;
use rivoli_artifact::glimmer_config::{GlimmerTextConfig, LayerKind};
use rivoli_core::residency::{Bytes, Floor};

/// How many layer-sized streaming slots the tier reserves. Raising it pins one layer fewer,
/// and a streamed layer is **967.942 MB of host memcpy per visit**.
///
/// **The write-after-read dependency is paid** — `pin::GlimmerPin::layer` performs a
/// `device_sync` before every refill — so raising this is not gated on writing that. What it
/// still needs is the OTHER half: a synchronous fill buys no overlap, so a second slot is only
/// worth its extra streamed layer once the fill is async, and that swap replaces the
/// whole-device join with an event on the fetch stream.
pub const STREAM_SLOTS: usize = 1;

/// Alignment slack in the tier request. `DeviceTier::place` starts every reservation at a
/// 256-byte boundary and the pin makes at most `3 + 12·n_layers` = 627 placements on the
/// shipped model, so the padding is under 160 KB; 1 MiB is ~6x that bound and 0.002% of the
/// 55.712 GB it can sit beside.
pub const PIN_SLACK: usize = 1 << 20;

/// How many prompt positions a layer-major prefill batch carries.
///
/// **The chunk size is a memory trade with a very flat payoff curve.** Layer-major prefill
/// fetches each streamed layer once per CHUNK instead of once per token, so a 2048-token
/// prompt at 39 streamed layers goes from 2048 fetches per layer to 8 — 99.6% of the saving a
/// whole-prompt batch would give, for 1/256th of the residual-stream memory. Whole-prompt
/// would cost `n_ctx · hidden · 4`, which is **3.49 GB** at this model's 131072-position
/// ceiling and on top of the KV cache; this costs 6.8 MB.
///
/// Correct at any value: chunk `c` runs EVERY layer before chunk `c+1` starts, so layer `l`'s
/// KV cache already holds every position below the chunk when the chunk's attends run.
pub const PREFILL_CHUNK: usize = 256;

/// Where one layer's keys and values live, and how the attend kernel must be told to read
/// them. The three fields are derived together and never separately — see [`window_of`].
pub struct Window {
    /// `sliding_window` on a sliding layer, 0 (the whole causal prefix) on a full one.
    pub win: usize,
    /// The ring's capacity, or 0 for a linear cache the kernel indexes BY POSITION.
    pub ring_cap: usize,
    /// Which cache slot this position's key and value belong in.
    pub slot: usize,
}

/// This layer's window, ring capacity and slot — all three from ONE match on its kind.
///
/// **Returning all three together is what makes the fluent-wrong combination
/// unconstructible.** `rivoli_gqa_attend` ACCEPTS `win != 0` with `ring_cap == 0` — its guard
/// rejects only the inverse — and that pair silently truncates a full layer's causal prefix to
/// its last `sliding_window` rows. Deriving all three from one match is why that is expressed
/// as a shape rather than as a check.
pub fn window_of(kind: LayerKind, win: usize, n_ctx: usize, pos: usize) -> Result<Window> {
    match kind {
        LayerKind::SlidingAttention => {
            // A `Result` rather than `old:`'s `assert!`. This is `pub`, so a caller that never
            // went through the engine constructor's refusal — a bench, a sizing tool, a future
            // serve path — reaches it directly, and `old:` had it panicking out of a function
            // whose neighbours all return `Result`. "Remainder with a divisor of zero" names
            // neither the field that was zero nor why it may not be.
            ensure!(
                win > 0 && n_ctx > 0,
                "a sliding layer needs a positive window and context, got win {win} and n_ctx \
                 {n_ctx} — a layer typed `sliding_attention` with `sliding_window` 0 is a \
                 contradiction in the manifest"
            );
            // **The window is clamped WITH the ring, and clamping only one of them is a hard
            // refusal at layer 0.** `rivoli_gqa_attend` rejects `ring_cap < win + tq - 1`
            // (code 1005), which at `tq == 1` is `ring_cap < win` — so a ring sized to a
            // context shorter than the window, beside a `win` left at the model's value, is
            // the one pair the launcher refuses outright. `old:`'s first version returned
            // exactly that, and since a short bench gives `n_ctx` ≈ 70 against a 2048-row
            // window, the default invocation could not emit a single token (review,
            // 2026-08-13).
            //
            // Clamping both is not a compromise, it is the same attention: the clamp only
            // fires when `n_ctx <= win`, and then every position is below `win`, so the
            // kernel's `lo = (win > 0 && pos >= win) ? pos - win + 1 : 0` is 0 either way —
            // the whole causal prefix. A sliding layer in a run shorter than its window has
            // nothing to slide past.
            let cap = win.min(n_ctx);
            Ok(Window {
                win: cap,
                ring_cap: cap,
                slot: pos % cap,
            })
        }
        // `ring_cap == 0` makes the slot the position itself, so the cache must run from
        // position 0 — which it does: nothing here ever trims it.
        LayerKind::FullAttention => Ok(Window {
            win: 0,
            ring_cap: 0,
            slot: pos,
        }),
    }
}

/// The two scales the weightless QK-norm takes: `qk_scale_factor` on Q, 1.0 on K.
///
/// **What the model depends on is their PRODUCT, and the assignment is fidelity rather than
/// correctness — measured 2026-08-13, against what `old:`'s own docs had said for weeks.**
/// Both operands are weightless-RMS-normed before the scale, so the score is
/// `(a·q̂)·(b·k̂)·head_dim^-0.5` and only `a·b` enters; RoPE is a norm-preserving rotation
/// applied afterwards and commutes with a scalar, and a cached key scaled by 3.87 dotted with
/// a later query scaled by 1.0 gives the same product. `old:tests/glimmer_chain.rs` red-proves
/// it both ways: SWAPPING the two leaves the logits inside 2.3e-6 reduction noise, while
/// DROPPING the factor moves them by 1.7.
///
/// So the pair stays named — a swap is still a rename rather than an argument reorder, 3.87 on
/// Q is where the reference puts it and where the intermediate magnitudes belong, and it is
/// what any future consumer of `q` or `k` alone (a trace, a probe) would see — but it is no
/// longer described as a hazard that produces wrong text, because it cannot.
///
/// It does NOT replace the softmax scale. `head_dim^-0.5` still applies, for an effective Q
/// factor of `3.87 / sqrt(head_dim)`.
pub struct QkScale {
    pub q: f32,
    pub k: f32,
}

/// [`QkScale`] from the config's `qk_scale_factor`.
pub fn qk_scales(qk_scale_factor: f32) -> QkScale {
    QkScale {
        q: qk_scale_factor,
        k: 1.0,
    }
}

/// How many KV slots layer kind `k` gets at `n_ctx`.
///
/// **Derived from [`window_of`], and read by BOTH the allocation and the budget accounting.**
/// The two must agree exactly: a cache sized from one expression and indexed by another is a
/// device write past the end, and a budget that charges for a different number of slots than
/// were allocated is a report the operator cannot act on. jscpd reported the second copy in
/// `old:` the moment it was written, which is the gate arriving at the same conclusion.
pub fn slots_of(k: LayerKind, win: usize, n_ctx: usize) -> Result<usize> {
    Ok(match window_of(k, win, n_ctx, 0)?.ring_cap {
        0 => n_ctx,
        cap => cap,
    })
}

/// Bytes ONE layer's twelve tensors occupy — **967.942 MB** for the shipped model.
///
/// Sized from [`GlimmerTextConfig::layer_tensor_shape`] rather than restating the shapes, so
/// this number and the checks the placers make cannot drift apart. Every layer is identical —
/// Glimmer has no dense/MoE split — so this is exact rather than an average, and that is what
/// makes a static prefix partition expressible at all.
///
/// **The norms are f32 here and bf16 in the checkpoint** — `convert_glimmer` widens them, the
/// house convention every architecture follows — so a layer here is 53,248 bytes larger than
/// the checkpoint's. (`old:` carried 967.889 MB, the checkpoint figure, in a file whose own
/// 55.712 GB total only reconciles with the f32 one; corrected there 2026-08-12.)
pub fn layer_bytes(cfg: &GlimmerTextConfig) -> Result<usize> {
    let mut per_layer = 0usize;
    for t in GLIMMER_LAYER_TENSORS {
        let shape = cfg.layer_tensor_shape(t)?;
        let n: usize = shape.iter().product();
        per_layer += n * if shape.len() == 1 { 4 } else { 2 };
    }
    Ok(per_layer)
}

/// Bytes the model-level tensors occupy — 5.380 GB for the shipped model.
///
/// `embed_tokens` and `lm_head`, both `[vocab, hidden]` bf16 and both shipped
/// (`tie_word_embeddings` is false, so this is 2x2.690 GB and not one of them), plus the final
/// norm at f32.
///
/// **These are unconditionally resident at every budget, and that is arithmetic rather than
/// convenience:** they are read once per TOKEN each (5.380 GB against a layer's 0.968), so
/// streaming them would buy 5.4 GB of residency and pay for it on every token, while pinning
/// them costs 9.7% of the model. They are therefore part of [`Floor::always_resident`], where
/// a budget that cannot hold them is refused rather than partitioned.
pub fn global_bytes(cfg: &GlimmerTextConfig) -> usize {
    2 * cfg.vocab * cfg.hidden * 2 + cfg.hidden * 4
}

/// The KV cache's device bytes at `n_ctx` — keys AND values, each layer sized by its own kind.
pub fn kv_bytes(cfg: &GlimmerTextConfig, n_ctx: usize) -> Result<usize> {
    let kvd = cfg.num_key_value_heads * cfg.head_dim;
    let mut kv = 0usize;
    for &k in &cfg.layer_types {
        kv += 2 * slots_of(k, cfg.sliding_window, n_ctx)? * kvd;
    }
    kv.checked_mul(4)
        .context("the KV cache footprint overflows a usize")
}

/// The activation scratch's device bytes.
///
/// **This list enumerates every allocation `engine::GlimmerEngine::new` makes rather than the
/// ones a given edit touched**, because it is the only cross-check anyone auditing that
/// constructor has. In `old:` it briefly carried two lines, each missing what the other had.
///
/// `x`, `xn`, `br`, plus `xs` at one row per position in a prefill chunk; `q`, `attn`, `gate`;
/// `mg`, `mu`, `mh`; `logits`; `pick` (2 words).
pub fn scratch_bytes(cfg: &GlimmerTextConfig, n_ctx: usize) -> Result<usize> {
    let qd = cfg.n_heads * cfg.head_dim;
    let act = (3 + PREFILL_CHUNK.min(n_ctx)) * cfg.hidden + 3 * qd + 3 * cfg.inter + cfg.vocab + 2;
    act.checked_mul(4)
        .context("the activation footprint overflows a usize")
}

/// The guards both footprint functions rest on, run before either multiplies anything.
///
/// **They live here and not in the constructor**, moved down after `old:`'s review found each
/// of them arriving too late. `n_ctx` is a caller's prompt plus an unbounded `--bench`, and at
/// `--bench 1e17` a full-attention layer wants more elements than a `u64` holds, so the
/// accumulation in [`kv_bytes`] overflows BEFORE `checked_mul(4)` is ever reached: on the dev
/// profile that is a bare "attempt to multiply with overflow" panic instead of the message the
/// operator needs, and under `--release` it wraps silently and reports a nonsense footprint.
/// `sliding_window` is the other: [`slots_of`] reaches [`window_of`]'s zero-window refusal,
/// and every function here is `pub` over a config that need not have been validated.
pub fn check_footprint_inputs(cfg: &GlimmerTextConfig, n_ctx: usize) -> Result<()> {
    ensure!(n_ctx > 0, "n_ctx must be positive");
    ensure!(
        n_ctx <= cfg.max_position_embeddings,
        "n_ctx {n_ctx} is past this model's {} trained positions",
        cfg.max_position_embeddings
    );
    let sliding = cfg
        .layer_types
        .iter()
        .filter(|k| **k == LayerKind::SlidingAttention)
        .count();
    ensure!(
        sliding == 0 || cfg.sliding_window > 0,
        "the config types {sliding} layers `sliding_attention` and gives `sliding_window` 0"
    );
    Ok(())
}

/// Everything a Glimmer run pays before ONE layer is resident, as
/// [`rivoli_core::residency::partition`] takes it.
///
/// **`kv_at_max_ctx` and `scratch` are filled in, where `glm::pin` passes 0 for both.** That
/// is not an inconsistency: GLM's `--max-mem` has always budgeted weights only and every
/// recorded benchmark reads it that way, so folding them in there would be a semantic change
/// to the flag owed its own measurement. Glimmer has no such history, and `old:` found the gap
/// as a live defect — its pin sized a tier from the budget and cleared `guard_capacity`, then
/// every activation and KV allocation was an unguarded `hipMalloc` on top of it. At the
/// 131072-position ceiling the KV cache alone is ~3.4 GiB, 85% of the reserve that exists for
/// driver scratch, and the residency line the operator reads counted none of it. `Floor` has
/// these two fields precisely so a pin can state them.
/// **`slots` is the caller's question, not a constant, and that is the subtle part.** A
/// streaming slot is a charge the run pays only IF it streams, and `Floor` has no way to say
/// "conditionally" — so the pin asks twice: once with `slots = 0` ("does the whole model fit
/// with no slot at all?", which is P1's degenerate happy case where the streaming path idles
/// and allocates nothing), and, only if something streams, again with the slot real.
///
/// Charging it unconditionally costs a whole layer at exactly the budgets where the model
/// just fits: at `budget = everything_but_a_slot + n_layers · layer` the pin would hold 51 of
/// 52 and stream the last, which is **967.942 MB of host memcpy per token** bought for
/// nothing. `old:` avoided it with an early return before its partition ran; here it is the
/// same fact expressed as an argument to the one placement author.
pub fn floor_of(cfg: &GlimmerTextConfig, n_ctx: usize, slots: usize) -> Result<Floor> {
    check_footprint_inputs(cfg, n_ctx)?;
    Ok(Floor {
        always_resident: Bytes((global_bytes(cfg) + PIN_SLACK) as u64),
        kv_at_max_ctx: Bytes(kv_bytes(cfg, n_ctx)? as u64),
        scratch: Bytes(scratch_bytes(cfg, n_ctx)? as u64),
        slot_bytes: Bytes((slots * layer_bytes(cfg)?) as u64),
    })
}

#[cfg(test)]
mod geometry_tests {
    // No `allow(clippy::unwrap_used)`: every test below returns `Result` and uses `?`, which
    // is both what this workspace's lints want and the better failure — a `?` shows the
    // refusal message these functions were written to produce, where `.unwrap()` shows only
    // "called `Result::unwrap()` on an `Err` value" and throws the argument away.
    use super::{
        LayerKind, PIN_SLACK, STREAM_SLOTS, floor_of, global_bytes, qk_scales, slots_of, window_of,
    };
    use anyhow::Result;

    /// The full-attention layer's cache is LINEAR and its slot is the position itself, so
    /// nothing may ever trim it; the sliding one is a ring. **`win != 0` with `ring_cap == 0`
    /// is the pair that truncates a full layer's causal prefix, and the launcher ACCEPTS it**
    /// — its guard rejects only the inverse — so the property worth asserting is that no
    /// input to this function can produce it.
    #[test]
    fn no_input_can_produce_the_pair_that_truncates_a_full_layers_prefix() -> Result<()> {
        for n_ctx in [1usize, 7, 2047, 2048, 2049, 131072] {
            for pos in [0usize, 1, 2047, 2048] {
                if pos >= n_ctx {
                    continue;
                }
                for kind in [LayerKind::SlidingAttention, LayerKind::FullAttention] {
                    let w = window_of(kind, 2048, n_ctx, pos)?;
                    assert!(
                        !(w.win != 0 && w.ring_cap == 0),
                        "{kind:?} at n_ctx {n_ctx} pos {pos} produced win {} ring_cap {}",
                        w.win,
                        w.ring_cap
                    );
                    // And the slot is always inside the cache `slots_of` sizes — the two
                    // must not be able to disagree, because that is a device write past the
                    // end that neither the launcher nor HIP would report.
                    let slots = slots_of(kind, 2048, n_ctx)?;
                    assert!(w.slot < slots, "slot {} past {slots} slots", w.slot);
                }
            }
        }
        Ok(())
    }

    /// The clamp fires TOGETHER on both fields. A ring clamped to a short context beside a
    /// `win` left at the model's 2048 is `ring_cap < win`, which `rivoli_gqa_attend` refuses
    /// outright (code 1005) — the shape that could not emit a single token under the default
    /// bench in `old:`.
    #[test]
    fn a_context_shorter_than_the_window_clamps_the_ring_and_the_window_together() -> Result<()> {
        let w = window_of(LayerKind::SlidingAttention, 2048, 70, 69)?;
        assert_eq!((w.win, w.ring_cap), (70, 70), "the clamp must move both");
        assert!(w.ring_cap >= w.win, "ring_cap < win is launcher code 1005");
        // Past the window it is a genuine ring, and the window stays at the model's value.
        let w = window_of(LayerKind::SlidingAttention, 2048, 4096, 3000)?;
        assert_eq!((w.win, w.ring_cap, w.slot), (2048, 2048, 3000 % 2048));
        Ok(())
    }

    /// A sliding layer with no window is a contradiction in the manifest, and it must be an
    /// error rather than `pos % 0`. These functions are `pub`, so callers that never met the
    /// engine's constructor reach them directly.
    #[test]
    fn a_sliding_layer_with_a_zero_window_is_refused_and_not_a_division_by_zero() {
        assert!(window_of(LayerKind::SlidingAttention, 0, 64, 0).is_err());
        assert!(window_of(LayerKind::SlidingAttention, 2048, 0, 0).is_err());
        // A full layer has no window to be zero, so the same config is fine for it.
        assert!(window_of(LayerKind::FullAttention, 0, 64, 0).is_ok());
    }

    /// 3.87 multiplies Q and 1.0 multiplies K. Only the PRODUCT enters the score, so this
    /// pins the ASSIGNMENT — which is fidelity, and which a trace or probe of `q` alone
    /// would see.
    #[test]
    fn the_qk_scale_factor_is_qs_alone() {
        let s = qk_scales(3.87);
        assert_eq!((s.q, s.k), (3.87, 1.0));
    }

    /// The floor's four charges are each non-zero and each land in their own field.
    ///
    /// **Anti-vacuity for the thing this arm does differently from GLM's pin**, which passes
    /// `kv_at_max_ctx: 0` and `scratch: 0` with an argument about `--max-mem`'s history. A
    /// Glimmer floor that quietly did the same would partition against a budget it had not
    /// reserved the KV cache out of — the live defect `old:` shipped — and nothing downstream
    /// reads a floor field to notice.
    #[test]
    fn the_floor_charges_for_the_kv_cache_and_the_scratch_and_not_only_the_weights() -> Result<()> {
        let cfg = tiny_config();
        let f = floor_of(&cfg, 64, STREAM_SLOTS)?;
        assert_eq!(
            f.always_resident.0,
            (global_bytes(&cfg) + PIN_SLACK) as u64,
            "always_resident is the model-level tensors plus alignment slack"
        );
        assert!(f.kv_at_max_ctx.0 > 0, "the KV cache must be charged for");
        assert!(
            f.scratch.0 > 0,
            "the activation scratch must be charged for"
        );
        assert!(
            f.slot_bytes.0 > 0,
            "{STREAM_SLOTS} slot(s) must be charged for"
        );
        // A longer context charges strictly more KV and never less of anything else.
        let g = floor_of(&cfg, 128, STREAM_SLOTS)?;
        assert!(
            g.kv_at_max_ctx.0 > f.kv_at_max_ctx.0,
            "KV must grow with n_ctx"
        );
        assert_eq!(g.slot_bytes, f.slot_bytes, "weights do not move with n_ctx");
        Ok(())
    }

    /// A context past the trained positions is refused before anything multiplies, which is
    /// what keeps `--bench` from reaching an overflow instead of a message.
    #[test]
    fn a_context_past_the_trained_positions_is_refused_before_the_arithmetic() {
        let cfg = tiny_config();
        assert!(floor_of(&cfg, cfg.max_position_embeddings + 1, STREAM_SLOTS).is_err());
        assert!(floor_of(&cfg, 0, STREAM_SLOTS).is_err());
    }

    /// A Glimmer-shaped config small enough to reason about, built through serde so the
    /// serde renames and the `LayerKind` spellings are exercised rather than bypassed —
    /// constructing the struct literally would test this module against a shape no
    /// `config.json` can produce.
    fn tiny_config() -> super::GlimmerTextConfig {
        let doc = serde_json::json!({
            "model_type": "muse_glimmer_text",
            "num_hidden_layers": 4,
            "hidden_size": 64,
            "vocab_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 128,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.196,
            "final_logit_softcapping": 20.0,
            "sliding_window": 16,
            "layer_types": ["sliding_attention", "sliding_attention",
                            "sliding_attention", "full_attention"],
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0],
            "rope_parameters": { "rope_theta": 1000000.0, "rope_type": "default" },
            "max_position_embeddings": 4096,
            "tie_word_embeddings": false,
            "hidden_activation": "silu",
            "attention_bias": false,
        });
        // `expect` in a fixture builder, not in a test body: a malformed literal above is a
        // broken test rather than a failed assertion, and there is nothing for `?` to
        // propagate to from a helper every test calls.
        #[allow(clippy::expect_used)]
        serde_json::from_value(doc).expect("the fixture is a valid GlimmerTextConfig")
    }
}
