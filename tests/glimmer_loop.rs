//! **S3: the two decisions the layer loop makes per layer, gated without a device.**
//!
//! Two of the loop's per-layer decisions are pure functions of the config — which cache a layer
//! reads, and which scale each of Q and K takes — and both are traps this port recorded as
//! unreachable-until-a-call-site-exists. They are reachable now, and gated here without a device.
//!
//! The third test DOES take a device, and it is the one that was missing: it runs the loop.
//!
//! > **The first version of this header said `glimmer_gpu.rs` "needs 55.7 GB of weights to run one
//! > token", and used that to explain why nothing here executes it. That is false, and review said
//! > so the same day.** The toy Glimmer checkpoint `tests/common` already builds — 4 layers, hidden
//! > 8, 2 Q heads over 1 KV head, `head_dim` 8, `sliding_window` 2 — converts and pins exactly like
//! > the real one, so `Glimmer::new` needs a device and not a checkpoint. The premise had cost
//! > something concrete by the time it was challenged: the commit that wrote it shipped a loop that
//! > could not decode at all (see [`the_loop_runs_end_to_end_on_the_toy_checkpoint`]).
//!
//! # What this covers, and what it does NOT
//!
//! It covers SELECTION and EXECUTION, not arithmetic. `gqa_attend` and `rmsnorm_weightless_batch`
//! are scored against the anchor goldens by `glimmer_attend.rs` and `glimmer_qk_norm.rs`; the
//! decode test below would pass with either replaced by a stub that writes plausible numbers.
//!
//! **Numeric proof of the whole chain is still G3's and still unbuilt.** The vendored goldens hold
//! activations only — 1,099 captured intermediates over 4 layers and 7 steps, and not one
//! parameter — so a zero-tolerance comparison needs either the tiny checkpoint's weights or a
//! host oracle over the toy fixture's. That decision is not made here.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

mod common;
use common::{GLIMMER_FIXTURE_DIM as DIM, TempRoot, glimmer_convert_fixture};
use rivoli::artifact::model as gm;
use rivoli::artifact::model::LayerKind;
use rivoli::glimmer_gpu::{Glimmer, Window, qk_scales, window_of};

/// Muse Glimmer's shipped window, and a context comfortably past it so the ring wraps.
const WIN: usize = 2048;
const CTX: usize = 5000;

/// **The pair that silently truncates a full layer cannot be constructed.**
///
/// `launch_gqa_attend` derives its causal bound from `(start_pos, win)` and its slot map from
/// `ring_cap`, and its guard rejects only `ring_cap != 0` with `ring_cap < win + tq - 1`. The
/// INVERSE — `win != 0` with `ring_cap == 0` on a full-attention layer — is accepted, and it
/// bounds a global layer to its last `win` rows: fluent, wrong, and invisible to any shape check.
/// The plan named it as the first thing this loop could get wrong.
///
/// Swept over both kinds and a range of positions rather than asserted at one, because the failure
/// this guards against is a `match` arm, and a single position cannot tell an arm from a constant.
#[test]
fn a_full_layer_can_never_be_handed_a_window_without_a_ring() {
    let mut full = 0;
    let mut sliding = 0;
    for pos in [0, 1, WIN - 1, WIN, WIN + 1, 2 * WIN, CTX - 1] {
        for kind in [LayerKind::SlidingAttention, LayerKind::FullAttention] {
            let w = window_of(kind, WIN, CTX, pos);
            assert_window_is_launchable(kind, CTX, pos, &w);
            match kind {
                // A full layer attends the whole prefix and indexes the cache BY POSITION, so its
                // slot is the position and its cache must run from 0.
                LayerKind::FullAttention => {
                    assert_eq!((w.win, w.ring_cap, w.slot), (0, 0, pos));
                    full += 1;
                }
                // A sliding layer's ring is exactly `sliding_window` — the launcher's floor of
                // `win + tq - 1` at the `tq == 1` this loop decodes at.
                LayerKind::SlidingAttention => {
                    assert_eq!((w.win, w.ring_cap, w.slot), (WIN, WIN, pos % WIN));
                    sliding += 1;
                }
            }
        }
    }
    // A census, because both arms above are inside a loop whose iteration count a future edit
    // could take to zero while every assertion still "passes".
    assert_eq!((full, sliding), (7, 7), "the sweep did not run both arms");
}

/// **Every window this function can return is one `launch_gqa_attend` will ACCEPT.**
///
/// `kernels/attn.hip` refuses two pairs: `win == 0` with `ring_cap != 0` (637), and
/// `ring_cap != 0` with `ring_cap < win + tq - 1` (647, code 1005) — which at the `tq == 1` this
/// loop decodes at is `ring_cap < win`. A window function whose output the launcher rejects is not
/// a subtle defect; it is a hard refusal on the first attention launch of the first layer.
///
/// > **THIS TEST EXISTS BECAUSE ITS PREDECESSOR CERTIFIED THE DEFECT.** The version shipped with
/// > the loop asserted `window_of(Sliding, 2048, 12, 5) == (2048, 12, 5)` — reasoning about the
/// > allocation and never about the guard — and 12 < 2048 is exactly the pair 1005 rejects. Since
/// > `--bench` defaults to 64, the default invocation could not emit one token, and BOTH reviews
/// > found it (2026-08-13). A test that pins a literal triple asserts what the code does; this one
/// > asserts what the consumer requires.
fn assert_window_is_launchable(kind: LayerKind, n_ctx: usize, pos: usize, w: &Window) {
    assert!(
        !(w.win != 0 && w.ring_cap == 0),
        "{kind:?} n_ctx {n_ctx} pos {pos}: a {}-row window against a LINEAR cache — the launcher \
         ACCEPTS that pair and truncates the causal prefix to the last {} rows",
        w.win,
        w.win
    );
    assert!(
        w.ring_cap == 0 || w.ring_cap >= w.win,
        "{kind:?} n_ctx {n_ctx} pos {pos}: ring {} against window {} — attn.hip:647 rejects \
         `ring_cap < win + tq - 1` with code 1005, so this refuses at the first launch",
        w.ring_cap,
        w.win
    );
    assert!(
        w.ring_cap == 0 || w.slot < w.ring_cap,
        "{kind:?} n_ctx {n_ctx} pos {pos}: slot {} is outside a {}-slot ring",
        w.slot,
        w.ring_cap
    );
    assert!(
        w.ring_cap != 0 || w.slot < n_ctx,
        "{kind:?} pos {pos}: a linear cache is indexed BY POSITION, so slot {} must be inside the \
         {n_ctx} positions it was allocated for",
        w.slot
    );
}

/// **The guard holds at every context, including the ones shorter than the window.**
///
/// Swept over `n_ctx` because that is the axis the shipped defect lived on and the axis the
/// original test held fixed.
///
/// **This sweep, and not the decode below, is what catches that defect — MEASURED, by restoring the
/// pre-fix `window_of` and running both.** The toy fixture's `sliding_window` is 2, so
/// `min(2, n_ctx)` is 2 for any context a decode can use and the ring is never short:
/// [`the_loop_runs_end_to_end_on_the_toy_checkpoint`] stayed GREEN on the broken window while this
/// went red at `n_ctx 1`. A fixture cannot be relied on to reach a defect whose axis it does not
/// span, which is why the sweep is deliberate rather than a by-product of running the loop.
#[test]
fn every_context_yields_a_window_the_launcher_accepts() {
    let mut cells = 0;
    for n_ctx in [1, 2, 12, WIN - 1, WIN, WIN + 1, CTX] {
        for pos in [0, 1, WIN - 1, WIN, CTX - 1] {
            if pos >= n_ctx {
                continue;
            }
            for kind in [LayerKind::SlidingAttention, LayerKind::FullAttention] {
                let w = window_of(kind, WIN, n_ctx, pos);
                assert_window_is_launchable(kind, n_ctx, pos, &w);
                cells += 1;
            }
        }
    }
    assert!(cells >= 20, "the sweep ran only {cells} cells");
    // The clamp fires only when the context cannot fill the window, and then it clamps BOTH — a
    // sliding layer with fewer positions than its window has nothing to slide past, so `win = cap`
    // is the same attention (`pos >= win` is never true either way, so the kernel's `lo` is 0).
    let short = window_of(LayerKind::SlidingAttention, WIN, 12, 5);
    assert_eq!((short.win, short.ring_cap, short.slot), (12, 12, 5));
    let long = window_of(LayerKind::SlidingAttention, WIN, CTX, WIN + 3);
    assert_eq!((long.win, long.ring_cap, long.slot), (WIN, WIN, 3));
    // And a full layer is untouched by the clamp at either length.
    assert_eq!(window_of(LayerKind::FullAttention, WIN, 12, 5).ring_cap, 0);
}

/// **Q takes `qk_scale_factor`, K takes 1.0, and their PRODUCT is `qk_scale_factor`.**
///
/// The product is the load-bearing half and the assignment is not, which inverts what this file
/// said when it was written. Both operands are normed before the scale, so the attention score
/// carries only `a·b`; `tests/glimmer_chain.rs` measured a swap leaving the logits inside 2.3e-6
/// reduction noise and a DROPPED factor moving them by 1.7. See [`rivoli::glimmer_gpu::QkScale`].
///
/// So this asserts the product first — that is what a wrong value breaks — and the assignment
/// second, as the fidelity point it is.
#[test]
fn the_qk_scale_is_qs_alone_and_k_takes_unity() {
    let s = qk_scales(3.87);
    assert_eq!(
        s.q, 3.87,
        "Q takes qk_scale_factor, after the weightless norm"
    );
    assert_eq!(
        s.k, 1.0,
        "K takes 1.0 — a K scaled by qk_scale_factor is fluent and wrong, and the goldens gate \
         the reference rather than the caller"
    );
    // The product is what enters the score, so it is asserted as such rather than inferred from
    // the two fields — a version that scaled BOTH by `sqrt(3.87)` would satisfy the fields' spirit
    // and the model, and one that scaled both by 3.87 would satisfy neither.
    assert_eq!(
        s.q * s.k,
        3.87,
        "the product is what the attention score carries"
    );
    let one = qk_scales(1.0);
    assert_eq!((one.q, one.k), (1.0, 1.0));
    // And the scale is passed through, not derived: a version that returned a constant would pass
    // the first assertion alone.
    assert_eq!(qk_scales(0.5).q, 0.5);
}

/// Convert the toy checkpoint and read back its config. The caller owns the temp root.
///
/// Shared by the two device tests below rather than written twice — `build.rs`'s jscpd gate
/// rejected the second copy at 86 tokens, and it is right about the substance: two setups that
/// could disagree about which artifact is under test would let one of them pass on the other's.
fn fixture(tag: &str) -> (TempRoot, gm::GlimmerConfig) {
    let root = TempRoot::new(tag);
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let cfg = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    (root, cfg)
}

/// An engine over that artifact at the whole budget — the fixture is kilobytes, so the partition
/// is all-resident and the streaming path is `glimmer_residency.rs`'s to gate.
fn engine(root: &TempRoot, gt: &gm::GlimmerTextConfig, n_ctx: usize) -> anyhow::Result<Glimmer> {
    Glimmer::new(root.join("out").to_str().unwrap(), gt, None, n_ctx)
}

/// **The loop runs, end to end, on the toy checkpoint — the check that was missing.**
///
/// A GPU arm, so this binary needs the flock and `--test-threads=1`. Everything above it is
/// deviceless; this one builds a `DeviceTier`.
///
/// It asserts almost nothing about the NUMBERS, and that is honest rather than lazy: the fixture's
/// weights are synthetic, so there is no right answer to compare against. What it proves is that
/// **every launcher in the chain accepts the arguments this loop hands it, at every layer kind,
/// across the sliding ring's wrap** — which is the class of defect the commit that first landed the
/// loop shipped: `window_of` handed `gqa_attend` a ring shorter than its window, the launcher
/// refused with code 1005, and nothing in the tree executed a single layer to notice. Both reviews
/// found it by reading; this finds it in 200 ms.
///
/// The prompt and generation are sized to cross the fixture's `sliding_window` of 2 several times,
/// so the ring wraps and the full-attention layer's linear cache grows past it.
#[test]
fn the_loop_runs_end_to_end_on_the_toy_checkpoint() {
    let (root, cfg) = fixture("glimmer-loop");
    let gt = &cfg.text;
    // The fixture's `vocab_size` is `DIM + 4`; `glimmer_fixture_eos` reserves the top two ids as
    // stop tokens, so a prompt below them cannot terminate the run by accident.
    let prompt: Vec<u32> = vec![0, 1, 2, 3, 4];
    let max_new = 6;
    let mut e = engine(&root, gt, prompt.len() + max_new)
        .expect("build the engine over the converted fixture");
    let out = e
        .decode(&prompt, max_new, &[])
        .expect("the loop must complete every layer at every position");
    assert_eq!(
        out.len(),
        max_new,
        "with no stop tokens the run emits exactly what it was asked for"
    );
    for t in &out {
        assert!(
            (*t as usize) < gt.vocab,
            "emitted token {t} is past the fixture's vocabulary of {}",
            gt.vocab
        );
    }
    // Both layer kinds ran: the fixture is one period of `[s,s,s,full]`, and the sweep above
    // covers what each was handed. Asserted here so a fixture edit that made every layer sliding
    // would not quietly turn this into a single-kind test.
    let has = |want: LayerKind| gt.layer_types.contains(&want);
    assert!(
        has(LayerKind::SlidingAttention) && has(LayerKind::FullAttention),
        "the fixture must exercise both layer kinds, got {:?}",
        gt.layer_types
    );
    // And the run crossed the ring: `sliding_window` is 2 against 11 positions, so every sliding
    // layer's ring wrapped four times. A run shorter than the window would not have.
    assert!(
        prompt.len() + max_new > 2 * gt.sliding_window,
        "the fixture's window is {}, so this run does not wrap the ring",
        gt.sliding_window
    );
}

/// **A stop token ends the run and is NOT part of the output.**
///
/// `GpuEngine::generate` returns before pushing, and two decode drivers in one binary disagreeing
/// about whether the terminator is output is the kind of thing a `serve.rs` port inherits silently.
/// The first version of `decode` pushed and then tested (review, 2026-08-13).
#[test]
fn a_stop_token_ends_the_run_without_appearing_in_it() {
    let (root, cfg) = fixture("glimmer-loop-eos");
    let gt = &cfg.text;
    let prompt: Vec<u32> = vec![0, 1, 2];
    let mut e = engine(&root, gt, prompt.len() + 8).unwrap();
    // Every id is a stop token, so whatever the head picks ends the run immediately. That makes
    // the assertion independent of the synthetic weights — the alternative, guessing which token
    // this fixture emits, would be a test of the weights rather than of the loop.
    let all: Vec<u32> = (0..gt.vocab as u32).collect();
    let out = e.decode(&prompt, 8, &all).unwrap();
    assert!(
        out.is_empty(),
        "every token is a stop token, so nothing may be emitted — got {out:?}"
    );
    // And with no stop tokens the same engine emits the full count, which is what makes the line
    // above a statement about EOS and not about a loop that emits nothing.
    let mut e2 = engine(&root, gt, prompt.len() + 8).unwrap();
    assert_eq!(e2.decode(&prompt, 8, &[]).unwrap().len(), 8);
}
