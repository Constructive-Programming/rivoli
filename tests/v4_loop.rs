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
//! | `attn_norm_out`, `attn_out`, `ffn_norm_out`, `ffn_out` | **no** — emitted, not readable per-sublayer | `ffn_out` would be worst, by a named cause: the shared expert is unclamped, one contribution in seven |
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
//! `q` — is `tests/v4_attn.rs`'s attention cell, on tensors this file does not read. No break has
//! ever been measured through `check` on `L*.out`. Those figures are the right order of magnitude to
//! calibrate against and they do not transfer directly, for a reason stated below: `hc_post` mixes
//! the sublayer output with four residual copies, so a sublayer error is DILUTED in `.out`. So 5e-2
//! is chosen to sit far above re-association and far below a wiring error, and the tight numbers are
//! reported rather than asserted. Injecting a break through THIS gate and recording what it measures
//! is the work that would make the bound measured.

// `rocm`, not `any(rocm, vulkan)`: `v4gpu` is `rocm`-gated because every launcher it drives is
// `backend::hip`'s. Nothing here claims a Vulkan parity that has not been measured, which is
// `tests/kernel_coverage.rs`'s standing rule for this port.
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
    total: usize,
    nonfinite: usize,
}

fn gap(got: &[f32], want: &[f32]) -> Gap {
    assert_eq!(got.len(), want.len(), "comparing tensors of different lengths");
    let mut g = Gap {
        max_rel: 0.0,
        max_abs: 0.0,
        bf16_differing: 0,
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
        g.max_abs = g.max_abs.max(d);
        // Relative to the REFERENCE's magnitude, with a floor so a near-zero element cannot
        // manufacture a huge ratio out of a one-ulp absolute difference.
        let scale = b.abs().max(1e-3);
        g.max_rel = g.max_rel.max(d / scale);
    }
    g
}

/// Report every gap, and gate on the one thing that separates a wiring error from a named
/// numeric deviation.
fn check(what: &str, got: &[f32], want: &[f32], bound: f32) {
    let g = gap(got, want);
    eprintln!(
        "  {what:<28} max_rel {:.3e}  max_abs {:.3e}  bf16 differ {}/{} ({:.2}%)",
        g.max_rel,
        g.max_abs,
        g.bf16_differing,
        g.total,
        100.0 * g.bf16_differing as f64 / g.total as f64,
    );
    assert_eq!(g.nonfinite, 0, "{what}: {} non-finite elements — a defect, not a tolerance", g.nonfinite);
    assert!(
        g.max_rel < bound,
        "{what}: max_rel {:.3e} exceeds the {bound:.0e} separation. That bound is CHOSEN, not measured here; the port's \
         4.2e-1..1.06 figures are `tests/v4_attn.rs`'s, on tensors this file does not read. It sits \
         far above re-association and far below a wiring error. {g:?}",
        g.max_rel
    );
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
            e.probe_layer(l, &here, start_pos).expect("probe_layer");
            let got = e.residual(m).expect("residual readback");
            // The two stages inside the block are not separately readable — `attention`'s output
            // and the MoE's both land in one scratch buffer that the next `hc_post` consumes — so
            // the block's OUTPUT is what is compared. A wrong sublayer still moves it, but this
            // comparison cannot attribute WHICH half moved — and the 4.2e-1 the port measured for a
            // dropped persist copy was measured on `attn_derot`, in another file, not here.
            check(&format!("L{l}.{tag}.out"), &got, golden(gs, &format!("L{l}.{tag}.out")), 5e-2);

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
            check(&format!("L{l}.{tag}.router_weights"), &wg, &ww, 1e-2);

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
    check("head.probe.logits", &logits, golden(gs, "head.probe.logits"), 5e-2);
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
}
