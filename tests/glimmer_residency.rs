//! **R1's gates: the budget trades speed, never bytes.**
//!
//! `investigations/glimmer-integration.md` §R1, `reference/principles.md` **P4** and **P6**.
//! What this suite exists to hold is one claim — **`GlimmerPin::layer(l)` resolves to the same
//! BYTES at every budget**, from all-resident down to the floor — because that is what makes
//! "the budget is a performance knob" true rather than aspirational.
//!
//! > **The plan's G-R1(a) asked for something this stage cannot give, and this is the
//! > substitute.** It reads "tiny-model DECODE output is bit-identical across every budget".
//! > There is no decode until S3 — the layer loop is that stage's whole content — so an
//! > end-to-end gate here would have to wait, and R1 would ship its partition ungated. The
//! > per-layer form below is available now and is *stronger where it overlaps*: it compares
//! > 12 tensors per layer per budget rather than one token stream, so it localises a wrong
//! > partition to the tensor instead of to the run. S3 still owes the end-to-end version,
//! > because a loop can consume correct bytes in the wrong ORDER and only a decode sees that.
//!
//! **A GPU arm** — `DeviceTier::new` allocates — except the two partition-arithmetic tests,
//! which are pure and run with no device. Those are what CI covers, and CI has no rocm job:
//! `..._at_every_boundary` at the fixture's widths, `..._at_the_shipped_widths` at the real ones,
//! because review showed several fixture-width assertions cannot fail there (1 MiB of alignment
//! slack is 99.6% of every fixture budget).

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
// Aliased module + const bindings rather than a flat `use` list, for the reason
// `glimmer_convert.rs` records about its own preamble: jscpd normalizes identifiers, so two test
// binaries importing the same names from the same two modules match as a clone. This is the
// fourth Glimmer binary and the preamble is where they collide — the gate is right that the
// text is identical, and the honest fix is to stop writing it a fourth time.
use common as fx;
use fx::{TempRoot, glimmer_convert_fixture};
use rivoli::artifact::model as gm;

const DIM: usize = fx::GLIMMER_FIXTURE_DIM;
const L: usize = fx::GLIMMER_FIXTURE_LAYERS;

/// The fixture's config — for the deviceless arithmetic test.
///
/// Converting needs no device, which is what keeps the partition arithmetic testable on a
/// machine with no GPU. (This doc claimed the test "needs no temp directory either", describing
/// a rejected draft that built the config by hand; the body has always converted. Review, 2026-08-12.)
fn tiny_cfg() -> gm::GlimmerTextConfig {
    let root = TempRoot::new("glimmer-residency-cfg");
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let cfg: gm::GlimmerConfig = gm::load_config(root.join("out").to_str().unwrap()).unwrap();
    cfg.text
}

/// **The partition, at every boundary, with no device.**
///
/// Six claims, each one a way the arithmetic could be wrong while looking right:
/// `None` pins everything; a budget above the model pins everything (and must NOT allocate
/// slots nothing fills); the floor pins ZERO layers and is accepted; one byte under the floor
/// is REFUSED and the message carries the numbers; each additional layer's worth of budget
/// pins exactly one more layer; and a budget between two layer boundaries pins the lower one
/// rather than rounding up into memory it does not have.
#[test]
fn the_partition_arithmetic_holds_at_every_boundary() {
    let cfg = tiny_cfg();
    let want = cfg.resident_bytes().unwrap();
    let layer = cfg.layer_bytes().unwrap();
    let floor = cfg.floor_bytes().unwrap();
    assert_eq!(cfg.n_layers, L, "the fixture's layer count moved");
    // Kept for shape, but it is the SHIPPED-widths twin that can fail: at 416-byte globals and
    // 1,920-byte layers a floor charging zero slots is still 1,048,992 > 2,336. Review, 2026-08-12.
    assert!(
        floor > cfg.global_bytes(),
        "the floor must cover the globals"
    );

    assert_eq!(
        cfg.partition(None).unwrap().0,
        L,
        "None must pin every layer"
    );
    // **"Above the model" is `want + SLACK`, not a multiple of `want`.** At the fixture's widths
    // the whole model is a few KB and the 1 MiB alignment slack dwarfs it, so `want * 4` is
    // still under the floor and was REFUSED — the first draft asserted it pinned everything and
    // went red on the featureless run. The lesson generalises past the test: every budget
    // comparison here has to be against the floor and `want + SLACK`, never against a scaling of
    // the model, because the two orderings swap between tiny and real widths.
    let all_resident = want + gm::GLIMMER_PIN_SLACK;
    for b in [all_resident, all_resident * 4] {
        let (pinned, capacity) = cfg.partition(Some(b)).unwrap();
        assert_eq!(pinned, L, "a budget of {b} must pin every layer");
        // And it must not ASK for more than the all-resident set. `DeviceTier::new` allocates
        // its capacity rather than treating it as a ceiling, and also feeds `guard_capacity`, so
        // an over-request both wastes GTT and can turn a workable budget into a refusal.
        assert_eq!(
            capacity, all_resident,
            "an over-generous budget must request only what it uses"
        );
    }

    let (pinned, _) = cfg.partition(Some(floor)).unwrap();
    assert_eq!(
        pinned, 0,
        "exactly the floor buys the globals and the slots, and no layers"
    );

    let e = format!("{:#}", cfg.partition(Some(floor - 1)).unwrap_err());
    for fragment in [
        "below this artifact's floor",
        "read once per",
        "Weights only",
    ] {
        assert!(
            e.contains(fragment),
            "the refusal must say {fragment:?}: {e}"
        );
    }

    // One layer at a time, and the between-boundary case in the same loop — `floor + k·layer`
    // must pin exactly `k`, and `floor + k·layer - 1` exactly `k-1`.
    //
    // **Only up to the crossover, and the crossover is a property rather than an exception.**
    // `floor + k·layer` = globals + (SLOTS + k)·layer + slack, while pinning EVERY layer costs
    // globals + n_layers·layer + slack and needs no slots at all. So once `k + SLOTS >= n_layers`
    // the same budget buys the whole model and the partition correctly stops streaming. A loop
    // that expected `k` there would be asserting that the pin declines free residency; it went
    // red on exactly that. Derived from the constant, so it follows a change to the slot count.
    let crossover = L - gm::GLIMMER_STREAM_SLOTS;
    for k in 1..crossover {
        assert_eq!(
            cfg.partition(Some(floor + k * layer)).unwrap().0,
            k,
            "floor + {k} layers must pin {k}"
        );
        assert_eq!(
            cfg.partition(Some(floor + k * layer - 1)).unwrap().0,
            k - 1,
            "one byte short of {k} layers must pin {}, not round up",
            k - 1
        );
    }
    // The crossover itself, asserted rather than merely avoided: at `k + SLOTS == n_layers` the
    // budget must pin EVERYTHING and drop the slots.
    let (pinned, capacity) = cfg.partition(Some(floor + crossover * layer)).unwrap();
    assert_eq!(
        (pinned, capacity),
        (L, all_resident),
        "at the crossover the budget holds the whole model, so nothing should stream"
    );
}

/// **P4 as a gate: every budget resolves every layer to the same bytes.**
///
/// The all-resident pin is the reference — it is the partition this port shipped at S1a and the
/// one `glimmer_pin.rs` already checks tensor-by-tensor against the converter's output. Every
/// other budget must reproduce it exactly, including the budgets where most layers arrive
/// through a slot that has been overwritten several times.
///
/// **Read through `layer()` on both sides, never from a stored pointer.** A test that captured
/// the reference's addresses once and compared them later would be comparing a slot against
/// itself; the comparison has to re-resolve per budget, which is also how a real caller uses it.
#[test]
#[cfg(feature = "rocm")]
fn every_budget_resolves_every_layer_to_the_same_bytes() {
    use rivoli::memory::pin::GlimmerPin;
    let root = TempRoot::new("glimmer-residency-bytes");
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let dir = root.join("out");
    let dir = dir.to_str().unwrap();
    let cfg = gm::load_config::<gm::GlimmerConfig>(dir).unwrap().text;
    let layer = cfg.layer_bytes().unwrap();
    let floor = cfg.floor_bytes().unwrap();

    // The reference: all resident, no slots.
    let mut all = GlimmerPin::build(dir, &cfg, None).unwrap();
    assert_eq!(all.streamed_layers(), 0);
    let reference: Vec<Vec<Vec<u8>>> = (0..L).map(|l| tensors_of(all.layer(l).unwrap())).collect();
    drop(all);

    // Every partition from "nothing pinned" to "all but one pinned". `floor` is the interesting
    // end: all L layers cycle through 2 slots, so every layer but the last two is read out of a
    // slot that gets overwritten before the sweep ends — which is what a second pass catches.
    // Only the budgets that actually stream — past `L - SLOTS` the same budget buys the whole
    // model and there is nothing to compare (the crossover is gated in the arithmetic test).
    let mut checked = 0;
    let mut streaming_budgets = 0;
    for k in 0..(L - gm::GLIMMER_STREAM_SLOTS) {
        let budget = floor + k * layer;
        let mut pin = GlimmerPin::build(dir, &cfg, Some(budget)).unwrap();
        assert_eq!(pin.pinned_layers(), k, "budget for {k} layers");
        assert_eq!(pin.streamed_layers(), L - k);
        streaming_budgets += 1;
        // TWICE, ascending, so a slot that is refilled between the two passes is exercised. A
        // single pass would pass on a pin that handed out a stale slot on a revisit.
        for _pass in 0..2 {
            for (l, want) in reference.iter().enumerate() {
                assert_eq!(
                    &tensors_of(pin.layer(l).unwrap()),
                    want,
                    "budget pinning {k} of {L}: layer {l} resolved to different bytes than the \
                     all-resident pin — the budget changed the MODEL, not just its speed"
                );
                checked += 1;
            }
        }
        // A streamed layer must actually have been filled: `fills` counts slot writes, and a
        // partition that quietly pinned everything would report zero and pass every byte
        // comparison above for the wrong reason.
        // **Bounded above as well as below.** `> 0` alone passes for a map that thrashes, and
        // thrash is invisible to the byte comparison (which reads after the fill). With one slot
        // and two ascending passes over L layers, every visit to a streamed layer is a miss, so
        // the exact count is derivable — review found the unbounded version.
        let (_, fills) = pin.slot_stats();
        assert_eq!(
            fills,
            ((L - k) * 2) as u64,
            "budget pinning {k} of {L}: expected one fill per streamed layer per pass"
        );
    }
    // Anti-vacuity, and derived from a DIFFERENT quantity than the loop bound: the sweep must
    // have run at least two streaming budgets (nothing pinned, and one layer pinned), or the
    // whole test is a restatement of `glimmer_pin.rs`.
    assert!(
        streaming_budgets >= 2,
        "only {streaming_budgets} budget(s) streamed anything"
    );
    assert_eq!(
        checked,
        streaming_budgets * L * 2,
        "the sweep did not cover every (budget, layer)"
    );
    println!("{checked} (budget, layer, pass) resolutions, all byte-identical to all-resident");
}

/// Layer `l`'s twelve tensors, read back out of the tier as bytes.
///
/// Sizes come from the pin's own dims rather than from the config: a field wired to the wrong
/// tensor would then read the wrong LENGTH too, and show up as a length mismatch instead of
/// silently comparing a prefix.
#[cfg(feature = "rocm")]
fn tensors_of(p: &rivoli::memory::pin::GlimmerLayerPin) -> Vec<Vec<u8>> {
    // Safe for the reason `glimmer_pin.rs` gives: the tier is a host-fillable VMM mapping, so
    // every pointer the pin hands out is readable here.
    let f32s = |ptr: *const f32, n: usize| unsafe {
        std::slice::from_raw_parts(ptr as *const u8, n * 4).to_vec()
    };
    let mat = |w: rivoli::memory::pin::Bf16Weight| unsafe {
        std::slice::from_raw_parts(w.packed as *const u8, w.o_dim * w.i_dim * 2).to_vec()
    };
    // `hidden` is recovered from a projection's input dim rather than passed in, so this
    // helper needs nothing but the pin.
    let hidden = p.q.i_dim;
    vec![
        f32s(p.input_ln, hidden),
        f32s(p.post_attn_ln, hidden),
        f32s(p.pre_ffn_ln, hidden),
        f32s(p.post_ffn_ln, hidden),
        mat(p.q),
        mat(p.k),
        mat(p.v),
        mat(p.o),
        mat(p.attn_gate),
        mat(p.mlp_gate),
        mat(p.mlp_up),
        mat(p.mlp_down),
    ]
}

/// **The partition at PRODUCTION widths, with no device — where the fixture's arithmetic lies.**
///
/// Two reviews found that every boundary assertion above is checked at widths where the 1 MiB
/// alignment slack is 99.6% of the budget, so several of them cannot fail there. The worked
/// example: `assert!(floor > global + layer)` is documented as catching "a floor that forgot the
/// slots", and at the fixture's 416-byte globals and 1,920-byte layers a floor charging ZERO
/// slots is still 1,048,992 > 2,336 — it passes. At the shipped widths it would go red.
///
/// So this runs the same arithmetic against the real config. Deviceless and therefore the second
/// thing CI covers, which matters because CI has no rocm job at all.
#[test]
fn the_partition_arithmetic_holds_at_the_shipped_widths() {
    let cfg: gm::GlimmerConfig = serde_json::from_str(fx::GLIMMER_SHIPPED_CONFIG).unwrap();
    let cfg = cfg.text;
    let layer = cfg.layer_bytes().unwrap();
    let global = cfg.global_bytes();
    let want = cfg.resident_bytes().unwrap();
    let floor = cfg.floor_bytes().unwrap();

    // Re-derived by hand from config.json, not quoted from any doc — the figure in this repo's
    // prose was 967.889 MB in four places, which is the CHECKPOINT's bf16-norm arithmetic; the
    // artifact widens the four norms to f32, so a layer is 53,248 bytes larger. Both reviews
    // caught it, and `resident_bytes`' 55.712 GB only reconciles with this value.
    assert_eq!(layer, 967_942_144, "per-layer bytes at the shipped widths");
    assert_eq!(global, 5_379_352_576, "embed + lm_head + final norm");
    assert_eq!(want, 55_712_344_064, "the whole text side");
    assert_eq!(cfg.n_layers, 52);

    // The floor assertion that is vacuous at fixture widths and load-bearing here.
    assert!(
        floor > global + layer,
        "the floor must cover the globals and more than one layer: {floor} vs {}",
        global + layer
    );
    // Every budget in the band below all-resident must pin fewer than every layer, and must ask
    // for what it uses rather than for the whole budget — the over-allocation review found.
    for b in [floor, floor + layer, want, want - 1] {
        let (pinned, capacity) = cfg.partition(Some(b)).unwrap();
        assert!(pinned < cfg.n_layers, "{b} must leave something streaming");
        assert!(
            capacity <= b,
            "budget {b} asked for {capacity}, more than it was given"
        );
        assert_eq!(
            capacity,
            global + (pinned + gm::GLIMMER_STREAM_SLOTS) * layer + 1_048_576,
            "budget {b} must request exactly the globals, its {pinned} pinned layers, its slots \
             and the alignment slack"
        );
    }
}

/// **A single tensor larger than `i32::MAX` BYTES, through the allocator the PIN uses.**
///
/// `lm_head.weight` is `[202048, 6656]` bf16 = **2,689,662,976 bytes = 1.252x `i32::MAX`**, and
/// R1 is where a budgeted pin first has to place it beside a partition.
///
/// > **Repointed 2026-08-12 by review, and the first version tested the wrong allocator.** It
/// > exercised `DeviceBuf::new` + `copy_in_at` + `copy_out` — `hipMalloc` and `hipMemcpy`. The pin
/// > places every tensor through `DeviceTier::place`, which bump-allocates inside a `VmmBuf`
/// > (`rivoli_vmm_alloc`) and fills it with `ptr::copy_nonoverlapping`. Not one of those three
/// > calls is on the pin's path, so 2.69 GB of GTT was being spent proving something about code
/// > R1 does not run, under a doc that said "R1 is where it first gets placed for real".
///
/// Sentinels straddle `i32::MAX` in both directions: a 32-bit truncation of a LENGTH copies a
/// prefix and leaves the tail untouched, a truncation of an OFFSET wraps to the start, and the
/// two are distinguishable only by probing on both sides of the boundary.
#[test]
#[cfg(feature = "rocm")]
#[ignore = "allocates 2.69 GB of GTT; run explicitly under the GPU flock with --ignored"]
fn a_tensor_past_i32_max_bytes_survives_the_placement() {
    const N: usize = 202_048 * 6656 * 2;
    const { assert!(N > i32::MAX as usize, "the point of this test is the size") };
    let mut host = vec![0u8; N];
    let probes = [
        0usize,
        (i32::MAX as usize) - 1,
        i32::MAX as usize,
        (i32::MAX as usize) + 1,
        N / 2,
        N - 2,
        N - 1,
    ];
    for (i, &p) in probes.iter().enumerate() {
        assert!(p < N, "probe {p} is outside the {N}-byte tensor");
        host[p] = (i + 1) as u8;
    }
    assert!(
        probes.iter().any(|&p| p > i32::MAX as usize),
        "at least one probe must sit past i32::MAX or this test checks nothing"
    );

    // The pin's own path: one tier, one `place`, read back through the returned pointer.
    let mut tier = rivoli::memory::device::DeviceTier::new(N + (1 << 20)).unwrap();
    let ptr = tier.place(&host).unwrap();
    // Safe for the reason `glimmer_pin.rs` gives: the tier is a host-fillable VMM mapping, so the
    // pointer `place` returns is readable here — which is also exactly why `Slot::fill` can write
    // through it.
    let back = unsafe { std::slice::from_raw_parts(ptr as *const u8, N) };
    for (i, &p) in probes.iter().enumerate() {
        assert_eq!(
            back[p],
            (i + 1) as u8,
            "byte at offset {p} ({:.3} GB in) did not survive `DeviceTier::place` — a 32-bit cast \
             anywhere on that path would look exactly like this",
            p as f64 / 1e9
        );
    }
    println!(
        "{N} bytes ({:.3} GB) placed and read back through DeviceTier, {} probes",
        N as f64 / 1e9,
        probes.len()
    );
}
