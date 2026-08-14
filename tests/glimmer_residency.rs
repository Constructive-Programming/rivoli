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
    converted("glimmer-residency-cfg", DIM).2
}

/// A converted Glimmer artifact at `dim`: its temp root, its `out` directory, and its text config.
///
/// **The `TempRoot` is RETURNED, not dropped**: it owns the directory the returned path names, so a
/// caller that binds it to `_` deletes the artifact it is about to open.
fn converted(tag: &str, dim: usize) -> (TempRoot, String, gm::GlimmerTextConfig) {
    let root = TempRoot::new(tag);
    let _ = glimmer_convert_fixture(root.path(), dim);
    let dir = root
        .join("out")
        .to_str()
        .expect("utf-8 temp path")
        .to_string();
    let cfg: gm::GlimmerConfig = gm::load_config(&dir).unwrap();
    (root, dir, cfg.text)
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
    let want = cfg.resident_bytes(gm::GlimmerFormat::Bf16).unwrap();
    let layer = cfg.layer_bytes(gm::GlimmerFormat::Bf16).unwrap();
    let floor = cfg.floor_bytes(gm::GlimmerFormat::Bf16).unwrap();
    assert_eq!(cfg.n_layers, L, "the fixture's layer count moved");
    // Kept for shape, but it is the SHIPPED-widths twin that can fail: at 416-byte globals and
    // 1,920-byte layers a floor charging zero slots is still 1,048,992 > 2,336. Review, 2026-08-12.
    assert!(
        floor > cfg.global_bytes(),
        "the floor must cover the globals"
    );

    assert_eq!(
        cfg.partition(None, gm::GlimmerFormat::Bf16).unwrap().0,
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
        let (pinned, capacity) = cfg.partition(Some(b), gm::GlimmerFormat::Bf16).unwrap();
        assert_eq!(pinned, L, "a budget of {b} must pin every layer");
        // And it must not ASK for more than the all-resident set. `DeviceTier::new` allocates
        // its capacity rather than treating it as a ceiling, and also feeds `guard_capacity`, so
        // an over-request both wastes GTT and can turn a workable budget into a refusal.
        assert_eq!(
            capacity, all_resident,
            "an over-generous budget must request only what it uses"
        );
    }

    let (pinned, _) = cfg.partition(Some(floor), gm::GlimmerFormat::Bf16).unwrap();
    assert_eq!(
        pinned, 0,
        "exactly the floor buys the globals and the slots, and no layers"
    );

    let e = format!(
        "{:#}",
        cfg.partition(Some(floor - 1), gm::GlimmerFormat::Bf16)
            .unwrap_err()
    );
    // `"what is LEFT for weights"` replaced `"Weights only"` on 2026-08-14. The old fragment
    // pinned a disclaimer that was FALSE for the number beside it — both callers hand `partition`
    // a budget `weight_budget` has already taken the KV cache out of, so "KV ... on top of this"
    // told the operator to add back what had just been subtracted. The assertion's job is
    // unchanged: the refusal has to say what the figure it quotes actually is.
    for fragment in [
        "below this artifact's floor",
        "read once per",
        "what is LEFT for weights",
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
            cfg.partition(Some(floor + k * layer), gm::GlimmerFormat::Bf16)
                .unwrap()
                .0,
            k,
            "floor + {k} layers must pin {k}"
        );
        assert_eq!(
            cfg.partition(Some(floor + k * layer - 1), gm::GlimmerFormat::Bf16)
                .unwrap()
                .0,
            k - 1,
            "one byte short of {k} layers must pin {}, not round up",
            k - 1
        );
    }
    // The crossover itself, asserted rather than merely avoided: at `k + SLOTS == n_layers` the
    // budget must pin EVERYTHING and drop the slots.
    let (pinned, capacity) = cfg
        .partition(Some(floor + crossover * layer), gm::GlimmerFormat::Bf16)
        .unwrap();
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
    let (_root, dir, cfg) = converted("glimmer-residency-bytes", DIM);
    let dir = dir.as_str();
    let layer = cfg.layer_bytes(gm::GlimmerFormat::Bf16).unwrap();
    let floor = cfg.floor_bytes(gm::GlimmerFormat::Bf16).unwrap();

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
    let bytes = |ptr: *const u8, n: usize| unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };
    // One or two blobs per projection, matching what `GlimmerLayerPin::addrs` hands `Slot::fill`:
    // an fp8 projection's SCALE GRID is a placement too, and a byte-identity gate that compared
    // only the weights would pass a streamed layer whose scales came from the wrong layer — every
    // magnitude wrong by a per-tile factor, every shape right.
    let mat = |w: rivoli::memory::pin::GlimmerProj| -> Vec<Vec<u8>> {
        let [o, i] = w.dims();
        match w {
            rivoli::memory::pin::GlimmerProj::Bf16(w) => {
                vec![bytes(w.packed as *const u8, o * i * 2)]
            }
            rivoli::memory::pin::GlimmerProj::Fp8(w) => {
                let b = rivoli::artifact::quant::FP8_BLOCK;
                vec![
                    bytes(w.packed, o * i),
                    bytes(w.scale as *const u8, o.div_ceil(b) * i.div_ceil(b) * 4),
                ]
            }
        }
    };
    // `hidden` is recovered from a projection's input dim rather than passed in, so this
    // helper needs nothing but the pin.
    let hidden = p.q.dims()[1];
    let mut v = vec![
        f32s(p.input_ln, hidden),
        f32s(p.post_attn_ln, hidden),
        f32s(p.pre_ffn_ln, hidden),
        f32s(p.post_ffn_ln, hidden),
    ];
    for w in [
        p.q,
        p.k,
        p.v,
        p.o,
        p.attn_gate,
        p.mlp_gate,
        p.mlp_up,
        p.mlp_down,
    ] {
        v.extend(mat(w));
    }
    v
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
    let layer = cfg.layer_bytes(gm::GlimmerFormat::Bf16).unwrap();
    let global = cfg.global_bytes();
    let want = cfg.resident_bytes(gm::GlimmerFormat::Bf16).unwrap();
    let floor = cfg.floor_bytes(gm::GlimmerFormat::Bf16).unwrap();

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
        let (pinned, capacity) = cfg.partition(Some(b), gm::GlimmerFormat::Bf16).unwrap();
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

// ---- S3 item 0: the write-after-read fence -------------------------------------------------

/// The width this one fixture runs at, and it is not [`DIM`].
///
/// At `DIM` = 8 the largest matrix in a layer is **16x8**, and a gemm that small retires before the
/// host has finished the `write_bytes` behind it — so the hazard this test exists to catch could
/// not be made to happen at any row count. **Width buys kernel DURATION, and only duration keeps
/// the disturbance inside a live kernel**; [`FENCE_ROWS`] buys the same thing along the other axis
/// and neither substitutes for the other, which the first design got wrong. At 256, `q` is [512, 256] and a
/// layer is **1,839,104 bytes** (six 262,144-byte projections, two 131,072-byte KV ones, four f32
/// norms — re-derived from `layer_bytes`, not estimated; an earlier "~1.3 MB" here was neither).
///
/// The fixture is fully parametric in `dim` (`glimmer_fixture`: heads 2, kv 1, head_dim = dim,
/// inter = 2·dim), so this costs one wider temp checkpoint (~5 MB) and one more `convert_glimmer`
/// subprocess, which three other tests in this file already pay.
#[cfg(feature = "rocm")]
const FENCE_DIM: usize = 256;

/// How many output rows ONE gemm computes while the slot is overwritten under it.
///
/// > **This was `FENCE_LAUNCHES`, 4096 separate launches, and that design was a COIN FLIP — measured
/// > 2026-08-12.** The idea was that enqueueing many launches leaves some pending when the host
/// > writes. It does not: a `[512, 256]` gemm costs the device about what a `hipLaunchKernel` costs
/// > the host, so the queue never grows and whether anything is still in flight at the moment of the
/// > write is scheduling noise. Two runs of the SAME binary gave 2732 of 4096 and **0 of 4096** — the
/// > second one tripping the very assert that exists to catch a red proof that stopped proving.
/// >
/// > One launch with `m` rows fixes it by construction: the host returns from the launch in
/// > microseconds and the kernel then runs for **4-6.5 ms, measured**, so the write lands inside a
/// > live kernel rather than hopefully — `polluted_rows` asserts that with a timestamp. (A first
/// > version justified "milliseconds" with "~1 GB of weight traffic", which is the cache-blind
/// > upper bound: `q` is 262 KB and fits in cache, so DRAM traffic is ~256 KB and the figure
/// > supported nothing. Review, 2026-08-12.)
/// > **A racing gate has to be made deterministic, not made likely** — this repo's rule that a proof
/// > which refuses to go red is itself evidence cuts both ways, and a proof that goes red at random
/// > is worth no more than one that never does.
#[cfg(feature = "rocm")]
const FENCE_ROWS: usize = 4096;

/// One gemm computing `FENCE_ROWS` rows against layer 0's `q`, then `disturb` while it runs, then a
/// sync and a count of how many rows differ from a reference taken with nothing else in flight.
///
/// One launch, not one per row — [`FENCE_ROWS`] carries that argument and the measurement behind it.
///
/// **The arms share everything a shared function can make them share, and NOT their timing.** An
/// earlier version of this doc claimed they "differ only in `disturb`"; review showed that is false
/// in two ways. Arm A's `pin.layer(0)` here is a HIT (the test body already resolved layer 0) while
/// arm B's is a MISS with a full twelve-tensor fill in front of it, and arm A's disturbance is one
/// `write_bytes` issued the instant the launch returns while arm B's walks a bounds check, the slot
/// map, the sync under test, and four norm memcpys before it reaches `q` — the fifth tensor in
/// `GLIMMER_LAYER_TENSORS`.
///
/// That difference matters: **arm A does not evidence that arm B's window is open.** What does is
/// the no-fence column of the table below, where arm B corrupts all 4096 rows — a measurement of arm
/// B's own latency profile, on this machine. Arm A's job is narrower and still worth its lines: it
/// shows the FIXTURE can produce the hazard at all, which is the thing that decays silently.
#[cfg(feature = "rocm")]
fn polluted_rows(
    pin: &mut rivoli::memory::pin::GlimmerPin,
    x: &rivoli::memory::device::DeviceBuf,
    out: &rivoli::memory::device::DeviceBuf,
    delay: std::time::Duration,
    disturb: impl FnOnce(&mut rivoli::memory::pin::GlimmerPin, &rivoli::memory::pin::Bf16Weight),
) -> (usize, std::time::Duration) {
    let q = common::bf16_of(pin.layer(0).unwrap().q);
    let o_dim = q.o_dim;
    let (xp, op) = (x.ptr() as *const f32, out.ptr() as *mut f32);
    let gemm = |rows: usize| {
        // SAFETY: `x` is `rows * i_dim` live f32, `q.packed` is `o_dim * i_dim` live u16 inside the
        // tier, `out` is `rows * o_dim` writable f32, none aliasing another. Null stream: this
        // test's subject is what happens when nothing orders the HOST against a launch.
        unsafe {
            fx::gemm_bf16_launch(xp, q.packed, op, rows, o_dim, q.i_dim, std::ptr::null_mut())
        };
    };
    // The reference, one row, synced before anything can move.
    gemm(1);
    rivoli::backend::hip::device_sync().unwrap();
    let want = fx::f32v(&fx::back(out))[..o_dim].to_vec();
    // All-zero and no overwrite can change it; NON-FINITE and every `!=` below is true regardless,
    // which is how this gate first went red on a fixture that could not produce a finite product at
    // all. See `bf16_blob`.
    assert!(
        want.iter().all(|v| v.is_finite()) && want.iter().any(|v| *v != 0.0),
        "the reference product is all-zero or non-finite, so the comparisons below cannot mean \
         what they claim"
    );

    // Time the launch so the claims about WHEN the disturbance lands are measurements. A launch
    // that retires before `disturb` runs makes arm A a write-BEFORE-read, which corrupts the same
    // bytes but does not exercise the ordering the fence is about — review found the first version
    // asserting "inside a live kernel by construction" with nothing timing it (2026-08-12).
    let t0 = std::time::Instant::now();
    gemm(FENCE_ROWS);
    let launched = t0.elapsed();
    std::thread::sleep(delay);
    disturb(pin, &q);
    let disturbed = t0.elapsed();
    rivoli::backend::hip::device_sync().unwrap();
    let total = t0.elapsed();
    println!(
        "  launch returned at {launched:?}, disturbance done at {disturbed:?}, kernel drained at \
         {total:?}"
    );
    // **The discriminator, and it is a timestamp rather than an argument.** A total divergence count
    // cannot tell "the write landed inside a live kernel" from "the write landed on an idle device
    // because the kernel had already retired" — different events, same number. This settles it:
    // the disturbance must complete before the kernel drains. Review found the first version
    // asserting the property in prose ("inside a live kernel by construction") with nothing timing
    // it, 2026-08-12. Measured on gfx1151: launch returns at ~27 µs, the raw write is done by ~35 µs,
    // the kernel drains at ~6.5 ms — the write lands 0.5% into the kernel's life.
    //
    // > **CORRECTED 2026-08-13, found by two independent reviews of S3 item 4's sibling gate.**
    // > The assert below is a TAUTOLOGY, not a discriminator: `disturbed` and `total` are two reads
    // > of the same monotonic clock taken in program order with a `device_sync` between them, so
    // > `disturbed < total` is a theorem of the code and cannot go red for any input or any device
    // > state. The paragraph above claims it "settles" the idle-device case; it does not.
    // >
    // > It is kept, and NOT strengthened to a ratio, because the two arms want different bounds and
    // > the function is shared: arm A's write finishes ~35 µs into a ~6.5 ms kernel (a 180x margin a
    // > ratio bar would check), while arm B's `layer()` refill spends the whole kernel inside
    // > `device_sync` and finishes 0.35 µs before the drain — 1.0001x — so any ratio that validates
    // > A red-lines B by construction. **What actually evidences arm B's window is the no-fence
    // > column of the table on `a_slot_refill_cannot_land_under_a_live_kernel`**, which is what that
    // > doc already says. `tests/glimmer_stream_order.rs` had the same shape and could fix it, by
    // > comparing against a duration measured in a DIFFERENT arm rather than a later clock read.
    assert!(
        disturbed < total,
        "the disturbance finished at {disturbed:?} but the kernel had already drained by {total:?} \
         — this arm measured a write to an IDLE device, not the write-after-read hazard"
    );
    let got = fx::f32v(&fx::back(out));
    let bad = (0..FENCE_ROWS)
        .filter(|i| got[i * o_dim..(i + 1) * o_dim] != want[..])
        .count();
    (bad, total)
}

/// **The hazard `GlimmerPin::layer`'s invariant describes, made to happen — and then closed.**
///
/// A slot refill is a host `memcpy` and kernel launches are asynchronous, so a host that runs one
/// layer ahead of the device overwrites weights a live GEMV is still reading. The symptom is
/// position-dependent nondeterministic wrong text, this repo's arena-relocation signature, and no
/// finite slot count prevents it — only a dependency does. `GLIMMER_STREAM_SLOTS` was cut from 2 to
/// 1 on exactly that argument, which left the fence owed and is what this pays.
///
/// **Two arms, and arm A is why arm B's green means anything.**
///
/// * **Arm A writes the slot from the HOST directly**, through the same tier pointers `tensors_of`
///   reads, while the gemm runs — an unfenced fill, spelled out. It asserts the outputs DIVERGE.
///   That is a standing red proof: it fires every run, and if it ever stops firing it says so
///   instead of letting arm B pass for the wrong reason.
/// * **Arm B goes through `pin.layer()`**, the shipped path, and asserts NOTHING diverges. Without a
///   fence inside `layer` this arm is arm A with extra steps.
///
/// An arm-B-only test is the false-green this repo keeps writing: passing could mean "the fence
/// works" or "the race did not fire". An arm A that fires only sometimes is no better, which is what
/// [`FENCE_ROWS`] records.
///
/// **MEASURED 2026-08-12 on gfx1151, both directions, from two binaries differing only in the fence:**
///
/// | | arm A (raw host write) | arm B (`layer()` refill) | |
/// |---|---|---|---|
/// | fence removed | 4096 of 4096 | **4096 of 4096**, disturbance at 233 µs of a 3.84 ms kernel | FAILED |
/// | fence present | 4096 of 4096 | **0 of 4096**, 2 fills | ok |
///
/// **The fenced run's timestamps are the fence's own fingerprint**, better evidence than the row
/// count: with it, arm B's disturbance completes at **3.8796 ms** against a kernel draining at
/// **3.87995 ms** — 0.35 µs apart, because `layer()` spent that entire 3.88 ms inside `device_sync`
/// waiting for the launch. Without it the same disturbance finishes at 233 µs and the kernel runs on
/// for another 3.6 ms with its weights already zeroed.
///
/// The no-fence column is what makes arm B's zero a measurement rather than a hope: the identical
/// sequence, on the identical fixture, differing only by one `device_sync` in `GlimmerPin::layer`,
/// corrupts every row.
#[test]
#[cfg(feature = "rocm")]
fn a_slot_refill_cannot_land_under_a_live_kernel() {
    use rivoli::memory::pin::GlimmerPin;
    use std::time::Duration;
    let (_root, dir, cfg) = converted("glimmer-fence", FENCE_DIM);
    let dir = dir.as_str();

    // The floor pins ZERO layers, so every layer streams and layers 0 and 1 map to the same slot.
    // That is the configuration the hazard needs and the one a tight budget produces.
    let floor = cfg.floor_bytes(gm::GlimmerFormat::Bf16).unwrap();
    let mut pin = GlimmerPin::build(dir, &cfg, Some(floor)).unwrap();
    // **Both of these are load-bearing, not sanity checks.** If the floor pinned layer 0, `q` would
    // be a PINNED pointer, `layer(1)` would refill a slot in unrelated memory, and every arm below
    // would report 0 while looking healthy — the margin is only 1.75x (1 MiB of slack against a
    // 1,839,104-byte layer), so a smaller FENCE_DIM crosses it. And every claim here about layers 0
    // and 1 sharing a slot is a consequence of the slot COUNT: at 2 they land on different slots and
    // arm B goes silently vacuous. Both were residual risks a review had to point out rather than
    // read off the test (2026-08-12).
    assert_eq!(pin.pinned_layers(), 0, "the floor must pin nothing");
    assert!(L >= 2, "the hazard needs two layers to map to one slot");
    assert_eq!(
        gm::GLIMMER_STREAM_SLOTS,
        1,
        "this gate's premise is that layers 0 and 1 share a slot, which holds only at one slot"
    );
    let [o_dim, i_dim] = pin.layer(0).unwrap().q.dims();
    // `x` carries FENCE_ROWS identical activation rows: the gemm reads `x + r*k` per row, so a
    // single row would read past the allocation for every r > 0.
    let x = fx::dev(&fx::f32b(&vec![1.0f32; FENCE_ROWS * i_dim]));
    let out = fx::zeros(FENCE_ROWS * o_dim * 4);

    // ---- arm A: the unfenced fill, spelled out, twice ----
    // The raw write, byte-for-byte what `Slot::fill` does to `q` — same mapping, no synchronization
    // of any kind.
    // SAFETY (both uses): `q.packed` is `o_dim * i_dim` live u16 in a host-fillable VMM mapping
    // owned by the pin, which outlives the write.
    let zap = |_: &mut GlimmerPin, q: &rivoli::memory::pin::Bf16Weight| {
        unsafe { std::ptr::write_bytes(q.packed as *mut u16, 0, q.o_dim * q.i_dim) };
    };
    let (diverged, kernel) = polluted_rows(&mut pin, &x, &out, Duration::ZERO, zap);
    println!("arm A: {diverged} of {FENCE_ROWS} rows, kernel drained in {kernel:?}");
    assert!(
        diverged > 0,
        "an unsynchronized host write during a live {FENCE_ROWS}-row gemm changed NOTHING. FIRST \
         CHECK `AMD_SERIALIZE_KERNEL` and `HIP_LAUNCH_BLOCKING` — either makes the launch block, \
         which defeats this gate and no fixture size can fix it. Otherwise the fixture can no \
         longer produce the hazard and arm B proves nothing: raise FENCE_ROWS or FENCE_DIM rather \
         than deleting this assert"
    );

    // > **The window is the kernel's FIRST FETCH, not its lifetime — measured 2026-08-12 and it
    // > refuted the model this gate was built on.** An arm A2 was added on review's suggestion:
    // > delay the same write to the kernel's midpoint and assert a STRICTLY PARTIAL count, which
    // > only a mid-kernel write could produce. It measured **0 of 4096 after a 3.24 ms delay into a
    // > 6.47 ms kernel** — a write halfway through changes nothing at all. The reason is coherence,
    // > not scheduling: `q` is 262 KB, the GPU pulls it into cache in the kernel's first microseconds
    // > and never re-reads it, so a host write after that point is invisible to the launch. The
    // > premise both review and I were reasoning from — that waves fetch weights progressively
    // > across the kernel's life, so the corrupted fraction tracks the write's position — is simply
    // > false here.
    // >
    // > So the arm was deleted rather than kept green by relaxing its assert: what it would test is
    // > a cache property of one GPU, and the question it existed to settle (is the kernel LIVE when
    // > the write lands?) is answered deterministically by the timestamp assert above. The practical
    // > consequence is worth carrying: the hazard window is tens of microseconds wide and its
    // > consequence is total, which is why "it will probably have finished by then" is not a defence
    // > anywhere in S3's loop.

    // ---- arm B: the shipped path ----
    // A fresh pin: the arms above left the slot zeroed, and `slot_layer` still claims layer 0, so
    // reusing one would take the hit path and score against zeros.
    drop(pin);
    let mut pin = GlimmerPin::build(dir, &cfg, Some(floor)).unwrap();
    let (bad, _) = polluted_rows(&mut pin, &x, &out, Duration::ZERO, |p, _| {
        // Layer 1 maps to the same slot, so this is a refill — the one `layer` must fence.
        let _ = p.layer(1).unwrap();
    });
    assert_eq!(
        bad, 0,
        "{bad} of {FENCE_ROWS} rows read weights that `layer()` overwrote under them"
    );
    // **Exactly two**, not `>= 2`: this file's neighbour records the review that established the
    // rule, because `> 0` also passes for a slot map that thrashes. Both fills are derivable —
    // `polluted_rows`' own `layer(0)` on the fresh pin, and `disturb`'s `layer(1)`.
    let (_, fills) = pin.slot_stats();
    assert_eq!(fills, 2, "expected exactly two slot fills, got {fills}");
    println!("arm B: {FENCE_ROWS} rows, 0 polluted, {fills} fills");
}
