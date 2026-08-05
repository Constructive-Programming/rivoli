//! The `.f4` routed streaming pool, end to end on the shipped artifact.
//!
//! **Needs the GPU** — a pool is a VMM allocation plus an io_uring reaper writing into it —
//! so this follows the repo's GPU-test idiom (`tests/v4_pin.rs`, `tests/vk.rs`): it simply
//! runs, and fails when another tenant holds the device. The host-only half of the same
//! path (what an `.f4` set reports about itself, what `TierFmt` derives) is in
//! `tests/v4_loading.rs` and stays there.
//!
//! What this exists to establish, in order of how much it would cost to get wrong:
//!
//! 1. **The bytes in a resolved slot are that expert's bytes.** Everything else in the pool
//!    is bookkeeping over an address; this is the only assertion that connects
//!    `(layer, expert)` to the 13,369,344 bytes the kernel will decode. Compared against an
//!    independent `pread` of the file at the offset `read_spec` names — not against anything
//!    the pool computed.
//! 2. **Absolute layer ids.** The `l3-5` fixture is the only one that can catch the read
//!    table's `layer - first_layer` being wrong; over `0..3` the subtraction is the identity.
//! 3. **Eviction actually happens.** V4's routed set is 137 GiB against a ~115 GiB budget,
//!    so a V4 decode streams by construction. A pool that silently held everything would
//!    pass every other test here and tell us nothing about the case that matters.
//!
//! Deliberate breaks that proved each of these can fail are recorded in
//! `docs/investigations/v4-flash-port.md`.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
// **`rocm`, not `any(rocm, vulkan)`**, and for two reasons rather than one. The readback
// below reads a DEVICE address as a host pointer, which only unified addressing makes legal;
// and `Stream::new()` is HIP's signature — the Vulkan `Stream` takes a device, a queue family
// and a timeline semaphore, so this file would not compile there even without the readback.
// The pool itself is backend-neutral; nothing here claims a Vulkan parity that has not been
// measured, which is `tests/kernel_coverage.rs`'s standing rule for this port.
#![cfg(feature = "rocm")]

use rivoli::artifact::format::RoutedFmt;
use rivoli::artifact::model::{V4Config, load_config};
use rivoli::artifact::quant::{f4_expert_stride, f4_slot_offsets};
use rivoli::backend::{Stream, device_sync};
use rivoli::fetch::asyncfetch::Ticket;
use rivoli::memory::pin::V4Pin;
use rivoli::memory::routed::ExpertSlot;

#[path = "common/v4_artifact_dir.rs"]
mod v4_artifact_dir;

/// A device budget for a fixture run: the pin sizes its resident tier off the artifact and
/// the pool gets the rest, so this is `resident + pool`.
///
/// **5 GiB, and the margin is the whole point.** The first draft used 12 GiB, which against
/// a 3-layer fixture's 2.43 GiB resident leaves a 9.56 GiB pool for a 9.56 GiB routed set:
/// short by **0.49 of one 13.37 MB expert**, so the eviction test below passed by evicting
/// exactly one key, and only because that key happened to be the LRU at that moment. The
/// comment defending it compared a binary figure against a decimal one ("~9.5 GiB of pool
/// for a 10.27 GB routed set") so a 0.06% margin read as 8%.
///
/// At 5 GiB the pool is ~2.5 GiB against the same 9.56 GiB — under 27% residency, so the
/// sweep must evict most of what it touches. The test asserts the margin rather than
/// trusting this constant, because a fixture whose `resident.safetensors` grows would move
/// it silently.
const CAPACITY: usize = 5 << 30;

/// Submit `sel` at `layer`, wait for every ticket, and return the resolved slots.
///
/// The waits go on a real stream and are followed by a full `device_sync`, because the
/// reaper signals its timeline from ANOTHER THREAD when the io_uring read lands — a
/// `device_sync` alone would return before any of that happened and the readback below would
/// see whatever was in the slot. This is the host-side spelling of what `gpu.rs`'s expert
/// loop does per launch.
fn submit_and_land(pin: &mut V4Pin, stream: &Stream, layer: usize, sel: &[usize]) -> Vec<ExpertSlot> {
    let (mut out, mut fmt, mut tickets) = (Vec::new(), Vec::new(), Vec::new());
    pin.routed
        .submit(layer, sel, &[], &[], &mut out, &mut fmt, &mut tickets)
        .unwrap_or_else(|e| panic!("submit layer {layer}: {e:#}"));
    assert_eq!(out.len(), sel.len());
    assert!(
        fmt.iter().all(|&f| f == RoutedFmt::F4),
        "a V4 pool's tiers are both `.f4`; anything else would dispatch the wrong kernel"
    );
    for &t in &tickets {
        pin.routed.wait_on(t, stream.raw()).unwrap();
    }
    device_sync().unwrap();
    out
}

/// Read one expert's block straight from the file, at the offset the set's own `read_spec`
/// names. Independent of the pool: this is the `pread` the comparison needs to be against
/// something other than the code under test.
///
/// **Opened fresh and BUFFERED, deliberately.** The obvious shortcut — reuse the fd
/// `read_spec` hands back — is wrong twice: that fd is `O_DIRECT` (`format.rs::open_direct`),
/// so a read into a plain `Vec<u8>` succeeds only while the allocator happens to return a
/// page-aligned block, and borrowing or duplicating it puts a second owner on a descriptor
/// the streamer is using. The offset is what the test needs from `read_spec`, not the
/// descriptor. No `unsafe` at all this way.
fn block_from_disk(dir: &str, pin: &V4Pin, layer: usize, expert: usize) -> Vec<u8> {
    use std::os::unix::fs::FileExt;
    let (_, begin, len) = pin.f4.read_spec(layer, expert).unwrap();
    let f = std::fs::File::open(format!("{dir}/L{layer:02}.f4")).unwrap();
    let mut buf = vec![0u8; len];
    f.read_exact_at(&mut buf, begin as u64).unwrap();
    buf
}

/// The slot the pool resolved, read back as bytes.
///
/// **Reads the DEVICE address as a host pointer, which is legal only under HIP.** Unified
/// addressing on gfx1151 makes `VmmBuf::ptr_mut` and `host_mut` the same number
/// (`memory/device.rs` says so and says why relying on it elsewhere is forbidden). A test is
/// the one place the coincidence may be used: the alternative is a `pub fn read_slot` on
/// `RoutedPool` whose only caller is this file, which is the callerless-helper shape this
/// port has already had to delete twice. Under Vulkan the two are unrelated addresses, so
/// this whole file is gated on `rocm` — see the module attribute.
fn slot_bytes(base: *const u8, len: usize) -> &'static [u8] {
    // SAFETY: `base` is inside the pool VMM, which is host-coherent under HIP unified
    // addressing and lives as long as the pin; `len` is one expert block, which the slot
    // stride covers. Called after `device_sync`, so no copy is in flight into it.
    unsafe { std::slice::from_raw_parts(base, len) }
}

/// Bytes of routed experts in `n_layers` of this model — the number the whole design is
/// against. 43 layers is 137 GiB; a 3-layer fixture is 9.56 GiB. Returned in BYTES and
/// converted at each use, because the mis-stated margin this replaced came from a GiB
/// figure being compared against a GB one.
fn routed_bytes(cfg: &V4Config, n_layers: usize) -> usize {
    n_layers * cfg.n_experts * f4_expert_stride(cfg.hidden, cfg.moe_inter)
}

/// One fixture, opened: the pin, a stream to enqueue ticket waits on, and the two things the
/// cases index by. A struct because the three cases below take exactly these and `jscpd`
/// refuses the third copy of the parameter list — which is the right call, since a
/// `(&V4Config, &mut V4Pin, &Stream, usize, &str)` list is four same-shaped references and a
/// transposition in it would compile.
struct Case<'a> {
    cfg: &'a V4Config,
    pin: &'a mut V4Pin,
    stream: &'a Stream,
    layer: usize,
    dir: &'a str,
}

fn open(dir: &str) -> (V4Config, V4Pin) {
    let cfg: V4Config = load_config(dir).unwrap();
    let pin = V4Pin::build(dir, &cfg, CAPACITY, "2q", Default::default(), None)
        .unwrap_or_else(|e| panic!("{dir} must load with a pool: {e:#}"));
    (cfg, pin)
}

/// The `l0-2` fixture with a pool, a stream to enqueue ticket waits on, and its first layer
/// id. `None` when this machine has no artifact — the caller returns, like every other
/// artifact-gated test in this tree. One function because the four-line preamble was
/// identical in three tests, which `build.rs`'s duplication gate refuses.
fn l0_2() -> Option<(V4Config, V4Pin, Stream, usize, String)> {
    let dir = v4_artifact_dir::v4_artifact("L00.f4")?;
    let (cfg, pin) = open(&dir);
    let layer = pin.range().start;
    Some((cfg, pin, Stream::new().unwrap(), layer, dir))
}

/// **The load-bearing one: a resolved slot holds that expert's bytes, at the offsets the
/// descriptor will be built from.**
///
/// Three separate claims, because a weaker one is satisfied by a wrong pool:
///  * the whole block matches the file, byte for byte — not a checksum, so a permutation
///    inside it cannot pass;
///  * each of the six `ExpertSlot` addresses sits at the `.f4` slot offset from the block
///    base, so `w3`'s scales are not `w2`'s;
///  * two different experts land in DIFFERENT slots holding DIFFERENT bytes, which is what
///    a pool that resolved every key to slot 0 would fail.
fn a_resolved_slot_holds_that_experts_bytes_at_the_f4_offsets(c: &mut Case<'_>) {
    let Case { cfg, pin, stream, layer, dir } = c;
    let (cfg, layer, dir) = (&**cfg, *layer, &**dir);
    // Not 0..k: expert ids that are far apart in the file, so a block-offset error of one
    // stride is visible rather than landing on a neighbour with similar content.
    let sel = [0usize, 7, 128, cfg.n_experts - 1];
    let out = submit_and_land(pin, stream, layer, &sel);

    let off = f4_slot_offsets(cfg.hidden, cfg.moe_inter);
    let (_, _, len) = pin.f4.read_spec(layer, 0).unwrap();
    for (i, &e) in sel.iter().enumerate() {
        let want = block_from_disk(dir, pin, layer, e);
        assert_eq!(want.len(), len);
        let base = out[i].gate.packed;
        assert_eq!(
            slot_bytes(base, len),
            &want[..],
            "layer {layer} expert {e}: the streamed slot is not the file's block"
        );
        // The six addresses, against the layout — `gate.packed` IS the block base, so these
        // are deltas and a wrong `slot_offsets` moves at least one.
        let got = [
            out[i].gate.packed, out[i].gate.scale,
            out[i].up.packed, out[i].up.scale,
            out[i].down.packed, out[i].down.scale,
        ];
        for (k, &p) in got.iter().enumerate() {
            assert_eq!(
                p as usize - base as usize,
                off[k],
                "expert {e}: projection address {k} is not at the .f4 slot offset"
            );
        }
    }
    // Distinct experts, distinct slots, distinct bytes. Experts 0 and 128 of a real layer
    // are independently trained, so equal content would mean the pool aliased them.
    assert_ne!(out[0].gate.packed, out[2].gate.packed, "two experts share a slot");
    assert_ne!(
        slot_bytes(out[0].gate.packed, 4096),
        slot_bytes(out[2].gate.packed, 4096),
        "two experts' slots hold the same bytes"
    );
}

/// **Absolute layer ids through the pool, on the only fixture that can see them.**
///
/// `l3-5` holds layers 3..6, so the read table's row for layer 3 is row 0. A pool that used
/// the absolute id — or subtracted the wrong first layer — would read layer 5's expert for
/// layer 3 and fail nothing else in this file. The comparison is against the FILE, resolved
/// through the same `read_spec` the streamer used, so it is the bytes and not the arithmetic
/// that is checked.
fn the_pool_streams_by_absolute_layer_id_on_a_set_that_starts_at_three() {
    let Some(dir) = v4_artifact_dir::v4_artifact_l3_5("L03.f4") else { return };
    let (cfg, mut pin) = open(&dir);
    let stream = Stream::new().unwrap();
    let range = pin.range();
    assert_eq!(range, 3..6, "this fixture is the non-zero-start case");

    // The SAME expert id in every layer: the block offset is a function of the expert only,
    // so if the layer→file mapping were wrong these would be byte-identical to each other.
    let mut seen: Vec<Vec<u8>> = Vec::new();
    for l in range.clone() {
        let out = submit_and_land(&mut pin, &stream, l, &[11]);
        let (_, _, len) = pin.f4.read_spec(l, 11).unwrap();
        let want = block_from_disk(&dir, &pin, l, 11);
        assert_eq!(
            slot_bytes(out[0].gate.packed, len),
            &want[..],
            "layer {l} expert 11 streamed the wrong block"
        );
        seen.push(want);
    }
    assert_ne!(seen[0], seen[1], "layers 3 and 4 must not hold the same expert 11");
    assert_ne!(seen[1], seen[2], "layers 4 and 5 must not hold the same expert 11");

    // **Outside the range is refused, and the refusal leaves the pool UNCHANGED.** The
    // second half is the one that had a bug: `submit` used to range-check in phase 1c, after
    // phase 1b had admitted the key, taken an arena slot and counted a miss — so the failed
    // call left `resident(2, 0)` answering true, and a re-submit took the HIT path and
    // handed back an `ExpertSlot` for a slot no read ever targeted. Found by review
    // 2026-08-05; asserting only `is_err()` passes either way, which is why it did.
    let (mut o, mut f, mut t) = (Vec::new(), Vec::new(), Vec::new());
    let (h, m) = (pin.routed.hits(), pin.routed.misses());
    assert!(
        pin.routed.submit(2, &[0], &[], &[], &mut o, &mut f, &mut t).is_err(),
        "layer 2 is below this artifact's range and must not resolve"
    );
    assert!(!pin.routed.resident(2, 0), "a refused submit admitted the key anyway");
    assert_eq!(
        (pin.routed.hits(), pin.routed.misses()),
        (h, m),
        "a refused submit moved the counters"
    );
    // The same shape one dimension over: an expert id past `n_experts`. On any row but the
    // last, `row * n_experts + expert` lands inside the read table on a LATER layer's row, so
    // a `table.get()` alone returns Ok with the wrong layer's fd. Layer 3 is row 0 of 3 here,
    // so this is exactly that case.
    let e = cfg.n_experts;
    assert!(
        pin.routed.submit(3, &[e], &[], &[], &mut o, &mut f, &mut t).is_err(),
        "expert {e} is past n_experts and must not resolve to layer 4's expert 0"
    );
    assert!(!pin.routed.resident(3, e));

    // A selection wider than `submit`'s fixed `[bool; MAX_BATCH]` hit scratch. Refused, not
    // indexed past — and it must be an `ensure!` and not a `debug_assert!`, because under
    // `--release` (which every recorded benchmark runs) a `debug_assert!` is compiled out and
    // this becomes an out-of-bounds panic mid-decode. V4 has 256 experts, so 33 is reachable.
    let wide: Vec<usize> = (0..33).collect();
    let e = format!(
        "{:#}",
        pin.routed
            .submit(3, &wide, &[], &[], &mut o, &mut f, &mut t)
            .err()
            .expect("33 experts must be refused by the 32-slot batch scratch")
    );
    assert!(e.contains("batch scratch"), "got: {e}");
}

/// The pool refuses a budget it cannot make progress on, rather than failing mid-run with
/// `arena NeedFree after policy eviction — byte-accounting bug`, which blames the arena for
/// the user's `--max-mem`.
///
/// The number is the batch, not `top_k`: V4's are equal (single-row FP4 kernel), GLM's are
/// not. Sized from the artifact's own resident footprint so the case is reached by a budget
/// too small for six experts and not by one too small for the resident set.
fn a_budget_too_small_for_one_batch_is_refused_at_build() {
    let Some(dir) = v4_artifact_dir::v4_artifact("L00.f4") else { return };
    let cfg: V4Config = load_config(&dir).unwrap();
    // Resident + 16 MiB slack + room for five of the six experts one layer demands.
    let resident = std::fs::metadata(format!("{dir}/resident.safetensors")).unwrap().len() as usize;
    let stride = f4_expert_stride(cfg.hidden, cfg.moe_inter);
    let capacity = resident + (16 << 20) + 5 * stride;
    let e = format!(
        "{:#}",
        V4Pin::build(&dir, &cfg, capacity, "2q", Default::default(), None)
            .err()
            .expect("a pool too small for one batch must be refused at build")
    );
    assert!(
        e.contains("cannot hold one batch"),
        "the refusal must name the batch bound, got: {e}"
    );
}

/// **A re-submitted expert is a HIT, and one that has been evicted is not.**
///
/// This is the property the whole 137-GiB-against-115 design rests on, and it is the one a
/// pool can fake: a pool that never evicted would report all-hits and look better. So both
/// directions are asserted — the immediate re-submit must hit, and a sweep large enough to
/// exceed the budget must force the first batch back out.
///
/// The eviction sweep is sized from the pool's own arithmetic rather than from a guess: at
/// `CAPACITY` the fixture leaves ~9.5 GiB for a 10.27 GB routed set, so touching every
/// expert of every layer must evict.
fn the_pool_hits_on_a_resubmit_and_evicts_when_the_working_set_exceeds_the_budget(c: &mut Case<'_>) {
    let Case { cfg, pin, stream, layer, .. } = c;
    let (cfg, layer) = (&**cfg, *layer);
    let sel = [3usize, 9, 200];

    let (h0, m0) = (pin.routed.hits(), pin.routed.misses());
    submit_and_land(pin, stream, layer, &sel);
    assert_eq!(
        (pin.routed.hits() - h0, pin.routed.misses() - m0),
        (0, sel.len() as u64),
        "a cold pool must miss every expert — an all-hit first batch would mean the \
         residency map is answering for bytes that were never read"
    );

    // **The premise, asserted rather than assumed.** Everything below is about what happens
    // when the working set exceeds the pool; if it does not, the sweep evicts nothing and
    // the failure message blames the pool for a property the fixture never asked of it.
    let (budget, routed) = (pin.routed.budget(), routed_bytes(cfg, pin.range().len()));
    assert!(
        budget * 2 < routed,
        "this fixture must be oversubscribed by a real margin, not by a rounding: pool \
         {:.2} GiB vs routed set {:.2} GiB",
        budget as f64 / (1u64 << 30) as f64,
        routed as f64 / (1u64 << 30) as f64,
    );

    let (h1, m1) = (pin.routed.hits(), pin.routed.misses());
    submit_and_land(pin, stream, layer, &sel);
    assert_eq!(
        (pin.routed.hits() - h1, pin.routed.misses() - m1),
        (sel.len() as u64, 0),
        "the same batch submitted twice must hit"
    );
    assert!(sel.iter().all(|&e| pin.routed.resident(layer, e)));

    // Now push the working set past the budget: every expert of every layer, against a pool
    // that the assertion above proved holds under half of them.
    for l in pin.range() {
        for chunk in (0..cfg.n_experts).collect::<Vec<_>>().chunks(cfg.top_k) {
            submit_and_land(pin, stream, l, chunk);
        }
    }
    // **Counted over every key, not sampled on three.** `!sel.iter().all(resident)` passes
    // when the sweep evicted exactly ONE key and it happened to be one of the three — which
    // is what the 12 GiB first draft of this test actually did. Residency has to be bounded
    // by the arithmetic: at most `budget / stride` slots can be occupied, whatever the policy
    // does, and the pool holds under half the set.
    let slots = budget / f4_expert_stride(cfg.hidden, cfg.moe_inter);
    let live = pin
        .range()
        .flat_map(|l| (0..cfg.n_experts).map(move |e| (l, e)))
        .filter(|&(l, e)| pin.routed.resident(l, e))
        .count();
    assert!(
        live <= slots,
        "{live} keys resident in a pool of {slots} slots — the pool is not bounded by its \
         budget, which is the one property a 137 GiB routed set against a 115 GiB machine \
         depends on"
    );
    assert!(
        live * 2 < pin.range().len() * cfg.n_experts,
        "a sweep of {:.2} GiB left {live} of {} keys resident — this fixture is not \
         oversubscribed and the test is measuring nothing",
        routed as f64 / (1u64 << 30) as f64,
        pin.range().len() * cfg.n_experts,
    );
    // And an evicted key re-reads correctly rather than resolving to a stale slot.
    let out = submit_and_land(pin, stream, layer, &sel);
    assert_eq!(out.len(), sel.len());
}

/// A resident expert carries `Ticket::RESIDENT`; a missing one does not. INV-5's encoding,
/// asserted on the V4 path because `gpu.rs`'s launch loop branches on `is_resident` to
/// choose a stream, and a pool that handed out `RESIDENT` for a slot still being written
/// would put the kernel on the compute stream with no wait at all.
fn a_miss_carries_a_real_ticket_and_a_hit_carries_the_resident_one(c: &mut Case<'_>) {
    let Case { pin, stream, layer, .. } = c;
    let layer = *layer;
    let (mut o, mut f, mut t) = (Vec::new(), Vec::new(), Vec::new());

    pin.routed.submit(layer, &[42], &[], &[], &mut o, &mut f, &mut t).unwrap();
    assert!(!t[0].is_resident(), "a cold expert must carry a real dependency");
    pin.routed.wait_on(t[0], stream.raw()).unwrap();
    device_sync().unwrap();

    pin.routed.submit(layer, &[42], &[], &[], &mut o, &mut f, &mut t).unwrap();
    assert_eq!(t[0], Ticket::RESIDENT, "a hit must carry the satisfied ticket");
}

/// **ONE `#[test]`, and that is not tidiness — it is the only thing that keeps the device
/// from being oversubscribed.**
///
/// libtest runs `#[test]` fns on parallel threads. Each case below builds a `V4Pin`, and a
/// `V4Pin` is a `DeviceTier` allocation plus a pool VMM plus an io_uring ring — so five of
/// them start at once, five tiers compete for a budget sized for one, and five `AsyncFetch`
/// reapers race. Run that way on 2026-08-05 it **wedged the device**: 19 threads, two in
/// `kfd_wait_on_events`, **four `io_sq_thread`s** (the tell — four rings, four pools), zero
/// test output in 12 minutes, and it had to be killed by PID. That is CLAUDE.md's recorded
/// `gpustream` hang, and here it was self-inflicted.
///
/// `--test-threads=1` fixes it and is the wrong fix: it lives in whoever remembers to type
/// it, and the failure mode for forgetting is a wedged sole-tenant GPU. `tests/v4_pin.rs`
/// already made this call — one test, an internal loop over fixtures — so this follows it.
/// The cost is coarser test names; every assertion below carries its own message, which is
/// what a failure actually needs.
///
/// **Order is load-bearing.** The `l0-2` cases share one pin to avoid re-reading its 2.6 GB
/// resident set four times, so the residency-destroying sweep goes LAST: `hits_on_a_resubmit`
/// asserts a cold pool misses everything, which a preceding sweep would have falsified.
#[test]
fn the_f4_streaming_pool() {
    // No pin at all: this one asserts a build is REFUSED, so it must not hold the device.
    a_budget_too_small_for_one_batch_is_refused_at_build();
    // Its own artifact and its own pin — the only fixture whose range does not start at 0.
    the_pool_streams_by_absolute_layer_id_on_a_set_that_starts_at_three();

    let Some((cfg, mut pin, stream, layer, dir)) = l0_2() else { return };
    let mut c = Case { cfg: &cfg, pin: &mut pin, stream: &stream, layer, dir: &dir };
    a_resolved_slot_holds_that_experts_bytes_at_the_f4_offsets(&mut c);
    a_miss_carries_a_real_ticket_and_a_hit_carries_the_resident_one(&mut c);
    // LAST: evicts most of the pool.
    the_pool_hits_on_a_resubmit_and_evicts_when_the_working_set_exceeds_the_budget(&mut c);
}
