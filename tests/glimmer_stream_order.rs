//! **S3 item 4: the stream a call site hands a launcher, and what a null one costs.**
//!
//! Run with `--nocapture` to see the counts below; libtest swallows them otherwise. This binary
//! needs `--test-threads=1` like every device arm here, and for a sharper reason than usual: the
//! quantity it measures IS timing, and a sibling test's blocking H2D on the process-wide null
//! stream queues ahead of this one's consumer.
//!
//! # The item's own premise was wrong, and that is the finding
//!
//! `docs/investigations/glimmer-integration.md` item 4 reads "`sigmoid_gate` and `logit_softcap`
//! take the trailing stream now; the loop passes its compute stream at both call sites. A null
//! there is the unordered-read bug `linalg.hip`'s swiglu note describes, **and no fixture can see
//! it**". Half of that is already true (both launchers took the parameter at S2), half is
//! unbuildable (there is no layer loop, so there is no call site to hold to anything) — and the
//! clause in bold is **false**. This file is the fixture.
//!
//! **The first clause is CONFIRMED, not refuted.** A null stream there really is an unordered
//! read. Only "no fixture can see it" fell.
//!
//! # What it measured, gfx1151, 2026-08-13 — 14 measurements over 7 runs
//!
//! A **3.48-3.86 ms** `gemm_bf16` on a real stream writes the consumer's operand; the consumer is
//! enqueued **5.7-34.5 µs** later, **108-656x inside** the producer.
//!
//! | consumer's stream | elements disagreeing with the ordered answer, of 2,097,152 |
//! |---|---:|
//! | **null** | **2,095,258 - 2,097,152 (99.910-100.000%)**, both launchers |
//! | **the producer's** | **0**, every run |
//!
//! The red proof is the launcher dropping its stream argument (`(hipStream_t)stream` →
//! `(hipStream_t)0` in `kernels/fwd.hip`, run and reverted): the stream arm goes from 0 to
//! **2,097,152** and **2,095,292**, so the green column is the stream being honoured and not the
//! race failing to fire. It stays self-red-proving — dropping the argument again trips `o == 0`.
//!
//! **The two launchers fail in DIFFERENT ways, and the counts now say which.** `sigmoid_gate`
//! reads `dst` and writes `x`, so all 2,095,258+ wrong elements are stale reads and **0 are lost
//! writes, every run** — asserted, not observed. `logit_softcap` read-modify-writes `dst`, and
//! there the split inverts: **0-261 elements kept the capped stale value and 2,095,254-2,097,152
//! lost the consumer's write entirely.** So at the head, a null stream does not compute a wrong
//! softcap — it computes the right one and has it overwritten by the raw logits. **The softcap
//! simply does not happen**, which is exactly the failure no greedy gate in this repo can see.
//! (The first version of this file asserted the survivor "is the uncapped logit" in prose and
//! measured nothing; review, 2026-08-13.)
//!
//! **The residue that is not stale is the honest part.** `gemm_bf16` is one wave per output
//! element over a 262,144-block grid, so its earliest blocks retire in microseconds and a consumer
//! dispatched at 10 µs finds a couple of thousand elements already written (**1,894** in the
//! weakest run, 0.090%). Hence the floor is `> OUT/2` rather than a tight fraction: the mechanism
//! guarantees staleness, the exact count is scheduling.
//!
//! # The two assertions are NOT the same kind of claim
//!
//! * `o == 0` is about **rivoli**: `rivoli_sigmoid_gate` and `rivoli_logit_softcap` forward their
//!   `void* stream` to `hipLaunchKernelGGL` rather than dropping it. This is the durable half.
//! * `u > OUT/2` is about **HIP**: a `hipStreamNonBlocking` stream (`kernels/async.hip:20`) carries
//!   no implicit ordering against the legacy null stream. It holds for every kernel, and
//!   `gemm_bf16` stands in for `gqa_attend` only because it runs for 3.8 ms.
//!
//! **If the second half ever goes red for an environmental reason, do not delete the file** — the
//! first half is the only thing in the tree gating that these launchers use their stream at all.
//!
//! # What this does NOT gate, and it is the important half
//!
//! **Nothing here says a call site passes the right stream, because there is no call site.** These
//! two launchers have no `src/` caller — `tests/kernel_coverage.rs`'s OWNERS rows carry empty
//! slices, and that census fires the moment one appears. What this file adds is the price tag.
//!
//! **And the contract the plan states is too weak — the item is not "non-null at these two
//! sites".** A compute stream at exactly these two launches inside an otherwise null-stream layer
//! is the SAME bug inverted, and it satisfies the plan's wording. What has to hold is that every
//! launch touching one buffer is on one stream, or separated by an explicit event or sync.
//!
//! That full invariant needs interprocedural dataflow — which buffer a launcher touches, and where
//! the syncs land — so it does not mechanise. `src/f4gpu.rs` shows both halves of why: `pre_norm`
//! and `layer` hand `launch_hc_pre` / `launch_hc_post` a null on purpose because those functions
//! are null-stream throughout, while `routed_experts` mixes `compute_stream`, `miss_stream` and a
//! null in one body and relies on four explicit `device_sync` boundaries its header enumerates.
//! Null is correct in both, for reasons no grep can see.
//!
//! **A NARROWER rule does mechanise and is not gated today** (found by review, 2026-08-13, left as
//! an open item rather than done here because it edits three engine files this port does not own):
//! `src/backend.rs:86` defines `NULL_STREAM` precisely so a deliberate null reads as a decision,
//! and about fourteen `src/` call sites still pass a bare `std::ptr::null_mut()` in the stream
//! position — including `src/f4gpu.rs:1686`, in the file that imports `NULL_STREAM` and uses it
//! eight lines away. A source census in `kernel_coverage.rs`'s style would cost one test and no
//! device.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::backend::Stream;
use rivoli::backend::hip::{device_sync, launch_logit_softcap, launch_sigmoid_gate};
use rivoli::memory::device::DeviceBuf;
use std::ffi::c_void;
use std::time::{Duration, Instant};

#[path = "common/glimmer_fixture.rs"]
mod fixture;
use fixture::{dev, f32b, gemm_bf16_launch, sync_read};

/// The producer's shape, taken from `glimmer_residency.rs`'s fence gate where it was MEASURED to
/// keep a gemm live for 4-6.5 ms. Duration is the whole point: the consumer has to be enqueued
/// while the producer is still running, or the arm measures enqueue order rather than stream order.
const M: usize = 4096;
const N: usize = 512;
const K: usize = 256;
const OUT: usize = M * N;

/// How far inside the producer's lifetime the consumer must be enqueued for an arm to count.
///
/// Measured 6-42 µs against a 3.7-4.9 ms producer, so 90-800x; this bar keeps an order of
/// magnitude of headroom and still cannot pass on an idle device, where the two are equal.
const INSIDE: u32 = 10;

/// What the producer's destination holds BEFORE it runs. A consumer that reads it has read
/// pre-production values.
///
/// **-1e6 SATURATES BOTH KERNELS, and that is the whole reason for the value.** `expf(1e6)` is
/// `+Inf`, so the gate's `1/(1+Inf)` is exactly `0.0f`; `tanhf(-9806)` is exactly `-1`, so the
/// softcap's answer is exactly `-CAP`. Neither is reachable from a real product: the producer's
/// outputs span **-46.769 to +88.152** (std 22.20 over all 2,097,152, measured on the host at
/// these exact widths), whose sigmoids bottom out at 4.88e-21 — nonzero — and whose softcaps peak
/// at |13.970| against a cap of 20. So the two answers are separated by SATURATION, structurally,
/// for any producer distribution that stays inside f32's exponent range.
///
/// > **This constant was -30.0 for two commits and the argument for it was wrong three times**
/// > (2026-08-13, both reviews). It claimed no product reached it — 188,778 of them are more
/// > negative — then claimed the kernels are injective, which `sigmoid` in f32 is emphatically
/// > not: 381,486 products already map to exactly `1.0f`. What actually held was a **37-ulp**
/// > margin from the single closest product (-29.9999026), i.e. roughly a 5% chance that any
/// > change to `M`/`N`/`K`/either salt would have re-rolled it into a collision and reported the
/// > discriminator as broken. One review proposed `+200` as the fix; that maps to `1.0f` and would
/// > have collided on **381,486 elements**, which is why a proposed constant gets measured before
/// > it gets taken.
const STALE: f32 = -1.0e6;

/// Muse Glimmer's own two, from `glimmer_head.rs` — the launcher refuses `mult >= cap`.
const MULT: f32 = 0.196_116_14;
const CAP: f32 = 20.0;

/// One long producer on a real stream, one consumer, and the buffers both touch.
///
/// `dst` is the producer's output and the consumer's operand — `g` for [`launch_sigmoid_gate`],
/// `x` in place for [`launch_logit_softcap`]. `x` is the gate's value operand; the softcap never
/// reads or writes it, and it lives here only so `Rig` serves both consumers rather than being
/// duplicated. (It costs that test one unused 8 MB refill per arm. What makes the two launchers'
/// counts comparable is [`score`] being one function, not this field.)
struct Rig {
    a: DeviceBuf,
    w: DeviceBuf,
    dst: DeviceBuf,
    x: DeviceBuf,
    s: Stream,
    stale: Vec<u8>,
    ones: Vec<u8>,
}

/// The four runs one launcher gets. `want` is the ordered answer, and the other three are what
/// each way of getting the ordering wrong produces.
struct Arms {
    /// Producer, `device_sync`, consumer — the answer a correct call site computes.
    want: Vec<f32>,
    /// Consumer on the NULL stream, enqueued while the producer runs on `s`. The bug.
    unordered: Vec<f32>,
    /// Consumer on `s`, enqueued the same way. The fix, and the same enqueue pattern.
    ordered: Vec<f32>,
    /// Consumer with NO producer at all — what a fully lost race computes.
    unproduced: Vec<f32>,
    /// How long the producer occupies the device, from the joined arm. The unjoined arms are only
    /// meaningful if their consumer was enqueued well inside this.
    producer: Duration,
}

impl Rig {
    fn new() -> Self {
        let a = dev(&f32b(&fixture::fill(M * K, 1, 1.0)));
        let wb: Vec<u8> = fixture::to_bf16(&fixture::fill(N * K, 2, 1.0))
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let stale = f32b(&vec![STALE; OUT]);
        let ones = f32b(&vec![1.0f32; OUT]);
        Rig {
            a,
            w: dev(&wb),
            dst: dev(&stale),
            x: dev(&ones),
            s: Stream::compute().expect("stream"),
            stale,
            ones,
        }
    }

    /// Both operand buffers back to their pre-production contents, joined before anything runs.
    fn reset(&mut self) {
        self.dst.copy_in_at(0, &self.stale).expect("dst reset");
        self.x.copy_in_at(0, &self.ones).expect("x reset");
        device_sync().expect("reset join");
    }

    /// The long gemm, enqueued on `s` and NOT waited for.
    fn produce(&self) {
        // SAFETY: `a` is `M*K` live f32, `w` is `N*K` live u16, `dst` is `M*N` writable f32, three
        // distinct allocations, all live for the whole test. `s` is a live stream.
        unsafe {
            gemm_bf16_launch(
                self.a.ptr() as *const f32,
                self.w.ptr() as *const u16,
                self.dst.ptr() as *mut f32,
                M,
                N,
                K,
                self.s.raw(),
            )
        };
    }

    /// Producer, consumer, join. `inside` carries the producer's measured duration for the arms
    /// that do NOT join between the two, and its absence marks the arm that measures it.
    ///
    /// **The unjoined arms have to prove their consumer was submitted while the producer was still
    /// running**, or a divergence count cannot tell "the consumer read early" from "the producer
    /// had already retired". An earlier version compared two reads of the same clock taken in
    /// program order (`enqueued < total`), which is a theorem rather than a measurement and could
    /// not go red for any input — both reviews found it, 2026-08-13. Comparing against the
    /// producer's own duration is what makes the arm mean what it claims.
    fn run(
        &mut self,
        inside: Option<Duration>,
        stream: *mut c_void,
        consume: &dyn Fn(&Rig, *mut c_void),
        observe: &dyn Fn(&Rig) -> Vec<f32>,
    ) -> (Vec<f32>, Duration) {
        self.reset();
        let t0 = Instant::now();
        self.produce();
        if inside.is_none() {
            device_sync().expect("producer join");
        }
        let producer = t0.elapsed();
        consume(self, stream);
        let enqueued = t0.elapsed();
        device_sync().expect("join");
        // On the unjoined arms `producer` is only the LAUNCH's return, tens of microseconds — the
        // producer is still running. Printing it as a duration there would read as a 10 µs
        // producer, so the two arms report different things because they measured different things.
        match inside {
            None => println!("    producer alone ran {producer:?}"),
            Some(d) => {
                println!(
                    "    consumer enqueued at {enqueued:?}, {:.0}x inside a {d:?} producer",
                    d.as_secs_f64() / enqueued.as_secs_f64()
                );
                assert!(
                    enqueued * INSIDE < d,
                    "the consumer was enqueued at {enqueued:?}, not inside the first 1/{INSIDE} \
                     of a {d:?} producer — this arm measured a launch onto an idle or nearly-idle \
                     device, not the unordered-read hazard"
                );
            }
        }
        (observe(self), producer)
    }

    /// The producer's four arms for one consumer.
    fn arms(
        &mut self,
        consume: &dyn Fn(&Rig, *mut c_void),
        observe: &dyn Fn(&Rig) -> Vec<f32>,
    ) -> Arms {
        let null = std::ptr::null_mut();
        // **Warm-up, and it is load-bearing.** HIP loads a kernel's code object on its FIRST
        // launch, and paying that inside a timed arm would put the consumer on an idle device. Here
        // rather than relying on the joined arm running first, because that ordering would be
        // enforced by nothing but the order of the struct fields below.
        self.reset();
        self.produce();
        consume(self, null);
        device_sync().expect("warm-up join");

        let raw = self.s.raw();
        let (want, producer) = self.run(None, null, consume, observe);
        let inside = Some(producer);
        Arms {
            want,
            unordered: self.run(inside, null, consume, observe).0,
            ordered: self.run(inside, raw, consume, observe).0,
            unproduced: {
                self.reset();
                consume(self, null);
                observe(self)
            },
            producer,
        }
    }
}

/// The claims every launcher's arms have to support, scored and asserted the same way.
///
/// One function rather than a tail in each test: jscpd rejected the second copy, and it is right
/// about the substance too — a red arm and a green arm are only comparable if the same code
/// decides what "wrong" counts as.
///
/// `in_place` says whether the consumer writes the buffer the producer writes, which changes what
/// a wrong element MEANS. `sigmoid_gate` reads `dst` and writes `x`, so every wrong element is a
/// stale read. `logit_softcap` read-modify-writes `dst`, so an unordered element has three
/// possible survivors — the ordered answer, the raw product (the consumer's write was lost), or
/// the capped stale value (the consumer's write won but read stale) — and the last two are both
/// "wrong" for different reasons. The split is counted and printed rather than asserted in prose,
/// which the first version did: it claimed the survivor "is the uncapped logit" and measured
/// nothing.
fn score(name: &str, in_place: bool, a: &Arms) {
    assert!(
        a.want.iter().all(|v| v.is_finite()) && a.want.iter().any(|v| *v != 0.0),
        "{name}: the ordered answer is all-zero or non-finite, so every comparison below is a \
         comparison against noise"
    );
    // `unproduced` is the other operand of the discriminator, and a NaN there would make every
    // `==` below false and the `blind` count trivially 0 — the exact hole this repo's fence gate
    // fell into. Asserted rather than argued, for the same one line it costs.
    assert!(
        a.unproduced.iter().all(|v| v.is_finite()),
        "{name}: the unproduced answer is not finite, so `blind` below is vacuously 0"
    );
    // **The discriminator has to be able to fire everywhere, or a zero count is ambiguous.** If the
    // ordered answer and the unproduced one agreed anywhere, an element that read stale would be
    // scored correct and the red arm would under-count for a reason that has nothing to do with
    // ordering. See [`STALE`] for why saturation makes this structural rather than lucky.
    let blind = (0..OUT).filter(|&i| a.want[i] == a.unproduced[i]).count();
    assert_eq!(
        blind, 0,
        "{name}: {blind} of {OUT} elements have the same value produced or not, so the counts \
         below cannot see a stale read there"
    );
    let bad = |v: &[f32]| (0..OUT).filter(|&i| v[i] != a.want[i]).count();
    let (u, o) = (bad(&a.unordered), bad(&a.ordered));
    let stale = (0..OUT)
        .filter(|&i| a.unordered[i] == a.unproduced[i])
        .count();
    println!(
        "  {name}: null stream {u} of {OUT} wrong ({:.3}%) — {stale} kept the stale answer, {} \
         lost the consumer's write; its own stream {o}; producer {:?}",
        100.0 * u as f64 / OUT as f64,
        u - stale,
        a.producer
    );
    if !in_place {
        assert_eq!(
            stale,
            u,
            "{name}: the consumer writes a different buffer than the producer, so every wrong \
             element must be a stale read — {} are neither the ordered answer nor the stale one",
            u - stale
        );
    }
    assert!(
        u > OUT / 2,
        "{name}: only {u} of {OUT} elements read stale — the race did not fire, so this gate \
         proves nothing and must not be read as evidence that null is safe. FIRST CHECK the \
         environment: `AMD_SERIALIZE_KERNEL`, `HIP_LAUNCH_BLOCKING`, rocprof/rocgdb with \
         serialized dispatch, or `--test-threads` above 1 all defeat this by construction. NEXT \
         check `kernels/async.hip`: if streams stopped being `hipStreamNonBlocking` they carry \
         implicit ordering against the null stream and this hazard is GONE, which is a green \
         outcome wearing a red hat"
    );
    assert_eq!(
        o, 0,
        "{name}: the SAME enqueue pattern with the stream passed got {o} of {OUT} elements wrong, \
         so the launcher is not honouring its stream argument"
    );
}

/// **The hazard `kernels/fwd.hip` describes for both launchers, made to happen.**
///
/// The gate sits between the attend and the o_proj GEMV, the softcap between the head GEMV and
/// `argmax`. Give either the compute stream and it sees what the producer wrote; give it null and
/// rivoli's `hipStreamNonBlocking` streams carry no implicit ordering against it.
///
/// The softcap is the worse spelling. It is the one operation every greedy gate here is provably
/// blind to (the anchor measured `softcap_off` leaving `emitted.ids` bit-identical), so a null
/// stream at that call site silently deletes it and nothing downstream of a decode can tell.
///
/// ONE test rather than two: both consumers launch on the process-wide null stream, and libtest
/// would otherwise run them concurrently by default — a sibling's blocking H2D queueing ahead of
/// this one's consumer perturbs the only quantity being measured (review, 2026-08-13).
#[test]
fn a_null_stream_consumer_reads_its_operand_before_the_producer_writes_it() {
    let mut rig = Rig::new();
    let gate = rig.arms(
        &|r, s| {
            // SAFETY: `x` and `dst` are two distinct `OUT`-element live f32 allocations (the
            // kernel's parameters are `__restrict__`), both outliving the join in `run`. The
            // stream is null or `r.s`, and which one it is IS the subject.
            unsafe { launch_sigmoid_gate(r.x.ptr() as *mut f32, r.dst.ptr() as *const f32, OUT, s) }
                .expect("sigmoid_gate launch")
        },
        &|r| sync_read(&r.x),
    );
    score("sigmoid_gate", false, &gate);

    let cap = rig.arms(
        &|r, s| {
            // SAFETY: `dst` is `OUT` writable live f32 outliving the join in `run`; `MULT` and
            // `CAP` satisfy the launcher's guards (both finite, positive, `MULT < CAP`).
            unsafe { launch_logit_softcap(r.dst.ptr() as *mut f32, OUT, MULT, CAP, s) }
                .expect("logit_softcap launch")
        },
        &|r| sync_read(&r.dst),
    );
    score("logit_softcap", true, &cap);
}
