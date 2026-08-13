//! **S3 item 4: the stream a call site hands a launcher, and what a null one costs.**
//!
//! # The item's own premise was wrong, and that is the finding
//!
//! `docs/investigations/glimmer-integration.md` item 4 reads "`sigmoid_gate` and `logit_softcap`
//! take the trailing stream now; the loop passes its compute stream at both call sites. A null
//! there is the unordered-read bug `linalg.hip`'s swiglu note describes, **and no fixture can see
//! it**". Half of that is already true (both launchers took the parameter at S2), half is
//! unbuildable (there is no layer loop, so there is no call site to hold to anything) — and the
//! clause in bold is **false**. This file is the fixture. It reproduces the bug on **more than
//! 99.9% of elements in every run**, deterministically enough to assert on.
//!
//! # What it measured, gfx1151, 2026-08-13
//!
//! A 3.7-4.9 ms `gemm_bf16` on a real stream writes the consumer's operand; the consumer is
//! enqueued **6-42 µs** later — inside the producer's first 1% — on the null stream or on the
//! producer's own.
//!
//! | | elements disagreeing with the ordered answer, of 2,097,152 |
//! |---|---:|
//! | consumer on the **null stream** | **2,095,272 - 2,097,152 (99.91-100.00%)**, 14 measurements |
//! | consumer on the **producer's stream** | **0**, every run |
//!
//! Both launchers, same shape. The red proof is the launcher dropping its stream argument
//! (`(hipStream_t)stream` → `(hipStream_t)0` in `kernels/fwd.hip`, run and reverted): the stream
//! arm goes from 0 to **2,097,152** and **2,095,292**, so the green column is the stream being
//! honoured and not the race failing to fire.
//!
//! **The 0.08% that is not stale is the honest part of the number.** `gemm_bf16` is one wave per
//! output element over a 262,144-block grid, so its earliest blocks retire in microseconds and a
//! consumer dispatched at 7 µs finds a few hundred elements already written. That is why the
//! assertion is `> 0` rather than a fraction: the mechanism guarantees staleness, the exact count
//! is scheduling.
//!
//! # What this does NOT gate, and it is the important half
//!
//! **Nothing here says a call site passes the right stream, because there is no call site.** These
//! two launchers have no `src/` caller at all — `tests/kernel_coverage.rs`'s OWNERS rows carry
//! empty slices, and that census fires the moment one appears. What this file adds is the price
//! tag: whoever fills those rows in can read what a null costs rather than take it on the comment.
//!
//! **And the contract the plan states is too weak — the item is not "non-null at these two
//! sites".** A compute stream at exactly these two launches inside an otherwise null-stream layer
//! is the SAME bug inverted, and it satisfies the plan's wording. What has to hold is that every
//! launch touching one buffer is on one stream, or separated by an explicit event or sync.
//!
//! That is not statically checkable, and `src/f4gpu.rs` is the proof: it deliberately mixes four
//! streams in one function, and its header records `launch_hc_pre`, `launch_hc_post` and
//! `launch_moe_acc_drain` as taking a stream and being HANDED NULL on purpose — "correct today,
//! because everything around them is null-stream, so a non-null one would reorder against the
//! norms". Same reasoning, opposite conclusion, from the same premise. So there is no rule to
//! mechanise here beyond the census, and pretending otherwise would put a gate on the loop that
//! is wrong about the engine it already has.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::backend::Stream;
use rivoli::backend::hip::{device_sync, launch_logit_softcap, launch_sigmoid_gate};
use rivoli::memory::device::DeviceBuf;
use std::ffi::c_void;
use std::time::Instant;

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

/// What the producer's destination holds BEFORE it runs. A consumer that reads it has read
/// pre-production values.
///
/// **The separation between the two answers is CHECKED, not argued** — `score`'s `blind` count
/// asserts that no element's ordered answer equals its unproduced one, because an element where
/// they agreed would score a stale read as *correct* and quietly shrink the red arm.
///
/// That check is doing real work here, and the argument a reader would expect is FALSE: these
/// products span **-46.769 to +88.152**, std 22.20 over all 2,097,152 of them (measured on the
/// host at these exact widths, 2026-08-13), so **188,778 are more negative than `STALE`** and
/// `sigmoid` maps them BELOW `sigmoid(-30)` = 9.36e-14 rather than above it. What actually holds
/// is exact equality: not one product is exactly -30.0, and both kernels are injective, so
/// neither can map a real product onto the stale answer.
const STALE: f32 = -30.0;

/// Muse Glimmer's own two, from `glimmer_head.rs` — the launcher refuses `mult >= cap`.
const MULT: f32 = 0.196_116_14;
const CAP: f32 = 20.0;

/// One long producer on a real stream, one consumer, and the buffers both touch.
///
/// `dst` is the producer's output and the consumer's operand — `g` for [`launch_sigmoid_gate`],
/// `x` in place for [`launch_logit_softcap`]. `x` is the gate's value operand and the softcap
/// ignores it; carrying it here rather than in one test is what lets both launchers travel the
/// same four runs, which is the only reason their counts are comparable.
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
}

impl Rig {
    fn new() -> Self {
        // `fill` is the fixture's deterministic pseudo-random spread; salts differ so the
        // activation and the weight are not the same sequence.
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

    /// Producer, consumer, join — with `joined` deciding whether the host waits BETWEEN them.
    ///
    /// The unjoined arms assert the consumer was submitted while the producer was still running.
    /// A total-divergence count cannot tell "the consumer read early" from "the producer had
    /// already retired", which is `glimmer_residency.rs`'s lesson one hazard over: the timestamp
    /// is what makes the arm mean what it claims.
    fn run(
        &mut self,
        joined: bool,
        stream: *mut c_void,
        consume: &dyn Fn(&Rig, *mut c_void),
        observe: &dyn Fn(&Rig) -> Vec<f32>,
    ) -> Vec<f32> {
        self.reset();
        let t0 = Instant::now();
        self.produce();
        if joined {
            device_sync().expect("producer join");
        }
        consume(self, stream);
        let enqueued = t0.elapsed();
        device_sync().expect("join");
        let total = t0.elapsed();
        println!("    consumer enqueued at {enqueued:?}, device idle at {total:?}");
        if !joined {
            assert!(
                enqueued < total,
                "the consumer was enqueued at {enqueued:?} but the device was already idle by \
                 {total:?} — this arm measured a launch onto an IDLE device, not the \
                 unordered-read hazard"
            );
        }
        observe(self)
    }

    /// The producer's four arms for one consumer.
    fn arms(
        &mut self,
        consume: &dyn Fn(&Rig, *mut c_void),
        observe: &dyn Fn(&Rig) -> Vec<f32>,
    ) -> Arms {
        let null = std::ptr::null_mut();
        let raw = self.s.raw();
        Arms {
            want: self.run(true, null, consume, observe),
            unordered: self.run(false, null, consume, observe),
            ordered: self.run(false, raw, consume, observe),
            unproduced: {
                self.reset();
                consume(self, null);
                observe(self)
            },
        }
    }
}

/// The claims every launcher's arms have to support, scored and asserted the same way.
///
/// One function rather than a tail in each test: jscpd rejected the second copy, and it is right
/// about the substance too — a red arm and a green arm are only comparable if the same code
/// decides what "wrong" counts as.
fn score(name: &str, a: &Arms) {
    assert!(
        a.want.iter().all(|v| v.is_finite()),
        "{name}: the ordered answer is not finite, so every comparison below is a comparison \
         against noise"
    );
    // **The discriminator has to be able to fire everywhere, or a zero count is ambiguous.** If the
    // ordered answer and the unproduced one agreed anywhere, an element that read stale would be
    // scored correct and the red arm would under-count for a reason that has nothing to do with
    // ordering. Checked rather than argued: the same class of hole made this repo's fence gate go
    // green on a fixture that could only produce NaN.
    let blind = (0..OUT).filter(|&i| a.want[i] == a.unproduced[i]).count();
    assert_eq!(
        blind, 0,
        "{name}: {blind} of {OUT} elements have the same value produced or not, so the counts \
         below cannot see a stale read there"
    );
    let bad = |v: &[f32]| (0..OUT).filter(|&i| v[i] != a.want[i]).count();
    let (u, o) = (bad(&a.unordered), bad(&a.ordered));
    println!(
        "  {name}: null stream {u} of {OUT} wrong ({:.2}%), its own stream {o}",
        100.0 * u as f64 / OUT as f64
    );
    assert!(
        u > 0,
        "{name}: the null-stream arm computed the ordered answer everywhere — the race did not \
         fire, so this gate proves nothing and must not be read as evidence that null is safe"
    );
    assert_eq!(
        o, 0,
        "{name}: the SAME enqueue pattern with the stream passed got {o} of {OUT} elements wrong, \
         so the launcher is not honouring its stream argument"
    );
}

/// **The hazard `kernels/fwd.hip` describes for both launchers, made to happen.**
///
/// The gate sits between the attend and the o_proj GEMV. Give it the compute stream and it reads
/// what the attend wrote; give it null and rivoli's `hipStreamNonBlocking` streams carry no
/// implicit ordering against it, so it reads whatever was in the buffer when it was dispatched.
#[test]
fn a_null_stream_gate_reads_its_operand_before_the_producer_writes_it() {
    let mut rig = Rig::new();
    let arms = rig.arms(
        &|r, s| {
            // SAFETY: `x` and `dst` are two distinct `OUT`-element live f32 allocations (the
            // kernel's parameters are `__restrict__`), both outliving the join in `run`. The
            // stream is null or `r.s`, and which one it is IS the subject.
            unsafe { launch_sigmoid_gate(r.x.ptr() as *mut f32, r.dst.ptr() as *const f32, OUT, s) }
                .expect("sigmoid_gate launch")
        },
        &|r| sync_read(&r.x),
    );
    score("sigmoid_gate", &arms);
}

/// The same, one launch later in the model: the softcap is in place on the head's output, so an
/// unordered call is a write-WRITE race and the surviving value is the uncapped logit.
///
/// That is the worst spelling of this bug in the tree. `logit_softcap` is the one operation every
/// greedy gate here is provably blind to (the anchor measured `softcap_off` leaving `emitted.ids`
/// bit-identical), so a null stream at this call site silently deletes it and nothing downstream
/// of a decode can tell.
#[test]
fn a_null_stream_softcap_is_overwritten_by_the_head_it_was_meant_to_cap() {
    let mut rig = Rig::new();
    let arms = rig.arms(
        &|r, s| {
            // SAFETY: `dst` is `OUT` writable live f32 outliving the join in `run`; `MULT` and
            // `CAP` satisfy the launcher's guards (both finite, positive, `MULT < CAP`).
            unsafe { launch_logit_softcap(r.dst.ptr() as *mut f32, OUT, MULT, CAP, s) }
                .expect("logit_softcap launch")
        },
        &|r| sync_read(&r.dst),
    );
    score("logit_softcap", &arms);
}
