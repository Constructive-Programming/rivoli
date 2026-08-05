//! Vulkan kernels vs their CPU oracles — the `tests/kernel.rs` story for the second
//! backend. Compiles to nothing without `vulkan`.
//!
//! Separate file because `tests/kernel.rs` is `#![cfg(feature = "rocm")]` end to end.
//! The helpers below are deliberately the same ones it uses, so the kernel-porting
//! phase can hoist both files onto a shared module instead of rewriting either.
//!
//! # Two rules for whoever writes the next tranche
//!
//! **1. Do not read the Vulkan shader sources while writing an oracle.** Derive it from the
//! HIP original and from `src/math.rs`, which are the specification. An oracle written
//! by someone who has seen the shader is a consistency check wearing a correctness
//! check's clothes: it agrees with the implementation because it was copied from it,
//! and it will happily ratify a shared misreading of the HIP. The `fwd.hip` oracles
//! here were written under that constraint deliberately. It costs a little rework when
//! the two disagree, which is the entire point — that disagreement is the signal.
//!
//! No tripwire enforces this. `tests/oracle_independence.rs` used to grep test sources
//! for a mention of `kernels/vk`, which caught only the laziest spelling of the
//! violation — a copied path — and cost 112 lines and an allowlist to say so. Deleted:
//! the rule is a review obligation, and pretending otherwise bought a green check for
//! the one case that leaves an artifact while the real failure (reading a shader, then
//! writing the oracle from memory) was always invisible to it.
//!
//! **2. A byte-exact oracle must prove its INPUTS are unambiguous.** Where a test
//! compares quantised bytes, the test DATA is a source of cross-driver flake
//! independent of the code: a value landing on a rounding midpoint of the target
//! format can legitimately quantise either way, so the comparison is decided by the
//! driver's arithmetic accuracy rather than by the shader. Green here, red on someone
//! else's machine, shader innocent — and whoever debugs it starts in the kernel,
//! because that is where the failure appears.
//!
//! Use [`assert_quantization_unambiguous`] with the margin the RELEVANT SPEC
//! guarantees, not the accuracy the hardware happens to deliver. Vulkan promises 2.5
//! ULP on `FDiv` and correctly-rounded `FAdd`/`FMul`; a dot product accumulates. This
//! matters most in the fp8 MoE tranche, which compares quantised bytes throughout and
//! is where a seed-dependent flake would be most expensive to diagnose. Make the
//! failure message name the SEED, not the shader.
#![cfg(feature = "vulkan")]
#![allow(clippy::expect_used)]

use rivoli::backend::block_on;
use rivoli::backend::vk::{
    Buf, ExpertDesc, Q, ROWS_PER_BLOCK, VALIDATION_ERRORS, device_sync, fill_u32, gpu,
    launch_append_kv, launch_argmax, launch_attend, launch_embed_i8_row, launch_flag_nonfinite,
    launch_gather_rope, launch_gemv_f32, launch_gemv_fp8, launch_gemv_i8, launch_index_append,
    launch_index_head_route, launch_index_pool_push, launch_index_score, launch_index_topk,
    launch_layernorm, launch_mla_absorb_fp8, launch_mla_value_fp8, launch_moe_acc_drain,
    launch_moe_expert_range, launch_moe_expert_range_i4, launch_rmsnorm, launch_rope,
    launch_swiglu, launch_vadd, launch_vaxpy, memcpy_dtod,
};
use rivoli::math::{E4M3_BLOCK, E4M3_MAX, e4m3_to_f32, f32_to_bf16, f32_to_e4m3, silu};
use rivoli::memory::device::{DeviceBuf, DeviceTier};
use std::sync::atomic::Ordering;

mod common;
use common::{
    Att, Lcg, Mla, MoeRange, assert_close, block_scales, f32b, f32v, gemv_fp8_case, report, u16b,
    want_i8,
};

/// Upload `b` to a fresh device buffer. Backend-typed, so it stays here rather than in
/// `common`: this is `Buf` under Vulkan and `DeviceBuf` under HIP.
fn dev(b: &[u8]) -> Buf {
    let mut d = Buf::new(b.len()).expect("alloc");
    d.write_at(0, b).expect("fill");
    d
}
/// Collects mismatches across shapes so a multi-shape oracle reports ALL of them.
///
/// `assert_close` panics on the first bad shape, which aborts the test and leaves every
/// later shape UNEXECUTED — the same first-failure-abort that used to hide a second
/// broken shader in build.rs, now inside a test. It cost real evidence: gemv_fp8's
/// wave-per-row shape failed and the split-K correctness shape never ran, so "is split-K
/// also wrong on values?" was unanswerable from a red suite.
#[derive(Default)]
struct Shapes {
    bad: Vec<String>,
}

impl Shapes {
    /// Like `assert_close` but records instead of panicking. Always prints the margin.
    /// Shares `err_tol` with it: two copies of a tolerance formula is two tolerances.
    fn close(&mut self, want: &[f32], got: &[f32], label: &str) {
        let (err, tol) = report(want, got, label);
        if err > tol {
            self.bad
                .push(format!("{label}: err={err:.3e} > tol={tol:.3e}"));
        }
    }

    fn assert_all_passed(&self, what: &str) {
        assert!(
            self.bad.is_empty(),
            "\n\n{} of {what}'s shapes failed — ALL of them, not just the first:\n  {}\n",
            self.bad.len(),
            self.bad.join("\n  ")
        );
    }
}

/// Plain f32 GEMV, ascending summation — the same shape as `quant.rs`'s matvec
/// oracles. The kernel reduces in a fixed 32-lane shuffle ladder, so it differs by
/// f32 rounding only, which is what `assert_close`'s tolerance covers.
fn matvec_f32(y: &mut [f32], x: &[f32], w: &[f32], i_dim: usize) {
    for (o, out) in y.iter_mut().enumerate() {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        *out = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// One `gemv_f32` dispatch + sync, returning the raw output bytes.
fn gemv(x: &Buf, w: &Buf, y: &mut Buf, o_dim: usize, i_dim: usize) -> Vec<u8> {
    launch(x, w, y, o_dim, i_dim);
    device_sync().expect("sync");
    read(y, o_dim)
}

/// A `Signal` armed on the FETCH queue, already recorded.
///
/// The two cross-queue tests below both hand-roll this handshake, and it is the thing they
/// are actually testing rather than incidental setup — so it is spelled once, with one
/// panic message, rather than twice with two.
fn armed_on_fetch() -> rivoli::backend::Signal {
    let sig = rivoli::backend::Signal::pending();
    gpu()
        .expect("vulkan init")
        .arm_on(Q::Fetch, &sig)
        .expect("arm on fetch");
    sig
}

/// Enqueue only — no sync. Used to build a multi-dispatch command buffer.
fn launch(x: &Buf, w: &Buf, y: &mut Buf, o_dim: usize, i_dim: usize) {
    // SAFETY: live Buf device addresses of the documented sizes; nothing is dropped
    // before the caller's device_sync.
    unsafe { launch_at(x.ptr(), w.ptr(), y.ptr_mut(), o_dim, i_dim) }
}

/// The same dispatch from BARE device addresses, which is the arm [`launch`] cannot serve:
/// `DeviceTier::place` hands back an address rather than a `Buf`, and two tests deliberately
/// hold a `DeviceBuf`. Those three sites plus `launch` are why the launcher's arguments are
/// spelled here and nowhere else.
///
/// Untyped `u8` addresses, so a caller passes `ptr()`/`ptr_mut()` straight through and the
/// three f32 casts live here with the `unsafe` they justify rather than at each site.
///
/// # Safety
/// `x` must address `i_dim` live f32, `w` `o_dim`·`i_dim`, `y` `o_dim`, all still mapped
/// when the caller next joins.
unsafe fn launch_at(x: *const u8, w: *const u8, y: *mut u8, o_dim: usize, i_dim: usize) {
    let (x, w, y) = (x as *const f32, w as *const f32, y as *mut f32);
    unsafe { launch_gemv_f32(x, w, o_dim, i_dim, 1, y, std::ptr::null_mut()) }.expect("launch");
}

/// Random `w` and `x` for an `o_dim × i_dim` GEMV, and the host result — the opening four
/// lines of five tests below. `w` is drawn BEFORE `x`, which is what makes a seed mean the
/// same data across them.
fn gemv_case(r: &mut Lcg, o_dim: usize, i_dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    matvec_f32(&mut want, &x, &w, i_dim);
    (w, x, want)
}

/// One `moe_acc_drain` on `stream`.
fn drain(
    out: &mut Buf,
    acc: &mut Buf,
    n: usize,
    rows: usize,
    gain: f32,
    stream: *mut std::ffi::c_void,
) {
    // SAFETY: live Buf device addresses holding `rows`·`n` u64 and `n` f32, both outliving
    // the caller's join.
    unsafe {
        launch_moe_acc_drain(
            out.ptr_mut() as *mut f32,
            acc.ptr_mut() as *mut u64,
            n,
            rows,
            gain,
            stream,
        )
    }
    .expect("moe_acc_drain");
}

/// One `gemv_fp8` dispatch + sync, returning `bytes` of `y`.
#[allow(clippy::too_many_arguments)]
fn gemv_fp8(
    x: &Buf,
    p: &Buf,
    s: &Buf,
    y: &mut Buf,
    o_dim: usize,
    i_dim: usize,
    block: usize,
    bytes: usize,
) -> Vec<u8> {
    // SAFETY: live Buf device addresses of the documented sizes — `scale` is
    // ⌈o_dim/block⌉·⌈i_dim/block⌉ f32 — and nothing is dropped before the sync.
    unsafe {
        launch_gemv_fp8(
            x.ptr() as *const f32,
            p.ptr(),
            s.ptr() as *const f32,
            o_dim,
            i_dim,
            block,
            1,
            y.ptr_mut() as *mut f32,
        )
    }
    .expect("launch gemv_fp8");
    sync_readb(y, bytes)
}

/// One `rmsnorm` dispatch + sync, returning `bytes` of `out`.
fn rmsnorm(x: &Buf, w: &Buf, out: &mut Buf, n: usize, eps: f32, bytes: usize) -> Vec<u8> {
    // SAFETY: `x` and `w` are n f32 and `out` is at least `bytes`; all three outlive the sync.
    unsafe {
        launch_rmsnorm(
            x.ptr() as *const f32,
            w.ptr() as *const f32,
            n,
            eps,
            out.ptr_mut() as *mut f32,
        )
    }
    .expect("launch rmsnorm");
    sync_readb(out, bytes)
}

/// The three destination slabs one `append_kv` writes a row into.
///
/// Bundled, with the shape beside it as [`KvShape`], because this dispatch took NINE
/// positional arguments and the reproducibility test forwards all nine through a closure
/// to a second, differently-shaped case. Four bare `usize`s in a row is a transposition
/// the type checker cannot see, and it would move the real case and the decoy together.
struct KvDst<'a> {
    lc8: &'a mut Buf,
    lscale: &'a mut Buf,
    rc: &'a mut Buf,
}

impl<'a> KvDst<'a> {
    fn new(lc8: &'a mut Buf, lscale: &'a mut Buf, rc: &'a mut Buf) -> Self {
        Self { lc8, lscale, rc }
    }
}

/// `(pos, kvl, ropn, n_blocks)` — the row written and the strides it is written with.
type KvShape = (usize, usize, usize, usize);

/// One `append_kv` dispatch + sync.
fn append_kv(lat: &Buf, rop: &Buf, dst: &mut KvDst<'_>, shape: KvShape) {
    let (pos, kvl, ropn, n_blocks) = shape;
    // SAFETY: `lat` is kvl f32 and `rop` ropn f32; the three slabs hold whole rows of their
    // documented stride and row `pos` is in bounds. All five are borrowed for the call.
    unsafe {
        launch_append_kv(
            lat.ptr() as *const f32,
            rop.ptr() as *const f32,
            dst.lc8.ptr_mut(),
            dst.lscale.ptr_mut() as *mut f32,
            dst.rc.ptr_mut() as *mut u16,
            pos,
            kvl,
            ropn,
            n_blocks,
        )
    }
    .expect("launch append_kv");
    device_sync().expect("sync");
}

/// Comparing two buffers of ZEROS and calling it determinism would pass every
/// reproducibility test in this file, so each one ends here.
///
/// Shared by the three whose output is f32 magnitudes. `append_kv`'s and `argmax`'s guards
/// are deliberately NOT this function: append_kv's untouched rows are legitimately zero so
/// only the row it wrote can be checked, and argmax returns an i32 beside an f32 and must
/// not read both as floats. A guard weak enough to cover all five catches none of them.
fn assert_not_all_zero(first: &[u8]) {
    assert!(
        f32v(first).iter().any(|v| v.abs() > 1e-6),
        "output is all zero — the test proves nothing"
    );
}

/// The scratch every guard test dispatches against: one read-only source and one
/// destination. Owning both keeps them mapped across the `unsafe` block, which matters for
/// the ACCEPTED controls — the rejected cases never reach a pointer at all.
struct Scratch {
    src: Buf,
    dst: Buf,
}

impl Scratch {
    /// The pair AND their device addresses, because every caller wants all three and the
    /// owner exists only to keep the two mappings alive. `Buf::ptr` hands back a stored
    /// device address rather than a pointer into the struct, so moving the `Scratch` out of
    /// here does not invalidate what it returned.
    fn new(bytes: usize) -> (Self, *const u8, *mut u8) {
        let mut s = Self {
            src: dev(&vec![0u8; bytes]),
            dst: dev(&vec![0u8; bytes]),
        };
        let (p, q) = (s.src.ptr(), s.dst.ptr_mut());
        (s, p, q)
    }
}

fn read(y: &Buf, n: usize) -> Vec<u8> {
    readb(y, n * 4)
}

/// Join, then read `bytes` of `y` — the launch/join/read-back tail every single-dispatch
/// oracle here ends with. One place to forget the `device_sync`, rather than one per kernel.
fn sync_readb(y: &Buf, bytes: usize) -> Vec<u8> {
    device_sync().expect("sync");
    readb(y, bytes)
}

/// `read` in BYTES. The fwd kernels write u8 and u16 as well as f32, and their
/// contracts are bit patterns rather than numbers, so most of the checks below never
/// leave byte space.
fn readb(y: &Buf, bytes: usize) -> Vec<u8> {
    let mut out = Vec::new();
    y.read_into(&mut out, bytes).expect("out");
    out
}

/// Bytes of guard band allocated past the end of every output buffer.
const GUARD: usize = 256;

/// An output buffer with every byte 0xFF. One poison pattern covers all four formats
/// these kernels emit — f32 NaN, bf16 NaN, e4m3 NaN, and a u8 no scale path produces —
/// so the guard band is the same check everywhere. `f32_to_e4m3` only ever returns
/// 0xFF for a NaN input, and no test here feeds one, so a surviving 0xFF is always
/// "nothing wrote here" rather than a legitimate result.
fn poison(bytes: usize) -> Buf {
    dev(&vec![0xFFu8; bytes])
}

/// Nothing was written at or past byte `end`. Cheap, and the standard — an overrun
/// into an allocation's rounding slack is otherwise invisible.
fn assert_untouched(got: &[u8], end: usize, label: &str) {
    let n = got[end..].iter().filter(|&&b| b != 0xFF).count();
    assert_eq!(n, 0, "{label}: wrote {n} bytes past byte {end}");
}

/// Byte-exact compare, reporting the FIRST difference and how many there are.
/// `assert_eq!` on two multi-kilobyte `Vec<u8>`s dumps both in full and buries the one
/// index that matters.
fn assert_bytes(want: &[u8], got: &[u8], label: &str) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    let diff = want.iter().zip(got).filter(|(a, b)| a != b).count();
    match want.iter().zip(got).position(|(a, b)| a != b) {
        None => println!("{label}: {} bytes exact", want.len()),
        Some(i) => panic!(
            "{label}: {diff} of {} bytes differ; first at {i}: want {:#04x} got {:#04x}",
            want.len(),
            want[i],
            got[i]
        ),
    }
}

/// The repeat/decoy recipe every `*_is_bit_reproducible` test below runs: take `real`'s
/// bytes, then four more times dispatch `decoy` — a differently-shaped launch, so two
/// `real` runs never see identical queue state — and require `real` to reproduce its
/// bytes exactly. `decoy`'s output is dropped; it exists only to perturb scheduling. See
/// `gemv_f32_is_bit_reproducible` for why this is bit equality and not a tolerance.
///
/// Returns `first` rather than checking it: the "not all zero, so the test proves
/// something" guard is NOT shared. `append_kv`'s untouched rows are legitimately zero so
/// only the row it wrote can be checked, `argmax` returns an i32 beside an f32 and must
/// not read both as floats, and the rest compare f32 magnitudes — which also rejects a
/// buffer of denormal noise. A guard weak enough to cover all three catches none of them.
fn assert_bit_reproducible(
    label: &str,
    mut real: impl FnMut() -> Vec<u8>,
    mut decoy: impl FnMut() -> Vec<u8>,
) -> Vec<u8> {
    let first = real();
    for i in 1..5 {
        decoy();
        assert_bytes(&first, &real(), &format!("{label} repeat {i}"));
    }
    first
}

/// Snapshots the validation counter on construction and asserts NOTHING NEW arrived
/// by the time it is checked.
///
/// A delta, not an absolute. `VALIDATION_ERRORS` is process-global and cargo runs this
/// binary's tests on parallel threads, so asserting `== 0` means a message caused by
/// test A fails whichever test happens to read the counter next — the failure names
/// the wrong test and the maintainer debugs the wrong code. That misattribution is
/// worst for exactly the findings that matter here, since a threading diagnostic is
/// caused by the interaction BETWEEN two tests.
///
/// The delta still cannot attribute a concurrent message to its true source; it only
/// stops a test being blamed for one that predates it. Run with `--test-threads=1`
/// when a message actually appears.
struct Validation {
    at_entry: usize,
}

impl Validation {
    fn new() -> Self {
        Self {
            at_entry: VALIDATION_ERRORS.load(Ordering::Relaxed),
        }
    }

    /// Costs nothing when the layer is absent, and prints whether it was there at all
    /// — a green oracle from an unvalidated run must not read like a validated one.
    fn check(&self, label: &str) {
        let g = gpu().expect("vulkan init");
        let now = VALIDATION_ERRORS.load(Ordering::Relaxed);
        let n = now.saturating_sub(self.at_entry);
        println!(
            "{label}: validation layer {}",
            if g.validation() {
                "ON"
            } else {
                "OFF — THIS RUN IS UNVALIDATED"
            }
        );
        assert_eq!(n, 0, "{label}: {n} validation messages during this test");
    }
}

#[test]
fn gemv_f32_matches_oracle() {
    let v = Validation::new();
    let mut r = Lcg(0x3F2);
    // Shapes chosen for the edges, not for realism:
    //   256x5120  the router-gate shape the kernel exists for;
    //   255x512   o_dim not a multiple of ROWS_PER_BLOCK — the tail workgroup has
    //             idle subgroups that must not write;
    //   1x96      a single row, grid of one;
    //   33x97     NEITHER dim is a multiple of WAVE=32, so the strided inner loop
    //             leaves a partial final iteration and lanes diverge before wave_sum;
    //   7x33      both tiny and both ragged.
    for (o_dim, i_dim) in [
        (256usize, 5120usize),
        (255, 512),
        (1, 96),
        (33, 97),
        (7, 33),
    ] {
        let (w, x, want) = gemv_case(&mut r, o_dim, i_dim);
        let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
        // Guard band: allocate a whole extra workgroup's worth of rows and poison
        // them. `o_dim = 255` leaves subgroup 7 of the tail workgroup out of range and
        // it must not write — but at exactly o_dim*4 bytes an overrun lands in the
        // allocation's rounding slack and nothing notices.
        let guard = ROWS_PER_BLOCK as usize;
        let mut yb = dev(&f32b(&vec![f32::NAN; o_dim + guard]));
        launch(&xb, &wb, &mut yb, o_dim, i_dim);
        device_sync().expect("sync");
        let out = f32v(&read(&yb, o_dim + guard));
        assert!(
            out[o_dim..].iter().all(|v| v.is_nan()),
            "gemv_f32 {o_dim}x{i_dim}: wrote past row {o_dim}"
        );
        assert_close(&want, &out[..o_dim], &format!("gemv_f32 {o_dim}x{i_dim}"));
    }
    v.check("gemv_f32_matches_oracle");
}

/// The inter-dispatch barrier in `Gpu::enqueue`, exercised hard enough that its
/// ABSENCE is detectable — and repeated often enough that one execution is a real
/// guard rather than a coin flip.
///
/// SIZING. Measured, not guessed. With the barrier deleted:
///
///   32-step chain, 2048x2048  ->  2 of 8 executions FAIL
///    4-step chain, 2048x2048  ->  0 of 8 (undetectable)
///    2-step chain, 64x96      ->  0 of 8 (undetectable, the original size)
///
/// and 8 of 8 pass in every configuration with the barrier present. So a missing
/// barrier is invisible below some threshold, and the shape that crosses it is: enough
/// workgroups to fill the machine several times over (256 per dispatch here), every
/// output row depending on every input element, and a chain long enough that one
/// escaped hazard survives to the end.
///
/// DETECTION RATE — MEASURED, AND WORSE THAN THE ARITHMETIC PREDICTS.
///
///   1 chain per execution, 7.4 s   ->  2 of 8 executions detect a removed barrier
///   16 chains per execution, 1.8 s ->  1 of 8
///   control (barrier present)      ->  0 of 8 in both configurations
///
/// Sixteen times the chains, HALF the detection. I predicted ~99% from
/// 1 - 0.75^16 and that was wrong, because the repeats are not independent and,
/// worse, because the change that added them also made each chain faster: fixing a
/// 512 MB duplicate-upload waste dropped the runtime 7.4 s -> 1.8 s, and those 32
/// interleaved 16 MB uploads were part of what opened the timing window. Optimising
/// the test removed the conditions that exposed the race.
///
/// So: **~12% detection per execution, measured.** A green run is very weak evidence
/// that a refactor kept the barrier. This test's value is now mostly that it CAN fail
/// at all — it is a smoke alarm with a flat battery, not a guard. Do not treat a
/// passing CI run as proof of ordering.
///
/// If you want a real guard back, the lever is not the repeat count: it is restoring
/// whatever made the chains slow and irregular. Interleaving unrelated large transfers
/// between the dispatches is the obvious candidate, since that is what was removed.
/// Measure the removal arm again after any change here — the whole point is that this
/// number does not follow from reasoning.
///
/// COST. The host oracle (32 chained 2048x2048 matvecs, in a debug build) dominates
/// and is computed once; the repeats add GPU work only. Worth it — this is the only
/// empirical check on the most consequential unverifiable property in the backend,
/// since synchronisation validation on this stack sees no compute hazard class at all
/// (docs/measurement/probes/README.md).
#[test]
fn chained_dispatch_respects_the_barrier() {
    let v = Validation::new();
    let mut r = Lcg(0xBA2);
    let n = 2048usize;
    const STEPS: usize = 32;
    /// Measured at ~12% detection per execution, NOT the ~99% that
    /// 1 - 0.75^REPEATS would suggest — see the docstring. Kept at 16 because it is
    /// cheap (1.8 s) and more chains cannot hurt; the number is not load-bearing.
    const REPEATS: usize = 16;

    // Near-identity: unit diagonal plus small noise. A random matrix chained 32 deep
    // either explodes or collapses into noise, and a corrupted intermediate then hides
    // inside the arithmetic. This keeps every step O(1) so a stale row survives to the
    // output as a visible error.
    //
    // A DISTINCT matrix per step, 32 x 16 MB = 512 MB. THIS IS NOT AN OVERSIGHT AND
    // MUST NOT BE "OPTIMISED" INTO ONE REUSED BUFFER. It was, once: reusing a single
    // matrix left it resident in cache, removed the DRAM traffic between dispatches,
    // took the runtime from 7.4 s to 1.8 s — and HALVED the detection rate, 2/8 to 1/8,
    // because the memory pressure was part of what opened the timing window. The waste
    // is the instrument. Re-measure the removal arm before changing it (see the
    // detection-rate note above); the number does not follow from reasoning.
    //
    // Cheap to build despite the size: only the diagonal and superdiagonal are nonzero,
    // so it is 2n random values over a zeroed allocation, not n^2.
    let mats: Vec<Vec<f32>> = (0..STEPS)
        .map(|_| {
            let mut m = vec![0.0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
                m[i * n + (i + 1) % n] = 0.05 * r.f();
            }
            m
        })
        .collect();
    let x: Vec<f32> = (0..n).map(|_| r.f()).collect();

    let mut want = x.clone();
    for m in &mats {
        let mut next = vec![0.0f32; n];
        matvec_f32(&mut next, &want, m, n);
        want = next;
    }

    let mbufs: Vec<Buf> = mats.iter().map(|m| dev(&f32b(m))).collect();
    let xb = dev(&f32b(&x));
    let mut ping = dev(&f32b(&vec![0.0f32; n]));
    let mut pong = dev(&f32b(&vec![0.0f32; n]));

    for rep in 0..REPEATS {
        launch(&xb, &mbufs[0], &mut ping, n, n);
        for (i, mb) in mbufs.iter().enumerate().skip(1) {
            if i % 2 == 1 {
                launch(&ping, mb, &mut pong, n, n);
            } else {
                launch(&pong, mb, &mut ping, n, n);
            }
        }
        device_sync().expect("sync"); // ONE sync per chain, after all STEPS dispatches
        // Launch k writes ping for even k and pong for odd k, so after STEPS launches
        // the result is in pong when STEPS is even. (Had this inverted first; the test
        // then failed identically with AND without the barrier, which is what caught
        // it — see docs/measurement/probes/README.md, "A test built to fail needs its passing arm
        // checked too".)
        let out = if STEPS.is_multiple_of(2) {
            &pong
        } else {
            &ping
        };
        assert_close(
            &want,
            &f32v(&read(out, n)),
            &format!("{STEPS}-step chain #{rep}"),
        );
    }
    v.check("chained_dispatch");
}

/// Greedy decode must be reproducible run to run, which is a STRICTER property than
/// the oracle's 1e-3 accuracy: a reduction whose order varies with workgroup
/// scheduling passes that tolerance happily. So compare BIT PATTERNS across repeats,
/// with a differently-shaped dispatch interleaved to perturb the scheduler.
///
/// This is why `wave_sum` is a fixed `subgroupShuffleDown` ladder and not
/// `subgroupAdd`. Every kernel with a reduction gets this test as it is ported —
/// `gemv_fp8_splitk` and `mla_attend_combine` especially, since they reduce across
/// workgroup partials in LDS rather than within one subgroup.
#[test]
fn gemv_f32_is_bit_reproducible() {
    let v = Validation::new();
    let mut r = Lcg(0xDE7);
    let (o_dim, i_dim) = (255usize, 1024usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);

    // A decoy of a different shape, dispatched between repeats so the two runs do not
    // see identical queue state.
    let (dx, dw) = (dev(&f32b(&x[..96])), dev(&f32b(&w[..96 * 33])));
    let mut dy = dev(&[0u8; 33 * 4]);

    let first = assert_bit_reproducible(
        "gemv_f32",
        || gemv(&xb, &wb, &mut yb, o_dim, i_dim),
        || gemv(&dx, &dw, &mut dy, 33, 96),
    );
    assert_not_all_zero(&first);
    v.check("gemv_f32_is_bit_reproducible");
}

/// `memcpy_dtod` (arena slot relocation) reading what a dispatch just wrote, in ONE
/// command buffer. The copy's TRANSFER_READ must be ordered after the shader's
/// SHADER_WRITE, which is a COMPUTE -> TRANSFER dependency — a class the barrier in
/// `Gpu::enqueue` did not cover until this test forced it to.
///
/// Run this under `VK_LAYER_VALIDATE_SYNC=1`. The numeric assertion alone is weak
/// here: an unsynchronised copy usually still returns the right bytes on this driver,
/// which is exactly why the hazard is worth a checker rather than an oracle.
#[test]
fn memcpy_dtod_after_dispatch_is_ordered() {
    let v = Validation::new();
    let mut r = Lcg(0xC0B);
    let (o_dim, i_dim) = (128usize, 256usize);
    let (w, x, want) = gemv_case(&mut r, o_dim, i_dim);
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));
    let zb = dev(&f32b(&vec![f32::NAN; o_dim]));

    // Dispatch writes y, then the copy reads y — no sync in between, so both land in
    // the same command buffer and only the barrier separates them.
    launch(&xb, &wb, &mut yb, o_dim, i_dim);
    // SAFETY: both are live Buf device addresses, o_dim*4 bytes, distinct allocations.
    unsafe { memcpy_dtod(zb.ptr() as *mut u8, yb.ptr(), o_dim * 4).expect("dtod") };

    let got = f32v(&read(&zb, o_dim));
    assert_close(&want, &got, "memcpy_dtod after dispatch");
    v.check("memcpy_dtod_after_dispatch_is_ordered");
}

/// The timeline-semaphore Signal resolves, and at what cost.
///
/// The HIP side gets this from `hipLaunchHostFunc` at ~19 us/signal, measured by
/// `gpustream::tests::signal_resolves_and_latency`. Vulkan has no host-callback-on-
/// queue, so this is one waiter thread in `vkWaitSemaphores` instead — and the plan
/// explicitly says to MEASURE whether a thread wakeup beats 19 us rather than assume.
/// The decode path arms roughly 9 per layer over 78 layers per token, so the number
/// decides whether the async overlap is affordable at all.
#[test]
fn timeline_signal_resolves_and_latency() {
    let v = Validation::new();
    let g = gpu().expect("vulkan init");
    // Warm the waiter thread and the first submit.
    block_on(g.signal_on(Q::Main).expect("arm"));

    let n = 200u32;
    let t = std::time::Instant::now();
    for _ in 0..n {
        block_on(g.signal_on(Q::Main).expect("arm"));
    }
    let us = t.elapsed().as_nanos() as f64 / f64::from(n) / 1000.0;
    println!("\nVK TIMELINE-SIGNAL round-trip: {us:.2} us/signal ({n} iters)\n");
    // Sanity ceiling only, matching the HIP test: if a bare signal costs >1 ms the
    // pipeline is a non-starter and the number matters more than the assertion.
    assert!(us < 1000.0, "signal latency {us:.1}us implausibly high");
    v.check("timeline_signal_resolves_and_latency");
}

/// `resolve` is idempotent and immediately ready — `resolve_all` calls it on every waiter of
/// a retired stamp, and a second call must not hang or double-wake.
///
/// It used to also assert `Signal::ready()`, and the doc used to say "the error path in the
/// reaper depends on both, so awaiters never hang". That was true and then quietly stopped
/// being: the ticketed dataflow made the timeline the dependency, so the reaper's error path
/// releases tickets (INV-6) and awaits no signal at all. `ready()` had no callers left and
/// is deleted — `Ticket::RESIDENT` is what says "already present" now.
#[test]
fn signal_resolve_is_idempotent_and_immediate() {
    let s = rivoli::backend::Signal::pending();
    s.resolve();
    s.resolve(); // idempotent
    block_on(s);
}

// ---------------------------------------------------------------------------
// Phase 4 increment 2: the three queues, the ring, timestamps, async staging.
//
// These are the oracles for the CONCURRENCY structure rather than for arithmetic, and
// they exist because the port's headline gate cannot see any of it: token IDs depend on
// arithmetic, not on overlap, so a fully serialised backend passes it (docs/investigations/vulkan-port.md).
// ---------------------------------------------------------------------------

/// A [`Signal`] armed on a queue covers work that was merely RECORDED, not only work
/// already submitted — the gap the command-buffer ring closes.
///
/// This is the property `asyncfetch.rs` depends on and could not have before: it enqueues
/// a copy and arms a signal on the same stream with no `device_sync` between, so if arming
/// covered only submitted work the signal would fire while the copy sat in an open buffer
/// and the engine would read an unwritten slot. The read below happens with NO
/// `device_sync` anywhere, deliberately — the signal is the only thing ordering it.
#[test]
fn a_signal_covers_recorded_work_without_a_device_sync() {
    let v = Validation::new();
    let mut x = dev(&f32b(&[1.0, 2.0, 3.0, 4.0]));
    let mut y = dev(&f32b(&[10.0, 20.0, 30.0, 40.0]));
    // Records a copy on the fetch queue and nothing else. Under HIP this is
    // `hipMemcpyAsync` on the fetch stream; here it is `vkCmdCopyBuffer`.
    // SAFETY: both mappings are live, 16 bytes each, distinct allocations.
    unsafe {
        rivoli::backend::vk::copy_h2d_async(y.host_mut(), x.host_mut(), 16).expect("async copy");
    }
    block_on(armed_on_fetch());
    let got = f32v(&{
        let mut out = Vec::new();
        y.read_into(&mut out, 16).expect("read back");
        out
    });
    assert_eq!(
        got,
        vec![1.0, 2.0, 3.0, 4.0],
        "the copy had not landed when its Signal resolved"
    );
    v.check("a_signal_covers_recorded_work_without_a_device_sync");
}

/// The cross-queue hazard, end to end: the FETCH queue writes bytes and a dispatch on the
/// MoE QUEUE reads them, ordered only by the host awaiting the copy's `Signal`.
///
/// This is the pair the barrier review is about, and the one synchronisation validation on
/// this stack **cannot see** — it reports transfer↔transfer only, so a clean run says
/// nothing either way (docs/investigations/vulkan-port.md, "Risks"). Execution order comes from the await;
/// visibility comes from the barrier at the head of the reading queue's command buffer. A
/// missing acquire barrier would show up here as stale bytes in `out`, and nowhere else.
///
/// It is the decode path's exact shape: `moe_acc_drain` on the compute stream folding an
/// expert accumulator that arrived by H2D copy. `launch_vadd` would NOT do — it takes no
/// stream, so it records on the MAIN queue, and a test that awaited the MoE stream for it
/// would assert nothing while looking like it asserted the whole thing.
#[test]
fn a_moe_dispatch_sees_what_the_fetch_queue_wrote() {
    let v = Validation::new();
    const N: usize = 256;
    const ROWS: usize = 2;
    // `moe_acc_drain` reads a FIXED-POINT accumulator: one unit is 2^-MOE_ACC_SHIFT, so
    // 1<<44 per row over ROWS rows at gain 3 is exactly 6.0 — the same value this asserted
    // when it drove `moe_reduce`, with no rounding for the schedule to reach.
    let acc_bytes: Vec<u8> = (0..N * ROWS)
        .flat_map(|_| (1u64 << 44).to_le_bytes())
        .collect();
    let mut src = dev(&acc_bytes);
    // `staged` is what the fetch copy writes and the MoE dispatch reads.
    let mut staged = dev(&vec![0u8; N * ROWS * 8]);
    // Zero, not `poison`: the drain is `x[o] +=`, so a poisoned x could not be read as a
    // sum. A copy that had NOT landed leaves the accumulator at zero and x at 0.0, which is
    // exactly the stale-read this test exists to catch.
    let mut out = dev(&f32b(&vec![0.0f32; N]));
    let moe = rivoli::backend::Stream::compute().expect("compute stream");
    // SAFETY: live mappings, N*ROWS*8 bytes each, distinct allocations.
    unsafe {
        rivoli::backend::vk::copy_h2d_async(staged.host_mut(), src.host_mut(), N * ROWS * 8)
            .expect("copy");
    }
    block_on(armed_on_fetch());
    // Now a dispatch on the MoE queue reads those bytes: out[o] = 3·Σ_r staged[r][o]/2^44 = 6.
    drain(&mut out, &mut staged, N, ROWS, 3.0, moe.raw());
    block_on(rivoli::backend::stream_signal(moe.raw()).expect("moe signal"));
    let got = f32v(&read(&out, N));
    assert!(
        got.iter().all(|&g| g == 6.0),
        "a MoE-queue dispatch read stale bytes the fetch queue had already written: {:?}",
        &got[..8]
    );
    v.check("a_moe_dispatch_sees_what_the_fetch_queue_wrote");
}

/// GPU timestamps are a MEASUREMENT, not a stub — and a pair that was never recorded
/// REFUSES rather than reading zero.
///
/// The second half is the regression test for increment 1's actual defect. `elapsed_ms`
/// returning 0.0 made `compute_gpu_ms` zero, which made `ProfileSummary`'s
/// `fetch_hidden_pct` print "0% hidden" as an arithmetic artifact of the stub — a number
/// that reads as a finding. An error cannot be mistaken for a measurement; a zero can.
#[test]
fn timestamps_measure_gpu_time_and_refuse_when_absent() {
    let v = Validation::new();
    let start = rivoli::backend::Event::new().expect("event");
    let end = rivoli::backend::Event::new().expect("event");
    assert!(
        rivoli::backend::Event::elapsed_ms(&start, &end).is_err(),
        "an unrecorded stamp pair reported a duration; that is how a stub reads as data"
    );

    // A dispatch big enough to take measurable GPU time: 2048x2048 f32 GEMV.
    let (o, i) = (2048usize, 2048usize);
    let mut r = Lcg(0x5AA);
    let x = dev(&f32b(&(0..i).map(|_| r.f()).collect::<Vec<_>>()));
    let w = dev(&f32b(&(0..o * i).map(|_| r.f()).collect::<Vec<_>>()));
    let mut y = dev(&vec![0u8; o * 4]);
    let wall = std::time::Instant::now();
    start.record(std::ptr::null_mut()).expect("record start");
    for _ in 0..8 {
        launch(&x, &w, &mut y, o, i);
    }
    end.record(std::ptr::null_mut()).expect("record end");
    device_sync().expect("sync");
    let wall_ms = wall.elapsed().as_secs_f64() * 1000.0;
    let ms = rivoli::backend::Event::elapsed_ms(&start, &end).expect("elapsed");
    println!("VK TIMESTAMP span: {ms:.3} ms GPU against {wall_ms:.3} ms wall");
    assert!(
        ms > 0.0,
        "GPU span measured {ms} ms — the query pool is not being written"
    );
    // A sanity CEILING, not a comparison against `wall_ms`. The main queue is
    // process-global and cargo runs these tests on parallel threads, so another test's
    // dispatches can legitimately land between these two stamps and make the GPU span
    // exceed this test's own wall. That would be a flake, not a finding — the wall is
    // printed for the reader, and the assertion is set only far enough out to catch a bad
    // scale factor (`timestampPeriod` misapplied turns 3 ms into 3 ns or 3 s).
    assert!(
        ms < 10_000.0,
        "GPU span {ms} ms is implausible for 8 dispatches; check timestampPeriod scaling"
    );
    // A SECOND read, with no fresh `record`, must refuse rather than hand back the same span
    // again. A stale duration is indistinguishable from a live one at the call site, which is
    // the whole failure mode this test's first assertion exists for, one step further along.
    assert!(
        rivoli::backend::Event::elapsed_ms(&start, &end).is_err(),
        "a stamp pair reported its span twice; the second read was stale data"
    );
    v.check("timestamps_measure_gpu_time_and_refuse_when_absent");
}

/// The ring wraps, in the two regimes that fail differently.
///
/// Deliberately more iterations than `RING`: at exactly `RING` the wrap never happens, which
/// is the off-by-one this test exists to catch.
///
/// **Part 1, slots still PENDING.** The MoE stream is eager, so each launch is its own
/// submit and none is joined — after `RING` of them the next `open` must wait a fence and
/// reset a slot before reusing it. Reusing a pending command buffer is a CORE validation
/// error (not a synchronisation one, so the layer really does catch this class), and a
/// missing `reset_fences` turns the next wait into a hang. This is the regime the decode
/// path runs in.
///
/// **Part 2, across joins.** `device_sync` per iteration retires everything, so this checks
/// the other direction: nothing is lost when the ring wraps through joins. The data is
/// cumulative, so a dropped batch shows up as a wrong total — which a single idempotent
/// dispatch could not show.
#[test]
fn the_command_buffer_ring_wraps_without_a_stale_slot() {
    let v = Validation::new();
    const N: usize = 64;
    const ROWS: usize = 2;
    const LAPS: usize = 40; // RING is 16; 40 is two and a half laps
    // 1<<44 is one unit at MOE_ACC_SHIFT, so ROWS rows at gain 2 drain to exactly 4.0.
    // The drain RESETS the accumulator, so laps 2..LAPS add zero and the total is the
    // first lap's — part 1 was never the cumulative half (part 2 is), and the fence
    // reset/pending-slot reuse this exercises fails as a validation error or a hang, both
    // of which `LAPS > RING` unjoined submits are what provoke.
    let acc_bytes: Vec<u8> = (0..N * ROWS)
        .flat_map(|_| (1u64 << 44).to_le_bytes())
        .collect();
    let mut acc = dev(&acc_bytes);
    let mut out = dev(&f32b(&vec![0.0f32; N]));
    let moe = rivoli::backend::Stream::compute().expect("compute stream");
    for _ in 0..LAPS {
        drain(&mut out, &mut acc, N, ROWS, 2.0, moe.raw());
    }
    block_on(rivoli::backend::stream_signal(moe.raw()).expect("signal"));
    let got = f32v(&read(&out, N));
    assert!(
        got.iter().all(|&g| g == 4.0),
        "the eager MoE stream produced {:?} after {LAPS} unjoined submits",
        &got[..8]
    );

    let one = dev(&f32b(&vec![1.0f32; N]));
    let mut acc = dev(&f32b(&vec![0.0f32; N]));
    for _ in 0..LAPS {
        // `vadd` takes no stream, so this is the MAIN queue — the lazy one. The
        // `device_sync` per iteration is what makes the total deterministic: without it,
        // another test's join on this process-global stream decides how much of the batch
        // has executed by the time the mapping is read.
        // SAFETY: device addresses of live Bufs holding N f32 each.
        unsafe {
            launch_vadd(acc.ptr_mut() as *mut f32, one.ptr() as *const f32, N).expect("vadd");
        }
        device_sync().expect("sync");
    }
    let got = f32v(&read(&acc, N));
    assert!(
        got.iter().all(|&g| g == LAPS as f32),
        "expected {LAPS} accumulated adds across the ring wrap, got {:?}",
        &got[..8]
    );
    v.check("the_command_buffer_ring_wraps_without_a_stale_slot");
}

/// A launcher handed a stream token that is neither NULL nor one of the three queue tags
/// REFUSES, rather than defaulting to a queue and computing plausible numbers on the wrong
/// one. `Q::parse` is total; this is the arm that proves the error branch exists.
#[test]
fn an_unknown_stream_token_is_refused() {
    let mut y = dev(&[0u8; 64]);
    let p = y.ptr_mut() as *mut f32;
    // SAFETY: `Q::parse` runs before the argument guards and before any pointer is read.
    let r =
        unsafe { launch_moe_acc_drain(p, p as *mut u64, 8, 1, 1.0, 0x99 as *mut std::ffi::c_void) };
    assert!(
        r.is_err(),
        "a bogus stream token silently picked a queue instead of failing"
    );
}

/// A kernel reads weights placed through `DeviceTier`, at the address `place`
/// returned.
///
/// This is the one line in the backend that converts between the two bases —
/// `place` writes through the HOST mapping and hands back a DEVICE address
/// (`slab.ptr() as usize + off`) — and docs/investigations/vulkan-port.md calls that split the biggest
/// structural difference in the port. The unit test in device.rs proves only the
/// host-side arithmetic: it reads back through the host mapping, so a base swap, a
/// sign error, or an offset that does not track between the two mappings would pass
/// it and then produce garbage weights at integration.
///
/// Handing the returned address to a real dispatch is the only thing that closes
/// that. It is also a miniature of what `pin.rs` does with every resident weight.
#[test]
fn kernel_reads_weights_placed_through_the_tier() {
    let v = Validation::new();
    let mut r = Lcg(0x71E);
    let (o_dim, i_dim) = (64usize, 128usize);
    let (w, x, want) = gemv_case(&mut r, o_dim, i_dim);
    let mut tier = DeviceTier::new(4 << 20).expect("tier");
    // Two placements, so the second sits at a non-zero bump offset — an offset that
    // failed to track between the host and device bases would only show up there.
    let wp = tier.place(&f32b(&w)).expect("place w");
    let xp = tier.place(&f32b(&x)).expect("place x");
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));

    // SAFETY: both are DEVICE addresses returned by `place`, sized as the launcher
    // documents, inside a tier that outlives the sync below.
    unsafe { launch_at(xp, wp, yb.ptr_mut(), o_dim, i_dim) };
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32v(&read(&yb, o_dim)),
        "gemv over tier-placed weights",
    );
    v.check("kernel_reads_weights_placed_through_the_tier");
}

/// `DeviceBuf::copy_out_into` must see what a just-launched kernel wrote, WITHOUT the
/// caller syncing.
///
/// Its HIP twin is a blocking `hipMemcpy(..., D2H)` and every caller was written against
/// that contract. `gpu.rs:789` does `launch_gemv_f32( 1,...)` then immediately
/// `gate_logits.copy_out_into(...)`; if the Vulkan version were a bare read of the host
/// mapping it would return the PREVIOUS token's gate logits, and routing would pick the
/// wrong experts on every layer of every token — coherently, with no error.
///
/// Found by review, not by a test, and untested until now because `gpu.rs` is rocm-only.
/// This closes it at unit scale, where a failure implicates one function, rather than at
/// integration where every kernel is simultaneously a suspect.
///
/// DETERMINISTIC, not a race: without the internal `device_sync` the dispatch is still
/// sitting unsubmitted in the open command buffer, so the read returns the sentinel
/// every time. No scaling needed — unlike the barrier test, which needed 32 chained
/// steps at 2048x2048 before a missing barrier became observable at all.
#[test]
fn copy_out_into_sees_the_dispatch_that_preceded_it() {
    let v = Validation::new();
    let mut r = Lcg(0x0D2);
    let (o_dim, i_dim) = (128usize, 256usize);
    let (w, x, want) = gemv_case(&mut r, o_dim, i_dim);
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    // A sentinel the kernel cannot produce, so "stale" and "correct" are never confusable.
    let sentinel = vec![-12345.0f32; o_dim];
    let mut out = DeviceBuf::new(o_dim * 4).expect("alloc");
    out.copy_in_at(0, &f32b(&sentinel)).expect("seed");

    // SAFETY: live device addresses of the documented sizes; nothing is dropped before
    // the copy_out_into below, which is what retires the dispatch.
    unsafe { launch_at(xb.ptr(), wb.ptr(), out.ptr_mut(), o_dim, i_dim) };
    // NO device_sync here. That is the whole test.
    let mut host = Vec::new();
    out.copy_out_into(&mut host).expect("copy_out_into");
    let got = f32v(&host);
    assert!(
        got.iter().all(|&g| g != -12345.0),
        "copy_out_into returned the SENTINEL — it read the host mapping without \
         retiring the dispatch, so a caller gets the previous contents (gpu.rs:789 \
         would route on the previous token's gate logits)"
    );
    assert_close(&want, &got, "copy_out_into after dispatch");
    v.check("copy_out_into_sees_the_dispatch_that_preceded_it");
}

/// `DeviceBuf::copy_in_at` must not overwrite bytes a recorded-but-unsubmitted dispatch
/// is going to read.
///
/// The mirror hazard of the test above, and the one that is easy to miss because it
/// runs the other way: HIP's H2D `hipMemcpy` blocks, so it is ordered AFTER any kernel
/// still reading the buffer. A bare host write into a mapped allocation is not — the
/// write lands before the submit, and the dispatch reads the NEW bytes. `gpu.rs:843`
/// (descs_vq / wexpert_buf) depends on the blocking behaviour.
///
/// Deterministic for the same reason: the dispatch has not been submitted, so without
/// the internal sync it is guaranteed to see the overwrite rather than merely likely to.
#[test]
fn copy_in_at_does_not_clobber_a_pending_dispatch() {
    let v = Validation::new();
    let mut r = Lcg(0x1A7);
    let (o_dim, i_dim) = (64usize, 128usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x_old: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    // Distinct enough that a result computed from the wrong one cannot pass the oracle.
    let x_new: Vec<f32> = x_old.iter().map(|v| v * 7.0 + 3.0).collect();
    let mut want = vec![0.0f32; o_dim];
    matvec_f32(&mut want, &x_old, &w, i_dim);

    let wb = dev(&f32b(&w));
    let mut xb = DeviceBuf::new(i_dim * 4).expect("alloc x");
    xb.copy_in_at(0, &f32b(&x_old)).expect("seed old");
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));

    // SAFETY: as above; `xb` outlives the sync inside copy_in_at.
    unsafe { launch_at(xb.ptr(), wb.ptr(), yb.ptr_mut(), o_dim, i_dim) };
    // Overwrite the INPUT the pending dispatch reads. copy_in_at must retire it first.
    xb.copy_in_at(0, &f32b(&x_new)).expect("clobber");
    device_sync().expect("sync");

    let got = f32v(&read(&yb, o_dim));
    assert_close(&want, &got, "gemv used x_old despite the overwrite");
    v.check("copy_in_at_does_not_clobber_a_pending_dispatch");
}

/// The guards report errors rather than tripping a Vulkan VUID or allocating nothing.
#[test]
fn guards_reject_degenerate_arguments() {
    let v = Validation::new();
    assert!(Buf::new(0).is_err(), "Buf::new(0) must be rejected");

    let mut b = dev(&[0u8; 64]);
    assert!(b.write_at(0, &[0u8; 65]).is_err(), "overlong write");
    assert!(b.write_at(64, &[0u8; 1]).is_err(), "write at the end");
    // The bounds check must not wrap: off + len overflows usize here, and an
    // `off + len <= self.len` test would compute a small number and let it through.
    assert!(
        b.write_at(usize::MAX, &[0u8; 8]).is_err(),
        "wrapping offset"
    );
    assert!(b.read_into(&mut Vec::new(), 65).is_err(), "overlong read");

    let mut y = dev(&[0u8; 16]);
    // The same dispatch with one dim zeroed, so the two cases differ by exactly the
    // argument under test and nothing else can be what rejected them.
    let mut degenerate = |o_dim, i_dim| {
        // SAFETY: zero dims are rejected before any pointer is used.
        unsafe {
            let p = b.ptr() as *const f32;
            launch_gemv_f32(p, p, o_dim, i_dim, 1, y.ptr_mut() as *mut f32, std::ptr::null_mut())
        }
    };
    assert!(degenerate(0, 4).is_err(), "o_dim = 0 must be rejected");
    assert!(degenerate(4, 0).is_err(), "i_dim = 0 must be rejected");
    // Rejecting a bad argument must not itself trip the layer — a guard that returns
    // Err while leaving a half-built Vulkan object behind would show up here.
    v.check("guards_reject_degenerate_arguments");
}

// ---------------------------------------------------------------------------
// fwd.hip glue: embed_i8_row, gather_rope, vadd, append_kv, argmax_reduce.
//
// Oracles are written against kernels/fwd.hip and src/math.rs, NOT against the GLSL —
// a test derived from the shader it checks agrees with the shader by construction.
// ---------------------------------------------------------------------------

/// `x[i] = (i8)packed[token·hidden + i] · scale[token]`.
///
/// The shader loads a `u32` word and sign-extends one byte out of it by hand, so the
/// failure mode that matters is a byte read as UNSIGNED: invisible for every value
/// below 0x80 and a 256·scale error at every value above it. Every row therefore
/// starts with the four extremes {-128, -1, 127, 0}.
///
/// `hidden = 37` is neither a multiple of 4 nor of 32, so (a) the last workgroup has
/// idle threads and (b) the row base `token·37` walks byte phases 0,1,2,3,0 over the
/// five tokens — the same four extremes are extracted from a different position within
/// the word on each iteration, which is the whole reason to loop over tokens at all.
#[test]
fn embed_i8_row_sign_extends_and_scales() {
    let v = Validation::new();
    let (vocab, hidden) = (5usize, 37usize);
    let mut packed: Vec<u8> = (0..vocab * hidden)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect();
    for t in 0..vocab {
        packed[t * hidden..t * hidden + 4].copy_from_slice(&[0x80, 0xFF, 0x7F, 0x00]);
    }
    // Distinct per row, so a kernel indexing `scale[0]` instead of `scale[token]` is a
    // failure rather than a coincidence.
    let scale: Vec<f32> = (0..vocab).map(|t| 0.013 * (t as f32 + 1.0)).collect();
    let (pb, sb) = (dev(&packed), dev(&f32b(&scale)));

    for token in 0..vocab {
        let want: Vec<f32> = (0..hidden)
            .map(|i| f32::from(packed[token * hidden + i] as i8) * scale[token])
            .collect();
        let mut xb = poison(hidden * 4 + GUARD);
        // SAFETY: live Buf device addresses of the documented sizes — `packed` is
        // vocab·hidden bytes, `scale` vocab f32, `x` hidden f32 plus the guard band.
        // Nothing is dropped before the sync.
        unsafe {
            launch_embed_i8_row(
                pb.ptr(),
                sb.ptr() as *const f32,
                token,
                hidden,
                xb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        let got = sync_readb(&xb, hidden * 4 + GUARD);
        let label = format!("embed_i8_row token {token}");
        assert_untouched(&got, hidden * 4, &label);
        // A sign-extension bug is a 256·scale error and clears this tolerance by three
        // orders of magnitude — `assert_close`'s printed margin says which happened.
        assert_close(&want, &f32v(&got[..hidden * 4]), &label);
    }
    v.check("embed_i8_row_sign_extends_and_scales");
}

/// `qrope[head·ropn + d] = q[head·qh + nope + d]` — gather each head's roped segment
/// out of the strided `q` into a contiguous buffer.
#[test]
fn gather_rope_gathers_the_strided_segment() {
    let v = Validation::new();
    let mut r = Lcg(0x604);
    // (h, qh, nope, ropn), all ragged. The caller guarantees nope + ropn <= qh; the
    // last shape sits EXACTLY at qh, so the final head reads q's very last element and
    // an off-by-one in the offset runs off the end of the allocation.
    for (h, qh, nope, ropn) in [
        (7usize, 100usize, 37usize, 51usize),
        (1, 64, 0, 64), // nope = 0: the no-offset path, single head, grid of one
        (33, 97, 65, 32),
    ] {
        let q: Vec<f32> = (0..h * qh).map(|_| r.f()).collect();
        let want: Vec<f32> = (0..h * ropn)
            .map(|i| q[(i / ropn) * qh + nope + i % ropn])
            .collect();
        let out_bytes = h * ropn * 4;

        let qb = dev(&f32b(&q));
        let mut ob = poison(out_bytes + GUARD);
        // SAFETY: `q` is h·qh f32, `qrope` h·ropn f32 plus the guard band; both live
        // until the sync below.
        unsafe {
            launch_gather_rope(
                qb.ptr() as *const f32,
                ob.ptr_mut() as *mut f32,
                h,
                qh,
                nope,
                ropn,
            )
            .expect("launch");
        }
        let got = sync_readb(&ob, out_bytes + GUARD);
        let label = format!("gather_rope h{h} qh{qh} nope{nope} ropn{ropn}");
        assert_untouched(&got, out_bytes, &label);
        // BYTES, not `assert_close`. This kernel performs no arithmetic, so any
        // tolerance at all would be slack for a bug rather than for rounding.
        assert_bytes(&f32b(&want), &got[..out_bytes], &label);
    }
    v.check("gather_rope_gathers_the_strided_segment");
}

/// `x[i] += y[i]`, in place.
#[test]
fn vadd_adds_into_x_in_place() {
    let v = Validation::new();
    let mut r = Lcg(0xADD);
    // 1 (one live thread in one workgroup), 255/257 either side of the 256-thread
    // workgroup, then ragged multi-workgroup sizes. None is a multiple of 256.
    for n in [1usize, 255, 257, 1000, 4097] {
        let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let y: Vec<f32> = (0..n).map(|_| r.f()).collect();
        let want: Vec<f32> = x.iter().zip(&y).map(|(a, b)| a + b).collect();

        let mut xb = poison(n * 4 + GUARD);
        xb.write_at(0, &f32b(&x)).expect("fill x");
        let yb = dev(&f32b(&y));
        // SAFETY: `x` is n f32 plus the guard band, `y` is n f32; both live until the
        // sync. `x` is read AND written by the kernel, which is the contract.
        unsafe {
            launch_vadd(xb.ptr_mut() as *mut f32, yb.ptr() as *const f32, n).expect("launch");
        }
        device_sync().expect("sync");
        // Read back x's OWN buffer, so this also proves the sum landed in place rather
        // than in some scratch the launcher forgot to point at x.
        let got = readb(&xb, n * 4 + GUARD);
        let label = format!("vadd n={n}");
        assert_untouched(&got, n * 4, &label);
        assert_close(&want, &f32v(&got[..n * 4]), &label);
    }
    v.check("vadd_adds_into_x_in_place");
}

/// The CPU side of `append_kv`, exactly as fwd.hip specifies it. Returns
/// (lc8 bytes, block scales, rc bytes) for ONE row.
///
/// `f32_to_e4m3` and `f32_to_bf16` are bit-exact contracts — the kernel comment says
/// "mirrors math.rs::f32_to_e4m3 bit-for-bit" — so the caller compares bytes. A float
/// tolerance would accept a quantizer that rounds the other way at every tie, which is
/// precisely the bug this file exists to catch.
fn append_kv_oracle(latent: &[f32], rope: &[f32]) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
    let scales: Vec<f32> = latent
        .chunks_exact(E4M3_BLOCK)
        .map(|blk| {
            let amax = blk.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            // amax == 0 -> scale 1.0, or the quantizer would divide by zero.
            if amax > 0.0 { amax / E4M3_MAX } else { 1.0 }
        })
        .collect();
    let lc8 = latent
        .iter()
        .enumerate()
        .map(|(i, x)| f32_to_e4m3(x / scales[i / E4M3_BLOCK]))
        .collect();
    let rc = rope
        .iter()
        .flat_map(|x| f32_to_bf16(*x).to_le_bytes())
        .collect();
    (lc8, scales, rc)
}

/// One latent of `kvl` values whose 128-blocks sit in deliberately different scale
/// regimes, plus `ropn` rope values.
fn append_kv_input(kvl: usize, ropn: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut r = Lcg(seed);
    let latent: Vec<f32> = (0..kvl)
        .map(|i| match i / E4M3_BLOCK {
            1 => 0.0,            // amax == 0 -> the scale = 1.0 branch
            2 => r.f() * 3000.0, // amax >> 448 -> a scale above one
            _ => r.f(),          // amax ~ 1 -> a scale near 1/448
        })
        .collect();
    let mut rope: Vec<f32> = (0..ropn).map(|_| r.f()).collect();
    // bf16 is a pure bit operation, so these are unconditional: the RNE tie (rounds
    // DOWN to the even neighbour 0x3f80), the value just past it (0x3f81), zero, and a
    // magnitude f32 carries exactly but that leaves no mantissa room in bf16.
    // ponytail: no -0.0 here. Its bf16 IS 0x8000, but Vulkan does not promise signed
    // zero preservation without an execution mode, so a 0x0000 would be ambiguous
    // between a shader bug and a missing SignedZeroInfNanPreserve — untestable from
    // here, and a probe's job rather than an oracle's.
    let specials = [
        1.0 + 2f32.powi(-8),
        1.0 + 2f32.powi(-8) + 2f32.powi(-16),
        0.0,
        65504.0,
    ];
    rope[..specials.len()].copy_from_slice(&specials);
    (latent, rope)
}

/// Every value in `values` must quantise to the same result `margin_ulp` either side
/// of itself — i.e. none of them sits near a rounding midpoint of the target format.
///
/// THE STANDING PRECONDITION FOR ANY BYTE-EXACT ORACLE (see the module header). A test
/// that compares quantised bytes is only meaningful if the inputs cannot legitimately
/// quantise both ways: otherwise the driver's arithmetic accuracy decides the result,
/// the test passes here and fails elsewhere, and the shader is innocent.
///
/// `margin_ulp` must come from what the SPEC guarantees for the operations that produce
/// these values, not from what this GPU happens to do. Vulkan guarantees 2.5 ULP on
/// `FDiv`; `FAdd`/`FMul` are correctly rounded but a reduction accumulates, so a dot
/// product needs a margin that grows with its length.
///
/// The message names the SEED deliberately. When this fires the data is wrong, not the
/// kernel, and the reader should not start by auditing a shader.
fn assert_quantization_unambiguous<T: PartialEq>(
    label: &str,
    values: &[f32],
    margin_ulp: u32,
    quantize: impl Fn(f32) -> T,
) {
    for (i, &q) in values.iter().enumerate() {
        if q == 0.0 || !q.is_finite() {
            continue; // exact or saturating; no midpoint to sit on
        }
        let up = f32::from_bits(q.to_bits().wrapping_add(margin_ulp));
        let dn = f32::from_bits(q.to_bits().wrapping_sub(margin_ulp));
        let want = quantize(q);
        assert!(
            want == quantize(up) && want == quantize(dn),
            "{label}[{i}] = {q:e} quantises ambiguously within {margin_ulp} ULP: this \
             SEED cannot support a byte-exact compare — pick another. The kernel is not \
             implicated; see the byte-exact-oracle rule in this file's header."
        );
    }
}

/// `append_kv`'s use of the rule above: 8 ULP covers both divisions in the chain
/// (amax/448, then latent/scale) at Vulkan's 2.5 ULP `FDiv` guarantee, with room.
fn assert_quantization_is_unambiguous(latent: &[f32], scales: &[f32]) {
    let quotients: Vec<f32> = latent
        .iter()
        .enumerate()
        .map(|(i, x)| x / scales[i / E4M3_BLOCK])
        .collect();
    assert_quantization_unambiguous("latent/scale", &quotients, 8, f32_to_e4m3);
}

/// `append_kv`: latent -> fp8-e4m3 with a per-128 block scale, roped key -> bf16, both
/// at row `pos`.
///
/// The block amax is an LDS tree reduction over 128 lanes, which is where the
/// reproducibility test below points; this one is about the quantizer agreeing with
/// `math.rs` bit for bit and about the row offset.
#[test]
fn append_kv_quantizes_bit_exactly() {
    let v = Validation::new();
    // kvl must be a multiple of 128 in [128, 1024], so it cannot be ragged — ropn is
    // the only free size here and 100 is deliberately not a multiple of 32, leaving the
    // rope write live on a partial subgroup.
    // Three shapes, and the two new ones each isolate ONE previously untested path so a
    // failure is attributable:
    //   512/100  the original: both paths in their already-covered state.
    //   1024/100 kvl = MAX_BLOCKS*128, so every subgroup carries a block and
    //            `gl_NumSubgroups == 8` is load-bearing rather than slack. Rope stays
    //            under THREADS, so this shape can only implicate the block mapping.
    //   512/300  ropn > THREADS, so the rope loop takes a second grid-stride iteration
    //            that was dead code until now. Block count stays at the covered 4, so
    //            this shape can only implicate the rope loop.
    // Combining them (1024/300) would exercise both at once and, on a failure, tell you
    // nothing about which.
    for (kvl, ropn, rows, pos) in [
        (512usize, 100usize, 5usize, 3usize),
        (1024, 100, 3, 2),
        (512, 300, 4, 1),
    ] {
        let n_blocks = kvl / E4M3_BLOCK;
        let (latent, rope) = append_kv_input(kvl, ropn, 0xA55);
        let (want_lc8, want_scl, want_rc) = append_kv_oracle(&latent, &rope);
        assert_quantization_is_unambiguous(&latent, &want_scl);
        // The two branches the test exists for, asserted on the oracle so a future edit to
        // `append_kv_input` cannot silently drop either.
        assert_eq!(
            want_scl[1], 1.0,
            "block 1 must exercise the amax == 0 branch"
        );
        assert!(want_scl[2] > 1.0, "block 2 must exercise a large amax");

        let (lb, rb) = (dev(&f32b(&latent)), dev(&f32b(&rope)));
        // Whole slabs, poisoned. Rows other than `pos` are as much of a guard band as the
        // trailing GUARD bytes: a row-offset bug lands in one of them, and an `i < ropn`
        // check missing from the rope write spills straight into row pos+1 of `rc`.
        let (lc8_n, lscale_n, rc_n) = (rows * kvl, rows * n_blocks * 4, rows * ropn * 2);
        let mut lc8 = poison(lc8_n + GUARD);
        let mut lscale = poison(lscale_n + GUARD);
        let mut rc = poison(rc_n + GUARD);
        let mut dst = KvDst::new(&mut lc8, &mut lscale, &mut rc);
        append_kv(&lb, &rb, &mut dst, (pos, kvl, ropn, n_blocks));
        let g_lc8 = readb(&lc8, lc8_n + GUARD);
        let g_scl = readb(&lscale, lscale_n + GUARD);
        let g_rc = readb(&rc, rc_n + GUARD);

        for (got, stride, name) in [
            (&g_lc8, kvl, "lc8"),
            (&g_scl, n_blocks * 4, "lscale"),
            (&g_rc, ropn * 2, "rc"),
        ] {
            assert!(
                got[..pos * stride].iter().all(|&b| b == 0xFF),
                "append_kv {name}: wrote BEFORE row {pos}"
            );
            assert_untouched(
                got,
                (pos + 1) * stride,
                &format!("append_kv {name} row {pos}"),
            );
        }
        assert_bytes(
            &want_lc8,
            &g_lc8[pos * kvl..(pos + 1) * kvl],
            "append_kv lc8",
        );
        assert_bytes(
            &want_rc,
            &g_rc[pos * ropn * 2..(pos + 1) * ropn * 2],
            "append_kv rc",
        );

        // The scale is the one float here — a division, for which Vulkan promises 2.5 ULP.
        // Compared per element RELATIVE rather than through `assert_close`: these four
        // scales span 1/448 to ~6.7, and a shared `mx` would hand the smallest of them
        // three orders of magnitude of slack. 1e-6 is ~8 ULP.
        let got_scl = f32v(&g_scl[pos * n_blocks * 4..(pos + 1) * n_blocks * 4]);
        let rel = want_scl
            .iter()
            .zip(&got_scl)
            .fold(0.0f32, |m, (w, g)| m.max((w - g).abs() / w.abs()));
        println!("append_kv {kvl}/{ropn} lscale: max rel err={rel:.3e} tol=1.0e-6");
        assert!(
            rel <= 1e-6,
            "append_kv lscale: rel err {rel:.3e} > 1e-6, want {want_scl:?} got {got_scl:?}"
        );
    }
    v.check("append_kv_quantizes_bit_exactly");
}

/// `append_kv` reduces the block amax in LDS, so it gets the bit-reproducibility test
/// every reduction in this backend gets — a scheduling-dependent reduce would still
/// clear any tolerance, and here it would silently move the quantization scale.
#[test]
fn append_kv_is_bit_reproducible() {
    let v = Validation::new();
    let (kvl, ropn, pos) = (512usize, 100usize, 2usize);
    let n_blocks = kvl / E4M3_BLOCK;
    let (latent, rope) = append_kv_input(kvl, ropn, 0xB16);
    let (lb, rb) = (dev(&f32b(&latent)), dev(&f32b(&rope)));

    // A decoy of a DIFFERENT shape (one block, half the rope, row 0) dispatched between
    // repeats, into its own buffers, so the two runs do not see identical queue state.
    let (dlat, drope) = append_kv_input(E4M3_BLOCK, 64, 0xD00);
    let (dlb, drb) = (dev(&f32b(&dlat)), dev(&f32b(&drope)));

    // Zero-filled, not poisoned: the all-zero guard at the end has to be able to fail.
    let rows = pos + 1;
    let mut lc8 = dev(&vec![0u8; rows * kvl]);
    let mut lscale = dev(&vec![0u8; rows * n_blocks * 4]);
    let mut rc = dev(&vec![0u8; rows * ropn * 2]);
    let (mut dlc8, mut dlscale, mut drc) =
        (dev(&[0u8; E4M3_BLOCK]), dev(&[0u8; 4]), dev(&[0u8; 128]));

    let run = |lat: &Buf, rop: &Buf, dst: &mut KvDst<'_>, shape: KvShape| {
        append_kv(lat, rop, dst, shape);
        let (pos, kvl, ropn, n_blocks) = shape;
        let mut out = readb(dst.lc8, (pos + 1) * kvl);
        out.extend(readb(dst.lscale, (pos + 1) * n_blocks * 4));
        out.extend(readb(dst.rc, (pos + 1) * ropn * 2));
        out
    };

    let mut real = KvDst::new(&mut lc8, &mut lscale, &mut rc);
    let mut decoy = KvDst::new(&mut dlc8, &mut dlscale, &mut drc);
    let first = assert_bit_reproducible(
        "append_kv",
        || run(&lb, &rb, &mut real, (pos, kvl, ropn, n_blocks)),
        || run(&dlb, &drb, &mut decoy, (0, E4M3_BLOCK, 64, 1)),
    );
    // Row `pos` of lc8 alone — rows 0..pos are legitimately still zero, so checking the
    // whole buffer would be a weaker guard than it looks.
    assert!(
        first[pos * kvl..(pos + 1) * kvl].iter().any(|&b| b != 0),
        "lc8 row {pos} is all zero — the test proves nothing"
    );
    v.check("append_kv_is_bit_reproducible");
}

/// One `argmax` dispatch + sync -> the two output words as RAW BYTES, idx then val.
///
/// Each word gets its own poisoned tail. They are written by thread 0 only, and a
/// shader that spilled the whole workgroup's partials would still leave the right
/// value in the first word.
fn argmax_raw(logits: &[f32]) -> Vec<u8> {
    let lb = dev(&f32b(logits));
    let mut ib = poison(4 + GUARD);
    let mut vb = poison(4 + GUARD);
    // SAFETY: `logits` is logits.len() f32; each output is one word plus a guard band.
    // All three live until the sync.
    unsafe {
        launch_argmax(
            lb.ptr() as *const f32,
            logits.len(),
            ib.ptr_mut() as *mut i32,
            vb.ptr_mut() as *mut f32,
        )
    }
    .expect("launch argmax");
    device_sync().expect("sync");
    let (gi, gv) = (readb(&ib, 4 + GUARD), readb(&vb, 4 + GUARD));
    assert_untouched(&gi, 4, "argmax out_idx");
    assert_untouched(&gv, 4, "argmax out_val");
    let mut out = gi[..4].to_vec();
    out.extend_from_slice(&gv[..4]);
    out
}

fn argmax(logits: &[f32]) -> (i32, f32) {
    let b = argmax_raw(logits);
    (
        i32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        f32::from_le_bytes([b[4], b[5], b[6], b[7]]),
    )
}

/// Both halves of the contract, always. An argmax that returns the right index with
/// the wrong value still breaks the caller, whose `!bv.is_finite()` bail reads the
/// value and not the index.
fn chk_argmax(label: &str, logits: &[f32], want_idx: i32, want_val: f32) {
    let (gi, gv) = argmax(logits);
    assert_eq!(gi, want_idx, "{label}: index (value was {gv})");
    assert_eq!(gv, want_val, "{label}: value (index was {gi})");
    println!("argmax {label}: n={} -> ({gi}, {gv})", logits.len());
}

/// `argmax_reduce` against the host fold it reproduces, adversarially.
///
/// This is the last op before the sampled token, so a wrong answer is a wrong output
/// with no numeric smell — no tolerance to blow, no NaN downstream, just a different
/// word. The contract, from fwd.hip: the LOWEST index of the maximum value wins, NaN
/// NEVER wins, and the returned value is `logits[best]`.
#[test]
fn argmax_matches_the_host_fold() {
    let v = Validation::new();

    chk_argmax("max at index 0", &[9.0, 1.0, 2.0, -3.0, 0.5], 0, 9.0);

    // n = 7: fewer elements than the 256-thread workgroup, so most lanes contribute
    // nothing but the identity and the LDS tree must not let one of them win.
    chk_argmax(
        "plateau -> lowest index",
        &[0.1, 0.5, 0.5, 0.3, 0.5, -1.0, 0.5],
        1,
        0.5,
    );

    let mut last = vec![-1.0f32; 1000];
    last[999] = 5.0;
    chk_argmax("max at the final position", &last, 999, 5.0);

    // 261 = 5 + 256: with a 256-thread block the SAME lane sees both, so this tie is
    // broken inside the grid-stride loop rather than in the LDS tree. The cross-lane
    // case is the 100_003 one below.
    let mut same_lane = vec![0.0f32; 300];
    same_lane[5] = 2.0;
    same_lane[261] = 2.0;
    chk_argmax("plateau within one lane", &same_lane, 5, 2.0);

    chk_argmax("all elements equal", &vec![1.5f32; 300], 0, 1.5);

    // NaN loses to any finite value: every compare against it is false, so it never
    // displaces the running best — including when it is the running best's only rival.
    chk_argmax(
        "NaN loses to finite",
        &[0.1, f32::NAN, 0.7, f32::NAN, 0.3, 0.7],
        2,
        0.7,
    );
    // Index 0 specifically: it is the identity's index, so a NaN there is the one
    // position where "NaN loses" and "ties go to the lowest index" could conspire.
    chk_argmax("NaN at index 0", &[f32::NAN, 0.2, 0.9, 0.4, -7.0], 2, 0.9);

    // Every element is the identity: only the tie rule decides, and logits[0] is -inf
    // so the returned value is exact rather than merely non-finite.
    chk_argmax(
        "all -inf",
        &vec![f32::NEG_INFINITY; 257],
        0,
        f32::NEG_INFINITY,
    );

    // All negative — catches an oracle (or a shader) that initialises the running best
    // to 0 instead of -inf, which is invisible whenever any element is positive.
    chk_argmax("all negative", &[-5.0, -2.0, -9.0, -2.5, -1e30], 1, -2.0);
    let big_neg: Vec<f32> = (0..1000)
        .map(|i| -((i as f32 - 613.0).abs()) - 3.0)
        .collect();
    chk_argmax("all negative at scale", &big_neg, 613, -3.0);

    // 100_003: ragged, ~391 grid-stride passes per lane, a NaN in lane 0's first slot,
    // and a plateau whose two members land on DIFFERENT lanes (12_345 % 256 = 57,
    // 98_765 % 256 = 205) so the tie is resolved by the LDS tree. `Lcg::f` is strictly
    // inside [-1, 1), so 3.0 is the unique maximum.
    let mut r = Lcg(0x9A2);
    let mut big: Vec<f32> = (0..100_003).map(|_| r.f()).collect();
    big[0] = f32::NAN;
    big[12_345] = 3.0;
    big[98_765] = 3.0;
    chk_argmax("cross-lane plateau at 100_003", &big, 12_345, 3.0);

    // ALL NaN. Nothing can win, so the reduction returns its identity. fwd.hip
    // documents that identity as best = 0, bv = -inf, chosen so the host's
    // `!bv.is_finite()` bail fires — the requirement is therefore not a particular
    // value but a NON-FINITE one at a defined index, and the value is printed rather
    // than asserted so a change in it is visible without being a failure.
    let (gi, gv) = argmax(&vec![f32::NAN; 300]);
    println!(
        "argmax all-NaN: -> ({gi}, {gv}) bits {:#010x}",
        gv.to_bits()
    );
    assert_eq!(gi, 0, "all-NaN: index");
    assert!(
        !gv.is_finite(),
        "all-NaN: value {gv} is finite — the caller's bail would not fire"
    );

    v.check("argmax_matches_the_host_fold");
}

/// `argmax_reduce` is a reduction, so it gets the bit-reproducibility test. Weaker
/// than it sounds for the index (an i32 is exact either way) and exactly the point for
/// the value: a reduce whose order varies with scheduling returns a different
/// `logits[best]` bit pattern on a plateau, and greedy decode would drift run to run.
#[test]
fn argmax_is_bit_reproducible() {
    let v = Validation::new();
    let mut r = Lcg(0x1D3);
    let mut logits: Vec<f32> = (0..100_003).map(|_| r.f() * 10.0).collect();
    // Nonzero index AND nonzero value, so the all-zero guard below is not vacuous.
    logits[77_777] = 42.5;
    // A differently-shaped dispatch between repeats, to perturb queue state.
    let decoy: Vec<f32> = (0..1291).map(|_| r.f()).collect();

    let first = assert_bit_reproducible("argmax", || argmax_raw(&logits), || argmax_raw(&decoy));
    assert!(
        first.iter().any(|&b| b != 0),
        "both output words are zero — the test proves nothing"
    );
    v.check("argmax_is_bit_reproducible");
}

/// The fwd launchers' argument guards report errors rather than dispatching a
/// degenerate grid or tripping a VUID.
///
/// `token`, `pos` and `nope` are `usize` here, so the HIP guards' `< 0` arms are
/// unreachable from Rust and are not tested — the only negative value that could reach
/// the shader is one produced by the `as i32` narrowing inside the launcher, which is
/// its business and not this file's.
#[test]
fn fwd_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let src = dev(&[0u8; 4096]);
    let mut d1 = dev(&[0u8; 4096]);
    let mut d2 = dev(&[0u8; 4096]);
    let mut d3 = dev(&[0u8; 4096]);
    let (p, a, b, c) = (src.ptr(), d1.ptr_mut(), d2.ptr_mut(), d3.ptr_mut());

    let akv = |kvl: usize, ropn: usize, n_blocks: usize| {
        // SAFETY: every one of these must be rejected by a shape guard before a pointer
        // is used; the four buffers are live and 4096 bytes regardless.
        unsafe {
            launch_append_kv(
                p as *const f32,
                p as *const f32,
                a,
                b as *mut f32,
                c as *mut u16,
                0,
                kvl,
                ropn,
                n_blocks,
            )
        }
    };
    assert!(akv(0, 8, 1).is_err(), "append_kv kvl = 0");
    assert!(akv(128, 0, 1).is_err(), "append_kv ropn = 0");
    assert!(akv(128, 8, 0).is_err(), "append_kv n_blocks = 0");
    // The one-block per-128 reduction needs kvl a multiple of 128 in [128, 1024], and
    // the rope half rides the same block so ropn cannot exceed it.
    assert!(akv(64, 8, 1).is_err(), "append_kv kvl < 128");
    assert!(
        akv(200, 8, 1).is_err(),
        "append_kv kvl not a multiple of 128"
    );
    assert!(akv(1152, 8, 9).is_err(), "append_kv kvl > 1024");
    assert!(akv(256, 300, 2).is_err(), "append_kv ropn > kvl");
    // The one guard deliberately STRICTER than HIP: the shader packs u16 keys into u32
    // words, so an odd ropn would straddle a word and drop the tail. Untested until
    // now, which meant the divergence from HIP was asserted only in a comment.
    assert!(akv(256, 99, 2).is_err(), "append_kv odd ropn");
    assert!(
        akv(256, 98, 2).is_ok(),
        "append_kv even ropn must still be accepted"
    );
    // POSITIVE but wrong. Every previous n_blocks case was rejected by an earlier arm
    // (kvl > 1024 fires before the equality check), so `n_blocks == kvl/128` could have
    // been deleted with the whole suite still green.
    assert!(akv(256, 8, 1).is_err(), "append_kv n_blocks too small");
    assert!(akv(256, 8, 3).is_err(), "append_kv n_blocks too large");

    // SAFETY: as above — a zero dimension is rejected before any pointer is used.
    unsafe {
        assert!(
            launch_embed_i8_row(p, p as *const f32, 0, 0, a as *mut f32).is_err(),
            "embed_i8_row hidden = 0"
        );
        assert!(
            launch_vadd(a as *mut f32, p as *const f32, 0).is_err(),
            "vadd n = 0"
        );
        assert!(
            launch_argmax(p as *const f32, 0, a as *mut i32, b as *mut f32).is_err(),
            "argmax n = 0"
        );
        for (h, qh, ropn, why) in [
            (0usize, 8usize, 8usize, "h"),
            (8, 0, 8, "qh"),
            (8, 8, 0, "ropn"),
        ] {
            assert!(
                launch_gather_rope(p as *const f32, a as *mut f32, h, qh, 0, ropn).is_err(),
                "gather_rope {why} = 0"
            );
        }
    }
    // Rejecting a bad argument must not itself trip the layer.
    v.check("fwd_guards_reject_degenerate_arguments");
}

// ---------------------------------------------------------------------------
// linalg.hip glue: swiglu, rmsnorm, rope_interleave.
//
// Oracles derived from kernels/linalg.hip and src/math.rs — the SPECIFICATION — and
// from nothing else. See the header: a test written from the shader it checks agrees
// with the shader by construction.
// ---------------------------------------------------------------------------

/// `h[i] = silu(g[i])·u[i]`, in and out of place.
///
/// TOLERANCE, NOT BYTES, AND DELIBERATELY. linalg.hip writes `gv / (1 + expf(-gv))`;
/// `math.rs::silu` writes `x * (1 / (1 + (-x).exp()))`. Those are the same function and
/// different roundings — a divide versus a multiply by a reciprocal — and underneath
/// both sits `exp`, a library routine whose last bits are unspecified in HIP and in
/// Vulkan alike (the SPIR-V extended instruction set promises 3 ULP on `Exp`, not
/// correct rounding). There is no bit pattern to agree on here, so the byte-exact rule
/// in this file's header does not apply and `assert_close` is the honest comparison.
///
/// ALIASING IS THE POINT OF THE SECOND ARM. linalg.hip states `h` may alias `g`, and
/// promises it by having each thread read `g[i]` before writing `h[i]`. A shader that
/// widened the write, reordered it ahead of a neighbouring read, or staged through a
/// shared tile would still be perfectly correct with distinct buffers and wrong here.
/// Running only the distinct case would leave the documented in-place contract untested.
#[test]
fn swiglu_matches_silu_times_u() {
    let v = Validation::new();
    let mut r = Lcg(0x51D);
    // 1 (one live thread in one workgroup), 255/257 either side of the 256-thread
    // workgroup, then ragged multi-workgroup sizes. None is a multiple of 256.
    for n in [1usize, 255, 257, 1000, 4097] {
        let mut g: Vec<f32> = (0..n).map(|_| r.f() * 12.0).collect();
        // Both saturating ends, planted rather than hoped for: silu(-30) is ~-2.8e-12
        // (the sigmoid has underflowed), silu(30) is 30 to within f32, and silu(0) is
        // 0. -0.5 leads so that even n = 1 exercises a value the shader has to compute
        // rather than one any implementation returns by accident.
        let specials = [-0.5f32, 0.0, -30.0, 30.0];
        let k = specials.len().min(n);
        g[..k].copy_from_slice(&specials[..k]);
        let u: Vec<f32> = (0..n).map(|_| r.f() * 2.0).collect();
        let want: Vec<f32> = g.iter().zip(&u).map(|(a, b)| silu(*a) * b).collect();

        let ub = dev(&f32b(&u));
        for alias in [false, true] {
            let mut gb = poison(n * 4 + GUARD);
            gb.write_at(0, &f32b(&g)).expect("fill g");
            let mut hb = poison(n * 4 + GUARD);
            let (gp, hp) = if alias {
                // ONE buffer, handed in as both operands — the in-place contract.
                let p = gb.ptr_mut();
                (p as *const f32, p as *mut f32)
            } else {
                (gb.ptr() as *const f32, hb.ptr_mut() as *mut f32)
            };
            // SAFETY: `g`/`u`/`h` are live Buf device addresses holding n f32 plus a
            // guard band; aliasing g and h is explicitly permitted by the kernel. None
            // is dropped before the sync.
            unsafe {
                launch_swiglu(gp, ub.ptr() as *const f32, n, hp).expect("launch");
            }
            device_sync().expect("sync");
            let out = if alias {
                readb(&gb, n * 4 + GUARD)
            } else {
                readb(&hb, n * 4 + GUARD)
            };
            let label = format!(
                "swiglu n={n} {}",
                if alias { "aliased" } else { "distinct" }
            );
            assert_untouched(&out, n * 4, &label);
            assert_close(&want, &f32v(&out[..n * 4]), &label);
            // With distinct buffers `g` is an input and must come back untouched. BYTES
            // here, unlike the result: no arithmetic is supposed to have happened to it,
            // so any tolerance would be slack for a bug. This is what makes a shader
            // that writes through the wrong pointer fail — in the aliased arm that bug
            // is invisible, because the wrong pointer is the right one.
            if !alias {
                assert_bytes(
                    &f32b(&g),
                    &readb(&gb, n * 4),
                    &format!("{label}: g unmodified"),
                );
            }
        }
    }
    v.check("swiglu_matches_silu_times_u");
}

/// `y[i] = x[i]·(1/sqrt(mean(x²)+eps))·w[i]`, exactly as linalg.hip orders it:
/// `(x·inv)·w`, so the only thing that can differ from the kernel is `inv` itself.
///
/// The mean accumulates in f64 — the kernel sums a strided partial per thread and then
/// an LDS tree, an order no host loop reproduces, so a "reference" f32 sum would be
/// just another arbitrary order rather than the right answer. f64 is the answer both
/// are approximating.
fn rmsnorm_oracle(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let sum: f64 = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    let mean = (sum / x.len() as f64) as f32;
    let inv = 1.0f32 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * inv * b).collect()
}

/// RMSNorm against the host oracle, over sizes and over two regimes the reduction can
/// get wrong in opposite directions.
#[test]
fn rmsnorm_matches_the_host_oracle() {
    let v = Validation::new();
    let mut r = Lcg(0x8A5);
    let eps = 1e-6f32;

    let chk = |x: &[f32], w: &[f32], label: &str| {
        let n = x.len();
        let want = rmsnorm_oracle(x, w, eps);
        let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
        let mut yb = poison(n * 4 + GUARD);
        let got = rmsnorm(&xb, &wb, &mut yb, n, eps, n * 4 + GUARD);
        assert_untouched(&got, n * 4, label);
        assert_close(&want, &f32v(&got[..n * 4]), label);
    };

    // 1 (one live thread; the other 255 tree slots hold the identity and must not
    // corrupt it), 255/256/257 around the workgroup, and 5000 — ~20 grid-stride passes
    // per thread, ragged, so the last pass is partial.
    for n in [1usize, 255, 256, 257, 5000] {
        let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
        // A DISTINCT weight per element: a kernel reading `w[0]`, or dropping `w`
        // entirely, is a coincidence away from passing on a uniform weight vector.
        let w: Vec<f32> = (0..n).map(|_| r.f()).collect();
        chk(&x, &w, &format!("rmsnorm n={n}"));
    }

    // LARGE DYNAMIC RANGE. One element carries the entire sum of squares: 500² = 2.5e5
    // against ~1e-8 apiece from the other 999. Lose that one slot and the mean falls by
    // ~1e8, `inv` rises ~1.6e4×, and EVERY output is wrong by that factor — so a
    // dropped reduction slot is visible rather than buried under the tolerance.
    // 777 % 256 = 9, so it lands in thread 9's strided partial and then in tree slot 9,
    // neither of which is the identity-heavy edge of the reduction.
    let n = 1000usize;
    let mut x: Vec<f32> = (0..n).map(|_| r.f() * 1e-4).collect();
    x[777] = 500.0;
    let w: Vec<f32> = (0..n).map(|_| r.f()).collect();
    // Only the spike's own output is checked meaningfully here — `assert_close` scales
    // its tolerance to the largest wanted value, which is the spike's. That is fine:
    // this case exists to move `inv`, and every other size above checks the elementwise
    // scaling at a uniform magnitude.
    chk(&x, &w, "rmsnorm dynamic range");

    // EPS-DOMINATED. mean(x²) ~ 1e-13, three orders below eps, so `inv` is decided
    // almost entirely by eps. Drop the eps term and `inv` goes from ~1e3 to ~1e6 — the
    // one arrangement in which the `+ eps` could be deleted and still look right
    // everywhere else, since at unit scale it moves the answer by 5e-7 relative.
    let x: Vec<f32> = (0..500).map(|_| r.f() * 1e-6).collect();
    let w: Vec<f32> = (0..500).map(|_| r.f()).collect();
    chk(&x, &w, "rmsnorm eps-dominated");

    v.check("rmsnorm_matches_the_host_oracle");
}

/// RMSNorm reduces Σx² in LDS, so it gets the bit-reproducibility test every reduction
/// in this backend gets. Stricter than the oracle above: a reduce whose order varies
/// with workgroup scheduling clears a 1e-3 tolerance easily, and here it would move
/// `inv` and therefore EVERY element of the output, silently, run to run.
///
/// The input spans six decades on purpose. Summation order only matters when the
/// addends differ in magnitude — over a uniform [-1,1) the f32 sum is nearly
/// order-independent and a nondeterministic reduction would produce identical bytes
/// anyway, making the test green for the wrong reason.
#[test]
fn rmsnorm_is_bit_reproducible() {
    let v = Validation::new();
    let mut r = Lcg(0x2E5);
    let (n, eps) = (5003usize, 1e-6f32);
    let x: Vec<f32> = (0..n)
        .map(|i| r.f() * 10f32.powi(i as i32 % 7 - 3))
        .collect();
    let w: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    // Zero-filled, not poisoned: the all-zero guard at the end has to be able to fail.
    let mut yb = dev(&vec![0u8; n * 4]);

    // A decoy of a different shape, dispatched between repeats so the two runs do not
    // see identical queue state. 301 is under one workgroup, so it is a different
    // grid-stride shape as well as a different size.
    let (dx, dw) = (dev(&f32b(&x[..301])), dev(&f32b(&w[..301])));
    let mut dy = dev(&vec![0u8; 301 * 4]);

    let run = |x: &Buf, w: &Buf, y: &mut Buf, n: usize| rmsnorm(x, w, y, n, eps, n * 4);

    let first = assert_bit_reproducible(
        "rmsnorm",
        || run(&xb, &wb, &mut yb, n),
        || run(&dx, &dw, &mut dy, 301),
    );
    assert_not_all_zero(&first);
    v.check("rmsnorm_is_bit_reproducible");
}

/// One row of `rope_interleave`, transcribed from linalg.hip: the angle in f64, `cos`
/// and `sin` rounded to f32, and the rotation itself in f32.
///
/// EVERY READ COMES FROM `v` AND EVERY WRITE GOES TO `out`. That is the contract the
/// HIP spends a `__syncthreads()` on — the destination halves `(j, half+j)` overlap the
/// source pairs `(2j, 2j+1)`, so an implementation that writes as it goes shreds its own
/// input. An oracle that updated in place would reproduce that bug instead of catching
/// it.
fn rope_row(v: &[f32], seg: usize, pos: usize, theta: f64) -> Vec<f32> {
    let half = seg / 2;
    let mut out = v[..seg].to_vec();
    for j in 0..half {
        let (a, b) = (v[2 * j], v[2 * j + 1]);
        let inv = theta.powf(-2.0 * j as f64 / seg as f64);
        let ang = pos as f64 * inv;
        let (cs, sn) = (ang.cos() as f32, ang.sin() as f32);
        out[j] = a * cs - b * sn;
        out[half + j] = b * cs + a * sn;
    }
    out
}

/// Interleaved RoPE over `count` rows of `stride`, rotating each row's first `seg`.
///
/// WHAT THIS DOES AND DOES NOT PROVE. The trig table is a f64 `powf`/`cos`/`sin` chain,
/// and both this oracle and the launcher evaluate it with the same Rust libm — so the
/// ANGLES are a consistency check, not an independent one, and a misread exponent in
/// the HIP would be reproduced identically on both sides. What is independently checked
/// is everything downstream of the table: the interleave-to-halves permutation, the
/// rotation's sign convention, the per-row `stride` addressing, and the read-before-
/// write discipline. The `pos = 0` shape below is the strongest of these precisely
/// because it removes the trig from the picture entirely.
///
/// TOLERANCE for pos > 0: the rotation is `a·cs - b·sn` in f32, which SPIR-V may
/// contract into an FMA unless the shader forbids it. That is a legitimate one-ULP
/// difference from the oracle's two roundings, so bytes are not available.
#[test]
fn rope_interleave_rotates_each_row() {
    let v = Validation::new();
    let mut r = Lcg(0x60E);
    // (count, stride, seg, pos, theta):
    //   3/40/32/0      pos = 0 -> a pure permutation, checked as BYTES (see below).
    //   4/100/34/4096  half = 17: ragged, a partial subgroup, and every lane past 17
    //                  idle. pos = 4096 is where the angle actually has to be right —
    //                  the relative error in theta^(-2j/seg) is multiplied by pos.
    //   2/2100/2048    seg/2 = 1024, the documented cap, and four times the workgroup
    //                  so the in-block loop over `half` runs several iterations.
    //   1/16/16        count = 1, stride == seg: no in-row tail, grid of one.
    for (count, stride, seg, pos, theta) in [
        (3usize, 40usize, 32usize, 0usize, 10000.0f64),
        (4, 100, 34, 4096, 10000.0),
        (2, 2100, 2048, 7, 10000.0),
        (1, 16, 16, 1, 500000.0),
    ] {
        // Distinct data per row, so a kernel that ignored blockIdx and rotated row 0
        // `count` times fails rather than coincides.
        let rows: Vec<Vec<f32>> = (0..count)
            .map(|_| (0..seg).map(|_| r.f()).collect())
            .collect();
        let bytes = count * stride * 4;
        let mut bb = poison(bytes + GUARD);
        for (s, row) in rows.iter().enumerate() {
            bb.write_at(s * stride * 4, &f32b(row)).expect("fill row");
        }
        // SAFETY: `base` is a live Buf device address holding count*stride f32 plus the
        // guard band; it outlives the sync.
        unsafe {
            launch_rope(bb.ptr_mut() as *mut f32, count, stride, seg, pos, theta).expect("launch");
        }
        let got = sync_readb(&bb, bytes + GUARD);
        let label = format!("rope c{count} st{stride} seg{seg} pos{pos}");
        assert_untouched(&got, bytes, &label);

        for (s, row) in rows.iter().enumerate() {
            let base = s * stride * 4;
            // The row's tail past `seg` is a guard region of its own — the kernel is
            // documented to touch only the first `seg` elements of each row, and a
            // stride/seg mix-up (or a `half` computed from stride) lands here, INSIDE
            // the allocation where the trailing guard band cannot see it.
            assert!(
                got[base + seg * 4..base + stride * 4]
                    .iter()
                    .all(|&b| b == 0xFF),
                "{label} row {s}: wrote past element {seg} of the row"
            );
            let want = rope_row(row, seg, pos, theta);
            let g = &got[base..base + seg * 4];
            let rl = format!("{label} row {s}");
            if pos == 0 {
                // BYTES. At pos = 0 every angle is 0, so cs = 1.0 and sn = 0.0 exactly
                // and the kernel reduces to `v[j] = a`, `v[half+j] = b` — a pure
                // de-interleave in which no value is rounded at all, contraction or not.
                // The header's precondition for a byte-exact compare is therefore met
                // trivially: the inputs are the outputs. `Lcg::f` cannot return exactly
                // zero (it would need (u>>32) = 2147483647.5), so the one ambiguity
                // that could arise — a signed zero out of `a - b*0.0`, which Vulkan does
                // not promise to preserve — is out of reach of this data.
                assert_bytes(&f32b(&want), g, &rl);
            } else {
                assert_close(&want, &f32v(g), &rl);
            }
        }
    }
    v.check("rope_interleave_rotates_each_row");
}

/// The linalg launchers' argument guards report errors rather than dispatching a
/// degenerate grid or tripping a VUID.
///
/// Each rope guard gets a POSITIVE control beside its negative, because every one of
/// them could be deleted and replaced by something stricter with the negatives still
/// green — `stride >= seg` unconditionally would reject the legal single-row case, and
/// `seg <= 32` would reject everything this kernel is for.
#[test]
fn linalg_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let (_s, p, q) = Scratch::new(4096);

    let rope = |count: usize, stride: usize, seg: usize| {
        // SAFETY: `q` is the Scratch destination — a live 4096-byte Buf that `_s` owns to
        // the end of this test, so it outlives the sync below. The cases ACCEPTED here
        // write at most 32 f32 into it; the rejected ones never reach a pointer.
        unsafe { launch_rope(q as *mut f32, count, stride, seg, 3, 10000.0) }
    };

    // SAFETY: n = 0 is rejected before any pointer is used.
    unsafe {
        assert!(
            launch_swiglu(p as *const f32, p as *const f32, 0, q as *mut f32).is_err(),
            "swiglu n = 0"
        );
        assert!(
            launch_rmsnorm(p as *const f32, p as *const f32, 0, 1e-6, q as *mut f32).is_err(),
            "rmsnorm n = 0"
        );
    }

    assert!(rope(0, 16, 16).is_err(), "rope count = 0");
    assert!(rope(1, 16, 0).is_err(), "rope seg = 0");
    // Odd seg: `half = seg/2` would truncate and the last element would never be read
    // or written, so the guard is the only thing standing between that and silence.
    assert!(rope(1, 16, 15).is_err(), "rope odd seg");
    assert!(
        rope(1, 16, 16).is_ok(),
        "rope even seg must still be accepted"
    );
    // seg/2 > 1024 — one past the cap. The ACCEPTED boundary, seg = 2048, is exercised
    // for real in `rope_interleave_rotates_each_row` rather than here, where the 4096
    // byte buffer could not hold its output.
    assert!(rope(1, 4096, 2050).is_err(), "rope seg/2 > 1024");
    // count > 1 with stride < seg: the rows would overlap and each block would rotate
    // bytes another block is mid-rotation on.
    assert!(rope(2, 8, 16).is_err(), "rope count > 1 with stride < seg");
    assert!(rope(1, 8, 16).is_ok(), "rope count = 1 ignores stride");
    assert!(
        rope(2, 16, 16).is_ok(),
        "rope stride == seg must be accepted"
    );

    // The quantised GEMVs' guards, none of which had a test. Two of them are this
    // backend's own additions rather than mirrors of a HIP arm — the word-alignment and
    // multiple-of-4 checks exist because the shaders read packed weights as 32-bit WORDS
    // (VK_KHR_8bit_storage is deliberately not required), which HIP does not do. An
    // untested guard is indistinguishable from its absence, and these two are the ones
    // standing between a sub-word-aligned slab pointer and a silently wrong decode.
    //
    // SAFETY: every case here is rejected before any pointer is dereferenced. `p` and `q`
    // are live 4096-byte Bufs regardless.
    unsafe {
        let fp8 = |o: usize, i: usize, block: usize, packed: *const u8| {
            launch_gemv_fp8(
                p as *const f32,
                packed,
                p as *const f32,
                o,
                i,
                block,
                1,
                q as *mut f32,
            )
        };
        assert!(fp8(0, 64, 128, p).is_err(), "gemv_fp8 o_dim = 0");
        assert!(fp8(8, 0, 128, p).is_err(), "gemv_fp8 i_dim = 0");
        assert!(fp8(8, 64, 0, p).is_err(), "gemv_fp8 block = 0");
        // blk_shift is a SHIFT, so a non-power-of-two tile would index the scale by a
        // floor rather than a quotient — mirrors HIP's 1003.
        assert!(
            fp8(8, 64, 96, p).is_err(),
            "gemv_fp8 block not a power of two"
        );
        assert!(
            fp8(8, 64, 128, p).is_ok(),
            "gemv_fp8 power-of-two block must be accepted"
        );
        // i_dim not a multiple of 4: row bases would stop being word-aligned after row 0.
        assert!(
            fp8(8, 66, 128, p).is_err(),
            "gemv_fp8 i_dim not a multiple of 4"
        );
        // A packed base that is not word-aligned. The shader's offsets are relative to
        // this address, so every word load would straddle. GPU-AV checks bounds, not
        // alignment, so nothing else would catch it.
        assert!(
            fp8(8, 64, 128, p.add(1)).is_err(),
            "gemv_fp8 packed not word-aligned"
        );

        assert!(
            launch_gemv_i8(p as *const f32, p, p as *const f32, 0, 64, 1, q as *mut f32).is_err(),
            "gemv_i8 o_dim = 0"
        );
        assert!(
            launch_gemv_i8(p as *const f32, p, p as *const f32, 8, 0, 1, q as *mut f32).is_err(),
            "gemv_i8 i_dim = 0"
        );
    }

    // Retire the accepted dispatches while their buffer is still alive. The guard test
    // is about the rejections, but the controls are real launches.
    device_sync().expect("sync");
    // Rejecting a bad argument must not itself trip the layer — a guard that returns
    // Err while leaving a half-built Vulkan object behind would show up here.
    v.check("linalg_guards_reject_degenerate_arguments");
}

// ---------------------------------------------------------------------------
// linalg.hip: the quantised GEMVs
// ---------------------------------------------------------------------------

/// `y[o] = scale[o] * Σ x[i]·(i8)packed[o·i_dim+i]` — lm_head to logits.
///
/// Like `embed_i8_row`, the shader reads packed int8 as u32 WORDS and sign-extends by
/// hand, so the failure that matters is a byte read as unsigned: invisible below 0x80,
/// a 256·scale error above it. Every row therefore carries the four extremes, and
/// `i_dim` is not a multiple of 4 so the extraction runs at every byte phase.
#[test]
fn gemv_i8_matches_the_host_oracle() {
    let v = Validation::new();
    let mut r = Lcg(0x18);
    // o_dim not a multiple of ROWS_PER_BLOCK (tail workgroup has idle subgroups);
    // i_dim not a multiple of 4 or 32 (byte phases, and a ragged wave stride).
    for (o_dim, i_dim) in [(37usize, 67usize), (256, 1024)] {
        let mut packed: Vec<u8> = (0..o_dim * i_dim)
            .map(|i| (i as u8).wrapping_mul(29).wrapping_add(7))
            .collect();
        for o in 0..o_dim {
            packed[o * i_dim..o * i_dim + 4].copy_from_slice(&[0x80, 0xFF, 0x7F, 0x00]);
        }
        let scale: Vec<f32> = (0..o_dim).map(|o| 0.011 * (o as f32 + 1.0)).collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
        let want = want_i8(&x, &packed, &scale, o_dim, i_dim);
        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = poison(o_dim * 4 + GUARD);
        // SAFETY: live Buf device addresses of the documented sizes; nothing dropped
        // before the sync. The addresses are taken first so the dispatch reads as one
        // line — `yp` is a raw address, so `yb` is free to be borrowed again below.
        let (xp, pp, sp) = (xb.ptr() as *const f32, pb.ptr(), sb.ptr() as *const f32);
        let yp = yb.ptr_mut() as *mut f32;
        unsafe { launch_gemv_i8(xp, pp, sp, o_dim, i_dim, 1, yp) }.expect("launch");
        let got = sync_readb(&yb, o_dim * 4 + GUARD);
        let label = format!("gemv_i8 {o_dim}x{i_dim}");
        assert_untouched(&got, o_dim * 4, &label);
        assert_close(&want, &f32v(&got[..o_dim * 4]), &label);
    }
    v.check("gemv_i8_matches_the_host_oracle");
}

/// `gemv_fp8` against `quant.rs::matvec_fp8`, in BOTH dispatch geometries.
///
/// The launcher picks split-K at `i_dim >= 4096`, so the two shapes below straddle that
/// threshold DELIBERATELY and each is reported separately. Testing only one side leaves
/// an entire geometry — a different thread-to-row mapping and a whole LDS combine —
/// unexecuted while the suite goes green, which is the same trap the 1024/300 append_kv
/// shape had. Combined shapes are efficient right up to the moment one goes red.
///
/// The block tile must be a power of two: `blk_shift` indexes the scale by a SHIFT since
/// the HIP side replaced its signed divide, and the launcher mirrors that rejection.
///
/// RAGGED `o_dim` IS THE THIRD SHAPE, AND IT EXISTS TO MAKE ONE `if` FIRE. Both original
/// shapes were multiples of `ROWS_PER_BLOCK`, so every workgroup was full and the
/// wave-per-row row guard — `if (o >= o_dim) return;` — was indistinguishable from its
/// absence: delete it and the suite stayed green. 253 = 31·8 + 5 leaves the last
/// workgroup with three waves that must retire without writing, and those three would
/// otherwise write rows 253..255 of a 253-row output, past the end.
///
/// The SPLIT-K arm's identical-looking guard is still unreachable, and that is a property
/// of the dispatch rather than an oversight: its grid is exactly `o_dim` workgroups, so
/// `o = gl_WorkGroupID.x` cannot exceed it. Stated rather than quietly left as apparent
/// coverage — it is kept because a future change to that grid would need it.
#[test]
fn gemv_fp8_matches_the_host_oracle() {
    let v = Validation::new();
    let mut shapes = Shapes::default();
    let block = 128usize;
    for (o_dim, i_dim, geometry) in [
        (256usize, 512usize, "wave-per-row"),
        (253, 512, "wave-per-row ragged"),
        (128, 4096, "split-K"),
    ] {
        // Seed off BOTH dims: the two i_dim = 512 shapes would otherwise draw identical
        // data, so a bug that happened to cancel on one would cancel on the other too.
        let mut r = Lcg(0xF8 ^ (i_dim as u64) ^ ((o_dim as u64) << 20));
        let sc_cols = i_dim / block;
        let (packed, scale, x, want) =
            gemv_fp8_case(&mut r, o_dim, i_dim, block, o_dim.div_ceil(block) * sc_cols);
        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = poison(o_dim * 4 + GUARD);
        let got = gemv_fp8(
            &xb,
            &pb,
            &sb,
            &mut yb,
            o_dim,
            i_dim,
            block,
            o_dim * 4 + GUARD,
        );
        let label = format!("gemv_fp8 {geometry} {o_dim}x{i_dim}");
        assert_untouched(&got, o_dim * 4, &label);
        shapes.close(&want, &f32v(&got[..o_dim * 4]), &label);
    }
    shapes.assert_all_passed("gemv_fp8");
    v.check("gemv_fp8_matches_the_host_oracle");
}

/// The split-K path reduces across workgroup partials in a fixed-order LDS combine, so
/// it gets the reproducibility test every reduction here gets — bytes, repeats, a decoy
/// between them, and a not-all-zero guard.
#[test]
fn gemv_fp8_splitk_is_bit_reproducible() {
    let v = Validation::new();
    let (o_dim, i_dim, block) = (64usize, 4096usize, 128usize);
    let mut r = Lcg(0x5B7);
    let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| f32_to_e4m3(r.f())).collect();
    let scale = block_scales(&mut r, o_dim.div_ceil(block) * (i_dim / block));
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = dev(&f32b(&vec![0.0f32; o_dim]));
    // A differently-shaped decoy, below the split-K threshold so it also perturbs which
    // pipeline geometry ran last.
    let (dx, dp, ds) = (
        dev(&f32b(&x[..512])),
        dev(&packed[..32 * 512]),
        dev(&f32b(&scale[..4])),
    );
    let mut dy = dev(&f32b(&[0.0f32; 32]));

    let run = |xb: &Buf, pb: &Buf, sb: &Buf, yb: &mut Buf, o: usize, i: usize| -> Vec<u8> {
        gemv_fp8(xb, pb, sb, yb, o, i, block, o * 4)
    };

    let first = assert_bit_reproducible(
        "gemv_fp8 split-K",
        || run(&xb, &pb, &sb, &mut yb, o_dim, i_dim),
        || run(&dx, &dp, &ds, &mut dy, 32, 512),
    );
    assert_not_all_zero(&first);
    v.check("gemv_fp8_splitk_is_bit_reproducible");
}

/// Twelve `(pos, j)` points of the RoPE trig table, against values computed OUTSIDE this
/// codebase's arithmetic.
///
/// WHY THIS EXISTS. `launch_rope` computes the table host-side in f64 — which is what
/// makes it bit-identical to HIP — and `rope_interleave_rotates_each_row`'s oracle
/// computes it the same way, with the same Rust libm. So that test is a CONSISTENCY
/// check on the table: a misread exponent would reproduce identically on both sides and
/// pass. The fix that bought bit-identity also removed the table from independent view,
/// and both halves are consequences of the same change.
///
/// These literals break the coupling. They were computed in Python with `decimal` at 60
/// digits: `inv = exp(-2j/seg · ln θ)` via Decimal's own ln/exp, exact range reduction
/// mod 2π against a 60-digit constant, then sin/cos by Taylor series — no `powf`, no
/// libm, nothing shared with the launcher.
///
/// SCOPE, stated so the coverage claim stays honest: THESE TWELVE POINTS are
/// independently verified. The rest of the table remains consistency-checked.
///
/// Extraction trick: with a row of pairs all set to (1, 0), the rotation reduces to
/// `v[j] = cos` and `v[half+j] = sin`, so the table is read straight out of the output.
#[test]
fn rope_trig_table_matches_independent_reference() {
    let v = Validation::new();
    // (pos, j, seg, cos bits, sin bits) — see the docstring for provenance.
    const REF: &[(usize, usize, usize, u32, u32)] = &[
        (0, 0, 64, 0x3f800000, 0x00000000),
        (0, 7, 64, 0x3f800000, 0x00000000),
        (1, 0, 64, 0x3f0a5140, 0x3f576aa4),
        (1, 1, 64, 0x3f3b54b0, 0x3f2e7ace),
        (4096, 0, 64, 0x3f4dd254, 0xbf183a75),
        (4096, 3, 64, 0x3f5241cc, 0xbf120a82),
        (4096, 31, 64, 0x3f5ac075, 0x3f04fadb),
        (137, 5, 64, 0x3ef4f90e, 0x3f60cbae),
        (100000, 0, 64, 0xbf7fd61c, 0x3d126d55),
        (100000, 17, 64, 0xbf15a6fd, 0x3f4fb3c1),
        (7, 2, 128, 0x3f02ee57, 0xbf5bfbf8),
        (65535, 63, 128, 0x3e908076, 0x3f7597c0),
    ];
    let theta = 10000.0f64;
    for &(pos, j, seg, want_cos, want_sin) in REF {
        let half = seg / 2;
        // All pairs (1, 0) so the rotation returns the table itself.
        let mut row = vec![0.0f32; seg];
        for k in 0..half {
            row[2 * k] = 1.0;
            row[2 * k + 1] = 0.0;
        }
        let mut b = dev(&f32b(&row));
        // SAFETY: one row of `seg` f32, live until the sync.
        unsafe { launch_rope(b.ptr_mut() as *mut f32, 1, seg, seg, pos, theta).expect("launch") };
        device_sync().expect("sync");
        let got = f32v(&read(&b, seg));
        let (gc, gs) = (got[j].to_bits(), got[half + j].to_bits());
        // Within 1 ULP, not bit-equal: the reference is a DIFFERENT algorithm at higher
        // precision, so the two f64 intermediates can straddle an f32 rounding boundary.
        // Exact agreement is expected and 1 ULP is the honest bound to assert.
        let ulp = |a: u32, b: u32| (a as i64 - b as i64).abs();
        assert!(
            ulp(gc, want_cos) <= 1,
            "rope cos(pos={pos}, j={j}, seg={seg}): got {gc:#010x}, independent reference              {want_cos:#010x} — the host-side trig table disagrees with a computation              that shares no arithmetic with it"
        );
        assert!(
            ulp(gs, want_sin) <= 1,
            "rope sin(pos={pos}, j={j}, seg={seg}): got {gs:#010x}, independent reference              {want_sin:#010x}"
        );
    }
    v.check("rope_trig_table_matches_independent_reference");
}

// ---------------------------------------------------------------------------
// mla.hip: the two fp8 MLA projections.
//
// Oracles derived from kernels/mla.hip, kernels/common.hpp and src/math.rs — the
// SPECIFICATION — and from nothing else. The usual rule bites harder here: the same
// person wrote these shaders, so an oracle they also wrote would be a consistency check
// wearing a correctness check's clothes even if derived honestly. The two `mla_*_oracle`
// functions were therefore written by someone working ONLY from the HIP sources, with the
// shader files withheld.
// ---------------------------------------------------------------------------

/// Every output element must match the oracle BIT FOR BIT.
///
/// THIS IS THE ASSERTION THAT GIVES THE SUMMATION-ORDER MODELLING ANY FORCE, and it
/// replaced a tolerance that could not do the job. `assert_close` compares at
/// `1e-3·mx + 1e-3`; two different summation orders of the same terms differ by ~1e-7.
/// Three orders of magnitude apart, so the tolerance cannot see order at all — MEASURED,
/// not assumed: replacing `oracle_wave_sum` with `partials.iter().sum()` left the test
/// passing at 27001x margin. Enlarging the shape does not help and no shape can, because
/// the gap is between the SIZE of an ordering perturbation and the SIZE of the tolerance.
///
/// Bit-identity is attainable here only because the oracle models what the hardware
/// actually does, INCLUDING FMA CONTRACTION — see `oracle_fp8_dot_strided`. That was
/// established by experiment rather than assumed: without the fused form, 10 of 15
/// elements differed; with it, err is exactly 0.
fn assert_bit_identical(want: &[f32], got: &[f32], label: &str) {
    let bad: Vec<String> = want
        .iter()
        .zip(got)
        .enumerate()
        .filter(|(_, (w, g))| w.to_bits() != g.to_bits())
        .map(|(i, (w, g))| format!("[{i}] want {:#010x} got {:#010x}", w.to_bits(), g.to_bits()))
        .collect();
    assert!(
        bad.is_empty(),
        "\n\n{label} is not BIT-IDENTICAL to its oracle ({} of {} elements):\n  {}\n\n\
         The oracle reproduces this kernel's arithmetic exactly, so a mismatch means the \
         shader's operation ORDER, GROUPING, or CONTRACTION has changed. INVESTIGATE; do \
         not relax this to a tolerance — the tolerance is blind to precisely this class.\n",
        bad.len(),
        want.len(),
        bad.join("\n  ")
    );
}

/// gfx1151 native wave width — `WAVE` in kernels/common.hpp.
const ORACLE_WAVE: usize = 32;

/// `common.hpp::wave_sum` — the fixed `__shfl_down` halving ladder, modelled over the 32
/// lane partials. Each rung reads the PRE-step values, because a shuffle is simultaneous.
///
/// A lane whose source `l + o` falls outside the group gets its own value back under HIP,
/// so it doubles; that is modelled rather than summarised. It is dead weight for lane 0,
/// which is the only lane the kernels store — and lane 0 is also the only lane whose
/// chain reads in-range values throughout, which is why the GLSL side is correct at lane
/// 0 despite SPIR-V leaving the out-of-range shuffle UNDEFINED (see common.glsl).
fn oracle_wave_sum(partials: [f32; ORACLE_WAVE]) -> f32 {
    let mut v = partials;
    let mut o = ORACLE_WAVE / 2;
    while o > 0 {
        let src = v;
        for l in 0..ORACLE_WAVE {
            let up = l + o;
            v[l] = src[l] + if up < ORACLE_WAVE { src[up] } else { src[l] };
        }
        o >>= 1;
    }
    v[0]
}

/// `common.hpp::fp8_dot_strided` — one lane's partial, NO cross-thread reduction.
///
/// Two loops with DELIBERATELY different groupings, both preserved: the dword loop takes
/// four consecutive columns and applies the block scale to the PARENTHESISED GROUP once
/// (using column `i0`'s scale for all four — valid only because `block >= 4` and `i0` is
/// a multiple of 4), while the scalar tail applies the scale PER ELEMENT.
///
/// FUSED MULTIPLY-ADD, AND IT IS NOT DECORATION. The GLSL source reads
/// `acc += s * (a + b + c + d)` with `a = x[i0]*lut[..]`, but the driver's backend
/// contracts each multiply-then-add into an FMA — invisible in SPIR-V, which still shows
/// OpFMul and OpFAdd, because contraction happens below it. Modelling the plain form
/// leaves 10 of 15 outputs differing in their low bits; modelling the fused chain makes
/// the oracle EXACT. Measured both ways.
///
/// The consequence worth knowing: a CPU oracle can be bit-exact with this GPU, but only
/// by reproducing a decision the shader source does not express. If a future driver
/// contracts differently, this is where it will show — as a bit mismatch, loudly, which
/// is the point of asserting bit-identity rather than a tolerance.
fn oracle_fp8_dot_strided(
    x: &[f32],
    wrow: &[u8],
    scalerow: &[f32],
    i_dim: usize,
    block: usize,
    start: usize,
    stride: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let n4 = i_dim >> 2;
    let mut j = start;
    while j < n4 {
        let i0 = j << 2;
        let s = scalerow[i0 / block];
        let mut t = x[i0] * e4m3_to_f32(wrow[i0]);
        t = x[i0 + 1].mul_add(e4m3_to_f32(wrow[i0 + 1]), t);
        t = x[i0 + 2].mul_add(e4m3_to_f32(wrow[i0 + 2]), t);
        t = x[i0 + 3].mul_add(e4m3_to_f32(wrow[i0 + 3]), t);
        acc = s.mul_add(t, acc);
        j += stride;
    }
    let mut i = (n4 << 2) + start;
    while i < i_dim {
        acc += x[i] * e4m3_to_f32(wrow[i]) * scalerow[i / block];
        i += stride;
    }
    acc
}

/// CPU oracle for `mla_absorb_fp8`.
///
/// One GPU thread owns one `(head, i)`, so the `d`-loop is a private accumulator summed
/// strictly left to right — no shuffle, no LDS beyond the read-only LUT. `w = e4m3·scale`
/// is formed FIRST and then multiplied by `q[d]`, matching the kernel's grouping.
///
/// Scale traversal: `i` is FIXED and the ROW varies with `d`, so this walks DOWN a column
/// of the block-scale grid — the transpose of every other fp8 kernel here.
fn mla_absorb_oracle(q: &[f32], kvb: &[u8], kvb_scale: &[f32], m: Mla) -> Vec<f32> {
    let (h, qh, nope) = (m.h, m.qh, m.nope);
    let (vh, kvl, block) = (m.vh, m.kvl, m.block);
    m.assert_guarded();
    assert!(qh > 0, "guard 1001");
    // NOT launcher-guarded, but the kernel reads q[head·qh .. +nope).
    assert!(qh >= nope, "q head stride {qh} shorter than nope {nope}");

    let rows = m.rows();
    let sc_cols = kvl.div_ceil(block);
    assert_eq!(
        kvb.len(),
        rows * kvl,
        "kv_b is [H·(nope+vh), kvl], row stride kvl BYTES"
    );
    assert_eq!(kvb_scale.len(), rows.div_ceil(block) * sc_cols);

    let mut qabs = vec![0.0f32; h * kvl];
    for head in 0..h {
        let rbase = head * (nope + vh);
        let qrow = &q[head * qh..head * qh + nope];
        for i in 0..kvl {
            let mut acc = 0.0f32;
            for (d, qd) in qrow.iter().enumerate() {
                let row = rbase + d;
                let w = e4m3_to_f32(kvb[row * kvl + i])
                    * kvb_scale[(row / block) * sc_cols + i / block];
                // `acc += q·w` contracts to a fused multiply-add on this driver; see
                // oracle_fp8_dot_strided.
                acc = qd.mul_add(w, acc);
            }
            qabs[head * kvl + i] = acc;
        }
    }
    qabs
}

/// CPU oracle for `mla_value_fp8`.
///
/// One WAVE per output row, and this reproduces that STRUCTURE rather than summing
/// sequentially: 32 independent lane partials, lane `l` owning dword groups `l, l+32, …`
/// plus its slice of the scalar tail, then the fixed halving ladder.
///
/// Scale traversal: the ROW is fixed per output element and the COLUMN varies with `i` —
/// the opposite of `mla_absorb_oracle`. The kernel hoists the row out; so does this.
fn mla_value_oracle(clat: &[f32], kvb: &[u8], kvb_scale: &[f32], m: Mla) -> Vec<f32> {
    let (h, nope, vh) = (m.h, m.nope, m.vh);
    let (kvl, block) = (m.kvl, m.block);
    m.assert_guarded();
    // Not launcher-guarded on the HIP side, but required for the dword load to be
    // well-defined: rows are `kvl` bytes apart, so an odd kvl misaligns every other row.
    // The Vulkan launcher DOES reject it — one place this backend is stricter.
    assert_eq!(
        kvl % 4,
        0,
        "kv_b rows must stay 4-byte aligned for the dword load"
    );
    // Below 4, the dword group's single scale would straddle a scale-tile boundary.
    assert!(
        block >= 4,
        "the 4-wide dword group must sit inside one scale tile"
    );

    let rows = h * (nope + vh);
    let sc_cols = kvl.div_ceil(block);
    assert_eq!(clat.len(), h * kvl);
    assert_eq!(kvb.len(), rows * kvl);
    assert_eq!(kvb_scale.len(), rows.div_ceil(block) * sc_cols);

    let mut ctx = vec![0.0f32; h * vh];
    for r in 0..h * vh {
        let head = r / vh;
        let j = r % vh;
        let row = head * (nope + vh) + nope + j; // skip the head's nope absorb rows
        let x = &clat[head * kvl..head * kvl + kvl];
        let wrow = &kvb[row * kvl..row * kvl + kvl];
        let scalerow = &kvb_scale[(row / block) * sc_cols..][..sc_cols];

        let mut lanes = [0.0f32; ORACLE_WAVE];
        for (lane, partial) in lanes.iter_mut().enumerate() {
            *partial = oracle_fp8_dot_strided(x, wrow, scalerow, kvl, block, lane, ORACLE_WAVE);
        }
        ctx[head * vh + j] = oracle_wave_sum(lanes);
    }
    ctx
}

/// The fp8 kv_b block and its block-scale grid, as device addresses.
///
/// The only two of these launchers' eleven arguments that always travel together, and
/// pairing them is what keeps each wrapper's signature on ONE line — which is in turn what
/// keeps the two from being two copies of the same five-line parameter list.
#[derive(Clone, Copy)]
struct KvPtr {
    b: *const u8,
    scale: *const f32,
}

impl KvPtr {
    fn new(b: *const u8, scale: *const f32) -> Self {
        Self { b, scale }
    }
}

/// One `mla_absorb_fp8` dispatch, from BARE device addresses and `nrow = 1`.
///
/// Raw addresses because the two callers hold different things — the oracle test owns
/// `Buf`s, the guard test dispatches from a [`Scratch`]'s pair of pointers — and this
/// launcher's eleven arguments are worth spelling exactly once either way.
///
/// # Safety
/// `q` addresses `h`·`qh` live f32, `kv.b` `Mla::rows()`·`kvl` bytes, `kv.scale` their
/// block-scale grid, and `out` `h`·`kvl` f32; all four outlive the caller's next join.
unsafe fn mla_absorb_at(m: Mla, q: *const f32, kv: KvPtr, out: *mut f32) -> anyhow::Result<()> {
    let (b, sc) = (kv.b, kv.scale);
    unsafe { launch_mla_absorb_fp8(q, b, sc, m.h, m.qh, m.nope, m.vh, m.kvl, m.block, 1, out) }
}

/// One `mla_value_fp8` dispatch, the same way. `qh` is absent: this kernel reads `clat`,
/// which is already one contiguous `kvl` row per head.
///
/// # Safety
/// As [`mla_absorb_at`], with `clat` addressing `h`·`kvl` f32 and `out` `h`·`vh`.
unsafe fn mla_value_at(m: Mla, clat: *const f32, kv: KvPtr, out: *mut f32) -> anyhow::Result<()> {
    let (b, sc) = (kv.b, kv.scale);
    unsafe { launch_mla_value_fp8(clat, b, sc, m.h, m.nope, m.vh, m.kvl, m.block, 1, out) }
}

/// Buffers shared by both MLA oracles: fp8 weights, their block scales, and the f32 input.
///
/// `rows` is the full kv_b row count, `h·(nope+vh)`; the block scale is a 2-D tile grid
/// over (rows × kvl), so its length is `⌈rows/block⌉·⌈kvl/block⌉`.
fn mla_inputs(
    seed: u64,
    xn: usize,
    rows: usize,
    kvl: usize,
    block: usize,
) -> (Vec<f32>, Vec<u8>, Vec<f32>) {
    let mut r = Lcg(seed);
    let x: Vec<f32> = (0..xn).map(|_| r.f()).collect();
    let kvb: Vec<u8> = (0..rows * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let scale = block_scales(&mut r, rows.div_ceil(block) * kvl.div_ceil(block));
    (x, kvb, scale)
}

/// `mla_value_fp8` against the CPU oracle.
///
/// `h·vh = 15` is DELIBERATELY not a multiple of `ROWS_PER_BLOCK`: the kernel maps one
/// wave per output row, so a full-workgroup shape would leave `if (r >= h·vh) return;`
/// unexercised and the trailing waves would write past `ctx`.
///
/// `kvl = 256` IS ALSO DELIBERATE, AND AN EARLIER 64 MADE THIS TEST PROVE LESS THAN IT
/// LOOKED LIKE. The dot's dword loop runs `n4 = kvl/4` iterations shared across `WAVE`
/// lanes. At kvl = 64, `n4 = 16 < 32`: lanes 16..31 did nothing at all, and every active
/// lane ran EXACTLY ONE iteration — so the strided accumulation `oracle_fp8_dot_strided`
/// exists to model was never exercised, and a naive ascending sum agreed to well within
/// tolerance. The oracle was faithful and the test could not tell. At 256, `n4 = 64`, so
/// all 32 lanes are busy and each loops twice; the per-lane order and the full 32-lane
/// ladder both become load-bearing.
#[test]
fn mla_value_fp8_matches_the_host_oracle() {
    let v = Validation::new();
    let (h, nope, vh, kvl, block) = (3usize, 8usize, 5usize, 256usize, 16usize);
    let m = Mla::value_dims(h, nope, vh, kvl, block);
    let (clat, kvb, scale) = mla_inputs(0x5A1, h * kvl, m.rows(), kvl, block);

    let want = mla_value_oracle(&clat, &kvb, &scale, m);

    let (cb, kb, sb) = (dev(&f32b(&clat)), dev(&kvb), dev(&f32b(&scale)));
    let mut out = poison(h * vh * 4 + GUARD);
    // SAFETY: live Bufs of the documented sizes; `out` holds h·vh f32 plus a guard band.
    unsafe {
        let kv = KvPtr::new(kb.ptr(), sb.ptr() as *const f32);
        mla_value_at(m, cb.ptr() as *const f32, kv, out.ptr_mut() as *mut f32)
    }
    .expect("launch");
    let got = sync_readb(&out, h * vh * 4 + GUARD);
    assert_untouched(&got, h * vh * 4, "mla_value_fp8");
    let got_f = f32v(&got[..h * vh * 4]);
    assert_close(&want, &got_f, "mla_value_fp8 h3 vh5 kvl256");
    assert_bit_identical(&want, &got_f, "mla_value_fp8");
    v.check("mla_value_fp8_matches_the_host_oracle");
}

/// `mla_absorb_fp8` against the CPU oracle.
///
/// `kvl = 37` is RAGGED on purpose, and this kernel is the only one that can take it. It
/// reads a single byte per thread by masking the containing word's address down, so an
/// odd row stride is handled rather than banned — unlike `mla_value_fp8`, which walks
/// rows with the word-loading shared MAC and rejects a `kvl` that is not a multiple of 4.
/// A tidy `kvl` would leave that masking untested, and it is the whole of the difference
/// between this kernel's memory access and every other fp8 kernel's.
///
/// `h·kvl = 111` also leaves most of the single 256-thread workgroup retiring early,
/// which is the bounds guard.
#[test]
fn mla_absorb_fp8_matches_the_host_oracle() {
    let v = Validation::new();
    let (h, qh, nope, vh, kvl, block) = (3usize, 12usize, 8usize, 5usize, 37usize, 16usize);
    let m = Mla::new(h, qh, nope, vh, kvl, block);
    let (q, kvb, scale) = mla_inputs(0xAB50, h * qh, m.rows(), kvl, block);

    let want = mla_absorb_oracle(&q, &kvb, &scale, m);

    let (qb, kb, sb) = (dev(&f32b(&q)), dev(&kvb), dev(&f32b(&scale)));
    let mut out = poison(h * kvl * 4 + GUARD);
    // SAFETY: live Bufs of the documented sizes; `out` holds h·kvl f32 plus a guard band.
    unsafe {
        let kv = KvPtr::new(kb.ptr(), sb.ptr() as *const f32);
        mla_absorb_at(m, qb.ptr() as *const f32, kv, out.ptr_mut() as *mut f32)
    }
    .expect("launch");
    let got = sync_readb(&out, h * kvl * 4 + GUARD);
    assert_untouched(&got, h * kvl * 4, "mla_absorb_fp8");
    let got_f = f32v(&got[..h * kvl * 4]);
    assert_close(&want, &got_f, "mla_absorb_fp8 h3 kvl37 nope8");
    assert_bit_identical(&want, &got_f, "mla_absorb_fp8");
    v.check("mla_absorb_fp8_matches_the_host_oracle");
}

// ---------------------------------------------------------------------------
// attn.hip: MLA flash attention.
//
// THE ORACLE SPLITS AT THE `exp` BOUNDARY, and that is a decision recorded in
// docs/investigations/vulkan-port.md before this code was written. Everything upstream of the first `exp` is
// ordinary arithmetic and gets BIT-EXACT treatment; everything downstream cannot, because
// Rust's `exp` and GLSL's `exp` are different functions and no care closes that. Since a
// tolerance is categorically blind to reordering (see `assert_bit_identical`), accepting
// one across the whole kernel would have left the score reduction's ordering — the biggest
// hazard in the kernel — untested. Splitting keeps it testable.
// ---------------------------------------------------------------------------

const ATT_HB: usize = 8;
const ATT_TILE: usize = 16;
const ATT_MAX_SPLITS: usize = 16;
const ATT_MIN_TILES_PER_SPLIT: usize = 4;
const ATT_TARGET_BLOCKS: usize = 80;

/// The attention's five input arrays: the per-head queries, then the token cache.
///
/// Owned rather than borrowed because every caller builds them from one [`att_inputs`]
/// draw and then hands the same set to both the oracle and the kernel — which is the
/// property that makes the comparison mean anything.
struct AttIn {
    qabs: Vec<f32>,
    qrope: Vec<f32>,
    lc8: Vec<u8>,
    lscale: Vec<f32>,
    rc: Vec<u16>,
}

/// `attn.hip::mla_plan_splits`, transliterated. The cut fixes which rows each split
/// reduces, so it fixes the summation order and therefore the bits.
fn att_plan(h: usize, nr: usize, have_scratch: bool) -> (usize, usize) {
    let ntiles = nr.div_ceil(ATT_TILE);
    let hblocks = h.div_ceil(ATT_HB);
    let by_work = ntiles / ATT_MIN_TILES_PER_SPLIT;
    let by_grid = ATT_TARGET_BLOCKS.div_ceil(hblocks);
    let mut n = by_work.min(by_grid).min(ATT_MAX_SPLITS);
    if n < 1 || !have_scratch {
        n = 1;
    }
    let tps = ntiles.div_ceil(n);
    (ntiles.div_ceil(tps), tps)
}

/// The staged tile, widened exactly as the shader widens it: latent fp8 × its per-128
/// block scale, roped key bf16 → f32. `exp`-free, so this part is bit-exact.
fn att_widen(src: &AttIn, t: usize, d: Att) -> (Vec<f32>, Vec<f32>) {
    let (kvl, rope, n_blocks) = (d.kvl, d.rope, d.n_blocks);
    let l: Vec<f32> = (0..kvl)
        .map(|i| e4m3_to_f32(src.lc8[t * kvl + i]) * src.lscale[t * n_blocks + i / 128])
        .collect();
    let r: Vec<f32> = (0..rope)
        .map(|d| f32::from_bits((src.rc[t * rope + d] as u32) << 16))
        .collect();
    (l, r)
}

/// One split's online-softmax pass, returning its unnormalised accumulator and (m, l).
///
/// The per-lane score partials and the halving ladder are modelled explicitly, for the
/// same reason `mla_value_oracle` models them: a left-to-right sum is a different number.
/// The accumulator update reproduces the FORCED contraction — a plain multiply for
/// `acc*corr`, a fused multiply-add for `p*Lt` — matching what the shader now spells with
/// an explicit `fma`, and what hipcc's ISA shows.
fn att_split(
    qa: &[f32],
    qr: &[f32],
    src: &AttIn,
    rows: Option<&[u32]>,
    (r_begin, r_end): (usize, usize),
    d: Att,
) -> (Vec<f32>, f32, f32) {
    let (kvl, rope, scale) = (d.kvl, d.rope, d.scale);
    let mut acc = vec![0.0f32; kvl];
    let mut m = f32::NEG_INFINITY;
    let mut l = 0.0f32;

    let mut t0 = r_begin;
    while t0 < r_end {
        let tcount = ATT_TILE.min(r_end - t0);
        for tt in 0..tcount {
            let t = rows.map_or(t0 + tt, |r| r[t0 + tt] as usize);
            let (lt, rt) = att_widen(src, t, d);

            // Per-lane strided partials: kvl first, then rope, into the SAME accumulator,
            // in that order — as the two loops in attn.hip do.
            // FUSED, for the same measured reason as `oracle_fp8_dot_strided`: the driver
            // contracts `part += a*b` into an FMA below SPIR-V, and hipcc does too. This
            // loop models the strided ORDER; modelling the order while dropping the
            // GROUPING would leave the oracle disagreeing with the shader in the low bits
            // of every score — invisible under `assert_close`, and a spurious failure for
            // whoever later tries to widen the bit-exact assertion past nr = 1.
            let mut lanes = [0.0f32; ORACLE_WAVE];
            for (lane, part) in lanes.iter_mut().enumerate() {
                let mut i = lane;
                while i < kvl {
                    *part = qa[i].mul_add(lt[i], *part);
                    i += ORACLE_WAVE;
                }
                let mut d = lane;
                while d < rope {
                    *part = qr[d].mul_add(rt[d], *part);
                    d += ORACLE_WAVE;
                }
            }
            let s = oracle_wave_sum(lanes) * scale;

            let m_new = m.max(s);
            let corr = (m - m_new).exp();
            let p = (s - m_new).exp();
            for (i, a) in acc.iter_mut().enumerate() {
                *a = p.mul_add(lt[i], *a * corr);
            }
            l = l.mul_add(corr, p);
            m = m_new;
        }
        t0 += ATT_TILE;
    }
    (acc, m, l)
}

/// CPU oracle for the whole attention, splits and all.
fn attend_oracle(src: &AttIn, rows: Option<&[u32]>, d: Att, have_scratch: bool) -> Vec<f32> {
    let (h, nr, kvl, rope) = (d.h, d.nr, d.kvl, d.rope);
    let (n_splits, tps) = att_plan(h, nr, have_scratch);
    let mut clat = vec![0.0f32; h * kvl];
    for head in 0..h {
        let qa = &src.qabs[head * kvl..][..kvl];
        let qr = &src.qrope[head * rope..][..rope];
        let mut parts = Vec::new();
        for s in 0..n_splits {
            let r_begin = s * tps * ATT_TILE;
            let r_end = nr.min(r_begin + tps * ATT_TILE);
            parts.push(att_split(qa, qr, src, rows, (r_begin, r_end), d));
        }
        if n_splits == 1 {
            let (acc, _, l) = &parts[0];
            let inv = if *l > 0.0 { 1.0 / *l } else { 0.0 };
            for i in 0..kvl {
                clat[head * kvl + i] = acc[i] * inv;
            }
            continue;
        }
        // The combine: ascending split order in both reductions, as the kernel does.
        let m_g = parts
            .iter()
            .fold(f32::NEG_INFINITY, |a, (_, m, _)| a.max(*m));
        let finite = !m_g.is_infinite() && !m_g.is_nan();
        let w: Vec<f32> = parts
            .iter()
            .map(|(_, m, _)| if finite { (*m - m_g).exp() } else { 0.0 })
            .collect();
        let mut l_g = 0.0f32;
        for (s, (_, _, l)) in parts.iter().enumerate() {
            l_g = l.mul_add(w[s], l_g);
        }
        let inv = if l_g > 0.0 { 1.0 / l_g } else { 0.0 };
        for i in 0..kvl {
            let mut sum = 0.0f32;
            for (s, (acc, _, _)) in parts.iter().enumerate() {
                sum = acc[i].mul_add(w[s], sum);
            }
            clat[head * kvl + i] = sum * inv;
        }
    }
    clat
}

/// Inputs for the attention tests. `tokens` is the CACHE depth, which the DSA test makes
/// larger than `d.nr` so a selected row lands on a live token rather than out of bounds.
fn att_inputs(seed: u64, tokens: usize, d: Att) -> AttIn {
    let (h, kvl, rope) = (d.h, d.kvl, d.rope);
    let mut r = Lcg(seed);
    let qabs: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| r.f()).collect();
    let lc8: Vec<u8> = (0..tokens * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let lscale = block_scales(&mut r, tokens * d.n_blocks);
    let rc: Vec<u16> = (0..tokens * rope).map(|_| f32_to_bf16(r.f())).collect();
    AttIn {
        qabs,
        qrope,
        lc8,
        lscale,
        rc,
    }
}

/// Dispatch the attention and read back `clat`.
fn run_attend(src: &AttIn, rows: Option<&[u32]>, d: Att, with_scratch: bool) -> Vec<f32> {
    let (h, nr, kvl, rope) = (d.h, d.nr, d.kvl, d.rope);
    let rcb: Vec<u8> = src.rc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let (qb, rb) = (dev(&f32b(&src.qabs)), dev(&f32b(&src.qrope)));
    let (lb, sb, kb) = (dev(&src.lc8), dev(&f32b(&src.lscale)), dev(&rcb));
    let mut out = poison(h * kvl * 4 + GUARD);
    let mut scratch = with_scratch.then(|| {
        Buf::new(rivoli::backend::vk::attend_scratch_floats(h, kvl) * 4).expect("scratch")
    });
    let pp = scratch
        .as_mut()
        .map_or(std::ptr::null_mut(), |b| b.ptr_mut() as *mut f32);
    // The DSA row-selection buffer, or null for dense. Kept alive across the launch.
    let rowsb = rows.map(|r| dev(&r.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()));
    let rowsp = rowsb
        .as_ref()
        .map_or(std::ptr::null(), |b| b.ptr() as *const u32);
    // SAFETY: every buffer is live and of the documented size; `rows` is null for dense
    // or an `nr`-entry u32 buffer outliving the sync.
    unsafe {
        launch_attend(
            qb.ptr() as *const f32,
            rb.ptr() as *const f32,
            lb.ptr(),
            sb.ptr() as *const f32,
            kb.ptr() as *const u16,
            rowsp,
            h,
            nr,
            kvl,
            rope,
            d.n_blocks,
            d.scale,
            out.ptr_mut() as *mut f32,
            pp,
        )
    }
    .expect("launch attend");
    let got = sync_readb(&out, h * kvl * 4 + GUARD);
    assert_untouched(&got, h * kvl * 4, "mla_attend");
    f32v(&got[..h * kvl * 4])
}

/// The `exp`-FREE HALF, ASSERTED BIT-EXACTLY.
///
/// At `nr = 1` the softmax is degenerate: `m` starts at −inf so `m_new = s`. The output
/// becomes the widened latent row, which isolates the tile widening — `e4m3(Lc8) × Lscale`,
/// the whole fp8 decode path — and pins it bit for bit.
///
/// THE ARGUMENT, PRECISELY, because a sloppier version of it named the wrong dependency.
/// `corr = exp(m − m_new)` is IRRELEVANT whatever it evaluates to: `acc` and `l` both start
/// at zero, so `acc*corr` is zero for any finite `corr` and `l = fma(l, corr, p)` collapses
/// to `p`. What the test actually rests on is `p = exp(0)` being exactly `1.0` — the shader
/// computes `fma(p, Lt, 0) = p·Lt`, then `inv = 1/p`, then writes `(p·Lt)·(1/p)`, and those
/// three roundings cancel only at `p == 1.0` exactly. At `1.0 + 1 ulp` they do not.
///
/// GLSL specifies `exp` to 3 ULP and says nothing about `exp(0)`, so this is an OBSERVED
/// property of this driver, not a guarantee. It holds everywhere anyone has looked, and the
/// failure would be loud rather than silent — but `assert_bit_identical`'s message tells the
/// reader to investigate the shader's order, grouping or contraction, which would be the
/// wrong file. Hence stating the real dependency here.
///
/// It does NOT cover the score reduction: with one row the score cannot influence the
/// output at all. That gap is stated rather than papered over, and it is why the tolerance
/// test below still matters.
#[test]
fn attend_tile_widening_is_bit_exact() {
    let v = Validation::new();
    let d = Att::new(3, 1, 128, 64, 0.125);
    let src = att_inputs(0xA77, d.nr, d);
    let want = attend_oracle(&src, None, d, false);
    let got = run_attend(&src, None, d, false);
    assert_close(&want, &got, "attend widening h3 kvl128 nr1");
    assert_bit_identical(&want, &got, "mla_latent_attend (nr=1, exp-free)");
    v.check("attend_tile_widening_is_bit_exact");
}

/// The full attention, at tolerance because the answer is downstream of `exp`.
///
/// `nr = 128` and `h = 8` drive the split planner to TWO splits, so `mla_attend_combine`
/// runs — a single-split shape would leave that kernel unexecuted while the suite went
/// green. The single-split path is covered by the bit-exact test above.
///
/// `kvl = 512` IS THE PRODUCTION SHAPE AND IT COVERS AN INDEX NOTHING ELSE DOES. It is the
/// GLM `kv_lora_rank`, it fills all 16 `acc` registers, it fills the static LDS tile, and
/// — the reason it is here — it makes `n_blocks = 4`, so the tile load's
/// `Lscale[t*n_blocks + i/128]` actually VARIES with `i`. At kvl = 128 that term is
/// identically zero and a wrong block-scale index inside a token row would be invisible.
#[test]
fn attend_matches_the_host_oracle() {
    let v = Validation::new();
    let d = Att::new(8, 128, 512, 64, 0.125);
    let src = att_inputs(0xA78, d.nr, d);
    // The plan must agree with the launcher's, or the two are reducing different rows.
    assert_eq!(
        att_plan(d.h, d.nr, true).0,
        2,
        "shape must exercise the split combine"
    );
    let want = attend_oracle(&src, None, d, true);
    let got = run_attend(&src, None, d, true);
    assert_close(&want, &got, "attend h8 nr128 kvl512 splits2 nblocks4");
    v.check("attend_matches_the_host_oracle");
}

/// The DSA `rows` path — the indirection that was DEAD CODE in both the shader and the
/// oracle until this test existed.
///
/// `rows[j]` is the cached token index of the j-th attended row, so the tile load reads
/// `rows[t0 + tt]` instead of `t0 + tt`. Every other line of the kernel is row-POSITION
/// arithmetic and cannot tell the difference, which is exactly why nothing else covers it.
///
/// THE SELECTION IS DELIBERATELY NOT THE IDENTITY, and not even a permutation of a
/// contiguous prefix. It picks a strided, reversed subset out of a token pool four times
/// larger than `nr`, so:
///
///   - an implementation that ignored `rows` entirely would read tokens 0..nr and differ;
///   - one that read `rows` but forgot the `t0` tile offset would still be wrong after the
///     first tile, which is why `nr` spans two tiles rather than one;
///   - and because the pool is larger than `nr`, a wrong index lands on a REAL token's
///     data rather than out of bounds, so this fails as wrong numbers rather than as a
///     GPU-AV report. That is the failure mode worth building the test around.
#[test]
fn attend_honours_the_dsa_row_selection() {
    let v = Validation::new();
    let d = Att::new(3, 24, 128, 64, 0.125);
    let tokens = 96usize; // four times nr, so selected indices are scattered and live
    let src = att_inputs(0xD5A, tokens, d);

    // Strided and descending: rows[j] = tokens - 2 - 4j — the 24 values {2, 6, …, 94},
    // sparse within the live 96-token pool, which is what makes a wrong index land on a
    // real token's data rather than out of bounds.
    //
    // THE BASE IS CONSTRAINED, AND THE OBVIOUS `- 1` VIOLATES THE ASSERT BELOW. Any
    // descending selection starting above 0 and ending below `nr - 1` must CROSS the
    // diagonal; the only question is whether it crosses ON an integer. For
    // `rows[j] = tokens - base - s*j` it does iff `(s + 1)` divides `(tokens - base)` with
    // the quotient below `nr`. At `s = 4` that means `tokens - base` must not be a multiple
    // of 5: `base = 1` gives 95 = 5·19, so `rows[19] == 19`. `base = 2` gives 94, crossing
    // at j = 18.8 — between samples. Re-check the divisibility after changing `tokens`,
    // `nr` or the stride. Bases 2, 3 and 4 are all fixed-point-free and in range (base 5
    // runs `rows` negative); 2 is the smallest of them and so spans widest, up to 94.
    //
    // It is a DESIGN invariant, not a detection guarantee, and the distinction is worth
    // keeping: the output is indexed by (head, latent dim) and never by row position, so
    // one shared entry costs almost nothing — the control below separates at 27x the
    // tolerance here and still managed 26x with the `- 1` form. Keep the assert because it
    // is cheap and machine-checks "the two mappings are unrelated", not because the test
    // was meaningfully weak without it.
    let rows: Vec<u32> = (0..d.nr).map(|j| (tokens - 2 - 4 * j) as u32).collect();
    assert!(
        rows.iter().enumerate().all(|(j, r)| *r as usize != j),
        "the selection must share no fixed point with the dense mapping"
    );

    let want = attend_oracle(&src, Some(&rows), d, false);
    let got = run_attend(&src, Some(&rows), d, false);
    assert_close(&want, &got, "attend dsa rows h3 nr24 kvl128");

    // CONTROL: the same shape run DENSE must differ. Without this the test would pass for
    // an implementation that ignored `rows` and happened to agree — the oracle would be
    // reading the same tokens the shader did, both wrongly.
    let dense = run_attend(&src, None, d, false);
    // THE CONTROL MEASURES WHAT THE ASSERTION MEASURES. An earlier version counted BIT
    // inequality while the assertion it protects is `assert_close` at `1e-3*mx + 1e-3` —
    // three orders of magnitude apart, so the control could pass while the test it guards
    // was vacuous. Concretely: shrink the token pool to `tokens = nr` and the two paths
    // consume the SAME tokens in a different order, differing only in accumulation, so
    // every element differs in its low bits (the old control reports 384/384 and passes)
    // while `max|want − dense|` is ~1e-7 — far inside the tolerance. A shader ignoring
    // `rows` would then produce `got == dense` and `assert_close` would pass, with the
    // indirection back to being dead code.
    let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let sep = want
        .iter()
        .zip(&dense)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let tol = 1e-3 * mx + 1e-3;
    assert!(
        sep > tol,
        "the dense and row-selected results must differ by more than assert_close's own \
         tolerance (max separation {sep:.3e} vs tol {tol:.3e}) — otherwise a shader that \
         ignored `rows` entirely would still pass this test"
    );
    v.check("attend_honours_the_dsa_row_selection");
}

/// `launch_attend`'s guards, including the one this backend adds.
#[test]
fn attend_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let (_s, p, q) = Scratch::new(8192);

    // SAFETY: every case is rejected before a pointer is dereferenced.
    unsafe {
        let att = |h: usize, nr: usize, kvl: usize, rope: usize, nb: usize, lc8: *const u8| {
            launch_attend(
                p as *const f32,
                p as *const f32,
                lc8,
                p as *const f32,
                p as *const u16,
                std::ptr::null(),
                h,
                nr,
                kvl,
                rope,
                nb,
                0.125,
                q as *mut f32,
                std::ptr::null_mut(),
            )
        };
        assert!(att(0, 1, 128, 64, 1, p).is_err(), "attend h = 0");
        assert!(att(1, 0, 128, 64, 1, p).is_err(), "attend nr = 0");
        assert!(att(1, 1, 0, 64, 1, p).is_err(), "attend kvl = 0");
        assert!(att(1, 1, 128, 0, 1, p).is_err(), "attend rope = 0");
        assert!(att(1, 1, 128, 64, 0, p).is_err(), "attend n_blocks = 0");
        // kvl must be a multiple of 128: 160 would satisfy %SUBW, give n_blocks = 1, and
        // silently read the NEXT token's block scale for i in [128, 160).
        assert!(
            att(1, 1, 160, 64, 1, p).is_err(),
            "attend kvl not a multiple of 128"
        );
        assert!(
            att(1, 1, 640, 64, 5, p).is_err(),
            "attend kvl over the register cap"
        );
        // STRICTER THAN HIP: the static LDS tile bounds rope, where attn.hip sizes it
        // dynamically. Deliberately this backend's own limit, so it gets its own case.
        assert!(
            att(1, 1, 128, 128, 1, p).is_err(),
            "attend rope over the static LDS tile"
        );
        assert!(
            att(1, 1, 128, 64, 1, p.add(1)).is_err(),
            "attend lc8 not word-aligned"
        );
    }
    device_sync().expect("sync");
    v.check("attend_guards_reject_degenerate_arguments");
}

/// Both MLA launchers' argument guards, mirroring `mla.hip`'s 1001/1003 arms plus this
/// backend's word-addressing additions.
#[test]
fn mla_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let (_s, p, q) = Scratch::new(4096);

    // SAFETY: every case is rejected before a pointer is dereferenced; the accepted
    // controls dispatch over live 4096-byte Bufs and are retired by the sync below.
    unsafe {
        let val = |h, nope, vh, kvl, block, kvb: *const u8| {
            let m = Mla::value_dims(h, nope, vh, kvl, block);
            let kv = KvPtr::new(kvb, p as *const f32);
            mla_value_at(m, p as *const f32, kv, q as *mut f32)
        };
        assert!(val(0, 4, 4, 16, 16, p).is_err(), "mla_value h = 0");
        assert!(val(1, 0, 4, 16, 16, p).is_err(), "mla_value nope = 0");
        assert!(val(1, 4, 0, 16, 16, p).is_err(), "mla_value vh = 0");
        assert!(val(1, 4, 4, 0, 16, p).is_err(), "mla_value kvl = 0");
        assert!(val(1, 4, 4, 16, 0, p).is_err(), "mla_value block = 0");
        assert!(
            val(1, 4, 4, 16, 12, p).is_err(),
            "mla_value block not a power of two"
        );
        assert!(
            val(1, 4, 4, 18, 16, p).is_err(),
            "mla_value kvl not a multiple of 4"
        );
        assert!(
            val(1, 4, 4, 16, 16, p.add(1)).is_err(),
            "mla_value kvb not word-aligned"
        );
        assert!(
            val(1, 4, 4, 16, 16, p).is_ok(),
            "mla_value valid dims must be accepted"
        );

        // `nope` and `vh` are parameters here rather than hardcoded: fixing them at 4
        // would leave two of this launcher's five dimension arms untested, and an
        // untested guard is indistinguishable from its absence — which is the whole
        // rationale for this test.
        let abs = |h, qh, nope, vh, kvl, block, kvb: *const u8| {
            let m = Mla::new(h, qh, nope, vh, kvl, block);
            let kv = KvPtr::new(kvb, p as *const f32);
            mla_absorb_at(m, p as *const f32, kv, q as *mut f32)
        };
        assert!(abs(0, 4, 4, 4, 16, 16, p).is_err(), "mla_absorb h = 0");
        assert!(abs(1, 0, 4, 4, 16, 16, p).is_err(), "mla_absorb qh = 0");
        assert!(abs(1, 4, 0, 4, 16, 16, p).is_err(), "mla_absorb nope = 0");
        assert!(abs(1, 4, 4, 0, 16, 16, p).is_err(), "mla_absorb vh = 0");
        assert!(abs(1, 4, 4, 4, 0, 16, p).is_err(), "mla_absorb kvl = 0");
        assert!(abs(1, 4, 4, 4, 16, 0, p).is_err(), "mla_absorb block = 0");
        assert!(
            abs(1, 4, 4, 4, 16, 12, p).is_err(),
            "mla_absorb block not a power of two"
        );
        assert!(
            abs(1, 4, 4, 4, 16, 16, p.add(1)).is_err(),
            "mla_absorb kvb not word-aligned"
        );
        // RAGGED kvl IS ACCEPTED HERE, and that asymmetry with mla_value is the point:
        // this kernel masks the byte address rather than loading whole rows as words.
        assert!(
            abs(1, 4, 4, 4, 17, 16, p).is_ok(),
            "mla_absorb ragged kvl must be accepted"
        );
    }

    device_sync().expect("sync");
    v.check("mla_guards_reject_degenerate_arguments");
}

// ---------------------------------------------------------------------------
// moe.hip: the fused VQ-int3 expert batch.
//
// Oracles derived from moe.hip, common.hpp, quant.rs and math.rs — the SPECIFICATION —
// by someone working with the shader files WITHHELD, for the same reason as the MLA
// tranche: the person who wrote these shaders cannot also write their reference.
// ---------------------------------------------------------------------------

use rivoli::artifact::quant::{VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, vq_groups, vq_row_bytes};

/// Subvectors sharing one bf16 group scale — `VQ_SUBS_PER_GROUP` in common.hpp.
const VQ_SUBS: usize = VQ_GROUP / VQ_DIM;

/// One expert's six byte spans, in `ExpertDescVq` field order. Scales are the raw
/// little-endian bf16 bytes exactly as they sit on device, so the oracle decodes the
/// same bytes the kernel does.
#[derive(Clone, Default)]
struct ExpertBytes {
    gate_indices: Vec<u8>,
    gate_scales: Vec<u8>,
    up_indices: Vec<u8>,
    up_scales: Vec<u8>,
    down_indices: Vec<u8>,
    down_scales: Vec<u8>,
}

/// fp16 bit pattern -> f32. Exact for every finite value: 10 mantissa bits into 23, and
/// the 2^-24 subnormals are f32 normals. `math.rs` exports only the forward direction, and
/// this widening is `half`'s job rather than 12 lines of bit surgery — the same delegation
/// `math::f32_to_f16` makes, so oracle and engine widen through one implementation.
fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// The 12-bit index of subvector `t`, in `dot_vq_wave`'s own form.
///
/// TWO ADJACENT BYTES, and that is the whole hazard. `byte = (t*12)>>3` gives
/// `0,1,3,4,6,7,9,...`, so the pair straddles a 4-byte word exactly when
/// `t = 2 or 5 (mod 8)` — derived two ways, by enumeration and algebraically, agreeing.
fn vq_index(idxrow: &[u8], t: usize) -> usize {
    let bitpos = t * VQ_INDEX_BITS;
    let byte = bitpos >> 3;
    let shift = bitpos & 7;
    let raw = (idxrow[byte] as u32) | ((idxrow[byte + 1] as u32) << 8);
    ((raw >> shift) & 0xFFF) as usize
}

/// Write `idx` at subvector `t`, mirroring `quant.rs::set_idx`.
fn vq_set(idxrow: &mut [u8], t: usize, idx: usize) {
    let bitpos = t * VQ_INDEX_BITS;
    let (byte, shift) = (bitpos >> 3, bitpos & 7);
    let v = (idx as u32) << shift;
    idxrow[byte] |= (v & 0xff) as u8;
    idxrow[byte + 1] |= ((v >> 8) & 0xff) as u8;
}

/// `moe.hip::siluf` — a DIVIDE, not `math::silu`'s reciprocal-then-multiply. The two are
/// not bit-identical; see docs/investigations/vulkan-port.md.
fn siluf(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `common.hpp::dot_vq_wave`, wave-strided with the fixed ladder and hipcc's contraction:
/// term 1 a plain multiply, terms 2-4 fused, and the group scale fused.
fn oracle_dot_vq(idxrow: &[u8], scalerow: &[u8], cb: &[u16], x: &[f32], i_dim: usize) -> f32 {
    let nsub = i_dim / VQ_DIM;
    let mut lanes = [0.0f32; ORACLE_WAVE];
    for (lane, out) in lanes.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        let mut t = lane;
        while t < nsub {
            let idx = vq_index(idxrow, t);
            // A fixed array, not a Vec: this is the innermost loop of an oracle that runs
            // 32 lanes x 128 subvectors x 512 rows x 3 experts, and a heap allocation per
            // subvector made the test allocation-bound rather than arithmetic-bound.
            let c = [
                f16_to_f32(cb[idx * VQ_DIM]),
                f16_to_f32(cb[idx * VQ_DIM + 1]),
                f16_to_f32(cb[idx * VQ_DIM + 2]),
                f16_to_f32(cb[idx * VQ_DIM + 3]),
            ];
            let i0 = t * VQ_DIM;
            let mut dot = x[i0] * c[0];
            dot = x[i0 + 1].mul_add(c[1], dot);
            dot = x[i0 + 2].mul_add(c[2], dot);
            dot = x[i0 + 3].mul_add(c[3], dot);
            let g = t / VQ_SUBS;
            let s = rivoli::math::bf16_to_f32(u16::from_le_bytes([
                scalerow[g * 2],
                scalerow[g * 2 + 1],
            ]));
            acc = s.mul_add(dot, acc);
            t += ORACLE_WAVE;
        }
        *out = acc;
    }
    oracle_wave_sum(lanes)
}

fn moe_gateup_oracle(
    x: &[f32],
    descs: &[ExpertBytes],
    gate_cb: &[u16],
    up_cb: &[u16],
    g: MoeRange,
) -> Vec<f32> {
    let (hidden, inter) = (g.hidden, g.inter);
    let (rb, ng) = (vq_row_bytes(hidden), vq_groups(hidden));
    let mut h = vec![0.0f32; g.e_end() * inter];
    for e in g.e_start..g.e_end() {
        let d = &descs[e];
        for j in 0..inter {
            let g = oracle_dot_vq(
                &d.gate_indices[j * rb..(j + 1) * rb],
                &d.gate_scales[j * ng * 2..(j + 1) * ng * 2],
                gate_cb,
                x,
                hidden,
            );
            let u = oracle_dot_vq(
                &d.up_indices[j * rb..(j + 1) * rb],
                &d.up_scales[j * ng * 2..(j + 1) * ng * 2],
                up_cb,
                x,
                hidden,
            );
            h[e * inter + j] = siluf(g) * u;
        }
    }
    h
}

fn moe_down_oracle(
    descs: &[ExpertBytes],
    down_cb: &[u16],
    wexpert: &[f32],
    h: &[f32],
    g: MoeRange,
) -> Vec<f32> {
    let (hidden, inter) = (g.hidden, g.inter);
    let (rb, ng) = (vq_row_bytes(inter), vq_groups(inter));
    let mut partial = vec![0.0f32; g.e_end() * hidden];
    for e in g.e_start..g.e_end() {
        let d = &descs[e];
        let he = &h[e * inter..(e + 1) * inter];
        for o in 0..hidden {
            let dv = oracle_dot_vq(
                &d.down_indices[o * rb..(o + 1) * rb],
                &d.down_scales[o * ng * 2..(o + 1) * ng * 2],
                down_cb,
                he,
                inter,
            );
            partial[e * hidden + o] = wexpert[e] * dv;
        }
    }
    partial
}

/// Host twin of `common.glsl::moe_fixed` — the SCALE and the rounding mode are the whole
/// contract, so a drifted constant fails here rather than as a mystery 1e-4 in decode.
fn moe_fixed(v: f32) -> i64 {
    (v.clamp(-16384.0, 16384.0) * 17592186044416.0).round_ties_even() as i64
}

/// Host twin of `moe_acc_drain`: quantise each expert, sum EXACTLY, convert once.
fn moe_acc_drain_oracle(partial: &[f32], e_count: usize, hidden: usize) -> Vec<f32> {
    (0..hidden)
        .map(|o| {
            let t: i64 = (0..e_count)
                .map(|e| moe_fixed(partial[e * hidden + o]))
                .sum();
            (t as f64 / 17592186044416.0) as f32
        })
        .collect()
}

/// Build one expert's six spans. `plant` places a 12-bit index with EVERY BIT SET at the
/// straddling subvectors, which is what makes the word-boundary bug detectable.
fn expert_bytes(seed: u64, hidden: usize, inter: usize, plant: bool) -> ExpertBytes {
    let mut r = Lcg(seed);
    let mut mk = |rows: usize, i_dim: usize| -> (Vec<u8>, Vec<u8>) {
        let (rb, ng) = (vq_row_bytes(i_dim), vq_groups(i_dim));
        let nsub = i_dim / VQ_DIM;
        let mut idx = vec![0u8; rows * rb];
        for row in 0..rows {
            let span = &mut idx[row * rb..(row + 1) * rb];
            for t in 0..nsub {
                // 0xFFF at every straddling subvector (t = 2 or 5 mod 8). A naive
                // single-word load truncates the high nibble there, so an index whose
                // high bits are zero would NOT reveal the bug.
                let v = if plant && (t % 8 == 2 || t % 8 == 5) {
                    0xFFF
                } else {
                    (r.0 >> 20) as usize & 0xFFF
                };
                r.f();
                vq_set(span, t, v);
            }
        }
        let scales: Vec<u8> = (0..rows * ng)
            .flat_map(|_| f32_to_bf16((r.f() * 0.5).abs() + 0.05).to_le_bytes())
            .collect();
        (idx, scales)
    };
    let (gi, gs) = mk(inter, hidden);
    let (ui, us) = mk(inter, hidden);
    let (di, ds) = mk(hidden, inter);
    ExpertBytes {
        gate_indices: gi,
        gate_scales: gs,
        up_indices: ui,
        up_scales: us,
        down_indices: di,
        down_scales: ds,
    }
}

/// An fp16 codebook of `VQ_K` entries x VQ_DIM centroids, as raw bit patterns.
fn vq_codebook(seed: u64) -> Vec<u16> {
    let mut r = Lcg(seed);
    (0..4096 * VQ_DIM)
        .map(|_| rivoli::math::f32_to_f16(r.f()))
        .collect()
}

/// The whole VQ MoE path — gate/up, down, reduce — against the CPU oracles.
///
/// SHAPE CHOSEN ADVERSARIALLY, not for convenience. `hidden = 512` gives 128 subvectors
/// per gate/up row, so the straddling indices at `t = 2, 5 (mod 8)` occur 32 times per
/// row. `inter = 64` gives only 16 subvectors per DOWN row — fewer than `WAVE` — so half
/// the lanes contribute exactly zero and the halving ladder runs in its degenerate
/// regime, which no other shape here exercises.
#[test]
fn moe_vq_matches_the_host_oracles() {
    let v = Validation::new();
    let (hidden, inter, e_count) = (512usize, 64usize, 3usize);
    let mut r = Lcg(0x30E);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let wexpert: Vec<f32> = (0..e_count).map(|_| r.f().abs() + 0.1).collect();
    let descs: Vec<ExpertBytes> = (0..e_count)
        .map(|e| expert_bytes(0xE0 + e as u64, hidden, inter, true))
        .collect();
    let (gate_cb, up_cb, down_cb) = (vq_codebook(1), vq_codebook(2), vq_codebook(3));

    let g = MoeRange::new(hidden, inter, 0, e_count);
    let want_h = moe_gateup_oracle(&x, &descs, &gate_cb, &up_cb, g);
    let want_p = moe_down_oracle(&descs, &down_cb, &wexpert, &want_h, g);
    let want_o = moe_acc_drain_oracle(&want_p, e_count, hidden);

    // Device-side: the six spans per expert, then a descriptor array of their addresses.
    let bufs: Vec<[Buf; 6]> = descs
        .iter()
        .map(|d| {
            [
                dev(&d.gate_indices),
                dev(&d.gate_scales),
                dev(&d.up_indices),
                dev(&d.up_scales),
                dev(&d.down_indices),
                dev(&d.down_scales),
            ]
        })
        .collect();
    let desc_bytes: Vec<u8> = bufs
        .iter()
        .flat_map(|b| b.iter().flat_map(|s| (s.ptr() as u64).to_le_bytes()))
        .collect();
    let db = dev(&desc_bytes);
    let (gcb, ucb, dcb) = (
        dev(&u16b(&gate_cb)),
        dev(&u16b(&up_cb)),
        dev(&u16b(&down_cb)),
    );
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&wexpert)));
    let mut hb = dev(&vec![0u8; e_count * inter * 4]);
    // ONE accumulator row (u64), zeroed, plus a zeroed `x` for the drain to add into —
    // the drain is the residual add, so its output starts at 0 rather than poisoned. The
    // GUARD past `hidden` stays poisoned and must survive.
    let mut ab = dev(&vec![0u8; hidden * 8]);
    let mut ob = dev(&[vec![0u8; hidden * 4], vec![0xFFu8; GUARD]].concat());

    // SAFETY: every buffer is live and of the documented size; descs holds e_count
    // six-address descriptors whose targets outlive the sync.
    unsafe {
        launch_moe_expert_range(
            xb.ptr() as *const f32,
            hidden,
            inter,
            0,
            e_count,
            db.ptr() as *const rivoli::backend::vk::ExpertDesc,
            gcb.ptr() as *const u16,
            ucb.ptr() as *const u16,
            dcb.ptr() as *const u16,
            wb.ptr() as *const f32,
            hb.ptr_mut() as *mut f32,
            ab.ptr_mut() as *mut u64,
            1,
            std::ptr::null_mut(),
        )
    }
    .expect("expert range");
    drain(&mut ob, &mut ab, hidden, 1, 1.0, std::ptr::null_mut());
    device_sync().expect("sync");

    let got_h = f32v(&readb(&hb, e_count * inter * 4));
    let got_o = readb(&ob, hidden * 4 + GUARD);
    assert_untouched(&got_o, hidden * 4, "moe_acc_drain");
    // The accumulator was DRAINED, so it must read back all zeroes — the reset is what
    // lets the next layer skip a memset, and a drain that forgot it would double-count
    // silently from layer 1 onward.
    assert!(
        readb(&ab, hidden * 8).iter().all(|&b| b == 0),
        "moe_acc_drain left the accumulator dirty"
    );

    let mut shapes = Shapes::default();
    shapes.close(&want_h, &got_h, "moe_gateup_vq h512 i64 e3");
    shapes.close(
        &want_o,
        &f32v(&got_o[..hidden * 4]),
        "moe_acc_drain h512 e3",
    );
    shapes.assert_all_passed("moe vq");

    // THE FUSION PIN, WHICH THE TOLERANCE ABOVE CANNOT SEE.
    //
    // `vq.glsl` spells `mul, fma, fma, fma` plus a fused scale because hipcc's ISA does —
    // and a 1e-3 comparison is blind to contraction and association by three orders of
    // magnitude, so a wrong grouping in the four-term dot would ship green above. The pass
    // that can be pinned exactly is `moe_down_vq`: it is `exp`-FREE (one multiply after the
    // dot), so feeding the GPU's OWN `h` into the oracle removes silu — the only
    // unreproducible step — and what remains is the VQ dot and nothing else.
    //
    // The fixed-point sum pins exactly too, and for a stronger reason than the f32 reduce
    // did: integer addition has no rounding for a schedule to reorder, so this holds no
    // matter which order the experts' atomics landed in.
    let pin_p = moe_down_oracle(&descs, &down_cb, &wexpert, &got_h, g);
    let pin_o = moe_acc_drain_oracle(&pin_p, e_count, hidden);
    assert_bit_identical(
        &pin_o,
        &f32v(&got_o[..hidden * 4]),
        "moe_acc_drain (fed the GPU's own h)",
    );
    v.check("moe_vq_matches_the_host_oracles");
}

/// The VQ launchers' argument guards, including the two stricter than HIP.
#[test]
fn moe_guards_reject_degenerate_arguments() {
    let v = Validation::new();
    let (_s, p, q) = Scratch::new(4096);
    // SAFETY: every case is rejected before a pointer is dereferenced.
    unsafe {
        let mo = |hidden: usize, inter: usize, e_count: usize| {
            launch_moe_expert_range(
                p as *const f32,
                hidden,
                inter,
                0,
                e_count,
                p as *const rivoli::backend::vk::ExpertDesc,
                p as *const u16,
                p as *const u16,
                p as *const u16,
                p as *const f32,
                q as *mut f32,
                q as *mut u64,
                1,
                std::ptr::null_mut(),
            )
        };
        assert!(mo(0, 64, 1).is_err(), "moe hidden = 0");
        assert!(mo(64, 0, 1).is_err(), "moe inter = 0");
        assert!(mo(64, 64, 0).is_err(), "moe e_count = 0");
        // STRICTER THAN HIP: moe.hip's vq_rb/vq_ng truncate silently on a dimension that
        // is not a whole number of VQ groups, mis-sizing every row with no diagnostic.
        assert!(
            mo(96, 64, 1).is_err(),
            "moe hidden not a multiple of VQ_GROUP"
        );
        assert!(
            mo(64, 96, 1).is_err(),
            "moe inter not a multiple of VQ_GROUP"
        );
        // One dispatch with one dim zeroed, so the two cases differ by exactly the
        // argument under test and nothing else can be what rejected them.
        let acc_drain = |n: usize, rows: usize| {
            launch_moe_acc_drain(
                q as *mut f32,
                q as *mut u64,
                n,
                rows,
                1.0,
                std::ptr::null_mut(),
            )
        };
        assert!(acc_drain(0, 1).is_err(), "moe_acc_drain n = 0");
        assert!(acc_drain(64, 0).is_err(), "moe_acc_drain rows = 0");
    }
    v.check("moe_guards_reject_degenerate_arguments");
}

/// THE PLANTED INDICES MUST ACTUALLY BE ABLE TO CATCH THE BUG.
///
/// The word-boundary defect needs the device to demonstrate end to end — a naive shader, a
/// red oracle, a restore. But half of it is checkable on the CPU now, and it is the half
/// that is easy to get wrong: whether the test DATA distinguishes the correct unpack from
/// the naive one at all. A planted index whose high nibble happened to be zero would leave
/// the GPU test green under a broken shader and nobody would know.
///
/// So model the naive single-word read — load the `uint` containing `byte`, shift within
/// it, drop whatever lives in the next word — and assert it DISAGREES at every straddling
/// subvector and AGREES everywhere else. The second half matters as much: if the model
/// disagreed everywhere it would not be modelling this bug.
#[test]
fn planted_straddle_indices_would_catch_a_naive_unpack() {
    let (hidden, inter) = (512usize, 64usize);
    let d = expert_bytes(0xE0, hidden, inter, true);
    let rb = vq_row_bytes(hidden);
    let nsub = hidden / VQ_DIM;

    let naive = |row: &[u8], t: usize| -> usize {
        let bitpos = t * VQ_INDEX_BITS;
        let byte = bitpos >> 3;
        let w = byte & !3;
        let word = u32::from_le_bytes([row[w], row[w + 1], row[w + 2], row[w + 3]]);
        ((word >> (((byte - w) * 8) + (bitpos & 7))) & 0xFFF) as usize
    };

    let row = &d.gate_indices[..rb];
    let (mut straddling, mut caught) = (0usize, 0usize);
    for t in 0..nsub {
        let correct = vq_index(row, t);
        if ((t * VQ_INDEX_BITS) >> 3) % 4 == 3 {
            straddling += 1;
            assert_eq!(
                correct, 0xFFF,
                "t={t} should carry the planted all-ones index"
            );
            if naive(row, t) != correct {
                caught += 1;
            }
        } else {
            assert_eq!(
                naive(row, t),
                correct,
                "t={t} does not straddle; both must agree"
            );
        }
    }
    assert_eq!(
        straddling, 32,
        "expected 32 straddling subvectors at nsub=128"
    );
    assert_eq!(
        caught, straddling,
        "the planted data must distinguish the naive unpack at EVERY straddling subvector \
         — below {straddling} and the GPU test could go green against a broken shader"
    );
    println!("{caught}/{straddling} straddling subvectors distinguish a naive unpack");
}

// ---------------------------------------------------------------------------
// The phase-4 integration surface: fill_u32, flag_nonfinite, and the refusals.
//
// Oracles from kernels/fwd.hip::flag_nonfinite and kernels/vmm.hip::fill_u32.
// ---------------------------------------------------------------------------

/// Read one u32 back from a buffer, after a sync.
fn read_u32(b: &Buf) -> u32 {
    let raw = readb(b, 4);
    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

/// `vkCmdFillBuffer` must behave as `vmm.hip::fill_u32`: every 32-bit word in
/// `[dst, dst+bytes)` becomes `pat`, and NOTHING past it moves.
///
/// The pattern is the one `pin.rs` actually poisons with — a quiet NaN in f32 and in both
/// bf16 halves — because a fill that swapped byte order would be invisible under a
/// palindromic pattern and 0x7FC0_7FC0 is one. So the guard band carries a DIFFERENT,
/// non-palindromic word, and a partial fill is what the overrun check catches.
#[test]
fn fill_u32_writes_the_pattern_and_nothing_past_it() {
    let v = Validation::new();
    const PAT: u32 = 0x7FC0_7FC0;
    // 4 (one word), 1020/1028 either side of a 256-thread workgroup's 1024 bytes, and a
    // ragged multi-workgroup size. None is a multiple of the workgroup byte span.
    for bytes in [4usize, 1020, 1028, 5000] {
        let mut b = poison(bytes + GUARD);
        // SAFETY: `b` owns `bytes + GUARD` device bytes and `bytes` is a multiple of 4.
        unsafe { fill_u32(b.ptr_mut(), PAT, bytes).expect("fill") };
        let got = sync_readb(&b, bytes + GUARD);
        let label = format!("fill_u32 bytes={bytes}");
        assert_untouched(&got, bytes, &label);
        let want: Vec<u8> = PAT.to_le_bytes().repeat(bytes / 4);
        assert_bytes(&want, &got[..bytes], &label);
    }

    // The guards. A `bytes` that is not a multiple of 4 cannot be expressed as whole
    // words, and vkCmdFillBuffer would silently round it — so the launcher refuses,
    // matching `rivoli_fill_u32`'s rc 1001.
    let mut b = dev(&[0u8; 64]);
    // SAFETY: every call is rejected on its arguments before the pointer is used.
    unsafe {
        assert!(
            fill_u32(b.ptr_mut(), PAT, 0).is_err(),
            "0 bytes must be rejected"
        );
        assert!(
            fill_u32(b.ptr_mut(), PAT, 6).is_err(),
            "6 bytes must be rejected"
        );
        assert!(
            fill_u32(std::ptr::null_mut(), PAT, 4).is_err(),
            "an address in no live Buf must be rejected"
        );
    }
    v.check("fill_u32_writes_the_pattern_and_nothing_past_it");
}

/// `read_raw` resolves a bare device address back to its host mapping, INCLUDING at an
/// offset into the buffer — which is the only case its caller ever uses.
///
/// The `trace` expert-checksum probe hashes weights straight out of the pool slab, and the
/// engine addresses those as raw pointers inside descriptors: never a buffer base. So a
/// lookup that returned the mapping base and ignored the offset would satisfy an
/// offset-zero test and mis-hash every expert but the first. Hence the sub-range read, and
/// hence the guards: one byte past the end, and an address in no live buffer at all.
#[test]
fn read_raw_resolves_a_device_address_at_an_offset() {
    let v = Validation::new();
    let bytes: Vec<u8> = (0..1024u32)
        .map(|i| (i.wrapping_mul(31) & 0xff) as u8)
        .collect();
    let b = dev(&bytes);
    let mut out = Vec::new();

    // SAFETY: `b` is live and holds 1024 bytes; every range below is inside it, and no
    // kernel is writing (nothing was launched).
    unsafe {
        rivoli::backend::vk::read_raw(b.ptr(), bytes.len(), &mut out).expect("whole buffer");
        assert_bytes(&bytes, &out, "read_raw whole");

        let off = 517usize; // not word-aligned, so a word-granular lookup would show
        let len = 300usize;
        rivoli::backend::vk::read_raw(b.ptr().add(off), len, &mut out).expect("sub-range");
        assert_bytes(&bytes[off..off + len], &out, "read_raw at offset");
    }

    // SAFETY: each call is rejected on its arguments; nothing is dereferenced.
    unsafe {
        assert!(
            rivoli::backend::vk::read_raw(b.ptr(), 1025, &mut out).is_err(),
            "one byte past the end must be rejected"
        );
        assert!(
            rivoli::backend::vk::read_raw(b.ptr().add(1024), 1, &mut out).is_err(),
            "an address at the end must be rejected"
        );
        assert!(
            rivoli::backend::vk::read_raw(b.ptr(), 0, &mut out).is_err(),
            "a zero-length read must be rejected"
        );
        assert!(
            rivoli::backend::vk::read_raw(std::ptr::null(), 4, &mut out).is_err(),
            "an address in no live Buf must be rejected"
        );
    }
    v.check("read_raw_resolves_a_device_address_at_an_offset");
}

/// `flag_nonfinite` records `tag` iff some element is non-finite, and FIRST WRITER WINS.
///
/// The oracle is `kernels/fwd.hip::flag_nonfinite`: `atomicCAS(flag, 0, tag)` on any `x[i]`
/// failing `-inf < x[i] < inf`. Four properties, and the last is the one that makes the NaN
/// localizer's answer meaningful rather than merely non-zero:
///
/// 1. An all-finite buffer leaves the flag at 0 — the "clean run" reading.
/// 2. A NaN anywhere sets it. Positions are swept across the workgroup boundary because a
///    shader that only checked lane 0, or only the first workgroup, would pass a
///    first-element test and miss every real fault.
/// 3. +inf and -inf set it too. A shader written with `isnan` instead of the two-sided
///    compare passes on the NaN and silently ignores an overflow to infinity, which is the
///    more common numerical failure of the two.
/// 4. A SECOND call with a different tag over an already-flagged buffer does not overwrite
///    the first — so the tag names the EARLIEST non-finite layer, which is the whole
///    diagnostic. A plain store instead of the CAS would report the LAST layer, i.e. would
///    point at a consequence rather than a cause.
#[test]
fn flag_nonfinite_records_the_first_tag_only() {
    let v = Validation::new();
    let mut r = Lcg(0xFACE);
    // Sizes either side of one workgroup (256 threads), plus a ragged multi-workgroup one.
    for n in [1usize, 255, 257, 1000] {
        let finite: Vec<f32> = (0..n).map(|_| r.f()).collect();

        // 1. Clean.
        let xb = dev(&f32b(&finite));
        let mut flag = dev(&0u32.to_le_bytes());
        // SAFETY: `xb` is n live f32, `flag` one live u32, both until the sync.
        unsafe {
            launch_flag_nonfinite(xb.ptr() as *const f32, n, 0x11, flag.ptr_mut() as *mut u32)
        }
        .expect("launch clean");
        device_sync().expect("sync");
        assert_eq!(
            read_u32(&flag),
            0,
            "n={n}: an all-finite buffer must not flag"
        );

        // 2/3. One bad element at a time, at positions that straddle the workgroup edge.
        for (label, bad) in [
            ("nan", f32::NAN),
            ("+inf", f32::INFINITY),
            ("-inf", f32::NEG_INFINITY),
        ] {
            for &pos in &[0usize, n / 2, n - 1] {
                let mut x = finite.clone();
                x[pos] = bad;
                let xb = dev(&f32b(&x));
                let mut flag = dev(&0u32.to_le_bytes());
                // SAFETY: as above.
                unsafe {
                    launch_flag_nonfinite(
                        xb.ptr() as *const f32,
                        n,
                        0x2A,
                        flag.ptr_mut() as *mut u32,
                    )
                }
                .expect("launch bad");
                device_sync().expect("sync");
                assert_eq!(
                    read_u32(&flag),
                    0x2A,
                    "n={n} {label} at {pos}: must be flagged with the tag"
                );
            }
        }
    }

    // 4. First writer wins across two calls on one flag.
    let n = 512usize;
    let mut x: Vec<f32> = (0..n).map(|_| r.f()).collect();
    x[100] = f32::NAN;
    let xb = dev(&f32b(&x));
    let mut flag = dev(&0u32.to_le_bytes());
    // SAFETY: as above; the two launches are ordered by `enqueue`'s barrier.
    unsafe {
        launch_flag_nonfinite(xb.ptr() as *const f32, n, 0x07, flag.ptr_mut() as *mut u32)
            .expect("first tag");
        launch_flag_nonfinite(xb.ptr() as *const f32, n, 0x63, flag.ptr_mut() as *mut u32)
            .expect("second tag");
    }
    device_sync().expect("sync");
    assert_eq!(
        read_u32(&flag),
        0x07,
        "the FIRST tag must survive — a plain store would report the last layer to see the \
         NaN instead of the one that produced it"
    );

    // Guards. `n = 0` has nothing to scan; tag 0 is the "nothing flagged" sentinel and a
    // CAS against it writes nothing, so a caller passing it would get silence it could not
    // distinguish from a clean run.
    // SAFETY: both are rejected on their arguments before any pointer is used.
    unsafe {
        assert!(
            launch_flag_nonfinite(xb.ptr() as *const f32, 0, 1, flag.ptr_mut() as *mut u32)
                .is_err(),
            "n = 0 must be rejected"
        );
        assert!(
            launch_flag_nonfinite(xb.ptr() as *const f32, n, 0, flag.ptr_mut() as *mut u32)
                .is_err(),
            "tag 0 must be rejected"
        );
    }
    v.check("flag_nonfinite_records_the_first_tag_only");
}

/// The DEFERRED launchers must return `Err`, every one of them.
///
/// This is an oracle, not a formality. The alternative implementation — a no-op that
/// returns `Ok` — compiles, dispatches nothing, and produces numbers: `layernorm` leaves
/// the DSA indexer selecting rows on unnormalised keys, `moe_expert_range_i4` leaves the
/// partial slab holding the previous token's experts. Both are silently wrong and neither
/// would fail any other test in this file, because no other test calls them. So the
/// property under test is "refuses", and it is checked directly.
///
/// `Config::validate` is what a USER hits first (docs/investigations/vulkan-port.md: a Vulkan build rejects
/// `--attn dsa|misa` and `--mode int4|hybrid` at startup). These are the backstop for a
/// path that reaches a kernel without passing that gate.
///
/// Null pointers throughout, deliberately: a launcher that dereferenced one would fault
/// here rather than pass, which is the second thing this test pins down.
#[test]
fn deferred_launchers_refuse_rather_than_no_op() {
    let nf: *mut f32 = std::ptr::null_mut();
    let cf: *const f32 = std::ptr::null();
    // SAFETY: every launcher below is a deferred stub that returns Err before touching a
    // pointer — which is precisely what this test asserts. A regression to a real
    // implementation would fault on the nulls instead of passing quietly.
    let refusals: Vec<(&str, bool)> = unsafe {
        vec![
            (
                "moe_expert_range_i4",
                launch_moe_expert_range_i4(
                    cf,
                    64,
                    64,
                    0,
                    1,
                    std::ptr::null::<ExpertDesc>(),
                    cf,
                    nf,
                    std::ptr::null_mut::<u64>(),
                    1,
                    std::ptr::null_mut(),
                )
                .is_err(),
            ),
            (
                "layernorm",
                launch_layernorm(cf, cf, cf, 64, 1e-6, nf).is_err(),
            ),
            (
                "index_append",
                launch_index_append(cf, std::ptr::null_mut(), 0, 64).is_err(),
            ),
            (
                "index_score",
                launch_index_score(
                    cf,
                    cf,
                    std::ptr::null(),
                    std::ptr::null(),
                    8,
                    4,
                    4,
                    64,
                    1.0,
                    1.0,
                    nf,
                )
                .is_err(),
            ),
            (
                "index_topk",
                launch_index_topk(cf, 8, 4, std::ptr::null_mut()).is_err(),
            ),
            (
                "index_pool_push",
                launch_index_pool_push(cf, nf, 0, 64).is_err(),
            ),
            (
                "index_head_route",
                launch_index_head_route(cf, cf, cf, 2, 4, 64, nf).is_err(),
            ),
            ("vaxpy", launch_vaxpy(nf, cf, 2.0, 64).is_err()),
        ]
    };
    let quiet: Vec<&str> = refusals
        .iter()
        .filter(|(_, e)| !e)
        .map(|(n, _)| *n)
        .collect();
    assert!(
        quiet.is_empty(),
        "these deferred launchers returned Ok instead of refusing: {quiet:?}. A no-op that \
         reports success is the failure mode this test exists to prevent — it produces \
         plausible numbers with no diagnostic."
    );
    println!("{} deferred launchers all refuse", refusals.len());
}

/// **INV-4 (Vulkan half): a wait may be enqueued BEFORE its producer exists, and still
/// waits.** The rocm half lives in `src/backend/gpustream.rs`; both are required, because the two
/// backends reach the property by different mechanisms and only a per-backend test can show
/// each one actually has it.
///
/// HIP enqueues a wait op into the stream (`hipStreamWaitValue64`). Vulkan attaches the wait
/// to a *submit* and relies on timeline semaphores permitting wait-before-signal. A Vulkan
/// implementation that quietly resolved the wait at record time — or dropped it because
/// nothing had signalled yet — would pass every kernel test in this file and fail only as
/// silent corruption under load, which is exactly how the `hit`-mask bug behaved.
#[cfg(feature = "vulkan")]
#[test]
fn inv_4_wait_enqueued_before_signal_still_waits() -> anyhow::Result<()> {
    use rivoli::backend::{Stream, Timeline, stream_signal};

    let t = Timeline::new()?;
    assert_eq!(t.completed(), 0, "a fresh timeline starts at 0");

    let compute = Stream::compute()?;
    let fetch = Stream::fetch()?;

    // Order is the whole test: the WAIT is registered first, against a value nothing has
    // produced. If it were snapshot-at-record (HIP events' failure mode) this would sail
    // through and the assertion below would read 0.
    t.wait(compute.raw(), 1)?;
    t.signal(fetch.raw(), 1)?;

    block_on(stream_signal(compute.raw())?);
    assert!(
        t.completed() >= 1,
        "the waited-on value must have been reached, not skipped"
    );
    Ok(())
}

/// **INV-6: a wait can always be released, so a dead producer cannot hang the device.**
/// `gpustream.rs::inv_6_…`'s Vulkan half — separate because the two backends reach it by
/// different mechanisms: HIP does a monotone CAS into signal memory, Vulkan calls
/// `vkSignalSemaphore`, which is the host-side signal operation proper.
///
/// The property is the same and it is the other half of INV-4. A wait enqueued before its
/// producer is the design; a producer that dies owing that value must still be answerable,
/// or the fetch error surfaces as a hang. It did — see the reaper's `release`.
#[cfg(feature = "vulkan")]
#[test]
fn inv_6_a_host_release_retires_an_enqueued_wait() -> anyhow::Result<()> {
    use rivoli::backend::{Stream, Timeline, stream_signal};

    let t = Timeline::new()?;
    let compute = Stream::compute()?;
    // Against a value NOTHING will ever signal on any queue: the dead-producer case, not a
    // race with a slow one.
    t.wait(compute.raw(), 7)?;
    t.release(7);

    block_on(stream_signal(compute.raw())?);
    assert!(
        t.completed() >= 7,
        "the host release must be what retired the wait"
    );
    // Monotone: releasing backwards would un-free a slot another consumer is gated on.
    t.release(3);
    assert_eq!(
        t.completed(),
        7,
        "release must never move a timeline backwards"
    );
    Ok(())
}
