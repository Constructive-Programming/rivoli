//! The MoE expert-range kernels vs their CPU oracles on SYNTHETIC quantized fixtures:
//! `moe_expert_range` (VQ codebooks), `moe_expert_range_i4` (group scales), and the two
//! fixed-point accumulator drains. Everything here draws its own weights from `Lcg` and runs
//! wherever there is a GPU; the same kernels on the shipped `.i4` artifact are
//! `kernel_moe_artifact.rs`, which skips loudly without it.
//!
//! **Split out of `kernel.rs` on 2026-08-15 — by COHESION, not by size alone.** That file had
//! grown to 2263 lines and 79 functions covering nine unrelated kernel families, and CodeScene
//! scored it 8.03 on exactly those two rules ("Low Cohesion", and 79 functions "at risk of
//! evolving into a Brain Class"). The expert-range oracles are the group that comes away
//! clean: they share `ExpertDesc` construction, the `moe_reference` fusion, the fixed-point
//! accumulator and its drain, and share NOTHING with the GEMV, MLA, attend, rope, swiglu or
//! `index_topk` suites beyond the assert helpers in `common`.
//!
//! Everything below travelled VERBATIM with its comments — in this repo a comment carries the
//! measurement that justified the choice, so a re-worded one loses evidence. What changed is
//! listed where it happens:
//!
//! * `batched_moe_rows_are_bit_identical_to_single_rows` was an arm of `kernel.rs`'s umbrella
//!   test and is now its own `#[test]` with its own seed;
//! * `moe_vq_matches_reference` reads its destination back ONCE (it read it twice, through a
//!   helper that stayed behind);
//! * `assert_rows` and `assert_guard` moved to `common` because both halves now need them, and
//!   so did the dispatch chain `i4_launch_drain` drags in — see `common`'s own note;
//! * the positional `new` constructors are gone in favour of field-named struct literals, on
//!   `common::Mla`'s argument about six bare `usize` in a row.
//!
//! `tests/kernel_coverage.rs` scans every `.rs` under `crates/engine/tests`, so the census
//! followed this move with no edit of its own: the `launch_moe_*` names are still counted,
//! they are just counted here.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_artifact::quant::{
    I4_GROUP, RowScaledW, VQ_DIM, VQ_K, VqW, matvec_i4, matvec_vq, quant_i4, quant_vq,
    vq_expert_layout,
};
use rivoli_backend::gpustream::HipStream;
use rivoli_backend::hip::{
    ExpertDesc, device_sync, launch_moe_acc_drain_to, launch_moe_expert_range,
};
use rivoli_core::num::silu;

mod common;
use common::moe::{
    Dims, Drain, MoeCtx, MoeIo, desc_buf, drain, expert_range_i4, i4_launch_drain, moe_bufs,
};
use common::{
    DeviceBuf, Lcg, MoeRange, assert_bits, assert_close, assert_guard, assert_rows, dev, err_tol,
    f16b, f32b, f32v, u16b,
};

/// `moe_acc_drain_to` — the latent-width sibling, `n` columns over `rows` accumulator rows.
/// `Result`-returning for the same reason `kernel.rs`'s `gemv_fp8` is: its guard arm must reach
/// the assertion rather than panic.
fn drain_to(d: Drain<'_>, n: usize, rows: usize, stream: &HipStream) -> anyhow::Result<()> {
    // SAFETY: `out` is `n` f32 and `acc` `rows·n` u64 in every caller; the stream is live for
    // the call, and a rejected dimension never reaches a dereference.
    unsafe {
        launch_moe_acc_drain_to(
            d.out.ptr_mut() as *mut f32,
            d.acc.ptr_mut() as *mut u64,
            n,
            rows,
            stream.raw(),
        )
    }
}

/// Both drains RESET what they consume, and that is load-bearing rather than tidy: it is why
/// steady-state decode needs no memset before each layer's experts. A drain that forgot it would
/// pass every single-layer check here and then double-count from layer 1 onward — silently, since
/// the result stays finite and merely wrong.
///
/// One function rather than the three copies of `copy_out().iter().all(zero)` this file grew — the
/// single-row drain, the batched one, and the latent-width sibling all make the same claim about
/// the same buffer, and `what` is the only thing that ever differed between them.
fn assert_acc_drained(acc: &DeviceBuf, what: &str) {
    assert!(
        acc.copy_out().expect("acc").iter().all(|&b| b == 0),
        "{what}"
    );
}

/// Upload `b`, park it in `bufs`, and hand back its device address.
///
/// An `ExpertDesc` is a struct of raw pointers and owns nothing, so something has to keep
/// the six spans alive for the length of the dispatch; `bufs` is that something.
fn push(b: Vec<u8>, bufs: &mut Vec<DeviceBuf>) -> *const u8 {
    bufs.push(dev(&b));
    bufs.last().expect("just pushed").ptr()
}

/// `[(gate o,i), (up o,i), (down o,i)]` for one int4 expert, over a [`Dims`].
///
/// It spelled the three pairs out until 2026-08-15, in ONE place because `build.rs`'s jscpd
/// gate rejected the copies inlined at its call sites — and noted beside itself that
/// "`vq_expert_layout` says the same thing for the real-data tests". It does, exactly: that is
/// the PRODUCTION layout function, `quant.rs` builds every `*_expert_bytes` on it, and the
/// artifact tests were already calling it. Two spellings of one layout is one of them able to
/// drift, so this is now the `Dims`-typed way to ask the production one.
fn i4_expert_dims(d: Dims) -> [(usize, usize); 3] {
    vq_expert_layout(d.hidden, d.inter)
}

/// The caller's matvec for projection `p` of expert `ex`: `mv(out, in, ex, p, o_dim, i_dim)`.
type Matvec<'a> = &'a dyn Fn(&mut [f32], &[f32], usize, usize, usize, usize);

/// `e` experts × 3 projections of (packed weights, scales) — what [`encode_experts`] returns
/// and what every MoE oracle and descriptor block in this file reads.
type Enc<S> = Vec<Vec<(Vec<u8>, Vec<S>)>>;

/// `Σ_e w[e]·down(silu(gate·x) ⊙ up·x)` — the MoE reference both format tests check against.
///
/// `mv` (0 gate, 1 up, 2 down) is the only step that differs between int4 and VQ; the fusion
/// around it is the same arithmetic in the same order, and a second copy of it is a second
/// place for the accumulation order to drift from the kernel's.
fn moe_reference(x: &[f32], w: &[f32], d: Dims, mv: Matvec<'_>) -> Vec<f32> {
    let (hidden, inter) = (d.hidden, d.inter);
    let mut want = vec![0.0f32; hidden];
    for (ex, we) in w.iter().enumerate() {
        let mut g = vec![0.0f32; inter];
        let mut u = vec![0.0f32; inter];
        mv(&mut g, x, ex, 0, inter, hidden);
        mv(&mut u, x, ex, 1, inter, hidden);
        let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
        let mut down = vec![0.0f32; hidden];
        mv(&mut down, &h, ex, 2, hidden, inter);
        for (o, d) in down.iter().enumerate() {
            // THE FIXTURE'S PRECONDITION, asserted where it can name the expert. The GPU
            // path runs every per-expert contribution through `moe_fixed`, which SATURATES
            // at +/-MOE_ACC_MAX = 2^(58 - MOE_ACC_SHIFT) = 2^14 (common.hpp; not exported to
            // Rust, so derived here) — this reference does not, so a fixture whose
            // magnitudes exceed the clamp makes every comparison against it garbage in a way
            // that reads as a kernel bug. MEASURED 2026-08-09, the hard way: the first
            // multi-trip fixture drew weights up to +/-4 at 1280/1024, its reference peaked
            // at 1.055e5, the GPU clamped at 1.6384e4, and the device test failed by 845x
            // against a STOCK kernel that three other tests were passing at 1e-6 — err
            // equalled max - 2^14 to three figures. The hazard was already written down in
            // dot_bench.rs's `run_glm_i4` ("scales small enough that partials stay far
            // under moe_fixed's +/-2^14 saturation") and the fixture ignored it.
            let contrib = we * d;
            assert!(
                contrib.abs() < 16384.0,
                "fixture drives moe_fixed into saturation: |{contrib:.3e}| >= 2^14 at expert \
                 {ex} output {o} — the GPU clamps here and this reference does not, so the \
                 comparison would fail against a CORRECT kernel. Shrink the fixture's \
                 magnitudes; do not widen any tolerance."
            );
            want[o] += contrib;
        }
    }
    want
}

/// `e` experts' worth of per-projection quantised weights, `one(r, p, o_dim, i_dim)` doing
/// each projection.
///
/// The DRAW ORDER is what this exists to hold fixed: expert-major, then gate/up/down, every
/// weight matrix drawn from `r` before the next one starts. The two format tests differ only
/// in what `one` does with those weights, and a seed has to mean the same data in both.
fn encode_experts<S>(r: &mut Lcg, e: usize, dims: &[(usize, usize)], one: EncOne<'_, S>) -> Enc<S> {
    (0..e)
        .map(|_| {
            dims.iter()
                .enumerate()
                .map(|(p, &(o_dim, i_dim))| one(r, p, o_dim, i_dim))
                .collect()
        })
        .collect()
}

/// One projection's encoder: `one(rng, p, o_dim, i_dim)`, drawing its own weights.
type EncOne<'a, S> = &'a dyn Fn(&mut Lcg, usize, usize, usize) -> (Vec<u8>, Vec<S>);

/// `e` `ExpertDesc`s over an encoded set, each of the six spans uploaded through [`push`].
///
/// `sb` is the ONE difference between the two formats' descriptor blocks: VQ scales upload as
/// u16 and int4's as f32, and both are read through a `*const u16` whose only meaningful part
/// is the ADDRESS. Spelled per format, that is two ten-line blocks a descriptor layout change
/// has to be fixed in twice — and the second copy is what `build.rs`'s jscpd gate rejects.
fn expert_descs<S>(
    enc: &Enc<S>,
    bufs: &mut Vec<DeviceBuf>,
    sb: fn(&[S]) -> Vec<u8>,
) -> Vec<ExpertDesc> {
    enc.iter()
        .map(|p| ExpertDesc {
            gate_indices: push(p[0].0.clone(), bufs),
            gate_scales: push(sb(&p[0].1), bufs) as *const u16,
            up_indices: push(p[1].0.clone(), bufs),
            up_scales: push(sb(&p[1].1), bufs) as *const u16,
            down_indices: push(p[2].0.clone(), bufs),
            down_scales: push(sb(&p[2].1), bufs) as *const u16,
        })
        .collect()
}

/// int4 weights whose group `g` is `4^g` larger, so the stored group scales genuinely DIFFER
/// between groups. Uniform-magnitude weights would give near-equal group scales, and a kernel
/// reading the wrong group would still pass.
fn i4_group_varying(r: &mut Lcg, o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    let wv: Vec<f32> = (0..o_dim * i_dim)
        .map(|n| r.f() * 4f32.powi(((n % i_dim) / I4_GROUP) as i32))
        .collect();
    quant_i4(&wv, o_dim, i_dim)
}

/// The device state every VQ MoE arm holds fixed: the uploaded descriptor array, the three
/// per-projection codebooks, the stream, and the expert count.
///
/// The three arms below run the same two kernels and differ only in the token rows, the gate
/// weights, the destination buffers and `nrow` — so the rest is captured once rather than
/// re-spelled, and the dispatch reads as the argument LIST it is rather than fourteen lines
/// of `.ptr() as`.
struct VqCtx<'a> {
    descs: &'a DeviceBuf,
    cbs: [&'a DeviceBuf; 3],
    stream: &'a HipStream,
    e: usize,
}

impl<'a> VqCtx<'a> {
    fn new(descs: &'a DeviceBuf, cbs: [&'a DeviceBuf; 3], stream: &'a HipStream, e: usize) -> Self {
        Self {
            descs,
            cbs,
            stream,
            e,
        }
    }
}

/// The production path: each expert accumulates into the shared fixed-point row via a
/// single-expert range on a compute stream (exercising e_start indexing) — same as the async
/// expert stream, minus the load overlap. The caller drains.
fn vq_range(io: MoeIo<'_>, cx: &VqCtx<'_>, d: Dims, nrow: usize) {
    let (x, w, h, acc) = io.ptrs();
    let dp = cx.descs.ptr() as *const ExpertDesc;
    let cb = |b: &DeviceBuf| b.ptr() as *const u16;
    let (c0, c1, c2) = (cb(cx.cbs[0]), cb(cx.cbs[1]), cb(cx.cbs[2]));
    let (hi, it, st) = (d.hidden, d.inter, cx.stream.raw());
    for k in 0..cx.e {
        // SAFETY: every buffer is device-resident and sized for `nrow` rows of the
        // layout above; the stream is live.
        unsafe { launch_moe_expert_range(x, hi, it, k, 1, dp, c0, c1, c2, w, h, acc, nrow, st) }
            .expect("launch moe_expert_range");
    }
}

/// The VQ MoE fixture, device-free: three per-projection codebooks, the activation, the gate
/// weights, the quantized experts, and the `matvec_vq`+silu reference over the same bytes.
/// `r` rides along because the batched arm draws its second row from the same stream.
struct VqCase {
    cbs: [Vec<f32>; 3],
    x: Vec<f32>,
    w: Vec<f32>,
    enc: Enc<u16>,
    want: Vec<f32>,
    r: Lcg,
}

fn vq_case(d: Dims, e: usize) -> VqCase {
    let mut r = Lcg(0x33);
    // 3 codebooks
    let cbs: [Vec<f32>; 3] = std::array::from_fn(|_| (0..VQ_K * VQ_DIM).map(|_| r.f()).collect());
    let x: Vec<f32> = (0..d.hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..e).map(|_| r.f()).collect();

    // per expert per projection: quant to (indices, scales) against the right codebook
    let dims = vq_expert_layout(d.hidden, d.inter); // [(gate o,i),(up o,i),(down o,i)]
    let enc = encode_experts(&mut r, e, &dims, &|r, p, o_dim, i_dim| {
        let wv: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
        quant_vq(&wv, o_dim, i_dim, &cbs[p])
    });

    let want = moe_reference(&x, &w, d, &|out, inp, ex, p, o_dim, i_dim| {
        let vw = VqW::new(&enc[ex][p].0, &enc[ex][p].1, &cbs[p]);
        matvec_vq(out, inp, vw, [o_dim, i_dim]);
    });
    VqCase {
        cbs,
        x,
        w,
        enc,
        want,
        r,
    }
}

/// Fused VQ MoE (moe_gateup_vq → moe_down_vq → moe_acc_drain), 3 per-projection
/// codebooks, vs a matvec_vq+silu reference on the same quantized bytes.
#[test]
fn moe_vq_matches_reference() {
    let (d, e) = (Dims::new(128, 64), 3usize); // multi-group hidden, 1-group inter
    let mut c = vq_case(d, e);
    // device: hold bufs alive; descriptors point into them
    let mut bufs: Vec<DeviceBuf> = Vec::new();
    let descb = desc_buf(&expert_descs(&c.enc, &mut bufs, u16b));
    let (xb, wb) = (dev(&f32b(&c.x)), dev(&f32b(&c.w)));
    let g: [DeviceBuf; 3] = std::array::from_fn(|i| dev(&f16b(&c.cbs[i])));
    let stream = HipStream::new().expect("stream");
    let cx = VqCtx::new(&descb, [&g[0], &g[1], &g[2]], &stream, e);

    let (mut hbuf, mut pbuf, mut obuf) = moe_bufs(e, 1, d);
    vq_range(MoeIo::new(&xb, &wb, &mut hbuf, &mut pbuf), &cx, d, 1);
    drain(Drain::new(&mut obuf, &mut pbuf), 0, d.hidden, &stream);
    device_sync().expect("sync");
    // ONE readback, bound and then asserted. It was two — `kernel.rs`'s `assert_out` did its
    // own `copy_out`, and the batched arm below then read the same buffer a second time. That
    // helper stayed with the suites that need it against a DeviceBuf destination; here the
    // bytes are wanted twice, so reading them once is both shorter and one fewer chance for
    // the two reads to be of different buffers.
    let row0 = f32v(&obuf.copy_out().expect("out"));
    assert_close(&c.want, &row0, "moe_vq");
    assert_acc_drained(&pbuf, "moe_acc_drain left the accumulator dirty");
    // Row 0 of the batched arm reran these inputs exactly, so it must reproduce these bits.
    vq_batched_matches_singles(&cx, d, &mut c, &row0);
}

/// nrow=2: the batched verify pass, against the single-row path it must reproduce.
///
/// BIT-identical, not `assert_close`. The batched kernel is not an approximation of two
/// passes, it is the same arithmetic with the weight decode hoisted — same products,
/// same order, same `wave_sum`. Anything looser would accept a real reassociation bug,
/// and reassociation here is exactly what the fixed-point accumulator exists to make
/// impossible to hide.
///
/// Row 1 gets a DIFFERENT x and DIFFERENT weights, one of them zero. Two rows with the
/// same input would pass even if the kernel ignored the row index entirely; the zero
/// weight covers the "this row did not route here" skip, which is the case the union
/// batching produces on every real layer.
fn vq_batched_matches_singles(cx: &VqCtx<'_>, d: Dims, c: &mut VqCase, row0: &[f32]) {
    let (hidden, e) = (d.hidden, cx.e);
    let x2: Vec<f32> = (0..hidden).map(|_| c.r.f()).collect();
    let xr: Vec<f32> = c.x.iter().chain(&x2).copied().collect(); // x[t·hidden + i]
    let w2: Vec<f32> = (0..e).map(|k| if k == 1 { 0.0 } else { c.r.f() }).collect();
    // wexpert[e·2 + t] — token row fastest, matching the kernel's indexing.
    let wr: Vec<f32> = (0..e).flat_map(|k| [c.w[k], w2[k]]).collect();

    let (xrb, wrb) = (dev(&f32b(&xr)), dev(&f32b(&wr)));
    let (mut hbuf2, mut pbuf2, mut obuf2) = moe_bufs(e, 2, d);
    vq_range(MoeIo::new(&xrb, &wrb, &mut hbuf2, &mut pbuf2), cx, d, 2);
    for t in 0..2 {
        drain(Drain::new(&mut obuf2, &mut pbuf2), t, hidden, cx.stream);
    }
    device_sync().expect("sync");
    let got2 = f32v(&obuf2.copy_out().expect("out2"));

    // Row 1 against a fresh single-row run of the same inputs.
    let (x2b, w2b) = (dev(&f32b(&x2)), dev(&f32b(&w2)));
    let (mut hbuf1, mut pbuf1, mut obuf1) = moe_bufs(e, 1, d);
    vq_range(MoeIo::new(&x2b, &w2b, &mut hbuf1, &mut pbuf1), cx, d, 1);
    drain(Drain::new(&mut obuf1, &mut pbuf1), 0, hidden, cx.stream);
    device_sync().expect("sync");
    let row1 = f32v(&obuf1.copy_out().expect("out1"));
    assert_rows(&got2, &[row0.to_vec(), row1], hidden, "moe_vq nrow=2");
    assert_acc_drained(&pbuf2, "moe_acc_drain left a batched accumulator row dirty");
}

/// int4 MoE (moe_gateup_i4 → moe_down_i4 → moe_acc_drain), GROUP scales, vs a
/// matvec_i4+silu reference on the same quantized bytes. hidden ≥ 256 exercises
/// dot_i4_wave's dword fast path; inter < 256 its scalar tail.
///
/// `hidden = 2·I4_GROUP` is deliberate: one dword-fast-path step covers WAVE·8 = 256
/// columns, so lanes 0..15 and 16..31 sit in DIFFERENT groups within the same step —
/// a kernel that hoisted one scale per step, or per row, disagrees here. `inter` is
/// one whole group, so the scalar tail's group indexing is exercised too.
#[test]
fn moe_i4_matches_reference() {
    let mut r = Lcg(0x14);
    let (d, e) = (Dims::new(2 * I4_GROUP, I4_GROUP), 3usize);
    let x: Vec<f32> = (0..d.hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..e).map(|_| r.f()).collect();
    let dims = i4_expert_dims(d); // gate, up, down (o,i)
    let enc = encode_experts(&mut r, e, &dims, &|r, _p, o_dim, i_dim| {
        i4_group_varying(r, o_dim, i_dim)
    });

    let c = I4Case { enc, x, w, d };
    assert_close(&i4_reference(&c), &gpu_i4_moe(&c), "moe_i4");
}

/// The int4 MoE launch path shared by the synthetic-fixture tests: one `ExpertDesc` per
/// expert built from `enc`, `moe_gateup_i4` → `moe_down_i4` per expert, drain, read back.
///
/// ONE copy, for the reason `gpu_i4_expert` below gives about its own pair — a descriptor
/// layout change must not be fixable in one test and left stale in the other — and because
/// `build.rs`'s jscpd gate REJECTED the second copy the moment the multi-trip test was
/// written. That is the gate working, and it is why this is a function rather than a
/// paragraph in two tests.
///
/// Launches per expert rather than one `[0, e)` range: bit-identical by `moe_expert_range`'s
/// own argument (`e = e_start + row / inter`, every row independent), and it is what this path
/// has always done.
fn gpu_i4_moe(c: &I4Case) -> Vec<f32> {
    let mut bufs: Vec<DeviceBuf> = Vec::new();
    // One ExpertDesc for both formats: the int4 kernel reads the six pointers as
    // packed weights + f32 row-scale (the scale ptr is typed *const u16 but only its
    // ADDRESS matters — cast the f32 buffer ptr through it).
    let descs = expert_descs(&c.enc, &mut bufs, f32b);
    i4_launch_drain(&descs, &c.x, &c.w, c.d)
}

/// `moe_reference` over int4-encoded experts — the CPU oracle the synthetic tests compare
/// against, and which the defect-injection test below evaluates TWICE (clean bytes, then
/// defective ones). Three call sites, so it is a function for the same jscpd reason.
fn i4_reference(c: &I4Case) -> Vec<f32> {
    let enc = &c.enc;
    moe_reference(&c.x, &c.w, c.d, &|o, i, ex, p, od, id| {
        matvec_i4(
            o,
            i,
            RowScaledW::new(&enc[ex][p].0, &enc[ex][p].1),
            [od, id],
        );
    })
}

/// One int4 MoE fixture: quantized experts plus the activation, routing weights and dims they
/// were built for. A struct rather than five positional arguments because the two helpers above
/// otherwise carry an identical six-line parameter list, which `build.rs`'s jscpd gate rejects
/// — the same fix, for the same reason, that `oracle_cat`'s `CompCase` made in `tests/`.
struct I4Case {
    enc: Enc<f32>,
    x: Vec<f32>,
    w: Vec<f32>,
    d: Dims,
}

/// The multi-trip fixture the two tests below share: `hidden = 1280` (5 dword trips) and
/// `inter = 1024` (4), one expert, with group scales that genuinely differ between adjacent
/// groups.
///
/// **The dims are the whole point and they are not the obvious ones.** `dot_i4_wave_r`'s dword
/// loop advances `WAVE * 8 = 256` columns per trip, so trips = `dim / 256`, and an unroll of
/// depth D executes a main body D trips wide plus a REMAINDER loop of `trips % D`:
///
/// | dim | trips | rem @ depth 2 | rem @ depth 4 |
/// |---|---|---|---|
/// | hidden 6144 (engine) | 24 | 0 | 0 |
/// | inter 2048 (engine) | 8 | 0 | 0 |
/// | hidden 1280 | 5 | 1 | 1 |
/// | inter 1024 | 4 | **0** | **0** |
///
/// So **at every engine dimension the remainder is never entered** — a test at real dims would
/// be vacuous on the epilogue while looking thorough, which is exactly how the suite ended up
/// unable to see M11's fp4 pragma.
///
/// **The two dims cover the two DIFFERENT cases, and that is deliberate.** 5 trips gives an
/// unrolled body plus a remainder at depth 2 and at depth 4 — the path nothing else reaches.
/// 4 trips divides cleanly at both depths — the *production* geometry, since every engine dim
/// is an exact multiple. Making both dims leave a remainder was tried and **reverted**: it
/// tests the epilogue twice and stops testing the clean case at all, which is the one the
/// engine actually runs. `v4_kernel.rs`'s fp4 twin picked 1280/1024 for exactly this reason
/// and says so — "5 trips = unrolled body + remainder at unroll 2 AND at unroll 4; 4 = clean
/// groups at both". This is the same pair, for the same argument.
///
/// **NOTHING machine-checks the trip counts**, here or there: the int4 launcher guards
/// `hidden`/`inter` against `I4_GROUP` (128), not against `WAVE * 8` (256), so 1152 would
/// launch fine at a different count. A changed `WAVE` breaks the counts outright. Re-derive
/// from the table above rather than trusting the green.
///
/// Both dims are whole multiples of 256, so the scalar tail below the dword loop is NOT
/// entered; that path is covered by `moe_i4_matches_reference`'s `inter = I4_GROUP`.
fn i4_multi_trip_fixture() -> I4Case {
    let d = Dims::new(1280, 1024);
    let mut r = Lcg(0x14_7213);
    let x: Vec<f32> = (0..d.hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = vec![1.0];
    let enc = encode_experts(&mut r, 1, &i4_expert_dims(d), &|r, _p, o_dim, i_dim| {
        // Adjacent groups differ by 2x or 4x, so a kernel reading the wrong group scale
        // disagrees. `2^(g % 3)` rather than `moe_i4_matches_reference`'s `4^g`: at
        // `i_dim = 1280` there are 10 groups and `4^9` is 2.6e5, which would put the whole
        // comparison at the mercy of one group's magnitude instead of testing indexing.
        //
        // The 0.02 is NOT tuning — it is what makes the comparison against the unclamped CPU
        // reference valid at all, and it was measured in, not guessed (2026-08-09). At the
        // first draft's +/-4 weights, sigma(gate dot) ~ 31 over 1280 columns, h ~ silu(g)*u
        // reached O(1e4), and the down pass peaked at 1.055e5 — 6.4x past `moe_fixed`'s 2^14
        // saturation — so the DEVICE test failed by 845x against a stock kernel while the
        // host-only red-target test, which never crosses the GPU, passed and certified
        // nothing about it. At 0.02 the reference peaks O(1) with four orders of headroom,
        // and `moe_reference`'s clamp assert now makes this precondition loud instead of a
        // comment. Quantization noise scales down with the weights, so the tolerance's
        // relative form keeps the same discrimination.
        let wv: Vec<f32> = (0..o_dim * i_dim)
            .map(|n| r.f() * 0.02 * 2f32.powi((((n % i_dim) / I4_GROUP) % 3) as i32))
            .collect();
        quant_i4(&wv, o_dim, i_dim)
    });
    I4Case { enc, x, w, d }
}

/// Zero the DECODED weight of every column the `n7`-past-the-first-trip defect would drop:
/// nibble 7 of each dword, i.e. `i % 8 == 7`, on every trip after the first (`i >= WAVE * 8`).
///
/// Setting the stored nibble to 8 is exactly equivalent to M11's injection — `nib()` returns
/// `nibble - 8`, so a stored 8 decodes to 0.0 whatever the group scale is.
fn i4_drop_n7_after_first_trip(packed: &mut [u8], o_dim: usize, i_dim: usize) {
    let rb = i_dim.div_ceil(2);
    for o in 0..o_dim {
        for i in (WAVE_COLS..i_dim).filter(|i| i % 8 == 7) {
            let b = &mut packed[o * rb + i / 2];
            *b = if i % 2 == 0 {
                (*b & 0xF0) | 8
            } else {
                (*b & 0x0F) | (8 << 4)
            };
        }
    }
}

/// `WAVE * 8` — one dword-loop trip in columns. A kernel constant this file must not
/// re-derive: `common.hpp`'s loop bound is `base + WAVE * 8 <= dim`.
const WAVE_COLS: usize = 32 * 8;

/// **THE TOLERANCE CAN SEE A PAST-THE-FIRST-TRIP DEFECT — proven on the host, no GPU.**
///
/// This is the non-vacuity half of the test below, and it is separate precisely because it
/// needs no device: it compares the CPU oracle against the CPU oracle with the injected defect
/// applied to the same bytes, and asserts the disagreement EXCEEDS `err_tol`'s
/// `1e-3 * max + 1e-3`. Without it, "the multi-trip test would go red on a broken unroll" is a
/// claim about a gate nobody has seen fire — and this repo has already shipped one acceptance
/// bar that was arithmetically incapable of detecting its own red target.
///
/// What it does NOT prove: that the GPU test below is wired up correctly. That needs a
/// deliberately broken kernel and a device, and is recorded as outstanding in
/// `docs/investigations/int4-moe-unroll.md` §G4.
///
/// The defect is aimed at ARITHMETIC, not at fold order. A reassociation is invisible to any
/// tolerance test by construction — that is the fingerprint's job, and the `glmi4` round's X
/// arm discharged it.
#[test]
fn the_i4_multi_trip_tolerance_can_see_a_past_first_trip_defect() {
    let c = i4_multi_trip_fixture();
    let want = i4_reference(&c);
    // The same fixture with the defect injected into all three projections —
    // `dot_i4_wave_r` serves gate, up and down, so a broken unroll breaks all three.
    let mut bad = i4_multi_trip_fixture();
    for (p, &(o_dim, i_dim)) in i4_expert_dims(c.d).iter().enumerate() {
        i4_drop_n7_after_first_trip(&mut bad.enc[0][p].0, o_dim, i_dim);
    }
    let got = i4_reference(&bad);

    let (err, tol) = err_tol(&want, &got);
    assert!(
        err > tol,
        "the injected past-first-trip defect is INVISIBLE to this tolerance: \
         err={err:.3e} <= tol={tol:.3e}. The multi-trip test would pass on a broken unroll."
    );
    println!(
        "multi-trip red target: err={err:.3e} vs tol={tol:.3e} ({:.1}x)",
        err / tol
    );
}

/// **The int4 dword path against the oracle at MULTIPLE trips, with a remainder in both
/// passes.** The int4 twin of `v4_kernel.rs::the_dword_path_matches_the_oracle_at_multiple_
/// trips`, and the gate `docs/investigations/int4-moe-unroll.md` §G4 requires before
/// `#pragma unroll 4` on `dot_i4_wave_r` can merge (measured 2026-08-09: +12.6% / +16.4% /
/// +20.1%, fingerprint-identical).
///
/// **The coverage hole this closes, stated exactly.** `moe_i4_matches_reference` above runs
/// `hidden = 256, inter = 128`: gate/up execute the dword loop ONCE and `moe_down_i4` never
/// enters it at all (128 < `WAVE * 8`). `moe_i4_real_data_matches_cpu` does reach 24 and 8
/// trips — but it `return`s early when the artifact is absent, and 24 and 8 are both divisible
/// by 2 and by 4, so it cannot execute an unroll remainder on any machine. **Before this test,
/// nothing checked `dot_i4_wave_r` past its first trip on a box without the artifact, and
/// nothing anywhere checked the epilogue an unroll creates.**
///
/// Non-vacuity is proven separately and without a device by
/// `the_i4_multi_trip_tolerance_can_see_a_past_first_trip_defect`, which injects M11's
/// `n7`-zeroed-when-`base != 0` into the same bytes and asserts the disagreement exceeds this
/// tolerance. Read that one before trusting this one green.
#[test]
fn the_i4_dword_path_matches_the_oracle_at_multiple_trips() {
    let c = i4_multi_trip_fixture();
    assert_close(&i4_reference(&c), &gpu_i4_moe(&c), "moe_i4 multi-trip");
}

/// The int4 MoE pair (moe_gateup_i4 / moe_down_i4).
///
/// **The MoE arm of `kernel.rs::batched_rows_are_bit_identical_to_single_rows`**, which states
/// the claim for every kernel that takes `nrow` and still runs the gemv and MLA arms. It became
/// a `#[test]` of its own on 2026-08-15 with the split: the three arms that stayed need the
/// gemv and kv_b launch wrappers, this one needs the expert-range machinery, and keeping them
/// under one function would have kept both halves in one file. Splitting the umbrella does not
/// weaken it — each arm always asserted its own kernel and named it in the failure.
///
/// It draws from its OWN `Lcg` for the same reason: it used to take the umbrella's, third in
/// line behind the gemv arms, so its fixture was whatever that stream had reached. Any seed
/// works — the comparison is a kernel against itself at two row counts, not against a
/// reference — which is why the umbrella's seed is reused verbatim rather than a new one
/// invented.
///
/// Compares the FIXED-POINT accumulator as u64, not the drained f32: integer equality is
/// the strictest form of "bit-identical" available and it is the quantity the atomics
/// actually produce.
///
/// The weight matrix is deliberately ragged — row 1 carries 0.0 on expert 1 — so this
/// covers the union's correctness argument as well as the batching: a row that did not
/// route to a union expert must come out exactly as if that expert were never launched.
/// That is the property the whole speculative claim rests on, and `0 * dv` with a
/// non-finite `dv` would otherwise clamp to a FINITE extreme rather than vanish.
#[test]
fn batched_moe_rows_are_bit_identical_to_single_rows() {
    let mut r = Lcg(0xba7c);
    let (d, ne) = (Dims::new(2 * I4_GROUP, I4_GROUP), 2usize);
    let (hidden, inter) = (d.hidden, d.inter);
    let mut bufs: Vec<DeviceBuf> = Vec::new();
    let enc = encode_experts(&mut r, ne, &i4_expert_dims(d), &|r, _p, o_dim, i_dim| {
        i4_group_varying(r, o_dim, i_dim)
    });
    let descb = desc_buf(&expert_descs(&enc, &mut bufs, f32b));
    let xs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..hidden).map(|_| r.f()).collect());
    // Row 0 routes to both experts; row 1 only to expert 0. Indexed `[e * R + t]`.
    let w1: [Vec<f32>; 2] = [vec![0.75, 0.5], vec![0.25, 0.0]];
    let stream = HipStream::new().expect("stream");
    let cx = MoeCtx::new(&descb, &stream);
    let g = MoeRange::new(hidden, inter, 0, ne);
    // Launch and join only — the caller reads its OWN accumulator back, because that
    // is the buffer the comparison is about and `MoeIo` has already given up its borrow.
    let run = |io: MoeIo<'_>, nrow: usize| {
        expert_range_i4(io, &cx, g, nrow);
        device_sync().expect("sync");
    };

    let single: Vec<Vec<u8>> = xs
        .iter()
        .zip(&w1)
        .map(|(x, w)| {
            let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
            let mut hb = dev(&vec![0u8; ne * inter * 4]);
            let mut ab = dev(&vec![0u8; hidden * 8]);
            run(MoeIo::new(&xb, &wb, &mut hb, &mut ab), 1);
            ab.copy_out().expect("out")
        })
        .collect();
    // Row-fastest: [e0r0, e0r1, e1r0, e1r1] — expert 1's row-1 weight is the 0.0.
    let wboth = vec![w1[0][0], w1[1][0], w1[0][1], w1[1][1]];
    let xboth: Vec<f32> = xs[0].iter().chain(&xs[1]).copied().collect();
    let (xb, wb) = (dev(&f32b(&xboth)), dev(&f32b(&wboth)));
    let mut hb = dev(&vec![0u8; ne * 2 * inter * 4]);
    let mut ab = dev(&vec![0u8; 2 * hidden * 8]);
    run(MoeIo::new(&xb, &wb, &mut hb, &mut ab), 2);
    let got = ab.copy_out().expect("out");
    assert_rows(&got, &single, hidden * 8, "moe_i4");
}

/// `moe_acc_drain_to` — the latent-width drain Kimi-K3 needs, against `moe_acc_drain`.
///
/// The two share ONE templated body in `moe.hip` and differ in exactly one line — `=` against `+=`
/// — which is also the only difference the code cannot make visible. Two things are asserted here:
///
/// 1. **`=` not `+=`.** The destination is pre-filled with a poison value. The sibling would add to
///    it; this must overwrite it. That is what makes K3's aggregate correct — it goes on to be
///    RMSNormed and up-projected, so a stale addend from the previous layer would be a plausible
///    wrong aggregate on every layer, forever, with nothing finite to notice.
/// 2. **The accumulator is RESET.** Shared with the sibling and load-bearing for the same reason:
///    steady-state decode does no memset before a layer's experts, so a drain that forgot would
///    double-count from the next layer on. It lives in the shared body, so this covers both.
///
/// `n` is NOT asserted and cannot be: it is a CALLER contract — the kernel treats it exactly as its
/// sibling does — guarded where the width is chosen, at `launch_moe_acc_drain_to`'s doc and at
/// whatever K3 layer loop S3 writes.
///
/// Fixed-point in, exact out: `MOE_ACC_SHIFT` is 44, and every value here is a small multiple of
/// `2^-44` scaled by a power of two, so the expected result is exact in f32 and this is
/// `assert_bits` rather than a tolerance.
#[test]
fn moe_acc_drain_to_writes_the_latent_aggregate_and_resets() {
    // Two rows — one per stream, which is what `rows` counts. 8 keeps the readback small and
    // readable. It does NOT exercise the multi-block path: `(8 + 255) / 256` is 1, and so is every
    // +/-1 perturbation of that expression, so only a truncating `n / 256` would redden. Said plainly
    // because an earlier draft claimed "an off-by-one in the block math would show" and it would
    // not; the grid math is covered by the real widths at decode, not here.
    let (n, rows) = (8usize, 2usize);
    const SCALE: f64 = (1u64 << 44) as f64; // MOE_ACC_SHIFT, mirrored from kernels/moe.hip
    // Row r contributes (o + 1) · (r + 1) · 2^44, so every output is an exact small integer and
    // the two rows are distinguishable — a drain that read only row 0 returns a THIRD of the
    // answer (1·(o+1) of 3·(o+1)), not half.
    let acc: Vec<u64> = (0..rows)
        .flat_map(|r| (0..n).map(move |o| ((o + 1) * (r + 1)) as u64 * (1u64 << 44)))
        .collect();
    let want: Vec<f32> = (0..n)
        .map(|o| {
            let t: u64 = (0..rows).map(|r| ((o + 1) * (r + 1)) as u64).sum::<u64>() * (1u64 << 44);
            (t as f64 / SCALE) as f32
        })
        .collect();

    let bytes: Vec<u8> = acc.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut accb = dev(&bytes);
    // POISON, not zeros: this is difference 1, and a destination of zeros would make `=` and `+=`
    // indistinguishable — which is exactly the mistake a reader porting from the sibling makes.
    let poison = -1234.5f32;
    let mut outb = dev(&f32b(&vec![poison; n]));
    let stream = HipStream::new().expect("stream");

    let io = Drain::new(&mut outb, &mut accb);
    drain_to(io, n, rows, &stream).expect("moe_acc_drain_to");
    device_sync().expect("sync");

    let got = f32v(&outb.copy_out().expect("out"));
    // `assert_bits` is bit-exact over every element and every `want` is positive, so it already
    // fails — naming the index — on a `+=` kernel (`poison + want`), on a no-write kernel (`poison`),
    // and on a transposition. No `!got.contains(&poison)` follow-up: there is no input where the
    // bits match and the poison survives.
    assert_bits(&want, &got, "moe_acc_drain_to");
    assert_acc_drained(
        &accb,
        "moe_acc_drain_to left the accumulator dirty — steady-state decode does no memset, so \
         the next layer would double-count",
    );

    // The dimension guard, by CODE rather than by is_err: 1001 is `moe.hip`'s non-positive-dimension
    // class. There is no float guard to test — the kernel takes no scalar, which is the point argued
    // at `launch_moe_acc_drain_to`. It briefly took a `gain` with a 1006 guard against `0`, `-1` and
    // `+inf`; deleting the parameter deleted the only way to reach any of them.
    for (bad_n, bad_rows) in [(0, rows), (n, 0)] {
        let io = Drain::new(&mut outb, &mut accb);
        let r = drain_to(io, bad_n, bad_rows, &stream);
        assert_guard(r, Some(1001), &format!("n={bad_n} rows={bad_rows}"));
    }
}
