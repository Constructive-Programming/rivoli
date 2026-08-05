//! **The S2a kernels, scored against the S1b oracle.**
//!
//! `src/v4oracle/` is a CPU transliteration of `inference/model.py` that was proved before
//! it was trusted (`tests/v4_oracle.rs`: ~40 deliberate breakages, each asserted both to
//! fire where it should and to be silent where it should not). This file is the other half
//! of that arrangement: it runs the MoE-side HIP kernels against the same toy checkpoint
//! and asks whether they compute what the oracle computes.
//!
//! # Why numerical comparison and nothing else
//!
//! Every defect available on this path is silent. An unclamped SwiGLU, a `w1`/`w3` swap, a
//! high-first nibble read, a group stride of 128 instead of 32, a missing activation
//! quantization, a bias that reached the routing weights — none crash, all leave every
//! shape, magnitude, norm and code histogram plausible, and `distinct`/`longest repeated
//! block` fire identically on all of them (CLAUDE.md; they have misled three
//! investigations here). So the tests below do not check that the kernels produce numbers.
//!
//! # What this file CANNOT detect, measured rather than assumed
//!
//! 1. **The Sinkhorn iteration count.** At `hc_sinkhorn_iters = 20` a 4x4 positive matrix
//!    is far past convergence: 19 and 20 agree BIT-FOR-BIT, so no golden distinguishes
//!    them (`tests/v4_oracle.rs::sinkhorn_has_converged_long_before_iteration_20`, which is
//!    why `Defect::SinkhornOneFewerIter` is excluded from the oracle's own matrix). What
//!    [`sinkhorn_iteration_count_is_live`] proves is strictly weaker and is all that is
//!    available: the parameter reaches the arithmetic (2 and 20 disagree). The exact value
//!    is gated by SOURCING, not by measurement — it is passed from `V4Config`, which
//!    `V4Config::assert_matches_reference_json` pins to `config.json`.
//! 2. **The shared expert.** It is fp8 e4m3 at 128x128, not FP4, and is a different kernel
//!    that is already in the tree. [`ffn_out_matches_the_golden`] fills it in from the
//!    ORACLE, so that test says nothing about rivoli's fp8 path.
//! 3. **Batch > 1.** Nothing here covers it, and nothing can: the oracle itself is
//!    `bsz = 1` only (`forward.rs`, "Out of scope"). The FP4 launcher therefore REFUSES
//!    `nrow != 1` rather than shipping an unscoreable second row —
//!    [`expert_range_f4_guards`] pins that. Speculative decode is a `--features rocm`
//!    capability on the GLM formats and is not one here yet.
//! 4. **The real checkpoint's values.** Everything runs on `V4Config::toy`, which
//!    preserves every discriminant and shrinks every extent. `toy` has `moe_inter_dim =
//!    128`, so `moe_down_f4` never enters `dot_f4_wave_r`'s 8-nibble dword fast path (it
//!    needs `WAVE·8 = 256` columns) — only `moe_gateup_f4` does, at exactly one iteration.
//!    Both PATHS run; neither runs at depth.
//! 5. **The e8m0 endpoints and the e2m1/e8m0 codes the fixture happens not to contain.**
//!    There is no exhaustive codec probe here — the accumulator's dynamic range forbids one
//!    (see [`the_fixture_exercises_the_codes_the_decoders_are_credited_with`], which
//!    measures what IS covered instead of assuming it). `e8m0f`'s `0x00` (2^-127) and
//!    `0xff` (NaN) arms are executed by nothing in this file.
//!
//! 6. **`hc_pre`'s mixes GEMV re-association is unnamed upstream.** `forward.rs`'s fidelity
//!    note lists `fp8_gemm`, `sparse_attn` and `hc_split_sinkhorn`'s warp reductions as
//!    reproduced only up to summation order. It does NOT list the `hc_fn` GEMV, which this
//!    kernel also wave-reduces where the oracle sums sequentially. Same class, and it rides
//!    the same [`TOL`] — but it is an unstated consumer of it, and the oracle's scope note
//!    should say so.
//!
//! Runs on the toy config, not the checkpoint: the questions are structural, and this way
//! they are re-answered in seconds on every `cargo test --release --features rocm`.

#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli::backend::gpustream::HipStream;
use rivoli::backend::hip::{
    ExpertDescF4, device_sync, launch_act_quant_f8, launch_hc_post, launch_hc_pre,
    launch_moe_acc_drain, launch_moe_expert_range_f4, launch_moe_gate_v4, launch_rmsnorm,
};
use rivoli::memory::device::DeviceBuf;
use rivoli::v4oracle::{
    forward::{Capture, Counters, Defect, ExpertW, LayerW, Oracle, Step},
    numerics::{act_quant_inplace, bf16_decode, bf16_encode},
    toy::{self, ToyModel},
    weights::{NamedRng, V4Config, WMat},
};
use std::sync::OnceLock;

mod common;
use common::{f32b, f32v, max_abs, report_rel};

// =======================================================================================
// fixture
// =======================================================================================

/// The toy model AND a defect-free oracle over it, built once.
///
/// `V4Config::toy` keeps `n_hash_layers = 3` over 4 layers, so layers 0-2 route by
/// `tid2eid` and layer 3 by score — both router modes reachable from one fixture, which is
/// the only reason a 4-layer toy has 4 layers.
///
/// The `Oracle` is cached here and NOT in `tests/v4_oracle.rs`'s equivalent, because that
/// file constructs a fresh one per `Defect` — running the breakage matrix is its whole job.
/// This file only ever wants `Defect::None`: the deliberate breaks here live in the KERNEL
/// side, not in the oracle.
fn fixture() -> &'static (V4Config, ToyModel, Oracle) {
    static M: OnceLock<(V4Config, ToyModel, Oracle)> = OnceLock::new();
    M.get_or_init(|| {
        let cfg = V4Config::toy();
        let m = toy::build(&cfg);
        let o = Oracle::new(cfg.clone(), Defect::None);
        (cfg, m, o)
    })
}

/// Upload to a fresh device buffer. `max(1)` because a zero-length allocation is not a
/// thing this allocator does and an empty span is never what a caller meant.
fn to_device(bytes: &[u8]) -> DeviceBuf {
    let mut buf = DeviceBuf::new(bytes.len().max(1)).expect("device alloc");
    buf.copy_in_at(0, bytes).expect("host to device");
    buf
}

/// A zeroed device buffer of `n` bytes.
fn zeros(n: usize) -> DeviceBuf {
    to_device(&vec![0u8; n])
}

/// The bit patterns of a slice, for the assertions that claim EXACTNESS rather than
/// closeness.
///
/// `assert_eq!` on `Vec<f32>` is `PartialEq`, which is neither: it reports `-0.0 == 0.0`
/// (so a sign-flipped zero passes an "identical" claim) and `NaN != NaN` (so a
/// NaN-poisoned kernel passes a "these must differ" claim for the wrong reason). Four
/// assertions here say "bit for bit" in as many words; this is what makes them true.
/// `act_quant_f8_is_bit_identical_to_the_oracle` open-codes the same comparison a fifth
/// time, as a find-first loop, so its failure can name the offending element.
fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Join the device, then read a buffer back as f32.
///
/// The sync is INSIDE this rather than at each call site on purpose: a readback that
/// skipped it would return whatever was in the staging buffer, and the resulting
/// comparison would be against stale data rather than against nothing — a green test on an
/// unwritten result.
fn sync_f32(b: &DeviceBuf) -> Vec<f32> {
    device_sync().expect("device sync");
    f32v(&b.copy_out().expect("device to host"))
}

/// The packed nibbles and e8m0 scale bytes of an FP4 weight, as the checkpoint stores them
/// and as `dot_f4_wave_r` reads them — the SAME bytes the oracle's `WMat::Fp4::row`
/// decodes. That is the whole design of this comparison: one byte array, two independently
/// written decoders, and any disagreement is a decoder bug rather than a fixture artefact.
fn fp4_spans(m: &WMat) -> (&[u8], &[u8]) {
    match m {
        WMat::Fp4 { w, s, .. } => (w, s),
        WMat::Dense { .. } | WMat::Fp8 { .. } => panic!("expected an fp4 weight"),
    }
}

/// The device-side FP4 expert set: one `ExpertDescF4` per routed expert, plus the buffers
/// that keep the six spans alive.
///
/// `parts` is what stops the descriptors from dangling — an `ExpertDescF4` is six raw
/// addresses and owns nothing.
struct F4Experts {
    descs: DeviceBuf,
    n: usize,
    #[allow(dead_code)]
    parts: Vec<DeviceBuf>,
}

/// Which projection slot a `WMat` is uploaded into. Named because the whole point of the
/// `w1`/`w3` tests is that the two are the same SHAPE, so nothing but the name distinguishes
/// them and a swap is invisible to every structural check (`quant.rs::V4_PROJ`).
#[derive(Clone, Copy)]
enum Wiring {
    /// `gate = w1, up = w3` — the reference (`Expert.forward`: `gate = self.w1(x)`).
    Correct,
    /// `gate = w3, up = w1`. Same shapes, same byte counts, same scale grids.
    SwapGateUp,
    /// The reference wiring with every weight byte's nibbles exchanged — a permutation
    /// INSIDE each 32-element scale group, so the group boundaries, the amax/scale relation
    /// and the code histogram are all invariant under it.
    SwapNibbles,
}

fn nibble_swapped(bytes: &[u8]) -> Vec<u8> {
    bytes.iter().map(|b| b.rotate_left(4)).collect()
}

impl F4Experts {
    fn upload(experts: &[&ExpertW], wiring: Wiring) -> Self {
        let mut parts = Vec::new();
        let mut descs = Vec::new();
        for e in experts {
            // ONE exhaustive match decides both halves. Deciding the nibble order separately
            // with `wiring == Wiring::SwapNibbles` would put a wildcard in disguise beside an
            // exhaustive match: a new variant would compile-error here and silently take the
            // `else` there, producing a "break" test that quietly ran the reference wiring.
            let (gate, up_proj, swap_nibbles) = match wiring {
                Wiring::Correct => (&e.w1, &e.w3, false),
                Wiring::SwapGateUp => (&e.w3, &e.w1, false),
                Wiring::SwapNibbles => (&e.w1, &e.w3, true),
            };
            let mut push = |m: &WMat| {
                let (w, sc) = fp4_spans(m);
                let wb = if swap_nibbles { to_device(&nibble_swapped(w)) } else { to_device(w) };
                let sb = to_device(sc);
                // Addresses taken BEFORE the move. Recovering them as `parts[n-2]`/`[n-1]`
                // after the pushes works until someone adds a third buffer, at which point the
                // descriptor silently points at another projection's weights — which is
                // precisely the class of defect this file exists to detect, presenting as an
                // arithmetic miss.
                let addrs = (wb.ptr(), sb.ptr());
                parts.push(wb);
                parts.push(sb);
                addrs
            };
            let (gp, gs) = push(gate);
            let (up_p, up_s) = push(up_proj);
            let (dp, ds) = push(&e.w2);
            descs.push(ExpertDescF4 {
                gate_packed: gp,
                gate_scale: gs,
                up_packed: up_p,
                up_scale: up_s,
                down_packed: dp,
                down_scale: ds,
            });
        }
        // SAFETY: `ExpertDescF4` is plain addresses, so the span is exactly the slice's bytes.
        let raw = unsafe {
            std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(&descs[..]))
        };
        Self { descs: to_device(raw), n: descs.len(), parts }
    }
}

// =======================================================================================
// the routed-expert half, on the GPU
// =======================================================================================

/// One FP4 MoE dispatch: the weights, the activation, the picks, and the two knobs the
/// deliberate-break tests turn.
///
/// A struct rather than seven positional parameters, for the reason `tests/kernel.rs`'s
/// `MoeIo` gives — the two entry points below take the identical list and differ only in how
/// the expert range is cut, so writing it twice would be two places for `hidden` and `inter`
/// to get swapped.
struct Dispatch<'a> {
    cfg: &'a V4Config,
    experts: &'a F4Experts,
    x: &'a [f32],
    /// One weight per UPLOADED expert, by ABSOLUTE id. `0.0` means this token did not route
    /// there — the kernel still runs the expert and adds exactly zero.
    wexpert: &'a [f32],
    swiglu_limit: f32,
    /// Quantize `x` to fp8 at block 128 first, as `model.py::linear` does. `false` is a
    /// deliberate break with exactly one caller.
    quantize_x: bool,
}

impl Dispatch<'_> {
    /// `Σ_e down_e(silu(clamp(w1_e·x)) ⊙ clamp(w3_e·x) · weight_e)` — the routed half of
    /// `MoE.forward`, without the shared expert and without the final bf16 store. Every
    /// expert in ONE range.
    fn run(&self) -> Vec<f32> {
        self.in_ranges(&[(0, self.experts.n)])
    }

    /// The same, dispatched as several `[e_start, e_start+e_count)` ranges into one
    /// accumulator — the shape S3's two streams will use, and the only thing here that
    /// exercises `e_start > 0`.
    fn in_ranges(&self, ranges: &[(usize, usize)]) -> Vec<f32> {
        let (hidden, inter) = (self.cfg.dim, self.cfg.moe_inter_dim);
        assert_eq!(self.wexpert.len(), self.experts.n, "one routing weight per expert");
        let stream = HipStream::new().expect("stream");
        let mut xb = to_device(&f32b(self.x));
        let wb = to_device(&f32b(self.wexpert));
        let mut hb = zeros(self.experts.n * inter * 4);
        let mut ab = zeros(hidden * 8);
        let mut ob = zeros(hidden * 4);
        if self.quantize_x {
            // SAFETY: `xb` is `hidden` live f32 and outlives the stream's completion below.
            unsafe { launch_act_quant_f8(xb.ptr_mut() as *mut f32, 1, hidden, stream.raw()) }
                .expect("act_quant_f8");
        }
        // Hoisted for line length, not for the borrow checker: `hb`, `ab` and `ob` are
        // distinct locals, so the `ptr_mut()` calls would borrow-check inline too.
        let (xp, wp) = (xb.ptr() as *const f32, wb.ptr() as *const f32);
        let dp = self.experts.descs.ptr() as *const ExpertDescF4;
        let (hp, ap, op) =
            (hb.ptr_mut() as *mut f32, ab.ptr_mut() as *mut u64, ob.ptr_mut() as *mut f32);
        // SAFETY: every buffer above is sized for `experts.n` ABSOLUTE expert slots and is
        // alive until the sync; every range is inside that bound.
        unsafe {
            for &(e_start, e_count) in ranges {
                launch_moe_expert_range_f4(
                    xp, hidden, inter, e_start, e_count, self.experts.n, dp, wp,
                    self.swiglu_limit, hp, ap, 1, stream.raw(),
                )
                .expect("moe_expert_range_f4");
            }
            // Same stream, so every range's atomics precede the drain.
            launch_moe_acc_drain(op, ap, hidden, 1, 1.0, stream.raw()).expect("moe_acc_drain");
        }
        sync_f32(&ob)
    }
}

/// The two experts every `Case` routes.
///
/// A constant rather than a `Case` field: `the_fixture_exercises_the_codes_...` is a
/// host-only histogram over those experts' weights, and reading them off a `Case` would make
/// it build device buffers and run an oracle expert pass to learn two integers.
const PICKS: [usize; 2] = [1, 5];

/// A residual-stream-shaped activation: bf16-representable, scaled so the caller can drive
/// the SwiGLU clamp on or off. `scale` is the knob the clamp test turns.
fn draw_x(tag: &str, n: usize, scale: f32) -> Vec<f32> {
    let mut r = NamedRng::new(tag);
    (0..n).map(|_| bf16_decode(bf16_encode(r.unit() * scale))).collect()
}

/// One expert-comparison case: a layer, an activation, a fixed set of picks, and the
/// oracle's answer for them.
///
/// Built once per test so the deliberate-break tests differ ONLY in the thing they break —
/// the wiring, the SwiGLU limit, or whether `x` was quantized. Four hand-rolled setups
/// would be four chances for a break test to accidentally change the fixture too, which
/// would make its disagreement prove nothing.
struct Case {
    cfg: &'static V4Config,
    /// This case's routed experts, in ABSOLUTE expert order. Held so `broken` re-uploads
    /// the SAME weights under a different wiring rather than re-deriving which layer they
    /// came from — a `broken` that hard-coded layer 0 would silently ignore its `Case`.
    all: Vec<&'static ExpertW>,
    x: Vec<f32>,
    /// One weight per routed expert, in ABSOLUTE expert order — every expert is uploaded,
    /// so the unrouted ones ride through the kernel at weight 0.
    wexpert: Vec<f32>,
    /// `Σ` over the routed experts through `Oracle::expert`, ascending expert id: exactly
    /// `MoE.forward` minus the shared expert and minus the final bf16 store.
    want: Vec<f32>,
    /// The oracle's OWN count of SwiGLU clamp events for this case, measured independently
    /// of whether any kernel clamped.
    clamp_events: usize,
    experts: F4Experts,
}

impl Case {
    /// `scale` sets how hard the activation drives the SwiGLU: 1.0 stays inside the clamp,
    /// large values cross it. Two picks at fixed ids with DIFFERENT, non-unit weights, so a
    /// kernel that dropped the routing weight or reused one for both experts fails.
    fn new(layer: usize, tag: &str, scale: f32) -> Self {
        let (cfg, m, o) = fixture();
        let lw = &m.layers[layer];
        let x = draw_x(tag, cfg.dim, scale);
        let mut wexpert = vec![0.0f32; cfg.n_routed_experts];
        wexpert[PICKS[0]] = 1.125;
        wexpert[PICKS[1]] = 0.375;

        let mut counters = Counters::default();
        let mut want = vec![0.0f32; cfg.dim];
        for (e, &w) in wexpert.iter().enumerate() {
            if w == 0.0 {
                // The kernel still runs this expert and adds exactly 0.0 — see
                // `an_unrouted_expert_contributes_exactly_zero`. Skipping here is not an
                // asymmetry, it is `MoE.forward`'s own `counts[i] == 0` skip.
                continue;
            }
            for (a, b) in want.iter_mut().zip(&o.expert(&lw.experts[&e], &x, 1, Some(&[w]), &mut counters)) {
                *a += b;
            }
        }
        // `moe_fixed` CLAMPS each expert contribution at `MOE_ACC_MAX = 2^(58-44) = 16384`
        // (`common.hpp`) and the oracle has no such clamp, so a case driven near it would
        // fail `assert_matches` for a reason unrelated to what the test is asking. The clamp
        // case runs at activation scale 48 — well inside, but close enough that leaving this
        // unstated would make a future scale bump a confusing red.
        let mx = max_abs(&want);
        assert!(mx < 8192.0, "case output {mx:.3e} is within 2x of moe_fixed's 2^14 clamp, \
                              which the oracle does not have — lower the activation scale");

        let all: Vec<&ExpertW> = (0..cfg.n_routed_experts).map(|e| &lw.experts[&e]).collect();
        let experts = F4Experts::upload(&all, Wiring::Correct);
        Self {
            cfg,
            all,
            x,
            wexpert,
            want,
            clamp_events: counters.swiglu_clamp_events,
            experts,
        }
    }

    /// This case as a dispatch, with `experts` and the two knobs overridable by `broken`.
    fn dispatch<'a>(&'a self, experts: &'a F4Experts, limit: f32, quantize_x: bool) -> Dispatch<'a> {
        Dispatch { cfg: self.cfg, experts, x: &self.x, wexpert: &self.wexpert,
                   swiglu_limit: limit, quantize_x }
    }

    /// The GPU answer for this case, under the reference wiring and the config's clamp.
    fn gpu(&self) -> Vec<f32> {
        self.dispatch(&self.experts, self.cfg.swiglu_limit, true).run()
    }

    /// The GPU answer with ONE thing broken, and nothing else changed.
    fn broken(&self, wiring: Wiring, swiglu_limit: f32, quantize_x: bool) -> Vec<f32> {
        let e = F4Experts::upload(&self.all, wiring);
        self.dispatch(&e, swiglu_limit, quantize_x).run()
    }
}

// =======================================================================================
// 1. the FP4 expert
// =======================================================================================

/// The load-bearing comparison: one MoE layer's routed experts, GPU against oracle.
///
/// A failure here is any of — a wrong e2m1 codebook, a wrong e8m0 decode, a group stride
/// of 128 instead of 32, a missing or misplaced bf16 store, the routing weight applied
/// after `w2` instead of to the intermediate, a missing activation quantization, or the
/// clamp on the wrong side of the gate. The tests that follow separate those out; this one
/// is what would catch a defect nobody thought to name.
#[test]
fn routed_experts_match_the_oracle() {
    let c = Case::new(0, "moe-x", 1.0);
    // Not vacuous: a comparison against an all-zero want would pass for any kernel.
    assert!(c.want.iter().any(|v| v.abs() > 1e-6), "the oracle produced nothing to compare");
    assert_eq!(c.clamp_events, 0, "this case is the UNCLAMPED half of the clamp bracket");
    assert_matches(&c.want, &c.gpu(), "routed experts (fp4)");
}

/// Three silent breaks, each asserted to be VISIBLE to the gate above.
///
/// Together they are the argument that `routed_experts_match_the_oracle` passing means
/// something. Each changes exactly one thing:
///
/// - **`w1`/`w3` swapped.** Identical shapes, identical byte counts, identical scale grids,
///   and S1a's repack maps both through the same name→slot table, so nothing structural can
///   see it (`quant.rs::V4_PROJ` says so). It is detectable ONLY because SwiGLU is
///   asymmetric in its two operands — `silu` applies to the gate alone. Were the combine
///   `g · u`, the swap would be a no-op and no instrument could ever find it.
/// - **Nibbles read high-first.** Swapping the BYTES is the same experiment from the other
///   end and needs no second kernel: a high-first kernel on real bytes and a low-first
///   kernel on swapped bytes compute the same wrong thing. `src/artifact/format.rs` records
///   that the `.f4` repack cannot check this and names a `matvec_f4` as where it becomes
///   checkable — this is that check.
/// - **`x` not fp8-quantized.** `model.py::linear` line 120 quantizes the activation in
///   front of the fp4 GEMM. Dropping it leaves every magnitude within 2^-3 of right.
#[test]
fn the_silent_fp4_breaks_are_visible() {
    let c = Case::new(0, "moe-x", 1.0);
    let lim = c.cfg.swiglu_limit;
    for (label, got) in [
        ("w1/w3 swapped", c.broken(Wiring::SwapGateUp, lim, true)),
        ("nibbles read high-first", c.broken(Wiring::SwapNibbles, lim, true)),
        ("x not fp8-quantized", c.broken(Wiring::Correct, lim, false)),
    ] {
        assert_disagrees(&c.want, &got, label);
    }
}

/// The clamped SwiGLU (`swiglu_limit = 10.0`), which rivoli's own SwiGLU does not have.
///
/// BIDIRECTIONAL, and the direction that matters is the second: a clamp test on activations
/// that never reach the limit passes for a kernel with no clamp at all. The oracle's own
/// `swiglu_clamp_events` is what makes the fixture's reachability a MEASUREMENT — it is
/// counted independently of whether any kernel clamped, and
/// `routed_experts_match_the_oracle` asserts the other end of the bracket (zero events).
#[test]
fn the_swiglu_clamp_is_live_and_the_fixture_reaches_it() {
    let c = Case::new(0, "moe-x-big", 48.0);
    assert!(
        c.clamp_events > 0,
        "the fixture never reaches the clamp, so this test could not distinguish a clamped \
         kernel from an unclamped one — raise the activation scale"
    );
    assert_matches(&c.want, &c.gpu(), "clamped swiglu");
    // Effectively unclamped, but positive — the launcher refuses 0 outright, which is the
    // stronger guarantee and the reason this arm has to go the long way round.
    assert_disagrees(
        &c.want,
        &c.broken(Wiring::Correct, 1e6, true),
        "swiglu limit raised to 1e6",
    );
}

/// An unrouted expert contributes exactly zero, and the routed sum does not depend on how
/// many unrouted experts rode along.
///
/// `moe_down_f4` takes no routing mask — the weight is already in `h` — so "did not route"
/// is `h == 0` and nothing else. That the contribution is EXACTLY 0.0 rather than merely
/// small is what makes the missing mask safe, and it is not obvious: it needs `0 · finite`
/// to stay zero through the fp8 re-quantization of `h` and through `moe_fixed`.
#[test]
fn an_unrouted_expert_contributes_exactly_zero() {
    let c = Case::new(0, "moe-x", 1.0);
    let full = c.gpu();
    // The same two picks, dispatched with ONLY the two experts they name uploaded. Every
    // unrouted expert is gone rather than zero-weighted, so if any of them was contributing
    // anything at all — a denormal, a NaN turned finite by `moe_fixed`'s clamp — the two
    // results differ. Compared as BIT PATTERNS, not as f32: the claim is exactness, and
    // `PartialEq` on f32 reports `-0.0 == 0.0` — which a zero contribution can produce.
    let named: Vec<&ExpertW> = PICKS.iter().map(|&e| c.all[e]).collect();
    let two = F4Experts::upload(&named, Wiring::Correct);
    let w = PICKS.map(|e| c.wexpert[e]);
    let just_two = Dispatch { cfg: c.cfg, experts: &two, x: &c.x, wexpert: &w,
                              swiglu_limit: c.cfg.swiglu_limit, quantize_x: true }
        .run();
    assert_eq!(bits(&full), bits(&just_two), "an unrouted expert perturbed the sum");
    assert!(full.iter().any(|v| v.abs() > 1e-6), "both arms produced zero — nothing compared");
}

/// `wexpert`, `h` and `descs` are indexed by ABSOLUTE expert id, so a dispatch split into
/// ranges gives the same answer as one range over everything.
///
/// This is the only test that passes `e_start > 0` to anything but a rejected arm, and the
/// convention it pins is one a caller can get wrong silently: reading `wexpert` as
/// range-relative and sizing it `e_count` compiles, and runs off the end the first time a
/// pipeline splits experts across two streams — which is the first thing `gpu.rs` does.
///
/// Two ranges that are not adjacent and do not start at 0, so a kernel that quietly used
/// `r / inter` as an absolute index, or offset `h` by the range rather than by the expert,
/// lands on different weights.
#[test]
fn a_dispatch_split_into_ranges_matches_one_range() {
    let c = Case::new(0, "moe-x", 1.0);
    // Exactly the two experts `Case` routes, each its own range — and nothing else, so the
    // sum is over the same terms as `c.want` by a different dispatch.
    let split = c
        .dispatch(&c.experts, c.cfg.swiglu_limit, true)
        .in_ranges(&[(PICKS[0], 1), (PICKS[1], 1)]);
    assert!(c.want.iter().any(|v| v.abs() > 1e-6), "nothing to compare");
    assert_matches(&c.want, &split, "routed experts, dispatched as two ranges");
    // And bit-identical to the single-range dispatch: the fixed-point accumulator makes the
    // sum associative, so splitting it must change nothing at all, not merely little.
    assert_eq!(bits(&c.gpu()), bits(&split), "range split perturbed the sum");
}

/// Every launcher guard, by CODE. Accepting any error would pass a build where an
/// unrelated dimension check swallowed the case first.
#[test]
fn expert_range_f4_guards() {
    let (cfg, m, _) = fixture();
    let lw = &m.layers[0];
    let experts = F4Experts::upload(&[&lw.experts[&0]], Wiring::Correct);
    let x = zeros(cfg.dim * 4);
    let w = zeros(4);
    let mut h = zeros(cfg.moe_inter_dim * 4);
    let mut acc = zeros(cfg.dim * 8);
    let stream = HipStream::new().expect("stream");

    let mut go = |(hidden, inter, e_start, e_count, n_desc, limit, nrow)| {
        // SAFETY: every rejected case returns before a dereference; the accepted case is
        // sized by the buffers above.
        let r = unsafe {
            launch_moe_expert_range_f4(
                x.ptr() as *const f32,
                hidden,
                inter,
                e_start,
                e_count,
                n_desc,
                experts.descs.ptr() as *const ExpertDescF4,
                w.ptr() as *const f32,
                limit,
                h.ptr_mut() as *mut f32,
                acc.ptr_mut() as *mut u64,
                nrow,
                stream.raw(),
            )
        };
        r.map_err(|e| format!("{e:#}"))
    };
    let (hid, int, lim) = (cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit);

    assert!(go((hid, int, 0, 1, 1, lim, 1)).is_ok(), "the accepted case must be accepted");
    // The accepted case LAUNCHED. Join before the buffers drop: a launcher's `Ok` is
    // `hipGetLastError()` immediately after the launch, so an asynchronous fault would
    // otherwise surface in whichever test calls `device_sync()` next — and `cargo test` runs
    // these in parallel threads. A guard test that poisons an unrelated one is the
    // false-green trap this suite is built to not have.
    device_sync().expect("device sync");
    let cases = vec![
        (1001, "zero hidden", go((0, int, 0, 1, 1, lim, 1))),
        // 129 is not a multiple of ACT_QUANT_BLOCK. `assert N % block_size == 0` is the
        // reference's own; a ragged tail would quantize against a scale it never computes.
        (1002, "hidden not a whole act-quant block", go((129, int, 0, 1, 1, lim, 1))),
        (1002, "inter not a whole act-quant block", go((hid, 96, 0, 1, 1, lim, 1))),
        // BOTH sides of 1, which is what separates `!= 1` from a `> 1` that would accept 0.
        // The FP4 path instantiates only R=1: no measurement justifies a second row, and the
        // oracle is bsz=1, so one could not be scored even if it existed.
        (1003, "nrow 2", go((hid, int, 0, 1, 1, lim, 2))),
        (1003, "nrow 0", go((hid, int, 0, 1, 1, lim, 0))),
        // THE `.f4` BOUNDARY. `.vq3`/`.i4` carry the shared expert as block `n_experts`;
        // `.f4` does not, because V4's shared expert is fp8 e4m3 at 128x128. One past the
        // end here is the wrong ARITHMETIC, not merely the wrong weights.
        (1004, "one expert past the descriptor array", go((hid, int, 0, 2, 1, lim, 1))),
        (1004, "e_start past the descriptor array", go((hid, int, 1, 1, 1, lim, 1))),
        // 0.0 and NaN, not 0.0 and -10.0: `x <= 0` would reject the negative too, so only
        // NaN distinguishes the `!(x > 0)` spelling that is actually there.
        (1006, "unclamped swiglu", go((hid, int, 0, 1, 1, 0.0, 1))),
        (1006, "NaN swiglu limit", go((hid, int, 0, 1, 1, f32::NAN, 1))),
    ];
    assert_guards(cases);
}

// =======================================================================================
// 2. the fp8 activation quantizer
// =======================================================================================

/// One 128-element `act_quant` block whose amax is EXACTLY 1.0, so the scale is pinned at
/// `fast_round_scale(1, 1/448) = 2^-8` and the tie values below land where they are meant to.
///
/// Returns the block. Contents, in order and each for a reason:
///   * `1.0`, which sets the amax and nothing else;
///   * every SUBNORMAL halfway tie, `k·2^-9·s` for k in {0.5 … 7.5} — the range where
///     `kernels/common.hpp::f2e4m3` (rivoli's own encoder) rounds half-AWAY-from-zero while
///     `f2e4m3_rne` and the oracle round half-to-EVEN. This is the only place the two rules
///     differ, so a block without it cannot tell them apart;
///   * the same eight negated, because RNE is sign-symmetric and half-away-from-zero is too
///     — the asymmetry to catch is in the tie direction, not the sign;
///   * NORMAL halfway ties `(1 + (m+0.5)/8)·2^e·s`, the other tie family;
///   * zeros, and a spread of ordinary magnitudes.
fn act_quant_block(seed: &str) -> Vec<f32> {
    const S: f32 = 1.0 / 256.0; // 2^-8, what fast_round_scale(1.0, 1/448) returns
    let mut v = vec![1.0f32];
    for k in 0..8 {
        let tie = (k as f32 + 0.5) * (1.0 / 512.0) * S; // subnormal quantum is 2^-9
        v.push(tie);
        v.push(-tie);
    }
    for m in 0..8 {
        for e in -4i32..3 {
            v.push((1.0 + (m as f32 + 0.5) / 8.0) * (e as f32).exp2() * S);
        }
    }
    v.push(0.0);
    v.push(-0.0);
    let mut r = NamedRng::new(seed);
    while v.len() < 128 {
        // Scaled down so nothing displaces the 1.0 amax, and spread over binades so the
        // normal path runs at more than one exponent.
        v.push(r.unit() * 0.5f32.powi(r.below(12) as i32));
    }
    v.truncate(128);
    v
}

/// `act_quant_f8` against `v4oracle::numerics::act_quant_inplace`, BIT FOR BIT.
///
/// Exactness is the right assertion and not an over-reach: every step is a deterministic
/// IEEE operation — an `fmaxf` reduction, `fast_round_scale`'s bit surgery, one `fdiv`, a
/// clamp, and an encode/decode pair that are both exhaustively specified rules. `build.rs`
/// compiles the kernels with `-O3` and no `-ffast-math`, so the divide stays a real `fdiv`.
/// A tolerance here would hide exactly the one-ulp tie disagreement the fixture is built to
/// expose.
///
/// What it pins is the ROUND TRIP `e4m3_decode(e4m3_encode(x/s))·s`, which is what a V4 GEMM
/// consumes. A hypothetical encoder and decoder that were both shifted by the same amount
/// would cancel and pass; nothing else does.
///
/// **`f2e4m3_rne`'s saturation arm is unreachable from here, and from the model.** `s` is
/// `2^ceil(log2(amax/448)) >= amax/448`, so `|x|/s <= 448` always and neither the clamp nor
/// the `a >= 464` early return can fire. They are the format's own bounds, executed by
/// nothing in this suite.
#[test]
fn act_quant_f8_is_bit_identical_to_the_oracle() {
    const ROWS: usize = 6;
    let mut host: Vec<f32> =
        (0..ROWS).flat_map(|r| act_quant_block(&format!("actq-{r}"))).collect();
    // A row of two blocks, to prove the tiling advances by 128 within a row rather than
    // treating each row as one block: the second block's amax differs from the first's, so
    // a kernel that reused one scale for the whole row produces different numbers.
    let wide: Vec<f32> = act_quant_block("actq-wide")
        .into_iter()
        .chain(act_quant_block("actq-wide2").into_iter().map(|v| v * 0.125))
        .collect();
    host.extend_from_slice(&wide);

    let mut want = host.clone();
    for row in want.chunks_mut(128) {
        act_quant_inplace(row, 128, true);
    }

    let mut b = to_device(&f32b(&host));
    let stream = HipStream::new().expect("stream");
    // 128-wide rows for the first ROWS blocks, then the 256-wide one — dispatched as two
    // calls over one buffer, which is also how the MoE uses it (`x` then `h`).
    // SAFETY: the buffer holds `ROWS + 2` blocks of 128 live f32.
    unsafe {
        let p = b.ptr_mut() as *mut f32;
        launch_act_quant_f8(p, ROWS, 128, stream.raw()).expect("act_quant_f8 rows");
        launch_act_quant_f8(p.add(ROWS * 128), 1, 256, stream.raw()).expect("act_quant_f8 wide");
    }
    let got = sync_f32(&b);

    assert_ne!(bits(&want), bits(&host),
               "the quantizer left the input unchanged — nothing was compared");
    assert_eq!(want.len(), got.len());
    if let Some(i) = (0..want.len()).find(|&i| want[i].to_bits() != got[i].to_bits()) {
        panic!(
            "act_quant_f8 differs at element {i} (block {}, lane {}): in={:e} want={:e} got={:e}",
            i / 128,
            i % 128,
            host[i],
            want[i],
            got[i]
        );
    }
}

/// `row_len` that is not a whole `ACT_QUANT_BLOCK` is refused, matching `kernel.py:112`'s
/// own `assert N % block_size == 0`.
#[test]
fn act_quant_f8_refuses_a_ragged_row() {
    let mut b = zeros(300 * 4);
    let stream = HipStream::new().expect("stream");
    // SAFETY: the accepted case is 2 blocks of the 300-f32 buffer; the rejected one returns
    // before a dereference.
    let p = b.ptr_mut() as *mut f32;
    let go = |n_rows, row_len| unsafe { launch_act_quant_f8(p, n_rows, row_len, stream.raw()) };
    assert!(go(1, 256).is_ok(), "a whole-block row must be accepted");
    device_sync().expect("device sync"); // the accepted case launched — see the guard test
    let e = |r: anyhow::Result<()>| r.map_err(|x| format!("{x:#}"));
    assert_guards(vec![
        (1002, "row_len 1", e(go(1, 1))),
        (1002, "row_len 127", e(go(1, 127))),
        (1002, "row_len 129", e(go(1, 129))),
        (1002, "row_len 192", e(go(1, 192))),
        (1001, "zero rows", e(go(0, 128))),
        (1001, "zero-length row", e(go(1, 0))),
    ]);
}

// =======================================================================================
// 3. what the fixture actually exercises
// =======================================================================================

/// What `routed_experts_match_the_oracle` passing does and does not say about the decoders.
///
/// There is no standalone exhaustive codec probe here, and that is a decision rather than
/// an omission. The obvious one — a synthetic weight whose columns cycle all 16 e2m1 codes
/// and whose rows carry every e8m0 code — cannot be read back through this pipeline: the
/// e8m0 range spans 2^-127 to 2^127, while `moe_fixed`'s accumulator is faithful only over
/// roughly `[2^-21, 2^14]` (`common.hpp`, MOE_ACC_SHIFT). Rows outside that band would be
/// truncated or clamped by the ACCUMULATOR and the test would be measuring the wrong thing.
///
/// So the decoders are covered by the end-to-end comparison, over whatever code
/// distribution the fixture contains — and this test measures that distribution instead of
/// assuming it is broad. It is what turns "the expert test covers the decode" into a
/// bounded claim, and it can go red: shrink the toy's weight scale far enough and the e2m1
/// histogram collapses onto a handful of codes while every other test here still passes.
#[test]
fn the_fixture_exercises_the_codes_the_decoders_are_credited_with() {
    // The experts `Case` actually routes — NOT all `n_routed_experts`. An unrouted expert's
    // decode is annihilated inside the kernel (`wexpert == 0` gives `h == 0` exactly, which
    // `an_unrouted_expert_contributes_exactly_zero` proves), so counting its codes would
    // credit the comparison with coverage it cannot see. Found by review, 2026-08-05: the
    // first version of this counted 4x the codes the gate can observe. Host-only: it reads
    // weights, not results, so it builds no `Case` and touches no device.
    let (_, m, _) = fixture();
    let mut nibbles = [0usize; 16];
    let mut scales = std::collections::BTreeSet::new();
    for &e in &PICKS {
        for w in [&m.layers[0].experts[&e].w1, &m.layers[0].experts[&e].w3,
                  &m.layers[0].experts[&e].w2] {
            let (packed, s) = fp4_spans(w);
            for b in packed {
                nibbles[(b & 0x0f) as usize] += 1;
                nibbles[(b >> 4) as usize] += 1;
            }
            scales.extend(s.iter().copied());
        }
    }
    let missing: Vec<usize> = (0..16).filter(|&n| nibbles[n] == 0).collect();
    assert!(missing.is_empty(), "e2m1 codes never exercised by a ROUTED expert: {missing:?}");
    // Printed, so it needs `cargo test -- --nocapture` to read: the BOUND is what a reader
    // wants and any threshold on it here would be a number picked to pass. `e8m0f`'s two
    // special codes — 0x00 (2^-127, an f32 subnormal) and 0xff (NaN) — are decoded by
    // nothing that runs in this file, which the assertion below states rather than prints.
    println!(
        "e2m1 code counts: {nibbles:?}\ne8m0 codes present: {} distinct, {:?}..={:?}",
        scales.len(),
        scales.iter().next(),
        scales.iter().next_back()
    );
    assert!(!scales.contains(&0u8) && !scales.contains(&0xffu8), "a special e8m0 code leaked in");
}

// =======================================================================================
// 4. the router
// =======================================================================================

/// `Gate.forward`'s logits: `linear(x.float(), weight.float())`, the DENSE branch — no
/// activation quantization, no fp8.
///
/// Computed here rather than by `gemv_f32` so the comparison isolates the gate kernel from
/// GEMV re-association: a re-associated logit can flip a near-tie in the top-k, and this
/// test is about the selection rule, not about a reduction order `tests/kernel.rs` already
/// covers. Summed left to right, which is bit-identical to `Oracle::linear`.
fn gate_logits(lw: &LayerW, x: &[f32]) -> Vec<f32> {
    let (rows, cols, v) = match &lw.gate_w {
        WMat::Dense { rows, cols, v } => (*rows, *cols, v),
        WMat::Fp4 { .. } | WMat::Fp8 { .. } => panic!("gate.weight is dense in the reference"),
    };
    (0..rows)
        .map(|e| {
            let mut acc = 0.0f32;
            for (i, xi) in x.iter().enumerate().take(cols) {
                acc += xi * v[e * cols + i];
            }
            acc
        })
        .collect()
}

/// One `moe_gate_v4` dispatch, spelling the launcher's argument list ONCE.
///
/// Takes raw pointers rather than a `LayerW`, because the guard test hands it the two
/// bias/`tid2eid` combinations the reference never produces and needs them REJECTED — a
/// helper that derived them from a layer could not express an illegal one. Returns the
/// `Result` for the same reason.
#[allow(clippy::too_many_arguments)]
fn gate_call(
    cfg: &V4Config,
    logits: *const f32,
    bias: *const f32,
    tid2eid: *const i64,
    input_id: usize,
    w: &mut DeviceBuf,
    i: &mut DeviceBuf,
) -> Result<(), String> {
    let stream = HipStream::new().expect("stream");
    // SAFETY: `logits` and `bias` (when non-null) hold `n_routed_experts` f32, `tid2eid`
    // (when non-null) is `vocab_size · k` and covers `input_id`, and both outputs hold `k`.
    // Every rejected combination returns before a dereference; the sync below is inside the
    // buffers' lifetimes.
    unsafe {
        launch_moe_gate_v4(
            logits,
            bias,
            tid2eid,
            input_id,
            cfg.vocab_size,
            cfg.n_routed_experts,
            cfg.n_activated_experts,
            cfg.route_scale,
            w.ptr_mut() as *mut f32,
            i.ptr_mut() as *mut i32,
            stream.raw(),
        )
    }
    .map_err(|e| format!("{e:#}"))
}

/// Run `moe_gate_v4` for one token of a real layer. `input_id` is read only where the layer
/// routes by hash.
fn gpu_gate(cfg: &V4Config, lw: &LayerW, logits: &[f32], input_id: usize) -> (Vec<f32>, Vec<i32>) {
    let k = cfg.n_activated_experts;
    let lb = to_device(&f32b(logits));
    let bias = lw.gate_bias.as_ref().map(|b| to_device(&f32b(b)));
    let hash = lw
        .tid2eid
        .as_ref()
        .map(|t| to_device(&t.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()));
    let mut wb = zeros(k * 4);
    let mut ib = zeros(k * 4);
    gate_call(
        cfg,
        lb.ptr() as *const f32,
        bias.as_ref().map_or(std::ptr::null(), |b| b.ptr() as *const f32),
        hash.as_ref().map_or(std::ptr::null(), |t| t.ptr() as *const i64),
        input_id,
        &mut wb,
        &mut ib,
    )
    .expect("moe_gate_v4");
    device_sync().expect("device sync");
    let iv = ib.copy_out().expect("indices");
    (
        f32v(&wb.copy_out().expect("weights")),
        (0..k).map(|j| i32::from_le_bytes(iv[j * 4..j * 4 + 4].try_into().unwrap())).collect(),
    )
}

/// Both router modes against the oracle: a hash layer (0-2, `tid2eid`, no bias) and a
/// scored layer (3+, bias, top-k).
///
/// Indices are compared EXACTLY — they are a selection, and no numeric tolerance stands in
/// for "a different expert ran".
#[test]
fn the_router_matches_the_oracle_in_both_modes() {
    let (cfg, m, o) = fixture();
    for layer in [0usize, 3] {
        let lw = &m.layers[layer];
        let hash = lw.tid2eid.is_some();
        assert_eq!(hash, layer < cfg.n_hash_layers, "layer {layer} is not the mode expected");
        let x = draw_x(&format!("gate-x-{layer}"), cfg.dim, 1.0);
        let ids = [7u32];
        let step = Step { lw, layer, s: 1, start_pos: 0, input_ids: &ids, phase: "probe" };
        let (want_w, want_i) = o.gate(&step, &x, &mut Counters::default());

        let (got_w, got_i) = gpu_gate(cfg, lw, &gate_logits(lw, &x), ids[0] as usize);
        let want_i: Vec<i32> = want_i.iter().map(|&e| e as i32).collect();
        assert_eq!(want_i, got_i, "layer {layer}: expert selection");
        assert_matches(&want_w, &got_w, &format!("layer {layer} routing weights"));
    }
}

/// **A hash layer bypasses the scores for SELECTION ONLY.** The gate still runs and its
/// scores still become the WEIGHTS (`model.py:585`). Reading `tid2eid` and skipping the
/// gate is the silent-wrong this test exists for, and it is the failure the S2a brief names.
///
/// Both halves, because either alone proves nothing: perturbing the logits must move the
/// weights (so the scores are live) and must NOT move the indices (so the selection really
/// is the hash).
#[test]
fn hash_routing_bypasses_the_scores_for_selection_only() {
    let (cfg, m, _) = fixture();
    let lw = &m.layers[0];
    assert!(lw.tid2eid.is_some(), "layer 0 must be a hash layer for this to mean anything");
    let x = draw_x("gate-x-0", cfg.dim, 1.0);
    let base = gate_logits(lw, &x);
    // A shift, not a scale: `sqrt(softplus(·))` is monotone, so a positive scale would
    // leave the top-k ORDER intact and the "indices did not move" half would pass for a
    // score-routed kernel too. Reversing the sign reverses the ranking.
    let flipped: Vec<f32> = base.iter().map(|v| -v).collect();

    let (w0, i0) = gpu_gate(cfg, lw, &base, 7);
    let (w1, i1) = gpu_gate(cfg, lw, &flipped, 7);
    assert_eq!(i0, i1, "hash selection moved when only the scores changed");
    assert!(
        w0.iter().zip(&w1).any(|(a, b)| (a - b).abs() > 1e-4),
        "the routing weights did not move when the scores did — the gate is being skipped"
    );

    // And the indices ARE the table's, not a top-k of anything.
    let table = lw.tid2eid.as_ref().unwrap();
    let k = cfg.n_activated_experts;
    let want: Vec<i32> = table[7 * k..8 * k].iter().map(|&e| e as i32).collect();
    assert_eq!(want, i0, "the hash indices are not tid2eid[input_id]");
}

/// `torch.topk`'s tie-break: descending by value, ties to the LOWER index.
///
/// This is the ENTIRE reason `moe_gate_v4`'s selection is a serial k-pass argmax rather
/// than a tree reduction over `(value, index)` pairs, and nothing else here measures it —
/// the oracle comparison runs on random weights, where an exact tie has measure zero.
///
/// Expectations are stated directly rather than taken from the oracle: with equal scores the
/// answer is the first `k` expert ids, and that is a fact about `torch.topk`, not about
/// either implementation. A comparison against `Oracle::gate` here would only show the two
/// agreeing about a rule neither had been asked to demonstrate.
#[test]
fn the_router_breaks_ties_towards_the_lower_expert_id() {
    let (cfg, ..) = fixture();
    let k = cfg.n_activated_experts;
    let zero_bias = to_device(&f32b(&vec![0.0f32; cfg.n_routed_experts]));
    let mut wb = zeros(k * 4);
    let mut ib = zeros(k * 4);
    let mut pick = |logits: &[f32]| {
        let lb = to_device(&f32b(logits));
        // A scored layer (bias, no table), because ties only matter where a top-k runs.
        gate_call(cfg, lb.ptr() as *const f32, zero_bias.ptr() as *const f32, std::ptr::null(),
                  0, &mut wb, &mut ib)
            .expect("moe_gate_v4");
        device_sync().expect("device sync");
        let iv = ib.copy_out().expect("indices");
        (0..k).map(|j| i32::from_le_bytes(iv[j * 4..j * 4 + 4].try_into().unwrap()))
            .collect::<Vec<i32>>()
    };

    // Every score identical: the k lowest ids, in ascending order.
    let all_tied: Vec<i32> = (0..k as i32).collect();
    assert_eq!(all_tied, pick(&vec![1.0f32; cfg.n_routed_experts]),
               "an all-tied row did not select the k lowest expert ids");

    // A tied PAIR at the top, below a run of lower scores: the two highest-scoring ids, and
    // the lower one first. Placed at the END of the row so a scan that kept the LAST maximum
    // (`>=` instead of `>`) would return them in the other order.
    let mut logits = vec![0.25f32; cfg.n_routed_experts];
    let (a, b) = (cfg.n_routed_experts - 2, cfg.n_routed_experts - 1);
    logits[a] = 4.0;
    logits[b] = 4.0;
    let got = pick(&logits);
    assert_eq!(vec![a as i32, b as i32], got[..2].to_vec(),
               "a tied pair did not come back lower-id first");
}

/// Every router guard, by CODE.
///
/// The two bias/`tid2eid` combinations the reference never produces are refused rather than
/// resolved by precedence inside the kernel: they are exactly "route a hash layer by score"
/// and "let the selection bias reach the weights".
///
/// `k > n_experts` matters more than it looks — the kernel's masking argmax sets each pick
/// to `-INFINITY`, so a `(k+1)`-th pass over an all-masked row hands back a DUPLICATED
/// index 0 rather than failing. And `input_id` past `tid2eid`'s rows is the overrun that can
/// actually happen; the guard here used to check `input_id < 0`, which the Rust launcher's
/// `usize` makes unreachable.
#[test]
fn the_router_refuses_what_it_claims_to() {
    let (cfg, ..) = fixture();
    let logits = to_device(&f32b(&vec![0.0f32; cfg.n_routed_experts]));
    let bias = to_device(&f32b(&vec![0.0f32; cfg.n_routed_experts]));
    let table = to_device(&vec![0u8; cfg.vocab_size * cfg.n_activated_experts * 8]);
    let mut wb = zeros(cfg.n_activated_experts * 4);
    let mut ib = zeros(cfg.n_activated_experts * 4);
    let (bp, tp) = (bias.ptr() as *const f32, table.ptr() as *const i64);
    let lp = logits.ptr() as *const f32;
    let mut go = |c, b, t, id| gate_call(c, lp, b, t, id, &mut wb, &mut ib);

    assert!(go(cfg, bp, std::ptr::null(), 0).is_ok(), "a scored layer must be accepted");
    assert!(go(cfg, std::ptr::null(), tp, 0).is_ok(), "a hash layer must be accepted");
    device_sync().expect("device sync"); // both accepted cases launched

    // `k > n_experts` needs a config the toy cannot supply, since `V4Config` pins k < n.
    let big = V4Config { n_activated_experts: cfg.n_routed_experts + 1, ..cfg.clone() };
    assert_guards(vec![
        (1002, "both a bias and a hash table", go(cfg, bp, tp, 0)),
        (1002, "neither", go(cfg, std::ptr::null(), std::ptr::null(), 0)),
        (1003, "input_id past the table", go(cfg, std::ptr::null(), tp, cfg.vocab_size)),
        (1001, "k > n_experts", go(&big, bp, std::ptr::null(), 0)),
    ]);
}

// =======================================================================================
// 5. mHC
// =======================================================================================

/// `hc_pre` for one sublayer, returning `(y, post, comb)` device-side readbacks.
fn gpu_hc_pre(
    cfg: &V4Config,
    h: &[f32],
    fnw: &[f32],
    scale: &[f32],
    base: &[f32],
    s: usize,
    iters: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let stream = HipStream::new().expect("stream");
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let (hb, fb, sb, bb) = (to_device(&f32b(h)), to_device(&f32b(fnw)), to_device(&f32b(scale)), to_device(&f32b(base)));
    let mut y = zeros(s * dim * 4);
    let mut post = zeros(s * hc * 4);
    let mut comb = zeros(s * hc * hc * 4);
    // SAFETY: every buffer is sized for (s, hc, dim) above.
    unsafe {
        launch_hc_pre(
            hb.ptr() as *const f32,
            fb.ptr() as *const f32,
            sb.ptr() as *const f32,
            bb.ptr() as *const f32,
            s,
            hc,
            dim,
            iters,
            cfg.norm_eps,
            cfg.hc_eps,
            y.ptr_mut() as *mut f32,
            post.ptr_mut() as *mut f32,
            comb.ptr_mut() as *mut f32,
            stream.raw(),
        )
    }
    .expect("hc_pre");
    (sync_f32(&y), sync_f32(&post), sync_f32(&comb))
}

/// `hc_post` over `s` tokens.
fn gpu_hc_post(
    cfg: &V4Config,
    x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    s: usize,
) -> Vec<f32> {
    let stream = HipStream::new().expect("stream");
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let (xb, rb) = (to_device(&f32b(x)), to_device(&f32b(residual)));
    let (pb, cb) = (to_device(&f32b(post)), to_device(&f32b(comb)));
    let mut y = zeros(s * hc * dim * 4);
    // SAFETY: sized for (s, hc, dim) above.
    unsafe {
        launch_hc_post(
            xb.ptr() as *const f32,
            rb.ptr() as *const f32,
            pb.ptr() as *const f32,
            cb.ptr() as *const f32,
            s,
            hc,
            dim,
            y.ptr_mut() as *mut f32,
            stream.raw(),
        )
    }
    .expect("hc_post");
    sync_f32(&y)
}

/// `rmsnorm` on the GPU, so the `hc_pre` comparison lands on a golden the oracle emits.
///
/// **rivoli's `rmsnorm` kernel does NOT bf16-round its output, and V4's `RMSNorm.forward`
/// returns bf16** (`model.py:197-202` computes in f32 and the module's dtype is bf16). That
/// is a real gap and it is NOT this stream's to close — `rmsnorm` is shared with the GLM
/// path, where adding a store would change shipped output. `mhc_reproduces_the_layer_
/// goldens` applies the missing round on the host and PRINTS what it was worth, so the
/// number is on the record rather than absorbed into a tolerance. **S3 owns supplying it.**
fn gpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let dim = w.len();
    assert!(x.len().is_multiple_of(dim), "x must be whole rows of the norm weight");
    let (xb, wb) = (to_device(&f32b(x)), to_device(&f32b(w)));
    let mut y = zeros(x.len() * 4);
    // ONE LAUNCH PER TOKEN. `rivoli_rmsnorm` is single-row — `dim3(1)`, one mean over its
    // whole `n`, and `w[i]` indexed over that same `n`. Handing it `s·dim` took a JOINT rms
    // over every token (the oracle's is per token, `x.chunks_mut(d)`) and read the norm
    // weight `s-1` rows past its allocation. Both were silent: the golden's length still
    // matched, so `compare` was happy, and the arithmetic error rode in as a plausible
    // scale. Found by review, 2026-08-05.
    for t in 0..x.len() / dim {
        // SAFETY: row `t` is `dim` live f32 inside both buffers, and `w` is `dim` long.
        unsafe {
            launch_rmsnorm(
                (xb.ptr() as *const f32).add(t * dim),
                wb.ptr() as *const f32,
                dim,
                eps,
                (y.ptr_mut() as *mut f32).add(t * dim),
            )
        }
        .expect("rmsnorm");
    }
    sync_f32(&y)
}

/// One prefill `run_layer` capture, for the tests that score against goldens rather than
/// against a re-derivation.
fn capture(layer: usize, s: usize) -> (Capture, Vec<f32>, Vec<u32>) {
    let (cfg, m, o) = fixture();
    let mut r = NamedRng::new("hc-h");
    let mut h: Vec<f32> = (0..s * cfg.hc_mult * cfg.dim)
        .map(|_| bf16_decode(bf16_encode(r.unit())))
        .collect();
    let mut ri = NamedRng::new("hc-ids");
    let ids: Vec<u32> = (0..s).map(|_| ri.below(cfg.vocab_size) as u32).collect();
    let h0 = h.clone();
    let mut cap = Capture::default();
    let mut st = o.fresh_state(layer);
    let step = Step { lw: &m.layers[layer], layer, s, start_pos: 0, input_ids: &ids, phase: "pre" };
    o.run_layer(&step, &mut st, &mut h, &mut cap);
    (cap, h0, ids)
}

/// mHC end to end, against `run_layer`'s own goldens.
///
/// Neither `hc_pre`'s reduction nor `hc_post`'s expansion is captured on its own — the
/// oracle records `.in`, `.attn_norm_out`, `.attn_out`, `.ffn_norm_out`, `.ffn_out` and
/// `.out`. So the two are gated in the CHAIN that connects them:
///
/// ```text
///   .in --hc_pre--> --rmsnorm--> .attn_norm_out          (gates hc_pre)
///   (.attn_out, .in) --hc_post--> h1
///   h1 --hc_pre--> --rmsnorm--> .ffn_norm_out            (gates hc_post, through hc_pre)
///   (.ffn_out, h1) --hc_post--> .out                     (gates hc_post directly)
/// ```
///
/// Every input is a golden or a layer weight, and the last line closes the loop: if the
/// mHC halves were both wrong in compensating ways, `.ffn_norm_out` could still match, but
/// `.out` is `hc_post`'s own output and cannot.
#[test]
fn mhc_reproduces_the_layer_goldens() {
    let (cfg, m, _) = fixture();
    const S: usize = 3;
    let (cap, h_in, _) = capture(0, S);
    let lw = &m.layers[0];
    let iters = cfg.hc_sinkhorn_iters;
    let g = |n: &str| cap.float(n).unwrap_or_else(|| panic!("golden {n}")).to_vec();

    assert_eq!(h_in, g("L0.pre.in"), "the driver's h is not what the oracle recorded");

    // The bf16 store `rmsnorm` is missing (see `gpu_rmsnorm`). Measured on the way past so
    // the size of the gap is recorded rather than inferred: `report` prints the unrounded
    // error next to the rounded one, and if the two are ever the same number the missing
    // store has stopped mattering and this wrapper can go.
    let norm = |v: &[f32], w: &[f32], label: &str| {
        let raw = gpu_rmsnorm(v, w, cfg.norm_eps);
        let rounded: Vec<f32> = raw.iter().map(|x| bf16_decode(bf16_encode(*x))).collect();
        // Asserted, not merely printed: `println!` is captured and discarded on a green run,
        // so "the number is on the record" would be a claim about output nobody sees. This
        // goes red exactly when the wrapper stops being needed.
        let (err, _) = compare(&raw, &rounded, &format!("{label}: rmsnorm's missing bf16 store"));
        assert!(
            err > 0.0,
            "{label}: rmsnorm's output is already bf16-representable — the missing store has \
             stopped mattering, so drop `norm` and call `gpu_rmsnorm` directly"
        );
        rounded
    };

    let (y, post, comb) =
        gpu_hc_pre(cfg, &h_in, &lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base, S, iters);
    assert_matches(&g("L0.pre.attn_norm_out"), &norm(&y, &lw.attn_norm, "attn"),
                 "hc_pre(attn) then rmsnorm");

    let h1 = gpu_hc_post(cfg, &g("L0.pre.attn_out"), &h_in, &post, &comb, S);
    let (y2, post2, comb2) =
        gpu_hc_pre(cfg, &h1, &lw.hc_ffn_fn, &lw.hc_ffn_scale, &lw.hc_ffn_base, S, iters);
    assert_matches(&g("L0.pre.ffn_norm_out"), &norm(&y2, &lw.ffn_norm, "ffn"),
                 "hc_post(attn) then hc_pre(ffn) then rmsnorm");

    let out = gpu_hc_post(cfg, &g("L0.pre.ffn_out"), &h1, &post2, &comb2, S);
    assert_matches(&g("L0.pre.out"), &out, "hc_post(ffn)");
}

/// Every mHC guard, by CODE.
///
/// `V4Config` already pins `hc_mult = 4`, so on the shipping path `hc != HC_MULT` can only
/// ever pass — and it is the guard `launch_hc_pre`'s doc leans on hardest, since
/// `mix_hc = (2+hc)·hc` is how the mHC weights are PACKED and this check is all that stands
/// between a foreign checkpoint and a wrong-stride read of `fnw`. Only a test that hands it
/// a wrong value separates "correct" from "unreachable".
#[test]
fn the_mhc_launchers_refuse_what_they_claim_to() {
    let (cfg, m, _) = fixture();
    let lw = &m.layers[0];
    let stream = HipStream::new().expect("stream");
    let h = zeros(cfg.hc_mult * cfg.dim * 4);
    let (f, sc, b) = (to_device(&f32b(&lw.hc_attn_fn)), to_device(&f32b(&lw.hc_attn_scale)),
                      to_device(&f32b(&lw.hc_attn_base)));
    // Sized for the ACCEPTED case each launcher runs, which is the half of a guard test
    // that actually touches memory: `hc_pre` writes `s·dim`, `hc_post` writes `s·hc·dim`.
    // A single shared output buffer sized for `hc_pre` would let `hc_post`'s accepted arm
    // overrun it by `hc`x — the first draft of this test did exactly that.
    let mut y = zeros(cfg.dim * 4);
    let mut expanded = zeros(cfg.hc_mult * cfg.dim * 4);
    let mut post = zeros(cfg.hc_mult * 4);
    let mut comb = zeros(cfg.hc_mult * cfg.hc_mult * 4);

    // Addresses hoisted so the two closures below vary only their guarded arguments.
    let (hp, fp, scp, bp_) = (h.ptr() as *const f32, f.ptr() as *const f32,
                              sc.ptr() as *const f32, b.ptr() as *const f32);
    let (yp, pp, cp) = (y.ptr_mut() as *mut f32, post.ptr_mut() as *mut f32,
                        comb.ptr_mut() as *mut f32);
    let ep = expanded.ptr_mut() as *mut f32;
    let pre = |s, hc, iters| {
        // SAFETY: every rejected case returns before a dereference, and the accepted one is
        // sized by the buffers above; all of them outlive the sync that follows it.
        unsafe {
            launch_hc_pre(hp, fp, scp, bp_, s, hc, cfg.dim, iters, cfg.norm_eps, cfg.hc_eps, yp,
                          pp, cp, stream.raw())
        }
        .map_err(|e| format!("{e:#}"))
    };
    let (hc, it) = (cfg.hc_mult, cfg.hc_sinkhorn_iters);
    assert!(pre(1, hc, it).is_ok(), "the accepted case must be accepted");
    device_sync().expect("device sync"); // the accepted case launched
    assert_guards(vec![
        (1001, "hc_pre zero tokens", pre(0, hc, it)),
        (1002, "hc_pre hc_mult 3", pre(1, 3, it)),
        (1002, "hc_pre hc_mult 8", pre(1, 8, it)),
        // A Sinkhorn that runs the leading column normalisation and no pairs at all.
        (1003, "hc_pre zero iterations", pre(1, hc, 0)),
    ]);

    let post_call = |s, hc| {
        // SAFETY: same — rejected before a dereference, accepted within the buffers. `y`
        // (`dim`) is the sublayer output and `expanded` (`hc·dim`) the destination; they are
        // DISTINCT, because `hc_post` reads `x[tok·dim+d]` while writing every copy.
        unsafe { launch_hc_post(yp, hp, pp, cp, s, hc, cfg.dim, ep, stream.raw()) }
            .map_err(|e| format!("{e:#}"))
    };
    assert!(post_call(1, hc).is_ok(), "the accepted case must be accepted");
    device_sync().expect("device sync");
    assert_guards(vec![
        (1001, "hc_post zero tokens", post_call(0, hc)),
        (1002, "hc_post hc_mult 2", post_call(1, 2)),
    ]);

}

/// The Sinkhorn iteration count reaches the arithmetic.
///
/// **This is NOT a check that the count is 20**, and it cannot be: at 20 passes the 4x4
/// matrix has converged and 19 and 20 agree bit-for-bit — the oracle's own matrix excludes
/// `Defect::SinkhornOneFewerIter` for that measured reason. What is provable is that the
/// parameter is live rather than ignored, which is what makes SOURCING it from `V4Config`
/// (and `V4Config::assert_matches_reference_json` pinning that to `config.json`) the actual
/// gate on the value.
#[test]
fn sinkhorn_iteration_count_is_live() {
    let (cfg, m, _) = fixture();
    const S: usize = 2;
    let lw = &m.layers[0];
    let mut r = NamedRng::new("sink-h");
    let h: Vec<f32> = (0..S * cfg.hc_mult * cfg.dim)
        .map(|_| bf16_decode(bf16_encode(r.unit())))
        .collect();
    assert!(cfg.hc_sinkhorn_iters >= 2, "this test subtracts one below");
    let run = |iters| {
        gpu_hc_pre(cfg, &h, &lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base, S, iters)
    };
    let (_, _, c20) = run(cfg.hc_sinkhorn_iters);
    let (_, _, c2) = run(2);
    let (_, _, c19) = run(cfg.hc_sinkhorn_iters - 1);
    // Bit-inequality, matching the claim the oracle's own test makes rather than a
    // threshold picked here: `sinkhorn_has_converged_long_before_iteration_20` asserts
    // `!identical(20, 2)`, and a magnitude threshold would be a weaker statement that could
    // pass for a kernel whose `iters` only tickled the low bits.
    assert_ne!(bits(&c20), bits(&c2),
               "2 and {} iterations agree — `iters` never reaches the kernel",
               cfg.hc_sinkhorn_iters);
    // The blind spot itself, asserted in the direction the oracle asserts it. If this ever
    // goes red, convergence has stopped holding on the GPU where it holds on the CPU — which
    // would mean `SinkhornOneFewerIter` becomes gateable AND that the two arithmetics have
    // diverged somewhere worth finding.
    assert_eq!(bits(&c20), bits(&c19),
               "19 and 20 iterations disagree on the GPU where they agree on the CPU");
}

// =======================================================================================
// 6. the MoE layer, end to end against a golden
// =======================================================================================

/// `.ffn_out` reproduced from `.ffn_norm_out` — the router kernel choosing the experts, the
/// FP4 kernels running them, and the SHARED expert filled in from the oracle.
///
/// The shared expert is fp8 e4m3 at 128x128, not FP4, and is explicitly out of S2a. It is
/// computed here by `Oracle::expert` so the comparison can reach a real golden at all; the
/// consequence is that **this test says nothing about rivoli's fp8 path**, and a defect
/// there would be invisible to it.
#[test]
fn ffn_out_matches_the_golden() {
    let (cfg, m, o) = fixture();
    let layer = 0usize;
    let lw = &m.layers[layer];
    let (cap, _, ids) = capture(layer, 1);
    let x = cap.float("L0.pre.ffn_norm_out").expect("ffn_norm_out golden").to_vec();

    let (gw, gi) = gpu_gate(cfg, lw, &gate_logits(lw, &x), ids[0] as usize);
    // The router's own goldens, exactly — a wrong selection here would otherwise show up
    // as a numeric miss at the end and be indistinguishable from an arithmetic bug.
    let want_i: Vec<i32> =
        cap.int("L0.pre.router_indices").expect("router_indices").iter().map(|&e| e as i32).collect();
    assert_eq!(want_i, gi, "router indices");
    assert_matches(cap.float("L0.pre.router_weights").expect("router_weights"), &gw,
                 "router weights");

    // ONE weight per expert, which is what `moe_gateup_f4`'s `wexpert` layout can express
    // — so a token routed TWICE to one expert would be unrepresentable, and the reference
    // does not fold it either (`MoE.forward` groups by `where(indices == i)` and runs the
    // expert once per PICK, bf16-rounding each pass separately).
    //
    // Measured 2026-08-05: 0 of 129,280 rows of the real `layers.0.ffn.gate.tid2eid` names
    // an expert twice, and `torch.topk` cannot. So the constraint is satisfied by the
    // checkpoint rather than by luck — but it is a CONSTRAINT, and S3 must assert it at
    // load rather than inherit this assumption silently.
    let mut wexpert = vec![0.0f32; cfg.n_routed_experts];
    for (j, &e) in gi.iter().enumerate() {
        assert_eq!(wexpert[e as usize], 0.0, "expert {e} was picked twice for one token");
        wexpert[e as usize] = gw[j];
    }
    let experts: Vec<&ExpertW> = (0..cfg.n_routed_experts).map(|e| &lw.experts[&e]).collect();
    let e = F4Experts::upload(&experts, Wiring::Correct);
    let routed = Dispatch { cfg, experts: &e, x: &x, wexpert: &wexpert,
                            swiglu_limit: cfg.swiglu_limit, quantize_x: true }
        .run();

    let shared = o.expert(&lw.shared, &x, 1, None, &mut Counters::default());
    let got: Vec<f32> = routed
        .iter()
        .zip(&shared)
        .map(|(a, b)| bf16_decode(bf16_encode(a + b)))
        .collect();
    assert_matches(cap.float("L0.pre.ffn_out").expect("ffn_out golden"), &got, "ffn_out");
}

// =======================================================================================
// shared assertions
// =======================================================================================

/// The tolerance every numerical comparison in this file uses, RELATIVE to the largest
/// element of the expectation.
///
/// Not `common::err_tol`, and the reason is arithmetic rather than taste: that formula is
/// `1e-3·max + 1e-3`, and its ABSOLUTE floor dominates at this fixture's scale — one routed
/// MoE layer's output on the toy weights is about 2e-2, so a 1e-3 floor is 5% of the
/// signal. A gate that loose would accept most of the defects this file exists to find, and
/// would weaken the deliberate-break tests by the same factor, since they must EXCEED it.
///
/// `2^-7` is two bf16 ulps. The reference stores bf16 at every step (each GEMM output, the
/// weighted SwiGLU intermediate, each expert's output), so the answer is quantized to 2^-8
/// relative and any upstream difference — the wave reduction's summation order against the
/// oracle's sequential sum, `expf` against Rust's `exp` — flips an element by a whole ulp
/// rather than by its own tiny magnitude. One ulp is the floor; two is the margin for a
/// flip in `h` that then propagates through `w2`.
const TOL: f32 = 1.0 / 128.0;

/// `(max abs error, tolerance)`, printed with the margin.
fn compare(want: &[f32], got: &[f32], label: &str) -> (f32, f32) {
    assert_eq!(want.len(), got.len(), "{label}: length mismatch");
    report_rel(want, got, label, TOL)
}

/// The two must agree within [`TOL`].
fn assert_matches(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = compare(want, got, label);
    assert!(err <= tol, "{label}: err={err:.3e} > tol={tol:.3e}");
}

/// Assert each launcher `Result` carries the guard CODE it is paired with.
///
/// The code, not `is_err`: a check that accepted any error would still pass if someone
/// replaced a power-of-two test with `block != 128`, or if an unrelated dimension guard
/// started swallowing the case first. Three guard tests share this, which is also what stops
/// three copies of the message format from drifting.
fn assert_guards(cases: Vec<(u32, &str, Result<(), String>)>) {
    for (want, case, r) in cases {
        let msg = r.expect_err(case);
        assert!(msg.contains(&want.to_string()), "{case}: want guard {want}, got {msg:?}");
    }
}

/// The negative of [`assert_matches`]: the two must NOT agree, at the SAME tolerance.
///
/// Every deliberate-break test in this file goes through here rather than through a bare
/// `assert_ne!`, and the shared threshold is what makes the pair meaningful: a break that
/// moved the result by less than [`TOL`] is a break the positive gate would NOT have
/// caught, so it must fail here rather than pass. The margin printed by [`compare`] is how
/// far from that line it actually landed.
fn assert_disagrees(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = compare(want, got, &format!("{label} (must differ)"));
    assert!(
        err > tol,
        "{label}: err={err:.3e} <= tol={tol:.3e} — the break is INVISIBLE to this gate, so \
         the corresponding positive test proves nothing about it"
    );
}
