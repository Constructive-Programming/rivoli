//! The V4 routed-FP4 dispatch harness, shared by `kernel_v4_moe.rs` (the expert comparison and
//! its deliberate breaks) and `kernel_v4_fp4_decode.rs` (what the fixture's CODES actually cover).
//!
//! One module because the two binaries drive the SAME dispatch: every number either reports comes
//! out of [`Dispatch::run`] over an [`F4Experts`] upload, and a second copy of either would be a
//! second launch sequence that could drift from the engine's while both stayed green.
//! `v4_compressor/mod.rs` and `glimmer_anchor/mod.rs` beside this are the precedent for the shape.
//!
//! Split out of `kernel_v4_moe.rs` on 2026-08-16 under the 800-line soft cap, by COHESION: the
//! upload, the dispatch, the case builder and the two shared assertions are the instrument; the
//! two files next door are the questions asked of it. Bodies and their comments travelled
//! verbatim.

#![allow(dead_code)] // this module compiles into BOTH binaries and neither names every item

use rivoli_backend::hip::{ExpertDescF4, device_sync, launch_moe_expert_range_f4};
use rivoli_engine::device::DeviceBuf;
use rivoli_oracles::v4oracle::forward::{Counters, ExpertOperand, ExpertW};
use rivoli_oracles::v4oracle::weights::{V4Config, WMat, fixed_bf16};

use super::common::moe::{Dims, Drain, desc_buf, drain, moe_bufs};
use super::common::{
    Toy, Want, assert_rel, assert_separates, dev, f32b, f32v, max_abs, stream, toy_fixture,
};

// =======================================================================================
// fixture
// =======================================================================================

/// `WAVE * 8`, one dword-loop iteration of `dot_f4_wave_r` — a kernel constant this file
/// cannot see.
pub const F4_COLS_PER_TRIP: usize = 256;

/// The tolerance every numerical comparison in this file uses, RELATIVE to the largest element
/// of the expectation.
///
/// Not `common::assert_close`, and the reason is arithmetic rather than taste: that formula is
/// `1e-3·max + 1e-3`, and its ABSOLUTE floor dominates at this fixture's scale — one routed MoE
/// layer's output on the toy weights is about 2e-2, so a 1e-3 floor is 5% of the signal. A gate
/// that loose would accept most of the defects this file exists to find, and would weaken the
/// deliberate-break tests by the same factor, since they must EXCEED it.
///
/// `2^-7` is two bf16 ulps. The reference stores bf16 at every step (each GEMM output, the
/// weighted SwiGLU intermediate, each expert's output), so the answer is quantized to `2^-8`
/// relative and any upstream difference — the wave reduction's summation order against the
/// oracle's sequential sum, `expf` against Rust's `exp` — flips an element by a whole ulp
/// rather than by its own tiny magnitude. One ulp is the floor; two is the margin for a flip in
/// `h` that then propagates through `w2`.
pub const TOL: f32 = 1.0 / 128.0;

/// The bound `moe_fixed` CLAMPS each expert contribution at, halved.
///
/// `MOE_ACC_MAX = 2^(58-44) = 16384` (`kernels/common.hpp`) and the oracle has no such clamp,
/// so a case driven near it fails for a reason unrelated to what the test is asking. Every
/// fixture here is asserted to stay under half of it.
pub const ACC_HEADROOM: f32 = 8192.0;

/// The packed nibbles and e8m0 scale bytes of an FP4 weight, as the checkpoint stores them and
/// as `dot_f4_wave_r` reads them — the SAME bytes the oracle's `WMat::Fp4::row` decodes.
///
/// That is the whole design of this comparison: one byte array, two independently written
/// decoders, and any disagreement is a decoder bug rather than a fixture artefact.
pub fn fp4_spans(m: &WMat) -> (&[u8], &[u8]) {
    match m {
        WMat::Fp4 { w, s, .. } => (w, s),
        WMat::Dense { .. } | WMat::Fp8 { .. } => panic!("expected an fp4 weight"),
    }
}

/// The device-side FP4 expert set: one `ExpertDescF4` per routed expert, plus the buffers that
/// keep the six spans alive.
///
/// `parts` is what stops the descriptors from dangling — an `ExpertDescF4` is six raw addresses
/// and owns nothing.
pub struct F4Experts {
    pub descs: DeviceBuf,
    pub n: usize,
    #[allow(dead_code)]
    pub parts: Vec<DeviceBuf>,
}

/// Which projection slot a `WMat` is uploaded into.
///
/// Named because the whole point of the `w1`/`w3` tests is that the two are the same SHAPE, so
/// nothing but the name distinguishes them and a swap is invisible to every structural check.
/// The two knobs the deliberate-break tests turn: the SwiGLU clamp and whether the activation was
/// fp8-quantized in front of the GEMM.
///
/// A pair rather than two trailing arguments, and the reason is the call site: `broken(w, 1e6,
/// true)` says nothing about which `true` that is, while `Knobs { quantize_x: false, ..reference }`
/// names the one thing the break changes and leaves the other visibly untouched.
#[derive(Clone, Copy)]
pub struct Knobs {
    pub swiglu_limit: f32,
    /// Quantize `x` to fp8 at block 128 first, as `model.py::linear` does. `false` is a deliberate
    /// break with exactly one caller.
    pub quantize_x: bool,
}

impl Knobs {
    /// The reference dispatch's knobs: the config's own clamp, activation quantized.
    pub fn reference(cfg: &V4Config) -> Self {
        Self {
            swiglu_limit: cfg.swiglu_limit,
            quantize_x: true,
        }
    }
}

/// One deliberate break: which wiring the weights were uploaded under, and which knobs the
/// dispatch ran with. Bundled so a caller states BOTH halves of what it changed, at one place.
#[derive(Clone, Copy)]
pub struct Break {
    pub wiring: Wiring,
    pub knobs: Knobs,
}

/// What a [`Case`] is drawn from: which layer, and the tag and amplitude of its activation.
///
/// A struct because `layer` is a bare `usize` beside a `&str` and an `f32` — and the amplitude is
/// the knob that decides whether the fixture reaches the SwiGLU clamp at all, which is the one
/// number in the list a reader has to find.
#[derive(Clone, Copy)]
pub struct CaseSpec<'a> {
    pub layer: usize,
    pub tag: &'a str,
    /// 1.0 stays inside the clamp; large values cross it.
    pub scale: f32,
}

#[derive(Clone, Copy)]
pub enum Wiring {
    /// `gate = w1, up = w3` — the reference (`Expert.forward`: `gate = self.w1(x)`).
    Correct,
    /// `gate = w3, up = w1`. Same shapes, same byte counts, same scale grids.
    SwapGateUp,
    /// The reference wiring with every weight byte's nibbles exchanged — a permutation INSIDE
    /// each 32-element scale group, so the group boundaries, the amax/scale relation and the
    /// code histogram are all invariant under it.
    SwapNibbles,
}

impl F4Experts {
    pub fn upload(experts: &[&ExpertW], wiring: Wiring) -> Self {
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
                // The nibble swap inline rather than behind a helper: it is one `rotate_left(4)`
                // per byte, and the whole content of `Wiring::SwapNibbles` is that this is a
                // permutation INSIDE each 32-element scale group, which the expression says and a
                // name would hide.
                let wb = match swap_nibbles {
                    true => dev(&w.iter().map(|b| b.rotate_left(4)).collect::<Vec<u8>>()),
                    false => dev(w),
                };
                let sb = dev(sc);
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
        Self {
            descs: desc_buf(&descs),
            n: descs.len(),
            parts,
        }
    }
}

// =======================================================================================
// the routed-expert half, on the GPU
// =======================================================================================

/// One FP4 MoE dispatch: the weights, the activation, the picks, and the two knobs the
/// deliberate-break tests turn.
///
/// A struct rather than seven positional parameters: the two entry points below take the
/// identical list and differ only in how the expert range is cut, so writing it twice would be
/// two places for `hidden` and `inter` to get swapped.
pub struct Dispatch<'a> {
    pub dims: Dims,
    pub experts: &'a F4Experts,
    pub x: &'a [f32],
    /// One weight per UPLOADED expert, by ABSOLUTE id. `0.0` means this token did not route
    /// there — the kernel still runs the expert and adds exactly zero.
    pub wexpert: &'a [f32],
    pub knobs: Knobs,
}

impl<'a> Dispatch<'a> {
    /// The reference dispatch — the config's own clamp, activation quantized. For the callers
    /// with nothing to break; [`Case::dispatch`] keeps the two knobs overridable because the
    /// deliberate-break tests exist to turn them.
    pub fn reference(
        cfg: &V4Config,
        experts: &'a F4Experts,
        x: &'a [f32],
        wexpert: &'a [f32],
    ) -> Self {
        Dispatch {
            dims: Dims::new(cfg.dim, cfg.moe_inter_dim),
            experts,
            x,
            wexpert,
            knobs: Knobs::reference(cfg),
        }
    }

    /// `Σ_e down_e(silu(clamp(w1_e·x)) ⊙ clamp(w3_e·x) · weight_e)` — the routed half of
    /// `MoE.forward`, without the shared expert and without the final bf16 store. Every expert
    /// in ONE range.
    pub fn run(&self) -> Vec<f32> {
        self.in_ranges(&[(0, self.experts.n)])
    }

    /// The same, dispatched as several `[e_start, e_start + e_count)` ranges into one
    /// accumulator — the shape a two-stream engine will use, and the only thing here that
    /// exercises `e_start > 0`.
    pub fn in_ranges(&self, ranges: &[(usize, usize)]) -> Vec<f32> {
        let (hidden, inter) = (self.dims.hidden, self.dims.inter);
        assert_eq!(
            self.wexpert.len(),
            self.experts.n,
            "one routing weight per expert"
        );
        let stream = stream();
        let mut xb = dev(&f32b(self.x));
        let wb = dev(&f32b(self.wexpert));
        let (mut hb, mut ab, mut ob) = moe_bufs(self.experts.n, 1, self.dims);
        if self.knobs.quantize_x {
            // SAFETY: `xb` is `hidden` live f32 and outlives the stream's completion below.
            unsafe {
                rivoli_backend::hip::launch_act_quant_f8(
                    xb.ptr_mut() as *mut f32,
                    1,
                    hidden,
                    stream.raw(),
                )
            }
            .expect("act_quant_f8");
        }
        let (xp, wp) = (xb.ptr() as *const f32, wb.ptr() as *const f32);
        let dp = self.experts.descs.ptr() as *const ExpertDescF4;
        let (hp, ap) = (hb.ptr_mut() as *mut f32, ab.ptr_mut() as *mut u64);
        // SAFETY: every buffer above is sized for `experts.n` ABSOLUTE expert slots by
        // `moe_bufs` and is alive until the sync; every range is inside that bound.
        unsafe {
            for &(e_start, e_count) in ranges {
                launch_moe_expert_range_f4(
                    xp,
                    hidden,
                    inter,
                    e_start,
                    e_count,
                    self.experts.n,
                    dp,
                    wp,
                    self.knobs.swiglu_limit,
                    hp,
                    ap,
                    1,
                    stream.raw(),
                )
                .expect("moe_expert_range_f4");
            }
        }
        // Same stream, so every range's atomics precede the drain.
        drain(Drain::new(&mut ob, &mut ab), 0, hidden, &stream);
        device_sync().expect("device sync");
        f32v(&ob.copy_out().expect("device to host"))
    }
}

/// The two experts every [`Case`] routes.
///
/// A constant rather than a `Case` field:
/// [`the_fixture_exercises_the_codes_the_decoders_are_credited_with`] is a host-only histogram
/// over those experts' weights, and reading them off a `Case` would make it build device
/// buffers and run an oracle expert pass to learn two integers.
pub const PICKS: [usize; 2] = [1, 5];

/// One expert-comparison case: a layer, an activation, a fixed set of picks, and the oracle's
/// answer for them.
///
/// Built once per test so the deliberate-break tests differ ONLY in the thing they break — the
/// wiring, the SwiGLU limit, or whether `x` was quantized. Four hand-rolled setups would be
/// four chances for a break test to accidentally change the fixture too, which would make its
/// disagreement prove nothing.
pub struct Case {
    pub cfg: &'static V4Config,
    /// This case's routed experts, in ABSOLUTE expert order. Held so [`Case::broken`]
    /// re-uploads the SAME weights under a different wiring rather than re-deriving which layer
    /// they came from.
    pub all: Vec<&'static ExpertW>,
    pub x: Vec<f32>,
    /// One weight per routed expert, in ABSOLUTE expert order — every expert is uploaded, so
    /// the unrouted ones ride through the kernel at weight 0.
    pub wexpert: Vec<f32>,
    /// `Σ` over the routed experts through `Oracle::expert`, ascending expert id: exactly
    /// `MoE.forward` minus the shared expert and minus the final bf16 store.
    pub want: Vec<f32>,
    /// The oracle's OWN count of SwiGLU clamp events for this case, measured independently of
    /// whether any kernel clamped.
    pub clamp_events: usize,
    pub experts: F4Experts,
}

impl Case {
    /// `scale` sets how hard the activation drives the SwiGLU: 1.0 stays inside the clamp,
    /// large values cross it. Two picks at fixed ids with DIFFERENT, non-unit weights, so a
    /// kernel that dropped the routing weight or reused one for both experts fails.
    /// A case on the standard toy fixture, at **layer 0** — the layer both suites drive, and the
    /// first of the toy's three hash-routed ones. The layer is fixed rather than a parameter
    /// because every case in this port is layer 0's and a spelled `0` at six call sites is six
    /// places for a fixture to move without its measurements; [`Case::at`] takes a full
    /// [`CaseSpec`] for the one test that needs another geometry.
    pub fn new(tag: &str, scale: f32) -> Self {
        Self::at(
            toy_fixture(),
            CaseSpec {
                layer: 0,
                tag,
                scale,
            },
        )
    }

    /// The same, against a chosen fixture — the multi-trip test needs this construction at
    /// other dims.
    pub fn at(fx: &'static Toy, spec: CaseSpec<'_>) -> Self {
        let (cfg, m, o) = fx;
        let lw = &m.layers[spec.layer];
        let x = fixed_bf16(spec.tag, cfg.dim, spec.scale);
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
            let rows = ExpertOperand {
                x: &x,
                m: 1,
                weight: Some(&[w]),
            };
            let out = o.expert(&lw.experts[&e], rows, &mut counters);
            for (a, b) in want.iter_mut().zip(&out) {
                *a += b;
            }
        }
        assert_headroom(&want, "case output");

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

    /// This case as a dispatch, with `experts` and the two knobs overridable by [`Case::broken`].
    pub fn dispatch<'a>(&'a self, experts: &'a F4Experts, knobs: Knobs) -> Dispatch<'a> {
        Dispatch {
            knobs,
            ..Dispatch::reference(self.cfg, experts, &self.x, &self.wexpert)
        }
    }

    /// This case's reference knobs — the config's clamp, activation quantized. Every break below
    /// is spelled as `Knobs { .., ..c.knobs() }`, so the field it does not name is visibly the
    /// reference's.
    pub fn knobs(&self) -> Knobs {
        Knobs::reference(self.cfg)
    }

    /// The GPU answer for this case, under the reference wiring and the config's clamp.
    pub fn gpu(&self) -> Vec<f32> {
        self.dispatch(&self.experts, self.knobs()).run()
    }

    /// The GPU answer with ONE thing broken, and nothing else changed.
    pub fn broken(&self, b: Break) -> Vec<f32> {
        let e = F4Experts::upload(&self.all, b.wiring);
        self.dispatch(&e, b.knobs).run()
    }
}

/// The oracle produced something to compare, and it is far enough under `moe_fixed`'s clamp
/// that the comparison is about the arithmetic.
///
/// Both halves, together, because they are the two ways a comparison here goes vacuous or
/// spurious and every fixture owes both: [`assert_matches`] scales its tolerance by
/// `max_abs(want)`, so an all-zero oracle result passes against an all-zero kernel result at
/// tol 0; and a `want` within 2x of `MOE_ACC_MAX` fails for a reason unrelated to the question.
pub fn assert_headroom(want: &[f32], what: &str) {
    let mx = max_abs(Want(want));
    assert!(mx > 1e-6, "{what}: the oracle produced nothing to compare");
    assert!(
        mx < ACC_HEADROOM,
        "{what} is {mx:.3e}, within 2x of moe_fixed's 2^14 clamp, which the oracle does not \
         have — lower the activation scale"
    );
}

/// The two must agree within [`TOL`]; `common::assert_rel` prints `err` and `tol` side by side,
/// which is the point — a green comparison that passed on 100x of headroom looks exactly like one
/// that passed on 2x.
pub fn assert_matches(want: &[f32], got: &[f32], label: &str) {
    assert_rel(want, got, label, TOL);
}

/// The NEGATIVE, at the SAME tolerance. Every deliberate-break arm goes through here rather than
/// through a bare `assert_ne!`: a break that moved the result by less than [`TOL`] is a break the
/// positive gate would NOT have caught, so it must fail here rather than pass.
/// `common::assert_separates` carries the full argument.
pub fn assert_disagrees(want: &[f32], got: &[f32], label: &str) {
    assert_separates(want, got, label, TOL);
}

/// "Effectively unclamped", spelled as a huge POSITIVE limit — the launcher refuses `0`, negatives,
/// NaN and `±inf`, which is the stronger guarantee and the reason the break arm goes the long way
/// round.
pub const NO_CLAMP: f32 = 1e6;
