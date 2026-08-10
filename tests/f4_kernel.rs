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
//! 1. **The Sinkhorn iteration count, ON THIS FIXTURE.** At `hc_sinkhorn_iters = 20` the toy
//!    fixture's 4x4 matrix reaches a bitwise fixed point: 19 and 20 agree BIT-FOR-BIT, so no
//!    golden built on it distinguishes them
//!    (`tests/v4_oracle.rs::sinkhorn_has_converged_long_before_iteration_20`, which is also
//!    why `Defect::SinkhornIterCountProbe` is excluded from the oracle's own matrix). What
//!    [`sinkhorn_iteration_count_is_live`] proves is strictly weaker and is all that is
//!    available here: the parameter reaches the arithmetic (2 and 20 disagree). The exact
//!    value is gated by SOURCING, not by measurement on this fixture — it is passed from
//!    `V4Config`, which `V4Config::assert_matches_reference_json` pins to `config.json`.
//!
//!    > **CORRECTED 2026-08-07.** This said flatly that "a 4x4 positive matrix is far past
//!    > convergence", as a fact about the arithmetic. It is a fact about these weights. On
//!    > the checkpoint 19 vs 20 moves 39,893/53,248 of `L0.pre.ffn_norm_out` and all 78
//!    > router weights, so a real-weights golden is not blind to the count — this file is.
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
//!
//!    > **CORRECTED 2026-08-09.** Depth is no longer uncovered:
//!    > `the_dword_path_matches_the_oracle_at_multiple_trips` runs a second fixture at
//!    > 1280/1024, where gate/up takes 5 dword trips and `moe_down_f4` DOES enter the fast
//!    > path at 4. The gap this point recorded is what let M11's `#pragma unroll` ship with
//!    > 27 green tests that never executed the unrolled body; the register stays so the next
//!    > geometry-bound claim is read as one.
//! 5. **The e8m0 endpoints and the e2m1/e8m0 codes the fixture happens not to contain.**
//!    There is no exhaustive codec probe here — the accumulator's dynamic range forbids one
//!    (see [`the_fixture_exercises_the_codes_the_decoders_are_credited_with`], which
//!    measures what IS covered instead of assuming it). `e8m0f`'s `0x00` (2^-127) and
//!    `0xff` (NaN) arms are executed by nothing in this file.
//!
//!    > **CORRECTED 2026-08-08, with the branchless-decode rewrite.** The e2m1 half of
//!    > this hole is now closed: [`every_byte_pattern_decodes_right_in_both_dot_paths`]
//!    > drives all 256 packed-byte patterns through every dword byte position of the fast
//!    > path AND through the scalar tail, at scales the accumulator is faithful over, and
//!    > [`the_branchless_decodes_match_the_oracle_bitwise`] pins both decode FORMULAS to
//!    > the oracle bit for bit (all 16 e2m1 codes including `-0.0`, all 256 e8m0 bytes
//!    > including the endpoints). What still holds: the e8m0 RANGE cannot ride through the
//!    > pipeline — the accumulator argument above is untouched — so the `0x00`/`0xff` arms
//!    > are covered on the host transliteration only, and by nothing that runs on the
//!    > device.
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
// No `.unwrap()` in this file — `expect` carries the message everywhere, so only that
// lint needs allowing. (Also what keeps this prelude from being a token-for-token jscpd
// clone of `tests/kvcompress_kernel.rs`'s.)
#![allow(clippy::expect_used)]

use rivoli::backend::gpustream::HipStream;
use rivoli::backend::hip::{
    ExpertDescF4, device_sync, launch_act_quant_f8, launch_act_quant_f8_prefix,
    launch_gemv_fp8_bf16, launch_hc_post, launch_hc_pre, launch_moe_acc_drain,
    launch_moe_expert_range_f4, launch_rmsnorm_single, launch_swiglu_clamped_bf16,
};
use rivoli::memory::device::DeviceBuf;
use rivoli::v4oracle::{
    forward::{Capture, Counters, Defect, ExpertW, LayerW, Oracle, wave_ladder},
    numerics::{
        act_quant_inplace, bf16_decode, bf16_encode, e2m1_decode, e4m3_decode, e8m0_decode, silu,
    },
    toy::{self, ToyModel},
    weights::{NamedRng, V4Config, WMat},
};
use std::sync::OnceLock;

mod common;
use common::{f32b, f32v, max_abs, report_rel, residual_probe};

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
fn fixture() -> &'static Fx {
    static M: OnceLock<Fx> = OnceLock::new();
    M.get_or_init(|| build_fixture(V4Config::toy()))
}

type Fx = (V4Config, ToyModel, Oracle);

/// Shared because `build.rs`'s jscpd gate rejects the second copy of these two lines.
fn build_fixture(cfg: V4Config) -> Fx {
    let m = toy::build(&cfg);
    let o = Oracle::new(cfg.clone(), Defect::None);
    (cfg, m, o)
}

/// `WAVE * 8`, one dword-loop iteration of `dot_f4_wave_r` — a kernel constant this file
/// cannot see.
const F4_COLS_PER_TRIP: usize = 256;

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
                let wb = if swap_nibbles {
                    to_device(&nibble_swapped(w))
                } else {
                    to_device(w)
                };
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
            let bytes = std::mem::size_of_val(&descs[..]);
            std::slice::from_raw_parts(descs.as_ptr() as *const u8, bytes)
        };
        Self {
            descs: to_device(raw),
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

impl<'a> Dispatch<'a> {
    /// The reference dispatch — the config's own clamp, activation quantized. For the
    /// callers with nothing to break; `Case::dispatch` keeps the two knobs overridable
    /// because the deliberate-break tests exist to turn them. Factored when the byte-sweep
    /// test became its third verbatim copy and the duplication gate said so.
    fn reference(
        cfg: &'a V4Config,
        experts: &'a F4Experts,
        x: &'a [f32],
        wexpert: &'a [f32],
    ) -> Self {
        Dispatch {
            cfg,
            experts,
            x,
            wexpert,
            swiglu_limit: cfg.swiglu_limit,
            quantize_x: true,
        }
    }

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
        assert_eq!(
            self.wexpert.len(),
            self.experts.n,
            "one routing weight per expert"
        );
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
        let (hp, ap, op) = (
            hb.ptr_mut() as *mut f32,
            ab.ptr_mut() as *mut u64,
            ob.ptr_mut() as *mut f32,
        );
        // SAFETY: every buffer above is sized for `experts.n` ABSOLUTE expert slots and is
        // alive until the sync; every range is inside that bound.
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
                    self.swiglu_limit,
                    hp,
                    ap,
                    1,
                    stream.raw(),
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

/// Two 128-element blocks covering EVERY finite e4m3 code, as `e4m3_decode(c) · 2^-8`.
///
/// Each block is 127 codes plus one pad, and the pad is what makes it 128 WIDE — which is
/// what `act_quant`'s blocking and the oracle's `chunks_mut(128)` both require. It does NOT
/// pin the scale and is not a new magnitude: it repeats `0x7e` (+448), a code the positive
/// block already holds, and the negative block's own extreme is `0xfe` (−448) of the same
/// magnitude. So `amax` is `448 · 2^-8 = 1.75` in both either way, and
/// `fast_round_scale(1.75, 1/448)` is exactly `2^-8` (1.75/448 IS 2^-8, mantissa zero).
/// Every element therefore divides back to the code's own decoded value, which is
/// representable by construction, so the block round-trips to ITSELF and any disagreement
/// with the oracle on ANY code shows up.
///
/// 0x7f and 0xff are the format's NaN; a NaN activation is fatal upstream, so it is not this
/// fixture's business.
fn e4m3_code_blocks() -> Vec<f32> {
    const S: f32 = 1.0 / 256.0;
    [0x00u8..=0x7e, 0x80..=0xfe]
        .into_iter()
        .flat_map(|codes| {
            codes
                .map(|c| e4m3_decode(c) * S)
                .chain([e4m3_decode(0x7e) * S])
        })
        .collect()
}

/// A residual-stream-shaped activation: bf16-representable, scaled so the caller can drive
/// the SwiGLU clamp on or off. `scale` is the knob the clamp test turns.
fn draw_x(tag: &str, n: usize, scale: f32) -> Vec<f32> {
    let mut r = NamedRng::new(tag);
    (0..n)
        .map(|_| bf16_decode(bf16_encode(r.unit() * scale)))
        .collect()
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
        Self::at(fixture(), layer, tag, scale)
    }

    /// The same, against a chosen fixture — the multi-trip test needs this construction at
    /// other dims.
    fn at(fx: &'static Fx, layer: usize, tag: &str, scale: f32) -> Self {
        let (cfg, m, o) = fx;
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
            let out = o.expert(&lw.experts[&e], &x, 1, Some(&[w]), &mut counters);
            for (a, b) in want.iter_mut().zip(&out) {
                *a += b;
            }
        }
        // `moe_fixed` CLAMPS each expert contribution at `MOE_ACC_MAX = 2^(58-44) = 16384`
        // (`common.hpp`) and the oracle has no such clamp, so a case driven near it would
        // fail `assert_matches` for a reason unrelated to what the test is asking. The clamp
        // case runs at activation scale 48 — well inside, but close enough that leaving this
        // unstated would make a future scale bump a confusing red.
        let mx = max_abs(&want);
        assert!(
            mx < 8192.0,
            "case output {mx:.3e} is within 2x of moe_fixed's 2^14 clamp, \
                              which the oracle does not have — lower the activation scale"
        );

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
    fn dispatch<'a>(
        &'a self,
        experts: &'a F4Experts,
        limit: f32,
        quantize_x: bool,
    ) -> Dispatch<'a> {
        Dispatch {
            cfg: self.cfg,
            experts,
            x: &self.x,
            wexpert: &self.wexpert,
            swiglu_limit: limit,
            quantize_x,
        }
    }

    /// The GPU answer for this case, under the reference wiring and the config's clamp.
    fn gpu(&self) -> Vec<f32> {
        self.dispatch(&self.experts, self.cfg.swiglu_limit, true)
            .run()
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
    assert!(
        c.want.iter().any(|v| v.abs() > 1e-6),
        "the oracle produced nothing to compare"
    );
    assert_eq!(
        c.clamp_events, 0,
        "this case is the UNCLAMPED half of the clamp bracket"
    );
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
        (
            "nibbles read high-first",
            c.broken(Wiring::SwapNibbles, lim, true),
        ),
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
    let just_two = Dispatch::reference(c.cfg, &two, &c.x, &w).run();
    assert_eq!(
        bits(&full),
        bits(&just_two),
        "an unrouted expert perturbed the sum"
    );
    assert!(
        full.iter().any(|v| v.abs() > 1e-6),
        "both arms produced zero — nothing compared"
    );
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
    assert_eq!(
        bits(&c.gpu()),
        bits(&split),
        "range split perturbed the sum"
    );
}

/// The dword fast path at a MULTI-TRIP shape — the only test in this file that reaches one.
/// At `V4Config::toy` gate/up runs this loop exactly ONCE (`dim 256` = `WAVE * 8`) and
/// `moe_down_f4` never enters it at all (`moe_inter_dim 128`), so `#pragma unroll N` executes
/// only the remainder copy everywhere else in this file. Why the test was owed and how
/// 1280/1024 was derived (including why the first-registered 768/512 would have been vacuous
/// at depth 4): `docs/investigations/v4-decode-decomposition.md` §M11b — re-derive there if
/// the pragma ever goes past 4.
///
/// **Measured against injected defects 2026-08-09, so nobody over-trusts it:**
///   * FIRES on arithmetic wrong only past the first trip (`n7` forced to 0 when `base != 0`):
///     `err=8.133e-2 > tol=1.247e-3`, 65x, while all 27 pre-existing tests stay green. That
///     pair is the whole claim for this test's existence.
///   * BLIND to a trip miscount (`<=` -> `<`; the scalar tail resumes from `base` and absorbs
///     the dropped trip) and to a pure reassociation (an even/odd fold split PASSES here — the
///     difference is far inside a tolerance that must admit bf16 rounding; no `err=` was
///     recorded for it). Fold order belongs to the `v4res` fingerprint, which that reassociation DID
///     move (`2e7c…` against stock `9a43…`, benchmarks.md "V4 M11 fp4 resident-kernel round").
///     The engine A/B's reply md5 would also catch it in principle, but the reassociation was
///     never put through a decode — do not cite it as demonstrated. No golden hash here, see
///     `the_fp4_dispatch_hash_pins_the_clamp_hoist`.
#[test]
fn the_dword_path_matches_the_oracle_at_multiple_trips() {
    // NOT a resized `toy`: `every_byte_pattern_decodes_right_in_both_dot_paths` asserts toy's
    // one-trip geometry on purpose for its own byte-position coverage. Leaked because `Case`
    // holds `&'static` into the fixture and exactly one test wants this shape.
    // The multipliers ARE the coverage claim, so they are written as trip counts rather than
    // as 1280/1024: 5 trips = unrolled body + remainder at unroll 2 AND at unroll 4; 4 = clean
    // groups at both. NOTHING machine-checks the trip counts — the launcher's rc 1002 guards
    // `% ACT_QUANT_BLOCK` (128), not 256, so 1152 or 1536 would launch fine at a different
    // count. The test keeps its power over multi-trip arithmetic if the pragma moves, but 5/4
    // stops guaranteeing an unrolled group PLUS a remainder past depth 4, and a changed `WAVE`
    // breaks the counts outright. Re-derive from §M11b rather than trusting the green.
    let cfg = V4Config {
        dim: 5 * F4_COLS_PER_TRIP,
        moe_inter_dim: 4 * F4_COLS_PER_TRIP,
        ..V4Config::toy()
    };
    let c = Case::at(
        Box::leak(Box::new(build_fixture(cfg))),
        0,
        "unroll-trips",
        1.0,
    );

    // `got` is bound rather than inlined: `assert_matches(&c.want, &c.gpu(), ..)` is
    // token-identical to `routed_experts_match_the_oracle`'s call and `build.rs`'s jscpd
    // gate rejects the build. Inlining it back is a build error, not a style choice.
    // `report_rel` scales the tolerance by `max_abs(want)`, so an all-zero oracle result would
    // pass against an all-zero kernel result at tol 0. Four comparison tests here guard that
    // with this exact assertion (grep "the oracle produced nothing to compare"); most do not,
    // and this one runs the only UNPROVEN geometry — where a silently-empty `want` is most
    // plausible.
    assert!(
        c.want.iter().any(|v| v.abs() > 1e-6),
        "the oracle produced nothing to compare"
    );
    let got = c.gpu();
    assert_matches(
        &c.want,
        &got,
        "fp4 routed experts at 5/4 dword trips (unrolled body + remainder)",
    );
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
        guard_err(r)
    };
    let (hid, int, lim) = (cfg.dim, cfg.moe_inter_dim, cfg.swiglu_limit);

    assert!(
        go((hid, int, 0, 1, 1, lim, 1)).is_ok(),
        "the accepted case must be accepted"
    );
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
        (
            1002,
            "hidden not a whole act-quant block",
            go((129, int, 0, 1, 1, lim, 1)),
        ),
        (
            1002,
            "inter not a whole act-quant block",
            go((hid, 96, 0, 1, 1, lim, 1)),
        ),
        // BOTH sides of 1, which is what separates `!= 1` from a `> 1` that would accept 0.
        // The FP4 path instantiates only R=1: no measurement justifies a second row, and the
        // oracle is bsz=1, so one could not be scored even if it existed.
        (1003, "nrow 2", go((hid, int, 0, 1, 1, lim, 2))),
        (1003, "nrow 0", go((hid, int, 0, 1, 1, lim, 0))),
        // THE `.f4` BOUNDARY. `.vq3`/`.i4` carry the shared expert as block `n_experts`;
        // `.f4` does not, because V4's shared expert is fp8 e4m3 at 128x128. One past the
        // end here is the wrong ARITHMETIC, not merely the wrong weights.
        (
            1004,
            "one expert past the descriptor array",
            go((hid, int, 0, 2, 1, lim, 1)),
        ),
        (
            1004,
            "e_start past the descriptor array",
            go((hid, int, 1, 1, 1, lim, 1)),
        ),
        // Every value that disables the clamp, and each row is chosen to distinguish a
        // SPELLING rather than to enumerate bad numbers. `-10.0` is absent because `x <= 0`
        // would reject it too, so it separates nothing. NaN separates `!(x > 0)` from
        // `x <= 0`. **+inf separates `!(x > 0 && x < INFINITY)` from `!(x > 0)`** — and that
        // row was missing until 2026-08-05, which is exactly how the infinity route stayed
        // open on the one clamp launcher that has callers: `fminf(gt, inf)` returns `gt`, so
        // the clamp is simply gone, on every fp4 expert of every layer, silently.
        (1006, "unclamped swiglu", go((hid, int, 0, 1, 1, 0.0, 1))),
        (
            1006,
            "NaN swiglu limit",
            go((hid, int, 0, 1, 1, f32::NAN, 1)),
        ),
        (
            1006,
            "infinite swiglu limit",
            go((hid, int, 0, 1, 1, f32::INFINITY, 1)),
        ),
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
    let mut host: Vec<f32> = (0..ROWS)
        .flat_map(|r| act_quant_block(&format!("actq-{r}")))
        .collect();
    // A row of two blocks, to prove the tiling advances by 128 within a row rather than
    // treating each row as one block: the second block's amax differs from the first's, so
    // a kernel that reused one scale for the whole row produces different numbers.
    let wide: Vec<f32> = act_quant_block("actq-wide")
        .into_iter()
        .chain(act_quant_block("actq-wide2").into_iter().map(|v| v * 0.125))
        .collect();
    host.extend_from_slice(&wide);
    // Every finite e4m3 code, so the round trip is pinned over the whole format rather than
    // over whatever magnitudes the tie fixture happened to reach. This is the ONLY
    // exhaustive codec coverage in the suite — `the_fixture_exercises_the_codes_...`
    // measures the fp4 side and finds it narrow (2 distinct e8m0 codes at toy scale).
    let code_blocks = e4m3_code_blocks();
    host.extend_from_slice(&code_blocks);

    let mut want = host.clone();
    for row in want.chunks_mut(128) {
        act_quant_inplace(row, 128, true);
    }

    let mut b = to_device(&f32b(&host));
    let stream = HipStream::new().expect("stream");
    // The launch PLAN is the data, and the assertion sums the launch extents — not the
    // fixture lengths. That distinction is the whole guard: `rows + wide + codes ==
    // host.len()` reads like a check and is a TAUTOLOGY, because `host` is the
    // concatenation of exactly those three pieces. It cannot fail, and it says nothing
    // about what was dispatched.
    //
    // It has to be the launch arguments, because nothing downstream can help: the code
    // blocks round-trip to THEMSELVES, so an undispatched one is bit-identical to a
    // correctly quantized one in `got`. Halve the third extent or delete it and the suite
    // went green — twice, under two earlier versions of this guard. Found by review
    // 2026-08-05, after the first fix for it turned out to be the tautology above.
    //
    // Asserting BEFORE the `unsafe` block is also what makes its SAFETY claim true: an
    // over-covering extent would write past the allocation before any later check ran.
    let plan = [(ROWS, 128), (1, wide.len()), (code_blocks.len() / 128, 128)];
    let covered: usize = plan.iter().map(|(r, n)| r * n).sum();
    assert_eq!(
        covered,
        host.len(),
        "the launches do not cover the fixture exactly"
    );
    println!(
        "act_quant_f8: {} blocks dispatched, {} of them e4m3-code blocks",
        covered / 128,
        plan[2].0
    );
    // SAFETY: `b` holds `host.len()` live f32, and the plan covers exactly that — asserted
    // immediately above, from the same extents the loop dispatches.
    unsafe {
        let p = b.ptr_mut() as *mut f32;
        let mut at = 0;
        for (rows, row_len) in plan {
            launch_act_quant_f8(p.add(at), rows, row_len, stream.raw()).expect("act_quant_f8");
            at += rows * row_len;
        }
    }
    let got = sync_f32(&b);

    assert_ne!(
        bits(&want),
        bits(&host),
        "the quantizer left the input unchanged — nothing was compared"
    );
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
        for w in [
            &m.layers[0].experts[&e].w1,
            &m.layers[0].experts[&e].w3,
            &m.layers[0].experts[&e].w2,
        ] {
            let (packed, s) = fp4_spans(w);
            for b in packed {
                nibbles[(b & 0x0f) as usize] += 1;
                nibbles[(b >> 4) as usize] += 1;
            }
            scales.extend(s.iter().copied());
        }
    }
    let missing: Vec<usize> = (0..16).filter(|&n| nibbles[n] == 0).collect();
    assert!(
        missing.is_empty(),
        "e2m1 codes never exercised by a ROUTED expert: {missing:?}"
    );
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
    assert!(
        !scales.contains(&0u8) && !scales.contains(&0xffu8),
        "a special e8m0 code leaked in"
    );
}

/// The branchless decode FORMULAS against the oracle, bit for bit, over every code.
///
/// `common.hpp`'s `e2m1f`/`e8m0f` were rewritten branchless on 2026-08-08 (the M3a
/// kernel-rate lever: the ternary forms compiled to an exec-mask branch region per nibble,
/// ~88 of the fp4 dot loop's 195 instructions). The two functions below are line-for-line
/// transliterations of the NEW bodies — if you change the kernel, change these or this
/// test lies. What each half pins, and what it cannot:
///
/// - This test proves the FORMULAS equal `v4oracle::numerics::{e2m1_decode, e8m0_decode}`
///   at the bit level — including code 8 → `-0.0` (the sign OR on a zero payload) and
///   e8m0's `0x00` (the f32 subnormal 2^-127) and `0xff` (the 0x7fc00000 NaN), none of
///   which any device test can observe: a `-0.0` weight is annihilated by the very next
///   multiply, and the e8m0 endpoints cannot ride through `moe_fixed` (module doc, item 5).
/// - It proves nothing about what hipcc COMPILED. That bridge is
///   [`every_byte_pattern_decodes_right_in_both_dot_paths`], which runs the real kernels
///   over every packed-byte pattern; a transliteration drifted from the kernel fails there.
#[test]
fn the_branchless_decodes_match_the_oracle_bitwise() {
    // kernels/common.hpp::e2m1f — magnitudes doubled are {0,1,2,3,4,6,8,12}, one immediate.
    fn e2m1f(nib: u32) -> f32 {
        let half = (0xC864_3210u32 >> ((nib & 7) << 2)) & 0xF;
        let mag = 0.5f32 * half as f32;
        f32::from_bits(mag.to_bits() | ((nib & 8) << 28))
    }
    // kernels/common.hpp::e8m0f — `max` against the b == 0 subnormal, quiet bit on 0xff.
    fn e8m0f(b: u8) -> f32 {
        let t = (b as u32) << 23;
        let quiet = if b == 0xff { 1u32 << 22 } else { 0 };
        f32::from_bits(t.max(1u32 << 22) | quiet)
    }
    for nib in 0..16u32 {
        assert_eq!(
            e2m1f(nib).to_bits(),
            e2m1_decode(nib as u8).to_bits(),
            "e2m1 code {nib}"
        );
    }
    for b in 0..=255u8 {
        assert_eq!(
            e8m0f(b).to_bits(),
            e8m0_decode(b).to_bits(),
            "e8m0 byte {b}"
        );
    }
}

/// Distinct byte values seen at dword byte position `p` of `w` — both byte-pattern
/// sweeps COUNT their coverage with this rather than trusting their constructions
/// (and jscpd caught them carrying the count verbatim before it was factored).
/// One output row's operands and geometry, shared by the fold transliterations below.
/// A struct rather than three seven-argument functions, because jscpd correctly reads a
/// transliterated parameter list repeated three times as a clone — the `matvec_*` lists in
/// `artifact/quant.rs` carry an exemption for exactly this shape, and a struct is the fix
/// that needs none.
struct FoldRow<'a> {
    x: &'a [f32],
    wrow: &'a [u8],
    srow: &'a [f32],
    lut: &'a [f32],
    bsh: usize,
    n4: usize,
    k: usize,
}

impl FoldRow<'_> {
    /// One THREAD's strided share of the fp8 dot, in the kernel's emitted contraction
    /// (`common.hpp::fp8_dot_strided`: `q = x1·l1` rounded, then three fmas, then
    /// `acc = fma(s, q, acc)`; the scalar tail is `acc = fma(x·l, s, acc)`) — the
    /// transliteration BOTH fold-order tests build their partials from. `start`/`stride`
    /// are the kernel's own arguments: `(lane, 32)` under wave-per-row `gemv_fp8_bf16`,
    /// `(threadIdx.x, 256)` under `gemv_fp8_bf16_splitk`. One definition, because the two
    /// tests pin the same chain against two different combine trees, and a drift between
    /// two copies would be indistinguishable from the kernel drift they exist to catch.
    fn chain(&self, start: usize, stride: usize) -> f32 {
        let (x, wrow, srow, lut) = (self.x, self.wrow, self.srow, self.lut);
        let mut acc = 0.0f32;
        for jj in (start..self.n4).step_by(stride) {
            let i0 = jj * 4;
            let mut q = x[i0 + 1] * lut[wrow[i0 + 1] as usize];
            q = x[i0].mul_add(lut[wrow[i0] as usize], q);
            q = x[i0 + 2].mul_add(lut[wrow[i0 + 2] as usize], q);
            q = x[i0 + 3].mul_add(lut[wrow[i0 + 3] as usize], q);
            acc = srow[i0 >> self.bsh].mul_add(q, acc);
        }
        for i in ((self.n4 * 4 + start)..self.k).step_by(stride) {
            acc = (x[i] * lut[wrow[i] as usize]).mul_add(srow[i >> self.bsh], acc);
        }
        acc
    }

    /// The row under `gemv_fp8_bf16`'s WAVE-PER-ROW fold: 32 lane chains at stride 32, one
    /// ladder (`v4oracle::forward::wave_ladder` — the shared definition). UNROUNDED — the
    /// caller applies the bf16 store where the kernel does.
    fn serial(&self) -> f32 {
        let mut lanes = [0.0f32; 32];
        for (l, acc) in lanes.iter_mut().enumerate() {
            *acc = self.chain(l, 32);
        }
        wave_ladder(lanes)
    }
}

/// Upload `(x, w, scales)` and dispatch `launch_gemv_fp8_bf16` at `(m, n_out, k, block)`,
/// groups = 1 — the device harness both fold-order tests share. Which KERNEL runs is the
/// launcher's shape dispatch, which is part of what the split-k test asserts.
fn gemv_fp8_on_device(
    x: &[f32],
    w: &[u8],
    scales: &[f32],
    m: usize,
    n_out: usize,
    k: usize,
    block: usize,
) -> Vec<f32> {
    let (wd, sd, xd) = (to_device(w), to_device(&f32b(scales)), to_device(&f32b(x)));
    let mut od = zeros(m * n_out * size_of::<f32>());
    let stream = HipStream::new().expect("stream");
    // SAFETY: `xd` is `m * k` f32, `wd` is `n_out * k` bytes, `sd` covers
    // `ceil(n_out/block) * ceil(k/block)` f32, `od` is `m * n_out` f32; `sync_f32` joins
    // the device before any buffer drops.
    unsafe {
        launch_gemv_fp8_bf16(
            xd.ptr().cast(),
            wd.ptr().cast(),
            sd.ptr().cast(),
            m,
            n_out,
            k,
            block,
            1,
            od.ptr_mut().cast(),
            stream.raw(),
        )
    }
    .expect("gemv_fp8_bf16 dispatch");
    sync_f32(&od)
}

/// The host reference for an `m = 1` wave-per-row fp8 launch: [`FoldRow::serial`] per
/// output row, bf16-stored where the kernel stores. ONE definition for both fold-order
/// tests — the k = 1152 sweep and the M10 fused-concat case — because a drift between
/// two copies of this arithmetic would be indistinguishable from the kernel drift they
/// exist to catch (the same argument [`FoldRow::chain`] carries for itself).
///
/// `sc` and not `scales`, deliberately: rustfmt breaks the longer name's signature over
/// seven lines whose token run then CLONES [`gemv_fp8_on_device`]'s parameter list — the
/// "rustfmt manufactured duplication" failure CLAUDE.md records, dodged at the cost of
/// one shorter name rather than an exemption.
fn serial_fold(x: &[f32], w: &[u8], sc: &[f32], n_out: usize, k: usize, block: usize) -> Vec<f32> {
    let bsh = block.trailing_zeros() as usize;
    let sc_cols = k.div_ceil(block);
    let n4 = if block >= 4 { k >> 2 } else { 0 };
    let lut: Vec<f32> = (0..256).map(|b| e4m3_decode(b as u8)).collect();
    (0..n_out)
        .map(|j| {
            let row = FoldRow {
                x,
                wrow: &w[j * k..(j + 1) * k],
                srow: &sc[(j >> bsh) * sc_cols..],
                lut: &lut,
                bsh,
                n4,
                k,
            };
            bf16_decode(bf16_encode(row.serial()))
        })
        .collect()
}

fn position_coverage(w: &[u8], p: usize) -> usize {
    let mut seen = [false; 256];
    for b in w.iter().skip(p).step_by(4) {
        seen[*b as usize] = true;
    }
    seen.iter().filter(|&&s| s).count()
}

/// Packed bytes that put every one of the 256 values at every byte position of the dword
/// fast path: byte `i` carries `(i/4 + 64·(i%4) + salt) mod 256`, so position `p`'s bytes
/// walk all 256 values once per 256 consecutive dwords (1024 bytes). `salt` decorrelates the three
/// projections.
fn covering_bytes(n: usize, salt: u8) -> Vec<u8> {
    (0..n)
        .map(|i| {
            ((i >> 2) as u8)
                .wrapping_add((64 * (i & 3)) as u8)
                .wrapping_add(salt)
        })
        .collect()
}

/// The compiled kernels' decode over EVERY packed-byte pattern, in both dot paths.
///
/// [`the_branchless_decodes_match_the_oracle_bitwise`] pins the decode formulas;
/// this is the bridge to what hipcc actually emitted. One synthetic expert whose `w1`/`w3`
/// bytes are [`covering_bytes`] runs the toy geometry's single dword-path iteration
/// (`dim = 256` = WAVE·8, module doc item 4), so every byte position of the weight dword
/// decodes every value 0..=255 — a wrong shift, mask or table constant at ANY position
/// fails against the oracle on thousands of terms. `w2` (`inter = 128` < WAVE·8) decodes
/// entirely in the scalar tail, whose bytes cover all 256 values too — both nibble
/// extraction parities included. Coverage is COUNTED below, not trusted from the
/// construction; scales cycle 2^-2..2^1 and the activation is small, so everything stays
/// inside `moe_fixed`'s faithful band (the constraint that forbids sweeping e8m0 the same
/// way) and outside the SwiGLU clamp, which would otherwise mask a gate-row decode error
/// behind a saturated `min`. The coverage claim is about byte POSITIONS, not loop trips:
/// at one iteration the dword loop's advance (`base += 256`, scale groups past 7) never
/// runs — untouched by the decode rewrite, and netted by the A/B's byte-identical gate at
/// the real multi-iteration dims.
#[test]
fn every_byte_pattern_decodes_right_in_both_dot_paths() {
    let (cfg, _, o) = fixture();
    let (hidden, inter) = (cfg.dim, cfg.moe_inter_dim);
    // The coverage claims are geometry-bound; a resized toy silently voids them.
    assert_eq!(
        hidden, 256,
        "gate/up must be exactly one dword-path iteration"
    );
    assert_eq!(inter, 128, "w2 must decode entirely in the scalar tail");
    let scales = |rows: usize, k: usize| -> Vec<u8> {
        (0..rows * (k / 32)).map(|i| 125 + (i % 4) as u8).collect()
    };
    let (w1w, w3w, w2w) = (
        covering_bytes(inter * hidden / 2, 0),
        covering_bytes(inter * hidden / 2, 101),
        covering_bytes(hidden * inter / 2, 202),
    );
    for (label, w) in [("w1", &w1w), ("w3", &w3w)] {
        for p in 0..4 {
            let n = position_coverage(w, p);
            assert_eq!(n, 256, "{label} dword byte position {p}: {n}/256 patterns");
        }
    }
    let mut seen = [false; 256];
    for &b in &w2w {
        seen[b as usize] = true;
    }
    let n = seen.iter().filter(|&&s| s).count();
    assert_eq!(n, 256, "scalar tail (w2): {n}/256 patterns");

    let mat = |rows, cols, w, s| WMat::Fp4 { rows, cols, w, s };
    let e = ExpertW {
        w1: mat(inter, hidden, w1w, scales(inter, hidden)),
        w2: mat(hidden, inter, w2w, scales(hidden, inter)),
        w3: mat(inter, hidden, w3w, scales(inter, hidden)),
    };
    let x = draw_x("byte-pattern-sweep-x", hidden, 0.05);
    let mut counters = Counters::default();
    let want = o.expert(&e, &x, 1, Some(&[1.125]), &mut counters);
    assert_eq!(
        counters.swiglu_clamp_events, 0,
        "a saturated clamp would mask gate-row decode errors — lower the activation scale"
    );
    // The two guards Case::new carries, for the same reasons: an all-zero want proves
    // nothing, and a want near moe_fixed's 2^14 clamp fails for an unrelated reason.
    assert!(
        max_abs(&want) > 1e-6,
        "the oracle produced nothing to compare"
    );
    assert!(
        max_abs(&want) < 8192.0,
        "sweep output too close to moe_fixed's clamp"
    );

    let experts = F4Experts::upload(&[&e], Wiring::Correct);
    let got = Dispatch::reference(cfg, &experts, &x, &[1.125]).run();
    assert_matches(&want, &got, "byte-pattern sweep (fp4)");
}

/// `gemv_fp8_bf16`'s summation ORDER, pinned bit-for-bit — the M7 unroll's oracle
/// (`common.hpp::fp8_dot_strided`, docs/investigations/v4-decode-decomposition.md §M7).
///
/// The fp8 twin of the sweep above cannot reuse its oracle: `Oracle::linear` folds
/// sequentially and the kernel wave-reduces, so that comparison rides [`TOL`]
/// (tests/f4_attn.rs) and would wave through an unroll that split the accumulator chain
/// — the exact failure the M7 change must not have. So the reference here is a host
/// transliteration of the KERNEL's own fold: per-lane strided accumulation in the
/// emitted contraction (`q = x1·l1` rounded, then three fmas, then `acc = fma(s, q,
/// acc)`; the tail is `acc = fma(x·l, s, acc)`), `wave_sum`'s shfl-down ladder, `rbf16`.
/// That pins two things a tolerance cannot: the unroll left the chain single and in
/// ascending-`j` order, and the compiler's contraction pattern did not drift — if a
/// future hipcc contracts differently this fails and the ISA gets re-read, which is
/// this repo's rule anyway.
///
/// `k = 1152` = 9 per-lane dword trips — one unrolled body plus one remainder pass, so
/// the unroll REMAINDER loop (unreachable at every engine dimension: all real trip
/// counts divide 8) is exercised here. The `block = 2` dispatch routes the SAME bytes
/// through the scalar tail (`n4 = 0` below a quad-wide scale tile), covering the other
/// loop entirely.
///
/// The two NaN codes (0x7F/0xFF) are EXCLUDED from the sweep: NaN payload propagation
/// through an FMA chain is not contractual across host and device, so a bitwise
/// comparison over them would pin an implementation accident. Their decode formula
/// stays covered by [`the_branchless_decodes_match_the_oracle_bitwise`]'s e4m3 sibling
/// (`e4m3_decode` against the LUT builder's own `e4m3f`, exercised on every non-NaN
/// code here). Coverage is COUNTED below, not trusted from the construction.
#[test]
fn the_fp8_dot_sums_in_source_order_through_both_loops() {
    assert!(
        e4m3_decode(0x7f).is_nan() && e4m3_decode(0xff).is_nan(),
        "the two excluded codes must be exactly the NaN ones"
    );
    const K: usize = 1152;
    const N_OUT: usize = 8;
    let allowed: Vec<u8> = (0u8..=255).filter(|b| !matches!(b, 0x7f | 0xff)).collect();
    let w: Vec<u8> = (0..N_OUT * K)
        .map(|i| allowed[(i / 4 + (i % 4) * 67) % allowed.len()])
        .collect();
    for p in 0..4 {
        let n = position_coverage(&w, p);
        assert_eq!(n, 254, "dword byte position {p}: {n}/254 patterns");
    }
    let x = draw_x("fp8-order-x", K, 0.05);
    // Sized for the block=2 dispatch's worst consumer: 4 row-blocks x 576 column tiles;
    // block=128 reads row 0's first 9 entries of the same buffer. Powers of two only by
    // habit — the host model replays the identical arithmetic whatever the scale.
    let scales: Vec<f32> = (0..(N_OUT / 2) * (K / 2))
        .map(|i| [0.25f32, 0.5, 1.0, 2.0][i % 4])
        .collect();

    // `K = 1152 < 4096`, so both blocks stay on the wave-per-row kernel — the split-k
    // dispatch is the other test's subject.
    for block in [128usize, 2] {
        let (want, got) = (
            serial_fold(&x, &w, &scales, N_OUT, K, block),
            gemv_fp8_on_device(&x, &w, &scales, 1, N_OUT, K, block),
        );
        assert_eq!(
            bits(&want),
            bits(&got),
            "fp8 dot order (block {block}): kernel fold differs from the source's"
        );
    }
}

/// The M10 `[wkv ‖ wq_a]` concat at the ENGINE's shapes — the per-row oracle coverage
/// for the fused `[1536 × 4096]` grid, seam included, in two bitwise claims:
///
/// 1. fused `out[0..512]` / `out[512..]` equal the two standalone launches the fusion
///    replaces. This is the load-time concat's layout contract executed by the real
///    kernel: fused row `512 + r` must read concatenated scale row `4 + r/128`. The
///    scale cycle's period (3) is COPRIME to the 32-entry scale rows, so every scale
///    row's phase differs from its neighbours' — a scale-grid concat shifted by one
///    block row in EITHER direction changes in-bounds values on every affected row
///    and fails on thousands of terms (a period dividing 32 would make within-tensor
///    neighbour rows identical and leave the +1 shift visible only as an
///    out-of-bounds read; review caught that first cut) — the "catch it by
///    arithmetic first" case.
/// 2. every fused row equals [`FoldRow::serial`] — the same source-order pin the
///    k = 1152 sweep holds, at the fused shape's 32 whole per-lane trips (no remainder
///    pass; that loop keeps its coverage in the sweep above).
///
/// Weight bytes are the sweep's covering pattern (NaN codes excluded, per its argument)
/// with different salts per tensor, so the two sides of the seam cannot mask each other.
#[test]
fn the_fused_qkv_gemv_is_bitwise_the_two_launches_it_replaces() {
    const K: usize = 4096;
    const N_KV: usize = 512;
    const N_QA: usize = 1024;
    const BLOCK: usize = 128;
    let allowed: Vec<u8> = (0u8..=255).filter(|b| !matches!(b, 0x7f | 0xff)).collect();
    let wrow = |n: usize, salt: usize| -> Vec<u8> {
        (0..n)
            .map(|i| allowed[(i / 4 + (i % 4) * 67 + salt) % allowed.len()])
            .collect()
    };
    let (w_kv, w_qa) = (wrow(N_KV * K, 0), wrow(N_QA * K, 131));
    // Power-of-two scales in a period-3 cycle (coprime to the 32-entry rows — see the
    // doc above), offset between the tensors: a seam or shift error lands rows on
    // scales that differ from their own, so equality would be impossible.
    let scl = |rows: usize, salt: usize| -> Vec<f32> {
        (0..rows.div_ceil(BLOCK) * K.div_ceil(BLOCK))
            .map(|i| [0.25f32, 0.5, 1.0][(i + salt) % 3])
            .collect()
    };
    let (s_kv, s_qa) = (scl(N_KV, 0), scl(N_QA, 1));
    let x = draw_x("qkv-fuse-x", K, 0.05);
    let kv = gemv_fp8_on_device(&x, &w_kv, &s_kv, 1, N_KV, K, BLOCK);
    let qa = gemv_fp8_on_device(&x, &w_qa, &s_qa, 1, N_QA, K, BLOCK);
    let wf = [w_kv.as_slice(), w_qa.as_slice()].concat();
    let sf = [s_kv.as_slice(), s_qa.as_slice()].concat();
    let f = gemv_fp8_on_device(&x, &wf, &sf, 1, N_KV + N_QA, K, BLOCK);
    assert_eq!(
        bits(&f[..N_KV]),
        bits(&kv),
        "kv rows through the fused grid"
    );
    assert_eq!(bits(&f[N_KV..]), bits(&qa), "wq_a rows across the seam");
    assert_eq!(
        bits(&serial_fold(&x, &wf, &sf, N_KV + N_QA, K, BLOCK)),
        bits(&f),
        "fused rows against the source-order fold"
    );
}

// =======================================================================================
// 5. mHC
// =======================================================================================

/// One sublayer's three mHC tensors, in the order `hc_pre` reads them.
///
/// A struct rather than three adjacent `&[f32]` parameters. `fn`, `scale` and `base` have
/// different LENGTHS but the same type, they were spelled adjacently at three call sites, and
/// a transposition compiles: `hc_pre` uploads each into its own buffer and reads them at the
/// strides its own arguments imply, so swapping two would produce finite, plausible numbers
/// from the wrong tensors — and both this and the golden it is scored against would be
/// reading the same wrong thing only if the ORACLE were wrong the same way, which it is not,
/// so it would surface as an unexplained numeric gap. Naming the constructors
/// [`HcW::attn`]/[`HcW::ffn`] also removes the second mistake available here: a caller
/// mixing `hc_attn_scale` into the ffn triple.
///
/// The same argument `tests/common/mod.rs`'s `Mla` makes about six `usize`.
#[derive(Clone, Copy)]
struct HcW<'a> {
    fnw: &'a [f32],
    scale: &'a [f32],
    base: &'a [f32],
}

impl<'a> HcW<'a> {
    fn attn(lw: &'a LayerW) -> Self {
        Self {
            fnw: &lw.hc_attn_fn,
            scale: &lw.hc_attn_scale,
            base: &lw.hc_attn_base,
        }
    }

    fn ffn(lw: &'a LayerW) -> Self {
        Self {
            fnw: &lw.hc_ffn_fn,
            scale: &lw.hc_ffn_scale,
            base: &lw.hc_ffn_base,
        }
    }
}

/// `hc_pre` for one sublayer, returning `(y, post, comb)` device-side readbacks.
fn gpu_hc_pre(
    cfg: &V4Config,
    h: &[f32],
    w: HcW<'_>,
    s: usize,
    iters: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let stream = HipStream::new().expect("stream");
    let (hc, dim) = (cfg.hc_mult, cfg.dim);
    let (hb, fb) = (to_device(&f32b(h)), to_device(&f32b(w.fnw)));
    let (sb, bb) = (to_device(&f32b(w.scale)), to_device(&f32b(w.base)));
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

/// `rmsnorm_single` on the GPU, so the `hc_pre` comparison lands on a golden the oracle emits.
///
/// **rivoli's `rmsnorm_single` kernel does NOT bf16-round its output, and V4's `RMSNorm.forward`
/// returns bf16** (`model.py:197-202` computes in f32 and the module's dtype is bf16). That
/// is a real gap and it is NOT this stream's to close — `rmsnorm_single` is shared with the GLM
/// path, where adding a store would change shipped output. `mhc_reproduces_the_layer_
/// goldens` applies the missing round on the host and PRINTS what it was worth, so the
/// number is on the record rather than absorbed into a tolerance. **S3 owns supplying it.**
fn gpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let dim = w.len();
    assert!(
        x.len().is_multiple_of(dim),
        "x must be whole rows of the norm weight"
    );
    let (xb, wb) = (to_device(&f32b(x)), to_device(&f32b(w)));
    let mut y = zeros(x.len() * 4);
    // ONE LAUNCH PER TOKEN. `rivoli_rmsnorm_single` is single-row — `dim3(1)`, one mean over its
    // whole `n`, and `w[i]` indexed over that same `n`. Handing it `s·dim` took a JOINT rms
    // over every token (the oracle's is per token, `x.chunks_mut(d)`) and read the norm
    // weight `s-1` rows past its allocation. Both were silent: the golden's length still
    // matched, so `compare` was happy, and the arithmetic error rode in as a plausible
    // scale. Found by review, 2026-08-05.
    for t in 0..x.len() / dim {
        // SAFETY: row `t` is `dim` live f32 inside both buffers, and `w` is `dim` long.
        unsafe {
            launch_rmsnorm_single(
                (xb.ptr() as *const f32).add(t * dim),
                wb.ptr() as *const f32,
                dim,
                eps,
                (y.ptr_mut() as *mut f32).add(t * dim),
            )
        }
        .expect("rmsnorm_single");
    }
    sync_f32(&y)
}

/// One prefill `run_layer` capture, for the tests that score against goldens rather than
/// against a re-derivation.
fn capture(layer: usize, s: usize) -> (Capture, Vec<f32>, Vec<u32>) {
    let (cfg, m, o) = fixture();
    let mut h = residual_probe(cfg, "hc-h", s);
    let mut ri = NamedRng::new("hc-ids");
    let ids: Vec<u32> = (0..s).map(|_| ri.below(cfg.vocab_size) as u32).collect();
    let h0 = h.clone();
    let cap = common::prefill_capture(o, &m.layers[layer], layer, &ids, &mut h);
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
    let g = |n: &str| {
        cap.float(n)
            .unwrap_or_else(|| panic!("golden {n}"))
            .to_vec()
    };

    assert_eq!(
        h_in,
        g("L0.pre.in"),
        "the driver's h is not what the oracle recorded"
    );

    // The bf16 store `rmsnorm_single` is missing (see `gpu_rmsnorm`). Measured on the way past so
    // the size of the gap is recorded rather than inferred: `report` prints the unrounded
    // error next to the rounded one, and if the two are ever the same number the missing
    // store has stopped mattering and this wrapper can go.
    let norm = |v: &[f32], w: &[f32], label: &str| {
        let raw = gpu_rmsnorm(v, w, cfg.norm_eps);
        let rounded: Vec<f32> = raw.iter().map(|x| bf16_decode(bf16_encode(*x))).collect();
        // Asserted, not merely printed: `println!` is captured and discarded on a green run,
        // so "the number is on the record" would be a claim about output nobody sees. This
        // goes red exactly when the wrapper stops being needed.
        let (err, _) = compare(
            &raw,
            &rounded,
            &format!("{label}: rmsnorm's missing bf16 store"),
        );
        assert!(
            err > 0.0,
            "{label}: rmsnorm's output is already bf16-representable — the missing store has \
             stopped mattering, so drop `norm` and call `gpu_rmsnorm` directly"
        );
        rounded
    };

    let (y, post, comb) = gpu_hc_pre(cfg, &h_in, HcW::attn(lw), S, iters);
    assert_matches(
        &g("L0.pre.attn_norm_out"),
        &norm(&y, &lw.attn_norm, "attn"),
        "hc_pre(attn) then rmsnorm",
    );

    let h1 = gpu_hc_post(cfg, &g("L0.pre.attn_out"), &h_in, &post, &comb, S);
    let (y2, post2, comb2) = gpu_hc_pre(cfg, &h1, HcW::ffn(lw), S, iters);
    assert_matches(
        &g("L0.pre.ffn_norm_out"),
        &norm(&y2, &lw.ffn_norm, "ffn"),
        "hc_post(attn) then hc_pre(ffn) then rmsnorm",
    );

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
    let (f, sc, b) = (
        to_device(&f32b(&lw.hc_attn_fn)),
        to_device(&f32b(&lw.hc_attn_scale)),
        to_device(&f32b(&lw.hc_attn_base)),
    );
    // Sized for the ACCEPTED case each launcher runs, which is the half of a guard test
    // that actually touches memory: `hc_pre` writes `s·dim`, `hc_post` writes `s·hc·dim`.
    // A single shared output buffer sized for `hc_pre` would let `hc_post`'s accepted arm
    // overrun it by `hc`x — the first draft of this test did exactly that.
    let mut y = zeros(cfg.dim * 4);
    let mut expanded = zeros(cfg.hc_mult * cfg.dim * 4);
    let mut post = zeros(cfg.hc_mult * 4);
    let mut comb = zeros(cfg.hc_mult * cfg.hc_mult * 4);

    // Addresses hoisted so the two closures below vary only their guarded arguments.
    let (hp, fp, scp, bp_) = (
        h.ptr() as *const f32,
        f.ptr() as *const f32,
        sc.ptr() as *const f32,
        b.ptr() as *const f32,
    );
    let (yp, pp, cp) = (
        y.ptr_mut() as *mut f32,
        post.ptr_mut() as *mut f32,
        comb.ptr_mut() as *mut f32,
    );
    let ep = expanded.ptr_mut() as *mut f32;
    let pre = |s, hc, iters| {
        // SAFETY: every rejected case returns before a dereference, and the accepted one is
        // sized by the buffers above; all of them outlive the sync that follows it.
        guard_err(unsafe {
            launch_hc_pre(
                hp,
                fp,
                scp,
                bp_,
                s,
                hc,
                cfg.dim,
                iters,
                cfg.norm_eps,
                cfg.hc_eps,
                yp,
                pp,
                cp,
                stream.raw(),
            )
        })
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
        guard_err(unsafe { launch_hc_post(yp, hp, pp, cp, s, hc, cfg.dim, ep, stream.raw()) })
    };
    assert!(
        post_call(1, hc).is_ok(),
        "the accepted case must be accepted"
    );
    device_sync().expect("device sync");
    assert_guards(vec![
        (1001, "hc_post zero tokens", post_call(0, hc)),
        (1002, "hc_post hc_mult 2", post_call(1, 2)),
    ]);
}

/// The Sinkhorn iteration count reaches the arithmetic.
///
/// **This is NOT a check that the count is 20**, and on this fixture it cannot be: at 20
/// passes the toy's 4x4 matrix has reached a bitwise fixed point and 19 and 20 agree
/// bit-for-bit — the oracle's own matrix excludes `Defect::SinkhornIterCountProbe` for that
/// measured reason. What is provable here is that the parameter is live rather than ignored,
/// which is what makes SOURCING it from `V4Config` (and
/// `V4Config::assert_matches_reference_json` pinning that to `config.json`) the actual gate
/// on the value. A golden emitted from the CHECKPOINT would gate it directly — see the
/// dated note on `sinkhorn_has_converged_long_before_iteration_20`.
#[test]
fn sinkhorn_iteration_count_is_live() {
    let (cfg, m, _) = fixture();
    const S: usize = 2;
    let lw = &m.layers[0];
    let h = residual_probe(cfg, "sink-h", S);
    assert!(cfg.hc_sinkhorn_iters >= 2, "this test subtracts one below");
    let run = |iters| gpu_hc_pre(cfg, &h, HcW::attn(lw), S, iters);
    let (_, _, c20) = run(cfg.hc_sinkhorn_iters);
    let (_, _, c2) = run(2);
    let (_, _, c19) = run(cfg.hc_sinkhorn_iters - 1);
    // Bit-inequality, matching the claim the oracle's own test makes rather than a
    // threshold picked here: `sinkhorn_has_converged_long_before_iteration_20` asserts
    // `!identical(20, 2)`, and a magnitude threshold would be a weaker statement that could
    // pass for a kernel whose `iters` only tickled the low bits.
    assert_ne!(
        bits(&c20),
        bits(&c2),
        "2 and {} iterations agree — `iters` never reaches the kernel",
        cfg.hc_sinkhorn_iters
    );
    // The blind spot itself, asserted in the direction the oracle asserts it. If this ever
    // goes red, the fixture's fixed point has stopped holding on the GPU where it holds on
    // the CPU — which would mean the two arithmetics have diverged somewhere worth finding.
    // It would NOT be news that the count is observable in general: on the checkpoint it
    // already is.
    assert_eq!(
        bits(&c20),
        bits(&c19),
        "19 and 20 iterations disagree on the GPU where they agree on the CPU"
    );
}

// =======================================================================================
// 6. the MoE layer, end to end against a golden
// =======================================================================================

/// `.ffn_out` reproduced from `.ffn_norm_out` — the FP4 kernels running the experts the
/// golden selected, and the SHARED expert filled in from the oracle.
///
/// The shared expert is fp8 e4m3 at 128x128, not FP4, and is explicitly out of S2a. It is
/// computed here by `Oracle::expert` so the comparison can reach a real golden at all; the
/// consequence is that **this test says nothing about rivoli's fp8 path**, and a defect
/// there would be invisible to it.
///
/// **The selection is READ from the golden, not computed, since `moe_gate_v4` was deleted
/// 2026-08-09.** It used to come from that kernel and be asserted equal to
/// `L0.pre.router_indices`/`router_weights` on the next two lines — so the golden was already
/// the authority, and taking it directly asserts the same thing about the FP4 path while
/// removing the last caller of a kernel the engine never reached. Routing in this engine is
/// HOST work (`math::route_into`, the router `architecture.md` INV-1 is stated about);
/// `f4gpu.rs::route_row` carries why. **This test therefore covers no router at all** — that
/// is the trade the deletion made, and it is recorded here rather than left to be discovered,
/// because a reader who sees a golden-fed selection will otherwise assume it was checked.
#[test]
fn ffn_out_matches_the_golden() {
    let (cfg, m, o) = fixture();
    let layer = 0usize;
    let lw = &m.layers[layer];
    let (cap, _, _) = capture(layer, 1);
    let x = cap
        .float("L0.pre.ffn_norm_out")
        .expect("ffn_norm_out golden")
        .to_vec();

    let gi: Vec<i32> = cap
        .int("L0.pre.router_indices")
        .expect("indices")
        .iter()
        .map(|&e| e as i32)
        .collect();
    let gw = cap
        .float("L0.pre.router_weights")
        .expect("router_weights")
        .to_vec();

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
        assert_eq!(
            wexpert[e as usize], 0.0,
            "expert {e} was picked twice for one token"
        );
        wexpert[e as usize] = gw[j];
    }
    let experts: Vec<&ExpertW> = (0..cfg.n_routed_experts).map(|e| &lw.experts[&e]).collect();
    let e = F4Experts::upload(&experts, Wiring::Correct);
    let routed = Dispatch::reference(cfg, &e, &x, &wexpert).run();

    let shared = o.expert(&lw.shared, &x, 1, None, &mut Counters::default());
    let got: Vec<f32> = routed
        .iter()
        .zip(&shared)
        .map(|(a, b)| bf16_decode(bf16_encode(a + b)))
        .collect();
    assert_matches(
        cap.float("L0.pre.ffn_out").expect("ffn_out golden"),
        &got,
        "ffn_out",
    );
}

// =======================================================================================
// 7. the resident fp8 SHARED expert, and the clamp it did not have
// =======================================================================================
//
// `MoE.__init__` hands `swiglu_limit` to `shared_experts` as well as to the routed ones
// (model.py:632) and `Expert.forward` clamps both. Until 2026-08-05 the shared expert's only
// available combine was `launch_swiglu`, which is GLM's, is unclamped, and rounds nothing —
// `Defect::SwigluUnclamped` on one contribution in seven of every one of the 43 layers,
// fluent and wrong. `launch_swiglu_clamped_bf16` is the fix and this section is its gate.
//
// It gates the KERNEL, not the engine. As of 2026-08-05 the shared expert has no caller — the
// V4 layer loop is being written separately — so what these tests establish is that the
// clamped combine is correct and available, NOT that anything uses it. The wiring, and a gate
// saying the caller reached for this launcher rather than for GLM's `launch_swiglu`, are owed
// by that loop.
//
// # What separates a clamped kernel from an unclamped one, and where
//
// Nowhere, at ordinary activation scales. `swiglu_limit` is 10.0 and the toy weights put
// `|w1·x|` and `|w3·x|` around 1, so a clamp test built on the natural fixture would pass
// against a kernel with no clamp at all. That is not a hypothetical failure mode — it is the
// one this repo has shipped five times, and `the_swiglu_clamp_is_live_and_the_fixture_reaches
// _it` above exists because of it on the fp4 side.
//
// So the fixture's reachability is MEASURED, from the oracle's own `swiglu_clamp_events`,
// which is counted while the oracle computes and is independent of what any kernel did. Both
// ends of the bracket are asserted: at scale 1 the count is zero and the clamp must be
// BIT-INERT, at scale 48 the count is positive and the clamp must change the answer.

/// The shared expert's three fp8 weights on device — `(e4m3 bytes, f32 block scales)` per
/// projection, in `[w1, w2, w3]` order.
///
/// The scale bytes are widened here through `e8m0_decode`, which is what
/// `format.rs::copy_fp8_e8m0` does at conversion and is exact: every e8m0 code is a power of
/// two. The oracle dequantizes the SAME bytes on its side through `WMat::row`, so a
/// disagreement between the two decoders is inside what this comparison covers.
///
/// Matched exhaustively rather than with a let-else: the shared expert being fp8 is the whole
/// reason it needs its own launcher, and a `.f4` block reaching here would be the wrong
/// ARITHMETIC rather than the wrong bytes — the distinction `launch_moe_expert_range_f4`'s
/// `n_desc` doc draws.
fn upload_fp8_shared(e: &ExpertW) -> Vec<(DeviceBuf, DeviceBuf)> {
    [&e.w1, &e.w2, &e.w3]
        .into_iter()
        .map(|m| match m {
            WMat::Fp8 { w, s, .. } => {
                let widened: Vec<f32> = s.iter().map(|&c| e8m0_decode(c)).collect();
                (to_device(w), to_device(&f32b(&widened)))
            }
            WMat::Dense { .. } | WMat::Fp4 { .. } => {
                panic!(
                    "the shared expert is fp8 e4m3 at 128x128 — `MoE.__init__` passes \
                        `expert_dtype` only to the ROUTED experts"
                )
            }
        })
        .collect()
}

/// One row of the resident fp8 shared expert on the GPU: three `gemv_fp8_bf16` and the clamped
/// combine, `Expert.forward` with `weights = None`.
///
/// The launch order IS the arithmetic, so it is spelled rather than abstracted:
/// `act_quant(x)` once (both `w1` and `w3` read the identical quantized row — the reference
/// runs a separate `act_quant` inside each `Linear`, on the same row at the same block, so
/// the bytes are identical), then `w1`/`w3`, then the combine, then `act_quant(h)` and `w2`.
/// Every `gemv_fp8_bf16` bf16-rounds its own output, which is where `Linear`'s bf16 store lives.
///
/// `limit` is a parameter and not `cfg.swiglu_limit` because the whole test below is an A/B
/// on it. There is no way to ask for the unclamped form: the launcher refuses `<= 0`, NaN
/// and `+/-inf`
/// and NaN, so "effectively unclamped" has to be spelled as a huge positive limit.
fn gpu_shared_expert(
    cfg: &V4Config,
    w: &[(DeviceBuf, DeviceBuf)],
    x: &[f32],
    limit: f32,
) -> Vec<f32> {
    let (dim, inter) = (cfg.dim, cfg.moe_inter_dim);
    let stream = HipStream::new().expect("hip stream");
    let (st, blk) = (stream.raw(), 128usize);
    let mut xq = to_device(&f32b(x));
    let mut g = zeros(inter * 4);
    let mut u = zeros(inter * 4);
    let mut out = zeros(dim * 4);
    let sc = |i: usize| w[i].1.ptr().cast::<f32>();
    // SAFETY: `xq` is one row of `dim` f32; `g`/`u` are `inter`; `out` is `dim`; each weight
    // is `[o_dim, i_dim]` e4m3 with a 128x128 f32 scale grid by `upload_fp8_shared`'s
    // contract. All five outlive the `device_sync` inside `sync_f32`.
    unsafe {
        launch_act_quant_f8_prefix(xq.ptr().cast(), xq.ptr_mut().cast(), 1, dim, dim, blk, st)
            .expect("act_quant x");
        let xp = xq.ptr().cast::<f32>();
        launch_gemv_fp8_bf16(
            xp,
            w[0].0.ptr(),
            sc(0),
            1,
            inter,
            dim,
            blk,
            1,
            g.ptr_mut().cast(),
            st,
        )
        .expect("w1");
        launch_gemv_fp8_bf16(
            xp,
            w[2].0.ptr(),
            sc(2),
            1,
            inter,
            dim,
            blk,
            1,
            u.ptr_mut().cast(),
            st,
        )
        .expect("w3");
        // IN PLACE into `g`: `h` becomes `w2`'s input, which is one fewer allocation and is
        // how `gpu.rs` already drives GLM's `swiglu`. Safe by the kernel's own note.
        launch_swiglu_clamped_bf16(
            g.ptr().cast(),
            u.ptr().cast(),
            inter,
            limit,
            g.ptr_mut().cast(),
            st,
        )
        .expect("clamped swiglu");
        launch_act_quant_f8_prefix(g.ptr().cast(), g.ptr_mut().cast(), 1, inter, inter, blk, st)
            .expect("act_quant h");
        launch_gemv_fp8_bf16(
            g.ptr().cast(),
            w[1].0.ptr(),
            sc(1),
            1,
            dim,
            inter,
            blk,
            1,
            out.ptr_mut().cast(),
            st,
        )
        .expect("w2");
    }
    sync_f32(&out)
}

/// `swiglu_clamp_events` for one shared-expert call at `defect`, and the oracle's answer.
///
/// The count comes from the oracle rather than from anything the kernel reports, which is
/// what makes the reachability claims below measurements instead of hopes.
fn oracle_shared(defect: Defect, x: &[f32], layer: usize) -> (Vec<f32>, usize) {
    let (cfg, m, _) = fixture();
    let o = Oracle::new(cfg.clone(), defect);
    let mut c = Counters::default();
    let y = o.expert(&m.layers[layer].shared, x, 1, None, &mut c);
    (y, c.swiglu_clamp_events)
}

/// The shared expert at an activation scale that NEVER reaches the clamp.
///
/// Two claims, and the second is the one a clamp test usually forgets. The fp8 path matches
/// the oracle; and where the oracle says the bound never binds, the clamp is **bit-inert** —
/// `limit = 10` and `limit = 1e6` produce identical bit patterns. That is the half of
/// "the clamp must separate exactly where it should and nowhere else" which says *nowhere
/// else*, and without it a kernel that clamped at the wrong threshold, or clamped `up` from
/// the wrong side, could still pass the positive gate below.
#[test]
fn the_shared_expert_matches_the_oracle_where_the_clamp_never_binds() {
    let (cfg, m, _) = fixture();
    let x = draw_x("shared-x", cfg.dim, 1.0);
    let (want, events) = oracle_shared(Defect::None, &x, 0);
    assert_eq!(
        events, 0,
        "this case is the UNCLAMPED half of the bracket — pick a lower scale"
    );
    assert!(
        want.iter().any(|v| v.abs() > 1e-6),
        "the oracle produced nothing to compare"
    );
    let w = upload_fp8_shared(&m.layers[0].shared);
    let got = gpu_shared_expert(cfg, &w, &x, cfg.swiglu_limit);
    assert_matches(&want, &got, "shared expert (fp8), clamp not binding");
    assert_eq!(
        bits(&got),
        bits(&gpu_shared_expert(cfg, &w, &x, 1e6)),
        "the clamp changed the answer on a case where the oracle counted ZERO clamp events, \
         so it is binding somewhere it must not — a wrong threshold, or `up` clamped on the \
         wrong side"
    );
}

/// The clamped SwiGLU on the shared expert, with the fixture measured to reach it.
///
/// Four arms, and each is here because it rules out a way the other three could be green
/// against a wrong kernel:
///
/// 1. **The fixture reaches the clamp**, from the oracle's own event count. Without this the
///    remaining three compare a clamp that never fired.
/// 2. **The kernel matches the clamped oracle.** The positive gate.
/// 3. **`limit = 1e6` disagrees with the clamped oracle**, so the clamp is what separates
///    them at these inputs — and it must exceed the same [`TOL`] the positive arm passes at,
///    which is what [`assert_disagrees`] enforces.
/// 4. **`limit = 1e6` MATCHES the oracle running `Defect::SwigluUnclamped`.** This is the arm
///    that makes 3 mean something: it says the break is *precisely* the unclamped form rather
///    than some unrelated perturbation that happens to move the answer. A "break" that moved
///    the result for the wrong reason would pass 3 and fail this.
///
/// The asymmetry gets its own test below; it is not visible from here.
#[test]
fn the_shared_expert_clamp_is_live_and_the_fixture_reaches_it() {
    let (cfg, m, _) = fixture();
    let x = draw_x("shared-x-big", cfg.dim, 48.0);
    let (want, events) = oracle_shared(Defect::None, &x, 0);
    assert!(
        events > 0,
        "the fixture never reaches `swiglu_limit`, so this test could not distinguish a \
         clamped kernel from an unclamped one — raise the activation scale"
    );
    println!("shared expert: {events} clamp events at scale 48");
    let w = upload_fp8_shared(&m.layers[0].shared);
    assert_matches(
        &want,
        &gpu_shared_expert(cfg, &w, &x, cfg.swiglu_limit),
        "clamped shared",
    );

    // Effectively unclamped, but POSITIVE — the launcher refuses 0 and NaN outright, which is
    // the stronger guarantee and the reason this arm goes the long way round.
    let unclamped = gpu_shared_expert(cfg, &w, &x, 1e6);
    assert_disagrees(
        &want,
        &unclamped,
        "shared expert with the limit raised to 1e6",
    );
    // No assertion that the defect oracle counted ZERO clamp events, and an earlier draft had
    // one. It could not fail: `Oracle::expert` sets `limit = 0.0` for this defect and both
    // increments of `swiglu_clamp_events` sit inside `if limit > 0.0`, so the count is
    // structurally zero. A guard that nothing could make red is what this port has shipped
    // five times, and three of the most recent four were added in answer to a review. The
    // claim it was reaching for — that the defect really disables the clamp — is what the
    // comparison on the next line says, from the numbers rather than from a counter.
    let (want_unclamped, _) = oracle_shared(Defect::SwigluUnclamped, &x, 0);
    assert_matches(
        &want_unclamped,
        &unclamped,
        "1e6 reproduces Defect::SwigluUnclamped",
    );
}

/// The clamp is ASYMMETRIC — `up` on both sides, `gate` only from above (model.py:606-607) —
/// and **this test CANNOT gate that. Measured, not assumed, and it is a property of the
/// reference rather than of the fixture.**
///
/// The plausible wrong version is `Defect::SwigluClampGateBothSides`: clamping the gate from
/// below too is one `fmaxf` and reads as a tidier symmetry. This test was written to reject it
/// through the expert and it does not, which was caught on the GPU on 2026-08-05 by the
/// test's own anti-vacuity arm rather than by a reviewer:
///
/// ```text
/// shared expert: 12 gate values below -10 at scale 48
/// the two clamp shapes on this fixture: err=3.125e-2  tol=9.424e-2   (max |want| = 1.206e1)
/// ```
///
/// So the fixture DOES reach the case — 12 elements of it — and the two clamp shapes still
/// agree to a third of the tolerance. Raising the activation scale cannot fix that, and the
/// reason is a closed-form bound rather than a fixture accident. For `g <= -limit` the
/// asymmetric form computes `silu(g)` and the symmetric one computes `silu(-limit)`, so the
/// difference is at most
///
/// ```text
/// |silu(g) - silu(-10)| <= |silu(-10)| = 4.540e-4      (silu -> 0 as g -> -inf)
/// ```
///
/// times `|up| <= limit`, i.e. **4.540e-3 per ELEMENT of `h`**. `silu` has already annihilated
/// the operand before the lower clamp could matter.
///
/// **The per-element bound is what is proved; the observed 3.125e-2 is not it.** That figure is
/// max-abs on the expert's OUTPUT, after `w2` accumulates over `moe_inter_dim = 128` — 6.9x the
/// per-element bound, which is accumulation, not a contradiction. Against a 9.424e-2 tolerance
/// the live margin is **3x**, and the per-element bound sits **20x** under it. An earlier draft
/// said "two orders below resolution", which is 1.3 orders for the bound and half an order for
/// the thing actually measured.
///
/// **And the scale claim is narrower than first written.** An earlier draft said the
/// non-separation held "at any activation scale" and was "provably impossible" to escape. The
/// bound is per-element and scale-free, but the NUMBER of affected elements grows with scale
/// while the tolerance saturates — `gt <= 10` and `|ut| <= 10` cap `h`, so `max|want|` stops
/// growing. 12 of 128 elements are affected at scale 48; at a high enough scale most of `inter`
/// would be, and the sum could clear `TOL`. So: **measured unresolvable at this fixture's
/// scale**, with a per-element bound explaining why pushing a little harder will not help. Not
/// a proof over all scales.
///
/// That is still a fact about `Expert.forward` worth recording: at `swiglu_limit = 10` the
/// gate's missing lower clamp is very nearly a no-op numerically. It is worth matching — the
/// reference is the spec — but it cannot be gated HERE.
///
/// **Where it IS gated:** [`the_clamped_combine_is_bit_exact_elementwise`], which compares the
/// kernel to a host transliteration BIT FOR BIT and carries an explicit symmetric-clamp arm.
/// At the combine there is no `w2` accumulation to bury a 4.540e-3 term in and no tolerance to
/// hide under: `silu(-12)` and `silu(-10)` are `-7.373e-5` and `-4.540e-4`, which are nowhere
/// near each other in bf16. Resolution is what moved, not the claim.
///
/// The pattern is `tests/kvcompress_kernel.rs`'s: record the measured separation, say plainly
/// which metric cannot resolve it, and name the instrument that can — rather than widening
/// [`TOL`] until a green appears, which here would have meant a 3x loosening that every other
/// test in this file pays for.
/// **This test still GATES the asymmetric clamp positively.** `assert_matches(&asym, &got)`
/// below is the check that the kernel implements the reference's clamp, and deleting this test
/// would lose it. Only the *rejection* of the symmetric variant moved elsewhere; the name is
/// long because it warns about the half that moved, not because the rest is a no-op.
#[test]
fn the_shared_expert_gate_clamp_matches_and_the_asymmetry_is_below_resolution() {
    let (cfg, m, _) = fixture();
    let x = draw_x("shared-x-big", cfg.dim, 48.0);
    let (asym, events) = oracle_shared(Defect::None, &x, 0);
    let (sym, events_sym) = oracle_shared(Defect::SwigluClampGateBothSides, &x, 0);
    // The fixture reaches the case: each `gi < -limit` is exactly one clamp event the
    // asymmetric form does not count, so the DIFFERENCE is the population size.
    assert!(
        events_sym > events,
        "the fixture has no gate value below -{}, so the measurement below would be reporting \
         an empty case rather than an unresolvable one ({events_sym} events vs {events})",
        cfg.swiglu_limit
    );
    println!(
        "shared expert: {} gate values below -{} at scale 48",
        events_sym - events,
        cfg.swiglu_limit
    );
    // The positive gate still stands, and it is the point of the test: the kernel reproduces
    // the ASYMMETRIC oracle.
    let w = upload_fp8_shared(&m.layers[0].shared);
    let got = gpu_shared_expert(cfg, &w, &x, cfg.swiglu_limit);
    assert_matches(&asym, &got, "shared expert, gate clamped from above only");

    // And the recorded non-separation, asserted so it cannot silently become a separation
    // nobody noticed. `compare` prints both numbers either way.
    //
    // Note what is and is not pinned. The 4.540e-3 figure in the doc is a bound on ONE
    // ELEMENT of `h`; the quantity asserted here is max-abs error on the expert's OUTPUT,
    // after `w2` has accumulated over `moe_inter_dim = 128`. Those are different quantities
    // and the measured 3.125e-2 is 6.9x the per-element bound, which is not a contradiction —
    // it is the accumulation. An earlier message here claimed the bound made a larger value
    // "impossible", which was false against the number printed on the very next line.
    let (err, tol) = compare(
        &asym,
        &sym,
        "the two clamp shapes (recorded as UNRESOLVABLE)",
    );
    assert!(
        err <= tol,
        "the asymmetric and symmetric clamps now differ by err={err:.3e} > tol={tol:.3e} \
         through the expert, against a recorded 3.125e-2. This test is a recorded \
         NON-separation, so a red here means the recording is stale — re-measure before \
         treating it as a new gate, and note the per-element bound in the doc does NOT govern \
         this post-w2 quantity"
    );
}

/// `swiglu_clamped_bf16` elementwise against a host transliteration, BIT FOR BIT.
///
/// Bitwise is legitimate here and nowhere else in this file: the kernel is one thread per
/// element with no reduction, so there is no summation order to diverge from and none of
/// `assert_close`'s justification applies. `docs/investigations/v4-flash-port.md`
/// §"`assert_close` over bitwise at real dims" retracted a bitwise gate over a WAVE-REDUCED
/// kernel at dim 4096; that argument is about the reduction, not about elementwise ops.
///
/// The inputs are adversarial by construction rather than by luck. `expf` is the only
/// transcendental and HIP's need not agree with Rust's to the last bit, so a disagreement
/// here is reported with the offending element rather than swept into a tolerance — if this
/// ever goes red on `expf` alone, the right response is to say so, not to widen it.
#[test]
fn the_clamped_combine_is_bit_exact_elementwise() {
    let limit = 10.0f32;
    // Straddling the bound on both sides and at it, plus the values that make an asymmetric
    // clamp distinguishable from a symmetric one and a bf16-first clamp from a clamp-first
    // one: `10.001` rounds DOWN to 10.0 in bf16 (bf16 has 8 mantissa bits, so the codes near
    // 10 are 2^-5 = 0.03125 apart), so clamping before the round and after it differ on it.
    let probes: Vec<f32> = vec![
        0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 9.9, -9.9, 9.999, -9.999, 10.0, -10.0, 10.001, -10.001,
        10.0625, -10.0625, 12.0, -12.0, 40.0, -40.0, 1e3, -1e3,
    ];
    let (mut g, mut u) = (Vec::new(), Vec::new());
    for &a in &probes {
        for &b in &probes {
            g.push(a);
            u.push(b);
        }
    }
    let n = g.len();
    // The host side, written as the reference reads: bf16 both, clamp `up` both sides, `gate`
    // per `gate_clamp`, `F.silu`'s MULTIPLY form, bf16 the product.
    //
    // ONE definition, taking the gate clamp as a function, because the two arms below differ in
    // exactly that expression and nothing else — which is also precisely the difference between
    // model.py:607 and the tidier wrong version. Two near-copies said the same thing worse, and
    // jscpd refused them (59 tokens).
    let combine = |gate_clamp: &dyn Fn(f32) -> f32| -> Vec<f32> {
        g.iter()
            .zip(&u)
            .map(|(&gv, &uv)| {
                let gt = gate_clamp(bf16_decode(bf16_encode(gv)));
                let ut = bf16_decode(bf16_encode(uv)).clamp(-limit, limit);
                bf16_decode(bf16_encode(silu(gt) * ut))
            })
            .collect()
    };
    // `torch.clamp(gate, max=self.swiglu_limit)` — ABOVE only. The reference.
    let want = combine(&|x: f32| x.min(limit));
    let (gb, ub) = (to_device(&f32b(&g)), to_device(&f32b(&u)));
    let mut hb = zeros(n * 4);
    let stream = HipStream::new().expect("hip stream");
    // SAFETY: three live `n`-element f32 buffers, outliving the sync in `sync_f32`.
    unsafe {
        launch_swiglu_clamped_bf16(
            gb.ptr().cast(),
            ub.ptr().cast(),
            n,
            limit,
            hb.ptr_mut().cast(),
            stream.raw(),
        )
    }
    .expect("swiglu_clamped_bf16");
    let got = sync_f32(&hb);
    // **THE ASYMMETRY GATE's anti-vacuity arm, and it is HOST vs HOST.** The same probes under
    // a SYMMETRIC gate clamp — the plausible wrong version, and the one
    // `the_shared_expert_gate_clamp_matches_and_the_asymmetry_is_below_resolution` measured itself
    // unable to see through the expert.
    //
    // `sym` against `want`, NOT against `got`, and the difference decides what a failure
    // MEANS. Comparing the symmetric host arm to the DEVICE output conflates two causes: a
    // narrowed probe table, and a kernel that genuinely is symmetric. The second is the whole
    // defect this arm exists to catch, and it would have been reported as "the probe table has
    // no gate value below -10" — sending the next reader to fix the fixture. Removing the
    // kernel from the comparison makes the claim unambiguous: *these two host functions differ
    // on this input set*, therefore a kernel can be asked which one it implements. That is the
    // same oracle-vs-oracle rule the relocated test's own history produced, applied here.
    //
    // The kernel-facing half is the bitwise `want` vs `got` check below, which is where a
    // symmetric kernel actually goes red.
    let sym = combine(&|x: f32| x.clamp(-limit, limit));
    let moved = sym
        .iter()
        .zip(&want)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(
        moved > 0,
        "the symmetric and asymmetric gate clamps are BIT-IDENTICAL over {} probe pairs, so \
         no comparison against them could tell the reference's clamp from the tidier wrong \
         one — the probe table has no gate value below -{limit}",
        g.len()
    );
    println!(
        "asymmetric vs symmetric gate clamp: {moved}/{} probe pairs differ",
        g.len()
    );
    // An EDIT TRIPWIRE, not a measurement, and worth being plain about which: `probes` is a
    // literal fifteen lines up, so this folds to a constant on today's tree and can only go
    // red if someone narrows that table. That is exactly the change it is here to stop — a
    // probe set inside the bound would compare two functions the clamp never touches — but it
    // is not evidence about anything the kernel does.
    let clamped_any = g
        .iter()
        .zip(&u)
        .any(|(&a, &b)| a > limit || b.abs() > limit);
    assert!(
        clamped_any,
        "no probe crosses the limit — the table was narrowed"
    );
    if let Some(i) = want
        .iter()
        .zip(&got)
        .position(|(a, b)| a.to_bits() != b.to_bits())
    {
        panic!(
            "element {i} differs: g={:?} u={:?} want={:?} ({:#010x}) got={:?} ({:#010x})",
            g[i],
            u[i],
            want[i],
            want[i].to_bits(),
            got[i],
            got[i].to_bits()
        );
    }
}

/// The new launcher's guards, by CODE — including the one that matters most.
///
/// Two rows matter here and they are the same defect from opposite ends of the float line —
/// both are values that make the clamp VANISH rather than values that make it fail loudly:
///
/// - **NaN**, which a `limit <= 0.0f` guard admits, because every comparison against NaN is
///   false. `fminf(gt, NaN)` returns `gt` (`fminf` returns the non-NaN operand).
/// - **+inf**, which `!(limit > 0.0f)` admits. `fminf(gt, inf)` is `gt` and
///   `fmaxf(ut, -inf)` is `ut`, so the clamp is simply gone.
///
/// The guard is therefore `!(limit > 0.0f && limit < INFINITY)` in `kernels/linalg.hip`, which
/// is the two-sided spelling — an earlier draft of this doc quoted the one-sided
/// `!(limit > 0.0f)`, which is what let the `+inf` row be missing in the first place. Code
/// 1006 is deliberately the same one `moe.hip`'s fp4 launcher returns for the same check on
/// the same argument, and that launcher had the identical hole.
#[test]
fn swiglu_clamped_bf16_guards() {
    let mut b = zeros(64);
    let (p, pm) = (b.ptr().cast::<f32>(), b.ptr_mut().cast::<f32>());
    let nul = std::ptr::null_mut();
    // `null_mut()` for the stream: every case is rejected before `hipLaunchKernelGGL`, so
    // there is no launch for a stream to order.
    // SAFETY: each call returns at an argument guard, before any pointer is read.
    assert_guards(unsafe {
        vec![
            (
                1001,
                "zero elements",
                launch_swiglu_clamped_bf16(p, p, 0, 10.0, pm, nul).map_err(|e| e.to_string()),
            ),
            (
                1006,
                "an unclamped limit",
                launch_swiglu_clamped_bf16(p, p, 16, 0.0, pm, nul).map_err(|e| e.to_string()),
            ),
            (
                1006,
                "a negative limit",
                launch_swiglu_clamped_bf16(p, p, 16, -1.0, pm, nul).map_err(|e| e.to_string()),
            ),
            (
                1006,
                "a NaN limit",
                launch_swiglu_clamped_bf16(p, p, 16, f32::NAN, pm, nul).map_err(|e| e.to_string()),
            ),
            // +inf is the case a `!(limit > 0.0f)` guard ADMITS, and it disables the clamp
            // exactly as thoroughly as `limit = 0` would: `fminf(gt, inf)` is `gt`. It sits
            // next to the NaN row deliberately — the two are the same defect from opposite
            // ends of the float line, and having only one of them on the page is how the
            // other stayed open.
            (
                1006,
                "an infinite limit",
                launch_swiglu_clamped_bf16(p, p, 16, f32::INFINITY, pm, nul)
                    .map_err(|e| e.to_string()),
            ),
        ]
    });
}

/// The bit pattern of the fp4 dispatch, printed — the instrument for the 2026-08-05 hoist of
/// the clamp out of `moe_gateup_f4_impl` and into `common.hpp::swiglu_clamped`.
///
/// That hoist had to be arithmetically inert, and "had to be" is not evidence. The hoist is
/// five lines moving across a `__forceinline__` boundary; the association is preserved by
/// hand (`swiglu_clamped` returns `(silu · up)` and the caller applies `· w`, which is the
/// `((silu · up) · w)` the inline form computed), but **FMA contraction is uncontrolled
/// tree-wide** — `build.rs:67` gives hipcc only `--offload-arch -O3 -fPIC` and clang's HIP
/// default is `-ffp-contract=fast` — so a function boundary can change codegen even where the
/// arithmetic is identical.
///
/// So it was measured, by running this test with `moe_gateup_f4_impl` reverted to the inline
/// form and again with the hoist, and comparing the printed hash. The result is in the commit
/// that made the change. This test stays because the hash is the cheapest possible tripwire
/// on any future edit to that helper: it does not assert a value (a hard-coded hash would be
/// a golden tied to one compiler and one GPU), it PRINTS one, next to the oracle comparison
/// that says the value is also correct.
#[test]
fn the_fp4_dispatch_hash_pins_the_clamp_hoist() {
    let c = Case::new(0, "moe-x-big", 48.0);
    assert!(
        c.clamp_events > 0,
        "the hoisted clamp must be exercised by the case that pins it"
    );
    let got = c.gpu();
    // FNV-1a over the bit patterns. Order-sensitive and 64-bit, so two runs that agree here
    // agree element for element; a sum or an XOR would not say that.
    let h = got.iter().fold(0xcbf2_9ce4_8422_2325u64, |a, v| {
        (a ^ u64::from(v.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    });
    println!("fp4 dispatch hash (clamp hoist tripwire): {h:#018x}");
    assert_matches(&c.want, &got, "fp4 dispatch behind the hoisted clamp");
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

/// `(max abs error, tolerance)`, printed side by side.
fn compare(want: &[f32], got: &[f32], label: &str) -> (f32, f32) {
    assert_eq!(want.len(), got.len(), "{label}: length mismatch");
    report_rel(want, got, label, TOL)
}

/// The two must agree within [`TOL`].
fn assert_matches(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = compare(want, got, label);
    assert!(err <= tol, "{label}: err={err:.3e} > tol={tol:.3e}");
}

/// A launcher's error rendered the way [`assert_guards`] matches on it.
///
/// **`{e:#}` and not `{e}`, which is the load-bearing part and the whole reason this is one
/// function.** The guard code lives in the CHAINED context, so `{e}` prints only the
/// outermost message and every code assertion in this file stops matching at once — and
/// `assert_guards` reports that as "want guard 1002, got ...", i.e. as a failing guard rather
/// than as a broken format. Four launcher wrappers ended this way; two of the copies were
/// what `build.rs`'s duplication gate found.
fn guard_err<T, E: std::fmt::Display>(r: Result<T, E>) -> Result<T, String> {
    r.map_err(|e| format!("{e:#}"))
}

/// Assert each launcher `Result` carries the guard CODE it is paired with.
///
/// The code, not `is_err`: a check that accepted any error would still pass if someone
/// replaced a power-of-two test with `block != 128`, or if an unrelated dimension guard
/// started swallowing the case first. Five guard tests share this across six call sites, which
/// is also what stops six copies of the message format from drifting.
fn assert_guards(cases: Vec<(u32, &str, Result<(), String>)>) {
    for (want, case, r) in cases {
        let msg = r.expect_err(case);
        assert!(
            msg.contains(&want.to_string()),
            "{case}: want guard {want}, got {msg:?}"
        );
    }
}

/// The negative of [`assert_matches`]: the two must NOT agree, at the SAME tolerance.
///
/// Every deliberate-break test in this file goes through here rather than through a bare
/// `assert_ne!`, and the shared threshold is what makes the pair meaningful: a break that
/// moved the result by less than [`TOL`] is a break the positive gate would NOT have
/// caught, so it must fail here rather than pass. [`compare`] prints `err` and `tol` for
/// both directions, so how far past the line it landed is on the page either way.
fn assert_disagrees(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = compare(want, got, &format!("{label} (must differ)"));
    assert!(
        err > tol,
        "{label}: err={err:.3e} <= tol={tol:.3e} — the break is INVISIBLE to this gate, so \
         the corresponding positive test proves nothing about it"
    );
}
