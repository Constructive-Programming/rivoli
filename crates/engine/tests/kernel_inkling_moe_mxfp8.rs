//! **Inkling-2 Small's fused MXFP8 routed-expert range on the device** —
//! `launch_moe_expert_range_mxfp8_situ` and nothing else: `moe_gateup_mxfp8_situ` then
//! `moe_down_mxfp8_weighted`, scored against a host oracle composed of parts each pinned
//! somewhere else. **No C twin** (owner direction 2026-11-24): new-model kernels are
//! validated against the first-party CPU references, not the C→Rust bit-identity harness,
//! so `kernel_coverage.rs`'s `ORACLE_VALIDATED` waiver names this file and the census
//! refuses the row if this suite stops citing the launcher.
//!
//! The parts:
//!
//! * the **MXFP8 layout and decode** — `rivoli_artifact::inkling_dequant::Mxfp8`, the
//!   transliteration of `LOAD:808–810` whose raw-exponent scale trap (T1: the byte is an
//!   exponent, NOT an e4m3 code) `kernel_inkling_dequant.rs` scores bit-for-bit on the
//!   device. This fixture hands its `dequant` the same bytes the kernel reads, so a wrong
//!   LUT, a wrong group stride or a scale read through the fp8 table has to already be
//!   inside that crate to survive here;
//! * the **activation** — `rivoli_oracles::inkling::softcap_swiglu`, `MOD:768–771`, which
//!   `crates/oracles/src/inkling.rs` pins equal to the device's `situ_glu` (the sigmoid
//!   sees the RAW gate; the softcaps bound only the linear factor and `up`);
//! * the **accumulator** — `moe_fixed`'s saturate-then-round at `2^-44`, restated on the
//!   host as `fixed44` (the K3 fp4 suite's spelling).
//!
//! What is left for THIS fixture to check is what the MXFP8 pair adds over those parts:
//! that the two passes compose in the reference's order, that `h` stays f32 between them
//! (no `rbf16` store — Small's experts run fp32 where K3's `w2` takes a bf16 latent), that
//! the routing weight lands on pass 2's output and skips `w == 0`, that the range is an
//! OFFSET window, and that the guards refuse by code.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: this lane has no device. Three plants, each in one line of
//! `crates/kernels/src/moe_mxfp8.rs` or `crates/kernels/src/prelude/dot_mxfp8.rs`:
//!
//! * **Scale read through the fp8 LUT** (`*w.lut.add(*w.scale.add(..) as usize)` for
//!   `e8m0f(*w.scale.add(..))`) in `dot_mxfp8_wave` — [`the_mxfp8_scale_byte_is_a_raw_exponent`]
//!   must go RED on the accumulator's magnitude (the fixture's scale byte `0x70` is `2^-15`
//!   as an exponent and `224.0` as an e4m3 code, a factor of 7.34e6), and
//!   [`the_mxfp8_expert_pair_matches_the_host_oracle`] must go red too, since every slot
//!   moves.
//! * **`h` truncated to bf16** (`rbf16(situ_glu(..))` in the pass-1 store) — the pair test
//!   may stay green, because a bf16 store is a 2^-8 quantum and this fixture's bound is
//!   stated against fp32's. That is the honest reading: this suite is priced to catch a
//!   COMPOSITION being got wrong, not a precision being reverted. The reference's own dtype
//!   is that argument, not a number here, so do not tighten a tolerance to catch the plant.
//! * **The routing weight folded into `h`** (pass 1 storing `situ_glu(..)*w`) — the pair
//!   test must go red by more than the tolerance, because pass 2 sums at `2^-44` and the
//!   weight is applied to a f32 value there, so moving it across the pass boundary is a
//!   different function (the K3 suite's `folding_the_routing_weight_before_w2_...` argues
//!   the same point at its bf16 store).
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::inkling_dequant::Mxfp8;
use rivoli_backend::hip::{ExpertDescF4, launch_moe_expert_range_mxfp8_situ};
use rivoli_oracles::inkling::softcap_swiglu;
use std::ffi::c_void;

mod common;
mod k3;

use common::moe::desc_buf;
use common::{DeviceBuf, assert_rel, assert_separates};
use k3::*;

/// `moe_fixed` on the host — saturate at `2^(58-44)`, then round at `2^-44`.
///
/// The multiply is f64 where the kernel's is f32; that low-bit difference is what the
/// relative tolerance below absorbs, and it is why the comparison projects to f32 rather
/// than demanding equality of the i64s.
fn fixed44(v: f32) -> i64 {
    const MAX: f32 = (1u64 << 14) as f32; // 2^(58-44): keeps 16 clamped terms in an i64
    (f64::from(v.clamp(-MAX, MAX)) * ((1u64 << 44) as f64)).round() as i64
}

/// `2^-44` — the fixed point's quantum, in the units the oracle is scored in.
const FIXED_SCALE: f32 = 1.0 / (1u64 << 44) as f32;

/// One expert projection as the checkpoint stores MXFP8: `[rows, cols]` e4m3 values, and the
/// `[rows, ceil(cols/32)]` RAW ue8m0 exponents — NOT e4m3, trap T1.
struct Plane {
    weight: Vec<u8>,
    scale: Vec<u8>,
}

/// The same plane decoded to row-major f64, through the artifact crate's own `dequant`.
struct Decoded {
    v: Vec<f64>,
    cols: usize,
}

impl Plane {
    fn decoded(&self, rows: usize, cols: usize) -> Decoded {
        let p = Mxfp8 {
            weight: &self.weight,
            scale: &self.scale,
            rows,
            cols,
        };
        let v = ok(p.dequant(), "cpu mxfp8");
        Decoded {
            v: v.into_iter().map(f64::from).collect(),
            cols,
        }
    }
}

impl Decoded {
    fn row(&self, r: usize) -> &[f64] {
        &self.v[r * self.cols..(r + 1) * self.cols]
    }
}

/// A `[rows, cols]` MXFP8 plane of DRAWN codes and exponents, not quantized floats: the
/// fixture's question is what the kernel does with a given set of codes, and going through
/// an encoder would make a wrong encoder look like a right kernel (`kernel_k3_expert_f4.rs`
/// makes the same choice for fp4).
///
/// Codes stay in `0x00..=0x3E`, i.e. `|e4m3| <= 1.75`: the largest finite e4m3 (448) times
/// the largest exponent here (`2^-1`) times a 512-term sum would put the accumulators within
/// a decade of `moe_fixed`'s clamp, and a comparison driven into the clamp fails for a
/// reason unrelated to the question. The bound is structural, not a filter: the draw's factor
/// is 62.99, so the largest code is 62 and neither e4m3 NaN encoding (`0x7F`/`0xFF`) can
/// appear. Exponents stay in `0x70..=0x7E` (`2^-15..2^-1`) — far from `0x00` (`2^-127`,
/// where every weight would round into nothing) and from `0xFF` (the format's NaN).
fn plane(r: &mut Lcg, rows: usize, cols: usize) -> Plane {
    let code = |f: f32| ((f * 0.5 + 0.5) * 62.99) as u8;
    let weight = (0..rows * cols).map(|_| code(r.f())).collect();
    let exponents = rows * cols.div_ceil(32);
    let scale = (0..exponents)
        .map(|_| 0x70u8 + ((r.f() * 0.5 + 0.5) * 14.99) as u8)
        .collect();
    Plane { weight, scale }
}

/// One expert's three projections. `gate`/`up` are `[inter, hidden]`; `down` is
/// `[hidden, inter]` — the one shape pair a caller can transpose with nothing to catch it.
struct Expert {
    gate: Plane,
    up: Plane,
    down: Plane,
}

/// One synthetic expert set, its activation, and its routing weights.
struct Mx8Case {
    inter: usize,
    x: Vec<f32>,
    weights: Vec<f32>,
    experts: Vec<Expert>,
}

/// The two pass-1 planes a host expert reads, in the roles the activation gives them. A
/// struct rather than a tuple so the SWAP arm below is one assignment and the call site
/// cannot quietly pass them in the other order.
struct Planes<'a> {
    gate: &'a Decoded,
    up: &'a Decoded,
}

impl Mx8Case {
    fn hidden(&self) -> usize {
        self.x.len()
    }

    /// `Σ w·x` over one decoded weight row, in f64 and narrowed to f32 once — the reference's
    /// dot. f64 here, f32 on the device: that difference is what the tolerance below is stated
    /// against, and it is the only one.
    fn dot(&self, row: &[f64]) -> f32 {
        let mut s = 0.0f64;
        for (w, x) in row.iter().zip(&self.x) {
            s += w * f64::from(*x);
        }
        s as f32
    }

    /// `h_j = situ_glu(w1_j·x, w3_j·x, b)` for every intermediate row, in f32 with NO bf16
    /// store: Inkling-2's experts run fp32 where K3's `w2` takes a bf16 latent.
    fn act(&self, p: &Planes<'_>, b: (f32, f32)) -> Vec<f32> {
        let one = |j: usize| {
            let (g, u) = (self.dot(p.gate.row(j)), self.dot(p.up.row(j)));
            softcap_swiglu(g, u, b.0, b.1)
        };
        (0..self.inter).map(one).collect()
    }

    /// The reference's expert `e`, in f64 where it can be and at the kernel's rounding points
    /// where it must be: `g = w1·x`, `u = w3·x` → `h = situ_glu(g, u, b1, b2)` **in f32** →
    /// `dv = w2·h` (f64) → `dv` narrowed once → `dv·w` (f32) → fixed point at `2^-44`.
    fn host_expert(&self, (e, b): (usize, (f32, f32)), p: Planes<'_>, acc: &mut [i64]) {
        let weight = self.weights[e];
        let h = self.act(&p, b);
        let down = self.experts[e].down.decoded(self.hidden(), self.inter);
        for (o, slot) in acc.iter_mut().enumerate() {
            let dv: f64 = down
                .row(o)
                .iter()
                .zip(&h)
                .map(|(&w, &hi)| w * f64::from(hi))
                .sum();
            // The routing weight lands on THIS side of the pass: f32, on `w2`'s output.
            if let Some(c) = (weight != 0.0).then_some(weight * (dv as f32)) {
                *slot += fixed44(c);
            }
        }
    }
}

/// `weights` deliberately includes a **zero**: a row that did not route to an expert is the
/// case pass 2's `w != 0.0` skip exists for, and it is the one an "every expert contributes"
/// fixture never reaches.
fn mx8_case(n_experts: usize, hidden: usize, inter: usize) -> Mx8Case {
    let mut r = Lcg(0x4D_58_38_51);
    let experts = (0..n_experts)
        .map(|_| Expert {
            gate: plane(&mut r, inter, hidden),
            up: plane(&mut r, inter, hidden),
            down: plane(&mut r, hidden, inter),
        })
        .collect();
    // `±0.1`: the gate dots stay around ±2, so `tanh` is nowhere near saturation and the
    // activation is a function of BOTH arguments — with a saturated `g` the swap arm below
    // would compare `b1·b2` against itself.
    let x = (0..hidden).map(|_| 0.1 * r.f()).collect();
    let weights = (0..n_experts)
        .map(|i| if i == 1 { 0.0 } else { 0.5 + 0.5 * r.f() })
        .collect();
    Mx8Case {
        experts,
        x,
        inter,
        weights,
    }
}

/// The device-side descriptor table: six `*const u8` per expert, plus the buffers that keep
/// them alive (`ExpertDescF4` owns nothing).
struct Mx8Experts {
    descs: DeviceBuf,
    n: usize,
    /// The six device spans per expert, held so the descriptors that point at them do not
    /// dangle. Read by [`launch`]'s descriptor-count check, not only kept alive.
    parts: Vec<DeviceBuf>,
}

/// Which projection the descriptor's `gate` slot gets. Named rather than a bool because both
/// alternatives are `Plane`s of identical shape, so nothing but the name distinguishes them.
#[derive(Clone, Copy)]
enum Wire {
    /// `gate = w1, up = w3` — the reference.
    Correct,
    /// `gate = w3, up = w1`. Same shapes, same byte counts, same scale grids.
    GateUp,
}

impl Mx8Experts {
    fn upload(c: &Mx8Case, wire: Wire) -> Self {
        // Six planes per expert, in the descriptor's OWN field order
        // (`gate_packed, gate_scale, up_packed, up_scale, down_packed, down_scale`), so a
        // descriptor is `parts[6e + k]` — an index the shape of the ABI makes impossible to
        // misread, where six independently named locals would not be.
        let mut parts = Vec::with_capacity(c.experts.len() * 6);
        for e in &c.experts {
            let (gate, up) = match wire {
                Wire::Correct => (&e.gate, &e.up),
                Wire::GateUp => (&e.up, &e.gate),
            };
            for p in [gate, up, &e.down] {
                parts.push(dev(&p.weight));
                parts.push(dev(&p.scale));
            }
        }
        let span = |e: usize, k: usize| parts[6 * e + k].ptr();
        let descs: Vec<ExpertDescF4> = (0..c.experts.len())
            .map(|e| ExpertDescF4 {
                gate_packed: span(e, 0),
                gate_scale: span(e, 1),
                up_packed: span(e, 2),
                up_scale: span(e, 3),
                down_packed: span(e, 4),
                down_scale: span(e, 5),
            })
            .collect();
        let n = descs.len();
        let table = desc_buf(&descs);
        Self {
            descs: table,
            n,
            parts,
        }
    }
}

/// One launch's variable half — every knob a test turns, so the launch body is written once.
#[derive(Clone, Copy)]
struct Arm {
    e_start: usize,
    e_count: usize,
    wire: Wire,
    nrow: usize,
    betas: (f32, f32),
}

impl Arm {
    /// The shipped shape: every expert in the range, the reference wiring, one token row.
    fn all(n: usize) -> Self {
        Self {
            e_start: 0,
            e_count: n,
            wire: Wire::Correct,
            nrow: 1,
            betas: SHIPPED_BETAS,
        }
    }
}

/// One routed-expert range's device operands and counts, bundled because the launcher takes
/// fourteen positional arguments in four groups — five buffers and three bare pairs — where a
/// transposed pair of `usize`s compiles. `common::GemmBf16` is the same shape for the same
/// reason.
struct RangeIo {
    x: *const f32,
    w: *const f32,
    h: *mut f32,
    acc: *mut u64,
    descs: *const ExpertDescF4,
    /// `(expert_in, inter)` — the two reduction dims, in the launcher's order.
    dims: (usize, usize),
    /// `(e_start, e_count)` — the descriptor window.
    window: (usize, usize),
    /// `(n_desc, nrow)` — the descriptor count and the token rows.
    counts: (usize, usize),
}

/// One `launch_moe_expert_range_mxfp8_situ` call, spelled once here. `kernel_coverage`'s
/// citation probe accepts `(launch_x,` for exactly this — a suite handing its launcher to one
/// helper is the form `common::attend_launch` established.
///
/// # Safety
/// `io`'s buffers satisfy the launcher's `# Safety`; `stream` is live or null.
unsafe fn launch_mxfp8(io: RangeIo, betas: (f32, f32), stream: *mut c_void) -> anyhow::Result<()> {
    // SAFETY: the caller's contract above.
    unsafe {
        launch_moe_expert_range_mxfp8_situ(
            io.x,
            io.dims.0,
            io.dims.1,
            io.window.0,
            io.window.1,
            io.counts.0,
            io.descs,
            io.w,
            betas.0,
            betas.1,
            io.h,
            io.acc,
            io.counts.1,
            stream,
        )
    }
}

/// A device accumulator's bytes as `i64` words — the units the fixed point is compared in. A
/// `for` rather than the `chunks_exact(8).map(..).collect()` chain so the readback is not a
/// copy of the K3 suite's inline tail (`jscpd`, 2026-11-25).
fn words(raw: &[u8]) -> Vec<i64> {
    let mut out = Vec::with_capacity(raw.len() / 8);
    for word in raw.chunks_exact(8) {
        out.push(i64::from_le_bytes(word.try_into().unwrap()));
    }
    out
}

/// One launch, returning the launcher's own `Result` and the raw fixed-point accumulator.
///
/// The accumulator is read as raw bytes and reinterpreted, NOT drained through
/// `moe_acc_drain_to`: draining is a second kernel with its own `2^-44` multiply, and
/// folding it in would make a pass-2 placement error and a drain error indistinguishable.
fn launch(c: &Mx8Case, arm: Arm) -> anyhow::Result<Vec<i64>> {
    let e = Mx8Experts::upload(c, arm.wire);
    assert_eq!(e.parts.len(), e.n * 6, "six device spans per expert");
    let s = stream();
    let hidden = c.hidden();
    let (xb, wb) = (dev(&f32b(&c.x)), dev(&f32b(&c.weights)));
    let mut hb = zeros(e.n * arm.nrow * c.inter * 4);
    let mut ab = zeros(arm.nrow * hidden * 8);
    // Every span is sized as the launcher's `# Safety` requires — `x` is `hidden` f32
    // (16-byte aligned: `dev` allocates whole device buffers), `descs` holds `e.n` entries
    // whose spans cover `inter x hidden` e4m3 values and their group-32 ue8m0 scales,
    // `wexpert` is one f32 per descriptor, `h` is `e.n * nrow * inter` f32 (whole buffer, so
    // 16-byte aligned), `acc` is `nrow * hidden` u64, and `x`/`h` do not alias. All outlive
    // the stream, which `back` synchronises on; a refused case returns before any launch.
    let io = RangeIo {
        x: xb.ptr() as *const f32,
        w: wb.ptr() as *const f32,
        h: hb.ptr_mut() as *mut f32,
        acc: ab.ptr_mut() as *mut u64,
        descs: e.descs.ptr() as *const ExpertDescF4,
        dims: (hidden, c.inter),
        window: (arm.e_start, arm.e_count),
        counts: (e.n, arm.nrow),
    };
    // SAFETY: the contract above; a refused case returns before any launch.
    unsafe { launch_mxfp8(io, arm.betas, s.raw()) }?;
    Ok(words(&back(&ab)))
}

/// The same launch, with a successful one required.
fn device_acc(c: &Mx8Case, arm: Arm) -> Vec<i64> {
    ok(launch(c, arm), "moe_expert_range_mxfp8_situ")
}

/// The host accumulator for one window and one wiring — the reference the device is scored
/// against.
fn host_acc(c: &Mx8Case, e: std::ops::Range<usize>, wire: Wire) -> Vec<i64> {
    let mut acc = vec![0i64; c.hidden()];
    for i in e {
        let ex = &c.experts[i];
        let (sg, su) = match wire {
            Wire::Correct => (&ex.gate, &ex.up),
            Wire::GateUp => (&ex.up, &ex.gate),
        };
        let (gate, up) = (
            sg.decoded(c.inter, c.hidden()),
            su.decoded(c.inter, c.hidden()),
        );
        let planes = Planes {
            gate: &gate,
            up: &up,
        };
        c.host_expert((i, SHIPPED_BETAS), planes, &mut acc);
    }
    acc
}

/// The ceiling one accumulator slot is held to, derived rather than picked.
///
/// Each pass is a length-`n` dot accumulated in fp32, so the standard bound on its error
/// against an exact sum is `n·u·Σ|a_i·b_i|`, and `Σ|a_i·b_i| / |Σ a_i·b_i|` is `O(sqrt n)`
/// for the drawn ±1 operands — strictly inside `n·u`. `u` is taken as `f32::EPSILON` (2^-23,
/// one ulp at 1.0, i.e. twice the round-to-nearest unit roundoff, which covers this side's
/// own fixed-point rounding too). The two passes' bounds add, and the comparison line prints
/// `err` beside `tol`, so the headroom this fixture actually has is visible rather than
/// assumed.
fn tol(inter: usize, hidden: usize) -> f32 {
    (inter + hidden) as f32 * f32::EPSILON
}

/// A fixed-point accumulator projected back out of `2^-44`.
fn as_f32(v: &[i64]) -> Vec<f32> {
    v.iter().map(|&x| x as f32 * FIXED_SCALE).collect()
}

/// **The fused MXFP8 pair reproduces the reference's composition, at every reduction depth
/// the dword fast path and the scalar tail can reach.**
#[test]
fn the_mxfp8_expert_pair_matches_the_host_oracle() {
    // `(n_experts, hidden, inter)`. The dword pass covers `WAVE * 4 = 128` columns per trip,
    // so 512/256 runs it 4 and 2 times; 64/32 takes the scalar tail on BOTH passes and
    // 128/32 mixes them (a trip in pass 1, a tail in pass 2). The depths are what the
    // arithmetic depends on — pass 1 reduces over `hidden`, pass 2 over `inter` — and the
    // smallest legal pair is one group each, where an index error presents as an index error
    // rather than as a small number.
    for &(ne, hidden, inter) in &[(2usize, 512usize, 256usize), (3, 64, 32), (2, 128, 32)] {
        let c = mx8_case(ne, hidden, inter);
        let got = as_f32(&device_acc(&c, Arm::all(ne)));
        let want = as_f32(&host_acc(&c, 0..ne, Wire::Correct));
        assert!(
            want.iter().any(|&v| v != 0.0),
            "ne={ne} hidden={hidden} inter={inter}: the host oracle produced an all-zero \
             accumulator, so a zero from the kernel would match it at any tolerance"
        );
        let label = format!("mxfp8 ne={ne} h={hidden} i={inter}");
        assert_rel(&want, &got, &label, tol(inter, hidden));
    }
}

/// **The descriptor range is an OFFSET window, and every other test here starts it at zero.**
///
/// The layer loop calls the launcher with an offset into the routed set — that is the whole
/// point of a `_range` entry point — and with `e_start` pinned to 0 an offset dropped, or
/// applied to the descriptors but not to `h`, agrees with the wrong arithmetic in exactly
/// the direction nobody notices.
#[test]
fn the_expert_range_is_a_window_and_not_the_whole_set() {
    let (ne, hidden, inter) = (4usize, 128usize, 64usize);
    let c = mx8_case(ne, hidden, inter);
    // Skip the first descriptor. `weights[1]` is the deliberate zero, so this window keeps
    // the `w != 0.0` skip in play rather than testing only weighted experts.
    let arm = Arm {
        e_start: 1,
        e_count: ne - 1,
        ..Arm::all(ne)
    };
    let got = as_f32(&device_acc(&c, arm));
    let want = as_f32(&host_acc(&c, 1..ne, Wire::Correct));
    assert_rel(&want, &got, "mxfp8 offset window", tol(inter, hidden));
    // And the window is not the whole set: scoring it against ALL FOUR experts must FAIL, or
    // the comparison above would pass for a kernel that ignored `e_start` entirely.
    let all = as_f32(&host_acc(&c, 0..ne, Wire::Correct));
    assert_separates(&all, &got, "mxfp8 offset window", 1e-4);
}

/// **`gate` and `up` are the same shape, so only the wiring distinguishes them — and
/// swapping them is a different function.**
///
/// `softcap_swiglu` is asymmetric: the sigmoid sees the raw GATE, and the two saturating
/// factors carry different betas, so `situ(g, u)` is nothing like `situ(u, g)`. Run on the
/// DEVICE with the descriptor's two spans exchanged, so what is proved is that this fixture
/// would catch a swapped upload, not merely that two host functions differ.
#[test]
fn swapping_gate_and_up_changes_the_answer() {
    let (ne, hidden, inter) = (2usize, 128usize, 64usize);
    let c = mx8_case(ne, hidden, inter);
    let want = as_f32(&host_acc(&c, 0..ne, Wire::Correct));
    let arm = Arm {
        wire: Wire::GateUp,
        ..Arm::all(ne)
    };
    let got = as_f32(&device_acc(&c, arm));
    // A separation claim, not a tolerance: the swap must move the accumulator by more than
    // `1e-4` of its scale — a decade above the positive gate's own bound — or this fixture
    // cannot see the wiring. The line prints how far it actually moved.
    assert_separates(&want, &got, "mxfp8 swapped gate/up", 1e-4);
}

/// **T1: the scale byte is a RAW exponent (`2^(s−127)`), not an e4m3 code.**
///
/// The fixture is built so the two readings are decades apart and the answer is a single
/// term: every `gate`/`up` value is `0x38` (e4m3 = 1.0) under exponent `127` (= 1.0), so
/// `h` is one constant `situ(1, 1, b1, b2)`; `down` row 0 is one `0x38` at column 0 under
/// exponent `0x70`, and every other down value is `0x00` (= 0), so `dv` IS that one weight
/// times `h`. `0x70` is `2^-15` as an exponent and `224.0` as an e4m3 code — a factor of
/// `224 · 2^15 = 7.34e6`, so a kernel that read the scale through the fp8 LUT would miss by
/// six decades rather than by a rounding.
#[test]
fn the_mxfp8_scale_byte_is_a_raw_exponent() {
    let (hidden, inter) = (32usize, 32usize);
    let one = |rows: usize, cols: usize, exponent: u8| Plane {
        weight: vec![0x38u8; rows * cols],
        scale: vec![exponent; rows * cols.div_ceil(32)],
    };
    // `down` row 0: a single 1.0 at column 0, zeros after; every row's scale is the trap's
    // exponent, so the reading is not hidden behind a coincidental 1.0 block.
    let mut down = Plane {
        weight: vec![0u8; hidden * inter],
        scale: vec![0x70u8; hidden * inter.div_ceil(32)],
    };
    down.weight[0] = 0x38;
    let c = Mx8Case {
        experts: vec![Expert {
            gate: one(inter, hidden, 127),
            up: one(inter, hidden, 127),
            down,
        }],
        x: vec![1.0 / hidden as f32; hidden],
        inter,
        weights: vec![1.0],
    };
    let got = as_f32(&device_acc(&c, Arm::all(1)));
    let want = as_f32(&host_acc(&c, 0..1, Wire::Correct));
    // The one predicted term, stated explicitly so the assertion below is about a NUMBER and
    // not only about agreement with another program: `h = situ(1, 1)`, `dv = 2^-15 · h`.
    let h = softcap_swiglu(1.0, 1.0, SHIPPED_BETAS.0, SHIPPED_BETAS.1);
    let predicted = [2.0f32.powi(-15) * h];
    assert_rel(&predicted, &want[..1], "mxfp8 T1 host", 1e-6);
    assert_rel(&want, &got, "mxfp8 T1 device", tol(inter, hidden));
    assert!(
        got[0] != 0.0 && got[1..].iter().all(|&v| v == 0.0),
        "T1: expected one nonzero slot and zeros after it, got {:?}",
        &got[..2]
    );
    // The trap's SIZE, measured rather than asserted about constants: the reference term is
    // six decades closer to the exponent reading than to the e4m3 one, so a kernel that read
    // the byte through the fp8 LUT could not land here by rounding.
    let as_e4m3 = 224.0f32 * h;
    assert!(
        (want[0] - predicted[0]).abs() * 1.0e6 < (want[0] - as_e4m3).abs(),
        "T1: the reference term {:e} is not six decades closer to the exponent reading {:e} \
         than to the e4m3 reading {:e}",
        want[0],
        predicted[0],
        as_e4m3
    );
}

/// **The shape guards, by CODE**, each case a plausible wrong value rather than a silly one —
/// and the single-knob rows go through ONE arm builder, so a row states the field it turns and
/// the fixture is visibly the same in all four.
#[test]
fn the_expert_shape_guards_refuse_by_code() {
    // One field, one arm: the single-knob refusals share a geometry, so the difference between
    // them is the field and not the fixture. (Without it each row restated the same five-line
    // `Arm` update and `jscpd` paired them, 2026-11-25.)
    let tuned = |f: fn(&mut Arm)| {
        let mut arm = Arm::all(2);
        f(&mut arm);
        launch(&mx8_case(2, 32, 32), arm)
    };
    assert_guards([
        // `span_checked` (1001): a zero extent, and — through the same code — a negative
        // start, which the C also refused before any pointer was touched. Every buffer stays
        // non-empty, so the refusal is about the extent and not about a degenerate allocation.
        (1001, "e_count = 0", tuned(|a| a.e_count = 0)),
        (
            1001,
            "e_start < 0",
            tuned(|a| a.e_start = (-1isize) as usize),
        ),
        // 1003: pass 1 is not row-templated.
        (1003, "nrow = 2", tuned(|a| a.nrow = 2)),
        // 1002: not a whole MXFP8 group, in each reduction dim. These change the CASE, not the
        // arm — 48 is a multiple of 16 (what the e4m3 byte stride alone would need) but not of
        // 32, which is the group the scale grid and the dot both stride by.
        (
            1002,
            "hidden not a group",
            launch(&mx8_case(2, 48, 32), Arm::all(2)),
        ),
        (
            1002,
            "inter not a group",
            launch(&mx8_case(2, 32, 48), Arm::all(2)),
        ),
        // 1004: one past the descriptor array. `wexpert` and `h` are sized by the DESCRIPTOR
        // index, so a range past `n_desc` is a read off the end of both.
        (
            1004,
            "range past the descriptor array",
            launch(
                &mx8_case(2, 32, 32),
                Arm {
                    e_start: 1,
                    e_count: 2,
                    ..Arm::all(2)
                },
            ),
        ),
    ]);
}

/// **The betas are guarded (1006), each quiet in its own way** — the shared table and its
/// argument are `k3::assert_betas_guarded`'s, and sharing it is what makes this kernel's
/// "same activation, same refusal" claim checkable beside the fp4 twin's.
#[test]
fn the_expert_betas_are_guarded() {
    let c = mx8_case(2, 32, 32);
    assert_betas_guarded("moe_expert_range_mxfp8_situ", |b1, b2| {
        let arm = Arm {
            betas: (b1, b2),
            ..Arm::all(2)
        };
        launch(&c, arm).is_err()
    });
}
