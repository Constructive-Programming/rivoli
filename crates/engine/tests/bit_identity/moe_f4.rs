//! Rows for `kernels/moe_f4.hip`'s two launchers on `moe.rs`'s range machinery: DeepSeek-V4's
//! range at its real 4096 × 2048 experts (one streamed expert, and the six-expert resident
//! batch) and Kimi-K3's at its 3584-wide latent × 3072, plus small widths at each launcher's
//! own divisibility floor (128 for V4, 32 for K3). e8m0 scales carry the format's NaN (0xff),
//! its subnormal floor (0x00 = 2^-127) and its largest power (0xfe = 2^127); `n_desc` runs
//! both exactly to the range's end and one past it.

use super::device::{I, O, Ref, S, Tally};
use super::inputs::uniform;
use super::moe::{Args, RangeBufs, U64, range_call, range_row};
use super::ranges::{K3, Range, V4, k3_betas};
use rivoli_backend::hip;
use std::ffi::c_void;

// One line per C signature, as `moe.rs`'s.
#[rustfmt::skip]
type F4Fn = unsafe extern "C" fn(I, i32, i32, i32, i32, i32, *const c_void, I, f32, O, U64, i32, S) -> i32;
#[rustfmt::skip]
type SituFn = unsafe extern "C" fn(I, i32, i32, i32, i32, i32, *const c_void, I, f32, f32, O, U64, i32, S) -> i32;

/// The range end exactly on odd seeds, one descriptor past it on even ones.
fn n_desc((_, _, es, ec, _): Range, seed: u64) -> usize {
    es + ec + usize::from(seed.is_multiple_of(2))
}

/// The fp4 refusals, `(args, n_desc)` plus the activation parameter(s) `p`, at reduction dims
/// of `w` (a multiple of the launcher's unit): `n_desc` 0 and a negative start (1001), a dim
/// off the unit (1002), two token rows (1003), a range past `n_desc` (1004), each refused
/// parameter (1006), then a pair pinning each adjacent two's order (1001 > 1002 > 1003 > 1004
/// > 1006).
fn f4_guards<P: Copy>(w: usize, ok: P, bad: [P; 4]) -> Vec<(Args, (usize, P))> {
    let mut g = vec![
        ((0, w, 0, 1, 1), (4, ok)),
        ((w, w, 0, 1, 1), (0, ok)),
        ((w, w, -1, 1, 1), (4, ok)),
        ((w + 2, w, 0, 1, 1), (4, ok)),
        ((w, w, 0, 1, 2), (4, ok)),
        ((w, w, 3, 2, 1), (4, ok)),
    ];
    g.extend(bad.map(|p| ((w, w, 0, 1, 1), (4, p))));
    g.extend([
        ((0, w + 2, 0, 1, 1), (4, ok)),
        ((w + 2, w, 0, 1, 2), (4, bad[0])),
        ((w, w, 3, 2, 2), (4, bad[0])),
        ((w, w, 3, 2, 1), (4, bad[0])),
    ]);
    g
}

pub fn moe_expert_range_f4(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_moe_expert_range_f4`'s C signature.
    let c: F4Fn = unsafe { r.sym(c"rivoli_moe_expert_range_f4") };
    // V4's shipped limit (10) on odd seeds; seeded positives elsewhere.
    let value = |range, seed: u64| {
        let limit = if seed % 2 == 1 {
            10.0
        } else {
            uniform(1, (seed, 7), 0.5..12.0)[0]
        };
        (n_desc(range, seed), limit)
    };
    let mut guards = f4_guards(128, 10.0, [f32::NAN, 0.0, f32::INFINITY, -1.0]);
    // V4's unit is `ACT_QUANT_BLOCK` (128), not `F4_GROUP` (32): dims that are whole groups but
    // not whole blocks must still be refused, by this launcher and before any pass runs.
    guards.extend([
        ((160, 128, 0, 1, 1), (4, 10.0)),
        ((128, 96, 0, 1, 1), (4, 10.0)),
    ]);
    let hip = |b: RangeBufs, i, (nd, limit): (usize, f32)| {
        // SAFETY: live buffers sized for the range, or the guard buffers under a refused guard.
        unsafe { range_call!(c, b, i; nd as i32, b.descs, b.w, limit) }
    };
    // SAFETY: as above.
    let rust = |b: RangeBufs, i, (nd, limit): (usize, f32)| unsafe {
        range_call!(hip::launch_moe_expert_range_f4, b, i; nd, b.descs.cast(), b.w, limit)
    };
    range_row(t, &V4, (value, &guards), (hip, rust));
}

pub fn moe_expert_range_f4_situ(r: &Ref, t: &mut Tally) {
    // SAFETY: `rivoli_moe_expert_range_f4_situ`'s C signature.
    let c: SituFn = unsafe { r.sym(c"rivoli_moe_expert_range_f4_situ") };
    // K3's betas (4, 25) on odd seeds; seeded positives elsewhere, so tanh's range moves.
    let value = |range, seed: u64| (n_desc(range, seed), k3_betas(seed));
    let bad = [
        (f32::NAN, 25.0),
        (4.0, f32::INFINITY),
        (0.0, 25.0),
        (4.0, -1.0),
    ];
    let guards = f4_guards(32, (4.0, 25.0), bad);
    let hip = |b: RangeBufs, i, (nd, (b1, b2)): (usize, (f32, f32))| {
        // SAFETY: live buffers sized for the range, or the guard buffers under a refused guard.
        unsafe { range_call!(c, b, i; nd as i32, b.descs, b.w, b1, b2) }
    };
    // SAFETY: as above.
    let rust = |b: RangeBufs, i, (nd, (b1, b2)): (usize, (f32, f32))| unsafe {
        range_call!(hip::launch_moe_expert_range_f4_situ, b, i; nd, b.descs.cast(), b.w, b1, b2)
    };
    range_row(t, &K3, (value, &guards), (hip, rust));
}
