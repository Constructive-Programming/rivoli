//! `gemv_bf16` (`y[o] = Σ_i x[i]·bf16(w[o, i])`) on the device, against a host f64 reference.
//!
//! **No C twin.** The kernel is new in the rewrite (owner direction 2026-11-24), so there is no
//! hipcc kernel to score byte-for-byte against; `kernel_coverage.rs`'s `ORACLE_VALIDATED` row
//! waives the C-twin gate in favour of this suite, and this suite is what that waiver cites.
//!
//! **Why the comparison is relative rather than bitwise.** The kernel reduces with 32 lanes and
//! a fixed shuffle ladder, in a different order from any host fold, so its answer is a
//! RE-ASSOCIATION of the same products. [`TOL`]'s provenance is measured rather than picked to
//! fit: this fixture was replayed host-side in f32 with the kernel's exact column partition and
//! accumulator order (four sequential products per chain, then `lo + hi`, then the tail, then
//! the ladder) against the f64 fold, and the worst case — `i_dim = 4608` — came out at
//! **1.85e-7 of `max |y|`**, so `1e-5` carries 54× headroom. The replay does not fuse (the
//! device's `v_fmac_f32` is the more accurate of the two), which makes it a conservative bound;
//! the first device run is what turns it into a device measurement.
//!
//! **The widths are Inkling's, and each enters a different path.** 2304 (Mini's `hidden`) and
//! 4608 (Small's) are multiples of the 256 columns one wave step covers, so the vector path
//! runs end to end with no tail — and 9 and 18 steps straddle `for_strided::<4>`'s remainder.
//! 256 is exactly one step, 272 is one step plus a 16-column tail, 16 is `d_rel` (tail only),
//! and `o_dim` 17 leaves the last of `ceil(o_dim / 8)` blocks partly filled. A one-element
//! offset on either span makes the row pointer 16-byte unaligned, which takes the whole row
//! down the scalar path.
//!
//! Run on the device under `flock /var/run/sys-gpu.lock`, single-threaded (`--test-threads=1`).
#![cfg(feature = "rocm")]

use rivoli_backend::hip::launch_gemv_bf16;
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};

mod common;
use common::{Lcg, assert_guard, assert_rel, back, dev, draws, f32b, f32v, ok, u16b, zeros};

/// The relative tolerance against `max |y|` — see this file's header for the measurement that
/// gives it 54× headroom at the widest case.
const TOL: f32 = 1e-5;

/// The three device addresses and the two extents a `gemv_bf16` launch takes.
///
/// A named bundle rather than six positional arguments: three of them are interchangeable raw
/// addresses and two are bare `usize`, so a transposed pair would be a wrong answer rather than
/// a compile error — the argument `common::AttendIo` carries for the same reason.
struct Operands {
    x: *const f32,
    w: *const u16,
    y: *mut f32,
    o_dim: usize,
    i_dim: usize,
}

impl Operands {
    /// The launch, returning the launcher's own `Result` so a guard arm reaches its assertion
    /// instead of panicking here.
    fn launch(&self) -> anyhow::Result<()> {
        // SAFETY: `x` is `i_dim` live f32, `w` `o_dim·i_dim` live u16 and `y` `o_dim` writable
        // f32, none aliasing another and all outliving the null-stream launch. A refused
        // extent returns its code before any device access.
        unsafe {
            launch_gemv_bf16(
                self.x,
                self.w,
                self.o_dim,
                self.i_dim,
                self.y,
                std::ptr::null_mut(),
            )
        }
    }
}

/// One case: the two extents and the element each span starts at.
///
/// Named fields rather than four bare `usize` in a row, where `i_dim`/`w_off` transposed is a
/// wrong answer instead of a compile error. A non-zero `x_off` (f32 elements) or `w_off` (u16
/// elements) is how the alignment fallback is entered.
#[derive(Clone, Copy, Debug)]
struct Case {
    o_dim: usize,
    i_dim: usize,
    x_off: usize,
    w_off: usize,
}
impl Case {
    const fn new(o_dim: usize, i_dim: usize, x_off: usize, w_off: usize) -> Self {
        Self {
            o_dim,
            i_dim,
            x_off,
            w_off,
        }
    }
}

/// Every path and every Inkling width, in one table: `d_rel`'s 16, one wave step's 256, a step
/// plus a 16-column tail (272), Mini's and Small's `hidden` (2304, 4608), and the two spans
/// one element off alignment.
const CASES: [Case; 7] = [
    Case::new(17, 16, 0, 0),
    Case::new(17, 256, 0, 0),
    Case::new(384, 272, 0, 0),
    Case::new(2304, 2304, 0, 0),
    Case::new(64, 4608, 0, 0),
    Case::new(17, 256, 1, 0),
    Case::new(17, 256, 0, 1),
];

/// `n` bf16 codes from `r`'s stream — the f32 draw narrowed to bf16, the only precision the
/// kernel ever sees.
fn draw_codes(r: &mut Lcg, n: usize) -> Vec<u16> {
    draws(r, n).into_iter().map(f32_to_bf16).collect()
}

/// The host reference: `y[o] = Σ_i x[i]·bf16(w[o, i])` in f64, decoded from the same codes the
/// kernel reads.
///
/// f64 because the device's answer is an f32 re-association of these products; a second f32
/// fold would measure its own rounding instead of the kernel's.
fn reference(codes: &[u16], x: &[f32], c: Case) -> Vec<f32> {
    (0..c.o_dim)
        .map(|o| {
            let row = &codes[o * c.i_dim..(o + 1) * c.i_dim];
            let dot: f64 = row
                .iter()
                .zip(x)
                .map(|(&w, &a)| f64::from(bf16_to_f32(w)) * f64::from(a))
                .sum();
            dot as f32
        })
        .collect()
}

/// One case end to end: draw, upload, launch, read back, compare against the f64 fold.
fn check(c: Case, salt: u64) {
    let mut r = Lcg(salt);
    let codes = draw_codes(&mut r, c.w_off + c.o_dim * c.i_dim);
    let x = draws(&mut r, c.x_off + c.i_dim);
    let (xb, wb) = (dev(&f32b(&x)), dev(&u16b(&codes)));
    let mut yb = zeros(c.o_dim * 4);
    // SAFETY: `c.x_off` and `c.w_off` are within the respective allocated buffer lengths.
    let ops = unsafe {
        Operands {
            x: (xb.ptr() as *const f32).add(c.x_off),
            w: (wb.ptr() as *const u16).add(c.w_off),
            y: yb.ptr_mut() as *mut f32,
            o_dim: c.o_dim,
            i_dim: c.i_dim,
        }
    };
    ok(ops.launch(), "gemv_bf16");
    let got = f32v(&back(&yb));
    let want = reference(&codes[c.w_off..], &x[c.x_off..], c);
    assert_rel(&want, &got, &format!("{c:?}"), TOL);
}

/// Every case in [`CASES`] against the f64 fold.
#[test]
fn gemv_bf16_matches_the_f64_reference() {
    for (k, &c) in CASES.iter().enumerate() {
        check(c, 0x5eed + k as u64);
    }
}

/// Both extents are refused before any device access (1001). The buffers hold one element
/// apiece, and that is enough: a refused launch returns its code, so nothing in it is
/// dereferenced.
#[test]
fn gemv_bf16_geometry_is_guarded() {
    let (xb, wb) = (dev(&f32b(&[1.0, 2.0])), dev(&u16b(&[0u16; 4])));
    let mut yb = zeros(8);
    let y = yb.ptr_mut() as *mut f32;
    let ops = |o_dim, i_dim| Operands {
        x: xb.ptr() as *const f32,
        w: wb.ptr() as *const u16,
        y,
        o_dim,
        i_dim,
    };
    assert_guard(ops(0, 4).launch(), Some(1001), "o_dim = 0");
    assert_guard(ops(4, 0).launch(), Some(1001), "i_dim = 0");
}
