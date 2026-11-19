//! The host halves of the Rust kernels — what the tag's `extern "C" int rivoli_*` launchers
//! did: the argument guards, the grid geometry, the launch. One module per kernel family; a
//! Phase 1 lane adds its family's file here.
//!
//! Each launcher keeps its C twin's name and return codes (`0`, POSITIVE guard code,
//! `-(hipError_t)`) exactly — `rivoli_hash_rows`' 1002 is the one code added, for a `usize`
//! walk the C's `int`s could not even be handed — so a `launch_x => rivoli_x(..)` row in a `launchers!`
//! invocation file maps the status unchanged and `crates/engine/tests/bit_identity/` can
//! compare guard codes against the tagged C launcher it replaces.
//!
//! What changed is the parameter list: a launcher takes the kernel's
//! `rivoli_abi::kernarg` block (built by its `launchers!` row) plus whatever only the HOST
//! reads — a stream, a count that becomes the grid, a checkpoint constant to refuse. The
//! guards' REASONS live in the C at tag `archive/hip-kernels` and in the `launch_*` docs;
//! these files add the geometry tie — every block and LDS size is the `rivoli_abi::geom`
//! constant the kernel itself strides by, never a literal.

#![cfg(feature = "rocm")]

// One family per pair, interleaved: as two plain `mod x;` / `use x::*;` runs this list matches
// `rivoli_abi::geom`'s line for line and crosses jscpd's 5-line floor (measured 2026-11-19).
mod activation;
pub(crate) use activation::*;
mod attn;
pub(crate) use attn::*;
mod blockindex;
pub(crate) use blockindex::*;
mod fwd;
pub(crate) use fwd::*;
mod headtail;
pub(crate) use headtail::*;
mod indexer;
pub(crate) use indexer::*;
mod kvcompress;
pub(crate) use kvcompress::*;
mod linalg;
pub(crate) use linalg::*;
mod mla;
pub(crate) use mla::*;
mod residual;
pub(crate) use residual::*;
mod touch;
pub use touch::rivoli_touch_region_async;

use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::ELEMENTWISE_BLOCK;
use std::ffi::c_void;

/// A launch's block count — one block per row, segment or token — as the C `int` the
/// launcher took. A unit, not a bare `i32`, because it sits beside other `int`s (`hc`, a
/// width, a row length) that every one of these signatures could transpose it with.
#[derive(Clone, Copy)]
pub(crate) struct Rows(pub(crate) i32);

/// A guard's verdict: the geometry to launch at, or the POSITIVE code to return.
type Checked = Result<Geometry, i32>;

/// A 2-D guard's verdict: the `(geometry, blocks_y)` grid, or the POSITIVE code.
type Checked2 = Result<(Geometry, u32), i32>;

/// Launch `k` with `args` if the guard passed, else return its code — the one tail every
/// launcher shares.
///
/// # Safety
/// `args` is `k`'s argument block and satisfies its buffer contract; `stream` is live or null.
unsafe fn run<A: Copy>(k: &Kernel<A>, checked: Checked, stream: *mut c_void, args: &A) -> i32 {
    unsafe { run_2d(k, checked.map(|geom| (geom, 1)), stream, args) }
}

/// [`run`] on a 2-D grid (`Kernel::launch_2d`).
///
/// # Safety
/// As [`run`].
unsafe fn run_2d<A: Copy>(k: &Kernel<A>, checked: Checked2, stream: *mut c_void, args: &A) -> i32 {
    match checked {
        Ok(grid) => unsafe { k.launch_2d(grid, stream, args) },
        Err(code) => code,
    }
}

/// `a <= 0 || b <= 0 || …` — the C guards' extent refusal (1001), one call per launcher.
fn any_nonpositive(extents: &[i32]) -> bool {
    extents.iter().any(|&x| x <= 0)
}

/// A row gather's refusal (1001: a negative `token`), then [`elementwise`] over its `hidden`.
fn row_gather(token: i32, hidden: i32) -> Checked {
    if token < 0 {
        return Err(1001);
    }
    elementwise(i64::from(hidden))
}

/// `!(x > 0 && x < INFINITY)` — the C guards' one-compare form, which refuses NaN, zero,
/// negatives and +inf together (`activation.hip`'s `swiglu_clamped_bf16` note says why each).
fn not_positive_finite(x: f32) -> bool {
    !(x > 0.0 && x < f32::INFINITY)
}

/// `dim3((n + 255) / 256), dim3(256)`, no LDS; `n <= 0` refused (1001).
fn elementwise(n: i64) -> Checked {
    if n <= 0 {
        return Err(1001);
    }
    // `dim3` takes `unsigned`: the C narrowed the same way.
    Ok(Geometry {
        blocks: (n as u64).div_ceil(u64::from(ELEMENTWISE_BLOCK)) as u32,
        threads: ELEMENTWISE_BLOCK,
        lds: LdsBytes(0),
    })
}
