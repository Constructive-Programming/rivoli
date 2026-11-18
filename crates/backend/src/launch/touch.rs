//! `async.hip`'s `touch_region` launcher, the fetch path's page-touch.

use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::TOUCH_BLOCK;
use rivoli_abi::kernarg::Touch;
use std::ffi::c_void;

static TOUCH_REGION: Kernel<Touch> = Kernel::new(c"touch_region");

/// async.hip's grid cap: at most this many blocks walk the region grid-stride. Host-only —
/// the kernel is handed the resulting stride, never the cap.
const TOUCH_MAX_BLOCKS: u64 = 1024;

/// Read `n` bytes of `src` at full width on `stream`, value discarded (`--arena-refresh`).
/// `n` must be a non-zero multiple of 16 (1002); the grid is `ceil(n/16/256)` blocks capped
/// at [`TOUCH_MAX_BLOCKS`]. Returns async.hip's convention — `0`, a POSITIVE guard code, or
/// `HIP_ERR_BASE - hipError_t` — because `fetch/stage.rs` was written against it.
///
/// The record behind this read (a MITIGATION for GLM's nondeterminism with a measured price,
/// not a root-cause fix; do not delete it without re-running `tests/determinism-glm.sh`) is
/// `kernels/async.hip` at tag `archive/hip-kernels` and
/// `docs/investigations/glm-nondeterminism-closeout.md`.
///
/// # Safety
/// `src` is valid for `n` bytes, `stream` is live, `sink` is one writable device-visible u64.
pub unsafe fn rivoli_touch_region_async(
    src: *const c_void,
    n: u64,
    stream: *mut c_void,
    sink: *mut u64,
) -> i32 {
    if n == 0 || !n.is_multiple_of(16) {
        return 1002;
    }
    let n4 = n / 16;
    let blocks = n4.div_ceil(u64::from(TOUCH_BLOCK)).min(TOUCH_MAX_BLOCKS) as u32;
    let stride = u64::from(blocks) * u64::from(TOUCH_BLOCK);
    let args = Touch {
        src,
        sink,
        n4,
        stride,
    };
    let geom = Geometry {
        blocks,
        threads: TOUCH_BLOCK,
        lds: LdsBytes(0),
    };
    // SAFETY: the caller's contract.
    let rc = unsafe { TOUCH_REGION.launch(geom, stream, &args) };
    // `launch` returns `-(e)`; async.hip folded a launch error to `HIP_ERR_BASE - e`.
    crate::hiprt::based(-rc)
}
