//! `touch_region` — the fetch path's page-touch, the twin of `kernels/async.hip`'s
//! `touch_region_kernel` at tag `archive/hip-kernels` (which carries the investigation and
//! why FULL WIDTH is load-bearing). It has no output, so no bit-identity row can score it;
//! what it must preserve is that every 16-byte unit of the region is loaded.

use crate::gpu::{bid, tid};
use rivoli_abi::geom::TOUCH_BLOCK;
use rivoli_abi::kernarg::Touch;

/// HIP's `uint4`: one 16-byte, 16-aligned vector load per unit.
#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct U4([u32; 4]);

/// Grid-stride over `n4` units by `stride` (`gridDim.x * blockDim.x`, passed by the
/// launcher that chose the grid — the Phase 0c rule: no dispatch-packet reads).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn touch_region(a: Touch) {
    let src = a.src.cast::<U4>();
    let mut acc = 0u64;
    let mut j = u64::from(bid()) * u64::from(TOUCH_BLOCK) + u64::from(tid());
    while j < a.n4 {
        let U4([x, y, z, w]) = *src.add(j as usize);
        acc = acc.wrapping_add(u64::from(x) ^ u64::from(y) ^ u64::from(z) ^ u64::from(w));
        j += a.stride;
    }
    // The store is what keeps the loads; the condition is never true for real payloads.
    if acc == u64::MAX {
        *a.sink = acc;
    }
}
