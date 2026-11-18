//! The staging operations' HIP FFI: allocate/free the pinned host arena, move one staged
//! read into its pool slot, and the investigation's touch-region read. Split out of
//! `stream.rs` 2026-08-19 when that file crossed its line cap; nothing here knows what a
//! [`super::stream::Streamer`] is, and with the Vulkan backend retired there is exactly one
//! implementation.
//!
//! Nothing else in `stream.rs` is backend-specific; this file is the whole of what is.

use std::ffi::c_void;

/// async.hip's four staging entry points, direct FFI since 2026-11-18 — same names, same
/// return convention (`rivoli_backend::hiprt` carries both and the touch-region record).
use rivoli_backend::hiprt as ffi;

/// A HIP-PINNED host arena, which is what makes the copy below a DMA rather than a
/// staged CPU memcpy. Null on failure.
pub fn alloc(bytes: usize) -> *mut u8 {
    // Null on failure.
    ffi::rivoli_pinned_alloc(bytes as u64) as *mut u8
}

/// HIP tracks the pinned registration's size itself, so this takes only the pointer.
///
/// It took an unused `_bytes` until 2026-08-06 so both backends' `free` were called
/// identically — the Vulkan one needed it because `std::alloc::dealloc` demands the
/// original `Layout`. With one backend that threaded a value through `Streamer`
/// construction and `Drop` purely to be discarded, so it is gone, on the same reasoning
/// as `device.rs::bump`'s `pad`. A backend whose deallocator needs the size needs it back.
///
/// # Safety
/// `p` came from [`alloc`] and is freed exactly once.
pub unsafe fn free(p: *mut u8) {
    unsafe { ffi::rivoli_pinned_free(p as *mut c_void) };
}

/// ASYNC bounce->slot copy — `hipMemcpyAsync` on the fetch stream: it returns before the
/// bytes land, and the read's `Signal` (armed on the same stream) is what says they have.
/// This is the op the load↔compute overlap is built on.
///
/// # Safety
/// `dst` owns `n` device bytes and stays valid until `stream`'s completion signal
/// fires; `src` is a live arena slot holding `n` bytes; `stream` is a live handle.
pub unsafe fn copy_to_slot(
    dst: *mut u8,
    src: *const u8,
    n: usize,
    stream: *mut c_void,
) -> Result<(), String> {
    // SAFETY: the caller's contract, forwarded.
    let rc = unsafe {
        ffi::rivoli_memcpy_h2d_async(dst as *mut c_void, src as *const c_void, n as u64, stream)
    };
    check(rc)
}

/// `0` is HIP success at this boundary; anything else is the negative HIP code the C side
/// folds through `HIP_ERR_BASE` (`hiprt`'s `HIP_ERR_BASE - hipError_t`, async.hip's convention). Factored because both staging calls end this way and the
/// duplication gate is right that one tail should exist once.
fn check(rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("hip rc {rc}"))
    }
}

/// ARENA REFRESH: read `n` bytes of the arena window at `src` on `stream` at full width,
/// discarding the value. Enqueued before the copy, so it is stream-ordered with it and
/// needs no host sync.
///
/// # Safety
/// `src` is a live arena window valid for `n` bytes, `stream` is a live handle, and `sink` is
/// a mapped writable `u64` the kernel stores to only on a value real payloads cannot produce.
pub unsafe fn touch_region(
    src: *const u8,
    n: usize,
    stream: *mut c_void,
    sink: *mut u64,
) -> Result<(), String> {
    // SAFETY: the caller's contract, forwarded.
    let rc =
        unsafe { ffi::rivoli_touch_region_async(src as *const c_void, n as u64, stream, sink) };
    check(rc)
}
