//! The staging operations' HIP FFI: allocate/free the pinned host arena, move one staged
//! read into its pool slot, and the investigation's touch-region read. Split out of
//! `stream.rs` 2026-08-19 when that file crossed its line cap; nothing here knows what a
//! [`super::stream::Streamer`] is, and with the Vulkan backend retired there is exactly one
//! implementation.
//!
//! Nothing else in `stream.rs` is backend-specific; this file is the whole of what is.

use std::ffi::c_void;

mod ffi {
    use std::ffi::c_void;
    unsafe extern "C" {
        /// Pinned host arena for the bounce path (kernels/async.hip). Null on failure.
        pub fn rivoli_pinned_alloc(bytes: u64, coherent: i32) -> *mut c_void;
        /// What the runtime ACTUALLY gave: 0 and `*out` on success, negative on failure.
        pub fn rivoli_pinned_flags(p: *mut c_void, out: *mut u32) -> i32;
        /// The bit `hipHostMallocCoherent` sets, so Rust does not hardcode a HIP constant.
        pub fn rivoli_pinned_coherent_bit() -> u32;
        pub fn rivoli_pinned_free(p: *mut c_void);
        /// Async H2D copy on `stream` (bounce slot → VMM slot). 0 ok, else negative.
        /// `by_kernel != 0` moves the bytes with an ordinary shader copy instead of the copy
        /// engine — Phase 3B. One entry point, because the two are one operation with a knob.
        pub fn rivoli_memcpy_h2d_async(
            dst: *mut c_void,
            src: *const c_void,
            n: u64,
            stream: *mut c_void,
            by_kernel: i32,
        ) -> i32;
        /// Device read of a pinned-arena window on `stream` at `stride4`-uint4 density
        /// (1 = full width), value discarded — the ARENA REFRESH mitigation
        /// (`kernels/async.hip` carries the evidence and the ceiling). `sink` is written only
        /// on an impossible value and exists solely to stop the loads being optimised away.
        pub fn rivoli_touch_region_async(
            src: *const c_void,
            n: u64,
            stream: *mut c_void,
            sink: *mut u64,
            stride4: u64,
        ) -> i32;
    }
}

/// A HIP-PINNED host arena, which is what makes the copy below a DMA rather than a
/// staged CPU memcpy. Null on failure.
///
/// `coherent` requests FINE-GRAINED host memory — a candidate fix for the ordering gap argued
/// at `kernels/async.hip::rivoli_pinned_alloc`, where nothing establishes that the GPU-side
/// copy observes the NVMe DMA's writes to this arena.
pub fn alloc(bytes: usize, coherent: bool) -> *mut u8 {
    // SAFETY: no pointer args; null on failure.
    unsafe { ffi::rivoli_pinned_alloc(bytes as u64, i32::from(coherent)) as *mut u8 }
}

/// What the runtime ACTUALLY returned for `p`: `(flags, coherent_bit_set)`, or `None` if the
/// query failed.
///
/// **A run must state what it GOT, not what it asked for.** A `--pinned-coherent` arm was run
/// and reported no effect, and it could not be believed because nothing observed whether the
/// allocation had changed — an intervention that never applied and an intervention that does
/// not work produce the same red. This is that observation.
pub fn flags(p: *mut u8) -> Option<(u32, bool)> {
    let mut f = 0u32;
    // SAFETY: `p` came from `alloc` and is live; `f` is a live local.
    let rc = unsafe { ffi::rivoli_pinned_flags(p as *mut c_void, &raw mut f) };
    // SAFETY: no arguments.
    let bit = unsafe { ffi::rivoli_pinned_coherent_bit() };
    (rc == 0).then_some((f, f & bit != 0))
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

/// ASYNC bounce->slot copy on the fetch stream — `hipMemcpyAsync` by default, or an ordinary
/// shader copy when `by_kernel` (Phase 3B, a candidate fix; see `kernels/async.hip`).
///
/// ORIGINALLY: `hipMemcpyAsync` on the fetch stream: it returns before the bytes land, and
/// the read's `Signal` (armed on the same stream) is what says they have. This is the
/// op the load↔compute overlap is built on.
///
/// # Safety
/// `dst` owns `n` device bytes and stays valid until `stream`'s completion signal
/// fires; `src` is a live arena slot holding `n` bytes; `stream` is a live handle.
pub unsafe fn copy_to_slot(
    dst: *mut u8,
    src: *const u8,
    n: usize,
    stream: *mut c_void,
    by_kernel: bool,
) -> Result<(), String> {
    // SAFETY: the caller's contract, forwarded.
    let rc = unsafe {
        ffi::rivoli_memcpy_h2d_async(
            dst as *mut c_void,
            src as *const c_void,
            n as u64,
            stream,
            i32::from(by_kernel),
        )
    };
    check(rc)
}

/// `0` is HIP success at this boundary; anything else is the negative HIP code the C side
/// folded through `HIP_ERR_BASE`. Factored because both staging calls end this way and the
/// duplication gate is right that one tail should exist once.
fn check(rc: i32) -> Result<(), String> {
    if rc == 0 {
        Ok(())
    } else {
        Err(format!("hip rc {rc}"))
    }
}

/// ARENA REFRESH: read `n` bytes of the arena window at `src` on `stream` at
/// `stride4`-uint4 density (1 = full width), discarding the value. Enqueued around the
/// copy, so it is stream-ordered with it and needs no host sync.
///
/// # Safety
/// `src` is a live arena window valid for `n` bytes, `stream` is a live handle, and `sink` is
/// a mapped writable `u64` the kernel stores to only on a value real payloads cannot produce.
pub unsafe fn touch_region(
    src: *const u8,
    n: usize,
    stream: *mut c_void,
    sink: *mut u64,
    stride4: u64,
) -> Result<(), String> {
    // SAFETY: the caller's contract, forwarded.
    let rc = unsafe {
        ffi::rivoli_touch_region_async(src as *const c_void, n as u64, stream, sink, stride4)
    };
    check(rc)
}
