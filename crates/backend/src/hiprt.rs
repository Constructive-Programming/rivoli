//! Direct FFI to `libamdhip64` — the host half of HIP, with no C++ in between.
//!
//! Until 2026-11-18 every host-side HIP call went through a hipcc-compiled shim
//! (`kernels/async.hip` for streams/events/timelines/the pinned arena,
//! `kernels/linalg.hip` for the device sync and the device-to-device copies,
//! `kernels/vmm.hip`'s `fill_u32` kernel). The shims added nothing but a return-code
//! convention, so they are gone and the conventions moved here **unchanged** — callers'
//! diffs are their import line, nothing else:
//!
//! - async.hip's functions return `0` or `HIP_ERR_BASE - hipError_t` (or `-1`, or null),
//!   exactly as before; their callers only test `!= 0`.
//! - linalg.hip's return `0` or `-(hipError_t)`, the launcher convention
//!   [`crate::hip::ensure_hip_status`] decodes.
//!
//! The `rivoli_` names are kept for the same reason: grep from any caller lands here.

#![cfg(feature = "rocm")]

use std::ffi::{c_char, c_void};
use std::sync::atomic::{AtomicU64, Ordering};

/// `hipError_t` — a C enum, `int`-sized. `0` is `hipSuccess`.
pub(crate) type HipError = i32;

const HIP_STREAM_NON_BLOCKING: u32 = 0x01; // hip_runtime_api.h:833
const HIP_HOST_MALLOC_DEFAULT: u32 = 0x0; // :899
const HIP_MALLOC_SIGNAL_MEMORY: u32 = 0x2; // :967
const HIP_STREAM_WAIT_VALUE_GTE: u32 = 0x0; // :1033
const HIP_MEMCPY_HOST_TO_DEVICE: i32 = 1; // driver_types.h:359
const HIP_MEMCPY_DEVICE_TO_DEVICE: i32 = 3; // driver_types.h:361

/// async.hip's distinct-nonzero fold: `HIP_ERR_BASE - e`, never colliding with 0.
pub(crate) const HIP_ERR_BASE: i32 = 1_000_000;

// The ABI mirror of `hip_runtime_api.h`'s prototypes, each line that header's signature.
unsafe extern "C" {
    pub(crate) fn hipModuleLoadData(module: *mut *mut c_void, image: *const c_void) -> HipError;
    pub(crate) fn hipModuleGetFunction(
        f: *mut *mut c_void,
        module: *mut c_void,
        name: *const c_char,
    ) -> HipError;
    pub(crate) fn hipModuleLaunchKernel(
        f: *mut c_void,
        gx: u32,
        gy: u32,
        gz: u32,
        bx: u32,
        by: u32,
        bz: u32,
        shared_bytes: u32,
        stream: *mut c_void,
        params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> HipError;
    fn hipDeviceSynchronize() -> HipError;
    fn hipMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> HipError;
    fn hipMemcpyAsync(
        dst: *mut c_void,
        src: *const c_void,
        bytes: usize,
        kind: i32,
        stream: *mut c_void,
    ) -> HipError;
    fn hipMemsetD32Async(
        dst: *mut c_void,
        value: i32,
        count: usize,
        stream: *mut c_void,
    ) -> HipError;
    fn hipStreamCreateWithFlags(s: *mut *mut c_void, flags: u32) -> HipError;
    fn hipStreamDestroy(s: *mut c_void) -> HipError;
    fn hipLaunchHostFunc(
        s: *mut c_void,
        f: extern "C" fn(*mut c_void),
        user: *mut c_void,
    ) -> HipError;
    fn hipEventCreate(e: *mut *mut c_void) -> HipError;
    fn hipEventRecord(e: *mut c_void, s: *mut c_void) -> HipError;
    fn hipEventElapsedTime(ms: *mut f32, start: *mut c_void, end: *mut c_void) -> HipError;
    fn hipEventDestroy(e: *mut c_void) -> HipError;
    fn hipHostMalloc(p: *mut *mut c_void, bytes: usize, flags: u32) -> HipError;
    fn hipHostFree(p: *mut c_void) -> HipError;
    fn hipExtMallocWithFlags(p: *mut *mut c_void, bytes: usize, flags: u32) -> HipError;
    fn hipFree(p: *mut c_void) -> HipError;
    fn hipStreamWaitValue64(
        s: *mut c_void,
        p: *mut c_void,
        value: u64,
        flags: u32,
        mask: u64,
    ) -> HipError;
    fn hipStreamWriteValue64(s: *mut c_void, p: *mut c_void, value: u64, flags: u32) -> HipError;
}

/// `0` or `-(hipError_t)` — the launcher convention.
fn neg(e: HipError) -> i32 {
    -e
}

/// `0` or `HIP_ERR_BASE - hipError_t` — async.hip's convention.
fn based(e: HipError) -> i32 {
    if e == 0 { 0 } else { HIP_ERR_BASE - e }
}

/// A handle out-parameter call: the handle on success, null on any error.
fn handle_or_null(f: impl FnOnce(*mut *mut c_void) -> HipError) -> *mut c_void {
    let mut h = std::ptr::null_mut();
    if f(&mut h) == 0 {
        h
    } else {
        std::ptr::null_mut()
    }
}

// ── linalg.hip's host half ──────────────────────────────────────────────────────────────

/// One join point per token: block until all launched kernels retire.
pub(crate) fn rivoli_device_sync() -> i32 {
    // SAFETY: no arguments.
    neg(unsafe { hipDeviceSynchronize() })
}

/// # Safety
/// `dst`/`src` are `bytes`-sized, non-overlapping device regions.
pub(crate) unsafe fn rivoli_memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> i32 {
    neg(unsafe { hipMemcpy(dst.cast(), src.cast(), bytes, HIP_MEMCPY_DEVICE_TO_DEVICE) })
}

/// # Safety
/// As [`rivoli_memcpy_dtod`], and both outlive `stream`'s completion.
pub(crate) unsafe fn rivoli_memcpy_dtod_async(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    stream: *mut c_void,
) -> i32 {
    neg(unsafe {
        hipMemcpyAsync(
            dst.cast(),
            src.cast(),
            bytes,
            HIP_MEMCPY_DEVICE_TO_DEVICE,
            stream,
        )
    })
}

/// vmm.hip's `fill_u32`: `bytes % 4` refused (1001), zero bytes a no-op, else the pattern
/// on the NULL stream — `hipMemsetD32Async(.., 0)` is the same stream position the old
/// null-stream kernel launch took, so ordering is unchanged.
///
/// Why it exists (carried from the deleted C): it POISONS a slot the moment it is admitted,
/// before its bytes land, to make a read-before-write DETERMINISTIC. Unpoisoned, an
/// unloaded slot holds uninitialised memory (NaN, ~6% of long runs) or a previous expert's
/// weights (finite, plausible, silently WRONG); a quiet-NaN pattern collapses both into the
/// loud case, so a fault seen one run in sixteen is seen every time it occurs.
///
/// UNVERIFIED on this ROCm: the old path was a kernel store, which cannot refuse an address;
/// a memset into a `hipMemMap`'d pool slot (`--features trace`'s admission fill) is owed a
/// device run (rust-kernels.md §5).
///
/// # Safety
/// `dst` owns `bytes` device bytes.
pub(crate) unsafe fn rivoli_fill_u32(dst: *mut u8, pat: u32, bytes: usize) -> i32 {
    if !bytes.is_multiple_of(4) {
        return 1001;
    }
    if bytes == 0 {
        return 0;
    }
    // The pattern is bits; the API spells them `int`.
    neg(unsafe { hipMemsetD32Async(dst.cast(), pat as i32, bytes / 4, std::ptr::null_mut()) })
}

// ── async.hip: streams, the host-func bridge, events ────────────────────────────────────

/// A non-blocking stream: work on it never implicitly serialises against the null stream,
/// so a compute launch truly overlaps a copy on another stream. Null on failure.
pub(crate) fn rivoli_stream_create() -> *mut c_void {
    // SAFETY: out-parameter only.
    handle_or_null(|h| unsafe { hipStreamCreateWithFlags(h, HIP_STREAM_NON_BLOCKING) })
}

/// # Safety
/// `s` came from [`rivoli_stream_create`] and is destroyed once.
pub(crate) unsafe fn rivoli_stream_destroy(s: *mut c_void) -> i32 {
    if unsafe { hipStreamDestroy(s) } == 0 {
        0
    } else {
        -1
    }
}

/// `cb(user)` runs on a HIP host thread once all prior work on `stream` completes; per the
/// HIP contract it must not call HIP or block.
///
/// # Safety
/// `stream` is live; `user` stays valid until `cb` has run.
pub(crate) unsafe fn rivoli_stream_host_signal(
    stream: *mut c_void,
    cb: extern "C" fn(*mut c_void),
    user: *mut c_void,
) -> i32 {
    based(unsafe { hipLaunchHostFunc(stream, cb, user) })
}

pub(crate) fn rivoli_event_create() -> *mut c_void {
    // SAFETY: out-parameter only.
    handle_or_null(|h| unsafe { hipEventCreate(h) })
}

/// # Safety
/// `ev` and `stream` are live handles.
pub(crate) unsafe fn rivoli_event_record(ev: *mut c_void, stream: *mut c_void) -> i32 {
    based(unsafe { hipEventRecord(ev, stream) })
}

/// # Safety
/// Both events are live and recorded; `ms` is writable.
pub(crate) unsafe fn rivoli_event_elapsed(
    start: *mut c_void,
    end: *mut c_void,
    ms: *mut f32,
) -> i32 {
    based(unsafe { hipEventElapsedTime(ms, start, end) })
}

/// # Safety
/// `ev` came from [`rivoli_event_create`] and is destroyed once.
pub(crate) unsafe fn rivoli_event_destroy(ev: *mut c_void) {
    // async.hip discarded the status; a teardown has no one to report it to.
    let _ = unsafe { hipEventDestroy(ev) };
}

// ── async.hip: the pinned bounce arena and its two device ops ───────────────────────────
//
// `kernels/async.hip` at tag `archive/hip-kernels` carries the GLM-nondeterminism record
// behind `rivoli_touch_region_async` (a MITIGATION with a measured price, not a root-cause
// fix; do not delete without re-running `tests/determinism-glm.sh`) and the reason the arena
// is plain `hipHostMallocDefault`. Neither changed in the move.

/// A HIP-pinned host arena. Null on failure.
pub fn rivoli_pinned_alloc(bytes: u64) -> *mut c_void {
    // SAFETY: out-parameter only.
    handle_or_null(|p| unsafe { hipHostMalloc(p, bytes as usize, HIP_HOST_MALLOC_DEFAULT) })
}

/// # Safety
/// `p` came from [`rivoli_pinned_alloc`] and is freed once.
pub unsafe fn rivoli_pinned_free(p: *mut c_void) {
    let _ = unsafe { hipHostFree(p) };
}

/// # Safety
/// `dst` owns `n` device bytes, `src` `n` pinned host bytes, both live until `stream`'s
/// completion.
pub unsafe fn rivoli_memcpy_h2d_async(
    dst: *mut c_void,
    src: *const c_void,
    n: u64,
    stream: *mut c_void,
) -> i32 {
    based(unsafe { hipMemcpyAsync(dst, src, n as usize, HIP_MEMCPY_HOST_TO_DEVICE, stream) })
}

/// The touch-region read lives with the other Rust launchers; re-exported so
/// `fetch/stage.rs` imports all four staging entry points from one module.
pub use crate::rk_launchers::rivoli_touch_region_async;

// ── async.hip: the timeline (signal-memory counter) ─────────────────────────────────────
//
// `hipStreamWaitValue64` waits on a VALUE, well defined before the signaller has run —
// which is why it, and not `hipStreamWaitEvent`, is the primitive the ticketed dataflow
// rests on (`kernels/async.hip` (at tag `archive/hip-kernels`), which also records the 2026-07-30 gfx1151 probe
// that a wait enqueued before its write retires correctly). The counter must live in signal
// memory. Why `release` exists at all — a fetch error once hung the device — is
// `gpustream.rs`'s timeline doc.

pub(crate) fn rivoli_timeline_create() -> *mut c_void {
    // SAFETY: out-parameter only.
    let p = handle_or_null(|p| unsafe {
        hipExtMallocWithFlags(p, size_of::<u64>(), HIP_MALLOC_SIGNAL_MEMORY)
    });
    if !p.is_null() {
        // SAFETY: a fresh, host-visible, 8-byte signal allocation.
        unsafe { p.cast::<u64>().write(0) };
    }
    p
}

/// # Safety
/// `t` came from [`rivoli_timeline_create`] (or is null) and is destroyed once.
pub(crate) unsafe fn rivoli_timeline_destroy(t: *mut c_void) {
    if !t.is_null() {
        let _ = unsafe { hipFree(t) };
    }
}

/// # Safety
/// `stream` and `t` are live.
pub(crate) unsafe fn rivoli_timeline_wait(stream: *mut c_void, t: *mut c_void, value: u64) -> i32 {
    unsafe { hipStreamWaitValue64(stream, t, value, HIP_STREAM_WAIT_VALUE_GTE, !0) }
}

/// # Safety
/// `stream` and `t` are live.
pub(crate) unsafe fn rivoli_timeline_signal(
    stream: *mut c_void,
    t: *mut c_void,
    value: u64,
) -> i32 {
    unsafe { hipStreamWriteValue64(stream, t, value, 0) }
}

/// # Safety
/// `t` is a live timeline; the device writes it, so it is only ever read atomically.
pub(crate) unsafe fn rivoli_timeline_completed(t: *mut c_void) -> u64 {
    unsafe { AtomicU64::from_ptr(t.cast()) }.load(Ordering::Acquire)
}

/// Host-side RELEASE for teardown: raise the counter to `value` so waits already enqueued
/// retire. Monotone (`fetch_max`, the CAS loop async.hip spelled by hand): lowering it
/// would un-release a slot another consumer is gated on.
///
/// # Safety
/// As [`rivoli_timeline_completed`].
pub(crate) unsafe fn rivoli_timeline_release(t: *mut c_void, value: u64) {
    unsafe { AtomicU64::from_ptr(t.cast()) }.fetch_max(value, Ordering::AcqRel);
}
