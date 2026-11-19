//! Direct FFI to `libamdhip64` — the host half of HIP, with no C++ in between.
//!
//! Until 2026-11-18 every host-side HIP call went through a hipcc-compiled shim
//! (`kernels/async.hip` for streams/events/timelines/the pinned arena,
//! `kernels/linalg.hip` for the device sync and the device-to-device copies,
//! `kernels/vmm.hip`'s `fill_u32` kernel, now `hipMemsetD32Async`). The shims added nothing
//! but a return-code convention, so they are gone and the conventions moved here
//! **unchanged** — callers' diffs are their import line, nothing else. Four conventions, one
//! per family of caller, each kept because its callers were written against it:
//!
//! - **`0` / `-(hipError_t)`** ([`neg`]) — linalg.hip's, the launcher convention
//!   [`crate::hip::ensure_hip_status`] decodes (device sync, copies, `fill_u32`).
//! - **`0` / `HIP_ERR_BASE - hipError_t`** ([`based`]) — async.hip's; its callers only test
//!   `!= 0` (streams, events, the staging copy, `rivoli_touch_region_async`).
//! - **`0` / `-1`** — the destroys (`rivoli_stream_destroy`; its one caller, `Drop`, ignores it).
//! - **a handle, or null** (`handle_or_null`) — the creators (streams, events, the pinned
//!   arena), where the caller has only a pointer to test.
//!
//! > **2026-11-19:** vmm.hip's allocator followed (Phase 1, the linalg lane), and it is the
//! > one exception to "callers' diffs are their import line": [`rivoli_vmm_alloc`] returns
//! > `Result<VmmMapping, i32>` — the C's codes (`1001`, `-(hipError_t)`) in the `Err` — where
//! > the C filled three out-pointers, and its one caller (`engine/src/device.rs`) changed
//! > with it.
//!
//! Beside all four, a POSITIVE guard code (1001…) is an argument refusal before any HIP call
//! (`fill_u32`'s `bytes % 4`, the touch launcher's 1002) — the codes the C launchers returned.
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
    pub(crate) fn hipModuleLoad(module: *mut *mut c_void, path: *const c_char) -> HipError;
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
    fn hipMemGetAllocationGranularity(g: *mut usize, prop: *const MemProp, opt: i32) -> HipError;
    fn hipMemCreate(h: *mut *mut c_void, size: usize, prop: *const MemProp, flags: u64)
    -> HipError;
    fn hipMemAddressReserve(
        p: *mut *mut c_void,
        size: usize,
        align: usize,
        addr: *mut c_void,
        flags: u64,
    ) -> HipError;
    fn hipMemMap(p: *mut c_void, size: usize, off: usize, h: *mut c_void, flags: u64) -> HipError;
    fn hipMemSetAccess(p: *mut c_void, size: usize, desc: *const MemAccess, n: usize) -> HipError;
    fn hipMemUnmap(p: *mut c_void, size: usize) -> HipError;
    fn hipMemRelease(h: *mut c_void) -> HipError;
    fn hipMemAddressFree(p: *mut c_void, size: usize) -> HipError;
}

/// `0` or `-(hipError_t)` — the launcher convention.
fn neg(e: HipError) -> i32 {
    -e
}

/// `0` or `HIP_ERR_BASE - hipError_t` — async.hip's convention.
pub(crate) fn based(e: HipError) -> i32 {
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
/// VERIFIED on the device 2026-11-18 (rust-kernels.md §5, the merge-review lease, arm T):
/// `--features trace`'s admission fill (`routed.rs`, rc propagated) ran into `hipMemMap`'d
/// pool slots during a 32-token GLM decode (4612 decode misses), rc 0, the first 4 ids at the
/// reference. That it ran on every admission rests on the code path; there is no per-fill
/// counter.
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
pub use crate::launch::rivoli_touch_region_async;

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

// ── vmm.hip: device-local memory the CPU can fill in place ──────────────────────────────
//
// The escape from the coherent read tax (docs/reference/architecture.md §1): on this APU
// `hipHostMalloc` of any grain maps system-domain (MTYPE_NC/UC, 4 KiB pages) and reads
// slower than `hipMalloc`; the VMM API allocates DEVICE-LOCAL physical memory (MTYPE_RW) and
// then grants the CPU access too, so weights are `pread` straight in and read in place. The
// ~9% read figure has no live measurement behind it (vmm.hip's SOURCES note at tag
// `archive/hip-kernels`: the probe was deleted, recoverable with `git show
// 3e1bd96:docs/probes/vmm_probe.cpp`); what is measured today is the write side,
// architecture.md §3.

/// `hipMemLocation` (driver_types.h): a location type and its id.
#[repr(C)]
#[derive(Clone, Copy)]
struct MemLocation {
    kind: i32,
    id: i32,
}

/// `hipMemAllocationProp` (hip_runtime_api.h:1919), with the fields this allocator leaves
/// zero spelled as such: the handle type (`hipMemHandleTypeNone`), the win32 metadata and
/// `allocFlags`.
#[repr(C)]
struct MemProp {
    kind: i32,
    handle_type: i32,
    location: MemLocation,
    win32: *mut c_void,
    alloc_flags: u32,
}
const _: () = assert!(size_of::<MemProp>() == 32);

/// `hipMemAccessDesc` (hip_runtime_api.h:1287).
#[repr(C)]
struct MemAccess {
    location: MemLocation,
    flags: i32,
}
const _: () = assert!(size_of::<MemAccess>() == 12);

const HIP_MEM_ALLOCATION_TYPE_PINNED: i32 = 1; // hip_runtime_api.h:1299
const HIP_MEM_LOCATION_TYPE_DEVICE: i32 = 1; // driver_types.h:439
const HIP_MEM_LOCATION_TYPE_HOST: i32 = 2; // driver_types.h:440
const HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE: i32 = 3; // hip_runtime_api.h:1279
const HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM: i32 = 0; // :1967

/// A live [`rivoli_vmm_alloc`] mapping: the VA, the physical handle, and the
/// granularity-rounded size [`rivoli_vmm_free`] needs verbatim.
#[derive(Clone, Copy, Debug)]
pub struct VmmMapping {
    pub ptr: *mut c_void,
    pub handle: *mut c_void,
    pub mapped: usize,
}

/// `hipMemSetAccess` read-write over `[ptr, ptr + size)` for one location.
///
/// # Safety
/// `[ptr, ptr + size)` is a live mapping.
unsafe fn grant(ptr: *mut c_void, size: usize, location: MemLocation) -> HipError {
    let desc = MemAccess {
        location,
        flags: HIP_MEM_ACCESS_FLAGS_PROT_READ_WRITE,
    };
    unsafe { hipMemSetAccess(ptr, size, &desc, 1) }
}

/// Device `dev` read-write, then the host (the domain-crossing grant): two calls, the proven
/// probe sequence. The first failure, or `0`.
///
/// # Safety
/// `[ptr, ptr + size)` is a live mapping.
unsafe fn grant_both(ptr: *mut c_void, size: usize, dev: i32) -> HipError {
    let at = |kind, id| MemLocation { kind, id };
    let e = unsafe { grant(ptr, size, at(HIP_MEM_LOCATION_TYPE_DEVICE, dev)) };
    if e != 0 {
        return e;
    }
    unsafe { grant(ptr, size, at(HIP_MEM_LOCATION_TYPE_HOST, 0)) }
}

/// Reserve a VA range for the physical allocation `h`, map it, and grant both sides. On
/// failure everything this acquired is released (not `h`) and the error returned.
///
/// # Safety
/// `h` is a live `msize`-byte physical allocation.
unsafe fn map_both(h: *mut c_void, msize: usize, dev: i32) -> Result<*mut c_void, HipError> {
    let mut ptr = std::ptr::null_mut();
    // SAFETY (every call): out-parameters and the range just reserved.
    let e = unsafe { hipMemAddressReserve(&mut ptr, msize, 0, std::ptr::null_mut(), 0) };
    if e != 0 {
        return Err(e);
    }
    let e = unsafe { hipMemMap(ptr, msize, 0, h, 0) };
    if e != 0 {
        unsafe { hipMemAddressFree(ptr, msize) };
        return Err(e);
    }
    let e = unsafe { grant_both(ptr, msize, dev) };
    if e != 0 {
        unsafe {
            hipMemUnmap(ptr, msize);
            hipMemAddressFree(ptr, msize);
        }
        return Err(e);
    }
    Ok(ptr)
}

/// Allocate `size` bytes of device-local memory on device `dev`, mapped for BOTH device and
/// host. `Err(1001)` for `size == 0`, else `Err(-(hipError_t))` with everything acquired so
/// far released — vmm.hip's codes.
pub fn rivoli_vmm_alloc(size: usize, dev: i32) -> Result<VmmMapping, i32> {
    if size == 0 {
        return Err(1001);
    }
    let prop = MemProp {
        kind: HIP_MEM_ALLOCATION_TYPE_PINNED,
        handle_type: 0,
        location: MemLocation {
            kind: HIP_MEM_LOCATION_TYPE_DEVICE,
            id: dev,
        },
        win32: std::ptr::null_mut(),
        alloc_flags: 0,
    };
    let mut gran = 0usize;
    // SAFETY (every call): out-parameters, and the handle just created.
    let e = unsafe {
        hipMemGetAllocationGranularity(&mut gran, &prop, HIP_MEM_ALLOCATION_GRANULARITY_MINIMUM)
    };
    if e != 0 {
        return Err(neg(e));
    }
    let msize = size.div_ceil(gran) * gran;
    let mut h = std::ptr::null_mut();
    let e = unsafe { hipMemCreate(&mut h, msize, &prop, 0) };
    if e != 0 {
        return Err(neg(e));
    }
    match unsafe { map_both(h, msize, dev) } {
        Ok(ptr) => Ok(VmmMapping {
            ptr,
            handle: h,
            mapped: msize,
        }),
        Err(e) => {
            unsafe { hipMemRelease(h) };
            Err(neg(e))
        }
    }
}

/// Tear down a [`rivoli_vmm_alloc`] mapping — unmap, release, free, all three attempted —
/// reporting the first failure as `-(hipError_t)`.
///
/// # Safety
/// `m` came from [`rivoli_vmm_alloc`] and is freed once.
pub unsafe fn rivoli_vmm_free(m: VmmMapping) -> i32 {
    let VmmMapping {
        ptr,
        handle,
        mapped,
    } = m;
    let codes = unsafe {
        [
            hipMemUnmap(ptr, mapped),
            hipMemRelease(handle),
            hipMemAddressFree(ptr, mapped),
        ]
    };
    neg(codes.into_iter().find(|&e| e != 0).unwrap_or(0))
}
