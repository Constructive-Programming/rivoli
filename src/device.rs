//! The resident device tier: one device-local, host-fillable VMM slab
//! ([`VmmBuf`]) allocated ONCE at startup, into which hot weights are filled in
//! place (`pread`/memcpy, no separate H2D) and handed to the kernels as raw device
//! pointers. This is the foundation the descriptor-array MoE kernel (D1) and the
//! flash attention kernel (D2) read from — weights stream through the GPU
//! exactly once, no host↔device copy on the hot path.
//!
//! Design constraints from the colibri campaign (PLAN.md bottleneck #6): the
//! amdgpu large-GTT wedge bug fires on mid-session re-allocations and when a
//! foreign tenant lands, so the tier is a single up-front allocation, never
//! resized, and construction refuses to proceed if another process already
//! holds GPU memory (sole-tenant rule). A bump allocator is the whole allocator:
//! the pin is filled once and freed as a unit, so a free-list would be dead
//! weight.
//!
//! The whole module is empty without the `rocm` feature — there is no device to
//! allocate on in a CPU-only dev build.

#[cfg(feature = "rocm")]
mod ffi {
    use std::ffi::c_void;
    // The HIP runtime memory API (libamdhip64, already linked by build.rs).
    // These are runtime calls, not kernels — no hipcc launcher needed.
    unsafe extern "C" {
        pub fn hipMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
        pub fn hipFree(ptr: *mut c_void) -> i32;
        pub fn hipMemcpy(dst: *mut c_void, src: *const c_void, size: usize, kind: i32) -> i32;
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    // VMM device-local host-fillable allocator (kernels/vmm.hip, in
    // librivolikernels.a). Gives MTYPE_RW device-bandwidth memory the CPU can fill
    // in place — see docs/hip-apu-memory.md.
    unsafe extern "C" {
        pub fn rivoli_vmm_alloc(
            size: usize,
            dev: i32,
            out_ptr: *mut *mut c_void,
            out_handle: *mut u64,
            out_mapped: *mut usize,
        ) -> i32;
        pub fn rivoli_vmm_free(ptr: *mut c_void, handle: u64, mapped: usize) -> i32;
    }
    pub const HIP_MEMCPY_H2D: i32 = 1;
    pub const HIP_MEMCPY_D2H: i32 = 2;
    pub const HIP_SUCCESS: i32 = 0;
}

#[cfg(feature = "rocm")]
pub use tier::{DeviceBuf, DeviceTier, VmmBuf, mem_info};

#[cfg(feature = "rocm")]
mod tier {
    use super::ffi::*;
    use anyhow::{Result, bail, ensure};
    use std::ffi::c_void;
    use tracing::warn;

    /// Free device memory and total, in bytes.
    pub fn mem_info() -> Result<(usize, usize)> {
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both out-pointers are valid for a usize write.
        let e = unsafe { hipMemGetInfo(&mut free, &mut total) };
        ensure!(e == HIP_SUCCESS, "hipMemGetInfo failed ({e})");
        Ok((free, total))
    }

    /// A resident device slab with a bump cursor. Weights placed into it stay
    /// resident for the run; the kernels read them by the returned device
    /// pointer. Freed as a unit on drop.
    pub struct DeviceTier {
        slab: VmmBuf, // device-local, host-fillable — weights load straight in
        capacity: usize,
        used: usize,
    }

    impl DeviceTier {
        /// Foreign GPU memory beyond this at startup ⇒ refuse to start (another
        /// tenant; a 21 GB foreign GTT allocation landing mid-run was the wedge
        /// aggravator behind most colibri device losses).
        const SOLE_TENANT_MAX_GTT: u64 = 1 << 30; // 1 GiB slack for the compositor
        /// Leave this much device memory free beyond the tier (driver scratch,
        /// kernel dispatch buffers, the cold-fetch slabs that arrive in M4).
        const HEADROOM: usize = 4 << 30; // 4 GiB

        /// Allocate the resident tier once. Fails loudly if a foreign tenant is
        /// present or the request doesn't fit free device memory with headroom.
        pub fn new(capacity: usize) -> Result<Self> {
            Self::guard_sole_tenant()?;
            let (free, _total) = mem_info()?;
            ensure!(
                capacity + Self::HEADROOM <= free,
                "device tier {capacity} + {} headroom > free {free}",
                Self::HEADROOM
            );
            // Device-local (MTYPE_RW, full bandwidth) AND host-fillable, so weights
            // load straight into the slab (no separate hipMalloc + H2D). `VmmBuf`
            // owns/frees the mapping.
            let slab = VmmBuf::new(capacity)?;
            Ok(Self {
                slab,
                capacity,
                used: 0,
            })
        }

        /// Refuse to start if another process already holds GPU memory. Reads the
        /// amdgpu GTT counter directly (the HIP API reports our own view, not the
        /// whole-device tenancy this guard needs).
        fn guard_sole_tenant() -> Result<()> {
            // Scan card0 then card1 — this box has one amdgpu today, but a second
            // GPU must not silently defeat the guard; the first readable node wins.
            // The guard fails OPEN (no readable node → proceed) so a non-amdgpu or
            // containerized box isn't blocked — but it says so LOUDLY, because a
            // silently-inert safety guard is worse than none (the operator would
            // believe they're protected). M5's clean-refusal gate depends on this.
            let gtt: u64 = 'read: {
                for card in ["card0", "card1"] {
                    let path = format!("/sys/class/drm/{card}/device/mem_info_gtt_used");
                    match std::fs::read_to_string(&path) {
                        Ok(s) => match s.trim().parse() {
                            Ok(v) => break 'read v,
                            Err(e) => {
                                warn!("sole-tenant guard DISABLED: {path} unparseable ({e})");
                                return Ok(());
                            }
                        },
                        Err(_) => continue, // absent card — try the next
                    }
                }
                warn!(
                    "sole-tenant guard DISABLED: no /sys/class/drm/card{{0,1}}/device/mem_info_gtt_used readable"
                );
                return Ok(());
            };
            if gtt > Self::SOLE_TENANT_MAX_GTT {
                bail!(
                    "refusing to start: {:.1} GiB GPU memory already in use by another \
                     tenant (sole-tenant rule; free the GPU and retry)",
                    gtt as f64 / (1u64 << 30) as f64
                );
            }
            Ok(())
        }

        /// Reserve `len` bytes (256-aligned) and return a host-writable pointer to
        /// fill in place (`pread`/memcpy). Errors if the tier is full — the pin is
        /// sized to fit, so OOM here is a budgeting bug, not a runtime condition.
        pub fn reserve(&mut self, len: usize) -> Result<*mut u8> {
            let off = (self.used + 255) & !255;
            ensure!(
                off + len <= self.capacity,
                "device tier OOM: need {len} at offset {off}, capacity {}",
                self.capacity
            );
            self.used = off + len;
            // SAFETY: off+len ≤ capacity (checked); within the slab.
            Ok(unsafe { self.slab.ptr_mut().add(off) })
        }

        pub fn used(&self) -> usize {
            self.used
        }
    }

    /// A standalone mutable device buffer — for per-token activations, the
    /// descriptor array, and the MoE accumulator that the kernels write. Unlike
    /// [`DeviceTier`] (append-only resident weights), a `DeviceBuf` is sized once
    /// and rewritten each token via `copy_in_at`. Freed on drop.
    pub struct DeviceBuf {
        ptr: *mut u8,
        len: usize,
    }

    impl DeviceBuf {
        /// Allocate `len` uninitialized device bytes (e.g. a cold-expert slot
        /// filled later via [`DeviceBuf::copy_in_at`]).
        pub fn new(len: usize) -> Result<Self> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: ptr is a valid out-pointer; owns `len` bytes freed in Drop.
            let e = unsafe { hipMalloc(&mut ptr, len) };
            ensure!(
                e == HIP_SUCCESS && !ptr.is_null(),
                "hipMalloc({len}) failed ({e})"
            );
            Ok(Self {
                ptr: ptr as *mut u8,
                len,
            })
        }

        /// Overwrite `bytes` at byte offset `off` (partial H2D) — grows a KV
        /// slab by one token or fills a cold-expert slot without reallocating.
        pub fn copy_in_at(&mut self, off: usize, bytes: &[u8]) -> Result<()> {
            ensure!(
                off + bytes.len() <= self.len,
                "copy_in_at {off}+{} > buf len {}",
                bytes.len(),
                self.len
            );
            // SAFETY: off+len ≤ len (checked); dst is within the buffer.
            let e = unsafe {
                hipMemcpy(
                    self.ptr.add(off) as *mut c_void,
                    bytes.as_ptr() as *const c_void,
                    bytes.len(),
                    HIP_MEMCPY_H2D,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy H2D failed ({e})");
            Ok(())
        }

        /// DIAGNOSTIC: read `len` bytes back from an arbitrary device pointer.
        ///
        /// Unlike the other copy_out family this is not tied to a `DeviceBuf` — it
        /// exists so the expert-checksum probe can hash weights straight out of the
        /// pool slab, which the engine addresses as raw pointers inside descriptors
        /// rather than as owned buffers.
        ///
        /// # Safety
        /// `src` must point to at least `len` readable device bytes, and no kernel
        /// may be concurrently writing them (call after a `device_sync`).
        pub unsafe fn copy_out_raw(src: *const u8, len: usize, out: &mut Vec<u8>) -> Result<()> {
            // No zero-fill: the D2H overwrites every byte, so reserve + set_len
            // after the copy (resize(len, 0) was a pure-waste memset per call).
            out.clear();
            out.reserve(len);
            // SAFETY: caller guarantees `src` is readable for `len`; dst owns
            // `len` reserved bytes.
            let e = unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    src as *const c_void,
                    len,
                    HIP_MEMCPY_D2H,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy D2H (raw) failed ({e})");
            // SAFETY: the copy above initialized the first `len` bytes.
            unsafe { out.set_len(len) };
            Ok(())
        }

        /// Copy the whole buffer back to host as a fresh `Vec` (the ergonomic form;
        /// the per-token decode path uses [`DeviceBuf::copy_out_into`] to reuse a
        /// buffer instead).
        pub fn copy_out(&self) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            self.copy_out_into(&mut out)?;
            Ok(out)
        }

        /// Copy the FIRST `len` bytes back into `out` (reused: cleared then
        /// resized). For partially-written buffers — e.g. the indexer's score
        /// slab, sized to max_ctx but holding only `nt` scores this step — so
        /// the D2H moves nt·4 bytes, not the whole slab.
        pub fn copy_out_prefix(&self, out: &mut Vec<u8>, len: usize) -> Result<()> {
            ensure!(
                len <= self.len,
                "copy_out_prefix {len} > buf len {}",
                self.len
            );
            // No zero-fill: the D2H overwrites the whole range (see copy_out_raw).
            out.clear();
            out.reserve(len);
            // SAFETY: source has `len <= self.len` bytes; dest owns `len` reserved bytes.
            let e = unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    len,
                    HIP_MEMCPY_D2H,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy D2H failed ({e})");
            // SAFETY: the copy above initialized the first `len` bytes.
            unsafe { out.set_len(len) };
            Ok(())
        }

        /// Copy the whole buffer back into `out` (a caller-owned buffer reused
        /// across tokens: cleared then refilled to `len`, so the per-token decode
        /// D2H allocates nothing once `out` has grown to size).
        pub fn copy_out_into(&self, out: &mut Vec<u8>) -> Result<()> {
            // No zero-fill: the D2H overwrites the whole buffer (see copy_out_raw).
            out.clear();
            out.reserve(self.len);
            // SAFETY: both regions are `len` bytes; dest owns them reserved.
            let e = unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    self.len,
                    HIP_MEMCPY_D2H,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy D2H failed ({e})");
            // SAFETY: the copy above initialized the first `len` bytes.
            unsafe { out.set_len(self.len) };
            Ok(())
        }

        pub fn ptr(&self) -> *const u8 {
            self.ptr
        }
        pub fn ptr_mut(&mut self) -> *mut u8 {
            self.ptr
        }
    }

    impl Drop for DeviceBuf {
        fn drop(&mut self) {
            // SAFETY: ptr came from hipMalloc and is freed exactly once.
            unsafe { hipFree(self.ptr as *mut c_void) };
        }
    }

    /// Device-local memory (MTYPE_RW, full ~220 GB/s GPU bandwidth) that the CPU
    /// can also fill IN PLACE — HIP VMM allocates the physical pages device-local
    /// and grants the host an access mapping (APU unified addressing). So a cold
    /// expert is `pread`/memcpy'd straight in ([`VmmBuf::write_at`], a plain host
    /// memcpy, no `hipMemcpy`/sync) and the GPU reads it with NO coherent-pool read
    /// tax — unlike a `hipHostMalloc` slot, which always maps system-domain and
    /// reads ~9% slower. `ptr()` is GPU-usable. See docs/hip-apu-memory.md.
    pub struct VmmBuf {
        ptr: *mut u8,
        handle: u64,
        mapped: usize, // granularity-rounded size, needed verbatim to free
    }

    impl VmmBuf {
        pub fn new(len: usize) -> Result<Self> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let mut handle: u64 = 0;
            let mut mapped: usize = 0;
            // SAFETY: out-pointers are valid; the shim owns the mapping until free.
            let e = unsafe { rivoli_vmm_alloc(len, 0, &mut ptr, &mut handle, &mut mapped) };
            ensure!(
                e == 0 && !ptr.is_null(),
                "rivoli_vmm_alloc({len}) failed ({e})"
            );
            Ok(Self {
                ptr: ptr as *mut u8,
                handle,
                mapped,
            })
        }

        /// GPU-usable pointer (device-local under unified addressing).
        pub fn ptr(&self) -> *const u8 {
            self.ptr
        }
        /// Host-writable pointer — the caller fills in place (`pread`/memcpy, or an
        /// io_uring O_DIRECT DMA), no `hipMemcpy`/sync.
        ///
        /// Ordering, CPU memcpy fill: the GPU sees the host stores at the next kernel
        /// launch on THIS (the single HIP/decode) thread — the launch's dispatch
        /// packet carries a release fence (drains the CPU store buffer) + a
        /// system-scope acquire (invalidates GPU caches), and gfx1151's coherent
        /// fabric lets the host-granted device-local mapping participate.
        ///
        /// Ordering, io_uring DMA fill (the cold-expert stream): the bytes are
        /// written by the kernel's DMA engine, NOT CPU stores, so the store-buffer
        /// release fence above does not apply to them. Visibility rests instead on
        /// (a) the dispatch packet's system-scope ACQUIRE invalidating the GPU caches
        /// before the reading kernel, and (b) the completed io_uring drain
        /// (`Streamer::drain`) happening-before that launch, plus the end-of-layer
        /// [`crate::hip::device_sync`] fencing slot reuse. No CPU store fence is
        /// involved on this path.
        ///
        /// Verified CPU->GPU coherent on this APU (docs/probes/vmm_probe.cpp, incl.
        /// `pread`). NOT a HIP contract for arbitrary hardware: a port off gfx1151,
        /// or a fill from a background HIP thread, must re-verify or insert an
        /// explicit fence.
        pub fn ptr_mut(&mut self) -> *mut u8 {
            self.ptr
        }
    }

    impl Drop for VmmBuf {
        fn drop(&mut self) {
            // SAFETY: (ptr,handle,mapped) came from rivoli_vmm_alloc, freed once.
            unsafe { rivoli_vmm_free(self.ptr as *mut c_void, self.handle, self.mapped) };
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;

        #[test]
        fn tier_roundtrips_placed_bytes() {
            // Small tier; reserve+fill two patterns via the host-writable ptr, read
            // both back in place (VMM is host-readable), check the 256-align gap
            // doesn't corrupt either.
            let mut tier = DeviceTier::new(4 << 20).expect("alloc tier");
            let a: Vec<u8> = (0..1000u32).map(|i| (i & 0xff) as u8).collect();
            let b: Vec<u8> = (0..500u32).map(|i| ((i * 7) & 0xff) as u8).collect();
            let pa = tier.reserve(a.len()).expect("reserve a");
            // SAFETY: pa owns a.len() host-writable bytes just reserved.
            unsafe { std::ptr::copy_nonoverlapping(a.as_ptr(), pa, a.len()) };
            let pb = tier.reserve(b.len()).expect("reserve b");
            // SAFETY: pb owns b.len() host-writable bytes just reserved.
            unsafe { std::ptr::copy_nonoverlapping(b.as_ptr(), pb, b.len()) };
            assert_ne!(pa, pb);
            // SAFETY: read back the bytes just written (device-local, host-mapped).
            let (ra, rb) = unsafe {
                (
                    std::slice::from_raw_parts(pa, a.len()),
                    std::slice::from_raw_parts(pb, b.len()),
                )
            };
            assert_eq!(ra, &a[..]);
            assert_eq!(rb, &b[..]);
            // 1000 bumps to 1024 (256-aligned) before b lands.
            assert_eq!(tier.used(), 1024 + 500);
        }

        #[test]
        fn tier_rejects_overflow() {
            let mut tier = DeviceTier::new(1 << 20).expect("alloc tier");
            assert!(tier.reserve((1 << 20) + 1).is_err());
        }
    }
}
