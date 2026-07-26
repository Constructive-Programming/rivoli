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
//! The whole module is empty without a backend feature — there is no device to
//! allocate on in a CPU-only dev build. `rocm` and `vulkan` each supply the same
//! four names (`mem_info`, `DeviceTier`, `DeviceBuf`, `VmmBuf`) so everything above
//! `crate::device::` reads identically either way; the Vulkan half lives in
//! `vktier` and differs in exactly one respect, which is the reason it is not a
//! mechanical transliteration: a host pointer and a device address are two unrelated
//! numbers there (docs/VULKAN.md, "Host pointer != device address").

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

        /// Reserve `bytes.len()` device bytes (256-aligned), fill them, and return
        /// the placed weight's DEVICE pointer.
        ///
        /// One call rather than reserve-then-memcpy, because the two are not
        /// separable in general: filling means writing at a HOST address, and only
        /// the tier knows the relationship between that and the device address it
        /// hands back. Under HIP's unified addressing they are the same number and
        /// this is a plain memcpy; that coincidence is not portable, and callers must
        /// not depend on it. Making this the only way in also means a caller cannot
        /// hold an unfilled device pointer, and costs `pin.rs` five `unsafe` blocks.
        ///
        /// Errors if the tier is full — the pin is sized to fit, so OOM here is a
        /// budgeting bug, not a runtime condition.
        pub fn place(&mut self, bytes: &[u8]) -> Result<*mut u8> {
            let dst = self.reserve(bytes.len())?;
            // SAFETY: `dst` owns the `bytes.len()` bytes just reserved, and the source
            // is a live slice; the regions cannot overlap (one is the mmap'd artifact
            // or a fresh Vec, the other the device slab).
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
            Ok(dst)
        }

        /// Bump the cursor by `len` (256-aligned) and return the slab pointer.
        /// Private: [`DeviceTier::place`] is the only way to obtain a tier pointer,
        /// so one cannot escape unfilled.
        fn reserve(&mut self, len: usize) -> Result<*mut u8> {
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

        /// Copy the whole buffer back to host as a fresh `Vec` (the ergonomic form
        /// the kernel oracle tests use; the per-token decode path uses
        /// [`DeviceBuf::copy_out_into`] to reuse a buffer instead).
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
        /// before the reading kernel, and (b) the read's `Signal` (armed on the fetch
        /// stream after [`Streamer::reap`] enqueued its copy) happening-before that
        /// launch, plus the end-of-layer [`crate::hip::device_sync`] fencing slot
        /// reuse. No CPU store fence is involved on this path.
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
            let pa = tier.place(&a).expect("place a");
            let pb = tier.place(&b).expect("place b");
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
            assert_eq!(tier.used, 1024 + 500);
        }

        #[test]
        fn tier_rejects_overflow() {
            let mut tier = DeviceTier::new(1 << 20).expect("alloc tier");
            assert!(tier.place(&vec![0u8; (1 << 20) + 1]).is_err());
        }
    }
}

#[cfg(feature = "vulkan")]
pub use vktier::{DeviceBuf, DeviceTier, VmmBuf, mem_info};

/// The same four names over Vulkan. Every allocation is a [`crate::vk::Buf`] — one
/// `VkBuffer` on its own `DEVICE_LOCAL | HOST_VISIBLE` allocation, permanently mapped
/// — which is the Vulkan spelling of what `kernels/vmm.hip` gives the HIP side, so the
/// shapes above (bump-allocated resident slab, rewritten per-token buffer) carry over
/// unchanged.
///
/// What does NOT carry over is that a HIP device pointer is also a host pointer. Here
/// the two bases are unrelated numbers, so every type below is explicit about which one
/// it hands out: the tier writes through the mapping and returns device addresses, and
/// `DeviceBuf`'s `copy_*` were already explicit transfers and need no change at all.
#[cfg(feature = "vulkan")]
mod vktier {
    use crate::vk::Buf;
    use anyhow::{Result, ensure};

    /// Free device memory and total, in bytes. Live free figure via
    /// VK_EXT_memory_budget — see [`crate::vk::Gpu::mem_info`] for what happens when
    /// the extension is absent.
    pub fn mem_info() -> Result<(usize, usize)> {
        Ok(crate::vk::gpu()?.mem_info())
    }

    /// A resident device slab with a bump cursor. Weights placed into it stay resident
    /// for the run; the kernels read them by the returned device address. Freed as a
    /// unit on drop.
    pub struct DeviceTier {
        slab: Buf,
        capacity: usize,
        used: usize,
    }

    impl DeviceTier {
        /// Leave this much device memory free beyond the tier (driver scratch, kernel
        /// dispatch buffers, the cold-fetch slabs that arrive in M4).
        const HEADROOM: usize = 4 << 30; // 4 GiB

        /// Allocate the resident tier once. Fails if the request doesn't fit free
        /// device memory with headroom.
        ///
        /// No sole-tenant guard here, unlike the HIP path. That guard reads amdgpu's
        /// `mem_info_gtt_used` sysfs node, which reports whole-device tenancy and knows
        /// nothing about which API allocated it — so running it again under Vulkan would
        /// be the same check against the same counter, and on a non-amdgpu driver (the
        /// portability this backend exists for) it fails open and warns, which is noise.
        /// ponytail: one copy of a backend-independent guard, and it stays where the
        /// wedge bug it defends against actually was.
        pub fn new(capacity: usize) -> Result<Self> {
            let (free, _total) = mem_info()?;
            ensure!(
                capacity + Self::HEADROOM <= free,
                "device tier {capacity} + {} headroom > free {free}",
                Self::HEADROOM
            );
            Ok(Self {
                slab: Buf::new(capacity)?,
                capacity,
                used: 0,
            })
        }

        /// Reserve `bytes.len()` device bytes (256-aligned), fill them, and return the
        /// placed weight's DEVICE address.
        ///
        /// This is where the HIP path's convenient coincidence stops being one. Filling
        /// means writing at a HOST address; the kernels need a DEVICE address; only the
        /// tier knows both bases, so it is the only thing that can translate. Fusing
        /// reserve and fill keeps that translation in one place, once per placement at
        /// startup, and means a caller cannot end up holding a device address for bytes
        /// that were never written.
        ///
        /// Errors if the tier is full — the pin is sized to fit, so OOM here is a
        /// budgeting bug, not a runtime condition.
        pub fn place(&mut self, bytes: &[u8]) -> Result<*mut u8> {
            let off = (self.used + 255) & !255;
            ensure!(
                off + bytes.len() <= self.capacity,
                "device tier OOM: need {} at offset {off}, capacity {}",
                bytes.len(),
                self.capacity
            );
            self.slab.write_at(off, bytes)?;
            self.used = off + bytes.len();
            Ok((self.slab.ptr() as usize + off) as *mut u8)
        }

        /// Read the slab's first `len` bytes back to the host.
        ///
        /// Test-only, and the tests need it: what [`DeviceTier::place`] returns is a
        /// device address, so there is no other way to observe what a placement wrote.
        #[cfg(test)]
        fn read_prefix(&self, len: usize) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            self.slab.read_into(&mut out, len)?;
            Ok(out)
        }
    }

    /// A standalone mutable device buffer — per-token activations, the descriptor
    /// array, the MoE accumulator. Sized once and rewritten each token via
    /// `copy_in_at`, exactly as under HIP: this type always spelled its transfers out,
    /// so it is the one that ports with no design change.
    ///
    /// The `copy_out*` family reads the mapping directly rather than issuing a
    /// `vkCmdCopyBuffer`, so unlike `hipMemcpy` it does NOT synchronise: call
    /// [`crate::vk::device_sync`] first if a kernel wrote the bytes you are reading.
    pub struct DeviceBuf {
        buf: Buf,
        len: usize,
    }

    impl DeviceBuf {
        /// Allocate `len` uninitialized device bytes (e.g. a cold-expert slot filled
        /// later via [`DeviceBuf::copy_in_at`]).
        pub fn new(len: usize) -> Result<Self> {
            Ok(Self {
                buf: Buf::new(len)?,
                len,
            })
        }

        /// Overwrite `bytes` at byte offset `off` — grows a KV slab by one token or
        /// fills a cold-expert slot without reallocating.
        pub fn copy_in_at(&mut self, off: usize, bytes: &[u8]) -> Result<()> {
            // Bounds-checked here as well as in `write_at` so the message names the
            // caller's operation; the engine greps these strings when a slab is
            // mis-sized, and "write_at" would point at the wrong layer.
            ensure!(
                off.checked_add(bytes.len()).is_some_and(|e| e <= self.len),
                "copy_in_at {off}+{} > buf len {}",
                bytes.len(),
                self.len
            );
            self.buf.write_at(off, bytes)
        }

        /// Copy the whole buffer back to host as a fresh `Vec` (the ergonomic form the
        /// kernel oracle tests use; the per-token decode path uses
        /// [`DeviceBuf::copy_out_into`] to reuse a buffer instead).
        pub fn copy_out(&self) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            self.copy_out_into(&mut out)?;
            Ok(out)
        }

        /// Copy the FIRST `len` bytes back into `out` (reused: cleared then resized).
        /// For partially-written buffers — e.g. the indexer's score slab, sized to
        /// max_ctx but holding only `nt` scores this step.
        pub fn copy_out_prefix(&self, out: &mut Vec<u8>, len: usize) -> Result<()> {
            ensure!(
                len <= self.len,
                "copy_out_prefix {len} > buf len {}",
                self.len
            );
            self.buf.read_into(out, len)
        }

        /// Copy the whole buffer back into `out` (a caller-owned buffer reused across
        /// tokens, so the per-token readback allocates nothing once it has grown).
        pub fn copy_out_into(&self, out: &mut Vec<u8>) -> Result<()> {
            self.buf.read_into(out, self.len)
        }

        // No `copy_out_raw`. Its HIP twin takes a bare device pointer with no owning
        // object, which works only because that number is also a host address. Here it
        // is not, and an address carries no identity — recovering the mapping means
        // asking `vk.rs`'s address registry which `Buf` contains the range and reading
        // through that. Its one caller is the `trace`-feature expert-checksum probe, so
        // that plumbing would exist purely to serve a diagnostic: build it if and when
        // the probe is wanted under Vulkan, and hash through the owning `DeviceBuf`
        // until then.

        pub fn ptr(&self) -> *const u8 {
            self.buf.ptr()
        }
        pub fn ptr_mut(&mut self) -> *mut u8 {
            self.buf.ptr_mut()
        }
    }

    /// The routed-expert pool's backing allocation: device-local at full bandwidth AND
    /// host-writable in place, so a cold expert lands in it without an H2D copy.
    ///
    /// Where the HIP `VmmBuf` hands out ONE pointer that `pin.rs` uses simultaneously as
    /// the io_uring O_DIRECT DMA target and as the base for every expert descriptor's
    /// six device pointers (`ArenaPool::ptr`, `src/pin.rs`), those are two numbers here
    /// and this type must hand them out separately — see docs/VULKAN.md, "Host pointer
    /// != device address". [`VmmBuf::ptr`] is the device base, for descriptor
    /// arithmetic, [`VmmBuf::host_mut`] is the DMA target, and callers must say which
    /// one they mean.
    pub struct VmmBuf {
        buf: Buf,
    }

    impl VmmBuf {
        pub fn new(len: usize) -> Result<Self> {
            let mut buf = Buf::new(len)?;
            // The cold-expert streamer DMAs O_DIRECT straight into this mapping — no
            // bounce, no staging copy — which is legal only while the base is
            // 4096-aligned. `vkMapMemory` guarantees `minMemoryMapAlignment`, which is
            // page-sized on every driver we have seen and is NOT required to be. That
            // "in practice" is exactly the kind of premise that fails on someone else's
            // machine, as EINVAL from the first read inside the reaper rather than
            // anything legible, so it is checked here instead of assumed.
            //
            // The other half of the requirement, the slot STRIDE, is VQ_ALIGN and is
            // enforced by the arena; see `crate::format` and `slot_span` in pin.rs.
            let base = buf.host_mut() as usize;
            ensure!(
                base.is_multiple_of(crate::vk::O_DIRECT_ALIGN),
                "mapped base {base:#x} is not {}-byte aligned, so io_uring O_DIRECT \
                 reads into the routed pool would fail with EINVAL",
                crate::vk::O_DIRECT_ALIGN
            );
            Ok(Self { buf })
        }

        /// The DEVICE base. Descriptor pointers are computed from this; it is not
        /// host-dereferenceable.
        pub fn ptr(&self) -> *const u8 {
            self.buf.ptr()
        }

        /// The HOST base — the io_uring O_DIRECT DMA target, and the only one of the
        /// two that may be dereferenced on the CPU.
        pub fn host_mut(&mut self) -> *mut u8 {
            self.buf.host_mut()
        }

        /// Host-fill `bytes` at byte offset `off`, in place — no staging buffer and no
        /// transfer command. The write is visible to any kernel submitted after it
        /// returns (HOST_COHERENT, and `vkQueueSubmit` implies the host-write barrier).
        pub fn write_at(&mut self, off: usize, bytes: &[u8]) -> Result<()> {
            self.buf.write_at(off, bytes)
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;

        #[test]
        fn tier_roundtrips_placed_bytes() {
            let mut tier = DeviceTier::new(4 << 20).expect("alloc tier");
            let a: Vec<u8> = (0..1000u32).map(|i| (i & 0xff) as u8).collect();
            let b: Vec<u8> = (0..500u32).map(|i| ((i * 7) & 0xff) as u8).collect();
            let pa = tier.place(&a).expect("place a");
            let pb = tier.place(&b).expect("place b");
            assert_ne!(pa, pb);
            // 1000 bumps to 1024 (256-aligned) before b lands, and the returned device
            // addresses track those offsets.
            assert_eq!(tier.used, 1024 + 500);
            assert_eq!(pb as usize - pa as usize, 1024);
            // NOT read back through `pa`/`pb` the way the HIP test does: those are
            // device addresses, and dereferencing one on the host is a segfault at best.
            // The tier's own slab is the only host view of these bytes.
            let got = tier.read_prefix(tier.used).expect("read back");
            assert_eq!(&got[..a.len()], &a[..], "a corrupted");
            assert_eq!(&got[1024..1024 + b.len()], &b[..], "b corrupted");
        }

        #[test]
        fn tier_rejects_overflow() {
            let mut tier = DeviceTier::new(1 << 20).expect("alloc tier");
            assert!(tier.place(&vec![0u8; (1 << 20) + 1]).is_err());
        }

        #[test]
        fn devicebuf_roundtrips() {
            let mut d = DeviceBuf::new(64).expect("alloc");
            d.copy_in_at(0, &[0u8; 64]).expect("zero");
            let bytes: Vec<u8> = (1..=32u8).collect();
            d.copy_in_at(8, &bytes).expect("copy in");
            let out = d.copy_out().expect("copy out");
            assert_eq!(out.len(), 64);
            assert_eq!(&out[8..40], &bytes[..]);
            assert!(out[..8].iter().all(|&v| v == 0), "wrote before the offset");
            assert!(out[40..].iter().all(|&v| v == 0), "wrote past the bytes");
            // The prefix form moves only `len` bytes, and `out` is reused, so its final
            // length is the proof it did not fall back to the whole buffer.
            let mut pre = vec![0xAAu8; 999];
            d.copy_out_prefix(&mut pre, 16).expect("prefix");
            assert_eq!(pre, &out[..16]);
            assert!(d.copy_out_prefix(&mut pre, 65).is_err());
            assert!(d.copy_in_at(48, &bytes).is_err());
        }

        // The spec's `VmmBuf::ptr()` vs `host_mut()` inequality test is deliberately
        // absent, not forgotten: `host_mut` does not exist yet (see the note on
        // `VmmBuf`). It is the property worth pinning — a maintainer "simplifying" two
        // accessors back into one is exactly the regression that reads as garbage
        // weights rather than a crash — so it belongs here the moment the accessor does.
        #[test]
        fn vmmbuf_allocates_distinct_device_bases() {
            let a = VmmBuf::new(4096).expect("alloc a");
            let b = VmmBuf::new(4096).expect("alloc b");
            assert!(!a.ptr().is_null());
            assert_ne!(a.ptr(), b.ptr());
        }
    }
}
