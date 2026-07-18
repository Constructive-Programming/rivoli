//! The resident device tier: one `hipMalloc` slab allocated ONCE at startup,
//! into which hot weights are placed and handed to the kernels as raw device
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
        pub fn hipMemset(ptr: *mut c_void, value: i32, size: usize) -> i32;
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
        base: *mut u8,
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
            let mut base: *mut c_void = std::ptr::null_mut();
            // SAFETY: base is a valid out-pointer; on success it owns `capacity`
            // bytes of device memory freed in Drop.
            let e = unsafe { hipMalloc(&mut base, capacity) };
            ensure!(
                e == HIP_SUCCESS && !base.is_null(),
                "hipMalloc({capacity}) failed ({e})"
            );
            Ok(Self {
                base: base as *mut u8,
                capacity,
                used: 0,
            })
        }

        /// Refuse to start if another process already holds GPU memory. Reads the
        /// amdgpu GTT counter directly (the HIP API reports our own view, not the
        /// whole-device tenancy this guard needs).
        fn guard_sole_tenant() -> Result<()> {
            // ponytail: card0 hardcoded — this box has one amdgpu. Generalize to
            // scan /sys/class/drm/card*/device only if a second GPU ever appears.
            let path = "/sys/class/drm/card0/device/mem_info_gtt_used";
            // The guard fails OPEN (a bad read → proceed) so a non-amdgpu or
            // containerized box isn't blocked — but it says so LOUDLY, because a
            // silently-inert safety guard is worse than none (the operator would
            // believe they're protected). M5's clean-refusal gate depends on this.
            let gtt: u64 = match std::fs::read_to_string(path) {
                Ok(s) => match s.trim().parse() {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("sole-tenant guard DISABLED: {path} unparseable ({e})");
                        return Ok(());
                    }
                },
                Err(e) => {
                    warn!("sole-tenant guard DISABLED: {path} unreadable ({e})");
                    return Ok(());
                }
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

        /// Copy `bytes` into the tier (256-aligned) and return the device pointer.
        /// Errors if the tier is full — the pin is sized to fit, so OOM here is a
        /// budgeting bug, not a runtime condition to absorb.
        pub fn place(&mut self, bytes: &[u8]) -> Result<*const u8> {
            let off = (self.used + 255) & !255;
            ensure!(
                off + bytes.len() <= self.capacity,
                "device tier OOM: need {} at offset {off}, capacity {}",
                bytes.len(),
                self.capacity
            );
            // SAFETY: off+len ≤ capacity (checked); dst is within the slab.
            let dst = unsafe { self.base.add(off) };
            let e = unsafe {
                hipMemcpy(
                    dst as *mut c_void,
                    bytes.as_ptr() as *const c_void,
                    bytes.len(),
                    HIP_MEMCPY_H2D,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy H2D failed ({e})");
            self.used = off + bytes.len();
            Ok(dst as *const u8)
        }

        /// Copy `len` bytes back from a device pointer (verification/debug — the
        /// hot path never reads device memory back to host).
        pub fn copy_out(&self, ptr: *const u8, len: usize) -> Result<Vec<u8>> {
            let mut out = vec![0u8; len];
            // SAFETY: caller passes a pointer + len returned by `place`.
            let e = unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    ptr as *const c_void,
                    len,
                    HIP_MEMCPY_D2H,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy D2H failed ({e})");
            Ok(out)
        }

        pub fn used(&self) -> usize {
            self.used
        }
    }

    impl Drop for DeviceTier {
        fn drop(&mut self) {
            // SAFETY: base came from hipMalloc and is freed exactly once.
            unsafe { hipFree(self.base as *mut c_void) };
        }
    }

    /// A standalone mutable device buffer — for per-token activations, the
    /// descriptor array, and the MoE accumulator that the kernels write. Unlike
    /// [`DeviceTier`] (append-only resident weights), a `DeviceBuf` is sized once
    /// and rewritten each token via `copy_in`/`zero`. Freed on drop.
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

        /// Allocate `len` device bytes, zeroed.
        pub fn zeroed(len: usize) -> Result<Self> {
            let b = Self::new(len)?;
            // SAFETY: ptr owns len bytes.
            let e = unsafe { hipMemset(b.ptr as *mut c_void, 0, len) };
            ensure!(e == HIP_SUCCESS, "hipMemset failed ({e})");
            Ok(b)
        }

        /// Allocate and fill from host bytes.
        pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
            let mut b = Self::new(bytes.len())?;
            b.copy_in(bytes)?;
            Ok(b)
        }

        /// Overwrite the buffer from host bytes (`bytes.len()` must be `len`).
        pub fn copy_in(&mut self, bytes: &[u8]) -> Result<()> {
            ensure!(
                bytes.len() == self.len,
                "copy_in {} != buf len {}",
                bytes.len(),
                self.len
            );
            // SAFETY: both regions are `len` bytes.
            let e = unsafe {
                hipMemcpy(
                    self.ptr as *mut c_void,
                    bytes.as_ptr() as *const c_void,
                    self.len,
                    HIP_MEMCPY_H2D,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy H2D failed ({e})");
            Ok(())
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

        /// Zero the buffer (before an accumulating kernel).
        pub fn zero(&mut self) -> Result<()> {
            // SAFETY: ptr owns len bytes.
            let e = unsafe { hipMemset(self.ptr as *mut c_void, 0, self.len) };
            ensure!(e == HIP_SUCCESS, "hipMemset failed ({e})");
            Ok(())
        }

        /// Copy the whole buffer back to host.
        pub fn copy_out(&self) -> Result<Vec<u8>> {
            let mut out = vec![0u8; self.len];
            // SAFETY: both regions are `len` bytes.
            let e = unsafe {
                hipMemcpy(
                    out.as_mut_ptr() as *mut c_void,
                    self.ptr as *const c_void,
                    self.len,
                    HIP_MEMCPY_D2H,
                )
            };
            ensure!(e == HIP_SUCCESS, "hipMemcpy D2H failed ({e})");
            Ok(out)
        }

        pub fn ptr(&self) -> *const u8 {
            self.ptr
        }
        pub fn ptr_mut(&mut self) -> *mut u8 {
            self.ptr
        }
        pub fn len(&self) -> usize {
            self.len
        }
        pub fn is_empty(&self) -> bool {
            self.len == 0
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
        len: usize,    // usable bytes requested
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
                len,
            })
        }

        /// Host memcpy `bytes` at `off` (device-local memory is host-writable via
        /// the VMM host grant — no `hipMemcpy`, no sync).
        ///
        /// Ordering: the GPU sees these stores at the next kernel launch on THIS
        /// (the single HIP/decode) thread — the launch's dispatch packet carries a
        /// release fence (drains the CPU store buffer) + a system-scope acquire
        /// (invalidates GPU caches), and gfx1151's coherent fabric lets the
        /// host-granted device-local mapping participate. Verified CPU->GPU
        /// coherent on this APU (docs/probes/vmm_probe.cpp). NOT a HIP contract for
        /// arbitrary hardware: a port off gfx1151, or issuing the fill from a
        /// background HIP thread, must re-verify or insert an explicit fence.
        pub fn write_at(&mut self, off: usize, bytes: &[u8]) -> Result<()> {
            ensure!(
                bytes.len() <= self.len.saturating_sub(off),
                "vmm write {off}+{} > len {}",
                bytes.len(),
                self.len
            );
            // SAFETY: off+len ≤ len (checked, overflow-safe); ptr is host-
            // addressable; src (mmap) and dst (this slab) are distinct regions.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(off), bytes.len());
            }
            Ok(())
        }

        /// GPU-usable pointer (device-local under unified addressing).
        pub fn ptr(&self) -> *const u8 {
            self.ptr
        }
        pub fn len(&self) -> usize {
            self.len
        }
        pub fn is_empty(&self) -> bool {
            self.len == 0
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
            // Small tier; place two patterns, read both back, check alignment gap
            // doesn't corrupt either.
            let mut tier = DeviceTier::new(4 << 20).expect("alloc tier");
            let a: Vec<u8> = (0..1000u32).map(|i| (i & 0xff) as u8).collect();
            let b: Vec<u8> = (0..500u32).map(|i| ((i * 7) & 0xff) as u8).collect();
            let pa = tier.place(&a).expect("place a");
            let pb = tier.place(&b).expect("place b");
            assert_ne!(pa, pb);
            assert_eq!(tier.copy_out(pa, a.len()).unwrap(), a);
            assert_eq!(tier.copy_out(pb, b.len()).unwrap(), b);
            // 1000 bumps to 1024 (256-aligned) before b lands.
            assert_eq!(tier.used(), 1024 + 500);
        }

        #[test]
        fn tier_rejects_overflow() {
            let mut tier = DeviceTier::new(1 << 20).expect("alloc tier");
            assert!(tier.place(&vec![0u8; (1 << 20) + 1]).is_err());
        }
    }
}
