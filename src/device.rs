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
        pub fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    }
    pub const HIP_MEMCPY_H2D: i32 = 1;
    pub const HIP_MEMCPY_D2H: i32 = 2;
    pub const HIP_SUCCESS: i32 = 0;
}

#[cfg(feature = "rocm")]
pub use tier::DeviceTier;

#[cfg(feature = "rocm")]
mod tier {
    use super::ffi::*;
    use anyhow::{Result, bail, ensure};
    use std::ffi::c_void;

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
            let gtt: u64 = match std::fs::read_to_string(path) {
                Ok(s) => s.trim().parse().unwrap_or(0),
                // No sysfs (non-amdgpu or containerized) — can't verify tenancy;
                // proceed rather than block, the allocation itself will fail loud.
                Err(_) => return Ok(()),
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
        pub fn capacity(&self) -> usize {
            self.capacity
        }
    }

    impl Drop for DeviceTier {
        fn drop(&mut self) {
            // SAFETY: base came from hipMalloc and is freed exactly once.
            unsafe { hipFree(self.base as *mut c_void) };
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
