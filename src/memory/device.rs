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
//! `crate::memory::device::` reads identically either way; the Vulkan half lives in
//! `vktier` and differs in exactly one respect, which is the reason it is not a
//! mechanical transliteration: a host pointer and a device address are two unrelated
//! numbers there (docs/investigations/vulkan-port.md, "Host pointer != device address").

/// The bump cursor both backends' `DeviceTier::place` runs on — 256-byte aligned
/// offsets, `len` rounded up to `pad`, OOM refused rather than wrapped.
///
/// Shared because it is the one part of the tier that is NOT backend-specific: what
/// differs is how the bytes are written (a host memcpy under HIP, `Buf::write_at` under
/// Vulkan) and what address comes back, not where the placement lands. The two copies
/// this replaces had drifted in their OOM message and, more to the point, only one of
/// them was covered by a test for a ragged length.
///
/// `pad` is 1 under HIP (byte reads, no word hazard) and [`crate::backend::vk::WORD`] under
/// Vulkan, whose shaders read the slab a `uint` at a time. Returns the offset; the caller
/// advances nothing itself.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn bump(used: &mut usize, capacity: usize, len: usize, pad: usize) -> anyhow::Result<usize> {
    let off = used.next_multiple_of(256);
    let span = len.next_multiple_of(pad);
    anyhow::ensure!(
        off + span <= capacity,
        "device tier OOM: need {len} (padded to {span}) at offset {off}, capacity {capacity}"
    );
    *used = off + span;
    Ok(off)
}

/// The sizing gate both backends' `DeviceTier::new` runs: the tier must fit free device
/// memory with [`HEADROOM`] left over. Shared for the same reason as [`bump`] — it is
/// arithmetic about the device budget, not about how bytes get written — and because the
/// two copies it replaces had to keep one error message in step by hand.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn guard_capacity(capacity: usize, free: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        capacity + HEADROOM <= free,
        "device tier {capacity} + {HEADROOM} headroom > free {free}"
    );
    Ok(())
}

/// Leave this much device memory free beyond the tier (driver scratch, kernel dispatch
/// buffers, the cold-fetch slabs that arrive in M4).
#[cfg(any(feature = "rocm", feature = "vulkan"))]
const HEADROOM: usize = 4 << 30; // 4 GiB

/// `copy_out` for either backend's `DeviceBuf`: a fresh `Vec` filled by that type's own
/// `copy_out_into`. Backend-independent by construction — the transfer is `copy_out_into`'s
/// and the allocation is nobody's business but this function's.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn copy_out_owned(
    fill: impl FnOnce(&mut Vec<u8>) -> anyhow::Result<()>,
) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    fill(&mut out)?;
    Ok(out)
}

/// Emit the `#[cfg(test)]` bump-cursor accessor both backends' `DeviceTier` carry.
///
/// Shared for the same reason as [`bump`] and [`guard_capacity`]: the cursor is the part
/// of the tier that is NOT backend-specific, and the shared `tier_tests` at file scope
/// asserts on it for both. It has to be an accessor rather than a `pub(super)` field so
/// the field stays private in a real build, and it has to be a macro rather than a trait
/// because either backend's `impl` block is the only place that can name the field —
/// which is what left two literal copies once `cargo fmt` normalised them.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
macro_rules! tier_used_accessor {
    () => {
        #[cfg(test)]
        pub(super) fn used(&self) -> usize {
            self.used
        }
    };
}

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
    // in place — see docs/reference/architecture.md §1.
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

        /// Allocate the resident tier once. Fails loudly if a foreign tenant is
        /// present or the request doesn't fit free device memory with headroom.
        pub fn new(capacity: usize) -> Result<Self> {
            Self::guard_sole_tenant()?;
            let (free, _total) = mem_info()?;
            super::guard_capacity(capacity, free)?;
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
            // `pad` 1: HIP kernels read the slab bytewise, so there is no word-read
            // hazard to pad against (the Vulkan tier passes `WORD` for that reason).
            let off = super::bump(&mut self.used, self.capacity, bytes.len(), 1)?;
            // SAFETY: off+len ≤ capacity (checked by `bump`); within the slab. The source
            // is a live slice and the regions cannot overlap (one is the mmap'd artifact
            // or a fresh Vec, the other the device slab).
            let dst = unsafe { self.slab.ptr_mut().add(off) };
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
            Ok(dst)
        }

        /// Read the slab's first `len` bytes back to the host. Test-only, and it exists
        /// so ONE `tier_roundtrips_placed_bytes` covers both backends: the Vulkan tier
        /// hands out device addresses that cannot be dereferenced on the host, so a test
        /// reading through `place`'s return value could only ever have run here.
        #[cfg(test)]
        pub(super) fn read_prefix(&mut self, len: usize) -> Result<Vec<u8>> {
            // SAFETY: the slab is host-mapped and at least `len` bytes (callers pass
            // `used`), and no kernel runs during these tests.
            Ok(unsafe { std::slice::from_raw_parts(self.slab.host_mut(), len) }.to_vec())
        }

        tier_used_accessor!();
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
            super::copy_out_owned(|out| self.copy_out_into(out))
        }

        /// Copy the FIRST `len` bytes back into `out` (reused: cleared then
        /// resized). For partially-written buffers — e.g. the indexer's score
        /// slab, sized to max_ctx but holding only `nt` scores this step — so
        /// the D2H moves nt·4 bytes, not the whole slab.
        ///
        /// The bounds check is here and the transfer is [`DeviceBuf::copy_out_raw`]'s:
        /// three copies of the same `clear`/`reserve`/`hipMemcpy`/`set_len` sequence is
        /// three places for the `set_len` to stop matching the copy.
        pub fn copy_out_prefix(&self, out: &mut Vec<u8>, len: usize) -> Result<()> {
            ensure!(
                len <= self.len,
                "copy_out_prefix {len} > buf len {}",
                self.len
            );
            // SAFETY: `self.ptr` owns `self.len >= len` device bytes, and the caller's
            // sync obligation is the same one `copy_out_raw` documents.
            unsafe { Self::copy_out_raw(self.ptr, len, out) }
        }

        /// Copy the whole buffer back into `out` (a caller-owned buffer reused
        /// across tokens: cleared then refilled to `len`, so the per-token decode
        /// D2H allocates nothing once `out` has grown to size).
        pub fn copy_out_into(&self, out: &mut Vec<u8>) -> Result<()> {
            self.copy_out_prefix(out, self.len)
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
    /// tax — unlike a `hipHostMalloc` slot, which always maps system-domain.
    /// `ptr()` is GPU-usable.
    ///
    /// **The "reads ~9% slower" this claimed until 2026-08-01 has no LIVE source.**
    /// It cited `docs/hip-apu-memory.md`, deleted in the empty-slate rebuild; the probe
    /// behind it went in the same commit and is recoverable as
    /// `git show 3e1bd96:docs/probes/vmm_probe.cpp`. Treat the figure as the rationale of
    /// record rather than a standing number — a neighbouring read-tax claim (a ~40%
    /// MoE-dot tax from host-mapped VMM) was retracted outright in
    /// `docs/reference/architecture.md` §3. The allocator choice stands on the write
    /// side, which IS measured there: DMA into VMM pages runs 5.66 GB/s against
    /// 12.4 GB/s into the pinned arena.
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
        /// launch, plus the end-of-layer [`crate::backend::hip::device_sync`] fencing slot
        /// reuse. No CPU store fence is involved on this path.
        ///
        /// Verified CPU->GPU coherent on this APU (`git show 3e1bd96:docs/probes/vmm_probe.cpp` — the probe
        /// predates the empty-slate rebuild and is not in the tree; incl.
        /// `pread`). NOT a HIP contract for arbitrary hardware: a port off gfx1151,
        /// or a fill from a background HIP thread, must re-verify or insert an
        /// explicit fence.
        pub fn ptr_mut(&mut self) -> *mut u8 {
            self.ptr
        }

        /// The HOST base, for symmetry with the Vulkan `VmmBuf` — under HIP unified
        /// addressing it is the SAME NUMBER as [`VmmBuf::ptr_mut`], and the whole point of
        /// spelling it separately is that `pin.rs` cannot then rely on that coincidence.
        /// See docs/investigations/vulkan-port.md, "Host pointer != device address"; the ordering rules for
        /// filling through it are on `ptr_mut` above. Spelled as a call to it rather than
        /// as a second read of `self.ptr`, so the coincidence is stated once.
        pub fn host_mut(&mut self) -> *mut u8 {
            self.ptr_mut()
        }
    }

    impl Drop for VmmBuf {
        fn drop(&mut self) {
            // SAFETY: (ptr,handle,mapped) came from rivoli_vmm_alloc, freed once.
            unsafe { rivoli_vmm_free(self.ptr as *mut c_void, self.handle, self.mapped) };
        }
    }
}

#[cfg(feature = "vulkan")]
pub use vktier::{DeviceBuf, DeviceTier, VmmBuf, mem_info};

/// The same four names over Vulkan. Every allocation is a [`crate::backend::vk::Buf`] — one
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
    use crate::backend::vk::Buf;
    use anyhow::{Result, ensure};

    /// Free device memory and total, in bytes. Live free figure via
    /// VK_EXT_memory_budget — see [`crate::backend::vk::Gpu::mem_info`] for what happens when
    /// the extension is absent.
    pub fn mem_info() -> Result<(usize, usize)> {
        Ok(crate::backend::vk::gpu()?.mem_info())
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
            super::guard_capacity(capacity, free)?;
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
        /// The cursor advances by `bytes.len()` rounded up to [`crate::backend::vk::WORD`].
        ///
        /// BELT, NOT BRACES — and the earlier version of this comment overstated it.
        /// It claimed the padding is what stops one placement's 32-bit read reaching
        /// into the next placement's data. It is not: `off` is already rounded to 256,
        /// so `round_up_256(off + len) == round_up_256(off + span)` for every `len`
        /// (the gap is at most 3), and the returned addresses are byte-identical with
        /// and without this rounding. The end-of-slab case is likewise already covered
        /// by `Buf::new` allocating `capacity.next_multiple_of(WORD)`.
        ///
        /// So today this changes nothing observable, and it is kept for one reason: it
        /// makes the invariant hold on `span` rather than on the 256-byte alignment
        /// happening to be larger than a word. Anyone lowering that alignment — a
        /// plausible tightening, since 256 is generous for f32 weights — would
        /// otherwise turn a benign read into a live overrun into the next placement's
        /// bytes. Cheap insurance against a change that would look safe.
        ///
        /// Vulkan-only: HIP reads bytes directly and has no word-read hazard, so
        /// shifting the ROCm arena's offsets for this would be a real behaviour change
        /// for nothing.
        ///
        /// Errors if the tier is full — the pin is sized to fit, so OOM here is a
        /// budgeting bug, not a runtime condition.
        // NO `device_sync()` here, unlike `DeviceBuf::copy_in_at` which needs one. That
        // asymmetry is deliberate and narrow: placement happens once at startup, before
        // any dispatch is recorded, so there is no in-flight kernel for the host write to
        // race. If placement ever becomes something the engine does mid-decode — a
        // re-pin, a hot-swap — this needs the same sync `copy_in_at` got, for the same
        // reason.
        pub fn place(&mut self, bytes: &[u8]) -> Result<*mut u8> {
            let off = super::bump(
                &mut self.used,
                self.capacity,
                bytes.len(),
                crate::backend::vk::WORD,
            )?;
            self.slab.write_at(off, bytes)?;
            Ok((self.slab.ptr() as usize + off) as *mut u8)
        }

        /// Read the slab's first `len` bytes back to the host.
        ///
        /// Test-only, and the tests need it: what [`DeviceTier::place`] returns is a
        /// device address, so there is no other way to observe what a placement wrote.
        /// The HIP tier carries the same accessor so one test covers both.
        #[cfg(test)]
        pub(super) fn read_prefix(&mut self, len: usize) -> Result<Vec<u8>> {
            let mut out = Vec::new();
            self.slab.read_into(&mut out, len)?;
            Ok(out)
        }

        // `place_pads_the_cursor_to_a_word` in this module reads the field straight; the
        // shared `tier_tests` at file scope cannot, hence the accessor.
        tier_used_accessor!();
    }

    /// A standalone mutable device buffer — per-token activations, the descriptor
    /// array, the MoE accumulator. Sized once and rewritten each token via
    /// `copy_in_at`, exactly as under HIP: this type always spelled its transfers out,
    /// so it is the one that ports with no design change.
    ///
    /// The `copy_out*` family reads the mapping directly rather than issuing a
    /// `vkCmdCopyBuffer`, so unlike `hipMemcpy` it does NOT synchronise: call
    /// [`crate::backend::vk::device_sync`] first if a kernel wrote the bytes you are reading.
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
            // SYNC FIRST. The HIP twin is `hipMemcpy(..., H2D)`, which BLOCKS and so
            // orders itself after any kernel still reading this buffer. A bare host
            // write into a mapped allocation does not: a dispatch already recorded into
            // the open command buffer but not yet submitted would read the NEW bytes,
            // turning a write-after-read hazard into wrong data. `gpu.rs:843` (descs_vq
            // / wexpert_buf) depends on the blocking behaviour and has no device_sync
            // of its own.
            crate::backend::vk::device_sync()?;
            // `write_at` already bounds-checks, and its message names the offset,
            // length and capacity.
            self.buf.write_at(off, bytes)
        }

        /// Copy the whole buffer back to host as a fresh `Vec` — the ergonomic form the
        /// kernel oracle tests use. SYNCS FIRST, like every `copy_out*` here, because it
        /// goes through [`DeviceBuf::copy_out_into`].
        pub fn copy_out(&self) -> Result<Vec<u8>> {
            super::copy_out_owned(|out| self.copy_out_into(out))
        }

        /// Copy the FIRST `len` bytes back into `out` (reused: cleared then resized).
        /// For partially-written buffers — e.g. the indexer's score slab, sized to
        /// max_ctx but holding only `nt` scores this step.
        pub fn copy_out_prefix(&self, out: &mut Vec<u8>, len: usize) -> Result<()> {
            crate::backend::vk::device_sync()?;
            self.buf.read_into(out, len)
        }

        /// Copy the whole buffer back into `out` (a caller-owned buffer reused across
        /// tokens, so the per-token readback allocates nothing once it has grown).
        /// SYNCS FIRST via [`DeviceBuf::copy_out_prefix`], for the same reason as
        /// [`DeviceBuf::copy_in_at`] — the HIP twin is a blocking `hipMemcpy(..., D2H)`
        /// and callers were written against that.
        ///
        /// Without it, `gpu.rs:789` (`launch_gemv_f32` then immediately
        /// `gate_logits.copy_out_into`) would read the PREVIOUS token's gate logits,
        /// because the dispatch is still sitting unsubmitted in the open command
        /// buffer. Routing would then pick the wrong experts every layer of every
        /// token, coherently, with no error. `gpu.rs:972` (`launch_argmax` then
        /// `argmax_dev.copy_out_into`) is the same shape and yields the wrong token.
        pub fn copy_out_into(&self, out: &mut Vec<u8>) -> Result<()> {
            self.copy_out_prefix(out, self.len)
        }

        /// DIAGNOSTIC: read `len` bytes back from an arbitrary device pointer.
        ///
        /// This was deliberately ABSENT, on the grounds that its HIP twin only works
        /// because a device pointer is also a host pointer, and that recovering the mapping
        /// from a bare address would exist purely to serve a diagnostic. It is here now for
        /// one reason the earlier note did not weigh: without it `--features vulkan,trace`
        /// does not COMPILE, and a feature combination that cannot be built is worse than a
        /// registry lookup nobody times. `crate::backend::vk::read_raw` carries the argument in full.
        ///
        /// **UN-GATED 2026-08-02. This said "`trace`-only, so it is never on the decode
        /// path", and that stopped being true.** `GpuEngine::prefill_layer_major` reads the
        /// prompt's last hidden row back through the host to normalise `x` row 0, on the
        /// ordinary decode path, on both backends — so a `trace` gate here means
        /// `--features vulkan` alone does not compile. It was caught by
        /// `tests/feature-matrix.sh` on that combination and by nothing else: the
        /// prescribed union names `rocm` and `trace` together, so both halves of the gate
        /// were satisfied and the hole only opened for a backend nobody had built.
        ///
        /// The HIP twin was never gated, which is why this was a one-backend break.
        ///
        /// # Safety
        /// `src` must be a device address inside a live buffer, readable for `len` bytes,
        /// and no kernel may be concurrently writing them (call after a `device_sync`).
        pub unsafe fn copy_out_raw(src: *const u8, len: usize, out: &mut Vec<u8>) -> Result<()> {
            // Reads the host mapping, which is not ordered against a recorded dispatch —
            // hence the sync, matching what the blocking `hipMemcpy` gives the HIP twin for
            // free. Every other `copy_out*` here does the same for the same reason.
            crate::backend::vk::device_sync()?;
            // SAFETY: the caller's contract, forwarded.
            unsafe { crate::backend::vk::read_raw(src, len, out) }
        }

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
    /// six device pointers (`ArenaPool::ptr`, `src/memory/pin.rs`), those are two numbers here
    /// and this type must hand them out separately — see docs/investigations/vulkan-port.md, "Host pointer
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
            // enforced by the arena; see `crate::artifact::format` and `slot_span` in pin.rs.
            let base = buf.host_mut() as usize;
            ensure!(
                base.is_multiple_of(crate::backend::vk::O_DIRECT_ALIGN),
                "mapped base {base:#x} is not {}-byte aligned, so io_uring O_DIRECT \
                 reads into the routed pool would fail with EINVAL",
                crate::backend::vk::O_DIRECT_ALIGN
            );
            Ok(Self { buf })
        }

        /// The DEVICE base. Descriptor pointers are computed from this; it is not
        /// host-dereferenceable.
        pub fn ptr(&self) -> *const u8 {
            self.buf.ptr()
        }

        /// The DEVICE base as `*mut`, matching the HIP `VmmBuf`'s spelling so `pin.rs`
        /// takes both bases the same way on both backends. Still not host-dereferenceable —
        /// the mutability is about what the KERNELS do to these bytes, not the CPU.
        pub fn ptr_mut(&mut self) -> *mut u8 {
            self.buf.ptr_mut()
        }

        /// The HOST base — the io_uring O_DIRECT DMA target, and the only one of the
        /// two that may be dereferenced on the CPU.
        pub fn host_mut(&mut self) -> *mut u8 {
            self.buf.host_mut()
        }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used, clippy::expect_used)]
    mod tests {
        use super::*;

        /// `place` advances the cursor by a WORD-rounded span. Neither length in
        /// `tier_roundtrips_placed_bytes` is ragged (1000 and 500 are both multiples of
        /// 4), so `used` is identical with and without the padding and that test could
        /// not tell the two apart. 1001 can.
        #[test]
        fn place_pads_the_cursor_to_a_word() {
            let mut tier = DeviceTier::new(4 << 20).expect("alloc tier");
            tier.place(&vec![7u8; 1001]).expect("place ragged");
            assert_eq!(
                tier.used, 1004,
                "cursor must advance by the WORD-rounded span, so a shader's 32-bit \
                 read of the final byte cannot reach the next placement"
            );
        }

        /// The device base and the host base are DIFFERENT NUMBERS. This is the
        /// central structural claim of the Vulkan port (docs/investigations/vulkan-port.md, "Host pointer
        /// != device address"), and the one a maintainer would erase by "simplifying"
        /// two accessors back into one — a regression that reads as garbage weights
        /// rather than a crash, because both values are plausible pointers.
        #[test]
        fn vmmbuf_device_and_host_bases_differ() {
            let mut b = VmmBuf::new(1 << 20).expect("alloc");
            let dev = b.ptr() as usize;
            let host = b.host_mut() as usize;
            assert_ne!(
                dev, host,
                "device base {dev:#x} == host base {host:#x}: either the accessors have \
                 been collapsed, or this driver maps them identically — if the latter, \
                 say so here rather than deleting the test, because the code must not \
                 start depending on it"
            );
        }
    }
}

/// The tier/buffer tests that are the SAME question on both backends, written once
/// against the four re-exported names.
///
/// They used to be two near-identical `mod tests` blocks, one per backend, which is how
/// the ROCm side ended up with no `DeviceBuf` coverage at all and the Vulkan side with
/// the only ragged-length cursor check. What genuinely differs stays inside each backend
/// module: the WORD padding (`place_pads_the_cursor_to_a_word`, Vulkan-only, since HIP
/// pads by 1) and the device/host base distinction (`vmmbuf_device_and_host_bases_differ`,
/// where HIP's two bases are deliberately the same number).
///
/// Needs a real device, so it runs only under a backend feature — a featureless
/// `cargo test` skips the module rather than failing to link.
#[cfg(all(test, any(feature = "rocm", feature = "vulkan")))]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tier_tests {
    use super::{DeviceBuf, DeviceTier, VmmBuf};

    /// Two placements land at 256-aligned offsets, keep their bytes, and the returned
    /// addresses track the cursor. Read back through `read_prefix` rather than through
    /// the returned pointer: under Vulkan that pointer is a device address and
    /// dereferencing it on the host is a segfault at best.
    #[test]
    fn tier_roundtrips_placed_bytes() {
        let mut tier = DeviceTier::new(4 << 20).expect("alloc tier");
        let a: Vec<u8> = (0..1000u32).map(|i| (i & 0xff) as u8).collect();
        let b: Vec<u8> = (0..500u32).map(|i| ((i * 7) & 0xff) as u8).collect();
        let pa = tier.place(&a).expect("place a");
        let pb = tier.place(&b).expect("place b");
        assert_ne!(pa, pb);
        // 1000 bumps to 1024 (256-aligned) before b lands.
        assert_eq!(tier.used(), 1024 + 500);
        assert_eq!(pb as usize - pa as usize, 1024);
        let used = tier.used();
        let got = tier.read_prefix(used).expect("read back");
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
        // length is the proof it did not fall back to the whole buffer. This also
        // exercises the shared D2H path: `copy_out_into` -> `copy_out_prefix` ->
        // `copy_out_raw` under HIP, so a broken `set_len` shows up here.
        let mut pre = vec![0xAAu8; 999];
        d.copy_out_prefix(&mut pre, 16).expect("prefix");
        assert_eq!(pre, &out[..16]);
        assert!(d.copy_out_prefix(&mut pre, 65).is_err());
        assert!(d.copy_in_at(48, &bytes).is_err());
    }

    #[test]
    fn vmmbuf_allocates_distinct_device_bases() {
        // `ptr_mut` on both backends (HIP has no `ptr`), hence the `mut` bindings — the
        // mutability is about what the kernels do to these bytes, not the CPU.
        let mut a = VmmBuf::new(4096).expect("alloc a");
        let mut b = VmmBuf::new(4096).expect("alloc b");
        assert!(!a.ptr_mut().is_null());
        assert_ne!(a.ptr_mut(), b.ptr_mut());
    }
}
