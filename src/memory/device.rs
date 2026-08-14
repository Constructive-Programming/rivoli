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
//! allocate on in a CPU-only dev build. `rocm` supplies the four names (`mem_info`,
//! `DeviceTier`, `DeviceBuf`, `VmmBuf`) that everything above `crate::memory::device::`
//! reads. A parallel `vktier` module supplied the same four over Vulkan until 2026-08-06
//! (tag `archive/vulkan-backend-hb16`); it differed in exactly one respect, and that
//! difference is why `VmmBuf` still hands out a device base and a host base separately —
//! see [`VmmBuf::host_mut`].

/// The bump cursor both backends' `DeviceTier::place` runs on — 256-byte aligned
/// offsets, `len` rounded up to `pad`, OOM refused rather than wrapped.
///
/// Factored out when there were two tiers, because placement is the one part that was NOT
/// backend-specific: what differed was how the bytes are written (a host memcpy under HIP,
/// `Buf::write_at` under Vulkan) and what address comes back, not where the placement
/// lands. The two copies this replaced had drifted in their OOM message and, more to the
/// point, only one of
/// them was covered by a test for a ragged length.
///
/// Returns the offset; the caller advances nothing itself.
///
/// It took a `pad` argument until 2026-08-06 — 1 under HIP (byte reads, no word hazard) and
/// `WORD` under Vulkan, whose shaders read the slab a `uint` at a time. With one backend the
/// only caller passed the constant 1, so `len.next_multiple_of(pad)` was the identity; the
/// parameter is gone rather than left as a knob nothing turns. A backend that reads the slab
/// wider than a byte needs it back.
#[cfg(feature = "rocm")]
fn bump(used: &mut usize, capacity: usize, len: usize) -> anyhow::Result<usize> {
    let off = used.next_multiple_of(256);
    anyhow::ensure!(
        off + len <= capacity,
        "device tier OOM: need {len} at offset {off}, capacity {capacity}"
    );
    *used = off + len;
    Ok(off)
}

/// The sizing gate both backends' `DeviceTier::new` runs: the tier must fit free device
/// memory with [`HEADROOM`] left over. Shared for the same reason as [`bump`] — it is
/// arithmetic about the device budget, not about how bytes get written — and because the
/// two copies it replaces had to keep one error message in step by hand.
#[cfg(feature = "rocm")]
fn guard_capacity(capacity: usize, free: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        capacity + HEADROOM <= free,
        "device tier {capacity} + {HEADROOM} headroom > free {free}"
    );
    Ok(())
}

/// Leave this much device memory free beyond the tier (driver scratch, kernel dispatch
/// buffers, the cold-fetch slabs that arrive in M4).
#[cfg(feature = "rocm")]
const HEADROOM: usize = 4 << 30; // 4 GiB

/// `copy_out` for either backend's `DeviceBuf`: a fresh `Vec` filled by that type's own
/// `copy_out_into`. Backend-independent by construction — the transfer is `copy_out_into`'s
/// and the allocation is nobody's business but this function's.
#[cfg(feature = "rocm")]
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
#[cfg(feature = "rocm")]
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
            // No padding: HIP kernels read the slab bytewise, so there is no word-read
            // hazard to pad against. See `bump` for what the removed `pad` argument was.
            let off = super::bump(&mut self.used, self.capacity, bytes.len())?;
            // SAFETY: off+len ≤ capacity (checked by `bump`); within the slab. The source
            // is a live slice and the regions cannot overlap (one is the mmap'd artifact
            // or a fresh Vec, the other the device slab).
            let dst = unsafe { self.slab.ptr_mut().add(off) };
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
            Ok(dst)
        }

        /// Overwrite a placed region with `bytes` — a REFILL, for a caller that recycles one
        /// placement across many tensors.
        ///
        /// **It exists so the unified-addressing assumption stays in one file.** [`Self::place`]'s
        /// doc says the device pointer and the host address coincide under HIP, that the
        /// coincidence is not portable, and that callers must not depend on it — so `GlimmerPin`'s
        /// streaming slots (which memcpy a layer's weights over the previous layer's, once per
        /// visit) call this rather than dereferencing the pointer `place` handed back.
        ///
        /// > **The offset parameter went 2026-08-12, by review, and so did a contract nothing
        /// > could satisfy.** This was `write_at(base, off, bytes)` and required `off +
        /// > bytes.len()` to lie inside "that same placement's reservation" — but its only caller
        /// > passed the FIRST tensor's address as `base` and offsets reaching eleven placements
        /// > past it, so every call but one violated the stated contract while being perfectly
        /// > sound (the provenance is the whole slab). A contract no caller satisfies cannot be
        /// > used to audit a caller, which on an `unsafe` boundary is the whole point of writing
        /// > one. Taking the destination directly makes the contract true and the arithmetic
        /// > disappear.
        ///
        /// # Safety
        /// `dst` must be a pointer this tier returned from [`Self::place`], and `bytes.len()`
        /// must not exceed that placement's extent. The tier does not retain per-placement
        /// extents, so the caller supplies it (see `Slot::fill`, which checks the incoming
        /// tensor against the length its own placement was made with).
        pub unsafe fn write_to(dst: *mut u8, bytes: &[u8]) {
            // SAFETY: the caller's contract above; source and destination cannot overlap (one
            // is the mmap'd artifact, the other the device slab).
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
            }
        }

        /// Read the slab's first `len` bytes back to the host. Test-only, and it exists
        /// so ONE `tier_roundtrips_placed_bytes` covered both backends: the Vulkan tier
        /// handed out device addresses that cannot be dereferenced on the host, so a test
        /// reading through `place`'s return value could only ever have run here. Kept after
        /// that backend's retirement because the test reads better through it than through
        /// a raw pointer, not because anything still forces the indirection.
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

        /// The HOST base. Under HIP unified addressing it is the SAME NUMBER as
        /// [`VmmBuf::ptr_mut`], and the whole point of spelling it separately is that
        /// `pin.rs` cannot then rely on that coincidence — descriptors take the device
        /// base, the io_uring DMA target takes the host base, and each says which it means.
        ///
        /// It existed for symmetry with a Vulkan `VmmBuf` where the two were unrelated
        /// numbers. That backend was retired 2026-08-06 and the distinction is now
        /// unenforced by the type system, so KEEPING the two spellings is what stops the
        /// coincidence from being silently baked into call sites. The ordering rules for
        /// filling through it are on `ptr_mut` above. Spelled as a call to it rather than
        /// as a second read of `self.ptr`, so the coincidence is stated once.
        pub fn host_mut(&mut self) -> *mut u8 {
            self.ptr_mut()
        }
    }

    impl Drop for VmmBuf {
        /// **Joins the device before unmapping.** `rivoli_vmm_free` is `hipMemUnmap` +
        /// `hipMemRelease` + `hipMemAddressFree`, none of which synchronise — unlike `hipFree`,
        /// which joins implicitly. Tearing a mapping out from under running kernels is a real
        /// hazard, and `glimmer_gpu.rs`'s `decode` already carried the argument but fixed only
        /// its ERROR path, on the reasoning that success is "already joined by `sample`". The
        /// join belongs where the unmap is, so no caller has to re-derive it.
        ///
        /// > **It was written while chasing the §4b SIGSEGV and is NOT what fixed that** —
        /// > measured, 2 crashes in 27 runs before and 1 in 25 after, indistinguishable. That
        /// > turned out to be an upstream bug, gone in ROCm 7.14. This survives because the
        /// > hazard it closes is real on its own and was documented independently by
        /// > `glimmer_gpu.rs`'s `decode`, which fixed only its error path on the reasoning that
        /// > success is "already joined by `sample`" — two of the captured cores were
        /// > success-path drops, so that reasoning was too narrow.
        ///
        /// A `device_sync` per teardown is free in the only sense that matters: buffers of this
        /// kind are built once per engine, not per token.
        fn drop(&mut self) {
            // Deliberately unreported: a `Drop` cannot return an error, and a device already in
            // a bad state is exactly when the unmap below must still run.
            let _ = crate::backend::hip::device_sync();
            // SAFETY: (ptr,handle,mapped) came from rivoli_vmm_alloc, freed once.
            unsafe { rivoli_vmm_free(self.ptr as *mut c_void, self.handle, self.mapped) };
        }
    }
}

/// The tier/buffer tests that are the SAME question on both backends, written once
/// against the four re-exported names.
///
/// They used to be two near-identical `mod tests` blocks, one per backend, which is how
/// the ROCm side ended up with no `DeviceBuf` coverage at all and the Vulkan side with the
/// only ragged-length cursor check. Unifying them is what kept that coverage when the
/// Vulkan module was deleted on 2026-08-06 — had they still been split, this half would
/// have gone with it. What genuinely differed stayed inside each backend module, so the
/// WORD-padding check (`place_pads_the_cursor_to_a_word`, Vulkan-only, since HIP pads by 1)
/// went, and so did `vmmbuf_device_and_host_bases_differ`, which lived in `vktier`'s own
/// `mod tests`. That left `VmmBuf::host_mut` with NO coverage at all while `routed.rs`
/// depends on it, so `vmmbuf_host_and_device_bases_coincide_under_hip` below replaces it —
/// asserting the opposite fact, which is the one that is now true and load-bearing.
///
/// Needs a real device, so it runs only under a backend feature — a featureless
/// `cargo test` skips the module rather than failing to link.
#[cfg(all(test, feature = "rocm"))]
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

    /// `host_mut()` and `ptr_mut()` are the SAME NUMBER under HIP unified addressing.
    ///
    /// This is the inverse of the Vulkan-side `vmmbuf_device_and_host_bases_differ`, deleted
    /// with that backend on 2026-08-06, and it exists because deleting it left `host_mut`
    /// with no caller in any test while `routed.rs` resolves both bases through it.
    ///
    /// Asserting the coincidence is not asserting that call sites may RELY on it — see
    /// `VmmBuf::host_mut`, which keeps the two spellings precisely so nobody bakes it in.
    /// What this pins is that the HIP half still behaves as that note claims.
    #[test]
    fn vmmbuf_host_and_device_bases_coincide_under_hip() {
        let mut b = VmmBuf::new(4096).expect("alloc");
        assert!(!b.host_mut().is_null());
        assert_eq!(
            b.host_mut(),
            b.ptr_mut(),
            "HIP unified addressing: one number is both the host and the device base"
        );
    }
}
