//! Minimal HIP surface: under `rocm` this binds the hipcc-built kernel launchers
//! (fp8/int8/f32 linalg, VQ-int3 and int4 MoE, MLA, fwd glue). Without the feature
//! the whole module compiles away.
//!
//! # The wall is four files, and this one is its shared quarter
//!
//! Split 2026-08-15 under the 800-line file ceiling (and again 2026-08-16, when M9's
//! launchers pushed the fused-block half past it), by cohesion rather than by line count:
//! this file keeps what every launcher needs — the descriptor structs, the `launchers!` DSL
//! that emits both halves of the ABI wall from one declaration, and the return-code check —
//! plus the handful of entry points that are not launchers at all (the device sync, the two
//! device-to-device copies, the fill, the scratch sizing). The macro invocations moved
//! out whole:
//!
//! - `hip_linalg.rs` — the PRIMITIVE wall: one launch per matvec, matmul, embedding row,
//!   elementwise op, activation quantizer, normalization or RoPE.
//! - `hip_blocks.rs` — the RESIDUAL-STREAM blocks: streaming MoE ranges and their
//!   accumulator, and the residual mixers (mHC, K3's attn_res).
//! - `hip_attn.rs` — the ATTENTION blocks: MLA/GQA/MHA attention, the KV slabs, the sparse
//!   indexer, the KV compressor, and the KDA recurrent-state family.
//!
//! Both are re-exported here, so `rivoli_backend::hip::launch_*` and `waist.rs`'s
//! `pub use crate::hip::*` resolve exactly as they did when this was one file — which module
//! a launcher is declared in is an authoring convenience and deliberately not part of any
//! path. `crates/cli/tests/kernel_coverage.rs` scans `crates/backend/src` recursively for
//! both declaration forms, so the census is unaffected by where they live.

#![cfg(feature = "rocm")]

use crate::abi::ScoreDims;
use anyhow::{Result, bail, ensure};
use std::ffi::c_void;

// Glob, not a curated list: a curated one is a third place a launcher has to be spelled, and
// the two that already exist (the declaration and `kernel_coverage.rs`'s census) are both
// checked. This one would not be — a launcher left out of it would simply be invisible to
// every consumer, which is the failure mode a wall like this cannot afford.
pub use crate::hip_attn::*;
pub use crate::hip_blocks::*;
pub use crate::hip_linalg::*;

/// One routed expert's six device pointers (per projection: a data ptr + a scale
/// ptr). ONE layout for both formats — byte-identical six-pointer `repr(C)` structs,
/// the kernel picks the interpretation: for int3-VQ (`launch_moe_expert_range`) the
/// pairs are packed 12-bit indices + bf16 group scales (`moe.hip ExpertDescVq`); for
/// int4 (`launch_moe_expert_range_i4`) they are packed 4-bit weights + f32 group
/// scales, one per `I4_GROUP` weights (`moe.hip ExpertDescI4`). Both formats are
/// group-scaled; only the scale WIDTH and the weight coding differ. The scale pointer
/// is typed `*const u16` here
/// (built from the VQ carrier) but its VALUE just addresses whatever the kernel reads.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDesc {
    pub gate_indices: *const u8,
    pub gate_scales: *const u16,
    pub up_indices: *const u8,
    pub up_scales: *const u16,
    pub down_indices: *const u8,
    pub down_scales: *const u16,
}

/// One DeepSeek-V4 routed expert's six device pointers — `moe.hip`'s `ExpertDescF4`.
///
/// Separate from [`ExpertDesc`] rather than a third interpretation of it. Dispatching a
/// `.f4` block through [`launch_moe_expert_range_i4`] would decode e2m1 nibbles as
/// `nibble − 8` at group 128 instead of group 32 — plausible magnitudes from the wrong
/// codebook, and there is no shape, size or scale check downstream that could find it. The
/// scales are `*const u8` because e8m0 IS one byte, a third width beside VQ's bf16 and
/// int4's f32, which is where [`ExpertDesc`]'s "one layout, kernel picks the
/// interpretation" stops being honest.
///
/// **The separation is a signpost, not a proof.** Every real dispatch reaches its
/// descriptor array through `buf.ptr() as *const _`, and that cast compiles either way —
/// only construction sites are type-checked. Making it a proof needs the keep-alive buffer
/// and the typed address to be ONE value (a `DescArray<T>` owning the `DeviceBuf` and
/// handing out only `*const T`), which reaches `gpu.rs`'s own
/// `self.descs_buf.ptr() as *const ExpertDesc` and so belongs to S3's wiring rather than
/// here.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDescF4 {
    /// `w1` — the GATE projection, `[inter, hidden]` as e2m1 nibble pairs.
    pub gate_packed: *const u8,
    /// `[inter, ceil(hidden / F4_GROUP)]` e8m0 bytes, row-major, one scale ROW per output
    /// row — not the 128x128 tile grid fp8 uses.
    pub gate_scale: *const u8,
    /// `w3` — the UP projection. Same shape as `w1`, which is exactly why a swap of the
    /// two is invisible to every structural check (`quant.rs::V4_PROJ`).
    pub up_packed: *const u8,
    /// `[inter, ceil(hidden / F4_GROUP)]` — same grid as `gate_scale`.
    pub up_scale: *const u8,
    /// `w2` — the DOWN projection, `[hidden, inter]`.
    pub down_packed: *const u8,
    /// `[hidden, ceil(inter / F4_GROUP)]` — the reduction dim is `inter` here, not `hidden`.
    pub down_scale: *const u8,
}

impl ExpertDescF4 {
    /// A descriptor that faults if it is ever read.
    ///
    /// Launch-order descriptor tables are sized `n_experts`, so most entries sit past any
    /// one token's selection and no launch names them. Filling those with nulls rather than
    /// with a copy of some resolved expert is the difference between a fault and a
    /// plausible wrong weight the day a range is computed wrongly. On the type because
    /// every arm with an F4 table wants the same defence — the V4 and K3 engines each
    /// carried a verbatim copy under its own name.
    pub fn null() -> Self {
        let n = std::ptr::null();
        Self {
            gate_packed: n,
            gate_scale: n,
            up_packed: n,
            up_scale: n,
            down_packed: n,
            down_scale: n,
        }
    }
}

/// The four device buffers the sparse indexer's scoring reads and writes — the pointer half
/// of [`launch_index_score_blocks`], whose `# Safety` section remains the single home for
/// their sizing and non-aliasing contract.
///
/// Grouped for [`ScoreDims`]'s reason and on stronger evidence: `q`, `kv` and `w` are three
/// `*const f32` in a row, so no type check can tell any of them from another, and a
/// transposed pair still addresses real f32 — finite, plausible, wrong. Named fields move
/// that mistake to the construction site, where it has a name.
///
/// NOT an ABI type, deliberately not `repr(C)`, and not in `abi.rs`: nothing here crosses
/// the wall. The launcher destructures it back into four positional arguments before the
/// call, so the C signature still reads 1:1 off the Rust one.
#[derive(Clone, Copy)]
pub struct ScoreBufs {
    pub q: *const f32,
    pub kv: *const f32,
    pub w: *const f32,
    pub score: *mut f32,
}

/// The `extern` type for one launcher argument: the `as`-cast target when the Rust side
/// narrows at the call (`o_dim: usize as i32`), and the Rust type unchanged when it does not
/// (`x: *const f32`). Exists only because `macro_rules!` cannot say "this one if present,
/// otherwise that one" inline — two arms can.
macro_rules! abi_ty {
    ($rt:ty) => {
        $rt
    };
    ($rt:ty, $ct:ty) => {
        $ct
    };
}

/// Declare a HIP entry point ONCE and get both halves of the ABI wall from it: the
/// `extern "C"` declaration and the `pub unsafe fn launch_*` wrapper that casts, calls and
/// maps the status through [`check`].
///
/// # Why this is not the macro the exemption below argued against
///
/// The note under the ignore-start marker rejected "a macro that declares each signature
/// once" — it sits with the invocation in `hip_linalg.rs` since the 2026-08-15 split, not
/// below this definition —
/// on two grounds, and it was right about one of them. It observed that ~25 launchers are
/// **different kernels that merely take the same shape** (`gemv_fp8`/`i8`/`i4`/`vq` all take
/// `x, packed, scale, o_dim, i_dim, y`) — "there is one copy of each already and nothing to
/// merge". That is correct and this macro does not merge them: each still has its own
/// declaration below, in full.
///
/// What it removes is the OTHER duplication, on an axis that note did not consider: every
/// kernel was written out **twice**, once as an `extern` decl in `i32` and once as a wrapper
/// in `usize` that restates the same parameters to cast them. Measured 2026-08-06 before this
/// change: 408 code lines of `extern` decls carrying **one** comment line between them, and
/// 795 code lines of wrappers of which 342 were the call re-listing its own parameters. The
/// two halves had to agree and nothing checked that they did.
///
/// **The other objection stands and is why the DSL is shaped like this.** "Breaking
/// goto-definition on every launcher" is a real cost in a repo whose orientation doc says
/// *"everything else: grep it"*. So the invocation spells `launch_gemv_vq` and
/// `rivoli_gemv_vq` **literally** rather than pasting them together from a stem — grep finds
/// both names at the declaration site exactly as before, and `tests/kernel_coverage.rs` can
/// still key its census on the text. Deriving them from one short ident would have saved
/// nothing (the names share a line either way) and cost every search in the file.
///
/// Doc comments and attributes ride at the invocation site: **the macro emits the
/// boilerplate, the declaration keeps the prose.** The `# Safety` blocks, the measurements
/// and the per-launcher notes are the reason this file is 752 comment lines, and none of
/// them are generated.
///
/// # Verified by expansion, not by ISA
///
/// G3 hashes the AMDGCN in `kernels/*.hip` and is structurally blind to this change — the
/// Rust wrapper is not in those objects. The gate that applies is
/// `RUSTC_BOOTSTRAP=1 cargo rustc --lib --features rocm -- -Zunpretty=expanded`, diffed
/// against the same output before the change. Seen red 2026-08-06 by transposing
/// `o_dim`/`i_dim` in `launch_gemv_vq`'s call — same types, compiles clean, and the
/// expansion diff caught it in 6 lines. That is the defect class a macro can introduce.
macro_rules! launchers {
    ($(
        $(#[$m:meta])*
        $rust:ident -> $sym:ident, $tag:literal (
            $($arg:ident : $rt:ty $(as $ct:ty)? ,)*
        );
    )*) => {
        // ONE block, not one per launcher, so the expansion stays comparable to the
        // hand-written original. The block-level `allow` replaces 17 per-item copies:
        // `too_many_arguments` on an ABI mirror is noise by construction.
        #[allow(clippy::too_many_arguments)]
        unsafe extern "C" {
            $( fn $sym($($arg: abi_ty!($rt $(, $ct)?)),*) -> i32; )*
        }

        $(
            $(#[$m])*
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn $rust($($arg: $rt),*) -> Result<()> {
                // SAFETY: caller's pointer contract.
                let r = unsafe { $sym($($arg $(as $ct)?),*) };
                ensure_hip_status(r, $tag)
            }
        )*
    };
}

// Both macros are `pub(crate)` items rather than `#[macro_export]`ed: the invocation files are
// siblings in this crate and nothing outside it may declare a launcher, so exporting them at
// the crate root would widen the ABI wall's authoring surface to every consumer of the waist.
pub(crate) use {abi_ty, launchers};

//
// EXEMPT FROM THE DUPLICATION GATE, and this is the only kind of thing that is.
//
// What follows is not code, it is an ABI: the argument lists of the C entry points in
// `kernels/*.hip`. Every item mirrors one of those entry points, and the mirroring IS the
// contract — identical signatures, not copy-paste. Until 2026-08-06 they also had to match
// the Vulkan launchers in `backend/vk.rs`, because `backend.rs` cfg-selected one glob of the
// SAME names; that second obligation is gone with the backend, but the first is unchanged
// and is the whole reason for the exemption.
//
// It cannot be deduplicated without making the code worse. The retired `vk.rs` stated the
// reason above its own `launch_gemv_fp8`, and it generalises: "the two must stay readable
// side by side for the bit-exactness comparison to be checkable by eye. Bundling them into a
// struct here and not there would put a translation step between the two signatures, which is the one
// place this port cannot afford one." A macro that declares each signature once would
// remove ~15 of these while breaking goto-definition on every launcher, and roughly 25 of
// the rest are DIFFERENT kernels that merely take the same shape (`gemv_fp8`/`i8`/`i4`/`vq`
// all take `x, packed, scale, o_dim, i_dim, y`) — there is one copy of each already and
// nothing to merge.
//
// The real instrument for "do the two backends agree" was behavioural, not syntactic:
// `tests/xbackend.rs` ran each arm under its own feature and compared raw output bytes. It
// was deleted 2026-08-06 with the second backend — a comparison needs two — and is preserved
// at `archive/vulkan-backend-hb16`. What it found is recorded in
// docs/investigations/vulkan-port.md §"1463 of 4096".
//
// The gate stays live over everything else in this file — it found four genuine
// duplicated-logic clones here on 2026-08-01 (`record_barrier`, `Stream::live`,
// `refill_from_mapping`, `dispatch_on`), all outside this block.
// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — the residual hand-written half of the ABI wall.
//
// ONE launcher does not fit `launchers!` above, and the macro refuses what it cannot prove
// mechanical rather than reshaping it.
//
//   `launch_index_score_blocks`  destructures `ScoreBufs` and `ScoreDims` into four pointers
//                              and four i32s before the call, with prose in place saying each
//                              struct exists to keep its own four in the right order. A
//                              positional 1:1 DSL cannot express one parameter becoming four
//                              arguments, and the alternative — four bare `usize` beside three
//                              indistinguishable `*const f32` — deletes what the structs are
//                              for.
//
// > **FIVE MORE WERE CONVERTED 2026-08-06, and how says more than what.** This list read six.
// > `launch_attend` was never a real exception: the extern scanner terminated on `-> i32;`, and
// > `rivoli_mla_attend_scratch_floats` returns `usize`, so it swallowed the following
// > declaration whole and `rivoli_mla_attend` looked absent. The other four — one
// > `i32::from(bool)` and three `&T` arguments passed to `*const T` by coercion — were declined
// > on the argument that converting them would move the text the expansion gate compares.
// >
// > That argument was backwards, and correcting it is the useful part. The gate is not the
// > thing being protected; the CODE is. So the gate was re-baselined to the intended text
// > FIRST, run to confirm it went red against the tree as it stood, and only then were the five
// > moved — turning it green again. A gate you may never re-baseline is not a safety net, it is
// > a freeze, and the way to keep it honest is to state the new expectation before writing the
// > code that meets it, not to leave code unconverted because the snapshot would move.
//
// The exemption itself is the same argument as the block above — these are same-shaped
// parameter lists for different kernels. Re-measured 2026-08-06 with the markers deleted: 6
// clones, all of them parameter lists, none logic.
unsafe extern "C" {
    fn rivoli_device_sync() -> i32;
    fn rivoli_memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> i32;
    fn rivoli_memcpy_dtod_async(
        dst: *mut u8,
        src: *const u8,
        bytes: usize,
        stream: *mut c_void,
    ) -> i32;
    fn rivoli_fill_u32(dst: *mut u8, pat: u32, bytes: usize) -> i32;
    fn rivoli_hash_rows(
        x: *const f32,
        n: i32,
        stride: i32,
        i_base: u64,
        out: *mut u64,
        stream: *mut c_void,
    ) -> i32;

    fn rivoli_mla_attend_scratch_floats(h: i32, kvl: i32) -> usize;

    // DSA lightning indexer (indexer.hip).

    fn rivoli_index_score_blocks(
        q: *const f32,
        kv: *const f32,
        w: *const f32,
        score: *mut f32,
        s: i32,
        n_comp: i32,
        heads: i32,
        hd: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// Launcher return-code check: 0 = ok, POSITIVE = arg guard, NEGATIVE = -(hipError_t).
// `pub(crate)` since the 2026-08-15 split: `launchers!` expands in `hip_linalg.rs` and
// `hip_blocks.rs`, and every wrapper it emits ends in this call.
pub(crate) fn ensure_hip_status(r: i32, name: &str) -> Result<()> {
    if r == 0 {
        Ok(())
    } else if r > 0 {
        bail!("{name}: argument guard rejected ({r})")
    } else {
        bail!("{name}: HIP error {}", -r)
    }
}

/// Block until all launched kernels retire — one join per token.
pub fn device_sync() -> Result<()> {
    // SAFETY: hipDeviceSynchronize, no pointers.
    ensure_hip_status(unsafe { rivoli_device_sync() }, "device_sync")
}

/// Synchronous device-to-device copy of `bytes` from `src` to `dst` — the routed
/// arena's slot relocation (compaction). BLOCKS, so the moved expert is in place before
/// any later kernel reads the new slot.
///
/// # Safety
/// `dst` and `src` must be valid, `bytes`-sized, NON-OVERLAPPING device regions (the
/// arena guarantees distinct slots).
pub unsafe fn memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
    ensure_hip_status(
        unsafe { rivoli_memcpy_dtod(dst, src, bytes) },
        "memcpy_dtod",
    )
}

/// STREAM-ORDERED device-to-device copy — for a pipeline whose kernels are on a stream.
///
/// A second entry point rather than a `stream` argument on [`memcpy_dtod`], because the two
/// are not one operation with a knob: that one **blocks the host**, and the arena relocation
/// it serves needs exactly that. `memory/routed.rs` compacts a slot while later reads may
/// still resolve to the old address, and this repo carries a measured defect from a read
/// outliving its layer and having its slot copied out from under it — 9 of 8452 reads at
/// `--max-mem 30`, non-deterministic output, and pins do not stop compaction. Replacing that
/// host barrier with a per-stream one would reopen it.
///
/// Added 2026-08-05 for `attn::v4::attention`, which interleaves six device-to-device copies
/// with sixteen launches. Those launches now take a stream, and a blocking `hipMemcpy`
/// between two of them does NOT wait on it: rivoli's streams are `hipStreamNonBlocking`, so
/// the null stream has no implicit ordering against them, and the copy would read `s.qr`
/// before `rmsnorm_batch` wrote it. See the banner above the V4 attention launchers.
///
/// # Safety
/// `dst` and `src` must be valid, `bytes`-sized, NON-OVERLAPPING device regions, and must
/// outlive `stream`'s completion — this returns as soon as the copy is *enqueued*, which is
/// the whole difference from [`memcpy_dtod`] and the whole hazard. `stream` is a live
/// `hipStream_t`, or null for the default stream.
pub unsafe fn memcpy_dtod_async(
    dst: *mut u8,
    src: *const u8,
    bytes: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    ensure_hip_status(
        unsafe { rivoli_memcpy_dtod_async(dst, src, bytes, stream) },
        "memcpy_dtod_async",
    )
}

/// Fill `bytes` at `dst` with the 32-bit pattern `pat` (`bytes` must be a multiple of 4).
///
/// Poisons a freshly admitted slot so a read-before-write is DETERMINISTIC rather than
/// dependent on what happened to be in memory. See kernels/vmm.hip::fill_u32.
///
/// # Safety
/// `dst` must be a device pointer owning at least `bytes`.
pub unsafe fn fill_u32(dst: *mut u8, pat: u32, bytes: usize) -> Result<()> {
    ensure_hip_status(unsafe { rivoli_fill_u32(dst, pat, bytes) }, "fill_u32")
}

//
// THE LAUNCHER WALL — same exemption, and for the same reason, as the `extern "C"` block
// above: from here to the end of the file every item is a signature mirroring a
// `kernels/*.hip` entry point, and the mirroring is the contract. (It mirrored the Vulkan
// launcher of the same name too, until that backend was retired 2026-08-06.) See the note
// above the extern block for the full argument, and for why a signature-declaring macro
// would not pay.
//
// The gate stays live over everything above — `check`, `device_sync` and the memcpy/fill
// helpers, which is where this file has logic rather than declarations.
//
// > **FALSE, and measured false 2026-08-15.** The marker opens ABOVE the `extern` block, so
// > every one of those helpers is inside the region, not above it — and the exemption is not
// > idle over them: deleting the marker fires a 5-line clone between `rivoli_memcpy_dtod_async`'s
// > `extern` declaration and `memcpy_dtod_async`'s wrapper, which is the one place in this
// > region where a wrapper still restates its own parameters. That is the ONE launcher
// > `launchers!` could not absorb plus its one uncollapsed pair, so the region is doing exactly
// > the job the sentence above says it is not. Left in place rather than deleted because a
// > reader who believes it will look for a gate that is not there; the honest boundary is
// > "everything in the two invocation files and this block is exempt, and `gpustream.rs` and
// > `waist.rs` are where this crate's logic is gated".

/// f32 count for the split-KV partial scratch — allocate once per session (never
/// per token). Mirrors the kernel's worst-case (MLA_MAX_SPLITS) sizing.
pub fn attend_scratch_floats(h: usize, kvl: usize) -> usize {
    // SAFETY: pure arithmetic, no pointers.
    unsafe { rivoli_mla_attend_scratch_floats(h as i32, kvl as i32) }
}

// ── DSA lightning indexer ───────────────────────────────────────────────────────

// ── DeepSeek-V4-Flash attention (S2b) ───────────────────────────────────────────────
//
// **All six take a `stream`, and none did before 2026-08-05.** They are the whole of
// `attn::v4::attention`, and a stream was declined here once on the grounds that "there is
// nothing to overlap with". That premise died when the `.f4` routed streaming pool landed
// and was measured at 1.082 ms/miss, 12.36 GB/s, `slot_stalls` 0 over the real 137.06 GiB
// expert set — the layer's routed fetch is now real work to hide compute behind.
//
// The set converts together or not at all, and that is correctness, not neatness. rivoli's
// streams are `hipStreamNonBlocking` (`async.hip`), so the null stream carries NO implicit
// ordering against them: one stream-capable launcher beside a null-stream neighbour over the
// same buffer is an unordered read, which is silently wrong, where leaving the whole block
// on the null stream is merely slow. A half-converted set is worse than either end.
//
// The seventh member of the set is not a launcher: [`memcpy_dtod`] is a BLOCKING
// `hipMemcpy` and `attn::v4::attention` interleaves six of them with these launches, so it
// needed [`memcpy_dtod_async`]. Counting six LAUNCHERS and paying six would have reproduced
// the very race the earlier decline predicted. `docs/investigations/v4-flash-port.md` requirement 9.
//
// A GATE NOW SAYS THESE ARE EXERCISED. None did until 2026-08-06: this block used to read
// "NO AUTOMATIC GATE SAYS THESE ARE EXERCISED, and that is now true of every launcher in
// this file" — `tests/kernel_coverage.rs::every_launcher_has_an_oracle` was keyed on
// `backend/vk.rs` and was deleted with that backend rather than re-keyed. It has been
// restored, keyed on `src/backend/` as a whole so the next file move does not repeat the
// deletion. On arrival it found 18 of this file's 48 launchers with no oracle at all.
//
// Read what it does and does not claim before trusting it: it counts a NAME in `tests/`,
// not an assertion, and it is not feature-gated, so it passes under a featureless build
// where none of the oracles it counts can even compile. Its original motivation is worth
// keeping in view: a tranche once shipped two of its hardest kernels unexercised while the
// suite's test COUNT rose, which is exactly what a census catches and a green suite does
// not.
//
// The substance behind the count, for these six: `tests/f4_attn.rs` scores four of them
// inside the whole attention block, and `tests/headtail.rs` scores `rmsnorm_batch` and
// `qk_norm` on their own. An earlier version of this line said `f4_attn.rs` covered all
// six and was wrong about two.

// --- head tail (S3) -------------------------------------------------------------------
// Appended as one block so the merge against the other V4 stages stays reviewable.

/// `Indexer.forward`'s scoring (model.py:425-427): `einsum("bshd,btd->bsht")`, `relu_()`,
/// the per-head `weights` multiply, and the sum over heads — into `[s, n_comp]`.
///
/// Writes the FULL pre-top-k score matrix, not a selection. That is deliberate and it is
/// what makes this scoreable: the shipped goldens' selected sets are invariant at
/// `index_topk = 512` (`docs/investigations/v4-flash-port.md`, "A hole S3 inherits"), so a
/// set comparison accepts an arbitrarily wrong ranking, while the score matrix cannot hide
/// one. The causal mask and the top-k are the caller's, exactly as `Oracle::indexer` splits
/// them.
///
/// Bit-exact against a faithful host reference by construction rather than by tolerance —
/// the kernel's note says why the reduction is not parallelised, and why it accumulates in
/// f32 and rounds once, which is what `torch.sum` over a bf16 tensor measurably does.
/// `Oracle::indexer` still folds per term; until that is fixed the two disagree, and the
/// disagreement is the oracle's.
///
/// # Safety
/// `q` is `s · heads · hd` f32; `kv` is `n_comp · hd` f32; `w` is `s · heads` f32; `score`
/// is `s · n_comp` writable f32. **None may alias another** — every kernel parameter is
/// `__restrict__`, so that covers the three inputs against each other and not only `score`
/// against them. All 4-byte aligned, device-resident, and outliving `stream`'s completion;
/// `stream` is a live `hipStream_t`, or null for the default stream.
pub unsafe fn launch_index_score_blocks(
    bufs: ScoreBufs,
    dims: ScoreDims,
    stream: *mut c_void,
) -> Result<()> {
    // Both groups are destructured here rather than field-accessed at the call, so the
    // argument list below stays readable 1:1 against the C signature — the wall's point.
    let ScoreBufs { q, kv, w, score } = bufs;
    // `dims`, not `d`: `d` means a head width everywhere else in this file, including in
    // `launch_act_quant_f4_rotated` directly above.
    //
    // Narrowed once, all four together, so the `as i32` soup is not interleaved with the
    // pointers at the call. `ScoreDims` is what keeps the four in the right order.
    let ScoreDims {
        s,
        n_comp,
        heads,
        hd,
    } = dims;
    let (s, n_comp) = (s as i32, n_comp as i32);
    let (heads, hd) = (heads as i32, hd as i32);
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_index_score_blocks(q, kv, w, score, s, n_comp, heads, hd, stream) };
    ensure_hip_status(r, "index_score_blocks")
}
// jscpd:ignore-end

// Deliberately BELOW the duplication-gate exemption that closes just above — the placement is
// the point. This first landed INSIDE that region, which would have put brand-new code under a
// build-error gate that had been told not to look at it. The region's argument is about the ABI
// wall's same-shaped parameter lists; this is an ordinary wrapper and has no business borrowing
// the exemption (review, 2026-08-17). An over-broad exemption is a hole in the gate.
//
// (The marker is named rather than spelled, because `crates/cli/tests/docs.rs` refuses a
// mid-sentence mention of it: jscpd treats one as a real marker, so a comment that quoted it
// would silently move where the exemption ends. That check caught this very comment.)

/// The sampled index space one hash fold walks — the scalar half of [`launch_hash_rows`],
/// grouped for [`ScoreBufs`]'s reason on the integer axis: `n` and `stride` are two bare
/// counts side by side, so no type check tells one from the other, and a transposed pair
/// still launches a fold that runs and reports — finite, plausible, wrong. Named fields
/// move that mistake to the construction site, where it has a name.
///
/// Like [`ScoreBufs`], NOT an ABI type and deliberately not `repr(C)`: nothing here
/// crosses the wall (see that struct's note for the destructure-before-the-call
/// convention this follows). Defined BELOW the exemption markers, beside the launcher it
/// belongs to, for the placement argument recorded above.
#[derive(Clone, Copy)]
pub struct HashSpan {
    /// Element count — the fold visits indices `0, stride, 2*stride, … < n`.
    pub n: usize,
    /// SAMPLES the buffer; pass 1 for every element. It exists so one probe arm can touch
    /// every CACHE LINE of a slot while reading a small fraction of its bytes; the mixed-in
    /// index is the element index, so folds at different strides stay sensitive to WHERE a
    /// difference is.
    pub stride: usize,
    /// Offsets the index space so several disjoint buffers can fold into ONE accumulator
    /// and still be sensitive to which buffer each element came from; pass 0 for a single
    /// buffer. The reference is `rivoli_core::hash::xor_fold_from`.
    pub i_base: u64,
}

/// XOR-fold the exact bits of `x[0..n]` into `*out` — the `--divergence-log` probe.
///
/// The device twin of `rivoli_core::hash::xor_fold`, which carries the argument for why the
/// fold is an XOR and adds no sync; `crates/engine/tests/fwd_kernel.rs::
/// hash_rows_matches_the_host_fold` scores the two against each other.
///
/// Hand-written here rather than a `launchers!` row in `hip_linalg.rs`, and the reason is the
/// 800-line soft cap: that file sits at 797 and the row would have carried it over. It also
/// belongs with `fill_u32` and `memcpy_dtod` on cohesion — the three are utilities over raw
/// device bytes rather than operators the model graph names.
///
/// The walk itself — count, stride, index offset — rides in [`HashSpan`], whose field docs
/// carry what `stride` and `i_base` mean.
///
/// `stream` is trailing and null is the null stream. On the fetch path it MUST be the fetch
/// stream: the folds bracket the bounce->slot copy, and rivoli's streams are
/// `hipStreamNonBlocking`, so a null-stream launch would race the copy it is meant to bracket.
///
/// # Safety
/// `x` must be `n` device f32; `out` one device u64, ZEROED before the first fold into it —
/// XOR against uninitialised memory is silently wrong, and `hipMalloc` does not zero. `stream`
/// must be a live `hipStream_t`, or null.
pub unsafe fn launch_hash_rows(
    x: *const f32,
    span: HashSpan,
    out: *mut u64,
    stream: *mut c_void,
) -> Result<()> {
    // Destructured here rather than field-accessed at the call — the same move
    // `launch_index_score_blocks` makes, and for its reason: the raw call below stays
    // readable 1:1 against the C signature, which is the wall's point.
    let HashSpan { n, stride, i_base } = span;
    // The ABI takes an i32. Every current call site is a few times 10^4 elements, so this is
    // defensive — but a silent wrap would fold a NEGATIVE length, which the kernel's `n <= 0`
    // guard turns into a no-op and the probe would report two runs agreeing about a quantity it
    // never hashed. That is the instrument's one unacceptable failure, so it is a check.
    ensure!(
        n <= i32::MAX as usize,
        "hash_rows: {n} elements exceeds the i32 ABI"
    );
    // SAFETY: caller's pointer contract.
    ensure_hip_status(
        unsafe { rivoli_hash_rows(x, n as i32, stride as i32, i_base, out, stream) },
        "hash_rows",
    )
}
