//! Minimal HIP surface: under `rocm` this binds the hipcc-built kernel launchers
//! (fp8/int8/f32 linalg, VQ-int3 and int4 MoE, MLA, fwd glue). Without the feature
//! the whole module compiles away.

#![cfg(feature = "rocm")]

use crate::v4compress::{Finish, GeomAbi, ScoreDims};
use anyhow::{Result, bail};
use std::ffi::c_void;

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
/// The note under `jscpd:ignore-start` rejected "a macro that declares each signature once"
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
                ensure_hip_decode_status(r, $tag)
            }
        )*
    };
}

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — the declarations below, and nothing else in this file.
//
// This is the surviving half of an argument made before Track 2 and re-tested after it. The
// original note exempted two regions, the `extern` block and the wrappers, on two grounds. One
// of them is now obsolete: "every kernel is written out twice, once as a decl and once as a
// wrapper" was true and is what `launchers!` above removed — 1307 code lines to 685, with the
// comment count going UP.
//
// The other ground is unchanged and is why this marker is still here, though the number that
// came with it was wrong and is corrected below.
//
// > **MEASURED 2026-08-06, correcting the inherited claim.** The original note said "roughly 25
// > are DIFFERENT kernels that merely take the same shape — `gemv_fp8`/`i8`/`i4`/`vq` all take
// > `x, packed, scale, o_dim, i_dim, y`". Counted rather than asserted: the 41 declarations
// > below have **41 distinct parameter lists — zero exact duplicates**, and exactly one PAIR
// > shares even a type sequence under different names (`act_quant_f8` and `v4_indexer_spread`,
// > both `*mut f32, i32, i32, *mut c_void`). The `gemv` family does not actually agree:
// > `gemv_vq` takes seven parameters (`indices`, `scales`, `codebook`), `gemv_i4` takes six.
//
// What jscpd matches is a shared PREFIX, not a shared list. `moe_expert_range` and
// `moe_expert_range_i4` agree on `x, hidden, inter, e_start, e_count, descs` and then diverge.
// Re-measured with the markers deleted: **11 clones, every one a prefix run between two
// adjacent declarations**, none of them logic.
//
// **A named-prefix DSL would work and is still refused.** `@moe_head, gate_cb: …` is a
// tt-muncher away and would save perhaps 40-60 lines across the ~10 launchers sharing a prefix.
// It costs the one property this wall exists for — you could no longer read the C signature off
// the Rust declaration, and "the mirroring IS the contract" is the whole argument for the file.
// The clearest candidate is also the worst one: `moe_expert_range` takes `*const ExpertDesc` and
// `_f4` takes `*const ExpertDescF4` plus an extra `n_desc`, and [`ExpertDescF4`]'s own note says
// dispatching an `.f4` block through the i4 path decodes e2m1 nibbles at the wrong group size
// with **no downstream check that could find it**. Factoring their common prefix behind one name
// moves those two toward each other, which is the opposite of what that separation is for.
//
// **The region is narrower than what it replaced, and deliberately so.** The old markers
// bracketed lines 67-524 and 602-2061 — nearly the whole file. Track 2 moved the code out from
// under them and they stayed where they were, silently exempting whatever had drifted into
// those line ranges. That is the "stale exemption is a hole in the gate" failure happening
// inside the change that was supposed to prevent it. This one is anchored to the macro
// invocation, so anything added outside it is gated.
launchers! {
    /// VQ-int3 GEMV `y = W·x` (group scales applied inside the decode).
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`].
    launch_gemv_vq -> rivoli_gemv_vq, "gemv_vq" (
        x: *const f32,
        indices: *const u8,
        scales: *const u16,
        codebook: *const u16,
        o_dim: usize as i32,
        i_dim: usize as i32,
        y: *mut f32,
    );

    /// Streaming MoE: gate/up + down for the absolute expert range `[e_start,
    /// e_start+e_count)` on `stream`, atomically accumulating into the shared fixed-point
    /// `acc` row. Drain it with [`launch_moe_acc_drain`] once every range has landed.
    ///
    /// `acc` is `hidden` u64 per token row — ONE row per row, not `e·hidden`. Ranges on
    /// DIFFERENT streams may accumulate concurrently and the result is unchanged, because
    /// integer addition is associative; that is the whole reason this is not an f32 slab plus
    /// a reduce.
    ///
    /// `nrow` token rows (1 or 2) share ONE read of the expert weights — the batched verify
    /// pass a speculative decode needs. Every buffer puts the token row FASTEST:
    /// `x[t·hidden + i]`, `h[(e·nrow + t)·inter + j]`, `wexpert[e·nrow + t]`,
    /// `acc[t·hidden + o]`. `wexpert[e·nrow + t] == 0` means row `t` did not route to expert
    /// `e`, which is how the caller passes the UNION of two tokens' picks with no mask.
    ///
    /// At `nrow == 1` every one of those indices collapses to the single-row form, so the
    /// shipping decode path's layout and arithmetic are unchanged.
    ///
    /// # Safety
    /// Every device pointer (`descs`/codebooks/`wexpert`/`x`/`h`/`acc`) must outlive
    /// `stream`'s completion — await its [`Signal`](crate::backend::gpustream::Signal), and each must
    /// own `nrow` rows in the layout above.
    launch_moe_expert_range -> rivoli_moe_expert_range, "moe_expert_range" (
        x: *const f32,
        hidden: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        descs: *const ExpertDesc,
        gate_cb: *const u16,
        up_cb: *const u16,
        down_cb: *const u16,
        wexpert: *const f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: usize as i32,
        stream: *mut c_void,
    );

    /// int4 counterpart of [`launch_moe_expert_range`]: gate/up + down for the absolute
    /// range `[e_start, e_start+e_count)` on `stream`, decoding int4 (f32 group scales).
    /// `descs` are [`ExpertDesc`]; contributions land in the same fixed-point `acc` row,
    /// so int4 and VQ experts of one layer mix freely within a batch.
    ///
    /// # Safety
    /// Every device pointer (`descs`/packed weights/`wexpert`/`x`/`h`/`acc`) must
    /// outlive `stream`'s completion — await its [`Signal`](crate::backend::gpustream::Signal).
    launch_moe_expert_range_i4 -> rivoli_moe_expert_range_i4, "moe_expert_range_i4" (
        x: *const f32,
        hidden: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        descs: *const ExpertDesc,
        wexpert: *const f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: usize as i32,
        stream: *mut c_void,
    );

    /// DeepSeek-V4 counterpart of [`launch_moe_expert_range_i4`]: FP4 experts (e2m1 nibbles,
    /// one e8m0 scale per 32 weights along the reduction dim) for the absolute range
    /// `[e_start, e_start+e_count)` on `stream`. Contributions land in the same fixed-point
    /// `acc` row, so this shares [`launch_moe_acc_drain`] with the other two formats.
    ///
    /// **`x` must already be fp8-quantized** by [`launch_act_quant_f8`] — V4 quantizes the
    /// activation in front of every quantized `Linear` and this path cannot do it per output
    /// row (see `linalg.hip::act_quant_f8`). The `h` re-quantization between the two passes IS
    /// done here, because forgetting it is silent.
    ///
    /// `n_desc` is the length of the `descs` array. It exists because `.f4` holds
    /// `n_experts` blocks and **no shared block** — V4's shared expert is fp8 e4m3 at 128x128
    /// and stays resident — unlike `.vq3`/`.i4`, which hold `n_experts + 1` with the shared
    /// expert last. An index one past the end there reads the wrong expert; here it reads
    /// something that is not e2m1 nibbles at all, i.e. the wrong ARITHMETIC.
    ///
    /// `swiglu_limit` comes from the config (`10.0`); the launcher refuses every value that
    /// would disable the clamp — `<= 0`, NaN and `+/-inf` — because an
    /// unclamped SwiGLU on this path is a known silent defect, not a configuration.
    ///
    /// # Safety
    /// Every device pointer (`descs`/packed weights/`wexpert`/`x`/`h`/`acc`) must outlive
    /// `stream`'s completion — await its [`Signal`](crate::backend::gpustream::Signal).
    ///
    /// **`wexpert` and `h` are indexed by ABSOLUTE expert id, not by position within
    /// `[e_start, e_start+e_count)`** — the same convention as `descs`, and the same one
    /// [`launch_moe_expert_range`] uses. So both must be sized for `n_desc`, not for `e_count`:
    /// `wexpert` is `n_desc·nrow` f32 and `h` is `n_desc·nrow·inter` f32. A caller that read
    /// these as range-relative and allocated `e_count` of them would run off the end the first
    /// time it passed `e_start > 0`, which is the first thing a two-stream pipeline does.
    /// `x` is `nrow` rows of `hidden` f32 and `acc` is `nrow` rows of `hidden` u64.
    ///
    /// `x` and `h` must be **16-byte aligned**, and this is UNCHECKED: `dot_f4_wave_r`'s fast
    /// path gates on the WEIGHT row's 4-byte alignment and then issues `float4` loads on the
    /// activation regardless, so a misaligned `x` faults rather than falling back to the scalar
    /// tail. Every `DeviceBuf` allocation satisfies it (`hipMalloc`); a pointer into the middle
    /// of one need not.
    ///
    /// `x` and `h` must not ALIAS: both are `__restrict__` in the kernel.
    launch_moe_expert_range_f4 -> rivoli_moe_expert_range_f4, "moe_expert_range_f4" (
        x: *const f32,
        hidden: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        n_desc: usize as i32,
        descs: *const ExpertDescF4,
        wexpert: *const f32,
        swiglu_limit: f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: usize as i32,
        stream: *mut c_void,
    );

    /// `kernel.py::act_quant(v, 128, "ue8m0", inplace=True)` over `n_rows x row_len` f32, in
    /// place — the fp8 activation quantization V4 performs in front of every quantized
    /// `Linear`, fp4-weight ones included.
    ///
    /// Fused quantize-then-dequantize: the buffer stays f32 and holds `e4m3(v/s)·s`. That is
    /// what the reference's `inplace=True` does and what the oracle models, so the values a
    /// following GEMV consumes are the reference's own.
    ///
    /// # Safety
    /// `v` is `n_rows · row_len` live f32 for `stream`'s duration.
    launch_act_quant_f8 -> rivoli_act_quant_f8, "act_quant_f8" (
        v: *mut f32,
        n_rows: usize as i32,
        row_len: usize as i32,
        stream: *mut c_void,
    );

    /// `model.py::Gate.forward` for ONE token: `logits` (a dense f32 GEMV against
    /// `gate.weight`) into `k` routing weights and `k` expert indices.
    ///
    /// Exactly one of `bias` and `tid2eid` may be non-null, and the launcher refuses any other
    /// combination. A hash layer (`layer_id < n_hash_layers`) has `tid2eid` and no bias; a
    /// scored layer has a bias and no `tid2eid`. The two refused combinations are the two
    /// silent defects this router invites: routing a hash layer by score, and letting the
    /// selection bias reach the weights.
    ///
    /// `tid2eid` is `[vocab_size, k]` **i64** — the checkpoint's dtype. `model.py` declares the
    /// parameter `torch.int32`, but every `layers.N.ffn.gate.tid2eid` on disk is `I64`.
    ///
    /// `vocab_size` is `tid2eid`'s row count and is checked against `input_id` (guard 1003); it is
    /// ignored on a scored layer, which has no table to run off. **Entries of `tid2eid` are not
    /// range-checked** — see `moe.hip`'s note; S3 validates the table at load.
    ///
    /// # Safety
    /// `logits` is `n_experts` f32; `bias`, when non-null, the same; `tid2eid`, when non-null,
    /// `vocab_size · k` i64. `weights`/`indices` hold `k` elements. All must outlive `stream`'s
    /// completion — await its [`Signal`](crate::backend::gpustream::Signal).
    launch_moe_gate_v4 -> rivoli_moe_gate_v4, "moe_gate_v4" (
        logits: *const f32,
        bias: *const f32,
        tid2eid: *const i64,
        input_id: usize as i32,
        vocab_size: usize as i32,
        n_experts: usize as i32,
        k: usize as i32,
        route_scale: f32,
        weights: *mut f32,
        indices: *mut i32,
        stream: *mut c_void,
    );

    /// `model.py::Block.hc_pre` over `s` tokens: reduce the `hc` residual copies to one with
    /// Sinkhorn-normalised learned weights, and emit the `post`/`comb` the matching
    /// [`launch_hc_post`] consumes.
    ///
    /// `iters` is `hc_sinkhorn_iters` from the config. **A numerical comparison cannot gate
    /// its exact value**: at 20 passes a 4x4 positive matrix is far past convergence, so 19 and
    /// 20 agree bit-for-bit (`tests/v4_oracle.rs::sinkhorn_has_converged_long_before_
    /// iteration_20`). Passing it from `V4Config` rather than baking it in is what keeps the
    /// count from drifting from `config.json`; what tests can and do prove is that the
    /// parameter is LIVE (2 and 20 disagree).
    ///
    /// `hc` is checked against the kernel's `HC_MULT`, not merely passed: `mix_hc = (2+hc)·hc`
    /// is how the mHC weights are packed on disk, so a mismatch is a different checkpoint.
    ///
    /// # Safety
    /// `h` is `s · hc · dim` f32, `fnw` is `(2+hc)·hc` rows of `hc·dim`, `scale` is 3 and
    /// `base` is `(2+hc)·hc`. Outputs: `y` `s·dim`, `post` `s·hc`, `comb` `s·hc·hc`. All must
    /// outlive `stream`'s completion — await its
    /// [`Signal`](crate::backend::gpustream::Signal) — and no output may alias `h`, which is
    /// `__restrict__` in the kernel.
    launch_hc_pre -> rivoli_hc_pre, "hc_pre" (
        h: *const f32,
        fnw: *const f32,
        scale: *const f32,
        base: *const f32,
        s: usize as i32,
        hc: usize as i32,
        dim: usize as i32,
        iters: usize as i32,
        norm_eps: f32,
        hc_eps: f32,
        y: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        stream: *mut c_void,
    );

    /// `model.py::Block.hc_post`: expand the sublayer output `x` back to `hc` residual copies,
    /// mixing the pre-sublayer `residual` through `comb`.
    ///
    /// `comb` is indexed `[source, dest]`. Transposing it leaves every output row a
    /// combination of the same vectors, so no magnitude or norm check can see it.
    ///
    /// # Safety
    /// `x` is `s·dim` f32, `residual` and `y` are `s·hc·dim`, `post` is `s·hc`, `comb` is
    /// `s·hc·hc`. All must outlive `stream`'s completion — await its
    /// [`Signal`](crate::backend::gpustream::Signal).
    ///
    /// **`y` must not alias `residual`.** An in-place residual expansion is the obvious thing to
    /// want and it is wrong twice over: the two are `__restrict__`, and thread `i` writes
    /// `y[i]` while other threads are still reading every source copy of `residual`, with no
    /// barrier between them.
    launch_hc_post -> rivoli_hc_post, "hc_post" (
        x: *const f32,
        residual: *const f32,
        post: *const f32,
        comb: *const f32,
        s: usize as i32,
        hc: usize as i32,
        dim: usize as i32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// Drain the fixed-point MoE accumulator into the residual:
    /// `x[o] += gain·(Σ_r acc[r][o])·2⁻⁴⁴`, resetting `acc` to zero for the next layer.
    ///
    /// `rows` is ONE ROW PER STREAM, not per expert. Every expert on a given stream shares a
    /// row; separate streams get separate rows because sharing one measured +825 µs on a
    /// 6-miss layer — same atomic count as a 0-miss layer, so the cost was cache lines
    /// bouncing between queues, not the atomics themselves.
    ///
    /// This IS the residual add on a MoE layer — it replaces [`launch_vadd`] there rather
    /// than running before it, so the convert costs no extra pass and needs no barrier of
    /// its own: the end-of-layer `device_sync` already stands between this and the next
    /// layer's first atomic.
    ///
    /// # Safety
    /// `x` and `acc` hold `n` f32 / `n` u64; EVERY stream that accumulated into `acc` must
    /// already have completed.
    launch_moe_acc_drain -> rivoli_moe_acc_drain_s, "moe_acc_drain" (
        x: *mut f32,
        acc: *mut u64,
        n: usize as i32,
        rows: usize as i32,
        gain: f32,
        stream: *mut c_void,
    );

    /// fp8-e4m3 block-scaled GEMV `y = W·x` (attention/dense projections).
    ///
    /// `nrow` token rows (1 or 2) share ONE read of the weights: `x[r·i_dim + i]` →
    /// `y[r·o_dim + o]`. That read is the cost — the attention projections are 165 MB of fp8
    /// per layer against a 24 KB `x` — so a batched verify pass is where this earns its
    /// keep. At `nrow == 1` both indices are the single-row ones and nothing changes.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `x` (`nrow·i_dim` f32),
    /// `packed` (`o_dim·i_dim` bytes), `scale` (block-scale f32), `y` (`nrow·o_dim` f32).
    launch_gemv_fp8 -> rivoli_gemv_fp8, "gemv_fp8" (
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        block: usize as i32,
        nrow: usize as i32,
        y: *mut f32,
    );

    /// MLA absorb: `qabs[head][i] = Σ_d q[head·qh+d]·kv_b[rbase+d][i]` over kv_b's `nope`
    /// absorb rows (rbase = head·(nope+vh)), head-batched. kv_b fp8-e4m3 block-scaled.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `q` (`h·qh` f32),
    /// `kvb` (`h·(nope+vh)·kvl` bytes), `kvb_scale` (block-scale f32), `qabs` (`h·kvl` f32).
    launch_mla_absorb_fp8 -> rivoli_mla_absorb_fp8, "mla_absorb_fp8" (
        q: *const f32,
        kvb: *const u8,
        kvb_scale: *const f32,
        h: usize as i32,
        qh: usize as i32,
        nope: usize as i32,
        vh: usize as i32,
        kvl: usize as i32,
        block: usize as i32,
        nrow: usize as i32,
        qabs: *mut f32,
    );

    /// MLA value: `ctx[head][j] = Σ_i clat[head][i]·kv_b[rbase+nope+j][i]` over kv_b's `vh`
    /// value rows, head-batched. kv_b fp8-e4m3 block-scaled.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `clat` (`h·kvl` f32),
    /// `kvb` (`h·(nope+vh)·kvl` bytes), `kvb_scale` (block-scale f32), `ctx` (`h·vh` f32).
    launch_mla_value_fp8 -> rivoli_mla_value_fp8, "mla_value_fp8" (
        clat: *const f32,
        kvb: *const u8,
        kvb_scale: *const f32,
        h: usize as i32,
        nope: usize as i32,
        vh: usize as i32,
        kvl: usize as i32,
        block: usize as i32,
        nrow: usize as i32,
        ctx: *mut f32,
    );

    /// Group-scaled int4 GEMV `y[o] = Σ_i x·(nibble-8)·scale[o, i/I4_GROUP]` — the MoE
    /// `dot_i4_wave` wave-per-row, for the dot-throughput microbench. `scale` is
    /// `o_dim · i4_groups(i_dim)` f32.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`].
    launch_gemv_i4 -> rivoli_gemv_i4, "gemv_i4" (
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        y: *mut f32,
    );

    /// int8 embedding row lookup: `x[i] = embed[token][i]·scale[token]`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `packed` (`≥(token+1)·hidden`
    /// bytes), `scale` (`≥token+1` f32), `x` (`hidden` f32).
    launch_embed_i8_row -> rivoli_embed_i8_row, "embed_i8_row" (
        packed: *const u8,
        scale: *const f32,
        token: usize as i32,
        hidden: usize as i32,
        x: *mut f32,
    );

    /// Append one token's latent (fp8-e4m3 + per-128 block scale) + roped key (bf16) to
    /// the KV slabs at row `pos`. `kvl` must be a multiple of 128 in `[128, 1024]`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `latent` (`kvl` f32), `rope`
    /// (`ropn` f32), `lc8`/`lscale`/`rc` the KV slabs (row `pos` in-bounds; `n_blocks =
    /// kvl/128`).
    launch_append_kv -> rivoli_append_kv, "append_kv" (
        latent: *const f32,
        rope: *const f32,
        lc8: *mut u8,
        lscale: *mut f32,
        rc: *mut u16,
        pos: usize as i32,
        kvl: usize as i32,
        ropn: usize as i32,
        n_blocks: usize as i32,
    );

    /// Gather each head's roped query segment: `qrope[head·ropn+d] = q[head·qh+nope+d]`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `q` (`h·qh` f32), `qrope`
    /// (`h·ropn` f32).
    launch_gather_rope -> rivoli_gather_rope, "gather_rope" (
        q: *const f32,
        qrope: *mut f32,
        h: usize as i32,
        qh: usize as i32,
        nope: usize as i32,
        ropn: usize as i32,
    );

    /// Record `tag` in `*flag` if any of `x[0..n]` is non-finite (first writer wins).
    ///
    /// The localiser for the intermittent non-finite-logits bug. Adds no sync — the caller
    /// reads `flag` on the argmax D2H the tail already pays — because the host-copy
    /// alternative (`--checksum-x`) perturbs timing enough to hide the fault entirely.
    ///
    /// # Safety
    /// `x` must be `n` device f32; `flag` one device u32, zeroed before the run.
    launch_flag_nonfinite -> rivoli_flag_nonfinite, "flag_nonfinite" (
        x: *const f32,
        n: usize as i32,
        tag: u32,
        flag: *mut u32,
    );

    /// `x += y` — the residual add on a dense-MLP layer.
    ///
    /// It said "`--moe-gain != 1` takes `launch_vaxpy` instead" until 2026-08-06, and that had
    /// been false for some time: `--moe-gain` folds into [`launch_moe_acc_drain`]'s `gain`
    /// multiply, which is the MoE layer's residual add, and the 3 dense layers must NOT be
    /// attenuated with it. `vaxpy` was deleted — this comment is why it survived, since a
    /// launcher that a doc comment says is on a live path reads as reachable to every grep.
    ///
    /// # Safety
    /// `x` and `y` must be device pointers to at least `n` f32.
    launch_vadd -> rivoli_vadd, "vadd" (
        x: *mut f32,
        y: *const f32,
        n: usize as i32,
    );

    /// Greedy argmax over `logits[0..n]` → (`out_idx`, `out_val`); lowest index on a
    /// tie, NaN never wins (matches the host fold).
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `logits` (`n` f32),
    /// `out_idx` (one i32), `out_val` (one f32).
    launch_argmax -> rivoli_argmax, "argmax" (
        logits: *const f32,
        n: usize as i32,
        out_idx: *mut i32,
        out_val: *mut f32,
    );

    /// Per-row int8 GEMV `y = W·x` (lm_head → logits).
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `x` (`i_dim` f32),
    /// `packed` (`o_dim·i_dim` bytes), `scale` (`o_dim` f32), `y` (`o_dim` f32).
    launch_gemv_i8 -> rivoli_gemv_i8, "gemv_i8" (
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        nrow: usize as i32,
        y: *mut f32,
    );

    /// f32 GEMV `y = W·x` (the MoE router gate).
    ///
    /// Takes a `stream`, and it is the only one of GLM's GEMVs that does. The reason is V4: this
    /// is the launcher `Gate.forward` maps to (`linear(x.float(), weight.float())`), so on the V4
    /// layer path it sits between `launch_v4_rmsnorm` and `launch_moe_expert_range_f4`, which both
    /// take one. Left on the null stream it would read a norm that no stream had been waited on —
    /// rivoli's streams are `hipStreamNonBlocking` — giving wrong logits and therefore a wrong
    /// SELECTION, which is the one V4 defect class no downstream numeric comparison can attribute.
    ///
    /// Not a V4-only twin launcher, and the contrast with [`memcpy_dtod_async`] earlier in this
    /// file is the reason: those two are separate entry points because they differ in whether the
    /// HOST blocks, which is a contract split the arena relocation depends on. Two spellings of
    /// one dispatch differing only in a stream handle has no such justification.
    ///
    /// All seven existing call sites pass `null_mut()` and are unchanged in behaviour, on both
    /// backends: `vk::launch_gemv_f32` routes the same argument through `Q::parse`, which maps 0
    /// and 1 alike to `Q::Main` for exactly this case.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion. `stream` is a live `hipStream_t`, or
    /// null for the default stream.
    launch_gemv_f32 -> rivoli_gemv_f32, "gemv_f32" (
        x: *const f32,
        w: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        nrow: usize as i32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// SwiGLU combine `h = silu(g)·u` (dense fp8 MLP; safe in place, `h` may alias `g`).
    ///
    /// # Safety
    /// Device pointers (`g`, `u`, `h` each `n` f32) live until the next [`device_sync`].
    launch_swiglu -> rivoli_swiglu, "swiglu" (
        g: *const f32,
        u: *const f32,
        n: usize as i32,
        h: *mut f32,
    );

    /// DeepSeek-V4's `Expert.forward` combine, for the resident fp8 **shared** expert:
    /// `h = bf16( silu(min(bf16(g), limit)) · clamp(bf16(u), ±limit) )`. Safe in place.
    ///
    /// `MoE.__init__` hands `swiglu_limit` to `shared_experts` as well as to the routed ones
    /// (model.py:632) and `Expert.forward` clamps both, so the shared expert needs the same
    /// clamped arithmetic `launch_moe_expert_range_f4` already runs on the fp4 routed experts.
    /// **NOT YET WIRED, as of 2026-08-05, and this launcher existing does not fix anything by
    /// itself.** The only caller of the unclamped [`launch_swiglu`] is GLM's dense MLP in
    /// `gpu.rs`; V4's MoE layer loop, which is what would call this, does not exist yet. So the
    /// state of the tree is: the clamped combine is available and gated
    /// (`tests/v4_kernel.rs` §7), and the first thing to wire the shared expert must call THIS
    /// and not [`launch_swiglu`] — because reaching for that one gives
    /// `v4oracle::Defect::SwigluUnclamped` on one contribution in seven of all 43 layers,
    /// fluent and wrong.
    ///
    /// # Why this is not [`launch_swiglu`] with a `limit`
    ///
    /// **At no value of `limit` would the two agree**, so a parameter could not have expressed
    /// it. [`launch_swiglu`] is `(g/(1+e^-g))·u` and rounds nothing. Three differences besides
    /// the clamp, each of them a defect the oracle names:
    ///
    /// - both operands are bf16-rounded **before** the clamp (`Linear` stores bf16 and
    ///   `Expert.forward` reads it back with `.float()`) — `Defect::NoBf16Rounding`;
    /// - the product is bf16-rounded, the reference's `x.to(dtype)` in front of `w2`;
    /// - `F.silu`'s multiply form `g·sigmoid(g)`, not the division form, which `moe.hip` records
    ///   as one rounding apart that "would normally vanish under the bf16 store ... except
    ///   exactly at a rounding boundary". This one is **true by construction and unexercised**:
    ///   swapping the forms was measured bit-identical over all 512 outputs of the fp4 fixture
    ///   (2026-08-05), which is what that comment predicts when the boundary is not reached.
    ///   The decision below does not rest on it — the two bf16 roundings and the clamp are each
    ///   demonstrated by a break that goes red.
    ///
    /// GLM has no `swiglu_limit` and should never acquire one, so passing an "unclamped"
    /// sentinel from its four call sites would put a value in the tree that no config can
    /// produce. This refuses `limit <= 0`, **NaN and `+/-inf`** (guard 1006, the same code `moe.hip`
    /// returns for the same check on the same argument), so unclamped is not spellable here.
    ///
    /// The clamp itself is one `kernels/common.hpp::swiglu_clamped`, shared with
    /// `moe_gateup_f4_impl`: the routed and shared paths must agree bit for bit, and `kernels/`
    /// is not scanned by `build.rs`'s jscpd gate, so a second copy would have drifted unseen.
    ///
    /// # Safety
    /// `g`, `u` and `h` are each `n` f32 and must outlive `stream`'s completion. `h` may alias
    /// `g` or `u` — every thread reads both, then writes once, and that write depends on both
    /// reads. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_v4_swiglu_clamped -> rivoli_v4_swiglu_clamped, "v4_swiglu_clamped" (
        g: *const f32,
        u: *const f32,
        n: usize as i32,
        limit: f32,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// RMSNorm `y = x·rsqrt(mean(x²)+eps)·w`.
    ///
    /// # Safety
    /// Device pointers (`x`, `w`, `y` each `n` f32) live until the next [`device_sync`].
    launch_rmsnorm -> rivoli_rmsnorm, "rmsnorm" (
        x: *const f32,
        w: *const f32,
        n: usize as i32,
        eps: f32,
        y: *mut f32,
    );

    /// Interleaved RoPE in place over `count` segments of `seg` at `stride`.
    ///
    /// # Safety
    /// `base` is a device buffer of `count·stride` f32, live until the next [`device_sync`].
    launch_rope -> rivoli_rope, "rope" (
        base: *mut f32,
        count: usize as i32,
        stride: usize as i32,
        seg: usize as i32,
        pos: usize as i32,
        theta: f64,
    );

    /// Batched VQ encode (offline converter accelerator): `idx[i] = argmin_k …`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`].
    launch_vq_encode -> rivoli_vq_encode, "vq_encode" (
        sub: *const f32,
        codebook: *const f32,
        cbnorm: *const f32,
        n: usize as i32,
        idx: *mut u16,
    );

    /// LayerNorm with bias `y = (x-mean)/sqrt(var+eps)·w + b` (the indexer k_norm).
    ///
    /// # Safety
    /// Device pointers (`x`, `w`, `b`, `y` each `n` f32) live until the next [`device_sync`].
    launch_layernorm -> rivoli_layernorm, "layernorm" (
        x: *const f32,
        w: *const f32,
        b: *const f32,
        n: usize as i32,
        eps: f32,
        y: *mut f32,
    );

    /// Append one indexer key row (bf16) at `pos`: `kcache[pos·hd+i] = bf16(k[i])`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `k` (`hd` f32), `kcache`
    /// (row `pos` in-bounds).
    launch_index_append -> rivoli_index_append, "index_append" (
        k: *const f32,
        kcache: *mut u16,
        pos: usize as i32,
        hd: usize as i32,
    );

    /// Score every cached token against the indexer query heads:
    /// `scores[t] = Σ_{h∈active} w[h]·wscale·ReLU((q_h·k_t)·dscale)`. `heads` (nullable)
    /// lists the `nact` active heads (MISA); null = all `nh` heads (DSA).
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `q` (`nh·hd` f32), `w` (`nh`
    /// f32), `kcache` (`nt·hd` bf16), `heads` (`nact` u32 or null), `scores` (`nt` f32).
    launch_index_score -> rivoli_index_score, "index_score" (
        q: *const f32,
        w: *const f32,
        kcache: *const u16,
        heads: *const u32,
        nt: usize as i32,
        nh: usize as i32,
        nact: usize as i32,
        hd: usize as i32,
        wscale: f32,
        dscale: f32,
        scores: *mut f32,
    );

    /// Select the DSA attend row set on device: `rows[0..min(k,nt))`, ASCENDING by index.
    ///
    /// Writes device-side only — no D2H, no host top-k, and no `device_sync`: the attend
    /// consumes `rows` on the same stream, so program order is the whole requirement.
    ///
    /// **Intended** to be bit-identical to the `topk_into(..) + sort_unstable()` it
    /// replaces; `tests/kernel.rs::index_topk_matches_host_selection` is the gate for that
    /// claim. The tiebreak rule and its rationale live at the kernel, once.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `scores` (`nt` f32), `rows`
    /// (at least `min(k, nt)` u32 — the kernel writes exactly that many).
    launch_index_topk -> rivoli_index_topk, "index_topk" (
        scores: *const f32,
        nt: usize as i32,
        k: usize as i32,
        rows: *mut u32,
    );

    /// Fold token `t`'s indexer key into its MISA block pool running mean.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `k` (`hd` f32), `pool`
    /// (block `t/MISA_BLOCK` in-bounds).
    launch_index_pool_push -> rivoli_index_pool_push, "index_pool_push" (
        k: *const f32,
        pool: *mut f32,
        t: usize as i32,
        hd: usize as i32,
    );

    /// MISA head-router estimate `e[j] = mean_b |w[j]·ReLU(q_j·k̄_b)|` over the block pool.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `q` (`nh·hd` f32), `w` (`nh`
    /// f32), `pool` (`m_blocks·hd` f32), `e` (`nh` f32).
    launch_index_head_route -> rivoli_index_head_route, "index_head_route" (
        q: *const f32,
        w: *const f32,
        pool: *const f32,
        m_blocks: usize as i32,
        nh: usize as i32,
        hd: usize as i32,
        e: *mut f32,
    );

    /// `kernel.py::act_quant(x, block, "ue8m0", inplace=True)` in place over `rows` rows of
    /// `row_stride` floats, quantizing the first `n` of each. `n < row_stride` is the KV
    /// entry's PARTIAL quantization (model.py:512, dims `[0, head_dim - rope_head_dim)` at
    /// block 64); `n == row_stride` at block 128 is what every quantized `Linear` does to
    /// its activation before the GEMM.
    ///
    /// # Safety
    /// `x` is a device buffer of at least `rows * row_stride` f32, and must outlive `stream`'s
    /// completion. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_v4_act_quant -> rivoli_v4_act_quant, "v4_act_quant" (
        x: *mut f32,
        rows: usize as i32,
        row_stride: usize as i32,
        n: usize as i32,
        block: usize as i32,
        stream: *mut c_void,
    );

    /// `RMSNorm.forward` over `rows` rows of `d` floats, in place: f32 statistic, learned
    /// weight, bf16-rounded store.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`rows * d` f32), `w` (`d` f32).
    /// `stream` is a live `hipStream_t`, or null for the default stream.
    launch_v4_rmsnorm -> rivoli_v4_rmsnorm, "v4_rmsnorm" (
        x: *mut f32,
        w: *const f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        stream: *mut c_void,
    );

    /// The weightless per-head QK-norm of model.py:504, in place over `rows = s * n_heads`
    /// rows of `head_dim`. Must be launched BEFORE the RoPE — see the kernel's note: the
    /// oracle provably cannot see the order, so it comes from the reference.
    ///
    /// # Safety
    /// `q` is a device buffer of at least `rows * d` f32, and must outlive `stream`'s
    /// completion. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_v4_qk_norm -> rivoli_v4_qk_norm, "v4_qk_norm" (
        q: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        stream: *mut c_void,
    );

    /// fp8-e4m3 GEMV with 128x128 block scales and a bf16-rounded output.
    ///
    /// `x` is `m` rows of `groups` consecutive `k`-wide slices; output row `j` reads slice
    /// `j / (n_out / groups)` of its row. `groups = 1` is a plain `Linear` (every output row
    /// sees the whole activation); `groups = o_groups` is the grouped `wo_a` einsum, whose
    /// input groups are contiguous runs of heads and so need no gather.
    ///
    /// Does NOT quantize the activation — [`launch_v4_act_quant`] is a separate launch where
    /// the reference performs one, and `wo_a` gets none at all.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`m * groups * k` f32), `w`
    /// (`n_out * k` bytes), `wscale` (`ceil(n_out/block) * ceil(k/block)` f32), `out`
    /// (`m * n_out` f32). The `x` bound is exact and follows from the arguments; an earlier
    /// signature took the row and group strides separately, where the in-bounds relation was
    /// a three-way inequality nothing checked. `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_v4_gemv_fp8 -> rivoli_v4_gemv_fp8, "v4_gemv_fp8" (
        x: *const f32,
        w: *const u8,
        wscale: *const f32,
        m: usize as i32,
        n_out: usize as i32,
        k: usize as i32,
        block: usize as i32,
        groups: usize as i32,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// `kernel.py::sparse_attn` — MQA over one `d`-wide entry that is both key and value for
    /// all `h` heads, gathered by `idxs` (`-1` masks a slot), with `sink` entering the
    /// softmax DENOMINATOR only.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `q` (`m * h * d` f32), `kv` (`d` f32
    /// per row, indexed by `idxs`, so at least `max(idxs) + 1` rows), `sink` (`h` f32), `idxs`
    /// (`m * topk` i32), `o` (`m * h * d` f32). `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_v4_sparse_attn -> rivoli_v4_sparse_attn, "v4_sparse_attn" (
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        m: usize as i32,
        h: usize as i32,
        d: usize as i32,
        topk: usize as i32,
        scale: f32,
        o: *mut f32,
        stream: *mut c_void,
    );

    /// `out[m, n] = x[m, k] · w[n, k]^T` with `w` in **bf16** — the un-quantized `F.linear`
    /// path, which is the one `Compressor.wkv`/`wgate` take (`Linear(..., dtype=float32)`,
    /// model.py:302).
    ///
    /// Deliberately NOT [`launch_v4_gemv_fp8`]: that one quantizes the activation to fp8 at
    /// block 128 in front of the GEMM, which the reference does only for quantized `Linear`s.
    /// Sending the compressor through it would introduce a quantization the reference never
    /// applies, and the resulting error would be indistinguishable from a pooling bug.
    ///
    /// # Safety
    /// `x` is `m · k` live f32, `w` is `n · k` live u16, `out` is `m · n` writable f32, none
    /// aliasing another (every kernel parameter is `__restrict__`), all live until `stream`
    /// completes. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_v4_dense_gemm_bf16 -> rivoli_v4_dense_gemm_bf16, "v4_dense_gemm_bf16" (
        x: *const f32,
        w: *const u16,
        out: *mut f32,
        m: usize as i32,
        n: usize as i32,
        k: usize as i32,
        stream: *mut c_void,
    );

    /// `Transformer.forward` 914-916: gather token `token`'s bf16 embedding row and broadcast it
    /// into `hc` copies. `x` receives `hc * hidden` f32.
    ///
    /// # Safety
    /// `w` is `>= (token + 1) * hidden` live u16, `x` is `hc * hidden` writable f32, they do not
    /// alias (both are `__restrict__`), and both outlive `stream`'s completion. `stream` is a
    /// live `hipStream_t`, or null for the default stream.
    launch_v4_embed_bf16_row -> rivoli_v4_embed_bf16_row, "v4_embed_bf16_row" (
        w: *const u16,
        token: usize as i32,
        hidden: usize as i32,
        hc: usize as i32,
        x: *mut f32,
        stream: *mut c_void,
    );

    /// `Block.hc_head` (model.py:709-716): collapse `[s, hc, dim]` to `[s, dim]`, bf16-rounded.
    ///
    /// Two kernels on one stream, so no host sync sits between them. `pre` is `s * hc` f32 of
    /// SCRATCH — the gate vector — and is written before it is read.
    ///
    /// # Safety
    /// `h` is `s * hc * dim` live f32; `fnw` is `hc * hc * dim`; `base` is `hc`; `scale` is 1;
    /// `pre` is `s * hc` writable; `y` is `s * dim` writable. None aliases another (every kernel
    /// parameter is `__restrict__`) and all outlive `stream`'s completion. `stream` is a live
    /// `hipStream_t`, or null for the default stream.
    launch_v4_hc_head -> rivoli_v4_hc_head, "v4_hc_head" (
        h: *const f32,
        fnw: *const f32,
        base: *const f32,
        scale: *const f32,
        pre: *mut f32,
        y: *mut f32,
        s: usize as i32,
        hc: usize as i32,
        dim: usize as i32,
        eps: f32,
        hc_eps: f32,
        stream: *mut c_void,
    );

    /// `rotate_activation` then `fp4_act_quant(·, 32, inplace=True)` over `rows` rows of `d`
    /// floats, in place — `Oracle::indexer_spread` (forward.rs:1130-1138) and the finish
    /// `Compressor.forward` performs when `rotate = true` (model.py:374-376).
    ///
    /// Applied to BOTH the indexer's `q` rows and its nested compressor's pooled rows, which is
    /// why it is one launcher rather than a step inside either. [`launch_v4_act_quant`] is the
    /// *other* compressor's finish and takes a partial extent; this one has none — the Hadamard
    /// covers the whole row, RoPE tail included. Handing either the other's extent is finite,
    /// plausible and wrong, so `v4compress::Geom` carries which is due and
    /// `v4compress::compress` matches on it.
    ///
    /// `d` must be a power of two no greater than 256 and a multiple of 32; the launcher
    /// refuses otherwise (guards 1002/1003/1004) rather than transforming a length the
    /// reference would have zero-padded, or quantizing a ragged tail against its own amax.
    ///
    /// # Safety
    /// `x` is `rows · d` writable, 4-byte-aligned, device-resident f32, read and written in
    /// place, and outlives `stream`'s completion. `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_v4_indexer_spread -> rivoli_v4_indexer_spread, "v4_indexer_spread" (
        x: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        stream: *mut c_void,
    );

    /// MLA flash attention `clat = Σ_t softmax((qabs·L_t + qrope·R_t)·scale)·L_t` over
    /// the fp8-e4m3 latent cache (per-128 block scales) + bf16 roped key, head-batched,
    /// split-KV when `partial` is non-null.
    ///
    /// `rows` (nullable) lists the `nr` attended token indices for DSA sparse attention;
    /// null = dense over the whole `0..nr` causal prefix.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `qabs` (`h·kvl` f32),
    /// `qrope` (`h·rope` f32), `lc8`/`lscale`/`rc` the KV cache (indexed by token — up to
    /// `pos+1` rows; `n_blocks = kvl/128`), `rows` (`nr` u32 or null), `clat` (`h·kvl`
    /// f32), `partial` ([`attend_scratch_floats`] f32 or null = single split).
    launch_attend -> rivoli_mla_attend, "mla_attend" (
        qabs: *const f32,
        qrope: *const f32,
        lc8: *const u8,
        lscale: *const f32,
        rc: *const u16,
        rows: *const u32,
        h: usize as i32,
        nr: usize as i32,
        kvl: usize as i32,
        rope: usize as i32,
        n_blocks: usize as i32,
        scale: f32,
        clat: *mut f32,
        partial: *mut f32,
    );

    /// `apply_rotary_emb` over the last `rd` dims of each of `rows` rows, ADJACENT-PAIR
    /// (`view_as_complex`), from a precomputed `(cos, sin)` table. Row `r` takes position
    /// `pos0 + r / rows_per_pos`. `inverse` conjugates it — the output de-rotation.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`rows * row_len` f32), `tbl`
    /// (at least `(pos0 + rows / rows_per_pos) * rd` f32, interleaved cos/sin). `stream` is a
    /// live `hipStream_t`, or null for the default stream.
    launch_v4_rope -> rivoli_v4_rope, "v4_rope" (
        x: *mut f32,
        tbl: *const f32,
        rows: usize as i32,
        row_len: usize as i32,
        rd: usize as i32,
        pos0: usize as i32,
        rows_per_pos: usize as i32,
        inverse: bool as i32,
        stream: *mut c_void,
    );

    /// The state deposit of `Compressor.forward` — **both phases**, which are one operation
    /// distinguished only by `slot0`.
    ///
    /// A prefill of `s` tokens deposits its `s % ratio` trailing rows starting at slot 0; a
    /// decode deposits its single row at slot `start_pos % ratio`. See
    /// `kernels/v4compress.hip::v4_compress_state` for why that is a unification and not a
    /// coincidence.
    ///
    /// Must be launched on **every** call, including one that emits no block: the reference
    /// writes the state and only then returns `None`. At ratio 128 that is every prompt under
    /// 128 tokens and 127 of every 128 decode steps.
    ///
    /// Refuses `s <= 0` (guard 1005) and a `slot0` whose run would leave the `[ratio, cd]`
    /// `ape` table (guard 1008).
    ///
    /// # Safety
    /// `kv`/`score` are `s · p.cd()` live f32; `ape` is `p.ratio() · p.cd()`; the two state
    /// buffers are `p.state_len()` writable f32. None may alias another — every kernel
    /// parameter is `__restrict__`. `p` is read host-side before the launch; the device buffers
    /// must outlive `stream`'s completion. `stream` is a live `hipStream_t`, or null for the
    /// default stream.
    launch_v4_compress_state -> rivoli_v4_compress_state, "v4_compress_state" (
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        kv_state: *mut f32,
        score_state: *mut f32,
        p: &GeomAbi as *const GeomAbi,
        s: usize as i32,
        slot0: usize as i32,
        stream: *mut c_void,
    );

    /// Prefill pooling for `nblk` compressed blocks — `overlap_transform`, the per-feature
    /// softmax over the pooling window, the bf16 store, `RMSNorm`, and the RoPE at each block's
    /// FIRST absolute position.
    ///
    /// Does **not** run `act_quant`; call [`launch_v4_act_quant`] over dims `[0, d - rd)` at
    /// block 64 afterwards, which is the order and the partial extent model.py:373-378 uses.
    ///
    /// Refuses `nblk <= 0` (guard 1006) rather than launching nothing and returning success,
    /// which would hand the caller an unwritten `out`.
    ///
    /// # Safety
    /// `kv`/`score` are at least `nblk · p.ratio() · p.cd()` live f32; `ape` is
    /// `p.ratio() · p.cd()`; `f` satisfies [`Finish`]'s field contract with `out` sized
    /// `nblk · p.d()` and `freqs` covering position `(nblk - 1) · p.ratio()`. None may alias
    /// another. All must outlive `stream`'s completion; `stream` is a live `hipStream_t`, or
    /// null for the default stream.
    launch_v4_compress_prefill -> rivoli_v4_compress_prefill, "v4_compress_prefill" (
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        f: &Finish as *const Finish,
        p: &GeomAbi as *const GeomAbi,
        nblk: usize as i32,
        stream: *mut c_void,
    );

    /// Pool one COMPLETED decode window out of the compressor state into a single block, and
    /// slide the window.
    ///
    /// Reads no activation: this step's row was already deposited by
    /// [`launch_v4_compress_state`], `ape` included. Call **only** when
    /// `(start_pos + 1) % ratio == 0`; the launcher refuses otherwise (guard 1009) rather than
    /// pooling a half-filled window into finite, plausible, wrong numbers.
    ///
    /// # Safety
    /// The two state buffers are `p.state_len()` f32 and are read-modify-written; `f` satisfies
    /// [`Finish`]'s field contract with `out` sized one row of `p.d()` and `freqs` covering
    /// position `(start_pos / ratio) * ratio`. None may alias another. All must outlive
    /// `stream`'s completion; `stream` is a live `hipStream_t`, or null for the default stream.
    launch_v4_compress_pool_decode -> rivoli_v4_compress_pool_decode, "v4_compress_pool_decode" (
        kv_state: *mut f32,
        score_state: *mut f32,
        f: &Finish as *const Finish,
        p: &GeomAbi as *const GeomAbi,
        start_pos: usize as i32,
        stream: *mut c_void,
    );
}
// jscpd:ignore-end

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
//   `launch_v4_indexer_score`  destructures `ScoreDims` into four i32s before the call, with
//                              prose in place saying the struct exists to keep the four in the
//                              right order. A positional 1:1 DSL cannot express one parameter
//                              becoming four arguments, and the alternative — four bare `usize`
//                              on the wrapper — deletes the thing the struct is for.
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

    fn rivoli_mla_attend_scratch_floats(h: i32, kvl: i32) -> usize;

    // DSA lightning indexer (indexer.hip).

    fn rivoli_v4_indexer_score(
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
fn ensure_hip_decode_status(r: i32, name: &str) -> Result<()> {
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
    ensure_hip_decode_status(unsafe { rivoli_device_sync() }, "device_sync")
}

/// Synchronous device-to-device copy of `bytes` from `src` to `dst` — the routed
/// arena's slot relocation (compaction). BLOCKS, so the moved expert is in place before
/// any later kernel reads the new slot.
///
/// # Safety
/// `dst` and `src` must be valid, `bytes`-sized, NON-OVERLAPPING device regions (the
/// arena guarantees distinct slots).
pub unsafe fn memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
    ensure_hip_decode_status(
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
/// before `v4_rmsnorm` wrote it. See the banner above the V4 attention launchers.
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
    ensure_hip_decode_status(
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
    ensure_hip_decode_status(unsafe { rivoli_fill_u32(dst, pat, bytes) }, "fill_u32")
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
// The substance behind the count, for these six: `tests/v4_attn.rs` scores four of them
// inside the whole attention block, and `tests/v4_head_tail.rs` scores `v4_rmsnorm` and
// `v4_qk_norm` on their own. An earlier version of this line said `v4_attn.rs` covered all
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
pub unsafe fn launch_v4_indexer_score(
    q: *const f32,
    kv: *const f32,
    w: *const f32,
    score: *mut f32,
    dims: ScoreDims,
    stream: *mut c_void,
) -> Result<()> {
    // `dims`, not `d`: `d` means a head width everywhere else in this file, including in
    // `launch_v4_indexer_spread` directly above.
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
    let r = unsafe { rivoli_v4_indexer_score(q, kv, w, score, s, n_comp, heads, hd, stream) };
    ensure_hip_decode_status(r, "v4_indexer_score")
}
// jscpd:ignore-end
