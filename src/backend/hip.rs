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

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE, and this is the only kind of thing that is.
//
// What follows is not code, it is an ABI: the argument lists of the C entry points in
// `kernels/*.hip`. They necessarily match the Vulkan launchers in `backend/vk.rs`, because
// `rocm` and `vulkan` are mutually exclusive and `backend.rs` cfg-selects one glob of the
// SAME names — `gpu.rs` calls `launch_attend` and gets whichever was compiled. Identical
// signatures are the contract, not copy-paste.
//
// It cannot be deduplicated without making the code worse. `vk.rs`'s own note above
// `launch_gemv_fp8` states the reason: "the two must stay readable side by side for the
// bit-exactness comparison to be checkable by eye. Bundling them into a struct here and
// not there would put a translation step between the two signatures, which is the one
// place this port cannot afford one." A macro that declares each signature once would
// remove ~15 of these while breaking goto-definition on every launcher, and roughly 25 of
// the rest are DIFFERENT kernels that merely take the same shape (`gemv_fp8`/`i8`/`i4`/`vq`
// all take `x, packed, scale, o_dim, i_dim, y`) — there is one copy of each already and
// nothing to merge.
//
// The real instrument for "do the two backends agree" is behavioural, not syntactic:
// `tests/xbackend.rs` runs each arm under its own feature and compares raw output bytes.
//
// The gate stays live over everything else in this file — it found four genuine
// duplicated-logic clones here on 2026-08-01 (`record_barrier`, `Stream::live`,
// `refill_from_mapping`, `dispatch_on`), all outside this block.
unsafe extern "C" {
    fn rivoli_device_sync() -> i32;
    fn rivoli_memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> i32;
    fn rivoli_fill_u32(dst: *mut u8, pat: u32, bytes: usize) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_expert_range_i4(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e_start: i32,
        e_count: i32,
        descs: *const ExpertDesc,
        wexpert: *const f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: i32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_expert_range(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e_start: i32,
        e_count: i32,
        descs: *const ExpertDesc,
        gate_cb: *const u16,
        up_cb: *const u16,
        down_cb: *const u16,
        wexpert: *const f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: i32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_expert_range_f4(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e_start: i32,
        e_count: i32,
        n_desc: i32,
        descs: *const ExpertDescF4,
        wexpert: *const f32,
        swiglu_limit: f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: i32,
        stream: *mut c_void,
    ) -> i32;

    fn rivoli_act_quant_f8(v: *mut f32, n_rows: i32, row_len: i32, stream: *mut c_void) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_gate_v4(
        logits: *const f32,
        bias: *const f32,
        tid2eid: *const i64,
        input_id: i32,
        vocab_size: i32,
        n_experts: i32,
        k: i32,
        route_scale: f32,
        weights: *mut f32,
        indices: *mut i32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_hc_pre(
        h: *const f32,
        fnw: *const f32,
        scale: *const f32,
        base: *const f32,
        s: i32,
        hc: i32,
        dim: i32,
        iters: i32,
        norm_eps: f32,
        hc_eps: f32,
        y: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_hc_post(
        x: *const f32,
        residual: *const f32,
        post: *const f32,
        comb: *const f32,
        s: i32,
        hc: i32,
        dim: i32,
        y: *mut f32,
        stream: *mut c_void,
    ) -> i32;

    fn rivoli_moe_acc_drain_s(
        x: *mut f32,
        acc: *mut u64,
        n: i32,
        rows: i32,
        gain: f32,
        stream: *mut c_void,
    ) -> i32;

    fn rivoli_gemv_fp8(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        block: i32,
        nrow: i32,
        y: *mut f32,
    ) -> i32;

    fn rivoli_gemv_vq(
        x: *const f32,
        indices: *const u8,
        scales: *const u16,
        codebook: *const u16,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;
    fn rivoli_gemv_i4(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_absorb_fp8(
        q: *const f32,
        kvb: *const u8,
        kvb_scale: *const f32,
        h: i32,
        qh: i32,
        nope: i32,
        vh: i32,
        kvl: i32,
        block: i32,
        nrow: i32,
        qabs: *mut f32,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_value_fp8(
        clat: *const f32,
        kvb: *const u8,
        kvb_scale: *const f32,
        h: i32,
        nope: i32,
        vh: i32,
        kvl: i32,
        block: i32,
        nrow: i32,
        ctx: *mut f32,
    ) -> i32;

    fn rivoli_mla_attend_scratch_floats(h: i32, kvl: i32) -> usize;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_attend(
        qabs: *const f32,
        qrope: *const f32,
        lc8: *const u8,
        lscale: *const f32,
        rc: *const u16,
        rows: *const u32,
        h: i32,
        nr: i32,
        kvl: i32,
        rope: i32,
        n_blocks: i32,
        scale: f32,
        clat: *mut f32,
        partial: *mut f32,
    ) -> i32;

    fn rivoli_embed_i8_row(
        packed: *const u8,
        scale: *const f32,
        token: i32,
        hidden: i32,
        x: *mut f32,
    ) -> i32;
    fn rivoli_append_kv(
        latent: *const f32,
        rope: *const f32,
        lc8: *mut u8,
        lscale: *mut f32,
        rc: *mut u16,
        pos: i32,
        kvl: i32,
        ropn: i32,
        n_blocks: i32,
    ) -> i32;
    fn rivoli_gather_rope(
        q: *const f32,
        qrope: *mut f32,
        h: i32,
        qh: i32,
        nope: i32,
        ropn: i32,
    ) -> i32;
    fn rivoli_vadd(x: *mut f32, y: *const f32, n: i32) -> i32;
    fn rivoli_flag_nonfinite(x: *const f32, n: i32, tag: u32, flag: *mut u32) -> i32;
    fn rivoli_vaxpy(x: *mut f32, y: *const f32, g: f32, n: i32) -> i32;
    fn rivoli_argmax(logits: *const f32, n: i32, out_idx: *mut i32, out_val: *mut f32) -> i32;

    fn rivoli_gemv_i8(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        nrow: i32,
        y: *mut f32,
    ) -> i32;
    fn rivoli_gemv_f32(
        x: *const f32,
        w: *const f32,
        o_dim: i32,
        i_dim: i32,
        nrow: i32,
        y: *mut f32,
    ) -> i32;
    fn rivoli_swiglu(g: *const f32, u: *const f32, n: i32, h: *mut f32) -> i32;
    fn rivoli_rmsnorm(x: *const f32, w: *const f32, n: i32, eps: f32, y: *mut f32) -> i32;
    fn rivoli_rope(base: *mut f32, count: i32, stride: i32, seg: i32, pos: i32, theta: f64) -> i32;
    fn rivoli_vq_encode(
        sub: *const f32,
        codebook: *const f32,
        cbnorm: *const f32,
        n: i32,
        idx: *mut u16,
    ) -> i32;

    // DSA lightning indexer (indexer.hip).
    fn rivoli_layernorm(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        n: i32,
        eps: f32,
        y: *mut f32,
    ) -> i32;
    fn rivoli_index_append(k: *const f32, kcache: *mut u16, pos: i32, hd: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn rivoli_index_score(
        q: *const f32,
        w: *const f32,
        kcache: *const u16,
        heads: *const u32,
        nt: i32,
        nh: i32,
        nact: i32,
        hd: i32,
        wscale: f32,
        dscale: f32,
        scores: *mut f32,
    ) -> i32;
    fn rivoli_index_topk(scores: *const f32, nt: i32, k: i32, rows: *mut u32) -> i32;
    fn rivoli_index_pool_push(k: *const f32, pool: *mut f32, t: i32, hd: i32) -> i32;
    fn rivoli_index_head_route(
        q: *const f32,
        w: *const f32,
        pool: *const f32,
        m_blocks: i32,
        nh: i32,
        hd: i32,
        e: *mut f32,
    ) -> i32;
    fn rivoli_v4_act_quant(x: *mut f32, rows: i32, row_stride: i32, n: i32, block: i32) -> i32;
    fn rivoli_v4_rmsnorm(x: *mut f32, w: *const f32, rows: i32, d: i32, eps: f32) -> i32;
    fn rivoli_v4_qk_norm(q: *mut f32, rows: i32, d: i32, eps: f32) -> i32;
    fn rivoli_v4_rope(
        x: *mut f32,
        tbl: *const f32,
        rows: i32,
        row_len: i32,
        rd: i32,
        pos0: i32,
        rows_per_pos: i32,
        inverse: i32,
    ) -> i32;
    fn rivoli_v4_gemv_fp8(
        x: *const f32,
        w: *const u8,
        wscale: *const f32,
        m: i32,
        n_out: i32,
        k: i32,
        block: i32,
        groups: i32,
        out: *mut f32,
    ) -> i32;
    fn rivoli_v4_sparse_attn(
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        m: i32,
        h: i32,
        d: i32,
        topk: i32,
        scale: f32,
        o: *mut f32,
    ) -> i32;

    fn rivoli_v4_dense_gemm_bf16(
        x: *const f32,
        w: *const u16,
        out: *mut f32,
        m: i32,
        n: i32,
        k: i32,
        stream: *mut c_void,
    ) -> i32;
    fn rivoli_v4_compress_state(
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        kv_state: *mut f32,
        score_state: *mut f32,
        p: *const GeomAbi,
        s: i32,
        slot0: i32,
        stream: *mut c_void,
    ) -> i32;
    fn rivoli_v4_compress_prefill(
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        f: *const Finish,
        p: *const GeomAbi,
        nblk: i32,
        stream: *mut c_void,
    ) -> i32;
    fn rivoli_v4_compress_pool_decode(
        kv_state: *mut f32,
        score_state: *mut f32,
        f: *const Finish,
        p: *const GeomAbi,
        start_pos: i32,
        stream: *mut c_void,
    ) -> i32;
    fn rivoli_v4_indexer_spread(x: *mut f32, rows: i32, d: i32, stream: *mut c_void) -> i32;
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
// jscpd:ignore-end

/// Launcher return-code check: 0 = ok, POSITIVE = arg guard, NEGATIVE = -(hipError_t).
fn check(r: i32, name: &str) -> Result<()> {
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
    check(unsafe { rivoli_device_sync() }, "device_sync")
}

/// Synchronous device-to-device copy of `bytes` from `src` to `dst` — the routed
/// arena's slot relocation (compaction). BLOCKS, so the moved expert is in place before
/// any later kernel reads the new slot.
///
/// # Safety
/// `dst` and `src` must be valid, `bytes`-sized, NON-OVERLAPPING device regions (the
/// arena guarantees distinct slots).
pub unsafe fn memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> Result<()> {
    check(
        unsafe { rivoli_memcpy_dtod(dst, src, bytes) },
        "memcpy_dtod",
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
    check(unsafe { rivoli_fill_u32(dst, pat, bytes) }, "fill_u32")
}

// jscpd:ignore-start
//
// THE LAUNCHER WALL — same exemption, and for the same reason, as the `extern "C"` block
// above and `vk.rs`'s wall: from here to the end of the file every item is a signature
// that mirrors either a `kernels/*.hip` entry point or the Vulkan launcher of the same
// name, and the mirroring is the contract. See the note above the extern block for the
// full argument, and for why a signature-declaring macro would not pay.
//
// The gate stays live over everything above — `check`, `device_sync` and the memcpy/fill
// helpers, which is where this file has logic rather than declarations.

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_expert_range(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e_start: usize,
    e_count: usize,
    descs: *const ExpertDesc,
    gate_cb: *const u16,
    up_cb: *const u16,
    down_cb: *const u16,
    wexpert: *const f32,
    h: *mut f32,
    acc: *mut u64,
    nrow: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_moe_expert_range(
            x,
            hidden as i32,
            inter as i32,
            e_start as i32,
            e_count as i32,
            descs,
            gate_cb,
            up_cb,
            down_cb,
            wexpert,
            h,
            acc,
            nrow as i32,
            stream,
        )
    };
    check(r, "moe_expert_range")
}

/// int4 counterpart of [`launch_moe_expert_range`]: gate/up + down for the absolute
/// range `[e_start, e_start+e_count)` on `stream`, decoding int4 (f32 group scales).
/// `descs` are [`ExpertDesc`]; contributions land in the same fixed-point `acc` row,
/// so int4 and VQ experts of one layer mix freely within a batch.
///
/// # Safety
/// Every device pointer (`descs`/packed weights/`wexpert`/`x`/`h`/`acc`) must
/// outlive `stream`'s completion — await its [`Signal`](crate::backend::gpustream::Signal).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_expert_range_i4(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e_start: usize,
    e_count: usize,
    descs: *const ExpertDesc,
    wexpert: *const f32,
    h: *mut f32,
    acc: *mut u64,
    nrow: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_moe_expert_range_i4(
            x,
            hidden as i32,
            inter as i32,
            e_start as i32,
            e_count as i32,
            descs,
            wexpert,
            h,
            acc,
            nrow as i32,
            stream,
        )
    };
    check(r, "moe_expert_range_i4")
}

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
/// `swiglu_limit` comes from the config (`10.0`); the launcher refuses `<= 0` because an
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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_expert_range_f4(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e_start: usize,
    e_count: usize,
    n_desc: usize,
    descs: *const ExpertDescF4,
    wexpert: *const f32,
    swiglu_limit: f32,
    h: *mut f32,
    acc: *mut u64,
    nrow: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_moe_expert_range_f4(
            x,
            hidden as i32,
            inter as i32,
            e_start as i32,
            e_count as i32,
            n_desc as i32,
            descs,
            wexpert,
            swiglu_limit,
            h,
            acc,
            nrow as i32,
            stream,
        )
    };
    check(r, "moe_expert_range_f4")
}

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
pub unsafe fn launch_act_quant_f8(
    v: *mut f32,
    n_rows: usize,
    row_len: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_act_quant_f8(v, n_rows as i32, row_len as i32, stream) };
    check(r, "act_quant_f8")
}

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_gate_v4(
    logits: *const f32,
    bias: *const f32,
    tid2eid: *const i64,
    input_id: usize,
    vocab_size: usize,
    n_experts: usize,
    k: usize,
    route_scale: f32,
    weights: *mut f32,
    indices: *mut i32,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_moe_gate_v4(
            logits,
            bias,
            tid2eid,
            input_id as i32,
            vocab_size as i32,
            n_experts as i32,
            k as i32,
            route_scale,
            weights,
            indices,
            stream,
        )
    };
    check(r, "moe_gate_v4")
}

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_hc_pre(
    h: *const f32,
    fnw: *const f32,
    scale: *const f32,
    base: *const f32,
    s: usize,
    hc: usize,
    dim: usize,
    iters: usize,
    norm_eps: f32,
    hc_eps: f32,
    y: *mut f32,
    post: *mut f32,
    comb: *mut f32,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_hc_pre(
            h,
            fnw,
            scale,
            base,
            s as i32,
            hc as i32,
            dim as i32,
            iters as i32,
            norm_eps,
            hc_eps,
            y,
            post,
            comb,
            stream,
        )
    };
    check(r, "hc_pre")
}

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_hc_post(
    x: *const f32,
    residual: *const f32,
    post: *const f32,
    comb: *const f32,
    s: usize,
    hc: usize,
    dim: usize,
    y: *mut f32,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_hc_post(x, residual, post, comb, s as i32, hc as i32, dim as i32, y, stream)
    };
    check(r, "hc_post")
}

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
pub unsafe fn launch_moe_acc_drain(
    x: *mut f32,
    acc: *mut u64,
    n: usize,
    rows: usize,
    gain: f32,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_moe_acc_drain_s(x, acc, n as i32, rows as i32, gain, stream) };
    check(r, "moe_acc_drain")
}

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemv_fp8(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    block: usize,
    nrow: usize,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_gemv_fp8(
            x,
            packed,
            scale,
            o_dim as i32,
            i_dim as i32,
            block as i32,
            nrow as i32,
            y,
        )
    };
    check(r, "gemv_fp8")
}

/// f32 count for the split-KV partial scratch — allocate once per session (never
/// per token). Mirrors the kernel's worst-case (MLA_MAX_SPLITS) sizing.
pub fn attend_scratch_floats(h: usize, kvl: usize) -> usize {
    // SAFETY: pure arithmetic, no pointers.
    unsafe { rivoli_mla_attend_scratch_floats(h as i32, kvl as i32) }
}

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_attend(
    qabs: *const f32,
    qrope: *const f32,
    lc8: *const u8,
    lscale: *const f32,
    rc: *const u16,
    rows: *const u32,
    h: usize,
    nr: usize,
    kvl: usize,
    rope: usize,
    n_blocks: usize,
    scale: f32,
    clat: *mut f32,
    partial: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_mla_attend(
            qabs,
            qrope,
            lc8,
            lscale,
            rc,
            rows,
            h as i32,
            nr as i32,
            kvl as i32,
            rope as i32,
            n_blocks as i32,
            scale,
            clat,
            partial,
        )
    };
    check(r, "mla_attend")
}

/// MLA absorb: `qabs[head][i] = Σ_d q[head·qh+d]·kv_b[rbase+d][i]` over kv_b's `nope`
/// absorb rows (rbase = head·(nope+vh)), head-batched. kv_b fp8-e4m3 block-scaled.
///
/// # Safety
/// Async device pointers live until the next [`device_sync`]: `q` (`h·qh` f32),
/// `kvb` (`h·(nope+vh)·kvl` bytes), `kvb_scale` (block-scale f32), `qabs` (`h·kvl` f32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_mla_absorb_fp8(
    q: *const f32,
    kvb: *const u8,
    kvb_scale: *const f32,
    h: usize,
    qh: usize,
    nope: usize,
    vh: usize,
    kvl: usize,
    block: usize,
    nrow: usize,
    qabs: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_mla_absorb_fp8(
            q,
            kvb,
            kvb_scale,
            h as i32,
            qh as i32,
            nope as i32,
            vh as i32,
            kvl as i32,
            block as i32,
            nrow as i32,
            qabs,
        )
    };
    check(r, "mla_absorb_fp8")
}

/// MLA value: `ctx[head][j] = Σ_i clat[head][i]·kv_b[rbase+nope+j][i]` over kv_b's `vh`
/// value rows, head-batched. kv_b fp8-e4m3 block-scaled.
///
/// # Safety
/// Async device pointers live until the next [`device_sync`]: `clat` (`h·kvl` f32),
/// `kvb` (`h·(nope+vh)·kvl` bytes), `kvb_scale` (block-scale f32), `ctx` (`h·vh` f32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_mla_value_fp8(
    clat: *const f32,
    kvb: *const u8,
    kvb_scale: *const f32,
    h: usize,
    nope: usize,
    vh: usize,
    kvl: usize,
    block: usize,
    nrow: usize,
    ctx: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_mla_value_fp8(
            clat,
            kvb,
            kvb_scale,
            h as i32,
            nope as i32,
            vh as i32,
            kvl as i32,
            block as i32,
            nrow as i32,
            ctx,
        )
    };
    check(r, "mla_value_fp8")
}

/// VQ-int3 GEMV `y = W·x` (group scales applied inside the decode).
///
/// # Safety
/// Device pointers live until the next [`device_sync`].
pub unsafe fn launch_gemv_vq(
    x: *const f32,
    indices: *const u8,
    scales: *const u16,
    codebook: *const u16,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_vq(x, indices, scales, codebook, o_dim as i32, i_dim as i32, y) };
    check(r, "gemv_vq")
}

/// Group-scaled int4 GEMV `y[o] = Σ_i x·(nibble-8)·scale[o, i/I4_GROUP]` — the MoE
/// `dot_i4_wave` wave-per-row, for the dot-throughput microbench. `scale` is
/// `o_dim · i4_groups(i_dim)` f32.
///
/// # Safety
/// Device pointers live until the next [`device_sync`].
pub unsafe fn launch_gemv_i4(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_i4(x, packed, scale, o_dim as i32, i_dim as i32, y) };
    check(r, "gemv_i4")
}

/// int8 embedding row lookup: `x[i] = embed[token][i]·scale[token]`.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `packed` (`≥(token+1)·hidden`
/// bytes), `scale` (`≥token+1` f32), `x` (`hidden` f32).
pub unsafe fn launch_embed_i8_row(
    packed: *const u8,
    scale: *const f32,
    token: usize,
    hidden: usize,
    x: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_embed_i8_row(packed, scale, token as i32, hidden as i32, x) };
    check(r, "embed_i8_row")
}

/// Append one token's latent (fp8-e4m3 + per-128 block scale) + roped key (bf16) to
/// the KV slabs at row `pos`. `kvl` must be a multiple of 128 in `[128, 1024]`.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `latent` (`kvl` f32), `rope`
/// (`ropn` f32), `lc8`/`lscale`/`rc` the KV slabs (row `pos` in-bounds; `n_blocks =
/// kvl/128`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_append_kv(
    latent: *const f32,
    rope: *const f32,
    lc8: *mut u8,
    lscale: *mut f32,
    rc: *mut u16,
    pos: usize,
    kvl: usize,
    ropn: usize,
    n_blocks: usize,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_append_kv(
            latent,
            rope,
            lc8,
            lscale,
            rc,
            pos as i32,
            kvl as i32,
            ropn as i32,
            n_blocks as i32,
        )
    };
    check(r, "append_kv")
}

/// Gather each head's roped query segment: `qrope[head·ropn+d] = q[head·qh+nope+d]`.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `q` (`h·qh` f32), `qrope`
/// (`h·ropn` f32).
pub unsafe fn launch_gather_rope(
    q: *const f32,
    qrope: *mut f32,
    h: usize,
    qh: usize,
    nope: usize,
    ropn: usize,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gather_rope(q, qrope, h as i32, qh as i32, nope as i32, ropn as i32) };
    check(r, "gather_rope")
}

/// Record `tag` in `*flag` if any of `x[0..n]` is non-finite (first writer wins).
///
/// The localiser for the intermittent non-finite-logits bug. Adds no sync — the caller
/// reads `flag` on the argmax D2H the tail already pays — because the host-copy
/// alternative (`--checksum-x`) perturbs timing enough to hide the fault entirely.
///
/// # Safety
/// `x` must be `n` device f32; `flag` one device u32, zeroed before the run.
pub unsafe fn launch_flag_nonfinite(
    x: *const f32,
    n: usize,
    tag: u32,
    flag: *mut u32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_flag_nonfinite(x, n as i32, tag, flag) };
    check(r, "flag_nonfinite")
}

/// `x += y` — the residual add. (`--moe-gain != 1` takes [`launch_vaxpy`] instead.)
///
/// # Safety
/// `x` and `y` must be device pointers to at least `n` f32.
pub unsafe fn launch_vadd(x: *mut f32, y: *const f32, n: usize) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_vadd(x, y, n as i32) };
    check(r, "vadd")
}

/// `x += g·y` — the residual add with a branch gain (see kernels/fwd.hip::vaxpy).
///
/// # Safety
/// `x` and `y` must be device pointers to at least `n` f32.
pub unsafe fn launch_vaxpy(x: *mut f32, y: *const f32, g: f32, n: usize) -> Result<()> {
    // SAFETY: caller guarantees both buffers hold `n` device f32.
    let r = unsafe { rivoli_vaxpy(x, y, g, n as i32) };
    check(r, "vaxpy")
}

/// Greedy argmax over `logits[0..n]` → (`out_idx`, `out_val`); lowest index on a
/// tie, NaN never wins (matches the host fold).
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `logits` (`n` f32),
/// `out_idx` (one i32), `out_val` (one f32).
pub unsafe fn launch_argmax(
    logits: *const f32,
    n: usize,
    out_idx: *mut i32,
    out_val: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_argmax(logits, n as i32, out_idx, out_val) };
    check(r, "argmax")
}

/// Per-row int8 GEMV `y = W·x` (lm_head → logits).
///
/// # Safety
/// Async device pointers live until the next [`device_sync`]: `x` (`i_dim` f32),
/// `packed` (`o_dim·i_dim` bytes), `scale` (`o_dim` f32), `y` (`o_dim` f32).
pub unsafe fn launch_gemv_i8(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    nrow: usize,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_i8(x, packed, scale, o_dim as i32, i_dim as i32, nrow as i32, y) };
    check(r, "gemv_i8")
}

/// f32 GEMV `y = W·x` (the MoE router gate).
///
/// # Safety
/// Device pointers live until the next [`device_sync`].
pub unsafe fn launch_gemv_f32(
    x: *const f32,
    w: *const f32,
    o_dim: usize,
    i_dim: usize,
    nrow: usize,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_f32(x, w, o_dim as i32, i_dim as i32, nrow as i32, y) };
    check(r, "gemv_f32")
}

/// SwiGLU combine `h = silu(g)·u` (dense fp8 MLP; safe in place, `h` may alias `g`).
///
/// # Safety
/// Device pointers (`g`, `u`, `h` each `n` f32) live until the next [`device_sync`].
pub unsafe fn launch_swiglu(g: *const f32, u: *const f32, n: usize, h: *mut f32) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_swiglu(g, u, n as i32, h) };
    check(r, "swiglu")
}

/// RMSNorm `y = x·rsqrt(mean(x²)+eps)·w`.
///
/// # Safety
/// Device pointers (`x`, `w`, `y` each `n` f32) live until the next [`device_sync`].
pub unsafe fn launch_rmsnorm(
    x: *const f32,
    w: *const f32,
    n: usize,
    eps: f32,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_rmsnorm(x, w, n as i32, eps, y) };
    check(r, "rmsnorm")
}

/// Interleaved RoPE in place over `count` segments of `seg` at `stride`.
///
/// # Safety
/// `base` is a device buffer of `count·stride` f32, live until the next [`device_sync`].
pub unsafe fn launch_rope(
    base: *mut f32,
    count: usize,
    stride: usize,
    seg: usize,
    pos: usize,
    theta: f64,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_rope(
            base,
            count as i32,
            stride as i32,
            seg as i32,
            pos as i32,
            theta,
        )
    };
    check(r, "rope")
}

/// Batched VQ encode (offline converter accelerator): `idx[i] = argmin_k …`.
///
/// # Safety
/// Device pointers live until the next [`device_sync`].
pub unsafe fn launch_vq_encode(
    sub: *const f32,
    codebook: *const f32,
    cbnorm: *const f32,
    n: usize,
    idx: *mut u16,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_vq_encode(sub, codebook, cbnorm, n as i32, idx) };
    check(r, "vq_encode")
}

// ── DSA lightning indexer ───────────────────────────────────────────────────────

/// LayerNorm with bias `y = (x-mean)/sqrt(var+eps)·w + b` (the indexer k_norm).
///
/// # Safety
/// Device pointers (`x`, `w`, `b`, `y` each `n` f32) live until the next [`device_sync`].
pub unsafe fn launch_layernorm(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    n: usize,
    eps: f32,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_layernorm(x, w, b, n as i32, eps, y) };
    check(r, "layernorm")
}

/// Append one indexer key row (bf16) at `pos`: `kcache[pos·hd+i] = bf16(k[i])`.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `k` (`hd` f32), `kcache`
/// (row `pos` in-bounds).
pub unsafe fn launch_index_append(
    k: *const f32,
    kcache: *mut u16,
    pos: usize,
    hd: usize,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_index_append(k, kcache, pos as i32, hd as i32) };
    check(r, "index_append")
}

/// Score every cached token against the indexer query heads:
/// `scores[t] = Σ_{h∈active} w[h]·wscale·ReLU((q_h·k_t)·dscale)`. `heads` (nullable)
/// lists the `nact` active heads (MISA); null = all `nh` heads (DSA).
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `q` (`nh·hd` f32), `w` (`nh`
/// f32), `kcache` (`nt·hd` bf16), `heads` (`nact` u32 or null), `scores` (`nt` f32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_score(
    q: *const f32,
    w: *const f32,
    kcache: *const u16,
    heads: *const u32,
    nt: usize,
    nh: usize,
    nact: usize,
    hd: usize,
    wscale: f32,
    dscale: f32,
    scores: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_index_score(
            q,
            w,
            kcache,
            heads,
            nt as i32,
            nh as i32,
            nact as i32,
            hd as i32,
            wscale,
            dscale,
            scores,
        )
    };
    check(r, "index_score")
}

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
pub unsafe fn launch_index_topk(
    scores: *const f32,
    nt: usize,
    k: usize,
    rows: *mut u32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_index_topk(scores, nt as i32, k as i32, rows) };
    check(r, "index_topk")
}

/// Fold token `t`'s indexer key into its MISA block pool running mean.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `k` (`hd` f32), `pool`
/// (block `t/MISA_BLOCK` in-bounds).
pub unsafe fn launch_index_pool_push(
    k: *const f32,
    pool: *mut f32,
    t: usize,
    hd: usize,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_index_pool_push(k, pool, t as i32, hd as i32) };
    check(r, "index_pool_push")
}

/// MISA head-router estimate `e[j] = mean_b |w[j]·ReLU(q_j·k̄_b)|` over the block pool.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `q` (`nh·hd` f32), `w` (`nh`
/// f32), `pool` (`m_blocks·hd` f32), `e` (`nh` f32).
pub unsafe fn launch_index_head_route(
    q: *const f32,
    w: *const f32,
    pool: *const f32,
    m_blocks: usize,
    nh: usize,
    hd: usize,
    e: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r =
        unsafe { rivoli_index_head_route(q, w, pool, m_blocks as i32, nh as i32, hd as i32, e) };
    check(r, "index_head_route")
}

// ── DeepSeek-V4-Flash attention (S2b) ───────────────────────────────────────────────
//
// HIP-ONLY, deliberately, and this is the one thing about them that is not obvious.
// `tests/kernel_coverage.rs::every_launcher_has_an_oracle` is keyed on `backend/vk.rs`,
// so a launcher that exists only here is invisible to it — there is no automatic gate
// saying these are exercised. `tests/v4_attn.rs` is the whole of their coverage, and
// every caller must be `#[cfg(feature = "rocm")]` or `--features vulkan` will not
// resolve the name. S3 decides whether V4 gets a Vulkan arm at all; until it does,
// adding stubs to `vk.rs` would claim a parity that does not exist.

/// `kernel.py::act_quant(x, block, "ue8m0", inplace=True)` in place over `rows` rows of
/// `row_stride` floats, quantizing the first `n` of each. `n < row_stride` is the KV
/// entry's PARTIAL quantization (model.py:512, dims `[0, head_dim - rope_head_dim)` at
/// block 64); `n == row_stride` at block 128 is what every quantized `Linear` does to
/// its activation before the GEMM.
///
/// # Safety
/// `x` is a device buffer of at least `rows * row_stride` f32, live until the next
/// [`device_sync`].
pub unsafe fn launch_v4_act_quant(
    x: *mut f32,
    rows: usize,
    row_stride: usize,
    n: usize,
    block: usize,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r =
        unsafe { rivoli_v4_act_quant(x, rows as i32, row_stride as i32, n as i32, block as i32) };
    check(r, "v4_act_quant")
}

/// `RMSNorm.forward` over `rows` rows of `d` floats, in place: f32 statistic, learned
/// weight, bf16-rounded store.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `x` (`rows * d` f32), `w` (`d`
/// f32).
pub unsafe fn launch_v4_rmsnorm(
    x: *mut f32,
    w: *const f32,
    rows: usize,
    d: usize,
    eps: f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_v4_rmsnorm(x, w, rows as i32, d as i32, eps) };
    check(r, "v4_rmsnorm")
}

/// The weightless per-head QK-norm of model.py:504, in place over `rows = s * n_heads`
/// rows of `head_dim`. Must be launched BEFORE the RoPE — see the kernel's note: the
/// oracle provably cannot see the order, so it comes from the reference.
///
/// # Safety
/// `q` is a device buffer of at least `rows * d` f32, live until the next
/// [`device_sync`].
pub unsafe fn launch_v4_qk_norm(q: *mut f32, rows: usize, d: usize, eps: f32) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_v4_qk_norm(q, rows as i32, d as i32, eps) };
    check(r, "v4_qk_norm")
}

/// `apply_rotary_emb` over the last `rd` dims of each of `rows` rows, ADJACENT-PAIR
/// (`view_as_complex`), from a precomputed `(cos, sin)` table. Row `r` takes position
/// `pos0 + r / rows_per_pos`. `inverse` conjugates it — the output de-rotation.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `x` (`rows * row_len` f32),
/// `tbl` (at least `(pos0 + rows / rows_per_pos) * rd` f32, interleaved cos/sin).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_v4_rope(
    x: *mut f32,
    tbl: *const f32,
    rows: usize,
    row_len: usize,
    rd: usize,
    pos0: usize,
    rows_per_pos: usize,
    inverse: bool,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_v4_rope(
            x,
            tbl,
            rows as i32,
            row_len as i32,
            rd as i32,
            pos0 as i32,
            rows_per_pos as i32,
            i32::from(inverse),
        )
    };
    check(r, "v4_rope")
}

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
/// Device pointers live until the next [`device_sync`]: `x` (`m * groups * k` f32), `w`
/// (`n_out * k` bytes), `wscale` (`ceil(n_out/block) * ceil(k/block)` f32), `out`
/// (`m * n_out` f32). The `x` bound is exact and follows from the arguments; an earlier
/// signature took the row and group strides separately, where the in-bounds relation was
/// a three-way inequality nothing checked.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_v4_gemv_fp8(
    x: *const f32,
    w: *const u8,
    wscale: *const f32,
    m: usize,
    n_out: usize,
    k: usize,
    block: usize,
    groups: usize,
    out: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_v4_gemv_fp8(
            x,
            w,
            wscale,
            m as i32,
            n_out as i32,
            k as i32,
            block as i32,
            groups as i32,
            out,
        )
    };
    check(r, "v4_gemv_fp8")
}

/// `kernel.py::sparse_attn` — MQA over one `d`-wide entry that is both key and value for
/// all `h` heads, gathered by `idxs` (`-1` masks a slot), with `sink` entering the
/// softmax DENOMINATOR only.
///
/// # Safety
/// Device pointers live until the next [`device_sync`]: `q` (`m * h * d` f32), `kv`
/// (`d` f32 per row, indexed by `idxs`, so at least `max(idxs) + 1` rows), `sink` (`h`
/// f32), `idxs` (`m * topk` i32), `o` (`m * h * d` f32).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_v4_sparse_attn(
    q: *const f32,
    kv: *const f32,
    sink: *const f32,
    idxs: *const i32,
    m: usize,
    h: usize,
    d: usize,
    topk: usize,
    scale: f32,
    o: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_v4_sparse_attn(
            q,
            kv,
            sink,
            idxs,
            m as i32,
            h as i32,
            d as i32,
            topk as i32,
            scale,
            o,
        )
    };
    check(r, "v4_sparse_attn")
}

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
pub unsafe fn launch_v4_dense_gemm_bf16(
    x: *const f32,
    w: *const u16,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_v4_dense_gemm_bf16(x, w, out, m as i32, n as i32, k as i32, stream) };
    check(r, "v4_dense_gemm_bf16")
}

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
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_v4_compress_state(
    kv: *const f32,
    score: *const f32,
    ape: *const f32,
    kv_state: *mut f32,
    score_state: *mut f32,
    p: &GeomAbi,
    s: usize,
    slot0: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_v4_compress_state(
            kv,
            score,
            ape,
            kv_state,
            score_state,
            p,
            s as i32,
            slot0 as i32,
            stream,
        )
    };
    check(r, "v4_compress_state")
}

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
pub unsafe fn launch_v4_compress_prefill(
    kv: *const f32,
    score: *const f32,
    ape: *const f32,
    f: &Finish,
    p: &GeomAbi,
    nblk: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_v4_compress_prefill(kv, score, ape, f, p, nblk as i32, stream) };
    check(r, "v4_compress_prefill")
}

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
pub unsafe fn launch_v4_compress_pool_decode(
    kv_state: *mut f32,
    score_state: *mut f32,
    f: &Finish,
    p: &GeomAbi,
    start_pos: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe {
        rivoli_v4_compress_pool_decode(kv_state, score_state, f, p, start_pos as i32, stream)
    };
    check(r, "v4_compress_pool_decode")
}
// jscpd:ignore-end

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
pub unsafe fn launch_v4_indexer_spread(
    x: *mut f32,
    rows: usize,
    d: usize,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_v4_indexer_spread(x, rows as i32, d as i32, stream) };
    check(r, "v4_indexer_spread")
}

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
    let ScoreDims { s, n_comp, heads, hd } = dims;
    let (s, n_comp) = (s as i32, n_comp as i32);
    let (heads, hd) = (heads as i32, hd as i32);
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_v4_indexer_score(q, kv, w, score, s, n_comp, heads, hd, stream) };
    check(r, "v4_indexer_score")
}
