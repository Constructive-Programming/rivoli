//! Minimal HIP surface: under `rocm` this binds the hipcc-built kernel launchers
//! (fp8/int8/f32 linalg, VQ-int3 and int4 MoE, MLA, fwd glue). Without the feature
//! the whole module compiles away.

#![cfg(feature = "rocm")]

use anyhow::{Result, bail};
use std::ffi::c_void;

/// VQ-int3 expert descriptor (mirrors `struct ExpertDescVq` in moe.hip): per
/// projection the packed 12-bit indices + bf16 group scales. The 3 per-projection
/// codebooks are shared across all experts and passed to [`launch_moe_expert_range`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDescVq {
    pub gate_indices: *const u8,
    pub gate_scales: *const u16,
    pub up_indices: *const u8,
    pub up_scales: *const u16,
    pub down_indices: *const u8,
    pub down_scales: *const u16,
}

/// int4 expert descriptor (mirrors `struct ExpertDescI4` in moe.hip): per projection
/// the packed 4-bit weights + a per-output-row f32 scale (colibri's `.qs`). The
/// "warm expert" format — passed to [`launch_moe_expert_range_i4`].
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDescI4 {
    pub gate_packed: *const u8,
    pub gate_scale: *const f32,
    pub up_packed: *const u8,
    pub up_scale: *const f32,
    pub down_packed: *const u8,
    pub down_scale: *const f32,
}

unsafe extern "C" {
    fn rivoli_device_sync() -> i32;
    fn rivoli_memcpy_dtod(dst: *mut u8, src: *const u8, bytes: usize) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_expert_range_i4(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e_start: i32,
        e_count: i32,
        descs: *const ExpertDescI4,
        wexpert: *const f32,
        h: *mut f32,
        partial: *mut f32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_expert_range(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e_start: i32,
        e_count: i32,
        descs: *const ExpertDescVq,
        gate_cb: *const u16,
        up_cb: *const u16,
        down_cb: *const u16,
        wexpert: *const f32,
        h: *mut f32,
        partial: *mut f32,
        stream: *mut c_void,
    ) -> i32;

    fn rivoli_moe_reduce_s(
        partial: *const f32,
        e: i32,
        hidden: i32,
        out: *mut f32,
        stream: *mut c_void,
    ) -> i32;

    fn rivoli_gemv_fp8(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        block: i32,
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
    fn rivoli_argmax(logits: *const f32, n: i32, out_idx: *mut i32, out_val: *mut f32) -> i32;

    fn rivoli_gemv_i8(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;
    fn rivoli_gemv_f32(x: *const f32, w: *const f32, o_dim: i32, i_dim: i32, y: *mut f32) -> i32;
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
}

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
    check(unsafe { rivoli_memcpy_dtod(dst, src, bytes) }, "memcpy_dtod")
}

/// Streaming MoE: gate/up + down for the absolute expert range `[e_start,
/// e_start+e_count)` on `stream`, writing each expert's own `h`/`partial` rows.
/// Reduce with [`launch_moe_reduce`] once every range's partials have landed.
///
/// # Safety
/// Every device pointer (`descs`/codebooks/`wexpert`/`x`/`h`/`partial`) must outlive
/// `stream`'s completion — await its [`Signal`](crate::gpustream::Signal).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_expert_range(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e_start: usize,
    e_count: usize,
    descs: *const ExpertDescVq,
    gate_cb: *const u16,
    up_cb: *const u16,
    down_cb: *const u16,
    wexpert: *const f32,
    h: *mut f32,
    partial: *mut f32,
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
            partial,
            stream,
        )
    };
    check(r, "moe_expert_range")
}

/// int4 counterpart of [`launch_moe_expert_range`]: gate/up + down for the absolute
/// range `[e_start, e_start+e_count)` on `stream`, decoding int4 (per-row scale).
/// `descs` are [`ExpertDescI4`]; partials land in the same `partial` slab and are
/// summed by [`launch_moe_reduce`], so int4 experts share the VQ reduce.
///
/// # Safety
/// Every device pointer (`descs`/packed weights/`wexpert`/`x`/`h`/`partial`) must
/// outlive `stream`'s completion — await its [`Signal`](crate::gpustream::Signal).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_expert_range_i4(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e_start: usize,
    e_count: usize,
    descs: *const ExpertDescI4,
    wexpert: *const f32,
    h: *mut f32,
    partial: *mut f32,
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
            partial,
            stream,
        )
    };
    check(r, "moe_expert_range_i4")
}

/// Fixed-order reduce `out[o] = Σ_e partial[e][o]` over all `e` experts, on `stream`.
///
/// # Safety
/// `partial` holds `e·hidden` f32 (all ranges landed); `out` holds `hidden` f32.
pub unsafe fn launch_moe_reduce(
    partial: *const f32,
    e: usize,
    hidden: usize,
    out: *mut f32,
    stream: *mut c_void,
) -> Result<()> {
    // SAFETY: caller's pointer contract; stream is a live HipStream handle.
    let r = unsafe { rivoli_moe_reduce_s(partial, e as i32, hidden as i32, out, stream) };
    check(r, "moe_reduce")
}

/// fp8-e4m3 block-scaled GEMV `y = W·x` (attention/dense projections).
///
/// # Safety
/// Async device pointers live until the next [`device_sync`]: `x` (`i_dim` f32),
/// `packed` (`o_dim·i_dim` bytes), `scale` (block-scale f32), `y` (`o_dim` f32).
pub unsafe fn launch_gemv_fp8(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    block: usize,
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

/// Per-row int4 GEMV `y[o] = scale[o]·Σ x·(nibble-8)` — the MoE `dot_i4_wave`
/// wave-per-row, for the dot-throughput microbench.
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

/// Residual add `x[i] += y[i]`.
///
/// # Safety
/// Device pointers `x`, `y` (each `n` f32) live until the next [`device_sync`].
pub unsafe fn launch_vadd(x: *mut f32, y: *const f32, n: usize) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_vadd(x, y, n as i32) };
    check(r, "vadd")
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
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_i8(x, packed, scale, o_dim as i32, i_dim as i32, y) };
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
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_f32(x, w, o_dim as i32, i_dim as i32, y) };
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
