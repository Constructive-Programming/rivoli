//! Minimal HIP surface: under `rocm` this binds the hipcc-built kernel launchers.
//! Without the feature the whole module compiles away. int4-free; grows as kernels
//! land (linalg now; moe/mla/attn/fwd next).

#![cfg(feature = "rocm")]

use anyhow::{Result, bail};

/// VQ-int3 expert descriptor (mirrors `struct ExpertDescVq` in moe.hip): per
/// projection the packed 12-bit indices + bf16 group scales. The 3 per-projection
/// codebooks are shared across all experts and passed to [`launch_moe_vq`].
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

unsafe extern "C" {
    fn rivoli_device_sync() -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_experts_vq(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e: i32,
        descs: *const ExpertDescVq,
        gate_cb: *const f32,
        up_cb: *const f32,
        down_cb: *const f32,
        wexpert: *const f32,
        h: *mut f32,
        partial: *mut f32,
        out: *mut f32,
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
        codebook: *const f32,
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
        lc: *const u16,
        rc: *const u16,
        h: i32,
        nr: i32,
        kvl: i32,
        rope: i32,
        scale: f32,
        clat: *mut f32,
        partial: *mut f32,
    ) -> i32;

    fn rivoli_gemv_f32(x: *const f32, w: *const f32, o_dim: i32, i_dim: i32, y: *mut f32) -> i32;
    fn rivoli_rmsnorm(x: *const f32, w: *const f32, n: i32, eps: f32, y: *mut f32) -> i32;
    fn rivoli_rope(base: *mut f32, count: i32, stride: i32, seg: i32, pos: i32, theta: f64) -> i32;
    fn rivoli_vq_encode(
        sub: *const f32,
        codebook: *const f32,
        cbnorm: *const f32,
        n: i32,
        idx: *mut u16,
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

/// Launch a fused VQ-int3 MoE expert batch: `out += Σ_e w[e]·down(silu(gate·x)⊙up·x)`.
/// gate/up/down decode against `gate_cb`/`up_cb`/`down_cb`. Routed + shared experts
/// fold into one call (shared appended, weight 1.0).
///
/// # Safety
/// Async, device pointers live until the next [`device_sync`]. `descs` ≥ `e`
/// `ExpertDescVq`; `wexpert` `e` f32; `h` ≥ `e·inter` f32; `partial` `e·hidden` f32;
/// `out` `hidden` f32; each codebook `VQ_K·VQ_DIM` f32.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_vq(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e: usize,
    descs: *const ExpertDescVq,
    gate_cb: *const f32,
    up_cb: *const f32,
    down_cb: *const f32,
    wexpert: *const f32,
    h: *mut f32,
    partial: *mut f32,
    out: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_moe_experts_vq(
            x,
            hidden as i32,
            inter as i32,
            e as i32,
            descs,
            gate_cb,
            up_cb,
            down_cb,
            wexpert,
            h,
            partial,
            out,
        )
    };
    check(r, "moe_vq")
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
/// the bf16 latent cache, head-batched, split-KV when `partial` is non-null.
///
/// # Safety
/// Async device pointers live until the next [`device_sync`]: `qabs` (`h·kvl` f32),
/// `qrope` (`h·rope` f32), `lc` (`nr·kvl` bf16), `rc` (`nr·rope` bf16), `clat`
/// (`h·kvl` f32), `partial` ([`attend_scratch_floats`] f32 or null = single split).
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_attend(
    qabs: *const f32,
    qrope: *const f32,
    lc: *const u16,
    rc: *const u16,
    h: usize,
    nr: usize,
    kvl: usize,
    rope: usize,
    scale: f32,
    clat: *mut f32,
    partial: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe {
        rivoli_mla_attend(
            qabs, qrope, lc, rc, h as i32, nr as i32, kvl as i32, rope as i32, scale, clat, partial,
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
    codebook: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    // SAFETY: caller's pointer contract.
    let r = unsafe { rivoli_gemv_vq(x, indices, scales, codebook, o_dim as i32, i_dim as i32, y) };
    check(r, "gemv_vq")
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
