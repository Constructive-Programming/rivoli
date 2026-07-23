//! Minimal HIP surface: under `rocm` this binds the hipcc-built kernel launchers.
//! Without the feature the whole module compiles away. int4-free; grows as kernels
//! land (linalg now; moe/mla/attn/fwd next).

#![cfg(feature = "rocm")]

use anyhow::{Result, bail};

unsafe extern "C" {
    fn rivoli_device_sync() -> i32;

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
