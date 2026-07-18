//! Minimal HIP surface. Under the `rocm` feature this binds the hipcc-built
//! kernel launchers; without it, the calls return a "not built" error so the
//! single-engine contract (zero launches = hard error) is visible even in a
//! CPU-only dev build rather than silently pretending success.

use anyhow::{Result, bail};

/// One MoE expert's six weight tensors, as raw DEVICE pointers into the resident
/// tier (or a cold-fetch slab). `repr(C)` matching `struct ExpertDesc` in
/// moe_fused.hip — the kernel reads a `[E]` array of these so experts may live
/// at arbitrary device offsets (no contiguous-stride assumption; PLAN.md D1).
#[cfg(feature = "rocm")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDesc {
    pub gate_packed: *const u8,
    pub gate_scale: *const f32,
    pub up_packed: *const u8,
    pub up_scale: *const f32,
    pub down_packed: *const u8,
    pub down_scale: *const f32,
}

#[cfg(feature = "rocm")]
unsafe extern "C" {
    fn rivoli_probe(n: i32) -> i32;

    fn rivoli_moe_experts(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e: i32,
        descs: *const ExpertDesc,
        wexpert: *const f32,
        out: *mut f32,
    ) -> i32;

    fn rivoli_device_sync() -> i32;

    fn rivoli_gemv_i4(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;

    fn rivoli_gemv_i8(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;

    fn rivoli_rmsnorm(x: *const f32, w: *const f32, n: i32, eps: f32, y: *mut f32) -> i32;

    fn rivoli_rope(base: *mut f32, count: i32, stride: i32, seg: i32, pos: i32, theta: f64) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_attend(
        qabs: *const f32,
        qrope: *const f32,
        lc: *const u16,
        rc: *const u16,
        h: i32,
        nt: i32,
        kvl: i32,
        rope: i32,
        scale: f32,
        clat: *mut f32,
    ) -> i32;
}

/// Launch MLA flash attention over the resident KV cache: for each head, writes
/// the attention-weighted latent `clat_h = Σ_t softmax((qabs_h·L_t + qrope_h·R_t)
/// ·scale)·L_t` as `H` contiguous `kvl`-length rows. All arguments are DEVICE
/// pointers — `qabs`/`qrope` (`h*kvl`/`h*rope` f32), the bf16 cache `lc`/`rc`
/// (`nt*kvl`/`nt*rope` u16), and `clat` (`h*kvl` f32, fully written). Does NOT
/// synchronize — call [`device_sync`] once per token.
///
/// # Safety
/// The launch is ASYNCHRONOUS: the kernel reads/writes the pointers below AFTER
/// this call returns, so all must stay valid until the next [`device_sync`]
/// RETURNS. Shapes (all device pointers in the current HIP context): `qabs`
/// `h*kvl` f32, `qrope` `h*rope` f32, `lc` `nt*kvl` u16, `rc` `nt*rope` u16,
/// `clat` `h*kvl` f32.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_attend(
    qabs: *const f32,
    qrope: *const f32,
    lc: *const u16,
    rc: *const u16,
    h: usize,
    nt: usize,
    kvl: usize,
    rope: usize,
    scale: f32,
    clat: *mut f32,
) -> Result<()> {
    anyhow::ensure!(
        h <= i32::MAX as usize
            && nt <= i32::MAX as usize
            && kvl <= i32::MAX as usize
            && rope <= i32::MAX as usize,
        "attn dims exceed i32 (h={h} nt={nt} kvl={kvl} rope={rope})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_mla_attend(
            qabs,
            qrope,
            lc,
            rc,
            h as i32,
            nt as i32,
            kvl as i32,
            rope as i32,
            scale,
            clat,
        )
    };
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("launch_attend failed ({kind}, code {r})");
    }
    Ok(())
}

/// Liveness probe: launch the axpy kernel and confirm the device computed the
/// expected value. Returns Ok(()) only if a real launch reached the GPU.
pub fn probe() -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        // SAFETY: FFI to the hipcc-built launcher; it owns its own device
        // allocations and frees them before returning.
        let r = unsafe { rivoli_probe(4096) };
        if r == 2 {
            Ok(())
        } else {
            bail!("HIP probe returned {r} (expected 2) — GPU launch failed")
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        bail!("built without the `rocm` feature — no GPU engine compiled in")
    }
}

/// Launch one fused MoE expert-batch over resident device memory: every expert
/// `e` shares `x` and the `(hidden, inter)` shape, accumulating
/// `out += Σ_e w[e] · down_e(silu(gate_e·x) ⊙ up_e·x)`. All arguments are DEVICE
/// pointers — `x`, `out`, the `[e]` weight array `wexpert`, the `[e]` descriptor
/// array `descs`, and the per-expert weights the descriptors point at. `out`
/// must be pre-zeroed (the kernel accumulates). Does NOT synchronize — call
/// [`device_sync`] once per token (≤1 join/token).
///
/// # Safety
/// The launch is ASYNCHRONOUS: the kernel dereferences every pointer below
/// AFTER this call returns, so all of them — and the device buffers the
/// `ExpertDesc` weight pointers address — must stay valid until the next
/// [`device_sync`] RETURNS. Dropping any of them before that join is a GPU
/// use-after-free. Shapes the kernel assumes (all device pointers in the current
/// HIP context):
///   - `x`: `hidden` f32; `out`: `hidden` f32, pre-zeroed; `wexpert`: `e` f32
///   - `descs`: ≥ `e` contiguous `ExpertDesc`, each pointing at
///     - `gate_packed`, `up_packed`: `inter * ((hidden+1)/2)` bytes
///     - `gate_scale`, `up_scale`: `inter` f32
///     - `down_packed`: `hidden * ((inter+1)/2)` bytes
///     - `down_scale`: `hidden` f32
#[cfg(feature = "rocm")]
pub unsafe fn launch_moe(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e: usize,
    descs: *const ExpertDesc,
    wexpert: *const f32,
    out: *mut f32,
) -> Result<()> {
    anyhow::ensure!(
        hidden <= i32::MAX as usize && inter <= i32::MAX as usize && e <= i32::MAX as usize,
        "moe dims exceed i32 (hidden={hidden} inter={inter} e={e})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity.
    let r = unsafe {
        rivoli_moe_experts(
            x,
            hidden as i32,
            inter as i32,
            e as i32,
            descs,
            wexpert,
            out,
        )
    };
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("launch_moe failed ({kind}, code {r})");
    }
    Ok(())
}

/// Launch a batch-1 int4 GEMV `y = scale ⊙ (W·x)` over resident device memory
/// (`W` = `packed` per-row nibbles + per-row `scale`, `o_dim × i_dim`). Device
/// pointers, launch-only. The workhorse for the attention projections
/// (q_a/q_b/kv_a/o_proj) that must stay on-device to avoid per-layer joins.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `packed` (`o_dim·⌈i_dim/2⌉` bytes), `scale`
/// (`o_dim` f32), `y` (`o_dim` f32) must be valid device pointers that stay live
/// until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_i4(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    anyhow::ensure!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_i4 dims exceed i32 (o={o_dim} i={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_i4(x, packed, scale, o_dim as i32, i_dim as i32, y) };
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("launch_gemv_i4 failed ({kind}, code {r})");
    }
    Ok(())
}

/// Launch a batch-1 int8 GEMV `y = scale ⊙ (W·x)` (`W` = `packed` signed bytes +
/// per-row `scale`, `o_dim × i_dim`) — the lm_head projection. Device pointers,
/// launch-only.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `packed` (`o_dim·i_dim` bytes), `scale` (`o_dim`
/// f32), `y` (`o_dim` f32) must be valid device pointers live until the next
/// [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_i8(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    anyhow::ensure!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_i8 dims exceed i32 (o={o_dim} i={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_i8(x, packed, scale, o_dim as i32, i_dim as i32, y) };
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("launch_gemv_i8 failed ({kind}, code {r})");
    }
    Ok(())
}

/// Launch RMSNorm `y = x·(1/√(mean(x²)+eps))·w` over a device vector of `n` f32.
///
/// # Safety
/// Async — `x`/`w`/`y` (each `n` f32 device pointers) must stay valid until the
/// next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_rmsnorm(
    x: *const f32,
    w: *const f32,
    n: usize,
    eps: f32,
    y: *mut f32,
) -> Result<()> {
    anyhow::ensure!(n <= i32::MAX as usize, "rmsnorm n exceeds i32 ({n})");
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_rmsnorm(x, w, n as i32, eps, y) };
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("launch_rmsnorm failed ({kind}, code {r})");
    }
    Ok(())
}

/// Launch interleaved RoPE at position `pos` on `count` device segments of `seg`
/// f32 each, segment `s` at `base[s*stride ..]` (matching attn.rs). Used for the
/// KV key (count=1) and the per-head query rope segments (count=H, stride=qh).
///
/// # Safety
/// Async — `base` must cover `count*stride` f32 and stay valid until the next
/// [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_rope(
    base: *mut f32,
    count: usize,
    stride: usize,
    seg: usize,
    pos: usize,
    theta: f64,
) -> Result<()> {
    anyhow::ensure!(
        count <= i32::MAX as usize
            && stride <= i32::MAX as usize
            && seg <= i32::MAX as usize
            && pos <= i32::MAX as usize,
        "rope dims exceed i32 (count={count} stride={stride} seg={seg} pos={pos})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
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
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("launch_rope failed ({kind}, code {r})");
    }
    Ok(())
}

/// Block until all launched kernels retire (one join point per token), surfacing
/// any async execution fault.
#[cfg(feature = "rocm")]
pub fn device_sync() -> Result<()> {
    // SAFETY: no arguments; hipDeviceSynchronize is always safe to call.
    let r = unsafe { rivoli_device_sync() };
    if r != 0 {
        bail!("device_sync failed (HIP runtime, code {r})");
    }
    Ok(())
}
