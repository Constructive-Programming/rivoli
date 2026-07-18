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
/// `x`/`out`/`wexpert` must be valid device pointers for `hidden`/`hidden`/`e`
/// elements; `descs` must point at `e` valid `ExpertDesc` in device memory whose
/// weight pointers cover the projection shapes; all must outlive the launch.
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

/// MLA flash attention over the compressed KV cache: for each head, returns the
/// attention-weighted latent `clat_h = Σ_t softmax((qabs_h·L_t + qrope_h·R_t)·
/// scale)·L_t` as `H` contiguous `kvl`-length rows. `lc`/`rc` are the bf16 cache
/// (`nt` rows of `kvl`/`rope`). Query-side absorb through `kv_b` and the value
/// projection back stay on the reference path for M2; this validates the
/// long-context score/softmax/weighted-sum kernel against the scalar oracle.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn mla_attend(
    qabs: &[f32],
    qrope: &[f32],
    lc: &[u16],
    rc: &[u16],
    h: usize,
    nt: usize,
    kvl: usize,
    rope: usize,
    scale: f32,
) -> Result<Vec<f32>> {
    // Dims cross the FFI as i32; guard the truncating cast (see moe_experts).
    anyhow::ensure!(
        h <= i32::MAX as usize
            && nt <= i32::MAX as usize
            && kvl <= i32::MAX as usize
            && rope <= i32::MAX as usize,
        "attn dims exceed i32 (h={h} nt={nt} kvl={kvl} rope={rope})"
    );
    anyhow::ensure!(
        qabs.len() == h * kvl,
        "qabs len {} != h*kvl {}",
        qabs.len(),
        h * kvl
    );
    anyhow::ensure!(
        qrope.len() == h * rope,
        "qrope len {} != h*rope {}",
        qrope.len(),
        h * rope
    );
    anyhow::ensure!(
        lc.len() == nt * kvl,
        "lc len {} != nt*kvl {}",
        lc.len(),
        nt * kvl
    );
    anyhow::ensure!(
        rc.len() == nt * rope,
        "rc len {} != nt*rope {}",
        rc.len(),
        nt * rope
    );
    let mut clat = vec![0.0f32; h * kvl];
    // SAFETY: every pointer is valid for the length asserted above; the C
    // launcher copies inputs in, launches, writes exactly h*kvl floats into
    // `clat`, and frees its device allocations before returning.
    let r = unsafe {
        rivoli_mla_attend(
            qabs.as_ptr(),
            qrope.as_ptr(),
            lc.as_ptr(),
            rc.as_ptr(),
            h as i32,
            nt as i32,
            kvl as i32,
            rope as i32,
            scale,
            clat.as_mut_ptr(),
        )
    };
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("rivoli_mla_attend failed ({kind}, code {r})");
    }
    Ok(clat)
}
