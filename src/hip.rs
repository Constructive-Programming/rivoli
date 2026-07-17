//! Minimal HIP surface. Under the `rocm` feature this binds the hipcc-built
//! kernel launchers; without it, the calls return a "not built" error so the
//! single-engine contract (zero launches = hard error) is visible even in a
//! CPU-only dev build rather than silently pretending success.

use anyhow::{Result, bail};

#[cfg(feature = "rocm")]
unsafe extern "C" {
    fn rivoli_probe(n: i32) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_moe_experts(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e: i32,
        gate_packed: *const u8,
        gate_scale: *const f32,
        up_packed: *const u8,
        up_scale: *const f32,
        down_packed: *const u8,
        down_scale: *const f32,
        wexpert: *const f32,
        out: *mut f32,
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

/// One fused MoE expert-batch on the GPU: every expert `e` shares the input `x`
/// and the `(hidden, inter)` shape, and the batch is computed in a single kernel
/// launch as `out = Σ_e w[e] · down_e(silu(gate_e·x) ⊙ up_e·x)`. Returns the
/// batch's `hidden`-length contribution (the caller adds it to the residual;
/// routed and shared experts are separate batches because their `inter` differ).
///
/// M2 correctness path — copies weights in per call. The resident device tier
/// and zero-copy feed are M3; this validates the kernel numerics against the
/// scalar oracle first.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub fn moe_experts(
    x: &[f32],
    hidden: usize,
    inter: usize,
    e: usize,
    gate_packed: &[u8],
    gate_scale: &[f32],
    up_packed: &[u8],
    up_scale: &[f32],
    down_packed: &[u8],
    down_scale: &[f32],
    wexpert: &[f32],
) -> Result<Vec<f32>> {
    let rb_h = hidden.div_ceil(2);
    let rb_i = inter.div_ceil(2);
    anyhow::ensure!(x.len() == hidden, "x len {} != hidden {hidden}", x.len());
    anyhow::ensure!(wexpert.len() == e, "wexpert len {} != e {e}", wexpert.len());
    let gate_bytes = e * inter * rb_h;
    anyhow::ensure!(
        gate_packed.len() == gate_bytes && up_packed.len() == gate_bytes,
        "gate/up packed len ({}, {}) != e*inter*rb_h {gate_bytes}",
        gate_packed.len(),
        up_packed.len()
    );
    anyhow::ensure!(
        down_packed.len() == e * hidden * rb_i,
        "down packed len {} != e*hidden*rb_i {}",
        down_packed.len(),
        e * hidden * rb_i
    );
    anyhow::ensure!(
        gate_scale.len() == e * inter && up_scale.len() == e * inter,
        "gate/up scale len ({}, {}) != e*inter {}",
        gate_scale.len(),
        up_scale.len(),
        e * inter
    );
    anyhow::ensure!(
        down_scale.len() == e * hidden,
        "down scale len {} != e*hidden {}",
        down_scale.len(),
        e * hidden
    );

    let mut out = vec![0.0f32; hidden];
    // SAFETY: every pointer is valid for the length asserted above; the C
    // launcher copies inputs in, launches, writes exactly `hidden` floats into
    // `out`, and frees its device allocations before returning.
    let r = unsafe {
        rivoli_moe_experts(
            x.as_ptr(),
            hidden as i32,
            inter as i32,
            e as i32,
            gate_packed.as_ptr(),
            gate_scale.as_ptr(),
            up_packed.as_ptr(),
            up_scale.as_ptr(),
            down_packed.as_ptr(),
            down_scale.as_ptr(),
            wexpert.as_ptr(),
            out.as_mut_ptr(),
        )
    };
    if r != 0 {
        bail!("rivoli_moe_experts failed (code {r})");
    }
    Ok(out)
}
