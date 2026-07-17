//! Minimal HIP surface. Under the `rocm` feature this binds the hipcc-built
//! kernel launchers; without it, the calls return a "not built" error so the
//! single-engine contract (zero launches = hard error) is visible even in a
//! CPU-only dev build rather than silently pretending success.

use anyhow::{Result, bail};

#[cfg(feature = "rocm")]
unsafe extern "C" {
    fn rivoli_probe(n: i32) -> i32;
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
