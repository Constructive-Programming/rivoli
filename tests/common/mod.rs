//! The backend-neutral half of the kernel-oracle scaffolding, shared by `tests/vk.rs`,
//! `tests/kernel.rs`, `tests/docs.rs` and `tests/invariants.rs`.
//!
//! It was copy-pasted per file until 2026-07-30, and the copies had already started to
//! drift: two spellings of the same `Lcg` bug note, two `assert_close` bodies with the
//! same tolerance, and `f16b`/`u16b` present in one file and re-derived in the other.
//! Anything that touches a device TYPE stays in the test file that owns it — `dev` is
//! `DeviceBuf` under HIP and `Buf` under Vulkan, and that difference is the point of
//! having two files.
//!
//! `dead_code` is allowed because this module is compiled into EACH test binary, and
//! neither uses every helper. The alternative is per-consumer cfg gates on a test
//! utility, which is more machinery than the warning is worth.
#![allow(dead_code)]
#![allow(clippy::expect_used)]

/// Every file under `root` with extension `ext`, recursively. Unsorted.
///
/// WALK, do not list files. The two registry checks that call this had each grown their
/// own copy — `docs.rs` recursive, `invariants.rs` an explicit stack — and both exist for
/// the same reason: the hand-maintained path list `invariants.rs` replaced named five
/// files, and moving `hybrid`/`gpustream`/`pin` into subsystem folders on 2026-07-31
/// silently emptied it, after which the registry reported every INV-n as untested. A
/// coverage check keyed on a remembered list fails in the direction that looks like a real
/// regression, which costs more than the walk.
pub fn walk(root: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = e.path();
            match p.is_dir() {
                true => stack.push(p),
                false if p.extension().is_some_and(|x| x == ext) => out.push(p),
                false => {}
            }
        }
    }
    out
}

/// f32 slice → little-endian bytes, the form every device upload takes.
pub fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u16 slice → little-endian bytes (bf16 scales, fp16 codebooks, roped keys).
pub fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// f32 → fp16 bytes — the VQ codebook is uploaded fp16 (the kernel decodes `__half`),
/// while the CPU reference keeps the f32 codebook, so these oracles measure exactly the
/// fp16 codebook-rounding error against the tol.
pub fn f16b(v: &[f32]) -> Vec<u8> {
    u16b(&v.iter().map(|&x| rivoli::math::f32_to_f16(x)).collect::<Vec<_>>())
}

/// Little-endian bytes → f32 vec, the inverse of [`f32b`] for readback.
pub fn f32v(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Report the max error AND the threshold it was compared against. Printing the MARGIN
/// is the point: a green oracle that passed on 100x of headroom looks exactly like one
/// that passed on 2x, and only one of them is evidence of anything.
pub fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = err_tol(want, got);
    let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    println!(
        "{label}: err={err:.3e} tol={tol:.3e} margin={:.1}x",
        tol / err.max(f32::MIN_POSITIVE)
    );
    assert!(err <= tol, "{label}: err={err:.3e} > tol={tol:.3e} max={mx:.3e}");
}

/// `(max abs error, tolerance)` for a want/got pair — the shared arithmetic behind
/// [`assert_close`] and `vk.rs`'s multi-shape `Shapes::close`, which records instead of
/// panicking. Two copies of a tolerance formula is two tolerances.
pub fn err_tol(want: &[f32], got: &[f32]) -> (f32, f32) {
    let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = want
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    (err, 1e-3 * mx + 1e-3)
}

/// The oracles' deterministic input source.
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in [-1, 1).
    ///
    /// `>> 32`, not `>> 33`. The old shift kept only 31 bits, so dividing by `u32::MAX`
    /// gave [0, 0.5) and `*2 - 1` gave [-1, 0) — **every sample negative**, for the whole
    /// life of both test files. In a matvec oracle that makes every `x[i]*w[i]` product
    /// positive, so the partial sums GROW instead of cancelling: `mx` inflates, the
    /// `1e-3 * mx` relative tolerance inflates with it, and the oracles were passing on
    /// roughly two orders of magnitude of headroom. It also meant no oracle here had ever
    /// exercised floating-point cancellation — the only regime where summation order
    /// matters, and the entire reason the kernels reduce with a fixed shuffle ladder
    /// (`__shfl_down` / `wave_sum`) instead of an atomic.
    pub fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
