//! Vulkan kernels vs their CPU oracles — the `tests/kernel.rs` story for the second
//! backend. Compiles to nothing without `vulkan`.
//!
//! Separate file because `tests/kernel.rs` is `#![cfg(feature = "rocm")]` end to end.
//! The helpers below are deliberately the same ones it uses, so the kernel-porting
//! phase can hoist both files onto a shared module instead of rewriting either.
#![cfg(feature = "vulkan")]
#![allow(clippy::expect_used)]

use rivoli::vk::{Buf, device_sync, launch_gemv_f32};

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f32v(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn dev(b: &[u8]) -> Buf {
    let mut d = Buf::new(b.len()).expect("alloc");
    d.write_at(0, b).expect("fill");
    d
}
fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = want
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        err <= 1e-3 * mx + 1e-3,
        "{label}: err={err:.3e} max={mx:.3e}"
    );
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// Plain f32 GEMV, ascending summation — the same shape as `quant.rs`'s matvec
/// oracles. The kernel reduces in a fixed 32-lane shuffle ladder, so it differs by
/// f32 rounding only, which is what `assert_close`'s tolerance covers.
fn matvec_f32(y: &mut [f32], x: &[f32], w: &[f32], i_dim: usize) {
    for (o, out) in y.iter_mut().enumerate() {
        let row = &w[o * i_dim..(o + 1) * i_dim];
        *out = row.iter().zip(x).map(|(a, b)| a * b).sum();
    }
}

/// One `gemv_f32` dispatch, returning the raw output bytes.
fn gemv(x: &Buf, w: &Buf, y: &mut Buf, o_dim: usize, i_dim: usize) -> Vec<u8> {
    // SAFETY: live Buf device addresses of the documented sizes; nothing is dropped
    // before the device_sync below.
    unsafe {
        launch_gemv_f32(
            x.ptr() as *const f32,
            w.ptr() as *const f32,
            o_dim,
            i_dim,
            y.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    let mut out = Vec::new();
    y.read_into(&mut out, o_dim * 4).expect("out");
    out
}

/// Greedy decode must be reproducible run to run, which is a STRICTER property than
/// the oracle's 1e-3 accuracy: a reduction whose order varies with workgroup
/// scheduling passes that tolerance happily. So compare BIT PATTERNS across repeats,
/// with a differently-shaped dispatch interleaved to perturb the scheduler.
///
/// This is why `wave_sum` is a fixed `subgroupShuffleDown` ladder and not
/// `subgroupAdd`. Every kernel with a reduction gets this test as it is ported —
/// `gemv_fp8_splitk` and `mla_attend_combine` especially, since they reduce across
/// workgroup partials in LDS rather than within one subgroup.
#[test]
fn gemv_f32_is_bit_reproducible() {
    let mut r = Lcg(0xDE7);
    let (o_dim, i_dim) = (255usize, 1024usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);

    // A decoy of a different shape, dispatched between repeats so the two runs do not
    // see identical queue state.
    let (dx, dw) = (dev(&f32b(&x[..96])), dev(&f32b(&w[..96 * 33])));
    let mut dy = dev(&[0u8; 33 * 4]);

    let first = gemv(&xb, &wb, &mut yb, o_dim, i_dim);
    for i in 1..5 {
        gemv(&dx, &dw, &mut dy, 33, 96);
        let again = gemv(&xb, &wb, &mut yb, o_dim, i_dim);
        assert_eq!(first, again, "gemv_f32 not bit-reproducible on repeat {i}");
    }
    // Guard against comparing two buffers of zeros and calling it determinism.
    assert!(
        f32v(&first).iter().any(|v| v.abs() > 1e-6),
        "output is all zero — the test proves nothing"
    );
}

#[test]
fn gemv_f32_matches_oracle() {
    let mut r = Lcg(0x3F2);
    // n_experts × hidden — the router-gate shape the kernel exists for. o_dim is
    // deliberately NOT a multiple of ROWS_PER_BLOCK, so the tail block's partly-idle
    // subgroups are exercised.
    for (o_dim, i_dim) in [(256usize, 5120usize), (255, 512), (1, 96)] {
        let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();

        let mut want = vec![0.0f32; o_dim];
        matvec_f32(&mut want, &x, &w, i_dim);

        let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
        let mut yb = dev(&vec![0u8; o_dim * 4]);
        let out = gemv(&xb, &wb, &mut yb, o_dim, i_dim);
        assert_close(&want, &f32v(&out), &format!("gemv_f32 {o_dim}x{i_dim}"));
    }
}
