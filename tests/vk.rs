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
        // SAFETY: every pointer is a live Buf device address of the size the launcher
        // documents, and nothing is dropped before the device_sync below.
        unsafe {
            launch_gemv_f32(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                o_dim,
                i_dim,
                yb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        let mut out = Vec::new();
        yb.read_into(&mut out, o_dim * 4).expect("out");
        assert_close(&want, &f32v(&out), &format!("gemv_f32 {o_dim}x{i_dim}"));
    }
}
