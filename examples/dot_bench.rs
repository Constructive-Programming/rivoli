//! MoE dot-decode throughput microbench — int4 (dot_i4_wave) vs int3-VQ (dot_vq_wave)
//! vs fp8 (dot_fp8_wave) at the gate/up and down projection dims, isolated from the
//! routing/miss-count confound the decode bench has (there, a numerics change shifts
//! the greedy sequence → hit rate → compute bubbles). All wave-per-row (the MoE kernel
//! structure); fp8 at i_dim≥4096 dispatches to split-K (its live behaviour). Finding:
//! int4 decodes ~1.8× faster than vq3/fp8 — the all-int4 decode-bench slowdown was
//! residency (bigger experts → fewer slots → bubbles), not compute.
//! Run: cargo run --release --features rocm --example dot_bench
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]
use rivoli::device::DeviceBuf;
use rivoli::hip::{device_sync, launch_gemv_fp8, launch_gemv_i4, launch_gemv_vq};
use rivoli::math::{f32_to_e4m3, f32_to_f16};
use rivoli::quant::{matvec_i4, quant_i4, quant_vq, VQ_DIM, VQ_K};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}
fn dev(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len()).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn f16b(v: &[f32]) -> Vec<u8> {
    u16b(&v.iter().map(|&x| f32_to_f16(x)).collect::<Vec<_>>())
}
fn f32v(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn time(iters: u32, f: &dyn Fn()) -> f64 {
    f();
    device_sync().expect("s");
    let t = std::time::Instant::now();
    for _ in 0..iters {
        f();
    }
    device_sync().expect("s");
    t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0 // us/launch
}

fn run(name: &str, o_dim: usize, i_dim: usize) {
    let block = 128usize;
    let mut r = Rng(0xD07 ^ i_dim as u64);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f() * 0.1).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let xb = dev(&f32b(&x));
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    let (xp, yp) = (xb.ptr() as *const f32, yb.ptr_mut() as *mut f32);
    let iters = 300u32;
    let gelem = (o_dim * i_dim) as f64; // decode+MAC ops ≈ o·i

    // int4
    let (i4p, i4s) = quant_i4(&w, o_dim, i_dim);
    let (i4pb, i4sb) = (dev(&i4p), dev(&f32b(&i4s)));
    let us_i4 = time(iters, &|| unsafe {
        launch_gemv_i4(xp, i4pb.ptr(), i4sb.ptr() as *const f32, o_dim, i_dim, yp).expect("i4");
    });
    // correctness of gemv_i4 vs the CPU oracle (so the timing is trustworthy).
    let mut want = vec![0f32; o_dim];
    matvec_i4(&mut want, &x, &i4p, &i4s, o_dim, i_dim);
    let got = f32v(&yb.copy_out().expect("o"));
    let mx = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = want.iter().zip(&got).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let i4_ok = if err <= 1e-3 * mx + 1e-3 { "ok" } else { "MISMATCH" };

    // int3-VQ
    let cb: Vec<f32> = (0..VQ_K * VQ_DIM).map(|_| r.f()).collect();
    let (vqi, vqs) = quant_vq(&w, o_dim, i_dim, &cb);
    let (vqib, vqsb, cbb) = (dev(&vqi), dev(&u16b(&vqs)), dev(&f16b(&cb)));
    let us_vq = time(iters, &|| unsafe {
        launch_gemv_vq(xp, vqib.ptr(), vqsb.ptr() as *const u16, cbb.ptr() as *const u16, o_dim, i_dim, yp).expect("vq");
    });

    // fp8 (scale=1 blocks — decode cost is representative; accuracy irrelevant here)
    let fp8p: Vec<u8> = w.iter().map(|&v| f32_to_e4m3(v)).collect();
    let fp8s: Vec<f32> = vec![1.0; (o_dim / block) * (i_dim / block)];
    let (fp8pb, fp8sb) = (dev(&fp8p), dev(&f32b(&fp8s)));
    let us_fp8 = time(iters, &|| unsafe {
        launch_gemv_fp8(xp, fp8pb.ptr(), fp8sb.ptr() as *const f32, o_dim, i_dim, block, yp).expect("fp8");
    });

    let ge = |us: f64| gelem / (us * 1e-6) / 1e9;
    println!("{name} [{o_dim}x{i_dim}]  (gemv_i4 vs oracle: {i4_ok}, err {err:.2e}/{mx:.2})");
    println!("  int4 {us_i4:7.1}us  {:6.1} GElem/s  (1.00x)", ge(us_i4));
    println!("  vq3  {us_vq:7.1}us  {:6.1} GElem/s  ({:.2}x int4)", ge(us_vq), us_i4 / us_vq);
    println!("  fp8  {us_fp8:7.1}us  {:6.1} GElem/s  ({:.2}x int4){}", ge(us_fp8), us_i4 / us_fp8,
        if i_dim >= 4096 { "  [split-K]" } else { "" });
}

fn main() {
    println!("MoE dot decode throughput (wave-per-row, isolated):");
    run("gate/up", 2048, 6144); // hidden reduction
    run("down", 6144, 2048); // inter reduction
}
