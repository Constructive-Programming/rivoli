//! MoE dot-decode throughput microbench — int4 (dot_i4_wave) vs int3-VQ (dot_vq_wave)
//! vs fp8 (dot_fp8_wave) at the gate/up and down projection dims, isolated from the
//! routing/miss-count confound the decode bench has (there, a numerics change shifts
//! the greedy sequence → hit rate → compute bubbles). All wave-per-row (the MoE kernel
//! structure); fp8 at i_dim≥4096 dispatches to split-K (its live behaviour). Finding:
//! int4 decodes ~1.8× faster than vq3/fp8 — the all-int4 decode-bench slowdown was
//! residency (bigger experts → fewer slots → bubbles), not compute.
//! A second section adds the two `docs/PERF.md` per-kernel targets (o_proj, lm_head) at
//! their real engine shapes in GB/s. The MoE rows above are untouched, so numbers already
//! recorded in benchmarks.md stay comparable.
//! Run: cargo run --release --features rocm --example dot_bench
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]
use rivoli::device::DeviceBuf;
use rivoli::hip::{
    attend_scratch_floats, device_sync, launch_argmax, launch_attend, launch_gemv_fp8,
    launch_gemv_i4, launch_gemv_i8, launch_gemv_vq, launch_mla_absorb_fp8, launch_mla_value_fp8,
    launch_rmsnorm,
};
use rivoli::math::{f32_to_e4m3, f32_to_f16};
use rivoli::quant::{matvec_i4, quant_i4, quant_vq, VQ_DIM, VQ_K};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        // `>> 32`, not `>> 33`. Shifting 33 keeps 31 bits, so the quotient lands in
        // [0, 0.5) and EVERY sample is negative — the identical defect commit 01b3de9
        // fixed in tests/kernel.rs::Lcg, which had survived here. It matters for the
        // `run` rows: all-negative weights and inputs make every product positive, so
        // the i4 oracle check below never exercises cancellation, and the VQ codebook
        // indices come from half the distribution. The MoE rows recorded in
        // benchmarks.md predate this fix and are NOT comparable to rows measured after.
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
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

/// `n` bytes from a repeated 4 KiB pattern. These two shapes are BANDWIDTH measurements
/// — the kernel reads every weight byte exactly once regardless of its value — so
/// generating o·i real quantized weights (a 3.8 GB host `Vec<f32>` at the lm_head shape,
/// plus a CPU quantize) would cost minutes to measure nothing extra. Correctness for
/// these kernels lives in tests/kernel.rs, against the CPU oracles. The pattern VARIES
/// rather than being a constant fill because the fp8 path decodes through an LDS LUT: a
/// constant byte would make every lane hit one LDS address, turning a scattered read
/// into a broadcast and timing something the real kernel never does.
fn pattern(n: usize, mut byte: impl FnMut(f32) -> u8) -> Vec<u8> {
    let mut r = Rng(0xB07);
    let p: Vec<u8> = (0..4096).map(|_| byte(r.f())).collect();
    // `repeat` preallocates; `cycle().take(n).collect()` has no size hint and would
    // realloc its way to 951 MB at the lm_head shape.
    let mut v = p.repeat(n.div_ceil(p.len()));
    v.truncate(n);
    v
}

fn report(name: &str, kind: &str, o_dim: usize, i_dim: usize, us: f64) {
    let gbs = (o_dim * i_dim) as f64 / (us * 1e-6) / 1e9;
    println!("{name} {kind} [{o_dim}x{i_dim}]  {us:8.1}us  {gbs:6.1} GB/s  ({:.0}% of 256)", gbs / 2.56);
}

/// The two BIG real shapes, throughput only. Separate from `run` because `run`
/// materializes an `o·i` f32 host buffer and CPU-quantizes it three ways — at
/// [154880,6144] the VQ argmin alone would never finish. Reported as GB/s of weight
/// traffic against the 256 GB/s LPDDR5 peak, which is the number these kernels are
/// actually bounded by.
///
/// THROUGHPUT ONLY. `x` is a constant vector — fine here, because addresses and traffic
/// are identical to a varying one — but it means these rows cannot surface any
/// value-dependent effect. A clean number here is NOT evidence about numerics; that is
/// what the oracles in tests/kernel.rs and the perplexity harness are for.
fn run_fp8(name: &str, o_dim: usize, i_dim: usize) {
    let block = 128usize;
    // |v| <= 0.1 never encodes to the e4m3 NaN pattern, so no fixup is needed.
    let packed = pattern(o_dim * i_dim, |v| f32_to_e4m3(v * 0.1));
    let scale: Vec<f32> = vec![1.0; (o_dim / block) * (i_dim / block)];
    let x: Vec<f32> = vec![0.5; i_dim];
    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    let (xp, yp) = (xb.ptr() as *const f32, yb.ptr_mut() as *mut f32);
    let us = time(60, &|| unsafe {
        launch_gemv_fp8(xp, pb.ptr(), sb.ptr() as *const f32, o_dim, i_dim, block, yp).expect("fp8");
    });
    report(name, "fp8", o_dim, i_dim, us);
}

fn run_i8(name: &str, o_dim: usize, i_dim: usize) {
    let packed = pattern(o_dim * i_dim, |v| (v * 127.0) as i8 as u8);
    let scale: Vec<f32> = vec![1.0; o_dim];
    let x: Vec<f32> = vec![0.5; i_dim];
    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    let (xp, yp) = (xb.ptr() as *const f32, yb.ptr_mut() as *mut f32);
    let us = time(60, &|| unsafe {
        launch_gemv_i8(xp, pb.ptr(), sb.ptr() as *const f32, o_dim, i_dim, yp).expect("i8");
    });
    report(name, "i8 ", o_dim, i_dim, us);
}

/// The two MLA kv_b kernels at GLM dims. These have no other isolated instrument — the
/// only alternative is the engine's whole `route` bucket, which cannot separate them —
/// so without these rows `mla_absorb_fp8` cannot be given a matched before/after at all.
/// GB/s counts the kv_b bytes each kernel reads: H*nope*kvl for absorb, H*vh*kvl for
/// value. `mla_value` is the stated 254 GB/s reference that `mla_absorb`'s 99 GB/s was
/// judged against, so both belong in the same run.
fn run_mla(h: usize, qh: usize, nope: usize, vh: usize, kvl: usize) {
    let block = 128usize;
    let rows = h * (nope + vh);
    let kvb = dev(&pattern(rows * kvl, |v| f32_to_e4m3(v * 0.1)));
    let sc_cols = kvl.div_ceil(block);
    let kvb_scale = dev(&f32b(&vec![1.0f32; rows.div_ceil(block) * sc_cols]));
    let q = dev(&f32b(&vec![0.3f32; h * qh]));
    let clat = dev(&f32b(&vec![0.3f32; h * kvl]));
    let mut qabs = dev(&vec![0u8; h * kvl * 4]);
    let mut ctx = dev(&vec![0u8; h * vh * 4]);
    let (kp, sp) = (kvb.ptr(), kvb_scale.ptr() as *const f32);
    let (qp, ap) = (q.ptr() as *const f32, qabs.ptr_mut() as *mut f32);
    let (cp, xp) = (clat.ptr() as *const f32, ctx.ptr_mut() as *mut f32);

    let us = time(60, &|| unsafe {
        launch_mla_absorb_fp8(qp, kp, sp, h, qh, nope, vh, kvl, block, ap).expect("absorb");
    });
    let gbs = (h * nope * kvl) as f64 / (us * 1e-6) / 1e9;
    println!("mla_absorb  [{h}x{nope}x{kvl}]   {us:8.1}us  {gbs:6.1} GB/s");

    let us = time(60, &|| unsafe {
        launch_mla_value_fp8(cp, kp, sp, h, nope, vh, kvl, block, xp).expect("value");
    });
    let gbs = (h * vh * kvl) as f64 / (us * 1e-6) / 1e9;
    println!("mla_value   [{h}x{vh}x{kvl}]   {us:8.1}us  {gbs:6.1} GB/s");
}

/// MLA flash attention at two context lengths. This kernel's cost grows with context,
/// and its LDS/register change is aimed at exactly that, so a single nr is not enough to
/// characterise it — nr=512 is the profiled operating point and nr=2048 shows the slope.
/// Reported in µs (it is not a simple bandwidth kernel: the KV sweep is re-read ⌈H/HB⌉×
/// and the cost is a mix of DRAM, LDS and the online-softmax rescale).
fn run_attend(h: usize, kvl: usize, rope: usize) {
    let n_blocks = kvl / 128;
    for nr in [512usize, 2048] {
        let qabs = dev(&f32b(&vec![0.02f32; h * kvl]));
        let qrope = dev(&f32b(&vec![0.02f32; h * rope]));
        let lc8 = dev(&pattern(nr * kvl, |v| f32_to_e4m3(v * 0.1)));
        let lscale = dev(&f32b(&vec![1.0f32; nr * n_blocks]));
        let rc = dev(&u16b(&vec![0x3c00u16; nr * rope]));
        let mut clat = dev(&vec![0u8; h * kvl * 4]);
        let mut part = dev(&vec![0u8; attend_scratch_floats(h, kvl) * 4]);
        let (qa, qr) = (qabs.ptr() as *const f32, qrope.ptr() as *const f32);
        let (lp, ls) = (lc8.ptr(), lscale.ptr() as *const f32);
        let rp = rc.ptr() as *const u16;
        let (cp, pp) = (clat.ptr_mut() as *mut f32, part.ptr_mut() as *mut f32);
        let us = time(30, &|| unsafe {
            launch_attend(qa, qr, lp, ls, rp, std::ptr::null(), h, nr, kvl, rope, n_blocks, 0.08, cp, pp)
                .expect("attend");
        });
        println!("mla_attend  [h{h} nr{nr} kvl{kvl}]  {us:8.1}us");
    }
}

/// The rest of the `tail` bucket, so it can be attributed by SUBTRACTION rather than
/// assumed. `tail` measures ~16 ms/tok and lm_head was taken to be "almost all" of it;
/// lm_head measures ~8 ms, so ~8 ms belongs to something else. These are the only other
/// ops in the bucket. Neither kernel is modified by this branch — the rows exist to
/// decompose the budget, not to show a before/after.
fn run_tail_rest(vocab: usize, hidden: usize) {
    let logits = dev(&f32b(&(0..vocab).map(|i| (i % 977) as f32 * 1e-3).collect::<Vec<_>>()));
    let mut idx = dev(&[0u8; 4]);
    let mut val = dev(&[0u8; 4]);
    let (lp, ip, vp) = (
        logits.ptr() as *const f32,
        idx.ptr_mut() as *mut i32,
        val.ptr_mut() as *mut f32,
    );
    let us = time(60, &|| unsafe {
        launch_argmax(lp, vocab, ip, vp).expect("argmax");
    });
    println!("argmax      [{vocab}]         {us:8.1}us");

    let x = dev(&f32b(&vec![0.7f32; hidden]));
    let w = dev(&f32b(&vec![1.0f32; hidden]));
    let mut y = dev(&vec![0u8; hidden * 4]);
    let (xp, wp, yp) = (
        x.ptr() as *const f32,
        w.ptr() as *const f32,
        y.ptr_mut() as *mut f32,
    );
    let us = time(60, &|| unsafe {
        launch_rmsnorm(xp, wp, hidden, 1e-5, yp).expect("rmsnorm");
    });
    println!("rmsnorm     [{hidden}]           {us:8.1}us  (single workgroup, dim3(1))");
}

fn main() {
    println!("MoE dot decode throughput (wave-per-row, isolated):");
    run("gate/up", 2048, 6144); // hidden reduction
    run("down", 6144, 2048); // inter reduction

    // The two per-kernel targets in docs/PERF.md, at their real engine shapes.
    println!("\nRoute/tail GEMV bandwidth (real shapes, 256 GB/s peak):");
    run_fp8("o_proj", 6144, 16384); // ~half of `route`; split-K path
    run_i8("lm_head", 154880, 6144); // ~half of `tail`, measured — not all of it

    // GLM-5.2 manifest: num_attention_heads 64, qk_head_dim 256 (= qk_nope 192 + rope
    // 64), v_head_dim 256, kv_lora_rank 512. Getting these wrong understates the work:
    // absorb reads H*nope*kvl and value reads H*vh*kvl, so a guessed 128/128 would have
    // measured 4.2 MB where the engine moves 6.3 and 8.4.
    println!("\nMLA kv_b kernels (route):");
    run_mla(64, 256, 192, 256, 512);

    println!("\nMLA flash attention (route; scales with context):");
    run_attend(64, 512, 64);

    println!("\nRest of the tail bucket (attribution by subtraction):");
    run_tail_rest(154880, 6144);
}
