//! MoE dot-decode throughput microbench — int4 (dot_i4_wave) vs int3-VQ (dot_vq_wave)
//! vs fp8 (dot_fp8_wave) at the gate/up and down projection dims, isolated from the
//! routing/miss-count confound the decode bench has (there, a numerics change shifts
//! the greedy sequence → hit rate → compute bubbles). All wave-per-row (the MoE kernel
//! structure); fp8 at i_dim≥4096 dispatches to split-K (its live behaviour). Finding:
//! int4 decodes ~1.8× faster than vq3/fp8 — the all-int4 decode-bench slowdown was
//! residency (bigger experts → fewer slots → bubbles), not compute.
//! A second section adds the two `docs/measurement/perf-roadmap.md` per-kernel targets (o_proj, lm_head) at
//! their real engine shapes in GB/s. The MoE rows above are untouched, so numbers already
//! recorded in docs/measurement/benchmarks.md stay comparable.
//! The `v4gemv` section (2026-08-08) measures `gemv_fp8_bf16` serially at V4's seven decode
//! shapes — the isolated kernel rate the M7 A/B record names as missing (its in-engine
//! spans confound rate with exposure/contention; `docs/investigations/v4-decode-decomposition.md` §M8).
//! The `v4res` section (2026-08-09) does the same for the fp4 routed experts' RESIDENT range
//! — the `res` span, the engine's largest single bucket — and takes its working set as a
//! parameter so the 32 MB MALL confound §M11 names first is measured rather than argued.
//! The `glmi4` section (2026-08-09) is that same instrument pointed at GLM-5.2's int4 routed
//! experts — `dot_i4_wave_r`, the shipping default's MoE compute — at R = 1 and R = 2, which
//! is the question `docs/investigations/int4-moe-unroll.md` opens.
//! Run: cargo run --release --features rocm --example dot_bench
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]
use rivoli::artifact::quant::{
    VQ_DIM, VQ_K, f4_groups, f4_row_bytes, i4_expert_bytes, i4_groups, i4_row_bytes, matvec_fp8,
    matvec_i4, quant_i4, quant_vq,
};
use rivoli::backend::hip::{
    ExpertDesc, ExpertDescF4, attend_scratch_floats, device_sync, launch_act_quant_f8,
    launch_argmax, launch_attend, launch_gemv_fp8, launch_gemv_fp8_bf16, launch_gemv_i4,
    launch_gemv_i8, launch_gemv_vq, launch_mla_absorb_fp8, launch_mla_value_fp8,
    launch_moe_acc_drain, launch_moe_expert_range_f4, launch_moe_expert_range_i4,
    launch_rmsnorm_single,
};
use rivoli::math::{f32_to_e4m3, f32_to_f16};
use rivoli::memory::device::DeviceBuf;

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
        // docs/measurement/benchmarks.md predate this fix and are NOT comparable to rows measured after.
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
/// `n` uniform [-1,1) f32 as device bytes, and the same with the positive offset the
/// block scales need. Every fingerprinted buffer draws from one of these two.
fn rnd(r: &mut Rng, n: usize) -> Vec<u8> {
    f32b(&(0..n).map(|_| r.f()).collect::<Vec<_>>())
}
fn rnd_scale(r: &mut Rng, n: usize) -> Vec<u8> {
    f32b(
        &(0..n)
            .map(|_| (r.f() * 0.1).abs() + 0.01)
            .collect::<Vec<_>>(),
    )
}
fn f32v(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// FNV-1a over a kernel's raw output bytes — the only instrument here that separates
/// "bit-identical" from "within tolerance". See docs/measurement/benchmarks.md, "A fingerprint is the only
/// instrument that shows bit-identity", including why the inputs below must VARY.
fn fnv(b: &[u8]) -> u64 {
    b.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &x| {
        (h ^ x as u64).wrapping_mul(0x100_0000_01b3)
    })
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
    let err = want
        .iter()
        .zip(&got)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    let i4_ok = if err <= 1e-3 * mx + 1e-3 {
        "ok"
    } else {
        "MISMATCH"
    };

    // int3-VQ
    let cb: Vec<f32> = (0..VQ_K * VQ_DIM).map(|_| r.f()).collect();
    let (vqi, vqs) = quant_vq(&w, o_dim, i_dim, &cb);
    let (vqib, vqsb, cbb) = (dev(&vqi), dev(&u16b(&vqs)), dev(&f16b(&cb)));
    let us_vq = time(iters, &|| unsafe {
        launch_gemv_vq(
            xp,
            vqib.ptr(),
            vqsb.ptr() as *const u16,
            cbb.ptr() as *const u16,
            o_dim,
            i_dim,
            yp,
        )
        .expect("vq");
    });

    // fp8 (scale=1 blocks — decode cost is representative; accuracy irrelevant here)
    let fp8p: Vec<u8> = w.iter().map(|&v| f32_to_e4m3(v)).collect();
    let fp8s: Vec<f32> = vec![1.0; (o_dim / block) * (i_dim / block)];
    let (fp8pb, fp8sb) = (dev(&fp8p), dev(&f32b(&fp8s)));
    let us_fp8 = time(iters, &|| unsafe {
        launch_gemv_fp8(
            xp,
            fp8pb.ptr(),
            fp8sb.ptr() as *const f32,
            o_dim,
            i_dim,
            block,
            1,
            yp,
        )
        .expect("fp8");
    });

    let ge = |us: f64| gelem / (us * 1e-6) / 1e9;
    println!("{name} [{o_dim}x{i_dim}]  (gemv_i4 vs oracle: {i4_ok}, err {err:.2e}/{mx:.2})");
    println!("  int4 {us_i4:7.1}us  {:6.1} GElem/s  (1.00x)", ge(us_i4));
    println!(
        "  vq3  {us_vq:7.1}us  {:6.1} GElem/s  ({:.2}x int4)",
        ge(us_vq),
        us_i4 / us_vq
    );
    println!(
        "  fp8  {us_fp8:7.1}us  {:6.1} GElem/s  ({:.2}x int4){}",
        ge(us_fp8),
        us_i4 / us_fp8,
        if i_dim >= 4096 { "  [split-K]" } else { "" }
    );
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

/// One throughput row. `bytes` is the WEIGHT traffic the shape moves per launch — the
/// denominator the byte bands in `docs/investigations/v4-decode-decomposition.md` price
/// against, NOT a claim that every row is bounded by it (§M11's probe D measures the fp4
/// row as drain-bound, with the activation making up 8/9 of its vector-memory requests).
/// `shape` is free-form because the fp4 rows below are sized by an expert count rather than
/// by an `o x i` pair. Returns the GB/s so a caller deriving a second figure from the same
/// rate cannot compute it a second, silently divergent way.
fn report_bytes(name: &str, kind: &str, shape: &str, bytes: usize, us: f64) -> f64 {
    let gbs = bytes as f64 / (us * 1e-6) / 1e9;
    println!(
        "{name} {kind} {shape}  {us:8.1}us  {gbs:6.1} GB/s  ({:.0}% of 256)",
        gbs / 2.56
    );
    gbs
}

fn report(name: &str, kind: &str, o_dim: usize, i_dim: usize, us: f64) {
    report_bytes(name, kind, &format!("[{o_dim}x{i_dim}]"), o_dim * i_dim, us);
}

/// The two BIG real shapes, throughput only. Separate from `run` because `run`
/// materializes an `o·i` f32 host buffer and CPU-quantizes it three ways — at
/// [154880,6144] the VQ argmin alone would never finish. Reported as GB/s of weight
/// traffic against the 256 GB/s LPDDR5 peak, which is the number these kernels are
/// actually bounded by.
///
/// THROUGHPUT PLUS A FINGERPRINT. `x` and the block scales VARY (they used to be
/// constant); addresses and traffic are identical either way, so no timing row moves.
/// The fingerprint is NOT an accuracy check — the oracles in tests/kernel.rs are.
fn run_fp8(name: &str, o_dim: usize, i_dim: usize) {
    let block = 128usize;
    let seed = 0xF8 ^ i_dim as u64;
    let mut r = Rng(seed);
    // |v| <= 0.1 never encodes to the e4m3 NaN pattern, so no fixup is needed.
    let packed = pattern(o_dim * i_dim, |v| f32_to_e4m3(v * 0.1));
    let scale = rnd_scale(&mut r, (o_dim / block) * (i_dim / block));
    // Two token rows' worth of x and y; the nrow=1 arm uses the first row of each, so both
    // arms read the SAME weight bytes and the ratio is purely "what does a second row cost".
    let (xb, pb, sb) = (dev(&rnd(&mut r, 2 * i_dim)), dev(&packed), dev(&scale));
    let mut yb = dev(&vec![0u8; 2 * o_dim * 4]);
    let (xp, yp) = (xb.ptr() as *const f32, yb.ptr_mut() as *mut f32);
    let us = time(60, &|| unsafe {
        launch_gemv_fp8(
            xp,
            pb.ptr(),
            sb.ptr() as *const f32,
            o_dim,
            i_dim,
            block,
            1,
            yp,
        )
        .expect("fp8");
    });
    report(name, "fp8", o_dim, i_dim, us);
    // Row 0's bytes, before the 2-row arm rewrites them. `x` row 0 is the same first
    // `i_dim` draws it always was, so this fingerprint stays comparable to values recorded
    // before batching existed.
    let row0 = yb.copy_out().expect("out")[..o_dim * 4].to_vec();
    let us2 = time(60, &|| unsafe {
        launch_gemv_fp8(
            xp,
            pb.ptr(),
            sb.ptr() as *const f32,
            o_dim,
            i_dim,
            block,
            2,
            yp,
        )
        .expect("fp8 r2");
    });
    report(name, "fp8 r2", o_dim, i_dim, us2);
    // BIT-identity at the real engine shape, which the unit test's toy dims cannot reach:
    // o_proj is [6144,16384] and goes down the split-K path, where the 2-row kernel folds
    // K partials per row out of a [R][K] LDS tile. A row-indexing slip there produces
    // plausible numbers, not a crash.
    let after = yb.copy_out().expect("out2");
    assert!(
        after[..o_dim * 4] == row0[..],
        "{name}: nrow=2 row 0 differs from nrow=1 — the batched kernel is not the same arithmetic"
    );
    println!(
        "            2-row cost ratio {:.3}x  (row 0 bit-identical)",
        us2 / us
    );
    // The hash is only comparable against a run with the SAME generator, so print what
    // determines it: a recorded bare hash goes stale the first time a seed or a draw
    // order moves, and reads as a numerics regression when it does.
    println!(
        "            fnv {:016x}  (seed {seed:#x}, {o_dim}x{i_dim} blk{block})",
        fnv(&row0)
    );
}

/// `gemv_fp8_bf16` at the engine's real decode shapes (m = 1) — the kernel M6 fired one
/// kill across three spans on (qkv 2.31×, oproj 1.99×, shared 2.82× bytes) and M7's
/// unroll-8 attacked to 1.44–1.89×. Run SERIAL on an otherwise idle device, this is the
/// instrument the M7 record said was missing: the engine's spans mix kernel rate with
/// exposure and cross-stream contention (`gpu shared` was ruled unscorable for exactly
/// that), so only an isolated rate can say whether the residual above bytes is the
/// kernel's own or the machine's effective ceiling for this access pattern.
///
/// GB/s counts weight bytes only — they are the traffic (33.55 MB/tensor against a
/// ≤32 KB `x` that stays L2-resident in the engine too). `groups` is benched at 1:
/// wo_a's `o_groups` changes which slice of `x` a row reads, not the weight stream the
/// timing is bound by. Weights ROTATE over enough device copies to spill the 32 MB
/// MALL, not just the 2 MB L2 — benchmarks.md records a single 33.5 MB buffer
/// replaying at 372 GB/s (above the 256 GB/s bus, i.e. MALL-served) and 4 rotating
/// copies (134 MB) bringing it back to 157; the engine reads every weight byte exactly
/// once per token out of GTT, so a cache-served row here misclassifies the §M8
/// floor-vs-mechanics decision. Budget ≥ 8× the MALL per shape.
fn run_v4_gemv(name: &str, n_out: usize, k: usize) {
    let block = 128usize;
    let bytes = n_out * k;
    // Seeded on the PAIR, not on `bytes`: wq_b/wo_a/wo_b all multiply to 33,554,432, and
    // a bytes-derived seed would hand three shapes one scale/x draw sequence.
    let seed = 0x74F8 ^ ((n_out as u64) << 20) ^ k as u64;
    let mut r = Rng(seed);
    let copies = ((256 << 20) / bytes).clamp(4, 128);
    let packed = pattern(bytes, |v| f32_to_e4m3(v * 0.1));
    let sc_len = n_out.div_ceil(block) * k.div_ceil(block);
    let scale_f = f32v(&rnd_scale(&mut r, sc_len));
    let x = f32v(&rnd(&mut r, k));
    let wb: Vec<DeviceBuf> = (0..copies).map(|_| dev(&packed)).collect();
    let (xb, sb) = (dev(&f32b(&x)), dev(&f32b(&scale_f)));
    let mut yb = dev(&vec![0u8; n_out * 4]);
    let (xp, yp) = (xb.ptr() as *const f32, yb.ptr_mut() as *mut f32);
    let turn = std::cell::Cell::new(0usize);
    let us = time(60, &|| unsafe {
        let w = &wb[turn.get() % copies];
        turn.set(turn.get() + 1);
        launch_gemv_fp8_bf16(
            xp,
            w.ptr(),
            sb.ptr() as *const f32,
            1,
            n_out,
            k,
            block,
            1,
            yp,
            std::ptr::null_mut(),
        )
        .expect("v4 fp8");
    });
    report(name, "v4fp8", n_out, k, us);
    // Trustworthiness, i4-row style: right layout and addresses, not bit-identity — the
    // wave reduction re-associates the fold and the kernel rounds its output to bf16
    // (`rbf16`), so the bound is bf16's 2^-9 step plus reassociation, against the host
    // oracle whose scale grid this kernel documents itself as matching.
    let mut want = vec![0f32; n_out];
    matvec_fp8(&mut want, &x, &packed, &scale_f, k, block);
    let out = yb.copy_out().expect("v4 out");
    let got = f32v(&out);
    let mx = want.iter().fold(0f32, |m, v| m.max(v.abs()));
    let err = want
        .iter()
        .zip(&got)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        err <= 4e-3 * mx + 1e-3,
        "{name}: gemv_fp8_bf16 disagrees with matvec_fp8 (err {err:.2e} vs max {mx:.2e})"
    );
    // `seed` governs scales and `x` only — `pattern()` draws the weight bytes from its
    // own fixed generator — so the printed (seed, shape, blk) triple plus pattern()'s
    // constant is what determines this row's inputs.
    println!(
        "            fnv {:016x}  (seed {seed:#x}, {n_out}x{k} blk{block}, {copies} copies, oracle err {err:.1e})",
        fnv(&out)
    );
}

/// 5.885 residents x 43 MoE layers — the expert-weight reads the engine's `res` span makes
/// per token, which is what turns a rate here back into the ms/token §M11 scores. Kept as
/// the two counts with the BYTE term taken from `f4_expert_bytes` at the print site: a baked
/// 3.383 GB/token would carry an unpinned copy of the same 13.37 MB the GB/s denominator is
/// pinned to, and would drift from the artifact silently.
const V4_RES_EXPERT_READS_PER_TOKEN: f64 = 5.885 * 43.0;

/// `rivoli_moe_expert_range_f4` at V4's real resident shape, serial — the isolated rate
/// behind the engine's `res` span (§M11 probe A). The engine launches exactly this call
/// ONCE per MoE layer for the layer's residents (M3b's boundary), so the unit timed here is
/// the unit the span measures: gate/up, the `h` re-quantization, and down.
///
/// **`ranges` is the whole instrument.** gfx1151 has a 32 MB MALL and one layer's residents
/// are only ~80 MB, so a probe that replays ONE range's weights gets partial cache service
/// and overstates the rate, while the engine rotates ~3.4 GB of DISTINCT experts per token
/// and never gets reuse. The caller runs this twice — at 14 ranges (1.12 GB, the engine's
/// condition) and at 1 (the naive harness) — so the confound is MEASURED rather than argued.
/// `run_v4_gemv` records the same effect on the fp8 side: 372 GB/s replaying one 33.5 MB
/// buffer against 157 rotating four. The row prints its own working set in MB.
///
/// Returns the `e_start = 0` fingerprint so the caller can check the two arms agree — see
/// `main`, where the expectation is asserted rather than left for the eye.
///
/// **`run_glm_i4` below is this function's twin and is NOT factored with it.** `examples/` is
/// outside `build.rs`'s jscpd gate, so nothing will report it if the two drift — this
/// cross-reference is the only thing pinning them, and an edit here should ask whether it
/// belongs there too. What is deliberately DIFFERENT there, so a "fix" does not undo it: int4
/// group scales are f32 where these are one e8m0 byte; there is no `act_quant_f8` (GLM's int4
/// path consumes raw f32 activations); the launcher takes `nrow` instead of `n_desc` and a
/// swiglu limit; and it derives NO ms/token, because GLM has no booked span to project onto.
fn run_v4_res(name: &str, hidden: usize, inter: usize, e_count: usize, ranges: usize) -> u64 {
    // V4's config value, hand-copied. The launcher only refuses a clamp-DISABLING limit (rc
    // 1006); a finite-but-wrong one runs silently, and only this section's fingerprint
    // disagreeing with `tests/f4_kernel.rs` would ever show it.
    const SWIGLU_LIMIT: f32 = 10.0;
    // Row and group sizing from `quant.rs`, not a local `32` that could drift from `F4_GROUP`
    // into a scale ROW STRIDE error. `per_expert` is then summed from the spans this probe
    // actually uploads, so the GB/s denominator cannot disagree with the traffic;
    // `quant.rs::f4_expert_bytes` is the same closed form, so asserting the two match would be
    // comparing an expression with itself.
    let (gu_packed, dn_packed) = (inter * f4_row_bytes(hidden), hidden * f4_row_bytes(inter));
    let (gu_scale, dn_scale) = (inter * f4_groups(hidden), hidden * f4_groups(inter));
    let per_expert = 2 * (gu_packed + gu_scale) + dn_packed + dn_scale; // 13.37 MB here
    let n_desc = ranges * e_count;

    let seed = 0xF4E5 ^ hidden as u64 ^ ((inter as u64) << 24);
    let mut r = Rng(seed);
    // `x` is drawn FIRST and at a length that does not depend on `n_desc`, so both rows draw
    // the same `x` and the same first `e_count` routing weights. The weight bytes come off a
    // SEPARATE generator below for the same reason — `wexpert`'s length is the only draw here
    // that varies with `ranges`, and letting it shift the weight stream would make the two
    // rows' fingerprints incomparable.
    let mut xb = dev(&rnd(&mut r, hidden));
    let wb = dev(&rnd_scale(&mut r, n_desc));

    // NOT `pattern()`. Its repeating 4 KiB block aliases with every row stride at this shape
    // — gate/up rows are 2048 B, down rows 1024 B, their scale rows 128/64 B — so the drained
    // residual would hold 64 distinct floats replicated 64x. §M11 D.5 makes this fingerprint
    // probe C's bit-identity gate, and 4096 independent dot products make a stronger one than
    // 64 for 9 MB of host LCG, drawn once.
    let mut wr = Rng(0xF4B0);
    let packed_of = |r: &mut Rng, n: usize| -> Vec<u8> {
        (0..n).map(|_| ((r.f() + 1.0) * 127.5) as u8).collect()
    };
    // e8m0 bytes in [0x71, 0x7d] = 2^-14..2^-2, NOT the full byte range. 0x00 is 2^-127,
    // below f32's smallest normal, which zeroes a whole 32-weight group; 0xff is the format's
    // NaN, which `moe_fixed` launders into a finite extreme. Both ends drive the drained
    // output to a constant, which is what the DEGENERATE line below reports.
    let scale_of = |r: &mut Rng, n: usize| -> Vec<u8> {
        (0..n)
            .map(|_| (0x77i32 + (r.f() * 6.0) as i32) as u8)
            .collect()
    };
    let packed_gu = packed_of(&mut wr, gu_packed);
    let packed_dn = packed_of(&mut wr, dn_packed);
    let scale_gu = scale_of(&mut wr, gu_scale);
    let scale_dn = scale_of(&mut wr, dn_scale);

    // One set of bytes, `n_desc` copies at DISTINCT device addresses — `run_v4_gemv`'s pattern
    // and its reason: the kernel reads every weight byte exactly once whatever its value (the
    // e2m1/e8m0 decodes have been branchless register-immediate tables since M3c, so there is
    // no data-dependent path and no LDS LUT for a shared fill to collapse into a broadcast),
    // and what must differ between experts is which DRAM pages they occupy. Gate and up
    // therefore share bytes, so `g == u`; that makes a w1/w3 swap invisible here, which is not
    // this probe's job — `tests/f4_kernel.rs::Wiring::SwapGateUp` covers it against an oracle,
    // and `a_dispatch_split_into_ranges_matches_one_range` covers absolute-vs-range-relative
    // descriptor indexing at two non-adjacent non-zero starts, bit-identically. **Every arm of
    // §M11's round patches only `dot_f4_wave_r`'s inner loop**, so fold order is the one thing
    // that can differ between them and the one thing this fingerprint has to catch.
    let mut parts: Vec<DeviceBuf> = Vec::with_capacity(n_desc * 6);
    let mut descs: Vec<ExpertDescF4> = Vec::with_capacity(n_desc);
    for _ in 0..n_desc {
        // Address taken BEFORE the move into `parts`, for the reason tests/f4_kernel.rs's
        // `F4Experts::upload` gives: recovering it by index afterwards works until the buffer
        // count changes, and then a descriptor silently points at another projection.
        let mut push = |b: &[u8]| {
            let d = dev(b);
            let p = d.ptr();
            parts.push(d);
            p
        };
        descs.push(ExpertDescF4 {
            gate_packed: push(&packed_gu),
            gate_scale: push(&scale_gu),
            up_packed: push(&packed_gu),
            up_scale: push(&scale_gu),
            down_packed: push(&packed_dn),
            down_scale: push(&scale_dn),
        });
    }
    // SAFETY: `ExpertDescF4` is `#[repr(C)]` plain addresses, so the span is exactly the
    // slice's bytes.
    let descb = dev(unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            std::mem::size_of_val(&descs[..]),
        )
    });
    // `wexpert` and `h` are indexed by the ABSOLUTE descriptor index, not by position in the
    // range — so both are sized for `n_desc`, which is what lets `e_start > 0` rotate at all.
    let mut hb = dev(&vec![0u8; n_desc * inter * 4]);
    let mut ab = dev(&vec![0u8; hidden * 8]);
    let mut ob = dev(&vec![0u8; hidden * 4]);
    let (xp, wp) = (xb.ptr() as *const f32, wb.ptr() as *const f32);
    let dp = descb.ptr() as *const ExpertDescF4;
    let (hp, ap, op) = (
        hb.ptr_mut() as *mut f32,
        ab.ptr_mut() as *mut u64,
        ob.ptr_mut() as *mut f32,
    );
    // The engine quantizes `x` before the range call (every quantized `Linear` in V4
    // quantizes its own activation), so the numbers this kernel consumes are `e4m3(x/s)·s`
    // here too. SAFETY: `xb` is `hidden` live f32 and outlives the sync below.
    unsafe { launch_act_quant_f8(xb.ptr_mut() as *mut f32, 1, hidden, std::ptr::null_mut()) }
        .expect("act_quant_f8");
    // SAFETY: every buffer above is sized for `n_desc` ABSOLUTE expert slots and stays alive
    // until this function returns; every range below is inside that bound.
    let launch = |e_start: usize| unsafe {
        launch_moe_expert_range_f4(
            xp,
            hidden,
            inter,
            e_start,
            e_count,
            n_desc,
            dp,
            wp,
            SWIGLU_LIMIT,
            hp,
            ap,
            1,
            std::ptr::null_mut(),
        )
        .expect("moe_expert_range_f4");
    };

    // The fingerprint first, off the freshly zeroed `acc`/`out` allocations — the drain resets
    // `acc`, so nothing has to be cleared by hand and the timing loop's atomics cannot ride
    // into the reading.
    launch(0);
    // SAFETY: same stream (the null one), so the range's atomics precede the drain.
    unsafe { launch_moe_acc_drain(op, ap, hidden, 1, 1.0, std::ptr::null_mut()) }.expect("drain");
    device_sync().expect("s");
    let out = ob.copy_out().expect("out");
    let vals = f32v(&out);
    let distinct = vals
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>()
        .len();
    // Rotating through DISTINCT experts between launches is the instrument; `iters` is a whole
    // number of sweeps so every expert is read the same number of times, and `time`'s one
    // untimed warm-up only shifts the phase.
    let iters = (56 / ranges).max(1) * ranges;
    let turn = std::cell::Cell::new(0usize);
    let us = time(iters as u32, &|| {
        let t = turn.get();
        turn.set(t + 1);
        launch((t % ranges) * e_count);
    });
    let bytes = e_count * per_expert;
    let ws_mb = n_desc * per_expert / 1_000_000; // decimal, to match the GB/s denominator
    let gbs = report_bytes(
        name,
        "f4res",
        &format!("[{e_count}e {hidden}x{inter} ws{ws_mb}MB]"),
        bytes,
        us,
    );
    // BOTH derived lines are for the engine-condition arm only. The control arm is a MALL-
    // served rate by construction, and converting one into an engine ms/token is the exact
    // failure §M11 opens by naming — it would be the only line here in engine units, so it is
    // also the one most likely to be lifted into benchmarks.md.
    if ranges > 1 {
        // SERIAL-IDLE, and not the engine's `res`. M9 measured that a microbench rate does NOT
        // transfer 1:1 into the overlapped engine (its split-k `dqkv` came in at −0.9 against a
        // −2.3..−2.9 projection), so this is what the span WOULD be if it did — a ceiling for
        // the projection, not a wall.
        let gb_per_token = V4_RES_EXPERT_READS_PER_TOKEN * per_expert as f64 / 1e9;
        println!(
            "            res {:5.1} ms/token serial-idle at this rate  (booked 24.3; \
             {gb_per_token:.3} GB/token; no M9 transfer discount applied)",
            gb_per_token / gbs * 1e3
        );
        // §M11's registered band, printed where the number is rather than left in the doc.
        println!(
            "            probe A band 130-150 GB/s; KILL >=170 or <=110 (harness, not kernel)"
        );
    }
    // Reported, NOT asserted. §M11 probe B's ballast arms drive every row past `moe_fixed`'s
    // ±2^14 saturation, so their residual IS a constant and this would fire on 2 of the 5
    // planned arms — a guard expected to go red is one an operator learns to skip, and an
    // abort here would also cost the second `run_v4_res` call (the MALL control row) which
    // runs in the same process. What it protects is real: a constant residual fingerprints
    // identically under ANY kernel, so probe C's bit-identity kill would pass on it. Full
    // entropy puts `distinct` at ~`hidden`; a collapse to a handful is the tell.
    let degenerate = if distinct * 4 > hidden {
        ""
    } else {
        "  DEGENERATE — constant residual, this fnv matches any kernel (expected on probe B)"
    };
    // `seed` governs `x` and the routing weights; the weight and scale bytes come off the
    // separate fixed generator above. Both constants plus the shape determine this row.
    let h = fnv(&out);
    println!(
        "            fnv {h:016x}  (seed {seed:#x}, {n_desc} experts, {ranges} ranges x {iters} \
         iters, {distinct} distinct){degenerate}"
    );
    h
}

/// GLM-5.2's int4 routed-expert range at the artifact's real dims — the serial rate of
/// `dot_i4_wave_r`, which is `--mode int4` outright and, through the HOT format,
/// `--mode hybrid`'s resident half. `docs/investigations/int4-moe-unroll.md` G1.
///
/// **This is the only instrument that can see a change to this loop, and that is a
/// deliberate scope choice, not a shortcut.** GLM decodes 483 ms/token with ~260 ms of it
/// fetch exposure (180.4 miss/token x 1.44 ms/miss), so a kernel win of tens of percent on
/// a fraction of the compute is below the wall's resolving power at any n this project can
/// afford. A wall A/B staged against it would be a gate structurally incapable of seeing
/// the thing under test. So this row prints a RATE and **derives no ms/token from it** —
/// unlike `run_v4_res`, which can, because V4 has a booked `res` span to project onto.
///
/// **It also cannot be validated against the engine the way `run_v4_res`'s probe A was.**
/// V4 had a booked 24.3 ms serial `res` span, so its band was a reproduction gate. There is
/// no equivalent booked GLM int4 expert-compute span, so the band printed below is only a
/// sanity band — "is this measuring DRAM at all" — and the `mall-ctl` row beside it is what
/// actually makes the number trustworthy.
///
/// `ranges` is that instrument, for the reason `run_v4_res` gives at length: gfx1151 has a
/// 32 MB MALL, so a probe that replays one range gets partial cache service and overstates
/// the rate. One range here is 6 x 20.05 MB = 120 MB — already 3.8x the MALL, which is why
/// the control has to be MEASURED rather than assumed to be free.
///
/// `nrow` is 1 or 2: GLM instantiates `dot_i4_wave_r<2>` because speculative decode was
/// measured to pay (1.108x at `--mtp-min-conf 0.8`), and R = 2 is the whole reason M11's fp4
/// result cannot be copied across — the loop body holds `2R` accumulators and reads `2R`
/// `float4`s per step. **GB/s is WEIGHT bytes over time at both**, because one read of the
/// weight row serves both token rows; so the two `nrow` rows are not competing arms, and
/// only depth-vs-depth WITHIN an `nrow` is a comparison.
///
/// Returns `(row 0's fingerprint, row 1's if `nrow == 2`)`, and **both halves are gated**.
///
/// Row 0 is arithmetically independent of `nrow` — `a0[t]` accumulates per `t` with no
/// cross-row term, and `moe_down_i4_impl` reads `he = h_in + e*R*inter`, so `t = 0` is always
/// the same slice — so every arm here, R = 1 and R = 2 alike, must return the same row-0
/// value. Row 1's inputs are just as `ranges`-independent (separate generator, `n_desc`-
/// independent lengths), so the two R = 2 arms must agree on row 1 as well.
///
/// **Row 1 is fingerprinted because without it this probe is blind to the one failure it
/// exists to price.** An unroll that broke `vt = v + t * v_stride` for `t = 1` past the first
/// trip would leave row 0 bit-identical and sail through a row-0-only gate — and the R = 2
/// path is precisely what this stretch is measuring. `tests/kernel.rs::
/// batched_rows_are_bit_identical_to_single_rows` covers row 0 against `nrow = 1`; nothing
/// covered row 1 at depth.
fn run_glm_i4(
    name: &str,
    hidden: usize,
    inter: usize,
    e_count: usize,
    ranges: usize,
    nrow: usize,
) -> (u64, Option<u64>) {
    // The four spans this probe actually uploads, so the GB/s denominator cannot disagree
    // with the traffic. int4 group scales are **f32**, so each scale span is `groups * 4`
    // BYTES; using `i4_groups(dim)` with a one-byte scale — the fp4 twin's e8m0 width —
    // understates `per_expert` by 884,736 B (4.41%) and would report every GB/s below 4.41%
    // **LOW**, since `report_bytes` divides bytes by time.
    //
    // Note, because it is a trap in the other direction: copying `f4_row_bytes`/`f4_groups`
    // in verbatim gives the IDENTICAL number here, not a wrong one — `I4_GROUP`(128) /
    // `F4_GROUP`(32) == 4 == `sizeof(f32)`, so `i4_groups(d) * 4 == f4_groups(d) * 1` at every
    // dim. The hazard is the scale WIDTH, not the helper names.
    let (gu_packed, dn_packed) = (inter * i4_row_bytes(hidden), hidden * i4_row_bytes(inter));
    let (gu_scale, dn_scale) = (inter * i4_groups(hidden) * 4, hidden * i4_groups(inter) * 4);
    let per_expert = 2 * (gu_packed + gu_scale) + dn_packed + dn_scale;
    // What this catches, stated precisely, because `run_v4_res` above DECLINES the same assert
    // as "comparing an expression with itself" and the two should not give opposite advice on
    // the same question: both sides compose `i4_row_bytes`/`i4_groups`, so this does NOT check
    // the scale width or the group size — it checks that the probe reduces gate/up over
    // `hidden` and down over `inter`, i.e. a transposed `(o, i)` pair. That is worth one line
    // here and is not worth it there, because `run_v4_res` uploads its spans from the same two
    // `f4_*` helpers its denominator is summed from, while this one is the FIRST consumer of
    // the int4 layout outside `src/`.
    //
    // The check that is not weak is external and recorded in the investigation doc: at
    // 6144/2048 this is 20,054,016 bytes and `/var/db/rivoli/glm52-vq3-full/L03.i4` is
    // 5,153,882,112 = 257 x that exactly (256 routed + the shared expert).
    assert_eq!(
        per_expert,
        i4_expert_bytes(hidden, inter),
        "probe uploads a different expert block than the artifact stores"
    );
    let n_desc = ranges * e_count;
    let ws_mb = n_desc * per_expert / 1_000_000; // decimal, to match the GB/s denominator
    // Checked HERE, before a byte is uploaded or a launch is timed — a guard placed after
    // `time()` fires only once the measurement it was meant to prevent has been paid for.
    // `ranges > 1` is what marks an engine-condition arm; the control arm is a cache artifact
    // on purpose and has no floor.
    assert!(
        ranges == 1 || ws_mb >= 1000,
        "engine-condition arm must rotate >= 1 GB past the 32 MB MALL; this one rotates {ws_mb} MB"
    );

    // TOKEN ROW 0's inputs are drawn first and at `nrow`- and `n_desc`-independent lengths, so
    // every arm's row-0 fingerprint is comparable: `x` row 0 is the same `hidden` draws
    // everywhere, and `w`'s first `e_count` entries are the same whether `n_desc` is 6 or 54.
    // Row 1's inputs come off a SEPARATE generator for that reason — letting them share the
    // stream would make `n_desc` shift row 0's routing weights.
    let seed = 0x14E5 ^ hidden as u64 ^ ((inter as u64) << 24);
    let mut r = Rng(seed);
    let x0 = rnd(&mut r, hidden);
    let w0 = f32v(&rnd_scale(&mut r, n_desc));
    let mut r1 = Rng(seed ^ 0x5EC0_0000);
    let x1 = rnd(&mut r1, hidden);
    let w1 = f32v(&rnd_scale(&mut r1, n_desc));
    // `wexpert` is indexed `[e * nrow + t]` (token row fastest), so row 0's weight for expert
    // `e` sits at `e * nrow` and is `w0[e]` at either width.
    let mut wex = Vec::with_capacity(n_desc * nrow);
    for e in 0..n_desc {
        wex.push(w0[e]);
        if nrow == 2 {
            wex.push(w1[e]);
        }
    }
    let xb = dev(&if nrow == 2 {
        [x0.as_slice(), x1.as_slice()].concat()
    } else {
        x0.clone()
    });
    let wb = dev(&f32b(&wex));

    // One set of weight bytes, `n_desc` copies at DISTINCT device addresses — `run_v4_res`'s
    // pattern and its reason: the kernel reads every weight byte exactly once whatever its
    // value (the nibble decode is `bfe`/`add`/`cvt`, register-only, with no data-dependent
    // path and no LDS table a shared fill could collapse into a broadcast), and what must
    // differ between experts is which DRAM pages they occupy. NOT `pattern()`: its repeating
    // 4 KiB block aliases with these row strides (gate/up rows 3072 B, down rows 1024 B), which
    // would collapse the drained residual's entropy and weaken the fingerprint.
    //
    // Gate and up therefore share bytes, so `g == u` and a w1/w3 swap is invisible here. That
    // is not this probe's job — `tests/kernel.rs` covers int4 against the CPU oracle. What the
    // fingerprint has to catch is FOLD ORDER, because that is the only thing an unroll of this
    // loop can change.
    let mut wr = Rng(0x14B0);
    let bytes_of = |r: &mut Rng, n: usize| -> Vec<u8> {
        (0..n).map(|_| ((r.f() + 1.0) * 127.5) as u8).collect()
    };
    let packed_gu = bytes_of(&mut wr, gu_packed);
    let packed_dn = bytes_of(&mut wr, dn_packed);
    // f32 group scales in [0.01, 0.11) — the same band `rnd_scale` uses everywhere here, and
    // small enough that the accumulated partials stay far under `moe_fixed`'s +/-2^14
    // saturation, which would flatten the residual into a constant and make the fingerprint
    // match any kernel. The `distinct` line below is what reports it if that ever stops
    // holding.
    let scale_gu = rnd_scale(&mut wr, gu_scale / 4);
    let scale_dn = rnd_scale(&mut wr, dn_scale / 4);

    let mut parts: Vec<DeviceBuf> = Vec::with_capacity(n_desc * 6);
    let mut descs: Vec<ExpertDesc> = Vec::with_capacity(n_desc);
    for _ in 0..n_desc {
        // Address taken BEFORE the move into `parts`, for the reason `run_v4_res` gives:
        // recovering it by index afterwards works until the buffer count changes, and then a
        // descriptor silently points at another projection.
        let mut push = |b: &[u8]| {
            let d = dev(b);
            let p = d.ptr();
            parts.push(d);
            p
        };
        // `ExpertDesc`'s scale fields are typed `*const u16` from the VQ carrier; the int4
        // kernel reads them as `const float*`. One layout, the kernel picks the
        // interpretation — see the type's own doc.
        descs.push(ExpertDesc {
            gate_indices: push(&packed_gu),
            gate_scales: push(&scale_gu) as *const u16,
            up_indices: push(&packed_gu),
            up_scales: push(&scale_gu) as *const u16,
            down_indices: push(&packed_dn),
            down_scales: push(&scale_dn) as *const u16,
        });
    }
    // SAFETY: `ExpertDesc` is `#[repr(C)]` plain addresses, so the span is exactly the slice's
    // bytes.
    let descb = dev(unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            std::mem::size_of_val(&descs[..]),
        )
    });
    // `wexpert` and `h` are indexed by the ABSOLUTE descriptor index, not by position in the
    // range, so both are sized for `n_desc` — which is what lets `e_start > 0` rotate at all.
    let mut hb = dev(&vec![0u8; n_desc * nrow * inter * 4]);
    // ONE accumulator stream row (this probe launches on the null stream only), `nrow` token
    // rows, so the drain below takes `n = nrow * hidden` and `rows = 1`. The engine passes
    // `rows = MOE_ACC_ROWS` because it has a second, miss-stream row to fold in.
    let mut ab = dev(&vec![0u8; nrow * hidden * 8]);
    let mut ob = dev(&vec![0u8; nrow * hidden * 4]);
    let (xp, wp) = (xb.ptr() as *const f32, wb.ptr() as *const f32);
    let dp = descb.ptr() as *const ExpertDesc;
    let (hp, ap, op) = (
        hb.ptr_mut() as *mut f32,
        ab.ptr_mut() as *mut u64,
        ob.ptr_mut() as *mut f32,
    );
    // NO `act_quant_f8` here, unlike the fp4 twin: GLM's int4 path consumes raw f32
    // activations. Quantizing them would measure a different kernel's inputs.
    //
    // SAFETY: every buffer above is sized for `n_desc` ABSOLUTE expert slots and `nrow` token
    // rows, and stays alive until this function returns; every range below is inside that
    // bound.
    let launch = |e_start: usize| unsafe {
        launch_moe_expert_range_i4(
            xp,
            hidden,
            inter,
            e_start,
            e_count,
            dp,
            wp,
            hp,
            ap,
            nrow,
            std::ptr::null_mut(),
        )
        .expect("moe_expert_range_i4");
    };

    // The fingerprint first, off the freshly zeroed `acc`/`out` allocations — the drain resets
    // `acc`, so nothing has to be cleared by hand and the timing loop's atomics cannot ride
    // into the reading.
    launch(0);
    // SAFETY: same stream (the null one), so the range's atomics precede the drain.
    unsafe { launch_moe_acc_drain(op, ap, nrow * hidden, 1, 1.0, std::ptr::null_mut()) }
        .expect("drain");
    device_sync().expect("s");
    let out = ob.copy_out().expect("out");
    let row0 = &out[..hidden * 4];
    let row1 = (nrow == 2).then(|| fnv(&out[hidden * 4..]));
    let distinct = f32v(row0)
        .iter()
        .map(|v| v.to_bits())
        .collect::<std::collections::HashSet<_>>()
        .len();

    let bytes = e_count * per_expert;
    // A whole number of sweeps, so every expert is read the same number of times and `time`'s
    // one untimed warm-up only shifts the phase. 6 GB is a CEILING on the timed weight traffic,
    // not a floor — this floors the sweep count — so the rows land at 5.4 / 5.9 / 5.4 / 5.9 /
    // 5.0 GB. That is enough work that launch overhead and clock noise are small shares at both
    // a 120 MB launch and a 20 MB one, which is the only property being bought here.
    let iters = ((6_000_000_000usize / (ranges * bytes)).max(1) * ranges) as u32;
    let turn = std::cell::Cell::new(0usize);
    let us = time(iters, &|| {
        let t = turn.get();
        turn.set(t + 1);
        launch((t % ranges) * e_count);
    });
    report_bytes(
        name,
        "i4res",
        &format!("[{e_count}e r{nrow} {hidden}x{inter} ws{ws_mb}MB]"),
        bytes,
        us,
    );
    // THE PRE-DEVICE SANITY BAND WAS HERE AND IS DELETED, 2026-08-09, by its own result.
    // It read "130-175 GB/s, KILL >=200 or <=90" and was registered for ONE round against the
    // stock kernel; that registration is spent and recorded (benchmarks.md "GLM int4 MoE
    // unroll round"). Keeping it would have been actively harmful: the winning arm measured
    // 190.6 / 190.1 GB/s, so `unroll 4` prints OUT OF BAND and sits 5% under a KILL — if it
    // merges, every future run cries wolf against its own default kernel. The `mall-ctl` rows
    // are what actually make these numbers trustworthy, and they stay.
    //
    // Reported, NOT asserted — `run_v4_res`'s argument: a guard expected to go red on a
    // planned arm is one an operator learns to skip, and an abort here would also cost the
    // control row that runs after it in the same process. Full entropy puts `distinct` at
    // ~`hidden`; a collapse to a handful means `moe_fixed` saturated and every fingerprint
    // this arm reports then matches ANY kernel — B1's did, and its four cross-arm fingerprint
    // checks passed VACUOUSLY as a result. A `DEGENERATE` row proves nothing about bit-identity.
    let degenerate = distinct * 4 <= hidden;
    let marker = if degenerate {
        "  DEGENERATE — constant residual, this fnv matches any kernel"
    } else {
        ""
    };
    let h = fnv(row0);
    println!(
        "            fnv {h:016x}  row 0 of {nrow}  (seed {seed:#x}, {n_desc} experts, {ranges} \
         ranges x {iters} iters, {distinct} distinct){marker}"
    );
    if let Some(h1) = row1 {
        println!("            fnv {h1:016x}  row 1 of {nrow}");
        // A REAL gate, and it lives here because this is the only scope where `distinct` — the
        // predicate it has to be conditioned on — exists.
        //
        // MEASURED 2026-08-09, and this is the second attempt. It began as an unconditional
        // `assert!` in `main`, which panicked the B1 ballast (rc 101, both passes) because a
        // saturated constant residual makes its two rows legitimately equal. The fix was then
        // written as an `if/else` that printed in BOTH branches and could never fail, while its
        // own comment and two docs claimed it was "gated on the same non-degeneracy condition"
        // — it was not gated on anything, because `distinct` never left this function. That is
        // a comment asserting a property the code does not have, this repo's most-cited defect
        // class, caught by review before it was committed. Now it is what it always claimed:
        // silent on a degenerate arm, RED on any other where row 1 equals row 0.
        assert!(
            degenerate || h1 != h,
            "row 1 == row 0 on a NON-degenerate arm ({distinct} distinct): row 1 is reading \
             row 0's activations. `dot_i4_wave_r` accumulates `a0[t]` per `t` with no cross-row \
             term, so equal rows on real data is an indexing bug, not a numerical coincidence."
        );
    }
    (h, row1)
}

fn run_i8(name: &str, o_dim: usize, i_dim: usize) {
    let packed = pattern(o_dim * i_dim, |v| (v * 127.0) as i8 as u8);
    let scale: Vec<f32> = vec![1.0; o_dim];
    let x: Vec<f32> = vec![0.5; i_dim];
    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    let (xp, yp) = (xb.ptr() as *const f32, yb.ptr_mut() as *mut f32);
    let us = time(60, &|| unsafe {
        launch_gemv_i8(xp, pb.ptr(), sb.ptr() as *const f32, o_dim, i_dim, 1, yp).expect("i8");
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
    const SEED: u64 = 0x11A;
    let mut r = Rng(SEED);
    let rows = h * (nope + vh);
    let kvb = dev(&pattern(rows * kvl, |v| f32_to_e4m3(v * 0.1)));
    let sc_cols = kvl.div_ceil(block);
    let kvb_scale = dev(&rnd_scale(&mut r, rows.div_ceil(block) * sc_cols));
    let q = dev(&rnd(&mut r, h * qh));
    let clat = dev(&rnd(&mut r, h * kvl));
    let mut qabs = dev(&vec![0u8; h * kvl * 4]);
    let mut ctx = dev(&vec![0u8; h * vh * 4]);
    let (kp, sp) = (kvb.ptr(), kvb_scale.ptr() as *const f32);
    let (qp, ap) = (q.ptr() as *const f32, qabs.ptr_mut() as *mut f32);
    let (cp, xp) = (clat.ptr() as *const f32, ctx.ptr_mut() as *mut f32);

    let us = time(60, &|| unsafe {
        launch_mla_absorb_fp8(qp, kp, sp, h, qh, nope, vh, kvl, block, 1, ap).expect("absorb");
    });
    let gbs = (h * nope * kvl) as f64 / (us * 1e-6) / 1e9;
    println!("mla_absorb  [{h}x{nope}x{kvl}]   {us:8.1}us  {gbs:6.1} GB/s");
    println!(
        "            fnv {:016x}  (seed {SEED:#x}, {h}x{nope}x{kvl} blk{block})",
        fnv(&qabs.copy_out().expect("out"))
    );

    let us = time(60, &|| unsafe {
        launch_mla_value_fp8(cp, kp, sp, h, nope, vh, kvl, block, 1, xp).expect("value");
    });
    let gbs = (h * vh * kvl) as f64 / (us * 1e-6) / 1e9;
    println!("mla_value   [{h}x{vh}x{kvl}]   {us:8.1}us  {gbs:6.1} GB/s");
    println!(
        "            fnv {:016x}  (seed {SEED:#x}, {h}x{vh}x{kvl} blk{block})",
        fnv(&ctx.copy_out().expect("out"))
    );
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
            launch_attend(
                qa,
                qr,
                lp,
                ls,
                rp,
                std::ptr::null(),
                h,
                nr,
                kvl,
                rope,
                n_blocks,
                0.08,
                cp,
                pp,
            )
            .expect("attend");
        });
        println!("mla_attend  [h{h} nr{nr} kvl{kvl}]  {us:8.1}us");
        // HB and MLA_MIN_TILES_PER_SPLIT both feed `mla_plan_splits`, so a sweep of them
        // is bit-identical only while the PLAN is unchanged — and whether it is cannot be
        // read off the timing. At H=64/nr=512 three of the four (HB, MIN) cells land on
        // n_splits=8 and one lands on 16; this line is what tells them apart. The other
        // rows in this file grew the same fingerprint for the same reason (docs/measurement/benchmarks.md,
        // "A fingerprint is the only instrument that shows bit-identity").
        println!(
            "            fnv {:016x}  (h{h} nr{nr} kvl{kvl} rope{rope})",
            fnv(&clat.copy_out().expect("out"))
        );
    }
}

/// The rest of the `tail` bucket, so it can be attributed by SUBTRACTION rather than
/// assumed. `tail` measures ~16 ms/tok and lm_head was taken to be "almost all" of it;
/// lm_head measures ~8 ms, so ~8 ms belongs to something else. These are the only other
/// ops in the bucket. Neither kernel is modified by this branch — the rows exist to
/// decompose the budget, not to show a before/after.
fn run_tail_rest(vocab: usize, hidden: usize) {
    let logits = dev(&f32b(
        &(0..vocab)
            .map(|i| (i % 977) as f32 * 1e-3)
            .collect::<Vec<_>>(),
    ));
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
        launch_rmsnorm_single(xp, wp, hidden, 1e-5, yp).expect("rmsnorm_single");
    });
    println!("rmsnorm_single [{hidden}]       {us:8.1}us  (single workgroup, dim3(1))");
}

/// `-- mla gemv` runs only those sections; no argument runs all of them, so a per-kernel
/// A/B can book only the rows it is about to compare.
fn main() {
    // `v4gemv` and `v4res` KEEP their model-derived names, deliberately, where the kernels
    // they drive were renamed for behaviour on 2026-08-09 (`gemv_fp8_bf16` and the fp4
    // resident range). These strings are the CLI of a benchmark whose runs are recorded in
    // `docs/measurement/benchmarks.md` and in the git history before its 2026-08-10
    // compaction. Renaming them would make every recorded invocation a lie, and a recorded
    // command that no longer runs cannot be re-run to settle a question — the reason
    // benchmarks.md keeps command forms even though it dropped the per-arm tables. So the
    // inconsistency below is load-bearing, not an oversight. (`glmi4`, 2026-08-10, is the
    // same deal.) See benchmarks.md, "Section tokens recorded rounds invoke".
    const SECTIONS: [&str; 8] = [
        "moe", "gemv", "v4gemv", "v4res", "glmi4", "mla", "attend", "tail",
    ];
    let want: Vec<String> = std::env::args().skip(1).collect();
    // A typo must NOT silently select nothing. Every number this branch rests on — the
    // timings and the fnv fingerprints both — is read off this binary's stdout, so an
    // empty run that exits 0 reads as "no regression" in exactly the A/B it exists for.
    for w in &want {
        assert!(
            SECTIONS.contains(&w.as_str()),
            "unknown section {w:?}, want one of {SECTIONS:?}"
        );
    }
    let on = |s: &str| want.is_empty() || want.iter().any(|w| w == s);

    if on("moe") {
        println!("MoE dot decode throughput (wave-per-row, isolated):");
        run("gate/up", 2048, 6144); // hidden reduction
        run("down", 6144, 2048); // inter reduction
    }
    // The two per-kernel targets in docs/measurement/perf-roadmap.md, at their real engine shapes.
    if on("gemv") {
        println!("\nRoute/tail GEMV bandwidth (real shapes, 256 GB/s peak):");
        run_fp8("o_proj", 6144, 16384); // ~half of `route`; split-K path
        run_i8("lm_head", 154880, 6144); // ~half of `tail`, measured — not all of it
    }
    // V4 shapes from the artifact header M4's byte table was derived from (dim 4096,
    // q_lora 1024, nh*hd 32768, hd 512, gr 8192, gd 4096, moe_inter 2048): wq_a 4.19 MB,
    // wq_b 33.55, wkv 2.10, wo_a 33.55, wo_b 33.55, shared gate/up 8.39 ×2, down 8.39 —
    // a wrong pair here books the wrong denominator into GB/s exactly as run_mla warns.
    if on("v4gemv") {
        println!("\nV4 fp8 GEMV, decode shapes m=1 (serial rate; the M6-kill/M7-unroll kernel):");
        run_v4_gemv("v4 wq_a ", 1024, 4096);
        run_v4_gemv("v4 wq_b ", 32768, 1024);
        run_v4_gemv("v4 wkv  ", 512, 4096);
        run_v4_gemv("v4 wo_a ", 8192, 4096);
        run_v4_gemv("v4 wo_b ", 4096, 8192);
        run_v4_gemv("v4 sh_gu", 2048, 4096);
        run_v4_gemv("v4 sh_dn", 4096, 2048);
    }
    // The fp4 ROUTED-expert path at the engine's resident shape (dim 4096, moe_inter 2048).
    // `e_count = 6` rounds the engine's 5.885 residents/layer UP, and the direction matters:
    // 6 experts launch 12,288 gate/up waves where 5 launch 10,240, and M8 measured grid width
    // as first-order on this box (222 GB/s at full grid against 66 at 512-1024 waves). So
    // probe A's band is being tested on the OPTIMISTIC side, and an `e_count = 5` row is the
    // first thing to add if the band is missed high.
    // Two working sets on purpose — see `run_v4_res`: 14 ranges = 84 experts = 1.12 GB = 35x
    // the 32 MB MALL, against 1 range = ~80 MB, the naive harness §M11 names as the confound
    // that decides whether any of this is true. Both rows draw the same experts 0..6, so
    // their `e_start 0` fingerprints are expected to agree.
    if on("v4res") {
        println!(
            "\nV4 fp4 resident-expert range, nrow=1 (serial rate; the §M11 `res` span). The \
             `mall-ctl` row replays ONE range and is a cache artifact by design:"
        );
        let engine = run_v4_res("v4 res         ", 4096, 2048, 6, 14);
        let ctl = run_v4_res("v4 res mall-ctl", 4096, 2048, 6, 1);
        // The one cross-arm invariant this section can check in-process, and it is not
        // trivial: `ranges` is the ONLY thing that differs between the two calls, `x` and the
        // routing weights come off a `ranges`-independent draw order, and the weight bytes come
        // off a separate fixed generator whose first `e_count` experts are a prefix in both.
        // So a `ranges`-dependent allocation or descriptor slip is exactly what this catches.
        assert_eq!(
            engine, ctl,
            "the two working sets disagree on the [0, e_count) range — a ranges-dependent bug, \
             not a cache effect"
        );
    }
    // GLM-5.2's int4 routed experts at the shape read out of the artifact manifest
    // (`/var/db/rivoli/glm52-vq3-full/manifest.json`: hidden_size 6144,
    // moe_intermediate_size 2048, n_routed_experts 256) — NOT from prose. The per-expert byte
    // count and its check against the artifact are at `run_glm_i4`'s `assert_eq!`.
    //
    // `e_count = 6`: GLM does NOT launch one range per layer the way V4 does. `gpu.rs`
    // batches each RUN of consecutive resident selections among the 9 descriptors (top-8 +
    // shared) and launches each miss singly on the miss stream, so real runs are short —
    // DERIVED, not counted: at the recorded 67.7% decode hit the expected run length is
    // 1/(1 - 0.677) ~= 3.1, so ~1-3. The mechanism is read off `gpu.rs`; the distribution is
    // not, and the engine's selection trace could settle it if it ever matters. The
    // `e1` row below is here because of that — it measures what launch size costs instead of
    // asserting that the grid is saturated either way (at `e_count = 1` gate/up is already
    // `1 x inter` = 2048 waves and down is 6144, against ~1280 machine slots).
    //
    // Two working sets per width on purpose: `ranges = 9` is 54 experts = 1.083 GB = 33x the
    // 32 MB MALL, against `ranges = 1` = 120 MB, the naive harness. Both draw the same
    // experts 0..6 and the same token row 0, so every `e_count = 6` row's fingerprint is
    // expected to agree — across the MALL control AND across R = 1 vs R = 2.
    if on("glmi4") {
        println!(
            "\nGLM-5.2 int4 resident-expert range (serial rate; the shipping default's MoE \
             compute). The `mall-ctl` rows replay ONE range and are a cache artifact by design:"
        );
        // Built in call order, not collected afterwards: each `run_glm_i4` allocates and frees
        // its own ~1 GB working set, so the rows must run one at a time and in this order.
        let fp = [
            (
                "R1         ",
                run_glm_i4("glm i4 R1        ", 6144, 2048, 6, 9, 1),
            ),
            (
                "R1 mall-ctl",
                run_glm_i4("glm i4 R1 mall-ctl", 6144, 2048, 6, 1, 1),
            ),
            (
                "R2         ",
                run_glm_i4("glm i4 R2        ", 6144, 2048, 6, 9, 2),
            ),
            (
                "R2 mall-ctl",
                run_glm_i4("glm i4 R2 mall-ctl", 6144, 2048, 6, 1, 2),
            ),
        ];
        // `e_count = 1`, 50 ranges = 1.003 GB — and its OWN control beside it, because the rule
        // is "a single-range control beside EVERY >= 1 GB rotating row", and this row is one.
        // It shipped without one until the coordinator asked; a rotating row whose control is
        // missing is exactly the confound the two rows above exist to convert into a number.
        //
        // Its control is 20 MB, i.e. BELOW the 32 MB MALL rather than merely near it, so it is
        // the strongest cache-served reading available at this shape — an upper bound on what
        // the MALL can do for this kernel, not just a naive-harness comparison.
        //
        // Both drain ONE expert, so their fingerprints agree with each other and with NEITHER
        // of the `e_count = 6` groups above; that pair is asserted separately below.
        let e1 = run_glm_i4("glm i4 R1 e1     ", 6144, 2048, 1, 50, 1);
        let e1_ctl = run_glm_i4("glm i4 R1 e1 mall-ctl", 6144, 2048, 1, 1, 1);
        for (tag, (h0, _)) in &fp[1..] {
            assert_eq!(
                *h0, fp[0].1.0,
                "{tag} disagrees with R1 on token row 0 — row 0's arithmetic is independent \
                 of `nrow` and of `ranges`, so this is a descriptor/indexing bug or a fold-order \
                 change, not a cache effect"
            );
        }
        // The row-1 equality check that used to sit here is DELETED, 2026-08-09, and the reason
        // is that it could not see the failure it was justified by. It compared row 1 across the
        // two R = 2 arms of the SAME binary, differing only in `ranges` — so an unroll that broke
        // `vt = v + t * v_stride` past the first trip breaks BOTH arms identically and the
        // comparison stays green. What actually caught that class is comparing row-1 fnvs across
        // ARMS (X moved `21842ae3faa86fc0` to `01ff6e0362de32b4`), i.e. the printed value read
        // out of the round's logs, not an in-process assert. The residual it did cover — a
        // `ranges`-dependent slip at `t = 1` — is already caught by the row-0 loop above, since
        // `he = h_in + e * R * inter` moves row 0 too. Row 1 is still fingerprinted and printed,
        // and `run_glm_i4` now asserts row 1 != row 0 on every non-degenerate arm.
        // The `e_count = 1` pair, on the same argument as the pairs above: `ranges` is the only
        // thing that differs, and expert 0's bytes and token row 0's inputs are drawn
        // `n_desc`-independently, so a disagreement here is an indexing bug and not a cache
        // effect. Also pinned NOT to equal the six-expert fingerprint — six experts summing
        // into one accumulator must not produce what one expert does, and an `e_count` that
        // silently failed to reach the kernel would show up exactly there.
        assert_eq!(
            e1.0, e1_ctl.0,
            "the two e_count = 1 working sets disagree on token row 0 — a ranges-dependent bug, \
             not a cache effect"
        );
        assert_ne!(
            e1.0, fp[0].1.0,
            "one expert and six experts drained the same value — `e_count` is not reaching the \
             kernel, which would make every rate row above a 1-expert measurement mislabelled \
             as 6"
        );
    }
    // GLM-5.2 manifest: num_attention_heads 64, qk_head_dim 256 (= qk_nope 192 + rope
    // 64), v_head_dim 256, kv_lora_rank 512. Getting these wrong understates the work:
    // absorb reads H*nope*kvl and value reads H*vh*kvl, so a guessed 128/128 would have
    // measured 4.2 MB where the engine moves 6.3 and 8.4.
    if on("mla") {
        println!("\nMLA kv_b kernels (route):");
        run_mla(64, 256, 192, 256, 512);
    }
    if on("attend") {
        println!("\nMLA flash attention (route; scales with context):");
        run_attend(64, 512, 64);
    }
    if on("tail") {
        println!("\nRest of the tail bucket (attribution by subtraction):");
        run_tail_rest(154880, 6144);
    }
}
