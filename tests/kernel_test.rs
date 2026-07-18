//! M2 gate: the fused MoE expert-batch HIP kernel vs the scalar oracle.
//!
//! The reference is built from the crate's own `matvec_i4` + `silu` (not a
//! reimplementation), so there is no oracle drift — the test locks that the GPU
//! kernel reproduces the reference's int4 dequant → gate/up → silu⊙ → down →
//! weighted-accumulate to within int4 dequant tolerance. Whole file compiles to
//! nothing without the `rocm` feature (the kernel isn't linked in a CPU build).
#![cfg(feature = "rocm")]
// A test binary: expect/panic on setup failure is the correct, readable idiom.
#![allow(clippy::expect_used)]

use rivoli::device::{DeviceBuf, DeviceTier};
use rivoli::hip::{
    ExpertDesc, device_sync, launch_attend, launch_gemv_f32, launch_gemv_i4, launch_gemv_i8,
    launch_mla_absorb, launch_mla_value, launch_moe, launch_rmsnorm, launch_rope,
};
use rivoli::math::{bf16_to_f32, f32_to_bf16, silu, softmax};
use rivoli::quant::{addrow, matvec_f32_bytes, matvec_i4, matvec_i4_rows, matvec_i8, row_bytes};
use rivoli::snapshot::{Int4Matrix, Int8Matrix};

/// Little-endian bytes of an f32 slice (device is LE, matching this host).
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Little-endian bytes of a u16 (bf16) slice.
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// f32 values from little-endian device bytes (kernel output readback).
fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Assert a kernel result matches its scalar reference within int4/f32 tolerance.
fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let max_ref = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let max_err = want
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let tol = 1e-3 * max_ref + 1e-3;
    assert!(
        max_err <= tol,
        "{label}: max_err={max_err:.3e} tol={tol:.3e} (max_ref={max_ref:.3e})"
    );
}

/// Deterministic PRNG (SplitMix-style LCG). No `rand` dependency, and no
/// `Math.random`-style nondeterminism — same seed reproduces the same weights.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform f32 in [-1, 1).
    fn f(&mut self) -> f32 {
        (self.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
    fn nibble(&mut self) -> u8 {
        (self.next_u32() & 0x0F) as u8
    }
}

/// A random per-row int4 matrix `[o_dim, i_dim]`: packed nibbles and the
/// little-endian per-row f32 scale bytes (read both by the `Int4Matrix`
/// reference and, placed as-is, by the kernel via its descriptor).
struct Mat {
    packed: Vec<u8>,
    scale_bytes: Vec<u8>,
}

fn gen_mat(rng: &mut Lcg, o_dim: usize, i_dim: usize) -> Mat {
    let rb = row_bytes(i_dim);
    let mut packed = vec![0u8; o_dim * rb];
    for row in packed.chunks_exact_mut(rb) {
        for i in 0..i_dim {
            let nib = rng.nibble();
            if i & 1 == 0 {
                row[i >> 1] |= nib;
            } else {
                row[i >> 1] |= nib << 4;
            }
        }
    }
    let scale_bytes: Vec<u8> = (0..o_dim)
        .flat_map(|_| (rng.f() * 0.05).to_le_bytes())
        .collect();
    Mat {
        packed,
        scale_bytes,
    }
}

/// Scalar reference for one expert-batch: `Σ_e w[e]·down_e(silu(gate_e·x)⊙up_e·x)`.
/// Uses the crate's `matvec_i4` (the oracle the kernel must match).
#[allow(clippy::too_many_arguments)]
fn reference(
    x: &[f32],
    hidden: usize,
    inter: usize,
    gates: &[Mat],
    ups: &[Mat],
    downs: &[Mat],
    w: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    let mut gate = vec![0.0f32; inter];
    let mut up = vec![0.0f32; inter];
    let mut h = vec![0.0f32; inter];
    let mut down = vec![0.0f32; hidden];
    for e in 0..w.len() {
        let gw = Int4Matrix {
            packed: &gates[e].packed,
            scale: &gates[e].scale_bytes,
            o_dim: inter,
            i_dim: hidden,
        };
        let uw = Int4Matrix {
            packed: &ups[e].packed,
            scale: &ups[e].scale_bytes,
            o_dim: inter,
            i_dim: hidden,
        };
        let dw = Int4Matrix {
            packed: &downs[e].packed,
            scale: &downs[e].scale_bytes,
            o_dim: hidden,
            i_dim: inter,
        };
        matvec_i4(&mut gate, x, &gw);
        matvec_i4(&mut up, x, &uw);
        for j in 0..inter {
            h[j] = silu(gate[j]) * up[j];
        }
        matvec_i4(&mut down, &h, &dw);
        for o in 0..hidden {
            out[o] += w[e] * down[o];
        }
    }
    out
}

fn check(seed: u64, hidden: usize, inter: usize, e: usize) {
    let mut rng = Lcg(seed);
    let x: Vec<f32> = (0..hidden).map(|_| rng.f()).collect();
    let gates: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, inter, hidden)).collect();
    let ups: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, inter, hidden)).collect();
    let downs: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, hidden, inter)).collect();
    let w: Vec<f32> = (0..e).map(|_| rng.f()).collect();

    let want = reference(&x, hidden, inter, &gates, &ups, &downs, &w);

    // Place every expert's weights in the resident tier and address them by
    // device pointer through a descriptor array — the D1 zero-copy path.
    let mut tier = DeviceTier::new(256 << 20).expect("alloc tier");
    let descs: Vec<ExpertDesc> = (0..e)
        .map(|i| {
            let gp = tier.place(&gates[i].packed).expect("place gate");
            let gs = tier.place(&gates[i].scale_bytes).expect("place gate scale");
            let up = tier.place(&ups[i].packed).expect("place up");
            let us = tier.place(&ups[i].scale_bytes).expect("place up scale");
            let dp = tier.place(&downs[i].packed).expect("place down");
            let ds = tier.place(&downs[i].scale_bytes).expect("place down scale");
            ExpertDesc {
                gate_packed: gp,
                gate_scale: gs as *const f32,
                up_packed: up,
                up_scale: us as *const f32,
                down_packed: dp,
                down_scale: ds as *const f32,
            }
        })
        .collect();

    // Descriptor array, x, per-expert weights, and the accumulator all device-side.
    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            std::mem::size_of_val(&descs[..]),
        )
    };
    let descs_buf = DeviceBuf::from_bytes(desc_bytes).expect("place descs");
    let x_buf = DeviceBuf::from_bytes(&f32_bytes(&x)).expect("place x");
    let w_buf = DeviceBuf::from_bytes(&f32_bytes(&w)).expect("place w");
    let mut out_buf = DeviceBuf::zeroed(hidden * 4).expect("alloc out");

    // SAFETY: all pointers are device-resident for the dims; out is zeroed.
    unsafe {
        launch_moe(
            x_buf.ptr() as *const f32,
            hidden,
            inter,
            e,
            descs_buf.ptr() as *const ExpertDesc,
            w_buf.ptr() as *const f32,
            out_buf.ptr_mut() as *mut f32,
        )
        .expect("launch moe");
    }
    device_sync().expect("device sync");
    let got: Vec<f32> = out_buf
        .copy_out()
        .expect("copy out")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let max_ref = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let max_err = want
        .iter()
        .zip(&got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    // Per-row int4 dot products sum in identical order to the reference, so the
    // only sources of drift are silu(expf) and cross-expert atomicAdd ordering:
    // ~1e-6 relative. A generous bound still catches any real dequant/index bug.
    let tol = 1e-3 * max_ref + 1e-3;
    assert!(
        max_err <= tol,
        "hidden={hidden} inter={inter} e={e}: max_err={max_err:.3e} tol={tol:.3e} (max_ref={max_ref:.3e})"
    );
}

#[test]
fn moe_batch_matches_scalar_realistic() {
    // GLM-shaped-ish: wide hidden, top-8 routed batch.
    check(0x1234_5678, 2048, 768, 8);
}

#[test]
fn moe_batch_matches_scalar_odd_dims() {
    // Odd hidden AND odd inter exercise the row-byte ceil + tail nibble on both
    // the gate/up (i_dim=hidden) and down (i_dim=inter) projections.
    check(0x9e37_79b9, 129, 65, 3);
}

#[test]
fn moe_batch_single_expert() {
    // Batch of one: isolates the fused path from cross-expert accumulation.
    check(0xdead_beef, 512, 256, 1);
}

/// Scalar reference for the MLA latent attention core, mirroring attn.rs: score
/// each cached token (kvl dot then rope dot, bf16 widened), two-pass softmax,
/// weighted sum of latents. The kernel's flash online-softmax must reproduce it.
#[allow(clippy::too_many_arguments)]
fn attend_reference(
    qabs: &[f32],
    qrope: &[f32],
    lc: &[u16],
    rc: &[u16],
    h: usize,
    nt: usize,
    kvl: usize,
    rope: usize,
    scale: f32,
) -> Vec<f32> {
    let mut clat = vec![0.0f32; h * kvl];
    let mut scores = vec![0.0f32; nt];
    for head in 0..h {
        let qa = &qabs[head * kvl..(head + 1) * kvl];
        let qr = &qrope[head * rope..(head + 1) * rope];
        for (t, sc) in scores.iter_mut().enumerate() {
            let lrow = &lc[t * kvl..(t + 1) * kvl];
            let rrow = &rc[t * rope..(t + 1) * rope];
            let mut a = 0.0f32;
            for (i, &lb) in lrow.iter().enumerate() {
                a += qa[i] * bf16_to_f32(lb);
            }
            for (d, &rb) in rrow.iter().enumerate() {
                a += qr[d] * bf16_to_f32(rb);
            }
            *sc = a * scale;
        }
        softmax(&mut scores);
        let out = &mut clat[head * kvl..(head + 1) * kvl];
        for (t, &sc) in scores.iter().enumerate() {
            let lrow = &lc[t * kvl..(t + 1) * kvl];
            for (i, &lb) in lrow.iter().enumerate() {
                out[i] += sc * bf16_to_f32(lb);
            }
        }
    }
    clat
}

fn check_attend(seed: u64, h: usize, nt: usize, kvl: usize, rope: usize) {
    let mut rng = Lcg(seed);
    let qabs: Vec<f32> = (0..h * kvl).map(|_| rng.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| rng.f()).collect();
    // Cache values round-tripped through bf16 so kernel and reference widen the
    // exact same u16 bits — the only algorithmic difference under test is the
    // flash (online) vs two-pass softmax.
    let lc: Vec<u16> = (0..nt * kvl).map(|_| f32_to_bf16(rng.f())).collect();
    let rc: Vec<u16> = (0..nt * rope).map(|_| f32_to_bf16(rng.f())).collect();
    let scale = 1.0 / ((kvl + rope) as f32).sqrt();

    let want = attend_reference(&qabs, &qrope, &lc, &rc, h, nt, kvl, rope, scale);

    // Device-resident query + KV cache; the kernel reads them in place.
    let qabs_buf = DeviceBuf::from_bytes(&f32_bytes(&qabs)).expect("place qabs");
    let qrope_buf = DeviceBuf::from_bytes(&f32_bytes(&qrope)).expect("place qrope");
    let lc_buf = DeviceBuf::from_bytes(&u16_bytes(&lc)).expect("place lc");
    let rc_buf = DeviceBuf::from_bytes(&u16_bytes(&rc)).expect("place rc");
    let mut clat_buf = DeviceBuf::zeroed(h * kvl * 4).expect("alloc clat");

    // SAFETY: all pointers are device-resident for the dims.
    unsafe {
        launch_attend(
            qabs_buf.ptr() as *const f32,
            qrope_buf.ptr() as *const f32,
            lc_buf.ptr() as *const u16,
            rc_buf.ptr() as *const u16,
            h,
            nt,
            kvl,
            rope,
            scale,
            clat_buf.ptr_mut() as *mut f32,
        )
        .expect("launch attend");
    }
    device_sync().expect("device sync");
    let got: Vec<f32> = clat_buf
        .copy_out()
        .expect("copy out")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    let max_ref = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let max_err = want
        .iter()
        .zip(&got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let tol = 1e-3 * max_ref + 1e-4;
    assert!(
        max_err <= tol,
        "h={h} nt={nt} kvl={kvl} rope={rope}: max_err={max_err:.3e} tol={tol:.3e} (max_ref={max_ref:.3e})"
    );
}

#[test]
fn mla_attend_matches_scalar_glm_dims() {
    // GLM MLA: kv_lora=512, qk_rope=64, a few hundred cached tokens.
    check_attend(0x0a5e_1102, 16, 300, 512, 64);
}

#[test]
fn mla_attend_matches_scalar_tiny() {
    // Small/odd: single cached token and short context stress the online-softmax
    // init (m=-inf on the first token) and tail behavior.
    check_attend(0x00c0_ffee, 4, 1, 32, 8);
    check_attend(0xfeed_face, 3, 17, 40, 8);
}

#[test]
fn mla_attend_head_tiling_boundaries() {
    // Real GLM head count (64 = 8 full HB=8 blocks), and H=20 → a partial second
    // block (4 active + 4 inactive lanes) exercising the head<H guard. nt spans
    // many TILE=16 steps.
    check_attend(0xb10c_c0de, 64, 130, 512, 64);
    check_attend(0x2020_2020, 20, 96, 512, 64);
}

/// Random int8 matrix `[o, i]`: signed bytes + per-row f32 scale bytes.
fn gen_i8(rng: &mut Lcg, o: usize, i: usize) -> (Vec<u8>, Vec<u8>) {
    let packed: Vec<u8> = (0..o * i).map(|_| (rng.next_u32() & 0xff) as u8).collect();
    let scale: Vec<u8> = (0..o)
        .flat_map(|_| (rng.f() * 0.05).to_le_bytes())
        .collect();
    (packed, scale)
}

fn check_gemv_i4(seed: u64, o: usize, i: usize) {
    let mut rng = Lcg(seed);
    let m = gen_mat(&mut rng, o, i);
    let x: Vec<f32> = (0..i).map(|_| rng.f()).collect();
    let mut want = vec![0.0f32; o];
    matvec_i4(
        &mut want,
        &x,
        &Int4Matrix {
            packed: &m.packed,
            scale: &m.scale_bytes,
            o_dim: o,
            i_dim: i,
        },
    );

    let mut tier = DeviceTier::new(64 << 20).expect("alloc tier");
    let pp = tier.place(&m.packed).expect("place packed");
    let ps = tier.place(&m.scale_bytes).expect("place scale");
    let x_buf = DeviceBuf::from_bytes(&f32_bytes(&x)).expect("place x");
    let mut y_buf = DeviceBuf::zeroed(o * 4).expect("alloc y");
    // SAFETY: device pointers valid for the dims; y outlives the sync.
    unsafe {
        launch_gemv_i4(
            x_buf.ptr() as *const f32,
            pp,
            ps as *const f32,
            o,
            i,
            y_buf.ptr_mut() as *mut f32,
        )
        .expect("launch gemv_i4");
    }
    device_sync().expect("device sync");
    assert_close(
        &want,
        &f32_vec(&y_buf.copy_out().expect("copy out")),
        "gemv_i4",
    );
}

#[test]
fn gemv_i4_matches_scalar() {
    check_gemv_i4(0x0991_1337, 1536, 2048); // even i_dim
    check_gemv_i4(0x0991_1338, 100, 257); // odd i_dim → last byte's high nibble is pad
}

#[test]
fn gemv_f32_matches_scalar() {
    // MoE router gate shape: o_dim = n_experts, i_dim = hidden.
    let mut rng = Lcg(0x0f32_1337);
    let (o, i) = (256usize, 6144usize);
    let w: Vec<f32> = (0..o * i).map(|_| rng.f() * 0.1).collect();
    let x: Vec<f32> = (0..i).map(|_| rng.f()).collect();
    let w_bytes = f32_bytes(&w);
    let mut want = vec![0.0f32; o];
    matvec_f32_bytes(&mut want, &x, &w_bytes, i);

    let w_buf = DeviceBuf::from_bytes(&w_bytes).expect("place w");
    let x_buf = DeviceBuf::from_bytes(&f32_bytes(&x)).expect("place x");
    let mut y_buf = DeviceBuf::zeroed(o * 4).expect("alloc y");
    // SAFETY: device pointers valid for the dims; y outlives the sync.
    unsafe {
        launch_gemv_f32(
            x_buf.ptr() as *const f32,
            w_buf.ptr() as *const f32,
            o,
            i,
            y_buf.ptr_mut() as *mut f32,
        )
        .expect("launch gemv_f32");
    }
    device_sync().expect("device sync");
    assert_close(
        &want,
        &f32_vec(&y_buf.copy_out().expect("copy out")),
        "gemv_f32",
    );
}

#[test]
fn gemv_i8_matches_scalar() {
    let mut rng = Lcg(0x0088_1337);
    let (o, i) = (2048usize, 1024usize);
    let (packed, scale) = gen_i8(&mut rng, o, i);
    let x: Vec<f32> = (0..i).map(|_| rng.f()).collect();
    let mut want = vec![0.0f32; o];
    matvec_i8(
        &mut want,
        &x,
        &Int8Matrix {
            packed: &packed,
            scale: &scale,
            o_dim: o,
            i_dim: i,
        },
    );

    let mut tier = DeviceTier::new(64 << 20).expect("alloc tier");
    let pp = tier.place(&packed).expect("place packed");
    let ps = tier.place(&scale).expect("place scale");
    let x_buf = DeviceBuf::from_bytes(&f32_bytes(&x)).expect("place x");
    let mut y_buf = DeviceBuf::zeroed(o * 4).expect("alloc y");
    // SAFETY: device pointers valid for the dims; y outlives the sync.
    unsafe {
        launch_gemv_i8(
            x_buf.ptr() as *const f32,
            pp,
            ps as *const f32,
            o,
            i,
            y_buf.ptr_mut() as *mut f32,
        )
        .expect("launch gemv_i8");
    }
    device_sync().expect("device sync");
    assert_close(
        &want,
        &f32_vec(&y_buf.copy_out().expect("copy out")),
        "gemv_i8",
    );
}

fn check_mla(seed: u64, h: usize, qh: usize, nope: usize, vh: usize, kvl: usize) {
    // MLA-ish dims: kv_b [H*(nope+vh), kvl]; q [H*qh]; clat [H*kvl].
    let mut rng = Lcg(seed);
    let kvb = gen_mat(&mut rng, h * (nope + vh), kvl);
    let q: Vec<f32> = (0..h * qh).map(|_| rng.f()).collect();
    let clat: Vec<f32> = (0..h * kvl).map(|_| rng.f()).collect();
    let kvb_m = Int4Matrix {
        packed: &kvb.packed,
        scale: &kvb.scale_bytes,
        o_dim: h * (nope + vh),
        i_dim: kvl,
    };

    // Scalar oracle: absorb via addrow, value via matvec_i4_rows (attn.rs core).
    let mut qabs_ref = vec![0.0f32; h * kvl];
    let mut ctx_ref = vec![0.0f32; h * vh];
    for head in 0..h {
        let rbase = head * (nope + vh);
        let qnope = &q[head * qh..head * qh + nope];
        let seg = &mut qabs_ref[head * kvl..(head + 1) * kvl];
        for (d, &qd) in qnope.iter().enumerate() {
            addrow(&kvb_m, rbase + d, qd, seg);
        }
        matvec_i4_rows(
            &mut ctx_ref[head * vh..(head + 1) * vh],
            &clat[head * kvl..(head + 1) * kvl],
            &kvb_m,
            rbase + nope,
        );
    }

    let mut tier = DeviceTier::new(64 << 20).expect("alloc tier");
    let kp = tier.place(&kvb.packed).expect("place kv_b packed");
    let ks = tier.place(&kvb.scale_bytes).expect("place kv_b scale") as *const f32;
    let q_buf = DeviceBuf::from_bytes(&f32_bytes(&q)).expect("place q");
    let clat_buf = DeviceBuf::from_bytes(&f32_bytes(&clat)).expect("place clat");
    let mut qabs_buf = DeviceBuf::zeroed(h * kvl * 4).expect("alloc qabs");
    let mut ctx_buf = DeviceBuf::zeroed(h * vh * 4).expect("alloc ctx");

    // SAFETY: device pointers valid for the dims; outputs outlive the sync.
    unsafe {
        launch_mla_absorb(
            q_buf.ptr() as *const f32,
            kp,
            ks,
            h,
            qh,
            nope,
            vh,
            kvl,
            qabs_buf.ptr_mut() as *mut f32,
        )
        .expect("launch absorb");
        launch_mla_value(
            clat_buf.ptr() as *const f32,
            kp,
            ks,
            h,
            nope,
            vh,
            kvl,
            ctx_buf.ptr_mut() as *mut f32,
        )
        .expect("launch value");
    }
    device_sync().expect("device sync");
    assert_close(
        &qabs_ref,
        &f32_vec(&qabs_buf.copy_out().expect("copy qabs")),
        "mla_absorb",
    );
    assert_close(
        &ctx_ref,
        &f32_vec(&ctx_buf.copy_out().expect("copy ctx")),
        "mla_value",
    );
}

#[test]
fn mla_absorb_and_value_match_scalar() {
    // nope≠vh so a value-row offset swap (rbase+nope vs rbase+vh) can't hide;
    // qh≠nope covers the absorb stride. Plus an odd-kvl case for the rb ceil.
    check_mla(0x3c0f_fee5, 8, 192, 128, 96, 512);
    check_mla(0x0dd_c0de5, 4, 96, 64, 48, 129);
}

#[test]
fn rmsnorm_matches_scalar() {
    let mut rng = Lcg(0x5551_0001);
    let n = 6144usize;
    let x: Vec<f32> = (0..n).map(|_| rng.f() * 3.0).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.f()).collect();
    let mut want = x.clone();
    rivoli::math::rmsnorm(&mut want, &w, 1e-5);

    let x_buf = DeviceBuf::from_bytes(&f32_bytes(&x)).expect("place x");
    let w_buf = DeviceBuf::from_bytes(&f32_bytes(&w)).expect("place w");
    let mut y_buf = DeviceBuf::zeroed(n * 4).expect("alloc y");
    // SAFETY: device pointers valid for n; y outlives the sync.
    unsafe {
        launch_rmsnorm(
            x_buf.ptr() as *const f32,
            w_buf.ptr() as *const f32,
            n,
            1e-5,
            y_buf.ptr_mut() as *mut f32,
        )
        .expect("launch rmsnorm");
    }
    device_sync().expect("device sync");
    assert_close(
        &want,
        &f32_vec(&y_buf.copy_out().expect("copy out")),
        "rmsnorm",
    );
}

#[test]
fn rope_matches_scalar() {
    // Batched per-head rope: H segments of `seg` at stride `qh`, rope applied to
    // the [nope, nope+seg) slice of each head (as in attention). Compare to
    // attn::rope_interleave on each segment.
    let mut rng = Lcg(0x600d_600d);
    let (h, qh, nope, seg, pos) = (16usize, 192usize, 128usize, 64usize, 137usize);
    let theta = 8_000_000.0f64;
    let mut q: Vec<f32> = (0..h * qh).map(|_| rng.f()).collect();

    let mut want = q.clone();
    for head in 0..h {
        let off = head * qh + nope;
        rivoli::attn::rope_interleave(&mut want[off..off + seg], pos, theta);
    }

    let mut q_buf = DeviceBuf::from_bytes(&f32_bytes(&q)).expect("place q");
    // base points at the first head's rope segment; stride qh strides heads.
    // SAFETY: base+count*stride within the h*qh buffer; outlives the sync.
    unsafe {
        let base = (q_buf.ptr_mut() as *mut f32).add(nope);
        launch_rope(base, h, qh, seg, pos, theta).expect("launch rope");
    }
    device_sync().expect("device sync");
    q = f32_vec(&q_buf.copy_out().expect("copy out"));
    assert_close(&want, &q, "rope");
}

#[test]
fn mla_attend_long_context() {
    // On-device tile loop at scale: 40k tokens = ~2500 TILE=16 steps, so the
    // per-lane bit-exact online-softmax replication is OBSERVED over thousands of
    // iterations (not just proven), at the GLM latent width.
    check_attend(0x1eaf_1eaf, 8, 40_000, 512, 64);
}
