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
use rivoli::hip::{ExpertDesc, device_sync, launch_attend, launch_moe};
use rivoli::math::{bf16_to_f32, f32_to_bf16, silu, softmax};
use rivoli::quant::{matvec_i4, row_bytes};
use rivoli::snapshot::Int4Matrix;

/// Little-endian bytes of an f32 slice (device is LE, matching this host).
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Little-endian bytes of a u16 (bf16) slice.
fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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
