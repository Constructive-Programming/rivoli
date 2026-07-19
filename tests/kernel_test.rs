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

/// Test helper: place synthetic bytes into the tier (no shard file to `pread`
/// from), filling the device-local host-mapped slab in place.
fn place_bytes(tier: &mut DeviceTier, bytes: &[u8]) -> *const u8 {
    let dst = tier.reserve(bytes.len()).expect("reserve");
    // SAFETY: dst owns bytes.len() host-writable bytes just reserved.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len()) };
    dst as *const u8
}

/// Test helper: a fresh device buffer filled with `bytes`. Production `DeviceBuf`
/// exposes `new` + `copy_in_at`; the old `from_bytes`/`zeroed` conveniences were
/// removed as dead in production, so the tests build the equivalents here.
fn dev_bytes(bytes: &[u8]) -> DeviceBuf {
    let mut b = DeviceBuf::new(bytes.len()).expect("alloc dev buf");
    b.copy_in_at(0, bytes).expect("fill dev buf");
    b
}

/// Test helper: a zero-initialized device buffer of `len` bytes.
fn dev_zeroed(len: usize) -> DeviceBuf {
    dev_bytes(&vec![0u8; len])
}
use rivoli::hip::{
    ExpertDesc, device_sync, launch_attend, launch_gemv_bf16, launch_gemv_f32, launch_gemv_i4,
    launch_gemv_i8, launch_index_head_route, launch_index_pool_push, launch_index_score,
    launch_layernorm, launch_mla_absorb, launch_mla_value, launch_moe, launch_moe_batched,
    launch_rmsnorm, launch_rope,
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

/// Little-endian bytes of a u32 slice (row/selection uploads).
fn u32_bytes(v: &[u32]) -> Vec<u8> {
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
            let gp = place_bytes(&mut tier, &gates[i].packed);
            let gs = place_bytes(&mut tier, &gates[i].scale_bytes);
            let up = place_bytes(&mut tier, &ups[i].packed);
            let us = place_bytes(&mut tier, &ups[i].scale_bytes);
            let dp = place_bytes(&mut tier, &downs[i].packed);
            let ds = place_bytes(&mut tier, &downs[i].scale_bytes);
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
    let descs_buf = dev_bytes(desc_bytes);
    let x_buf = dev_bytes(&f32_bytes(&x));
    let w_buf = dev_bytes(&f32_bytes(&w));
    let mut h_buf = DeviceBuf::new(e * inter * 4).expect("alloc h");
    let mut partial_buf = DeviceBuf::new(e * hidden * 4).expect("alloc partial");
    let mut out_buf = DeviceBuf::new(hidden * 4).expect("alloc out");

    // SAFETY: all pointers are device-resident for the dims; h/partial/out are
    // fully written by the two passes + reduce (no pre-zero needed).
    unsafe {
        launch_moe(
            x_buf.ptr() as *const f32,
            hidden,
            inter,
            e,
            descs_buf.ptr() as *const ExpertDesc,
            w_buf.ptr() as *const f32,
            h_buf.ptr_mut() as *mut f32,
            partial_buf.ptr_mut() as *mut f32,
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

/// Batched MoE (S positions over the UNION of e experts) must equal S separate
/// S=1 MoE calls: position 0 weights experts [0,e/2), position 1 weights
/// [e/2,e) (each zero elsewhere), so the batched kernel — which applies every
/// union expert to both positions but zeroes the non-selected weights — matches
/// two independent `reference()` outputs.
fn check_moe_batched(seed: u64, hidden: usize, inter: usize, e: usize) {
    let mut rng = Lcg(seed);
    let x0: Vec<f32> = (0..hidden).map(|_| rng.f()).collect();
    let x1: Vec<f32> = (0..hidden).map(|_| rng.f()).collect();
    let gates: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, inter, hidden)).collect();
    let ups: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, inter, hidden)).collect();
    let downs: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, hidden, inter)).collect();
    // Per-position weights: position 0 selects the first half, position 1 the
    // second half (0 elsewhere) — exercises the union + per-position weighting.
    let half = e / 2;
    let w0: Vec<f32> = (0..e)
        .map(|i| if i < half { rng.f() } else { 0.0 })
        .collect();
    let w1: Vec<f32> = (0..e)
        .map(|i| if i >= half { rng.f() } else { 0.0 })
        .collect();

    let want0 = reference(&x0, hidden, inter, &gates, &ups, &downs, &w0);
    let want1 = reference(&x1, hidden, inter, &gates, &ups, &downs, &w1);

    let mut tier = DeviceTier::new(256 << 20).expect("alloc tier");
    let descs: Vec<ExpertDesc> = (0..e)
        .map(|i| {
            let gp = place_bytes(&mut tier, &gates[i].packed);
            let gs = place_bytes(&mut tier, &gates[i].scale_bytes);
            let up = place_bytes(&mut tier, &ups[i].packed);
            let us = place_bytes(&mut tier, &ups[i].scale_bytes);
            let dp = place_bytes(&mut tier, &downs[i].packed);
            let ds = place_bytes(&mut tier, &downs[i].scale_bytes);
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
    let desc_bytes = unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            std::mem::size_of_val(&descs[..]),
        )
    };
    let descs_buf = dev_bytes(desc_bytes);
    // x = [x0 | x1]; wexpert = [w0 | w1] (S*e).
    let mut xcat = f32_bytes(&x0);
    xcat.extend(f32_bytes(&x1));
    let mut wcat = f32_bytes(&w0);
    wcat.extend(f32_bytes(&w1));
    let x_buf = dev_bytes(&xcat);
    let w_buf = dev_bytes(&wcat);
    let mut h_buf = DeviceBuf::new(2 * e * inter * 4).expect("alloc h");
    let mut partial_buf = DeviceBuf::new(2 * e * hidden * 4).expect("alloc partial");
    let mut out_buf = DeviceBuf::new(2 * hidden * 4).expect("alloc out");
    // SAFETY: device buffers sized for S=2; kernels fully write h/partial/out.
    unsafe {
        launch_moe_batched(
            x_buf.ptr() as *const f32,
            hidden,
            inter,
            e,
            2,
            descs_buf.ptr() as *const ExpertDesc,
            w_buf.ptr() as *const f32,
            h_buf.ptr_mut() as *mut f32,
            partial_buf.ptr_mut() as *mut f32,
            out_buf.ptr_mut() as *mut f32,
        )
        .expect("launch moe_batched");
    }
    device_sync().expect("device sync");
    let got = f32_vec(&out_buf.copy_out().expect("copy out"));
    let mut want = want0.clone();
    want.extend(&want1);
    assert_close(
        &want,
        &got,
        &format!("moe_batched hidden={hidden} inter={inter} e={e}"),
    );
}

#[test]
fn moe_batched_matches_two_s1() {
    // Realistic GLM MoE dims, e experts split across the two positions.
    check_moe_batched(0xb47c_4ed0, 6144, 2048, 8);
    check_moe_batched(0x0002_0002, 512, 256, 4);
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
    rows: &[u32],
    kvl: usize,
    rope: usize,
    scale: f32,
) -> Vec<f32> {
    let mut clat = vec![0.0f32; h * kvl];
    let mut scores = vec![0.0f32; rows.len()];
    for head in 0..h {
        let qa = &qabs[head * kvl..(head + 1) * kvl];
        let qr = &qrope[head * rope..(head + 1) * rope];
        for (&r, sc) in rows.iter().zip(scores.iter_mut()) {
            let t = r as usize;
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
        for (&r, &sc) in rows.iter().zip(scores.iter()) {
            let t = r as usize;
            let lrow = &lc[t * kvl..(t + 1) * kvl];
            for (i, &lb) in lrow.iter().enumerate() {
                out[i] += sc * bf16_to_f32(lb);
            }
        }
    }
    clat
}

/// `sel` = None → dense over all `nt` rows (kernel gets a null rows pointer);
/// Some(rows) → sparse gather of those rows (kernel reads them from a device
/// buffer). The scalar reference always iterates the effective row list, so
/// both paths check the same math.
fn check_attend_sel(seed: u64, h: usize, nt: usize, kvl: usize, rope: usize, sel: Option<&[u32]>) {
    let mut rng = Lcg(seed);
    let qabs: Vec<f32> = (0..h * kvl).map(|_| rng.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| rng.f()).collect();
    // Cache values round-tripped through bf16 so kernel and reference widen the
    // exact same u16 bits — the only algorithmic difference under test is the
    // flash (online) vs two-pass softmax.
    let lc: Vec<u16> = (0..nt * kvl).map(|_| f32_to_bf16(rng.f())).collect();
    let rc: Vec<u16> = (0..nt * rope).map(|_| f32_to_bf16(rng.f())).collect();
    let scale = 1.0 / ((kvl + rope) as f32).sqrt();

    let dense: Vec<u32> = (0..nt as u32).collect();
    let rows = sel.unwrap_or(&dense);
    let want = attend_reference(&qabs, &qrope, &lc, &rc, h, rows, kvl, rope, scale);

    // Device-resident query + KV cache; the kernel reads them in place.
    let qabs_buf = dev_bytes(&f32_bytes(&qabs));
    let qrope_buf = dev_bytes(&f32_bytes(&qrope));
    let lc_buf = dev_bytes(&u16_bytes(&lc));
    let rc_buf = dev_bytes(&u16_bytes(&rc));
    let mut clat_buf = dev_zeroed(h * kvl * 4);
    let rows_buf = dev_bytes(&u32_bytes(rows)); // kept alive even on the dense path
    let rows_ptr = if sel.is_some() {
        rows_buf.ptr() as *const u32
    } else {
        std::ptr::null()
    };

    // SAFETY: all pointers are device-resident for the dims.
    unsafe {
        launch_attend(
            qabs_buf.ptr() as *const f32,
            qrope_buf.ptr() as *const f32,
            lc_buf.ptr() as *const u16,
            rc_buf.ptr() as *const u16,
            rows_ptr,
            h,
            rows.len(),
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
        "h={h} nt={nt} kvl={kvl} rope={rope} nr={}: max_err={max_err:.3e} tol={tol:.3e} (max_ref={max_ref:.3e})",
        rows.len()
    );
}

fn check_attend(seed: u64, h: usize, nt: usize, kvl: usize, rope: usize) {
    check_attend_sel(seed, h, nt, kvl, rope, None);
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

#[test]
fn mla_attend_gather_matches_scalar() {
    // Sparse gather at GLM dims: a scattered subset (every 3rd row + the last,
    // non-contiguous strides through the slab) and a streaming-shaped subset
    // (sinks 0..4 + trailing window) — both against the row-list reference.
    // Rows counts straddle TILE=16 boundaries (101 and 36).
    let scattered: Vec<u32> = (0..300u32).step_by(3).chain([299]).collect();
    check_attend_sel(0x5ca7_7e8d, 64, 300, 512, 64, Some(&scattered));
    let streaming: Vec<u32> = (0..4u32).chain(268..300).collect();
    check_attend_sel(0x51de_ca8e, 20, 300, 512, 64, Some(&streaming));
    // Single selected row (the degenerate softmax-of-one path).
    check_attend_sel(0x0000_0001, 4, 50, 32, 8, Some(&[49]));
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
    let pp = place_bytes(&mut tier, &m.packed);
    let ps = place_bytes(&mut tier, &m.scale_bytes);
    let x_buf = dev_bytes(&f32_bytes(&x));
    let mut y_buf = dev_zeroed(o * 4);
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

    let w_buf = dev_bytes(&w_bytes);
    let x_buf = dev_bytes(&f32_bytes(&x));
    let mut y_buf = dev_zeroed(o * 4);
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
    let pp = place_bytes(&mut tier, &packed);
    let ps = place_bytes(&mut tier, &scale);
    let x_buf = dev_bytes(&f32_bytes(&x));
    let mut y_buf = dev_zeroed(o * 4);
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
    let kp = place_bytes(&mut tier, &kvb.packed);
    let ks = place_bytes(&mut tier, &kvb.scale_bytes) as *const f32;
    let q_buf = dev_bytes(&f32_bytes(&q));
    let clat_buf = dev_bytes(&f32_bytes(&clat));
    let mut qabs_buf = dev_zeroed(h * kvl * 4);
    let mut ctx_buf = dev_zeroed(h * vh * 4);

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

    let x_buf = dev_bytes(&f32_bytes(&x));
    let w_buf = dev_bytes(&f32_bytes(&w));
    let mut y_buf = dev_zeroed(n * 4);
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

    let mut q_buf = dev_bytes(&f32_bytes(&q));
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

#[test]
fn argmax_matches_host_fold() {
    use rivoli::hip::launch_argmax;
    // The EXACT host fold from gpu.rs::argmax: best=0, bv=-inf; strict `>` so ties
    // keep the first index and NaN never wins. Returns (index, winning value).
    fn host(logits: &[f32]) -> (i32, f32) {
        let mut best = 0i32;
        let mut bv = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > bv {
                bv = l;
                best = i as i32;
            }
        }
        (best, bv)
    }
    let mut rng = Lcg(0x0a11_5eed);
    let mut cases: Vec<Vec<f32>> = vec![
        vec![1.0],                     // single element
        vec![-3.0, -3.0, -3.0],        // all tie -> lowest index 0
        vec![0.5, 2.0, 2.0, 1.0],      // tie at max -> first (index 1)
        vec![-1.0, -5.0, -0.25, -9.0], // negatives
        vec![0.1, 0.2, f32::NAN, 0.4], // NaN interior, finite max 0.4@3
        vec![f32::NAN, 5.0, 3.0],      // NaN at 0, max 5@1
    ];
    // A vocab-scale vector: exercises the grid-stride loop + full tree reduce, with
    // many ties (both host and device must pick the LOWEST index of the max).
    cases.push(
        (0..154_880usize)
            .map(|i| (i.wrapping_mul(2_654_435_761) & 0xffff) as f32)
            .collect(),
    );
    // A random vector with a planted unique max.
    cases.push({
        let mut v: Vec<f32> = (0..1000).map(|_| rng.f()).collect();
        v[737] = 9.9;
        v
    });

    for (ci, logits) in cases.iter().enumerate() {
        let (wi, wv) = host(logits);
        let mut lbuf = dev_bytes(&f32_bytes(logits));
        let mut out = DeviceBuf::new(8).expect("alloc argmax out");
        // SAFETY: lbuf holds logits.len() f32; out holds 8 bytes [i32 idx|f32 val].
        unsafe {
            launch_argmax(
                lbuf.ptr_mut() as *const f32,
                logits.len(),
                out.ptr_mut() as *mut i32,
                out.ptr_mut().add(4) as *mut f32,
            )
            .expect("launch argmax");
        }
        device_sync().expect("device sync");
        let bytes = out.copy_out().expect("copy out");
        let gi = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let gv = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        assert_eq!(gi, wi, "case {ci}: device index != host fold");
        // Value bits identical: the reduce only compares + copies logits, no math.
        assert_eq!(
            gv.to_bits(),
            wv.to_bits(),
            "case {ci}: device value bits differ"
        );
    }
}

// --- DSA lightning-indexer device kernels vs the scalar oracle (src/indexer.rs) ---

/// gemv_bf16: y[o] = Σ_i x[i]·bf16(w[o·i_dim+i]). Reference widens the same u16
/// bits the kernel does, so the only difference under test is reduction order.
#[test]
fn gemv_bf16_matches_scalar() {
    let mut rng = Lcg(0x9e37_79b9);
    let (o_dim, i_dim) = (4096usize, 2048usize); // wq_b shape
    let x: Vec<f32> = (0..i_dim).map(|_| rng.f()).collect();
    let w: Vec<u16> = (0..o_dim * i_dim)
        .map(|_| f32_to_bf16(rng.f() * 0.1))
        .collect();
    let want: Vec<f32> = (0..o_dim)
        .map(|o| {
            let row = &w[o * i_dim..(o + 1) * i_dim];
            x.iter()
                .zip(row)
                .map(|(&xi, &wb)| xi * bf16_to_f32(wb))
                .sum()
        })
        .collect();
    let x_buf = dev_bytes(&f32_bytes(&x));
    let w_buf = dev_bytes(&u16_bytes(&w));
    let mut y_buf = dev_zeroed(o_dim * 4);
    // SAFETY: device pointers sized for the dims.
    unsafe {
        launch_gemv_bf16(
            x_buf.ptr() as *const f32,
            w_buf.ptr() as *const u16,
            o_dim,
            i_dim,
            y_buf.ptr_mut() as *mut f32,
        )
        .expect("gemv_bf16");
    }
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32_vec(&y_buf.copy_out().expect("out")),
        "gemv_bf16",
    );
}

/// layernorm: y = (x-mean)/sqrt(var+eps)·w + b, matching math.rs::layernorm.
#[test]
fn layernorm_matches_scalar() {
    let mut rng = Lcg(0x1234_5678);
    let n = 128usize; // index_head_dim
    let x: Vec<f32> = (0..n).map(|_| rng.f() * 3.0).collect();
    let w: Vec<f32> = (0..n).map(|_| rng.f() * 0.5 + 1.0).collect();
    let b: Vec<f32> = (0..n).map(|_| rng.f() * 0.1).collect();
    let mut want = x.clone();
    rivoli::math::layernorm(&mut want, &w, &b, 1e-6);
    let x_buf = dev_bytes(&f32_bytes(&x));
    let w_buf = dev_bytes(&f32_bytes(&w));
    let b_buf = dev_bytes(&f32_bytes(&b));
    let mut y_buf = dev_zeroed(n * 4);
    // SAFETY: device pointers sized n.
    unsafe {
        launch_layernorm(
            x_buf.ptr() as *const f32,
            w_buf.ptr() as *const f32,
            b_buf.ptr() as *const f32,
            n,
            1e-6,
            y_buf.ptr_mut() as *mut f32,
        )
        .expect("layernorm");
    }
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32_vec(&y_buf.copy_out().expect("out")),
        "layernorm",
    );
}

/// index_score: scores[t] = Σ_h w[h]·wscale·ReLU((q_h·k_t)·dscale) — the DSA
/// scoring core, all 32 heads active (DSA; heads=null). Reference mirrors the
/// scalar indexer's inner loop over the same bf16-widened keys.
#[test]
fn index_score_matches_scalar() {
    let mut rng = Lcg(0xabcd_1234);
    let (nt, nh, hd) = (600usize, 32usize, 128usize); // GLM indexer dims
    let q: Vec<f32> = (0..nh * hd).map(|_| rng.f()).collect();
    let w: Vec<f32> = (0..nh).map(|_| rng.f()).collect();
    let kc: Vec<u16> = (0..nt * hd).map(|_| f32_to_bf16(rng.f())).collect();
    let wscale = 1.0 / (nh as f32).sqrt();
    let dscale = 1.0 / (hd as f32).sqrt();
    let want: Vec<f32> = (0..nt)
        .map(|t| {
            let krow = &kc[t * hd..(t + 1) * hd];
            (0..nh)
                .map(|hh| {
                    let qh = &q[hh * hd..(hh + 1) * hd];
                    let dot: f32 = qh
                        .iter()
                        .zip(krow)
                        .map(|(&a, &kb)| a * bf16_to_f32(kb))
                        .sum();
                    w[hh] * wscale * (dot * dscale).max(0.0)
                })
                .sum()
        })
        .collect();
    let q_buf = dev_bytes(&f32_bytes(&q));
    let w_buf = dev_bytes(&f32_bytes(&w));
    let kc_buf = dev_bytes(&u16_bytes(&kc));
    let mut sc_buf = dev_zeroed(nt * 4);
    // SAFETY: device pointers sized for the dims; heads=null → all nh active.
    unsafe {
        launch_index_score(
            q_buf.ptr() as *const f32,
            w_buf.ptr() as *const f32,
            kc_buf.ptr() as *const u16,
            std::ptr::null(),
            nt,
            nh,
            nh,
            hd,
            wscale,
            dscale,
            sc_buf.ptr_mut() as *mut f32,
        )
        .expect("index_score");
    }
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32_vec(&sc_buf.copy_out().expect("out")),
        "index_score",
    );
}

/// MISA_BLOCK, mirrored from src/indexer.rs (and the `#define` in indexer.hip).
const MISA_BLOCK: usize = 1024;

/// Host reference for the MISA block pool — the exact 5-line running mean from
/// src/indexer.rs::pool_push, but resizing a flat buffer (the device kernel
/// writes into a pre-sized slab, so opening a block just fills its row).
fn pool_push_ref(pool: &mut Vec<f32>, k: &[f32], t: usize, hd: usize) {
    let b = t / MISA_BLOCK;
    let in_block = t % MISA_BLOCK;
    if pool.len() < (b + 1) * hd {
        pool.resize((b + 1) * hd, 0.0);
    }
    let m = &mut pool[b * hd..(b + 1) * hd];
    if in_block == 0 {
        m.copy_from_slice(k);
    } else {
        let inv = 1.0 / (in_block + 1) as f32;
        for (mi, &ki) in m.iter_mut().zip(k) {
            *mi += (ki - *mi) * inv;
        }
    }
}

/// index_pool_push: the block-pooled running-mean key pool. Drives a sequence of
/// tokens (a partial block 0, then across the block boundary into block 1)
/// through the kernel and the scalar reference, asserting the pool matches. Each
/// token's key uses a fresh device buffer (kept alive) so the in-place folds on
/// the null stream serialize against distinct inputs.
#[test]
fn index_pool_push_matches_scalar() {
    let mut rng = Lcg(0x5115_a001);
    let hd = 128usize; // index_head_dim
    // Two blocks' worth of rows (block 0 + block 1). The kernel writes row
    // t/MISA_BLOCK, so a 2-block slab covers t up to 2·MISA_BLOCK-1.
    let mut pool_buf = dev_zeroed(2 * hd * 4);
    let mut want: Vec<f32> = Vec::new();
    // Token stream: 6 in block 0, then the block-1 open + two folds.
    let tokens: [usize; 9] = [0, 1, 2, 3, 4, 5, MISA_BLOCK, MISA_BLOCK + 1, MISA_BLOCK + 2];
    let mut keep: Vec<DeviceBuf> = Vec::new(); // hold key buffers alive across launches
    for &t in &tokens {
        let k: Vec<f32> = (0..hd).map(|_| rng.f()).collect();
        pool_push_ref(&mut want, &k, t, hd);
        let k_buf = dev_bytes(&f32_bytes(&k));
        // SAFETY: k_buf/pool_buf sized hd / 2·hd·f32; row t/MISA_BLOCK in range.
        unsafe {
            launch_index_pool_push(
                k_buf.ptr() as *const f32,
                pool_buf.ptr_mut() as *mut f32,
                t,
                hd,
            )
            .expect("index_pool_push");
        }
        keep.push(k_buf);
    }
    device_sync().expect("sync");
    // want only spans 2·hd (block 0 + block 1); the slab is exactly that.
    assert_eq!(want.len(), 2 * hd);
    let got = f32_vec(&pool_buf.copy_out().expect("out"));
    assert_close(&want, &got, "index_pool_push");
}

/// index_head_route: E_j = mean_b |w[j]·ReLU(q_j·pool_b)| — the MISA router
/// (paper Eq. 7-8). Reference mirrors the scalar `(w[j]·dot.max(0)).abs()` mean
/// form exactly (relu before abs, w inside abs, no wscale/dscale).
#[test]
fn index_head_route_matches_scalar() {
    let mut rng = Lcg(0x1100_1e00);
    let (nh, hd, m_blocks) = (32usize, 128usize, 5usize); // GLM indexer dims, 5 pooled blocks
    let q: Vec<f32> = (0..nh * hd).map(|_| rng.f()).collect();
    let w: Vec<f32> = (0..nh).map(|_| rng.f()).collect();
    let pool: Vec<f32> = (0..m_blocks * hd).map(|_| rng.f()).collect();
    let want: Vec<f32> = (0..nh)
        .map(|j| {
            let qj = &q[j * hd..(j + 1) * hd];
            let wj = w[j];
            let sum: f32 = (0..m_blocks)
                .map(|b| {
                    let kb = &pool[b * hd..(b + 1) * hd];
                    let dot: f32 = qj.iter().zip(kb).map(|(&a, &c)| a * c).sum();
                    (wj * dot.max(0.0)).abs()
                })
                .sum();
            sum / m_blocks as f32
        })
        .collect();
    let q_buf = dev_bytes(&f32_bytes(&q));
    let w_buf = dev_bytes(&f32_bytes(&w));
    let pool_buf = dev_bytes(&f32_bytes(&pool));
    let mut e_buf = dev_zeroed(nh * 4);
    // SAFETY: device pointers sized nh·hd / nh / m_blocks·hd / nh.
    unsafe {
        launch_index_head_route(
            q_buf.ptr() as *const f32,
            w_buf.ptr() as *const f32,
            pool_buf.ptr() as *const f32,
            m_blocks,
            nh,
            hd,
            e_buf.ptr_mut() as *mut f32,
        )
        .expect("index_head_route");
    }
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32_vec(&e_buf.copy_out().expect("out")),
        "index_head_route",
    );
}

// --- fp8 latent-cache device kernels vs the scalar path ---

/// append_kv_fp8 must produce byte-identical fp8 + scales to the host
/// quantizer (math.rs::quantize_latent_fp8) — the device f2e4m3 mirrors it.
#[test]
fn append_kv_fp8_matches_host_quantize() {
    use rivoli::math::{E4M3_BLOCK, quantize_latent_fp8};
    let mut rng = Lcg(0x0f_08_00_01u64);
    let (kvl, ropn) = (512usize, 64usize);
    let nb = kvl / E4M3_BLOCK;
    let latent: Vec<f32> = (0..kvl).map(|_| rng.f() * 3.0).collect();
    let rope: Vec<f32> = (0..ropn).map(|_| rng.f()).collect();
    // Host reference.
    let mut want_data = vec![0u8; kvl];
    let mut want_scale = vec![0.0f32; nb];
    quantize_latent_fp8(&latent, &mut want_data, &mut want_scale);
    // Device.
    let lat_buf = dev_bytes(&f32_bytes(&latent));
    let rope_buf = dev_bytes(&f32_bytes(&rope));
    let mut lc8 = dev_zeroed(kvl); // pos 0 only
    let mut lscale = dev_zeroed(nb * 4);
    let mut rc = dev_zeroed(ropn * 2);
    // SAFETY: device pointers sized for one token at pos 0.
    unsafe {
        rivoli::hip::launch_append_kv_fp8(
            lat_buf.ptr() as *const f32,
            rope_buf.ptr() as *const f32,
            lc8.ptr_mut(),
            lscale.ptr_mut() as *mut f32,
            rc.ptr_mut() as *mut u16,
            0,
            kvl,
            ropn,
            nb,
        )
        .expect("append_kv_fp8");
    }
    device_sync().expect("sync");
    assert_eq!(lc8.copy_out().expect("out"), want_data, "fp8 bytes differ");
    let got_scale = f32_vec(&lscale.copy_out().expect("out"));
    for (g, w) in got_scale.iter().zip(&want_scale) {
        assert!((g - w).abs() <= w.abs() * 1e-6 + 1e-9, "scale {g} vs {w}");
    }
}

/// mla_latent_attend_fp8 over an fp8 latent cache must match the scalar attend
/// reference computed over the SAME dequantized latents (so the only thing
/// under test is the kernel's dequant+flash, not the quantization error).
#[test]
fn mla_attend_fp8_matches_scalar() {
    use rivoli::math::{E4M3_BLOCK, dequant_latent_fp8, quantize_latent_fp8};
    let mut rng = Lcg(0x0f_08_a1_1e_u64);
    let (h, nt, kvl, rope) = (16usize, 200usize, 512usize, 64usize);
    let nb = kvl / E4M3_BLOCK;
    let qabs: Vec<f32> = (0..h * kvl).map(|_| rng.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| rng.f()).collect();
    // Quantize each token's latent to fp8 (host), keep both the bytes/scales
    // (for the device) and the exact dequantized f32 (for the reference).
    let mut lc8 = vec![0u8; nt * kvl];
    let mut lscale = vec![0.0f32; nt * nb];
    let mut lc_deq = vec![0u16; nt * kvl]; // dequantized then bf16-packed for attend_reference
    for t in 0..nt {
        let row: Vec<f32> = (0..kvl).map(|_| rng.f()).collect();
        quantize_latent_fp8(
            &row,
            &mut lc8[t * kvl..(t + 1) * kvl],
            &mut lscale[t * nb..(t + 1) * nb],
        );
        for i in 0..kvl {
            let f = dequant_latent_fp8(lc8[t * kvl + i], lscale[t * nb + i / E4M3_BLOCK]);
            lc_deq[t * kvl + i] = f32_to_bf16(f); // reference reads bf16; f is already e4m3-exact
        }
    }
    let rc: Vec<u16> = (0..nt * rope).map(|_| f32_to_bf16(rng.f())).collect();
    let scale = 1.0 / ((kvl + rope) as f32).sqrt();
    let rows: Vec<u32> = (0..nt as u32).collect();
    let want = attend_reference(&qabs, &qrope, &lc_deq, &rc, h, &rows, kvl, rope, scale);

    let qabs_buf = dev_bytes(&f32_bytes(&qabs));
    let qrope_buf = dev_bytes(&f32_bytes(&qrope));
    let lc8_buf = dev_bytes(&lc8);
    let lscale_buf = dev_bytes(&f32_bytes(&lscale));
    let rc_buf = dev_bytes(&u16_bytes(&rc));
    let mut clat_buf = dev_zeroed(h * kvl * 4);
    // SAFETY: device pointers sized for the dims; rows=null → dense over nt.
    unsafe {
        rivoli::hip::launch_attend_fp8(
            qabs_buf.ptr() as *const f32,
            qrope_buf.ptr() as *const f32,
            lc8_buf.ptr(),
            lscale_buf.ptr() as *const f32,
            rc_buf.ptr() as *const u16,
            std::ptr::null(),
            h,
            nt,
            kvl,
            rope,
            nb,
            scale,
            clat_buf.ptr_mut() as *mut f32,
        )
        .expect("attend_fp8");
    }
    device_sync().expect("sync");
    assert_close(
        &want,
        &f32_vec(&clat_buf.copy_out().expect("out")),
        "attend_fp8",
    );
}
