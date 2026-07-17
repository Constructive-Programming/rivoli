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

use rivoli::math::silu;
use rivoli::quant::{matvec_i4, row_bytes};
use rivoli::snapshot::Int4Matrix;

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

/// A random per-row int4 matrix `[o_dim, i_dim]`: packed nibbles, f32 scales,
/// plus the little-endian scale bytes the `Int4Matrix` reference reads.
struct Mat {
    packed: Vec<u8>,
    scale_f32: Vec<f32>,
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
    let scale_f32: Vec<f32> = (0..o_dim).map(|_| rng.f() * 0.05).collect();
    let scale_bytes: Vec<u8> = scale_f32.iter().flat_map(|s| s.to_le_bytes()).collect();
    Mat {
        packed,
        scale_f32,
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

/// Concatenate per-expert matrices into the flat buffers the kernel expects.
fn flatten(mats: &[Mat]) -> (Vec<u8>, Vec<f32>) {
    let mut packed = Vec::new();
    let mut scale = Vec::new();
    for m in mats {
        packed.extend_from_slice(&m.packed);
        scale.extend_from_slice(&m.scale_f32);
    }
    (packed, scale)
}

fn check(seed: u64, hidden: usize, inter: usize, e: usize) {
    let mut rng = Lcg(seed);
    let x: Vec<f32> = (0..hidden).map(|_| rng.f()).collect();
    let gates: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, inter, hidden)).collect();
    let ups: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, inter, hidden)).collect();
    let downs: Vec<Mat> = (0..e).map(|_| gen_mat(&mut rng, hidden, inter)).collect();
    let w: Vec<f32> = (0..e).map(|_| rng.f()).collect();

    let want = reference(&x, hidden, inter, &gates, &ups, &downs, &w);

    let (gp, gs) = flatten(&gates);
    let (up, us) = flatten(&ups);
    let (dp, ds) = flatten(&downs);
    let got = rivoli::hip::moe_experts(&x, hidden, inter, e, &gp, &gs, &up, &us, &dp, &ds, &w)
        .expect("kernel launch");

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
