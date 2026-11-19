//! The CPU halves of the attention rows' red proofs (`docs/measurement/gate-red-proofs.md`
//! §20): each family's plant (`tests/bit-identity/plants/attn-*.patch`) is shown to move
//! values on the gate's OWN draws before its device arm runs — §1's lesson, as for `rope`.
//! The host's `exp` and summation order are not the device's, so each model measures the
//! plant's RATE on these operands; the device arm is the measurement.
//!
//! Each model demands the plant move ≥ 90% of seeds, so it cannot be the kind that hides
//! in a lucky draw.

use super::attn_inputs::{Mla, ROPE, mla, mla_inputs, nasty, plant_nonfinite};
use super::inputs::{SEEDS, draw};
use rivoli_core::num::{bf16_to_f32, e4m3_to_f32};

/// Run `pairs(seed)` — every `[unplanted, planted]` value the plant reaches for that seed —
/// over `seeds`, print the rate, and demand the plant move ≥ 90% of them.
fn rate(what: &str, seeds: &[u64], pairs: impl Fn(u64) -> Vec<[f32; 2]>) {
    let (mut moved, mut diff, mut total) = (0usize, 0usize, 0usize);
    for &seed in seeds {
        let p = pairs(seed);
        let d = p.iter().filter(|p| differ(p[0], p[1])).count();
        (moved, diff, total) = (moved + usize::from(d > 0), diff + d, total + p.len());
    }
    let n = seeds.len();
    println!("{what}: {diff}/{total} values differ; {moved}/{n} eligible seeds affected");
    assert!(
        moved * 10 >= n * 9,
        "plant moved only {moved}/{n} seeds — sub-threshold"
    );
}

/// Two bit patterns that differ as values (both NaN counts as equal — the gate's rule).
fn differ(a: f32, b: f32) -> bool {
    a.to_bits() != b.to_bits() && !(a.is_nan() && b.is_nan())
}

/// Every seed the device rows draw.
fn all_seeds() -> Vec<u64> {
    (1..=SEEDS).collect()
}

/// One MLA head's final accumulators over the cell's `nr` rows, `[left-fused, right-fused]`
/// per column (`ins` is `attn_inputs::mla_inputs`'s cell).
fn mla_head(s: Mla, ins: &[Vec<f32>], h: usize, scale: f32) -> Vec<[f32; 2]> {
    let byte = |i: usize| ins[2][i / 4].to_bits().to_le_bytes()[i % 4];
    let half = |i: usize| (ins[4][i / 2].to_bits() >> (16 * (i % 2))) as u16;
    let lat = |t: usize, i: usize| e4m3_to_f32(byte(t * s.kvl + i)) * ins[3][t * 4 + i / 128];
    let q = |i: usize| ins[0][h * s.kvl + i];
    let (mut m, mut acc) = (f32::NEG_INFINITY, vec![[0.0f32; 2]; s.kvl]);
    for t in 0..s.nr {
        let part = (0..s.kvl).fold(0.0f32, |a, i| q(i).mul_add(lat(t, i), a));
        let part = (0..ROPE).fold(part, |a, d| {
            ins[1][h * ROPE + d].mul_add(bf16_to_f32(half(t * ROPE + d)), a)
        });
        let m_new = m.max(scale * part);
        let (corr, p) = ((m - m_new).exp(), scale.mul_add(part, -m_new).exp());
        for (i, a) in acc.iter_mut().enumerate() {
            a[0] = a[0].mul_add(corr, p * lat(t, i));
            a[1] = p.mul_add(lat(t, i), a[1] * corr);
        }
        m = m_new;
    }
    acc
}

/// `attn-mla.patch`: the accumulator `acc·corr + p·L` fused on the RIGHT product
/// (`fma(p, L, acc·corr)`), where hipcc fuses the LEFT (`v_mul` of `p·L`, then `v_fmac`) —
/// the exact defect `hc_post` shipped with and the device caught (rust-kernels.md §5, Phase
/// 0a lease 1). Measured at the row's two-split shape (64 heads × 130 rows; the first 4
/// heads), over every column's final accumulator.
///
/// **Eligible seeds exclude two degenerate classes, by construction of the inputs:** the nasty
/// seeds (NaN in the queries and cache — every score NaN, which the gate compares as a class)
/// and the edge seeds whose block SCALES start with `EDGES`' ±100 (one token's latent ×100
/// dominates, every other `p` underflows to 0, and `p·L = 0` hides which product fused). Both
/// classes stay in the device rows for what they do test; they cannot test this site.
#[test]
fn attn_mla_right_fused_acc_plant_is_adversarial_on_cpu() {
    let s: Mla = mla([64, 130, 512], false, true);
    let eligible: Vec<u64> = (1..=SEEDS)
        .filter(|&x| !nasty(x) && !x.is_multiple_of(4))
        .collect();
    rate("attn-mla right-fused acc", &eligible, |seed| {
        let ins = mla_inputs(s, seed);
        let scale = [0.072_168_78, 1.0, 0.01][(seed % 3) as usize];
        (0..4).flat_map(|h| mla_head(s, &ins, h, scale)).collect()
    });
}

/// One head's online softmax over `rows` of K/V head 0 (of 2), the accumulator updated both
/// ways: `[a·r + w·v, fma(a, r, w·v)]` per column.
fn gqa_both(q: &[f32], k: &[f32], v: &[f32], rows: usize) -> Vec<[f32; 2]> {
    let (d, hkv) = (q.len(), 2);
    let (mut m, mut acc) = (f32::NEG_INFINITY, vec![[0.0f32; 2]; d]);
    for j in 0..rows {
        let at = |c: usize| (j * hkv) * d + c;
        let s = (0..d).fold(0.0f32, |a, c| a + q[c] * k[at(c)]) / (d as f32).sqrt();
        let mx = m.max(s);
        let (rescale, w) = ((m - mx).exp(), (s - mx).exp());
        for (c, a) in acc.iter_mut().enumerate() {
            a[0] = a[0] * rescale + w * v[at(c)];
            a[1] = a[1].mul_add(rescale, w * v[at(c)]);
        }
        m = mx;
    }
    acc
}

/// `attn-gqa.patch`: the accumulator update fused (`fma(acc, rescale, w·v)`) where the C has
/// `contract(off)` and hipcc emits a multiply and an add. Measured at Glimmer's decode shape
/// (32 Q / 2 KV heads, d 128) over a 256-row global prefix, on the row's draws, the first 4
/// heads (all on KV head 0).
#[test]
fn attn_gqa_fused_acc_plant_is_adversarial_on_cpu() {
    let (hq, hkv, d, rows) = (32, 2, 128, 256);
    rate("attn-gqa fused acc", &all_seeds(), |seed| {
        let mut q = draw(hq * d, (seed, 1), 0.5, seed % 4 == 1);
        plant_nonfinite(&mut q, (seed, 1));
        let k = draw(rows * hkv * d, (seed, 2), 1.0, seed % 4 == 2);
        let mut v = draw(rows * hkv * d, (seed, 3), 1.0, seed % 4 == 3);
        plant_nonfinite(&mut v, (seed, 3));
        (0..4)
            .flat_map(|h| gqa_both(&q[h * d..][..d], &k, &v, rows))
            .collect()
    });
}

/// Logits `q · k_j` for every row `j` of `k` (`d` = `q.len()`), both ways: `[plain, fused]`.
fn dots(q: &[f32], k: &[f32]) -> Vec<[f32; 2]> {
    k.chunks_exact(q.len())
        .map(|kr| {
            let plain = q.iter().zip(kr).fold(0.0f32, |a, (x, y)| a + x * y);
            let fused = q.iter().zip(kr).fold(0.0f32, |a, (x, y)| x.mul_add(*y, a));
            [plain, fused]
        })
        .collect()
}

/// `attn-dense.patch`: the score dot fused (`acc = fma(q, k, acc)`) in both two-pass kernels,
/// where the C has `contract(off)`. Measured on each row's own draws: the gather's V4 decode
/// (d 512, one shared 128-slot KV, the first 8 heads) and the MHA row's unmasked cell (3
/// heads × 300 rows × d 64, a KV per head) — one logit per (head, slot).
#[test]
fn attn_dense_fused_dot_plant_is_adversarial_on_cpu() {
    rate("attn-dense fused dot (gather)", &all_seeds(), |seed| {
        let mut q = draw(64 * 512, (seed, 1), 0.1, seed % 4 == 1);
        plant_nonfinite(&mut q, (seed, 1));
        let kv = draw(128 * 512, (seed, 2), 1.0, seed % 4 == 2);
        (0..8)
            .flat_map(|h| dots(&q[h * 512..][..512], &kv))
            .collect()
    });
    rate("attn-dense fused dot (mha)", &all_seeds(), |seed| {
        let mut q = draw(3 * 64, (seed, 1), 0.3, seed % 4 == 1);
        plant_nonfinite(&mut q, (seed, 1));
        let k = draw(3 * 300 * 64, (seed, 2), 1.0, seed % 4 == 2);
        (0..3)
            .flat_map(|h| dots(&q[h * 64..][..64], &k[h * 300 * 64..][..300 * 64]))
            .collect()
    });
}
