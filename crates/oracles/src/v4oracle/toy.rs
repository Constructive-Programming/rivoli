//! A structurally-identical miniature of V4-Flash, generated deterministically.
//!
//! This is the substrate the defect matrix runs on. It is not a stand-in for the real
//! checkpoint's *values* and no golden is emitted from it — it exists so the question "can
//! this gate fire, and is it silent where it should be" can be answered in under a second,
//! offline, and re-answered on every `cargo test`. See `V4Config::toy` for which
//! discriminants are preserved and why each one had to be.
//!
//! The weights are genuinely quantized, not random bytes: f32 values are drawn and then put
//! through the same fp8/fp4 block quantization the checkpoint uses. A checkpoint of random
//! bytes would carry arbitrary e8m0 scales and could make a defect look inert for reasons
//! that have nothing to do with the model.

use crate::v4oracle::forward::{CompressorW, ExpertW, HeadTailW, IndexerW, LayerW};
use crate::v4oracle::numerics::{
    FP4_MAX, FP8_MAX, e2m1_encode, e4m3_encode, fast_log2_ceil, fast_pow2,
};
use crate::v4oracle::weights::{NamedRng, V4Config, WMat, draw, fixed_bf16};

/// A whole toy model: all layers, plus the head tail.
///
/// No embedding. `Oracle::embed` is reached only from `bin/v4-oracle`, which loads the real
/// `embed.weight`; a toy embedding matrix was built on every `model()` and read by nothing.
pub struct ToyModel {
    pub layers: Vec<LayerW>,
    pub head_tail: HeadTailW,
}

/// A bf16 tensor, exactly as the checkpoint would store it.
fn dense(name: &str, rows: usize, cols: usize, scale: f32) -> WMat {
    let v = fixed_bf16(name, rows * cols, scale);
    WMat::Dense { rows, cols, v }
}

/// A norm weight vector. `draw` is centred on zero and the checkpoint's norms on one, so
/// every one of them is drawn as `1.0 + x`.
fn norm_vec(name: &str, n: usize, scale: f32) -> Vec<f32> {
    draw(name, n, scale).iter().map(|x| 1.0 + x).collect()
}

/// The e8m0 code for `fast_round_scale(amax, 1/max)`, and the scale it decodes to.
fn block_scale(amax: f32, max: f32) -> (u8, f32) {
    let e = fast_log2_ceil(amax.max(1e-30) / max);
    let code = (e + 127).clamp(0, 254) as u8;
    (code, fast_pow2(code as i32 - 127))
}

/// One 128x128 tile of a row-major `[rows][cols]` draw: the half-open span it covers, plus
/// the row stride every index into the flat vector needs.
struct Tile {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
    cols: usize,
}

/// The largest magnitude in the tile — the quantity that picks its shared e8m0 scale.
fn tile_amax(v: &[f32], t: &Tile) -> f32 {
    let mut amax = 0.0f32;
    for r in t.r0..t.r1 {
        for c in t.c0..t.c1 {
            amax = amax.max(v[r * t.cols + c].abs());
        }
    }
    amax
}

/// Encode one tile in place and return its scale code. Two passes over the tile, because
/// nothing in it can be encoded until all of it has been seen: the scale is a function of
/// the tile's own amax.
fn fp8_tile(v: &[f32], t: &Tile, w: &mut [u8]) -> u8 {
    let (code, sv) = block_scale(tile_amax(v, t), FP8_MAX);
    for r in t.r0..t.r1 {
        for c in t.c0..t.c1 {
            w[r * t.cols + c] = e4m3_encode((v[r * t.cols + c] / sv).clamp(-FP8_MAX, FP8_MAX));
        }
    }
    code
}

/// Quantize to fp8 on the 128x128 grid the checkpoint uses for attention and shared-expert
/// weights.
fn fp8(name: &str, rows: usize, cols: usize, scale: f32) -> WMat {
    let v = draw(name, rows * cols, scale);
    let (sr, sc) = (rows.div_ceil(128), cols.div_ceil(128));
    let mut w = vec![0u8; rows * cols];
    let mut s = vec![0u8; sr * sc];
    for br in 0..sr {
        for bc in 0..sc {
            let (r0, r1) = (br * 128, ((br + 1) * 128).min(rows));
            let (c0, c1) = (bc * 128, ((bc + 1) * 128).min(cols));
            let t = Tile {
                r0,
                r1,
                c0,
                c1,
                cols,
            };
            s[br * sc + bc] = fp8_tile(&v, &t, &mut w);
        }
    }
    WMat::Fp8 { rows, cols, w, s }
}

/// Pack one 32-element K group of `row` against the scale its own amax chose, two nibbles
/// per byte LOW NIBBLE FIRST — the order `inference/convert.py` documents by unpacking
/// `stack([FP4_TABLE[low], FP4_TABLE[high]]).flatten()`.
fn fp4_group(seg: &[f32], sv: f32, row: &mut [u8], g: usize) {
    for (i, &x) in seg.iter().enumerate() {
        let nib = e2m1_encode((x / sv).clamp(-FP4_MAX, FP4_MAX));
        let k = g * 32 + i;
        let byte = &mut row[k / 2];
        if k % 2 == 0 {
            *byte = (*byte & 0xf0) | nib;
        } else {
            *byte = (*byte & 0x0f) | (nib << 4);
        }
    }
}

/// Quantize to fp4 with one e8m0 scale per 32 elements of K.
fn fp4(name: &str, rows: usize, cols: usize, scale: f32) -> WMat {
    assert!(
        cols.is_multiple_of(32),
        "fp4 needs K divisible by the group size"
    );
    let v = draw(name, rows * cols, scale);
    let groups = cols / 32;
    let mut w = vec![0u8; rows * cols / 2];
    let mut s = vec![0u8; rows * groups];
    for r in 0..rows {
        let row = &mut w[r * (cols / 2)..(r + 1) * (cols / 2)];
        for g in 0..groups {
            let seg = &v[r * cols + g * 32..r * cols + g * 32 + 32];
            let amax = seg.iter().fold(0.0f32, |a, x| a.max(x.abs()));
            let (code, sv) = block_scale(amax, FP4_MAX);
            s[r * groups + g] = code;
            fp4_group(seg, sv, row, g);
        }
    }
    WMat::Fp4 { rows, cols, w, s }
}

fn expert(name: &str, c: &V4Config, quantized: bool) -> ExpertW {
    let (dim, inter) = (c.dim, c.moe_inter_dim);
    let mk = |suffix: &str, rows, cols| {
        let n = format!("{name}.{suffix}");
        if quantized {
            fp4(&n, rows, cols, 0.03)
        } else {
            fp8(&n, rows, cols, 0.03)
        }
    };
    ExpertW {
        w1: mk("w1", inter, dim),
        w2: mk("w2", dim, inter),
        w3: mk("w3", inter, dim),
    }
}

/// The three discriminants a compressor is shaped by, which always travel together: the
/// stride, the per-head width, and whether the indexer's Hadamard rotation applies.
struct CompressorShape {
    ratio: usize,
    d: usize,
    rotate: bool,
}

fn compressor(name: &str, c: &V4Config, shape: CompressorShape) -> CompressorW {
    let CompressorShape { ratio, d, rotate } = shape;
    let overlap = ratio == 4;
    let coff = 1 + usize::from(overlap);
    CompressorW {
        ratio,
        overlap,
        d,
        rotate,
        ape: draw(&format!("{name}.ape"), ratio * coff * d, 0.5),
        wkv: dense(&format!("{name}.wkv"), coff * d, c.dim, 0.05),
        wgate: dense(&format!("{name}.wgate"), coff * d, c.dim, 0.05),
        norm: norm_vec(&format!("{name}.norm"), d, 0.3),
    }
}

/// The sparse-attention indexer, which exists only where `compress_ratio == 4`.
fn indexer(p: &str, c: &V4Config, ratio: usize) -> IndexerW {
    IndexerW {
        wq_b: fp8(
            &format!("{p}.idx.wq_b"),
            c.index_n_heads * c.index_head_dim,
            c.q_lora_rank,
            0.03,
        ),
        weights_proj: dense(&format!("{p}.idx.wproj"), c.index_n_heads, c.dim, 0.05),
        compressor: compressor(
            &format!("{p}.idx.comp"),
            c,
            CompressorShape {
                ratio,
                d: c.index_head_dim,
                rotate: true,
            },
        ),
    }
}

/// The hash router's `[vocab_size, n_activated_experts]` table, materialised in full — which
/// only the toy vocabulary makes affordable.
fn tid2eid(p: &str, c: &V4Config) -> Vec<i64> {
    let mut r = NamedRng::new(&format!("{p}.gate.tid2eid"));
    (0..c.vocab_size)
        .flat_map(|_| {
            // Distinct experts per token, as a hash assignment would give.
            let mut pool: Vec<usize> = (0..c.n_routed_experts).collect();
            (0..c.n_activated_experts)
                .map(|_| pool.swap_remove(r.below(pool.len())) as i64)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// One layer. The three layer classes differ only in the tail: `compressor` wherever the
/// ratio is non-zero, `indexer` only at ratio 4, and the router table by `hash`.
fn layer(c: &V4Config, l: usize) -> LayerW {
    let p = format!("layers.{l}");
    let ratio = c.compress_ratio(l);
    let hash = l < c.n_hash_layers;
    LayerW {
        attn_sink: draw(&format!("{p}.attn_sink"), c.n_heads, 2.0),
        wq_a: fp8(&format!("{p}.wq_a"), c.q_lora_rank, c.dim, 0.03),
        q_norm: norm_vec(&format!("{p}.q_norm"), c.q_lora_rank, 0.3),
        wq_b: fp8(
            &format!("{p}.wq_b"),
            c.n_heads * c.head_dim,
            c.q_lora_rank,
            0.03,
        ),
        wkv: fp8(&format!("{p}.wkv"), c.head_dim, c.dim, 0.03),
        kv_norm: norm_vec(&format!("{p}.kv_norm"), c.head_dim, 0.3),
        wo_a: dense(
            &format!("{p}.wo_a"),
            c.o_groups * c.o_lora_rank,
            c.n_heads * c.head_dim / c.o_groups,
            0.03,
        ),
        wo_b: fp8(
            &format!("{p}.wo_b"),
            c.dim,
            c.o_groups * c.o_lora_rank,
            0.03,
        ),
        attn_norm: norm_vec(&format!("{p}.attn_norm"), c.dim, 0.3),
        ffn_norm: norm_vec(&format!("{p}.ffn_norm"), c.dim, 0.3),
        hc_attn_fn: draw(&format!("{p}.hc_attn_fn"), c.mix_hc() * c.hc_dim(), 0.05),
        hc_attn_base: draw(&format!("{p}.hc_attn_base"), c.mix_hc(), 1.0),
        hc_attn_scale: norm_vec(&format!("{p}.hc_attn_scale"), 3, 0.5),
        hc_ffn_fn: draw(&format!("{p}.hc_ffn_fn"), c.mix_hc() * c.hc_dim(), 0.05),
        hc_ffn_base: draw(&format!("{p}.hc_ffn_base"), c.mix_hc(), 1.0),
        hc_ffn_scale: norm_vec(&format!("{p}.hc_ffn_scale"), 3, 0.5),
        gate_w: dense(&format!("{p}.gate.weight"), c.n_routed_experts, c.dim, 0.05),
        gate_bias: (!hash).then(|| draw(&format!("{p}.gate.bias"), c.n_routed_experts, 0.2)),
        tid2eid: hash.then(|| tid2eid(&p, c)),
        compressor: (ratio != 0).then(|| {
            compressor(
                &format!("{p}.comp"),
                c,
                CompressorShape {
                    ratio,
                    d: c.head_dim,
                    rotate: false,
                },
            )
        }),
        indexer: (ratio == 4).then(|| indexer(&p, c, ratio)),
        experts: (0..c.n_routed_experts)
            .map(|e| (e, expert(&format!("{p}.e{e}"), c, true)))
            .collect(),
        shared: expert(&format!("{p}.shared"), c, false),
    }
}

/// The mHC head weights, the final norm, and the logits GEMM.
fn head_tail(c: &V4Config) -> HeadTailW {
    HeadTailW {
        // Drawn at the same scales as the per-block mHC weights above, because the head's
        // are the same kind of parameter: `hc_head_fn` is a projection of the flattened
        // `hc_mult * dim` row, and `hc_head_base` a per-copy bias into a sigmoid. A base
        // drawn much wider would saturate every gate and make `HeadHcNoRsqrt` inert for a
        // reason that has nothing to do with the model.
        hc_head_fn: draw("hc_head_fn", c.hc_mult * c.hc_dim(), 0.05),
        hc_head_base: draw("hc_head_base", c.hc_mult, 1.0),
        // `[1]`, matching the checkpoint. Centred on 1.0 like the block scales so the
        // sigmoid argument keeps the same order of magnitude as the mixes.
        hc_head_scale: norm_vec("hc_head_scale", 1, 0.5),
        norm: norm_vec("norm", c.dim, 0.3),
        // bf16 in the checkpoint, so bf16 here: `dense` rounds. The logits GEMM is the
        // one place in the model with no activation quantization at all, and a toy that
        // stored f32 weights would understate the noise the real one carries.
        lm_head: dense("head", c.vocab_size, c.dim, 0.05),
    }
}

/// Build the toy model for `cfg`, which must be [`V4Config::toy`]-shaped (the expert set is
/// materialised in full, which only makes sense at toy scale).
pub fn build(c: &V4Config) -> ToyModel {
    ToyModel {
        layers: (0..c.n_layers).map(|l| layer(c, l)).collect(),
        head_tail: head_tail(c),
    }
}
