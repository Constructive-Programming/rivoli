//! MLA (multi-head latent attention) decode — the absorb path, ported from
//! colibri `attention_rows` (glm.c). Reference scalar implementation for a
//! single new token (S=1), dense over the whole cached context (DSA sparse
//! selection is a later addition once the indexer weights are re-exported).
//!
//! The compressed KV cache stores, per token per layer, only the normed latent
//! `L[kv_lora]` and the roped shared key `R[qk_rope]` — in bf16 (decided
//! 2026-07-18). k_nope and value are never materialized: q_nope is absorbed
//! through `kv_b`'s nope rows into latent space (`qabs`), scored against the
//! latents directly, and the attention-weighted latent is projected back
//! through `kv_b`'s value rows.

use crate::math::{bf16_to_f32, f32_to_bf16, rmsnorm, softmax};
use crate::model::ModelConfig;
use crate::quant::{addrow, matvec_i4, matvec_i4_rows, read_f32};
use crate::snapshot::Snapshot;
use anyhow::Result;

/// Compressed MLA KV cache: per layer, bf16 latents and roped keys, one row per
/// cached token. Grows by one row per layer per decoded token.
pub struct KvCache {
    lc: Vec<Vec<u16>>, // [n_layers] flat, len = tokens * kv_lora
    rc: Vec<Vec<u16>>, // [n_layers] flat, len = tokens * qk_rope
    kv_lora: usize,
}

impl KvCache {
    pub fn new(n_layers: usize, kv_lora: usize) -> Self {
        Self {
            lc: vec![Vec::new(); n_layers],
            rc: vec![Vec::new(); n_layers],
            kv_lora,
        }
    }

    /// Number of tokens cached for a layer.
    fn tokens(&self, layer: usize) -> usize {
        self.lc[layer].len() / self.kv_lora
    }

    fn append(&mut self, layer: usize, latent: &[f32], rope: &[f32]) {
        self.lc[layer].extend(latent.iter().map(|&v| f32_to_bf16(v)));
        self.rc[layer].extend(rope.iter().map(|&v| f32_to_bf16(v)));
    }
}

/// Reused buffers so a decode step allocates nothing per layer.
pub struct AttnScratch {
    qr: Vec<f32>,     // q_lora
    q: Vec<f32>,      // n_heads * qk_head
    comp: Vec<f32>,   // kv_lora + qk_rope
    qabs: Vec<f32>,   // kv_lora
    clat: Vec<f32>,   // kv_lora
    ctx: Vec<f32>,    // n_heads * v_head
    scores: Vec<f32>, // up to context length
}

impl AttnScratch {
    pub fn new(cfg: &ModelConfig) -> Self {
        Self {
            qr: vec![0.0; cfg.q_lora_rank],
            q: vec![0.0; cfg.n_heads * cfg.qk_head_dim()],
            comp: vec![0.0; cfg.kv_lora_rank + cfg.qk_rope_head_dim],
            qabs: vec![0.0; cfg.kv_lora_rank],
            clat: vec![0.0; cfg.kv_lora_rank],
            ctx: vec![0.0; cfg.n_heads * cfg.v_head_dim],
            scores: Vec::new(),
        }
    }
}

/// Interleaved RoPE on a `qk_rope`-length vector at position `pos` (colibri
/// `rope_interleave`): pairs (2j, 2j+1) rotate by angle `pos·θ^(-2j/dim)`,
/// output halves are [rotated-first | rotated-second].
fn rope_interleave(v: &mut [f32], pos: usize, theta: f32) {
    let n = v.len();
    let half = n / 2;
    let inbuf: Vec<f32> = v.to_vec();
    for j in 0..half {
        let inv = theta.powf(-2.0 * j as f32 / n as f32);
        let ang = pos as f32 * inv;
        let (cs, sn) = (ang.cos(), ang.sin());
        let (a, b) = (inbuf[2 * j], inbuf[2 * j + 1]);
        v[j] = a * cs - b * sn;
        v[half + j] = b * cs + a * sn;
    }
}

/// One MLA attention decode step for `layer`. `x` is the RMSNorm'd hidden
/// (input_layernorm applied by the caller). Appends this token's latent/key to
/// `kv` and writes the attention output (pre-residual) into `out[hidden]`.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    snap: &Snapshot,
    cfg: &ModelConfig,
    layer: usize,
    x: &[f32],
    pos: usize,
    kv: &mut KvCache,
    s: &mut AttnScratch,
    out: &mut [f32],
) -> Result<()> {
    let h = cfg.n_heads;
    let qh = cfg.qk_head_dim();
    let nope = cfg.qk_nope_head_dim;
    let rope = cfg.qk_rope_head_dim;
    let kvl = cfg.kv_lora_rank;
    let vh = cfg.v_head_dim;
    let eps = cfg.rms_norm_eps as f32;
    let theta = cfg.rope_theta() as f32;
    let scale = 1.0 / (qh as f32).sqrt();
    let base = format!("model.layers.{layer}.self_attn");

    // 1) Q path: q_a → rmsnorm → q_b.
    let q_a = snap.int4(&format!("{base}.q_a_proj"), cfg.hidden)?;
    let q_b = snap.int4(&format!("{base}.q_b_proj"), cfg.q_lora_rank)?;
    let q_a_ln = read_f32(snap.require(&format!("{base}.q_a_layernorm.weight"))?);
    matvec_i4(&mut s.qr, x, &q_a);
    let qr_norm = s.qr.clone();
    rmsnorm(&mut s.qr, &qr_norm, &q_a_ln, eps);
    matvec_i4(&mut s.q, &s.qr, &q_b);

    // 2) KV path: kv_a → split latent/rope, normalize latent, RoPE the key.
    let kv_a = snap.int4(&format!("{base}.kv_a_proj_with_mqa"), cfg.hidden)?;
    let kv_a_ln = read_f32(snap.require(&format!("{base}.kv_a_layernorm.weight"))?);
    matvec_i4(&mut s.comp, x, &kv_a);
    let (latent_raw, rope_raw) = s.comp.split_at(kvl);
    let mut latent = latent_raw.to_vec();
    let latent_in = latent.clone();
    rmsnorm(&mut latent, &latent_in, &kv_a_ln, eps);
    let mut rkey = rope_raw.to_vec();
    rope_interleave(&mut rkey, pos, theta);
    kv.append(layer, &latent, &rkey);

    // RoPE each head's query rope segment.
    for head in 0..h {
        let off = head * qh + nope;
        rope_interleave(&mut s.q[off..off + rope], pos, theta);
    }

    // 3) Absorb-path attention core, per head.
    let kv_b = snap.int4(&format!("{base}.kv_b_proj"), kvl)?;
    let nt = kv.tokens(layer);
    if s.scores.len() < nt {
        s.scores.resize(nt, 0.0);
    }
    for head in 0..h {
        let qp = &s.q[head * qh..head * qh + qh];
        let (qnope, qrope) = qp.split_at(nope);
        let rbase = head * (nope + vh);

        // Absorb q_nope through kv_b nope rows → qabs in latent space.
        s.qabs.iter_mut().for_each(|a| *a = 0.0);
        for (d, &qd) in qnope.iter().enumerate() {
            addrow(&kv_b, rbase + d, qd, &mut s.qabs);
        }

        // Scores over every cached token: qabs·L + qrope·R, scaled.
        for (t, sc) in s.scores[..nt].iter_mut().enumerate() {
            let lrow = &kv.lc[layer][t * kvl..(t + 1) * kvl];
            let rrow = &kv.rc[layer][t * rope..(t + 1) * rope];
            let mut a = 0.0f32;
            for (i, &lb) in lrow.iter().enumerate() {
                a += s.qabs[i] * bf16_to_f32(lb);
            }
            for (d, &rb) in rrow.iter().enumerate() {
                a += qrope[d] * bf16_to_f32(rb);
            }
            *sc = a * scale;
        }
        softmax(&mut s.scores[..nt]);

        // Weighted sum of latents, then project through kv_b value rows.
        s.clat.iter_mut().for_each(|c| *c = 0.0);
        for (t, &sc) in s.scores[..nt].iter().enumerate() {
            let lrow = &kv.lc[layer][t * kvl..(t + 1) * kvl];
            for (i, &lb) in lrow.iter().enumerate() {
                s.clat[i] += sc * bf16_to_f32(lb);
            }
        }
        let cx = &mut s.ctx[head * vh..head * vh + vh];
        matvec_i4_rows(cx, &s.clat, &kv_b, rbase + nope);
    }

    // 4) Output projection.
    let o_proj = snap.int4(&format!("{base}.o_proj"), h * vh)?;
    matvec_i4(out, &s.ctx, &o_proj);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_interleave_pos0_is_identity() {
        // At pos 0 all angles are 0 → cos=1 sin=0 → v[j]=a, v[half+j]=b, i.e. a
        // deinterleave (even→first half, odd→second half).
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        rope_interleave(&mut v, 0, 8_000_000.0);
        assert_eq!(v, vec![1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn bf16_cache_roundtrip_is_within_tolerance() {
        for &x in &[0.0f32, 1.0, -2.5, 0.3333, 1234.5] {
            let r = bf16_to_f32(f32_to_bf16(x));
            let tol = x.abs() * 0.01 + 1e-3;
            assert!((r - x).abs() <= tol, "{x} -> {r}");
        }
    }
}
