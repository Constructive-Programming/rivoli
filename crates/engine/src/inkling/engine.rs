//! The Inkling 2 engine: the device scratch the looped decoder reads — the residual stream, the
//! block-residual bank, the per-cache-layer short-convolution states, the per-pass KV cache, the
//! loop-pass snapshots, and the attention/MoE/logit scratch. `forward.rs` drives the state this
//! file allocates; `geometry.rs` owns every shape and byte count it reads.
//!
//! The loop's extra passes each carry their OWN KV cache copies (T16): the cache has
//! `num_layers + (num_passes−1)·(core_end−core_start)` layers, not `num_layers`. The sconv state is
//! per cache layer for the same reason — two passes must not shift one another's windows. The block
//! bank and the residual stream restart from the pre-loop snapshot on each extra pass (T8).

use super::geometry::{cache_layer_count, kv_tensor_bytes, residual_bytes, sconv_state_bytes};
use crate::device::DeviceBuf;
use anyhow::{Result, ensure};
use rivoli_artifact::inkling_config::InklingTextConfig;

/// The Inkling decode path's activation scratch and caches. Weights live in the pin.
pub struct InklingScratch {
    /// The residual stream for one position (`hidden` f32).
    pub x: DeviceBuf,
    /// The normed hidden feeding a branch (`hidden` f32).
    pub xn: DeviceBuf,
    /// The full layer's separate k/v input norms (`hidden` f32 each); sliding layers reuse `xn`.
    pub xn_k: DeviceBuf,
    pub xn_v: DeviceBuf,
    /// The block-residual bank: one `hidden` vector per block boundary, plus the partial.
    pub block_bank: DeviceBuf,
    /// The four short-convolution states (k, v, post-attn, post-mlp) of EACH cache layer, each
    /// `channels·(k−1)` f32. Indexed by cache layer: an extra loop pass must not continue the
    /// window a previous pass left behind, and the layers of one pass must not share a window.
    pub sconv: Vec<[DeviceBuf; 4]>,
    /// Per-cache-layer K and V, `kv_heads·head_dim·ctx` f32 EACH. The extra loop passes append
    /// their own copies (T16), so this is longer than `num_hidden_layers`.
    pub kc: Vec<DeviceBuf>,
    pub vc: Vec<DeviceBuf>,
    /// The residual stream as it stood when the pre-loop prefix ended, and the bank beside it —
    /// every extra pass restarts from these (T8).
    pub pre_loop_x: DeviceBuf,
    pub pre_loop_bank: DeviceBuf,
    /// The PREVIOUS pass's residual and bank, which the looped layers' MessageGates read
    /// (LOOP:537, T8). Pass 1 has no predecessor; the caller seeds these from the live state.
    pub prev_pass_x: DeviceBuf,
    pub prev_pass_bank: DeviceBuf,
    /// The query projection (`num_heads·head_dim` f32).
    pub q: DeviceBuf,
    /// The attention output (`num_heads·head_dim` f32).
    pub attn: DeviceBuf,
    /// The projected k/v for one token (`kv_heads·head_dim` f32 each), post-sconv, pre-cache.
    pub k: DeviceBuf,
    pub v: DeviceBuf,
    /// The rel-bias path: `rel_states` (`heads·d_rel`), `rel_logits` (`heads·rel_extent`), and the
    /// distance-gathered `bias` (`heads·n_ctx`).
    pub rel_states: DeviceBuf,
    pub rel_logits: DeviceBuf,
    pub bias: DeviceBuf,
    /// The per-head sink logit (`heads`) — read ON-DEVICE by `sink_attend` — and the sigmoid
    /// output gate (`heads·head_dim`).
    pub sink: DeviceBuf,
    pub gate: DeviceBuf,
    /// The logits (`unpadded_vocab_size` f32 — the live rows, T14).
    pub logits: DeviceBuf,
}

/// The number of cache layers the loop needs: the stored layers plus the extra passes' copies.
pub fn cache_layers(text: &InklingTextConfig) -> usize {
    let core = text
        .loop_config
        .as_ref()
        .map(|lc| (lc.core_start, lc.core_end, lc.num_passes));
    cache_layer_count(text.num_hidden_layers, core)
}

/// `n_cache` KV layers: K and V each [`kv_tensor_bytes`] — one tensor's worth per buffer, not the
/// K+V layer total twice (the 2× over-allocation this pair carried while it read
/// `kv_layer_bytes` into both).
fn kv_caches(
    n_cache: usize,
    kv_heads: usize,
    head_dim: usize,
    n_ctx: usize,
) -> Result<(Vec<DeviceBuf>, Vec<DeviceBuf>)> {
    let bytes = kv_tensor_bytes(kv_heads, head_dim, n_ctx);
    let mut kc = Vec::with_capacity(n_cache);
    let mut vc = Vec::with_capacity(n_cache);
    for _ in 0..n_cache {
        kc.push(DeviceBuf::new(bytes)?);
        vc.push(DeviceBuf::new(bytes)?);
    }
    Ok((kc, vc))
}

/// One `[k, v, post-attn, post-mlp]` sconv state set per cache layer: the k/v sides are
/// `kv_heads·head_dim` channels wide, the post convolutions `hidden`.
fn sconv_states(
    n_cache: usize,
    kv_channels: usize,
    hidden: usize,
    k: usize,
) -> Result<Vec<[DeviceBuf; 4]>> {
    let kv_bytes = sconv_state_bytes(kv_channels, k);
    let h_bytes = sconv_state_bytes(hidden, k);
    let mut states = Vec::with_capacity(n_cache);
    for _ in 0..n_cache {
        states.push([
            DeviceBuf::new(kv_bytes)?,
            DeviceBuf::new(kv_bytes)?,
            DeviceBuf::new(h_bytes)?,
            DeviceBuf::new(h_bytes)?,
        ]);
    }
    Ok(states)
}

impl InklingScratch {
    /// Allocate the scratch for `n_ctx` positions. `n_ctx` is the caller's prompt plus intended
    /// generation, not `model_max_length` (1048576) — a full layer's cache is linear in it.
    pub fn new(text: &InklingTextConfig, n_ctx: usize) -> Result<Self> {
        ensure!(n_ctx > 0, "n_ctx is 0 — no positions to decode");
        ensure!(
            text.num_key_value_heads > 0 && text.head_dim > 0 && text.hidden_size > 0,
            "the config carries a zero attention dimension"
        );
        let hidden = text.hidden_size;
        let kv_heads = text.num_key_value_heads;
        let hd = text.head_dim;
        let qd = text.num_attention_heads * hd;
        let block = text.n_layers_per_attn_res_block;
        ensure!(block > 0, "n_layers_per_attn_res_block is 0");
        // One bank slot per block boundary across the stored layers, plus the partial block.
        let bank_bytes = (text.num_hidden_layers / block + 1) * residual_bytes(hidden);
        let n_cache = cache_layers(text);
        let (kc, vc) = kv_caches(n_cache, kv_heads, hd, n_ctx)?;
        // The k/v sconv channels are the KV projection width; post-attn/post-mlp are hidden.
        let sconv = sconv_states(n_cache, kv_heads * hd, hidden, text.sconv_kernel_size)?;
        Ok(Self {
            x: DeviceBuf::new(residual_bytes(hidden))?,
            xn: DeviceBuf::new(residual_bytes(hidden))?,
            xn_k: DeviceBuf::new(residual_bytes(hidden))?,
            xn_v: DeviceBuf::new(residual_bytes(hidden))?,
            block_bank: DeviceBuf::new(bank_bytes)?,
            sconv,
            kc,
            vc,
            pre_loop_x: DeviceBuf::new(residual_bytes(hidden))?,
            pre_loop_bank: DeviceBuf::new(bank_bytes)?,
            prev_pass_x: DeviceBuf::new(residual_bytes(hidden))?,
            prev_pass_bank: DeviceBuf::new(bank_bytes)?,
            q: DeviceBuf::new(qd * 4)?,
            attn: DeviceBuf::new(qd * 4)?,
            k: DeviceBuf::new(kv_heads * hd * 4)?,
            v: DeviceBuf::new(kv_heads * hd * 4)?,
            rel_states: DeviceBuf::new(text.num_attention_heads * text.d_rel * 4)?,
            rel_logits: DeviceBuf::new(text.num_attention_heads * text.rel_extent * 4)?,
            bias: DeviceBuf::new(text.num_attention_heads * n_ctx * 4)?,
            sink: DeviceBuf::new(text.num_attention_heads * 4)?,
            gate: DeviceBuf::new(qd * 4)?,
            logits: DeviceBuf::new(text.unpadded_vocab_size * 4)?,
        })
    }
}
