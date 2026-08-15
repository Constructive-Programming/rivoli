//! Everything ONE `Block.forward` call is handed and everything it carries between calls:
//! the layer's weights ([`LayerW`] and the three sub-bundles), the per-call context
//! ([`LayerCtx`]), and the host-side caches that outlive a step ([`LayerRings`],
//! [`CompState`]).
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut is by COHESION: these are the plain data the
//! block body reads and writes, with no arithmetic of their own — `LayerCtx::tag` is the
//! only behaviour in the file, and it is here because the naming rule it enforces belongs
//! to the context rather than to any one stage. `forward.rs` re-exports every item at its
//! original path, so `v4oracle::forward::{LayerW, LayerCtx, CompState, …}` still resolves.

use crate::v4oracle::weights::WMat;

// ---------------------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------------------

/// One `Compressor`'s parameters. `wkv`/`wgate` are `Linear(..., dtype=torch.float32)` in
/// the reference, so they take the un-quantized `F.linear` path — the checkpoint stores
/// them in bf16 and the module holds them in f32.
#[derive(Clone)]
pub struct CompressorW {
    pub ratio: usize,
    pub overlap: bool,
    /// `head_dim` of the *compressor*: `args.head_dim` for the attention one,
    /// `args.index_head_dim` for the indexer's.
    pub d: usize,
    /// `rotate=True` only for the indexer's compressor: Hadamard + fp4 instead of
    /// partial fp8.
    pub rotate: bool,
    /// `[ratio, coff * d]`.
    pub ape: Vec<f32>,
    pub wkv: WMat,
    pub wgate: WMat,
    pub norm: Vec<f32>,
}

impl CompressorW {
    pub fn coff(&self) -> usize {
        1 + usize::from(self.overlap)
    }
}

#[derive(Clone)]
pub struct IndexerW {
    pub wq_b: WMat,
    pub weights_proj: WMat,
    pub compressor: CompressorW,
}

#[derive(Clone)]
pub struct ExpertW {
    pub w1: WMat,
    pub w2: WMat,
    pub w3: WMat,
}

#[derive(Clone)]
pub struct LayerW {
    pub attn_sink: Vec<f32>,
    pub wq_a: WMat,
    pub q_norm: Vec<f32>,
    pub wq_b: WMat,
    pub wkv: WMat,
    pub kv_norm: Vec<f32>,
    /// fp8 on disk, dequantized to bf16 at load exactly as `convert.py`'s `wo_a` branch
    /// does, because `Attention.forward` consumes it raw in an einsum rather than through
    /// `Linear.forward` — there is no activation quantization on this one.
    pub wo_a: WMat,
    pub wo_b: WMat,
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub hc_attn_fn: Vec<f32>,
    pub hc_attn_base: Vec<f32>,
    pub hc_attn_scale: Vec<f32>,
    pub hc_ffn_fn: Vec<f32>,
    pub hc_ffn_base: Vec<f32>,
    pub hc_ffn_scale: Vec<f32>,
    pub gate_w: WMat,
    /// `Some` iff the layer routes by score (`layer_id >= n_hash_layers`).
    pub gate_bias: Option<Vec<f32>>,
    /// `Some` iff the layer routes by hash. `[vocab_size, n_activated_experts]`.
    pub tid2eid: Option<Vec<i64>>,
    pub compressor: Option<CompressorW>,
    /// Present only where `compress_ratio == 4` — 21 of the 43 layers.
    pub indexer: Option<IndexerW>,
    /// Routed experts, indexed by expert id. Sparse: only the ones a run actually reaches
    /// are loaded, since one is 13.37 MB.
    pub experts: std::collections::HashMap<usize, ExpertW>,
    pub shared: ExpertW,
}

/// Everything one `Block.forward` call needs that does not vary within it.
///
/// A struct rather than eight more parameters: `attention`, `moe`, `gate` and `run_layer`
/// all took the same tail, and four copies of a parameter list is four places for `s` and
/// `start_pos` to get swapped. `s` is the number of query rows — the prompt length at
/// prefill, and 1 at decode, which is also `start_pos`'s discriminant (`start_pos == 0`
/// means prefill throughout the reference).
pub struct LayerCtx<'a> {
    pub lw: &'a LayerW,
    pub layer: usize,
    pub s: usize,
    pub start_pos: usize,
    pub input_ids: &'a [u32],
    /// Which call this is: `"pre"` for the prefill, `"dec0"`, `"dec1"`, ... for the decode
    /// steps. NOT the golden prefix -- see [`LayerCtx::tag`].
    pub step_tag: &'a str,
}

impl LayerCtx<'_> {
    /// The prefix every recorded golden carries: `L{layer}.{step_tag}`.
    ///
    /// A method, not a field, because it must be impossible to apply inconsistently. When
    /// the layer id was prepended in `run_layer` alone, the goldens pushed inside
    /// `attention` and `moe` kept the bare step tag, a four-layer run wrote `pre.q` four
    /// times, and `Capture::float` -- which returns the FIRST match -- silently hid three of
    /// them. Every push in this file goes through here.
    pub fn tag(&self) -> String {
        format!("L{}.{}", self.layer, self.step_tag)
    }
}

// ---------------------------------------------------------------------------------------
// mutable state
// ---------------------------------------------------------------------------------------

#[derive(Clone)]
pub struct CompState {
    /// `[coff * ratio, coff * d]`, f32, zero-initialised.
    pub kv_state: Vec<f32>,
    /// `[coff * ratio, coff * d]`, f32, `-inf`-initialised.
    pub score_state: Vec<f32>,
    /// `[max_seq_len / ratio, d]` — the compressed region this compressor writes into.
    /// For the attention compressor this is a VIEW of `kv_cache[window_size..]` in the
    /// reference; here it is a separate buffer that `Attention` concatenates, which is the
    /// same values in the same order.
    pub cache: Vec<f32>,
}

/// One layer's host-side caches, carried across decode steps: the sliding-window ring
/// plus the two compressors' pooling state. Named for the ring because that is the part
/// `Oracle::attention` indexes modulo the window; the two `CompState` halves are
/// append-only.
pub struct LayerRings {
    /// `[window_size, head_dim]` — the sliding-window ring only. The compressed region
    /// lives in `comp.cache`.
    pub win_cache: Vec<f32>,
    pub comp: Option<CompState>,
    pub idx_comp: Option<CompState>,
}
