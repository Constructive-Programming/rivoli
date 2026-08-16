//! The scored compressed-block selection: the lightning indexer's device pipeline and the
//! per-layer state it needs — what retired `crate::v4`'s second declared deviation (M15).
//!
//! Every kernel here already existed and was already scored before this file had a caller:
//! `wq_b` rides `gemv_fp8_bf16` and the finish rides `rope_adjacent` (both on the attention
//! q path since M8), the spread and the scorer are `blockindex.hip`'s pair, bit-identical
//! against the oracle in `tests/kernel_v4_indexer.rs`, and the nested compressor is
//! [`LayerCompressor`] under [`Geom::indexer`](super::geometry::Geom::indexer)'s
//! Hadamard-fp4 finish. What this file adds is the WIRING and the two host steps no kernel
//! covers: the `weights_proj` scale (a `[m, index_n_heads]` round-trip — 256 bytes at
//! decode) and the top-k assembly, which is `super::select::scored_rows` and is gated
//! deviceless against the frozen oracle in `tests/v4_scored_selection.rs`.
//!
//! # When this runs, and when it deliberately does not
//!
//! * **`max_ctx` below [`positional_context_limit`]: never.** [`IndexerBank::new`] returns
//!   `None`, no state is allocated, no weight is narrowed, and the engine is the pre-M15
//!   arm BYTE FOR BYTE — which is what keeps the M8 parity record binding on it.
//! * **Above it, the nested compressor runs on EVERY indexed-layer call from position 0**,
//!   scoring or not: its pooling state and cache are cumulative, and a compressor first
//!   consulted at the boundary would score queries against an empty cache — fluent, wrong,
//!   and permanent, the exact failure the positional refusal used to make impossible.
//! * **The scoring kernels run only once the causal set outgrows `index_topk`**
//!   (`end_pos / ratio > index_topk`). Below that the top-k keeps everything, so
//!   [`scored_rows`]'s ascending output equals the positional fill whatever the scores say
//!   — the selection is decided without computing them, and the positional path is taken
//!   as the cheaper spelling of the same bytes.
//!
//! # What the selection can and cannot claim against the reference
//!
//! Given bit-identical inputs the scorer is bit-identical to the oracle, but its inputs are
//! not: the projections and the pooling upstream are tolerance-gated kernels, so near a
//! score TIE the engine and the oracle can legitimately pick different blocks — the
//! oracle's own scoring note names this ("no numeric tolerance would show it"). The gates
//! are therefore set-equality where sets are determined (below the cap, and the deviceless
//! scored_rows-vs-oracle comparison at equal scores) plus the NLL-smoothness measurement
//! across the old boundary, never a bitwise selection claim at real depth.

use super::engine::{V4Engine, gemv_fp8};
use super::geometry::LayerKind;
use super::kvcompress::{CompInput, LayerCompressor, narrow_to_bf16};
use super::select::{Extent, compress_dst, compress_offset, positional_context_limit, scored_rows};
use crate::device::DeviceBuf;
use crate::v4::pin::V4Pin;
use anyhow::{Context, Result, ensure};
use rivoli_artifact::quant::read_f32;
use rivoli_artifact::v4_config::V4Config;
use rivoli_backend::abi::ScoreDims;
use rivoli_backend::hip::ScoreBufs;
use rivoli_backend::{
    NULL_STREAM, launch_act_quant_f4_rotated, launch_gemm_bf16, launch_index_score_blocks,
    launch_rope_adjacent, memcpy_dtod,
};
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};

/// One indexed layer's persistent scored-selection state.
struct IdxLayer {
    /// The indexer's nested compressor — [`Geom::indexer`](super::geometry::Geom::indexer)'s
    /// geometry, so its emit finishes with the Hadamard-fp4 spread the scorer's `kv` side
    /// requires.
    comp: LayerCompressor,
    /// `[max_blocks, index_head_dim]` f32 — the indexer's own compressed cache. No ring
    /// prefix: this buffer exists only to be scored against, so its region base is 0
    /// ([`compress_dst`]'s doc names this exact caller).
    cache: DeviceBuf,
    /// Rows [`IdxLayer::cache`] holds — carried for the same reason every other
    /// `DeviceBuf` in this arm carries its row count: the buffer has no length.
    cache_rows: usize,
    /// `weights_proj` narrowed to the bf16 `launch_gemm_bf16` indexes,
    /// `[index_n_heads, hidden]`. See [`narrow_to_bf16`] for why the pin's f32 pointer
    /// cannot be handed over directly (a row-stride error, not a precision choice).
    wproj: DeviceBuf,
}

/// The scored-selection bank: per-layer indexer state plus the step scratch every indexed
/// layer shares. Exists only on an engine whose `max_ctx` can outgrow the positional set.
pub(super) struct IndexerBank {
    /// Parallel to the engine's layer vector; `None` on layers without an indexer.
    layers: Vec<Option<IdxLayer>>,
    /// `[max_m, index_n_heads * index_head_dim]` f32 — the roped, spread query rows.
    iq: DeviceBuf,
    /// `[max_m, index_n_heads]` f32 — `weights_proj`'s output, then (after the host
    /// round-trip) the scaled per-head weights the scorer consumes.
    w_dev: DeviceBuf,
    /// `[max_m, max_blocks]` f32 — the full pre-top-k score matrix.
    score_dev: DeviceBuf,
    /// D2H staging, reused across steps so the decode loop allocates nothing.
    w_host: Vec<u8>,
    score_host: Vec<u8>,
    /// `index_head_dim^-1/2 * index_n_heads^-1/2` — the reference's
    /// `softmax_scale * n_heads ** -0.5`, folded into the per-head weights exactly where
    /// the oracle folds it (after `weights_proj`, in bf16).
    wscale: f32,
}

impl IndexerBank {
    /// Build the bank, or `None` for an engine the boundary cannot reach.
    ///
    /// `kinds` spans the ARTIFACT's layers in engine order; the pin is consulted for each
    /// indexed one and the two are ASSERTED to agree (config-driven placement on one side,
    /// config-driven classification on the other — a mismatch is a loader bug surfacing
    /// here rather than as a null launch).
    pub(super) fn new(
        pin: &V4Pin,
        cfg: &V4Config,
        kinds: &[(usize, LayerKind)],
        max_ctx: usize,
        max_m: usize,
    ) -> Result<Option<Self>> {
        if max_ctx < positional_context_limit(cfg.index_topk) {
            return Ok(None);
        }
        let (h, hd) = (cfg.index_n_heads, cfg.index_head_dim);
        let f32s = |n: usize| DeviceBuf::new(n * size_of::<f32>());
        // Sized at the INDEXED ratio, not the tightest-of-all — the two coincide (4), but
        // this buffer serves only layers that HAVE an indexer, so its arithmetic should
        // name their class. `div_ceil` for `DeviceLayer::new`'s reason.
        let ratio = LayerKind::Overlap
            .compressor_ratio()
            .context("the indexed layer class has a ratio")?;
        let max_blocks = max_ctx.div_ceil(ratio);
        let mut layers = Vec::with_capacity(kinds.len());
        for &(l, kind) in kinds {
            let iw = &pin.layer(l)?.indexer;
            ensure!(
                kind.has_indexer() == iw.is_some(),
                "layer {l}: the pin and the layer class disagree about the indexer"
            );
            let Some(iw) = iw else {
                layers.push(None);
                continue;
            };
            let mut ix = IdxLayer {
                comp: LayerCompressor::indexer(cfg, kind, max_m, &iw.compressor)
                    .with_context(|| format!("layer {l} indexer compressor"))?,
                cache: f32s(max_blocks * hd)?,
                cache_rows: max_blocks,
                wproj: narrow_to_bf16(iw.weights_proj, h * cfg.hidden)?,
            };
            ix.reset()?;
            layers.push(Some(ix));
        }
        Ok(Some(Self {
            layers,
            iq: f32s(max_m * h * hd)?,
            w_dev: f32s(max_m * h)?,
            score_dev: f32s(max_m * max_blocks)?,
            w_host: Vec::new(),
            score_host: Vec::new(),
            wscale: (hd as f32).powf(-0.5) * (h as f32).powf(-0.5),
        }))
    }

    /// Clear every layer's pooling state and cache — between sequences, with the engine's
    /// own reset.
    pub(super) fn reset(&mut self) -> Result<()> {
        for ix in self.layers.iter_mut().flatten() {
            ix.reset()?;
        }
        Ok(())
    }
}

impl IdxLayer {
    fn reset(&mut self) -> Result<()> {
        self.comp.reset()?;
        // The cache needs no zero for correctness — every block index below `n_comp` is
        // rewritten by this sequence before the scorer can name it — but an unwritten row
        // read through a future bound slip would be PLAUSIBLE garbage, so one fill per
        // sequence buys the loud failure mode. Same trade `DeviceLayer::reset` makes.
        let bytes = self.cache_rows_bytes();
        // SAFETY: `cache` was allocated at exactly this size; reset runs at construction
        // and between sequences, nothing in flight.
        unsafe { rivoli_backend::fill_u32(self.cache.ptr_mut(), 0, bytes) }
    }

    fn cache_rows_bytes(&self) -> usize {
        self.cache_rows * self.comp.d() * size_of::<f32>()
    }
}

impl V4Engine<'_> {
    /// Run this layer's indexer for the step and, where the causal block set outgrows
    /// `index_topk`, return the scored per-row selection for
    /// [`Sel::gather_scored`](super::select::Sel::gather_scored). `None` means the
    /// positional fill is exact — an unindexed layer, an engine below the boundary, or a
    /// step whose legal set still fits under the top-k.
    ///
    /// Must run AFTER `qkv_project` (it consumes this step's quantized `qr`) and BEFORE the
    /// selection upload — the reference's own order inside `Attention.forward`.
    pub(super) fn scored_selection(
        &mut self,
        layer: usize,
        at: Extent,
    ) -> Result<Option<Vec<Vec<i32>>>> {
        // Everything read from `self` beside the bank is resolved FIRST, so the bank's
        // `&mut` borrow below stays disjoint — `run_compress` takes the same shape for the
        // same reason.
        let li = layer - self.range.start;
        let kind = self.layers[li].kind;
        if self.indexer.is_none() || !kind.has_indexer() {
            return Ok(None);
        }
        let freqs = self.rope.for_layer(kind)?;
        let (dim, topk) = (self.cfg.hidden, self.cfg.index_topk);
        let (h, hd) = (self.cfg.index_n_heads, self.cfg.index_head_dim);
        let (q_lora, rd) = (self.cfg.q_lora_rank, self.cfg.qk_rope_head_dim);
        let win = self.dims.window;
        let iw = self
            .pin
            .layer(layer)?
            .indexer
            .as_ref()
            .with_context(|| format!("layer {layer} is indexed but its pin has no indexer"))?;
        let (wq_b, cw) = (iw.wq_b, iw.compressor);
        let x = self.xw.ptr().cast::<f32>();
        let qrq = self.a_qrq.ptr().cast::<f32>();
        let bank = self.indexer.as_mut().context("checked Some above")?;
        let ix = bank.layers[li]
            .as_mut()
            .with_context(|| format!("layer {layer} is indexed but the bank has no state"))?;

        // 1. The nested compressor, EVERY call — the module header carries why this cannot
        // wait for the boundary. Same launch sequence, same contract, different geometry
        // and destination from the attention compressor's.
        let w = CompInput::of(x, dim, &cw, freqs);
        // SAFETY: every pointer is a pin placement or a live DeviceBuf at its documented
        // shape; the null stream is what this whole arm runs on.
        let blocks = unsafe { ix.comp.run(w, at, NULL_STREAM) }
            .with_context(|| format!("layer {layer} indexer compressor at {at:?}"))?;
        if blocks > 0 {
            let (row, want) = compress_dst(kind, 0, at)
                .context("the compressor emitted where compress_dst names no destination")?;
            ensure!(
                blocks == want,
                "layer {layer}: indexer emitted {blocks} block(s) where compress_dst \
                 reserved {want}"
            );
            ensure!(
                row + blocks <= ix.cache_rows,
                "layer {layer}: {blocks} indexer block(s) at row {row} overrun the \
                 {}-row cache",
                ix.cache_rows
            );
            let rowb = hd * size_of::<f32>();
            // SAFETY: `emitted()` holds `blocks * hd` f32 by the compressor's sizing; the
            // destination range was bounded above; distinct allocations.
            unsafe {
                memcpy_dtod(
                    ix.cache.ptr_mut().cast::<f32>().add(row * hd).cast(),
                    ix.comp.emitted(),
                    blocks * rowb,
                )
                .context("persisting the indexer's compressed cache")?;
            }
        }

        // 2. Scores, only once they can DECIDE something. The ratio is the compressor's
        // own — the same geometry that sized the cache the scorer reads.
        let n_comp = at.end_pos() / ix.comp.ratio();
        if n_comp <= topk {
            return Ok(None);
        }
        let m = at.query_rows();
        let iq = bank.iq.ptr_mut().cast::<f32>();
        let wp = bank.w_dev.ptr_mut().cast::<f32>();
        // SAFETY: `qrq` holds this step's `m * q_lora` quantized rows (qkv_project ran),
        // `iq` is `max_m * h * hd`, `wp` is `max_m * h`, the score slab is
        // `max_m * max_blocks >= m * n_comp`, and `ix.cache` holds `n_comp` finished rows
        // by step 1 — all on the null stream, so ordering holds.
        unsafe {
            gemv_fp8(qrq, wq_b, m, (h * hd, q_lora), iq, NULL_STREAM)?;
            launch_rope_adjacent(
                iq,
                freqs,
                m * h,
                hd,
                rd,
                at.start_pos,
                h,
                false,
                NULL_STREAM,
            )?;
            launch_act_quant_f4_rotated(iq, m * h, hd, NULL_STREAM)?;
            launch_gemm_bf16(x, ix.wproj.ptr().cast(), wp, m, h, dim, NULL_STREAM)?;
        }
        // The scale, on host: `round_bf16(weights_proj(x))`, then `* wscale` landing in
        // bf16 — the oracle's own two stores (`Oracle::indexer`, model.py:424). The GEMM
        // writes raw f32 accumulations, so both rounds happen here; the traffic is
        // `m * index_n_heads` floats each way, 256 bytes at decode.
        bank.w_dev
            .copy_out_prefix(&mut bank.w_host, m * h * size_of::<f32>())?;
        let scaled: Vec<u8> = read_f32(&bank.w_host)
            .into_iter()
            .flat_map(|v| {
                let r = bf16_to_f32(f32_to_bf16(v));
                bf16_to_f32(f32_to_bf16(r * bank.wscale)).to_le_bytes()
            })
            .collect();
        bank.w_dev.copy_in_at(0, &scaled)?;
        let bufs = ScoreBufs {
            q: bank.iq.ptr().cast(),
            kv: ix.cache.ptr().cast(),
            w: bank.w_dev.ptr().cast(),
            score: bank.score_dev.ptr_mut().cast(),
        };
        let dims = ScoreDims {
            s: m,
            n_comp,
            heads: h,
            hd,
        };
        // SAFETY: sized in the launch's own terms two comments up; distinct allocations.
        unsafe { launch_index_score_blocks(bufs, dims, NULL_STREAM)? };
        bank.score_dev
            .copy_out_prefix(&mut bank.score_host, m * n_comp * size_of::<f32>())?;
        let rows = scored_rows(
            &read_f32(&bank.score_host),
            n_comp,
            kind,
            topk,
            at,
            compress_offset(win, at),
        )?;
        Ok(Some(rows))
    }
}
