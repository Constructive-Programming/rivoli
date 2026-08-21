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
use super::select::{
    Extent, SelRule, compress_dst, compress_offset, positional_context_limit, scored_rows,
};
use crate::device::DeviceBuf;
use crate::resident::Fp8Weight;
use crate::v4::pin::V4Pin;
use anyhow::{Context, Result, ensure};
use rivoli_artifact::quant::read_f32;
use rivoli_artifact::v4_config::V4Config;
use rivoli_backend::abi::ScoreDims;
use rivoli_backend::hip::ScoreBufs;
use rivoli_backend::{
    NULL_STREAM, launch_act_quant_f4_rotated, launch_gemm_bf16, launch_index_score_blocks,
    launch_rope_adjacent, memcpy_dtod_async,
};
use rivoli_core::num::{bf16_to_f32, f32_to_bf16};

/// One indexed layer's persistent scored-selection state.
struct IdxLayer {
    /// The ABSOLUTE layer id, for error attribution: every refusal in [`IdxLayer::ingest`]
    /// names its layer, and carrying the id here is what keeps `ingest` from taking it as a
    /// parameter beside every call.
    layer: usize,
    /// The layer's class — [`IdxLayer::ingest`]'s destination arithmetic ([`compress_dst`])
    /// derives from it. Copied from the same `kinds` entry [`IndexerBank::new`] ASSERTS
    /// against the pin, and the bank is already "parallel to the engine's layer vector", so
    /// it cannot disagree with the engine's own classification without that assertion — or
    /// the parallelism everything here indexes by — being broken first.
    kind: LayerKind,
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
    /// D2H staging BYTES, reused across steps. Narrower than the arm's "a decode
    /// allocates nothing on the hot path" rule and deliberately stated so: `read_f32`
    /// hands back a fresh `Vec<f32>` view of each of these, the scaled weights collect
    /// into another, and `scored_rows` allocates a row per query. Only the two byte
    /// buffers below are actually reused; the rest is per-step and unpriced.
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
                layer: l,
                kind,
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

    /// Step 2 of [`V4Engine::scored_selection`]: the q-side launches, the host scale fold,
    /// and the scoring kernel — the raw `[m, n_comp]` score matrix, for
    /// [`scored_rows`] to rank. The caller owns the trigger: this runs only once the causal
    /// set has outgrown the top-k.
    ///
    /// Takes the layer by INDEX rather than as `&mut IdxLayer` because that reference would
    /// borrow `self.layers` — the caller cannot hold it and hand the bank over too.
    fn raw_scores(&mut self, li: usize, q: ScoreStep<'_>, at: Extent) -> Result<Vec<f32>> {
        let (h, hd) = (q.cfg.index_n_heads, q.cfg.index_head_dim);
        let (q_lora, rd) = (q.cfg.q_lora_rank, q.cfg.qk_rope_head_dim);
        let (dim, m, n_comp) = (q.cfg.hidden, at.query_rows(), q.n_comp);
        let ix = self.layers[li]
            .as_ref()
            .context("scored_selection resolved this layer's state before ingest")?;
        let iq = self.iq.ptr_mut().cast::<f32>();
        let wp = self.w_dev.ptr_mut().cast::<f32>();
        // SAFETY: `qrq` holds this step's `m * q_lora` quantized rows (qkv_project ran),
        // `iq` is `max_m * h * hd`, `wp` is `max_m * h`, the score slab is
        // `max_m * max_blocks >= m * n_comp`, and `ix.cache` holds `n_comp` finished rows —
        // written by ingest at prefill, but at DECODE ingest contributes at most the ONE
        // row that just closed and the rest came from earlier steps, so the invariant is
        // that the decode loop visits every position from 0, not anything ingest does. All
        // on the null stream, so ordering holds.
        unsafe {
            gemv_fp8(q.qrq, q.wq_b, m, (h * hd, q_lora), iq, NULL_STREAM)?;
            launch_rope_adjacent(
                iq,
                q.freqs,
                m * h,
                hd,
                rd,
                at.start_pos,
                h,
                false,
                NULL_STREAM,
            )?;
            launch_act_quant_f4_rotated(iq, m * h, hd, NULL_STREAM)?;
            launch_gemm_bf16(q.x, ix.wproj.ptr().cast(), wp, m, h, dim, NULL_STREAM)?;
        }
        // The scale, on host: `round_bf16(weights_proj(x))`, then `* wscale` landing in
        // bf16 — the oracle's own two stores (`Oracle::indexer`, model.py:424). The GEMM
        // writes raw f32 accumulations, so both rounds happen here; the traffic is
        // `m * index_n_heads` floats each way, 256 bytes at decode.
        self.w_dev
            .copy_out_prefix(&mut self.w_host, m * h * size_of::<f32>())?;
        let scaled: Vec<u8> = read_f32(&self.w_host)
            .into_iter()
            .flat_map(|v| {
                let r = bf16_to_f32(f32_to_bf16(v));
                bf16_to_f32(f32_to_bf16(r * self.wscale)).to_le_bytes()
            })
            .collect();
        self.w_dev.copy_in_at(0, &scaled)?;
        let bufs = ScoreBufs {
            q: self.iq.ptr().cast(),
            kv: ix.cache.ptr().cast(),
            w: self.w_dev.ptr().cast(),
            score: self.score_dev.ptr_mut().cast(),
        };
        let dims = ScoreDims {
            s: m,
            n_comp,
            heads: h,
            hd,
        };
        // SAFETY: sized in the launch's own terms two comments up; distinct allocations.
        unsafe { launch_index_score_blocks(bufs, dims, NULL_STREAM)? };
        self.score_dev
            .copy_out_prefix(&mut self.score_host, m * n_comp * size_of::<f32>())?;
        Ok(read_f32(&self.score_host))
    }
}

/// One step's q-side inputs to [`IndexerBank::raw_scores`] — resolved by the caller because
/// they live on [`V4Engine`], whose borrow the caller has already split to hand the bank out
/// `&mut`.
struct ScoreStep<'a> {
    /// The engine's config — the indexer's dims are read from it here rather than copied
    /// into the bank, so there is exactly one authority for them.
    cfg: &'a V4Config,
    /// This step's quantized `qr` rows — why the scoring must run after `qkv_project`.
    qrq: *const f32,
    /// The indexer's own `wq_b`, fp8 like the attention's.
    wq_b: Fp8Weight,
    /// The UNQUANTIZED pre-attention norm output `weights_proj` consumes.
    x: *const f32,
    /// This layer's rotary table, from the ONE selection site.
    freqs: *const f32,
    /// Causally-complete block count — the score matrix's width.
    n_comp: usize,
}

impl IdxLayer {
    /// Step 1 of [`V4Engine::scored_selection`]: run the nested compressor for this call and
    /// persist what it emitted into this layer's compressed cache. Runs on EVERY indexed-layer
    /// call, scoring or not — the module header carries why this cannot wait for the boundary.
    ///
    /// Same launch sequence, same contract, different geometry and destination from the
    /// attention compressor's.
    fn ingest(&mut self, w: CompInput, at: Extent) -> Result<()> {
        let layer = self.layer;
        // SAFETY: every pointer is a pin placement or a live DeviceBuf at its documented
        // shape; the null stream is what this whole arm runs on.
        let blocks = unsafe { self.comp.run(w, at, NULL_STREAM) }
            .with_context(|| format!("layer {layer} indexer compressor at {at:?}"))?;
        // The drift tripwire, BOTH ways, which is `kvcompress::compress_and_place`'s shape
        // and for its reason: the run and `compress_dst` decide from the same `(kind, at)`
        // and today cannot disagree, so what this catches is a future edit to one of them.
        // The `Some(_) && blocks == 0` corner is not cosmetic here — it would leave
        // `self.cache[row]` at its reset zeros, and a zero row scores EXACTLY 0.0 through the
        // scorer's `relu` while a real block can score negative (`weights_proj` is a bare
        // Linear), so the empty row would outrank real ones. Silent, plausible, permanent.
        match compress_dst(self.kind, 0, at) {
            None => ensure!(
                blocks == 0,
                "layer {layer}: the indexer's compressor emitted {blocks} block(s) where \
                 compress_dst names no destination — the two have drifted"
            ),
            Some((row, want)) => {
                ensure!(
                    blocks == want,
                    "layer {layer}: indexer emitted {blocks} block(s) at {at:?} where \
                     compress_dst reserved {want}"
                );
                ensure!(
                    row + blocks <= self.cache_rows,
                    "layer {layer}: {blocks} indexer block(s) at row {row} overrun the \
                     {}-row cache",
                    self.cache_rows
                );
                // The block WIDTH off the emitting geometry, not off the config a second
                // time — which is what `LayerCompressor::d`'s doc promised and this is the
                // caller it named.
                let d = self.comp.d();
                // SAFETY: `emitted()` holds `blocks * d` f32 by the compressor's sizing;
                // the destination range was bounded above; distinct allocations. Async and
                // stream-ordered like `attn::write_ring`'s copies — the blocking
                // `memcpy_dtod` is the arena-relocation entry point, and forcing a host
                // sync per indexed layer per step is not what this path wants.
                unsafe {
                    memcpy_dtod_async(
                        self.cache.ptr_mut().cast::<f32>().add(row * d).cast(),
                        self.comp.emitted(),
                        blocks * d * size_of::<f32>(),
                        NULL_STREAM,
                    )
                    .context("persisting the indexer's compressed cache")?;
                }
            }
        }
        Ok(())
    }

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
        let (cfg, topk) = (self.cfg, self.cfg.index_topk);
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

        // 1. The nested compressor, EVERY call.
        ix.ingest(CompInput::of(x, cfg.hidden, &cw, freqs), at)?;

        // 2. Scores, only once they can DECIDE something. The ratio is the compressor's
        // own — the same geometry that sized the cache the scorer reads.
        let n_comp = at.end_pos() / ix.comp.ratio();
        if n_comp <= topk {
            return Ok(None);
        }
        let step = ScoreStep {
            cfg,
            qrq,
            wq_b,
            x,
            freqs,
            n_comp,
        };
        let rule = SelRule {
            kind,
            index_topk: topk,
            at,
            offset: compress_offset(win, at),
        };
        let rows = scored_rows(&bank.raw_scores(li, step, at)?, n_comp, rule)?;
        Ok(Some(rows))
    }
}
