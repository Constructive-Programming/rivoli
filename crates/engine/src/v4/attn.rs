//! The V4 attention sublayer: shared-K=V MQA over a sliding-window ring plus the compressed
//! blocks, in four phases.
//!
//! Ported from `old:src/attn.rs`'s `v4::attention` and `old:src/f4gpu.rs`'s
//! `attention_block`/`io_for`. One kv entry `head_dim` wide serves as both K and V for every
//! head, which is the architecture's whole KV story and why there is no latent, no absorb and
//! no value projection here — [`crate::glm::attn`] is the counterpart and shares not one step.
//!
//! # Everything runs on ONE stream, and that is a correctness requirement
//!
//! The block is a straight-line data dependency: every step reads what the one before it
//! wrote. rivoli's streams are created `hipStreamNonBlocking`, so the null stream carries no
//! implicit ordering against them, and a single operation left on a different stream reads a
//! buffer whose producer has not been waited on — an unordered activation, which is fluent
//! wrong text where a fully-serialised version is merely slow.
//!
//! **This arm runs the whole block, the compressor and the norms on the NULL stream**, which
//! is what the reference did and what its own header recorded as an owed rebase. The device
//! copies are the part that is easy to miss: `memcpy_dtod` is a BLOCKING `hipMemcpy`, so
//! host-blocking hides the hazard while everything sits on stream 0 — the moment the launchers
//! move, it becomes a read racing a stream-ordered write. The set to convert is seven, not
//! six: the six launcher FUNCTIONS plus the async copy.
//!
//! # Three obligations no plausible caller satisfies by accident
//!
//! * **The compressor must already have run for this same step, and must have written BOTH
//!   destinations.** This block only READS the compressed rows — the attention scratch's tail
//!   at prefill, the cache's tail at decode — and writes neither. Running it after would hand
//!   the attend kernel uninitialised device memory in rows that every later decode step
//!   selects BY POSITION and weights with `exp(l - max)`. [`V4Engine::attention_block`] does
//!   both here rather than at two call sites, which is what makes it impossible to get right
//!   in prefill and wrong in decode.
//! * **The selection must name rows that exist in what the step attends.** The kernel
//!   dereferences `kv + idx * head_dim` for every non-negative entry, and the bound is
//!   `seqlen + n_comp` at prefill and `window + n_comp` at decode — NOT `seqlen` and `window`,
//!   because the compressed selection legitimately names rows past the window region. The
//!   SHAPE is checked below; the values are [`super::select`]'s to get right.
//! * **`Dims` is validated at entry**, not only at construction: it is `Copy` with public
//!   fields, so a struct literal or a later assignment skips `from_config` entirely.

use super::engine::{V4Engine, gemv_fp8};
use super::geometry::{FP8_BLOCK, KV_QUANT_BLOCK};
use super::select::{Extent, Sel};
use crate::device::as_le_bytes;
use crate::resident::Fp8Weight;
use anyhow::Result;
use rivoli_backend::{
    NULL_STREAM, launch_act_quant_f8_prefix, launch_gather_attn_shared_kv, launch_gemv_fp8_bf16,
    launch_qk_norm, launch_rmsnorm_batch, launch_rope_adjacent, memcpy_dtod_async,
};

/// One layer's attention weights, resolved once per call.
///
/// `wo_a` is an [`Fp8Weight`] like the rest and goes through the same GEMV, but it is NOT
/// arithmetically the same: **no activation quantization is performed against it**, and it is
/// the one GROUPED projection — output row `j` reads input group `j / o_lora_rank`, where a
/// group is a contiguous run of heads and so needs no gather.
#[derive(Clone, Copy)]
struct Weights {
    wq_a: Fp8Weight,
    q_norm: *const f32,
    wq_b: Fp8Weight,
    wkv: Fp8Weight,
    kv_norm: *const f32,
    /// `[n_heads]` f32 — the softmax DENOMINATOR's per-head term. It never enters the
    /// numerator, which is what makes it a sink rather than a bias.
    attn_sink: *const f32,
    wo_a: Fp8Weight,
    wo_b: Fp8Weight,
    /// `[wkv ‖ wq_a]` as ONE weight — see [`super::pin::place_fp8_qkv`]. `Some` on this
    /// checkpoint, and read only at DECODE: at `m > 1` the fused `[m, hd + q_lora]` output
    /// would interleave the two destinations row-wise.
    wqkv: Option<Fp8Weight>,
}

/// This call's device pointers and per-call extents, resolved once so the phases share a
/// single derivation.
///
/// All `Copy` raw pointers — holding them across `&mut self` calls borrows nothing, which is
/// the same shape [`crate::glm::attn`]'s `AttnCall` takes and for the same reason.
///
/// `qr` and `qrq` are separate buffers holding the same values, and that is not waste: `qr` is
/// `q_norm(wq_a(x))` and a ratio-4 layer's indexer consumes it AFTER the q path is done with
/// it, so quantizing in place would destroy an input. `xq` is separate for the same reason on
/// `x`, which the compressor and the router both read UNQUANTIZED. Neither pair costs a copy:
/// the activation quantizer reads its source and writes the quantized copy in one launch,
/// which preserves exactly the property the separation exists for.
#[derive(Clone, Copy)]
struct AttnCall {
    /// `[m, dim]` — the pre-attention norm's output. Not modified.
    x: *const f32,
    /// `[m, dim]` — the activation-quantized copy of `x`.
    xq: *mut f32,
    qr: *mut f32,
    qrq: *mut f32,
    /// `[m, n_heads * head_dim]`
    q: *mut f32,
    /// See [`super::engine::V4Engine::a_kv`] for why this is wider than `[m, head_dim]`.
    kv: *mut f32,
    o: *mut f32,
    y: *mut f32,
    /// The persistent `[ring ‖ compressed]` cache for this layer.
    cache: *mut f32,
    /// This layer's rotary table, interleaved `(cos, sin)`, from the ONE selection site.
    freqs: *const f32,
    idxs: *const i32,
    /// `[m, dim]` — the sublayer output the hyper-connection expansion consumes.
    out: *mut f32,
}

impl V4Engine<'_> {
    /// The whole attention sublayer for `layer` over the rows `at` names: the compressor and
    /// its placements, the q/kv projections, the selection, then the attend and the output.
    ///
    /// The order is the reference's (`Attention.forward`) and it is not optional — see this
    /// module's first obligation. The selection moved BEHIND `qkv_project` with M15, which
    /// is also the reference's order: the indexer consumes this step's `qr`, so a selection
    /// built before the q chain could never be scored. The positional path does not care
    /// where it runs; the scored one does.
    pub(super) fn attention_block(&mut self, layer: usize, at: Extent) -> Result<()> {
        self.dims.validate()?;
        self.compress_and_place(layer, at)?;
        let li = layer - self.range.start;
        let c = self.attn_call(li)?;
        let w = self.attn_weights(layer)?;
        self.qkv_project(w, c, at)?;
        let scored = self.scored_selection(layer, at)?;
        let (_, topk) = self.upload_selection(layer, at, scored)?;
        self.attend_rows(w, c, at, topk)?;
        self.output_project(w, c, at)
    }

    /// Build this step's selection, upload it, and report the `(rows, cols)` it filled.
    ///
    /// **The `Sel` is built HERE and nowhere else.** The reference let its caller supply `win`
    /// and then overwrote it from `Dims`, because two reviewers independently found the hole:
    /// `Dims::window` drives every ring write (`pos % window`), so a `Sel` disagreeing with it
    /// produced a selection over the wrong slot space that matched its own shape and passed
    /// every guard — in-bounds reads, no crash, fluent wrong text. One construction site is the
    /// version of that fix with nothing left to override.
    ///
    /// `scored` is [`V4Engine::scored_selection`]'s answer: `Some` rows go through
    /// [`Sel::gather_scored`], `None` means the positional fill is the same selection for
    /// less work (or the only one, on an unindexed layer) and goes through [`Sel::gather`].
    fn upload_selection(
        &mut self,
        layer: usize,
        at: Extent,
        scored: Option<Vec<Vec<i32>>>,
    ) -> Result<(usize, usize)> {
        let sel = Sel {
            win: self.dims.window,
            kind: self.layers[layer - self.range.start].kind,
            index_topk: self.cfg.index_topk,
            at,
        };
        self.idx_host.clear();
        let shape = match &scored {
            Some(rows) => sel.gather_scored(rows, &mut self.idx_host),
            None => sel.gather(&mut self.idx_host),
        }
        .map_err(|e| e.context(format!("layer {layer} selection at {at:?}")))?;
        self.idx_dev.copy_in_at(0, as_le_bytes(&self.idx_host))?;
        Ok(shape)
    }

    /// This layer's resident attention weights.
    fn attn_weights(&self, layer: usize) -> Result<Weights> {
        let lp = self.pin.layer(layer)?;
        Ok(Weights {
            wq_a: lp.wq_a,
            q_norm: lp.q_norm,
            wq_b: lp.wq_b,
            wkv: lp.wkv,
            kv_norm: lp.kv_norm,
            attn_sink: lp.attn_sink,
            wo_a: lp.wo_a,
            wo_b: lp.wo_b,
            wqkv: lp.wqkv,
        })
    }

    /// Resolve the layer's scratch and cache pointers once.
    ///
    /// The selection's column count is NOT here: since the selection moved behind the q
    /// chain it does not exist yet when the pointers are resolved, so it travels from
    /// `upload_selection`'s return straight into `attend_rows` — one producer, one hop,
    /// nothing to disagree with.
    fn attn_call(&mut self, li: usize) -> Result<AttnCall> {
        let freqs = self.rope.for_layer(self.layers[li].kind)?;
        Ok(AttnCall {
            x: self.xw.ptr().cast(),
            xq: self.xq.ptr_mut().cast(),
            qr: self.a_qr.ptr_mut().cast(),
            qrq: self.a_qrq.ptr_mut().cast(),
            q: self.a_q.ptr_mut().cast(),
            kv: self.a_kv.ptr_mut().cast(),
            o: self.a_o.ptr_mut().cast(),
            y: self.a_y.ptr_mut().cast(),
            cache: self.layers[li].cache.ptr_mut().cast(),
            freqs,
            idxs: self.idx_dev.ptr().cast(),
            out: self.sub.ptr_mut().cast(),
        })
    }

    /// Phase 1: quantize the activation once, then the q and kv chains.
    ///
    /// `x` is quantized ONCE and read by both `wq_a` and `wkv`. The reference runs a separate
    /// `act_quant` inside each `Linear`, but on the same row at the same block size, so the two
    /// produce identical bytes.
    ///
    /// **A DECODE takes the fused `[wkv ‖ wq_a]` GEMV when the pin built one.** Same kernel,
    /// same per-row `k`, same fold — only the grid is taller — so every output value is
    /// bit-identical; what moves is WHERE the q intermediate lands (`kv + head_dim`, because
    /// the fused output must be contiguous and the KV entry must stay at the base where its
    /// consumers already look). The one LOCAL fact: the kv rows compute BEFORE `q_norm` rather
    /// than after the query's rotary, and nothing between those points reads or writes them.
    fn qkv_project(&mut self, w: Weights, p: AttnCall, at: Extent) -> Result<()> {
        let (d, m) = (self.dims, at.query_rows());
        let (nh, hd, rd, dim) = (d.n_heads, d.head_dim, d.rope_head_dim, d.dim);
        let (q_lora, eps, pos0) = (d.q_lora_rank, d.norm_eps, at.start_pos);
        let fused = if at.is_prefill() { None } else { w.wqkv };
        // SAFETY: every pointer is live device scratch or a resident weight for its dims; each
        // launch's inputs are produced by a prior launch on the same (null) stream, so ordering
        // holds. A `Some(wqkv)` caller granted `kv` the `hd + q_lora` floats the fused row
        // needs — `V4Engine::a_kv`'s sizing is where that slack is explicit.
        unsafe {
            launch_act_quant_f8_prefix(p.x, p.xq, m, dim, dim, FP8_BLOCK, NULL_STREAM)?;
            let qr = match fused {
                Some(f) => {
                    // The seam arithmetic (kernel scale row = `j >> log2(block)`) is exact only
                    // on a block-aligned seam, and the pin REFUSES to build a concat otherwise
                    // — so a `Some` here with a misaligned `head_dim` is unreachable through
                    // `V4Pin::build`.
                    debug_assert!(hd.is_multiple_of(FP8_BLOCK));
                    gemv_fp8(p.xq, f, m, (hd + q_lora, dim), p.kv, NULL_STREAM)?;
                    p.kv.add(hd)
                }
                None => {
                    gemv_fp8(p.xq, w.wq_a, m, (q_lora, dim), p.qr, NULL_STREAM)?;
                    p.qr
                }
            };
            launch_rmsnorm_batch(qr, w.q_norm, m, q_lora, eps, NULL_STREAM)?;
            launch_act_quant_f8_prefix(qr, p.qrq, m, q_lora, q_lora, FP8_BLOCK, NULL_STREAM)?;
            gemv_fp8(p.qrq, w.wq_b, m, (nh * hd, q_lora), p.q, NULL_STREAM)?;
            // QK-norm BEFORE the rotary. Read off the reference rather than inferred from a
            // green test: the two orders differ only through a bf16-rounded statistic, so the
            // distance between them measures the ROUNDING scale rather than a defect's reach —
            // which is exactly the kind of difference a tolerance is free to call noise.
            launch_qk_norm(p.q, m * nh, hd, eps, NULL_STREAM)?;
            launch_rope_adjacent(p.q, p.freqs, m * nh, hd, rd, pos0, nh, false, NULL_STREAM)?;

            // At a fused decode the kv entry is already at `p.kv`'s base — the concat's kv rows
            // came first precisely so this chain runs unchanged from here down.
            if fused.is_none() {
                gemv_fp8(p.xq, w.wkv, m, (hd, dim), p.kv, NULL_STREAM)?;
            }
            launch_rmsnorm_batch(p.kv, w.kv_norm, m, hd, eps, NULL_STREAM)?;
            launch_rope_adjacent(p.kv, p.freqs, m, hd, rd, pos0, 1, false, NULL_STREAM)?;
            // PARTIAL, at block 64 rather than 128: the rotated tail keeps bf16 precision.
            // Quantizing the whole entry corrupts the positional dims and produces noise
            // without a crash — the llama.cpp failure mode this architecture is known for.
            launch_act_quant_f8_prefix(p.kv, p.kv, m, hd, hd - rd, KV_QUANT_BLOCK, NULL_STREAM)?;
        }
        Ok(())
    }

    /// Phase 2: write this step's KV into the ring, then attend.
    ///
    /// **What the attend reads differs by phase, and so does the index space:** prefill attends
    /// the prompt's own KV by ABSOLUTE POSITION, decode attends the ring by SLOT.
    /// [`super::select`] owns that split.
    ///
    /// A compressed layer needs no extra arm, and that is the whole point of the buffer layout:
    /// at prefill the compressor has already written its blocks to the attention scratch's
    /// TAIL, so that buffer IS the concatenation; at decode the compressed region is the
    /// cache's tail behind the ring. Both are one contiguous buffer the kernel indexes
    /// straight into.
    fn attend_rows(&mut self, w: Weights, p: AttnCall, at: Extent, topk: usize) -> Result<()> {
        let d = self.dims;
        let (nh, hd, m) = (d.n_heads, d.head_dim, at.query_rows());
        // SAFETY: every copy below stays inside `cache[0, window)` — the ring region — and the
        // scratch rows it reads were written by phase 1.
        let kv_src: *const f32 = unsafe { self.write_ring(p, at)? };
        // SAFETY: `q` is `m * nh * hd`, `kv_src` covers every row the selection names (this
        // module's second obligation), `attn_sink` is `nh` resident f32, `idxs` is the
        // `m * topk` buffer just uploaded — `topk` IS that upload's column count, handed
        // straight from `upload_selection`'s return — and `o` is `m * nh * hd`.
        unsafe {
            launch_gather_attn_shared_kv(
                p.q,
                kv_src,
                w.attn_sink,
                p.idxs,
                m,
                nh,
                hd,
                topk,
                (hd as f32).powf(-0.5),
                p.o,
                NULL_STREAM,
            )?;
        }
        Ok(())
    }

    /// Write this step's KV entry (or the prompt's last `window` of them) into the ring, and
    /// report which buffer the attend should index.
    ///
    /// Three arms and not two: a prompt SHORTER than the ring seeds it from row 0, a longer one
    /// seeds the last `window` positions with the rotation slot `t % window` holds — and
    /// seeding with the FIRST window instead is right exactly when the prompt fits, which is
    /// why a short fixture cannot see it.
    ///
    /// # Safety
    /// `p.cache` holds at least `window` rows of `hd` f32 and `p.kv` holds the step's rows;
    /// the two are distinct allocations, which [`memcpy_dtod_async`] requires.
    unsafe fn write_ring(&self, p: AttnCall, at: Extent) -> Result<*const f32> {
        // Read off `Dims` here rather than taken as arguments: the ring's width IS
        // `Dims::window`, and a caller free to pass a different one would write into a slot
        // space the selection does not index — in bounds, finite, and wrong.
        let (win, hd) = (self.dims.window, self.dims.head_dim);
        let row = hd * size_of::<f32>();
        // SAFETY: forwarded from this function's contract; every offset below is inside the
        // ring region by the arm's own arithmetic.
        unsafe {
            if !at.is_prefill() {
                let slot = at.start_pos % win;
                memcpy_dtod_async(p.cache.add(slot * hd).cast(), p.kv.cast(), row, NULL_STREAM)?;
                return Ok(p.cache);
            }
            let s = at.seqlen;
            if s <= win {
                memcpy_dtod_async(p.cache.cast(), p.kv.cast(), s * row, NULL_STREAM)?;
                return Ok(p.kv);
            }
            let cut = s % win;
            memcpy_dtod_async(
                p.cache.add(cut * hd).cast(),
                p.kv.add((s - win) * hd).cast(),
                (win - cut) * row,
                NULL_STREAM,
            )?;
            if cut > 0 {
                memcpy_dtod_async(
                    p.cache.cast(),
                    p.kv.add((s - cut) * hd).cast(),
                    cut * row,
                    NULL_STREAM,
                )?;
            }
            Ok(p.kv)
        }
    }

    /// Phase 3: de-rotate the attention output, then the grouped `wo_a` and the `wo_b`
    /// projection back into the sublayer buffer.
    ///
    /// The de-rotation is IN PLACE, which is why no probe can read the attend core's own
    /// output: by the time this returns the pre-image is gone.
    fn output_project(&mut self, w: Weights, p: AttnCall, at: Extent) -> Result<()> {
        let (d, m) = (self.dims, at.query_rows());
        let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
        let gr = d.o_groups * d.o_lora_rank;
        // SAFETY: as in `qkv_project` — live scratch, resident weights, null-stream ordering.
        unsafe {
            // `at.start_pos`, the SAME position phase 1 rotated the query at: `inverse`
            // conjugates the table, so this undoes that rotation exactly. A 0 here would
            // de-rotate every row against position 0 — finite, plausible, and wrong on every
            // decode step past the first.
            launch_rope_adjacent(
                p.o,
                p.freqs,
                m * nh,
                hd,
                rd,
                at.start_pos,
                nh,
                true,
                NULL_STREAM,
            )?;
            // `groups = o_groups`: `o` is `m` rows of `o_groups` contiguous `group_width`-wide
            // head runs, and output row `j` takes run `j / o_lora_rank`. No gather is needed
            // because a group IS a contiguous run of heads, all `head_dim` dims of each.
            launch_gemv_fp8_bf16(
                p.o,
                w.wo_a.packed,
                w.wo_a.scale,
                m,
                gr,
                d.group_width(),
                FP8_BLOCK,
                d.o_groups,
                p.y,
                NULL_STREAM,
            )?;
            launch_act_quant_f8_prefix(p.y, p.y, m, gr, gr, FP8_BLOCK, NULL_STREAM)?;
            gemv_fp8(p.y, w.wo_b, m, (d.dim, gr), p.out, NULL_STREAM)?;
        }
        Ok(())
    }
}
