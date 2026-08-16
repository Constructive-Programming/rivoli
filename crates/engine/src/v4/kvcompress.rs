//! The KV compressor on device: one layer's pooling state and narrowed weights, the launch
//! sequence of `Compressor.forward`, and the placement of the blocks it emits.
//!
//! Ported from `old:src/kvcompress.rs`'s `device` module and `old:src/f4gpu.rs`'s
//! `compress_and_place` / `run_compress`. The deviceless half — which layers compress, what
//! shape their state is, which rows a query may read, and where a block BELONGS — is
//! [`super::geometry`] and [`super::select`], and nothing here re-derives any of it.
//!
//! # Why the two halves are one file here
//!
//! The reference put the launch sequence in `kvcompress.rs` and the placement in `f4gpu.rs`,
//! and the placement then had to restate the launch sequence's own contract twice — "the
//! caller must place `out` at BOTH destinations at prefill" and "a device copy after ONE
//! call, never a second call". Both sentences are about the SAME hazard: `Compressor.forward`
//! does two writes and only one of them is the return value, while every path through the
//! launch sequence read-modify-writes the pooling state before it decides whether to emit. A
//! second call re-deposits the same rows and slides the window again — finite, plausible,
//! wrong. Keeping the emit and the two placements in one file is what makes that one rule
//! with one reader instead of a contract stated in two places.

use super::engine::{NEG_INF_BITS, V4Engine};
use super::geometry::{Geom, KV_QUANT_BLOCK, LayerKind, Quantize};
use super::pin::CompressorWeights;
use super::select::{Extent, compress_dst, compress_offset};
use crate::device::{DeviceBuf, as_le_bytes};
use anyhow::{Context, Result, ensure};
use rivoli_artifact::quant::read_f32;
use rivoli_artifact::v4_config::V4Config;
use rivoli_backend::abi::CompFinish;
use rivoli_backend::{
    NULL_STREAM, fill_u32, launch_act_quant_f4_rotated, launch_act_quant_f8_prefix,
    launch_gemm_bf16, launch_kv_compress_decode, launch_kv_compress_deposit,
    launch_kv_compress_prefill, memcpy_dtod,
};
use rivoli_core::num::f32_to_bf16;
use std::ffi::c_void;

/// Narrow a `[n]` f32 device buffer to the bf16 `launch_gemm_bf16` reads.
///
/// **This exists because handing that launcher the pin's f32 pointer is not a precision loss
/// — it is a ROW-STRIDE error**, and a review caught it in the reference before any device
/// ran. The kernel indexes `w + c * k` in `unsigned short` units, so output row `c` would read
/// f32 elements `[c·k/2, (c+1)·k/2)` — a different row's data, not the low halves of its own.
/// Every compressing layer would pool the wrong weights, finitely and without ever reading out
/// of bounds.
///
/// **The narrowing is EXACT, which is why this is the right fix rather than a conversion
/// cost.** `compressor.{wkv,wgate}.weight` are BF16 in the checkpoint; the converter widens
/// them to f32 because `Compressor.__init__` declares the module fp32, and this narrows the
/// same values back. A widened bf16 round-trips bit-identically, so no value moves — which
/// also means it is not a deviation to name.
///
/// It costs ~0.5 GB of device memory over 43 layers beside the ~1 GB of f32 the pin already
/// holds, plus one read-back per tensor at startup. Placing bf16 in [`super::pin`] instead
/// would be strictly better — it would REPLACE the f32 rather than adding to it — but that
/// changes `CompressorWeights`' field types and every reader with them, and this is the
/// engine's bug to fix, not the loader's.
fn narrow_to_bf16(src: *const f32, n: usize) -> Result<DeviceBuf> {
    let mut bytes = Vec::new();
    // SAFETY: `src` is a pin placement of at least `n` f32 (its `[cd, dim]` extent, computed
    // by the caller from the same `Geom` the kernel is handed), and no kernel is in flight —
    // this runs once, at engine construction, after the pin is fully built.
    unsafe { DeviceBuf::copy_out_raw(src.cast(), n * size_of::<f32>(), &mut bytes)? };
    // `copy_out_raw` sets `bytes` to exactly the length it was given, so the half-width vector
    // is `n` long by construction. What a caller CAN get wrong — passing the wrong extent —
    // is invisible from inside here; that extent is `LayerCompressor::new`'s, computed from
    // the same `Geom` the kernel is handed.
    let half: Vec<u16> = read_f32(&bytes).into_iter().map(f32_to_bf16).collect();
    let mut d = DeviceBuf::new(std::mem::size_of_val(half.as_slice()))?;
    d.copy_in_at(0, as_le_bytes(&half))?;
    Ok(d)
}

/// What one [`LayerCompressor::run`] reads that does not live on the compressor: the
/// activation, and the three pin/rotary pointers the finish stage needs.
///
/// A struct because `ape`, `norm` and `freqs` are three `*const f32` in a row and no type
/// check can tell any of them from another. `norm` is the COMPRESSOR's own RMSNorm weight and
/// not the layer's `kv_norm`, which is also `[d]` and also f32; `freqs` must be the layer's
/// table as [`super::engine::RopeTables`] resolved it, and the plain and YaRN tables have the
/// same type, stride and shape.
#[derive(Clone, Copy)]
pub(super) struct CompInput {
    /// `[rows, dim]` f32 activations — the reference's `x.float()`, unquantized.
    pub x: *const f32,
    /// The model dim, and the reduction extent of both projections. NOT derivable from
    /// [`Geom`], which describes the compressor's OUTPUT geometry only.
    pub dim: usize,
    /// `[ratio, cd]` f32 — the absolute positional table both pool kernels add.
    pub ape: *const f32,
    /// `Compressor.norm`'s weight, `[d]`.
    pub norm: *const f32,
    /// This layer's rotary table, interleaved `(cos, sin)`.
    pub freqs: *const f32,
}

/// One compressed layer's compressor: geometry, narrowed weights, pooling state, scratch.
///
/// `None` on a ratio-0 layer, which is the same answer [`Geom::attention`] gives — so the two
/// cannot disagree about whether a layer compresses.
///
/// The four `[.., cd]` buffers are named `state_*`/`proj_*` rather than after the roles the
/// kernels call them (`kv_state`/`score_state`/`kv`/`score`). That is deliberate: all four are
/// `[.., cd]` f32 and two of them are the pooling STATE while two are this call's
/// PROJECTIONS, so the short names are the pair most worth being unable to confuse.
pub(super) struct LayerCompressor {
    geom: Geom,
    /// `[cd, dim]` bf16 — the pin's f32 `wkv`/`wgate` narrowed to what the GEMM indexes. See
    /// [`narrow_to_bf16`] for why the pin's own pointers cannot be handed over directly.
    w_kv: DeviceBuf,
    w_gate: DeviceBuf,
    state_kv: DeviceBuf,
    state_score: DeviceBuf,
    proj_kv: DeviceBuf,
    proj_score: DeviceBuf,
    /// `[max_m/ratio, d]` — where the emitted blocks land before they are placed.
    blocks: DeviceBuf,
    /// Rows [`Self::proj_kv`] and [`Self::proj_score`] hold. Carried WITH the pointers rather
    /// than beside them: a scratch sized for decode and handed a prefill overruns every buffer
    /// silently, because a `*mut f32` has no length.
    scratch_rows: usize,
}

impl LayerCompressor {
    /// `None` exactly when the layer has no compressor, which [`Geom::attention`] decides.
    ///
    /// `cw` is the pin's answer to the same question, reached through a different file off the
    /// same config. They are ASSERTED to agree rather than assumed: a layer with one and not
    /// the other is a loader bug that would otherwise surface as a null-pointer launch.
    pub(super) fn new(
        cfg: &V4Config,
        kind: LayerKind,
        max_m: usize,
        cw: Option<&CompressorWeights>,
    ) -> Result<Option<Self>> {
        let eps = cfg.rms_norm_eps as f32;
        let geom = Geom::attention(kind, cfg.head_dim, cfg.qk_rope_head_dim, eps);
        let (Some(geom), Some(cw)) = (geom, cw) else {
            ensure!(
                geom.is_none() && cw.is_none(),
                "this layer's Geom and its pin disagree about whether it compresses"
            );
            return Ok(None);
        };
        let f32s = |n: usize| DeviceBuf::new(n * size_of::<f32>());
        let (cd, d, ratio) = (geom.cd(), geom.d(), geom.ratio());
        let mut c = Self {
            geom,
            w_kv: narrow_to_bf16(cw.wkv, cd * cfg.hidden)?,
            w_gate: narrow_to_bf16(cw.wgate, cd * cfg.hidden)?,
            state_kv: f32s(geom.state_len())?,
            state_score: f32s(geom.state_len())?,
            proj_kv: f32s(max_m * cd)?,
            proj_score: f32s(max_m * cd)?,
            blocks: f32s(max_m.div_ceil(ratio) * d)?,
            scratch_rows: max_m,
        };
        c.reset()?;
        Ok(Some(c))
    }

    /// `kv_state` zeroed, `score_state` `-inf`.
    ///
    /// **Not an assumption about the allocator.** `hipMalloc` does not zero and nothing else
    /// in the tree does either, so a state buffer read before it is written is garbage — and
    /// if some future allocator DID zero it, the `score_state` half would become silent rather
    /// than loud, because zeros are live pooling entries at `exp(0 - m)` and not absent ones
    /// ([`NEG_INF_BITS`] carries that argument). This also serves as the between-sequences
    /// clear: a compressor keeping its pooling window across a sequence boundary pools the
    /// previous prompt's tail into this one's first block.
    pub(super) fn reset(&mut self) -> Result<()> {
        let bytes = self.geom.state_len() * size_of::<f32>();
        // SAFETY: both buffers were allocated at exactly `state_len` f32 above, and no kernel
        // is in flight — `reset` runs at construction and between sequences only.
        unsafe {
            fill_u32(self.state_kv.ptr_mut(), 0, bytes)?;
            fill_u32(self.state_score.ptr_mut(), NEG_INF_BITS, bytes)?;
        }
        Ok(())
    }

    /// Run `Compressor.forward` for one call, and report how many blocks landed in
    /// [`Self::blocks`] — **zero** where the reference returns `None`, which at prefill is any
    /// prompt shorter than `ratio` and at decode is every position that does not complete a
    /// block.
    ///
    /// A zero return is not a failure; it is the reference's own control flow, and **the state
    /// writes still happened**. That is the load-bearing half: `Compressor.forward` writes
    /// `kv_state`/`score_state` in BOTH phases and only THEN decides whether to emit, so a
    /// step that emits nothing still deposits. At ratio 128 that is every prompt under 128
    /// tokens and 127 of every 128 decode steps, and skipping the call on a non-emitting step
    /// would build the pooling window out of every 128th token.
    ///
    /// # Safety
    /// Every pointer in `w` must satisfy its field's shape contract and must outlive
    /// `stream`'s completion — not merely the call, which returns as soon as the last
    /// operation is ENQUEUED. This synchronizes nothing. `stream` is a live `hipStream_t`, or
    /// null for the default stream.
    unsafe fn run(&mut self, w: CompInput, at: Extent, stream: *mut c_void) -> Result<usize> {
        at.check_single_row_decode()?;
        ensure!(
            at.seqlen <= self.scratch_rows,
            "v4 compressor: {} rows into scratch sized for {}",
            at.seqlen,
            self.scratch_rows
        );
        // SAFETY: forwarded to both halves from this function's own contract.
        unsafe {
            self.deposit(w, at, stream)?;
            self.emit(w, at, stream)
        }
    }

    /// Both projections and the state deposit — the half that runs on EVERY call.
    ///
    /// `kv = wkv(x)`, `score = wgate(x)`, then the read-modify-write of the pooling state.
    /// `slot0` is the whole difference between the phases and needs no branch: prefill has
    /// `start_pos == 0`, and `0 % ratio` is the 0 it wants.
    ///
    /// # Safety
    /// As [`LayerCompressor::run`], which is the only caller.
    unsafe fn deposit(&mut self, w: CompInput, at: Extent, stream: *mut c_void) -> Result<()> {
        let g = self.geom;
        let (kv, score) = (
            self.proj_kv.ptr_mut().cast::<f32>(),
            self.proj_score.ptr_mut().cast::<f32>(),
        );
        // SAFETY: caller's contract on `w`; both GEMMs write scratch `run` bounded, and the
        // deposit's five buffers are this struct's own at the shapes `Geom` derived.
        unsafe {
            // One loop rather than two spelled-out launches: they differ ONLY in
            // (weight, destination), and writing the other five arguments twice is how
            // `cd` and `dim` get transposed in one copy and not the other — a projection
            // of the right shape onto the wrong extent, which pools finite wrong numbers.
            for (wt, dst) in [(&self.w_kv, kv), (&self.w_gate, score)] {
                launch_gemm_bf16(w.x, wt.ptr().cast(), dst, at.seqlen, g.cd(), w.dim, stream)?;
            }
            launch_kv_compress_deposit(
                kv,
                score,
                w.ape,
                self.state_kv.ptr_mut().cast(),
                self.state_score.ptr_mut().cast(),
                g.abi(),
                at.seqlen,
                at.start_pos % g.ratio(),
                stream,
            )
        }
    }

    /// Pool the completed window(s) into blocks and apply the finish this geometry owes —
    /// **zero** where the reference returns `None`.
    ///
    /// The emission COUNT comes from [`compress_dst`] at region base 0 rather than from a
    /// second `seqlen / ratio` here: `compress_and_place` bounds both destinations with the
    /// same function, so a divergence between the count written and the count reserved is not
    /// expressible.
    ///
    /// # Safety
    /// As [`LayerCompressor::run`], which is the only caller, and the deposit must already
    /// have run for this same call.
    unsafe fn emit(&mut self, w: CompInput, at: Extent, stream: *mut c_void) -> Result<usize> {
        let g = self.geom;
        let (d, rd) = (g.d(), g.rd());
        // Reconstructed rather than passed in. `Geom` was BUILT from a `LayerKind` and stores
        // its ratio, so a second `kind` parameter could only ever disagree with it — layer
        // 42's `Geom` beside layer 41's kind gives a plausible wrong emission count and no
        // guard anywhere sees it. Exact here: `Geom::attention` refuses `Plain`.
        let Some((_, emitted)) = compress_dst(LayerKind::from_ratio(g.ratio()), 0, at) else {
            return Ok(0);
        };
        let out = self.blocks.ptr_mut().cast::<f32>();
        let fin = CompFinish {
            norm: w.norm,
            freqs: w.freqs,
            out,
        };
        // SAFETY: caller's contract; `out` is `emitted * d` writable f32 by this struct's
        // sizing, which `compress_dst` and the `blocks` allocation derive from the same ratio.
        unsafe {
            if at.is_prefill() {
                let (kv, score) = (
                    self.proj_kv.ptr().cast::<f32>(),
                    self.proj_score.ptr().cast::<f32>(),
                );
                launch_kv_compress_prefill(kv, score, w.ape, &fin, g.abi(), emitted, stream)?;
            } else {
                launch_kv_compress_decode(
                    self.state_kv.ptr_mut().cast(),
                    self.state_score.ptr_mut().cast(),
                    &fin,
                    g.abi(),
                    at.start_pos,
                    stream,
                )?;
            }
            // `if self.rotate:` — the two arms are a different ALGORITHM over an identical
            // shape, which is why the choice is a field of `Geom` and not a parameter here:
            // both arms accept every geometry this function can hold, so nothing downstream
            // would reject the wrong one. Matched exhaustively and with no wildcard, so a
            // third `Quantize` cannot silently take the fp8 path.
            match g.quantize() {
                // Dims `[0, d - rd)` at block 64, leaving the rotary tail in bf16 to match how
                // the checkpoint was trained.
                Quantize::PartialFp8 => launch_act_quant_f8_prefix(
                    out,
                    out,
                    emitted,
                    d,
                    d - rd,
                    KV_QUANT_BLOCK,
                    stream,
                )?,
                // The Hadamard spread then fp4 over the WHOLE row, `rd` included. Note what
                // this does NOT do: it takes no `rd` at all, because the indexer's finish has
                // no partial extent. Passing `d - rd` here is the same silent-wrong in the
                // other direction.
                Quantize::HadamardFp4 => launch_act_quant_f4_rotated(out, emitted, d, stream)?,
            }
        }
        Ok(emitted)
    }
}

impl V4Engine<'_> {
    /// Run this layer's compressor for the step and place its blocks at every destination the
    /// reference writes.
    ///
    /// **ONE run; one or two placements.** `Compressor.forward` performs two writes and only
    /// one of them is the return value: it assigns `self.kv_cache[:, :seqlen // ratio]` — the
    /// persistent region every later decode step selects BY POSITION — *and* returns the same
    /// blocks for `Attention.forward` to concatenate onto this step's prompt KV. [`CompFinish`]
    /// carries a single `out`, so the second destination is a device COPY and never a second
    /// run; see [`LayerCompressor::run`] for what a second run would corrupt.
    ///
    /// **Both destinations come from [`compress_dst`]; neither is re-derived.** The two bases
    /// are the SELECTION space's ([`compress_offset`], which is `seqlen` at prefill and the
    /// window at decode) and the persistent `[ring ‖ compressed]` buffer's (the window,
    /// always). At decode the two coincide, and that is not a coincidence needing a branch:
    /// decode's selection space IS the persistent buffer. The `if` below tests the two BASES,
    /// not the phase.
    pub(super) fn compress_and_place(&mut self, layer: usize, at: Extent) -> Result<()> {
        // Off `Dims`, not `cfg`, for `attn::write_ring`'s reason: the placement bases below
        // and the ring writes deposit into the SAME cache buffer, so they must share one
        // authority for its width — "equal by construction" is two authors one edit apart.
        let (win, hd) = (self.dims.window, self.dims.head_dim);
        let li = layer - self.range.start;
        let kind = self.layers[li].kind;
        if self.layers[li].comp.is_none() {
            // A ratio-0 layer has no `Compressor` object at all in the reference, so there is
            // nothing to run and nothing to place. Every other early return below still runs.
            return Ok(());
        }
        // Both destinations, computed BEFORE anything runs. `compress_dst` is pure, so these
        // bounds are PRE-FLIGHT — which matters because the run deposits into the pooling
        // state and (at decode) slides the window, so a bound that failed after it would leave
        // the compressor advanced with no way to retry the step.
        let persist = compress_dst(kind, win, at);
        let sel = compress_dst(kind, compress_offset(win, at), at);
        self.check_block_room(layer, li, persist, sel)?;

        let (blocks, src) = self.run_compress(layer, at)?;
        let Some((sel_base, _)) = sel else {
            // `None` exactly where the reference returns `None`. A DRIFT TRIPWIRE and not a
            // runtime guard, which is the honest reading: the run and `compress_dst` decide
            // from the same `(kind, at)` and today cannot disagree. What it catches is a
            // future edit to one of the two.
            ensure!(
                blocks == 0,
                "layer {layer}: the compressor emitted {blocks} block(s) where compress_dst \
                 names no destination — the two have drifted"
            );
            return Ok(());
        };
        let (persist_base, want) =
            persist.context("compress_dst named a selection destination and no persistent one")?;
        // The same tripwire in the other direction, comparing COUNTS: both copies below are
        // sized from `blocks`, so a divergence would write a different number of rows than the
        // placement reserved.
        ensure!(
            blocks == want,
            "layer {layer}: the compressor emitted {blocks} block(s) at {at:?} where \
             compress_dst reserved {want}"
        );
        let row = hd * size_of::<f32>();
        let cache = self.layers[li].cache.ptr_mut().cast::<f32>();
        let tail = self.a_kv.ptr_mut().cast::<f32>();
        // SAFETY: `src` holds `blocks * head_dim` f32 by `LayerCompressor`'s sizing, and both
        // destinations were bounded by `check_block_room`. `memcpy_dtod` requires non-overlap,
        // which holds: `cache` and `a_kv` are distinct allocations and `src` is a third.
        unsafe {
            // A BLOCKING `hipMemcpy` on the null stream, which is a full serialisation point
            // in the middle of an otherwise-enqueued sequence. It is correct here because this
            // whole arm's attention and compressor run on the null stream; converting the six
            // attention launchers to a real stream without also converting these two copies
            // would leave the hand-off unordered, which is the reference's recorded rebase.
            memcpy_dtod(cache.add(persist_base * hd).cast(), src, blocks * row)
                .context("persisting the compressed region")?;
            if sel_base != persist_base {
                // Prefill only: the transient concatenation this step's attend indexes. At
                // decode the two bases are equal and the copy above already IS the selection
                // space.
                memcpy_dtod(tail.add(sel_base * hd).cast(), src, blocks * row)
                    .context("the prefill kv concatenation tail")?;
            }
        }
        Ok(())
    }

    /// Bound both destinations against the BUFFERS, not against the arithmetic that produced
    /// the row.
    ///
    /// A compressed region sized `max_ctx/ratio` and a slot from `start_pos/ratio` agree only
    /// while `start_pos < max_ctx`, which `forward` enforces; these are the checks that say so
    /// at the write, where a wrong row is an out-of-bounds device write.
    ///
    /// **The selection bound fires only when `a_kv` is actually written.** The reference's
    /// first version sat outside the branch, so at decode it bounded a PERSISTENT-cache row
    /// (window + `start_pos/ratio`) against the attention scratch's row count — two different
    /// coordinate systems with the same type — and fired at the first decode position that
    /// completed a block on any compressing layer.
    fn check_block_room(
        &self,
        layer: usize,
        li: usize,
        persist: Option<(usize, usize)>,
        sel: Option<(usize, usize)>,
    ) -> Result<()> {
        let Some((persist_base, blocks)) = persist else {
            return Ok(());
        };
        ensure!(
            persist_base + blocks <= self.layers[li].cache_rows,
            "layer {layer}: {blocks} block(s) at cache row {persist_base} overrun the {} rows \
             the cache holds",
            self.layers[li].cache_rows
        );
        let (sel_base, _) =
            sel.context("compress_dst named a persistent destination and no selection one")?;
        if sel_base != persist_base {
            let rows = self.a_kv_rows;
            ensure!(
                sel_base + blocks <= rows,
                "layer {layer}: {blocks} block(s) at kv row {sel_base} overrun the {rows}-row \
                 attention scratch"
            );
        }
        Ok(())
    }

    /// Assemble the compressor's inputs, make the call, and hand back `(blocks emitted, the
    /// buffer they are in)`.
    ///
    /// Split from [`V4Engine::compress_and_place`] for the BORROW: the compressor is
    /// `&mut self.layers[li].comp` while the placement reads `self.a_kv` and
    /// `self.layers[li].cache`. Returning `src` rather than letting the caller reach for the
    /// blocks buffer again is what removes a third `Option<LayerCompressor>` probe.
    fn run_compress(&mut self, layer: usize, at: Extent) -> Result<(usize, *const u8)> {
        let li = layer - self.range.start;
        let cw = *self
            .pin
            .layer(layer)?
            .compressor
            .as_ref()
            .with_context(|| format!("layer {layer} compresses but its pin has no compressor"))?;
        let freqs = self.rope.for_layer(self.layers[li].kind)?;
        let (dim, x) = (self.cfg.hidden, self.xw.ptr().cast::<f32>());
        // Guaranteed `Some` by the caller's early return, reported rather than panicked.
        let c = self.layers[li].comp.as_mut().with_context(|| {
            format!("layer {layer} compresses but its DeviceLayer has no compressor state")
        })?;
        let w = CompInput {
            x,
            dim,
            ape: cw.ape,
            norm: cw.norm,
            // The layer's rotary table, resolved through the ONE site.
            freqs,
        };
        // SAFETY: every pointer above is either a pin placement outliving this engine or a
        // `DeviceBuf` field of `self`, at the shape `CompInput` documents. The null stream is
        // what the rest of this arm's attention path runs on.
        let blocks = unsafe { c.run(w, at, NULL_STREAM) }
            .with_context(|| format!("layer {layer} compressor at {at:?}"))?;
        Ok((blocks, c.blocks.ptr()))
    }
}
