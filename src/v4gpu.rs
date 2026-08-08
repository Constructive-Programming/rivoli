//! The DeepSeek-V4-Flash layer loop — `Transformer.forward` on device.
//!
//! GLM's counterpart is [`crate::gpu`]. This is a separate module rather than a second arm
//! inside it, and the reason is not file size: the two share **no** per-layer step. V4 carries
//! a `[s, hc_mult, dim]` hyper-connection residual where GLM carries `[s, dim]`; its attention
//! is shared-K=V MQA over a sliding window plus a compressed region where GLM's is MLA; its
//! router is `sqrt(softplus(·))` with a renormalisation GLM does not perform; and its experts
//! are fp4 with an `ExpertDescF4` that GLM's dispatch explicitly refuses (`gpu.rs`'s
//! `launch_expert_range` bails on `RoutedFmt::F4` naming this file's launcher). What the two DO
//! share is the pool — `memory::routed::RoutedPool` — and the ticket protocol over it, which is
//! exactly why that moved out of `Pin` rather than being copied.
//!
//! # What gates this loop, and what does not
//!
//! `distinct` / `longest repeated block` cannot see anything wrong here (CLAUDE.md; they have
//! misled three investigations in this repo). The gate is `src/v4oracle`'s per-layer goldens at
//! REAL weights, reachable through `bin/v4-oracle emit`, and `tests/v4_loop.rs` is what drives
//! it. **Fluent text out of this file is evidence of nothing.**
//!
//! Two deviations from the reference are live and named at their call sites, so no reader has
//! to discover them from a number:
//!
//! * the resident fp8 **shared expert is unclamped** — `v4oracle::Defect::SwigluUnclamped`, one
//!   contribution in seven, on every layer. See [`V4Engine::shared_expert`].
//! * on the 21 ratio-4 layers the compressed block set is chosen **positionally** rather than by
//!   the lightning indexer's scores. Below `4 * (index_topk + 1)` positions the SET is the same
//!   (fixed by the causal mask) and only the online-softmax fold ORDER differs; above it the two
//!   disagree on the set, which is why [`V4Engine::new`] refuses that context outright.
//! * **the MoE output is not bf16-rounded.** `MoE.forward` ends `return y.type_as(x)`
//!   (model.py:649) and `Oracle::moe` ends `round_bf16(&mut y)`; the engine's `sub` after
//!   [`launch_moe_acc_drain`] holds a bf16 shared-expert output plus a fixed-point routed sum with
//!   no final round, and `hc_post` consumes that. Attention's output IS rounded (the `wo_b` GEMV
//!   does it), so the two sublayers are inconsistent. It needs a kernel — there is no bare
//!   round-to-bf16 launcher and `launch_v4_rmsnorm` also norms — so it is NAMED here rather than
//!   fixed. Magnitude: ~half a bf16 ULP on `hc_post`'s `post * x` term, well inside the gate's
//!   bound, which is exactly why naming it matters: a reader seeing `ffn_out` move would otherwise
//!   attribute all of it to the unclamped shared expert and stop looking. Found by review.
//!
//! # Synchronisation this loop OWNS
//!
//! Six of the launchers `attn::v4::attention` uses take no stream, neither does
//! `v4compress::compress`, and `memcpy_dtod` is a blocking `hipMemcpy` on the null stream
//! (`kernels/linalg.hip:692`). So attention, the compressor, the norms and the router GEMV all
//! run on the **null stream**, and only the MoE experts run on `compute_stream`/`miss_stream` —
//! which are `hipStreamNonBlocking` and therefore do NOT implicitly join the null stream. Every
//! boundary between the two is an explicit sync in this file, named at its site:
//!
//! 1. before the expert launches, so `xq` (null stream) is complete when a non-blocking stream
//!    reads it;
//! 2. after them and before [`launch_moe_acc_drain`], whose own contract is that "EVERY stream
//!    that accumulated into `acc` must already have completed";
//! 3. one per layer, so layer `L+1`'s first atomic cannot race `L`'s drain;
//! 4. one at the end of a forward, before the argmax D2H.
//!
//! All four are `device_sync`, which is stronger than the two `stream_signal` awaits GLM uses
//! and was chosen for a reason worth recording: this loop has no other work in flight to
//! overlap a narrower wait with, and a nested executor (`block_on` inside `block_on`) is the
//! failure that shape invites. When the attention set takes a stream, (1)-(3) become the
//! redundant belt they should be and narrowing them is a measured change, not a free one.
//!
//! **This loop does not claim to be race-free and no comment here should be read as saying
//! so.** Every function below takes `stream` and threads it to whichever launcher accepts one,
//! so the conversion is an argument per call site rather than a restructuring. The launchers exist
//! on another branch; the one item requirement 9 never named is the blocking `memcpy_dtod` above —
//! converting the six without also taking `memcpy_dtod_async` leaves the overlap defeated and this
//! header claiming otherwise. Marked at the two placements that use it.
//!
//! **And three launchers here ALREADY take a stream and are handed `null_mut()`:**
//! [`launch_hc_pre`], [`launch_hc_post`] and [`launch_moe_acc_drain`]. That is correct today —
//! everything around them is null-stream, so a non-null one would reorder against the norms — but
//! it is the reason (1)-(3) above cannot narrow the day the other six convert: the
//! hyper-connections and the accumulator drain would still serialise the layer, and the six-launcher
//! count would read as complete. Three one-token edits, on no requirement's list; found by review,
//! which is the only reason they are written down at all.

use crate::artifact::model::V4Config;
use crate::artifact::quant::{FP8_BLOCK, f4_expert_stride, read_f32};
use crate::attn::{Sel, v4, v4_topk_idxs};
use crate::backend::hip::{
    ExpertDescF4, device_sync, fill_u32, launch_act_quant_f8, launch_argmax, launch_gemv_f32,
    launch_hc_post, launch_hc_pre, launch_moe_acc_drain, launch_moe_expert_range_f4, launch_swiglu,
    launch_v4_dense_gemm_bf16, launch_v4_embed_bf16_row, launch_v4_gemv_fp8, launch_v4_hc_head,
    launch_v4_rmsnorm, memcpy_dtod,
};
use crate::backend::{Event, NULL_STREAM, Stream};
use crate::gpu::as_le_bytes;
use crate::math::{Scoring, f32_to_bf16, route_into};
use crate::memory::device::DeviceBuf;
use crate::memory::pin::{Fp8Weight, V4Compressor, V4Pin, V4Route};
use crate::memory::routed::ExpertSlot;
use crate::v4compress::{
    Buffers, Finish, Geom, LayerKind, RopeParams, compress, compress_dst, compress_offset,
    freqs_cis, rope_for_layer,
};
use anyhow::{Context, Result, ensure};
use std::ffi::c_void;

/// Rows of the fixed-point MoE accumulator — ONE PER STREAM, which is what
/// [`launch_moe_acc_drain`]'s `rows` argument means and not one per expert. Residents
/// accumulate into row 0 on the compute stream and misses into row 1 on the miss stream, so the
/// two never contend for a cache line and there is no cross-stream join. Same value and same
/// reason as `gpu.rs`'s.
const MOE_ACC_ROWS: usize = 2;

/// `-inf` as f32 bits, for [`fill_u32`].
///
/// `score_state` must be `-inf`-initialised and **not** zeroed: a never-written pooling slot has
/// to weigh `exp(-inf - m) == 0`, where a zero weighs `exp(0 - m)` — a plausible number and a
/// wrong pooling window. S3 requirement 3. Named rather than spelled at each of its call sites
/// because `fill_u32` takes a `u32` and the defect is a `0` there.
const NEG_INF_BITS: u32 = f32::NEG_INFINITY.to_bits();

/// The ratio whose positional compressed selection is the tightest constraint, and the only
/// layer class that has an indexer at all.
///
/// `Attention.__init__` builds an `Indexer` only where `compress_ratio == 4` (model.py:474), so
/// the ratio-128 layers have nothing to stand in for and `Sel::n_comp` never refuses there.
const INDEXED_RATIO: usize = 4;

/// The context past which the positional compressed selection stops agreeing with
/// `Indexer.forward` on the block SET.
///
/// `Sel::n_comp` is what actually refuses, on `start_pos + seqlen`. This is the same bound
/// stated at STARTUP, so a caller learns it before the pin reads 9 GB rather than 41 layers into
/// the first token. Takes `index_topk` rather than `&V4Config` so it is testable without one,
/// and computed rather than written as 2052, because both factors are config values.
fn positional_context_limit(index_topk: usize) -> usize {
    INDEXED_RATIO * (index_topk + 1)
}

/// Both rotary tables, device-resident, with the per-layer selection happening at exactly ONE
/// site.
///
/// **This is the enforcing construction `docs/investigations/v4-flash-port.md` records as owed
/// and has handed back three times** — "`Io` must be built by something that takes `LayerKind`
/// and calls `rope_for_layer` itself, so the two-table selection has exactly one site".
/// `attn::v4::Io::freqs` is a bare `*const f32`; the ratio-0 table and the YaRN one have the
/// same type, stride and shape, and substituting one for the other is `Defect::RopeNoYarn` —
/// plausible frequencies at every scale and fluent wrong text. The port has MEASURED that its
/// numeric gate cannot see it at `ratio4/decode` (separation 8 bf16 codes against a `RESOLVABLE`
/// floor of 64, i.e. half an e4m3 step), so no tolerance anywhere would have caught a caller
/// that got this wrong.
///
/// **The cache is content-addressed, not an arm per [`LayerKind`].** Keying on
/// `(theta, original_seq_len)` — the pair `rope_for_layer` moves together, and the only two
/// fields the model's two tables differ in — makes this a *memo over* `rope_for_layer` rather
/// than a second copy of its decision. A `match kind { Plain => .., _ => .. }` accessor was
/// written first and rejected: it is exactly the "second place to state the same fact, and
/// therefore a second place to state it wrongly" that got the `RopeTable` newtype deleted in
/// `2445645`.
///
/// One construction serves both classes. `freqs_cis(rope_for_layer(.., Plain))` is asserted
/// equal to `attn::v4_rope_table_ratio0` by
/// `tests/v4_attn.rs::the_two_rope_table_constructions_agree_on_the_un_yarned_table`, so there
/// is no reason to carry the second builder here as well.
struct RopeTables {
    /// `(theta bits, original_seq_len) -> table`. At most two entries on any real config; a
    /// `Vec` because a two-entry linear scan beats a hash and the key half is `f32::to_bits`,
    /// which has no `Hash` worth reaching for.
    tables: Vec<((u32, usize), DeviceBuf)>,
    compressed: RopeParams,
    rope_theta: f32,
    max_pos: usize,
}

impl RopeTables {
    /// `max_pos` is the context this decode was sized for; it enters no weight and only sizes
    /// the tables.
    fn new(cfg: &V4Config, max_pos: usize) -> Self {
        Self {
            // Built ONCE, from the config's own rotary fields. `RopeParams` the type is what
            // keeps `compress_rope_theta` and `original_seq_len` travelling together: selecting
            // one without the other is `Defect::RopeNoYarn` in one direction and
            // `Defect::RopeBaseThetaEverywhere` in the other, two distinct silent-wrongs the
            // oracle enumerates separately.
            compressed: RopeParams {
                rope_head_dim: cfg.qk_rope_head_dim,
                theta: cfg.compress_rope_theta as f32,
                original_seq_len: cfg.rope_scaling.original_max_position_embeddings,
                factor: cfg.rope_scaling.factor as f32,
                beta_fast: cfg.rope_scaling.beta_fast as f32,
                beta_slow: cfg.rope_scaling.beta_slow as f32,
            },
            rope_theta: cfg.rope_theta as f32,
            max_pos,
            tables: Vec::new(),
        }
    }

    /// This layer's rotary table. **The one site where the two-table selection happens.**
    ///
    /// `&mut self` because it uploads on first use: a table is `max_pos * rope_head_dim` f32
    /// (512 KB at 2048 positions), and building both eagerly would upload one a run may never
    /// touch — a 43-layer artifact reaches both, a 2-layer fixture only one.
    fn for_layer(&mut self, kind: LayerKind) -> Result<*const f32> {
        let p = rope_for_layer(self.compressed, self.rope_theta, kind);
        let key = (p.theta.to_bits(), p.original_seq_len);
        if let Some((_, buf)) = self.tables.iter().find(|(k, _)| *k == key) {
            return Ok(buf.ptr().cast());
        }
        // Interleaved `(cos, sin)` — the layout `launch_v4_rope` indexes, and the one
        // `v4_rope_table_ratio0` produces. Written as a push pair rather than a `flat_map` over
        // the tuple so it reads as the layout it is.
        let pairs = freqs_cis(p, self.max_pos);
        let mut flat = Vec::with_capacity(pairs.len() * 2);
        for (c, s) in pairs {
            flat.push(c);
            flat.push(s);
        }
        let mut buf = DeviceBuf::new(std::mem::size_of_val(flat.as_slice()))?;
        buf.copy_in_at(0, as_le_bytes(&flat))?;
        let ptr = buf.ptr().cast();
        self.tables.push((key, buf));
        Ok(ptr)
    }
}

/// The rows one compressor/attention call covers, and where they start.
///
/// A struct because `seqlen` and `start_pos` are two `usize` in a row, every permutation
/// type-checks, and the failure is not a panic: `compress`, `should_compress`, `compress_dst`,
/// `compress_offset` and `window_topk` all take exactly this pair, and swapping it pools the
/// wrong window at the wrong position. Same argument `attn::Sel` makes about its own four.
#[derive(Clone, Copy, Debug)]
struct Extent {
    /// Query rows: the prompt length at prefill, 1 at decode.
    seqlen: usize,
    /// 0 means prefill, throughout the reference.
    start_pos: usize,
}

/// Narrow a `[n]` f32 device buffer to the bf16 `launch_v4_dense_gemm_bf16` reads.
///
/// **This exists because handing that launcher the pin's f32 pointer is not a precision loss —
/// it is a row-stride error, and a review caught it before any device ran.** The kernel indexes
/// `w + (size_t)c * k` in `unsigned short` units (`kernels/v4compress.hip:94`), so output row `c`
/// would read f32 elements `[c·k/2, (c+1)·k/2)` — a different row's data, not the low halves of
/// its own. Every one of the 41 compressing layers would pool the wrong weights, finitely and
/// without ever reading out of bounds.
///
/// **The narrowing is EXACT, which is why this is the right fix rather than a conversion cost.**
/// `layers.N.attn.compressor.{wkv,wgate}.weight` are **BF16** in the checkpoint (verified:
/// `[1024, 4096]` at ratio 4, `[512, 4096]` at ratio 128, matching `[cd, dim]`); `convert_v4`
/// widens them to f32 because `Compressor.__init__` declares the module fp32, and this narrows
/// the same values back. A widened bf16 round-trips bit-identically, so no value moves — which
/// also means it is not a deviation to name. `tests/v4_attn.rs::LayerCompressor::new` has always done this
/// (`u16b(&bf16_rows(&cw.wkv))`); the engine simply had no counterpart.
///
/// Adds **≈0.5 GB** of device memory over 43 layers — 21 ratio-4 layers at 16.8 MB for the
/// `wkv`+`wgate` pair plus 20 ratio-128 layers at 8.4 MB — beside the ≈1 GB of f32 the pin already
/// holds, plus one read-back per tensor at startup. (An earlier note here said "~1 GB", which is
/// the f32 that is already there, not the addition.) Placing bf16 in `V4Pin` instead would be strictly better —
/// it would REPLACE the f32 rather than adding to it — but that changes `V4Compressor`'s field
/// types and `tests/v4_pin.rs` with it, and this is the loop's bug to fix.
fn narrow_to_bf16(src: *const f32, n: usize) -> Result<DeviceBuf> {
    let mut bytes = Vec::new();
    // SAFETY: `src` is a pin placement of at least `n` f32 (its `[cd, dim]` extent, computed by
    // the caller from the same `Geom` the kernel is handed), and no kernel is in flight — this
    // runs once, at engine construction, after the pin is fully built.
    unsafe { DeviceBuf::copy_out_raw(src.cast(), n * size_of::<f32>(), &mut bytes)? };
    // `copy_out_raw` sets `bytes` to exactly the length it was given, so `half.len() == n` holds
    // by construction. A check for it was written here and cut: it could not fire, and what a
    // caller CAN get wrong — passing the wrong extent — is invisible from inside this function.
    // The extent is `LayerCompressor::new`'s, computed from the same `Geom` the kernel is handed.
    let half: Vec<u16> = read_f32(&bytes).into_iter().map(f32_to_bf16).collect();
    let mut d = DeviceBuf::new(std::mem::size_of_val(half.as_slice()))?;
    d.copy_in_at(0, as_le_bytes(&half))?;
    Ok(d)
}

/// One compressed layer's compressor: geometry, pooling state, scratch.
///
/// `None` on a ratio-0 layer in both halves at once — `Geom::attention` refuses
/// [`LayerKind::Plain`] and `V4LayerPin::compressor` is `None` there — so the pair cannot
/// disagree about whether a layer compresses.
///
/// The four buffers are named `state_*`/`proj_*` rather than after `Buffers`' own fields, which
/// call them `kv_state`/`score_state`/`kv`/`score`. That is deliberate: all four are `[.., cd]`
/// f32 and two of them are the pooling STATE while two are this call's PROJECTIONS, so the
/// short names are the pair most worth being unable to confuse at the literal below.
struct LayerCompressor {
    geom: Geom,
    /// `[cd, dim]` bf16 — the pin's f32 `wkv`/`wgate` narrowed to what the GEMM indexes. See
    /// [`narrow_to_bf16`] for why the pin's own pointers cannot be handed over directly.
    w_kv: DeviceBuf,
    w_gate: DeviceBuf,
    state_kv: DeviceBuf,
    state_score: DeviceBuf,
    proj_kv: DeviceBuf,
    proj_score: DeviceBuf,
    blocks: DeviceBuf,
}

impl LayerCompressor {
    fn new(
        cfg: &V4Config,
        kind: LayerKind,
        max_m: usize,
        cw: Option<&V4Compressor>,
    ) -> Result<Option<Self>> {
        let eps = cfg.rms_norm_eps as f32;
        let geom = Geom::attention(kind, cfg.head_dim, cfg.qk_rope_head_dim, eps);
        let (Some(geom), Some(cw)) = (geom, cw) else {
            // Asserted rather than assumed: `Geom::attention` refuses `LayerKind::Plain` and
            // `V4LayerPin::compressor` is `None` there, and the two are decided in different
            // files off the same config. A layer with one and not the other is a loader bug that
            // would otherwise surface as a null-pointer launch.
            ensure!(
                geom.is_none() && cw.is_none(),
                "this layer's Geom and its pin disagree about whether it compresses"
            );
            return Ok(None);
        };
        let f32s = |n: usize| DeviceBuf::new(n * size_of::<f32>());
        let (cd, ents, d, ratio) = (geom.cd(), geom.ents(), geom.d(), geom.ratio());
        let mut c = Self {
            geom,
            w_kv: narrow_to_bf16(cw.wkv, cd * cfg.hidden)?,
            w_gate: narrow_to_bf16(cw.wgate, cd * cfg.hidden)?,
            state_kv: f32s(ents * cd)?,
            state_score: f32s(ents * cd)?,
            proj_kv: f32s(max_m * cd)?,
            proj_score: f32s(max_m * cd)?,
            blocks: f32s(max_m.div_ceil(ratio) * d)?,
        };
        c.reset()?;
        Ok(Some(c))
    }

    /// `kv_state` zeroed, `score_state` `-inf`.
    ///
    /// **Not an assumption about the allocator.** `hipMalloc` does not zero and nothing else in
    /// the tree does either, so a state buffer read before it is written is garbage — and if some
    /// future allocator DID zero it, the `score_state` half would become silent rather than loud,
    /// because zeros are live pooling entries at `exp(0 - m)` and not absent ones. This also
    /// serves as the between-sequences clear: a compressor keeping its pooling window across a
    /// sequence boundary pools the previous prompt's tail into this one's first block.
    fn reset(&mut self) -> Result<()> {
        let bytes = self.geom.state_len() * size_of::<f32>();
        // SAFETY: both buffers were allocated at exactly `state_len` f32 above, and no kernel is
        // in flight — `reset` runs at construction and between sequences only.
        unsafe {
            fill_u32(self.state_kv.ptr_mut(), 0, bytes)?;
            fill_u32(self.state_score.ptr_mut(), NEG_INF_BITS, bytes)?;
        }
        Ok(())
    }
}

/// One layer's persistent decode state.
struct DeviceLayer {
    kind: LayerKind,
    /// `[window_size + max_ctx/ratio, head_dim]` f32 — the ring FIRST, then the compressed
    /// region, contiguous and in that order. That is what `attn::v4::Io::cache` requires and
    /// what makes the reference's `compressor.kv_cache = self.kv_cache[:, win:]` a *view* rather
    /// than a second buffer; decode attends the whole thing and the selection's compressed
    /// columns are `window_size + block`.
    cache: DeviceBuf,
    /// Rows in `cache`. Carried because `DeviceBuf` has no length and the placement indexes it
    /// by row — the same reason `tests/v4_attn.rs::Gpu` carries `ring_rows`.
    cache_rows: usize,
    comp: Option<LayerCompressor>,
}

impl DeviceLayer {
    fn new(
        cfg: &V4Config,
        kind: LayerKind,
        max_ctx: usize,
        max_m: usize,
        cw: Option<&V4Compressor>,
    ) -> Result<Self> {
        // `max_ctx / ratio` is the reference's own sizing of `kv_cache[:, window_size:]`, and 0
        // on a ratio-0 layer. `div_ceil` and not `/`: at `max_ctx = 13, ratio = 4` the block
        // completed at position 11 is real and `13 / 4 == 3` sizes the region exactly, but at
        // `max_ctx = 14` a plain divide is one row short of the slot `compress_dst` computes.
        let blocks = kind.compressor_ratio().map_or(0, |r| max_ctx.div_ceil(r));
        let rows = cfg.sliding_window + blocks;
        let mut s = Self {
            kind,
            cache: DeviceBuf::new(rows * cfg.head_dim * size_of::<f32>())?,
            cache_rows: rows,
            comp: LayerCompressor::new(cfg, kind, max_m, cw)?,
        };
        s.reset(cfg)?;
        Ok(s)
    }

    /// Clear everything a new sequence must not inherit.
    ///
    /// **The compressed region specifically**, which was asserted nowhere — in `src/` or in
    /// `tests/` — when this loop was written. It matters twice.
    ///
    /// A stale compressed row is attended BY POSITION on every later step, because the
    /// compressed selection is the positional full prefix: leftover blocks from a previous
    /// prompt are weighted `exp(l - max)` and silently mixed in. And it is the second of the two
    /// premises under which the port measured requirement 2 (the decode slot) to be invisible to
    /// attention output — "a compressed region reused across sequences without clearing holds
    /// stale rows rather than zeros, so the two rules read different values". Clearing keeps that
    /// measurement's premise true; not clearing would make the append bug observable *and* the
    /// output wrong, which is not a trade worth taking.
    ///
    /// The ring half needs it for the same reason with a shorter tail: `window_topk` masks slots
    /// the sequence has not reached with `-1`, so unwritten ring slots are unread — but only
    /// while that mask is right, and a zeroed ring costs one `fill_u32` per layer per sequence.
    fn reset(&mut self, cfg: &V4Config) -> Result<()> {
        // SAFETY: `cache` was allocated at exactly this size; `reset` runs at construction and
        // between sequences, with no kernel in flight.
        unsafe {
            let bytes = self.cache_rows * cfg.head_dim * size_of::<f32>();
            fill_u32(self.cache.ptr_mut(), 0, bytes)?;
        }
        if let Some(c) = &mut self.comp {
            c.reset()?;
        }
        Ok(())
    }
}

/// Build one expert's `ExpertDescF4` from its resolved pool slot.
///
/// Six byte addresses in, six byte addresses out — which is the whole reason `ExpertSlot`
/// stopped carrying a typed `scales` pointer. `.f4`'s e8m0 scales are ONE byte, `.i4`'s are f32
/// and `.vq3`'s bf16, and `gpu.rs::desc_of_vq` casts to `*const u16` for its two formats; this
/// one casts nothing, because `ExpertDescF4` already says `*const u8`.
///
/// What it cannot check: `.f4` and `.i4` tile IDENTICALLY for 25% of all `i_dim`, both models'
/// dimensions included (`quant::f4_slot_offsets` carries the identity), so a slot resolved
/// through the wrong format's offsets finds every projection at exactly the right address and
/// then decodes e2m1 nibbles against the wrong scale grid. The header magic and the descriptor
/// TYPE are the entire separation — the type is this function's return and the magic is
/// `ExpertSet::open_routed`'s.
fn desc_of_f4(s: &ExpertSlot) -> ExpertDescF4 {
    ExpertDescF4 {
        gate_packed: s.gate.packed,
        gate_scale: s.gate.scale,
        up_packed: s.up.packed,
        up_scale: s.up.scale,
        down_packed: s.down.packed,
        down_scale: s.down.scale,
    }
}

/// A descriptor that faults if it is ever read.
///
/// `descs` is written in LAUNCH order and sized `n_desc`, so 250 of its 256 entries sit past
/// this token's selection and no launch names them (the dispatches below cover exactly
/// `[0, sel.len())`: one resident range plus one single-descriptor range per miss). Filling
/// them with nulls rather than with a copy of some resolved expert is the difference between
/// a fault and a plausible wrong weight the day a range is computed wrongly.
fn desc_never_read() -> ExpertDescF4 {
    let n = std::ptr::null();
    ExpertDescF4 {
        gate_packed: n,
        gate_scale: n,
        up_packed: n,
        up_scale: n,
        down_packed: n,
        down_scale: n,
    }
}

/// The first `n` f32 of a device buffer, host-side.
///
/// Every probe in this file wants exactly this and four of them wanted it in one function, which
/// `build.rs`'s jscpd gate does not permit spelled four times. Takes an element count and not a
/// byte count, because every caller has the former and the `* size_of::<f32>()` was the part that
/// was easy to drop.
///
/// **Does NOT sync.** The caller owns that, and every one of them does it once for the whole
/// group rather than once per tensor — `device_sync` after four blocking D2H copies would be
/// three joins that prove nothing.
fn read_prefix(b: &DeviceBuf, n: usize) -> Result<Vec<f32>> {
    let mut bytes = Vec::new();
    b.copy_out_prefix(&mut bytes, n * size_of::<f32>())?;
    Ok(read_f32(&bytes))
}

/// Everything ONE [`V4Engine::probe_attn_stages`] call leaves readable, in pipeline order.
///
/// A struct and not a 4-tuple: all four are `Vec<f32>`, so every permutation type-checks and the
/// failure is a comparison against the wrong golden. Same argument [`Extent`] makes about its two
/// `usize`.
///
/// `attn_core_out` is deliberately absent and **cannot** be added — the output de-rotation is IN
/// PLACE on `s.o` (`launch_v4_rope(s.o, .., inverse = true)`), so by the time `attention` returns
/// the pre-image is gone. `tests/v4_attn.rs` drives `sparse_attn` separately for exactly that
/// reason; this makes the same partition available at real weights, minus that one cell.
///
/// # These four are NOT equally sharp
///
/// [`AttnStages::kv_entry`] is the sharpest instrument here and [`AttnStages::attn_out`] is the
/// bluntest — the opposite of how the two read, since `attn_out` is the one with the familiar name
/// and the block-shaped meaning. **Bisect from `kv_entry` outwards, not from `attn_out` inwards.**
///
/// The reason is structural rather than a property of this port: the block performs three fp8
/// activation requantizations, each a step function that amplifies re-association noise ~16x, and
/// a tensor's distance from the oracle is set by how many of them sit upstream of it. The
/// mechanism, the measured amplification and the per-tensor ladder are stated ONCE, in
/// `tests/v4_loop.rs`'s `# CORRECTED 2026-08-05` header. They are deliberately not restated here:
/// an earlier draft copied the table into this doc and the two had already contradicted each other
/// on the size of a bf16 ULP within the same session — jscpd does not see comments, so nothing
/// would have caught it.
pub struct AttnStages {
    /// `[m, n_heads, head_dim]` — `s.q` after QK-norm and RoPE. Golden `L{l}.{tag}.q`.
    pub q: Vec<f32>,
    /// `[m, head_dim]` — `s.kv` after `kv_norm`, RoPE and the partial block-64 `act_quant`.
    /// Golden `L{l}.{tag}.kv_entry`.
    pub kv_entry: Vec<f32>,
    /// `[m, n_heads, head_dim]` — `s.o` after `sparse_attn` AND the in-place de-rotation.
    /// Golden `L{l}.{tag}.attn_derot`.
    pub attn_derot: Vec<f32>,
    /// `[m, dim]` — `io.out`: the grouped `wo_a`, `act_quant`, then `wo_b`.
    /// Golden `L{l}.{tag}.attn_out`.
    pub attn_out: Vec<f32>,
}

impl AttnStages {
    /// Each tensor with the golden SUFFIX that names it and the `max_rel` bound derived for it,
    /// **sharpest first** — the order of the amplifier ladder, so a caller printing these prints
    /// the most diagnostic row first and the first line to move is the one to believe.
    ///
    /// This exists because the struct's whole argument for not being a 4-tuple — that four
    /// `Vec<f32>` in a row are permutable and the failure is a silent comparison against the wrong
    /// golden — was defeated at its only call site, which paired the fields with their names in a
    /// hand-written array. Found by review. Pairing them HERE, next to the field declarations,
    /// is the only place the two can be checked against each other by eye.
    ///
    /// # Where the bounds come from, and how little they buy
    ///
    /// **Derived, not chosen**: each is `sqrt(envelope_max * weakest_defect_above_the_envelope)`,
    /// where the envelope is what a correct implementation produces given the `attn_norm_out`
    /// deviation the device actually has (26 of 53,248 elements at 1 bf16 ULP, measured on
    /// gfx1151 2026-08-06) and the defect column is measured over the **in-scope subset of
    /// `Defect::ALL`**, not over a hand-picked pair. §8 of
    /// `docs/measurement/probes/v4_attn_amplification.py` prints both inputs.
    ///
    /// | tensor | envelope max | weakest defect ABOVE it | bound | ratio | device |
    /// |---|---:|---|---:|---:|---:|
    /// | `kv_entry` | 2.56 | `RopeHalfSplit` 116 | **17** | 45x | 0.95 |
    /// | `q` | 49.5 | `RopeFirstDims` 1,482 | **275** | 30x | 27.1 |
    /// | `attn_derot` | 20.1 | `SkipKvActQuant` 26.8 | **23** | **1.3x** | 6.93 |
    /// | `attn_out` | 53.8 | `KvActQuantWholeTensor` 94.6 | **71** | **1.6x** | 35.0 |
    ///
    /// **CORRECTED 2026-08-06.** A first version computed the defect column from TWO variants
    /// (`RopeHalfSplit`, `SkipQkNorm`) and shipped 17/420/130/350 with a claimed "41x to 71x on
    /// every row". Adversarial review measured all eighteen in-scope defects: `RopeHalfSplit` is
    /// the weakest only for `kv_entry`; for `q` it is the STRONGEST of three, and for `attn_derot`
    /// and `attn_out` the true weakest are 30x and 24x smaller. The old bounds were up to 5.6x
    /// looser than the rule they cited, and **seven of eighteen defects cleared all four at once**
    /// — including `QkNormAfterRope`, which moves `attn_out` by 32.7, i.e. LESS than the device's
    /// own 35.0, so no bound on that tensor can ever separate it.
    ///
    /// **So read the ratio column, not the bound.** `kv_entry` and `q` separate by 45x and 30x and
    /// are real gates. `attn_derot` and `attn_out` separate by 1.3x and 1.6x: they are barely
    /// gates at all, and that is a property of `max_rel` on a tensor two and three fp8
    /// requantizations deep, not something a better constant fixes. `max_rel` is floor-dominated
    /// and near-blind to SCALING defects — `SkipQkNorm` roughly doubles every element of `q` and
    /// reads 1.07. The statistic that separated every defect review could construct is the
    /// differing-element FRACTION, and nothing in this tree asserts it yet; that is the owed work
    /// recorded in `tests/v4_loop.rs`.
    ///
    /// Two further limits. The envelope draws its perturbed element POSITIONS uniformly at random,
    /// while the device's are wherever `hc_pre`'s fold order crossed a bf16 boundary — clustering
    /// them instead moves `attn_derot`'s envelope over an 18x range. And every number here is
    /// `L0.pre`; the same four constants gate `L1` and both decode cells, where the input
    /// deviation is up to 5x larger.
    pub fn scored(&self) -> [(&'static str, &[f32], f32); 4] {
        [
            ("kv_entry", &self.kv_entry, 1.7e1),
            ("q", &self.q, 2.75e2),
            ("attn_derot", &self.attn_derot, 2.3e1),
            ("attn_out", &self.attn_out, 7.1e1),
        ]
    }
}

/// Per-token decode phase buckets — V4's half of GLM's always-on PROFILE line, under
/// GLM's discipline (`gpu.rs::Profile`): every span wraps host work or a blocking call the
/// decode already pays, so accumulating costs two `Instant` reads per bracket and adds no
/// device sync, event, or join. Three fields and not `ProfileSummary`, because that struct's
/// other axes (class spans, the indexer, MTP) would print zeros here and a zero that means
/// "not measured" reading as "measured zero" is this repo's named telemetry trap.
///
/// What each bucket brackets, exactly — the names follow GLM's and approximate like GLM's:
///
/// * `route_ns` — the gate-logits D2H plus [`V4Engine::route_row`]'s host math. The D2H is
///   the layer's FIRST blocking call and the null stream carries everything before it, so
///   the wait drains the attention half, both `hc_pre`/norm launches and the gate GEMV
///   still in flight: most of `route` is attention GPU time, not routing. GLM's `route`
///   contains the same drain behind the same D2H, which is what keeps the columns
///   comparable — and is why the HB=16 attention win was quoted in `route` there.
/// * `moe_ns` — [`V4Engine::shared_expert`] plus [`V4Engine::routed_experts`]: the shared
///   expert's launches, the pool submit, both existing `device_sync`s (expert compute and
///   whatever fetch was not hidden behind it), and the drain launch. The analog of GLM's
///   `moe` block_on wall.
/// * fetch, miss count, ms/miss and GB/token come from [`crate::memory::RoutedPool`]'s own
///   off-thread counters (`fetch_ns`, `misses`), exactly the values GLM's summary reads —
///   the reaper's wall overlaps the decode's, so fetch is NOT a share of wall there or here.
///
/// The remainder printed with them is `wall − route − moe`: the decode thread outside the
/// two brackets — attention/norm/hc LAUNCH time (their GPU time drains into `route`), the
/// end-of-layer sync (acc drain + `hc_post`), and the whole head tail (`hc_head`, final
/// norm, the bf16 lm_head GEMV, argmax sync + D2H). Non-negative by construction: both
/// buckets are sub-spans of the decode wall on one thread.
///
/// * `tail_ns` — M2's ranking item (3), the remainder's own decomposition: the head-tail
///   launches ([`V4Engine::head_tail`]) plus [`V4Engine::argmax`] — whose `device_sync` is
///   where the whole tail's GPU time drains, the existing join the M1 rule requires. Printed
///   INSIDE the remainder term, not beside it: the remainder stays `wall − route − moe` and
///   `tail` names the head share of it, leaving `remainder − tail` = the per-layer
///   launch/sync residue (candidate 4's other half). A sub-span of the remainder by
///   construction — disjoint from both brackets on the same thread.
///
/// # The route split (M4): device sub-spans of the work the gate D2H drains
///
/// Four HIP-event-pair accumulators say WHICH phase `route`'s wait drains (M2 measured it
/// ~76 ms against ~31 ms of bytes and could not say). Same instrument as GLM's
/// `idx_gpu_ns`, and like it a SPAN, gaps included, not a kernel sum: eight marks per
/// layer (six of M4's, renumbered when M6's two landed between SEL and ATTN_DONE) on the
/// null stream, read at the gate-logits D2H — the join the route bracket already
/// closes on — so no new sync, event wait, or join. Marks record on EVERY layer; a phase
/// a layer class skips (the two ratio-0 layers run no compressor) reads ~zero-width, so
/// the accumulation is uniform across classes rather than conditional. The coverage map
/// (marks are program order; the full residual argument and the per-span byte budgets
/// live in `docs/investigations/v4-decode-decomposition.md` §M4 and §M6):
///
/// * `hcn_ns` — marks 0→1 + 5→6: `hc_pre` + `attn_norm`, and `hc_post` + `hc_pre` +
///   `ffn_norm` — the hyper-connection application and both sublayer norm chains.
/// * `cmp_ns` — marks 1→2: `compress_and_place` (the deposit and both placement copies)
///   plus the GPU idle while the host builds and uploads the positional selection.
/// * `attn_ns` — marks 2→5: the `attn::v4::attention` call, whole — q/kv projections,
///   cache write, `sparse_attn`, o_proj. M6 split it (marks 3 and 4, recorded inside
///   `attention` — the next section); this whole-call span is kept as the tiled sum's
///   independent restatement.
/// * `gate_ns` — marks 6→7: the gate GEMV, the `xq` copy and its act_quant.
/// * `win_wall_ns` — the HOST wall containing them: layer top to the D2H's return. The
///   four spans tile marks 0→7 INSIDE that wall (mark 0 records after the wall clock
///   starts; mark 7 retires before the D2H's data lands), so the printed
///   `resid = win − Σspans` is non-negative by construction: it holds the D2H copy, the
///   pre-mark-0 lag (layer 0's includes the step's pending embed gather), and GPU-vs-host
///   clock skew. `resid` is defined against `win` and NOT against `route`, because
///   `route` does not contain the spans — GPU time overlapping the host's pre-D2H
///   traversal is invisible to `route`'s clock but visible to the events; §M4 carries the
///   reconciliation (`win − d2h` = the traversal the PROFILE remainder holds).
/// * `route_host_ns` — `route_row`'s host share of `route_ns`, from clock reads the
///   bracket already pays, so the print can restate `route` as its `d2h + host` halves.
///
/// # The attn split (M6): three sub-spans of `attn_ns`, tiled by construction
///
/// Two more marks per layer — recorded INSIDE `attn::v4::attention` via its
/// [`v4::SplitMarks`] argument (that struct carries the exact launch coverage of each
/// bracket) — cut the 2→5 attn span into `qkv` (2→3), `attend` (3→4) and `oproj` (4→5).
/// Same stream, same read site, no new join. Every endpoint is shared with the next
/// span, so `qkv + attend + oproj ≡ attn` up to per-query float rounding — the printed
/// ATTN-SPLIT residual is that identity checked on every run, and it is allowed EITHER
/// sign (four independent `hipEventElapsedTime` reads of the same four marks), unlike
/// `win`'s structurally-positive one.
///
/// * `qkv_ns` — marks 2→3: the q and kv projection chains, ~39.9 MB/layer of weights.
/// * `attend_ns` — marks 3→4: the KV cache write + `sparse_attn`, ≤~1 MB gathered.
/// * `oproj_ns` — marks 4→5: de-rotation + `wo_a`/`wo_b` projection, ~67.1 MB/layer.
///
/// # The moe split (M6): host tiling of `moe_ns`, plus three device attributions
///
/// `moe_ns` is a HOST wall (two `Instant` brackets in [`V4Engine::moe`]), so its split
/// is host sub-intervals that tile it — disjoint by construction, `resid ≥ 0` — read at
/// clocks the path already holds, plus three same-stream event pairs read behind the
/// SECOND `device_sync` the routed path already pays. What each holds, exactly:
///
/// * `moe_sh_ns` — the [`V4Engine::shared_expert`] call: five launch ENQUEUES (the
///   chain's GPU time drains later, see `moe_h2d_ns`).
/// * `moe_desc_ns` — routed entry to the first H2D: `RoutedPool::submit` (miss fetches
///   enter flight here), the format check, and the 256-descriptor launch-order rebuild.
/// * `moe_h2d_ns` — the two blocking H2D copies (~13.3 KB), plus the pointer/scalar
///   derivations between them and the sync (ns-scale, spans keep their gaps). Legacy
///   null-stream semantics order the copies behind the shared-expert chain still in
///   flight, so the chain's UNHIDDEN tail exposes here — the copies' own bytes are
///   ~30 µs/token.
/// * `moe_sync1_ns` — the first `device_sync`. Expected ~0 (the H2D just drained the
///   null stream); it is the correctness guarantee, not the exposure site.
/// * `moe_launch_ns` — ticket waits + the resident range launch + per-miss launches:
///   host enqueue only.
/// * `moe_sync2_ns` — the closing `device_sync`: the exposure of resident-batch
///   compute and miss stragglers, whichever stream drains last.
/// * `moe_drain_ns` — the accumulator drain launch (enqueue; its GPU time retires at
///   the end-of-layer sync, in the PROFILE remainder — said here so the bracket is not
///   read as containing it).
/// * resid (printed, not stored) — `moe − Σ` above: the inter-bracket clock gaps and
///   the instrument's own event reads, which sit deliberately between `sync2` and
///   `drain` so their cost lands in resid, not in a named span.
///
/// The three device pairs attribute what the host spans expose — they are NOT part of
/// the sum (their work overlaps `route_row`'s host math and each other across streams):
///
/// * `moe_shg_ns` — null-stream pair around the shared-expert chain: gate/up GEMVs,
///   swiglu, `act_quant`, down GEMV — the fp8 shared GEMV's true device span.
/// * `moe_res_ns` — compute-stream pair around the single resident range launch:
///   resident-batch expert compute, ~78.7 MB/layer at the measured residency.
/// * `moe_miss_ns` — miss-stream pair from before the first straggler's ticket wait to
///   after the last straggler's launch: fetch exposure + straggler compute. Both
///   miss-pair records are SKIPPED on a layer with no miss (and the res pair when
///   `n_res == 0`), because a reused event retains its previous recording — reading a
///   pair this call did not record would book a stale span, not zero.
///
/// What stays unbucketed and why: the per-miss split of `moe_miss_ns` into fetch-wait
/// vs kernel time would need an event pair PER MISS (a variable-length pool) or a
/// stream query between wait and launch — the off-thread `fetch`/`ms/miss` counters
/// already price that side, so it is not bought here.
///
/// Cost bound, extending the O(10 µs) argument above: M4's 258 records + 215 queries,
/// plus M6's ≤8 records (2 attn + 2 shared + 2 resident + 2 miss, the last two usually
/// skipped) and ≤6 queries per layer — worst case ~600 records + ~470 queries per
/// token, O(µs) each, O(2 ms) against a 130 ms token — plus 8 `Instant` reads per
/// routed call (O(20 ns) each, noise). A bound by argument, not a control; the M6 wall
/// gate (±3% of the recorded 130.7) is the control, and a breach is reported before
/// any reading of either split.
#[derive(Default)]
struct V4Profile {
    route_ns: u128,
    /// `route_row`'s host share of `route_ns` — accumulated beside it, never beyond it.
    route_host_ns: u128,
    moe_ns: u128,
    tail_ns: u128,
    win_wall_ns: u128,
    hcn_ns: u128,
    cmp_ns: u128,
    attn_ns: u128,
    gate_ns: u128,
    qkv_ns: u128,
    attend_ns: u128,
    oproj_ns: u128,
    moe_sh_ns: u128,
    moe_desc_ns: u128,
    moe_h2d_ns: u128,
    moe_sync1_ns: u128,
    moe_launch_ns: u128,
    moe_sync2_ns: u128,
    moe_drain_ns: u128,
    moe_shg_ns: u128,
    moe_res_ns: u128,
    moe_miss_ns: u128,
}

/// The route-split marks' indices into [`V4Engine::route_ev`], program order — named so
/// the record sites and the pair reads spell the same map [`V4Profile`] documents, and a
/// transposed pair is visible at the call site instead of plausible. M6 renumbered the
/// tail three when the two intra-attention marks landed between SEL and ATTN_DONE; the
/// names are why no pair read had to change for it.
const M_TOP: usize = 0;
const M_ATTN_NORMED: usize = 1;
const M_SEL: usize = 2;
/// Recorded inside `attn::v4::attention` via [`v4::SplitMarks`], as is the next one.
const M_QKV_DONE: usize = 3;
const M_ATTEND_DONE: usize = 4;
const M_ATTN_DONE: usize = 5;
const M_FFN_NORMED: usize = 6;
const M_GATE: usize = 7;

/// The moe-split pairs' indices into [`V4Engine::moe_ev`] — [`V4Profile`]'s "moe split"
/// section is the coverage map. Pairs, not a program-order chain: SH is a null-stream
/// pair, RES a compute-stream pair, MISS a miss-stream pair, so only same-index-pair
/// spans are meaningful.
const MO_SH0: usize = 0;
const MO_SH1: usize = 1;
const MO_RES0: usize = 2;
const MO_RES1: usize = 3;
const MO_MISS0: usize = 4;
const MO_MISS1: usize = 5;

/// Completed-event span in ns. Both events must have retired behind a join the path
/// already pays — every caller sits after the gate D2H or the routed path's closing
/// `device_sync`, so this is a query, never a wait.
fn ev_span_ns(a: &Event, b: &Event, what: &str) -> Result<u128> {
    let ms = Event::elapsed_ms(a, b)?;
    // A negative pair would saturate to 0 below and read as a plausible zero-width
    // span; same-stream program order makes it impossible, so it is a bug.
    debug_assert!(ms >= 0.0, "{what} event pair out of order: {ms} ms");
    Ok((f64::from(ms) * 1e6) as u128)
}

/// Everything one V4 decode needs that does not vary between tokens.
///
/// Every device buffer is allocated once in [`V4Engine::new`] and none per token, which is
/// `gpu.rs`'s rule and why a decode does not allocate on the hot path.
pub struct V4Engine {
    pin: V4Pin,
    cfg: V4Config,
    dims: v4::Dims,
    rope: RopeTables,
    layers: Vec<DeviceLayer>,
    /// Which layers this pin holds. `0..n_layers` for a whole-model decode; a shorter prefix is
    /// a golden comparison and [`V4Engine::new`] says so at startup.
    range: std::ops::Range<usize>,
    max_ctx: usize,
    /// Query rows every `[m, ..]` buffer is sized for — the prompt length, because V4's prefill
    /// attends the whole prompt in ONE `attention` call (both the ring seeding and the
    /// compressor's block pooling are whole-prompt by construction).
    max_m: usize,
    /// The token ids of the step in flight. Held because a hash layer's `tid2eid` is indexed by
    /// token id, and [`V4Engine::moe`] does not otherwise see them.
    step_ids: Vec<u32>,
    /// Rows the last [`V4Engine::set_residual`] (or [`V4Engine::forward`]) loaded.
    ///
    /// Closes the one way the probe API can produce a plausible wrong number instead of an error:
    /// loading a 13-row residual and then driving one row reads row 0 and scores it against a
    /// golden for row 12, silently. A review named it; nothing else on that path knows the two
    /// counts have to agree.
    loaded_rows: usize,

    /// `h` and its double buffer. **Two and not one:** `launch_hc_post`'s contract is that `y`
    /// must not alias `residual` — both are `__restrict__`, and thread `i` writes `y[i]` while
    /// other threads still read every source copy of `residual`, with no barrier between them.
    /// So `hc_post` reads `h[cur]` and writes `h[1 - cur]`. An in-place residual expansion is the
    /// obvious thing to want and it is wrong twice over.
    h: [DeviceBuf; 2],
    cur: usize,

    /// `hc_pre`'s `y` — the `[m, dim]` tensor the norms and the sublayer see. NOT the residual,
    /// which is why `attn_norm`/`ffn_norm` may be in-place `launch_v4_rmsnorm` here.
    xw: DeviceBuf,
    /// The fp8-quantized copy of `xw`. Separate because the ROUTER must see the unquantized
    /// activation — `Gate.forward` is `linear(x.float(), weight.float())`, with no activation
    /// quantization anywhere — while every expert projection must see the quantized one.
    /// Doubles as `attn::v4::Scratch::xq`, which re-derives it from `io.x` before any read.
    xq: DeviceBuf,
    post: DeviceBuf,
    comb: DeviceBuf,
    /// `launch_v4_hc_head`'s `[s, hc]` scratch gate vector.
    head_pre: DeviceBuf,
    /// The sublayer output `hc_post` consumes — attention's `io.out`, then the MoE's.
    sub: DeviceBuf,

    a_qr: DeviceBuf,
    a_qrq: DeviceBuf,
    a_q: DeviceBuf,
    a_kv: DeviceBuf,
    a_o: DeviceBuf,
    a_y: DeviceBuf,
    idx_host: Vec<i32>,
    idx_dev: DeviceBuf,

    gate_logits: DeviceBuf,
    gl_host: Vec<u8>,
    scores: Vec<f32>,
    choice: Vec<f32>,
    /// `n_experts` zeros, for the hash layers' `route_into` call. **Not an empty slice:**
    /// `route_into` computes `choice` by zipping `scores` with `bias`, so an empty `bias` leaves
    /// `choice` holding the PREVIOUS layer's values and `topk_into` then selects on them. A hash
    /// layer discards that selection, so it would be harmless today and a landmine tomorrow;
    /// zeros make `choice == scores`, which is what a bias-free gate means.
    zero_bias: Vec<f32>,
    sel: Vec<usize>,
    /// `[n_desc]` f32 indexed by ABSOLUTE expert id, zero for every expert this token did not
    /// route to. The kernel skips a zero weight, so the zeros are correctness and not thrift.
    /// The scatter source for [`V4Engine::routed_experts`]'s launch-order gather, and read
    /// directly by the probe API's last-row weights — which is why it stays absolute.
    wexpert_host: Vec<f32>,
    /// `[n_desc]` f32 in LAUNCH order — the descriptor index space the device buffers share
    /// (residents at `[0, n_res)`, misses after; see [`V4Engine::routed_experts`]). Gathered
    /// from [`V4Engine::wexpert_host`] per token row. Entries past this row's selection are
    /// never written after construction, so they stay zero — and a zero weight makes a
    /// wrongly-computed launch range write `h = 0` instead of plausible values, the same
    /// defense the null descriptors give the pointer side.
    wexpert_launch: Vec<f32>,
    wexpert: DeviceBuf,
    descs_host: Vec<ExpertDescF4>,
    descs: DeviceBuf,
    slots: Vec<ExpertSlot>,
    fmts: Vec<crate::artifact::format::RoutedFmt>,
    tickets: Vec<crate::fetch::asyncfetch::Ticket>,
    moe_acc: DeviceBuf,
    /// `[n_desc, nrow, inter]` f32 — indexed by the same launch-order descriptor index as
    /// `descs`/`wexpert` per the launcher's contract, so it is sized for `n_desc` and not
    /// for one range's `e_count`.
    moe_h: DeviceBuf,
    sh_g: DeviceBuf,
    sh_u: DeviceBuf,

    head_x: DeviceBuf,
    logits: DeviceBuf,
    argmax_dev: DeviceBuf,
    argmax_host: Vec<u8>,

    compute_stream: Stream,
    miss_stream: Stream,
    hits0: u64,
    misses0: u64,
    prof: V4Profile,
    /// The eight route-split marks (six of M4's, two of M6's inside `attention`),
    /// program order (see [`V4Profile`]). ONE array reused every layer: each layer's
    /// pairs are read at its own gate D2H, before the next layer records over them, so
    /// reuse cannot cross-read.
    route_ev: [Event; 8],
    /// The three moe-split pairs (see [`V4Profile`]'s moe section and the `MO_*`
    /// constants). Reused per layer like `route_ev` — read behind the routed path's own
    /// closing `device_sync` before the next layer records over them — with one further
    /// rule the route marks do not need: a pair a call did NOT record this layer is not
    /// read either, because a reused event retains its previous recording.
    moe_ev: [Event; 6],
    /// `Some` from layer top until the gate D2H returns — the token saying "this `moe`
    /// call closes a window [`V4Engine::layer`] opened". Today `moe` is only entered from
    /// `layer`, which always opens the window first; the `Option` is what keeps a FUTURE
    /// probe-driven `moe` from booking a window that never opened or reading marks it
    /// never recorded (`probe_attn_stages` already records marks 2..=5 and never reads).
    win_t0: Option<std::time::Instant>,
}

impl V4Engine {
    /// Build the engine over a loaded [`V4Pin`]. `prompt_len + ngen + 1` is the context.
    pub fn new(pin: V4Pin, cfg: V4Config, prompt_len: usize, ngen: usize) -> Result<Self> {
        let range = pin.range();
        // **A decode has to start at layer 0.** `V4Pin::build` deliberately does not enforce
        // this — which layers a file holds is a property of the LOADER, and refusing a partial
        // artifact there made every one but the first unloadable — but a forward pass has no
        // residual stream to enter at layer 3. `V4Pin::layer` takes ABSOLUTE ids, so a pin over
        // 3..6 answers every lookup correctly and the arithmetic is a different model's, with
        // nothing anywhere to notice. The check was in neither place before this line.
        ensure!(
            range.start == 0,
            "this artifact holds layers [{}, {}) and a decode must start at layer 0 — there is no \
             residual stream to enter the model at layer {}. Convert from layer 0.",
            range.start,
            range.end,
            range.start
        );
        ensure!(prompt_len > 0, "a decode needs a prompt");
        if range.end < cfg.n_layers {
            // Not refused: a 3-layer prefix IS what the `l0-2` fixture is for and what gates this
            // loop against the oracle's real-weight goldens. But three layers of a 43-layer model
            // is a golden comparison and not a decode, and calling it one is the reading this
            // port has had to retract twice — so it says so, loudly, once.
            tracing::warn!(
                "PARTIAL ARTIFACT: layers [0, {}) of {}. This is NOT the model — the logits are a \
                 {}-layer prefix's, and any text decoded from them is meaningless. Use it for the \
                 per-layer golden comparison only.",
                range.end,
                cfg.n_layers,
                range.end
            );
        }

        let max_ctx = prompt_len + ngen + 1;
        let limit = positional_context_limit(cfg.index_topk);
        // Refused at startup rather than 41 layers into some later token. `Sel::n_comp` refuses
        // on `start_pos + seqlen`, so a 13-token prompt crosses this after 2039 generated
        // tokens — not at the prompt, which is why a prompt-length check would not be this.
        // `<`, not `<=`. `Sel::n_comp` refuses when `(start_pos + rows) / ratio > index_topk`,
        // i.e. at `end_pos >= limit` — and `forward` admits `start_pos + m == max_ctx`. So
        // `max_ctx == limit` passes here and still refuses at the last position. `generate`'s own
        // accounting never reaches it (its last `forward` is at `max_ctx - 2`), which is exactly
        // the kind of slack that makes a boundary bug invisible until someone calls `forward`
        // directly. Found by review.
        ensure!(
            max_ctx < limit,
            "context {max_ctx} (prompt {prompt_len} + {ngen} generated + 1) reaches {limit}, past \
             which the compressed block set is decided by the lightning indexer's SCORES. This \
             loop selects blocks POSITIONALLY, which agrees with the indexer on the block SET \
             only below that length; above it, keeping the first {} blocks keeps the OLDEST and \
             silently stops attending everything newer, on the {} ratio-{INDEXED_RATIO} layers.",
            cfg.index_topk,
            cfg.compress_ratios
                .iter()
                .take(cfg.n_layers)
                .filter(|&&r| r == INDEXED_RATIO)
                .count(),
        );

        let dims = v4::Dims::from_config(&cfg).context("v4 attention dims from the artifact")?;
        let (dim, hc, hd) = (cfg.hidden, cfg.hc_mult, cfg.head_dim);
        let nhd = cfg.n_heads * hd;
        let (max_m, n_desc) = (prompt_len, cfg.n_experts);
        let f32s = |n: usize| DeviceBuf::new(n * size_of::<f32>());

        // The widest selection any step can ask for. Prefill is `m` rows of
        // `min(m, win) + m/ratio`; decode is one row of `win + max_ctx/ratio`. Taken as the
        // bound of both rather than assumed to be one of them.
        let idx_cols = cfg.sliding_window + max_ctx.div_ceil(INDEXED_RATIO);
        let mut layers = Vec::with_capacity(range.len());
        for l in range.clone() {
            let kind = LayerKind::from_config(&cfg, l)
                .with_context(|| format!("classifying layer {l}"))?;
            // The pin's compressor for this layer, so `LayerCompressor` can narrow its weights. Read
            // through `V4Pin::layer`, which applies the artifact-order offset exactly once.
            let cw = pin.layer(l)?.compressor;
            layers.push(DeviceLayer::new(&cfg, kind, max_ctx, max_m, cw.as_ref())?);
        }

        let (hits0, misses0) = (pin.routed.hits(), pin.routed.misses());
        // A `Vec` because `[Event::new()?; 14]` cannot exist (`Event` is not `Copy`) and
        // fourteen spelled-out calls are the duplication gate's food. One loop for both
        // arrays; the moe pairs are the tail six.
        let mut marks = Vec::with_capacity(14);
        for _ in 0..14 {
            marks.push(Event::new()?);
        }
        let moe_marks = marks.split_off(8);
        let mut e = Self {
            dims,
            rope: RopeTables::new(&cfg, max_ctx),
            layers,
            range,
            max_ctx,
            max_m,
            step_ids: Vec::with_capacity(max_m),
            loaded_rows: 0,
            h: [f32s(max_m * hc * dim)?, f32s(max_m * hc * dim)?],
            cur: 0,
            xw: f32s(max_m * dim)?,
            xq: f32s(max_m * dim)?,
            post: f32s(max_m * hc)?,
            comb: f32s(max_m * hc * hc)?,
            head_pre: f32s(hc)?,
            sub: f32s(max_m * dim)?,
            a_qr: f32s(max_m * cfg.q_lora_rank)?,
            a_qrq: f32s(max_m * cfg.q_lora_rank)?,
            a_q: f32s(max_m * nhd)?,
            // `[m + m/ratio, head_dim]`, not `[m, head_dim]`: at prefill `sparse_attn` reads
            // `torch.cat([kv, kv_compress])` and the selection indexes that concatenation as ONE
            // space, so the compressor's blocks live in this buffer's tail. Sized at the tightest
            // ratio because one buffer serves every layer class.
            a_kv: f32s((max_m + max_m.div_ceil(INDEXED_RATIO)) * hd)?,
            a_o: f32s(max_m * nhd)?,
            a_y: f32s(max_m * cfg.o_groups * cfg.o_lora_rank)?,
            idx_host: Vec::with_capacity(max_m * idx_cols),
            idx_dev: DeviceBuf::new(max_m * idx_cols * size_of::<i32>())?,
            gate_logits: f32s(max_m * n_desc)?,
            gl_host: Vec::new(),
            scores: vec![0.0; n_desc],
            choice: vec![0.0; n_desc],
            zero_bias: vec![0.0; n_desc],
            sel: Vec::with_capacity(cfg.top_k),
            wexpert_host: vec![0.0; n_desc],
            wexpert_launch: vec![0.0; n_desc],
            wexpert: f32s(n_desc)?,
            descs_host: vec![desc_never_read(); n_desc],
            descs: DeviceBuf::new(n_desc * size_of::<ExpertDescF4>())?,
            slots: Vec::with_capacity(cfg.top_k),
            fmts: Vec::with_capacity(cfg.top_k),
            tickets: Vec::with_capacity(cfg.top_k),
            // `MOE_ACC_ROWS` rows of ONE token's hidden width: the routed experts are launched
            // per token, because the fp4 kernel refuses `nrow != 1`.
            moe_acc: DeviceBuf::new(MOE_ACC_ROWS * dim * size_of::<u64>())?,
            moe_h: f32s(n_desc * cfg.moe_inter)?,
            sh_g: f32s(max_m * cfg.moe_inter)?,
            sh_u: f32s(max_m * cfg.moe_inter)?,
            head_x: f32s(dim)?,
            logits: f32s(cfg.vocab)?,
            argmax_dev: DeviceBuf::new(size_of::<i32>() + size_of::<f32>())?,
            argmax_host: Vec::new(),
            compute_stream: Stream::compute()?,
            miss_stream: Stream::miss()?,
            hits0,
            misses0,
            prof: V4Profile::default(),
            // Infallible at runtime — the loop above pushed exactly fourteen and the
            // split took six — but `try_into`'s error type is the `Vec` back, so
            // `context` is the honest spelling.
            route_ev: marks.try_into().ok().context("eight route-split events")?,
            moe_ev: moe_marks.try_into().ok().context("six moe-split events")?,
            win_t0: None,
            pin,
            cfg,
        };
        e.reset()?;
        Ok(e)
    }

    /// Clear every layer's persistent state and the accumulator. **Between sequences, not
    /// between tokens.**
    fn reset(&mut self) -> Result<()> {
        for st in &mut self.layers {
            st.reset(&self.cfg)?;
        }
        // `launch_moe_acc_drain` resets `acc` to zero as it converts, so this is only the FIRST
        // use's initialisation — but `hipMalloc` does not zero, and an accumulator that starts
        // at garbage adds a fixed-point garbage vector to layer 0's first token and nothing else.
        // One `fill_u32` per sequence; GLM pays the same one, once, for the same reason.
        // SAFETY: allocated at exactly this size, nothing in flight.
        unsafe {
            fill_u32(
                self.moe_acc.ptr_mut(),
                0,
                MOE_ACC_ROWS * self.cfg.hidden * size_of::<u64>(),
            )?;
        }
        self.cur = 0;
        // Re-baselined, so a second `generate` on one engine reports ITS lookups and not the first
        // run's folded in. These were captured once in `new` and left there, which a review caught:
        // the hit-rate line is the number every later measurement is explained by, and a cumulative
        // one reads as a residency change.
        self.hits0 = self.pin.routed.hits();
        self.misses0 = self.pin.routed.misses();
        self.loaded_rows = 0;
        Ok(())
    }

    /// Run this layer's compressor for the step and place its blocks at every destination the
    /// reference writes.
    ///
    /// **ONE `compress` call; one or two placements.** `Compressor.forward` performs two writes
    /// and only one of them is the return value: it assigns `self.kv_cache[:, :seqlen // ratio]`
    /// — the persistent region every later decode step selects by position — *and* returns the
    /// same blocks for `Attention.forward` to `torch.cat` onto this step's prompt KV. `Finish`
    /// carries a single `out`, so the second destination is a device COPY and never a second
    /// `compress`. Why a second call is the hazard is stated once, at the call below.
    ///
    /// **Both destinations come from `compress_dst`; neither is re-derived.** The two
    /// `region_base`s are:
    ///
    /// * the SELECTION space — `compress_offset(win, seqlen, start_pos)`, which is `seqlen` at
    ///   prefill (the transient `torch.cat`, i.e. `a_kv`'s tail) and `window_size` at decode;
    /// * the PERSISTENT `[ring ‖ compressed]` buffer — `window_size`, always.
    ///
    /// At decode the two coincide, and that is not a coincidence needing a branch: decode's
    /// selection space IS the persistent buffer, so there is exactly one destination and the
    /// equality is what says so. The `if` below tests the two bases, not the phase.
    ///
    /// **A disagreement, reported rather than resolved.** The port's carried note 3 proposes
    /// making `region_base` unrepresentably wrong by having `compress_dst` take `&Geom` and
    /// derive the base from its `Quantize` — `sliding_window` for the attention compressor, `0`
    /// for the indexer's nested one. That fix is **incompatible with the prefill call below**:
    /// its selection-space base is `seqlen`, which is neither of those two values, so a
    /// `compress_dst` that derived its own base could not express the destination
    /// `attn::v4::attention` actually reads at prefill. `compress_dst`'s doc scopes it to the
    /// PERSISTENT cache, so the first call here uses it a little outside its stated meaning —
    /// deliberately, because the arithmetic is identical and a second placement function would be
    /// a second place for the rule to be wrong.
    fn compress_and_place(&mut self, layer: usize, p: Extent, stream: *mut c_void) -> Result<()> {
        let (win, hd) = (self.cfg.sliding_window, self.cfg.head_dim);
        let li = layer - self.range.start;
        let kind = self.layers[li].kind;
        if self.layers[li].comp.is_none() {
            // A ratio-0 layer has no `Compressor` object at all in the reference
            // (`Attention.__init__` builds one only when `compress_ratio` is truthy), so there is
            // nothing to run and nothing to place. Every other early return below still runs the
            // call.
            return Ok(());
        }
        // **The one `compress` call site in this engine**, and unconditional on a compressing
        // layer. `Compressor.forward` writes `kv_state`/`score_state` in BOTH phases and only
        // THEN decides whether to emit, so a step that emits nothing still deposits — at ratio
        // 128 that is 127 of every 128 decode steps. Skipping the call on a non-emitting step
        // would skip the deposit with it, and the pooling window would be built from every
        // 128th token.
        //
        // Both destinations, computed BEFORE anything runs. `compress_dst` is pure, so the
        // bounds below are PRE-FLIGHT — which matters because `run_compress` deposits into
        // `kv_state`/`score_state` and (at decode) slides the pooling window, so a bound that
        // failed after it would leave the compressor advanced with no way to retry the step.
        // `RoutedPool::submit` had exactly this shape as a real defect; this is the same lesson
        // applied before it could become one.
        let persist = compress_dst(kind, win, p.seqlen, p.start_pos);
        let sel = compress_dst(
            kind,
            compress_offset(win, p.seqlen, p.start_pos),
            p.seqlen,
            p.start_pos,
        );
        if let Some((persist_base, blocks)) = persist {
            // Bounded against the BUFFERS, not against the arithmetic that produced the row. A
            // compressed region sized `max_ctx/ratio` and a slot from `start_pos/ratio` agree
            // only while `start_pos < max_ctx`, which `forward` enforces — these are the checks
            // that say so at the write, where a wrong row is an out-of-bounds device write.
            ensure!(
                persist_base + blocks <= self.layers[li].cache_rows,
                "layer {layer}: {blocks} block(s) at cache row {persist_base} overrun the {} rows \
                 the cache holds",
                self.layers[li].cache_rows
            );
            let (sel_base, _) =
                sel.context("compress_dst named a persistent destination and no selection one")?;
            // **Only when `a_kv` is actually written.** The first version of this check sat
            // outside the branch, so at decode it bounded a PERSISTENT-cache row (`window_size +
            // start_pos/ratio`, i.e. 131 at the goldens' prompt) against the attention scratch's
            // row count (`max_m + max_m/4`, i.e. 17) — and fired at the first decode position
            // that completed a block on any compressing layer. Two different coordinate systems
            // with the same type. Found by review; invisible to `tests/v4_loop.rs`, which scores
            // ratio-0 layers only.
            if sel_base != persist_base {
                let rows = self.max_m + self.max_m.div_ceil(INDEXED_RATIO);
                ensure!(
                    sel_base + blocks <= rows,
                    "layer {layer}: {blocks} block(s) at kv row {sel_base} overrun the {rows}-row \
                     attention scratch"
                );
            }
        }

        // **Twice is the specific failure, and this is the one place that says so.** Every path
        // in `compress` runs `launch_v4_compress_state` before the emit decision, which is a
        // read-modify-write of `kv_state`/`score_state` — and the decode path also slides the
        // pooling window. So a second call re-deposits the same rows and slides again, which
        // corrupts exactly the state S3 requirement 3 is about, finitely and plausibly.
        let (blocks, src) = self.run_compress(layer, p, stream)?;

        let Some((sel_base, _)) = sel else {
            // `None` exactly where the reference returns `None`: a prefill shorter than `ratio`,
            // or a decode position that does not complete a block.
            //
            // A DRIFT TRIPWIRE and not a runtime guard, which is the honest reading: `compress`
            // and `compress_dst` both decide by calling `should_compress` on the same
            // `(kind, seqlen, start_pos)`, so today they cannot disagree. What it catches is a
            // future edit to one of the two — they live in the same file but in different
            // functions, and `compress`'s emit arm is where a `>=` could become a `>`.
            ensure!(
                blocks == 0,
                "layer {layer}: compress emitted {blocks} block(s) where compress_dst names no \
                 destination — should_compress has drifted between the two"
            );
            return Ok(());
        };
        let (persist_base, want) =
            persist.context("compress_dst named a selection destination and no persistent one")?;
        // The same tripwire in the other direction, and this one compares COUNTS rather than
        // presence: `compress` emits `seqlen / ratio` at prefill and 1 at decode, and
        // `compress_dst` says the same thing from the same inputs. Both memcpys below are sized
        // from `blocks`, so a divergence would write a different number of rows than the
        // placement reserved.
        ensure!(
            blocks == want,
            "layer {layer}: compress emitted {blocks} block(s) at {p:?} where compress_dst \
             reserved {want}"
        );
        let row = hd * size_of::<f32>();
        let cache = self.layers[li].cache.ptr_mut().cast::<f32>();
        let tail = self.a_kv.ptr_mut().cast::<f32>();
        // SAFETY: `out` holds `blocks * head_dim` f32 by `LayerCompressor`'s sizing, and both destinations
        // were bounded above. `memcpy_dtod` requires non-overlap, which holds: `cache` and
        // `a_kv` are distinct allocations and `out` is a third.
        unsafe {
            // REBASE: `memcpy_dtod_async` here and below. This is a blocking `hipMemcpy` on the
            // null stream (`kernels/linalg.hip:692`), so it is a full serialisation point in the
            // middle of an otherwise-streamed sequence — converting the six V4 launchers without
            // also converting these two leaves the overlap defeated.
            memcpy_dtod(cache.add(persist_base * hd).cast(), src, blocks * row)
                .context("persisting the compressed region")?;
            if sel_base != persist_base {
                // Prefill only: the transient `torch.cat([kv, kv_compress])` this step's
                // `sparse_attn` indexes. At decode the two bases are equal and the copy above
                // already IS the selection space.
                memcpy_dtod(tail.add(sel_base * hd).cast(), src, blocks * row)
                    .context("the prefill kv concatenation tail")?;
            }
        }
        Ok(())
    }

    /// Assemble `Buffers`, make the call, and hand back `(blocks emitted, the buffer they are
    /// in)`.
    ///
    /// Split from [`V4Engine::compress_and_place`] for the BORROW and not for safety — an
    /// earlier doc here claimed the split was what kept `compress` to one call site, which is
    /// backwards: this function *is* the second entry point, and what keeps the call to one is
    /// that this one is private with a single caller. Returning `src` rather than letting the
    /// caller re-reach for `comp.blocks` is what removed a third `Option<LayerCompressor>` probe whose
    /// comment had to open "Unreachable:".
    fn run_compress(
        &mut self,
        layer: usize,
        p: Extent,
        _stream: *mut c_void,
    ) -> Result<(usize, *const u8)> {
        let li = layer - self.range.start;
        let lp = self.pin.layer(layer)?;
        let cw = *lp
            .compressor
            .as_ref()
            .with_context(|| format!("layer {layer} compresses but its pin has no compressor"))?;
        let freqs = self.rope.for_layer(self.layers[li].kind)?;
        let (dim, max_m) = (self.cfg.hidden, self.max_m);
        let x = self.xw.ptr().cast::<f32>();
        // As above: guaranteed `Some` by the caller's early return, reported rather than panicked.
        let c = self.layers[li].comp.as_mut().with_context(|| {
            format!("layer {layer} compresses but its DeviceLayer has no compressor state")
        })?;
        let b = Buffers {
            x,
            dim,
            // The NARROWED copies, not the pin's f32 pointers. Casting those to `*const u16`
            // was this loop's worst bug and it is worth naming at the call site: the kernel
            // strides `w + c * k` in u16 units, so it would read a different row's data
            // entirely. See [`narrow_to_bf16`].
            wkv: c.w_kv.ptr().cast(),
            wgate: c.w_gate.ptr().cast(),
            ape: cw.ape,
            fin: Finish {
                norm: cw.norm,
                // The layer's COMPRESSED rotary table, resolved through the ONE site.
                freqs,
                out: c.blocks.ptr_mut().cast(),
            },
            kv_state: c.state_kv.ptr_mut().cast(),
            score_state: c.state_score.ptr_mut().cast(),
            kv: c.proj_kv.ptr_mut().cast(),
            score: c.proj_score.ptr_mut().cast(),
            scratch_rows: max_m,
        };
        // SAFETY: every pointer above is either a pin placement outliving this engine or a
        // `DeviceBuf` field of `self`, at the shape `Buffers` documents. `compress` takes no
        // stream today (see this module's header); `_stream` is what it gets when it does.
        let blocks = unsafe { compress(&c.geom, &b, p.seqlen, p.start_pos, NULL_STREAM) }
            .with_context(|| format!("layer {layer} compressor at {p:?}"))?;
        Ok((blocks, c.blocks.ptr()))
    }

    /// One attention block: the compressor, its placements, then `attn::v4::attention`.
    ///
    /// The order is the reference's (`Attention.forward`, model.py:523-538 — 538 is the decode `sparse_attn`, which is what the compressor-then-attend order is about) and it is not
    /// optional. `attention`'s own safety contract says the compressor must already have run for
    /// this same step and must have written BOTH destinations, because `attention` only READS the
    /// compressed rows and writes neither. Running it after would hand `sparse_attn`
    /// uninitialised device memory in rows every later decode step selects BY POSITION and
    /// weights with `exp(l - max)`. Doing both here rather than at two call sites is what makes
    /// it impossible to get right in prefill and wrong in decode.
    fn attention_block(&mut self, layer: usize, p: Extent, stream: *mut c_void) -> Result<()> {
        let (m, start_pos) = (p.seqlen, p.start_pos);
        let li = layer - self.range.start;
        let kind = self.layers[li].kind;
        self.compress_and_place(layer, p, stream)?;

        // `win`, `seqlen` and `start_pos` are overwritten by `attention` from its own `Dims` and
        // `Pass`; the caller supplies only the layer's class and `index_topk`. That is not
        // redundancy — a `Sel` whose `win` disagreed with `Dims::window` produced a selection
        // over the wrong slot space that matched its own `idxs_shape` and passed every guard, so
        // the values here must be the ones `attention` would compute anyway.
        let sel = Sel {
            win: self.cfg.sliding_window,
            kind,
            index_topk: self.cfg.index_topk,
            seqlen: m,
            start_pos,
        };
        self.idx_host.clear();
        let shape = v4_topk_idxs(sel, &mut self.idx_host)
            .with_context(|| format!("layer {layer} selection at ({m}, {start_pos})"))?;
        self.idx_dev.copy_in_at(0, as_le_bytes(&self.idx_host))?;

        let lp = self.pin.layer(layer)?;
        let fp8 = |w: Fp8Weight| v4::Fp8W {
            w: w.packed,
            scale: w.scale,
        };
        let w = v4::Weights {
            wq_a: fp8(lp.wq_a),
            q_norm: lp.q_norm,
            wq_b: fp8(lp.wq_b),
            wkv: fp8(lp.wkv),
            kv_norm: lp.kv_norm,
            attn_sink: lp.attn_sink,
            wo_a: fp8(lp.wo_a),
            wo_b: fp8(lp.wo_b),
        };
        let s = v4::Scratch {
            rows: self.max_m,
            xq: self.xq.ptr_mut().cast(),
            qr: self.a_qr.ptr_mut().cast(),
            qrq: self.a_qrq.ptr_mut().cast(),
            q: self.a_q.ptr_mut().cast(),
            kv: self.a_kv.ptr_mut().cast(),
            o: self.a_o.ptr_mut().cast(),
            y: self.a_y.ptr_mut().cast(),
        };
        let io = self.io_for(kind, li, shape)?;
        let step = if start_pos == 0 {
            v4::Pass::Prefill { seqlen: m }
        } else {
            v4::Pass::Decode { pos: start_pos }
        };
        // Mark 2: selection uploaded, nothing of `attention` queued yet — `attn`'s span
        // opens here, so the compressor span behind it keeps the selection-build idle.
        self.route_ev[M_SEL].record(NULL_STREAM)?;
        // Marks 3 and 4 record INSIDE the call, between its three sections — the M6
        // attn split ([`v4::SplitMarks`] carries the per-bracket launch coverage).
        let mk = v4::SplitMarks {
            qkv_done: &self.route_ev[M_QKV_DONE],
            attend_done: &self.route_ev[M_ATTEND_DONE],
        };
        // SAFETY: every pointer in `w` is a pin placement outliving this engine; every pointer in
        // `s`/`io` is a `DeviceBuf` field of `self` at the size its field documents, and no two
        // are the same allocation — `xq` is attention's own scratch here, re-derived from `io.x`
        // inside `attention` before any read.
        unsafe { v4::attention(&self.dims, sel, &w, &s, &io, step, Some(mk), NULL_STREAM) }
            .with_context(|| format!("layer {layer} attention at ({m}, {start_pos})"))?;
        // Mark 5: `attn` closes on the call's last enqueue.
        self.route_ev[M_ATTN_DONE].record(NULL_STREAM)
    }

    /// Bind one step's `Io`.
    ///
    /// **This takes `LayerKind` and resolves `freqs` through [`RopeTables::for_layer`], and that
    /// is the whole reason it exists** — it is the enforcing construction
    /// `docs/investigations/v4-flash-port.md` has handed back three times, whose stated blocker
    /// each time was that its only correct home is a layer loop. `Io::freqs` is a bare
    /// `*const f32` and cannot tell the ratio-0 table from the YaRN one; nothing in the engine
    /// detects the mismatch, and the port has measured the numeric gate blind to it at
    /// `ratio4/decode`. Every `Io` in this file comes from here, so there is one site rather than
    /// one per call.
    fn io_for(&mut self, kind: LayerKind, li: usize, idxs_shape: (usize, usize)) -> Result<v4::Io> {
        let freqs = self.rope.for_layer(kind)?;
        Ok(v4::Io {
            x: self.xw.ptr().cast(),
            freqs,
            idxs: self.idx_dev.ptr().cast(),
            idxs_shape,
            cache: self.layers[li].cache.ptr_mut().cast(),
            out: self.sub.ptr_mut().cast(),
        })
    }

    /// `Gate.forward` for one row, on the host, into `self.sel` and `self.wexpert_host`.
    ///
    /// # `launch_moe_gate_v4` is DECLINED, and this is the decision the port asked for
    ///
    /// That kernel exists, is verified, is 8-test covered, and takes `tid2eid` as a device
    /// `*const i64` — while `V4Pin` parses the table to a host `Vec<u32>` and argues that
    /// "placing 6.2 MB of `tid2eid` per hash layer on the device to index it there would buy
    /// nothing". Both are defensible and they are opposite; the port recorded it so this stage
    /// would decide deliberately rather than discover it at the call site. The reasons:
    ///
    /// * routing is HOST work in this engine and `math::route_into` is the router that
    ///   `docs/reference/architecture.md` INV-1 is stated about. A second router on the device is
    ///   a second place for "the selection bias must not reach the weights" to be wrong, and
    ///   that rule is invisible to every magnitude check.
    /// * the indices must reach the host regardless, because `RoutedPool::submit` is host code.
    ///   So the kernel does not remove a D2H, it moves one: 48 bytes of picks instead of 1 KB of
    ///   logits, against an 18.6 MB `tid2eid` upload and a second scatter to rebuild `wexpert` by
    ///   absolute id.
    /// * `parse_tid2eid`'s range check is only expressible host-side, and `moe.hip`'s own note
    ///   records that the kernel does not perform it.
    ///
    /// So `launch_moe_gate_v4` has no reachable caller after this stage either — the shape
    /// `Dims::compress_slot` was in when it was deleted. Recorded rather than acted on: removing
    /// a verified kernel is not this stage's call.
    fn route_row(&mut self, layer: usize, t: usize) -> Result<()> {
        let (k, n_desc) = (self.cfg.top_k, self.cfg.n_experts);
        let logits = &self.gl_host[t * n_desc * 4..(t + 1) * n_desc * 4];
        let lp = self.pin.layer(layer)?;
        let (bias, hash) = match &lp.route {
            V4Route::Scored { bias } => (bias.as_slice(), None),
            // A hash layer has no bias. It still RUNS the gate: the scores become the WEIGHTS
            // even though the selection ignores them, and reading `tid2eid` while skipping the
            // gate leaves the weights uniform — which decodes fluently and wrongly
            // (`Defect::HashRoutingIgnored`'s mirror image).
            V4Route::Hash { tid2eid } => (self.zero_bias.as_slice(), Some(tid2eid)),
        };
        route_into(
            logits,
            bias,
            k,
            Scoring::SqrtSoftplus,
            &mut self.scores,
            &mut self.choice,
            &mut self.sel,
        );
        if let Some(tid2eid) = hash {
            // `tid2eid[token * top_k + j]` REPLACES the selection and nothing else. Values are
            // valid by construction: `parse_tid2eid` range-checked them into a `Vec<u32>` at
            // load, which is the only place that can, since the kernel's own note says it does
            // not and the descriptor array it would index is `n_desc` long.
            let tok = self.step_ids[t] as usize;
            let base = tok * k;
            let picks = tid2eid
                .get(base..base + k)
                .with_context(|| format!("layer {layer}: tid2eid has no row for token {tok}"))?;
            self.sel.clear();
            self.sel.extend(picks.iter().map(|&e| e as usize));
        }
        // Renormalise, then scale: `weights /= weights.sum()` then `*= route_scale`
        // (model.py:586-588). `route_into` does NEITHER — it stops at the scores, because GLM's
        // `norm_topk_prob` is false — so both are this loop's. The weights come from `scores`,
        // never from `choice`: `Defect::RouterBiasedWeights` is the one-line "simplification"
        // that lets the selection bias reach them, and it changes every routed magnitude by an
        // amount that looks like ordinary variation.
        let sum: f32 = self.sel.iter().map(|&e| self.scores[e]).sum();
        ensure!(
            sum > 0.0 && sum.is_finite(),
            "layer {layer}: routing weights sum to {sum}"
        );
        let scale = self.cfg.routed_scale as f32 / sum;
        self.wexpert_host.fill(0.0);
        // Indexed by ABSOLUTE expert id — this is the scatter [`V4Engine::routed_experts`]
        // gathers into launch order (and the probe API reads directly), sized `n_desc` rather
        // than `e_count`. The id is in range on both paths and by two independent
        // mechanisms: the scored path selects indices OF this
        // `n_desc`-long array, and the hash path's values were range-checked into a `Vec<u32>` by
        // `parse_tid2eid` at load — which is the only place that can, since the kernel's own note
        // says it does not. `submit` checks it a third time. An `ensure!` here was written and cut
        // for being triple-dead; the `Vec` index is what makes it total.
        for &e in &self.sel {
            self.wexpert_host[e] = self.scores[e] * scale;
        }
        Ok(())
    }

    /// The routed experts for token row `t`, accumulated onto `sub[t]`.
    ///
    /// One row at a time, and that is structural rather than a simplification:
    /// `kernels/moe.hip:409` refuses `nrow != 1` (guard 1003, only `R = 1` instantiated). So a
    /// prefill of `s` tokens performs `s` MoE dispatches per layer, while attention runs once
    /// over the whole prompt — attention is the only op with a cross-token dependency.
    ///
    /// `sub[t]` must already hold the SHARED expert's output: [`launch_moe_acc_drain`] does
    /// `x += ...`. In GLM that `+=` IS the residual add; here it is not — V4's MoE output feeds
    /// `hc_post`'s `x` argument, and `MoE.forward` starts from `y = zeros` and adds the shared
    /// expert raw (model.py:648, no `weights` argument). That is why the shared expert *writes*
    /// this buffer and the routed drain adds on top.
    fn routed_experts(&mut self, layer: usize, t: usize) -> Result<()> {
        let (dim, inter, n_desc) = (self.cfg.hidden, self.cfg.moe_inter, self.cfg.n_experts);
        // The M6 moe-split host brackets — every boundary below is a clock read beside
        // a call the path already makes; [`V4Profile`]'s moe section is the coverage map.
        let t_entry = std::time::Instant::now();
        self.pin.routed.submit(
            layer,
            &self.sel,
            // Empty: `window`/`choice` feed the v2 access trace and nothing else, and this loop
            // writes no trace. Passing the real arrays would cost a clone per token per layer for
            // a file nobody reads.
            &[],
            &[],
            &mut self.slots,
            &mut self.fmts,
            &mut self.tickets,
        )?;
        // `submit` clears all three and pushes exactly one entry per `sel` element, so the three
        // `zip`/index pairings below are total. A length `ensure!` was written and cut: it
        // restated `submit`'s postcondition and could not fire.
        // **`fmt` is READ, not ignored.** `submit` returns it so the caller can dispatch, which is
        // exactly what GLM's `launch_expert_range` does — and `desc_of_f4` below builds an F4
        // descriptor unconditionally. The pool is single-format today (`V4Pin::build` hands
        // `RoutedPool::new` the same `TierFmt` twice), so this cannot fire; it is here because the
        // consequence if a second container is ever paired with `.f4` is not a fault. `.f4` and
        // `.i4` tile IDENTICALLY for 25% of all `i_dim`, both models' dimensions included
        // (`quant::f4_slot_offsets` carries the identity), so every projection would be found at
        // exactly the right address and then decoded as e2m1 nibbles against the wrong scale grid:
        // right bytes, wrong arithmetic, and no length, offset or descriptor check able to see it.
        // It also gives `fmts` its only reader — a `Vec` that `submit` fills and nobody inspects is
        // how the last two dead fields in this engine got there.
        for (&e, f) in self.sel.iter().zip(&self.fmts) {
            ensure!(
                *f == crate::artifact::format::RoutedFmt::F4,
                "layer {layer} expert {e}: the pool resolved a {f:?} slot and this path builds an \
                 ExpertDescF4 — the bytes would be found at the right addresses and decoded with \
                 the wrong arithmetic"
            );
        }
        // **Refilled with nulls first, and the first version of this did not.** It wrote only the
        // selected entries and its comment claimed "the rest stay `desc_never_read()`" — true for
        // the first token of the first layer and false ever after: the previous token's six
        // descriptors survive. That is strictly worse than the null it was
        // meant to preserve. A stale descriptor names a pool SLOT, and a slot the policy has since
        // evicted holds a different expert's bytes at exactly the right addresses — so a
        // wrongly-computed range would read plausible wrong weights on the one path where the
        // ticket protocol cannot help, instead of faulting. 256 pointer-sextuple writes per token
        // per layer against a 13.37 MB expert fetch is not a cost worth trading for that.
        //
        // **Written in LAUNCH order, not at absolute ids (M3b, 2026-08-07):** residents at
        // `[0, n_res)`, misses after, so the residents form the one CONTIGUOUS range the
        // launcher's `[e_start, e_count)` form was left as a hook for (`kernels/moe.hip:173`)
        // — one 3-launch range call where M2 counted 3 per expert, ~820 launches/token.
        // Byte-identity across the regrouping is the fixed-point accumulator's contract:
        // every expert's `h` row and `moe_fixed` term is computed from the same descriptor,
        // `x` and weight regardless of which launch carries it, and integer addition is
        // associative and commutative, so the `acc` sums cannot depend on the grouping
        // (`common.hpp`'s MOE_ACC_SHIFT block). A duplicate pick (reachable only through a
        // hash-table row) gets one compact slot per occurrence and accumulates once each —
        // the same total the absolute layout produced by launching its shared slot twice.
        self.descs_host.fill(desc_never_read());
        let mut n_res = 0;
        let mut c = 0;
        for resident in [true, false] {
            for i in 0..self.sel.len() {
                if self.tickets[i].is_resident() != resident {
                    continue;
                }
                self.descs_host[c] = desc_of_f4(&self.slots[i]);
                self.wexpert_launch[c] = self.wexpert_host[self.sel[i]];
                c += 1;
            }
            if resident {
                n_res = c;
            }
        }
        let t_h2d = std::time::Instant::now();
        self.descs.copy_in_at(0, as_le_bytes(&self.descs_host))?;
        self.wexpert
            .copy_in_at(0, as_le_bytes(&self.wexpert_launch))?;

        // SAFETY: `xq` holds `max_m * dim` f32 and `t < m <= max_m`.
        let x = unsafe { self.xq.ptr().cast::<f32>().add(t * dim) };
        // 16-byte alignment of `x` and `h` is the launcher's contract, is UNCHECKED there, and
        // faults rather than falling back. It holds by construction and the argument is written
        // here rather than as an `ensure!` that cannot fire: `DeviceBuf` is `hipMalloc`'d, which
        // is 256-byte aligned, and the only offset applied is `t * dim` floats — `dim` is 4096, so
        // every row base is 16 KiB-aligned. The check that WOULD earn its keep is one on `dim`
        // itself, and `Dims::validate` already refuses a `hidden` that is not a multiple of 128.
        let acc = self.moe_acc.ptr_mut().cast::<u64>();
        let h = self.moe_h.ptr_mut().cast::<f32>();
        let descs = self.descs.ptr().cast::<ExpertDescF4>();
        let wexpert = self.wexpert.ptr().cast::<f32>();
        // Narrowed here rather than held on `self`: the narrowing is deterministic, and
        // `V4Config::validate` already checks the f32 it produces is positive AND finite — the
        // second half because a finite f64 above ~3.4e38 saturates to `f32::INFINITY`, which passes
        // every `> 0.0` test and makes `fminf(gt, inf)` a no-op, i.e. `Defect::SwigluUnclamped`
        // silently. A copy of that check here was written and removed: `validate` runs on every
        // path into this engine, and the version here was one-sided anyway.
        let limit = self.cfg.swiglu_limit as f32;
        let (cs, ms) = (self.compute_stream.raw(), self.miss_stream.raw());
        let null: *mut c_void = std::ptr::null_mut();

        // Header item 1. The `xq` these read was produced on the NULL stream, and both expert
        // streams are `hipStreamNonBlocking`, so they do not implicitly join it. This sync is
        // what makes the read safe; it goes away when the attention set takes a stream, not
        // before.
        let t_sync1 = std::time::Instant::now();
        device_sync()?;
        let t_launch = std::time::Instant::now();
        // Residents first, then misses — measured, not tidy: inverting the order cost GLM
        // 3.05 -> 2.44 tok/s, because a resident expert's compute is what overlaps the in-flight
        // ones' reads. Every launch enqueues its ticket's wait first: `wait_on` is the only way
        // to consume a ticket, so a launch cannot happen without its data dependency.
        //
        // SAFETY (both calls below): `descs`/`wexpert`/`h`/`acc`/`x` are `DeviceBuf` fields of
        // `self` sized per the launcher's contract, and every launched range lies within
        // `[0, sel.len())` — compact indices written just above, and `sel.len() = top_k <=
        // n_experts = n_desc` by `V4Config::validate`. Misses accumulate into `acc` row 1 so
        // the two streams never share a cache line.
        let launch_range = |e_start: usize, e_count: usize, acc: *mut u64, stream| unsafe {
            launch_moe_expert_range_f4(
                x, dim, inter, e_start, e_count, n_desc, descs, wexpert, limit, h, acc, 1, stream,
            )
        };
        // The residents' waits are timeline waits on value 0 (`AsyncFetch::wait` early-returns
        // on them, enqueuing nothing) — kept because consuming the ticket is the protocol,
        // not because they gate anything — so ONE range launch computes every resident
        // without waiting on any fetch.
        for i in 0..self.sel.len() {
            if self.tickets[i].is_resident() {
                self.pin.routed.wait_on(self.tickets[i], cs)?;
            }
        }
        if n_res > 0 {
            // The resident device pair brackets exactly this one range launch on the
            // compute stream (the ticket waits above enqueue nothing — value-0
            // timeline waits early-return): its span is the resident-batch compute.
            self.moe_ev[MO_RES0].record(cs)?;
            launch_range(0, n_res, acc, cs)?;
            self.moe_ev[MO_RES1].record(cs)?;
        }
        // Misses stay ONE LAUNCH EACH, each behind its own ticket only: folding them into the
        // resident range (or into one miss range) would gate the whole batch on the LAST fetch
        // to land — 2.40 ms/miss measured at M2 — serialising hits behind misses. The compact
        // index keeps ascending in the same order the placement loop wrote, so straggler `j`
        // reads descriptor `n_res + j`.
        let any_miss = n_res < self.sel.len();
        if any_miss {
            // Opens the miss-stream pair: from here to `MO_MISS1` the stream holds every
            // straggler's ticket wait (the fetch exposure) and every straggler's kernels.
            self.moe_ev[MO_MISS0].record(ms)?;
        }
        for (j, i) in (0..self.sel.len())
            .filter(|&i| !self.tickets[i].is_resident())
            .enumerate()
        {
            self.pin.routed.wait_on(self.tickets[i], ms)?;
            // SAFETY: row 1 of an `MOE_ACC_ROWS * dim` accumulator; see the block above.
            launch_range(n_res + j, 1, unsafe { acc.add(dim) }, ms)?;
        }
        if any_miss {
            self.moe_ev[MO_MISS1].record(ms)?;
        }
        // Header item 2: the drain's contract is that EVERY stream which accumulated into `acc`
        // has already completed.
        let t_sync2 = std::time::Instant::now();
        device_sync()?;
        let t_read = std::time::Instant::now();
        // Pair reads, behind the sync that retired them. ONLY the pairs this call
        // recorded: a reused event keeps its previous recording, so reading an
        // unrecorded pair would book a stale span as this layer's (see [`V4Profile`]).
        if n_res > 0 {
            self.prof.moe_res_ns += ev_span_ns(
                &self.moe_ev[MO_RES0],
                &self.moe_ev[MO_RES1],
                "moe-split resident",
            )?;
        }
        if any_miss {
            self.prof.moe_miss_ns += ev_span_ns(
                &self.moe_ev[MO_MISS0],
                &self.moe_ev[MO_MISS1],
                "moe-split miss",
            )?;
        }
        let t_drain = std::time::Instant::now();
        // The host tiling lands in one place, ordered as the brackets are: the reads
        // between `sync2` and `drain` (the pair queries above) deliberately fall in NO
        // span — they are the instrument's own cost and surface in the printed resid.
        self.prof.moe_desc_ns += t_h2d.duration_since(t_entry).as_nanos();
        self.prof.moe_h2d_ns += t_sync1.duration_since(t_h2d).as_nanos();
        self.prof.moe_sync1_ns += t_launch.duration_since(t_sync1).as_nanos();
        self.prof.moe_launch_ns += t_sync2.duration_since(t_launch).as_nanos();
        self.prof.moe_sync2_ns += t_read.duration_since(t_sync2).as_nanos();
        let dst = self.sub.ptr_mut();
        // `gain` is 1.0 and there is no `--moe-gain` on this path: that flag exists for GLM's
        // int3-vq magnitude sweep and V4 has no such measurement. A knob whose only value is the
        // identity is a knob that can be set wrongly.
        // SAFETY: `sub` row `t` is `dim` f32 and holds the shared expert's output; `acc` is
        // `MOE_ACC_ROWS * dim` u64 and both contributing streams have drained.
        unsafe {
            let row = dst.cast::<f32>().add(t * dim);
            launch_moe_acc_drain(row, acc, dim, MOE_ACC_ROWS, 1.0, null)
        }
        .with_context(|| format!("layer {layer}: draining the MoE accumulator for row {t}"))?;
        self.prof.moe_drain_ns += t_drain.elapsed().as_nanos();
        Ok(())
    }

    /// The resident fp8 shared expert, batched over all `m` rows.
    ///
    /// Batched where the routed experts cannot be: `launch_v4_gemv_fp8` takes `m`, and the shared
    /// expert is one weight set read by every row, so its fp8 weights are read once for the whole
    /// prompt.
    ///
    /// # This is KNOWN WRONG, and the wrongness is named rather than measured later
    ///
    /// `MoE.__init__` passes `swiglu_limit` to `shared_experts` as well as to the routed ones
    /// (model.py:632), and `Expert.forward` clamps `up` on both sides and `gate` from above.
    /// `launch_swiglu` is GLM's, is `silu(g)·u`, and **takes no limit** — so this runs
    /// `v4oracle::Defect::SwigluUnclamped` on all 43 layers, one contribution in seven. Three
    /// further differences ride along, which is why the fix is a kernel and not a parameter
    /// passed to this one: V4 bf16-rounds both operands BEFORE the clamp, bf16-rounds the
    /// product, and uses `F.silu`'s `g·sigmoid(g)` rather than `g/(1 + e^-g)`.
    ///
    /// `launch_v4_swiglu_clamped(g, u, n, limit, h, stream)` is being written elsewhere and this
    /// is the one call site it replaces. Until it lands, **no output from this loop is
    /// reference-faithful**, and `tests/v4_loop.rs` scores against a `Defect::SwigluUnclamped`
    /// oracle rather than the clean one for exactly this reason — which is a gate on the wiring
    /// with a named hole, not a gate on the model.
    fn shared_expert(&mut self, layer: usize, m: usize) -> Result<()> {
        let (dim, inter) = (self.cfg.hidden, self.cfg.moe_inter);
        // BEFORE the launches, not after: a bad `layer` has to return without having issued
        // anything to the device, which is what the single pin lookup here used to guarantee.
        let down = self.pin.layer(layer)?.shared.down;
        self.shared_gate_up(layer, m)?;
        let g = self.sh_g.ptr_mut().cast::<f32>();
        let u = self.sh_u.ptr_mut().cast::<f32>();
        let out = self.sub.ptr_mut().cast::<f32>();
        // SAFETY: `down` is a pin placement outliving this engine; `g`/`u` are `max_m * inter`
        // and `out` is `max_m * dim`, three distinct allocations, and `shared_gate_up` has just
        // filled `g` and `u` for these same `m` rows.
        unsafe {
            // See the doc above: unclamped, and the wrong silu form. One line, and one
            // contribution in seven of every layer's FFN.
            launch_swiglu(g, u, m * inter, g)?;
            // The `w2` input is act-quantized. The routed path does this for itself inside
            // `launch_moe_expert_range_f4` ("The `h` re-quantization between the two passes IS
            // done here, because forgetting it is silent"); here it has to be explicit.
            launch_act_quant_f8(g, m, inter, std::ptr::null_mut())?;
            // WRITES `sub` — does not accumulate into it. `MoE.forward` starts from `y = zeros`
            // and the routed drain adds on top; see `routed_experts`.
            launch_v4_gemv_fp8(
                g,
                down.packed,
                down.scale,
                m,
                dim,
                inter,
                FP8_BLOCK,
                1,
                out,
                NULL_STREAM,
            )?;
        }
        Ok(())
    }

    /// `Expert.forward`'s two input projections for the SHARED expert — `w1` into `sh_g` and
    /// `w3` into `sh_u`, over all `m` rows.
    ///
    /// Split out because [`V4Engine::probe_shared_operands`] has to run exactly these two and
    /// cannot read them back from `shared_expert`: that method writes the SwiGLU product over
    /// `sh_g` in place.
    fn shared_gate_up(&mut self, layer: usize, m: usize) -> Result<()> {
        let (dim, inter) = (self.cfg.hidden, self.cfg.moe_inter);
        let lp = self.pin.layer(layer)?;
        let (gate, up) = (lp.shared.gate, lp.shared.up);
        let xq = self.xq.ptr().cast::<f32>();
        let g = self.sh_g.ptr_mut().cast::<f32>();
        let u = self.sh_u.ptr_mut().cast::<f32>();
        // One loop rather than two launches written out: the two differ ONLY in (weight, dest),
        // and spelling the other eight arguments twice is how `m`/`inter`/`dim` get transposed
        // in one copy and not the other. jscpd does not catch this pair — 25 tokens against a
        // `minTokens` of 15, but the two blocks are not textually adjacent enough to register
        // until the threshold drops to 10, which is below what the gate runs at (measured
        // 2026-08-07 while sizing Track I). So this one is on the reader, and now it is not.
        //
        // SAFETY: `xq` holds `m * dim` fp8-quantized f32; both weights are pin placements
        // outliving this engine; `g` and `u` are `max_m * inter`, three distinct allocations.
        for (w, dst) in [(gate, g), (up, u)] {
            unsafe {
                launch_v4_gemv_fp8(
                    xq,
                    w.packed,
                    w.scale,
                    m,
                    inter,
                    dim,
                    FP8_BLOCK,
                    1,
                    dst,
                    NULL_STREAM,
                )?;
            }
        }
        Ok(())
    }

    /// `MoE.forward` over `m` rows: the gate, the shared expert batched, then the routed experts
    /// one row at a time.
    fn moe(&mut self, layer: usize, m: usize) -> Result<()> {
        let (dim, n_experts) = (self.cfg.hidden, self.cfg.n_experts);
        let gate_w = self.pin.layer(layer)?.gate_w;
        let (xw, xq) = (self.xw.ptr(), self.xq.ptr_mut());
        // SAFETY: `xw` is `m * dim` f32, `gate_logits` is `max_m * n_experts`, `xq` is
        // `max_m * dim`, and `gate_w` is a `[n_experts, hidden]` pin placement.
        unsafe {
            // The gate reads the UNQUANTIZED activation: `Gate.forward` is
            // `linear(x.float(), self.weight.float())`, a dense f32 GEMV with no fp8 anywhere.
            // That is the whole reason `xq` is a separate buffer rather than an in-place
            // quantization of `xw` — quantizing first would feed the router e4m3 values and the
            // error would look like ordinary routing variation.
            //
            // **ONE ROW PER LAUNCH, and not because routing is per-token.** `rivoli_gemv_f32`
            // refuses any `nrow` but 1 or 2 (`kernels/linalg.hip:582`, guard 1004 — `R` is a
            // template parameter and only those two are instantiated). Passing `m` here was this
            // loop's other critical bug: it aborted layer 0's FFN on the FIRST forward of any
            // prompt longer than two tokens, with `gemv_f32: argument guard rejected (1004)`, so
            // no decode and no golden comparison could ever have run. Found by review before any
            // device did. `nrow == 2` is reachable and deliberately unused: V4 is structurally
            // single-row (`kernels/moe.hip:409`), so pairing rows here would buy one fewer
            // launch of a 256x4096 GEMV against a second index space to get wrong. The loop is
            // NOT a null-stream artefact: the guard is on `nrow`, so it survives the rebase.
            let logits = self.gate_logits.ptr_mut().cast::<f32>();
            let x0 = xw.cast::<f32>();
            for t in 0..m {
                launch_gemv_f32(
                    x0.add(t * dim),
                    gate_w,
                    n_experts,
                    dim,
                    1,
                    logits.add(t * n_experts),
                    NULL_STREAM,
                )?;
            }
            // Now quantize, for the experts. In place at block 128 over the full row, which is
            // what every quantized `Linear` in the reference performs.
            memcpy_dtod(xq, xw, m * dim * size_of::<f32>())?;
            launch_act_quant_f8(xq.cast(), m, dim, std::ptr::null_mut())?;
        }
        // Mark 7: the gate chain's last enqueue — the D2H below is what retires it.
        self.route_ev[M_GATE].record(NULL_STREAM)?;
        // The one blocking D2H on the per-layer path, and GLM pays the same one: `route_into` is
        // host math. `m * n_experts` f32 — 1 KB at decode. The bracket is free — the D2H is a
        // join the path already pays (see [`V4Profile`] for what its wait actually contains).
        let t0 = std::time::Instant::now();
        self.gate_logits.copy_out_into(&mut self.gl_host)?;
        self.prof.route_ns += t0.elapsed().as_nanos();
        // The route-split read: the D2H just drained the null stream, so all eight of
        // this layer's marks have retired and each pair is a completed-event query, not
        // a wait.
        if let Some(w0) = self.win_t0.take() {
            self.prof.win_wall_ns += w0.elapsed().as_nanos();
            let span = |a: usize, b: usize| {
                ev_span_ns(&self.route_ev[a], &self.route_ev[b], "route-split")
            };
            self.prof.hcn_ns += span(M_TOP, M_ATTN_NORMED)? + span(M_ATTN_DONE, M_FFN_NORMED)?;
            self.prof.cmp_ns += span(M_ATTN_NORMED, M_SEL)?;
            self.prof.attn_ns += span(M_SEL, M_ATTN_DONE)?;
            // The M6 attn split: three sub-spans tiling `attn` at shared endpoints, so
            // their sum restates it (checked at the print, both signs allowed there).
            self.prof.qkv_ns += span(M_SEL, M_QKV_DONE)?;
            self.prof.attend_ns += span(M_QKV_DONE, M_ATTEND_DONE)?;
            self.prof.oproj_ns += span(M_ATTEND_DONE, M_ATTN_DONE)?;
            self.prof.gate_ns += span(M_FFN_NORMED, M_GATE)?;
        }
        // The shared-expert device pair opens on a just-drained null stream (the D2H
        // above), so its span is the chain's execution, gaps included, span-convention.
        self.moe_ev[MO_SH0].record(NULL_STREAM)?;
        let t0 = std::time::Instant::now();
        self.shared_expert(layer, m)?;
        self.moe_ev[MO_SH1].record(NULL_STREAM)?;
        let sh = t0.elapsed().as_nanos();
        self.prof.moe_ns += sh;
        self.prof.moe_sh_ns += sh;
        for t in 0..m {
            let r0 = std::time::Instant::now();
            self.route_row(layer, t)?;
            let r1 = std::time::Instant::now();
            self.routed_experts(layer, t)?;
            let host = r1.duration_since(r0).as_nanos();
            self.prof.route_ns += host;
            self.prof.route_host_ns += host;
            self.prof.moe_ns += r1.elapsed().as_nanos();
        }
        // Read ONCE per layer, after the row loop: the last `routed_experts`' closing
        // `device_sync` retired the whole device, `MO_SH1` included. Reading inside the
        // row loop would book the pair `m` times at prefill.
        self.prof.moe_shg_ns += ev_span_ns(
            &self.moe_ev[MO_SH0],
            &self.moe_ev[MO_SH1],
            "moe-split shared",
        )?;
        Ok(())
    }

    /// `hc_pre` then the sublayer's `RMSNorm`, into `xw`/`post`/`comb`.
    ///
    /// Split out of [`V4Engine::layer`] so [`V4Engine::probe_pre_norm`] can stop here — the first
    /// gate run localised a real error to somewhere in `hc_pre -> attn_norm -> attention ->
    /// hc_post -> hc_pre -> ffn_norm` and could not say which, because `ffn_norm_out` is the
    /// earliest tensor the loop leaves readable. This is the one before it.
    ///
    /// Idempotent: it reads the residual and writes only scratch, touching no KV ring and no
    /// pooling state. That is what makes it safe to run before a full `layer` on the same input.
    fn pre_norm(&mut self, layer: usize, ffn: bool, m: usize) -> Result<()> {
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        let null = std::ptr::null_mut();
        let lp = self.pin.layer(layer)?;
        let (hcw, norm) = if ffn {
            (lp.hc_ffn, lp.ffn_norm)
        } else {
            (lp.hc_attn, lp.attn_norm)
        };
        // SAFETY: `h[cur]` is `m * hc * dim`; `xw` is `m * dim`, `post` `m * hc`, `comb`
        // `m * hc * hc`, and none of the three aliases `h` — `launch_hc_pre` requires exactly
        // that, since `h` is `__restrict__` in the kernel.
        unsafe {
            launch_hc_pre(
                self.h[self.cur].ptr().cast(),
                hcw.func,
                // `scale` THEN `base`. `launch_v4_hc_head` takes them the other way round and both
                // are `*const f32`, so a swap compiles, runs, and is finite.
                hcw.scale,
                hcw.base,
                m,
                hc,
                dim,
                self.cfg.hc_sinkhorn_iters,
                self.dims.norm_eps,
                self.cfg.hc_eps as f32,
                self.xw.ptr_mut().cast(),
                self.post.ptr_mut().cast(),
                self.comb.ptr_mut().cast(),
                null,
            )?;
            // `launch_v4_rmsnorm`, in place on `hc_pre`'s output — NOT GLM's `linalg.hip::rmsnorm`,
            // which is out-of-place, does not bf16-round, and is SINGLE-ROW (`dim3(1)`, one mean
            // over its whole `n`), so handing it `m * dim` would take a joint statistic over every
            // token and read the norm weight past its allocation. V4's `RMSNorm` returns bf16 and
            // this kernel rounds, which is S3 requirement 1 satisfied by SELECTION rather than by
            // editing a shared kernel.
            launch_v4_rmsnorm(
                self.xw.ptr_mut().cast(),
                norm,
                m,
                dim,
                self.dims.norm_eps,
                NULL_STREAM,
            )?;
        }
        Ok(())
    }

    /// One `Block.forward` over `m` rows at `start_pos`.
    ///
    /// The order is `Block.forward`'s (model.py:695-707) and the oracle's `run_layer`:
    /// `residual = h; hc_pre; attn_norm; attention; hc_post;` then `residual = h; hc_pre;
    /// ffn_norm; moe; hc_post`. Note the SECOND `residual = h` — the FFN's residual is the
    /// POST-ATTENTION `h`, not the block's input. Taking the block input for both is a
    /// silent-wrong no shape check sees, and it is why the loop below re-reads `h[self.cur]`
    /// after the flip rather than hoisting the pointer.
    fn layer(&mut self, layer: usize, m: usize, start_pos: usize) -> Result<()> {
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        let null = std::ptr::null_mut();
        // The route-split window opens (see [`V4Profile`]): the wall clock first, then mark
        // 0, so the window contains the mark by construction. The stream is drained here on
        // every layer but the step's first (the previous layer's end-of-layer sync), where
        // only the embed gather can be pending.
        self.win_t0 = Some(std::time::Instant::now());
        self.route_ev[M_TOP].record(NULL_STREAM)?;
        for ffn in [false, true] {
            self.pre_norm(layer, ffn, m)?;
            // Marks 1 and 6: the sublayer's hc/norm chain is queued — `hcn`'s two spans.
            self.route_ev[if ffn { M_FFN_NORMED } else { M_ATTN_NORMED }].record(NULL_STREAM)?;

            if ffn {
                self.moe(layer, m)?;
            } else {
                self.attention_block(
                    layer,
                    Extent {
                        seqlen: m,
                        start_pos,
                    },
                    null,
                )?;
            }

            let dst = 1 - self.cur;
            // SAFETY: `sub` is `m * dim`, `post`/`comb` are as above, and `h[cur]`/`h[dst]` are
            // DISTINCT allocations — that is `launch_hc_post`'s "`y` must not alias `residual`".
            unsafe {
                launch_hc_post(
                    self.sub.ptr().cast(),
                    self.h[self.cur].ptr().cast(),
                    self.post.ptr().cast(),
                    self.comb.ptr().cast(),
                    m,
                    hc,
                    dim,
                    self.h[dst].ptr_mut().cast(),
                    null,
                )?;
            }
            self.cur = dst;
        }
        // Header item 3: one join per layer, so layer `L+1`'s first atomic into `moe_acc` cannot
        // race `L`'s drain. GLM pays the same one for the same reason.
        device_sync()?;
        Ok(())
    }

    /// Bind this step's token ids and bound its position; returns the row count.
    ///
    /// **One function because the two entry points into [`V4Engine::layer`] fell out of step.**
    /// `forward` had the position bound and `probe_layer` did not, which a review found: the
    /// rotary tables are built at exactly `max_ctx` positions and `launch_v4_rope` reads
    /// `tbl + pos0 * rd` with no bound of its own, so a probe driver walking further than the
    /// engine was sized for read past the table — plausible garbage frequencies, which is
    /// `Defect::RopeNoYarn`'s shape. Two copies of two guards is two places for them to diverge
    /// again, and `build.rs` said so.
    ///
    /// `ids` and not just `m`, because a hash layer's gate indexes `tid2eid` by TOKEN ID: a
    /// caller that passed row counts alone would route every layer by the previous step's tokens,
    /// and the difference looks like ordinary routing variation.
    fn begin_step(&mut self, ids: &[u32], start_pos: usize) -> Result<usize> {
        let m = ids.len();
        ensure!(
            m > 0 && m <= self.max_m,
            "{m} rows into buffers sized for {}",
            self.max_m
        );
        ensure!(
            start_pos + m <= self.max_ctx,
            "position {} exceeds the {} this engine was sized for",
            start_pos + m,
            self.max_ctx
        );
        self.step_ids.clear();
        self.step_ids.extend_from_slice(ids);
        Ok(m)
    }

    /// One forward pass over `m` rows starting at `start_pos`, leaving the LAST row's logits in
    /// `self.logits`.
    fn forward(&mut self, ids: &[u32], start_pos: usize) -> Result<()> {
        let m = self.begin_step(ids, start_pos).context("forward")?;
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        // `h[0]` is where the embedding lands, so every pass starts from the same buffer
        // regardless of how many times the previous one flipped (`layer` flips twice, so the count
        // is always even and `cur` returns to 0 — this is belt, not the mechanism).
        self.cur = 0;
        self.loaded_rows = m;
        // `embed` then the `hc_mult` copies — `Transformer.forward` 914-916. One launch per token
        // because `launch_v4_embed_bf16_row` gathers ONE row and broadcasts it into `hc` copies.
        for (t, &tok) in ids.iter().enumerate() {
            ensure!(
                (tok as usize) < self.cfg.vocab,
                "token id {tok} is outside the {} the artifact holds",
                self.cfg.vocab
            );
            // SAFETY: `embed.packed` is `[vocab, hidden]` bf16 and `tok < vocab` was just
            // checked; the destination is row `t` of an `m * hc * dim` allocation with
            // `t < m <= max_m`.
            unsafe {
                launch_v4_embed_bf16_row(
                    self.pin.embed.packed,
                    tok as usize,
                    dim,
                    hc,
                    self.h[0].ptr_mut().cast::<f32>().add(t * hc * dim),
                    NULL_STREAM,
                )?;
            }
        }
        for l in self.range.clone() {
            self.layer(l, m, start_pos)?;
        }
        // Launch time only — the tail's GPU wall drains into `argmax`'s sync, the other half
        // of the same bracket.
        let t0 = std::time::Instant::now();
        let r = self.head_tail(m);
        self.prof.tail_ns += t0.elapsed().as_nanos();
        r
    }

    /// `hc_head`, the final `RMSNorm`, `ParallelHead` — the last three ops of
    /// `Transformer.forward`, over the LAST of `m` rows.
    ///
    /// Split from [`V4Engine::forward`] so [`V4Engine::probe_head_tail`] can drive it on a
    /// residual the caller supplied. That is what makes it gateable at all: the port records
    /// that these three ops "have neither an implementation nor a golden" and that "the first
    /// decode's logits are ungated by construction" — `bin/v4-oracle`'s `head.probe.*` goldens
    /// exist precisely because the oracle runs them on a declared probe rather than on the layer
    /// chain, and this split is the engine side of that.
    fn head_tail(&mut self, m: usize) -> Result<()> {
        let (dim, hc) = (self.cfg.hidden, self.cfg.hc_mult);
        // `Transformer.forward:923` is `self.head(self.norm(h))` and `ParallelHead` slices
        // `x[:, -1]` AFTER the norm — but RMSNorm is per row, so norming one row and norming all
        // `m` then keeping the last are the same arithmetic. `Defect::HeadNormOverAllTokens` is
        // the defect that would make them differ, and it is a defect precisely because the
        // statistic is per row.
        let last = m - 1;
        let h_last = self.h[self.cur].ptr();
        // SAFETY: `h[cur]` row `last` is `hc * dim` f32 within an `m * hc * dim` allocation;
        // `head_pre` is `hc` writable; `head_x` is `dim`; `logits` is `vocab`. None aliases
        // another, which every parameter of these three launchers requires.
        unsafe {
            launch_v4_hc_head(
                h_last.cast::<f32>().add(last * hc * dim),
                self.pin.hc_head.func,
                // `base` THEN `scale` — the OPPOSITE order from `launch_hc_pre` above. Both are
                // `*const f32`, and `hc_head_scale` is `[1]` where a block's `hc_*_scale` is
                // `[3]`, so a swap reads one of three floats as the scalar and three floats of
                // `base` as the `[3]`: finite, plausible, wrong.
                self.pin.hc_head.base,
                self.pin.hc_head.scale,
                self.head_pre.ptr_mut().cast(),
                self.head_x.ptr_mut().cast(),
                1,
                hc,
                dim,
                self.dims.norm_eps,
                self.cfg.hc_eps as f32,
                std::ptr::null_mut(),
            )?;
            launch_v4_rmsnorm(
                self.head_x.ptr_mut().cast(),
                self.pin.final_norm,
                1,
                dim,
                self.dims.norm_eps,
                NULL_STREAM,
            )?;
            // `head.weight` is bf16 in the artifact and there is no int8 head to reach for.
            // `launch_v4_dense_gemm_bf16` computes exactly this at `m = 1`: runtime `(m, n, k)` over
            // a bf16 `[n, k]` weight is a head GEMV at `m = 1, n = vocab, k = hidden`.
            //
            // **Verified at TOY extents, not these.** `tests/v4_head_tail.rs`'s
            // `the_lm_head_needs_no_kernel_of_its_own` runs it at vocab 1024 / dim 512 against
            // `rel < 1e-3`, and says outright that the real `(1, 129280, 4096)` shape is not
            // covered — four literal assertions claiming otherwise were deleted there for being
            // unfireable. An earlier version of this comment quoted "rel 2.33e-7, verified on
            // device", a figure that appears nowhere in this tree; corrected rather than sourced.
            //
            // The objection to reusing it is SHAPE, not capability: one wave per output element over
            // a one-row activation is a 129,280-wave launch. That is a performance argument with no
            // measurement attached, so the honest instruction was "call it first, then price it".
            // Called; S4 prices it.
            launch_v4_dense_gemm_bf16(
                self.head_x.ptr().cast(),
                self.pin.head.packed,
                self.logits.ptr_mut().cast(),
                1,
                self.cfg.vocab,
                dim,
                std::ptr::null_mut(),
            )?;
        }
        Ok(())
    }

    /// Greedy argmax over `self.logits`, with the finiteness check that turns a NaN blow-up into
    /// a message instead of a plausible token.
    ///
    /// Header item 4 lives here: the `device_sync` before the D2H. `logits` was produced on the
    /// null stream and `launch_argmax` runs there too, so it is nominally redundant today — and
    /// it is exactly the redundancy that stops being redundant the moment the head tail moves
    /// onto a stream. This port's record is that the sync someone forgets is the failure mode, so
    /// it is written down as owned rather than inferred from where the launchers happen to run.
    fn argmax(&mut self) -> Result<u32> {
        // The tail bracket's blocking half: this `device_sync` is where the head GEMV and
        // everything else still on the null stream drains, a join the decode already pays —
        // so the bracket is two `Instant` reads and no new sync.
        let t0 = std::time::Instant::now();
        // SAFETY: `logits` holds `vocab` f32; `argmax_dev` holds an i32 then an f32, and the
        // second pointer is offset by exactly the first's width.
        unsafe {
            launch_argmax(
                self.logits.ptr().cast(),
                self.cfg.vocab,
                self.argmax_dev.ptr_mut().cast(),
                self.argmax_dev.ptr_mut().add(size_of::<i32>()).cast(),
            )?;
        }
        device_sync()?;
        // `argmax_dev` is allocated at exactly `i32 + f32` and `copy_out_into` yields that or
        // errors, so the slicing below is total. A length `ensure!` was written here and cut for
        // being unable to fire — the failure mode this stage was briefed on.
        self.argmax_dev.copy_out_into(&mut self.argmax_host)?;
        self.prof.tail_ns += t0.elapsed().as_nanos();
        let idx = i32::from_le_bytes(self.argmax_host[..4].try_into()?);
        let val = f32::from_le_bytes(self.argmax_host[4..8].try_into()?);
        // A non-finite top logit is a defect and not a sampling outcome. It is also the visible
        // half of the read-before-write hazard the ticket protocol exists to prevent — an
        // unloaded `.f4` slot decodes to large-but-FINITE garbage (`0x7FC0_7FC0` is e8m0 `0x7f`
        // and `0xc0`, scales of 2^0 and 2^65), so this catches the loud case only and the silent
        // one is the oracle's job.
        ensure!(
            val.is_finite(),
            "argmax read a non-finite logit ({val}): the forward pass produced NaN or inf"
        );
        u32::try_from(idx).context("argmax returned a negative index")
    }

    // ── probes ─────────────────────────────────────────────────────────────────────────────
    //
    // The per-layer gate's only entry, and the reason this loop is gateable at all.
    //
    // **Not behind a feature, and the rule that looks broken does not apply.** CLAUDE.md's
    // "instruments go behind a feature AND a flag" is about instruments that put work on the
    // per-token decode path — `--ppl` and `--pred-probe` both do, which is why a tok/s from a
    // probe build means nothing. These put none: nothing inside this engine calls them, so a
    // stock build pays for them exactly what it pays for `DeviceBuf::copy_out`. Each is a
    // DIAGNOSTIC in the sense `DeviceBuf::copy_out_raw`'s own doc means it.
    //
    // They exist in this shape because it is the oracle's. `bin/v4-oracle`'s `drive` supplies
    // each phase's residual from `h_for` rather than from the previous phase, and its `defects`
    // command hands back a fixed probe "so a defect's effect is isolated to the layer rather
    // than inherited from the layers before it". Scoring a layer on the engine's own accumulated
    // residual would make every layer's number a function of every earlier layer's.

    /// Overwrite the residual stream with `h` — `[m, hc_mult, dim]` f32, row-major.
    pub fn set_residual(&mut self, h: &[f32]) -> Result<()> {
        let row = self.cfg.hc_mult * self.cfg.hidden;
        ensure!(
            h.len().is_multiple_of(row) && !h.is_empty() && h.len() / row <= self.max_m,
            "set_residual: {} floats is not 1..={} rows of {row}",
            h.len(),
            self.max_m
        );
        self.cur = 0;
        self.loaded_rows = h.len() / row;
        self.h[0].copy_in_at(0, as_le_bytes(h))
    }

    /// Read the residual stream back — the first `m` rows of `[m, hc_mult, dim]`.
    pub fn residual(&self, m: usize) -> Result<Vec<f32>> {
        device_sync()?;
        read_prefix(&self.h[self.cur], m * self.cfg.hc_mult * self.cfg.hidden)
    }

    /// Drive ONE layer over `ids` at `start_pos`, on whatever residual is loaded.
    ///
    /// `ids` is taken (rather than only `m`) because a hash layer's gate indexes `tid2eid` by
    /// TOKEN ID: a probe that passed row counts alone would route every layer by the previous
    /// step's tokens and the difference would look like ordinary routing variation.
    ///
    /// **NOT idempotent, and a driver that retries a step corrupts it.** This writes the layer's KV
    /// ring at `pos % window` and — on a compressing layer — deposits into the pooling state and
    /// slides the window. Calling it twice for one `(layer, start_pos)`, which is a plausible thing
    /// for a driver re-scoring a comparison to do, double-deposits exactly the state S3
    /// requirement 3 is about. `tests/v4_attn.rs` splits `step` from `attend` for this reason;
    /// there is no such split here, so the obligation is the caller's and is written down.
    pub fn probe_layer(&mut self, layer: usize, ids: &[u32], start_pos: usize) -> Result<()> {
        ensure!(
            self.range.contains(&layer),
            "probe_layer: {layer} is outside this artifact's [{}, {})",
            self.range.start,
            self.range.end
        );
        let m = self.begin_step(ids, start_pos).context("probe_layer")?;
        ensure!(
            self.loaded_rows == m,
            "probe_layer: the residual holds {} row(s) and this step drives {m} — the result would \
             be row 0 scored as row {}",
            self.loaded_rows,
            m.saturating_sub(1)
        );
        self.layer(layer, m, start_pos)
    }

    /// The head tail on the loaded residual: `hc_head`, the final norm, `ParallelHead`. Returns
    /// the `[vocab]` logits of row `m - 1`.
    ///
    /// **This is what closes the hole the port names as structural** — "the last three ops of
    /// `Transformer.forward` have neither an implementation nor a golden … the first decode's
    /// logits are ungated by construction". They have both now: the implementation is
    /// [`V4Engine::head_tail`] and the golden is `bin/v4-oracle`'s `head.probe.logits`, taken on
    /// a DECLARED probe rather than on the layer chain — which is exactly why it is a golden
    /// (composing 43 layers at `--layers 2` would produce a logits vector that is not any
    /// quantity the model computes, and `fixed_probe`'s doc says so).
    pub fn probe_head_tail(&mut self, m: usize) -> Result<Vec<f32>> {
        ensure!(m > 0 && m <= self.max_m, "probe_head_tail: {m} rows");
        ensure!(
            self.loaded_rows == m,
            "probe_head_tail: the residual holds {} row(s), not {m} — the head tail slices row {}",
            self.loaded_rows,
            m.saturating_sub(1)
        );
        self.head_tail(m)?;
        device_sync()?;
        let mut bytes = Vec::new();
        self.logits.copy_out_into(&mut bytes)?;
        Ok(read_f32(&bytes))
    }

    /// This engine's pool counters, so a test can assert the streaming path actually ran.
    /// Oversubscription is not the same as a miss.
    pub fn pool_hits(&self) -> u64 {
        self.pin.routed.hits() - self.hits0
    }

    /// See [`V4Engine::pool_hits`].
    pub fn pool_misses(&self) -> u64 {
        self.pin.routed.misses() - self.misses0
    }

    /// One `attention` call's four readable tensors. See [`AttnStages`] for what each implicates,
    /// for why they are not equally sharp, and for why `attn_core_out` cannot be among them.
    ///
    /// **Only sound on a ratio-0 layer, and that is not a caller's convention — it is checked.**
    /// It runs the attention half, which writes the KV ring. On a `Plain` layer that write is
    /// idempotent (prefill copies the same rows, decode rewrites the same slot with the same
    /// bytes), so a later `probe_layer` on the same step is unaffected. On a compressing layer it
    /// is emphatically not: `compress` read-modify-writes the pooling state and the decode path
    /// slides the window, so a second pass double-deposits — the exact trap
    /// `v4compress::compress` documents at its "never a second call".
    pub fn probe_attn_stages(
        &mut self,
        layer: usize,
        ids: &[u32],
        start_pos: usize,
    ) -> Result<AttnStages> {
        let kind = self.layers[layer - self.range.start].kind;
        ensure!(
            kind.compressor_ratio().is_none(),
            "probe_attn_stages on layer {layer} ({kind:?}): re-running the attention half of a \
             compressing layer double-deposits into its pooling state"
        );
        let m = self
            .begin_step(ids, start_pos)
            .context("probe_attn_stages")?;
        ensure!(
            self.loaded_rows == m,
            "probe_attn_stages: residual holds {} rows, not {m}",
            self.loaded_rows
        );
        let p = Extent {
            seqlen: m,
            start_pos,
        };
        self.pre_norm(layer, false, m)?;
        self.attention_block(layer, p, NULL_STREAM)?;
        // ONE join for all four readbacks below. `read_prefix` deliberately does not sync.
        device_sync()?;
        let (dim, hd) = (self.cfg.hidden, self.cfg.head_dim);
        let nhd = self.cfg.n_heads * hd;
        Ok(AttnStages {
            q: read_prefix(&self.a_q, m * nhd)?,
            kv_entry: read_prefix(&self.a_kv, m * hd)?,
            attn_derot: read_prefix(&self.a_o, m * nhd)?,
            attn_out: read_prefix(&self.sub, m * dim)?,
        })
    }

    /// `hc_pre` + the sublayer norm ALONE, returning `xw`.
    ///
    /// The earliest comparable tensor in a block: `attn_norm_out` at `ffn = false`. Idempotent, so
    /// it may run before [`V4Engine::probe_layer`] on the same loaded residual.
    pub fn probe_pre_norm(
        &mut self,
        layer: usize,
        ffn: bool,
        ids: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        let m = self.begin_step(ids, start_pos).context("probe_pre_norm")?;
        ensure!(
            self.loaded_rows == m,
            "probe_pre_norm: residual holds {} rows, not {m}",
            self.loaded_rows
        );
        self.pre_norm(layer, ffn, m)?;
        self.probe_working(m)
    }

    /// The `[m, dim]` working tensor — `hc_pre`'s output after its `RMSNorm`.
    ///
    /// **The bisection point the block-output comparison lacks.** After [`V4Engine::probe_layer`]
    /// this holds the FFN's `ffn_norm_out`: `moe` copies it into `xq` and quantizes THAT, leaving
    /// this intact. So everything up to and including attention, `hc_post`, the second `hc_pre` and
    /// its norm is UPSTREAM of it, and the entire MoE — router, routed experts, shared expert,
    /// accumulator drain — is downstream. `L{l}.{tag}.ffn_norm_out` is the golden.
    ///
    /// Added after the first gate run came back at `max_rel 23.78` on `L0.pre.out` with no way to
    /// say which half moved, which is the limitation that run's own comment predicted.
    pub fn probe_working(&self, m: usize) -> Result<Vec<f32>> {
        device_sync()?;
        read_prefix(&self.xw, m * self.cfg.hidden)
    }

    /// The routing decision of the LAST row the previous [`V4Engine::probe_layer`] drove:
    /// `(expert ids, their weights)` in selection order.
    ///
    /// **The router is otherwise ungated, and it carries the invisible defect.**
    /// `Defect::RouterBiasedWeights` — taking the weights from the bias-shifted `choice` instead
    /// of from `scores` — is a one-line "simplification" that changes every routed magnitude by an
    /// amount that looks like ordinary variation, and `hc_post` then mixes the FFN output with
    /// four residual copies, so it dilutes below any bound a block-output comparison would set.
    /// `bin/v4-oracle` emits `router_weights` and `router_indices` per layer per phase; this is
    /// what they can be compared against. A review found the test asserting on two of seven
    /// available goldens and this being one of the five missing.
    ///
    /// The LAST row, because `sel`/`wexpert_host` hold whatever the final `moe_row` wrote — which
    /// at prefill is row `m - 1`. Comparing against any other row of the golden would silently
    /// pass or fail on the wrong data, so the caller is told which row this is.
    pub fn probe_route(&self) -> (Vec<usize>, Vec<f32>) {
        let w = self.sel.iter().map(|&e| self.wexpert_host[e]).collect();
        (self.sel.clone(), w)
    }

    /// The shared expert's two SwiGLU operands, recomputed and read back — `(gate, up)`, each
    /// `[m, moe_inter]`.
    ///
    /// **The one measurement that says whether [`V4Engine::shared_expert`]'s missing clamp
    /// actually binds at this prompt.** `Expert.forward` does `up.clamp(-limit, limit)` and
    /// `gate.min(limit)` at `swiglu_limit = 10.0`; if neither operand reaches 10 the clamp is
    /// INERT here and the whole deviation reduces to `F.silu`'s multiply form plus one missing
    /// bf16 round of the product. If either does reach it, the deviation is the clamp and the
    /// separation is unbounded. Those are very different findings and no golden distinguishes
    /// them, because the golden only sees the sum of seven contributions.
    ///
    /// Must be called immediately after the layer whose FFN is in question: it reads `xq`, which
    /// still holds that layer's quantized `ffn_norm` output until the next layer's attention
    /// overwrites it. It recomputes the two GEMVs rather than reading `sh_g` back, because
    /// `shared_expert` writes the SwiGLU product over `sh_g` in place.
    pub fn probe_shared_operands(
        &mut self,
        layer: usize,
        m: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let inter = self.cfg.moe_inter;
        self.shared_gate_up(layer, m)?;
        device_sync()?;
        Ok((
            read_prefix(&self.sh_g, m * inter)?,
            read_prefix(&self.sh_u, m * inter)?,
        ))
    }

    /// Prefill, then greedy decode. Returns the generated ids.
    ///
    /// **Speculative decode is not available on this path and cannot be.**
    /// `kernels/moe.hip:409` refuses `nrow != 1` (guard 1003; only `R = 1` is instantiated, and
    /// its comment records that the `1.108x` measurement justifying `R = 2` for GLM's VQ and
    /// int4 kernels does not exist for V4 — the S1b oracle is `bsz = 1` only, so a two-row FP4
    /// kernel could not be scored). So there is no `mtp` argument here to switch off; `main` says
    /// so once at startup, the way it already does for Vulkan.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new: usize,
        eos: &[u32],
        on_tok: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Vec<u32>> {
        self.reset()?;
        let t0 = std::time::Instant::now();
        // Prefill: ONE call over the whole prompt. Not a choice — `attention`'s prefill arm seeds
        // the ring from the prompt's last `window` positions, and `compress`'s prefill arm pools
        // every complete block in one go. Both are whole-prompt by construction, which is also
        // why every `[m, ..]` buffer is sized for the prompt rather than for `MAXROW`.
        self.forward(prompt, 0)?;
        let prefill = t0.elapsed();
        let mut cur = self.argmax()?;
        let mut out = Vec::with_capacity(max_new);
        let t1 = std::time::Instant::now();
        // Decode-only buckets: whatever the prefill accumulated is discarded here, and the
        // pool counters are re-read here, so the PROFILE line below decomposes exactly the
        // wall that `tok/s` is computed over — not the prefill's.
        self.prof = V4Profile::default();
        let fetch0 = self.pin.routed.fetch_ns();
        let miss1 = self.pin.routed.misses();
        let mut pos = prompt.len();
        loop {
            if eos.contains(&cur) {
                break;
            }
            out.push(cur);
            if !on_tok(cur) || out.len() >= max_new {
                break;
            }
            // **Prefill and decode index different spaces**, and nothing in the types says so:
            // `start_pos == 0` means the selection's window columns are ABSOLUTE positions
            // `0..seqlen` over the prompt's own KV, and `start_pos > 0` means they are ring SLOTS
            // `0..window_size`. `v4compress::compress_offset` owns that split and
            // `attn::v4::Pass` is the discriminant. What this loop contributes is that `pos` is
            // never 0 here — the prefill consumed position 0 — so the decode arm is unreachable
            // with a prefill's index space and vice versa.
            self.forward(&[cur], pos)?;
            cur = self.argmax()?;
            pos += 1;
        }
        let (hits, misses) = (self.pin.routed.hits(), self.pin.routed.misses());
        let (dh, dm) = (hits - self.hits0, misses - self.misses0);
        let stride = f4_expert_stride(self.cfg.hidden, self.cfg.moe_inter);
        let decode = t1.elapsed().as_secs_f64();
        tracing::info!(
            "v4 decode: prefill {:.2}s over {} tokens; {} generated in {decode:.2}s = {:.3} tok/s; \
             expert lookups {dh} hit / {dm} miss ({:.1}% hit), {:.2} GB fetched, {:.1} misses/token",
            prefill.as_secs_f64(),
            prompt.len(),
            out.len(),
            out.len() as f64 / decode.max(f64::EPSILON),
            100.0 * dh as f64 / (dh + dm).max(1) as f64,
            dm as f64 * stride as f64 / 1e9,
            dm as f64 / out.len().max(1) as f64,
        );
        // The per-token phase decomposition, decode-only (see [`V4Profile`] for what each
        // bucket brackets). Same denominator as `tok/s` above — `out.len()` — so the terms
        // and the wall stay one arithmetic. `fetch`/`miss` here are the DECODE deltas, where
        // the hit/miss line above spans prefill too; both GLM conventions, kept so the two
        // engines' lines read alike.
        let n = out.len().max(1) as f64;
        let wall_ms = decode * 1e3 / n;
        let per = |ns: u128| ns as f64 / 1e6 / n;
        let route_ms = per(self.prof.route_ns);
        let moe_ms = per(self.prof.moe_ns);
        let tail_ms = per(self.prof.tail_ns);
        let fetch_ns = (self.pin.routed.fetch_ns() - fetch0) as f64;
        let dec_miss = self.pin.routed.misses() - miss1;
        let remainder_ms = wall_ms - route_ms - moe_ms;
        debug_assert!(
            remainder_ms >= 0.0,
            "negative PROFILE remainder ({remainder_ms:.3} ms/tok): a bucket exceeds the wall"
        );
        // 1e-6: the three buckets and the wall are independently-rounded f64 sums, so the
        // comparison needs a slack far below the µs a real violation would show.
        debug_assert!(
            tail_ms <= remainder_ms + 1e-6,
            "tail ({tail_ms:.3} ms/tok) exceeds the remainder ({remainder_ms:.3}): the tail \
             bracket overlaps a phase bucket"
        );
        // No `(gpu Nms)` term: GLM's is a HIP-event bracket on its compute stream, and V4 has
        // no event pair to read at an existing join yet. Absent rather than zero. moe/fetch at
        // 0.1 ms where GLM prints whole ms: its moe wall is ~250 ms, V4's terms are expected
        // an order smaller, and the decomposition exists to resolve them. The raw miss count
        // rides beside the rounded rate because M2 could recover it only to ±2 from the
        // per-token print; `tail` prints inside the remainder because it is a sub-span of it,
        // not a fourth bucket.
        tracing::info!(
            "PROFILE/tok: {wall_ms:.1}ms wall | route {route_ms:.1}ms | moe {moe_ms:.1}ms | \
             fetch {:.1}ms | {:.2} miss ({dec_miss} raw), {:.2}ms/miss, {:.2} GB | \
             remainder {remainder_ms:.1}ms (tail {tail_ms:.1}ms)",
            fetch_ns / 1e6 / n,
            dec_miss as f64 / n,
            fetch_ns / 1e6 / dec_miss.max(1) as f64,
            dec_miss as f64 * stride as f64 / 1e9 / n,
        );
        // The route split (M4, see [`V4Profile`]): where the D2H wait's GPU time goes. The
        // four spans tile marks 0→7 inside `win`, the host wall that contains them; `resid`
        // is the wall's unexplained share (the D2H copy + pre-mark-0 lag) and the summing
        // check itself. `d2h + host` restates `route` as its halves, same accumulators.
        let (win_ms, hcn_ms) = (per(self.prof.win_wall_ns), per(self.prof.hcn_ns));
        let (cmp_ms, attn_ms) = (per(self.prof.cmp_ns), per(self.prof.attn_ns));
        let (gate_ms, host_ms) = (per(self.prof.gate_ns), per(self.prof.route_host_ns));
        let resid_ms = win_ms - (hcn_ms + cmp_ms + attn_ms + gate_ms);
        let d2h_ms = route_ms - host_ms;
        // 5e-2 ms and not 1e-6: `win` is a host clock and the spans are GPU timestamps, so
        // the slack must cover their rate skew over ~80 ms of summed spans — ~100 ppm ≈
        // 10 µs, and 100 ppm is a typical oscillator figure rather than a bound, hence the
        // headroom. In practice `resid` carries a structurally POSITIVE floor (43 layers of
        // D2H copy + pre-mark-0 lag), so a fire means a mark recorded outside its window,
        // which shows whole milliseconds.
        debug_assert!(
            resid_ms >= -5e-2,
            "negative ROUTE-SPLIT residual ({resid_ms:.3} ms/tok): an event span exceeds \
             the window wall that contains it by construction"
        );
        tracing::info!(
            "ROUTE-SPLIT/tok: attn {attn_ms:.1}ms | cmp {cmp_ms:.1}ms | hcn {hcn_ms:.1}ms | \
             gate {gate_ms:.1}ms | win {win_ms:.1}ms (resid {resid_ms:.1}ms) | \
             d2h {d2h_ms:.1}ms + host {host_ms:.1}ms"
        );
        // The attn split (M6, see [`V4Profile`]): three sub-spans tiling `attn` at
        // SHARED marks, so the residual is an identity check, not a remainder — four
        // independent float queries of the same four events, allowed either sign.
        let (qkv_ms, attend_ms) = (per(self.prof.qkv_ns), per(self.prof.attend_ns));
        let oproj_ms = per(self.prof.oproj_ns);
        let attn_resid_ms = attn_ms - (qkv_ms + attend_ms + oproj_ms);
        // 5e-2 as the route-split slack, and two-sided where that one is one-sided:
        // there is no structurally positive floor here — the spans share endpoints, so
        // anything past query rounding (~0.5 µs/read × 43 layers) is a transposed pair.
        debug_assert!(
            attn_resid_ms.abs() <= 5e-2,
            "ATTN-SPLIT does not tile attn (resid {attn_resid_ms:.3} ms/tok): the three \
             sub-spans share endpoints with the attn span by construction"
        );
        tracing::info!(
            "ATTN-SPLIT/tok: qkv {qkv_ms:.1}ms | attend {attend_ms:.1}ms | \
             oproj {oproj_ms:.1}ms | attn {attn_ms:.1}ms (resid {attn_resid_ms:.2}ms)"
        );
        // The moe split (M6, see [`V4Profile`]): seven host spans tiling `moe` (resid =
        // clock gaps + the instrument's own pair reads, non-negative by construction),
        // then the three device attributions — NOT addends: shared overlaps route_row's
        // host math, res and miss overlap each other across streams.
        let (sh_ms, desc_ms) = (per(self.prof.moe_sh_ns), per(self.prof.moe_desc_ns));
        let (h2d_ms, sync1_ms) = (per(self.prof.moe_h2d_ns), per(self.prof.moe_sync1_ns));
        let (launch_ms, sync2_ms) = (per(self.prof.moe_launch_ns), per(self.prof.moe_sync2_ns));
        let drain_ms = per(self.prof.moe_drain_ns);
        let moe_resid_ms =
            moe_ms - (sh_ms + desc_ms + h2d_ms + sync1_ms + launch_ms + sync2_ms + drain_ms);
        // -1e-6: pure f64 rounding slack — the spans are disjoint sub-intervals of the
        // two host brackets `moe` sums, on one thread, so a real negative means a
        // bracket left its interval.
        debug_assert!(
            moe_resid_ms >= -1e-6,
            "negative MOE-SPLIT residual ({moe_resid_ms:.3} ms/tok): a host span exceeds \
             the moe bucket that contains it by construction"
        );
        let (shg_ms, res_ms) = (per(self.prof.moe_shg_ns), per(self.prof.moe_res_ns));
        let miss_ms = per(self.prof.moe_miss_ns);
        tracing::info!(
            "MOE-SPLIT/tok: sh_enq {sh_ms:.1}ms | desc {desc_ms:.1}ms | h2d {h2d_ms:.1}ms | \
             sync1 {sync1_ms:.1}ms | launch {launch_ms:.1}ms | sync2 {sync2_ms:.1}ms | \
             drain {drain_ms:.1}ms | moe {moe_ms:.1}ms (resid {moe_resid_ms:.1}ms) | \
             gpu: shared {shg_ms:.1}ms, res {res_ms:.1}ms, miss {miss_ms:.1}ms"
        );
        self.pin.routed.flush_trace()?;
        Ok(out)
    }
}

/// The device-free half of this loop's own checks.
///
/// Everything here runs with no GPU: it is arithmetic and placement rules. The device-touching
/// half is `tests/v4_loop.rs`, which drives the loop against `bin/v4-oracle`'s real-weight
/// per-layer goldens — and the split matters, because the two things this module gets wrong
/// most easily are **invisible to a numeric golden**. The port measured that: injecting the
/// append rule left every attention golden bit-identical, `attn_out` included, on a script built
/// specifically to expose it, because the two rules differ by a permutation of a region the
/// selection covers uniformly and `sparse_attn`'s softmax is permutation-invariant over a set.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    /// The shipped config's ceiling, and the two facts that make it 2052 rather than 2048.
    ///
    /// `index_topk + 1` and not `index_topk`: truncation begins when the block COUNT exceeds
    /// `index_topk`, and the count reaches `index_topk` at position `ratio * index_topk`, which
    /// is still fine. Off by one here is 4 positions of silently-dropped context.
    #[test]
    fn the_positional_selection_ceiling_is_the_indexers_truncation_point() {
        assert_eq!(
            positional_context_limit(512),
            2052,
            "the shipped config's bound"
        );
        // The bound is on `start_pos + seqlen`, so it scales with `index_topk` and not with the
        // layer count or the window. A config with a smaller indexer is a tighter engine.
        assert_eq!(positional_context_limit(0), INDEXED_RATIO);
        assert_eq!(positional_context_limit(1), 2 * INDEXED_RATIO);
    }

    /// INV-7. A compressed block's destination row is a pure function of its POSITION, in
    /// **both coordinate systems the layer loop writes** — which is the half of the invariant
    /// that belongs to `v4gpu` rather than to `v4compress`.
    ///
    /// `tests/v4_compress.rs::compress_dst_is_positional_and_an_appending_placer_disagrees`
    /// already proves the other half comprehensively (the gapped script against an appending
    /// placer, the ratio-128 boundary, `region_base = 0` for the indexer's nested compressor,
    /// and the `Plain` refusal before either division), so the §8b row cites BOTH — the same
    /// two-halves shape INV-4 and INV-6 use. Re-deriving it here would be a duplicate that
    /// `build.rs` happened not to catch. The short gap check below stays because the registry
    /// row's "never the next free slot" clause needs a proof `tests/invariants.rs` can SEE, and
    /// that test walks `src/` only.
    ///
    /// **The table is hand-spelled and that is the point.** Comparing the loop's placement
    /// against `compress_dst` would be circular — the loop calls `compress_dst`. These rows are
    /// the only non-circular statement of the rule available, which is the argument
    /// `docs/investigations/v4-flash-port.md`'s carried note 2 makes about keeping `COMP_SLOTS`
    /// when the harness switched onto the shipped function.
    #[test]
    fn inv_7_a_compressed_blocks_row_is_a_pure_function_of_its_position() {
        const RATIO: usize = 8;
        const WIN: usize = 8;
        let kind = LayerKind::from_ratio(RATIO);

        // THE NEW FACT: at prefill, ONE block has TWO destinations, and they differ. The
        // persistent one is the region base `WIN`; the selection-space one is the prompt length,
        // because `sparse_attn` reads `torch.cat([kv, kv_compress])` and indexes it as one space.
        let seqlen = 12usize;
        let sel_base = compress_offset(WIN, seqlen, 0);
        assert_eq!(
            compress_dst(kind, WIN, seqlen, 0),
            Some((WIN, seqlen / RATIO))
        );
        assert_eq!(
            compress_dst(kind, sel_base, seqlen, 0),
            Some((seqlen, seqlen / RATIO))
        );
        assert_ne!(
            WIN, seqlen,
            "the two prefill destinations must differ, or this proves nothing"
        );

        // ...and at decode they COINCIDE, which is what says decode has one destination rather
        // than two. The loop branches on this equality and not on the phase, so it is the
        // property that makes the single branch correct.
        for pos in [15usize, 23, 31] {
            assert_eq!(
                compress_offset(WIN, 1, pos),
                WIN,
                "decode's selection base IS the ring base"
            );
            assert_eq!(
                compress_dst(kind, WIN, 1, pos),
                Some((WIN + pos / RATIO, 1))
            );
        }

        // **The loop's OWN half: the predicate its single branch rests on.** `compress_and_place`
        // writes `a_kv` only when `sel_base != persist_base`, and it tests THAT rather than the
        // phase — so the invariant is only true of the loop if the two bases differ at prefill and
        // coincide at decode. A review found the registry row claiming a property of the layer loop
        // while both cited tests exercised only the pure functions; this is the one line that ties
        // them together, and it is device-free.
        assert_ne!(
            compress_offset(WIN, seqlen, 0),
            WIN,
            "prefill: two destinations"
        );
        for pos in [15usize, 23, 31] {
            assert_eq!(compress_offset(WIN, 1, pos), WIN, "decode: one destination");
        }
        // ...and it must not be an accident of these numbers: the prefill bases differ because the
        // prompt length is not the window, so a prompt of exactly `WIN` tokens collapses them.
        // Asserted so a reader knows the branch takes the one-write path there too, correctly —
        // both destinations are row `WIN` and one memcpy serves both.
        assert_eq!(
            compress_offset(WIN, WIN, 0),
            WIN,
            "a prompt of exactly `window` tokens coincides"
        );

        // The registry row's "never the next free slot", on the only script shape that separates
        // the two rules: skip 23. The rules AGREE on every contiguous script, which is asserted
        // second so that "tidying" this into 15, 23, 31 fails here instead of passing vacuously.
        let rows = |script: &[usize]| -> (Vec<usize>, Vec<usize>) {
            let (mut positional, mut appending, mut next) = (Vec::new(), Vec::new(), WIN);
            for &pos in script {
                let (row, n) = compress_dst(kind, WIN, 1, pos).expect("a completing position");
                positional.push(row);
                appending.push(next);
                next += n;
            }
            (positional, appending)
        };
        let (pos_gap, app_gap) = rows(&[15, 31]);
        assert_eq!(
            (pos_gap.as_slice(), app_gap.as_slice()),
            (&[9, 11][..], &[8, 9][..])
        );
        assert_ne!(
            pos_gap, app_gap,
            "a skipped step MUST separate the two placers"
        );
        let (pos_c, app_c) = rows(&[7, 15, 23, 31]);
        assert_eq!(
            pos_c, app_c,
            "contiguous: the two rules agree, which is why the gap is needed"
        );
    }
}
