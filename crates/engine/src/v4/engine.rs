//! The V4 decode engine: the two device-resident rotary tables, each layer's persistent KV
//! and pooling state, and the per-token scratch — everything `forward.rs`, `attn.rs`,
//! `kvcompress.rs`, `moe.rs` and `decode.rs` drive.
//!
//! Ported from `old:src/f4gpu.rs`'s `F4Engine` under the same narrowings the other arms took:
//! no per-phase `Profile`, no probe API, no teacher-forcing hook. What remains is the decode
//! loop's minimum state, and each deferred field returns with the feature that reads it rather
//! than as dead weight now. The reference's `Profile` was 250 lines of doc over fourteen
//! accumulators and six HIP events; it measured a decomposition of a wall this tree has not
//! measured once, and a bucket that reads zero because nothing filled it is this repo's named
//! telemetry trap.
//!
//! **Every device buffer is allocated once, here.** A decode allocates nothing on the hot
//! path, which is `crate::glm::engine`'s rule and the reason both engines' `new` are long.

use super::geometry::{Dims, LayerKind};
use super::kvcompress::LayerCompressor;
use super::rope::{self, Params};
use super::select::positional_context_limit;
use crate::device::{DeviceBuf, as_le_bytes};
use crate::resident::Fp8Weight;
use crate::routed::{ExpertSlot, ResolvedBatch};
use crate::v4::pin::{CompressorWeights, V4Pin};
use anyhow::{Context, Result, ensure};
use rivoli_artifact::v4_config::V4Config;
use rivoli_backend::gpustream::HipStream;
use rivoli_backend::{ExpertDescF4, fill_u32};

/// Rows of the fixed-point MoE accumulator — ONE PER STREAM, which is what
/// `launch_moe_acc_drain`'s `rows` argument means and not one per expert. Residents
/// accumulate into row 0 on the compute stream and misses into row 1 on the miss stream, so
/// the two never contend for a cache line and there is no cross-stream join. Same value and
/// same reason as [`crate::glm::engine::MOE_ACC_ROWS`].
pub(super) const MOE_ACC_ROWS: usize = 2;

/// `-inf` as f32 bits, for [`fill_u32`].
///
/// **`score_state` must be `-inf`-initialised and NOT zeroed.** A never-written pooling slot
/// has to weigh `exp(-inf - m) == 0`; a zero weighs `exp(0 - m)` — a plausible number and a
/// wrong pooling window. Named rather than spelled at its call site because `fill_u32` takes a
/// `u32` and the defect is a `0` there.
pub(super) const NEG_INF_BITS: u32 = f32::NEG_INFINITY.to_bits();

/// Device argmax result bytes: one i32 index then one f32 value.
///
/// V4 decodes one row ([`super::ROWS`]), so unlike GLM's there is no per-row array here —
/// and no non-finite tag either. GLM's tag localises a NaN to a `(pos, layer)` by riding a
/// D2H the tail already pays; V4's equivalent would need a `launch_flag_nonfinite` per layer,
/// which is a diagnostic this arm has not needed yet and would be a one-line addition beside
/// the end-of-layer join if it ever does.
pub(super) const ARGMAX_BYTES: usize = 8;

/// Both rotary tables, device-resident, with the per-layer selection happening at exactly ONE
/// site.
///
/// **This is the enforcing construction the reference recorded as owed and handed back three
/// times.** A layer's rotary table reaches the kernels as a bare `*const f32`; the plain table
/// and the YaRN one have the same element type, the same stride and the same shape, so nothing
/// downstream can tell them apart, and substituting one for the other keeps the frequencies
/// plausible at every scale and the text fluent. The reference measured its numeric gate BLIND
/// to the swap at `ratio4/decode` (separation 8 bf16 codes against a resolvable floor of 64,
/// i.e. half an e4m3 step), so no tolerance anywhere would have caught a caller that got it
/// wrong. The answer is not a check — it is that there is only one place to get it wrong.
///
/// **The cache is content-addressed, not an arm per [`LayerKind`].** Keying on
/// `(theta, original_seq_len)` — the pair [`rope::for_layer`] moves together, and the only two
/// fields the model's two sets differ in — makes this a MEMO OVER that function rather than a
/// second copy of its decision. A `match kind { Plain => .., _ => .. }` accessor was written
/// first in the reference and rejected as "a second place to state the same fact, and
/// therefore a second place to state it wrongly".
pub(super) struct RopeTables {
    /// `(theta bits, original_seq_len) -> table`. At most two entries on any real config; a
    /// `Vec` because a two-entry linear scan beats a hash and the key half is `f32::to_bits`,
    /// which has no `Hash` worth reaching for.
    tables: Vec<((u32, usize), DeviceBuf)>,
    compressed: Params,
    rope_theta: f32,
    max_pos: usize,
}

impl RopeTables {
    /// `max_pos` is the context this decode was sized for; it enters no weight and only sizes
    /// the tables.
    fn new(cfg: &V4Config, max_pos: usize) -> Self {
        Self {
            // Built ONCE, from the config's own rotary fields. `Params` the type is what keeps
            // `compress_rope_theta` and `original_seq_len` travelling together.
            compressed: Params::compressed(cfg),
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
    pub(super) fn for_layer(&mut self, kind: LayerKind) -> Result<*const f32> {
        let p = rope::for_layer(self.compressed, self.rope_theta, kind);
        let key = (p.theta.to_bits(), p.original_seq_len);
        if let Some((_, buf)) = self.tables.iter().find(|(k, _)| *k == key) {
            return Ok(buf.ptr().cast());
        }
        // `rope::table` already produces the interleaved `(cos, sin)` layout
        // `launch_rope_adjacent` indexes, so there is no flattening step here — which is the
        // other half of "one site": a second interleave would be a second thing to get wrong.
        let flat = rope::table(p, self.max_pos);
        let mut buf = DeviceBuf::new(std::mem::size_of_val(flat.as_slice()))?;
        buf.copy_in_at(0, as_le_bytes(&flat))?;
        let ptr = buf.ptr().cast();
        self.tables.push((key, buf));
        Ok(ptr)
    }
}

/// One layer's persistent decode state.
pub(super) struct DeviceLayer {
    pub(super) kind: LayerKind,
    /// `[window + max_ctx/ratio, head_dim]` f32 — the ring FIRST, then the compressed region,
    /// contiguous and in that order.
    ///
    /// That order is what makes the reference's `compressor.kv_cache = self.kv_cache[:, win:]`
    /// a VIEW rather than a second buffer: decode attends the whole thing and the selection's
    /// compressed columns are `window + block`. `super::select`'s two coordinate systems are
    /// the reader's half of the same fact.
    pub(super) cache: DeviceBuf,
    /// Rows in [`Self::cache`]. Carried because `DeviceBuf` has no length and the placement
    /// indexes it by ROW, so the bound check at the write has nothing else to compare against.
    pub(super) cache_rows: usize,
    pub(super) comp: Option<LayerCompressor>,
}

impl DeviceLayer {
    fn new(
        cfg: &V4Config,
        kind: LayerKind,
        max_ctx: usize,
        max_m: usize,
        cw: Option<&CompressorWeights>,
    ) -> Result<Self> {
        // `max_ctx / ratio` is the reference's own sizing of `kv_cache[:, window_size:]`, and
        // 0 on a ratio-0 layer. `div_ceil` and not `/`: at `max_ctx = 13, ratio = 4` the block
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
    /// **The compressed region specifically**, which the reference asserted nowhere. It
    /// matters twice. A stale compressed row is attended BY POSITION on every later step,
    /// because this arm's compressed selection is the positional full prefix: leftover blocks
    /// from a previous prompt are weighted `exp(l - max)` and silently mixed in. And it is a
    /// premise of the measurement that made the decode placement rule invisible to attention
    /// output — a region reused across sequences holds stale rows rather than zeros, so the
    /// two placement rules would read different values.
    ///
    /// The ring half needs it for the same reason with a shorter tail: `super::select`'s
    /// window fill masks slots the sequence has not reached with `-1`, so unwritten ring slots
    /// are unread — but only while that mask is right, and a zeroed ring costs one `fill_u32`
    /// per layer per sequence.
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

/// Everything one V4 decode needs that does not vary between tokens.
pub struct V4Engine<'a> {
    pub(super) pin: V4Pin,
    pub(super) cfg: &'a V4Config,
    pub(super) dims: Dims,
    pub(super) rope: RopeTables,
    pub(super) layers: Vec<DeviceLayer>,
    /// Which layers this pin holds. `0..n_layers` for a whole-model decode; a shorter prefix
    /// is a golden comparison and [`V4Engine::new`] says so at startup.
    pub(super) range: std::ops::Range<usize>,
    pub(super) max_ctx: usize,
    /// Query rows every `[m, ..]` buffer is sized for — **`max_ctx - 1`**, the largest prompt
    /// this engine can be handed.
    ///
    /// V4's prefill attends the whole prompt in ONE `attention` call: both the ring seeding
    /// and the compressor's block pooling are whole-prompt by construction, so there is no
    /// `MAXROW`-shaped scratch here and no layer-major schedule to test. The reference sized
    /// this from the RUN's prompt because it had no server path; this tree's seam opens an
    /// engine before any request exists, so the ceiling is the only honest bound.
    ///
    /// **That makes the activation scratch O(`max_ctx`) and it is this arm's largest
    /// unbudgeted allocation** — `h` alone is `2 · max_m · hc_mult · hidden` f32 and the two
    /// `n_heads · head_dim` buffers are the same order. It is NOT charged to the partition;
    /// `crate::resident::weights_only_floor` carries what that omission means and why it is a
    /// change to what `--max-mem` MEANS rather than a line inside a port.
    pub(super) max_m: usize,
    /// The token ids of the step in flight. Held because a hash layer's `tid2eid` is indexed
    /// by TOKEN ID, and the MoE block does not otherwise see them — a caller that passed row
    /// counts alone would route every hash layer by the previous step's tokens, and the
    /// difference looks like ordinary routing variation.
    pub(super) step_ids: Vec<u32>,

    /// `h` and its double buffer. **Two and not one:** `launch_hc_post`'s contract is that `y`
    /// must not alias `residual` — both are `__restrict__`, and thread `i` writes `y[i]` while
    /// other threads still read every source copy of `residual`, with no barrier between them.
    /// So `hc_post` reads `h[cur]` and writes `h[1 - cur]`. An in-place residual expansion is
    /// the obvious thing to want and it is wrong twice over.
    pub(super) h: [DeviceBuf; 2],
    pub(super) cur: usize,

    /// `hc_pre`'s `y` — the `[m, dim]` tensor the norms and the sublayer see. NOT the
    /// residual, which is why the two sublayer norms may be in-place `launch_rmsnorm_batch`.
    pub(super) xw: DeviceBuf,
    /// The fp8-quantized copy of [`Self::xw`]. Separate because the ROUTER must see the
    /// UNQUANTIZED activation — `Gate.forward` is `linear(x.float(), weight.float())`, with no
    /// activation quantization anywhere — while every expert projection must see the quantized
    /// one. Quantizing first would feed the router e4m3 values and the error would look like
    /// ordinary routing variation.
    pub(super) xq: DeviceBuf,
    pub(super) post: DeviceBuf,
    pub(super) comb: DeviceBuf,
    /// `launch_hc_head_collapse`'s `[s, hc]` scratch gate vector.
    pub(super) head_pre: DeviceBuf,
    /// The sublayer output `hc_post` consumes — attention's output, then the MoE's.
    pub(super) sub: DeviceBuf,

    pub(super) a_qr: DeviceBuf,
    pub(super) a_qrq: DeviceBuf,
    pub(super) a_q: DeviceBuf,
    /// `[m + m/ratio, head_dim] + q_lora_rank` f32 — **not** `[m, head_dim]`.
    ///
    /// At prefill the attend kernel reads `cat([kv, kv_compress])` and the selection indexes
    /// that concatenation as ONE space, so the compressor's blocks live in this buffer's tail;
    /// it is sized at the TIGHTEST ratio because one buffer serves every layer class. The
    /// `+ q_lora_rank` is the fused decode GEMV's one output row (`head_dim + q_lora_rank`
    /// floats at the base) — decode touches no other row, and the compressed-tail sizing
    /// happens to cover the spill for `max_m >= 2` but not at `max_m == 1`, so the slack is
    /// explicit rather than counted out of rows.
    pub(super) a_kv: DeviceBuf,
    pub(super) a_o: DeviceBuf,
    pub(super) a_y: DeviceBuf,
    pub(super) idx_host: Vec<i32>,
    pub(super) idx_dev: DeviceBuf,

    pub(super) gate_logits: DeviceBuf,
    pub(super) gl_host: Vec<u8>,
    pub(super) scores: Vec<f32>,
    pub(super) choice: Vec<f32>,
    /// `n_experts` zeros, for a HASH layer's routing call. **Not an empty slice:**
    /// `route_into` computes `choice` by zipping `scores` with `bias`, so an empty `bias`
    /// leaves `choice` holding the PREVIOUS layer's values and the top-k then selects on them.
    /// A hash layer discards that selection, so it would be harmless today and a landmine
    /// tomorrow; zeros make `choice == scores`, which is what a bias-free gate means.
    pub(super) zero_bias: Vec<f32>,
    pub(super) sel: Vec<usize>,
    /// `[n_experts]` f32 indexed by ABSOLUTE expert id, zero for every expert this token did
    /// not route to. The kernel SKIPS a zero weight, so the zeros are correctness and not
    /// thrift. This is the scatter `moe.rs`'s launch-order gather reads from.
    pub(super) wexpert_host: Vec<f32>,
    /// `[n_experts]` f32 in LAUNCH order — the descriptor index space the device buffers
    /// share (residents first, misses after). Entries past this row's selection are never
    /// written after construction, so they stay zero, and a zero weight makes a
    /// wrongly-computed launch range write `h = 0` instead of plausible values — the same
    /// defence the null descriptors give the pointer side.
    pub(super) wexpert_launch: Vec<f32>,
    pub(super) wexpert: DeviceBuf,
    pub(super) descs_host: Vec<ExpertDescF4>,
    pub(super) descs: DeviceBuf,
    /// The pool's answer for the current layer: slots + tickets, in selection order.
    pub(super) resolved: ResolvedBatch,
    pub(super) moe_acc: DeviceBuf,
    /// `[n_experts, inter]` f32 — indexed by the same launch-order descriptor index as
    /// `descs`/`wexpert` per the launcher's contract, so it is sized for the DESCRIPTOR count
    /// and not for one range's `e_count`. A caller that read these as range-relative would run
    /// off the end the first time it passed `e_start > 0`, which is the first thing a
    /// two-stream pipeline does.
    pub(super) moe_h: DeviceBuf,
    pub(super) sh_g: DeviceBuf,
    pub(super) sh_u: DeviceBuf,

    pub(super) head_x: DeviceBuf,
    pub(super) logits: DeviceBuf,
    pub(super) argmax_dev: DeviceBuf,
    pub(super) argmax_host: Vec<u8>,

    /// Resident experts' partials run here, concurrently with the fetch stream's loads.
    pub(super) compute_stream: HipStream,
    /// Experts whose bytes are still arriving launch HERE, not on the compute stream — a
    /// stream is FIFO, so a wait enqueued on the compute stream is only reached after the
    /// residents finish, putting the GPU's wake latency on the critical path.
    pub(super) miss_stream: HipStream,
    /// The shared-expert chain's stream: off the null stream so the routed path's blocking
    /// H2D copies stop ordering behind it. See `moe.rs`, which owns the joins.
    pub(super) shared_stream: HipStream,
    /// Pool counters at the last [`V4Engine::reset`], so a second `generate` on one engine
    /// reports ITS lookups and not the first run's folded in. Captured once in the
    /// constructor and left there, a cumulative hit rate reads as a residency change.
    pub(super) hits0: u64,
    pub(super) misses0: u64,
}

impl<'a> V4Engine<'a> {
    /// Build the engine over `pin`: allocate every per-token buffer and every layer's
    /// persistent state ONCE, so the decode loop allocates nothing.
    ///
    /// `max_ctx` is the run's hard ceiling AND what the whole-prompt pass is sized for — see
    /// [`V4Engine::max_m`] for why this arm has no smaller number to use.
    pub fn new(pin: V4Pin, cfg: &'a V4Config, max_ctx: usize) -> Result<Self> {
        let range = pin.range();
        // **A decode has to start at layer 0.** `V4Pin::build` deliberately does not enforce
        // this — which layers a file holds is a property of the LOADER, and refusing a partial
        // artifact there made every one but the first unloadable — but a forward pass has no
        // residual stream to enter at layer 3. `V4Pin::layer` takes ABSOLUTE ids, so a pin
        // over 3..6 answers every lookup correctly and the arithmetic is a different model's,
        // with nothing anywhere to notice.
        ensure!(
            range.start == 0,
            "this artifact holds layers [{}, {}) and a decode must start at layer 0 — there is \
             no residual stream to enter the model at layer {}. Convert from layer 0.",
            range.start,
            range.end,
            range.start
        );
        // The largest prompt this engine can be handed, and therefore what every `[m, ..]`
        // buffer is sized for. `max_ctx - 1` and not `max_ctx`: a run that spent the whole
        // ceiling on its prompt has no room to generate, so `Emit` would stop before the
        // first token.
        ensure!(
            max_ctx >= 2,
            "--ctx {max_ctx} leaves no room for a prompt and a generated token"
        );
        let max_m = max_ctx - 1;
        if range.end < cfg.n_layers {
            // Not refused: a short prefix IS what a per-layer golden comparison is for. But
            // three layers of a 43-layer model is not a decode, and calling it one is a
            // reading the reference had to retract twice — so it says so, loudly, once.
            tracing::warn!(
                "PARTIAL ARTIFACT: layers [0, {}) of {}. This is NOT the model — the logits \
                 are a {}-layer prefix's, and any text decoded from them is meaningless.",
                range.end,
                cfg.n_layers,
                range.end
            );
        }
        // Checked a SECOND time here: `Engine::open` refuses this at the door, before the pin
        // reads nine gigabytes, and a hand-built caller never passed that door.
        check_context(cfg, max_ctx)?;

        let dims = Dims::from_config(cfg).context("v4 attention dims from the artifact")?;
        let (dim, hc, hd) = (cfg.hidden, cfg.hc_mult, cfg.head_dim);
        let (nhd, n_desc) = (cfg.n_heads * hd, cfg.n_experts);
        let f32s = |n: usize| DeviceBuf::new(n * size_of::<f32>());
        // The widest selection any step can ask for: prefill is `m` rows of
        // `min(m, win) + m/ratio`, decode is one row of `win + max_ctx/ratio`. Taken as the
        // bound of BOTH rather than assumed to be one of them, and at the tightest ratio
        // because one buffer serves every layer class.
        let tightest = LayerKind::Overlap.compressor_ratio().unwrap_or(1);
        let idx_cols = cfg.sliding_window + max_ctx.div_ceil(tightest);
        let mut layers = Vec::with_capacity(range.len());
        for l in range.clone() {
            let kind =
                LayerKind::from_config(cfg, l).with_context(|| format!("classifying layer {l}"))?;
            // The pin's compressor for this layer, so `LayerCompressor` can narrow its
            // weights. Read through `V4Pin::layer`, which applies the artifact-order offset
            // exactly once, and passed rather than looked up again inside — the two answers
            // (`Geom::attention` and the pin's `Option`) are ASSERTED to agree there.
            let cw = pin.layer(l)?.compressor;
            layers.push(DeviceLayer::new(cfg, kind, max_ctx, max_m, cw.as_ref())?);
        }

        let mut e = Self {
            dims,
            rope: RopeTables::new(cfg, max_ctx),
            layers,
            range,
            max_ctx,
            max_m,
            step_ids: Vec::with_capacity(max_m),
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
            a_kv: f32s((max_m + max_m.div_ceil(tightest)) * hd + cfg.q_lora_rank)?,
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
            resolved: ResolvedBatch::default(),
            // `MOE_ACC_ROWS` rows of ONE token's hidden width: the routed experts are launched
            // per token, because the FP4 kernel refuses `nrow != 1`.
            moe_acc: DeviceBuf::new(MOE_ACC_ROWS * dim * size_of::<u64>())?,
            moe_h: f32s(n_desc * cfg.moe_inter)?,
            sh_g: f32s(max_m * cfg.moe_inter)?,
            sh_u: f32s(max_m * cfg.moe_inter)?,
            head_x: f32s(dim)?,
            logits: f32s(cfg.vocab)?,
            argmax_dev: DeviceBuf::new(ARGMAX_BYTES)?,
            argmax_host: Vec::new(),
            compute_stream: HipStream::compute()?,
            miss_stream: HipStream::miss()?,
            shared_stream: HipStream::shared()?,
            hits0: pin.routed.hits(),
            misses0: pin.routed.misses(),
            pin,
            cfg,
        };
        // The pool is single-format and this arm builds `ExpertDescF4` unconditionally, so a
        // `.vq3` or `.i4` slot reaching the dispatch would be found at exactly the right
        // addresses and decoded with the wrong arithmetic: `.f4` and `.i4` tile IDENTICALLY
        // for a quarter of all `i_dim`, this model's included, so no length, offset or
        // descriptor check downstream could see it. `V4Pin::build` opens the set as `F4`, so
        // this cannot fire today — it is here because the consequence if a second FP4
        // container is ever paired with `.f4` is not a fault.
        ensure!(
            e.pin.routed.fmt() == rivoli_artifact::format::RoutedFmt::F4,
            "the V4 pool resolved a {:?} set and this arm dispatches ExpertDescF4",
            e.pin.routed.fmt()
        );
        e.reset()?;
        Ok(e)
    }

    /// Clear every layer's persistent state and the accumulator. **Between sequences, not
    /// between tokens.**
    pub(super) fn reset(&mut self) -> Result<()> {
        for st in &mut self.layers {
            st.reset(self.cfg)?;
        }
        // `launch_moe_acc_drain` resets `acc` to zero as it converts, so this is only the
        // FIRST use's initialisation — but `hipMalloc` does not zero, and an accumulator that
        // starts at garbage adds a fixed-point garbage vector to layer 0's first token and
        // nothing else. One `fill_u32` per sequence; GLM pays the same one for the same
        // reason.
        // SAFETY: allocated at exactly this size, nothing in flight.
        unsafe {
            fill_u32(
                self.moe_acc.ptr_mut(),
                0,
                MOE_ACC_ROWS * self.cfg.hidden * size_of::<u64>(),
            )?;
        }
        self.cur = 0;
        self.hits0 = self.pin.routed.hits();
        self.misses0 = self.pin.routed.misses();
        Ok(())
    }

    /// The KV ceiling this engine allocated for, in tokens — see `Engine::max_ctx`, whose only
    /// job is to hand this number to a caller that must refuse an over-long request before
    /// decoding it.
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub(super) fn hits(&self) -> u64 {
        self.pin.routed.hits()
    }

    pub(super) fn misses(&self) -> u64 {
        self.pin.routed.misses()
    }
}

/// Refuse a context at or past the point where a POSITIONAL compressed selection stops
/// agreeing with the trained-in indexer.
///
/// Called at the DOOR — `Engine::open`'s V4 arm runs this before `V4Pin::build` reads nine
/// gigabytes — and again inside [`V4Engine::new`], which is not redundant: a caller that
/// constructs the engine directly never passed the door.
/// [`super::select::Sel::shape`] refuses it a third time at the call, per query, which is the
/// only one of the three that sees a hand-built selection.
///
/// **`<`, not `<=`.** The selection refuses when the block count EXCEEDS `index_topk`, i.e. at
/// `end_pos >= limit`, and `forward` admits `start_pos + m == max_ctx`. So `max_ctx == limit`
/// would pass here and still refuse at the last position. The decode loop's own accounting
/// never reaches it, which is exactly the kind of slack that makes a boundary bug invisible
/// until someone drives `forward` directly.
///
/// **The shipped `--ctx` default does not satisfy this**, and that is deliberate rather than a
/// bug in either: at `index_topk = 512` the ceiling is 2052, so `rivoli V4DIR --bench 4` on
/// the CLI's 4096 default is REFUSED with this message, which names the flag and the number.
/// Clamping instead would contradict `rivoli_core::legality`'s `Support` on `--ctx` — a user
/// who asked for a 4096-token conversation and got 2048 would have it silently truncated —
/// and the flag is the knob, so the refusal is the honest answer until the indexer runs.
pub fn check_context(cfg: &V4Config, max_ctx: usize) -> Result<()> {
    let limit = positional_context_limit(cfg.index_topk);
    let indexed = (0..cfg.n_layers)
        .filter(|&l| cfg.layer_has_indexer(l).unwrap_or(false))
        .count();
    ensure!(
        max_ctx < limit,
        "--ctx {max_ctx} reaches {limit}, past which the compressed block set is decided by \
         the lightning indexer's SCORES. This arm selects blocks POSITIONALLY, which agrees \
         with the indexer on the block SET only below that length; above it, keeping the \
         first {} blocks keeps the OLDEST and silently stops attending everything newer, on \
         the {indexed} indexed layer(s). Pass --ctx below {limit}.",
        cfg.index_topk,
    );
    Ok(())
}

/// A `groups = 1` fp8-e4m3 GEMV with a bf16-rounded output — every projection on this arm
/// except the grouped output one.
///
/// One helper for the attention block's nine and the shared expert's three, because they
/// differ ONLY in `(weight, extents, destination)` and spelling the other six arguments twelve
/// times is how `m`/`k`/`n_out` get transposed in one copy and not the others. The block size
/// is the config's, checked equal to [`super::geometry::FP8_BLOCK`] at pin build, so it is
/// spelled once here rather than read off each weight.
///
/// # Safety
/// `x` holds `m * k` **already fp8-quantized** f32, `w` is a pin placement, `out` holds
/// `m * n_out` f32, none aliasing another, all live until `stream` completes.
pub(super) unsafe fn gemv_fp8(
    x: *const f32,
    w: Fp8Weight,
    m: usize,
    (n_out, k): (usize, usize),
    out: *mut f32,
    stream: *mut std::ffi::c_void,
) -> Result<()> {
    // SAFETY: forwarded verbatim from this function's own contract.
    unsafe {
        rivoli_backend::launch_gemv_fp8_bf16(
            x,
            w.packed,
            w.scale,
            m,
            n_out,
            k,
            super::geometry::FP8_BLOCK,
            1,
            out,
            stream,
        )
    }
}

/// Build one expert's `ExpertDescF4` from its resolved pool slot.
///
/// Six byte addresses in, six byte addresses out — which is the whole reason [`ExpertSlot`]
/// stopped carrying a typed `scales` pointer. `.f4`'s e8m0 scales are ONE byte, `.i4`'s are
/// f32 and `.vq3`'s bf16, and GLM's builder casts to `*const u16` for its two formats; this
/// one casts nothing, because `ExpertDescF4` already says `*const u8`.
///
/// What it cannot check: `.f4` and `.i4` tile IDENTICALLY for 25% of all `i_dim`, both models'
/// dimensions included, so a slot resolved through the wrong format's offsets finds every
/// projection at exactly the right address and then decodes e2m1 nibbles against the wrong
/// scale grid. The header magic and the descriptor TYPE are the entire separation — the type
/// is this function's return and the magic is `ExpertSet::open_routed`'s.
pub(super) fn desc_of_f4(s: &ExpertSlot) -> ExpertDescF4 {
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
/// The descriptor table is written in LAUNCH order and sized `n_experts`, so most of its
/// entries sit past any one token's selection and no launch names them. Filling those with
/// nulls rather than with a copy of some resolved expert is the difference between a fault and
/// a plausible wrong weight the day a range is computed wrongly.
pub(super) fn desc_never_read() -> ExpertDescF4 {
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
