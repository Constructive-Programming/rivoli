//! The K3 decode engine: every per-layer persistent state (KDA recurrent slabs and conv
//! rings, MLA caches), the one-token residual arena, and the per-token scratch — everything
//! `forward.rs` and `decode.rs` drive. `geometry.rs` owns every width it reads; `pin.rs`
//! owns every weight.
//!
//! **Every device buffer is allocated once, here.** A decode allocates nothing on the hot
//! path — `crate::glm::engine`'s rule, held by every arm.
//!
//! # This model's twelve traps, and which construction owns each
//!
//! `k3:docs/reference/k3-architecture.md` §10 — each runs cleanly and produces a wrong
//! model. "kernel-internal" means the composed kernel takes RAW inputs, so the engine
//! CANNOT commit the trap; those are priced by the kernel suite against the anchor's
//! fixtures, not re-checked here.
//!
//! | trap | where |
//! |---|---|
//! | 1. `A_log` per HEAD (96 of a shipped `[128]`), `dt_bias` per channel (12288) | `pin::place_kda` asserts both shapes; they reach the kernel as separate pointers |
//! | 2. recurrence order: decay → read u → delta write → read o | kernel-internal (`launch_gated_delta_recurrent_f32`) |
//! | 3. `a*(z + dt_bias)`, not `a*z + dt_bias` | kernel-internal — `z` and `dt_bias` enter RAW, the engine never pre-adds |
//! | 4. `gate_lb` MULTIPLIES the sigmoid | kernel-internal; guard 1006 refuses a bound outside `[-5, 0)`, and the engine passes the config's value verbatim |
//! | 5. SiLU FUSED into the conv output | `forward.rs` launches conv → recurrence with no activation between — there is no separate silu to misplace |
//! | 6. L2Norm after conv, q and k only, eps on the SUM | kernel-internal — the recurrence takes q/k PRE-norm and `v` has no normed path by signature |
//! | 7. `q * d^-0.5` after the L2Norm | kernel-internal — the launcher HAS no scale parameter |
//! | 8. MLA softmax scale over the FULL 192 | `geometry::Dims::mla_scale`, tested deviceless |
//! | 9. the 64 unrotated rope dims are STILL SCORED | the cache row is `Dims::q_head` = 192 wide — the rope slot is appended into every row, so dropping the term is unrepresentable |
//! | 10. MLA gates with NO norm; KDA norms THEN gates | two different launcher selections in `forward.rs` (`launch_sigmoid_gate` vs `launch_rmsnorm_gate_heads_f32`) |
//! | 11. combining weights from the UNBIASED sigmoid | the ONE call site, `forward.rs::moe_ffn`'s `route` — `scores` and `choice` are both `Vec<f32>`, so the signature does NOT argue (a claim this table made falsely until review 2026-08-16); `combine_weights` itself is tested deviceless |
//! | 12. RMSNorm the AGGREGATE; shared expert AFTER the up-projection, unweighted | `forward.rs::moe_ffn`'s drain_to → norm → up → shared order |
//!
//! # The MLA cache is masked-full-width, and that is a sizing decision
//!
//! `launch_mha_attend` reads `k` as `[heads][kv][d]` with the per-head stride derived from
//! `kv` — so a cache allocated at `max_ctx` rows per head can only be scored at
//! `kv == max_ctx`. This arm therefore attends every allocated row on every step behind an
//! additive `[max_ctx]` mask: unreached positions carry `-inf` and weigh
//! `exp(-inf - m) == 0`. The waste is score arithmetic over unreached rows; the alternative
//! is a compacting append per step. Priced when a benchmark exists; `geometry::check_context`
//! is where the same fact becomes a `--ctx` ceiling.

use super::geometry::{Dims, check_context};
use super::pin::{Attn, K3Pin};
use crate::device::DeviceBuf;
use crate::routed::ResolvedBatch;
use crate::telemetry::Phases;
use anyhow::{Context as _, Result, ensure};
use rivoli_artifact::k3_config::K3TextConfig;
use rivoli_backend::gpustream::HipStream;
use rivoli_backend::{ExpertDescF4, fill_u32};

/// `-inf` as f32 bits, for [`fill_u32`] — the mask's "position not reached yet". A zeroed
/// mask instead weighs every unreached row `exp(0 - m)`: plausible, wrong, and exactly the
/// defect class `crate::v4::engine::NEG_INF_BITS` records for its pooling slots.
pub(super) const NEG_INF_BITS: u32 = f32::NEG_INFINITY.to_bits();

/// Device argmax result bytes: one i32 index then one f32 value.
pub(super) const ARGMAX_BYTES: usize = 8;

/// Rows of the fixed-point MoE accumulator — one per STREAM (residents on the compute
/// stream, misses on the miss stream), same value and same no-cross-stream-join reason as
/// `crate::v4::engine::MOE_ACC_ROWS`, and beside the engine like both siblings'.
pub(super) const MOE_ACC_ROWS: usize = 2;

/// One layer's persistent decode state — the enum mirrors `pin::Attn` so a KDA layer
/// cannot be driven with an MLA cache or vice versa: the dispatch in `forward.rs` matches
/// BOTH and the compiler refuses the cross pairings.
pub(super) enum LayerState {
    Kda {
        /// `[heads][head_dim][head_dim]` f32, KEY-major (`[key][value]` — fla's own buffer
        /// is `[V][K]` and this tree declines that layout for coalescing; the kernel doc
        /// states the axis order, and the anchor's `KdaStateLayout` defect is the priced
        /// consequence of getting it backwards). Updated in place by the recurrence.
        state: DeviceBuf,
        /// Three `[ch][taps]` conv windows, advanced in place by the conv launcher. They
        /// hold PRE-conv, PRE-SiLU inputs — the launcher's contract, not a choice here.
        win_q: DeviceBuf,
        win_k: DeviceBuf,
        win_v: DeviceBuf,
    },
    Mla {
        /// `[heads][max_ctx][q_head]` f32 — nope ‖ rope per row (trap 9 lives in this
        /// width) — and `[heads][max_ctx][v_head]`.
        kc: DeviceBuf,
        vc: DeviceBuf,
    },
}

/// Everything one K3 decode needs that does not vary between tokens.
pub struct K3Engine<'a> {
    pub(super) pin: K3Pin,
    pub(super) cfg: &'a K3TextConfig,
    pub(super) d: Dims,
    pub(super) layers: Vec<LayerState>,
    pub(super) max_ctx: usize,

    /// The AttnRes arena: `[res_blocks + 1][hidden]` f32, ONE token wide — snapshots at
    /// rows `0..stack`, the prefix sum always at row `stack` (`state.rs`'s representation
    /// argument). Sequential prefill is what keeps this one token wide; the module header
    /// of `crate::k3` carries that deviation.
    pub(super) arena: DeviceBuf,
    /// The fold output / current hidden state `h` — distinct from the prefix row, because
    /// §3's loop REPLACES `h` at each fold while the residual chain lives in the arena.
    pub(super) hbuf: DeviceBuf,
    /// The normed sublayer input. Both gates read THIS (`k3:docs` §4 step 9, §5): "the
    /// layer input x" is the normed vector the projections consumed, not the pre-norm sum.
    pub(super) xn: DeviceBuf,
    /// The sublayer output at hidden width (attention's, then the FFN's).
    pub(super) sub: DeviceBuf,
    /// A second hidden-width temporary: the shared MLP's down-projection lands here so its
    /// add onto [`Self::sub`] is a `vadd` of two distinct buffers.
    pub(super) tmp_h: DeviceBuf,

    // KDA scratch, all `[kda_ch]` unless noted.
    pub(super) kq: DeviceBuf,
    pub(super) kk: DeviceBuf,
    pub(super) kv: DeviceBuf,
    pub(super) kqc: DeviceBuf,
    pub(super) kkc: DeviceBuf,
    pub(super) kvc: DeviceBuf,
    /// `f_a`'s `[head_dim]` intermediate — one shared rank-`head_dim` pair feeds all heads.
    pub(super) f_mid: DeviceBuf,
    /// `z = f_b(f_a(x))`, the RAW decay input the kernel finishes (traps 3/4 stay inside).
    pub(super) z: DeviceBuf,
    /// `[kda_heads]` pre-sigmoid beta.
    pub(super) beta: DeviceBuf,
    /// The output gate projection — `[kda_ch]` on KDA layers, `[heads * v_head]` on MLA
    /// layers; the two widths are equal on this checkpoint (both 12288) but the buffer is
    /// sized at the max so a config where they differ still fits.
    pub(super) gate: DeviceBuf,
    /// The recurrence output, then (separate buffer) the normed-and-gated head output.
    pub(super) ko: DeviceBuf,
    pub(super) kon: DeviceBuf,

    // MLA scratch. The LoRA norms are OUT-OF-PLACE (`launch_rmsnorm_single`'s shape), so
    // each normed vector has its own buffer rather than a may-alias assumption nobody
    // verified against the kernel.
    pub(super) qa: DeviceBuf,
    pub(super) qan: DeviceBuf,
    pub(super) qb: DeviceBuf,
    pub(super) kva: DeviceBuf,
    pub(super) kvan: DeviceBuf,
    pub(super) kvb: DeviceBuf,
    pub(super) attn: DeviceBuf,
    /// `[max_ctx]` additive mask shared by every MLA layer: position `p` flips to 0 when
    /// the token at `p` is fed. One 4-byte H2D per step, not per layer.
    pub(super) mask: DeviceBuf,

    // MoE scratch.
    pub(super) gate_logits: DeviceBuf,
    pub(super) gl_host: Vec<u8>,
    pub(super) scores: Vec<f32>,
    pub(super) choice: Vec<f32>,
    pub(super) sel: Vec<usize>,
    /// `[n_experts]` by ABSOLUTE id (the `combine_weights` scatter), then in LAUNCH order.
    pub(super) wexpert_host: Vec<f32>,
    pub(super) wexpert_launch: Vec<f32>,
    pub(super) wexpert: DeviceBuf,
    pub(super) descs_host: Vec<ExpertDescF4>,
    pub(super) descs: DeviceBuf,
    pub(super) resolved: ResolvedBatch,
    /// Selection indices reordered residents-first — the launch order. A reused scratch so
    /// the hot path allocates nothing.
    pub(super) launch_idx: Vec<usize>,
    /// The 7168 -> 3584 latent the experts read (plain f32 — K3's fp4 pair quantizes no
    /// activation; that is the V4-only `act_quant_f8` step, absent by kernel contract).
    pub(super) z_lat: DeviceBuf,
    /// `[n_experts][moe_inter]` staging, indexed by DESCRIPTOR like the weights.
    pub(super) moe_h: DeviceBuf,
    /// `[MOE_ACC_ROWS][latent]` u64 fixed-point accumulator — LATENT wide, not hidden:
    /// passing `hidden` here is the drain's documented last-row overrun.
    pub(super) moe_acc: DeviceBuf,
    /// The drained, then normed, aggregate at latent width.
    pub(super) latent: DeviceBuf,
    /// Dense/shared MLP gate & up staging, sized `max(dense_inter, shared_inter)`.
    pub(super) mg: DeviceBuf,
    pub(super) mu: DeviceBuf,

    /// Resident experts' partials run here, overlapping the fetch stream's loads; misses
    /// launch on their own stream behind their tickets — `crate::v4::moe`'s measured order.
    pub(super) compute_stream: HipStream,
    pub(super) miss_stream: HipStream,
    /// Pool counters at the last [`K3Engine::reset`], so a second `generate` reports ITS
    /// lookups (`crate::v4::engine`'s rule).
    pub(super) hits0: u64,
    pub(super) misses0: u64,

    pub(super) logits: DeviceBuf,
    pub(super) argmax_out: DeviceBuf,
    pub(super) argmax_bytes: Vec<u8>,
    /// Whether any forward has completed — [`K3Engine::logits`]'s guard, on Muse Glimmer's
    /// argument: before the first step the buffer is whatever `hipMalloc` handed back, and
    /// a garbage-but-plausible logit vector is exactly what a readback consumer exists to
    /// catch.
    pub(super) stepped: bool,
    /// Decode-thread phase spans — `forward.rs`/`decode.rs` stamp them around this arm's
    /// existing joins. Unexercised on a device until a K3 checkpoint lands; the stamps
    /// stand so the first real decode profiles without a second edit.
    pub(super) prof: Phases,
}

impl<'a> K3Engine<'a> {
    /// Build the engine over `pin`: allocate every buffer and every layer's persistent
    /// state ONCE.
    pub fn new(pin: K3Pin, cfg: &'a K3TextConfig, max_ctx: usize) -> Result<Self> {
        // Checked a second time here (the seam checks at the door): a hand-built caller
        // never passed the door, and the mask/cache sizing below assumes the ceiling.
        check_context(max_ctx)?;
        // The widths the PIN was placed under, not a re-derivation — `K3Pin::build` already
        // ran `Dims::from_config` and its refusals.
        let d = pin.d;
        let la = &cfg.linear_attn_config;
        // The `.f4` range must START at the dense prefix's end: the resident file of a
        // partial artifact only covers `0..first_dense ∪ range`, and a forward pass has no
        // residual stream to skip a hole. A short END is a golden-comparison prefix.
        let moe = pin.moe_layers();
        ensure!(
            moe.start == cfg.first_k_dense_replace,
            "this artifact's experts start at layer {} but the dense prefix ends at {} — \
             the layers between have no weights anywhere. Convert from layer {}.",
            moe.start,
            cfg.first_k_dense_replace,
            cfg.first_k_dense_replace,
        );
        let n_layers = pin.layers();
        if n_layers < cfg.n_layers {
            tracing::warn!(
                "PARTIAL ARTIFACT: layers [0, {n_layers}) of {}. This is NOT the model — \
                 the logits are a prefix's, and any text decoded from them is meaningless.",
                cfg.n_layers
            );
        }
        let f32s = |n: usize| DeviceBuf::new(n * size_of::<f32>());
        let zeroed = |n: usize| -> Result<DeviceBuf> {
            let mut b = DeviceBuf::new(n * size_of::<f32>())?;
            // SAFETY: the fill is the allocation's own byte count, one line above.
            unsafe { fill_u32(b.ptr_mut(), 0, n * size_of::<f32>())? };
            Ok(b)
        };
        let (hid, ch, ne) = (cfg.hidden, d.kda_ch, cfg.n_experts);
        let mut layers = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            layers.push(match pin.layer(l)?.attn {
                Attn::Kda(_) => LayerState::Kda {
                    state: f32s(la.num_heads * la.head_dim * la.head_dim)?,
                    win_q: f32s(ch * la.short_conv_kernel_size)?,
                    win_k: f32s(ch * la.short_conv_kernel_size)?,
                    win_v: f32s(ch * la.short_conv_kernel_size)?,
                },
                // ZEROED at allocation, once — hipMalloc hands back recycled in-process
                // memory, and the additive mask only makes an unwritten row unread when its
                // garbage is FINITE: a NaN/inf K-row poisons pass 2's softmax denominator
                // (`fmaxf` skips the NaN, `expf(NaN - m)` does not), and `0.0 * inf` in the
                // V mix is NaN at weight zero. Review 2026-08-16; the reset path still
                // skips re-zeroing, because rows a PREVIOUS sequence wrote are finite and
                // the mask genuinely does the rest.
                Attn::Mla(_) => LayerState::Mla {
                    kc: zeroed(cfg.n_heads * max_ctx * d.q_head)?,
                    vc: zeroed(cfg.n_heads * max_ctx * cfg.v_head_dim)?,
                },
            });
        }
        let mut e = Self {
            d,
            layers,
            max_ctx,
            arena: f32s((d.res_blocks + 1) * hid)?,
            hbuf: f32s(hid)?,
            xn: f32s(hid)?,
            sub: f32s(hid)?,
            tmp_h: f32s(hid)?,
            kq: f32s(ch)?,
            kk: f32s(ch)?,
            kv: f32s(ch)?,
            kqc: f32s(ch)?,
            kkc: f32s(ch)?,
            kvc: f32s(ch)?,
            f_mid: f32s(la.head_dim)?,
            z: f32s(ch)?,
            beta: f32s(la.num_heads)?,
            gate: f32s(ch.max(cfg.n_heads * cfg.v_head_dim))?,
            ko: f32s(ch)?,
            kon: f32s(ch)?,
            qa: f32s(cfg.q_lora_rank)?,
            qan: f32s(cfg.q_lora_rank)?,
            qb: f32s(cfg.n_heads * d.q_head)?,
            kva: f32s(d.kv_a_out)?,
            kvan: f32s(cfg.kv_lora_rank)?,
            kvb: f32s(cfg.n_heads * d.kv_b_head)?,
            attn: f32s(cfg.n_heads * cfg.v_head_dim)?,
            mask: f32s(max_ctx)?,
            gate_logits: f32s(ne)?,
            gl_host: Vec::new(),
            scores: vec![0.0; ne],
            choice: vec![0.0; ne],
            sel: Vec::with_capacity(cfg.top_k),
            wexpert_host: vec![0.0; ne],
            wexpert_launch: vec![0.0; ne],
            wexpert: f32s(ne)?,
            descs_host: vec![ExpertDescF4::null(); ne],
            descs: DeviceBuf::new(ne * size_of::<ExpertDescF4>())?,
            resolved: ResolvedBatch::default(),
            launch_idx: Vec::with_capacity(cfg.top_k),
            z_lat: f32s(cfg.expert_in)?,
            moe_h: f32s(ne * cfg.moe_inter)?,
            moe_acc: DeviceBuf::new(MOE_ACC_ROWS * cfg.expert_in * size_of::<u64>())?,
            latent: f32s(cfg.expert_in)?,
            mg: f32s(cfg.dense_inter.max(d.shared_inter))?,
            mu: f32s(cfg.dense_inter.max(d.shared_inter))?,
            compute_stream: HipStream::compute()?,
            miss_stream: HipStream::miss()?,
            // Zeros, not the pool's counters: `reset` below is the one author of the
            // baseline, and a second copy of that read here was a jscpd clone of the V4
            // constructor besides.
            hits0: 0,
            misses0: 0,
            logits: f32s(cfg.vocab)?,
            argmax_out: DeviceBuf::new(ARGMAX_BYTES)?,
            argmax_bytes: Vec::new(),
            stepped: false,
            prof: Phases::default(),
            pin,
            cfg,
        };
        e.reset()?;
        Ok(e)
    }

    /// Clear everything a new sequence must not inherit. **Between sequences, not between
    /// tokens** — the KDA state and conv windows are the sequence
    /// (`k3:docs/reference/k3-architecture.md` §4: zeroed once per sequence, never between
    /// steps), and a stale state is a different conversation's memory decoded fluently.
    pub(super) fn reset(&mut self) -> Result<()> {
        let la = &self.cfg.linear_attn_config;
        let state_bytes = la.num_heads * la.head_dim * la.head_dim * size_of::<f32>();
        let win_bytes = self.d.kda_ch * la.short_conv_kernel_size * size_of::<f32>();
        for st in &mut self.layers {
            // SAFETY: each fill length is the byte count the buffer was allocated with in
            // `new`, derived from the same dims; reset runs at construction and between
            // sequences, with no kernel in flight.
            unsafe {
                match st {
                    LayerState::Kda {
                        state,
                        win_q,
                        win_k,
                        win_v,
                    } => {
                        fill_u32(state.ptr_mut(), 0, state_bytes)?;
                        for w in [win_q, win_k, win_v] {
                            fill_u32(w.ptr_mut(), 0, win_bytes)?;
                        }
                    }
                    // The caches need no RE-zeroing: they were zeroed once at allocation
                    // (see `zeroed` — finite garbage is what the mask's weight-0 argument
                    // requires), rows a previous sequence wrote are finite, and re-zeroing
                    // gigabytes per sequence is a cost with no defect behind it. The MASK
                    // is the thing that must reset.
                    LayerState::Mla { .. } => {}
                }
            }
        }
        // SAFETY: both fills are the allocated byte counts, same derivation as `new`.
        unsafe {
            fill_u32(
                self.mask.ptr_mut(),
                NEG_INF_BITS,
                self.max_ctx * size_of::<f32>(),
            )?;
            fill_u32(
                self.moe_acc.ptr_mut(),
                0,
                MOE_ACC_ROWS * self.cfg.expert_in * size_of::<u64>(),
            )?;
        }
        let f = self.fetched();
        (self.hits0, self.misses0) = (f.hits, f.misses);
        self.stepped = false;
        Ok(())
    }

    /// The KV ceiling this engine allocated for, in tokens — `Engine::max_ctx`'s source.
    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    /// The pool's cumulative fetch counters — one accessor, because every consumer wants
    /// the pair and two single-counter getters were a jscpd clone of V4's besides. Named
    /// `Fetched` (was a bare `(u64, u64)`, which read the same with the counters swapped);
    /// the arm rebases with `Fetched::since`.
    pub(super) fn fetched(&self) -> crate::seam::Fetched {
        self.pin.routed.fetched()
    }

    /// The routed pool's byte budget — the anchor gate's residency DISCRIMINATOR: two
    /// engines whose budgets resolved equal are one residency state wearing two names,
    /// which is exactly what its P4 test must refuse to certify.
    pub fn pool_budget(&self) -> usize {
        self.pin.routed.budget()
    }

    /// The last forwarded position's logit vector — the decode's one public observable
    /// beyond its ids, and what the anchor gate's residency and state-carry comparisons
    /// score bit-for-bit (Muse Glimmer's `logits()` is the precedent, guard included).
    pub fn logits(&self) -> Result<Vec<f32>> {
        ensure!(
            self.stepped,
            "no forward has completed on this engine, so the logit buffer is uninitialised \
             allocator memory"
        );
        let raw = self.logits.copy_out()?;
        rivoli_core::num::f32s_le(&raw).context("a logit buffer of whole f32s")
    }
}
