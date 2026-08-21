//! The Muse Glimmer engine: the device scratch, the KV cache, and the one call that opens
//! both beside a placed pin. `forward.rs` and `decode.rs` drive the state this file
//! allocates; `geometry.rs` owns every shape and byte count it reads.
//!
//! Ported from `old:src/glimmer_gpu.rs::Glimmer::new` under this tree's split of authorship.
//! `old:` had the constructor subtract a runtime footprint from the budget and hand the
//! remainder to a pin that partitioned it — two authors, and the CLI could not reach the
//! refusal that lived between them. Here [`GlimmerEngine::open`] states the footprint once,
//! `partition()` decides placement, and this constructor allocates what the floor charged for.
//!
//! # This model's traps, and which file answers each
//!
//! | trap | where |
//! |---|---|
//! | `qk_scale_factor` 3.87 on **Q alone**, K gets 1.0 | `geometry::qk_scales`, one call each in `forward.rs` |
//! | the gate reads the LAYER INPUT, not the attend output | `forward.rs::attention` |
//! | two eps three orders apart, assigned by POSITION | [`GlimmerEngine::eps_pre`] vs `eps_post` |
//! | post-norms sit on the BRANCH, before the residual add | `forward.rs::branch_add` |
//! | `layer_types`, never `l % 4 == 3` | [`GlimmerEngine::attn_window`] |
//! | NoPE on full layers — `layer_rope_theta` read as a boolean | [`GlimmerEngine::rotated`] |
//! | the window is a RING on sliding layers and linear on full ones | `geometry::window_of` |
//! | centered norm (`1+w`) per layer, plain (`w`) at the head | two launchers, never a flag |

use super::geometry::{PREFILL_CHUNK, Window, check_footprint_inputs, slots_of, window_of};
use super::pin::{GlimmerPin, GlimmerPinCfg, scratch};
use crate::device::DeviceBuf;
use anyhow::{Context, Result, ensure};
use rivoli_artifact::glimmer_config::GlimmerTextConfig;

/// The Muse Glimmer decode path: weights, activations, KV cache, and the geometry the loop
/// over them reads.
pub struct GlimmerEngine<'a> {
    pub(super) pin: GlimmerPin,
    pub(super) cfg: &'a GlimmerTextConfig,
    pub(super) hq: usize,
    pub(super) hkv: usize,
    pub(super) hd: usize,
    /// `head_dim^-0.5`, the softmax scale. Applies IN ADDITION to `qk_scale_factor`.
    pub(super) attn_scale: f32,
    /// 1e-5 — the two pre-norms, the weightless QK-norm, the embedding norm, the final norm.
    pub(super) eps_pre: f32,
    /// 1e-8 — the two POST norms, and nothing else. Three orders of magnitude from `eps_pre`,
    /// assigned by position rather than by name.
    pub(super) eps_post: f32,
    /// Whether layer `l` rotates, from `layer_rope_theta != 0` — the boolean that field really
    /// is, resolved once at construction so no call site re-reads it as a per-layer base. The
    /// first-party code builds ONE table from `rope_parameters.rope_theta` and passes it or
    /// `None`, so a port that builds 52 tables is doing arithmetic nobody asked for.
    pub(super) rotated: Vec<bool>,

    // Activations. One set, reused every layer and every token.
    /// The residual stream for ONE position — the decode path's. A prefill reads its last row
    /// out of `xs` instead.
    pub(super) x: DeviceBuf,
    /// The residual streams of one prefill CHUNK, `PREFILL_CHUNK · hidden`. Layer-major
    /// prefill advances a layer at a time across a batch of positions, so every stream in the
    /// batch has to survive until the next layer reaches it — that is the memory the reorder
    /// costs, and `geometry::scratch_bytes` charges it.
    pub(super) xs: DeviceBuf,
    /// A pre-norm's output — the attention block's and the MLP's input.
    pub(super) xn: DeviceBuf,
    /// The branch, from the attention output or the MLP output up to its post-norm.
    pub(super) br: DeviceBuf,
    pub(super) q: DeviceBuf,
    pub(super) attn: DeviceBuf,
    pub(super) gate: DeviceBuf,
    pub(super) mg: DeviceBuf,
    pub(super) mu: DeviceBuf,
    pub(super) mh: DeviceBuf,
    pub(super) logits: DeviceBuf,
    /// `argmax`'s two outputs, one i32 then one f32.
    pub(super) pick: DeviceBuf,

    /// Per layer, keys then values, sized by that layer's own window.
    pub(super) kc: Vec<DeviceBuf>,
    pub(super) vc: Vec<DeviceBuf>,
    /// How many positions the linear (full-attention) layers can hold. A sliding layer's
    /// capacity is its clamped window and needs no field.
    pub(super) n_ctx: usize,
    /// Whether [`Self::sample`](super::forward) has ever run, so the logit accessor cannot
    /// hand back an uninitialised buffer.
    pub(super) sampled: bool,
    /// Decode-thread phase spans — `forward.rs` stamps them around this arm's existing
    /// sync points (`telemetry::ProfileSummary` documents what each bucket covers here:
    /// the slot-fill memcpy is the real fetch-wait; `sample`'s `device_sync` drains the
    /// whole layer stack into `head`).
    pub(super) prof: crate::telemetry::Phases,
}

impl<'a> GlimmerEngine<'a> {
    /// Build the engine over `pin`: allocate activations and a KV cache sized for `n_ctx`
    /// positions.
    ///
    /// `n_ctx` is the caller's prompt plus what it intends to generate, not
    /// `max_position_embeddings` — this model's is 131072, and a full-attention layer's cache
    /// is linear in it, so sizing from the config would ask for 3.5 GB of cache to decode
    /// twelve tokens.
    pub fn new(pin: GlimmerPin, cfg: &'a GlimmerTextConfig, n_ctx: usize) -> Result<Self> {
        check_footprint_inputs(cfg, n_ctx)?;
        // **The geometry is asserted here rather than inherited from
        // `GlimmerTextConfig::validate`, WHICH THIS CONSTRUCTOR DOES NOT CALL.** `new` is
        // `pub` on a `pub` type, so a caller can hand it a config the artifact loader never
        // saw. The KV vectors below are built by iterating `layer_types` while everything that
        // indexes them runs `0..n_layers`, so a short array is an index panic mid-token after
        // the pin is already placed.
        ensure!(
            cfg.layer_types.len() == cfg.n_layers && cfg.layer_rope_theta.len() == cfg.n_layers,
            "the config declares {} layers but carries {} layer_types and {} layer_rope_theta",
            cfg.n_layers,
            cfg.layer_types.len(),
            cfg.layer_rope_theta.len()
        );
        let (hq, hkv, hd) = (cfg.n_heads, cfg.num_key_value_heads, cfg.head_dim);
        let (qd, kvd) = (hq * hd, hkv * hd);
        // **The allocation is sized BY `window_of`, not alongside it.** A ring the loop
        // indexes modulo `cap` and a buffer sized from a second copy of that expression is a
        // device write past the end the first time the two disagree, and neither the launcher
        // nor HIP would say so.
        let mut kc = Vec::with_capacity(cfg.n_layers);
        let mut vc = Vec::with_capacity(cfg.n_layers);
        for &k in &cfg.layer_types {
            let slots = slots_of(k, cfg.sliding_window, n_ctx)?;
            kc.push(scratch(slots * kvd)?);
            vc.push(scratch(slots * kvd)?);
        }
        Ok(Self {
            pin,
            cfg,
            hq,
            hkv,
            hd,
            attn_scale: (hd as f64).powf(-0.5) as f32,
            eps_pre: cfg.rms_norm_eps as f32,
            eps_post: cfg.post_norm_eps as f32,
            rotated: cfg.layer_rope_theta.iter().map(|t| *t != 0.0).collect(),
            x: scratch(cfg.hidden)?,
            xs: scratch(PREFILL_CHUNK.min(n_ctx) * cfg.hidden)?,
            xn: scratch(cfg.hidden)?,
            br: scratch(cfg.hidden)?,
            q: scratch(qd)?,
            attn: scratch(qd)?,
            gate: scratch(qd)?,
            mg: scratch(cfg.inter)?,
            mu: scratch(cfg.inter)?,
            mh: scratch(cfg.inter)?,
            logits: scratch(cfg.vocab)?,
            pick: DeviceBuf::new(8)?,
            kc,
            vc,
            n_ctx,
            sampled: false,
            prof: crate::telemetry::Phases::default(),
        })
    }

    /// Open `dir` as a Glimmer engine: let `partition()` place the weights against a floor
    /// derived from `(cfg, n_ctx)`, then allocate what that floor charged for.
    ///
    /// **`n_ctx` reaches the pin rather than two byte counts computed here.** The KV cache and
    /// the activation scratch are charges the floor must cover AND allocations
    /// [`Self::new`] then makes, and both sides derive them from `geometry`'s one pair of
    /// functions — so the bytes the partition was refused or granted against are the bytes
    /// that get allocated, by construction rather than by two call sites agreeing.
    pub fn open(
        dir: &str,
        cfg: &'a GlimmerTextConfig,
        capacity: usize,
        n_ctx: usize,
    ) -> Result<Self> {
        let pin = GlimmerPin::build(dir, cfg, GlimmerPinCfg { capacity, n_ctx })?;
        Self::new(pin, cfg, n_ctx)
    }

    /// This layer's window — [`window_of`] against its kind from `layer_types`.
    ///
    /// **`layer_types[l]`, never `l % 4 == 3`.** The `[s,s,s,full]` period is a fact about
    /// this checkpoint and not a rule about the architecture, so a loop that computes it is
    /// right until the first checkpoint whose pattern differs — and wrong fluently when one
    /// does.
    pub(super) fn attn_window(&self, l: usize, pos: usize) -> Result<Window> {
        let kind = *self
            .cfg
            .layer_types
            .get(l)
            .with_context(|| format!("layer {l} is past this model's layer_types"))?;
        window_of(kind, self.cfg.sliding_window, self.n_ctx, pos)
    }

    /// The KV ceiling this engine allocated for, in tokens — see
    /// [`Engine::max_ctx`](crate::seam::Engine::max_ctx), whose only job is to hand this
    /// number to a caller that must refuse an over-long request before decoding it.
    pub fn max_ctx(&self) -> usize {
        self.n_ctx
    }

    /// Slot hits and fills — see [`GlimmerPin::slot_stats`]. The decode loops read the
    /// same numbers through [`Self::fetched`]; this raw pair stays for the fp8 decode
    /// test, which asserts on it directly.
    pub fn slot_stats(&self) -> (u64, u64) {
        self.pin.slot_stats()
    }

    /// [`Self::slot_stats`] as one named pair — this arm's answer to the seam's
    /// `hits`/`misses` question, at its own whole-layer granularity; the arm rebases with
    /// `Fetched::since`.
    pub fn fetched(&self) -> crate::seam::Fetched {
        let (hits, misses) = self.slot_stats();
        crate::seam::Fetched { hits, misses }
    }

    /// How many layers the budget pinned, and how many stream.
    pub fn residency(&self) -> (usize, usize) {
        (self.pin.pinned_layers(), self.pin.streamed_layers())
    }
}
