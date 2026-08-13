//! **Muse Glimmer's decode loop — S3.** The first `src/` caller of the S2 kernel family.
//!
//! Everything below consumes R1's contract ([`GlimmerPin::layer`]) and S2's launchers. Nothing
//! here decides residency: `layer(l)` returns the same shape whether the budget pinned that layer
//! or streamed it, and the write-after-read fence that makes streaming safe lives inside it.
//!
//! # This loop runs entirely on the NULL stream, and that is a decision
//!
//! `tests/glimmer_stream_order.rs` measures what a null stream costs a consumer whose producer is
//! on a real one: 99.9% of a 2M-element operand read stale. The conclusion it draws is **not**
//! "never pass null" — it is that every launch touching one buffer must be on ONE stream, and that
//! a compute stream at two call sites inside an otherwise null-stream layer is the same bug
//! inverted. **Four of the twelve launchers this chain drives take no stream parameter at all** —
//! `rmsnorm_single`, `rope_split_half`, `vadd` and `argmax` — so a "compute stream" loop would be
//! exactly that inversion: a real stream at the eight launches that accept one and the legacy
//! default at the four that cannot.
//!
//! > **CORRECTED 2026-08-13, same day, by review.** This said SIX and listed `rope_interleave` and
//! > "`swiglu`'s siblings". `swiglu` takes a stream, and this file passes it `NULL_STREAM` like
//! > every other; `rope_interleave` is the INTERLEAVED convention — the one this model must never
//! > use, since Glimmer is split-half and applying one where the other belongs is trap 9, fluent
//! > and wrong. Writing it into the inventory of "launchers this chain needs" put a forbidden
//! > kernel on S5's to-do list. The decision below is unaffected; its stated reason was wrong on
//! > two of its six entries, and a closure gets inherited together with its justification.
//!
//! So the whole chain is null, uniformly, and the legacy stream's own ordering carries the
//! dependencies. `f4gpu.rs::pre_norm` is the precedent and makes the same argument. **The cost is
//! that nothing overlaps**, which is S5's to change — and the change is not "pass a stream here",
//! it is giving those four launchers one first. Until then a stream argument in this file would be
//! decoration that reads as a guarantee.
//!
//! # What the loop must not get wrong, and where each is handled
//!
//! | trap | where |
//! |---|---|
//! | `qk_scale_factor` 3.87 on **Q alone**, K gets 1.0 | [`Glimmer::attention`], one call each |
//! | the gate reads the LAYER INPUT, not the attend output | [`Glimmer::attention`], `gate` is built from `xn` |
//! | two eps three orders apart, assigned by POSITION | [`Glimmer::layer`], `eps_pre` vs `eps_post` |
//! | post-norms sit on the BRANCH, before the residual add | [`Glimmer::layer`], norm-then-`vadd` |
//! | `layer_types`, never `l % 4 == 3` | [`Glimmer::attn_window`] |
//! | NoPE on full layers — `layer_rope_theta` read as a boolean | [`Glimmer::attention`], `rotated` |
//! | the window is a RING on sliding layers and linear on full ones | [`Glimmer::attn_window`] |
//! | centered norm (`1+w`) per layer, plain (`w`) at the head | two launchers, never a flag |
//!
//! **`attn_window` returns `win`, `ring_cap` and the slot together for one reason:** the launcher
//! accepts `win != 0` with `ring_cap == 0`, which silently truncates a full layer's causal prefix
//! to its last `sliding_window` rows — fluent and wrong. Deriving all three from one `match` on
//! the layer's kind is what makes that combination unconstructible here rather than merely
//! unlikely.

use crate::artifact::model::{GlimmerTextConfig, LayerKind};
use crate::backend::NULL_STREAM;
use crate::backend::hip::{
    device_sync, launch_argmax, launch_embed_bf16_row_bcast, launch_gemm_bf16, launch_gqa_attend,
    launch_logit_softcap, launch_rmsnorm_centered_single, launch_rmsnorm_single,
    launch_rmsnorm_weightless_batch, launch_rope_split_half, launch_sigmoid_gate, launch_swiglu,
    launch_vadd,
};
use crate::memory::device::DeviceBuf;
use crate::memory::pin::{Bf16Weight, GlimmerLayerPin, GlimmerPin};
use anyhow::{Context, Result, ensure};

/// f32 scratch of `n` elements.
fn scratch(n: usize) -> Result<DeviceBuf> {
    DeviceBuf::new(n * 4)
}

/// One bf16 projection applied to a single row: `out = w · x`, `w` being `[o_dim, i_dim]`.
///
/// A GEMM at `m = 1` rather than a GEMV because there is no bf16 GEMV in the tree and adding one
/// is S5's arithmetic to price, not S3's. The shape check is here rather than trusted: every
/// caller below passes a buffer sized from the config, and a projection whose `i_dim` disagrees
/// with the activation it is handed reads past the end of a live allocation and stays fluent.
///
/// # Safety
/// `x` is `w.i_dim` live f32, `out` is `w.o_dim` writable f32, the two do not alias, and both
/// outlive the next [`device_sync`].
unsafe fn proj(x: *const f32, w: &Bf16Weight, out: *mut f32, i_dim: usize) -> Result<()> {
    ensure!(
        w.i_dim == i_dim,
        "projection expects an activation of {} but was handed {i_dim} — a shape mismatch here \
         reads past the end of a live buffer and produces fluent wrong text",
        w.i_dim
    );
    // SAFETY: the caller's contract, plus the `i_dim` agreement checked above.
    unsafe { launch_gemm_bf16(x, w.packed, out, 1, w.o_dim, w.i_dim, NULL_STREAM) }
}

/// The first `n` f32 of a device buffer, on the host.
///
/// Shared by the two readbacks — `build.rs`'s jscpd gate rejected the second copy, and it is right
/// about the substance too: the `copy_out_prefix` length and the `chunks_exact` width are one fact
/// written twice, and a pair that disagreed would return a shorter vector rather than an error.
fn read_f32(b: &DeviceBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = Vec::new();
    b.copy_out_prefix(&mut raw, n * 4)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Where one layer's keys and values live, and how the kernel must be told to read them.
///
/// The three fields are derived together and never separately — see this module's header.
pub struct Window {
    /// `sliding_window` on a sliding layer, 0 (the whole causal prefix) on a full one.
    pub win: usize,
    /// The ring's capacity, or 0 for a linear cache the kernel indexes BY POSITION.
    pub ring_cap: usize,
    /// Which cache slot this position's key and value belong in.
    pub slot: usize,
}

/// This layer's window, ring capacity and slot — all three from ONE match on its kind.
///
/// A full layer gets `win = 0` AND `ring_cap = 0`: the kernel's causal bound is derived from
/// `(start_pos, win)`, and its slot map from `ring_cap`, so the pair that truncates a full layer's
/// prefix (`win != 0` with `ring_cap == 0`) cannot be produced by any input here. That combination
/// is ACCEPTED by the launcher — its guard rejects only the inverse — which is why the constraint
/// is expressed as a shape rather than as a check.
///
/// A free function rather than a method so `tests/glimmer_loop.rs` can gate it without a device: a
/// rule whose test needs 55 GB of weights to run is a rule nothing runs.
///
/// `win` must be positive on a sliding layer — [`Glimmer::new`] refuses a config where it is not,
/// because a zero here is a division by zero one line down and a contradiction in the manifest
/// (a layer typed `sliding_attention` with no window).
pub fn window_of(kind: LayerKind, win: usize, n_ctx: usize, pos: usize) -> Window {
    match kind {
        LayerKind::SlidingAttention => {
            // Asserted rather than left to `pos % 0`. This is `pub`, so a caller that never went
            // through `Glimmer::new`'s refusal — a bench, a future server path, a test — reaches
            // it directly, and "attempt to calculate the remainder with a divisor of zero" names
            // neither the field that was zero nor why it may not be.
            assert!(
                win > 0 && n_ctx > 0,
                "a sliding layer needs a positive window and context, got win {win} and n_ctx \
                 {n_ctx} — a layer typed `sliding_attention` with `sliding_window` 0 is a \
                 contradiction in the manifest"
            );
            // **The window is clamped WITH the ring, and clamping only one of them is a hard
            // refusal at layer 0.** `rivoli_gqa_attend` rejects `ring_cap < win + tq - 1` (code
            // 1005, `kernels/attn.hip:647`), which at `tq == 1` is `ring_cap < win` — so a ring
            // sized to a context shorter than the window and a `win` left at the model's value is
            // the one pair the launcher refuses outright. The first version returned exactly that,
            // and since the default `--bench 64` gives `n_ctx` ≈ 70 against a 2048-row window, the
            // default invocation could not emit a single token (review, 2026-08-13).
            //
            // Clamping both is not a compromise, it is the same attention: the clamp only fires
            // when `n_ctx <= win`, and then every position is below `win`, so the kernel's
            // `lo = (win > 0 && pos >= win) ? pos - win + 1 : 0` is 0 either way — the whole causal
            // prefix. A sliding layer in a run shorter than its window has nothing to slide past.
            let cap = win.min(n_ctx);
            Window {
                win: cap,
                ring_cap: cap,
                slot: pos % cap,
            }
        }
        // `ring_cap == 0` makes the slot the position itself, so the cache must run from position
        // 0 — which it does: nothing here ever trims it.
        LayerKind::FullAttention => Window {
            win: 0,
            ring_cap: 0,
            slot: pos,
        },
    }
}

/// One layer's twelve device handles, resolved from the pin ONCE per layer.
///
/// **This exists to make [`GlimmerPin::layer`]'s narrowest rule structural.** That contract says
/// in as many words: *do not launch from pointers captured across a `layer()` call* — the fence it
/// performs syncs kernels already enqueued and cannot see a caller that captures layer `l`'s
/// addresses, requests another layer, and only then launches. The first version of this loop
/// requested the pin three times per layer and launched from the first request's pointers after
/// the second and third, which was safe only because all three asked for the SAME `l` and a repeat
/// request takes the hit path. Safe by accident is the state S5's prefetch turns into wrong text,
/// so the request happens once and the handles are passed down (review, 2026-08-13).
///
/// It also fixes a live measurement defect: `GlimmerPin`'s `hits` counter is documented as the way
/// to tell a working partition from a thrashing one, and three requests per layer inflated it
/// threefold.
///
/// A copy of `GlimmerLayerPin` rather than a borrow, because holding the borrow would keep
/// `self.pin` mutably borrowed across every `&mut self` call in the layer body.
struct Handles {
    input_ln: *const f32,
    post_attn_ln: *const f32,
    pre_ffn_ln: *const f32,
    post_ffn_ln: *const f32,
    q: Bf16Weight,
    k: Bf16Weight,
    v: Bf16Weight,
    o: Bf16Weight,
    attn_gate: Bf16Weight,
    mlp_gate: Bf16Weight,
    mlp_up: Bf16Weight,
    mlp_down: Bf16Weight,
}

impl Handles {
    fn of(p: &GlimmerLayerPin) -> Self {
        Self {
            input_ln: p.input_ln,
            post_attn_ln: p.post_attn_ln,
            pre_ffn_ln: p.pre_ffn_ln,
            post_ffn_ln: p.post_ffn_ln,
            q: p.q,
            k: p.k,
            v: p.v,
            o: p.o,
            attn_gate: p.attn_gate,
            mlp_gate: p.mlp_gate,
            mlp_up: p.mlp_up,
            mlp_down: p.mlp_down,
        }
    }
}

/// The two scales the weightless QK-norm takes: `qk_scale_factor` on Q, 1.0 on K.
///
/// **What the model depends on is their PRODUCT, and the assignment is fidelity rather than
/// correctness — measured 2026-08-13, against what this tree and `glimmer-architecture.md` trap 3
/// had said for weeks.** Both operands are weightless-RMS-normed before the scale, so the score is
/// `(a·q̂)·(b·k̂)·head_dim^-0.5` and only `a·b` enters; RoPE is a norm-preserving rotation applied
/// afterwards and commutes with a scalar, and a cached key scaled by 3.87 dotted with a later
/// query scaled by 1.0 gives the same product. `tests/glimmer_chain.rs` red-proves it both ways:
/// SWAPPING the two leaves the logits inside 2.3e-6 reduction noise, while DROPPING the factor
/// moves them by 1.7.
///
/// So the pair stays named — a swap is still a rename rather than an argument reorder, and 3.87 on
/// Q is where the reference puts it, where the intermediate magnitudes belong, and what any future
/// consumer of `q` or `k` alone (a trace, a probe) would see — but it is no longer described as a
/// hazard that produces wrong text, because it cannot.
///
/// It does NOT replace the softmax scale. `head_dim^-0.5` still applies, for an effective Q factor
/// of `3.87 / sqrt(128)`.
pub struct QkScale {
    pub q: f32,
    pub k: f32,
}

/// [`QkScale`] from the config's `qk_scale_factor`. Free, and gated deviceless, for the reason
/// [`window_of`] is.
pub fn qk_scales(qk_scale_factor: f32) -> QkScale {
    QkScale {
        q: qk_scale_factor,
        k: 1.0,
    }
}

/// How many KV slots layer kind `k` gets at `n_ctx`.
///
/// **Derived from [`window_of`], and read by BOTH the allocation and the budget accounting.** The
/// two must agree exactly: a cache sized from one expression and indexed by another is a device
/// write past the end, and a budget that charges for a different number of slots than were
/// allocated is a report the operator cannot act on. jscpd reported the second copy the moment it
/// was written, which is the gate arriving at the same conclusion.
fn slots_of(k: LayerKind, win: usize, n_ctx: usize) -> usize {
    match window_of(k, win, n_ctx, 0).ring_cap {
        0 => n_ctx,
        cap => cap,
    }
}

/// The device bytes a decode needs BESIDE its weights: the KV cache at `n_ctx`, plus the
/// activation scratch.
///
/// **It has to be subtracted from the budget before the pin is built, and it was not.**
/// `GlimmerPin::build` sizes a `DeviceTier` from the budget and clears `guard_capacity`'s
/// `capacity + 4 GiB HEADROOM <= free`; every allocation below was then an unguarded `hipMalloc`
/// on top. At the 131072-position ceiling the KV cache alone is ~3.4 GiB — 85% of the reserve that
/// exists for driver scratch — and the `residency:` line the operator reads counted none of it.
/// `GlimmerTextConfig::floor_bytes`' own doc had named the gap ("KV at the configured context,
/// activation scratch and the DFlash drafter are not here"); it was inert until S3 allocated a KV
/// cache for the first time (review, 2026-08-13).
///
/// `pub` because `run_glimmer` reports the partition, and a reported split that used a different
/// budget from the built one is the disagreement this repo has already been bitten by.
pub fn runtime_bytes(gt: &GlimmerTextConfig, n_ctx: usize) -> Result<usize> {
    ensure!(n_ctx > 0, "n_ctx must be positive");
    let kvd = gt.num_key_value_heads * gt.head_dim;
    // Sized BY `window_of`, exactly as the allocation is — the two must not be able to disagree
    // about how many slots a layer gets.
    let mut kv = 0usize;
    for &k in &gt.layer_types {
        // Keys AND values.
        kv += 2 * slots_of(k, gt.sliding_window, n_ctx) * kvd;
    }
    let qd = gt.n_heads * gt.head_dim;
    // `x`, `xn`, `br`; `q`, `attn`, `gate`; `mg`, `mu`, `mh`; `logits`; `pick` (2 words).
    let act = 3 * gt.hidden + 3 * qd + 3 * gt.inter + gt.vocab + 2;
    (kv + act)
        .checked_mul(4)
        .context("the runtime footprint overflows a usize")
}

/// The Muse Glimmer decode path: weights, activations, KV cache, and the loop over them.
pub struct Glimmer {
    pin: GlimmerPin,
    n_layers: usize,
    hidden: usize,
    inter: usize,
    vocab: usize,
    hq: usize,
    hkv: usize,
    hd: usize,
    /// `head_dim^-0.5`, the softmax scale. Applies IN ADDITION to `qk_scale_factor`.
    attn_scale: f32,
    /// 1e-5 — the two pre-norms, the weightless QK-norm, the embedding norm, the final norm.
    eps_pre: f32,
    /// 1e-8 — the two POST norms, and nothing else. Three orders of magnitude from `eps_pre`,
    /// assigned by position rather than by name.
    eps_post: f32,
    /// 3.87, and it multiplies Q only.
    qk_scale: f32,
    /// `output_multiplier` and `final_logit_softcapping`.
    mult: f32,
    cap: f32,
    win: usize,
    theta: f64,
    kinds: Vec<LayerKind>,
    /// Whether layer `l` rotates, from `layer_rope_theta != 0` — the boolean that field really
    /// is, resolved once at construction so no call site re-reads it as a per-layer base.
    rotated: Vec<bool>,

    // Activations. One set, reused every layer and every token.
    /// The residual stream.
    x: DeviceBuf,
    /// A pre-norm's output — the attention block's and the MLP's input.
    xn: DeviceBuf,
    /// The branch, from the attention output or the MLP output up to its post-norm.
    br: DeviceBuf,
    q: DeviceBuf,
    attn: DeviceBuf,
    gate: DeviceBuf,
    mg: DeviceBuf,
    mu: DeviceBuf,
    mh: DeviceBuf,
    logits: DeviceBuf,
    /// `argmax`'s two outputs, one i32 then one f32.
    pick: DeviceBuf,

    /// Per layer, keys then values, sized by that layer's own window.
    kc: Vec<DeviceBuf>,
    vc: Vec<DeviceBuf>,
    /// How many positions the linear (full-attention) layers can hold. A sliding layer's
    /// capacity is `win` and needs no field.
    n_ctx: usize,
    /// Whether [`Self::sample`] has ever run, so [`Self::logits`] cannot hand back an
    /// uninitialised buffer.
    sampled: bool,
}

impl Glimmer {
    /// Build the engine: pin the weights the budget allows, then allocate activations and a KV
    /// cache sized for `n_ctx` positions.
    ///
    /// `n_ctx` is the caller's prompt plus what it intends to generate, not
    /// `max_position_embeddings` — this model's is 131072, and a full-attention layer's cache is
    /// linear in it, so sizing from the config would ask for 3.5 GB of cache to decode twelve
    /// tokens.
    pub fn new(
        dir: &str,
        gt: &GlimmerTextConfig,
        budget: Option<usize>,
        n_ctx: usize,
    ) -> Result<Self> {
        ensure!(n_ctx > 0, "n_ctx must be positive");
        // **The geometry is asserted here rather than inherited from `GlimmerTextConfig::validate`,
        // WHICH THIS CONSTRUCTOR DOES NOT CALL.** `new` is `pub` on a `pub` type, so a caller can
        // hand it a config the artifact loader never saw. Two of these matter beyond tidiness: the
        // KV vectors below are built by iterating `layer_types` while everything that indexes them
        // runs `0..n_layers`, so a short array is an index panic mid-token after the pin is
        // already placed; and a `sliding_window` of 0 under a `sliding_attention` layer reaches
        // `window_of` as a division by zero (review, 2026-08-13).
        ensure!(
            gt.layer_types.len() == gt.n_layers && gt.layer_rope_theta.len() == gt.n_layers,
            "the config declares {} layers but carries {} layer_types and {} layer_rope_theta",
            gt.n_layers,
            gt.layer_types.len(),
            gt.layer_rope_theta.len()
        );
        let sliding = gt
            .layer_types
            .iter()
            .filter(|k| **k == LayerKind::SlidingAttention)
            .count();
        ensure!(
            sliding == 0 || gt.sliding_window > 0,
            "the config types {sliding} layers `sliding_attention` and gives `sliding_window` 0"
        );
        ensure!(
            n_ctx <= gt.max_position_embeddings,
            "n_ctx {n_ctx} is past this model's {} trained positions",
            gt.max_position_embeddings
        );
        // **The pin gets what is left after the KV cache and the activations**, not the whole
        // budget — see [`runtime_bytes`]. Subtracted BEFORE the tier is sized, because the tier is
        // what `guard_capacity` checks against free memory and everything below it is unguarded.
        let overhead = runtime_bytes(gt, n_ctx)?;
        if let Some(b) = budget {
            ensure!(
                b > overhead,
                "the KV cache and activations for {n_ctx} positions need {:.3} GB, which is the \
                 whole {:.3} GB budget — there is nothing left to pin weights into. Lower the \
                 context or raise --max-mem",
                overhead as f64 / 1e9,
                b as f64 / 1e9
            );
        }
        let pin = GlimmerPin::build(dir, gt, budget.map(|b| b - overhead))?;
        let (hq, hkv, hd) = (gt.n_heads, gt.num_key_value_heads, gt.head_dim);
        let qd = hq * hd;
        let kvd = hkv * hd;
        // **The allocation is sized BY [`window_of`], not alongside it.** A ring the loop indexes
        // modulo `cap` and a buffer sized from a second copy of that expression is a device write
        // past the end the first time the two disagree, and neither the launcher nor HIP would say
        // so. A `ring_cap` of 0 means the linear cache, whose extent is the context.
        let mut kc = Vec::with_capacity(gt.n_layers);
        let mut vc = Vec::with_capacity(gt.n_layers);
        for &k in &gt.layer_types {
            let slots = slots_of(k, gt.sliding_window, n_ctx);
            kc.push(scratch(slots * kvd)?);
            vc.push(scratch(slots * kvd)?);
        }
        Ok(Self {
            pin,
            n_layers: gt.n_layers,
            hidden: gt.hidden,
            inter: gt.inter,
            vocab: gt.vocab,
            hq,
            hkv,
            hd,
            attn_scale: (hd as f64).powf(-0.5) as f32,
            eps_pre: gt.rms_norm_eps as f32,
            eps_post: gt.post_norm_eps as f32,
            qk_scale: gt.qk_scale_factor as f32,
            mult: gt.output_multiplier as f32,
            cap: gt.final_logit_softcapping as f32,
            win: gt.sliding_window,
            theta: gt.rope_parameters.rope_theta,
            kinds: gt.layer_types.clone(),
            // `layer_rope_theta` is 500000 on sliding layers and 0 on full ones, and the
            // first-party code builds ONE table and passes it or `None` — so the field is a
            // boolean wearing a float's clothes. Resolved here, once.
            rotated: gt.layer_rope_theta.iter().map(|t| *t != 0.0).collect(),
            x: scratch(gt.hidden)?,
            xn: scratch(gt.hidden)?,
            br: scratch(gt.hidden)?,
            q: scratch(qd)?,
            attn: scratch(qd)?,
            gate: scratch(qd)?,
            mg: scratch(gt.inter)?,
            mu: scratch(gt.inter)?,
            mh: scratch(gt.inter)?,
            logits: scratch(gt.vocab)?,
            pick: DeviceBuf::new(8)?,
            kc,
            vc,
            n_ctx,
            sampled: false,
        })
    }

    /// This layer's window — [`window_of`] against its kind from `layer_types`.
    ///
    /// **`self.kinds[l]`, never `l % 4 == 3`.** The `[s,s,s,full]` period is a fact about this
    /// checkpoint and not a rule about the architecture, so a loop that computes it is right until
    /// the first checkpoint whose pattern differs — and wrong fluently when one does.
    fn attn_window(&self, l: usize, pos: usize) -> Window {
        window_of(self.kinds[l], self.win, self.n_ctx, pos)
    }

    /// The attention block for layer `l` at absolute position `pos`, reading `xn` and leaving the
    /// gated, projected output in `br`.
    ///
    /// Order is **norm → scale → rope → cache → attend**, and the cache therefore holds
    /// post-QK-norm, post-RoPE keys. Each projection writes where its result is consumed: `k` and
    /// `v` go STRAIGHT into their cache slot, so the norm and the rotation run in place on the
    /// cache and no copy exists to get wrong.
    fn attention(&mut self, l: usize, pos: usize, w: &Window, h: &Handles) -> Result<()> {
        let (qd, kvd) = (self.hq * self.hd, self.hkv * self.hd);
        let (wq, wk, wv, wo, wg) = (h.q, h.k, h.v, h.o, h.attn_gate);
        let xn = self.xn.ptr() as *const f32;
        let kslot = unsafe { (self.kc[l].ptr() as *mut f32).add(w.slot * kvd) };
        let vslot = unsafe { (self.vc[l].ptr() as *mut f32).add(w.slot * kvd) };
        // SAFETY: `xn` is `hidden` live f32 written by this layer's pre-norm; each destination is
        // its projection's `o_dim` f32 inside a live allocation — `q` its own buffer, `k`/`v` one
        // `hkv*hd` slot of a cache with at least `slot+1` slots by `attn_window`'s construction.
        // None aliases `xn` or another. All outlive the `device_sync` in `decode`.
        unsafe {
            proj(xn, &wq, self.q.ptr() as *mut f32, self.hidden)?;
            proj(xn, &wk, kslot, self.hidden)?;
            proj(xn, &wv, vslot, self.hidden)?;
            // **The 3.87 is Q's alone**, and it arrives here by NAME — see [`QkScale`], which
            // also records what a swap actually costs (nothing: only the product enters the
            // score). What is NOT free is dropping or doubling the factor, and that the chain gate
            // does see.
            let s = qk_scales(self.qk_scale);
            launch_rmsnorm_weightless_batch(
                self.q.ptr() as *mut f32,
                self.hq,
                self.hd,
                self.eps_pre,
                s.q,
                NULL_STREAM,
            )?;
            launch_rmsnorm_weightless_batch(
                kslot,
                self.hkv,
                self.hd,
                self.eps_pre,
                s.k,
                NULL_STREAM,
            )?;
            // NoPE: the full-attention layers carry no rotation at all. `rotated` is the boolean
            // `layer_rope_theta` really is; the base comes from the ONE `rope_parameters` table.
            if self.rotated[l] {
                launch_rope_split_half(
                    self.q.ptr() as *mut f32,
                    self.hq,
                    self.hd,
                    self.hd,
                    pos,
                    self.theta,
                )?;
                launch_rope_split_half(kslot, self.hkv, self.hd, self.hd, pos, self.theta)?;
            }
            launch_gqa_attend(
                self.q.ptr() as *const f32,
                self.kc[l].ptr() as *const f32,
                self.vc[l].ptr() as *const f32,
                self.hq,
                self.hkv,
                self.hd,
                1,
                pos,
                w.win,
                w.ring_cap,
                self.attn_scale,
                self.attn.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            // **The gate reads the layer input, not the attend output.** `wg` consumes `xn`; a
            // gate built from `self.attn` has the right shapes, the right tensor and the wrong
            // model (trap 3's sibling, `glimmer-architecture.md` §4 item 3).
            proj(xn, &wg, self.gate.ptr() as *mut f32, self.hidden)?;
            launch_sigmoid_gate(
                self.attn.ptr() as *mut f32,
                self.gate.ptr() as *const f32,
                qd,
                NULL_STREAM,
            )?;
            proj(
                self.attn.ptr() as *const f32,
                &wo,
                self.br.ptr() as *mut f32,
                qd,
            )?;
        }
        Ok(())
    }

    /// A sandwich norm's PRE half: `xn = centered_norm(x, w, eps_pre)`, leaving the residual
    /// stream untouched for the add that closes the block.
    ///
    /// # Safety
    /// `w` is `hidden` live f32 and stays valid until the next [`GlimmerPin::layer`] call.
    unsafe fn pre_norm(&mut self, w: *const f32) -> Result<()> {
        // SAFETY: `x` and `xn` are each `hidden` f32 this struct owns for its whole life; `w` is
        // the caller's obligation.
        unsafe {
            launch_rmsnorm_centered_single(
                self.x.ptr() as *const f32,
                w,
                self.hidden,
                self.eps_pre,
                self.xn.ptr() as *mut f32,
                NULL_STREAM,
            )
        }
    }

    /// A sandwich norm's POST half: `br = centered_norm(br, w, eps_post); x += br`.
    ///
    /// **This is the shape of the whole trap.** The norm runs on the BRANCH, before the residual
    /// add — not on the sum, and not on the stream. One function rather than two copies because
    /// the attention block and the MLP close identically, and two copies of "which operand does
    /// the post-norm take" is the pair that can drift with only one of them wrong.
    ///
    /// `eps_post` is 1e-8 against `eps_pre`'s 1e-5, and the assignment is by POSITION: both halves
    /// take it from the field rather than from a parameter, so no call site can pass the other.
    ///
    /// # Safety
    /// `w` is `hidden` live f32 and stays valid until the next [`GlimmerPin::layer`] call.
    unsafe fn branch_add(&mut self, w: *const f32) -> Result<()> {
        // SAFETY: `br` and `x` are each `hidden` f32 this struct owns; the centered norm's own
        // contract permits `y` aliasing `x`, which is what makes the in-place form legal.
        unsafe {
            launch_rmsnorm_centered_single(
                self.br.ptr() as *const f32,
                w,
                self.hidden,
                self.eps_post,
                self.br.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            launch_vadd(
                self.x.ptr() as *mut f32,
                self.br.ptr() as *const f32,
                self.hidden,
            )
        }
    }

    /// The MLP: `down(silu(gate(xn)) · up(xn))`, leaving its output in `br` for the post-norm.
    fn mlp(&mut self, h: &Handles) -> Result<()> {
        let (wg, wu, wd) = (h.mlp_gate, h.mlp_up, h.mlp_down);
        let xn = self.xn.ptr() as *const f32;
        // SAFETY: `xn` is `hidden` live f32 from the pre-norm; `mg`/`mu`/`mh` are each `inter` and
        // `br` is `hidden`, all owned by this struct. `swiglu` writes `mh`, which aliases neither
        // of its operands. The three weights are live until the next `pin.layer` call.
        unsafe {
            proj(xn, &wg, self.mg.ptr() as *mut f32, self.hidden)?;
            proj(xn, &wu, self.mu.ptr() as *mut f32, self.hidden)?;
            launch_swiglu(
                self.mg.ptr() as *const f32,
                self.mu.ptr() as *const f32,
                self.inter,
                self.mh.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            proj(
                self.mh.ptr() as *const f32,
                &wd,
                self.br.ptr() as *mut f32,
                self.inter,
            )
        }
    }

    /// One decoder layer: sandwich norms around an attention block and an MLP.
    ///
    /// Reads as the reference's twelve lines because it is those twelve lines —
    /// `glimmer-architecture.md` §3. Each norm's eps comes from the half it belongs to, so the
    /// three-orders-of-magnitude swap is not expressible at this level.
    ///
    /// # The pin invariant this body relies on, stated as what actually holds
    ///
    /// The four norm pointers are captured once and then used ACROSS the `pin.layer` calls that
    /// [`Self::attention`] and [`Self::mlp`] make — `post_attn_ln` after the first, `post_ffn_ln`
    /// after the second. `GlimmerPin::layer`'s own doc names that as the hazard its fence does NOT
    /// cover: the fence syncs kernels already enqueued, and cannot see a caller that captures layer
    /// `l`'s pointers, calls `layer(l + 1)`, and only then launches.
    ///
    /// **What makes this sound is narrower and it is the thing to preserve: every `pin.layer` call
    /// inside this function is for the SAME `l`.** A repeat visit takes the `slot_layer[s] ==
    /// Some(l)` hit path, which neither fences nor refills, so no pointer can move under a capture.
    /// Prefetching `l + 1`, splitting the sandwich halves across layers, or reordering these calls
    /// breaks it with no compile error and no fence to catch it — position-dependent wrong text.
    /// The first version of this comment claimed the opposite ordering and cited the fence as the
    /// mechanism; both were wrong (review, 2026-08-13).
    fn layer(&mut self, l: usize, pos: usize) -> Result<()> {
        let w = self.attn_window(l, pos);
        let h = Handles::of(self.pin.layer(l)?);
        // SAFETY: every pointer in `h` is live in the pin and stays at its address for as long as
        // its slot holds layer `l` — which is the whole of this function, per the section above.
        unsafe { self.pre_norm(h.input_ln)? };
        self.attention(l, pos, &w, &h)?;
        // SAFETY: as above.
        unsafe { self.branch_add(h.post_attn_ln)? };
        // SAFETY: as above.
        unsafe { self.pre_norm(h.pre_ffn_ln)? };
        self.mlp(&h)?;
        // SAFETY: as above.
        unsafe { self.branch_add(h.post_ffn_ln) }
    }

    /// Embed `token`, run every layer at `pos`, and leave the final hidden state in `x`.
    ///
    /// **The embedding is NORMED, by the weightless form** (`MuseGlimmerTextNormedEmbedding`) —
    /// and it cannot be folded into the matrix, because the DFlash drafter shares that matrix
    /// unnormed. A port that folds it is correct until S6.
    fn hidden_state(&mut self, token: u32, pos: usize) -> Result<()> {
        ensure!(
            (token as usize) < self.vocab,
            "token {token} is past this model's vocabulary of {}",
            self.vocab
        );
        // **Cleared HERE, not only set in `sample`.** A `decode` that errors mid-layer leaves the
        // previous position's logits in the buffer, and a caller that logs the error and reads
        // `logits()` would get a plausible vector for the wrong position. Clearing at the start of
        // every forward makes the accessor's answer "the last COMPLETED sample or nothing"
        // (review, 2026-08-13).
        self.sampled = false;
        // SAFETY: `embed` is `[vocab, hidden]` bf16 in the pin and `token < vocab`; `x` is
        // `hidden` writable f32 this struct owns.
        unsafe {
            launch_embed_bf16_row_bcast(
                self.pin.embed.packed,
                token as usize,
                self.hidden,
                1,
                self.x.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            launch_rmsnorm_weightless_batch(
                self.x.ptr() as *mut f32,
                1,
                self.hidden,
                self.eps_pre,
                1.0,
                NULL_STREAM,
            )?;
        }
        for l in 0..self.n_layers {
            self.layer(l, pos)?;
        }
        Ok(())
    }

    /// The logit path: final norm, head, softcap, argmax.
    ///
    /// **The final norm is the PLAIN form** (`x·rsqrt(mean(x²)+eps)·w`), not the centered one the
    /// four per-layer norms take. Two launchers rather than one with a flag, and the reason is
    /// that the wrong substitution here is silent in one direction: a centered norm's weight is
    /// initialised to zeros, so `plain(centered_weight)` multiplies the residual stream by ≈0 and
    /// crashes into garbage, while `centered(plain_weight)` scales by ≈2 and stays fluent.
    ///
    /// **The softcap cannot move the argmax below it**, so this function's return value is the
    /// same with or without it. It is here because every probability depends on it — and that is
    /// why a greedy gate cannot be this path's evidence.
    fn sample(&mut self) -> Result<u32> {
        // SAFETY: `x` is the `hidden` f32 the layer loop left; `final_norm` is `hidden` f32 in the
        // pin; `logits` is `vocab` writable f32; `pick` is 8 bytes taking one i32 then one f32.
        unsafe {
            launch_rmsnorm_single(
                self.x.ptr() as *const f32,
                self.pin.final_norm,
                self.hidden,
                self.eps_pre,
                self.xn.ptr() as *mut f32,
            )?;
            let head = self.pin.head;
            proj(
                self.xn.ptr() as *const f32,
                &head,
                self.logits.ptr() as *mut f32,
                self.hidden,
            )?;
            launch_logit_softcap(
                self.logits.ptr() as *mut f32,
                self.vocab,
                self.mult,
                self.cap,
                NULL_STREAM,
            )?;
            launch_argmax(
                self.logits.ptr() as *const f32,
                self.vocab,
                self.pick.ptr() as *mut i32,
                (self.pick.ptr() as *mut f32).add(1),
            )?;
        }
        device_sync()?;
        self.sampled = true;
        let mut out = Vec::new();
        self.pick.copy_out_prefix(&mut out, 8)?;
        let idx = i32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        let val = f32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        // **The VALUE is the fault detector, and the index is not.** `argmax_reduce` initialises
        // `bidx = 0` and `amax_combine` only ever assigns one of its two input indices, so
        // `out_idx` is always in `[0, n)` — an all-NaN logit vector returns index 0 with no
        // complaint. This read `ensure!(idx >= 0, "argmax found no finite logit")` for one commit,
        // which is a branch that cannot be taken carrying the message of one that matters: it
        // deleted the ONLY post-final-layer fault detector this model has, one launch after
        // `logit_softcap` deliberately passes non-finite values through to preserve it. GLM's tail
        // (`gpu.rs`) bails on the value and keeps the index as a debug assertion; so does this now.
        // Found by review, 2026-08-13.
        debug_assert!(idx >= 0, "argmax returned {idx}");
        ensure!(
            val.is_finite(),
            "the head produced a non-finite winning logit ({val}) at token index {idx} — every \
             candidate was NaN or Inf, and an argmax over those picks index 0 silently"
        );
        ensure!(
            (idx as usize) < self.vocab,
            "argmax returned {idx}, past the vocabulary of {}",
            self.vocab
        );
        Ok(idx as u32)
    }

    /// The logit vector the last [`Self::sample`] produced — post-softcap, `vocab` long.
    ///
    /// **The only way to see this model's logit path at all.** `output_multiplier` and
    /// `final_logit_softcapping` are argmax-invariant by construction, so a greedy gate, a
    /// teacher-forced argmax check and a byte-identical-output comparison are all blind to their
    /// being wrong or absent; every probability, NLL and confidence value is wrong regardless.
    /// G3's probability-space check reads this, and so does `tests/glimmer_chain.rs`, which scores
    /// the whole chain against a host reference — a 12-way argmax is one integer of evidence about
    /// a 52-layer composition.
    ///
    /// **Not an instrument in the sense `CLAUDE.md` puts behind a feature and a flag.** Those are
    /// gated because they perturb what they measure — `--pred-probe` puts a blocking D2H on the
    /// per-layer path. This is a D2H of a buffer that already exists, after the `device_sync`
    /// `sample` already performs, and nothing on the decode path calls it.
    pub fn logits(&self) -> Result<Vec<f32>> {
        // `DeviceBuf::new` is a bare `hipMalloc` with no zero-fill, so before the first `sample`
        // this buffer is whatever the allocator handed back. Returning that is worse than useless
        // for the one thing this accessor exists for — a garbage-but-plausible logit vector is
        // exactly what G3's probability-space check is meant to catch (review, 2026-08-13).
        ensure!(
            self.sampled,
            "no logits yet: `sample` has not run on this engine, so this buffer holds whatever \
             `hipMalloc` returned"
        );
        read_f32(&self.logits, self.vocab)
    }

    /// The branch buffer as the last completed layer left it — its post-FFN norm output, `hidden`
    /// long, before the residual add.
    ///
    /// **This is the only window onto an INTERMEDIATE, and it exists for one defect.** The two
    /// epsilons are 1e-5 and 1e-8, assigned by position, and a transposition is invisible to every
    /// whole-chain gate in this tree — measured, on the synthetic fixture AND on Muse Glimmer's own
    /// weights. The reason is the residual add: the branch enters a stream that dominates it, so by
    /// the logits the difference is ~5e-6. One layer upstream it is 41.8-56.6x
    /// (`tests/glimmer_norm.rs`, on the reference's own captures), and that is where this reads.
    ///
    /// Same argument as [`Self::logits`] for why it is a plain accessor rather than an instrument
    /// behind a feature and a flag: a D2H of a buffer that already exists, after a sync that has
    /// already happened, and nothing on the decode path calls it.
    ///
    /// **Which layer's branch it holds is the CALLER's arrangement** — the engine keeps one branch
    /// buffer and every layer overwrites it, so reading a particular layer means running a model
    /// truncated to end there. `tests/glimmer_reference.rs` does exactly that, which is also why
    /// this needs no layer argument.
    pub fn branch(&self) -> Result<Vec<f32>> {
        ensure!(
            self.sampled,
            "no branch yet: no forward has completed on this engine"
        );
        read_f32(&self.br, self.hidden)
    }

    /// Greedy decode: consume `prompt`, then emit until an `eos` or `max_new`.
    ///
    /// **Prefill is token-major — one position per forward, `tq == 1` throughout.** That is what
    /// lets every sliding layer's ring be exactly `sliding_window` slots, since the launcher's
    /// floor is `win + tq - 1`. It is also the slow way: layer-major prefill is 2.15x on the GLM
    /// path and a streamed Glimmer layer is 967.942 MB of memcpy per token, so token-major prefill
    /// re-streams the whole model for every prompt token. **S5's, not S3's** — a wider `tq` needs
    /// a ring of `win + tq - 1` and a `q` buffer of `tq · hq · hd`, and neither is a change to the
    /// chain above.
    // ponytail: token-major prefill, layer-major when S5 prices the wider ring.
    pub fn decode(&mut self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            !prompt.is_empty(),
            "the prompt is empty — nothing to decode from"
        );
        // `checked_add`, because the sum is the ONLY thing standing between a caller's numbers and
        // a device write past the end of `kc[l]`: a wrapped sum satisfies this bound and then
        // `slot = pos` on a full layer indexes wherever it likes. Release builds compile the
        // overflow check out, so the guard has to be the explicit one.
        let need = prompt
            .len()
            .checked_add(max_new)
            .filter(|n| *n <= self.n_ctx);
        ensure!(
            need.is_some(),
            "{} prompt tokens plus {max_new} new is past the {} positions this engine's KV cache \
             was built for",
            prompt.len(),
            self.n_ctx
        );
        // **Every error path joins the device before returning.** `Glimmer` has no `Drop`, so its
        // fields drop in declaration order and `pin` is FIRST — `DeviceTier`'s `VmmBuf` calls
        // `hipMemUnmap`/`hipMemRelease` with no synchronisation, while the activation `DeviceBuf`s
        // whose `hipFree` would implicitly join drop AFTER it. So an error returned mid-layer
        // unmaps the weight slab with that layer's kernels still in flight. The success path is
        // already joined by `sample`; this covers the rest (review, 2026-08-13).
        let r = self.decode_inner(prompt, max_new, eos);
        if r.is_err() {
            let _ = device_sync();
        }
        r
    }

    /// [`Self::decode`]'s body. Split out so the join above covers every `?` in it rather than
    /// each one carrying its own.
    fn decode_inner(&mut self, prompt: &[u32], max_new: usize, eos: &[u32]) -> Result<Vec<u32>> {
        let mut next = 0u32;
        for (pos, &t) in prompt.iter().enumerate() {
            self.hidden_state(t, pos)?;
            // The last prompt position is the only one whose logits are wanted; the rest run for
            // their KV cache alone. Sampling them all would cost a 2.69 GB head GEMM per prompt
            // token for a value nothing reads.
            if pos + 1 == prompt.len() {
                next = self.sample()?;
            } else {
                device_sync()?;
            }
        }
        // **The stop token is NOT part of the output**, matching `GpuEngine::generate`'s `emit`
        // (`gpu.rs`), which returns before pushing. The first version pushed and then tested, so an
        // EOS-terminated run rendered the marker into the printed completion and reported one token
        // more than it produced — two decode drivers in one binary disagreeing about whether the
        // terminator is output, which a `serve.rs` port or a golden comparison inherits silently
        // (review, 2026-08-13).
        let mut out = Vec::with_capacity(max_new);
        for i in 0..max_new {
            if eos.contains(&next) {
                break;
            }
            out.push(next);
            if i + 1 == max_new {
                break;
            }
            self.hidden_state(next, prompt.len() + i)?;
            next = self.sample()?;
        }
        Ok(out)
    }
}
