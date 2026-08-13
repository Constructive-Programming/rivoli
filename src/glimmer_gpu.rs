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
//! inverted. Six of the launchers this chain needs (`rope_split_half`, `rmsnorm_single`, `vadd`,
//! `argmax`, `rope_interleave`, `swiglu`'s siblings) take no stream parameter at all, so a
//! "compute stream" loop would be exactly that inversion — a real stream at the launches that
//! accept one and the legacy default at the six that cannot.
//!
//! So the whole chain is null, uniformly, and the legacy stream's own ordering carries the
//! dependencies. `f4gpu.rs::pre_norm` is the precedent and makes the same argument. **The cost is
//! that nothing overlaps**, which is S5's to change — and the change is not "pass a stream here",
//! it is giving those six launchers one first. Until then a stream argument in this file would be
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
use crate::memory::pin::{Bf16Weight, GlimmerPin};
use anyhow::{Result, ensure};

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
pub fn window_of(kind: LayerKind, win: usize, n_ctx: usize, pos: usize) -> Window {
    match kind {
        LayerKind::SlidingAttention => {
            // The ring is `sliding_window` slots — the launcher's floor of `win + tq - 1` at the
            // `tq == 1` this file decodes at. Clamped to `n_ctx` because a run shorter than the
            // window would otherwise allocate slots no position can reach.
            let cap = win.min(n_ctx);
            Window {
                win,
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

/// The two scales the weightless QK-norm takes, which are NOT the same number.
///
/// **Q gets `qk_scale_factor` (3.87) and K gets 1.0** — `glimmer-architecture.md` trap 3. Returned
/// as a named pair rather than passed positionally so that swapping them is a rename rather than
/// an argument reorder: nothing downstream of a K scaled by 3.87 reads as an error, and no numeric
/// test in this tree can see which scale a caller chose, because every scoring path hands the same
/// one to the kernel and to the oracle.
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
        ensure!(
            n_ctx <= gt.max_position_embeddings,
            "n_ctx {n_ctx} is past this model's {} trained positions",
            gt.max_position_embeddings
        );
        let pin = GlimmerPin::build(dir, gt, budget)?;
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
            let slots = match window_of(k, gt.sliding_window, n_ctx, 0).ring_cap {
                0 => n_ctx,
                cap => cap,
            };
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
    fn attention(&mut self, l: usize, pos: usize, w: &Window) -> Result<()> {
        let (qd, kvd) = (self.hq * self.hd, self.hkv * self.hd);
        let p = self.pin.layer(l)?;
        let (wq, wk, wv, wo, wg) = (p.q, p.k, p.v, p.o, p.attn_gate);
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
            // **The 3.87 is Q's alone**, and it arrives here by NAME — see [`QkScale`]. Written as
            // `s.q` and `s.k` rather than as `self.qk_scale` and a literal `1.0` so that the
            // trap-3 swap is a rename: `nothing downstream of a K scaled by 3.87 reads as an
            // error, and no numeric test in this tree can see which scale a caller chose.
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
    fn mlp(&mut self, l: usize) -> Result<()> {
        let (wg, wu, wd) = {
            let p = self.pin.layer(l)?;
            (p.mlp_gate, p.mlp_up, p.mlp_down)
        };
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
    fn layer(&mut self, l: usize, pos: usize) -> Result<()> {
        let w = self.attn_window(l, pos);
        let (input_ln, post_attn_ln, pre_ffn_ln, post_ffn_ln) = {
            let p = self.pin.layer(l)?;
            (p.input_ln, p.post_attn_ln, p.pre_ffn_ln, p.post_ffn_ln)
        };
        // SAFETY: the four norm weights are `hidden` f32 in the pin, valid until the next
        // `pin.layer` call — which the calls below make, so each is used before the one that could
        // move it. That ordering is the borrow this block cannot express and the fence in
        // `GlimmerPin::layer` is what makes safe: it syncs before any refill.
        unsafe { self.pre_norm(input_ln)? };
        self.attention(l, pos, &w)?;
        // SAFETY: as above.
        unsafe { self.branch_add(post_attn_ln)? };
        // SAFETY: as above.
        unsafe { self.pre_norm(pre_ffn_ln)? };
        self.mlp(l)?;
        // SAFETY: as above.
        unsafe { self.branch_add(post_ffn_ln) }
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
        let mut out = Vec::new();
        self.pick.copy_out_prefix(&mut out, 4)?;
        let idx = i32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        ensure!(
            idx >= 0,
            "argmax found no finite logit — every candidate was NaN or the head produced none"
        );
        Ok(idx as u32)
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
        ensure!(
            prompt.len() + max_new <= self.n_ctx,
            "{} prompt tokens plus {max_new} new is past the {} positions this engine's KV cache \
             was built for",
            prompt.len(),
            self.n_ctx
        );
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
        let mut out = Vec::with_capacity(max_new);
        for i in 0..max_new {
            out.push(next);
            if eos.contains(&next) {
                break;
            }
            if i + 1 == max_new {
                break;
            }
            self.hidden_state(next, prompt.len() + i)?;
            next = self.sample()?;
        }
        Ok(out)
    }
}
