//! Muse Glimmer's forward pass: one layer's sandwich-normed attention and MLP, the
//! layer-major prefill that feeds them, and the logit tail. Ported from
//! `old:src/glimmer_gpu.rs`.
//!
//! Everything here consumes [`GlimmerPin::layer`](super::pin::GlimmerPin::layer)'s contract
//! and the launcher family. Nothing here decides residency: `layer(l)` returns the same shape
//! whether the budget pinned that layer or streamed it, and the write-after-read fence that
//! makes streaming safe lives inside it.
//!
//! # This loop runs entirely on the NULL stream, and that is a decision
//!
//! `old:tests/glimmer_stream_order.rs` measures what a null stream costs a consumer whose
//! producer is on a real one: 99.9% of a 2M-element operand read stale. The conclusion it
//! draws is **not** "never pass null" — it is that every launch touching one buffer must be on
//! ONE stream, and that a compute stream at two call sites inside an otherwise null-stream
//! layer is the same bug inverted. **Four of the launchers this chain drives take no stream
//! parameter at all** — `rmsnorm_single`, `rope_split_half`, `vadd` and `argmax` — so a
//! "compute stream" loop would be exactly that inversion: a real stream at the launches that
//! accept one and the legacy default at the four that cannot.
//!
//! So the whole chain is null, uniformly, and the legacy stream's own ordering carries the
//! dependencies. **The cost is that nothing overlaps**, and the change is not "pass a stream
//! here", it is giving those four launchers one first. Until then a stream argument in this
//! file would be decoration that reads as a guarantee.
//!
//! > `old:` carried this argument with SIX launchers named, two of them wrong —
//! > `rope_interleave` is the INTERLEAVED convention, the one this model must never use, and
//! > writing it into the inventory put a forbidden kernel on the next milestone's to-do list.
//! > The decision is unaffected; a closure gets inherited together with its justification, so
//! > the corrected list is the one that travels.

use super::engine::GlimmerEngine;
use super::geometry::{PREFILL_CHUNK, Window, qk_scales};
use super::pin::{Bf16Weight, GlimmerLayerPin};
use anyhow::{Result, ensure};
use rivoli_backend::{
    NULL_STREAM, device_sync, launch_argmax, launch_embed_bf16_row_bcast, launch_gemm_bf16,
    launch_gqa_attend, launch_logit_softcap, launch_rmsnorm_centered_single, launch_rmsnorm_single,
    launch_rmsnorm_weightless_batch, launch_rope_split_half, launch_sigmoid_gate, launch_swiglu,
    launch_vadd,
};

/// One bf16 projection applied to a single row: `out = w · x`, `w` being `[o_dim, i_dim]`.
///
/// A GEMM at `m = 1` rather than a GEMV because there is no bf16 GEMV in the tree and adding
/// one is arithmetic to price, not to assume. The shape check is here rather than trusted:
/// every caller below passes a buffer sized from the config, and a projection whose `i_dim`
/// disagrees with the activation it is handed reads past the end of a live allocation and
/// stays fluent.
///
/// # Safety
/// `x` is `w.i_dim` live f32, `out` is `w.o_dim` writable f32, the two do not alias, and both
/// outlive the next [`device_sync`].
unsafe fn proj(x: *const f32, w: &Bf16Weight, out: *mut f32, i_dim: usize) -> Result<()> {
    ensure!(
        w.i_dim == i_dim,
        "projection expects an activation of {} but was handed {i_dim} — a shape mismatch \
         here reads past the end of a live buffer and produces fluent wrong text",
        w.i_dim
    );
    // SAFETY: the caller's contract, plus the `i_dim` agreement checked above.
    unsafe { launch_gemm_bf16(x, w.packed, out, 1, w.o_dim, w.i_dim, NULL_STREAM) }
}

/// One layer's twelve device handles, resolved from the pin ONCE per layer.
///
/// **This exists to make [`GlimmerPin::layer`](super::pin::GlimmerPin::layer)'s narrowest rule
/// structural.** That contract says in as many words: *do not launch from pointers captured
/// across a `layer()` call* — the fence it performs syncs kernels already enqueued and cannot
/// see a caller that captures layer `l`'s addresses, requests another layer, and only then
/// launches. `old:`'s first loop requested the pin three times per layer and launched from the
/// first request's pointers after the second and third, which was safe only because all three
/// asked for the SAME `l` and a repeat request takes the hit path. Safe by accident is the
/// state a prefetch turns into wrong text, so the request happens once and the handles are
/// passed down.
///
/// It also fixes a live measurement defect: the pin's `hits` counter is documented as the way
/// to tell a working partition from a thrashing one, and three requests per layer inflated it
/// threefold.
///
/// A copy of [`GlimmerLayerPin`] rather than a borrow, because holding the borrow would keep
/// `pin` mutably borrowed across every `&mut self` call in the layer body.
pub(super) struct Handles {
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

/// Where in the model a launch is: which layer, and the absolute position of the token row.
///
/// **Two bare `usize` side by side that index different axes.** `layer(x, &h, l, pos)`
/// transposed still compiles, still addresses real memory, and produces fluent wrong text —
/// the KV slot would be derived from the layer id and the weights indexed by position. Named
/// fields move that mistake to the construction site, where it has a name. Same argument, and
/// the same shape, as `glm::forward`'s `Rows`.
///
/// CodeScene also reads it: `attention` and `layer` were 5-argument functions before this
/// (Excess Number of Function Arguments, 1.0 penalty). **A tuple parameter would have scored
/// the same green and fixed nothing** — `(usize, usize)` is exactly the transposition hazard
/// with a shorter spelling.
#[derive(Clone, Copy)]
pub(super) struct At {
    pub layer: usize,
    pub pos: usize,
}

/// The first `n` f32 of a device buffer, on the host.
///
/// Shared by the two readbacks — jscpd rejected the second copy in `old:`, and it is right
/// about the substance too: the `copy_out_prefix` length and the `chunks_exact` width are one
/// fact written twice, and a pair that disagreed would return a shorter vector rather than an
/// error.
fn read_f32(b: &crate::device::DeviceBuf, n: usize) -> Result<Vec<f32>> {
    let mut raw = Vec::new();
    b.copy_out_prefix(&mut raw, n * 4)?;
    Ok(raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

impl GlimmerEngine<'_> {
    /// The attention block for layer `l` at absolute position `pos`, reading `xn` and leaving
    /// the gated, projected output in `br`.
    ///
    /// Order is **norm → scale → rope → cache → attend**, so the cache holds post-QK-norm,
    /// post-RoPE keys. Each projection writes where its result is consumed: `k` and `v` go
    /// STRAIGHT into their cache slot, so the norm and the rotation run in place on the cache
    /// and no copy exists to get wrong.
    fn attention(&mut self, at: At, w: &Window, h: &Handles) -> Result<()> {
        let (qd, kvd) = (self.hq * self.hd, self.hkv * self.hd);
        let xn = self.xn.ptr() as *const f32;
        // SAFETY: `w.slot` is inside this layer's cache by `attn_window`'s construction, which
        // is also what sized the allocation.
        let kslot = unsafe { (self.kc[at.layer].ptr() as *mut f32).add(w.slot * kvd) };
        // SAFETY: as above.
        let vslot = unsafe { (self.vc[at.layer].ptr() as *mut f32).add(w.slot * kvd) };
        // SAFETY: `xn` is `hidden` live f32 written by this layer's pre-norm; each destination
        // is its projection's `o_dim` f32 inside a live allocation — `q` its own buffer,
        // `k`/`v` one `hkv·hd` slot of a cache with at least `slot+1` slots. None aliases `xn`
        // or another. All outlive the `device_sync` in `sample`.
        unsafe {
            proj(xn, &h.q, self.q.ptr() as *mut f32, self.cfg.hidden)?;
            proj(xn, &h.k, kslot, self.cfg.hidden)?;
            proj(xn, &h.v, vslot, self.cfg.hidden)?;
            // **The 3.87 is Q's alone**, and it arrives here by NAME — see `qk_scales`, which
            // also records what a swap actually costs (nothing: only the product enters the
            // score). What is NOT free is dropping or doubling the factor.
            let s = qk_scales(self.cfg.qk_scale_factor as f32);
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
            // NoPE: the full-attention layers carry no rotation at all. `rotated` is the
            // boolean `layer_rope_theta` really is; the base comes from the ONE
            // `rope_parameters` table.
            if self.rotated[at.layer] {
                let theta = self.cfg.rope_parameters.rope_theta;
                let qp = self.q.ptr() as *mut f32;
                launch_rope_split_half(qp, self.hq, self.hd, self.hd, at.pos, theta)?;
                launch_rope_split_half(kslot, self.hkv, self.hd, self.hd, at.pos, theta)?;
            }
            launch_gqa_attend(
                self.q.ptr() as *const f32,
                self.kc[at.layer].ptr() as *const f32,
                self.vc[at.layer].ptr() as *const f32,
                self.hq,
                self.hkv,
                self.hd,
                1,
                at.pos,
                w.win,
                w.ring_cap,
                self.attn_scale,
                self.attn.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            // **The gate reads the layer input, not the attend output.** `attn_gate` consumes
            // `xn`; a gate built from `self.attn` has the right shapes, the right tensor and
            // the wrong model.
            proj(
                xn,
                &h.attn_gate,
                self.gate.ptr() as *mut f32,
                self.cfg.hidden,
            )?;
            launch_sigmoid_gate(
                self.attn.ptr() as *mut f32,
                self.gate.ptr() as *const f32,
                qd,
                NULL_STREAM,
            )?;
            proj(
                self.attn.ptr() as *const f32,
                &h.o,
                self.br.ptr() as *mut f32,
                qd,
            )
        }
    }

    /// A sandwich norm's PRE half: `xn = centered_norm(x, w, eps_pre)`, leaving the residual
    /// stream untouched for the add that closes the block.
    ///
    /// # Safety
    /// `w` is `hidden` live f32 and stays valid until the next `pin.layer` call.
    unsafe fn pre_norm(&mut self, x: *mut f32, w: *const f32) -> Result<()> {
        // SAFETY: `x` is `hidden` live f32 and `xn` is this struct's own; `w` is the caller's
        // obligation.
        unsafe {
            launch_rmsnorm_centered_single(
                x,
                w,
                self.cfg.hidden,
                self.eps_pre,
                self.xn.ptr() as *mut f32,
                NULL_STREAM,
            )
        }
    }

    /// A sandwich norm's POST half: `br = centered_norm(br, w, eps_post); x += br`.
    ///
    /// **This is the shape of the whole trap.** The norm runs on the BRANCH, before the
    /// residual add — not on the sum, and not on the stream. One function rather than two
    /// copies because the attention block and the MLP close identically, and two copies of
    /// "which operand does the post-norm take" is the pair that can drift with only one of
    /// them wrong.
    ///
    /// `eps_post` is 1e-8 against `eps_pre`'s 1e-5, and the assignment is by POSITION: both
    /// halves take it from the field rather than from a parameter, so no call site can pass
    /// the other.
    ///
    /// # Safety
    /// `w` is `hidden` live f32 and stays valid until the next `pin.layer` call.
    unsafe fn branch_add(&mut self, x: *mut f32, w: *const f32) -> Result<()> {
        // SAFETY: `br` and `x` are each `hidden` f32 this struct owns; the centered norm's own
        // contract permits `y` aliasing `x`, which is what makes the in-place form legal.
        unsafe {
            launch_rmsnorm_centered_single(
                self.br.ptr() as *const f32,
                w,
                self.cfg.hidden,
                self.eps_post,
                self.br.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            launch_vadd(x, self.br.ptr() as *const f32, self.cfg.hidden)
        }
    }

    /// The MLP: `down(silu(gate(xn)) · up(xn))`, leaving its output in `br` for the post-norm.
    fn mlp(&mut self, h: &Handles) -> Result<()> {
        let xn = self.xn.ptr() as *const f32;
        let (hidden, inter) = (self.cfg.hidden, self.cfg.inter);
        // SAFETY: `xn` is `hidden` live f32 from the pre-norm; `mg`/`mu`/`mh` are each `inter`
        // and `br` is `hidden`, all owned by this struct. `swiglu` writes `mh`, which aliases
        // neither of its operands. The three weights are live until the next `pin.layer` call.
        unsafe {
            proj(xn, &h.mlp_gate, self.mg.ptr() as *mut f32, hidden)?;
            proj(xn, &h.mlp_up, self.mu.ptr() as *mut f32, hidden)?;
            launch_swiglu(
                self.mg.ptr() as *const f32,
                self.mu.ptr() as *const f32,
                inter,
                self.mh.ptr() as *mut f32,
                NULL_STREAM,
            )?;
            proj(
                self.mh.ptr() as *const f32,
                &h.mlp_down,
                self.br.ptr() as *mut f32,
                inter,
            )
        }
    }

    /// One decoder layer: sandwich norms around an attention block and an MLP.
    ///
    /// # The pin invariant this body relies on, stated as what actually holds
    ///
    /// The twelve device handles in `h` are resolved from the pin ONCE and then held across
    /// every launch below. The pin's own doc names that as the hazard its fence does NOT
    /// cover: the fence syncs kernels already enqueued, and cannot see a caller that captures
    /// layer `l`'s pointers, calls `layer(l + 1)`, and only then launches.
    ///
    /// **The invariant is the CALLER's, and that is the thing to preserve: no `pin.layer`
    /// request may be issued between resolving `h` and the last launch that reads it.** This
    /// function makes no pin calls at all, so nothing it does can violate the rule — which
    /// means reading the rule here and checking only this body proves nothing.
    /// [`Self::prefill`] resolves `h` once and then runs up to [`PREFILL_CHUNK`] layer bodies
    /// from it; [`Self::hidden_state`] resolves it per layer. A prefetch of `l + 1` from
    /// inside `prefill`'s inner loop would satisfy every word of a rule stated about `layer`
    /// and break the real one — position-dependent wrong text, no compile error, no fence to
    /// catch it. The `Handles` resolution is the only thing keeping that unreachable today.
    fn layer(&mut self, x: *mut f32, h: &Handles, at: At) -> Result<()> {
        let w = self.attn_window(at.layer, at.pos)?;
        // SAFETY: every pointer in `h` is live in the pin and stays at its address for as long
        // as its slot holds layer `l` — which is the whole of this function, per the section
        // above.
        unsafe { self.pre_norm(x, h.input_ln)? };
        self.attention(at, &w, h)?;
        // SAFETY: as above.
        unsafe { self.branch_add(x, h.post_attn_ln)? };
        // SAFETY: as above.
        unsafe { self.pre_norm(x, h.pre_ffn_ln)? };
        self.mlp(h)?;
        // SAFETY: as above.
        unsafe { self.branch_add(x, h.post_ffn_ln) }
    }

    /// Look `token` up in the embedding matrix and leave the NORMED row in `dst`.
    ///
    /// `dst` rather than `self.x`, because the prefill writes one row of `xs` per position and
    /// runs no layers here.
    ///
    /// **The embedding is normed, by the weightless form** — and it cannot be folded into the
    /// matrix, because the drafter shares that matrix unnormed. A port that folds it is
    /// correct until the drafter lands.
    fn embed(&mut self, token: u32, dst: *mut f32) -> Result<()> {
        ensure!(
            (token as usize) < self.cfg.vocab,
            "token {token} is past this model's vocabulary of {}",
            self.cfg.vocab
        );
        // **Cleared HERE, not only set in `sample`.** A forward that errors mid-layer leaves
        // the previous position's logits in the buffer, and a caller that logs the error and
        // reads `logits()` would get a plausible vector for the wrong position. Clearing at
        // the start of every forward makes the accessor's answer "the last COMPLETED sample
        // or nothing".
        self.sampled = false;
        let hidden = self.cfg.hidden;
        // SAFETY: `embed` is `[vocab, hidden]` bf16 in the pin and `token < vocab`; `dst` is
        // `hidden` writable f32, the caller's obligation.
        unsafe {
            launch_embed_bf16_row_bcast(
                self.pin.embed.packed,
                token as usize,
                hidden,
                1,
                dst,
                NULL_STREAM,
            )?;
            launch_rmsnorm_weightless_batch(dst, 1, hidden, self.eps_pre, 1.0, NULL_STREAM)
        }
    }

    /// One token at `pos`, every layer, leaving the hidden state in `x`.
    pub(super) fn hidden_state(&mut self, token: u32, pos: usize) -> Result<()> {
        let x = self.x.ptr() as *mut f32;
        self.embed(token, x)?;
        for l in 0..self.cfg.n_layers {
            let h = Handles::of(self.pin.layer(l)?);
            self.layer(x, &h, At { layer: l, pos })?;
        }
        Ok(())
    }

    /// **The prompt, LAYER-MAJOR: every token through layer `l` before any token reaches
    /// `l+1`.**
    ///
    /// Identical arithmetic to running the tokens one at a time — the same launches with the
    /// same arguments, only reordered, so the logits are bit-for-bit what a token-major loop
    /// produced.
    ///
    /// **What it buys is the residency, and that is the whole point.** `pin.layer(l)` is
    /// called ONCE per layer per chunk instead of once per layer per token, so a streamed
    /// layer is fetched once for the whole chunk rather than once for every position — and a
    /// streamed Glimmer layer is a synchronous **967.942 MB** host memcpy. At a 2048-token
    /// prompt with 39 layers streaming that is the difference between ~77 TB of memcpy and
    /// ~38 GB.
    ///
    /// **It does NOT batch the math.** Every projection is still a GEMM at `m = 1` and the
    /// attend is still `tq = 1`, which is why the rings stay at their clamped window and the
    /// launcher's `ring_cap >= win + tq - 1` union hazard is not reachable from here. Batching
    /// the math is a further step: it needs a rows dimension on the centered norm and a
    /// per-row position on the rope, and it changes the arithmetic — so it belongs to whoever
    /// can re-measure the gates.
    // ponytail: reorder only; batch the math when someone can price the kernel changes.
    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        ensure!(!tokens.is_empty(), "an empty prompt has nothing to prefill");
        for (c, chunk) in tokens.chunks(PREFILL_CHUNK).enumerate() {
            self.prefill_chunk(chunk, c * PREFILL_CHUNK)?;
        }
        Ok(())
    }

    /// One chunk of the prompt, layer-major: embed every position, then walk the layers.
    ///
    /// Split out of [`Self::prefill`] so the chunk loop and the layer-major walk are one
    /// statement each — CodeScene read the three-deep nest as Bumpy Road Ahead, and it is
    /// right that the two nests answer different questions: the outer one is a MEMORY bound
    /// ([`PREFILL_CHUNK`], the residual-stream trade) and this one is the residency reorder.
    ///
    /// `base` is the absolute position of `chunk[0]`. Every chunk runs EVERY layer before the
    /// next chunk starts, so layer `l`'s KV cache already holds every position below `base`
    /// when this chunk's attends run — which is what makes the chunk size free to choose.
    fn prefill_chunk(&mut self, chunk: &[u32], base: usize) -> Result<()> {
        let (xs, hidden) = (self.xs.ptr() as *mut f32, self.cfg.hidden);
        for (i, &tok) in chunk.iter().enumerate() {
            // SAFETY: `xs` is `PREFILL_CHUNK · hidden` f32 and `i < chunk.len() <= CHUNK`.
            self.embed(tok, unsafe { xs.add(i * hidden) })?;
        }
        for l in 0..self.cfg.n_layers {
            // **ONE pin request per layer per chunk** — the saving, and also what keeps the
            // captured-pointer rule satisfied: no other layer is requested between this
            // capture and the launches that read it.
            let h = Handles::of(self.pin.layer(l)?);
            for i in 0..chunk.len() {
                // SAFETY: as above.
                let at = At {
                    layer: l,
                    pos: base + i,
                };
                self.layer(unsafe { xs.add(i * hidden) }, &h, at)?;
            }
        }
        Ok(())
    }

    /// Prefill `prompt` and sample the token that follows it.
    pub(super) fn prefill_and_sample(&mut self, prompt: &[u32]) -> Result<u32> {
        self.prefill(prompt)?;
        // The head reads the last position, which is the last row of the last chunk.
        let last = (prompt.len() - 1) % PREFILL_CHUNK;
        // SAFETY: `xs` holds `PREFILL_CHUNK` rows of `hidden` f32 and `last` is inside it.
        let x = unsafe { (self.xs.ptr() as *const f32).add(last * self.cfg.hidden) };
        self.sample(x)
    }

    /// The hidden state `hidden_state` left, for the decode path's sampler.
    pub(super) fn sample_x(&mut self) -> Result<u32> {
        self.sample(self.x.ptr() as *const f32)
    }

    /// The logit path: final norm, head, softcap, argmax.
    ///
    /// **The final norm is the PLAIN form** (`x·rsqrt(mean(x²)+eps)·w`), not the centered one
    /// the four per-layer norms take. Two launchers rather than one with a flag, and the
    /// reason is that the wrong substitution here is silent in one direction: a centered
    /// norm's weight is initialised to zeros, so `plain(centered_weight)` multiplies the
    /// residual stream by ≈0 and crashes into garbage, while `centered(plain_weight)` scales
    /// by ≈2 and stays fluent.
    ///
    /// **The softcap cannot move the argmax below it**, so this function's return value is the
    /// same with or without it. It is here because every probability depends on it — and that
    /// is why a greedy gate cannot be this path's evidence.
    fn sample(&mut self, x: *const f32) -> Result<u32> {
        let (hidden, vocab) = (self.cfg.hidden, self.cfg.vocab);
        // SAFETY: `x` is the `hidden` f32 the layer loop left; `final_norm` is `hidden` f32 in
        // the pin; `logits` is `vocab` writable f32; `pick` is 8 bytes taking one i32 then one
        // f32.
        unsafe {
            launch_rmsnorm_single(
                x,
                self.pin.final_norm,
                hidden,
                self.eps_pre,
                self.xn.ptr() as *mut f32,
            )?;
            let head = self.pin.head;
            proj(
                self.xn.ptr() as *const f32,
                &head,
                self.logits.ptr() as *mut f32,
                hidden,
            )?;
            launch_logit_softcap(
                self.logits.ptr() as *mut f32,
                vocab,
                self.cfg.output_multiplier as f32,
                self.cfg.final_logit_softcapping as f32,
                NULL_STREAM,
            )?;
            launch_argmax(
                self.logits.ptr() as *const f32,
                vocab,
                self.pick.ptr() as *mut i32,
                (self.pick.ptr() as *mut f32).add(1),
            )?;
        }
        device_sync()?;
        self.sampled = true;
        self.read_pick()
    }

    /// `pick`'s two words, checked. Split out so [`Self::sample`]'s body stays the launch
    /// chain and this stays the fault analysis.
    ///
    /// **The VALUE is the fault detector, and the index is not.** `argmax_reduce` initialises
    /// its index to 0 and only ever assigns one of its two candidate indices, so the result is
    /// always in `[0, n)` — an all-NaN logit vector returns index 0 with no complaint.
    /// `old:` read `ensure!(idx >= 0, "argmax found no finite logit")` for one commit, which
    /// is a branch that cannot be taken carrying the message of one that matters: it deleted
    /// the ONLY post-final-layer fault detector this model has, one launch after
    /// `logit_softcap` deliberately passes non-finite values through to preserve it.
    fn read_pick(&self) -> Result<u32> {
        let mut out = Vec::new();
        self.pick.copy_out_prefix(&mut out, 8)?;
        let idx = i32::from_le_bytes([out[0], out[1], out[2], out[3]]);
        let val = f32::from_le_bytes([out[4], out[5], out[6], out[7]]);
        debug_assert!(idx >= 0, "argmax returned {idx}");
        ensure!(
            val.is_finite(),
            "the head produced a non-finite winning logit ({val}) at token index {idx} — \
             every candidate was NaN or Inf, and an argmax over those picks index 0 silently"
        );
        ensure!(
            (idx as usize) < self.cfg.vocab,
            "argmax returned {idx}, past the vocabulary of {}",
            self.cfg.vocab
        );
        Ok(idx as u32)
    }

    /// The logit vector the last sample produced — post-softcap, `vocab` long.
    ///
    /// **The only way to see this model's logit path at all.** `output_multiplier` and
    /// `final_logit_softcapping` are argmax-invariant by construction, so a greedy gate, a
    /// teacher-forced argmax check and a byte-identical-output comparison are all blind to
    /// their being wrong or absent; every probability, NLL and confidence value is wrong
    /// regardless.
    ///
    /// **Not an instrument in the sense `CLAUDE.md` puts behind a feature and a flag.** Those
    /// are gated because they perturb what they measure. This is a D2H of a buffer that
    /// already exists, after the `device_sync` `sample` already performs, and nothing on the
    /// decode path calls it.
    pub fn logits(&self) -> Result<Vec<f32>> {
        // `DeviceBuf::new` is a bare `hipMalloc` with no zero-fill, so before the first
        // `sample` this buffer is whatever the allocator handed back. Returning that is worse
        // than useless for the one thing this accessor exists for — a garbage-but-plausible
        // logit vector is exactly what a probability-space check is meant to catch.
        ensure!(
            self.sampled,
            "no logits yet: `sample` has not run on this engine, so this buffer holds \
             whatever `hipMalloc` returned"
        );
        read_f32(&self.logits, self.cfg.vocab)
    }

    /// The branch buffer as the last completed layer left it — its post-FFN norm output,
    /// `hidden` long, before the residual add.
    ///
    /// **This is the only window onto an INTERMEDIATE, and it exists for one defect.** The two
    /// epsilons are 1e-5 and 1e-8, assigned by position, and a transposition is invisible to
    /// every whole-chain gate — measured, on a synthetic fixture AND on Muse Glimmer's own
    /// weights. The reason is the residual add: the branch enters a stream that dominates it,
    /// so by the logits the difference is ~5e-6. One layer upstream it is 41.8-56.6x
    /// (`old:tests/glimmer_norm.rs`, on the reference's own captures), and that is where this
    /// reads.
    ///
    /// **Which layer's branch it holds is the CALLER's arrangement** — the engine keeps one
    /// branch buffer and every layer overwrites it, so reading a particular layer means
    /// running a model truncated to end there. That is also why this needs no layer argument.
    pub fn branch(&self) -> Result<Vec<f32>> {
        ensure!(
            self.sampled,
            "no branch yet: no forward has completed on this engine"
        );
        read_f32(&self.br, self.cfg.hidden)
    }
}
