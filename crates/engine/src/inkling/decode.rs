//! Inkling 2's decode driver: the per-position layer loop (the prefix, the core's passes with
//! their per-pass messages, the suffix), the block-residual bank's lifecycle, the routed MoE, and
//! the seam's [`decode`](InklingEngine::decode)/[`score`](InklingEngine::score) entry points.
//!
//! `forward.rs` owns the launch composition (one attention block, one MLP block); this file owns
//! the ORDER the blocks run in, which is where the loop's own traps live — **T7** a boundary's
//! residual is the attention output alone, **T8** an extra pass restarts from the pre-loop states
//! and skips the layer-`core_start` boundary, **T16** each pass has its own cache layers, **T5**
//! the router divides by `Σ scores + 1e-20`.

use super::engine::InklingEngine;
use super::forward::{
    AttnCall, BlockRes, LayerCall, LayerDims, LogitTail, MsgIo, attention, copy_dev, logit_tail, p,
    pc, proj,
};
use super::geometry::{
    is_block_boundary, is_block_end, log_scale_tau, loop_cache_offset, sconv_static_divisor,
};
use crate::seam::{Decoded, Emit, GenSpec, TokenSink};
use anyhow::{Result, ensure};
use rivoli_backend::{
    NULL_STREAM, device_sync, fill_u32, launch_argmax, launch_embed_bf16_row_bcast,
    launch_rmsnorm_single, launch_sconv, launch_situ_glu_f32, launch_vadd, launch_vaxpy,
};
use rivoli_core::routing::{RoutePolicy, RouteScratch, route_into};

/// One stored layer's application: stored layer, cache layer (T16), position, full-attention flag.
#[derive(Clone, Copy)]
pub struct LayerStep {
    pub layer: usize,
    pub cache_layer: usize,
    pub pos: usize,
    pub full: bool,
}

/// The loop's cursor over one position.
struct LoopCtx {
    pos: usize,
    block: usize,
    n_bank: usize,
}

/// One decoder layer: the attention half, then the MLP half.
///
/// **Only the attention half is stamped here.** The MLP half stamps itself — `Ffn` around
/// everything that is not the routed MoE, and the MoE's own dispatch stamps `Ffn`/`FetchWait` at
/// its awaits. Wrapping the whole half in one `Ffn` span (what this loop did before the `.mxfp8`
/// miss path grew a fetch-wait stamp) would put the fetch wait INSIDE ffn and drive
/// `PROFILE/tok`'s `other` remainder negative — the double-count its bucket-sum gate reads as a
/// broken instrument.
impl InklingEngine<'_> {
    pub(super) fn lap_ffn<T>(&mut self, t: std::time::Instant, res: Result<T>) -> Result<T> {
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        res
    }

    fn layer(&mut self, s: LayerStep, c: &mut LoopCtx, msg: Option<*const f32>) -> Result<()> {
        let t = std::time::Instant::now();
        self.attn_phase(s, c, msg)?;
        self.prof.lap(crate::telemetry::Phase::Attend, t);
        self.mlp_phase(s, (c.n_bank, msg))
    }
}

/// The attention half: the block query (or layer 0's input), the gates, the norms, the attention,
/// the post-attention convolution, and the residual/boundary rule.
impl InklingEngine<'_> {
    fn attn_phase(&mut self, s: LayerStep, c: &mut LoopCtx, msg: Option<*const f32>) -> Result<()> {
        let (n, tau, inputs) = self.attn_norms(s, c.n_bank, msg)?;
        let mut dims = LayerDims::of(self.cfg, s.full, self.n_ctx);
        dims.rel.tau = tau;
        let call = AttnCall {
            step: s,
            n,
            tau,
            inputs,
            out: p(&self.sc.branch),
        };
        // SAFETY: the pin and scratch are live for the call; the NULL stream orders the chain.
        let io = LayerCall {
            pin: &self.pin.layers[s.layer],
            sc: &self.sc,
            call: &call,
            dims: &dims,
        };
        unsafe { attention(&io)? };
        let hidden = dims.hidden;
        let (conv, state) = (
            self.pin.layers[s.layer].post_attn_conv,
            p(&self.sc.sconv[s.cache_layer][2]),
        );
        // The conv is in place: `branch` is both the input and the output.
        self.post_sconv(conv, state, pc(&self.sc.branch))?;
        if is_block_boundary(s.layer, c.block) {
            self.bank_push((c.n_bank, hidden))?;
            c.n_bank += 1;
            Ok(())
        } else {
            // SAFETY: `x` and `branch` are distinct `hidden`-f32 buffers.
            unsafe { launch_vadd(p(&self.sc.x), pc(&self.sc.branch), hidden) }
        }
    }

    /// A block boundary (T7, INCLUDING layer 0): append the layer INPUT to the bank, then make the
    /// attention output the residual — no residual add, the bank entry carries the input forward.
    fn bank_push(&mut self, (n_bank, hidden): (usize, usize)) -> Result<()> {
        // SAFETY: `block_bank` has one free row per boundary (the scratch sized one per boundary
        // plus the partial); `x`/`branch` are `hidden` f32.
        unsafe {
            copy_dev(
                (pc(&self.sc.block_bank) as *mut f32).add(n_bank * hidden),
                pc(&self.sc.x),
                hidden,
            )?;
            copy_dev(p(&self.sc.x), pc(&self.sc.branch), hidden)
        }
    }

    /// One post convolution into `branch`: this cache layer's window state, `hidden` channels,
    /// divisor `√(k+1)`. `src` is either distinct from `branch` or `branch` itself — the
    /// attention conv is in place — which the kernel's read-then-write order allows.
    fn post_sconv(&self, conv: *const f32, state: *mut f32, src: *const f32) -> Result<()> {
        let (hidden, k) = (self.cfg.hidden_size, self.cfg.sconv_kernel_size);
        let out = p(&self.sc.branch);
        // SAFETY: `state` is this cache layer's window and `src`/`out` are `hidden` f32; the
        // caller's contract covers the aliasing.
        unsafe {
            launch_sconv(
                conv,
                state,
                src,
                hidden,
                k,
                sconv_static_divisor(k),
                out,
                NULL_STREAM,
            )
        }
    }

    /// The block-residual sources a query reads: the bank, the partial row, and the `nq`-row query
    /// bank at `query`. ONE author, so no two queries can disagree about which buffers they read.
    fn block_res(&self, n_bank: usize, query: *const f32, nq: usize) -> BlockRes {
        BlockRes {
            bank: pc(&self.sc.block_bank),
            n_bank,
            partial: pc(&self.sc.x),
            query,
            nq,
        }
    }

    /// The window, τ, and the three pre-normed attention inputs.
    ///
    /// **T11 lives here.** Sliding: one block query, gates if this pass has messages, then ALL
    /// THREE gated vectors through the SAME `input_layernorm`. Full: three distinct queries,
    /// gated individually, then the three distinct q/k/v norms.
    fn attn_norms(
        &mut self,
        s: LayerStep,
        n_bank: usize,
        msg: Option<*const f32>,
    ) -> Result<(usize, Option<f32>, [*const f32; 3])> {
        let cfg = self.cfg;
        let tau = s
            .full
            .then(|| log_scale_tau(s.pos, cfg.log_scaling_alpha as f32, cfg.log_scaling_n_floor));
        let n = match s.full {
            true => s.pos + 1,
            false => (s.pos + 1).min(cfg.sliding_window_size),
        };
        // The kernel's own ceiling, not a transcribed number: `sink_attend` refuses past it
        // (1001), and `open` already refuses a `--ctx` that could reach it — this is the guard
        // that keeps the two agreeing if either moves.
        ensure!(
            n <= rivoli_backend::abi::SINK_ATTEND_MAX_N as usize,
            "position {} wants {n} keys, past the sink_attend LDS cap {}",
            s.pos,
            rivoli_backend::abi::SINK_ATTEND_MAX_N
        );
        let x = pc(&self.sc.x);
        if s.layer == 0 {
            // No `attention_input` (the boundary opens the bank); one shared norm.
            let (xn, ln0, eps) = (
                p(&self.sc.xn),
                self.pin.layers[0].input_ln[0],
                cfg.rms_norm_eps as f32,
            );
            // SAFETY: `x` is `hidden` f32 into the `hidden`-f32 `xn`.
            unsafe { launch_rmsnorm_single(x, ln0, cfg.hidden_size, eps, xn)? };
            return Ok((n, tau, [xn as *const f32; 3]));
        }
        let out = self.block_query(s, n_bank)?;
        let gated = self.gate_inputs(s, out, msg)?;
        let shared = !s.full && msg.is_none();
        Ok((n, tau, self.norm_inputs(s, gated, shared)?))
    }

    /// The block-residual query: 3 rows on a full layer, 1 on a sliding one.
    fn block_query(&mut self, s: LayerStep, n_bank: usize) -> Result<*mut f32> {
        let nq = match s.full {
            true => 3,
            false => 1,
        };
        let br = self.block_res(n_bank, self.pin.layers[s.layer].attn_query, nq);
        let out = p(&self.sc.res_out);
        self.block_res_attend(&br, out)?;
        Ok(out)
    }

    /// Apply the layer's MessageGates to the query rows in `(q, k, v)` order (T13); without
    /// messages the rows pass through untouched.
    fn gate_inputs(
        &mut self,
        s: LayerStep,
        out: *mut f32,
        msg: Option<*const f32>,
    ) -> Result<[*const f32; 3]> {
        let hidden = self.cfg.hidden_size;
        let mut gated = if s.full {
            // SAFETY: `out` holds 3 rows of `hidden` on full layers.
            unsafe {
                [
                    out as *const f32,
                    (out as *const f32).add(hidden),
                    (out as *const f32).add(2 * hidden),
                ]
            }
        } else {
            [out as *const f32; 3]
        };
        if let Some(m) = msg {
            self.apply_qkv_gates(s.layer, &mut gated, m)?;
        }
        Ok(gated)
    }

    /// Run the three attention MessageGates `(q, k, v)` in order into their temp buffers.
    fn apply_qkv_gates(
        &mut self,
        layer: usize,
        gated: &mut [*const f32; 3],
        m: *const f32,
    ) -> Result<()> {
        let hidden = self.cfg.hidden_size;
        let tmp = [
            p(&self.sc.xn_q_tmp),
            p(&self.sc.xn_k_tmp),
            p(&self.sc.xn_v_tmp),
        ];
        for which in 0..3 {
            let io = MsgIo {
                layer,
                which,
                hidden: gated[which],
                // SAFETY: the message bank holds 4 rows of `hidden` (T13).
                message: unsafe { m.add(which * hidden) },
                out: tmp[which],
            };
            self.message_gate(&io)?;
            gated[which] = tmp[which] as *const f32;
        }
        Ok(())
    }

    /// The input norms (T11): a sliding layer's ONE shared norm (a single vector when nothing was
    /// gated), a full layer's three distinct q/k/v norms.
    fn norm_inputs(
        &mut self,
        s: LayerStep,
        gated: [*const f32; 3],
        shared: bool,
    ) -> Result<[*const f32; 3]> {
        let hidden = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let lp = &self.pin.layers[s.layer];
        let norms = match s.full {
            true => [lp.input_ln[0], lp.input_ln[1], lp.input_ln[2]],
            false => [lp.input_ln[0]; 3],
        };
        let (xn, xnk, xnv) = (p(&self.sc.xn), p(&self.sc.xn_k), p(&self.sc.xn_v));
        let (dsts, count): ([*mut f32; 3], usize) = match shared {
            true => ([xn, xn, xn], 1),
            false => ([xn, xnk, xnv], 3),
        };
        for which in 0..count {
            // SAFETY: `gated[which]` is `hidden` f32 into the `hidden` writable `dsts[which]`.
            unsafe { launch_rmsnorm_single(gated[which], norms[which], hidden, eps, dsts[which])? };
        }
        Ok(dsts.map(|d| d.cast_const()))
    }
}

/// The MLP half: block-residual query, optional message gate, pre-MLP norm, the routed MoE, the
/// post-MLP convolution, and the residual add.
///
/// **Two `Ffn` spans with the MoE's own stamps between them**, so every millisecond of the half
/// lands in exactly one bucket: the MoE reports its launches as `Ffn` and, on the miss path, the
/// residual miss-stream wait as `FetchWait` (see `moe_mxfp8`). The dispatch call itself is the
/// only unstamped gap, and it is a call.
impl InklingEngine<'_> {
    fn mlp_phase(
        &mut self,
        s: LayerStep,
        (n_bank, msg): (usize, Option<*const f32>),
    ) -> Result<()> {
        let t = std::time::Instant::now();
        let cfg = self.cfg;
        let (hidden, eps) = (cfg.hidden_size, cfg.rms_norm_eps as f32);
        let lp = &self.pin.layers[s.layer];
        let (mlp_query, post_ln, post_conv, sconv3) = (
            lp.mlp_query,
            lp.post_attn_ln,
            lp.post_mlp_conv,
            p(&self.sc.sconv[s.cache_layer][3]),
        );
        let br = self.block_res(n_bank, mlp_query, 1);
        let out = p(&self.sc.res_out);
        self.block_res_attend(&br, out)?;
        let mlp_in = match msg {
            Some(m) => {
                let io = MsgIo {
                    layer: s.layer,
                    which: 3, // T13: the fourth message is the MLP's
                    hidden: out,
                    // SAFETY: the message bank holds 4 rows of `hidden`; this is the MLP's (T13).
                    message: unsafe { m.add(3 * hidden) },
                    out: p(&self.sc.xn_q_tmp),
                };
                self.message_gate(&io)?;
                pc(&self.sc.xn_q_tmp)
            }
            None => out as *const f32,
        };
        // SAFETY: `mlp_in` is `hidden` f32 into the `hidden`-f32 `xn`.
        unsafe { launch_rmsnorm_single(mlp_in, post_ln, hidden, eps, p(&self.sc.xn))? };
        let scale = self.route(s.layer)?;
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        self.moe(s.layer, scale)?;
        let t = std::time::Instant::now();
        // The post-MLP conv reads `moe_acc` and writes `branch` (distinct).
        self.post_sconv(post_conv, sconv3, pc(&self.sc.moe_acc))?;
        // SAFETY: `x` and `branch` are distinct `hidden`-f32 buffers.
        let r = unsafe { launch_vadd(p(&self.sc.x), pc(&self.sc.branch), hidden) };
        self.lap_ffn(t, r)
    }

    /// The router: a bf16 gate GEMV on the normed input, one blocking D2H, then host top-k.
    /// Returns the combine scale `route_scale / (Σ scores + 1e-20)` (T5's epsilon); the bias
    /// shifts SELECTION only.
    fn route(&mut self, layer: usize) -> Result<f32> {
        let hidden = self.cfg.hidden_size;
        let router = self.pin.layers[layer].router;
        // SAFETY: `xn` is `hidden` f32, `router_logits` is `n_routed_experts` writable.
        unsafe { proj(pc(&self.sc.xn), &router, p(&self.sc.router_logits), hidden)? };
        // The layer's one blocking readback; it also drains the null stream, so the attention half
        // has retired before the host picks experts.
        self.sc.router_logits.copy_out_into(&mut self.router_host)?;
        let policy = RoutePolicy {
            top_k: self.cfg.num_experts_per_tok,
            scoring: rivoli_core::num::Scoring::Sigmoid, // the config's `scoring_func` default, pinned (INV-1)
        };
        route_into(
            &self.router_host,
            &self.pin.router_bias_host[layer],
            policy,
            RouteScratch {
                scores: &mut self.route_scores,
                choice: &mut self.route_choice,
                sel: &mut self.route_sel,
            },
        );
        let sum: f32 = self
            .route_sel
            .iter()
            .map(|&e| self.route_scores[e])
            .sum::<f32>();
        Ok(self.cfg.route_scale as f32 / (sum + 1e-20))
    }

    /// The routed MoE — the ONE branch the two members take differently.
    ///
    /// Mini's whole-layer slabs are sliced on the host ([`Self::moe_bf16`]); Small's `.mxfp8`
    /// experts come from the pool's slots, their fills overlapped on two streams when the layer
    /// misses at least one expert and on the null stream alone when it misses none
    /// ([`Self::moe_mxfp8`]). Everything before and after this call — the router, the norms, the
    /// post-MLP convolution, the residual — is the same code for both.
    fn moe(&mut self, layer: usize, scale: f32) -> Result<()> {
        if self.pin.is_mxfp8() {
            return self.moe_mxfp8(layer, scale);
        }
        self.moe_bf16(layer, scale)
    }

    /// Mini's MoE: per expert `situ_glu(gate_up · x)`, then `down`, accumulated in fp32 with
    /// `scores[e] · scale`. The softcaps are K3's SiTU-GLU at b1=4, b2=25 — the sigmoid sees the
    /// RAW gate (T6).
    fn moe_bf16(&mut self, layer: usize, scale: f32) -> Result<()> {
        let t = std::time::Instant::now();
        let (hidden, inter) = (self.cfg.hidden_size, self.cfg.intermediate_size);
        let (b1, b2) = (
            self.cfg.mlp_gate_softcap as f32,
            self.cfg.mlp_up_softcap as f32,
        );
        let acc = p(&self.sc.moe_acc);
        // SAFETY: `moe_acc` is `hidden` f32, zeroed by an async memset on the null stream.
        unsafe { fill_u32(acc as *mut u8, 0, hidden * 4)? };
        let xn = pc(&self.sc.xn);
        // **The accumulation ORDER deliberately follows `route_sel` (descending router score),
        // while the reference torch path folds in ASCENDING expert id.** fp32 addition is not
        // associative, so the two sums differ in the last bits — and this is why the MoE is
        // scored against a relative floor rather than bit-identity. The same applies to the
        // weighting: `score[e] · (route_scale / (Σ + 1e-20))` is applied per expert here, where
        // torch normalises the scores first; the interval the tolerance carries is calibrated on
        // that difference (`docs/measurement/how-to-measure.md`).
        for i in 0..self.route_sel.len() {
            let e = self.route_sel[i];
            let w = self.route_scores[e] * scale;
            let (gate_up, down) = self.pin.expert_weights(layer, e)?;
            // SAFETY: `moe_gu` is `2·inter`, `moe_h` `inter`, `moe_out`/`moe_acc` `hidden` f32;
            // `proj` checks each weight's input width against the activation.
            unsafe {
                proj(xn, &gate_up, p(&self.sc.moe_gu), hidden)?;
                let gu = p(&self.sc.moe_gu);
                launch_situ_glu_f32(
                    gu,
                    gu.add(inter),
                    inter,
                    b1,
                    b2,
                    p(&self.sc.moe_h),
                    NULL_STREAM,
                )?;
                proj(pc(&self.sc.moe_h), &down, p(&self.sc.moe_out), inter)?;
                launch_vaxpy(acc, pc(&self.sc.moe_out), w, hidden, NULL_STREAM)
            }?;
        }
        // Launch time, not execution: nothing on this arm joins before the next router
        // readback, so the span is the enqueue cost — the same semantics GLM's resident half
        // carries, and the reason Mini's `fetch-wait` stays 0 (its experts are pinned weights,
        // not pooled fills).
        self.lap_ffn(t, Ok(()))
    }
}

/// The per-position loop: embedding, the prefix, the core's passes with their messages, the
/// suffix, and the final block-residual query.
impl InklingEngine<'_> {
    /// Run one position's whole stack, leaving the final hidden state in `res_out`.
    pub(super) fn hidden_state(&mut self, tok: u32, pos: usize) -> Result<()> {
        let (hidden, vocab) = (self.cfg.hidden_size, self.cfg.vocab_size);
        ensure!(
            (tok as usize) < vocab,
            "token id {tok} names no embedding row (the vocab is {vocab})"
        );
        ensure!(
            pos < self.n_ctx,
            "position {pos} is outside the {} positions this KV cache was sized for",
            self.n_ctx
        );
        self.sampled = false;
        let eps = self.cfg.rms_norm_eps as f32;
        // SAFETY: the embed table is `[vocab, hidden]` bf16 with `tok < vocab` checked; `xn`/`x`
        // are `hidden` f32.
        unsafe {
            launch_embed_bf16_row_bcast(
                self.pin.globals.embed.packed,
                tok as usize,
                hidden,
                1,
                p(&self.sc.xn),
                NULL_STREAM,
            )?;
            launch_rmsnorm_single(
                p(&self.sc.xn),
                self.pin.globals.embed_norm,
                hidden,
                eps,
                p(&self.sc.x),
            )?;
        }
        let n_bank = self.run_layers(pos)?;
        let br = self.block_res(n_bank, self.pin.globals.final_query, 1);
        let out = p(&self.sc.res_out);
        self.block_res_attend(&br, out)
    }

    /// The layers: prefix, the loop's passes, suffix. Returns the bank length the final query reads.
    fn run_layers(&mut self, pos: usize) -> Result<usize> {
        let n_layers = self.cfg.num_hidden_layers;
        let block = self.cfg.n_layers_per_attn_res_block;
        let core = self
            .cfg
            .loop_config
            .as_ref()
            .map(|lc| (lc.core_start, lc.core_end, lc.num_passes));
        let mut c = LoopCtx {
            pos,
            block,
            n_bank: 0,
        };
        let (pre, post) = match core {
            Some((start, end, _)) => (start, end),
            None => (n_layers, n_layers),
        };
        for l in 0..pre {
            self.step((l, l), &mut c, None)?;
        }
        if let Some((start, end, passes)) = core {
            self.save_pre_loop(&c)?;
            for l in start..end {
                self.step((l, l), &mut c, None)?;
            }
            for loop_idx in 1..passes {
                self.replay_pass(loop_idx, (start, end), &mut c)?;
            }
        }
        for l in post..n_layers {
            self.step((l, l), &mut c, None)?;
        }
        Ok(c.n_bank)
    }

    /// One stored layer's application; `slot` is `(stored_layer, cache_layer)` (T16).
    fn step(
        &mut self,
        (layer, cache_layer): (usize, usize),
        c: &mut LoopCtx,
        msg: Option<*const f32>,
    ) -> Result<()> {
        let s = LayerStep {
            layer,
            cache_layer,
            pos: c.pos,
            full: is_block_end(layer, c.block),
        };
        self.layer(s, c, msg)
    }

    /// Snapshot the pre-loop residual and bank — every extra pass restarts from these (T8).
    fn save_pre_loop(&mut self, c: &LoopCtx) -> Result<()> {
        let hidden = self.cfg.hidden_size;
        // SAFETY: `pre_loop_x` is `hidden` f32; `pre_loop_bank` holds `n_bank` rows (the bank's own
        // length bounds it).
        unsafe {
            copy_dev(p(&self.sc.pre_loop_x), pc(&self.sc.x), hidden)?;
            copy_dev(
                p(&self.sc.pre_loop_bank),
                pc(&self.sc.block_bank),
                c.n_bank * hidden,
            )
        }
    }

    /// One extra loop pass (LOOP:537–585): snapshot the previous pass's outputs, restart from the
    /// pre-loop states, and hand every core layer its four messages.
    ///
    /// **T8 is the `pre_blocks + 1` below**: the previous pass's bank minus the layer-`core_start`
    /// boundary state is what the messages attend — pass 2 sees only the layer-12 state.
    fn replay_pass(
        &mut self,
        loop_idx: usize,
        core: (usize, usize),
        c: &mut LoopCtx,
    ) -> Result<()> {
        let (start, end) = core;
        let hidden = self.cfg.hidden_size;
        let pre_blocks = start / c.block.max(1); // `num_pre_loop_blocks` (LOOP:516)
        // SATURATING here hid a real defect class: a pass whose bank never grew past the pre-loop
        // boundary has NO previous-pass states to snapshot, and `prev_len = 0` would silently
        // hand the MessageGates an empty bank instead of saying the schedule is wrong.
        ensure!(
            c.n_bank > pre_blocks + 1,
            "loop pass {loop_idx}: no previous-pass block states beyond pre-loop boundary \
             (n_bank={}, pre_blocks={pre_blocks})",
            c.n_bank
        );
        let prev_len = c.n_bank - (pre_blocks + 1);
        // SAFETY: each destination is sized for its source: `hidden` for the residual, `prev_len`
        // rows for the snapshot, `pre_blocks` rows back out of the pre-loop copy.
        unsafe {
            copy_dev(p(&self.sc.prev_pass_x), pc(&self.sc.x), hidden)?;
            copy_dev(
                p(&self.sc.prev_pass_bank),
                pc(&self.sc.block_bank).add((pre_blocks + 1) * hidden),
                prev_len * hidden,
            )?;
            copy_dev(p(&self.sc.x), pc(&self.sc.pre_loop_x), hidden)?;
            copy_dev(
                p(&self.sc.block_bank),
                pc(&self.sc.pre_loop_bank),
                pre_blocks * hidden,
            )?;
        }
        c.n_bank = pre_blocks;
        let offset = loop_cache_offset(self.cfg.num_hidden_layers, start, end, loop_idx);
        let message_query = self.pin.globals.message_query;
        ensure!(
            !message_query.is_null(),
            "a looped member must carry model.message_input.query"
        );
        for (i, l) in (start..end).enumerate() {
            let br = BlockRes {
                bank: pc(&self.sc.prev_pass_bank),
                n_bank: prev_len,
                partial: pc(&self.sc.prev_pass_x),
                // T13: the four messages are the row block of this loop layer.
                query: unsafe { message_query.add(i * 4 * hidden) },
                nq: 4,
            };
            let msg = p(&self.sc.msg_buf);
            self.block_res_attend(&br, msg)?;
            self.step((l, l + offset), c, Some(msg))?;
        }
        Ok(())
    }
}

/// The logit path and the prefill/decode entry points.
impl InklingEngine<'_> {
    /// The final norm, the lm_head, the softcap, and the device argmax — the token's ONE join.
    pub(super) fn sample_x(&mut self) -> Result<u32> {
        let t = std::time::Instant::now();
        let cfg = self.cfg;
        let tail = LogitTail {
            norm_w: self.pin.globals.final_norm,
            head: &self.pin.globals.lm_head,
            hidden: cfg.hidden_size,
            vocab: cfg.unpadded_vocab_size,
            eps: cfg.rms_norm_eps as f32,
            cap: cfg.final_logit_softcapping as f32,
        };
        // SAFETY: `res_out` is the final query's `hidden` row, `xn` `hidden` f32, `logits` `vocab`
        // f32, and `pick` takes one i32 then one f32.
        unsafe {
            logit_tail(
                pc(&self.sc.res_out),
                p(&self.sc.xn),
                p(&self.sc.logits),
                &tail,
            )?;
            launch_argmax(
                pc(&self.sc.logits),
                cfg.unpadded_vocab_size,
                p(&self.sc.pick) as *mut i32,
                p(&self.sc.pick).add(1),
            )?;
        }
        // No `device_sync` here: `copy_out_into` is a blocking null-stream D2H `hipMemcpy`, which
        // returns only after every launch enqueued on the null stream before it has retired
        // (`device.rs::copy_out_joined` states the contract it restates). A sync here joined the
        // same work twice and is measurable on the per-token path.
        self.sc.pick.copy_out_into(&mut self.pick_host)?;
        self.prof.lap(crate::telemetry::Phase::Head, t);
        self.sampled = true;
        Ok(i32::from_le_bytes([
            self.pick_host[0],
            self.pick_host[1],
            self.pick_host[2],
            self.pick_host[3],
        ]) as u32)
    }

    /// Feed the whole prompt, then sample the token the last position predicts.
    pub(super) fn prefill_and_sample(&mut self, prompt: &[u32]) -> Result<u32> {
        ensure!(!prompt.is_empty(), "empty prompt");
        for (pos, &tok) in prompt.iter().enumerate() {
            self.hidden_state(tok, pos)?;
        }
        self.sample_x()
    }

    /// The last forward's logit row on the host — the teacher-forcing row (post-softcap, so a TF
    /// gate can see the softcap at all).
    pub fn logits(&mut self) -> Result<Vec<f32>> {
        ensure!(
            self.sampled,
            "logits() before sample_x — no forward has produced a row"
        );
        self.sc.logits.copy_out_into(&mut self.row_host)?;
        rivoli_core::num::f32s_le(&self.row_host).ok_or_else(|| {
            anyhow::anyhow!(
                "logits readback length {} not a multiple of 4",
                self.row_host.len()
            )
        })
    }
}

/// The seam's decode entry point.
impl InklingEngine<'_> {
    /// Greedy-decode up to `spec.ngen` tokens continuing `spec.prompt`, stopping on any `spec.eos`.
    ///
    /// **Every short-convolution state is zeroed first** (`reset_state`): the state belongs to a
    /// REQUEST, and a second prompt that inherited the first one's windows would let tokens the
    /// caller never wrote influence the text.
    ///
    /// **Every error path joins the device before returning.** The engine has no `Drop` and its
    /// field order frees the weight slab FIRST, so an error returned mid-layer would unmap weights
    /// with that layer's kernels still in flight. `inspect_err` covers every `?` in the body.
    pub fn decode(&mut self, spec: GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        self.decode_inner(&spec, sink)
            .inspect_err(|_| drop(device_sync()))
    }

    fn decode_inner(&mut self, spec: &GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        ensure!(!spec.prompt.is_empty(), "empty prompt");
        // `checked_add`: a wrapped sum would satisfy the bound and then write past the KV cache.
        let fits = spec.prompt.len().checked_add(spec.ngen);
        let ctx = self.max_ctx();
        ensure!(
            fits.is_some_and(|n| n <= ctx),
            "a {} token prompt plus {} new does not fit the {} positions this engine's KV cache \
             holds",
            spec.prompt.len(),
            spec.ngen,
            ctx
        );
        self.reset_state()?;
        let prefill_wall = std::time::Instant::now();
        let mut cur = self.prefill_and_sample(spec.prompt)?;
        self.flush_trace()?;
        let f0 = self.fetched();
        let (pinned, streamed) = self.residency();
        let prefill_s = prefill_wall.elapsed().as_secs_f64();
        tracing::info!(
            "PREFILL: {} tokens in {:.1} s | {pinned} pinned / {streamed} streamed | {} fills, \
             {} hits",
            spec.prompt.len(),
            prefill_s,
            f0.misses,
            f0.hits,
        );
        // The counters and the phase buckets rebase together: stats describe steady-state decode,
        // and the prefill's fills are warm-up.
        let decode_wall = std::time::Instant::now();
        let prof0 = self.prof;
        let mut emit = Emit::new(spec);
        let mut pos = spec.prompt.len();
        while emit.offer(cur, sink) {
            cur = self.advance(cur, pos)?;
            pos += 1;
        }
        let phases = self.prof.since(&prof0);
        let fetched = self.fetched().since(f0);
        Ok(emit.finish(decode_wall.elapsed(), fetched, &phases))
    }

    /// Feed `cur` at `pos` and decide the token after it — one greedy step, the unit both the
    /// decode loop and the teacher-forced walk are built from.
    fn advance(&mut self, cur: u32, pos: usize) -> Result<u32> {
        self.hidden_state(cur, pos)?;
        let decided = self.sample_x()?;
        // The trace is flushed at the ONE step both the decode loop and the teacher-forced walk
        // go through, so neither can capture a run whose sink silently stopped writing.
        self.flush_trace()?;
        Ok(decided)
    }
}

/// Teacher-forced scoring: the SAME per-position calls the decode loop makes (`prefill_and_sample`,
/// `hidden_state`, `sample_x`), so the TF path cannot diverge from the decode path.
#[cfg(feature = "teacher-forcing")]
impl InklingEngine<'_> {
    /// One NLL per forced position, over `ids`.
    pub fn score(&mut self, ids: &[u32]) -> Result<crate::seam::Scored> {
        crate::score::admit(ids, self.max_ctx())?;
        self.score_inner(ids).inspect_err(|_| drop(device_sync()))
    }

    fn score_inner(&mut self, ids: &[u32]) -> Result<crate::seam::Scored> {
        self.reset_state()?;
        let mut own = self.prefill_and_sample(&ids[..1])?;
        // Counters rebase AFTER position 0: its fill is warm-up, exactly as in `decode`.
        let f0 = self.fetched();
        let tally = crate::score::walk(ids, |next| {
            let row = self.logits()?;
            let decided = own;
            if let Some((tok, pos)) = next {
                own = self.advance(tok, pos)?;
            }
            Ok((row, decided))
        })?;
        tally.into_scored(self.fetched().since(f0))
    }
}
