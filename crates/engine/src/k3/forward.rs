//! The K3 forward pass: one token through 93 layers — the AttnRes fold discipline, the two
//! attention families, the latent-sandwich MoE, and the head tail. This file owns the ORDER
//! (`k3:docs/reference/k3-architecture.md` §3's layer loop, §4-§7's sublayers); the widths
//! are `geometry`'s, the schedules `state`'s, the weights `pin`'s.
//!
//! # The residual is a STACK plus a prefix row, and the arena representation carries it
//!
//! `h` on entry to a layer IS the prefix sum; the two folds REPLACE `h` to produce module
//! inputs while the residual chain lives in the arena (`state.rs`'s header). Everything this
//! loop does to the arena is: fold rows `0..=stack` into `hbuf`, bump `stack` at a boundary,
//! and write/add the sublayer output at row `stack`. Norms apply to the AGGREGATED `h`,
//! never to the prefix row.
//!
//! # Weights are EXTRACTED before dispatch, not borrowed across it
//!
//! Every sublayer starts by copying the `Copy` weight handles out of its `pin` match. That
//! is borrow discipline, not style: the alternative holds a `&pin` across `&mut self`
//! sublayer calls, and the compiler's refusal of that shape is what forces the extraction —
//! the match against the WRONG family is the thing that cannot compile, which is the layer
//! map (`layer_is_mla`, never a modulus) doing its job at the second site.
//!
//! # Streams
//!
//! Everything here runs on the null stream except the routed experts (compute stream for the
//! resident range, miss stream per straggler — `crate::v4::moe`'s measured order), joined by
//! `device_sync` before the drain. The shared MLP stays on the null stream in this first
//! landing: overlapping it is a measured change, and the join discipline it would need is
//! exactly the class of bug the house records most.

use super::engine::{K3Engine, LayerState, MOE_ACC_ROWS};
use super::geometry::MLA_LORA_EPS;
use super::pin::{Attn, Ffn, SituMlp};
use super::state::{combine_weights, final_sources, fold_at};
use crate::device::as_le_bytes;
use crate::resident::Bf16Weight;
use crate::routed::Selection;
use anyhow::{Result, bail, ensure};
use rivoli_backend::{
    NULL_STREAM, device_sync, launch_attn_res, launch_embed_bf16_row_bcast,
    launch_gated_delta_recurrent_f32, launch_gemm_bf16, launch_mha_attend, launch_moe_acc_drain_to,
    launch_moe_expert_range_f4_situ, launch_rmsnorm_gate_heads_f32, launch_rmsnorm_single,
    launch_short_conv_silu_f32, launch_sigmoid_gate, launch_situ_glu_f32, launch_vadd,
    memcpy_dtod_async,
};
// One nested use, not two lines: split, this preamble ends `v4/moe.rs`'s token for token
// and the jscpd gate reports the pair — `golden_read.rs`'s fewer-imports rule again.
use rivoli_core::{
    num::Scoring,
    routing::{RoutePolicy, RouteScratch, route_into},
};

/// One bf16 projection over one row: `out = w · x`, dims from the placed weight. A GEMM at
/// `m = 1` because there is no bf16 GEMV in the tree and adding one is arithmetic to price,
/// not to assume (`crate::v4::forward`'s head-tail note carries the wave-shape objection and
/// its answer).
///
/// # Safety
/// `x` holds `w.i_dim` live f32 on the device, `out` holds `w.o_dim` writable f32, and
/// neither aliases the other or the weight (all three are `__restrict__` in the kernel).
unsafe fn bproj(x: *const u8, w: Bf16Weight, out: *mut u8) -> Result<()> {
    // SAFETY: forwarded verbatim from this function's contract.
    unsafe {
        launch_gemm_bf16(
            x.cast(),
            w.packed,
            out.cast(),
            1,
            w.o_dim,
            w.i_dim,
            NULL_STREAM,
        )
    }
}

/// One RMSNorm's three operands, which always travel together: the weight, its extent, and
/// WHICH eps — the model-wide 1e-5 or the MLA LoRA norms' 1e-6, this arm's quietest trap.
#[derive(Clone, Copy)]
struct Norm {
    w: *const f32,
    n: usize,
    eps: f32,
}

/// One out-of-place RMSNorm over a bare vector — every eps-bearing call site (the two LoRA
/// norms, the latent aggregate, the sandwich norms) reaches `launch_rmsnorm_single` through
/// here, so the cast pattern and the eps choice have one author.
///
/// # Safety
/// `x` holds `norm.n` live f32, `norm.w` is a placed weight of that extent, `y` holds
/// `norm.n` writable f32 distinct from `x` (the launcher is out-of-place by contract).
unsafe fn rms_vec(x: *const u8, norm: Norm, y: *mut u8) -> Result<()> {
    // SAFETY: forwarded verbatim from this function's contract.
    unsafe { launch_rmsnorm_single(x.cast(), norm.w, norm.n, norm.eps, y.cast()) }
}

/// The arena's residual bookkeeping between layers: how many snapshots exist, and whether
/// the prefix row currently holds a live sum (false only between a boundary push and the
/// attention output that re-seeds it — §3's `prefix_sum = NONE`).
#[derive(Clone, Copy)]
struct Pref {
    stack: usize,
    live: bool,
}

impl K3Engine<'_> {
    /// One token at `pos`, leaving this position's logits in the logit buffer.
    pub(super) fn forward(&mut self, tok: u32, pos: usize) -> Result<()> {
        let (vocab, ceiling) = (self.cfg.vocab, self.max_ctx);
        ensure!(
            (tok as usize) < vocab,
            "token id {tok} is past the vocab {vocab}"
        );
        ensure!(
            pos < ceiling,
            "position {pos} is past the --ctx {ceiling} sizing"
        );
        // Open this position in the shared additive mask: every MLA layer attends the full
        // allocated cache, and this 4-byte write is what makes position `pos` visible.
        self.mask.copy_in_at(pos * 4, &0f32.to_le_bytes())?;
        // The embedding IS the initial prefix sum — a plain bf16→f32 gather, no scale
        // factor (`k3:docs/reference/k3-architecture.md` §7) — landing at arena row 0.
        let table = self.pin.embed.packed;
        // SAFETY: `table` is `[vocab, hidden]` bf16 with `tok < vocab` checked above; row 0
        // of the arena is `hidden` writable f32.
        let hid = self.cfg.hidden;
        unsafe {
            let row0 = self.arena.ptr_mut().cast();
            launch_embed_bf16_row_bcast(table, tok as usize, hid, 1, row0, NULL_STREAM)?;
        }
        let mut pref = Pref {
            stack: 0,
            live: true,
        };
        for l in 0..self.pin.layers() {
            pref = self.layer(l, pos, pref)?;
        }
        self.head_tail(pref.stack)
    }

    /// One layer of §3's loop. Returns the arena bookkeeping for the next layer.
    fn layer(&mut self, l: usize, pos: usize, p: Pref) -> Result<Pref> {
        let f = fold_at(l, self.cfg.attn_res_block_size);
        // Layer entry: `h = attn_res(blocks ‖ pref)` — GUARDED (an empty stack skips),
        // unlike the mlp fold below.
        match f.entry_sources {
            Some(nsrc) => self.fold_into_h(self.pin.layer(l)?.attn_fold, nsrc)?,
            None => {
                // `entry_sources` is `None` only at layer 0, before any push, so the prefix
                // row IS row 0 (`state::fold_at`) — checked on the dev profile, where the
                // suites run.
                debug_assert_eq!(p.stack, 0, "an empty entry fold implies an empty stack");
                self.copy_pref_to_h()?;
            }
        }
        let mut p = p;
        if f.push {
            // The prefix row becomes the newest snapshot; the prefix restarts as NONE and
            // is re-seeded by the attention output below (`state.rs`'s representation).
            p = Pref {
                stack: p.stack + 1,
                live: false,
            };
        }
        // Both sublayer norms take the model eps and read the AGGREGATED h.
        self.norm_h_into_xn(self.pin.layer(l)?.input_norm)?;
        match matches!(self.pin.layer(l)?.attn, Attn::Kda(_)) {
            true => self.kda_attention(l)?,
            false => self.mla_attention(l, pos)?,
        }
        p = self.add_sub_to_pref(p)?;
        // The pre-FFN fold is UNCONDITIONAL — no empty guard, §3's one asymmetry between
        // the two calls.
        self.fold_into_h(self.pin.layer(l)?.mlp_fold, f.mlp_sources)?;
        self.norm_h_into_xn(self.pin.layer(l)?.post_norm)?;
        match matches!(self.pin.layer(l)?.ffn, Ffn::Dense(_)) {
            true => self.dense_ffn(l)?,
            false => self.moe_ffn(l)?,
        }
        p = self.add_sub_to_pref(p)?;
        // ONE join per layer: the miss stream's last expert must retire before the next
        // layer's first accumulator atomic, and GLM/V4 pay the same sync for the same
        // reason.
        device_sync()?;
        Ok(p)
    }

    /// `hbuf = softmax(⟨RMSNorm(row_s), fold⟩) @ rows` over arena rows `0..nsrc`.
    fn fold_into_h(&mut self, fold: *const f32, nsrc: usize) -> Result<()> {
        let hid = self.cfg.hidden;
        // SAFETY: the arena holds `(res_blocks + 1) * hidden` f32 and every caller's `nsrc`
        // is ≤ `res_blocks + 1` by `state::fold_at`'s construction; `hbuf` is `hidden`
        // writable f32 and does not alias the arena. Stride `nsrc * hid` satisfies the
        // kernel's `src_stride >= nsrc * n` with one token.
        unsafe {
            launch_attn_res(
                self.arena.ptr().cast(),
                fold,
                1,
                nsrc,
                hid,
                nsrc * hid,
                self.cfg.rms_norm_eps as f32,
                self.hbuf.ptr_mut().cast(),
            )
        }
    }

    /// `h = pref`, the layer-0 case where there is no stack to fold — the prefix row is
    /// arena row 0, per the call site's invariant.
    fn copy_pref_to_h(&mut self) -> Result<()> {
        let bytes = self.cfg.hidden * size_of::<f32>();
        // SAFETY: row 0 is in the arena and `hbuf` is `bytes` writable; null-stream
        // ordering keeps this behind the embed gather.
        unsafe { memcpy_dtod_async(self.hbuf.ptr_mut(), self.arena.ptr(), bytes, NULL_STREAM) }
    }

    /// `xn = rmsnorm(hbuf, w)` at the model eps.
    fn norm_h_into_xn(&mut self, w: *const f32) -> Result<()> {
        let norm = Norm {
            w,
            n: self.cfg.hidden,
            eps: self.cfg.rms_norm_eps as f32,
        };
        // SAFETY: `hbuf` and `xn` are distinct `hidden`-f32 buffers.
        unsafe { rms_vec(self.hbuf.ptr(), norm, self.xn.ptr_mut()) }
    }

    /// `pref (+)= sub`: add the sublayer output into the prefix row, or seed the row when
    /// the prefix is NONE after a boundary push. Returns the bookkeeping with a live prefix.
    fn add_sub_to_pref(&mut self, p: Pref) -> Result<Pref> {
        let bytes = self.cfg.hidden * size_of::<f32>();
        // SAFETY: row `p.stack` is in bounds (≤ res_blocks); `sub` holds `hidden` live
        // f32; the two never alias (arena and sub are distinct allocations).
        unsafe {
            let row = self.arena.ptr_mut().add(p.stack * bytes);
            match p.live {
                true => launch_vadd(row.cast(), self.sub.ptr().cast(), self.cfg.hidden)?,
                false => memcpy_dtod_async(row, self.sub.ptr(), bytes, NULL_STREAM)?,
            }
        }
        Ok(Pref {
            stack: p.stack,
            live: true,
        })
    }

    /// §4: the KDA sublayer. Projections → short conv (SiLU fused, no separate activation
    /// to misplace — trap 5) → the gated delta recurrence, which finishes the decay, both
    /// L2 norms and the q-scale ITSELF (traps 2, 3, 4, 6, 7 are inside its signature: the
    /// inputs arrive raw) → the fused head norm THEN gate (trap 10's KDA half) → o_proj.
    fn kda_attention(&mut self, l: usize) -> Result<()> {
        let Attn::Kda(w) = &self.pin.layer(l)?.attn else {
            bail!("layer {l}: KDA dispatch on a non-KDA layer");
        };
        let o_w = w.o;
        let LayerState::Kda {
            state,
            win_q,
            win_k,
            win_v,
        } = &mut self.layers[l]
        else {
            bail!("layer {l}: KDA weights beside a non-KDA state");
        };
        let la = &self.cfg.linear_attn_config;
        let (ch, taps) = (self.d.kda_ch, la.short_conv_kernel_size);
        let xn = self.xn.ptr();
        // SAFETY throughout: every buffer below is an engine allocation sized in `new` for
        // exactly these widths; `xn` holds `hidden` live f32; all launches ride the null
        // stream in dependency order; no launcher argument aliases another (distinct
        // allocations, and each conv window aliases neither its input nor its output).
        unsafe {
            bproj(xn, w.q, self.kq.ptr_mut())?;
            bproj(xn, w.k, self.kk.ptr_mut())?;
            bproj(xn, w.v, self.kv.ptr_mut())?;
            bproj(xn, w.b, self.beta.ptr_mut())?;
            // One shared rank-`head_dim` pair feeds all heads: z = f_b(f_a(x)).
            bproj(xn, w.f_a, self.f_mid.ptr_mut())?;
            bproj(self.f_mid.ptr(), w.f_b, self.z.ptr_mut())?;
            bproj(xn, w.g, self.gate.ptr_mut())?;
            for (cur, cw, win, out) in [
                (&self.kq, w.q_conv, win_q, &mut self.kqc),
                (&self.kk, w.k_conv, win_k, &mut self.kkc),
                (&self.kv, w.v_conv, win_v, &mut self.kvc),
            ] {
                launch_short_conv_silu_f32(
                    cur.ptr().cast(),
                    cw,
                    ch,
                    taps,
                    win.ptr_mut().cast(),
                    out.ptr_mut().cast(),
                    NULL_STREAM,
                )?;
            }
            launch_gated_delta_recurrent_f32(
                self.kqc.ptr().cast(),
                self.kkc.ptr().cast(),
                self.kvc.ptr().cast(),
                self.z.ptr().cast(),
                self.beta.ptr().cast(),
                w.a_log,
                w.dt_bias,
                la.num_heads,
                la.head_dim,
                la.gate_lower_bound as f32,
                state.ptr_mut().cast(),
                self.ko.ptr_mut().cast(),
                NULL_STREAM,
            )?;
            // Norm, THEN gate, then project — the opposite order from the MLA path below.
            // The eps is the MODEL-WIDE `rms_norm_eps`, not `MLA_LORA_EPS`: the gate-norm
            // fixtures read it off the golden's own tiny config and the device kernel
            // matched the first-party `o_norm` captures under `kda_op`'s tolerance — so
            // 1e-5 here is anchored, where the LoRA norms' 1e-6 had to be read.
            launch_rmsnorm_gate_heads_f32(
                self.ko.ptr().cast(),
                self.gate.ptr().cast(),
                w.o_norm,
                la.num_heads,
                la.head_dim,
                self.cfg.rms_norm_eps as f32,
                self.kon.ptr_mut().cast(),
                NULL_STREAM,
            )?;
        }
        self.project_sub(o_w, self.kon.ptr())
    }

    /// The output projection both attention families end with: `sub = w · x`. One author,
    /// because the two tails were the first self-clone the jscpd gate reported here.
    ///
    /// Takes `x` as a pointer its caller derives from the buffer it just filled — the
    /// SAFETY obligation (a live `w.i_dim`-f32 source) is the caller's, restated at each of
    /// the two call sites by construction: each passes the buffer its own last launch wrote.
    fn project_sub(&mut self, w: Bf16Weight, x: *const u8) -> Result<()> {
        // SAFETY: `sub` is `hidden` writable f32 and distinct from every attention scratch.
        unsafe { bproj(x, w, self.sub.ptr_mut()) }
    }

    /// §5: the gated-MLA sublayer, NoPE. The LoRA norms take [`MLA_LORA_EPS`]; the cache
    /// row is the FULL 192 (nope ‖ rope — trap 9 lives in that width); the gate has NO
    /// norm (trap 10's MLA half); the scale is `geometry`'s, over 192 (trap 8).
    fn mla_attention(&mut self, l: usize, pos: usize) -> Result<()> {
        // Copy the handles out — see the module header on why dispatch extracts.
        let (q_a, q_a_norm, q_b, kv_a, kv_a_norm, kv_b, g_w, o_w) = {
            let Attn::Mla(w) = &self.pin.layer(l)?.attn else {
                bail!("layer {l}: MLA dispatch on a non-MLA layer");
            };
            (
                w.q_a,
                w.q_a_norm,
                w.q_b,
                w.kv_a,
                w.kv_a_norm,
                w.kv_b,
                w.g,
                w.o,
            )
        };
        let (heads, vh) = (self.cfg.n_heads, self.cfg.v_head_dim);
        let xn = self.xn.ptr();
        // SAFETY throughout: engine allocations at the widths `new` sized them for; the
        // norms are out-of-place into their own buffers; null-stream dependency order.
        unsafe {
            bproj(xn, q_a, self.qa.ptr_mut())?;
            bproj(xn, g_w, self.gate.ptr_mut())?;
            let ql = Norm {
                w: q_a_norm,
                n: self.cfg.q_lora_rank,
                eps: MLA_LORA_EPS,
            };
            rms_vec(self.qa.ptr(), ql, self.qan.ptr_mut())?;
            bproj(self.qan.ptr(), q_b, self.qb.ptr_mut())?;
            // ONE projection emits latent ‖ rope; the norm covers the LATENT half only —
            // its `n` is `kv_lora_rank`, so the rope slot never enters the statistic.
            bproj(xn, kv_a, self.kva.ptr_mut())?;
            let kl = Norm {
                w: kv_a_norm,
                n: self.cfg.kv_lora_rank,
                eps: MLA_LORA_EPS,
            };
            rms_vec(self.kva.ptr(), kl, self.kvan.ptr_mut())?;
            bproj(self.kvan.ptr(), kv_b, self.kvb.ptr_mut())?;
        }
        self.append_kv(l, pos)?;
        let LayerState::Mla { kc, vc } = &self.layers[l] else {
            bail!("layer {l}: MLA weights beside a non-MLA state");
        };
        // SAFETY: `qb` is `[heads][192]`, the caches are `[heads][max_ctx][192/128]` and
        // scored at `kv == max_ctx` behind the additive mask (unreached rows are -inf);
        // `attn` is `[heads][128]` writable; the gate is applied in place on `attn` after
        // the attend retires (both on the null/0 stream).
        unsafe {
            launch_mha_attend(
                self.qb.ptr().cast(),
                kc.ptr().cast(),
                vc.ptr().cast(),
                self.mask.ptr().cast(),
                heads,
                self.max_ctx,
                self.d.q_head,
                vh,
                self.d.mla_scale,
                self.attn.ptr_mut().cast(),
            )?;
            launch_sigmoid_gate(
                self.attn.ptr_mut().cast(),
                self.gate.ptr().cast(),
                heads * vh,
                NULL_STREAM,
            )?;
        }
        self.project_sub(o_w, self.attn.ptr())
    }

    /// Write position `pos` into this layer's caches: per head, `k = kv_b's nope half ‖ the
    /// SHARED rope slot` and `v = kv_b's v half`. The rope slot is one 64-wide vector
    /// broadcast to every head (`k3:docs/reference/k3-architecture.md` §5) — the broadcast
    /// is paid in cache bytes so the attend kernel's `[heads][kv][d]` layout holds.
    fn append_kv(&mut self, l: usize, pos: usize) -> Result<()> {
        let f = size_of::<f32>();
        let (heads, ctx) = (self.cfg.n_heads, self.max_ctx);
        let (nope, rope, vh) = (
            self.cfg.qk_nope_head_dim,
            self.cfg.qk_rope_head_dim,
            self.cfg.v_head_dim,
        );
        let (dk, dkv) = (self.d.q_head, self.d.kv_b_head);
        let LayerState::Mla { kc, vc } = &mut self.layers[l] else {
            bail!("layer {l}: appending KV on a non-MLA layer");
        };
        // SAFETY: `kvb` is `[heads][nope + vh]`, `kva`'s tail at `kv_lora_rank` is the
        // 64-wide rope slot; each destination offset is `(h * ctx + pos)` rows into a cache
        // allocated for `heads * ctx` rows, with `pos < max_ctx` checked in `forward`. All
        // copies are null-stream ordered behind the projections that produced the sources.
        unsafe {
            let rope_src = self.kva.ptr().add(self.cfg.kv_lora_rank * f);
            for h in 0..heads {
                let src = self.kvb.ptr().add(h * dkv * f);
                let krow = kc.ptr_mut().add((h * ctx + pos) * dk * f);
                let vrow = vc.ptr_mut().add((h * ctx + pos) * vh * f);
                memcpy_dtod_async(krow, src, nope * f, NULL_STREAM)?;
                memcpy_dtod_async(krow.add(nope * f), rope_src, rope * f, NULL_STREAM)?;
                memcpy_dtod_async(vrow, src.add(nope * f), vh * f, NULL_STREAM)?;
            }
        }
        Ok(())
    }

    /// Layer 0's dense FFN (`k3:docs` §7, same betas as everything else).
    fn dense_ffn(&mut self, l: usize) -> Result<()> {
        let Ffn::Dense(m) = self.pin.layer(l)?.ffn else {
            bail!("layer {l}: dense dispatch on a MoE layer");
        };
        let out = self.sub.ptr_mut();
        self.situ_mlp(m, out)
    }

    /// One SiTU-GLU MLP: gate/up → SiTU-GLU → down into `out`. The dense FFN and the fused
    /// shared MLP are this same chain at different widths, so it has ONE author — and the
    /// activation's sigmoid takes the UNCAPPED gate inside the launcher, which refuses
    /// non-positive betas (its guard, not a check here).
    fn situ_mlp(&mut self, m: SituMlp, out: *mut u8) -> Result<()> {
        let xn = self.xn.ptr();
        // SAFETY: `mg`/`mu` are sized `max(dense_inter, shared_inter)` and every caller's
        // `m.gate.o_dim` is one of those two; the situ output MAY alias its gate input (the
        // one launcher with a documented alias allowance); `out` is `hidden` writable and
        // distinct from `mg`.
        unsafe {
            bproj(xn, m.gate, self.mg.ptr_mut())?;
            bproj(xn, m.up, self.mu.ptr_mut())?;
            launch_situ_glu_f32(
                self.mg.ptr().cast(),
                self.mu.ptr().cast(),
                m.gate.o_dim,
                self.cfg.activation_situ_beta as f32,
                self.cfg.activation_linear_beta as f32,
                self.mg.ptr_mut().cast(),
                NULL_STREAM,
            )?;
            bproj(self.mg.ptr(), m.down, out)?;
        }
        Ok(())
    }

    /// §6, in the documented order: route on FULL width → down-project → experts in the
    /// LATENT → drain → norm the AGGREGATE → up-project → add the shared MLP unweighted,
    /// AFTER the up-projection, at full width (trap 12).
    fn moe_ffn(&mut self, l: usize) -> Result<()> {
        self.route(l)?;
        let (down, latent_norm, up, shared) = {
            let Ffn::Moe(t) = &self.pin.layer(l)?.ffn else {
                bail!("layer {l}: MoE dispatch on the dense layer");
            };
            (t.down, t.latent_norm, t.up, t.shared)
        };
        // SAFETY: `xn` is `hidden` live f32 and `z_lat` is `expert_in` writable.
        unsafe { bproj(self.xn.ptr(), down, self.z_lat.ptr_mut())? };
        self.routed_latent(l)?;
        let lat = self.cfg.expert_in;
        // SAFETY: `latent` and `z_lat` are `expert_in` f32; the drain's `n` is the LATENT
        // width (passing `hidden` is the drain's documented last-row overrun); both expert
        // streams drained inside `routed_latent`. The norm is out-of-place back into
        // `z_lat`, whose expert readers have retired.
        unsafe {
            launch_moe_acc_drain_to(
                self.latent.ptr_mut().cast(),
                self.moe_acc.ptr_mut().cast(),
                lat,
                MOE_ACC_ROWS,
                NULL_STREAM,
            )?;
            let agg = Norm {
                w: latent_norm,
                n: lat,
                eps: self.cfg.rms_norm_eps as f32,
            };
            rms_vec(self.latent.ptr(), agg, self.z_lat.ptr_mut())?;
            bproj(self.z_lat.ptr(), up, self.sub.ptr_mut())?;
        }
        // The shared MLP reads the ORIGINAL full-width input (`xn`), lands in `tmp_h`, and
        // adds with NO routing weight and NO scaling.
        let out = self.tmp_h.ptr_mut();
        self.situ_mlp(shared, out)?;
        // SAFETY: `sub` and `tmp_h` are distinct `hidden`-f32 buffers.
        unsafe {
            launch_vadd(
                self.sub.ptr_mut().cast(),
                self.tmp_h.ptr().cast(),
                self.cfg.hidden,
            )
        }
    }

    /// The router: a bf16 gate GEMV on the FULL width, one blocking D2H, then host top-k —
    /// host math for the reason `crate::v4::moe` argues at length (the pool's `submit` is
    /// host code, and `route_into` is the router INV-1 is stated about). The combining
    /// weights are `state::combine_weights`' (trap 11: unbiased scores, renormalised).
    fn route(&mut self, l: usize) -> Result<()> {
        let Ffn::Moe(t) = &self.pin.layer(l)?.ffn else {
            bail!("layer {l}: routing on the dense layer");
        };
        // SAFETY: `xn` is `hidden` live f32, `gate_logits` is `n_experts` writable.
        unsafe { bproj(self.xn.ptr(), t.gate, self.gate_logits.ptr_mut())? };
        // The one blocking D2H on the per-layer path (~3.5 KB); it also drains the null
        // stream, so everything upstream has retired when the host reads the logits.
        self.gate_logits.copy_out_into(&mut self.gl_host)?;
        // The config REFUSES any other `moe_router_activation_func` at load; the policy
        // restates its answer where the selection needs it (§6: independent sigmoids, no
        // softmax).
        let policy = RoutePolicy {
            top_k: self.cfg.top_k,
            scoring: Scoring::Sigmoid,
        };
        let (scores, choice, sel) = (&mut self.scores, &mut self.choice, &mut self.sel);
        route_into(
            &self.gl_host,
            &t.bias,
            policy,
            RouteScratch {
                scores,
                choice,
                sel,
            },
        );
        // Beside the SELECTION it describes, before the weights are derived — the trace's
        // candidate record ranks `choice`, the weights read `scores`.
        self.pin.routed.record_candidates(&self.choice);
        combine_weights(
            &self.scores,
            &self.sel,
            self.cfg.routed_scale as f32,
            &mut self.wexpert_host,
        )
    }

    /// The streamed experts over the latent: submit, reorder residents-first, launch the
    /// resident range on the compute stream and each miss alone behind its ticket on the
    /// miss stream (`crate::v4::moe`'s measured order — inverting it cost GLM 20%), then
    /// join both.
    fn routed_latent(&mut self, layer: usize) -> Result<()> {
        let picks = Selection {
            layer,
            experts: &self.sel,
            // `--divergence-log` is a GLM-only instrument (the arm that does not reproduce itself), so
            // this arm has no fold targets to point at.
            fold: crate::fetch::asyncfetch::FetchFolds::OFF,
        };
        self.pin
            .routed
            .submit(picks, &self.choice, &mut self.resolved)?;
        // Launch order: residents first. A stable sort by residency keeps each half in
        // selection order, which is the order the trace recorded.
        self.launch_idx.clear();
        self.launch_idx.extend(0..self.sel.len());
        let tickets = &self.resolved.tickets;
        self.launch_idx.sort_by_key(|&i| !tickets[i].is_resident());
        let n_res = self
            .launch_idx
            .iter()
            .take_while(|&&i| tickets[i].is_resident())
            .count();
        // Descriptor table refilled with FAULTING nulls first — a stale entry names a pool
        // slot the policy may since have evicted, which is plausible wrong weights at
        // exactly the right addresses on the one path the ticket protocol cannot help.
        self.descs_host.fill(rivoli_backend::ExpertDescF4::null());
        for (c, &i) in self.launch_idx.iter().enumerate() {
            let (g, u, dn) = {
                let s = &self.resolved.slots[i];
                (&s.gate, &s.up, &s.down)
            };
            self.descs_host[c] = rivoli_backend::ExpertDescF4 {
                gate_packed: g.packed,
                gate_scale: g.scale,
                up_packed: u.packed,
                up_scale: u.scale,
                down_packed: dn.packed,
                down_scale: dn.scale,
            };
            self.wexpert_launch[c] = self.wexpert_host[self.sel[i]];
        }
        self.descs.copy_in_at(0, as_le_bytes(&self.descs_host))?;
        self.wexpert
            .copy_in_at(0, as_le_bytes(&self.wexpert_launch))?;
        // JOIN before the stream launches: `z_lat` was produced on the null stream and both
        // expert streams are non-blocking, so nothing implicit orders them behind it.
        device_sync()?;
        let (cs, ms) = (self.compute_stream.raw(), self.miss_stream.raw());
        for k in 0..self.launch_idx.len() {
            let ticket = self.resolved.tickets[self.launch_idx[k]];
            match k < n_res {
                // Residents: enqueue every wait (a resident ticket's wait enqueues
                // nothing), then ONE contiguous range once all are enqueued.
                true => {
                    self.pin.routed.wait_on(ticket, cs)?;
                    if k + 1 == n_res {
                        self.expert_range(0..n_res, 0, cs)?;
                    }
                }
                // Misses: one launch each behind its own ticket — folding them into a
                // range would gate the whole batch on the LAST fetch to land.
                false => {
                    self.pin.routed.wait_on(ticket, ms)?;
                    self.expert_range(k..k + 1, 1, ms)?;
                }
            }
        }
        // Both accumulating streams must retire before the drain reads the accumulator —
        // the drain launcher's own contract.
        device_sync()
    }

    /// One descriptor range of MXFP4 situ experts into accumulator row `acc_row`.
    fn expert_range(
        &self,
        experts: std::ops::Range<usize>,
        acc_row: usize,
        stream: *mut std::ffi::c_void,
    ) -> Result<()> {
        let (lat, inter) = (self.cfg.expert_in, self.cfg.moe_inter);
        // SAFETY: `z_lat` is `expert_in` f32, 16-byte aligned (hipMalloc); `descs`,
        // `wexpert` and `moe_h` are sized for `n_experts` DESCRIPTORS (the launcher indexes
        // them by descriptor, not range-relative); `acc_row` is 0 or 1 of `MOE_ACC_ROWS`,
        // each row `expert_in` u64 wide and owned by one stream; every expert in the range
        // had its ticket wait enqueued on `stream` by the caller. `x` and `h` are distinct
        // allocations.
        unsafe {
            let acc = self
                .moe_acc
                .ptr()
                .cast_mut()
                .cast::<u64>()
                .add(acc_row * lat);
            launch_moe_expert_range_f4_situ(
                self.z_lat.ptr().cast(),
                lat,
                inter,
                experts.start,
                experts.len(),
                self.cfg.n_experts,
                self.descs.ptr().cast(),
                self.wexpert.ptr().cast(),
                self.cfg.activation_situ_beta as f32,
                self.cfg.activation_linear_beta as f32,
                self.moe_h.ptr().cast_mut().cast(),
                acc,
                super::ROWS,
                stream,
            )
        }
    }

    /// §7's tail: the model-level fold (whose omission is silent), the final norm, and the
    /// head — no bias, no logit scaling, no tied embeddings.
    fn head_tail(&mut self, stack: usize) -> Result<()> {
        let want = final_sources(self.pin.layers(), self.cfg.attn_res_block_size);
        ensure!(
            stack + 1 == want,
            "the layer loop left {stack} snapshots + prefix where the model-level fold \
             expects {want} sources — the boundary schedule and the loop have diverged"
        );
        self.fold_into_h(self.pin.output_fold, want)?;
        self.norm_h_into_xn(self.pin.final_norm)?;
        // SAFETY: `xn` holds the normed `hidden` row; `logits` is `vocab` writable.
        unsafe { bproj(self.xn.ptr(), self.pin.head, self.logits.ptr_mut())? };
        self.stepped = true;
        Ok(())
    }
}
