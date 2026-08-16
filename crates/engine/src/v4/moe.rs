//! The V4 MoE sublayer: the host router, the resident fp8 shared expert, and the `.f4` routed
//! dispatch — one token row at a time, because the FP4 expert kernel has no other shape.
//!
//! Ported from `old:src/f4gpu.rs`'s `route_row` / `routed_experts` / `shared_expert` /
//! `shared_gate_up` / `moe`. What separates this from [`crate::glm::mlp`] is not size: V4's
//! router is `sqrt(softplus(·))` with a renormalisation GLM does not perform, three of its
//! layers select by a TOKEN-ID hash rather than by score, its shared expert is fp8 and
//! resident rather than a routed-format block folded into the batch, and its experts are fp4
//! behind a descriptor type GLM's dispatch refuses by name.
//!
//! # The two writes into `sub`, in order
//!
//! `MoE.forward` starts from `y = zeros` and adds the shared expert RAW — no routing weight —
//! then adds the routed sum. So the shared expert **writes** the sublayer buffer and the
//! accumulator drain **adds** on top of it. In GLM the drain's `+=` IS the residual add; here
//! it is not, because V4's MoE output feeds the hyper-connection expansion rather than the
//! residual. Getting that order backwards loses the shared expert's contribution entirely on
//! every layer, which is one seventh of the FFN and reads as ordinary drift.

use super::ROWS;
use super::engine::{MOE_ACC_ROWS, V4Engine, desc_never_read, desc_of_f4, gemv_fp8};
use super::geometry::FP8_BLOCK;
use crate::device::as_le_bytes;
use crate::routed::Selection;
use anyhow::{Result, ensure};
use rivoli_backend::{
    ExpertDescF4, NULL_STREAM, device_sync, launch_act_quant_f8, launch_gemv_f32,
    launch_moe_acc_drain, launch_moe_expert_range_f4, launch_swiglu_clamped_bf16,
};
use rivoli_core::num::Scoring;
use rivoli_core::routing::{RoutePolicy, RouteScratch, route_into};

/// One routed dispatch's per-call operands; everything else comes off the engine.
///
/// `acc` is the accumulator ROW this stream owns — see [`MOE_ACC_ROWS`], which is one row per
/// STREAM and not one per expert.
#[derive(Clone, Copy)]
struct MoeLane {
    acc: *mut u64,
    stream: *mut std::ffi::c_void,
}

impl V4Engine<'_> {
    /// `MoE.forward` over `m` rows: the gate, then the routed experts one row at a time — with
    /// the shared expert (batched over ALL rows at once) enqueued from inside row 0's dispatch
    /// onto its own stream.
    pub(super) fn moe_sublayer(&mut self, layer: usize, m: usize) -> Result<()> {
        self.gate(layer, m)?;
        for t in 0..m {
            self.route_row(layer, t)?;
            // Only row 0 carries the shared expert, and it carries ALL `m` rows of it: the
            // shared weights are one set read by every row, so `launch_gemv_fp8_bf16`'s `m`
            // reads them once for the whole prompt where the routed path cannot.
            let shared_rows = (t == 0).then_some(m);
            self.routed_experts(layer, t, shared_rows)?;
        }
        Ok(())
    }

    /// The router GEMV and the activation quantization the experts (but NOT the router) see.
    ///
    /// **ONE ROW PER LAUNCH, and not because routing is per-token.** `rivoli_gemv_f32` refuses
    /// any `nrow` but 1 or 2 — `R` is a template parameter and only those two are instantiated
    /// — so passing `m` here aborts layer 0's FFN on the FIRST forward of any prompt longer
    /// than two tokens, with an opaque argument-guard code. That was the reference's other
    /// critical bug, found by review before any device ran it. `nrow == 2` is reachable and
    /// deliberately unused: this arm is structurally single-row, so pairing rows would buy one
    /// fewer launch of a `[n_experts, hidden]` GEMV against a second index space to get wrong.
    fn gate(&mut self, layer: usize, m: usize) -> Result<()> {
        let (dim, n_experts) = (self.cfg.hidden, self.cfg.n_experts);
        let gate_w = self.pin.layer(layer)?.gate_w;
        let (xw, xq) = (self.xw.ptr(), self.xq.ptr_mut());
        // SAFETY: `xw` is `m * dim` f32, `gate_logits` is `max_m * n_experts`, `xq` is
        // `max_m * dim`, and `gate_w` is a `[n_experts, hidden]` pin placement. Every launch
        // is on the null stream, which is what the attention half before it ran on.
        unsafe {
            // The gate reads the UNQUANTIZED activation: `Gate.forward` is
            // `linear(x.float(), self.weight.float())`, a dense f32 GEMV with no fp8 anywhere.
            // That is the whole reason `xq` is a separate buffer rather than an in-place
            // quantization of `xw` — quantizing first would feed the router e4m3 values and
            // the error would look like ordinary routing variation.
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
            // Now quantize, for the experts: block 128 over the full row, which is what every
            // quantized `Linear` in the reference performs. Reading `xw` and writing `xq` in
            // ONE launch is what keeps `xw` intact for its other readers (the router above,
            // and the compressor on the next layer).
            quantize_activation(xw.cast(), xq.cast(), m, dim)?;
        }
        // The one blocking D2H on the per-layer path, and GLM pays the same one: routing is
        // host math. `m * n_experts` f32 — 1 KB at decode. It is also a JOIN: it drains the
        // null stream, so everything the attention half and the gate enqueued has retired by
        // the time the shared-expert chain reads `xq` on another stream.
        self.gate_logits.copy_out_into(&mut self.gl_host)?;
        Ok(())
    }

    /// `Gate.forward` for one row, on the host, into `sel` and `wexpert_host`.
    ///
    /// # Why routing is host work here, and stays so
    ///
    /// A device router existed in the reference, was verified and carried four tests, and was
    /// DELETED rather than wired in. The indices must reach the host regardless, because the
    /// pool's `submit` is host code — so a kernel does not remove a D2H, it moves one: 48 bytes
    /// of picks instead of 1 KB of logits, against an 18.6 MB `tid2eid` upload and a second
    /// scatter to rebuild the weights by absolute id. And `rivoli_core::routing::route_into` is
    /// the router INV-1 is stated about; a second router on the device is a second place for
    /// "the selection bias must not reach the weights" to be wrong, which is invisible to every
    /// magnitude check.
    fn route_row(&mut self, layer: usize, t: usize) -> Result<()> {
        let (k, n_desc) = (self.cfg.top_k, self.cfg.n_experts);
        let logits = &self.gl_host[t * n_desc * 4..(t + 1) * n_desc * 4];
        let lp = self.pin.layer(layer)?;
        let (bias, hash) = match &lp.route {
            super::pin::GateRoute::Scored { bias } => (bias.as_slice(), None),
            // A hash layer has no bias. It still RUNS the gate: the scores become the WEIGHTS
            // even though the selection ignores them, and reading `tid2eid` while skipping the
            // gate leaves the weights uniform — which decodes fluently and wrongly.
            super::pin::GateRoute::Hash { tid2eid } => (self.zero_bias.as_slice(), Some(tid2eid)),
        };
        route_into(
            logits,
            bias,
            RoutePolicy {
                top_k: k,
                // Read off `V4Config`, which REFUSES any other `scoring_func` at load — so
                // this is the config's answer restated where the kernel needs it, not a second
                // authority. A wrong affinity picks different experts, which no numeric
                // comparison downstream can attribute.
                scoring: Scoring::SqrtSoftplus,
            },
            RouteScratch {
                scores: &mut self.scores,
                choice: &mut self.choice,
                sel: &mut self.sel,
            },
        );
        if let Some(tid2eid) = hash {
            // `tid2eid[token * top_k + j]` REPLACES the selection and nothing else. The values
            // are valid by construction: `super::pin::parse_tid2eid` range-checked them into a
            // `Vec<u32>` at load, which is the only place that can — the kernel's own note says
            // it does not, and the descriptor array it would index is `n_experts` long.
            let tok = self.step_ids[t] as usize;
            let base = tok * k;
            let picks = tid2eid.get(base..base + k).ok_or_else(|| {
                anyhow::anyhow!("layer {layer}: tid2eid has no row for token {tok}")
            })?;
            self.sel.clear();
            self.sel.extend(picks.iter().map(|&e| e as usize));
        }
        // Recorded beside the SELECTION it describes and before the weights are derived: the
        // candidate window ranks `choice` and the weights come from `scores`, so the two read
        // different arrays and only this one is what a v2 trace line is about.
        self.pin.routed.record_candidates(&self.choice);
        self.weigh_row(layer)
    }

    /// This row's routed weights: **renormalise, then scale.**
    ///
    /// `weights /= weights.sum()` then `*= routed_scale`. `route_into` does NEITHER — it stops
    /// at the scores, because GLM's `norm_topk_prob` is false — so both are this arm's. The
    /// weights come from `scores`, never from `choice`: letting the selection BIAS reach them
    /// is the one-line "simplification" that changes every routed magnitude by an amount which
    /// looks like ordinary variation.
    fn weigh_row(&mut self, layer: usize) -> Result<()> {
        let sum: f32 = self.sel.iter().map(|&e| self.scores[e]).sum();
        ensure!(
            sum > 0.0 && sum.is_finite(),
            "layer {layer}: routing weights sum to {sum}"
        );
        let scale = self.cfg.routed_scale as f32 / sum;
        self.wexpert_host.fill(0.0);
        // Indexed by ABSOLUTE expert id — this is the scatter `routed_experts` gathers into
        // launch order. The id is in range by two independent mechanisms: the scored path
        // selects indices OF this `n_experts`-long array, and the hash path's values were
        // range-checked at load. `submit` checks a third time.
        for &e in &self.sel {
            self.wexpert_host[e] = self.scores[e] * scale;
        }
        Ok(())
    }

    /// The routed experts for token row `t`, accumulated onto the sublayer buffer's row `t`.
    ///
    /// One row at a time, and that is structural rather than a simplification: `moe.hip`
    /// refuses `nrow != 1` (only `R = 1` is instantiated). So a prefill of `s` tokens performs
    /// `s` MoE dispatches per layer while attention runs ONCE over the whole prompt —
    /// attention is the only operation here with a cross-token dependency.
    fn routed_experts(&mut self, layer: usize, t: usize, shared_rows: Option<usize>) -> Result<()> {
        let dim = self.cfg.hidden;
        {
            let (out, choice, sel) = (&mut self.resolved, &self.choice, &self.sel);
            self.pin.routed.submit(
                Selection {
                    layer,
                    experts: sel,
                },
                choice,
                out,
            )?;
        }
        let n_res = self.stage_launch_order();
        self.descs.copy_in_at(0, as_le_bytes(&self.descs_host))?;
        self.wexpert
            .copy_in_at(0, as_le_bytes(&self.wexpert_launch))?;

        // JOIN 1. The `xq` these read was produced on the NULL stream, and both expert streams
        // are `hipStreamNonBlocking`, so they do not implicitly join it. This is also the WAR
        // join for the shared chain: the sublayer buffer is attention's output, and the chain's
        // down-GEMV overwrites it, so a device-wide sync is what proves attention's last reader
        // retired before the chain can run.
        device_sync()?;
        // AFTER the join, deliberately: enqueued any earlier, that sync would wait for the
        // chain and re-serialise exactly what the shared stream exists to overlap.
        if let Some(rows) = shared_rows {
            self.shared_expert(layer, rows)?;
        }
        self.launch_ranges(t, n_res)?;
        // SAFETY: the sublayer buffer's row `t` is `dim` f32 and holds the shared expert's
        // output; `acc` is `MOE_ACC_ROWS * dim` u64 and both contributing streams drained in
        // `launch_ranges`. `gain` is 1.0: this arm has no magnitude sweep, and a knob whose
        // only value is the identity is a knob that can be set wrongly.
        unsafe {
            let row = self.sub.ptr_mut().cast::<f32>().add(t * dim);
            let acc = self.moe_acc.ptr_mut().cast::<u64>();
            launch_moe_acc_drain(row, acc, dim, MOE_ACC_ROWS, 1.0, NULL_STREAM)?;
        }
        Ok(())
    }

    /// Rewrite the descriptor table and the weight vector in LAUNCH order — residents at
    /// `[0, n_res)`, misses after — and report `n_res`.
    ///
    /// **Refilled with nulls FIRST.** The reference's first version wrote only the selected
    /// entries and claimed the rest stayed null: true for the first token of the first layer
    /// and false ever after, because the previous token's descriptors survive. A stale
    /// descriptor names a pool SLOT, and a slot the policy has since evicted holds a different
    /// expert's bytes at exactly the right addresses — so a wrongly-computed range would read
    /// plausible wrong weights on the one path where the ticket protocol cannot help, instead
    /// of faulting. A few hundred pointer-sextuple writes per token per layer against a 13 MB
    /// expert fetch is not a cost worth trading for that.
    ///
    /// **Launch order, not absolute ids.** The residents then form the ONE contiguous range
    /// the launcher's `[e_start, e_count)` form exists for — one range call where a per-expert
    /// loop would issue `top_k`. Byte-identity across the regrouping is the fixed-point
    /// accumulator's contract: every expert's contribution is computed from the same
    /// descriptor, activation and weight regardless of which launch carries it, and integer
    /// addition is associative and commutative, so the sums cannot depend on the grouping. A
    /// duplicate pick (reachable only through a hash row) gets one compact slot per occurrence
    /// and accumulates once each — the same total.
    fn stage_launch_order(&mut self) -> usize {
        self.descs_host.fill(desc_never_read());
        let mut n_res = 0;
        let mut c = 0;
        for resident in [true, false] {
            for i in 0..self.sel.len() {
                if self.resolved.tickets[i].is_resident() != resident {
                    continue;
                }
                self.descs_host[c] = desc_of_f4(&self.resolved.slots[i]);
                self.wexpert_launch[c] = self.wexpert_host[self.sel[i]];
                c += 1;
            }
            if resident {
                n_res = c;
            }
        }
        n_res
    }

    /// Residents as ONE range on the compute stream, then each miss alone on the miss stream.
    ///
    /// **Synchronous, where GLM's twin awaits two stream signals.** The join here is a
    /// `device_sync` because this loop has no other work in flight to overlap a narrower wait
    /// with — attention, the compressor, the norms and the gate all ran on the null stream and
    /// have already drained — and because a nested executor (`block_on` inside `block_on`) is
    /// the failure that shape invites. It becomes a redundant belt the day the attention set
    /// takes a real stream, which is a measured change and not a free one.
    ///
    /// Residents first is measured, not tidy: inverting the order cost GLM 3.05 → 2.44 tok/s,
    /// because a resident expert's compute is what overlaps the in-flight ones' reads. Every
    /// launch enqueues its ticket's wait first — `wait_on` is the only way to consume a ticket,
    /// so a launch cannot happen without its data dependency, and a resident ticket is a
    /// value-0 timeline wait that enqueues nothing.
    ///
    /// **Misses stay ONE LAUNCH EACH**, each behind its own ticket only: folding them into the
    /// resident range (or into one miss range) would gate the whole batch on the LAST fetch to
    /// land, serialising hits behind misses.
    fn launch_ranges(&mut self, t: usize, n_res: usize) -> Result<()> {
        let dim = self.cfg.hidden;
        let (cs, ms) = (self.compute_stream.raw(), self.miss_stream.raw());
        let acc = self.moe_acc.ptr_mut().cast::<u64>();
        // SAFETY: `moe_acc` is `MOE_ACC_ROWS * dim` u64; this is row 1, in bounds.
        let acc_miss = unsafe { acc.add(dim) };
        for i in 0..self.sel.len() {
            if self.resolved.tickets[i].is_resident() {
                self.pin.routed.wait_on(self.resolved.tickets[i], cs)?;
            }
        }
        if n_res > 0 {
            // SAFETY: `descs`/`wexpert`/`moe_h`/`moe_acc` are `DeviceBuf` fields sized per the
            // launcher's contract, `xq` row `t` is 16-byte aligned by construction
            // (`hipMalloc` is 256-byte aligned and `dim` is a 128-multiple), and every expert
            // in `[0, n_res)` had its ticket wait enqueued above.
            unsafe {
                self.expert_range(t, (0, n_res), MoeLane { acc, stream: cs })?;
            }
        }
        // The compact index ascends in the same order `stage_launch_order` wrote, so straggler
        // `j` reads descriptor `n_res + j`.
        for (j, i) in (0..self.sel.len())
            .filter(|&i| !self.resolved.tickets[i].is_resident())
            .enumerate()
        {
            self.pin.routed.wait_on(self.resolved.tickets[i], ms)?;
            // SAFETY: as above; this expert's bytes are gated by the wait just enqueued on the
            // same stream, and row 1 of the accumulator is this stream's alone.
            unsafe {
                self.expert_range(
                    t,
                    (n_res + j, 1),
                    MoeLane {
                        acc: acc_miss,
                        stream: ms,
                    },
                )?;
            }
        }
        // JOIN 2: `launch_moe_acc_drain`'s own contract is that EVERY stream which accumulated
        // into `acc` has already completed — and the shared chain, which WROTE the buffer the
        // drain adds into, retires here too.
        device_sync()?;
        Ok(())
    }

    /// Enqueue one descriptor range of fp4 experts.
    ///
    /// # Safety
    /// Every device pointer must outlive `lane.stream`'s completion, `x` and `h` must be
    /// 16-byte aligned and must not alias (both are `__restrict__`), and every expert in the
    /// range must already have its ticket wait enqueued on `lane.stream`.
    unsafe fn expert_range(
        &self,
        t: usize,
        (e_start, e_count): (usize, usize),
        lane: MoeLane,
    ) -> Result<()> {
        let (dim, inter, n_desc) = (self.cfg.hidden, self.cfg.moe_inter, self.cfg.n_experts);
        // SAFETY: `xq` holds `max_m * dim` f32 and `t < m <= max_m`.
        let x = unsafe { self.xq.ptr().cast::<f32>().add(t * dim) };
        // SAFETY: forwarded verbatim from this function's own contract. `wexpert` and `h` are
        // indexed by the DESCRIPTOR index, so both are sized for `n_desc` and not `e_count` —
        // a caller that read them as range-relative would run off the end the first time it
        // passed `e_start > 0`, which is the first thing this two-stream shape does.
        unsafe {
            launch_moe_expert_range_f4(
                x,
                dim,
                inter,
                e_start,
                e_count,
                n_desc,
                self.descs.ptr().cast::<ExpertDescF4>(),
                self.wexpert.ptr().cast::<f32>(),
                self.cfg.swiglu_limit as f32,
                self.moe_h.ptr() as *mut f32,
                lane.acc,
                ROWS,
                lane.stream,
            )
        }
    }

    /// The resident fp8 shared expert, batched over all `m` rows, on its own stream.
    ///
    /// # This is where the port DIVERGES from the reference engine, deliberately
    ///
    /// `MoE.__init__` passes `swiglu_limit` to the shared expert as well as to the routed ones
    /// and `Expert.forward` clamps both, so the shared expert needs the same clamped
    /// arithmetic the fp4 routed kernel already runs. The reference called the plain
    /// `launch_swiglu` here and named the deviation at the call
    /// (`v4oracle::Defect::SwigluUnclamped`, one contribution in seven on every layer); this
    /// arm calls [`launch_swiglu_clamped_bf16`], which is the kernel written for exactly this
    /// and the reason it exists.
    ///
    /// **The two are not one parameter apart, which is why it is a different kernel.** Besides
    /// the clamp: both operands are bf16-rounded BEFORE it (`Linear` stores bf16 and
    /// `Expert.forward` reads it back with `.float()`), the product is bf16-rounded, and the
    /// silu is the multiply form `g·sigmoid(g)`. The launcher refuses `limit <= 0`, NaN and
    /// infinities, so "unclamped" is not spellable through it.
    fn shared_expert(&mut self, layer: usize, m: usize) -> Result<()> {
        let (dim, inter) = (self.cfg.hidden, self.cfg.moe_inter);
        // BEFORE the launches: a bad `layer` has to return without having issued anything.
        let sh = self.pin.layer(layer)?.shared;
        // The whole chain rides ONE stream — five launches reading each other's buffers, so a
        // single null-stream member between stream-ordered neighbours is an unordered
        // activation. Its input `xq` retired at the gate-logits D2H and at join 1; its output
        // is read no earlier than the drain, which sits behind join 2.
        let stream = self.shared_stream.raw();
        let g = self.sh_g.ptr_mut().cast::<f32>();
        let u = self.sh_u.ptr_mut().cast::<f32>();
        let out = self.sub.ptr_mut().cast::<f32>();
        let xq = self.xq.ptr().cast::<f32>();
        // One loop rather than two spelled-out launches: the two differ ONLY in (weight,
        // dest), and spelling the other eight arguments twice is how `m`/`inter`/`dim` get
        // transposed in one copy and not the other. jscpd does not catch that pair — it is
        // under the token floor — so it is on the reader, and now it is not.
        //
        // SAFETY: `xq` holds `m * dim` fp8-quantized f32; both weights are pin placements
        // outliving this engine; `g` and `u` are `max_m * inter`, three distinct allocations.
        for (w, dst) in [(sh.gate, g), (sh.up, u)] {
            unsafe { gemv_fp8(xq, w, m, (inter, dim), dst, stream)? };
        }
        // SAFETY: `g`/`u` hold `m * inter` live f32 and `out` is `max_m * dim`; the combine is
        // safe in place (each thread reads both operands, then writes once).
        unsafe {
            launch_swiglu_clamped_bf16(g, u, m * inter, self.cfg.swiglu_limit as f32, g, stream)?;
            // The down projection's input is act-quantized. The routed path does this for
            // itself inside the expert kernel; here it has to be explicit, and forgetting it
            // is silent.
            launch_act_quant_f8(g, m, inter, stream)?;
            // WRITES the sublayer buffer — does not accumulate into it. See this module's
            // header for why that order is the architecture's and not a convenience.
            gemv_fp8(g, sh.down, m, (dim, inter), out, stream)?;
        }
        Ok(())
    }
}

/// The activation quantization every quantized `Linear` performs on its input: block 128 over
/// the whole row, reading `src` and writing `dst` in one launch.
///
/// A named function rather than a bare launch at its one call site, because the arguments are
/// three `usize` in a row (`rows`, `row_stride`, `n`) and passing a PARTIAL extent here — the
/// KV entry's `hd - rd` at block 64 — is the same silent-wrong the attention block's own
/// partial call is about, in the other direction.
///
/// # Safety
/// `src` and `dst` are device buffers of at least `rows * dim` f32 and must outlive the null
/// stream's drain; they may be the same buffer or non-overlapping ones.
unsafe fn quantize_activation(
    src: *const f32,
    dst: *mut f32,
    rows: usize,
    dim: usize,
) -> Result<()> {
    // SAFETY: forwarded from this function's contract; the full-width form is the one
    // `launch_act_quant_f8_prefix` accepts from a distinct source.
    unsafe {
        rivoli_backend::launch_act_quant_f8_prefix(src, dst, rows, dim, dim, FP8_BLOCK, NULL_STREAM)
    }
}
