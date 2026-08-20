//! The GLM MLP sublayer: fp8 SwiGLU on the three dense layers, the routed-expert MoE
//! (host routing → ticketed pool submit → two-stream launch → fixed-point drain) on
//! the rest. Ported from `old:src/gpu.rs`'s MoE half; the ticketed-dataflow arguments
//! travel with their loops.

use super::engine::{GlmEngine, MOE_ACC_ROWS};
use super::forward::{ResidualBase, Rows};
use crate::device::as_le_bytes;
use crate::fetch::asyncfetch::Ticket;
use crate::glm::pin::{Fp8Mlp, Fp8Weight, LayerMlp};
use crate::routed::{ExpertSlot, Selection};
use anyhow::Result;
// Only the divergence probe's two `.context()` calls need this, and an unconditional import is
// an `unused_imports` error under this workspace's deny-level warnings.
#[cfg(feature = "corruption-probe")]
use anyhow::Context;
use rivoli_artifact::format::RoutedFmt;
use rivoli_backend::{
    ExpertDesc, NULL_STREAM, launch_gemv_f32, launch_gemv_fp8, launch_moe_acc_drain,
    launch_moe_expert_range, launch_moe_expert_range_i4, launch_swiglu, launch_vadd, stream_signal,
};
use rivoli_core::routing::{RoutePolicy, RouteScratch, route_into};

/// One MoE dispatch's per-call operands; everything else comes off the engine.
#[derive(Clone, Copy)]
struct MoeLane {
    acc: *mut u64,
    stream: *mut std::ffi::c_void,
    nrow: usize,
}

/// Six-pointer descriptor from a resolved slot. One builder for routed and shared
/// experts; the int4 kernel reinterprets the same bytes at its own slot offsets, which
/// is why one descriptor type serves both formats.
fn desc_of(m: &ExpertSlot) -> ExpertDesc {
    ExpertDesc {
        gate_indices: m.gate.packed,
        gate_scales: m.gate.scale as *const u16,
        up_indices: m.up.packed,
        up_scales: m.up.scale as *const u16,
        down_indices: m.down.packed,
        down_scales: m.down.scale as *const u16,
    }
}

impl GlmEngine<'_> {
    /// The dense-layer MLP sublayer: fp8 SwiGLU into `moe_out`, then the residual add.
    /// Null stream end to end.
    pub(super) fn dense_sublayer(&mut self, m: Fp8Mlp, rows: Rows, xp: ResidualBase) -> Result<()> {
        let inter = m.gate.o_dim;
        let xnp = self.xn.ptr() as *const f32;
        let gp = self.mlp_g.ptr_mut() as *mut f32;
        let up = self.mlp_u.ptr_mut() as *mut f32;
        let outp = self.moe_out.ptr_mut() as *mut f32;
        let gemv = |x: *const f32, w: Fp8Weight, y: *mut f32| -> Result<()> {
            // SAFETY: weights resident; x/y live device scratch for nrow rows.
            unsafe {
                launch_gemv_fp8(
                    x, w.packed, w.scale, w.o_dim, w.i_dim, w.block, rows.nrow, y,
                )
            }
        };
        gemv(xnp, m.gate, gp)?;
        gemv(xnp, m.up, up)?;
        // SAFETY: gp/up hold nrow·inter live f32; one elementwise launch over the
        // contiguous row-minor buffers, in place: h = silu(gate)·up.
        unsafe { launch_swiglu(gp, up, rows.nrow * inter, gp, NULL_STREAM)? };
        gemv(gp as *const f32, m.down, outp)?;
        // SAFETY: xp is nrow rows of live residual; moe_out was fully written above on
        // the null stream.
        unsafe { launch_vadd(xp.0, outp as *const f32, rows.nrow * self.cfg.hidden) }
    }

    /// The MoE sublayer: host routing, pool submit, the ticketed two-stream launch,
    /// then the drain — which IS the residual add (it converts the fixed-point
    /// accumulator straight into `x` and resets it, so the conversion costs no extra
    /// pass; both MoE streams were awaited, so no barrier is needed despite both
    /// having written `moe_acc`).
    pub(super) async fn moe_sublayer(
        &mut self,
        l: usize,
        rows: Rows,
        xp: ResidualBase,
    ) -> Result<()> {
        // Sampled BEFORE the layer runs: what `--divergence-log` records is the per-layer
        // DELTA, because the question is whether a divergence coordinate coincides with this
        // layer's misses and relocations, not with the run's totals.
        #[cfg(feature = "corruption-probe")]
        let before = (self.pin.routed.misses(), self.pin.routed.relocs());
        self.route_layer(l, rows)?;
        self.moe_layer(l, rows).await?;
        #[cfg(feature = "corruption-probe")]
        self.probe_moe(l, rows, before)?;
        // `ac`: the fixed-point accumulator BEFORE the drain resets it — the consumer-output
        // witness for pass 2, exactly as `h` is for pass 1. Both MoE lanes were awaited inside
        // `moe_layer`, so every writer has retired; the drain below is the next reader.
        //
        // It has to be here and not in `probe_moe`, because `probe_moe` runs before this point in
        // the call order and the drain runs immediately after — this is the only instant at which
        // the accumulator holds the layer's complete result.
        #[cfg(feature = "corruption-probe")]
        {
            let n = MOE_ACC_ROWS * rows.nrow * self.cfg.hidden * 2; // u64 slots as f32 words
            let acc = self.moe_acc.ptr() as *const f32;
            if let Some(p) = self.probe.as_mut().filter(|p| p.folds().ac) {
                // SAFETY: `moe_acc` holds MOE_ACC_ROWS*MAXROW*hidden u64 and `nrow <= MAXROW`, so
                // `n` f32 words are in bounds; both lanes' atomics retired at `launch_moe`'s awaits.
                unsafe { p.fold(crate::probe::Q::Ac, l, acc, n)? };
            }
        }
        // SAFETY: xp nrow rows live; moe_acc's writers completed.
        unsafe {
            launch_moe_acc_drain(
                xp.0,
                self.moe_acc.ptr_mut() as *mut u64,
                rows.nrow * self.cfg.hidden,
                MOE_ACC_ROWS,
                1.0,
                NULL_STREAM,
            )
        }
    }

    /// `--divergence-log`'s MoE-layer record: the `h` fold plus the host columns.
    ///
    /// **This is the middle of a three-way cut and the other two ends are in `forward.rs`.**
    /// `xn` (the MLP's input, folded there for every layer including the dense ones) differing
    /// means attention; `h` — the SwiGLU intermediate, folded here — differing with `xn` equal
    /// means the gate/up expert BYTES or that kernel, the routed-pool hypothesis; and the exit
    /// residual differing with both equal means the down projection, the fixed-point
    /// accumulator or the drain, the accumulation hypothesis. Two runs' logs diff to a
    /// coordinate AND a mechanism.
    ///
    /// Runs AFTER `moe_layer`, which is what makes it legal: that call host-awaits both MoE
    /// streams, so `h`'s writers have retired and a fold on the null stream reads settled
    /// bytes. It also means the record can carry the slot placement, which does
    /// not exist until `submit` has run.
    #[cfg(feature = "corruption-probe")]
    fn probe_moe(&mut self, l: usize, rows: Rows, before: (u64, u64)) -> Result<()> {
        if self.probe.is_none() {
            return Ok(());
        }
        // EVERY ROW OF THE PASS, not just row 0 — `nrow * ne` bytes.
        //
        // Row 0 alone was the first version, on the argument that `gate_logits_host` is the whole
        // `MAXROW`-wide D2H and its tail is uninitialised at `nrow == 1`. True, but the bound
        // that fixes it is `nrow * ne`, not `ne`: MAXROW is 2, so EVERY layer-major prefill pass
        // runs nrow=2, and a row-0 fold left a divergence entering row 1's logits invisible in
        // `gl` and `pk` while `xn` and `sl` moved — which is `pk`'s documented reading
        // ("equal `gl` and unequal `pk` means routing consulted something outside its inputs")
        // silently narrowed to a claim about one row. Found by review, 2026-08-17.
        let ne = self.cfg.n_experts * 4;
        let glen = rows.nrow * ne;
        let cols = crate::probe::Cols {
            gl: rivoli_core::hash::fnv1a(
                self.gate_logits_host
                    .get(..glen)
                    .context("divergence probe: gate-logits staging shorter than the pass")?,
            ),
            pk: rivoli_core::hash::fnv1a_u64s(
                self.sel_row[..rows.nrow]
                    .iter()
                    .flat_map(|r| r.iter().map(|&e| e as u64)),
            ),
            sl: self.pin.routed.slot_fold(l, &self.union),
            miss: self.pin.routed.misses() - before.0,
            reloc: self.pin.routed.relocs() - before.1,
        };
        // `se`: the WHOLE union's pool slots, folded now that the layer's compute has been
        // awaited — see `fold_slots` for why the union and not just the misses, and what it costs.
        //
        // UNCONDITIONAL on a MoE layer, including one that missed nothing. A layer of pure hits
        // can still read corrupt bytes: the corruption would have arrived on whatever earlier
        // token loaded them. Guarding this on `misses > 0` (as an earlier draft did) would have
        // made the instrument blind to exactly the case `bh`/`sc` cannot see, which is the reason
        // this column exists.
        //
        // The `se` pointer is taken in its own scope so the `&mut self.probe` borrow ends before
        // `fold_slots` needs `&self.pin`, and before the `p` binding below takes the probe again.
        if self.probe.as_ref().is_some_and(|p| p.folds().se) {
            // Its own scope: the `&mut self.probe` borrow ends before `fold_slots` needs
            // `&self.pin`, and before the `p` binding below takes the probe again.
            let se = {
                let p = self
                    .probe
                    .as_mut()
                    .context("unreachable: probe checked present above")?;
                p.fold_slot(l, crate::probe::Q::Se)?
            };
            // SAFETY: `se` is one live device u64 in the slab; every writer of these slots retired
            // at `launch_moe`'s awaits.
            unsafe { self.pin.routed.fold_slots(l, &self.union, se)? };
        }
        let inter = self.cfg.moe_inter;
        // EXACTLY the written extent, and the bound is load-bearing: the kernel writes
        // `h[(e·R + t)·inter + j]` for `e < descs.len()` and `R == nrow`, so anything past
        // `descs.len() · nrow · inter` is untouched `hipMalloc` memory whose bits differ per
        // run. Folding one word of it would manufacture a divergence on every line.
        let h = (
            self.moe_hidden.ptr() as *const f32,
            self.descs.len() * rows.nrow * inter,
        );
        // Re-taken rather than held across the reads above: `cols` needs `&self.pin` and
        // `&self.cfg`, which a live `&mut self.probe` borrow would forbid. The `is_none`
        // guard at the top is what keeps the folds free when the flag is off; this `else` is
        // the borrow checker's copy of it and cannot be reached.
        let Some(p) = self.probe.as_mut() else {
            return Ok(());
        };
        // SAFETY: `h` is live device f32 written by launches this layer already host-awaited
        // (`launch_moe` awaits both lanes), bounded to exactly the written region. What keeps it
        // readable on the NULL stream is that `run_layer`'s trailing `device_sync` is what stands
        // between this and the NEXT layer's overwrite of the same buffer — the two MoE lanes are
        // `hipStreamNonBlocking`, so the host await orders the WRITE and the device_sync orders
        // the reuse.
        unsafe { p.fold(crate::probe::Q::H, l, h.0, h.1)? };
        p.set_cols(l, cols)
    }

    /// Host routing for a MoE layer: gate GEMV, the blocking logits D2H, per-row
    /// routing, and the UNION build. Row 0's picks come first and in order, so an
    /// `nrow == 1` pass submits exactly what the unbatched engine submitted. Routing
    /// is a pure function of (logits, bias, top_k) — it does NOT consult residency
    /// (INV-1).
    fn route_layer(&mut self, l: usize, rows: Rows) -> Result<()> {
        let cfg = self.cfg;
        let LayerMlp::Moe { gate_w, .. } = self.pin.layers[l].mlp else {
            anyhow::bail!("route_layer on dense layer {l}")
        };
        let (xnp, glp) = (
            self.xn.ptr() as *const f32,
            self.gate_logits.ptr_mut() as *mut f32,
        );
        // SAFETY: gate_w resident F32; glp device scratch for nrow·n_experts.
        unsafe {
            launch_gemv_f32(
                xnp,
                gate_w,
                cfg.n_experts,
                cfg.hidden,
                rows.nrow,
                glp,
                NULL_STREAM,
            )?
        };
        // The gate-logits D2H is the layer's one blocking join on the host path. Its
        // span goes to the ATTEND bucket: what it drains is the attention execution the
        // layer launched (plus this gate GEMV's own — a 6144×256 f32 GEMV, small
        // against MLA over a growing KV; the approximation is stated on
        // `telemetry::ProfileSummary`).
        let t = std::time::Instant::now();
        self.gate_logits.copy_out_into(&mut self.gate_logits_host)?;
        let t = self.prof.lap(crate::telemetry::Phase::Attend, t);
        // Descending, so `scores`/`choice` are left holding ROW 0 for the trace window
        // below — the trace measures the router against the real token, and row 0 is
        // the real token.
        let ne = cfg.n_experts * 4;
        let policy = RoutePolicy {
            top_k: cfg.top_k,
            scoring: cfg.scoring(),
        };
        for r in (0..rows.nrow).rev() {
            route_into(
                &self.gate_logits_host[r * ne..(r + 1) * ne],
                self.pin.moe_bias(l),
                policy,
                RouteScratch {
                    scores: &mut self.scores,
                    choice: &mut self.choice,
                    sel: &mut self.sel_row[r],
                },
            );
            self.weigh_row(r);
        }
        self.build_union(rows.nrow);
        self.pin.routed.record_candidates(&self.choice);
        // Host routing — sigmoid/bias/top-k over 256 experts per row plus the union
        // build — is FFN-side work.
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        Ok(())
    }

    /// Row `r`'s routed weights: the affinity score BEFORE the bias (the bias steers
    /// selection only), sum-normalized over THIS row's picks, then scaled. Runs while
    /// `scores` still belongs to row `r` — the next `route_into` overwrites it.
    fn weigh_row(&mut self, r: usize) {
        let wr = &mut self.wrow[r];
        wr.clear();
        for &e in &self.sel_row[r] {
            wr.push(self.scores[e]);
        }
        let mut sm: f32 = wr.iter().sum();
        if self.cfg.norm_topk_prob {
            sm += 1e-20;
            for wi in wr.iter_mut() {
                *wi /= sm;
            }
        }
        for wi in wr.iter_mut() {
            *wi *= self.cfg.routed_scale as f32;
        }
    }

    /// The deduplicated union of every row's picks, row 0's first and in order.
    fn build_union(&mut self, nrow: usize) {
        self.union.clear();
        for r in 0..nrow {
            for &e in &self.sel_row[r] {
                if !self.union.contains(&e) {
                    self.union.push(e);
                }
            }
        }
    }

    /// The routed pool round: submit the union's cold reads (each selected expert gets
    /// a ticket — hit → RESIDENT, miss → resolves when its bytes land; the slot
    /// ADDRESSES are known after submit, so the descriptors are valid pointers), stage
    /// the batch, launch. The UNION, not one row's picks: every expert any row routed
    /// to must be resident before the batch launches.
    async fn moe_layer(&mut self, l: usize, rows: Rows) -> Result<()> {
        let t = std::time::Instant::now();
        // The fetch-path fold targets travel WITH the selection, so they are resolved here where
        // the probe is in hand rather than pushed into the pool as state beforehand.
        #[cfg(feature = "corruption-probe")]
        let fold = match self.probe.as_mut() {
            // Each position is armed ONLY if `--divergence-folds` asked for it. The all-three
            // configuration is what suppressed the defect over 2,048 tokens, so a cell enables one
            // at a time and the others must be genuinely absent, not merely unread.
            Some(p) => {
                let f = p.folds();
                crate::fetch::asyncfetch::FetchFolds {
                    bh: match f.bh {
                        crate::fetch::asyncfetch::FoldProbe::Off => std::ptr::null_mut(),
                        _ => p.fold_slot(l, crate::probe::Q::Bh)?,
                    },
                    bh_mode: f.bh,
                    sc: match f.sc {
                        crate::fetch::asyncfetch::FoldProbe::Off => std::ptr::null_mut(),
                        _ => p.fold_slot(l, crate::probe::Q::Sc)?,
                    },
                    decoy: p.decoy(),
                    line_stride: f.line_stride,
                    sc_mode: f.sc,
                }
            }
            None => crate::fetch::asyncfetch::FetchFolds::OFF,
        };
        #[cfg(not(feature = "corruption-probe"))]
        let fold = crate::fetch::asyncfetch::FetchFolds::OFF;
        let (out, choice, union) = (&mut self.resolved, &self.choice, &self.union);
        self.pin.routed.submit(
            Selection {
                layer: l,
                experts: union,
                fold,
            },
            choice,
            out,
        )?;
        self.stage_batch(l, rows)?;
        // Pool submit + host staging + the two H2D uploads — FFN-side host work.
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        self.launch_moe(rows).await
    }

    /// Host staging after submit: the `[descriptor][row]` weight matrix, the
    /// descriptor table (routed + shared), the ticket list, and the two device
    /// uploads.
    fn stage_batch(&mut self, l: usize, rows: Rows) -> Result<()> {
        let nrow = rows.nrow;
        // Scatter each row's weights into the layout the kernel reads as
        // `wexpert[e*R + t]`. Driven from the union rather than from each row's picks,
        // so "this row did not route here" is the natural `None` and leaves the 0.0
        // the resize put there — the kernel SKIPS a zero weight rather than
        // multiplying by it (`0 * dv` with a non-finite `dv` is NaN, which the
        // fixed-point clamp would turn into a finite extreme). That skip is what makes
        // row 0 of a batched pass bit-identical to an unbatched one.
        self.wexpert.clear();
        self.wexpert.resize(self.union.len() * nrow, 0.0);
        for (u, &e) in self.union.iter().enumerate() {
            for r in 0..nrow {
                if let Some(i) = self.sel_row[r].iter().position(|&x| x == e) {
                    self.wexpert[u * nrow + r] = self.wrow[r][i];
                }
            }
        }
        self.descs.clear();
        for m in &self.resolved.slots {
            self.descs.push(desc_of(m));
        }
        self.tickets.clear();
        self.tickets.extend_from_slice(&self.resolved.tickets);
        let LayerMlp::Moe { shared, .. } = self.pin.layers[l].mlp else {
            anyhow::bail!("stage_batch on dense layer {l}")
        };
        self.descs.push(desc_of(&shared));
        // Weight 1.0 for EVERY row: the shared expert is unconditional. It is in the
        // RESIDENT tier, never streamed, so its dependency is already satisfied — but
        // it must still grow `tickets` with `descs`, or the launch loop indexes past
        // the end (it did once in the old tree: "len is 8 but the index is 8").
        self.wexpert.extend(std::iter::repeat_n(1.0, nrow));
        self.tickets.push(Ticket::RESIDENT);
        self.descs_buf.copy_in_at(0, as_le_bytes(&self.descs))?;
        self.wexpert_buf.copy_in_at(0, as_le_bytes(&self.wexpert))?;
        Ok(())
    }

    /// TICKETED DATAFLOW. Every expert is enqueued behind a DEVICE-SIDE wait on its
    /// own data, with no branch on residency and no host round trip — resident,
    /// missing and in-flight take the same path (a resident expert carries
    /// `Ticket::RESIDENT`, value 0, satisfied on arrival). The wait is enqueued BEFORE
    /// the producer has run, which is what `hipStreamWaitEvent` cannot do and
    /// `hipStreamWaitValue64` can (INV-4). Do not "simplify" this to events.
    ///
    /// ORDER MATTERS, and getting it wrong cost 20% in the old tree: the compute
    /// stream is FIFO, so enqueueing in `sel` order puts every resident expert BEHIND
    /// the first miss's wait, and nothing computes while that fetch is in flight —
    /// which is the overlap the whole engine is built on. So residents go first
    /// (batched into maximal runs), misses after ON THE MISS STREAM (their waits then
    /// start at the top of the layer, and the GPU's wake latency is absorbed by the
    /// resident compute running beside it — measured +382 µs per layer-with-misses
    /// when the compute stream carried them). Reordering LAUNCHES is safe by
    /// construction: each expert accumulates into fixed-point atomically, and integer
    /// addition is associative — the two-stream split is a contention fix, not a
    /// correctness one.
    ///
    /// This branches on `ticket.is_resident()`, and that is NOT the old `hit` mask
    /// coming back: the mask decided whether to WAIT (a wrong bit read unwritten
    /// memory, silently); this decides only the ORDER of launches that each enqueue
    /// their wait unconditionally. A wrong bit costs throughput and cannot cost
    /// correctness.
    async fn launch_moe(&mut self, rows: Rows) -> Result<()> {
        let ndesc = self.descs.len();
        debug_assert_eq!(
            self.tickets.len(),
            ndesc,
            "every descriptor needs a ticket (routed picks + the shared expert)"
        );
        let acc = self.moe_acc.ptr_mut() as *mut u64;
        // The miss stream's accumulator block — see `moe_acc`'s layout doc.
        // SAFETY: `moe_acc` is MOE_ACC_ROWS·MAXROW·hidden u64; this is block 1, in
        // bounds for nrow ≤ MAXROW.
        let acc_miss = unsafe { acc.add(rows.nrow * self.cfg.hidden) };
        let (cs_raw, ms_raw) = (self.compute_stream.raw(), self.miss_stream.raw());
        let t = std::time::Instant::now();
        self.launch_residents(MoeLane {
            acc,
            stream: cs_raw,
            nrow: rows.nrow,
        })?;
        self.launch_misses(MoeLane {
            acc: acc_miss,
            stream: ms_raw,
            nrow: rows.nrow,
        })?;
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        // BOTH streams, because neither waits for the other: with a fixed-point
        // accumulator nothing between them needs ordering, and the only consumer of
        // all experts (the drain) runs after this returns.
        //
        // The two awaits are also the phase profile's one honest window onto fetch
        // exposure, with no sync added: the compute stream carries resident experts
        // (compute-bound), the miss stream's experts are gated on their own fetches —
        // so the RESIDUAL wait on the miss stream after the compute await returned is
        // fetch cost the resident compute did NOT hide. 0 means fully hidden, which is
        // the overlap the whole engine is built on.
        let t = std::time::Instant::now();
        stream_signal(cs_raw)?.await;
        let t = self.prof.lap(crate::telemetry::Phase::Ffn, t);
        stream_signal(ms_raw)?.await;
        self.prof.lap(crate::telemetry::Phase::FetchWait, t);
        Ok(())
    }

    /// Residents, batched into maximal runs. A hit's ticket is already resolved, so
    /// awaiting before launching buys nothing and costs the GPU an idle gap (~7 of 9
    /// experts per layer at the measured ~76% hit rate). The wait is enqueued anyway
    /// rather than skipped: `wait_on` is the only way to consume a ticket, a resident
    /// one short-circuits on value 0, and the unconditional call is what makes "every
    /// launch is behind its dependency" true by reading the code rather than trusting
    /// this loop's classification.
    fn launch_residents(&mut self, lane: MoeLane) -> Result<()> {
        let ndesc = self.descs.len();
        let mut i = 0usize;
        while i < ndesc {
            if !self.tickets[i].is_resident() {
                i += 1;
                continue;
            }
            let j = (i..ndesc)
                .find(|&k| !self.tickets[k].is_resident())
                .unwrap_or(ndesc);
            for k in i..j {
                self.pin.routed.wait_on(self.tickets[k], lane.stream)?;
            }
            // SAFETY: descs/codebooks resident; every expert in [i, j) has its
            // dependency enqueued above; scratch live; the stream is live.
            unsafe { self.expert_range(i..j, lane)? };
            i = j;
        }
        Ok(())
    }

    /// The misses, one wait + one launch each, on the miss stream.
    fn launch_misses(&mut self, lane: MoeLane) -> Result<()> {
        for e in 0..self.descs.len() {
            if self.tickets[e].is_resident() {
                continue;
            }
            self.pin.routed.wait_on(self.tickets[e], lane.stream)?;
            // SAFETY: as in `launch_residents`; this expert's bytes are gated by the
            // wait just enqueued on the same stream.
            unsafe { self.expert_range(e..e + 1, lane)? };
        }
        Ok(())
    }

    /// Enqueue the `experts` slice of the descriptor table in the run's ONE format.
    /// The int4 kernel reinterprets the same descriptor bytes at its own slot offsets,
    /// which is why one descriptor buffer serves both formats; `.f4` needs a different
    /// descriptor type and is refused (spelled out so a fourth variant cannot fall
    /// through a `_`).
    ///
    /// # Safety
    /// `descs_buf`/codebooks resident; scratch live; `lane.stream` live; every expert in
    /// `experts` already gated by a wait enqueued on `lane.stream`.
    unsafe fn expert_range(&self, experts: std::ops::Range<usize>, lane: MoeLane) -> Result<()> {
        let (x, h, acc) = (
            self.xn.ptr() as *const f32,
            self.moe_hidden.ptr() as *mut f32,
            lane.acc,
        );
        let descs = self.descs_buf.ptr() as *const ExpertDesc;
        let wexpert = self.wexpert_buf.ptr() as *const f32;
        let (hidden, inter) = (self.cfg.hidden, self.cfg.moe_inter);
        let [cb0, cb1, cb2] = self.codebooks;
        let (e_start, e_count) = (experts.start, experts.len());
        let (nrow, stream) = (lane.nrow, lane.stream);
        // SAFETY: forwarded verbatim from this function's own contract.
        unsafe {
            match self.fmt {
                RoutedFmt::I4 => launch_moe_expert_range_i4(
                    x, hidden, inter, e_start, e_count, descs, wexpert, h, acc, nrow, stream,
                ),
                RoutedFmt::Vq3 => launch_moe_expert_range(
                    x, hidden, inter, e_start, e_count, descs, cb0, cb1, cb2, wexpert, h, acc,
                    nrow, stream,
                ),
                RoutedFmt::F4 => anyhow::bail!(
                    "an .f4 expert reached GLM's MoE dispatch — it needs ExpertDescF4 and \
                     launch_moe_expert_range_f4, not this descriptor"
                ),
            }
        }
    }
}
