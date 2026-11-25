//! Inkling 2 Small's routed MoE dispatch: the `.mxfp8` experts out of the routed pool, the
//! compute and miss streams their fills overlap on, and the fixed-point accumulator they land in.
//!
//! Split from `decode.rs` when the two-stream dispatch crossed the workspace's 800-line soft
//! cap — a split by cohesion, not convenience: `decode.rs` owns the ORDER the blocks run in,
//! and this file owns one block's device composition.
//!
//! **The order is `crate::v4::moe`'s, measured there**: residents first as one contiguous range
//! on the compute stream, then each miss alone behind its own ticket on the miss stream. The
//! pool, the tickets and the O_DIRECT fills are the same machinery, so the shape that overlaps
//! them well is the same shape.
//!
//! **Two shapes, one decision.** A layer with at least one miss splits across the compute and
//! miss streams so the fills overlap the resident compute; a layer with NO miss runs entirely on
//! the null stream, because a second stream with no fill to hide behind buys nothing and costs
//! two joins. That branch is the whole reason this block no longer calls `device_sync`: a
//! fully-resident layer's MoE, post-conv, residual add and the next layer's attention queue
//! behind one another on the null stream until the next router readback drains it.

use super::engine::{InklingEngine, MOE_ACC_ROWS, MxFp8MoeScratch};
use super::forward::{p, pc};
use super::pin::InklingPin;
use crate::device::as_le_bytes;
use crate::fetch::asyncfetch::FetchFolds;
use crate::routed::{RoutedPool, Selection};
use anyhow::{Context, Result};
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_backend::{
    ExpertDescF4, NULL_STREAM, block_on, launch_moe_acc_drain_to,
    launch_moe_expert_range_mxfp8_situ, stream_signal,
};
use std::ffi::c_void;

/// The activation one range reads and the config its extents come from — bundled because they
/// travel together and unbundled they would push the range call past the argument budget.
#[derive(Clone, Copy)]
struct MoeX<'a> {
    x: *const f32,
    cfg: &'a InklingTextConfig,
}

/// One dispatch's per-stream operands: the accumulator ROW this stream owns and the stream.
#[derive(Clone, Copy)]
struct MoeLane {
    acc: *mut u64,
    stream: *mut c_void,
}

/// One dispatch's launch-order split — the accumulator's row 0, the resident count
/// `stage_launch_order` reports, and `top_k` — bundled for [`MoeX`]'s reason: unbundled they
/// would push both launch shapes past the argument budget.
#[derive(Clone, Copy)]
struct MoeLaunch {
    acc: *mut u64,
    n_res: usize,
    k: usize,
}

impl InklingEngine<'_> {
    /// Small's MoE: the pool's `.mxfp8` slots, one range of residents overlapping the misses.
    ///
    /// **The shape is `crate::v4::moe`'s measured order**, not an invention: residents first as
    /// ONE contiguous range on the compute stream, then each miss alone behind its own ticket on
    /// the miss stream. Folding the misses into one range would gate the whole batch on the LAST
    /// fetch to land, serialising hits behind stragglers; launching without the waits would run
    /// a kernel over an expert's bytes before its O_DIRECT fill arrived.
    ///
    /// **Two accumulator rows, one per stream** ([`MOE_ACC_ROWS`]): sharing a row across streams
    /// measured +825 µs on a 6-miss layer. The drain sums and RESETS exactly the rows it reads,
    /// so a zero-miss layer drains row 0 alone and leaves row 1 at the zero the previous miss
    /// layer's drain left there — which is why the two shapes can share one accumulator.
    pub(super) fn moe_mxfp8(&mut self, layer: usize, scale: f32) -> Result<()> {
        let t = std::time::Instant::now();
        let n_res = self.stage_moe8(layer, scale)?;
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        let k = self.route_sel.len();
        let x_in = MoeX {
            x: pc(&self.sc.xn),
            cfg: self.cfg,
        };
        let acc = self
            .moe8
            .as_mut()
            .context("the .mxfp8 backend has no routed scratch")?
            .acc
            .ptr_mut()
            .cast::<u64>();
        let d = MoeLaunch { acc, n_res, k };
        if n_res == k {
            return self.moe_resident_only(x_in, d);
        }
        self.moe_split(x_in, d)
    }

    /// The `.mxfp8` scratch and the routed pool, or the message naming which one is missing.
    ///
    /// ONE borrow of each for a whole dispatch: they are disjoint fields, and the pin has no
    /// other pool reader on this path. Written once because three call sites need the pair, and
    /// three copies of two `context` calls is a clone the duplication gate counts.
    ///
    /// **It takes the two FIELDS, not `&mut self`**, and that is what keeps the borrows
    /// field-precise: `stage_moe8` reads `self.route_choice`/`self.route_scores` beside the pair,
    /// and every caller stamps `self.prof`, which a whole-`self` borrow would refuse.
    fn moe8_parts<'s>(
        moe8: &'s mut Option<MxFp8MoeScratch>,
        pin: &'s mut InklingPin,
    ) -> Result<(&'s mut MxFp8MoeScratch, &'s mut RoutedPool)> {
        let m = moe8
            .as_mut()
            .context("the .mxfp8 backend has no routed scratch")?;
        let routed = pin
            .routed_mut()
            .context("the .mxfp8 pin holds no routed pool")?;
        Ok((m, routed))
    }

    /// The host half of one dispatch: resolve the selection against the pool (issuing whatever
    /// fills it needs), rewrite the descriptor table into launch order, and upload it.
    ///
    /// **No device join between this and the launches, and that is an invariant rather than a
    /// removed safeguard.** `route()`'s `router_logits.copy_out_into` is a blocking null-stream
    /// D2H (`device.rs`), so the null stream — and with it `xn` — has retired before this runs;
    /// `copy_in_at` is a blocking `hipMemcpy` H2D, so `descs`/`wexpert` are globally visible on
    /// return (`docs/investigations/rust-kernels.md` §5 measured that ordering in the CLR).
    /// Nothing is enqueued between that readback and these launches, so a sync here could only
    /// wait on work that has already retired.
    ///
    /// **The readback also carries the pool's own safety argument**, which is why the sync was
    /// not merely slow: `submit` may relocate a slot or (under `--trace`) poison a freshly
    /// assigned one, and BOTH are wrong if a previous layer's expert kernel is still reading that
    /// slot. The same drain retires those kernels, so the reuse is ordered by the router's
    /// readback rather than by a `device_sync` of its own.
    fn stage_moe8(&mut self, layer: usize, scale: f32) -> Result<usize> {
        let (m, routed) = Self::moe8_parts(&mut self.moe8, &mut self.pin)?;
        // The descriptor table names POOL SLOTS, so a stale entry would read a since-evicted
        // expert's bytes at plausible addresses — a wrong answer, not a fault.
        // `stage_launch_order` rewrites every one of the `k` entries below, before the upload.
        // The candidate window first: `--trace` v2 records the demand keys a layer looked up
        // AND the top-ranked candidates that produced them, and `submit` consumes the window.
        routed.record_candidates(&self.route_choice);
        routed.submit(
            Selection {
                layer,
                experts: &self.route_sel,
                // `--divergence-log` folds GLM's own routed quantities; this arm has no fold
                // targets to point at.
                fold: FetchFolds::OFF,
            },
            &self.route_choice,
            &mut m.resolved,
        )?;
        let n_res = stage_launch_order(m, (&self.route_scores, &self.route_sel), scale);
        m.descs.copy_in_at(0, as_le_bytes(&m.descs_host))?;
        m.wexpert.copy_in_at(0, as_le_bytes(&m.wexpert_launch))?;
        Ok(n_res)
    }

    /// Zero misses: every ticket is resident, so the whole dispatch is ONE range on the null
    /// stream and nothing is joined.
    ///
    /// **This is the steady state on a pin that holds the routed pool** (Mini's 64 GiB: every
    /// layer of every token), and it is the branch that removes the layer's syncs entirely — no
    /// compute stream, no miss stream, no join, so the MoE, the post-MLP conv, the residual add
    /// and the next layer's attention queue behind one another on the null stream until the next
    /// router readback drains them. Waiting on a RESIDENT ticket is a no-op (`Fetch::wait`
    /// returns before touching a timeline); the loop stays because it is what makes "every
    /// launch is behind its dependency" true by reading the code.
    fn moe_resident_only(&mut self, x_in: MoeX<'_>, d: MoeLaunch) -> Result<()> {
        let t = std::time::Instant::now();
        let (m, routed) = Self::moe8_parts(&mut self.moe8, &mut self.pin)?;
        for &i in &m.launch_idx[..d.k] {
            routed.wait_on(m.resolved.tickets[i], NULL_STREAM)?;
        }
        // `k` is `top_k`, and a zero-`top_k` config is legal — the router then selects nothing
        // and the range launcher refuses `e_count == 0` (1001), exactly as the old resident
        // guard refused it. The drain below still runs, so row 0 stays the zero the next layer
        // needs.
        if d.k > 0 {
            // SAFETY: every expert in the range has its wait enqueued above, `descs` holds `k`
            // filled entries, `moe_h` is `k · inter`, and row 0 is the only row this layer
            // touches.
            unsafe {
                m.range(
                    x_in,
                    (0, d.k),
                    MoeLane {
                        acc: d.acc,
                        stream: NULL_STREAM,
                    },
                )?
            };
        }
        let out = p(&self.sc.moe_acc);
        let hidden = x_in.cfg.hidden_size;
        // SAFETY: `out` is `hidden` f32 writable and `acc` is `MOE_ACC_ROWS · hidden` u64; ONE
        // row was accumulated into, and the drain resets exactly the rows it sums, so row 1
        // keeps the zero every earlier miss layer left there.
        let r = unsafe { launch_moe_acc_drain_to(out, d.acc, hidden, 1, NULL_STREAM) };
        self.lap_ffn(t, r)
    }

    /// At least one miss: residents as one range on the compute stream, each miss alone behind
    /// its ticket on the miss stream, then the two awaits — which are also this arm's only
    /// honest window onto fetch exposure.
    ///
    /// **The split is `glm/mlp.rs`'s, and so is the reading**: the compute stream carries the
    /// residents (compute-bound) and the miss stream's experts are gated on their own fills, so
    /// the RESIDUAL wait on the miss stream after the compute await returned is fetch cost the
    /// resident compute did NOT hide. 0 means fully hidden.
    fn moe_split(&mut self, x_in: MoeX<'_>, d: MoeLaunch) -> Result<()> {
        let (m, routed) = Self::moe8_parts(&mut self.moe8, &mut self.pin)?;
        let (cs, ms) = (m.compute_stream.raw(), m.miss_stream.raw());
        let t = std::time::Instant::now();
        for &i in &m.launch_idx[..d.n_res] {
            routed.wait_on(m.resolved.tickets[i], cs)?;
        }
        if d.n_res > 0 {
            // SAFETY: every expert in the range has its wait enqueued above, `descs` holds `k`
            // filled entries, and `moe_h` is `k · inter`.
            unsafe {
                m.range(
                    x_in,
                    (0, d.n_res),
                    MoeLane {
                        acc: d.acc,
                        stream: cs,
                    },
                )?
            };
        }
        // SAFETY: `acc` is `MOE_ACC_ROWS * hidden` u64; row 1 is in bounds.
        let miss = MoeLane {
            acc: unsafe { d.acc.add(x_in.cfg.hidden_size) },
            stream: ms,
        };
        // The compact index ascends in the order `stage_launch_order` wrote, so straggler `j`
        // reads descriptor `n_res + j`.
        for (j, &i) in m.launch_idx[d.n_res..].iter().enumerate() {
            routed.wait_on(m.resolved.tickets[i], ms)?;
            // SAFETY: this expert's bytes are gated by the wait just enqueued on the same
            // stream, and row 1 of the accumulator is this stream's alone.
            unsafe { m.range(x_in, (d.n_res + j, 1), miss)? };
        }
        if d.n_res > 0 {
            block_on(stream_signal(cs)?);
        }
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        let t = std::time::Instant::now();
        block_on(stream_signal(ms)?);
        self.prof.lap(crate::telemetry::Phase::FetchWait, t);
        let t = std::time::Instant::now();
        let out = p(&self.sc.moe_acc);
        let hidden = x_in.cfg.hidden_size;
        // SAFETY: BOTH streams that accumulated into `acc` retired above, `out` is `hidden` f32,
        // and the drain sums and resets both rows.
        unsafe { launch_moe_acc_drain_to(out, d.acc, hidden, MOE_ACC_ROWS, NULL_STREAM) }?;
        // The drain is a launch, not a wait, and it is stamped so the split path's `Ffn` holds
        // the same thing the resident path's does — otherwise its enqueue time would fall into
        // `PROFILE/tok`'s `other` on exactly the arm this change exists to explain.
        self.prof.lap(crate::telemetry::Phase::Ffn, t);
        Ok(())
    }
}

impl MxFp8MoeScratch {
    /// Enqueue one descriptor range on `lane`, accumulating into its row.
    ///
    /// A method here rather than on the engine: the caller holds `&mut` on this struct for the
    /// whole dispatch, and an `&self` engine method would borrow the pin against it.
    ///
    /// # Safety
    /// Every expert in `range` must already have its ticket wait enqueued on `lane.stream`;
    /// `descs`/`wexpert`/`moe_h` must be sized for `top_k` and outlive the stream's completion.
    unsafe fn range(
        &self,
        x_in: MoeX<'_>,
        (e_start, e_count): (usize, usize),
        lane: MoeLane,
    ) -> Result<()> {
        let cfg = x_in.cfg;
        // SAFETY: forwarded from this method's contract; `x` is the normed hidden the router
        // read, `hidden` wide and 256-byte aligned by construction.
        unsafe {
            launch_moe_expert_range_mxfp8_situ(
                x_in.x,
                cfg.hidden_size,
                cfg.intermediate_size,
                e_start,
                e_count,
                self.descs_host.len(),
                self.descs.ptr().cast::<ExpertDescF4>(),
                self.wexpert.ptr().cast::<f32>(),
                cfg.mlp_gate_softcap as f32,
                cfg.mlp_up_softcap as f32,
                self.moe_h.ptr() as *mut f32,
                lane.acc,
                1,
                lane.stream,
            )
        }
    }
}

/// Rewrite the descriptor table into LAUNCH order — residents at `[0, n_res)`, misses after —
/// and report `n_res`. `scores`/`sel` are the router's own arrays and `scale` its combine
/// normaliser, so the weight written is exactly the one the bf16 path applies.
///
/// **Launch order, not absolute ids.** The residents then form the ONE contiguous range the
/// launcher's `[e_start, e_count)` form exists for, and the fixed-point accumulator makes the
/// regrouping invisible: every expert's contribution comes from the same descriptor, activation
/// and weight whichever launch carries it, and integer addition is associative.
fn stage_launch_order(
    m: &mut MxFp8MoeScratch,
    (scores, sel): (&[f32], &[usize]),
    scale: f32,
) -> usize {
    m.launch_idx.clear();
    let mut n_res = 0;
    for resident in [true, false] {
        for i in 0..m.resolved.tickets.len() {
            if m.resolved.tickets[i].is_resident() == resident {
                m.launch_idx.push(i);
            }
        }
        if resident {
            n_res = m.launch_idx.len();
        }
    }
    for c in 0..m.launch_idx.len() {
        let i = m.launch_idx[c];
        fill_desc(&mut m.descs_host[c], &m.resolved.slots[i]);
        m.wexpert_launch[c] = scores[sel[i]] * scale;
    }
    n_res
}

/// One descriptor's six pointers, field by field. Field-at-a-time rather than a struct literal
/// because `crate::v4::engine::desc_of_f4` is the same literal and a second copy is a jscpd
/// clone; the token runs here are four tokens long.
fn fill_desc(d: &mut ExpertDescF4, s: &crate::routed::ExpertSlot) {
    d.gate_packed = s.gate.packed;
    d.gate_scale = s.gate.scale;
    d.up_packed = s.up.packed;
    d.up_scale = s.up.scale;
    d.down_packed = s.down.packed;
    d.down_scale = s.down.scale;
}
