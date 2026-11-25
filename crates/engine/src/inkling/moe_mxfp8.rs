//! Inkling 2 Small's routed MoE dispatch: the `.mxfp8` experts out of the routed pool, the two
//! streams their compute and their fills run on, and the fixed-point accumulator they land in.
//!
//! Split from `decode.rs` when the two-stream dispatch crossed the workspace's 800-line soft
//! cap — a split by cohesion, not convenience: `decode.rs` owns the ORDER the blocks run in,
//! and this file owns one block's device composition.
//!
//! **The order is `crate::v4::moe`'s, measured there**: residents first as one contiguous range
//! on the compute stream, then each miss alone behind its own ticket on the miss stream. The
//! pool, the tickets and the O_DIRECT fills are the same machinery, so the shape that overlaps
//! them well is the same shape.

use super::engine::{InklingEngine, MOE_ACC_ROWS, MxFp8MoeScratch};
use super::forward::{p, pc};
use crate::device::as_le_bytes;
use crate::fetch::asyncfetch::FetchFolds;
use crate::routed::Selection;
use anyhow::{Context, Result};
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_backend::{
    ExpertDescF4, NULL_STREAM, device_sync, launch_moe_acc_drain_to,
    launch_moe_expert_range_mxfp8_situ,
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
    /// measured +825 µs on a 6-miss layer. The drain adds both and resets both.
    pub(super) fn moe_mxfp8(&mut self, layer: usize, scale: f32) -> Result<()> {
        let hidden = self.cfg.hidden_size;
        let k = self.route_sel.len();
        let m = self
            .moe8
            .as_mut()
            .context("the .mxfp8 backend has no routed scratch")?;
        // One borrow of the pool for the whole dispatch: it and `m` are disjoint fields, and
        // the pin has no other pool reader on this path.
        let routed = self
            .pin
            .routed_mut()
            .context("the .mxfp8 pin holds no routed pool")?;
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
        // JOIN 1: `xn` was written on the NULL stream and both expert streams are
        // `hipStreamNonBlocking`, so they do not implicitly join it.
        device_sync()?;
        let x_in = MoeX {
            x: pc(&self.sc.xn),
            cfg: self.cfg,
        };
        let acc = m.acc.ptr_mut().cast::<u64>();
        let (cs, ms) = (m.compute_stream.raw(), m.miss_stream.raw());
        // SAFETY: `acc` is `MOE_ACC_ROWS * hidden` u64; row 1 is in bounds.
        let miss = MoeLane {
            acc: unsafe { acc.add(hidden) },
            stream: ms,
        };
        let lane = MoeLane { acc, stream: cs };
        for &i in &m.launch_idx[..n_res] {
            routed.wait_on(m.resolved.tickets[i], cs)?;
        }
        if n_res > 0 {
            // SAFETY: every expert in the range has its wait enqueued above, `descs` holds
            // `k` filled entries, and `moe_h` is `k · inter`.
            unsafe { m.range(x_in, (0, n_res), lane)? };
        }
        // The compact index ascends in the order `stage_launch_order` wrote, so straggler `j`
        // reads descriptor `n_res + j`.
        if k > n_res {
            for (j, &i) in m.launch_idx[n_res..].iter().enumerate() {
                routed.wait_on(m.resolved.tickets[i], ms)?;
                // SAFETY: this expert's bytes are gated by the wait just enqueued on the same
                // stream, and row 1 of the accumulator is this stream's alone.
                unsafe { m.range(x_in, (n_res + j, 1), miss)? };
            }
        }
        // JOIN 2: the drain's own contract — every stream that accumulated into `acc` retired.
        device_sync()?;
        // SAFETY: `moe_acc` is `hidden` f32 writable, `acc` is `MOE_ACC_ROWS · hidden` u64, and
        // both accumulate rows retired at JOIN 2.
        unsafe {
            launch_moe_acc_drain_to(p(&self.sc.moe_acc), acc, hidden, MOE_ACC_ROWS, NULL_STREAM)
        }
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
