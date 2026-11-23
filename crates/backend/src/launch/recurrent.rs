//! `recurrent.hip`'s launchers: the two gated delta rules, the short convolution and the
//! gated head norm. Each guard keeps its C code and order; the C's reasons are at tag
//! `archive/hip-kernels` and in the `launch_*` docs.

use super::{Checked, Rows, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{gated_delta_lds_bytes, gdn_lds_bytes, reduce_block, reduce_lds_bytes};
use rivoli_abi::kernarg::{GateHeads, GatedDelta, Gdn, ShortConv};
use std::ffi::c_void;

static GATED_DELTA_RECURRENT_F32: Kernel<GatedDelta> = Kernel::new(c"gated_delta_recurrent_f32");
static SHORT_CONV_SILU_F32: Kernel<ShortConv> = Kernel::new(c"short_conv_silu_f32");
static RMSNORM_GATE_HEADS_F32: Kernel<GateHeads> = Kernel::new(c"rmsnorm_gate_heads_f32");
static GDN_RECURRENT_F32: Kernel<Gdn> = Kernel::new(c"gdn_recurrent_f32");

/// The widest block HIP launches, which every width that IS (part of) the block inherits.
const MAX_BLOCK: i32 = 1024;

/// The C launchers' first two refusals: a count or width `<= 0` (1001), then a width past
/// [`MAX_BLOCK`] (1002).
fn extents(counts: &[i32], widths: &[i32]) -> Result<(), i32> {
    if counts.iter().chain(widths).any(|&x| x <= 0) {
        return Err(1001);
    }
    if widths.iter().any(|&w| w > MAX_BLOCK) {
        return Err(1002);
    }
    Ok(())
}

/// One block per head of `threads`, with `lds` bytes.
fn per_head(Rows(heads): Rows, threads: u32, lds: u32) -> Geometry {
    Geometry {
        blocks: heads as u32,
        threads,
        lds: LdsBytes(lds),
    }
}

fn gated_delta_checked(heads: Rows, a: &GatedDelta) -> Checked {
    let d = a.head_dim;
    extents(&[heads.0], &[d])?;
    if d & (d - 1) != 0 {
        return Err(1003);
    }
    // `!(lb >= -5 && lb < 0)`: fla's own range, NaN refused with it.
    if !(a.lower_bound >= -5.0 && a.lower_bound < 0.0) {
        return Err(1006);
    }
    Ok(per_head(heads, d as u32, gated_delta_lds_bytes(d as u32)))
}

/// # Safety
/// The `launch_gated_delta_recurrent_f32` contract.
pub(crate) unsafe fn rivoli_gated_delta_recurrent_f32(
    heads: Rows,
    a: GatedDelta,
    stream: *mut c_void,
) -> i32 {
    let checked = gated_delta_checked(heads, &a);
    unsafe { run(&GATED_DELTA_RECURRENT_F32, checked, stream, &a) }
}

fn short_conv_checked(a: &ShortConv) -> Checked {
    if a.channels <= 0 {
        return Err(1001);
    }
    if a.taps < 2 || a.taps > 16 {
        return Err(1002);
    }
    elementwise(i64::from(a.channels))
}

/// # Safety
/// The `launch_short_conv_silu_f32` contract.
pub(crate) unsafe fn rivoli_short_conv_silu_f32(a: ShortConv, stream: *mut c_void) -> i32 {
    unsafe { run(&SHORT_CONV_SILU_F32, short_conv_checked(&a), stream, &a) }
}

fn gate_heads_checked(heads: Rows, a: &GateHeads) -> Checked {
    extents(&[heads.0], &[a.head_dim])?;
    // Zero is legal (an exact RMSNorm); negative, NaN and inf are not.
    if !(a.eps >= 0.0 && a.eps < f32::INFINITY) {
        return Err(1006);
    }
    let block = reduce_block(a.head_dim as u32);
    Ok(per_head(heads, block, reduce_lds_bytes(block)))
}

/// # Safety
/// The `launch_rmsnorm_gate_heads_f32` contract.
pub(crate) unsafe fn rivoli_rmsnorm_gate_heads_f32(
    heads: Rows,
    a: GateHeads,
    stream: *mut c_void,
) -> i32 {
    let checked = gate_heads_checked(heads, &a);
    unsafe { run(&RMSNORM_GATE_HEADS_F32, checked, stream, &a) }
}

/// `v_heads` (the grid) against `qk_heads`, and `a.rep` — which the row derives as their
/// quotient — launches only once 1003 has refused a ragged ratio.
fn gdn_checked(v_heads: Rows, qk_heads: i32, a: &Gdn) -> Checked {
    extents(&[qk_heads, v_heads.0], &[a.dk, a.dv])?;
    if v_heads.0 % qk_heads != 0 {
        return Err(1003);
    }
    let block = reduce_block(a.dk.max(a.dv) as u32);
    Ok(per_head(v_heads, block, gdn_lds_bytes(a.dk as u32, block)))
}

/// # Safety
/// The `launch_gdn_recurrent_f32` contract.
pub(crate) unsafe fn rivoli_gdn_recurrent_f32(
    v_heads: Rows,
    qk_heads: i32,
    a: Gdn,
    stream: *mut c_void,
) -> i32 {
    let checked = gdn_checked(v_heads, qk_heads, &a);
    // `rep` is derived here, AFTER the guard: `gdn_checked` refuses `qk_heads <= 0` and a ragged
    // ratio (1003), so this division never meets a zero divisor and the kernel never receives a
    // ratio the guard did not admit. Until 2026-11-23 the `launchers!` row computed it, before
    // any guard, as `v_heads / qk_heads.max(1)` — the `.max(1)` hid the zero instead of refusing it.
    let a = Gdn {
        rep: checked.map_or(0, |_| v_heads.0 / qk_heads),
        ..a
    };
    unsafe { run(&GDN_RECURRENT_F32, checked, stream, &a) }
}
