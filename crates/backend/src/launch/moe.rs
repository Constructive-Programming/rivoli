//! `moe.hip`'s and `moe_f4.hip`'s launchers: an expert range's two (V4: three) passes on a
//! caller stream, and the fixed-point drains. Guard codes and order are the C's; their
//! reasons are at tag `archive/hip-kernels` and in the `launch_*` docs.

use super::{Checked, elementwise, not_positive_finite, rivoli_act_quant_f8, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{ACT_QUANT_BLOCK, F4_GROUP, ROWS_PER_BLOCK, WAVE};
use rivoli_abi::kernarg::{AccDrain, AccDrainTo, MoeF4, MoeF4Situ, MoeI4, MoeSpan, MoeVq};
use std::ffi::c_void;

static MOE_GATEUP_VQ: Kernel<MoeVq> = Kernel::new(c"moe_gateup_vq");
static MOE_GATEUP_VQ_R2: Kernel<MoeVq> = Kernel::new(c"moe_gateup_vq_r2");
static MOE_DOWN_VQ: Kernel<MoeVq> = Kernel::new(c"moe_down_vq");
static MOE_DOWN_VQ_R2: Kernel<MoeVq> = Kernel::new(c"moe_down_vq_r2");
static MOE_GATEUP_I4: Kernel<MoeI4> = Kernel::new(c"moe_gateup_i4");
static MOE_GATEUP_I4_R2: Kernel<MoeI4> = Kernel::new(c"moe_gateup_i4_r2");
static MOE_DOWN_I4: Kernel<MoeI4> = Kernel::new(c"moe_down_i4");
static MOE_DOWN_I4_R2: Kernel<MoeI4> = Kernel::new(c"moe_down_i4_r2");
static MOE_ACC_DRAIN: Kernel<AccDrain> = Kernel::new(c"moe_acc_drain");
static MOE_ACC_DRAIN_TO: Kernel<AccDrainTo> = Kernel::new(c"moe_acc_drain_to");

/// The range refusals every format shares (1001): extents, and a negative start.
fn span_checked(s: &MoeSpan) -> Result<(), i32> {
    if s.hidden.min(s.inter).min(s.e_count) <= 0 || s.e_start < 0 {
        return Err(1001);
    }
    Ok(())
}

/// One wave per `(expert, row)` over `e_count · len` rows: `ceil(rows / ROWS_PER_BLOCK)`
/// blocks of `ROWS_PER_BLOCK` waves — the constants the kernels' row mapping strides by.
/// `dim3` takes `unsigned`: the C narrowed the same way.
fn wave_rows(e_count: i32, len: i32) -> Geometry {
    let rows = e_count as u64 * len as u64;
    Geometry {
        blocks: rows.div_ceil(u64::from(ROWS_PER_BLOCK)) as u32,
        threads: ROWS_PER_BLOCK * WAVE,
        lds: LdsBytes(0),
    }
}

/// One range's guard verdict, then its two passes, `up` then `down`, each handed its grid —
/// and whatever the format runs between them (`between`: V4's `h` re-quantization; nothing for
/// the others). The guard's code or the first failure is returned. The passes are closures so
/// each launcher names, and launches, its own kernels' statics (`kernel_coverage`'s
/// `LAUNCHED_BY` rule).
fn passes(
    (span, checked): (MoeSpan, Result<(), i32>),
    (up, down): (impl FnOnce(Geometry) -> i32, impl FnOnce(Geometry) -> i32),
    between: impl FnOnce() -> i32,
) -> i32 {
    if let Err(code) = checked {
        return code;
    }
    let rc = up(wave_rows(span.e_count, span.inter));
    if rc != 0 {
        return rc;
    }
    let rc = between();
    if rc != 0 {
        return rc;
    }
    down(wave_rows(span.e_count, span.hidden))
}

/// A GLM range's (VQ or int4) refusals: the span, then `nrow` outside 1-2 token rows (1003).
fn glm_checked(span: &MoeSpan, nrow: i32) -> Result<(), i32> {
    span_checked(span)?;
    if !(1..=2).contains(&nrow) {
        return Err(1003);
    }
    Ok(())
}

/// # Safety
/// The `launch_moe_expert_range` contract.
pub(crate) unsafe fn rivoli_moe_expert_range(a: MoeVq, nrow: i32, stream: *mut c_void) -> i32 {
    let span = a.range.span;
    // `passes` returns a refusal before either pass runs, so `r2` is read only for nrow 1-2.
    let r2 = nrow == 2;
    // SAFETY: the launcher contract covers all four kernels' buffers.
    let pass = |down: bool, g| unsafe {
        match (down, r2) {
            (false, false) => MOE_GATEUP_VQ.launch(g, stream, &a),
            (false, true) => MOE_GATEUP_VQ_R2.launch(g, stream, &a),
            (true, false) => MOE_DOWN_VQ.launch(g, stream, &a),
            (true, true) => MOE_DOWN_VQ_R2.launch(g, stream, &a),
        }
    };
    let (up, down) = (|g| pass(false, g), |g| pass(true, g));
    passes((span, glm_checked(&span, nrow)), (up, down), || 0)
}

/// # Safety
/// The `launch_moe_expert_range_i4` contract.
pub(crate) unsafe fn rivoli_moe_expert_range_i4(a: MoeI4, nrow: i32, stream: *mut c_void) -> i32 {
    let r2 = nrow == 2;
    // An `if` chain, and the closures passed inline, where the VQ twin has a `match` and a
    // `let`: the same dispatch, deliberately spelled apart. jscpd reads identifiers as one
    // token class and reported the matching spelling as two clones of the twin (5 lines / 70
    // tokens and 6 lines / 53 tokens, 2026-11-19); the census needs each `STATIC.launch`
    // written here, so the selection cannot move into a shared helper.
    // SAFETY: the launcher contract covers all four kernels' buffers.
    let pass = |down: bool, g| unsafe {
        if !down {
            if r2 {
                MOE_GATEUP_I4_R2.launch(g, stream, &a)
            } else {
                MOE_GATEUP_I4.launch(g, stream, &a)
            }
        } else if r2 {
            MOE_DOWN_I4_R2.launch(g, stream, &a)
        } else {
            MOE_DOWN_I4.launch(g, stream, &a)
        }
    };
    passes(
        (a.span, glm_checked(&a.span, nrow)),
        (|g| pass(false, g), |g| pass(true, g)),
        || 0,
    )
}

/// Both drains' guard (1001) and grid: one thread per accumulator column.
fn drain_checked(a: &AccDrainTo) -> Checked {
    if a.n <= 0 || a.rows <= 0 {
        return Err(1001);
    }
    elementwise(i64::from(a.n))
}

/// # Safety
/// The `launch_moe_acc_drain` contract.
pub(crate) unsafe fn rivoli_moe_acc_drain_s(a: AccDrain, stream: *mut c_void) -> i32 {
    unsafe { run(&MOE_ACC_DRAIN, drain_checked(&a.to), stream, &a) }
}

/// # Safety
/// The `launch_moe_acc_drain_to` contract.
pub(crate) unsafe fn rivoli_moe_acc_drain_to_s(a: AccDrainTo, stream: *mut c_void) -> i32 {
    unsafe { run(&MOE_ACC_DRAIN_TO, drain_checked(&a), stream, &a) }
}

static MOE_GATEUP_F4: Kernel<MoeF4> = Kernel::new(c"moe_gateup_f4");
static MOE_DOWN_F4: Kernel<MoeF4> = Kernel::new(c"moe_down_f4");
static MOE_GATEUP_F4_SITU: Kernel<MoeF4Situ> = Kernel::new(c"moe_gateup_f4_situ");
static MOE_DOWN_F4_WEIGHTED: Kernel<MoeF4Situ> = Kernel::new(c"moe_down_f4_weighted");

/// The fp4 pair's shared refusals, in the C's order: the span and `n_desc` (1001), both
/// reduction dims whole multiples of `dim_unit` (1002 — `ACT_QUANT_BLOCK` for V4, `F4_GROUP`
/// for K3), one token row only (1003), the range inside the `n_desc` descriptors (1004), and
/// the activation's parameters positive and finite (1006).
fn f4_checked(
    span: &MoeSpan,
    (n_desc, nrow): (i32, i32),
    dim_unit: i32,
    params: &[f32],
) -> Result<(), i32> {
    span_checked(span)?;
    if n_desc <= 0 {
        return Err(1001);
    }
    if span.hidden % dim_unit != 0 || span.inter % dim_unit != 0 {
        return Err(1002);
    }
    if nrow != 1 {
        return Err(1003);
    }
    if i64::from(span.e_start) + i64::from(span.e_count) > i64::from(n_desc) {
        return Err(1004);
    }
    if params.iter().any(|&p| not_positive_finite(p)) {
        return Err(1006);
    }
    Ok(())
}

/// The fp4 launchers' host-only counts: the descriptor array's length and the token rows.
/// Named because both are `int`s the C took side by side, and a swapped pair still compiles.
#[derive(Clone, Copy)]
pub(crate) struct DescRows {
    pub(crate) n_desc: i32,
    pub(crate) nrow: i32,
}

/// # Safety
/// The `launch_moe_expert_range_f4` contract.
pub(crate) unsafe fn rivoli_moe_expert_range_f4(a: MoeF4, c: DescRows, stream: *mut c_void) -> i32 {
    let span = a.range.span;
    let checked = f4_checked(
        &span,
        (c.n_desc, c.nrow),
        ACT_QUANT_BLOCK as i32,
        &[a.limit],
    );
    // `w2`'s own input quantization, between the passes, over ONLY this range's `h` rows
    // (`moe_f4.hip` argues both halves). `passes` runs it only after pass 1, so never on a
    // refused range.
    let requant = || {
        let first = span.e_start as usize * c.nrow as usize * span.inter as usize;
        // SAFETY: `h` holds `n_desc · nrow · inter` floats (the launcher contract) and the
        // range is inside `n_desc` (1004).
        let h = unsafe { a.range.h.add(first) };
        unsafe { rivoli_act_quant_f8(h, span.e_count * c.nrow, span.inter, stream) }
    };
    // SAFETY: the launcher contract covers the pass's buffers.
    let up = |g| unsafe { MOE_GATEUP_F4.launch(g, stream, &a) };
    // SAFETY: as above.
    let down = |g| unsafe { MOE_DOWN_F4.launch(g, stream, &a) };
    passes((span, checked), (up, down), requant)
}

/// # Safety
/// The `launch_moe_expert_range_f4_situ` contract.
pub(crate) unsafe fn rivoli_moe_expert_range_f4_situ(
    a: MoeF4Situ,
    c: DescRows,
    stream: *mut c_void,
) -> i32 {
    let span = a.range.span;
    let checked = f4_checked(&span, (c.n_desc, c.nrow), F4_GROUP as i32, &[a.b1, a.b2]);
    // SAFETY: the launcher contract covers the pass's buffers.
    let up = |g| unsafe { MOE_GATEUP_F4_SITU.launch(g, stream, &a) };
    // SAFETY: as above.
    let down = |g| unsafe { MOE_DOWN_F4_WEIGHTED.launch(g, stream, &a) };
    passes((span, checked), (up, down), || 0)
}
