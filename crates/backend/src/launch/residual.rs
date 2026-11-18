//! `residual.hip`'s launchers: the mHC pair, the attention residual and the gated residual
//! collapse.

use super::{Checked, Rows, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{
    ATTN_RES_BLOCK, ATTN_RES_LDS_BYTES, ATTN_RES_MAX_SRC, GATED_RESIDUAL_MAX_NR, HC_MULT,
    HC_PRE_BLOCK, HC_PRE_LDS_BYTES,
};
use rivoli_abi::kernarg::{AttnRes, GatedResidual, HcPost, HcPre};
use std::ffi::c_void;
use std::ptr::null_mut;

static HC_PRE: Kernel<HcPre> = Kernel::new(c"hc_pre");
static HC_POST: Kernel<HcPost> = Kernel::new(c"hc_post");
static ATTN_RES: Kernel<AttnRes> = Kernel::new(c"attn_res");
static GATED_RESIDUAL_COLLAPSE_F32: Kernel<GatedResidual> =
    Kernel::new(c"gated_residual_collapse_f32");

/// The mHC pair's shared refusals: extents (1001), then the packing constant (1002).
fn hc_checked(Rows(s): Rows, hc: i32, dim: i32) -> Result<(), i32> {
    if s <= 0 || dim <= 0 {
        return Err(1001);
    }
    if hc != HC_MULT as i32 {
        return Err(1002);
    }
    Ok(())
}

/// # Safety
/// The `launch_hc_pre` contract.
pub(crate) unsafe fn rivoli_hc_pre(s: Rows, hc: i32, a: HcPre, stream: *mut c_void) -> i32 {
    let checked = hc_checked(s, hc, a.dim).and_then(|()| {
        // `iters < 1` (1003): the Sinkhorn needs its first pass.
        let geom = Geometry {
            blocks: s.0 as u32,
            threads: HC_PRE_BLOCK,
            lds: LdsBytes(HC_PRE_LDS_BYTES),
        };
        (a.iters >= 1).then_some(geom).ok_or(1003)
    });
    unsafe { run(&HC_PRE, checked, stream, &a) }
}

/// # Safety
/// The `launch_hc_post` contract.
pub(crate) unsafe fn rivoli_hc_post(hc: i32, a: HcPost, stream: *mut c_void) -> i32 {
    let n = i64::from(a.s) * i64::from(HC_MULT) * i64::from(a.dim);
    let checked = hc_checked(Rows(a.s), hc, a.dim).and_then(|()| elementwise(n));
    unsafe { run(&HC_POST, checked, stream, &a) }
}

/// The attention-residual refusals: extents (1001), stack depth (1003), and a token stride
/// shorter than the live stack (1005), which would overlap consecutive tokens.
fn attn_res_checked(Rows(tokens): Rows, a: &AttnRes) -> Checked {
    if a.n <= 0 || tokens <= 0 {
        return Err(1001);
    }
    if a.nsrc <= 0 || a.nsrc > ATTN_RES_MAX_SRC as i32 {
        return Err(1003);
    }
    if a.src_stride < a.nsrc as u64 * a.n as u64 {
        return Err(1005);
    }
    Ok(Geometry {
        blocks: tokens as u32,
        threads: ATTN_RES_BLOCK,
        lds: LdsBytes(ATTN_RES_LDS_BYTES),
    })
}

/// # Safety
/// The `launch_attn_res` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_attn_res(tokens: Rows, a: AttnRes) -> i32 {
    unsafe { run(&ATTN_RES, attn_res_checked(tokens, &a), null_mut(), &a) }
}

/// # Safety
/// The `launch_gated_residual_collapse_f32` contract.
pub(crate) unsafe fn rivoli_gated_residual_collapse_f32(
    a: GatedResidual,
    stream: *mut c_void,
) -> i32 {
    // The C's order: both extents (1001) before the width cap (1003).
    let checked = match (
        a.n_r <= 0 || a.dim <= 0,
        a.n_r > GATED_RESIDUAL_MAX_NR as i32,
    ) {
        (true, _) => Err(1001),
        (false, true) => Err(1003),
        (false, false) => elementwise(i64::from(a.dim)),
    };
    unsafe { run(&GATED_RESIDUAL_COLLAPSE_F32, checked, stream, &a) }
}
