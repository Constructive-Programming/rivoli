//! `blockindex.hip`'s launchers: V4's rotated fp4 activation simulator and the bf16 block scorer.

use super::{Rows, any_nonpositive, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{F4_ROTATE_BLOCK, F4_ROTATE_LDS_BYTES, FP4_QUANT_BLOCK, ROTATE_D_MAX};
use rivoli_abi::kernarg::{F4Rotate, ScoreBlocks};
use std::ffi::c_void;

static ACT_QUANT_F4_ROTATED: Kernel<F4Rotate> = Kernel::new(c"act_quant_f4_rotated");
static INDEX_SCORE_BLOCKS: Kernel<ScoreBlocks> = Kernel::new(c"index_score_blocks");

/// Extents (1001), the LDS tile `d <= ROTATE_D_MAX` (1002), a power-of-two `d` for the Hadamard
/// (1003), then whole fp4 blocks (1004 — reachable: `d = 16` clears 1003). One block per row.
///
/// # Safety
/// The `launch_act_quant_f4_rotated` contract.
pub(crate) unsafe fn rivoli_act_quant_f4_rotated(
    rows: Rows,
    a: F4Rotate,
    stream: *mut c_void,
) -> i32 {
    let d = a.d;
    let checked = if any_nonpositive(&[rows.0, d]) {
        Err(1001)
    } else if d > ROTATE_D_MAX as i32 {
        Err(1002)
    } else if d & (d - 1) != 0 {
        Err(1003)
    } else if d % FP4_QUANT_BLOCK as i32 != 0 {
        Err(1004)
    } else {
        Ok(Geometry {
            blocks: rows.0 as u32,
            threads: F4_ROTATE_BLOCK,
            lds: LdsBytes(F4_ROTATE_LDS_BYTES),
        })
    };
    unsafe { run(&ACT_QUANT_F4_ROTATED, checked, stream, &a) }
}

/// `s`, `heads`, `hd` positive (1001), then `n_comp` (1005 — NOT a no-op success: `n_comp == 0`
/// is a real state where the reference produces an empty score tensor, so a launch there is the
/// caller's misunderstanding, named). `ceil(s · n_comp / 256)` blocks of 256.
///
/// # Safety
/// The `launch_index_score_blocks` contract.
pub(crate) unsafe fn rivoli_index_score_blocks(a: ScoreBlocks, stream: *mut c_void) -> i32 {
    let checked = if any_nonpositive(&[a.s, a.heads, a.hd]) {
        Err(1001)
    } else if a.n_comp <= 0 {
        Err(1005)
    } else {
        elementwise(i64::from(a.s) * i64::from(a.n_comp))
    };
    unsafe { run(&INDEX_SCORE_BLOCKS, checked, stream, &a) }
}
