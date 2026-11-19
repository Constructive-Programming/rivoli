//! `kernels/blockindex.hip`'s geometry.
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

use super::{F32, FP4_QUANT_BLOCK};

/// `act_quant_f4_rotated`'s block: `d/2` butterflies at the widest stage, so 128 threads saturate
/// every stage at `d = 256` and waste nothing at 128.
pub const F4_ROTATE_BLOCK: u32 = 128;
/// The widest row the LDS tile holds; the launcher refuses wider (1002).
pub const ROTATE_D_MAX: u32 = 256;
/// The tile, `sh[ROTATE_D_MAX]`.
pub const F4_ROTATE_LDS_BYTES: u32 = ROTATE_D_MAX * F32;
const _: () =
    assert!(ROTATE_D_MAX.is_multiple_of(FP4_QUANT_BLOCK) && F4_ROTATE_BLOCK * 2 == ROTATE_D_MAX);
