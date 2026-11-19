//! `kernels/blockindex.hip`'s kernels: V4's Hadamard-rotated fp4 activation simulator. (Its block
//! scorer takes `indexer::ScoreBlocks`, shared with the fp32 twin.)
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

/// `act_quant_f4_rotated` — rows of `d`, rotated and quantized IN PLACE; the row count is the grid.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct F4Rotate {
    pub x: *mut f32,
    pub d: i32,
}
const _: () = assert!(size_of::<F4Rotate>() == 16);
