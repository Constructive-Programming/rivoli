//! `linalg.hip`'s launchers: the wave-per-row GEMVs (`ceil(o_dim / ROWS_PER_BLOCK)` blocks
//! of `GEMV_BLOCK`), fp8's split-K route, and the VQ encoder.

use super::{Checked, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{GEMV_BLOCK, GEMV_FP8_LDS_BYTES, ROWS_PER_BLOCK, gemv_fp8_splitk_lds_bytes};
use rivoli_abi::kernarg::{Gemv, GemvFp8, Scaled, Vq, VqEncode};
use std::ffi::c_void;
use std::ptr::null_mut;

/// `[R = 1, R = 2]` for each batched GEMV — `nrow` picks the instantiation.
static GEMV_FP8: [Kernel<GemvFp8>; 2] = [Kernel::new(c"gemv_fp8"), Kernel::new(c"gemv_fp8_r2")];
static GEMV_FP8_SPLITK: [Kernel<GemvFp8>; 2] = [
    Kernel::new(c"gemv_fp8_splitk"),
    Kernel::new(c"gemv_fp8_splitk_r2"),
];
static GEMV_I8: [Kernel<Gemv<Scaled>>; 2] = [Kernel::new(c"gemv_i8"), Kernel::new(c"gemv_i8_r2")];
static GEMV_F32: [Kernel<Gemv<*const f32>>; 2] =
    [Kernel::new(c"gemv_f32"), Kernel::new(c"gemv_f32_r2")];
static GEMV_I4: Kernel<Gemv<Scaled>> = Kernel::new(c"gemv_i4");
static GEMV_VQ: Kernel<Gemv<Vq>> = Kernel::new(c"gemv_vq");
static VQ_ENCODE: Kernel<VqEncode> = Kernel::new(c"vq_encode");

/// Wave-per-row saturates DRAM only once `o_dim` fills the machine; below that (o_proj:
/// o_dim 6144, i_dim 16384) too few blocks launch, so long reductions go split-K. The plain
/// kernel measured 255-283 GB/s at i_dim <= 2048 and ~128 at >= 8192 (the C's note).
const GEMV_SPLITK_MIN_IDIM: i32 = 4096;

/// The GEMVs' extent refusal (1001), else the row grid: `ceil(o_dim / ROWS_PER_BLOCK)` blocks
/// of `GEMV_BLOCK` with `lds` bytes — the C's `ROW_GRID`/`ROW_BLOCK`.
fn row_grid<W>(g: &Gemv<W>, lds: u32) -> Checked {
    if g.o_dim <= 0 || g.i_dim <= 0 {
        return Err(1001);
    }
    Ok(Geometry {
        blocks: (g.o_dim as u32).div_ceil(ROWS_PER_BLOCK),
        threads: GEMV_BLOCK,
        lds: LdsBytes(lds),
    })
}

/// `nrow` in `{1, 2}` (1004) — `R` is a const generic, so only the two instantiations exist.
/// Yields `nrow`'s kernel index.
fn nrow_index(nrow: i32) -> Result<usize, i32> {
    match nrow {
        1 | 2 => Ok(nrow as usize - 1),
        _ => Err(1004),
    }
}

/// The fp8 GEMV's refusals in the C's order: extents or `block` (1001), a tile that is not a
/// power of two (1003, `blk_shift` needs one), then `nrow` (1004).
fn fp8_checked(a: &GemvFp8, nrow: i32) -> Result<usize, i32> {
    if a.block <= 0 {
        return Err(1001);
    }
    row_grid(&a.gemv, 0)?;
    if a.block & (a.block - 1) != 0 {
        return Err(1003);
    }
    nrow_index(nrow)
}

/// # Safety
/// The `launch_gemv_fp8` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_gemv_fp8(a: GemvFp8, nrow: i32) -> i32 {
    let g = &a.gemv;
    let r = match fp8_checked(&a, nrow) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let (kernel, geom) = if g.i_dim >= GEMV_SPLITK_MIN_IDIM {
        // One block per output row, every wave striding it.
        let geom = Geometry {
            blocks: g.o_dim as u32,
            threads: GEMV_BLOCK,
            lds: LdsBytes(gemv_fp8_splitk_lds_bytes(nrow as u32)),
        };
        (&GEMV_FP8_SPLITK[r], Ok(geom))
    } else {
        (&GEMV_FP8[r], row_grid(g, GEMV_FP8_LDS_BYTES))
    };
    unsafe { run(kernel, geom, null_mut(), &a) }
}

/// # Safety
/// The `launch_gemv_vq` contract.
pub(crate) unsafe fn rivoli_gemv_vq(a: Gemv<Vq>) -> i32 {
    unsafe { run(&GEMV_VQ, row_grid(&a, 0), null_mut(), &a) }
}

/// # Safety
/// The `launch_gemv_i4` contract.
pub(crate) unsafe fn rivoli_gemv_i4(a: Gemv<Scaled>) -> i32 {
    unsafe { run(&GEMV_I4, row_grid(&a, 0), null_mut(), &a) }
}

/// One of a batched GEMV's two instantiations, by `nrow`, on the row grid.
///
/// # Safety
/// `a` satisfies the kernels' contract; `stream` is live or null.
unsafe fn batched<W: Copy>(
    k: &[Kernel<Gemv<W>>; 2],
    a: &Gemv<W>,
    nrow: i32,
    stream: *mut c_void,
) -> i32 {
    // The C's order: extents (1001) before `nrow` (1004).
    let checked = row_grid(a, 0).and_then(|geom| Ok((nrow_index(nrow)?, geom)));
    match checked {
        Ok((r, geom)) => unsafe { run(&k[r], Ok(geom), stream, a) },
        Err(code) => code,
    }
}

/// # Safety
/// The `launch_gemv_i8` contract.
pub(crate) unsafe fn rivoli_gemv_i8(a: Gemv<Scaled>, nrow: i32) -> i32 {
    unsafe { batched(&GEMV_I8, &a, nrow, null_mut()) }
}

/// # Safety
/// The `launch_gemv_f32` contract.
pub(crate) unsafe fn rivoli_gemv_f32(a: Gemv<*const f32>, nrow: i32, stream: *mut c_void) -> i32 {
    unsafe { batched(&GEMV_F32, &a, nrow, stream) }
}

/// # Safety
/// The `launch_vq_encode` contract.
pub(crate) unsafe fn rivoli_vq_encode(a: VqEncode) -> i32 {
    unsafe { run(&VQ_ENCODE, elementwise(i64::from(a.n)), null_mut(), &a) }
}
