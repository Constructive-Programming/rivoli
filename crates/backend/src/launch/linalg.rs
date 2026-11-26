//! `linalg.hip`'s launchers: the wave-per-row GEMVs (`ceil(o_dim / ROWS_PER_BLOCK)` blocks
//! of `GEMV_BLOCK`), fp8's split-K route, and the VQ encoder.

use super::{Checked, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{GEMV_BLOCK, GEMV_FP8_LDS_BYTES, ROWS_PER_BLOCK, gemv_fp8_splitk_lds_bytes};
use rivoli_abi::kernarg::{Gemv, GemvFp8, Scaled, Vq, VqEncode};
use std::ffi::c_void;
use std::ptr::null_mut;

// One handle per instantiation: `nrow` (1 or 2) picks `R`, and each launcher names every
// static it launches in its own body (the census's `LAUNCHED_BY` rule).
static GEMV_FP8: Kernel<GemvFp8> = Kernel::new(c"gemv_fp8");
static GEMV_FP8_R2: Kernel<GemvFp8> = Kernel::new(c"gemv_fp8_r2");
static GEMV_FP8_SPLITK: Kernel<GemvFp8> = Kernel::new(c"gemv_fp8_splitk");
static GEMV_FP8_SPLITK_R2: Kernel<GemvFp8> = Kernel::new(c"gemv_fp8_splitk_r2");
static GEMV_I8: Kernel<Gemv<Scaled>> = Kernel::new(c"gemv_i8");
static GEMV_I8_R2: Kernel<Gemv<Scaled>> = Kernel::new(c"gemv_i8_r2");
static GEMV_F32: Kernel<Gemv<*const f32>> = Kernel::new(c"gemv_f32");
static GEMV_F32_R2: Kernel<Gemv<*const f32>> = Kernel::new(c"gemv_f32_r2");
static GEMV_BF16: Kernel<Gemv<*const u16>> = Kernel::new(c"gemv_bf16");
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
fn nrow_checked(nrow: i32) -> Result<(), i32> {
    if nrow == 1 || nrow == 2 {
        Ok(())
    } else {
        Err(1004)
    }
}

/// A batched GEMV's refusals in the C's order — extents (1001), then `nrow` (1004) — else its
/// row grid.
fn batched_checked<W>(a: &Gemv<W>, nrow: i32) -> Checked {
    let geom = row_grid(a, 0)?;
    nrow_checked(nrow)?;
    Ok(geom)
}

/// The fp8 GEMV's refusals in the C's order — extents or `block` (1001), a tile that is not a
/// power of two (1003, `blk_shift` needs one), then `nrow` (1004) — else its geometry: the
/// row grid, or on the split-K route one block per output row.
fn fp8_checked(a: &GemvFp8, nrow: i32, splitk: bool) -> Checked {
    let g = &a.gemv;
    if a.block <= 0 {
        return Err(1001);
    }
    let rows = row_grid(g, GEMV_FP8_LDS_BYTES)?;
    if a.block & (a.block - 1) != 0 {
        return Err(1003);
    }
    nrow_checked(nrow)?;
    if !splitk {
        return Ok(rows);
    }
    Ok(Geometry {
        blocks: g.o_dim as u32,
        threads: GEMV_BLOCK,
        lds: LdsBytes(gemv_fp8_splitk_lds_bytes(nrow as u32)),
    })
}

/// # Safety
/// The `launch_gemv_fp8` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_gemv_fp8(a: GemvFp8, nrow: i32) -> i32 {
    // ONE route decision, so the kernel and its geometry cannot disagree.
    let splitk = a.gemv.i_dim >= GEMV_SPLITK_MIN_IDIM;
    let checked = fp8_checked(&a, nrow, splitk);
    // `_` is `nrow == 1`: any other `nrow` is already a refusal in `checked`, which `run`
    // returns unlaunched.
    match (splitk, nrow) {
        (true, 2) => unsafe { run(&GEMV_FP8_SPLITK_R2, checked, null_mut(), &a) },
        (true, _) => unsafe { run(&GEMV_FP8_SPLITK, checked, null_mut(), &a) },
        (false, 2) => unsafe { run(&GEMV_FP8_R2, checked, null_mut(), &a) },
        (false, _) => unsafe { run(&GEMV_FP8, checked, null_mut(), &a) },
    }
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

/// # Safety
/// The `launch_gemv_i8` contract.
pub(crate) unsafe fn rivoli_gemv_i8(a: Gemv<Scaled>, nrow: i32) -> i32 {
    let checked = batched_checked(&a, nrow);
    // `else` is `nrow == 1`: any other `nrow` is already a refusal in `checked`, which `run`
    // returns unlaunched.
    if nrow == 2 {
        unsafe { run(&GEMV_I8_R2, checked, null_mut(), &a) }
    } else {
        unsafe { run(&GEMV_I8, checked, null_mut(), &a) }
    }
}

/// # Safety
/// The `launch_gemv_f32` contract.
pub(crate) unsafe fn rivoli_gemv_f32(a: Gemv<*const f32>, nrow: i32, stream: *mut c_void) -> i32 {
    let checked = batched_checked(&a, nrow);
    // `else` as in `rivoli_gemv_i8`.
    if nrow == 2 {
        unsafe { run(&GEMV_F32_R2, checked, stream, &a) }
    } else {
        unsafe { run(&GEMV_F32, checked, stream, &a) }
    }
}

/// The bf16 GEMV's refusals and grid are the family's: extents (1001), then the row grid with
/// no LDS. Single-row only — a batched caller keeps `gemm_bf16`, whose `R`-row shapes this
/// kernel does not have.
///
/// # Safety
/// The `launch_gemv_bf16` contract.
pub(crate) unsafe fn rivoli_gemv_bf16(a: Gemv<*const u16>, stream: *mut c_void) -> i32 {
    unsafe { run(&GEMV_BF16, row_grid(&a, 0), stream, &a) }
}

/// # Safety
/// The `launch_vq_encode` contract.
pub(crate) unsafe fn rivoli_vq_encode(a: VqEncode) -> i32 {
    unsafe { run(&VQ_ENCODE, elementwise(i64::from(a.n)), null_mut(), &a) }
}
