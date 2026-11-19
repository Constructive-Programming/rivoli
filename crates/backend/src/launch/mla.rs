//! `mla.hip`'s launchers (tag `archive/hip-kernels`): MLA's absorb and value through the fp8
//! `kv_b` (each launching one of two kernels by token-row count — the `R` accumulators live in
//! registers, so only the instantiated widths exist), and V4's attention-side helpers.

use super::{Checked, Rows, any_nonpositive, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{
    FP8_LUT_LDS_BYTES, ROW_BLOCK, ROW_LDS_BYTES, ROWS_PER_BLOCK, WAVE_ROWS_BLOCK, absorb_nquad,
};
use rivoli_abi::kernarg::{
    ActQuantPrefix, Fp8KvB, GemvFp8Bf16, MlaAbsorb, MlaValue, QkNorm, RmsNormRows, RopeAdjacent,
};
use std::{ffi::c_void, ptr::null_mut};

static MLA_ABSORB_FP8: Kernel<MlaAbsorb> = Kernel::new(c"mla_absorb_fp8");
static MLA_ABSORB_FP8_R2: Kernel<MlaAbsorb> = Kernel::new(c"mla_absorb_fp8_r2");
static MLA_VALUE_FP8: Kernel<MlaValue> = Kernel::new(c"mla_value_fp8");
static MLA_VALUE_FP8_R2: Kernel<MlaValue> = Kernel::new(c"mla_value_fp8_r2");
static ACT_QUANT_F8_PREFIX: Kernel<ActQuantPrefix> = Kernel::new(c"act_quant_f8_prefix");
static RMSNORM_BATCH: Kernel<RmsNormRows> = Kernel::new(c"rmsnorm_batch");
static QK_NORM: Kernel<QkNorm> = Kernel::new(c"qk_norm");
static ROPE_ADJACENT: Kernel<RopeAdjacent> = Kernel::new(c"rope_adjacent");
static GEMV_FP8_BF16: Kernel<GemvFp8Bf16> = Kernel::new(c"gemv_fp8_bf16");

/// The two MLA launchers' shared refusals after their extents: a power-of-two tile (1003, the
/// kernels shift rather than divide) and an instantiated row count (1004), in the C's order.
fn kvb_checked(k: &Fp8KvB, nrow: i32) -> Result<(), i32> {
    if k.block & (k.block - 1) != 0 {
        Err(1003)
    } else if nrow != 1 && nrow != 2 {
        Err(1004)
    } else {
        Ok(())
    }
}

/// `count` waves of `ROWS_PER_BLOCK` per block, with the fp8 LUT.
fn wave_rows(count: i64) -> Geometry {
    Geometry {
        blocks: (count as u64).div_ceil(u64::from(ROWS_PER_BLOCK)) as u32,
        threads: WAVE_ROWS_BLOCK,
        lds: LdsBytes(FP8_LUT_LDS_BYTES),
    }
}

/// # Safety
/// The `launch_mla_absorb_fp8` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_mla_absorb_fp8(nrow: i32, a: MlaAbsorb) -> i32 {
    let k = &a.kvb;
    let checked = if any_nonpositive(&[k.h, a.qh, k.nope, k.vh, k.kvl, k.block]) {
        Err(1001)
    } else {
        kvb_checked(k, nrow).and_then(|()| {
            let mut g = elementwise(i64::from(k.h) * i64::from(absorb_nquad(k.kvl)))?;
            g.lds = LdsBytes(FP8_LUT_LDS_BYTES);
            Ok(g)
        })
    };
    let kernel = if nrow == 2 {
        &MLA_ABSORB_FP8_R2
    } else {
        &MLA_ABSORB_FP8
    };
    unsafe { run(kernel, checked, null_mut(), &a) }
}

/// The extents (1001), the tile and row count, then `kvl % 4 == 0` (1002) — the value rows'
/// dword load, which the absorb has a fallback for and this kernel does not.
///
/// # Safety
/// The `launch_mla_value_fp8` contract. Null stream.
pub(crate) unsafe fn rivoli_mla_value_fp8(nrow: i32, a: MlaValue) -> i32 {
    let k = &a.kvb;
    let checked = if any_nonpositive(&[k.h, k.nope, k.vh, k.kvl, k.block]) {
        Err(1001)
    } else {
        kvb_checked(k, nrow).and_then(|()| {
            if k.kvl & 3 != 0 {
                Err(1002)
            } else {
                Ok(wave_rows(i64::from(k.h) * i64::from(k.vh)))
            }
        })
    };
    let kernel = if nrow == 2 {
        &MLA_VALUE_FP8_R2
    } else {
        &MLA_VALUE_FP8
    };
    unsafe { run(kernel, checked, null_mut(), &a) }
}

/// Extents and `row_stride >= n` (1001), a strided copy (1002: `src != dst` needs contiguous
/// rows), whole groups (1004). One thread per group.
///
/// # Safety
/// The `launch_act_quant_f8_prefix` contract.
pub(crate) unsafe fn rivoli_act_quant_f8_prefix(a: ActQuantPrefix, stream: *mut c_void) -> i32 {
    let checked = if any_nonpositive(&[a.rows, a.n, a.block]) || a.row_stride < a.n {
        Err(1001)
    } else if a.src != a.dst.cast_const() && a.n != a.row_stride {
        Err(1002)
    } else if a.n % a.block != 0 {
        Err(1004)
    } else {
        elementwise(i64::from(a.rows) * i64::from(a.n / a.block))
    };
    unsafe { run(&ACT_QUANT_F8_PREFIX, checked, stream, &a) }
}

/// One block of `ROW_BLOCK` per row, `lds` bytes; `rows, d` positive (1001).
fn row_blocks(Rows(rows): Rows, d: i32, lds: u32) -> Checked {
    if any_nonpositive(&[rows, d]) {
        return Err(1001);
    }
    Ok(Geometry {
        blocks: rows as u32,
        threads: ROW_BLOCK,
        lds: LdsBytes(lds),
    })
}

/// # Safety
/// The `launch_rmsnorm_batch` contract.
pub(crate) unsafe fn rivoli_rmsnorm_batch(rows: Rows, a: RmsNormRows, stream: *mut c_void) -> i32 {
    let checked = row_blocks(rows, a.d, ROW_LDS_BYTES);
    unsafe { run(&RMSNORM_BATCH, checked, stream, &a) }
}

/// # Safety
/// The `launch_qk_norm` contract.
pub(crate) unsafe fn rivoli_qk_norm(rows: Rows, a: QkNorm, stream: *mut c_void) -> i32 {
    unsafe { run(&QK_NORM, row_blocks(rows, a.d, ROW_LDS_BYTES), stream, &a) }
}

/// Extents and `pos0 >= 0` (1001), `rd <= row_len` (1002), whole pairs (1005), whole position
/// groups (1004) — the C's order. One block per row, no LDS.
///
/// # Safety
/// The `launch_rope_adjacent` contract.
pub(crate) unsafe fn rivoli_rope_adjacent(rows: Rows, a: RopeAdjacent, stream: *mut c_void) -> i32 {
    let extents = [rows.0, a.row_len, a.rd, a.rows_per_pos];
    let checked = if any_nonpositive(&extents) || a.pos0 < 0 {
        Err(1001)
    } else if a.rd > a.row_len {
        Err(1002)
    } else if a.rd & 1 != 0 {
        Err(1005)
    } else if rows.0 % a.rows_per_pos != 0 {
        Err(1004)
    } else {
        row_blocks(rows, a.row_len, 0)
    };
    unsafe { run(&ROPE_ADJACENT, checked, stream, &a) }
}

/// Extents (1001), a power-of-two tile (1003), whole groups (1004). One wave per output.
///
/// # Safety
/// The `launch_gemv_fp8_bf16` contract.
pub(crate) unsafe fn rivoli_gemv_fp8_bf16(a: GemvFp8Bf16, stream: *mut c_void) -> i32 {
    let checked = if any_nonpositive(&[a.m, a.n_out, a.k, a.block, a.groups]) {
        Err(1001)
    } else if a.block & (a.block - 1) != 0 {
        Err(1003)
    } else if a.n_out % a.groups != 0 {
        Err(1004)
    } else {
        Ok(wave_rows(i64::from(a.m) * i64::from(a.n_out)))
    };
    unsafe { run(&GEMV_FP8_BF16, checked, stream, &a) }
}
