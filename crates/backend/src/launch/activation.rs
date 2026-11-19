//! `activation.hip`'s launchers: the gated combines, the row norms, the fp8 activation
//! quantizer and RoPE.

use super::{Checked, Rows, elementwise, not_positive_finite, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{ACT_QUANT_BLOCK, ACT_QUANT_THREADS, ROW_BLOCK, ROW_LDS_BYTES, WAVE};
use rivoli_abi::kernarg::{ActQuant, Gate, GateClamped, GateSitu, Rope, RowNorm, WeightlessNorm};
use std::ffi::c_void;
use std::ptr::null_mut;

static SWIGLU: Kernel<Gate> = Kernel::new(c"swiglu");
static SWIGLU_CLAMPED_BF16: Kernel<GateClamped> = Kernel::new(c"swiglu_clamped_bf16");
static SITU_GLU_F32: Kernel<GateSitu> = Kernel::new(c"situ_glu_f32");
static RMSNORM_SINGLE: Kernel<RowNorm> = Kernel::new(c"rmsnorm_single");
static RMSNORM_CENTERED_SINGLE: Kernel<RowNorm> = Kernel::new(c"rmsnorm_centered_single");
static RMSNORM_WEIGHTLESS_BATCH: Kernel<WeightlessNorm> = Kernel::new(c"rmsnorm_weightless_batch");
static ACT_QUANT_F8: Kernel<ActQuant> = Kernel::new(c"act_quant_f8");
static ROPE_INTERLEAVE: Kernel<Rope> = Kernel::new(c"rope_interleave");
static ROPE_SPLIT_HALF: Kernel<Rope> = Kernel::new(c"rope_split_half");

/// [`elementwise`] over a gated combine's `n`, then each extra positive-finite operand (1006).
fn gate_checked(a: &Gate, operands: &[f32]) -> Checked {
    let geom = elementwise(i64::from(a.n))?;
    if operands.iter().any(|&x| not_positive_finite(x)) {
        return Err(1006);
    }
    Ok(geom)
}

/// # Safety
/// The `launch_swiglu` contract.
pub(crate) unsafe fn rivoli_swiglu(a: Gate, stream: *mut c_void) -> i32 {
    unsafe { run(&SWIGLU, gate_checked(&a, &[]), stream, &a) }
}

/// # Safety
/// The `launch_swiglu_clamped_bf16` contract.
pub(crate) unsafe fn rivoli_swiglu_clamped_bf16(a: GateClamped, stream: *mut c_void) -> i32 {
    let checked = gate_checked(&a.gate, &[a.limit]);
    unsafe { run(&SWIGLU_CLAMPED_BF16, checked, stream, &a) }
}

/// # Safety
/// The `launch_situ_glu_f32` contract.
pub(crate) unsafe fn rivoli_situ_glu_f32(a: GateSitu, stream: *mut c_void) -> i32 {
    let checked = gate_checked(&a.gate, &[a.b1, a.b2]);
    unsafe { run(&SITU_GLU_F32, checked, stream, &a) }
}

/// One block of `ROW_BLOCK` threads per row, `ROW_LDS_BYTES` of tree.
fn rows(Rows(count): Rows) -> Geometry {
    Geometry {
        blocks: count as u32,
        threads: ROW_BLOCK,
        lds: LdsBytes(ROW_LDS_BYTES),
    }
}

/// # Safety
/// The `launch_rmsnorm_single` contract. No stream: the C launched on the null stream.
pub(crate) unsafe fn rivoli_rmsnorm_single(a: RowNorm) -> i32 {
    let checked = if a.n <= 0 {
        Err(1001)
    } else {
        Ok(rows(Rows(1)))
    };
    unsafe { run(&RMSNORM_SINGLE, checked, null_mut(), &a) }
}

/// # Safety
/// The `launch_rmsnorm_centered_single` contract.
pub(crate) unsafe fn rivoli_rmsnorm_centered_single(a: RowNorm, stream: *mut c_void) -> i32 {
    let checked = match (a.n <= 0, not_positive_finite(a.eps)) {
        (true, _) => Err(1001),
        (false, true) => Err(1002),
        (false, false) => Ok(rows(Rows(1))),
    };
    unsafe { run(&RMSNORM_CENTERED_SINGLE, checked, stream, &a) }
}

/// The weightless norm's three refusals, in the C's order. `scale` comes from a config
/// field, so it earns `eps`'s guard (the C's note).
fn weightless_checked(n_rows: Rows, a: &WeightlessNorm) -> Checked {
    if n_rows.0 <= 0 || a.d <= 0 {
        return Err(1001);
    }
    if not_positive_finite(a.eps) {
        return Err(1002);
    }
    if not_positive_finite(a.scale) {
        return Err(1003);
    }
    Ok(rows(n_rows))
}

/// # Safety
/// The `launch_rmsnorm_weightless_batch` contract.
pub(crate) unsafe fn rivoli_rmsnorm_weightless_batch(
    n_rows: Rows,
    a: WeightlessNorm,
    stream: *mut c_void,
) -> i32 {
    unsafe {
        run(
            &RMSNORM_WEIGHTLESS_BATCH,
            weightless_checked(n_rows, &a),
            stream,
            &a,
        )
    }
}

/// # Safety
/// The `launch_act_quant_f8` contract. Also called between the V4 fp4 expert range's passes
/// (`launch/moe.rs`); until that range was Rust it was a C symbol for `moe_f4.hip`.
pub(crate) unsafe fn rivoli_act_quant_f8(
    v: *mut f32,
    n_rows: i32,
    row_len: i32,
    stream: *mut c_void,
) -> i32 {
    if n_rows <= 0 || row_len <= 0 {
        return 1001;
    }
    if row_len % ACT_QUANT_BLOCK as i32 != 0 {
        return 1002;
    }
    // `nblk` sizes both the kernel's bound and the grid — one expression (the C's note).
    let nblk = n_rows as u64 * (row_len as u64 / u64::from(ACT_QUANT_BLOCK));
    let groups_per_block = u64::from(ACT_QUANT_THREADS / WAVE);
    let geom = Geometry {
        blocks: nblk.div_ceil(groups_per_block) as u32,
        threads: ACT_QUANT_THREADS,
        lds: LdsBytes(0),
    };
    unsafe { run(&ACT_QUANT_F8, Ok(geom), stream, &ActQuant { v, nblk }) }
}

/// Both RoPE launchers' guard and geometry — one block per segment, `seg / 2` threads. The
/// kernels index by `seg` and never read blockDim, so there is no constant to tie; `> 1024`
/// is HIP's block-size ceiling.
fn rope_checked(Rows(count): Rows, a: &Rope) -> Checked {
    // The C's two 1001 tests: a block HIP can launch (`seg` even, `seg/2 <= 1024`), and
    // segments that do not overlap when there is more than one.
    let half = a.seg / 2;
    let launchable = count > 0 && a.seg > 0 && a.seg % 2 == 0 && half <= 1024;
    let disjoint = count <= 1 || a.stride >= a.seg;
    if launchable && disjoint {
        Ok(Geometry {
            blocks: count as u32,
            threads: half as u32,
            lds: LdsBytes(0),
        })
    } else {
        Err(1001)
    }
}

/// # Safety
/// The `launch_rope_interleave` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_rope_interleave(count: Rows, a: Rope) -> i32 {
    unsafe { run(&ROPE_INTERLEAVE, rope_checked(count, &a), null_mut(), &a) }
}

/// # Safety
/// The `launch_rope_split_half` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_rope_split_half(count: Rows, a: Rope) -> i32 {
    unsafe { run(&ROPE_SPLIT_HALF, rope_checked(count, &a), null_mut(), &a) }
}
