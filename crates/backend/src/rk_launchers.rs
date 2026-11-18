//! The host halves of the Rust kernels — what `kernels/activation.hip`, `residual.hip` and
//! `async.hip`'s `extern "C" int rivoli_*` launchers did: the argument guards, the grid
//! geometry, the launch. Each keeps its C twin's name and return codes (`0`, POSITIVE guard
//! code, `-(hipError_t)`) exactly, so a `launch_x => rivoli_x(..)` row in
//! `hip_linalg.rs`/`hip_blocks.rs` maps the status unchanged and
//! `crates/engine/tests/bit_identity/` can compare guard codes against the tagged C
//! launcher it replaces.
//!
//! What changed is the parameter list: a launcher takes the kernel's
//! `rivoli_abi::kernarg` block (built by its `launchers!` row) plus whatever only the HOST
//! reads — a stream, a count that becomes the grid, a checkpoint constant to refuse. The
//! guards' REASONS live in the C at tag `archive/hip-kernels` and in the `launch_*` docs;
//! this file adds the geometry tie — every block and LDS size below is the
//! `rivoli_abi::geom` constant the kernel itself strides by, never a literal.

#![cfg(feature = "rocm")]

use crate::module::{Geometry, Kernel};
use rivoli_abi::geom::{
    ACT_QUANT_BLOCK, ACT_QUANT_THREADS, ATTN_RES_BLOCK, ATTN_RES_LDS_BYTES, ATTN_RES_MAX_SRC,
    ELEMENTWISE_BLOCK, GATED_RESIDUAL_MAX_NR, HC_MULT, HC_PRE_BLOCK, HC_PRE_LDS_BYTES, ROW_BLOCK,
    ROW_LDS_BYTES, TOUCH_BLOCK, WAVE,
};
use rivoli_abi::kernarg::{
    ActQuant, AttnRes, Gate, GateClamped, GateSitu, GatedResidual, HcPost, HcPre, Rope, RowNorm,
    Touch, WeightlessNorm,
};
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
static HC_PRE: Kernel<HcPre> = Kernel::new(c"hc_pre");
static HC_POST: Kernel<HcPost> = Kernel::new(c"hc_post");
static ATTN_RES: Kernel<AttnRes> = Kernel::new(c"attn_res");
static GATED_RESIDUAL_COLLAPSE_F32: Kernel<GatedResidual> =
    Kernel::new(c"gated_residual_collapse_f32");

/// A launch's block count — one block per row, segment or token — as the C `int` the
/// launcher took. A unit, not a bare `i32`, because it sits beside other `int`s (`hc`, a
/// width, a row length) that every one of these signatures could transpose it with.
#[derive(Clone, Copy)]
pub(crate) struct Rows(pub(crate) i32);

/// A guard's verdict: the geometry to launch at, or the POSITIVE code to return.
type Checked = Result<Geometry, i32>;

/// Launch `k` with `args` if the guard passed, else return its code — the one tail every
/// launcher below shares.
///
/// # Safety
/// `args` is `k`'s argument block and satisfies its buffer contract; `stream` is live or null.
unsafe fn run<A: Copy>(k: &Kernel<A>, checked: Checked, stream: *mut c_void, args: &A) -> i32 {
    match checked {
        Ok(geom) => unsafe { k.launch(geom, stream, args) },
        Err(code) => code,
    }
}

/// `!(x > 0 && x < INFINITY)` — the C guards' one-compare form, which refuses NaN, zero,
/// negatives and +inf together (`activation.hip`'s `swiglu_clamped_bf16` note says why each).
fn not_positive_finite(x: f32) -> bool {
    !(x > 0.0 && x < f32::INFINITY)
}

/// `dim3((n + 255) / 256), dim3(256)`, no LDS; `n <= 0` refused (1001).
fn elementwise(n: i64) -> Checked {
    if n <= 0 {
        return Err(1001);
    }
    // `dim3` takes `unsigned`: the C narrowed the same way.
    Ok((
        (n as u64).div_ceil(u64::from(ELEMENTWISE_BLOCK)) as u32,
        ELEMENTWISE_BLOCK,
        0,
    ))
}

/// [`elementwise`] over a gated combine's `n`, then each extra positive-finite operand (1006).
fn gate_checked(a: &Gate, operands: &[f32]) -> Checked {
    let geom = elementwise(i64::from(a.n))?;
    if operands.iter().any(|&x| not_positive_finite(x)) {
        return Err(1006);
    }
    Ok(geom)
}

// ── activation.hip ──────────────────────────────────────────────────────────────────────

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
    (count as u32, ROW_BLOCK, ROW_LDS_BYTES)
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
/// The `launch_act_quant_f8` contract.
///
/// **Also a C symbol**, and the only one in this file: `kernels/moe_f4.hip`'s fp4 expert
/// launcher calls `rivoli_act_quant_f8` between its two passes, C to C, and that file is
/// still hipcc's until Phase 1's L2 lane ports it. The signature was C-shaped already, so
/// the export costs the two attributes; drop them (and this paragraph) when moe_f4 lands.
#[unsafe(no_mangle)]
pub(crate) unsafe extern "C" fn rivoli_act_quant_f8(
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
    let geom = (nblk.div_ceil(groups_per_block) as u32, ACT_QUANT_THREADS, 0);
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
        Ok((count as u32, half as u32, 0))
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

// ── residual.hip ────────────────────────────────────────────────────────────────────────

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
        (a.iters >= 1)
            .then_some((s.0 as u32, HC_PRE_BLOCK, HC_PRE_LDS_BYTES))
            .ok_or(1003)
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
    Ok((tokens as u32, ATTN_RES_BLOCK, ATTN_RES_LDS_BYTES))
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

// ── async.hip ───────────────────────────────────────────────────────────────────────────

static TOUCH_REGION: Kernel<Touch> = Kernel::new(c"touch_region");

/// async.hip's grid cap: at most this many blocks walk the region grid-stride. Host-only —
/// the kernel is handed the resulting stride, never the cap.
const TOUCH_MAX_BLOCKS: u64 = 1024;

/// Read `n` bytes of `src` at full width on `stream`, value discarded (`--arena-refresh`).
/// `n` must be a non-zero multiple of 16 (1002); the grid is `ceil(n/16/256)` blocks capped
/// at [`TOUCH_MAX_BLOCKS`]. Returns async.hip's convention — `0`, a POSITIVE guard code, or
/// `HIP_ERR_BASE - hipError_t` — because `fetch/stage.rs` was written against it.
///
/// The record behind this read (a MITIGATION for GLM's nondeterminism with a measured price,
/// not a root-cause fix; do not delete it without re-running `tests/determinism-glm.sh`) is
/// `kernels/async.hip` at tag `archive/hip-kernels` and
/// `docs/investigations/glm-nondeterminism-closeout.md`.
///
/// # Safety
/// `src` is valid for `n` bytes, `stream` is live, `sink` is one writable device-visible u64.
pub unsafe fn rivoli_touch_region_async(
    src: *const c_void,
    n: u64,
    stream: *mut c_void,
    sink: *mut u64,
) -> i32 {
    if n == 0 || !n.is_multiple_of(16) {
        return 1002;
    }
    let n4 = n / 16;
    let blocks = n4.div_ceil(u64::from(TOUCH_BLOCK)).min(TOUCH_MAX_BLOCKS) as u32;
    let stride = u64::from(blocks) * u64::from(TOUCH_BLOCK);
    let args = Touch {
        src,
        sink,
        n4,
        stride,
    };
    // SAFETY: the caller's contract.
    let rc = unsafe { TOUCH_REGION.launch((blocks, TOUCH_BLOCK, 0), stream, &args) };
    // `launch` returns `-(e)`; async.hip folded a launch error to `HIP_ERR_BASE - e`.
    if rc == 0 {
        0
    } else {
        crate::hiprt::HIP_ERR_BASE + rc
    }
}
