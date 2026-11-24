//! `activation.hip`'s launchers: the gated combines, the row norms, the fp8 activation
//! quantizer and RoPE.

use super::{Checked, Rows, any_nonpositive, elementwise, not_positive_finite, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{
    ACT_QUANT_BLOCK, ACT_QUANT_THREADS, ELEMENTWISE_BLOCK, GEMV_FP8_LDS_BYTES, ROW_BLOCK,
    ROW_LDS_BYTES, WAVE,
};
use rivoli_abi::kernarg::{
    ActQuant, Gate, GateClamped, GateSitu, KvScatter, L2Clip, LogScaledL2Clip, MessageGate,
    Mxfp8Dequant, Nvfp4Dequant, RelBiasGather, Rope, RowNorm, ScaledL2Clip, Sconv, SinkAttend,
    WeightlessNorm,
};
use std::{ffi::c_void, ptr::null_mut};

static SWIGLU: Kernel<Gate> = Kernel::new(c"swiglu");
static SWIGLU_CLAMPED_BF16: Kernel<GateClamped> = Kernel::new(c"swiglu_clamped_bf16");
static SITU_GLU_F32: Kernel<GateSitu> = Kernel::new(c"situ_glu_f32");
static L2_CLIP: Kernel<L2Clip> = Kernel::new(c"l2_clip");
static SCALED_L2_CLIP: Kernel<ScaledL2Clip> = Kernel::new(c"scaled_l2_clip");
static LOG_SCALED_L2_CLIP: Kernel<LogScaledL2Clip> = Kernel::new(c"log_scaled_l2_clip");
static MXFP8_DEQUANT: Kernel<Mxfp8Dequant> = Kernel::new(c"mxfp8_dequant");
static NVFP4_DEQUANT: Kernel<Nvfp4Dequant> = Kernel::new(c"nvfp4_dequant");
static SCONV: Kernel<Sconv> = Kernel::new(c"sconv");
static SINK_ATTEND: Kernel<SinkAttend> = Kernel::new(c"sink_attend");
static MESSAGE_GATE: Kernel<MessageGate> = Kernel::new(c"message_gate");
static KV_SCATTER: Kernel<KvScatter> = Kernel::new(c"kv_scatter");
static REL_BIAS_GATHER: Kernel<RelBiasGather> = Kernel::new(c"rel_bias_gather");
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

/// The clip family's shared guard: `rows > 0` and `n > 0` (1001), then a positive-finite
/// `threshold` (1006). One block of `ROW_BLOCK` threads per row — the batching is why `rows` is a
/// launcher argument rather than one launch per head.
fn clip_checked(n_rows: Rows, n: i32, threshold: f32) -> Checked {
    if n_rows.0 <= 0 || n <= 0 {
        return Err(1001);
    }
    if not_positive_finite(threshold) {
        return Err(1006);
    }
    Ok(rows(n_rows))
}

/// # Safety
/// The `launch_l2_clip` contract. inkling's per-head L2 clip; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_l2_clip(n_rows: Rows, a: L2Clip, stream: *mut c_void) -> i32 {
    let checked = clip_checked(n_rows, a.n, a.threshold);
    unsafe { run(&L2_CLIP, checked, stream, &a) }
}

/// # Safety
/// The `launch_scaled_l2_clip` contract. inkling's sliding q/k norm; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_scaled_l2_clip(
    n_rows: Rows,
    a: ScaledL2Clip,
    stream: *mut c_void,
) -> i32 {
    let checked = clip_checked(n_rows, a.clip.n, a.clip.threshold);
    unsafe { run(&SCALED_L2_CLIP, checked, stream, &a) }
}

/// # Safety
/// The `launch_log_scaled_l2_clip` contract. inkling's full-layer q/k norm; no C twin. `tau` is a
/// second positive-finite operand (1006): it multiplies the clipped vector, so a zero or
/// non-finite value would zero or NaN the whole head while every other guard held.
pub(crate) unsafe fn rivoli_log_scaled_l2_clip(
    n_rows: Rows,
    a: LogScaledL2Clip,
    stream: *mut c_void,
) -> i32 {
    let c = a.scaled.clip;
    let checked = match clip_checked(n_rows, c.n, c.threshold) {
        Err(code) => Err(code),
        Ok(_) if not_positive_finite(a.tau) => Err(1006),
        Ok(geom) => Ok(geom),
    };
    unsafe { run(&LOG_SCALED_L2_CLIP, checked, stream, &a) }
}

/// Elementwise geometry carrying the fp8 LUT's LDS (the e4m3 block scale both expert dequants
/// read), over the 64-bit grid `n` gives — an expert layer's `n` passes `i32`. `n == 0` and a
/// block count past `u32` are refused (1001) rather than truncated into a short launch.
fn elementwise_fp8_lds_u64(n: u64) -> Checked {
    if n == 0 {
        return Err(1001);
    }
    let blocks = n.div_ceil(u64::from(ELEMENTWISE_BLOCK));
    if blocks > u64::from(u32::MAX) {
        return Err(1001);
    }
    Ok(Geometry {
        blocks: blocks as u32,
        threads: ELEMENTWISE_BLOCK,
        lds: LdsBytes(GEMV_FP8_LDS_BYTES),
    })
}

/// The MXFP8 decode's geometry: elementwise over `n` carrying the fp8 LUT's LDS, `cols` a
/// positive multiple of 32 (the row-wise scale block, T1) that divides `n` — the kernel's
/// `row = i / cols` is only an expert/row split while the grid is a whole number of rows. Bad
/// extent or grid → 1001.
fn mxfp8_checked(n: u64, cols: i32) -> Checked {
    if n == 0 || cols <= 0 {
        return Err(1001);
    }
    if cols % 32 != 0 || !n.is_multiple_of(cols as u64) {
        return Err(1001);
    }
    elementwise_fp8_lds_u64(n)
}

/// # Safety
/// The `launch_mxfp8_dequant` contract. inkling's MXFP8 expert decode; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_mxfp8_dequant(a: Mxfp8Dequant, stream: *mut c_void) -> i32 {
    unsafe { run(&MXFP8_DEQUANT, mxfp8_checked(a.n, a.cols), stream, &a) }
}

/// The NVFP4 decode's geometry: elementwise over `n` carrying the fp8 LUT's LDS (the e4m3 block
/// scale), `cols_packed` a positive multiple of 8 (logical width a multiple of 16), `n` a whole
/// number of `rows·cols_packed·2` expert planes — the kernel's `row / rows` is only the expert
/// index under that divisibility — and `denom` positive and finite (1006). Bad extent or grid →
/// 1001.
fn nvfp4_checked(a: &Nvfp4Dequant) -> Checked {
    if a.n == 0 || any_nonpositive(&[a.rows, a.cols_packed]) {
        return Err(1001);
    }
    let plane = (a.rows as u64) * (a.cols_packed as u64) * 2;
    if a.cols_packed % 8 != 0 || !a.n.is_multiple_of(plane) {
        return Err(1001);
    }
    if not_positive_finite(a.denom) {
        return Err(1006);
    }
    elementwise_fp8_lds_u64(a.n)
}

/// # Safety
/// The `launch_nvfp4_dequant` contract. inkling's NVFP4 expert decode; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_nvfp4_dequant(a: Nvfp4Dequant, stream: *mut c_void) -> i32 {
    unsafe { run(&NVFP4_DEQUANT, nvfp4_checked(&a), stream, &a) }
}

/// The sconv geometry: elementwise over `channels` (no LDS — a short per-channel dot, no fp8 LUT),
/// `channels > 0` and `k > 0` (1001), `divisor` positive finite (1006).
fn sconv_checked(a: &Sconv) -> Checked {
    if a.channels <= 0 || a.k <= 0 {
        return Err(1001);
    }
    if not_positive_finite(a.divisor) {
        return Err(1006);
    }
    let n = i64::from(a.channels);
    elementwise(n)
}

/// # Safety
/// The `launch_sconv` contract. inkling's static short-conv decode step; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_sconv(a: Sconv, stream: *mut c_void) -> i32 {
    unsafe { run(&SCONV, sconv_checked(&a), stream, &a) }
}

/// The sink-attend geometry: one block of `ROW_BLOCK` threads per query head, and
/// `(n + ROW_BLOCK)` f32 of LDS — the `n` logits plus the max/sum tree. `n`, `d`, `heads`,
/// `kv_heads` > 0 and `heads % kv_heads == 0` (the GQA mapping divides by it), `kv_stride >= n`
/// (a head's window lies inside its stride) → 1001; `scaling` positive finite → 1006; the LDS
/// total within the gfx1151 budget of 65536 bytes → 1001 (the engine chunks a longer context).
fn sink_attend_checked(a: &SinkAttend) -> Checked {
    if any_nonpositive(&[a.n, a.d, a.heads, a.kv_heads]) {
        return Err(1001);
    }
    if a.kv_stride < a.n || a.heads % a.kv_heads != 0 {
        return Err(1001);
    }
    if not_positive_finite(a.scaling) {
        return Err(1006);
    }
    let lds = (a.n as u64 + u64::from(ROW_BLOCK)) * 4;
    if lds > 65536 {
        return Err(1001);
    }
    Ok(Geometry {
        blocks: a.heads as u32,
        threads: ROW_BLOCK,
        lds: LdsBytes(lds as u32),
    })
}

/// # Safety
/// The `launch_sink_attend` contract. inkling's single-query sink attention; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_sink_attend(a: SinkAttend, stream: *mut c_void) -> i32 {
    unsafe { run(&SINK_ATTEND, sink_attend_checked(&a), stream, &a) }
}

/// # Safety
/// The `launch_message_gate` contract. inkling's loop MessageGate combine; no C twin (oracle-validated).
/// `elementwise` refuses `n <= 0` (1001); the four operand vectors carry no further guard.
pub(crate) unsafe fn rivoli_message_gate(a: MessageGate, stream: *mut c_void) -> i32 {
    unsafe { run(&MESSAGE_GATE, elementwise(i64::from(a.n)), stream, &a) }
}

/// `kv_scatter`'s refusals: extents (1001), then the position inside the context and the source a
/// whole number of heads (1001). A pure data movement — no divisor to guard.
fn kv_scatter_checked(a: &KvScatter) -> Checked {
    if a.n <= 0 || a.head_dim <= 0 {
        return Err(1001);
    }
    if a.ctx <= 0 || a.pos < 0 {
        return Err(1001);
    }
    if a.pos >= a.ctx {
        return Err(1001);
    }
    if a.n % a.head_dim != 0 {
        return Err(1001);
    }
    let n = i64::from(a.n);
    elementwise(n)
}

/// # Safety
/// The `launch_kv_scatter` contract. inkling's KV-cache append scatter; no C twin (oracle-validated).
pub(crate) unsafe fn rivoli_kv_scatter(a: KvScatter, stream: *mut c_void) -> i32 {
    unsafe { run(&KV_SCATTER, kv_scatter_checked(&a), stream, &a) }
}

/// `rel_bias_gather`'s refusals: a zero head/key/extent count, a negative position or window
/// start, and a `heads·n` past the elementwise grid's `i32` (all 1001). A pure gather + mask.
fn rel_bias_gather_checked(a: &RelBiasGather) -> Checked {
    if any_nonpositive(&[a.heads, a.n, a.rel_extent]) {
        return Err(1001);
    }
    if a.pos < 0 || a.start < 0 {
        return Err(1001);
    }
    let total = i64::from(a.heads) * i64::from(a.n);
    if total > i64::from(i32::MAX) {
        return Err(1001);
    }
    elementwise(total)
}

/// # Safety
/// inkling's rel-bias distance gather; no C twin (oracle-validated). `launch_rel_bias_gather` contract.
pub(crate) unsafe fn rivoli_rel_bias_gather(a: RelBiasGather, stream: *mut c_void) -> i32 {
    unsafe { run(&REL_BIAS_GATHER, rel_bias_gather_checked(&a), stream, &a) }
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
