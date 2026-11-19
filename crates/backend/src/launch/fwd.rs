//! `fwd.hip`'s launchers: the forward pass's glue kernels.

use super::{Checked, elementwise, not_positive_finite, row_gather, run};
use crate::hip::HashSpan;
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{
    APPEND_KV_GROUP, ARGMAX_BLOCK, ARGMAX_LDS_BYTES, HASH_BLOCK, HASH_LDS_BYTES,
    append_kv_lds_bytes,
};
use rivoli_abi::kernarg::{AppendKv, Argmax, EmbedI8, GatherRope, InPlaceBinary, Softcap};
// The two divergence instruments.
use rivoli_abi::kernarg::{FlagNonfinite, HashRows};
use std::ffi::c_void;
use std::ptr::null_mut;

static EMBED_I8_ROW: Kernel<EmbedI8> = Kernel::new(c"embed_i8_row");
static APPEND_KV: Kernel<AppendKv> = Kernel::new(c"append_kv");
static GATHER_ROPE: Kernel<GatherRope> = Kernel::new(c"gather_rope");
static VADD: Kernel<InPlaceBinary> = Kernel::new(c"vadd");
static SIGMOID_GATE: Kernel<InPlaceBinary> = Kernel::new(c"sigmoid_gate");
static LOGIT_SOFTCAP: Kernel<Softcap> = Kernel::new(c"logit_softcap");
static ARGMAX: Kernel<Argmax> = Kernel::new(c"argmax");
static FLAG_NONFINITE: Kernel<FlagNonfinite> = Kernel::new(c"flag_nonfinite");
static HASH_ROWS: Kernel<HashRows> = Kernel::new(c"hash_rows");

/// The largest block `append_kv` launches (`kvl` threads, one per latent element) — HIP's
/// ceiling, and the C's bound. Host-only: the kernel reads `threadIdx.x`, never a width.
const APPEND_KV_MAX: i32 = 1024;

/// `hash_rows`'s grid cap: the kernel is grid-stride, so a bounded grid bounds the atomics
/// (one per block) whatever `n` is; ~25x the 40 CUs (the C's note). Host-only.
const HASH_MAX_BLOCKS: i64 = 1024;

/// # Safety
/// The `launch_embed_i8_row` contract.
pub(crate) unsafe fn rivoli_embed_i8_row(a: EmbedI8) -> i32 {
    unsafe { run(&EMBED_I8_ROW, row_gather(a.token, a.hidden), null_mut(), &a) }
}

/// `append_kv`'s refusals: extents (1001), then a `kvl` the one-block per-group reduction
/// cannot take (a multiple of `APPEND_KV_GROUP` in `[GROUP, 1024]`) or `ropn > kvl` (1002).
fn append_kv_checked(a: &AppendKv) -> Checked {
    let extents = [a.kvl, a.ropn, a.n_blocks].iter().all(|&v| v > 0);
    if a.pos < 0 || !extents {
        return Err(1001);
    }
    let g = APPEND_KV_GROUP as i32;
    let one_block = (g..=APPEND_KV_MAX).contains(&a.kvl) && a.kvl % g == 0;
    if !one_block || a.ropn > a.kvl {
        return Err(1002);
    }
    Ok(Geometry {
        blocks: 1,
        threads: a.kvl as u32,
        lds: LdsBytes(append_kv_lds_bytes(a.kvl as u32)),
    })
}

/// # Safety
/// The `launch_append_kv` contract.
pub(crate) unsafe fn rivoli_append_kv(a: AppendKv) -> i32 {
    unsafe { run(&APPEND_KV, append_kv_checked(&a), null_mut(), &a) }
}

/// # Safety
/// The `launch_gather_rope` contract.
pub(crate) unsafe fn rivoli_gather_rope(a: GatherRope) -> i32 {
    let extents = [a.h, a.qh, a.ropn].iter().all(|&v| v > 0);
    let checked = if !extents || a.nope < 0 {
        Err(1001)
    } else {
        // The C's `int` product, which the kernel bounds by too.
        elementwise(i64::from(a.h.wrapping_mul(a.ropn)))
    };
    unsafe { run(&GATHER_ROPE, checked, null_mut(), &a) }
}

/// # Safety
/// The `launch_vadd` contract.
pub(crate) unsafe fn rivoli_vadd(a: InPlaceBinary) -> i32 {
    unsafe { run(&VADD, elementwise(i64::from(a.n)), null_mut(), &a) }
}

/// # Safety
/// The `launch_sigmoid_gate` contract.
pub(crate) unsafe fn rivoli_sigmoid_gate(a: InPlaceBinary, stream: *mut c_void) -> i32 {
    unsafe { run(&SIGMOID_GATE, elementwise(i64::from(a.n)), stream, &a) }
}

/// The softcap's refusals: `n` (1001), a non-positive or non-finite `cap`/`mult` (1001 —
/// a config field not read), then `mult >= cap` (1002 — the silent transposed pair; the C's
/// note has the argument).
fn softcap_checked(a: &Softcap) -> Checked {
    let geom = elementwise(i64::from(a.n))?;
    if not_positive_finite(a.cap) || not_positive_finite(a.mult) {
        return Err(1001);
    }
    if a.mult >= a.cap {
        return Err(1002);
    }
    Ok(geom)
}

/// # Safety
/// The `launch_logit_softcap` contract.
pub(crate) unsafe fn rivoli_logit_softcap(a: Softcap, stream: *mut c_void) -> i32 {
    unsafe { run(&LOGIT_SOFTCAP, softcap_checked(&a), stream, &a) }
}

/// # Safety
/// The `launch_argmax` contract. One block.
pub(crate) unsafe fn rivoli_argmax(a: Argmax) -> i32 {
    let checked = if a.n <= 0 {
        Err(1001)
    } else {
        Ok(Geometry {
            blocks: 1,
            threads: ARGMAX_BLOCK,
            lds: LdsBytes(ARGMAX_LDS_BYTES),
        })
    };
    unsafe { run(&ARGMAX, checked, null_mut(), &a) }
}

/// # Safety
/// The `launch_flag_nonfinite` contract.
pub(crate) unsafe fn rivoli_flag_nonfinite(a: FlagNonfinite) -> i32 {
    unsafe { run(&FLAG_NONFINITE, elementwise(i64::from(a.n)), null_mut(), &a) }
}

/// The fold's walk and its grid: `n` and `stride` positive (1001); the grid sized on SAMPLES
/// and capped at [`HASH_MAX_BLOCKS`], its `blocks · HASH_BLOCK` handed to the kernel as the
/// grid stride.
fn hash_checked(n: i32, stride: i32) -> Result<(Geometry, i64), i32> {
    if n <= 0 || stride <= 0 {
        return Err(1001);
    }
    let samples = (i64::from(n) + i64::from(stride) - 1) / i64::from(stride);
    let blocks = (samples + i64::from(HASH_BLOCK) - 1) / i64::from(HASH_BLOCK);
    let blocks = blocks.min(HASH_MAX_BLOCKS);
    let geom = Geometry {
        blocks: blocks as u32,
        threads: HASH_BLOCK,
        lds: LdsBytes(HASH_LDS_BYTES),
    };
    Ok((geom, blocks * i64::from(HASH_BLOCK)))
}

/// # Safety
/// The `launch_hash_rows` contract. Refuses (1002) a walk the `i32` ABI cannot carry: `n` or
/// `stride` past `i32`, or `n + stride - 1` (the kernel's sample count) overflowing it — the
/// one refusal the C, which took `int`s, did not have.
pub(crate) unsafe fn rivoli_hash_rows(
    x: *const f32,
    HashSpan { n, stride, i_base }: HashSpan,
    out: *mut u64,
    stream: *mut c_void,
) -> i32 {
    let walk = (i32::try_from(n), i32::try_from(stride));
    let (Ok(n), Ok(stride)) = walk else {
        return 1002;
    };
    if n.checked_add(stride.saturating_sub(1)).is_none() {
        return 1002;
    }
    let (geom, grid_stride) = match hash_checked(n, stride) {
        Ok(g) => g,
        Err(code) => return code,
    };
    let a = HashRows {
        x,
        out,
        i_base,
        grid_stride,
        n,
        stride,
    };
    unsafe { run(&HASH_ROWS, Ok(geom), stream, &a) }
}
