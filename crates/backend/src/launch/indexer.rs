//! `indexer.hip`'s launchers: the DSA/MISA indexer chain, the device top-k, Qwen's QSA block-key
//! stage, and the fp32 block scorer (its bf16 twin is `blockindex.rs`'s, over the same
//! `ScoreBlocks` operand set). Three kernels take a run-time block size; their launchers compute it
//! exactly as the C did and hand it over in `Blocked`, never through the dispatch packet.

use super::{Checked, Rows, any_nonpositive, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{LAYERNORM_BLOCK, LAYERNORM_LDS_BYTES, TOPK_BLOCK, TOPK_LDS_BYTES};
use rivoli_abi::kernarg::{
    Blocked, HeadRoute, IndexAppend, IndexScore, LayerNorm, PoolNorm, PoolPush, ScoreBlocks, TopK,
};
use std::ffi::c_void;
use std::ptr::null_mut;

const F32: u32 = size_of::<f32>() as u32;

static LAYERNORM: Kernel<LayerNorm> = Kernel::new(c"layernorm");
static INDEX_APPEND: Kernel<IndexAppend> = Kernel::new(c"index_append");
static INDEX_SCORE: Kernel<Blocked<IndexScore>> = Kernel::new(c"index_score");
static INDEX_POOL_PUSH: Kernel<PoolPush> = Kernel::new(c"index_pool_push");
static INDEX_HEAD_ROUTE: Kernel<Blocked<HeadRoute>> = Kernel::new(c"index_head_route");
static INDEX_TOPK: Kernel<TopK> = Kernel::new(c"index_topk");
static INDEX_POOL_NORM_F32: Kernel<Blocked<PoolNorm>> = Kernel::new(c"index_pool_norm_f32");
static INDEX_SCORE_BLOCKS_F32: Kernel<ScoreBlocks> = Kernel::new(c"index_score_blocks_f32");

/// `index_head_route`'s block cap — the C's `red[256]`, and the LDS this launcher allocates for it.
const HEAD_ROUTE_MAX_BLOCK: u32 = 256;
/// `index_pool_norm_f32`'s block cap: the block is `next_pow2(hd)` (1002 past it).
const POOL_NORM_MAX_BLOCK: i32 = 1024;
/// `index_score`'s refusals past these (1002): the block, and the dynamic LDS of `hd + block` f32.
const SCORE_MAX_BLOCK: u32 = 1024;
const SCORE_MAX_LDS: u64 = 65536;

/// `reduce.hpp::next_pow2` for the positive extents every caller has already guarded. (Past
/// 2^30 the C's `int` doubling overflows — undefined there; here it saturates into a refusal.)
fn next_pow2(n: i32) -> u32 {
    (n as u32).next_power_of_two()
}

/// `n` blocks of `threads` with `lds` bytes of dynamic LDS.
fn grid(Rows(n): Rows, threads: u32, lds: u32) -> Geometry {
    Geometry {
        blocks: n as u32,
        threads,
        lds: LdsBytes(lds),
    }
}

/// Launch `k` over `a` with its run-time block, the C's geometry (`threads` = that block).
///
/// # Safety
/// As [`run`].
unsafe fn run_blocked<A: Copy>(
    k: &Kernel<Blocked<A>>,
    checked: Checked,
    stream: *mut c_void,
    a: A,
) -> i32 {
    let block = checked.map_or(0, |g| g.threads);
    unsafe { run(k, checked, stream, &Blocked { a, block }) }
}

/// The two one-row writers' shared refusal: a negative position or a non-positive width (1001),
/// then one thread per element of the row.
fn row_write(pos: i32, hd: i32) -> Checked {
    if pos < 0 {
        Err(1001)
    } else {
        elementwise(i64::from(hd))
    }
}

/// # Safety
/// The `launch_layernorm` contract. Null stream, as the C.
pub(crate) unsafe fn rivoli_layernorm(a: LayerNorm) -> i32 {
    let checked = if a.row.n <= 0 {
        Err(1001)
    } else {
        Ok(grid(Rows(1), LAYERNORM_BLOCK, LAYERNORM_LDS_BYTES))
    };
    unsafe { run(&LAYERNORM, checked, null_mut(), &a) }
}

/// # Safety
/// The `launch_index_append` contract. Null stream.
pub(crate) unsafe fn rivoli_index_append(a: IndexAppend) -> i32 {
    unsafe { run(&INDEX_APPEND, row_write(a.pos, a.hd), null_mut(), &a) }
}

/// `nh` is the guard's alone: `nact` heads of `nh`. The block is `max(32, next_pow2(nact))` — the
/// tree needs a power of two, and 32 threads to stage the key row.
///
/// # Safety
/// The `launch_index_score` contract. Null stream.
pub(crate) unsafe fn rivoli_index_score(nh: i32, a: IndexScore) -> i32 {
    let checked = if any_nonpositive(&[a.nt, nh, a.nact, a.hd]) || a.nact > nh {
        Err(1001)
    } else {
        let bd = next_pow2(a.nact).max(32);
        let lds = (u64::from(a.hd as u32) + u64::from(bd)) * F32 as u64;
        if bd > SCORE_MAX_BLOCK || lds > SCORE_MAX_LDS {
            Err(1002)
        } else {
            Ok(grid(Rows(a.nt), bd, lds as u32))
        }
    };
    unsafe { run_blocked(&INDEX_SCORE, checked, null_mut(), a) }
}

/// # Safety
/// The `launch_index_pool_push` contract. Null stream.
pub(crate) unsafe fn rivoli_index_pool_push(a: PoolPush) -> i32 {
    unsafe { run(&INDEX_POOL_PUSH, row_write(a.t, a.hd), null_mut(), &a) }
}

/// One block per head of `min(256, next_pow2(hd))` threads.
///
/// # Safety
/// The `launch_index_head_route` contract. Null stream.
pub(crate) unsafe fn rivoli_index_head_route(a: HeadRoute) -> i32 {
    let checked = if any_nonpositive(&[a.m_blocks, a.nh, a.hd]) {
        Err(1001)
    } else {
        let bd = next_pow2(a.hd).min(HEAD_ROUTE_MAX_BLOCK);
        Ok(grid(Rows(a.nh), bd, HEAD_ROUTE_MAX_BLOCK * F32))
    };
    unsafe { run_blocked(&INDEX_HEAD_ROUTE, checked, null_mut(), a) }
}

/// `k` is clamped to `nt` (the host `topk_into`'s `k.min(len)`); `k == 0` is refused where the host
/// returns an empty selection — the C's documented divergence, kept.
///
/// # Safety
/// The `launch_index_topk` contract. Null stream.
pub(crate) unsafe fn rivoli_index_topk(mut a: TopK) -> i32 {
    let checked = if any_nonpositive(&[a.nt, a.k]) {
        Err(1001)
    } else {
        a.k = a.k.min(a.nt);
        Ok(grid(Rows(1), TOPK_BLOCK, TOPK_LDS_BYTES))
    };
    unsafe { run(&INDEX_TOPK, checked, null_mut(), &a) }
}

/// Extents (1001), the block cap `hd <= 1024` (1002), `ratio >= 2` (1004, trap T8), then a
/// non-negative finite `eps` (1006; zero is an exact RMSNorm) — the C's order.
///
/// # Safety
/// The `launch_index_pool_norm_f32` contract.
pub(crate) unsafe fn rivoli_index_pool_norm_f32(
    n_blocks: Rows,
    a: PoolNorm,
    stream: *mut c_void,
) -> i32 {
    let checked = if any_nonpositive(&[n_blocks.0, a.ratio, a.hd]) {
        Err(1001)
    } else if a.hd > POOL_NORM_MAX_BLOCK {
        Err(1002)
    } else if a.ratio < 2 {
        Err(1004)
    } else if !(0.0..f32::INFINITY).contains(&a.eps) {
        Err(1006)
    } else {
        let block = next_pow2(a.hd);
        Ok(grid(n_blocks, block, block * F32))
    };
    unsafe { run_blocked(&INDEX_POOL_NORM_F32, checked, stream, a) }
}

/// # Safety
/// The `launch_index_score_blocks_f32` contract.
pub(crate) unsafe fn rivoli_index_score_blocks_f32(a: ScoreBlocks, stream: *mut c_void) -> i32 {
    // Every extent at once (1001) — its bf16 twin answers `n_comp == 0` with 1005 instead.
    let checked = if any_nonpositive(&[a.s, a.n_comp, a.heads, a.hd]) {
        Err(1001)
    } else {
        elementwise(i64::from(a.s) * i64::from(a.n_comp))
    };
    unsafe { run(&INDEX_SCORE_BLOCKS_F32, checked, stream, &a) }
}
