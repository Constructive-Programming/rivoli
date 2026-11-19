//! `attn.hip`'s launchers: MLA's split-KV flash attend (attend + combine), V4's gather attend,
//! the GQA pair and K3's dense MHA. Guards, order and codes are the tagged C's; what is new is
//! the geometry tie (every block size is the `rivoli_abi::geom` constant the kernel strides
//! by) and the LDS the C declared `static` (`red`, `w`), which the Rust kernels carve from the
//! dynamic allocation and the launcher therefore requests.

use super::{Checked2, Rows, run_2d};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::{
    GATHER_ATTN_ACC, GATHER_ATTN_THREADS, GQA_ACC, GQA_THREADS, GQA_TILE, MHA_THREADS,
    MLA_ACC_REGS, MLA_ATTEND_THREADS, MLA_COMBINE_LDS_BYTES, MLA_COMBINE_THREADS, MLA_HB,
    MLA_MAX_SPLITS, MLA_TILE, WAVE,
};
use rivoli_abi::kernarg::{
    GatherAttn, Gqa, GqaBlock, GqaRing, MhaAttend, MlaAttend, MlaCombine, MlaIo,
};
use std::ffi::c_void;
use std::ptr::null_mut;

static MLA_LATENT_ATTEND: Kernel<MlaAttend> = Kernel::new(c"mla_latent_attend");
static MLA_ATTEND_COMBINE: Kernel<MlaCombine> = Kernel::new(c"mla_attend_combine");
static GATHER_ATTN_SHARED_KV: Kernel<GatherAttn> = Kernel::new(c"gather_attn_shared_kv");
static GQA_ATTEND: Kernel<GqaRing> = Kernel::new(c"gqa_attend");
static GQA_BLOCK_ATTEND: Kernel<GqaBlock> = Kernel::new(c"gqa_block_attend");
static MHA_ATTEND: Kernel<MhaAttend> = Kernel::new(c"mha_attend");

/// gfx1151's LDS per workgroup — the C guards' `65536`.
const LDS_MAX: i64 = 65536;
const F32: i64 = 4;

// The split plan's two host-only knobs (the kernel is handed only the plan): ~2× the 40 CUs,
// and the fewest tiles a split must own before split + combine pays (the C's measurements).
const MLA_TARGET_BLOCKS: i32 = 80;
const MLA_MIN_TILES_PER_SPLIT: i32 = 4;

/// `mha_attend`'s score-staging ceiling: `kv` past it is refused (1004), not truncated.
const ATTEND_MAX_KV: i32 = 8192;

/// How the `nr` rows are cut: `tps` tile steps per split, `n_splits` splits. A pure function
/// of `(h, nr, scratch present)` — same context, same cut, same summation order (greedy
/// decode needs it).
struct SplitPlan {
    n_splits: i32,
    tps: i32,
}

/// The C's `mla_plan_splits`.
fn plan_splits(h: i32, nr: i32, have_scratch: bool) -> SplitPlan {
    let tile = MLA_TILE as i32;
    let ntiles = (nr + tile - 1) / tile;
    let hblocks = (h + MLA_HB as i32 - 1) / MLA_HB as i32;
    let by_work = ntiles / MLA_MIN_TILES_PER_SPLIT;
    let by_grid = (MLA_TARGET_BLOCKS + hblocks - 1) / hblocks;
    let mut n = by_work.min(by_grid).min(MLA_MAX_SPLITS as i32);
    if n < 1 || !have_scratch {
        n = 1; // degenerate = exactly the unsplit path
    }
    let tps = (ntiles + n - 1) / n;
    SplitPlan {
        n_splits: (ntiles + tps - 1) / tps,
        tps,
    }
}

/// The attend's refusals in the C's order — extents (1001), the tile's LDS (1002), a `kvl`
/// past the accumulator registers or not a multiple of 128 (1004, the C argues both) — or
/// the tile's LDS bytes.
fn mla_checked(io: &MlaIo) -> Result<u32, i32> {
    let extents = [io.h, io.nr, io.kvl, io.rope, io.n_blocks]
        .iter()
        .all(|&x| x > 0);
    // i64: the extents may be negative here, and the C's `size_t` sum only ever met positives.
    let lds = i64::from(MLA_TILE) * (i64::from(io.kvl) + i64::from(io.rope)) * F32;
    let registers = io.kvl <= (MLA_ACC_REGS * WAVE) as i32 && io.kvl % 128 == 0;
    match () {
        () if !extents => Err(1001),
        () if lds > LDS_MAX => Err(1002),
        () if !registers => Err(1004),
        () => Ok(lds as u32),
    }
}

/// # Safety
/// The `launch_attend` contract. Null stream, as the C.
///
/// Two kernels, one launcher: the attend (`⌈h/MLA_HB⌉ × n_splits` blocks) and, when the plan
/// splits, the combine (`h` blocks) — `kernel_coverage.rs`'s `LAUNCHED_BY` rows both under
/// this launcher, so each launch below names its kernel's static.
pub(crate) unsafe fn rivoli_mla_attend(io: MlaIo, partial: *mut f32) -> i32 {
    let lds = match mla_checked(&io) {
        Ok(lds) => lds,
        Err(code) => return code,
    };
    let plan = plan_splits(io.h, io.nr, !partial.is_null());
    let split = plan.n_splits > 1;
    let a = MlaAttend {
        io,
        tps: plan.tps,
        partial: if split { partial } else { null_mut() },
    };
    let geom = Geometry {
        blocks: (io.h as u32).div_ceil(MLA_HB),
        threads: MLA_ATTEND_THREADS,
        lds: LdsBytes(lds),
    };
    let grid = (geom, plan.n_splits as u32);
    // SAFETY: the caller's contract; `partial` holds the scratch whenever `split`.
    let rc = unsafe { MLA_LATENT_ATTEND.launch_2d(grid, null_mut(), &a) };
    if rc != 0 || !split {
        return rc; // the attend wrote `clat` itself
    }
    let c = MlaCombine {
        partial,
        clat: io.clat,
        h: io.h,
        kvl: io.kvl,
        n_splits: plan.n_splits,
    };
    let geom = Geometry {
        blocks: io.h as u32,
        threads: MLA_COMBINE_THREADS,
        lds: LdsBytes(MLA_COMBINE_LDS_BYTES),
    };
    // SAFETY: as above.
    unsafe { MLA_ATTEND_COMBINE.launch(geom, null_mut(), &c) }
}

/// The gather attend's refusals in the C's order: extents (1001), `d` past the per-thread
/// accumulator (1002), and LDS (1006) — counted as the C counts it, with its static
/// `red[GATHER_ATTN_THREADS]` and `bc[2]` — then a `(h, m)` grid of `GATHER_ATTN_THREADS`.
/// The Rust kernel carves `qs | lg | red` from the dynamic allocation (no `bc`).
fn gather_checked(Rows(m): Rows, a: &GatherAttn) -> Checked2 {
    let extents = [m, a.h, a.d, a.topk].iter().all(|&x| x > 0);
    let lds = (i64::from(a.d) + i64::from(a.topk)) * F32;
    let statics = F32 * (i64::from(GATHER_ATTN_THREADS) + 2);
    let geom = Geometry {
        blocks: a.h as u32,
        threads: GATHER_ATTN_THREADS,
        lds: LdsBytes((lds + F32 * i64::from(GATHER_ATTN_THREADS)) as u32),
    };
    match () {
        () if !extents => Err(1001),
        () if a.d > (GATHER_ATTN_THREADS * GATHER_ATTN_ACC) as i32 => Err(1002),
        () if lds + statics > LDS_MAX => Err(1006),
        () => Ok((geom, m as u32)),
    }
}

/// # Safety
/// The `launch_gather_attn_shared_kv` contract.
pub(crate) unsafe fn rivoli_gather_attn_shared_kv(
    m: Rows,
    a: GatherAttn,
    stream: *mut c_void,
) -> i32 {
    unsafe { run_2d(&GATHER_ATTN_SHARED_KV, gather_checked(m, &a), stream, &a) }
}

/// Both GQA attends' refusals in the C's order — extents and the launcher's own argument
/// checks (`bad_args`, 1001), an inexact group (1003), `d` past the lane registers (1002), the
/// launcher's geometry check (`bad_geometry`, 1005) — then the `(hq, tq)` grid of waves with
/// a K and a V tile of `GQA_TILE` rows. No LDS guard: the `d` cap bounds it under 64 KiB
/// (`rivoli_abi::geom`'s assert).
fn gqa_checked(g: &Gqa, Rows(tq): Rows, [bad_args, bad_geometry]: [bool; 2]) -> Checked2 {
    let extents = [g.hq, g.hkv, g.d, tq].iter().all(|&x| x > 0);
    // i64, and narrowed only once accepted: `d` may be anything here.
    let lds = 2 * i64::from(GQA_TILE) * i64::from(g.d) * F32;
    let geom = Geometry {
        blocks: g.hq as u32,
        threads: GQA_THREADS,
        lds: LdsBytes(lds as u32),
    };
    match () {
        () if !extents || bad_args => Err(1001),
        () if g.hq % g.hkv != 0 => Err(1003),
        () if g.d > (GQA_THREADS * GQA_ACC) as i32 => Err(1002),
        () if bad_geometry => Err(1005),
        () => Ok((geom, tq as u32)),
    }
}

/// `gqa_attend`'s own checks: a negative `start_pos`, `win` or `ring_cap` (1001); a ring on a
/// global layer, or a ring shorter than the batch's union of windows, `win + tq − 1` (1005).
fn gqa_ring_checked(tq: Rows, a: &GqaRing) -> Checked2 {
    let g = &a.gqa;
    let span = i64::from(g.win) + i64::from(tq.0) - 1;
    let short = a.ring_cap != 0 && i64::from(a.ring_cap) < span;
    let ring_on_global = g.win == 0 && a.ring_cap != 0;
    let bad_args = a.start_pos < 0 || g.win < 0 || a.ring_cap < 0;
    gqa_checked(g, tq, [bad_args, ring_on_global || short])
}

/// # Safety
/// The `launch_gqa_attend` contract.
pub(crate) unsafe fn rivoli_gqa_attend(tq: Rows, a: GqaRing, stream: *mut c_void) -> i32 {
    unsafe { run_2d(&GQA_ATTEND, gqa_ring_checked(tq, &a), stream, &a) }
}

/// `gqa_block_attend`'s own checks: `kv_len ≤ 0`, `q_offset < 0`, and `win ≤ 0` — a
/// bidirectional window of 0 is no operator this model has (1001); the block inside its own
/// K/V buffer — `kv_len ≥ tq`, and row `tq − 1`'s lower edge at most `kv_len − 1` (1005).
fn gqa_block_checked(tq: Rows, a: &GqaBlock) -> Checked2 {
    let g = &a.gqa;
    let last_lo = i64::from(a.q_offset) + i64::from(tq.0) - 1 - i64::from(g.win);
    let outside = a.kv_len < tq.0 || last_lo > i64::from(a.kv_len) - 1;
    let bad_args = a.kv_len <= 0 || a.q_offset < 0 || g.win <= 0;
    gqa_checked(g, tq, [bad_args, outside])
}

/// # Safety
/// The `launch_gqa_block_attend` contract.
pub(crate) unsafe fn rivoli_gqa_block_attend(tq: Rows, a: GqaBlock, stream: *mut c_void) -> i32 {
    unsafe { run_2d(&GQA_BLOCK_ATTEND, gqa_block_checked(tq, &a), stream, &a) }
}

/// # Safety
/// The `launch_mha_attend` contract. Null stream, as the C.
///
/// Refusals: extents (1001), then `kv` outside `1..=ATTEND_MAX_KV` (1004). LDS is the score
/// row plus the C's static `red[MHA_THREADS]`.
pub(crate) unsafe fn rivoli_mha_attend(Rows(heads): Rows, a: MhaAttend) -> i32 {
    let extents = [heads, a.d, a.dv].iter().all(|&x| x > 0);
    match () {
        () if !extents => 1001,
        () if !(1..=ATTEND_MAX_KV).contains(&a.kv) => 1004,
        () => {
            let geom = Geometry {
                blocks: heads as u32,
                threads: MHA_THREADS,
                lds: LdsBytes((a.kv as u32 + MHA_THREADS) * F32 as u32),
            };
            // SAFETY: the caller's contract.
            unsafe { MHA_ATTEND.launch(geom, null_mut(), &a) }
        }
    }
}
