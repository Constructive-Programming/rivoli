//! MLA's flash attend over the fp8 latent cache, and the combine that merges its split-KV
//! partials.
//!
//! hipcc fuses seven sites here, in five groups (its ISA, `hipcc -O3 --cuda-device-only -S` on the tag):
//! - both score dots, `part += qa·L` and `part += qr·R` — `v_fmac`, `fma(qa, L, part)`;
//! - `p = expf(s − m_new)` with `s = scale·total`: the subtraction absorbs the product,
//!   `v_fma_f32 v, scale, total, −m_new`, while `m_new = fmaxf(m, s)` takes the rounded `s`;
//! - the accumulator, `acc·corr + p·L` — the LEFT product: `v_mul` of `p·L`, then `v_fmac`;
//! - `l·corr + p` — `fma(l, corr, p)`;
//! - in the combine, `l_g += p₁·w` and `sum += partial·w`.

use super::{Running, wave_total};
use crate::gpu::{bid, bid_y, fma, for_i32, lds, tid};
use crate::prelude::formats::{bf16f, e4m3f};
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::sync_threads;
use core::intrinsics::{exp, unchecked_div};
use rivoli_abi::geom::{
    MLA_ACC_REGS, MLA_ATTEND_THREADS, MLA_COMBINE_THREADS, MLA_HB, MLA_MAX_SPLITS, MLA_TILE, WAVE,
};
use rivoli_abi::kernarg::{MlaAttend, MlaCombine, MlaIo};

const ACC: usize = MLA_ACC_REGS as usize;
const TILE: i32 = MLA_TILE as i32;
const MAX_SPLITS: usize = MLA_MAX_SPLITS as usize;

/// One staged token tile in LDS: `kvl` widened latent floats then `rope` key floats per token.
#[derive(Clone, Copy)]
struct Tile {
    lt: *mut f32,
    rt: *mut f32,
}

/// The cooperative tile load for rows `t0 .. t0 + tcount` (every thread of the block):
/// latent e4m3 × its per-128 block scale, key bf16 → f32. `rows` null = dense.
#[inline(always)]
unsafe fn load_tile(io: &MlaIo, tile: Tile, t0: i32, tcount: i32) {
    let (kvl, rope) = (io.kvl, io.rope);
    let token = |tt: i32| -> usize {
        let r = (t0 + tt) as usize;
        if io.rows.is_null() {
            r
        } else {
            *io.rows.add(r) as usize
        }
    };
    for_i32(tid(), tcount * kvl, MLA_ATTEND_THREADS, |idx| {
        // SAFETY: `kvl > 0`, `rope > 0` (1001).
        let tt = unchecked_div(idx as i32, kvl);
        let i = idx as i32 - tt * kvl;
        let t = token(tt);
        let l8 = *io.lc8.add(t * kvl as usize + i as usize);
        let s = *io.lscale.add(t * io.n_blocks as usize + (i / 128) as usize);
        *tile.lt.add(idx) = e4m3f(l8) * s;
    });
    for_i32(tid(), tcount * rope, MLA_ATTEND_THREADS, |idx| {
        // SAFETY: as above.
        let tt = unchecked_div(idx as i32, rope);
        let d = idx as i32 - tt * rope;
        *tile.rt.add(idx) = bf16f(*io.rc.add(token(tt) * rope as usize + d as usize));
    });
}

/// One head's queries and online-softmax state: the lane's `ACC` owned columns
/// (`i ≡ lane mod WAVE`), the running max and the running sum, replicated across the head's
/// lanes.
struct Online {
    qa: *const f32,
    qr: *const f32,
    r: Running<ACC>,
}

/// `f(k, i)` for each of this lane's live accumulator registers `k`, column `i`. The bound is
/// the wave-uniform `nacc = kvl / WAVE`, never `i < kvl` (the C's INVARIANT note: a
/// lane-divergent bound costs an exec-mask save/restore per unrolled step).
#[inline(always)]
fn each_acc(kvl: i32, mut f: impl FnMut(usize, usize)) {
    let (nacc, lane) = (kvl / WAVE as i32, (tid() & (WAVE - 1)) as usize);
    for k in 0..ACC {
        if (k as i32) < nacc {
            f(k, lane + k * WAVE as usize);
        }
    }
}

/// Fold token `tt` of the staged tile into `st`: the subgroup's score, then the rescale.
#[inline(always)]
unsafe fn fold_token(io: &MlaIo, tile: Tile, tt: i32, st: &mut Online) {
    let (kvl, lane) = (io.kvl, tid() & (WAVE - 1));
    let lrow = tile.lt.add((tt * kvl) as usize);
    let rrow = tile.rt.add((tt * io.rope) as usize);
    let mut part = 0.0f32;
    for_i32(lane, kvl, WAVE, |i| {
        part = fma(*st.qa.add(i), *lrow.add(i), part)
    });
    for_i32(lane, io.rope, WAVE, |d| {
        part = fma(*st.qr.add(d), *rrow.add(d), part)
    });
    let total = wave_total(part);
    let m_new = maxnum(st.r.m, io.scale * total);
    let corr = exp(st.r.m - m_new);
    let p = exp(fma(io.scale, total, -m_new));
    each_acc(kvl, |k, i| {
        st.r.acc[k] = fma(st.r.acc[k], corr, p * *lrow.add(i))
    });
    st.r.l = fma(st.r.l, corr, p);
    st.r.m = m_new;
}

/// The single-split result: the normalised row of `clat` (`l > 0 ? 1/l : 0`).
#[inline(always)]
unsafe fn write_clat(io: &MlaIo, head: usize, st: &Online) {
    let out = io.clat.add(head * io.kvl as usize);
    let inv = if st.r.l > 0.0 { 1.0 / st.r.l } else { 0.0 };
    each_acc(io.kvl, |k, i| *out.add(i) = st.r.acc[k] * inv);
}

/// A split's result: the UNNORMALISED accumulator and its `(m, l)` — normalising here would
/// lose what the combine needs. An empty split stores `−inf, 0, 0`, which the combine's
/// `exp(−inf − m_g) = 0` weight annihilates.
#[inline(always)]
unsafe fn write_partial(a: &MlaAttend, head: usize, st: &Online) {
    let (h, kvl) = (a.io.h as usize, a.io.kvl as usize);
    let at = bid_y() as usize * h + head;
    let out = a.partial.add(at * kvl);
    each_acc(a.io.kvl, |k, i| *out.add(i) = st.r.acc[k]);
    if tid() & (WAVE - 1) == 0 {
        let pml = a.partial.add(MAX_SPLITS * h * kvl + at * 2);
        *pml = st.r.m;
        *pml.add(1) = st.r.l;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mla_latent_attend(a: MlaAttend) {
    let io = a.io;
    let lt = lds::<f32>();
    let tile = Tile {
        lt,
        rt: lt.add((TILE * io.kvl) as usize),
    };
    let head = (bid() * MLA_HB + tid() / WAVE) as i32;
    let active = head < io.h;
    let qh = if active { head as usize } else { 0 };
    let mut st = Online {
        qa: io.qabs.add(qh * io.kvl as usize),
        qr: io.qrope.add(qh * io.rope as usize),
        r: Running::START,
    };
    // This split's rows — block-uniform, so every barrier is reached by every thread.
    let r_begin = bid_y() as i32 * a.tps * TILE;
    let r_end = io.nr.min(r_begin + a.tps * TILE);
    let mut t0 = r_begin;
    while t0 < r_end {
        let tcount = TILE.min(r_end - t0);
        load_tile(&io, tile, t0, tcount);
        sync_threads();
        if active {
            for tt in 0..tcount {
                fold_token(&io, tile, tt, &mut st);
            }
        }
        sync_threads(); // the tile is consumed before the next load overwrites it
        t0 += TILE;
    }
    match (active, a.partial.is_null()) {
        (false, _) => {}
        (true, true) => write_clat(&io, head as usize, &st),
        (true, false) => write_partial(&a, head as usize, &st),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mla_attend_combine(a: MlaCombine) {
    let w = lds::<f32>();
    let inv_l = w.add(MAX_SPLITS);
    let head = bid() as i32;
    if head >= a.h {
        return;
    }
    let (h, kvl, head) = (a.h as usize, a.kvl as usize, head as usize);
    let ml = a.partial.add(MAX_SPLITS * h * kvl);
    let splits = 0..a.n_splits as usize;
    if tid() == 0 {
        let m_g = splits.clone().fold(f32::NEG_INFINITY, |m, s| {
            maxnum(m, *ml.add((s * h + head) * 2))
        });
        // Every split empty (or every score −inf): `exp(m_s − m_g)` would be NaN, so the
        // weights are 0 and the output is the single-block kernel's `l > 0 ? 1/l : 0` zero.
        let mut l_g = 0.0f32;
        for s in splits.clone() {
            let p = ml.add((s * h + head) * 2);
            let ws = if m_g.is_finite() { exp(*p - m_g) } else { 0.0 };
            *w.add(s) = ws;
            l_g = fma(*p.add(1), ws, l_g);
        }
        *inv_l = if l_g > 0.0 { 1.0 / l_g } else { 0.0 };
    }
    sync_threads();
    for_i32(tid(), a.kvl, MLA_COMBINE_THREADS, |i| {
        let sum = splits.clone().fold(0.0f32, |sum, s| {
            fma(*a.partial.add((s * h + head) * kvl + i), *w.add(s), sum)
        });
        *a.clat.add(head * kvl + i) = sum * *inv_l;
    });
}
