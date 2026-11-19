//! Grouped-query attention, one wave per (query row, Q head): `gqa_attend` (causal, with an
//! optional sliding-window ring) and `gqa_block_attend` (the DFlash drafter's bidirectional
//! block). The C spells the two bodies twice and they differ only in the row window and the
//! slot a position reads; here they are one body over a [`Window`]. Contraction is off in the
//! C, so nothing here is fused.

use super::{Running, wave_total};
use crate::gpu::{bid, bid_y, lds, tid};
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::sync_threads;
use core::intrinsics::exp;
use rivoli_abi::geom::{GQA_ACC, GQA_THREADS, GQA_TILE};
use rivoli_abi::kernarg::{Gqa, GqaBlock, GqaRing};

const ACC: usize = GQA_ACC as usize;
const TILE: i32 = GQA_TILE as i32;

/// The KV positions one query row attends, `lo..=hi`, and the cache slot holding position
/// `j` (`ring_cap` 0 = the slot is the position).
#[derive(Clone, Copy)]
struct Window {
    lo: i32,
    hi: i32,
    ring_cap: i32,
}

/// This lane's column `c` of a `d`-wide row — `lane + c·GQA_THREADS` — or `None` past the
/// `nacc` registers or past `d` (a `d` that does not tile the wave leaves tail lanes idle).
#[inline(always)]
fn column(c: usize, d: i32) -> Option<usize> {
    let i = tid() as i32 + c as i32 * GQA_THREADS as i32;
    let nacc = (d + GQA_THREADS as i32 - 1) / GQA_THREADS as i32;
    ((c as i32) < nacc && i < d).then_some(i as usize)
}

/// `f(c, i)` for each of this lane's live columns, in register order.
#[inline(always)]
fn each_column(d: i32, mut f: impl FnMut(usize, usize)) {
    for c in 0..ACC {
        if let Some(i) = column(c, d) {
            f(c, i);
        }
    }
}

/// The staged K and V tiles (`GQA_TILE · d` floats each) and the KV head this block reads.
#[derive(Clone, Copy)]
struct Tile {
    k: *mut f32,
    v: *mut f32,
    kvh: usize,
}

/// The cooperative load of positions `j0 .. j0 + cnt`: K and V share the slot, so one index
/// serves both.
#[inline(always)]
unsafe fn stage(g: &Gqa, win: Window, tile: Tile, (j0, cnt): (i32, i32)) {
    let d = g.d;
    let mut idx = tid() as i32;
    while idx < cnt * d {
        let tt = idx / d;
        let j = j0 + tt;
        let slot = if win.ring_cap != 0 {
            (j % win.ring_cap) as usize
        } else {
            j as usize
        };
        let at = (slot * g.hkv as usize + tile.kvh) * d as usize + (idx - tt * d) as usize;
        *tile.k.add(idx as usize) = *g.k.add(at);
        *tile.v.add(idx as usize) = *g.v.add(at);
        idx += GQA_THREADS as i32;
    }
}

/// One query row's lane-owned query columns and online-softmax state, all in registers.
struct Online {
    q: [f32; ACC],
    r: Running<ACC>,
}

/// Fold staged row `tt` into `st`: the wave's score, then the rescale.
#[inline(always)]
unsafe fn fold(g: &Gqa, tile: Tile, tt: i32, st: &mut Online) {
    let d = g.d;
    let at = |i: usize| (tt * d) as usize + i;
    let mut part = 0.0f32;
    each_column(d, |c, i| part += st.q[c] * *tile.k.add(at(i)));
    let s = wave_total(part) * g.scale;
    let mx = maxnum(st.r.m, s);
    let (rescale, w) = (exp(st.r.m - mx), exp(s - mx));
    each_column(d, |c, i| {
        st.r.acc[c] = st.r.acc[c] * rescale + w * *tile.v.add(at(i))
    });
    st.r.l = st.r.l * rescale + w;
    st.r.m = mx;
}

/// The shared body: stage `GQA_TILE` K/V rows at a time, fold each into the online softmax,
/// write `acc / l` — unguarded, so a query that attended nothing poisons its row (the C's
/// argument).
#[inline(always)]
unsafe fn attend(g: Gqa, row: usize, win: Window) {
    let d = g.d;
    let kt = lds::<f32>();
    let head = bid() as i32;
    let tile = Tile {
        k: kt,
        v: kt.add((TILE * d) as usize),
        kvh: (head / (g.hq / g.hkv)) as usize, // a per-head block repeat, not an interleave
    };
    let at = (row * g.hq as usize + head as usize) * d as usize;
    let mut st = Online {
        q: [0.0; ACC],
        r: Running::START,
    };
    each_column(d, |c, i| st.q[c] = *g.q.add(at + i));
    let mut j0 = win.lo;
    while j0 <= win.hi {
        let cnt = TILE.min(win.hi - j0 + 1);
        stage(&g, win, tile, (j0, cnt));
        sync_threads();
        for tt in 0..cnt {
            fold(&g, tile, tt, &mut st);
        }
        sync_threads(); // the tile is consumed before the next load overwrites it
        j0 += TILE;
    }
    let inv = 1.0 / st.r.l;
    each_column(d, |c, i| *g.out.add(at + i) = st.r.acc[c] * inv);
}

/// Causal: `[pos − win + 1, pos]` INCLUSIVE of `pos` (trap 14), the whole prefix at `win` 0.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gqa_attend(a: GqaRing) {
    let row = bid_y() as i32;
    let pos = a.start_pos + row;
    let win = a.gqa.win;
    let lo = if win > 0 { (pos - win + 1).max(0) } else { 0 };
    let window = Window {
        lo,
        hi: pos,
        ring_cap: a.ring_cap,
    };
    attend(a.gqa, row as usize, window);
}

/// Bidirectional: `[pos − win, pos + win] ∩ [0, kv_len)`, inclusive BOTH sides, no ring.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gqa_block_attend(a: GqaBlock) {
    let row = bid_y() as i32;
    let pos = a.q_offset + row;
    let win = a.gqa.win;
    let window = Window {
        lo: 0.max(pos - win),
        hi: (a.kv_len - 1).min(pos + win),
        ring_cap: 0,
    };
    attend(a.gqa, row as usize, window);
}
