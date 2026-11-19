//! The DSA/MISA lightning indexer and Qwen's QSA block-key stage — the twins of
//! `kernels/indexer.hip` at tag `archive/hip-kernels`, whose comments carry each kernel's contract
//! (the MISA pool, the top-k's tie rule and its scan bound, why `index_score_blocks_f32` is not the
//! bf16 scorer with a flag). This file carries what the port decided: the fused sites (read off
//! hipcc's ISA — every `+= a * b` in a dot is a `v_fmac`, `layernorm`'s variance is
//! `fma(-mean, mean, Σx²/n)` and its output `fma((x - mean)·inv, w, b)`, and
//! `index_score_blocks_f32` has none, being `#pragma clang fp contract(off)`), and where blockDim
//! went: a `geom` constant for the fixed blocks, `Blocked::block` for the three the launcher sizes
//! per call.

use crate::gpu::{bid, elementwise_id, fma, for_i32, lds, tid};
use crate::prelude::formats::{bf16f, f2bf16};
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{block_sum_lds, sync_threads, tree};
use core::intrinsics::sqrt;
use core::sync::atomic::{AtomicU32, Ordering::Relaxed};
use rivoli_abi::geom::{ELEMENTWISE_BLOCK, LAYERNORM_BLOCK, MISA_BLOCK, TOPK_BINS, TOPK_BLOCK};
use rivoli_abi::kernarg::{
    Blocked, HeadRoute, IndexAppend, IndexScore, LayerNorm, PoolNorm, PoolPush, ScoreBlocks,
    ScoreBufs, TopK,
};

/// `p[i]` for a C `int` offset `i * w + j`, computed in `usize` as `(size_t)` arithmetic would
/// for the non-negative extents every launcher guarantees.
#[inline(always)]
unsafe fn row<T>(p: *const T, i: i32, w: i32) -> *const T {
    p.add(i as usize * w as usize)
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn layernorm(l: LayerNorm) {
    let a = l.row;
    let t = tid() as usize;
    let sred = lds::<f32>();
    let sqred = sred.add(LAYERNORM_BLOCK as usize);
    let (mut s, mut sq) = (0.0f32, 0.0f32);
    for_i32(tid(), a.n, LAYERNORM_BLOCK, |i| {
        let v = *a.x.add(i);
        s += v;
        sq = fma(v, v, sq);
    });
    *sred.add(t) = s;
    *sqred.add(t) = sq;
    sync_threads();
    tree(t, LAYERNORM_BLOCK as usize, |t, st| {
        *sred.add(t) += *sred.add(st);
        *sqred.add(t) += *sqred.add(st);
    });
    let n = a.n as f32;
    let mean = *sred / n;
    let var = fma(-mean, mean, *sqred / n);
    let inv = 1.0 / sqrt(var + a.eps);
    for_i32(tid(), a.n, LAYERNORM_BLOCK, |i| {
        *a.y.add(i) = fma((*a.x.add(i) - mean) * inv, *a.w.add(i), *l.b.add(i));
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_append(a: IndexAppend) {
    let i = elementwise_id();
    if i < a.hd {
        *row(a.kcache, a.pos, a.hd).cast_mut().add(i as usize) = f2bf16(*a.k.add(i as usize));
    }
}

/// One block per cached token, `b.block` threads (the launcher's `max(32, next_pow2(nact))`):
/// the key row staged in LDS, one head per thread, then a `block`-leaf tree.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_score(b: Blocked<IndexScore>) {
    let (a, bd) = (b.a, b.block);
    let t = bid() as i32;
    if t >= a.nt {
        return;
    }
    let th = tid() as usize;
    let kf = lds::<f32>();
    let part = kf.add(a.hd as usize);
    let krow = row(a.kcache, t, a.hd);
    for_i32(tid(), a.hd, bd, |i| *kf.add(i) = bf16f(*krow.add(i)));
    sync_threads();
    let mut contrib = 0.0f32;
    if (th as i32) < a.nact {
        let h = if a.heads.is_null() {
            th as i32
        } else {
            *a.heads.add(th) as i32
        };
        let qh = row(a.q, h, a.hd);
        let mut dot = 0.0f32;
        for i in 0..a.hd as usize {
            dot = fma(*qh.add(i), *kf.add(i), dot);
        }
        contrib = *a.w.add(h as usize) * a.wscale * maxnum(dot * a.dscale, 0.0);
    }
    *part.add(th) = contrib;
    sync_threads();
    tree(th, bd as usize, |x, s| *part.add(x) += *part.add(s));
    if th == 0 {
        *a.scores.add(t as usize) = *part;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_pool_push(a: PoolPush) {
    let i = elementwise_id();
    if i >= a.hd {
        return;
    }
    let in_block = a.t % MISA_BLOCK as i32;
    let m = row(a.pool, a.t / MISA_BLOCK as i32, a.hd)
        .cast_mut()
        .add(i as usize);
    let k = *a.k.add(i as usize);
    if in_block == 0 {
        *m = k;
    } else {
        *m += (k - *m) / (in_block + 1) as f32;
    }
}

/// One block per head, `b.block` threads (`min(256, next_pow2(hd))`), one tree per pooled block.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_head_route(b: Blocked<HeadRoute>) {
    let (a, bd) = (b.a, b.block);
    let j = bid() as i32;
    if j >= a.nh {
        return;
    }
    let th = tid() as usize;
    let red = lds::<f32>();
    let qj = row(a.q, j, a.hd);
    let wj = *a.w.add(j as usize);
    let mut sum = 0.0f32;
    for blk in 0..a.m_blocks {
        let kb = row(a.pool, blk, a.hd);
        let mut p = 0.0f32;
        for_i32(tid(), a.hd, bd, |i| p = fma(*qj.add(i), *kb.add(i), p));
        *red.add(th) = p;
        sync_threads();
        tree(th, bd as usize, |x, s| *red.add(x) += *red.add(s));
        if th == 0 {
            sum += (wj * maxnum(*red, 0.0)).abs();
        }
        sync_threads();
    }
    if th == 0 {
        *a.e.add(j as usize) = sum / a.m_blocks.max(1) as f32;
    }
}

/// The C's order-preserving key: `-0.0` canonicalised to `+0.0`, then the sign-flip map.
#[inline(always)]
fn topk_key(v: f32) -> u32 {
    let b = if v == 0.0 { 0 } else { v.to_bits() };
    b ^ (((b as i32) >> 31) as u32 | 0x8000_0000)
}

/// Block-wide EXCLUSIVE scan of `v` through `tmp` (`TOPK_BLOCK` u32), the block total in `*total`.
/// Every thread calls it: the barriers are unconditional.
#[inline(always)]
unsafe fn topk_scan(tmp: *mut u32, v: u32, th: usize, total: &mut u32) -> u32 {
    *tmp.add(th) = v;
    sync_threads();
    let mut off = 1;
    while off < TOPK_BLOCK as usize {
        let add = if th >= off { *tmp.add(th - off) } else { 0 };
        sync_threads();
        *tmp.add(th) = (*tmp.add(th)).wrapping_add(add);
        sync_threads();
        off <<= 1;
    }
    let incl = *tmp.add(th);
    *total = *tmp.add(TOPK_BLOCK as usize - 1);
    sync_threads();
    incl.wrapping_sub(v)
}

/// One radix pass's histogram: `hist[(key >> shift) & 0xff]` over every key whose fixed bits
/// (`mask`) equal `pivot`'s. LDS atomics: the counts are order-free, so the bin totals are the C's
/// whatever the arrival order.
#[inline(always)]
unsafe fn topk_hist(a: &TopK, hist: *mut u32, (pivot, mask, shift): (u32, u32, u32)) {
    for_i32(tid(), TOPK_BINS as i32, TOPK_BLOCK, |b| *hist.add(b) = 0);
    sync_threads();
    for_i32(tid(), a.nt, TOPK_BLOCK, |i| {
        let u = topk_key(*a.s.add(i));
        if u & mask == pivot {
            AtomicU32::from_ptr(hist.add(((u >> shift) & 0xff) as usize)).fetch_add(1, Relaxed);
        }
    });
    sync_threads();
}

/// The bin holding the `need`-th largest surviving key, walking high to low, and the count above
/// it. Bin 0 is never tested — the C's fall-through, with `acc < need` on every exit.
#[inline(always)]
unsafe fn topk_bin(hist: *const u32, need: u32) -> (u32, u32) {
    let (mut acc, mut b) = (0u32, TOPK_BINS - 1);
    while b > 0 && acc.wrapping_add(*hist.add(b as usize)) < need {
        acc = acc.wrapping_add(*hist.add(b as usize));
        b -= 1;
    }
    (b, acc)
}

/// Four 8-bit radix passes converge `pivot` on the k-th largest key and `need` on how many entries
/// tied at it the selection takes; returns `(pivot, need)`. LDS: `hist`, then `sh = [pivot, need]`
/// (thread 0's walk, broadcast).
#[inline(always)]
unsafe fn topk_pivot(a: &TopK, hist: *mut u32, sh: *mut u32) -> (u32, u32) {
    let (mut pivot, mut mask, mut need) = (0u32, 0u32, a.k as u32);
    for shift in [24u32, 16, 8, 0] {
        topk_hist(a, hist, (pivot, mask, shift));
        if tid() == 0 {
            let (b, acc) = topk_bin(hist, need);
            *sh = pivot | (b << shift);
            *sh.add(1) = need.wrapping_sub(acc);
        }
        sync_threads();
        pivot = *sh;
        need = *sh.add(1);
        mask |= 0xff << shift;
        sync_threads();
    }
    (pivot, need)
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_topk(a: TopK) {
    let hist = lds::<u32>();
    let scan = hist.add(TOPK_BINS as usize);
    let (pivot, need) = topk_pivot(&a, hist, scan.add(TOPK_BLOCK as usize));
    // Compaction, ascending: slot(i) = #(key > pivot before i) + min(#(key == pivot before i),
    // need), from one packed scan per chunk — no atomics, so the order is deterministic.
    let th = tid() as usize;
    let (mut base_gt, mut base_eq) = (0u32, 0u32);
    let mut base = 0i32;
    while base < a.nt {
        let i = base + th as i32;
        let live = i < a.nt;
        let u = if live {
            topk_key(*a.s.add(i as usize))
        } else {
            0
        };
        // 0/1 flags combined bitwise: the C's `live && …` and `gt || (eq && e < need)`, as
        // integers.
        let gt = u32::from(live) & u32::from(u > pivot);
        let eq = u32::from(live) & u32::from(u == pivot);
        let mut total = 0u32;
        let pre = topk_scan(scan, gt | (eq << 16), th, &mut total);
        let g = base_gt + (pre & 0xffff);
        let e = base_eq + (pre >> 16);
        if gt | (eq & u32::from(e < need)) != 0 {
            *a.rows.add((g + e.min(need)) as usize) = i as u32;
        }
        base_gt += total & 0xffff;
        base_eq += total >> 16;
        base += TOPK_BLOCK as i32;
    }
}

/// One block row per HIP block, `b.block` threads (`next_pow2(hd)`), threads past `hd` masked on
/// the VALUE so every thread reaches `block_sum_lds`'s barriers.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_pool_norm_f32(b: Blocked<PoolNorm>) {
    let a = b.a;
    let (blk, t) = (bid() as i32, tid() as i32);
    let mut m = 0.0f32;
    if t < a.hd {
        let mut acc = 0.0f32;
        for r in 0..a.ratio {
            acc += *row(a.keys, blk * a.ratio + r, a.hd).add(t as usize);
        }
        m = acc / a.ratio as f32;
    }
    let sum = block_sum_lds(m * m, lds(), t as usize, b.block as usize);
    if t >= a.hd {
        return;
    }
    *a.out.add(blk as usize * a.hd as usize + t as usize) =
        m * (1.0 / sqrt(sum / a.hd as f32 + a.eps)) * *a.weight.add(t as usize);
}

/// Both block scorers' body: thread `id` scores query row `id / n_comp` against block
/// `id % n_comp`, summing `term(dot_h, w_h)` over the heads in an f32 accumulator and storing
/// `finish(acc)`. No `fma` anywhere: both C kernels are `#pragma clang fp contract(off)`, so every
/// product rounds before its sum. The `(size_t)` index arithmetic is the C's — in `int` it
/// overflows at a long prefill.
///
/// # Safety
/// `a`'s four buffers are sized as `ScoreBlocks` documents.
#[inline(always)]
pub(crate) unsafe fn score_blocks(
    a: ScoreBlocks,
    term: impl Fn(f32, f32) -> f32,
    finish: impl Fn(f32) -> f32,
) {
    let ScoreBufs { q, kv, w, score } = a.bufs;
    let id = bid() as usize * ELEMENTWISE_BLOCK as usize + tid() as usize;
    let (n_comp, heads, hd) = (a.n_comp as usize, a.heads as usize, a.hd as usize);
    if id >= a.s as usize * n_comp {
        return;
    }
    let (t, c) = (id / n_comp, id % n_comp);
    let kvr = kv.add(c * hd);
    let mut acc = 0.0f32;
    for hh in 0..heads {
        let qh = q.add((t * heads + hh) * hd);
        let mut dot = 0.0f32;
        for i in 0..hd {
            dot += *qh.add(i) * *kvr.add(i);
        }
        acc += term(dot, *w.add(t * heads + hh));
    }
    *score.add(id) = finish(acc);
}

/// Qwen's fp32 contract: the ReLU sees the unrounded dot, and nothing rounds (the bf16 twin is
/// `blockindex.rs`'s).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn index_score_blocks_f32(a: ScoreBlocks) {
    score_blocks(a, |dot, w| maxnum(dot, 0.0) * w, |acc| acc);
}
