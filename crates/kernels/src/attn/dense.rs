//! The two-pass (non-flash) softmax attends, scores staged in LDS: V4's
//! `gather_attn_shared_kv` (MQA over an index list, a sink in the denominator) and K3's
//! `mha_attend` (dense over explicit per-head K/V, an additive mask). Both reduce over a
//! 256-leaf LDS tree; contraction is off in the C, so nothing here is fused.

use crate::gpu::{bid, bid_y, for_i32, lds, tid};
use crate::prelude::formats::rbf16;
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{sync_threads, tree};
use core::intrinsics::exp;
use rivoli_abi::geom::{GATHER_ATTN_ACC, GATHER_ATTN_THREADS, MHA_THREADS};
use rivoli_abi::kernarg::{GatherAttn, MhaAttend};

/// Block-wide `op` over `v` through the LDS tree `red` (`leaves` floats, one per thread),
/// the result read by every thread. Both kernels' C stores, barriers, halves with a barrier
/// per level, then reads `red[0]` (gather through a broadcast slot); the trailing barrier
/// frees `red` for the next reduction.
///
/// # Safety
/// Every thread of the block calls this; `red` holds `leaves` floats of LDS.
#[inline(always)]
unsafe fn block_reduce(v: f32, red: *mut f32, leaves: u32, op: impl Fn(f32, f32) -> f32) -> f32 {
    let t = tid() as usize;
    *red.add(t) = v;
    sync_threads();
    tree(t, leaves as usize, |t, s| {
        *red.add(t) = op(*red.add(t), *red.add(s))
    });
    let r = *red;
    sync_threads();
    r
}

/// One (head, query row)'s gather: this head's query row staged in LDS, the row's index list,
/// and the logit/weight row.
#[derive(Clone, Copy)]
struct Gather {
    qs: *mut f32,
    lg: *mut f32,
    ix: *const i32,
}

/// Pass 1: one key per thread, the d-loop sequential (the oracle's order); a masked slot
/// (`idx < 0`) is −inf and never dereferenced. Returns this thread's max.
#[inline(always)]
unsafe fn logits(a: &GatherAttn, g: Gather) -> f32 {
    let d = a.d as usize;
    let mut mx = f32::NEG_INFINITY;
    for_i32(tid(), a.topk, GATHER_ATTN_THREADS, |j| {
        let p = *g.ix.add(j);
        let mut l = f32::NEG_INFINITY;
        if p >= 0 {
            let kr = a.kv.add(p as usize * d);
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += *g.qs.add(i) * *kr.add(i);
            }
            l = acc * a.scale;
        }
        *g.lg.add(j) = l;
        mx = maxnum(mx, l);
    });
    mx
}

/// `acc[c] += e · kr[col(c)]` over this thread's columns inside `d`.
#[inline(always)]
unsafe fn accumulate(acc: &mut [f32; GATHER_ATTN_ACC as usize], e: f32, kr: *const f32, d: usize) {
    for c in 0..GATHER_ATTN_ACC {
        let i = col(c);
        if i < d {
            acc[c as usize] += e * *kr.add(i);
        }
    }
}

/// Pass 3: the values, `j` ascending over every slot (the oracle's order), this thread's
/// columns `GATHER_ATTN_THREADS` apart; returns the thread's `GATHER_ATTN_ACC` sums.
#[inline(always)]
unsafe fn values(a: &GatherAttn, g: Gather) -> [f32; GATHER_ATTN_ACC as usize] {
    let d = a.d as usize;
    let mut acc = [0.0f32; GATHER_ATTN_ACC as usize];
    for_i32(0, a.topk, 1, |j| {
        let p = *g.ix.add(j);
        if p >= 0 {
            accumulate(&mut acc, *g.lg.add(j), a.kv.add(p as usize * d), d);
        }
    });
    acc
}

/// This thread's column `c`: `tid + c·GATHER_ATTN_THREADS`.
#[inline(always)]
fn col(c: u32) -> usize {
    (tid() + c * GATHER_ATTN_THREADS) as usize
}

/// One block per (head, query row): logits, max, exp-and-sum, values, `bf16(acc / denom)`.
/// LDS: `qs[d] | lg[topk] | red[GATHER_ATTN_THREADS]`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gather_attn_shared_kv(a: GatherAttn) {
    let (d, topk) = (a.d as usize, a.topk);
    let qs = lds::<f32>();
    let red = qs.add(d + topk as usize);
    let (head, row) = (bid() as usize, bid_y() as usize);
    let g = Gather {
        qs,
        lg: qs.add(d),
        ix: a.idxs.add(row * topk as usize),
    };
    let qrow = a.q.add((row * a.h as usize + head) * d);
    for_i32(tid(), a.d, GATHER_ATTN_THREADS, |i| {
        *qs.add(i) = *qrow.add(i)
    });
    sync_threads();
    let mx = block_reduce(logits(&a, g), red, GATHER_ATTN_THREADS, |x, y| maxnum(x, y));
    let mut ps = 0.0f32;
    for_i32(tid(), topk, GATHER_ATTN_THREADS, |j| {
        let e = if *g.ix.add(j) >= 0 {
            exp(*g.lg.add(j) - mx)
        } else {
            0.0
        };
        *g.lg.add(j) = e;
        ps += e;
    });
    let total = block_reduce(ps, red, GATHER_ATTN_THREADS, |x, y| x + y);
    // The sink joins the denominator only; a fully-masked query (`mx` −inf) is NaN.
    let denom = if mx.is_finite() {
        total + exp(*a.sink.add(head) - mx)
    } else {
        f32::NAN
    };
    let acc = values(&a, g);
    let orow = a.o.add((row * a.h as usize + head) * d);
    for c in 0..GATHER_ATTN_ACC {
        if col(c) < d {
            *orow.add(col(c)) = rbf16(acc[c as usize] / denom);
        }
    }
}

/// One block per head: scores one position per thread, max, exp in place and sum, then one
/// output column per thread. A fully-masked row is NaN, deliberately (the C's note).
/// LDS: `sc[kv] | red[MHA_THREADS]`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mha_attend(a: MhaAttend) {
    let (kv, d, dv) = (a.kv as usize, a.d as usize, a.dv as usize);
    let sc = lds::<f32>();
    let red = sc.add(kv);
    let (t, head) = (tid(), bid() as usize);
    let qh = a.q.add(head * d);
    let kh = a.k.add(head * kv * d);
    let vh = a.v.add(head * kv * dv);
    let mut local = f32::NEG_INFINITY;
    for_i32(t, a.kv, MHA_THREADS, |s| {
        let ks = kh.add(s * d);
        let mut acc = 0.0f32;
        for i in 0..d {
            acc += *qh.add(i) * *ks.add(i);
        }
        let mask = if a.mask.is_null() {
            0.0
        } else {
            *a.mask.add(s)
        };
        let acc = acc * a.scale + mask;
        *sc.add(s) = acc;
        local = maxnum(local, acc);
    });
    let m = block_reduce(local, red, MHA_THREADS, |x, y| maxnum(x, y));
    let mut ls = 0.0f32;
    for_i32(t, a.kv, MHA_THREADS, |s| {
        let e = exp(*sc.add(s) - m);
        *sc.add(s) = e;
        ls += e;
    });
    let inv_z = 1.0 / block_reduce(ls, red, MHA_THREADS, |x, y| x + y);
    for_i32(t, a.dv, MHA_THREADS, |j| {
        let mut acc = 0.0f32;
        for s in 0..kv {
            acc += *sc.add(s) * *vh.add(s * dv + j);
        }
        *a.out.add(head * dv + j) = acc * inv_z;
    });
}
