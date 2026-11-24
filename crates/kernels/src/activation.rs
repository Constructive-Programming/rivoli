//! Per-token activation transforms — the twins of `kernels/activation.hip` at tag
//! `archive/hip-kernels`: the gated combines, RMSNorm in three forms, `act_quant_f8`, and
//! RoPE in two. That file's comments carry each kernel's contract (aliasing, which model
//! uses it, why a near-twin is a different function); they are not repeated here. What
//! this file adds is what the port had to decide: where hipcc fused, and what replaced
//! `blockDim`.
//!
//! In-place callers (`h == g`, `y == x`) are why no pointer here is `noalias`: HIP's
//! `__restrict__` on an aliased call is UB the C gets away with by one-read-then-one-write;
//! Rust raw pointers make no such promise to the optimizer in the first place.

use crate::gpu::{bid, elementwise, fma, for_i32, fp8_lut, lds, ocml, tid};
use crate::prelude::actquant::{
    ACT_QUANT_AMAX_FLOOR, FP8_MAX_INV, act_quant_roundtrip, fast_round_scale,
};
use crate::prelude::formats::{e2m1f, rbf16};
use crate::prelude::llvm::{ldexp, maxnum, minnum};
use crate::prelude::reduce::{block_sum_lds, sync_threads, tree, wave_max};
use crate::prelude::activation::tanh_ocml;
use crate::prelude::{situ_glu, swiglu_clamped};
use core::intrinsics::{exp, sqrt, unchecked_div, unchecked_rem};
use rivoli_abi::geom::{ACT_QUANT_BLOCK, ACT_QUANT_THREADS, ELEMENTWISE_BLOCK, ROW_BLOCK, WAVE};
use rivoli_abi::kernarg::{ActQuant, Gate, GateClamped, GateSitu, KvScatter, L2Clip, LogScaledL2Clip, MessageGate, Mxfp8Dequant, Nvfp4Dequant, RelBiasGather, Rope, RowNorm, ScaledL2Clip, Sconv, SinkAttend, WeightlessNorm};

/// The gated combines' shared shape: `h[i] = f(g[i], u[i])` for the thread's one `i < n`.
#[inline(always)]
unsafe fn gate_map(a: Gate, f: impl FnOnce(f32, f32) -> f32) {
    elementwise(a.n, |i| *a.h.add(i) = f(*a.g.add(i), *a.u.add(i)));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn swiglu(a: Gate) {
    gate_map(a, |g, u| (g / (1.0 + exp(-g))) * u);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn swiglu_clamped_bf16(a: GateClamped) {
    gate_map(a.gate, |g, u| rbf16(swiglu_clamped(g, u, a.limit)));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn situ_glu_f32(a: GateSitu) {
    gate_map(a.gate, |g, u| situ_glu(g, u, a.b1, a.b2));
}

/// `Σ value(i)²` over this thread's `ROW_BLOCK`-strided slice of `0..n`, fused per term the way
/// hipcc fuses `local += v * v`.
#[inline(always)]
unsafe fn strided_sum_sq_by(n: i32, value: impl Fn(usize) -> f32) -> f32 {
    let mut local = 0.0f32;
    for_i32(tid(), n, ROW_BLOCK, |i| {
        let v = value(i);
        local = fma(v, v, local);
    });
    local
}

/// `Σ x[i]²` over this thread's `ROW_BLOCK`-strided slice of `x[0..n]`.
#[inline(always)]
unsafe fn strided_sum_sq(x: *const f32, n: i32) -> f32 {
    strided_sum_sq_by(n, |i| *x.add(i))
}

/// `y[i] = x[i] · inv · weight(w[i])` over the thread's strided slice — `(x·inv)·weight`,
/// the C's association.
#[inline(always)]
unsafe fn row_scale(a: RowNorm, inv: f32, weight: impl Fn(f32) -> f32) {
    for_i32(tid(), a.n, ROW_BLOCK, |k| {
        *a.y.add(k) = *a.x.add(k) * inv * weight(*a.w.add(k));
    });
}

/// One row; the LDS ladder is spelled out rather than `block_sum_lds`, as in the C (which
/// argues why: that helper's trailing barrier is unmeasured on GLM's decode path).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_single(a: RowNorm) {
    let red = lds::<f32>();
    *red.add(tid() as usize) = strided_sum_sq(a.x, a.n);
    sync_threads();
    tree(tid() as usize, ROW_BLOCK as usize, |t, s| {
        *red.add(t) += *red.add(s)
    });
    let inv = 1.0 / sqrt(*red / a.n as f32 + a.eps);
    row_scale(a, inv, |w| w);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_centered_single(a: RowNorm) {
    let local = strided_sum_sq(a.x, a.n);
    let inv = 1.0
        / sqrt(
            block_sum_lds(local, lds(), tid() as usize, ROW_BLOCK as usize) / a.n as f32 + a.eps,
        );
    row_scale(a, inv, |w| 1.0 + w);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rmsnorm_weightless_batch(a: WeightlessNorm) {
    // `(size_t)blockIdx.x * d`: `d` sign-extends to the 64-bit product, as in C.
    let row =
        a.x.wrapping_add((bid() as usize).wrapping_mul(a.d as isize as usize));
    let p = strided_sum_sq(row, a.d);
    let rs = a.scale
        / sqrt(block_sum_lds(p, lds(), tid() as usize, ROW_BLOCK as usize) / a.d as f32 + a.eps);
    for_i32(tid(), a.d, ROW_BLOCK, |i| *row.add(i) *= rs);
}

/// `Σ (x·scale)[i]²` over this thread's `ROW_BLOCK`-strided slice — the scaled clip's statistic.
#[inline(always)]
unsafe fn strided_sum_sq_scaled(x: *const f32, scale: *const f32, n: i32) -> f32 {
    strided_sum_sq_by(n, |i| *x.add(i) * *scale.add(i))
}

/// One row's clip factor: `min(threshold/(‖·‖+1e-6), 1)` from a leaf sum. The block reduction
/// lands in LDS, the norm is the raw L2 (no mean), the `+1e-6` is in the denominator (trap T4),
/// and the clamp at 1 keeps the clip scaling DOWN only.
#[inline(always)]
unsafe fn clip_factor(local: f32, threshold: f32) -> f32 {
    let norm = sqrt(block_sum_lds(local, lds(), tid() as usize, ROW_BLOCK as usize));
    minnum(threshold / (norm + 1e-6), 1.0)
}

/// One block per row: the row's three pointers and its clip factor. `bid()` is the row, so the
/// pointers advance by `row·n` — the batching is what turns `heads` launches into one.
#[inline(always)]
unsafe fn scaled_row(a: ScaledL2Clip) -> (f32, *const f32, *const f32, *mut f32) {
    let c = a.clip;
    let off = bid() as usize * c.n as usize;
    let (x, y, s) = (c.x.add(off), c.y.add(off), a.scale.add(off));
    (clip_factor(strided_sum_sq_scaled(x, s, c.n), c.threshold), x, s, y)
}

/// inkling's per-head L2 clip — no C twin (owner direction 2026-11-24), oracle-validated against
/// `rivoli_oracles::inkling::l2_clip` and the S3 anchor. One block per row of `n`:
/// `y[i] = x[i]·min(threshold/(‖x‖+1e-6), 1)`. `block_sum_lds` reduces the sum of squares (its
/// trailing barrier is fine here — this is not GLM's decode path, and it keeps the ladder off
/// `rmsnorm_single`'s hand-rolled spelling). Safe in place (`y` may alias `x`): the barrier
/// separates every read of `x` from every write of `y`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn l2_clip(a: L2Clip) {
    let off = bid() as usize * a.n as usize;
    let (x, y) = (a.x.add(off), a.y.add(off));
    let scale = clip_factor(strided_sum_sq(x, a.n), a.threshold);
    for_i32(tid(), a.n, ROW_BLOCK, |i| *y.add(i) = *x.add(i) * scale);
}

/// inkling's sliding-layer q/k norm — a learned per-head `scale` applied FIRST, then the L2 clip.
/// No C twin (oracle-validated against `rivoli_oracles::inkling::scaled_l2_clip`). One block per
/// row. Safe in place (`y` may alias `x`): the barrier separates every read from every write.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn scaled_l2_clip(a: ScaledL2Clip) {
    let (clip, x, s, y) = scaled_row(a);
    for_i32(tid(), a.clip.n, ROW_BLOCK, |i| *y.add(i) = (*x.add(i) * *s.add(i)) * clip);
}

/// inkling's full-layer q/k norm — the scaled clip then the position-dependent LogScale `tau`
/// (T11: sliding layers omit it). No C twin (oracle-validated against
/// `rivoli_oracles::inkling::log_scaled_l2_clip`). One block per row. Safe in place.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn log_scaled_l2_clip(a: LogScaledL2Clip) {
    let (clip, x, s, y) = scaled_row(a.scaled);
    for_i32(tid(), a.scaled.clip.n, ROW_BLOCK, |i| {
        *y.add(i) = (*x.add(i) * *s.add(i)) * clip * a.tau
    });
}

/// One wave per `ACT_QUANT_BLOCK`-element group. `b >= nblk` is uniform per wave, so the
/// early return never splits a wave before `wave_max`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn act_quant_f8(a: ActQuant) {
    let lane = tid() & (WAVE - 1);
    let b = u64::from(bid()) * u64::from(ACT_QUANT_THREADS / WAVE) + u64::from(tid() / WAVE);
    if b >= a.nblk {
        return;
    }
    let seg = a.v.add(b as usize * ACT_QUANT_BLOCK as usize);
    let mut amax = 0.0f32;
    for_i32(lane, ACT_QUANT_BLOCK as i32, WAVE, |i| {
        amax = maxnum(amax, (*seg.add(i)).abs());
    });
    let s = fast_round_scale(maxnum(wave_max(amax), ACT_QUANT_AMAX_FLOOR), FP8_MAX_INV);
    for_i32(lane, ACT_QUANT_BLOCK as i32, WAVE, |i| {
        let e = seg.add(i);
        *e = act_quant_roundtrip(*e, s);
    });
}

/// Both RoPE conventions: thread `j` reads the pair at the indices `read(j, seg / 2)` names,
/// barriers (every pair is read before any is written — the in-place race-freedom), and
/// writes the rotation to `(j, seg/2 + j)`. Both products that multiply `cs` are the fused
/// ones — hipcc's ISA: `v_fma_f32 lo, a, cs, -(b·sn)` and `v_fmac_f32 hi += b·cs` over
/// `a·sn`. The block is `seg / 2` threads and nothing here reads blockDim.
#[inline(always)]
unsafe fn rope(a: Rope, read: impl Fn(i32, i32) -> (i32, i32)) {
    let j = tid() as i32;
    let half = a.seg / 2;
    // `(size_t)s * stride`, with C's sign extension of `stride`.
    let v = a
        .base
        .wrapping_add((bid() as usize).wrapping_mul(a.stride as isize as usize));
    let (ra, rb) = read(j, half);
    let (x, y) = (*v.add(ra as usize), *v.add(rb as usize));
    sync_threads();
    let inv = ocml::pow(a.theta, -2.0 * f64::from(j) / f64::from(a.seg));
    let ang = f64::from(a.pos) * inv;
    let (cs, sn) = (ocml::cos(ang) as f32, ocml::sin(ang) as f32);
    *v.add(j as usize) = fma(x, cs, -(y * sn));
    *v.add((half + j) as usize) = fma(y, cs, x * sn);
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rope_interleave(a: Rope) {
    rope(a, |j, _| (2 * j, 2 * j + 1));
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rope_split_half(a: Rope) {
    rope(a, |j, half| (j, half + j));
}

/// The 64-bit elementwise id, `blockIdx·ELEMENTWISE_BLOCK + threadIdx`, for the expert dequants
/// whose `n` is `u64` (one Small expert layer is ~8e9 elements). `None` past `n`.
#[inline(always)]
fn elementwise_id_u64(n: u64) -> Option<u64> {
    let i = u64::from(bid()) * u64::from(ELEMENTWISE_BLOCK) + u64::from(tid());
    (i < n).then_some(i)
}

/// `mxfp8_dequant` — inkling's MXFP8 expert decode (no C twin; oracle-validated): one e4m3 value
/// per byte and one power-of-two scale per 32 values along the input dim, ROW-WISE `[1,32]` (T1:
/// the scale is a raw U8 exponent, not e4m3). The e4m3 decode is the shared fp8 LUT; `ldexp(1, e)`
/// is clang's fold of `exp2f((float)e)`, exact for the integer exponent the oracle's `powi` uses,
/// so the product matches `inkling_dequant::Mxfp8::dequant` bit for bit. The index and the two
/// extent divisions are 64-bit and native-integer: `unchecked_div`/`unchecked_rem` are the
/// device's divide instructions, where Rust's `/` would emit a zero-divisor trap branch the
/// launcher has already ruled out (`n % cols == 0`, `cols > 0`).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn mxfp8_dequant(a: Mxfp8Dequant) {
    let lut = fp8_lut();
    let cols = a.cols as u64;
    let Some(i) = elementwise_id_u64(a.n) else {
        return;
    };
    let c = unchecked_rem(i, cols);
    let row = unchecked_div(i, cols);
    let s = *a
        .scale
        .add((row * unchecked_div(cols, 32) + unchecked_div(c, 32)) as usize);
    let sc = ldexp(1.0, i32::from(s) - 127);
    *a.out.add(i as usize) = *lut.add(*a.weight.add(i as usize) as usize) * sc;
}

/// `nvfp4_dequant` — inkling's NVFP4 expert decode (no C twin; oracle-validated): two E2M1 codes
/// per byte (low nibble even), one e4m3 scale per 16 values along the input dim, and a per-expert
/// factor `amax[e]/denom`. The association is `(E2M1[code]·e4m3(scale))·factor` — NOT
/// `code·(scale·factor)`, f32 multiply is not associative and the bf16 golden is bit-compared
/// (LOAD:813). The e4m3 scale decode is the shared fp8 LUT; the E2M1 code is the prelude's
/// `e2m1f` (the same immediate `0xC8643210` fold, so code 8 is `-0.0` by its sign bit).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn nvfp4_dequant(a: Nvfp4Dequant) {
    let lut = fp8_lut();
    let cols_packed = a.cols_packed as u64;
    let Some(i) = elementwise_id_u64(a.n) else {
        return;
    };
    let pair = unchecked_div(i, 2);
    let cp = unchecked_rem(pair, cols_packed);
    let row = unchecked_div(pair, cols_packed);
    let e = unchecked_div(row, a.rows as u64) as usize;
    let byte = *a.weight.add(pair as usize);
    let code = if unchecked_rem(i, 2) == 0 {
        byte & 0xF
    } else {
        byte >> 4
    };
    let bs = *lut.add(
        *a.scale
            .add((row * unchecked_div(cols_packed, 8) + unchecked_div(cp, 8)) as usize) as usize,
    );
    let factor = *a.amax.add(e) / a.denom;
    *a.out.add(i as usize) = (e2m1f(u32::from(code)) * bs) * factor;
}

/// `sconv` — inkling's static short-convolution decode step, state included (no C twin;
/// oracle-validated): one thread per channel accumulates the k-tap cross-correlation over the
/// window `[state (past k−1), current]` in tap order, adds the residual (the current token), and
/// divides by `√(k+1)` (T15). It then ADVANCES the state in place (shift left one, append current —
/// MOD:254), so the caller keeps no separate window copy. The reduction is short (k=4); the oracle
/// comparison is relative, the clip family's convention for a sum whose accumulation may contract
/// to FMA on the device.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn sconv(a: Sconv) {
    elementwise(a.channels, |c| {
        let k = a.k as usize;
        let hist = k - 1;
        let (wb, sb) = (c * k, c * hist);
        let cur = *a.current.add(c);
        let mut acc = 0.0f32;
        for j in 0..hist {
            acc += *a.weight.add(wb + j) * *a.state.add(sb + j);
        }
        acc += *a.weight.add(wb + hist) * cur;
        *a.out.add(c) = (acc + cur) / a.divisor;
        // Shift left one, append current. Increasing j reads state[j+1] before overwriting it at
        // step j+1, so the in-place shift is safe; each channel's slice is thread-private.
        for j in 0..hist {
            let nxt = if j + 1 < hist { *a.state.add(sb + j + 1) } else { cur };
            *a.state.add(sb + j) = nxt;
        }
    });
}

/// `sink_attend` — inkling's single-query attention decode step with a sink (no C twin;
/// oracle-validated). One block per query head: the block stripes the `n` QK dots into LDS
/// logits, reduces their max (with the head's on-device `sink` logit as thread 0's leaf), turns
/// them into exponentials in place while summing them, and then stripes the `d` outputs of the
/// weight·V sum. The sink column is joined into the max and the sum but never written back
/// (T10): its mass stays out of the output, so the real weights sum to < 1.
///
/// **The softmax is a block reduction, not thread 0's serial loop** (CHANGED 2026-11-24 at the
/// batching review): one block per head means the old single-thread softmax would have left a
/// 256-thread block with 255 idle threads across `n` exponentials, and a long context
/// (`n = 1024`) is exactly where that shows. Two LDS reductions of `ROW_BLOCK` leaves — a max
/// tree and `block_sum_lds` — replace it; both are re-associated against the oracle's sequential
/// fold, so the comparison stays relative.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn sink_attend(a: SinkAttend) {
    let logits = lds::<f32>();
    let red = logits.add(a.n as usize);
    let t = tid();
    let d = a.d as usize;
    let n = a.n as usize;
    let h = bid() as usize;
    let kvh = unchecked_div(h as i32, unchecked_div(a.heads, a.kv_heads)) as usize;
    let (q_h, out_h, bias_h) = (a.q.add(h * d), a.out.add(h * d), a.bias.add(h * n));
    let kv = kvh * a.kv_stride as usize * d;
    let (k_h, v_h) = (a.k.add(kv), a.v.add(kv));
    let sink = *a.sink.add(h);
    // Phase 1: the QK dots into LDS, thread 0 seeding the max with the sink logit.
    let mut local_max = if t == 0 { sink } else { f32::NEG_INFINITY };
    for_i32(t, a.n, ROW_BLOCK, |j| {
        let kj = k_h.add(j * d);
        let mut dot = 0.0f32;
        for i in 0..d {
            dot += *q_h.add(i) * *kj.add(i);
        }
        let val = dot * a.scaling + *bias_h.add(j);
        *logits.add(j) = val;
        local_max = maxnum(local_max, val);
    });
    *red.add(t as usize) = local_max;
    sync_threads();
    tree(t as usize, ROW_BLOCK as usize, |x, y| {
        *red.add(x) = maxnum(*red.add(x), *red.add(y))
    });
    let m = *red;
    // The barrier `tree` ended on orders the leaves, not this read: `block_sum_lds` below
    // overwrites `red[0]`, so every thread must have its `m` before any thread writes it again.
    sync_threads();
    // Phase 2: exponentials in place (ONE `exp` per key, reused by the AV sum) and the sum.
    let mut local_z = if t == 0 { exp(sink - m) } else { 0.0f32 };
    for_i32(t, a.n, ROW_BLOCK, |j| {
        let e = exp(*logits.add(j) - m);
        *logits.add(j) = e;
        local_z += e;
    });
    let z = block_sum_lds(local_z, red, t as usize, ROW_BLOCK as usize);
    // Phase 3: the weight·V sum, the single `1/z` applied once per output.
    let inv_z = 1.0f32 / z;
    for_i32(t, a.d, ROW_BLOCK, |i| {
        let mut acc = 0.0f32;
        for j in 0..n {
            acc += *logits.add(j) * *v_h.add(j * d + i);
        }
        *out_h.add(i) = acc * inv_z;
    });
}

/// `message_gate` — inkling's loop MessageGate pointwise tail (no C twin; oracle-validated):
/// `out[i] = (1−tanh(forget_logit[i]))·hidden[i] + tanh(gain_logit[i])·message[i]`. The tanh is
/// ocml's `tanhf` (the softcap/SiTU kernel's same body); the oracle comparison is relative, the
/// ~1-ulp gap between ocml's tanh and the host's. The projections are the engine's GEMVs.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn message_gate(a: MessageGate) {
    elementwise(a.n, |i| {
        let f = tanh_ocml(*a.forget_logit.add(i));
        let g = tanh_ocml(*a.gain_logit.add(i));
        *a.out.add(i) = (1.0 - f) * *a.hidden.add(i) + g * *a.message.add(i);
    });
}

/// `kv_scatter` — inkling's KV-cache append (no C twin; oracle-validated): one thread per element
/// scatters one token's projected vector into the per-head-deinterleaved cache, so each head's `n`
/// vectors are contiguous for `sink_attend`. Element `e` (head `e/head_dim`, lane `e%head_dim`)
/// lands at `head·ctx·head_dim + pos·head_dim + lane`. A pure data movement — the oracle comparison
/// is bitwise.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn kv_scatter(a: KvScatter) {
    let hd = a.head_dim as usize;
    elementwise(a.n, |e| {
        // The launcher's guards make `hd > 0` and `e % hd` exact; `unchecked_*` is the device's
        // divide, not Rust's `/` (which would carry a zero-divisor trap branch).
        let (head, lane) = (unchecked_div(e, hd), unchecked_rem(e, hd));
        let idx = head * a.ctx as usize * hd + a.pos as usize * hd + lane;
        *a.dst.add(idx) = *a.src.add(e);
    });
}

/// `rel_bias_gather` — inkling's relative-position bias distance gather (no C twin; oracle-validated):
/// one thread per (head, key) pair reads `rel_logits[h][pos−(start+t)]` when that distance is in
/// `[0, rel_extent)`, else writes 0 (MOD:224–227). Key slot `t` is absolute position
/// `start + t` — the sliding window's base, which is `0` for a global layer and the window's
/// first position past the start. A pure gather + mask — the oracle comparison is bitwise.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn rel_bias_gather(a: RelBiasGather) {
    let n = a.n as usize;
    let extent = a.rel_extent as usize;
    let pos = a.pos as usize;
    elementwise(a.heads * a.n, |i| {
        // `n > 0` is a launcher guard, so the split needs no trap branch (`unchecked_*`).
        let (h, t) = (unchecked_div(i, n), unchecked_rem(i, n));
        let key_pos = a.start as usize + t;
        let mut v = 0.0f32;
        if pos >= key_pos && pos - key_pos < extent {
            v = *a.rel_logits.add(h * extent + (pos - key_pos));
        }
        *a.bias.add(i) = v;
    });
}
