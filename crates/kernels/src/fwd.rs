//! The forward pass's glue kernels — the twins of `kernels/fwd.hip` at tag
//! `archive/hip-kernels`, which carries each kernel's contract and record (the sigmoid-gate
//! trap, the softcap's argmax blindness, the instruments' no-sync argument). What this file
//! adds is what the port decided.
//!
//! **Fused sites, read from hipcc's gfx1151 ISA:** one — `f2e4m3`'s subnormal `a·512 + 0.5`
//! is `v_fma_f32 v3, 0x44000000, |v3|, 0.5`. The softcap's `tanhf` is the prelude's ocml
//! transliteration (hipcc's IR inlines the same `fmuladd` body). Everything else is a single
//! operation, a divide or a compare.
//!
//! **Atomics:** hipcc's `atomicCAS`/`atomicXor` are `syncscope("agent") monotonic`; Rust's
//! `Atomic*` are system scope. The scope moves no value — the CAS's winner and the XOR's
//! result are each defined by the operation — but it can move the instruction: the system
//! XOR became a CAS loop, so `hash_rows` writes the agent-scope XOR out (`agent_xor`).

use crate::gpu::{bid, elementwise, fma, for_i32, lds, tid};
use crate::prelude::activation::tanh_ocml;
use crate::prelude::formats::f2bf16;
use crate::prelude::llvm::maxnum;
use crate::prelude::reduce::{sync_threads, tree};
use core::intrinsics::{exp, fabs, floor, unchecked_div, unchecked_rem};
use core::sync::atomic::{AtomicU32, Ordering};
use rivoli_abi::geom::{APPEND_KV_GROUP, ARGMAX_BLOCK, HASH_BLOCK};
use rivoli_abi::kernarg::{
    AppendKv, Argmax, EmbedI8, FlagNonfinite, GatherRope, HashRows, InPlaceBinary, Softcap,
};

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn embed_i8_row(a: EmbedI8) {
    // `token >= 0` — the launcher refuses a negative one (1001).
    let t = a.token as usize;
    elementwise(a.hidden, |i| {
        let q = *a.packed.add(t * a.hidden as usize + i) as i8;
        *a.x.add(i) = f32::from(q) * *a.scale.add(t);
    });
}

/// f32 → OCP e4m3, `fwd.hip`'s own quantizer (NOT the prelude's `f2e4m3_rne`): RNE on the
/// normal mantissa, round-half-AWAY on the subnormal, saturating to ±448, `0x7f` for NaN.
#[inline(always)]
fn f2e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    let sign: u8 = if x.is_sign_negative() { 0x80 } else { 0 };
    let a = fabs(x);
    let magnitude = if a >= 448.0 {
        0x7e
    } else if a < 0.000_976_562_5 {
        0
    } else {
        e4m3_magnitude(a)
    };
    sign | magnitude
}

/// The e4m3 magnitude bits of `a` in `[2^-10, 448)`.
#[inline(always)]
fn e4m3_magnitude(a: f32) -> u8 {
    let bits = a.to_bits();
    let e = ((bits >> 23) & 0xff) as i32 - 127;
    if e < -6 {
        // The one fused site (module doc). `m == 8` promotes to the smallest normal.
        let m = floor(fma(a, 512.0, 0.5)) as i32;
        return if m >= 8 { 0x08 } else { m as u8 };
    }
    let mant = bits & 0x007f_ffff;
    let (m3, rem, half) = ((mant >> 20) as u8, mant & 0x000f_ffff, 0x0008_0000);
    let odd_tie = rem == half && m3 & 1 != 0;
    let m3 = m3 + u8::from(rem > half || odd_tie);
    // A rounded-up mantissa of 8 carries into the exponent.
    let (m3, exp) = if m3 == 8 { (0, e + 8) } else { (m3, e + 7) };
    if exp >= 15 && m3 >= 7 {
        0x7e
    } else {
        ((exp as u8) << 3) | m3
    }
}

/// One block of `kvl` threads (`kvl` a launch-time argument, so no block-size constant —
/// the kernel reads only `threadIdx.x`): each `APPEND_KV_GROUP`-lane group reduces its amax
/// in LDS (`geom::append_kv_lds_bytes(kvl)`, one f32 leaf per thread), then every thread
/// quantizes its element.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn append_kv(a: AppendKv) {
    let red = lds::<f32>();
    let i = tid() as usize;
    let g = APPEND_KV_GROUP as usize;
    *red.add(i) = fabs(*a.latent.add(i));
    sync_threads();
    let (b, lane) = (i / g, i % g);
    let grp = red.add(b * g);
    tree(lane, g, |t, s| {
        *grp.add(t) = maxnum(*grp.add(t), *grp.add(s))
    });
    let amax = *grp;
    let scl = if amax > 0.0 { amax / 448.0 } else { 1.0 };
    let pos = a.pos as usize;
    if lane == 0 {
        *a.lscale.add(pos * a.n_blocks as usize + b) = scl;
    }
    *a.lc8.add(pos * a.kvl as usize + i) = f2e4m3(*a.latent.add(i) / scl);
    if (i as i32) < a.ropn {
        *a.rc.add(pos * a.ropn as usize + i) = f2bf16(*a.rope.add(i));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn gather_rope(a: GatherRope) {
    elementwise(a.h.wrapping_mul(a.ropn), |idx| {
        // SAFETY: `ropn > 0` — the launcher refuses `ropn <= 0` (1001).
        let (head, d) = unsafe {
            (
                unchecked_div(idx as i32, a.ropn),
                unchecked_rem(idx as i32, a.ropn),
            )
        };
        *a.qrope.add(idx) =
            *a.q.add(head as usize * a.qh as usize + a.nope as usize + d as usize);
    });
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn vadd(a: InPlaceBinary) {
    elementwise(a.n, |i| *a.x.add(i) += *a.y.add(i));
}

/// `1/(1+exp(-g))`, the division form `torch.sigmoid` computes (the C's note).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn sigmoid_gate(a: InPlaceBinary) {
    elementwise(a.n, |i| *a.x.add(i) *= 1.0 / (1.0 + exp(-*a.y.add(i))));
}

/// Non-finite passes THROUGH (the C's note: laundering ±Inf deletes the argmax detector).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn logit_softcap(a: Softcap) {
    elementwise(a.n, |i| {
        let v = *a.x.add(i);
        if v.is_finite() {
            *a.x.add(i) = a.cap * tanh_ocml(v * a.mult / a.cap);
        }
    });
}

/// The host fold's `(value, index)` combine — associative and commutative, so the tree yields
/// the sequential `>` fold's answer: the max wins, NaN never wins, a tie takes the smaller
/// index (the C's `amax_combine`, branch for branch).
#[inline(always)]
fn amax_combine((va, ia): (f32, i32), (vb, ib): (f32, i32)) -> (f32, i32) {
    if vb > va {
        (vb, ib)
    } else if va > vb {
        (va, ia)
    } else if va.is_nan() {
        (vb, ib)
    } else if vb.is_nan() || ia <= ib {
        (va, ia)
    } else {
        (vb, ib)
    }
}

/// One block, LDS tree. Named for its launcher (`rivoli_argmax`); the C kernel was
/// `argmax_reduce`.
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn argmax(a: Argmax) {
    let sval = lds::<f32>();
    let sidx = sval.add(ARGMAX_BLOCK as usize).cast::<i32>();
    let t = tid();
    let mut best = (f32::NEG_INFINITY, 0i32);
    for_i32(t, a.n, ARGMAX_BLOCK, |i| {
        best = amax_combine(best, (*a.logits.add(i), i as i32));
    });
    let t = t as usize;
    (*sval.add(t), *sidx.add(t)) = best;
    sync_threads();
    tree(t, ARGMAX_BLOCK as usize, |t, s| {
        let at = |k: usize| (*sval.add(k), *sidx.add(k));
        (*sval.add(t), *sidx.add(t)) = amax_combine(at(t), at(s));
    });
    if t == 0 {
        *a.out_idx = *sidx;
        *a.out_val = *sval;
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn flag_nonfinite(a: FlagNonfinite) {
    elementwise(a.n, |i| {
        let v = *a.x.add(i);
        // The C's compare form, not `is_finite`: it is what the C wrote.
        if !(v > f32::NEG_INFINITY && v < f32::INFINITY) {
            let _ = AtomicU32::from_ptr(a.flag).compare_exchange(
                0,
                a.tag,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
        }
    });
}

/// splitmix64's finalizer over `(i_base + i) << 32 ^ bits`.
#[inline(always)]
fn mix(i_base: u64, i: i64, bits: u32) -> u64 {
    let z = (i_base.wrapping_add(i as u64) << 32) ^ u64::from(bits);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Grid-stride over SAMPLES, XOR-tree in LDS, ONE atomic per block (the C's notes).
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn hash_rows(a: HashRows) {
    let red = lds::<u64>();
    let samples: i64 = if a.stride > 0 {
        // SAFETY: `stride > 0` on this branch.
        i64::from(unchecked_div(a.n.wrapping_add(a.stride - 1), a.stride))
    } else {
        0
    };
    let bits = a.x.cast::<u32>();
    let mut acc = 0u64;
    let mut s_i = i64::from(bid().wrapping_mul(HASH_BLOCK).wrapping_add(tid()));
    while s_i < samples {
        let i = s_i * i64::from(a.stride);
        acc ^= mix(a.i_base, i, *bits.add(i as usize));
        s_i += a.grid_stride;
    }
    let t = tid() as usize;
    *red.add(t) = acc;
    sync_threads();
    tree(t, HASH_BLOCK as usize, |t, s| *red.add(t) ^= *red.add(s));
    if t == 0 {
        agent_xor(a.out, *red);
    }
}

/// `atomicXor(out, v)` as hipcc emits it: one relaxed, agent-scope `global_atomic_xor_b64`,
/// no return. Core's atomics are system scope, and LLVM expands a system-scope 64-bit XOR into a
/// load + `global_atomic_cmpswap_b64` retry loop, which timed `hash_rows` at 1.9× hipcc
/// (rust-kernels.md §5, "Perf follow-up"); nightly 1.102 has no syncscope route.
///
/// - **Scope.** Agent scope is hipcc's, and the readback is unchanged: `out` is device memory
///   the host reads only through a sync and a D2H copy.
/// - **Ordering.** The asm keeps its default memory clobber: a stronger compiler barrier than
///   relaxed, never a weaker one.
/// - **Counters.** `SIInsertWaitcnts` does not see the asm's `vscnt` increment. Nothing after it
///   reads memory and `s_endpgm` needs no wait; code added after it that reads `out` must wait
///   `vscnt` itself.
///
/// # Safety
/// `out` is a valid, 8-byte-aligned device address.
#[inline(always)]
unsafe fn agent_xor(out: *mut u64, v: u64) {
    // SAFETY: the caller's contract; the instruction writes only `*out`.
    unsafe {
        core::arch::asm!(
            "global_atomic_xor_b64 {a}, {v}, off",
            a = in(vgpr64) out,
            v = in(vgpr64) v,
            options(nostack, preserves_flags)
        );
    }
}
