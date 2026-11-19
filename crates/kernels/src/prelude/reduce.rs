//! Wave/block geometry, the workgroup barrier, and the three reductions over them. Port of
//! `reduce.hpp` at tag `archive/hip-kernels`.
//!
//! No `fmuladd`: the only float operators are `+` and `maxnum` over shuffles.
//!
//! **No helper here reads the block size from the dispatch packet** (Phase 0c, 2026-11-18:
//! that read costs ~1.1–1.2 µs per launch at tens of workgroups). `block_sum_lds` takes it
//! from the kernel, which knows it as a constant or a kernel argument.

use core::arch::amdgpu::mbcnt_lo;
use core::sync::atomic::{Ordering::SeqCst, compiler_fence};

use super::llvm::{maxnum, wave_shuffle};

/// gfx1151 is native wave32 — `rivoli_abi`'s constant, so a launcher and this prelude
/// cannot disagree about it.
pub const WAVE: usize = rivoli_abi::geom::WAVE as usize;
/// block = 256 threads = 8 waves = 8 output rows — `rivoli_abi`'s, so a launcher's grid and
/// the kernel's row index cannot disagree.
pub const ROWS_PER_BLOCK: usize = rivoli_abi::geom::ROWS_PER_BLOCK as usize;

/// `__syncthreads()` — HIP's workgroup-scope release fence, `s_barrier`, acquire fence, as
/// gfx1151 (WGP mode) lowers them: LDS (`lgkmcnt`), vector loads (`vmcnt`) AND stores
/// (`vscnt`) drained, the barrier, then the L0 invalidate. hipcc emits
/// `s_waitcnt_vscnt null, 0x0` before the barrier after a global store (0b's probe,
/// 2026-11-18) and drops each wait when its counter is provably zero; asm cannot see the
/// counters, so it always waits — a no-op when nothing is outstanding, never weaker.
///
/// The barrier is the `llvm.amdgcn.s.barrier` intrinsic, not `s_barrier` in the asm
/// (CHANGED 2026-11-18 at the 0a/0b merge; rust-kernels.md §5 has the record): the intrinsic
/// is `convergent`, so LLVM cannot duplicate it into divergent branches, and inline asm is
/// not. It is also `IntrNoMem`, so something else must keep memory operations from crossing.
///
/// **That something is two compiler fences, not the asm** (CHANGED 2026-11-19, rust-kernels.md
/// §5 "Perf follow-up"). As opaque memory clobbers the asm counted as a write to every
/// address, so AMDGPU's uniform-load analysis (`isClobberedInFunction`, which skips fences)
/// kept every uniform global load after a barrier a per-lane `global_load` where hipcc gets an
/// `s_load`. A fence of any scope is a MemoryDef that alias analysis treats as reading and
/// writing all non-constant memory, so no load or store crosses one in IR; in machine code the
/// fences and the asm all have side effects, which keeps memory instructions on their side.
/// `singlethread` fences emit nothing, so the asm stays the whole hardware sequence. The
/// cross-lane ordering rests on that alias-analysis and side-effect behaviour, as hipcc's
/// does, not on LLVM's formal synchronization rules.
#[inline(always)]
pub fn sync_threads() {
    // SAFETY: waits and a cache invalidate; no registers or memory are written. Every caller
    // must reach the barrier uniformly across the block, as `__syncthreads` requires.
    unsafe {
        compiler_fence(SeqCst);
        core::arch::asm!(
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt_vscnt null, 0x0",
            options(nomem, nostack, preserves_flags)
        );
        core::arch::amdgpu::s_barrier();
        core::arch::asm!("buffer_gl0_inv", options(nomem, nostack, preserves_flags));
        compiler_fence(SeqCst);
    }
}

/// The LDS halving ladder every block reduction shares: for `s = leaves/2, …, 1`, threads
/// `t < s` run `step(t, t + s)`, then the whole block barriers. The caller has stored its
/// leaves and barriered first; `step` is one level (one tree, or two side by side, as
/// `attn_res` does). `leaves` comes from the kernel — the Phase 0c rule.
///
/// # Safety
/// Every thread of the block calls this (it barriers), `tid` is this thread's index, and
/// `step` touches only LDS the caller owns.
#[inline(always)]
pub unsafe fn tree(tid: usize, leaves: usize, mut step: impl FnMut(usize, usize)) {
    let mut s = leaves >> 1;
    while s > 0 {
        if tid < s {
            step(tid, tid + s);
        }
        sync_threads();
        s >>= 1;
    }
}

/// This lane's index in its wave (`__lane_id` on wave32).
#[inline(always)]
pub fn lane_id() -> u32 {
    mbcnt_lo(u32::MAX, 0)
}

/// `__shfl_down(v, delta, WAVE)`: a lane whose source would leave the wave reads itself.
#[inline(always)]
fn shfl_down(v: f32, delta: u32) -> f32 {
    let me = lane_id();
    let src = if (me & (WAVE as u32 - 1)) + delta >= WAVE as u32 {
        me
    } else {
        me + delta
    };
    wave_shuffle(v, src)
}

/// `__shfl_xor(v, mask, WAVE)`.
#[inline(always)]
fn shfl_xor(v: f32, mask: u32) -> f32 {
    let me = lane_id();
    let src = me ^ mask;
    let src = if src >= ((me + WAVE as u32) & !(WAVE as u32 - 1)) {
        me
    } else {
        src
    };
    wave_shuffle(v, src)
}

/// Sum a wave's values into lane 0, fixed shuffle-down order (deterministic). Other lanes
/// hold partials; they are reproduced bit for bit too.
#[inline(always)]
pub fn wave_sum(v: f32) -> f32 {
    [16, 8, 4, 2, 1]
        .into_iter()
        .fold(v, |v, o| v + shfl_down(v, o))
}

/// Max across a wave, on EVERY lane. A butterfly is legal because `maxnum` is exactly
/// associative. REQUIRES A FULL WAVE: a shuffle from an inactive lane is undefined.
#[inline(always)]
pub fn wave_max(v: f32) -> f32 {
    [16, 8, 4, 2, 1]
        .into_iter()
        .fold(v, |v, o| maxnum(v, shfl_xor(v, o)))
}

/// Block-wide sum of `v` into every thread through the caller's LDS `red` (`block_dim`
/// floats); `tid` is this thread's index and `block_dim` the launch width.
///
/// REQUIRES a power-of-two `block_dim` (the halving ladder drops the odd element otherwise)
/// and that `block_dim` IS the launch width. Re-associated against the oracle's sequential
/// fold. The trailing barrier makes `red` reusable for a second reduction — `recurrent.rs`
/// reduces twice back to back — so the count is 2 + log2(block_dim), as in C++.
///
/// # Safety
/// `red` must point at `block_dim` floats of workgroup memory, and every thread of the block
/// must call this uniformly.
#[inline(always)]
pub unsafe fn block_sum_lds(v: f32, red: *mut f32, tid: usize, block_dim: usize) -> f32 {
    // SAFETY: `tree` touches `t + s < block_dim` only for `t < s <= block_dim/2`; the
    // barriers order every write before the reads that follow it.
    unsafe {
        *red.add(tid) = v;
        sync_threads();
        tree(tid, block_dim, |t, s| *red.add(t) += *red.add(s));
        let t = *red;
        sync_threads();
        t
    }
}
