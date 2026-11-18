//! Wave/block geometry, the workgroup barrier, and the three reductions over them. Port of
//! `reduce.hpp` at tag `archive/hip-kernels`.
//!
//! No `fmuladd`: the only float operators are `+` and `maxnum` over shuffles.
//!
//! **No helper here reads the block size from the dispatch packet** (Phase 0c, 2026-11-18:
//! that read costs ~1.1–1.2 µs per launch at tens of workgroups). `block_sum_lds` takes it
//! from the kernel, which knows it as a constant or a kernel argument.

use core::arch::amdgpu::mbcnt_lo;

use super::llvm::{maxnum, wave_shuffle};

/// gfx1151 is native wave32.
pub const WAVE: usize = 32;
/// block = 256 threads = 8 waves = 8 output rows.
pub const ROWS_PER_BLOCK: usize = 8;

/// `__syncthreads()`: wait, barrier, invalidate L0. Rust has no workgroup-scope fence
/// (`atomic::fence` is system scope and adds a `buffer_gl1_inv`), so it is spelled in asm;
/// without `nomem` the asm is also a compiler memory barrier.
///
/// The waits are hipcc's workgroup RELEASE in WGP mode, which these kernels run in: LDS
/// (`lgkmcnt`), vector loads (`vmcnt`) AND stores (`vscnt`). hipcc emits
/// `s_waitcnt_vscnt null, 0x0` before the barrier after a global store (probe, 2026-11-18)
/// and drops each wait when its counter is provably zero; this asm cannot see the counters,
/// so it always waits — a no-op when nothing is outstanding, and never weaker than hipcc.
#[inline(always)]
pub fn sync_threads() {
    // SAFETY: a workgroup barrier. Every caller must reach it uniformly across the block,
    // the same obligation `__syncthreads` carries.
    unsafe {
        core::arch::asm!(
            "s_waitcnt vmcnt(0) lgkmcnt(0)",
            "s_waitcnt_vscnt null, 0x0",
            "s_barrier",
            "buffer_gl0_inv"
        )
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
/// fold. The trailing barrier makes `red` reusable for a second reduction — `recurrent.hip`
/// reduces twice back to back — so the count is 2 + log2(block_dim), as in C++.
///
/// # Safety
/// `red` must point at `block_dim` floats of workgroup memory, and every thread of the block
/// must call this uniformly.
#[inline(always)]
pub unsafe fn block_sum_lds(v: f32, red: *mut f32, tid: usize, block_dim: usize) -> f32 {
    // SAFETY: `tid < block_dim` and `tid + o < block_dim` for `tid < o <= block_dim/2`; the
    // barriers order every write before the reads that follow it.
    unsafe {
        *red.add(tid) = v;
        sync_threads();
        let mut o = block_dim >> 1;
        while o > 0 {
            if tid < o {
                *red.add(tid) += *red.add(tid + o);
            }
            sync_threads();
            o >>= 1;
        }
        let t = *red;
        sync_threads();
        t
    }
}
