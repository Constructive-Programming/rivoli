//! inkling's KV-cache append scatter on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::kv_scatter` — **no C twin** (owner direction 2026-11-24): a new-model
//! kernel validated against the first-party reference, not the C→Rust bit-identity harness. The
//! scatter is a pure data movement (no arithmetic), so the comparison is BITWISE.
//!
//! `ORACLE_VALIDATED` in `kernel_coverage.rs` waives the C-twin row; this suite is the oracle it cites.
//!
//! Run on the device under `flock /var/run/sys-gpu.lock`, single-threaded (`--test-threads=1`).
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_backend::hip::launch_kv_scatter;
use rivoli_oracles::inkling::kv_scatter as cpu_kv_scatter;

mod common;
use common::{assert_bits, assert_guard, dev, f32b, fill, launch_to_f32, ok};

/// One geometry's operands. `Copy` so the guard test can vary one field with struct-update syntax.
#[derive(Clone, Copy)]
struct ScatterCase<'a> {
    src: &'a [f32],
    head_dim: usize,
    ctx: usize,
    pos: usize,
}

impl<'a> ScatterCase<'a> {
    fn new(src: &'a [f32], head_dim: usize, ctx: usize, pos: usize) -> Self {
        Self {
            src,
            head_dim,
            ctx,
            pos,
        }
    }
}

/// One geometry's device launch. The destination is the zeroed buffer `launch_to_f32` provides;
/// its length is `src.len()·ctx` (= `kv_heads·ctx·head_dim` when the geometry is valid), which
/// avoids dividing by `head_dim` so a `head_dim = 0` guard case reaches the launcher rather than
/// panicking here.
fn scatter_launch(c: &ScatterCase) -> anyhow::Result<Vec<f32>> {
    let sb = dev(&f32b(c.src));
    let dst_n = c.src.len() * c.ctx;
    // SAFETY: `src` `n` f32, the destination `kv_heads·ctx·head_dim` f32; both outlive the stream
    // `launch_to_f32` syncs on. The launcher guards the geometry before device access.
    launch_to_f32(dst_n, |op, st| unsafe {
        launch_kv_scatter(
            sb.ptr() as *const f32,
            op,
            c.pos,
            c.head_dim,
            c.ctx,
            c.src.len(),
            st,
        )
    })
}

/// The scatter matches the CPU oracle bitwise over ragged head counts, widths, and positions.
#[test]
fn kv_scatter_matches_the_cpu_oracle() {
    for &(heads, hd, ctx, pos) in &[
        (1usize, 4, 3, 0),
        (8, 128, 5, 2),
        (3, 16, 7, 6),
        (2, 8, 4, 1),
    ] {
        let src = fill(heads * hd, 0xfeed + heads as u64, 1.0);
        let c = ScatterCase::new(&src, hd, ctx, pos);
        let want = cpu_kv_scatter(&src, hd, ctx, pos);
        let got = ok(scatter_launch(&c), "gpu kv_scatter");
        assert_bits(
            &want,
            &got,
            &format!("kv_scatter h={heads} d={hd} ctx={ctx} pos={pos}"),
        );
    }
}

/// Each head's token lands contiguous in its own block: a head/lane transposition would put head
/// 1's pair in head 0's block. Two heads, width 2, ctx 4, pos 2 — a deinterleave trap.
#[test]
fn kv_scatter_keeps_heads_in_their_own_blocks() {
    let src = [5.0, 6.0, 7.0, 8.0];
    let c = ScatterCase::new(&src, 2, 4, 2);
    let got = ok(scatter_launch(&c), "kv_scatter blocks");
    assert_eq!(
        got,
        vec![
            0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0
        ]
    );
}

/// The geometry refusals (all 1001): a zero extent, a position past the context, and a source that
/// is not a whole number of heads.
#[test]
fn kv_scatter_geometry_is_guarded() {
    let src = fill(8, 1, 1.0);
    let base = ScatterCase::new(&src, 2, 3, 1);
    assert_guard(
        scatter_launch(&ScatterCase {
            head_dim: 0,
            ..base
        }),
        Some(1001),
        "head_dim = 0",
    );
    assert_guard(
        scatter_launch(&ScatterCase { ctx: 0, ..base }),
        Some(1001),
        "ctx = 0",
    );
    assert_guard(
        scatter_launch(&ScatterCase { pos: 3, ..base }),
        Some(1001),
        "pos = ctx",
    );
    assert_guard(
        scatter_launch(&ScatterCase {
            head_dim: 3,
            ..base
        }),
        Some(1001),
        "n % head_dim != 0",
    );
    assert_guard(scatter_launch(&base), None, "valid geometry");
}
