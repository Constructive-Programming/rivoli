//! inkling's rel-bias distance gather on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::rel_bias_gather` — **no C twin** (owner direction 2026-11-24). The
//! gather is a pure data movement plus a distance mask (no arithmetic), so the comparison is BITWISE.
//!
//! `ORACLE_VALIDATED` in `kernel_coverage.rs` waives the C-twin row; this suite is the oracle it cites.
//!
//! Run on the device under `flock /var/run/sys-gpu.lock`, single-threaded (`--test-threads=1`).
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{assert_bits, assert_guard, dev, f32b, fill, launch_to_f32, ok};
use rivoli_backend::hip::launch_rel_bias_gather;
use rivoli_oracles::inkling::rel_bias_gather as cpu_rel_bias_gather;
/// One geometry's operands. `heads` is carried explicitly (not derived by division) so a
/// `rel_extent = 0` guard case reaches the launcher instead of dividing by zero here.
#[derive(Clone, Copy)]
struct GatherCase<'a> {
    rel_logits: &'a [f32],
    heads: usize,
    rel_extent: usize,
    pos: usize,
    start: usize,
    n: usize,
}

impl<'a> GatherCase<'a> {
    /// `geom` is `(rel_extent, pos, start, n)` — the four extents in the order the launcher takes
    /// them, grouped so a `pos`/`start` swap is one visible literal rather than two bare `usize`.
    fn new(rel_logits: &'a [f32], heads: usize, geom: (usize, usize, usize, usize)) -> Self {
        Self {
            rel_logits,
            heads,
            rel_extent: geom.0,
            pos: geom.1,
            start: geom.2,
            n: geom.3,
        }
    }
}

/// One geometry's device launch; the destination is the zeroed buffer `launch_to_f32` provides.
fn gather_launch(c: &GatherCase) -> anyhow::Result<Vec<f32>> {
    let bias_n = c.heads * c.n;
    let rb = dev(&f32b(c.rel_logits));
    // SAFETY: `rel_logits` `heads·rel_extent` f32, the destination `heads·n` f32; both outlive the
    // stream `launch_to_f32` syncs on. The launcher guards the geometry before device access.
    launch_to_f32(bias_n.max(1), |op, st| unsafe {
        launch_rel_bias_gather(
            rb.ptr() as *const f32,
            op,
            c.pos,
            c.start,
            c.n,
            c.heads,
            c.rel_extent,
            st,
        )
    })
}

/// The gather matches the CPU oracle bitwise over ragged head counts, extents, positions, and
/// window starts.
#[test]
fn rel_bias_gather_matches_the_cpu_oracle() {
    for &(heads, ext, pos, start, n) in &[
        (1usize, 4, 0, 0, 1),
        (24, 16, 7, 0, 8),
        (3, 8, 15, 0, 16),
        (2, 5, 2, 0, 5),
        // A sliding window past its start: slot t is absolute key `start + t`.
        (2, 8, 10, 8, 3),
        (4, 16, 40, 30, 8),
    ] {
        let rel_logits = fill(heads * ext, 0xabc + heads as u64, 1.0);
        let c = GatherCase::new(&rel_logits, heads, (ext, pos, start, n));
        let want = cpu_rel_bias_gather(&rel_logits, ext, pos, (start, n));
        let got = ok(gather_launch(&c), "gpu rel_bias_gather");
        assert_bits(
            &want,
            &got,
            &format!("rel_bias h={heads} ext={ext} pos={pos} start={start} n={n}"),
        );
    }
}

/// Distances past the query or beyond the extent are masked to zero, and the kept distances read
/// the profile backwards: pos 2, n 4, extent 3 → indices 2,1,0 then a masked −1.
#[test]
fn rel_bias_gather_masks_out_of_range_distances() {
    let rel_logits = [10.0, 11.0, 12.0, 20.0, 21.0, 22.0];
    let c = GatherCase::new(&rel_logits, 2, (3, 2, 0, 4));
    let got = ok(gather_launch(&c), "rel_bias mask");
    assert_eq!(got, vec![12.0, 11.0, 10.0, 0.0, 22.0, 21.0, 20.0, 0.0]);
}

/// **The `start` field is load-bearing.** A sliding window at `pos = 1500` over `start = 477`,
/// `n = 1024`, `rel_extent = 1024`: slot 0 is key 477 (distance 1023 → `rel_logits[h][1023]`) and
/// slot 1023 is key 1500 (distance 0 → `rel_logits[h][0]`). A kernel that ignored `start` and read
/// the slot as the position would mask the whole window to zero at this `pos`.
#[test]
fn rel_bias_gather_reads_the_sliding_window_start() {
    let (heads, ext, pos, start, n) = (2usize, 1024usize, 1500usize, 477usize, 1024usize);
    let rel_logits = fill(heads * ext, 0x51de, 1.0);
    let c = GatherCase::new(&rel_logits, heads, (ext, pos, start, n));
    let got = ok(gather_launch(&c), "rel_bias sliding");
    assert_bits(
        &cpu_rel_bias_gather(&rel_logits, ext, pos, (start, n)),
        &got,
        "sliding window == oracle",
    );
    for h in 0..heads {
        let row = &rel_logits[h * ext..(h + 1) * ext];
        let bias = &got[h * n..(h + 1) * n];
        assert_eq!(
            bias[0], row[1023],
            "head {h}: slot 0 is the oldest distance"
        );
        assert_eq!(
            bias[1023], row[0],
            "head {h}: slot 1023 is the query itself"
        );
    }
}

/// The geometry refusals (all 1001): a zero head count, a zero extent, and a zero key count.
#[test]
fn rel_bias_gather_geometry_is_guarded() {
    let rel_logits = fill(8, 1, 1.0);
    let base = GatherCase::new(&rel_logits, 2, (4, 1, 0, 3));
    for (why, case) in [
        ("heads = 0", GatherCase { heads: 0, ..base }),
        (
            "rel_extent = 0",
            GatherCase {
                rel_extent: 0,
                ..base
            },
        ),
        ("n = 0", GatherCase { n: 0, ..base }),
    ] {
        assert_guard(gather_launch(&case), Some(1001), why);
    }
    assert_guard(gather_launch(&base), None, "valid geometry");
}
