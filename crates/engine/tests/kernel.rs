//! The quantized GEMV family vs their CPU oracles in quant.rs — fp8, int8, VQ, int4 — and the
//! bit-identity claim two of them make about batched rows. Compiles to nothing without rocm.
//!
//! **Split on 2026-08-15, then again on 2026-08-16 — by COHESION, not by size alone.** This
//! file had reached 2263 lines and 79 functions over nine unrelated kernel families, and
//! CodeScene scored it 8.03 on "Low Cohesion" and on that function count. Three groups have
//! come away whole and left:
//!
//! * `kernel_moe.rs` — the MoE expert-range oracles, which share `ExpertDesc` construction,
//!   the `moe_reference` fusion and the fixed-point accumulator with each other and with
//!   nothing here;
//! * `kernel_vector.rs` — `argmax`/`vadd`, `index_topk`, `swiglu`, `rope_interleave` and
//!   `vq_encode`, each of which is one launcher against one inline host reference and shares
//!   no scaffolding at all;
//! * `kernel_attend.rs` — the two fp8 kv_b MLA projections and the flash attend, which share
//!   `MlaIo`, `AttIo` and `kvb_fp8` with each other and with nothing here. It was the last of
//!   the nine families still sharing this file with GEMV, and once it left, `kernel.rs` sat at
//!   804 lines against the build's 800-line soft cap — no new test drove it there, the cap
//!   alone did.
//!
//! What stayed is what shares scaffolding: `GemvIo` bundles the four-buffer operand list every
//! quantized GEMV takes, `assert_out` (now in `common`, since `kernel_attend.rs` needs it too)
//! reads a device destination back against the CPU oracle, and
//! `batched_rows_are_bit_identical_to_single_rows` drives the two GEMV arms that stayed through
//! `single_vs_batched`. Every body is where it was, with its comments.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use anyhow::Result;
use rivoli_artifact::quant::{
    RowScaledW, VQ_DIM, VQ_K, VqW, matvec_i4, matvec_vq, quant_i4, quant_vq,
};
use rivoli_backend::hip::{
    device_sync, launch_gemv_f32, launch_gemv_fp8, launch_gemv_i4, launch_gemv_i8, launch_gemv_vq,
};

mod common;
// Two `use` statements rather than one shared bracket: a single `DeviceBuf, Lcg, <lowercase
// helpers…>` list is the exact shape `kernel_vector.rs`'s own `use common::{…}` takes — same
// two capitalized types first (rustfmt sorts uppercase before lowercase), same bracketed run —
// and `build.rs`'s jscpd gate matched the two AS ONE FRAGMENT on that structure. There is
// nothing shared in substance: this file and that one name mostly different helpers. Splitting
// the types from the functions breaks the token run without renaming anything.
use common::{DeviceBuf, Lcg};
use common::{
    assert_guard, assert_out, assert_rel, assert_rows, back, dev, f16b, f32b, f32v, gemv_fp8_case,
    i8_weights, u16b, want_i8, zeros,
};

// `dev` lived here until 2026-08-06, "backend-typed, so it stays here rather than in
// `common`: this is `DeviceBuf` under HIP and `Buf` under Vulkan." There is one backend,
// so the argument is gone and so is the copy — see `common`'s header.

// ---------------------------------------------------------------------------
// Launch wrappers. Each spells its launcher's argument list ONCE.
//
// All the launchers take raw device addresses, so the only thing these add is the `.ptr()`
// extraction — and that is exactly why they cannot move to `common`: `DeviceBuf` is the HIP
// half of the type pair the two oracle files exist to keep apart.
// ---------------------------------------------------------------------------

/// The four buffers a quantized GEMV takes: activations, packed weights, block scales, and
/// the destination.
///
/// `gemv_fp8` and `gemv_i8` take the same four in the same order and differ only in whether
/// a `block` follows — written out twice, that is two four-line preambles that must stay in
/// step with each other and with the launcher.
struct GemvIo<'a> {
    x: &'a DeviceBuf,
    p: &'a DeviceBuf,
    s: &'a DeviceBuf,
    y: &'a mut DeviceBuf,
}

impl<'a> GemvIo<'a> {
    fn new(x: &'a DeviceBuf, p: &'a DeviceBuf, s: &'a DeviceBuf, y: &'a mut DeviceBuf) -> Self {
        Self { x, p, s, y }
    }

    /// The four device addresses, in launcher order. Consuming, because the destination is
    /// a unique borrow and the address outlives the reborrow that produced it.
    fn ptrs(self) -> (*const f32, *const u8, *const f32, *mut f32) {
        let (x, s) = (self.x.ptr() as *const f32, self.s.ptr() as *const f32);
        (x, self.p.ptr(), s, self.y.ptr_mut() as *mut f32)
    }
}

/// `gemv_fp8`, single-row. Returns the `Result` rather than expecting it: the guard test
/// hands it dims it requires to be REJECTED, and that arm must reach its assertion rather
/// than panic.
///
/// `nrow` is fixed at 1 rather than taken, like [`gemv_i4`] beside it: both oracle arms here
/// are single-row, and the batched claim is made against `gemv_i8`/`gemv_f32` in
/// `batched_rows_are_bit_identical_to_single_rows` instead.
fn gemv_fp8(io: GemvIo<'_>, o_dim: usize, i_dim: usize, block: usize) -> Result<()> {
    let (x, p, s, y) = io.ptrs();
    // SAFETY: the buffers are borrowed for the call, so they outlive it; every caller sizes
    // them for the dims it passes, and a rejected dim never reaches a dereference.
    unsafe { launch_gemv_fp8(x, p, s, o_dim, i_dim, block, 1, y) }
}

/// `gemv_i8` — lm_head's projection.
fn gemv_i8(io: GemvIo<'_>, o_dim: usize, i_dim: usize, nrow: usize) {
    let (x, p, s, y) = io.ptrs();
    // SAFETY: as `gemv_fp8`; `x` holds `nrow` rows of i_dim and `y` `nrow` rows of o_dim.
    unsafe { launch_gemv_i8(x, p, s, o_dim, i_dim, nrow, y) }.expect("gemv_i8");
}

/// `gemv_i4` — the standalone `dot_i4_wave` microbench kernel. Single-row only.
///
/// Takes [`GemvIo`] like its two neighbours even though its `scale` is per GROUP rather
/// than per row or per tile: the four BUFFERS are the same four in the same order, and the
/// point of the shared operand is that a transposed pair moves both a launcher and its
/// oracle at once. What differs is the sizes the caller allocates, which is its own job.
fn gemv_i4(io: GemvIo<'_>, o_dim: usize, i_dim: usize) {
    let (x, p, s, y) = io.ptrs();
    // SAFETY: as `gemv_fp8`; `packed` is `o_dim · i4_row_bytes(i_dim)` and `scale` is
    // `o_dim · i4_groups(i_dim)` f32.
    unsafe { launch_gemv_i4(x, p, s, o_dim, i_dim, y) }.expect("gemv_i4");
}

/// The comparison the two GEMV arms of `batched_rows_are_bit_identical_to_single_rows`
/// run: each row of `xs` dispatched alone, then both as one two-row batch. Returns
/// `(batched, [single0, single1])`.
///
/// `launch` enqueues and nothing else; the join and the readback are here, so an arm cannot
/// accidentally read one row before the other has landed. The singles go first, in index
/// order, and the batch last — that order is the one under test.
fn single_vs_batched(
    xs: &[Vec<f32>; 2],
    o_dim: usize,
    launch: impl Fn(&DeviceBuf, &mut DeviceBuf, usize),
) -> (Vec<f32>, [Vec<f32>; 2]) {
    let run = |x: &[f32], rows: usize| -> Vec<f32> {
        let xb = dev(&f32b(x));
        let mut yb = dev(&vec![0u8; rows * o_dim * 4]);
        launch(&xb, &mut yb, rows);
        device_sync().expect("sync");
        f32v(&yb.copy_out().expect("out"))
    };
    let single: [Vec<f32>; 2] = std::array::from_fn(|i| run(&xs[i], 1));
    // Row 0 first — the batch layout every `nrow = 2` launch here takes.
    let both: Vec<f32> = xs[0].iter().chain(&xs[1]).copied().collect();
    (run(&both, 2), single)
}

#[test]
fn gemv_fp8_matches_oracle() {
    // block-scaled fp8 GEMV vs matvec_fp8. Two shapes: a short reduction (plain
    // wave-per-row path) and a long one (i_dim ≥ 4096 → the split-K path
    // launch_gemv_fp8 dispatches to for the o_proj-class projections).
    // Shape 3 is DELIBERATELY ragged: 130 rows at block 128 give a PARTIAL final scale
    // row, which shapes 1-2 (exact multiples) never produce. That is what forced the
    // scale allocation below from `(o_dim / block)` to `o_dim.div_ceil(block)` — the old
    // expression sizes one row for 130 and the oracle reads off the end of it.
    // Shapes 4-5 pin the NARROW tiles the guard test below insists on ACCEPTING: the dot
    // reads four columns per dword and one block scale per quad, so a tile narrower than
    // a quad must fall back to the per-column path. Both values are covered because they
    // index differently — block=1 gives bsh=0 and sc_cols == i_dim (a scale per column,
    // no shift at all), block=2 puts two columns of each quad in the right tile and two
    // in the next. Accepting them in the launcher while silently mis-scaling most of
    // every quad, which is what shipped, is the worse of the two failures.
    for (o_dim, i_dim, block, label) in [
        (256usize, 512usize, 128usize, "gemv_fp8"),
        (128, 16384, 128, "gemv_fp8_splitk"),
        (130, 8192, 128, "gemv_fp8_splitk ragged o_dim"),
        (64, 512, 2, "gemv_fp8 block=2"),
        (64, 512, 1, "gemv_fp8 block=1"),
    ] {
        let mut r = Lcg(0xF8 ^ i_dim as u64 ^ (block as u64) << 20);
        // The scale grid is `gemv_fp8_case`'s own, div_ceil on BOTH axes, mirroring the
        // kernel — see its doc for why this stopped being the caller's to state.
        let (packed, scale, x, want) = gemv_fp8_case(&mut r, o_dim, i_dim, block);

        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = dev(&vec![0u8; o_dim * 4]);
        let io = GemvIo::new(&xb, &pb, &sb, &mut yb);
        gemv_fp8(io, o_dim, i_dim, block).expect("launch");
        device_sync().expect("sync");
        assert_out(&want, &yb, label);
    }
}

#[test]
fn gemv_fp8_rejects_non_power_of_two_block() {
    // The fp8 dot indexes the block scale with a SHIFT (`blk_shift`), which is only
    // the same index as `i / block` for a power-of-two tile. The launcher must reject
    // anything else rather than compute a silently wrong scale index.
    //
    // Asserts the guard CODE, not just is_err(): a test that accepted any error would
    // still pass if someone replaced the power-of-two test with `block != 128`, or if
    // the dim guard (1001) started swallowing these first. Both i_dim arms are covered
    // because `rivoli_gemv_fp8` dispatches to the split-K launcher at i_dim >= 4096
    // (GEMV_SPLITK_MIN_IDIM) and that launcher carries its own copy of the guard.
    let o_dim = 64usize;
    for i_dim in [512usize, 8192] {
        let packed = dev(&vec![0u8; o_dim * i_dim]);
        let scale = dev(&f32b(&vec![1.0f32; o_dim * i_dim]));
        let x = dev(&f32b(&vec![0.0f32; i_dim]));
        let mut y = dev(&vec![0u8; o_dim * 4]);
        // 1 is a power of two (bsh = 0, `i >> 0` == `i / 1`), so it must be ACCEPTED.
        for (block, want) in [
            (128usize, None),
            (96, Some(1003)),
            (127, Some(1003)),
            (1, None),
        ] {
            let io = GemvIo::new(&x, &packed, &scale, &mut y);
            let r = gemv_fp8(io, o_dim, i_dim, block);
            assert_guard(r, want, &format!("i_dim={i_dim} block={block}"));
        }
        device_sync().expect("sync");
    }
}

#[test]
fn gemv_i8_matches_oracle() {
    // lm_head's GEMV — the last op before argmax, and until now the only quantized
    // kernel in the engine with NO oracle at all.
    //
    // Swept over dims because the kernel is now a dword-quad fast path plus a scalar
    // tail, and lm_head alone (6144) only ever exercises the fast path. Each dim below
    // reaches a different arm: 6148 leaves a 4-column tail, 100 is shorter than one
    // WAVE*4 step so it is tail-only, and 6143 gives an odd row stride that
    // de-aligns every row but the first and must fall out of the quad path entirely.
    for (o_dim, i_dim) in [(512usize, 6144usize), (96, 6148), (96, 100), (96, 6143)] {
        let mut r = Lcg(0x18);
        let (packed, scale) = i8_weights(&mut r, o_dim, i_dim);
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = dev(&vec![0u8; o_dim * 4]);
        let want = want_i8(&x, &packed, &scale, [o_dim, i_dim]);
        gemv_i8(GemvIo::new(&xb, &pb, &sb, &mut yb), o_dim, i_dim, 1);
        device_sync().expect("sync");
        assert_out(&want, &yb, &format!("gemv_i8 o={o_dim} i={i_dim}"));
    }
}

#[test]
fn gemv_vq_matches_oracle() {
    let mut r = Lcg(0x53);
    let (o_dim, i_dim) = (2048usize, 512usize);
    let codebook: Vec<f32> = (0..VQ_K * VQ_DIM).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let (indices, scales) = quant_vq(&w, o_dim, i_dim, &codebook);
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();

    let mut want = vec![0.0f32; o_dim];
    matvec_vq(
        &mut want,
        &x,
        VqW::new(&indices, &scales, &codebook),
        [o_dim, i_dim],
    );

    let (xb, ib, sb, cb) = (
        dev(&f32b(&x)),
        dev(&indices),
        dev(&u16b(&scales)),
        dev(&f16b(&codebook)),
    );
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    // SAFETY: all five buffers are device-resident, sized for these dims, and live until
    // the sync below.
    unsafe {
        launch_gemv_vq(
            xb.ptr() as *const f32,
            ib.ptr(),
            sb.ptr() as *const u16,
            cb.ptr() as *const u16,
            o_dim,
            i_dim,
            yb.ptr_mut() as *mut f32,
        )
    }
    .expect("launch");
    device_sync().expect("sync");
    assert_out(&want, &yb, "gemv_vq");
}

/// **The batched forward's correctness gate.** Every kernel that takes `nrow` must be
/// BIT-IDENTICAL, per row, to running the same input as a single row — not close, equal.
///
/// That is what lets speculative decode claim to emit exactly the bytes greedy sequential
/// decode would. Row 0 of a verify pass IS the real token: if these kernels are
/// bit-identical at row 0, and the MoE union cannot perturb row 0 (unselected experts
/// carry weight 0 and are skipped), then the speculative engine has no freedom to differ.
/// A tolerance here would let that claim rot into "usually the same text".
///
/// Row 1 is checked the same way against its own single-row run, which is what catches
/// the other failure mode: a kernel that batches correctly but leaks row 0's `x` into
/// row 1 (a missing `r * stride`) would pass a row-0-only test.
///
/// **The claim covers the MoE pair too, and its arm is
/// `kernel_moe.rs::batched_moe_rows_are_bit_identical_to_single_rows`** — it left with the
/// expert-range machinery on 2026-08-15 and is a `#[test]` in its own right there. **The MLA
/// pair's arm left the same way on 2026-08-16, as
/// `kernel_attend.rs::batched_mla_rows_are_bit_identical_to_single_rows`**, when the
/// MLA/attend suites left for their own file. Neither departure weakened the claim — each arm
/// still asserts its own kernel and names it in the failure, and the MoE arm still asserts the
/// fixed-point accumulator as u64, which is the strictest form of the claim anything in this
/// suite makes.
#[test]
fn batched_rows_are_bit_identical_to_single_rows() {
    let mut r = Lcg(0xba7c);
    batched_gemv_f32(&mut r);
    batched_gemv_i8(&mut r);
}

/// gemv_f32 — the MoE router gate.
fn batched_gemv_f32(r: &mut Lcg) {
    let (o_dim, i_dim) = (64usize, 128usize);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let xs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..i_dim).map(|_| r.f()).collect());
    let wb = dev(&f32b(&w));
    let (got, single) = single_vs_batched(&xs, o_dim, |xb, yb, nrow| {
        // SAFETY: `x` holds `nrow` rows of i_dim and `y` `nrow` rows of o_dim; both
        // nrow 1 and nrow 2 are instantiated.
        unsafe {
            launch_gemv_f32(
                xb.ptr() as *const f32,
                wb.ptr() as *const f32,
                o_dim,
                i_dim,
                nrow,
                yb.ptr_mut() as *mut f32,
                std::ptr::null_mut(),
            )
        }
        .expect("gemv_f32");
    });
    assert_rows(&got, &single, o_dim, "gemv_f32");
}

/// gemv_i8 (lm_head). i_dim % 4 == 0 so both rows take the dword-quad path.
fn batched_gemv_i8(r: &mut Lcg) {
    let (o_dim, i_dim) = (96usize, 260usize);
    let (packed, scale) = i8_weights(r, o_dim, i_dim);
    let xs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..i_dim).map(|_| r.f()).collect());
    let (pb, sb) = (dev(&packed), dev(&f32b(&scale)));
    let (got, single) = single_vs_batched(&xs, o_dim, |xb, yb, nrow| {
        gemv_i8(GemvIo::new(xb, &pb, &sb, yb), o_dim, i_dim, nrow)
    });
    assert_rows(&got, &single, o_dim, "gemv_i8");
}

/// `gemv_i4` — the group-scaled int4 dot, against `quant.rs::matvec_i4`.
///
/// **This kernel has no caller in `src/`.** It exists so `dot_i4_wave`'s decode throughput
/// can be measured against `gemv_vq`/`gemv_fp8` free of the routing confound, and
/// `examples/dot_bench.rs` is its only user — which checked it against this same oracle
/// and then threw the result away in a `println!`. An example is not a test: nothing runs
/// it, and `cargo test` never noticed that. The kernel is the same `dot_i4_wave` the MoE
/// int4 experts run, so a change to it silently changes what the benchmark measures.
///
/// Not bitwise: the dot is a wave reduction, so its summation order is not the oracle's and
/// cannot be made to be. Nor `assert_close` — measured here, the disagreement is 6.68e-6
/// against that shared tolerance's 1.79e-2, which is 2677x of headroom, and its absolute
/// floor alone is 13.6% of the smallest output (0.1315).
#[test]
fn gemv_i4_matches_oracle() {
    let (o_dim, i_dim) = (64usize, 256usize);
    let mut r = Lcg(0x1D40);
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let (packed, scale) = quant_i4(&w, o_dim, i_dim);
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    let rw = RowScaledW::new(&packed, &scale);
    matvec_i4(&mut want, &x, rw, [o_dim, i_dim]);

    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = zeros(o_dim * 4);
    gemv_i4(GemvIo::new(&xb, &pb, &sb, &mut yb), o_dim, i_dim);
    // 1e-5 relative: 25x above the measured 4e-7, and still far tighter than a wrong
    // group-scale index or a dropped `-8` nibble bias, both of which are O(1).
    assert_rel(&want, &f32v(&back(&yb)), "gemv_i4", 1e-5);
}
