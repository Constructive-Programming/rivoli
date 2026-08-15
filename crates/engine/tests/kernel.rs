//! The weight-matrix kernels vs their CPU oracles in quant.rs: the quantized GEMV family
//! (fp8, int8, VQ, int4), the two fp8 kv_b MLA projections, the flash attend, and the
//! bit-identity claim that every one of them makes about batched rows. Compiles to nothing
//! without rocm.
//!
//! **Split on 2026-08-15 — by COHESION, not by size alone.** This file had reached 2263
//! lines and 79 functions over nine unrelated kernel families, and CodeScene scored it 8.03
//! on "Low Cohesion" and on that function count. Two groups came away whole and left:
//!
//! * `kernel_moe.rs` — the MoE expert-range oracles, which share `ExpertDesc` construction,
//!   the `moe_reference` fusion and the fixed-point accumulator with each other and with
//!   nothing here;
//! * `kernel_vector.rs` — `argmax`/`vadd`, `index_topk`, `swiglu`, `rope_interleave` and
//!   `vq_encode`, each of which is one launcher against one inline host reference and shares
//!   no scaffolding at all.
//!
//! What stayed is what shares scaffolding: `GemvIo`, `MlaIo` and `AttIo` bundle the operand
//! lists, `kvb_fp8` draws the block-scaled kv_b both MLA suites use, and
//! `batched_rows_are_bit_identical_to_single_rows` drives three of these kernels through
//! `single_vs_batched`. Every body is where it was, with its comments.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use anyhow::Result;
use rivoli_artifact::quant::{
    RowScaledW, VQ_DIM, VQ_K, VqW, matvec_i4, matvec_vq, quant_i4, quant_vq,
};
use rivoli_backend::hip::{
    attend_scratch_floats, device_sync, launch_attend, launch_gemv_f32, launch_gemv_fp8,
    launch_gemv_i4, launch_gemv_i8, launch_gemv_vq, launch_mla_absorb_fp8, launch_mla_value_fp8,
};
use rivoli_core::num::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3};
use rivoli_core::routing::softmax;

mod common;
use common::{
    Att, DeviceBuf, Lcg, Mla, assert_close, assert_guard, assert_rel, assert_rows, back,
    block_scales, dev, f16b, f32b, f32v, gemv_fp8_case, i8_weights, u16b, want_i8, zeros,
};

// `dev` lived here until 2026-08-06, "backend-typed, so it stays here rather than in
// `common`: this is `DeviceBuf` under HIP and `Buf` under Vulkan." There is one backend,
// so the argument is gone and so is the copy — see `common`'s header.

/// [`assert_close`] against a device destination, read back. The join belongs to the caller —
/// several oracles enqueue two kernels and sync once, which is how the engine runs them.
fn assert_out(want: &[f32], got: &DeviceBuf, label: &str) {
    assert_close(want, &f32v(&got.copy_out().expect("out")), label);
}

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

/// The fp8 kv_b block, its block-scale grid, and the destination — everything one MLA
/// dispatch touches except the f32 input, which is the ONE argument the two kernels differ
/// in (`q` for absorb, `clat` for value).
///
/// Bundled for the same reason as [`Mla`] beside it: these launchers take eleven arguments
/// each, and two copies of that list is two chances for the pair to stop reading the same
/// kv_b — which no oracle in this file would notice, since each checks its own kernel.
struct MlaIo<'a> {
    kvb: &'a DeviceBuf,
    scale: &'a DeviceBuf,
    out: &'a mut DeviceBuf,
}

impl<'a> MlaIo<'a> {
    fn new(kvb: &'a DeviceBuf, scale: &'a DeviceBuf, out: &'a mut DeviceBuf) -> Self {
        Self { kvb, scale, out }
    }

    /// The three device addresses, in launcher order.
    fn ptrs(&mut self) -> (*const u8, *const f32, *mut f32) {
        let sc = self.scale.ptr() as *const f32;
        (self.kvb.ptr(), sc, self.out.ptr_mut() as *mut f32)
    }
}

/// `mla_absorb_fp8`. `Result`-returning for the same reason as [`gemv_fp8`].
fn mla_absorb(q: &DeviceBuf, io: &mut MlaIo<'_>, m: Mla, nrow: usize) -> Result<()> {
    let (kvb, sc, out) = io.ptrs();
    let q = q.ptr() as *const f32;
    // SAFETY: as `gemv_fp8`; kv_b is [h·(nope+vh), kvl] bytes and `out` `nrow` rows of h·kvl.
    unsafe {
        launch_mla_absorb_fp8(
            q, kvb, sc, m.h, m.qh, m.nope, m.vh, m.kvl, m.block, nrow, out,
        )
    }
}

/// `mla_value_fp8` — the same kv_b, without `qh`, into `ctx`.
fn mla_value(clat: &DeviceBuf, io: &mut MlaIo<'_>, m: Mla, nrow: usize) -> Result<()> {
    let (kvb, sc, out) = io.ptrs();
    let clat = clat.ptr() as *const f32;
    // SAFETY: as `mla_absorb`; `out` holds `nrow` rows of h·vh.
    unsafe { launch_mla_value_fp8(clat, kvb, sc, m.h, m.nope, m.vh, m.kvl, m.block, nrow, out) }
}

/// `mla_latent_attend`'s five inputs: the two query halves, the fp8 latent cache and its
/// per-128 block scales, and the bf16 roped key cache.
///
/// Bundled because the launcher takes fourteen arguments, five of them interchangeable raw
/// addresses, and the guard test and the oracle both spell the list — two copies that a
/// transposed pair would move together while both stayed green.
struct AttIo<'a> {
    qabs: &'a DeviceBuf,
    qrope: &'a DeviceBuf,
    lc8: &'a DeviceBuf,
    lscale: &'a DeviceBuf,
    rc: &'a DeviceBuf,
}

/// `mla_latent_attend`, dense (no row gather). `Result`-returning for the same reason as
/// [`gemv_fp8`]: the guard test hands it a kvl it requires to be REJECTED.
fn attend(io: &AttIo<'_>, d: Att, clat: &mut DeviceBuf, part: &mut DeviceBuf) -> Result<()> {
    // SAFETY: every buffer is sized for the largest kvl its caller tries, `part` for
    // `attend_scratch_floats`; the rejected cases never dereference them at all.
    unsafe {
        launch_attend(
            io.qabs.ptr() as *const f32,
            io.qrope.ptr() as *const f32,
            io.lc8.ptr(),
            io.lscale.ptr() as *const f32,
            io.rc.ptr() as *const u16,
            std::ptr::null(), // dense (no row gather)
            d.h,
            d.nr,
            d.kvl,
            d.rope,
            d.n_blocks,
            d.scale,
            clat.ptr_mut() as *mut f32,
            part.ptr_mut() as *mut f32,
        )
    }
}

/// fp8 kv_b bytes for `rows × kvl`, and the block-scale grid over them, drawn in that order.
fn kvb_fp8(r: &mut Lcg, rows: usize, kvl: usize, block: usize) -> (Vec<u8>, Vec<f32>) {
    let packed: Vec<u8> = (0..rows * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let scale = block_scales(r, rows.div_ceil(block) * kvl.div_ceil(block));
    (packed, scale)
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
fn mla_value_rejects_ragged_kvl_but_absorb_accepts_it() {
    // The two kv_b kernels have DIFFERENT legal kvl, and the asymmetry is load width.
    // `mla_value_fp8` goes through the word-loading shared MAC, so a kvl that is not a
    // multiple of 4 leaves 3 rows in 4 misaligned for its dword load — guard 1002, the
    // same rejection src/backend/vk.rs makes. `mla_absorb_fp8` reads the ragged case a column at
    // a time and must keep accepting it, because the Vulkan absorb launcher does.
    //
    // Asserts BOTH directions. A test that only checked the rejection would still pass
    // if someone "fixed the asymmetry" by adding the guard to absorb as well, which
    // would silently drop a shape the other backend supports.
    let (h, qh, nope, vh, block) = (2usize, 32usize, 8usize, 4usize, 128usize);
    for (kvl, want) in [(64usize, None), (37, Some(1002)), (66, Some(1002))] {
        let rows = h * (nope + vh);
        let kvb = dev(&vec![0u8; rows * kvl]);
        let scale = dev(&f32b(&vec![1.0f32; rows * kvl]));
        let x = dev(&f32b(&vec![0.0f32; h * qh.max(kvl)]));
        let mut out = dev(&vec![0u8; h * kvl * 4]);
        let m = Mla::new([h, qh, nope, vh, kvl, block]);
        let mut io = MlaIo::new(&kvb, &scale, &mut out);
        assert_guard(
            mla_value(&x, &mut io, m, 1),
            want,
            &format!("mla_value kvl={kvl}"),
        );
        let abs = mla_absorb(&x, &mut io, m, 1);
        assert!(abs.is_ok(), "mla_absorb must accept kvl={kvl}: {abs:?}");
    }
    device_sync().expect("sync the accepted kv_b dispatches");
}

#[test]
fn mla_attend_rejects_unsupported_kvl() {
    // `mla_latent_attend` keeps its online accumulator in MLA_ACC_REGS*SUBW = 512
    // registers per lane and indexes it by k = (i - lane)/SUBW, so a kvl over that cap
    // — or one that is not a multiple of 128 — must be REJECTED (guard 1004), never run
    // with columns silently dropped or the wrong per-128 block scale applied.
    let (h, kvl, rope) = (8usize, 512usize, 64usize);
    let mut scratch = dev(&vec![0u8; attend_scratch_floats(h, kvl) * 4]);
    // Every buffer is sized for the largest kvl tried (1024 floats/head).
    let b = dev(&vec![0u8; h * 1024 * 4]);
    let io = AttIo {
        qabs: &b,
        qrope: &b,
        lc8: &b,
        lscale: &b,
        rc: &b,
    };
    let mut clat = dev(&vec![0u8; h * kvl * 4]);
    for (bad_kvl, want) in [(kvl, None), (640usize, Some(1004)), (160, Some(1004))] {
        let d = Att::new(h, 16, bad_kvl, rope);
        let r = attend(&io, d, &mut clat, &mut scratch);
        assert_guard(r, want, &format!("kvl={bad_kvl}"));
    }
    device_sync().expect("sync");
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

/// The two halves of a query row: the absorbed part that dots the latent cache and the roped
/// part that dots the key cache. Both are `&[f32]` and interchangeable to the compiler.
struct Qh<'a> {
    abs: &'a [f32],
    rope: &'a [f32],
}

/// Dense two-pass scalar softmax over the whole cache — the oracle the flash
/// (online) kernel must match. `lat` is the latent cache already dequantized exactly the way
/// the kernel dequantizes it (fp8-e4m3 × per-128 block scale); the roped key is bf16.
fn attend_reference(q: Qh<'_>, lat: &[f32], rc: &[u16], d: Att) -> Vec<f32> {
    let (h, kvl, rope) = (d.h, d.kvl, d.rope);
    let mut clat = vec![0.0f32; h * kvl];
    for head in 0..h {
        let qa = &q.abs[head * kvl..(head + 1) * kvl];
        let qr = &q.rope[head * rope..(head + 1) * rope];
        // ONE running sum, latent terms then roped ones in that order: `Iterator::sum` folds
        // left from 0.0, so the chain accumulates exactly what the scalar loop it replaced
        // did. `lat` is indexed rather than zipped because it is one flat `nt × kvl` slab.
        let mut scores: Vec<f32> = (0..d.nr)
            .map(|t| {
                let a = (0..kvl).map(|i| qa[i] * lat[t * kvl + i]);
                let b = rc[t * rope..(t + 1) * rope].iter().zip(qr);
                a.chain(b.map(|(&rb, x)| x * bf16_to_f32(rb))).sum::<f32>() * d.scale
            })
            .collect();
        softmax(&mut scores);
        let out = &mut clat[head * kvl..(head + 1) * kvl];
        for (t, &sc) in scores.iter().enumerate() {
            for (i, o) in out.iter_mut().enumerate() {
                *o += sc * lat[t * kvl + i];
            }
        }
    }
    clat
}

fn check_attend(seed: u64, d: Att) {
    let (h, nt, kvl, rope) = (d.h, d.nr, d.kvl, d.rope);
    assert_eq!(kvl % 128, 0, "fp8 latent cache needs kvl a multiple of 128");
    let nb = d.n_blocks;
    let mut r = Lcg(seed);
    let qabs: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| r.f()).collect();
    // Latent as fp8 bytes + positive per-128 block scales; key round-tripped through
    // bf16. Kernel and reference decode identical bits — the only difference under
    // test is flash (online) vs two-pass softmax.
    let lc8: Vec<u8> = (0..nt * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let lscale: Vec<f32> = (0..nt * nb).map(|_| (r.f() * 0.1).abs() + 0.01).collect();
    let rc: Vec<u16> = (0..nt * rope).map(|_| f32_to_bf16(r.f())).collect();
    // The kernel's own dequant, hoisted out of the reference's inner loops so both passes
    // read one slab rather than recomputing a code-times-block-scale product per element.
    let lat: Vec<f32> = (0..nt * kvl)
        .map(|n| e4m3_to_f32(lc8[n]) * lscale[n / kvl * nb + (n % kvl) / 128])
        .collect();
    let q = Qh {
        abs: &qabs,
        rope: &qrope,
    };
    let want = attend_reference(q, &lat, &rc, d);

    let (qab, qrb) = (dev(&f32b(&qabs)), dev(&f32b(&qrope)));
    let (lcb, lsb, rcb) = (dev(&lc8), dev(&f32b(&lscale)), dev(&u16b(&rc)));
    let mut clatb = dev(&vec![0u8; h * kvl * 4]);
    // Non-null worst-case scratch → the multi-split + combine path is what's checked.
    let mut partb = dev(&vec![0u8; attend_scratch_floats(h, kvl) * 4]);
    let io = AttIo {
        qabs: &qab,
        qrope: &qrb,
        lc8: &lcb,
        lscale: &lsb,
        rc: &rcb,
    };
    attend(&io, d, &mut clatb, &mut partb).expect("launch attend");
    device_sync().expect("sync");
    assert_out(&want, &clatb, "mla_attend");
}

#[test]
fn mla_attend_glm_dims() {
    // GLM MLA (kv_lora=512, qk_rope=64); H=64 = 8 full HB blocks, nt spans many
    // TILE steps → the split-KV planner picks n_splits>1.
    check_attend(0x0a5e_1102, Att::new(64, 300, 512, 64));
}

/// Long context, where the split PLANNER changes regime — the case every other attend
/// test misses. `mla_attend_glm_dims` runs nt=300 (ntiles=19), so `by_work` binds and
/// `n_splits` is 4; nothing above exercises the two regimes the engine actually reaches:
/// `by_grid` binding (nt ≳ 640) and the `MLA_MAX_SPLITS` clamp (nt ≳ 1024).
///
/// Added 2026-08-02 after an `HB` 8→16 sweep measured **2.08× on the kernel** and then
/// failed the perplexity gate at **+0.108 nats** — damage that is exactly zero below
/// nt≈640, symmetric noise to ~4600, and catastrophic past it (max |dNLL| 16 nats). The
/// whole suite passed at HB=16 because its longest case stops at 300. A knob whose only
/// effect is on the split plan needs a case where the split plan is non-trivial.
///
/// nt=4608 is the smallest size reproducing the blow-up; the f32 reference is O(nt·kvl·h)
/// and runs in about a second, which is why this is a plain case and not `#[ignore]`d.
#[test]
fn mla_attend_long_context_split_regimes() {
    // just past by_grid binding
    check_attend(0x5f11_7bad, Att::new(64, 704, 512, 64));
    // past the MLA_MAX_SPLITS clamp
    check_attend(0xc1a3_9ed0, Att::new(64, 4608, 512, 64));
}

#[test]
fn mla_attend_edges() {
    // Single cached token stresses the online init (m=-inf on the first token);
    // H=20 → a partial second HB block (4 active + 4 inactive lanes). kvl=128 = the
    // smallest legal fp8-block latent (n_blocks=1).
    check_attend(0x00c0_ffee, Att::new(4, 1, 128, 8));
    check_attend(0xb10c_c0de, Att::new(20, 130, 512, 64));
}

/// MLA absorb + value kernels vs an f32 reference on the dequantized kv_b. kv_b is
/// fp8-e4m3 block-scaled: [H·(nope+vh) rows, kvl cols].
///
/// `mla_absorb_fp8` has one fast path and ONE fallback branch with two independent
/// triggers, and the GLM shape only ever reaches the fast path. The fast path gives a
/// thread an i-QUAD: four contiguous columns read as one dword sharing one block scale.
/// It needs `kvl % 4 == 0` (or the rows are unaligned for the dword load) and
/// `block >= 4` (or a quad straddles scale tiles). Each trigger, and the boundary value
/// `block == 4` itself, needs its own shape — an off-by-one in that predicate is
/// invisible to a suite that only tries 2 and 128.
#[test]
fn mla_fp8_matches_reference() {
    // The six shapes as a TABLE in `Mla::new`'s order, `[h, qh, nope, vh, kvl, block]`. Six
    // `Mla { .. }` literals instead is one line per field per shape under rustfmt, and three of
    // these six then share four identical lines — they differ in ONE dim at a time, which is the
    // point of the sweep. `build.rs`'s duplication gate reported exactly that on 2026-08-15.
    for (seed, dims) in [
        (0x77u64, [4, 128, 128, 64, 256, 128]), // GLM-like: i-quad fast path
        (0x78, [3, 64, 48, 32, 37, 128]),       // kvl % 4 != 0 -> scalar fallback
        (0x79, [2, 32, 16, 8, 64, 2]),          // block < 4 -> scalar fallback
        (0x7a, [2, 32, 16, 8, 64, 4]),          // block == 4: fast/fallback BOUNDARY
        (0x7b, [2, 32, 16, 8, 68, 64]),         // fast path, PARTIAL last scale tile
        (0x7c, [2, 16, 8, 4, 2, 128]),          // kvl < 4: the fallback's i0+k clamp
    ] {
        check_mla_fp8(seed, Mla::new(dims));
    }
}

/// kv_b as both kernels see it, `rows × kvl` of `e4m3 · block-scale` — the exact dequant, so
/// the oracles below measure the arithmetic and not a second decode of the same bytes.
///
/// The scale ROW index is clamped: `rows` need not be a multiple of `block`, and the partial
/// final tile has no row of its own.
fn kvb_dequant(packed: &[u8], scale: &[f32], m: Mla) -> Vec<f32> {
    let (rows, kvl, block) = (m.rows(), m.kvl, m.block);
    let (sc_rows, sc_cols) = (rows.div_ceil(block), kvl.div_ceil(block));
    (0..rows * kvl)
        .map(|n| {
            let (row, i) = (n / kvl, n % kvl);
            e4m3_to_f32(packed[n]) * scale[(row / block).min(sc_rows - 1) * sc_cols + i / block]
        })
        .collect()
}

/// `out[n] = Σ_{k < len} f(n, k)` — the shell both kv_b oracles are. They differ only in which
/// index runs the dot and how it addresses kv_b, and `build.rs`'s jscpd gate reported the
/// second copy of the shell. `sum` folds left from 0.0, which is the accumulation the scalar
/// `acc +=` loops these replaced performed.
fn per_output_sum(rows: usize, len: usize, f: impl Fn(usize, usize) -> f32) -> Vec<f32> {
    (0..rows)
        .map(|n| (0..len).map(|k| f(n, k)).sum::<f32>())
        .collect()
}

/// `qabs[head][i] = Σ_d q[head·qh+d]·kvb[rbase+d][i]`, the absorb oracle. Its own function so
/// neither reference's loop nest sits inside the other's.
fn mla_absorb_reference(kvb: &[f32], q: &[f32], m: Mla) -> Vec<f32> {
    let (qh, nope, vh, kvl) = (m.qh, m.nope, m.vh, m.kvl);
    per_output_sum(m.h * kvl, nope, |n, d| {
        let (head, i) = (n / kvl, n % kvl);
        q[head * qh + d] * kvb[(head * (nope + vh) + d) * kvl + i]
    })
}

/// `ctx[head][j] = Σ_i clat[head][i]·kvb[rbase+nope+j][i]`, the value oracle.
fn mla_value_reference(kvb: &[f32], clat: &[f32], m: Mla) -> Vec<f32> {
    let (nope, vh, kvl) = (m.nope, m.vh, m.kvl);
    per_output_sum(m.h * vh, kvl, |n, i| {
        let (head, j) = (n / vh, n % vh);
        clat[head * kvl + i] * kvb[(head * (nope + vh) + nope + j) * kvl + i]
    })
}

/// Checks absorb always, and value only where value is DEFINED — `mla_value_fp8` requires
/// `kvl % 4 == 0` (guard 1002) and absorb does not; see
/// `mla_value_rejects_ragged_kvl_but_absorb_accepts_it`.
fn check_mla_fp8(seed: u64, m: Mla) {
    let (h, vh, kvl, block) = (m.h, m.vh, m.kvl, m.block);
    let value_defined = kvl.is_multiple_of(4);
    let mut r = Lcg(seed);
    let (packed, scale) = kvb_fp8(&mut r, m.rows(), kvl, block);
    let kvb = kvb_dequant(&packed, &scale, m);
    let q: Vec<f32> = (0..h * m.qh).map(|_| r.f()).collect();
    let clat: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();
    let want_abs = mla_absorb_reference(&kvb, &q, m);
    let want_val = mla_value_reference(&kvb, &clat, m);

    let (kb, sb) = (dev(&packed), dev(&f32b(&scale)));
    let (qb, clb) = (dev(&f32b(&q)), dev(&f32b(&clat)));
    let mut absb = dev(&vec![0u8; h * kvl * 4]);
    let mut valb = dev(&vec![0u8; h * vh * 4]);
    // Both enqueued before either sync, as the engine's attention step runs them.
    let mut aio = MlaIo::new(&kb, &sb, &mut absb);
    mla_absorb(&qb, &mut aio, m, 1).expect("launch absorb");
    if value_defined {
        let mut vio = MlaIo::new(&kb, &sb, &mut valb);
        mla_value(&clb, &mut vio, m, 1).expect("launch value");
    }
    device_sync().expect("sync");
    assert_out(
        &want_abs,
        &absb,
        &format!("mla_absorb kvl{kvl} block{block}"),
    );
    if value_defined {
        assert_out(
            &want_val,
            &valb,
            &format!("mla_value kvl{kvl} block{block}"),
        );
    }
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
/// expert-range machinery on 2026-08-15 and is a `#[test]` in its own right there. The
/// sentence about the union above is about that arm; nothing here weakened, and the arm still
/// asserts the fixed-point accumulator as u64, which is the strictest form of the claim
/// anything in this suite makes.
#[test]
fn batched_rows_are_bit_identical_to_single_rows() {
    let mut r = Lcg(0xba7c);
    batched_gemv_f32(&mut r);
    batched_gemv_i8(&mut r);
    batched_mla_kvb(&mut r);
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

/// mla_absorb_fp8 / mla_value_fp8 (both through kv_b). kvl % 4 == 0 and block >= 4, so this
/// exercises the quad path both kernels actually run.
fn batched_mla_kvb(r: &mut Lcg) {
    let m = Mla::new([3, 20, 12, 8, 16, 4]);
    let (h, vh, kvl) = (m.h, m.vh, m.kvl);
    let (packed, scale) = kvb_fp8(r, m.rows(), kvl, m.block);
    let (kb, sb) = (dev(&packed), dev(&f32b(&scale)));
    let qs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..h * m.qh).map(|_| r.f()).collect());
    let cs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..h * kvl).map(|_| r.f()).collect());

    // Launch only — the sync stays at the call sites so both kernels are still in
    // flight together before either result is read, which is how the engine's own
    // attention step runs them and how the nrow=1 arm below has always run them.
    let absorb = |qb: &DeviceBuf, ab: &mut DeviceBuf, nrow: usize| {
        let mut io = MlaIo::new(&kb, &sb, ab);
        mla_absorb(qb, &mut io, m, nrow).expect("absorb");
    };
    let value = |clb: &DeviceBuf, vb: &mut DeviceBuf, nrow: usize| {
        let mut io = MlaIo::new(&kb, &sb, vb);
        mla_value(clb, &mut io, m, nrow).expect("value");
    };

    let mut abs1 = Vec::new();
    let mut val1 = Vec::new();
    for i in 0..2 {
        let (qb, clb) = (dev(&f32b(&qs[i])), dev(&f32b(&cs[i])));
        let mut ab = dev(&vec![0u8; h * kvl * 4]);
        let mut vb = dev(&vec![0u8; h * vh * 4]);
        absorb(&qb, &mut ab, 1);
        value(&clb, &mut vb, 1);
        device_sync().expect("sync");
        abs1.push(f32v(&ab.copy_out().expect("out")));
        val1.push(f32v(&vb.copy_out().expect("out")));
    }
    let qboth: Vec<f32> = qs[0].iter().chain(&qs[1]).copied().collect();
    let cboth: Vec<f32> = cs[0].iter().chain(&cs[1]).copied().collect();
    let (qb, clb) = (dev(&f32b(&qboth)), dev(&f32b(&cboth)));
    let mut ab = dev(&vec![0u8; 2 * h * kvl * 4]);
    let mut vb = dev(&vec![0u8; 2 * h * vh * 4]);
    absorb(&qb, &mut ab, 2);
    value(&clb, &mut vb, 2);
    device_sync().expect("sync");
    assert_rows(
        &f32v(&ab.copy_out().expect("out")),
        &abs1,
        h * kvl,
        "mla_absorb",
    );
    assert_rows(
        &f32v(&vb.copy_out().expect("out")),
        &val1,
        h * vh,
        "mla_value",
    );
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
