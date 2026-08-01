//! GPU kernels vs their CPU oracles in quant.rs. Compiles to nothing without rocm.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli::memory::device::DeviceBuf;
use rivoli::backend::gpustream::HipStream;
use rivoli::backend::hip::{
    ExpertDesc, attend_scratch_floats, device_sync, launch_argmax, launch_attend, launch_gemv_fp8,
    launch_gemv_i8, launch_gemv_vq, launch_index_topk, launch_mla_absorb_fp8, launch_mla_value_fp8,
    launch_moe_acc_drain, launch_moe_expert_range, launch_moe_expert_range_i4, launch_vadd,
};
use rivoli::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3, silu, softmax};
use rivoli::artifact::quant::{VQ_DIM, VQ_K, matvec_vq, quant_vq};

mod common;
use common::{Lcg, assert_close, block_scales, f16b, f32b, f32v, gemv_fp8_case, u16b, want_i8};

/// Upload `b` to a fresh device buffer. Backend-typed, so it stays here rather than in
/// `common`: this is `DeviceBuf` under HIP and `Buf` under Vulkan.
fn dev(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len()).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}

/// A launcher result against an expected guard code: `None` must be ACCEPTED, `Some(n)`
/// rejected with `n` somewhere in the message.
///
/// The CODE is asserted rather than merely `is_err`, and that is the whole value of these
/// tests: one that accepted any error would still pass if someone replaced a power-of-two
/// check with `block != 128`, or if an unrelated dimension guard started swallowing the
/// case first.
fn assert_guard<T: std::fmt::Debug>(r: anyhow::Result<T>, want: Option<u32>, what: &str) {
    match want {
        None => assert!(r.is_ok(), "{what}: {r:?}"),
        Some(code) => {
            let msg = format!("{:#}", r.expect_err("expected a guard rejection"));
            assert!(msg.contains(&code.to_string()), "{what}: want guard {code}, got {msg:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Launch wrappers. Each spells its launcher's argument list ONCE.
//
// All the launchers take raw device addresses, so the only thing these add is the `.ptr()`
// extraction — and that is exactly why they cannot move to `common`: `DeviceBuf` is the HIP
// half of the type pair the two oracle files exist to keep apart.
// ---------------------------------------------------------------------------

/// `gemv_fp8`. Returns the `Result` rather than expecting it: the guard test hands it dims
/// it requires to be REJECTED, and that arm must reach its assertion rather than panic.
#[allow(clippy::too_many_arguments)]
fn gemv_fp8(x: &DeviceBuf, p: &DeviceBuf, s: &DeviceBuf, y: &mut DeviceBuf,
            o_dim: usize, i_dim: usize, block: usize, nrow: usize) -> anyhow::Result<()> {
    // SAFETY: the buffers are borrowed for the call, so they outlive it; every caller sizes
    // them for the dims it passes, and a rejected dim never reaches a dereference.
    unsafe {
        launch_gemv_fp8(x.ptr() as *const f32, p.ptr(), s.ptr() as *const f32,
                        o_dim, i_dim, block, nrow, y.ptr_mut() as *mut f32)
    }
}

/// `gemv_i8` — lm_head's projection.
fn gemv_i8(x: &DeviceBuf, p: &DeviceBuf, s: &DeviceBuf, y: &mut DeviceBuf,
           o_dim: usize, i_dim: usize, nrow: usize) {
    // SAFETY: as `gemv_fp8`; `x` holds `nrow` rows of i_dim and `y` `nrow` rows of o_dim.
    unsafe {
        launch_gemv_i8(x.ptr() as *const f32, p.ptr(), s.ptr() as *const f32,
                       o_dim, i_dim, nrow, y.ptr_mut() as *mut f32)
    }
    .expect("gemv_i8");
}

/// `mla_absorb_fp8`. `Result`-returning for the same reason as [`gemv_fp8`].
#[allow(clippy::too_many_arguments)]
fn mla_absorb(q: &DeviceBuf, kvb: &DeviceBuf, scale: &DeviceBuf, out: &mut DeviceBuf,
              h: usize, qh: usize, nope: usize, vh: usize, kvl: usize, block: usize,
              nrow: usize) -> anyhow::Result<()> {
    // SAFETY: as `gemv_fp8`; kv_b is [h·(nope+vh), kvl] bytes and `out` `nrow` rows of h·kvl.
    unsafe {
        launch_mla_absorb_fp8(q.ptr() as *const f32, kvb.ptr(), scale.ptr() as *const f32,
                              h, qh, nope, vh, kvl, block, nrow, out.ptr_mut() as *mut f32)
    }
}

/// `mla_value_fp8` — the same kv_b, without `qh`, into `ctx`.
#[allow(clippy::too_many_arguments)]
fn mla_value(clat: &DeviceBuf, kvb: &DeviceBuf, scale: &DeviceBuf, out: &mut DeviceBuf,
             h: usize, nope: usize, vh: usize, kvl: usize, block: usize,
             nrow: usize) -> anyhow::Result<()> {
    // SAFETY: as `mla_absorb`; `out` holds `nrow` rows of h·vh.
    unsafe {
        launch_mla_value_fp8(clat.ptr() as *const f32, kvb.ptr(), scale.ptr() as *const f32,
                             h, nope, vh, kvl, block, nrow, out.ptr_mut() as *mut f32)
    }
}

/// `moe_expert_range_i4` over experts `[e_start, e_start+e_count)`.
#[allow(clippy::too_many_arguments)]
fn expert_range_i4(x: &DeviceBuf, descs: &DeviceBuf, w: &DeviceBuf, h: &mut DeviceBuf,
                   acc: &mut DeviceBuf, hidden: usize, inter: usize, e_start: usize,
                   e_count: usize, nrow: usize, stream: &HipStream) {
    // SAFETY: `x` is `nrow` rows of [hidden], `w` is [e_count·nrow], `h` [e_count·nrow·inter]
    // and `acc` `nrow` rows of [hidden] u64; the stream is live for the call.
    unsafe {
        launch_moe_expert_range_i4(
            x.ptr() as *const f32, hidden, inter, e_start, e_count,
            descs.ptr() as *const ExpertDesc, w.ptr() as *const f32,
            h.ptr_mut() as *mut f32, acc.ptr_mut() as *mut u64, nrow, stream.raw(),
        )
    }
    .expect("moe_expert_range_i4");
}

/// One `moe_acc_drain` over row `row` of the accumulator.
///
/// `row` is what lets a batched arm drain its rows with the same launch the single-row arms
/// use — the drain itself is always single-row.
fn drain(out: &mut DeviceBuf, acc: &mut DeviceBuf, row: usize, hidden: usize, stream: &HipStream) {
    // SAFETY: `row` is inside both buffers, which every caller sizes for it; the stream is
    // live for the call.
    unsafe {
        launch_moe_acc_drain(
            out.ptr_mut().add(row * hidden * 4) as *mut f32,
            acc.ptr_mut().add(row * hidden * 8) as *mut u64,
            hidden, 1, 1.0, stream.raw(),
        )
    }
    .expect("moe_acc_drain");
}

/// Upload `b`, park it in `bufs`, and hand back its device address.
///
/// An `ExpertDesc` is a struct of raw pointers and owns nothing, so something has to keep
/// the six spans alive for the length of the dispatch; `bufs` is that something.
fn push(b: Vec<u8>, bufs: &mut Vec<DeviceBuf>) -> *const u8 {
    bufs.push(dev(&b));
    bufs.last().expect("just pushed").ptr()
}

/// The descriptor ARRAY on device — the addresses themselves, uploaded verbatim.
fn desc_buf(descs: &[ExpertDesc]) -> DeviceBuf {
    // SAFETY: `ExpertDesc` is plain pointers, and the span is exactly the slice's own bytes.
    dev(unsafe {
        std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(descs))
    })
}

/// The three MoE destination buffers for `nrow` token rows: per-expert `h` staging, the
/// fixed-point accumulator, and the f32 output.
///
/// ONE u64 accumulator row per token, not `e` partial rows; the output starts at zero
/// because the drain ADDS into it — it is the residual add.
fn moe_bufs(e: usize, nrow: usize, hidden: usize, inter: usize) -> (DeviceBuf, DeviceBuf, DeviceBuf) {
    (
        dev(&vec![0u8; e * nrow * inter * 4]),
        dev(&vec![0u8; nrow * hidden * 8]),
        dev(&vec![0u8; nrow * hidden * 4]),
    )
}

/// `Σ_e w[e]·down(silu(gate·x) ⊙ up·x)` — the MoE reference both format tests check against.
///
/// `mv(out, in, ex, p, o_dim, i_dim)` is the caller's matvec for projection `p` (0 gate,
/// 1 up, 2 down) of expert `ex`. Only that step differs between int4 and VQ; the fusion
/// around it is the same arithmetic in the same order, and a second copy of it is a second
/// place for the accumulation order to drift from the kernel's.
fn moe_reference(
    x: &[f32],
    w: &[f32],
    hidden: usize,
    inter: usize,
    mv: impl Fn(&mut [f32], &[f32], usize, usize, usize, usize),
) -> Vec<f32> {
    let mut want = vec![0.0f32; hidden];
    for (ex, we) in w.iter().enumerate() {
        let mut g = vec![0.0f32; inter];
        let mut u = vec![0.0f32; inter];
        mv(&mut g, x, ex, 0, inter, hidden);
        mv(&mut u, x, ex, 1, inter, hidden);
        let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
        let mut down = vec![0.0f32; hidden];
        mv(&mut down, &h, ex, 2, hidden, inter);
        for (o, d) in down.iter().enumerate() {
            want[o] += we * d;
        }
    }
    want
}

/// `e` experts' worth of per-projection quantised weights, `one(r, p, o_dim, i_dim)` doing
/// each projection.
///
/// The DRAW ORDER is what this exists to hold fixed: expert-major, then gate/up/down, every
/// weight matrix drawn from `r` before the next one starts. The two format tests differ only
/// in what `one` does with those weights, and a seed has to mean the same data in both.
fn encode_experts<S>(
    r: &mut Lcg,
    e: usize,
    dims: &[(usize, usize)],
    one: impl Fn(&mut Lcg, usize, usize, usize) -> (Vec<u8>, Vec<S>),
) -> Vec<Vec<(Vec<u8>, Vec<S>)>> {
    (0..e)
        .map(|_| {
            dims.iter()
                .enumerate()
                .map(|(p, &(o_dim, i_dim))| one(r, p, o_dim, i_dim))
                .collect()
        })
        .collect()
}

/// `argmax_reduce` into the two output words, idx then val.
fn argmax(logits: &DeviceBuf, n: usize, idx: &mut DeviceBuf, val: &mut DeviceBuf) {
    // SAFETY: `logits` is n live f32 and each output buffer is one live word.
    unsafe {
        launch_argmax(logits.ptr() as *const f32, n,
                      idx.ptr_mut() as *mut i32, val.ptr_mut() as *mut f32)
    }
    .expect("launch argmax");
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
    let mut both: Vec<f32> = xs[0].clone();
    both.extend_from_slice(&xs[1]);
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
        let sc_cols = i_dim.div_ceil(block); // div_ceil on BOTH axes, mirroring the kernel
        let (packed, scale, x, want) =
            gemv_fp8_case(&mut r, o_dim, i_dim, block, o_dim.div_ceil(block) * sc_cols);

        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = dev(&vec![0u8; o_dim * 4]);
        gemv_fp8(&xb, &pb, &sb, &mut yb, o_dim, i_dim, block, 1).expect("launch");
        device_sync().expect("sync");
        assert_close(&want, &f32v(&yb.copy_out().expect("out")), label);
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
        for (block, want) in [(128usize, None), (96, Some(1003)), (127, Some(1003)), (1, None)] {
            let r = gemv_fp8(&x, &packed, &scale, &mut y, o_dim, i_dim, block, 1);
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
        let val = mla_value(&x, &kvb, &scale, &mut out, h, nope, vh, kvl, block, 1);
        assert_guard(val, want, &format!("mla_value kvl={kvl}"));
        let abs = mla_absorb(&x, &kvb, &scale, &mut out, h, qh, nope, vh, kvl, block, 1);
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
    let big = dev(&vec![0u8; h * 1024 * 4]);
    let mut clat = dev(&vec![0u8; h * kvl * 4]);
    for (bad_kvl, want) in [(kvl, None), (640usize, Some(1004)), (160, Some(1004))] {
        // SAFETY: every buffer is sized for the largest kvl tried (1024 floats/head);
        // the rejected cases never dereference them at all.
        let r = unsafe {
            launch_attend(
                big.ptr() as *const f32,
                big.ptr() as *const f32,
                big.ptr(),
                big.ptr() as *const f32,
                big.ptr() as *const u16,
                std::ptr::null(),
                h,
                16,
                bad_kvl,
                rope,
                bad_kvl / 128,
                1.0,
                clat.ptr_mut() as *mut f32,
                scratch.ptr_mut() as *mut f32,
            )
        };
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
        let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| (r.f() * 127.0) as i8 as u8).collect();
        let scale: Vec<f32> = (0..o_dim).map(|_| (r.f() * 0.01).abs() + 1e-4).collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = dev(&vec![0u8; o_dim * 4]);
        let want = want_i8(&x, &packed, &scale, o_dim, i_dim);
        gemv_i8(&xb, &pb, &sb, &mut yb, o_dim, i_dim, 1);
        device_sync().expect("sync");
        let got = f32v(&yb.copy_out().expect("out"));
        assert_close(&want, &got, &format!("gemv_i8 o={o_dim} i={i_dim}"));
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
    matvec_vq(&mut want, &x, &indices, &scales, &codebook, o_dim, i_dim);

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
        launch_gemv_vq(xb.ptr() as *const f32, ib.ptr(), sb.ptr() as *const u16,
                       cb.ptr() as *const u16, o_dim, i_dim, yb.ptr_mut() as *mut f32)
    }
    .expect("launch");
    device_sync().expect("sync");
    assert_close(&want, &f32v(&yb.copy_out().expect("out")), "gemv_vq");
}

/// argmax_reduce: max value wins, ties → lowest index, NaN never wins. Plus vadd
/// as a residual-add smoke check (both fwd.hip glue).
#[test]
fn fwd_argmax_and_vadd() {
    // argmax: a plateau (tie → lowest index) with a NaN that must lose.
    let mut logits = vec![0.1f32, 0.5, 0.5, f32::NAN, 0.3, 0.5, -1.0];
    let want_idx = 1i32; // first 0.5
    let lb = dev(&f32b(&logits));
    let mut ib = dev(&[0u8; 4]);
    let mut vb = dev(&[0u8; 4]);
    argmax(&lb, logits.len(), &mut ib, &mut vb);
    device_sync().expect("sync");
    let got_idx = i32::from_le_bytes(
        ib.copy_out().expect("out")[..4]
            .try_into()
            .expect("4 bytes"),
    );
    let got_val = f32::from_le_bytes(
        vb.copy_out().expect("out")[..4]
            .try_into()
            .expect("4 bytes"),
    );
    assert_eq!(got_idx, want_idx, "argmax idx");
    assert_eq!(got_val, 0.5, "argmax val");

    // vadd: x += y, elementwise.
    let y: Vec<f32> = logits
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    for l in logits.iter_mut() {
        if l.is_nan() {
            *l = 0.0;
        }
    }
    let mut xb = dev(&f32b(&logits));
    let yb = dev(&f32b(&y));
    unsafe {
        launch_vadd(
            xb.ptr_mut() as *mut f32,
            yb.ptr() as *const f32,
            logits.len(),
        )
        .expect("vadd");
    }
    device_sync().expect("sync");
    let got = f32v(&xb.copy_out().expect("out"));
    let want: Vec<f32> = logits.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_close(&want, &got, "vadd");
}

/// Dense two-pass scalar softmax over the whole cache — the oracle the flash
/// (online) kernel must match. Latent is fp8-e4m3 × per-128 block scale (the exact
/// dequant the kernel does); the roped key is bf16.
#[allow(clippy::too_many_arguments)]
fn attend_reference(
    qabs: &[f32], qrope: &[f32], lc8: &[u8], lscale: &[f32], rc: &[u16],
    h: usize, nt: usize, kvl: usize, rope: usize, n_blocks: usize, scale: f32,
) -> Vec<f32> {
    let lat = |t: usize, i: usize| e4m3_to_f32(lc8[t * kvl + i]) * lscale[t * n_blocks + i / 128];
    let mut clat = vec![0.0f32; h * kvl];
    let mut scores = vec![0.0f32; nt];
    for head in 0..h {
        let qa = &qabs[head * kvl..(head + 1) * kvl];
        let qr = &qrope[head * rope..(head + 1) * rope];
        for (t, sc) in scores.iter_mut().enumerate() {
            let rrow = &rc[t * rope..(t + 1) * rope];
            let mut a = 0.0f32;
            // `lat(t, i)` needs the index, so a range loop is the clear form here.
            #[allow(clippy::needless_range_loop)]
            for i in 0..kvl {
                a += qa[i] * lat(t, i);
            }
            for (d, &rb) in rrow.iter().enumerate() {
                a += qr[d] * bf16_to_f32(rb);
            }
            *sc = a * scale;
        }
        softmax(&mut scores);
        let out = &mut clat[head * kvl..(head + 1) * kvl];
        for (t, &sc) in scores.iter().enumerate() {
            for (i, o) in out.iter_mut().enumerate() {
                *o += sc * lat(t, i);
            }
        }
    }
    clat
}

fn check_attend(seed: u64, h: usize, nt: usize, kvl: usize, rope: usize) {
    assert_eq!(kvl % 128, 0, "fp8 latent cache needs kvl a multiple of 128");
    let n_blocks = kvl / 128;
    let mut r = Lcg(seed);
    let qabs: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| r.f()).collect();
    // Latent as fp8 bytes + positive per-128 block scales; key round-tripped through
    // bf16. Kernel and reference decode identical bits — the only difference under
    // test is flash (online) vs two-pass softmax.
    let lc8: Vec<u8> = (0..nt * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let lscale: Vec<f32> = (0..nt * n_blocks)
        .map(|_| (r.f() * 0.1).abs() + 0.01)
        .collect();
    let rc: Vec<u16> = (0..nt * rope).map(|_| f32_to_bf16(r.f())).collect();
    let scale = 1.0 / ((kvl + rope) as f32).sqrt();
    let want = attend_reference(
        &qabs, &qrope, &lc8, &lscale, &rc, h, nt, kvl, rope, n_blocks, scale,
    );

    let (qab, qrb) = (dev(&f32b(&qabs)), dev(&f32b(&qrope)));
    let (lcb, lsb, rcb) = (dev(&lc8), dev(&f32b(&lscale)), dev(&u16b(&rc)));
    let mut clatb = dev(&vec![0u8; h * kvl * 4]);
    // Non-null worst-case scratch → the multi-split + combine path is what's checked.
    let mut partb = dev(&vec![0u8; attend_scratch_floats(h, kvl) * 4]);
    unsafe {
        launch_attend(
            qab.ptr() as *const f32,
            qrb.ptr() as *const f32,
            lcb.ptr(),
            lsb.ptr() as *const f32,
            rcb.ptr() as *const u16,
            std::ptr::null(), // dense (no row gather)
            h,
            nt,
            kvl,
            rope,
            n_blocks,
            scale,
            clatb.ptr_mut() as *mut f32,
            partb.ptr_mut() as *mut f32,
        )
        .expect("launch attend");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&clatb.copy_out().expect("out")), "mla_attend");
}

#[test]
fn mla_attend_glm_dims() {
    // GLM MLA (kv_lora=512, qk_rope=64); H=64 = 8 full HB blocks, nt spans many
    // TILE steps → the split-KV planner picks n_splits>1.
    check_attend(0x0a5e_1102, 64, 300, 512, 64);
}

#[test]
fn mla_attend_edges() {
    // Single cached token stresses the online init (m=-inf on the first token);
    // H=20 → a partial second HB block (4 active + 4 inactive lanes). kvl=128 = the
    // smallest legal fp8-block latent (n_blocks=1).
    check_attend(0x00c0_ffee, 4, 1, 128, 8);
    check_attend(0xb10c_c0de, 20, 130, 512, 64);
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
    //          seed  h  qh nope vh  kvl block
    check_mla_fp8(0x77, 4, 128, 128, 64, 256, 128); // GLM-like: the i-quad fast path
    check_mla_fp8(0x78, 3, 64, 48, 32, 37, 128); // kvl % 4 != 0 → scalar fallback
    check_mla_fp8(0x79, 2, 32, 16, 8, 64, 2); // block < 4 → scalar fallback
    check_mla_fp8(0x7a, 2, 32, 16, 8, 64, 4); // block == 4: the fast/fallback BOUNDARY
    check_mla_fp8(0x7b, 2, 32, 16, 8, 68, 64); // fast path over a PARTIAL last scale tile
    check_mla_fp8(0x7c, 2, 16, 8, 4, 2, 128); // kvl < 4: the fallback's i0+k clamp
}

/// Checks absorb always, and value only where value is DEFINED — `mla_value_fp8` requires
/// `kvl % 4 == 0` (guard 1002) and absorb does not; see
/// `mla_value_rejects_ragged_kvl_but_absorb_accepts_it`.
fn check_mla_fp8(seed: u64, h: usize, qh: usize, nope: usize, vh: usize, kvl: usize, block: usize) {
    let value_defined = kvl.is_multiple_of(4);
    let mut r = Lcg(seed);
    let rows = h * (nope + vh);
    let sc_cols = kvl.div_ceil(block);
    let (packed, scale) = kvb_fp8(&mut r, rows, kvl, block);
    // reference reads the exact dequant the kernels see: kvb[row][i] = e4m3·block-scale.
    let sc_rows = rows.div_ceil(block);
    let kvbf = |row: usize, i: usize| -> f32 {
        e4m3_to_f32(packed[row * kvl + i])
            * scale[(row / block).min(sc_rows - 1) * sc_cols + i / block]
    };
    let q: Vec<f32> = (0..h * qh).map(|_| r.f()).collect();
    let clat: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();

    // absorb: qabs[head][i] = Σ_d q[head·qh+d]·kvb[rbase+d][i]
    let mut want_abs = vec![0.0f32; h * kvl];
    for head in 0..h {
        let rbase = head * (nope + vh);
        for i in 0..kvl {
            let mut acc = 0.0f32;
            for d in 0..nope {
                acc += q[head * qh + d] * kvbf(rbase + d, i);
            }
            want_abs[head * kvl + i] = acc;
        }
    }
    // value: ctx[head][j] = Σ_i clat[head][i]·kvb[rbase+nope+j][i]
    let mut want_val = vec![0.0f32; h * vh];
    for head in 0..h {
        let rbase = head * (nope + vh);
        for j in 0..vh {
            let mut acc = 0.0f32;
            for i in 0..kvl {
                acc += clat[head * kvl + i] * kvbf(rbase + nope + j, i);
            }
            want_val[head * vh + j] = acc;
        }
    }

    let (kb, sb) = (dev(&packed), dev(&f32b(&scale)));
    let (qb, clb) = (dev(&f32b(&q)), dev(&f32b(&clat)));
    let mut absb = dev(&vec![0u8; h * kvl * 4]);
    let mut valb = dev(&vec![0u8; h * vh * 4]);
    // Both enqueued before either sync, as the engine's attention step runs them.
    mla_absorb(&qb, &kb, &sb, &mut absb, h, qh, nope, vh, kvl, block, 1).expect("launch absorb");
    if value_defined {
        mla_value(&clb, &kb, &sb, &mut valb, h, nope, vh, kvl, block, 1).expect("launch value");
    }
    device_sync().expect("sync");
    assert_close(
        &want_abs,
        &f32v(&absb.copy_out().expect("out")),
        &format!("mla_absorb kvl{kvl} block{block}"),
    );
    if value_defined {
        assert_close(
            &want_val,
            &f32v(&valb.copy_out().expect("out")),
            &format!("mla_value kvl{kvl} block{block}"),
        );
    }
}

/// Fused VQ MoE (moe_gateup_vq → moe_down_vq → moe_acc_drain), 3 per-projection
/// codebooks, vs a matvec_vq+silu reference on the same quantized bytes.
#[test]
fn moe_vq_matches_reference() {
    use rivoli::artifact::quant::{vq_expert_layout, vq_groups, vq_row_bytes};
    let mut r = Lcg(0x33);
    let (hidden, inter, e) = (128usize, 64usize, 3usize); // multi-group hidden, one-group inter
    // 3 codebooks
    let cbs: [Vec<f32>; 3] = std::array::from_fn(|_| (0..VQ_K * VQ_DIM).map(|_| r.f()).collect());
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..e).map(|_| r.f()).collect();

    // per expert per projection: quant to (indices, scales) against the right codebook
    let dims = vq_expert_layout(hidden, inter); // [(gate o,i),(up o,i),(down o,i)]
    let enc = encode_experts(&mut r, e, &dims, |r, p, o_dim, i_dim| {
        let wv: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
        quant_vq(&wv, o_dim, i_dim, &cbs[p])
    });

    let want = moe_reference(&x, &w, hidden, inter, |out, inp, ex, p, o_dim, i_dim| {
        matvec_vq(out, inp, &enc[ex][p].0, &enc[ex][p].1, &cbs[p], o_dim, i_dim)
    });

    // device: hold bufs alive; descriptors point into them
    let mut bufs: Vec<DeviceBuf> = Vec::new();
    let descs: Vec<ExpertDesc> = (0..e)
        .map(|ex| ExpertDesc {
            gate_indices: push(enc[ex][0].0.clone(), &mut bufs),
            gate_scales: push(u16b(&enc[ex][0].1), &mut bufs) as *const u16,
            up_indices: push(enc[ex][1].0.clone(), &mut bufs),
            up_scales: push(u16b(&enc[ex][1].1), &mut bufs) as *const u16,
            down_indices: push(enc[ex][2].0.clone(), &mut bufs),
            down_scales: push(u16b(&enc[ex][2].1), &mut bufs) as *const u16,
        })
        .collect();
    let descb = desc_buf(&descs);
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let (g0, g1, g2) = (
        dev(&f16b(&cbs[0])),
        dev(&f16b(&cbs[1])),
        dev(&f16b(&cbs[2])),
    );
    let (mut hbuf, mut pbuf, mut obuf) = moe_bufs(e, 1, hidden, inter);
    let _ = (vq_groups(hidden), vq_row_bytes(hidden)); // (layout used by quant/vq_expert)
    // The production path: each expert accumulates into the shared fixed-point row via a
    // single-expert range on a compute stream (exercising e_start indexing), then one
    // drain — same as the async expert stream, minus the load overlap.
    let stream = HipStream::new().expect("stream");
    // Three arms below run the same two kernels and differ only in the token rows, the
    // gate weights, the destination buffers and `nrow`. Geometry, descriptors and
    // codebooks are identical in all three, so they are captured rather than re-spelled.
    let range =
        |xb: &DeviceBuf, wb: &DeviceBuf, hb: &mut DeviceBuf, ab: &mut DeviceBuf, nrow: usize| {
            for k in 0..e {
                // SAFETY: every buffer is device-resident and sized for `nrow` rows of the
                // layout above; the stream is live.
                unsafe {
                    launch_moe_expert_range(
                        xb.ptr() as *const f32, hidden, inter, k, 1,
                        descb.ptr() as *const ExpertDesc, g0.ptr() as *const u16,
                        g1.ptr() as *const u16, g2.ptr() as *const u16, wb.ptr() as *const f32,
                        hb.ptr_mut() as *mut f32, ab.ptr_mut() as *mut u64, nrow, stream.raw(),
                    )
                }
                .expect("launch moe_expert_range");
            }
        };

    range(&xb, &wb, &mut hbuf, &mut pbuf, 1);
    drain(&mut obuf, &mut pbuf, 0, hidden, &stream);
    device_sync().expect("sync");
    assert_close(&want, &f32v(&obuf.copy_out().expect("out")), "moe_vq");
    // The drain RESETS the accumulator, and that is load-bearing rather than tidy: it is
    // why steady-state decode needs no memset before each layer's experts. A drain that
    // forgot it would pass every single-layer check here and then double-count from layer
    // 1 onward — silently, since the result stays finite and merely wrong.
    assert!(
        pbuf.copy_out().expect("acc").iter().all(|&b| b == 0),
        "moe_acc_drain left the accumulator dirty"
    );

    // --- nrow=2: the batched verify pass, against the single-row path it must reproduce.
    //
    // BIT-identical, not `assert_close`. The batched kernel is not an approximation of two
    // passes, it is the same arithmetic with the weight decode hoisted — same products,
    // same order, same `wave_sum`. Anything looser would accept a real reassociation bug,
    // and reassociation here is exactly what the fixed-point accumulator exists to make
    // impossible to hide.
    //
    // Row 1 gets a DIFFERENT x and DIFFERENT weights, one of them zero. Two rows with the
    // same input would pass even if the kernel ignored the row index entirely; the zero
    // weight covers the "this row did not route here" skip, which is the case the union
    // batching produces on every real layer.
    let x2: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let xr: Vec<f32> = x.iter().chain(&x2).copied().collect(); // x[t·hidden + i]
    let w2: Vec<f32> = (0..e).map(|k| if k == 1 { 0.0 } else { r.f() }).collect();
    // wexpert[e·2 + t] — token row fastest, matching the kernel's indexing.
    let wr: Vec<f32> = (0..e).flat_map(|k| [w[k], w2[k]]).collect();

    let xrb = dev(&f32b(&xr));
    let wrb = dev(&f32b(&wr));
    let (mut hbuf2, mut pbuf2, mut obuf2) = moe_bufs(e, 2, hidden, inter);
    range(&xrb, &wrb, &mut hbuf2, &mut pbuf2, 2);
    for t in 0..2 {
        drain(&mut obuf2, &mut pbuf2, t, hidden, &stream);
    }
    device_sync().expect("sync");
    let got2 = f32v(&obuf2.copy_out().expect("out2"));

    // Row 0 reran the FIRST arm's inputs exactly, so it must reproduce its bits.
    let row0 = f32v(&obuf.copy_out().expect("out"));
    assert_eq!(
        got2[..hidden],
        row0[..],
        "nrow=2 row 0 disagrees with the single-row kernel"
    );
    // Row 1 against a fresh single-row run of the same inputs.
    let x2b = dev(&f32b(&x2));
    let w2b = dev(&f32b(&w2));
    let (mut hbuf1, mut pbuf1, mut obuf1) = moe_bufs(e, 1, hidden, inter);
    range(&x2b, &w2b, &mut hbuf1, &mut pbuf1, 1);
    drain(&mut obuf1, &mut pbuf1, 0, hidden, &stream);
    device_sync().expect("sync");
    assert_eq!(
        got2[hidden..],
        f32v(&obuf1.copy_out().expect("out1"))[..],
        "nrow=2 row 1 disagrees with the single-row kernel (expert 1 carried weight 0)"
    );
    assert!(
        pbuf2.copy_out().expect("acc2").iter().all(|&b| b == 0),
        "moe_acc_drain left a batched accumulator row dirty"
    );
}

/// int4 MoE (moe_gateup_i4 → moe_down_i4 → moe_acc_drain), GROUP scales, vs a
/// matvec_i4+silu reference on the same quantized bytes. hidden ≥ 256 exercises
/// dot_i4_wave's dword fast path; inter < 256 its scalar tail.
///
/// `hidden = 2·I4_GROUP` is deliberate: one dword-fast-path step covers WAVE·8 = 256
/// columns, so lanes 0..15 and 16..31 sit in DIFFERENT groups within the same step —
/// a kernel that hoisted one scale per step, or per row, disagrees here. `inter` is
/// one whole group, so the scalar tail's group indexing is exercised too.
#[test]
fn moe_i4_matches_reference() {
    use rivoli::artifact::quant::{I4_GROUP, matvec_i4, quant_i4};
    let mut r = Lcg(0x14);
    let (hidden, inter, e) = (2 * I4_GROUP, I4_GROUP, 3usize);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..e).map(|_| r.f()).collect();
    let dims = [(inter, hidden), (inter, hidden), (hidden, inter)]; // gate, up, down (o,i)
    let enc = encode_experts(&mut r, e, &dims, |r, _p, o_dim, i_dim| {
        // Group `g` of every row is 4^g larger, so the stored scales genuinely DIFFER
        // between groups. Uniform-magnitude weights would give near-equal group scales
        // and a kernel reading the wrong group would still pass.
        let wv: Vec<f32> = (0..o_dim * i_dim)
            .map(|n| r.f() * 4f32.powi(((n % i_dim) / I4_GROUP) as i32))
            .collect();
        quant_i4(&wv, o_dim, i_dim)
    });

    let want = moe_reference(&x, &w, hidden, inter, |out, inp, ex, proj, o_dim, i_dim| {
        matvec_i4(out, inp, &enc[ex][proj].0, &enc[ex][proj].1, o_dim, i_dim)
    });

    let mut bufs: Vec<DeviceBuf> = Vec::new();
    // One ExpertDesc for both formats: the int4 kernel reads the six pointers as
    // packed weights + f32 row-scale (the scale ptr is typed *const u16 but only its
    // ADDRESS matters — cast the f32 buffer ptr through it).
    let descs: Vec<ExpertDesc> = (0..e)
        .map(|ex| ExpertDesc {
            gate_indices: push(enc[ex][0].0.clone(), &mut bufs),
            gate_scales: push(f32b(&enc[ex][0].1), &mut bufs) as *const u16,
            up_indices: push(enc[ex][1].0.clone(), &mut bufs),
            up_scales: push(f32b(&enc[ex][1].1), &mut bufs) as *const u16,
            down_indices: push(enc[ex][2].0.clone(), &mut bufs),
            down_scales: push(f32b(&enc[ex][2].1), &mut bufs) as *const u16,
        })
        .collect();
    let (descb, xb, wb) = (desc_buf(&descs), dev(&f32b(&x)), dev(&f32b(&w)));
    let (mut hbuf, mut pbuf, mut obuf) = moe_bufs(e, 1, hidden, inter);
    let stream = HipStream::new().expect("stream");
    for k in 0..e {
        expert_range_i4(&xb, &descb, &wb, &mut hbuf, &mut pbuf, hidden, inter, k, 1, 1, &stream);
    }
    drain(&mut obuf, &mut pbuf, 0, hidden, &stream);
    device_sync().expect("sync");
    assert_close(&want, &f32v(&obuf.copy_out().expect("out")), "moe_i4");
}

/// Run ONE int4 expert block through the real GPU path — `moe_gateup_i4` →
/// `moe_down_i4` → `moe_acc_drain`, weight 1.0 — and return `down(silu(gate·x) ⊙ up·x)`.
/// `blk` is an expert's on-disk bytes; `off` its `i4_slot_offsets`. Shared by the two
/// real-data tests so they exercise byte-for-byte the same launch, and a change to the
/// descriptor layout cannot be fixed in one test and left stale in the other.
fn gpu_i4_expert(blk: &[u8], off: &[usize; 6], x: &[f32], hidden: usize, inter: usize) -> Vec<f32> {
    let slot = dev(blk);
    let base = slot.ptr();
    // SAFETY: every offset lies within `blk`, which `slot` holds device-resident.
    let desc = ExpertDesc {
        gate_indices: unsafe { base.add(off[0]) },
        gate_scales: unsafe { base.add(off[1]) } as *const u16,
        up_indices: unsafe { base.add(off[2]) },
        up_scales: unsafe { base.add(off[3]) } as *const u16,
        down_indices: unsafe { base.add(off[4]) },
        down_scales: unsafe { base.add(off[5]) } as *const u16,
    };
    let descb = desc_buf(std::slice::from_ref(&desc));
    let (xb, wb) = (dev(&f32b(x)), dev(&f32b(&[1.0f32])));
    let (mut hbuf, mut pbuf, mut obuf) = moe_bufs(1, 1, hidden, inter);
    let stream = HipStream::new().expect("stream");
    expert_range_i4(&xb, &descb, &wb, &mut hbuf, &mut pbuf, hidden, inter, 0, 1, 1, &stream);
    drain(&mut obuf, &mut pbuf, 0, hidden, &stream);
    device_sync().expect("sync");
    f32v(&obuf.copy_out().expect("out"))
}

/// GPU int4 MoE on REAL colibri `.i4` bytes in the actual slot layout
/// (`i4_slot_offsets`) vs `matvec_i4` on the same bytes. The gap neither
/// `moe_i4_matches_reference` (synthetic `quant_i4`, separate buffers) nor the host
/// probe (CPU only) covers. Skips if the artifact is absent.
#[test]
fn moe_i4_real_data_matches_cpu() {
    use rivoli::artifact::quant::{
        i4_expert_bytes, i4_groups, i4_row_bytes, i4_slot_offsets, matvec_i4, vq_expert_layout,
    };
    use std::os::unix::fs::FileExt;
    let path = "/var/db/rivoli/glm52-vq3-full/L03.i4";
    let Ok(f) = std::fs::File::open(path) else {
        eprintln!("skip moe_i4_real_data: {path} absent");
        return;
    };
    let (hidden, inter) = (6144usize, 2048usize);
    let mut blk = vec![0u8; i4_expert_bytes(hidden, inter)];
    f.read_exact_at(&mut blk, 0).expect("read expert 0"); // routed expert 0, layer 3 (block 0)
    let off = i4_slot_offsets(hidden, inter);
    let dims = vq_expert_layout(hidden, inter); // [(gate o,i),(up),(down)]

    let mut r = Lcg(0x99);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let proj = |k: usize| -> (Vec<u8>, Vec<f32>) {
        let (o, i) = dims[k];
        let p = blk[off[k * 2]..off[k * 2] + o * i4_row_bytes(i)].to_vec();
        let sc = blk[off[k * 2 + 1]..off[k * 2 + 1] + o * i4_groups(i) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (p, sc)
    };
    let (gp, gs) = proj(0);
    let (up, us) = proj(1);
    let (dp, ds) = proj(2);
    let mut g = vec![0f32; inter];
    matvec_i4(&mut g, &x, &gp, &gs, inter, hidden);
    let mut u = vec![0f32; inter];
    matvec_i4(&mut u, &x, &up, &us, inter, hidden);
    let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
    let mut want = vec![0f32; hidden];
    matvec_i4(&mut want, &h, &dp, &ds, hidden, inter);

    let got = gpu_i4_expert(&blk, &off, &x, hidden, inter);
    let dot: f64 = want.iter().zip(&got).map(|(a, b)| *a as f64 * *b as f64).sum();
    let (na, nb): (f64, f64) = (
        want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt(),
        got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt(),
    );
    eprintln!(
        "moe_i4_real: cosine(GPU,CPU)={:.4} want[0..3]={:?} got[0..3]={:?}",
        dot / (na * nb + 1e-30),
        &want[..3],
        &got[..3]
    );
    assert_close(&want, &got, "moe_i4_real");
}

/// The int4 path against INDEPENDENT ground truth: the original fp8 checkpoint the
/// `.i4` set was derived from, dequantized and dotted in **f64** through the same
/// `down(silu(gate·x) ⊙ up·x)` chain. `moe_i4_real_data_matches_cpu` compares the GPU
/// kernel to our own `matvec_i4`; nothing in this reference touches `matvec_i4`,
/// `quant_i4`, or a nibble.
///
/// **What it catches, stated honestly.** Errors present in the DERIVATION — wrong
/// tensor, wrong dims, wrong `weight_scale_inv` tiling, an `.i4` set rebuilt from a
/// different checkpoint or through the old `fp8→vq3→int4` chain. Those move `gain`
/// toward 0 and `rel_l2` past 1. It does **not** catch a nibble order or zero point
/// that `quant_i4` and the kernel SHARE: `quant_i4` wrote these bytes, so a shared
/// convention cancels end to end. Nothing in this repo can pin that — both the writer
/// and the reader are ours; the anchor is colibri's `.qs` format, off-tree.
///
/// **It is also a COARSE gate, deliberately.** Two aggregate statistics over 6144
/// outputs cannot see corruption confined to a few percent of rows — a simulated 2%
/// K-tail truncation sits inside any band wide enough to hold the real spread. The
/// tight per-element gate is the sibling test's `assert_close` (max-abs at
/// `1e-3·max`), which catches that trivially. `max_err/max|ref|` here is the cheap
/// complement, not a substitute.
///
/// **The bands are tight because the measurement is deterministic.** `x` is a fixed
/// seed, the artifact is fixed, and the kernels reduce with a fixed `__shfl_down`
/// ladder — so this is not a sample from a distribution, it is one number. `bin/i4_audit`
/// (same generator, same `CHAIN_SEED`) reproduces it on the CPU to four decimals:
/// rel_l2 **0.2951**, gain **1.0009**, max_err/max|ref| **0.1603** for L3 expert 0.
/// Bands sit ~12% around those, which is roughly 3× more sensitive than a band sized
/// for x-draw spread would be — and rel-L2 through `silu` really does vary ~15% across
/// draws, which is exactly why quoting a same-distribution-but-different-seed anchor
/// (as an earlier revision did) is not good enough.
///
///     cargo run --release --bin i4_audit -- /var/db/rivoli/glm52-vq3-full <fp8-dir> \
///         --layer 3 --experts 0,7,256
#[test]
fn moe_i4_real_data_vs_fp8_ground_truth() {
    use rivoli::artifact::format::{FormatMeta, I4Source, Safetensors};
    use rivoli::artifact::model::ModelConfig;
    use rivoli::artifact::quant::{i4_expert_bytes, i4_slot_offsets, vq_expert_layout};
    use std::os::unix::fs::FileExt;
    const ART: &str = "/var/db/rivoli/glm52-vq3-full";
    const LAYER: usize = 3; // first MoE layer; block 0 = routed expert 0
    let path = format!("{ART}/L{LAYER:02}.i4");
    let Ok(f) = std::fs::File::open(&path) else {
        eprintln!("skip moe_i4_real_data_vs_fp8: {path} absent");
        return;
    };
    // The checkpoint comes from the artifact's OWN stamp, and the bands below only
    // describe the `fp8->int4` chain. The retired `vq3_to_i4` rewrote `L{l}.i4` IN PLACE
    // in this very directory with the other chain, and artifacts it produced are still
    // on disk; without this the test would keep running and quietly certify the
    // derivation it exists to distinguish.
    let Some(prov) = I4Source::load(ART).expect("read i4_source") else {
        eprintln!("skip moe_i4_real_data_vs_fp8: artifact carries no i4_source stamp");
        return;
    };
    assert_eq!(
        prov.chain, "fp8->int4",
        "the assertion bands characterise the fp8->int4 chain; this artifact is {}",
        prov.chain
    );
    // A set quantized at a different group size is a differently-shaped scale array,
    // not a slightly-different one: reading it walks the wrong strides and yields
    // rel_l2=NaN rather than a large error. Skip loudly instead of asserting on NaN,
    // which reports "systematic gain error" and sends the reader hunting a numerics
    // bug that is really a stale artifact.
    if prov.group != Some(rivoli::artifact::quant::I4_GROUP) {
        eprintln!(
            "skip moe_i4_real_data_vs_fp8: artifact is group {:?}, this build reads group {} \
             — rebuild with fp8_to_i4",
            prov.group,
            rivoli::artifact::quant::I4_GROUP
        );
        return;
    }
    assert!(
        prov.layers[0] <= LAYER && LAYER < prov.layers[1],
        "layer {LAYER} outside the stamped range {:?}",
        prov.layers
    );
    let Ok(src) = Safetensors::open_dir(&prov.src) else {
        eprintln!("skip moe_i4_real_data_vs_fp8: checkpoint {} absent", prov.src);
        return;
    };
    let block = FormatMeta::load(ART).expect("format meta").fp8_block;
    // From the manifest, not hardcoded: the slot offsets below are functions of these
    // dims, so a shape the constants disagreed with would read the wrong bytes and
    // still produce a plausible-looking number.
    let cfg = ModelConfig::load(ART).expect("artifact manifest");
    let (hidden, inter) = (cfg.hidden, cfg.moe_inter);
    let mut blk = vec![0u8; i4_expert_bytes(hidden, inter)];
    f.read_exact_at(&mut blk, 0).expect("read expert 0");
    let off = i4_slot_offsets(hidden, inter);
    let dims = vq_expert_layout(hidden, inter);

    // ── the reference: fp8 → f64, no quantized code in the path ─────────────────
    let base = format!("model.layers.{LAYER}.mlp.experts.0");
    let wref: Vec<Vec<f32>> = ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .zip(&dims)
        .map(|(p, &(o, i))| {
            src.dequant_fp8(&format!("{base}.{p}"), o, i, block)
                .expect("dequant fp8")
        })
        .collect();
    let mv64 = |w: &[f32], x: &[f32], o_dim: usize, i_dim: usize| -> Vec<f32> {
        (0..o_dim)
            .map(|o| {
                w[o * i_dim..(o + 1) * i_dim]
                    .iter()
                    .zip(x)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    };
    let mut r = Lcg(0x5A17);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let g = mv64(&wref[0], &x, inter, hidden);
    let u = mv64(&wref[1], &x, inter, hidden);
    let hv: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
    let want = mv64(&wref[2], &hv, hidden, inter);

    // ── what the shipped int4 path produces, on the GPU, from the real bytes ────
    let got = gpu_i4_expert(&blk, &off, &x, hidden, inter);

    let (mut num, mut den, mut dot) = (0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(&want) {
        let (a, b) = (a as f64, b as f64);
        num += (a - b) * (a - b);
        den += b * b;
        dot += a * b;
    }
    let (rel_l2, gain) = ((num / den).sqrt(), dot / den);
    let (mx_ref, mx_err) = got.iter().zip(&want).fold((0f64, 0f64), |(r, e), (&a, &b)| {
        (r.max(b.abs() as f64), e.max((a - b).abs() as f64))
    });
    let rel_max = mx_err / mx_ref;
    // Margins, not just pass/fail — a run drifting from 0.24 toward 0.32 stays green
    // and unremarked right up until it crosses (the reason `assert_close` prints them).
    println!(
        "moe_i4_vs_fp8: rel_l2={rel_l2:.4} (band 0.14..0.20) gain={gain:.4} (band 1.01..1.09) \
         max_err/max|ref|={rel_max:.4} (bound 0.16)"
    );
    // These bands describe GROUP-128 (`quant::I4_GROUP`). The per-row format they replaced
    // anchored at rel_l2 0.2951 / gain 1.0009; group-128 measures 0.1698 / 1.0493 on the
    // same artifact coordinates and seed — a 42% error reduction, and the reason `--mode
    // int4` went from PPL 73.43 to 5.12. Changing `I4_GROUP` moves these ON PURPOSE:
    // re-anchor with `bin/i4_audit`, do not widen. (A mismatched artifact never reaches
    // here — the group gate above skips it, because reading one yields rel_l2=NaN and this
    // assertion would then blame a "systematic gain error" for a stale file.)
    assert!(
        (1.01..=1.09).contains(&gain),
        "SYSTEMATIC gain error vs fp8 ground truth: gain={gain:.4}. Cosine scores 1.0000 \
         for any uniform or per-group scale error, so nothing else here would see this. \
         Group-128 at an amax/7 step runs slightly hot (~1.05) because the step is set by \
         each group's own extreme; a value near 1.00 means the loading factor changed."
    );
    assert!(
        (0.14..=0.20).contains(&rel_l2),
        "rel_l2={rel_l2:.4} != the deterministic 0.1698 this artifact and seed produce. \
         Below 0.14 the reference has collapsed into the thing it checks (a 128-wide group \
         at an amax/7 step still loses ~0.12, ~0.17 through the silu chain); above 0.20 the \
         derivation is wrong, not merely coarse — and 0.26+ is the per-row format, i.e. an \
         artifact that predates group scales."
    );
    assert!(
        rel_max <= 0.16,
        "max_err/max|ref|={rel_max:.4} > 0.16 — a few corrupted output rows move this \
         while leaving rel_l2 inside its band"
    );
}

/// `index_topk` vs the host selection it replaces, on the shapes that actually occur.
///
/// The oracle is the engine's own two lines — `topk_into(scores, k, &mut sel)` then
/// `sel.sort_unstable()` — not a reimplementation, so this pins the kernel to what the
/// attend has always consumed rather than to my reading of it.
///
/// **The row buffer is sentinel-filled and the tail is asserted untouched.** Without
/// that, over-selection is invisible: the readback would be truncated to `want.len()`,
/// and whenever the correct answer is an index *prefix* — which it is for the
/// `ReLU-sparse` case, since its non-zero scores sit at indices 0..300 — a kernel that
/// emitted every tied row would still match on the first k. Measured against a serial
/// simulation of this kernel: of three mutations (drop the cross-chunk tie carry; use
/// `<= need` for the tie budget; drop the -0.0 canonicalisation), the tie carry is
/// caught by seven cases on selection alone, the canonicalisation only by
/// `mixed +0.0/-0.0`, and **`<= need` by nothing except the tail check**.
///
/// The `ReLU-sparse` and `scattered zeros` cases are tie-DOMINATED, which is the regime
/// where the index-ascending rule decides the bulk of the selection rather than a
/// handful of boundary entries. Whether the engine actually produces such an array is
/// unmeasured (docs/investigations/npu-offload.md), so these are chosen as the hardest case for the tiebreak,
/// not as a claim about production data. `scattered zeros` additionally makes the answer
/// non-prefix, which is the combination nothing else here covers — and note the two
/// differ in ORDER as well as scatter: `ReLU-sparse` is pre-sorted into the host
/// comparator's own order, which is its best case and a trap when timing rather than
/// checking. nt = 5185 and k = 2048 are the longer in-engine context and `index_topk`.
#[test]
fn index_topk_matches_host_selection() {
    const SENTINEL: u32 = 0xFFFF_FFFF;
    fn host(scores: &[f32], k: usize) -> Vec<u32> {
        let mut sel = Vec::new();
        rivoli::math::topk_into(scores, k, &mut sel);
        sel.sort_unstable();
        sel.iter().map(|&i| i as u32).collect()
    }
    let mut rng = Lcg(0x7071_C0DE);
    let nt = 5185usize;
    // Realistic shape, answer is a prefix: real scores at the front, rest ReLU'd to 0.0.
    let mut relu_sparse = vec![0.0f32; nt];
    for (i, x) in relu_sparse.iter_mut().enumerate().take(300) {
        *x = (300 - i) as f32 * 0.25;
    }
    // Realistic shape, answer is NOT a prefix: the same sparsity, scattered.
    let mut scattered = vec![0.0f32; nt];
    for j in 0..300 {
        scattered[(j * 7919) % nt] = (300 - j) as f32 * 0.25;
    }
    let dense: Vec<f32> = (0..nt).map(|_| rng.f() * 8.0).collect();
    let heavy_ties: Vec<f32> = (0..nt).map(|_| (rng.f() * 4.0).floor()).collect();
    let cases: Vec<(&str, Vec<f32>, usize)> = vec![
        (
            "mixed +0.0/-0.0",
            (0..4096)
                .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
                .collect(),
            2048,
        ),
        (
            "negatives only",
            (0..4096).map(|i| -((i % 11) as f32)).collect(),
            2048,
        ),
        ("k == nt", (0..2048).map(|i| (i % 7) as f32).collect(), 2048),
        ("k == nt - 1", (0..2049).map(|i| (i % 7) as f32).collect(), 2048),
        ("k > nt (wrapper clamp)", (0..500).map(|i| (i % 7) as f32).collect(), 2048),
        ("single block", (0..200).map(|i| (i % 5) as f32).collect(), 64),
        ("ReLU-sparse (engine shape, prefix answer)", relu_sparse, 2048),
        ("scattered zeros (engine shape, non-prefix answer)", scattered, 2048),
        ("dense random", dense, 2048),
        ("heavy ties", heavy_ties, 2048),
    ];
    for (name, scores, k) in cases {
        let n = scores.len();
        let want = host(&scores, k);
        let written = k.min(n);
        assert_eq!(
            want.len(),
            written,
            "oracle wrote {} rows, expected {written} on {name}",
            want.len()
        );
        let sb = dev(&f32b(&scores));
        // Sentinel fill: anything the kernel writes past `written` shows up below.
        let slots = n.max(k);
        let mut rb = dev(&vec![0xFFu8; slots * 4]);
        // SAFETY: scores holds n f32; rows holds >= min(k,n) u32.
        unsafe {
            launch_index_topk(sb.ptr() as *const f32, n, k, rb.ptr_mut() as *mut u32)
                .expect("index_topk");
        }
        device_sync().expect("sync");
        let raw = rb.copy_out().expect("rows out");
        let got: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            &got[..written],
            &want[..],
            "index_topk selection differs on {name}"
        );
        assert!(
            got[written..].iter().all(|&v| v == SENTINEL),
            "index_topk wrote past min(k,nt)={written} on {name} — over-selection"
        );
        assert!(
            got[..written].windows(2).all(|w| w[0] < w[1]),
            "index_topk output not strictly ascending on {name}"
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
#[test]
fn batched_rows_are_bit_identical_to_single_rows() {
    use rivoli::backend::hip::launch_gemv_f32;
    let mut r = Lcg(0xba7c);

    // --- gemv_f32 (the MoE router gate) ---
    {
        let (o_dim, i_dim) = (64usize, 128usize);
        let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
        let xs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..i_dim).map(|_| r.f()).collect());
        let wb = dev(&f32b(&w));
        let (got, single) = single_vs_batched(&xs, o_dim, |xb, yb, nrow| {
            // SAFETY: `x` holds `nrow` rows of i_dim and `y` `nrow` rows of o_dim; both
            // nrow 1 and nrow 2 are instantiated.
            unsafe {
                launch_gemv_f32(xb.ptr() as *const f32, wb.ptr() as *const f32,
                                o_dim, i_dim, nrow, yb.ptr_mut() as *mut f32)
            }
            .expect("gemv_f32");
        });
        assert_eq!(got[..o_dim], single[0][..], "gemv_f32 row 0 must be bit-identical");
        assert_eq!(got[o_dim..], single[1][..], "gemv_f32 row 1 must be bit-identical");
    }

    // --- gemv_i8 (lm_head). i_dim % 4 == 0 so both rows take the dword-quad path. ---
    {
        let (o_dim, i_dim) = (96usize, 260usize);
        let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| (r.f() * 127.0) as i8 as u8).collect();
        let scale: Vec<f32> = (0..o_dim).map(|_| (r.f() * 0.01).abs() + 1e-4).collect();
        let xs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..i_dim).map(|_| r.f()).collect());
        let (pb, sb) = (dev(&packed), dev(&f32b(&scale)));
        let (got, single) =
            single_vs_batched(&xs, o_dim, |xb, yb, nrow| gemv_i8(xb, &pb, &sb, yb, o_dim, i_dim, nrow));
        assert_eq!(got[..o_dim], single[0][..], "gemv_i8 row 0 must be bit-identical");
        assert_eq!(got[o_dim..], single[1][..], "gemv_i8 row 1 must be bit-identical");
    }

    // --- mla_absorb_fp8 / mla_value_fp8 (both through kv_b). kvl % 4 == 0 and
    //     block >= 4, so this exercises the quad path both kernels actually run. ---
    {
        let (h, qh, nope, vh, kvl, block) = (3usize, 20usize, 12usize, 8usize, 16usize, 4usize);
        let (packed, scale) = kvb_fp8(&mut r, h * (nope + vh), kvl, block);
        let (kb, sb) = (dev(&packed), dev(&f32b(&scale)));
        let qs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..h * qh).map(|_| r.f()).collect());
        let cs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..h * kvl).map(|_| r.f()).collect());

        // Launch only — the sync stays at the call sites so both kernels are still in
        // flight together before either result is read, which is how the engine's own
        // attention step runs them and how the nrow=1 arm below has always run them.
        let absorb = |qb: &DeviceBuf, ab: &mut DeviceBuf, nrow: usize| {
            mla_absorb(qb, &kb, &sb, ab, h, qh, nope, vh, kvl, block, nrow).expect("absorb");
        };
        let value = |clb: &DeviceBuf, vb: &mut DeviceBuf, nrow: usize| {
            mla_value(clb, &kb, &sb, vb, h, nope, vh, kvl, block, nrow).expect("value");
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
        let mut qboth = qs[0].clone();
        qboth.extend_from_slice(&qs[1]);
        let mut cboth = cs[0].clone();
        cboth.extend_from_slice(&cs[1]);
        let (qb, clb) = (dev(&f32b(&qboth)), dev(&f32b(&cboth)));
        let mut ab = dev(&vec![0u8; 2 * h * kvl * 4]);
        let mut vb = dev(&vec![0u8; 2 * h * vh * 4]);
        absorb(&qb, &mut ab, 2);
        value(&clb, &mut vb, 2);
        device_sync().expect("sync");
        let ga = f32v(&ab.copy_out().expect("out"));
        let gv = f32v(&vb.copy_out().expect("out"));
        assert_eq!(ga[..h * kvl], abs1[0][..], "mla_absorb row 0 must be bit-identical");
        assert_eq!(ga[h * kvl..], abs1[1][..], "mla_absorb row 1 must be bit-identical");
        assert_eq!(gv[..h * vh], val1[0][..], "mla_value row 0 must be bit-identical");
        assert_eq!(gv[h * vh..], val1[1][..], "mla_value row 1 must be bit-identical");
    }

    // --- the int4 MoE pair (moe_gateup_i4 / moe_down_i4) ---
    //
    // Compares the FIXED-POINT accumulator as u64, not the drained f32: integer equality is
    // the strictest form of "bit-identical" available and it is the quantity the atomics
    // actually produce.
    //
    // The weight matrix is deliberately ragged — row 1 carries 0.0 on expert 1 — so this
    // covers the union's correctness argument as well as the batching: a row that did not
    // route to a union expert must come out exactly as if that expert were never launched.
    // That is the property the whole speculative claim rests on, and `0 * dv` with a
    // non-finite `dv` would otherwise clamp to a FINITE extreme rather than vanish.
    {
        use rivoli::artifact::quant::{I4_GROUP, quant_i4};
        let (hidden, inter, ne) = (2 * I4_GROUP, I4_GROUP, 2usize);
        let dims = [(inter, hidden), (inter, hidden), (hidden, inter)]; // gate, up, down
        let mut bufs: Vec<DeviceBuf> = Vec::new();
        let mut descs: Vec<ExpertDesc> = Vec::new();
        for _ in 0..ne {
            let mut p: Vec<*const u8> = Vec::new();
            for &(o_dim, i_dim) in &dims {
                // Group-varying magnitudes, as in `moe_i4_matches_reference`: equal group
                // scales would let a kernel reading the wrong group still pass.
                let wv: Vec<f32> = (0..o_dim * i_dim)
                    .map(|n| r.f() * 4f32.powi(((n % i_dim) / I4_GROUP) as i32))
                    .collect();
                let (packed, scale) = quant_i4(&wv, o_dim, i_dim);
                bufs.push(dev(&packed));
                p.push(bufs.last().expect("packed").ptr());
                bufs.push(dev(&f32b(&scale)));
                p.push(bufs.last().expect("scale").ptr());
            }
            descs.push(ExpertDesc {
                gate_indices: p[0],
                gate_scales: p[1] as *const u16,
                up_indices: p[2],
                up_scales: p[3] as *const u16,
                down_indices: p[4],
                down_scales: p[5] as *const u16,
            });
        }
        let descb = desc_buf(&descs);
        let xs: [Vec<f32>; 2] = std::array::from_fn(|_| (0..hidden).map(|_| r.f()).collect());
        // Row 0 routes to both experts; row 1 only to expert 0. Indexed `[e * R + t]`.
        let w1: [Vec<f32>; 2] = [vec![0.75, 0.5], vec![0.25, 0.0]];
        let stream = HipStream::new().expect("stream");
        let run = |xb: &DeviceBuf, wb: &DeviceBuf, hb: &mut DeviceBuf,
                   ab: &mut DeviceBuf, nrow: usize| -> Vec<u8> {
            expert_range_i4(xb, &descb, wb, hb, ab, hidden, inter, 0, ne, nrow, &stream);
            device_sync().expect("sync");
            ab.copy_out().expect("out")
        };

        let single: Vec<Vec<u8>> = xs
            .iter()
            .zip(&w1)
            .map(|(x, w)| {
                let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
                let mut hb = dev(&vec![0u8; ne * inter * 4]);
                let mut ab = dev(&vec![0u8; hidden * 8]);
                run(&xb, &wb, &mut hb, &mut ab, 1)
            })
            .collect();
        let mut xboth = xs[0].clone();
        xboth.extend_from_slice(&xs[1]);
        // Row-fastest: [e0r0, e0r1, e1r0, e1r1] — expert 1's row-1 weight is the 0.0.
        let wboth = vec![w1[0][0], w1[1][0], w1[0][1], w1[1][1]];
        let (xb, wb) = (dev(&f32b(&xboth)), dev(&f32b(&wboth)));
        let mut hb = dev(&vec![0u8; ne * 2 * inter * 4]);
        let mut ab = dev(&vec![0u8; 2 * hidden * 8]);
        let got = run(&xb, &wb, &mut hb, &mut ab, 2);
        assert_eq!(got[..hidden * 8], single[0][..], "moe_i4 row 0 must be bit-identical");
        assert_eq!(got[hidden * 8..], single[1][..], "moe_i4 row 1 must be bit-identical");
    }
}
