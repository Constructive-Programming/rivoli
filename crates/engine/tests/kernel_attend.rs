//! The MLA kernels vs their CPU oracles in quant.rs: the two fp8 kv_b projections (absorb and
//! value) and the flash attend, plus the bit-identity claim the kv_b pair makes about batched
//! rows. Compiles to nothing without rocm.
//!
//! **Split out of `kernel.rs` on 2026-08-16 — by COHESION, not by size alone**, the same
//! argument `kernel_moe.rs` and `kernel_vector.rs` made on 2026-08-15 when that file first came
//! down from 2263 lines and 79 functions over nine unrelated kernel families. This was the last
//! of those nine families still sharing the file with GEMV, and after that first split settled,
//! `kernel.rs` sat at 804 lines against the build's 800-line soft cap — no new test drove it
//! there, the cap alone did.
//!
//! What travelled: `MlaIo` and `AttIo` bundle the operand lists the two kv_b launchers and the
//! attend launcher take, `kvb_fp8` draws the block-scaled kv_b every MLA oracle here reads, and
//! `check_attend`/`check_mla_fp8` are the two dense CPU references the guard tests and the
//! shape-sweep tests both drive. `batched_mla_kvb` was an arm of `kernel.rs`'s umbrella
//! bit-identity test and is now its own `#[test]` with its own seed, on the same argument
//! `kernel_moe.rs`'s header gives for its own MoE arm.
//!
//! `GemvIo` did NOT come: the two kv_b projections and attend take their own bundles, and
//! nothing here is a vector-against-matrix dot like `gemv_fp8`/`gemv_i8`/`gemv_i4`. `assert_out`
//! DID move, but to `common` rather than here — both this file and `kernel.rs` read a device
//! destination back against a CPU oracle, and a second copy of that join is what `build.rs`'s
//! jscpd gate is for.
//!
//! Every body below travelled VERBATIM with its comments — in this repo a comment carries the
//! measurement that justified the choice, so a re-worded one loses evidence.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use anyhow::Result;
use rivoli_backend::hip::{
    attend_scratch_floats, device_sync, launch_attend, launch_mla_absorb_fp8, launch_mla_value_fp8,
};
use rivoli_core::num::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3};
use rivoli_core::routing::softmax;

mod common;
use common::{
    Att, DeviceBuf, Lcg, Mla, assert_guard, assert_out, assert_rows, block_scales, dev, f32b, f32v,
    u16b,
};

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

/// `mla_absorb_fp8`. `Result`-returning for the same reason as `gemv_fp8` in `kernel.rs`.
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
/// `gemv_fp8`: the guard test hands it a kvl it requires to be REJECTED.
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

/// The MLA kv_b pair (mla_absorb_fp8 / mla_value_fp8).
///
/// **The MLA arm of `kernel.rs::batched_rows_are_bit_identical_to_single_rows`**, which states
/// the claim for every kernel that takes `nrow` and still runs the two GEMV arms that stayed. It
/// became a `#[test]` of its own on 2026-08-16 with this split, on the same argument
/// `kernel_moe.rs`'s header gives for its own MoE arm the day before: the arms that stayed need
/// only `GemvIo`, this one needs `MlaIo` and `Mla`, and keeping them under one function would
/// have kept both halves in one file. Splitting the umbrella does not weaken it — each arm
/// always asserted its own kernel and named it in the failure.
///
/// It draws from its OWN `Lcg`, seeded identically to the arm it left (`0xba7c`) — any seed
/// works, since the comparison is a kernel against itself at two row counts, not against a
/// reference, which is why the umbrella's seed is reused verbatim rather than a new one
/// invented.
#[test]
fn batched_mla_rows_are_bit_identical_to_single_rows() {
    let mut r = Lcg(0xba7c);
    batched_mla_kvb(&mut r);
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
