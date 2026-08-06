//! `kernels/fwd.hip`'s glue kernels against host references. Compiles to nothing without
//! rocm.
//!
//! These four are the small device ops that stitch a resident token together — an
//! embedding lookup, the KV append, a query gather, and the non-finite localiser. All were
//! uncovered until 2026-08-06, when `tests/kernel_coverage.rs` was re-keyed onto
//! `src/backend/hip.rs` and named them. `argmax` and `vadd`, their neighbours in the same
//! file, have been covered since the M-series and live in `tests/kernel.rs`; these landed
//! here rather than there because that file is already 1.6k lines and these share no
//! scaffolding with its GEMV and MoE oracles.
//!
//! **`vaxpy` is the one `fwd.hip` launcher left without an oracle, and that is deliberate.**
//! It has NO caller anywhere in the tree: `--moe-gain` reaches the residual add through
//! `launch_moe_acc_drain`'s gain multiply on MoE layers (`gpu.rs`), and dense layers take
//! the unscaled `launch_vadd`. Its last caller was a Vulkan stub that existed only to
//! satisfy this census. An oracle written to turn the census green would keep a dead kernel
//! compiled and call that coverage, so the census is left RED on `launch_vaxpy` pending a
//! decision to delete it — `docs/investigations/refactor-2026-08.md` §Track 1.
//!
//! **Every comparison in this file is BITWISE**, and that is a property of the kernels
//! rather than a standard of strictness: each is one thread per output element with no
//! reduction. The one apparent exception, `append_kv`'s per-128 block amax, reduces with
//! `fmaxf` — and max is order-independent EXACTLY where a sum is not.
//! `common::assert_bitwise` carries the measurement for when this is not available.
//!
//! Device tests: run with `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
//! Parallel libtest threads each build their own tier, pool and io_uring ring, which is the
//! diagnosed cause of the "intermittent gpustream hang".
#![cfg(feature = "rocm")]

use rivoli::backend::hip::{
    launch_append_kv, launch_embed_i8_row, launch_flag_nonfinite, launch_gather_rope,
};
use rivoli::math::{E4M3_BLOCK, E4M3_MAX, f32_to_bf16, f32_to_e4m3};

mod common;
use common::{
    Lcg, assert_bits, assert_bitwise, back, dev, f32b, f32v, i8_weights, ok, u16v, u32v, zeros,
};

/// `embed_i8_row` — `x[i] = (i8)embed[token][i] · scale[token]`.
///
/// The row INDEX is what this is really about. The kernel addresses
/// `packed[(size_t)token * hidden + i]`, and every other row of the table is populated with
/// different data, so reading the wrong row produces numbers of the right shape and the
/// wrong content. A single-row fixture would pass on that.
///
/// It does NOT pin the `(size_t)`, and nothing here can: the real table is 154880 x 6144,
/// whose largest linear index is 951,582,720 — 44% of `INT_MAX`. `int` would overflow only
/// at token 349,525, 2.26x the vocabulary, so the cast is defensive rather than
/// load-bearing. This fixture's largest index is 5119. `gather_rope` and `append_kv` below
/// are smaller again.
#[test]
fn embed_i8_row_gathers_the_requested_row() {
    let (rows, hidden, token) = (6usize, 1024usize, 4usize);
    let mut r = Lcg(0xE3B0);
    // Rows-as-tokens, columns-as-hidden: exactly `i8_weights`' `[o_dim, i_dim]` layout with
    // one scale per row, which is what the embedding table is.
    let (packed, scale) = i8_weights(&mut r, rows, hidden);
    let want: Vec<f32> = (0..hidden)
        .map(|i| f32::from(packed[token * hidden + i] as i8) * scale[token])
        .collect();

    let (pb, sb) = (dev(&packed), dev(&f32b(&scale)));
    let mut xb = zeros(hidden * 4);
    // SAFETY: `packed` is `rows·hidden` bytes and `scale` `rows` f32, so row `token` is in
    // bounds; `x` is `hidden` f32. All live for the call.
    ok(
        unsafe {
            launch_embed_i8_row(
                pb.ptr(),
                sb.ptr() as *const f32,
                token,
                hidden,
                xb.ptr_mut() as *mut f32,
            )
        },
        "embed_i8_row",
    );

    assert_bits(&want, &f32v(&back(&xb)), "embed_i8_row");
}

/// `gather_rope` — `qrope[head·ropn + d] = q[head·qh + nope + d]`.
///
/// Pure data movement, so the only thing that can be wrong is an index, and the fixture is
/// built so every index error is visible: `qh > nope + ropn` leaves a tail after each
/// head's roped segment, and `nope > 0` puts a prefix before it. A kernel that dropped
/// `nope`, walked `ropn` instead of `qh`, or ran off the end of a head lands in data that
/// belongs to a different head.
#[test]
fn gather_rope_extracts_each_heads_roped_segment() {
    let (h, qh, nope, ropn) = (8usize, 192usize, 96usize, 64usize);
    assert!(
        nope + ropn < qh,
        "the fixture needs a tail after the segment"
    );
    let mut r = Lcg(0x60A7);
    let q: Vec<f32> = (0..h * qh).map(|_| r.f()).collect();
    let want: Vec<f32> = (0..h * ropn)
        .map(|i| q[(i / ropn) * qh + nope + i % ropn])
        .collect();

    let qb = dev(&f32b(&q));
    let mut ob = zeros(h * ropn * 4);
    // SAFETY: `q` is `h·qh` f32 and `qrope` `h·ropn` f32, both live for the call.
    ok(
        unsafe {
            launch_gather_rope(
                qb.ptr() as *const f32,
                ob.ptr_mut() as *mut f32,
                h,
                qh,
                nope,
                ropn,
            )
        },
        "gather_rope",
    );

    assert_bits(&want, &f32v(&back(&ob)), "gather_rope");
}

/// `append_kv` — one token's latent as fp8-e4m3 with per-128 block scales, plus its roped
/// key as bf16, written at row `pos` of three slabs.
///
/// Bit-exact against the host, and the reduction does not spoil that: the block amax is a
/// `fmaxf` tree, and max is order-independent EXACTLY where a sum is not. Everything after
/// it — `amax/448`, `latent[i]/scl`, `f32_to_e4m3` — is one operation per element.
///
/// Three slabs are allocated with FIVE rows and only row 3 is written, and the other four
/// are asserted to still be zero. That is the point of the fixture rather than
/// thoroughness: `pos` multiplies three different strides (`kvl`, `n_blocks`, `ropn`), all
/// distinct here, so a kernel that used the wrong one for any slab writes a real row of a
/// real slab and a single-row fixture could not tell.
#[test]
fn append_kv_quantizes_one_row_and_leaves_the_slab_alone() {
    let (rows, kvl, ropn, pos) = (5usize, 256usize, 64usize, 3usize);
    let n_blocks = kvl / E4M3_BLOCK;
    assert_eq!(n_blocks, 2, "the fixture needs more than one block scale");
    let mut r = Lcg(0xA99E);
    // The two blocks are scaled DIFFERENTLY, 1x and 20x, giving block amaxes 0.9949 and
    // 19.9455. A uniform factor cannot do this: scaling the whole row leaves the RATIO of
    // the two block maxima unchanged, and measured at this seed a flat `* 20.0` puts them
    // 0.24% apart — under which a kernel taking ONE scale for the whole row normalises the
    // low block by the high block's amax, dropping it ~4.3 binades and moving 4 of 256
    // codes. At the 20x split the same defect moves all 128 low-block codes. Detection was
    // never the issue — the `lscale` comparison below sees one scale where there are two,
    // and bitwise has no margin to lose. What the split buys is that the CODE comparison
    // fails loudly too, rather than by four elements.
    let latent: Vec<f32> = (0..kvl)
        .map(|i| r.f() * if i < E4M3_BLOCK { 1.0 } else { 20.0 })
        .collect();
    let rope: Vec<f32> = (0..ropn).map(|_| r.f()).collect();

    let mut want_lc8 = vec![0u8; rows * kvl];
    let mut want_ls = vec![0.0f32; rows * n_blocks];
    let mut want_rc = vec![0u16; rows * ropn];
    for b in 0..n_blocks {
        let blk = &latent[b * E4M3_BLOCK..(b + 1) * E4M3_BLOCK];
        let amax = blk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        // `E4M3_MAX`, not a literal 448.0: the kernel divides by the same saturation point
        // the host quantizer clamps to, and a drift between them is silent.
        let scl = if amax > 0.0 { amax / E4M3_MAX } else { 1.0 };
        want_ls[pos * n_blocks + b] = scl;
        for (i, &v) in blk.iter().enumerate() {
            want_lc8[pos * kvl + b * E4M3_BLOCK + i] = f32_to_e4m3(v / scl);
        }
    }
    for (i, &v) in rope.iter().enumerate() {
        want_rc[pos * ropn + i] = f32_to_bf16(v);
    }

    let (lb, rb) = (dev(&f32b(&latent)), dev(&f32b(&rope)));
    let mut c8 = zeros(rows * kvl);
    let mut ls = zeros(rows * n_blocks * 4);
    let mut rc = zeros(rows * ropn * 2);
    // SAFETY: `latent` is `kvl` f32 and `rope` `ropn` f32; the three slabs hold `rows`
    // rows of `kvl` u8, `n_blocks` f32 and `ropn` u16, and `pos < rows`. All live.
    ok(
        unsafe {
            launch_append_kv(
                lb.ptr() as *const f32,
                rb.ptr() as *const f32,
                c8.ptr_mut(),
                ls.ptr_mut() as *mut f32,
                rc.ptr_mut() as *mut u16,
                pos,
                kvl,
                ropn,
                n_blocks,
            )
        },
        "append_kv",
    );

    // Anti-vacuity on the fixture: e4m3 saturates at ±448 and the scale is chosen so the
    // block extreme lands exactly there, so a latent that quantized to all-zero codes
    // would make the whole comparison agree with the untouched slab.
    assert!(
        want_lc8[pos * kvl..(pos + 1) * kvl]
            .iter()
            .filter(|&&c| c != 0)
            .count()
            > kvl / 2,
        "the fixture quantized to mostly zero codes — the comparison would be vacuous"
    );
    assert_bitwise(&want_lc8, &back(&c8), "append_kv latent codes");
    assert_bits(&want_ls, &f32v(&back(&ls)), "append_kv block scales");
    assert_bitwise(&want_rc, &u16v(&back(&rc)), "append_kv roped key");
}

/// `flag_nonfinite` — record `tag` in `*flag` if any of `x[0..n]` is non-finite, first
/// writer wins.
///
/// Four launches in sequence, and the fourth is the one that makes the third mean
/// anything. The third asserts the flag does NOT move when a second, later fault is
/// reported — first-writer-wins, which is the entire reason the kernel exists: the tag
/// names the EARLIEST (pos, layer) that went bad, not the last. But "the flag did not
/// change" is also what a kernel that cannot see `+inf` at all would produce. The fourth
/// re-runs the same array against a FRESH flag and requires the tag to land, so the two
/// readings are separated.
///
/// This is the file's one test with no reference vector: the output is a single word and
/// the reference is the `atomicCAS(flag, 0, tag)` contract stated in the launcher's doc.
#[test]
fn flag_nonfinite_records_the_first_fault_only() {
    let n = 1024;
    let mut r = Lcg(0xF1A6);
    let clean: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let poison = |at: usize, v: f32| -> Vec<f32> {
        let mut x = clean.clone();
        x[at] = v;
        x
    };

    let mut flag = zeros(4);
    let fire = |x: &[f32], tag: u32, flag: &mut rivoli::memory::device::DeviceBuf| -> u32 {
        let xb = dev(&f32b(x));
        // SAFETY: `x` is `n` live device f32 and `flag` one live device u32.
        ok(
            unsafe {
                launch_flag_nonfinite(xb.ptr() as *const f32, n, tag, flag.ptr_mut() as *mut u32)
            },
            "flag_nonfinite",
        );
        u32v(&back(flag))[0]
    };

    assert_eq!(fire(&clean, 5, &mut flag), 0, "a finite pass must not flag");
    assert_eq!(
        fire(&poison(700, f32::NAN), 5, &mut flag),
        5,
        "NaN must flag"
    );
    assert_eq!(
        fire(&poison(3, f32::NEG_INFINITY), 9, &mut flag),
        5,
        "the FIRST tag must survive a later fault"
    );
    // Tag 0 is RESERVED for "clean": it is the `atomicCAS` comparand, so a fault reported
    // with tag 0 would be indistinguishable from no fault. This pins the KERNEL's half of
    // that contract — a kernel that remapped tag 0 to a non-zero sentinel goes red here and
    // nowhere else in the tree. It does NOT pin `gpu.rs`'s `1 + (pos * 256 + l)`, which is
    // what makes the reservation safe for the caller; delete that `1 +` and this stays
    // green. Closing that needs a caller-level test, not a kernel oracle.
    let mut sentinel = zeros(4);
    assert_eq!(
        fire(&poison(3, f32::NAN), 0, &mut sentinel),
        0,
        "tag 0 is the atomicCAS comparand and cannot be recorded as a fault"
    );

    let mut fresh = zeros(4);
    assert_eq!(
        fire(&poison(3, f32::NEG_INFINITY), 9, &mut fresh),
        9,
        "-inf must flag on its own — without this the line above proves only that the \
         kernel cannot see it"
    );
}
