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
//! **`fwd.hip` had a fifth launcher, `vaxpy`, and this file is why it is gone.** The census
//! named it as uncovered and the honest answer was not an oracle: it had NO caller anywhere
//! in the tree. `--moe-gain` reaches the residual add through `launch_moe_acc_drain`'s gain
//! multiply on MoE layers, and dense layers take the unscaled `launch_vadd`; its last caller
//! was a Vulkan stub that existed only to satisfy this census. A test written to turn the
//! census green would have kept a dead kernel compiled and called that coverage — the exact
//! substitution of a name for a check that the census exists to prevent. Deleted 2026-08-06,
//! kernel and launcher together, along with the two comments on `launch_vadd` and
//! `--moe-gain` that asserted a live path and kept it looking reachable.
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

use rivoli_backend::hip::{
    launch_append_kv, launch_embed_i8_row, launch_flag_nonfinite, launch_gather_rope,
    launch_hash_rows,
};
use rivoli_core::num::{E4M3_BLOCK, E4M3_MAX, f32_to_bf16, f32_to_e4m3};

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
    let fire = |x: &[f32], tag: u32, flag: &mut rivoli_engine::device::DeviceBuf| -> u32 {
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

/// `hash_rows` — the `--divergence-log` fold, scored against its host twin.
///
/// **The instrument gets a gate of its own, and P7 is the whole argument.** Every conclusion
/// the GLM-nondeterminism investigation reaches is read off a pair of these hashes; a fold
/// that were subtly wrong would produce confident coordinates pointing at nothing, and no
/// other test in the tree would notice. `rivoli_core::hash::xor_fold` is the reference,
/// and it is a REFERENCE rather than a second copy: `probe::Probe` never computes a hash on
/// the host, so there is one implementation of the fold on each side of the ABI and this test
/// is the only thing that makes them agree.
///
/// Four assertions, each pinning a property the instrument's usefulness depends on:
///
/// 1. **Bit-exact against the host fold**, over a length that is NOT a multiple of the 256
///    block so the `i >= n` guard is exercised. This is the correctness assertion.
/// 2. **A one-ULP change in ONE element moves the hash.** Without this, assertion 1 is
///    satisfied by a fold that ignores its input entirely (e.g. an XOR of indices), and the
///    probe would report "identical" for every divergence there is. This is the
///    anti-vacuity assertion, and one ulp is the resolution the investigation needs: a
///    fixed-point accumulator differing by one count is exactly the perturbation being
///    hunted.
/// 3. **A PERMUTATION of the same values moves the hash.** This is what the index mix-in
///    buys, and it matters because XOR is self-inverse: without the index, two elements
///    holding the same bit pattern would cancel out of the fold and a reordering would be
///    invisible.
/// 4. **Folding the same array twice returns to the starting value.** The fold's
///    self-inverse property IS why `Probe::drain` must re-zero the slab, so the property is
///    pinned here rather than left as a comment there.
#[test]
fn hash_rows_matches_the_host_fold() {
    // 1000, deliberately: 3 full 256-blocks plus 232, so the last block's `i >= n` guard
    // decides the result. A multiple of 256 would leave that branch untested.
    let n = 1000;
    let mut r = Lcg(0x8AD5);
    let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
    // ONE launch site. Spelling the four-argument call twice made the two bodies identical text,
    // which the duplication gate reported the moment the `stream` parameter widened them.
    let launch = |x: *const f32, n: usize, out: *mut u64| {
        // SAFETY: the callers pass a live device f32 span of `n` and one live device u64.
        ok(
            unsafe { launch_hash_rows(x, n, out, rivoli_backend::NULL_STREAM) },
            "hash_rows",
        );
    };
    let fold = |v: &[f32]| -> u64 {
        let xb = dev(&f32b(v));
        let mut out = zeros(8);
        launch(xb.ptr() as *const f32, v.len(), out.ptr_mut() as *mut u64);
        u64le(&back(&out))
    };

    let want = rivoli_core::hash::xor_fold(&x);
    assert_eq!(
        fold(&x),
        want,
        "hash_rows must equal the host fold bit for bit"
    );

    let mut ulp = x.clone();
    ulp[617] = f32::from_bits(x[617].to_bits() ^ 1);
    assert_ne!(
        rivoli_core::hash::xor_fold(&ulp),
        want,
        "a ONE-ULP change in one element must move the fold — a probe blind to that is \
         blind to the divergence it exists to localise"
    );
    assert_eq!(
        fold(&ulp),
        rivoli_core::hash::xor_fold(&ulp),
        "and the kernel must agree with the host on the perturbed array too"
    );

    let mut swapped = x.clone();
    swapped.swap(4, 900);
    assert_ne!(
        fold(&swapped),
        want,
        "a permutation of the same values must move the fold (this is what mixing the \
         INDEX in buys — XOR is self-inverse)"
    );

    // Folding twice into the same slot cancels. This is the property `Probe::drain`'s
    // re-zero exists for, so it is asserted rather than asserted-in-prose.
    let xb = dev(&f32b(&x));
    let mut twice = zeros(8);
    for _ in 0..2 {
        launch(xb.ptr() as *const f32, n, twice.ptr_mut() as *mut u64);
    }
    assert_eq!(
        u64le(&back(&twice)),
        0,
        "XOR is self-inverse: two folds of one array must cancel, which is why an \
         un-zeroed slot silently corrupts the next token's hash"
    );
}

/// First 8 bytes of `b` as a little-endian u64.
///
/// Review asked for this to be inlined as `u64::from_le_bytes(b[..8].try_into().expect(..))`
/// (2026-08-17). It cannot be: `expect_used` is deny-level workspace-wide and this file carries no
/// `#![allow]` for it, so the inline form does not compile. `copy_from_slice` panics on a length
/// mismatch without going through `Result`, which is the same loudness with no lint to opt out of.
fn u64le(b: &[u8]) -> u64 {
    let mut w = [0u8; 8];
    w.copy_from_slice(&b[..8]);
    u64::from_le_bytes(w)
}
