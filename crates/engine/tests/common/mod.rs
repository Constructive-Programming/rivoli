//! The device-free half of the kernel-oracle scaffolding — the GENERIC slice of the old
//! tree's tests/common/mod.rs (lines 1-737 there). The V4-checkpoint helpers that shared
//! the file (Checkpoint/Oracle wiring, `/var/db/rivoli/...` paths) arrive with M8, their
//! first consumer in this tree.
//!
//! Originally shared by ten test binaries:
//! `docs`, `invariants`, `kernel`, `f4_attn`, `kvcompress_kernel`, `kvcompress_probe`,
//! `headtail`, `blockindex_kernel`, `f4_kernel` and `v4_oracle`.
//!
//! It was copy-pasted per file until 2026-07-30, and the copies had already started to
//! drift: two spellings of the same `Lcg` bug note, two `assert_close` bodies with the
//! same tolerance, and `f16b`/`u16b` present in one file and re-derived in the other.
//!
//! **A helper that touches a device TYPE is `#[cfg(feature = "rocm")]`-gated here, not
//! banished to the file that owns it.**
//!
//! > **CORRECTED 2026-08-06, reconciling two same-day corrections that disagreed.** The old
//! > rule — device types stay in the owning test file — was argued from the two backends:
//! > `dev` was `DeviceBuf` under HIP and `Buf` under Vulkan, "and that difference is the
//! > point of having two files". Vulkan was retired 2026-08-06, so that argument names
//! > something that no longer exists.
//! >
//! > Two agents corrected this independently within hours and landed on opposite answers.
//! > One deleted the rule and moved `dev`/`zeros`/`back`/`ok` here. The other kept it,
//! > re-derived on stronger ground: this module compiles into EVERY binary listed above, and
//! > `docs` and `invariants` are GPU-free registry checks, so a device type here would put
//! > both behind a device.
//! >
//! > **The second argument is right and the first is safe, because of a fact neither stated:
//! > the moved helpers are `#[cfg(feature = "rocm")]`.** Verified — `cargo check --test docs`
//! > featureless, with them present, is 0 errors. So they can live here, and the constraint
//! > that survives is the gate, not the location. Move a device helper here ungated and you
//! > break `docs` and `invariants`, which is the failure the second agent predicted.
//!
//! **Five older oracle files still spell their own uploader** (`f4_kernel`, `f4_attn`,
//! `headtail`, `kvcompress_kernel`, `blockindex_kernel`) and survive the duplication
//! gate only because their `.expect` strings differ — a half-migration, not a design.
//!
//! `dead_code` is allowed because this module is compiled into EACH test binary and none
//! uses every helper. The alternative is per-consumer cfg gates on a test utility, which is
//! more machinery than the warning is worth.
#![allow(dead_code)]
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use rivoli_artifact::quant::{Fp8W, RowScaledW};

// (`walk` stayed with its consumers — the cli crate's registry meta-gates own the one
// copy; the kernel-oracle scaffolding here never walks the tree.)

/// `1/sqrt(mean(x²) + eps)` — the factor all three norm forms share.
pub fn rms_inv(x: &[f32], eps: f32) -> f32 {
    1.0 / (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps).sqrt()
}

// **MOVED HERE from `glimmer_chain.rs` 2026-08-13**, the third helper to make this trip and for
// the same reason each time: a second host-side reader appeared (`glimmer_reference.rs`, which
// needs it to recover which embedding row the reference used) and jscpd reported the copy. The
// formula is `glimmer-architecture.md` §5's weightless norm; sharing it between two files that
// both transcribe the reference does not weaken either — what would weaken them is transcribing
// the CHAIN twice, and neither does that.
/// The weightless norm's shape and constants: `rows` segments of `d`, `eps` inside the mean,
/// `scale` after it.
///
/// One value rather than four trailing arguments, on [`Mla`]'s argument below about its own six:
/// `(rows, d)` and `(eps, scale)` are each interchangeable to the type checker, and a transposed
/// pair moves the reference and the thing it scores together, so the comparison still agrees.
#[derive(Clone, Copy)]
pub struct Weightless {
    pub rows: usize,
    pub d: usize,
    pub eps: f32,
    pub scale: f32,
}

impl Weightless {
    /// The weightless form over `rows` segments of `d`, times `scale`. The QK-norm (per head) and
    /// the embedding norm (`rows = 1`, `d = hidden`, `scale = 1`) are the same operator.
    pub fn apply(self, x: &mut [f32]) {
        for r in 0..self.rows {
            let seg = &mut x[r * self.d..(r + 1) * self.d];
            let f = self.scale * rms_inv(seg, self.eps);
            seg.iter_mut().for_each(|v| *v *= f);
        }
    }
}

/// The sliding window's lower bound for a query at absolute position `pos`: `[pos - win + 1, pos]`,
/// INCLUSIVE of `pos` itself, and 0 on a global layer. Trap 14 is the `+ 1`.
///
/// **Shared, because two copies of a bound is how one of them drifts.** `glimmer_attend.rs` (host
/// oracle and mask comparison) and `glimmer_chain.rs` (the loop's reference) both need it, and
/// jscpd caught the second copy the moment it was written. The kernel has its own, in HIP, which
/// is the implementation those files exist to check — this is deliberately not that one.
pub fn window_lo(pos: usize, win: usize) -> usize {
    if win > 0 && pos >= win {
        pos - win + 1
    } else {
        0
    }
}

/// The REFERENCE side of a comparison — the tensor every tolerance in this module is scaled by.
///
/// A newtype, with [`Got`] as its opposite, because this module spells the pair in **both**
/// orders: `rel(got, want)` and [`worst_rel`]`(got, want)` against [`report`]`(want, got)`,
/// [`err_tol`]`(want, got)` and every `assert_*`. Both sides are `&[f32]`, so a swap compiles —
/// and it is not cosmetic: every bound here is `…max_abs(want)`, so a swapped pair scales the
/// tolerance by the KERNEL's own output and the gate ends up grading itself. Wrapped, the swap is
/// an `E0308` instead of a green run.
#[derive(Clone, Copy)]
pub struct Want<'a>(pub &'a [f32]);

/// The MEASURED side of a comparison — the thing under test. See [`Want`] for why it is a newtype.
#[derive(Clone, Copy)]
pub struct Got<'a>(pub &'a [f32]);

// **MOVED HERE from `glimmer_fixture.rs` 2026-08-13**, for the reason `window_lo` moved the same
// day and one commit earlier: a test binary that includes only this module cannot reach that one,
// so `glimmer_chain.rs` wrote its own scorer — and reintroduced the NaN trap the history below
// records, for the THIRD time. A guard that lives where half the callers cannot see it is a guard
// with a hole in it.
/// `max|got - want| / max|want|` — **the metric every Glimmer tolerance is stated in**, and the one
/// `glimmer_anchor_driver.py::by_operator` computes to produce the floors. Stated once, here,
/// because a fixture that scores against a row in a different metric is comparing two numbers that
/// are not the same quantity.
///
/// Scaled by the reference side's own magnitude, once per tensor, not per element: a per-element
/// ratio divides one rounding error by another wherever the reference is near zero.
pub fn worst_rel(got: Got, want: Want) -> f32 {
    let (got, want) = (got.0, want.0);
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().copied().fold(0.0f32, |m, w| m.max(w.abs()));
    // An all-zero reference has no scale to divide by; any difference is then infinitely relative,
    // and reporting infinity is more honest than dividing by an epsilon.
    if scale == 0.0 {
        return if got.iter().all(|g| *g == 0.0) {
            0.0
        } else {
            f32::INFINITY
        };
    }
    // A non-finite `got` is INFINITY, checked BEFORE the max — `f32::max` returns the other
    // argument when one side is NaN, so the fold below silently discards every NaN difference
    // and an all-NaN kernel output would otherwise score 0.0, a perfect match. That is not
    // hypothetical: a broken kernel in this repo once passed 9 of 9 comparisons that way
    // (2026-08-05), and a review found this helper reintroducing the trap on 2026-08-12.
    if got.iter().any(|g| !g.is_finite()) {
        return f32::INFINITY;
    }
    // The SAME trap on the reference side, and it needs a different answer. `scale` above is
    // another `f32::max` fold, so a NaN in `want` is silently skipped there too — but returning
    // INFINITY would report it as the kernel being wrong, which is a diagnosis of the wrong side.
    // Added 2026-08-12 when the chain gates put golden bytes on this side of a score for the first
    // time; `glimmer_anchor.rs` asserts the captures are finite, so this fires only if that gate
    // and this one disagree, and then the message has to say so.
    assert!(
        want.iter().all(|w| w.is_finite()),
        "the REFERENCE side holds a non-finite value — this is a corrupt or mis-read capture, not \
         a kernel result"
    );
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
        / scale
}

/// f32 slice → little-endian bytes, the form every device upload takes.
pub fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u16 slice → little-endian bytes (bf16 scales, fp16 codebooks, roped keys).
pub fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// f32 → fp16 bytes — the VQ codebook is uploaded fp16 (the kernel decodes `__half`),
/// while the CPU reference keeps the f32 codebook, so these oracles measure exactly the
/// fp16 codebook-rounding error against the tol.
pub fn f16b(v: &[f32]) -> Vec<u8> {
    u16b(
        &v.iter()
            .map(|&x| rivoli_core::num::f32_to_f16(x))
            .collect::<Vec<_>>(),
    )
}

/// One `WMat::Dense` weight as the bf16 codes a kernel decodes with `bf16f`.
///
/// Asserts the round-trip is EXACT rather than assuming it. The checkpoint stores these in
/// bf16 and `Checkpoint::dense` widens them to f32, so re-encoding must be lossless — if it
/// ever is not, the kernel is being fed a different matrix from the oracle and every
/// comparison downstream silently measures that instead of the arithmetic.
///
/// Here rather than in one test file because two suites now upload the compressor's
/// `wkv`/`wgate` — `kvcompress_kernel.rs` at the real checkpoint and `f4_attn.rs` at the toy
/// — and `build.rs`'s duplication gate sees a second copy.
pub fn bf16_rows(w: &rivoli_oracles::v4oracle::weights::WMat) -> Vec<u16> {
    let (rows, cols) = (w.rows(), w.cols());
    let mut out = Vec::with_capacity(rows * cols);
    let mut buf = Vec::new();
    for r in 0..rows {
        w.row(r, &mut buf);
        for &v in &buf {
            let code = rivoli_core::num::f32_to_bf16(v);
            assert_eq!(
                rivoli_core::num::bf16_to_f32(code),
                v,
                "compressor weight row {r} is not bf16-exact: the oracle and the kernel \
                 would be reading different numbers"
            );
            out.push(code);
        }
    }
    out
}

/// `(cos, sin)` pairs flattened to the `[pos][2*i], [pos][2*i+1]` layout every V4 rotary
/// consumer indexes — `compress_finish_row` on the device and `Io::freqs` in `attn::v4`.
pub fn flat_freqs(t: &[(f32, f32)]) -> Vec<f32> {
    t.iter().flat_map(|&(c, s)| [c, s]).collect()
}

/// Little-endian bytes → f32 vec, the inverse of [`f32b`] for readback.
///
/// Delegates to the engine's own decoder rather than repeating it: an oracle that read
/// bytes back differently from the code under test could agree with itself while both were
/// wrong about the file format.
pub fn f32v(b: &[u8]) -> Vec<f32> {
    rivoli_artifact::quant::read_f32(b)
}

/// Little-endian bytes → fixed-width words, `f` being the `from_le_bytes` that decodes one.
///
/// ONE body for both widths, and it is load-bearing rather than tidy. `chunks_exact(N)
/// .map(from_le_bytes)` is already spelled in `quant.rs::read_f32` and in `bin/convert.rs`'s
/// VQ encoder; writing [`u16v`] and [`u32v`] out as two more bodies takes the tree from
/// **0 clones to 2** under `build.rs`'s gate. Measured both ways on 2026-08-06, because a
/// reviewer proposed exactly that simplification.
///
/// What makes it worth a comment is where jscpd points: the two clones it reports are
/// `quant.rs`<->`v4oracle/weights.rs` and `quant.rs`<->`bin/convert.rs` — **neither names
/// this file.** The copies here are the members that tip an existing pair over the
/// threshold, so the gate sends you to `src/` for a duplicate you introduced in `tests/`.
fn le_words<const N: usize, T>(b: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
    b.chunks_exact(N)
        .map(|c| f(c.try_into().expect("chunks_exact yields exactly N")))
        .collect()
}

/// Little-endian bytes → u16 vec — bf16 key caches, fp16 codebooks, VQ indices.
pub fn u16v(b: &[u8]) -> Vec<u16> {
    le_words(b, u16::from_le_bytes)
}

/// Little-endian bytes → u32 vec — the non-finite flag and `index_topk`'s row set.
pub fn u32v(b: &[u8]) -> Vec<u32> {
    le_words(b, u32::from_le_bytes)
}

// ---------------------------------------------------------------------------------------
// Device scaffolding. See the CORRECTED note in this module's header for why it is no
// longer kept out of here.
// ---------------------------------------------------------------------------------------

// (The Glimmer-config-typed fixture helper that lived here arrives at M7 with its
// consumer tests; its parameter type is not ported yet.)

#[cfg(feature = "rocm")]
pub fn dev(b: &[u8]) -> rivoli_engine::device::DeviceBuf {
    let mut d = rivoli_engine::device::DeviceBuf::new(b.len().max(1)).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}

/// The device buffer type every helper here returns and every oracle file names.
///
/// Re-exported rather than imported per file: `use rivoli_engine::device::DeviceBuf;` sat
/// directly above `mod common;` in three of the four oracle files, and that five-line run of
/// boilerplate is a jscpd clone with nothing in it worth sharing. Named through the module
/// that hands you `dev()` instead, it joins the `use common::{…}` list those files already have.
// `#[allow(unused_imports)]` for the reason the module header gives `dead_code`: this compiles
// into EVERY test binary and most of them never name the buffer type. A re-export nobody in a
// given binary uses is that binary's business, not a defect — and the allow is on this ONE item
// rather than the file, so a genuinely dead `use` elsewhere still reports.
#[allow(unused_imports)]
#[cfg(feature = "rocm")]
pub use rivoli_engine::device::DeviceBuf;

/// A zeroed device buffer of `n` bytes — a kernel destination.
///
/// ZEROED rather than uninitialised, and load-bearing for the oracles that compare a
/// destination the kernel only PARTLY writes — `append_kv` fills one row of a five-row slab,
/// `index_pool_push` one block of a three-block pool. The untouched remainder is asserted
/// too, so a wrote-the-wrong-row defect shows up as a mismatch against zero rather than as
/// noise.
#[cfg(feature = "rocm")]
pub fn zeros(n: usize) -> rivoli_engine::device::DeviceBuf {
    dev(&vec![0u8; n])
}

/// One `gemm_bf16` launch's operands and dims, and the one place they are spelled.
///
/// `glimmer_fixture.rs`'s `gemv_bf16` and `glimmer_residency.rs`'s fence gate both drive this
/// kernel and jscpd matched their call blocks (2026-08-12). The gate was right about the substance
/// too: seven positional arguments where `n` and `k` are both bare `usize` is a place a mistake is
/// a wrong answer rather than a compile error — so the six that describe the operands are named
/// fields here, and [`gemm_bf16_launch`] takes this and the stream.
#[cfg(feature = "rocm")]
#[derive(Clone, Copy)]
pub struct GemmBf16 {
    pub x: *const f32,
    pub w: *const u16,
    pub out: *mut f32,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// One `gemm_bf16` launch.
///
/// > **Two things were tried first and are recorded so they are not tried again.** Hoisting it here
/// > UNGATED is an `E0433` on the featureless build — this module compiles into `docs` and
/// > `invariants`, which are GPU-free, and `rivoli::backend` is `rocm`-gated; that is the exact
/// > failure the module header above predicts, and it reached a review rather than a run.
/// > Hoisting the CASTS at the call site instead, to make the two blocks structurally unalike and
/// > delete this helper, does NOT satisfy jscpd — measured, still 29 tokens matched.
///
/// # Safety
/// `g.x` is `g.m * g.k` live f32, `g.w` is `g.n * g.k` live u16, `g.out` is `g.m * g.n` writable
/// f32, none aliasing another, all live until the caller's next `device_sync`.
#[cfg(feature = "rocm")]
pub unsafe fn gemm_bf16_launch(g: GemmBf16, stream: *mut std::ffi::c_void) {
    // SAFETY: the caller's contract above.
    unsafe { rivoli_backend::hip::launch_gemm_bf16(g.x, g.w, g.out, g.m, g.n, g.k, stream) }
        .expect("gemm_bf16 launch");
}

/// A launch that must succeed, with the launcher named.
///
/// Every oracle here passes dims the kernel's own guards accept, so an `Err` is a guard that
/// MOVED, not a case to handle.
///
/// It exists for `fwd_kernel.rs` and `indexer_kernel.rs` specifically, which is why the
/// older oracle files still use `.expect` under their own blanket allow. Those two gained
/// the same allow with the same three-line preamble and `build.rs`'s duplication gate
/// rejected it — a clone produced by suppressing a lint instead of removing its cause. This
/// removes the cause, and `expect_used = "deny"` stays live in both files.
///
/// `{e:#}` so an `anyhow` chain prints its causes; a launcher's guard code is in the
/// innermost one and `{e}` would drop it.
#[cfg(feature = "rocm")]
pub fn ok<T>(r: anyhow::Result<T>, what: &str) -> T {
    r.unwrap_or_else(|e| panic!("{what} refused the launch: {e:#}"))
}

/// Join the device, then read a buffer back. The join is HERE rather than at each call
/// site because forgetting it reads the destination before the kernel has written it,
/// which fails as a wrong ANSWER rather than as a missing sync — the most expensive
/// possible spelling of the mistake.
#[cfg(feature = "rocm")]
pub fn back(d: &rivoli_engine::device::DeviceBuf) -> Vec<u8> {
    rivoli_backend::hip::device_sync().expect("device_sync");
    d.copy_out().expect("copy_out")
}

/// Assert two slices are BIT-IDENTICAL, reporting the first disagreement and how many
/// there are.
///
/// For the kernels with **one thread per output element and no reduction**, where exact
/// agreement with a host transliteration is a property of the arithmetic rather than luck.
/// Anything that reduces gets [`assert_rel`] instead — or [`assert_close`] where its shared
/// `1e-3·max + 1e-3` floor is honest for the fixture's scale. Measured on this tree, a
/// correct wave-reduced kernel differs from its oracle on ~0.08% of bf16 elements at dim
/// 4096, so a bitwise gate there rejects correct code.
///
/// Prints the element count on success: "identical" over 4 elements and over 4096 are not
/// the same evidence.
pub fn assert_bitwise<T: PartialEq + std::fmt::Debug>(want: &[T], got: &[T], label: &str) {
    assert_eq!(want.len(), got.len(), "{label}: length");
    let bad: Vec<usize> = (0..want.len()).filter(|&i| want[i] != got[i]).collect();
    match bad.first() {
        None => println!("{label}: {} elements, bit-identical", want.len()),
        Some(&i) => panic!(
            "{label}: {} of {} elements differ; first at {i}: want {:?}, got {:?}",
            bad.len(),
            want.len(),
            want[i],
            got[i]
        ),
    }
}

/// [`assert_bitwise`] over f32 BIT PATTERNS.
///
/// Not `assert_bitwise(want, got)` directly on the floats: `PartialEq` for f32 says
/// `-0.0 == 0.0`, so a sign-dropping defect passes an assertion that claims exactness, and
/// says `NaN != NaN`, so a NaN-poisoned buffer fails one for the wrong reason. Five call
/// sites spelled the `.to_bits()` fold on both operands; `tests/f4_kernel.rs` and
/// `tests/hadamard_basis.rs` each keep a private `bits()` for the same reason.
pub fn assert_bits(want: &[f32], got: &[f32], label: &str) {
    let b = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_bitwise(&b(want), &b(got), label);
}

// **MOVED HERE from `kernel.rs` 2026-08-15**, when the MoE expert-range oracles left it for
// `kernel_moe.rs`: the batched-row claim and the guard-code assertion are each made on BOTH
// sides of that split, and a second copy of either is what `build.rs`'s duplication gate is
// for. `assert_out` did NOT come — it is `DeviceBuf`-typed and only the file that kept the
// GEMV/MLA destinations still calls it.
/// Both rows of a two-row batch against their own single-row runs, named per row.
///
/// Row 0 alone would pass a kernel that batches correctly but leaks row 0's input into row 1
/// (a missing `r * stride`), so both rows are asserted and the message says which failed.
pub fn assert_rows<T: PartialEq + std::fmt::Debug>(got: &[T], want: &[Vec<T>], w: usize, k: &str) {
    assert_eq!(got[..w], want[0][..], "{k} row 0 must be bit-identical");
    assert_eq!(got[w..], want[1][..], "{k} row 1 must be bit-identical");
}

/// A launcher result against an expected guard code: `None` must be ACCEPTED, `Some(n)`
/// rejected with `n` somewhere in the message.
///
/// The CODE is asserted rather than merely `is_err`, and that is the whole value of these
/// tests: one that accepted any error would still pass if someone replaced a power-of-two
/// check with `block != 128`, or if an unrelated dimension guard started swallowing the
/// case first.
///
/// (That paragraph sat on [`assert_rows`] in `kernel.rs` — two doc blocks stacked on one
/// function, describing two. It is re-anchored, not rewritten.)
pub fn assert_guard<T: std::fmt::Debug>(r: anyhow::Result<T>, want: Option<u32>, what: &str) {
    match want {
        None => assert!(r.is_ok(), "{what}: {r:?}"),
        Some(code) => {
            let msg = format!("{:#}", r.expect_err("expected a guard rejection"));
            assert!(
                msg.contains(&code.to_string()),
                "{what}: want guard {code}, got {msg:?}"
            );
        }
    }
}

/// `golden.rs::Diff.rel` — max absolute disagreement over the largest expected magnitude.
///
/// The metric the oracle's own gate uses, so an anti-vacuity arm here is scored the same way
/// the goldens are. Moved out of `tests/headtail.rs` on 2026-08-06 when
/// `tests/indexer_kernel.rs` reimplemented it under another name.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    // Length first: `zip` truncates, so a short `got` would score 0.0 and read as perfect
    // agreement.
    assert_eq!(
        got.len(),
        want.len(),
        "comparing tensors of different length"
    );
    max_err(Want(want), Got(got)) / max_abs(Want(want)).max(1e-30)
}

/// [`report_rel`] promoted to an assertion, for oracles whose agreement is far tighter
/// than [`err_tol`]'s `1e-3·max + 1e-3` floor and would pass on two orders of headroom
/// under it.
///
/// Takes its ratio per call rather than sharing one: a single `TOL` shared across a file is
/// how a widening made for one comparison silently degrades every other, and these
/// oracles' honest tolerances differ by four orders of magnitude — `swiglu` is one `expf`
/// apart from the host, `index_head_route` is an LDS tree reduction.
pub fn assert_rel(want: &[f32], got: &[f32], label: &str, ratio: f32) {
    let (err, tol) = report_rel(Want(want), Got(got), label, ratio);
    assert!(
        err <= tol,
        "{label}: err={err:.3e} > tol={tol:.3e} (rel={ratio:.1e} of max={:.3e})",
        max_abs(Want(want))
    );
}

/// Report the max error AND the threshold it was compared against. Printing BOTH is the
/// point: a green oracle that passed on 100x of headroom looks exactly like one that passed
/// on 2x, and only one of them is evidence of anything.
pub fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = report(Want(want), Got(got), label);
    assert!(
        err <= tol,
        "{label}: err={err:.3e} > tol={tol:.3e} max={:.3e}",
        max_abs(Want(want))
    );
}

/// The largest magnitude in a slice — the scale every tolerance in this suite is stated
/// against.
///
/// Extracted because a second tolerance FORMULA now exists: `tests/f4_kernel.rs` bounds
/// relative to the bf16 quantum instead of [`err_tol`]'s `1e-3·max + 1e-3`, whose absolute
/// floor is 5% of the signal at that fixture's scale. The formulas differ on purpose; the
/// SCALE they are stated against must not, and three copies of this fold were the duplicate
/// the gate found.
///
/// Takes [`Want`] rather than a bare slice because every caller here folds the REFERENCE: a
/// tolerance scaled by the measured side is a gate that grades itself.
pub fn max_abs(v: Want) -> f32 {
    v.0.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// [`err_tol`] plus the comparison line, returning the pair so the caller decides what a
/// failure means. It had two callers with two answers — [`assert_close`] panics, and the
/// retired `vk.rs`'s `Shapes::close` recorded and kept going. The PRINT is what they shared,
/// and a second copy of the format string is a second format.
pub fn report(want: Want, got: Got, label: &str) -> (f32, f32) {
    let (err, tol) = err_tol(want.0, got.0);
    report_line(
        label,
        Scored {
            err,
            tol,
            mx: max_abs(want),
        },
    )
}

/// [`report`] against a tolerance RELATIVE to the largest expected element, for callers
/// whose signal is too small for [`err_tol`]'s `1e-3` absolute floor to mean anything —
/// `tests/f4_kernel.rs`, where one routed MoE layer's output is ~2e-2 and that floor would
/// be 5% of it.
///
/// Takes the ratio and computes the metric itself. The `(err, tol, mx)` an earlier version took
/// bare is now [`Scored`], which carries that argument.
pub fn report_rel(want: Want, got: Got, label: &str, rel: f32) -> (f32, f32) {
    let mx = max_abs(want);
    report_line(
        label,
        Scored {
            err: max_err(want, got),
            tol: rel * mx,
            mx,
        },
    )
}

/// The three numbers a comparison line prints: the error, the bound it was held to, and the
/// scale both are stated against.
///
/// One value rather than three positional `f32`, and the argument is [`report_rel`]'s own, moved
/// here with the fields it is about: an earlier version of that function took `(err, tol, mx)`
/// bare — three interchangeable `f32`s, where swapping the first two turns the caller's
/// `err <= tol` into `tol <= err`, a gate that goes green on every failure. That is this module's
/// argument about six bare `usize` in a row, made about `f32`; naming the fields answers it.
#[derive(Clone, Copy)]
struct Scored {
    err: f32,
    tol: f32,
    mx: f32,
}

/// The comparison LINE, given an error and whatever bound the caller holds it to. Named for
/// what it emits: it was `report_margin` until 2026-08-05 and the margin is gone. Private:
/// [`report`] and [`report_rel`] are the two ways in, and a third caller would be a third
/// tolerance with no argument attached to it.
///
/// **Prints `err` and `tol` side by side, not a ratio.** It printed `margin = tol/err`
/// until 2026-08-05, and that number is pathological at both ends of its range: a bit-exact
/// result rendered as `margin=532543503195029799199619132512272384.0x`, which reads as
/// corruption rather than as the best possible outcome, and a deliberate-break test — where
/// passing means err EXCEEDS tol — rendered as `margin=0.0x`, which reads as failure beside
/// a green test. Two numbers the reader compares themselves have neither pathology, and the
/// distance is still on the page.
fn report_line(label: &str, s: Scored) -> (f32, f32) {
    let Scored { err, tol, mx } = s;
    println!("{label}: err={err:.3e} tol={tol:.3e} max={mx:.3e}");
    (err, tol)
}

/// `(max abs error, tolerance)` for a want/got pair — the shared arithmetic behind
/// [`assert_close`] and [`report`]. Two copies of a tolerance formula is two tolerances.
pub fn err_tol(want: &[f32], got: &[f32]) -> (f32, f32) {
    let (want, got) = (Want(want), Got(got));
    (max_err(want, got), 1e-3 * max_abs(want) + 1e-3)
}

/// The largest absolute disagreement between two slices — the error metric every
/// comparison in this suite uses, whatever tolerance it is held to.
fn max_err(want: Want, got: Got) -> f32 {
    want.0
        .iter()
        .zip(got.0)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
}

/// `n` positive per-block scales, `|f·0.1| + 0.01`. Every fp8 oracle draws them this way,
/// and the 0.01 floor is load-bearing rather than tidy: a tile whose
/// scale rounds to zero makes the comparison for that tile vacuous, and `assert_close`'s
/// relative tolerance would not show it.
pub fn block_scales(r: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| (r.f() * 0.1).abs() + 0.01).collect()
}

/// An fp8 GEMV case: e4m3 weights, the block-scale grid over them, the input, and the host
/// result.
///
/// `n_scales` was the caller's, not computed here, because the two backends spelled the scale
/// grid differently (`i_dim / block` against `i_dim.div_ceil(block)`); with one backend left it
/// was "a knob nothing turns and could be computed", noted here and **collapsed 2026-08-15**,
/// when CodeScene's excess-argument rule made the price of keeping it explicit. `div_ceil` on
/// BOTH axes, mirroring the kernel — the caller passed exactly this, and the ragged shape
/// (`o_dim = 130` at `block = 128`) is the one that made `o_dim / block` wrong.
///
/// The DRAW ORDER — weights, scales, x — is the part that has to be shared: it is what makes a
/// seed mean the same data at both call sites.
pub fn gemv_fp8_case(
    r: &mut Lcg,
    o_dim: usize,
    i_dim: usize,
    block: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let packed: Vec<u8> = (0..o_dim * i_dim)
        .map(|_| rivoli_core::num::f32_to_e4m3(r.f()))
        .collect();
    let scale = block_scales(r, o_dim.div_ceil(block) * i_dim.div_ceil(block));
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    rivoli_artifact::quant::matvec_fp8(&mut want, &x, Fp8W::new(&packed, &scale, block), i_dim);
    (packed, scale, x, want)
}

/// Random int8 weights and their per-row scales, drawn the way every int8 oracle here
/// draws them.
///
/// The `1e-4` floor is load-bearing for the same reason [`block_scales`]'s `0.01` is: a row
/// whose scale rounds to zero makes that row's comparison vacuous, and a relative tolerance
/// would not show it. The DRAW ORDER — weights, then scales — is what makes a seed mean the
/// same data at both call sites.
pub fn i8_weights(r: &mut Lcg, o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    let packed: Vec<u8> = (0..o_dim * i_dim)
        .map(|_| (r.f() * 127.0) as i8 as u8)
        .collect();
    let scale: Vec<f32> = (0..o_dim).map(|_| (r.f() * 0.01).abs() + 1e-4).collect();
    (packed, scale)
}

/// `matvec_i8` into a fresh `o_dim` vector. Returned rather than written through an
/// out-param so the caller binds it in one line; the two int8 oracles generate their
/// weights differently and share only this step.
///
/// `dims` is `matvec_i8`'s own `[o_dim, i_dim]` rather than two trailing `usize` — the pair is
/// passed straight through, and two bare interchangeable dimensions at the end of an argument
/// list is the defect [`Mla`] below exists to answer, made small (2026-08-15).
pub fn want_i8(x: &[f32], packed: &[u8], scale: &[f32], dims: [usize; 2]) -> Vec<f32> {
    let mut want = vec![0.0f32; dims[0]];
    rivoli_artifact::quant::matvec_i8(&mut want, x, RowScaledW::new(packed, scale), dims);
    want
}

/// The kv_b geometry both MLA launchers take.
///
/// Six bare `usize` in a row, every one of them plausible in any other's position, spelled
/// in an oracle, a launch wrapper and a guard closure PER BACKEND — five copies of the same
/// order, and a transposed pair would have moved the oracle and the kernel together. Pure
/// dimensions, so it belongs here rather than beside either backend's buffer type.
#[derive(Clone, Copy)]
pub struct Mla {
    pub h: usize,
    /// The q head stride. `mla_value_fp8` never reads q, so callers for that kernel leave this
    /// zero — cheaper than a second five-field shape whose only difference from this one is
    /// a field nothing reads.
    ///
    /// A `value_dims(h, nope, vh, kvl, block)` constructor did that zeroing until 2026-08-15.
    /// It was deleted rather than reshaped: five positional `usize` is the same excess-argument
    /// defect this struct exists to answer, and with the fields public `Mla { qh: 0, .. }` says
    /// it without an order to get wrong. [`Mla::new`] took six the same way and was reshaped to
    /// take ONE the same day — see its own note for why the fix was an array and not six named
    /// fields at every call site.
    pub qh: usize,
    pub nope: usize,
    pub vh: usize,
    pub kvl: usize,
    pub block: usize,
}

impl Mla {
    /// The six dims in kv_b order: `[h, qh, nope, vh, kvl, block]`.
    ///
    /// **One array argument, and the order is spelled HERE and nowhere else.** It took the six
    /// as six parameters until 2026-08-15, which is CodeScene's excess-argument rule at full
    /// size. The obvious fix — delete the constructor, write `Mla { h: 4, qh: 128, .. }` at each
    /// call site — was tried and REVERTED the same day: rustfmt's `struct_lit_width` is 18, so
    /// every literal wider than that becomes one line per field, and `kernel.rs`'s six MLA
    /// shapes (which differ in one dim at a time, deliberately) then shared four identical lines
    /// three ways. `build.rs`'s duplication gate reported it, correctly.
    ///
    /// So the array is not "positional again by accident". It keeps the six TOGETHER, names
    /// their order in one place, and leaves the public fields for a caller who wants
    /// `Mla { qh: 0, .. }` — which the value-kernel callers do.
    pub fn new(dims: [usize; 6]) -> Self {
        let [h, qh, nope, vh, kvl, block] = dims;
        Self {
            h,
            qh,
            nope,
            vh,
            kvl,
            block,
        }
    }

    /// kv_b's full row count, `h·(nope + vh)`.
    pub fn rows(self) -> usize {
        self.h * (self.nope + self.vh)
    }

    /// The two launcher guards both CPU oracles restate. An oracle that accepted a shape
    /// the launcher rejects would be checking the kernel against a case it can never run.
    ///
    /// `qh` is deliberately absent: the value kernel's callers leave it zero and
    /// `mla_value_fp8` never reads it, so the absorb oracle asserts it separately.
    pub fn assert_guarded(self) {
        let (h, nope, vh) = (self.h, self.nope, self.vh);
        let (kvl, block) = (self.kvl, self.block);
        assert!(
            h > 0 && nope > 0 && vh > 0 && kvl > 0 && block > 0,
            "guard 1001"
        );
        assert!(
            block.is_power_of_two(),
            "guard 1003: blk_shift needs a power-of-two tile"
        );
    }
}

/// The MLA attention's shape.
///
/// Five `usize` and an f32 that travel together through the split planner, the tile
/// widener, the CPU reference and the dispatch — and the reference and the dispatch take
/// the SAME six, so every test spelled them twice. A transposed pair would have moved both
/// sides identically and the comparison would still have agreed.
#[derive(Clone, Copy)]
pub struct Att {
    pub h: usize,
    pub nr: usize,
    pub kvl: usize,
    pub rope: usize,
    pub n_blocks: usize,
    pub scale: f32,
}

impl Att {
    /// Neither `n_blocks` nor `scale` is a free parameter, and both are derived here for the
    /// same reason.
    ///
    /// The fp8 latent cache carries one block scale per 128 latent dims, so `n_blocks`
    /// FOLLOWS from `kvl` and every test derived it the same way; deriving it once removes the
    /// only way a reference and a launcher could have been handed different block-scale
    /// strides for the same cache.
    ///
    /// The softmax `scale` follows from `kvl + rope` the same way. `kernel.rs` carried an
    /// `att(h, nt, kvl, rope)` wrapper that did nothing else, arguing that "five call sites
    /// spelling `1.0 / ((kvl + rope) as f32).sqrt()` is five places for it to drift from the
    /// kernel's" — true, and the wrapper left a fifth argument here that could still be handed
    /// a wrong number. **Folded in 2026-08-15** with the `kernel.rs` split, which deleted the
    /// wrapper: there is now no way to state a scale that does not follow from the shape. The
    /// guard test, which only asks whether a `kvl` is accepted, previously passed a literal
    /// `1.0` and now takes the derived value — it never reads the result.
    pub fn new(h: usize, nr: usize, kvl: usize, rope: usize) -> Self {
        Self {
            h,
            nr,
            kvl,
            rope,
            n_blocks: kvl / 128,
            scale: 1.0 / ((kvl + rope) as f32).sqrt(),
        }
    }
}

/// One MoE dispatch's geometry: the two matrix dims and the half-open expert range
/// `[e_start, e_start + e_count)`.
///
/// The same four, in the same order, in `moe_expert_range`'s wrapper and in both of the
/// VQ oracles that check it — three copies per backend of a list whose middle two entries
/// are interchangeable to the type checker.
#[derive(Clone, Copy)]
pub struct MoeRange {
    pub hidden: usize,
    pub inter: usize,
    pub e_start: usize,
    pub e_count: usize,
}

impl MoeRange {
    pub fn new(hidden: usize, inter: usize, e_start: usize, e_count: usize) -> Self {
        Self {
            hidden,
            inter,
            e_start,
            e_count,
        }
    }

    /// One past the last expert this range writes — the oracles size their staging by it.
    pub fn e_end(self) -> usize {
        self.e_start + self.e_count
    }
}

/// The oracles' deterministic input source.
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in [-1, 1).
    ///
    /// `>> 32`, not `>> 33`. The old shift kept only 31 bits, so dividing by `u32::MAX`
    /// gave [0, 0.5) and `*2 - 1` gave [-1, 0) — **every sample negative**, for the whole
    /// life of both test files. In a matvec oracle that makes every `x[i]*w[i]` product
    /// positive, so the partial sums GROW instead of cancelling: `mx` inflates, the
    /// `1e-3 * mx` relative tolerance inflates with it, and the oracles were passing on
    /// roughly two orders of magnitude of headroom. It also meant no oracle here had ever
    /// exercised floating-point cancellation — the only regime where summation order
    /// matters, and the entire reason the kernels reduce with a fixed shuffle ladder
    /// (`__shfl_down` / `wave_sum`) instead of an atomic.
    pub fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

/// MoE expert-range dispatch scaffolding, shared by `kernel_moe.rs` and
/// `kernel_moe_artifact.rs`.
///
/// **MOVED HERE from `kernel.rs` (via `kernel_moe.rs`) 2026-08-15**, when the MoE oracles
/// split into `kernel_moe.rs` (synthetic quantized fixtures, runs on any GPU) and
/// `kernel_moe_artifact.rs` (the shipped `.i4` set and its fp8 checkpoint, skips loudly
/// without them). `i4_launch_drain` is what both drive, and it drags in the whole chain
/// below — a second copy of any of it is what `build.rs`'s duplication gate is for.
///
/// It sits beside [`MoeRange`], which was already in the parent for the same reason, and
/// follows [`GemmBf16`]/[`gemm_bf16_launch`]: a device-typed launch operand and its wrapper
/// live here GATED, so the featureless registry binaries (`docs`, `invariants`) never see
/// them. **One `#[cfg]` on the module rather than nine on the items** — nine copies of the
/// attribute is nine identical token runs, and jscpd reported two of the structs as a clone
/// of each other on the strength of it.
#[cfg(feature = "rocm")]
pub mod moe {
    use super::{MoeRange, dev, f32b, f32v};
    use rivoli_backend::gpustream::HipStream;
    use rivoli_backend::hip::ExpertDesc;
    use rivoli_engine::device::DeviceBuf;

    /// The four per-dispatch buffers of one MoE expert range: the token rows, the gate weights,
    /// the per-expert `h` staging, and the fixed-point accumulator.
    ///
    /// Descriptors, codebooks and geometry are fixed for a whole test; these four are what a
    /// batched arm swaps. Bundled so the two batching tests can each drive their range through
    /// a closure taking ONE operand rather than the same five-parameter list written twice.
    pub struct MoeIo<'a> {
        x: &'a DeviceBuf,
        w: &'a DeviceBuf,
        h: &'a mut DeviceBuf,
        acc: &'a mut DeviceBuf,
    }

    impl<'a> MoeIo<'a> {
        pub fn new(
            x: &'a DeviceBuf,
            w: &'a DeviceBuf,
            h: &'a mut DeviceBuf,
            a: &'a mut DeviceBuf,
        ) -> Self {
            Self { x, w, h, acc: a }
        }

        /// The four device addresses, in launcher order. Consuming, because two of them are
        /// unique borrows and the address outlives the reborrow that produced it.
        pub fn ptrs(self) -> (*const f32, *const f32, *mut f32, *mut u64) {
            let (x, w) = (self.x.ptr() as *const f32, self.w.ptr() as *const f32);
            let acc = self.acc.ptr_mut() as *mut u64;
            (x, w, self.h.ptr_mut() as *mut f32, acc)
        }
    }

    /// What every int4 expert-range dispatch holds fixed for a whole test: the uploaded
    /// descriptor array and the stream it runs on. Only [`MoeIo`] and `nrow` change between arms.
    pub struct MoeCtx<'a> {
        descs: &'a DeviceBuf,
        stream: &'a HipStream,
    }

    impl<'a> MoeCtx<'a> {
        pub fn new(descs: &'a DeviceBuf, stream: &'a HipStream) -> Self {
            Self { descs, stream }
        }
    }

    /// `moe_expert_range_i4` over experts `[g.e_start, g.e_end())`.
    pub fn expert_range_i4(io: MoeIo<'_>, cx: &MoeCtx<'_>, g: MoeRange, nrow: usize) {
        let (x, w, h, acc) = io.ptrs();
        let d = cx.descs.ptr() as *const ExpertDesc;
        let (hidden, inter, st) = (g.hidden, g.inter, cx.stream.raw());
        // SAFETY: `x` is `nrow` rows of [hidden], `w` is [e_count·nrow], `h` [e_count·nrow·inter]
        // and `acc` `nrow` rows of [hidden] u64; the stream is live for the call.
        unsafe {
            rivoli_backend::hip::launch_moe_expert_range_i4(
                x, hidden, inter, g.e_start, g.e_count, d, w, h, acc, nrow, st,
            )
        }
        .expect("moe_expert_range_i4");
    }

    /// The drain's two buffers: the fixed-point accumulator it consumes and the f32 destination
    /// it writes. Always allocated and drained as a pair, and both are unique borrows.
    pub struct Drain<'a> {
        pub out: &'a mut DeviceBuf,
        pub acc: &'a mut DeviceBuf,
    }

    impl<'a> Drain<'a> {
        pub fn new(out: &'a mut DeviceBuf, acc: &'a mut DeviceBuf) -> Self {
            Self { out, acc }
        }
    }

    /// One `moe_acc_drain` over row `row` of the accumulator.
    ///
    /// `row` is what lets a batched arm drain its rows with the same launch the single-row arms
    /// use — the drain itself is always single-row.
    pub fn drain(d: Drain<'_>, row: usize, hidden: usize, stream: &HipStream) {
        // SAFETY: `row` is inside both buffers, which every caller sizes for it; the stream is
        // live for the call.
        unsafe {
            rivoli_backend::hip::launch_moe_acc_drain(
                d.out.ptr_mut().add(row * hidden * 4) as *mut f32,
                d.acc.ptr_mut().add(row * hidden * 8) as *mut u64,
                hidden,
                1,
                1.0,
                stream.raw(),
            )
        }
        .expect("moe_acc_drain");
    }

    /// The descriptor ARRAY on device — the addresses themselves, uploaded verbatim.
    pub fn desc_buf(descs: &[ExpertDesc]) -> DeviceBuf {
        // SAFETY: `ExpertDesc` is plain pointers, and the span is exactly the slice's own bytes.
        dev(unsafe {
            std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(descs))
        })
    }

    /// One MoE expert's two matrix dims. Both are bare `usize` and each is plausible in the
    /// other's position at every launcher, oracle and buffer size that takes the pair.
    #[derive(Clone, Copy)]
    pub struct Dims {
        pub hidden: usize,
        pub inter: usize,
    }

    impl Dims {
        pub fn new(hidden: usize, inter: usize) -> Self {
            Self { hidden, inter }
        }
    }

    /// The three MoE destination buffers for `nrow` token rows: per-expert `h` staging, the
    /// fixed-point accumulator, and the f32 output.
    ///
    /// ONE u64 accumulator row per token, not `e` partial rows; the output starts at zero
    /// because the drain ADDS into it — it is the residual add.
    pub fn moe_bufs(e: usize, nrow: usize, d: Dims) -> (DeviceBuf, DeviceBuf, DeviceBuf) {
        let z = |n: usize| dev(&vec![0u8; n]);
        (
            z(e * nrow * d.inter * 4),
            z(nrow * d.hidden * 8),
            z(nrow * d.hidden * 4),
        )
    }

    /// Launch `[0, descs.len())` int4 experts ONE AT A TIME and drain — the tail every int4 MoE
    /// test shares, extracted because `gpu_i4_moe` and `gpu_i4_expert` had it verbatim and the
    /// duplication gate said so. Per expert rather than one range: bit-identical by
    /// `moe_expert_range`'s own argument (`e = e_start + row / inter`, every row independent).
    ///
    /// The caller must keep the buffers the descriptors point INTO alive across this call.
    pub fn i4_launch_drain(descs: &[ExpertDesc], x: &[f32], w: &[f32], d: Dims) -> Vec<f32> {
        let e = descs.len();
        let (descb, xb, wb) = (desc_buf(descs), dev(&f32b(x)), dev(&f32b(w)));
        let (mut hbuf, mut pbuf, mut obuf) = moe_bufs(e, 1, d);
        let stream = HipStream::new().expect("stream");
        let cx = MoeCtx::new(&descb, &stream);
        for k in 0..e {
            let io = MoeIo::new(&xb, &wb, &mut hbuf, &mut pbuf);
            expert_range_i4(io, &cx, MoeRange::new(d.hidden, d.inter, k, 1), 1);
        }
        drain(Drain::new(&mut obuf, &mut pbuf), 0, d.hidden, &stream);
        rivoli_backend::hip::device_sync().expect("sync");
        f32v(&obuf.copy_out().expect("out"))
    }
}

// ---------------------------------------------------------------------------------------
// V4-Flash checkpoint scaffolding, shared by `kvcompress_probe.rs` and
// `kvcompress_kernel.rs`
// ---------------------------------------------------------------------------------------
//
// These moved here from `kvcompress_probe.rs` when the kernel test needed the same loader:
// both drive the SAME two compressors (layer 2 at ratio 4, layer 3 at ratio 128), one
// against the oracle alone and one against the GPU, and a second copy of `compressor_w`
// would be a second set of shape assertions that could drift apart while both stayed green.
// `build.rs`'s duplication gate watches `tests/`, so it would also be a build error.
//
// jscpd:ignore-free by construction: the loader is spelled once and both suites call it.
