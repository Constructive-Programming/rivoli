//! **The head tail on the device, scored against the oracle.**
//!
//! Before this stage the first decode's logits were ungated by construction: `hc_head`, the
//! final `RMSNorm` and `ParallelHead` existed nowhere in the engine, so every per-layer
//! golden could be perfect and the sampled token still wrong.
//!
//! # Read this before tightening anything here
//!
//! **The comparison is deliberately NOT bit-exact, and that is a measured decision.** An
//! earlier revision of `tests/v4_oracle.rs` concluded from TOY dimensions that a device head
//! gate "must be bitwise". That is false at real ones:
//! `the_reassociation_floor_bounds_any_tolerance_these_goldens_can_have` measures a CORRECT
//! wave-reduced kernel differing from the oracle on ~0.08% of bf16 elements at `dim = 4096`,
//! because the oracle sums sequentially and `common.hpp::wave_sum` sums as a ladder. A
//! bitwise gate would reject correct code.
//!
//! What each half can promise is different, and the tests say which is which:
//! - `hc_head_collapse_blend` sums over `hc_mult` copies SEQUENTIALLY, in the oracle's own ORDER,
//!   so given the same gate vector it is exactly reproducible -- as an FMA reduction, which
//!   is what hipcc's default `-ffp-contract=fast` makes it. Asserted bitwise against the
//!   device's own `pre`, and against `mul_add` rather than `+ p * x`; see `kernels/headtail.hip`
//!   for the measurement.
//! - `hc_head_collapse_gate` reduces 16384 terms with `wave_sum`, so it re-associates and is
//!   compared under a tolerance.
//!
//! And one thing this suite CANNOT do, stated so nobody reads its green as more than it is:
//! `Defect::HeadHcRsqrtPerCopy` — the mHC denominator taken per copy instead of over the
//! flattened row — has a signal of the same ORDER as that re-association noise at real
//! dimensions. **No tolerance here can separate it.** It is settled by reading
//! `kernels/headtail.hip`, and pinned at small dimensions by
//! `tests/v4_oracle.rs::the_head_tail_matches_torch_absolutely`, which compares against
//! PyTorch itself.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli_backend::gpustream::HipStream;
use rivoli_backend::hip::{
    device_sync, launch_embed_bf16_row_bcast, launch_gemm_bf16, launch_hc_head_collapse,
    launch_qk_norm, launch_rmsnorm_batch,
};
use rivoli_engine::device::DeviceBuf;
use rivoli_oracles::v4oracle::{
    forward::{Capture, Defect, HeadTailW, Oracle},
    numerics::{bf16_decode, bf16_encode},
    weights::{NamedRng, V4Config, WMat},
};

mod common;
use common::{assert_bits, f32b, f32v, rel, u16b};

/// This file's device plumbing.
///
/// Owned here rather than shared, and the reason it USED to give is gone: `tests/common/mod.rs`
/// stated that "anything that touches a device TYPE stays in the test file that owns it",
/// because `dev` was `DeviceBuf` under HIP and `Buf` under Vulkan. That rule died with the
/// second backend on 2026-08-06 and `common` now owns `dev`/`zeros`/`back`. What keeps this
/// struct is narrower: the typed `p()`/`pm()`/`read()` accessors, which the four bare
/// functions do not provide. `tests/kvcompress_kernel.rs` wraps it the same way.
struct Dev(DeviceBuf);

impl Dev {
    fn of(bytes: Vec<u8>) -> Self {
        let mut b = DeviceBuf::new(bytes.len().max(1)).expect("device alloc");
        b.copy_in_at(0, &bytes).expect("host to device");
        Self(b)
    }
    fn f32s(v: &[f32]) -> Self {
        Self::of(f32b(v))
    }
    fn u16s(v: &[u16]) -> Self {
        Self::of(u16b(v))
    }
    /// `n` BYTES, zeroed.
    fn blank(n: usize) -> Self {
        Self::of(vec![0u8; n])
    }
    fn p(&self) -> *const f32 {
        self.0.ptr() as *const f32
    }
    fn pm(&mut self) -> *mut f32 {
        self.0.ptr_mut() as *mut f32
    }
    /// Join the device, THEN read. A readback that skips the join compares against memory the
    /// kernel may not have written -- a green test on a result that never existed.
    fn read(&self) -> Vec<f32> {
        device_sync().expect("device sync");
        f32v(&self.0.copy_out().expect("device to host"))
    }
}

fn bf(x: f32) -> f32 {
    bf16_decode(bf16_encode(x))
}
/// bf16 halves of `v`, which must already be bf16-valued.
fn as_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16_encode(*x)).collect()
}
// `rel` — `golden.rs::Diff.rel`, the metric the oracle's own gate uses — moved to
// `common` on 2026-08-06 when `tests/indexer_kernel.rs` reimplemented it under another name.

/// A head-tail fixture at the given `dim`, with everything bf16 where the checkpoint is.
struct Fixture {
    cfg: V4Config,
    hw: HeadTailW,
    h: Vec<f32>,
    s: usize,
}

impl Fixture {
    fn new(dim: usize, vocab: usize, s: usize) -> Self {
        let cfg = V4Config {
            dim,
            vocab_size: vocab,
            ..V4Config::toy()
        };
        let hcd = cfg.hc_dim();
        let mut r = NamedRng::new("v4-head-tail-device");
        let hw = HeadTailW {
            // F32 on disk, so f32 here -- a bf16 fixture would not exercise the mixes dot.
            hc_head_fn: (0..cfg.hc_mult * hcd).map(|_| r.unit() * 0.05).collect(),
            hc_head_base: (0..cfg.hc_mult).map(|_| r.unit()).collect(),
            hc_head_scale: vec![1.0 + r.unit() * 0.5],
            norm: (0..dim).map(|_| bf(1.0 + r.unit() * 0.3)).collect(),
            lm_head: WMat::Dense {
                rows: vocab,
                cols: dim,
                v: (0..vocab * dim).map(|_| bf(r.unit() * 0.05)).collect(),
            },
        };
        let h = (0..s * hcd).map(|_| bf(r.unit())).collect();
        Self { cfg, hw, h, s }
    }

    /// The oracle's three head-tail goldens.
    fn oracle(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut cap = Capture::default();
        Oracle::new(self.cfg.clone(), Defect::None)
            .head_tail(&self.hw, &self.h, self.s, "d", &mut cap);
        let g = |n: &str| cap.float(&format!("head.d.{n}")).expect(n).to_vec();
        (g("hc_head_out"), g("final_norm_out"), g("logits"))
    }

    /// The same three, on the device: `hc_head_collapse` -> `rmsnorm_batch` -> `gemm_bf16`,
    /// plus the gate vector `pre`, which `the_blend_is_bit_exact_given_the_gate` needs in order
    /// to test the blend WITHOUT the gate's re-association already in its input.
    fn device(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let (dim, hc, vocab) = (self.cfg.dim, self.cfg.hc_mult, self.cfg.vocab_size);
        let stream = HipStream::new().expect("stream");
        let hb = Dev::f32s(&self.h);
        let fnb = Dev::f32s(&self.hw.hc_head_fn);
        let bb = Dev::f32s(&self.hw.hc_head_base);
        let sb = Dev::f32s(&self.hw.hc_head_scale);
        let nb = Dev::f32s(&self.hw.norm);
        let WMat::Dense { v: lm, .. } = &self.hw.lm_head else {
            panic!("lm_head is Dense")
        };
        let lb = Dev::u16s(&as_bf16(lm));
        let mut preb = Dev::blank(self.s * hc * 4);
        let mut yb = Dev::blank(self.s * dim * 4);
        let mut lgb = Dev::blank(vocab * 4);
        // SAFETY: every buffer above is sized exactly as each launcher's contract requires
        // and outlives the sync below; none aliases another.
        unsafe {
            launch_hc_head_collapse(
                hb.p(),
                fnb.p(),
                bb.p(),
                sb.p(),
                preb.pm(),
                yb.pm(),
                self.s,
                hc,
                dim,
                self.cfg.norm_eps,
                self.cfg.hc_eps,
                stream.raw(),
            )
            .expect("hc_head_collapse");
        }
        let hc_out = yb.read();
        // `rmsnorm_batch` is one block per ROW and bf16-rounds on the way out, which is what
        // `RMSNorm.forward` does. `linalg.hip::rmsnorm_single` is single-row and would take one
        // statistic over every token -- `Defect::HeadNormOverAllTokens`, invisible at s == 1.
        // SAFETY: `yb` is `s * dim` live f32, `nb` is `dim`; both outlive the sync.
        unsafe {
            launch_rmsnorm_batch(
                yb.pm(),
                nb.p(),
                self.s,
                dim,
                self.cfg.norm_eps,
                stream.raw(),
            )
            .expect("rmsnorm_batch");
        }
        let norm_out = yb.read();
        // `ParallelHead.forward` slices `x[:, -1]` -- the LAST row only.
        let last = yb.p().wrapping_add((self.s - 1) * dim);
        // SAFETY: `last` is the final `dim` f32 of `yb`; `lb` is `vocab * dim` live u16;
        // `lgb` is `vocab` writable f32. m=1 n=vocab k=dim matches the kernel's [m,k]x[n,k].
        unsafe {
            launch_gemm_bf16(
                last,
                lb.p() as *const u16,
                lgb.pm(),
                1,
                vocab,
                dim,
                stream.raw(),
            )
            .expect("gemm_bf16");
        }
        (hc_out, norm_out, lgb.read(), preb.read())
    }
}

#[test]
fn the_embedding_gather_is_bit_exact_and_broadcasts_every_copy() {
    // A gather and a widen -- no arithmetic, so there is nothing to re-associate and bit
    // -exactness is the right bar. It is also the ONLY head-tail kernel of which that is
    // true at real dimensions.
    let (vocab, hidden, hc) = (512usize, 256usize, 4usize);
    let mut r = NamedRng::new("v4-embed-bf16");
    let table: Vec<f32> = (0..vocab * hidden).map(|_| bf(r.unit())).collect();
    let cfg = V4Config {
        dim: hidden,
        vocab_size: vocab,
        ..V4Config::toy()
    };
    let w = WMat::Dense {
        rows: vocab,
        cols: hidden,
        v: table.clone(),
    };
    let o = Oracle::new(cfg, Defect::None);
    let stream = HipStream::new().expect("stream");
    let tb = Dev::u16s(&as_bf16(&table));
    for token in [0usize, 1, 37, vocab - 1] {
        let mut xb = Dev::blank(hc * hidden * 4);
        // SAFETY: `tb` holds `vocab * hidden` u16 and `token < vocab`; `xb` is `hc * hidden`
        // writable f32. Both outlive the download below.
        unsafe {
            launch_embed_bf16_row_bcast(
                tb.p() as *const u16,
                token,
                hidden,
                hc,
                xb.pm(),
                stream.raw(),
            )
            .expect("embed_bf16_row_bcast");
        }
        let got = xb.read();
        let want = o.embed(&w, &[token as u32]);
        assert_eq!(want.len(), got.len(), "token {token}: shape");
        for (i, (g, x)) in got.iter().zip(&want).enumerate() {
            assert_eq!(
                g.to_bits(),
                x.to_bits(),
                "token {token} element {i}: {g:e} vs {x:e}"
            );
        }
        // The broadcast is the part a port forgets: `h.unsqueeze(2).repeat(1, 1, hc, 1)`.
        // Asserted directly, because a kernel writing only copy 0 would leave the rest zero
        // and still match `want` on the first `hidden` values if `want` were sliced.
        for c in 1..hc {
            assert_eq!(
                &got[c * hidden..(c + 1) * hidden],
                &got[..hidden],
                "copy {c} differs"
            );
        }
        assert!(
            got[..hidden].iter().any(|v| *v != 0.0),
            "token {token} row is all zero"
        );
    }
}

#[test]
fn the_head_tail_matches_the_oracle_at_toy_dimensions() {
    // `dim = 256`, so the gate's reduction is 1024 terms over 32 lanes -- 32 per lane. Small
    // enough that re-association is a handful of ulps, which is what makes a tight bound
    // meaningful here and NOT at 4096.
    let f = Fixture::new(256, 64, 3);
    let (w_hc, w_nm, w_lg) = f.oracle();
    let (g_hc, g_nm, g_lg, _) = f.device();
    let (r_hc, r_nm, r_lg) = (rel(&g_hc, &w_hc), rel(&g_nm, &w_nm), rel(&g_lg, &w_lg));
    println!("toy dims: hc_head_out {r_hc:.3e}, final_norm_out {r_nm:.3e}, logits {r_lg:.3e}");
    for (n, r) in [
        ("hc_head_out", r_hc),
        ("final_norm_out", r_nm),
        ("logits", r_lg),
    ] {
        assert!(r < 1e-3, "{n} disagrees with the oracle by {r:.3e}");
    }
    assert!(
        g_lg.len() == 64 && g_lg.iter().any(|v| *v != 0.0),
        "logits are empty or all zero"
    );
}

/// One `(token, d)` element of the blend: the `hc_mult` copies summed SEQUENTIALLY, then
/// bf16-rounded exactly where the kernel rounds. `fma` picks the contraction form, because
/// which one hipcc emitted is what the caller is measuring.
fn blend_copies(copies: impl Iterator<Item = (f32, f32)>, fma: bool) -> f32 {
    let mut acc = 0.0f32;
    for (p, x) in copies {
        acc = if fma { p.mul_add(x, acc) } else { acc + p * x };
    }
    bf(acc)
}

/// The blend half of `hc_head_collapse` on the host, over a caller-supplied gate vector so the
/// gate's re-association never enters the comparison.
fn blend_on_host(f: &Fixture, pre: &[f32], fma: bool) -> Vec<f32> {
    let (dim, hc) = (f.cfg.dim, f.cfg.hc_mult);
    let mut out = vec![0.0f32; f.s * dim];
    for t in 0..f.s {
        for d in 0..dim {
            let copies = (0..hc).map(|c| (pre[t * hc + c], f.h[(t * hc + c) * dim + d]));
            out[t * dim + d] = blend_copies(copies, fma);
        }
    }
    out
}

#[test]
fn the_blend_is_bit_exact_given_the_gate() {
    // The two halves of `hc_head` promise DIFFERENT things, and conflating them is how a gate
    // ends up either too loose to catch anything or red against correct code.
    //
    // `hc_head_collapse_blend` sums over `hc_mult` copies SEQUENTIALLY, in the oracle's own order,
    // so it is exactly reproducible -- and stays so at any `dim`, because `hc_mult` does not
    // grow. That is asserted against the device's OWN `pre`, not the oracle's: feeding the
    // oracle's gate vector would fold the gate's re-association back in and measure the sum
    // of both halves, which is what the first version of this test did while its name
    // promised otherwise.
    //
    // `hc_head_collapse_gate` reduces `hc_mult * dim` with `wave_sum` and cannot be bit-exact; its
    // contribution is reported here and bounded, not pinned.
    let f = Fixture::new(4096, 32, 8);
    let (dim, hc) = (f.cfg.dim, f.cfg.hc_mult);
    let (w_hc, _, _) = f.oracle();
    let (g_hc, _, _, pre) = f.device();
    assert_eq!(pre.len(), f.s * hc, "one gate weight per (token, copy)");
    assert!(
        pre.iter().all(|p| *p > 0.0 && *p < 1.0 + 1e-3),
        "gate is a sigmoid plus hc_eps"
    );

    // The blend alone, on the device's own gate vector, in the reference's order -- computed
    // BOTH ways, because whether hipcc contracts `acc += p * x` into an FMA decides which one
    // the kernel is actually doing and the build does not say.
    let (plain, fma) = (
        blend_on_host(&f, &pre, false),
        blend_on_host(&f, &pre, true),
    );
    let miss = |w: &[f32]| {
        // `zip` stops at the shorter side, so a truncated `got` would score ZERO mismatches
        // and pass as agreement. The oracle documents the same hazard for `RMSNorm`'s weight.
        assert_eq!(
            g_hc.len(),
            w.len(),
            "length mismatch would make the comparison vacuous"
        );
        g_hc.iter()
            .zip(w)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    let (m_plain, m_fma) = (miss(&plain), miss(&fma));
    println!(
        "dim {dim} s {}: blend vs plain-mul {m_plain}, vs mul_add {m_fma}",
        f.s
    );
    // MEASURED 2026-08-05 at this fixture: 3 and 0. The kernel contracts, so `mul_add` is what
    // it actually computes and `+ p * x` is not. Asserting the latter bitwise would have been
    // FLAKY, not wrong-looking -- roughly a 1-in-5 spurious red at the fixture this test
    // originally used, and indistinguishable from a real regression when it fired.
    assert_eq!(
        m_fma, 0,
        "the blend no longer reproduces an FMA reduction over the copies"
    );
    assert!(
        m_plain > 0,
        "the blend now matches a NON-contracted sum, so hipcc has stopped contracting (or the \
         fixture stopped reaching a rounding boundary). Re-measure before relying on either \
         form -- this bound is what keeps the assertion above from being a coincidence"
    );

    // And the gate's own contribution, which is NOT zero and must not be pinned to zero.
    let differing = g_hc
        .iter()
        .zip(&w_hc)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let r = rel(&g_hc, &w_hc);
    println!(
        "dim 1024: vs oracle {differing}/{} elements differ, rel {r:.3e}",
        w_hc.len()
    );
    assert!(
        r < 1e-2,
        "hc_head_out moved by {r:.3e}, past a rounding-scale disagreement"
    );
}

#[test]
fn the_lm_head_needs_no_kernel_of_its_own() {
    // `gemm_bf16` computes `out[m,n] = x[m,k] . w[n,k]` for f32 activations against
    // bf16 weights. `ParallelHead.forward` is `F.linear(x.float(), weight)` with `weight`
    // `[vocab, dim]` bf16 on disk -- the same shape at `m = 1, n = vocab, k = dim`. Verified
    // rather than argued, because writing a second GEMV that differed only in extents would
    // be duplication that `kernels/` -- outside jscpd's scan (build.rs:618) -- cannot catch.
    //
    // NOT covered here: the real `vocab` of 129280. Allocating that weight is 1.06 GB, which
    // is not a unit test. Four literal assertions over `(1, 129280, 4096)` used to sit at the
    // end of this test claiming to cover it; they referenced neither the kernel nor the
    // launcher, so the compiler folded them and no change to either could turn them red. That
    // is the "guard that cannot fire" this port has shipped twice, and it is deleted rather
    // than left to look like coverage. The real-extent argument lives in
    // `kvcompress.hip`'s own comment, beside the `(size_t)` cast that carries it.
    let f = Fixture::new(512, 1024, 2);
    let (_, _, w_lg) = f.oracle();
    let (_, _, g_lg, _) = f.device();
    assert_eq!(g_lg.len(), 1024, "one row of logits, whatever s was");
    let r = rel(&g_lg, &w_lg);
    println!("lm_head via gemm_bf16 at vocab 1024: rel {r:.3e}");
    assert!(
        r < 1e-3,
        "the existing dense GEMM does not reproduce the lm_head: rel {r:.3e}"
    );
}

/// `kernels/mla.hip::qk_norm` against `Oracle::qk_norm`.
///
/// Here rather than in `tests/f4_attn.rs` because of what it IS rather than where it is
/// called: it is a per-row bf16-rounded RMS normalisation, the direct sibling of
/// `rmsnorm_batch` above, and the two differ in exactly the way this file already exists to
/// pin — where the statistic is taken and in what precision. `f4_attn.rs` scores the whole
/// attention block, in which this kernel's contribution is one bf16 step wide.
///
/// **Bit-exact, and that is forced rather than chosen.** The kernel bf16-rounds `rs`, so
/// its output lands on a ~0.4% lattice; any tolerance loose enough to absorb a
/// re-association would be looser than the quantum and would absorb a wrong `rs` with it.
/// The two sides can only disagree if the f32 variance falls within a reduction's worth of
/// a bf16 tie point — the tree sum against the oracle's sequential one — which then costs a
/// whole step. Should this ever flake, it is that, and the fix is a different seed, not a
/// tolerance.
///
/// **No learnable weight**, model.py:504. The kernel takes none, and `Defect::
/// QkNormUsesQNormWeight` is the oracle's name for supplying one — so `q_norm_w` below is a
/// single 1.0 that `Defect::None` never reads, not a fixture with meaning.
///
/// Six rows, not one. `attn::v4::attention` launches this as `m * n_heads` rows in a single
/// grid, and the statistic is per row: a kernel that reduced over the whole buffer, or that
/// indexed `blockIdx.x` against the wrong stride, is invisible at one row.
#[test]
fn qk_norm_matches_the_oracle_bit_for_bit() {
    let (rows, head_dim) = (6usize, 128usize);
    let cfg = V4Config::toy();
    let mut r = NamedRng::new("v4-qk-norm-device");
    // Per-row scales spanning 32x (`1 << (i / head_dim)` over six rows is 1..32), so each
    // row's `rs` is a different number — measured, they run 1.758 down to 0.0532. Drawn from
    // one distribution every row would normalise by nearly the same factor, and a cross-row
    // statistic bug would land inside the bf16 quantum.
    let q: Vec<f32> = (0..rows * head_dim)
        .map(|i| bf(r.unit() * (1 << (i / head_dim)) as f32))
        .collect();

    let mut want = q.clone();
    Oracle::new(cfg.clone(), Defect::None).qk_norm(&mut want, head_dim, &[1.0]);

    let stream = HipStream::new().expect("stream");
    let mut qb = Dev::f32s(&q);
    // SAFETY: `qb` is `rows * head_dim` live f32, written in place, and outlives the sync
    // inside `read`. `stream` is a live HipStream handle.
    unsafe {
        launch_qk_norm(qb.pm(), rows, head_dim, cfg.norm_eps, stream.raw()).expect("qk_norm");
    }
    let got = qb.read();

    // Anti-vacuity, host against host: the normalisation must have MOVED the input, or
    // "the kernel matches the oracle" is satisfied by a kernel that writes nothing. The
    // per-row scale spread above is what makes this large.
    let moved = rel(&want, &q);
    println!("qk_norm: the norm moves the input by rel {moved:.3e}");
    assert!(
        moved > 0.5,
        "the fixture is already normalised; the comparison is vacuous"
    );
    assert_bits(&want, &got, "qk_norm");
}
