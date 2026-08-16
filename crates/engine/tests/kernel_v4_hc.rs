//! **DeepSeek-V4's hyper-connected residual, scored against the frozen host oracle** — `hc_pre`,
//! `hc_post`, and the `rmsnorm_single` that stands between them.
//!
//! Ported from `old:tests/f4_kernel.rs` §5. V4's residual is not one `[hidden]` stream but
//! `[hc_mult, hidden]` — four of them and a learned mix — so every sublayer is bracketed by a
//! REDUCTION (`hc_pre`: Sinkhorn-normalise the mixing matrix, collapse the copies to one, emit
//! the `post`/`comb` its partner consumes) and an EXPANSION (`hc_post`: expand the sublayer
//! output back to `hc` copies, mixing the pre-sublayer residual through `comb`). Neither is
//! captured on its own by the oracle, which records `.in`, `.attn_norm_out`, `.attn_out`,
//! `.ffn_norm_out`, `.ffn_out` and `.out`.
//!
//! # So the two are gated in the CHAIN that connects them
//!
//! ```text
//!   .in --hc_pre--> --rmsnorm--> .attn_norm_out          (gates hc_pre)
//!   (.attn_out, .in) --hc_post--> h1
//!   h1 --hc_pre--> --rmsnorm--> .ffn_norm_out            (gates hc_post, through hc_pre)
//!   (.ffn_out, h1) --hc_post--> .out                     (gates hc_post directly)
//! ```
//!
//! Every input is a golden or a layer weight, and the last line closes the loop: if the two mHC
//! halves were both wrong in compensating ways, `.ffn_norm_out` could still match, but `.out` is
//! `hc_post`'s own output and cannot.
//!
//! # What this file provably cannot detect
//!
//! 1. **The Sinkhorn iteration count, ON THIS FIXTURE.** At `hc_sinkhorn_iters = 20` the toy's
//!    4x4 matrix reaches a bitwise fixed point: 19 and 20 agree BIT-FOR-BIT, so no golden built
//!    on it distinguishes them, and the oracle's own matrix excludes
//!    `Defect::SinkhornIterCountProbe` for that measured reason. What
//!    [`sinkhorn_iteration_count_is_live`] proves is strictly weaker and is all that is available
//!    here: the parameter REACHES the arithmetic (2 and 20 disagree). The exact value is gated by
//!    SOURCING — it is passed from the config — not by measurement on this fixture.
//!
//!    > This is a fact about these WEIGHTS, not about the arithmetic. On the checkpoint, 19 vs 20
//!    > moves 39,893/53,248 of `L0.pre.ffn_norm_out` and all 78 router weights, so a
//!    > real-weights golden is not blind to the count — this file is.
//! 2. **`rmsnorm_single`'s missing bf16 store.** The kernel does not bf16-round its output and
//!    V4's `RMSNorm.forward` returns bf16 (`model.py:197-202` computes in f32 and the module's
//!    dtype is bf16). That is a real gap and it is NOT this file's to close: `rmsnorm_single` is
//!    shared with the GLM path, where adding a store would change shipped output. [`norm`]
//!    applies the missing round on the host and ASSERTS that it was worth something, so the gap
//!    is on the record rather than absorbed into a tolerance — and that assertion goes red
//!    exactly when the wrapper stops being needed.
//! 3. **The `hc_fn` GEMV's re-association.** The oracle's fidelity note lists `fp8_gemm`,
//!    `sparse_attn` and `hc_split_sinkhorn`'s warp reductions as reproduced only up to summation
//!    order. It does NOT list this GEMV, which the kernel also wave-reduces where the oracle sums
//!    sequentially. Same class, and it rides the same [`TOL`] — but it is an unstated consumer of
//!    it, and the oracle's scope note should say so.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, and no GPU for this port. One mutation each:
//!
//! * In `kernels/linalg.hip::rivoli_hc_post`, TRANSPOSE the `comb` index (`comb[dest*hc+src]`
//!   for `comb[src*hc+dest]`). [`mhc_reproduces_the_layer_goldens`] must go RED on its
//!   `hc_post(ffn)` arm — and note WHY nothing cheaper would catch it: transposing leaves every
//!   output row a combination of the same vectors, so no magnitude, norm or shape check can see
//!   it, which is exactly why the golden chain is the instrument. `hc_pre`'s two arms must stay
//!   green, which is what says the failure is attributed to the right half.
//! * In `kernels/linalg.hip::rivoli_hc_pre`, replace the `iters` parameter with a literal `2`.
//!   [`sinkhorn_iteration_count_is_live`] must go RED on its first assertion
//!   (`bits(c20) != bits(c2)`) and GREEN on its second (`bits(c20) == bits(c19)`), because at 2
//!   iterations both arms compute the same thing. The second assertion going red instead is a
//!   different and more interesting failure: the toy's fixed point holding on the CPU and not on
//!   the GPU.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_backend::hip::{launch_hc_post, launch_hc_pre, launch_rmsnorm_single};
use rivoli_engine::device::DeviceBuf;
use rivoli_oracles::v4oracle::forward::{Capture, LayerW};
use rivoli_oracles::v4oracle::numerics::{bf16_decode, bf16_encode};
use rivoli_oracles::v4oracle::weights::{V4Config, fixed_bf16};

mod common;
use common::{
    Got, Prefill, Want, assert_bits, assert_guard, assert_guards, assert_rel, back, dev, f32b,
    f32v, prefill, report_rel, stream, toy_fixture, zeros,
};

/// Two bf16 ulps, relative to the largest expected element — the tolerance the golden-chain
/// comparisons are held to.
///
/// Not `common::assert_close`'s `1e-3·max + 1e-3`, whose ABSOLUTE floor dominates at this
/// fixture's scale. The reference stores bf16 at every step, so any upstream difference — the
/// `hc_fn` GEMV's wave reduction against the oracle's sequential sum, the Sinkhorn's block
/// reductions, `expf` against Rust's `exp` — flips an element by a whole ulp rather than by its
/// own magnitude. One ulp is the floor; two is the margin for a flip that then propagates.
const TOL: f32 = 1.0 / 128.0;

/// One sublayer's three mHC tensors, in the order `hc_pre` reads them.
///
/// A struct rather than three adjacent `&[f32]` parameters. `fnw`, `scale` and `base` have
/// different LENGTHS but the same type, they were spelled adjacently at three call sites, and a
/// transposition compiles: `hc_pre` uploads each into its own buffer and reads them at the
/// strides its own arguments imply, so swapping two would produce finite, plausible numbers from
/// the wrong tensors. Naming the constructors [`HcW::attn`]/[`HcW::ffn`] also removes the second
/// mistake available here: a caller mixing `hc_attn_scale` into the ffn triple.
#[derive(Clone, Copy)]
struct HcW<'a> {
    fnw: &'a [f32],
    scale: &'a [f32],
    base: &'a [f32],
}

/// WHICH sublayer's mHC triple. An enum matched exhaustively rather than two constructors with a
/// literal each: the two literals differ only in the tensor names, so spelling them twice is the
/// one place a caller could pair `hc_attn_scale` with `hc_ffn_fn` and get finite numbers.
#[derive(Clone, Copy, Debug)]
enum Sub {
    Attn,
    Ffn,
}

impl<'a> HcW<'a> {
    /// ONE struct literal, with the three slices chosen by an exhaustive match. A sublayer added
    /// later is a compile error here rather than a silent fall-through to the attention triple.
    fn of(lw: &'a LayerW, which: Sub) -> Self {
        let (fnw, scale, base) = match which {
            Sub::Attn => (&lw.hc_attn_fn, &lw.hc_attn_scale, &lw.hc_attn_base),
            Sub::Ffn => (&lw.hc_ffn_fn, &lw.hc_ffn_scale, &lw.hc_ffn_base),
        };
        Self { fnw, scale, base }
    }
}

/// One mHC call's shape: how many token rows, how many residual copies, how wide.
///
/// Three bare `usize` travelling to both launchers, and `hc` and `dim` are each plausible in the
/// other's position at every buffer size below. `common::geometry`'s `Mla` makes this argument
/// about six of them.
#[derive(Clone, Copy)]
struct Shape {
    s: usize,
    hc: usize,
    dim: usize,
}

impl Shape {
    fn of(cfg: &V4Config, s: usize) -> Self {
        Self {
            s,
            hc: cfg.hc_mult,
            dim: cfg.dim,
        }
    }
}

/// The three device buffers `hc_pre` writes, read back — `(y, post, comb)`.
///
/// `post` and `comb` are not compared against anything on their own: the oracle captures neither,
/// and they exist here to be fed straight into `hc_post`, which is what makes the chain a chain.
fn gpu_hc_pre(
    cfg: &V4Config,
    h: &[f32],
    w: HcW<'_>,
    g: Shape,
    iters: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let stream = stream();
    let (hb, fb) = (dev(&f32b(h)), dev(&f32b(w.fnw)));
    let (sb, bb) = (dev(&f32b(w.scale)), dev(&f32b(w.base)));
    let mut y = zeros(g.s * g.dim * 4);
    let mut post = zeros(g.s * g.hc * 4);
    let mut comb = zeros(g.s * g.hc * g.hc * 4);
    // SAFETY: `h` is `s·hc·dim`, `fnw` is `(2+hc)·hc` rows of `hc·dim`, `scale` is 3 and `base`
    // is `(2+hc)·hc`, all by the toy's own construction; the three destinations are sized for
    // `(s, hc, dim)` immediately above and none aliases `h`. All outlive the join inside `back`.
    unsafe {
        launch_hc_pre(
            hb.ptr() as *const f32,
            fb.ptr() as *const f32,
            sb.ptr() as *const f32,
            bb.ptr() as *const f32,
            g.s,
            g.hc,
            g.dim,
            iters,
            cfg.norm_eps,
            cfg.hc_eps,
            y.ptr_mut() as *mut f32,
            post.ptr_mut() as *mut f32,
            comb.ptr_mut() as *mut f32,
            stream.raw(),
        )
    }
    .expect("hc_pre");
    (f32v(&back(&y)), f32v(&back(&post)), f32v(&back(&comb)))
}

/// What `hc_post` expands: this call's sublayer output, the residual it was computed from, and
/// the pair `hc_pre` emitted alongside.
///
/// Four `&[f32]` in a row — the same argument [`HcW`] makes, and here the failure is sharper:
/// `post` and `comb` come from the SAME `hc_pre` call, and pairing one call's `post` with
/// another's `comb` is finite, plausible and wrong.
#[derive(Clone, Copy)]
struct HcPost<'a> {
    x: &'a [f32],
    residual: &'a [f32],
    post: &'a [f32],
    comb: &'a [f32],
}

/// `hc_post` over `g.s` tokens.
fn gpu_hc_post(a: HcPost<'_>, g: Shape) -> Vec<f32> {
    let stream = stream();
    let (xb, rb) = (dev(&f32b(a.x)), dev(&f32b(a.residual)));
    let (pb, cb) = (dev(&f32b(a.post)), dev(&f32b(a.comb)));
    let mut y = zeros(g.s * g.hc * g.dim * 4);
    // SAFETY: sized for `(s, hc, dim)` above; `y` is a fresh allocation and so cannot alias
    // `residual`, which the kernel's own note requires.
    unsafe {
        launch_hc_post(
            xb.ptr() as *const f32,
            rb.ptr() as *const f32,
            pb.ptr() as *const f32,
            cb.ptr() as *const f32,
            g.s,
            g.hc,
            g.dim,
            y.ptr_mut() as *mut f32,
            stream.raw(),
        )
    }
    .expect("hc_post");
    f32v(&back(&y))
}

/// `rmsnorm_single` per token row, so the `hc_pre` comparison lands on a golden the oracle emits.
///
/// **ONE LAUNCH PER TOKEN.** `rivoli_rmsnorm_single` is single-row — `dim3(1)`, one mean over its
/// whole `n`, and `w[i]` indexed over that same `n`. Handing it `s·dim` takes a JOINT rms over
/// every token (the oracle's is per token) and reads the norm weight `s-1` rows past its
/// allocation. Both are silent: the golden's length still matches, so the comparison is happy,
/// and the arithmetic error rides in as a plausible scale.
fn gpu_rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let dim = w.len();
    assert!(
        x.len().is_multiple_of(dim),
        "x must be whole rows of the norm weight"
    );
    let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
    let mut y: DeviceBuf = zeros(x.len() * 4);
    for t in 0..x.len() / dim {
        // SAFETY: row `t` is `dim` live f32 inside both buffers, and `w` is `dim` long.
        unsafe {
            launch_rmsnorm_single(
                (xb.ptr() as *const f32).add(t * dim),
                wb.ptr() as *const f32,
                dim,
                eps,
                (y.ptr_mut() as *mut f32).add(t * dim),
            )
        }
        .expect("rmsnorm_single");
    }
    f32v(&back(&y))
}

/// [`gpu_rmsnorm`] plus the bf16 store the kernel is missing, with the size of the gap ASSERTED
/// rather than printed.
///
/// `println!` is captured and discarded on a green run, so "the number is on the record" would be
/// a claim about output nobody sees. This goes red exactly when the wrapper stops being needed —
/// i.e. when `rmsnorm_single`'s output is already bf16-representable and the missing store has
/// stopped mattering.
fn norm(v: &[f32], w: &[f32], eps: f32, label: &str) -> Vec<f32> {
    let raw = gpu_rmsnorm(v, w, eps);
    let rounded: Vec<f32> = raw.iter().map(|x| bf16_decode(bf16_encode(*x))).collect();
    let (err, _) = report_rel(
        Want(&raw),
        Got(&rounded),
        &format!("{label}: rmsnorm's missing bf16 store"),
        TOL,
    );
    assert!(
        err > 0.0,
        "{label}: rmsnorm's output is already bf16-representable — the missing store has stopped \
         mattering, so drop this wrapper and call `gpu_rmsnorm` directly"
    );
    rounded
}

/// One prefill `run_layer` capture, and the residual it was driven from.
///
/// `h` is the mHC residual block, `[s, hc_mult · dim]`, drawn through the ORACLE's own
/// `fixed_bf16` so the fixture is reproducible from the name alone and a rerun compares the same
/// numbers. `s` is `ids.len()` by construction rather than a second parameter: a `LayerCtx` whose
/// `s` disagreed with its `input_ids` length silently makes every golden a capture of a prompt
/// nobody wrote.
fn capture(layer: usize, s: usize) -> (Capture, Vec<f32>) {
    let fx = toy_fixture();
    prefill(
        fx,
        Prefill {
            o: &fx.2,
            layer,
            tag: "hc-h",
            s,
            scale: 1.0,
        },
    )
}

/// mHC end to end, against `run_layer`'s own goldens.
#[test]
fn mhc_reproduces_the_layer_goldens() {
    let (cfg, m, _) = toy_fixture();
    const S: usize = 3;
    let (cap, h_in) = capture(0, S);
    let lw = &m.layers[0];
    let g = Shape::of(cfg, S);
    let iters = cfg.hc_sinkhorn_iters;
    let golden = |n: &str| {
        cap.float(n)
            .unwrap_or_else(|| panic!("golden {n}"))
            .to_vec()
    };

    assert_bits(
        &golden("L0.pre.in"),
        &h_in,
        "the driver's h is not what the oracle recorded",
    );

    let (y, post, comb) = gpu_hc_pre(cfg, &h_in, HcW::of(lw, Sub::Attn), g, iters);
    assert_rel(
        &golden("L0.pre.attn_norm_out"),
        &norm(&y, &lw.attn_norm, cfg.norm_eps, "attn"),
        "hc_pre(attn) then rmsnorm",
        TOL,
    );

    let attn_out = golden("L0.pre.attn_out");
    let h1 = gpu_hc_post(
        HcPost {
            x: &attn_out,
            residual: &h_in,
            post: &post,
            comb: &comb,
        },
        g,
    );
    let (y2, post2, comb2) = gpu_hc_pre(cfg, &h1, HcW::of(lw, Sub::Ffn), g, iters);
    assert_rel(
        &golden("L0.pre.ffn_norm_out"),
        &norm(&y2, &lw.ffn_norm, cfg.norm_eps, "ffn"),
        "hc_post(attn) then hc_pre(ffn) then rmsnorm",
        TOL,
    );

    let ffn_out = golden("L0.pre.ffn_out");
    let out = gpu_hc_post(
        HcPost {
            x: &ffn_out,
            residual: &h1,
            post: &post2,
            comb: &comb2,
        },
        g,
    );
    assert_rel(&golden("L0.pre.out"), &out, "hc_post(ffn)", TOL);
}

/// The Sinkhorn iteration count reaches the arithmetic.
///
/// **NOT a check that the count is 20**, and on this fixture it cannot be — see the module
/// header's item 1. What is provable here is that the parameter is LIVE rather than ignored,
/// which is what makes SOURCING it from the config the actual gate on the value.
#[test]
fn sinkhorn_iteration_count_is_live() {
    let (cfg, m, _) = toy_fixture();
    const S: usize = 2;
    let lw = &m.layers[0];
    let h = fixed_bf16("sink-h", S * cfg.hc_mult * cfg.dim, 1.0);
    assert!(cfg.hc_sinkhorn_iters >= 2, "this test subtracts one below");
    let g = Shape::of(cfg, S);
    let run = |iters| gpu_hc_pre(cfg, &h, HcW::of(lw, Sub::Attn), g, iters).2;
    let (c20, c2, c19) = (
        run(cfg.hc_sinkhorn_iters),
        run(2),
        run(cfg.hc_sinkhorn_iters - 1),
    );
    // Bit-INEQUALITY, matching the claim the oracle's own test makes rather than a threshold
    // picked here: a magnitude threshold would be a weaker statement that could pass for a kernel
    // whose `iters` only tickled the low bits.
    assert_ne!(
        c20.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        c2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "2 and {} iterations agree — `iters` never reaches the kernel",
        cfg.hc_sinkhorn_iters
    );
    // The blind spot itself, asserted in the direction the oracle asserts it. A red here means
    // the fixture's fixed point has stopped holding on the GPU where it holds on the CPU, which
    // would mean the two arithmetics have diverged somewhere worth finding. It would NOT be news
    // that the count is observable in general: on the checkpoint it already is.
    assert_bits(
        &c19,
        &c20,
        "19 and 20 iterations disagree on the GPU where they agree on the CPU",
    );
}

/// Every mHC guard, by CODE.
///
/// The config already pins `hc_mult = 4`, so on the shipping path `hc != HC_MULT` can only ever
/// pass — and it is the guard `launch_hc_pre`'s doc leans on hardest, since `mix_hc = (2+hc)·hc`
/// is how the mHC weights are PACKED and this check is all that stands between a foreign
/// checkpoint and a wrong-stride read of `fnw`. Only a test that hands it a wrong value separates
/// "correct" from "unreachable".
#[test]
fn the_mhc_launchers_refuse_what_they_claim_to() {
    let (cfg, m, _) = toy_fixture();
    let lw = &m.layers[0];
    let stream = stream();
    let h = zeros(cfg.hc_mult * cfg.dim * 4);
    let (f, sc, b) = (
        dev(&f32b(&lw.hc_attn_fn)),
        dev(&f32b(&lw.hc_attn_scale)),
        dev(&f32b(&lw.hc_attn_base)),
    );
    // Sized for the ACCEPTED case each launcher runs, which is the half of a guard test that
    // actually touches memory: `hc_pre` writes `s·dim`, `hc_post` writes `s·hc·dim`. A single
    // shared output buffer sized for `hc_pre` would let `hc_post`'s accepted arm overrun it by
    // `hc`x — the first draft of this test in the reference tree did exactly that.
    let mut y = zeros(cfg.dim * 4);
    let mut expanded = zeros(cfg.hc_mult * cfg.dim * 4);
    let mut post = zeros(cfg.hc_mult * 4);
    let mut comb = zeros(cfg.hc_mult * cfg.hc_mult * 4);

    // Addresses hoisted so the two closures below vary only their guarded arguments.
    let (hp, fp, scp, bp) = (
        h.ptr() as *const f32,
        f.ptr() as *const f32,
        sc.ptr() as *const f32,
        b.ptr() as *const f32,
    );
    let (yp, pp, cp) = (
        y.ptr_mut() as *mut f32,
        post.ptr_mut() as *mut f32,
        comb.ptr_mut() as *mut f32,
    );
    let ep = expanded.ptr_mut() as *mut f32;
    let pre = |s, hc, iters| {
        // SAFETY: every rejected case returns before a dereference, and the accepted one is sized
        // by the buffers above; all of them outlive the sync that follows it.
        unsafe {
            launch_hc_pre(
                hp,
                fp,
                scp,
                bp,
                s,
                hc,
                cfg.dim,
                iters,
                cfg.norm_eps,
                cfg.hc_eps,
                yp,
                pp,
                cp,
                stream.raw(),
            )
        }
    };
    let (hc, it) = (cfg.hc_mult, cfg.hc_sinkhorn_iters);
    assert_guard(pre(1, hc, it), None, "hc_pre's accepted case");
    rivoli_backend::hip::device_sync().expect("device sync"); // the accepted case LAUNCHED
    let post_call = |s, hc| {
        // SAFETY: same — rejected before a dereference, accepted within the buffers. `y` (`dim`)
        // is the sublayer output and `expanded` (`hc·dim`) the destination; they are DISTINCT,
        // because `hc_post` reads `x[tok·dim+d]` while writing every copy.
        unsafe { launch_hc_post(yp, hp, pp, cp, s, hc, cfg.dim, ep, stream.raw()) }
    };
    assert_guard(post_call(1, hc), None, "hc_post's accepted case");
    rivoli_backend::hip::device_sync().expect("device sync");
    assert_guards([
        (1001, "hc_pre zero tokens", pre(0, hc, it)),
        (1002, "hc_pre hc_mult 3", pre(1, 3, it)),
        (1002, "hc_pre hc_mult 8", pre(1, 8, it)),
        // A Sinkhorn that runs the leading column normalisation and no pairs at all.
        (1003, "hc_pre zero iterations", pre(1, hc, 0)),
        (1001, "hc_post zero tokens", post_call(0, hc)),
        (1002, "hc_post hc_mult 2", post_call(1, 2)),
    ]);
}
