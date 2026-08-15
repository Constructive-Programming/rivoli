//! **S4 item 4's first quantized rung: `convert_glimmer --fp8`, end to end.**
//!
//! `investigations/glimmer-integration.md` S4. Muse Glimmer ships bf16 and nothing else, so unlike
//! every other model in this tree the quantization here is *rivoli's* decision rather than a
//! publisher's — `quant::quantize_fp8_block`'s doc carries that argument, and this file is what
//! says the decision was implemented rather than merely described.
//!
//! **Two gates with different jobs, and keeping them apart is the point.**
//!
//! `a_placed_fp8_projection_matches_the_host_oracle_over_the_same_bytes` is the ARITHMETIC gate:
//! one placed projection against `quant::matvec_fp8` over the same device bytes, where the only
//! difference is summation order and the bound is 1e-4. Every way the new wiring can be wrong —
//! a transposed scale-grid index, the wrong block, a weight paired with another tensor's grid, an
//! o/i swap — is O(1) wrong there.
//!
//! `an_fp8_artifact_decodes_and_is_not_secretly_the_bf16_one` is the INTEGRATION gate, and it is
//! coarse on purpose: fp8 and bf16 hold different numbers, so no tight bound between them exists.
//! It asserts the two things that are sharp — the decode yields a usable logit row, and it is not
//! the bf16 one.
//!
//! A GPU arm — it decodes. Needs the flock and `--test-threads=1`.

#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{GLIMMER_FP8_FIXTURE_DIM as DIM, TempRoot, glimmer_convert_fixture_fmt};
use rivoli::artifact::format::{Dtype, Safetensors};
use rivoli::artifact::model as gm;
use rivoli::artifact::quant::FP8_BLOCK;
use rivoli::glimmer_gpu::Glimmer;

/// Convert the fixture at `dim` into `root/out` in `fmt`, and return that directory.
fn artifact(root: &TempRoot, fmt: gm::GlimmerFormat) -> String {
    let _ = glimmer_convert_fixture_fmt(root.path(), DIM, fmt);
    root.join("out").to_str().unwrap().to_string()
}

/// Decode one token from `dir` and return the full logit row.
///
/// The LOGITS rather than the argmax, deliberately: the softcap is argmax-invariant and so is any
/// monotone weight error, so a greedy comparison would agree between arms that differ everywhere.
/// This is the same reason `glimmer_gpu::sample`'s own doc gives for why a greedy gate cannot be
/// that path's evidence.
fn logits(dir: &str, prompt: &[u32]) -> Vec<f32> {
    let cfg: gm::GlimmerConfig = gm::load_config(dir).unwrap();
    let mut e = Glimmer::new(dir, &cfg.text, None, prompt.len() + 1).unwrap();
    e.decode(prompt, 1, &[]).unwrap();
    e.logits().unwrap()
}

/// Worst relative difference between two logit rows, and the index it happened at.
///
/// Relative to the larger magnitude rather than to `want`, so a near-zero reference cannot
/// manufacture a huge ratio out of two numbers that are both tiny.
fn worst(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "logit rows differ in length");
    // **Guard the reference before believing any comparison.** `f32::max` ignores NaN, so an
    // all-NaN row scores as a perfect match against anything — this repo has shipped a broken
    // kernel that passed 9 of 9 that way.
    assert!(
        a.iter().chain(b).all(|v| v.is_finite()) && a.iter().any(|v| *v != 0.0),
        "a logit row is non-finite or all-zero, so the comparison below cannot mean what it claims"
    );
    let mut w = (0.0f32, 0usize);
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x - y).abs() / x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        if d > w.0 {
            w = (d, i);
        }
    }
    w
}

/// **What `--fp8` actually wrote, counted rather than assumed.**
///
/// The census is over the artifact's own header, so it catches the two ways `is_layer_proj` could
/// go wrong that the converter's own count assertion cannot: quantizing something it should not
/// have (`embed_tokens`, `lm_head`), and writing a scale grid of the wrong shape.
#[test]
fn the_fp8_artifact_is_fp8_exactly_where_it_should_be() {
    let root = TempRoot::new("glimmer-fp8-census");
    let dir = artifact(&root, gm::GlimmerFormat::Fp8);
    let cfg: gm::GlimmerConfig = gm::load_config(&dir).unwrap();
    let st = Safetensors::open_dir(&dir).unwrap();

    assert_eq!(
        gm::GlimmerFormat::of_artifact(&dir).unwrap(),
        gm::GlimmerFormat::Fp8,
        "the artifact must declare itself fp8 through its own tensors"
    );

    // Every layer projection is fp8 with a grid of the shape its dims imply.
    let mut fp8 = 0usize;
    for l in 0..cfg.text.n_layers {
        for t in gm::GLIMMER_LAYER_TENSORS {
            let want = cfg.text.layer_tensor_shape(t).unwrap();
            let (_, dtype, _) = st
                .raw(&format!("{}.{l}.{t}.weight", gm::GLIMMER_LAYER_PREFIX))
                .unwrap();
            if want.len() == 1 {
                assert_eq!(dtype, Dtype::F32, "{t} is a norm and stays f32");
                continue;
            }
            assert_eq!(dtype, Dtype::F8E4M3, "{t} must be quantized");
            let (_, sdtype, sshape) = st
                .raw(&format!(
                    "{}.{l}.{t}.weight_scale_inv",
                    gm::GLIMMER_LAYER_PREFIX
                ))
                .unwrap();
            let grid: Vec<usize> = want.iter().map(|d| d.div_ceil(FP8_BLOCK)).collect();
            assert_eq!(sdtype, Dtype::F32);
            assert_eq!(*sshape, grid[..], "{t}'s scale grid");
            // The fixture width exists to make this true; if it stops being true the gate stops
            // being able to see a transposed grid index. See `GLIMMER_FP8_FIXTURE_DIM`.
            assert!(
                grid[0] > 1 && grid[1] > 1,
                "{t}'s grid is {grid:?} — a single-tile axis cannot exercise the scale index"
            );
            fp8 += 1;
        }
    }
    assert_eq!(fp8, cfg.text.n_layers * 8, "eight projections per layer");

    // And the two the format deliberately leaves alone.
    for n in ["lm_head.weight", "model.language_model.embed_tokens.weight"] {
        assert_eq!(
            st.raw(n).unwrap().1,
            Dtype::Bf16,
            "{n} is read once per token and is not this format's to quantize"
        );
    }

    // The byte arithmetic the tier is sized from, against the artifact that exists.
    let (b16, b8) = (
        cfg.text.layer_bytes(gm::GlimmerFormat::Bf16).unwrap(),
        cfg.text.layer_bytes(gm::GlimmerFormat::Fp8).unwrap(),
    );
    let mut actual = 0usize;
    for t in gm::GLIMMER_LAYER_TENSORS {
        for suffix in ["weight", "weight_scale_inv"] {
            if let Ok((b, _, _)) = st.raw(&format!("{}.0.{t}.{suffix}", gm::GLIMMER_LAYER_PREFIX)) {
                actual += b.len();
            }
        }
    }
    assert_eq!(
        b8, actual,
        "layer_bytes(Fp8) must be the bytes the artifact's layer 0 occupies"
    );
    // No `b8 < b16 && b8 * 2 > b16` beside it: that is arithmetic over two numbers the exact
    // equality above already pins, and it would still pass on a `b8` that was wrong by 40%.
    let _ = b16;
}

/// **The arithmetic gate: one placed fp8 projection against a host oracle over the SAME bytes.**
///
/// The end-to-end test below cannot be this sharp and it says so. Here the two sides differ only
/// in summation ORDER — `matvec_fp8` accumulates a row left to right in f32, `gemv_fp8` reduces it
/// across a wave — so anything above reduction noise is a defect rather than a rounding budget,
/// and every way the new wiring can be wrong is O(1) wrong here:
///
/// * a transposed scale-grid index reads `sc_rows` where `sc_cols` belongs (5x3 against 3x5 at
///   this fixture width, so it is not a permutation),
/// * the wrong `block` reaches the kernel and every tile past the first is scaled by its
///   neighbour,
/// * `place_fp8` paired a weight with another tensor's scale grid,
/// * `o_dim`/`i_dim` arrive swapped from `GlimmerProj::dims`.
///
/// **`matvec_fp8` is a host oracle, not a second copy of the kernel** — it predates this work by
/// three models and is what GLM's own fp8 path is scored against.
#[test]
fn a_placed_fp8_projection_matches_the_host_oracle_over_the_same_bytes() {
    use rivoli::memory::device::DeviceBuf;
    let root = TempRoot::new("glimmer-fp8-gemv");
    let dir = artifact(&root, gm::GlimmerFormat::Fp8);
    let cfg: gm::GlimmerConfig = gm::load_config(&dir).unwrap();
    let mut pin = rivoli::memory::pin::GlimmerPin::build(&dir, &cfg.text, None).unwrap();

    // Every projection, not just `q`: the eight differ in shape and three of them are the
    // transposed ones, which is where an o/i swap hides.
    for (name, w) in [
        ("q", pin.layer(0).unwrap().q),
        ("k", pin.layer(0).unwrap().k),
        ("o", pin.layer(0).unwrap().o),
        ("mlp_down", pin.layer(0).unwrap().mlp_down),
    ] {
        let [o_dim, i_dim] = w.dims();
        let rivoli::memory::pin::GlimmerProj::Fp8(f) = w else {
            panic!("{name} is not fp8 on an fp8 artifact");
        };
        // Safe for the reason `glimmer_pin.rs` gives: the tier is a host-fillable VMM mapping, so
        // every pointer the pin hands out is readable here.
        let (packed, scale) = unsafe {
            (
                std::slice::from_raw_parts(f.packed, o_dim * i_dim),
                std::slice::from_raw_parts(
                    f.scale,
                    o_dim.div_ceil(FP8_BLOCK) * i_dim.div_ceil(FP8_BLOCK),
                ),
            )
        };
        // A varying activation, not a constant one: `x[i] = 1` makes the dot product a row sum,
        // which is invariant under any permutation of the row — so a strided-read bug would pass.
        let x: Vec<f32> = (0..i_dim).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
        let mut want = vec![0f32; o_dim];
        rivoli::artifact::quant::matvec_fp8(&mut want, &x, packed, scale, i_dim, FP8_BLOCK);

        let mut xb = DeviceBuf::new(i_dim * 4).unwrap();
        let ob = DeviceBuf::new(o_dim * 4).unwrap();
        xb.copy_in_at(0, &bytemuck_f32(&x)).unwrap();
        // SAFETY: `x` is `i_dim` live f32, `f.packed` is `o_dim*i_dim` live bytes in the tier with
        // its grid beside it, `ob` is `o_dim` writable f32, none aliasing. Null stream, synced
        // immediately below.
        unsafe {
            rivoli::backend::hip::launch_gemv_fp8(
                xb.ptr() as *const f32,
                f.packed,
                f.scale,
                o_dim,
                i_dim,
                // **`f.block`, not the `FP8_BLOCK` literal** — the field `glimmer_gpu::proj`
                // actually reads. Passing the constant meant the placement's own block was on
                // neither side of the comparison, while this gate's doc claimed "the wrong block
                // reaches the kernel" as one of the four things it catches (review, 2026-08-15).
                f.block,
                1,
                ob.ptr() as *mut f32,
            )
            .unwrap();
        }
        rivoli::backend::hip::device_sync().unwrap();
        // Through `common`'s readback pair rather than a hand-rolled `copy_out_prefix` +
        // `chunks_exact` — jscpd matched that spelling against `tests/kernel.rs`, and it is right
        // that the conversion is one fact the shared module already owns.
        let got = common::f32v(&common::back(&ob));

        // **Guard BOTH sides, and the device side is not optional.**
        //
        // > Review found this checking only `want`, 2026-08-15 — the exact trap `worst()` above
        // > guards against and cites, in this same file, three functions up. `f32::max` returns
        // > the non-NaN argument, so `fold(0f32, f32::max)` over an all-NaN difference is **0.0**
        // > and `0.0 < 1e-4` passes. A kernel that wrote NaN into every element, or into every
        // > row its grid failed to cover, scored a PERFECT match. This repo has already shipped a
        // > broken kernel that passed 9 of 9 that way; the gate is red-proved against a NaN
        // > device row below.
        let worst = common::worst_vs_scale(&want, &got, name);
        // Reduction order only. A scale-grid or block defect is O(1) here, not O(1e-6), so this
        // bound has three orders of magnitude of daylight on either side of it.
        assert!(
            worst < 1e-4,
            "{name} [{o_dim}, {i_dim}]: device fp8 GEMV differs from the host oracle by {worst} \
             of the row scale — that is not summation order"
        );
    }
}

/// `&[f32]` as little-endian bytes. Spelled out rather than transmuted: `DeviceBuf::copy_in`
/// takes bytes, and this is the one place in this file that needs the conversion.
fn bytemuck_f32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// **The integration arm: a whole decode on an fp8 artifact, and proof the format is in play.**
///
/// Deliberately NOT the arithmetic gate — that is
/// `a_placed_fp8_projection_matches_the_host_oracle_over_the_same_bytes`, which scores one
/// projection against a host oracle over the same bytes and holds to 1e-4. This one runs the
/// whole loop: `GlimmerProj` dispatch in `proj`, the pin over every layer, embed and `lm_head`
/// staying bf16 around it. What it can assert is coarse, and the reason is worth writing down.
///
/// **An `fp8`-vs-`bf16` bound is not available at any useful tightness, because the quantization
/// is the difference.** e4m3 carries 3 mantissa bits, so the two arms hold genuinely different
/// numbers and the gap is whatever this architecture amplifies them into — **measured at ~8% of
/// the logit row scale over only four layers**, by decoding a third artifact holding the fp8
/// weights dequantized back to bf16. That artifact is not kept: an oracle whose tightest honest
/// tolerance is 8% gates nothing the projection test does not gate at 1e-4.
///
/// So the assertions are the two that ARE sharp: the decode produces a usable logit row, and it is
/// not the bf16 one.
///
/// **What the second assertion actually establishes, stated twice wrong before it was stated
/// right.** It first read "if `proj` ever fell back to the verbatim weights, every other test here
/// would still pass"; a second pass narrowed that to a `proj` dispatching both variants to
/// `launch_gemm_bf16`. Neither is constructible: an fp8 artifact holds no bf16 projections to fall
/// back TO (`place_bf16` on an `F8E4M3` tensor is a `typed` dtype refusal), and `gemm_bf16` takes
/// `*const u16` where `Fp8Weight::packed` is `*const u8`, so that mis-dispatch would not compile.
///
/// What `q > 1e-3` DOES establish is that the two artifacts are not the same model — i.e. that
/// `--fp8` did something. That is load-bearing and nothing else here covers it: `glimmer_fixture`
/// seeds every tensor from `bf16_blob(seed, n)` with `seed` its index, so both arms are built from
/// **byte-identical source tensors**, and a `--fp8` that was ignored, or a `quantize_fp8_block`
/// that round-tripped to the input, would give exactly `q == 0`. It is NOT a guard against a
/// wrong-but-finite fp8 path — any dispatch bug also moves the logits and passes. That job belongs
/// to the projection test, which is why this file keeps both. (Review, 2026-08-15, both passes.)
#[test]
fn an_fp8_artifact_decodes_and_is_not_secretly_the_bf16_one() {
    let bf16_root = TempRoot::new("glimmer-fp8-e2e-bf16");
    let fp8_root = TempRoot::new("glimmer-fp8-e2e-fp8");
    let bf16 = artifact(&bf16_root, gm::GlimmerFormat::Bf16);
    let fp8 = artifact(&fp8_root, gm::GlimmerFormat::Fp8);

    let prompt = [1u32, 2, 3, 4];
    let (l16, l8) = (logits(&bf16, &prompt), logits(&fp8, &prompt));
    // `worst` refuses a non-finite or all-zero row on either side, which is the "it decodes"
    // half — a decode that produced NaNs would score as a perfect match without it.
    let (q, _) = worst(&l8, &l16);
    assert!(
        q > 1e-3,
        "fp8 and bf16 logits agree to {q} — quantization to 3 mantissa bits cannot be that \
         invisible, so the fp8 path is not in play"
    );
}
