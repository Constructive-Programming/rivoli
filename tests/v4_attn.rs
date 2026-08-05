//! **The V4 attention kernels, scored against S1b's oracle.** S2b of
//! `docs/investigations/v4-flash-port.md`, plus S3's compressed-layer cell.
//!
//! # Two cells, and the second one is 41 of the model's 43 layers
//!
//! `LAYER` is ratio-0 and was the only cell until 2026-08-05. It is the layer class the
//! model has **two** of. `COMP_LAYER` is `compress_ratio == 8` — a compressor, YaRN, and a
//! selection whose columns are `window + n_comp` wide — and it is the class the other 41
//! have. Until that cell existed, the `io.cache` tail layout, the prefill persist copy into
//! it, the decode slot `window + start_pos/ratio`, and compressed columns arriving at
//! `sparse_attn` were executed by no test in this tree.
//!
//! The compressed cell drives the whole block step the way S3's layer loop must:
//! `v4compress::compress`, its output placed at BOTH destinations from ONE call, then
//! `attention`. What it does NOT cover is stated at `COMP_LAYER` — the ratio-4 class, whose
//! `Indexer` returns score-ORDERED blocks where `v4_topk_idxs` returns them positionally.
//!
//! Every defect available in this path is silent-wrong — a missing QK-norm, RoPE on the
//! wrong pairing, `attn_sink` treated as a real key, a mis-grouped `wo_a`. None crash and
//! `distinct`/`longest repeated block` cannot see any of them (CLAUDE.md). So the kernels
//! are compared against `src/v4oracle/`'s goldens, and this file is written so that the
//! comparison is *shown* to have the resolution it claims rather than assumed to.
//!
//! # How the block is partitioned, and why that is all one call
//!
//! `v4::attention` leaves four of the five goldens in scratch buffers it does not
//! overwrite — `.q`, `.kv_entry`, `.attn_derot`, `.attn_out` — so one call is compared at
//! four points and each disagreement is attributable to a stage:
//!
//! | golden | what a disagreement implicates |
//! |---|---|
//! | `.q` | `wq_a` → `q_norm` → `wq_b` → QK-norm → RoPE |
//! | `.kv_entry` | `wkv` → `kv_norm` → RoPE → the partial block-64 `act_quant` |
//! | `.attn_core_out` | `sparse_attn` ALONE — driven separately, from the oracle's own `.q` and `.kv_entry` |
//! | `.attn_derot` | the output de-rotation, given the three above |
//! | `.attn_out` | the grouped `wo_a` and `wo_b` |
//!
//! The pipeline is deliberately NOT re-spelled here: re-running the launch sequence in a
//! test would duplicate `src/attn.rs` (a build error under jscpd) and would test a second
//! copy of the wiring rather than the shipped one.
//!
//! # The tolerance is measured, and its resolution is proved
//!
//! Every tensor compared holds bf16 values on BOTH sides, so the natural unit is the
//! bf16 ULP and it is exact — no epsilon is chosen. What separates the kernels from the
//! oracle is re-association: `dot_fp8_wave`'s wave reduction and this file's block
//! reductions fold in a different order than the oracle's sequential `for`. FP
//! contraction is off in the V4 kernels precisely so that this is the *only* difference
//! (see `kernels/mla.hip`), which is what keeps the floor low enough to be useful.
//!
//! `each_in_scope_defect_is_further_away_than_the_kernels_are` is the half that is
//! usually missing. It measures, for every breakage in S2b's scope, the distance from
//! the GPU output to the oracle running WITH that defect, and requires it to dwarf the
//! distance to the clean oracle. That proves the comparison can reject a wrong
//! implementation without putting a break switch in a shipped kernel — and a break
//! switch is what would otherwise be needed, since a kernel cannot be asked to be wrong
//! on purpose without shipping the means to make it wrong.
//!
//! # What this file provably cannot detect
//!
//! Two defects in S2b's scope are excluded from the defect list below rather than silently
//! passing inside it: the QK-norm's position relative to the RoPE, and the KV `act_quant`'s
//! block size. Both are argued at their call sites in `src/attn.rs` from `model.py`.
//!
//! **Their two exclusions are NOT the same fact, and this said they were until 2026-08-05.**
//! `KvActQuantBlock128` is bit-inert **on these fixtures**, which `expect_moves` asserts as a
//! measurement and not as a theorem — `src/attn.rs`'s `KV_QUANT_BLOCK` doc records the
//! scale-invariance derivation as CORRECTED, with counter-evidence at the real weights
//! (`ratio4/prefill`: 5/32768 differing clean, 6/32768 broken), so "powers of two, therefore
//! invariant" is right in kind and wrong at a rounding boundary. It holds here because the
//! toy's in-block dynamic range does not reach one.
//! `QkNormAfterRope` is **not** inert: measured on `COMP_LAYER` it moves four goldens, on
//! 4–13% of their elements, at `rel` 2e-3..9e-3. The mathematical argument for invisibility
//! is sound (RoPE preserves `q.square().mean(-1)`, a scalar commutes with a rotation) but
//! `Oracle::qk_norm` computes that statistic in **bf16**, so `rs` is quantized to ~0.4%
//! steps and the two orders land on different steps. It is a rounding difference, which
//! `tests/v4_oracle.rs::qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one` bounds
//! against the cost of dropping bf16 rounding entirely — it is not an invisibility.
//!
//! A THIRD hole is in this file's own metric rather than in the oracle: `mono` rounds
//! both sides to bf16 before differencing, so a kernel that stopped rounding its stores
//! would score zero ULP. That one IS closed, by `Score::unrounded` — but by a property
//! of the GPU output, not by the goldens, which is why it is named here and not in
//! `in_scope()`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

// Both V4 configs are in scope in this file and they are deliberately separate types
// (`src/v4oracle/weights.rs`: the instrument must not share code with the thing it
// judges). Aliased to name WHICH one -- `V4Cfg` next to `V4Config` is one abbreviation
// apart and survives no skim.
use rivoli::artifact::model::V4Config as EngineV4Config;
use rivoli::artifact::quant::e8m0;
use rivoli::v4compress::{
    Buffers, Finish, Geom, LayerKind, RopeParams, compress, freqs_cis, rope_for_layer,
};
use rivoli::attn::{
    v4::{Dims, Fp8W, Io, Scratch, Step, Weights, attention},
    v4_rope_table_ratio0, v4_topk_idxs, Sel,
};
use rivoli::backend::hip::{
    device_sync, launch_v4_act_quant, launch_v4_gemv_fp8, launch_v4_rope,
    launch_v4_sparse_attn, memcpy_dtod,
};
use rivoli::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3};
use rivoli::v4oracle::numerics::{FP8_MAX, act_quant_inplace, fast_round_scale};
use rivoli::memory::device::DeviceBuf;
use rivoli::v4oracle::forward::{Capture, Defect, Oracle, Step as OStep};
use rivoli::v4oracle::toy::{self, ToyModel};
use rivoli::v4oracle::weights::{NamedRng, V4Config, WMat};

mod common;
use common::{bf16_rows, f32b, f32v, flat_freqs};

/// The ratio-0 layer S2b is scored on. Toy layer 0 has `compress_ratio == 0`: no
/// compressor, no indexer, no YaRN, base `rope_theta` — exactly the shape S2b owns, and
/// exactly the shape `tests/v4_oracle.rs` warns is the least representative layer in the
/// model. That is fine here and NOT fine for the port: S2c owns the rest.
const LAYER: usize = 0;

/// The COMPRESSED layer, and the shape 41 of the model's 43 layers actually have.
///
/// Toy layer 3 is `compress_ratio == 8`: [`LayerKind::NonOverlap`], a compressor, **no
/// indexer**, YaRN and `compress_rope_theta`. Layer 2 (ratio 4) is the other compressed
/// class and is deliberately NOT the cell here — its `Indexer` returns
/// **score-ordered** blocks (`forward.rs:776-782`) where [`v4_topk_idxs`] returns them
/// positionally, so engine and oracle would fold `sparse_attn`'s softmax over the same SET
/// in a different ORDER and every disagreement below would be uninterpretable. That is the
/// gap `docs/investigations/v4-flash-port.md` §"The pre-indexer shortcut is narrower than it
/// sounds" records, and it is named in this file's header rather than papered over. What is
/// covered here — the `io.cache` tail layout, the persist copy, the decode slot and the
/// compressed columns reaching `sparse_attn` — is identical arithmetic on both classes:
/// `attention` branches on neither the ratio nor `has_indexer`, only on
/// [`Sel::shape`]'s `n_comp`.
const COMP_LAYER: usize = 3;

/// Prompt long enough to outrun the toy's 8-slot window, so the ring wraps and
/// `PrefillRingWritesFirstWindow` is reachable at all. A prompt that fits the window
/// makes that defect structurally unable to fire.
///
/// It is also `>= ratio` on [`COMP_LAYER`] (`should_compress` is `seqlen >= ratio` at
/// prefill, `src/v4compress.rs`), so the prefill emits a compressed block — below 8 it
/// emits none and every compressed assertion would pass vacuously.
const PROMPT: usize = 12;
const DECODES: usize = 2;

// ═══ device plumbing ════════════════════════════════════════════════════════════════

fn dev_f32(v: &[f32]) -> DeviceBuf {
    dev_bytes(&f32b(v))
}

fn dev_i32(v: &[i32]) -> DeviceBuf {
    dev_bytes(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
}

fn dev_bytes(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len()).expect("v4: device alloc");
    d.copy_in_at(0, b).expect("v4: device fill");
    d
}

fn read(b: &DeviceBuf) -> Vec<f32> {
    f32v(&b.copy_out().expect("v4: device read"))
}

/// An fp8 weight uploaded as the engine's real path holds it: the checkpoint's e4m3
/// bytes verbatim, and its `F8_E8M0` scale codes WIDENED to f32.
///
/// The widening goes through `artifact::quant::e8m0` — the engine's decoder, the one
/// `format.rs::copy_fp8_e8m0` uses at conversion — and not through the oracle's
/// `e8m0_decode`. If the two disagreed, sharing one here would hide it; the oracle
/// dequantizes the same bytes on its side through `WMat::row`, so the comparison covers
/// the pair.
struct Fp8Buf {
    w: DeviceBuf,
    s: DeviceBuf,
}

impl Fp8Buf {
    fn new(m: &WMat) -> Self {
        let WMat::Fp8 { w, s, .. } = m else {
            panic!("v4: expected an fp8 weight, the kernels read no other format here");
        };
        let scales: Vec<f32> = s.iter().map(|&c| e8m0(c).expect("e8m0 scale code")).collect();
        Self { w: dev_bytes(w), s: dev_f32(&scales) }
    }
    fn ptr(&self) -> Fp8W {
        Fp8W { w: self.w.ptr(), scale: self.s.ptr().cast() }
    }
}

// ═══ the fixture ════════════════════════════════════════════════════════════════════

/// The toy model with `wo_a` replaced by an fp8 weight on **both** layers this file drives.
///
/// `toy::build` stores `wo_a` DENSE, because that is what the reference holds after
/// `convert.py` dequantizes it. The engine reads the checkpoint's fp8 bytes and
/// dequantizes on the fly instead, which is not an approximation over the scale range
/// weight tensors use — `fp8_times_a_power_of_two_is_exact_in_bf16_over_the_range_the_checkpoint_uses` in `tests/v4_attn_host.rs` checks every e4m3 code against scale codes
/// 40..=200 and exhibits the tiny-scale boundary where it would stop. The real
/// `layers.0.attn.wo_a` carries 33,554,432 weight bytes byte-identically from the
/// checkpoint and its 2048 scales widen from `F8_E8M0` to f32 bit-exactly, with codes
/// spanning 115..=117 — measured against the source shard once, by hand, on 2026-08-04.
/// That was a one-off reading and nothing in this tree re-checks it, so treat it as
/// provenance for the choice and not as a live gate; the live gate is the exactness
/// sweep named above.
///
/// Swapping in an `Fp8` `WMat` therefore makes both sides read the SAME values by
/// construction, and keeps the comparison about arithmetic rather than about a format
/// difference the plan's DECIDED note already settled.
fn fixture() -> (V4Config, ToyModel) {
    let cfg = V4Config::toy();
    let mut m = toy::build(&cfg);
    let (rows, cols) = (cfg.o_groups * cfg.o_lora_rank, cfg.n_heads * cfg.head_dim / cfg.o_groups);
    // BOTH layers this file drives. `Gpu::new` calls `Fp8Buf::new` on `wo_a`, which panics
    // on a `Dense`, so a layer added to the drive list without being added here fails
    // loudly at upload rather than silently comparing a different weight.
    //
    // ONE RNG, hoisted, keeping the ORIGINAL name — so `LAYER` (drawn first) gets exactly
    // the byte stream it got before `COMP_LAYER` existed and `COMP_LAYER` gets the
    // continuation. A per-layer seed was written here first and is a silent re-fixturing:
    // `NamedRng::new` is FNV-1a over the name, so `…-L0` is a different stream, and the
    // ratio-0 cell asserts its measurements EXACTLY (`floor == 0`, `r >= 8`, both dated
    // "MEASURED 2026-08-05 on gfx1151"). It is the same hazard the `h-{tag}` seed comment in
    // `drive_script` argues against, and it was violated here in the same change.
    let mut r = NamedRng::new("v4-s2b-wo_a-fp8");
    for layer in [LAYER, COMP_LAYER] {
        // e4m3 codes, NaN (S.1111.111) excluded — the checkpoint contains none and a NaN
        // weight would make every comparison below vacuously "different".
        let w: Vec<u8> = (0..rows * cols)
            .map(|_| {
                let c = r.below(256) as u8;
                if c & 0x7f == 0x7f { 0 } else { c }
            })
            .collect();
        // Scale codes in a narrow band around 2^0 so the dequantized weight has the
        // magnitude a trained tensor does; the real layer 0's codes span 115..=117.
        let s: Vec<u8> = (0..rows.div_ceil(128) * cols.div_ceil(128))
            .map(|_| (120 + r.below(8)) as u8)
            .collect();
        m.layers[layer].wo_a = WMat::Fp8 { rows, cols, w, s };
    }
    (cfg, m)
}

/// The ratio-0 descriptor, for the tests that drive [`LAYER`] directly.
///
/// `index_topk` is 0 because `Sel::n_comp` reads it only under `kind.has_indexer()`, which
/// `Plain` is not — so the value is unreachable rather than chosen. **`Gpu::new` passes
/// `cfg.index_topk` for the same layer**, and the two disagree harmlessly today for exactly
/// that reason. Recorded rather than reconciled because reconciling means threading a
/// `V4Config` to three call sites that have no other use for one; if `index_topk` ever
/// becomes meaningful on a `Plain` layer, this is the divergence to close first.
///
/// This said "Every layer this file drives is ratio-0" until 2026-08-05. It drives two, and
/// the claim about being the one place a new `Sel` field lands moved to [`base_sel`].
fn plain_sel(d: &Dims) -> Sel {
    base_sel(d, LayerKind::Plain, 0)
}

/// A `Sel` for a layer of any class, and **the one place a field added to `Sel` lands** —
/// every other `Sel` in this file is built from a `base_sel` with `..`.
fn base_sel(d: &Dims, kind: LayerKind, index_topk: usize) -> Sel {
    Sel { win: d.window, kind, index_topk, seqlen: 1, start_pos: 0 }
}

fn dims(cfg: &V4Config) -> Dims {
    Dims {
        dim: cfg.dim,
        n_heads: cfg.n_heads,
        head_dim: cfg.head_dim,
        rope_head_dim: cfg.rope_head_dim,
        q_lora_rank: cfg.q_lora_rank,
        o_groups: cfg.o_groups,
        o_lora_rank: cfg.o_lora_rank,
        window: cfg.window_size,
        norm_eps: cfg.norm_eps,
    }
}

/// One captured step: which phase, how many query rows, and at what position.
struct Phase {
    tag: String,
    m: usize,
    start_pos: usize,
    cap: Capture,
}

/// The ratio-0 schedule: one prefill, then [`DECODES`] contiguous decode steps.
fn plain_script() -> Vec<(usize, usize)> {
    std::iter::once((PROMPT, 0)).chain((0..DECODES).map(|i| (1, PROMPT + i))).collect()
}

/// The last decode position of [`comp_script`], reached by SKIPPING one that would have
/// completed a block.
///
/// `31` because at ratio 8 the emitting positions after the prefill are 15, 23 and 31, and
/// this script runs 15 and 31 but not 23. The gap is the whole reason the position exists:
/// the decode slot is `start_pos / ratio` (blocks 1 then 3) and the plausible wrong rule is
/// "the next free slot" (blocks 1 then 2). **The two agree on every contiguous script**, and
/// a block is skipped only when positions are, so without a gap requirement 2 is untestable
/// rather than merely untested.
///
/// # This is a SYNTHETIC state, and what it does and does not certify
///
/// An earlier version of this said speculative decode "produces exactly this gap in
/// production". **It does not**, and the claim is retracted: `compress` refuses `seqlen > 1`
/// at `start_pos > 0`, so an engine accepting two speculative tokens must call it once per
/// POSITION. A correct engine never skips one; a gap is a bug, not a mode.
///
/// So the state at position 31 is off-reference in a way worth naming. Both implementations
/// deposit at slot `start_pos % ratio` and never clear, so block 3 is pooled from the slots
/// last written at positions {16, 9, 10, 11, 12, 13, 14, 31} rather than 24..31, and it is
/// RoPE'd at `(31/8)*8 = 24`. Block 2 is never written and both sides read it as zeros (the
/// GPU cache is zero-initialised in `Gpu::new`, the oracle's `CompState::cache` is
/// `vec![0.0]`), while the selection at 31 names blocks 0..3.
///
/// That makes the last step a **state-machine** probe, not a reference-faithful one: it
/// certifies that the engine and the oracle implement the same deposit/emit/slide machine and
/// that the emitted block lands at `start_pos / ratio`. It does NOT certify that the value is
/// what `model.py` would compute for a real sequence — no real sequence reaches this state.
/// The reference-faithful evidence is the prefill and the contiguous decodes before it.
const COMP_SKIP_TO: usize = 31;

/// The compressed-layer schedule.
///
/// Only the properties NOT observable downstream are asserted here. Two more were written
/// and cut on review: "the prefill emits" and "the prompt outruns the ring" are both checked
/// from the oracle's own MEASURED counters in
/// `attention_matches_the_oracle_on_a_compressed_layer_in_both_phases` and in
/// `each_defect_moves_exactly_the_compressed_layer_goldens_it_should`'s
/// `PrefillRingWritesFirstWindow` classification — a measurement beats an arithmetic
/// prediction of the same fact.
fn comp_script(cfg: &V4Config) -> Vec<(usize, usize)> {
    let ratio = cfg.compress_ratio(COMP_LAYER);
    let mut v = vec![(PROMPT, 0)];
    // Through PROMPT+4 contiguously, which reaches the first decode-completed block (15)
    // and then one step past it (16) so that block is read on a LATER call and not only on
    // the one that wrote it.
    v.extend((PROMPT..=PROMPT + 4).map(|p| (1, p)));
    v.push((1, COMP_SKIP_TO));
    let emits = |p: usize| (p + 1).is_multiple_of(ratio);
    assert!(
        v.iter().filter(|&&(s, p)| s == 1 && emits(p)).count() >= 2,
        "at least two decode-completed blocks, else the RoPE position `(start_pos/ratio)*ratio` \
         and the block index `start_pos/ratio` are both 0 and cannot be told apart"
    );
    // The gap, asserted as a gap: some position this script SKIPS would have emitted.
    let ran: Vec<usize> = v.iter().filter(|&&(s, _)| s == 1).map(|&(_, p)| p).collect();
    let last = *ran.last().expect("the script has decode steps");
    assert!(
        (PROMPT..last).any(|p| emits(p) && !ran.contains(&p)),
        "no skipped emitting position: `start_pos / ratio` and `next free slot` agree on this \
         script, so requirement 2 would pass vacuously"
    );
    assert!(
        last / ratio < cfg.max_seq_len / ratio,
        "block {} does not fit the compressed region",
        last / ratio
    );
    v
}

/// Run the oracle over one layer and one `(seqlen, start_pos)` script, capturing every
/// golden.
///
/// `h` is drawn fresh per step: nothing on the attention path depends on where the residual
/// stream came from, and a deterministic draw keeps the fixture from depending on the MoE
/// half of the block agreeing first.
///
/// The script is a parameter because the compressed cell needs one the ratio-0 cell does
/// not — see [`comp_script`], whose deliberate GAP is what makes the decode slot
/// arithmetic observable at all.
fn drive_script(
    cfg: &V4Config,
    m: &ToyModel,
    defect: Defect,
    layer: usize,
    script: &[(usize, usize)],
) -> Vec<Phase> {
    let o = Oracle::new(cfg.clone(), defect);
    let mut st = o.fresh_state(layer);
    let mut out = Vec::new();
    for (k, &(s, start_pos)) in script.iter().enumerate() {
        let tag = if k == 0 { "pre".to_string() } else { format!("dec{}", k - 1) };
        // Seeded by PHASE only, not by layer: the ratio-0 cell's measured floor and every
        // separation in `each_in_scope_defect_is_further_away_than_the_kernels_are` were
        // taken against these activations, and re-seeding per layer would silently move
        // numbers this file asserts exactly.
        let mut h = draw(&format!("h-{tag}"), s * cfg.hc_mult * cfg.dim);
        let ids: Vec<u32> =
            (0..s).map(|i| ((start_pos + i) * 7 % cfg.vocab_size) as u32).collect();
        let mut cap = Capture::default();
        let step =
            OStep { lw: &m.layers[layer], layer, input_ids: &ids, phase: &tag, s, start_pos };
        o.run_layer(&step, &mut st, &mut h, &mut cap);
        out.push(Phase { tag: step.tag(), m: s, start_pos, cap });
    }
    out
}

fn draw(name: &str, n: usize) -> Vec<f32> {
    let mut r = NamedRng::new(name);
    (0..n).map(|_| r.unit() * 0.5).collect()
}

fn golden<'a>(p: &'a Phase, suffix: &str) -> &'a [f32] {
    captured(p, suffix).unwrap_or_else(|| {
        panic!("golden {}.{suffix} is missing -- the oracle no longer emits it", p.tag)
    })
}

/// The same lookup, for the goldens whose ABSENCE is a fact to compare rather than a
/// failure: `.compressed` exists only on a step the compressor emitted on.
fn captured<'a>(p: &'a Phase, suffix: &str) -> Option<&'a [f32]> {
    p.cap.float(&format!("{}.{suffix}", p.tag))
}

// ═══ scoring ════════════════════════════════════════════════════════════════════════

/// How far apart two bf16-valued tensors are.
///
/// The ULP is the unit because every tensor compared is bf16 on both sides, which makes
/// the distance DISCRETE and exact rather than a chosen epsilon: 0 is bit-identical, 1
/// is the smallest difference the format can express, and a real defect is thousands.
/// `rel` is kept only so the printed line is readable next to the oracle's own metric.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Score {
    max_ulp: i32,
    rel: f32,
    differing: usize,
    total: usize,
    nans: usize,
    /// Elements of `got` that are NOT bf16-representable.
    ///
    /// **Without this the ULP metric has a hole it cannot see.** `mono` puts BOTH sides
    /// through `f32_to_bf16` before differencing, so a kernel that stopped rounding its
    /// stores would keep extra f32 mantissa and still score `max_ulp = 0` — every value
    /// would round back to the same bf16. `Defect::NoBf16Rounding` is exactly that
    /// breakage and `rbf16` appears in every kernel S2b adds, so it is in scope. The
    /// oracle cannot supply the check (its goldens are bf16 on both sides by
    /// construction); it has to be a property of the GPU output on its own.
    unrounded: usize,
}

/// bf16 bit patterns ordered as the numbers they represent, so a subtraction is a ULP
/// count across zero and across the sign.
fn mono(x: f32) -> i32 {
    let b = i32::from(f32_to_bf16(x));
    if b & 0x8000 != 0 { 0x8000 - b } else { b }
}

fn score(got: &[f32], want: &[f32]) -> Score {
    assert_eq!(got.len(), want.len(), "shape disagreement is not a tolerance question");
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
    let mut s =
        Score { max_ulp: 0, rel: 0.0, differing: 0, total: got.len(), nans: 0, unrounded: 0 };
    for (&a, &b) in got.iter().zip(want) {
        // The low 16 bits of a bf16-valued f32 are zero. Checked on `got` only: `want`
        // is the oracle's, which rounds by construction.
        if !a.is_nan() && a.to_bits() & 0xffff != 0 {
            s.unrounded += 1;
        }
        if a.is_nan() != b.is_nan() {
            s.nans += 1;
            continue;
        }
        if a.to_bits() != b.to_bits() {
            s.differing += 1;
            s.max_ulp = s.max_ulp.max((mono(a) - mono(b)).abs());
            s.rel = s.rel.max((a - b).abs() / scale);
        }
    }
    s
}

fn show(what: &str, s: Score) {
    println!(
        "  {what:<26} max_ulp={:<6} differing={:>6}/{:<6} rel={:.3e} nans={}",
        s.max_ulp, s.differing, s.total, s.rel, s.nans
    );
}

/// The bf16-ULP budget a correct kernel must stay inside.
///
/// NOT an arbitrary epsilon. The kernels differ from the oracle only by re-association
/// (FP contraction is off; every elementwise value is reproduced exactly), so a
/// disagreement can only arise when a re-associated f32 sum lands on the other side of a
/// bf16 rounding boundary — which moves a value by exactly one ULP. Anything past that
/// is a second error source, and the point of a tight budget is that it goes red when
/// one appears. `each_in_scope_defect_is_further_away_than_the_kernels_are` shows the
/// separation this buys.
const ULP_BUDGET: i32 = 1;

fn assert_within(what: &str, got: &[f32], want: &[f32]) -> Score {
    let s = score(got, want);
    show(what, s);
    assert_eq!(s.nans, 0, "{what}: NaN on one side only -- {s:?}");
    assert_eq!(
        s.unrounded, 0,
        "{what}: {} values are not bf16-representable, so the kernel stopped rounding a \
         store the reference makes -- and the ULP metric alone could not have seen it, \
         since it rounds both sides. {s:?}",
        s.unrounded
    );
    assert!(s.max_ulp <= ULP_BUDGET, "{what}: over the {ULP_BUDGET}-ULP budget -- {s:?}");
    s
}

// ═══ the harness ════════════════════════════════════════════════════════════════════

/// One layer's COMPRESSOR, device-resident: its weights, its pooling state, and the
/// geometry that pairs with them.
///
/// Present only on a compressed layer. `Geom::attention` returns `None` for
/// [`LayerKind::Plain`], so the `Option` here is that refusal carried through rather than a
/// second decision about which layers have a compressor.
struct Comp {
    geom: Geom,
    wkv: DeviceBuf,
    wgate: DeviceBuf,
    ape: DeviceBuf,
    norm: DeviceBuf,
    kv_state: DeviceBuf,
    score_state: DeviceBuf,
    kv: DeviceBuf,
    score: DeviceBuf,
    out: DeviceBuf,
}

impl Comp {
    /// `None` on a ratio-0 layer, in both halves at once: `Geom::attention` refuses
    /// `LayerKind::Plain` and `toy::build` gives such a layer no `CompressorW`. Asserted
    /// rather than assumed — the two must agree, and a layer with one and not the other is
    /// a fixture bug that would otherwise surface as a null-pointer launch.
    fn new(
        cfg: &V4Config,
        lw: &rivoli::v4oracle::forward::LayerW,
        kind: LayerKind,
        max_m: usize,
    ) -> Option<Self> {
        let geom = Geom::attention(kind, cfg.head_dim, cfg.rope_head_dim, cfg.norm_eps);
        let (Some(geom), Some(cw)) = (geom, lw.compressor.as_ref()) else {
            assert!(
                geom.is_none() && lw.compressor.is_none(),
                "the layer's Geom and its CompressorW disagree about whether it compresses"
            );
            return None;
        };
        let (cd, ents) = (geom.cd(), geom.ents());
        assert_eq!(cw.ape.len(), geom.ratio() * cd, "ape is [ratio, coff*d]");
        Some(Self {
            geom,
            wkv: dev_bytes(&common::u16b(&bf16_rows(&cw.wkv))),
            wgate: dev_bytes(&common::u16b(&bf16_rows(&cw.wgate))),
            ape: dev_f32(&cw.ape),
            norm: dev_f32(&cw.norm),
            kv_state: dev_f32(&vec![0.0f32; ents * cd]),
            // **-inf**, not zero. A never-written slot must weigh `exp(-inf - m) == 0` in
            // the pooling softmax; zero makes it a live entry at `exp(0 - m)`, which is a
            // plausible number and a wrong window. S3 requirement 3.
            score_state: dev_f32(&vec![f32::NEG_INFINITY; ents * cd]),
            kv: dev_f32(&vec![0.0f32; max_m * cd]),
            score: dev_f32(&vec![0.0f32; max_m * cd]),
            out: dev_f32(&vec![0.0f32; max_m.div_ceil(geom.ratio()) * geom.d()]),
        })
    }
}

/// The compressed layers' rotary parameter set, from the oracle's config.
///
/// A function with TWO callers — `Gpu::new` and
/// `the_two_rope_table_constructions_agree_on_the_un_yarned_table` — which is what earns it.
/// It was inlined for one round on the argument that `RopeParams` the TYPE is what keeps
/// `compress_rope_theta` and `original_seq_len` travelling together (true, and the reason
/// `rope_for_layer` takes the struct); the second caller made the literal a duplicate that
/// `build.rs`'s gate refused, which is the better argument.
fn compressed_rope(cfg: &V4Config) -> RopeParams {
    RopeParams {
        rope_head_dim: cfg.rope_head_dim,
        theta: cfg.compress_rope_theta,
        original_seq_len: cfg.original_seq_len,
        factor: cfg.rope_factor,
        beta_fast: cfg.beta_fast,
        beta_slow: cfg.beta_slow,
    }
}

/// Every device buffer one layer's attention needs, allocated once for the largest step.
struct Gpu {
    _w: Vec<Fp8Buf>,
    _norms: Vec<DeviceBuf>,
    weights: Weights,
    xq: DeviceBuf,
    qr: DeviceBuf,
    qrq: DeviceBuf,
    q: DeviceBuf,
    kv: DeviceBuf,
    o: DeviceBuf,
    y: DeviceBuf,
    ring: DeviceBuf,
    out: DeviceBuf,
    freqs: DeviceBuf,
    /// Rows in `ring` — `window + max_seq_len/ratio`. Carried because `DeviceBuf` has no
    /// length and [`Gpu::poke`] indexes it by row; the same reason
    /// `tests/v4_compress_kernel.rs::Dev` carries an element count.
    ring_rows: usize,
    max_m: usize,
    /// This layer's class and `index_topk`, fixed at construction. `attention` overwrites
    /// `win`/`seqlen`/`start_pos` itself, so the only fields that travel from here are the
    /// two a caller could get wrong — and getting `kind` wrong is a layer's compressed
    /// columns silently vanishing.
    sel: Sel,
    comp: Option<Comp>,
}

impl Gpu {
    fn new(cfg: &V4Config, model: &ToyModel, d: &Dims, max_m: usize, layer: usize) -> Self {
        let lw = &model.layers[layer];
        let kind = LayerKind::from_ratio(cfg.compress_ratio(layer));
        let w: Vec<Fp8Buf> =
            [&lw.wq_a, &lw.wq_b, &lw.wkv, &lw.wo_a, &lw.wo_b].map(Fp8Buf::new).into();
        let norms: Vec<DeviceBuf> =
            [&lw.q_norm, &lw.kv_norm, &lw.attn_sink].map(|v| dev_f32(v)).into();
        let weights = Weights {
            wq_a: w[0].ptr(),
            q_norm: norms[0].ptr().cast(),
            wq_b: w[1].ptr(),
            wkv: w[2].ptr(),
            kv_norm: norms[1].ptr().cast(),
            attn_sink: norms[2].ptr().cast(),
            wo_a: w[3].ptr(),
            wo_b: w[4].ptr(),
        };
        let z = |n: usize| dev_f32(&vec![0.0f32; n]);
        let nhd = d.n_heads * d.head_dim;
        // The compressed region's capacity, in BLOCKS. `max_seq_len / ratio` is the
        // reference's own sizing of `kv_cache[:, window_size:]`, and 0 on a ratio-0 layer.
        let blocks = kind.compressor_ratio().map_or(0, |r| cfg.max_seq_len / r);
        Self {
            _w: w,
            _norms: norms,
            weights,
            max_m,
            ring_rows: d.window + blocks,
            sel: base_sel(d, kind, cfg.index_topk),
            comp: Comp::new(cfg, lw, kind, max_m),
            xq: z(max_m * d.dim),
            qr: z(max_m * d.q_lora_rank),
            qrq: z(max_m * d.q_lora_rank),
            q: z(max_m * nhd),
            // `[rows, head_dim]` on a ratio-0 layer, `[rows + rows/ratio, head_dim]` on a
            // compressed one: at prefill `sparse_attn` reads `torch.cat([kv, kv_compress])`
            // and the selection indexes that concatenation as ONE space, so the compressor's
            // blocks live in this buffer's tail. `Scratch`'s own doc states the rule and
            // says nothing can check it.
            kv: z((max_m + kind.compressor_ratio().map_or(0, |r| max_m / r)) * d.head_dim),
            o: z(max_m * nhd),
            y: z(max_m * d.o_groups * d.o_lora_rank),
            // `[window_size + max_seq_len/ratio, head_dim]`: the ring FIRST, then the
            // compressed region. Contiguous and in that order because decode attends the
            // whole thing and the selection's compressed columns are `window_size + block`.
            // Zeroed, and that matters: a block the script never emits is read as zeros by
            // BOTH sides (the oracle's `CompState::cache` is zero-initialised too), so a
            // skipped block is an agreed value rather than an unmapped read.
            ring: z((d.window + blocks) * d.head_dim),
            out: z(max_m * d.dim),
            // The layer's own table. `rope_for_layer` is the ONE selector — it moves
            // `compress_rope_theta` and `original_seq_len` together, which is what makes
            // `Defect::RopeNoYarn` unrepresentable for a caller that goes through it
            // (`docs/investigations/v4-flash-port.md`, the owed "Io built by something that
            // takes LayerKind"). The ratio-0 arm keeps `v4_rope_table_ratio0` because
            // that is the function the ratio-0 ENGINE path calls, so the ratio-0 cell scores
            // the shipped construction rather than a second one built for the test; the
            // compressed arm does the same for `freqs_cis`/`rope_for_layer`. This called it a
            // "cross-check" for one round, which named a comparison nothing performed —
            // `the_two_rope_table_constructions_agree_on_the_un_yarned_table` now performs
            // it, and `src/attn.rs`'s own note says the function had no such credit before.
            freqs: dev_f32(&match kind {
                LayerKind::Plain => {
                    v4_rope_table_ratio0(d.rope_head_dim, cfg.max_seq_len, cfg.rope_theta)
                }
                LayerKind::Overlap | LayerKind::NonOverlap(_) => flat_freqs(&freqs_cis(
                    rope_for_layer(compressed_rope(cfg), cfg.rope_theta, kind),
                    cfg.max_seq_len,
                )),
            }),
        }
    }

    fn scratch(&mut self) -> Scratch {
        Scratch {
            rows: self.max_m,
            xq: self.xq.ptr_mut().cast(),
            qr: self.qr.ptr_mut().cast(),
            qrq: self.qrq.ptr_mut().cast(),
            q: self.q.ptr_mut().cast(),
            kv: self.kv.ptr_mut().cast(),
            o: self.o.ptr_mut().cast(),
            y: self.y.ptr_mut().cast(),
        }
    }

    /// Bind one step's input and selection to the persistent ring, table and output.
    /// Spelled once: the six pointers are easy to permute and five of them are the same
    /// type, so a second copy is a second chance to swap `ring` for `out`.
    fn io(&mut self, x: &DeviceBuf, idxs: &DeviceBuf, idxs_shape: (usize, usize)) -> Io {
        Io {
            x: x.ptr().cast(),
            freqs: self.freqs.ptr().cast(),
            idxs: idxs.ptr().cast(),
            idxs_shape,
            cache: self.ring.ptr_mut().cast(),
            out: self.out.ptr_mut().cast(),
        }
    }

    /// One whole block-step for `p`: the compressor, the two placements, then `attention`.
    ///
    /// The order is the reference's (`Attention.forward` model.py:523-537) and it is not
    /// optional — `attention`'s own safety contract says the compressor must already have
    /// run for this same step and must have written BOTH destinations, because `attention`
    /// only READS the compressed rows. Doing it here rather than at the call sites is what
    /// makes that impossible to forget in one test and not another.
    fn step(&mut self, d: &Dims, p: &Phase) -> StepOut {
        let x = dev_f32(golden(p, "attn_norm_out"));
        let placed = self.compress_and_place(d, &x, p);
        let mut out = self.attend(d, p);
        // Read back from where `sparse_attn` read them, NOT from the compressor's own
        // `out` buffer: a placement that landed the blocks somewhere else entirely would
        // still make that buffer compare clean, and WHERE they land is this cell's subject.
        if let Some((off, blocks)) = placed {
            let src = if p.start_pos == 0 { &self.kv } else { &self.ring };
            out.compressed = read(src)[off * d.head_dim..(off + blocks) * d.head_dim].to_vec();
        }
        out
    }

    /// The `attention` half of a step, WITHOUT the compressor.
    ///
    /// Separate because it is **idempotent** and [`Gpu::compress_and_place`] is not: a decode
    /// call recomputes `q`/`kv` from the same `x` and rewrites ring slot `pos % window` with
    /// the same bytes, while `compress` read-modify-writes `kv_state`/`score_state` before
    /// the emit decision and slides the pooling window. So a probe that wants to re-attend
    /// over a perturbed cache must call THIS, and calling `step` twice would corrupt exactly
    /// the state S3 requirement 3 is about — the trap
    /// `src/v4compress.rs::compress` documents at its "never a second call".
    fn attend(&mut self, d: &Dims, p: &Phase) -> StepOut {
        let x = dev_f32(golden(p, "attn_norm_out"));
        let mut idx = Vec::new();
        let sel = Sel { seqlen: p.m, start_pos: p.start_pos, ..self.sel };
        let shape = v4_topk_idxs(sel, &mut idx).unwrap();
        let idxb = dev_i32(&idx);
        let io = self.io(&x, &idxb, shape);
        let step = if p.start_pos == 0 {
            Step::Prefill { seqlen: p.m }
        } else {
            Step::Decode { pos: p.start_pos }
        };
        let s = self.scratch();
        // SAFETY: every buffer above outlives the `device_sync` on the next line.
        unsafe { attention(d, sel, &self.weights, &s, &io, step) }.expect("v4 attention");
        device_sync().expect("sync");
        let n = |b: &DeviceBuf, len: usize| read(b)[..len].to_vec();
        let nhd = d.n_heads * d.head_dim;
        StepOut {
            compressed: Vec::new(),
            // The window half is THIS selection with the compressor removed, taken from
            // `Sel::shape` rather than re-derived: a second copy of
            // `if start_pos == 0 { seqlen.min(win) } else { win }` in the harness could
            // drift from the engine's and would then report a compressed column count that
            // was wrong in exactly the direction that hides a missing one.
            n_comp: shape.1 - Sel { kind: LayerKind::Plain, ..sel }.shape().unwrap().1,
            // The largest index the selection actually names. `n_comp > 0` says the
            // compressed COLUMNS exist; this says they are not all `-1`, which is the
            // vacuous way for a compressed selection to be present and read nothing.
            max_idx: idx.iter().copied().max().unwrap_or(-1),
            v: [
                n(&self.q, p.m * nhd),
                n(&self.kv, p.m * d.head_dim),
                n(&self.o, p.m * nhd),
                n(&self.out, p.m * d.dim),
            ],
        }
    }

    /// Overwrite one `[head_dim]` row of the persistent cache — ring slot `row` when
    /// `row < window`, compressed block `row - window` otherwise.
    ///
    /// The instrument the compressed region has no other way to get: `sparse_attn` reading
    /// a row is not observable from any golden, but its output moving when that row changes
    /// is. Returns the bytes it replaced, so a probe can put them back and the next one
    /// starts from the same state.
    fn poke(&mut self, d: &Dims, row: usize, fill: f32) -> Vec<f32> {
        let hd = d.head_dim;
        // In bounds, with a message. The slice below would panic on overrun anyway, but as
        // an index panic — and the caller most likely to overrun is a future probe at
        // `window + n_comp` on a step where the selection covers the whole region.
        assert!(row < self.ring_rows, "cache row {row} is past the {} the ring holds", self.ring_rows);
        let was = read(&self.ring)[row * hd..(row + 1) * hd].to_vec();
        self.ring.copy_in_at(row * hd * size_of::<f32>(), &f32b(&vec![fill; hd])).expect("poke");
        was
    }

    fn unpoke(&mut self, d: &Dims, row: usize, was: &[f32]) {
        self.ring.copy_in_at(row * d.head_dim * size_of::<f32>(), &f32b(was)).expect("unpoke");
    }

    /// `Compressor.forward` for this step, with its output placed where the reference puts
    /// it. Returns `(first row, block count)` in whichever buffer holds it, or `None` when
    /// this step emits nothing (a ratio-0 layer, a short prefill, or a decode position that
    /// does not complete a block — all three are the reference's own `return None`).
    ///
    /// **Both destinations at prefill, from ONE call.** `Compressor.forward` assigns
    /// `self.kv_cache[:, :seqlen // ratio]` — the persistent region every LATER decode step
    /// selects by position — *and* returns the same blocks for `Attention.forward` to
    /// `torch.cat` onto the prompt's KV for THIS step. `Finish` carries a single `out`, so
    /// the second destination is a device COPY: `compress` read-modify-writes
    /// `kv_state`/`score_state` before it decides whether to emit, so a second call would
    /// re-deposit the pooling window and slide it again.
    ///
    /// **At decode the slot is `start_pos / ratio`, never "the next free one."** That is
    /// requirement 2 of `docs/investigations/v4-flash-port.md`, and the two rules agree on
    /// every contiguous script — which is why [`comp_script`] has a gap in it.
    fn compress_and_place(
        &mut self,
        d: &Dims,
        x: &DeviceBuf,
        p: &Phase,
    ) -> Option<(usize, usize)> {
        let c = self.comp.as_mut()?;
        let fin = Finish {
            norm: c.norm.ptr().cast(),
            freqs: self.freqs.ptr().cast(),
            out: c.out.ptr_mut().cast(),
        };
        let b = Buffers {
            x: x.ptr().cast(),
            dim: d.dim,
            wkv: c.wkv.ptr().cast(),
            wgate: c.wgate.ptr().cast(),
            ape: c.ape.ptr().cast(),
            fin,
            kv_state: c.kv_state.ptr_mut().cast(),
            score_state: c.score_state.ptr_mut().cast(),
            kv: c.kv.ptr_mut().cast(),
            score: c.score.ptr_mut().cast(),
            scratch_rows: self.max_m,
        };
        // SAFETY: every pointer above is a live `DeviceBuf` field of `self`, at the shape
        // `Buffers` documents; nothing here drops before the `device_sync` in `step`.
        let blocks = unsafe { compress(&c.geom, &b, p.m, p.start_pos) }.expect("v4 compress");
        if blocks == 0 {
            return None;
        }
        let (ratio, hd) = (c.geom.ratio(), d.head_dim);
        let row = hd * size_of::<f32>();
        let src: *const u8 = c.out.ptr();
        let cache = self.ring.ptr_mut().cast::<f32>();
        // SAFETY: as above. Both destinations were sized for the whole compressed region
        // in `Gpu::new`, and `blocks` is bounded by `max_seq_len / ratio` there.
        unsafe {
            if p.start_pos == 0 {
                let tail = self.kv.ptr_mut().cast::<f32>().add(p.m * hd);
                memcpy_dtod(tail.cast(), src, blocks * row).expect("prefill kv tail");
                memcpy_dtod(cache.add(d.window * hd).cast(), src, blocks * row)
                    .expect("prefill cache persist");
                Some((p.m, blocks))
            } else {
                let slot = d.window + p.start_pos / ratio;
                memcpy_dtod(cache.add(slot * hd).cast(), src, blocks * row)
                    .expect("decode cache slot");
                Some((slot, blocks))
            }
        }
    }
}

/// What one `Gpu::step` produced.
///
/// The four `attention` leaves in scratch, plus the two things only a compressed layer has:
/// the blocks as they sit in the buffer `sparse_attn` indexed, and how wide the selection
/// was. `n_comp` is carried so the anti-vacuity assertions can be about a MEASURED column
/// count rather than about the arithmetic that produced it.
struct StepOut {
    v: [Vec<f32>; 4],
    compressed: Vec<f32>,
    n_comp: usize,
    max_idx: i32,
}

/// Everything every test below needs: the fixture, its dimensions, the clean captures
/// and the device buffers.
///
/// Built in one place because the four are a matched set — `gpu` uploads THIS `model`'s
/// weights and `clean` is THIS `model`'s oracle run, and assembling them separately is
/// three chances to score a GPU carrying one fixture against goldens from another.
struct Harness {
    cfg: V4Config,
    model: ToyModel,
    d: Dims,
    clean: Vec<Phase>,
    gpu: Gpu,
    layer: usize,
    script: Vec<(usize, usize)>,
}

impl Harness {
    fn new() -> Self {
        Self::at(LAYER, &plain_script())
    }

    /// The same, for any layer and script. `layer` reaches `drive_script` AND `Gpu::new`
    /// from this one argument, so a harness carrying layer 0's weights against layer 3's
    /// goldens is not constructible.
    fn at(layer: usize, script: &[(usize, usize)]) -> Self {
        let (cfg, model) = fixture();
        let d = dims(&cfg);
        let clean = drive_script(&cfg, &model, Defect::None, layer, script);
        let max_m = script.iter().map(|&(s, _)| s).max().expect("an empty script gates nothing");
        let gpu = Gpu::new(&cfg, &model, &d, max_m, layer);
        Self { cfg, model, d, clean, gpu, layer, script: script.to_vec() }
    }

    /// The compressed cell's harness. The script comes from the harness's OWN `cfg`, so a
    /// test cannot hold a second `V4Config::toy()` beside the one `fixture()` built — the
    /// coupling this struct's doc says it exists to prevent.
    fn compressed() -> Self {
        Self::at(COMP_LAYER, &comp_script(&V4Config::toy()))
    }
}

// ═══ tests ══════════════════════════════════════════════════════════════════════════

#[test]
fn attention_matches_the_oracle_at_every_stage_of_a_ratio_zero_layer() {
    let Harness { d, clean, mut gpu, .. } = Harness::new();
    for p in &clean {
        let got = gpu.step(&d, p);
        assert_eq!(got.n_comp, 0, "a ratio-0 layer has no compressed columns");
        assert_stages(p, &got);
    }
}

/// Score one step's four `attention` goldens against the oracle's.
///
/// Ordered so the FIRST failure is the earliest stage: a wrong `.q` makes every later tensor
/// wrong too, and reporting the last one first would send the reader to `wo_b` for a bug in
/// `wq_a`. One function because both cells need exactly this and `build.rs`'s duplication
/// gate found the second copy.
fn assert_stages(p: &Phase, got: &StepOut) {
    println!("{} (m={}, start_pos={})", p.tag, p.m, p.start_pos);
    for (name, blame, v) in stages(got) {
        assert_within(blame, v, golden(p, name));
    }
}

/// `sparse_attn` alone, driven from the oracle's own `.q` and `.kv_entry`.
///
/// Isolated because it is the only stage whose output `attention` overwrites in place,
/// and because it is where `attn_sink` lives: feeding the oracle's exact inputs means a
/// disagreement here cannot be blamed on an upstream projection.
#[test]
fn sparse_attn_alone_matches_the_oracle_including_the_sink() {
    let Harness { d, clean, model, .. } = Harness::new();
    let sink = dev_f32(&model.layers[LAYER].attn_sink);
    // Prefill only: at prefill `sparse_attn` reads the prompt's own KV, so `.kv_entry`
    // IS the whole of what it attends. At decode it reads the ring, which is state this
    // test does not own -- that path is covered end-to-end by the test above.
    let p = &clean[0];
    let q = dev_f32(golden(p, "q"));
    let kv = dev_f32(golden(p, "kv_entry"));
    let mut idx = Vec::new();
    let (rows, cols) = v4_topk_idxs(Sel { seqlen: p.m, start_pos: 0, ..plain_sel(&d) }, &mut idx).unwrap();
    assert_eq!(rows, p.m);
    let idxb = dev_i32(&idx);
    let mut o = dev_f32(&vec![0.0f32; p.m * d.n_heads * d.head_dim]);
    // SAFETY: all six buffers outlive the sync below.
    unsafe {
        launch_v4_sparse_attn(
            q.ptr().cast(),
            kv.ptr().cast(),
            sink.ptr().cast(),
            idxb.ptr().cast(),
            p.m,
            d.n_heads,
            d.head_dim,
            cols,
            (d.head_dim as f32).powf(-0.5),
            o.ptr_mut().cast(),
        )
    }
    .expect("v4_sparse_attn");
    device_sync().expect("sync");
    println!("{} sparse_attn in isolation", p.tag);
    assert_within("attn_core_out", &read(&o), golden(p, "attn_core_out"));
}

/// The breakages S2b's kernels could actually contain, each of which these goldens must
/// be able to reject. Deliberately NOT the whole `Defect::ALL` set: a defect outside this
/// scope (the compressor, the indexer, the router, mHC) is S2a's or S2c's, and listing it
/// here would claim coverage this file does not provide.
///
/// `QkNormAfterRope` and `KvActQuantBlock128` are absent ON PURPOSE, and — CORRECTED
/// 2026-08-05 — for two DIFFERENT reasons, which this said were one. `KvActQuantBlock128`
/// really is bit-inert on this fixture, so a separation could not be produced.
/// `QkNormAfterRope` is not inert: it moves four goldens, at the bf16 ROUNDING scale (`rel`
/// 2e-3..9e-3, i.e. under one bf16 step), so its distance would measure the rounding floor
/// rather than a defect's reach. Both are excluded; only the first is a blind spot. See the
/// module header and `expect_moves`.
/// Each carries the stage it belongs to, so the printed margin table says WHICH part of
/// the block a given separation is evidence about — a defect in the q path and one in
/// the output projection are not interchangeable evidence, and a flat list of names
/// invites reading them as if they were.
fn in_scope() -> Vec<(&'static str, Defect)> {
    vec![
        ("q path", Defect::SkipQkNorm),
        ("q path", Defect::QkNormUsesQNormWeight),
        ("q/kv rope", Defect::RopeAllDims),
        ("q/kv rope", Defect::RopeFirstDims),
        ("q/kv rope", Defect::RopeHalfSplit),
        ("kv quant", Defect::SkipKvActQuant),
        ("kv quant", Defect::KvActQuantWholeTensor),
        ("kv quant", Defect::KvActQuantNoRoundScale),
        ("attn core", Defect::SkipAttnSink),
        ("attn core", Defect::AttnSinkNotMaxShifted),
        ("kv ring", Defect::PrefillRingWritesFirstWindow),
        ("de-rotation", Defect::SkipOutputDerotation),
        ("de-rotation", Defect::OutputDerotationForward),
        ("wo_a grouping", Defect::WoGroupsSplitHeadDim),
        ("wo_a grouping", Defect::WoGroupsInterleaved),
    ]
}

/// The four goldens one `attention` call leaves behind, paired with the names the
/// capture files them under. Spelled once so the two scoring loops below cannot drift
/// into comparing `.attn_derot` against `.attn_out`.
fn stages(o: &StepOut) -> impl Iterator<Item = (&'static str, &'static str, &[f32])> {
    STAGES.iter().zip(&o.v).map(|(&(name, blame), v)| (name, blame, v.as_slice()))
}

/// The four goldens one `attention` call leaves behind: the name the capture files each
/// under, and what a disagreement there implicates.
///
/// **ONE table, not two parallel arrays.** It was `STAGE_NAMES` and `STAGE_BLAME` for one
/// round, under a comment claiming the split kept the two "from being swapped for each
/// other, which a single tuple list invites" — which is backwards, and both reviews said so.
/// Two `[_; 4]` joined by `zip` is exactly the shape that drifts silently under a reorder;
/// a tuple writes the pairing once and the compiler carries it.
const STAGES: [(&str, &str); 4] = [
    ("q", "q (wq_a..qk_norm..rope)"),
    ("kv_entry", "kv_entry (wkv..act_quant)"),
    ("attn_derot", "attn_derot (de-rotation)"),
    ("attn_out", "attn_out (wo_a, wo_b)"),
];

/// How far the largest move at any stage of any step is, scoring `mine` against the
/// captures in `refs`.
fn reach(refs: &[Phase], mine: &[StepOut]) -> i32 {
    // The zip below TRUNCATES to the shorter side, so a caller pairing `refs` and `mine`
    // from DIFFERENT runs is scored over the intersection and can pass on a smaller reach.
    //
    // Note what this does NOT guard, because the assertion it replaced claimed it and was
    // wrong: **no `Defect` can change the step count.** `drive_script` pushes one `Phase` per
    // script entry unconditionally and nothing in `run_layer` is defect-conditional, so
    // "the defect changed the step schedule" — the message this carried for one round, moved
    // here from the ratio-0 sweep — named a state that cannot occur. What CAN occur is the
    // caller error: this file now has two cells with two scripts of different lengths, and
    // `reach(&ratio0_clean, &compressed_mine)` type-checks.
    assert_eq!(refs.len(), mine.len(), "refs and mine come from different runs");
    refs.iter()
        .zip(mine)
        .flat_map(|(p, v)| stages(v).map(|(name, _, got)| score(got, golden(p, name)).max_ulp))
        .max()
        .expect("no steps captured")
}

#[test]
fn each_in_scope_defect_is_further_away_than_the_kernels_are() {
    let Harness { cfg, model, d, clean, mut gpu, layer, script } = Harness::new();
    // The GPU's own output for every step, taken ONCE and reused: every distance below
    // is measured from the same point, so the numbers are comparable to each other and
    // not merely each to its own baseline.
    let mine: Vec<StepOut> = clean.iter().map(|p| gpu.step(&d, p)).collect();
    let floor = reach(&clean, &mine);
    println!("kernel-vs-oracle floor: {floor} bf16 ULP over {} steps", clean.len());
    // MEASURED 2026-08-05 on gfx1151: 0 ULP at prefill and both decode steps.
    //
    // That is 0 ULP, which is NOT by itself bit-identity: `mono` rounds both sides to
    // bf16 before differencing, so `max_ulp == 0` means "identical after rounding" —
    // the very hole `Score::unrounded` exists to cover, and this test never calls
    // `assert_within`, so it never checks it. Bit-identity is a fact BORROWED from
    // `attention_matches_the_oracle_at_every_stage_of_a_ratio_zero_layer`, which asserts
    // `unrounded == 0` on these same four tensors. Stated as a borrow because writing
    // "i.e. bit-identical" here would assert a check this test does not make.
    //
    // Pinned at 0 rather
    // than at `ULP_BUDGET`, because the budget is what the argument allows and this is
    // what the kernels do — and a silent drift from 0 to 1 is the first observable sign
    // of a second error source appearing.
    //
    // A 1-ULP flip IS theoretically reachable: the block reductions re-associate, and a
    // re-associated f32 sum can land on the other side of a bf16 rounding boundary. If
    // this ever goes red, establish that it is that before relaxing it — the same
    // re-association would move a handful of elements by one ULP, where a real defect
    // moves thousands by tens of thousands. The assertion is DEFERRED to the end of this
    // test so the per-defect table below still prints when it fires: asserting here
    // aborted the run and withheld exactly the evidence this comment sends the reader to
    // weigh.

    let mut worst: Option<(Defect, i32)> = None;
    for (stage, defect) in in_scope() {
        let r = reach(&drive_script(&cfg, &model, defect, layer, &script), &mine);
        println!("  {stage:<14} {defect:<32?} reach={r} ULP");
        if worst.is_none_or(|(_, w)| r < w) {
            worst = Some((defect, r));
        }
        assert!(
            r > floor && r >= 8,
            "{defect:?} ({stage}) moves the goldens by only {r} bf16 ULP against a \
             kernel-vs-oracle floor of {floor}: this comparison could not tell a kernel \
             carrying that defect from a correct one"
        );
    }
    // The tightest separation is the gate's real resolution; print it so a regression in
    // the floor is visible as a shrinking margin before it is visible as a failure.
    let (d_worst, r_worst) = worst.expect("in_scope() is empty");
    println!("tightest margin: {d_worst:?} at {r_worst} ULP against a floor of {floor}");
    assert_eq!(floor, 0, "the kernels are no longer bit-exact against the oracle");

    // ANTI-DRIFT. The oracle owns the defect set; this file names a subset of it. If a
    // breakage is added there, the complement changes and this fails — which forces S2b's
    // scope to be re-decided rather than silently excluding the new one. The two counts
    // are recorded, not derived, for exactly that reason.
    let listed: Vec<Defect> = in_scope().into_iter().map(|(_, x)| x).collect();
    let outside = Defect::breakages().filter(|x| !listed.contains(x)).count();
    // 28 -> 35 on 2026-08-05, by the head-tail stage. Seven added, ALL outside S2b's scope
    // and re-decided rather than assumed: `IndexerBf16RunningSum` is the indexer's per-head
    // score reduction, and the six `Head*` variants live after the last block entirely, so
    // none of them can reach an attention golden. Classification only; `in_scope()` is
    // unchanged, which is why the first count still reads 15.
    assert_eq!(
        (listed.len(), outside),
        (15, 35),
        "the oracle's defect set changed: {} in S2b's scope, {outside} outside. Re-decide \
         which side each new breakage falls on -- and note that `QkNormAfterRope` and \
         `KvActQuantBlock128` are outside because this METRIC cannot resolve them, not \
         because they are someone else's stage. `KvActQuantBlock128` is bit-inert here; \
         `QkNormAfterRope` moves four goldens at the bf16 rounding scale (measured \
         2026-08-05, `expect_moves`). They are not the same case",
        listed.len()
    );
}

/// The selection-shape check, both ways.
///
/// It is the one guard here that protects against a SILENT wiring bug rather than a
/// crash: prefill indices are absolute positions into the prompt's KV and decode indices
/// are ring slots, and a `cols` that is right for one phase reads past the buffer in the
/// other. The rejection is exercised at the shape that actually collides — a prefill
/// SHORTER than the window, where the reference narrows `cols` to `seqlen` and a caller
/// that assumed `window` would not notice — and the acceptance is exercised too, because
/// a guard that rejects everything is not a guard.
#[test]
fn the_selection_shape_guard_rejects_a_short_prefill_and_accepts_a_decode() {
    let Harness { d, clean, mut gpu, .. } = Harness::new();
    let p = &clean[0];
    // A prefill of 4 against a window of 8: `v4_topk_idxs` returns 4 columns, and the
    // whole point is that `window` is the plausible wrong answer.
    let short = 4usize;
    assert!(short < d.window, "the collision this test needs does not exist");
    let x = dev_f32(&golden(p, "attn_norm_out")[..short * d.dim]);
    let mut idx = Vec::new();
    let right = v4_topk_idxs(Sel { seqlen: short, start_pos: 0, ..plain_sel(&d) }, &mut idx).unwrap();
    assert_eq!(right, (short, short), "a short prefill no longer narrows its columns");
    let idxb = dev_i32(&idx);
    let mut io = Io {
        x: x.ptr().cast(),
        freqs: gpu.freqs.ptr().cast(),
        idxs: idxb.ptr().cast(),
        idxs_shape: (short, d.window),
        cache: gpu.ring.ptr_mut().cast(),
        out: gpu.out.ptr_mut().cast(),
    };
    let s = gpu.scratch();
    // SAFETY: buffers outlive the call; it returns before any launch.
    let e = unsafe { attention(
                &d,
                plain_sel(&d), &gpu.weights, &s, &io, Step::Prefill { seqlen: short }) }
        .expect_err("a 4-row prefill must not accept an 8-column selection");
    assert!(format!("{e}").contains("selection"), "rejected for the wrong reason: {e}");

    io.idxs_shape = right;
    // SAFETY: as above; this one does launch, and the sync below joins it.
    unsafe { attention(
                &d,
                plain_sel(&d), &gpu.weights, &s, &io, Step::Prefill { seqlen: short }) }
        .expect("the correct shape must be accepted");
    device_sync().expect("sync");

    // The decode arm, which the first draft of this test named and never ran. Decode
    // always wants `window` columns whatever the prompt was, so the plausible wrong
    // answer here is the narrowed prefill shape -- the exact inverse of the mistake
    // above, and it must be rejected too.
    io.idxs_shape = (1, short);
    let e = unsafe { attention(
                &d,
                plain_sel(&d), &gpu.weights, &s, &io, Step::Decode { pos: PROMPT }) }
        .expect_err("a decode must not accept a narrowed prefill selection");
    assert!(format!("{e}").contains("selection"), "rejected for the wrong reason: {e}");
    let mut one = Vec::new();
    let want = v4_topk_idxs(Sel { seqlen: 1, start_pos: PROMPT, ..plain_sel(&d) }, &mut one).unwrap();
    assert_eq!(want, (1, d.window), "decode no longer wants the full window");
    let oneb = dev_i32(&one);
    io.idxs = oneb.ptr().cast();
    io.idxs_shape = want;
    // SAFETY: as above.
    unsafe { attention(
                &d,
                plain_sel(&d), &gpu.weights, &s, &io, Step::Decode { pos: PROMPT }) }
        .expect("the correct decode shape must be accepted");
    device_sync().expect("sync");
}

/// The C ABI's argument guards, which nothing else reaches.
///
/// Each returns before any launch and before any pointer is read, so one scratch buffer
/// stands in for all of them. These exist because a guard nobody exercises is a guard
/// nobody knows is inverted — and `v4_sparse_attn`'s `d` cap in particular is the only
/// thing between a `head_dim` past the per-thread accumulator and output dims that are
/// silently never written. The model runs 512 against a cap of 1024, so nothing else in
/// this suite comes near it.
#[test]
fn the_c_abi_argument_guards_reject_out_of_domain_shapes() {
    let mut b = dev_f32(&vec![0.0f32; 64]);
    let (p, pm) = (b.ptr().cast::<f32>(), b.ptr_mut().cast::<f32>());
    let guard = |r: anyhow::Result<()>, code: &str, what: &str| {
        let e = format!("{}", r.expect_err(what));
        assert!(e.contains(code), "{what}: expected guard {code}, got {e}");
    };
    // SAFETY: every call below is rejected by an argument guard before any launch, so no
    // pointer is dereferenced and the shapes never have to be real.
    unsafe {
        // head_dim past V4_ATTN_THREADS * V4_ATTN_ACC -- silently dropped dims otherwise.
        guard(
            launch_v4_sparse_attn(p, p, p, b.ptr().cast(), 1, 1, 1025, 8, 1.0, pm),
            "1002",
            "head_dim over the accumulator cap",
        );
        // ...and 1024 exactly is accepted, so the cap is a boundary and not a blanket no.
        guard(
            launch_v4_sparse_attn(p, p, p, b.ptr().cast(), 1, 1, 1024, 1 << 20, 1.0, pm),
            "1006",
            "a topk that overflows LDS",
        );
        // A `groups` that does not divide `n_out` would index a slice no input was sized
        // for. This is the guard the three-parameter form could not express at all.
        guard(
            launch_v4_gemv_fp8(p, b.ptr(), p, 1, 10, 128, 128, 3, pm),
            "1004",
            "groups not dividing n_out",
        );
        guard(launch_v4_gemv_fp8(p, b.ptr(), p, 1, 8, 128, 96, 1, pm), "1003", "non-power-of-two block");
        // `view_as_complex` cannot pair an odd count.
        guard(launch_v4_rope(pm, p, 1, 8, 3, 0, 1, false), "1005", "odd rope_head_dim");
        guard(launch_v4_rope(pm, p, 1, 8, 16, 0, 1, false), "1002", "rope span over the row");
        // The ONLY assertion of the ragged-span guard, deliberately. 2026-08-05: it was
        // also asserted inside `act_quant_matches_the_oracle_on_the_subnormal_ties_...`,
        // holding the pre-renumbering code 1002; the kernel and this test moved to 1004
        // together and that copy did not, so the run failed on a stale string AFTER the
        // numerics comparison had already passed. It cost two wrong diagnoses, because a
        // guard rejection reads like a numerics failure in a log -- the test name says
        // "matches the oracle" and the output says "argument guard rejected". One guard,
        // one assertion.
        guard(launch_v4_act_quant(pm, 1, 64, 60, 64), "1004", "a ragged quantization span");
    }
}

/// `Dims::from_config` against the artifact the port will actually run on.
///
/// It is the only path that reads S1a's `V4Config`, including `sliding_window` and
/// `rms_norm_eps`, which that config did not parse at all until `b5d4083`.
///
/// A MISSING ARTIFACT IS A FAILURE, not a skip. The first draft of this printed a SKIP
/// line and returned green, which is worse than useless: libtest captures stdout on a
/// passing test, so the run was indistinguishable from one that had checked the real
/// config. There is no CI here (CLAUDE.md), so a silently-skipped gate is a gate nobody
/// learns is dead.
#[test]
fn dims_accept_the_real_artifact_and_reject_a_ragged_kv_span() {
    const DIR: &str = "/var/db/rivoli/v4-f4-l0-2";
    assert!(
        std::path::Path::new(DIR).join("manifest.json").exists(),
        "no V4 artifact at {DIR}: S2b's only check against the SHIPPED config cannot run. \
         Produce it with `bin/convert_v4` (S1a) rather than letting this pass."
    );
    let cfg: EngineV4Config = rivoli::artifact::model::load_config(DIR).expect("V4 config");
    let d = Dims::from_config(&cfg).expect("the shipped config must be runnable");
    assert_eq!((d.head_dim, d.rope_head_dim, d.n_heads), (512, 64, 64));
    assert_eq!((d.head_dim - d.rope_head_dim) % 64, 0, "the partial act_quant needs whole blocks");
    // The two fields `V4Config` gained in `b5d4083`. What these pin is the WIRING —
    // that `from_config` puts `cfg.sliding_window` into `Dims.window` and
    // `cfg.rms_norm_eps` into `Dims.norm_eps`, rather than another field of the same
    // type. They do NOT catch a defaulting parser, which an earlier comment here claimed:
    // a `#[serde(default = "…")]` returning 128 passes this identically, and a bare
    // `#[serde(default)]` yields 0 and is caught upstream by the zero sweep. The guard
    // against defaults is `every_v4_field_is_required`, which covers both since
    // `b5d4083` put them in `V4_BASE`.
    assert_eq!(d.window, 128, "sliding_window is not wired through to Dims.window");
    assert!((d.norm_eps - 1e-6).abs() < 1e-12, "rms_norm_eps is not wired through to Dims.norm_eps");

    // The rejection half. A `rope_head_dim` that leaves a ragged non-RoPE span would
    // make `act_quant` quantize a short tail block against its own amax — values the
    // reference cannot produce, and silent, since every shipped shape divides evenly.
    let mut ragged = cfg.clone();
    ragged.qk_rope_head_dim = 66;
    let e = Dims::from_config(&ragged).expect_err("a ragged KV span must be refused");
    assert!(format!("{e}").contains("not a multiple of 64"), "wrong rejection: {e}");
    // Zero extents. `is_multiple_of` admits zero, so without `from_config`'s explicit
    // sweep each of these reached a launcher as an opaque guard code.
    //
    // ALL EIGHT the sweep covers, not a sample. The production side is one loop over a
    // literal list, so six cases would not exercise six code paths — they would exercise
    // one branch six times. What this can prove is MEMBERSHIP: that every extent the
    // kernels index with is in that list. A subset proves neither, and the first draft
    // shipped six of eight, omitting `n_heads` and `hidden`.
    //
    // The rejection is matched on the FIELD NAME, and what that buys is the field->label
    // MAPPING — the only thing in this file that would catch `q_lora_rank` being wired to
    // `cfg.o_lora_rank`, since both are 1024 in the shipped config and no value assertion
    // separates them.
    //
    // It does NOT pin the sweep's position, though an earlier draft of this comment
    // claimed it did. Traced against a sweep moved to the end of `from_config`: six of the
    // cases pass every intervening check and still reach it with their own correct
    // message, and the two that do not (`head_dim`, `o_groups`) are intercepted with
    // messages containing no "is zero" at all — so a strict and a lax assertion have
    // identical reordering sensitivity in all nine cases.
    /// One named extent, and how to zero it. Named so the array below is a table of
    /// FIELDS rather than a tuple soup, and typed so the count is checked: `[ZeroCase; 9]`
    /// stops being 9 the moment someone drops a case; it shipped 6 of 8, then 8 of 9.
    type ZeroCase = (&'static str, fn(&mut EngineV4Config));
    let cases: [ZeroCase; 9] = [
        // DERIVED, and the reason this list is 9 and not 8: the KV entry's non-RoPE span
        // is what `act_quant` sizes on, and no config field holds it.
        ("head_dim - qk_rope_head_dim", |c| c.qk_rope_head_dim = c.head_dim),
        ("sliding_window", |c| c.sliding_window = 0),
        ("n_heads", |c| c.n_heads = 0),
        ("o_groups", |c| c.o_groups = 0),
        ("hidden", |c| c.hidden = 0),
        ("head_dim", |c| c.head_dim = 0),
        ("qk_rope_head_dim", |c| c.qk_rope_head_dim = 0),
        ("q_lora_rank", |c| c.q_lora_rank = 0),
        ("o_lora_rank", |c| c.o_lora_rank = 0),
    ];
    for (name, mutate) in cases {
        let mut bad = cfg.clone();
        mutate(&mut bad);
        let e = Dims::from_config(&bad).expect_err("a zero extent must be refused");
        let want = format!("{name} is zero");
        assert!(format!("{e}").contains(&want), "expected `{want}`, got: {e}");
    }
}

/// `v4_act_quant` against the oracle, on data CHOSEN to reach e4m3's subnormal range and
/// sit exactly on its rounding ties.
///
/// The model fixture cannot cover this and no amount of it would. `act_quant`'s
/// power-of-two scale puts a block's largest element in [224, 448], so an element only
/// reaches e4m3's subnormals when it is ~2^15 below its block's peak — which drawn
/// activations essentially never are. That range is precisely where `f2e4m3_rne` and
/// rivoli's own `math.rs::f32_to_e4m3` disagree: the kernel rounds subnormal ties to
/// nearest-EVEN because V4 was trained against CUDA's `cvt.rn.satfinite.e4m3x2.f32`,
/// while rivoli's rule for GLM is half-away-from-zero.
///
/// So the block below pins the scale with a 448 and fills the rest with exact multiples
/// and exact HALF-multiples of the 2^-9 subnormal quantum, and the assertion before the
/// comparison proves that this data separates the two rules — without it, agreeing with
/// the oracle here would be evidence of nothing.
#[test]
fn act_quant_matches_the_oracle_on_the_subnormal_ties_that_pick_the_rounding_rule() {
    const BLOCK: usize = 64;
    const Q: f32 = 1.0 / 512.0; // e4m3's subnormal quantum, 2^-9
    let mut row = vec![0.0f32; BLOCK];
    row[0] = 448.0; // pins the block scale, and is itself the saturation edge
    for (i, v) in row[1..].iter_mut().enumerate() {
        // 0, 0.5, 1.0, ... 7.5 quanta — every representable subnormal AND every midpoint
        // between two of them, in both signs so a sign-dependent tie rule shows up.
        let m = (i % 16) as f32 * 0.5;
        *v = if i % 2 == 0 { m * Q } else { -m * Q };
    }

    let mut want = row.clone();
    act_quant_inplace(&mut want, BLOCK, true);

    // ANTI-VACUITY, in two parts.
    // 1. The data must actually land in the subnormal band, or it tests the normal path
    //    twice. The band is |x| < 2^-6 * s.
    let s = fast_round_scale(row.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-4), 1.0 / FP8_MAX);
    let sub = want.iter().filter(|v| **v != 0.0 && v.abs() < s * 0.015625).count();
    assert!(sub >= 8, "only {sub} outputs are subnormal — this block does not reach the branch");
    // 2. The data must SEPARATE the two rounding rules. `math.rs::f32_to_e4m3` is
    //    rivoli's half-away-from-zero encoder; if it produced the same block, then
    //    matching the oracle below would not be evidence that the kernel uses RNE.
    let half_away: Vec<f32> = row
        .iter()
        .map(|v| e4m3_to_f32(f32_to_e4m3((v / s).clamp(-FP8_MAX, FP8_MAX))) * s)
        .collect();
    assert_ne!(
        half_away, want,
        "half-away-from-zero and round-to-nearest-even agree on this block, so it cannot \
         tell which rule the kernel implements"
    );

    let mut buf = dev_f32(&row);
    // SAFETY: `buf` is one row of BLOCK f32 and outlives the sync below.
    unsafe { launch_v4_act_quant(buf.ptr_mut().cast(), 1, BLOCK, BLOCK, BLOCK) }
        .expect("v4_act_quant");
    device_sync().expect("sync");
    let got = read(&buf);
    // Bit-exact, not within a tolerance: `act_quant` is comparisons, a power-of-two
    // scale and a table lookup. There is no re-association in it to excuse a difference.
    assert!(
        got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()),
        "v4_act_quant disagrees with the oracle on the subnormal ties:\n  got  {:?}\n  want {:?}",
        &got[..16],
        &want[..16]
    );
}

/// **The comparator itself, proved able to go red.** Needs no device.
///
/// `score` carries three independent signals and `assert_within` asserts on all three.
/// Each is a claim that some class of wrongness is detectable, and a signal that cannot
/// fire is worse than no signal — it reads as coverage. So each is driven here with data
/// that must trip it, and with data that must not.
///
/// `unrounded` is the one that most needed this. It was added because `mono` rounds both
/// sides to bf16 before differencing, which makes `max_ulp` blind to a kernel that
/// stopped rounding its stores; a blindness fixed by a guard that could not fire would
/// have left the file claiming a coverage it did not have.
#[test]
fn the_comparator_fires_on_each_class_of_wrongness_and_stays_quiet_otherwise() {
    let bf = |x: f32| bf16_to_f32(f32_to_bf16(x));
    let clean: Vec<f32> = (0..64).map(|i| bf(i as f32 * 0.37 - 11.0)).collect();

    // Identical input: every signal silent. Without this the three below prove only that
    // the signals fire, not that they discriminate.
    let s = score(&clean, &clean);
    assert_eq!((s.max_ulp, s.differing, s.nans, s.unrounded), (0, 0, 0, 0), "{s:?}");

    // 1. ULP: one value moved by a single bf16 step.
    let mut off = clean.clone();
    off[7] = bf16_to_f32(f32_to_bf16(off[7]) + 1);
    let s = score(&off, &clean);
    assert_eq!((s.max_ulp, s.differing), (1, 1), "a one-step move must read as 1 ULP: {s:?}");

    // 2. `unrounded`: a value that is NOT bf16-representable. It rounds back to exactly
    //    the golden, so `max_ulp` stays 0 -- which is the blindness this signal exists
    //    for, and asserting it here is what makes that concrete rather than argued.
    let mut extra = clean.clone();
    extra[3] = f32::from_bits(clean[3].to_bits() | 0x0000_1234);
    let s = score(&extra, &clean);
    assert_eq!(s.unrounded, 1, "extra f32 mantissa must be seen: {s:?}");
    assert_eq!(s.max_ulp, 0, "the ULP metric is supposed to be blind here -- {s:?}");
    assert_eq!(s.differing, 1, "and the bit compare is supposed to see it");

    // 3. NaN, one side only. Counted separately, never folded into a distance.
    let mut nan = clean.clone();
    nan[11] = f32::NAN;
    let s = score(&nan, &clean);
    assert_eq!(s.nans, 1, "a one-sided NaN must be counted: {s:?}");

    // `mono` across zero and across the sign -- the ordering the ULP count rests on.
    // -0.0 and +0.0 are the same number one step apart in bits, and must score 0 ULP.
    assert_eq!(score(&[-0.0f32], &[0.0f32]).max_ulp, 0, "signed zero is not a ULP apart");
    // The smallest positive and smallest negative bf16 straddle zero, two steps apart
    // (+min, 0, -min); a naive bit subtraction would call them 2^15 apart.
    //
    // 0x0001 is the smallest SUBNORMAL. The first draft of this used 0x0080 — the
    // smallest NORMAL — and asserted 2, which is wrong by 128 codes in each direction
    // and went red. `mono` was right and the expectation was not, which is the more
    // useful way round for an assertion about a metric to fail.
    let (tiny_p, tiny_n) = (bf16_to_f32(0x0001), bf16_to_f32(0x8001));
    assert_eq!(score(&[tiny_p], &[tiny_n]).max_ulp, 2, "mono is not monotone across zero");
    // ...and the normal boundary really is 128 subnormal codes above zero, so the ULP
    // count is a count of representable values and not of exponent steps.
    assert_eq!(score(&[bf16_to_f32(0x0080)], &[0.0f32]).max_ulp, 128);
    // ...and it is monotone across the whole ladder, not just near zero.
    let (a, b) = (bf(-3.5), bf(-3.25));
    assert!(mono(a) < mono(b), "mono is not increasing on negatives");
    assert!(mono(bf(-1.0)) < mono(bf(1.0)), "mono does not order across the sign");
}

// ═══ the compressed layer — 41 of the model's 43 ════════════════════════════════════

/// Every golden this file can observe on one block-step, from the block's input to its
/// output, in execution order.
///
/// `attn_norm_out` is first and it is not decoration: it is what the GPU is FED, so a
/// defect that moves it has moved the comparison's own input and every later disagreement
/// is about mHC rather than about attention. `compressed` exists only on a step that
/// emitted, which is itself a fact worth comparing — see [`moved`].
const OBSERVED: [&str; 7] = [
    "attn_norm_out",
    "q",
    "kv_entry",
    "compressed",
    "attn_core_out",
    "attn_derot",
    "attn_out",
];

/// Which of [`OBSERVED`] differ between two oracle runs, with the worst [`Score`] over the
/// script, in [`OBSERVED`] order.
///
/// **Membership is BITWISE and the magnitude is evidence, not a threshold.** A name appears
/// when the two runs disagree on a value.
///
/// A tensor present on ONE side only **panics**; it is not reported as a move. There is no
/// distance to report, so a list entry would carry a fabricated `Score`, and a changed
/// emission schedule invalidates every other row rather than being one more row.
///
/// **Those two arms are unreachable today and are kept for totality, not for coverage.**
/// `.compressed` is the only conditionally-pushed name in [`OBSERVED`], and whether it exists
/// is a pure function of `(seqlen, start_pos, ratio)` — `Oracle::attention`'s compressor gate
/// and `Oracle::compressor`'s two `should` tests read no `self.defect`. Verified over all 50
/// breakages, 2026-08-05: every one either moves `compressed` or leaves it bit-identical, and
/// none changes whether it is emitted. Stated because an earlier draft of this doc sold the
/// arms as this instrument's strongest protection, and a dead guard advertised as protection
/// is worse than none.
///
/// The `Score` is printed and nothing here asserts on it, deliberately. `max_ulp` is a
/// bf16-CODE distance and it is blind at zero — an element driven near zero by cancellation
/// reports tens of thousands of codes for a negligible absolute move, which is why
/// `QkNormAfterRope` reads 29,131 here and is a rounding difference by the scale-relative
/// yardstick `tests/v4_oracle.rs::qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one`
/// applies (its move on `.q` bounded by what dropping bf16 rounding entirely costs).
/// `rel` — max absolute move over the golden's own peak — is printed beside it so the two
/// can be read together, and whether a defect is REJECTABLE is settled on the device by
/// [`each_compressed_defect_is_further_away_than_the_kernels_are`], not here.
fn moved(clean: &[Phase], broken: &[Phase]) -> Vec<(&'static str, Score)> {
    // Not "the defect changed the schedule", which no `Defect` can do — see the doc above.
    // This catches the two runs having been driven from different scripts.
    assert_eq!(clean.len(), broken.len(), "the two runs walked different scripts");
    OBSERVED
        .into_iter()
        .filter_map(|name| {
            let mut worst: Option<Score> = None;
            for (a, b) in clean.iter().zip(broken) {
                // Two arms, worded separately: one message for both directions sends the
                // reader hunting for a DROPPED emission when the defect ADDED one.
                let s = match (captured(a, name), captured(b, name)) {
                    (Some(x), Some(y)) => score(y, x),
                    (None, None) => continue,
                    (Some(_), None) => panic!(
                        "{name}: the clean run emits it at {} and the defect run does not — \
                         the emission schedule changed, so no row below is comparable",
                        a.tag
                    ),
                    (None, Some(_)) => panic!(
                        "{name}: the defect run emits it at {} and the clean run does not — \
                         the emission schedule changed, so no row below is comparable",
                        b.tag
                    ),
                };
                if s.differing > 0 && worst.is_none_or(|w| s.differing > w.differing) {
                    worst = Some(s);
                }
            }
            worst.map(|s| (name, s))
        })
        .collect()
}

/// Which of [`OBSERVED`] a defect must move on [`COMP_LAYER`], and by construction which it
/// must leave BIT-IDENTICAL — the complement is asserted just as hard as the list.
///
/// **Exhaustive and wildcard-free.** A `Defect` added by a later stage is a compile error
/// here, which is the moment someone must decide whether a compressed layer can see it; a
/// `_ =>` arm would silently classify it as "moves nothing" and the empty list is this
/// file's strongest claim, not its weakest.
///
/// **Every list must be written in [`OBSERVED`] order.** `moved` iterates `OBSERVED`, and
/// the caller compares as a SET (symmetric difference) rather than positionally, so a
/// mis-ordered list still passes — but it will read as disagreeing with every neighbour and
/// the failure messages elsewhere print in this order.
///
/// The empty entries are the point of the whole table. Four groups are asserted INERT on a
/// compressed layer, each for a structural reason rather than because a run happened to come
/// back equal, and each is a separate arm below carrying its own argument:
///
/// 1. **Wrong layer class.** `CompressorNoOverlap` ([`COMP_LAYER`] is `NonOverlap`, so there
///    is no overlap term to disable — the pair `tests/v4_compress_kernel.rs` asserts at the
///    real weights) and `RopeYarnEverywhere` (keys off `!compressed` in `Oracle::freqs`, so
///    it is the ratio-0 layers' defect). Its sibling `RopeBaseThetaEverywhere` DOES reach
///    this layer, and having both here is what stops "a rope defect" reading as one thing.
/// 2. **No indexer** at ratio 8 — all five `Indexer*`.
/// 3. **`KvActQuantBlock128` alone**, because a ue8m0 scale is a power of two and e4m3 is
///    exactly scale-invariant under those.
/// 4. **Downstream of `attention`** — the twenty router / expert / Sinkhorn / head variants.
///    `run_layer` runs the MoE, but AFTER `attention`, and `h` is drawn fresh per step.
///
/// **`QkNormAfterRope` is NOT one of them**, though this file's header and `src/attn.rs`
/// both called it invisible until 2026-08-05. It is classified with the q path and moves
/// four goldens; see that arm. `HcPreNoRsqrt` and `NoBf16Rounding` move all seven — they are
/// upstream of `attn_norm_out`, which is what the GPU is fed.
fn expect_moves(d: Defect) -> &'static [&'static str] {
    // The three most common shapes, named once so a typo in one arm cannot masquerade as a
    // considered classification.
    const AFTER_Q: &[&str] = &["q", "attn_core_out", "attn_derot", "attn_out"];
    const AFTER_KV: &[&str] = &["kv_entry", "compressed", "attn_core_out", "attn_derot", "attn_out"];
    const AFTER_CORE: &[&str] = &["attn_core_out", "attn_derot", "attn_out"];
    const EVERY_ROPE: &[&str] =
        &["q", "kv_entry", "compressed", "attn_core_out", "attn_derot", "attn_out"];
    match d {
        Defect::None => &[],
        // The q path: `qk_norm` is q-only, so `kv_entry` and `compressed` must not move.
        //
        // **`QkNormAfterRope` is here, and that CORRECTED two comments** — both now carry a
        // dated note at their own site (this file's header, and `src/attn.rs`'s launch
        // sequence, which said "the oracle cannot see this order"). Measured 2026-08-05 it
        // moves all four of these goldens: `q` on 1287/24576 elements at rel 7.4e-3,
        // `attn_out` on 2098/3072 at rel 9.1e-3. The mathematical argument behind the old
        // claim is sound — RoPE rotates adjacent pairs so it preserves `q.square().mean(-1)`,
        // and a scalar commutes with a rotation — but `Oracle::qk_norm` computes that
        // statistic in BF16 (`forward.rs:768`, faithfully: it is bf16 in the reference too),
        // so `rs` is quantized to ~0.4% steps and the two orders land on different steps.
        // A ROUNDING difference, which
        // `tests/v4_oracle.rs::qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one`
        // bounds against dropping bf16 rounding entirely — not an INVISIBILITY.
        // [`compressed_sweep`] excludes it for that reason (its distance would measure the
        // rounding floor), not because nothing moves.
        Defect::SkipQkNorm | Defect::QkNormUsesQNormWeight | Defect::QkNormAfterRope => AFTER_Q,
        // RoPE is shared by the q path, the KV entry, the compressor's finish AND the output
        // de-rotation, so a pairing defect moves everything downstream of the block's input.
        Defect::RopeAllDims | Defect::RopeFirstDims | Defect::RopeHalfSplit => EVERY_ROPE,
        // The layer's TABLE: `Oracle::freqs` hands a compressed layer the YaRN one, and both
        // of these swap it. `RopeNoYarn` is S3 requirement 4 and `v4_compress_kernel.rs`
        // records that it cannot see it at `ratio4/decode`; here it moves six goldens.
        Defect::RopeNoYarn | Defect::RopeBaseThetaEverywhere => EVERY_ROPE,
        // The KV quantizer, which the compressor's finish also calls (`forward.rs:1253`) —
        // which is why `compressed` is in this list and not only `kv_entry`.
        Defect::SkipKvActQuant
        | Defect::KvActQuantWholeTensor
        | Defect::KvActQuantNoRoundScale => AFTER_KV,
        Defect::SkipAttnSink | Defect::AttnSinkNotMaxShifted => AFTER_CORE,
        // Prefill-only, and it shows up at DECODE: the prefill attends `kv ++ compressed`
        // and never reads the ring it just seeded, so its own output is untouched and the
        // later steps' are not. `moved` is over the whole script, which is what lets one
        // entry state that.
        Defect::PrefillRingWritesFirstWindow => AFTER_CORE,
        // `attn_core_out` is captured BEFORE the de-rotation, so it must stay identical —
        // that is the silent half the oracle keeps a pre-image for.
        Defect::SkipOutputDerotation | Defect::OutputDerotationForward => {
            &["attn_derot", "attn_out"]
        }
        Defect::WoGroupsSplitHeadDim | Defect::WoGroupsInterleaved => &["attn_out"],
        // The compressor proper. `q` and `kv_entry` MUST be bit-identical: nothing in the
        // compressor is upstream of either, and a harness that had accidentally routed the
        // compressor's output into the KV entry would fail here and nowhere else.
        Defect::CompressorNoApe | Defect::CompressorRopeAtBlockEnd => {
            &["compressed", "attn_core_out", "attn_derot", "attn_out"]
        }
        // Upstream of `attn_norm_out`, so it moves the comparison's own input.
        Defect::HcPreNoRsqrt => &OBSERVED,
        // `round_bf16` runs in `hc_pre` too, so this one also reaches the input.
        Defect::NoBf16Rounding => &OBSERVED,
        // ---- inert, in four groups by WHY, which is what the arms below are for ----
        //
        // jscpd:ignore-start
        //
        // THE ARGUMENT FOR THE EXEMPTION, in place as `build.rs` requires. This arm and
        // `tests/v4_compress_kernel.rs::in_compressor_scope` are two EXHAUSTIVE matches over
        // one 51-variant enum, so they necessarily share its variant list, and jscpd sees
        // that as a clone. Being exhaustive is the point in both: a `Defect` added later must
        // be classified in each, because the two answer different questions — "can a
        // NonOverlap layer's attention goldens see it" here, "does the attention compressor
        // touch it" there — and the answers differ for `CompressorNoOverlap`,
        // `QkNormAfterRope` and every `Head*`. Factoring them into one classifier would
        // merge two judgements that must stay separable.
        //
        // The first attempt at this was to REORDER the variants so the token runs did not
        // match, with a comment saying so. Review was right to reject it: nothing semantic
        // depends on the order, so the constraint is invisible, and the next stage to add
        // defects to both files in matching positions — which is exactly what happened on
        // 2026-08-05 — re-breaks the build for a reason documented sixty lines away. An
        // `ignore` marker is greppable, stable, and this repo's own documented mechanism.
        //
        // The better fix, if a third consumer ever appears: `Defect::stage(self) -> Stage` on
        // the enum itself in `src/v4oracle/forward.rs`, with both files matching exhaustively
        // over the ~12-variant `Stage` instead. Not done here because it edits the oracle and
        // a second suite mid-flight, for two consumers.
        //
        // 1. WRONG LAYER CLASS. Structurally unreachable at ratio 8, and each is the
        //    positive half of some other cell's coverage: `CompressorNoOverlap` is live at
        //    ratio 4, `RopeYarnEverywhere` at ratio 0.
        Defect::RopeYarnEverywhere | Defect::CompressorNoOverlap => &[],
        // 2. NO INDEXER at ratio 8 — `Indexer` exists only where `compress_ratio == 4`
        //    (model.py:474). All five, so a sixth added later has to be classified.
        Defect::IndexerNoRelu
        | Defect::IndexerNoFp4Quant
        | Defect::IndexerNoHadamard
        | Defect::IndexerNoWeights
        | Defect::IndexerBf16RunningSum => &[],
        // 3. BIT-IDENTICAL **on this fixture**, and that qualifier is the whole comment: a
        //    ue8m0 scale is a power of two and e4m3 is scale-invariant under those, so
        //    re-blocking moves nothing until an in-block dynamic range of ~2^13, which the
        //    toy does not reach. It is NOT a theorem — `src/attn.rs`'s `KV_QUANT_BLOCK` doc
        //    records the same derivation as CORRECTED, and `tests/v4_compress_kernel.rs`
        //    finds the defect live on one cell of four at the real weights, one e4m3 step,
        //    6 of 32768 elements. An empty list here is an observation this run reproduces.
        //
        //    `QkNormAfterRope` is deliberately NOT here — it moves four goldens; the
        //    measurement and the argument are at the q-path arm above.
        Defect::KvActQuantBlock128 => &[],
        // 4. DOWNSTREAM OF `attention`. `run_layer` runs the MoE after it and `h` is drawn
        //    fresh per step, so nothing here can reach an attention golden. The Sinkhorn
        //    pair is in this group and not group 3: `comb` is consumed by `hc_post`, and
        //    `pre` — the only part `attn_norm_out` depends on — is computed before the
        //    iterations begin.
        //
        //    In `Defect::ALL`'s declaration order, so a new variant has an obvious place and
        //    a diff against the enum reads straight down. That was NOT possible before the
        //    `ignore` marker above.
        Defect::SwigluUnclamped
        | Defect::SwigluClampGateBothSides
        | Defect::RouterSoftmax
        | Defect::RouterNoSoftplusThreshold
        | Defect::RouterBiasedWeights
        | Defect::RouterNoRenorm
        | Defect::RouterNoScale
        | Defect::HashRoutingIgnored
        | Defect::RouteWeightAfterW2
        | Defect::SharedExpertWeighted
        | Defect::Fp4NibbleSwap
        | Defect::SinkhornOneFewerIter
        | Defect::SinkhornCombTransposed
        | Defect::HcPostNoComb
        | Defect::HeadHcNoRsqrt
        | Defect::HeadHcRsqrtPerCopy
        | Defect::HeadNormSkipped
        | Defect::HeadNormNotBf16
        | Defect::HeadNormOverAllTokens
        | Defect::HeadLogitsFromFirstRow => &[],
        // jscpd:ignore-end
    }
}

/// **The oracle's own discrimination on a compressed layer, measured bidirectionally and
/// with no device.**
///
/// Every breakage must move exactly the goldens [`expect_moves`] names and leave every
/// other one BIT-IDENTICAL. The second half is what makes this more than a reachability
/// sweep: `Defect::CompressorNoApe` moving `attn_out` says only that something changed,
/// while `CompressorNoApe` moving `attn_out` *and* leaving `q` and `kv_entry` bit-identical
/// says the compressor is the only thing that moved — which is the claim every compressed
/// column assertion below rests on.
///
/// Runs on the CPU. It gates the INSTRUMENT, not the kernels, and a machine with no free
/// GPU can still tell whether the table has gone stale.
#[test]
fn each_defect_moves_exactly_the_compressed_layer_goldens_it_should() {
    let (cfg, model) = fixture();
    let script = comp_script(&cfg);
    let clean = drive_script(&cfg, &model, Defect::None, COMP_LAYER, &script);
    // ANTI-VACUITY. Every name in `OBSERVED` must actually be emitted by this script,
    // otherwise "did not move" is "was never there" and the inert half of the table would
    // be satisfied by an oracle that recorded nothing at all.
    for name in OBSERVED {
        assert!(
            clean.iter().any(|p| captured(p, name).is_some()),
            "{name} is emitted by no step of this script, so nothing about it is asserted"
        );
    }
    let mut bad = Vec::new();
    for def in Defect::breakages() {
        let got = moved(&clean, &drive_script(&cfg, &model, def, COMP_LAYER, &script));
        // Both magnitudes on every moved stage, because they disagree and the disagreement
        // is the point (see `moved`): `ulp` is a bf16-CODE distance and blind at zero, `rel`
        // is scale-relative. A `differing`/`total` count separates a systematic move from a
        // handful of boundary flips.
        let evidence: Vec<String> = got
            .iter()
            .map(|(n, s)| {
                format!("{n}[ulp={} rel={:.2e} {}/{}]", s.max_ulp, s.rel, s.differing, s.total)
            })
            .collect();
        println!("  {def:<32?} {}", evidence.join(" "));
        // Compared as a SET, and reported as a symmetric difference. Positional equality
        // would fail on a correctly-classified defect whose list was written in a different
        // order, with a message showing two lists that look alike to anyone scanning for a
        // missing name — and the two halves mean opposite things, so they are named apart.
        let want = expect_moves(def);
        let names: Vec<&str> = got.iter().map(|&(n, _)| n).collect();
        let extra: Vec<&str> = names.iter().copied().filter(|n| !want.contains(n)).collect();
        let missing: Vec<&str> = want.iter().copied().filter(|n| !names.contains(n)).collect();
        if !extra.is_empty() || !missing.is_empty() {
            bad.push(format!(
                "{def:?}: reached {extra:?} which it is classified as NOT touching; \
                 left {missing:?} bit-identical which it is classified as moving"
            ));
        }
    }
    assert!(
        bad.is_empty(),
        "the compressed-layer defect classification is wrong. A defect that moves MORE than \
         its entry has reached a stage it should not; one that moves LESS is not reachable \
         on this layer and is claiming coverage this cell does not have:\n  {}",
        bad.join("\n  ")
    );
}

/// **The compressed layer, end to end, at both destinations and in both phases.**
///
/// This is the path 41 of the model's 43 layers take and, until this cell, the path nothing
/// executed: `LAYER` is ratio-0, so the `io.cache` tail layout, the prefill persist copy,
/// the decode slot and the compressed columns arriving at `sparse_attn` were reached by no
/// test at all.
///
/// The four `attention` goldens are compared exactly as the ratio-0 cell compares them, and
/// the compressed rows are compared as a fifth — read back out of the buffer `sparse_attn`
/// indexed, not out of the compressor's own `out`, so a placement that landed them anywhere
/// else fails here rather than passing on a clean-looking scratch.
///
/// The anti-vacuity assertions are what make a green run mean something, and each names a
/// way this could otherwise pass while testing the ratio-0 path twice:
///
/// * `n_comp > 0` on every step — there ARE compressed columns.
/// * `max_idx >= offset` — they are not all `-1`. A `compress_topk` that produced a
///   correctly-shaped all-masked row would satisfy the first and read nothing.
/// * the prefill emits, and the FIRST decode step emits nothing while still selecting a
///   compressed block — so the block it attends can only have come from the prefill's
///   persist copy into `io.cache`, and dropping that copy cannot pass.
/// * some later decode step emits — so the decode slot is exercised too.
#[test]
fn attention_matches_the_oracle_on_a_compressed_layer_in_both_phases() {
    let Harness { d, clean, mut gpu, .. } = Harness::compressed();
    let blocks = |p: &Phase| p.cap.counters.compressed_blocks;
    assert!(blocks(&clean[0]) > 0, "the prefill emits no block; the persist copy gates nothing");
    assert_eq!(
        blocks(&clean[1]),
        0,
        "the first decode step must NOT emit — a step that both emits and selects could read \
         its own block and would pass with the prefill's persist copy deleted"
    );
    assert!(
        clean[1..].iter().any(|p| blocks(p) > 0),
        "no decode step emits, so the decode slot `window + start_pos/ratio` is never written"
    );

    for p in &clean {
        let got = gpu.step(&d, p);
        // The compressed region is where the window region is not, so the offset a
        // selection index must reach is the width of the window half — `seqlen` at prefill
        // (the prompt IS the window region there) and `window` at decode.
        let offset = if p.start_pos == 0 { p.m } else { d.window };
        assert!(
            got.n_comp > 0 && got.max_idx >= offset as i32,
            "{}: {} compressed columns past a window half of {offset}, largest selection \
             index {} — nothing reaches the compressed region, so this step scores the \
             ratio-0 path",
            p.tag,
            got.n_comp,
            got.max_idx
        );
        assert_stages(p, &got);
        // The fifth golden, and the only one that says WHERE the blocks landed.
        match captured(p, "compressed") {
            Some(want) => {
                assert_eq!(
                    got.compressed.len(),
                    want.len(),
                    "{}: the engine placed {} compressed values where the reference emits {}",
                    p.tag,
                    got.compressed.len(),
                    want.len()
                );
                assert_within("compressed (pool..rope..act_quant)", &got.compressed, want);
            }
            None => assert!(
                got.compressed.is_empty(),
                "{}: the reference emits no block here and the engine placed {} values",
                p.tag,
                got.compressed.len() / d.head_dim
            ),
        }
    }
}

/// **The compressed region is read exactly where the selection names it, and nowhere else.**
///
/// Agreement with the oracle says the right values are in the right place; it does not say
/// `sparse_attn` READ them, because a layout in which the compressed rows were unreachable
/// would agree just as well if the oracle's blocks happened to contribute little. This
/// corrupts one cache row at a time and requires the output to move — or, for a row the
/// selection does not name, to stay BIT-IDENTICAL.
///
/// Three probes, and the pair is the instrument:
///
/// | row | must |
/// |---|---|
/// | compressed block 0 | MOVE — it is selected, so a layout that never reaches it fails here |
/// | compressed block `n_comp` | stay IDENTICAL — the first block past the selection, so an engine reading one row too far fails here |
/// | ring slot 0 | MOVE — the window half is live, and the compressed offset is not pointing into it |
///
/// Re-attends with [`Gpu::attend`], never [`Gpu::step`]: `compress` read-modify-writes the
/// pooling state before its emit decision, so a second `step` at the same position would
/// deposit the row twice and slide the window twice, and the probe would be measuring that
/// instead.
#[test]
fn the_compressed_region_is_read_exactly_where_the_selection_names_it() {
    let Harness { cfg, d, clean, mut gpu, .. } = Harness::compressed();
    for p in &clean {
        gpu.step(&d, p);
    }
    let p = clean.last().expect("the script has steps");
    let base = gpu.attend(&d, p);
    let ratio = cfg.compress_ratio(COMP_LAYER);
    let capacity = cfg.max_seq_len / ratio;
    assert!(
        base.n_comp < capacity,
        "the selection names every one of the {capacity} compressed blocks, so there is no \
         unselected row to prove the read is BOUNDED — only that it happens"
    );
    // FIRST, not last. The probes are evidence only if `attend` is otherwise deterministic,
    // and asserting it afterwards means a non-idempotent `attend` surfaces as the
    // past-the-selection probe reporting "MOVED" — pointing the reader at an out-of-bounds
    // read that is not there. Establish the baseline is stable, then perturb it.
    let again = gpu.attend(&d, p);
    assert!(
        again.v[3].iter().zip(&base.v[3]).all(|(a, b)| a.to_bits() == b.to_bits()),
        "`attend` is not idempotent, so nothing below is attributable to the poison"
    );

    // A value no KV entry can hold: the fixture's activations are O(1) and every row here
    // is bf16, so 64 is finite (no NaN in the softmax, which would fail as a NaN count
    // rather than as a distance) and dominant.
    const POISON: f32 = 64.0;
    // A third probe — a selected RING slot must move — was cut on review: the window half is
    // already proved live by the ratio-0 cell and by `floor == 0`, so it repeated coverage.
    // These two do not: probe 1 survives `floor == 0` (the harness could be placing at the
    // same wrong offset the kernel reads, which is the write-the-test-to-match-the-bug case),
    // and probe 2 is what turns probe 1 from "reads at least here" into "reads exactly here".
    for (row, what, must_move) in [
        (d.window, "compressed block 0 (selected)", true),
        (d.window + base.n_comp, "the first compressed block PAST the selection", false),
    ] {
        let was = gpu.poke(&d, row, POISON);
        let got = gpu.attend(&d, p);
        gpu.unpoke(&d, row, &was);
        let s = score(&got.v[3], &base.v[3]);
        println!("  poisoned cache row {row:>3} ({what}): {s:?}");
        assert_eq!(
            s.differing > 0,
            must_move,
            "poisoning {what} at cache row {row} {} the block output. {s:?}",
            if must_move { "did NOT move" } else { "MOVED" }
        );
    }
    // ...and the buffers are back where they started, which is what `unpoke` owes and the
    // only check that it delivered.
    let restored = gpu.attend(&d, p);
    assert!(
        restored.v[3].iter().zip(&base.v[3]).all(|(a, b)| a.to_bits() == b.to_bits()),
        "`unpoke` did not restore the cache: a later test sharing this harness would run on \
         a poisoned buffer"
    );
}

/// The two constructions of the un-YaRN'd rotary table agree, bit for bit.
///
/// `Gpu::new` builds the ratio-0 table with `v4_rope_table_ratio0` and the compressed one
/// with `freqs_cis(rope_for_layer(..))`, and a comment there used to call that a
/// "cross-check" while nothing compared them. This is the comparison, and it is not
/// bookkeeping: `src/attn.rs` records that `v4_rope_table_ratio0`'s only cross-check was an
/// out-of-tree numpy transliteration plus the end-to-end `.q` golden. `rope_for_layer` with
/// `LayerKind::Plain` passes `original_seq_len = 0`, which disables the YaRN branch, so the
/// two must produce identical tables — and if they ever do not, one of the two cells in this
/// file is being scored against a table the other would call wrong.
///
/// Needs no device.
#[test]
fn the_two_rope_table_constructions_agree_on_the_un_yarned_table() {
    let cfg = V4Config::toy();
    let direct = v4_rope_table_ratio0(cfg.rope_head_dim, cfg.max_seq_len, cfg.rope_theta);
    let compressed = compressed_rope(&cfg);
    let via_selector = flat_freqs(&freqs_cis(
        rope_for_layer(compressed, cfg.rope_theta, LayerKind::Plain),
        cfg.max_seq_len,
    ));
    assert_eq!(direct.len(), via_selector.len(), "the two tables are not even the same shape");
    let differing =
        direct.iter().zip(&via_selector).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
    assert_eq!(differing, 0, "{differing} of {} entries differ", direct.len());
    // ANTI-VACUITY: `rope_for_layer` must actually be making a CHOICE here. Handing it a
    // compressed `LayerKind` has to produce a different table, or the equality above would
    // hold for a selector that ignored its argument entirely.
    let yarn = flat_freqs(&freqs_cis(
        rope_for_layer(compressed, cfg.rope_theta, LayerKind::from_ratio(8)),
        cfg.max_seq_len,
    ));
    assert_ne!(direct, yarn, "`rope_for_layer` returns the same table for both layer classes");
}

/// The compressed cell's separation sweep: every breakage [`expect_moves`] says reaches one
/// of the four tensors [`stages`] scores, minus three documented exclusions.
///
/// DERIVED from the classification rather than spelled as a second list. A list would
/// silently omit any variant added later; this way a new breakage classified as reaching an
/// attention golden lands in the sweep automatically, and the count below is what forces
/// someone to look when that happens.
///
/// The exclusions, each because the sweep would measure something other than the kernels'
/// resolution:
///
/// * `QkNormAfterRope` — a bf16-rounding difference (see [`expect_moves`]), so its distance
///   is the rounding floor rather than a defect's reach. The ratio-0 sweep excludes it too.
/// * `HcPreNoRsqrt` and `NoBf16Rounding` — both move `attn_norm_out`, which is what the GPU
///   is FED. Their distance would measure a different input, not a different kernel.
///   `NoBf16Rounding` is covered instead by `Score::unrounded`, a property of the GPU output.
fn compressed_sweep() -> Vec<Defect> {
    Defect::breakages()
        .filter(|d| expect_moves(*d).iter().any(|n| STAGES.iter().any(|(s, _)| s == n)))
        .filter(|d| {
            !matches!(
                d,
                Defect::QkNormAfterRope | Defect::HcPreNoRsqrt | Defect::NoBf16Rounding
            )
        })
        .collect()
}

/// **The compressed cell has the resolution it claims** — the same instrument
/// `each_in_scope_defect_is_further_away_than_the_kernels_are` applies to the ratio-0 layer,
/// on the layer class 41 of 43 layers actually have.
///
/// The floor is what this cell is really about. If the compressed blocks were placed at the
/// wrong row, or the persist copy were missing, or the decode slot were "the next free one",
/// the GPU would be far from the CLEAN oracle and this fails before any defect is tried.
#[test]
fn each_compressed_defect_is_further_away_than_the_kernels_are() {
    let Harness { cfg, model, d, clean, mut gpu, layer, script } = Harness::compressed();
    let mine: Vec<StepOut> = clean.iter().map(|p| gpu.step(&d, p)).collect();
    let floor = reach(&clean, &mine);
    println!("compressed-layer kernel-vs-oracle floor: {floor} bf16 ULP over {} steps", clean.len());

    // No `sweep.len() == N` anti-drift record, deliberately: a new `Defect` variant is
    // already a COMPILE error at `expect_moves`, which is the same moment and the same
    // reader, and a mis-CLASSIFIED variant is caught by
    // `each_defect_moves_exactly_the_compressed_layer_goldens_it_should`, which measures
    // rather than predicts. A hand-maintained count on top of both buys nothing and is a
    // second thing to update.
    let mut worst: Option<(Defect, i32)> = None;
    for def in compressed_sweep() {
        let r = reach(&drive_script(&cfg, &model, def, layer, &script), &mine);
        println!("  {def:<32?} reach={r} ULP");
        if worst.is_none_or(|(_, w)| r < w) {
            worst = Some((def, r));
        }
        assert!(
            r > floor && r >= 8,
            "{def:?} moves the compressed layer's goldens by only {r} bf16 ULP against a \
             kernel-vs-oracle floor of {floor}: this cell could not tell a kernel carrying \
             that defect from a correct one"
        );
    }
    let (dw, rw) = worst.expect("the sweep is empty");
    println!("tightest margin: {dw:?} at {rw} ULP against a floor of {floor}");
    // Asserted LAST so the per-defect table above still prints when it fires, for the reason
    // the ratio-0 sweep gives. Pinned at what the kernels DO, not at what `ULP_BUDGET`
    // allows: a silent drift from 0 to 1 is the first observable sign of a second error
    // source, and on this cell it would most likely be the compressor's e4m3 boundary.
    assert_eq!(floor, 0, "the compressed layer is no longer bit-exact against the oracle");
}
