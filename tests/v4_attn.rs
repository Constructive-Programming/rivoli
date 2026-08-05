//! **The V4 attention kernels, scored against S1b's oracle.** S2b of
//! `docs/investigations/v4-flash-port.md`.
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
//! Two defects in S2b's scope are invisible to these goldens and are NOT claimed here:
//! the QK-norm's position relative to the RoPE, and the KV `act_quant`'s block size.
//! Both are argued at their call sites in `src/attn.rs` from `model.py`, and
//! `tests/v4_oracle.rs` measures why the oracle cannot see them. They are excluded from
//! the defect list below rather than silently passing inside it.
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
use rivoli::attn::{
    v4::{Dims, Fp8W, Io, Scratch, Step, Weights, attention},
    v4_rope_table_ratio0, v4_window_topk,
};
use rivoli::backend::hip::{
    device_sync, launch_v4_act_quant, launch_v4_gemv_fp8, launch_v4_rope,
    launch_v4_sparse_attn,
};
use rivoli::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3};
use rivoli::v4oracle::numerics::{FP8_MAX, act_quant_inplace, fast_round_scale};
use rivoli::memory::device::DeviceBuf;
use rivoli::v4oracle::forward::{Capture, Defect, Oracle, Step as OStep};
use rivoli::v4oracle::toy::{self, ToyModel};
use rivoli::v4oracle::weights::{NamedRng, V4Config, WMat};

mod common;
use common::{f32b, f32v};

/// The ratio-0 layer S2b is scored on. Toy layer 0 has `compress_ratio == 0`: no
/// compressor, no indexer, no YaRN, base `rope_theta` — exactly the shape S2b owns, and
/// exactly the shape `tests/v4_oracle.rs` warns is the least representative layer in the
/// model. That is fine here and NOT fine for the port: S2c owns the rest.
const LAYER: usize = 0;

/// Prompt long enough to outrun the toy's 8-slot window, so the ring wraps and
/// `PrefillRingWritesFirstWindow` is reachable at all. A prompt that fits the window
/// makes that defect structurally unable to fire.
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

/// The toy model with layer 0's `wo_a` replaced by an fp8 weight.
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
    let mut r = NamedRng::new("v4-s2b-wo_a-fp8");
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
    m.layers[LAYER].wo_a = WMat::Fp8 { rows, cols, w, s };
    (cfg, m)
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

/// Run the oracle over layer `LAYER` for a prefill and `DECODES` decode steps, capturing
/// every golden. `h` is drawn fresh per step: nothing on the attention path depends on
/// where the residual stream came from, and a deterministic draw keeps the fixture from
/// depending on the MoE half of the block agreeing first.
fn drive(cfg: &V4Config, m: &ToyModel, defect: Defect) -> Vec<Phase> {
    let o = Oracle::new(cfg.clone(), defect);
    let mut st = o.fresh_state(LAYER);
    let mut out = Vec::new();
    for (k, (s, start_pos)) in
        std::iter::once((PROMPT, 0)).chain((0..DECODES).map(|i| (1, PROMPT + i))).enumerate()
    {
        let tag = if k == 0 { "pre".to_string() } else { format!("dec{}", k - 1) };
        let mut h = draw(&format!("h-{tag}"), s * cfg.hc_mult * cfg.dim);
        let ids: Vec<u32> =
            (0..s).map(|i| ((start_pos + i) * 7 % cfg.vocab_size) as u32).collect();
        let mut cap = Capture::default();
        let step = OStep {
            lw: &m.layers[LAYER],
            layer: LAYER,
            s,
            start_pos,
            input_ids: &ids,
            phase: &tag,
        };
        o.run_layer(&step, &mut st, &mut h, &mut cap);
        out.push(Phase { tag: format!("L{LAYER}.{tag}"), m: s, start_pos, cap });
    }
    out
}

fn draw(name: &str, n: usize) -> Vec<f32> {
    let mut r = NamedRng::new(name);
    (0..n).map(|_| r.unit() * 0.5).collect()
}

fn golden<'a>(p: &'a Phase, suffix: &str) -> &'a [f32] {
    p.cap.float(&format!("{}.{suffix}", p.tag)).unwrap_or_else(|| {
        panic!("golden {}.{suffix} is missing -- the oracle no longer emits it", p.tag)
    })
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
    /// breakage and `v4_rbf16` appears in every kernel S2b adds, so it is in scope. The
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
}

impl Gpu {
    fn new(cfg: &V4Config, model: &ToyModel, d: &Dims, max_m: usize) -> Self {
        let lw = &model.layers[LAYER];
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
        Self {
            _w: w,
            _norms: norms,
            weights,
            xq: z(max_m * d.dim),
            qr: z(max_m * d.q_lora_rank),
            qrq: z(max_m * d.q_lora_rank),
            q: z(max_m * nhd),
            kv: z(max_m * d.head_dim),
            o: z(max_m * nhd),
            y: z(max_m * d.o_groups * d.o_lora_rank),
            ring: z(d.window * d.head_dim),
            out: z(max_m * d.dim),
            // The ratio-0 table: base `rope_theta`, NO YaRN. A compressed layer would
            // need the other one and `v4_rope_table_ratio0` cannot build it.
            freqs: dev_f32(&v4_rope_table_ratio0(
                d.rope_head_dim,
                cfg.max_seq_len,
                cfg.rope_theta,
            )),
        }
    }

    fn scratch(&mut self) -> Scratch {
        Scratch {
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
            ring: self.ring.ptr_mut().cast(),
            out: self.out.ptr_mut().cast(),
        }
    }

    /// One `attention` call for `p`, returning the four goldens it leaves behind.
    fn step(&mut self, d: &Dims, p: &Phase) -> [Vec<f32>; 4] {
        let x = dev_f32(golden(p, "attn_norm_out"));
        let mut idx = Vec::new();
        let shape = v4_window_topk(d.window, p.m, p.start_pos, &mut idx);
        let idxb = dev_i32(&idx);
        let io = self.io(&x, &idxb, shape);
        let step = if p.start_pos == 0 {
            Step::Prefill { seqlen: p.m }
        } else {
            Step::Decode { pos: p.start_pos }
        };
        let s = self.scratch();
        // SAFETY: every buffer above outlives the `device_sync` on the next line.
        unsafe { attention(d, &self.weights, &s, &io, step) }.expect("v4 attention");
        device_sync().expect("sync");
        let n = |b: &DeviceBuf, len: usize| read(b)[..len].to_vec();
        let nhd = d.n_heads * d.head_dim;
        [
            n(&self.q, p.m * nhd),
            n(&self.kv, p.m * d.head_dim),
            n(&self.o, p.m * nhd),
            n(&self.out, p.m * d.dim),
        ]
    }
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
}

impl Harness {
    fn new() -> Self {
        let (cfg, model) = fixture();
        let d = dims(&cfg);
        let clean = drive(&cfg, &model, Defect::None);
        let gpu = Gpu::new(&cfg, &model, &d, PROMPT);
        Self { cfg, model, d, clean, gpu }
    }
}

// ═══ tests ══════════════════════════════════════════════════════════════════════════

#[test]
fn attention_matches_the_oracle_at_every_stage_of_a_ratio_zero_layer() {
    let Harness { d, clean, mut gpu, .. } = Harness::new();
    for p in &clean {
        println!("{} (m={}, start_pos={})", p.tag, p.m, p.start_pos);
        let [q, kv, derot, out] = gpu.step(&d, p);
        // Ordered so the FIRST failure is the earliest stage: a wrong `.q` makes every
        // later tensor wrong too, and reporting the last one first would send the reader
        // to `wo_b` for a bug in `wq_a`.
        assert_within("q (wq_a..qk_norm..rope)", &q, golden(p, "q"));
        assert_within("kv_entry (wkv..act_quant)", &kv, golden(p, "kv_entry"));
        assert_within("attn_derot (de-rotation)", &derot, golden(p, "attn_derot"));
        assert_within("attn_out (wo_a, wo_b)", &out, golden(p, "attn_out"));
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
    let (rows, cols) = v4_window_topk(d.window, p.m, 0, &mut idx);
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
/// `QkNormAfterRope` and `KvActQuantBlock128` are absent ON PURPOSE — see the module
/// header. They are the two blind spots, and asserting a separation the oracle provably
/// cannot produce would turn a documented limitation into a passing test.
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
fn stages(v: &[Vec<f32>; 4]) -> [(&'static str, &[f32]); 4] {
    [("q", &v[0]), ("kv_entry", &v[1]), ("attn_derot", &v[2]), ("attn_out", &v[3])]
}

/// How far the largest move at any stage of any step is, scoring `mine` against the
/// captures in `refs`.
fn reach(refs: &[Phase], mine: &[[Vec<f32>; 4]]) -> i32 {
    refs.iter()
        .zip(mine)
        .flat_map(|(p, v)| stages(v).map(|(name, got)| score(got, golden(p, name)).max_ulp))
        .max()
        .expect("no steps captured")
}

#[test]
fn each_in_scope_defect_is_further_away_than_the_kernels_are() {
    let Harness { cfg, model, d, clean, mut gpu } = Harness::new();
    // The GPU's own output for every step, taken ONCE and reused: every distance below
    // is measured from the same point, so the numbers are comparable to each other and
    // not merely each to its own baseline.
    let mine: Vec<[Vec<f32>; 4]> = clean.iter().map(|p| gpu.step(&d, p)).collect();
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
        let broken = drive(&cfg, &model, defect);
        assert_eq!(broken.len(), clean.len(), "the defect changed the step schedule");
        let r = reach(&broken, &mine);
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
    assert_eq!(
        (listed.len(), outside),
        (15, 28),
        "the oracle's defect set changed: {} in S2b's scope, {outside} outside. Re-decide \
         which side each new breakage falls on -- and note that `QkNormAfterRope` and \
         `KvActQuantBlock128` are outside because the oracle CANNOT see them, not because \
         they are someone else's stage",
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
    // A prefill of 4 against a window of 8: `v4_window_topk` returns 4 columns, and the
    // whole point is that `window` is the plausible wrong answer.
    let short = 4usize;
    assert!(short < d.window, "the collision this test needs does not exist");
    let x = dev_f32(&golden(p, "attn_norm_out")[..short * d.dim]);
    let mut idx = Vec::new();
    let right = v4_window_topk(d.window, short, 0, &mut idx);
    assert_eq!(right, (short, short), "a short prefill no longer narrows its columns");
    let idxb = dev_i32(&idx);
    let mut io = Io {
        x: x.ptr().cast(),
        freqs: gpu.freqs.ptr().cast(),
        idxs: idxb.ptr().cast(),
        idxs_shape: (short, d.window),
        ring: gpu.ring.ptr_mut().cast(),
        out: gpu.out.ptr_mut().cast(),
    };
    let s = gpu.scratch();
    // SAFETY: buffers outlive the call; it returns before any launch.
    let e = unsafe { attention(&d, &gpu.weights, &s, &io, Step::Prefill { seqlen: short }) }
        .expect_err("a 4-row prefill must not accept an 8-column selection");
    assert!(format!("{e}").contains("selection"), "rejected for the wrong reason: {e}");

    io.idxs_shape = right;
    // SAFETY: as above; this one does launch, and the sync below joins it.
    unsafe { attention(&d, &gpu.weights, &s, &io, Step::Prefill { seqlen: short }) }
        .expect("the correct shape must be accepted");
    device_sync().expect("sync");

    // The decode arm, which the first draft of this test named and never ran. Decode
    // always wants `window` columns whatever the prompt was, so the plausible wrong
    // answer here is the narrowed prefill shape -- the exact inverse of the mistake
    // above, and it must be rejected too.
    io.idxs_shape = (1, short);
    let e = unsafe { attention(&d, &gpu.weights, &s, &io, Step::Decode { pos: PROMPT }) }
        .expect_err("a decode must not accept a narrowed prefill selection");
    assert!(format!("{e}").contains("selection"), "rejected for the wrong reason: {e}");
    let mut one = Vec::new();
    let want = v4_window_topk(d.window, 1, PROMPT, &mut one);
    assert_eq!(want, (1, d.window), "decode no longer wants the full window");
    let oneb = dev_i32(&one);
    io.idxs = oneb.ptr().cast();
    io.idxs_shape = want;
    // SAFETY: as above.
    unsafe { attention(&d, &gpu.weights, &s, &io, Step::Decode { pos: PROMPT }) }
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
/// activations essentially never are. That range is precisely where `v4_f2e4m3_rne` and
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
