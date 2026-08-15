//! Shared drivers for the `v4_oracle*` test binaries — the V4-checkpoint-free slice of the
//! old tests/common tail (its Checkpoint/LayerKind-coupled half stays deferred to M8).
//!
//! Six sibling binaries `#[path]`-include this file: `v4_oracle` (the defect matrix),
//! `v4_oracle_codecs`, `v4_oracle_gate`, `v4_oracle_head_tail`, `v4_oracle_reduction` and
//! `v4_oracle_targeted`. They ask different questions of the same instrument, and every one
//! of them drives the SAME toy through the SAME grid — a second copy of [`model`] or [`run`]
//! would be a second fixture, and two fixtures that drift are two different claims wearing
//! one name. So the driver lives here and the questions live there.
#![allow(dead_code)]
// ^ each includer uses a subset of this file; without the allow, every binary warns about
// the helpers the other five need. The cost is that a helper nobody calls goes unreported,
// which is why anything moved in here has to be reachable from at least two of them.
#![allow(clippy::unwrap_used, clippy::expect_used)]
// ^ a test harness panics loudly on a broken fixture; that is the correct failure here.

use rivoli_oracles::golden::{Diff, GoldenSet};
use rivoli_oracles::v4oracle::forward::{Capture, Defect, LayerCtx, LayerW, Oracle};
use rivoli_oracles::v4oracle::numerics::{bf16_decode, bf16_encode};
use rivoli_oracles::v4oracle::toy::{self, ToyModel};
use rivoli_oracles::v4oracle::weights::{NamedRng, V4Config};
use std::sync::OnceLock;

/// The layer a probe drives: its weights and its index, together — the pair every probe
/// call threads and a bare `(usize, &LayerW)` invites swapping against other indices.
pub struct ProbeLayer<'a> {
    pub idx: usize,
    pub w: &'a LayerW,
}

pub fn prefill_capture(o: &Oracle, at: ProbeLayer<'_>, ids: &[u32], h: &mut Vec<f32>) -> Capture {
    let ProbeLayer { idx: layer, w: lw } = at;
    let mut st = o.fresh_state(layer);
    let mut cap = Capture::default();
    let step = LayerCtx {
        lw,
        layer,
        s: ids.len(),
        start_pos: 0,
        input_ids: ids,
        step_tag: "pre",
    };
    o.run_layer(&step, &mut st, h, &mut cap);
    cap
}

/// A deterministic RESIDUAL-STREAM block, `[s, hc_mult * dim]`, seeded by `tag`.
///
/// [`probe`] with the one row width that is not arbitrary. `hc_mult * dim` is what the mHC
/// residual is, and it was spelled at three call sites in two files under two different
/// treatments — `v4_oracle`'s `fixed_h` wrapped it and argued the wrapper was worth it, while
/// `f4_kernel` inlined the identical product twice with no comment. jscpd sees none of that
/// (each site was a single expression, far under its default `minLines: 5`), which makes it a
/// "known, not merely unseen" case rather than a licence to leave it.
///
/// Fixed per `tag` so a defect at prefill cannot change a later step's INPUT: only the
/// layer's own cached state carries a defect forward, which is what makes "this case is
/// unaffected" a statement about the defect rather than about propagation.
pub fn residual_probe(cfg: &V4Config, tag: &str, s: usize) -> Vec<f32> {
    probe(tag, s, cfg.hc_mult * cfg.dim)
}

/// A deterministic bf16 activation block, `[n, dim]`, seeded by `name`.
///
/// **Changing the draw or the `NamedRng` sequence re-bases goldens in five suites at once** —
/// `v4_oracle`, `f4_kernel`, `blockindex_kernel`, `kvcompress_kernel` and
/// `kvcompress_probe`. `v4_oracle` and `f4_kernel` reach it only through
/// [`residual_probe`], so neither file can see its own exposure from its own source.
///
/// This doc line was orphaned onto `indexer_w` until 2026-08-06, which is how a shared
/// fixture source ended up with nothing at its definition saying what it is shared by.
pub fn probe(name: &str, n: usize, dim: usize) -> Vec<f32> {
    let mut r = NamedRng::new(name);
    (0..n * dim)
        .map(|_| bf16_decode(bf16_encode(r.unit())))
        .collect()
}

// =======================================================================================
// the grid: one toy, one driver, six questions
// =======================================================================================

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Case {
    pub layer: usize,
    /// Prompt length. 5 fits the toy's `window_size = 8`; 12 does not, which is what makes
    /// the ring rotation and the ratio-8 compressor reachable at all.
    pub prompt: usize,
    pub phase: Phase,
}

pub const PROMPTS: [usize; 2] = [5, 12];
/// Enough decode steps that BOTH compress ratios complete a block from either prompt
/// length: `(start_pos + 1) % ratio == 0` needs 4 steps to be guaranteed for ratio 4 and 8
/// at both starts.
pub const DECODE_STEPS: usize = 4;

/// Both captures for one (layer, prompt, defect).
pub struct Run {
    pub pre: Capture,
    pub dec: Capture,
}

impl Run {
    pub fn of(&self, p: Phase) -> &Capture {
        match p {
            Phase::Prefill => &self.pre,
            Phase::Decode => &self.dec,
        }
    }
}

pub fn model() -> &'static (V4Config, ToyModel) {
    static M: OnceLock<(V4Config, ToyModel)> = OnceLock::new();
    M.get_or_init(|| {
        let cfg = V4Config::toy();
        let m = toy::build(&cfg);
        (cfg, m)
    })
}

pub fn fixed_ids(cfg: &V4Config, tag: &str, s: usize) -> Vec<u32> {
    let mut r = NamedRng::new(tag);
    (0..s).map(|_| r.below(cfg.vocab_size) as u32).collect()
}

pub fn run(layer: usize, prompt: usize, defect: Defect) -> Run {
    let (cfg, m) = model();
    let o = Oracle::new(cfg.clone(), defect);
    let lw = &m.layers[layer];
    let mut st = o.fresh_state(layer);
    let mut caps = [Capture::default(), Capture::default()];
    // One state, driven prefill-then-decode as the engine will drive it. The two captures
    // are separate so a phase can be asserted on its own; the STATE is shared, which is what
    // lets a prefill-only defect show up at decode and nowhere else.
    let steps = std::iter::once((0usize, "pre".to_string(), prompt, 0usize))
        .chain((0..DECODE_STEPS).map(|i| (1usize, format!("dec{i}"), 1, prompt + i)));
    for (slot, tag, s, start_pos) in steps {
        let mut h = residual_probe(cfg, &format!("h-{tag}"), s);
        let ids = fixed_ids(cfg, &format!("ids-{tag}"), s);
        let step = LayerCtx {
            lw,
            layer,
            s,
            start_pos,
            input_ids: &ids,
            step_tag: &tag,
        };
        o.run_layer(&step, &mut st, &mut h, &mut caps[slot]);
        // The head tail, on THIS layer's output. `bin/v4-oracle` deliberately refuses to do
        // that (see `HeadTailW`) because a logits vector taken at 4 of 43 layers is not a
        // quantity the model computes and would be misread as one. Here nothing is ever read
        // as a model quantity -- the whole family is a structural gate on the toy -- and the
        // composition buys the thing a standalone head fixture could not: every layer defect
        // is shown to REACH the logits, so the head tail is proved unable to mask one.
        o.head_tail(&m.head_tail, &h, s, &tag, &mut caps[slot]);
    }
    let [pre, dec] = caps;
    Run { pre, dec }
}

pub fn cases() -> Vec<Case> {
    let (cfg, _) = model();
    let mut v = Vec::new();
    for layer in 0..cfg.n_layers {
        for prompt in PROMPTS {
            for phase in [Phase::Prefill, Phase::Decode] {
                v.push(Case {
                    layer,
                    prompt,
                    phase,
                });
            }
        }
    }
    v
}

pub fn matching<'a>(ds: &'a [Diff], suffix: &str) -> Vec<&'a Diff> {
    // The whole suffix scheme rests on the leading dot. Without it `"norm_out"` would match
    // `head.*.final_norm_out` AND `.attn_norm_out` AND `.ffn_norm_out`, and `"_out"` would
    // sweep up `.attn_out`, `.ffn_out` and `.out` together -- a silent widening that reads
    // exactly like a correct row. With every suffix dotted there is no collision in the
    // current name set. That was previously only a naming convention and a comment; this is
    // what actually enforces it.
    assert!(
        suffix.starts_with('.'),
        "golden suffix {suffix:?} must carry its leading dot"
    );
    ds.iter().filter(|d| d.name.ends_with(suffix)).collect()
}

/// A fingerprint of a whole capture, for the "no two defects are the same defect" check.
///
/// Hashes the SERIALIZED form rather than walking the fields, so it covers names and shapes
/// as well as values -- and so there is only one place that knows how a capture is laid out.
pub fn fingerprint(c: &Capture) -> u64 {
    let mut buf = Vec::new();
    GoldenSet::from_capture(Vec::new(), c.clone())
        .write(&mut buf)
        .unwrap();
    buf.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A bf16 STORE, as the reference performs it. Spelled once because the fixtures that have to
/// be bf16-exact build their draws through it and [`all_bf16`] asks the same question in
/// reverse — two spellings of one rounding is two chances to round only one of them.
pub fn bf16_round(x: f32) -> f32 {
    bf16_decode(bf16_encode(x))
}

/// Is every value in `v` exactly representable in bf16 — i.e. would a [`bf16_round`] store
/// leave it alone?
pub fn all_bf16(v: &[f32]) -> bool {
    v.iter().all(|&x| bf16_round(x).to_bits() == x.to_bits())
}
