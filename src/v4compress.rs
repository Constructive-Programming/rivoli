//! **V4-Flash KV compressor and sparse indexer — the host half.**
//!
//! The engine-side counterpart to `Compressor` (model.py:285-386) and `Indexer`
//! (model.py:386-442): the two per-layer RoPE tables, the shapes each `compress_ratio`
//! implies, and the two *arithmetic* selection paths (`get_window_topk_idxs` :261,
//! `get_compress_topk_idxs` :275) that need no learned weights and so belong on the host.
//!
//! The pooling itself, and the indexer's scoring, are device work. Those kernels and their
//! launchers do **not exist yet** — S2c's device half is unwritten as of this commit, and
//! naming a file here that `ls kernels/` does not show would send the next reader to prove a
//! negative.
//!
//! # This is NOT the DSA indexer in `indexer.rs`
//!
//! `indexer.rs` holds GLM's lightning-indexer constants (`MISA_BLOCK`, `K_NORM_EPS`). V4's
//! `Indexer` shares the name and nothing else: it has **no `wk` and no `k_norm`**, and
//! instead carries its own nested `Compressor` plus `wq_b` and `weights_proj`. S1a's first
//! cut guessed GLM's tensor names here and failed on the first real convert
//! (`docs/investigations/v4-flash-port.md` §S1a), so the two live in separate modules to
//! make the confusion cost a compile error rather than a silent-wrong load.
//!
//! # Why three functions are duplicated from `v4oracle`, and what that does NOT buy
//!
//! [`freqs_cis`], [`window_topk`] and [`compress_topk`] restate functions `src/v4oracle/`
//! also carries. The engine must not call into the oracle — it is the gate S2 and S3 are
//! scored against, and a dependency would close the loop — so the copies stay, under the
//! `jscpd:ignore` region below.
//!
//! **They are not independent derivations, and an earlier version of this doc claimed they
//! were.** They were written from the oracle's transliteration rather than re-derived from
//! `model.py`, so a shared misreading of the reference is present in both and nothing in
//! this repo currently detects it. The region's comment states the full argument; read it
//! before treating agreement between the two as evidence of anything but drift.

use crate::artifact::model::V4Config;
use anyhow::Result;

/// `compress_ratios` is `[0, 0, 4, 128, 4, 128, …, 4, 0, 0, 0]` — **46 entries for 43
/// layers**, the trailing three being the MTP blocks that `Transformer.forward` never runs.
///
/// The trap in the neighbourhood: the tail *looks* like the ratio-0 layers run out at the
/// end, so an implementation that trims to `n_layers` from the wrong end loses **layer 42**,
/// which is ratio-4 and carries both a compressor and an indexer. S1b's first cut did
/// exactly that (`docs/investigations/v4-flash-port.md` §S1b).
///
/// **This type does not catch that**, and saying otherwise would be the load-bearing lie:
/// [`LayerKind::from_ratio`] receives one already-extracted ratio and never sees the layer
/// index, `n_layers`, or the 46-entry vector. The guard that does catch it is
/// [`V4Config::compress_ratio`], which bounds-checks against `n_layers` rather than against
/// the vector — so prefer [`LayerKind::from_config`], which goes through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    /// `compress_ratio == 0`: no compressor, no indexer, base `rope_theta`, **no YaRN**.
    /// Layers 0, 1 and — in the config's 46-entry list — nothing else that actually runs.
    Plain,
    /// `compress_ratio == 4`: overlapping compressor **and** an `Indexer`.
    Overlap,
    /// `compress_ratio != 0 && != 4`: a non-overlapping compressor and **no** indexer;
    /// selection falls back to the arithmetic [`compress_topk`]. 128 in the shipped config,
    /// but the ratio is carried rather than hard-coded into the name because nothing in
    /// `model.py` fixes it — `Compressor.overlap` keys on `== 4` alone.
    ///
    /// [`LayerKind::from_ratio`] is the only constructor, so `NonOverlap(0)` and
    /// `NonOverlap(4)` — each of which would make `compressor_ratio()` disagree with
    /// `overlap()` and `has_indexer()` — are unrepresentable outside this module.
    NonOverlap(usize),
}

impl LayerKind {
    /// Classify `layer` through [`V4Config::compress_ratio`], which bounds-checks against
    /// `n_layers` rather than against the 46-entry vector.
    ///
    /// Preferred over [`LayerKind::from_ratio`]: same classification, reached via the one
    /// accessor that refuses to hand back an MTP block's ratio for a main-path layer.
    /// `V4Config::layer_has_compressor`/`layer_has_indexer` answer two of the same questions
    /// one at a time; this returns the whole classification at once, so a caller cannot pair
    /// a compressor decision from one call with a pooling width from another.
    pub fn from_config(cfg: &V4Config, layer: usize) -> Result<Self> {
        Ok(Self::from_ratio(cfg.compress_ratio(layer)?))
    }

    /// Classify one layer from a raw `compress_ratios` entry.
    ///
    /// Prefer [`LayerKind::from_config`] where a `V4Config` is in hand — this one trusts the
    /// caller to have indexed the vector correctly, which is the S1b trap above.
    pub fn from_ratio(ratio: usize) -> Self {
        match ratio {
            0 => Self::Plain,
            4 => Self::Overlap,
            r => Self::NonOverlap(r),
        }
    }

    /// `Compressor.overlap` — true iff `compress_ratio == 4` (model.py:296).
    pub fn overlap(self) -> bool {
        match self {
            Self::Overlap => true,
            Self::Plain | Self::NonOverlap(_) => false,
        }
    }

    /// `coff = 1 + overlap` (model.py:298). The width multiplier on `ape`, `wkv` and
    /// `wgate`: at ratio 4 the compressor projects to `2 * head_dim` and splits it into an
    /// overlapping half and a normal half, so `ape` is `[4, 1024]` at `head_dim = 512` but
    /// `[128, 512]` at ratio 128. **A shape assumption that holds on L2 breaks on L3.**
    pub fn coff(self) -> usize {
        1 + usize::from(self.overlap())
    }

    /// The ratio, if this layer has a `Compressor` at all — `None` for [`LayerKind::Plain`].
    ///
    /// The ONLY ratio accessor, deliberately. A companion returning a bare `usize` (0 for
    /// `Plain`) existed and was cut: every arithmetic path in this module divides by the
    /// ratio, so offering both would make each future caller choose between a `0` that
    /// panics the divide and a `None` that cannot — reintroducing exactly the hazard this
    /// type was added to remove. The reference's own guard is `if self.compress_ratio:`.
    pub fn compressor_ratio(self) -> Option<usize> {
        match self {
            Self::Plain => None,
            Self::Overlap => Some(4),
            Self::NonOverlap(r) => Some(r),
        }
    }

    /// An `Indexer` exists **only** where `compress_ratio == 4` (model.py:474) — 21 of the
    /// 43 layers, at indices `[2, 4, …, 42]`.
    pub fn has_indexer(self) -> bool {
        self.overlap()
    }
}

/// One resolved rotary parameter set — everything [`freqs_cis`] needs for one layer.
///
/// `theta` and `original_seq_len` are the only two fields that differ between the model's
/// two tables; see [`rope_for_layer`], which is the only thing that should ever set them.
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    /// Rotary width. Only the **last** `rope_head_dim` dims of a row are rotated.
    pub rope_head_dim: usize,
    /// `base` in `1 / base ** (2i / dim)`.
    pub theta: f32,
    /// `original_seq_len`. **Zero disables YaRN**, which is how a ratio-0 layer is expressed
    /// rather than by a separate flag.
    pub original_seq_len: usize,
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

/// The per-layer table selection `Attention.__init__` performs (model.py:481-488).
///
/// Takes the *compressed* parameter set and the one value a ratio-0 layer swaps in, so the
/// ratio-0 set is built with `..compressed`. That is deliberate: the reference's two tables
/// "differ ONLY in the pair" (`v4oracle/forward.rs:507`), and struct update syntax makes
/// that a fact the compiler maintains rather than a comment that rots — a rotary parameter
/// added later cannot silently apply to one table and not the other.
///
/// Selecting one value of the pair without the other is `Defect::RopeNoYarn` and
/// `Defect::RopeBaseThetaEverywhere` — two *distinct* silent-wrong outcomes the oracle
/// enumerates separately, and both are the slip a caller makes when theta and
/// `original_seq_len` travel as loose same-typed arguments. Here they move together, once.
///
/// A function rather than a two-table struct: the struct held two values, was built once,
/// and its accessor was this `if`.
pub fn rope_for_layer(compressed: RopeParams, rope_theta: f32, kind: LayerKind) -> RopeParams {
    // YaRN is keyed to COMPRESSION, not to layer index or to overlap: every ratio != 0
    // layer gets it, ratio 4 and ratio 128 alike.
    if kind.compressor_ratio().is_some() {
        compressed
    } else {
        // Not merely "skip the blend": the reference passes `original_seq_len = 0`, and the
        // base theta travels with it. Keeping 160000 here while dropping YaRN is
        // `Defect::RopeNoYarn` -- the frequencies stay plausible at every scale, which is
        // what makes it the insidious half of the selection.
        RopeParams { theta: rope_theta, original_seq_len: 0, ..compressed }
    }
}

// jscpd:ignore-start
//
// ENGINE COPIES of three functions `src/v4oracle/forward.rs` also carries: `freqs_cis`,
// `window_topk`, `compress_topk`. What the exemption buys, stated precisely, because the
// obvious stronger claim is FALSE and was written here before review caught it:
//
// WHAT IT BUYS. The engine must not call into `v4oracle`. The oracle is the gate S2 and S3
// are scored against and it must stay a pure judge -- `v4oracle/mod.rs` ("Not the engine")
// states the rule in the other direction, and a `use crate::v4oracle::...` here would close
// the loop and make the engine depend on its own instrument at runtime.
//
// WHAT IT DOES NOT BUY, and what the comment claimed until review caught it: these are NOT
// independent derivations. They were written by reading the oracle's transliteration, not by
// re-deriving from `model.py` with the oracle closed, and a mechanical diff shows
// `window_topk` and `compress_topk` are character-identical to the oracle's modulo
// `i32`/`i64`. A shared misreading of `model.py` is therefore present in BOTH copies and no
// test in this repo can currently see it. `window_topk_matches_the_oracle` is a DRIFT
// TRIPWIRE -- it catches one copy being edited without the other -- not the independent
// cross-check its name suggests.
//
// WHAT WOULD ACTUALLY CLOSE IT: a comparison against something that did not come from the
// oracle's source -- the reference implementation itself, or the goldens routed through a
// different code path. Recorded as an open gap in the S2c report, not described as covered.

/// `model.py::precompute_freqs_cis` — `[seqlen * rope_head_dim/2]` pairs of `(cos, sin)`.
///
/// Faithful to two quirks that are easy to "clean up" into a one-ulp disagreement:
///
/// 1. **`find_correction_range` works in `dim` units while `linear_ramp_factor` indexes
///    `dim // 2` entries.** The ramp bounds are computed against `dim` (64) and then applied
///    to 32 frequency indices. That is what the reference does; matching the *intent*
///    instead of the code shifts the interpolation band by a factor of two.
/// 2. **Only the correction range is f64.** `find_correction_dim` goes through Python's
///    `math.log`/`floor`/`ceil` and so is double precision, but `freqs`, the YaRN blend, the
///    outer product and `polar` are all float32 tensors. Computing the whole table in f64
///    leaves a sub-bf16-ulp angle error that still surfaces as sporadic one-ulp
///    disagreement against a faithful implementation -- and a one-ulp angle can flip a
///    `topk` tie, which is a *set* difference, not a tolerance one.
pub fn freqs_cis(p: RopeParams, seqlen: usize) -> Vec<(f32, f32)> {
    let dim = p.rope_head_dim;
    let half = dim / 2;
    let mut freqs: Vec<f32> =
        (0..half).map(|i| 1.0 / p.theta.powf((2 * i) as f32 / dim as f32)).collect();
    if p.original_seq_len > 0 {
        let base = f64::from(p.theta);
        let fcd = |rot: f64| {
            dim as f64 * (p.original_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * base.ln())
        };
        let low = fcd(p.beta_fast as f64).floor().max(0.0);
        let high = fcd(p.beta_slow as f64).ceil().min(dim as f64 - 1.0);
        // `linear_ramp_factor`'s `if min == max: max += 0.001` -- a guard against a zero
        // denominator, kept because dropping it turns a degenerate config into a NaN table
        // rather than into the reference's (arbitrary but finite) ramp.
        let (min, max) = if low == high { (low, high + 0.001) } else { (low, high) };
        let (min, max) = (min as f32, max as f32);
        for (i, f) in freqs.iter_mut().enumerate() {
            let smooth = 1.0 - ((i as f32 - min) / (max - min)).clamp(0.0, 1.0);
            *f = *f / p.factor * (1.0 - smooth) + *f * smooth;
        }
    }
    let mut out = Vec::with_capacity(seqlen * half);
    for t in 0..seqlen {
        for f in &freqs {
            let a = t as f32 * f;
            out.push((a.cos(), a.sin()));
        }
    }
    out
}

/// `model.py::get_window_topk_idxs` (:261) — which sliding-window ring slots each query row
/// may read, `-1` for "nothing there yet".
///
/// Returns one row per query: `seqlen` rows at prefill, exactly **one** at decode. The
/// values are ring *slots*, not positions, and the decode branch is a rotation rather than a
/// range precisely because slot `t % win` holds position `t`.
pub fn window_topk(win: usize, seqlen: usize, start_pos: usize) -> Vec<Vec<i32>> {
    // See `compress_topk`: the decode branches ignore `seqlen`, so a caller passing more
    // than one query row at `start_pos > 0` gets one row back for N queries and no error.
    // rivoli ships speculative decode on by default, so that caller is not hypothetical.
    debug_assert!(start_pos == 0 || seqlen == 1, "decode is one query row (bsz=1 scope cut)");
    if start_pos >= win.saturating_sub(1) && start_pos > 0 {
        // Ring is full: read oldest-first, starting just past the slot about to be
        // overwritten. `start_pos %= window_size` in the reference.
        let sp = start_pos % win;
        let mut v: Vec<i32> = ((sp + 1)..win).map(|i| i as i32).collect();
        v.extend((0..=sp).map(|i| i as i32));
        vec![v]
    } else if start_pos > 0 {
        // Ring not yet full: slots [0, start_pos] are live, the rest padded to `win`.
        let mut v: Vec<i32> = (0..=start_pos).map(|i| i as i32).collect();
        v.resize(win, -1);
        vec![v]
    } else {
        (0..seqlen)
            .map(|t| {
                let base = t.saturating_sub(win - 1);
                (0..win.min(seqlen))
                    .map(|j| {
                        let v = base + j;
                        // Causal: a slot beyond this query's own position is not yet written.
                        if v > t { -1 } else { v as i32 }
                    })
                    .collect()
            })
            .collect()
    }
}

/// `model.py::get_compress_topk_idxs` (:275) — the **indexer-free** compressed selection,
/// used on every layer whose ratio is neither 0 nor 4.
///
/// Pure arithmetic: with no `Indexer` there is no ranking, so every causally-legal
/// compressed block is selected and `index_topk` never applies. `offset` shifts the indices
/// into the attention's combined KV space (the window region occupies the low slots).
///
/// Takes [`LayerKind`], not a raw ratio, because ratio 0 has no compressor and every path
/// here divides by the ratio. The reference cannot reach that state — `Attention.__init__`
/// builds a `Compressor` only when `compress_ratio` is truthy and `Attention.forward` guards
/// the call with `if self.compress_ratio:` — but a `usize` parameter re-opens it, and
/// `LayerKind::Plain.ratio()` hands back the 0 that turns it into a panic.
pub fn compress_topk(
    kind: LayerKind,
    seqlen: usize,
    start_pos: usize,
    offset: usize,
) -> Vec<Vec<i32>> {
    let Some(ratio) = kind.compressor_ratio() else { return Vec::new() };
    // One row per query. `Attention.forward` zips this against the window list, so a decode
    // call carrying more than one query row would silently score every speculative row
    // against row 0's index set. The oracle asserts the same thing at its own call site
    // (`forward.rs:1412`) and calls it what ENFORCES the bsz=1 scope cut.
    debug_assert!(start_pos == 0 || seqlen == 1, "decode is one query row (bsz=1 scope cut)");
    if start_pos > 0 {
        vec![(0..(start_pos + 1) / ratio).map(|i| (i + offset) as i32).collect()]
    } else {
        (0..seqlen)
            .map(|t| {
                (0..seqlen / ratio)
                    // Block `c` covers positions [c*ratio, (c+1)*ratio); query `t` may read
                    // it only once the block is COMPLETE, i.e. c < (t+1)/ratio. The
                    // reference writes this as `arange(1, seqlen+1) // ratio`, so row `t`
                    // compares against `(t+1) // ratio` -- off-by-one here silently grants
                    // each query one block of the future.
                    .map(|c| if c >= (t + 1) / ratio { -1 } else { (c + offset) as i32 })
                    .collect()
            })
            .collect()
    }
}
// jscpd:ignore-end

/// Where the compressed region starts in the attention's combined KV index space
/// (`Attention.forward`, model.py:515 —
/// `offset = kv.size(1) if start_pos == 0 else win`, evaluated BEFORE the compressed rows
/// are concatenated at :531, which is why the prefill value is the bare prompt length).
///
/// At prefill the window region is the prompt itself, so the compressed rows are appended at
/// `seqlen`; at decode the window region is the full `window_size` ring. Two different
/// values for the same variable, and passing the prefill one at decode aims every selected
/// index at the wrong buffer -- fluent wrong text, no crash.
pub fn compress_offset(window_size: usize, seqlen: usize, start_pos: usize) -> usize {
    if start_pos == 0 { seqlen } else { window_size }
}

/// Does `Compressor.forward` emit a block on this call? (model.py:332 / :350.)
///
/// Prefill pools every complete block in one go and needs `seqlen >= ratio` to emit
/// anything at all; decode emits one block only on the position that completes it. **At
/// ratio 128 a 13-token prompt emits nothing**, which is why the ratio-128 pooling path is
/// unexercised by the oracle's shipped goldens (`bin/v4-oracle`'s `caveats` metadata).
pub fn should_compress(kind: LayerKind, seqlen: usize, start_pos: usize) -> bool {
    // `Plain` has no `Compressor` object at all, so the answer is false in BOTH phases. Read
    // from a raw `usize` this was `seqlen >= 0` (always TRUE) at prefill and
    // `is_multiple_of(0)` (always false) at decode -- two different answers for one layer,
    // and the prefill one would send every ratio-0 layer into `compress_topk` to divide by
    // zero. A decode-only smoke test would not have shown it.
    let Some(ratio) = kind.compressor_ratio() else { return false };
    if start_pos == 0 { seqlen >= ratio } else { (start_pos + 1).is_multiple_of(ratio) }
}
