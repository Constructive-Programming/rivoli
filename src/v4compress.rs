//! **V4-Flash KV compressor and sparse indexer — the host half.**
//!
//! The engine-side counterpart to `Compressor` (model.py:285-386) and `Indexer`
//! (model.py:386-442): the two per-layer RoPE tables, the shapes each `compress_ratio`
//! implies, and the two *arithmetic* selection paths (`get_window_topk_idxs` :261,
//! `get_compress_topk_idxs` :275) that need no learned weights and so belong on the host.
//!
//! The pooling itself is device work: `kernels/v4compress.hip`, launched in the reference's
//! own order by [`compress`] below and scored against `Oracle::compressor` by
//! `tests/v4_compress_kernel.rs`.
//!
//! The indexer's device half landed 2026-08-05 and is `kernels/v4indexer.hip`:
//! `act_quant_f4_rotated` (the Hadamard + fp4 finish, which is also [`Geom::indexer`]'s) and
//! `index_score_blocks` (the `einsum`/relu/weight/sum chain), scored by
//! `tests/v4_indexer_kernel.rs` — the spread against the oracle, the score against a host
//! transliteration of `model.py`, because the oracle's own head-sum is a confirmed defect as
//! of 2026-08-05. Read that file's header before treating a green run as a verdict. **The selection is not there** — the causal mask and the
//! top-k stay with the caller, because `Oracle::indexer` exports the full pre-top-k score
//! matrix and comparing THAT is strictly stronger than comparing the selected sets, which
//! are invariant at the shipped `index_topk` (see the plan doc's "A hole S3 inherits").
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
        RopeParams {
            theta: rope_theta,
            original_seq_len: 0,
            ..compressed
        }
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
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / p.theta.powf((2 * i) as f32 / dim as f32))
        .collect();
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
        let (min, max) = if low == high {
            (low, high + 0.001)
        } else {
            (low, high)
        };
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
    debug_assert!(
        start_pos == 0 || seqlen == 1,
        "decode is one query row (bsz=1 scope cut)"
    );
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
    let Some(ratio) = kind.compressor_ratio() else {
        return Vec::new();
    };
    // One row per query. `Attention.forward` zips this against the window list, so a decode
    // call carrying more than one query row would silently score every speculative row
    // against row 0's index set. The oracle asserts the same thing at its own call site
    // (`forward.rs:1412`) and calls it what ENFORCES the bsz=1 scope cut.
    debug_assert!(
        start_pos == 0 || seqlen == 1,
        "decode is one query row (bsz=1 scope cut)"
    );
    if start_pos > 0 {
        vec![
            (0..(start_pos + 1) / ratio)
                .map(|i| (i + offset) as i32)
                .collect(),
        ]
    } else {
        (0..seqlen)
            .map(|t| {
                (0..seqlen / ratio)
                    // Block `c` covers positions [c*ratio, (c+1)*ratio); query `t` may read
                    // it only once the block is COMPLETE, i.e. c < (t+1)/ratio. The
                    // reference writes this as `arange(1, seqlen+1) // ratio`, so row `t`
                    // compares against `(t+1) // ratio` -- off-by-one here silently grants
                    // each query one block of the future.
                    .map(|c| {
                        if c >= (t + 1) / ratio {
                            -1
                        } else {
                            (c + offset) as i32
                        }
                    })
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
    let Some(ratio) = kind.compressor_ratio() else {
        return false;
    };
    if start_pos == 0 {
        seqlen >= ratio
    } else {
        (start_pos + 1).is_multiple_of(ratio)
    }
}

/// Where this call's compressed block(s) belong in the **persistent** KV cache — the
/// `[ring ‖ compressed]` buffer `attn::v4::Io::cache` describes — as `(first_row, count)`.
/// `None` exactly when [`should_compress`] is false.
///
/// **A different coordinate system from [`compress_offset`], not the other half of one.**
/// That function indexes the SELECTION space — at prefill the transient
/// `torch.cat([kv, kv_compress])` (model.py:515/:531), so its prefill value is `seqlen`.
/// This one indexes rows of the PERSISTENT `[ring ‖ compressed]` buffer, whose base is
/// `window_size`. The two coincide at decode and disagree at prefill, and passing one where
/// the other is due is the fluent-wrong `compress_offset`'s own doc warns about. They are
/// two places because the reference keeps them in two places: `Compressor.kv_cache` is the
/// **view** `Attention.kv_cache[:, win:]` (model.py:497), so the compressor's own
/// `self.kv_cache[:, :seqlen // ratio]` at prefill and `[start_pos // ratio]` at decode
/// (model.py:380 and :382) land at `window_size + …` in the one flat buffer this engine keeps.
///
/// # `region_base` is 0 for the INDEXER's compressor
///
/// There are two `Compressor`s on every ratio-4 layer and they write into differently-shaped
/// buffers. `Attention.forward` hands its own the **view** above, so its blocks start at
/// `win`. `Indexer.forward` hands its nested one the *whole* buffer —
/// `self.compressor.kv_cache = self.kv_cache` (model.py:415) — and that buffer is registered
/// as `[max_batch_size, max_seq_len // compress_ratio, head_dim]` (model.py:405): **no ring,
/// no window prefix.** So the indexer's destination is `start_pos / ratio` outright, which is
/// this function with `region_base = 0`.
///
/// Spelled out because the natural wiring is to pass `cfg.sliding_window` at every call site,
/// and the 21 layers that need the zero are exactly the 21 that also need the non-zero. The
/// symptom would appear at the indexer, 128 rows past the end of a buffer this function's own
/// test says nothing about — an out-of-bounds write, or a silent overwrite if the two land in
/// one arena. The parameter is taken rather than derived from [`LayerKind`] precisely so both
/// callers can exist; it cannot be defaulted.
///
/// # `start_pos / ratio`, never "the next free slot"
///
/// The decode row is a pure function of the POSITION, and that is the whole content of
/// `docs/investigations/v4-flash-port.md` requirement 2. A caller that appends — keeps a
/// running count and writes the next row each time a block is emitted — agrees with this on
/// every sequence that visits every position in order, and diverges the first time a step is
/// skipped.
///
/// **The usual example is not currently reachable on V4, and saying otherwise would be the
/// load-bearing lie.** Requirement 2 motivates this with "speculative decode skips steps by
/// construction", which is true of rivoli's GLM path — but `kernels/moe.hip:409` refuses
/// `nrow != 1` (guard 1003, only `R = 1` instantiated), so a V4 decode is structurally
/// single-row and no verify pass can advance `start_pos` by two. The rule is still right and
/// still worth having: a skipped step is not exclusive to speculation, and the day the fp4
/// MoE gains an `R = 2` this becomes live with nothing to warn anyone.
///
/// The divergence is silent. An appended block is a real pooled row of real numbers sitting
/// one slot early, so every later query selects it BY POSITION, weights it `exp(l - max)`,
/// and attends a window that is off by `ratio` tokens on 41 of 43 layers — fluent, wrong, and
/// permanent for the rest of the sequence.
/// `tests/v4_compress.rs::compress_dst_is_positional_and_an_appending_placer_disagrees` pins it by
/// running a skipped step against an appending placer and asserting they disagree.
///
/// `seqlen` is read only at prefill and `start_pos` only at decode; both are taken because
/// [`should_compress`] needs the pair, and splitting this into two functions would put the
/// phase discriminant in the caller's hands twice.
pub fn compress_dst(
    kind: LayerKind,
    // NOT `window_size`, which is what this was called for one round. It is the compressed
    // region's BASE ROW, and that is `sliding_window` for the attention compressor and **0**
    // for the indexer's — see the section above. A parameter named for the value one of its
    // two callers passes is the shape `Dims::compress_slot` was deleted in: correct code
    // whose signature reads wrong at the call site.
    region_base: usize,
    seqlen: usize,
    start_pos: usize,
) -> Option<(usize, usize)> {
    let ratio = kind.compressor_ratio()?;
    if !should_compress(kind, seqlen, start_pos) {
        return None;
    }
    Some(if start_pos == 0 {
        // Prefill emits every COMPLETE block at once, starting at the region's base.
        (region_base, seqlen / ratio)
    } else {
        (region_base + start_pos / ratio, 1)
    })
}

// ---------------------------------------------------------------------------------------
// the device half — S2c's kernels and the order they run in
// ---------------------------------------------------------------------------------------

/// Which quantization `Compressor.forward` finishes with — the `rotate` flag of
/// `Compressor.__init__` (model.py:290), which is the ONLY thing that differs between the
/// attention compressor and the indexer's nested one.
///
/// An enum rather than a `bool`, and carried inside [`Geom`] rather than passed beside it,
/// because the two are a different *algorithm* over an identical *shape*. `compress` matches
/// on it exhaustively, so a third finish cannot be added without the compiler naming this
/// site — and there is no wildcard arm to swallow it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantize {
    /// `rotate = false` — model.py:378, `act_quant(kv[..., :-rd], 64, …)`: block-64 e4m3
    /// over dims `[0, d - rd)` only, leaving the RoPE tail in bf16 to match QAT.
    PartialFp8,
    /// `rotate = true` — model.py:374-376, `rotate_activation` then
    /// `fp4_act_quant(·, 32)`: the Hadamard spread and fp4 over the WHOLE row.
    HadamardFp4,
}

/// The `repr(C)` mirror of `kernels/v4compress.hip`'s `V4Comp`, and the only part of
/// [`Geom`] that crosses the ABI wall.
///
/// Split out from [`Geom`] when [`Geom::indexer`] landed. It would have been simpler to add
/// an eighth field to the one struct, and that is exactly what was rejected: `Geom` now
/// carries a [`Quantize`] that no kernel reads, and a `repr(C)` type whose Rust half has
/// fields the C half does not is the hazard the layout assertion below exists to catch —
/// "then the two structs are different objects and the guard is reading whichever bytes
/// happen to line up". Two types keeps one of them an exact mirror.
///
/// Fields are PRIVATE and [`Geom`]'s constructors are the only way in, because three of the
/// six integers are derived and a caller that computed one from a stale `coff` gets a kernel
/// that reads the right shape at the wrong stride — `ape`'s row stride is `cd`, and at ratio
/// 4 that is 1024 while at ratio 128 it is 512. Handing the kernel six loose `i32`s would
/// make every one of those positions plausible in the others.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeomAbi {
    ratio: i32,
    coff: i32,
    d: i32,
    cd: i32,
    ents: i32,
    rd: i32,
    eps: f32,
}

/// One compressor's geometry **and the finish it owes**.
///
/// Built from [`LayerKind`], not a raw ratio, so [`LayerKind::Plain`] — which has no
/// `Compressor` object at all in the reference — cannot produce one.
///
/// The pairing is the whole point of the type. `docs/investigations/v4-flash-port.md`
/// requirement 5 banks it: the indexer's compressor has the attention compressor's *shape*
/// and a different *algorithm*, so a geometry built for one and finished as the other passes
/// every dimension guard and runs block-64 e4m3 where fp4 over all 128 is due. Finite,
/// plausible, wrong. Here the algorithm is a field, chosen once by whichever constructor ran.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geom {
    abi: GeomAbi,
    quant: Quantize,
}

// The `repr(C)` mirror, pinned at compile time. `compress_guard` already catches a REORDER of
// the six `int`s at run time — swap `ratio`/`coff` or `cd`/`ents` and guard 1003 fires,
// swap `d`/`rd` and 1002 does — which a size check cannot do, since a reorder is
// size-invariant. What the guard cannot catch is a field being ADDED, REMOVED, or changed
// WIDTH on one side only, because then the two structs are different objects and the guard
// is reading whichever bytes happen to line up. These two catch that, at the file where the
// edit was made. (`backend/vk.rs`'s `ExpertDesc` assert was the same idiom until that file
// was deleted 2026-08-06; `src/backend/hip.rs`'s `ExpertDesc` carries it now.)
const _: () = assert!(
    size_of::<GeomAbi>() == 28 && align_of::<GeomAbi>() == 4,
    "GeomAbi must stay six i32 and one f32 — the layout kernels/v4compress.hip's V4Comp declares"
);
const _: () = assert!(
    size_of::<Finish>() == 3 * size_of::<*const f32>(),
    "Finish must stay three pointers — kernels/v4compress.hip's V4Fin"
);

impl Geom {
    /// The **attention** compressor's geometry — `rotate = false` (model.py:377-378: the
    /// partial block-64 `act_quant` over dims `[0, d - rd)`).
    ///
    /// **Named `attention`, not `new`, and that name is the guard.** The reference's other
    /// `Compressor` is the indexer's nested one, `rotate = true`, which does
    /// `rotate_activation` + `fp4_act_quant` over the WHOLE row instead — a different
    /// algorithm, not a different parameter. Its geometry is otherwise identical in shape
    /// (`Overlap`, `index_head_dim`, same `rope_head_dim`), so a `Geom` built for it would
    /// pass every launcher guard and [`compress`] would run the attention finish over it:
    /// block-64 e4m3 on dims `[0, 64)` where Hadamard-plus-fp4 on all 128 was due. Finite,
    /// plausible, wrong. Its constructor is [`Geom::indexer`], which landed **together with**
    /// the fp4 finish on 2026-08-05 as the plan's requirement 5 demands; the state stays
    /// unrepresentable because [`Quantize`] is a field chosen by whichever constructor ran,
    /// not an argument [`compress`] could be handed wrongly.
    ///
    /// `None` for [`LayerKind::Plain`]: the reference's `Attention.__init__` builds a
    /// `Compressor` only when `compress_ratio` is truthy, so there is no geometry to
    /// describe and every arithmetic path below would divide by zero.
    ///
    /// `d` is the *compressor's* head_dim — `args.head_dim` (512) for the attention
    /// compressor, `args.index_head_dim` (128) for the indexer's. They are different
    /// geometries on the SAME layer, and taking one for the other still "works".
    ///
    /// Only the ratio is validated here, by the `LayerKind` that carries it. `d`,
    /// `rope_head_dim` and `norm_eps` are taken on trust and checked by `compress_guard` at the
    /// launch (1002 for `rd > d` or odd `rd`, 1001 for a zero). That split is deliberate
    /// rather than an omission: those three come from the config, so a bad one is a bad
    /// checkpoint and belongs at the wall where every launcher sees it — but it does mean a
    /// `Geom` can exist that no launcher will accept, and the error names a code and not a
    /// field.
    pub fn attention(
        kind: LayerKind,
        d: usize,
        rope_head_dim: usize,
        norm_eps: f32,
    ) -> Option<Self> {
        // The one derivation of `cd`/`ents`: [`Geom::indexer`] refines what this returns
        // rather than repeating it, because two copies of the arithmetic are two places for
        // a `coff` to go stale.
        let ratio = kind.compressor_ratio()?;
        let coff = kind.coff();
        Some(Self {
            abi: GeomAbi {
                ratio: ratio as i32,
                coff: coff as i32,
                d: d as i32,
                cd: (coff * d) as i32,
                ents: (coff * ratio) as i32,
                rd: rope_head_dim as i32,
                eps: norm_eps,
            },
            quant: Quantize::PartialFp8,
        })
    }

    /// The **indexer's** compressor geometry — `rotate = true` (model.py:404,
    /// `Compressor(args, compress_ratio, self.head_dim, True)`).
    ///
    /// Landed together with the fp4 finish and not before, which is
    /// `docs/investigations/v4-flash-port.md` requirement 5 discharged: the state this type
    /// must not permit is a geometry that says "indexer" while [`compress`] runs the
    /// attention finish over it, and the way it is prevented is that [`Quantize`] is a field
    /// rather than a second argument to `compress`.
    ///
    /// `None` unless the layer is [`LayerKind::Overlap`]. An `Indexer` exists **only** where
    /// `compress_ratio == 4` (model.py:474 — 21 of the 43 layers, at indices `[2, 4, …, 42]`),
    /// and its nested `Compressor` is constructed with that same ratio, so ratio 128 and
    /// ratio 0 are both unrepresentable here rather than merely undocumented. That is
    /// stricter than [`Geom::attention`], which accepts every ratio with a compressor.
    ///
    /// `d` is `args.index_head_dim` (128), **not** `args.head_dim` (512). They are different
    /// geometries on the SAME layer and taking one for the other still "works" — every
    /// downstream shape is a multiple of both.
    pub fn indexer(kind: LayerKind, d: usize, rope_head_dim: usize, norm_eps: f32) -> Option<Self> {
        // `has_indexer`, not `compressor_ratio().is_some()`: the question here is whether the
        // layer HAS an indexer, and routing it through the accessor that answers exactly that
        // means a future change to which ratios carry one lands in a single place.
        if !kind.has_indexer() {
            return None;
        }
        // Sound because `quant` is not an input to any derived integer, which is not left to
        // this comment: `tests/v4_indexer_kernel.rs` asserts `a.abi() == i.abi()` for exactly
        // this pair. The `?` cannot fire — `has_indexer()` implies ratio 4, which
        // `compressor_ratio()` answers `Some` for.
        Some(Self {
            quant: Quantize::HadamardFp4,
            ..Self::attention(kind, d, rope_head_dim, norm_eps)?
        })
    }

    /// The `repr(C)` half, for the launchers. `pub` because `backend::hip` is a different
    /// module and Rust has no friend declaration; it hands out a shared reference to an
    /// all-private-field type, so a caller can pass it across the wall and cannot build or
    /// mutate one.
    pub fn abi(&self) -> &GeomAbi {
        &self.abi
    }

    /// Which finish this geometry owes. See [`Quantize`].
    pub fn quantize(self) -> Quantize {
        self.quant
    }

    /// `coff * d` — the width `wkv`/`wgate` project to and `ape`'s row stride.
    pub fn cd(self) -> usize {
        self.abi.cd as usize
    }

    /// `compress_ratio`. Public because [`crate::backend::hip`]'s safety contracts are
    /// stated in it — `ape` is `ratio * cd`, and a caller outside this module that cannot
    /// read the field cannot check the contract it is being asked to uphold.
    pub fn ratio(self) -> usize {
        self.abi.ratio as usize
    }

    /// This compressor's `head_dim` — the width of ONE pooled block. Not [`Geom::cd`],
    /// which is twice this on an overlapping layer.
    pub fn d(self) -> usize {
        self.abi.d as usize
    }

    /// `rope_head_dim`: only the LAST this-many dims of a pooled row are rotated, and the
    /// first `d - rd` are what `act_quant` covers.
    pub fn rd(self) -> usize {
        self.abi.rd as usize
    }

    /// `coff * ratio` — entries in one pooling window, and the row count of both state
    /// buffers.
    pub fn ents(self) -> usize {
        self.abi.ents as usize
    }

    /// Elements in `kv_state` / `score_state`: `[coff*ratio, coff*d]`.
    pub fn state_len(self) -> usize {
        self.ents() * self.cd()
    }
}

/// What the compressor's finish stage reads and writes — the same three pointers for both
/// pool kernels, in `kernels/v4compress.hip`'s `V4Fin` layout.
///
/// One struct rather than three parameters because the three must AGREE with each other and
/// nothing in a flat argument list makes them:
///
/// * `freqs` must be the layer's **compressed** table (`compress_rope_theta = 160000` WITH
///   YaRN). The ratio-0 table has the same type, stride and shape, and substituting it is
///   `Defect::RopeNoYarn` — plausible frequencies at every scale, fluent wrong text.
///   [`rope_for_layer`] is the only thing that should have produced it.
/// * `norm` is `[d]`, the **compressor's own** `RMSNorm` weight — not the layer's `kv_norm`,
///   which is also `[d]` and also f32.
/// * `out` is `[nblk, d]` at prefill and one row at decode.
///
/// The grouping was forced by `build.rs`'s duplication gate, which found the three repeated
/// across the two pool call sites. It is kept because the gate was right about the shape of
/// the problem: `Defect::RopeNoYarn` is a mismatch BETWEEN these pointers, so they should be
/// chosen once, together, and not three times each.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Finish {
    pub norm: *const f32,
    pub freqs: *const f32,
    pub out: *mut f32,
}

/// The four extents `Indexer.forward`'s scoring contracts over
/// (`einsum("bshd,btd->bsht")`, model.py:425).
///
/// A struct because they are four bare `usize` in a row and every one is plausible in any
/// other's position — the same argument `Finish` makes about its three pointers, and the one
/// `tests/common/mod.rs::Mla` makes about six. `heads` and `hd` in particular are 64 and 128
/// on every layer that has an indexer, so a transposed pair indexes a real row of `q` and
/// produces a finite score.
///
/// What it does NOT buy, stated because the neighbouring types do enforce something: there
/// are no derived fields here and no relation between them to check, so this is naming and
/// nothing else. `n_comp` is `end_pos / ratio` and the rest come from the config; a wrong
/// one is caught by the launcher's bounds or not at all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreDims {
    /// Query rows: the prompt length at prefill, 1 at decode.
    pub s: usize,
    /// Compressed blocks visible to this call, `(start_pos + seqlen) / ratio`.
    pub n_comp: usize,
    /// `args.index_n_heads` (64) — NOT the attention's `n_heads`, which is also 64.
    pub heads: usize,
    /// `args.index_head_dim` (128) — NOT `args.head_dim` (512).
    pub hd: usize,
}

#[cfg(feature = "rocm")]
pub use device::{Buffers, compress};

/// The launch sequence for one `Compressor.forward`.
///
/// Split out from `attn.rs`'s equivalent deliberately: the compressor is scored on its own
/// against `Oracle::compressor`, and S3 wires it into the layer loop. Nothing here touches
/// `gpu.rs`.
#[cfg(feature = "rocm")]
mod device {
    use super::{Finish, Geom, LayerKind, Quantize, should_compress};
    use crate::backend::hip::{
        launch_act_quant_f4_rotated, launch_act_quant_f8_prefix, launch_gemm_bf16,
        launch_kv_compress_decode, launch_kv_compress_deposit, launch_kv_compress_prefill,
    };
    use anyhow::{Result, ensure};
    use std::ffi::c_void;

    /// Every device pointer one `compress` call reads or writes.
    ///
    /// `scratch_rows` is carried WITH the two scratch pointers rather than beside them.
    /// `docs/investigations/v4-flash-port.md` §"A hole S3 inherits" records the matching
    /// hazard on S2b's `Scratch`: one sized for decode and handed a prefill overruns every
    /// buffer, silently, because a `*mut f32` has no length. Here the capacity travels with
    /// the pointers and [`compress`] checks it against the phase before any launch.
    pub struct Buffers {
        /// `[rows, dim]` f32 activations — the reference's `x.float()`.
        pub x: *const f32,
        /// `args.dim` — the model dim, and the reduction extent of both projections. It is
        /// NOT derivable from [`Geom`], which describes the compressor's output geometry
        /// only, and it is the one dimension `wkv`/`wgate`/`x` must agree on.
        pub dim: usize,
        /// `[cd, dim]` bf16. Checkpoint dtype; `bf16f` reproduces it exactly on device.
        pub wkv: *const u16,
        /// `[cd, dim]` bf16.
        pub wgate: *const u16,
        /// `[ratio, cd]` f32.
        pub ape: *const f32,
        /// `Compressor.norm`'s weight, the layer's COMPRESSED rotary table, and the
        /// destination. See [`Finish`] for what each must be and why they are one field.
        pub fin: Finish,
        /// `[ents, cd]` f32, zero-initialised.
        pub kv_state: *mut f32,
        /// `[ents, cd]` f32, **-inf**-initialised. Zero-initialising it instead makes every
        /// never-written slot a live pooling entry with weight `exp(0 - m)`, which is a
        /// plausible number and a wrong window.
        pub score_state: *mut f32,
        /// `[scratch_rows, cd]` f32.
        pub kv: *mut f32,
        /// `[scratch_rows, cd]` f32.
        pub score: *mut f32,
        pub scratch_rows: usize,
    }

    /// Run `Compressor.forward` for one layer, one call.
    ///
    /// Returns the number of compressed blocks written to `out` — **zero** where the
    /// reference returns `None`, which at prefill is any prompt shorter than `ratio` and at
    /// decode is every position that does not complete a block. A zero return is not a
    /// failure; it is the reference's own control flow, and the state writes still happened.
    ///
    /// # Runs entirely on `stream`, and the earlier decline recorded here is REVERSED
    ///
    /// This function used to run on the null stream, handing `null` to the four compress
    /// launchers that already took one, and this doc argued the fifth step
    /// ([`launch_act_quant_f8_prefix`]) should not gain a stream because that would leave it on a
    /// different stream from the four before it. Both halves of that have gone:
    ///
    /// - The premise it rested on — "with the `.f4` streaming pool that would give them
    ///   something to overlap with; neither exists yet" — died when the pool landed and was
    ///   measured at **1.082 ms/miss, 12.36 GB/s, `slot_stalls` 0** over the real 137.06 GiB
    ///   expert set. There is now a routed fetch on every layer to hide compute behind.
    /// - The count it stated was **short**, and by exactly the mechanism it was warning
    ///   about. It said "**six** of the launchers the V4 path uses take no stream, and they
    ///   are exactly S2b's set — the whole of `attn::v4::attention`". But
    ///   `attn::v4::attention` also makes six `memcpy_dtod` calls, and `memcpy_dtod` is a
    ///   BLOCKING `hipMemcpy`, not a launcher: host-blocking hid the hazard while the whole
    ///   block sat on stream 0, and it becomes a read racing a stream-ordered write the
    ///   moment the launchers move. The set is seven, and
    ///   [`crate::backend::hip::memcpy_dtod_async`] is the seventh.
    ///
    /// This function is on `stream` because the hand-off requires it: it writes `b.fin.out`,
    /// which at prefill is `s.kv`'s tail — the same buffer `attn::v4::attention` reads through
    /// `sparse_attn`. rivoli's streams are `hipStreamNonBlocking` (`async.hip`), so leaving
    /// the compressor on the null stream while attention runs on a stream would leave that
    /// hand-off unordered. **Pass the SAME stream to both**, in the reference's order
    /// (compressor, then attend).
    ///
    /// That pairing does not exist yet — as of 2026-08-05 the only caller of either function is
    /// a test harness, and the V4 layer loop that would put them back to back is being written
    /// separately. So the paragraph above is a REQUIREMENT ON THAT CALLER rather than a
    /// description of something the tree does, and it is stated here because this is where its
    /// reader is.
    ///
    /// The names and the history are in `docs/investigations/v4-flash-port.md`
    /// §"The prerequisite inventory was taken again" (item 5) and requirement 9.
    ///
    /// # The caller must place `out` at BOTH destinations at prefill
    ///
    /// `Compressor.forward` does two writes, and only one of them is `out`. It assigns
    /// `self.kv_cache[:, :seqlen // ratio]` — the PERSISTENT compressed region, which every
    /// later decode step selects by position — **and** returns the same blocks for
    /// `Attention.forward` to `torch.cat` onto the prompt's KV for THIS step's attention.
    ///
    /// So at prefill a caller must land these blocks at `s.kv + seqlen * head_dim` (the
    /// concatenation `attn::v4::attention` reads now) and COPY them to
    /// `cache + window_size * head_dim` (what the next decode reads).
    ///
    /// **A device copy after ONE call — never a second call.** [`Finish`] carries a single
    /// `out`, so "write both destinations" reads like two invocations; it is not. Every
    /// path here runs [`launch_kv_compress_deposit`] before the emit decision, and that is a
    /// read-modify-write of `kv_state`/`score_state` (the decode path also slides the
    /// window). A second call re-deposits the same rows into the pooling state and slides
    /// again — finite, plausible, wrong, and it corrupts exactly the state S3 requirement 3
    /// is about. At decode the single
    /// block goes to `cache + (window_size + start_pos / ratio) * head_dim` — the slot
    /// `start_pos / ratio`, NOT "the next free one": a caller that appends is right only
    /// until it skips a step, and speculative decode skips steps by construction.
    ///
    /// **`attention` does neither, deliberately** — it only reads those rows. Omitting the
    /// persistent write is silent and one phase delayed: the prefill attends correctly and
    /// the first decode step then reads an unwritten compressed region. Nothing in the tree
    /// zeroes that allocation and `hipMalloc` does not, so the reading is either garbage (a
    /// loud NaN blow-up) or — if some future allocator zeroes it — silent, because zeros are
    /// live entries at weight `exp(0 - max)` rather than absent ones. Recorded
    /// here rather than at the reading end because nobody editing this function greps
    /// `attn.rs`.
    ///
    /// # Safety
    /// Every pointer in `b` must satisfy the shape contract on its field and must outlive
    /// `stream`'s completion — not merely the call, which returns as soon as the last
    /// operation is *enqueued*. This function synchronizes nothing. `stream` is a live
    /// `hipStream_t`, or null for the default stream.
    pub unsafe fn compress(
        g: &Geom,
        b: &Buffers,
        seqlen: usize,
        start_pos: usize,
        stream: *mut c_void,
    ) -> Result<usize> {
        // `(seqlen, start_pos)`, the same pair `should_compress`, `window_topk` and
        // `compress_topk` above take and the same pair `Compressor.forward` takes. An enum
        // over the two phases lived here briefly and was cut: it made `start_pos == 0` at
        // decode unrepresentable, which is worth something, but it did so by introducing a
        // THIRD spelling of the pair in a module whose other three entry points use the
        // pair — and the state it actually needed to exclude is `seqlen > 1` at decode,
        // which it did not exclude either. That one is checked here.
        ensure!(
            start_pos == 0 || seqlen == 1,
            "compress: decode consumes one row (bsz=1 scope cut), got {seqlen}"
        );
        ensure!(
            seqlen <= b.scratch_rows,
            "compress: {seqlen} rows into scratch sized for {}",
            b.scratch_rows
        );
        let (ratio, d, rd) = (g.ratio(), g.d(), g.rd());
        // Reconstructed rather than passed in. `Geom` was BUILT from a `LayerKind` and
        // stores its ratio, so a second `kind` parameter could only ever disagree with it —
        // layer 42's `Geom` beside layer 41's `kind` gives a plausible wrong emission count
        // and no guard anywhere sees it. `from_ratio` is exact here: `Geom::attention` refuses
        // `Plain`, so the only ratios that reach this are 4 and the non-overlapping ones.
        let kind = LayerKind::from_ratio(ratio);

        // `kv = self.wkv(x)`, `score = self.wgate(x)` (model.py:329-330), then the state
        // deposit — which runs in BOTH phases and BEFORE the emit decision, because the
        // reference writes the state and only then returns `None`.
        //
        // `slot0` is the whole difference between the phases and needs no branch: prefill
        // has `start_pos == 0`, and `0 % ratio` is the 0 it wants.
        let slot0 = start_pos % ratio;
        // SAFETY: caller's contract on `b`; both GEMMs write scratch checked above.
        unsafe {
            launch_gemm_bf16(b.x, b.wkv, b.kv, seqlen, g.cd(), b.dim, stream)?;
            launch_gemm_bf16(b.x, b.wgate, b.score, seqlen, g.cd(), b.dim, stream)?;
            launch_kv_compress_deposit(
                b.kv,
                b.score,
                b.ape,
                b.kv_state,
                b.score_state,
                g.abi(),
                seqlen,
                slot0,
                stream,
            )?;
        }

        // Zero is the reference's own `return` before `self.norm(...)`, not a failure: at
        // ratio 128 it is every prompt under 128 tokens and 127 of every 128 decode steps.
        // The state deposit above already happened either way.
        let emitted = if !should_compress(kind, seqlen, start_pos) {
            0
        } else if start_pos == 0 {
            let nblk = seqlen / ratio;
            // SAFETY: as above; `b.fin.out` is `nblk · d` by its field contract.
            unsafe {
                launch_kv_compress_prefill(b.kv, b.score, b.ape, &b.fin, g.abi(), nblk, stream)?;
            }
            nblk
        } else {
            // SAFETY: as above; `b.fin.out` is one row of `d` by its field contract.
            unsafe {
                launch_kv_compress_decode(
                    b.kv_state,
                    b.score_state,
                    &b.fin,
                    g.abi(),
                    start_pos,
                    stream,
                )?;
            }
            1
        };

        if emitted > 0 {
            // `if self.rotate:` (model.py:374). The two arms are a different ALGORITHM over
            // an identical shape, which is why the choice is a field of `Geom` and not a
            // parameter here: both arms accept every geometry this function can hold, so
            // nothing downstream would reject the wrong one.
            //
            // Matched exhaustively and with no wildcard. `docs/investigations/v4-flash-port.md`
            // requirement 5 is that `Geom::indexer` may not exist without this arm, and a
            // `_ =>` here would let a third `Quantize` silently take the fp8 path.
            match g.quant {
                // model.py:378 `act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)`
                // — dims [0, d-rd) at block **64**, leaving the RoPE tail in bf16 to match
                // QAT. S2b's kernel, not a second one: `n` separate from `row_stride` is
                // exactly this partial extent, and a second e4m3 quantizer in this tree
                // would be the third (see `kernels/v4compress.hip`'s header).
                // SAFETY: `b.fin.out` is `emitted · d` writable f32 by the field's contract.
                Quantize::PartialFp8 => unsafe {
                    launch_act_quant_f8_prefix(
                        b.fin.out,
                        b.fin.out,
                        emitted,
                        d,
                        d - rd,
                        64,
                        stream,
                    )?
                },
                // model.py:374-376 `rotate_activation(kv)` then `fp4_act_quant(kv, 32, True)`
                // — the WHOLE row, `rd` included. Note what this does NOT do: it takes no
                // `rd` argument at all, because the indexer's finish has no partial extent.
                // Passing `d - rd` here would be the same silent-wrong in the other
                // direction.
                // SAFETY: as above; the spread reads and writes `emitted · d` in place.
                Quantize::HadamardFp4 => unsafe {
                    launch_act_quant_f4_rotated(b.fin.out, emitted, d, stream)?
                },
            }
        }
        Ok(emitted)
    }
}
