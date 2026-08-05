//! Attention row-selection modes. The MLA absorb + flash-attend core (gpu.rs) is
//! row-set-agnostic; only which cached tokens a step attends over differs by
//! [`AttnMode`]. The DSA/MISA lightning-indexer selection itself runs on device
//! (gpu.rs `dsa_select_layer` + indexer.hip); this module holds the mode enum and
//! the position-based StreamingLLM row set.
//!
//! It ALSO holds the whole DeepSeek-V4-Flash attention block, in `mod v4` and the two
//! free functions above it — see the banner below `streaming_rows`. V4 is MQA and shares
//! none of the MLA machinery, so nothing before that banner applies to it.

use anyhow::{Result, bail, ensure};
use crate::v4compress::LayerKind;

/// Which tokens each decode step attends over. Selected once per layer per token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnMode {
    /// Full softmax over every cached token. Exactly the trained model at
    /// ≤ index_topk context; mildly out-of-distribution beyond.
    Dense,
    /// StreamingLLM: first `sinks` tokens + last `window` tokens, position based,
    /// no weights. Bounds attention BANDWIDTH, not cache memory.
    Streaming { sinks: usize, window: usize },
    /// Native DSA: the trained lightning indexer picks top-index_topk tokens per
    /// full layer; shared layers reuse the nearest preceding full layer's selection
    /// (IndexShare). Needs the resident indexer weights.
    Dsa,
    /// DSA with MISA head routing (arXiv 2605.07363): only `active_heads` of the
    /// indexer heads score tokens (routed by a block-pool estimate per full layer).
    Misa { active_heads: usize },
}

/// StreamingLLM row set over `nt` cached tokens: the first `sinks` tokens plus the
/// last `window` tokens, ascending, overlap-free. Never empty for `nt ≥ 1` (a
/// zero-sink zero-window config still attends the current token — the window floor
/// is the row that was just appended).
pub fn streaming_rows(nt: usize, sinks: usize, window: usize, rows: &mut Vec<u32>) {
    rows.clear();
    let sink_end = sinks.min(nt);
    let win_start = nt.saturating_sub(window.max(1)).max(sink_end);
    rows.extend(0..sink_end as u32);
    rows.extend(win_start as u32..nt as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_rows_shapes() {
        let mut r = Vec::new();
        // Fewer tokens than sinks+window → everything (dense-equivalent).
        streaming_rows(5, 4, 100, &mut r);
        assert_eq!(r, vec![0, 1, 2, 3, 4]);
        // Disjoint sinks + window.
        streaming_rows(100, 4, 10, &mut r);
        assert_eq!(&r[..4], &[0, 1, 2, 3]);
        assert_eq!(&r[4..], (90u32..100).collect::<Vec<_>>().as_slice());
        // Window overlapping the sinks clips, no duplicates.
        streaming_rows(10, 8, 5, &mut r);
        assert_eq!(r, (0u32..10).collect::<Vec<_>>());
        // Degenerate zero-sink zero-window still attends the current token.
        streaming_rows(50, 0, 0, &mut r);
        assert_eq!(r, vec![49]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════════
// DeepSeek-V4-Flash attention — S2b of docs/investigations/v4-flash-port.md
// ═══════════════════════════════════════════════════════════════════════════════════
//
// V4 shares nothing with the MLA path above. It is MQA: one `head_dim`-wide `wkv` entry
// is simultaneously the key and the value for all `n_heads` heads, its last
// `rope_head_dim` dims carry the position, and there is no `--attn` mode to pick
// (`arch.rs::attn_modes` returns `None` for `Arch::DeepseekV4`) — the row set is the
// sliding window, plus a compressed region that S2c adds.
//
// The two host functions below are pure, and they sit at MODULE scope rather than in
// `mod v4` for a reason that is easy to undo by tidying: `mod v4` is
// `#[cfg(feature = "rocm")]`, while `tests/v4_attn_host.rs` deliberately carries no
// feature gate so the selection can be scored against the oracle on any machine. Moving
// them inside breaks that test's build, not its assertions.
//
// Only ONE of them is independent of the oracle in the way that matters.
// `v4_topk_idxs` is written against `model.py`, not against
// `v4oracle::forward::window_topk` — that one is `pub` for the defect matrix's use, and
// calling it here would make the selection gate vacuous, since the engine and the
// instrument would then agree by construction. `v4_rope_table_ratio0` gets no such
// credit: the oracle's `precompute_freqs_cis` is private, so there was never an option
// to share it, and its only cross-check is an out-of-tree numpy transliteration of
// `precompute_freqs_cis` (agreed to <= 16 f32 ULP, pure libm spread) plus the end-to-end
// `.q` golden.

/// The `(cos, sin)` table `apply_rotary_emb` reads, interleaved as
/// `tbl[pos * rope_head_dim + 2*i]` = cos, `+ 2*i + 1` = sin, for `i` in `0..rd/2`.
///
/// **`compress_ratio == 0` LAYERS ONLY.** `Attention.__init__` (model.py:481-488) builds
/// two tables: a compressed layer passes `original_seq_len` and `compress_rope_theta` and
/// gets the YaRN interpolation, a ratio-0 layer passes `original_seq_len = 0` — which
/// disables the branch entirely — and the base `rope_theta`. This function has no YaRN
/// parameters at all rather than a flag defaulting to off, because a table built with the
/// right theta and no interpolation is `Defect::RopeNoYarn`: the frequencies stay
/// plausible at every scale and the text stays fluent. S2c owns the compressed table.
pub fn v4_rope_table_ratio0(rd: usize, max_pos: usize, theta: f32) -> Vec<f32> {
    let inv: Vec<f32> =
        (0..rd / 2).map(|i| 1.0 / theta.powf((2 * i) as f32 / rd as f32)).collect();
    let mut out = Vec::with_capacity(max_pos * rd);
    for t in 0..max_pos {
        for f in &inv {
            let a = t as f32 * f;
            out.push(a.cos());
            out.push(a.sin());
        }
    }
    out
}

/// What one selection is over — the arguments [`v4_topk_idxs`] and [`Sel::shape`] share.
///
/// A struct because three of its fields are `usize` and any permutation of
/// `win`/`seqlen`/`start_pos`/`index_topk` type-checks, while the failure is not a panic:
/// it attends to real vectors at the wrong positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sel {
    /// `sliding_window` — the ring's size, and the window part's column count at decode.
    pub win: usize,
    /// The layer class. [`LayerKind`], not a `usize` ratio, and that is not style: every
    /// path below divides by the ratio, and `LayerKind::Plain` — which has no compressor —
    /// would hand back the 0 that turns the division into a panic. `v4compress` carries
    /// the same argument at its own `compress_topk`.
    pub kind: LayerKind,
    /// `index_topk` from the config (512). Read only on a layer that `has_indexer`, where
    /// it is the length past which this positional selection REFUSES — see
    /// [`v4_topk_idxs`].
    pub index_topk: usize,
    /// Query rows: the prompt length at prefill, 1 at decode.
    pub seqlen: usize,
    /// 0 means prefill, throughout the reference.
    pub start_pos: usize,
}

impl Sel {
    /// The `(rows, cols)` a buffer for this selection must hold.
    ///
    /// Fallible for the reason [`v4_topk_idxs`] is, and derived from the same two
    /// `v4compress` functions the fill uses, so `attention` can check a caller's uploaded
    /// shape without building the selection twice.
    pub fn shape(&self) -> Result<(usize, usize)> {
        if self.win == 0 {
            return Ok((0, 0));
        }
        let rows = if self.start_pos == 0 { self.seqlen } else { 1 };
        let win_cols =
            if self.start_pos == 0 { self.seqlen.min(self.win) } else { self.win };
        Ok((rows, win_cols + self.n_comp()?))
    }

    /// Compressed columns — `end_pos // ratio`, refused past the truncation point.
    fn n_comp(&self) -> Result<usize> {
        let Some(ratio) = self.kind.compressor_ratio() else { return Ok(0) };
        let rows = if self.start_pos == 0 { self.seqlen } else { 1 };
        let live = (self.start_pos + rows) / ratio;
        // **REFUSE, do not cap.** `Indexer.forward` keeps the top `index_topk` blocks BY
        // SCORE; taking the first `index_topk` positionally keeps the OLDEST, so past this
        // length the engine would attend blocks 0..511 and nothing newer — every position
        // between `ratio * index_topk` and `pos - window` unattended, on 21 of 43 layers,
        // for the rest of the sequence. Fluent, wrong, and permanent.
        //
        // `min(live, index_topk)` was written here first and is exactly that bug. The
        // truncation point is `ratio * (index_topk + 1)` = 2052 at the shipped config,
        // which is the coverage cliff `docs/investigations/v4-flash-port.md` records: below
        // it the set is fixed by the causal mask and this function is right, above it only
        // the real indexer is.
        if self.kind.has_indexer() && live > self.index_topk {
            bail!(
                "v4 selection: {live} compressed blocks at position {} exceeds index_topk {} on a \
                 ratio-{ratio} layer. Past {} positions the block set is decided by the \
                 indexer's SCORES, which are not wired yet; a positional selection here \
                 keeps the OLDEST blocks and silently stops attending everything newer.",
                self.start_pos + rows,
                self.index_topk,
                ratio * (self.index_topk + 1)
            );
        }
        Ok(live)
    }
}

/// `model.py`'s `torch.cat([get_window_topk_idxs(...), get_compress_topk_idxs(...)], -1)`
/// as one row-major `i32` buffer, which is what the kernel indexes.
///
/// Appends `rows * cols` entries to `out` and returns `(rows, cols)`; `-1` masks a slot.
///
/// **This function ports nothing.** Both halves are [`crate::v4compress`]'s — a review
/// found this file had re-derived `window_topk`, `compress_topk`, `compress_offset` and
/// `should_compress` alongside that module's existing ports of the same four reference
/// functions, with different spellings, so jscpd could not see it. The values agreed; they
/// were not equivalent by construction, and they had already diverged on whether
/// `index_topk` applies. This is the flattening and the concatenation, nothing else.
///
/// **The two phases index DIFFERENT spaces.** At prefill the window columns are absolute
/// positions `0..seqlen` and the compressed ones continue from `seqlen`; at decode they are
/// ring SLOTS `0..window_size` and the compressed ones continue from `window_size`.
/// `v4compress::compress_offset` owns that split; the write-side half belongs to whoever
/// places the compressor's output, which is `v4compress::compress`'s caller.
///
/// **This is the POSITIONAL compressed selection.** For a ratio-128 layer that is the whole
/// story — the reference has no `Indexer` there. For a ratio-4 layer it stands in for
/// `Indexer.forward` and agrees with it only on the SET, only below `ratio * (index_topk +
/// 1)` = 2052 positions, and never on the score ORDER that `sparse_attn`'s online softmax
/// folds in. Past that length it REFUSES rather than degrading — see [`Sel::n_comp`].
pub fn v4_topk_idxs(sel: Sel, out: &mut Vec<i32>) -> Result<(usize, usize)> {
    if sel.win == 0 {
        return Ok((0, 0));
    }
    // `out` is APPENDED to, so every length check below is against this mark and not
    // against `out.len()` outright.
    let start = out.len();
    let (rows, cols) = sel.shape()?;
    let win = crate::v4compress::window_topk(sel.win, sel.seqlen, sel.start_pos);
    // The window columns `window_topk` actually produced; `n_comp` is the rest of `cols`.
    // Derived from the fill rather than re-running `Sel::n_comp` — which `shape()` has
    // already called, so a second call could only return the same value or diverge.
    let n_comp = cols - win.first().map_or(0, Vec::len);
    let comp = if n_comp == 0 {
        Vec::new()
    } else {
        let offset = crate::v4compress::compress_offset(sel.win, sel.seqlen, sel.start_pos);
        crate::v4compress::compress_topk(sel.kind, sel.seqlen, sel.start_pos, offset)
    };
    for (t, w) in win.iter().enumerate() {
        out.extend_from_slice(w);
        if let Some(c) = comp.get(t) {
            out.extend_from_slice(c);
        }
    }
    // ONE check, and it must be `ensure!` rather than `debug_assert!`: this crate's
    // `[profile.release]` sets no `debug-assertions` and CLAUDE.md prescribes
    // `cargo test --release`, so a `debug_assert!` here is compiled out of every build
    // this repo runs — verified, `cfg!(debug_assertions)` is false under `-O` and the
    // body is not even evaluated. A ragged selection buffer is a silent-wrong: the kernel
    // reads a row's worth of whatever follows as attention indices.
    //
    // Against `start`, not `out.len()`, because this function APPENDS — a `% cols == 0`
    // test on the whole buffer fires on a legitimate non-empty append and passes on a
    // short final row. It covers a short `win`, a short `comp` row and a missing `comp`
    // row together, so the per-row checks it replaced were three ways to say this.
    ensure!(
        out.len() - start == rows * cols,
        "v4 selection: wrote {} entries, {rows}x{cols} needs {}",
        out.len() - start,
        rows * cols
    );
    Ok((rows, cols))
}

#[cfg(test)]
mod v4_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    #[test]
    fn window_topk_prefill_is_causal_and_masks_nothing_reachable() {
        let mut v = Vec::new();
        // Prompt shorter than the window: `cols` shrinks to the prompt, and row `t` sees
        // exactly `0..=t`. A caller that assumed `win` columns would read past the row.
        let (rows, cols) = v4_topk_idxs(Sel { win: 8, kind: LayerKind::Plain, index_topk: 0, seqlen: 3, start_pos: 0 }, &mut v).unwrap();
        assert_eq!((rows, cols), (3, 3));
        assert_eq!(v, vec![0, -1, -1, 0, 1, -1, 0, 1, 2]);

        // Prompt past the window: the oldest position drops out of each row.
        v.clear();
        let (rows, cols) = v4_topk_idxs(Sel { win: 2, kind: LayerKind::Plain, index_topk: 0, seqlen: 4, start_pos: 0 }, &mut v).unwrap();
        assert_eq!((rows, cols), (4, 2));
        assert_eq!(v, vec![0, -1, 0, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn window_topk_decode_rotates_only_after_the_ring_fills() {
        let mut v = Vec::new();
        // Before the wrap: slots 0..=start_pos, then masked. Ascending, NOT rotated —
        // rotating here would name slots the prefill never wrote.
        let (rows, cols) = v4_topk_idxs(Sel { win: 4, kind: LayerKind::Plain, index_topk: 0, seqlen: 1, start_pos: 2 }, &mut v).unwrap();
        assert_eq!((rows, cols), (1, 4));
        assert_eq!(v, vec![0, 1, 2, -1]);

        // At exactly `win - 1` the ring is full and the rotation starts; `start_pos = 3`
        // wrote slot 3, so the oldest slot is 0 and the list is already in order.
        v.clear();
        v4_topk_idxs(Sel { win: 4, kind: LayerKind::Plain, index_topk: 0, seqlen: 1, start_pos: 3 }, &mut v).unwrap();
        assert_eq!(v, vec![0, 1, 2, 3]);

        // Past the wrap: slot `start_pos % win` holds the newest token and must come
        // LAST. At start_pos=5, win=4 the newest is slot 1, so the order is 2,3,0,1.
        v.clear();
        v4_topk_idxs(Sel { win: 4, kind: LayerKind::Plain, index_topk: 0, seqlen: 1, start_pos: 5 }, &mut v).unwrap();
        assert_eq!(v, vec![2, 3, 0, 1]);
        // Every slot appears exactly once once the ring is full -- a rotation that
        // dropped or repeated one would still look ordered.
        let mut seen = v.clone();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2, 3]);
    }

    #[test]
    fn rope_table_is_the_reference_frequency_ladder() {
        // Position 0 is (1, 0) for every pair -- the identity rotation. A table that
        // indexed `pos` wrong would almost always still start here, so the real check is
        // the pair below.
        let t = v4_rope_table_ratio0(4, 3, 10000.0);
        assert_eq!(&t[..4], &[1.0, 0.0, 1.0, 0.0]);
        // Pair `i` turns at theta^(-2i/rd): pair 0 at 1 rad/step, pair 1 at 1/100.
        for (pos, row) in t.chunks_exact(4).enumerate() {
            let (a0, a1) = (pos as f32, pos as f32 * 0.01);
            assert_eq!((row[0], row[1]), (a0.cos(), a0.sin()));
            assert_eq!((row[2], row[3]), (a1.cos(), a1.sin()));
        }
    }
}

/// The V4 attention block on device — `Attention.forward` (model.py:490-548), for
/// **any** layer class.
///
/// It was ratio-0 only until S3: `attention` derived the selection shape itself and
/// refused anything else, which made it unable to run 41 of the 43 layers. It now takes a
/// [`Sel`] carrying the layer's [`LayerKind`], and the compressed region is the tail of
/// `s.kv` at prefill and of `io.cache` at decode.
///
/// HIP-only, because [`crate::backend::hip`] is where the `v4_*` launchers live and
/// `backend/vk.rs` has no twin. That is S3's decision to make, not a gap to paper over
/// with stubs that would claim a parity nothing has measured.
///
/// This does not touch `gpu.rs`'s layer loop: it takes device pointers the caller owns
/// and performs one block's launches, so `tests/v4_attn.rs` drives exactly what S3 will.
#[cfg(feature = "rocm")]
pub mod v4 {
    use crate::artifact::model::V4Config;
    use crate::backend::hip::{
        launch_v4_act_quant, launch_v4_gemv_fp8, launch_v4_qk_norm, launch_v4_rmsnorm,
        launch_v4_rope, launch_v4_sparse_attn, memcpy_dtod,
    };
    use anyhow::{Result, bail, ensure};

    /// The block size V4's attention weights are quantized on — `weight_block_size:
    /// [128, 128]` in the checkpoint's `quantization_config`, and also `kernel.py`'s
    /// `block_size` for the ACTIVATION quantization every quantized `Linear` performs.
    /// The KV entry's partial quantization is the one place that is not 128; it is
    /// spelled at its call site below.
    const FP8_BLOCK: usize = 128;

    /// One fp8-e4m3 weight and its block scales, device-resident.
    #[derive(Clone, Copy)]
    pub struct Fp8W {
        pub w: *const u8,
        /// `ceil(rows/128) * ceil(cols/128)` f32. The artifact widens the checkpoint's
        /// `F8_E8M0` byte to f32 at conversion (`format.rs::copy_fp8_e8m0`), exactly —
        /// every e8m0 code is a power of two and so is exact in f32.
        pub scale: *const f32,
    }

    /// One layer's attention weights, device-resident.
    ///
    /// `wo_a` is an [`Fp8W`] like the rest and is read through the same GEMV, but it is
    /// NOT arithmetically the same: no activation quantization is performed against it.
    /// See the DECIDED note of 2026-08-05 in `docs/investigations/v4-flash-port.md`.
    ///
    /// Holding it fp8 rather than as the bf16 `convert.py` produces is not a deviation
    /// from that note: over the scale range weight tensors use, dequantizing an e4m3
    /// value against a power-of-two block scale gives bit-identical bf16 — measured over
    /// every e4m3 code by `tests/v4_attn_host.rs::fp8_times_a_power_of_two_is_exact_in_bf16_over_the_range_the_checkpoint_uses`.
    #[derive(Clone, Copy)]
    pub struct Weights {
        pub wq_a: Fp8W,
        pub q_norm: *const f32,
        pub wq_b: Fp8W,
        pub wkv: Fp8W,
        pub kv_norm: *const f32,
        /// `[n_heads]` f32 — the softmax DENOMINATOR's per-head term.
        pub attn_sink: *const f32,
        pub wo_a: Fp8W,
        pub wo_b: Fp8W,
    }

    /// Per-call scratch, device-resident, sized for `rows` query rows.
    ///
    /// `qr` and `qrq` are separate buffers holding the same values, and that is not
    /// waste: `qr` is `q_norm(wq_a(x))` and a ratio-4 layer's `Indexer` consumes it
    /// AFTER the q path is done with it (model.py:509/519), so quantizing in place here
    /// would hand S2c a destroyed input. `xq` is separate for the same reason on `x`,
    /// which the compressor and the indexer both read.
    #[derive(Clone, Copy)]
    pub struct Scratch {
        /// How many query rows every buffer below is sized for, checked against the `m`
        /// the step implies. The failure it catches is REUSE — allocate once at `max_m`,
        /// then hand a `Prefill` to a decode-sized scratch. Same hazard and same fix as
        /// `v4compress::Buffers::scratch_rows`, which cites this struct for its reason.
        pub rows: usize,
        /// `[m, dim]` — the activation-quantized copy of `x`.
        pub xq: *mut f32,
        /// `[m, q_lora_rank]`
        pub qr: *mut f32,
        /// `[m, q_lora_rank]`
        pub qrq: *mut f32,
        /// `[m, n_heads * head_dim]`
        pub q: *mut f32,
        /// `[rows, head_dim]` on a ratio-0 layer; **`[rows + rows/ratio, head_dim]` on a
        /// compressed one**, because at prefill `sparse_attn` reads
        /// `torch.cat([kv, kv_compress], dim=1)` and the selection indexes that
        /// concatenation as one space. The compressor writes the tail, at
        /// `v4compress::compress`. Nothing here can check it — see [`Scratch::rows`].
        pub kv: *mut f32,
        /// `[m, n_heads * head_dim]`
        pub o: *mut f32,
        /// `[m, o_groups * o_lora_rank]`
        pub y: *mut f32,
    }

    /// The block's inputs, persistent state and output.
    #[derive(Clone, Copy)]
    pub struct Io {
        /// `[m, dim]` — `attn_norm`'s output. Not modified.
        pub x: *const f32,
        /// Interleaved `(cos, sin)` for THIS layer's class.
        ///
        /// **Build it as `v4compress::freqs_cis(rope_for_layer(compressed, rope_theta,
        /// kind), ..)`, with the same `kind` this call gets.** A ratio-0 layer uses
        /// `rope_theta = 10000` with YaRN off; every compressed layer uses
        /// `compress_rope_theta = 160000` with it. The two tables have the same type,
        /// stride and shape, so nothing here can tell them apart, and swapping them is
        /// `Defect::RopeNoYarn` — plausible frequencies at every scale, fluent wrong text.
        /// S3 requirement 4.
        ///
        /// **Nothing here checks this.** A `RopeTable` newtype tagged `yarn: bool` did,
        /// and was dropped as a second place to state the same fact wrongly;
        /// `rope_for_layer` is the single source and calling it is on the caller. Note what
        /// that trade actually is — the mismatch is no less REPRESENTABLE than before, it
        /// is merely undiagnosed, and the enforcing caller does not exist yet.
        ///
        /// Why this needs saying at all: **no numeric gate catches it at RATIO-4 decode**,
        /// and that is now recorded rather than incidental. Not "at decode" — an earlier
        /// draft of this said that and it is the over-generalization the suite itself
        /// retracted: `ratio128/decode` is NOT in the exempt registry and is still required
        /// to separate.
        /// `tests/v4_compress_kernel.rs`'s `ratio4/decode` cell scores the no-yarn swap
        /// bit-identical to the defect oracle and 8 bf16 codes from clean — under the
        /// `RESOLVABLE` floor of 64 (`4 * E4M3_ULP`, and `E4M3_ULP` is 16 bf16 codes, so 8
        /// is half an e4m3 step). That suite lists the cell in `NO_YARN_BELOW_RESOLUTION`
        /// with its measured separation and asserts it reproduces the defect exactly,
        /// rather than lowering `RESOLVABLE` to admit it, and says outright that it "cannot
        /// see requirement 4 at RATIO-4 decode". `ratio4/prefill` separates at 31,215 and is
        /// what gates the requirement — though note it is asserted only at `>= RESOLVABLE`,
        /// so a 500x collapse in that separation would still pass.
        pub freqs: *const f32,
        /// `[idxs_shape.0, idxs_shape.1]` i32, from [`super::v4_topk_idxs`].
        pub idxs: *const i32,
        /// The `(rows, cols)` the caller actually uploaded. Checked against what this
        /// step requires, because the two disagree silently: the shapes differ between
        /// prefill and decode AND between a short prompt and a long one, and a wrong
        /// `cols` reads whatever follows the buffer as attention indices.
        pub idxs_shape: (usize, usize),
        /// The persistent KV cache: `[window_size, head_dim]` on a ratio-0 layer,
        /// **`[window_size + max_seq_len/ratio, head_dim]` on a compressed one** — the
        /// ring first, then the compressed region, which is exactly the reference's
        /// `self.kv_cache` with `compressor.kv_cache = self.kv_cache[:, win:]` as a VIEW
        /// of its tail rather than a second buffer.
        ///
        /// Decode attends this whole thing, so the two regions must be contiguous and in
        /// that order; the selection's compressed columns are `window_size + block`.
        pub cache: *mut f32,
        /// `[m, dim]`
        pub out: *mut f32,
    }

    /// Which call this is. The reference's discriminant is `start_pos == 0`
    /// (model.py:523); making it a sum type means every place that branches on it is
    /// exhaustive, and the two phases differ in more than one way — the ring write, what
    /// `sparse_attn` reads, and which index SPACE the selection is in.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Step {
        Prefill { seqlen: usize },
        Decode { pos: usize },
    }

    /// Everything `attention` needs that does not vary between calls.
    #[derive(Clone, Copy, Debug)]
    pub struct Dims {
        pub dim: usize,
        pub n_heads: usize,
        pub head_dim: usize,
        pub rope_head_dim: usize,
        pub q_lora_rank: usize,
        pub o_groups: usize,
        pub o_lora_rank: usize,
        pub window: usize,
        pub norm_eps: f32,
    }

    impl Dims {
        /// Derive from the artifact's config, validating every relation the kernels
        /// assume. Parsed once here so `attention` can trust its dimensions.
        ///
        /// `sliding_window` and `rms_norm_eps` come from the config like everything
        /// else. They did NOT until `b5d4083`, which declared them for the first time.
        ///
        /// Worth recording why the gap survived that long: `every_v4_field_is_required` drives off
        /// `V4_BASE` and can only prove that *declared* fields lack a `#[serde(default)]`.
        /// A field never declared at all is invisible to it. Both are in `V4_BASE` now.
        pub fn from_config(cfg: &V4Config) -> Result<Self> {
            // `f64` in the config because JSON numbers are, `f32` in `Dims` because the
            // kernels are. The narrowing ROUNDS — 1e-6 is representable in
            // neither format — but it lands on the same bits as the f32 literal `1e-6`, so
            // there is no double-rounding surprise, and the ~5e-14 absolute error is
            // thirteen orders below the norm it perturbs.
            let (window, norm_eps) = (cfg.sliding_window, cfg.rms_norm_eps as f32);
            let d = Self {
                dim: cfg.hidden,
                n_heads: cfg.n_heads,
                head_dim: cfg.head_dim,
                rope_head_dim: cfg.qk_rope_head_dim,
                q_lora_rank: cfg.q_lora_rank,
                o_groups: cfg.o_groups,
                o_lora_rank: cfg.o_lora_rank,
                window,
                norm_eps,
            };
            d.validate()?;
            Ok(d)
        }

        /// Every relation the kernels assume, checked against the values actually held.
        ///
        /// Called by `from_config` AND by [`attention`]: `Dims` is `Copy` with `pub`
        /// fields, so a struct literal or a later `d.head_dim = 0` skips `from_config`
        /// entirely — and the struct literal is what `tests/v4_attn.rs::dims` does today,
        /// which is the shape a layer loop copies by default (S3 requirement 7). Sealing
        /// the fields stops the literal and not the mutation, and costs every reader an
        /// accessor; checking at the point of use stops both, for ~15 integer compares
        /// against a 4096-wide GEMV.
        pub fn validate(&self) -> Result<()> {
            // `self` under a short name, so the checks below read as they did when they
            // were inlined in `from_config` rather than gaining 40 `self.`.
            let d = self;
            // EVERY extent, not just the three that read as counts. `is_multiple_of`
            // admits zero (`0.is_multiple_of(128)` is true) and so do `0 > 0` and
            // `0.is_multiple_of(2)`, so without this a `head_dim` or `q_lora_rank` of 0
            // passed every check below and surfaced as an opaque "argument guard rejected
            // (1001)" from whichever launcher happened to run first. `rope_head_dim == 0`
            // is the interesting one: it means no RoPE at all, which is a legal-looking
            // config and a completely different model.
            for (v, what) in [
                (d.window, "sliding_window"),
                (d.n_heads, "n_heads"),
                (d.o_groups, "o_groups"),
                (d.dim, "hidden"),
                (d.head_dim, "head_dim"),
                (d.rope_head_dim, "qk_rope_head_dim"),
                (d.q_lora_rank, "q_lora_rank"),
                (d.o_lora_rank, "o_lora_rank"),
            ] {
                if v == 0 {
                    bail!("v4 attention: {what} is zero");
                }
            }
            if d.rope_head_dim > d.head_dim || !d.rope_head_dim.is_multiple_of(2) {
                bail!(
                    "v4 attention: rope_head_dim {} must be even and at most head_dim {}",
                    d.rope_head_dim,
                    d.head_dim
                );
            }
            // `act_quant` asserts `N % block_size == 0` on every row it quantizes: the
            // three `Linear` inputs at 128, and the KV entry's non-RoPE span at 64.
            for (n, what) in [
                (d.dim, "hidden"),
                (d.q_lora_rank, "q_lora_rank"),
                (d.o_groups * d.o_lora_rank, "o_groups*o_lora_rank"),
            ] {
                if !n.is_multiple_of(FP8_BLOCK) {
                    bail!("v4 attention: {what} = {n} is not a multiple of {FP8_BLOCK}");
                }
            }
            // The one extent the sweep above cannot reach, because it is DERIVED. A
            // config with `qk_rope_head_dim == head_dim` — "rotate the whole head", which
            // looks entirely ordinary — passes the rope bound (`512 > 512` is false), is
            // even, and then satisfies the multiple-of-64 test below because
            // `0.is_multiple_of(64)` is TRUE. That is the same `is_multiple_of`-admits-zero
            // property the zero sweep was added for. It reached `launch_v4_act_quant` with
            // `n = 0` and came back as `argument guard rejected (1001)`: late, opaque, and
            // exactly what the sweep exists to prevent.
            if d.head_dim == d.rope_head_dim {
                bail!("v4 attention: head_dim - qk_rope_head_dim is zero");
            }
            if !(d.head_dim - d.rope_head_dim).is_multiple_of(KV_QUANT_BLOCK) {
                bail!(
                    "v4 attention: head_dim - rope_head_dim = {} is not a multiple of {}",
                    d.head_dim - d.rope_head_dim,
                    KV_QUANT_BLOCK
                );
            }
            if !(d.n_heads * d.head_dim).is_multiple_of(d.o_groups) {
                bail!("v4 attention: n_heads*head_dim is not divisible by o_groups");
            }
            Ok(())
        }

    }

    /// `act_quant(kv[..., :-rope_head_dim], **64**, …)` — model.py:512, and NOT the 128
    /// every `Linear` uses.
    ///
    /// **CORRECTED 2026-08-05: the oracle CAN see this number; the max-ULP metric cannot.**
    /// This said "provably cannot see", from the argument that a ue8m0 scale is a power of
    /// two and e4m3 is exactly scale-invariant under those. The argument is sound only
    /// while both blockings keep every value in range, and it breaks at a rounding
    /// boundary. Measured on the GPU at `78796eb` and `e213df1` alike, `ratio4/prefill`:
    ///
    /// ```text
    /// clean:              max=16  differing=5/32768
    /// KvActQuantBlock128: max=16  differing=6/32768
    /// ```
    ///
    /// Same max ULP, one more differing element — so `broken != clean` and the defect is
    /// observable. What is true is narrower: a gate that ranks on max ULP alone cannot
    /// separate it, which is why it sits at `sep=16` under a 64-code floor.
    ///
    /// Worth keeping as a warning about its own shape: a first-principles argument strong
    /// enough to read as settled ("powers of two, therefore exactly invariant") survived
    /// three readers who each had the contradicting numbers in front of them. Where a
    /// comment is about to say something *cannot* be observed, check a run, not a
    /// derivation.
    ///
    /// It is 64 because model.py:512 says 64.
    const KV_QUANT_BLOCK: usize = 64;

    /// Run one V4 attention block.
    ///
    /// # Safety
    /// Every pointer in `w`, `s` and `io` must be a live device allocation of at least
    /// the size documented on its field — `s`'s buffers for `s.rows` rows, which this
    /// checks against the `m` the step implies — and must stay live until the next
    /// [`crate::backend::hip::device_sync`]. `io.freqs` must cover position
    /// `start_pos + m - 1`.
    ///
    /// Four further obligations, none of which a plausible caller satisfies by
    /// accident:
    ///
    /// - **Everything must be pairwise NON-OVERLAPPING.** `memcpy_dtod`'s own contract
    ///   forbids overlap, and every GEMV needs its input and output disjoint. The
    ///   [`Scratch`] docs explain why `qr`/`qrq` are separate buffers, which reads as
    ///   though that were the only separation this needs; it is not.
    /// - **`io.idxs` must name rows that exist in what the step attends.** The kernel
    ///   dereferences `kv + idx * head_dim` for every non-negative entry. The bound is
    ///   `seqlen + n_comp` at prefill and `window + n_comp` at decode — NOT `seqlen` and
    ///   `window`, which is what it was before compressed columns existed: the compressed
    ///   selection legitimately names rows past the window region, which is the whole
    ///   point of `v4compress::compress_offset`. The shape is checked below; the VALUES
    ///   are not and cannot be cheaply, which is the cost of letting the caller own the
    ///   selection buffer.
    /// - **On a compressed layer the compressor must already have run for this same
    ///   `step`, and must have written BOTH destinations.** `attention` only reads the
    ///   compressed rows — `s.kv`'s tail at prefill, `io.cache`'s tail at decode — and
    ///   writes neither. Running it first hands `sparse_attn` uninitialised device memory
    ///   in rows that every later decode step selects BY POSITION and weights with
    ///   `exp(l - max)`. The reference's order is compressor-then-attend, and so is this.
    /// - **`io.x` and `io.out` must hold `s.rows` rows too.** They are the other two
    ///   `[m, dim]` buffers, and `s.rows` is deliberately the ONE number that governs all
    ///   nine: a second capacity field on [`Io`] would be a second thing to get right, and
    ///   the check below would then pass while `io.out` — the buffer the next layer reads
    ///   — still overran. Sized off the same allocation as [`Scratch`]'s and they cannot
    ///   disagree.
    pub unsafe fn attention(
        d: &Dims,
        sel: super::Sel,
        w: &Weights,
        s: &Scratch,
        io: &Io,
        step: Step,
    ) -> Result<()> {
        d.validate()?;
        let (m, pos0) = match step {
            Step::Prefill { seqlen } => (seqlen, 0),
            Step::Decode { pos } => (1, pos),
        };
        if m == 0 {
            bail!("v4 attention: zero query rows");
        }
        ensure!(
            m <= s.rows,
            "v4 attention: {step:?} needs {m} scratch rows, caller allocated {}",
            s.rows
        );
        let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
        let (nhd, gd) = (nh * hd, nh * hd / d.o_groups);
        let gr = d.o_groups * d.o_lora_rank;
        // `win`, `seqlen` and `start_pos` ALL come from this call, not from `sel` — the
        // caller supplies the layer's class and `index_topk`, and this function owns the
        // geometry. `win` was left to the caller for one round and two reviewers
        // independently found the hole: `Dims::window` still drives every ring write
        // (`pos % d.window`), so a `Sel` disagreeing with it produced
        // a selection over the wrong slot space that matched its own `idxs_shape` and
        // passed the guard below — in-bounds reads, no crash, fluent wrong text. It also
        // took `win == 0` out of reach, which `Dims::validate` already refuses but
        // `Sel::shape`'s early return would otherwise have turned into a silent `(0, 0)`
        // that skipped the `index_topk` refusal entirely.
        let sel = super::Sel { win: d.window, seqlen: m, start_pos: pos0, ..sel };
        let want_idxs = sel.shape()?;
        if io.idxs_shape != want_idxs {
            bail!(
                "v4 attention: {step:?} needs a {want_idxs:?} selection, caller uploaded {:?}",
                io.idxs_shape
            );
        }
        let topk = want_idxs.1;

        // -- q -------------------------------------------------------------------------
        // `x` is quantized ONCE and read by both `wq_a` and `wkv`. The reference runs a
        // separate `act_quant` inside each `Linear` (model.py:120/123), but on the same
        // row at the same block size, so the two produce identical bytes.
        // SAFETY: caller's contract; `xq` is `[m, dim]` and `x` is not modified.
        unsafe {
            memcpy_dtod(s.xq.cast(), io.x.cast(), m * d.dim * size_of::<f32>())?;
            launch_v4_act_quant(s.xq, m, d.dim, d.dim, FP8_BLOCK)?;
            let (q_lora, dim) = (d.q_lora_rank, d.dim);
            launch_v4_gemv_fp8(s.xq, w.wq_a.w, w.wq_a.scale, m, q_lora, dim, FP8_BLOCK, 1, s.qr)?;
            launch_v4_rmsnorm(s.qr, w.q_norm, m, q_lora, d.norm_eps)?;
            memcpy_dtod(s.qrq.cast(), s.qr.cast(), m * q_lora * size_of::<f32>())?;
            launch_v4_act_quant(s.qrq, m, q_lora, q_lora, FP8_BLOCK)?;
            launch_v4_gemv_fp8(s.qrq, w.wq_b.w, w.wq_b.scale, m, nhd, q_lora, FP8_BLOCK, 1, s.q)?;
            // QK-norm BEFORE RoPE (model.py:504 then :505). Read off the reference, not
            // inferred from a green test -- which is the part that matters and is unchanged.
            //
            // CORRECTED 2026-08-05: this said "the oracle cannot see this order". It can.
            // The argument was that RoPE rotates adjacent pairs so it preserves `mean(q^2)`
            // and a scalar commutes with a rotation -- exact, and about exact arithmetic.
            // `Oracle::qk_norm` computes that statistic in BF16, faithfully (it is bf16 in
            // the reference), so `rs` is quantized to ~0.4% steps and the two orders land on
            // different steps. Measured on a compressed layer: `Defect::QkNormAfterRope`
            // moves `.q` on 1287/24576 elements at rel 7.4e-3, and three goldens downstream
            // (`tests/v4_attn.rs::expect_moves`). What IS true is narrower and is why the
            // defect stays out of both separation sweeps: its distance measures the bf16
            // ROUNDING scale rather than a defect's reach, so ranking it beside the others
            // would misstate the sweep's resolution. (It is not that the sweep cannot see it
            // -- the max-ULP metric prints 29,131 for it against a floor of 0, because that
            // metric is a bf16-CODE distance and is blind at zero. An earlier wording of
            // this line said "cannot separate it from the floor" and was wrong about the
            // instrument.) `tests/v4_oracle.rs::qk_norm_order_is_a_rounding_difference_not_an_arithmetic_one`
            // bounds it by what dropping bf16 rounding entirely costs. Third time in this
            // port that an exact-arithmetic equivalence was taken for a bitwise one; see
            // `KV_QUANT_BLOCK` above for the second.
            launch_v4_qk_norm(s.q, m * nh, hd, d.norm_eps)?;
            launch_v4_rope(s.q, io.freqs, m * nh, hd, rd, pos0, nh, false)?;

            // -- kv --------------------------------------------------------------------
            launch_v4_gemv_fp8(s.xq, w.wkv.w, w.wkv.scale, m, hd, dim, FP8_BLOCK, 1, s.kv)?;
            launch_v4_rmsnorm(s.kv, w.kv_norm, m, hd, d.norm_eps)?;
            launch_v4_rope(s.kv, io.freqs, m, hd, rd, pos0, 1, false)?;
            // PARTIAL: the RoPE'd tail keeps bf16 precision. Quantizing the whole entry
            // is the llama.cpp failure mode v4-flash-port.md §0.2 names -- it corrupts
            // the positional dims and produces noise without a crash.
            launch_v4_act_quant(s.kv, m, hd, hd - rd, KV_QUANT_BLOCK)?;
        }

        // -- cache and attention ---------------------------------------------------------
        // What `sparse_attn` reads differs by phase, and so does the index space:
        // prefill attends the prompt's own KV by ABSOLUTE POSITION, decode attends the
        // ring by SLOT. See `v4_topk_idxs`.
        //
        // A compressed layer needs no extra arm here, and that is the whole point of the
        // buffer layout: at prefill the compressor has already written its blocks to
        // `s.kv`'s TAIL, so `s.kv` IS `torch.cat([kv, kv_compress])`; at
        // decode the compressed region is `io.cache`'s tail behind the ring, which is the
        // reference's own `compressor.kv_cache = self.kv_cache[:, win:]` view. Both are
        // one contiguous buffer and `sparse_attn` indexes straight into it.
        let row = hd * size_of::<f32>();
        // SAFETY: caller's contract; every copy below stays inside `cache[0, window)`.
        let kv_src: *const f32 = unsafe {
            match step {
                Step::Prefill { seqlen } if seqlen <= d.window => {
                    memcpy_dtod(io.cache.cast(), s.kv.cast(), seqlen * row)?;
                    s.kv
                }
                Step::Prefill { seqlen } => {
                    // Slot `t % window` holds position `t`, for the last `window`
                    // positions. Seeding with the FIRST window instead is right exactly
                    // when the prompt fits, which is why a short fixture cannot see it.
                    let cut = seqlen % d.window;
                    memcpy_dtod(
                        io.cache.add(cut * hd).cast(),
                        s.kv.add((seqlen - d.window) * hd).cast(),
                        (d.window - cut) * row,
                    )?;
                    if cut > 0 {
                        memcpy_dtod(
                            io.cache.cast(),
                            s.kv.add((seqlen - cut) * hd).cast(),
                            cut * row,
                        )?;
                    }
                    s.kv
                }
                Step::Decode { pos } => {
                    memcpy_dtod(io.cache.add((pos % d.window) * hd).cast(), s.kv.cast(), row)?;
                    io.cache
                }
            }
        };

        // `attention` does NOT persist the prompt's compressed blocks; it only reads them.
        // `Compressor.forward` does two writes — `self.kv_cache[:, :seqlen//ratio]` AND the
        // return value `Attention.forward` concatenates — and doing only the second here
        // meant reaching into `s.kv`'s tail for rows this function never wrote. The
        // obligation is on `v4compress::compress`'s doc, where its reader is.
        //
        // `Finish` has ONE `out`, so the second destination is a device COPY after a single
        // `compress` call, never a second call: `compress` read-modify-writes
        // `kv_state`/`score_state` before it decides whether to emit, so calling it twice
        // deposits the same rows into the pooling window twice and slides it twice.

        // SAFETY: caller's contract.
        unsafe {
            launch_v4_sparse_attn(
                s.q,
                kv_src,
                w.attn_sink,
                io.idxs,
                m,
                nh,
                hd,
                topk,
                (hd as f32).powf(-0.5),
                s.o,
            )?;

            // -- output ----------------------------------------------------------------
            launch_v4_rope(s.o, io.freqs, m * nh, hd, rd, pos0, nh, true)?;
            // `o.view(b, s, o_groups, -1)` needs no gather: group `g` is the contiguous
            // run of heads `[g * n_heads/o_groups, ...)`, all `head_dim` dims of each.
            // `groups = o_groups`: `o` is `m` rows of `o_groups` contiguous `gd`-wide
            // head runs, and output row `j` takes run `j / o_lora_rank`.
            launch_v4_gemv_fp8(
                s.o, w.wo_a.w, w.wo_a.scale, m, gr, gd, FP8_BLOCK, d.o_groups, s.y,
            )?;
            launch_v4_act_quant(s.y, m, gr, gr, FP8_BLOCK)?;
            launch_v4_gemv_fp8(s.y, w.wo_b.w, w.wo_b.scale, m, d.dim, gr, FP8_BLOCK, 1, io.out)?;
        }
        Ok(())
    }
}

/// The guards `attention` gained in S3, exercised — requirements 6 and 7.
///
/// **Every guard fires before any device call**, which is what makes this testable without a GPU
/// and is also the property being pinned: `d.validate()` and the `s.rows` check are the
/// first two statements, so every pointer below can be dangling and never be read. If a
/// guard stopped firing the test would not quietly pass — it would run on into
/// `memcpy_dtod` with a null destination.
///
/// **This is the rejecting half only, and deliberately so.** The accepting half is the
/// whole of `tests/v4_attn.rs`, which drives `attention` to completion against the oracle
/// with `rows == max_m` and valid dims; if either guard were inverted, all five of its
/// comparisons would fail to run at all. What a rejection-only test cannot do by itself is
/// tell "the guard fired" from "something failed" — so each case below asserts on the
/// MESSAGE, and the two cases are chosen to cross: the `Dims` case has ample rows, and the
/// rows case has valid dims. Neither can be passing for the other's reason.
///
/// **SEEN RED, 2026-08-05, before being trusted green — the two requirement-6/7 tests
/// specifically.** The window-vs-ring test carries its own mutation record in-line. With
/// `d.validate()?` deleted and
/// the `ensure!` weakened to `m <= usize::MAX`, both tests failed — and failed on the
/// backstop's message (`needs a (4, 4) selection, caller uploaded (0, 0)` and
/// `needs a (1, 128) selection, …`), which is the evidence that a disabled guard lands on
/// the selection bail rather than on a launcher. This repo has shipped a tautological
/// anti-vacuity assertion twice and reported it working both times; the mutation is why
/// this one is not a third.
#[cfg(all(test, feature = "rocm"))]
mod v4_guard_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::v4::{Dims, Fp8W, Io, Scratch, Step, Weights, attention};
    use super::Sel;
    use crate::v4compress::LayerKind;

    /// The shipped V4 geometry, so a rejection below is about the guard and not about a
    /// shape the kernels would refuse anyway.
    fn dims() -> Dims {
        Dims {
            dim: 4096,
            n_heads: 64,
            head_dim: 512,
            rope_head_dim: 64,
            q_lora_rank: 1024,
            o_groups: 8,
            o_lora_rank: 1024,
            window: 128,
            norm_eps: 1e-6,
        }
    }

    /// A ratio-0 layer's selection descriptor. The guards under test precede anything
    /// that reads `kind`, so `Plain` keeps these cases about the guards.
    fn plain() -> Sel {
        Sel { win: 128, kind: LayerKind::Plain, index_topk: 0, seqlen: 1, start_pos: 0 }
    }

    /// Dangling, and never dereferenced: both guards return before the first launch.
    fn parts(rows: usize) -> (Weights, Scratch, Io) {
        let n = std::ptr::null_mut::<f32>();
        let c = std::ptr::null::<f32>();
        let f = Fp8W { w: std::ptr::null(), scale: c };
        let w = Weights {
            wq_a: f,
            q_norm: c,
            wq_b: f,
            wkv: f,
            kv_norm: c,
            attn_sink: c,
            wo_a: f,
            wo_b: f,
        };
        let s = Scratch { rows, xq: n, qr: n, qrq: n, q: n, kv: n, o: n, y: n };
        let io = Io {
            x: c,
            freqs: c,
            idxs: std::ptr::null(),
            // DELIBERATELY a shape no step wants, so the pre-existing selection guard is a
            // BACKSTOP: if either guard under test stopped firing, control reaches that
            // bail instead of a launcher, and the assertions below fail on the message
            // rather than the process dying on a null device pointer. That is what lets
            // these two tests be mutated to red safely — which they were, before they were
            // trusted green (see the module doc).
            idxs_shape: (0, 0),
            cache: n,
            out: n,
        };
        (w, s, io)
    }

    #[test]
    fn a_decode_sized_scratch_refuses_a_prefill() {
        let d = dims();
        let (w, s, io) = parts(1);
        // SAFETY: every pointer is null and none is read — the `rows` check precedes the
        // first launch, which is the property this asserts.
        let e = unsafe { attention(&d, plain(), &w, &s, &io, Step::Prefill { seqlen: 4 }) }
            .expect_err("a 4-row prefill into a 1-row scratch must be refused");
        let msg = format!("{e}");
        assert!(msg.contains("scratch rows"), "wrong rejection: {msg}");

        // ...and the same call with room does NOT fail for this reason. It still fails —
        // the pointers are null — but the message proves the guard is a bound and not a
        // constant `false`, which is the shape S2 shipped twice.
        let (w, s, io) = parts(4);
        let msg = match unsafe { attention(&d, plain(), &w, &s, &io, Step::Prefill { seqlen: 4 }) } {
            Ok(()) => String::new(),
            Err(e) => format!("{e}"),
        };
        assert!(!msg.contains("scratch rows"), "the guard rejects a scratch that fits: {msg}");
    }

    /// Requirement 12 — the two index spaces — at the one place a mismatch is silent.
    /// The invariant two reviewers found missing, independently.
    ///
    /// `Sel::win` must not be able to disagree with `Dims::window`, which drives every ring
    /// write (`pos % d.window`). `attention` overrides it for exactly
    /// that reason; SEEN RED with the override removed, where the call accepts a
    /// 64-column selection over a ring that rotates modulo 128.
    #[test]
    fn the_selection_window_cannot_disagree_with_the_ring_window() {
        let d = dims(); // window 128
        let (w, s, io) = parts(64);
        let lying = Sel { win: 64, ..plain() };
        // SAFETY: the shape guard precedes every launch.
        let e = unsafe { attention(&d, lying, &w, &s, &io, Step::Decode { pos: 200 }) }
            .expect_err("the (0, 0) idxs_shape in `parts` must be refused");
        assert!(
            format!("{e}").contains("(1, 128)"),
            "attention took the selection width from the caller, not from Dims: {e}"
        );
    }

    #[test]
    fn attention_revalidates_dims_that_never_saw_from_config() {
        // A `Dims` mutated AFTER construction — exactly what `from_config` cannot prevent
        // and what requirement 7 is about. Ample rows, so the other guard is not in play.
        let mut d = dims();
        d.head_dim = 0;
        let (w, s, io) = parts(64);
        // SAFETY: as above.
        let e = unsafe { attention(&d, plain(), &w, &s, &io, Step::Decode { pos: 7 }) }
            .expect_err("a zero head_dim must be refused before any launch");
        let msg = format!("{e}");
        assert!(msg.contains("head_dim is zero"), "wrong rejection: {msg}");

        // The derived extent, which no config field holds and the zero sweep cannot reach:
        // `head_dim == rope_head_dim` makes the KV span zero and `0.is_multiple_of(64)` is
        // true. It reached a launcher as guard code 1001 before `from_config` grew this.
        let mut d = dims();
        d.rope_head_dim = d.head_dim;
        let e = unsafe { attention(&d, plain(), &w, &s, &io, Step::Decode { pos: 7 }) }
            .expect_err("head_dim == rope_head_dim must be refused");
        assert!(format!("{e}").contains("is zero"), "wrong rejection: {e}");
    }
}
