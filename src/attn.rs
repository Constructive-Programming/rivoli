//! Attention row-selection modes. The MLA absorb + flash-attend core (gpu.rs) is
//! row-set-agnostic; only which cached tokens a step attends over differs by
//! [`AttnMode`]. The DSA/MISA lightning-indexer selection itself runs on device
//! (gpu.rs `dsa_select_layer` + indexer.hip); this module holds the mode enum and
//! the position-based StreamingLLM row set.
//!
//! It ALSO holds the whole DeepSeek-V4-Flash attention block, in `mod v4` and the two
//! free functions above it — see the banner below `streaming_rows`. V4 is MQA and shares
//! none of the MLA machinery, so nothing before that banner applies to it.

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
// `v4_window_topk` is written against `model.py`, not against
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

/// `model.py::get_window_topk_idxs` — which cache slots each query row may attend.
/// Appends `rows * cols` entries to `out` and returns `(rows, cols)`; `-1` masks a slot.
///
/// **The two phases index DIFFERENT spaces, and nothing in the types says so.** At
/// prefill (`start_pos == 0`) `sparse_attn` reads the prompt's own `kv`, so these are
/// absolute positions `0..seqlen`; at decode it reads the ring, so they are ring SLOTS
/// `0..window_size`. Feeding one to the other is silent — it attends to real vectors at
/// the wrong positions.
///
/// `cols` is `min(seqlen, win)` at prefill and `win` at decode, following the
/// reference's `torch.arange(min(seqlen, window_size))`. It is returned rather than
/// derived by the caller because a caller that assumed `win` would read `win - seqlen`
/// columns of whatever followed the buffer.
pub fn v4_window_topk(
    win: usize,
    seqlen: usize,
    start_pos: usize,
    out: &mut Vec<i32>,
) -> (usize, usize) {
    // `win == 0` is not a model state — `Dims::from_config` refuses it — but this is a
    // `pub` free function and the oracle's twin uses `win.saturating_sub(1)` throughout.
    // Wrapping where the instrument saturates would make the two disagree at a shape the
    // comparison could then never be run at, so the floor is matched rather than argued
    // to be unreachable.
    if win == 0 {
        return (0, 0);
    }
    if start_pos == 0 {
        // Causal window per query row: row `t` covers positions `t-win+1 ..= t`, clamped
        // at 0, with the acausal tail masked.
        let cols = seqlen.min(win);
        for t in 0..seqlen {
            let base = t.saturating_sub(win - 1);
            out.extend((0..cols).map(|j| if base + j > t { -1 } else { (base + j) as i32 }));
        }
        return (seqlen, cols);
    }
    // One decode row over the ring. Past the first wrap the ring is full and the slots
    // are listed oldest-first, which is `start_pos % win` rotated to the end; before it,
    // only slots `0..=start_pos` hold anything and the rest are masked.
    if start_pos >= win - 1 {
        let sp = start_pos % win;
        out.extend((sp + 1..win).chain(0..=sp).map(|i| i as i32));
    } else {
        out.extend((0..=start_pos).map(|i| i as i32));
        out.resize(out.len() + win - start_pos - 1, -1);
    }
    (1, win)
}

#[cfg(test)]
mod v4_tests {
    use super::*;

    #[test]
    fn window_topk_prefill_is_causal_and_masks_nothing_reachable() {
        let mut v = Vec::new();
        // Prompt shorter than the window: `cols` shrinks to the prompt, and row `t` sees
        // exactly `0..=t`. A caller that assumed `win` columns would read past the row.
        let (rows, cols) = v4_window_topk(8, 3, 0, &mut v);
        assert_eq!((rows, cols), (3, 3));
        assert_eq!(v, vec![0, -1, -1, 0, 1, -1, 0, 1, 2]);

        // Prompt past the window: the oldest position drops out of each row.
        v.clear();
        let (rows, cols) = v4_window_topk(2, 4, 0, &mut v);
        assert_eq!((rows, cols), (4, 2));
        assert_eq!(v, vec![0, -1, 0, 1, 1, 2, 2, 3]);
    }

    #[test]
    fn window_topk_decode_rotates_only_after_the_ring_fills() {
        let mut v = Vec::new();
        // Before the wrap: slots 0..=start_pos, then masked. Ascending, NOT rotated —
        // rotating here would name slots the prefill never wrote.
        let (rows, cols) = v4_window_topk(4, 1, 2, &mut v);
        assert_eq!((rows, cols), (1, 4));
        assert_eq!(v, vec![0, 1, 2, -1]);

        // At exactly `win - 1` the ring is full and the rotation starts; `start_pos = 3`
        // wrote slot 3, so the oldest slot is 0 and the list is already in order.
        v.clear();
        v4_window_topk(4, 1, 3, &mut v);
        assert_eq!(v, vec![0, 1, 2, 3]);

        // Past the wrap: slot `start_pos % win` holds the newest token and must come
        // LAST. At start_pos=5, win=4 the newest is slot 1, so the order is 2,3,0,1.
        v.clear();
        v4_window_topk(4, 1, 5, &mut v);
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

/// The V4 attention block on device — `Attention.forward` (model.py:490-548) for a
/// `compress_ratio == 0` layer.
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
    use anyhow::{Result, bail};

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

    /// Per-call scratch, device-resident, sized for `m` query rows.
    ///
    /// `qr` and `qrq` are separate buffers holding the same values, and that is not
    /// waste: `qr` is `q_norm(wq_a(x))` and a ratio-4 layer's `Indexer` consumes it
    /// AFTER the q path is done with it (model.py:509/519), so quantizing in place here
    /// would hand S2c a destroyed input. `xq` is separate for the same reason on `x`,
    /// which the compressor and the indexer both read.
    #[derive(Clone, Copy)]
    pub struct Scratch {
        /// `[m, dim]` — the activation-quantized copy of `x`.
        pub xq: *mut f32,
        /// `[m, q_lora_rank]`
        pub qr: *mut f32,
        /// `[m, q_lora_rank]`
        pub qrq: *mut f32,
        /// `[m, n_heads * head_dim]`
        pub q: *mut f32,
        /// `[m, head_dim]`
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
        /// Interleaved `(cos, sin)`, from [`super::v4_rope_table_ratio0`].
        pub freqs: *const f32,
        /// `[idxs_shape.0, idxs_shape.1]` i32, from [`super::v4_window_topk`].
        pub idxs: *const i32,
        /// The `(rows, cols)` the caller actually uploaded. Checked against what this
        /// step requires, because the two disagree silently: the shapes differ between
        /// prefill and decode AND between a short prompt and a long one, and a wrong
        /// `cols` reads whatever follows the buffer as attention indices.
        pub idxs_shape: (usize, usize),
        /// `[window_size, head_dim]` — the sliding-window ring, persistent across steps.
        pub ring: *mut f32,
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
            // kernels are. Narrowing is exact for the shipped 1e-6 and for any eps a
            // config would carry.
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
            // EVERY extent, not just the three that read as counts. `is_multiple_of`
            // admits zero (`0.is_multiple_of(128)` is true) and so do `0 > 0` and
            // `0.is_multiple_of(2)`, so without this a `head_dim` or `q_lora_rank` of 0
            // passed every check below and surfaced as an opaque "argument guard rejected
            // (1001)" from whichever launcher happened to run first. `rope_head_dim == 0`
            // is the interesting one: it means no RoPE at all, which is a legal-looking
            // config and a completely different model.
            for (v, what) in [
                (window, "sliding_window"),
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
            Ok(d)
        }
    }

    /// `act_quant(kv[..., :-rope_head_dim], **64**, …)` — model.py:512, and NOT the 128
    /// every `Linear` uses.
    ///
    /// **The oracle provably cannot see this number.** A ue8m0 scale is a power of two
    /// and e4m3 is exactly scale-invariant under those, so re-blocking changes no value
    /// until a block spans ~2^13 of dynamic range, which activations do not
    /// (`tests/v4_oracle.rs::act_quant_block_size_is_almost_invisible_under_ue8m0_scales`
    /// measures this and demotes `Defect::KvActQuantBlock128` out of the defect matrix
    /// for it). It is 64 because model.py:512 says 64.
    const KV_QUANT_BLOCK: usize = 64;

    /// Run one V4 attention block.
    ///
    /// # Safety
    /// Every pointer in `w`, `s` and `io` must be a live device allocation of at least
    /// the size documented on its field, for the `m` implied by `step` (`seqlen` at
    /// prefill, 1 at decode), and must stay live until the next
    /// [`crate::backend::hip::device_sync`]. `io.freqs` must cover position
    /// `start_pos + m - 1`.
    ///
    /// Two further obligations, neither of which a plausible caller satisfies by
    /// accident:
    ///
    /// - **Everything must be pairwise NON-OVERLAPPING.** `memcpy_dtod`'s own contract
    ///   forbids overlap, and every GEMV needs its input and output disjoint. The
    ///   [`Scratch`] docs explain why `qr`/`qrq` are separate buffers, which reads as
    ///   though that were the only separation this needs; it is not.
    /// - **`io.idxs` must name rows that exist in what the step attends.** The kernel
    ///   dereferences `kv + idx * head_dim` for every non-negative entry, so at prefill
    ///   every entry must be `< seqlen` and at decode `< window`. The shape is checked
    ///   below; the VALUES are not and cannot be cheaply, which is the cost of letting
    ///   the caller own the selection buffer.
    pub unsafe fn attention(
        d: &Dims,
        w: &Weights,
        s: &Scratch,
        io: &Io,
        step: Step,
    ) -> Result<()> {
        let (m, pos0) = match step {
            Step::Prefill { seqlen } => (seqlen, 0),
            Step::Decode { pos } => (1, pos),
        };
        if m == 0 {
            bail!("v4 attention: zero query rows");
        }
        let (nh, hd, rd) = (d.n_heads, d.head_dim, d.rope_head_dim);
        let (nhd, gd) = (nh * hd, nh * hd / d.o_groups);
        let gr = d.o_groups * d.o_lora_rank;
        // The selection's shape is a function of the step, so it is derived here and
        // checked against what the caller uploaded rather than trusted.
        let want_idxs = match step {
            Step::Prefill { seqlen } => (seqlen, seqlen.min(d.window)),
            Step::Decode { .. } => (1, d.window),
        };
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
            // QK-norm BEFORE RoPE (model.py:504 then :505). The oracle cannot see this
            // order -- RoPE rotates adjacent pairs, so it preserves `mean(q^2)`, and a
            // scalar commutes with a rotation. Read off the reference, not inferred from
            // a green test.
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
        // ring by SLOT. See `v4_window_topk`.
        let row = hd * size_of::<f32>();
        // SAFETY: caller's contract; every copy below stays inside `ring[0, window)`.
        let kv_src: *const f32 = unsafe {
            match step {
                Step::Prefill { seqlen } if seqlen <= d.window => {
                    memcpy_dtod(io.ring.cast(), s.kv.cast(), seqlen * row)?;
                    s.kv
                }
                Step::Prefill { seqlen } => {
                    // Slot `t % window` holds position `t`, for the last `window`
                    // positions. Seeding with the FIRST window instead is right exactly
                    // when the prompt fits, which is why a short fixture cannot see it.
                    let cut = seqlen % d.window;
                    memcpy_dtod(
                        io.ring.add(cut * hd).cast(),
                        s.kv.add((seqlen - d.window) * hd).cast(),
                        (d.window - cut) * row,
                    )?;
                    if cut > 0 {
                        memcpy_dtod(
                            io.ring.cast(),
                            s.kv.add((seqlen - cut) * hd).cast(),
                            cut * row,
                        )?;
                    }
                    s.kv
                }
                Step::Decode { pos } => {
                    memcpy_dtod(io.ring.add((pos % d.window) * hd).cast(), s.kv.cast(), row)?;
                    io.ring
                }
            }
        };

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
