//! The activation family of the HIP ABI wall's primitive half: the gated combines, the
//! activation quantizers, the normalizations and RoPE — split out of `hip_linalg.rs`
//! 2026-11-18 by kernel family, matching `src/launch/` (the Rust launchers' host halves),
//! so a Phase 1 lane's edits stay inside one invocation file and below the 800-line cap.
//! A move: every declaration below is byte-identical to the one it replaced, and `hip.rs`
//! re-exports this module, so `rivoli_backend::hip::launch_swiglu` resolves as before.
//!
//! Read `hip.rs` first for the DSL, the descriptor structs and the return-code check.

use crate::hip::{ensure_hip_status, launchers};
use crate::launch::{
    Rows, rivoli_act_quant_f4_rotated, rivoli_act_quant_f8, rivoli_act_quant_f8_prefix,
    rivoli_kv_scatter, rivoli_l2_clip, rivoli_layernorm, rivoli_log_scaled_l2_clip,
    rivoli_logit_softcap, rivoli_message_gate, rivoli_mxfp8_dequant, rivoli_nvfp4_dequant,
    rivoli_qk_norm, rivoli_rel_bias_gather, rivoli_rmsnorm_batch, rivoli_rmsnorm_centered_single,
    rivoli_rmsnorm_single, rivoli_rmsnorm_weightless_batch, rivoli_rope_adjacent,
    rivoli_rope_interleave, rivoli_rope_split_half, rivoli_scaled_l2_clip, rivoli_sconv,
    rivoli_sigmoid_gate, rivoli_sink_attend, rivoli_situ_glu_f32, rivoli_swiglu,
    rivoli_swiglu_clamped_bf16,
};
use anyhow::Result;
use rivoli_abi::kernarg::{
    ActQuantPrefix, F4Rotate, Gate, GateClamped, GateSitu, InPlaceBinary, KvScatter, L2Clip,
    LayerNorm, LogScaledL2Clip, MessageGate, Mxfp8Dequant, Nvfp4Dequant, QkNorm, RelBiasGather,
    RmsNormRows, Rope, RopeAdjacent, RowNorm, ScaledL2Clip, Sconv, SinkAttend, Softcap,
    WeightlessNorm,
};
use std::ffi::c_void;

// Doc links only — see the same block in `hip_linalg.rs` for why these are imports.
#[allow(unused_imports)]
use crate::hip::device_sync;

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — the declarations below, and nothing else in this file.
// The argument is `hip_linalg.rs`'s marker (every clone in a `launchers!` invocation is a
// parameter-list prefix run between two declarations, never logic; the mirroring IS the
// contract), and it is not restated. This file exists because the 2026-11-18 family split
// made the primitive half two invocation files, and a marker pair cannot span files — the
// same reason `hip_blocks.rs` and `hip_attn.rs` carry their own. MEASURED 2026-11-18 with
// both markers deleted: 6 clones touch this file (5 inside it, 1 against `hip_linalg.rs`'s
// `gemv_f32` row), all parameter-list prefix runs — so this marker is earned, not inherited.
//
launchers! {
    // ── activations and gates ───────────────────────────────────────────────────────────────

    /// SwiGLU combine `h = silu(g)·u` (GLM's dense fp8 MLP, and V4's shared-expert chain;
    /// safe in place, `h` may alias `g`).
    ///
    /// `stream` is trailing and null is the null stream, per the `mla.hip` (tag
    /// `archive/hip-kernels`) V4-launcher contract: V4's shared-expert chain is stream-ordered as a SET (§M7), and GLM's one
    /// call site passes null — unchanged to the byte.
    ///
    /// # Safety
    /// Device pointers (`g`, `u`, `h` each `n` f32) live until the next [`device_sync`];
    /// `stream` is null or a live stream ordering every producer of `g`/`u`.
    launch_swiglu => rivoli_swiglu(Gate { g, u, h, n: n as i32 }, stream), "swiglu" (
        g: *const f32,
        u: *const f32,
        n: usize,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's per-head L2 clip `y = x·min(threshold/(‖x‖+1e-6), 1)` (the q/k/rel norm clips).
    /// No C twin — oracle-validated against `rivoli_oracles::inkling` (owner direction 2026-11-24).
    /// BATCHED: one block per row, `rows` rows of `n` — `x`/`y` are `rows·n` f32. Safe in place
    /// (`y` may alias `x`); `threshold` positive and finite (1006).
    ///
    /// # Safety
    /// Device pointers (`x`, `y` each `rows·n` f32) live until the next [`device_sync`]; `stream`
    /// is null or a live stream ordering every producer of `x`.
    launch_l2_clip => rivoli_l2_clip(Rows(rows as i32), L2Clip { x, y, n: n as i32, threshold }, stream), "l2_clip" (
        x: *const f32,
        rows: usize,
        n: usize,
        threshold: f32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's sliding-layer q/k norm: a learned per-head `scale` first, then the L2 clip.
    /// No C twin — oracle-validated (owner direction 2026-11-24). BATCHED: one block per row,
    /// `x`/`scale`/`y` each `rows·n` f32 (row `r` reads `scale + r·n`). Safe in place (`y` may
    /// alias `x`); `threshold` positive and finite (1006).
    ///
    /// # Safety
    /// Device pointers (`x`, `scale`, `y` each `rows·n` f32) live until the next [`device_sync`];
    /// `stream` is null or a live stream ordering every producer of `x`/`scale`.
    launch_scaled_l2_clip => rivoli_scaled_l2_clip(Rows(rows as i32), ScaledL2Clip { clip: L2Clip { x, y, n: n as i32, threshold }, scale }, stream), "scaled_l2_clip" (
        x: *const f32,
        scale: *const f32,
        rows: usize,
        n: usize,
        threshold: f32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's full-layer q/k norm: the scaled clip then the position-dependent LogScale `tau`
    /// (T11: sliding layers omit it). No C twin — oracle-validated. BATCHED like
    /// [`launch_scaled_l2_clip`]. Safe in place; `threshold` AND `tau` positive and finite (1006).
    ///
    /// # Safety
    /// Device pointers (`x`, `scale`, `y` each `rows·n` f32) live until the next [`device_sync`];
    /// `stream` is null or a live stream ordering every producer of `x`/`scale`.
    launch_log_scaled_l2_clip => rivoli_log_scaled_l2_clip(Rows(rows as i32), LogScaledL2Clip { scaled: ScaledL2Clip { clip: L2Clip { x, y, n: n as i32, threshold }, scale }, tau }, stream), "log_scaled_l2_clip" (
        x: *const f32,
        scale: *const f32,
        rows: usize,
        n: usize,
        threshold: f32,
        tau: f32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's MXFP8 expert decode: one e4m3 value per byte, one power-of-two scale per 32
    /// values along the input dim, ROW-WISE `[1,32]` (T1: the scale is a raw U8 exponent, not
    /// e4m3). No C twin — oracle-validated. `cols` a positive multiple of 32 that DIVIDES `n`
    /// (1001 otherwise); `n` is carried 64-bit into the kernarg for an expert layer past `i32`.
    ///
    /// # Safety
    /// Device pointers (`weight` `n` bytes, `scale` `n/32` bytes, `out` `n` f32) live until the
    /// next [`device_sync`]; `stream` is null or a live stream ordering every producer.
    launch_mxfp8_dequant => rivoli_mxfp8_dequant(Mxfp8Dequant { weight, scale, out, n: n as u64, cols: cols as i32 }, stream), "mxfp8_dequant" (
        weight: *const u8,
        scale: *const u8,
        n: usize,
        cols: usize,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's NVFP4 expert decode: two E2M1 codes per byte (low nibble even), one e4m3 scale
    /// per 16 values along the input dim, and a per-expert factor `amax[e]/denom`. No C twin —
    /// oracle-validated. `cols_packed` a positive multiple of 8 (1001), `n` a whole number of
    /// `rows·cols_packed·2` expert planes (1001 — `row / rows` is the expert index), `denom`
    /// positive finite (1006); `n` is carried 64-bit into the kernarg.
    ///
    /// # Safety
    /// Device pointers (`weight` `n/2` bytes, `scale` `n/16` bytes, `amax` `E` f32, `out` `n` f32)
    /// live until the next [`device_sync`]; `stream` is null or a live stream ordering every producer.
    launch_nvfp4_dequant => rivoli_nvfp4_dequant(Nvfp4Dequant { weight, scale, amax, out, denom, n: n as u64, rows: rows as i32, cols_packed: cols_packed as i32 }, stream), "nvfp4_dequant" (
        weight: *const u8,
        scale: *const u8,
        amax: *const f32,
        denom: f32,
        n: usize,
        rows: usize,
        cols_packed: usize,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's static short-convolution decode step, state included: a depthwise causal k-tap
    /// cross-correlation of the window `[state (past k−1), current]` plus the residual (the current
    /// token), divided by `divisor = √(k+1)` (T15); advances `state` in place. No C twin —
    /// oracle-validated. `channels > 0`, `k > 0` (1001), `divisor` positive finite (1006).
    ///
    /// # Safety
    /// Device pointers (`weight` `channels·k` f32, `state` `channels·(k−1)` f32, `current`/`out`
    /// `channels` f32) live until the next [`device_sync`]; `stream` is null or a live stream
    /// ordering every producer.
    launch_sconv => rivoli_sconv(Sconv { weight, state, current, out, channels: channels as i32, k: k as i32, divisor }, stream), "sconv" (
        weight: *const f32,
        state: *mut f32,
        current: *const f32,
        channels: usize,
        k: usize,
        divisor: f32,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's single-query attention decode step with a sink: the softmax joins one extra sink
    /// logit then drops its column (T10), so the real weights sum to < 1. No C twin —
    /// oracle-validated. BATCHED over `heads` query heads in one launch (one block each), with
    /// the per-head sink logits already on the device (`sink` is `heads` f32) so the caller
    /// needs no sync-and-copy per layer.
    ///
    /// `n`, `d`, `heads`, `kv_heads` > 0 with `heads % kv_heads == 0` (the GQA map), `kv_stride >=
    /// n` (a head's window inside its stride) → 1001; `scaling` positive finite → 1006; the LDS
    /// total `(n + ROW_BLOCK)·4` within the gfx1151 budget → 1001.
    ///
    /// # Safety
    /// Device pointers (`q`/`out` `heads·d` f32, `bias` `heads·n` f32, `sink` `heads` f32,
    /// `k`/`v` `kv_heads·kv_stride·d` f32) live until the next [`device_sync`]; `stream` is null
    /// or a live stream ordering every producer.
    launch_sink_attend => rivoli_sink_attend(SinkAttend { sink, bias, out, q, k, v, scaling, n: n as i32, d: d as i32, heads: heads as i32, kv_heads: kv_heads as i32, kv_stride: kv_stride as i32 }, stream), "sink_attend" (
        q: *const f32,
        k: *const f32,
        v: *const f32,
        bias: *const f32,
        sink: *const f32,
        scaling: f32,
        n: usize,
        d: usize,
        heads: usize,
        kv_heads: usize,
        kv_stride: usize,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's loop MessageGate pointwise combine: `(1−tanh(forget_logit))·hidden +
    /// tanh(gain_logit)·message`, fp32. No C twin — oracle-validated. The down/forget/gain
    /// projections are the engine's GEMVs; `n > 0` (1001).
    ///
    /// # Safety
    /// Device pointers (the four inputs and `out`, each `n` f32) live until the next [`device_sync`];
    /// `stream` is null or a live stream ordering every producer.
    launch_message_gate => rivoli_message_gate(MessageGate { forget_logit, gain_logit, hidden, message, out, n: n as i32 }, stream), "message_gate" (
        forget_logit: *const f32,
        gain_logit: *const f32,
        hidden: *const f32,
        message: *const f32,
        n: usize,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// inkling's KV-cache append: scatter one token's projected `src` (`n = kv_heads·head_dim`) into
    /// the per-head-deinterleaved cache `dst` (`[kv_heads, ctx, head_dim]`) at `pos`, so each head's
    /// vectors are contiguous for `sink_attend`. No C twin — oracle-validated. `n > 0`,
    /// `head_dim > 0`, `ctx > 0`, `n % head_dim == 0`, `0 <= pos < ctx` (1001).
    ///
    /// # Safety
    /// Device pointers (`src` `n` f32, `dst` `kv_heads·ctx·head_dim` f32) live until the next
    /// [`device_sync`]; `stream` is null or a live stream ordering every producer.
    launch_kv_scatter => rivoli_kv_scatter(KvScatter { src, dst, pos: pos as i32, head_dim: head_dim as i32, ctx: ctx as i32, n: n as i32 }, stream), "kv_scatter" (
        src: *const f32,
        dst: *mut f32,
        pos: usize,
        head_dim: usize,
        ctx: usize,
        n: usize,
        stream: *mut c_void,
    );

    /// inkling's rel-bias distance gather: for one decode query at `pos` over the key window
    /// `start..start+n`, key slot `t` sits at absolute position `start + t` and
    /// `bias[h·n+t] = rel_logits[h·rel_extent + pos−(start+t)]` when that distance is in
    /// `[0, rel_extent)`, else 0 (MOD:224–227). `rel_logits` `[heads, rel_extent]`, `bias`
    /// `[heads, n]`. No C twin — oracle-validated. `heads > 0`, `n > 0`, `rel_extent > 0`,
    /// `pos >= 0`, `start >= 0`, `heads·n <= i32::MAX` (1001).
    ///
    /// # Safety
    /// Device pointers (`rel_logits` `heads·rel_extent` f32, `bias` `heads·n` f32) live until the
    /// next [`device_sync`]; `stream` is null or a live stream ordering every producer.
    launch_rel_bias_gather => rivoli_rel_bias_gather(RelBiasGather { rel_logits, bias, pos: pos as i32, start: start as i32, n: n as i32, heads: heads as i32, rel_extent: rel_extent as i32 }, stream), "rel_bias_gather" (
        rel_logits: *const f32,
        bias: *mut f32,
        pos: usize,
        start: usize,
        n: usize,
        heads: usize,
        rel_extent: usize,
        stream: *mut c_void,
    );

    /// DeepSeek-V4's `Expert.forward` combine, for the resident fp8 **shared** expert:
    /// `h = bf16( silu(min(bf16(g), limit)) · clamp(bf16(u), ±limit) )`. Safe in place.
    ///
    /// `MoE.__init__` hands `swiglu_limit` to `shared_experts` as well as to the routed ones
    /// (model.py:632) and `Expert.forward` clamps both, so the shared expert needs the same
    /// clamped arithmetic `launch_moe_expert_range_f4` already runs on the fp4 routed experts.
    /// **NOT YET WIRED, as of 2026-08-05, and this launcher existing does not fix anything by
    /// itself.** *[CORRECTED 2026-08-08: the V4 loop exists now and calls the unclamped
    /// [`launch_swiglu`] from its shared-expert chain — `f4gpu.rs::shared_expert` names the
    /// deviation (`Defect::SwigluUnclamped`) at the call. So this launcher's caller-to-be
    /// exists and still does not call it; the sentence below about what the wiring must do
    /// stands.]* The clamped combine is available and gated (`tests/f4_kernel.rs` §7), and
    /// the shared expert's fix is to call THIS and not [`launch_swiglu`] — because that one
    /// gives `v4oracle::Defect::SwigluUnclamped` on one contribution in seven of all 43
    /// layers, fluent and wrong.
    ///
    /// # Why this is not [`launch_swiglu`] with a `limit`
    ///
    /// **At no value of `limit` would the two agree**, so a parameter could not have expressed
    /// it. [`launch_swiglu`] is `(g/(1+e^-g))·u` and rounds nothing. Three differences besides
    /// the clamp, each of them a defect the oracle names:
    ///
    /// - both operands are bf16-rounded **before** the clamp (`Linear` stores bf16 and
    ///   `Expert.forward` reads it back with `.float()`) — `Defect::NoBf16Rounding`;
    /// - the product is bf16-rounded, the reference's `x.to(dtype)` in front of `w2`;
    /// - `F.silu`'s multiply form `g·sigmoid(g)`, not the division form, which `moe.hip` records
    ///   as one rounding apart that "would normally vanish under the bf16 store ... except
    ///   exactly at a rounding boundary". This one is **true by construction and unexercised**:
    ///   swapping the forms was measured bit-identical over all 512 outputs of the fp4 fixture
    ///   (2026-08-05), which is what that comment predicts when the boundary is not reached.
    ///   The decision below does not rest on it — the two bf16 roundings and the clamp are each
    ///   demonstrated by a break that goes red.
    ///
    /// GLM has no `swiglu_limit` and should never acquire one, so passing an "unclamped"
    /// sentinel from its four call sites would put a value in the tree that no config can
    /// produce. This refuses `limit <= 0`, **NaN and `+/-inf`** (guard 1006, the same code `moe.hip`
    /// returns for the same check on the same argument), so unclamped is not spellable here.
    ///
    /// The clamp itself is one `crates/kernels/src/prelude/activation.rs::swiglu_clamped` (was
    /// `kernels/common.hpp`), shared with `moe_gateup_f4`: the routed and shared paths must
    /// agree bit for bit.
    ///
    /// # Safety
    /// `g`, `u` and `h` are each `n` f32 and must outlive `stream`'s completion. `h` may alias
    /// `g` or `u` — every thread reads both, then writes once, and that write depends on both
    /// reads. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_swiglu_clamped_bf16 => rivoli_swiglu_clamped_bf16(GateClamped { gate: Gate { g, u, h, n: n as i32 }, limit }, stream), "swiglu_clamped_bf16" (
        g: *const f32,
        u: *const f32,
        n: usize,
        limit: f32,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// **SiTU-GLU over a gate/up pair — Kimi-K3's activation**, `k3-architecture.md` §8:
    /// `y = (b1·tanh(g/b1)·sigmoid(g)) · (b2·tanh(u/b2))`, so `|y| <= b1·b2` (100 at the shipped
    /// 4 and 25). The arithmetic is `crates/kernels/src/prelude/activation.rs::situ_glu`;
    /// was `kernels/common.hpp::situ_glu`, which argues it (tag `archive/hip-kernels`); ported
    /// from `k3:src/backend/hip.rs` (M9).
    ///
    /// **Not [`launch_swiglu_clamped_bf16`] with different constants.** The sigmoid takes the
    /// UNCAPPED gate, `up` is transformed rather than clamped, neither operand is bf16-rounded,
    /// and the store is f32. Merging the two behind a parameter is the refactor `common.hpp`
    /// warns against for `swiglu`/`swiglu_clamped` — the same hazard, one model over.
    ///
    /// **This is the DENSE path only** — layer 0's MLP and the shared MLP. The routed experts
    /// fuse the same helper inside the fp4 expert kernel (`crates/kernels/src/moe_f4.rs`) and
    /// never come through here,
    /// so a change made here alone leaves every routed expert wrong (plan §3b).
    ///
    /// Both betas must be finite and positive; `<= 0`, NaN and `+/-inf` are refused (1006, the
    /// same code the `swiglu_limit` guards return for the same shape of check). A zero or NaN
    /// beta does not degrade gracefully — it makes `tanh(x/b)` saturate or go NaN for every
    /// element, which no magnitude check on the output would catch.
    ///
    /// # Safety
    /// `g`, `u` and `h` are each `n` f32 and must outlive `stream`'s completion. `h` may alias
    /// `g` or `u` — every thread reads both, then writes once, and that write depends on both
    /// reads. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_situ_glu_f32 => rivoli_situ_glu_f32(GateSitu { gate: Gate { g, u, h, n: n as i32 }, b1, b2 }, stream), "situ_glu_f32" (
        g: *const f32,
        u: *const f32,
        n: usize,
        b1: f32,
        b2: f32,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// `x[i] *= sigmoid(g[i])` — Muse Glimmer's attention output gate, applied between the attend
    /// and `o_proj`.
    ///
    /// **`g` must be `gate_proj(LAYER INPUT)`, not anything derived from `x`.** The reference
    /// computes the gate from the post-`input_layernorm` activation (`glimmer-architecture.md`
    /// §4 item 3); gating on the attention output has the right shapes and the wrong model, and no
    /// signature can prevent it — `tests/glimmer_gate.rs` is what holds a caller to it.
    ///
    /// Not [`launch_swiglu`]: that is `silu(g)*u`, which carries an extra factor of `g`.
    ///
    /// USED BY Kimi-K3's gated MLA too (M9): §5's output gate is this same arithmetic at the
    /// same seam — between [`launch_mha_attend`](crate::hip_attn::launch_mha_attend) and
    /// `o_proj` — so the k3 tree's out-of-place copy was deliberately NOT ported
    /// (`kernels/fwd.hip` (tag `archive/hip-kernels`) records the decision, and trap 10's twin is
    /// [`launch_rmsnorm_gate_heads_f32`](crate::hip_attn::launch_rmsnorm_gate_heads_f32),
    /// which norms first and must not be reached for here).
    ///
    /// A `g` of ±Inf gates by the LIMIT (exactly 1 or 0) — finite output from non-finite input,
    /// a reviewed decision recorded on the kernel.
    ///
    /// # Safety
    /// `x` is `n` writable f32 and `g` is `n` readable f32, both live until the next
    /// [`device_sync`], and they must not alias — both are `__restrict__`. `stream` is null or a
    /// live stream ordering every producer of `x` and `g` — this kernel runs BETWEEN the attend
    /// and the o_proj GEMV, so at any stream-ordered call site null is the unordered-read bug,
    /// not a default.
    launch_sigmoid_gate => rivoli_sigmoid_gate(InPlaceBinary { x, y: g, n: n as i32 }, stream), "sigmoid_gate" (
        x: *mut f32,
        g: *const f32,
        n: usize as i32,
        stream: *mut c_void,
    );

    /// `x[i] = cap * tanh(x[i] * mult / cap)` — Muse Glimmer's logit path, applied to the head's
    /// output. `mult` is `output_multiplier`, `cap` is `final_logit_softcapping`; the launcher
    /// refuses a transposed pair (`mult >= cap`, code 1002) because the swap is a silent
    /// sign-quantiser, and refuses non-finite or non-positive values of either (1001).
    ///
    /// **Every greedy gate in this repo is provably blind to omitting this.** `mult > 0` and `tanh`
    /// is strictly increasing, so it cannot move an argmax — the anchor measured `softcap_off`
    /// leaving `emitted.ids` bit-identical while the logits moved. It changes every probability,
    /// so its evidence must come from logits and never from what was decoded.
    ///
    /// **A non-finite logit passes through unchanged.** The naive form maps ±Inf to exactly ±cap,
    /// which would launder an overflowed head output into a finite argmax winner one launch
    /// before the engine's ONLY post-final-layer fault detector (`argmax`'s non-finite bail).
    ///
    /// # Safety
    /// `x` is `n` writable f32, live until the next [`device_sync`]; `stream` is null or a live
    /// stream ordering the head GEMV that produced `x`.
    launch_logit_softcap => rivoli_logit_softcap(Softcap { x, n: n as i32, mult, cap }, stream), "logit_softcap" (
        x: *mut f32,
        n: usize as i32,
        mult: f32,
        cap: f32,
        stream: *mut c_void,
    );

    // ── activation quantizers ───────────────────────────────────────────────────────────────

    /// `kernel.py::act_quant(v, 128, "ue8m0", inplace=True)` over `n_rows x row_len` f32, in
    /// place — the fp8 activation quantization V4 performs in front of every quantized
    /// `Linear`, fp4-weight ones included.
    ///
    /// Fused quantize-then-dequantize: the buffer stays f32 and holds `e4m3(v/s)·s`. That is
    /// what the reference's `inplace=True` does and what the oracle models, so the values a
    /// following GEMV consumes are the reference's own.
    ///
    /// # Safety
    /// `v` is `n_rows · row_len` live f32 for `stream`'s duration.
    launch_act_quant_f8 => rivoli_act_quant_f8(v, n_rows as i32, row_len as i32, stream), "act_quant_f8" (
        v: *mut f32,
        n_rows: usize,
        row_len: usize,
        stream: *mut c_void,
    );

    /// `kernel.py::act_quant(x, block, "ue8m0", inplace=True)` over `rows` rows of
    /// `row_stride` floats, quantizing the first `n` of each — reading `src`, writing
    /// `dst` at the same offsets. `src == dst` is the reference's in-place form;
    /// `n < row_stride` is then the KV entry's PARTIAL quantization (model.py:512, dims
    /// `[0, head_dim - rope_head_dim)` at block 64). `src != dst` is the M10 fused
    /// quantize-copy — one launch where the qkv chain ran `memcpy_dtod_async` + this,
    /// leaving `src` untouched for its other readers — and the launcher REFUSES it at
    /// `n != row_stride` (code 1002), because a partial-width quant-from-source would
    /// leave `dst`'s row tails stale where the copy it replaces filled them.
    /// `n == row_stride` at block 128 is what every quantized `Linear` does to its
    /// activation before the GEMM.
    ///
    /// # Safety
    /// `src` and `dst` are device buffers of at least `rows * row_stride` f32 (the same
    /// buffer, or non-overlapping ones), and must outlive `stream`'s completion. `stream`
    /// is a live `hipStream_t`, or null for the default stream.
    launch_act_quant_f8_prefix => rivoli_act_quant_f8_prefix(ActQuantPrefix { src, dst, rows: rows as i32, row_stride: row_stride as i32, n: n as i32, block: block as i32 }, stream), "act_quant_f8_prefix" (
        src: *const f32,
        dst: *mut f32,
        rows: usize as i32,
        row_stride: usize as i32,
        n: usize as i32,
        block: usize as i32,
        stream: *mut c_void,
    );

    /// `rotate_activation` then `fp4_act_quant(·, 32, inplace=True)` over `rows` rows of `d`
    /// floats, in place — `Oracle::indexer_spread` (forward.rs:1130-1138) and the finish
    /// `Compressor.forward` performs when `rotate = true` (model.py:374-376).
    ///
    /// Applied to BOTH the indexer's `q` rows and its nested compressor's pooled rows, which is
    /// why it is one launcher rather than a step inside either. [`launch_act_quant_f8_prefix`] is the
    /// *other* compressor's finish and takes a partial extent; this one has none — the Hadamard
    /// covers the whole row, RoPE tail included. Handing either the other's extent is finite,
    /// plausible and wrong, so `kvcompress::Geom` carries which is due and
    /// `kvcompress::compress` matches on it.
    ///
    /// `d` must be a power of two no greater than 256 and a multiple of 32; the launcher
    /// refuses otherwise (guards 1002/1003/1004) rather than transforming a length the
    /// reference would have zero-padded, or quantizing a ragged tail against its own amax.
    ///
    /// # Safety
    /// `x` is `rows · d` writable, 4-byte-aligned, device-resident f32, read and written in
    /// place, and outlives `stream`'s completion. `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_act_quant_f4_rotated => rivoli_act_quant_f4_rotated(Rows(rows as i32), F4Rotate { x, d: d as i32 }, stream), "act_quant_f4_rotated" (
        x: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        stream: *mut c_void,
    );

    // ── normalization ───────────────────────────────────────────────────────────────────────

    /// RMSNorm `y = x·rsqrt(mean(x²)+eps)·w`.
    ///
    /// # Safety
    /// Device pointers (`x`, `w`, `y` each `n` f32) live until the next [`device_sync`].
    launch_rmsnorm_single => rivoli_rmsnorm_single(RowNorm { x, w, y, n: n as i32, eps }), "rmsnorm_single" (
        x: *const f32,
        w: *const f32,
        n: usize,
        eps: f32,
        y: *mut f32,
    );

    /// `y[i] = x[i]·(1/sqrt(mean(x²)+eps))·(1 + w[i])` — Muse Glimmer's CENTERED RMSNorm, one row.
    ///
    /// **The `(1 + w)` is the whole difference from [`launch_rmsnorm_single`], and this model uses
    /// BOTH forms.** Glimmer's four per-layer sandwich norms are centered with `w` initialised to
    /// ZEROS; its final norm, weightless qk_norm and embedding norm are the plain form with `w`
    /// ones (`glimmer-architecture.md` §5). NEITHER substitution announces itself — the anchor's
    /// `norm_not_centered` run leaves zero non-finite values and decodes on, so §5's "crashes into
    /// garbage" is wrong and §9 trap 5's "runs clean" is right. Two entry points rather than a flag, the
    /// `launch_rope_split_half` argument: the wrong form is silent in one direction (a plain weight
    /// through here scales by ≈2 and stays fluent) and a bool would put it one argument from every
    /// GLM and V4 call site.
    ///
    /// `eps` is refused unless positive and finite (code 1002). Glimmer's four norms carry TWO eps
    /// three orders of magnitude apart, and passing one where the other belongs is the anchor
    /// defect that sets this operator's tolerance row.
    ///
    /// # Safety
    /// `x`, `w` and `y` are each `n` live f32 — `x`/`w` readable, `y` writable — until the next
    /// [`device_sync`]; `y` may alias `x` (each thread reads and writes only index `i` after the
    /// block-wide reduction has completed). `stream` is null or a live stream ordering every
    /// producer of `x` and `w`.
    launch_rmsnorm_centered_single => rivoli_rmsnorm_centered_single(RowNorm { x, w, y, n: n as i32, eps }, stream), "rmsnorm_centered_single" (
        x: *const f32,
        w: *const f32,
        n: usize,
        eps: f32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// `RMSNorm.forward` over `rows` rows of `d` floats, in place: f32 statistic, learned
    /// weight, bf16-rounded store.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`rows * d` f32), `w` (`d` f32).
    /// `stream` is a live `hipStream_t`, or null for the default stream.
    launch_rmsnorm_batch => rivoli_rmsnorm_batch(Rows(rows as i32), RmsNormRows { x, w, d: d as i32, eps }, stream), "rmsnorm_batch" (
        x: *mut f32,
        w: *const f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        stream: *mut c_void,
    );

    /// Muse Glimmer's weightless QK-norm over `rows` heads of `d`, in place, times `scale`.
    ///
    /// Q passes `qk_scale_factor` (3.87) and K passes 1.0 — the scale is Q's alone
    /// (`glimmer-architecture.md` trap 3), and it is folded here rather than given a second pass
    /// over the tensor. **Not `rmsnorm_batch`** (which multiplies a learned weight and stores bf16)
    /// **and not `mla.rs::qk_norm`** (whose statistic is bf16 by DeepSeek-V4's reference); the
    /// kernel's own comment carries both arguments.
    ///
    /// # Safety
    /// `x` is a device buffer of `rows · d` live f32, written in place, live until the next
    /// [`device_sync`]. `stream` is null or a live stream ordering every producer of `x`.
    ///
    /// **`x` is DESTROYED — the pre-norm values do not survive this call**, so no consumer of them
    /// may be enqueued after this launch on any stream. The clause above constrains what may WRITE
    /// `x` before; this one constrains what may READ it after, and only the first was written down
    /// (review, 2026-08-13). The realistic violation is a `--trace` or `--pred-probe` readback of q
    /// expecting `q_proj`'s output and getting post-norm, post-3.87 bytes; no fixture can see it,
    /// which is the same shape as the null-stream finding this port already carries.
    launch_rmsnorm_weightless_batch => rivoli_rmsnorm_weightless_batch(Rows(rows as i32), WeightlessNorm { x, d: d as i32, eps, scale }, stream), "rmsnorm_weightless_batch" (
        x: *mut f32,
        rows: usize,
        d: usize,
        eps: f32,
        scale: f32,
        stream: *mut c_void,
    );

    /// LayerNorm with bias `y = (x-mean)/sqrt(var+eps)·w + b` (the indexer k_norm).
    ///
    /// # Safety
    /// Device pointers (`x`, `w`, `b`, `y` each `n` f32) live until the next [`device_sync`].
    launch_layernorm => rivoli_layernorm(LayerNorm { row: RowNorm { x, w, y, n: n as i32, eps }, b }), "layernorm" (
        x: *const f32,
        w: *const f32,
        b: *const f32,
        n: usize as i32,
        eps: f32,
        y: *mut f32,
    );

    /// The weightless per-head QK-norm of model.py:504, in place over `rows = s * n_heads`
    /// rows of `head_dim`. Must be launched BEFORE the RoPE — see the kernel's note: the
    /// oracle provably cannot see the order, so it comes from the reference.
    ///
    /// # Safety
    /// `q` is a device buffer of at least `rows * d` f32, and must outlive `stream`'s
    /// completion. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_qk_norm => rivoli_qk_norm(Rows(rows as i32), QkNorm { q, d: d as i32, eps }, stream), "qk_norm" (
        q: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        stream: *mut c_void,
    );

    // ── RoPE — three pairing conventions, never interchangeable ─────────────────────────────

    /// Interleaved RoPE in place over `count` segments of `seg` at `stride`.
    ///
    /// # Safety
    /// `base` is a device buffer of `count·stride` f32, live until the next [`device_sync`].
    launch_rope_interleave => rivoli_rope_interleave(Rows(count as i32), Rope { base, theta, stride: stride as i32, seg: seg as i32, pos: pos as i32 }), "rope_interleave" (
        base: *mut f32,
        count: usize,
        stride: usize,
        seg: usize,
        pos: usize,
        theta: f64,
    );

    /// Split-half RoPE in place — transformers' `rotate_half`, where the pair is
    /// `(x[j], x[j+seg/2])` rather than two adjacent elements. Muse Glimmer's convention.
    ///
    /// **Same arithmetic as [`launch_rope_interleave`], different pairing, and the two are NOT
    /// interchangeable.** Applying one where the other is meant produces fluent wrong text and no
    /// error — `glimmer-architecture.md` §9 trap 9. They are separate entry points rather than one
    /// with a flag precisely so a GLM or V4 call site cannot reach this convention by changing an
    /// argument; `kernels/linalg.hip` (tag `archive/hip-kernels`) carries the argument, and `swiglu`/`swiglu_clamped_bf16` is
    /// the precedent.
    ///
    /// # Safety
    /// `base` is a device buffer of `count·stride` f32, live until the next [`device_sync`].
    launch_rope_split_half => rivoli_rope_split_half(Rows(count as i32), Rope { base, theta, stride: stride as i32, seg: seg as i32, pos: pos as i32 }), "rope_split_half" (
        base: *mut f32,
        count: usize,
        stride: usize,
        seg: usize,
        pos: usize,
        theta: f64,
    );

    /// `apply_rotary_emb` over the last `rd` dims of each of `rows` rows, ADJACENT-PAIR
    /// (`view_as_complex`), from a precomputed `(cos, sin)` table. Row `r` takes position
    /// `pos0 + r / rows_per_pos`. `inverse` conjugates it — the output de-rotation.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`rows * row_len` f32), `tbl`
    /// (at least `(pos0 + rows / rows_per_pos) * rd` f32, interleaved cos/sin). `stream` is a
    /// live `hipStream_t`, or null for the default stream.
    launch_rope_adjacent => rivoli_rope_adjacent(Rows(rows as i32), RopeAdjacent { x, tbl, row_len: row_len as i32, rd: rd as i32, pos0: pos0 as i32, rows_per_pos: rows_per_pos as i32, inverse: i32::from(inverse) }, stream), "rope_adjacent" (
        x: *mut f32,
        tbl: *const f32,
        rows: usize as i32,
        row_len: usize as i32,
        rd: usize as i32,
        pos0: usize as i32,
        rows_per_pos: usize as i32,
        inverse: bool as i32,
        stream: *mut c_void,
    );
}
// jscpd:ignore-end
