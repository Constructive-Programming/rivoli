//! `kernels/activation.hip`'s kernels: the gated combines, the row norms, the fp8 activation
//! quantizer and RoPE.

/// `swiglu` — and the operand triple under every gated combine: `h[i] = f(g[i], u[i])` for
/// `i < n`. `h` may alias `g` (each thread reads both operands before its one write).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Gate {
    pub g: *const f32,
    pub u: *const f32,
    pub h: *mut f32,
    pub n: i32,
}
const _: () = assert!(size_of::<Gate>() == 32);

/// `swiglu_clamped_bf16` — a [`Gate`] plus the clamp; `limit` is refused unless positive
/// and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateClamped {
    pub gate: Gate,
    pub limit: f32,
}
const _: () = assert!(size_of::<GateClamped>() == 40);

/// `situ_glu_f32` — a [`Gate`] plus SiTU's two betas, each positive and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GateSitu {
    pub gate: Gate,
    pub b1: f32,
    pub b2: f32,
}
const _: () = assert!(size_of::<GateSitu>() == 40);

/// `l2_clip` — inkling's per-head L2 clip (no C twin; oracle-validated). Batched: one block per
/// row, `rows` of `n`: `y[i] = x[i]·min(threshold/(‖x‖+1e-6), 1)`. `bid()` selects the row and
/// the block offsets both pointers by `row·n`, so `x`/`y` are `rows·n` f32. `y` may alias `x`
/// (the norm reads every element before the scale writes any, behind the reduction barrier).
/// `threshold > 0` and finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct L2Clip {
    pub x: *const f32,
    pub y: *mut f32,
    pub n: i32,
    pub threshold: f32,
}
const _: () = assert!(size_of::<L2Clip>() == 24);

/// `scaled_l2_clip` — inkling's sliding-layer q/k norm: a learned per-head `scale` applied
/// FIRST, then the L2 clip (no C twin; oracle-validated). Batched like [`L2Clip`]: one block per
/// row, `rows` of `n`, `scale` is `rows·n` f32 (row `r` reads `scale + r·n`).
/// `y[i] = (x[i]·scale[i])·min(threshold/(‖x·scale‖+1e-6), 1)`. `y` may alias `x`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScaledL2Clip {
    pub clip: L2Clip,
    pub scale: *const f32,
}
const _: () = assert!(size_of::<ScaledL2Clip>() == 32);

/// `log_scaled_l2_clip` — the full-layer q/k norm: the scaled clip then a position-dependent
/// LogScale factor `tau` (no C twin; oracle-validated). `tau = log_scale(1, pos, alpha, n_floor)`,
/// precomputed host-side, and refused by the launcher unless positive and finite (1006) — `tau`
/// is a multiply, so a zero or non-finite value is a silent zero/NaN for the whole head. Batched
/// like [`ScaledL2Clip`]. `y` may alias `x`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LogScaledL2Clip {
    pub scaled: ScaledL2Clip,
    pub tau: f32,
}
const _: () = assert!(size_of::<LogScaledL2Clip>() == 40);

/// `rmsnorm_single` and `rmsnorm_centered_single` — one row of `n`: `y` from `x` and `w`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RowNorm {
    pub x: *const f32,
    pub w: *const f32,
    pub y: *mut f32,
    pub n: i32,
    pub eps: f32,
}
const _: () = assert!(size_of::<RowNorm>() == 32);

/// `rmsnorm_weightless_batch` — rows of `d`, normed IN PLACE and scaled.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WeightlessNorm {
    pub x: *mut f32,
    pub d: i32,
    pub eps: f32,
    pub scale: f32,
}
const _: () = assert!(size_of::<WeightlessNorm>() == 24);

/// `act_quant_f8` — `nblk` groups of `ACT_QUANT_BLOCK`, quantized IN PLACE. `nblk` is the
/// launcher's, so the kernel's bound and the grid come from one expression.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ActQuant {
    pub v: *mut f32,
    pub nblk: u64,
}
const _: () = assert!(size_of::<ActQuant>() == 16);

/// `rope_interleave` / `rope_split_half` — segment `s` at `base + s * stride`, `seg` wide,
/// rotated to position `pos` at base `theta`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Rope {
    pub base: *mut f32,
    pub theta: f64,
    pub stride: i32,
    pub seg: i32,
    pub pos: i32,
}
const _: () = assert!(size_of::<Rope>() == 32);

/// `mxfp8_dequant` — inkling's MXFP8 expert decode (no C twin; oracle-validated). `weight` is
/// `n` e4m3 bytes (`E·rows·cols`), `scale` is `E·rows·(cols/32)` raw U8 exponents — ROW-WISE
/// block 32, the scale a raw exponent NOT e4m3 (trap T1). `out` is `n` f32; `cols` is the
/// logical width, a positive multiple of 32. `out[i] = e4m3(w[i])·2^(scale[row·(cols/32)+(i%cols)/32]−127)`.
/// `n` is 64-bit because a Small expert's layer is ~8e9 elements over a single launch, past
/// `i32`; the kernel indexes it with `blockIdx·ELEMENTWISE_BLOCK + threadIdx` in `u64`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Mxfp8Dequant {
    pub weight: *const u8,
    pub scale: *const u8,
    pub out: *mut f32,
    pub n: u64,
    pub cols: i32,
}
const _: () = assert!(size_of::<Mxfp8Dequant>() == 40);

/// `nvfp4_dequant` — inkling's NVFP4 expert decode (no C twin; oracle-validated). `weight` is
/// `E·rows·cols_packed` bytes (two E2M1 codes each, the LOW nibble the even element), `scale` is
/// `E·rows·(cols_packed/8)` e4m3 bytes (block 16), `amax` is `E` f32, `denom` the scalar
/// `out` is `n = E·rows·cols_packed·2` f32; `rows` is per-expert.
/// Association is `(E2M1[code]·e4m3(scale))·(amax[e]/denom)` — f32 multiply is not associative
/// (LOAD:813), and the bf16 golden is bit-compared. `cols_packed` a positive multiple of 8.
/// `n` is 64-bit for [`Mxfp8Dequant`]'s reason; the launcher also requires it to be a whole
/// number of `rows·cols_packed·2` expert planes, so `row / rows` selects the expert.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nvfp4Dequant {
    pub weight: *const u8,
    pub scale: *const u8,
    pub amax: *const f32,
    pub out: *mut f32,
    pub denom: f32,
    pub n: u64,
    pub rows: i32,
    pub cols_packed: i32,
}
const _: () = assert!(size_of::<Nvfp4Dequant>() == 56);

/// `sconv` — inkling's static short-convolution decode step, state included (no C twin;
/// oracle-validated). A depthwise causal k-tap cross-correlation of the window `[state (past k−1
/// tokens), current]`, plus the residual (the current token), divided by `divisor = √(k+1)`
/// (T15: `/√5` for k=4). `weight` is `[channels, k]`; `state` `[channels, k−1]` is ADVANCED in
/// place (shift left one, append current); `current`/`out` `[channels]`. `divisor` positive finite.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sconv {
    pub weight: *const f32,
    pub state: *mut f32,
    pub current: *const f32,
    pub out: *mut f32,
    pub channels: i32,
    pub k: i32,
    pub divisor: f32,
}
const _: () = assert!(size_of::<Sconv>() == 48);

/// `sconv_dynamic` — inkling's dynamic short-convolution decode step: the static kernel's
/// k-tap window with a per-HEAD learned tap vector added to every static weight (no C twin;
/// oracle-validated). `static_w` is `[k, channels]` row-major (`lag 0` = current … `lag k−1` =
/// oldest) and `dyn_w` `[kv_heads, k]` row-major, head `c / head_dim`, same lag order.
/// `state` `[channels, k−1]` is chronological (`[0]` = oldest) and ADVANCED in place;
/// `current`/`out` `[channels]`, `out` may alias `current`. `channels % head_dim == 0`,
/// each of `channels`/`head_dim`/`k` positive; `divisor` positive finite.
///
/// Field order is scalars FIRST (4·4 B) then pointers (5·8 B): 56 bytes, zero padding. Callers
/// name their fields, so the order is free to be the one that packs.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SconvDynamic {
    pub divisor: f32,
    pub channels: i32,
    pub head_dim: i32,
    pub k: i32,
    pub static_w: *const f32,
    pub dyn_w: *const f32,
    pub state: *mut f32,
    pub current: *const f32,
    pub out: *mut f32,
}
const _: () = assert!(size_of::<SconvDynamic>() == 56);

/// `vaxpy` — `x[i] += alpha·y[i]` over `n`, the loop inkling's decode folds into a vector
/// update (no C twin; oracle-validated). `x`/`y` are `n` f32; `n > 0` and `alpha` finite are
/// the launcher's guards, so `alpha = 0.0` and a negative scale are accepted.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Axpy {
    pub x: *mut f32,
    pub y: *const f32,
    pub alpha: f32,
    pub n: i32,
}
const _: () = assert!(size_of::<Axpy>() == 24);

/// `sink_attend` — inkling's single-query attention decode step with a sink (no C twin;
/// oracle-validated). Batched over the query heads: `bid()` is the head, `q`/`out` are
/// `[heads, d]`, `bias` is `[heads, n]` (rel-bias + mask folded), and `sink` is a DEVICE pointer
/// to `heads` f32 — one on-device per-head logit, so the launcher needs no host round-trip.
///
/// `k`/`v` are the per-head-deinterleaved cache `[kv_heads, kv_stride, d]`, entered already
/// offset to the window start; head `h` reads KV head `h / (heads/kv_heads)`, so a GQA mapping
/// is one division and a single KV head is `kv_heads = 1`.
///
/// The softmax joins the extra `sink` logit then drops its column (T10): the sink absorbs mass,
/// so the real weights sum to < 1. `scaling` positive finite; `kv_stride >= n` so a head's
/// window is inside its stride; the launcher sizes `(n + ROW_BLOCK)` f32 of LDS — the logits and
/// the block-max/sum scratch — and refuses more than 65536 bytes (the gfx1151 LDS budget).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SinkAttend {
    pub sink: *const f32,
    pub bias: *const f32,
    pub out: *mut f32,
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub scaling: f32,
    pub n: i32,
    pub d: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub kv_stride: i32,
}
const _: () = assert!(size_of::<SinkAttend>() == 72);

/// `message_gate` — inkling's loop MessageGate pointwise tail (no C twin; oracle-validated).
/// `out[i] = (1−tanh(forget_logit[i]))·hidden[i] + tanh(gain_logit[i])·message[i]`, fp32. The four
/// inputs and `out` are each `n` f32 (one hidden width). The down/forget/gain projections are GEMVs
/// the engine runs separately; this kernel is the gate's combine (MOD:196–200).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MessageGate {
    pub forget_logit: *const f32,
    pub gain_logit: *const f32,
    pub hidden: *const f32,
    pub message: *const f32,
    pub out: *mut f32,
    pub n: i32,
}
const _: () = assert!(size_of::<MessageGate>() == 48);

/// `kv_scatter` — inkling's KV-cache append (no C twin; oracle-validated). Scatters one token's
/// projected `src` (`n = kv_heads·head_dim` f32) into the per-head-deinterleaved cache `dst`
/// (`[kv_heads, ctx, head_dim]`) at position `pos`: element `e` lands at
/// `(e/head_dim)·ctx·head_dim + pos·head_dim + e%head_dim`, so each head's vectors are contiguous
/// for `sink_attend`. A pure data movement. `n > 0`, `head_dim > 0`, `ctx > 0`, `n % head_dim == 0`,
/// `0 <= pos < ctx` (1001).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvScatter {
    pub src: *const f32,
    pub dst: *mut f32,
    pub pos: i32,
    pub head_dim: i32,
    pub ctx: i32,
    pub n: i32,
}
const _: () = assert!(size_of::<KvScatter>() == 32);

/// `rel_bias_gather` — inkling's relative-position bias distance gather (no C twin; oracle-validated).
/// For one decode query at `pos` over the key window `start..start+n`, the key at slot `t` sits
/// at absolute position `start + t`, and `bias[h·n+t] = rel_logits[h·rel_extent + pos−(start+t)]`
/// when `0 <= pos−(start+t) < rel_extent`, else 0 (MOD:224–227). `start` is what makes a sliding
/// window past its own start correct; without it every key would be read at its slot index.
/// `rel_logits` is `[heads, rel_extent]`, `bias` `[heads, n]`. A pure gather + mask.
/// `heads > 0`, `n > 0`, `rel_extent > 0`, `pos >= 0`, `start >= 0`, `heads·n <= i32::MAX` (1001).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RelBiasGather {
    pub rel_logits: *const f32,
    pub bias: *mut f32,
    pub pos: i32,
    pub start: i32,
    pub n: i32,
    pub heads: i32,
    pub rel_extent: i32,
}
const _: () = assert!(size_of::<RelBiasGather>() == 40);
