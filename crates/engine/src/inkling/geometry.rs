//! Inkling 2's shapes and schedule: the looped application order, the block-residual boundaries,
//! the sliding/full layer split, and the short-convolution divisors. **Touching no device** —
//! pure functions over plain data so the deviceless CI arm builds and tests them (the pin and
//! loop body that consume these are `rocm`-gated; a rule whose test needs the artifact is a rule
//! nothing runs).
//!
//! The two functions that carry a trap are free for the same reason as glimmer's `window_of`:
//! [`application_schedule`]'s loop order and [`is_block_boundary`]'s inclusion of layer 0 (T7)
//! are the things a loop is most likely to get fluently wrong.

use rivoli_artifact::inkling_config::InklingTextConfig;

/// The decoder's application order as a sequence of stored-layer indices. The non-looped member
/// (Small) is `0..num_layers`; the looped member (Mini) runs the pre-loop prefix `[0, core_start)`
/// once, the core `[core_start, core_end)` `num_passes` times, then the suffix `[core_end,
/// num_layers)` once. Mini: 6 + 12·3 + 6 = 48 applications of 24 stored layers.
pub fn application_schedule(num_layers: usize, core: Option<(usize, usize, usize)>) -> Vec<usize> {
    let Some((core_start, core_end, num_passes)) = core else {
        return (0..num_layers).collect();
    };
    let mut seq = Vec::with_capacity(num_layers + (core_end - core_start) * (num_passes - 1));
    seq.extend(0..core_start);
    for _ in 0..num_passes {
        seq.extend(core_start..core_end);
    }
    seq.extend(core_end..num_layers);
    seq
}

/// Layer `i` opens a block-residual block iff `i % block == 0` — **including layer 0** (trap T7;
/// MOD:1433–1434). The boundary residual reset fires here; the block END (full attention) is the
/// complement, [`is_block_end`].
pub fn is_block_boundary(i: usize, block: usize) -> bool {
    i.is_multiple_of(block)
}

/// Layer `i` closes a block (full attention) iff `(i + 1) % block == 0`.
pub fn is_block_end(i: usize, block: usize) -> bool {
    (i + 1).is_multiple_of(block)
}

/// The static short-convolution divisor `√(k+1)` (T15): k=4 → √5. The `(conv+residual)/√(k+1)`
/// of `Inkling2ShortConvolution` (MOD:349).
pub fn sconv_static_divisor(k: usize) -> f32 {
    ((k + 1) as f32).sqrt()
}

/// The dynamic short-convolution divisor `√(2k+1)` (T15): k=4 → 3. The `(sum+residual)/√(2k+1)`
/// of `Inkling2DynamicShortConvolution` (MOD:412).
pub fn sconv_dynamic_divisor(k: usize) -> f32 {
    ((2 * k + 1) as f32).sqrt()
}

/// The attention QK scale `head_dim^−0.5` (MOD:530).
pub fn qk_scale(head_dim: usize) -> f32 {
    1.0 / (head_dim as f32).sqrt()
}

/// The cache-layer offset for an extra loop pass (T16; LOOP:542). Pass 0 (the main forward) uses
/// offset 0; extra pass `loop_idx` (1-based, `1..num_passes`) appends its own cache copies after
/// the model's layers: `num_layers + (loop_idx−1)·(core_end−core_start) − core_start`. Mini:
/// pass 2 → 18, pass 3 → 30. The extra passes restart from the pre-loop states (T8).
pub fn loop_cache_offset(
    num_layers: usize,
    core_start: usize,
    core_end: usize,
    loop_idx: usize,
) -> usize {
    num_layers + (loop_idx - 1) * (core_end - core_start) - core_start
}

/// The LogScale factor at `pos` (T9): `τ = 1 + α·ln(max((pos+1)/n_floor, 1))`. Full layers
/// multiply their q/k normalisation by it; sliding layers never do. `n_floor` is 8192 in BOTH
/// shipped configs — the class default (128000) is the trap. The ratio is clamped at 1.0, so
/// `pos < n_floor` is exactly `τ = 1.0`.
pub fn log_scale_tau(pos: usize, alpha: f32, n_floor: usize) -> f32 {
    let floor = n_floor.max(1) as f32;
    1.0 + alpha * (((pos + 1) as f32 / floor).max(1.0)).ln()
}

/// One streamed expert's staging-slot bytes: the w13 `[2I, H]` plus the down `[H, I]` bf16
/// pair at `bytes_per` storage. This is the charge the floor reserves for the streaming path
/// itself — `InklingScratch` does not hold it; [`super::pin`] does.
pub fn single_expert_bytes(hidden: usize, inter: usize, bytes_per: usize) -> usize {
    (2 * inter * hidden + hidden * inter) * bytes_per
}

/// Bytes per f32 — the engine's activations and caches are f32 (the residual stream and KV
/// allocate `DeviceBuf::new(n*4)`; the glimmer drafter verdict records the same f32-cache rule).
const F32: usize = 4;

/// Bytes per bf16 — every dense projection and both expert slabs are stored bf16.
const BF16: usize = 2;

/// The bytes the six GLOBAL tensors occupy in the tier: `embed_tokens` bf16 (all
/// `vocab_size` rows — T14, the embed is NOT unpadded), `embed_norm`/`norm`/`final_input.query`
/// f32 (`hidden` each), `message_input.query` f32 (`core_size·4·hidden`, present exactly when
/// the member loops), and `lm_head` bf16 (the live `unpadded_vocab_size` rows, T14).
///
/// The pin PLACES each of these from the artifact, which carries its own byte length; this is
/// the deviceless restatement of that length, and the two are checked against each other by
/// the pin's own floor listing rather than compared by eye.
pub fn global_weight_bytes(text: &InklingTextConfig) -> usize {
    let h = text.hidden_size;
    let mut bytes = text.vocab_size * h * BF16; // embed_tokens
    bytes += 3 * h * F32; // embed_norm, norm, final_input.query
    bytes += text.unpadded_vocab_size * h * BF16; // lm_head
    if let Some(lc) = &text.loop_config {
        bytes += (lc.core_end - lc.core_start) * 4 * h * F32; // message_input.query
    }
    bytes
}

/// Every byte [`super::engine::InklingScratch`] allocates, named once.
///
/// **One author for the scratch's size.** The pin's [`rivoli_core::residency::Floor`] charges
/// for what the engine will allocate, and [`super::engine::InklingScratch::new`] allocates
/// exactly these fields — so a buffer added to the scratch without a field here is a floor the
/// partition never saw, and the pin OOMs on `DeviceBuf::new` after the placement decision has
/// already been made. `total` is the sum of the same numbers `new` reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScratchSizes {
    /// One position's residual stream (`hidden`).
    pub residual: usize,
    /// The block-residual bank: one residual vector per boundary slot, plus the partial.
    pub bank: usize,
    pub bank_slots: usize,
    /// One query projection (`heads·head_dim`).
    pub qd: usize,
    /// One KV projection (`kv_heads·head_dim`).
    pub kv_vec: usize,
    /// One dynamic short-convolution's per-KV-head taps (`kv_heads·k`). NOT `kv_vec`: the tap
    /// vectors are `k` wide, so charging the projection's width over-reserved the buffer 32×.
    pub dyn_w: usize,
    /// Cache layers, each with its own KV pair and its own four sconv states (T16).
    pub n_cache: usize,
    /// One KV cache tensor over `n_ctx` positions.
    pub kv_tensor: usize,
    /// One k/v short-convolution state.
    pub sconv_kv: usize,
    /// One post-attention/post-mlp short-convolution state.
    pub sconv_h: usize,
    pub rel_states: usize,
    pub rel_logits: usize,
    pub bias: usize,
    /// One sink logit per query head.
    pub sink: usize,
    /// The live logit row (T14: `unpadded_vocab_size`).
    pub logits: usize,
    /// The MessageGate's rank-width scratch, at least one element wide.
    pub gate_low: usize,
    pub router: usize,
    pub moe_gu: usize,
    pub moe_h: usize,
}

impl ScratchSizes {
    /// The sizes for `n_ctx` positions.
    pub fn of(text: &InklingTextConfig, n_ctx: usize) -> Self {
        let hidden = text.hidden_size;
        let heads = text.num_attention_heads;
        let kv_heads = text.num_key_value_heads;
        let hd = text.head_dim;
        let block = text.n_layers_per_attn_res_block.max(1);
        let core = text
            .loop_config
            .as_ref()
            .map(|lc| (lc.core_start, lc.core_end, lc.num_passes));
        let gate_rank = text
            .loop_config
            .as_ref()
            .map_or(1, |lc| lc.gate_rank)
            .max(1);
        // ONE computation, and `div_ceil`: floor division loses a partial block
        // (`num_hidden_layers % block != 0`), which leaves the LAST boundary — plus the partial
        // row — past a bank sized on the quotient, so `block_res_attend` writes out of bounds.
        let bank_slots = text.num_hidden_layers.div_ceil(block) + 1;
        Self {
            residual: residual_bytes(hidden),
            bank_slots,
            bank: bank_slots * residual_bytes(hidden),
            qd: heads * hd * F32,
            kv_vec: kv_heads * hd * F32,
            dyn_w: kv_heads * text.sconv_kernel_size * F32,
            n_cache: cache_layer_count(text.num_hidden_layers, core),
            kv_tensor: kv_tensor_bytes(kv_heads, hd, n_ctx),
            sconv_kv: sconv_state_bytes(kv_heads * hd, text.sconv_kernel_size),
            sconv_h: sconv_state_bytes(hidden, text.sconv_kernel_size),
            rel_states: heads * text.d_rel * F32,
            // The full layers' extent is `rel_extent`, the sliding layers' the window (T17):
            // one scratch serves both, so it holds the larger.
            rel_logits: heads * text.rel_extent.max(text.sliding_window_size) * F32,
            bias: heads * n_ctx * F32,
            sink: heads * F32,
            logits: text.unpadded_vocab_size * F32,
            gate_low: gate_rank * F32,
            router: text.n_routed_experts * F32,
            moe_gu: 2 * text.intermediate_size * F32,
            moe_h: text.intermediate_size * F32,
        }
    }

    /// The total bytes [`super::engine::InklingScratch::new`] allocates — ONE term per buffer,
    /// in the order `new` allocates them, so a missing term is a visible line rather than a
    /// rounding difference.
    pub fn total(&self) -> usize {
        let r = self.residual;
        let mut n = 0;
        n += 4 * r; // x, xn, xn_k, xn_v
        n += self.bank; // block_bank
        n += self.n_cache * (2 * self.sconv_kv + 2 * self.sconv_h); // sconv states
        n += 2 * self.n_cache * self.kv_tensor; // kc, vc
        n += 2 * r + 2 * self.bank; // pre_loop_x, pre_loop_bank, prev_pass_x, prev_pass_bank
        n += 3 * self.qd; // q, attn, gate
        n += 2 * self.kv_vec; // k, v
        n += self.rel_states + self.rel_logits + self.bias + self.sink;
        n += self.logits;
        n += self.bank; // stacked
        n += 4 * r; // res_out
        n += 4 * self.bank_slots * F32; // res_bias
        n += 4 * F32; // res_sink
        n += 4 * r; // msg_buf
        n += self.dyn_w; // dyn_w
        n += r; // branch
        n += 4 * r; // gate_cat, gate_f, gate_g
        n += self.gate_low;
        n += 3 * r; // xn_q_tmp, xn_k_tmp, xn_v_tmp
        n += self.router;
        n += self.moe_gu + self.moe_h;
        n += 2 * r; // moe_out, moe_acc
        n += 8; // pick: one i32 index and one f32 value
        n
    }

    /// The KV caches' share of [`Self::total`] — the floor's `kv_at_max_ctx` bucket.
    pub fn kv_total(&self) -> usize {
        2 * self.n_cache * self.kv_tensor
    }
}

/// One position's residual stream: `hidden` f32.
pub fn residual_bytes(hidden: usize) -> usize {
    hidden * F32
}

/// ONE of a layer's two KV tensors over `ctx` positions: `kv_heads · head_dim · ctx` f32. The cache
/// allocates this for K and again for V — the two are separate `DeviceBuf`s, so a layer's cache is
/// [`kv_layer_bytes`] while a single allocation is this (the 2× over-allocation the scratch's
/// `kc`/`vc` pair had while it asked for the layer total twice).
pub fn kv_tensor_bytes(kv_heads: usize, head_dim: usize, ctx: usize) -> usize {
    kv_heads * head_dim * ctx * F32
}

/// One layer's K AND V cache over `ctx` positions: `2 ·` [`kv_tensor_bytes`].
pub fn kv_layer_bytes(kv_heads: usize, head_dim: usize, ctx: usize) -> usize {
    2 * kv_tensor_bytes(kv_heads, head_dim, ctx)
}

/// One short-convolution state: `channels · (k−1)` f32 — the past k−1 tokens the causal k-tap
/// window reads (the current token is the residual, not stored).
pub fn sconv_state_bytes(channels: usize, k: usize) -> usize {
    channels * (k - 1) * F32
}

/// One layer's routed-expert weight bytes — the Mini pin's streamed unit: `n_experts ·
/// (2·intermediate·hidden + hidden·intermediate)` elements (the w13 gate/up pair plus down) at
/// `bytes_per` storage. Mini's artifact dequantizes experts to bf16, so `bytes_per = 2`.
pub fn expert_layer_bytes(
    n_experts: usize,
    intermediate: usize,
    hidden: usize,
    bytes_per: usize,
) -> usize {
    n_experts * (2 * intermediate * hidden + hidden * intermediate) * bytes_per
}

/// The number of cache layers the loop needs (T16): the stored layers plus the extra passes'
/// copies. Non-looped: `num_layers`. Looped: `num_layers + (num_passes−1)·(core_end−core_start)`.
/// Mini: 24 + 2·12 = 48 cache layers.
pub fn cache_layer_count(num_layers: usize, core: Option<(usize, usize, usize)>) -> usize {
    match core {
        None => num_layers,
        Some((start, end, passes)) => num_layers + (passes - 1) * (end - start),
    }
}

/// Layer `layer`'s DENSE weight element count — everything except the routed experts (the streamed
/// unit): the q/k/v/o projections and their norm/sconv children, the sink and rel-bias, the input
/// and post-attention norms, the post-attention/post-mlp convolutions, the query banks, the router
/// gate + bias, and (on a looped layer) its four MessageGates.
///
/// The per-layer shape is NOT a function of one flag: **every** term below is read off
/// `docs/measurement/inkling2-reference/mini/text-model-names.tsv`, and the test asserts the sum
/// against that file for all 24 Mini layers rather than against a transcribed budget.
///
/// - `full` = [`is_block_end`]: the three q/k/v input norms (T11) replace the single
///   `input_layernorm`, `attention_input.query` is `[3, H]` instead of `[H]`, and the k/v sconv is
///   the DYNAMIC pair (`k·kv_heads` taps + a `(kv_heads·k, H)` weight projection per side) rather
///   than the static `conv1d`.
/// - `layer > 0`: `attention_input.query` exists (layer 0 has none — the block boundary opens the
///   first bank, it does not consume one).
/// - `core_start <= layer < core_end` with a `loop_config`: the four MessageGates, each
///   `gate_rank·2H + 2·H·gate_rank` (down 2H→rank, forget and message rank→H).
///
/// The pin multiplies by the storage bytes per element; the experts ride [`expert_layer_bytes`]
/// separately.
pub fn dense_layer_elements(text: &InklingTextConfig, layer: usize) -> usize {
    let h = text.hidden_size;
    let nh = text.num_attention_heads;
    let qd = nh * text.head_dim;
    let kv_heads = text.num_key_value_heads;
    let kvd = kv_heads * text.head_dim;
    let k = text.sconv_kernel_size;
    let dr = text.d_rel;
    let full = is_block_end(layer, text.n_layers_per_attn_res_block);
    // The dynamic sconv's `weight_proj` is `[kv_heads·k, H]` — one input tap per KV head per tap.
    let sconv = if full {
        k * kvd + (kv_heads * k) * h
    } else {
        kvd * k
    };
    let mut e = qd * h + qd; // q_proj.linear + q_proj.norm.scale
    e += 2 * (kvd * h + sconv); // k_proj and v_proj: linear + sconv each
    e += 2 * qd * h; // o_proj.gate_proj + o_proj.linear
    e += nh * h; // sink_proj
    e += dr * text.rel_extent + (nh * dr) * h + nh * dr; // rel-bias proj + linear + norm.scale
    e += if full { 3 * h } else { h }; // input norms (T11)
    e += h; // post_attention_layernorm
    e += 2 * h * k; // post_attention_convolution + post_mlp_convolution
    e += h; // mlp_input.query
    if layer > 0 {
        e += if full { 3 * h } else { h }; // attention_input.query
    }
    e += text.n_routed_experts * h + text.n_routed_experts; // mlp.gate.weight + bias
    if let Some(rank) = text
        .loop_config
        .as_ref()
        .filter(|lc| layer >= lc.core_start && layer < lc.core_end)
        .map(|lc| lc.gate_rank)
    {
        e += 4 * (rank * 2 * h + 2 * h * rank);
    }
    e
}

// The loop schedule and block boundaries are the trap-carrying rules; scored deviceless here.
#[cfg(test)]
mod tests {
    use super::*;

    /// The vendored Mini member's config — the ONE reader of the pinned fixture, so a test that
    /// needs it cannot spell the path (or the parse) a second way.
    #[allow(clippy::unwrap_used)] // test: the vendored config is a pinned fixture
    fn mini_cfg() -> rivoli_artifact::inkling_config::Inkling2LoopConfig {
        serde_json::from_str(include_str!(
            "../../../../docs/measurement/inkling2-reference/mini/config.json"
        ))
        .unwrap()
    }

    #[test]
    fn mini_schedule_runs_the_core_three_times() {
        // 24 stored layers, core [6,18), 3 passes → 6 + 36 + 6 = 48 applications.
        let seq = application_schedule(24, Some((6, 18, 3)));
        assert_eq!(seq.len(), 48);
        assert_eq!(&seq[..6], &[0, 1, 2, 3, 4, 5]);
        assert_eq!(&seq[42..], &[18, 19, 20, 21, 22, 23]);
        // each core layer appears exactly three times, consecutively per pass.
        for layer in 6..18 {
            assert_eq!(seq.iter().filter(|&&l| l == layer).count(), 3);
        }
        assert_eq!(&seq[6..18], &(6..18).collect::<Vec<_>>()[..]);
        assert_eq!(&seq[18..30], &(6..18).collect::<Vec<_>>()[..]);
    }

    #[test]
    fn small_schedule_is_plain_order() {
        let seq = application_schedule(48, None);
        assert_eq!(seq, (0..48).collect::<Vec<_>>());
    }

    #[test]
    fn layer_zero_is_a_block_boundary() {
        // T7: the boundary reset fires at layer 0, not only at layer 6.
        assert!(is_block_boundary(0, 6));
        assert!(is_block_boundary(6, 6));
        assert!(!is_block_boundary(5, 6));
        assert!(is_block_end(5, 6));
        assert!(!is_block_end(6, 6));
    }

    #[test]
    fn sconv_divisors_are_sqrt5_and_3() {
        // T15: static /√5, dynamic /3 at k=4 — distinct, not interchangeable.
        let s = sconv_static_divisor(4);
        let d = sconv_dynamic_divisor(4);
        assert!(
            (s * s - 5.0).abs() < 1e-5,
            "static divisor squared must be 5"
        );
        assert!((d - 3.0).abs() < 1e-6, "dynamic divisor must be 3");
        assert_ne!(s, d);
    }

    #[test]
    fn qk_scale_is_head_dim_inverse_sqrt() {
        assert!((qk_scale(128) - 1.0 / 128.0f32.sqrt()).abs() < 1e-7);
    }

    #[test]
    fn log_scale_tau_is_flat_inside_the_floor() {
        // T9: 8192 is BOTH shipped members' floor; the class default (128000) is the trap.
        let alpha = 0.1f32;
        for pos in [0, 1, 100, 8190, 8191] {
            assert_eq!(log_scale_tau(pos, alpha, 8192), 1.0, "pos {pos}");
        }
        let scaled = log_scale_tau(8192, alpha, 8192);
        assert!(
            scaled > 1.0,
            "the first position past the floor scales: {scaled}"
        );
        assert!((scaled - (1.0 + 0.1 * (8193.0f32 / 8192.0).ln())).abs() < 1e-6);
        // alpha = 0 is the identity everywhere (the knob-off arm).
        assert_eq!(log_scale_tau(1 << 20, 0.0, 8192), 1.0);
    }

    /// The pin's floor charges [`ScratchSizes::total`], and `InklingScratch::new` allocates
    /// `ScratchSizes`' fields — the two are ONE type, so this asserts the shape a reader can check
    /// deviceless: the KV is a proper subset, and the loop needs 48 cache layers (T16).
    #[test]
    fn scratch_total_is_the_scratch_allocation_and_exceeds_its_kv() {
        let cfg = mini_cfg();
        let text = &cfg.text;
        let s = ScratchSizes::of(text, 129);
        assert_eq!(
            s.n_cache, 48,
            "T16: 24 stored layers + 2 extra passes of 12"
        );
        assert!(s.kv_total() > 0);
        assert!(
            s.total() > s.kv_total(),
            "the activations ride on top of the KV cache"
        );
        assert!(single_expert_bytes(2304, 1152, 2) > 0);
        assert!(global_weight_bytes(text) > 0);
    }

    /// A PARTIAL final block must still buy its own bank slot. `num_hidden_layers % block != 0`
    /// is the shape floor division under-counts, and the last boundary plus the partial row then
    /// writes past a bank sized on the quotient. 25 layers over 6 → ⌈25/6⌉ + 1 = 6, not 5.
    #[test]
    fn bank_slots_count_the_partial_final_block() {
        let mut text = mini_cfg().text;
        text.num_hidden_layers = 25;
        text.n_layers_per_attn_res_block = 6;
        assert_eq!(ScratchSizes::of(&text, 129).bank_slots, 6);
    }

    #[test]
    fn loop_cache_offset_matches_t16() {
        // Mini: 24 layers, core [6,18). pass2 (loop_idx 1) → 18, pass3 (loop_idx 2) → 30.
        assert_eq!(loop_cache_offset(24, 6, 18, 1), 18);
        assert_eq!(loop_cache_offset(24, 6, 18, 2), 30);
    }

    #[test]
    fn scratch_bytes_are_f32() {
        assert_eq!(residual_bytes(2304), 9216);
        assert_eq!(kv_layer_bytes(8, 128, 1024), 8 * 128 * 2 * 1024 * 4);
        assert_eq!(sconv_state_bytes(1024, 4), 1024 * 3 * 4);
    }

    #[test]
    fn expert_layer_bytes_counts_w13_and_down() {
        // Mini: 128 experts, intermediate 1152, hidden 2304, bf16. w13 = 2·1152·2304, down = 2304·1152.
        let per_expert = 2 * 1152 * 2304 + 2304 * 1152;
        assert_eq!(expert_layer_bytes(128, 1152, 2304, 2), 128 * per_expert * 2);
    }

    #[test]
    fn cache_layer_count_includes_extra_passes() {
        assert_eq!(cache_layer_count(24, Some((6, 18, 3))), 48);
        assert_eq!(cache_layer_count(48, None), 48);
    }

    /// A TSV `shape` column (`2304`, `3072x2304`, `128x2304x2304`) as its element count.
    fn row_elements(shape: &str) -> usize {
        shape
            .split('x')
            .map(|d| d.parse::<usize>().unwrap_or(0))
            .product()
    }

    /// The dense count is asserted against the artifact census, layer by layer — a formula a
    /// slug of prose "cross-checks" is a formula nothing scores. The two expert rows per layer
    /// are the streamed unit ([`expert_layer_bytes`]) and are excluded here BY NAME, so an
    /// accidental extra tensor in either list reddens one side.
    #[test]
    #[allow(clippy::unwrap_used)] // test: the vendored census is a pinned fixture; a parse failure is a test failure
    fn dense_layer_elements_matches_the_mini_census_for_every_layer() {
        let cfg = mini_cfg();
        let text = &cfg.text;
        let tsv = include_str!(
            "../../../../docs/measurement/inkling2-reference/mini/text-model-names.tsv"
        );
        let n = text.num_hidden_layers;
        assert_eq!(n, 24, "the vendored Mini census is the 24-layer member");
        let mut dense = vec![0usize; n];
        let mut rows = vec![0usize; n];
        let mut dense_rows = 0usize;
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let mut cols = line.split('\t');
            let name = cols.next().unwrap();
            let shape = cols.next().unwrap();
            let Some(rest) = name.strip_prefix("model.layers.") else {
                continue; // the six global tensors are not per-layer
            };
            let (layer, suffix) = rest.split_once('.').unwrap();
            let layer = layer.parse::<usize>().unwrap();
            rows[layer] += 1;
            if suffix.starts_with("mlp.experts.") {
                continue; // streamed unit, scored by `expert_layer_bytes`
            }
            dense_rows += 1;
            dense[layer] += row_elements(shape);
        }
        // The examined counts are asserted, not trusted: 687 layer rows, less the 2 expert rows of
        // each of the 24 layers.
        assert_eq!(
            rows.iter().sum::<usize>(),
            687,
            "layer tensor rows in the census"
        );
        assert_eq!(dense_rows, 639, "dense rows in the census");
        for (layer, &want) in dense.iter().enumerate() {
            assert_eq!(
                dense_layer_elements(text, layer),
                want,
                "layer {layer}: dense elements against the census"
            );
        }
    }
}
