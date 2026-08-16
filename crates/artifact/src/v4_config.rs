//! DeepSeek-V4-Flash's config schema — one architecture, one file, per the rule that
//! per-model config types stay separate. Ported from `old:src/artifact/model.rs`'s V4 slice
//! (`wt/glimmer-s2` @ 6b7f496), bodies and comments travelling verbatim: in this repo a comment
//! carries the measurement that justified the choice.
//!
//! A separate struct rather than optional fields on [`crate::glm_config::ModelConfig`], and
//! separate serde declarations rather than a shared core. The duplication is the POINT: a
//! shared schema is exactly the mechanism by which a field added for one architecture would
//! silently satisfy the other's parse, which is the defect the load boundary exists to prevent.
//! [`crate::schema`]'s header carries the full argument and the five fields V4's config lacks
//! *because it is not MLA*.
//!
//! **What did NOT come with it, and why.** The reference's V4 slice sits inside a 3452-line
//! `model.rs` beside GLM's, K3's and Glimmer's; only V4's half is here, and the two helpers it
//! shares with those (`ensure_f4_group_aligned`, `parse_config`/`load_config`) stay in
//! `schema.rs`, which is where this tree already put the shared validation vocabulary.
//!
//! **Every field is REQUIRED and several have no reader in this crate yet.** That is
//! deliberate rather than an oversight, and each such field says so at its declaration —
//! `routed_scale`, `index_topk`, `hc_sinkhorn_iters` and `hc_eps` are the four. The
//! alternative is the decode loop reaching for `v4oracle::weights`'s own transliterated
//! constants, which is the failure `index_topk`'s doc names: three types declare a field of
//! that name and the wrong one is the easy reach.

use crate::arch::Arch;
use crate::schema::{ArchConfig, ensure_f4_group_aligned};
use anyhow::{Result, ensure};
use serde::Deserialize;

/// The `quantization_config` block. Checked rather than ignored: `.f4`'s repack and the
/// resident fp8 path each assume one specific scheme, and a checkpoint quantized another
/// way would decode to plausible-but-wrong weights with no error anywhere.
#[derive(Debug, Clone, Deserialize)]
struct QuantConfig {
    fmt: String,
    scale_fmt: String,
    weight_block_size: Vec<usize>,
}

/// DeepSeek-V4-Flash-0731. Shared-KV MQA (one `wkv` entry serving as both K and V for
/// all heads), grouped low-rank output projection, hyper-connection residuals, a
/// hash-routed prefix, and routed experts shipped as FP4 nibbles with e8m0 block scales.
///
/// **The model name is deliberate and load-bearing** (kept through the 2026-08-09
/// rename-for-behaviour pass): this struct IS that checkpoint's `config.json` — every
/// `#[serde(rename)]` below is one of its JSON keys, the unrenamed fields are its keys
/// verbatim — so deserializing another model's config through it is a refusal by design.
///
/// Every field is REQUIRED. See [`crate::schema`]'s header for why that is not negotiable.
#[derive(Debug, Clone, Deserialize)]
pub struct V4Config {
    // The three dimension renames below coincide with `glm_config::ModelConfig`'s and
    // `glimmer_config::GlimmerTextConfig`'s, because all three checkpoints declare them under the
    // SAME HuggingFace-standard JSON names. **Not factored, and not exempted either**: the
    // design argument is this crate's one-type-per-architecture rule (see the module header),
    // and the mechanical answer is the one `glimmer_config.rs` already uses — a doc comment per
    // field breaks the run so jscpd cannot see it. Measured 2026-08-16: with the runs bare, the
    // gate reported both pairs as 36-token clones on the first compile; with these three
    // comments it reports 0 over `crates/`. An exemption that suppresses nothing is a hole in
    // the gate, so none is added.
    /// The checkpoint ships 43. Also the bound `compress_ratio` checks against, which is NOT
    /// `compress_ratios.len()` — that vector is 46 long and its tail is the mtp blocks'.
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    /// The checkpoint ships 4096, and on V4 this is ALSO the routed block's entry width
    /// (`expert_in`), which is why `validate` passes it to `ensure_f4_group_aligned`. On K3 the
    /// two differ.
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    /// The checkpoint ships 129280, and `convert_v4`'s router check reads it: a hash-routed
    /// layer's `ffn.gate.tid2eid` is `[vocab, top_k]`, so a wrong vocab here sizes the one
    /// tensor that decides expert selection on the first three layers.
    #[serde(rename = "vocab_size")]
    pub vocab: usize,

    // --- shared-KV MQA. `head_dim` (512) is the FULL per-head width and the width of the
    // single KV entry; `qk_rope_head_dim` (64) is its RoPE'd tail. There is no
    // nope/rope SPLIT of separate tensors as in MLA — the last 64 dims of one 512-wide
    // vector are rotated in place, which is why `qk_nope_head_dim` has no meaning here.
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    pub head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub q_lora_rank: usize,
    /// Groups the output projection is split into (`wo_a` is `[o_groups·o_lora_rank, …]`).
    pub o_groups: usize,
    pub o_lora_rank: usize,
    /// One KV head, shared by all `n_heads` queries. Validated == 1: the whole attention
    /// frontend is written against a single shared entry.
    pub num_key_value_heads: usize,
    /// The sliding-window span, and the size of the KV ring. Required, not defaulted:
    /// `Attention.forward` indexes the cache as `kv_cache[:, start_pos % win]`, so a wrong
    /// `win` silently attends to the wrong rows rather than failing. S2b had to pass this
    /// in explicitly because it was missing here.
    pub sliding_window: usize,
    /// RMSNorm epsilon, used by `q_norm`/`kv_norm` AND by the weightless QK-norm. Required
    /// for the same reason: a default that differs from the checkpoint shifts every norm
    /// slightly and produces fluent, wrong text.
    pub rms_norm_eps: f64,

    // --- per-layer KV compression. `compress_ratios[l] == 0` means pure sliding-window
    // (no compressor, no indexer, base `rope_theta`, YaRN OFF); `!= 0` selects
    // `compress_rope_theta` WITH YaRN; `== 4` additionally carries an indexer.
    pub compress_ratios: Vec<usize>,

    // --- RoPE. Carried WHOLE, even though this crate reads none of it — the one exception to
    // the "a field must have a reader" rule above, and it is earned. `precompute_freqs_cis`
    // takes seven arguments and the reference's own `ModelArgs` DEFAULTS disagree with this
    // checkpoint on two: `rope_factor: 40` against the config's 16, `compress_rope_theta:
    // 40000.0` against 160000. A type exposing half the group invites the engine to take the
    // rest from those defaults and build a wrong RoPE table on all 41 compressor layers — no
    // error, just wrong positions. All of it, or none of it.
    pub compress_rope_theta: f64,
    pub rope_theta: f64,
    pub rope_scaling: RopeScaling,
    pub max_position_embeddings: usize,

    // --- MoE ---
    //
    // The four serde renames below coincide with `glm_config::ModelConfig`'s, because both
    // architectures declare these under the SAME JSON names.
    //
    // The copy is the point, for the same reason this crate keeps a type per architecture at
    // all: the two agreeing on four JSON names today is a coincidence of the checkpoints, not
    // a shared contract, and a shared struct would become the attractor for a fifth field that
    // is NOT shared.
    //
    // **The old tree wraps this run in a `jscpd:ignore` region. Here it needs none**, and for
    // the same measured reason the dimension run above gives: a doc comment per field breaks
    // the token run. The old tree also priced `#[serde(flatten)]` and rejected it on the
    // arithmetic — nesting `ModelConfig`'s public fields turns ~100 `cfg.n_experts` /
    // `.top_k` / `.moe_inter` / `.n_shared` call sites into `cfg.moe.*` to delete four lines of
    // attribute — and that pricing is unchanged here.
    //
    // What must NOT be shared, and is not: the fields around them. `ModelConfig` gives
    // `norm_topk_prob` and `scoring_func` `#[serde(default)]` because GLM snapshots predate
    // them; on V4 a missing `scoring_func` or `swiglu_limit` is silent-wrong arithmetic,
    // not an old file, so both are required here.
    /// The checkpoint ships 256, and every one of them is FP4 and streams from NVMe. It is also
    /// the SHARED expert's index — `v4_expert_base(l, n_experts, n_experts)` is the boundary's
    /// one definition.
    #[serde(rename = "n_routed_experts")]
    pub n_experts: usize,
    /// The checkpoint ships 6. Doubles as the second dimension of a hash layer's `tid2eid`.
    #[serde(rename = "num_experts_per_tok")]
    pub top_k: usize,
    /// The checkpoint ships 2048 — the down projection's input width, so it is the second
    /// width `ensure_f4_group_aligned` must divide.
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    /// Asserted to be exactly one. The shared expert is fp8 e4m3 and RESIDENT, not FP4 and
    /// streamed, so it is not interchangeable with a routed one — see the note at the end of
    /// this file.
    #[serde(rename = "n_shared_experts")]
    pub n_shared: usize,

    /// `Gate.forward`'s last line, `weights *= self.route_scale` (`model.py:588`), applied
    /// to every routed expert's weight after the top-k renormalization.
    ///
    /// **Required, and it is the RoPE block's argument again — the one place that exception
    /// was earned.** `ModelArgs.route_scale` defaults to `1.` (`model.py:56`) while this
    /// checkpoint ships `1.5` (both `config.json`'s `routed_scaling_factor` and the
    /// reference's own `inference/config.json` `"route_scale": 1.5`). A reader taking the
    /// default would scale every routed contribution by 1/1.5 — fluent, wrong, no crash.
    /// Carried with no reader in this crate yet, deliberately, so the MoE combine cannot
    /// reach for the default when it lands.
    ///
    /// There is no `norm_topk_prob` twin, and that is not an oversight: `Gate.forward`
    /// renormalizes on `score_func != "softmax"` (`model.py:587`), NOT on the config flag,
    /// and V4's `scoring_func` is `sqrtsoftplus` — so the renormalization is unconditional
    /// here and the config's `norm_topk_prob: true` has no effect to carry.
    #[serde(rename = "routed_scaling_factor")]
    pub routed_scale: f64,
    scoring_func: String,
    /// The first `n_hash_layers` layers route by a `tid2eid[vocab, top_k]` table indexed
    /// by token id — the router scores are computed but the SELECTION bypasses them.
    /// Those layers carry `ffn.gate.tid2eid` and NO `ffn.gate.bias`; the rest are the
    /// reverse. [`V4Config::layer_routes_by_hash`] is the one reader.
    #[serde(rename = "num_hash_layers")]
    pub n_hash_layers: usize,
    /// SwiGLU clamp. rivoli's SwiGLU is unclamped, so a lost `10.0` here is silent-wrong,
    /// not a crash — required, and validated non-zero.
    pub swiglu_limit: f64,
    /// `"fp4"`. The `.f4` repack copies e2m1 nibbles verbatim; against an `"fp8"` export
    /// of the same model it would read fp8 bytes as nibble pairs and produce noise.
    pub expert_dtype: String,

    // --- attention-tensor quantization ---
    quantization_config: QuantConfig,

    // --- lightning indexer, on the `compress_ratio == 4` layers. `indexer.wq_b` is
    // `[index_n_heads · index_head_dim, q_lora_rank]`.
    //
    // **CORRECTED 2026-08-16 (M15).** This said the shape was "confronted with the tensor by
    // `convert_v4::write_layer_resident`". It is not: that function's table names
    // `attn.{wq_a, wq_b, wkv, wo_a, wo_b}` and no `attn.indexer.*` entry, and `convert_v4`
    // does not mention `index_n_heads` at all. Harmless while nothing read those tensors;
    // load-bearing since M15, because `v4::blocksel` hands `gemv_fp8` the CONFIG-derived
    // `(index_n_heads · index_head_dim, q_lora_rank)` and ignores the `Fp8Weight`'s own
    // dims, and `narrow_to_bf16` reads config-derived byte counts out of `place_f32`
    // placements, which do no shape check. So a checkpoint whose `attn.indexer.*` shapes
    // disagree with these two fields gets out-of-bounds device reads instead of a
    // load-time refusal. Bounded, not closed, by `place_fp8`, which does confront `wq_b`'s
    // scale grid against the weight's own shape. `attn.compressor.*` has the identical
    // gap and predates M15; both want one confront table, which is its own change. ---
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    /// How many compressed blocks the lightning indexer keeps:
    /// `index_score.topk(min(index_topk, end_pos // ratio))` (`model.py:433`).
    ///
    /// **Required, and the reason is NOT the one `routed_scale` carries.** There the
    /// reference's default (`1.`) disagrees with the checkpoint (`1.5`), so a reader taking
    /// the default is silently wrong. Here they agree — `ModelArgs.index_topk` is 512
    /// (`model.py:77`) and so is the shipped config, a pairing
    /// `v4_config.rs`'s shipped-document tests check against the real file. The hazard is the
    /// other one: serde's `usize` default is **0**, which is a legal-looking value rather than
    /// an obviously-absent one. `topk(min(0, n))` yields a zero-width selection,
    /// `Attention.forward` concatenates it with the sliding-window list (`model.py:519`), and
    /// the `cat` is perfectly legal — so every `compress_ratio == 4` layer silently degrades to
    /// pure sliding-window attention. Fluent, wrong, no error: the same class as
    /// `routed_scale`, reached by the other route. (An earlier version of this comment claimed
    /// it "fails loudly"; it does not, and the oracle agrees — `forward.rs`'s
    /// `topk_idx(&score, 0)` returns an empty row.)
    ///
    /// No upper bound is available or needed: `min(index_topk, end_pos / ratio)` clamps from
    /// above, so an absurd value is a no-op, and nothing in this crate is sized by it.
    ///
    /// **Carried with no reader in this crate yet, deliberately** — like `routed_scale`, and
    /// for a sharper reason: THREE types declare a field of this name (`ModelConfig`'s;
    /// `v4oracle::weights::V4Config`'s, hard-coded to 512; and this one), which is exactly the
    /// setup where the decode path reaches for the wrong one.
    ///
    /// For anyone who finds it looking inert: at 512 it does not truncate until **2052
    /// tokens** — `4 * (index_topk + 1)` — which the old tree records in
    /// `docs/investigations/v4-flash-port.md`, "A hole S3 inherits". Below that the selection
    /// is decided entirely by the causal mask, so a wrong value changes nothing observable.
    /// That is a property of the prompt length, not evidence the field is unused.
    pub index_topk: usize,

    /// Hyper-connection streams. `hc_*_fn` is `[_, hc_mult · hidden]`, likewise confronted
    /// with the tensor by `convert_v4`.
    pub hc_mult: usize,

    /// Sinkhorn passes in `hc_split_sinkhorn` — `Block.hc_pre`'s row/column normalization
    /// of the mHC mixing matrix (model.py:686).
    ///
    /// **Added 2026-08-05 in the old tree by S3, because the layer loop could not be written
    /// without it.** `launch_hc_pre` takes it as a parameter *specifically* so the kernel and
    /// `config.json` cannot drift, and its doc says "passing it from `V4Config`" — but no such
    /// field existed, only `v4oracle::weights::V4Config::hc_sinkhorn_iters`, which is the
    /// oracle's own transliteration and must not be what the engine reads. That is the exact
    /// shape `index_topk`'s doc warns about: three types declare a field of one name and the
    /// decode path reaches for the wrong one.
    ///
    /// Required like every other field here, so the `ensure!` in `validate` reaches exactly
    /// one case: a config that writes `0` explicitly. `rivoli_hc_pre` already refuses that
    /// (`kernels/linalg.hip:642`, `if (iters < 1) return 1003`), so this is a load-boundary
    /// restatement of a check the kernel makes — worth the two lines because the load
    /// boundary names the FIELD and the kernel names a code, and because a config is read
    /// once while the launcher fires 43 times a token.
    ///
    /// **What a zero would cost is smaller than it sounds, and an earlier draft of this
    /// comment overstated it as "the matrix is never normalized at all".** It is not:
    /// `hc_sinkhorn` runs a row softmax and one column divide BEFORE the loop
    /// (`kernels/linalg.hip:370`, `norm(1, HC_MULT);`, and `kernel.py:401-415` agrees), and
    /// only the remaining `iters - 1` passes are the plain row/column refinements. So zero
    /// loses the refinement, not the normalization.
    ///
    /// The reason the value must be *read* rather than *checked* is unchanged: it is a
    /// config value, and the engine reading its own is not something a gate substitutes for.
    ///
    /// > **CORRECTED 2026-08-07.** This said a numeric gate *cannot* pin the shipped 20,
    /// > "because at 20 a 4x4 positive matrix is far past convergence and 19 and 20 agree
    /// > bit-for-bit". That is the toy fixture's behaviour, not the checkpoint's: on real
    /// > weights 19 vs 20 moves 39,893/53,248 of `L0.pre.ffn_norm_out` and all 78 router
    /// > weights, so a golden emitted from the checkpoint DOES distinguish them.
    ///
    /// **Carried with no reader in this crate yet, deliberately**, exactly as `index_topk`
    /// and `routed_scale` are: the layer loop that would read it does not exist. The
    /// alternative to declaring it now is that loop reaching for
    /// `v4oracle::weights::V4Config`, which is the whole reason this field is here.
    pub hc_sinkhorn_iters: usize,

    /// `hc_eps` — and it is **five** things, not the one an earlier draft of this comment
    /// named. It is the floor added to `hc_head`'s sigmoid gate (`model.py:714`,
    /// `pre = sigmoid(...) + hc_eps`), and inside `hc_split_sinkhorn` it is *also* the
    /// `+ eps` after the comb softmax and in every row and column divide — `2·iters - 1` of
    /// them per token (`kernel.py:408, :413, :419, :423`; `kernels/linalg.hip:347` and the
    /// `norm` lambda at :358). So a zero moves `comb` as well as `pre`, in opposite
    /// directions, and removes the guard from those divisions — harmless there, since the
    /// sums are strictly positive, but that is a reason rather than an absence.
    ///
    /// `model.py:686` is the `hc_split_sinkhorn(...)` CALL; the expression itself is in
    /// `inference/kernel.py`, a different file.
    ///
    /// Added with `hc_sinkhorn_iters` and for the same reason: `launch_hc_pre` and
    /// `launch_hc_head_collapse` both take it and nothing supplied it. `f64` because JSON
    /// numbers are; the kernels narrow to f32 at the call, as `rms_norm_eps` already does.
    ///
    /// Required rather than defaulted for the reason a default of 0.0 would be *nearly*
    /// right — 1e-6 against a sigmoid output in (0, 1). It perturbs every gate by a hair, in
    /// the direction of less signal from every stream, uniformly across 43 layers. Small,
    /// systematic and unattributable is the worst shape a numeric error can have here.
    ///
    /// Carried with no reader yet, deliberately — see [`V4Config::hc_sinkhorn_iters`].
    pub hc_eps: f64,
}

/// The YaRN block. Required, and its `type` is checked — see the RoPE note in [`V4Config`].
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    pub beta_fast: u32,
    pub beta_slow: u32,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
    #[serde(rename = "type")]
    pub kind: String,
}

impl ArchConfig for V4Config {
    const ARCH: Arch = Arch::DeepseekV4;

    /// Cross-field checks. Each one guards a failure that produces text rather than an
    /// error, which is the whole hazard class of this port.
    ///
    /// **Split into five bodies rather than the reference's one**, on the precedent
    /// `glimmer_config::GlimmerTextConfig::validate` set and for the same reason: the
    /// reference's `validate` is a 165-line body whose only structure is comment headings,
    /// which is the shape the CodeScene gate refuses. The groups are the headings, and the
    /// ORDER is the reference's — a refusal message that moved would be a behaviour change.
    fn validate(&self) -> Result<()> {
        self.validate_layer_tables()?;
        self.validate_moe_counts()?;
        self.validate_attention_widths()?;
        self.validate_rope()?;
        self.validate_named_settings()?;
        // The FP4 group scale runs along the INPUT dim, so both expert input widths must
        // divide it exactly — `f4_groups` rounds up, and a ragged tail would give the
        // last group a scale covering fewer weights than the kernel assumes.
        // `self.hidden` is the `expert_in` argument, and on V4 those are equal.
        ensure_f4_group_aligned(self.hidden, self.moe_inter)
    }
}

impl V4Config {
    /// Load from the artifact's `manifest.json`, falling back to a bare `config.json` for
    /// reading a raw checkpoint.
    ///
    /// An inherent wrapper over [`crate::schema::load_config`] because the converter and its
    /// gates both spell it, and `V4Config::load(dir)` is what a reader greps for. GLM's
    /// `ModelConfig::load` and `GlimmerConfig::load` carry the same argument.
    pub fn load(dir: &str) -> Result<Self> {
        crate::schema::load_config(dir)
    }

    /// The two per-layer tables that are indexed by layer id, and the count that bounds one.
    fn validate_layer_tables(&self) -> Result<()> {
        // `compress_ratios` is indexed by layer id in `Attention.__init__`. The shipped
        // config carries 46 entries for 43 layers (the tail belongs to the mtp blocks), so
        // this is a floor, not an equality — but one entry short is an index panic mid-load
        // or, worse, a layer silently treated as ratio 0.
        ensure!(
            self.compress_ratios.len() >= self.n_layers,
            "compress_ratios has {} entries, need at least n_layers={}",
            self.compress_ratios.len(),
            self.n_layers
        );
        // Only 0 (sliding-window), 4 (compressor + indexer) and 128 (compressor only)
        // appear, and `Attention.__init__` branches on `== 4` exactly. An unseen ratio
        // would land in the "compressor, no indexer" arm by default; refuse instead.
        for (l, &r) in self.compress_ratios.iter().take(self.n_layers).enumerate() {
            ensure!(
                matches!(r, 0 | 4 | 128),
                "compress_ratios[{l}] = {r}; implemented: 0, 4, 128"
            );
        }
        ensure!(
            self.n_hash_layers <= self.n_layers,
            "num_hash_layers {} > n_layers {}",
            self.n_hash_layers,
            self.n_layers
        );
        Ok(())
    }

    /// The two expert counts, whose relation decides how many blocks a layer streams.
    fn validate_moe_counts(&self) -> Result<()> {
        ensure!(
            self.top_k <= self.n_experts,
            "top_k {} > n_experts {}",
            self.top_k,
            self.n_experts
        );
        // `MoE.__init__` asserts this outright; the shared expert is a single always-on
        // FFN in both the reference and rivoli's resident set.
        ensure!(
            self.n_shared == 1,
            "n_shared_experts {} != 1",
            self.n_shared
        );
        Ok(())
    }

    /// The attention geometry — the three relations a wrong value would reshape rather than
    /// refuse.
    fn validate_attention_widths(&self) -> Result<()> {
        ensure!(
            self.num_key_value_heads == 1,
            "num_key_value_heads {} != 1 — this is shared-KV MQA, one entry for all heads",
            self.num_key_value_heads
        );
        ensure!(
            self.qk_rope_head_dim < self.head_dim,
            "qk_rope_head_dim {} must be inside head_dim {}",
            self.qk_rope_head_dim,
            self.head_dim
        );
        // `wo_a` is viewed as `(o_groups, o_lora_rank, n_heads·head_dim/o_groups)`, so a
        // ragged split would reshape into the wrong stride rather than fail.
        ensure!(
            self.o_groups > 0 && (self.n_heads * self.head_dim).is_multiple_of(self.o_groups),
            "n_heads·head_dim ({}) not divisible by o_groups {}",
            self.n_heads * self.head_dim,
            self.o_groups
        );
        Ok(())
    }

    /// Both thetas and the YaRN block. A wrong RoPE is the archetype of this file's hazard
    /// class: every frequency stays plausible and every token lands at the wrong position.
    fn validate_rope(&self) -> Result<()> {
        // A zero theta is a silently wrong RoPE, not a crash — `ModelConfig::validate`
        // refuses one for the same reason. Both are live: ratio-0 layers use `rope_theta`,
        // the other 41 use `compress_rope_theta`.
        for (what, theta) in [
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
        ] {
            ensure!(theta > 0.0, "{what} {theta} must be positive");
        }
        ensure!(
            self.rope_scaling.kind == "yarn"
                && self.rope_scaling.factor > 0.0
                && self.rope_scaling.original_max_position_embeddings > 0,
            "rope_scaling type {:?} / factor {} / original {}: only YaRN is implemented, \
             and a zero original length disables the interpolation branch entirely",
            self.rope_scaling.kind,
            self.rope_scaling.factor,
            self.rope_scaling.original_max_position_embeddings
        );
        Ok(())
    }

    /// The named settings that change arithmetic without changing a shape, so that nothing
    /// downstream would refuse them.
    fn validate_named_settings(&self) -> Result<()> {
        ensure!(
            self.scoring_func == "sqrtsoftplus",
            "scoring_func {:?}: V4 is sqrtsoftplus. A wrong router affinity picks \
             plausible-but-wrong experts and never crashes",
            self.scoring_func
        );
        ensure!(
            self.index_topk > 0,
            "index_topk must be positive — at 0 the indexer selects no compressed blocks \
             and every compress_ratio == 4 layer silently falls back to sliding-window \
             attention"
        );
        // A zero or negative scale silently zeroes or flips every routed contribution
        // while the shared expert keeps working — degraded fluent text, not a crash.
        ensure!(
            self.routed_scale > 0.0,
            "routed_scaling_factor {} must be positive — it multiplies every routed \
             expert's weight (`Gate.forward`'s `weights *= route_scale`)",
            self.routed_scale
        );
        self.validate_swiglu_limit()?;
        ensure!(
            self.expert_dtype == "fp4",
            "expert_dtype {:?}: the .f4 repack reads e2m1 nibble pairs, and an fp8 export \
             of the same model would decode as noise",
            self.expert_dtype
        );
        self.validate_quantization()?;
        // Both mHC scalars are checked non-zero here rather than left to the kernels, and
        // the two cases differ. `hc_sinkhorn_iters == 0` IS refused downstream —
        // `kernels/linalg.hip:642` returns guard 1003 — so this is a restatement at the load
        // boundary, which names the field where the kernel names a code. `hc_eps == 0` is
        // refused by nothing: it is arithmetic, not a shape, and it perturbs every gate and
        // every Sinkhorn divide by 1e-6 uniformly, which no per-layer tolerance would read
        // as anything but depth. That one is the reason this pair is here at all.
        ensure!(
            self.hc_sinkhorn_iters > 0,
            "hc_sinkhorn_iters is 0 — `hc_split_sinkhorn`'s refinement passes would all be \
             skipped, on every layer (and `rivoli_hc_pre` refuses it with guard 1003)"
        );
        ensure!(
            self.hc_eps > 0.0,
            "hc_eps {} must be positive — it floors `hc_head`'s sigmoid gate AND every \
             row/column divide inside hc_split_sinkhorn",
            self.hc_eps
        );
        Ok(())
    }

    /// The SwiGLU clamp, checked in the f32 domain the KERNELS work in.
    ///
    /// `kernels/moe.hip:413` guards `!(swiglu_limit > 0.0f)` — spelled that way rather than
    /// `<= 0.0f` because NaN fails every comparison and `fminf(gt, NaN)` returns `gt`, so `<=`
    /// would admit the one value that silently disables the clamp. A bare `> 0.0` here misses
    /// BOTH narrowing failures, and they fail in opposite ways:
    ///
    ///   * UNDERFLOW. `as f32` rounds to nearest even, so any `0 < x <= 2^-150`
    ///     (7.006492321624085e-46) becomes `0.0f32` — and 7.1e-46 does NOT, it rounds up to the
    ///     min subnormal. The consequence is LOUD: guard 1006 at the first MoE layer of prefill.
    ///   * OVERFLOW. Float-to-float `as` saturates, so any finite `x > ~3.4e38` becomes
    ///     `f32::INFINITY`, which passes every `> 0.0` test — and `fminf(gt, inf) == gt`, so the
    ///     clamp becomes a NO-OP. That is exactly `v4oracle::Defect::SwigluUnclamped`, SILENT,
    ///     and wrong all the way to the output.
    ///
    /// The silent direction is the one worth the check, which is why this is `is_finite()` and
    /// not just a positivity test. Both verified numerically 2026-08-05.
    ///
    /// **Not routed through [`crate::schema::ensure_f32_positive`]**, which applies the same
    /// narrowing rule to Glimmer's and K3's epsilons: that helper takes a slice of pairs and
    /// produces a message about "the domain the kernels work in", and this one has to name the
    /// clamp, the guard code and the reason a bare positivity test is insufficient. Folding it
    /// in would trade this argument for one shared line — and the argument is the check.
    fn validate_swiglu_limit(&self) -> Result<()> {
        let narrowed = self.swiglu_limit as f32;
        ensure!(
            narrowed > 0.0 && narrowed.is_finite(),
            "swiglu_limit {} narrows to {narrowed} in f32, which is the domain the MoE kernel's \
             clamp works in. V4's SwiGLU is CLAMPED (10.0 in the shipped config) and rivoli's is \
             not, so a zero here is a rejected launch and an infinity is silently unclamped \
             arithmetic",
            self.swiglu_limit
        );
        Ok(())
    }

    /// The attention-tensor quantization scheme, which the resident path assumes rather than
    /// discovers.
    fn validate_quantization(&self) -> Result<()> {
        let q = &self.quantization_config;
        ensure!(
            q.fmt == "e4m3" && q.scale_fmt == "ue8m0",
            "quantization_config {:?}/{:?}: the resident path decodes e4m3 weights with \
             ue8m0 block scales",
            q.fmt,
            q.scale_fmt
        );
        ensure!(
            q.weight_block_size == [crate::quant::FP8_BLOCK; 2],
            "quantization_config.weight_block_size {:?} != [{}, {}]",
            q.weight_block_size,
            crate::quant::FP8_BLOCK,
            crate::quant::FP8_BLOCK
        );
        Ok(())
    }

    /// `compress_ratios[layer]`, bounds-checked against `n_layers` rather than against the
    /// vector — the vector is longer (the mtp tail), and reading past `n_layers` would
    /// return an mtp block's ratio for a main-path layer.
    pub fn compress_ratio(&self, layer: usize) -> Result<usize> {
        ensure!(layer < self.n_layers, "layer {layer} >= {}", self.n_layers);
        // In bounds by the `ensure!` above AND by `validate_layer_tables`, which requires the
        // vector to be at least `n_layers` long. `get` rather than `[]` anyway: the workspace
        // lint table does not deny `indexing_slicing`, so the panic would be real, and this
        // function's whole contract is that an out-of-range layer is an error and not a crash.
        self.compress_ratios
            .get(layer)
            .copied()
            .ok_or_else(|| anyhow::anyhow!("compress_ratios is shorter than n_layers"))
    }

    /// Whether this layer carries `attn.compressor.*` (ratio != 0). Also selects
    /// `compress_rope_theta` + YaRN over the base theta with YaRN off.
    pub fn layer_has_compressor(&self, layer: usize) -> Result<bool> {
        Ok(self.compress_ratio(layer)? != 0)
    }

    /// Whether this layer carries `attn.indexer.*`. `Attention.__init__` builds one only
    /// at ratio EXACTLY 4 — 21 of the 41 compressor layers in the shipped checkpoint.
    pub fn layer_has_indexer(&self, layer: usize) -> Result<bool> {
        Ok(self.compress_ratio(layer)? == 4)
    }

    /// Whether this layer's gate selects experts from `tid2eid` instead of from the
    /// scores. Such a layer has `ffn.gate.tid2eid` and no `ffn.gate.bias`.
    pub fn layer_routes_by_hash(&self, layer: usize) -> bool {
        layer < self.n_hash_layers
    }

    // No `experts_per_layer` twin of `ModelConfig`'s. On V4 the two kinds of expert are
    // not interchangeable: `top_k` routed experts are FP4 and stream from NVMe, while the
    // one shared expert is fp8 e4m3 and is resident. A single `top_k + n_shared` count is
    // right for a MoE launch and WRONG for per-token stream traffic, and it is the traffic
    // number this port keeps needing — so the two are spelled out separately at each use.
}
