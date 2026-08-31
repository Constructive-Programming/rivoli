//! Qwen3.8-Flash-Next's config schema — one architecture, one file, per the rule that
//! per-model config types stay separate ([`crate::schema`]'s header carries the argument).
//!
//! **The nesting is not optional here, and it is not the K3 nesting either.** K3 wraps a text
//! model whose `text_config` re-declares BOTH identity fields; this file's `text_config`
//! declares `model_type` alone (`qwen4_exp_text`) and no `architectures`, and the wrapper's
//! `vision_config.model_type` is the ROOT string verbatim (`qwen4_exp`). Five spellings are in
//! play and only the two at the document root resolve — the table is
//! `docs/investigations/qwen-flash-next-port.md` ("Identity decisions already fixed") and the
//! reading is enforced by [`crate::schema`]'s root-only, both-fields sniff. What this file adds
//! is the other end of the same check: the nested `model_type` must be the TEXT half's
//! spelling, so a descent that landed in `vision_config` — which carries the root string and
//! nothing else — is a refusal rather than a config with 27 layers of vision dimensions.
//!
//! **Every field is REQUIRED**, for [`crate::schema`]'s reason: a defaulted dimension does not
//! crash, it produces fluent wrong text. The two exceptions each say so at their declaration
//! and both are shapes this repo already has a precedent for — an `Option` whose `None` is the
//! FIRST-PARTY default, cited to the reference line that sets it, never to a Rust default that
//! happens to agree today.
//!
//! **What is deliberately NOT bound.** `vision_config` (v1 is text-only, owner scope 2026-08-30
//! — its 21 tensor families are excluded BY NAME in [`crate::census::qwen`], which is a
//! stronger statement than a schema that cannot see them); the `mtp` sub-dict beyond its layer
//! count (the 35 MTP families are excluded the same way); `initializer_range`,
//! `attention_dropout`, `router_aux_loss_coef`, `output_router_logits` (training-side, and none
//! of them changes a decode); `pad_token_id` (`null` in the shipped file). Each absence is a
//! decision, and none of them is a field this port reads from somewhere else.
//!
//! **Bare `usize` for dimensions, not newtypes, and the argument is consistency.** The house
//! rule asks for newtypes over bytes/tokens/layers; all four sibling configs (`glm_config`,
//! `v4_config`, `k3_config`, `glimmer_config`) declare their dimensions as bare `usize`, and
//! the shared validators they hand values to ([`crate::schema::ensure_f32_positive`],
//! `ensure_group_aligned`) take `usize`. A fifth config that newtyped only its own fields would
//! either fork those helpers or unwrap at every call — so the newtype belongs to a change that
//! moves all five, not to this file.

/// The derived widths — `hc_stream`, `gdn_qkv_width`, `ple_host_layer` and their siblings. A
/// sibling module under the 800-line soft cap, split by JOB: this file declares the schema and
/// refuses a document; that one answers how wide a tensor is. The methods stay inherent on
/// [`QwenTextConfig`], so no call site knows about the split.
pub mod geometry;

use crate::arch::Arch;
use crate::quant::FP8_BLOCK;
use crate::schema::{ArchConfig, ensure_f32_positive};
use anyhow::{Result, ensure};
use serde::Deserialize;

/// The checkpoint's spelling for a layer that runs GatedDeltaNet.
pub const LINEAR_ATTENTION: &str = "linear_attention";

/// The checkpoint's spelling for a layer that runs QSA — **and it is an ALIAS.**
///
/// `Qwen4ExpTextConfig.__post_init__` rewrites every `full_attention` entry to
/// `qwen_sparse_attention` before `validate_architecture` looks at the array, with the comment
/// "The real checkpoint contains \"full_attention\" entries for layers that are actually using
/// an indexer" (CFG:180-184; `docs/reference/qwen-architecture.md` §3). So these 12 layers RUN
/// THE INDEXER, and a port that reads the field at face value builds dense global attention on
/// them — which is bit-identical below 2051 cached tokens and therefore invisible to the free
/// oracle, the planned parity window and the smoke decode cell alike (trap T19).
///
/// The constant is named for what the FILE says because that is what serde must match; every
/// accessor below is named for what the layer DOES.
pub const FULL_ATTENTION_ALIAS: &str = "full_attention";

/// `rope_parameters` — carried WHOLE, for the reason `V4Config` carries all of RoPE: a type
/// exposing half of it invites the arm to take the rest from reference defaults.
///
/// It also carries the only cross-check the file offers for free: `partial_rotary_factor`
/// appears BOTH here and directly on `text_config`, so [`QwenTextConfig::validate`] can compare
/// two independent statements of the same number instead of trusting one.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenRopeParameters {
    /// `default`. **Not `yarn`** — YaRN is opt-in, absent from every released config, and its
    /// factor-4 form (including the non-unit `attention_factor` 1.138629436111989 a port
    /// forgets) is pinned in `qwen-architecture.md` §7 as a milestone, not a guess.
    pub rope_type: String,
    /// 1e7.
    pub rope_theta: f64,
    /// 0.25 → `rotary_dim = 64` of `head_dim` 256, the FIRST 64 dims, paired split-half
    /// `(i, i+32)`. Asserted equal to the sibling on `text_config`.
    pub partial_rotary_factor: f64,
    /// True. Inert for text-only v1 — 2-D `position_ids` expand to 3 identical grids, so
    /// `apply_interleaved_mrope` overwrites `freqs[0]` slices with identical values
    /// (`qwen-architecture.md` §7) — and asserted anyway: it is vision that makes it bite, and
    /// a `false` here would be a different rotary layout on the day vision lands.
    pub mrope_interleaved: bool,
    /// `[11, 11, 10]`, summing to the 32 frequencies of a 64-wide rotary. The sum is the check;
    /// the split is the vision milestone's.
    pub mrope_section: Vec<usize>,
}

/// Qwen3.8-Flash-Next, as its `config.json` ships it: a `Qwen4ExpForConditionalGeneration`
/// multimodal wrapper around the text model.
///
/// **Hold a [`QwenConfig`], not a [`QwenTextConfig`]** — the nesting is what separates them, and
/// only the outer type is evidence that [`crate::schema::parse_config`] ran both the
/// architecture check and `validate`. `K3Config`'s doc carries the same warning for the same
/// reason; here it is sharper, because a bare `QwenTextConfig` could also be deserialized out of
/// `vision_config` if that block ever grew the text half's keys.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenConfig {
    /// The ROOT `model_type`, `qwen4_exp`. Carried so `validate` can assert the WRAPPER
    /// spelling as well as the nested one: `parse_config` resolves it through
    /// [`crate::arch::from_manifest_str`], which maps four other strings to four other
    /// architectures, so "this resolved to `Arch::QwenFlashNext`" and "this document says
    /// `qwen4_exp`" are not the same claim once a manifest can be hand-edited.
    pub model_type: String,
    /// The other half of the root pair, `["Qwen4ExpForConditionalGeneration"]`. Both root fields
    /// are REQUIRED by the sniff (a sub-config promoted to a document carries `model_type`
    /// alone), and requiring them here too is what makes the manifest a converter writes
    /// re-parse exactly as the source did.
    pub architectures: Vec<String>,
    #[serde(rename = "text_config")]
    pub text: QwenTextConfig,
}

/// The `text_config` dict — every architecture dimension in the model.
#[derive(Debug, Clone, Deserialize)]
pub struct QwenTextConfig {
    /// `qwen4_exp_text` — the TEXT half's spelling, which the root-only sniff must not accept
    /// and which this struct must find. `vision_config.model_type` is the root string verbatim,
    /// so this field is the whole difference between "descended into the text model" and
    /// "descended into the vision tower".
    pub model_type: String,

    // --- trunk widths.
    /// 248320, padded. The n-gram hash's `half_bound` is derived from it
    /// (`qwen-architecture.md` §5), so a wrong value moves every hashed row.
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    /// 2560 — and the hyper-connection stream is FOUR of these wide; see [`Self::hc_stream`].
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    /// 48, as `12 x (3 x GatedDeltaNet -> 1 x QSA)`.
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    /// RMSNorm epsilon, 1e-6. In [`Self::validate`]'s f32 loop because the kernels narrow it:
    /// a `1e-46` eps passes an f64 positivity test and reaches every RMSNorm as `0.0f32`.
    pub rms_norm_eps: f64,
    /// `bfloat16` — the trunk dtype. Note the LEVEL: like K3 and unlike Glimmer, this sits
    /// inside `text_config`.
    pub dtype: String,
    /// `silu`.
    pub hidden_act: String,
    /// `float32`. The GDN recurrent state's accumulation dtype, and it is load-bearing rather
    /// than a hint: the state is 48 value heads x 128 x 128 per layer over 36 layers, and
    /// carrying it in bf16 would halve 108.0 MiB of state at the cost of the decay recurrence's
    /// precision. Asserted so a checkpoint that changed it is a refusal, not a silent change of
    /// arithmetic.
    pub mamba_ssm_dtype: String,
    /// False. A `true` would mean bias tensors on every projection — 0 of the census's 109
    /// families is a `*_proj.bias` on the text side, so this and the census agree.
    pub attention_bias: bool,
    /// False — a separate `lm_head`, which the census carries as its own 1,271,398,400-byte row.
    pub tie_word_embeddings: bool,
    /// 262144 native. YaRN beyond it is out of v1 scope.
    pub max_position_embeddings: usize,

    // --- the layer partition. TWO independent statements, and `validate` compares them.
    /// 48 entries: 36 [`LINEAR_ATTENTION`] + 12 [`FULL_ATTENTION_ALIAS`] at indices 3, 7, …, 47.
    ///
    /// `Vec<String>` rather than an enum, and the choice is deliberate: the refusal for an
    /// unrecognised spelling must be able to QUOTE it. A `#[derive(Deserialize)]` enum would
    /// fail inside serde with "unknown variant", naming the variant but not the layer index —
    /// and which layer is the actionable half when a checkpoint grows a third family.
    pub layer_types: Vec<String>,
    /// 4 — the group size of `3 x GatedDeltaNet -> 1 x QSA`. The reference validates it against
    /// nothing; [`QwenTextConfig::validate_layer_types`] validates it against `layer_types`.
    pub full_attention_interval: usize,

    // --- QSA (the 12 `full_attention`-spelled layers) and its indexer.
    /// 24 query heads. `q_proj` is [12288, 2560] = **2 x 24 x 256**; see [`Self::qsa_q_width`].
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    /// 2 — GQA, not the MQA `V4Config` asserts `== 1` and not K3's per-query-head MLA. Pinned
    /// as a divisibility relation against [`Self::n_heads`] rather than as the literal.
    pub num_key_value_heads: usize,
    /// 256, and NOT `hidden / n_heads` (2560/24 is not an integer). The two are independent
    /// here, which is the trap every fixture in this tree keeps visible.
    pub head_dim: usize,
    /// 0.25 → the FIRST 64 of 256 dims rotate. Duplicated inside `rope_parameters`; `validate`
    /// compares them.
    pub partial_rotary_factor: f64,
    pub rope_parameters: QwenRopeParameters,
    /// `sigmoid`. The output-gate activation, shared by GDN's `in_proj_z` gate and QSA's
    /// `q_proj` gate half. One of the nine named defect classes (`qwen-architecture.md`'s traps
    /// table), and it changes arithmetic without changing a shape.
    pub output_gate_type: String,
    /// 2048 tokens — 512 blocks x `indexer_compress_ratio` 4.
    pub indexer_budget: usize,
    /// 4 TOKENS per block. `block_topk = budget / ratio` must divide exactly (CFG:223-224).
    pub indexer_compress_ratio: usize,
    /// 128, half of which is partial-RoPEd, "matching the rotary dimension used in the core
    /// attention module" (TR p.6).
    pub indexer_head_dim: usize,
    /// 4 query heads.
    pub indexer_n_heads: usize,
    /// 1 shared key head — MQA. `index_qk_proj` is one [640, 2560] matrix split
    /// `[4*128 | 1*128]`; see [`Self::indexer_qk_width`].
    pub indexer_kv_heads: usize,

    // --- GatedDeltaNet (the 36 `linear_attention` layers). ASYMMETRIC head counts.
    /// 16 QK heads. Asymmetric against the 48 V heads, which is why the four `linear_*` fields
    /// are all bound: deriving any of them from another gives a self-consistent artifact with
    /// `in_proj_qkv` mis-split.
    pub linear_num_key_heads: usize,
    /// 128.
    pub linear_key_head_dim: usize,
    /// 48 V heads — three times the QK heads, and the count that owns the recurrent state.
    pub linear_num_value_heads: usize,
    /// 128, so each V head owns a 128-key x 128-value state.
    pub linear_value_head_dim: usize,
    /// 4, with the newest tap LAST and SiLU after (`qwen-architecture.md` §2).
    pub linear_conv_kernel_dim: usize,

    // --- MoE, on EVERY layer.
    /// 512 routed experts per layer, all 48 layers.
    pub num_experts: usize,
    /// 10 routed + 1 shared.
    #[serde(rename = "num_experts_per_tok")]
    pub top_k: usize,
    /// 640.
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    /// 640 — equal to [`Self::moe_inter`] in this checkpoint (`qwen-architecture.md` §6), and
    /// bound separately because they are separate keys and a shared expert at another width
    /// would change one tensor set and no count.
    pub shared_expert_intermediate_size: usize,
    /// **Absent from the shipped config, and its FIRST-PARTY default is `true`** (CFG:163), so
    /// the 10 kept probabilities are renormalised to sum to 1 *after* the top-k (MOD:912-913).
    ///
    /// `Option<bool>` rather than a defaulted `bool`, and this is the module header's rule read
    /// forward rather than broken: a Rust `bool` default would be OUR value dressed as the
    /// checkpoint's, while `None` says "the file does not state it" and lets `validate` name the
    /// reference line it is deferring to. `Some(false)` is REFUSED — this port renormalises.
    ///
    /// **The `#[serde(default)]` below is documentation, not the mechanism, and that is MEASURED**:
    /// removing it and re-running `the_shipped_config_parses_and_matches_the_architecture_doc` left
    /// the test green (2026-08-31, plant P20 — exit 0 after a real rebuild), because serde already
    /// treats an `Option<T>` field as absent-tolerant. Kept as a statement of intent at the one
    /// declaration where absence is expected; do not read it as an enforced property.
    #[serde(default)]
    pub norm_topk_prob: Option<bool>,

    // --- hyper-connections: a 4-wide residual stream through all 48 layers.
    /// Four branches. Every sublayer is wrapped pre/post, so the stream is `hc_count * hidden`
    /// wide — 10240, which is the width of every `hc_norm`, `block_inject_weight` and
    /// `input_mix_weight_*` row in the census.
    pub hc_count: usize,
    /// 320 = `hidden / 8`. Qwen's own Gated Residual, NOT Sinkhorn mHC
    /// (`qwen-architecture.md` §4).
    pub hc_lowrank: usize,

    // --- hashed n-gram embeddings (PLE), hosted on ONE layer.
    /// 3 → bigram + trigram, `ngram_heads = (3-1) * heads_per_ngram`.
    pub ngram_size: usize,
    /// 8.
    pub heads_per_ngram: usize,
    /// The prime floor, 20000000: each head's vocabulary is a distinct prime strictly above
    /// `ngram_vocab_size_base - 1`; the 16 primes and the 3 multipliers are re-derived by
    /// [`crate::census::qwen::ngram_hash`] and byte-compared against the checkpoint's own I64
    /// buffers.
    pub ngram_vocab_size_base: usize,
    /// 128 — the padding modulus for the summed head vocabularies.
    pub make_ngram_vocab_size_divisible_by: usize,
    /// 128 shards of exactly `padded_rows / 128` rows each. The shard split is a
    /// checkpoint-side convention with no counterpart in the reference module (OPEN-2), so the
    /// converter treats the shards as ordered row ranges and asserts they tile the table.
    pub split_ngram_parts: usize,
    /// 2560, gathered as `ngram_heads` x `ple_embed_dim / ngram_heads` = 16 x 160.
    pub ple_embed_dim: usize,
    /// 4, with `dilation = ngram_size` = 3, so token `t` reads taps `{t-9, t-6, t-3, t}`.
    pub ple_conv_kernel_size: usize,
    /// **ONE-INDEXED.** `[2]` in the shipped file, and the reference tests
    /// `layer_idx + 1 in ple_layer_ids` (MOD:1202), so the host is `layer_idx` **1** — the
    /// SECOND layer. The checkpoint's only PLE tensors sit under `layers.1.`, and a converter
    /// that emits `layers.2.ple.*` must fail its census (trap T12). [`Self::ple_host_layer`] is
    /// the one place the conversion happens.
    pub ple_layer_ids: Vec<usize>,

    // --- token ids, and the MTP head this port excludes.
    /// 248044 — and the n-gram history is padded with THIS scalar (MOD:1076), while
    /// `_shift_right_ignore_eos` resets the context at every occurrence, so an n-gram never
    /// spans a document boundary. Note that `generation_config.json`'s `eos_token_id: [248046, 248044]`
    /// is a DIFFERENT object and the hash uses this one (`qwen-architecture.md` §5).
    pub eos_token_id: usize,
    /// 248044 — the same id as the EOS in this checkpoint.
    pub bos_token_id: usize,
    /// One. The census's 35 `mtp.*` families are that one layer's worth (3,101 tensors,
    /// 2,698,026,496 B) and are excluded BY NAME for v1; a different value means a different
    /// family set, so this is asserted rather than ignored.
    pub mtp_num_hidden_layers: usize,
    /// False — the MTP head shares the trunk's embeddings. Bound with its sibling above so the
    /// excluded set's shape is stated by the config rather than only by the census.
    pub mtp_use_dedicated_embeddings: bool,
}

impl ArchConfig for QwenConfig {
    const ARCH: Arch = Arch::QwenFlashNext;

    fn validate(&self) -> Result<()> {
        // The WRAPPER pair, asserted as one `ensure!` for `K3TextConfig::validate_descent`'s
        // reason: which half disagrees is not the interesting fact, "this is not the document we
        // think it is" is, and a reader holding the file wants both spellings in front of them.
        ensure!(
            self.model_type == "qwen4_exp"
                && self.architectures == ["Qwen4ExpForConditionalGeneration".to_string()],
            "the document ROOT declares {:?} / {:?} — Qwen3.8-Flash-Next's wrapper is \
             \"qwen4_exp\" / [\"Qwen4ExpForConditionalGeneration\"]. Note that \
             `vision_config.model_type` is the root string VERBATIM, so a promoted vision block \
             is the shape this refusal is written to catch",
            self.model_type,
            self.architectures
        );
        self.text.validate()
    }
}

impl QwenConfig {
    /// Load from the artifact's `manifest.json`, falling back to a bare `config.json` for reading
    /// a raw checkpoint.
    ///
    /// An inherent wrapper over [`crate::schema::load_config`] because the converter and its gates
    /// both spell it, and `QwenConfig::load(dir)` is what a reader greps for.
    ///
    /// **The type parameter is spelled `QwenConfig` rather than `Self`, and that is gate work.**
    /// This is the FIFTH identical `load` wrapper in the crate; `K3Config::load`'s note records
    /// growing a `::<Self>` turbofish for exactly this reason in 2026-08-16 ("1 clone reported
    /// without it, 0 with"), and copying that spelling here made jscpd report the K3/V4 pair
    /// instead — three regions in one group, so breaking two of the three is what it takes. The
    /// concrete name also reads better at the one place it matters, which is a reader deciding
    /// which of five configs a directory is being loaded as.
    pub fn load(dir: &str) -> Result<Self> {
        crate::schema::load_config::<QwenConfig>(dir)
    }
}

impl QwenTextConfig {
    /// Cross-field checks. Split into named groups for `V4Config`/`K3TextConfig`'s reason: one
    /// body whose only structure is comment headings is the shape the CodeScene gate refuses,
    /// and the groups are those headings.
    pub fn validate(&self) -> Result<()> {
        self.validate_descent()?;
        self.validate_widths()?;
        self.validate_layer_types()?;
        self.validate_attention_geometry()?;
        self.validate_linear_attention()?;
        self.validate_hyper_connections()?;
        self.validate_moe()?;
        self.validate_ple()?;
        self.validate_named_settings()?;
        // Every f64 the kernels narrow to f32, checked in the f32 domain rather than only in
        // the f64 the JSON carries: underflow collapses the value and overflow passes any bare
        // `> 0.0` test. `rms_norm_eps` is in this list because the engine does
        // `cfg.rms_norm_eps as f32` at every RMSNorm — the k3 review's finding, inherited.
        ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            (
                "rope_parameters.rope_theta",
                self.rope_parameters.rope_theta,
            ),
            ("partial_rotary_factor", self.partial_rotary_factor),
        ])
    }

    /// The descent check: this is `text_config` and not one of its siblings.
    fn validate_descent(&self) -> Result<()> {
        ensure!(
            self.model_type == "qwen4_exp_text",
            "text_config declares model_type {:?} — the text half's spelling is \
             \"qwen4_exp_text\". `vision_config.model_type` is the ROOT string (\"qwen4_exp\") \
             verbatim, so this refusal is what separates a descent into the text model from a \
             descent into the vision tower",
            self.model_type
        );
        Ok(())
    }

    /// Every width is positive. A zero passes every divisibility check below (0 is a multiple of
    /// anything) and then sizes an expert row, an arena stride and a GEMV `dim` to nothing.
    fn validate_widths(&self) -> Result<()> {
        for (what, dim) in [
            ("vocab_size", self.vocab),
            ("hidden_size", self.hidden),
            ("num_hidden_layers", self.n_layers),
            ("num_attention_heads", self.n_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("indexer_budget", self.indexer_budget),
            ("indexer_compress_ratio", self.indexer_compress_ratio),
            ("indexer_head_dim", self.indexer_head_dim),
            ("indexer_n_heads", self.indexer_n_heads),
            ("indexer_kv_heads", self.indexer_kv_heads),
            ("linear_num_key_heads", self.linear_num_key_heads),
            ("linear_key_head_dim", self.linear_key_head_dim),
            ("linear_num_value_heads", self.linear_num_value_heads),
            ("linear_value_head_dim", self.linear_value_head_dim),
            ("linear_conv_kernel_dim", self.linear_conv_kernel_dim),
            ("num_experts", self.num_experts),
            ("num_experts_per_tok", self.top_k),
            ("moe_intermediate_size", self.moe_inter),
            (
                "shared_expert_intermediate_size",
                self.shared_expert_intermediate_size,
            ),
            ("hc_count", self.hc_count),
            ("hc_lowrank", self.hc_lowrank),
            ("ngram_size", self.ngram_size),
            ("heads_per_ngram", self.heads_per_ngram),
            ("ngram_vocab_size_base", self.ngram_vocab_size_base),
            (
                "make_ngram_vocab_size_divisible_by",
                self.make_ngram_vocab_size_divisible_by,
            ),
            ("split_ngram_parts", self.split_ngram_parts),
            ("ple_embed_dim", self.ple_embed_dim),
            ("ple_conv_kernel_size", self.ple_conv_kernel_size),
            ("max_position_embeddings", self.max_position_embeddings),
            ("mtp_num_hidden_layers", self.mtp_num_hidden_layers),
        ] {
            // The message names the CONSEQUENCE rather than restating the value, for
            // `K3TextConfig::validate_widths`'s mechanical reason as well as its argued one: an
            // identical (pointer, message) row in two gate files is a token run jscpd reports.
            ensure!(
                dim > 0,
                "{what} = 0 would size an expert row, a hyper-connection stride or a GEMV dim \
                 to nothing"
            );
        }
        Ok(())
    }

    /// **`layer_types` against `full_attention_interval`, which is the check the ALIAS makes
    /// necessary.**
    ///
    /// The array and the interval are two independent statements of the same partition, and the
    /// reference validates neither against the other. Every failure here is a layer running the
    /// wrong attention family — arithmetic, not a shape, so nothing downstream sees it — and the
    /// two families take the same `[hc_stream]` input and return the same shape.
    ///
    /// The interval is read as "one QSA layer closes each group of `interval`", i.e.
    /// `(idx + 1) % interval == 0`, which puts the 12 QSA layers at 3, 7, …, 47. That is the
    /// pattern `12 x (3 x GDN -> 1 x QSA)` states and the shipped array holds; the OTHER
    /// reading (`idx % interval == 0`, layers 0, 4, …, 44) is refused by comparing against the
    /// array rather than by preferring one, which is the whole point of having both.
    fn validate_layer_types(&self) -> Result<()> {
        ensure!(
            self.layer_types.len() == self.n_layers,
            "layer_types has {} entries against num_hidden_layers {} — the array is the \
             per-layer family map and a short one leaves the tail unassigned",
            self.layer_types.len(),
            self.n_layers
        );
        ensure!(
            self.full_attention_interval > 0,
            "full_attention_interval is 0; it is the group size of the \
             `3 x GatedDeltaNet -> 1 x QSA` pattern"
        );
        for (idx, kind) in self.layer_types.iter().enumerate() {
            // FULL NAMES, both of them. The reference admits `linear_attention` and
            // `qwen_sparse_attention` only, having already rewritten `full_attention` into the
            // second; a checkpoint spelling anything else is a family this port has no path for.
            ensure!(
                kind == LINEAR_ATTENTION || kind == FULL_ATTENTION_ALIAS,
                "layer_types[{idx}] is {kind:?} — this checkpoint's own spellings are \
                 {LINEAR_ATTENTION:?} and {FULL_ATTENTION_ALIAS:?} (the second an ALIAS for \
                 the layers that RUN THE INDEXER, rewritten to `qwen_sparse_attention` by the \
                 reference before it validates)"
            );
            let by_interval = (idx + 1).is_multiple_of(self.full_attention_interval);
            ensure!(
                (kind == FULL_ATTENTION_ALIAS) == by_interval,
                "layer_types[{idx}] is {kind:?} but full_attention_interval {} makes it \
                 {}. The array and the interval are two statements of one partition and \
                 nothing upstream compares them; a disagreement runs the wrong attention \
                 family on this layer, which is arithmetic and not a shape",
                self.full_attention_interval,
                if by_interval {
                    FULL_ATTENTION_ALIAS
                } else {
                    LINEAR_ATTENTION
                }
            );
        }
        // Anti-vacuity, and it is not implied by the loop: an array of 48 `linear_attention`
        // entries with `full_attention_interval` 49 satisfies every row above and leaves this
        // port with no QSA layer, no indexer and no KV cache at all.
        let sparse = self.sparse_attention_layers();
        ensure!(
            sparse > 0 && sparse < self.n_layers,
            "layer_types assigns {sparse} of {} layers to {FULL_ATTENTION_ALIAS} — this model \
             is a hybrid and needs both families",
            self.n_layers
        );
        Ok(())
    }

    /// QSA's head geometry and the partial-rotary width.
    fn validate_attention_geometry(&self) -> Result<()> {
        // GQA: the query heads must group onto the KV heads exactly. Pinned as the relation
        // rather than as `num_key_value_heads == 2`, so a re-sharded checkpoint is admitted and
        // a ragged grouping is not — and note this is neither V4's `== 1` MQA nor K3's
        // per-query-head MLA, both of which would pass a copied check from those files.
        ensure!(
            self.num_key_value_heads > 0 && self.n_heads.is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} does not group onto num_key_value_heads {} — QSA is GQA \
             here, and a ragged grouping silently drops or duplicates heads",
            self.n_heads,
            self.num_key_value_heads
        );
        // The two independent statements of the partial-rotary factor. The file carries it twice
        // and nothing upstream compares them; the arm reads one of the two.
        ensure!(
            self.partial_rotary_factor == self.rope_parameters.partial_rotary_factor,
            "partial_rotary_factor is {} on text_config and {} inside rope_parameters — the \
             file states it twice and the arm reads one of them",
            self.partial_rotary_factor,
            self.rope_parameters.partial_rotary_factor
        );
        let rot = self.rotary_dim();
        ensure!(
            rot > 0 && rot <= self.head_dim && rot.is_multiple_of(2),
            "partial_rotary_factor {} x head_dim {} gives a rotary width of {rot}; it must be a \
             positive EVEN width no wider than the head — `rotate_half` pairs `i` with \
             `i + {}` inside it",
            self.partial_rotary_factor,
            self.head_dim,
            rot / 2
        );
        // `mrope_section` sums to the FREQUENCY count, which is half the rotary width:
        // `inv_freq` has `rotary_dim / 2` entries and `emb = cat(freqs, freqs)`. Inert for
        // text-only v1 and asserted anyway — see the field.
        let sections: usize = self.rope_parameters.mrope_section.iter().sum();
        ensure!(
            sections == rot / 2,
            "mrope_section {:?} sums to {sections}, but a {rot}-wide partial rotary has {} \
             frequencies",
            self.rope_parameters.mrope_section,
            rot / 2
        );
        // The indexer's budget must split into whole blocks: `block_topk = budget / ratio`
        // floors in MOD:622 while TR Eq. 16 ceils, and CFG:223-224 requires exact divisibility,
        // so the two agree here and ONLY here.
        ensure!(
            self.indexer_budget
                .is_multiple_of(self.indexer_compress_ratio),
            "indexer_budget {} is not a multiple of indexer_compress_ratio {} — the reference \
             FLOORS `block_topk` where the tech report CEILS it, and they agree only when this \
             divides",
            self.indexer_budget,
            self.indexer_compress_ratio
        );
        Ok(())
    }

    /// GatedDeltaNet's asymmetric heads, checked as the widths the checkpoint's tensors have.
    ///
    /// The asymmetry is the point: 16 QK heads and 48 V heads, so `in_proj_qkv` is
    /// `2*16*128 + 48*128 = 10240` wide and `in_proj_z`/`out_proj` are `48*128 = 6144`. Deriving
    /// either width from a single head count produces a self-consistent artifact with the qkv
    /// split wrong — and both halves still have the right total length.
    fn validate_linear_attention(&self) -> Result<()> {
        ensure!(
            self.linear_key_head_dim == self.linear_value_head_dim,
            "linear_key_head_dim {} != linear_value_head_dim {} — the GDN state is \
             `key_dim x value_dim` per V head, and `linear_attn.norm.weight` is ONE vector of \
             the value width, so a split here needs a different norm than the checkpoint ships",
            self.linear_key_head_dim,
            self.linear_value_head_dim
        );
        ensure!(
            self.linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads),
            "linear_num_value_heads {} does not group onto linear_num_key_heads {} — GDN shares \
             each QK head across a whole group of V heads",
            self.linear_num_value_heads,
            self.linear_num_key_heads
        );
        ensure!(
            self.gdn_qkv_width() == 2 * self.gdn_key_width() + self.gdn_value_width(),
            "the GDN qkv width does not decompose into q + k + v"
        );
        Ok(())
    }

    /// The 4-wide residual stream, and the low-rank width it is mixed through.
    fn validate_hyper_connections(&self) -> Result<()> {
        ensure!(
            self.hc_count > 1,
            "hc_count is {} — at 1 there is no hyper-connection stream, and every `hc_norm`, \
             `block_inject_weight` and `input_mix_weight_*` tensor in this checkpoint is sized \
             `hc_count x hidden` wide",
            self.hc_count
        );
        ensure!(
            self.hc_lowrank > 0 && self.hc_lowrank <= self.hc_stream(),
            "hc_lowrank {} is not a rank BELOW the {}-wide hyper-connection stream — \
             `input_mix_weight_down` is [hc_lowrank, hc_count x hidden] and \
             `input_mix_weight_up` is its transpose",
            self.hc_lowrank,
            self.hc_stream()
        );
        Ok(())
    }

    /// Expert counts, the shared expert's width, and the two fp8 block alignments.
    fn validate_moe(&self) -> Result<()> {
        ensure!(
            self.top_k > 0 && self.top_k <= self.num_experts,
            "num_experts_per_tok {} is not in 1..={}",
            self.top_k,
            self.num_experts
        );
        ensure!(
            self.shared_expert_intermediate_size == self.moe_inter,
            "shared_expert_intermediate_size {} != moe_intermediate_size {} — they are equal in \
             this checkpoint (`qwen-architecture.md` §6) and the shared MLP's tensors are sized \
             from the second; a difference changes one tensor set and no count",
            self.shared_expert_intermediate_size,
            self.moe_inter
        );
        ensure!(
            self.norm_topk_prob != Some(false),
            "norm_topk_prob is false — this port renormalises the {} kept probabilities to sum \
             to 1 after the top-k, which is what the reference's own default (CFG:163, the \
             value this checkpoint omits) does. Turning it off rescales every routed \
             contribution and changes no shape",
            self.top_k
        );
        // Both routed-expert widths against the fp8 tile. NOT a `debug_assert` and not left to
        // `fp8_blocks`' `div_ceil`: a ragged edge is representable — the last tile's scale
        // simply covers fewer weights — so this is a claim about THIS checkpoint's widths, and
        // it is the claim `weight_scale_inv`'s [out/128, in/128] grid shape rests on. The
        // n-gram table is deliberately NOT here: its 160-wide rows do not divide 128 at all,
        // which is the structural half of why it carries a per-TENSOR scale instead of a grid.
        for (what, dim) in [
            ("hidden_size", self.hidden),
            ("moe_intermediate_size", self.moe_inter),
        ] {
            ensure!(
                dim.is_multiple_of(FP8_BLOCK),
                "{what} is {dim}, not a multiple of the fp8 tile {FP8_BLOCK} — every routed \
                 expert's `weight_scale_inv` grid is [out/{FP8_BLOCK}, in/{FP8_BLOCK}], and a \
                 ragged edge makes the grid's last row or column cover fewer weights than the \
                 dequant assumes"
            );
        }
        Ok(())
    }

    /// The n-gram table's geometry and — the trap — its ONE-INDEXED host layer.
    fn validate_ple(&self) -> Result<()> {
        // Exactly one PLE layer. More than one is not a bigger version of this model: the hash
        // reseeds per PLE layer AND the head vocabularies continue past the first layer's 16
        // primes (`qwen-architecture.md` §5), and a port that carried only one of those two
        // would collide both layers' heads onto the same table rows — plausible output, no
        // crash. Refused until there is a checkpoint to measure it against.
        ensure!(
            self.ple_layer_ids.len() == 1,
            "ple_layer_ids is {:?} — this port implements ONE PLE layer. Two would reseed the \
             multipliers (`base_seed = seed + 10007 * ple_layer_index`) AND continue the head \
             vocabularies past the first layer's primes, and carrying only one of those two \
             collides both layers onto the same rows",
            self.ple_layer_ids
        );
        let host = self.ple_host_layer()?;
        // The host must be a GDN layer: CFG:248-254 requires a `linear_attention` host, and the
        // PLE output is added at the TOP of that layer before the attention branch reads the
        // stream.
        ensure!(
            self.layer_types.get(host).map(String::as_str) == Some(LINEAR_ATTENTION),
            "ple_layer_ids {:?} is ONE-INDEXED, so the host is layer_idx {host} — which \
             layer_types calls {:?} rather than {LINEAR_ATTENTION:?}. The reference requires a \
             linear-attention host (CFG:248-254), and the checkpoint's only PLE tensors sit \
             under `layers.{host}.`",
            self.ple_layer_ids,
            self.layer_types.get(host)
        );
        ensure!(
            self.ngram_size > 1,
            "ngram_size {} gives (ngram_size - 1) x heads_per_ngram = 0 n-gram heads",
            self.ngram_size
        );
        let heads = self.ngram_heads();
        ensure!(
            self.ple_embed_dim.is_multiple_of(heads),
            "ple_embed_dim {} does not divide into {heads} n-gram heads — the gather is \
             `heads x (ple_embed_dim / heads)` and a ragged split reads each head at the wrong \
             stride",
            self.ple_embed_dim
        );
        Ok(())
    }

    /// Five strings, each of which selects an ARITHMETIC while every shape stays right — the
    /// class of defect nothing downstream refuses.
    ///
    /// Spelled with `as_str()` triples rather than `&self.field` ones, which is `k3_config.rs`'s
    /// width-gate deviation applied here for its mechanical co-benefit: the two files' settings
    /// loops were a 40-token clone the moment this one existed, and the response this repo
    /// prescribes is to change the code, never to exempt it.
    fn validate_named_settings(&self) -> Result<()> {
        for (key, got, want) in [
            ("dtype", self.dtype.as_str(), "bfloat16"),
            ("hidden_act", self.hidden_act.as_str(), "silu"),
            (
                "output_gate_type",
                self.output_gate_type.as_str(),
                "sigmoid",
            ),
            ("mamba_ssm_dtype", self.mamba_ssm_dtype.as_str(), "float32"),
            // `default`, never `yarn`: YaRN changes `inv_freq` AND puts a non-unit
            // `attention_factor` on cos/sin, and it is absent from every released config.
            (
                "rope_parameters.rope_type",
                self.rope_parameters.rope_type.as_str(),
                "default",
            ),
        ] {
            ensure!(
                got == want,
                "{key} is {got:?} where this port implements {want:?} only — every one of these \
                 five selects an arithmetic and leaves the shapes alone, so no length, count or \
                 dtype check downstream would refuse it"
            );
        }
        ensure!(
            self.rope_parameters.mrope_interleaved,
            "rope_parameters.mrope_interleaved is false — it is inert for text-only v1 (2-D \
             position_ids expand to three identical grids) and a different rotary layout the \
             day vision lands, so the flag is asserted rather than ignored"
        );
        ensure!(
            !self.attention_bias,
            "attention_bias is true, but no `*_proj.bias` family exists on the text side of \
             this checkpoint's 109-family census — the weights and the flag would disagree"
        );
        ensure!(
            !self.tie_word_embeddings,
            "tie_word_embeddings is true — this checkpoint ships a separate lm_head, and \
             reading the output projection out of the embedding table is a different set of \
             weights at the same shape"
        );
        ensure!(
            self.eos_token_id < self.vocab && self.bos_token_id < self.vocab,
            "eos_token_id {} / bos_token_id {} must index the {}-row vocabulary: the n-gram \
             history is PADDED with the eos scalar (MOD:1076), so an out-of-range id hashes \
             rows that do not exist",
            self.eos_token_id,
            self.bos_token_id,
            self.vocab
        );
        ensure!(
            self.mtp_num_hidden_layers == 1 && !self.mtp_use_dedicated_embeddings,
            "mtp_num_hidden_layers {} / mtp_use_dedicated_embeddings {} — the census's 35 \
             `mtp.*` families are ONE layer's worth sharing the trunk embeddings, and any other \
             shape is a different excluded set than the one the converter names",
            self.mtp_num_hidden_layers,
            self.mtp_use_dedicated_embeddings
        );
        Ok(())
    }
}
