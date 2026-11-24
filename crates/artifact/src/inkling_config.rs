//! Inkling 2's config schema — ONE port, TWO members, per the owner scope of 2026-11-23
//! (Mini first on the shared text stack, Small-MXFP8 the bigger-than-memory streaming case).
//! The math is `docs/reference/inkling2-architecture.md`; the plan and the stage legend are
//! `docs/investigations/inkling2-port.md`.
//!
//! **Two wrapper types over ONE shared [`InklingTextConfig`], and the sharing is the
//! reference's own shape, not a convenience.** The crate header warns against a shared
//! struct because that hazard caught two fields living in the wrong struct in the old tree —
//! but that hazard is about DIFFERENT models whose dimensions drift apart. Here the reference
//! declares `Inkling2LoopTextConfig` as a subclass of `Inkling2TextConfig` adding `loop_config`
//! (CFG loop:60-95 @ d8f10a82); the shared fields are one declaration in the reference and one
//! struct here. Two separate full-field structs would be a jscpd clone at the gate's 5-line floor
//! AND a second authority for the same numbers. The member identity lives in the WRAPPER
//! ([`Inkling2LoopConfig`] / [`Inkling2Config`]), and [`InklingTextConfig::validate`] takes the
//! resolved [`Arch`] so the member-specific fields are asserted present-or-absent per member
//! rather than defaulted.
//!
//! **Member-specific fields are `Option<T>` whose `None` means ABSENT FROM THIS MEMBER'S
//! SHIPPED FILE**, the qwen precedent's rule (`qwen_config.rs` header): never a Rust default
//! that happens to agree today. Mini's `text_config` carries `model_type` (`inkling2_loop_text`)
//! and `loop_config`; Small's carries NEITHER, and carries `pre_norm_pre_affine` /
//! `final_norm_pre_affine` (both `false`) which Mini lacks. `validate` asserts each side.
//!
//! **Field names are the JSON spellings verbatim, no serde renames.** The reference's
//! `attribute_map` already targets these exact keys (CONV @ d8f10a82), so a rename layer would
//! be a second authority between the file and the struct. The one exception is `text_config` →
//! `text`, the wrapper descent every sibling config spells the same way.
//!
//! **What is deliberately NOT bound.** `vision_config` (text-only v1, owner scope 2026-11-23 —
//! its 422 tensors are EXCLUDED-VISION in `docs/measurement/inkling2-reference/mini/tensor-families.tsv`,
//! excluded BY NAME at S2, a stronger statement than a schema that cannot see them);
//! `mtp_config` (the DSpark draft stack is deferred behind `Flag::Mtp`'s refusal, and
//! `mtp.safetensors` is excluded by name the same way); `eos_token_id` (the tokenizer pins the
//! specials and `generation_config.json` carries the stop id — binding it here would be a third
//! statement of one number); `scoring_func` (ABSENT from both shipped files — sigmoid is the
//! reference class default, LOOP:377 / MOD router, pinned as a comment at [`InklingTextConfig::route_scale`]
//! rather than bound to a key that does not exist); training-side fields. Each absence is a
//! decision.
//!
//! **Bare `usize` for dimensions, not newtypes**, for `qwen_config.rs:29-35`'s consistency
//! argument: all five sibling configs declare bare `usize` and the shared validators take
//! `usize`; a sixth that newtyped only its own fields would fork those helpers.

use crate::arch::Arch;
use crate::schema::{ArchConfig, ensure_f32_positive};
use anyhow::{Result, bail, ensure};
use serde::Deserialize;

/// The MXFP8 scale block along the input dim — 32, ROW-WISE `[1, 32]`.
///
/// **The config's own `hf_quant_config.json` declares `[32, 32]` and that is a lie the scale
/// shapes refute** (trap T1, `inkling2-architecture.md` §5): Mini's MXFP8 `w13_weight_scale` is
/// `[128, 2304, 72]` against a `[128, 2304, 2304]` weight — 2304/72 = 32 along the LAST dim
/// only, one scale row per weight row. Small's root `quantization_config` tells the truth
/// (`weight_block_size: [1, 32]`, `scale_fmt: "ue8m0"`). NVFP4's block is 16, finer, so a width
/// aligned to 32 is aligned to both schemes Mini uses.
const MXFP8_BLOCK: usize = 32;

/// `text_config.loop_config` — Mini ONLY. Small's shipped file has no such key and
/// [`InklingTextConfig::validate`] refuses one on that member.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingLoopConfig {
    /// 6 — the first looped layer, a block boundary (`core_start % n_layers_per_attn_res_block
    /// == 0`). The loop runs layers `[core_start, core_end)`.
    pub core_start: usize,
    /// 18 — exclusive end, also a block boundary. `core_end - core_start` = 12 looped layers.
    pub core_end: usize,
    /// 3 — passes over the looped block. Pass 1 carries no messages; passes 2–3 read the
    /// previous pass's block states (LOOP:537).
    pub num_passes: usize,
    /// 64 — the MessageGate down-projection rank (2H → 64, bf16).
    pub gate_rank: usize,
    /// `false`. A variable-pass loop is a different recurrence; refused, not defaulted.
    pub is_varloop: bool,
}

/// The root `quantization_config`, per member. Mini ships `{quant_method: "modelopt",
/// quant_algo: "NVFP4"}`; Small ships `{quant_method: "fp8", activation_scheme: "dynamic",
/// weight_block_size: [1, 32], scale_fmt: "ue8m0", modules_to_not_convert: [...]}`.
///
/// The per-LAYER split (Mini layers 0–19 NVFP4, 20–23 MXFP8) lives in `hf_quant_config.json`,
/// hash-pinned beside this file but parsed by the S2 converter, not here — the root header is
/// what a config gate can assert without a 67 KB census.
#[derive(Debug, Clone, Deserialize)]
pub struct InklingQuantConfig {
    /// `"modelopt"` (Mini) or `"fp8"` (Small).
    pub quant_method: String,
    /// `"NVFP4"` — Mini only; absent on Small.
    pub quant_algo: Option<String>,
    /// `"dynamic"` — Small only.
    pub activation_scheme: Option<String>,
    /// `[1, 32]` — Small only, and the truthful row-wise block (trap T1).
    pub weight_block_size: Option<Vec<usize>>,
    /// `"ue8m0"` — Small only: the MXFP8 scale is a biased exponent, `2^(u8 - 127)`.
    pub scale_fmt: Option<String>,
    /// Small's exclusion globs: `["model.llm.unembed", "model.visual",
    /// "model.llm.layers.*.attn.*"]`. The converter asserts its own exclusion list against
    /// these; binding them here makes a hand-edited config a refusal rather than a silent
    /// re-quantization of the attention weights.
    pub modules_to_not_convert: Option<Vec<String>>,
}

/// Inkling 2 Mini, as its `config.json` ships it: an `Inkling2LoopForConditionalGeneration`
/// wrapper around the looped text model.
///
/// **Hold a [`Inkling2LoopConfig`], not a bare [`InklingTextConfig`]** — only the outer type is
/// evidence that [`crate::schema::parse_config`] ran both the architecture check and `validate`,
/// and a bare text struct could be deserialized out of a promoted `text_config` with no member
/// identity at all. `QwenConfig`'s doc carries the same warning.
#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2LoopConfig {
    /// The ROOT `model_type`, `inkling2_loop`.
    pub model_type: String,
    /// `["Inkling2LoopForConditionalGeneration"]` — the other half of the root pair.
    pub architectures: Vec<String>,
    /// `"bfloat16"` — Mini's root carries `dtype`; Small's does NOT, which is why this field
    /// lives on the Mini wrapper and not on a shared root struct.
    pub dtype: String,
    #[serde(rename = "text_config")]
    pub text: InklingTextConfig,
    pub quantization_config: InklingQuantConfig,
}

/// Inkling 2 Small (the MXFP8 checkpoint), as its `config.json` ships it: an
/// `Inkling2ForConditionalGeneration` wrapper around the plain text model. No root `dtype` key,
/// no `loop_config`, no nested `text_config.model_type` — the absences are asserted, not skipped.
#[derive(Debug, Clone, Deserialize)]
pub struct Inkling2Config {
    /// The ROOT `model_type`, `inkling2`.
    pub model_type: String,
    /// `["Inkling2ForConditionalGeneration"]`.
    pub architectures: Vec<String>,
    #[serde(rename = "text_config")]
    pub text: InklingTextConfig,
    pub quantization_config: InklingQuantConfig,
}

/// The `text_config` dict — every architecture dimension, shared by both members. The 36
/// required fields are the reference's `Inkling2TextConfig` declaration; the four `Option`s are
/// the member differences (see the module header).
#[derive(Debug, Clone, Deserialize)]
pub struct InklingTextConfig {
    /// Mini: `Some("inkling2_loop_text")`. Small: `None` — the shipped file omits the key.
    /// Asserted per member; the root-only sniff refuses this spelling at the document root.
    pub model_type: Option<String>,
    /// 1048576.
    pub model_max_length: usize,
    /// `"bfloat16"` — the text stack computes bf16 with fp32 islands
    /// (`_keep_in_fp32_modules_strict`, LOOP:389-401).
    pub torch_dtype: String,
    /// 2304 (Mini) / 4608 (Small).
    pub hidden_size: usize,
    /// 24 / 48.
    pub num_hidden_layers: usize,
    /// 201024, PADDED. `unpadded_vocab_size` (200058) is the live row count; the lm_head keeps
    /// the first 200058 rows of the 201024-row `unembed` (LOAD:856), embed keeps all.
    pub vocab_size: usize,
    /// 24 / 48.
    pub num_attention_heads: usize,
    /// 8, both members.
    pub num_key_value_heads: usize,
    /// 128.
    pub head_dim: usize,
    /// 16 — the relative-position bias width (`d_rel`).
    pub d_rel: usize,
    /// 1024 — the relative bias extent.
    pub rel_extent: usize,
    /// `false`, both. Pinned: a bias on q/o is a different projection.
    pub q_bias: bool,
    /// `false`, both.
    pub o_bias: bool,
    /// 8192, BOTH members — the LogScale floor `n_floor` in
    /// `τ = 1 + α·log(max((pos+1)/n_floor, 1))`.
    pub log_scaling_n_floor: usize,
    /// 0.1 — the LogScale α.
    pub log_scaling_alpha: f64,
    /// 1e-6.
    pub rms_norm_eps: f64,
    /// Small: `Some(false)`. Mini: `None`. The pre-norm affine is disabled on Small; asserted.
    pub pre_norm_pre_affine: Option<bool>,
    /// Small: `Some(false)`. Mini: `None`.
    pub final_norm_pre_affine: Option<bool>,
    /// `true`, both — the value norm carries no gamma.
    pub use_value_norm_no_gamma: bool,
    /// 22.627416997969522 (= 16√2) — the QK L2Clip threshold.
    pub qk_l2_clip_threshold: f64,
    /// 8.0 — the relative-bias L2Clip threshold.
    pub rel_l2_clip_threshold: f64,
    /// The SLIDING-window layers. `validate` DERIVES the expected complement of the block-end
    /// (full-attention) layers and compares — a strong cross-check, not a trusted list.
    pub local_layer_ids: Vec<usize>,
    /// 4 — the short-convolution kernel. Pinned: the static sconv's `(conv+residual)/√5` and the
    /// dynamic sconv's `(sum+residual)/3` divisors are functions of k=4 (MOD sconv), so a
    /// different k is different math, not a wider kernel.
    pub sconv_kernel_size: usize,
    /// 200058 — the live vocabulary rows (see `vocab_size`).
    pub unpadded_vocab_size: usize,
    /// 30.0 — the final-logit softcap.
    pub final_logit_softcapping: f64,
    /// 128.
    pub swa_head_dim: usize,
    /// 24 / 48 — equals `num_attention_heads` on both members; compared, not trusted.
    pub swa_num_attention_heads: usize,
    /// 8 — equals `num_key_value_heads`; compared.
    pub swa_num_key_value_heads: usize,
    /// 1024.
    pub sliding_window_size: usize,
    /// 6 — the block-residual block size. Layer `i` is a block END (full attention) iff
    /// `(i+1) % 6 == 0`; the boundary residual reset fires at `i % 6 == 0` INCLUDING layer 0
    /// (MOD:1433-1434, trap T-table).
    pub n_layers_per_attn_res_block: usize,
    /// `true`, both.
    pub attn_res_normalize_values: bool,
    /// 128 (Mini) / 384 (Small).
    pub n_routed_experts: usize,
    /// 8, both.
    pub num_experts_per_tok: usize,
    /// 0, both — no shared expert. Pinned: a nonzero count is a different MoE.
    pub n_shared_experts: usize,
    /// 1152 / 2304 — the routed expert intermediate width.
    pub intermediate_size: usize,
    /// 4.0 — the SwiGLU gate softcap (fp32).
    pub mlp_gate_softcap: f64,
    /// 25.0 — the SwiGLU up softcap (fp32).
    pub mlp_up_softcap: f64,
    /// 8.0 — the router weight scale: `gather/(sum+1e-20) · route_scale`.
    ///
    /// `scoring_func` is NOT a shipped key on either member; the reference router is a fp32
    /// SIGMOID with `topk(scores+bias, k, sorted=False)` and a selection-only bias
    /// (`Inkling2LoopTopKRouter`, LOOP:377). Pinned here as a comment because binding a key the
    /// file does not carry would invent a default.
    pub route_scale: f64,
    /// `true`, both — the router carries an `e_score_correction_bias`.
    pub use_gate_bias: bool,
    /// Mini: `Some(...)`. Small: `None`, asserted.
    pub loop_config: Option<InklingLoopConfig>,
}

impl ArchConfig for Inkling2LoopConfig {
    const ARCH: Arch = Arch::Inkling2Loop;

    fn validate(&self) -> Result<()> {
        // One line, not the five-line `ensure!` block glimmer_config.rs:227 writes for the same
        // bfloat16-root rule: jscpd normalizes the message literal, so the block form clones it.
        ensure!(
            self.dtype == "bfloat16",
            "Mini root dtype {:?} is not bfloat16",
            self.dtype
        );
        validate_member(
            (&self.model_type, &self.architectures),
            &self.quantization_config,
            &self.text,
            Arch::Inkling2Loop,
        )
    }
}

impl ArchConfig for Inkling2Config {
    const ARCH: Arch = Arch::Inkling2;

    fn validate(&self) -> Result<()> {
        validate_member(
            (&self.model_type, &self.architectures),
            &self.quantization_config,
            &self.text,
            Arch::Inkling2,
        )
    }
}

/// Both wrappers validate the same way — root pair, quant header, then the shared text stack —
/// so the rule lives here once rather than as a cloned `validate_root`-call pair in two `impl`s
/// (jscpd's 5-line floor). Mini's root-dtype check is the one member-specific step and stays at
/// its call site; Small has no root `dtype` key to check.
fn validate_member(
    root: (&str, &[String]),
    quant: &InklingQuantConfig,
    text: &InklingTextConfig,
    arch: Arch,
) -> Result<()> {
    // The expected root spelling is a function of the member, so it is derived here rather
    // than passed: four arguments, under CodeScene's Rust threshold of five (the six-argument
    // form scored 9.68). The two call sites differ only in the `Arch` they name.
    let spelling = match arch {
        Arch::Inkling2Loop => ("inkling2_loop", "Inkling2LoopForConditionalGeneration"),
        Arch::Inkling2 => ("inkling2", "Inkling2ForConditionalGeneration"),
        other => bail!("validate_member called for {other:?}, which has no inkling wrapper"),
    };
    validate_root(root.0, root.1, spelling)?;
    validate_quant(quant, arch)?;
    text.validate(arch)
}

impl Inkling2LoopConfig {
    /// Load from the artifact's `manifest.json`, falling back to a bare `config.json`. The
    /// turbofish spells the concrete type for `QwenConfig::load`'s jscpd reason.
    pub fn load(dir: &str) -> Result<Self> {
        crate::schema::load_config::<Inkling2LoopConfig>(dir)
    }
}

impl Inkling2Config {
    /// Load from the artifact's `manifest.json`, falling back to a bare `config.json`.
    pub fn load(dir: &str) -> Result<Self> {
        crate::schema::load_config::<Inkling2Config>(dir)
    }
}

/// The root pair, asserted as one `ensure!` for `QwenConfig::validate`'s reason: which half
/// disagrees is not the interesting fact, "this is not the document we think it is" is. Shared
/// by both members because the check is one rule with two expected pairs, not two rules.
fn validate_root(model_type: &str, architectures: &[String], want: (&str, &str)) -> Result<()> {
    ensure!(
        model_type == want.0 && architectures == [want.1.to_string()],
        "the document ROOT declares {model_type:?} / {architectures:?} — this member's wrapper \
         is {:?} / [{:?}]. A promoted `text_config` or `vision_config` block carries a nested \
         spelling (or none) and is the shape this refusal catches",
        want.0,
        want.1
    );
    Ok(())
}

/// The root quant header, per member. The per-layer NVFP4/MXFP8 census is the S2 converter's;
/// this asserts the header the shipped file actually carries.
fn validate_quant(q: &InklingQuantConfig, arch: Arch) -> Result<()> {
    match arch {
        Arch::Inkling2Loop => {
            ensure!(
                q.quant_method == "modelopt" && q.quant_algo.as_deref() == Some("NVFP4"),
                "Mini's quantization_config is {:?}/{:?} — the shipped header is \
                 modelopt/NVFP4 (layers 0–19 NVFP4, 20–23 MXFP8 per hf_quant_config.json)",
                q.quant_method,
                q.quant_algo
            );
        }
        Arch::Inkling2 => {
            ensure!(
                q.quant_method == "fp8"
                    && q.activation_scheme.as_deref() == Some("dynamic")
                    && q.weight_block_size.as_deref() == Some([1, MXFP8_BLOCK].as_slice())
                    && q.scale_fmt.as_deref() == Some("ue8m0"),
                "Small's quantization_config is {:?} — the shipped header is fp8/dynamic with \
                 weight_block_size [1,32] and scale_fmt ue8m0. The [32,32] in hf_quant_config.json \
                 is refuted by the scale shapes (trap T1); [1,32] row-wise is the truth",
                q.quant_method
            );
            ensure!(
                q.modules_to_not_convert.is_some(),
                "Small's quantization_config omits modules_to_not_convert — the exclusion globs \
                 (unembed, visual, attention) are what keep those weights out of the fp8 repack"
            );
        }
        // The two wrappers above are the only callers; a third member is a new port, not a
        // silently-accepted quant header.
        other => bail!("validate_quant called for {other:?}, which has no inkling wrapper"),
    }
    Ok(())
}

impl InklingTextConfig {
    /// Cross-field checks, split into named groups for `QwenTextConfig::validate`'s CodeScene
    /// reason: one body whose only structure is comment headings is the shape the gate refuses.
    pub fn validate(&self, arch: Arch) -> Result<()> {
        self.validate_member(arch)?;
        self.validate_widths()?;
        self.validate_attention_geometry()?;
        self.validate_moe()?;
        self.validate_named_settings()?;
        // Every f64 the kernels narrow to f32, checked in the f32 domain (the shared helper's
        // argument): the clip thresholds, softcaps, α and eps all reach fp32 islands.
        ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            ("log_scaling_alpha", self.log_scaling_alpha),
            ("qk_l2_clip_threshold", self.qk_l2_clip_threshold),
            ("rel_l2_clip_threshold", self.rel_l2_clip_threshold),
            ("final_logit_softcapping", self.final_logit_softcapping),
            ("mlp_gate_softcap", self.mlp_gate_softcap),
            ("mlp_up_softcap", self.mlp_up_softcap),
            ("route_scale", self.route_scale),
        ])
    }

    /// The member-specific presence/absence claims — the four `Option`s and the nested
    /// `model_type`. This is what makes one shared struct two members rather than one blurred
    /// average: Mini MUST carry the loop and the text spelling, Small MUST carry neither and
    /// MUST carry the two pre-affine flags.
    fn validate_member(&self, arch: Arch) -> Result<()> {
        match arch {
            Arch::Inkling2Loop => {
                ensure!(
                    self.model_type.as_deref() == Some("inkling2_loop_text"),
                    "Mini's text_config declares model_type {:?} — the text half's spelling is \
                     \"inkling2_loop_text\"; a descent into vision_config (inkling2_vision) is \
                     what this refuses",
                    self.model_type
                );
                let lc = self.loop_config.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("Mini's text_config omits loop_config — the looped member")
                })?;
                self.validate_loop(lc)?;
                ensure!(
                    self.pre_norm_pre_affine.is_none() && self.final_norm_pre_affine.is_none(),
                    "Mini's text_config carries pre_norm_pre_affine/final_norm_pre_affine — \
                     those keys are Small's; the shipped Mini file has neither"
                );
            }
            Arch::Inkling2 => {
                ensure!(
                    self.model_type.is_none(),
                    "Small's text_config declares model_type {:?} — the shipped Small file \
                     omits the key entirely; a declared spelling is a promoted sub-config",
                    self.model_type
                );
                ensure!(
                    self.loop_config.is_none(),
                    "Small's text_config carries loop_config — Small is the NON-looped member"
                );
                ensure!(
                    self.pre_norm_pre_affine == Some(false)
                        && self.final_norm_pre_affine == Some(false),
                    "Small's pre_norm_pre_affine/final_norm_pre_affine are {:?}/{:?} — the \
                     shipped file disables both affines (false/false)",
                    self.pre_norm_pre_affine,
                    self.final_norm_pre_affine
                );
            }
            other => {
                bail!("inkling text validate called for {other:?}, no inkling wrapper")
            }
        }
        Ok(())
    }

    /// The loop geometry: a whole number of blocks, at least one pass, a positive gate rank,
    /// and no variable-pass recurrence. `core_end <= num_hidden_layers` keeps the loop inside
    /// the model; the block-boundary alignment keeps the residual reset and the loop edges on
    /// the same grid.
    fn validate_loop(&self, lc: &InklingLoopConfig) -> Result<()> {
        let block = self.n_layers_per_attn_res_block;
        ensure!(
            block > 0,
            "n_layers_per_attn_res_block is 0 — no block grid"
        );
        ensure!(
            lc.core_start > 0
                && lc.core_start.is_multiple_of(block)
                && lc.core_end.is_multiple_of(block)
                && lc.core_end > lc.core_start
                && lc.core_end <= self.num_hidden_layers,
            "loop_config [{}, {}) is not a whole number of {block}-layer blocks inside \
             {} layers — the loop edges must sit on block boundaries",
            lc.core_start,
            lc.core_end,
            self.num_hidden_layers
        );
        ensure!(lc.num_passes >= 1, "loop_config.num_passes is 0 — no loop");
        ensure!(
            lc.gate_rank > 0,
            "loop_config.gate_rank is 0 — no message gate"
        );
        ensure!(
            !lc.is_varloop,
            "loop_config.is_varloop is true — a variable-pass loop is a different recurrence, \
             refused rather than stepped"
        );
        Ok(())
    }

    /// Every width is positive — a zero passes every divisibility check below and then sizes a
    /// tensor to nothing (`QwenTextConfig::validate_widths`'s consequence argument).
    fn validate_widths(&self) -> Result<()> {
        for (what, dim) in [
            ("model_max_length", self.model_max_length),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("vocab_size", self.vocab_size),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("d_rel", self.d_rel),
            ("rel_extent", self.rel_extent),
            ("log_scaling_n_floor", self.log_scaling_n_floor),
            ("sconv_kernel_size", self.sconv_kernel_size),
            ("unpadded_vocab_size", self.unpadded_vocab_size),
            ("swa_head_dim", self.swa_head_dim),
            ("swa_num_attention_heads", self.swa_num_attention_heads),
            ("swa_num_key_value_heads", self.swa_num_key_value_heads),
            ("sliding_window_size", self.sliding_window_size),
            (
                "n_layers_per_attn_res_block",
                self.n_layers_per_attn_res_block,
            ),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("intermediate_size", self.intermediate_size),
        ] {
            ensure!(
                dim > 0,
                "{what} is 0 — it sizes a tensor, a stride or a GEMV dim to nothing"
            );
        }
        Ok(())
    }

    /// The attention geometry: GQA divisibility, the sliding/full head-count cross-checks, and
    /// the DERIVED `local_layer_ids` complement. The full-attention layers are the block ENDS
    /// (`(i+1) % block == 0`); every other layer is sliding. Comparing the shipped list against
    /// the derived one catches a hand-edited list that a positivity check would pass.
    fn validate_attention_geometry(&self) -> Result<()> {
        ensure!(
            self.num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
                && self
                    .swa_num_attention_heads
                    .is_multiple_of(self.swa_num_key_value_heads),
            "head counts are not GQA-divisible: {}/{} and {}/{}",
            self.num_attention_heads,
            self.num_key_value_heads,
            self.swa_num_attention_heads,
            self.swa_num_key_value_heads
        );
        ensure!(
            self.num_attention_heads == self.swa_num_attention_heads
                && self.num_key_value_heads == self.swa_num_key_value_heads
                && self.head_dim == self.swa_head_dim,
            "the sliding and full attention geometries disagree — both members ship them equal \
             (heads {}, kv {}, head_dim {})",
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim
        );
        let block = self.n_layers_per_attn_res_block;
        let expected: Vec<usize> = (0..self.num_hidden_layers)
            .filter(|i| (i + 1) % block != 0)
            .collect();
        ensure!(
            self.local_layer_ids == expected,
            "local_layer_ids has {} entries, but the block-end complement of {} {block}-layer \
             blocks is {} — the sliding layers are every layer that is NOT a block end",
            self.local_layer_ids.len(),
            self.num_hidden_layers,
            expected.len()
        );
        Ok(())
    }

    /// The MoE: top-k within the expert count, no shared expert, and both expert widths aligned
    /// to the MXFP8 block (which also covers Mini's finer NVFP4 block-16).
    fn validate_moe(&self) -> Result<()> {
        ensure!(
            self.num_experts_per_tok <= self.n_routed_experts,
            "num_experts_per_tok {} exceeds n_routed_experts {}",
            self.num_experts_per_tok,
            self.n_routed_experts
        );
        ensure!(
            self.n_shared_experts == 0,
            "n_shared_experts is {} — both members ship 0; a shared expert is a different MoE",
            self.n_shared_experts
        );
        for (key, dim) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
        ] {
            ensure!(
                dim.is_multiple_of(MXFP8_BLOCK),
                "{key} is {dim}, not a multiple of the MXFP8 block {MXFP8_BLOCK} — expert rows \
                 would carry a ragged scale block (NVFP4's block-16 is finer and also covered)"
            );
        }
        Ok(())
    }

    /// The pinned settings: the booleans that select a math path, the sconv kernel the divisors
    /// depend on, the vocabulary padding relation, and the bf16 compute dtype.
    fn validate_named_settings(&self) -> Result<()> {
        ensure!(
            !self.q_bias && !self.o_bias,
            "q_bias/o_bias are {}/{} — both members ship false; a bias is a different projection",
            self.q_bias,
            self.o_bias
        );
        ensure!(
            self.use_value_norm_no_gamma && self.attn_res_normalize_values && self.use_gate_bias,
            "use_value_norm_no_gamma/attn_res_normalize_values/use_gate_bias are {}/{}/{} — \
             both members ship true/true/true",
            self.use_value_norm_no_gamma,
            self.attn_res_normalize_values,
            self.use_gate_bias
        );
        ensure!(
            self.sconv_kernel_size == 4,
            "sconv_kernel_size is {} — the static sconv's /√5 and the dynamic sconv's /3 are \
             functions of k=4; a different kernel is different math",
            self.sconv_kernel_size
        );
        ensure!(
            self.unpadded_vocab_size <= self.vocab_size,
            "unpadded_vocab_size {} exceeds the padded vocab_size {}",
            self.unpadded_vocab_size,
            self.vocab_size
        );
        ensure!(
            self.torch_dtype == "bfloat16",
            "text_config.torch_dtype is {:?} — the text stack computes bfloat16",
            self.torch_dtype
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

    use crate::inkling_config::{Inkling2Config, Inkling2LoopConfig};
    use crate::schema::{ArchConfig, parse_config};
    use serde_json::{Value, json};
    const MINI: &str =
        include_str!("../../../docs/measurement/inkling2-reference/mini/config.json");
    const SMALL: &str =
        include_str!("../../../docs/measurement/inkling2-reference/small-mxfp8/config.json");

    /// One negative row: the pointer to the OBJECT the key lives in, the key, the value written
    /// there, and the fragment the refusal must name.
    ///
    /// A TABLE rather than one call per row: `refuses::<T>(BASE, |d| …, "…")` is a five-line call
    /// whose normalized tokens are the same for every row, which is a jscpd clone the moment
    /// there are two of them. The pointer addresses the PARENT because two rows write a key the
    /// shipped file does not carry, and `Value::pointer_mut` cannot create a missing key.
    type Row<'a> = (&'a str, &'a str, Value, &'a str);

    /// Apply every row to a fresh copy of the shipped `base` document, parse it as `T`, and
    /// require the refusal to name that row's fragment.
    ///
    /// Each row starts from the vendored FILE, so there is no transcribed base for the
    /// checkpoint's values to drift away from; a row that stopped refusing would score the
    /// unmutated document, which the positive cell below proves parses.
    fn refuses<T: ArchConfig>(base: &str, rows: &[Row<'_>]) {
        for (parent, key, value, want) in rows {
            let mut document: Value =
                serde_json::from_str(base).expect("the vendored config is JSON");
            let object = document
                .pointer_mut(parent)
                .expect("the row's parent pointer must name an object");
            object[*key] = value.clone();
            let err = parse_config::<T>(&document.to_string())
                .err()
                .unwrap_or_else(|| panic!("{parent}/{key} was mutated and still parsed"));
            let text = format!("{err:#}");
            assert!(
                text.contains(*want),
                "{parent}/{key} refused, but not for the reason under test: {text}"
            );
        }
    }

    /// The base every mutation starts from: both shipped members parse.
    #[test]
    fn both_shipped_members_parse() {
        parse_config::<Inkling2LoopConfig>(MINI).expect("Mini's own config must load");
        parse_config::<Inkling2Config>(SMALL).expect("Small's own config must load");
    }

    /// Every setting Mini's file pins: the loop recurrence, the member split, the arithmetic
    /// selectors, the widths, and the DERIVED sliding-layer list. Each row names the branch it
    /// holds, so a guard that stopped refusing reddens on its own row rather than on a bare
    /// `is_err()`.
    #[test]
    fn mini_refuses_every_setting_the_shipped_file_pins() {
        refuses::<Inkling2LoopConfig>(
            MINI,
            &[
                (
                    "/text_config/loop_config",
                    "is_varloop",
                    json!(true),
                    "is_varloop",
                ),
                // 6 is the block size; 7 sits mid-block, off the residual grid.
                (
                    "/text_config/loop_config",
                    "core_start",
                    json!(7),
                    "loop_config",
                ),
                (
                    "/quantization_config",
                    "quant_method",
                    json!("fp8"),
                    "quantization_config",
                ),
                (
                    "/text_config",
                    "pre_norm_pre_affine",
                    json!(false),
                    "pre_norm_pre_affine",
                ),
                (
                    "/text_config",
                    "sconv_kernel_size",
                    json!(3),
                    "sconv_kernel_size",
                ),
                (
                    "/text_config",
                    "n_shared_experts",
                    json!(1),
                    "n_shared_experts",
                ),
                ("/text_config", "hidden_size", json!(0), "hidden_size is 0"),
                (
                    "/text_config",
                    "local_layer_ids",
                    json!([0, 1]),
                    "local_layer_ids",
                ),
            ],
        );
    }

    /// Small is the NON-looped member with no nested text spelling, and its `[1, 32]` scale block
    /// is the truth `hf_quant_config.json`'s `[32, 32]` refutes (trap T1) — a document claiming
    /// the square block is refused rather than admitted.
    #[test]
    fn small_refuses_mini_only_keys_and_the_square_scale_block() {
        refuses::<Inkling2Config>(
            SMALL,
            &[
                (
                    "/text_config",
                    "loop_config",
                    json!({
                        "core_start": 6,
                        "core_end": 18,
                        "num_passes": 3,
                        "gate_rank": 64,
                        "is_varloop": false,
                    }),
                    "carries loop_config",
                ),
                (
                    "/text_config",
                    "model_type",
                    json!("inkling2_text"),
                    "declares model_type",
                ),
                (
                    "/quantization_config",
                    "weight_block_size",
                    json!([32, 32]),
                    "weight_block_size",
                ),
            ],
        );
    }
}
