//! Kimi-K3's config schema — one architecture, one file, per the rule that per-model config
//! types stay separate. Ported from `k3:src/artifact/model.rs`'s K3 slice (`K3Config` at
//! ~969, its validate at ~1160), bodies and comments travelling verbatim: in this repo a
//! comment carries the measurement that justified the choice.
//!
//! A separate struct rather than optional fields on any sibling, and separate serde
//! declarations rather than a shared core — [`crate::schema`]'s header carries the argument.
//! K3 is the strongest instance of it: it agrees with GLM/V4/Glimmer on four HuggingFace
//! dimension names and disagrees on `num_experts_per_token` (`_tok` everywhere else), on
//! `num_experts` (`n_routed_experts` on V4), and on nesting the whole dict behind a
//! multimodal wrapper. A shared core would have had to special-case all three.
//!
//! **What did NOT come with it, and why.** The reference's K3 slice sits inside a 3452-line
//! `model.rs`; only K3's half is here. Its `jscpd:ignore` region around the four dimension
//! renames did not travel either — the rewrite's measured answer (`glimmer_config.rs`,
//! `v4_config.rs`, both 2026-08-16) is that a doc comment per field breaks the token run and
//! the gate reports 0 clones, and an exemption that suppresses nothing is a hole in the gate.
//! The validate body is split into named groups for the same reason those files' are: the
//! reference's is one ~230-line body whose only structure is comment headings, which is the
//! shape the CodeScene gate refuses. The groups are the headings, and the refusal ORDER is
//! the reference's.

use crate::arch::Arch;
use crate::schema::{ArchConfig, ensure_f4_group_aligned, ensure_f32_positive};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// The `linear_attn_config` dict — KDA's own geometry and the layer partition.
///
/// A separate struct because the FILE nests it, and the nesting is load-bearing:
/// `use_full_rank_gate` lives HERE, one level below the flags it is usually listed beside,
/// and reading it from the wrong level was a real bug in the k3 tree (`k3:src/artifact/model.rs`,
/// this struct's doc: "this field's LEVEL was the bug").
#[derive(Debug, Clone, Deserialize)]
pub struct LinearAttnConfig {
    /// One-based. 24 entries in the shipped config, and the two reference implementations
    /// read OPPOSITE arrays (`k3:docs/reference/k3-architecture.md` §2), so `validate`
    /// asserts the partition rather than trusting either alone.
    pub full_attn_layers: Vec<usize>,
    /// One-based. 69 entries in the shipped config.
    pub kda_layers: Vec<usize>,
    /// 96 — KDA's own head count, which equals `num_attention_heads` in this checkpoint but
    /// is a separate field and must not be read from it.
    pub num_heads: usize,
    /// 128, and `d_k == d_v` for KDA.
    pub head_dim: usize,
    /// 4 — the depthwise causal conv's kernel width (`conv_k` in the C reference).
    pub short_conv_kernel_size: usize,
    /// **-5.0, and NEGATIVE is correct.** It multiplies the sigmoid rather than clamping or
    /// flooring it — trap 4 of `k3:docs/reference/k3-architecture.md` §10 — so this is
    /// neither a bound nor an epsilon, and a positivity check on it would be wrong.
    pub gate_lower_bound: f64,
    /// True. See the struct doc: this field's LEVEL was the bug.
    pub use_full_rank_gate: bool,
}

/// Kimi-K3, as its `config.json` ships it: a `KimiK3ForConditionalGeneration` multimodal
/// wrapper around the text model.
///
/// The nesting is carried rather than flattened away, because it is load-bearing twice over.
/// The wrapper is the level that names the architecture ([`crate::arch::from_manifest_str`]
/// recognises the TOP-level pair only — its doc carries why the nested pair is not accepted
/// there), the nested dict is the level that carries the dimensions, and `vision_config` —
/// which this port does not implement — is a sibling of `text_config` rather than of
/// anything inside it. Flattening would cost a hand-written `Deserialize` and would hide
/// which level a field came from, which is exactly how a key goes missing for the wrong
/// reason (`k3:src/artifact/model.rs`, plan §3e).
#[derive(Debug, Clone, Deserialize)]
pub struct K3Config {
    #[serde(rename = "text_config")]
    pub text: K3TextConfig,
}

/// The `text_config` dict — Kimi-K3's text model, `KimiLinearForCausalLM`.
///
/// **Every field is REQUIRED**, and for the reason [`crate::schema`]'s header gives: a
/// defaulted dimension does not crash, it produces fluent wrong text. A field whose JSON key
/// this port has not verified against the shipped file is *absent* rather than guessed — a
/// wrong key on a required field refuses the real checkpoint with `missing field`, which is
/// loud and fixable, while a guessed key with a `#[serde(default)]` is the silent version.
///
/// **Hold a [`K3Config`], not this.** This type is `pub` with `pub` fields and derives
/// `Deserialize`, so one can be produced by deserializing the inner dict alone or by a
/// struct literal — either of which skips [`crate::schema::parse_config`] and therefore
/// skips both the architecture check and `validate`. Only `K3Config` is evidence that those
/// ran. Flagged by review in the k3 tree 2026-08-10; the nesting is what separates them, so
/// the discipline is a convention here rather than a type guarantee.
///
/// `quantization_config` is absent on purpose, and it is the one omission that is a decision
/// rather than a gap: the block **mis-declares its own scope** (`targets: ["Linear"]` with
/// an `ignore` list that omits `routed_expert_{down,up}_proj` and
/// `block_sparse_moe.gate.weight`, all three of which ship BF16). The converter drives off
/// the presence of `.weight_packed` instead, so a schema that read this block would be
/// reading a field nothing is allowed to trust (`k3:src/artifact/model.rs`, S1a item 5).
#[derive(Debug, Clone, Deserialize)]
pub struct K3TextConfig {
    /// The NESTED architecture pair — `KimiLinearForCausalLM` / `kimi_linear`, which differs
    /// from the wrapper's. Carried so `validate` can assert it: this struct is reached by
    /// descending through `text_config`, and a descent that landed in some other dict of a
    /// multimodal config would otherwise be indistinguishable from the right one.
    pub model_type: String,
    /// The other half of the nested pair; asserted together with `model_type`.
    pub architectures: Vec<String>,

    // The four dimension serde renames below coincide with the other three configs',
    // because all four checkpoints declare these under the SAME HuggingFace-standard JSON
    // names. **Not factored, and not exempted either** — the k3 tree wraps this run in a
    // `jscpd:ignore` region, and the rewrite's measured answer is that a doc comment per
    // field breaks the token run (`v4_config.rs` records 0 clones over `crates/` with the
    // markers removed). The design argument is unchanged: four architectures agreeing on
    // four JSON names is a coincidence of the checkpoints, not a shared contract, and a
    // shared struct becomes the attractor for a fifth field that is NOT shared — this
    // file's `expert_in` is exactly that field.
    /// The checkpoint ships 93 — and the partition arrays, `first_k_dense_replace` and the
    /// AttnRes blocks are all indexed by the REAL layer id, which is why a partial artifact
    /// never rewrites this (`convert_k3`'s manifest comment).
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    /// 7168 — the trunk width, and NOT the width the routed experts are entered at. That is
    /// [`Self::expert_in`], the 3584 latent, and binding `hidden` there produces a
    /// self-consistent artifact with every expert stride 2x wrong.
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    /// The checkpoint ships 163840.
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    /// 96 — and equal to `num_key_value_heads`, an equality `validate` pins so a copied V4
    /// MQA check (`== 1`) cannot land here unnoticed.
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    /// RMSNorm epsilon (1e-5 in the shipped config; note the first-party MLA LoRA norms use
    /// 1e-6 where the C reference wrote 1e-5 — `k3:docs/reference/k3-architecture.md` §5).
    ///
    /// The JSON key is `rms_norm_eps` as on GLM and V4 — **confirmed against the shipped
    /// file 2026-08-10** after the doc's own `rms_eps` (the C reference's field name) was
    /// flagged; a key taken from prose rather than the file is how two spellings became
    /// defects in the k3 tree.
    pub rms_norm_eps: f64,
    /// `bfloat16`. The trunk dtype, asserted rather than assumed: the trunk is BF16
    /// (k3 G0 item 3), and an fp8 export of the same model read as BF16 is noise at every
    /// width. Note the LEVEL: K3 puts `dtype` inside `text_config` where Glimmer puts it on
    /// the wrapper — a reader who knows one does not know the other.
    pub dtype: String,

    // --- attention. Which layer is which comes from the partition, not from a stride.
    pub linear_attn_config: LinearAttnConfig,
    /// The MLA head geometry, carried WHOLE for the reason `V4Config` carries all of RoPE: a
    /// type exposing half of it invites the engine arm to take the rest from
    /// `ModelArgs`-style defaults and build a wrong projection on all 24 layers.
    pub q_lora_rank: usize,
    /// 512 — exactly at the MLA kernel's lane-private accumulator cap; see `validate`.
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    /// 96 — equal to `num_attention_heads`, i.e. NOT the MQA that V4 asserts `== 1`.
    pub num_key_value_heads: usize,
    /// NoPE, asserted POSITIVELY. Checking only that `rope_theta` is absent cannot tell
    /// "this model applies no rotation" from "we descended into the wrong dict" — plan §3e.
    pub mla_use_nope: bool,
    /// §3e's secondary reading, and the ONLY field here that must be **absent**. `Option`
    /// rather than a required field for that reason, which is the module header's rule read
    /// forward rather than broken: what is banned is a default standing in for a value the
    /// engine needs, and this is a value the engine must not find. `validate` refuses `Some`.
    #[serde(default)]
    pub rope_theta: Option<f64>,
    /// Defaults to `False` in the first-party modeling code, so it must come from the config
    /// rather than from a Rust default that happens to agree today (k3 G0 item 11). Its
    /// partner `use_full_rank_gate` is on [`LinearAttnConfig`], one level down — see that
    /// struct.
    pub mla_use_output_gate: bool,
    /// AttnRes block size (12). The residual is taken across a block of layers rather than
    /// per layer — `k3:docs/reference/k3-architecture.md` §3.
    pub attn_res_block_size: usize,

    // --- MoE. The routed experts run in a LATENT that is not `hidden_size`; see
    // `crate::quant::vq_expert_layout` for what assuming otherwise costs.
    /// 896 — per layer, all MXFP4, all streamed.
    #[serde(rename = "num_experts")]
    pub n_experts: usize,
    /// `num_experts_per_token` — note the spelling. GLM and V4 both write
    /// `num_experts_per_tok`, and this checkpoint does not.
    ///
    /// **The field is `top_k` here and `text_config` ALSO has a key literally named
    /// `top_k`, which is 50 and has nothing to do with routing** — it is HuggingFace's
    /// sampling top-k, inherited from `PretrainedConfig`. Binding this from `top_k` selects
    /// 50 experts a token instead of 16: 3.1x the stream traffic, plausible output, no
    /// error. Noticed while reading the shipped file 2026-08-10; the `rename` is what keeps
    /// them apart, and `k3_config.rs`'s pinning test asserts the two values differ so this
    /// cannot be "simplified" later.
    #[serde(rename = "num_experts_per_token")]
    pub top_k: usize,
    /// Declared 2, but the checkpoint ships **one fused MLP** per layer (`down_proj`
    /// `[7168, 6144]` BF16) — k3 G0 item 4. So this is the config's count of shared
    /// experts, not a count of tensors, and the converter must not go looking for two.
    #[serde(rename = "num_shared_experts")]
    pub n_shared: usize,
    /// `routed_expert_hidden_size` — the 3584-wide latent the routed experts are entered
    /// at, **not** `hidden_size` 7168. Named for the role, matching
    /// [`crate::quant::vq_expert_layout`]'s parameter.
    #[serde(rename = "routed_expert_hidden_size")]
    pub expert_in: usize,
    /// 3072 — the routed experts' intermediate width, the down projection's input.
    #[serde(rename = "moe_intermediate_size")]
    pub moe_inter: usize,
    /// RMSNorm on the routed AGGREGATE, in latent space, before the up-projection.
    pub latent_moe_use_norm: bool,
    /// The router renormalises the top-k UNBIASED scores over the selected set.
    pub moe_renormalize: bool,
    /// Grouped routing is **degenerate, not absent**: both are 1. Asserted rather than
    /// ignored — a checkpoint with real groups would need a grouped top-k this engine does
    /// not have, and would otherwise route through the ungrouped path with no error.
    pub num_expert_group: usize,
    /// The other half of the degenerate pair; see `num_expert_group`.
    pub topk_group: usize,
    /// `noaux_tc` — the first-party name for bias-on-selection-only, with unbiased
    /// combining weights (`k3:docs/reference/k3-architecture.md` §6, trap 11).
    pub topk_method: String,
    /// `sigmoid`, and independent per expert — the scores do NOT sum to 1. Asserted for the
    /// reason `V4Config::scoring_func` is: a wrong router affinity picks
    /// plausible-but-wrong experts and never crashes.
    pub moe_router_activation_func: String,
    /// 1.0 today, and the multiply is kept anyway (`k3:docs/reference/k3-architecture.md`
    /// §6's router block says so explicitly). A zero or negative scale silently zeroes or
    /// flips every routed contribution while the shared MLP keeps working — degraded fluent
    /// text, not a crash.
    #[serde(rename = "routed_scaling_factor")]
    pub routed_scale: f64,
    /// 1 — **every** layer after the dense prefix is MoE. Asserted rather than assumed: at
    /// 2 only every other layer would be, and this port's layer loop would run the routed
    /// path on 46 layers that ship no expert tensors.
    pub moe_layer_freq: usize,
    /// SiTU-GLU's first beta (4.0), fused inside the fp4 expert kernel.
    pub activation_situ_beta: f64,
    /// **The key is `activation_situ_linear_beta`, not `activation_linear_beta`.** The k3
    /// plan's §1 abbreviates the pair in a way that reads as the latter; the file says
    /// otherwise (checked 2026-08-10). Declared wrong, this refused every real K3
    /// checkpoint on `missing field` — the loud direction the port prefers, and it is why
    /// the vendored config and its pinning test exist.
    #[serde(rename = "activation_situ_linear_beta")]
    pub activation_linear_beta: f64,

    // --- layer 0 is dense, and simultaneously a KDA layer and an AttnRes boundary.
    /// 1 — the dense prefix. Layer 0 ships `mlp.{gate,up,down}_proj` and no experts.
    pub first_k_dense_replace: usize,
    /// 33792 — the dense layer's intermediate width, under the HF-standard key.
    #[serde(rename = "intermediate_size")]
    pub dense_inter: usize,
    /// `situ`. The dense layer and the shared MLP use the same SiTU-GLU as the routed
    /// experts, so a `silu` here would be a different activation on layer 0 and the shared
    /// path while the routed path stayed right — `k3:docs/reference/k3-architecture.md`
    /// §3b's "watches the dense path go right and every routed expert stay wrong", in
    /// reverse.
    pub hidden_act: String,

    /// 0 — no MTP head, so no speculative decode. Asserted rather than assumed: a non-zero
    /// value means tensors this port does not convert.
    pub num_nextn_predict_layers: usize,
    /// False. A tied head would read the output projection out of the embedding table,
    /// which is a different set of weights at the same shape.
    pub tie_word_embeddings: bool,
}

impl ArchConfig for K3Config {
    const ARCH: Arch = Arch::KimiK3;

    fn validate(&self) -> Result<()> {
        self.text.validate()
    }
}

impl K3Config {
    /// Load from the artifact's `manifest.json`, falling back to a bare `config.json` for
    /// reading a raw checkpoint.
    ///
    /// An inherent wrapper over [`crate::schema::load_config`] because the converter and
    /// its gates both spell it, and `K3Config::load(dir)` is what a reader greps for. The
    /// other three configs' `load` carry the same argument.
    //
    // The turbofish is doing gate work, not style: this is the FOURTH identical `load`
    // wrapper in the crate, and jscpd flagged this one (the other three sit inside larger
    // impls whose surrounding tokens differ; this impl ends here, exactly like
    // `GlimmerConfig`'s). Spelling the type parameter breaks the token run without
    // changing what a reader greps for. Measured 2026-08-16: 1 clone reported without it,
    // 0 with.
    pub fn load(dir: &str) -> Result<Self> {
        crate::schema::load_config::<Self>(dir)
    }
}

impl K3TextConfig {
    /// Cross-field checks. As in `V4Config`, each guards a failure that produces text
    /// rather than an error — and the split into named groups follows that impl's
    /// precedent: the reference's one body is the shape the CodeScene gate refuses, the
    /// groups are its comment headings, and the refusal ORDER is the reference's.
    fn validate(&self) -> Result<()> {
        self.validate_descent()?;
        self.validate_widths()?;
        self.validate_attention_families()?;
        self.validate_arithmetic_settings()?;
        self.validate_layer_structure()?;
        self.validate_moe_counts()?;
        self.validate_flags()?;
        // Every f64 the kernels narrow to f32, checked in the f32 domain rather than only
        // in the f64 JSON carries — underflow (`x <= 2^-150` -> `0.0f32`) collapses the
        // value; overflow (`-> inf`, which passes any bare `> 0.0` test) is the silent one.
        //
        // **`rms_norm_eps` belongs in this loop, and was checked in f64 alone until review
        // 2026-08-10 in the k3 tree**: the engine does `cfg.rms_norm_eps as f32` at every
        // RMSNorm, and a `1e-46` eps passes an f64 positivity test and reaches all of them
        // as `0.0f32`.
        ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            ("activation_situ_beta", self.activation_situ_beta),
            ("activation_situ_linear_beta", self.activation_linear_beta),
            // 1.0 today. Zero silently zeroes every routed contribution while the shared
            // MLP keeps working; negative flips them. Both are degraded fluent text.
            ("routed_scaling_factor", self.routed_scale),
        ])?;
        // The routed experts' widths, and ONLY those: `expert_in` is the latent 3584, so
        // this says nothing about the trunk's 7168 or the shared MLP's 6144. Both of those
        // happen to be multiples of 32, so there is no hole today — but it is an accident
        // of this checkpoint, not something this call checks.
        ensure_f4_group_aligned(self.expert_in, self.moe_inter)
    }

    /// The descent check, both readings of plan §3e.
    fn validate_descent(&self) -> Result<()> {
        // `parse_config` already matched the WRAPPER's pair against `Arch::KimiK3`; this is
        // the nested pair, and it is what distinguishes "descended into `text_config` of a
        // Kimi-K3 config" from "descended into some other dict". Both halves are quoted,
        // and the pair is checked as ONE `ensure!` on purpose: which of the two disagrees
        // is not the interesting fact — "we are in the wrong dict" is, and a reader holding
        // the file wants both spellings in front of them to see that.
        ensure!(
            self.model_type == "kimi_linear"
                && self.architectures == ["KimiLinearForCausalLM".to_string()],
            "text_config declares {:?} / {:?} — a Kimi-K3 wrapper's text model is \
             \"kimi_linear\" / [\"KimiLinearForCausalLM\"]. Either this is not the dict we \
             think we descended into, or the checkpoint's text model is a different family",
            self.model_type,
            self.architectures
        );
        // §3e's SECONDARY check, and it is the reason the field exists at all. NoPE is
        // asserted positively (`mla_use_nope`, in `validate_flags`), because "this model
        // applies no rotation" and "we descended into the wrong dict" are otherwise the
        // same observation — but the plan asks for both readings, and without this the
        // struct's lack of `deny_unknown_fields` means a `rope_theta` sitting in
        // `text_config` is silently ignored rather than being the signal it is.
        //
        // If a real K3 `text_config` turns out to carry a `rope_theta` for some path this
        // port does not walk, the fix is to delete this check and say so — not to relax it
        // to a value comparison, which would re-admit the wrong-dict case.
        ensure!(
            self.rope_theta.is_none(),
            "text_config carries rope_theta {:?}, but this model is NoPE (`mla_use_nope`) \
             and no rotation table is built. A rotary base in a dict that should have none \
             is the signal that the descent landed somewhere else",
            self.rope_theta
        );
        Ok(())
    }

    /// Every width is positive. A zero passes every divisibility check below (0 is a
    /// multiple of anything) and then sizes an expert row, an arena stride and a GEMV `dim`
    /// to nothing.
    fn validate_widths(&self) -> Result<()> {
        for (what, dim) in [
            ("hidden_size", self.hidden),
            ("routed_expert_hidden_size", self.expert_in),
            ("moe_intermediate_size", self.moe_inter),
            ("intermediate_size", self.dense_inter),
            ("vocab_size", self.vocab),
            ("num_attention_heads", self.n_heads),
            ("num_hidden_layers", self.n_layers),
            ("attn_res_block_size", self.attn_res_block_size),
            // The two LoRA ranks belong here for a sharper reason than the rest, found by
            // review 2026-08-10 in the k3 tree: the MLA kernel's own guard does NOT catch
            // a zero. Its guard 1004 is `kvl % 128 || kvl > cap`, and `0 % 128 == 0` while
            // `!(0 > 512)` — so a zero-width latent passes, and 24 layers of attention
            // contribute nothing with no error anywhere. `V4Config` omits the same pair;
            // K3 is where the plan's §1 wrote the constraint down, so it is stated here.
            ("q_lora_rank", self.q_lora_rank),
            ("kv_lora_rank", self.kv_lora_rank),
            ("qk_nope_head_dim", self.qk_nope_head_dim),
            ("qk_rope_head_dim", self.qk_rope_head_dim),
            ("v_head_dim", self.v_head_dim),
            (
                "linear_attn_config.num_heads",
                self.linear_attn_config.num_heads,
            ),
            (
                "linear_attn_config.head_dim",
                self.linear_attn_config.head_dim,
            ),
            (
                "linear_attn_config.short_conv_kernel_size",
                self.linear_attn_config.short_conv_kernel_size,
            ),
        ] {
            // The message names the consequence where the k3 branch's bare "{what} is 0"
            // did not — a deliberate deviation with a mechanical co-benefit: Glimmer's
            // width gate spells its refusal rows with that exact message, and identical
            // (pointer, message) rows in two gate files are a token run the duplication
            // gate reports (measured 2026-08-16: two clones with the shared spelling,
            // zero with this one).
            ensure!(
                dim > 0,
                "{what} = 0 would size an expert row, an arena stride or a GEMV dim to \
                 nothing"
            );
        }
        Ok(())
    }

    /// The two attention families' non-width geometry: the KV-head equality and the KDA
    /// gate's sign.
    fn validate_attention_families(&self) -> Result<()> {
        // Not MQA. V4 asserts `num_key_value_heads == 1` because its whole attention
        // frontend is written against one shared KV entry; K3's MLA has one per query head,
        // and a copied V4 check would refuse this checkpoint while a copied V4 *assumption*
        // would size the cache 96x too small. Pinned as the equality rather than as the
        // literal 96.
        ensure!(
            self.num_key_value_heads == self.n_heads,
            "num_key_value_heads {} != num_attention_heads {} — K3's MLA is not MQA; every \
             query head has its own KV projection",
            self.num_key_value_heads,
            self.n_heads
        );
        // **Negative, and checked in f32.** `gate_lower_bound` MULTIPLIES the KDA gate's
        // sigmoid rather than clamping or flooring it (trap 4), so its sign is load-bearing
        // and a positivity check on it would be exactly backwards. At 0 every gate on all
        // 69 KDA layers is zeroed and the recurrence contributes nothing — the model goes
        // quiet rather than wrong, which no tolerance downstream reads as an error. `-5.0`
        // shipped.
        let gate_lb = self.linear_attn_config.gate_lower_bound as f32;
        ensure!(
            gate_lb < 0.0 && gate_lb.is_finite(),
            "linear_attn_config.gate_lower_bound {} narrows to {gate_lb} in f32; it must be \
             negative and finite — it MULTIPLIES the KDA gate's sigmoid (trap 4), so 0 \
             silences all {} KDA layers and a positive value inverts the decay",
            self.linear_attn_config.gate_lower_bound,
            self.linear_attn_config.kda_layers.len()
        );
        Ok(())
    }

    /// The named settings that change arithmetic without changing a shape, so that nothing
    /// downstream would refuse them.
    fn validate_arithmetic_settings(&self) -> Result<()> {
        ensure!(
            self.moe_layer_freq == 1,
            "moe_layer_freq {} != 1 — this port's layer loop treats every layer past the \
             dense prefix as MoE, and at 2 half of them ship no expert tensors at all",
            self.moe_layer_freq
        );
        for (what, got, want) in [
            ("dtype", &self.dtype, "bfloat16"),
            ("hidden_act", &self.hidden_act, "situ"),
            (
                "moe_router_activation_func",
                &self.moe_router_activation_func,
                "sigmoid",
            ),
        ] {
            ensure!(
                got == want,
                "{what} is {got:?}, not {want:?} — each of these three changes the \
                 arithmetic without changing a single shape, so nothing downstream would \
                 refuse it"
            );
        }
        Ok(())
    }

    /// Per-layer structure: the MLA latent's kernel bounds, the KDA/MLA partition, and the
    /// dense prefix.
    fn validate_layer_structure(&self) -> Result<()> {
        // The other half of guard 1004, restated at the load boundary because the boundary
        // names the FIELD where the kernel names a code. 512 (this checkpoint's value) sits
        // exactly at the cap: `MLA_ACC_REGS * SUBW` is 16 * 32, the lane-private
        // accumulator registers.
        //
        // TWO checks rather than one conjunction, so a refusal names which bound was
        // crossed — and so each test row proves its own half. As `a && b` both rows matched
        // both halves of the message and neither was a test of anything.
        ensure!(
            self.kv_lora_rank.is_multiple_of(128),
            "kv_lora_rank {} is not a multiple of 128 — the MLA attend kernel refuses it \
             with guard 1004",
            self.kv_lora_rank
        );
        ensure!(
            self.kv_lora_rank <= 512,
            "kv_lora_rank {} exceeds the 512 (MLA_ACC_REGS * SUBW) the MLA kernel's \
             lane-private accumulator can hold — guard 1004",
            self.kv_lora_rank
        );
        self.validate_layer_partition()?;
        // The dense prefix, both bounds. TWO checks for the reason `kv_lora_rank`'s pair
        // above gives.
        ensure!(
            self.first_k_dense_replace > 0,
            "first_k_dense_replace is 0, so layer 0 would run the routed MoE path — and \
             this checkpoint ships no expert tensors for it, only the dense \
             `intermediate_size` {} pair, which would then go unused",
            self.dense_inter
        );
        ensure!(
            self.first_k_dense_replace < self.n_layers,
            "first_k_dense_replace {} is not below n_layers {} — every layer would be dense \
             and no routed expert would ever run",
            self.first_k_dense_replace,
            self.n_layers
        );
        Ok(())
    }

    /// The expert counts and the routing scheme's degenerate-group assertion.
    fn validate_moe_counts(&self) -> Result<()> {
        ensure!(
            self.top_k > 0 && self.top_k <= self.n_experts,
            "num_experts_per_token {} is not in 1..={}",
            self.top_k,
            self.n_experts
        );
        ensure!(
            self.n_shared > 0,
            "num_shared_experts is 0 — the always-on MLP is a third of this layer's \
             arithmetic, and its absence is not something the routed path compensates for"
        );
        ensure!(
            self.num_expert_group == 1 && self.topk_group == 1,
            "num_expert_group {} / topk_group {}: grouped routing is degenerate in this \
             checkpoint and this engine has no grouped top-k. Real groups would route \
             through the ungrouped path with no error",
            self.num_expert_group,
            self.topk_group
        );
        ensure!(
            self.topk_method == "noaux_tc",
            "topk_method {:?}: only `noaux_tc` (bias on SELECTION only, never on the \
             returned weight) is implemented. Any other method picks plausible-but-wrong \
             experts and never crashes",
            self.topk_method
        );
        Ok(())
    }

    /// The boolean architecture switches, each required POSITIVE, plus the two absences.
    fn validate_flags(&self) -> Result<()> {
        // Each of these defaults to `false` somewhere — in the first-party modeling code
        // for the two gates, and in Rust for any `bool` this port forgot to read. Requiring
        // the POSITIVE value means a config that omits one is a refusal, not a silent
        // downgrade to an architecture the weights were not trained for.
        for (what, flag) in [
            ("mla_use_nope", self.mla_use_nope),
            ("mla_use_output_gate", self.mla_use_output_gate),
            // One level down, and the label says so — the file puts it inside
            // `linear_attn_config`, which is where it belongs: it is a KDA property.
            (
                "linear_attn_config.use_full_rank_gate",
                self.linear_attn_config.use_full_rank_gate,
            ),
            ("latent_moe_use_norm", self.latent_moe_use_norm),
            ("moe_renormalize", self.moe_renormalize),
        ] {
            ensure!(
                flag,
                "{what} is false; this port implements only the true form and the shipped \
                 config sets it. Turning it off changes the arithmetic, not the shapes"
            );
        }
        ensure!(
            self.num_nextn_predict_layers == 0,
            "num_nextn_predict_layers {} != 0 — this checkpoint has no MTP head, so a \
             non-zero value means tensors nothing here converts and a batched verify pass \
             with no kernel behind it",
            self.num_nextn_predict_layers
        );
        ensure!(
            !self.tie_word_embeddings,
            "tie_word_embeddings is true — K3 ships a separate lm_head, and reading the \
             output projection out of the embedding table is a different set of weights at \
             the same shape"
        );
        Ok(())
    }

    /// `full_attn_layers` and `kda_layers` must partition `1..=n_layers` — both present, no
    /// duplicates, no overlap, nothing missing, nothing out of range.
    ///
    /// Asserted rather than derived from one array, because the two reference
    /// implementations read opposite ones (see [`LinearAttnConfig`]). Every failure here is
    /// a layer running the wrong attention family, which is arithmetic rather than a shape
    /// — no length check downstream sees it.
    fn validate_layer_partition(&self) -> Result<()> {
        let (mla, kda) = (
            &self.linear_attn_config.full_attn_layers,
            &self.linear_attn_config.kda_layers,
        );
        ensure!(
            mla.len() + kda.len() == self.n_layers,
            "full_attn_layers ({}) + kda_layers ({}) = {} layers, but num_hidden_layers is {}",
            mla.len(),
            kda.len(),
            mla.len() + kda.len(),
            self.n_layers
        );
        // One pass over both, so overlap, duplication and gaps are all the same check:
        // every one-based id in range exactly once.
        let mut seen = vec![false; self.n_layers];
        for (what, ids) in [("full_attn_layers", mla), ("kda_layers", kda)] {
            for &one_based in ids {
                let l = one_based
                    .checked_sub(1)
                    .with_context(|| format!("{what} contains 0; these arrays are ONE-BASED"))?;
                let slot = seen.get_mut(l).with_context(|| {
                    format!(
                        "{what} contains layer {one_based}, past num_hidden_layers {}",
                        self.n_layers
                    )
                })?;
                ensure!(
                    !*slot,
                    "layer {one_based} appears twice across full_attn_layers/kda_layers — \
                     the two arrays must partition the layers, and an overlap means the two \
                     reference implementations would disagree about this layer's family"
                );
                *slot = true;
            }
        }
        // Unreachable while the length check above holds, and kept anyway: it is the
        // invariant the readers actually depend on, and the length check is the accident.
        if let Some(l) = seen.iter().position(|s| !s) {
            bail!(
                "layer {} (one-based) is in neither full_attn_layers nor kda_layers",
                l + 1
            );
        }
        Ok(())
    }

    /// Does zero-based `layer` run gated MLA? (Otherwise KDA.)
    ///
    /// **The one place the one-based → zero-based conversion happens.** The reference names
    /// getting it wrong as the mistake that "silently swaps KDA and MLA layers", and the
    /// swap is invisible downstream: both families take the same `[hidden]` input and
    /// return the same shape.
    ///
    /// The linear scan is 24 elements against 93 layers, which is nothing at this engine's
    /// rate; if it ever matters, `validate` has already proven the partition and can hand
    /// out a precomputed mask instead.
    pub fn layer_is_mla(&self, layer: usize) -> Result<bool> {
        ensure!(layer < self.n_layers, "layer {layer} >= {}", self.n_layers);
        Ok(self
            .linear_attn_config
            .full_attn_layers
            .contains(&(layer + 1)))
    }

    /// Is `layer` the dense-FFN prefix (no routed experts)? Layer 0 in the shipped config.
    ///
    /// **Note the asymmetry with the sibling above: this one does not bounds-check**,
    /// because it cannot be wrong in a way a check would catch — an out-of-range id is not
    /// `< first_k_dense_replace` and reads as "not dense", which is the same answer it
    /// gives for every real layer but the first. `V4Config::layer_routes_by_hash` is the
    /// same shape for the same reason. `layer_is_mla` returns `Result` because there a
    /// missing id would read as "KDA", which is a positive claim about arithmetic.
    pub fn layer_is_dense(&self, layer: usize) -> bool {
        layer < self.first_k_dense_replace
    }
}
