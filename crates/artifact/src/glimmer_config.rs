//! Muse Glimmer-30B's config schema — one architecture, one file, per the rule that
//! per-model config types stay separate. Ported from `old:src/artifact/model.rs`'s Glimmer
//! slice (`wt/glimmer-s2` @ 6b7f496), bodies and comments travelling verbatim: in this repo a
//! comment carries the measurement that justified the choice.
//!
//! **This is the first DENSE model in the tree**, so the vocabulary it shares with
//! [`crate::glm_config`] is smaller than it looks — no experts, no routing, no dense/MoE
//! split. What it does share is the shape of the guard: refuse by architecture before serde
//! reads a dimension, then make every cross-field check that separates "this checkpoint" from
//! "a checkpoint that produces fluent wrong text".
//!
//! **What did NOT come with it, and why.** `GLIMMER_STREAM_SLOTS`, `GLIMMER_PIN_SLACK`,
//! `global_bytes`, `layer_bytes`, `resident_bytes`, `floor_bytes` and `partition` are the
//! reference's residency arithmetic. Their arguments are about a *device* — how far a host may
//! run ahead of an asynchronous kernel launch, what `DeviceTier::place` charges for alignment,
//! which layer a budget deficit evicts — and none of that is decidable in this crate, which is
//! host-only and has no pin to size. They land with the engine's Glimmer loop, next to
//! `GlimmerPin`, which is their only caller. A `pub fn` with no caller is the shape
//! `schema.rs` records itself having deleted twice.

use crate::arch::Arch;
use crate::schema::{ArchConfig, ensure_f32_positive};
use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

/// Muse Glimmer-30B, as its `config.json` ships it: a `MuseGlimmerForConditionalGeneration`
/// wrapper around the text model, with a sibling `vision_config` this port does not implement.
///
/// `dtype` sits at the WRAPPER level here rather than inside `text_config`, which is worth
/// naming because the other multimodal checkpoint this tree will meet (Kimi-K3) puts it
/// inside — so a reader who knows one does not know the other.
#[derive(Debug, Clone, Deserialize)]
pub struct GlimmerConfig {
    #[serde(rename = "text_config")]
    pub text: GlimmerTextConfig,
    /// `bfloat16`. The whole checkpoint is BF16 — 59.553 GB across two shards, reconciled
    /// against the index's own `total_size` (`old:docs/reference/glimmer-architecture.md` §7).
    /// The model card's "approximately 4-bit precision, under 20 GB" describes separate GGUF
    /// releases; reading an fp8 or 4-bit export as BF16 is noise at every width, so this is
    /// asserted.
    pub dtype: String,
    /// Must be **absent**. `Option` because what this module bans is a default standing in for
    /// a value the engine needs, and this is a value the engine must not find.
    ///
    /// **`dtype` alone does not prove the weights are unquantized**, which is the whole reason
    /// this field exists. The counter-example is a real checkpoint: Kimi-K3's `config.json`
    /// declares `dtype: "bfloat16"` *and* a `quantization_config` with
    /// `format: "mxfp4-pack-quantized"`. Without this field serde would ignore such a block,
    /// and a packed Glimmer export would parse clean and be read as BF16 at every width —
    /// exactly what the `dtype` message above claims to prevent. Found by review 2026-08-11.
    ///
    /// > **CORRECTED 2026-08-16, by porting it against the real file.** The reference declares
    /// > this field on the WRAPPER only, and its test inserted the block at the wrapper level to
    /// > prove the guard — but K3's vendored `config.json`, the checkpoint the whole argument
    /// > rests on, carries `quantization_config` **inside `text_config`**, with only `dtype` at
    /// > the top. So the guard was checking the one level its own counter-example does not use,
    /// > and a Glimmer export shaped like the checkpoint being cited would have parsed clean.
    /// > [`GlimmerTextConfig::quantization_config`] closes it; `validate` now checks both levels
    /// > and names the one it found, and `glimmer_config.rs`'s test builds its packed document
    /// > from K3's actual bytes rather than from a hand-written block.
    ///
    /// Untyped `Value` on purpose: nothing may act on its contents. K3's schema records why a
    /// `quantization_config` is not trustworthy even when present — its `targets`/`ignore`
    /// lists mis-declare their own scope — so the only claim supported here is the negative
    /// one, that this converter reads unquantized checkpoints.
    #[serde(default)]
    pub quantization_config: Option<serde_json::Value>,
}

/// The `text_config` dict — Muse Glimmer's text model.
///
/// **Every field is REQUIRED**, and for the reason [`crate::schema`]'s header gives: a
/// defaulted dimension does not crash, it produces fluent wrong text. **Hold a
/// [`GlimmerConfig`], not this** — a bare text dict is not evidence that the wrapper around it
/// was Glimmer's, and the wrapper is where `dtype` and `quantization_config` live.
///
/// The fields here are not the interesting part; what is absent is. Eight load-bearing
/// operations appear in no marketing surface. **Four have a config key and are therefore
/// guardable here** — `qk_scale_factor`, `post_norm_eps`, `output_multiplier`,
/// `final_logit_softcapping`, and [`GlimmerTextConfig::validate`] checks all four. The other
/// four are code-only facts no schema can see: the weightless QK-norm, the centered `x*(1+w)`
/// norm form, the sandwich-norm placement, and the normed embedding. Those are the engine's
/// fixtures. `old:docs/reference/glimmer-architecture.md` §9 lists all fifteen traps.
///
/// > **CORRECTED 2026-08-11** in the old tree, by review. It said "only two of them are
/// > visible as a config key at all (`qk_scale_factor`, `post_norm_eps`)" and routed "the
/// > rest" to a fixture — which sent away the two fields `validate` deliberately checks HERE,
/// > and for the sharpest reason in this port: `output_multiplier` and
/// > `final_logit_softcapping` are argmax-invariant, so no greedy gate downstream can ever see
/// > them wrong.
#[derive(Debug, Clone, Deserialize)]
pub struct GlimmerTextConfig {
    /// The NESTED spelling, `muse_glimmer_text`. Carried so `validate` can assert the descent
    /// landed in the text dict rather than in `vision_config` — which is a sibling, declares
    /// `muse_glimmer_vision`, and has a `hidden_size` and a `num_attention_heads` of its own.
    /// That last point is why this check is not decoration: the vision dict would parse
    /// several of the fields below and refuse only on the ones it lacks.
    pub model_type: String,

    // The four dimension serde renames below coincide with `glm_config::ModelConfig`'s, because
    // both checkpoints declare these under the SAME HuggingFace-standard JSON names.
    //
    // **Not factored, and NOT exempted either.** The design argument is this crate's
    // one-type-per-architecture rule: two architectures agreeing on four JSON names is a
    // coincidence of the checkpoints, not a shared contract, and a shared struct becomes the
    // attractor for a FIFTH field that is not shared — `head_dim` below is exactly that field,
    // since GLM derives its head width and Glimmer cannot.
    //
    // The old tree wraps this run in a `jscpd:ignore` region. **Here it needs none: measured
    // 2026-08-16, jscpd reports 0 clones over `crates/` with the markers removed**, because
    // each field carries its own doc comment and the two structs' runs are broken up
    // differently. An exemption that suppresses nothing is a hole in the gate, which is the
    // argument CLAUDE.md makes for deleting four of them, so it is not carried over on faith.
    #[serde(rename = "num_hidden_layers")]
    pub n_layers: usize,
    #[serde(rename = "hidden_size")]
    pub hidden: usize,
    #[serde(rename = "vocab_size")]
    pub vocab: usize,
    #[serde(rename = "num_attention_heads")]
    pub n_heads: usize,
    /// GQA: 2, against 32 query heads — 16 query heads per KV head.
    pub num_key_value_heads: usize,
    /// **128, and NOT `hidden / n_heads`.** 32 x 128 = 4096 while `hidden_size` is 6656, so
    /// `q_proj` is `[4096, 6656]` and is not square. A port that derives it builds a 208-wide
    /// head and indexes past the end of every projection.
    pub head_dim: usize,
    #[serde(rename = "intermediate_size")]
    pub inter: usize,
    /// 1e-5, on `input_layernorm` and `pre_feedforward_layernorm` — and on the weightless
    /// QK-norm and embedding norm.
    pub rms_norm_eps: f64,
    /// **1e-8, and a different value on purpose**: the two POST norms
    /// (`post_attention_layernorm`, `post_feedforward_layernorm`) use this. Three orders of
    /// magnitude apart and assigned by position, so one eps for all four norms is wrong in a
    /// way nothing downstream reads as an error.
    pub post_norm_eps: f64,
    /// 3.87, multiplying **Q only** and applied AFTER the weightless QK-norm. It does not
    /// replace the `head_dim^-0.5` softmax scale; both apply.
    pub qk_scale_factor: f64,
    /// 1/sqrt(26). Pre-multiplies the logits before the tanh softcap below.
    pub output_multiplier: f64,
    /// 20.0. `logits = T * tanh(logits * output_multiplier / T)`.
    ///
    /// **Argmax-invariant, which makes it this model's gate blind spot** rather than a routine
    /// field: `tanh` is strictly increasing and `output_multiplier` is positive, so omitting
    /// both cannot change a greedy pick. Every probability, NLL and confidence value is wrong
    /// regardless. `old:docs/reference/glimmer-architecture.md` §5.
    pub final_logit_softcapping: f64,
    /// 2048, on the `sliding_attention` layers only. The window is inclusive of the current
    /// position: `[p-2047, p]`, exactly 2048 rows.
    pub sliding_window: usize,
    /// 52 entries. **Consumed as the array, never re-derived from a stride** — the
    /// `[s,s,s,full]` period is a fact about this checkpoint, not a rule, and a port that
    /// computes `i % 4 == 3` produces a model that is right until the first checkpoint whose
    /// pattern differs.
    ///
    /// Typed rather than `Vec<String>` so an unknown spelling is refused by **serde**, at
    /// deserialize time and unconditionally, instead of by a `validate` that a caller holding
    /// this struct directly can skip. The realistic wrong value is one dict away: this file's
    /// own `vision_config.layer_types` is `["window_attention", …]`.
    pub layer_types: Vec<LayerKind>,
    /// 52 entries: 500000.0 on sliding layers, **0 on full ones**.
    ///
    /// Read as a BOOLEAN, not as a per-layer base. The first-party code builds ONE cos/sin
    /// table from `rope_parameters.rope_theta` and passes it or `None` per layer, so a port
    /// that builds 52 tables is doing arithmetic nobody asked for — and one that reads the
    /// top-level theta and applies it everywhere rotates the 13 NoPE layers.
    pub layer_rope_theta: Vec<f64>,
    pub rope_parameters: GlimmerRope,
    pub max_position_embeddings: usize,
    /// False — `lm_head.weight` and `embed_tokens.weight` both ship, 2.690 GB each. The
    /// first-party class declares a tied-weights mapping that this checkpoint does not use, so
    /// the class is not evidence and the config is.
    pub tie_word_embeddings: bool,
    /// `silu`. Named `hidden_activation` here, where GLM says `hidden_act`.
    pub hidden_activation: String,
    /// False. No projection in the attention block carries a bias, and none ships.
    pub attention_bias: bool,
    /// Must be **absent here too** — see [`GlimmerConfig::quantization_config`]'s dated
    /// correction for why this level is the one the cited counter-example actually uses.
    ///
    /// `pub(crate)` rather than `pub`: nothing outside this module may act on its contents, and
    /// the only supported claim is the negative one `validate` makes. `Option<Value>` for the
    /// same reason as its wrapper twin — a default standing in for "not packed" is precisely
    /// what is being refused.
    #[serde(default)]
    pub(crate) quantization_config: Option<serde_json::Value>,
}

/// What a Glimmer layer's attention attends over. One entry per layer in
/// [`GlimmerTextConfig::layer_types`].
///
/// An enum rather than the checkpoint's raw string so that an unrecognised spelling cannot
/// reach the engine at all: serde refuses it while deserializing, which is before `validate`
/// and therefore not skippable. As `Vec<String>` a typo in any ONE comparison site read as
/// "not sliding" — i.e. as a positive claim of full attention over the whole prefix, on a
/// layer trained with a 2048 window. That is fluent wrong text, and the kind this port cannot
/// otherwise see.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Attends over `[p-2047, p]` — exactly `sliding_window` rows, inclusive of `p`.
    SlidingAttention,
    /// Attends over the whole prefix, and carries no rotation (`layer_rope_theta` is 0).
    FullAttention,
}

/// Glimmer's `rope_parameters`. Distinct from `glm_config`'s private `RopeParameters` rather
/// than an extension of it, on this crate's one-type-per-architecture rule: GLM's carries theta
/// alone, and adding a `rope_type` there would either be a required field GLM's config need not
/// have or a defaulted one — and a default standing in for a scaling scheme is exactly what
/// this asserts against.
///
/// `rope_type` is asserted `default` rather than ignored: a scaling scheme silently
/// unimplemented keeps every frequency plausible and the text fluent.
#[derive(Debug, Clone, Deserialize)]
pub struct GlimmerRope {
    pub rope_theta: f64,
    pub rope_type: String,
}

impl ArchConfig for GlimmerConfig {
    const ARCH: Arch = Arch::MuseGlimmer;

    fn validate(&self) -> Result<()> {
        ensure!(
            self.dtype == "bfloat16",
            "dtype is {:?}, not \"bfloat16\" — this checkpoint is BF16 throughout (59.553 GB, \
             reconciled against the index's own total_size). The 4-bit artifacts the model \
             card advertises are separate GGUF releases and are not what this converter reads",
            self.dtype
        );
        // The other half of that claim. `dtype` says how the tensors are TYPED; this says they
        // are not additionally packed. K3 ships both at once, so neither check implies the
        // other and the `dtype` message would otherwise promise more than it delivers.
        //
        // **BOTH levels, and the loop is the fix rather than a generalisation** — see the field
        // doc's 2026-08-16 correction. K3 puts the block in `text_config` and only `dtype` at
        // the top, so a single wrapper-level check was blind to the exact shape it cited. The
        // level found is named in the message because that is what the reader has to go look at.
        for (level, block) in [
            ("at the top level", &self.quantization_config),
            ("inside text_config", &self.text.quantization_config),
        ] {
            ensure!(
                block.is_none(),
                "this config carries a quantization_config {level}, so its weights are packed \
                 rather than plain BF16 — `dtype: \"bfloat16\"` does not exclude that (Kimi-K3's \
                 checkpoint declares both, and puts the block inside text_config). This \
                 converter reads the unquantized release"
            );
        }
        self.text.validate()
    }
}

impl GlimmerConfig {
    /// Load from the artifact's `manifest.json`, falling back to a bare `config.json` for
    /// reading a raw checkpoint.
    ///
    /// An inherent wrapper over [`crate::schema::load_config`] because the converter and its
    /// gates both spell it, and `GlimmerConfig::load(dir)` is what a reader greps for. GLM's
    /// `ModelConfig::load` carries the same argument.
    pub fn load(dir: &str) -> Result<Self> {
        crate::schema::load_config(dir)
    }
}

impl GlimmerTextConfig {
    /// True when layer `layer` attends over a sliding window rather than the whole prefix.
    ///
    /// **`Result`, not `Option`, and the difference is not taste.** An absent answer here has
    /// to be an error the caller must handle, because every ergonomic way to collapse an
    /// `Option<bool>` — `unwrap_or(false)`, `unwrap_or_default()`, `matches!(_, Some(true))` —
    /// yields `false`, which is a *positive claim of full attention* about a layer that does
    /// not exist. `unwrap()` is not the escape hatch either: the workspace lint table denies
    /// `unwrap_used`. So the shape steered its only caller into the silently-wrong branch, and
    /// in the old tree it did: `main.rs` wrote `.unwrap_or(false)` on the first try.
    ///
    /// `Result` collapses to `?` instead. Flagged by both standing reviews 2026-08-11 — latent
    /// there (`validate` pins the length and the loop is `0..n_layers`), and it is the line the
    /// engine's layer loop copies.
    pub fn layer_is_sliding(&self, layer: usize) -> Result<bool> {
        let kind = self.layer_types.get(layer).with_context(|| {
            format!(
                "layer {layer} is out of range: this model has {} layers",
                self.layer_types.len()
            )
        })?;
        Ok(*kind == LayerKind::SlidingAttention)
    }

    /// The shape of one [`crate::glimmer::GLIMMER_LAYER_TENSORS`] entry, derived from this
    /// config. `[o, i]` for a projection, `[hidden]` for a norm.
    ///
    /// **One shape table, because the alternatives were three.** In the old tree the pin needed
    /// it to check what it placed, the residency arithmetic needed it to size the tier, and the
    /// name gate needed it to compare against the shipped checkpoint — and the third is what
    /// makes the other two trustworthy: that gate resolves every entry here against
    /// `model.safetensors.index.json`, so this table is not a belief about the checkpoint, it
    /// is checked against it.
    ///
    /// > **PORT NOTE 2026-08-16. That third caller does not exist in this tree yet**, so the
    /// > paragraph above is the reference's provenance rather than a live property here. The
    /// > shipped index is not on this machine; what checks the table today is
    /// > `crates/cli/tests/glimmer_convert.rs`, which builds its synthetic checkpoint FROM this
    /// > function — so the converter's completeness walk and the fixture cannot disagree about
    /// > a shape, and neither can disagree with the config. What that does not close is a name
    /// > or shape wrong in both. The index-side gate lands with the real checkpoint work.
    ///
    /// The pairs matter more than the individual entries. `q_proj` and `self_attn.gate_proj`
    /// are both `[n_heads·head_dim, hidden]` and `k_proj`/`v_proj` are both
    /// `[kv_heads·head_dim, hidden]`, so within each pair a shape check proves nothing and only
    /// the NAME separates them; `o_proj` is the one that is transposed, and `down_proj` the
    /// other. Reading `hidden` for `n_heads·head_dim` — 6656 for 4096 — is the mistake this
    /// exists to make impossible, since `head_dim` is not `hidden / n_heads` here.
    pub fn layer_tensor_shape(&self, tensor: &str) -> Result<Vec<usize>> {
        let q = self.n_heads * self.head_dim;
        let kv = self.num_key_value_heads * self.head_dim;
        Ok(match tensor {
            "self_attn.q_proj" | "self_attn.gate_proj" => vec![q, self.hidden],
            "self_attn.k_proj" | "self_attn.v_proj" => vec![kv, self.hidden],
            "self_attn.o_proj" => vec![self.hidden, q],
            "mlp.gate_proj" | "mlp.up_proj" => vec![self.inter, self.hidden],
            "mlp.down_proj" => vec![self.hidden, self.inter],
            // All four norms, as a suffix rather than four literals.
            //
            // > **CORRECTED 2026-08-11**, by review. This said the suffix means "a fifth norm
            // > is covered rather than silently unmatched". Covered is not correct: the only
            // > fifth norm this architecture could grow is the QK-norm, which is per-head and
            // > `[head_dim]` = 128, not `[hidden]` = 6656 — so the arm would hand back a WRONG
            // > shape where the reader was promised a `bail!`. Inert today (the QK-norm is
            // > weightless and ships no tensor at all, trap 2), and recorded because this is
            // > the one place where "covered" and "correct" diverge.
            t if t.ends_with("layernorm") => vec![self.hidden],
            _ => bail!(
                "{tensor} is not a Muse Glimmer layer tensor — GLIMMER_LAYER_TENSORS and this \
                 table disagree, which means one of them was extended and the other was not"
            ),
        })
    }

    /// Cross-field checks. Each guards a failure that produces text rather than an error.
    fn validate(&self) -> Result<()> {
        // The descent check. `parse_config` matched the WRAPPER's pair; this is the nested
        // spelling, and it is what separates "descended into text_config" from "descended into
        // vision_config" — a sibling dict that carries its own `hidden_size`,
        // `num_attention_heads`, `layer_types` and `rope_parameters`, and would therefore
        // satisfy a good fraction of the schema above before failing on the rest.
        ensure!(
            self.model_type == "muse_glimmer_text",
            "text_config declares model_type {:?} — a Muse Glimmer wrapper's text model is \
             \"muse_glimmer_text\". Either this is not the dict we think we descended into, or \
             (the case worth naming) it is `vision_config`, which carries several of the same \
             keys",
            self.model_type
        );
        self.validate_widths()?;
        self.validate_layer_arrays()?;
        self.validate_named_settings()?;
        // Narrowed to f32, the domain the kernels work in: an f64 positivity test passes
        // values that reach every kernel as 0.0f32.
        //
        // `output_multiplier` and `final_logit_softcapping` are here even though both are
        // argmax-invariant (see the field docs). That is exactly why they need a load-boundary
        // check: no greedy gate downstream can see them being wrong.
        ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            ("post_norm_eps", self.post_norm_eps),
            ("qk_scale_factor", self.qk_scale_factor),
            ("output_multiplier", self.output_multiplier),
            ("final_logit_softcapping", self.final_logit_softcapping),
            (
                "rope_parameters.rope_theta",
                self.rope_parameters.rope_theta,
            ),
        ])
    }

    /// Every width is positive, and the GQA broadcast divides.
    ///
    /// Split out of [`Self::validate`] rather than inlined, and the same for its two siblings
    /// below: the reference's `validate` is one 90-line body whose only structure is comment
    /// headings, which is the shape the CodeScene gate refuses. The groups are the headings.
    fn validate_widths(&self) -> Result<()> {
        // A zero width passes every divisibility check below and then sizes a projection, a
        // KV row or a GEMV `dim` to nothing.
        for (what, dim) in [
            ("hidden_size", self.hidden),
            ("intermediate_size", self.inter),
            ("vocab_size", self.vocab),
            ("num_attention_heads", self.n_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("num_hidden_layers", self.n_layers),
            ("sliding_window", self.sliding_window),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            ensure!(dim > 0, "{what} is 0");
        }
        // GQA, and the direction of the broadcast is the trap. 32 query heads share 2 KV
        // heads, so query head j reads KV head `j / 16` — NOT `j % 2`. Both mappings
        // type-check, both decode fluently, and only one is this model. The divisibility is
        // what makes `j / groups` well-defined at all.
        ensure!(
            // No `kv <= n_heads` conjunct: the zero loop above guarantees both are positive,
            // and for `0 < n_heads < kv` the multiple test is already false. A conjunct that
            // can never be the sole cause of a refusal makes the message ambiguous for free.
            self.n_heads.is_multiple_of(self.num_key_value_heads),
            "num_attention_heads {} is not a positive multiple of num_key_value_heads {} — \
             GQA needs a whole number of query heads per KV head",
            self.n_heads,
            self.num_key_value_heads
        );
        Ok(())
    }

    /// The two per-layer arrays: their lengths, and the invariant that binds them.
    fn validate_layer_arrays(&self) -> Result<()> {
        // **Both must be exactly n_layers long**, and this is the load-bearing length check of
        // the whole schema: everything downstream indexes them by layer id. A short array is an
        // out-of-bounds panic at best; a LONG one is worse, because the extra entries are
        // silently ignored and the file that was meant to describe a different model parses
        // cleanly.
        for (what, got) in [
            ("layer_types", self.layer_types.len()),
            ("layer_rope_theta", self.layer_rope_theta.len()),
        ] {
            ensure!(
                got == self.n_layers,
                "{what} has {got} entries but num_hidden_layers is {} — this array is indexed \
                 by layer id and is the only statement of which layers slide and which rotate",
                self.n_layers
            );
        }
        // The pairing invariant, and it is the reason both arrays are carried rather than one.
        // In this checkpoint a layer is sliding IFF it is rotated: `layer_rope_theta[i] == 0`
        // exactly on the `full_attention` layers. The two arrays are independent in the file,
        // so they CAN disagree — and a disagreement is not a shape error anywhere downstream.
        // It is a model that attends over the wrong rows or rotates a layer that must not be
        // rotated, and either one is fluent.
        //
        // This is the strongest statement the config alone can make, so it is made here rather
        // than left to a fixture. If a future Glimmer ships a rotated full layer, this refuses
        // it — correctly, because this port's attention would not implement it.
        // `zip` + `enumerate` rather than `0..n_layers` and two index expressions: the indices
        // are in bounds only by the statement order of the length check above, and nothing in
        // the workspace lint table denies `indexing_slicing`. Iterating the pair is total, so
        // reordering this function cannot turn it into a panic.
        //
        // An unknown layer kind needs no check here — `LayerKind` refuses it at deserialize
        // time, which is earlier and not skippable.
        for (i, (kind, &theta)) in self
            .layer_types
            .iter()
            .zip(self.layer_rope_theta.iter())
            .enumerate()
        {
            let sliding = *kind == LayerKind::SlidingAttention;
            ensure!(
                sliding == (theta != 0.0),
                "layer {i} is {kind:?} with layer_rope_theta {theta} — in this architecture a \
                 layer is rotated IFF it slides. The arrays disagreeing is not a shape error \
                 downstream: it is a layer attending over the wrong rows, or rotated when it \
                 must not be, and both produce fluent text"
            );
            // Every rotated layer must share the one base the table is built from. The
            // first-party code builds a SINGLE cos/sin table from `rope_parameters.rope_theta`
            // and selects per layer, so a per-layer base that differed from it would be
            // silently ignored rather than honoured.
            ensure!(
                !sliding || theta == self.rope_parameters.rope_theta,
                "layer {i} asks for rope theta {theta} but rope_parameters.rope_theta is {} — \
                 one table is built for the whole model, so a differing per-layer base would \
                 be read and then ignored",
                self.rope_parameters.rope_theta
            );
        }
        Ok(())
    }

    /// The four named settings that change arithmetic without changing a shape, so that
    /// nothing downstream would refuse them.
    fn validate_named_settings(&self) -> Result<()> {
        ensure!(
            self.rope_parameters.rope_type == "default",
            "rope_parameters.rope_type is {:?}, not \"default\" — this port builds an \
             unscaled table, and an unimplemented scaling scheme keeps every frequency \
             plausible and the text fluent (the V4 port's `Defect::RopeNoYarn`)",
            self.rope_parameters.rope_type
        );
        ensure!(
            !self.tie_word_embeddings,
            "tie_word_embeddings is true — this port reads `lm_head.weight` and \
             `embed_tokens.weight` as two separate 2.690 GB tensors, and the shipped \
             checkpoint declares them untied"
        );
        ensure!(
            !self.attention_bias,
            "attention_bias is true — no projection in this port's attention block reads a \
             bias tensor, and none ships in the checkpoint"
        );
        // SwiGLU with a different activation changes the arithmetic without changing one
        // shape, so nothing downstream would refuse it.
        ensure!(
            self.hidden_activation == "silu",
            "hidden_activation is {:?}, not \"silu\" — the MLP is SwiGLU and the gate's \
             activation is not a shape",
            self.hidden_activation
        );
        Ok(())
    }
}
