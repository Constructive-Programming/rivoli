//! Recognising which model architecture an artifact holds — the manifest half of the
//! identity. The enum itself is [`rivoli_core::legality::Arch`], re-exported here.
//!
//! **Why the enum moved down to core and the recogniser did not.** The (arch × flag)
//! legality table is the one place allowed to judge a configuration, and it lives in
//! `rivoli-core`, which by the workspace's DAG cannot depend on this crate. So the
//! architecture *identity* is core's and the knowledge of which manifest spellings name it
//! stays here, with the manifest reader. That is the split `ModelConfig::scoring` already
//! uses: core owns the vocabulary, this crate maps its raw input into it.
//!
//! There is deliberately **no `--model`/`--arch` flag**, an unrecognised value refuses at
//! startup, and the old tree's presentation policy (`attn_modes`, `hidden_flags`) is not
//! ported — `rivoli_core::legality` owns all three arguments; this header does not repeat
//! them (a second copy is the next drifted doc).

pub use rivoli_core::legality::Arch;

/// Recognise an architecture from a manifest `architectures[0]` or `model_type` string.
/// Both spellings are accepted because the two checkpoints disagree about which field
/// carries the answer: GLM's manifest ships `architectures: ["GlmMoeDsaForCausalLM"]`,
/// DeepSeek-V4 ships both `architectures: ["DeepseekV4ForCausalLM"]` and
/// `model_type: "deepseek_v4"`.
///
/// **Kimi-K3 is recognised on the TOP level only, and that is the whole subtlety.** Its
/// `config.json` is a `KimiK3ForConditionalGeneration` multimodal wrapper whose nested
/// `text_config` declares a *different* pair — `KimiLinearForCausalLM` / `kimi_linear`,
/// the linear-attention family K3's text model belongs to. `K3Config` descends into that
/// dict, so a recogniser that descended first would look for `kimi_k3` where the file says
/// `kimi_linear` and refuse the real checkpoint. The nested pair is not accepted here
/// either: `KimiLinear` names a family, not this checkpoint, and admitting it would let a
/// foreign member of that family resolve as K3. `K3Config::validate` asserts the nested
/// spelling as a secondary check, where it can quote the key it descended through.
///
/// **Muse Glimmer has the same wrapper shape and is recognised the same way** — top level
/// only. Its `text_config` declares `muse_glimmer_text`, which is deliberately NOT accepted
/// here: it names the text half of this checkpoint, so admitting it would let a bare text
/// dict resolve as the whole model. `GlimmerTextConfig::validate` asserts it instead.
///
/// **Qwen3.8-Flash-Next is the third wrapper shape, and its near-miss is a NEW one: the
/// nested `vision_config.model_type` is the ROOT string verbatim.** The checkpoint's
/// `config.json` (revision `236dfdf2…`, vendored under `docs/measurement/qwen-reference/`) is
/// a `Qwen4ExpForConditionalGeneration` / `qwen4_exp` wrapper whose `text_config.model_type`
/// is `qwen4_exp_text` and whose `vision_config.model_type` is `qwen4_exp` — the same string
/// the root carries. So the mistake K3 documents (descend, then look for the wrapper's
/// spelling) is joined here by a second one: a recogniser that SEARCHED the document rather
/// than reading its root would match the vision block, and a vision-only or hand-trimmed
/// config would resolve as the whole model. [`crate::schema::arch_of_named`] reads the root
/// only. The nested `qwen4_exp_text` spelling is refused here for K3's and Glimmer's reason —
/// it names the text half, not this checkpoint — and is asserted by the config type's own
/// `validate` when that lands, where it can quote the key it descended through.
///
/// > The port plan's identity note read "a nested `text_config` spelling is asserted by config
/// > validate, never accepted by the sniff" as if the nesting were hypothetical. It is not:
/// > the nesting is mandatory and present, and every architecture dimension lives under
/// > `text_config` (S0, 2026-08-31). The rule it states is unchanged and now load-bearing.
///
/// A free function rather than an inherent `Arch::from_manifest_str`, because the type is
/// core's now and an inherent impl has to live with its type. Nothing else changed: this
/// is the same match, in the same crate as the manifests it knows about.
pub fn from_manifest_str(s: &str) -> Option<Arch> {
    match s {
        "GlmMoeDsaForCausalLM" | "glm_moe_dsa" => Some(Arch::GlmMoeDsa),
        "DeepseekV4ForCausalLM" | "deepseek_v4" => Some(Arch::DeepseekV4),
        "KimiK3ForConditionalGeneration" | "kimi_k3" => Some(Arch::KimiK3),
        "MuseGlimmerForConditionalGeneration" | "muse_glimmer" => Some(Arch::MuseGlimmer),
        "Qwen4ExpForConditionalGeneration" | "qwen4_exp" => Some(Arch::QwenFlashNext),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fifth architecture's two accepted spellings, and the two near-misses that must not
    /// resolve.
    ///
    /// Its own test — the first in this file — because "recognised" and "recognised at the
    /// right LEVEL" are different claims here, and only the second is true: `qwen4_exp` is
    /// both the root `model_type` and `vision_config`'s, so the doc's root-only rule is the
    /// whole content of the recognition. `qwen4_exp_text` is what a descending recogniser
    /// would look for; `qwen3_next` is a real, foreign member of the same family, which is
    /// the shape that must not be admitted by a prefix or a "close enough" match.
    ///
    /// > **RENAMED 2026-08-31 (S1 review).** This was called
    /// > `qwen_resolves_from_the_root_spellings_only`, which overstated it: this function
    /// > takes a bare `&str` and never sees a document, so it cannot observe what LEVEL a
    /// > spelling came from. The name now says what it checks — four strings, two admitted
    /// > and two refused. The root-only claim belongs to whoever reads the document, and is
    /// > gated there, against the vendored `config.json`:
    /// > `schema::tests::the_shipped_qwen_wrapper_resolves_from_its_root_and_a_promoted_block_does_not`.
    #[test]
    fn qwen_accepts_its_two_root_spellings_and_refuses_the_two_near_misses() {
        for s in ["Qwen4ExpForConditionalGeneration", "qwen4_exp"] {
            assert_eq!(from_manifest_str(s), Some(Arch::QwenFlashNext), "{s:?}");
        }
        for s in ["qwen4_exp_text", "qwen3_next"] {
            assert_eq!(
                from_manifest_str(s),
                None,
                "{s:?} must not resolve: it names the text half or another family member, and \
                 admitting it would let a trimmed or foreign config choose a decode path"
            );
        }
    }
}
