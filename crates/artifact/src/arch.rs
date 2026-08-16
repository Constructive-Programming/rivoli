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
/// A free function rather than an inherent `Arch::from_manifest_str`, because the type is
/// core's now and an inherent impl has to live with its type. Nothing else changed: this
/// is the same match, in the same crate as the manifests it knows about.
pub fn from_manifest_str(s: &str) -> Option<Arch> {
    match s {
        "GlmMoeDsaForCausalLM" | "glm_moe_dsa" => Some(Arch::GlmMoeDsa),
        "DeepseekV4ForCausalLM" | "deepseek_v4" => Some(Arch::DeepseekV4),
        "KimiK3ForConditionalGeneration" | "kimi_k3" => Some(Arch::KimiK3),
        "MuseGlimmerForConditionalGeneration" | "muse_glimmer" => Some(Arch::MuseGlimmer),
        _ => None,
    }
}
