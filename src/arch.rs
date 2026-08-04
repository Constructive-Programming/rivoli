//! Which architecture an artifact is.
//!
//! # PLACEHOLDER — owned by the multi-model branch, not by S1a
//!
//! The coordinator is landing the real `src/arch.rs` on `worktree-feat+multi-model`: this
//! enum plus the static capability table (which `--mode`/`--attn`/… each architecture
//! admits) and the `--help` re-rendering that reads it. This file exists only so S1a's
//! artifact parse has the agreed type to produce before that branch is available to rebase
//! onto. **At rebase, take the branch's version of this file wholesale.** It is deliberately
//! a single file with no S1a-specific content, so that resolution is "take theirs".
//!
//! The one rule it must keep: there is exactly ONE architecture discriminant in the tree.
//! `src/artifact/model.rs` produces it and nobody re-derives it. **At rebase, take the
//! branch's file — and if its `from_manifest_str` lacks the `deepseek_v4` /
//! `DeepseekV4ForCausalLM` arms, re-add them, or "take theirs" silently un-ports S1a.**
//! `src/artifact/model.rs` depends on those four strings and on both variant names.

/// The architecture an artifact declares. Derived from the manifest, never from a flag —
/// the artifact IS the model, so an explicit override could only ever disagree with the
/// weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// GLM-5.2: MLA attention with a DSA lightning indexer, a dense prefix, int3-vq/int4
    /// experts. Everything this engine runs today.
    GlmMoeDsa,
    /// DeepSeek-V4-Flash-0731: shared-KV MQA, hyper-connection residuals, a hash-routed
    /// prefix, and FP4 experts.
    DeepseekV4,
}

impl Arch {
    /// Resolve one manifest string. Accepts BOTH vocabularies a HuggingFace-style config
    /// uses — `model_type` (`"glm_moe_dsa"`) and an `architectures` entry
    /// (`"GlmMoeDsaForCausalLM"`) — because the two never collide and a caller holding one
    /// string should not have to say which kind it is.
    ///
    /// `None` is "not a name this build knows". The caller turns that into a refusal that
    /// quotes the string; it must NEVER be turned into a default. An artifact whose
    /// architecture we cannot name is one whose decode path we cannot choose.
    pub fn from_manifest_str(s: &str) -> Option<Self> {
        Some(match s {
            "glm_moe_dsa" | "GlmMoeDsaForCausalLM" => Arch::GlmMoeDsa,
            "deepseek_v4" | "DeepseekV4ForCausalLM" => Arch::DeepseekV4,
            _ => return None,
        })
    }
}
