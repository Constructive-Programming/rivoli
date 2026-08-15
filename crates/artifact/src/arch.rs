//! Which model architecture an artifact holds — the one discriminant, as IDENTITY only.
//!
//! There is deliberately **no `--model`/`--arch` flag**: the artifact IS the model, and a
//! flag naming the architecture is a flag that can disagree with the weights it
//! describes. Disagreeing is not a crash — it launches the wrong decode path and produces
//! fluent wrong text. An unrecognised value REFUSES at startup; falling back to GLM is
//! the specific mistake worth naming, because it is the only value that would look like
//! it worked.
//!
//! The old tree's `arch.rs` also carried presentation policy (`attn_modes`,
//! `hidden_flags` — what `--help` shows per arch). That half is deliberately NOT ported:
//! the M4 legality table (`decide(arch, flag) -> Outcome`) replaces it with one decider,
//! because two places that can independently judge a configuration is the silent-wrong
//! hazard the old file itself warned about.
/// The architectures the engine has a decode path for. Parsed from the artifact manifest's
/// The architectures the engine has a decode path for. Parsed from the artifact manifest's
/// `architectures` / `model_type`; an unrecognised value must REFUSE at startup rather than
/// fall back to a default. Falling back to GLM is the specific mistake worth naming: it is
/// the only value that would look like it worked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arch {
    /// GLM-5.2: MLA (multi-head latent attention) + q-LoRA, with the DSA lightning indexer.
    GlmMoeDsa,
    /// DeepSeek-V4-Flash-0731: shared-K=V MQA, sliding window + per-layer KV compression.
    DeepseekV4,
    /// Kimi-K3: 69 KDA (linear-attention) layers interleaved with 24 gated MLA layers, and
    /// routed experts that run in a 3584-wide LATENT rather than at `hidden_size` 7168.
    KimiK3,
    /// Muse Glimmer-30B: the first DENSE model here — no experts, no routing, nothing
    /// streamed. 52 layers of GQA 32Q/2KV with a sigmoid output gate, three sliding-window
    /// (2048) layers to every full one, and RoPE on the sliding layers only.
    MuseGlimmer,
}

impl Arch {
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
    pub fn from_manifest_str(s: &str) -> Option<Self> {
        match s {
            "GlmMoeDsaForCausalLM" | "glm_moe_dsa" => Some(Arch::GlmMoeDsa),
            "DeepseekV4ForCausalLM" | "deepseek_v4" => Some(Arch::DeepseekV4),
            "KimiK3ForConditionalGeneration" | "kimi_k3" => Some(Arch::KimiK3),
            "MuseGlimmerForConditionalGeneration" | "muse_glimmer" => Some(Arch::MuseGlimmer),
            _ => None,
        }
    }

    /// Short kebab name, for help headers and log lines.
    pub fn name(self) -> &'static str {
        match self {
            Arch::GlmMoeDsa => "glm-moe-dsa",
            Arch::DeepseekV4 => "deepseek-v4",
            Arch::KimiK3 => "kimi-k3",
            Arch::MuseGlimmer => "muse-glimmer",
        }
    }

    /// One line naming the attention family, since that is what actually differs.
    pub fn summary(self) -> &'static str {
        match self {
            Arch::GlmMoeDsa => "MLA + q-LoRA, DSA lightning indexer",
            Arch::DeepseekV4 => "shared-K=V MQA, sliding window + per-layer KV compression",
            Arch::KimiK3 => "69 KDA + 24 gated MLA (NoPE), latent-space routed experts",
            Arch::MuseGlimmer => "dense GQA 32Q/2KV, gated, 3 sliding (2048) per full layer",
        }
    }
}
