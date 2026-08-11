//! Which model architecture an artifact holds, and which flags that makes meaningful.
//!
//! There is deliberately **no `--model`/`--arch` flag**. `main.rs`'s opening paragraph
//! states the design — *the artifact IS the model* — and a flag naming the architecture is
//! a flag that can disagree with the weights it describes. Disagreeing is not a crash: it
//! launches the MLA decode path against an MQA checkpoint and produces fluent wrong text.
//! `--attn auto` is the precedent already in the tree: sniff the artifact, and make the
//! EXPLICIT form bail when it contradicts what is there.
//!
//! This module is **presentation policy only** — what `--help` shows. The refusals
//! themselves stay where they already are (`Config::validate_backend`, `ModelConfig`'s
//! boundary checks, `resolve_attn`). Two places that can independently decide whether a
//! configuration is legal is the same silent-wrong hazard in a different coat; so what is
//! here is the *shown* list, and `arch_help_matches_the_parser` below pins it to the
//! parser rather than trusting the two to stay in step.

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

    /// The values `--attn` accepts, or `None` when the flag does not apply at all.
    ///
    /// `None` is not "no modes" — it is "this is not a choice". V4's attention is fixed by
    /// the weights: window 128, per-layer compression at ratio 4 or 128, and an indexer
    /// only on the ratio-4 layers. There is no row-selection policy to pick, so offering
    /// the flag at all would be offering a knob that cannot turn.
    ///
    /// K3 is `None` for the same reason arrived at differently: which of its 93 layers is KDA
    /// and which is gated MLA is a *partition declared in the config* (`linear_attn_config`)
    /// and confirmed by which layers ship `kv_a_proj_with_mqa`. A KDA layer has no row set to
    /// select from at all — it carries a recurrent state, not a KV cache.
    ///
    /// Muse Glimmer is `None` for the third distinct reason, and it is the most literal: the
    /// config ships a 52-entry `layer_types` array naming each layer `sliding_attention` or
    /// `full_attention`, and a matching `layer_rope_theta`. The window is 2048 because the
    /// weights were trained that way. `--attn streaming` would be offering to re-derive, at
    /// runtime and differently, a row set the checkpoint already states.
    pub fn attn_modes(self) -> Option<&'static [&'static str]> {
        match self {
            Arch::GlmMoeDsa => Some(&["auto", "dense", "streaming", "dsa", "misa"]),
            Arch::DeepseekV4 | Arch::KimiK3 | Arch::MuseGlimmer => None,
        }
    }

    /// Why `--attn` is absent, shown in place of the flag. Only meaningful when
    /// [`attn_modes`](Self::attn_modes) is `None`.
    pub fn attn_fixed_note(self) -> &'static str {
        match self {
            Arch::GlmMoeDsa => "",
            Arch::DeepseekV4 => {
                "attention is fixed by the weights (window 128, per-layer compression \
                 4/128, indexer on the ratio-4 layers)"
            }
            Arch::KimiK3 => {
                "attention is fixed by the weights (`linear_attn_config` partitions the 93 \
                 layers into 69 KDA and 24 gated MLA; a KDA layer keeps a recurrent state, \
                 not a KV cache, so there are no rows to select)"
            }
            Arch::MuseGlimmer => {
                "attention is fixed by the weights (`layer_types` names each of the 52 layers \
                 sliding or full, `layer_rope_theta` says which are rotated, and the sliding \
                 window is 2048 because that is what was trained)"
            }
        }
    }

    /// Flags that only mean something on one architecture, hidden from the resolved help
    /// of the others. Listed by clap id.
    ///
    /// These are the *attention-shaped* knobs; `--mode` is deliberately NOT here, because
    /// which formats an artifact admits is a property of the files in it (does it carry
    /// `.vq3`, `.i4`, both?) rather than of the architecture, and the engine already
    /// decides that from the artifact.
    pub fn hidden_flags(self) -> &'static [&'static str] {
        match self {
            Arch::GlmMoeDsa => &[],
            // Every one of these is a `--attn streaming`/`misa` knob, and none of V4, K3 or
            // Glimmer has an `--attn`. The MTP flags are deliberately NOT hidden on any, even
            // though K3 ships `num_nextn_predict_layers: 0` and so has no head to run: V4's
            // branch says so with an `info!` instead (its fp4 MoE kernel refuses nrow != 1),
            // and one mechanism for one fact beats two that can disagree.
            Arch::DeepseekV4 | Arch::KimiK3 => &["attn", "sinks", "window", "misa_heads"],
            // **Glimmer hides four more, and they are the residency knobs.** V4 and K3 both
            // stream routed experts, so `--mode`, `--cache-policy`, `--max-mem` and `--trace`
            // name real things for them. Glimmer is dense: there is no pool to evict from, no
            // remainder of the budget to grow it with, no second routed format and no
            // routed-expert access to trace. `run_glimmer` refuses all four.
            //
            // Split from the shared arm 2026-08-11, by review, which found `run_glimmer`
            // telling the user each refused flag "is hidden from this artifact's --help"
            // while four of the eight were still listed there. Fixing the sentence was the
            // other option and is the wrong one: a flag a model cannot honour should not be
            // advertised in that model's help, which is what this function is for.
            Arch::MuseGlimmer => &[
                "attn",
                "sinks",
                "window",
                "misa_heads",
                "mode",
                "cache_policy",
                "max_mem",
                "trace",
                "moe_gain",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    /// Every architecture, so a new variant cannot ship with an arm nobody looked at.
    ///
    /// Added 2026-08-10 with `KimiK3`, after review found that four of its six arms were
    /// unasserted — and so were V4's. The gap that matters: splitting a shared
    /// `DeepseekV4 | KimiK3` arm and giving one of them an empty `hidden_flags` reddened
    /// nothing, and `--help` would then advertise `--attn` on an architecture whose only
    /// refusal path is a `bail!` in a branch that architecture does not go through.
    ///
    /// Written as an INVARIANT over the list rather than a table of expected strings: the
    /// strings are prose that will be reworded, while "no `--attn` means those four knobs are
    /// hidden" is the property `artifact_resolved_help` actually depends on.
    #[test]
    fn every_arch_answers_its_own_help_questions() {
        // The `--attn`-shaped knobs. Named here rather than imported, because the point is to
        // fail if `hidden_flags` and this list drift — see `arch_help_matches_the_parser` for
        // the same idea against the parser.
        const ATTN_KNOBS: [&str; 4] = ["attn", "sinks", "window", "misa_heads"];
        for arch in [
            Arch::GlmMoeDsa,
            Arch::DeepseekV4,
            Arch::KimiK3,
            Arch::MuseGlimmer,
        ] {
            let name = arch.name();
            assert!(
                !name.is_empty() && name == name.to_lowercase() && !name.contains(' '),
                "{name:?} is not a kebab name fit for a help header"
            );
            assert!(!arch.summary().is_empty(), "{name} has no summary");
            match arch.attn_modes() {
                // An architecture WITH `--attn` must not hide the flag it offers.
                Some(modes) => {
                    assert!(
                        modes.contains(&"auto"),
                        "{name}: --attn has no default mode"
                    );
                    assert!(
                        !arch.hidden_flags().contains(&"attn"),
                        "{name} offers --attn and also hides it"
                    );
                }
                // ...and one WITHOUT it must hide all four knobs and say why, since `--help`
                // hiding a flag is the only notice a user gets.
                None => {
                    for knob in ATTN_KNOBS {
                        assert!(
                            arch.hidden_flags().contains(&knob),
                            "{name} has no --attn but leaves --{knob} in its help"
                        );
                    }
                    assert!(
                        !arch.attn_fixed_note().is_empty(),
                        "{name} hides --attn without saying why"
                    );
                }
            }
        }
    }

    /// An unknown architecture must not resolve. The engine refuses it at startup; this
    /// pins the *recogniser* so a future rename cannot quietly make an unknown checkpoint
    /// look like a known one.
    #[test]
    fn unknown_architectures_do_not_resolve() {
        for s in [
            "",
            "DeepseekV3ForCausalLM",
            "GlmForCausalLM",
            "glm_moe",
            "deepseek_v4 ",
            // K3's own NESTED spellings, which must not resolve here. See
            // `from_manifest_str`: they name the linear-attention family the text model
            // belongs to, not this checkpoint, and `K3Config::validate` is where they are
            // asserted — with the key it descended through available to quote.
            "KimiLinearForCausalLM",
            "kimi_linear",
            // Glimmer's NESTED spelling, on the same argument: it names this checkpoint's
            // text half, and `GlimmerTextConfig::validate` is where it is asserted. Also the
            // vision half's, which must never resolve as a decode path at all.
            "muse_glimmer_text",
            "muse_glimmer_vision",
        ] {
            assert_eq!(Arch::from_manifest_str(s), None, "{s:?} must not resolve");
        }
        // ...and the two that must, so this test can fail in BOTH directions. A recogniser
        // test that only ever asserts rejection passes just as well on a function that
        // returns None unconditionally.
        assert_eq!(
            Arch::from_manifest_str("GlmMoeDsaForCausalLM"),
            Some(Arch::GlmMoeDsa)
        );
        assert_eq!(
            Arch::from_manifest_str("deepseek_v4"),
            Some(Arch::DeepseekV4)
        );
        // Both of K3's TOP-level spellings, since the wrapper is the level that names it.
        for s in ["KimiK3ForConditionalGeneration", "kimi_k3"] {
            assert_eq!(Arch::from_manifest_str(s), Some(Arch::KimiK3), "{s:?}");
        }
        for s in ["MuseGlimmerForConditionalGeneration", "muse_glimmer"] {
            assert_eq!(Arch::from_manifest_str(s), Some(Arch::MuseGlimmer), "{s:?}");
        }
    }

    /// The shown list must be the parsed list. This module is presentation policy, so the
    /// failure it invites is help that advertises a mode the parser rejects (or omits one
    /// it accepts) — drift that no user-facing test would otherwise catch, because help
    /// text is the one output nothing asserts on.
    ///
    /// It is pinned against the literal in `main.rs`'s `#[arg(value_parser = [...])]` for
    /// `--attn`; if you change one, this fails and tells you about the other.
    #[test]
    fn arch_help_matches_the_parser() {
        let main_rs = include_str!("main.rs");
        let line = main_rs
            .lines()
            .find(|l| l.contains("value_parser") && l.contains("\"misa\""))
            .expect("the --attn value_parser literal moved; re-point this test");
        // Only the values INSIDE `value_parser = [...]`. The same line carries
        // `default_value = "auto"`, so counting quotes across the whole line reads six
        // values where the parser accepts five — an off-by-one that would have made this
        // test fail against correct code and then be "fixed" by loosening it.
        // Anchored on `value_parser = [`, not on the first `[` — the attribute opens with
        // `#[arg(`, so a bare `split_once('[')` captures the attribute itself and reports
        // `arg(long` as an accepted value.
        let list = line
            .split_once("value_parser = [")
            .and_then(|(_, r)| r.split_once(']'))
            .expect("value_parser list is not a bracketed literal any more")
            .0;
        let parsed: Vec<&str> = list
            .split(',')
            .map(|s| s.trim().trim_matches('"'))
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(
            Arch::GlmMoeDsa.attn_modes().expect("glm has --attn"),
            parsed.as_slice(),
            "the --attn help list and the parser's accepted values have drifted"
        );
    }
}
