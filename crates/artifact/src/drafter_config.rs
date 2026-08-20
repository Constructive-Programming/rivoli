//! **The block drafter's config and tensor census — the one statement of what a drafter
//! sub-artifact contains.**
//!
//! The DFlash drafter (`meta-models/Muse-Glimmer-30B-assistant`; `old:docs/reference/glimmer-architecture.md`
//! §11) is a **sub-artifact, not an architecture**: it decodes nothing on its own — it owns
//! neither embedding nor lm_head — so it deliberately does NOT implement
//! [`crate::schema::ArchConfig`] and mints no `Arch` variant. An `Arch` is a row in core's (arch × flag)
//! legality table, and a drafter alone must never be servable; what M17d must make `--mtp`
//! legal by is a GLIMMER artifact that HAS a drafter, which is a fact about the artifact, not a
//! fifth architecture — and core's `decide(arch, flag)` takes no such input today, so that is a
//! signature change M17d owes, not a description of the current table. The `model_type` refusal below plays the role `parse_config`'s arch
//! check plays — wrong-document reported as itself, before serde names a missing width.
//!
//! [`DrafterConfig::census`] is read by the converter today (what must be in the checkpoint).
//! Two more readers are RESERVED and neither exists yet: the converter's end-to-end fixture
//! (what to synthesize) and the engine's drafter pin (what to map), both M17b/c. Stated as
//! reservations rather than as fact — the point of one statement is that a shape wrong in the
//! schema reddens in a gate instead of being agreed with, and that only pays once those land.
//! `GLIMMER_LAYER_TENSORS` next door is the same pattern for the target.

use anyhow::{Context, Result, ensure};
use serde::Deserialize;

/// The eleven tensors every drafter layer ships, as `layers.{l}.{}.weight`.
///
/// Four attention projections, the two WEIGHTED per-head QK-norms (`[head_dim]` — tensors
/// the target's weightless QK-norm does not even ship), three MLP projections, two plain
/// pre-norm layernorms. No sandwich norms, no attention gate, no biases — the drafter's
/// layer is not the target's, and defaulting any of this from `GLIMMER_LAYER_TENSORS` is
/// the S6 item-1 mistake the spec warns about.
pub const DRAFTER_LAYER_TENSORS: [&str; 11] = [
    "input_layernorm",
    "post_attention_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

/// `config.json` of the assistant checkpoint (or the drafter sub-artifact's `manifest.json`,
/// which is that config plus a `format` section). Field names are the reference's
/// (`configuration_muse_glimmer_assistant.py`); defaults mirror the reference's defaults so
/// a config that omits a defaulted field parses as the reference would build it.
/// Absent fields take the REFERENCE's defaults (`configuration_muse_glimmer_assistant.py`,
/// verbatim), so a config that omits one parses as the reference would build it. One block
/// rather than ten one-value serde callables: the block can be read against
/// `glimmer-architecture.md` §11 in a glance, which ten scattered `fn`s cannot.
#[derive(Deserialize, Debug)]
#[serde(default)]
pub struct DrafterConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    /// `None` means all-sliding, exactly as the reference's `__post_init__` fills it.
    pub layer_types: Option<Vec<String>>,
    pub block_size: usize,
    pub mask_token_id: usize,
    /// `None` means the reference default `[1, 13, 25, 37, 49]`.
    pub target_layer_ids: Option<Vec<usize>>,
    /// §11's `rope_theta 500000` — bound because the TARGET's is a different number, and a
    /// drafter whose RoPE base is inherited from it is §11's own "off by `ctx_len` here is a
    /// silent quality loss, not a crash" in another guise. Added 2026-08-16 after review found
    /// it the one parameter of §11's block this schema did not read.
    pub rope_theta: f64,
}

impl Default for DrafterConfig {
    fn default() -> Self {
        Self {
            // Empty, and `validate` refuses it: the wrong document must be reported as itself.
            // There IS no default architecture — every other field below is a width the
            // reference ships, and this one is the claim that the file is the drafter at all.
            model_type: String::new(),
            hidden_size: 6656,
            intermediate_size: 19968,
            num_hidden_layers: 5,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-5,
            sliding_window: 2048,
            layer_types: None,
            block_size: 16,
            mask_token_id: 201_818,
            target_layer_ids: None,
            rope_theta: 500_000.0,
        }
    }
}

impl DrafterConfig {
    /// Parse and validate one document, refusing anything that does not declare itself the
    /// assistant. The check comes FIRST so a target config handed to the drafter path is
    /// reported as the wrong document rather than as whichever field it lacks.
    fn parse(text: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(text)
            .context("the assistant config does not parse as a drafter schema")?;
        cfg.validate().map(|()| cfg)
    }

    /// [`Self::parse`] over `<dir>/manifest.json`, falling back to `<dir>/config.json` —
    /// the same two spellings `schema::load_config` accepts, so the sub-artifact reads
    /// exactly as its source checkpoint did.
    pub fn load(dir: &str) -> Result<Self> {
        let manifest = format!("{dir}/manifest.json");
        let path = if std::path::Path::new(&manifest).is_file() {
            manifest
        } else {
            format!("{dir}/config.json")
        };
        let text = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
        Self::parse(&text).with_context(|| format!("parse {path}"))
    }

    /// The reference's `__post_init__` default for absent `target_layer_ids`.
    pub fn target_layer_ids(&self) -> Vec<usize> {
        self.target_layer_ids
            .clone()
            .unwrap_or_else(|| vec![1, 13, 25, 37, 49])
    }

    fn validate(&self) -> Result<()> {
        // FIRST, so a target config handed to the drafter path is reported as the wrong
        // DOCUMENT rather than as whichever width happens to look implausible. Every width
        // below has a default, so nothing else in this function can name the real problem.
        ensure!(
            self.model_type == "muse_glimmer_assistant",
            "this config declares model_type {:?}, not \"muse_glimmer_assistant\" — it is not \
             a DFlash drafter checkpoint",
            self.model_type
        );
        // The zero guard stays and must stay FIRST: `is_multiple_of(0)` is `self == 0`, so a
        // config with no KV heads would otherwise pass this on `num_attention_heads == 0` and
        // fail one loop later with the wrong message.
        ensure!(
            self.num_key_value_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
            "heads {} do not divide into KV heads {}",
            self.num_attention_heads,
            self.num_key_value_heads
        );
        for (what, v) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("head_dim", self.head_dim),
            ("sliding_window", self.sliding_window),
            // Added 2026-08-16: `num_attention_heads: 0` used to pass EVERY refusal here —
            // `0.is_multiple_of(8)` is true, so the divisibility guard above waves it through —
            // and produced a census with `q_proj [0, 6656]`.
            ("num_attention_heads", self.num_attention_heads),
        ] {
            ensure!(v > 0, "{what} is zero");
        }
        // The eps is an f64 the kernels narrow to f32: `schema.rs`'s rule, and its failure —
        // "a 1e-46 eps passes an f64 positivity test and reaches every RMSNorm as 0.0f32".
        crate::schema::ensure_f32_positive(&[
            ("rms_norm_eps", self.rms_norm_eps),
            ("rope_theta", self.rope_theta),
        ])?;
        // One anchor token plus at least one candidate: the cycle slices index 0 off, so a
        // block of 1 drafts nothing and a block of 0 cannot carry its anchor.
        ensure!(
            self.block_size >= 2,
            "block_size {} leaves no candidate after the anchor row is sliced off",
            self.block_size
        );
        let ids = self.target_layer_ids();
        ensure!(
            !ids.is_empty() && ids.windows(2).all(|w| w[0] < w[1]),
            "target_layer_ids {ids:?} must be non-empty and strictly increasing — the \
             encoder concatenates one column block per entry, in order"
        );
        // The engine's bidirectional block-attend implements the sliding form only, which
        // is every layer of the shipped drafter. A full-attention layer type is a different
        // mask, refused here rather than silently attended with the wrong one.
        if let Some(kinds) = &self.layer_types {
            ensure!(
                kinds.len() == self.num_hidden_layers,
                "layer_types has {} entries for {} layers",
                kinds.len(),
                self.num_hidden_layers
            );
            ensure!(
                kinds.iter().all(|k| k == "sliding_attention"),
                "layer_types {kinds:?}: every shipped drafter layer is sliding_attention, \
                 and the engine's block-attend implements no other mask"
            );
        }
        Ok(())
    }

    /// **The tensor census: every name the checkpoint must carry, with its shape.** Derived
    /// from the config and nothing else, so the converter's completeness walk, its gate's
    /// fixture and the engine's pin are one statement.
    ///
    /// 11 per layer plus 3 globals — 58 for the real five-layer drafter. The spec's prose
    /// says "59 tensors" for the shipped checkpoint while its own per-layer enumeration
    /// derives 58; the converter compares this census against the real file by exact SET
    /// equality, so whichever tensor accounts for the difference (an `fc` bias is the
    /// candidate) is NAMED by the gate on first contact instead of silently copied or
    /// silently dropped.
    ///
    /// > **RESOLVED 2026-08-17 against the real file: it is 58, and there is no `fc` bias.** The
    /// > enumeration was right and the prose was wrong. The prediction is kept because the
    /// > *reason* it was made is the reusable part — a set-equality census names the
    /// > disagreement instead of choosing a side — and because it is the record of what this
    /// > schema knew before it had a checkpoint to read.
    pub fn census(&self) -> Vec<(String, Vec<usize>)> {
        let (h, hd) = (self.hidden_size, self.head_dim);
        let (nq, nkv, inter) = (
            self.num_attention_heads,
            self.num_key_value_heads,
            self.intermediate_size,
        );
        let shape = |t: &str| -> Vec<usize> {
            match t {
                "self_attn.q_proj" => vec![nq * hd, h],
                "self_attn.k_proj" | "self_attn.v_proj" => vec![nkv * hd, h],
                "self_attn.o_proj" => vec![h, nq * hd],
                "self_attn.q_norm" | "self_attn.k_norm" => vec![hd],
                "mlp.gate_proj" | "mlp.up_proj" => vec![inter, h],
                "mlp.down_proj" => vec![h, inter],
                "input_layernorm" | "post_attention_layernorm" => vec![h],
                // Reachable only by a name added to DRAFTER_LAYER_TENSORS and not here — which
                // moves `census().len()`, and that is an absolute in the gate next door. The
                // sibling `GlimmerTextConfig::layer_tensor_shape` bails instead; making this
                // one fallible ripples `?` through two callers to buy the same red.
                _ => vec![h],
            }
        };
        let mut out: Vec<(String, Vec<usize>)> = (0..self.num_hidden_layers)
            .flat_map(|l| {
                DRAFTER_LAYER_TENSORS
                    .iter()
                    .map(move |t| (format!("layers.{l}.{t}.weight"), shape(t)))
            })
            .collect();
        // The three model-level tensors. Not a constant: unlike `DRAFTER_LAYER_TENSORS`, which
        // `census` ITERATES, a three-name list would only be read for its length — and the
        // absent names matter more than the present ones. There is no `embed_tokens` and no
        // `lm_head`: the drafter borrows the target's, and the converter's set equality treats
        // their PRESENCE in a checkpoint as an error, not a bonus.
        out.push((
            "encoder.fc.weight".into(),
            vec![h, self.target_layer_ids().len() * h],
        ));
        out.push(("encoder.output_norm_enc.weight".into(), vec![h]));
        out.push(("norm.weight".into(), vec![h]));
        out
    }
}

#[cfg(test)]
mod tests {
    //! **The census against the SPEC, because the checkpoint is not on this machine.**
    //!
    //! > **CORRECTED 2026-08-17: the checkpoint IS on this machine now**, at
    //! > `/swarm/storage/ai/rivoli/muse-glimmer-30b-assistant` — the directory search recorded
    //! > below was right on 2026-08-16 and wrong the next day. The end-to-end gate this note
    //! > says "cannot exist yet" exists: `crates/cli/tests/drafter_convert.rs`, which gates the
    //! > REAL checkpoint's census against the checkpoint's own vendored safetensors header, and
    //! > `convert_glimmer_drafter` has run against the real file (174 s, exit 0). The tests
    //! > below keep their value unchanged and are not superseded: they pin the SCHEMA's
    //! > defaults, while that gate pins the vendored config — two independent literals about one
    //! > shipped model, and the pair is the cross-check. **The 58-vs-59 question these tests
    //! > were written unable to settle is settled: 58, and there is no `fc` bias.**
    //!
    //! `meta-models/Muse-Glimmer-30B-assistant` is absent from `/swarm/storage/ai/` (checked
    //! 2026-08-16: `glimmer-30b-bf16`, `glimmer-30b-fp8` and `muse-glimmer-30b` are all the
    //! TARGET). So the converter's end-to-end gate — read a real file, write an artifact, read
    //! it back — cannot exist yet, and these tests gate the half that can: that the shapes this
    //! schema derives are the shapes `old:docs/reference/glimmer-architecture.md` §11 states, and that the four
    //! refusals fire on the documents that reach them.
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: a firing panic IS the report
    use super::*;

    /// The nine integer widths as one value. Said once because two tests read the same nine in
    /// the same order, and `build.rs`'s jscpd gate correctly reported the second spelling of
    /// that list as a 54-token clone of the first. An ARRAY, not a tuple: rustfmt explodes a
    /// nine-`usize` tuple type one element per line, and jscpd then reports THAT against the
    /// nine field reads below — the gate was right both times.
    fn widths(c: &DrafterConfig) -> [usize; 9] {
        [
            c.hidden_size,
            c.intermediate_size,
            c.num_hidden_layers,
            c.num_attention_heads,
            c.num_key_value_heads,
            c.head_dim,
            c.sliding_window,
            c.block_size,
            c.mask_token_id,
        ]
    }

    /// The shipped config, minus every defaulted field — so the defaults ARE what is tested.
    fn shipped() -> DrafterConfig {
        DrafterConfig::parse(r#"{"model_type": "muse_glimmer_assistant"}"#).unwrap()
    }

    /// **§11's shapes, derived here and compared to the spec's own numbers.**
    ///
    /// `encoder.fc [6656, 33280]` is the load-bearing one: 33280 = 5 × 6656 is what proves the
    /// encoder concatenates one column block per `target_layer_ids` entry, and it is derived
    /// from `target_layer_ids().len()` rather than written, so a schema that lost an entry
    /// fails here instead of building the projection four-fifths as wide.
    #[test]
    fn the_census_derives_the_shapes_the_spec_states() {
        let c = shipped();
        let census = c.census();
        let get = |n: &str| {
            census
                .iter()
                .find(|(name, _)| name == n)
                .unwrap_or_else(|| panic!("no census entry {n}"))
                .1
                .clone()
        };
        assert_eq!(get("encoder.fc.weight"), vec![6656, 33280]);
        assert_eq!(get("encoder.output_norm_enc.weight"), vec![6656]);
        assert_eq!(get("norm.weight"), vec![6656]);
        // 32 Q heads x 128 = 4096, and 8 KV heads x 128 = 1024 — the 4:1 ratio that is NOT the
        // target's 16:1, expressed as the two projection widths a port would get wrong together.
        assert_eq!(get("layers.0.self_attn.q_proj.weight"), vec![4096, 6656]);
        assert_eq!(get("layers.0.self_attn.k_proj.weight"), vec![1024, 6656]);
        assert_eq!(get("layers.0.self_attn.v_proj.weight"), vec![1024, 6656]);
        assert_eq!(get("layers.0.self_attn.o_proj.weight"), vec![6656, 4096]);
        // WEIGHTED per-head QK-norms: tensors the target does not ship at all.
        assert_eq!(get("layers.4.self_attn.q_norm.weight"), vec![128]);
        assert_eq!(get("layers.4.self_attn.k_norm.weight"), vec![128]);
        assert_eq!(get("layers.4.mlp.gate_proj.weight"), vec![19968, 6656]);
        assert_eq!(get("layers.4.mlp.up_proj.weight"), vec![19968, 6656]);
        assert_eq!(get("layers.0.input_layernorm.weight"), vec![6656]);
        assert_eq!(get("layers.4.mlp.down_proj.weight"), vec![6656, 19968]);
        // TWO norms per layer, not the target's four: no post-FFN norm, no sandwich pair.
        assert_eq!(get("layers.4.post_attention_layernorm.weight"), vec![6656]);
        // **58, and the spec's prose says 59.** Stated as a number here so the difference is a
        // gate the converter's set-equality walk NAMES on first contact with a real checkpoint,
        // rather than a tensor silently copied or silently dropped. Whichever way it resolves,
        // this assert is the thing that has to move.
        // All eleven per-layer names are pinned above, so the `_ => vec![h]` arm in `census`
        // can only be reached by a TWELFTH name — and that moves this count.
        assert_eq!(census.len(), 58);
        // ...and the count alone does NOT stop the census SHRINKING: duplicating one entry of
        // DRAFTER_LAYER_TENSORS over another of the same shape keeps the length and the byte
        // total, and only loses a name. Found by review 2026-08-16.
        let names: std::collections::BTreeSet<&str> =
            census.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), census.len(), "a census name is duplicated");
        // No embedding and no lm_head: the drafter borrows the target's, and their ABSENCE is
        // asserted rather than assumed (the converter treats their presence as an error).
        assert!(
            !census
                .iter()
                .any(|(n, _)| n.contains("embed_tokens") || n.contains("lm_head")),
            "the drafter census must not contain a vocabulary tensor"
        );
    }

    /// **The census reconciles the PUBLISHED file size, which settles §11's own tensor count.**
    ///
    /// The spec states 5,111,976,608 B and 2.556 B params for `Muse-Glimmer-30B-assistant`, and
    /// says "59 tensors" while its own per-layer enumeration derives 58. Summing this census bf16
    /// gives 2,555,985,152 params = 5,111,970,304 B, leaving **6,304 B** — which is exactly a
    /// safetensors 8-byte length prefix plus a 6,296-byte JSON header, 108.6 B per entry over 58
    /// names like `layers.0.self_attn.q_proj.weight`. There is no room for a 59th tensor of any
    /// meaningful size, so **58 is the count and the spec's prose is a miscount** — derived here,
    /// 2026-08-16, not inherited. The converter still compares by exact SET equality, so if a real
    /// checkpoint ever disagrees it is NAMED rather than silently copied or dropped, and this
    /// assert is what has to move.
    ///
    /// The same arithmetic corrects the pin figure the M17 plan carries: 5,111,976,608 B is
    /// **5.112 GB = 4.761 GiB**, not "5.1 GiB".
    #[test]
    fn the_census_accounts_for_the_published_checkpoint_size() {
        let c = shipped();
        let params: usize = c
            .census()
            .iter()
            .map(|(_, shape)| shape.iter().product::<usize>())
            .sum();
        assert_eq!(params, 2_555_985_152, "census params (spec: 2.556 B)");
        let published = 5_111_976_608usize;
        let header = published - params * 2;
        assert_eq!(
            header, 6_304,
            "unaccounted bytes = safetensors prefix + JSON header"
        );
    }

    /// **Every field is read from JSON under the spelling written here.** With a default on all
    /// twelve and no `deny_unknown_fields`, a mis-spelled serde key is INDISTINGUISHABLE from an
    /// omitted one — the struct silently takes the default, and for the shipped drafter every
    /// default coincides with the real value, so the whole schema would be right by luck and
    /// wrong for any other checkpoint. `head_dim` is the sharpest case: 6656/32 = 208, not 128,
    /// so a misread there cannot be recovered from the other widths. This supplies all twelve at
    /// NON-default values and requires each to arrive. Found by review 2026-08-16.
    #[test]
    fn every_field_is_actually_read_from_its_json_key() {
        let c = DrafterConfig::parse(
            r#"{"model_type": "muse_glimmer_assistant", "hidden_size": 64,
                "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "rms_norm_eps": 2e-5,
                "sliding_window": 32, "block_size": 6, "mask_token_id": 41,
                "rope_theta": 10000.0, "target_layer_ids": [0, 1],
                "layer_types": ["sliding_attention", "sliding_attention"]}"#,
        )
        .unwrap();
        assert_eq!(widths(&c), [64, 96, 2, 4, 2, 8, 32, 6, 41]);
        assert!((c.rms_norm_eps - 2e-5).abs() < 1e-12);
        assert!((c.rope_theta - 10000.0).abs() < f64::EPSILON);
        assert_eq!(c.target_layer_ids(), vec![0, 1]);
        assert_eq!(c.layer_types.as_deref().map(<[String]>::len), Some(2));
        // The shapes follow the fields, so a key read into the wrong field shows up here.
        assert_eq!(
            c.census()
                .iter()
                .find(|(n, _)| n == "layers.1.self_attn.q_proj.weight")
                .unwrap()
                .1,
            vec![32, 64]
        );
        assert_eq!(c.census().len(), 2 * 11 + 3);
    }

    /// The defaults are the reference's, so a config that omits a field parses as the reference
    /// would BUILD it — the §11 block that a port otherwise transcribes by hand.
    #[test]
    fn the_defaults_are_the_shipped_drafters() {
        let c = shipped();
        assert_eq!(widths(&c), [6656, 19968, 5, 32, 8, 128, 2048, 16, 201_818]);
        assert!((c.rms_norm_eps - 1e-5).abs() < f64::EPSILON);
        assert_eq!(c.target_layer_ids(), vec![1, 13, 25, 37, 49]);
        assert!((c.rope_theta - 500_000.0).abs() < f64::EPSILON);
    }

    /// Every refusal, each on the document that reaches it and on nothing else. A refusal test
    /// that only asserts non-zero exit passes when the guard is deleted and something else
    /// fails, so each names a fragment of its own message.
    #[test]
    fn each_refusal_fires_on_its_own_document() {
        let with = |extra: &str| {
            DrafterConfig::parse(&format!(
                r#"{{"model_type": "muse_glimmer_assistant"{extra}}}"#
            ))
        };
        assert!(with("").is_ok(), "the shipped config must parse");
        for (extra, want) in [
            (
                r#", "num_key_value_heads": 7"#,
                "do not divide into KV heads",
            ),
            (r#", "hidden_size": 0"#, "hidden_size is zero"),
            (r#", "block_size": 1"#, "leaves no candidate"),
            (r#", "target_layer_ids": [13, 1]"#, "strictly increasing"),
            (r#", "target_layer_ids": []"#, "strictly increasing"),
            (
                r#", "layer_types": ["full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"]"#,
                "every shipped drafter layer is sliding_attention",
            ),
            (r#", "layer_types": ["sliding_attention"]"#, "for 5 layers"),
        ] {
            let e = with(extra)
                .err()
                .unwrap_or_else(|| panic!("{extra} must be refused"));
            let msg = format!("{e:#}");
            assert!(
                msg.contains(want),
                "{extra}: refused with {msg:?}, not {want:?}"
            );
        }
        // The wrong DOCUMENT, reported as itself rather than as a missing width — the whole
        // reason the model_type check runs before serde does.
        let e = DrafterConfig::parse(r#"{"model_type": "muse_glimmer_text"}"#).unwrap_err();
        assert!(format!("{e:#}").contains("not a DFlash drafter checkpoint"));
    }
}
