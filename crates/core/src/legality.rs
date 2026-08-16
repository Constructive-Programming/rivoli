//! The (architecture × flag) legality table — **one** decider for "may this run ask for
//! that", and the refusal text lives in the table beside the verdict.
//!
//! The old tree answered this question in two places: `arch.rs` carried presentation
//! policy (which flags `--help` shows per architecture) while `main.rs` carried nine
//! hand-written refusals. Two authorities that can independently judge one configuration
//! is the silent-wrong hazard — a flag hidden from help but still accepted, or accepted
//! and then dropped, and a recorded command line that carries a knob nothing spent. Here
//! [`decide`] is the only judge, [`Outcome`] is its only vocabulary, and a refusal's
//! wording is data rather than prose scattered through the CLI.
//!
//! **Why the identity enums live in `rivoli-core` and not in `rivoli-artifact`.** The DAG
//! runs core ← artifact ← engine ← cli, so core cannot name an artifact type. The table
//! has to key on the architecture, therefore the architecture *identity* is core's and
//! the *recognition* — which manifest spellings name it — stays with the manifest reader
//! (`rivoli_artifact::arch::from_manifest_str`, which re-exports [`Arch`]). That is the
//! shape `ModelConfig::scoring` already uses: core owns the vocabulary, artifact maps its
//! raw input into it.
//!
//! **[`Mode`] is not a weight format, deliberately.** `RoutedFmt` lives in
//! `rivoli-artifact` precisely so core cannot express "residency selects arithmetic" (the
//! old tree's hybrid defect, where `--max-mem` changed the output text). `Mode` is what
//! the user typed; `RoutedFmt` is what the pool stores; the CLI is the one place where one
//! becomes the other, and `Mode::Hybrid` has no `RoutedFmt` at all — which is exactly why
//! it refuses here rather than silently resolving to one of the two real formats.

/// The architectures the engine has, or will have, a decode path for. Parsed from the
/// artifact manifest's `architectures` / `model_type` by
/// `rivoli_artifact::arch::from_manifest_str`; an unrecognised value must REFUSE at
/// startup rather than fall back to a default. Falling back to GLM is the specific
/// mistake worth naming: it is the only value that would look like it worked.
///
/// There is deliberately **no `--model`/`--arch` flag**: the artifact IS the model, and a
/// flag naming the architecture is a flag that can disagree with the weights it describes.
/// Disagreeing is not a crash — it launches the wrong decode path and produces fluent
/// wrong text.
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

/// How many architectures [`Arch::ALL`] must carry. Hand-written on purpose: it is what
/// turns "someone added a variant and forgot the list" from a silent hole into a length
/// mismatch on `ALL`'s array type. See [`Arch::ordinal`].
const ARCH_COUNT: usize = 4;

impl Arch {
    /// Every architecture, exactly once — the legality product test's first axis.
    pub const ALL: [Arch; ARCH_COUNT] = [
        Arch::GlmMoeDsa,
        Arch::DeepseekV4,
        Arch::KimiK3,
        Arch::MuseGlimmer,
    ];

    /// Dense index, written out arm by arm rather than as `self as usize`, because the
    /// cast would keep compiling when a variant is added and the exhaustive match will
    /// not. Together with `[Arch; ARCH_COUNT]` and the permutation check in this module's
    /// tests, the only self-consistent way to add an architecture is: new arm, bumped
    /// `ARCH_COUNT`, new entry in [`Arch::ALL`].
    ///
    /// `cfg(test)` because the completeness check is its only consumer and a `pub fn`
    /// with no caller is surface this workspace deletes on sight. The gate still binds:
    /// the exhaustive match is compiled by `cargo test`, which is the build that runs it.
    #[cfg(test)]
    const fn ordinal(self) -> usize {
        match self {
            Arch::GlmMoeDsa => 0,
            Arch::DeepseekV4 => 1,
            Arch::KimiK3 => 2,
            Arch::MuseGlimmer => 3,
        }
    }

    /// Short kebab name, for help headers, log lines and refusals.
    pub fn name(self) -> &'static str {
        match self {
            Arch::GlmMoeDsa => "glm-moe-dsa",
            Arch::DeepseekV4 => "deepseek-v4",
            Arch::KimiK3 => "kimi-k3",
            Arch::MuseGlimmer => "muse-glimmer",
        }
    }

    /// One line naming the attention family, since that is what actually differs between
    /// the four — and what a "no decode path for this artifact" refusal wants to quote.
    pub fn summary(self) -> &'static str {
        match self {
            Arch::GlmMoeDsa => "MLA + q-LoRA, DSA lightning indexer",
            Arch::DeepseekV4 => "shared-K=V MQA, sliding window + per-layer KV compression",
            Arch::KimiK3 => "69 KDA + 24 gated MLA (NoPE), latent-space routed experts",
            Arch::MuseGlimmer => "dense GQA 32Q/2KV, gated, 3 sliding (2048) per full layer",
        }
    }
}

/// What the user asked the routed experts to be stored as. See this module's header for
/// why this is NOT `RoutedFmt`: `Hybrid` names a run with more than one format in it and
/// therefore has no single format to map onto.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// 3-bit vector quantization — the smallest, and the M6 default.
    Int3Vq,
    /// 4-bit scalar quantization — the best measured quality, and the slowest.
    Int4,
    /// int4 for the hot experts, vq3 for the cold ones.
    Hybrid,
}

/// How attention selects the rows it attends over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnKind {
    /// Every position attends every earlier position.
    Dense,
    /// A leading sink plus a trailing window.
    Streaming,
    /// The trained-in DSA lightning indexer picks `index_topk` rows.
    Dsa,
    /// The MISA paper's multi-head indexer selection.
    Misa,
}

/// `--mode`'s accepted spellings, in variant order. ONE table per vocabulary: the parser,
/// the `--dump-ids` header and any log line all read it, so the spelling a user types and
/// the spelling a dump file records cannot drift apart.
pub const MODES: [(&str, Mode); 3] = [
    ("int3-vq", Mode::Int3Vq),
    ("int4", Mode::Int4),
    ("hybrid", Mode::Hybrid),
];

/// `--attn`'s accepted spellings, in variant order. There is deliberately no `auto`: the
/// old tree's `auto` resolved to `dsa` when the artifact carried indexer weights, and
/// with only the dense path built it could resolve to exactly one thing — a value whose
/// whole purpose is to choose, that cannot choose, is a value that lies.
pub const ATTNS: [(&str, AttnKind); 4] = [
    ("dense", AttnKind::Dense),
    ("streaming", AttnKind::Streaming),
    ("dsa", AttnKind::Dsa),
    ("misa", AttnKind::Misa),
];

/// Resolve one spelling against a vocabulary table, or say what was accepted.
///
/// Generic over the vocabulary rather than written once per enum: two copies of
/// `match s { "..." => Ok(..) }` share a ~20-token prefix and are exactly the duplication
/// `build.rs`'s jscpd gate forbids. The error is a `String` because that is what clap's
/// `value_parser` takes — which costs `rivoli-core` no dependency on clap.
pub fn parse_in<T: Copy>(table: &[(&'static str, T)], flag: &str, s: &str) -> Result<T, String> {
    table
        .iter()
        .find(|(name, _)| *name == s)
        .map(|(_, v)| *v)
        .ok_or_else(|| {
            let accepted: Vec<&str> = table.iter().map(|(name, _)| *name).collect();
            format!(
                "unknown {flag} value {s:?} (accepted: {})",
                accepted.join(" | ")
            )
        })
}

/// The spelling of a value, from the same table its parser reads.
///
/// The `"?"` fallback is unreachable: `vocabularies_round_trip` asserts every variant
/// reachable through [`Flag::ALL`] is in its table. It exists because a total function is
/// cheaper than a `Result` no caller could act on.
pub fn name_in<T: Copy + PartialEq>(table: &[(&'static str, T)], v: T) -> &'static str {
    table
        .iter()
        .find(|(_, t)| *t == v)
        .map_or("?", |(name, _)| *name)
}

/// One thing a run can ask for, **whose answer can differ by architecture**. Closed on
/// purpose: the domain is exactly the flags whose legality is an architectural question, so
/// there is no second, informal list of "flags nobody checks".
///
/// Two kinds of flag are deliberately absent, and the distinction is the table's scope:
/// - `--bench` and `--port`, plus the positional artifact path, are the INVOCATION — which
///   of the two things this process is — not knobs on it. Their exclusivity is clap's.
/// - `--think` is a knob, but an architecture-independent one: it selects a framing the
///   tokenizer either renders or does not, and no architecture here answers it differently.
///   A row that reads `Support` for every arch is a row that can only ever be noise, and
///   the cost of adding it is real — [`Flag::ALL`], the ordinal permutation and the product
///   test all widen. **The rule: a flag earns a row when some architecture would answer it
///   differently, and not before.** A flag that later grows per-arch variance gets its row
///   then, which is also when there is something true to write in the cell.
///
/// Note the asymmetry among the rows that do exist: `Mode` and `Attn` carry their value
/// because the answer differs per value (int4 decodes, hybrid does not), while
/// `CachePolicy` does not because all three of its values stand or fall together on any
/// given architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flag {
    Mode(Mode),
    Attn(AttnKind),
    CachePolicy,
    MaxMem,
    Ctx,
    Prompt,
    Trace,
    Mtp,
    DumpIds,
}

/// Where the valueless flags' ordinals start.
const VALUELESS: usize = MODES.len() + ATTNS.len();

/// How many (arch, flag) columns exist. Same contract as [`ARCH_COUNT`]: the `7` is the
/// count of valueless [`Flag`] variants and is hand-written so that adding one cannot
/// quietly skip [`Flag::ALL`].
const FLAG_COUNT: usize = VALUELESS + 7;

impl Flag {
    /// Every flag value the CLI can present, exactly once — the product test's second
    /// axis, and the thing that makes "the test covers the whole product" a checkable
    /// claim rather than a comment.
    pub const ALL: [Flag; FLAG_COUNT] = [
        Flag::Mode(Mode::Int3Vq),
        Flag::Mode(Mode::Int4),
        Flag::Mode(Mode::Hybrid),
        Flag::Attn(AttnKind::Dense),
        Flag::Attn(AttnKind::Streaming),
        Flag::Attn(AttnKind::Dsa),
        Flag::Attn(AttnKind::Misa),
        Flag::CachePolicy,
        Flag::MaxMem,
        Flag::Ctx,
        Flag::Prompt,
        Flag::Trace,
        Flag::Mtp,
        Flag::DumpIds,
    ];

    /// Dense index over the whole flag domain — see [`Arch::ordinal`] for why this is an
    /// exhaustive match and not a cast. The value-carrying arms index their own
    /// vocabulary, so a `Mode` variant that is missing from [`MODES`] lands on
    /// `usize::MAX` and fails the permutation check, while one that is present shifts
    /// [`VALUELESS`] and makes [`Flag::ALL`]'s array length wrong — a compile error.
    ///
    /// `cfg(test)` for the same reason as [`Arch::ordinal`].
    #[cfg(test)]
    fn ordinal(self) -> usize {
        match self {
            Flag::Mode(m) => MODES
                .iter()
                .position(|(_, v)| *v == m)
                .unwrap_or(usize::MAX),
            Flag::Attn(a) => {
                MODES.len()
                    + ATTNS
                        .iter()
                        .position(|(_, v)| *v == a)
                        .unwrap_or(usize::MAX)
            }
            Flag::CachePolicy => VALUELESS,
            Flag::MaxMem => VALUELESS + 1,
            Flag::Ctx => VALUELESS + 2,
            Flag::Prompt => VALUELESS + 3,
            Flag::Trace => VALUELESS + 4,
            Flag::Mtp => VALUELESS + 5,
            Flag::DumpIds => VALUELESS + 6,
        }
    }

    /// How the user spelled it, so a refusal can quote the command line back instead of
    /// naming an enum variant the user has never seen.
    pub fn spelling(self) -> String {
        match self {
            Flag::Mode(m) => format!("--mode {}", name_in(&MODES, m)),
            Flag::Attn(a) => format!("--attn {}", name_in(&ATTNS, a)),
            Flag::CachePolicy => "--cache-policy".to_string(),
            Flag::MaxMem => "--max-mem".to_string(),
            Flag::Ctx => "--ctx".to_string(),
            Flag::Prompt => "--prompt".to_string(),
            Flag::Trace => "--trace".to_string(),
            Flag::Mtp => "--mtp".to_string(),
            Flag::DumpIds => "--dump-ids".to_string(),
        }
    }
}

/// The verdict on one (architecture, flag) cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The run may proceed and the flag does what it says.
    Support,
    /// The run proceeds, but not as asked. The message names what it fell back to AND
    /// why, and the caller must SAY it — a silent degrade is how a benchmark ends up
    /// measuring something other than its own command line.
    FallbackLoudly(&'static str),
    /// The run does not start. The message names the deferral, so the user learns "not
    /// yet, and here is what it is waiting on" rather than "invalid value".
    Refuse(&'static str),
}

// --- The table. One function per architecture, dispatched once; every message is a
// --- named const so the verdicts read as rows rather than as prose.

const NO_ENGINE_ARM: &str = "this architecture has no decode path in this build — GLM-5.2 \
     is the only arm so far. Every flag refuses here rather than a chosen few, because a \
     knob cannot be legal on a model that cannot decode at all";

const HYBRID_IS_A_PLAN: &str = "int4 for hot experts and vq3 for cold ones returns as an \
     explicit FormatPlan, not as a pool property. In the old tree the cache picked each \
     expert's FORMAT, so --max-mem and --cache-policy changed the output TEXT; this tree \
     makes that unwriteable (rivoli-core cannot name a weight format) and the mixed-format \
     run comes back once the plan is the thing being chosen. Use int3-vq or int4";

const SPARSE_ATTN_DEFERRED: &str = "only --attn dense decodes today. Sparse row selection \
     (streaming sinks+window, the trained-in DSA indexer, MISA) is the first post-dense \
     increment — the indexer weights are not even placed in the resident pin yet";

const MTP_DEFERRED: &str = "speculative decode is deferred past parity: the draft head is \
     not loaded and the verify pass is not built. The 2-row batch shape it rides is \
     already designed in (MAXROW), so this is a decode-loop increment, not a re-architecture";

const TRACE_IS_TOKEN_MAJOR: &str = "--trace forces the prefill back to TOKEN-MAJOR. A v2 \
     trace carries no token delimiter and recovers one from the layer id descending, which \
     a layer-major prefill never does — a capture under it is silently mis-segmented, the \
     worst outcome for a file that costs a sole-tenant GPU tens of minutes. So this run's \
     prefill wall and reads/token are NOT comparable to an untraced arm (the reference \
     engine measured layer-major prefill at 2.15x token-major, and 28.20 reads/token \
     against 159.56; both are that engine's numbers, not this one's)";

/// **The** legality decider. Total over `Arch × Flag` by the compiler: adding an
/// architecture breaks this match, adding a flag breaks [`glm`], and either way the build
/// stops until the new cell has an argued answer.
pub fn decide(arch: Arch, flag: Flag) -> Outcome {
    match arch {
        Arch::GlmMoeDsa => glm(flag),
        // Not a wildcard over the flag axis: the reason is architecture-level, so the
        // flag is genuinely not consulted. Splitting the dispatch this way is what keeps
        // a `_` out of the table entirely.
        Arch::DeepseekV4 | Arch::KimiK3 | Arch::MuseGlimmer => Outcome::Refuse(NO_ENGINE_ARM),
    }
}

/// GLM-5.2's row of the table.
fn glm(flag: Flag) -> Outcome {
    match flag {
        Flag::Mode(Mode::Int3Vq | Mode::Int4) => Outcome::Support,
        Flag::Mode(Mode::Hybrid) => Outcome::Refuse(HYBRID_IS_A_PLAN),
        Flag::Attn(AttnKind::Dense) => Outcome::Support,
        Flag::Attn(AttnKind::Streaming | AttnKind::Dsa | AttnKind::Misa) => {
            Outcome::Refuse(SPARSE_ATTN_DEFERRED)
        }
        Flag::Mtp => Outcome::Refuse(MTP_DEFERRED),
        // The engine already DOES this fallback (`GlmEngine::new` reads
        // `RoutedPool::tracing()`); the table is what makes it something the user is
        // TOLD rather than something buried in an info line.
        Flag::Trace => Outcome::FallbackLoudly(TRACE_IS_TOKEN_MAJOR),
        Flag::CachePolicy | Flag::MaxMem | Flag::Ctx | Flag::Prompt | Flag::DumpIds => {
            Outcome::Support
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ordinals` must be exactly `0..n`, each once. This is the guard that makes an
    /// `ALL` list's completeness checkable: a forgotten variant either collides with an
    /// existing ordinal or leaves a gap, and both read as false here.
    fn is_dense_permutation(n: usize, ordinals: impl Iterator<Item = usize>) -> bool {
        let mut seen = vec![false; n];
        let mut count = 0;
        for o in ordinals {
            match seen.get_mut(o) {
                Some(slot) if !*slot => *slot = true,
                _ => return false,
            }
            count += 1;
        }
        count == n
    }

    /// The product test's domain is only as honest as the two lists it iterates, so the
    /// lists are checked first. Dropping an entry is already a compile error (the arrays
    /// are fixed-size); what this catches is the shape that still compiles — an entry
    /// duplicated or mistyped, which leaves the product covering less than it claims.
    #[test]
    fn the_axis_lists_are_complete() {
        assert!(
            is_dense_permutation(ARCH_COUNT, Arch::ALL.iter().map(|a| a.ordinal())),
            "Arch::ALL is not every architecture exactly once"
        );
        assert!(
            is_dense_permutation(FLAG_COUNT, Flag::ALL.iter().map(|f| f.ordinal())),
            "Flag::ALL is not every flag exactly once"
        );
    }

    /// The FULL (arch × flag) product, every cell decided and every non-Support cell
    /// carrying a message that names its deferral. An empty or stub message is the
    /// failure mode this table exists to prevent — "refused" without a reason is the
    /// unhelpful half of a refusal.
    #[test]
    fn every_arch_flag_cell_is_decided_with_its_reason() {
        let mut cells = 0;
        for arch in Arch::ALL {
            for flag in Flag::ALL {
                let msg = match decide(arch, flag) {
                    Outcome::Support => "",
                    Outcome::FallbackLoudly(m) | Outcome::Refuse(m) => m,
                };
                assert!(
                    msg.is_empty() || msg.len() > 40,
                    "{} on {}: a one-word reason is not a reason ({msg:?})",
                    flag.spelling(),
                    arch.name()
                );
                cells += 1;
            }
        }
        assert_eq!(cells, ARCH_COUNT * FLAG_COUNT, "product iterated short");
    }

    /// Anti-vacuity. A table that answered `Support` everywhere would pass every check
    /// above while deciding nothing, and one that answered `Refuse` everywhere would
    /// pass them while refusing the working engine. All three verdicts must have a live
    /// cell — including `FallbackLoudly`, which is a real row (`--trace`) rather than a
    /// variant kept warm for later; if that row ever goes, so should the variant.
    #[test]
    fn all_three_verdicts_have_a_live_cell() {
        let all: Vec<Outcome> = Arch::ALL
            .iter()
            .flat_map(|&a| Flag::ALL.iter().map(move |&f| decide(a, f)))
            .collect();
        assert!(all.contains(&Outcome::Support), "nothing is supported");
        assert!(
            all.iter().any(|o| matches!(o, Outcome::Refuse(_))),
            "nothing is refused"
        );
        assert!(
            all.iter().any(|o| matches!(o, Outcome::FallbackLoudly(_))),
            "nothing falls back — delete the variant or restore the row"
        );
    }

    /// GLM's row, cell by cell, as of M6.
    ///
    /// A deliberate change-detector: the checks above prove the table is TOTAL and not
    /// VACUOUS, and neither of them would notice a cell being flipped — which is the one
    /// edit that changes what the engine accepts. Pinning the row makes flipping one a
    /// decision someone has to record here, next to the argument for it, rather than a
    /// diff nothing reads. Update it in the same commit that lands the arm.
    #[test]
    fn glm_row_is_the_m6_truth() {
        let glm = |f| decide(Arch::GlmMoeDsa, f);
        let refused = |f| matches!(glm(f), Outcome::Refuse(_));
        // The two single-format modes decode; hybrid returns as a FormatPlan.
        assert_eq!(glm(Flag::Mode(Mode::Int3Vq)), Outcome::Support);
        assert_eq!(glm(Flag::Mode(Mode::Int4)), Outcome::Support);
        assert!(refused(Flag::Mode(Mode::Hybrid)), "hybrid must refuse");
        // Dense attention only; the three sparse selections are the post-dense increment.
        assert_eq!(glm(Flag::Attn(AttnKind::Dense)), Outcome::Support);
        for kind in [AttnKind::Streaming, AttnKind::Dsa, AttnKind::Misa] {
            assert!(refused(Flag::Attn(kind)), "{kind:?} must refuse");
        }
        assert!(refused(Flag::Mtp), "speculative decode must refuse");
        // --trace is the one degrade-and-say-so cell.
        assert!(matches!(glm(Flag::Trace), Outcome::FallbackLoudly(_)));
        for f in [
            Flag::CachePolicy,
            Flag::MaxMem,
            Flag::Ctx,
            Flag::Prompt,
            Flag::DumpIds,
        ] {
            assert_eq!(glm(f), Outcome::Support, "{} must decode", f.spelling());
        }
    }

    /// Every spelling parses back to the variant it names, which is what makes
    /// [`name_in`]'s `"?"` fallback unreachable and lets `--dump-ids` headers be compared
    /// against command lines.
    #[test]
    fn vocabularies_round_trip() {
        for (name, m) in MODES {
            assert_eq!(parse_in(&MODES, "--mode", name), Ok(m));
            assert_eq!(name_in(&MODES, m), name);
        }
        for (name, a) in ATTNS {
            assert_eq!(parse_in(&ATTNS, "--attn", name), Ok(a));
            assert_eq!(name_in(&ATTNS, a), name);
        }
        assert!(parse_in(&MODES, "--mode", "int3").is_err());
    }
}
