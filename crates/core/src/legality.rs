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
/// - `--prompt` and `--dump-ids` had rows at first and lost them to the same rule
///   (review 2026-08-16): a prompt and an id dump are IO facts of the invocation, and no
///   architecture that decodes at all could answer either differently. `--ctx` stays: the
///   KV ceiling is answered per-architecture in principle — K3's recurrent KDA state has
///   no per-token KV growth at all, so its cell would not read like GLM's.
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
    Trace,
    Mtp,
}

/// Where the valueless flags' ordinals start.
const VALUELESS: usize = MODES.len() + ATTNS.len();

/// How many (arch, flag) columns exist. Same contract as [`ARCH_COUNT`]: the `7` is the
/// count of valueless [`Flag`] variants and is hand-written so that adding one cannot
/// quietly skip [`Flag::ALL`].
const FLAG_COUNT: usize = VALUELESS + 5;

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
        Flag::Trace,
        Flag::Mtp,
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
            Flag::Trace => VALUELESS + 3,
            Flag::Mtp => VALUELESS + 4,
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
            Flag::Trace => "--trace".to_string(),
            Flag::Mtp => "--mtp".to_string(),
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

const DENSE_HAS_NO_ROUTED_FORMAT: &str = "--mode names how ROUTED EXPERTS are stored, and \
     this architecture is DENSE — it has no experts and no routing, and its weights are \
     bf16 throughout. The flag is ignored, and nothing in the run is stored in the format \
     you named. Residency here is a partition of whole LAYERS (--max-mem decides how many \
     are pinned), never a choice of arithmetic";

const DENSE_CACHE_POLICY_IS_ONE_ANSWER: &str = "a dense model reads its layers in fixed \
     cyclic order, which is LRU's pathological case: at any deficit LRU evicts exactly the \
     layer needed next and the hit rate is 0, not pinned/n_layers. Belady on a cyclic scan \
     degenerates to holding a fixed subset, and every fixed subset of size k has the same \
     hit rate k/n — so all three policies are the same answer here and the pin is a static \
     prefix. Use --max-mem, which is the knob that actually moves it";

const DENSE_HAS_NO_EXPERT_TRACE: &str = "--trace captures the ROUTED-EXPERT access stream \
     for the offline residency sim, and a dense model makes no routing decisions to record \
     — every layer is read every token, so the trace is derivable from the layer count and \
     carries no information. Accepting it would write a file whose emptiness reads as a \
     measurement";

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

const V4_FORMAT_IS_THE_CHECKPOINTS: &str = "--mode names how ROUTED EXPERTS are stored, and \
     on this architecture the checkpoint decides: the experts are `.f4` (e2m1 nibbles, one \
     e8m0 scale per 32 weights) and there is no second container to choose between. The flag \
     is ignored, and nothing in the run is stored in the format you named. What DOES move \
     residency here is --max-mem, which decides how many of the 137 GiB of experts are pinned";

const V4_ATTENTION_IS_WINDOW_PLUS_BLOCKS: &str = "--attn names how attention SELECTS the rows \
     it reads, and this architecture does not offer the choice: every layer attends a \
     sliding_window ring PLUS the pooled blocks its per-layer KV compressor emitted, which is \
     neither the dense causal prefix nor any of the three sparse selections. `dense` is \
     ignored rather than honoured — there is no dense path here to fall back to";

const V4_SPARSE_ATTN_IS_NOT_A_CHOICE: &str = "--attn names how attention SELECTS the rows \
     it reads, and this architecture does not offer the choice: streaming sinks, the DSA \
     token indexer and MISA are all row selections over a DENSE cache, and this cache is a \
     sliding-window ring plus pooled blocks. Its own trained-in indexer ranks COMPRESSED \
     BLOCKS — a different object from GLM's — and since the scored selection landed it \
     decides the set natively, on every indexed layer, wherever the causally-legal set \
     outgrows index_topk. Below that the set already fits and both rules name it, so there \
     is no context ceiling and nothing here to pick. The flag toggles nothing; the run \
     proceeds with the checkpoint's own selection";

const V4_MTP_NEEDS_A_KERNEL: &str = "speculative decode on this architecture is blocked by a \
     missing KERNEL, not a missing head, and that is which of the two would have to arrive \
     first. moe.hip instantiates the FP4 expert range at R=1 only and its guard refuses \
     anything else, so a V4 decode is structurally single-row: there is no verify pass to \
     batch whatever a draft head might later offer, and no two-row FP4 kernel could be scored \
     against an oracle that is bsz=1 only";

const V4_TRACE_PREFILL_IS_ONE_PASS: &str = "--trace still captures the routed-expert access \
     stream, but this architecture's prefill is ONE whole-prompt pass — attention is its only \
     cross-token operation, so the ring seeding and the block pooling both take every row at \
     once. A v2 trace recovers its token delimiter from the layer id descending, so the whole \
     prefill lands as a single pseudo-token however long the prompt was. The DECODE half is \
     faithful; prefill reads/token derived from this file are not comparable with GLM's";

const K3_FORMAT_IS_THE_CHECKPOINTS: &str = "--mode names how ROUTED EXPERTS are stored, and \
     on this architecture the checkpoint decides: Kimi-K3 SHIPS its 896 experts per layer as \
     MXFP4 (e2m1 nibbles, one e8m0 scale per 32 weights), repacked byte-for-byte into `.f4` \
     with nothing quantized. The flag is ignored, and nothing in the run is stored in the \
     format you named. What moves residency here is --max-mem, which decides how much of the \
     ~1.3 TiB routed set is pinned";

const K3_SPARSE_ATTN_HAS_NO_SUBSTRATE: &str = "streaming sinks, the DSA lightning indexer \
     and MISA are all row selections over a dense KV cache, and 69 of this architecture's 93 \
     layers keep NO rows at all — the KDA recurrence folds every position into a fixed \
     [heads][128][128] state, so there is nothing for a selection to select from. The 24 \
     gated MLA layers do keep rows, but this checkpoint ships no indexer to rank them, so \
     `dsa` names weights that do not exist and the other two name an increment nobody has \
     measured on a quarter of a model";

const K3_MTP_NEEDS_TWO_KERNELS: &str = "speculative decode on this architecture is blocked \
     by TWO missing kernels, which is more than either other refusal: the MXFP4 situ expert \
     range is instantiated at one row only (its guard refuses nrow != 1), and a verify pass \
     would also need a MULTI-TOKEN KDA recurrence — the chunked kernel whose port must \
     reinstate the UT-transform inverse and the A_qk-diagonal retention together with gating \
     fixtures. There is no draft head loaded either, but the kernels are what would have to \
     arrive first";

/// **The** legality decider. Total over `Arch × Flag` by the compiler: adding an
/// architecture breaks this match, adding a flag breaks [`glm`], and either way the build
/// stops until the new cell has an argued answer.
pub fn decide(arch: Arch, flag: Flag) -> Outcome {
    match arch {
        Arch::GlmMoeDsa => glm(flag),
        Arch::MuseGlimmer => muse_glimmer(flag),
        Arch::DeepseekV4 => deepseek_v4(flag),
        Arch::KimiK3 => kimi_k3(flag),
    }
}

/// Kimi-K3's row of the table — M9's arm, and most of it is two facts worked out flag by
/// flag: **the checkpoint chose the expert format**, and **69 of 93 layers are a recurrence,
/// not an attention with rows to select.**
///
/// # `--mode` falls back loudly, forced the same way V4's was
///
/// `Support` would record a knob nothing spent and `Refuse` would kill the no-flags
/// invocation — [`muse_glimmer`]'s doc carries the argument in full.
///
/// # `--attn dense` is `Support`, on Muse Glimmer's precedent and not V4's
///
/// Glimmer's cell is `Support` although 39 of its 52 layers slide, because `dense` names
/// what its full layers genuinely do. K3's 24 MLA layers attend the whole causal prefix the
/// same way — NoPE, unconditionally causal — and the KDA layers are not an attention row
/// selection at all, so the flag contradicts nothing. V4's `FallbackLoudly` is the different
/// case: there EVERY layer attends a window plus pooled blocks, so `dense` described no
/// layer in the run.
///
/// # The rest
///
/// `--cache-policy` and `--max-mem` govern every token here even harder than on V4: the
/// routed set is ~1.3 TiB against ~115 GiB of budget, so residency is never the tail of a
/// working set. `--ctx` sizes the 24 MLA caches ONLY — the KDA state is context-free, which
/// is this row's contribution to the header's "a flag earns a row when some architecture
/// answers it differently". `--trace` is the one cell BETTER here than on GLM: this arm's
/// prefill is already token-sequential (the recurrence forces it), so the v2 trace's
/// token-major recovery is exact and nothing falls back. `--mtp` refuses with K3's OWN two
/// blockers, because quoting GLM's or V4's wording would tell a user to wait for the wrong
/// thing.
fn kimi_k3(flag: Flag) -> Outcome {
    match flag {
        // Every value together: the checkpoint's format is the same whichever was typed.
        Flag::Mode(Mode::Int3Vq | Mode::Int4 | Mode::Hybrid) => {
            Outcome::FallbackLoudly(K3_FORMAT_IS_THE_CHECKPOINTS)
        }
        Flag::Attn(AttnKind::Dense) => Outcome::Support,
        Flag::Attn(AttnKind::Streaming | AttnKind::Dsa | AttnKind::Misa) => {
            Outcome::Refuse(K3_SPARSE_ATTN_HAS_NO_SUBSTRATE)
        }
        Flag::CachePolicy | Flag::MaxMem | Flag::Ctx => Outcome::Support,
        // No fallback and no refusal: a token-sequential prefill IS token-major, so the
        // capture is faithful and costs the run nothing to be honest about.
        Flag::Trace => Outcome::Support,
        Flag::Mtp => Outcome::Refuse(K3_MTP_NEEDS_TWO_KERNELS),
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
        Flag::CachePolicy | Flag::MaxMem | Flag::Ctx => Outcome::Support,
    }
}

/// Muse Glimmer-30B's row of the table — the first DENSE architecture here, and most of
/// this row is that one fact worked out flag by flag.
///
/// # `--mode` falls back loudly rather than refusing, and the choice is forced
///
/// A dense model has no routed format, so `Support` is out: it would accept `--mode int4`
/// on a run that stores nothing in int4, which is precisely the "recorded command line
/// carrying a knob nothing spent" this table exists to prevent.
///
/// `Refuse` is out too, and for a reason that lives in the CALLER rather than here.
/// `--mode` carries a clap default, and `main::requested_flags` submits the value-carrying
/// flags on their RESOLVED value by design — so that a default which ever became illegal is
/// caught rather than exempted by never having been typed. A refusal here would therefore
/// kill `rivoli GLIMMER_DIR --bench 4`, an invocation with no `--mode` in it at all, on a
/// flag the user never typed. Making it presence-judged instead would mean two authorities
/// deciding when a value matters, which is the shape this module was written to end.
///
/// [`Outcome::FallbackLoudly`] is exactly "the run proceeds, but not as asked, and the
/// caller must SAY it". The cost is one warn line per Glimmer run; the alternative is a
/// silent drop, and this table's whole premise is that a silent drop is worse than noise.
///
/// # The rest
///
/// `--cache-policy` and `--trace` are presence-judged (`main::PRESENCE`), so refusing them
/// costs an untyped run nothing and tells a typed one why. `--attn dense` is genuinely what
/// this model does — GQA 32Q/2KV over the whole causal prefix on its full layers — so it is
/// the one attention cell here that is `Support` on its own merits rather than by default.
fn muse_glimmer(flag: Flag) -> Outcome {
    match flag {
        // Every value, together: on a dense model they do not differ, which is the same
        // shape the header notes for `CachePolicy` on any architecture.
        Flag::Mode(Mode::Int3Vq | Mode::Int4 | Mode::Hybrid) => {
            Outcome::FallbackLoudly(DENSE_HAS_NO_ROUTED_FORMAT)
        }
        Flag::Attn(AttnKind::Dense) => Outcome::Support,
        // The three sparse selections are GLM-shaped in different ways — `dsa` names a
        // trained-in indexer this checkpoint does not ship at all — but the deferral is the
        // same one, so it quotes the same reason rather than inventing a second wording.
        Flag::Attn(AttnKind::Streaming | AttnKind::Dsa | AttnKind::Misa) => {
            Outcome::Refuse(SPARSE_ATTN_DEFERRED)
        }
        Flag::CachePolicy => Outcome::Refuse(DENSE_CACHE_POLICY_IS_ONE_ANSWER),
        Flag::Trace => Outcome::Refuse(DENSE_HAS_NO_EXPERT_TRACE),
        Flag::Mtp => Outcome::Refuse(MTP_DEFERRED),
        // `--max-mem` is the knob that decides how many of the 52 layers are pinned, and
        // `--ctx` sizes a KV cache this arm's floor CHARGES for (unlike GLM's, which
        // budgets weights only) — so both are not merely accepted here, they are the two
        // numbers the partition is a function of.
        Flag::MaxMem | Flag::Ctx => Outcome::Support,
    }
}

/// DeepSeek-V4-Flash's row of the table — and most of it is one fact worked out flag by flag:
/// **this architecture chooses neither its expert format nor its attention selection.**
///
/// # EVERY value-carrying flag falls back loudly, and the choice is forced the same way
/// Glimmer's `--mode` was
///
/// `--mode` because the checkpoint owns the format, and every `--attn` value because the
/// checkpoint owns the attention too: a run here attends a ring plus pooled blocks, with
/// the trained-in block indexer ranking the blocks on the 21 indexed layers — natively and
/// always, since M15 placed its weights and wired its kernels. `Support` on any of the four
/// would record a knob nothing spent, and `Refuse` would kill runs on a flag that
/// contradicts nothing the user can change — [`muse_glimmer`]'s doc carries the
/// no-flags half of the argument. The sparse three were `Refuse` while the indexer was
/// unplaced and the positional stand-in imposed a context ceiling; with the ceiling gone,
/// refusing `--attn dsa` on the one arm whose attention IS natively block-sparse would
/// kill exactly the runs the flag's intent describes.
///
/// # The rest
///
/// `--cache-policy` and `--max-mem` are real here and for a sharper reason than on GLM: the
/// `.f4` set is 137 GiB against ~115 GiB of budget, so this arm CANNOT be fully resident on
/// any configuration this machine has and the policy governs every token. `--ctx` sizes the
/// KV ring, the compressed regions and the indexer's score scratch — and nothing refuses it
/// since M15: memory is its only bound.
fn deepseek_v4(flag: Flag) -> Outcome {
    match flag {
        // Every value together: the checkpoint's format is the same whichever was typed.
        Flag::Mode(Mode::Int3Vq | Mode::Int4 | Mode::Hybrid) => {
            Outcome::FallbackLoudly(V4_FORMAT_IS_THE_CHECKPOINTS)
        }
        Flag::Attn(AttnKind::Dense) => Outcome::FallbackLoudly(V4_ATTENTION_IS_WINDOW_PLUS_BLOCKS),
        // Every sparse value together, like `--mode`: the checkpoint's attention is the
        // same whichever was typed. Flipped from `Refuse` when M15 landed the scored
        // selection — the row doc carries the argument.
        Flag::Attn(AttnKind::Streaming | AttnKind::Dsa | AttnKind::Misa) => {
            Outcome::FallbackLoudly(V4_SPARSE_ATTN_IS_NOT_A_CHOICE)
        }
        // The three knobs that are MORE real here than on GLM: the `.f4` set cannot fit on
        // this machine, so residency governs every token rather than the tail of a working
        // set — and `--ctx` sizes both the KV ring and the compressed region.
        Flag::CachePolicy | Flag::MaxMem | Flag::Ctx => Outcome::Support,
        Flag::Trace => Outcome::FallbackLoudly(V4_TRACE_PREFILL_IS_ONE_PASS),
        // NOT the shared `MTP_DEFERRED`: that one says the head is not loaded and the verify
        // pass is not built, and describes a decode-loop increment on a batch shape already
        // designed in. Here the blocker is one instantiation lower down and no other arm
        // shares it, so quoting GLM's wording would tell a user to wait for the wrong thing.
        Flag::Mtp => Outcome::Refuse(V4_MTP_NEEDS_A_KERNEL),
    }
}

#[cfg(test)]
mod tests;
