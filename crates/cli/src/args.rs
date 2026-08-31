//! `--help`'s shape, and the two questions the legality table needs answered about it.
//!
//! **Split out of `main.rs` verbatim on 2026-08-31, to pay the 800-line soft cap before S6
//! needs the headroom.** `main.rs` was already over the cap and the S1 commit grew it (824 →
//! 838), with the shrink deferred to the one commit least able to afford it: S6 replaces
//! `main`'s fifth-architecture `bail!` with a real engine arm. The house rule attached to
//! that warning is that the NEXT edit shrinks the file, so this is that edit. Nothing here
//! changed behaviour — the clap declarations, both value parsers, [`parse_args`],
//! [`PRESENCE`] and [`requested_flags`] moved as they stood, and the only edits are the
//! `pub(crate)` markers a module boundary requires.
//!
//! **Why THIS cut.** `main.rs`'s header calls it "deliberately thin: parse → sniff → open →
//! loop", and 325 of its lines were the `parse` step's vocabulary rather than the pipeline.
//! The seam is narrow by construction: three items leave this module ([`Args`],
//! [`parse_args`], [`requested_flags`], plus [`Explicit`] and [`PRESENCE`] for the tests that
//! pin them), and everything else — the two `value_parser` adapters, the clap plumbing that
//! recovers what was TYPED — is private here for the first time.
//!
//! **The fields are `pub(crate)` and that is the one real cost.** Inside `main.rs` they were
//! private to the crate root and `bench`/`nll` reached them as descendant modules; from a
//! sibling module they need a visibility. `pub(crate)` and not `pub`: the binary's `Args` is
//! not crate API — `main.rs`'s note that publishing it "would invent a crate API nobody
//! imports" still holds.

use anyhow::Result;
use rivoli_core::legality::{ATTNS, AttnKind, Flag, MODES, Mode};
use std::collections::HashSet;

// NOTE: doc comments on this struct and its fields are USER-FACING — clap renders them as
// `--help`. Rationale for the code goes in `//` comments like this one, which clap ignores.
//
// No `wrap_help` feature, so clap does not re-wrap: the terminal does. That also means
// `max_term_width` would be inert, which is why it is not set.
/// Zero-knob by design: every flag is a benchmark or diagnostic override, and the artifact
/// path plus `--bench N` is a complete invocation.
#[derive(clap::Parser, Debug)]
#[command(
    name = "rivoli",
    version,
    // Architecture-neutral: the artifact decides which model this is, and a flag that is
    // legal here may be refused on the next artifact. `rivoli_core::legality` is the one
    // place that knows which, and it says so at startup with the reason.
    about = "MoE decode engine (HIP/ROCm). The artifact names its own architecture;\n\
             flags are checked against it at startup and refused with the reason."
)]
pub(crate) struct Args {
    /// The converted artifact directory (manifest.json + codebooks + resident.safetensors
    /// + per-layer expert files + tokenizer). The artifact IS the model.
    pub(crate) model: String,

    /// Decode this many tokens, print the text and the DECODE line, exit. Omit for the
    /// server path (`--port`); exactly one of the two is required.
    ///
    /// Spelled `-bench` in every recorded command line of the reference engine; `main`
    /// rewrites that single-dash form to `--bench` before clap sees it, since clap has no
    /// single-dash-long concept. Both work.
    // The exclusivity is clap's, not a runtime bail: `conflicts_with` + `required_unless_
    // present` make "neither" and "both" usage errors, rendered by clap's own error path
    // with the usage line attached. A hand-written message would say the same thing worse
    // and would not appear in `--help`.
    // `.range(1..)`: the decode loop decides token T before checking the budget, so
    // `--bench 0` would still generate one token (review 2026-08-16); zero is refused at
    // the door, matching serve's max_tokens floor.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u64).range(1..),
          conflicts_with = "port", required_unless_present_any = ["port", "ppl"])]
    pub(crate) bench: Option<u64>,

    /// Score this text file TEACHER-FORCED — per-token NLL over a fixed corpus, the
    /// producer for bin/ppl's paired-dNLL comparison — and write one NLL per predicted
    /// token to --ppl-out. Encoded RAW (a text to score, not a turn to answer). Needs a
    /// `--features teacher-forcing` build; a stock binary refuses at startup with the
    /// reason.
    // A third invocation beside --bench and --port, exclusive with both: a scoring run
    // walks a fixed corpus instead of decoding, so a token budget is meaningless and a
    // server has no corpus. `requires = "ppl_out"` mirrors the reference engine: the
    // per-token .nll file IS the deliverable (the PPL: line alone cannot be paired), so
    // a run that would discard it is refused at parse. The feature gate is checked at
    // the DOOR (`Engine::ensure_scoring`, before the tokenizer's vocab parse), not by
    // hiding the flag — main.rs stays free of #[cfg], and a visible flag that names its
    // build requirement beats one that vanishes.
    #[arg(long, value_name = "TEXT_FILE", requires = "ppl_out",
          conflicts_with_all = ["bench", "port", "think", "prompt", "dump_ids", "trace", "mtp"])]
    pub(crate) ppl: Option<String>,

    /// Where --ppl writes its per-token NLLs (the `# rivoli-nll v1` file bin/ppl reads).
    #[arg(long, value_name = "PATH", requires = "ppl")]
    pub(crate) ppl_out: Option<String>,

    /// Serve an OpenAI-compatible HTTP API on 127.0.0.1:PORT until killed — this is how
    /// llama-swap (and any OpenAI client) calls the engine. `POST /v1/chat/completions`
    /// with or without `stream`, plus `GET /health` and `GET /v1/models`.
    ///
    /// The port opens only once the model is loaded, so it doubles as the readiness
    /// signal. Sampling is NOT implemented — `temperature`/`top_p` are accepted and
    /// ignored, because the engine decodes greedy argmax.
    #[arg(long, value_name = "PORT", value_parser = clap::value_parser!(u16).range(1..))]
    pub(crate) port: Option<u16>,

    /// `--port`: reason before answering, unless the request says otherwise. A request's
    /// `enable_thinking` (or `reasoning_effort`) overrides this in either direction, and
    /// the reasoning comes back in `reasoning_content`, never mixed into `content`.
    // Why the default inverts the checkpoint's: `ChatOpts::thinking` owns that argument —
    // one home, because a copy here already started drifting once (review 2026-08-16).
    // `conflicts_with` and not `requires = "port"`: the `requires` form was tried and is
    // INERT here — `rivoli DIR --bench 4 --think` sailed past clap and loaded a model
    // (probed 2026-08-16, clap 4.6.6). An attribute that cannot refuse anything is
    // decoration, so the exclusivity is spelled the way this file has already proved works.
    #[arg(long, conflicts_with = "bench")]
    pub(crate) think: bool,

    /// Routed-expert format: int3-vq | int4 | hybrid. int4 scores best and is slowest;
    /// hybrid is refused at startup with the reason.
    #[arg(long, default_value = "int3-vq", value_parser = parse_mode)]
    pub(crate) mode: Mode,

    /// Attention row selection: dense | streaming | dsa | misa. What each architecture does
    /// with it is its own row of the legality table, and the run says which at startup:
    /// most refuse the other three with the reason; DeepSeek-V4, whose attention is
    /// natively block-sparse, warns and proceeds on all four.
    // No `auto`: it resolved to `dsa` when the artifact carried indexer weights, and with
    // one path built it could only ever resolve to one thing. A value whose whole purpose
    // is to choose, that cannot choose, is a value that lies.
    #[arg(long, default_value = "dense", value_parser = parse_attn)]
    pub(crate) attn: AttnKind,

    /// Routed-expert cache policy. Output-neutral by construction: routing never consults
    /// residency, so a policy change moves throughput and hit rate, never the tokens
    /// produced.
    #[arg(long, default_value = "2q", value_parser = ["lru", "2q", "arc"])]
    pub(crate) cache_policy: String,

    /// Device budget in GiB, taken LITERALLY — no OS reserve, so it may OOM while pinning.
    /// Without it the budget auto-sizes to `free − 16 GiB`.
    #[arg(long, value_name = "GIB", value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) max_mem: Option<u64>,

    /// The context window, in tokens. The KV cache is allocated ONCE at startup (there is
    /// no paging here), so this is a hard ceiling on prompt + generated: a `--bench` run
    /// that does not fit is refused before any weight is placed, and under `--port` a
    /// conversation that does not fit is refused with a 400 rather than silently
    /// truncated. Costs ~51 KB of device memory per token, on top of `--max-mem`'s expert
    /// pool.
    #[arg(long, value_name = "N", default_value_t = 4096, value_parser = clap::value_parser!(usize))]
    pub(crate) ctx: usize,

    /// Override the fixed bench prompt, for capturing traces of diverse inputs.
    // Refused under `--port` for the same reason the legality table refuses rather than
    // ignores: there every request brings its own prompt, so this flag would be silently
    // dropped, and a recorded command line carrying a knob that did nothing is exactly the
    // lie the table exists to stop. Spelled `conflicts_with` — see `--think` for why the
    // `requires` form is not used.
    #[arg(long, value_name = "TEXT", conflicts_with = "port")]
    pub(crate) prompt: Option<String>,

    /// Dump the routed-expert access trace (v2: demand keys plus the ranked candidate
    /// window) for the offline residency sim. Forces a token-major prefill — see the
    /// warning it prints.
    // `conflicts_with = "port"` for the reason --prompt and --dump-ids already refuse
    // there: one run, many replies — a served process would append every request's
    // selections into ONE v2 trace with no request delimiter, and the offline sim would
    // read it as a single decode (review 2026-08-16).
    #[arg(long, value_name = "PATH", conflicts_with = "port")]
    pub(crate) trace: Option<String>,

    /// Speculative decode with the multi-token-prediction head. Refused at startup with
    /// the reason: the head is not loaded and the verify pass is not built.
    // Exposed rather than omitted so a command line ported from the reference engine gets
    // a named deferral instead of clap's "unexpected argument", which reads as a typo.
    #[arg(long)]
    pub(crate) mtp: bool,

    /// Write the generated token ids, one per line, under a header naming the arm.
    ///
    /// Comparing decoded TEXT is not a substitute: different id sequences can decode to
    /// identical text, so a text diff reports only a lower bound on divergence. Any
    /// two-arm comparison across a refactor wants this rather than an eyeball.
    // Refused under `--port`: it dumps ONE run's ids under a header naming the arm, and a
    // server answers many. See `--prompt` for why this is a refusal and not a silent drop.
    #[arg(long, value_name = "PATH", conflicts_with = "port")]
    pub(crate) dump_ids: Option<String>,

    /// MITIGATION for the GLM decode nondeterminism: before each bounce->VMM copy, read the
    /// just-written arena window at full width on the fetch stream and discard the value.
    ///
    /// This is the ONLY intervention measured to make GLM decode reproduce itself — four clean
    /// pairs over 6144 tokens against a rate predicting P(all clean) ~ 1e-9 — and its mechanism is
    /// NOT understood. Fifteen alternatives are RED, each read back and confirmed to have applied;
    /// `docs/investigations/glm-nondeterminism-closeout.md` keeps that ablation matrix and
    /// `docs/investigations/glm-nondeterminism-worklog.md` the arm log.
    ///
    /// Not behind a feature, like `--copy-via-cpu`: the acceptance protocol
    /// compares two release binaries differing in exactly this flag. Costs **1-3%** of decode
    /// throughput (2.65 tok/s against 2.70-2.73 at 1536 tokens, measured 2026-08-18 — the ~10%
    /// this said before the protocol ran was the PROBE fold, which also hashed and wrote a 19 MB
    /// log per run). Measure it on the arm rather than quoting either number.
    #[arg(long)]
    pub(crate) arena_refresh: bool,

    /// CANDIDATE FIX: the bounce→slot hop as a HOST memcpy on the reaper thread instead of an
    /// async copy on the fetch stream.
    ///
    /// The defect the matrix localised is that a GPU-side reader (the SDMA copy engine, or a
    /// shader) can read bytes the NVMe's DMA wrote and get stale ones, and that the ONE clean
    /// cell repairs it with a full-width device read of the DMA'd region. This flag removes the
    /// hop rather than repairing it: the arena is read only by the CPU (the visibility the
    /// io_uring CQE actually guarantees — the same one btrfs's datasum verification spends on
    /// every read) and the pool slot is written only by the CPU (the CPU→GPU coherence
    /// `kernels/vmm.hip` was verified to have, which the resident tier's 281 GB startup load
    /// already relies on). The ticket still signals on the fetch stream, so the consumer side
    /// is unchanged. A flag and not a feature, like every fix arm here: the protocol compares
    /// two release binaries differing in exactly this argument.
    #[arg(long)]
    pub(crate) copy_via_cpu: bool,

    /// DIAGNOSTIC: write a per-layer divergence log here — three device-folded quantities
    /// (the MoE's input, its SwiGLU intermediate, the exit residual) plus what the router
    /// saw, picked and where the pool put it.
    ///
    /// `--dump-ids` says THAT two runs diverged; this says WHERE and in WHICH quantity.
    /// Diff two logs: the first differing line is the (position, layer) coordinate, the
    /// first differing column names the mechanism. GLM only, and refused elsewhere.
    ///
    /// Costs no device traffic and no I/O during the run, deliberately: the predecessor
    /// copied the residual to the host every layer and MASKED the fault it was built to
    /// find. For the same reason, do not combine it with `--trace`, which adds a
    /// `device_sync` per layer-with-misses.
    // Refused under `--port` for the reason `--dump-ids` and `--trace` already are: one
    // run, many replies, and no request delimiter in the file.
    // Gated on `rocm` too: the probe is part of the device path, and a flag a deviceless
    // build accepted and could not spend is the "knob nothing spends" `rivoli_core::legality`
    // exists to stop.
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    #[arg(long, value_name = "PATH", conflicts_with = "port")]
    pub(crate) divergence_log: Option<String>,

    /// DIAGNOSTIC: which OPTIONAL folds `--divergence-log` enables, comma-separated —
    /// `xa`, `ac`, `bh`, `sc`, `sc-nop`, `sc-decoy`, `sc-line`, `se`. Default: none (the light
    /// probe).
    ///
    /// The default is none because the all-on fetch-path configuration was measured to SUPPRESS
    /// the divergence it exists to localise (2,048 instrumented tokens, zero events), while the
    /// light probe diverges normally — so the suppressor is the hop folds specifically. Enable ONE
    /// at a time; whichever turns a red pair green is the mask, and its position names where the
    /// mechanism lives.
    ///
    /// Two kinds, and they are not read the same way. `xa`/`ac` are CONSUMER-OUTPUT folds (the
    /// residual before the norm; the MoE accumulator before the drain) — cheap, and a null on them
    /// genuinely constrains what the kernel consumed. `bh`/`sc`/`se` are BYTES-AT-AN-INSTANT folds
    /// on the fetch path; a null on those cannot exonerate a hop, because a corruption landing
    /// between the fold and the consumer's read is invisible to them.
    ///
    /// The `sc` forms are alternatives at one pipeline position, not additions, and each removes
    /// one ingredient of the known suppressor: `sc` reads the whole slot; `sc-nop` is the same
    /// launch with ~no work; `sc-decoy` moves the same bytes from a buffer that is NOT the slot;
    /// `sc-line` touches every cache line of the slot for ~1/32 of the reads. Run `sc-nop` first —
    /// if a bare launch suppresses, every other arm is confounded.
    // Parsed by clap, not in `run_bench`: a typo used to be refused only AFTER the artifact had
    // loaded, which on this model is minutes of a sole-tenant GPU to learn that `sc-lien` is not a
    // fold. The refusals are the point of this flag — a misparse would make a cell green for the
    // wrong reason — so they happen at the door.
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    #[arg(long, value_name = "LIST", requires = "divergence_log", value_parser = parse_folds)]
    pub(crate) divergence_folds: Option<rivoli_engine::probe::Folds>,
}

/// `--divergence-folds`' parser, so clap refuses a bad spec before the artifact loads.
///
/// The engine owns the grammar (`Folds::parse`); this only adapts its error to clap's `String`.
#[cfg(all(feature = "rocm", feature = "corruption-probe"))]
fn parse_folds(s: &str) -> Result<rivoli_engine::probe::Folds, String> {
    rivoli_engine::probe::Folds::parse(s).map_err(|e| format!("{e:#}"))
}

// The two vocabulary parsers. One line each, delegating to the table in `rivoli_core`, so
// the spellings clap accepts and the spellings `--dump-ids` records are the same list.
fn parse_mode(s: &str) -> Result<Mode, String> {
    rivoli_core::legality::parse_in(&MODES, "--mode", s)
}

fn parse_attn(s: &str) -> Result<AttnKind, String> {
    rivoli_core::legality::parse_in(&ATTNS, "--attn", s)
}

/// The clap ids the user actually typed, as opposed to the ones clap defaulted.
///
/// **Needed because "was this flag passed" and "does this flag hold a non-default value"
/// are different questions, and only the first is the one a refusal wants to ask.**
/// Comparing values accepts `--cache-policy 2q` on an architecture that has no cache,
/// because `2q` is what the flag would have held anyway — and a matrix script passes mode,
/// policy and attn explicitly, so that is the ordinary case rather than an exotic one. A
/// recorded command line carrying a knob that was silently dropped is precisely the lie
/// the legality table exists to stop. (Found by review in the reference engine, 2026-08-11:
/// the value-comparison version let a fully-spelled illegal command line through every
/// refusal it had.)
pub(crate) type Explicit = HashSet<String>;

pub(crate) fn parse_args() -> (Args, Explicit) {
    use clap::{CommandFactory, FromArgMatches, parser::ValueSource};
    // Only an exact `-bench` matches, so `-b`, `--bench` and a positional path are all
    // untouched. Known residual, port-faithful: the map has no position awareness, so a
    // VALUE that is literally `-bench` (e.g. `--prompt -bench`) is rewritten too and
    // clap then errors loudly — same behavior as the reference, wrong input impossible
    // to reach without asking for it.
    let argv: Vec<String> = std::env::args()
        .map(|a| if a == "-bench" { "--bench".into() } else { a })
        .collect();
    // `get_matches_from` + `from_arg_matches` rather than `parse_from`, which discards the
    // `ArgMatches` and with it `value_source` — the only thing that knows what was typed.
    // Errors still exit through clap's own renderer, so `--help` and a bad value read
    // exactly as they would otherwise.
    let m = Args::command().get_matches_from(argv);
    let explicit: Explicit = m
        .ids()
        .map(|id| id.as_str().to_string())
        .filter(|id| m.value_source(id) == Some(ValueSource::CommandLine))
        .collect();
    (
        Args::from_arg_matches(&m).unwrap_or_else(|e| e.exit()),
        explicit,
    )
}

/// The presence flags, paired with the clap id that reports them typed.
/// `presence_flag_ids_name_real_arguments` pins each string to a real argument, so renaming
/// a field cannot silently stop a flag from ever being checked.
pub(crate) const PRESENCE: [(&str, Flag); 5] = [
    ("cache_policy", Flag::CachePolicy),
    ("max_mem", Flag::MaxMem),
    ("ctx", Flag::Ctx),
    ("trace", Flag::Trace),
    ("mtp", Flag::Mtp),
];

/// Everything this run asks for, in the legality table's vocabulary.
pub(crate) fn requested_flags(a: &Args, explicit: &Explicit) -> Vec<Flag> {
    // Value-carrying flags are judged on their RESOLVED value, typed or not: the value IS
    // the question here (int4 decodes, hybrid does not), so a default that ever became
    // illegal must be caught rather than exempted by never having been typed.
    let mut flags = vec![Flag::Mode(a.mode), Flag::Attn(a.attn)];
    flags.extend(
        PRESENCE
            .iter()
            .filter(|(id, _)| explicit.contains(*id))
            .map(|(_, f)| *f),
    );
    flags
}
