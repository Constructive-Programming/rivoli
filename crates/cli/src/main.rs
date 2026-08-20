//! rivoli — MoE decode engine (HIP/ROCm). The artifact IS the model: point `rivoli` at a
//! converted artifact directory and it sniffs the architecture, checks every flag you
//! typed against that architecture, opens the engine and decodes.
//!
//! Zero-knob by design — the machine is auto-discovered; the flags below are benchmark and
//! diagnostic overrides, not required configuration.
//!
//! This file is deliberately thin: parse → sniff → open → loop. Everything that could
//! judge a configuration lives in `rivoli_core::legality`, and everything that could build
//! a device lives behind `rivoli_engine::Engine::open` — including the refusal a build with
//! no compute backend gives, which is why there is not one `#[cfg]` in this file.
//!
//! A run is exactly one of three things — `--bench N` (decode N tokens, print, exit),
//! `--ppl FILE` (teacher-forced scoring, the quality instrument) or `--port P` (serve
//! until killed) — and clap enforces the exclusivity, so no branch below has to defend
//! against another having been asked for too.

use anyhow::{Context, Result, bail};
// One nested use for the artifact types: as five separate lines this preamble reproduced
// another file's token for token the moment the K3 import landed, and the jscpd gate
// reported the pair — an import list being the one duplication Rust cannot factor.
use rivoli_artifact::{
    format::RoutedFmt, glimmer_config::GlimmerConfig, glm_config::ModelConfig, k3_config::K3Config,
    tokenizer::Tokenizer, v4_config::V4Config,
};
use rivoli_core::cache::TwoQSplit;
use rivoli_core::legality::{ATTNS, Arch, AttnKind, Flag, MODES, Mode, Outcome, decide};
use rivoli_engine::{ArchCfg, Engine, OpenSpec, PoolKnobs};
use std::collections::HashSet;

// Private to the binary rather than `pub` in `lib.rs`: these have exactly one consumer,
// the `rivoli` process, and publishing it would invent a crate API nobody imports. Its
// tests still run under `cargo test --workspace`, which builds and tests bin targets.
//
// BELOW the imports, not above: with `mod serve;` on top, this file's opening
// `use anyhow::{...}; use rivoli_artifact::format::RoutedFmt;` reproduced `routed.rs`'s
// preamble token for token and the jscpd gate reported it (46 tokens, 2026-08-16).
// rustfmt sorts the `use` list, so the module lines are the only movable item. The
// anyhow list shrank when `--bench` moved to `bench.rs`; the placement stays, because
// what made the clone was the ADJACENCY, not the exact import set.
mod bench;
mod nll;
mod serve;

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
struct Args {
    /// The converted artifact directory (manifest.json + codebooks + resident.safetensors
    /// + per-layer expert files + tokenizer). The artifact IS the model.
    model: String,

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
    bench: Option<u64>,

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
    ppl: Option<String>,

    /// Where --ppl writes its per-token NLLs (the `# rivoli-nll v1` file bin/ppl reads).
    #[arg(long, value_name = "PATH", requires = "ppl")]
    ppl_out: Option<String>,

    /// Serve an OpenAI-compatible HTTP API on 127.0.0.1:PORT until killed — this is how
    /// llama-swap (and any OpenAI client) calls the engine. `POST /v1/chat/completions`
    /// with or without `stream`, plus `GET /health` and `GET /v1/models`.
    ///
    /// The port opens only once the model is loaded, so it doubles as the readiness
    /// signal. Sampling is NOT implemented — `temperature`/`top_p` are accepted and
    /// ignored, because the engine decodes greedy argmax.
    #[arg(long, value_name = "PORT", value_parser = clap::value_parser!(u16).range(1..))]
    port: Option<u16>,

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
    think: bool,

    /// Routed-expert format: int3-vq | int4 | hybrid. int4 scores best and is slowest;
    /// hybrid is refused at startup with the reason.
    #[arg(long, default_value = "int3-vq", value_parser = parse_mode)]
    mode: Mode,

    /// Attention row selection: dense | streaming | dsa | misa. What each architecture does
    /// with it is its own row of the legality table, and the run says which at startup:
    /// most refuse the other three with the reason; DeepSeek-V4, whose attention is
    /// natively block-sparse, warns and proceeds on all four.
    // No `auto`: it resolved to `dsa` when the artifact carried indexer weights, and with
    // one path built it could only ever resolve to one thing. A value whose whole purpose
    // is to choose, that cannot choose, is a value that lies.
    #[arg(long, default_value = "dense", value_parser = parse_attn)]
    attn: AttnKind,

    /// Routed-expert cache policy. Output-neutral by construction: routing never consults
    /// residency, so a policy change moves throughput and hit rate, never the tokens
    /// produced.
    #[arg(long, default_value = "2q", value_parser = ["lru", "2q", "arc"])]
    cache_policy: String,

    /// Device budget in GiB, taken LITERALLY — no OS reserve, so it may OOM while pinning.
    /// Without it the budget auto-sizes to `free − 16 GiB`.
    #[arg(long, value_name = "GIB", value_parser = clap::value_parser!(u64).range(1..))]
    max_mem: Option<u64>,

    /// The context window, in tokens. The KV cache is allocated ONCE at startup (there is
    /// no paging here), so this is a hard ceiling on prompt + generated: a `--bench` run
    /// that does not fit is refused before any weight is placed, and under `--port` a
    /// conversation that does not fit is refused with a 400 rather than silently
    /// truncated. Costs ~51 KB of device memory per token, on top of `--max-mem`'s expert
    /// pool.
    #[arg(long, value_name = "N", default_value_t = 4096, value_parser = clap::value_parser!(usize))]
    ctx: usize,

    /// Override the fixed bench prompt, for capturing traces of diverse inputs.
    // Refused under `--port` for the same reason the legality table refuses rather than
    // ignores: there every request brings its own prompt, so this flag would be silently
    // dropped, and a recorded command line carrying a knob that did nothing is exactly the
    // lie the table exists to stop. Spelled `conflicts_with` — see `--think` for why the
    // `requires` form is not used.
    #[arg(long, value_name = "TEXT", conflicts_with = "port")]
    prompt: Option<String>,

    /// Dump the routed-expert access trace (v2: demand keys plus the ranked candidate
    /// window) for the offline residency sim. Forces a token-major prefill — see the
    /// warning it prints.
    // `conflicts_with = "port"` for the reason --prompt and --dump-ids already refuse
    // there: one run, many replies — a served process would append every request's
    // selections into ONE v2 trace with no request delimiter, and the offline sim would
    // read it as a single decode (review 2026-08-16).
    #[arg(long, value_name = "PATH", conflicts_with = "port")]
    trace: Option<String>,

    /// Speculative decode with the multi-token-prediction head. Refused at startup with
    /// the reason: the head is not loaded and the verify pass is not built.
    // Exposed rather than omitted so a command line ported from the reference engine gets
    // a named deferral instead of clap's "unexpected argument", which reads as a typo.
    #[arg(long)]
    mtp: bool,

    /// Write the generated token ids, one per line, under a header naming the arm.
    ///
    /// Comparing decoded TEXT is not a substitute: different id sequences can decode to
    /// identical text, so a text diff reports only a lower bound on divergence. Any
    /// two-arm comparison across a refactor wants this rather than an eyeball.
    // Refused under `--port`: it dumps ONE run's ids under a header naming the arm, and a
    // server answers many. See `--prompt` for why this is a refusal and not a silent drop.
    #[arg(long, value_name = "PATH", conflicts_with = "port")]
    dump_ids: Option<String>,

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
    arena_refresh: bool,

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
    copy_via_cpu: bool,

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
    divergence_log: Option<String>,

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
    divergence_folds: Option<rivoli_engine::probe::Folds>,
}

/// `--divergence-folds`' parser, so clap refuses a bad spec before the artifact loads.
///
/// The engine owns the grammar (`Folds::parse`); this only adapts its error to clap's `String`.
#[cfg(all(feature = "rocm", feature = "corruption-probe"))]
fn parse_folds(s: &str) -> Result<rivoli_engine::probe::Folds, String> {
    rivoli_engine::probe::Folds::parse(s).map_err(|e| format!("{e:#}"))
}

/// The default bench prompt.
///
/// A FIXED default is the point: two arms of this engine are only comparable if they
/// decoded the same tokens, and a prompt that drifted would invalidate the comparison with
/// nothing to point at. The spelling is the reference engine's, reused because it costs
/// nothing and keeps recorded command lines readable — NOT as a claim that numbers cross
/// engines. `--prompt` overrides it.
const BENCH_PROMPT: &str = "The sky is blue because";

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
type Explicit = HashSet<String>;

fn parse_args() -> (Args, Explicit) {
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
const PRESENCE: [(&str, Flag); 5] = [
    ("cache_policy", Flag::CachePolicy),
    ("max_mem", Flag::MaxMem),
    ("ctx", Flag::Ctx),
    ("trace", Flag::Trace),
    ("mtp", Flag::Mtp),
];

/// Everything this run asks for, in the legality table's vocabulary.
fn requested_flags(a: &Args, explicit: &Explicit) -> Vec<Flag> {
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

/// Put every requested flag past `arch`'s row of the table. The refusal text is the
/// table's, never this file's — a refusal written here would be a second authority on a
/// question that has one.
fn check_legality(arch: Arch, flags: &[Flag]) -> Result<()> {
    for &f in flags {
        match decide(arch, f) {
            Outcome::Support => {}
            Outcome::FallbackLoudly(why) => tracing::warn!("{}: {why}", f.spelling()),
            Outcome::Refuse(why) => bail!(
                "{} is not available on {} ({}): {why}",
                f.spelling(),
                arch.name(),
                arch.summary()
            ),
        }
    }
    Ok(())
}

/// The one place a user-facing mode becomes a stored weight format. `None` for `hybrid`,
/// which names a run with more than one format in it and therefore has no single value
/// here — the caller reports that as the internal inconsistency it would be, since
/// `check_legality` has already refused it.
fn routed_fmt(mode: Mode) -> Option<RoutedFmt> {
    match mode {
        Mode::Int3Vq => Some(RoutedFmt::Vq3),
        Mode::Int4 => Some(RoutedFmt::I4),
        Mode::Hybrid => None,
    }
}

fn main() -> Result<()> {
    // Logs on stderr: stdout carries the generated TEXT, and a log line interleaved into
    // it would corrupt the one output a reader (or a diff of two arms) is looking at.
    // `info` by default because the startup log — budget, placement, prefill — is how a run
    // says what it actually did, and a silent engine is one nobody can cite.
    let directives = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(tracing_subscriber::EnvFilter::new(directives))
        .init();
    let (a, explicit) = parse_args();

    // **The sniff, and it reads the manifest rather than a loaded config.** Until M7 this
    // was `ModelConfig::load` followed by `<ModelConfig as ArchConfig>::ARCH` — the config
    // in hand WAS the evidence, because that load refuses every other architecture before
    // serde reads a dimension. With two arms that shape can only ever answer "GLM": which
    // config type to load is now the question, so it cannot also be the answer. The
    // per-type refusal stays behind this as the check that sniff and parse agreed.
    let arch = rivoli_artifact::schema::arch_of_artifact(&a.model)?;
    tracing::info!("{}: {} ({})", a.model, arch.name(), arch.summary());
    check_legality(arch, &requested_flags(&a, &explicit))?;
    // The door check, BEFORE the tokenizer's 19 MB vocab parse and the prompt encode: a
    // backendless build refuses here or the seam doc's "at the door" claim is prose.
    Engine::ensure_backend()?;
    // Same door, same reason, for the scoring instrument: a stock build (no
    // `teacher-forcing`) refuses `--ppl` before anything expensive happens, citing the
    // one authority (`rivoli_engine::seam::TF_SCORING_NOT_BUILT`) that `Engine::score`
    // itself would cite minutes later.
    if a.ppl.is_some() {
        Engine::ensure_scoring()?;
    }

    let tok = Tokenizer::load(&a.model)?;
    // `--bench` and `--ppl` resolve their input BEFORE the engine opens, so a run that
    // outgrows the KV slab is refused before any weight is placed rather than failing in
    // the token loop, minutes in. `--port` has no prompt to check here: every request
    // brings its own, and `serve` checks each against `Engine::max_ctx` and answers 400.
    let bench = a
        .bench
        .map(|ngen| bench::bench_input(&tok, arch, &a, ngen as usize))
        .transpose()?;
    let ppl = a
        .ppl
        .as_deref()
        .map(|path| nll::ppl_input(&tok, path, &a))
        .transpose()?;
    let inv = invocation(&a, bench.as_ref(), ppl.as_ref())?;

    // **Each arm owns its config for the whole run, because the engine borrows it.** The
    // arms are two lines apiece and then rejoin: `open_and_run` is the shared tail, so a
    // third architecture cannot acquire a second bench loop or a second `serve` call.
    match arch {
        Arch::GlmMoeDsa => {
            let cfg = ModelConfig::load(&a.model)?;
            let arch_cfg = ArchCfg::Glm(&cfg, glm_fmt(&a)?, pool_knobs(&a));
            open_and_run(&a, &tok, inv, arch_cfg)
        }
        // No `RoutedSpec`: a dense model has no routed pool to configure, and the legality
        // table has already told the user so for each of the flags that would have filled
        // one in.
        Arch::MuseGlimmer => {
            let cfg = GlimmerConfig::load(&a.model)?;
            open_and_run(&a, &tok, inv, ArchCfg::Glimmer(&cfg))
        }
        // `PoolKnobs` and not `RoutedSpec`: this architecture HAS a routed pool and does not
        // have a routed FORMAT — the checkpoint's experts are `.f4` and there is nothing for
        // `--mode` to select, which is what the three `FallbackLoudly` cells told the user.
        Arch::DeepseekV4 => {
            let cfg = V4Config::load(&a.model)?;
            open_and_run(&a, &tok, inv, ArchCfg::V4(&cfg, pool_knobs(&a)))
        }
        // The match went EXHAUSTIVE when this arm landed (M9): the "no decode path" bail
        // that stood here would now be an unreachable pattern, which is the design working —
        // a fifth architecture reopens the hole and the compiler reports it.
        Arch::KimiK3 => {
            // `--port` refuses at THIS door, before a terabyte-class pin. Not a legality
            // row, because `--port` is the INVOCATION (the table's own scope rule) — but
            // the same courtesy: the reason, not "invalid".
            if a.port.is_some() {
                bail!("{K3_PORT_HAS_NO_CHAT_ENCODING}");
            }
            let cfg = K3Config::load(&a.model)?;
            open_and_run(&a, &tok, inv, ArchCfg::K3(&cfg, pool_knobs(&a)))
        }
    }
}

/// Why a K3 artifact cannot serve. A named const like the legality rows' and for the same
/// reason: the wording is data, cited by the refusal and pinned by a test.
const K3_PORT_HAS_NO_CHAT_ENCODING: &str = "--port needs a chat encoding and Kimi-K3 ships \
     NONE in any tree — its tokenizer_config.json has no chat_template, so `convert_k3` \
     copies no template, and inventing a framing here would feed the model turn markers it \
     never saw (an instruct model outside its turn structure never emits a stop token — the \
     failure that invalidated 56 benchmark runs in the old tree). Use --bench, whose prompt \
     is encoded RAW on this architecture.";

/// GLM's routed FORMAT — the one knob V4 does not have.
///
/// `--mode hybrid` has no [`RoutedFmt`] at all, which is why `legality::decide` refuses it
/// rather than resolving it to one of the two real formats; reaching the `context` below means
/// that refusal did not fire.
fn glm_fmt(a: &Args) -> Result<RoutedFmt> {
    routed_fmt(a.mode)
        .context("--mode hybrid reached the format mapping — legality::decide must refuse it first")
}

/// The routed pool's knobs, gathered in the one place they are legal.
///
/// Shared by both routed arms because they describe a POOL and both have one — see
/// [`PoolKnobs`]. Whether the flags that fill it are legal on this architecture is the
/// legality table's question and has already been asked by the time this runs.
fn pool_knobs(a: &Args) -> PoolKnobs<'_> {
    PoolKnobs {
        cache_policy: &a.cache_policy,
        two_q: TwoQSplit::default(),
        trace_path: a.trace.as_deref(),
        arena_refresh: a.arena_refresh,
        copy_via_cpu: a.copy_via_cpu,
    }
}

/// The one thing this process will do, resolved from the three exclusive flags. A closed
/// enum rather than three `Option`s threaded side by side, so `open_and_run`'s dispatch
/// is exhaustive and a fourth invocation breaks the match instead of falling through.
enum Invocation<'a> {
    Bench(&'a bench::Bench<'a>),
    Ppl(&'a nll::Ppl),
    Serve(u16),
}

/// Resolve the invocation, or report the contract undone.
fn invocation<'a>(
    a: &Args,
    bench: Option<&'a bench::Bench<'a>>,
    ppl: Option<&'a nll::Ppl>,
) -> Result<Invocation<'a>> {
    match (bench, ppl, a.port) {
        (Some(b), None, None) => Ok(Invocation::Bench(b)),
        (None, Some(p), None) => Ok(Invocation::Ppl(p)),
        (None, None, Some(port)) => Ok(Invocation::Serve(port)),
        // Unreachable through clap (`--bench` is `required_unless_present_any = ["port",
        // "ppl"]` and the three conflict pairwise), and a `bail!` rather than a
        // `debug_assert!`: this is the one place the contract is read, and under
        // `--release` a debug assert would enforce nothing at all.
        _ => bail!(
            "exactly one of --bench, --ppl and --port must be given, which clap should \
             have refused — the exclusivity attributes on `Args` have come undone"
        ),
    }
}

/// Open the engine on `cfg`'s arm and run whichever invocation was asked for.
///
/// **The shared tail.** Everything below the seam is architecture-shaped and everything here
/// is not, so this function names no architecture — which is what makes "a third arm is two
/// lines in `main`" true rather than aspirational. M11b needed the architecture in
/// `serve::Opts` — chat framing is a property of the CHECKPOINT and sits outside the engine
/// seam — and it comes from `cfg.arch()` rather than a fifth parameter, because the config
/// already knows which architecture it is.
fn open_and_run(a: &Args, tok: &Tokenizer, inv: Invocation<'_>, cfg: ArchCfg<'_>) -> Result<()> {
    // Read BEFORE the config moves into `Engine::open`. `ArchCfg` is not `Copy`, so this
    // ordering is the borrow checker's, not a preference.
    let arch = cfg.arch();
    let mut eng = Engine::open(
        &a.model,
        cfg,
        OpenSpec {
            max_mem_gib: a.max_mem,
            max_ctx: a.ctx,
        },
    )?;
    match inv {
        Invocation::Bench(b) => bench::run_bench(&mut eng, tok, a, b),
        Invocation::Ppl(p) => nll::run_ppl(&mut eng, a, p),
        Invocation::Serve(port) => serve::serve(
            &mut eng,
            tok,
            &serve::Opts {
                port,
                // The artifact directory's own name, so `/v1/models` and the echoed
                // `model` field say which checkpoint answered.
                model_id: std::path::Path::new(&a.model)
                    .file_name()
                    .map_or_else(|| "rivoli".into(), |n| n.to_string_lossy().into_owned()),
                think: a.think,
                // Which chat template frames a request, and reads the reply back — see
                // `serve::Opts::arch`. Read off the config rather than matched here, which is
                // what keeps this function's "names no architecture" claim true.
                arch,
            },
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// The invocation contract, PARSED rather than trusted to an attribute.
    ///
    /// This exists because the first spelling was `requires = "port"` / `requires =
    /// "bench"`, and it is INERT in clap 4.6.6: `rivoli DIR --bench 4 --think` parsed clean
    /// and went on to load a model. Nothing but a parse test tells two attribute spellings
    /// apart, and an exclusivity nobody can see fail is decoration. The accepted rows are
    /// here so this cannot pass by refusing everything.
    #[test]
    fn exactly_one_of_bench_and_port_and_the_bench_only_flags_say_so() {
        let refused: [&[&str]; 10] = [
            &["rivoli", "DIR"],                                       // no invocation
            &["rivoli", "DIR", "--bench", "4", "--port", "8080"],     // two invocations
            &["rivoli", "DIR", "--bench", "4", "--think"],            // --think is server-only
            &["rivoli", "DIR", "--port", "8080", "--prompt", "x"],    // requests bring prompts
            &["rivoli", "DIR", "--port", "8080", "--dump-ids", "/x"], // one run, many replies
            // --ppl is its own invocation: exclusive with the other two, inseparable
            // from --ppl-out (the .nll file IS the deliverable), and --ppl-out without
            // --ppl has nothing to write.
            &["rivoli", "DIR", "--ppl", "c.txt"],
            &["rivoli", "DIR", "--ppl-out", "/x.nll"],
            &[
                "rivoli",
                "DIR",
                "--bench",
                "4",
                "--ppl",
                "c.txt",
                "--ppl-out",
                "/x.nll",
            ],
            &[
                "rivoli",
                "DIR",
                "--port",
                "8080",
                "--ppl",
                "c.txt",
                "--ppl-out",
                "/x.nll",
            ],
            // A scoring run never decodes free-running text, so a trace of "the decode's
            // expert selections" would be a file about a run that did not happen.
            &[
                "rivoli",
                "DIR",
                "--ppl",
                "c",
                "--ppl-out",
                "/x",
                "--trace",
                "/t",
            ],
        ];
        for argv in refused {
            assert!(
                Args::command().try_get_matches_from(argv).is_err(),
                "clap accepted {argv:?}, which is not a legal invocation"
            );
        }
        let accepted: [&[&str]; 5] = [
            &["rivoli", "DIR", "--bench", "4"],
            &["rivoli", "DIR", "--port", "8080"],
            &["rivoli", "DIR", "--port", "8080", "--think"],
            &["rivoli", "DIR", "--ppl", "c.txt", "--ppl-out", "/x.nll"],
            // The sweep knobs stay legal under --ppl — mode/policy/budget ARE the cells
            // a paired comparison distinguishes.
            &[
                "rivoli",
                "DIR",
                "--ppl",
                "c",
                "--ppl-out",
                "/x",
                "--mode",
                "int4",
                "--max-mem",
                "70",
            ],
        ];
        for argv in accepted {
            assert!(
                Args::command().try_get_matches_from(argv).is_ok(),
                "clap refused {argv:?}, which is a legal invocation"
            );
        }
    }

    /// Every id in [`PRESENCE`] must name a real argument. Without this the strings are
    /// unchecked: renaming a field would leave its flag permanently unrequested, and the
    /// legality table would go quietly blind to it while every test above stayed green.
    #[test]
    fn presence_flag_ids_name_real_arguments() {
        let real: HashSet<String> = Args::command()
            .get_arguments()
            .map(|arg| arg.get_id().as_str().to_string())
            .collect();
        for (id, flag) in PRESENCE {
            assert!(
                real.contains(id),
                "{} is checked under an unknown clap id {id:?}",
                flag.spelling()
            );
        }
    }

    /// The value-carrying half of the same claim: `--mode` and `--attn` are judged on
    /// their resolved value, so they must not ALSO be in [`PRESENCE`] — a flag checked
    /// twice would report a refusal twice and, worse, invite the two checks to diverge.
    #[test]
    fn value_carrying_flags_are_not_also_presence_flags() {
        for (id, _) in PRESENCE {
            assert!(
                id != "mode" && id != "attn",
                "{id} is judged by value, not presence"
            );
        }
    }

    /// **The arms `main` dispatches to and the table's refusals must name the same set.**
    ///
    /// `main` sniffs, runs `check_legality`, and only then dispatches on the architecture.
    /// Since M9 the match is exhaustive with a real arm per variant, so the "no decode
    /// path" bail is gone — but the contract survives it: an architecture with NO arm must
    /// have `--mode` or `--attn` refused (that is what kept the bail unreachable while one
    /// existed), and an architecture WITH an arm must refuse neither, which is
    /// `rivoli_core::legality`'s
    /// `every_architecture_with_an_arm_decodes_with_no_flags_typed`. [`ARMS`] below is what
    /// ties the two lists together, and a fifth architecture lands red here until both
    /// sides answer for it.
    ///
    /// The defaults come from PARSING a bare invocation through the same two functions
    /// `main` uses, not from restating clap's `default_value` attributes — a restatement is
    /// free to drift from what the binary actually asks for, which is the whole failure this
    /// file's other parse tests exist for.
    #[test]
    fn the_arms_and_the_legality_table_agree_about_who_can_start() {
        // Hand-written, on the same argument as `legality::ARCH_COUNT`: no test can observe
        // a `match`'s arms, so the list is stated and the assertion is what binds it. Adding
        // an arm to `main` without a row here leaves that architecture unchecked.
        const ARMS: [Arch; 4] = [
            Arch::GlmMoeDsa,
            Arch::MuseGlimmer,
            Arch::DeepseekV4,
            Arch::KimiK3,
        ];
        let Ok(a) = <Args as clap::Parser>::try_parse_from(["rivoli", "DIR", "--bench", "1"])
        else {
            panic!("`rivoli DIR --bench 1` must parse — the invocation contract has moved");
        };
        let defaults = requested_flags(&a, &Explicit::new());
        assert_eq!(
            defaults.len(),
            2,
            "a no-flag run asks for --mode and --attn and nothing else; got {defaults:?}"
        );
        for arch in Arch::ALL {
            let refused = defaults
                .iter()
                .any(|&f| matches!(decide(arch, f), Outcome::Refuse(_)));
            let has_arm = ARMS.contains(&arch);
            assert_eq!(
                !refused,
                has_arm,
                "{}: main {} a decode arm, but the table {} the flags a bare invocation \
                 asks for — one of the two is wrong, and the visible symptom is either a \
                 model that cannot be started or `main`'s \"should have refused at the \
                 door\" bail becoming reachable",
                arch.name(),
                if has_arm { "has" } else { "has no" },
                if refused { "refuses" } else { "accepts" },
            );
        }
    }

    /// Only `hybrid` lacks a stored format; the other two must map, or a legal mode would
    /// die at the `context` line instead of decoding.
    #[test]
    fn every_supported_mode_has_a_stored_format() {
        for (name, mode) in MODES {
            let mapped = routed_fmt(mode).is_some();
            let legal = decide(Arch::GlmMoeDsa, Flag::Mode(mode)) == Outcome::Support;
            assert_eq!(
                mapped, legal,
                "{name}: legality and the format mapping disagree"
            );
        }
    }
}
