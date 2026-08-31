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
use rivoli_core::legality::{Arch, Flag, Mode, Outcome, QWEN_ARM_NOT_BUILT, decide};
use rivoli_engine::{ArchCfg, Engine, OpenSpec, PoolKnobs};

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
mod args;
mod bench;
mod nll;
mod serve;

// `Args` and the two questions the legality table asks about it. Imported rather than
// re-exported: `bench` and `nll` name `crate::args::Args` directly, so there is one path
// to the type and no second spelling for a reader to wonder about.
use args::{Args, parse_args, requested_flags};

/// The default bench prompt.
///
/// A FIXED default is the point: two arms of this engine are only comparable if they
/// decoded the same tokens, and a prompt that drifted would invalidate the comparison with
/// nothing to point at. The spelling is the reference engine's, reused because it costs
/// nothing and keeps recorded command lines readable — NOT as a claim that numbers cross
/// engines. `--prompt` overrides it.
const BENCH_PROMPT: &str = "The sky is blue because";

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
        // The fifth architecture, and the hole the comment above predicted: no arm, so the
        // "no decode path" bail is BACK, deliberately, quoting the table's own const rather
        // than inventing a second wording. `check_legality` keeps it unreachable — qwen's row
        // refuses `--mode` and `--attn`, which every invocation asks for — and the ARMS test
        // below ties those two halves together. S6 replaces this arm with a real one.
        Arch::QwenFlashNext => bail!("{QWEN_ARM_NOT_BUILT}"),
    }
}

/// Why a K3 artifact cannot serve. A named const like the legality rows', because the wording
/// is data rather than a string literal buried in a branch.
///
/// > **CORRECTED 2026-08-16, on first contact with the real checkpoint.** The message said
/// > Kimi-K3 "ships NONE in any tree", and that is false: `encoding_k3.py` (647 lines) is a
/// > FIRST-PARTY chat encoder, shipped in the checkpoint beside the weights. The refusal is
/// > unchanged and is now better supported — what K3 lacks is a `chat_template`, which is a
/// > narrower and checkable claim — but a refusal that misstates the checkpoint teaches the
/// > next reader the wrong thing about what porting would cost.
/// >
/// > The doc line here also claimed the wording was "pinned by a test". Nothing references
/// > this const but the `bail!` above — `grep` it. That half is deleted rather than fixed:
/// > the only test such a string admits asserts it equals itself, which this tree deletes on
/// > sight (`quant/naming.rs`'s removed `k3_expert_proj` carries the argument). The pin that
/// > would mean something is a smoke cell invoking `--port`, and K3 has no smoke script yet.
/// >
/// > **And the last sentence was corrected twice in one day.** It read "Use --bench, whose
/// > prompt is encoded RAW on this architecture" — true when written, and made false by this
/// > same commit: dropping `tokenizer.json` from `convert_k3`'s aux list (correctly — K3
/// > ships none) means `Tokenizer::load` at the top of `run` refuses every K3 artifact before
/// > the arch match, so `--bench` does not work either and **the `bail!` above is currently
/// > unreachable**. Caught by review. A refusal that tells the user to do something that also
/// > fails is worse than no advice, and "the fix removed the reader's only working path" is
/// > exactly what a correction pass is for.
const K3_PORT_HAS_NO_CHAT_ENCODING: &str = "--port needs a chat encoding and this build has \
     none for Kimi-K3. The checkpoint ships no `chat_template` (its tokenizer_config.json has \
     no such key and there is no .jinja), so `convert_k3` copies no template. It DOES ship a \
     first-party encoder, `encoding_k3.py` — an XTML framing over <|open|>/<|sep|>/<|close|>/\
     <|end_of_msg|> — and porting THAT, gated against its own output, is the only honest way \
     to fill this in; hand-inventing a framing here would feed the model turn markers it \
     never saw (an instruct model outside its turn structure never emits a stop token — the \
     failure that invalidated 56 benchmark runs in the old tree). Use --bench, whose prompt \
     is encoded RAW on this architecture — its tiktoken vocabulary loads via the \
     first-party-gated loader since the 2026-08-21 merge of wave/m19-k3 (the sentence here \
     said 'no K3 artifact opens at all' while the loader lived only on that branch; see \
     docs/investigations/k3-first-checkpoint.md sections 4 and 7).";

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
    // Only `expect_used`, and only here: a firing panic IS a test's report, and the workspace
    // denies it at bin level. `unwrap_used` stays denied — every panic below names what it
    // expected, which is the half of the rule worth keeping in a test.
    #![allow(clippy::expect_used)]

    use super::*;
    // Test-only, so they sit here rather than at the crate root: an import the binary does
    // not use is a deny-level warning, and `Explicit`/`PRESENCE`/`MODES` are read by the
    // assertions below and by nothing in the pipeline above.
    use crate::args::{Explicit, PRESENCE};
    use clap::CommandFactory;
    use rivoli_core::legality::MODES;
    use std::collections::HashSet;

    /// The rows clap must refuse, each with why beside it. The table is data and the test
    /// below is the loop, so a new illegal combination is one row here, not a new test.
    const REFUSED: [&[&str]; 10] = [
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

    /// The rows that must PARSE, so the refusal table cannot pass by refusing everything.
    const ACCEPTED: [&[&str]; 5] = [
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

    /// The invocation contract, PARSED rather than trusted to an attribute.
    ///
    /// This exists because the first spelling was `requires = "port"` / `requires =
    /// "bench"`, and it is INERT in clap 4.6.6: `rivoli DIR --bench 4 --think` parsed clean
    /// and went on to load a model. Nothing but a parse test tells two attribute spellings
    /// apart, and an exclusivity nobody can see fail is decoration. [`ACCEPTED`] is judged
    /// here too, so this cannot pass by refusing everything.
    #[test]
    fn exactly_one_of_bench_and_port_and_the_bench_only_flags_say_so() {
        for argv in REFUSED {
            assert!(
                Args::command().try_get_matches_from(argv).is_err(),
                "clap accepted {argv:?}, which is not a legal invocation"
            );
        }
        for argv in ACCEPTED {
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

    /// **Which refusal a `--mtp` invocation on the armless architecture actually PRINTS.**
    ///
    /// `rivoli_core::legality`'s qwen row answers `--mtp` with its own 964-character wording,
    /// and a reader — including whoever writes `tests/smoke-qwen.sh`'s `--mtp` cell at S6 —
    /// will assume a user sees it. They do not: [`requested_flags`] puts `Flag::Mode` first
    /// and [`check_legality`] bails on the FIRST `Refuse`, so the `--mode` cell answers and
    /// the run stops. The consequence is a smoke expectation, which is why it is gated here
    /// rather than left as a comment: until the arm row flips, that cell must quote
    /// `QWEN_ARM_NOT_BUILT`'s fragment, and after it flips this test's `Refuse` becomes a
    /// `Support` and reddens exactly where the expectation has to change.
    ///
    /// Asserted on the ORDER and on the outcome of the first flag, not on a formatted string:
    /// the wording belongs to the table, and re-asserting it here would be the second
    /// authority `check_legality`'s own doc refuses to become.
    #[test]
    fn a_qwen_mtp_invocation_is_refused_by_the_mode_cell_first() {
        let Ok(a) =
            <Args as clap::Parser>::try_parse_from(["rivoli", "DIR", "--bench", "1", "--mtp"])
        else {
            panic!("`rivoli DIR --bench 1 --mtp` must parse — the invocation contract has moved");
        };
        let flags = requested_flags(&a, &Explicit::from(["mtp".to_string()]));
        assert!(
            matches!(flags.first(), Some(Flag::Mode(_))),
            "--mode must be the first flag judged, or the refusal a --mtp run prints is not \
             the one this test and the smoke script reason about; got {flags:?}"
        );
        assert!(
            flags.contains(&Flag::Mtp),
            "--mtp was typed and must be among the requested flags: {flags:?}"
        );
        let first = *flags.first().expect("at least --mode");
        assert_eq!(
            decide(Arch::QwenFlashNext, first),
            Outcome::Refuse(QWEN_ARM_NOT_BUILT),
            "the FIRST cell judged on the armless architecture is what the user reads"
        );
    }

    /// **The arms `main` dispatches to and the table's refusals must name the same set.**
    ///
    /// `main` sniffs, runs `check_legality`, and only then dispatches on the architecture.
    /// The contract: an architecture with NO arm must have `--mode` or `--attn` refused (that
    /// is what keeps its bail unreachable), and an architecture WITH an arm must refuse
    /// neither, which is `rivoli_core::legality`'s
    /// `every_architecture_with_an_arm_decodes_with_no_flags_typed`. [`ARMS`] below is what
    /// ties the two lists together.
    ///
    /// > **CORRECTED 2026-08-31 (S1).** This said the "no decode path" bail "is gone" since
    /// > M9, every variant having a real arm — which left the `has_arm == false` branch of the
    /// > assertion below dead for four architectures running. `Arch::QwenFlashNext` restores
    /// > it: the bail is back in `main`, and this test's no-arm branch is EXERCISED for the
    /// > first time — the difference between a checked biconditional and half of one.
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
                "{}: main {} decode arm, but the table {} the flags a bare invocation \
                 asks for — one of the two is wrong, and the visible symptom is either a \
                 model that cannot be started or `main`'s \"should have refused at the \
                 door\" bail becoming reachable",
                arch.name(),
                // "has a" / "has no", because this reads as a sentence for the first time at
                // S1: the false branch had never fired (every architecture had an arm), and it
                // said "main has no a decode arm" when the qwen red proof finally fired it.
                if has_arm { "has a" } else { "has no" },
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
