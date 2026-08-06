//! rivoli — GLM-5.2 MoE decode engine (routed experts int3-vq/int4/hybrid; the default is
//! hybrid — see docs/reference/modes.md and `config::Mode`). The artifact IS the model: point
//! `rivoli` at a converted artifact directory (manifest.json + codebooks.f32 +
//! resident.safetensors + `L{ll}.vq3` + tokenizer) and it decodes on device.
//!
//! Zero-knob by design — the machine is auto-discovered (see config.rs); the flags
//! below are benchmark/diagnostic overrides, not required configuration.

use anyhow::{Result, bail};

// The decode path's imports and helpers. A featureless build compiles this file down to
// the refusal stub in `main` (see there for why the binary still builds at all), so
// everything only the real `main` reaches is gated with it rather than left to warn.
#[cfg(feature = "rocm")]
use anyhow::Context;
#[cfg(feature = "rocm")]
use rivoli::artifact::config::Config;
#[cfg(feature = "rocm")]
use tracing::info;

// NOTE: doc comments on this struct and its fields are USER-FACING — clap renders them as
// `--help`. Rationale for the code goes in `//` comments like this one, which clap ignores.
//
// `clap` derives the parse, the help text and the range checks from the struct. The
// hand-rolled loop it replaced was ~150 lines of
// `args.next().context(..)?.parse().context(..)?`, one block per flag, plus a `USAGE`
// const maintained separately from the flags it described — and by the end no longer
// matching them (it never mentioned `--misa-heads`, `--checksum-x`, or the bounds of the
// since-deleted `--2q-*` pair).
//
// No `wrap_help` feature, so clap does not re-wrap: the terminal does. That also means
// `max_term_width` would be inert, which is why it is not set.
/// Zero-knob by design: every flag is a benchmark or diagnostic override, and a bare
/// `rivoli <model-dir>` is a complete invocation.
#[derive(clap::Parser, Debug)]
// `version` reports CARGO_PKG_VERSION. It is here because tagged builds are now shipped as
// tarballs (.github/workflows/release.yml), and a binary that cannot say which one it is
// turns every "which build is this?" into a checksum hunt. The release workflow refuses to
// run unless the tag and Cargo.toml agree, so this number is the tag. The commit and
// toolchain that produced it are in BUILD-INFO.txt beside the binary.
#[command(
    name = "rivoli",
    version,
    // Architecture-neutral since the engine gained a second one: naming GLM here made the
    // bare help contradict `rivoli <v4-artifact> --help` two lines later.
    about = "MoE decode engine (HIP/ROCm). Architectures: glm-moe-dsa, deepseek-v4.\n\
             Flags marked ARCHITECTURE-DEPENDENT resolve against the artifact: run\n\
             `rivoli <artifact> --help` for the ones that model actually admits."
)]
struct Args {
    /// The converted artifact directory (manifest.json + codebooks + resident.safetensors
    /// + L{ll}.vq3 + tokenizer). The artifact IS the model.
    model: String,

    /// Decode this many tokens, print PROFILE, exit. Omit for the server path (`--port`).
    ///
    /// Spelled `-bench` in every recorded command line (docs/measurement/benchmarks.md, docs/reference/modes.md,
    /// tests/bench-matrix.sh); `main` rewrites that single-dash form to `--bench` before
    /// clap sees it, since clap has no single-dash-long concept. Both work.
    #[arg(long)]
    bench: Option<usize>,

    /// Serve an OpenAI-compatible HTTP API on 127.0.0.1:PORT until killed — this is how
    /// llama-swap (and any OpenAI client) calls the engine. `POST /v1/chat/completions`
    /// with or without `stream`, plus `GET /health` and `GET /v1/models`.
    ///
    /// The port opens only once the model is loaded, so it doubles as the readiness
    /// signal. Sampling is NOT implemented — `temperature`/`top_p` are accepted and
    /// ignored, because the engine decodes greedy argmax. See src/serve.rs.
    #[arg(long, value_name = "PORT", conflicts_with = "bench")]
    port: Option<u16>,

    /// `--port`: the context window, in tokens. The KV cache is allocated ONCE at startup
    /// (there is no paging here), so this is a hard per-request ceiling — a conversation
    /// that does not fit is refused with a 400, not silently truncated. Costs ~51 KB of
    /// device memory per token, on top of `--max-mem`'s expert pool.
    #[arg(long, value_name = "N", default_value_t = 8192, value_parser = clap::value_parser!(usize))]
    ctx: usize,

    /// `--port`: reason before answering, unless the request says otherwise.
    ///
    /// The checkpoint is a thinking model — its template ends the prompt at an OPEN
    /// `<think>` by default, and the model fills it before answering. rivoli defaults the
    /// other way: at ~2.7 tok/s a reasoning block is tens of seconds of silence, and most
    /// OpenAI clients have no way to turn it off once it is on. A request's
    /// `enable_thinking` (or `reasoning_effort`) overrides this in either direction, and
    /// the reasoning comes back in `reasoning_content`, never mixed into `content`.
    #[arg(long)]
    think: bool,

    /// Routed-expert format. Default: hybrid — see docs/reference/modes.md and
    /// `config::Mode`.
    #[arg(long, default_value_t, value_parser = rivoli::artifact::config::Mode::parse)]
    mode: rivoli::artifact::config::Mode,

    /// Routed-expert cache policy. All three are output-neutral: routing never consults
    /// residency (see math.rs `inv_1_routing_never_consults_the_cache`), so a policy change
    /// moves throughput and hit rate, never the tokens produced.
    #[arg(long, default_value = "2q", value_parser = ["lru", "2q", "arc"])]
    cache_policy: String,

    /// Device budget in GiB, taken LITERALLY — no OS reserve, so it may OOM at build.
    /// Without it the budget auto-sizes to `free − 16 GiB`.
    #[arg(long, value_name = "GIB", value_parser = clap::value_parser!(u64).range(1..))]
    max_mem: Option<u64>,

    /// Attention row selection. `auto` picks `dsa` when the artifact carries indexer
    /// weights AND the backend has the indexer kernels, else `dense` — see `resolve_attn`.
    /// ARCHITECTURE-DEPENDENT: run `rivoli <artifact> --help` for the values this model
    /// admits. Some architectures fix attention in the weights and do not take this flag.
    #[arg(long, default_value = "auto", value_parser = ["auto", "dense", "streaming", "dsa", "misa"])]
    attn: String,

    /// `--attn streaming`: the number of leading sink tokens kept.
    #[arg(long, value_name = "N", default_value_t = 4)]
    sinks: usize,

    /// `--attn streaming`: the trailing window kept.
    #[arg(long, value_name = "N", default_value_t = 8192)]
    window: usize,

    /// `--attn misa`: how many indexer heads score tokens (the MISA paper's validated GLM
    /// setting is 8).
    #[arg(long, value_name = "N", default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..))]
    misa_heads: u16,

    /// Dump the routed-expert access trace (v2: demand keys plus the ranked candidate
    /// window) for the offline `replay` sim.
    #[arg(long, value_name = "PATH")]
    trace: Option<String>,

    /// Override the fixed bench prompt, for capturing traces of diverse inputs.
    #[arg(long, value_name = "TEXT")]
    prompt: Option<String>,

    /// Score this file TEACHER-FORCED and write one NLL per predicted token to --ppl-out
    /// instead of generating. The quality instrument for any change that CAN move output
    /// (a format or a kernel); never a free-running decode. Cache changes no longer need it
    /// — routing is residency-blind, so they are output-neutral by construction.
    #[cfg(feature = "teacher-forcing")]
    #[arg(long, value_name = "TEXT_FILE", requires = "ppl_out")]
    ppl: Option<String>,

    /// Where `--ppl` writes its per-token NLLs.
    #[cfg(feature = "teacher-forcing")]
    #[arg(long, value_name = "PATH")]
    ppl_out: Option<String>,

    /// Measure whether a layer's experts can be predicted BEFORE its attention runs — the
    /// feasibility question under cross-layer prefetch. Reports recall against the top-k and
    /// against the MISSES (the only reads a prefetch could save), plus what it would spend.
    ///
    /// Costs an rmsnorm, a gemv and a blocking D2H per MoE layer, so it measures RECALL and
    /// nothing else — do not read a tok/s off a probe run. Pair it with `--no-mtp`: with
    /// speculation on, the union carries two routers' picks and a row-0 prediction is scored
    /// against a denominator it never saw. Answer: docs/investigations/cross-layer-prefetch.md, "Feasibility,
    /// settled".
    #[cfg(feature = "pred-probe")]
    #[arg(long)]
    pred_probe: bool,

    /// Seconds without a token before the wedge watchdog aborts the process
    /// (`--features trace`). Must comfortably exceed the slowest HEALTHY token — a
    /// cold-miss token here is 1-2 s — so it trips only on a real wedge.
    #[cfg(feature = "trace")]
    #[arg(long, value_name = "SECS", default_value_t = 60)]
    watchdog_secs: u64,

    /// Record the decode as real OTLP spans, up to BUDGET of them (bare `--spans` = 5000).
    /// Without it only the end-of-run metrics export; there is no timeline to build.
    ///
    /// The budget is spent by sampling WHOLE tokens spread across the run, so what you get
    /// is representative rather than the cold start. Cost is a `Vec` push behind a mutex —
    /// +0.15% on wall fully instrumented, +0.00% at the default stride — but leave it off
    /// in a benchmark arm regardless: "too small to measure" is not "zero".
    ///
    /// Was the `RIVOLI_SPANS` env var until 2026-08-01. An env var is invisible to `--help`,
    /// absent from the command line `docs/measurement/benchmarks.md` records, and silently
    /// active in a build that looks stock — this is the same rule `--ppl` and `--pred-probe`
    /// follow. See `docs/measurement/traces.md`.
    #[cfg(feature = "otlp")]
    #[arg(long, value_name = "BUDGET", num_args = 0..=1, default_missing_value = "5000")]
    spans: Option<usize>,

    /// Scale the whole MoE branch by `g` before the residual add. An EXPERIMENT knob, not
    /// a tuning parameter (kernels/fwd.hip::vaxpy). The band is generous but finite: a
    /// sweep that silently ran at 0 or a negative gain would produce a confidently
    /// degenerate arm. g = 1.0 is bit-identical to the plain vadd.
    #[arg(long, value_name = "G", default_value_t = 1.0, value_parser = moe_gain_in_band)]
    moe_gain: f32,

    /// Write the generated token ids, one per line.
    ///
    /// Written for **gate A** of the Vulkan acceptance gate (`archive/vulkan-backend-hb16`,
    /// docs/investigations/vulkan-kernels.md): agreement on token IDs for K tokens. That
    /// backend is retired, but the instrument outlived it and the argument is general —
    /// comparing decoded TEXT is not a substitute, because different id sequences can decode
    /// to identical text, so a text diff reports only a lower bound on divergence. Any
    /// two-arm comparison across a refactor wants this rather than an eyeball.
    #[arg(long, value_name = "PATH")]
    dump_ids: Option<String>,

    /// Turn OFF speculative decode. On by default whenever the artifact carries the MTP
    /// head: the head drafts `pos+1`, a 2-row verify pass checks the real token and the
    /// draft through one read of every weight, and an accepted draft costs no second pass.
    ///
    /// Output is BYTE-IDENTICAL either way — every batched kernel is bit-identical per row
    /// and row 0 of a verify pass is the real token — so this flag can only move speed.
    /// (The exception is `--mode hybrid`, whose output is not stable under ANY cache
    /// change, speculation included; see docs/reference/architecture.md §8b under INV-1.)
    ///
    /// **Measured 1.108x** — 2.97 vs 2.68 tok/s, int3-vq, 512 tokens. That is WITH the
    /// `--mtp-min-conf` gate, which is on by default. The verify pass costs ~1.53x a
    /// sequential one (the MoE launches the UNION of both rows' experts), so it needs ~53%
    /// acceptance to break even and ungated it lands at 0.93-0.95x. Gating is what turns it
    /// positive. See docs/measurement/benchmarks.md, "The MTP confidence gate".
    #[arg(long)]
    no_mtp: bool,

    /// Speculate only when the draft head's own confidence clears this. Below it the pass
    /// runs one row and the draft is scored for free against the plain result, so the
    /// histogram keeps filling and the gate never goes blind to the bins it skips.
    ///
    /// Default 0.8 from the measured calibration, which is PROMPT-INVARIANT: two unrelated
    /// 512-token prompts both put the ≥0.8 bin at 91% and the 0.6–0.8 bin at 57%, while the
    /// MASS moved (25% vs 52% of drafts in the top bin). Against a ~1.53x verify pass the
    /// break-even is ~53%, so 0.8 clears it with margin and 0.6 would not. 0 disables the
    /// gate and speculates on every draft (the pre-gate behaviour).
    #[arg(long, default_value_t = 0.8)]
    mtp_min_conf: f32,


    /// DIAGNOSTIC: hash the residual stream after every layer.
    #[cfg(feature = "trace")]
    #[arg(long)]
    checksum_x: bool,
}

/// `--moe-gain`'s band. A hand-written parser because clap's `range` value_parser is
/// integers only, and an unchecked float here is the failure mode the band exists for.
fn moe_gain_in_band(s: &str) -> Result<f32, String> {
    let g: f32 = s
        .parse()
        .map_err(|e| format!("{s:?} is not a float: {e}"))?;
    if (0.5..=1.5).contains(&g) {
        Ok(g)
    } else {
        Err(format!("{g} is outside [0.5, 1.5]"))
    }
}

/// Above this token budget, `-bench` arms [`BENCH_SCRIPT`] so EOS continues the
/// conversation instead of ending the run. Below it the single default prompt still fits,
/// and every historical `-bench 128/256/512` keeps its exact recorded behaviour — the
/// default prompt stops on EOS at ~318 tokens, which is why 512 was the largest budget
/// anyone recorded.
const BENCH_SCRIPT_MIN: usize = 512;

/// The opening turn of a scripted run, replacing the historical `-bench` prompt above
/// [`BENCH_SCRIPT_MIN`]. `--prompt` still overrides it.
///
/// **Why not keep "The sky is blue because".** That prompt answers in ~318 tokens and then
/// stops, which is the shape a two-line factual question has; every turn of a scripted run
/// inherits that shape and the run ends up dominated by turn *boundaries* rather than by
/// decoding. It is also a topic pivot — the follow-ups are about inference engines — and a
/// conversation that changes subject at turn 2 is neither realistic nor good for the thing
/// these runs measure, since incoherent context is what pushes this model toward the
/// repetition the degeneration detector fires on.
///
/// So: an open-ended engineering request of the kind a real user actually sends. It asks
/// for reasoning and for a bottleneck analysis, both of which are naturally long, and it
/// frames the subject the thirteen follow-ups then drill into.
const BENCH_SCRIPT_OPEN: &str = "I'm building a decode engine for a mixture-of-experts \
     language model whose expert weights are far too large to fit in GPU memory, so most of \
     them have to stream from an NVMe drive while the resident ones compute. Walk me \
     through the architecture you would recommend end to end, explain the reasoning behind \
     each choice, and be specific about where you expect the bottlenecks to be.";

/// The historical `-bench` prompt, kept for budgets at or below [`BENCH_SCRIPT_MIN`] so
/// every `-bench 128/256/512` in `docs/measurement/benchmarks.md` stays comparable. Those
/// command lines do not record the prompt, so changing it for them would silently
/// invalidate a year of recorded numbers with nothing to point at.
const BENCH_PROMPT_LEGACY: &str = "The sky is blue because";

/// The scripted follow-up turns, fed one per EOS. Thirteen turns on ONE subject, because
/// a determinism or long-context run needs several thousand tokens of *coherent* output:
/// `docs/00-orientation/TOUR.md` is explicit that a degenerate run looks fastest, and a
/// repetition loop would also make the arena's access pattern unrepresentative.
///
/// Shape: seven turns drilling into the thread [`BENCH_SCRIPT_OPEN`] opens, an eighth
/// asking for a full report (the longest single answer), then five rewrites of that report
/// for five different audiences — each forced to restate the same material differently
/// rather than continue it, which keeps output long without letting the model settle into
/// a loop.
///
/// **Every turn asks for reasoning, trade-offs or examples on purpose.** A question that
/// can be answered in two sentences makes a scripted run measure turn boundaries instead
/// of decoding; these are written to elicit several hundred tokens each, which is also what
/// a real user's turn looks like.
///
/// **These are a benchmark input and are frozen from the commit that adds them.** Editing
/// one changes the token sequence, the routing, and hence the hit rate and every number a
/// `-bench` above [`BENCH_SCRIPT_MIN`] produces, and the command line does not record the
/// prompt. Add a turn rather than reword one.
const BENCH_SCRIPT: [&str; 13] = [
    "Go deeper on the routing itself: how does the gate pick experts for a token, why \
     top-k rather than a softmax over all of them, and what goes wrong when the routing is \
     unbalanced across experts?",
    "Now the streaming path. Walk through everything that happens between the router \
     naming an expert and that expert's weights being ready to compute on, and explain \
     which parts of that can be overlapped with useful work and which cannot.",
    "Compare the cache policies you would consider for keeping the most useful experts \
     resident — LRU, 2Q, ARC, and anything else you think is worth considering — and \
     explain the trade-offs with concrete examples of workloads where each one wins.",
    "How would you measure whether that cache is doing its job? Name the specific metrics, \
     explain what each one can and cannot tell you, and describe the ways each one can \
     mislead someone who trusts it too much.",
    "Move on to quantization. Compare 3-bit vector quantization against 4-bit scalar \
     quantization for these expert weights, covering accuracy, decode cost, memory \
     footprint and implementation complexity, and say which you would pick and why.",
    "Explain the risks of evaluating a quantized model with free-running generation \
     instead of a fixed scored corpus, and then describe an evaluation protocol you would \
     actually trust, including how you would know it had enough statistical power.",
    "Explain speculative decoding with a draft head: the mechanism, the arithmetic that \
     decides whether it pays for itself, and the conditions under which it stops paying.",
    "Now write a full report covering everything we have discussed — routing, streaming, \
     caching, measurement, quantization, evaluation methodology and speculative decoding. \
     Structure it properly, include the reasoning behind each recommendation, and call out \
     the open questions you would want answered before committing to the design.",
    "Rewrite that report for an executive audience: someone who has ten minutes, controls \
     the budget, and has no machine learning background. Keep every recommendation but \
     change how you justify it.",
    "Rewrite it for a systems engineer who will implement the storage and caching layer. \
     Assume they know operating systems, NVMe and io_uring well, and know nothing about \
     transformers.",
    "Rewrite it as onboarding documentation for a new engineer who will maintain this \
     system, assuming they need to make changes safely before they understand the whole \
     thing.",
    "Rewrite it as a design review that argues against this approach as strongly as it \
     can, so the team can test its own assumptions. Steel-man the alternatives rather than \
     dismissing them.",
    "Rewrite it as a postmortem written a year after adoption, assuming the system shipped \
     and ran into exactly the problems you flagged, and include what the team should have \
     done differently.",
];

/// Parse argv, accepting the legacy single-dash `-bench` alongside `--bench`.
///
/// clap has no single-dash-long form, and `-bench N` is what every recorded command line
/// in docs/measurement/benchmarks.md, docs/reference/modes.md and tests/bench-matrix.sh uses. Rewriting the token is three
/// lines; re-recording a year of benchmark provenance is not. Only an exact `-bench`
/// matches, so `-b`, `--bench` and a positional path are all untouched.
fn parse_args() -> Args {
    use clap::Parser;
    let argv: Vec<String> = std::env::args()
        .map(|a| if a == "-bench" { "--bench".into() } else { a })
        .collect();
    // `--help` resolves against the artifact when one is named, BEFORE clap handles it.
    if artifact_resolved_help(&argv) {
        std::process::exit(0);
    }
    Args::parse_from(argv)
}

/// Render `--help` against the architecture of the artifact on the command line, hiding
/// flags that architecture has no use for. Returns `true` if help was printed.
///
/// The motivating case: `--attn`, `--sinks`, `--window` and `--misa-heads` are GLM
/// row-selection knobs, and on DeepSeek-V4 attention is fixed by the weights — so a V4
/// user reading generic help spends it on four flags that cannot do anything, and learns
/// nothing about the one thing that differs between the two models. Help is the only
/// output nothing else asserts on, which is why `arch.rs` pins its lists to the parser.
///
/// Degrades to clap's own help whenever it cannot do better: no artifact named, not a
/// directory, no manifest, or an architecture string it does not recognise. An
/// unrecognised architecture is NOT an error here — `main` refuses it a few lines later
/// with a better message than a help renderer can give.
fn artifact_resolved_help(argv: &[String]) -> bool {
    let all = argv.iter().any(|a| a == "--help-all");
    if !all && !argv.iter().any(|a| a == "--help" || a == "-h") {
        return false;
    }
    // The first argument that is a directory holding a config document. Scanning for that
    // rather than for "the first non-flag" keeps this honest around flag VALUES: a
    // `--ppl-out /some/dir` would fool a positional-counting parse, and getting it wrong
    // silently renders help for the wrong model.
    //
    // Resolution itself is `artifact::model::arch_of_artifact`, NOT re-derived here. This
    // read the manifest and both key spellings inline until a review caught it: a second
    // reader is a second thing to keep in step, and `arch.rs`'s header states the rule it
    // was breaking — there is exactly one architecture discriminant in the tree and nobody
    // re-derives it. jscpd could not see it because the two spellings differed.
    let dir = argv[1..].iter().find(|a| {
        ["manifest.json", "config.json"]
            .iter()
            .any(|f| std::path::Path::new(a).join(f).is_file())
    });
    let Some(arch) = dir.and_then(|d| rivoli::artifact::model::arch_of_artifact(d).ok()) else {
        return false; // clap's generic help, which is the right answer when we know nothing
    };

    use clap::CommandFactory;
    let mut cmd = Args::command().about(format!(
        "rivoli — {} decode engine (HIP/ROCm)\nArchitecture: {} — {}",
        arch.name(),
        arch.name(),
        arch.summary()
    ));
    let mut notes: Vec<String> = Vec::new();
    if !all {
        for id in arch.hidden_flags() {
            cmd = cmd.mut_arg(*id, |a| a.hide(true));
        }
        if !arch.hidden_flags().is_empty() {
            notes.push(format!(
                "{} flag(s) that do not apply to {} are hidden — show them with --help-all.",
                arch.hidden_flags().len(),
                arch.name()
            ));
        }
    }
    match arch.attn_modes() {
        Some(modes) => cmd = cmd.mut_arg("attn", |a| a.value_parser(modes.to_vec())),
        // Say why the flag is gone. "Absent" and "inapplicable" read identically otherwise,
        // and the first invites a bug report.
        None => notes.push(format!("--attn: {}", arch.attn_fixed_note())),
    }
    if !notes.is_empty() {
        cmd = cmd.after_help(notes.join("\n"));
    }
    // `.ok()` rather than `expect`: `rivoli <artifact> --help | head` closes the pipe early,
    // and a help renderer that panics on EPIPE turns a normal shell idiom into a crash.
    // There is nothing to recover *to* here — the next statement exits.
    cmd.print_help().ok();
    true
}

#[cfg(feature = "rocm")]
/// Does the artifact carry resident DSA indexer weights? (`auto` picks `dsa` iff so AND the
/// backend has the indexer kernels — see [`resolve_attn`].)
/// Checks layer 0's `wk` — `indexer_layout` guarantees layer 0 is a full layer, so
/// its indexer is present whenever any is. `open_dir` merges every *.safetensors in
/// the artifact, so this sees `indexer.safetensors` (added post-hoc) too.
fn artifact_has_indexer(model_dir: &str) -> bool {
    rivoli::artifact::format::Safetensors::open_dir(model_dir)
        .map(|st| st.has("model.layers.0.self_attn.indexer.wk.weight"))
        .unwrap_or(false)
}

/// Resolve `--attn` (with `auto` → dsa/dense by artifact contents) into an `AttnMode`.
///
/// `auto` used to ALSO ask what the backend could run: the Vulkan build had none of the five
/// DSA indexer kernels, so `auto` resolving to `dsa` on an artifact carrying indexer weights
/// made a bare `rivoli <model>` fail on its own default. That backend was retired 2026-08-06
/// and the only remaining one has every kernel, so the question is now purely about the
/// artifact.
///
/// The reason for the choice is still logged: "auto picked dense" is otherwise
/// indistinguishable from "this artifact has no indexer".
#[cfg(feature = "rocm")]
fn resolve_attn(a: &Args) -> Result<rivoli::attn::AttnMode> {
    use rivoli::attn::AttnMode;
    let mode = if a.attn == "auto" {
        if artifact_has_indexer(&a.model) {
            "dsa"
        } else {
            "dense"
        }
    } else {
        a.attn.as_str()
    };
    Ok(match mode {
        "dense" => AttnMode::Dense,
        "streaming" => AttnMode::Streaming {
            sinks: a.sinks,
            window: a.window,
        },
        "dsa" => AttnMode::Dsa,
        "misa" => AttnMode::Misa {
            active_heads: a.misa_heads as usize,
        },
        other => bail!("unknown --attn mode {other:?} (auto|dense|streaming|dsa|misa)"),
    })
}

/// A build with NO compute backend refuses at the door.
///
/// The binary still compiles featureless on purpose — `cargo test` and the featureless
/// clippy pass build every target, and `backend.rs` records why breaking those is not an
/// option: they are what keeps the backend-independent half (config, math, quant, arena,
/// cache, telemetry) honest. So this is a runtime refusal rather than a `compile_error!`,
/// and it is the FIRST thing that happens: the old version discovered memory, loaded the
/// manifest, built the tokenizer and encoded the prompt before admitting it could not
/// decode.
///
/// `parse_args` still runs, so `--help` and flag validation work in a featureless build —
/// which is the one useful thing it can do.
#[cfg(not(feature = "rocm"))]
fn main() -> Result<()> {
    let _ = parse_args();
    bail!(
        "rivoli was built with NO compute backend and cannot decode. Rebuild with \
         `--features rocm` (HIP/ROCm), the only backend (src/backend.rs)."
    )
}

/// The device bytes a pin may use.
///
/// Extracted so the GLM and V4 branches cannot drift: an over-count here comes straight out of
/// the routed pool's share, and a budget that differed between the two architectures would look
/// like a residency difference in every log line.
///
/// `--max-mem` is honoured LITERALLY — no OS reserve. The user asked for that size, so it is
/// allowed to OOM at pin build; the auto path just leaves `OS_RESERVE` free (there is no hard
/// footprint ceiling — the old NaN cliff was our own bug, since fixed).
#[cfg(feature = "rocm")]
fn device_budget(max_mem: Option<u64>) -> Result<usize> {
    use rivoli::artifact::config::OS_RESERVE;
    const GIB: f64 = (1u64 << 30) as f64;
    let (free, _total) = rivoli::memory::device::mem_info()?;
    Ok(match max_mem {
        Some(m) => {
            info!(
                "device pool budget {:.1} GiB (--max-mem, literal — no reserve; may OOM)",
                m as f64 / GIB
            );
            m as usize
        }
        None => {
            let cap = free.saturating_sub(OS_RESERVE as usize);
            info!(
                "device pool budget {:.1} GiB (auto: free {:.1} GiB − {:.0} GiB OS reserve)",
                cap as f64 / GIB,
                free as f64 / GIB,
                OS_RESERVE as f64 / GIB,
            );
            cap
        }
    })
}


/// The DeepSeek-V4-Flash decode, which shares no engine type with GLM's.
///
/// A separate function rather than a `match` arm inside `main` because it forks BEFORE
/// `ModelConfig::load` — every line of `main` after that point is `&ModelConfig`-typed, and
/// `ModelConfig::load` refuses a V4 manifest with a message about the two architectures not sharing
/// a decode path. That message is still correct and still reachable (`parse_config` is generic over
/// both schemas, so any OTHER caller handed a V4 manifest gets it); what changed is that the
/// dispatch now happens before `main` can reach it, so a user never sees it for the one case where
/// it is no longer true.
///
/// **Flags this path does not have, refused rather than ignored.** `arch.rs` already carries the
/// help policy — `Arch::DeepseekV4.hidden_flags()` hides `attn`/`sinks`/`window`/`misa_heads`,
/// and `attn_modes()` returns `None` because V4's attention is fixed by the weights — so `--help`
/// does not offer them. `--help` hiding a flag is not the same as the parser rejecting it, which
/// is why the explicit forms are refused below. `arch.rs`'s own header names `resolve_attn` as a
/// legitimate refusal site and `validate_backend` as not one, so this is the third: the branch
/// that knows both the architecture and the flags.
/// `attn`/`port`/`no_mtp` are passed individually rather than as `&Args`: `Config` takes
/// ownership of four of `Args`' `String`s just above the dispatch, so `&a` is not borrowable
/// there. The three are differently typed, so none is substitutable for another.
#[cfg(feature = "rocm")]
fn run_v4(cfg: &Config, attn: &str, port: Option<u16>, no_mtp: bool, watchdog_secs: u64) -> Result<()> {
    use rivoli::arch::Arch;
    // Inside the function rather than at the top of the file: it then inherits this `cfg`
    // instead of restating it, and a gate that cannot be restated cannot drift out of step.
    use rivoli::artifact::dsv4_encoding::{EncodeOpts, Message, ThinkingMode};
    let arch = Arch::DeepseekV4;
    let v4: rivoli::artifact::model::V4Config = rivoli::artifact::model::load_config(&cfg.model)?;
    info!(
        "model: {} ({}) — {} layers, hidden {}, {} heads x head_dim {}, {} experts top{}, \
         window {}, hc_mult {}, vocab {}",
        arch.name(),
        arch.summary(),
        v4.n_layers,
        v4.hidden,
        v4.n_heads,
        v4.head_dim,
        v4.n_experts,
        v4.top_k,
        v4.sliding_window,
        v4.hc_mult,
        v4.vocab
    );
    // Every flag this branch cannot honour, refused with the reason. `--attn` is compared
    // against the string rather than against the resolved `AttnMode`, because `resolve_attn`
    // already turned `auto` into `dense` — a value that is meaningless here rather than wrong,
    // and refusing it would refuse the default.
    if attn != "auto" {
        bail!(
            "--attn does not apply to a {} artifact: {}. It is hidden from this artifact's \
             --help for the same reason (arch.rs::attn_modes returns None).",
            arch.name(),
            arch.attn_fixed_note()
        );
    }
    if cfg.mode != rivoli::artifact::config::Mode::default() {
        bail!(
            "--mode {} selects a GLM routed-expert format. A {} artifact carries `.f4` \
             (fp4 e2m1 with e8m0 group scales) and there is no second format to pick.",
            cfg.mode,
            arch.name()
        );
    }
    if port.is_some() {
        bail!(
            "--port is not wired for {} yet: `serve::serve` takes a `&mut GpuEngine`, which is \
             GLM-typed. Nothing here is missing a kernel; it is a signature.",
            arch.name()
        );
    }
    // Speculative decode is OFF and cannot be turned on — say so once, and only when the
    // user did not already ask for it off. (This mirrored a Vulkan single-row downgrade
    // message that was deleted with that backend on 2026-08-06; the V4 reason below is
    // unrelated and still live.)
    if !no_mtp {
        info!(
            "speculative decode OFF for {}: the fp4 MoE kernel refuses nrow != 1 \
             (kernels/moe.hip:409, guard 1003 — only R = 1 is instantiated, and the 1.108x that \
             justifies R = 2 for GLM's VQ/int4 kernels has no V4 measurement behind it), so a V4 \
             decode is structurally single-row and a batched verify pass has no kernel.",
            arch.name()
        );
    }
    let ngen = cfg
        .bench
        .context("nothing to do: pass -bench <tokens> to decode")?;
    // NOTE for anyone comparing against a V4 figure recorded before 2026-08-06: with the chat
    // framing below, EOS is REACHABLE for the first time, so `-bench N` may now return fewer
    // than N tokens where it always ran to the budget. tok/s divides by the tokens actually
    // produced, so the rate stays comparable — the token count does not, and neither does the
    // text, which was measured off-template.

    let tok = rivoli::artifact::tokenizer::Tokenizer::load(&cfg.model)?;
    let bench_prompt = cfg.prompt.as_deref().unwrap_or("The sky is blue because");
    // CHAT-FRAMED. This was raw until 2026-08-06 and that was the whole of the "V4 repeats
    // itself and never stops" defect: raw text puts the model outside any assistant turn, where
    // it is doing document continuation and has no reason to emit `<｜end▁of▁sentence｜>` — so
    // every decode ran to `-bench` and the first one emitted "The sky is blue because of
    // Rayleigh scattering." three times over. EOS handling was never wrong. (Not
    // `encode_chat_turns`: that is GLM's Jinja template, whose literals this tokenizer lacks.
    // This checkpoint ships no template at all — `artifact::dsv4_encoding` ports the Python
    // its README points at instead.)
    //
    // `Chat`, not `Thinking`: the prompt ends `<｜Assistant｜></think>`, closing the reasoning
    // block before it opens so the model answers immediately. `Thinking` would end at an open
    // `<think>` and spend the whole `-bench` budget reasoning — at V4 decode rates a benchmark
    // would measure deliberation, not answering. Thinking is a PREFILL, not a flag.
    let prompt_ids = tok.encode_dsv4(
        vec![Message::user(bench_prompt)],
        &EncodeOpts::new(ThinkingMode::Chat),
    )?;
    info!(
        "tokenizer: prompt {bench_prompt:?} -> {} tokens, DeepSeek-V4 chat framing \
         (<｜begin▁of▁sentence｜> … <｜User｜> … <｜Assistant｜></think>); eos={:?}",
        prompt_ids.len(),
        tok.eos
    );

    let cap = device_budget(cfg.max_mem)?;
    let t = std::time::Instant::now();
    let pin = rivoli::memory::pin::V4Pin::build(
        &cfg.model,
        &v4,
        cap,
        &cfg.cache_policy,
        cfg.two_q,
        cfg.trace.as_deref(),
    )?;
    info!("v4 pin built in {:.1}s", t.elapsed().as_secs_f64());
    let mut engine = rivoli::v4gpu::V4Engine::new(pin, v4, prompt_ids.len(), ngen)?;
    // Wedge watchdog, same as GLM's. A V4 layer streams 6 of 256 experts against GLM's 8 of 256
    // over half as many layers, so the shared default is if anything generous here.
    //
    // `trace`-only with the deadline as a FLAG, matching the GLM path since 2026-08-03. The
    // env-var helper this replaced argued a watchdog is exempt from "feature AND flag, never
    // an env var" because it produces no measurement — but an env var is still invisible to
    // `--help` and absent from the command line benchmarks.md records, which is the whole
    // objection. Two spellings of one deadline would also make "the watchdog fired" ambiguous.
    #[cfg(feature = "trace")]
    let hb = rivoli::watchdog::spawn(std::time::Duration::from_secs(watchdog_secs))?;
    #[cfg(not(feature = "trace"))]
    let hb = rivoli::watchdog::inert();
    let ids = engine.generate(&prompt_ids, ngen, &tok.eos, &mut |_: u32| {
        hb.beat();
        true
    })?;
    // The text is printed and NOT assessed. `distinct`/`longest repeated block` fire identically
    // on a repetition loop, on spliced corruption and on prose that restates a paragraph on
    // purpose (CLAUDE.md — they have misled three investigations here), and this loop has two
    // NAMED deviations from the reference (the unclamped shared expert, and positional block
    // selection on the ratio-4 layers). A degeneration verdict on top of that would be a number
    // standing in for a gate. `tests/v4_loop.rs` is the gate.
    // Prompt and reply LABELLED and on separate lines. Concatenating them was right while the
    // prompt was raw and the model was continuing a document; under chat framing `generate`
    // returns the assistant's answer alone, and an answer normally restates the question — so
    // `{bench_prompt}{reply}` printed "The sky is blue becauseThe sky is blue because of
    // Rayleigh scattering", which is exactly the repetition signature this branch's framing
    // comment says was fixed. This repo has burned three investigations on misread repetition
    // (CLAUDE.md); a bench log must not manufacture a fourth. Review finding, 2026-08-06.
    info!("prompt: {bench_prompt:?}");
    info!("reply : {:?}", tok.decode_all(&ids)?);
    Ok(())
}

/// A build with no compute backend cannot reach this: see the stub above.
#[cfg(feature = "rocm")]
fn main() -> Result<()> {
    let a = parse_args();
    let attn = resolve_attn(&a)?;
    // Arm the span recorder before anything can record: it anchors a monotonic clock to a
    // wall clock here, and every interval in the run is stamped as a delta from that pair.
    #[cfg(feature = "otlp")]
    if let Some(budget) = a.spans {
        rivoli::telemetry::spans::init(budget);
    }
    // `--checksum-x` only exists in a `trace` build; elsewhere the field is dead-false.
    #[cfg(feature = "trace")]
    let checksum_x = a.checksum_x;
    #[cfg(not(feature = "trace"))]
    let checksum_x = false;
    // The one run-identity attribute the OTLP root span cannot read back off `cfg`, since
    // `attn` is moved into it below and `AttnMode` has no Display.
    let a_attn = format!("{attn:?}").to_lowercase();
    rivoli::artifact::config::check_budget(a.max_mem)?;
    // The flags ARE the config. This was `Config::discover`, a nine-argument passthrough
    // (carrying an `#[allow(clippy::too_many_arguments)]`) whose only work was the budget
    // check above — after which `main` overwrote two of the fields it had just defaulted.
    // Moving the strings in rather than cloning them out is also what dissolved the block
    // of `let a_* = a.field.clone()` bindings that worked around the partial move: every
    // field read after this point is either `Copy` or lives on `cfg`.
    let cfg = Config {
        model: a.model,
        bench: a.bench,
        trace: a.trace,
        prompt: a.prompt,
        cache_policy: a.cache_policy,
        two_q: rivoli::memory::cache::TwoQSplit::default(),
        checksum_x,
        mode: a.mode,
        // GiB -> BYTES. The flag is documented and value_named in GiB and every consumer
        // (`Config::max_mem`, the pool cap, the log line's `/ GIB`) is in bytes. The clap
        // migration dropped this multiply, so `--max-mem 115` asked for 115 BYTES and the
        // pool came out at "0.0 GiB (~0 slots)" before failing in `rivoli_vmm_alloc(0)`.
        // Loud rather than silent, but it invalidates any run that passed the flag.
        max_mem: a.max_mem.map(|g| g.saturating_mul(1 << 30)),
        attn,
    };
    // There is no startup capability gate any more. `Config::validate_backend` refused
    // `--mode int4|hybrid` and `--attn dsa|misa` at the door on Vulkan builds, and a
    // `--moe-gain != 1` check beside it refused the `vaxpy` kernel that backend deferred.
    // Both went with the backend on 2026-08-06: `rocm` has every kernel, so every
    // configuration the CLI accepts is one this build can run, and a gate that can only
    // return `Ok(())` is the shape this file's own history has deleted before.

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let version = env!("CARGO_PKG_VERSION");
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
    // Rule 1: the full discovered config is the first line of every run.
    info!("rivoli {version} | {cfg}");

    if !std::path::Path::new(&cfg.model).is_dir() {
        bail!("model dir not found: {}", cfg.model);
    }

    // **Which architecture, before anything reads a dimension.** `ModelConfig::load` below
    // hard-assumes GLM's schema and every line after it is `&ModelConfig`-typed; handed a V4
    // manifest it refuses with a message about the two architectures not sharing a decode path,
    // which was true and is now the wrong message. There is deliberately no `--arch` flag —
    // `arch.rs`'s header states why: a flag naming the architecture is a flag that can disagree
    // with the weights, and disagreeing launches the wrong attention path and produces fluent
    // wrong text rather than crashing.
    match rivoli::artifact::model::arch_of_artifact(&cfg.model)? {
        rivoli::arch::Arch::GlmMoeDsa => {}
        rivoli::arch::Arch::DeepseekV4 => return run_v4(&cfg, &a.attn, a.port, a.no_mtp, {
            #[cfg(feature = "trace")] { a.watchdog_secs }
            #[cfg(not(feature = "trace"))] { 60 }
        }),
    }

    // Model dimensions from the artifact's manifest.json.
    let mc = rivoli::artifact::model::ModelConfig::load(&cfg.model)?;
    info!(
        "model: {} layers ({} dense) hidden={} heads={} experts={} top{} moe_inter={} vocab={}",
        mc.n_layers,
        mc.dense_layers,
        mc.hidden,
        mc.n_heads,
        mc.n_experts,
        mc.top_k,
        mc.moe_inter,
        mc.vocab
    );
    // Which tool produced the `.i4` set, and from what — only for the modes that
    // actually read it. int4/hybrid quality numbers are interpretable only against
    // this line; unstamped means an artifact predating provenance, where a
    // `vq3_to_i4` set and an `fp8_to_i4` set are indistinguishable on disk.
    // The group size is not decoration: the engine indexes `scale[o·ngroups + i/G]`,
    // so reading a set quantized at a different G walks a differently-shaped array.
    // That does not fault — it silently yields garbage (a per-row set read as G=128
    // produced rel_l2=NaN, gain=NaN in the ground-truth oracle), which is the failure
    // mode a stamp exists to prevent. Refuse rather than decode nonsense.
    if cfg.mode.uses_int4() {
        match rivoli::artifact::format::I4Source::load(&cfg.model)? {
            Some(s) => {
                info!(
                    "i4 source: {} ({}, group {}) layers {}..{} from {}",
                    s.tool,
                    s.chain,
                    s.group.map_or("per-row".to_string(), |g| g.to_string()),
                    s.layers[0],
                    s.layers[1],
                    s.src
                );
                let want = rivoli::artifact::quant::I4_GROUP;
                match s.group {
                    Some(g) if g == want => {}
                    Some(g) => anyhow::bail!(
                        "`.i4` set was quantized at group {g}, this binary reads group {want}. \
                         Rebuild with `fp8_to_i4`, or run a binary built for group {g}."
                    ),
                    None => anyhow::bail!(
                        "`.i4` set is per-row (pre-group provenance), this binary reads group \
                         {want}. Rebuild with `fp8_to_i4`."
                    ),
                }
            }
            // UNSTAMPED is not unknown, and refusing here made it look like it was.
            // `ExpertSet::open` computes `(n_experts + 1) * i4_expert_stride` and hard-fails
            // on a byte mismatch, and the stride is a function of the group size — so the
            // slab length ALREADY proves the group, exactly, before a single expert is
            // read. (The `.i4` set on the reference artifact is 5,153,882,112 B =
            // 257 x 20,054,016, which only group 128 produces; group 64 would be
            // 21,233,664 and per-row 18,915,328.) format.rs::I4Source says as much —
            // "such a set has a different `.i4` file size and is rejected by
            // `ExpertSet::open`, so this is a diagnosis, not a load-time guard" — and
            // bailing here contradicted its own doc comment, locking out an artifact whose
            // bytes are provably correct because a JSON field went missing.
            //
            // A stamp that POSITIVELY DISAGREES still bails above: that is a claim in
            // conflict with the binary, not an absence of one.
            None => info!(
                "i4 source: unstamped (no manifest `i4_source`) — group size will be \
                 verified by the `.i4` slab length against group {}",
                rivoli::artifact::quant::I4_GROUP
            ),
        }
    }
    info!(
        "mla: q_lora={} kv_lora={} qk={}+{} v_head={} rope_theta={}",
        mc.q_lora_rank,
        mc.kv_lora_rank,
        mc.qk_nope_head_dim,
        mc.qk_rope_head_dim,
        mc.v_head_dim,
        mc.rope_theta()
    );

    // Tokenizer (tokenizer.json + generation_config.json, copied into the artifact).
    // Read from `cfg.bench` rather than `ngen` (computed below) because the prompt has to
    // be framed before the tokenizer log; `--ppl`/`--port` set `ngen` to 0 and never use
    // this string, so keying off the raw flag cannot pick the wrong one for a run that
    // matters.
    //
    // `--prompt` DISARMS the script rather than keeping its turns. The follow-ups drill
    // into what `BENCH_SCRIPT_OPEN` asked ("Go deeper on the routing itself…"), so bolting
    // them onto an unrelated prompt produces a conversation that changes subject at turn 2
    // — the incoherence this script exists to avoid. A custom prompt gets the old
    // behaviour: decode until EOS or the budget.
    let scripted = cfg.prompt.is_none() && cfg.bench.is_some_and(|n| n > BENCH_SCRIPT_MIN);
    let bench_prompt = match scripted {
        true => BENCH_SCRIPT_OPEN,
        false => cfg.prompt.as_deref().unwrap_or(BENCH_PROMPT_LEGACY),
    };
    let tok = rivoli::artifact::tokenizer::Tokenizer::load(&cfg.model)?;
    // Chat-framed, always: raw text leaves the model outside any assistant turn, so its
    // EOS ids (two of which are turn boundaries) are unreachable and decode runs to the
    // token limit every time. `--raw-prompt` opted out of the framing to reproduce numbers
    // recorded before templating existed; it was deleted 2026-08-01, because the only runs
    // it can produce are runs that cannot stop. `--ppl` deliberately does NOT go through
    // here — it scores a fixed corpus, not a turn.
    let prompt_ids = tok.encode_chat(bench_prompt)?;
    info!(
        "tokenizer: prompt {bench_prompt:?} -> {} tokens chat-framed {:?}; eos={:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(12)],
        tok.eos
    );

    // `--ppl` scores a fixed text instead of generating, so `-bench` is meaningless there.
    // Defined in BOTH configurations rather than gating each of its three readers: without
    // the feature there is no corpus to load, so it is a `None` the sizing below folds away.
    #[cfg(feature = "teacher-forcing")]
    let ppl_ids = match &a.ppl {
        Some(path) => Some(rivoli::eval::load_corpus(path, &tok)?),
        None => None,
    };
    #[cfg(not(feature = "teacher-forcing"))]
    let ppl_ids: Option<Vec<u32>> = None;
    // Three shapes of run, and only `-bench` carries a token budget: `--ppl` scores a
    // corpus, and the server takes its budget per request (bounded by `--ctx`).
    let ngen = match (ppl_ids.is_some(), a.port.is_some()) {
        (true, _) | (_, true) => 0,
        _ => cfg
            .bench
            .context("nothing to do: pass -bench <tokens> to decode, or --port <PORT> to serve")?,
    };

    #[cfg(feature = "rocm")]
    {
        let cap = device_budget(cfg.max_mem)?;
        // dsa/misa need the resident DSA indexer placed by Pin::build.
        let want_indexer = matches!(
            cfg.attn,
            rivoli::attn::AttnMode::Dsa | rivoli::attn::AttnMode::Misa { .. }
        );
        let t = std::time::Instant::now();
        let pin = rivoli::memory::pin::Pin::build(
            &cfg.model,
            &mc,
            cap,
            cfg.trace.as_deref(),
            &cfg.cache_policy,
            cfg.two_q,
            want_indexer,
            cfg.mode,
        )?;
        info!("pin built in {:.1}s", t.elapsed().as_secs_f64());
        // `--ppl` walks the CORPUS, not the bench prompt, and its `ngen` is 0 — sizing the
        // KV cache from `prompt_ids` would allocate ~17 positions for an ~900-position
        // scoring pass. The server sizes from `--ctx`, since it cannot know its prompts
        // yet and the slabs are allocated exactly once. Take whichever run this actually is.
        // Encoded here rather than beside `generate` because the KV cache is sized below
        // and has to cover them: a follow-up's tokens occupy POSITIONS without ever
        // entering `generated`, so `pos` outruns the token budget. Sizing from `ngen`
        // alone aborted `-bench 1200` mid-run with "pos 1212 + 1 rows exceeds engine
        // capacity max_ctx=1212" — the 13 turns are ~400 positions the budget never saw.
        // Encoding up front also means a bad turn fails at startup rather than 3000 tokens
        // into a sole-tenant run.
        let followups: Vec<Vec<u32>> = match scripted {
            true => BENCH_SCRIPT
                .iter()
                .map(|t| tok.encode_chat_continuation(t))
                .collect::<Result<_>>()?,
            false => Vec::new(),
        };
        let followup_pos: usize = followups.iter().map(Vec::len).sum();
        if !followups.is_empty() {
            info!(
                "-bench {ngen} > {BENCH_SCRIPT_MIN}: scripted opening + {} follow-up turns \
                 armed, {followup_pos} extra KV positions (EOS continues the conversation)",
                followups.len()
            );
        }
        let max_ctx = match (&ppl_ids, a.port) {
            (Some(ids), _) => ids.len() + 1,
            (None, Some(_)) => a.ctx,
            (None, None) => prompt_ids.len() + ngen + followup_pos + 1,
        };
        let mut engine = rivoli::gpu::GpuEngine::new(
            pin,
            &mc,
            max_ctx,
            cfg.attn.clone(),
        )?;
        #[cfg(feature = "trace")]
        engine.set_checksum_x(cfg.checksum_x);
        #[cfg(feature = "pred-probe")]
        engine.set_pred_probe(a.pred_probe);
        engine.set_moe_gain(a.moe_gain);
        // Wedge watchdog: a hung GPU join can't be caught inside the decode loop, so a
        // background thread aborts the process if no token lands for `--watchdog-secs`.
        //
        // `trace`-only, and the deadline is a FLAG. It was `RIVOLI_WATCHDOG_SECS` until
        // 2026-08-03 — an env var is invisible to `--help`, absent from the command line
        // `docs/measurement/benchmarks.md` records, and silently active in a build that
        // looks stock, which is the same objection that retired `RIVOLI_SPANS` and
        // `RIVOLI_TOPK`. The only env vars this engine still reads are `OTEL_*`, whose
        // names are the OpenTelemetry spec's rather than ours, and `TMPDIR`.
        //
        // Cloned rather than moved: server mode has to beat it from the accept loop too,
        // because an idle server produces no tokens and the watchdog cannot tell "waiting
        // for a request" from "wedged". See serve::serve.
        #[cfg(feature = "trace")]
        let hb = rivoli::watchdog::spawn(std::time::Duration::from_secs(a.watchdog_secs))?;
        #[cfg(not(feature = "trace"))]
        let hb = rivoli::watchdog::inert();
        engine.set_heartbeat(hb.clone());
        // --- `--ppl`: teacher-forced quality gate, not a decode -----------------------
        // Returns before `generate` because the two are mutually exclusive by design: a
        // free-running generation's own trajectory is the confound this measurement exists
        // to remove. The whole flow lives in `rivoli::eval` so the feature gate is a module
        // boundary rather than four `#[cfg]`s that can drift out of step.
        #[cfg(feature = "teacher-forcing")]
        if let Some(ids) = ppl_ids {
            // `bin/ppl` labels a cell with `split_whitespace().take(3)`, so these three
            // fields ARE the label. `moe_gain` is among them because a gain sweep differs
            // in nothing else — six arms would otherwise be indistinguishable downstream.
            let label = format!(
                "mode={} policy={} moe_gain={}",
                cfg.mode, cfg.cache_policy, a.moe_gain
            );
            rivoli::eval::run(&mut engine, &ids, a.ppl_out.as_ref(), &label)?;
            return Ok(());
        }

        let t0 = std::time::Instant::now();
        // Speculative decode is the default, but only where it is BUILDABLE. TWO conditions
        // can take it away and neither is the user's mistake: an artifact without the MTP
        // head (an artifact converted before 2026-07-31 has no L78.i4, so int4/hybrid runs
        // on one load without a head), and `--trace` (a verify pass routes twice per layer
        // and submits the union, which the v2 trace format cannot spell). Say which, once.
        //
        // **The attention mode is no longer one of them, and it is the one that drew blood.**
        // Every sparse mode was on this list, `main` never actually checked for any of them,
        // and `--attn auto` picks dsa on any artifact carrying indexer weights — so a bare
        // `rivoli <artifact> -bench 8` ran speculation under DSA, slipped past the engine's
        // `nrow > 1` guard on the one-row DRAFT pass, and PANICKED indexing the indexer's
        // per-layer slab table at the MTP head's layer ("len is 78 but the index is 78").
        // Found 2026-08-01 while building server mode. Fixed by BATCHING the row selection
        // rather than refusing it: dsa/misa select per row and the head attends dense, and
        // streaming uploads one row set per row. All four modes speculate. See §13.
        // A verify pass is TWO token rows. That used to be a backend question: the Vulkan
        // `.comp` shaders were single-row and `gemv_fp8` rejected `nrow=2` outright, so a
        // `batched_rows = cfg!(not(feature = "vulkan"))` gated speculation off at compile
        // time. Discovered 2026-08-03 when tests/mode-matrix.sh ran the cross under both
        // backends and every Vulkan cell loaded weights for ~2 minutes before dying
        // mid-decode. With that backend retired (2026-08-06) every kernel takes two rows,
        // so the only remaining reasons to skip speculation are properties of the ARTIFACT
        // and the RUN.
        let mtp = !a.no_mtp && engine.has_mtp() && !engine.tracing();
        if !a.no_mtp && !mtp {
            info!(
                "speculative decode OFF: {}",
                if engine.has_mtp() {
                    "--trace routes once per layer and a verify pass routes twice"
                } else {
                    "this artifact carries no MTP head (re-run bin/fp8_to_i4 to \
                     emit L78.i4 for int4/hybrid)"
                }
            );
        }
        // Server mode: the same engine, driven by HTTP instead of by one bench prompt.
        // Everything below (degeneration report, OTLP export, --dump-ids) is the bench
        // path's epilogue and does not apply — serve() carries its own per-request one.
        if let Some(port) = a.port {
            return rivoli::serve::serve(
                &mut engine,
                &tok,
                &hb,
                &rivoli::serve::Opts {
                    port,
                    ctx: a.ctx,
                    // The artifact directory's own name, so `/v1/models` and the echoed
                    // `model` field say which checkpoint answered.
                    model_id: std::path::Path::new(&cfg.model)
                        .file_name()
                        .map_or_else(|| "rivoli".into(), |n| n.to_string_lossy().into_owned()),
                    mtp,
                    mtp_min_conf: a.mtp_min_conf,
                    think: a.think,
                },
            );
        }

        let (ids, summary) = engine.generate(
            &prompt_ids,
            ngen,
            &tok.eos,
            mtp,
            a.mtp_min_conf,
            &mut |_: u32| true,
            &followups,
        )?;
        let dt = t0.elapsed().as_secs_f64();
        let (hits, misses) = (engine.hits(), engine.misses());
        info!(
            "GPU: {} tokens in {dt:.1}s ({:.2} tok/s) | expert hit {:.1}% ({hits} hit / {misses} miss)",
            ids.len(),
            summary.tok_per_s,
            summary.hit_pct,
        );
        // OTLP: one decode span carrying the always-on summary (opt-in via
        // OTEL_EXPORTER_OTLP_ENDPOINT; log-only otherwise). Exported synchronously on
        // drop — no async runtime.
        // Degeneration check. A looped generation's tok/s is an ARTIFACT — it routes to
        // a handful of experts, inflates the hit rate, and therefore benchmarks FAST —
        // so it must be surfaced loudly, not left for someone to spot in the text. Three
        // verbatim repeats of a block up to 64 tokens long; prose does not do that.
        let degenerate = rivoli::telemetry::detect_loop(&ids, 3, 64);
        // The restart signal. A tail cycle is LATE degeneration; a run that answers and
        // then answers again is already broken and may have no cycle at all. Threshold:
        // an eighth of the generation, floor 32 tokens. Healthy prose repeats phrases,
        // not paragraphs.
        let lrb_bar = std::cmp::max(32, ids.len() / 8);
        let restart = rivoli::telemetry::has_repeated_block(&ids, lrb_bar);
        if degenerate.is_none() && restart {
            tracing::warn!(
                "SUSPECT OUTPUT: some block of {lrb_bar} tokens occurs twice in the {} \
                 generated, with no verbatim tail cycle — the shape of a RESTART rather \
                 than a loop. tok/s here is still suspect: re-answering re-routes to the \
                 same experts, which inflates the hit rate.",
                ids.len(),
            );
        }
        // Structural repetition, the signal the two exact-matching detectors above are
        // blind to. Both passed a run that was 329 repetitions of "**Memory Product.**"
        // with a varying label, so this is not a belt-and-braces addition — it is the
        // detector that actually works on the common failure shape.
        let text = tok.decode_all(&ids).unwrap_or_default();
        let rep = rivoli::telemetry::repetition_report(&text);
        tracing::info!(
            "generation: {} tokens, {lrb_bar}-token block repeated: {restart}, top-line x{}, \
             distinct {:.3}",
            ids.len(),
            rep.top_line,
            rep.distinct,
        );
        if rivoli::telemetry::is_degenerate(&rep) {
            tracing::warn!(
                "STRUCTURALLY DEGENERATE: one line repeats {}x and the distinct-word ratio \
                 is {:.3} (healthy band 0.42-0.53). This is a near-miss loop — a varying \
                 slot in a repeated template — which the verbatim-cycle and \
                 repeated-block checks CANNOT see. tok/s is not usable: the hit \
                 rate rises as the output collapses.",
                rep.top_line,
                rep.distinct,
            );
        }
        if let Some(d) = degenerate {
            tracing::warn!(
                "DEGENERATE OUTPUT: the last {} of {} generated tokens are {} verbatim \
                 repeats of a {}-token block (loop starts at token {}). This run's tok/s \
                 is NOT comparable — a looped decode reuses a few experts, so its hit \
                 rate and speed are both inflated. Investigate before ranking it.",
                d.period * d.repeats,
                ids.len(),
                d.repeats,
                d.period,
                d.start,
            );
        }
        // The run's identity for the root span.
        let run = rivoli::telemetry::RunInfo {
            model: cfg.model.clone(),
            mode: cfg.mode.to_string(),
            cache_policy: cfg.cache_policy.clone(),
            attn: a_attn.clone(),
            max_mem_gib: a.max_mem,
            mtp_min_conf: mtp.then_some(a.mtp_min_conf),
            bench_tokens: a.bench,
            prompt: cfg.prompt.clone(),
            moe_gain: a.moe_gain,
            sinks: a.sinks,
            window: a.window,
            misa_heads: a.misa_heads as usize,
            degenerate,
        };
        rivoli::telemetry::export_decode(&summary, ids.len(), &run);
        if let Some(path) = &a.dump_ids {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(
                std::fs::File::create(path).with_context(|| format!("create {path}"))?,
            );
            // Header names the arm, so two files cannot be silently compared across
            // different backends/modes — the same discipline `--ppl-out` uses.
            writeln!(
                w,
                // `backend=` is spelled into the literal because `rocm` is the only value it
                // can take since 2026-08-06. Still EMITTED, though: the header exists so two
                // dump files cannot be silently compared across arms, and a field that
                // vanishes when it becomes constant makes old dumps unreadable against new
                // ones. A second backend puts the `{}` back.
                "# rivoli-ids v1 backend=rocm mode={} policy={} attn={} tokens={}",
                cfg.mode,
                cfg.cache_policy,
                a_attn,
                ids.len(),
            )?;
            for id in &ids {
                writeln!(w, "{id}")?;
            }
            w.flush().context("flush --dump-ids")?;
            info!("wrote {} token ids to {path}", ids.len());
        }
        info!("{bench_prompt}{}", tok.decode_all(&ids)?);
        Ok(())
    }
}
