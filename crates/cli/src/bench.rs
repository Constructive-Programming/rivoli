//! The `--bench` invocation: resolve the prompt, decode N tokens, report, and write the
//! `--dump-ids` record.
//!
//! **Split out of `main.rs` on 2026-08-16**, when `--ppl` pushed that file past the
//! 800-line soft cap the CLI's own build script warns at. The cut is the one the file
//! already had: `main` parses, sniffs and opens, and each of the three invocations owns
//! its own module — `serve`, `nll`, and now this.
//!
//! Moved verbatim EXCEPT the `RunInfo` + `export_decode` block at the end of `run_bench`,
//! which is new. It is worth saying why it is new: **`export_decode` had no caller at all
//! before this** (`git grep export_decode 264758c` finds the definition and a doc link,
//! nothing else), so the whole OTLP half of `telemetry` was exporting nothing — the same
//! "a profile nothing fills" trap `ProfileSummary`'s doc names, one level up. This is its
//! first caller.

use crate::{Args, BENCH_PROMPT};
use anyhow::{Context, Result, ensure};
use rivoli_artifact::tokenizer::Tokenizer;
use rivoli_core::legality::{ATTNS, Arch, MODES, name_in};
use rivoli_engine::{Engine, GenSpec};
use std::io::Write as _;

/// The `--bench` arm's fixed input, resolved before any weight is placed.
pub(crate) struct Bench<'a> {
    /// Printed back on stdout ahead of the completion, so the two read together.
    text: &'a str,
    ids: Vec<u32>,
    ngen: usize,
}

/// Frame the bench prompt in the architecture's OWN chat template.
///
/// **The two encoders share no framing**, which is why this is a match and not a default:
/// GLM builds a token-ID list from ids looked up by name, DeepSeek-V4 builds a string with its
/// markers written out and then tokenizes it. Feeding a V4 checkpoint GLM's `[gMASK] <sop>`
/// prefix is not a near-miss — it is text the model never saw, and an instruct model outside
/// its turn structure does document continuation and never emits a stop token, which is the
/// failure that invalidated 56 benchmark runs in the old tree.
///
/// **Muse Glimmer takes GLM's framing here and that is a KNOWN GAP, not a decision.**
/// `rivoli_artifact::glimmer_encoding` exists and is not wired to this door; it predates this
/// arm and is left exactly as it was, because changing it is a Glimmer change owed its own
/// id-pinned comparison rather than a line inside a V4 port.
fn frame_prompt(tok: &Tokenizer, arch: Arch, text: &str) -> Result<Vec<u32>> {
    use rivoli_artifact::v4_encoding::{EncodeOpts, Message, ThinkingMode};
    match arch {
        // `ThinkingMode::Chat` — the same default `ChatOpts::thinking` takes and for the same
        // measured reason: at a few tok/s a reasoning block is tens of seconds of silence
        // before the first word, and a `--bench` run is timing the decode, not the reasoning.
        Arch::DeepseekV4 => tok.encode_dsv4(
            vec![Message::user(text)],
            &EncodeOpts::new(ThinkingMode::Chat),
        ),
        Arch::GlmMoeDsa | Arch::MuseGlimmer => tok.encode_chat(text),
        // RAW, deliberately: K3 ships no chat template in ANY tree (`convert_k3`'s header
        // records it), so there is no "its own framing" to render — this line inherited
        // GLM's `encode_chat` until M9, which would have fed a K3 checkpoint `[gMASK]
        // <sop>` markers it never saw. A base-model bench prompt is document continuation,
        // and raw encoding is the honest spelling of that.
        Arch::KimiK3 => tok.encode(text),
    }
}

pub(crate) fn bench_input<'a>(
    tok: &Tokenizer,
    arch: Arch,
    a: &'a Args,
    ngen: usize,
) -> Result<Bench<'a>> {
    let text = a.prompt.as_deref().unwrap_or(BENCH_PROMPT);
    let ids = frame_prompt(tok, arch, text)?;
    // The KV slab is sized once from `--ctx`; a run that outgrows it fails somewhere in the
    // token loop, minutes in, so it is refused here instead.
    ensure!(
        ids.len() + ngen <= a.ctx,
        "{} prompt + {ngen} generated tokens exceed --ctx {} — raise it (~51 KB of device \
         memory per token) or shorten the run",
        ids.len(),
        a.ctx
    );
    Ok(Bench { text, ids, ngen })
}

pub(crate) fn run_bench(
    eng: &mut Engine<'_>,
    tok: &Tokenizer,
    a: &Args,
    b: &Bench<'_>,
) -> Result<()> {
    let out = eng.generate(
        GenSpec {
            prompt: &b.ids,
            ngen: b.ngen,
            eos: &tok.eos,
        },
        &mut |_| true,
    )?;

    // stdout: the prompt and its completion, the way the reference engine reports a bench —
    // reading them together is what tells you whether the run answered the question.
    println!("{}{}", b.text, tok.decode_all(&out.ids)?);
    // stderr, and NOT through `tracing`: the engine logs its own DECODE line at `info`, but
    // a benchmark number that disappears under `RUST_LOG=warn` is a number someone will one
    // day fail to find. The prompt length is here and not there, which is the other half of
    // why this is not a restatement: decode stats deliberately EXCLUDE the prefill.
    eprintln!(
        "BENCH {:.2} tok/s | {} prompt + {} generated tokens in {:.1} s | {} expert hits, \
         {} misses",
        out.stats.tok_s,
        b.ids.len(),
        out.ids.len(),
        out.stats.decode_s,
        out.stats.hits,
        out.stats.misses
    );
    // The OTLP export (a no-op without the `otlp` feature + endpoint): the phase profile the
    // engine just reported, labelled with WHAT THIS RUN WAS so two runs stay two series.
    let run = rivoli_engine::telemetry::RunInfo {
        model: a.model.clone(),
        mode: name_in(&MODES, a.mode).to_string(),
        cache_policy: a.cache_policy.clone(),
        attn: name_in(&ATTNS, a.attn).to_string(),
        max_mem_gib: a.max_mem,
        // Speculative decode does not exist in this tree yet (`--mtp` is refused on
        // every arm); the gate value arrives with M12.
        mtp_min_conf: None,
        bench_tokens: Some(b.ngen),
        prompt: Some(b.text.to_string()),
        // (3, 64): the reference engine's thresholds (`old:src/main.rs`) — at least 3
        // verbatim copies of a block up to 64 tokens long counts as a wedged tail.
        degenerate: rivoli_engine::telemetry::detect_loop(&out.ids, 3, 64),
    };
    rivoli_engine::telemetry::export_decode(&out.profile, out.ids.len(), &run);
    match &a.dump_ids {
        Some(path) => write_ids(path, a, &out.ids),
        None => Ok(()),
    }
}

/// Write the generated ids, one per line, under a header naming the configuration.
///
/// The header is why this exists at all: two dump files from different modes, policies or
/// attention settings must not be silently comparable. `backend=rocm` is spelled into the
/// literal because a build that reaches here has one — a backendless build refuses at
/// `Engine::open`, long before a token exists to dump. It is still EMITTED, because a
/// field that vanishes once it becomes constant makes old dumps unreadable against new
/// ones; a second backend puts the `{}` back.
fn write_ids(path: &str, a: &Args, ids: &[u32]) -> Result<()> {
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("create {path}"))?,
    );
    writeln!(
        w,
        "# rivoli-ids v1 backend=rocm mode={} policy={} attn={} tokens={}",
        name_in(&MODES, a.mode),
        a.cache_policy,
        name_in(&ATTNS, a.attn),
        ids.len(),
    )?;
    for id in ids {
        writeln!(w, "{id}")?;
    }
    w.flush().context("flush --dump-ids")?;
    tracing::info!("wrote {} token ids to {path}", ids.len());
    Ok(())
}
