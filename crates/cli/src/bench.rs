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
/// > **THE GLIMMER GAP IS CLOSED, M11b 2026-08-17.** This said: "Muse Glimmer takes GLM's
/// > framing here and that is a KNOWN GAP, not a decision. `rivoli_artifact::glimmer_encoding`
/// > exists and is not wired to this door; it predates this arm and is left exactly as it was,
/// > because changing it is a Glimmer change owed its own id-pinned comparison rather than a
/// > line inside a V4 port." The comparison exists now
/// > (`glimmer_template.rs::rendered_prompts_tokenize_to_the_vendored_ids`, 31/31 cases
/// > `render` → the shipped tokenizer → `apply_chat_template`'s own ids), so the arm is split
/// > and Glimmer renders its own template. **This CHANGES Glimmer bench ids**, which is why it
/// > is its own commit and why no fp8 A/B may straddle it.
fn frame_prompt(tok: &Tokenizer, arch: Arch, text: &str) -> Result<Vec<u32>> {
    use rivoli_artifact::glimmer_encoding;
    use rivoli_artifact::v4_encoding::{EncodeOpts, Message, ThinkingMode};
    match arch {
        // `ThinkingMode::Chat` — the same default `ChatOpts::thinking` takes and for the same
        // measured reason: at a few tok/s a reasoning block is tens of seconds of silence
        // before the first word, and a `--bench` run is timing the decode, not the reasoning.
        Arch::DeepseekV4 => tok.encode_dsv4(
            vec![Message::user(text)],
            &EncodeOpts::new(ThinkingMode::Chat),
        ),
        // **A STRING then tokenized, unlike GLM's id list** — Glimmer's template is a hand-port
        // that emits `<|start|>`/`<|message|>`/`<|eot|>` as literal text, and that those become
        // single ids is the shipped tokenizer's doing rather than this crate's. That dependency
        // is measured, not assumed: the id pin cited above runs `Tokenizer::encode` over
        // `render`'s bytes and compares against the ids `apply_chat_template` produced.
        //
        // **The date reaches the prompt, and the consequence is stated rather than dodged.**
        // The template's synthesised system block carries `Current date: YYYY-MM-DD.`, so a
        // Glimmer bench framed today and the same bench framed tomorrow differ in a handful of
        // ids. `utc_date`'s own doc puts that decision here — "the CLI, which is where 'what
        // day is it' is a legitimate question" — and the alternative, passing `""`, renders a
        // system block no reference renderer can produce. Prompt LENGTH is invariant (the date
        // is always ten characters), so tok/s is unaffected; what a recorded run must carry is
        // the DATE, which is why this arm logs it. Without that line a recorded Glimmer bench
        // is unreproducible for a reason nothing in its output names.
        Arch::MuseGlimmer => {
            let date = glimmer_encoding::utc_date(std::time::SystemTime::now());
            tracing::info!("glimmer framing: own chat template, current_date {date}");
            tok.encode(&glimmer_encoding::render(
                &[serde_json::json!({"role": "user", "content": text})],
                &glimmer_encoding::GlimmerChatOpts {
                    // **`"none"`, on the V4 arm's argument two lines up, not the template's
                    // default `"high"`.** Glimmer's Jinja has no thinking boolean — it always
                    // renders `Reasoning strength: <s>.` — so a bench that said nothing would
                    // opt INTO maximal reasoning, and `--bench N` would spend its whole budget
                    // in the `to=self` channel: a transcript of the model thinking rather than
                    // answering, which is exactly what `ThinkingMode::Chat` above exists to
                    // avoid. Review 2026-08-17 found the two arms disagreeing with nothing
                    // written down; this is the agreement.
                    reasoning_strength: Some("none"),
                    ..glimmer_encoding::GlimmerChatOpts::new(&date)
                },
            ))
        }
        Arch::GlmMoeDsa => tok.encode_chat(text),
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
    // Armed BEFORE the decode and written after it — the log is held in memory precisely so
    // the measurement is not perturbed by its own instrument.
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    if a.divergence_log.is_some() {
        let folds = a.divergence_folds.unwrap_or_default();
        tracing::info!("divergence probe: folds = {}", folds.label());
        eng.arm_divergence_log(folds)?;
    }
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
    // After the run, never during it (see `arm_divergence_log` above).
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    if let Some(path) = &a.divergence_log {
        eng.write_divergence_log(path)?;
    }
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
