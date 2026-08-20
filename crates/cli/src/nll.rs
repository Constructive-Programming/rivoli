//! The `--ppl` invocation: corpus load + cell label, the scoring call, and the `.nll`
//! v1 producer — the format contract with `bin/ppl.rs`, which pairs these files
//! position-by-position. Ported from `old:src/eval.rs`'s tail (`load_corpus` /
//! `report` / `write_nll_file` / `finish`), byte format unchanged: every historical
//! `.nll` file and a dozen closed investigations' paired-dNLL claims read as this
//! writes.
//!
//! One writer, deliberately private plumbing behind [`run_ppl`]: the old tree's note
//! stands — a `pub` writer invites a caller to skip `report`'s hit_pct computation and
//! re-open the "two similar-but-not-identical writers" problem this module exists to
//! close.

use crate::Args;
use anyhow::{Context, Result, ensure};
use rivoli_artifact::tokenizer::Tokenizer;
use rivoli_core::legality::{ATTNS, MODES, name_in};
use rivoli_engine::{Engine, Scored};
use std::io::Write as _;

/// FNV-1a 64 of `bytes`, hex — the corpus identity tag, logged and written into the
/// header label so a quality claim can be checked against the text that produced it.
// ponytail: not a cryptographic hash and does not need to be — the corpus is committed,
// so git already holds its real identity; this only has to catch "the file changed under
// us". A sha256 would mean taking a dependency to print one line.
fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ u64::from(b)).wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

/// The `--ppl` invocation's fixed input, resolved before any weight is placed — the
/// corpus ids and the label that names this cell downstream.
pub(crate) struct Ppl {
    pub(crate) ids: Vec<u32>,
    label: String,
}

/// Load and tag the corpus, RAW-encoded: it is a text to score, not a turn to answer
/// (`old:src/eval.rs::load_corpus` — chat framing would prepend tokens the text never
/// contained and shift every scored position).
pub(crate) fn ppl_input(tok: &Tokenizer, path: &str, a: &Args) -> Result<Ppl> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read --ppl text {path}"))?;
    let ids = tok.encode(&text)?;
    ensure!(
        ids.len() >= 2,
        "--ppl corpus {path:?} tokenizes to {} token(s); need at least 2 to score a position",
        ids.len()
    );
    ensure!(
        ids.len() <= a.ctx,
        "--ppl corpus is {} tokens, past --ctx {} — raise it (~51 KB of device memory \
         per token) or score a shorter text",
        ids.len(),
        a.ctx
    );
    let fnv = fnv1a64_hex(text.as_bytes());
    // The corpus is part of the measurement, so its identity goes in the log AND the
    // label — a quality claim whose text drifted is worth nothing later.
    tracing::info!(
        "ppl corpus {path:?}: {} bytes, {} tokens, fnv1a64 {fnv}",
        text.len(),
        ids.len(),
    );
    // First three fields name the cell (`bin/ppl`'s `cell()` reads exactly three); the
    // budget and corpus tag ride behind for the record.
    let label = format!(
        "mode={} policy={} attn={} max_mem={} corpus={fnv}",
        name_in(&MODES, a.mode),
        a.cache_policy,
        name_in(&ATTNS, a.attn),
        a.max_mem
            .map_or_else(|| "auto".to_string(), |g| g.to_string()),
    );
    Ok(Ppl { ids, label })
}

/// The `--ppl` run: score, log the coherence/agreement line, write the `.nll` file.
pub(crate) fn run_ppl(eng: &mut Engine<'_>, a: &Args, p: &Ppl) -> Result<()> {
    let t0 = std::time::Instant::now();
    let s = eng.score(&p.ids)?;
    let n = s.nlls.len();
    // Reaching this line at all means the per-position coherence check held on every
    // scored row (`score_row` refuses the run on the first device-vs-host argmax
    // mismatch); the agreement rate is next-token accuracy, reported for the record.
    tracing::info!(
        "TF row-coherence held on all {n} positions; own-argmax agreement {}/{n} ({:.1}%)",
        s.agree,
        100.0 * s.agree as f64 / n.max(1) as f64,
    );
    finish(
        &s,
        t0.elapsed().as_secs_f64(),
        a.ppl_out.as_deref(),
        &p.label,
    )
}

/// The `PPL:` log line and the hit-rate arithmetic it and the `.nll` header both need —
/// one formula, so a mean and a percentage have no room to drift between two copies.
fn report(s: &Scored, dt: f64) -> f64 {
    let mean = s.nlls.iter().map(|&v| f64::from(v)).sum::<f64>() / s.nlls.len().max(1) as f64;
    let hit_pct = 100.0 * s.hits as f64 / (s.hits + s.misses).max(1) as f64;
    tracing::info!(
        "PPL: {:.6} (mean NLL {mean:.6} over {} predicted tokens, {dt:.1}s) | hit {hit_pct:.2}%",
        mean.exp(),
        s.nlls.len(),
    );
    hit_pct
}

/// Write one `.nll` v1 file — the ONLY writer. The byte format is pinned by
/// `nll_file_byte_for_byte` below; `bin/ppl` parses the header prefix and one f64 per
/// line, and a stray space or a different float precision silently invalidates every
/// historical file it would be paired against.
fn write_nll_file(path: &str, label: &str, nlls: &[f32], hit_pct: f64) -> Result<()> {
    let f = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
    let mut w = std::io::BufWriter::new(f);
    writeln!(
        w,
        "# rivoli-nll v1 {label} tokens={} hit_pct={hit_pct:.4}",
        nlls.len()
    )?;
    for v in nlls {
        writeln!(w, "{v:.8}")?;
    }
    w.flush().context("flush --ppl-out")?;
    tracing::info!("wrote {} per-token NLLs to {path}", nlls.len());
    Ok(())
}

/// The tail of every scoring run: the `PPL:` line, then the per-token record when a
/// caller asked for one. The per-token NLLs are the actual deliverable — two runs over
/// the same text are PAIRED at every position, and differencing them detects a
/// systematic shift far smaller than the sampling noise in two independent
/// perplexities (`bin/ppl`'s whole argument).
fn finish(s: &Scored, dt: f64, out: Option<&str>, label: &str) -> Result<()> {
    let hit_pct = report(s, dt);
    if let Some(path) = out {
        write_nll_file(path, label, &s.nlls, hit_pct)?;
    }
    Ok(())
}

#[cfg(test)]
mod format_pin_tests {
    #![allow(clippy::unwrap_used)] // tests: panic-on-failure is the idiom
    use super::*;

    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rivoli-nll-{}-{:?}-{name}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    fn scored(nlls: Vec<f32>, hits: u64, misses: u64) -> Scored {
        Scored {
            nlls,
            agree: 0,
            hits,
            misses,
        }
    }

    /// **The format pin, inherited from `old:src/eval.rs`'s own test.** This asserts the
    /// exact bytes a consumer sees, so any edit that reflows the header or the per-line
    /// format (a stray space, a different float precision, `\r\n`) fails loudly instead
    /// of silently invalidating every historical `.nll` file `bin/ppl` pairs against.
    #[test]
    fn nll_file_byte_for_byte() {
        let path = tmp_path("cell.nll");
        let path_str = path.to_str().unwrap().to_string();
        let nlls = [1.5f32, 2.25, 0.0, 10.0];
        write_nll_file(&path_str, "mode=int4 policy=2q attn=dense", &nlls, 91.5).unwrap();
        let got = std::fs::read_to_string(&path).unwrap();
        let want = "# rivoli-nll v1 mode=int4 policy=2q attn=dense tokens=4 hit_pct=91.5000\n\
                     1.50000000\n\
                     2.25000000\n\
                     0.00000000\n\
                     10.00000000\n";
        assert_eq!(
            got, want,
            "the .nll v1 format changed — this is a regression, not a refactor"
        );
        std::fs::remove_file(&path).ok();
    }

    /// The written file round-trips through `bin/ppl.rs`'s own reader rules: the header
    /// prefix is exactly what its `HEADER` const strips, and every body line parses as
    /// f64. Asserted here because the two live in one crate but compile separately — a
    /// drift between them is precisely a file this side writes that the other refuses.
    #[test]
    fn the_header_is_the_one_ppl_strips() {
        let path = tmp_path("roundtrip.nll");
        let path_str = path.to_str().unwrap().to_string();
        let s = scored(vec![0.5, 1.25], 3, 1);
        finish(&s, 1.0, Some(&path_str), "mode=a policy=b attn=c").unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let mut lines = text.lines();
        let hdr = lines.next().unwrap();
        // bin/ppl.rs: const HEADER = "# rivoli-nll v1 " — strip_prefix must succeed.
        let label = hdr.strip_prefix("# rivoli-nll v1 ").unwrap();
        assert!(label.starts_with("mode=a policy=b attn=c"));
        assert!(label.contains("hit_pct=75.0000"), "got {label}");
        for l in lines {
            l.trim().parse::<f64>().unwrap();
        }
        std::fs::remove_file(&path).ok();
    }

    /// `report`'s hit_pct feeds straight into the header, so a wrong denominator here
    /// drifts into every cell's `hit_pct=` field with no `.nll` byte looking wrong.
    #[test]
    fn report_hit_pct_and_zero_guard() {
        let hit_pct = report(&scored(vec![0.0, f32::ln(2.0)], 3, 1), 0.0);
        assert!((hit_pct - 75.0).abs() < 1e-9);
        // A dense arm reports 0 hits / 0 fills without dividing by zero.
        assert!(report(&scored(vec![1.0], 0, 0), 0.0).abs() < 1e-9);
    }

    /// The corpus tag is stable and format-pinned — it goes into recorded labels, so a
    /// changed constant would silently break every stored label comparison.
    #[test]
    fn fnv_tag_is_stable() {
        assert_eq!(fnv1a64_hex(b""), "cbf29ce484222325");
        assert_ne!(fnv1a64_hex(b"rivoli"), fnv1a64_hex(b"rivolI"));
    }
}
