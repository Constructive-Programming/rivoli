//! Teacher-forced scoring — the quality instrument, behind `--features teacher-forcing`.
//!
//! Separate from the decode path on purpose. A free-running generation's own trajectory is
//! the confound this measurement exists to remove: `--ppl` walks a FIXED corpus, feeding
//! the true token at every position, so two runs over the same text are PAIRED and their
//! difference detects a systematic shift far smaller than the sampling noise in two
//! independent perplexities. See CLAUDE.md, "Measurement discipline".
//!
//! Gated because it is an instrument, not an engine feature: nothing in a decode reaches
//! this module. The statistics that consume its output live in `bin/ppl`, which is pure
//! host arithmetic over the `.nll` files written here and needs no backend at all.

use anyhow::{Context, Result};
use tracing::info;

/// FNV-1a 64 of `bytes`, hex. Identity tag for the `--ppl` corpus, logged beside the
/// numbers so a result can be checked against the text that produced it.
///
// ponytail: not a cryptographic hash and does not need to be — the corpus is committed,
// so git already holds its real identity; this only has to catch "the file changed under
// us". A sha256 would mean taking a dependency to print one line.
pub fn fnv1a64_hex(bytes: &[u8]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h = (h ^ b as u64).wrapping_mul(0x1000_0000_01b3);
    }
    format!("{h:016x}")
}

/// `-log softmax(logits)[target]` from a little-endian f32 logit vector.
///
/// Shifted by the max before exponentiating, which is load-bearing rather than tidy:
/// this model's raw logits routinely run past 80, `exp(80)` overflows f32 to `inf`, and
/// an `inf` in the sum yields a NaN NLL. A NaN would then propagate into a mean that
/// still *looks* like a plausible perplexity, so the failure would be silent and would
/// land in a number we are using to decide a feature. Accumulated in f64 for the same
/// reason — 154,880 f32 addends lose real precision to rounding.
pub fn nll_of(logits_le: &[u8], target: usize) -> anyhow::Result<f32> {
    let n = logits_le.len() / 4;
    anyhow::ensure!(n > 0 && target < n, "target {target} outside {n} logits");
    let z = |i: usize| {
        let b = &logits_le[4 * i..4 * i + 4];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    };
    let mut max = f32::NEG_INFINITY;
    for i in 0..n {
        let v = z(i);
        if v > max {
            max = v;
        }
    }
    anyhow::ensure!(max.is_finite(), "non-finite logits");
    let sum: f64 = (0..n).map(|i| ((z(i) - max) as f64).exp()).sum();
    let nll = (sum.ln() - (z(target) - max) as f64) as f32;
    anyhow::ensure!(nll.is_finite(), "non-finite NLL");
    Ok(nll)
}

/// Read the corpus, tag it, and tokenize it RAW (no chat framing — it is a text to score,
/// not a turn to answer).
pub fn load_corpus(path: &str, tok: &crate::artifact::tokenizer::Tokenizer) -> Result<Vec<u32>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read --ppl text {path}"))?;
    let ids = tok.encode(&text)?;
    // The corpus is part of the measurement, so its identity goes in the log next to the
    // numbers — a quality claim whose text drifted is worth nothing later.
    info!(
        "ppl corpus {path:?}: {} bytes, {} tokens, fnv1a64 {}",
        text.len(),
        ids.len(),
        fnv1a64_hex(text.as_bytes()),
    );
    Ok(ids)
}

/// Score `ids` teacher-forced and report; write per-token NLLs to `out` when given.
///
/// `label` is the first three whitespace fields of the `.nll` header, which is what
/// `bin/ppl` uses to name a cell — a gain sweep differs in NOTHING else, so six arms would
/// otherwise write six identical headers and be indistinguishable downstream.
pub fn run(
    engine: &mut crate::gpu::GpuEngine,
    ids: &[u32],
    out: Option<&String>,
    label: &str,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let nlls = engine.nll_forced(ids)?;
    let dt = t0.elapsed().as_secs_f64();
    let mean = nlls.iter().map(|&v| f64::from(v)).sum::<f64>() / nlls.len().max(1) as f64;
    let (hits, misses) = (engine.hits(), engine.misses());
    let hit_pct = 100.0 * hits as f64 / (hits + misses).max(1) as f64;
    info!(
        "PPL: {:.6} (mean NLL {mean:.6} over {} predicted tokens, {dt:.1}s) | hit {hit_pct:.2}%",
        mean.exp(),
        nlls.len(),
    );
    // Per-token NLLs are the actual deliverable: two runs over the same text are PAIRED at
    // every position, and differencing them detects a systematic shift far smaller than the
    // sampling noise in two independent perplexities.
    if let Some(path) = out {
        use std::io::Write;
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(path).with_context(|| format!("create {path}"))?,
        );
        writeln!(
            w,
            "# rivoli-nll v1 {label} tokens={} hit_pct={hit_pct:.4}",
            nlls.len()
        )?;
        for v in &nlls {
            writeln!(w, "{v:.8}")?;
        }
        w.flush().context("flush --ppl-out")?;
        info!("wrote {} per-token NLLs to {path}", nlls.len());
    }
    Ok(())
}
