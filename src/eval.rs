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
    let max = (0..n).map(z).fold(f32::NEG_INFINITY, f32::max);
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

/// The `PPL:` log line and the hit-rate arithmetic it and the `.nll` header both need.
/// Shared because GLM's [`run`] and V4's [`run_v4`] compute it from a differently-typed
/// engine's hit/miss counters but must not report it differently — a mean and a percentage
/// have no room to drift between two copies of the same formula.
fn report(nlls: &[f32], dt: f64, hits: u64, misses: u64) -> f64 {
    let mean = nlls.iter().map(|&v| f64::from(v)).sum::<f64>() / nlls.len().max(1) as f64;
    let hit_pct = 100.0 * hits as f64 / (hits + misses).max(1) as f64;
    info!(
        "PPL: {:.6} (mean NLL {mean:.6} over {} predicted tokens, {dt:.1}s) | hit {hit_pct:.2}%",
        mean.exp(),
        nlls.len(),
    );
    hit_pct
}

/// Write one `.nll` v1 file — the ONLY writer. GLM's [`run`] and V4's [`run_v4`] both call
/// this rather than each formatting their own header, so the load-bearing byte format
/// (`bin/ppl` and a dozen closed investigations' paired-dNLL claims read these files) cannot
/// drift between the two architectures. `label` is the first three whitespace fields of the
/// header, which is what `bin/ppl` uses to name a cell — a gain sweep differs in NOTHING
/// else, so six arms would otherwise write six identical headers and be indistinguishable
/// downstream.
///
/// Private: `report`/`finish` have the same "only `run`/`run_v4`/tests call this" shape and
/// are private too — a `pub` writer invites a future caller to skip `report`'s hit_pct
/// computation and re-open the "two similar-but-not-identical writers" problem this exists
/// to close. The tests below reach it via `use super::*`, which sees private items fine.
fn write_nll_file(path: &str, label: &str, nlls: &[f32], hit_pct: f64) -> Result<()> {
    use std::io::Write;
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
    info!("wrote {} per-token NLLs to {path}", nlls.len());
    Ok(())
}

/// The tail both [`run`] and [`run_v4`] share once they have a scored run: [`report`] the
/// `PPL:` line, then [`write_nll_file`] if a caller wants the per-token record. Pulled out
/// rather than left inline in both — `run`'s body and `run_v4`'s body would otherwise be
/// four lines of identical `(hits, misses) -> hit_pct -> maybe write` plumbing apiece, which
/// is exactly the shape this module exists to NOT have twice.
fn finish(
    nlls: Vec<f32>,
    dt: f64,
    hits: u64,
    misses: u64,
    out: Option<&String>,
    label: &str,
) -> Result<()> {
    let hit_pct = report(&nlls, dt, hits, misses);
    if let Some(path) = out {
        write_nll_file(path, label, &nlls, hit_pct)?;
    }
    Ok(())
}

/// Score `ids` teacher-forced (GLM) and report; write per-token NLLs to `out` when given.
/// Per-token NLLs are the actual deliverable: two runs over the same text are PAIRED at
/// every position, and differencing them detects a systematic shift far smaller than the
/// sampling noise in two independent perplexities.
pub fn run(
    engine: &mut crate::gpu::GpuEngine,
    ids: &[u32],
    out: Option<&String>,
    label: &str,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let nlls = engine.nll_forced(ids)?;
    let dt = t0.elapsed().as_secs_f64();
    let (hits, misses) = (engine.hits(), engine.misses());
    finish(nlls, dt, hits, misses, out, label)
}

/// The V4 twin of [`run`] — same tail ([`finish`]), a different scorer:
/// [`crate::v4gpu::V4Engine::nll_forced`] reuses `generate`'s free-run loop and its existing
/// force hook rather than a second bespoke forward loop (V4's `forward` only ever produces
/// logits for the LAST row it is given, so there is no per-token forward to duplicate
/// outside `generate`'s own decode arm — see that method's doc). `on_tok` is threaded
/// through rather than read off `self.heartbeat` the way GLM's [`crate::gpu::GpuEngine`]
/// does, because `V4Engine` has no heartbeat field of its own — its callers beat the
/// watchdog from `generate`'s callback instead.
pub fn run_v4(
    engine: &mut crate::v4gpu::V4Engine,
    ids: &[u32],
    out: Option<&String>,
    label: &str,
    on_tok: &mut dyn FnMut(u32) -> bool,
) -> Result<()> {
    let t0 = std::time::Instant::now();
    let nlls = engine.nll_forced(ids, on_tok)?;
    let dt = t0.elapsed().as_secs_f64();
    let (hits, misses) = (engine.pool_hits(), engine.pool_misses());
    finish(nlls, dt, hits, misses, out, label)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    /// **The dedup gate.** `write_nll_file` was extracted out of `run`'s body verbatim — this
    /// pins the exact bytes a caller sees, so any future edit that reflows the header or the
    /// per-line format (a stray space, a different float precision, `\r\n`) fails loudly
    /// instead of silently invalidating every historical `.nll` file `bin/ppl` and a dozen
    /// closed investigations' paired-dNLL claims depend on.
    #[test]
    fn write_nll_file_byte_for_byte() {
        let dir = std::env::temp_dir().join(format!(
            "rivoli-nll-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cell.nll");
        let path_str = path.to_str().unwrap().to_string();
        let nlls = [1.5f32, 2.25, 0.0, 10.0];
        write_nll_file(&path_str, "mode=int4 policy=2q moe_gain=1", &nlls, 91.5).unwrap();
        let got = std::fs::read_to_string(&path).unwrap();
        let want = "# rivoli-nll v1 mode=int4 policy=2q moe_gain=1 tokens=4 hit_pct=91.5000\n\
                     1.50000000\n\
                     2.25000000\n\
                     0.00000000\n\
                     10.00000000\n";
        assert_eq!(
            got, want,
            "the .nll v1 format changed — this is a regression, not a refactor"
        );
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }

    /// `report`'s hit_pct feeds straight into the header `write_nll_file` writes, so a wrong
    /// denominator here would drift into every cell's `hit_pct=` field without a single .nll
    /// byte looking wrong on its own.
    #[test]
    fn report_hit_pct_and_mean() {
        let nlls = [0.0f32, f32::ln(2.0)]; // exp(mean) = exp(ln(2)/2) = sqrt(2)
        let hit_pct = report(&nlls, 0.0, 3, 1);
        assert!((hit_pct - 75.0).abs() < 1e-9);
    }

    /// The toy-row check the brief asks for: a hand-computed softmax over 4 logits, scored
    /// against each possible target. This is exactly the computation `V4Engine::trace_step`
    /// now also drives — `nll_of` itself is untouched, but nothing in this repo pinned its
    /// output against arithmetic worked out by hand before today.
    #[test]
    fn nll_of_toy_row() {
        let logits = [1.0f32, 2.0, 3.0, 0.0];
        let le: Vec<u8> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
        // softmax([1,2,3,0]): shift by max=3 -> [-2,-1,0,-3], exp -> [.1353,.3679,1,.0498],
        // sum = 1.5530. -log(p[2]) = -log(1/1.5530) = log(1.5530) = 0.44024...
        let nll = nll_of(&le, 2).unwrap();
        assert!((nll - 0.440_24).abs() < 1e-3, "got {nll}");
        // The least likely target (index 3, logit 0) has the largest NLL of the four.
        let nll3 = nll_of(&le, 3).unwrap();
        assert!(nll3 > nll);
    }
}
