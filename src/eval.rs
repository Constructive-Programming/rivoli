//! Teacher-forced scoring — the quality instrument, behind `--features teacher-forcing`.
//!
//! Separate from the decode path on purpose. A free-running generation's own trajectory is
//! the confound this measurement exists to remove: `--ppl` walks a FIXED corpus, feeding
//! the true token at every position, so two runs over the same text are PAIRED and their
//! difference detects a systematic shift far smaller than the sampling noise in two
//! independent perplexities. See CLAUDE.md, "Measurement discipline".
//!
//! Gated because it is an instrument, not an engine feature: no STOCK decode reaches this
//! module. (Until 2026-08-08 this sentence said "nothing in a decode reaches it" — no
//! longer literally true: both engines' decode loops call [`LogitTrace::step`] and
//! [`nll_of`] when an instrument is ARMED, which only a `teacher-forcing` build can do and
//! only a `--logit-dump`/`--ppl` run does. The gate is the same; the call graph grew.)
//! The statistics that consume its output live in `bin/ppl`, which is pure host
//! arithmetic over the `.nll` files written here and needs no backend at all.

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

/// Per-position record/force/score state for ONE armed instrument — shared by BOTH
/// engines' decode hooks (`GpuEngine`'s and `F4Engine`'s `trace_step`/`next_token`), which
/// is the point: the forcing rule, the dump byte format, and the NLL accumulation live
/// here exactly once, so the two engines cannot drift on any of them. Lived in `f4gpu.rs`
/// until 2026-08-08, when GLM gained the same instruments.
///
/// Two constructors, two instruments:
/// - [`LogitTrace::for_dump`] — `--logit-dump`/`--force-tokens`, the drift A/B between two
///   builds. Writes every position's own-argmax + full logit row to a file.
/// - [`LogitTrace::for_scoring`] — `--ppl` on a V4 artifact. Writes nothing per step;
///   accumulates one NLL per forced position, collected by [`LogitTrace::into_nlls`].
///
/// Dump file format, read by `docs/measurement/probes/v4_logit_drift.py`: magic `b"V4LT"`,
/// vocab as u32 LE, then per decode position the engine's OWN argmax as u32 LE followed by
/// all `vocab` logits as f32 LE. The magic is a MISNOMER since 2026-08-08 — GLM dumps use
/// the identical layout (the header's vocab field is what varies: 154880 vs 129280) — and
/// is kept anyway: the probe script keys on these four bytes, and every recorded M9 dump
/// carries them, so changing the magic would fork the format to fix a name.
pub struct LogitTrace {
    /// `--logit-dump`'s file; `None` when armed for `--ppl` scoring.
    out: Option<std::io::BufWriter<std::fs::File>>,
    /// Token to CONSUME at each position, replacing the engine's own argmax — arm K of a
    /// drift A/B forces arm S's recorded stream so every position stays comparable instead
    /// of only the prefix before the first divergence. `--ppl` forces the corpus itself
    /// (`corpus[1..]`), so this same list is also what each position is scored against:
    /// `step` never needs a second, separate target array.
    force: Option<Vec<u32>>,
    pos: usize,
    /// `--ppl`'s accumulator; `None` when armed for dumping.
    nlls: Option<Vec<f32>>,
}

impl LogitTrace {
    /// Arm for `--logit-dump`: create the dump file and write its header.
    ///
    /// The force list is bounds-checked against `vocab` HERE because a forced id is the
    /// one token source neither engine's own plumbing vets: argmax output and tokenizer
    /// output are `< vocab` by construction, and while V4's `forward` re-checks every id,
    /// GLM's does not — `launch_embed_i8_row` indexes the embedding slab with the raw
    /// value, so an oversized id from a `--force-tokens` file would be a device
    /// out-of-bounds READ (garbage logits, no fault), the silent kind. One check at arm
    /// time covers both engines.
    pub fn for_dump(path: &str, vocab: usize, force: Option<Vec<u32>>) -> Result<Self> {
        use std::io::Write;
        if let Some(f) = &force {
            for (i, &t) in f.iter().enumerate() {
                anyhow::ensure!(
                    (t as usize) < vocab,
                    "--force-tokens line {}: id {t} outside vocab {vocab}",
                    i + 1
                );
            }
        }
        let f = std::fs::File::create(path).with_context(|| format!("create {path}"))?;
        let mut out = std::io::BufWriter::new(f);
        out.write_all(b"V4LT")?;
        out.write_all(&u32::try_from(vocab)?.to_le_bytes())?;
        Ok(Self {
            out: Some(out),
            force,
            pos: 0,
            nlls: None,
        })
    }

    /// Arm for `--ppl` scoring: force `corpus[1..]` (position 0 is context only) and
    /// accumulate one NLL per forced position. The caller checks `corpus.len() >= 2`.
    pub fn for_scoring(corpus: &[u32]) -> Self {
        Self {
            out: None,
            force: Some(corpus[1..].to_vec()),
            pos: 0,
            nlls: Some(Vec::with_capacity(corpus.len() - 1)),
        }
    }

    /// One armed step. `row_le` is the position's full logit row as LE f32 bytes (each
    /// engine reads its own device buffer; this function is the everything-after). Records
    /// `own` + the row when dumping, scores the row against the about-to-be-forced token
    /// when scoring, and returns the token the decode should CONSUME — the forced one
    /// where the list covers this position, the engine's own otherwise (a short list
    /// forces a prefix and free-runs the rest, which the arming log line states).
    pub fn step(&mut self, own: u32, row_le: &[u8]) -> Result<u32> {
        use std::io::Write;
        // The token about to be forced IS the corpus's true next token whenever
        // `for_scoring` armed this trace (it built `force` from the corpus itself).
        let next = self.force.as_ref().and_then(|f| f.get(self.pos)).copied();
        if let (Some(nlls), Some(target)) = (self.nlls.as_mut(), next) {
            nlls.push(nll_of(row_le, target as usize)?);
        }
        if let Some(out) = self.out.as_mut() {
            out.write_all(&own.to_le_bytes())?;
            out.write_all(row_le)?;
        }
        self.pos += 1;
        Ok(next.unwrap_or(own))
    }

    /// Flush the dump file if there is one. A `BufWriter` flushes on drop but swallows the
    /// error there; a truncated dump read as "no drift past position N" is exactly the
    /// silent kind this instrument exists to rule out. No-op when armed for scoring.
    pub fn flush(&mut self) -> Result<()> {
        use std::io::Write;
        if let Some(out) = self.out.as_mut() {
            out.flush()?;
        }
        Ok(())
    }

    /// The accumulated NLLs of a `for_scoring` trace.
    pub fn into_nlls(self) -> Result<Vec<f32>> {
        self.nlls
            .context("this LogitTrace was armed for dumping (for_dump), not scoring")
    }
}

/// Parse a `--force-tokens` file: one decimal id per line, blank lines skipped. Refused
/// loudly on any unparseable line rather than truncated to its valid prefix, because a
/// silently-shortened force list turns a positionally-comparable A/B into a free-running
/// one at some position nobody chose.
pub fn load_force_tokens(path: &str) -> Result<Vec<u32>> {
    std::fs::read_to_string(path)
        .with_context(|| format!("read --force-tokens {path}"))?
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            l.parse::<u32>()
                .map_err(|e| anyhow::anyhow!("--force-tokens {path}: {l:?}: {e}"))
        })
        .collect()
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
/// [`crate::f4gpu::F4Engine::nll_forced`] reuses `generate`'s free-run loop and its existing
/// force hook rather than a second bespoke forward loop (V4's `forward` only ever produces
/// logits for the LAST row it is given, so there is no per-token forward to duplicate
/// outside `generate`'s own decode arm — see that method's doc). `on_tok` is threaded
/// through rather than read off `self.heartbeat` the way GLM's [`crate::gpu::GpuEngine`]
/// does, because `F4Engine` has no heartbeat field of its own — its callers beat the
/// watchdog from `generate`'s callback instead.
pub fn run_v4(
    engine: &mut crate::f4gpu::F4Engine,
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

    /// A per-test scratch file path, unique per pid+thread so `cargo test`'s parallel
    /// default cannot make two tests collide. A bare file in the temp dir, not a subdir:
    /// nothing here needs one, and the shorter setup also keeps this from token-colliding
    /// with the tempdir fixtures `tests/f4_loading.rs`/`tests/v4_oracle.rs` carry (the
    /// jscpd gate found exactly that clone in an earlier draft of this module's tests).
    fn tmp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rivoli-eval-{}-{:?}-{name}",
            std::process::id(),
            std::thread::current().id()
        ))
    }

    /// **The dedup gate.** `write_nll_file` was extracted out of `run`'s body verbatim — this
    /// pins the exact bytes a caller sees, so any future edit that reflows the header or the
    /// per-line format (a stray space, a different float precision, `\r\n`) fails loudly
    /// instead of silently invalidating every historical `.nll` file `bin/ppl` and a dozen
    /// closed investigations' paired-dNLL claims depend on.
    #[test]
    fn write_nll_file_byte_for_byte() {
        let path = tmp_path("cell.nll");
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
    /// against each possible target. This is exactly the computation `F4Engine::trace_step`
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

    fn row_le(logits: &[f32]) -> Vec<u8> {
        logits.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The dump file's exact bytes, host-only — the format both engines now write and
    /// `v4_logit_drift.py` reads. Pins the (misnamed, kept) `V4LT` magic, the vocab
    /// header, and the per-position [argmax u32][row f32 x vocab] layout, so a change to
    /// any of them fails here rather than in a probe script mid-investigation.
    #[test]
    fn dump_trace_byte_layout_and_forcing() {
        let path = tmp_path("arm.lt");
        let path_str = path.to_str().unwrap().to_string();
        // Force list covers position 0 only: position 1 free-runs (returns own).
        let mut tr = LogitTrace::for_dump(&path_str, 3, Some(vec![1])).unwrap();
        let (r0, r1) = ([0.5f32, 2.0, -1.0], [4.0f32, 0.0, 1.0]);
        assert_eq!(tr.step(2, &row_le(&r0)).unwrap(), 1, "position 0 is forced");
        assert_eq!(
            tr.step(9, &row_le(&r1)).unwrap(),
            9,
            "past the list: free-run"
        );
        tr.flush().unwrap();
        let got = std::fs::read(&path).unwrap();
        let mut want = b"V4LT".to_vec();
        want.extend_from_slice(&3u32.to_le_bytes());
        want.extend_from_slice(&2u32.to_le_bytes());
        want.extend_from_slice(&row_le(&r0));
        want.extend_from_slice(&9u32.to_le_bytes());
        want.extend_from_slice(&row_le(&r1));
        assert_eq!(got, want, "the .lt dump layout changed");
        // A dump trace accumulates no NLLs, and saying so must be an error, not a panic.
        assert!(tr.into_nlls().is_err());
        // A forced id at or past the vocab is refused at arm time — GLM's forward would
        // otherwise read the embedding slab out of bounds (see `for_dump`'s doc).
        assert!(LogitTrace::for_dump(&path_str, 3, Some(vec![3])).is_err());
        std::fs::remove_file(&path).ok();
    }

    /// The scoring trace end-to-end at host scale: forces `corpus[1..]`, scores each row
    /// against the token it is about to force, writes nothing. The NLL value cross-checks
    /// against a direct `nll_of` call, so the two paths into that function cannot diverge.
    #[test]
    fn scoring_trace_forces_and_accumulates() {
        let corpus = [10u32, 1, 2];
        let mut tr = LogitTrace::for_scoring(&corpus);
        let (r0, r1) = ([1.0f32, 2.0, 3.0, 0.0], [0.0f32, 1.0, 0.5, 2.0]);
        // Position 0: engine's own argmax is irrelevant to what gets consumed.
        assert_eq!(tr.step(3, &row_le(&r0)).unwrap(), 1);
        assert_eq!(tr.step(0, &row_le(&r1)).unwrap(), 2);
        let nlls = tr.into_nlls().unwrap();
        assert_eq!(nlls.len(), 2, "one NLL per forced position");
        assert_eq!(nlls[0], nll_of(&row_le(&r0), 1).unwrap());
        assert_eq!(nlls[1], nll_of(&row_le(&r1), 2).unwrap());
    }

    /// `--force-tokens` parsing: blank lines skipped, any bad line refuses the whole file
    /// — a silently-shortened force list is the failure this loudness exists to prevent.
    #[test]
    fn force_tokens_parse_and_refuse() {
        let good = tmp_path("force-good.txt");
        std::fs::write(&good, "5\n\n  17 \n0\n").unwrap();
        assert_eq!(
            load_force_tokens(good.to_str().unwrap()).unwrap(),
            vec![5, 17, 0]
        );
        let bad = tmp_path("force-bad.txt");
        std::fs::write(&bad, "5\nnot-a-token\n7\n").unwrap();
        assert!(load_force_tokens(bad.to_str().unwrap()).is_err());
        std::fs::remove_file(&good).ok();
        std::fs::remove_file(&bad).ok();
    }
}
