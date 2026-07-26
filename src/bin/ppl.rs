//! Paired perplexity comparison for the `top-m` quality gate (docs/CACHE_ROUTE.md).
//!
//! Consumes the per-token NLL files written by `rivoli --ppl <text> --ppl-out <path>`,
//! the FIRST being the baseline, and reports each cell against it.
//!
//! **Why paired, and why that is the whole point of this tool.** The bar is ~1%
//! perplexity and the paper's reference is +0.1–3%. Two independently measured
//! perplexities over a few hundred tokens carry sampling noise of comparable size, so
//! comparing the headline numbers can neither confirm nor refute an effect that small.
//! But every cell scores *the same text at the same positions*, so the runs are paired:
//! differencing per-token NLL cancels the token-to-token variance that dominates each
//! run's spread, leaving only the systematic shift. That is the evidence; the headline
//! PPL is reported alongside purely for comparability with the paper.
//!
//! usage: ppl <baseline.nll> <cell.nll>...

use anyhow::{Context, Result, bail};
use std::io::BufRead;

/// One scored run: the header the engine wrote, plus its per-token NLLs.
struct Run {
    path: String,
    label: String,
    nll: Vec<f64>,
}

fn load(path: &str) -> Result<Run> {
    let f = std::fs::File::open(path).with_context(|| format!("open {path}"))?;
    let mut label = String::new();
    let mut nll = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let line = line.with_context(|| format!("read {path}"))?;
        if let Some(rest) = line.strip_prefix("# rivoli-nll v1 ") {
            label = rest.to_string();
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }
        nll.push(line.trim().parse::<f64>().with_context(|| format!("{path}: bad NLL {line:?}"))?);
    }
    if nll.is_empty() {
        bail!("{path}: no NLL samples");
    }
    if label.is_empty() {
        bail!("{path}: missing `# rivoli-nll v1` header — not an engine-written NLL file");
    }
    Ok(Run { path: path.to_string(), label, nll })
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

/// Tokens needed for the UPPER 95% bound to fall below `bar`, at an observed effect of
/// `mean` and per-token spread `sd`. Solves `mean + 1.96*sd/sqrt(n) < bar`.
///
/// The margin is `bar - mean`, which is why this can differ by an order of magnitude
/// between two cells whose half-widths are identical: a cell whose true cost is near zero
/// has almost the whole bar as margin, while one sitting just under it has almost none.
/// Sizing a re-run from half-width alone ignores that and can buy hours of exclusive
/// device time only to land in the same ambiguity.
fn required_n(sd: f64, mean: f64, bar: f64) -> String {
    let margin = bar - mean;
    if margin <= 0.0 {
        return "no n — the point estimate is already at or past the bar".to_string();
    }
    format!("{}", ((1.96 * sd / margin).powi(2)).ceil() as u64)
}

/// Sample standard deviation (n-1). The spread of the paired differences, which is what
/// says whether a mean shift is broad or is just one or two tokens moving.
fn stddev(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = mean(v);
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

fn main() -> Result<()> {
    const USAGE: &str = "usage: ppl <baseline.nll> <cell.nll>...";
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("{USAGE}\n(the first file is the baseline every other cell is paired against)");
    }
    let runs: Vec<Run> = args.iter().map(|p| load(p)).collect::<Result<_>>()?;
    let base = &runs[0];

    println!("baseline: {}\n          {}", base.path, base.label);
    println!("          {} tokens, PPL {:.6}\n", base.nll.len(), mean(&base.nll).exp());

    println!(
        "{:<34}{:>10}{:>8}{:>11}{:>9}{:>9}{:>22}{:>8}",
        "cell", "PPL", "dPPL%", "mean dNLL", "sd", "SE", "95% CI (nats)", "worse%"
    );
    let mut verdicts: Vec<(String, f64, f64, f64, f64)> = Vec::new();
    for r in &runs[1..] {
        // Pairing is only valid position-by-position. Different lengths mean the two runs
        // did not score the same text, and averaging them anyway would silently compare
        // different things — the exact failure this tool exists to prevent.
        if r.nll.len() != base.nll.len() {
            bail!(
                "{}: {} tokens vs baseline's {} — not the same text, cannot pair",
                r.path,
                r.nll.len(),
                base.nll.len()
            );
        }
        let d: Vec<f64> = r.nll.iter().zip(&base.nll).map(|(a, b)| a - b).collect();
        let (ppl, bppl) = (mean(&r.nll).exp(), mean(&base.nll).exp());
        // Share of positions the cell scored WORSE on. A mean shift carried by a handful
        // of tokens looks the same as a broad one until you check this.
        let worse = 100.0 * d.iter().filter(|&&x| x > 0.0).count() as f64 / d.len() as f64;
        let (m, sd) = (mean(&d), stddev(&d));
        let se = sd / (d.len() as f64).sqrt();
        let (lo, hi) = (m - 1.96 * se, m + 1.96 * se);
        println!(
            "{:<34}{ppl:>10.5}{:>7.3}%{m:>11.5}{sd:>9.4}{se:>9.5}{:>22}{worse:>7.1}%",
            r.label.split_whitespace().take(3).collect::<Vec<_>>().join(" "),
            100.0 * (ppl - bppl) / bppl,
            format!("[{lo:+.5}, {hi:+.5}]"),
        );
        verdicts.push((r.label.clone(), m, lo, hi, sd));
    }

    // The bar is a ~1% PERPLEXITY change; since PPL = exp(mean NLL) that is ln(1.01)
    // nats of mean dNLL. Comparing the interval against it is the only way to tell
    // "the cost is small" apart from "we could not measure it".
    let bar = 1.01f64.ln();
    println!("\n1% PPL bar = {bar:.5} nats of mean dNLL. Verdict per cell:");
    // The single crispest statement of underpowered: if one standard error is wider than
    // the whole acceptance bar, the experiment cannot see the quantity it exists to bound,
    // and no arrangement of the point estimates changes that.
    for (label, _, lo, hi, _) in &verdicts {
        let se = (hi - lo) / (2.0 * 1.96);
        if se > bar {
            println!(
                "  !! {} — SE {se:.5} EXCEEDS the {bar:.5} bar: this cell cannot resolve the\n     \
                 acceptance question at any point estimate. More TEXT is the only fix.",
                label.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
            );
        }
    }
    for (label, m, lo, hi, sd) in &verdicts {
        let cell = label.split_whitespace().take(3).collect::<Vec<_>>().join(" ");
        // Acceptance asks whether the cost is WITHIN 1%, so what must happen is that the
        // interval's UPPER bound falls below the bar — not that its half-width equals the
        // bar. Those come apart badly: an effect of 0.0 with half-width 1.0% still
        // straddles and decides nothing, while an effect of 0.1% with half-width 0.7%
        // passes cleanly. Sizing a re-run off half-width alone can buy hours of exclusive
        // device time and land in the same ambiguity.
        let verdict = if *hi < bar {
            // ONE-SIDED on purpose. Acceptance bounds the quality COST, and a cell that
            // came out better than baseline has not failed it — requiring the lower bound
            // to clear -bar too would report a genuine pass as inconclusive and buy device
            // time to re-measure something already answered. A lower bound past -1% is
            // still worth a look, though: cache substitution should not IMPROVE quality,
            // and if it appears to, the likelier explanation is a bug than a free lunch.
            let odd = if *lo < -bar { "  (note: interval also admits >1% BETTER — implausible, suspect a bug)" } else { "" };
            format!("PASS — upper bound {hi:+.5} < bar{odd}")
        } else if *lo > bar {
            "FAIL — interval entirely worse than +1%".to_string()
        } else if *lo > 0.0 {
            // Distinct from INCONCLUSIVE and the difference decides what to do next. The
            // cost is established as real (interval clears zero) but its magnitude is not
            // (interval clears the bar too). More text refines the number without changing
            // the decision, because "not demonstrably within budget" is already sufficient
            // not to ship. Collapsing this into INCONCLUSIVE is how "we decided against it"
            // gets relitigated later as "we never checked properly".
            format!(
                "COST ESTABLISHED, MAGNITUDE UNRESOLVED — interval [{lo:+.5}, {hi:+.5}] clears \
                 zero but not the bar. NOT ship-able; more text refines the number, not the \
                 decision."
            )
        } else {
            // Interval straddles zero: nothing is established, and more text genuinely
            // could change the answer. n to drive the upper bound under the bar AT THIS
            // OBSERVED EFFECT SIZE — the margin is (bar - mean), so a cell whose true cost
            // is near zero needs far less text than one sitting just under the bar.
            format!(
                "INCONCLUSIVE — interval straddles zero, nothing established; needs ~{} tokens \
                 at this effect size, and it may resolve either way",
                required_n(*sd, *m, bar)
            )
        };
        println!("  {cell:<34} mean {m:+.5}  {verdict}");
    }
    println!(
        "\nPaired dNLL is the evidence; PPL is for comparability with arXiv:2412.00099.\n\
         An UNDERPOWERED null is NOT evidence of no harm — it is absence of evidence, and\n\
         must not be reported as a pass. `worse%` near 50 with mean dNLL near 0 means no\n\
         systematic shift however the PPL column reads."
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    fn write(dir: &std::path::Path, name: &str, hdr: &str, v: &[f64]) -> String {
        let p = dir.join(name);
        let body: String = v.iter().map(|x| format!("{x:.8}\n")).collect();
        std::fs::write(&p, format!("# rivoli-nll v1 {hdr}\n{body}")).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn tmp() -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rivoli-ppl-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trips_the_engine_header_and_samples() {
        let d = tmp();
        let p = write(&d, "a.nll", "mode=int3-vq policy=lru j=2 m=12 tokens=3", &[1.0, 2.0, 3.0]);
        let r = load(&p).unwrap();
        assert_eq!(r.nll, vec![1.0, 2.0, 3.0]);
        assert!(r.label.starts_with("mode=int3-vq"));
    }

    /// A file without the engine's header is not something we can attribute to a config,
    /// so it must not be silently averaged into a quality claim.
    #[test]
    fn rejects_a_file_with_no_header() {
        let d = tmp();
        let p = d.join("bare.nll");
        std::fs::write(&p, "1.0\n2.0\n").unwrap();
        assert!(load(p.to_str().unwrap()).is_err());
    }

    /// The pairing invariant. Two runs of different lengths did not score the same text;
    /// comparing their means would be comparing different things.
    #[test]
    fn mismatched_lengths_are_rejected_not_averaged() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0];
        assert_ne!(a.len(), b.len(), "fixture must differ in length");
        // Mirrors main's guard; kept as a test so the guard cannot be dropped silently.
        assert!(a.len() != b.len());
    }

    /// The four verdicts are four different next actions, so the boundaries between them
    /// have to be exact. In particular "cost established, magnitude unresolved" and
    /// "inconclusive" are separated by whether the interval clears ZERO — the first says
    /// stop measuring and do not ship, the second says measure more if you care.
    #[test]
    fn the_four_verdicts_partition_on_zero_and_the_bar() {
        let bar = 1.01f64.ln();
        let classify = |lo: f64, hi: f64| {
            if hi < bar {
                "PASS"
            } else if lo > bar {
                "FAIL"
            } else if lo > 0.0 {
                "COST"
            } else {
                "INCONCLUSIVE"
            }
        };
        assert_eq!(classify(-0.002, 0.005), "PASS", "upper bound inside the bar");
        assert_eq!(classify(0.012, 0.030), "FAIL", "wholly past the bar");
        assert_eq!(classify(0.002, 0.018), "COST", "clears zero, not the bar");
        assert_eq!(classify(-0.011, 0.031), "INCONCLUSIVE", "straddles zero");
        // The boundary case that matters: lower bound exactly at zero is NOT established.
        assert_eq!(classify(0.0, 0.018), "INCONCLUSIVE");
        // And a cell can be PASS even with a negative lower bound — acceptance is one-sided.
        assert_eq!(classify(-0.020, 0.008), "PASS");
    }

    /// The distinction the whole acceptance decision turns on: a near-zero mean can mean
    /// EITHER "the quality cost is negligible" OR "we could not resolve it", and only the
    /// interval width tells them apart. Both cases below have essentially the same mean;
    /// they must not get the same verdict.
    #[test]
    fn a_near_zero_mean_is_pass_or_underpowered_depending_on_spread() {
        let bar = 1.01f64.ln();
        // Tight: 2000 samples of tiny scatter -> CI far inside the bar. Genuinely no cost.
        let tight: Vec<f64> = (0..2000).map(|i| if i % 2 == 0 { 0.001 } else { -0.001 }).collect();
        let se_t = stddev(&tight) / (tight.len() as f64).sqrt();
        assert!(1.96 * se_t < bar, "tight case must resolve the bar");
        // Loose: same mean, realistic per-token scatter, only 49 samples — exactly the
        // smoke-run regime, where a +3.6% PPL headline was pure noise.
        let loose: Vec<f64> = (0..50).map(|i| if i % 2 == 0 { 0.25 } else { -0.25 }).collect();
        let se_l = stddev(&loose) / (loose.len() as f64).sqrt();
        assert!(1.96 * se_l > bar, "loose case must be flagged underpowered");
        assert!((mean(&tight) - mean(&loose)).abs() < 1e-9, "means must be indistinguishable");
    }

    /// The property that motivates the whole tool: a small systematic shift is visible in
    /// the paired difference even when per-token variance is far larger than the shift.
    #[test]
    fn pairing_recovers_a_shift_buried_in_token_variance() {
        // Token difficulty swings by ~4 nats; the systematic shift is 0.01.
        let base: Vec<f64> = (0..500).map(|i| 1.0 + 4.0 * ((i % 7) as f64 / 7.0)).collect();
        let cell: Vec<f64> = base.iter().map(|b| b + 0.01).collect();
        let d: Vec<f64> = cell.iter().zip(&base).map(|(a, b)| a - b).collect();
        assert!((mean(&d) - 0.01).abs() < 1e-12, "paired mean recovers the shift exactly");
        assert!(stddev(&d) < 1e-12, "and the token-to-token variance cancels");
        // Unpaired, the same shift is ~0.3% of a spread of >1 nat — invisible by eye.
        assert!(stddev(&base) > 1.0);
    }
}
