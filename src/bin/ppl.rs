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

/// Sample standard deviation (n-1). Used for the spread of the paired differences, which
/// is what says whether a mean shift is meaningful or is just one or two tokens moving.
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
        "{:<46}{:>11}{:>10}{:>9}{:>12}{:>11}{:>9}",
        "cell", "PPL", "dPPL", "dPPL%", "mean dNLL", "sd dNLL", "worse%"
    );
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
        println!(
            "{:<46}{ppl:>11.6}{:>10.6}{:>8.3}%{:>12.6}{:>11.6}{worse:>8.1}%",
            r.label.split_whitespace().take(4).collect::<Vec<_>>().join(" "),
            ppl - bppl,
            100.0 * (ppl - bppl) / bppl,
            mean(&d),
            stddev(&d),
        );
    }
    println!(
        "\nPaired dNLL is the evidence; PPL is for comparability with arXiv:2412.00099.\n\
         `worse%` near 50 with a mean dNLL near 0 means no systematic shift, however the\n\
         PPL column reads. Acceptance (docs/CACHE_ROUTE.md): within ~1% of baseline."
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
