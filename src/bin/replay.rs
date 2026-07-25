//! Offline cache-policy A/B and 2Q Kin/Kout sweep. Replays a routed-expert access
//! trace (captured with `rivoli --trace`) through LRU / 2Q / ARC at a chosen slot
//! count and prints the residency split. Pure CPU, milliseconds: the whole point is
//! to compare policies without a full GPU decode run.
//!
//! The engine no longer prefetches (an A/B showed it bought zero throughput on the
//! bandwidth-bound VQ path — 78.3% vs 76.9% hit, identical tok/s), so traces carry
//! no predictions and `preloading` is always 0; `loaded` is the residency metric.
//! The `preloading` column and the prediction (` | `) parsing are retained only so
//! historical prefetch-era traces still replay.
//!
//! usage: replay <trace> <n_slots> [--kin <pct>] [--kout <pct>] [--sweep]

use anyhow::{Context, Result, bail};
use rivoli::cache::{Layer, Residency, TwoQSplit, replay};
use std::io::BufRead;

/// One captured MoE layer, owning its keys (`Layer` borrows from these).
struct Captured {
    demand: Vec<u32>,
    predicted: Vec<u32>,
}

/// Parse `k1 k2 k3 | p1 p2` (the ` | ` tail optional) into demand + predicted keys.
fn parse_line(line: &str) -> Captured {
    let (d, p) = line.split_once('|').unwrap_or((line, ""));
    let keys = |s: &str| {
        s.split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect()
    };
    Captured {
        demand: keys(d),
        predicted: keys(p),
    }
}

/// The Kin/Kout grid a `--sweep` walks. Kin is a resident probation bound so it is
/// swept fine-grained around the default; Kout is a key-only ghost (4 bytes an
/// entry), cheap enough to sweep well past 100 % of capacity.
const KIN_GRID: [u32; 11] = [3, 5, 8, 12, 16, 20, 25, 33, 40, 50, 66];
const KOUT_GRID: [u32; 6] = [25, 50, 100, 200, 400, 800];

fn print_row(label: &str, r: Residency) {
    let pct = |n: u64| 100.0 * n as f64 / r.accesses().max(1) as f64;
    println!(
        "{label:<10} {:>8.2}% {:>11.2}% {:>8.2}% {:>8.2}%",
        r.loaded_pct(),
        pct(r.preloading),
        pct(r.cold),
        r.hit_pct(),
    );
}

fn main() -> Result<()> {
    const USAGE: &str = "usage: replay <trace> <n_slots> [--kin <pct>] [--kout <pct>] [--sweep]";
    let mut args = std::env::args().skip(1);
    let trace_path = args.next().context(USAGE)?;
    let cap: usize = args
        .next()
        .context("n_slots required")?
        .parse()
        .context("n_slots must be an integer")?;
    let mut sweep = false;
    let default = TwoQSplit::default();
    let (mut kin, mut kout) = (default.kin_pct(), default.kout_pct());
    while let Some(a) = args.next() {
        let mut val = |what: &str| args.next().with_context(|| format!("{what} needs a value"));
        match a.as_str() {
            "--sweep" => sweep = true,
            "--kin" => kin = val("--kin")?.parse().context("--kin takes an integer percentage")?,
            "--kout" => kout = val("--kout")?.parse().context("--kout takes an integer percentage")?,
            other => bail!("unexpected arg {other}\n{USAGE}"),
        }
    }
    let split = TwoQSplit::new(kin, kout)?;

    // Load the trace: one MoE layer per line, `demand... [| predicted...]`.
    let f = std::fs::File::open(&trace_path).with_context(|| format!("open trace {trace_path}"))?;
    let captured: Vec<Captured> = std::io::BufReader::new(f)
        .lines()
        .map(|l| l.map(|l| parse_line(&l)))
        .collect::<std::io::Result<Vec<_>>>()
        .context("read trace")?
        .into_iter()
        .filter(|c| !c.demand.is_empty())
        .collect();
    let trace: Vec<Layer<'_>> = captured
        .iter()
        .map(|c| Layer {
            demand: &c.demand,
            predicted: &c.predicted,
        })
        .collect();

    let accesses: usize = captured.iter().map(|c| c.demand.len()).sum();
    let predictions: usize = captured.iter().map(|c| c.predicted.len()).sum();
    let uniq = captured
        .iter()
        .flat_map(|c| c.demand.iter().copied())
        .collect::<std::collections::HashSet<u32>>()
        .len();
    println!(
        "trace {trace_path}: {} layers, {accesses} accesses, {uniq} unique experts, cap={cap}",
        trace.len()
    );
    if predictions > 0 {
        // A historical prefetch-era trace: model the admission it recorded.
        println!(
            "prefetch: {predictions} predictions recorded ({:.2}/layer) — insert_cold modelled",
            predictions as f64 / trace.len().max(1) as f64
        );
    }
    // A prediction-less trace (the engine no longer prefetches) just means preloading
    // is 0 and `loaded` is the residency figure — no warning; that is now the norm.

    // No seed: the new artifact carries no `.coli_usage` frequency profile, so every
    // policy starts cold (online priming is what the trace measures anyway).
    let run = |pol: &str, split: TwoQSplit| -> Result<Residency> {
        replay(pol, cap, split, &[], &trace).with_context(|| format!("unknown policy {pol}"))
    };

    if sweep {
        println!(
            "\n2Q Kin/Kout sweep at cap={cap} — cells are `loaded %`, the only I/O-free\n\
             metric. Kin bounds the A1in probation queue, Kout the A1out ghost; both\n\
             are percentages of capacity.\n"
        );
        print!("{:<9}", "kin\\kout");
        for ko in KOUT_GRID {
            print!("{:>9}", format!("{ko}%"));
        }
        println!();
        let mut best = (default, f64::MIN);
        for ki in KIN_GRID {
            print!("{:<9}", format!("{ki}%"));
            for ko in KOUT_GRID {
                let s = TwoQSplit::new(ki, ko)?;
                let loaded = run("2q", s)?.loaded_pct();
                if loaded > best.1 {
                    best = (s, loaded);
                }
                print!("{loaded:>8.2}%");
            }
            println!();
        }
        let base = run("2q", default)?.loaded_pct();
        println!(
            "\ndefault (kin {}% / kout {}%): {base:.2}% loaded\n\
             best    (kin {}% / kout {}%): {:.2}% loaded  ({:+.2} pp)",
            default.kin_pct(),
            default.kout_pct(),
            best.0.kin_pct(),
            best.0.kout_pct(),
            best.1,
            best.1 - base,
        );
        return Ok(());
    }

    println!(
        "\n{:<10} {:>8} {:>11} {:>8} {:>8}",
        "policy", "loaded", "preloading", "cold", "hit"
    );
    for pol in ["lru", "2q", "arc"] {
        print_row(pol, run(pol, split)?);
    }
    if split != default {
        println!("\n(2q ran with kin {kin}% / kout {kout}%)");
    }
    Ok(())
}
