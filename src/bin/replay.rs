//! Offline cache-policy A/B and 2Q Kin/Kout sweep. Replays a routed-expert access
//! trace (captured with `rivoli --trace`) through LRU / 2Q / ARC at a chosen slot
//! count and prints the `loaded / preloading / cold` split — cold-start, and (with
//! `--seed`) with the protected segment pre-filled from `.coli_usage`. Pure CPU,
//! milliseconds: the whole point is to compare policies without ~90s GPU decode runs.
//!
//! **Prefetch fidelity.** The live engine's residency comes mostly from *prefetch
//! admission*: `prefetch_layer` cold-admits L+1's predicted experts into 2Q's A1in
//! probation, and their first real `get` promotes them into the protected Am set.
//! A trace therefore only replays faithfully if it records those predictions. Traces
//! captured after 2026-07-21 do (` | ` then the predicted keys); older ones, and any
//! `--no-prefetch` capture, do not — for those this tool prints a warning and its
//! numbers describe a NO-PREFETCH engine, which measures ~70 % residency where the
//! real one measures ~92 %. Do not tune Kin/Kout against a prediction-less trace.
//!
//! usage: replay <trace> <n_slots> [--seed <snapshot_dir>] [--kin <pct>] [--kout <pct>] [--sweep]

use anyhow::{Context, Result, bail};
use rivoli::cache::{Layer, Residency, TwoQSplit, replay};
use std::io::BufRead;

/// Same packing as pin::expert_key — `(layer,expert)` into one u32.
fn key(l: usize, e: usize) -> u32 {
    ((l as u32) << 16) | e as u32
}

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
/// swept fine-grained around the 25 % default; Kout is a key-only ghost (4 bytes an
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
    const USAGE: &str =
        "usage: replay <trace> <n_slots> [--seed <dir>] [--kin <pct>] [--kout <pct>] [--sweep]";
    let mut args = std::env::args().skip(1);
    let trace_path = args.next().context(USAGE)?;
    let cap: usize = args
        .next()
        .context("n_slots required")?
        .parse()
        .context("n_slots must be an integer")?;
    let (mut seed_dir, mut sweep) = (None, false);
    let default = TwoQSplit::default();
    let (mut kin, mut kout) = (default.kin_pct(), default.kout_pct());
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seed" => seed_dir = Some(args.next().context("--seed needs a snapshot dir")?),
            "--sweep" => sweep = true,
            "--kin" => {
                kin = args
                    .next()
                    .context("--kin needs a percentage")?
                    .parse()
                    .context("--kin takes an integer percentage")?;
            }
            "--kout" => {
                kout = args
                    .next()
                    .context("--kout needs a percentage")?
                    .parse()
                    .context("--kout takes an integer percentage")?;
            }
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
    if predictions == 0 {
        println!(
            "\n!! WARNING: this trace records NO prefetch predictions, so `insert_cold` is\n\
             !! never exercised and 2Q's probation split has nothing to promote from.\n\
             !! These numbers describe a NO-PREFETCH engine (~70 % residency), not the\n\
             !! shipping one (~92 %). Do NOT tune Kin/Kout against them — recapture with\n\
             !! `rivoli <snap> -bench 512 --pre-seed --trace <path>` (prefetch left on)."
        );
    } else {
        println!(
            "prefetch: {predictions} predictions recorded ({:.2}/layer) — insert_cold modelled",
            predictions as f64 / trace.len().max(1) as f64
        );
    }

    // Optional seed: the top-`cap` experts by `.coli_usage` frequency (same stale
    // filter the live pin applies), placed in each policy's protected segment.
    let seed: Vec<u32> = match &seed_dir {
        Some(dir) => {
            let mc = rivoli::model::ModelConfig::load(dir)?;
            let usage = rivoli::usage::Usage::load(dir)?;
            let s: Vec<u32> = usage
                .ranked()
                .into_iter()
                .filter_map(|((l, e), _)| {
                    let (l, e) = (l as usize, e as usize);
                    (l >= mc.dense_layers && l < mc.n_layers && e < mc.n_experts).then(|| key(l, e))
                })
                .take(cap)
                .collect();
            println!("seed: {} experts from .coli_usage", s.len());
            s
        }
        None => Vec::new(),
    };

    let run = |pol: &str, split: TwoQSplit| -> Result<Residency> {
        replay(pol, cap, split, &seed, &trace).with_context(|| format!("unknown policy {pol}"))
    };

    if sweep {
        println!(
            "\n2Q Kin/Kout sweep at cap={cap} — cells are `loaded %`, the only I/O-free\n\
             metric (see benchmarks.md). Kin bounds the A1in probation queue, Kout the\n\
             A1out ghost; both are percentages of capacity.\n"
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
    for pol in ["lru", "2q", "arc", "wtlfu"] {
        print_row(pol, run(pol, split)?);
    }
    if split != default {
        println!("\n(2q ran with kin {kin}% / kout {kout}%)");
    }
    Ok(())
}
