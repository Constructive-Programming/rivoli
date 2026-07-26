//! Offline cache-policy A/B and 2Q Kin/Kout sweep. Replays a routed-expert access
//! trace (captured with `rivoli --trace`) through the SAME byte-aware policies the
//! engine runs ([`rivoli::hybrid`]) at a chosen slot count — unit strides make the byte
//! budget a plain slot count — and prints the residency (loaded %). Pure CPU,
//! milliseconds: compare policies without a full GPU decode run.
//!
//! usage: replay <trace> <n_slots> [--kin <pct>] [--kout <pct>] [--sweep]

use anyhow::{Context, Result, bail};
use rivoli::cache::TwoQSplit;
use rivoli::hybrid::{self, HybridPolicy};
use std::io::BufRead;

/// The Kin/Kout grid a `--sweep` walks. Kin is a resident probation bound so it is
/// swept fine-grained around the default; Kout is a key-only ghost (4 bytes an entry),
/// cheap enough to sweep well past 100 % of capacity.
const KIN_GRID: [u32; 11] = [3, 5, 8, 12, 16, 20, 25, 33, 40, 50, 66];
const KOUT_GRID: [u32; 6] = [25, 50, 100, 200, 400, 800];

/// Replay `trace` (each inner Vec = one MoE layer's demand keys) through `policy` at
/// `cap` unit slots; return the `loaded` (resident-hit) count. Two-pass per layer —
/// hits first (protected), then misses — mirroring the pin's `submit_layer`.
fn loaded_count(policy: &str, cap: usize, split: TwoQSplit, trace: &[Vec<u32>]) -> Result<u64> {
    // Unit strides: budget `cap` bytes == `cap` slots, cold==hot so the split is by
    // slot count exactly as the single-format engine sees it.
    let mut p: Box<dyn HybridPolicy> =
        hybrid::make(policy, cap, 1, 1, split).with_context(|| format!("unknown policy {policy}"))?;
    let mut loaded = 0u64;
    let mut miss: Vec<u32> = Vec::new();
    for layer in trace {
        p.begin_batch(); // one MoE layer = one batch (mirrors the pin)
        miss.clear();
        for &k in layer {
            if p.get(k) {
                loaded += 1;
                p.protect(k);
            } else {
                miss.push(k);
            }
        }
        for &k in &miss {
            p.admit(k);
        }
    }
    Ok(loaded)
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

    // Load the trace: one MoE layer per line, whitespace-separated demand keys. (A
    // legacy `| predicted` tail from the retired prefetch era is ignored.)
    let f = std::fs::File::open(&trace_path).with_context(|| format!("open trace {trace_path}"))?;
    let trace: Vec<Vec<u32>> = std::io::BufReader::new(f)
        .lines()
        .map(|l| {
            l.map(|l| {
                let demand = l.split('|').next().unwrap_or("");
                demand.split_whitespace().filter_map(|t| t.parse().ok()).collect::<Vec<u32>>()
            })
        })
        .collect::<std::io::Result<Vec<_>>>()
        .context("read trace")?
        .into_iter()
        .filter(|l| !l.is_empty())
        .collect();

    let accesses: usize = trace.iter().map(|l| l.len()).sum();
    let uniq = trace
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::HashSet<u32>>()
        .len();
    println!(
        "trace {trace_path}: {} layers, {accesses} accesses, {uniq} unique experts, cap={cap}",
        trace.len()
    );
    let pct = |n: u64| 100.0 * n as f64 / accesses.max(1) as f64;

    if sweep {
        println!(
            "\n2Q Kin/Kout sweep at cap={cap} — cells are `loaded %`. Kin bounds the A1in\n\
             probation queue, Kout the A1out ghost; both are percentages of capacity.\n"
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
                let loaded = pct(loaded_count("2q", cap, s, &trace)?);
                if loaded > best.1 {
                    best = (s, loaded);
                }
                print!("{loaded:>8.2}%");
            }
            println!();
        }
        let base = pct(loaded_count("2q", cap, default, &trace)?);
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

    println!("\n{:<10} {:>8}", "policy", "loaded");
    for pol in ["lru", "2q", "arc"] {
        println!("{pol:<10} {:>7.2}%", pct(loaded_count(pol, cap, split, &trace)?));
    }
    if split != default {
        println!("\n(2q ran with kin {kin}% / kout {kout}%)");
    }
    Ok(())
}
