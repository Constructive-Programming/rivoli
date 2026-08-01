//! Offline cache-policy A/B and 2Q Kin/Kout sweep. Replays a routed-expert access trace
//! (captured with `rivoli --trace`) through the SAME byte-aware policies the engine runs
//! ([`rivoli::memory::hybrid`]) at a chosen slot count — unit strides make the byte budget
//! a plain slot count — and prints the residency (loaded %). Pure CPU, milliseconds:
//! compare policies without a full GPU decode run.
//!
//! A **v2** trace's `| key:score ...` tail — the ranked candidate window per routing
//! decision — is read past and dropped. It fed two models this tool no longer carries: the
//! `top-m` (J, M) substitution grid and the oracle-prefetch ceiling, both removed
//! 2026-08-01 along with the header parse and window validation they required. Each
//! modelled a mechanism the engine does not have and both investigations closed negative
//! (docs/investigations/cache-conditional-routing.md, docs/investigations/cross-layer-prefetch.md);
//! every cell they printed is tabulated in docs/measurement/benchmarks.md, and the code is
//! at git tag `archive/replay-oracle-prefetch`. Dropping the tail is what keeps a v2 trace
//! readable here at all.

use anyhow::{Context, Result};
use clap::Parser;
use rivoli::memory::cache::TwoQSplit;
use rivoli::memory::hybrid::{self, HybridPolicy};
use std::io::BufRead;

/// The Kin/Kout grid a `--sweep` walks. Kin is a resident probation bound so it is
/// swept fine-grained around the default; Kout is a key-only ghost (4 bytes an entry),
/// cheap enough to sweep well past 100 % of capacity.
const KIN_GRID: [u32; 11] = [3, 5, 8, 12, 16, 20, 25, 33, 40, 50, 66];
const KOUT_GRID: [u32; 6] = [25, 50, 100, 200, 400, 800];

/// One routing decision (one MoE layer of one token): the keys the engine actually looked
/// up, in router RANK order (they come straight from `topk_into`). The order is load-
/// bearing — the policies below are recency-sensitive, so they see it as an access
/// sequence, not a set.
type Decision = Vec<u32>;

// clap derives the parse and the help text from this struct — the same switch src/main.rs
// made, for the same reason: the hand-rolled `std::env::args()` loop this replaced kept a
// `USAGE` const maintained separately from the flags it described, and a second source of
// truth for a flag list only ever drifts apart from the first.
//
// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
// Rationale for the code goes in `//` comments like this one, which clap ignores. The
// struct itself carries no doc comment on purpose: `about` overrides it, so it would be a
// second description of this binary that nothing ever prints.
#[derive(Parser)]
#[command(name = "replay", about = "Offline cache-policy A/B and 2Q Kin/Kout sweep over a routing trace")]
struct Args {
    /// The routed-expert access trace to replay, captured with `rivoli --trace`. v1 and v2
    /// traces both read here; a v2 candidate-window tail is ignored.
    trace: String,

    /// Cache capacity, in unit slots. Strides are unit in this sim, so the engine's byte
    /// budget is a plain slot count.
    n_slots: usize,

    /// Walk the 2Q Kin/Kout grid at this capacity instead of printing the policy A/B.
    #[arg(long)]
    sweep: bool,

    /// 2Q only: the A1in probation bound, as a percentage of capacity.
    #[arg(long, value_name = "PCT", default_value_t = TwoQSplit::default().kin_pct())]
    kin: u32,

    /// 2Q only: the A1out ghost bound, as a percentage of capacity. May exceed 100 — the
    /// ghost holds keys only.
    #[arg(long, value_name = "PCT", default_value_t = TwoQSplit::default().kout_pct())]
    kout: u32,
}

/// Parse a trace into its demand keys. v1 lines are bare whitespace-separated keys; v2
/// lines add a `| key:score ...` ranked candidate window, dropped here (see the module
/// header for what read it and why that went).
///
/// Lenient by construction, and that is what lets one reader take both formats: a token
/// that does not parse as a `u32` is skipped, so the `# rivoli-trace v2 ...` header, a
/// comment and a blank line all yield an empty decision and are dropped.
fn parse_trace(r: impl BufRead) -> Result<Vec<Decision>> {
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line.context("read trace")?;
        let head = line.split_once('|').map_or(line.as_str(), |(h, _)| h);
        let demand: Decision = head.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if !demand.is_empty() {
            out.push(demand);
        }
    }
    Ok(out)
}

/// Replay `trace` through `policy` at `cap` unit slots; return the `loaded` (resident-hit)
/// count over the demand accesses. Two-pass per layer — hits first (protected), then
/// misses — mirroring the pin's `submit_layer`.
fn replay(policy: &str, cap: usize, split: TwoQSplit, trace: &[Decision]) -> Result<u64> {
    // Unit strides: budget `cap` bytes == `cap` slots, cold==hot so the split is by
    // slot count exactly as the single-format engine sees it.
    let mut p: Box<dyn HybridPolicy> = hybrid::make(policy, cap, 1, 1, split)
        .with_context(|| format!("unknown policy {policy}"))?;
    let mut loaded = 0u64;
    let mut miss: Vec<u32> = Vec::new();
    for d in trace {
        p.begin_batch(); // one MoE layer = one batch (mirrors the pin)
        miss.clear();
        for &k in d {
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
    let args = Args::parse();
    let cap = args.n_slots;
    let default = TwoQSplit::default();
    let split = TwoQSplit::new(args.kin, args.kout)?;

    let f = std::fs::File::open(&args.trace)
        .with_context(|| format!("open trace {}", args.trace))?;
    let trace = parse_trace(std::io::BufReader::new(f))?;

    let accesses: usize = trace.iter().map(Vec::len).sum();
    let uniq = trace
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::HashSet<u32>>()
        .len();
    println!(
        "trace {}: {} layers, {accesses} accesses, {uniq} unique experts, cap={cap}",
        args.trace,
        trace.len()
    );
    // Every replay is scored the same way: demand accesses served from residency.
    let pct = |loaded: u64| 100.0 * loaded as f64 / accesses.max(1) as f64;

    if args.sweep {
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
                let loaded = pct(replay("2q", cap, s, &trace)?);
                if loaded > best.1 {
                    best = (s, loaded);
                }
                print!("{loaded:>8.2}%");
            }
            println!();
        }
        let base = pct(replay("2q", cap, default, &trace)?);
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
        println!("{pol:<10} {:>7.2}%", pct(replay(pol, cap, split, &trace)?));
    }
    // Residency curve, ALWAYS printed. Two jobs, both learned the hard way.
    //
    // 1. It shows how steeply hit rate depends on capacity. Reading a single cap invites
    //    the assumption that the curve is flat there — that slots are not the binding
    //    constraint — and on this workload that assumption is badly wrong (+23% slots is
    //    worth ~6pp). It also converts a policy's pp gain into "how many slots would buy
    //    the same", which is the currency the engine actually thinks in.
    // 2. It makes CROSS-MODE comparison honest. Every mode decodes its own trajectory, so
    //    the traces are different workloads and each mode's own slot count is a different
    //    capacity — comparing two modes at their native caps conflates the two effects
    //    and silently attributes one to the other. Compare at a shared cap, from here.
    println!("\nresidency curve (lru) — hit% vs capacity, for reading any single cap in context");
    print!("{:<10}", "slots");
    let curve: Vec<usize> = [cap / 2, cap * 3 / 4, cap, cap * 3 / 2, cap * 2]
        .into_iter()
        .filter(|&c| c > 0)
        .collect();
    for c in &curve {
        print!("{c:>10}");
    }
    print!("\n{:<10}", "hit%");
    for &c in &curve {
        print!("{:>9.2}%", pct(replay("lru", c, split, &trace)?));
    }
    println!(
        "\n(cross-mode readings MUST use the same slot count — each mode's native capacity\n\
         differs AND its trace is a different workload, so native-cap comparisons conflate them.)"
    );
    if split != default {
        println!("\n(2q ran with kin {}% / kout {}%)", args.kin, args.kout);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    fn parse(s: &str) -> Vec<Decision> {
        parse_trace(std::io::BufReader::new(s.as_bytes())).expect("parse")
    }

    /// The leniency the format relies on, in one file: a v2 header and a blank line must
    /// vanish rather than become decisions, a v2 line must yield exactly its demand keys
    /// with the candidate-window tail dropped, and a v1 line mixed in must still parse as
    /// a bare demand list. That last pair is the compat property the residency sim rests
    /// on — one reader, both formats, identical demand-key sequences.
    #[test]
    fn parser_is_lenient_in_the_ways_the_format_relies_on() {
        let t = parse(
            "# rivoli-trace v2 top_k=2 window=4\n\n\
             7 8 | 7:0.9 8:0.8 9:0.7 10:0.6\n\
             9 10\n",
        );
        assert_eq!(t, vec![vec![7, 8], vec![9, 10]]);
    }
}
