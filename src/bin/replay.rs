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

use anyhow::{Context, Result};
use rivoli::cache::{Cache, Layer, Residency, Tier, TwoQ, TwoQSplit, replay};
use rivoli::quant::{i4_expert_stride, vq_expert_stride};
use std::collections::HashMap;
use std::io::BufRead;

// GLM-5.2 MoE dims — this crate targets one model, so the expert slot sizes are
// fixed (ponytail: constants, not CLI knobs nobody varies).
const HIDDEN: usize = 6144;
const MOE_INTER: usize = 2048;
/// Measured pool budget at `--max-mem 115` (98.6 GiB left after the 16.4 GiB resident
/// tier). Overridable with `--budget-gib` for other budgets.
const DEFAULT_BUDGET_GIB: f64 = 98.6;
/// Compute cost of one vq3 (COLD) expert relative to an int4 (HOT) expert = 1.0.
/// From the dot microbench: int4 decodes ~1.8× faster than int3-VQ.
const COLD_COMPUTE: f64 = 1.8;

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

/// One split's confound-free result: the SAME trace, this `(n_cold, n_hot)` fixed
/// partition. Hits/misses split by the slab (format) that served them.
struct HybridResult {
    hot_pct: u32,
    n_cold: usize,
    n_hot: usize,
    hot_hit: u64,
    cold_hit: u64,
    miss_hot: u64,  // a miss whose insert landed (fetched) into the HOT/int4 slab
    miss_cold: u64, // ... into the COLD/vq3 slab
}
impl HybridResult {
    fn accesses(&self) -> u64 {
        self.hot_hit + self.cold_hit + self.miss_hot + self.miss_cold
    }
    fn hit_pct(&self) -> f64 {
        100.0 * (self.hot_hit + self.cold_hit) as f64 / self.accesses().max(1) as f64
    }
    /// Throughput proxy in int4-compute units (LOWER = faster). A resident hit costs
    /// its slab's compute (int4=1.0, vq3=[`COLD_COMPUTE`]); a miss adds `miss_penalty`
    /// (the exposed fetch + host-gated launch bubble a stream miss pays on top).
    fn cost(&self, miss_penalty: f64) -> f64 {
        self.hot_hit as f64
            + self.cold_hit as f64 * COLD_COMPUTE
            + self.miss_hot as f64 * (1.0 + miss_penalty)
            + self.miss_cold as f64 * (COLD_COMPUTE + miss_penalty)
    }
}

/// Replay `trace` through a fixed-partition 2Q with `n_cold`/`n_hot` slabs, mirroring
/// `Pin::submit_spine`: touch every hit first (protect it so a same-layer miss cannot
/// evict it), then admit the misses — each insert's tier picks its slab. `slab_of`
/// mirrors the pin's key→slab map (a hit stays in its slab; only an evict+refetch
/// migrates format), so hits/misses split cleanly by format. Confound-free: identical
/// trace for every split, so only the partition varies.
fn hybrid_replay(
    trace: &[Layer<'_>],
    hot_pct: u32,
    n_cold: usize,
    n_hot: usize,
    kout: usize,
) -> HybridResult {
    let mut policy = TwoQ::fixed(n_cold, n_hot, kout);
    let mut slab_of: HashMap<u32, bool> = HashMap::new(); // key -> is_hot
    let (mut hot_hit, mut cold_hit, mut miss_hot, mut miss_cold) = (0u64, 0u64, 0u64, 0u64);
    let mut misses: Vec<u32> = Vec::new();
    for layer in trace {
        misses.clear();
        for &k in layer.demand {
            if policy.get(k) {
                if slab_of[&k] {
                    hot_hit += 1;
                } else {
                    cold_hit += 1;
                }
                policy.protect(k);
            } else {
                misses.push(k);
            }
        }
        for &k in &misses {
            let (evicted, tier) = policy.insert(k);
            if let Some(ev) = evicted {
                slab_of.remove(&ev);
            }
            let is_hot = tier == Tier::Hot;
            slab_of.insert(k, is_hot);
            if is_hot {
                miss_hot += 1;
            } else {
                miss_cold += 1;
            }
        }
    }
    HybridResult { hot_pct, n_cold, n_hot, hot_hit, cold_hit, miss_hot, miss_cold }
}

/// Sweep `--hot-pct` over the fixed trace: for each split derive `(n_cold, n_hot)`
/// from the byte budget + the two slot strides, replay, print the format breakdown +
/// the cost proxy, and report the min-cost split. This is the confound-free optimum
/// the empirical bench can't give (each bench run generates a DIFFERENT trace).
fn run_hybrid(trace: &[Layer<'_>], budget_gib: f64, kout_pct: u32, miss_penalty: f64) {
    let vq3 = vq_expert_stride(HIDDEN, MOE_INTER);
    let i4 = i4_expert_stride(HIDDEN, MOE_INTER);
    let budget = (budget_gib * (1u64 << 30) as f64) as usize;
    println!(
        "\nhybrid sim: budget {budget_gib:.1} GiB, vq3 slot {:.1}MB / int4 slot {:.1}MB, \
         kout {kout_pct}%, cold_compute {COLD_COMPUTE}×, miss_penalty {miss_penalty}",
        vq3 as f64 / 1e6,
        i4 as f64 / 1e6,
    );
    println!(
        "{:>4} {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>6} {:>9}",
        "hot%", "nCold", "nHot", "hotHit", "coldHit", "missH", "missC", "hit%", "cost",
    );
    let mut best: Option<(HybridResult, f64)> = None;
    for hp in (20..=90).step_by(5) {
        let hot_bytes = budget * hp / 100;
        let n_hot = (hot_bytes / i4).max(1);
        let n_cold = (budget.saturating_sub(hot_bytes) / vq3).max(1);
        let kout = ((n_cold + n_hot) * kout_pct as usize / 100).max(1);
        let r = hybrid_replay(trace, hp as u32, n_cold, n_hot, kout);
        let cost = r.cost(miss_penalty);
        println!(
            "{:>3}% {:>7} {:>7} {:>8} {:>8} {:>7} {:>7} {:>5.1}% {:>9.0}",
            hp, r.n_cold, r.n_hot, r.hot_hit, r.cold_hit, r.miss_hot, r.miss_cold, r.hit_pct(), cost,
        );
        if best.as_ref().is_none_or(|(_, bc)| cost < *bc) {
            best = Some((r, cost));
        }
    }
    if let Some((b, cost)) = best {
        println!(
            "\noptimum: hot {}% ({} cold vq3 + {} hot int4), hit {:.1}%, cost {cost:.0}",
            b.hot_pct, b.n_cold, b.n_hot, b.hit_pct(),
        );
    }
}

fn main() -> Result<()> {
    const USAGE: &str = "usage: replay <trace> <n_slots> [--kin <pct>] [--kout <pct>] [--sweep]\n\
                         \x20      replay <trace> --hybrid [--budget-gib <f>] [--kout <pct>] [--miss-penalty <f>]";
    let mut args = std::env::args().skip(1);
    let trace_path = args.next().context(USAGE)?;
    let mut cap: Option<usize> = None;
    let mut sweep = false;
    let mut hybrid = false;
    let default = TwoQSplit::default();
    let (mut kin, mut kout) = (default.kin_pct(), default.kout_pct());
    let mut budget_gib = DEFAULT_BUDGET_GIB;
    let mut miss_penalty = 0.5;
    while let Some(a) = args.next() {
        let mut val = |what: &str| args.next().with_context(|| format!("{what} needs a value"));
        match a.as_str() {
            "--sweep" => sweep = true,
            "--hybrid" => hybrid = true,
            "--kin" => kin = val("--kin")?.parse().context("--kin takes an integer percentage")?,
            "--kout" => kout = val("--kout")?.parse().context("--kout takes an integer percentage")?,
            "--budget-gib" => budget_gib = val("--budget-gib")?.parse().context("--budget-gib takes a number")?,
            "--miss-penalty" => {
                miss_penalty = val("--miss-penalty")?.parse().context("--miss-penalty takes a number")?
            }
            other => {
                cap = Some(other.parse().with_context(|| format!("unexpected arg {other}\n{USAGE}"))?)
            }
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
    if hybrid {
        println!(
            "trace {trace_path}: {} layers, {accesses} accesses, {uniq} unique experts",
            trace.len()
        );
        run_hybrid(&trace, budget_gib, kout, miss_penalty);
        return Ok(());
    }
    let cap = cap.context("n_slots required (or pass --hybrid)")?;
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
