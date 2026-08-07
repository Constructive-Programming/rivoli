//! Offline cache-policy A/B and 2Q Kin/Kout sweep. Replays a routed-expert access trace
//! (captured with `rivoli --trace`) through the SAME byte-aware policies the engine runs
//! ([`rivoli::memory::hybrid`]) at a chosen slot count — unit strides make the byte budget
//! a plain slot count — and prints the residency (loaded %). Pure CPU, milliseconds:
//! compare policies without a full GPU decode run.
//!
//! It also prints [`replay_opt`], the Belady bound, which is what makes any of those rows
//! readable: until 2026-08-02 the online policies were only ever ranked against each other,
//! so a 2Q that beat LRU by 5 pp could equally have been 2 pp or 20 pp short of what the
//! trace allows. On a disk-bound engine (`docs/reference/architecture.md` §3: 181 ms of
//! transfer against 117 ms of compute) hit rate is one of only two levers that move bytes,
//! so knowing whether that lever is spent is worth more than another policy.
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
use std::collections::{BTreeMap, HashMap, HashSet};
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
#[command(
    name = "replay",
    about = "Offline cache-policy A/B and 2Q Kin/Kout sweep over a routing trace"
)]
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

    /// Model a BATCHED pass over this many consecutive tokens: each layer reads the union
    /// of its rows' picks once instead of once per row. 1 (the default) replays the trace
    /// as captured. The figure of merit under batching is `reads/token`, not `loaded %` —
    /// the union shrinks the access count itself, so hit rate flatters a larger batch.
    #[arg(long, value_name = "B", default_value_t = 1, value_parser = clap::value_parser!(u16).range(1..))]
    batch: u16,

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
        let demand: Decision = head
            .split_whitespace()
            .filter_map(|t| t.parse().ok())
            .collect();
        if !demand.is_empty() {
            out.push(demand);
        }
    }
    Ok(out)
}

/// Split a flat trace into per-token runs. `expert_key` packs the layer into the high 16
/// bits, and a token walks its MoE layers in increasing order, so a layer id that fails to
/// increase starts a new token. Derived rather than assumed: the MoE layer count is an
/// artifact property and a trace carries no header for it.
fn tokens(trace: &[Decision]) -> Vec<&[Decision]> {
    let mut cuts = vec![0usize];
    for (i, d) in trace.iter().enumerate().skip(1) {
        if d[0] >> 16 <= trace[i - 1][0] >> 16 {
            cuts.push(i);
        }
    }
    cuts.push(trace.len());
    cuts.windows(2).map(|w| &trace[w[0]..w[1]]).collect()
}

/// The access sequence a BATCHED pass over `b` consecutive tokens would issue: per layer,
/// the union of that layer's picks across the batch's rows, read once. This is the whole
/// question batched prefill turns on — a batch pays the union, not the sum, so the saving
/// is however much consecutive tokens agree about which experts they want.
///
/// Layers come back in id order, which is the order a batched pass visits them. Within a
/// layer the union keeps FIRST-SEEN order, preserving the router rank the policies are
/// recency-sensitive to (see [`Decision`]).
fn coalesce(trace: &[Decision], b: usize) -> Vec<Decision> {
    if b <= 1 {
        return trace.to_vec();
    }
    let mut out = Vec::new();
    for chunk in tokens(trace).chunks(b) {
        let mut by_layer: BTreeMap<u32, Decision> = BTreeMap::new();
        for d in chunk.iter().copied().flatten() {
            let u = by_layer.entry(d[0] >> 16).or_default();
            // ponytail: linear dedup over a union bounded by b*top_k (512 at b=64), which
            // is microseconds against a trace this tool already scans in milliseconds. A
            // HashSet beside it is the upgrade path if a batch ever wants to be large.
            for &k in d {
                if !u.contains(&k) {
                    u.push(k);
                }
            }
        }
        out.extend(by_layer.into_values());
    }
    out
}

/// Replay `trace` through `policy` at `cap` unit slots; return the `loaded` (resident-hit)
/// count over the demand accesses. Two-pass per layer — hits first (protected), then
/// misses — mirroring the pool's `RoutedPool::submit`.
fn replay(policy: &str, cap: usize, split: TwoQSplit, trace: &[Decision]) -> Result<u64> {
    // Unit strides: budget `cap` bytes == `cap` slots, cold==hot so the split is by
    // slot count exactly as the single-format engine sees it.
    let mut p: Box<dyn HybridPolicy> = hybrid::policy_for(policy, cap, 1, 1, split)
        .with_context(|| format!("unknown policy {policy}"))?;
    let mut loaded = 0u64;
    let mut miss: Vec<u32> = Vec::new();
    for d in trace {
        p.begin_batch(); // one MoE layer = one batch (mirrors the pin)
        miss.clear();
        for &k in d {
            if p.hit(k) {
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

/// Belady's OPT: the `loaded` count a CLAIRVOYANT evictor reaches on this trace at `cap`
/// slots. No online policy can beat it, so it is the ceiling the rows above are short of.
///
/// **Deliberately not a [`HybridPolicy`], and deliberately not in `hybrid::policy_for`.** It needs
/// the whole future, which the trait does not offer (and `replay` calls `get` for every hit
/// before `admit` for any miss, so a policy-side access counter would not even see trace
/// order). Registering it would also put `--cache-policy opt` in the engine's `--help` for
/// something the engine cannot run — the mistake `top-m` made in the other direction.
///
/// It carries the pin set anyway (`pinned`), which is NOT part of textbook OPT: a key hit
/// this batch has had its slot handed to a kernel and cannot be evicted before the batch
/// closes ("expert not resident after alloc"). Relaxing it would raise the bound by making
/// it unreachable by construction, which is the opposite of what a bound is for.
fn replay_opt(cap: usize, trace: &[Decision]) -> u64 {
    // Next-use index per access, walking the flattened sequence BACKWARDS: `last.insert`
    // hands back the previous value, which — going this direction — is exactly the next
    // occurrence of that key. `usize::MAX` = never referenced again, i.e. evict me first.
    let mut next: Vec<Vec<usize>> = trace.iter().map(|d| vec![usize::MAX; d.len()]).collect();
    let mut last: HashMap<u32, usize> = HashMap::new();
    let mut i: usize = trace.iter().map(Vec::len).sum();
    for (di, d) in trace.iter().enumerate().rev() {
        for (ki, &k) in d.iter().enumerate().rev() {
            i -= 1;
            next[di][ki] = last.insert(k, i).unwrap_or(usize::MAX);
        }
    }

    let mut resident: HashMap<u32, usize> = HashMap::new(); // key -> its next-use index
    let mut pinned: HashSet<u32> = HashSet::new();
    let mut loaded = 0u64;
    for (di, d) in trace.iter().enumerate() {
        // Pass 1 — score the batch and RE-KEY its hits to the use AFTER this one. Ranking a
        // key by the reference being served now would make every hit look maximally urgent
        // and evict the wrong victim. Pinning misses too is a no-op (they are not resident)
        // and saves a second membership test below.
        pinned.clear();
        for (ki, &k) in d.iter().enumerate() {
            if let Some(nu) = resident.get_mut(&k) {
                *nu = next[di][ki];
                loaded += 1;
            }
            pinned.insert(k);
        }
        // Pass 2 — admit the misses, evicting the farthest next use. Ties break on the key
        // so the bound is reproducible: `HashMap` iteration order is not stable across runs
        // and a number quoted in a doc has to be the same number tomorrow.
        for (ki, &k) in d.iter().enumerate() {
            if resident.contains_key(&k) {
                continue;
            }
            // Linear scan, ~3.6k slots × ~300k misses ≈ a second — a heap would need a
            // second structure kept in sync with `resident` for a tool that runs once.
            // ponytail: if a sweep ever wants OPT per cell, that heap is the upgrade path.
            while resident.len() >= cap {
                let victim = resident
                    .iter()
                    .filter(|(v, _)| !pinned.contains(*v))
                    .max_by_key(|&(v, &nu)| (nu, *v))
                    .map(|(v, _)| *v);
                // Same `None => break` as `hybrid::evict_until_fits`: a batch with no legal
                // victim admits over budget rather than spinning.
                match victim {
                    Some(v) => resident.remove(&v),
                    None => break,
                };
            }
            resident.insert(k, next[di][ki]);
        }
    }
    loaded
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cap = args.n_slots;
    let default = TwoQSplit::default();
    let split = TwoQSplit::new(args.kin, args.kout)?;

    let f =
        std::fs::File::open(&args.trace).with_context(|| format!("open trace {}", args.trace))?;
    let trace = parse_trace(std::io::BufReader::new(f))?;
    // Token count comes from the CAPTURED trace, before coalescing collapses the runs it
    // is derived from. It is the denominator of `reads/token`, which is the only figure
    // comparable across batch sizes — bytes moved per token of output is what the fetch
    // bound is denominated in (docs/reference/architecture.md §3).
    let ntok = tokens(&trace).len();
    let trace = coalesce(&trace, usize::from(args.batch));

    let accesses: usize = trace.iter().map(Vec::len).sum();
    let uniq = trace
        .iter()
        .flatten()
        .copied()
        .collect::<std::collections::HashSet<u32>>()
        .len();
    println!(
        "trace {}: {} decisions over {ntok} tokens (batch={}), {accesses} accesses, \
         {uniq} unique experts, cap={cap}",
        args.trace,
        trace.len(),
        args.batch,
    );
    // Every replay is scored the same way: demand accesses served from residency.
    let pct = |loaded: u64| 100.0 * loaded as f64 / accesses.max(1) as f64;
    // Misses are reads, and a read is ~15.34 MB off NVMe. Per TOKEN, because that is the
    // denominator the fetch bound uses and the only one a batch cannot flatter: batching
    // removes accesses, so `loaded %` rises even where bytes moved per token does not fall.
    let per_tok =
        |loaded: u64| (accesses as u64).saturating_sub(loaded) as f64 / ntok.max(1) as f64;

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

    println!("\n{:<10} {:>8} {:>12}", "policy", "loaded", "reads/token");
    let mut best_online = f64::MIN;
    for pol in ["lru", "2q", "arc"] {
        let loaded = replay(pol, cap, split, &trace)?;
        best_online = best_online.max(pct(loaded));
        println!("{pol:<10} {:>7.2}% {:>12.2}", pct(loaded), per_tok(loaded));
    }
    // The bound, and what is left under it. This line is the whole point of the table: a
    // headroom near zero RETIRES residency as a lever and leaves format (`perf-roadmap.md`
    // #2) as the only remaining way to move bytes; a large one prices the policy work in
    // the currency that matters — at ~8 experts/layer a pp is ~0.6 fewer misses per token.
    let opt_loaded = replay_opt(cap, &trace);
    let opt = pct(opt_loaded);
    println!(
        "{:<10} {opt:>7.2}% {:>12.2}   <- Belady bound (clairvoyant, unreachable online)",
        "opt",
        per_tok(opt_loaded)
    );
    println!(
        "headroom:  {:.2} pp  — every online policy above is at least this far from optimal",
        opt - best_online
    );
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

    /// A batch pays the UNION of its rows, not the sum — the property the whole batched-
    /// prefill question turns on, so it is asserted rather than assumed. Two tokens over
    /// two layers, agreeing on one expert per layer: 4 accesses per layer collapse to 3.
    ///
    /// It also pins the two things a wrong union would silently break: the layer split is
    /// derived from `expert_key`'s high 16 bits (so `tokens` must find 2, not 1 or 4), and
    /// layers come back in id order with first-seen rank inside them, because the policies
    /// downstream are recency-sensitive and read this as an access sequence.
    #[test]
    fn a_batch_reads_the_union_of_its_rows_not_the_sum() {
        let k = |layer: u32, expert: u32| (layer << 16) | expert;
        let t = parse(&format!(
            "{} {}\n{} {}\n{} {}\n{} {}\n",
            k(3, 1),
            k(3, 2),
            k(4, 5),
            k(4, 6),
            k(3, 2),
            k(3, 3),
            k(4, 5),
            k(4, 7),
        ));
        assert_eq!(tokens(&t).len(), 2, "two tokens of two MoE layers each");
        assert_eq!(
            coalesce(&t, 1),
            t,
            "batch=1 is the captured trace, untouched"
        );
        assert_eq!(
            coalesce(&t, 2),
            vec![
                vec![k(3, 1), k(3, 2), k(3, 3)],
                vec![k(4, 5), k(4, 6), k(4, 7)],
            ],
        );
    }

    /// The textbook case OPT exists to catch, hand-computed. `1 2 3 1 2 3 1 2 3` at cap 2:
    /// OPT scores **3** — on the first 3 it drops 2 (next use at index 4) rather than 1
    /// (index 3), so 1 hits, and the pattern repeats. The exact 3 fails if the eviction rule
    /// inverts, if the next-use table is off by an access, or if pass 1 forgets to re-key a
    /// hit past the reference it is serving.
    ///
    /// The second assert is the property the whole tool rests on — **no online policy can
    /// exceed the bound** — checked against all three rather than one. Note this is a strict
    /// inequality, not `lru == 0`: `hybrid::evict_lru` runs two tiers on INDEPENDENT clocks
    /// and is documented as not a true global LRU, and that drift is worth 2 hits here. If
    /// this ever reads 0, the tier segmentation changed — that is a finding, not a typo.
    #[test]
    fn opt_bounds_every_online_policy() {
        let t = parse("1\n2\n3\n1\n2\n3\n1\n2\n3\n");
        let opt = replay_opt(2, &t);
        assert_eq!(opt, 3, "Belady keeps the sooner-used key");
        for pol in ["lru", "2q", "arc"] {
            let online = replay(pol, 2, TwoQSplit::default(), &t).unwrap();
            assert!(
                online < opt,
                "{pol} scored {online} against a bound of {opt}"
            );
        }
    }
}
