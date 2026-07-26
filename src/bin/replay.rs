//! Offline cache-policy A/B, 2Q Kin/Kout sweep, and the [CACHE_ROUTE](../../docs/CACHE_ROUTE.md)
//! offline screen. Replays a routed-expert access trace (captured with `rivoli --trace`)
//! through the SAME byte-aware policies the engine runs ([`rivoli::hybrid`]) at a chosen
//! slot count — unit strides make the byte budget a plain slot count — and prints the
//! residency (loaded %). Pure CPU, milliseconds: compare policies without a full GPU
//! decode run.
//!
//! A **v2** trace additionally carries the ranked candidate window per routing decision,
//! which unlocks two things a v1 trace cannot answer, both printed automatically:
//!
//! - the **(J, M) grid** — replays under `top-m` cache-conditional substitution
//!   (arXiv:2412.00099) and reports how much of the miss rate it removes;
//! - the **oracle-prefetch ceiling** — what [CACHE_PILOT](../../docs/CACHE_PILOT.md) could
//!   reach at 100% recall, i.e. with perfect knowledge of decision `L+h`'s true experts,
//!   admitted under the real byte policy.
//!
//! usage: replay <trace> <n_slots> [--kin <pct>] [--kout <pct>] [--sweep] [--policy <p>]

use anyhow::{Context, Result, bail};
use rivoli::cache::TwoQSplit;
use rivoli::hybrid::{self, HybridPolicy};
use std::io::BufRead;

/// The Kin/Kout grid a `--sweep` walks. Kin is a resident probation bound so it is
/// swept fine-grained around the default; Kout is a key-only ghost (4 bytes an entry),
/// cheap enough to sweep well past 100 % of capacity.
const KIN_GRID: [u32; 11] = [3, 5, 8, 12, 16, 20, 25, 33, 40, 50, 66];
const KOUT_GRID: [u32; 6] = [25, 50, 100, 200, 400, 800];

/// The `top-m` screen's grid. J is the sacred prefix (the paper uses 1 for Mixtral/Phi
/// and 2 for the Qwen/DeepSeek-class routers GLM-5.2 belongs to; 3 and 4 bound how much
/// the bar moves as substitution is throttled). M is the candidate window, from `top_k`
/// itself — where substitution is a no-op and the row must reproduce the baseline — out
/// to the full recorded window.
const J_GRID: [usize; 4] = [1, 2, 3, 4];
const M_GRID: [usize; 7] = [8, 10, 12, 16, 20, 24, 32];

/// The prefetch horizons the oracle ceiling is reported at. L+2 is the horizon
/// docs/CACHE_PILOT.md argues for (one layer of decode compute is shorter than one
/// expert load); L+1 is the cheaper prediction, and the gap between them is the cost of
/// reaching further.
const HORIZONS: [usize; 2] = [1, 2];

/// The one real recall datum anyone has: colibri measured 71.6% L+1 on GLM-5.2 (48 greedy
/// tokens). Ours is int3-vq quantized and L+2 is unmeasured anywhere — that is what LOOKA
/// is for. The curve marks whichever swept point lands nearest this.
const COLIBRI_L1_RECALL: f64 = 0.716;

/// One routing decision (one MoE layer of one token).
struct Decision {
    /// The keys the engine actually looked up, in access order. Always present.
    demand: Vec<u32>,
    /// v2 only: the ranked candidate window, best-first. Empty for a v1 trace, which
    /// silently disables the substitution grid and the oracle ceiling.
    window: Vec<u32>,
}

/// Parse a trace. v1 lines are bare whitespace-separated demand keys; v2 lines add
/// `| key:choice ...`. The `choice` scores are recorded for a future `route_kl` and are
/// deliberately dropped here — the (J, M) substitution needs only the rank ORDER, which
/// the window's ordering already carries.
///
/// Lenient by construction, and that is what makes v2 readable by a v1 reader too: a
/// token that does not parse as a `u32` is skipped, so the `# rivoli-trace v2 ...`
/// header yields an empty decision and is dropped.
fn parse_trace(r: impl BufRead) -> Result<Vec<Decision>> {
    let mut out = Vec::new();
    for line in r.lines() {
        let line = line.context("read trace")?;
        let (head, tail) = match line.split_once('|') {
            Some((h, t)) => (h, t),
            None => (line.as_str(), ""),
        };
        let demand: Vec<u32> = head.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if demand.is_empty() {
            continue; // header, blank line, or a v1 comment
        }
        // `key:choice` — take the key, drop the score.
        let window = tail
            .split_whitespace()
            .filter_map(|t| t.split(':').next()?.parse().ok())
            .collect();
        out.push(Decision { demand, window });
    }
    Ok(out)
}

/// Fill `out` with the `top-m` substituted selection for one decision (arXiv:2412.00099).
/// The top-`j` candidates are **sacred** — always selected, resident or not. The
/// remaining `k - j` slots prefer candidates that are already RESIDENT and ranked inside
/// the top-`m` window, then fall back to plain rank order. Weights are untouched by
/// construction here: this reorders *selection* only, and the offline sim scores nothing.
///
/// `resident` must not mutate the policy — [`HybridPolicy::contains`] takes `&self` and
/// does not refresh recency, so asking about the whole window is free of side effects on
/// the eviction clock.
fn substitute(window: &[u32], k: usize, j: usize, m: usize, resident: impl Fn(u32) -> bool, out: &mut Vec<u32>) {
    out.clear();
    let (j, m) = (j.min(k), m.min(window.len()));
    out.extend(window.iter().take(j));
    for &key in &window[j.min(m)..m] {
        if out.len() == k {
            break;
        }
        if resident(key) {
            out.push(key);
        }
    }
    // Fall back to rank order for whatever the window could not fill from residents.
    for &key in &window[j.min(window.len())..] {
        if out.len() == k {
            break;
        }
        if !out.contains(&key) {
            out.push(key);
        }
    }
}

/// A modelled CACHE_PILOT predictor: it names `k` experts for the decision `horizon`
/// layers ahead, of which a `recall` fraction are right.
///
/// **This is a MODEL, not a measurement,** and it is optimistic in one way that cannot be
/// fixed offline: errors here are independent across decisions, where a real predictor's
/// errors are correlated (the tokens it finds hard, it finds hard for many layers in a
/// row). Treat every number it produces as an upper bound. The real recall figure comes
/// from LOOKA (docs/CACHE_PILOT.md Step 1), on the device.
#[derive(Clone, Copy)]
struct Pilot {
    horizon: usize,
    /// How many of the decision's `top_k` experts it gets RIGHT — a count, not a
    /// fraction. `top_k` is 8, so recall only exists in eighths; sweeping a fraction
    /// would print 70%, 71.6% and 80% as three identical rows and need a footnote
    /// apologising for it. A count makes every row an exact, distinct configuration.
    keep: usize,
}

impl Pilot {
    /// Fill `out` with what this predictor would name for `next`. It always names `k`
    /// experts — a real pilot takes the top-`k` of its own predicted logits, so a false
    /// negative is necessarily also a false POSITIVE, and false positives are the whole
    /// cost: each one is an admission, an eviction and a wasted read. (Colibri measured
    /// unguarded speculation at +9–18% bytes for +0.5–0.7pt hit for exactly this reason.)
    ///
    /// The kept experts are the top-ranked ones, and the distractors are drawn from the
    /// ranks immediately OUTSIDE the true set in that decision's own recorded window.
    /// Both choices model a router-based predictor: it fails on the marginal experts, not
    /// the obvious ones, and it confuses them with their rank neighbours rather than with
    /// a uniform draw over all 256. Recording that window is what makes this possible at
    /// all — a v1 trace could only have modelled a uniform draw.
    fn predict(&self, next: &Decision, out: &mut Vec<u32>) {
        let k = next.demand.len();
        let keep = self.keep.min(k);
        out.clear();
        out.extend(next.demand.iter().take(keep));
        out.extend(next.window.iter().skip(k).take(k - keep));
    }
}

/// What one replay produced: demand accesses served from residency, and the speculative
/// admissions the oracle issued (0 unless prefetching).
#[derive(Default, Clone, Copy, PartialEq, Eq, Debug)]
struct Counts {
    loaded: u64,
    spec: u64,
}

/// Replay `trace` through `policy` at `cap` unit slots; return the `loaded`
/// (resident-hit) count over the DEMAND accesses. Two-pass per layer — hits first
/// (protected), then misses — mirroring the pin's `submit_layer`.
///
/// `sub = Some((j, m))` applies the `top-m` substitution above. `prefetch = Some(h)`
/// additionally admits decision `i + h`'s **true** keys at the end of decision `i`: the
/// oracle-prefetch ceiling, i.e. CACHE_PILOT with a perfect predictor, paying the real
/// policy's admission and eviction. Both `None` is the plain policy A/B, and is
/// byte-identical to what this tool did before the v2 format existed.
///
/// Also returns the speculative admissions issued, which is the byte story: each one is
/// a real read. At 100% recall it equals the misses the baseline would have taken — the
/// pilot moves those bytes off the critical path without adding any, which is precisely
/// CACHE_PILOT's "the lever is residency, not earlier bytes". Below 100% recall the
/// excess over that is wasted bandwidth.
fn replay(
    policy: &str,
    cap: usize,
    split: TwoQSplit,
    trace: &[Decision],
    sub: Option<(usize, usize)>,
    prefetch: Option<Pilot>,
) -> Result<Counts> {
    // Unit strides: budget `cap` bytes == `cap` slots, cold==hot so the split is by
    // slot count exactly as the single-format engine sees it.
    let mut p: Box<dyn HybridPolicy> =
        hybrid::make(policy, cap, 1, 1, split).with_context(|| format!("unknown policy {policy}"))?;
    let mut c = Counts::default();
    let mut miss: Vec<u32> = Vec::new();
    let mut chosen: Vec<u32> = Vec::new();
    let mut pred: Vec<u32> = Vec::new();
    for (i, d) in trace.iter().enumerate() {
        p.begin_batch(); // one MoE layer = one batch (mirrors the pin)
        // Substitution reads residency but must not touch the eviction clock, so it runs
        // before the two phases below and only through `contains`.
        let keys = match sub {
            Some((j, m)) if !d.window.is_empty() => {
                substitute(&d.window, d.demand.len(), j, m, |k| p.contains(k), &mut chosen);
                &chosen
            }
            _ => &d.demand,
        };
        miss.clear();
        for &k in keys {
            if p.get(k) {
                c.loaded += 1;
                p.protect(k);
            } else {
                miss.push(k);
            }
        }
        for &k in &miss {
            p.admit(k);
        }
        // Oracle prefetch: the speculative admissions land inside this batch, so they are
        // pinned for the rest of it — which is what an in-flight speculative read is.
        // Unguarded on purpose: this is the CEILING, not a proposal. The real loader must
        // drop a speculation rather than displace a hotter resident (CACHE_PILOT Step 2),
        // which can only cost recall relative to this number.
        if let Some(pilot) = prefetch
            && let Some(next) = trace.get(i + pilot.horizon)
        {
            pilot.predict(next, &mut pred);
            for &k in &pred {
                if !p.contains(k) {
                    c.spec += 1;
                    p.admit(k);
                }
            }
        }
    }
    Ok(c)
}

/// `(absolute pp on the hit rate, relative % of misses removed)` for `hit` against
/// `base`, both as percentages. Both numbers are reported for every grid cell: the pp
/// figure is the acceptance bar and is directly comparable to `benchmarks.md`'s hit%
/// column; the relative figure is what makes the result comparable to the paper's
/// ">50% cache-miss reduction".
fn delta(base: f64, hit: f64) -> (f64, f64) {
    let base_miss = 100.0 - base;
    (hit - base, if base_miss > 0.0 { 100.0 * (hit - base) / base_miss } else { 0.0 })
}

fn main() -> Result<()> {
    const USAGE: &str = "usage: replay <trace> <n_slots> [--kin <pct>] [--kout <pct>] [--sweep] [--policy <p>]";
    let mut args = std::env::args().skip(1);
    let trace_path = args.next().context(USAGE)?;
    let cap: usize = args
        .next()
        .context("n_slots required")?
        .parse()
        .context("n_slots must be an integer")?;
    let mut sweep = false;
    // The (J, M) grid and the oracle run under ONE policy: docs/CACHE_ROUTE.md builds
    // `top-m` on `HybridLru` (the paper evaluates on LRU) and hybrid+lru is the fastest
    // coherent config in benchmarks.md. Overridable, but one grid is the readable one.
    let mut grid_policy = "lru".to_string();
    let default = TwoQSplit::default();
    let (mut kin, mut kout) = (default.kin_pct(), default.kout_pct());
    while let Some(a) = args.next() {
        let mut val = |what: &str| args.next().with_context(|| format!("{what} needs a value"));
        match a.as_str() {
            "--sweep" => sweep = true,
            "--kin" => kin = val("--kin")?.parse().context("--kin takes an integer percentage")?,
            "--kout" => kout = val("--kout")?.parse().context("--kout takes an integer percentage")?,
            "--policy" => grid_policy = val("--policy")?,
            other => bail!("unexpected arg {other}\n{USAGE}"),
        }
    }
    let split = TwoQSplit::new(kin, kout)?;

    let f = std::fs::File::open(&trace_path).with_context(|| format!("open trace {trace_path}"))?;
    let trace = parse_trace(std::io::BufReader::new(f))?;

    let accesses: usize = trace.iter().map(|d| d.demand.len()).sum();
    let uniq = trace
        .iter()
        .flat_map(|d| &d.demand)
        .copied()
        .collect::<std::collections::HashSet<u32>>()
        .len();
    println!(
        "trace {trace_path}: {} layers, {accesses} accesses, {uniq} unique experts, cap={cap}",
        trace.len()
    );
    // Every replay is scored the same way: demand accesses served from residency.
    let pct = |c: Counts| 100.0 * c.loaded as f64 / accesses.max(1) as f64;

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
                let loaded = pct(replay("2q", cap, s, &trace, None, None)?);
                if loaded > best.1 {
                    best = (s, loaded);
                }
                print!("{loaded:>8.2}%");
            }
            println!();
        }
        let base = pct(replay("2q", cap, default, &trace, None, None)?);
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
        println!("{pol:<10} {:>7.2}%", pct(replay(pol, cap, split, &trace, None, None)?));
    }
    if split != default {
        println!("\n(2q ran with kin {kin}% / kout {kout}%)");
    }

    // --- v2 only: the CACHE_ROUTE offline screen -------------------------------------
    let windowed = trace.iter().filter(|d| !d.window.is_empty()).count();
    if windowed == 0 {
        println!(
            "\n(v1 trace — no candidate window recorded, so the top-m (J, M) grid and the\n\
             oracle-prefetch ceiling are unavailable. Recapture with a v2-format --trace.)"
        );
        return Ok(());
    }
    if windowed != trace.len() {
        bail!("mixed trace: {windowed}/{} decisions carry a candidate window", trace.len());
    }
    // Invariant: the window is ranked by the same `choice` with the same comparator that
    // produced `sel`, so its first `top_k` entries MUST be the demand keys. If that ever
    // fails the writer and the reader disagree and every number below is meaningless.
    if let Some(bad) = trace.iter().position(|d| !d.window.starts_with(&d.demand)) {
        bail!(
            "trace decision {bad}: window prefix {:?} does not match demand {:?} — the\n\
             candidate window is not the ranking that produced the selection",
            &trace[bad].window[..trace[bad].demand.len().min(trace[bad].window.len())],
            trace[bad].demand,
        );
    }

    let base_c = replay(&grid_policy, cap, split, &trace, None, None)?;
    let base = pct(base_c);
    let base_miss = accesses as u64 - base_c.loaded;
    // The trace does not need to be asked for `top_k` — every decision selects exactly
    // that many, and the window-prefix check above already proved the trace self-consistent.
    let top_k = trace.first().map_or(0, |d| d.demand.len());
    println!(
        "\ntop-m (J, M) grid — policy {grid_policy}, cap={cap}, baseline {base:.2}% hit.\n\
         Cells are `hit% (+pp / rel% of misses removed)`. J = sacred prefix, M = candidate\n\
         window. Acceptance screen: >= +5.00 pp absolute at some cell.\n"
    );
    print!("{:<5}", "J\\M");
    for m in M_GRID {
        print!("{:>24}", format!("M={m}"));
    }
    println!();
    let mut best = (0usize, 0usize, f64::MIN);
    for j in J_GRID {
        print!("{:<5}", format!("J={j}"));
        for m in M_GRID {
            let hit = pct(replay(&grid_policy, cap, split, &trace, Some((j, m)), None)?);
            let (pp, rel) = delta(base, hit);
            if hit > best.2 {
                best = (j, m, hit);
            }
            print!("{:>24}", format!("{hit:.2}% ({pp:+.2}/{rel:+.1}%)"));
        }
        println!();
    }
    let (pp, rel) = delta(base, best.2);
    println!(
        "\nbest (J={}, M={}): {:.2}% hit  ({pp:+.2} pp, {rel:+.1}% of misses removed)  — {}",
        best.0,
        best.1,
        best.2,
        if pp >= 5.0 { "PASSES the >= +5pp screen" } else { "FAILS the >= +5pp screen" },
    );

    println!(
        "\nCACHE_PILOT recall curve — policy {grid_policy}, cap={cap}, baseline {base:.2}% hit,\n\
         {base_miss} baseline misses (= baseline admissions, the byte reference).\n\
         Recall is a COUNT of a decision's {top_k} experts the predictor gets right, since\n\
         recall only exists in 1/{top_k} steps. The {top_k}/{top_k} row is the ORACLE CEILING:\n\
         perfect knowledge of L+h's true experts, admitted under the real byte policy with no\n\
         eviction guard — and it is close to vacuous, because {top_k} admissions fit in any pool\n\
         that holds one batch, so a perfect predictor removes every miss by construction and\n\
         moves exactly the baseline's bytes earlier rather than adding any.\n\
         Every row below it still names {top_k} experts, so each false negative is also a false\n\
         POSITIVE, drawn from that decision's own candidate window — the admissions column is\n\
         where that cost lands. THIS IS A MODEL, NOT A MEASUREMENT: errors are independent\n\
         across decisions here and correlated in reality, so every row is an upper bound.\n\
         Real recall comes from LOOKA, on the device.\n"
    );
    println!(
        "{:<6}{:>14}{:>10}{:>10}{:>13}{:>12}",
        "h", "recall", "hit%", "+pp", "admissions", "vs base"
    );
    // Nearest swept point to colibri's measured L+1 recall, marked so the one real datum
    // anyone has is findable in the curve rather than left to the reader's arithmetic.
    let colibri_keep = (COLIBRI_L1_RECALL * top_k as f64).round() as usize;
    for h in HORIZONS {
        // Half-right to all-right. Below half a predictor is not worth discussing, and
        // `keep` is a COUNT because recall only exists in 1/top_k steps.
        for keep in (top_k / 2)..=top_k {
            let c = replay(&grid_policy, cap, split, &trace, None, Some(Pilot { horizon: h, keep }))?;
            let (pp, _) = delta(base, pct(c));
            // Every admission, demand or speculative, moves one expert's bytes; the stride
            // is constant, so the admission RATIO is exactly the byte ratio.
            let admits = (accesses as u64 - c.loaded) + c.spec;
            println!(
                "L+{h:<4}{:>7.1}% ({keep}/{top_k}){:>9.2}%{pp:>+10.2}{admits:>13}{:>11.2}x{}",
                100.0 * keep as f64 / top_k as f64,
                pct(c),
                admits as f64 / base_miss.max(1) as f64,
                if keep == colibri_keep { "  <- nearest colibri L+1 (71.6%)" } else { "" },
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    /// Two decisions, written the way the v1 writer wrote them.
    const V1: &str = "10 11 12 13\n20 21 22 23\n";
    /// The same two decisions from the v2 writer: header, demand keys, then the ranked
    /// window (whose first `top_k` entries are the demand keys again) with scores.
    const V2: &str = "# rivoli-trace v2 top_k=4 window=6\n\
                      10 11 12 13 | 10:0.900000 11:0.800000 12:0.700000 13:0.600000 14:0.500000 15:0.400000\n\
                      20 21 22 23 | 20:0.900000 21:0.800000 22:0.700000 23:0.600000 24:0.500000 25:0.400000\n";

    fn parse(s: &str) -> Vec<Decision> {
        parse_trace(std::io::BufReader::new(s.as_bytes())).expect("parse")
    }

    /// The compat claim CACHE_ROUTE's staging rests on, as a test rather than an
    /// argument: a v2 trace must present the SAME demand-key sequence as the v1 trace of
    /// the same run. The header must vanish and the `| ...` tail must not leak into it.
    #[test]
    fn v2_demand_keys_are_identical_to_v1() {
        let (v1, v2) = (parse(V1), parse(V2));
        assert_eq!(v1.len(), v2.len(), "header line must not become a decision");
        for (a, b) in v1.iter().zip(&v2) {
            assert_eq!(a.demand, b.demand);
        }
        assert!(v1.iter().all(|d| d.window.is_empty()));
        assert!(v2.iter().all(|d| d.window.len() == 6));
    }

    /// ...and the residency numbers a v1 reader would print are unchanged, which is the
    /// property that actually matters: every policy, byte-identical, on both formats.
    #[test]
    fn v2_replays_identically_to_v1_under_every_policy() {
        let (v1, v2) = (parse(V1), parse(V2));
        for pol in ["lru", "2q", "arc"] {
            let a = replay(pol, 4, TwoQSplit::default(), &v1, None, None).expect("v1");
            let b = replay(pol, 4, TwoQSplit::default(), &v2, None, None).expect("v2");
            assert_eq!(a, b, "policy {pol} diverged between v1 and v2");
        }
    }

    /// The window's leading `top_k` entries are the selection, so substituting with a
    /// window no wider than `top_k` cannot change anything — whatever J is.
    #[test]
    fn m_equal_to_top_k_reproduces_the_baseline() {
        let t = parse(V2);
        let base = replay("lru", 4, TwoQSplit::default(), &t, None, None).expect("base");
        for j in 0..=4 {
            let sub = replay("lru", 4, TwoQSplit::default(), &t, Some((j, 4)), None).expect("sub");
            assert_eq!(base, sub, "J={j}, M=top_k must be a no-op");
        }
    }

    #[test]
    fn sacred_prefix_survives_even_when_nothing_is_resident() {
        let window: Vec<u32> = (0..8).collect();
        let mut out = Vec::new();
        substitute(&window, 4, 2, 8, |_| false, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3], "no residents => plain rank order");
        substitute(&window, 4, 2, 8, |k| k >= 6, &mut out);
        assert_eq!(out, vec![0, 1, 6, 7], "top-J kept, the rest swapped to residents");
    }

    /// A resident outside the window must NOT be promoted — that is the whole point of
    /// the top-M bound (a resident but genuinely irrelevant expert must not run).
    #[test]
    fn residents_outside_the_window_are_not_promoted() {
        let window: Vec<u32> = (0..8).collect();
        let mut out = Vec::new();
        substitute(&window, 4, 1, 4, |k| k >= 6, &mut out);
        assert_eq!(out, vec![0, 1, 2, 3], "residents at rank 6/7 are outside M=4");
    }

    /// Selection is always exactly `top_k` keys, distinct, however the residency falls.
    #[test]
    fn substitution_preserves_the_selection_size() {
        let window: Vec<u32> = (0..12).collect();
        let mut out = Vec::new();
        for j in 0..=4 {
            for m in [4usize, 6, 8, 12, 99] {
                for res in [0u32, 1, 3, 7] {
                    substitute(&window, 4, j, m, |k| k % 4 == res, &mut out);
                    assert_eq!(out.len(), 4, "J={j} M={m} res={res}");
                    let uniq: std::collections::HashSet<_> = out.iter().collect();
                    assert_eq!(uniq.len(), 4, "duplicate selection at J={j} M={m} res={res}");
                }
            }
        }
    }

    /// The oracle can only help: admitting a future decision's true keys never evicts
    /// something it then needs more (at a capacity that holds a batch), so its hit count
    /// is >= the baseline's.
    #[test]
    fn oracle_prefetch_beats_the_baseline() {
        // Three decisions that cycle, so L+1 knowledge is worth something.
        let src = "# rivoli-trace v2 top_k=2 window=2\n\
                   1 2 | 1:0.9 2:0.8\n3 4 | 3:0.9 4:0.8\n1 2 | 1:0.9 2:0.8\n3 4 | 3:0.9 4:0.8\n";
        let t = parse(src);
        let base = replay("lru", 2, TwoQSplit::default(), &t, None, None).expect("base");
        let pilot = Pilot { horizon: 1, keep: 4 };
        let o = replay("lru", 2, TwoQSplit::default(), &t, None, Some(pilot)).expect("oracle");
        assert!(o.loaded >= base.loaded, "oracle {o:?} < baseline {base:?}");
        assert!(o.spec > 0, "the oracle issued no speculative admissions");
    }

    /// At r=1.0 the modelled pilot must name exactly the true set — no distractors — so
    /// the recall curve's top row IS the oracle ceiling rather than an approximation of it.
    #[test]
    fn recall_one_predicts_exactly_the_true_set() {
        let t = parse(V2);
        let mut out = Vec::new();
        for d in &t {
            Pilot { horizon: 1, keep: 4 }.predict(d, &mut out);
            assert_eq!(out, d.demand);
        }
    }

    /// A false negative is also a false positive: the predictor always names `top_k`
    /// experts, and the ones it gets wrong come from just outside the true set in the
    /// SAME decision's window — not from a uniform draw. That is the false-positive cost
    /// the byte column has to show.
    #[test]
    fn degraded_recall_substitutes_rank_neighbour_distractors() {
        let t = parse(V2);
        let mut out = Vec::new();
        // k=4, r=0.5 => keep 2 true, fill 2 from window[4..] = the ranks just outside.
        Pilot { horizon: 1, keep: 2 }.predict(&t[0], &mut out);
        assert_eq!(out.len(), t[0].demand.len(), "a real pilot always names top_k");
        assert_eq!(out, vec![10, 11, 14, 15]);
        let hits = out.iter().filter(|k| t[0].demand.contains(k)).count();
        assert_eq!(hits, 2, "half right, half wrong");
    }

    /// A malformed score must not silently drop the key it belongs to, and a v1 line
    /// mixed into a v2 file must still parse as a bare demand list.
    #[test]
    fn parser_is_lenient_in_the_ways_the_format_relies_on() {
        let t = parse("# header\n\n7 8 | 7: 8:notanumber\n9 10\n");
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].demand, vec![7, 8]);
        assert_eq!(t[0].window, vec![7, 8], "keys survive an unparseable score");
        assert_eq!(t[1].demand, vec![9, 10]);
        assert!(t[1].window.is_empty());
    }
}
