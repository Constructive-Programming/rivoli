//! Offline cache-policy A/B. Replays a routed-expert access trace (captured with
//! `rivoli --trace`) through LRU / 2Q / ARC at a chosen slot count and prints hit
//! rates — cold, and (with `--seed`) with the protected segment pre-filled from
//! `.coli_usage`. Pure CPU, milliseconds: the whole point is to compare policies
//! without ~90s GPU decode runs. Validate against the live pin: sim `lru` at the
//! live slot count should match the live hit rate on the same trace.
//!
//! usage: replay <trace> <n_slots> [--seed <snapshot_dir>]

use anyhow::{Context, Result, bail};
use rivoli::cache::simulate;
use std::io::BufRead;

/// Same packing as pin::expert_key — `(layer,expert)` into one u32.
fn key(l: usize, e: usize) -> u32 {
    ((l as u32) << 16) | e as u32
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let trace = args
        .next()
        .context("usage: replay <trace> <n_slots> [--seed <snapshot_dir>]")?;
    let cap: usize = args
        .next()
        .context("n_slots required")?
        .parse()
        .context("n_slots must be an integer")?;
    let mut seed_dir = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--seed" => seed_dir = Some(args.next().context("--seed needs a snapshot dir")?),
            other => bail!("unexpected arg {other}"),
        }
    }

    // Load the trace: one batch (MoE layer) per line, space-separated u32 keys.
    let f = std::fs::File::open(&trace).with_context(|| format!("open trace {trace}"))?;
    let mut batches: Vec<Vec<u32>> = Vec::new();
    for line in std::io::BufReader::new(f).lines() {
        let b: Vec<u32> = line?
            .split_whitespace()
            .filter_map(|s| s.parse().ok())
            .collect();
        if !b.is_empty() {
            batches.push(b);
        }
    }
    let accesses: usize = batches.iter().map(Vec::len).sum();
    let uniq: usize = {
        let mut s: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for b in &batches {
            s.extend(b.iter().copied());
        }
        s.len()
    };
    println!(
        "trace {trace}: {} batches, {accesses} accesses, {uniq} unique experts, cap={cap}",
        batches.len()
    );

    // Optional seed: the top-`cap` experts by `.coli_usage` frequency (same stale
    // filter the live pin applies), placed in each policy's protected segment.
    let seed: Vec<u32> = if let Some(dir) = &seed_dir {
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
    } else {
        Vec::new()
    };

    let seeded = !seed.is_empty();
    println!(
        "\n{:<6} {:>9} {:>9}",
        "policy",
        "cold%",
        if seeded { "seeded%" } else { "" }
    );
    for pol in ["lru", "2q", "arc"] {
        let (h, t) = simulate(pol, cap, &[], &batches);
        let cold = 100.0 * h as f64 / t as f64;
        if seeded {
            let (hs, ts) = simulate(pol, cap, &seed, &batches);
            println!(
                "{pol:<6} {cold:>8.1}% {:>8.1}%",
                100.0 * hs as f64 / ts as f64
            );
        } else {
            println!("{pol:<6} {cold:>8.1}%");
        }
    }
    Ok(())
}
