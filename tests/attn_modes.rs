//! Attention-mode equivalence against the real snapshot: below the sparsity
//! thresholds every mode must select ALL causal rows, so dense / streaming /
//! dsa outputs are bit-identical. This pins (a) the row-gather refactor of the
//! absorb core, (b) the indexer's dense-fallback path, and (c) IndexShare
//! propagation through a shared layer — with the trained weights, not mocks.
//!
//! Needs a snapshot dir that includes the `out-idx-*` indexer shard: set
//! `RIVOLI_SNAPSHOT` or provide `~/glm52-snap`. Skips (pass, with a note) when
//! absent, so bare CI stays green.

use rivoli::attn::{AttnMode, AttnScratch, KvCache, attention};
use rivoli::indexer::Indexer;
use rivoli::model::ModelConfig;
use rivoli::snapshot::Snapshot;

fn snapshot_dir() -> Option<String> {
    let dir = std::env::var("RIVOLI_SNAPSHOT")
        .unwrap_or_else(|_| format!("{}/glm52-snap", std::env::var("HOME").unwrap_or_default()));
    std::path::Path::new(&dir).is_dir().then_some(dir)
}

/// Deterministic pseudo-random hidden vector, small-magnitude.
fn hidden(cfg: &ModelConfig, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..cfg.hidden)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 16) as f32 / 65536.0 - 0.5) * 0.1
        })
        .collect()
}

/// Run `steps` decode steps of `attention` on `layers` under `mode`, returning
/// every step's output for every layer.
fn run_mode(
    snap: &Snapshot,
    cfg: &ModelConfig,
    layers: &[usize],
    mode: &AttnMode,
    steps: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut kv = KvCache::new(cfg);
    let mut scr = AttnScratch::new(cfg);
    let mut indexer = match mode {
        AttnMode::Dsa | AttnMode::Misa { .. } => Some(Indexer::new(cfg)?),
        _ => None,
    };
    let mut outs = Vec::new();
    for pos in 0..steps {
        for &layer in layers {
            let x = hidden(cfg, (pos * 1000 + layer) as u32);
            let mut out = vec![0.0f32; cfg.hidden];
            attention(
                snap,
                cfg,
                layer,
                &x,
                pos,
                mode,
                indexer.as_mut(),
                &mut kv,
                &mut scr,
                &mut out,
            )?;
            outs.push(out);
        }
    }
    Ok(outs)
}

#[test]
fn modes_agree_below_sparsity_thresholds() -> anyhow::Result<()> {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: no snapshot (set RIVOLI_SNAPSHOT or provide ~/glm52-snap)");
        return Ok(());
    };
    let snap = Snapshot::open(&dir)?;
    let cfg = ModelConfig::load(&dir)?;
    if snap
        .bf16("model.layers.0.self_attn.indexer.wk.weight", cfg.hidden)
        .is_err()
    {
        eprintln!("skipping: snapshot has no out-idx indexer shard");
        return Ok(());
    }

    // Layer 0 is 'full', layer 3 is 'shared' (GLM-5.2 IndexShare layout) —
    // exercising both pins the reuse path.
    let layers = [0usize, 3];
    let steps = 5;
    let dense = run_mode(&snap, &cfg, &layers, &AttnMode::Dense, steps)?;
    let dsa = run_mode(&snap, &cfg, &layers, &AttnMode::Dsa, steps)?;
    let stream = run_mode(
        &snap,
        &cfg,
        &layers,
        &AttnMode::Streaming {
            sinks: 4,
            window: 8192,
        },
        steps,
    )?;

    for (i, d) in dense.iter().enumerate() {
        assert_eq!(d, &dsa[i], "dense vs dsa diverged at step-layer {i}");
        assert_eq!(
            d, &stream[i],
            "dense vs streaming diverged at step-layer {i}"
        );
    }
    Ok(())
}
