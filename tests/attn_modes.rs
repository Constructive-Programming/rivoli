//! Attention-mode equivalence against the real snapshot: below the sparsity
//! thresholds every mode must select ALL causal rows, so dense / streaming /
//! dsa outputs are bit-identical. This pins (a) the row-gather refactor of the
//! absorb core, (b) the indexer's dense-fallback path, and (c) IndexShare
//! propagation through a shared layer — with the trained weights, not mocks.
//!
//! Needs a snapshot dir that includes the `out-idx-*` indexer shard: set
//! `RIVOLI_SNAPSHOT` or provide `~/glm52-snap`. Skips (pass, with a note) when
//! absent, so bare CI stays green.

use rivoli::attn::{AttnMode, AttnScratch, AttnWeights, KvCache, attention};
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
    run_mode_kv(snap, cfg, layers, mode, steps, false)
}

fn run_mode_kv(
    snap: &Snapshot,
    cfg: &ModelConfig,
    layers: &[usize],
    mode: &AttnMode,
    steps: usize,
    kv_fp8: bool,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut kv = KvCache::new(cfg, kv_fp8);
    let mut scr = AttnScratch::new(cfg);
    let mut indexer = match mode {
        AttnMode::Dsa | AttnMode::Misa { .. } => Some(Indexer::new(snap, cfg)?),
        _ => None,
    };
    // Weights resolve once (as in Engine::new); the loop is pure math.
    let weights = layers
        .iter()
        .map(|&l| AttnWeights::load(snap, cfg, l))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut outs = Vec::new();
    for pos in 0..steps {
        for (li, &layer) in layers.iter().enumerate() {
            let x = hidden(cfg, (pos * 1000 + layer) as u32);
            let mut out = vec![0.0f32; cfg.hidden];
            attention(
                &weights[li],
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

    let misa = run_mode(
        &snap,
        &cfg,
        &layers,
        &AttnMode::Misa { active_heads: 8 },
        steps,
    )?;
    for (i, d) in dense.iter().enumerate() {
        assert_eq!(d, &misa[i], "dense vs misa diverged at step-layer {i}");
    }

    // The same below-threshold equivalence must hold with the fp8 latent cache:
    // fp8 changes the CACHE, not which rows are selected, so dsa/misa still fall
    // back to dense row selection and must match dense-fp8 EXACTLY. (The bf16
    // cross above would not catch a divergence that only appears under fp8.)
    let dense_fp8_eq = run_mode_kv(&snap, &cfg, &layers, &AttnMode::Dense, steps, true)?;
    let dsa_fp8 = run_mode_kv(&snap, &cfg, &layers, &AttnMode::Dsa, steps, true)?;
    let misa_fp8 = run_mode_kv(
        &snap,
        &cfg,
        &layers,
        &AttnMode::Misa { active_heads: 8 },
        steps,
        true,
    )?;
    for (i, d) in dense_fp8_eq.iter().enumerate() {
        assert_eq!(d, &dsa_fp8[i], "dense-fp8 vs dsa-fp8 diverged at {i}");
        assert_eq!(d, &misa_fp8[i], "dense-fp8 vs misa-fp8 diverged at {i}");
    }

    // fp8 latent cache: NOT bit-identical to bf16 (lossy by design) but close.
    // The attention output stays within a small tolerance of the bf16 result.
    let dense_fp8 = run_mode_kv(&snap, &cfg, &layers, &AttnMode::Dense, steps, true)?;
    for (i, d) in dense.iter().enumerate() {
        let max_ref = d.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let max_err = d
            .iter()
            .zip(&dense_fp8[i])
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            max_err <= 0.05 * max_ref + 1e-3,
            "fp8 latent cache diverged too far at step-layer {i}: max_err={max_err:.3e} (ref {max_ref:.3e})"
        );
    }
    Ok(())
}

/// Sparse-regime indexer mechanics (the equivalence test above never leaves
/// the dense-fallback path): push layer 0 well past index_topk cached tokens,
/// then check DSA returns exactly topk sorted unique rows, the selection is
/// score-driven (not a degenerate tie-break prefix), and MISA's 8-head routed
/// selection substantially agrees with the full 32-head DSA selection.
///
/// The margin matters: at nt = topk + m, ANY two selections overlap in at
/// least topk−m rows, so IoU ≥ (topk−m)/(topk+m) is forced. With m = 512 the
/// forced floor is 0.6 and a broken router (random/inverted heads) lands near
/// E[IoU] ≈ 0.66. The healthy measurement HERE is ~0.84, not the paper's >92%:
/// at nt=2560 the router sees only ⌈nt/1024⌉ = 3 pooled blocks (the paper's
/// contexts give it 8–128), and synthetic activations are not what the gates
/// were trained on. The 0.75 floor cleanly separates broken (≈0.66) from
/// healthy (≈0.84) in this regime; true routing QUALITY is a real-decode
/// long-context eval, not this harness's job.
#[test]
fn indexer_sparse_regime_topk_and_misa_overlap() -> anyhow::Result<()> {
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

    let steps = cfg.index_topk + 512;
    // Two indexers fed identical streams — selections at each step are
    // comparable because the key caches are identical. The stream must be
    // REALISTIC in shape, not iid noise: MISA's premise is that a query's
    // relevant heads drift slowly along the prefix, which holds for real
    // activation streams and fails for white noise. So: x is a slow random
    // walk, and qr is the model's ACTUAL q-LoRA residual of x (q_a projection
    // + rmsnorm with the trained weights), exactly what attention() computes.
    //
    // Only the FINAL step's selection is asserted, so the cache-building steps
    // run under an inflated index_topk (fast path: append + pool only, no
    // scoring) — identical kcache/pool state at a fraction of the cost — and
    // the last step scores once under the real config.
    let mut cfg_grow = cfg.clone();
    cfg_grow.index_topk = steps + 1;
    let q_a = snap.int4("model.layers.0.self_attn.q_a_proj", cfg.hidden)?;
    let q_a_ln = rivoli::quant::read_f32(snap.typed(
        "model.layers.0.self_attn.q_a_layernorm.weight",
        rivoli::snapshot::Dtype::F32,
    )?);
    let mut dsa = Indexer::new(&snap, &cfg)?;
    let mut misa = Indexer::new(&snap, &cfg)?;
    let mut rows_dsa = Vec::new();
    let mut rows_misa = Vec::new();
    let mut x = hidden(&cfg, 0);
    let mut qr = vec![0.0f32; cfg.q_lora_rank];
    for pos in 0..steps {
        for (xi, &n) in x.iter_mut().zip(&hidden(&cfg, pos as u32)) {
            *xi = 0.95 * *xi + 0.3 * n;
        }
        let last = pos == steps - 1;
        let step_cfg = if last { &cfg } else { &cfg_grow };
        if last {
            rivoli::quant::matvec_i4(&mut qr, &x, &q_a);
            rivoli::math::rmsnorm(&mut qr, &q_a_ln, cfg.rms_norm_eps as f32);
        }
        dsa.select(step_cfg, 0, &x, &qr, pos, None, &mut rows_dsa)?;
        misa.select(step_cfg, 0, &x, &qr, pos, Some(8), &mut rows_misa)?;
    }

    assert_eq!(rows_dsa.len(), cfg.index_topk, "dsa row count");
    assert!(
        rows_dsa.windows(2).all(|w| w[0] < w[1]),
        "rows not sorted-unique"
    );
    assert!(
        rows_dsa.iter().all(|&r| (r as usize) < steps),
        "row out of range"
    );
    assert_eq!(rows_misa.len(), cfg.index_topk, "misa row count");
    // Score-driven, not degenerate: all-zero scores would tie-break to the
    // contiguous prefix 0..topk. Real scores must have dropped some early
    // token in favor of a later one.
    let prefix: Vec<u32> = (0..cfg.index_topk as u32).collect();
    assert_ne!(rows_dsa, prefix, "dsa selection is the tie-break prefix");

    let set: std::collections::HashSet<u32> = rows_dsa.iter().copied().collect();
    let inter = rows_misa.iter().filter(|r| set.contains(r)).count();
    let iou = inter as f64 / (2 * cfg.index_topk - inter) as f64;
    eprintln!("misa-vs-dsa selection IoU at nt={steps}: {iou:.3}");
    assert!(
        iou > 0.75,
        "misa selection diverged from dsa (IoU {iou:.3})"
    );
    Ok(())
}
