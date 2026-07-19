//! MTP M3 gate: the speculative decoder (`GpuEngine::generate_spec`) produces
//! output **identical** to plain greedy `generate` — the draft only changes how
//! positions are batched, never which token is emitted. Runs both on the same
//! engine (each starts from a pos-0 prefill that overwrites the KV) and asserts
//! the two token streams match exactly.
//!
//! Heavy (GPU pin build + two full decodes), so `#[ignore]`d — run explicitly:
//!   cargo test --release --features rocm --test mtp_spec -- --ignored --nocapture
//! Needs the snapshot WITH out-mtp (RIVOLI_SNAPSHOT or ~/glm52-snap) and a GPU.
#![cfg(feature = "rocm")]

use rivoli::attn::AttnMode;
use rivoli::config::{MAX_POOL, OS_RESERVE};
use rivoli::gpu::GpuEngine;
use rivoli::pin::Pin;

fn snapshot_dir() -> Option<String> {
    let dir = std::env::var("RIVOLI_SNAPSHOT")
        .unwrap_or_else(|_| format!("{}/glm52-snap", std::env::var("HOME").unwrap_or_default()));
    std::path::Path::new(&dir).is_dir().then_some(dir)
}

#[test]
#[ignore = "slow: GPU pin build + two full decodes"]
fn spec_decode_matches_greedy() -> anyhow::Result<()> {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: no snapshot");
        return Ok(());
    };
    let snap = rivoli::snapshot::Snapshot::open(&dir)?;
    let cfg = rivoli::model::ModelConfig::load(&dir)?;
    if snap
        .bf16("model.layers.78.eh_proj.weight", 2 * cfg.hidden)
        .is_err()
    {
        eprintln!("skipping: snapshot has no out-mtp shard");
        return Ok(());
    }
    let tok = rivoli::tokenizer::Tokenizer::load(&dir)?;
    let usage = rivoli::usage::Usage::load(&dir)?;

    let ngen = 24usize;
    let toks = tok.encode("The capital of France is")?;
    let prompt_len = toks.len();
    // Accept-both can advance 2 positions/round, so cap the KV generously.
    let max_ctx = prompt_len + ngen + 4;
    let eos = tok.eos.clone();

    let (free, _) = rivoli::device::mem_info()?;
    let cap = free
        .saturating_sub(OS_RESERVE as usize)
        .min(MAX_POOL as usize);
    let pin = Pin::build(
        &snap, &cfg, &usage, cap, false, true, None, "arc", false, 2, false, false, true,
    )?;
    let mut gpu = GpuEngine::new(pin, &cfg, max_ctx, AttnMode::Dense, false)?;

    // Speculative first, then the plain greedy reference on the same engine
    // (its pos-0 prefill overwrites the KV, so the runs are independent).
    let spec = gpu.generate_spec(&toks, ngen, &eos)?;
    let greedy = gpu.generate(&toks, ngen, &eos)?;

    eprintln!("spec   ({:2}): {spec:?}", spec.len());
    eprintln!("greedy ({:2}): {greedy:?}", greedy.len());
    // Greedy-equivalent, and for this fixed prompt bit-stable. The batched MoE
    // reduces experts in union order vs generate's score-desc order, so a genuine
    // near-tie argmax *could* flip by a ULP on some other prompt — if this ever
    // trips on a new prompt, confirm the divergence is a single near-tie, not a
    // logic bug, before loosening to a prefix-match.
    assert_eq!(
        spec, greedy,
        "speculative decode diverged from greedy — the draft must never change the emitted token"
    );
    Ok(())
}
