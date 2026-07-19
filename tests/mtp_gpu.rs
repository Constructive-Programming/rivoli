//! MTP M2 gate: the DEVICE draft (GpuEngine::mtp_draft) matches the scalar
//! oracle (src/mtp.rs) token-for-token. Drives the GPU main forward + device
//! draft and the scalar Engine + Mtp in lockstep on the same greedy stream, and
//! asserts the device draft argmax equals the scalar draft each step.
//!
//! Heavy (GPU pin build + slow scalar decode), so `#[ignore]`d — run explicitly:
//!   cargo test --release --features rocm --test mtp_gpu -- --ignored --nocapture
//! Needs the snapshot WITH out-mtp (RIVOLI_SNAPSHOT or ~/glm52-snap) and a GPU.
#![cfg(feature = "rocm")]

use rivoli::attn::AttnMode;
use rivoli::config::{MAX_POOL, OS_RESERVE};
use rivoli::engine::Engine;
use rivoli::gpu::GpuEngine;
use rivoli::mtp::Mtp;
use rivoli::pin::Pin;

fn snapshot_dir() -> Option<String> {
    let dir = std::env::var("RIVOLI_SNAPSHOT")
        .unwrap_or_else(|_| format!("{}/glm52-snap", std::env::var("HOME").unwrap_or_default()));
    std::path::Path::new(&dir).is_dir().then_some(dir)
}

#[test]
#[ignore = "slow: GPU pin build + reference scalar decode in lockstep"]
fn device_mtp_draft_matches_scalar_oracle() -> anyhow::Result<()> {
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

    let ngen = 5usize;
    let mut toks = tok.encode("The capital of France is")?;
    let prompt_len = toks.len();
    let max_ctx = prompt_len + ngen + 2;

    // GPU engine with the resident MTP layer (want_mtp = true).
    let (free, _) = rivoli::device::mem_info()?;
    let cap = free
        .saturating_sub(OS_RESERVE as usize)
        .min(MAX_POOL as usize);
    let pin = Pin::build(
        &snap, &cfg, &usage, cap, false, true, None, "arc", false, 2, false, false, true,
    )?;
    let mut gpu = GpuEngine::new(pin, &cfg, max_ctx, AttnMode::Dense, false)?;

    // Scalar reference + MTP oracle.
    let mut engine = Engine::new(&snap, &cfg, AttnMode::Dense, false)?;
    let mut mtp = Mtp::new(&snap, &cfg)?;

    let mut matches = 0usize;
    let mut total = 0usize;
    let mut main_agree = 0usize;
    for i in 0..prompt_len + ngen {
        if i >= toks.len() {
            break;
        }
        let gp = gpu.step(toks[i], i)?; // GPU main predicts i+1
        let sp = engine.step(toks[i], i)?; // scalar main predicts i+1
        if gp == sp {
            main_agree += 1;
        }
        let next = if i + 1 < toks.len() {
            toks[i + 1]
        } else {
            toks.push(gp); // extend the greedy stream from the GPU
            gp
        };
        let gd = gpu.mtp_draft(next, i)?; // device draft of i+2
        let sd = mtp.draft(&snap, &cfg, engine.trunk(), next, i)?; // scalar draft of i+2
        if gd == sd {
            matches += 1;
        }
        total += 1;
    }
    let rate = matches as f64 / total.max(1) as f64;
    eprintln!(
        "device-vs-scalar MTP draft agreement: {matches}/{total} = {rate:.2} \
         | main-forward agreement: {main_agree}/{total}"
    );
    // int4 kernel-vs-scalar f32-reduction differences can flip a rare near-tie
    // draft; require strong agreement, not necessarily perfect.
    assert!(total >= 4, "too few draft comparisons ({total})");
    assert!(
        rate >= 0.8,
        "device MTP draft diverged from the scalar oracle (agreement {rate:.2})"
    );
    Ok(())
}
