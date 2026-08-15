//! First-decode smoke: open a REAL GLM artifact, greedy-decode from raw token ids,
//! print the generated ids one per line. An EXAMPLE rather than a `#[test]` because it
//! needs a 281 GB artifact, the sole-tenant GPU and the flock — none of which a
//! hermetic suite may assume. M5's parity gate formalises the comparison this enables:
//! the same ids through the old engine must match token for token (greedy decode on a
//! deterministic engine).
//!
//! Usage: `glm_smoke <artifact-dir> <max-mem-GiB> <ngen> <id> [<id>...]`
//! Run under `flock /var/run/sys-gpu.lock`, dev profile (this is a correctness run).

#![allow(clippy::unwrap_used, clippy::expect_used)] // a smoke harness dies loudly

use anyhow::{Context, Result};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [dir, mem_gib, ngen, ids @ ..] = args.as_slice() else {
        anyhow::bail!("usage: glm_smoke <artifact-dir> <max-mem-GiB> <ngen> <id> [<id>...]");
    };
    let capacity = mem_gib.parse::<usize>().context("max-mem GiB")? << 30;
    let ngen = ngen.parse::<usize>().context("ngen")?;
    let prompt: Vec<u32> = ids
        .iter()
        .map(|s| s.parse::<u32>().context("token id"))
        .collect::<Result<_>>()?;
    let cfg = rivoli_artifact::glm_config::ModelConfig::load(dir)?;
    let pin = rivoli_engine::glm::pin::GlmPin::build(
        dir,
        &cfg,
        rivoli_engine::glm::pin::GlmPinCfg {
            capacity,
            fmt: rivoli_artifact::format::RoutedFmt::Vq3,
            cache_policy: "2q",
            two_q: rivoli_core::cache::TwoQSplit::default(),
            trace_path: None,
        },
    )?;
    let mut eng = rivoli_engine::glm::engine::GlmEngine::new(pin, &cfg, 4096)?;
    let (out, stats) = eng.generate(
        rivoli_engine::glm::decode::GenSpec {
            prompt: &prompt,
            ngen,
            eos: &[],
        },
        &mut |t| {
            println!("{t}");
            true
        },
    )?;
    eprintln!(
        "generated {} ids at {:.2} tok/s ({} hits / {} misses)",
        out.len(),
        stats.tok_s,
        stats.hits,
        stats.misses
    );
    Ok(())
}
