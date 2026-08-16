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
    // Logs on stderr: stdout is the id stream the parity gate diffs, and a log line
    // interleaved into it would read as a mismatch.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
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
    // The routed FORMAT is GLM's own argument and the three pool knobs are the shared
    // `PinCfg` — see `rivoli_engine::PoolKnobs` for why V4 has the second and not the first.
    let pin = rivoli_engine::glm::pin::GlmPin::build(
        dir,
        &cfg,
        rivoli_artifact::format::RoutedFmt::Vq3,
        rivoli_engine::resident::PinCfg {
            capacity,
            cache_policy: "2q",
            two_q: rivoli_core::cache::TwoQSplit::default(),
            trace_path: None,
        },
    )?;
    let mut eng = rivoli_engine::glm::engine::GlmEngine::new(pin, &cfg, 4096)?;
    let out = eng.generate(
        rivoli_engine::GenSpec {
            prompt: &prompt,
            ngen,
            eos: &[],
        },
        &mut |t| {
            println!("{t}");
            true
        },
    )?;
    let s = &out.stats;
    eprintln!(
        "generated {} ids at {:.2} tok/s ({} hits / {} misses)",
        out.ids.len(),
        s.tok_s,
        s.hits,
        s.misses
    );
    Ok(())
}
