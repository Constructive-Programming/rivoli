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

mod common;

fn main() -> Result<()> {
    // `common::start` puts the log sink on stderr, which matters here: stdout is the id stream
    // the parity gate diffs and an interleaved log line reads as a mismatch.
    let args = common::start();
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
        // The STOCK configuration, spelled once at `PinCfg::stock`: this example is the
        // parity gate's arm and must stay the historical path — every candidate-fix and
        // diagnostic arm is evaluated through the CLI, never here.
        rivoli_engine::resident::PinCfg::stock(capacity),
    )?;
    let mut eng = rivoli_engine::glm::engine::GlmEngine::new(pin, &cfg, 4096)?;
    let (out, stats) = eng.generate(
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
    eprintln!(
        "generated {} ids at {:.2} tok/s ({} hits / {} misses)",
        out.len(),
        stats.tok_s,
        stats.hits,
        stats.misses
    );
    Ok(())
}
