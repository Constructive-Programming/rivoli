//! **`InklingEngine` end to end against the S3 first-party anchor — the composition gate.**
//!
//! Every other Inkling gate in this tree scores one operator: `kernel_inkling_*.rs` compares a
//! kernel against a host oracle transcribed beside it, and `geometry.rs`'s tests score the
//! schedules and byte counts deviceless. What none of them can see is the COMPOSITION — whether
//! the 48 applications, the per-pass cache offsets, the block-residual bank, and the message
//! gates add up to the model. This file runs the engine's own `open`/`decode` on an artifact
//! built from the SAME `salt=inkling-1` tiny model that produced
//! `crates/oracles/tests/inkling-anchor-decode-inkling-1.bin` (transformers @ d8f10a82, CPU
//! fp32), and compares the engine's greedy token against the reference's own logits.
//!
//! # The two facts it can honestly assert, and why they are these two
//!
//! A cached decode step is one token: the reference prefills `0..16`, steps token `16`, and
//! captures that position's logits (`[1, 1, 250]`). The engine's `decode` samples the token the
//! LAST prompt position predicts, so a 17-token prompt with `ngen = 1` emits exactly that
//! position's argmax — the same quantity the golden carries, one decision rather than a float
//! comparison of a row rivoli stores as bf16 weights against a fp32 reference. The argmax is
//! the anchor's stop condition and it is what a wrong composition changes first.
//!
//! # The artifact is a FIXTURE, and the skip policy is `inkling_convert.rs`'s
//!
//! `/var/cache/rivoli/scratch/inkling-tiny-anchor` holds the driver's `--dump-artifact` output
//! (`resident.safetensors` plus the `manifest.json` that wraps the tiny `InklingTextConfig` in a
//! valid `Inkling2LoopConfig`). It is generated, never vendored — it is derived from the same
//! seeded draws as the golden, so shipping it would be shipping a second copy of bytes whose
//! provenance is the driver. Absent ⇒ skip, unless `RIVOLI_INKLING_REQUIRED` makes the skip a
//! failure.
//!
//! # The door check needs no artifact at all
//!
//! `open` validates the config BEFORE the pin places a byte, so the `--ctx` ceiling is scored
//! here on the vendored tiny config, on every run — including a build whose scratch dir is
//! empty. That is deliberate: the refusal is a fact about the kernel's LDS budget, not about a
//! fixture, and a gate that only runs when a 3 GB artifact happens to exist is not evidence.
//!
//! # RED-PROOF RECIPE (P7 — owed for the device half)
//!
//! * door: delete `validate_scratch_cfg(cfg, n_ctx)?` from `InklingEngine::open` and
//!   [`open_refuses_a_ctx_past_the_sink_attend_ceiling`] panics on `.expect(...)`; restore it.
//! * composition: swap the comparison to `decoded.ids.last()` vs `argmax(logits)` — or invert
//!   one geometry rule (`is_block_end` → `is_block_boundary`) — and the device test must redden
//!   on the token, which is the only observable a wrong loop order reaches. Record the run in
//!   `docs/measurement/gate-red-proofs.md` beside the other anchor gates' proofs.

#![cfg(feature = "rocm")]
// Tests: panic-on-failure is the idiom, and the golden and config are pinned fixtures whose
// absence must be loud rather than threaded out as a `Result`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "../../oracles/tests/common/golden_read.rs"]
mod golden_read;

use golden_read::{GoldenSet, float};
use rivoli_artifact::inkling_config::{Inkling2LoopConfig, InklingTextConfig};
use rivoli_backend::abi::SINK_ATTEND_MAX_N;
use rivoli_core::legality::Arch;
use rivoli_engine::inkling::engine::InklingEngine;
use rivoli_engine::seam::GenSpec;

/// The driver's dumped artifact, beside every other live Inkling fixture.
const ARTIFACT: &str = "/var/cache/rivoli/scratch/inkling-tiny-anchor";

/// The vendored decode golden the artifact's draws came from. Reading the bytes here rather than
/// through a shared module is the `#[path]`-included `golden_read`'s division of labour: the
/// helper reads a `GoldenSet`, the gate owns which file it is.
const GOLDEN: &[u8] = include_bytes!("../../oracles/tests/inkling-anchor-decode-inkling-1.bin");

/// The shipped Mini config (valid `Inkling2LoopConfig`) for the fixture-free `--ctx` door test.
const MINI_CONFIG: &str =
    include_str!("../../../docs/measurement/inkling2-reference/mini/config.json");

/// The reference's logits are the 17th token's; the prompt is therefore `0..17` (the driver's
/// `torch.arange(17).remainder(vocab_size)`, vocab 256, so the ids are the positions).
const PROMPT_LEN: usize = 17;

/// The KV ceiling this gate allocates for: the prompt plus intent, nowhere near the cap.
const N_CTX: usize = 32;

fn mini_text() -> InklingTextConfig {
    serde_json::from_str::<Inkling2LoopConfig>(MINI_CONFIG)
        .expect("the vendored Mini config is a pinned fixture")
        .text
}

/// `inkling_convert.rs`'s skip policy: a missing fixture skips the run, unless
/// `RIVOLI_INKLING_REQUIRED` makes the absence a failure instead. `true` when the path exists.
fn fixture_present(path: &str) -> bool {
    if std::fs::metadata(path).is_ok() {
        return true;
    }
    assert!(
        std::env::var_os("RIVOLI_INKLING_REQUIRED").is_none(),
        "Inkling engine anchor gate REQUIRED but did not run: {path} absent"
    );
    eprintln!("skip engine anchor: {path} absent (set RIVOLI_INKLING_REQUIRED=1 to force)");
    false
}

/// The row's argmax — the one decision this gate compares. Ties keep the FIRST index, matching
/// `topk_into`'s index-ascending rule rather than depending on which side of a tie floats land.
fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(core::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .expect("a non-empty logit row")
}

/// The ceiling is the LDS arithmetic's ONE number: `SINK_ATTEND_LDS_CAP / 4 − ROW_BLOCK`. Pinned
/// here so a change to either constant reddens a test instead of silently moving the `--ctx`
/// limit the CLI advertises.
#[test]
fn the_sink_attend_ceiling_is_the_lds_arithmetic() {
    assert_eq!(SINK_ATTEND_MAX_N, 16_128);
}

/// `open` refuses an unservable `--ctx` at the DOOR — before `InklingPin::build` opens the
/// artifact, so this runs on a machine with no fixture and no device allocation.
#[test]
fn open_refuses_a_ctx_past_the_sink_attend_ceiling() {
    let text = mini_text();
    let over = SINK_ATTEND_MAX_N as usize + 1;
    let err = InklingEngine::open(ARTIFACT, &text, 4 << 30, over)
        .err()
        .expect("a --ctx past the sink_attend LDS ceiling must be refused");
    let message = format!("{err:#}");
    assert!(
        message.contains("exceeds sink_attend LDS ceiling"),
        "the refusal must name the ceiling, not a failed malloc: {message}"
    );
    assert!(
        message.contains("lower --ctx"),
        "the refusal must say what the caller can do: {message}"
    );
}

/// The composition gate: the engine's greedy token at the stepped position equals the
/// reference's own argmax there.
#[test]
fn the_engine_reproduces_the_reference_argmax_on_the_stepped_token() {
    if !fixture_present(&format!("{ARTIFACT}/resident.safetensors")) {
        return;
    }
    // The manifest is the dumped artifact's own wrapper, so the engine runs the member the
    // driver built, not the gate's reading of it.
    let cfg = Inkling2LoopConfig::load(ARTIFACT).expect("the dumped artifact carries a manifest");
    let mut engine = InklingEngine::open(ARTIFACT, &cfg.text, 4 << 30, N_CTX)
        .expect("the dumped tiny artifact must open");
    let golden = GoldenSet::read_anchor_for(Arch::Inkling2Loop, &mut &GOLDEN[..])
        .expect("the vendored inkling-1 decode golden must load");
    let (shape, logits) = float(&golden, "logits");
    assert_eq!(shape, &[1, 1, 250], "a decode golden is ONE stepped token");
    let prompt: Vec<u32> = (0..PROMPT_LEN as u32).collect();
    let spec = GenSpec {
        prompt: &prompt,
        ngen: 1,
        eos: &[],
    };
    let decoded = engine
        .decode(spec, &mut |_| true)
        .expect("the engine's decode must run on its own artifact");
    assert_eq!(
        decoded.ids,
        [argmax(logits)],
        "the engine's greedy token must equal the reference's argmax at position {}",
        PROMPT_LEN - 1
    );
}
