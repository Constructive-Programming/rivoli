//! **The DeepSeek-V4-Flash numerical oracle** — a CPU transliteration of
//! `inference/model.py`'s `Transformer.forward`, and the gate S2/S3 are scored against.
//!
//! The model name is deliberate (kept through the 2026-08-09 rename pass): a reference
//! transliteration is ABOUT one model, and generalising its name would be a lie in the
//! opposite direction. The engine it judges is behaviour-named (`f4gpu`); the yardstick is
//! named for what it measures against.
//!
//! Read `forward.rs`'s module doc first: it states what is reproduced exactly, what is
//! reproduced only up to summation order, and what is out of scope.
//!
//! > **AMENDED 2026-08-11.** One file here is no longer V4-only: `golden.rs` is the shared
//! > container, and it gained `read_k3` for Kimi-K3's S1b anchor goldens
//! > (`docs/measurement/k3-reference/anchor.md`). The argument above still holds for everything
//! > else — `forward.rs`, `numerics.rs`, `weights.rs` and `toy.rs` ARE a V4 transliteration and
//! > are named for what they measure against. `golden.rs` is not; it is a length-prefixed tensor
//! > file with a per-model magic, and K3's producer is the reference's own PyTorch, not a
//! > transliteration at all. It lives here on cost — moving it would touch this module, the
//! > `v4-oracle` bin, `tests/f4_loop.rs` and a probe script that asserts the magic bytes — and
//! > review is right that the reasoning bit at the SECOND model rather than a third. If that move
//! > is ever paid for, `golden.rs` is what goes.
//!
//! # Why this exists before the thing it judges
//!
//! Every defect in a V4 port is silent-wrong. A missing QK-norm, RoPE on the wrong dims, an
//! unclamped SwiGLU, a mis-grouped output projection, the partial fp8 KV quantization
//! applied to the whole tensor — none crash, all produce fluent wrong text, and the two
//! cheap corpus statistics this repo has (`distinct`, `longest repeated block`) fire
//! identically on a repetition loop, on spliced corruption and on ordinary prose
//! (CLAUDE.md). So the instrument is built first, and it is proved before it is trusted.
//!
//! # The gate is proved, not asserted
//!
//! [`forward::Defect`] enumerates ~40 deliberate breakages, each a transcription slip a
//! competent implementer actually makes. `tests/v4_oracle.rs` runs every one across a grid
//! of (layer class x prefill/decode x prompt length) and asserts BOTH halves for each: the
//! cases it must perturb, and the cases it must leave bit-identical. A breakage that moves
//! every golden proves nothing about the gate's resolution, so a defect with no declared
//! silent case is itself a test failure.
//!
//! # Not the engine
//!
//! Nothing here imports `gpu.rs`, `attn.rs`, `math.rs`, `artifact/` or any kernel, and it
//! must stay that way: an oracle that shares code with the implementation it judges cannot
//! see a bug they share. That includes the safetensors reader, which is written here rather
//! than borrowed from `src/artifact/`.

pub mod forward;
pub mod numerics;
pub mod toy;
pub mod weights;
