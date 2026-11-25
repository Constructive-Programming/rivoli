# Architecture

rivoli is a Rust workspace with one backend, HIP/ROCm. Its six crates are the layering:
crate edges are the only layering Rust enforces, so the boundaries that matter are crates.

| crate | is | may depend on |
|---|---|---|
| `rivoli-core` | pure planning: residency, spans, legality, gates, tolerances | nothing |
| `rivoli-artifact` | formats, per-model configs, sniffing, tokenizer, converters | core |
| `rivoli-oracles` | frozen references, golden container, anchor readers | core |
| `rivoli-backend` | the waist: HIP streams, launchers, kernels; owns `rocm` | nothing |
| `rivoli-engine` | the interpreter: fetch, executors, per-architecture loops | all above |
| `rivoli` (cli) | thin `main`, `serve` and tool binaries | engine |

## Pure core, imperative shell

`rivoli-core` is total functions over plain data. It emits values (directives, spans,
verdicts) and never acts. `rivoli-engine` is the interpreter that spends those values on
the device. Nothing in core can name a stream, a pointer or a weight format, and the
dependency graph enforces it. That is what makes "residency selects arithmetic" impossible
to write here, rather than merely tested against.

## Types carry the rules

- **Newtypes for units.** Bytes, tokens, layers and expert ids are never bare `usize`.
- **Typestate for lifecycles.** A fetch ticket is consumed by waiting on it, a writer is
  sealed, and an unloaded artifact is a different type from a loaded one.
- **No panicking shortcuts.** `unwrap` and `expect` are denied across the workspace.

## Kernels in Rust

The device kernels are written in Rust (`crates/kernels`) and built with a pinned nightly
toolchain against ROCm's device bitcode. hipcc is not a build input. Every kernel is checked
bit-for-bit against the original HIP C++ implementation it replaced, at a pinned tag.

## The artifact is the model

`rivoli <artifact>` takes a converted artifact directory: a manifest, codebooks, resident
tensors, per-layer expert files and a tokenizer. The artifact names its own architecture.
CLI flags are checked against it at startup, and an illegal one is refused with the reason.
