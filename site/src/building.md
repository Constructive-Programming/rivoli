# Building

## Requirements

- A ROCm unified memory system (an AMD APU), with ROCm's device bitcode (`ocml.bc` and friends)
  and `libamdhip64`.
- `rustup`, with the nightly toolchain pinned in `crates/kernels/rust-toolchain.toml` for
  the device crate.
- Fast NVMe for the streamed weights.

The default build **is** the ROCm build: `default = ["rocm"]` everywhere. The deviceless
build, which is what CI runs, is `--no-default-features`.

## Commands

```bash
git clone https://github.com/Constructive-Programming/rivoli
cd rivoli

# Deviceless tests (safe anywhere, what CI runs)
cargo test --workspace --no-default-features

# Full battery, host and device. Device suites must run single-threaded.
cargo test --workspace -- --test-threads=1

# Release build: for benchmarks and performance work only
cargo build --release
```

A run that is not timing something should use the dev profile: release builds compile out
every `debug_assert!`.

## Running

```bash
# Decode 64 tokens and print the text plus a DECODE timing line
rivoli /path/to/artifact --bench 64

# Serve an OpenAI-compatible API (/v1/models, chat completions, SSE streaming)
rivoli /path/to/artifact --port 8080
```

Artifacts are built from Hugging Face checkpoints with the per-model `convert_*` binaries.

## Container image

Prebuilt ROCm (`gfx1151`) images containing `rivoli` and all offline converters are
published to GitHub Packages on every version tag:

```bash
docker pull ghcr.io/constructive-programming/rivoli:latest
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v /var/db/rivoli:/models:ro ghcr.io/constructive-programming/rivoli:latest \
  /models/inkling-2-small-fp8-f4-art --max-mem 100 --bench 64
```

To put several models behind one OpenAI endpoint, with swapping, idle unload and a clean
SIGTERM path, run rivoli as a llama-swap backend: see
[Running behind llama-swap](llama-swap.md) for the verified `config.yaml` and the one
constraint that shapes it (rivoli's `--max-mem` is a literal GPU-memory claim, so it cannot
share the device with another loaded model).
