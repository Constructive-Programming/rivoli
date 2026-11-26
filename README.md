# rivoli

**Decode frontier LLMs bigger than memory on one AMD Strix Halo.** Weights stream from
NVMe **overlapped with ROCm compute** over unified LPDDR5 (GTT). The overlap is the
whole design: a model that fits in memory is the trivial case of the streaming path, and
one that does not is this engine's reason to exist.

Website: [rivoli.constructive.dev](https://rivoli.constructive.dev) ·
Container packages: [`ghcr.io/constructive-programming/rivoli`](https://github.com/Constructive-Programming/rivoli/pkgs/container/rivoli)

Target box: AMD Strix Halo APU (`gfx1151`, Radeon 8060S), 128 GiB unified LPDDR5 (~100 GiB
practical decode budget), dual-NVMe btrfs RAID0 (~14 GB/s O_DIRECT), Rust + HIP/ROCm
with every GPU kernel compiled in Rust on `amdgcn-amd-amdhsa` (no `hipcc` build
dependency).

## Supported models & witnessed speeds

The artifact IS the model — convert a Hugging Face checkpoint once, and `rivoli <dir>`
sniffs the architecture and weight format from `manifest.json` and the tensor headers
(no `--model` flag). All decode numbers below are witnessed release-build measurements on
hot local NVMe under `/var/run/sys-gpu.lock` with an empty KFD contention witness and
18 MiB pre-arm GTT:

| model | params (total / active) | experts (active) | attention | formats | witnessed tok/s | status |
|---|---:|---|---|---|---|---|
| **Inkling 2 Small** | 594B / 17B | 384 (8) | SWA + full, sinks, sconv | MXFP8, FP8 trunk, MXFP4 (`.f4`) | **6.39–7.54** (`FP8+.f4`, 100 GiB) · **3.31** (`FP8+.mxfp8`) · **3.05** (`BF16+.mxfp8`) | **decoding + serving** (`smoke-inkling-small.sh` 17/17; FP8 trunk paired-dNLL PASS) |
| **Inkling 2 Mini** | 26B / 5.4B | 128 (8), middle 12 looped ×3 | SWA + full, sinks, sconv | BF16, FP8 trunk, MXFP8, MXFP4 | **24.19** (`FP8+.mxfp8`, 64 GiB) · **21.78** (`BF16+.mxfp8`) · **12.60** (12 GiB streaming) | **decoding + serving** (8/8 first-party parity; `smoke-inkling.sh` 16/16; FP8 trunk paired-dNLL PASS @ 5,471 tok) |
| **DeepSeek-V4-Flash-0731** | 284B / 13B | 256+1 (6) | MQA, SWA + compressed blocks | native FP4 (`.f4`) | **9.17** (100 GiB, 512 tok) | **decoding + serving** (30/32 parity, near-ties; `smoke-v4.sh` 3/3) |
| **GLM-5.2** | 753B / 40B | 256 (8) | MLA + DSA lightning indexer | int3-vq, int4 | **3.03** (115 GiB, 512 tok, mean of 3 arms) | **decoding + serving** (32/32 parity; `smoke-glm.sh` 12/12) |
| **Muse Glimmer-30B** | 30B dense | — | GQA 32/2, SWA 2048 + full | BF16, FP8 | 2.56 (bf16, pre-framing baseline) | **decoding + serving** (3/3 goldens; DFlash drafter converter + oracle) |
| **Kimi-K3** | 2.8T / 104B | 896+2 (16) | 69 KDA linear + 24 gated MLA | native MXFP4 (`.f4`) | — (first 16-tok decode verified) | **decode-ready** (KDA 2.3e-7 vs reference) |
| **Qwen3.8-Flash-Next** | 180B / 6B | 512+1 (10) | 36 GDN linear + 12 sparse | FP8 | — | **under construction** (S0–S5 built; CLI refuses until S6–S8) |

See [docs/investigations/inkling2-port.md](docs/investigations/inkling2-port.md) and
[docs/00-orientation/INDEX.md](docs/00-orientation/INDEX.md) for full paired-dNLL quality
gates, `PROFILE/tok` breakdowns, and caveats.

## Install from GitHub Packages (GHCR) or Releases

Every semver tag push (`vX.Y.Z` matching `workspace.package.version` in `Cargo.toml`)
builds and publishes both:

1. **GitHub Container Package (GHCR)**: `ghcr.io/constructive-programming/rivoli:<tag>`
   (and `:latest` on stable semver releases), built on `rocm/dev-ubuntu-24.04:7.2.4`
   containing `rivoli` plus all offline converters and `ppl` under `/usr/local/bin/`.
2. **GitHub Release tarball**: `rivoli-<tag>-linux-x86_64-gfx1151.tar.gz` + `.sha256`
   extracted from the published image (requires a host ROCm runtime providing
   `libamdhip64.so`).

### Run the container image

```bash
docker pull ghcr.io/constructive-programming/rivoli:latest

# Benchmark 64 tokens from a local NVMe artifact
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /var/db/rivoli:/models:ro \
  ghcr.io/constructive-programming/rivoli:latest \
  /models/inkling-2-small-fp8-f4-art --max-mem 100 --bench 64

# Serve an OpenAI-compatible HTTP API on loopback
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v /var/db/rivoli:/models:ro \
  --network host \
  ghcr.io/constructive-programming/rivoli:latest \
  /models/inkling-2-small-fp8-f4-art --max-mem 100 --port 8080
```

> Mount artifacts from a filesystem that supports **O_DIRECT** (local NVMe with
> ext4/xfs/btrfs). Avoid NFS or container overlayfs for `.f4` / `.mxfp8` / `.vq3`
> streaming files.

### Cutting a new GitHub Package release

1. Bump `version` under `[workspace.package]` in `Cargo.toml` and update `Cargo.lock`
   (`cargo update --workspace --offline` or `cargo check --no-default-features`).
2. Commit to `main`, tag, and push:
   ```bash
   git tag v0.2.0
   git push origin main v0.2.0
   ```
3. `.github/workflows/release.yml` automatically validates the tag against `Cargo.toml`,
   builds `Dockerfile` for `gfx1151`, pushes `ghcr.io/constructive-programming/rivoli:v0.2.0`
   (plus `0.2.0`, `v0.2.0-gfx1151`, `sha-<short>`, and `latest`), smoke-tests the published
   image, and publishes the GitHub Release tarball.
4. You can also trigger a package build manually via **Actions → release → Run workflow**
   (`workflow_dispatch`, with optional `offload_arch` and `device_verify` inputs).

## Running behind `llama-swap`

[`llama-swap`](https://github.com/mostlygeek/llama-swap) provides lazy on-demand startup,
idle TTL unload, and model swapping in front of `rivoli`. Because `rivoli`'s `--max-mem`
is a literal GTT allocation and HIP cannot see Vulkan co-tenants (such as an embedding
model or whisper server), place all GPU-backed models in one **exclusive** group so
`llama-swap` evicts every other GPU tenant before spawning `rivoli`:

```yaml
healthCheckTimeout: 360   # cold pin build takes ~1 min
unloadTimeout: 20         # SIGTERM -> clean exit measured at 108 ms
startPort: 18000
logLevel: info

routing:
  router:
    use: group
    settings:
      groups:
        gpu:
          swap: true        # one member resident at a time
          exclusive: true   # evicts every other group's models too
          members: [inkling-2-mini, inkling-2-small]

models:
  inkling-2-mini:
    ttl: 900
    checkEndpoint: /health
    cmd: |
      /usr/local/bin/rivoli
      /var/db/rivoli/inkling-2-mini-fp8-mxfp8-art
      --ctx 4096
      --max-mem 64
      --port ${PORT}

  inkling-2-small:
    ttl: 900
    checkEndpoint: /health
    cmd: |
      /usr/local/bin/rivoli
      /var/db/rivoli/inkling-2-small-fp8-f4-art
      --ctx 4096
      --max-mem 100
      --port ${PORT}
```

```bash
llama-swap -config config.yaml -validate
flock /var/run/sys-gpu.lock -c 'llama-swap -config config.yaml -listen 127.0.0.1:18088'
```

`rivoli` binds `127.0.0.1` only (no `--host`), so in Kubernetes/Docker run `llama-swap` as
the parent process in the same container (deriving from
`ghcr.io/constructive-programming/rivoli` with `llama-swap` copied in). Full verified
experiment transcript, Kubernetes notes, and Inkling thinking behavior:
[docs/reference/llama-swap.md](docs/reference/llama-swap.md).

## Compile from source

Requirements:

- ROCm runtime with `libamdhip64.so` and device bitcode (`ocml.bc`,
  `oclc_isa_version_1151.bc`, etc.) — `hipcc` is **not** required.
- `rustup` with stable Rust plus the nightly channel + `rust-src` pinned in
  `crates/kernels/rust-toolchain.toml`.

```bash
# Build the server (release profile for real runs / benchmarks)
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release -p rivoli --bin rivoli

# Build server + all offline converters + paired-dNLL tool
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release -p rivoli --bins

# Optional: include teacher-forcing `--ppl` support
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release -p rivoli --features teacher-forcing --bin rivoli --bin ppl

# Install to user PATH
install -Dm755 /var/cache/rivoli/target/rivoli/release/rivoli "$HOME/.local/bin/rivoli"
```

## Convert checkpoints

One converter per architecture lives under `crates/cli/src/bin/`:

- `convert` (GLM-5.2 `.vq3`) + `fp8_to_i4` / `add_indexer`
- `convert_v4` (DeepSeek-V4-Flash `.f4`)
- `convert_glimmer` (`--fp8` optional) + `convert_glimmer_drafter`
- `convert_k3` (Kimi-K3 `.f4`)
- `convert_qwen` (Qwen3.8-Flash-Next)
- `convert_inkling` (Inkling 2 Mini: default BF16, `--mxfp8`, `--f4`, `--fp8` dense trunk)
- `convert_inkling_small` (Inkling 2 Small: native `.mxfp8`, `--fp8` dense trunk, `--f4`
  MXFP4 experts, and `--from-mxfp8` to convert an existing `.mxfp8` artifact locally at NVMe
  speed — hardlinking experts when `--f4` is omitted)

Example (Inkling 2 Small FP8 dense trunk + MXFP4 routed experts from an existing `.mxfp8`
artifact):

```bash
convert_inkling_small /var/db/rivoli/inkling-2-small-mxfp8-art \
  /var/db/rivoli/inkling-2-small-fp8-f4-art --from-mxfp8 --fp8 --f4
```

## Run

```bash
# Decode 64 tokens and print text + DECODE + PROFILE/tok
rivoli /var/db/rivoli/<model> --max-mem 100 --bench 64

# Serve OpenAI-compatible HTTP on 127.0.0.1:8080
rivoli /var/db/rivoli/<model> --max-mem 100 --port 8080

# Teacher-forced per-token NLL (requires `--features teacher-forcing` build)
rivoli /var/db/rivoli/<model> --max-mem 100 --ppl tests/ppl-corpus.txt --ppl-out run.nll
ppl baseline.nll candidate.nll
```

Flags are checked against a per-architecture legality table at startup (`--mode`, `--attn`,
`--cache-policy lru|2q|arc`, `--max-mem`, `--ctx`, `--think`, `--dump-ids`, `--trace`).
Serving implements `GET /health`, `GET /v1/models`, and `POST /v1/chat/completions`
(streaming SSE and non-streaming), greedy argmax, one request at a time.

## Work on rivoli

`default = ["rocm"]` at every level of the DAG: a bare `cargo build` is the device build,
and a bare `cargo test --workspace` touches the GPU — wrap it in `flock` and serialize
with `--test-threads=1`. The deviceless refusal-stub arm (`--no-default-features`) is what
CI runs on GitHub-hosted runners:

```bash
flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace -- --test-threads=1'
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo clippy --workspace --all-targets
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo clippy --workspace --all-targets --no-default-features
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release
```

## Workspace layout

| crate | role | dependencies |
|---|---|---|
| `rivoli-core` | pure planning: residency partition, legality table, cache policies, tolerances | none |
| `rivoli-abi` | `no_std` kernarg & geometry structs shared by host and device | none |
| `rivoli-kernels` | all GPU kernels in Rust (`amdgcn-amd-amdhsa`, pinned nightly) | `abi` |
| `rivoli-artifact` | formats, configs, sniffing, tiktoken/BPE tokenizers, dequant/quantizers | `core` |
| `rivoli-oracles` | frozen first-party references, golden container, anchor readers | `core` |
| `rivoli-backend` | HIP runtime FFI, streams/timelines, embedded HSACO loader, launchers | `abi` |
| `rivoli-engine` | io_uring O_DIRECT streamer, routed pool, per-arch decode loops, `Engine` | all above |
| `rivoli` (cli) | `rivoli` server/bench, offline converters, `ppl`, workspace meta-gates | `engine` |

## Reading the tree

1. [docs/00-orientation/TOUR.md](docs/00-orientation/TOUR.md) — two-page orientation.
2. [docs/00-orientation/INDEX.md](docs/00-orientation/INDEX.md) — every doc with status,
   scope, and one-line verdict (enforced by `crates/cli/tests/docs.rs`).
3. [docs/reference/principles.md](docs/reference/principles.md) — the seven principles.
4. [docs/reference/llama-swap.md](docs/reference/llama-swap.md) — live-verified
   `llama-swap` + Kubernetes deployment guide.
5. [CLAUDE.md](CLAUDE.md) — gates, build cache rules, and measurement discipline.
