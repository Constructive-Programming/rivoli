# Running behind llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) puts one OpenAI-compatible port in
front of several backends and starts them on demand. rivoli is one of those backends: it
speaks `/v1/models`, `/v1/chat/completions` (with or without `stream`) and `/health`, and it
does nothing front-door-shaped itself.

This page is the configuration that was run on a Strix Halo box on 2026-11-26 (llama-swap
v276, gfx1151), with the numbers it produced.

## The constraint that shapes the config

`--max-mem` is a literal claim on GPU memory, not a hint. If that many bytes are not free,
the run dies at the allocation:

```console
$ rivoli /models/inkling-2-small --port 18077 --max-mem 200
INFO rivoli_engine::seam: device pool budget 200.0 GiB (--max-mem, literal — no reserve; may OOM)
Error: rivoli_vmm_alloc(204332875776) failed (-2)
```

`-2` is out-of-memory, and it never names the other tenant — because the other tenant is
invisible to HIP. So a resident embedding model or a whisper server beside rivoli is not a
slow model, it is a model that will not start.

llama-swap's groups are the fix: one **exclusive** group holding every model that touches the
GPU. `swap: true` makes the members evict each other; `exclusive: true` makes loading a member
unload every other group's models too.

## Configuration

```yaml
healthCheckTimeout: 360   # a cold rivoli start is a ~1 min pin build
unloadTimeout: 20
startPort: 18000
logLevel: info

routing:
  router:
    use: group
    settings:
      groups:
        gpu:
          swap: true
          exclusive: true
          members: [inkling-2-mini, inkling-2-small]

models:
  inkling-2-mini:
    ttl: 900
    checkEndpoint: /health
    cmd: |
      /usr/local/bin/rivoli
      /models/inkling-2-mini-fp8-mxfp8-art
      --ctx 4096
      --max-mem 64
      --port ${PORT}

  inkling-2-small:
    ttl: 900
    checkEndpoint: /health
    cmd: |
      /usr/local/bin/rivoli
      /models/inkling-2-small-fp8-f4-art
      --ctx 4096
      --max-mem 64
      --port ${PORT}
```

`${PORT}` must appear in `cmd` — llama-swap substitutes a free port from `startPort` up, and
refuses a config that names `${PORT}` anywhere else. The port opens only after the model is
pinned and the KV slabs are built, so `/health` doubles as the readiness signal, which is why
`healthCheckTimeout` has to clear a cold pin rather than a llama.cpp-style warm-up.

Validate before running — but note that `-validate` accepts unknown keys, so a typo is
silent:

```bash
llama-swap -config config.yaml -validate
flock /var/run/sys-gpu.lock -c 'llama-swap -config config.yaml -listen 127.0.0.1:18088'
```

The `flock` is the cross-tenant guard on the box, not a llama-swap feature: llama-swap takes no
lock, and a bare-metal development run is invisible to its groups.

## What was measured

From a clean GPU (18 MiB of GTT used, no `/dev/kfd` holders):

| step | result |
|---|---|
| `GET /v1/models` | both keys, unloaded |
| chat, `stream:false`, 16 tokens | health check passed on port 18000, `200` in **4.9 s**; `reasoning_content` filled, `content` empty, `finish_reason: length` |
| chat, `stream:true`, 16 tokens | one `data:` delta per token, `finish_reason: length`, `data: [DONE]`; `200` in **1.35 s** |
| request the other model | first one SIGTERMed, second health-passed on port 18001, answered in **10.0 s** |
| `POST /api/models/unload` | `{"msg":"ok"}` in **0.097 s**, child gone, GTT back to **18 MiB**, `/dev/kfd` empty |

Direct `SIGTERM` to the child exits in **108 ms**. rivoli installs no signal handler; the
default disposition is the whole shutdown path, which is right for a server that answers one
request at a time and owns no durable state.

## Thinking is not switchable off on Inkling 2

Three arms on the same prompt — `enable_thinking: false`, `reasoning_effort: "none"`, and the
server default — produced **byte-identical** output, thinking included. The switch renders
`Thinking effort level: 0` into the prompt and this checkpoint thinks anyway. What does hold is
that thinking never leaks into `content`: it arrives in `reasoning_content`, and a 16-token
budget spent entirely on reasoning comes back as an empty `content` with
`finish_reason: length`. Budget `max_tokens` for both.

## Containers

rivoli binds **127.0.0.1 only**, and there is no `--host`. llama-swap starts it as a child
process, so the two must live in the same container: a sidecar shares the pod's network
namespace but llama-swap cannot execute a binary it does not have.

Two workable shapes:

- **Unified image.** Derive from `ghcr.io/constructive-programming/rivoli`, copy the
  `llama-swap` binary in, and make llama-swap the entrypoint. llama-swap owns the lifecycle,
  so `ttl`, `unload`, `/running` and the group policy all apply.
- **Proxy-only.** Give the model a `proxy:` and no `cmd` and run rivoli elsewhere in the pod.
  Nothing unloads it, so it holds its memory until it is killed.

Kubernetes specifics, from the deployment this was verified against:

- `privileged: true` (or `/dev/kfd` and `/dev/dri` as explicit devices) plus
  `supplementalGroups: [27, 28]` for the `video` and `render` groups.
- `-listen 0.0.0.0:8080` on llama-swap so a Service can reach it; rivoli stays on loopback.
- The artifact must sit on a filesystem that supports **O_DIRECT** — the cold-expert streamer
  reads with `O_DIRECT` and 4 KiB alignment. NVMe with ext4/xfs/btrfs is fine; NFS and most
  overlayfs layers are not, which is the usual reason a containerised run is far slower than
  the same binary on the host.
- No `mlock` or `RLIMIT_MEMLOCK` tuning: the 125 MiB bounce arena is `hipHostMalloc`, which
  the ROCm runtime pins itself and which counts as GPU memory.
- The container's memory limit is not the budget. `--max-mem` is a GPU-memory claim; weights
  are streamed from NVMe rather than held in the container's resident set.

A common and pragmatic variant is to let `cmd` exec a host-editable script — the live cluster
runs `/var/db/rivoli/run.sh --port ${PORT}` — so `--ctx` and `--max-mem` can be tuned without
rebuilding an image. It requires the rivoli binary to be reachable inside the container at the
path the script names; a host-mounted script that `exec`s a path nothing mounts is a model
that never becomes ready, and llama-swap only reports the child exiting.
