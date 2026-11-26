---
status: live
scope: engine
verdict: VERIFIED LIVE 2026-11-26 on rh-anine (gfx1151, llama-swap v276, rivoli release build) — llama-swap lazy-spawns rivoli on the first chat request, GET /health 200 is what its readiness check reads, non-stream and SSE chat both answer with the request's own model name echoed back, an exclusive group swaps one rivoli member for another, and POST /api/models/unload SIGTERMs the child (gone in 108 ms) leaving GTT at 18 MiB and /dev/kfd empty. rivoli binds 127.0.0.1 ONLY and has no --host, so llama-swap must be the PARENT PROCESS in the same container; a sidecar cannot spawn it and can only proxy a rivoli that something else started. Inkling 2's enable_thinking:false is advisory — the template renders "Thinking effort level: 0" and the checkpoint still thinks, though the channels stay separate.
---

# rivoli behind llama-swap

llama-swap is the front door: one OpenAI-compatible port in front of several backends, which
it starts on demand, evicts by policy and unloads on TTL. rivoli is one of those backends and
does nothing front-door-shaped itself — no model swapping, no auth, no TLS, no batching.

Everything below was executed on this box on 2026-11-26. Where a statement is derived rather
than observed it says so.

## The one hard constraint: the GTT budget is a literal claim

`--max-mem` is not a hint and not a soft cap. It is the size of the VMM allocation the routed
pool asks HIP for, and if that many bytes are not free the run dies at the allocation:

```console
$ rivoli /var/db/rivoli/inkling-2-small-fp8-f4-art --port 18077 --max-mem 200
INFO rivoli_engine::seam: device pool budget 200.0 GiB (--max-mem, literal — no reserve; may OOM)
INFO rivoli_engine::resident: Inkling-2-Small partition: 12068 of 18432 routed experts fit
     resident beyond the 8 batch slots (190.3 GiB pool)
Error: rivoli_vmm_alloc(204332875776) failed (-2)
$ echo $?
1
```

`-2` is `hipErrorOutOfMemory`. That is the shape of every "rivoli will not start" report: a
bare allocation failure, no mention of the other tenant, because the other tenant is invisible
to HIP. It is also why the memory rule is a **scheduling** rule and not a tuning rule.

Two consequences llama-swap's config must encode:

1. **No Vulkan model may be resident beside rivoli.** A resident embedder or whisper server
   holds ~5 GiB / ~3 GiB of the 116 GiB pool. With `--max-mem` at or near the pool, the
   allocation above is exactly what you get instead of a slow model.
   The automatic budget (`--max-mem` omitted: `free − 16 GiB`) does adapt, so a small co-tenant
   costs residency and hit rate rather than the run — but it also makes tok/s depend on who
   else happens to be loaded, which is the opposite of a benchmarkable box.
2. **`groups` with `swap: true` and `exclusive: true` is the fix, and it is not cosmetic.**
   `swap` makes the group's members evict each other; `exclusive` makes loading a member
   unload *every other group's* models. One exclusive group holding every model that touches
   the GPU is the whole configuration.

The live cluster runs this shape today, in the legacy top-level `matrix:` form, with rivoli as
its own branch precisely so that requesting it evicts the pinned embedder and whisper:

```yaml
matrix:
  sets:
    # r is its own branch, NOT OR'd into the "w & q & (...)" group: its --max-mem
    # defaults near the full GTT ceiling, so it cannot coexist with the pinned
    # embedder/whisper the way every other tier here does.
    main: "(w & q & ((wt & (l | m)) | h | u | o | g | n)) | r"
```

Both spellings are accepted by v276. The `routing.router` form below is the one the docs
upstream recommend for new configs and the one verified here.

## The config that was verified

`/var/cache/rivoli/scratch/llama-swap-exp/config.yaml`, validated with
`llama-swap -config config.yaml -validate` → `config is valid: 2 model(s), 0 peer(s)`:

```yaml
healthCheckTimeout: 360   # a cold rivoli start is a ~1 min pin build; the llama.cpp default is far too short
unloadTimeout: 20         # SIGTERM -> exit measured at 108 ms; this only has to cover a page-in
startPort: 18000
logLevel: info

routing:
  router:
    use: group
    settings:
      groups:
        gpu:
          swap: true        # one member at a time
          exclusive: true   # a member evicts every other group's models too
          members: [inkling-2-mini, inkling-2-small]

models:
  inkling-2-mini:
    ttl: 900
    checkEndpoint: /health
    cmd: |
      /var/cache/rivoli/target/loggerhead/release/rivoli
      /var/db/rivoli/inkling-2-mini-fp8-mxfp8-art
      --ctx 4096
      --max-mem 64
      --port ${PORT}

  inkling-2-small:
    ttl: 900
    checkEndpoint: /health
    cmd: |
      /var/cache/rivoli/target/loggerhead/release/rivoli
      /var/db/rivoli/inkling-2-small-fp8-f4-art
      --ctx 4096
      --max-mem 64
      --port ${PORT}
```

| key | why this value |
|---|---|
| `checkEndpoint: /health` | rivoli answers `GET /health` with `200 {"status":"ok"}`. It is the same route llama.cpp serves, so the default would also work; naming it is documentation, not necessity. |
| `cmd` with `${PORT}` | llama-swap substitutes a free port from `startPort` upward and passes it as `--port`. **`${PORT}` must appear in `cmd`**: a model whose `proxy:` names `${PORT}` while `cmd` does not is refused at validate (`proxy uses ${PORT} but cmd does not`). |
| multi-line `cmd` | newlines are whitespace; one flag per line keeps a long rivoli command line readable and diffable. |
| `healthCheckTimeout: 360` | the port only opens **after** the model is pinned and the KV slabs are built, so health is the readiness signal. Measured: 1.2 s with the artifact in page cache, ~1 min cold. Too short a timeout aborts the child mid-pin. |
| `ttl: 900` | `-1` (the default) inherits `globalTTL`, `0` never unloads, `>0` is idle seconds. 900 keeps a warm model through a conversation and frees 26–69 GiB afterwards. |
| `--max-mem 64` | a literal 64 GiB claim on the 116 GiB pool: Mini's whole 23.6 GiB routed set is resident inside it, Small's 290 GiB routed set is 64 GiB resident / the rest streamed. |
| `swap: true`, `exclusive: true` | see above. |

## What each side serves

| endpoint | served by | behaviour |
|---|---|---|
| `GET /health` | rivoli | `200 {"status":"ok"}`, `Content-Length: 15`, `Connection: close` |
| `GET /v1/models` | rivoli | one entry, id = the artifact directory's basename |
| `POST /v1/chat/completions` | rivoli | `stream` true or false; `reasoning_content` + `content` |
| `GET /v1/models` | llama-swap | its own model keys, `owned_by: llama-swap`, each with `status.value` |
| `GET /running` | llama-swap | loaded models with their `cmd`, `proxy` URL and `ttl` |
| `POST /api/models/unload[/MODEL]` | llama-swap | SIGTERM the child, wait `unloadTimeout`, then SIGKILL |
| `POST /api/models/pause` / `resume` | llama-swap | unload everything and 503 every local model request |
| `POST /v1/completions` | neither | rivoli answers `404`; the endpoint is deliberately absent — a raw prompt leaves the model outside an assistant turn, where its EOS ids are unreachable |

**Model identity.** rivoli reports the artifact directory's basename on its own `/v1/models`
(`inkling-2-mini-fp8-mxfp8-art`), while llama-swap advertises the config's model key
(`inkling-2-mini`). Responses echo the `model` field of the **request**, so a client that asks
for `inkling-2-mini` gets `"model":"inkling-2-mini"` back and never sees the directory name.
No `--model-id` flag exists and none is needed for the llama-swap path.

## The verified run

Under `flock /var/run/sys-gpu.lock`, from a clean GPU (GTT 18,702,336 B ≈ 18 MiB, `/dev/kfd`
with no holders), `llama-swap -config config.yaml -listen 127.0.0.1:18088`:

| step | request | observed |
|---|---|---|
| advertise | `GET /v1/models` | both keys, `status.value: unloaded`; `/running` → `{"running":[]}` |
| cold start + answer | `POST /v1/chat/completions` `stream:false`, `max_tokens:16` | `[INFO] <inkling-2-mini> Health check passed on http://localhost:18000/health`, then `200` in **4.91 s**; `finish_reason:"length"`, `content:""`, `reasoning_content:"The user asks for a one-sentence definition of a mixture-of"`, `usage` 27/16/43. GTT **29,681,303,552 B (27.6 GiB)** |
| `/running` | `GET /running` | `{"model":"inkling-2-mini","state":"ready","cmd":"…--port 18000","proxy":"http://localhost:18000","ttl":900}` |
| SSE | same request with `stream:true` | 16 `data:` chunks (`delta.reasoning_content` per token), a final chunk with `finish_reason:"length"`, then `data: [DONE]`; `200` in **1.35 s** |
| swap | `POST` for `inkling-2-small` with `max_tokens:8` | llama-swap SIGTERMed mini, health-passed small on port **18001**, answered `"content":"hello"`, `finish_reason:"stop"`; `200` in **9.97 s**. `/running` listed small only. GTT **68,884,598,784 B (64.1 GiB)** |
| unload | `POST /api/models/unload` | `{"msg":"ok"}` in **0.097 s**; `/running` → `{"running":[]}`; GTT back to **18,702,336 B**; `/dev/kfd` empty; no rivoli process left |

Direct SIGTERM to the child was measured separately: `kill -TERM` → process gone in **108 ms**,
GTT back to 18 MiB, and llama-swap logged
`[WARN] <inkling-2-mini> upstream process exited unexpectedly` and cleared `/running` on its
own. rivoli installs no signal handler — the default disposition is the whole shutdown path,
which is correct for a server that handles one request at a time and owns no durable state.

## Thinking, and what `enable_thinking:false` actually does

Measured on `inkling-2-mini`, same prompt, `max_tokens: 96`, greedy:

| request | `content` | `reasoning_content` | `usage.completion_tokens` |
|---|---|---|---|
| `enable_thinking: false` | `pong` | `The user asks to reply with exactly "pong". Need final exactly pong. No extra words. Ensure no punctuation.` | 33 |
| `reasoning_effort: "none"` | identical | identical | 33 |
| (nothing; server default) | identical | identical | 33 |

So on Inkling 2 the switch is **advisory**: it renders `Thinking effort level: 0` into the
prompt (the template's `none → 0.0`), and this checkpoint thinks anyway. The three arms were
byte-identical, which is the honest form of the finding — the knob changed nothing at all on
this model. What *does* hold is the channel separation: no `<|content_thinking|>` bytes leak
into `content`, and with a 16-token budget the reply is an empty `content` plus a
`finish_reason:"length"` because the budget went to reasoning. Size `max_tokens` for thinking
plus answer, and treat `reasoning_content` as the place thinking goes, not as something a
client can switch off.

## Deployment

### Bare metal (verified)

```bash
# 1. GPU idle: ~18 MiB GTT and no /dev/kfd holders
cat /sys/class/drm/card*/device/mem_info_gtt_used
fuser -v /dev/kfd

# 2. validate the config, then run llama-swap under the lock
llama-swap -config /var/cache/rivoli/scratch/llama-swap-exp/config.yaml -validate
flock /var/run/sys-gpu.lock -c \
  'llama-swap -config /var/cache/rivoli/scratch/llama-swap-exp/config.yaml -listen 127.0.0.1:18088'
```

`-validate` is worth running but it is not a linter: **unknown keys are silently ignored**
(measured — `bogusTop: 1` and a bogus per-model field both validate clean). A typo'd
`checkEndpoint` is accepted and simply does nothing.

llama-swap's `groups` only protect rivoli from llama-swap's *own* models. A bare-metal dev or
CI run holding `/var/run/sys-gpu.lock` is invisible to it, so every model whose `cmd` starts a
GPU process should begin with a lock-wait wrapper (the cluster's `gpu-lock-wait.sh`), exactly
as the live config does. The reverse is also true: run the experiment above *under* the lock,
because llama-swap itself takes no lock.

### Kubernetes / container

The live `ai/llama-swap-rh-anine` Deployment, read on 2026-11-26, is the reference shape:

| field | value | why |
|---|---|---|
| image | `ghcr.io/mostlygeek/llama-swap:unified-vulkan@sha256:c0f55fd2…` | ships `llama-swap`, `llama-server` and `whisper-server` in one container |
| args | `-config /etc/llama-swap/config/config.yaml -listen 0.0.0.0:8080 -watch-config` | `0.0.0.0` is required for the Service; `-watch-config` reloads on ConfigMap change |
| securityContext | `privileged: true`, `supplementalGroups: [27, 28]` | whole `/dev` (so `/dev/kfd` and `/dev/dri` both), plus the Debian `video`/`render` groups |
| volumeMounts | `/dev/dri`, `/var/run/sys-gpu.lock`, `/var/db/rivoli` (ro), `/swarm/storage/ai/models` (ro) | device nodes, the cross-tenant lock file, and the artifacts |
| probes | `GET /` on port 8080 | llama-swap's UI route, not a model's `/health` |
| resources | `requests 500m/24Gi`, `limits 48Gi` | rivoli's budget is GTT, not the container's memory limit; keep the limit above the host-side page cache you want it to keep |

**One container, not a sidecar.** rivoli binds `127.0.0.1` only — `TcpListener::bind(("127.0.0.1",
opts.port))`, with no `--host` flag — and llama-swap starts it as a child process. A sidecar
container shares the pod's network namespace, so a *proxy* to `127.0.0.1:PORT` would reach it,
but llama-swap cannot spawn a binary it does not have. Two workable shapes:

1. **Unified image (the one that lets llama-swap own the lifecycle).** Derive from
   `ghcr.io/constructive-programming/rivoli` (its runtime stage is
   `rocm/dev-ubuntu-24.04:7.2.4` with `rivoli` in `/usr/local/bin`), copy the `llama-swap`
   binary in, and make llama-swap the entrypoint. Same namespace, same PID tree, and
   `cmd: rivoli /models/<artifact> …` is a plain child process.
2. **Proxy-only (llama-swap does not own the lifecycle).** Give the model a `proxy:` and no
   `cmd` — `-validate` accepts `models: {ext: {proxy: http://127.0.0.1:19000}}` — and run
   rivoli elsewhere in the pod. The trade is that `ttl`, `unload`, `/running` and the group
   policy no longer apply: nothing unloads it, so it holds its GTT until it is killed.

The live cluster takes a third, pragmatic path that is worth copying: llama-swap runs from its
own image and `cmd` execs a **host-editable script** under a mounted path
(`/var/db/rivoli/run.sh --port ${PORT}`), so the flags that matter (`--ctx`, `--max-mem`, the
artifact) can be tuned without a fleet commit. That only works because the rivoli binary is
reachable from inside the container — mount it, or derive the image.

> **Observed defect in that path, 2026-11-26.** Inside the live container,
> `/var/db/rivoli/run.sh` exists (hostPath) but the binary its `exec` line names —
> `/home/rhansen/workspace/constructive.dev/rivoli/target/release/rivoli`, the OLD tree — does
> not, and no volume mounts it. The `glm-5.2-rivoli` row therefore cannot spawn: llama-swap
> reports the child exiting and the model never becomes ready. Mount the binary (or its
> directory) at the path the script names, or point the script at a binary that is inside the
> image. `/dev/kfd` and `/dev/dri` are both present under `privileged: true`, so the device
> side of that row is fine.

Storage notes, from the engine's own I/O path:

- The artifact must sit on a filesystem that supports **O_DIRECT** — the cold-expert streamer
  opens with `O_DIRECT` and a 4 KiB alignment requirement. NVMe with ext4/xfs/btrfs is fine;
  NFS and most overlayfs layers are not. This is the single most common container mistake:
  the artifact ends up on the container's writable layer instead of a mounted volume.
- Reads go through **io_uring** (`io-uring` crate, direct syscalls, no liburing). Needs a
  kernel with `io_uring_disabled=0`; nothing to configure in the container.
- **No `mlock` / `RLIMIT_MEMLOCK` requirement.** The 125 MiB bounce arena is `hipHostMalloc`,
  which the ROCm runtime pins itself and which counts as GTT, not as a memlock charge.
- The container's memory **limit** is not the budget. `--max-mem` is a GTT claim; a 48 Gi
  container limit with a 64 GiB `--max-mem` works because the two account for different
  things (and because the weights are streamed, not held in the container's RSS).

## Friction found in this pass

| finding | status |
|---|---|
| `--max-mem` over budget fails as `rivoli_vmm_alloc(N) failed (-2)` with no mention of a co-tenant | **open, reported** — the message names neither the budget nor the likely cause |
| `enable_thinking: false` is advisory on Inkling 2 (byte-identical output at effort 0 and 0.9) | **open** — a property of the checkpoint's template, not of serve; document, do not "fix" |
| SSE frames carry no `usage`, so llama-swap logs `[WARN] metrics: no valid JSON data found in stream, path=/v1/chat/completions, recording minimal metrics` | **open, harmless** — non-stream requests carry full `usage`; the warning costs the UI's per-token metrics on streaming calls |
| `llama-swap -validate` accepts unknown keys | upstream behaviour; a typo is silent |
| `--mode` warns loudly on every Inkling start (both artifacts store experts natively, so the flag is a no-op) | expected; the warning is the flag refusing to pretend |

## Reproducing

The two scripts that produced the tables above are
`/var/cache/rivoli/scratch/llama-swap-exp/run-exp.sh` (full lifecycle) and `run-exp2.sh`
(`/health` shape, model-id echo, thinking arms, explicit SIGTERM). Both are written to be run
as `flock /var/run/sys-gpu.lock -c '<script>'`, both trap EXIT to kill llama-swap and any
surviving rivoli child, and both print GTT and `/dev/kfd` holders before and after. A rerun is
only comparable with the artifact in page cache or not — the cold-start numbers above (4.9 s
and 10.0 s) are page-cache-hot, and a cold pin is ~1 min.
