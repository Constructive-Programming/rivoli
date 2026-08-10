---
scope: engine
status: live
verdict: llama-swap's pod shares /var/run/sys-gpu.lock — but only for SOME models (CORRECTED 2026-08-07, verified live 3x - whisper and the embedding model hold GTT with no lock). Until hr-fleet wraps every cmd, a GPU witness must sample mem_info_gtt_used, not just the flock and KFD. PROBING (2026-08-10): `fuser` is BLIND to an advisory flock holder in another namespace — it prints nothing while the lock is held — so use `flock -n ... -c true` (rc 1 = held), and check GTT separately, since a tenant can hold either without the other.
---

# GPU lock — coordinating with llama-swap on rh-anine

k3s (and with it, `llama-swap`'s Vulkan pod) was stopped on rh-anine from 2026-07-24 to
2026-08-01 specifically to give bare-metal rivoli runs sole tenancy of gfx1151. k3s is back
now, which reopens the hazard `flock /var/run/sys-gpu.lock -c '…'` has always guarded against
between concurrent bare-metal runs, but not against a `llama-swap`-managed `llama-server`/
`whisper-server` child starting at the same time — until now `llama-swap` had no idea that
lock file existed.

> **CORRECTED 2026-08-07 — the fix below is PARTIAL, verified live three times in one
> evening.** The `gpu-lock-wait.sh` wrapper covers only SOME models: `qwen3.6-medium` held a
> proper shared flock while resident (observed in `/proc/locks`), but `whisper-large-v3-turbo`
> and `qwen3-embedding-4b` loaded and held 1.6–4 GB of GTT with **no lock at all** — twice
> while a bare-metal exclusive flock was actively HELD. Consequences until hr-fleet's
> `fleet/ai/llama-swap.yaml` wraps every model's `cmd`: (1) the flock is NOT sufficient
> protection against llama-swap tenants; (2) `/sys/class/kfd/kfd/proc/` cannot see them
> either — they are Vulkan allocations via render nodes, so a GPU witness must also sample
> `/sys/class/drm/card0/device/mem_info_gtt_used` (the engine's own sole-tenant guard reads
> the same counter and refuses above 1 GiB, which is how all three intrusions were caught);
> (3) `curl http://<llama-swap>/running` names the tenant and `curl .../unload` clears it,
> reversibly — models reload on demand. Methodology recorded in
> `docs/measurement/benchmarks.md` ("GLM-5.2 vs DeepSeek-V4-Flash", 2026-08-07) and the
> m1a-stale-selection round.

**This is now fixed on the `llama-swap` side, with zero changes needed here.** `hr-fleet`'s
`fleet/ai/llama-swap.yaml` wraps every model's `cmd` in a script that takes a **shared** flock
on `/var/run/sys-gpu.lock` — the exact same host path already bind-mounted into the pod — before
starting the real upstream binary, and **holds it for as long as that model stays loaded**, not
just at start. Any command already following the existing convention
(`flock /var/run/sys-gpu.lock -c '…'`, exclusive by default) now correctly blocks until
`llama-swap` has fully unloaded whatever it has resident, and vice versa: `llama-swap` starting
a new child waits out an exclusive bare-metal holder.

This concerns every bare-metal use of the GPU that happens outside `llama-swap`'s own process
tree — which, even with rivoli registered as a `llama-swap` model (below), is still real and
ongoing:

- **Interactive dev/measurement** — a human wrapping a one-off `cargo test`/`-bench`/probe
  command in `flock` (`docs/measurement/`, `TOUR.md`).
- **CI on the self-hosted runner** — `.github/workflows/release.yml` runs
  `flock /var/run/sys-gpu.lock cargo test --release --features rocm` on every release tag, on the
  same box (`runs-on: [self-hosted, linux, rocm, gfx1151]`). That job already used this exact
  lock before `llama-swap` knew about it; nothing changed there.

`rivoli serve` registered as a `llama-swap` model entry (`glm-5.2-rivoli`, see
`docs/reference/serving.md`) gets **two separate, non-overlapping protections**: sole-tenancy
against `llama-swap`'s *other* models (`qwen3.6`, `gemma4`, etc.) comes from `llama-swap`'s own
group/swap matrix — no lock file involved there. Protection against a concurrent bare-metal
dev/CI run comes from this SAME `/var/run/sys-gpu.lock`, same as every other model — the matrix
has no way to know about a process outside `llama-swap`'s own tree, so `glm-5.2-rivoli` takes
the lock too.

## The contract, as it now actually stands

- Host path: `/var/run/sys-gpu.lock` (i.e. `/run/sys-gpu.lock` — `/var/run` is a symlink).
  **Renamed 2026-08-02** from the original `/tmp/rivoli-gpu.lock` (every reference in this repo
  — TOUR.md, docs/measurement/, the probes, tests/mtp-neutrality.sh, release.yml — was updated
  in the same pass). The lock semantics and usage are otherwise identical: keep wrapping GPU
  commands in `flock /var/run/sys-gpu.lock -c '…'`, exclusive for the command's duration.
- **`/run` behaves differently from `/tmp` and this bit, so read it once.** `/tmp` is
  world-writable (sticky bit), so any user's `flock` could silently create the file on first
  use. `/run` is tmpfs but `root:root 0755` — a non-root `flock` command **cannot** create it if
  missing, it can only open an already-existing one. Both `/tmp` and `/run` are wiped on every
  reboot, so this matters: a `systemd-tmpfiles.d` rule
  (`/etc/tmpfiles.d/sys-gpu-lock.conf` on rh-anine, `f /run/sys-gpu.lock 0644 rhansen rhansen -`)
  now recreates it on every boot, independent of which consumer (k3s pod, bare-metal, CI runner)
  happens to start first. Without that rule, a bare-metal `flock` run as `rhansen` right after a
  cold boot — before `llama-swap`'s pod has come up and hostPath-`FileOrCreate`'d the file —
  would fail with a permission error instead of just working.
- **`llama-swap` side (hr-fleet): new, and it HOLDS the lock, not just gates the start.**
  `fleet/ai/llama-swap.yaml`'s `gpu-lock-wait.sh` bind-mounts the same path (`hostPath` volume
  `gpu-lock`), opens it on a dedicated fd, and `flock --shared --wait 350`s that fd — then
  `exec`s into the real model command. Bash fds opened this way are not close-on-exec, so the
  lock survives into the real server and stays held for as long as that process runs (every
  model here `exec`s again at least once more internally, so it propagates all the way to the
  actual binary). The 350s figure only bounds the ACQUIRE wait — how long a NEW model load
  tolerates a bare-metal exclusive holder before failing loudly instead of silently eating
  `healthCheckTimeout`. Once acquired there is no hold timeout: release is tied to the process
  actually exiting — TTL, `/unload`, config reload, OOM-kill, anything — which is exactly when a
  waiting bare-metal `flock` should be let through, and the kernel does it automatically.

## What this does NOT do

- Does not make `llama-swap` proactively evict a resident model just because a bare-metal run
  wants the GPU — a bare-metal `flock` now correctly *waits* for it (verified: an exclusive
  attempt blocks the whole time a held shared lock's process is alive, and unblocks the instant
  it exits), but nothing forces that release early. If it's not going to unload on its own soon
  enough, hit `llama-swap`'s `/unload` or scale the deployment down.
- **Does not cover three specific small models, deliberately.** `qwen3-embedding-4b`,
  `whisper-large-v3-turbo`, and `whisper-translate` are all `ttl: -1` (pinned, near-permanently
  resident) on the hr-fleet side — small enough (~5 GiB, ~3 GiB, ~3 GiB) that making bare-metal
  work queue behind them indefinitely wasn't worth it, so they're exempt from
  `gpu-lock-wait.sh`. A bare-metal `flock` can proceed while any/all of them are loaded — they
  are genuinely, simultaneously on the GPU at that point, not evicted or paused. Every other
  model here (including `glm-5.2-rivoli`) still takes the lock and holds it for its whole run.

## Where the other half lives

`fleet/ai/llama-swap.yaml` in `hr-fleet`: ConfigMap key `gpu-lock-wait.sh`, volume `gpu-lock`
(hostPath `/var/run/sys-gpu.lock`, `FileOrCreate`), mounted into every model's `cmd`. The
`systemd-tmpfiles.d` rule that survives a reboot lives directly on rh-anine at
`/etc/tmpfiles.d/sys-gpu-lock.conf` — not tracked in any repo, since it's host config, not
application code.

## Probing the lock: `fuser` is blind to it, `flock -n` is not

Added 2026-08-10, after a device suite was misread as failing for ~40 minutes.

**`fuser -v /var/run/sys-gpu.lock` prints nothing while the lock is held.** It reports
processes with the file *open*; an advisory `flock(2)` holder in another mount namespace — which
is what llama-swap's pod is — does not show up. So an empty `fuser` is not evidence the lock is
free, and the tempting conclusion ("nobody holds it, so my failure is real") is wrong.

The only reliable probe is a non-blocking acquire:

```bash
flock -n /var/run/sys-gpu.lock -c true   # rc 0 = free, rc 1 = held
```

Pair it with the GTT counter, because the two failure modes are independent: a tenant can hold
GTT *without* the lock (the whole point of this file's verdict), and can hold the lock while
its GTT is still small (a model mid-load). Both must be clear before a device number counts.

Note the related trap recorded in `investigations/int4-moe-unroll.md`: `flock -w N` exits **1
silently** on timeout, which looks identical to a test failure with an empty log.
