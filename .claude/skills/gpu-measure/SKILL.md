---
name: gpu-measure
description: >-
  The per-arm GPU measurement procedure on rh-anine — target-dir isolation, the flock,
  --test-threads=1, the per-arm contention witness, and the READY-FOR-GPU / GO / DONE lease
  with the coordinator. Invoke BEFORE any run that touches the device or produces a citable
  number: a benchmark, an A/B, tests/parity-glm.sh, tests/ppl-gates.sh, tests/smoke-glm.sh, or
  a bare `cargo test --workspace` (which is a GPU arm, because default = ["rocm"]).
---

# Measuring on the shared GPU

The device is on **rh-anine** (`ssh -o BatchMode=yes rh-anine`); this session runs on
rh-desktop and `/home` is NFS. The GPU is sole-tenant, the lock is advisory, and other
agents skip it — so a number is only citable with its **witness verdict** attached.

## The lease dance — you do not take the GPU, you are handed it

1. **Do every deviceless thing first.** Build, `cargo clippy`, and
   `cargo test --workspace --no-default-features` need no lock and no lease. Finish them
   so the leased window holds nothing but the measurement.
2. **Post READY-FOR-GPU** to the coordinator: the exact commands you will run, in order,
   with their target dirs and flags already written out, plus your **wall-clock estimate**
   (parity ~1 h, ppl-gates ~28 min — ~34 if the strict branch takes its fourth arm, smoke
   ~45 min, + ~6 min per red proof).
3. **STOP.** Do not run anything on the device. Wait for the coordinator's explicit **GO**.
4. Run the arms exactly as posted. Deviating from the posted commands voids the lease —
   re-post instead.
5. **Report DONE** with per-arm results *and* per-arm witness verdicts.

## Per-arm procedure

```bash
# 1. BUILD OUTSIDE THE LOCK — every cargo invocation carries its own target dir.
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --tests
# <checkout> is the worktree's name; it is `rivoli` for the primary checkout.

# 2. TAKE THE LOCK, then run the arm.
flock /var/run/sys-gpu.lock -c \
  'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace -- --test-threads=1'
rc=$?    # captured from the command itself, UNPIPED
```

**The explicit `CARGO_TARGET_DIR` is mandatory on every invocation.**
`/etc/security/pam_env.conf:6` exports
`CARGO_TARGET_DIR=/var/cache/users/<user>/cargo-target` on every node, and cargo's
precedence (`--target-dir` > `CARGO_TARGET_DIR` > `build.target-dir`) means that login
variable beats a worktree's `.cargo/config.toml` in every shell an agent gets. A
worktree-local config is therefore **inert** here; relying on it is how W3 returned two
greens while certifying an isolation that had never happened
(`docs/measurement/gate-red-proofs.md` §12). `env -u CARGO_TARGET_DIR cargo …` is the
other working form — it strips the variable so the config can bind. A shared target dir
is a **correctness** defect, not a slow build (§7e-bis).

## The contention witness — per arm, not per session

The on-demand gates already do this via `tests/gpu-witness.sh` (`run_arm`, sourced with
`LOCK` and `SCRATCH` set); a hand-rolled arm owes the same two samples:

- **`/dev/kfd` holders**, sampled every 5 s while the arm runs, identified by *descent
  from the arm's own pid* — never by binary path, since a peer agent may run the identical
  binary and a path whitelist would wave it through.
- **`mem_info_gtt_used`** (`/sys/class/drm/card*/device/mem_info_gtt_used`) as a pre-arm
  baseline, **because KFD is blind to Vulkan tenants** — llama-swap once held 1.6 GB of GTT
  with zero KFD entries. An empty sysfs read is a *setup refusal*, not a clean box.

**A non-empty witness is exit 3: DISCARD the arm.** Do not average it in, do not caveat
it, do not silently rerun — report the discard to the coordinator and let the coordinator
decide whether to spend another lease on a rerun.

## Standing rules

- **`-- --test-threads=1` on anything that touches the device.** Parallel device tests
  build parallel io_uring rings and wedge. `cargo test --lib` counts as a GPU arm.
- **Never `cargo build` between the two arms of a benchmark.** Measured once: one build's
  page-cache eviction moved ms/miss **1.36 → 5.14**. Build all arms before the first one.
- **Dev profile unless you are timing.** `--release` is for benchmarks and performance
  evaluation only; it compiles out every `debug_assert!`.
- **Lock per arm of an A/B**, and build outside every one of them.
- **Read exit codes UNPIPED.** `cmd | grep | head` reports the pipeline's last stage; the
  false green that followed one was an unconditional `echo "CHECK OK"`. Redirect to a
  file, capture `$?` from the command, then read the file.
- **Scratch goes to `/var/cache/rivoli/scratch`**, never `/tmp` (63 GB tmpfs in RAM that
  competes with the unified-memory budget, and session-scoped).
- **Rank quality on paired dNLL from `bin/ppl`, never the PPL column**, and run the A-vs-A
  control first: the floor is per arm and a comparison below it is not a measurement.
- **Prefer measured spans to derived percentages.** A bracket timer that contains what it
  is used to rule out is a defect class.

## Reporting a number

Every number ships with the arm that produced it and that arm's witness verdict:

```
arm A  --max-mem 60  2.58 tok/s   witness: CLEAN (0 foreign KFD, GTT baseline 17.8 MiB)
arm B  --max-mem 30  1.84 tok/s   witness: DISCARDED (pid 41221 convert_k3, loadavg 7.31)
```

**An unwitnessed number is unciteable** — including one you are tempted to quote from an
earlier session. If the witness verdict is missing, the measurement has not happened yet.
