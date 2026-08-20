---
status: live
scope: engine
verdict: The v3.0.0 cut plan (owner decisions 2026-08-20) — gates BEFORE tag; release defaults are FAST (both determinism flags opt-in, the GLM defect ships labeled with --arena-refresh quoted at 1-3%); the gate day is §2's ordered runbook (parity, both smokes, ppl-gates with the M10 engine halves, fp8 decode + SSE, determinism-512 on the mitigation arm, K3 first real decode, CodeScene once CS_ACCESS_TOKEN lands); the baseline is re-taken with the pinned prompt before notes are written; §4 is what v3.0.0 explicitly does NOT claim. Tag and push only after every §2 row is green or its red is fixed forward.
---

# Cutting v3.0.0

Owner decisions, 2026-08-20: **(1)** release defaults are fast — `--arena-refresh` and
`--copy-via-cpu` both opt-in; every recorded number carries its flag state and the release
notes carry the nondeterminism warning. **(2)** Gates run BEFORE the tag; a red is fixed
forward, then the day re-runs from the failed row. **(3)** Scope additions: the K3 first
real decode joins the gate day; the CodeScene 10/10 gate joins once `CS_ACCESS_TOKEN` is
in the environment (it is the one external dependency).

Versioning: `v2` = the old tree's final state (tagged 2026-08-20). This cut is `v3.0.0`
on `main`.

## 1. What ships

The four-arm engine as merged 2026-08-20: GLM-5.2 (int3-vq/int4, streamed experts),
Muse Glimmer-30B (bf16/fp8-e4m3, own chat template since M11b), DeepSeek-V4-Flash (fp4),
Kimi-K3 (MXFP4, tiktoken loader) — one seam, `serve` with SSE, the phase profile,
teacher-forced scoring, the divergence localiser behind `corruption-probe`.

## 2. The gate day — one flocked sole-tenant session, in this order

Discipline for every row: `flock /var/run/sys-gpu.lock`, contention witness per arm
(`tests/gpu-witness.sh`), exit codes read UNPIPED, build OUTSIDE the lock, dev profile
unless the row is a timing row. A row's green is recorded with its command line.

| # | gate | est | notes |
|---|---|---|---|
| 1 | `tests/feature-matrix.sh` | ~10 m, no GPU | run first, deviceless |
| 2 | `tests/parity-glm.sh` | ~1 h | reference binary pinned at `archive/glimmer-s2` @ 6b7f496e, prebuilt in /var/cache/users/rhansen/ref-pin-target |
| 3 | `tests/smoke-glm.sh` | ~45 m | refusals asserted against the table's own fragments |
| 4 | `tests/smoke-v4.sh` | ~30 m | |
| 5 | `tests/ppl-gates.sh` — the three cells INCLUDING the M10 engine halves | ~1 h | the owed halves need a source mutation each; recipes are in gate-red-proofs §5 |
| 6 | `crates/engine/tests/glimmer_fp8_decode.rs` (anti-fallback assert) + live serve SSE round-trip | ~20 m | the two owed M11 device halves |
| 7 | `tests/determinism-glm.sh <artifact> 512` on the `--arena-refresh` arm, same-day stock control | ~1 h | a green is only interpretable WITH the control (gate's own rule); recorded as the mitigation arm's green, not the default's |
| 8 | K3 first real decode — correctness only | hours (NFS) | ids finite, sane text, no crash, ctx ≤ 8192 (`ATTEND_MAX_KV`), small token count. Perf disclaimed (owner Q1: artifact is NFS-resident). A starved-looking job may be alive — verify by /proc PID before restarting. §8 of k3-first-checkpoint.md lists two checkpoint leads (A_log shape, lying MXFP4 target list) to check DURING this run's load |
| 9 | CodeScene 10/10 (`RIVOLI_CS_REQUIRED=1`) | ~10 m, no GPU | WAITS ON `CS_ACCESS_TOKEN`; the standing red-proof fixture must still score <10 |

## 3. After the gates, before the tag

1. **Re-take the baseline** — all four arms, release profile, the ppl-gates-pinned prompt
   (the 2026-08-16 baseline is not byte-reproducible and its Glimmer row is superseded by
   M11b). Record flag state per row; this page becomes the release's numbers.
2. **Release notes** (`docs/` doc with front matter + INDEX row): the §1 inventory, the
   baseline table, and the §4 labels verbatim.
3. Tag `v3.0.0` (annotated), push `main` + tag.

## 4. What v3.0.0 explicitly does NOT claim (the labels)

- **GLM long-run determinism.** Open defect, root cause unnamed. Default decode can
  diverge run-to-run (~1 event per 299–578 token-forwards, per READ). Mitigation:
  `--arena-refresh` (1–3%, gated over thousands of tokens). Fix candidate:
  `--copy-via-cpu` (~11% quiet / ~4% loaded, exposure still accumulating). The closeout's
  four closure items stand.
- **V4 parity is 30/32 forced-history, bsz=1**, both flips at near-ties.
- **K3 decodes; nothing more.** No chat framing (bench/raw only — `--port` refuses),
  no benchmark (NFS-resident artifact), anchor tolerances carry one-draw floors,
  `moe_fixed` range unmeasured for K3.
- **The M17c block-attend kernel has never executed** — compiled for gfx1151, census-
  covered, not wired into any decode path; its duplication vs `gqa_attend` is OWED.
- **CI has no GPU arm.** Every device gate above is exactly as fresh as its recorded run.
- Deferred wholesale: `wave/m12-glm-chain` (M13 DSA), `wave/m19-k3` (K3 chain), the M1
  substrate deferrals, K3 both-draw tolerance refresh.

## 5. Standing risks for the day

The GPU is shared: the flock is advisory, so every arm carries the witness and a
non-empty witness discards the arm. `/home` is NFS — build into /var/cache. A 115 GiB pin
starves concurrent CPU jobs; liveness by /proc, never by ps/log-tail. tmpfs (/tmp) competes
with the GPU memory budget — no artifacts there.
