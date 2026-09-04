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
| 9 | CodeScene 10/10 (`RIVOLI_CS_REQUIRED=1`) | ~~10 m~~ **a burn-down, not a run** — see §2c | ARMED 2026-09-04 (`cs` on the shared home + the PAT): the red-proof fixture passed, and **21 files scored below 10**. The row's estimate was wrong because it assumed arming was the work |

## 2b. The device halves, one-shot (added 2026-09-04)

§2's rows 5, 6 and 9 each carry a half that is a RUN and not code, and two of the three
claims that they are blocked on something are now false. This section is what to type, what
each arm actually needs, and what its green is evidence of — so the gate day does not stop to
re-derive any of it.

**Precondition, recorded because it is the current blocker on all of rows A–E.** As of
2026-09-04 **rh-anine cannot start an ssh session**: the privileged parent authenticates (it
reads `~/.ssh/authorized_keys` over NFS and echoes the key options), `ssh -N` holds the
connection open, and `exec` of `echo`/`true` returns nothing at 20 s, 30 s and 75 s — from
rh-desktop *and* from hr-main — while `sftp` hangs identically, so no session child of any
kind starts. There is no `~/.ssh/rc`, and `~/.ssh/config`'s host-group `ForwardX11 Yes` is a
separate, lesser defect that hangs the *first* attempt on xauth (use `-o ForwardX11=no`).
That is a console fix, not a remote one. It also means **these arms cannot be evidenced after
the fact from another node**: the batteries write to `/var/cache/rivoli/scratch`, which is
rh-anine's local NVMe and invisible elsewhere, so each row must record its log path as it runs.

Every row below: build OUTSIDE the lock, `flock /var/run/sys-gpu.lock`, an explicit
`CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli`, `-- --test-threads=1`, witness per arm
(`tests/gpu-witness.sh`) with a non-empty witness discarding the arm, and exit codes read
UNPIPED (§12's stale-binary class and §7e-bis's pipeline-green class are both avoided by
those two lines, not by intention).

| row | command | needs | what green means |
|---|---|---|---|
| **A** M11 fp8 device half | `flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test -p rivoli-engine --features rocm --test glimmer_fp8_decode -- --test-threads=1 --nocapture'` (verbatim from the file's own header) | **no checkpoint** — the suite writes its own artifact via `glimmer_anchor::write_artifact` into `std::env::temp_dir()` | the fp8 arm's logits DIFFER from the bf16 arm's on the same input (anti-fallback) and the split is entered; the file's header warns a finite-but-wrong fp8 arithmetic survives both, so this is not a quality claim |
| **B** live serve SSE round-trip | `tests/smoke-glm.sh <artifact-dir>` (its `serve` cell) | the GLM artifact | already-written assertions, not missing code: `tests/smoke-glm.sh:111-117` requires `^data: ` frames AND a `data: [DONE]` terminator over a live decode, after readiness and `/v1/models`. §13's "only the live SSE round-trip is OWED" is owed as a **run** |
| **C** M17c block-attend execution | `flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test -p rivoli-engine --features rocm --test kernel_glimmer_block_attend -- --test-threads=1'` | none (host oracle + drawn weights, no skip path: 18 tests, no env gate, `#![cfg(feature = "rocm")]` is the only arm) | VERIFY-OR-PAY, and it is probably already paid: `crates/engine/tests/kernel_glimmer_block_attend.rs` is M17c's on-device gate and landed 2026-08-17, and both device batteries since (2026-09-01's 592 tests / 91 suites, 2026-09-02's 699 / 98) would have run it — but neither names suites, so neither is citeable for THIS claim. Record the suite name and its count, then correct §11's "has NEVER EXECUTED", CLAUDE.md's census paragraph, and §4's label **in the same commit**. The `gqa_attend` duplication stays owed regardless and is not a device matter: `.jscpd.json` is `format: ["rust"]`, so no gate can raise it |
| **D** CodeScene 10/10 | `RIVOLI_CS_REQUIRED=1 CS_ACCESS_TOKEN=… CARGO_TARGET_DIR=… cargo test -p rivoli --test codescene` | **the token, and nothing else** — `cs` is already installed at `~/.local/bin/cs` (shared NFS home, so on every node; it runs, and reports 1.0.40 pending) | `codescene.rs` panics on tool-absent only under `RIVOLI_CS_REQUIRED`; the standing red-proof fixture must still score < 10 in the same run. Since no device is involved this row was payable on rh-desktop immediately, **and it was paid on 2026-09-04**: the fixture half is green and §13's "blocked only on `CS_ACCESS_TOKEN`" is closed — which revealed the row was never about the token. See §2c |
| **E** fp8 paired dNLL + tok/s + partition bit-identity | build the instruments outside the lock first: `cargo build --release --features teacher-forcing --bin rivoli --bin ppl`, then `flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli tests/ppl-gates.sh <artifact-dir> all'` — the script reads `$PPL_BIN`/`$PPL_TOOL` from the target dir's `release/`, and `bin/ppl` consumes the per-token NLL files the engine writes under `--ppl <text> --ppl-out <path>` | BOTH Glimmer artifacts: bf16 55,712,428,144 B and fp8 30,554,903,564 B (the NFS pair measured in `docs/measurement/glimmer-fp8.md`); confirm both by length before the arm. The `tf` cell additionally needs the pinned reference at `$PPL_REF_BIN` (default `/var/cache/users/rhansen/m10-ref-tf-target/release/rivoli`, built with teacher-forcing) | **the stated blocker is stale.** `glimmer-fp8.md` says this is "blocked on M10's `--ppl`, which has zero commits" — `crates/cli/src/bin/ppl.rs` and `tests/ppl-gates.sh` both exist and their classifier and engine halves are PAID (§5, 2026-08-21). Run it, then correct that doc's blocker clause with a dated note |

**What row A's command must not quietly do.** `temp_dir()` honors `TMPDIR`, and the default is
`/tmp` — a 63 GB tmpfs that competes with the same unified-memory budget the decode under test
is spending (CLAUDE.md, *Build cache and worktrees*). Export `TMPDIR=/var/cache/rivoli/scratch`
for arm A and say so in the recorded command line, or the arm is measuring under memory
pressure it did not declare.

**Two things found while writing this table, recorded because they are the reason to read the
scripts rather than the summaries.** `ppl-gates.sh` refuses in three distinguishable ways — a
missing `release/rivoli`, a `--no-default-features` build that "cannot decode at all", and a
build that refuses `--ppl` for want of `--features teacher-forcing` — and the third is the one
that would otherwise read as a scoring red: `--ppl` is an unconditional clap field, so the
*binary* must be built with the feature while `main.rs` carries no `#[cfg]` at all. And
`crates/cli/src/bin/ppl.rs:1` cites `docs/investigations/cache-conditional-routing.md`, which
does not exist in this tree — the citation resolver `docs.rs` deliberately left unported is
what used to catch that class, so a dangling citation is currently a legal state of this repo.

**Nothing here is a TODO in the code.** Rows A, B, C and E are arms to run and record on the
device; row D is an external credential. Two of the five claims that A/D/E were blocked on
missing tree artifacts are wrong as of today, and the file that makes each one is named in its
row so the correction lands beside the measurement rather than in a new document.

## 2c. CodeScene, armed: the gate was never blocked on the token (2026-09-04)

`RIVOLI_CS_REQUIRED=1 cargo test -p rivoli --no-default-features --test codescene` on
rh-desktop, `cs` from `~/.local/bin/cs`, PAT supplied from outside the tree: **exit 101**
after 371 s. `the_red_proof_fixture_scores_below_ten` **passed**, so the reviewer is really
scoring and not returning 10.0s — which is what makes the other half's red evidence rather
than noise: **21 files below 10/10**, with `codescene.rs`'s own standing rule that an EXEMPT
row is for **frozen transliterations of the reference, never rivoli-authored code** (its
comment records the first armed run refactoring `glimmer_draft_oracle.rs` and
`v4_indexer_goldens.rs` to 10.0 on exactly that ground). So none of these 21 is exemption
work; every one is a fix, and the score distribution says so — 16 of 21 sit in [9.0, 9.9],
which is one long function or one over-wide argument list, not a structural disagreement
with the tool.

**Why the tree did not know.** Every one of these landed after 2026-08-21, the last day the
gate could run: §13's verdict and CLAUDE.md both carry "blocked only on `CS_ACCESS_TOKEN`",
and three weeks of merges — the whole Qwen S2–S5 arm (11 of the 21), M17b's templates, the
K3 test module — went past a gate that warn-skips when unarmed. A gate that degrades to a
warning when its credential is missing is a gate that reports green; `RIVOLI_CS_REQUIRED` is
CI's protection and CI is the only place it is set, so locally the red was invisible. That is
the §12 W3 class one level up: a silent skip that lets a claim stand as checked.

**The 21, triaged by what can verify a fix on this box** (`cs review --output-format json`,
per file):

| file | score | findings | verifiable here? |
|---|---|---|---|
| `oracles/tests/qwen_anchor_taps.py` | 7.92 | Complex Method/Conditional, Deep Nested Complexity, Bumpy Road, arity | needs the anchor venv (CPU, but not on this node) |
| `oracles/tests/qwen_anchor_driver.py` | 8.56 | Complex Method/Conditional, Overall Complexity | same |
| `engine/tests/kernel_qwen_gdn_recurrent.rs` | 8.88 | Complex Method, Deep Nested Complexity, Bumpy Road | **no — device arm** |
| `artifact/src/qwen_encoding.rs` | 9.09 | Complex Conditional, Deep Nested Complexity | yes (deviceless arm) |
| `artifact/src/census/qwen.rs` | 9.24 | Complex Conditional, Complex Method | yes |
| `engine/tests/kernel_qwen_indexer.rs` | 9.29 | Large Method, arity | **no — device arm** |
| `oracles/tests/common/golden_read.rs` | 9.38 | Code Duplication | yes — and jscpd is the neighbouring smell, so both tools must stay green |
| `oracles/tests/qwen_anchor_fixtures.rs` | 9.38 | Large Assertion Blocks | yes |
| `artifact/tests/qwen_config.rs` | 9.52 | Large Method | yes |
| `cli/build.rs` | 9.54 | Large Method | yes (every build runs it) |
| `cli/src/bin/convert_qwen.rs` | 9.54 | Large Method | yes |
| `artifact/tests/qwen_names.rs` | 9.58 | Large Method | yes |
| `oracles/tests/qwen_anchor_compare.py` | 9.60 | Complex Method | needs the venv |
| `artifact/tests/glimmer_template.rs` | 9.68 | arity | yes |
| `backend/src/hip_attn.rs` | 9.68 | arity | **no — ABI wall, rocm arm** |
| `engine/src/v4/engine.rs` | 9.68 | arity | **no — device arm** |
| `engine/tests/k3/mod.rs` | 9.68 | Primitive Obsession | **no — device arm** |
| `engine/tests/kernel_qwen_harness.rs` | 9.68 | arity | **no — device arm** |
| `oracles/tests/qwen_anchor_provenance.py` | 9.68 | Complex Method | needs the venv |
| `oracles/tests/qwen_anchor.rs` | 9.68 | String Heavy Function Arguments | yes (host oracle reads the goldens) |
| `engine/tests/kernel_qwen_gdn_out_norm.rs` | 9.84 | Bumpy Road Ahead | **no — device arm** |

**Seven need the device to verify, four need the anchor venv, ten are deviceless-verifiable
on this box.** The gate needs all 21, so row 9 stays red until the last arm lands, and the
split is the decision: burn down the ten now against `cargo test --workspace
--no-default-features` (each re-score is ~20 s and the full gate ~6 min, so the loop is
cheap), or hold the whole set for one reviewed branch that can verify the device nine too.
Refactoring the nine blind — `hip_attn.rs` is the hand-written launcher side of the ABI wall
and `v4/engine.rs` sits under the parity gate — is what §11 already declined once for the
`gqa_attend` duplication, on the grounds that a change to GPU-parity-gated code that its
author cannot verify is worse than an open debt.

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
