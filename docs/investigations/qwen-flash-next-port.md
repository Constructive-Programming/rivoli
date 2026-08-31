---
status: live
scope: qwen
verdict: NOTHING IS BUILT YET — the Qwen3.8-Flash-Next (qwen4_exp) port has a finalized plan and no code, and this is its in-tree plan of record: the transcribed model identity (125B MoE 6B-active + 51B n-gram table + 4B MTP, 48 layers, 12x(3x GatedDeltaNet+MoE -> 1x QSA+MoE), fp8 block-128 official checkpoint, thinking default, vision excluded from v1 — transcribed at an UNPINNED revision, so census item 1 still owes the pin), the census checklist, the A-E track split with exclusive file lists, the GPU lease protocol, and the exit-gate table whose tests/smoke-qwen.sh row lands WITH the port because smoke-k3.sh proved later means never. Part 1 platform work is in flight; anchors, tolerances and defect matrices are all OWED, so nothing here is measured evidence, and every figure quoted from another scope carries that scope inline.
---

# Qwen3.8-Flash-Next (`qwen4_exp`): the port, in tree

## Why this file exists at all

The plan is executable prose that lives outside the tree, and an out-of-tree plan is not
citable — `k3-first-checkpoint.md:10` had to say "that plan is not in this tree, so its
milestone numbers are not cited", which is the cost paid once already. This doc is the
in-tree, condensed plan of record: what the port is, what closes each item, and what is
still owed. It is **not** a measurement doc. Nothing below is evidence.

Out-of-tree provenance (read for the full argument, cite THIS file in code and commits):

- `/home/rhansen/.claude/plans/rivoli-part2-qwen4-exp-port.md` — the port plan (S0–S8).
- `/home/rhansen/.claude/plans/rivoli-part1-rocm10-platform-toolchain.md` — the platform
  work the fleet inherits (per-worktree build isolation, agent defs, this registry growth
  as its commit E, the GPU lease, MCP/skills, the CI trigger fix). On the toolchain: the
  fleet builds against whatever `docs/measurement/rocm10-migration.md`'s verdict closes
  on, and that verdict is **IN FLIGHT at this commit** — so no toolchain number transfers
  here until its ROCm 10 column is measured.

## Stage legend (S0–S8, cited throughout this file)

- **S0** vendor pins — reference bytes vendored, gates recompute FNV-1a from the live file.
- **S1** reddening commit — the fifth `Arch` refuses everywhere, fragments asserted.
- **S2** anchors + tolerances — goldens and measured floors BEFORE the code they score.
- **S3** kernels — launchers scored by S2's oracles, census checked both ends.
- **S4** artifact + converter — `qwen_config.rs`, `convert_qwen`, exact tensor counts.
- **S5** chat encoding — the hand-ported template against id-pinned reference cases.
- **S6** the arm — `engine/src/qwen/**`; legality flips atomic; `tests/smoke-qwen.sh` lands here.
- **S7** first decode + gate battery — parity window, ppl cells, floor; reference ids recorded.
- **S8** closeout — every deferral retired or milestone-named, standing gates green.

## Where the port stands (2026-08-31)

Nothing built. `Arch` still has four variants; no `qwen` config type, converter, arm,
kernel, anchor or smoke exists. What exists is: the plan, and — as of this commit — the
docs scope `qwen`, this file, and its INDEX row.

## Model identity as transcribed (revision NOT pinned)

Released **2026-08-26** as the open-weight preview of the Qwen4 architecture. Every field
below is **transcribed** from the HF model card + `config.json` **as read on 2026-08-30**
— the **revision is not pinned anywhere**, and pinning it plus vendoring the bytes for a
byte-compare IS census item 1 (S0). Until that lands, every figure below is *unverified
transcription, not evidence*, and this file's own cross-track figure rule (below) applies
to it like any other unwitnessed number. Tertiary write-ups contradicted each other (one
claimed 60B active) and are not used. The tech report PDF is **unread** — every math
detail marked **(TR)** is OWED and pinned from it before the code it governs is written.

- `model_type: qwen4_exp`, `architectures: [Qwen4ExpForConditionalGeneration]`.
- ~180B on disk: **125B MoE (6B active/token) + 51B hashed n-gram table + 4B MTP**.
  48 layers, hidden 2560, vocab 248,320 padded, RoPE theta 1e7, native ctx 262,144.
- Layer pattern **12 × (3 × (GatedDeltaNet → MoE) → 1 × (QSA → MoE))**; `layer_types`
  linear/full with `full_attention_interval: 4`.
- GDN: conv kernel 4, **16 QK heads @128 / 48 V heads @128** (asymmetric). fla lineage,
  not KDA (TR: decay/gate form).
- QSA: 24 Q / 2 KV heads @256, partial RoPE dim 64; lightning-indexer-style MQA indexer
  4 heads @128 + 1 shared key head, `indexer_budget: 2048`, compress ratio 4 (TR: scoring).
- MoE **every** layer: 512 experts, top-10 + 1 shared, moe_intermediate 640 (TR: router).
- Hyper-connections: `hc_count: 4`, `hc_lowrank: 320` — a 4-wide residual stream through
  all 48 layers, so every sublayer is wrapped pre/post (TR: static vs dynamic form).
- Hashed n-gram embeddings: 20M entries (~51B), bigram/trigram, injected at layer 2 via
  `ple_layer_ids` (TR: hash fn). Pure per-token row gather — an ordered-unit stream, P6.
- Source artifact: the **official FP8 checkpoint**, fine-grained **block-128** e4m3 (the
  scheme `Fp8W` already ingests); tensor dtypes BF16 + F8_E4M3 + I64.
- **Thinking mode is the default.** Sampling differs per mode (thinking T=1.0/top_p .95/
  top_k 20; non-thinking T=0.7/top_p .80/presence 1.5) — from `generation_config.json`,
  vendored under item 1.
- Owner scope decisions 2026-08-30: **text-only v1** (vision named as later milestones),
  FP8 official checkpoint as the source.

**Box constraint that shapes every gate:** ~180B parameters total (125B MoE + 51B n-gram
table + 4B MTP, per the unvendored config/card transcription above) at 1 byte/param in the
released fp8 is **≈ 180 GB against this box's ~124 GB usable** — so no released precision
fits resident on gfx1151, and the first-party reference cannot run here either. Anchors
are therefore operator-level taps plus a frozen teacher-forced window, never a live
side-by-side decode.

## Census checklist (the eight items; each closes with an artifact, not a memory)

| # | item | closes when | state |
|---|---|---|---|
| 1 | vendored config pins | `qwen-reference/config.json` **and `generation_config.json`** (the sampling-defaults source) vendored at a pinned revision, and the gate **recomputes** FNV-1a from the live file and byte-compares | OWED (S0) |
| 2 | tokenizer TYPE first | HF `tokenizer.json` confirmed as the TYPE **before** 180 GB moves, **and** the chat template's real home located (it has shipped only in the fp8 SOURCE repo before), **and** its hand-port scheduled with id-pinned cases — the GLM scar is a hand-ported template that drifted to another family's framing for months | OWED (S0) |
| 3 | converter `ensure!` counts | `convert_qwen` asserts exact consumed/emitted tensor counts, with MTP and vision excluded **by name** against `tensor-families.tsv` | OWED (S4) |
| 4 | first-party anchors + defect matrix | tiny-width real-structure anchor from the model's own stack, two weight salts, **≥2 rows per each of the nine named operator classes (so ≥18 rows)** each shown to redden AND to hold, tolerances from fp64/fp32 floors on ≥2 draws BEFORE the kernels | OWED (S2) |
| 5 | kernel census both ends | every new launcher has an oracle suite or a live DEFERRED row; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | OWED (S3) |
| 6 | registration in the SAME change | `SCOPES` entry + this doc + exactly one INDEX row | **DONE, this commit** |
| 7 | thin end-to-end smoke | `tests/smoke-qwen.sh` lands WITH the exit gate | OWED (S6) |
| 8 | behaviour+ABI naming | the rule binds **kernels, traits and structs** — none of those may be named `qwen_*`; membership is data in the census table. Arm directories and arch-scoped module/file names are the argued exception (`crates/engine/src/qwen/`, `qwen_config.rs`, `convert_qwen.rs`, `qwen_encoding.rs`): those name a *port boundary*, not a behaviour, and the exception is stated here so no reviewer has to guess | binding from S1 |

## Identity decisions already fixed

`Arch::QwenFlashNext` · kebab/docs scope `qwen` · manifest spellings
`"Qwen4ExpForConditionalGeneration"` and `"qwen4_exp"` at top level (a nested
`text_config` spelling is asserted by config validate, never accepted by the sniff) ·
`crates/artifact/src/qwen_config.rs` · arm `crates/engine/src/qwen/` · golden magic
`RIVQWGLD` ·
`docs/measurement/qwen-reference/{anchor.md,config.json,generation_config.json,tensor-families.tsv}`
· math in `docs/reference/qwen-architecture.md`. Milestone label MQ, stages S0–S8;
M-numbers are owner-assigned.

## Track split and exclusive files

Agents share one tree, so the file lists are exclusive — a second agent editing a listed
file is the defect, not a merge conflict.

```
S0 census (B: config pins · A: tensor-families.tsv)
 │
 ▼
S1 reddening commit (coordinator)
 │
 ├──► B anchors (S2) ──► C kernels (S3) ──┐
 ├──► A artifact (S4) ────────────────────┤
 └──► D chat encoding (S5) ───────────────┴──► E arm (S6) ──► S7 GPU ──► S8 closeout
```

The **B → C** edge is the vendoring dependency spelled out under the table. Every S0
deliverable has an owner — the config and `generation_config.json` pins are B's, the
tensor census tsv is A's — so nothing in the census node is unowned.

| track | exclusive files |
|---|---|
| Coordinator | `legality.rs` + tests, `arch.rs`, `main.rs`, `serve/mod.rs`, `docs.rs`, `golden.rs`, `seam.rs`, `engine/lib.rs`, `INDEX.md`, the how-to-measure floor row; hands the seam, `main.rs`, serve and the legality flip to E after A/C/D merge |
| A artifact | `qwen_config.rs` + tests, `bin/convert_qwen.rs`, `tests/qwen_convert.rs`, `tensor-families.tsv` |
| B anchors | `qwen_anchor*.{rs,py}`, `tests/qwen-anchor.sh`, `qwen-anchor-*.bin`, `common/tolerance.rs`, `qwen-reference/{anchor.md,config.json,generation_config.json}`, `qwen-architecture.md` |
| C kernels | `kernels/*.hip` (new sections), `hip_attn.rs`, `hip_blocks.rs`, `hip_linalg.rs`, `engine/tests/kernel_qwen_*.rs`, `kernel_coverage.rs` |
| D chat | `qwen_encoding.rs`, `tests/qwen_template.rs`, `qwen-chat-cases.json`, `qwen_template_driver.py` |
| E arm | `engine/src/qwen/**`, `tests/smoke-qwen.sh`, plus the coordinator-handed seam files |

C authors deviceless against B's **vendored** fixtures, so C starts when the goldens are
vendored — not when B's doc is finished.

## GPU lease protocol (no daemon, no lease file, no env token)

The lease IS coordinator-serialized dispatch. A self-asserted env-var token is against
house rules and was rejected.

1. Worker finishes its deviceless work, posts **READY-FOR-GPU** with the exact command
   list and expected wall-clock, and **STOPS**. A worker that "waits" never resumes —
   pollers do not fire.
2. Coordinator grants **GO** to exactly one worker, after checking occupancy via
   `/sys/class/kfd/kfd/proc/` — **never `pgrep`**, and never `ps | grep flock`, which has
   returned empty while a peer was demonstrably blocked. KFD is also blind to Vulkan
   tenants, so `mem_info_gtt_used` is read too.
3. Worker runs each arm `flock /var/run/sys-gpu.lock` + `--test-threads=1`, builds
   OUTSIDE the lock, `setsid` for long runs, and samples a **contention witness per arm**.
4. Worker reports **DONE** plus a per-arm witness verdict. A non-empty witness **discards
   that arm**; the coordinator decides the rerun.

Flock is the cross-tenant guard, the witness is the post-hoc detector, the coordinator is
the fleet-vs-fleet guard. Three layers, none of them optional. What runs **unlocked** is
any `--no-default-features` build, test or clippy — with its explicit
`CARGO_TARGET_DIR=…` prefix, since a shared target dir has already certified a green that
scanned a sibling's tree — which is how four of the five tracks author. The deviceless
docs test is the docs-registrar's ONLY invocation. Everything that touches the device
waits for a **GO**.

## Exit gates

| gate | what it is | lands with | red proof |
|---|---|---|---|
| docs registry | this doc + `SCOPES` + one INDEX row, deviceless | **this commit** | the two natural mid-change reds, recorded in the worklog below |
| S1 reddening commit | every census and legality surface refuses the fifth arch with **asserted message fragments** (`QWEN_ARM_NOT_BUILT` and its siblings), so no track can land silently | S1, before any track opens | the refusals themselves — each one observed firing, quoting its own fragment; a refusal that cannot be seen to fire is not a refusal |
| anchors | tiny-width real-structure anchor from the first-party stack, all 48 layers, real `layer_types`/`ple_layer_ids`/hc=4/conv=4, **two weight salts** with disjointness asserted, vendored `RIVQWGLD` bytes read deviceless | S2, BEFORE any kernel | truncated golden reds every load; a mutated structural assert reds exactly that test |
| defect matrix | **≥2 rows for each of the nine named classes, so ≥18 rows** (GDN decay form, conv tap order, output-gate activation, indexer relu/pool/budget, hc transpose + lowrank order, router bias/norm/w1w3, n-gram seed/order/layer, partial-rope width, eps homes), gated **both directions** | S2, with the anchor | a deleted EXPECT_GREEN row reds the both-directions gate |
| tolerances | per-operator rows from fp32/fp64 floors on **≥2 weight draws**, under the existing derived-policy gate; `ExactOnly` where the margin collapses | S2, before the kernels they score | an under-floor tolerance reds the tolerance-rule gate |
| kernel census | every new launcher has an oracle suite or a live DEFERRED row, checked both ends; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | S3 | the three-edge DEFERRED proof for any deferral; per-suite plants from the observed defect classes |
| chat encoding (track D) | ~31 **id-pinned** template cases scored against the reference tokenizer's own output (gate G5): thinking and non-thinking framing, tool turns, multi-turn, the trailing-generation prompt | S5, with the encoder | a deliberately mis-framed turn (another family's role framing, the GLM drift) reddens the case set |
| parity window | frozen teacher-forced forward over a pinned ~64-token window (the M8 substitution — no live reference fits this box): per-position argmax agreement with flips confined to measured near-ties, NLL deltas within compounded tolerance and above the floor | S7 | a shadow artifact perturbed at scale — a single-byte flip is BELOW a short run's detection floor (scope: glm) |
| ppl cells | `tests/ppl-gates.sh <artifact> profile` and `p4` (both arch-agnostic); the paired `tf` cell's analogue here is the parity window. **p4 self-calibrates: on a non-reproducing arm it reports UNCALIBRATED (exit 1)** — there it is a diagnostic, never a merge gate | S7 | p4's red-proof corpus |
| determinism floor | A-vs-A twice on `tests/ppl-corpus.txt`, **before any dNLL claim** | S7, cell 3 — a precondition, not a result | the standing instrument's own self-test, `tests/determinism-glm.sh --self-test`: the comparator reddens on a changed id AND on a truncated stream. The qwen arm inherits that instrument; this floor **row** stays uncalibrated until it is measured on the qwen artifact |
| `tests/smoke-qwen.sh` | the thin CLI door-to-door: `--mtp` refusal quoted from the table's own fragment, decode cell against recorded ids, `--attn` inertness cell, a prompt crossing the layer-2 n-gram injection decoding finite | **the script lands WITH the arm at S6 — not later** (`tests/smoke-k3.sh` was scheduled for "after the port" and never existed: the named later-means-never scar). Its legality cells are **green immediately**; its decode cell is **RED until S7 records the reference ids**, and that red is the TDD ordering, not a defect | a wrong message fragment must redden it; **a missing reference is RED, never a skip** — which is exactly what makes the S6→S7 red meaningful |
| closeout | docs registry + jscpd + CodeScene + line caps green; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | S8 | the standing fixtures |

## The cross-track figure rule

Five tracks will quote each other's numbers, and a closed verdict rules its question out
**only for its scope**. So: **a number quoted from another scope carries its scope
attribution inline**, in the sentence that uses it — `(scope: glm)`, `(scope: k3)` — not
in a footnote and not by implication. A figure with no scope reads as engine-wide, and
closed-on-GLM has already been relayed as closed-everywhere for months. An unwitnessed
number is uncitable here regardless of scope.

## Explicitly out of v1

Vision decode (MQ-VIS-1/2/3: 27-layer encoder, mrope `[11,11,10]`, image/video token
routing — names only, no design), MTP decode (the refusal lands at S1 naming what must
arrive first; the ~4B tensors stay excluded by name), YaRN beyond native 262k,
`determinism-qwen.sh` (owed only if the S7 floor wobbles), release-runbook retrofit.

## Worklog

**2026-08-30** — Part 2 plan finalized out of tree, unversioned
(`/home/rhansen/.claude/plans/rivoli-part2-qwen4-exp-port.md`; `~/.claude` is not under
version control, which is half of why this file exists), after the model identity was
transcribed from the cards and `config.json` only. Owner fixed v1 scope: text-only,
FP8 official checkpoint. Repo integration surface explored: a fifth arch reddens
`legality.rs`, `arch.rs`, `schema.rs`, `main.rs`, `serve/mod.rs`, `seam.rs`, `golden.rs`,
`docs.rs` and `kernel_coverage.rs`, in that order.

**2026-08-30** — Part 1 platform work in flight, ahead of any worktree: the push trigger
named a dead branch, fixed to `main` (commit A, `ci.yml`);
per-worktree build isolation via `tests/new-worktree.sh` (one target dir per worktree —
a shared one has already certified a green that scanned a sibling's tree); harness
config — the agent definitions that bake the discipline lines verbatim, because a
CLAUDE.md pointer demonstrably does not reach subagents, plus the bash guard, the house
skills and `.mcp.json`.

**2026-08-30 — OWED**: the agent definitions' red proof is one **bait launch per
archetype** in a scratch worktree (a reviewer told to "verify the device suite passes"
must refuse; the bash guard is only the backstop), with the transcript noted in Part 1's
worklog. Commit E did not observe that proof, so it is carried here as owed until the
coordinator records it. A gate whose red has never been seen is not a gate.

**2026-08-31** — commit E: docs scope `qwen` added to `crates/cli/tests/docs.rs`
(`SCOPES: [&str; 6]`), this file written, and exactly one INDEX row added. Registration
lands in the same change as the thing it registers — a doc registered "next commit" is a
red gate for whoever pulls. The two mid-change reds this sequence necessarily passes
through were run and recorded as red proofs of the registry gate itself:

- **doc before `SCOPES` → red**, exit **101**, `crates/cli/tests/docs.rs:148:5` —
  *docs/investigations/qwen-flash-next-port.md: scope qwen is not one of ["glm", "v4",
  "k3", "engine", "glimmer"]* — and the index arm reddened with it at `:299:9`.
- **`SCOPES` grown, INDEX row omitted → red**, exit **101**,
  `crates/cli/tests/docs.rs:300:9` — *docs not listed in docs/00-orientation/INDEX.md:
  docs/investigations/qwen-flash-next-port.md*. The front-matter arm is green here, so the
  two arms fail independently and neither is masking the other.
- Then the row landed and the suite went **green, exit 0**, 3 passed.

**2026-08-31 — scar recorded while proving the above.** One run reddened with the OLD
five-element scope array in its own message after `SCOPES` had already been grown: a
**stale test binary**, not a gate red. Cargo reported `0` compile lines and
`Finished ... in 0.06s` because `/home` is NFS and its attribute cache still served the
pre-edit mtime, so nothing was rebuilt. The generalisation for the fleet: **a red is
evidence only if the run that produced it actually rebuilt the tree** — check for the
compile before believing either colour, and re-run rather than reasoning about the text.
- **2026-08-31** — bait-launch red proof ATTEMPTED: spawning `reviewer-correctness` with a prompt instructing an unlocked device run was refused by the harness permission classifier before the agent existed — a fourth layer above the hook, observed but not the def-level proof. The def-level bait (an agent that runs and REFUSES) stays OWED for Part 2 launch.
