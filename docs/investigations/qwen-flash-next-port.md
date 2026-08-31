---
status: live
scope: qwen
verdict: S0 and S1 ARE BUILT, S2-S8 are not -- the Qwen3.8-Flash-Next (qwen4_exp) port has its reference bytes vendored at a pinned revision and hash-gated in tree, its math extracted from primary sources, and a fifth Arch that REFUSES at every legality cell and every CLI and serve door; this file is its in-tree plan of record. Measured identity: 180,007,507,860 parameters occupying 185,502,232,570 B = 172.76 GiB in the official fp8 checkpoint at revision 236dfdf2, of which 2.607 B parameters are the MTP draft layer excluded by name for v1 and 0.449 B are vision; 48 layers as 12x(3x GatedDeltaNet+MoE -> 1x QSA+MoE), where the checkpoint's 12 full_attention entries are an ALIAS for layers that run the indexer; block-128 fp8 with BF16 scale grids on the routed experts and a single per-tensor scale on the 51.2 G-element n-gram table hosted at layer_idx 1; thinking default. Six figures the first draft transcribed were measured FALSE and are corrected in place with dates. What it carries: the census checklist with items 1, 2 and 6 CLOSED and 3, 4, 5, 7 owed, the A-E track split with exclusive file lists, the GPU lease protocol, and the exit-gate table whose tests/smoke-qwen.sh row lands WITH the arm because smoke-k3.sh proved later means never. Anchors, tolerances and defect matrices are all still OWED, so no number here about THIS BOX is measured evidence, and every figure quoted from another scope carries that scope inline.
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

## Where the port stands (2026-08-31, after S0 and S1)

> **CORRECTED 2026-08-31.** This section said "Nothing built. `Arch` still has four variants."
> Both halves are now false, and this is the plan every track authors from.

**S0 is closed and S1 has landed.** `Arch` has **five** variants: `Arch::QwenFlashNext` exists
and REFUSES at every legality cell, at `main`'s dispatch, at `serve`'s two arch matches and at
`bench::frame_prompt`, each quoting `rivoli_core::legality`'s own const. The reference bytes are
vendored and hash-gated (`docs/measurement/qwen-reference/`), the math is extracted from primary
sources (`docs/reference/qwen-architecture.md`), the golden container has its fifth magic
(`RIVQWGLD`), and `crates/artifact/tests/qwen_names.rs` recomputes the census on every deviceless
run. What does NOT exist: a `qwen` config type, a converter, a kernel, an anchor, an arm, a chat
encoding, `tests/smoke-qwen.sh`. Stages S2-S8 below are all still owed.

## Model identity (revision PINNED and vendored, 2026-08-31)

> **CORRECTED 2026-08-31, S0.** This section was headed "as transcribed (revision NOT pinned)"
> and said "the revision is not pinned anywhere". It is:
> **`Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13`**, with
> `config.json` (72,514 B) and `generation_config.json` (202 B) vendored under
> `docs/measurement/qwen-reference/` and recomputed from the live bytes by
> `crates/artifact/tests/qwen_names.rs`. The tech report is **read** (28 pp., pinned by blob in
> `docs/reference/qwen-architecture.md`'s source table), so the `(TR)` markers below are
> answered there rather than owed. Six figures in the bullets below were measured FALSE against
> the vendored bytes in the same commit that vendored them; each is corrected at its own line.
> Tertiary write-ups contradicted each other (one claimed 60B active) and are not used.
>
> **Read `docs/reference/qwen-architecture.md` for any math**, and
> `docs/measurement/qwen-reference/tensor-families.tsv` for any count or byte figure. This
> section is the identity summary, not the evidence.

- `model_type: qwen4_exp`, `architectures: [Qwen4ExpForConditionalGeneration]` **at the
  document ROOT**; `text_config.model_type` is `qwen4_exp_text` and
  `vision_config.model_type` is `qwen4_exp`, the root string verbatim.
- **180,007,507,860 parameters** occupying **185,502,232,570 B = 185.50 GB (dec) = 172.76 GiB**
  on disk: 176.95 B v1 parameters + **2.607 B MTP** + 0.449 B vision. 48 layers, hidden 2560,
  vocab 248,320 padded, RoPE theta 1e7, native ctx 262,144.
  > **CORRECTED 2026-08-31, S0.** Read "~180B on disk: 125B MoE (6B active/token) + 51B hashed
  > n-gram table + **4B MTP**". The MTP figure is wrong — the `mtp.*` families sum to
  > **2,607,304,448** parameters, re-derived from shape × count and gated by
  > `qwen_names.rs::the_parameter_counts_split_by_v1_status_are_re_derivable`. And "180B on
  > disk" conflated two units: 180 G is the PARAMETER count, the BYTE count is 185.50 GB,
  > because 5.49 B of the parameters are BF16 at 2 B each. Report bytes/token budgets from the
  > byte column, never from "180 GB" (P5). The 125B-MoE / 51B-table split is unrecomputed here
  > and is not cited by any gate; the tsv's per-family rows are.
- Layer pattern **12 × (3 × (GatedDeltaNet → MoE) → 1 × (QSA → MoE))**; `layer_types` is 36
  `"linear_attention"` + 12 **`"full_attention"`** at indices 3, 7, …, 47, with
  `full_attention_interval: 4`.
  > **CORRECTED 2026-08-31, S0.** Read "`layer_types` linear/full", neither of which is a value
  > the file contains. **And the checkpoint's `full_attention` spelling is an ALIAS**: the
  > reference rewrites it to `qwen_sparse_attention` before validating
  > (`configuration_qwen4_exp.py:180-184`, comment: "layers that are actually using an
  > indexer"), so those 12 layers RUN THE INDEXER. A dense reading of them is bit-identical
  > below 2051 cached tokens — invisible to the free oracle, the parity window and the smoke
  > decode cell alike. See `qwen-architecture.md` §3 and trap T19; any gate that must
  > distinguish them needs > 2051 cached tokens.
- GDN: conv kernel 4, **16 QK heads @128 / 48 V heads @128** (asymmetric). fla lineage,
  not KDA (TR: decay/gate form).
- QSA: 24 Q / 2 KV heads @256, partial RoPE dim 64; lightning-indexer-style MQA indexer
  4 heads @128 + 1 shared key head, `indexer_budget: 2048`, compress ratio 4 (TR: scoring).
- MoE **every** layer: 512 experts, top-10 + 1 shared, moe_intermediate 640 (TR: router).
- Hyper-connections: `hc_count: 4`, `hc_lowrank: 320` — a 4-wide residual stream through
  all 48 layers, so every sublayer is wrapped pre/post (TR: static vs dynamic form).
- Hashed n-gram embeddings: 20M entries per head (~51.2 G elements), bigram/trigram, hosted on
  **`layer_idx = 1`, the second layer**, via `ple_layer_ids: [2]`.
  > **CORRECTED 2026-08-31, S0.** Read "injected at layer 2 via `ple_layer_ids`".
  > **`ple_layer_ids` is ONE-INDEXED**: the reference tests `layer_idx + 1 in ple_layer_ids`
  > (`modeling_qwen4_exp.py:1202`), its config validation rejects ids outside
  > `[1, num_hidden_layers]`, and the checkpoint's only PLE tensors sit under `layers.1.`. A
  > converter that emits `layers.2.ple.*` must fail its census (trap T12).
  Pure per-token row gather — an ordered-unit stream, P6.
- Source artifact: the **official FP8 checkpoint**; tensor dtypes BF16 + F8_E4M3 + I64.
  Fine-grained **block-128 e4m3 with BF16 `weight_scale_inv` grids** covers the **routed
  experts** — the only families not in the config's 943-entry `modules_to_not_convert` list —
  and the **n-gram table is per-TENSOR**: one BF16 `weight_scale` scalar
  (1.9931793212890625e-4) for all 51.2 G elements, listed under `modules_to_convert`.
  > **CORRECTED 2026-08-31, S0.** Read "fine-grained block-128 e4m3 (the scheme `Fp8W` already
  > ingests)". True of the routed experts only. `Fp8W`'s block path does **not** apply to the
  > n-gram table, which needs a per-tensor-scale dequant; the scales are **BF16, not F32**; and
  > the grid orientation is per projection — `down_proj` is [20, 5] while `gate_proj`/`up_proj`
  > are [5, 20], a silent-dequant trap over 80.5 GB (trap T24's pair, `qwen-architecture.md`
  > §5).
- **Thinking mode is the default.** The vendored `generation_config.json` (202 B) carries ONE
  sampling set — `temperature 1.0, top_p 0.95, top_k 20`, `do_sample true`,
  `eos_token_id [248046, 248044]` — and it is the thinking set.
  > **CORRECTED 2026-08-31, S0.** Read "Sampling differs per mode (thinking T=1.0/top_p .95/
  > top_k 20; non-thinking T=0.7/top_p .80/presence 1.5) — from `generation_config.json`".
  > The thinking numbers are right and are in that file. **The non-thinking numbers are not**:
  > the whole file is 202 bytes, it has no `presence_penalty` key at all, and it declares no
  > per-mode split. Their true source is **UNKNOWN until vendored** — probably the model card,
  > which is pinned as CARD but not vendored — so track D must not encode them from this line.
  > Whoever needs them vendors the source first and cites it here.
- Owner scope decisions 2026-08-30: **text-only v1** (vision named as later milestones),
  FP8 official checkpoint as the source.

**Box constraint that shapes every gate:** the released fp8 checkpoint is
**185,502,232,570 B = 172.76 GiB on disk** (of which 174.51 GB is F8_E4M3 at 1 B/param and
5.49 B parameters are BF16 at 2 B) against this box's **~124 GiB usable** — so no released
precision fits resident on gfx1151, and the first-party reference cannot run here either.
Anchors are therefore operator-level taps plus a frozen teacher-forced window, never a live
side-by-side decode.

> **CORRECTED 2026-08-31, S0.** This read "≈ 180 GB against this box's ~124 GB usable", from
> "~180B parameters at 1 byte/param". The conclusion is unchanged and the arithmetic was two
> errors that happened to cancel: 180 G is the parameter count (185.50 GB of bytes), and not
> every parameter is one byte. The figure now comes from `metadata.total_size` in the
> checkpoint's own index, re-summed from the family rows by `qwen_names.rs`. Excluding MTP and
> vision leaves 181,906,343,962 B = 169.44 GiB for v1, which is still 1.37x the budget — the
> conclusion survives with room to spare, which is why the wrong number was invisible.

## Census checklist (the eight items; each closes with an artifact, not a memory)

| # | item | closes when | state |
|---|---|---|---|
| 1 | vendored config pins | `qwen-reference/config.json` **and `generation_config.json`** (the sampling-defaults source) vendored at a pinned revision, and the gate **recomputes** FNV-1a from the live file and byte-compares | **DONE 2026-08-31** — `Qwen/Qwen3.8-Flash-Next-FP8` @ `236dfdf285828023ca3bcd3f37366c58a3469b13`; `docs/measurement/qwen-reference/{config.json,generation_config.json,linear_attn_norm_l0.bin}` + `tensor-families.tsv`'s VENDORED BYTES block; recomputed by `crates/artifact/tests/qwen_names.rs::the_vendored_files_hash_as_the_header_records`, red-proofed by a one-digit pin flip |
| 2 | tokenizer TYPE first | HF `tokenizer.json` confirmed as the TYPE **before** 172.76 GiB moves, **and** the chat template's real home located (it has shipped only in the fp8 SOURCE repo before), **and** its hand-port scheduled with id-pinned cases — the GLM scar is a hand-ported template that drifted to another family's framing for months | **DONE 2026-08-31** — `tokenizer.json` is `"model": {"type": "BPE"}` (first 8192 B fetched, sha pinned in the tsv) with `tokenizer_class: Qwen2Tokenizer`, so the `tokenizers` crate is the loader and no tiktoken path is needed. The template ships **in the FP8 repo itself, in two places that agree byte-for-byte**: `chat_template.jinja` (8,952 B) and `tokenizer_config.json`'s `chat_template` key (8,952 chars, identical modulo the file's trailing newline) — the GLM hazard (template only in the fp8 SOURCE) does not recur here, and a converter can copy either. Hand-port scheduled as track D / S5 with ~31 id-pinned cases (exit-gate table below) |
| 3 | converter `ensure!` counts | `convert_qwen` asserts exact consumed/emitted tensor counts, with MTP and vision excluded **by name** against `tensor-families.tsv` (109 family rows, Σcount 152,089, all three status sums now gated deviceless) | OWED (S4) |
| 4 | first-party anchors + defect matrix | tiny-width real-structure anchor from the model's own stack, two weight salts, **≥2 rows per each of the nine named operator classes (so ≥18 rows)** each shown to redden AND to hold, tolerances from fp64/fp32 floors on ≥2 draws BEFORE the kernels | OWED (S2) |
| 5 | kernel census both ends | every new launcher has an oracle suite or a live DEFERRED row; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | OWED (S3) |
| 6 | registration in the SAME change | `SCOPES` entry + this doc + exactly one INDEX row | **DONE, this commit** |
| 7 | thin end-to-end smoke | `tests/smoke-qwen.sh` lands WITH the exit gate | OWED (S6) |
| 8 | behaviour+ABI naming | the rule binds **kernels, traits and structs** — none of those may be named `qwen_*`; membership is data in the census table. Arm directories and arch-scoped module/file names are the argued exception (`crates/engine/src/qwen/`, `qwen_config.rs`, `convert_qwen.rs`, `qwen_encoding.rs`): those name a *port boundary*, not a behaviour, and the exception is stated here so no reviewer has to guess | binding from S1 |

## Identity decisions already fixed

`Arch::QwenFlashNext` · kebab/docs scope `qwen` · manifest spellings
`"Qwen4ExpForConditionalGeneration"` and `"qwen4_exp"`, **both read at the document ROOT and
both required** ·
`crates/artifact/src/qwen_config.rs` · arm `crates/engine/src/qwen/` · golden magic
`RIVQWGLD` ·
`docs/measurement/qwen-reference/{anchor.md,config.json,generation_config.json,tensor-families.tsv}`
· math in `docs/reference/qwen-architecture.md`. Milestone label MQ, stages S0–S8;
M-numbers are owner-assigned.

> **CORRECTED 2026-08-31, S0/S1.** The parenthesis read "a nested `text_config` spelling is
> asserted by config validate, never accepted by the sniff", which reads backwards now that the
> file is in tree. The nesting is not hypothetical: **the config IS nested**, and every
> architecture dimension lives under `text_config`. What the sniff does is read the document
> **ROOT only** — `crates/artifact/src/schema.rs::arch_of_named`, gated against the vendored
> `config.json`. Five spellings are in play and only two resolve:
>
> | spelling | where | resolves? |
> |---|---|---|
> | `qwen4_exp` | root `model_type` | **yes** |
> | `Qwen4ExpForConditionalGeneration` | root `architectures[0]` | **yes** |
> | `qwen4_exp_text` | `text_config.model_type` | no — it names the text half |
> | `qwen4_exp` | **`vision_config.model_type`, the root string VERBATIM** | no — root-only reading is what excludes it |
> | `qwen3_next` | a real, foreign family member | no — no prefix or "close enough" match |
>
> The `vision_config` collision is why root-only is a decision rather than a habit, and why the
> sniff now also requires **both** root fields: a promoted sub-config carries `model_type`
> alone, and one statement of identity has nothing to agree with. Both halves are tested —
> `schema::tests::the_shipped_qwen_wrapper_resolves_from_its_root_and_a_promoted_block_does_not`
> runs the vendored file through the real resolution path.

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
| Coordinator | `legality.rs` + tests, `arch.rs`, `schema.rs`, `main.rs`, **`args.rs`**, `serve/mod.rs`, `bench.rs`, `docs.rs`, `golden.rs`, `seam.rs`, `engine/lib.rs`, `INDEX.md`, the how-to-measure floor row; hands the seam, `main.rs`, `args.rs`, serve and the legality flip to E after A/C/D merge |
| A artifact | `qwen_config.rs` + tests, `bin/convert_qwen.rs`, `tests/qwen_convert.rs`, `tensor-families.tsv` |
| B anchors | `qwen_anchor*.{rs,py}`, `tests/qwen-anchor.sh`, `qwen-anchor-*.bin`, `common/tolerance.rs`, `qwen-reference/{anchor.md,config.json,generation_config.json}`, `qwen-architecture.md` |
| C kernels | `kernels/*.hip` (new sections), `hip_attn.rs`, `hip_blocks.rs`, `hip_linalg.rs`, `engine/tests/kernel_qwen_*.rs`, `kernel_coverage.rs` |
| D chat | `qwen_encoding.rs`, `tests/qwen_template.rs`, `qwen-chat-cases.json`, `qwen_template_driver.py` |
| E arm | `engine/src/qwen/**`, `tests/smoke-qwen.sh`, plus the coordinator-handed seam files |

C authors deviceless against B's **vendored** fixtures, so C starts when the goldens are
vendored — not when B's doc is finished.

> **EXTENDED 2026-08-31, S1.** Three files were missing from every row and are now the
> Coordinator's, because S1 could not compile without them: `crates/cli/src/bench.rs` (the
> `frame_prompt` refusal arm), `crates/artifact/src/schema.rs` (the root-only sniff and its
> tests), and `crates/cli/src/args.rs` — which did not exist before this round. `args.rs` is
> the verbatim `Args`/parse block split out of `main.rs` to pay the 800-line soft cap **before**
> S6 needs the headroom (838 → 566 lines), so E inherits both files together.
> Also on the census gate: `crates/artifact/tests/qwen_names.rs` is A's, beside
> `tensor-families.tsv`.

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
| S1 reddening commit | every census and legality surface refuses the fifth arch with **asserted message fragments** (`QWEN_ARM_NOT_BUILT` and its siblings), so no track can land silently | S1, before any track opens | the refusals themselves — each observed firing at the level it is REACHABLE from, quoting its own fragment; `QWEN_ARM_NOT_BUILT` fires from the CLI, `serve` and `bench`, and `QWEN_MTP_NOT_LOADED` from a unit test only until S6 (see the note below the table) |
| anchors | tiny-width real-structure anchor from the first-party stack, all 48 layers, real `layer_types`/`ple_layer_ids`/hc=4/conv=4, **two weight salts** with disjointness asserted, vendored `RIVQWGLD` bytes read deviceless | S2, BEFORE any kernel | truncated golden reds every load; a mutated structural assert reds exactly that test |
| defect matrix | **≥2 rows for each of the nine named classes, so ≥18 rows** (GDN decay form, conv tap order, output-gate activation, indexer relu/pool/budget, hc transpose + lowrank order, router bias/norm/w1w3, n-gram seed/order/layer, partial-rope width, eps homes), gated **both directions** | S2, with the anchor | a deleted EXPECT_GREEN row reds the both-directions gate |
| tolerances | per-operator rows from fp32/fp64 floors on **≥2 weight draws**, under the existing derived-policy gate; `ExactOnly` where the margin collapses | S2, before the kernels they score | an under-floor tolerance reds the tolerance-rule gate |
| kernel census | every new launcher has an oracle suite or a live DEFERRED row, checked both ends; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | S3 | the three-edge DEFERRED proof for any deferral; per-suite plants from the observed defect classes |
| chat encoding (track D) | ~31 **id-pinned** template cases scored against the reference tokenizer's own output (gate G5): thinking and non-thinking framing, tool turns, multi-turn, the trailing-generation prompt | S5, with the encoder | a deliberately mis-framed turn (another family's role framing, the GLM drift) reddens the case set |
| parity window | frozen teacher-forced forward over a pinned ~64-token window (the M8 substitution — no live reference fits this box): per-position argmax agreement with flips confined to measured near-ties, NLL deltas within compounded tolerance and above the floor | S7 | a shadow artifact perturbed at scale — a single-byte flip is BELOW a short run's detection floor (scope: glm) |
| ppl cells | `tests/ppl-gates.sh <artifact> profile` and `p4` (both arch-agnostic); the paired `tf` cell's analogue here is the parity window. **p4 self-calibrates: on a non-reproducing arm it reports UNCALIBRATED (exit 1)** — there it is a diagnostic, never a merge gate | S7 | p4's red-proof corpus |
| determinism floor | A-vs-A twice on `tests/ppl-corpus.txt`, **before any dNLL claim** | S7, cell 3 — a precondition, not a result | the standing instrument's own self-test, `tests/determinism-glm.sh --self-test`: the comparator reddens on a changed id AND on a truncated stream. The qwen arm inherits that instrument; this floor **row** stays uncalibrated until it is measured on the qwen artifact |
| `tests/smoke-qwen.sh` | the thin CLI door-to-door: the `--mtp` cell quoted from the table's own fragment — **`QWEN_ARM_NOT_BUILT`'s fragment until S6 flips the row, `QWEN_MTP_NOT_LOADED`'s after** (see the note under this table), decode cell against recorded ids, `--attn` inertness cell, a prompt crossing the **layer-1** n-gram injection decoding finite | **the script lands WITH the arm at S6 — not later** (`tests/smoke-k3.sh` was scheduled for "after the port" and never existed: the named later-means-never scar). Its legality cells are **green immediately**; its decode cell is **RED until S7 records the reference ids**, and that red is the TDD ordering, not a defect | a wrong message fragment must redden it; **a missing reference is RED, never a skip** — which is exactly what makes the S6→S7 red meaningful |
| closeout | docs registry + jscpd + CodeScene + line caps green; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | S8 | the standing fixtures |

> **CORRECTED 2026-08-31, S1.** The smoke row said "`--mtp` refusal quoted from the table's own
> fragment", which cannot pass as written before S6. `main`'s `requested_flags` puts
> `Flag::Mode` first and `check_legality` bails on the FIRST `Refuse`, so `rivoli DIR --mtp …`
> prints `QWEN_ARM_NOT_BUILT` and never reaches the `Mtp` cell — `QWEN_MTP_NOT_LOADED` is
> reachable only from a unit test until the `--mode` cell becomes `Support`. The ordering is
> written down beside the const in `legality.rs` and gated by
> `main.rs::tests::a_qwen_mtp_invocation_is_refused_by_the_mode_cell_first`, which turns red in
> the same commit that flips the row — exactly where the smoke expectation has to change.
> Collecting every refusal and reporting the most specific one was considered and declined: it
> changes the refusal contract for four shipped architectures to improve a message on a fifth
> that cannot start.

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

**2026-08-31 — S4, track A: the artifact lane. Config, census, converter, and 22 plants.**

What landed, and the two files that are additions to track A's row rather than entries in it
(flagged here so the coordinator can move or reject them rather than find them):
`crates/artifact/src/qwen_config.rs` + `qwen_config/geometry.rs` (the derived widths, split under
the 800-line cap), `crates/artifact/src/census.rs` + `census/qwen.rs` + `census/qwen/ngram_hash.rs`,
`crates/cli/src/bin/convert_qwen.rs`, and the three gates
`crates/artifact/tests/{qwen_config.rs,qwen_names.rs}` and `crates/cli/tests/qwen_convert.rs` — 35
tests. Also `crates/artifact/src/lib.rs` (two `pub mod` lines) and
`crates/artifact/tests/k3_names.rs` (the shared-parser debt, below); neither is in any track's
exclusive list.

**The S4 parser debt is PAID, and it went to the LIBRARY rather than to a test helper.**
`qwen_names.rs`'s header named it: "lifting one shared helper is owed at S4, when the converter
census gives the second reader a reason to exist". S4 gave it three readers, and one of them is a
BINARY — `convert_qwen`'s exclusion list IS the census, read from
`tensor-families.tsv` at compile time so there is exactly one copy of the 109 patterns rather than a
frozen second copy in code. So the reader is `rivoli_artifact::census::Tsv` (library), the two
column contracts stay with their own censuses, and `k3_names.rs::families`/`declared` moved onto it
in the same commit. jscpd had already reported the digits-walk pair.

**The census line, printed by the converter itself** (against a synthesized 152,089-name index; the
real source is not on this box). Every arm named with its own tensor count and byte total, because a
grand total balances while a class vanishes:

```
convert_qwen: census 152089 tensors / 185502232570 B, of which v1 148655/181906343962:
routed-weight 73728/120795955200, routed-scale 73728/14745600, ngram-shard 128/51200245760,
ngram-scale 1/2, hash-param 3/280, resident 1067/9895397120, excluded-mtp 3101/2698026496,
excluded-vision 333/897862112
```

Those arms are pinned string-by-string by `qwen_convert.rs::the_synthesized_index_passes_every_
check_before_the_weights`. Derived from them: the artifact is **181,921,089,562 B**, i.e. the v1
source plus exactly one more copy of the scale grids (**+14,745,600 B**, the only bytes this tool
produces); routed expert weight is **112.5000 GiB** and the resident set **9.2158 GiB**.

**The n-gram hash is re-derivable, and the near-miss is measured.** `census::qwen::ngram_hash`
reproduces the checkpoint's three I64 buffers byte-exactly at `ple_layer_index = 0` — 16 primes, 16
offsets, 3 multipliers — and NOT at index 1, which is what makes "this checkpoint's PLE index is 0"
evidence rather than an assumption. The gate §5 of `qwen-architecture.md` called owed is closed by
`qwen_names.rs::the_ngram_hash_parameters_are_re_derivable_at_index_zero_and_not_at_one`. The
converter byte-compares the derivation against the source's own buffers, which is also the
mitigation for `NGRAM_SEED` being a constant rather than a config field: the JSON key for the seed
is not recorded by any primary source this port has read, and a guessed `#[serde(default)]` key
would be a guess wearing a schema.

**22 plants, 19 red, and the three that did not go red each said something.** Every plant was
applied ON the remote box (so the mtime comes from its clock, not from NFS's attribute cache), its
file sha256 recorded before and after, its run checked for a `Compiling` line before its colour was
believed, and then reverted. Full transcript: `/var/cache/rivoli/scratch/plants{,2,3}*.log`; each
assertion's left/right is recorded beside the test it belongs to. The reds:

- **P2** validate stubbed → `/text_config/model_type was mutated to a wrong value and the config
  still parsed` (the ROOT string sitting in `text_config`).
- **P3** the layer_types-vs-interval `ensure!` short-circuited → same shape on
  `/text_config/full_attention_interval`.
- **P4** `ple_host_layer` returning the ONE-BASED id → the `[4]` row parsed clean (its true host is
  layer_idx 3, a QSA layer; the un-decremented 4 lands on a GDN layer), AND
  `left: 2 right: 1` on the host index. Trap T12, both ends.
- **P5'** the two rotary homes no longer compared → the `rope_parameters/partial_rotary_factor` row
  parsed clean.
- **P6a** `qsa_q_width` without its factor of 2 → `left: [12288, 2560] right: [6144, 2560]`.
- **P6b** `gdn_qkv_width` halved → reddened EARLIER than expected, at config load, with `the GDN qkv
  width does not decompose into q + k + v`; recorded in place, because the interesting fact is that
  the config self-checks that width and does not self-check `q_proj`'s.
- **P7'** the pattern lookup turned into a skip → the run walked past an unclassified family and
  died on the weights instead of refusing.
- **P9** `down_proj`'s grid transposed in the TSV → the DERIVED orientation check named
  `[5, 20]` against `[20, 5]` over 40,265,318,400 B.
- **P10** `splitmix64` without the golden-gamma increment →
  `left: [16547087256759, 23703573157769, 20109073645365]` against
  `right: [23703573157769, 20109073645365, 8052911324071]`: the sequence SHIFTS BY ONE and carries
  two of the three correct multipliers in the wrong slots, so a set comparison would have passed.
- **P11** the `NgramShard` role predicate stopped matching → caught by `check_role_layout` at census
  load (`role resident wants Bf16 … but the row is F8E4M3 [2500012, 160]`), one layer before the
  arm-tally cell; recorded in place.
- **P12** `sparse_attention_layers` counting the GDN layers → `left: (73728, 147456) right: (24576,
  49152)` on the QSA KV budget.
- **P13/P13b** the layer-family bound forced open → each of the two families' refusals named its own
  fragment, which is what says they are checked separately.
- **P14** the per-pattern count relaxed to `<=` → the vanished-tensor run proceeded.
- **P15** the chat-template comparison short-circuited; **P16** `chat_template.jinja` dropped from
  `AUX`; **P17** the `total_size` confrontation short-circuited; **P18** the layer range clamped
  instead of refused; **P1'** `#[serde(default)]` added to `mtp_use_dedicated_embeddings` → the key
  moved from required to tolerated, printed as left/right lists.
- **P19** the required mode's `ensure!(!required, …)` short-circuited → `the required mode must
  refuse an absent source: 0`, i.e. a required gate passing on zero examinations.

The three non-reds:

1. **P1 was aimed at a test that never walks an absent key** — `norm_topk_prob` is not in the
   shipped document, so the required-fields walk never sees it. Re-aimed as P1', which reddened.
2. **P5** deleted a `validate_*` call, which made the function dead code and turned the run into a
   COMPILE error under `-D warnings` — an exit 101 that is not a red. Re-plumbed as P5', an
   in-function short-circuit.
3. **P7**'s first form left an always-`Err` expression behind, so the test PASSED for the wrong
   reason. That is the false-green shape on record, produced by the proof rather than by the tree,
   and it is why the plant was re-read rather than believed.

And one plant run **deliberately** to settle a question rather than to prove a gate: **P20** removed
`#[serde(default)]` from `norm_topk_prob` and the shipped-config test stayed GREEN after a real
rebuild. serde already treats an `Option<T>` field as absent-tolerant, so that attribute is
documentation and not the mechanism — said so at its declaration rather than left implying
enforcement.

**Deviceless gates, unpiped exit codes** (worktree `qwen-a`, target dir
`/var/cache/rivoli/target/qwen-a`): `cargo fmt --all --check` **0**; `cargo test --workspace
--no-default-features` **0** (jscpd and both line caps are build-script gates on that run, so they
passed with it — 8 clones were reported and all 8 were FACTORED or re-shaped, none exempted);
`cargo clippy --workspace --all-targets --no-default-features` **0**.

**OWED out of S4**, each named rather than left implicit:

- **The live conversion.** The 172.76 GiB source is not on this box. Every arm above runs against a
  synthesized index; the `RIVOLI_QWEN_CKPT_REQUIRED` cell is the mechanism for the real thing and is
  proven in all three of its states (unset+absent, set+absent, set+present-against-a-fixture, via
  `RIVOLI_QWEN_CKPT`), but the run against the real bytes — and therefore the first `resident.safe
  tensors`, the first `L{ll}.experts.safetensors` and the first 128 n-gram shards — is owed.
- **Census item 3's checklist row, this file's `verdict:` and its INDEX row.** The converter now
  asserts exact per-family counts against the 109-row census, both ends, with MTP and vision named.
  Flipping the row is the coordinator's: the `verdict:` is relayed verbatim into `INDEX.md`, and the
  recorded scar is a body corrected while the verdict kept the false claim.
- **`docs/measurement/gate-red-proofs.md` §14** is still OWED, as the S0/S1 round recorded; the 22
  plants above are here for the same reason that round's five were.
- **The norms are copied BF16 verbatim.** `convert_v4` widens because its loader reads f32; this
  port's loader does not exist, so the decision belongs to whoever writes it (S6) and is written
  down at `write_resident` rather than left to be discovered.
- **`q_proj`'s doubled width is labelled, not confirmed.** [12288, 2560] against `o_proj`'s 6144
  input forces `2 x n_heads x head_dim`; "q ‖ output-gate" is the reading, and the ORDER of the two
  halves is S2's anchor to settle. Nothing downstream depends on the label.
- **The n-gram seed's JSON key.** Bind it when a source states it; until then the constant plus the
  byte-comparison against the checkpoint's own buffers is the guard.
- **`chat_template.jinja`'s pin is not yet recomputed by anything.** Track D vendored the 8,952
  bytes beside `tensor-families.tsv` in ITS worktree; track A's tree has the sha and the fnv1a64
  (`8b6b0871c5db260b`) in the census header only, transcribed from D's fetch. At merge the TSV line
  moves from NOT-VENDORED-HERE into VENDORED BYTES IN THIS DIRECTORY and the name joins
  `qwen_names.rs::the_vendored_files_hash_as_the_header_records`, which recomputes FNV-1a over the
  live bytes. Written at the line itself rather than only here. It could NOT be written tolerantly
  in advance — `include_bytes!` on an absent file is a compile error, not a skip — so no
  cross-worktree coupling was created to close it early.

**2026-08-31 — S0+S1 review-fix round: five red proofs run, and the gates they belong to.**
Recorded here rather than in `docs/measurement/gate-red-proofs.md`, whose §14 is **OWED** — that
file's verdict is a single paragraph summarising every section, so adding one is a rewrite of a
scale this round did not carry, and the coordinator owns the call. Each plant below was observed
to have CHANGED THE TREE (byte-compared against a saved copy) and to have reddened THE ASSERTION
ADDED, read off `left`/`right` rather than off the exit code, then reverted and the tree observed
green again with a genuine rebuild:

1. `qwen_names.rs`, one digit of `config.json`'s recorded fnv1a (`6e35…` → `6e34…`) —
   `the_vendored_files_hash_as_the_header_records` red, `left: "6e35d0820f1ec218"` (recomputed
   from the live bytes) vs `right: "6e34d0820f1ec218"` (the pin); the other five tests stayed
   green, so nothing was masking it.
2. Same gate, the declared family-row count `109` → `108` —
   `the_vendored_census_closes_against_its_own_header` red, `left: 109 right: 108`.
3. Same gate, one digit of the declared MTP parameter count — the split-by-status test red,
   `left: 2607304448 right: 2607304449`, message naming "summed over 35 rows".
4. `schema.rs`, the both-fields `ensure!` short-circuited to `true ||` (the pre-round
   behaviour) — BOTH new sniff tests red, including
   `one_field_is_refused_on_the_four_architectures_that_already_decode` on `glm_moe_dsa`, which
   is what shows the rule is not qwen-scoped.
5. `serve/mod.rs`, the refusal replaced by a silent fallback
   (`Ok((String::new(), text.to_string()))`) —
   `the_reply_is_read_back_with_the_same_template_that_framed_it` red with
   `the armless architecture must refuse: ("", " to=user<|message|>hi<|eot|>")`, i.e. the
   fallback handing a client another model's raw turn markers, which is the failure the
   `unreachable!` this replaced was defending against.

A sixth plant is recorded as a **FAILED proof, and it cost a wrong belief for one run**:
inverting `requested_flags`' flag order to redden
`a_qwen_mtp_invocation_is_refused_by_the_mode_cell_first` reported **36 passed, exit 0**, with
`Finished in 0.06s` and no `Compiling` line — `/home` is NFS and its attribute cache served the
pre-edit mtime, so nothing was rebuilt. `touch`ing the file over ssh (so the mtime comes from
the remote clock) and re-running produced the red, `got [Mtp, Mode(Int3Vq), Attn(Dense)]`, exit
101. This is the stale-binary scar this file's own worklog already records, biting the same day
it was written down. **Every verification in this round was afterwards checked for a `Compiling`
line before its colour was believed.**

**2026-08-31 — S0+S1 review pass: this file's own figures corrected in place.** Six statements
of the model's identity were measured FALSE against the bytes vendored in the same commit, and
this is the doc `CLAUDE.md` tells a reader to trust instead of opening the others — with its
`verdict:` relayed verbatim into `INDEX.md`, which is the recorded scar "correct the verdict,
not just the body". Corrected at their own sites, each with what measured it: the 4B MTP figure
(2,607,304,448 parameters), "~180 GB on disk" (that is the PARAMETER count; the bytes are
185.50 GB = 172.76 GiB), `layer_types` "linear/full" (the values are `linear_attention` and
`full_attention`, and the second is an ALIAS for indexer layers), "injected at layer 2"
(`ple_layer_ids` is ONE-indexed; the host is `layer_idx` 1), "the scheme `Fp8W` already ingests"
(routed experts only, BF16 scales not F32, and the n-gram table is per-tensor), and the
non-thinking sampling numbers cited to a 202-byte file that contains only the thinking set
(their source is unknown until vendored). The `verdict:` was REWRITTEN rather than appended to,
and the INDEX row rewritten with it. Census items 1 and 2 flipped OWED → DONE with pointers, and
the smoke row's `--mtp` cell was corrected to the fragment a user actually gets before S6.

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
