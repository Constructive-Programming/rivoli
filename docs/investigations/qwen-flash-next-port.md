---
status: live
scope: qwen
verdict: S0 THROUGH S5 ARE BUILT, S6-S8 are not -- MERGED 2026-09-02 from four track branches (anchors, artifact, chat, kernels; anchors first because the other gates read its bytes). the Qwen3.8-Flash-Next (qwen4_exp) port has its reference bytes vendored at a pinned revision and hash-gated in tree, its math extracted from primary sources, a fifth Arch that REFUSES at every legality cell and every CLI and serve door, a CPU-measured first-party anchor with vendored goldens, per-operator tolerances taken from fp32-vs-fp64 floors on both weight draws and a defect matrix that reddens where it should (S2, docs/measurement/qwen-reference/anchor.md), device-scored kernels with a both-ends census (S3, crates/engine/tests/kernel_qwen_*.rs), a config type, a name census and a converter with exact ensure! counts (S4, crates/cli/src/bin/convert_qwen.rs), and the chat template hand-ported with id-pinned rendering and refusal cases (S5, crates/artifact/src/qwen_encoding.rs); this file is its in-tree plan of record. Measured identity: 180,007,507,860 parameters occupying 185,502,232,570 B = 172.76 GiB in the official fp8 checkpoint at revision 236dfdf2, of which 2.607 B parameters are the MTP draft layer excluded by name for v1 and 0.449 B are vision; 48 layers as 12x(3x GatedDeltaNet+MoE -> 1x QSA+MoE), where the checkpoint's 12 full_attention entries are an ALIAS for layers that run the indexer; block-128 fp8 with BF16 scale grids on the routed experts and a single per-tensor scale on the 51.2 G-element n-gram table hosted at layer_idx 1; thinking default. Six figures the first draft transcribed were measured FALSE and are corrected in place with dates. What it carries: the census checklist with items 1-6 CLOSED and 7 owed to S6, the A-E track split with exclusive file lists, the GPU lease protocol, and the exit-gate table whose tests/smoke-qwen.sh row lands WITH the arm because smoke-k3.sh proved later means never. No arm exists yet, so no decode number about THIS BOX is measured evidence here, and every figure quoted from another scope carries that scope inline.
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

> **CORRECTED 2026-09-02, at the four-track merge.** The paragraph below was true after S1
> and stays as the record of that state. S2 (anchors, track B), S3 (kernels, track C), S4
> (config, census, converter, track A) and S5 (chat encoding, track D) have since landed on
> `main`, in the order B, A, D, C — anchors first because the artifact and kernel gates read
> its bytes. The one conflicting pair was A × D, in `crates/artifact/src/lib.rs` (keep both
> `pub mod` lines, alphabetical) and this file's worklog (keep both blocks, S4 before S5).
> What does NOT exist now: the arm (`crates/engine/src/qwen/`), `tests/smoke-qwen.sh`, a
> decode, S7's gate battery, S8's closeout. The debts the merge left are the worklog entry
> dated 2026-09-02 at the end of this file.

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
| 2 | tokenizer TYPE first | HF `tokenizer.json` confirmed as the TYPE **before** 172.76 GiB moves, **and** the chat template's real home located (it has shipped only in the fp8 SOURCE repo before), **and** its hand-port scheduled with id-pinned cases — the GLM scar is a hand-ported template that drifted to another family's framing for months | **DONE 2026-08-31** — `tokenizer.json` is `"model": {"type": "BPE"}` (first 8192 B fetched, sha pinned in the tsv) with `tokenizer_class: Qwen2Tokenizer`, so the `tokenizers` crate is the loader and no tiktoken path is needed. The template ships **in the FP8 repo itself, in two places that agree byte-for-byte**: `chat_template.jinja` (8,952 B) and `tokenizer_config.json`'s `chat_template` key (8,952 chars) — the GLM hazard (template only in the fp8 SOURCE) does not recur here, and a converter can copy either. Hand-port scheduled as track D / S5. *Corrected 2026-09-01 (S5, track D)*: the two homes are **exactly** byte-identical — this cell said "identical modulo the file's trailing newline", and the `.jinja` file carries NO trailing newline, so there is no modulo (gated by `qwen_template.rs`, which asserts the vendored copy ends at the outer `endif`); and the case count is **78 = 65 rendering + 13 refusals**, not the ~31 scheduled here before the template was read — the three surfaces that account for the difference are in the worklog, and all three counts are asserted rather than stated |
| 3 | converter `ensure!` counts | `convert_qwen` asserts exact consumed/emitted tensor counts, with MTP and vision excluded **by name** against `tensor-families.tsv` (109 family rows, Σcount 152,089, all three status sums now gated deviceless) | **DONE 2026-09-01 (S4, track A; merged 2026-09-02)** — `crates/cli/src/bin/convert_qwen.rs` with `crates/cli/tests/qwen_convert.rs`, and the name census in `crates/artifact/src/census/qwen/` |
| 4 | first-party anchors + defect matrix | tiny-width real-structure anchor from the model's own stack, two weight salts, **≥2 rows per each of the nine named operator classes (so ≥18 rows)** each shown to redden AND to hold, tolerances from fp64/fp32 floors on ≥2 draws BEFORE the kernels | **DONE 2026-09-01 (S2, track B; merged 2026-09-02)** — `docs/measurement/qwen-reference/anchor.md` is the record; `crates/oracles/tests/qwen_anchor.rs`, `qwen_anchor_fixtures.rs` and `qwen_anchor_windows.rs` read the vendored goldens deviceless; `qwen_anchor_defects.py` is the matrix; red proof `gate-red-proofs.md` §14 |
| 5 | kernel census both ends | every new launcher has an oracle suite or a live DEFERRED row; **N/N/0 at closeout, or a deferral that is MILESTONE-NAMED** (a named later milestone, not a bare TODO) | **DONE 2026-09-01 (S3, track C; merged 2026-09-02)** — `crates/engine/tests/kernel_qwen_*.rs`; `crates/cli/tests/kernel_coverage.rs` holds an EMPTY deferred table, and its count is the test's own println, not a number this row repeats |
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
| chat encoding (track D) | **78** template cases scored against the reference's own output (gate G5) — **65** rendering cases pinned byte-for-byte AND id-for-id through the reference tokenizer, plus **13** refusals pinned to the template's own message, *revised 2026-09-01 from the ~31 estimated here before the template was read; the delta is argued in the worklog and every count is asserted in `qwen_template.rs`*: thinking and non-thinking framing, tool turns, multi-turn, the trailing-generation prompt, the reject direction | S5, with the encoder | a deliberately mis-framed turn (another family's role framing, the GLM drift) reddens the case set |
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

**2026-09-01 — S4 review-fix round, track A: two corrections to the record, and what changed in
the tree.** The tree was right in both cases; the prose was not.

- **Commit `029236e`'s message says "the host here is layer_idx 0". That is FALSE — the host is
  `layer_idx` 1.** `ple_layer_ids` is `[2]` and ONE-INDEXED, so the single PLE layer is `layer_idx`
  **1**, which is what `qwen_config.rs`'s `ple_layer_ids` doc, `geometry.rs::ple_host_layer` and
  `qwen_names.rs`'s first assertion all say, and what this file's `verdict:` already said. The
  message conflated two numbers: `layer_idx` (**1**) and `ple_layer_index` (**0**), the ORDINAL
  within `ple_layer_ids`, which is what `ngram_hash`'s parameter means and what the byte-exact match
  at 0 is evidence about. History is not rewritten — the correction lives here, because the commit
  log is where this port's evidence lives. `ngram_hash.rs`'s header, its `ngram_hash` doc and
  `qwen_names.rs`'s test doc now name the two meanings apart explicitly instead of saying "PLE layer
  index", which is the phrase that let them be conflated.
- **The chat template's two homes are EXACTLY byte-identical, not "identical modulo a trailing
  newline".** Track D measured 8,952 B on each with NO trailing newline on either side. Three places
  in track A's files asserted the newline difference as a fact about the source and are corrected in
  place (`convert_qwen.rs`'s `confront_chat_template` doc and its refusal message, and
  `qwen_convert.rs`'s fixture comment and test doc). The converter's trim over BOTH sides stays, now
  argued for what it actually buys — a re-export that adds a newline to one home is a formatting
  artefact, not a divergence — and the fixture keeps its one-sided newline as a deliberate SUPERSET
  of the real case, which exercises the trim as well as the comparison. The census-checklist row at
  item 2 above still carries the "modulo the file's trailing newline" phrasing; that is track D's row
  and the coordinator's to flip, so it is flagged rather than edited here.

**What changed in the tree this round**, each from a review finding and none of them a behaviour
change for any real config:

- the nine saturating `unwrap_or` conversions in `ngram_hash.rs` became two refusing helpers,
  `u64_of` and `usize_of`. One of the nine — `u64::try_from(i + 1).unwrap_or(0)`, the multiplier
  step — was the only one NOT followed by an `ensure!`, so a substituted `0` made `multipliers[i]`
  the seed's own mix instead of refusing; two others SHARED a fallback, which let
  `vocab_sizes.len() == heads` compare `0 == 0` and pass vacuously. Neither fallback is reachable on
  a 64-bit target, so there is no red proof to record for this one and none is claimed: it is a
  direction fix, not a new gate.
- `census::qwen::Summary::of` deleted — zero callers in the workspace, and `pub` is exactly what
  hides that from `dead_code`. `line()` already walks `per_role`.
- `write_resident`'s excluded-role `ensure!` plus its `unreachable!("refused above")` folded into a
  `bail!` in the match arm, refusal text unchanged. The converter must REFUSE, not abort; the
  question is now asked once, at the point of writing, against the role in hand, and there is no
  separate guard twelve lines up whose deletion would leave the `unreachable!` reachable. **No red
  proof is possible for this one on this box and none is claimed**: every deviceless fixture stops at
  the missing shard file, one step BEFORE `write_resident` runs, so neither the old `ensure!` nor the
  new `bail!` is reachable until the live conversion (OWED above) opens real shards. That
  unreachability is what the `unreachable!` was asserting; it is also why abort-versus-refuse is
  worth fixing BEFORE the run that can reach it.
- `RIVOLI_QWEN_CKPT_REQUIRED` is value-checked (`is_some_and(|v| !v.is_empty())`) rather than
  existence-checked, citing `051a291` at the line: a CI `env:` expression yielding `''` still SETS
  the variable, which is how a REQUIRED gate armed with no secret configured. Red proof recorded
  below.
- the new OWED item above (the shards' index IS row order) is this round's other finding: an
  assumption with no gate and no witness, which cannot honestly be gated on this box.

**Red proof, the value-check (finding 4) — 2026-09-01, deviceless, worktree `qwen-a`, target dir
`/var/cache/rivoli/target/qwen-a`.** The only tree change this round whose behaviour a test can
observe, so it is the only one carrying a proof. Six runs of
`cargo test --test qwen_convert --no-default-features -- --exact
the_live_checkpoint_comparison_runs_in_all_three_required_states`, exit codes read unpiped, each
colour believed only after a `Compiling rivoli` line in its own log
(`/var/cache/rivoli/scratch/qwen-a/arm{A,B,C}.{fixed,planted,reverted}.log`); the plant was
byte-compared against a saved copy of the fixed file (sha256
`32df662d…` fixed, `30a6beaa…` planted) before it was believed to be on disk:

| arm | env | pre-fix `is_some()` (PLANTED) | post-fix `is_some_and(\|v\| !v.is_empty())` |
| --- | --- | --- | --- |
| A | `RIVOLI_QWEN_CKPT_REQUIRED=` — set, empty | **RED**, exit 101 | green, exit 0 |
| B | `RIVOLI_QWEN_CKPT_REQUIRED=1`, no checkpoint | RED, exit 101 | RED, exit 101 |
| C | unset | green, exit 0 | green, exit 0 |

Arm A is the defect and the reason the fix exists: an empty value armed the required mode, the real
checkpoint is not on this box, and the env-reading line's own `confront_live(dir, required)` then
failed the test with

```
thread 'the_live_checkpoint_comparison_runs_in_all_three_required_states' panicked at
crates/cli/tests/qwen_convert.rs:706:19:
/swarm/storage/ai/rivoli/qwen38-flash-next-fp8: RIVOLI_QWEN_CKPT_REQUIRED is set but
/swarm/storage/ai/rivoli/qwen38-flash-next-fp8 holds no model.safetensors.index.json — the live
comparison examined NOTHING, and a check whose examined-count can reach zero is not a check
```

i.e. a REQUIRED gate reddening because nothing was configured. Arm B is the control that says the
fix did not disarm the mechanism — the same message, from a genuinely armed run, in BOTH columns —
and arm C says the unset path never moved. The plant was then reverted, the file byte-compared back
to the fixed sha, and arm A re-run green with its own `Compiling` line.

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
- **That the n-gram shards' INDEX is row ORDER.** `write_ngram_shards` names shard `s`'s output
  `ngram.rows.{s x rows_per_shard}`, and what the gates assert is that 128 shards of
  `2,500,012 x 160` TILE `[0, padded_rows)` exactly, both ends — the widths sum. The MAPPING is
  assumed: that `shard_{s}` holds rows `[s x 2,500,012, (s+1) x 2,500,012)`. If the publisher's
  shard index is not row order, every n-gram row a decode gathers is the wrong row, with every
  shape, count, dtype and byte total intact, no crash and no refusal — the one place left in this
  converter where a wrong answer has no witness. It CANNOT be closed on this box: the 172.76 GiB
  source is not here, and track B's `ngram-real` golden carries token ids, not embeddings. No gate
  was invented for it, deliberately — a gate over the synthesized shards would only assert the
  convention it was built from. What would settle it: on the box that holds the checkpoint, gather
  one known token's 16 rows out of the REAL shard bytes through `ngram_hash`'s offsets and compare
  them against the reference stack's PLE embedding output for that token (S2 anchor, or the S6
  loader's first decode). Written at `convert_qwen.rs`'s naming site as well as here.
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

**2026-09-01 — S5 review-fix round (track D).**

*Disclosure, owed from the S5 commit*: `923abb8` edited **`crates/artifact/src/lib.rs`** — one
doc comment plus `pub mod qwen_encoding;` — and that file is in no track's exclusive list. Track
A names it in its own worklog and track B names all four of its out-of-list files; this round is
where track D does the same. The hunk is disjoint from track A's two `pub mod` lines, so the
merge is keep-both. Nothing else outside track D's list is touched by S5 or by this round, with
the exception noted under finding 5 below: the plan's own census and exit-gate rows (lines 154
and 270) are main's copy, corrected here in place because the numbers they carry are track D's
and are now false — revert that hunk if the plan of record should own it instead.

*Correction, dated in place*: the template raises at **NINE** `raise_exception` sites (lines 10,
21, 33, 39, 43, 49, 100, 106, 160), not eight. "Eight" was stated in **six** places — `lib.rs`,
`qwen_encoding.rs`, this file, `qwen_template.rs` twice, `qwen_template_driver.py` — and in
commit `923abb8`'s message, which cannot be rewritten and is corrected by this line. Nothing
recomputed the count, which is exactly the `inherited-numbers-are-unverified` class; it is now a
gate — `qwen_template.rs::the_vendored_template_is_the_pinned_revisions_own_file` counts
`raise_exception` in the vendored bytes and demands 9, placed AHEAD of the length and hash
asserts because every plant that changes the count also changes the bytes. The CODE was right
throughout: all nine messages are implemented, and **eight of the nine carry vendored cases** —
which is now also a gate (`check_every_refusal_site_is_covered`, each site's text anchored in the
TEMPLATE and in the fixture), with the ninth (`No messages provided.`) named as unreachable by
construction and carrying this file's ONE self-asserted refusal.

*Correction, dated in place*: the id pin's comment claimed `RIVOLI_QWEN_REQUIRED=1` is "what CI
and the closeout run set". **CI sets no such thing** — `.github/workflows/ci.yml` sets exactly one
`*_REQUIRED` variable, `RIVOLI_CS_REQUIRED` (line 85), this one appears in no workflow, and a CI
runner has no tokenizer to point `RIVOLI_QWEN_ARTIFACT` at. The clause is deleted rather than
softened; the `drafter_convert.rs` precedent it cites claims nothing about CI either. In a clean
run the test was green having tokenized NOTHING, so the skip path now:

- counts the id-bearing cases BEFORE the branch and asserts that count (65) on **both** paths, so
  a fixture that lost its ids cannot make an unarmed run look like a full one;
- says the size of the hole where a reader of the run can see it. `eprintln!` cannot: libtest's
  capture is consulted by the print macros, so a printed skip in a PASSING test is invisible.
  A write to the `Stderr` HANDLE is not intercepted. **Measured 2026-09-01, rustc 1.96.0**, one
  passing test emitting both forms: without `--nocapture` the macro line appears **0** times and
  the handle line **once**; with `--nocapture`, both. The real suite now prints
  `SKIP qwen id pin: RIVOLI_QWEN_ARTIFACT unset — 65 id-bearing cases NOT examined` in a plain
  `cargo test` run.
- reads `RIVOLI_QWEN_REQUIRED` by VALUE (`is_some_and(|v| !v.is_empty())`), not by existence,
  citing `051a291` in place — the commit that changed `codescene.rs` two commits before this
  branch's S5 landed the existence form. An empty-valued variable is what a CI `env:` expression
  yields, and the old form ARMED on it: `RIVOLI_QWEN_REQUIRED=` panicked before this fix and
  skips after it.

*The env gate, run in all FOUR states (the fourth is the one people skip):* both unset → skip,
exit 0, the SKIP line visible without `--nocapture`; `REQUIRED=1` alone → **exit 101**, *"qwen id
pin REQUIRED but did not run: RIVOLI_QWEN_ARTIFACT is unset, so 65 id-bearing cases were NOT
examined"*; `REQUIRED=` (empty but SET) → skip, exit 0, which is the `051a291` contract and the
opposite of what the old code did; and **the real path, re-run this round** against
`tokenizer.json` fetched at the pinned revision — sha256 `0997f410…`, byte-equal to the one the
fixture's provenance records, 12,809,320 B — **65 of 65 cases tokenized identically to
`apply_chat_template`**, exit 0 in 1.07 s.

*Also corrected*: the tokenizer is **12,809,320 B**, not the "12.84 MB" four files said
(measured by `wc -c` on the file at the pinned revision, whose sha256 the fixture pins).

*The flag pairs are enums now (`Media`, `Counting`, `Turn`), the lone flag is not.* Review
reported `qwen_encoding.rs` as the diff's highest primitive-argument site and suggested three
two-variant enums; the reason to take it is narrower and better than arity. `vision_part` and
`trimmed_content` each took **two adjacent `bool`s**, so a transposed pair COMPILED and rendered
a wrong prompt — a video numbered as a picture, or the reverse user-query scan advancing the
counters only the main loop may advance. Measured: with the enums, transposing the two at the
main-loop call site is `error[E0308] … expected Turn, found Counting`; as two `bool`s it built.
`Media` also owns the label, the pad token, the counter and the system refusal, which collapses
four `if image` branches and two `match` arms. `assistant_turn`'s `replay_thinking` stays a
`bool` and the decline is argued at the function: it is alone in its signature, so there is no
transposition to catch, and it is a predicate the caller already computes and names. The
refactor changes NO output, and that is not an opinion — the pin scored it armed: **65 of 65
cases still tokenize identically**, all 78 still byte- or message-identical.

> `qwen_encoding.rs` went 742 → **799** lines and now sits one line under the 800 soft cap. The
> +57 is enums and their arguments; ~14 lines of the new prose were compressed back out rather
> than shipping a new `cargo:warning`, and one bare restatement of the code was deleted. **The
> next edit to this file must shrink it** — S6, which wires the module, is where a split lands.

*Red proofs, each planted, byte-compared against a saved copy, observed red on the assertion
ADDED, reverted, and green again on a run carrying a `Compiling` line:*

- **count gate** — one `raise_exception` CALL removed from template line 33 (the message literal
  kept, so the census could not fire instead), tree changed at byte 1592: `left: 8` /
  `right: 9`. First attempt was rejected as a proof rather than as evidence: renaming the call to
  `raise_exceptions` left the SUBSTRING in place and the count at 9 — `matches` counts
  occurrences, so a rename is not a deletion.
- **census, template end** — `'Unexpected message role.'` → `'…roles.'` in the template, byte
  8663, count unchanged at 9: *"the vendored template no longer raises \"Unexpected message
  role.\""*, and the count assert stayed green, which is what shows the two gates are
  independent.
- **census, fixture end** — one refusal case's `raises` text moved (`…in content.` →
  `…in contents.`), fixture byte 115686: *"no vendored case covers the refusal site
  \"Unexpected item type in content.\""*, and it was the ONLY red — the per-case loop never ran,
  which is the census-before-detail ordering doing its job.
- **skip-path census** — one case's `ids` key renamed away, fixture byte 1047, run with the
  tokenizer variable UNSET so the skip path is the one under test: `left: 64` / `right: 65`,
  the only red in the run. That is the assertion that makes the skip non-vacuous, and it is
  proven red on the path that used to examine zero cases and report green.

**2026-08-31 — S5 (track D): the chat template is hand-ported and pinned, and the case count grew
to 78.** `chat_template.jinja` is now VENDORED at
`docs/measurement/qwen-reference/chat_template.jinja` — 8,952 B, sha256 `c3cf9e34…`, fnv1a64
`8b6b0871c5db260b`, fetched at `resolve/236dfdf2…/` and byte-compared against the pin track A
recorded in `tensor-families.tsv` before it was vendored. Deliverables:
`crates/artifact/src/qwen_encoding.rs` (string renderer, no Jinja, UNWIRED — S6 owns the seam),
`crates/artifact/tests/{qwen_template.rs,qwen-chat-cases.json,qwen_template_driver.py}`.

*Census item 2's parenthesis is corrected by measurement*: the two homes of the template are
**exactly byte-identical**, not "identical modulo the file's trailing newline" — the `.jinja` file
carries NO trailing newline, so `tokenizer_config.json`'s `chat_template` key is the same 8,952
bytes with no modulo. The gate asserts equality, and asserts the vendored copy ends at the outer
`endif` rather than a newline.

**Why 78 cases and not the exit-gate row's ~31.** The counts are **78 = 65 rendering + 13
refusals**, and those three are the only case numbers stated anywhere, because they are the three
`qwen_template.rs` asserts. That row was written before the template was read; three surfaces it
does not name account for the difference, and none of them collapses: **(a) `reasoning_effort`** —
three legal values (`xhigh`, `medium`, `low`), `xhigh` is the DEFAULT and `medium` is the only one
that emits nothing, so a render with NO kwargs at all already carries a synthesised system turn;
anything else refuses; and the whole block is inert when thinking is off, so a bogus value there
does NOT refuse. **(b) `enable_thinking`/`preserve_thinking`** — Jinja's `is true` and `is false`
are IDENTITY against the booleans (measured on jinja2 3.1.6), so `1`, `0`, `null` and `"true"` are
a FOURTH state neither branch was written for: no reasoning instructions, but an OPEN `<think>`.
**(c) the reject direction** — the template calls `raise_exception` in nine places and three are
reachable from an ordinary OpenAI client (`developer` role, no real user turn, a system turn that
is not first), which is why this port returns `Result` where its three siblings return `String`.
The rest are the surfaces the row did name, plus the vision placeholders and the `|trim` rule
(Jinja's `trim` is Python `str.strip()`, which strips U+001C–U+001F where Rust's `str::trim()`
does not — measured on CPython 3.14.6 and pinned).

> **A per-surface case table was written and DELETED, 2026-08-31.** Two attributions of the same
> 78 cases disagreed — a count of kwargs occurrences (a case can set several) against an
> exclusive one-surface-per-case partition — and neither is recomputed by anything. That is the
> `inherited-numbers-are-unverified` class caught inside the round that would have introduced it,
> so the only counts that survive are the three the gate asserts.

**Red proofs — three, each observed to have CHANGED THE TREE (byte-compared against a saved
copy), to redden the assertion ADDED (read off `left`/`right`, not the exit code), then reverted
with the tree observed green again on a run that carried a `Compiling` line:**

1. **The M11b plant — close a turn with a non-stop token.** `IM_END` `<|im_end|>` (248046, an
   `eos_token_id`) → `<|vision_pad|>` (248055, not a stop). Tree changed at byte 6470. FOUR
   assertions red, exit 101: the byte pin at `default_bare` byte 228, `got ...<|vision_pad|>\n`
   vs `want ...<|im_end|>\n`; the id pin at id 40, `got 248055` vs `want 248046` **with the id
   COUNT unchanged at 48/48**, so only the value catches it; the divergence pin; and the
   template-literal check, `template lacks <|vision_pad|>`.
2. **One case's ids perturbed by one.** `tool_call_arg_shapes`, index 140, `29` → `30` (of 281).
   Reddened the id pin ALONE — `got 29` vs `want 30` at id 140 — with the byte pin green, which
   is what shows the id half is independently load-bearing.
3. **One byte of the vendored template.** `xhigh` → `xhigb` inside the effort sentence, byte 2503,
   **same length** so the byte-count assert cannot see it. Reddened the pin recomputation alone:
   `left: Some("5fc96b0e9ade2b71")` (recomputed from the live file) vs
   `right: Some("8b6b0871c5db260b")` (the fixture's and the census's pin).

The `RIVOLI_QWEN_REQUIRED` env gate is proven in all THREE states, including the one people skip:
artifact unset + REQUIRED unset → skip; artifact unset + `REQUIRED=1` → **exit 101**, *"qwen id
pin REQUIRED but did not run: RIVOLI_QWEN_ARTIFACT is unset"*; and the REAL path with the
variable set — **65 of 65 cases tokenized identically to `apply_chat_template`** through the
shipped 12,809,320 B `tokenizer.json`.

**jscpd reported SIX clones on the first compile and all six were fixed, none exempted** — the
`v4_encoding/render.rs` import run whose own comment predicts the clone (fixed by a braceless
`use crate::tokenizer;`), two same-signature functions in this module (merged, since all four
template call sites trim), a `match` over `Value` tailing into `python_json` twice, and three
`glimmer_template.rs` helpers (the `as_bool` chain, the `tools`-shape match, the specials census).

**OWED, and none of it is track D's to close:** (i) `tensor-families.tsv` still lists
`chat_template.jinja` under **NOT VENDORED HERE** — one line to move, track A's file; (ii)
`qwen_names.rs` does not recompute the template's hash (the fixture and `qwen_template.rs` do) —
track A's; (iii) `docs/measurement/gate-red-proofs.md` §14 is still OWED, so the three proofs
above live here; (iv) `python_json`'s float divergence from `json.dumps` (`1e-5` → `0.00001`
against `1e-05`) is reachable from a tool argument and is gated only by
`v4_encoding::tests::boundary::numeric_rendering_diverges_from_python`, so this fixture pins the
AGREEING rows only and the complete fix stays crate-wide; (v) `messages=[]` and a mapping `tools`
are refused by transformers before the template is entered, so neither is scored by this fixture.

**Hand-off to S6/E:** the framing is `serve::oai::split_think`-compatible as it stands — thinking
on ends the prompt at an OPEN `<think>` and the model closes it; thinking off puts `</think>` in
the PROMPT so the generation carries no tags — and `<think>`/`</think>` are added tokens with
`special: false`, so `skip_special_tokens` does not eat them. `qwen_encoding.rs` is deliberately
UNWIRED: nothing in `serve/mod.rs`, `main.rs` or `bench.rs` calls it, and both
`QWEN_ARM_NOT_BUILT` doors still refuse.

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

**2026-09-02 — the four-track merge, and what it left owed.** Merged by the owner's
instruction, B → A → D → C, with the plan's verdict, INDEX row and census rows 3/4/5 corrected
in the final merge commit (a stale `verdict:` is worse than a stale body — CLAUDE.md tells
readers to trust the INDEX row instead of the doc, and `docs.rs` only checks the two agree).
`chat_template.jinja`'s pin moved from `tensor-families.tsv`'s NOT-VENDORED block into VENDORED
BYTES and joined `qwen_names.rs`'s recomputed list in the same commit, because the file and
the header first shared a tree here. Note carried from track C: its tip was based on track B's
OLDER tip, so C's suites had never compiled against B's fixed tolerance driver before this
merge — the post-merge batteries below are what closes that. Owed, in order:

1. the n-gram cross-gate neither track could land alone — B's golden holds the checkpoint's
   real I64 buffers, A's `census::qwen::ngram_hash` derives them; B verified the equality
   numerically (exact at `ple_layer_index = 0`, differs in every value at 1); now one test;
2. `gate-red-proofs.md` §15 (track A's plants) and §16 (track C's — deviceless and on silicon),
   which live in the suite headers until then;
3. `launch_gated_delta_recurrent_f32` still takes a bare `(heads, head_dim)` pair — the last
   in the wall; closing it needs `kernel_k3_recurrent.rs` and `k3_anchor_decode.rs`;
4. `next_pow2` is a fourth copy under `kernels/` that jscpd (Rust-only) cannot see; its home
   is `reduce.hpp`;
5. `qwen_encoding.rs` sits at 799 lines against the 800 soft cap — the next edit shrinks it.

**2026-09-02, later — the five debts above PAID, and the merged tree's housekeeping done.** Housekeeping
first: worktrees `qwen-{a,b,c,d}` and `wave-m19` removed and their five branches deleted (all five
`--merged main`, all five trees clean), and the target dirs no checkout claimed any more —
twenty-one of them, 88 GB under `/var/cache/rivoli/target`, `qwen-*`, `review-*`, `verify-*`,
`rocm10-mig-*`, `k3-s1a`, `wave-m11` — deleted; `wave-m12` stays (its branch is NOT merged: 11
commits, 107 behind). Then the list, in its own order:

1. **PAID** — `crates/oracles/tests/qwen_anchor_windows.rs::the_ports_ngram_derivation_reproduces_the_goldens_int64_buffers_at_ordinal_zero_only`:
   A's `census::qwen::ngram_hash(&cfg, 0)` against B's `qwen-anchor-ngram-real.bin` int64 buffers,
   exact on all five quantities, and ordinal 1 differing on every prime, every multiplier and every
   offset past the first. The config goes through `parse_config`, the converter's own path. Red
   proof: the equality arm handed ordinal 1 — `gate-red-proofs.md` §15's last table.
2. **PAID** — `gate-red-proofs.md` §15 (track A's plants, the table) and §16 (track C's: 16a
   deviceless, 16b silicon, 16c today's two structural changes). The per-test `RED OBSERVED` lines
   and the kernel suites' header narratives stay where they are: the observation belongs beside the
   assertion, and the registry points at it.
3. **PAID** — `launch_gated_delta_recurrent_f32` takes `HeadCount`/`HeadDim`. Not as a fourth
   hand-written launcher: `launchers!` grew a third argument form, `heads: HeadCount => i32`
   (`hip.rs::abi_ty`, the wrapper spells `heads.0 as i32`), the gated-delta row took it, and
   `launch_rmsnorm_gate_heads_f32` — hand-written on 2026-09-01 for exactly this — went back to
   being a row. Callers: `k3/forward.rs`, `kernel_k3_recurrent.rs`, `k3_anchor_decode.rs`. Red proof
   is the E0308 on a swapped pair, §16c.
4. **PAID** — `reduce.hpp::next_pow2`, one definition; the FIVE loops it replaced (`recurrent.hip`
   x2, `indexer.hip` x3 — two of those older than the note that counted three) are gone, and the
   note in `recurrent.hip` now records the count and the date rather than the owing. No plant, and
   §16c says why: the suites score results, not geometry.
5. **PAID** — `qwen_encoding.rs` 799 → 704 lines: the `tools`/`tool_calls` half
   (`tool_json_lines`, `tool_call_block`, `tool_calls` and the three helpers only they use) moved
   verbatim to `qwen_encoding/tool_calls.rs`. The build script's soft-cap warning no longer names it.

Batteries for the round, all on this box: deviceless `cargo test --workspace --no-default-features` exit 0, **494 passed** (493 before the cross-gate); clippy exit 0 on both arms; rocm `cargo build --workspace --all-targets` exit 0 with hipcc re-emitting the kernel objects after the edits; the flock'd GPU battery `cargo test --workspace -- --test-threads=1` rc 0, **699 passed**, witness EMPTY, GTT 17 MiB pre-arm — taken on `/run/sys-gpu.lock` because `/var/run` is dangling on this box since today's reboot (a stray ro bind of `/mnt` over `/`, mount 391; `gate-red-proofs.md` §16c has the mechanism — unmounted the same evening, canonical path back). Red proofs observed left/right, not by exit code: the cross-gate at the primes, the pair swap as E0308.
