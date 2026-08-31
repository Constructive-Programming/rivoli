# PROVENANCE — amd-magpie-kernel-evaluator

Vendored 2026-08-31. Nothing in this directory was authored here, and every file
except this one, `LICENSE` and `SKILL.md` is a byte-for-byte copy of upstream. **`SKILL.md`
carries two deliberate deviations and one subdirectory was DROPPED** — both recorded below,
because a vendored copy is allowed to be a subset and to be annotated, but it is not allowed
to be either one silently.

## Deviations from upstream (added 2026-08-31, review of commit D)

1. **`evals/` dropped at vendor time: unpinned fetch-and-execute infrastructure.** `evals/evals.json` plus four `evals/files/**` vector-add `.hip` fixtures for upstream's own eval runner, including the MI300X-peak-FP16 fixture that does not apply to this box (5 files).
   Nothing in this repo referenced it, nothing here can run it (the upstream CLI is not
   installed on this box), and it was committed, greppable and looked sanctioned. **`SKILL.md`
   is the vendored surface**; the eval harness is upstream's own CI and stays upstream.
2. **`name:` in the front matter now matches this DIRECTORY** (`was `name: magpie-kernel-evaluator`, now `name: amd-magpie-kernel-evaluator``), where upstream
   spells it without the `amd-` prefix. Claude Code validates the name against the directory,
   so as vendored the skill most likely did not load at all — which would have made
   the earn-or-drop clause unsatisfiable, since nothing could have invoked it.
3. **A house-constraint header is prepended to the SKILL.md BODY.** Every guardrail in this
   file used to live only here, and `PROVENANCE.md` is not what the model loads on trigger —
   grep confirmed no mention of the lock, the lease, gfx1151 or this file anywhere in
   `SKILL.md`. The header names the lease, the flock, the witness, and the fact that every
   roofline denominator this skill can emit belongs to another machine.

Consequently the tree hash below is recomputed over the directory AS IT NOW STANDS. The
upstream commit pin and the `.federated.json` figures are untouched and remain the provenance
of the CONTENT; this hash is the pin on what is actually checked in here.

## Upstream

| Field | Value |
|---|---|
| Repo | https://github.com/amd/skills (AMD Skills catalog, part of ROCm.AI GA 2026-08-27) |
| Pinned commit | `e867fa4ae4516f644221cb04dcdf24008a43cb99` (2026-08-28, "Bump `tracelens-analysis-orchestrator` to `9f62389` (#195)") |
| Upstream path | `skills/magpie-kernel-evaluator/` |
| License | MIT (`LICENSE`, copied from the catalog root so the notice travels with the code) |

The catalog is itself a federation. This skill's `.federated.json` pins its own
source of truth one level further up:

| Field | Value |
|---|---|
| source | `amd-agi-magpie` |
| repo | `AMD-AGI/Magpie` |
| ref / commit | `main` @ `12896a49a731ad72c791b7a23abcef7a0d6c4487` |
| path | `skills/magpie` |
| imported_at | 2026-08-11T21:37:43Z |
| content_hash | **absent** — this manifest carries no hash field |

Because upstream published no per-skill hash for Magpie, the integrity of this copy
rests on the catalog commit pin alone. Independent hash over the vendored tree **as checked in here** (5 files; the
as-fetched hash this file first recorded no longer applies — see Deviations):
`a4a87a11186c9bc8742ab9e55788769fddcc2515ec5a097ad93e068cb3839b05`. Reproduce with, from this directory:

```
find . -type f ! -name LICENSE ! -name PROVENANCE.md | LC_ALL=C sort | xargs sha256sum | sha256sum
```

## Review verdict

Every one of the 620 vendored lines was read. The skill is prose and YAML/HIP
fixtures only — no executable code ships with it, there is no `curl | sh`, and the
sole install instruction is a local `pip install -e .` against a Magpie checkout
(`SKILL.md:46`), so the network surface is PyPI dependency resolution rather than a
fetched script. Its epistemics are unusually good and much of it reads like house
doctrine: correctness is a hard gate before any performance ranking, "do not equate
successful execution with numerical correctness" (`SKILL.md:64`), an explicit warning
that Magpie's built-in PyTorch check only verifies finiteness and does not prove
numerical equivalence (`SKILL.md:76`, mirroring the `f32::max`-ignores-NaN trap this
tree already paid for), profiled runs are never the clean baseline (`SKILL.md:88`),
and a reproducibility checklist that demands commit, arch, driver, and build flags
(`reference.md:181-193`). Two things must be neutralized before use. First, the
entire CLI surface — `magpie --gpu-info`, `analyze`, `compare`, `benchmark` — compiles
and executes device work directly, with no notion of a lock, lease, or contention
witness; the preflight's "check GPU visibility" (`SKILL.md:42`) is the closest it
comes, and on a shared box that is not a mutex. Second, its worked examples assume
Instinct-class multi-GPU serving: `--tp 8` (`examples.md:91`), vLLM/SGLang/Atom
tensor-parallel benchmarking, and an eval fixture about MI300X peak FP16
(`evals/evals.json:87`). None of that maps onto one gfx1151 APU, and the benchmark
half of this skill has no subject here at all. What is genuinely useful is the narrow
kernel half — `analyze`/`compare` over a `.hip` source with a `hipcc` compile command
and a testcase whose exit code is the correctness oracle, which is the same shape as
this tree's own kernel oracles. Verdict: **admitted as a checklist, not as a runner.**
Treat it as a written procedure for structuring a kernel A/B; do not let it drive the
device.

## Neutralized by note

- **(b) Direct GPU execution.** Every `magpie analyze|compare|benchmark|--gpu-info`
  invocation in `SKILL.md`, `reference.md`, and `examples.md` is a device command and
  is NOT to be run as written. See the house constraint below.
- **(c) Instinct-class assumptions.** `--tp 8` (`examples.md:91`) and the multi-GPU
  serving frameworks do not apply to this box: there is no tensor parallelism here. The
  MI300X-peak-FP16 eval fixture that carried the same assumption is gone with `evals/`
  (Deviations 1).
- **(a) Network.** No fetch-and-execute. `pip install -e .` (`SKILL.md:46`) still
  resolves dependencies from PyPI; the two GitHub links in `reference.md:81,92` are
  documentation references, not fetches.
- **Inert without an external checkout.** The `magpie` CLI is not installed on this
  box and is not vendored here. This directory is documentation until an AMD-AGI/Magpie
  checkout exists, and the skill itself says the checked-out version is authoritative
  over its own flag lists (`reference.md:3`) — so treat every flag above as unverified.

## House constraint

GPU commands in this skill run ONLY under the gpu-measure skill's lease + flock +
witness discipline; this box is gfx1151, not Instinct — treat performance advice as
hypotheses to measure.

## Earn-or-drop

This skill must show **one recorded useful run on gfx1151 during the qwen port** — a
cited, witnessed run that changed a decision — or it is **dropped at port closeout**.
A vendored skill nobody used is dead weight carrying an upstream's assumptions.
