# PROVENANCE — amd-tracelens-analysis-orchestrator

Vendored 2026-08-31. Nothing in this directory was authored here, and every file
except this one, `LICENSE` and `SKILL.md` is a byte-for-byte copy of upstream. **`SKILL.md`
carries two deliberate deviations and one subdirectory was DROPPED** — both recorded below,
because a vendored copy is allowed to be a subset and to be annotated, but it is not allowed
to be either one silently.

## Deviations from upstream (added 2026-08-31, review of commit D)

1. **`evals/` dropped at vendor time: unpinned fetch-and-execute infrastructure.** `evals/hooks.py` clones TraceLens at an unpinned default `main`, builds a venv, pip-installs, and calls `tarfile.extractall()` with no `filter=` — the CVE-2007-4559 path-traversal shape Python 3.14 made an error precisely because it is unsafe — and `evals/evals.json` embeds trace URLs an agent would download (3 files).
   Nothing in this repo referenced it, nothing here can run it (the upstream CLI is not
   installed on this box), and it was committed, greppable and looked sanctioned. **`SKILL.md`
   is the vendored surface**; the eval harness is upstream's own CI and stays upstream.
2. **`name:` in the front matter now matches this DIRECTORY** (`was `name: tracelens-analysis-orchestrator`, now `name: amd-tracelens-analysis-orchestrator``), where upstream
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
| Upstream path | `skills/tracelens-analysis-orchestrator/` |
| License | MIT (`LICENSE`, copied from the catalog root so the notice travels with the code) |

The catalog is itself a federation. This skill's `.federated.json` pins its own
source of truth one level further up, and unlike Magpie it DOES carry a hash:

| Field | Value |
|---|---|
| source | `amd-agi-tracelens` |
| repo | `AMD-AGI/TraceLens` |
| ref / commit | `main` @ `9f62389b78887c545eb8a165736f05fc2763fd36` |
| path | `TraceLens/Agent/Analysis/skills/analysis-orchestrator` |
| imported_at | 2026-08-28T22:10:53Z |
| **content_hash** | `d3c12fc332bfb57b092de3ae5c885d32f8646f300e8bdc14b9a3fc6b2c59c905` |

That hash is upstream's own, over upstream's tree, by an algorithm this repo has not
reproduced — it is recorded, not verified. Independent hash over the vendored tree **as checked in here** (19 files; the
as-fetched hash this file first recorded no longer applies — see Deviations):
`7cbb16ed6c919c605cd01c256172c04169a65efa643e0e15359c4bb381e8e2e6`. Reproduce with, from this directory:

```
find . -type f ! -name LICENSE ! -name PROVENANCE.md | LC_ALL=C sort | xargs sha256sum | sha256sum
```

## Review verdict

All 4,162 vendored lines were read: `SKILL.md`, the 652-line `reference.md`, 13 agent
files, 2 templates, and the eval harness. The decisive fact in this skill's favour is
that **it never touches the GPU**: the entire pipeline consumes an already-captured
PyTorch profiler trace, so the analysis runs deviceless while the device stays free —
exactly the property this tree wants, since a trace can be taken once under lease and
mined afterwards without holding the lock. Its epistemics are the best part.
`templates/sub_agent_spec.md` carries an explicit CANNOT-Infer table declaring bank
conflicts, cache hit rates, wave occupancy, LDS usage and warp-shuffle efficiency to be
unobservable from a trace, with mandated fallback prose instead of speculation, and it
states plainly that "traces show WHAT, not WHY"; micro-architecture speculation is
listed as forbidden in the Reasoning field, and an empty `category_findings[]` must be
rendered as an honest "no actionable issues" rather than padded with sub-threshold
cards. That is the same instinct as this tree's own rule that an unwitnessed number is
unciteable. The serious flag is hardware. Step 0 offers a closed platform menu of
MI300X / MI325X / MI350X / MI355X / MI455X (`reference.md:68-72`), and the report then
resolves peaks by reading `utils/arch/<platform>.json` for peak HBM BW and peak MAF
(`reference.md:564`), which every efficiency number and every "expect >70% of peak HBM
BW" band (`elementwise-analyzer.md:82`, `norm-analyzer.md:118`,
`convolution-analyzer.md:124`, `gemm-analyzer.md:110`) is then scored against. This box
has no HBM at all — it is LPDDR5 unified memory reached through GTT — so choosing an MI
entry to satisfy the prompt would silently score rivoli against another machine's
roofline and emit efficiency percentages that are fiction while looking authoritative.
Three smaller defects: `evals/hooks.py` clones TraceLens from the network at an
unpinned default `main`, builds a venv, pip-installs, and calls
`tarfile.extractall()` with no `filter=` (an unfiltered extract trusts member paths);
`reference.md:145,151` installs via `pip install git+https://github.com/AMD-AGI/TraceLens`
with no ref, so it is not reproducible; and the cluster branch shells out to
`ssh <node> "find / -maxdepth 5 ..."` (`reference.md:118,121`). Verdict: **admitted as
the more useful of the two**, on the strength of being deviceless and epistemically
disciplined, provided the roofline denominators are never believed on this box.

## Neutralized by note

- **(c) Instinct-class hardware — the load-bearing flag.** The platform menu has no
  gfx1151 / Strix Halo entry and no arch JSON exists for one. Every roofline
  denominator, `efficiency_percent`, and HBM-bandwidth expectation band in the 13 agent
  files is Instinct-derived. Do NOT pick an MI platform to get past Step 0 and then
  quote the resulting percentages: relative rankings and the kernel census may still be
  informative, the absolute efficiencies are not.
- **(a) Network fetches — CLOSED BY DELETION 2026-08-31, see Deviations 1.** The
  `evals/` directory is gone from this copy rather than neutralized by a note: a note that
  says "do not run this" still leaves committed, greppable, sanctioned-looking
  fetch-and-execute code in the tree, and review found no reader of it. What it contained is
  recorded in the verdict above. The remaining network surface in the WORKFLOW is
  `reference.md:145,151`'s `pip install git+https://github.com/AMD-AGI/TraceLens` with no
  ref, which is not reproducible and must be pinned by whoever installs it.
- **(b) Remote/device discipline.** Use the **local** environment branch (blank
  prefix). The cluster branch's ad-hoc `ssh <node>` and a `find /` filesystem sweep are
  not how this tree reaches rh-anine; the box is reached under the house lease.
- **Vendored paths do not match the instructions.** `SKILL.md:46,55` and
  `reference.md:312,381,428,501,538` point subagents at
  `TraceLens/Agent/Analysis/skills/analysis-orchestrator/agents/<name>.md`, the upstream
  checkout layout. Here those files live at
  `.claude/skills/amd-tracelens-analysis-orchestrator/agents/`. Rewrite the path when
  launching, or every subagent fails to find its own instructions.
- **Pinned subagent model.** All 13 agent files declare `model: claude-opus-4-7-high`
  in front matter, overriding the session's model choice for each spawned subagent.
- **Requires an external TraceLens install.** The Python that does the actual work
  (`orchestrator_prepare.py`, `category_analyses/*.py`, `report_utils`,
  `validation_utils`) is NOT vendored here — only the agent prose is. Without a
  TraceLens checkout this directory is documentation.

## House constraint

GPU commands in this skill run ONLY under the gpu-measure skill's lease + flock +
witness discipline; this box is gfx1151, not Instinct — treat performance advice as
hypotheses to measure.

(This skill issues none of its own — trace capture happens elsewhere, under lease; the
analysis is post-hoc and deviceless. The line stands for whoever captures the trace.)

## Earn-or-drop

This skill must show **one recorded useful run on gfx1151 during the qwen port** — a
cited, witnessed run that changed a decision — or it is **dropped at port closeout**.
A vendored skill nobody used is dead weight carrying an upstream's assumptions.
