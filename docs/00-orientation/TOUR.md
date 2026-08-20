---
status: live
scope: engine
verdict: Two pages of orientation for the rewrite tree — what rivoli is, where the layering lives, which gates guard the tree, and where the old tree's evidence remains reachable.
---

# Tour

rivoli decodes LLMs bigger than memory on one box: AMD Strix Halo (gfx1151), unified
LPDDR5 via GTT, weights streamed from NVMe **overlapped with compute**. The overlap is the
whole design. This tree is the ground-up rewrite; the previous implementation lives on
tag `v2` / `archive/glimmer-s2` (pinned in CLAUDE.md; was branch `wt/glimmer-s2`) and stays running as the parity reference
until M5 retires the comparison.

## Why a rewrite

The old tree proved four model families (GLM-5.2 MoE, DeepSeek-V4-flash, Kimi-K3, Muse
Glimmer) and accumulated the measurements in `docs/` there — but its strain points were
all layering failures: two whole decode loops with no shared seam, three pin types in one
file, a 3,310-line config file, a serving layer hard-typed to one engine. The rewrite
keeps every hard-won lesson (see `../reference/principles.md`, owner-confirmed) and makes
the proven boundaries *crates*, because crate edges are the only layering Rust enforces.

## The workspace

| crate | is | may depend on |
|---|---|---|
| `rivoli-core` | pure planning: residency, spans, legality, gates, tolerances | nothing |
| `rivoli-artifact` | formats, per-model configs, sniffing, tokenizer, converters | core |
| `rivoli-oracles` | frozen references, golden container, anchor readers | core |
| `rivoli-backend` | the waist: HIP streams/launchers/kernels; owns `rocm` | nothing |
| `rivoli-engine` | the interpreter: fetch, executors, per-arch loops, `enum Engine` | all above |
| `rivoli` (cli) | thin main/serve/bins; hosts the workspace meta-gates | engine |

`rivoli-core` cannot name a weight format — that is what makes the old tree's worst open
defect (residency selecting arithmetic) un-writeable here rather than merely tested.

## The gates

Every claim is a gate that can go red (P7), and each gate was proven able to fail before
its green was believed. jscpd (duplication, zero budget) runs on every build; CodeScene
10/10, the docs registry, and the derived exemption ledger run in `crates/cli/tests/`;
clippy runs at `-D warnings` with `unwrap`/`expect` denied; CI runs the featureless
workspace. Anchors (first-party-reference goldens with defect matrices) land before the
code they score — that ordering is what makes this TDD rather than test-after.

## Reading order

1. This file.
2. [INDEX.md](INDEX.md) — every doc with a status, scope, and one-line verdict. Decide
   what NOT to open from the verdict column.
3. `../reference/principles.md` — the seven principles a plan is checked against.
4. The old tree's `docs/` (at the pinned SHA) for closed investigations — kept there for
   what they *eliminated*; grep it, don't read it end to end.
