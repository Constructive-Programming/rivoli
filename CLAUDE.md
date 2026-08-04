# rivoli — orientation for agents

GLM-5.2 MoE decode engine. 78 layers (3 dense, 75 MoE), 256 experts top-8, hidden 6144,
vocab 154880. AMD Strix Halo gfx1151, unified LPDDR5 via GTT. Rust + HIP/ROCm, with a
second Vulkan backend. The routed experts do not fit in memory, so they stream from NVMe
while the resident ones compute — that overlap is the whole design.

## Read this before opening anything in `docs/`

**`docs/` is ~500 KB and most of it is closed investigation.** It is kept for what it
*eliminated*, not as reference. Reading it end to end will cost your context and teach you
mostly about rejected options.

1. **`docs/00-orientation/TOUR.md`** — two pages. If you are new, read this and stop.
2. **`docs/00-orientation/INDEX.md`** — every doc with a `status:` and a one-line
   **verdict**. Use the verdict column to decide what *not* to open; if it answers your
   question, you are done.
3. **`docs/reference/architecture.md`** is the only doc meant to be read whole.
4. **Everything else: grep it.** `grep -n "^## " <file>` for the map, then read one section.

Layout: `reference/` = true today · `measurement/` = how to measure and what was measured ·
`investigations/` = asked, answered, closed. **A doc that stops being true moves directory**
— that move is the signal.

`tests/docs.rs` enforces that every doc declares `status:`/`verdict:` and that the index
agrees. If you change a verdict, change both; the test will tell you which one you forgot.

## Current state, so you don't go looking

| | |
|---|---|
| quality ladder | int4 **5.120** (best, slowest) > hybrid **5.189** (best overall, the default) > int3-vq **5.275** |
| speculative decode | on by default, **1.108×** via `--mtp-min-conf 0.8` (ungated it is 0.93–0.95×, a loss). All modes carry the head since 2026-07-31 |
| LOOKA hints (`--hint-k`) | **DELETED 2026-07-31** — measured inert (0.9% of evictions, ≤+0.1pp hit). `docs/investigations/cross-layer-prefetch.md` keeps the record |
| `top-m` routing | **RETIRED**, removed from the engine |
| Vulkan | decodes `--mode int3-vq --attn dense`; 16 of 29 kernels; 6 more are single-row; ~1.9× slower |
| MoE accumulation | fixed-point (`MOE_ACC_SHIFT 44`), no cross-stream join |

## Build and test

```bash
cargo build --release --features rocm        # or --features vulkan; NEVER both
cargo test  --release --features rocm        # 100 tests
cargo clippy --release --features rocm --all-targets
# Before you claim a change compiles, ALSO run the union — see below.
cargo clippy --release --features rocm,otlp,teacher-forcing,pred-probe,trace --all-targets
```

**`--features rocm` alone does not compile `mod otlp`, `src/eval.rs`, or the pred-probe and
trace paths.** That blind spot is not hypothetical: `otlp` sat broken for weeks on an
`E0609` — a `ProfileSummary` field renamed out from under it — while every prescribed
command passed, because nothing built it. There is no CI, so a feature-gated module is
checked exactly as often as someone remembers to name its feature. Add the union run to any
change that touches `telemetry.rs`, `eval.rs`, `gpu.rs` or a `ProfileSummary` field.

**Duplication is a build error.** `build.rs` runs `jscpd --min-tokens 15` over `src/`,
`tests/` and itself on every build and panics on any clone; `.jscpd.json` carries no
`threshold`, so there is no budget. Twelve regions are exempt via `jscpd:ignore-start`, each
carrying its argument in place — the two backends' ABI walls, `math.rs`'s frozen
`route_into_pre` oracle, and `glsl_numerics.rs`'s transliterations. Being a verbatim copy
is the POINT in all three; everywhere else, factor it. jscpd is skipped with a warning if
`npx` is absent, so the crate still builds without Node.

`src/` is grouped by subsystem — `artifact/ memory/ fetch/ backend/` plus `gpu math attn
indexer telemetry watchdog eval` at top level. See `docs/reference/architecture.md` §11.

`--features teacher-forcing` adds `--ppl` (teacher-forced scoring, `src/eval.rs`). Off by
default: it is a quality instrument, not part of decoding. `bin/ppl`, which does the paired
statistics over its `.nll` output, needs no feature — it never touches the engine.

`--features pred-probe` adds `--pred-probe` (pre-attention router recall, the cross-layer
prefetch feasibility question; `docs/investigations/cross-layer-prefetch.md` §"Feasibility, settled"). Same rule and
the same reason: it puts a blocking D2H on the per-layer path, so it measures recall and a
tok/s from a probe build means nothing.

**Instruments go behind a feature AND a flag, never an env var.** Both of the above did
briefly read one, and an env var is invisible to `--help`, absent from the recorded command
line in `docs/measurement/benchmarks.md`, and silently active in a build that looks stock.

A featureless build compiles to a refusal stub — that is deliberate, not breakage.

## Measurement discipline — these have all drawn blood

- **The GPU is sole-tenant.** Never run two benchmarks at once. This also breaks *tests*:
  `DeviceTier::new` fails to allocate while a decode holds the budget.
- **Never `cargo build` between the two arms of a benchmark.** It evicts page cache and
  moved `ms/miss` from 1.36 to 5.14 in one measured pair.
- **`distinct` / `longest repeated block` do NOT measure quality.** They fire identically on
  a repetition loop, on spliced corruption, and on prose that restates a paragraph on
  purpose. They have misled three investigations. Read the text.
- **Rank on paired dNLL from `bin/ppl`, not on the PPL column.** It reports its own power;
  an interval straddling zero is *inconclusive*, not a pass. `tests/ppl-corpus.txt` is 762
  tokens and often underpowered — `tests/ppl-corpus-5000.txt` exists.
- **Cache policy and `--max-mem` are output-neutral in `int3-vq` and `int4`** (INV-1:
  routing never consults residency). If output changes when only those change, that is a
  bug — **and in `--mode hybrid` it does, measured 2026-07-31.** Hybrid's cache picks each
  expert's *format* (HOT→int4, COLD→vq3), so residency selects the arithmetic; `--max-mem`
  115 vs 70 gives different text. Open defect, predates the MTP work — `docs/reference/architecture.md`
  §8b under INV-1. Do quality A/Bs on a single-format mode, or hold cache settings fixed.
- **`docs/reference/architecture.md` §8b is a registry with a test.** A documented INV-n with no
  `inv_n_*` test, or the reverse, fails `tests/invariants.rs`. Don't add one without the
  other.

## Conventions

- Comments explain *why*, and carry the measurement that justified the choice. Match that
  density; a bare restatement of the code is noise here.
- Superseded docs are **corrected in place with a dated note**, not deleted — what an
  investigation ruled out is worth as much as what it found. Follow that.
- `rtk proxy <cmd>` shows unfiltered cargo/git output.
- Verify sync with `git rev-parse HEAD origin/main`, not `git log origin/main..HEAD | wc -l`.
