# rivoli — orientation for agents

GLM-5.2 MoE decode engine. 78 layers (3 dense, 75 MoE), 256 experts top-8, hidden 6144,
vocab 154880. AMD Strix Halo gfx1151, unified LPDDR5 via GTT. Rust + HIP/ROCm, with a
second Vulkan backend. The routed experts do not fit in memory, so they stream from NVMe
while the resident ones compute — that overlap is the whole design.

## Read this before opening anything in `docs/`

**`docs/` is ~490 KB of markdown. Do not read it.** Most of it is investigation logs kept
for what they *eliminated*, not reference material. Reading `benchmarks.md` (105 KB),
`docs/VULKAN.md` (117 KB) or `docs/NPU.md` (57 KB) end to end will consume your context and
tell you mostly about options that were rejected.

1. **`docs/README.md` first** — the index. ~4 KB, and it answers most questions outright or
   names the one section to open.
2. **`docs/ARCHITECTURE.md`** is the only doc meant to be read whole (44 KB). It is the
   engine as it is today, not as it was proposed.
3. **Everything else: grep it.** `grep -n "^## " docs/X.md` for the map, then read the one
   section. Each big doc opens with a STATE block giving the current answer in ~15 lines.

## Current state, so you don't go looking

| | |
|---|---|
| quality ladder | int4 **5.120** (best, slowest) > hybrid **5.189** (best overall, the default) > int3-vq **5.275** |
| speculative decode (`--mtp`) | shipped, on by default where buildable, **0.93–0.95× — a loss.** Only `--mode int3-vq` carries the head |
| LOOKA hints (`--hint-k`) | built, **default 0 = OFF**, measured inert (0.9% of evictions) |
| `top-m` routing | **RETIRED**, removed from the engine |
| Vulkan | decodes `--mode int3-vq --attn dense`; 16 of 29 kernels; 6 more are single-row; ~1.9× slower |
| MoE accumulation | fixed-point (`MOE_ACC_SHIFT 44`), no cross-stream join |

## Build and test

```bash
cargo build --release --features rocm        # or --features vulkan; NEVER both
cargo test  --release --features rocm        # 104 tests
cargo clippy --release --features rocm --all-targets
```

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
- **Cache policy and `--max-mem` are output-neutral** by construction (INV-1): routing never
  consults residency. If output changes when only those change, that is a bug.
- **`docs/ARCHITECTURE.md` §8b is a registry with a test.** A documented INV-n with no
  `inv_n_*` test, or the reverse, fails `tests/invariants.rs`. Don't add one without the
  other.

## Conventions

- Comments explain *why*, and carry the measurement that justified the choice. Match that
  density; a bare restatement of the code is noise here.
- Superseded docs are **corrected in place with a dated note**, not deleted — what an
  investigation ruled out is worth as much as what it found. Follow that.
- `rtk proxy <cmd>` shows unfiltered cargo/git output.
- Verify sync with `git rev-parse HEAD origin/main`, not `git log origin/main..HEAD | wc -l`.
