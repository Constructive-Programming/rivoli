# rivoli — orientation for agents

GLM-5.2 MoE decode engine. 78 layers (3 dense, 75 MoE), 256 experts top-8, hidden 6144,
vocab 154880. AMD Strix Halo gfx1151, unified LPDDR5 via GTT. Rust + HIP/ROCm — one
backend; a second, Vulkan, was retired 2026-08-06. The routed experts do not fit in memory,
so they stream from NVMe while the resident ones compute — that overlap is the whole
design.

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

`tests/docs.rs` enforces that every doc declares `status:`/`scope:`/`verdict:` and that the
index agrees. If you change a verdict, change both; the test will tell you which one you
forgot. `scope:` (`glm` | `v4` | `k3` | `glimmer` | `engine`) names whose evidence backs the
verdict — **a closed verdict rules its question out only for its scope**; a GLM-only closure
has already been misread as engine-wide once (npu-offload, 2026-08-07).

## Current state, so you don't go looking

| | |
|---|---|
| quality ladder | int4 **5.120** (best, slowest) > hybrid **5.189** (best overall, the default) > int3-vq **5.275** |
| speculative decode | on by default, **1.108×** via `--mtp-min-conf 0.8` (ungated it is 0.93–0.95×, a loss). All modes carry the head since 2026-07-31 |
| LOOKA hints (`--hint-k`) | **DELETED 2026-07-31** — measured inert (0.9% of evictions, ≤+0.1pp hit). `docs/investigations/cross-layer-prefetch.md` keeps the record |
| `top-m` routing | **RETIRED**, removed from the engine |
| Vulkan | **RETIRED 2026-08-06** — classified an unfinished port, not a feature. At retirement: 6 of its own 36 mode-matrix cells decoded and 30 refused (the matrix was 72 -- 36 per backend), 16 of 29 kernels, ~1.9× slower, no DeepSeek-V4 path at all. Preserved at tag `archive/vulkan-backend-hb16`; `docs/investigations/vulkan-kernels.md` keeps the inventory. **`rocm` is the only backend** — there is no `--features vulkan` |
| MoE accumulation | fixed-point (`MOE_ACC_SHIFT 44`), no cross-stream join |
| long-run divergence | **OPEN defect, not root-caused — and the instrument for it is not in this tree.** Two witnessed sole-tenant `--ppl` pairs diverge run-to-run (up to 0.55 PPL); the divergence POSITION moves between pairs, so it is a **timing race, not residency** — INV-1 exonerated by direct measurement, 0 of 388,875 records per arm. Window: layer *L-1*'s MoE compute against layer *L*'s attention. The `--features corruption-probe` / `--checksum-route` probe that established this lives only in tag `archive/belady-residency-bound` (`544fea7`, base `9ffb468`) and needs a forward-port before it can be re-run — it predates `0f39cc4` (`v4gpu.rs`→`f4gpu.rs`) and `b8ff613` (device router deleted), and its +211 lines in `src/gpu.rs` are the part to re-place. `docs/measurement/benchmarks.md` "Long-run divergence". **Any repro needs a contention witness** — the GPU flock is advisory and other agents skip it |
| layer-major prefill | **default since 2026-08-03** (flag deleted; `--trace` falls back to token-major): prefill **2.15×**, reads **159.56 → 28.20/token** (the floor), output byte-identical, every `--attn` mode. Decode pays a ONE-OFF ~2.7 s warm-up (1.8% of the prefill saving; the "1.55× slower decode" reading is a 13-pass artifact). Closing the sweep token-major was tried and **reverted** — useless. `architecture.md` §14 |

## Build and test

```bash
# DEVELOPING a feature — the dev profile, where debug_assert! is LIVE. Default to this.
cargo test --features rocm                   # 100 tests

# BENCHMARKS and performance evaluation ONLY.
cargo build --release --features rocm        # the only backend
cargo test  --release --features rocm        # HANGS intermittently — see below; sweep per-binary
tests/feature-matrix.sh                      # every feature combo compiles (34 cells, no GPU)
tests/mode-matrix.sh <artifact>              # mode x policy x attn, 36 cells, all decode (~90 min, GPU)
tests/smoke-matrix.sh                        # BOTH models x settings, 12 tokens/cell + V4 refusal asserts (~30 min, GPU)

# REGENERATES a vendored fixture; not part of any cargo run. Needs a pinned python env
# (K3_ANCHOR_VENV) and a GPU, because Kimi-K3's KDA ops are triton kernels with no CPU path.
# It gates each defect run: reddened nothing, or reddened a layer UPSTREAM of itself, is a
# failure. `cargo test --test k3_anchor` then reads the vendored bytes with no device — but it
# is a fixture-INTEGRITY gate, not a correctness gate: it compares no rivoli output to
# anything, because at S1b there is no K3 kernel to score.
tests/k3-anchor.sh                           # K3's goldens + 11 defects x 2 salts (~25 min, GPU)
# Muse Glimmer's equivalent, and it takes NO GPU and no lock — that reference is plain PyTorch
# with a CPU path for every operator, unlike K3's triton-only KDA. Same contract otherwise:
# a defect run that reddens nothing, or reddens outside its declared green set, is a failure.
GLIMMER_ANCHOR_VENV=/home/rhansen/glimmer-anchor/venv tests/glimmer-anchor.sh   # 14 defects x 2 salts x 2 modes (~10 min, CPU)

cargo clippy --release --features rocm --all-targets
# Before you claim a change compiles, ALSO run the union — see below.
cargo clippy --release --features rocm,otlp,teacher-forcing,pred-probe,trace,stale-sel --all-targets
```

**Develop on the dev profile. Use `--release` for benchmarks and performance evaluation
only.** `[profile.release]` sets `lto` and `opt-level` and **no `debug-assertions`**, and no
`.cargo/config.toml` overrides it — so under `--release` every `debug_assert!` in `src/` is
compiled out. There are **43** across 13 files (`grep -ro 'debug_assert[_a-z]*!' src/ | wc -l`;
23 of them bare), and the two in `kvcompress.rs` are described by their own doc as "what
ENFORCES the bsz=1 scope cut". Under `--release` they enforce nothing. The distribution as of
2026-08-05 is in `docs/investigations/v4-flash-port.md`.

> **CORRECTED 2026-08-11.** This said "**32**" and named `v4compress.rs` — a path `0f39cc4`
> renamed to `kvcompress.rs`, so the one worked example pointed at a file that does not exist.
> That is the exact rot this file warns about two sections down for jscpd exemptions. The
> command is given above because a count of the tree written in prose is a number nothing
> checks: re-derive it rather than quoting this line.

That is not a defect in the release profile — it is what every number in
`docs/measurement/benchmarks.md` was measured under, and putting bounds and overflow checks
on the hot path would change the thing being measured. It was a defect in *habit*: this file
prescribed `--release` for everything, so nobody ever ran the checks. **A run that is not
timing something should be a dev-profile run.**

So `debug_assert!` is the right tool for a cheap internal check again — but it fires only for
someone who follows the line above. A check that must hold in a shipped binary is an
`assert!`/`ensure!` and pays its cost, and a `debug_assert!` whose comment claims it
*enforces* anything is this repo's most common review finding wearing a new hat.

**`--features rocm` alone does not compile `mod otlp`, `src/eval.rs`, or the pred-probe and
trace paths.** That blind spot is not hypothetical: `otlp` sat broken for weeks on an
`E0609` — a `ProfileSummary` field renamed out from under it — while every prescribed
command passed, because nothing built it. Add the union run to any change that touches
`telemetry.rs`, `eval.rs`, `gpu.rs` or a `ProfileSummary` field.

> **CORRECTED 2026-08-05.** This said "**there is no CI**", and used that as the reason.
> There is: `.github/workflows/ci.yml`, gated since `5ef1f9a`. What it actually runs is
> narrower than "CI exists" and is the useful thing to know —
>
> **UPDATED 2026-08-06:** the `vulkan` job went with the backend, so there is ONE job.
>
> | job | runs |
> |---|---|
> | `host` | `cargo fmt --check`; a proof that the jscpd gate is *armed*; `clippy --release --locked --all-targets`; `clippy --features otlp,teacher-forcing,pred-probe,trace --all-targets`; `cargo test --release --locked` — all **featureless** |
>
> So **there is no `rocm` arm and no GPU arm at all.** Every `--features rocm` build, every
> HIP kernel, and every device test is checked exactly as often as someone runs it here. The
> featureless union step does cover `mod otlp` (gated on its own feature, with no backend in
> the cfg — that is the E0609 class, and it is now watched). **`src/eval.rs` is
> `all(teacher-forcing, rocm)`, so no CI job compiles it** — but CI never had a rocm arm to
> lose, and the local union run below is its real and prescribed coverage. The
> union-clippy instruction above stands and is now the only thing
> that checks it. And `cargo fmt --check` **is** gated, so a change that
> adds rustfmt violations breaks CI even though nothing local reports it; the tree has since
> drifted, so fix only the hunks your diff touches and leave a tree-wide reformat to its own
> `style:` commit rather than burying it in a feature change.

**Duplication is a build error.** `build.rs` runs `jscpd --min-tokens 15` over `src/`,
`tests/` and itself on every build and panics on any clone; `.jscpd.json` carries no
`threshold`, so there is no budget.

**But `cargo clippy` does not reliably observe it.** Reported 2026-08-05: `clippy
--all-targets` came back green **twice** on a tree that `build.rs`'s jscpd gate then rejected
on the very next `cargo test --no-run` — the gate had not run, after an in-place `git apply`
left the build script's fingerprint stale. (The observation is direct; the fingerprint
mechanism is the reporter's diagnosis and is not independently confirmed.) The lesson does
not depend on the cause: **clippy-green is not duplication-green.** Run something that
actually re-runs `build.rs` before claiming a change is clone-free. The same reporter hit it
twice more, both times on real clones that appeared only after `cargo fmt` reflowed calls which
had gained a seventh argument.

> **CORRECTED 2026-08-11.** This said the clones were ones "**rustfmt had created**", and
> concluded "a mechanical formatting pass can manufacture duplication". **It cannot.** The
> duplication was already written; reflowing only brought the two blocks over `minTokens` so the
> tokenizer could see them. `build.rs` has said so in place since 2026-08-06 and this file
> contradicted it: *"Nothing was added — the formatter let the tokenizer see what was already
> there."* The wording matters because it decides what you do next — "the formatter did this"
> invites reverting the format or exempting the region, and the only correct response is to
> **factor the duplication out**. Hit a third time 2026-08-11 (`tests/k3_anchor.rs`'s two
> config-check loops, 35 tokens) and fixed by factoring, not by exempting. **Fix what `cargo fmt
> --check` and jscpd report; neither of them is the author of what it found.**

**13** regions are exempt via `jscpd:ignore-start`, and
`tests/docs.rs::the_jscpd_exemption_count_is_derived` derives that number and goes red when
this line drifts from it — so read the count here and do NOT hand-count it; the note below
says why the obvious `grep` gets it wrong. Today: `artifact/model.rs` 3
(the dimension serde renames, one per architecture that shares them), `backend/hip.rs` 2 (the
ABI wall), `v4oracle/weights.rs` 2 (`WMat::Fp8`/`Fp4`, see below), and one each in
`artifact/quant.rs`, `bin/convert_glimmer.rs` (clap's Args/main boundary), `kvcompress.rs`,
`math.rs`, `v4oracle/numerics.rs` and `tests/f4_attn.rs`. Of the six that
went, two were `glsl_numerics.rs`'s and the other **four were deleted because their entire
argument named a file the Vulkan retirement removed** — `gpustream.rs`'s `Stream::raw`
("mirrors `vkstream::Stream::raw`"), its `Timeline` Send/Sync twin, its INV-4 half, and
`gpu.rs`'s launcher import list. jscpd was re-run without each: still 0 clones, so they were
suppressing nothing. **A stale exemption is a hole in the gate** — when the justification
names a deleted file, delete the exemption rather than rewording it. The survivors each carry
their argument in place: the HIP ABI wall (`backend/hip.rs`, two regions), `math.rs`'s frozen
`route_into_pre` oracle, `v4oracle/numerics.rs`'s transliterations, `kvcompress.rs`'s three
functions restated from the oracle, `artifact/model.rs`'s serde renames, `artifact/quant.rs`'s
`matvec_*` parameter lists, `bin/convert_glimmer.rs`'s clap boundary, `tests/f4_attn.rs`, and
the two `WMat` ones. Being a verbatim copy is the POINT in each; everywhere else, factor it.

> **CORRECTED 2026-08-11.** This count has now been wrong three times in two days, which is
> why it is a test and no longer a prescription to re-grep.
>
> It was stale at **Ten** (written before the glimmer port added one, and before
> `v4compress.rs` → `kvcompress.rs` and `tests/v4_attn.rs` → `tests/f4_attn.rs` — the survivor
> list above still named both deleted files). It was then re-derived as **Fourteen** by grepping
> for the marker text, and this file went on to assert **Thirteen** on the grounds that the
> fourteenth hit was `backend/hip.rs`'s own doc comment merely *discussing* the marker, and
> therefore inert.
>
> > **CORRECTED 2026-08-11, and the correction runs the other way: that mention was NOT inert.**
> > Measured on jscpd 4.0.5 with synthetic pairs carrying a 141-token duplicate (a code review
> > independently measured 5.0.11 and agreed on every row): a bare `// marker` exempts, **a `///`
> > doc comment exempts, and a mid-sentence mention inside a comment exempts**; only a string
> > literal does not. So `hip.rs`'s prose line WAS a live start, 62 lines above the real one,
> > pairing with the `ignore-end` 1150 lines later — the exemption began where nobody decided it
> > should, and the count of things jscpd honoured really was fourteen. **The fix was to reword
> > the prose so it stops being a marker**, not to argue about counting it; the count is 13 again
> > because there are now 13 markers, and jscpd still reports 0 clones either way, so nothing was
> > being hidden. Also measured: **an unpaired start does NOT exempt to end of file** — this file
> > and the test both said it did, from no measurement at all. It yields one clone, exactly as if
> > no markers were present.
> >
> > What the test enforces now follows from the measurement rather than from a guess: **the marker
> > text may appear ONLY on a bare marker line**, anywhere under `src/`, `tests/` or `build.rs`.
> > That is the one convention under which counting is unambiguous and a prose mention is a
> > visible edit instead of a silent widening — and it is why neither the test nor this paragraph
> > spells the marker out. The pairs-balance check stays, for the reason that survived: an
> > unpaired start means a region someone meant to exempt is not exempt, or a pairing crossed two
> > regions the way `hip.rs`'s did.
> >
> > **Caveat that applies to all of the above:** `build.rs` pins no jscpd version, so the gate's
> > marker semantics are an unpinned dependency. Two versions agree today; a third need not.

> **The `WMat` pair, added 2026-08-06 with the `src/` dedup.** `WMat::Fp8` and `WMat::Fp4`
> carry the same four fields because that IS the checkpoint's storage layout for both
> quantized formats — only the scale grid differs, and nothing in the bytes says which, so the
> variant has to carry it. Unlike the others this is not "a verbatim copy is the point": the
> factoring that removes the text (`Fp8(Q)`/`Fp4(Q)` over one payload struct) would keep the
> distinction, and was declined on cost — a `.0` hop in every arm of `WMat::rows`/`cols`/`row`
> and at every construction and pattern site in `tests/`. **If that hop is ever paid for
> another reason, delete these two exemptions rather than keeping them.**
>
> **CORRECTED 2026-08-06, same day.** This said "one clone remains unresolved and is not
> exempted: `v4oracle::compress_topk`'s parameter list against
> `tests/v4_attn_host.rs::oracle_cat`'s, which only the `tests/` side can fix." The `tests/`
> side then fixed it, by giving `oracle_cat` a `CompCase` struct in place of four bare `usize`
> — see that struct's doc for why there was no `src`-side fix (`compress_topk` is `pub`, its
> signature is the reference's, and collapsing it to one line is 101 characters against a
> `max_width` of 100). **The tree is at ZERO clones, none of them exempted away.** The record
> of what the clone was is kept here rather than deleted, on the same argument as the `WMat`
> note above.
jscpd is skipped with a warning if `npx` is absent, so the crate still builds without Node.

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
- **The other tenant is Kubernetes, and you evict it by UNLOADING, not by draining.**
  `ai/llama-swap` runs on this node (`rh-anine`, the only one labelled
  `hr-home.xyz/rocm=true`). `POST http://10.43.48.47:8080/unload` — the ClusterIP, never a pod
  IP — frees it reversibly; models reload on demand and the service stays up (measured
  2026-08-11: 41.4 GB of GTT → 174 MB, three models). **`kubectl cordon` + `drain` looks like
  it works and does not**: the ReplicaSet re-scheduled onto the cordoned node within seconds,
  because its tolerations do not cover `unschedulable`, so the run that follows is not
  sole-tenant. Scaling the deployment to 0 does work and takes the AI service down for the
  window. And the tenant can be invisible to KFD — llama-swap has held 1.6 GB of GTT with
  **zero** `/sys/class/kfd/kfd/proc/` entries — so read `mem_info_gtt_used` too, not just the
  holder count. It does take the flock when a model loads through `gpu-lock-wait.sh`, so
  `flock -w N -E 66` blocking is a legitimate reading, not a bug.
- **Always `-- --test-threads=1` on any suite that touches the device.** The "intermittent
  `gpustream` hang" recorded here for months is **not intermittent** — diagnosed 2026-08-05.
  libtest runs `#[test]`s in parallel, and each device test builds its own tier, pool and
  io_uring ring; the tell in `/proc/<pid>/task/*/wchan` is **four `io_sq_thread`s**, meaning
  four rings. Serialised, the same suite goes from a 12-minute hang to **3.52 s**. This is
  why the per-binary sweeps in `docs/investigations/v4-flash-port.md` pass while a bare
  `cargo test --release --features rocm` wedges. **`cargo test --lib` is a GPU arm** — it
  contains device tests, so it needs the flock and the serialisation like everything else.
- **Read GPU occupancy with `find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 | wc -l`,
  not `ls … | wc -l`.** The `ls` form returned **1 for an empty directory** at least once on
  2026-08-05 (confirmed with `cat -A`: the literal string `(empty)`), which reads as a
  phantom holder and was twice mistaken for a stale KFD entry. When a count is non-zero,
  resolve the PID and confirm `/proc/<pid>` exists before believing it.
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
