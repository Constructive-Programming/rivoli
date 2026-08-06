---
status: live
verdict: A staged 39% code reduction (36,100 -> ~22,050 lines) over two waves and nine tracks. Retiring the Vulkan backend is 6,600 of it and must run ALONE because its cfg sites reach 21 files. ~43% of this tree is TEST code and is not a line-count target: consolidating duplicated scaffolding is in scope, cutting coverage is not. 50% remains unreachable without dropping features beyond Vulkan.
---

# Refactor 2026-08 — the plan

**Measured baseline, post-merge at `76db536`:**

| area | code lines |
|---|---:|
| `src/` | 21,309 |
| `tests/` | 11,328 |
| `kernels/` | 3,463 |
| **total** | **36,100** |

Comments (~20,000) are **not** a target. They carry the measurements that justify the code;
this repo's recurring failure is a claim outliving its evidence, and deleting the evidence
would accelerate it.

## The 50% question, answered honestly

50% is 18,050 lines. Wave 1's four tracks total **~10,700 (30%)**; wave 2 adds **~3,350**, reaching **~14,050
(39%)**. The remaining gap is not recoverable by being cleverer:

- **~2,500** is `src/v4oracle/`, a deliberate second implementation. Its whole value is being
  redundant — a 44-defect matrix that static goldens cannot replace. Deleting it deletes the
  gate that caught the bf16 running fold and the Hadamard basis order.
- **~300** is `ModelConfig`/`V4Config` separation. Merging them reintroduces the hazard S1a
  built them apart to prevent, and which caught `routed_scaling_factor` and `index_topk`
  living in the wrong struct this month.
- **~3,300** is `gpu.rs` + `v4gpu.rs`, two layer loops. The *skeleton* (prefill, decode,
  argmax, EOS, profile) is genuinely shared and worth ~600. The bodies are two different
  architectures and merging them is how a GLM-shaped path ends up running on a V4 config.

**If 50% is a hard requirement, it is a product decision about which features to drop, not an
engineering one about abstractions.** Options, priced: retire GLM's `misa`/`dsa` variants,
retire `i4_audit` (697), retire the OTLP exporter, or drop the second artifact format.

## Verification: the three gates

This tree has something most do not — **a corpus of recorded deliberate breaks**, each with
the assertion that fired and the message it printed. Every stage in `v4-flash-port.md` left
one. That corpus is the regression suite for this refactor.

A change is a **refactor** iff every gate that applies to it holds:

| gate | applies to | how |
|---|---|---|
| **G1 byte-identical output** | anything on a decode path | same prompt, same seed, `sha256` of the emitted text and of the logits probe. Not "looks the same" |
| **G2 byte-identical artifact** | converters, format, quant | `convert_v4 --verify` over the real 43-layer checkpoint: 0 bytes differ |
| **G3 byte-identical ISA** | launchers, kernels, anything a macro emits | `llvm-objdump` diff of the built object. Count narrowly — a naive `v_fma\|v_mac\|v_mad` grep is mostly `v_mad_u64_u32` address arithmetic |
| **G4 the break corpus** | anything touching tests | every recorded deliberate break still goes red, **with the same message naming the same subject**. A break that goes green is a regression; a break whose message changes subject is the "0-ULP compare wearing a slot-rule failure message" defect again |

**G4 is the one that makes Track 3 safe and it is non-negotiable.**

## Standing hazards for every track

- The GPU is sole-tenant; the coordinator holds it. `flock /var/run/sys-gpu.lock`, build
  **outside**, re-check `find /sys/class/kfd/kfd/proc/ -mindepth 1 -maxdepth 1 | wc -l`
  **inside** the lock. `ls … | wc -l` returned **1 for an empty directory** on 2026-08-05.
- **`-- --test-threads=1` on every device suite.** The "intermittent gpustream hang" is
  parallel libtest building one io_uring ring per test. `cargo test --lib` IS a GPU arm.
- **Forbid GPU use in review subagents' prompts** — they inherit Bash and do not inherit the
  lock discipline.
- Develop on the **dev profile** (`debug_assert!` is live there and dead under `--release`),
  but know the **17× tax** on oracle-heavy suites: `v4_compress_kernel` is 43 s under
  `--release` and 719 s on dev.
- **clippy-green is not duplication-green.** Run something that re-runs `build.rs`. rustfmt
  can *manufacture* clones by reflowing a call that gained an argument.
- CI exists (`.github/workflows/ci.yml`) and gates `cargo fmt --check`, but has **no rocm arm
  and no GPU arm**. The rocm union-clippy is genuinely unchecked.

---

## Track 0 — `attn_out`, in parallel, and it is not part of the refactor

**Owns `src/attn.rs`, `src/v4gpu.rs`. Nobody else touches these until it lands.**

The V4 decode produces fluent, on-topic, **wrong** output: `attn_out` differs from the oracle
on 30,841 of 53,248 elements (57.9%) at max_abs 7.8e-2, ~20 bf16 ULP. `attn_norm_out` at ~1
ULP is the port's own real-dims prediction met exactly, so this is not re-association.

`tests/v4_attn.rs` passes 13/13 on the same `attention`, same ratio-0 layer, same oracle, at
**toy dims, 0 ULP, in the same lock hold**. So it is real-dims-only, or a difference between
how that harness builds `attention`'s arguments and how `v4gpu` does. **Those two have never
been compared, and that comparison is the job.**

One unchecked difference already found: the `Fp8W` adapter carries only `w` and `scale`,
while the pin's `Fp8Weight` carries `block`, `o_dim`, `i_dim` — all three discarded, with
`attention` re-deriving every extent from `Dims`. A placed shape disagreeing with the config
is invisible.

**This track is why Track 2 must not touch the six V4 attention launchers in its first wave.**

## Track 0b — the prompt encoding, and it blocks any quality number

**Owns `src/artifact/tokenizer.rs` and a new encoder module.**

DeepSeek-V4-Flash ships **no chat template**, deliberately. Its README:

> This release does not include a Jinja-format chat template. Instead, we provide a dedicated
> `encoding` folder with Python scripts and test cases.

The canonical format is `encoding/encoding_dsv4.py` (29 KB, with `tests/`), producing
`<｜begin▁of▁sentence｜>…<｜User｜>…<｜Assistant｜><think>`. **Nothing in rivoli ports it.** The
first decode fed raw text to a model expecting turn markers, which is why it continued and
repeated rather than emitting `<｜end▁of▁sentence｜>` — EOS handling is correct
(`v4gpu.rs:1879` breaks before pushing); the model was never in an assistant turn.

Port it **against the reference's own test cases**, not by inference. Until it lands, any
benchmark or quality number measures the wrong thing, and any "degenerate output" reading is
an artifact of framing.

---

## Track 1 — Retire the Vulkan backend · ~~~6,600 lines~~ **DONE 2026-08-06** · RAN ALONE

> **DONE 2026-08-06.** Landed as one commit on top of `8293e50`. Measured
> **12,420 deletions / 519 insertions across 58 files**, of which 10,857 raw lines were the
> 28 deleted files; **5,427 code lines** (non-blank, non-comment) in the deleted `.rs` alone.
> Code preserved at tag **`archive/vulkan-backend-hb16`** — NOT `archive/vulkan-backend`,
> which predates `c434de3` (HB=16) by one commit and is 54 insertions short of what shipped.
>
> **The ~6,600 estimate was ~2x low**, and the gap is instructive rather than an arithmetic
> slip: it counted the five Vulkan-only files as *code* lines and missed four consumers
> entirely — `examples/moe_bench.rs` (16 cfg sites, more than any file but `main.rs`),
> `tests/xbackend.rs` (a cross-backend byte comparison that cannot exist with one backend),
> `tests/kernel_coverage.rs`, and the cfg sweep itself, which was **79 sites across 18
> files**, not 63 across 17. See "Vulkan-only files" below for the corrected inventory.
>
> **Gates:** G1/G2/G3 all RAN and held; `tests/feature-matrix.sh` shrank 34 → **18 cells**,
> asserted by `tests/matrix.rs`, not eyeballed; `mode-matrix.sh` lost its 36-cell Vulkan arm
> and its `refused` outcome, since a refusal now means a bug.
>
> **Two things it cost, disclosed rather than inherited.** (1) The `vulkan` CI job was the
> only one running the union clippy AND the only one running `cargo test` under a backend
> feature; a featureless union step in `host` recovers `mod otlp` (the `E0609` class) but
> **`src/eval.rs` is `all(teacher-forcing, rocm)` and is now compiled by no CI job**. That
> is a known gap, not a silent one, and NOT the `otlp` failure mode: CI never had a rocm arm
> to lose, and `eval.rs` is covered by the local rocm union run CLAUDE.md prescribes.
> (2) `tests/kernel_coverage.rs` was the launcher-to-oracle census, keyed on `vk.rs`; it was
> deleted rather than re-keyed, so **no launcher in the tree has a coverage census**.
> Re-keying it onto `src/backend/hip.rs` is the obvious repair and is unclaimed.
>
> > **CLOSED 2026-08-06.** (2) is repaired. The census is restored, keyed on `src/backend/`
> > rather than on one file, so a launcher relocating into a sibling module no longer leaves
> > it silently. It found **18 of 48 launchers with no oracle** — `hip.rs` had never been
> > scanned — and thirteen new device oracles closed that to one: `launch_vaxpy`, which had
> > **no caller anywhere in the tree**. It was deleted rather than tested, kernel and
> > launcher together, and the census is green at **47 of 47**. All 22 rows of
> > `tests/refactor-gates/breaks.tsv` were fired on device the same day: 22 red, right
> > subject, 0 problems. (1) is unchanged and still open.

**Rationale.** 6 of 36 cells decode (`tests/mode-matrix.sh`), 16 of 29 kernels, ~1.9× slower,
refuses `int4`/`hybrid`/`dsa`/`misa` at startup, and cannot run V4 at all. Every V4 launcher
signature change this month cost a parallel edit to a backend that cannot use it. Classified
by the user as an **unfinished port, not a feature**.

**Vulkan-only files (6,210 measured):** `tests/vk.rs` 2550 · `src/backend/vk.rs` 2407 ·
`tests/glsl_numerics.rs` 303 · `src/backend/vkstream.rs` 38 · `kernels/vk/*` ~900.

> **CORRECTED 2026-08-06.** Raw lines, as deleted: `src/backend/vk.rs` **4148** (2413 code —
> the merge brought `c434de3`'s HB=16 work, +54) · `tests/vk.rs` **4060** (2550 code) ·
> `kernels/vk/*` **1770** across 22 files · `tests/glsl_numerics.rs` **499** (303) ·
> `src/backend/vkstream.rs` **123** (38). Plus two files the estimate did not list at all:
> `tests/xbackend.rs` **160** (81) and `tests/kernel_coverage.rs` **97** (42). Total
> **10,857 raw**. The modified-file sweep removed a further net 1,069 lines across 28 files.

**cfg sites in 21 further files:** `main.rs` 11 · `tests/xbackend.rs` 9 · `memory/device.rs` 8
· `artifact/config.rs` 6 · `backend.rs` 5 · `lib.rs` 4 · `serve.rs` 2 · `memory.rs` 2 ·
`fetch/stream.rs` 2 · and one each in `gpu.rs`, `memory/pin.rs`, `memory/routed.rs`,
`fetch.rs`, `fetch/asyncfetch.rs`, `tests/v4_pin.rs`, `tests/v4_pool.rs`, `bin/convert.rs`.

**Why it runs alone:** that list is most of the tree. Any parallel track conflicts.

**The trap.** `backend.rs` carries no `compile_error!` for the neither-backend case *on
purpose* — a featureless build is the backend-independent half and must still compile to a
refusal stub. Removing `vulkan` must not collapse `any(rocm, vulkan)` into `rocm` where the
featureless build depended on the disjunction. `lib.rs` documents this; read it first.

**Gates.** G1 on a rocm decode before and after. Featureless build still compiles.
`tests/feature-matrix.sh` shrinks to exactly the surviving cells and its cell count is
asserted, not eyeballed. `cargo clippy --release --features rocm,otlp,teacher-forcing,pred-probe,trace --all-targets`
clean. Delete the `vulkan` feature from `Cargo.toml` and the CI job with it.

**Also delete:** the Vulkan rows from `CLAUDE.md`'s state table and `mode-matrix.sh`'s
Vulkan arm, and move `docs/investigations/vulkan-port.md` to `closed-negative` with a verdict
naming the measurement that retired it.

## Track 2 — Macro-generate the ABI wall · ~1,000 lines · after Track 1

**Owns `src/backend/hip.rs`, the `extern "C"` wrappers in `kernels/*.hip`.**

Post-Track-1 surface: **47 HIP launchers** (839 lines, median 13, mean 17.9), **51 extern
decls**, **61 C wrappers**. Every one is the same five moves — declare the symbol, take raw
pointers, call, map the int through `check()`, document safety.

**The constraint that decides whether this is worth doing.** These launchers carry
measurements: `# Safety` blocks reading "must outlive `stream`'s completion", guards like
`!(limit > 0.0f && limit < INFINITY)` with the NaN-and-infinity argument in place, and
`contract(off)` pragmas justified by an ISA count. **The macro emits boilerplate; the
invocation site keeps the prose.** A macro that swallows the *why* costs more than the lines
it saves, and these regions are `jscpd:ignore` today precisely because the duplication was
argued for.

**Scope exclusion, first wave:** the six V4 attention launchers and `memcpy_dtod_async` stay
untouched while Track 0 hunts `attn_out`. They join in a second wave.

**Gate.** G3, per launcher: convert one, `llvm-objdump` diff the object, confirm
byte-identical, then batch. Any launcher whose generated form moves the ISA is one the macro
got wrong. G1 on a decode at the end.

## Track 4 — Crates for hand-rolled subsystems · ~1,000 lines · after Track 1, parallel with Track 2

**Owns `src/artifact/format.rs`, `src/v4oracle/weights.rs`, `src/serve.rs`,
`src/bin/convert*.rs`.** Disjoint from Track 2's file set — verified.

1. **safetensors** (~500). Hand-rolled parsing in `artifact/format.rs`, `v4oracle/weights.rs`
   and both converters. The `safetensors` crate is mature. **The oracle may use a crate** —
   engine-independence means not sharing code with *the engine*, and a third-party parser is
   not the engine.
2. **HTTP** (~400). `serve.rs` is 715 lines of hand-rolled HTTP/1.1 over `std::net`. Its
   header says this is deliberate: *"no HTTP crate, no async runtime, one request at a
   time."* **`tiny_http` respects that constraint; `axum` would not.** If the sync/no-runtime
   property cannot be kept, do not do this one and say why.
3. **`half`** (~100). Already a dependency — audit for hand-rolled bf16/fp8 conversions that
   duplicate it. `common.hpp`'s device-side codecs are NOT candidates; they are HIP.

**Gate.** G2 — `convert_v4 --verify` over the full checkpoint, 0 bytes differ, on both
artifact formats. The existing HTTP framing tests are pure host code and already the gate for
(2). Note `convert_v4 --verify` deliberately compares the **file against the mmap'd source**
rather than against the buffer it just wrote, because that comparison could never fail —
preserve that reasoning if the skeleton moves.

## Track 3 — One golden / defect-matrix harness · ~2,200 lines · LAST, alone

**Owns `tests/**`.** Runs after Tracks 1, 2 and 4 have landed, because Track 1 deletes
`tests/vk.rs` (2,550) and Track 2 changes launcher signatures.

`tests/common/` is **285 lines** against 11,328 in test files. Seven files independently load
and parse goldens; `v4_oracle.rs` alone defines 11 ULP/golden helpers, `v4_kernel.rs` 6. Every
V4 test re-implements: load golden → compare on a ULP metric → report per-tensor stats →
drive a defect matrix → assert anti-vacuity.

**Why it is worth the risk.** That last step is where the copies actively hurt. In one month
the five copies produced: a tautological anti-vacuity assert (twice, reported working both
times), a guard that could never fire, one that could never pass (`12 <= 8`), and one green
in the wrong dimension (`COMP_SLOTS` checking block 0 of a range). **Five instances of one bug
class, in five separately-written copies of the same harness.**

**Why it is the most dangerous refactor here.** These tests are the only thing between this
port and fluent wrong output.

**Gate — G4, and nothing less.** Every recorded deliberate break re-runs and must go red with
the same message naming the same subject. Build the corpus **first**, as a runnable script,
before changing a line of harness. If the corpus cannot be made runnable, this track does not
proceed.

**Design requirements, each from a defect this month:**
- An **anti-vacuity arm that involves no code under test** — assert the defect oracle differs
  from the clean oracle *before* asserting anything about a kernel. Its failure cannot be
  misread as a kernel fault, whereas the kernel-facing form reads as a tolerance needing
  widening, and `TOL` is shared so that "repair" silently degrades every other comparison.
- **Record non-separations as expected values, not skips**, with the metric that cannot
  resolve them named — the `BELOW_RESOLUTION` / `NO_YARN_BELOW_RESOLUTION` pattern. A dead
  record must not absorb a regression, so assert every entry was *reached*.
- **Tolerance is a property of the dimension**, not a constant. At `dim 4096` a correct
  wave-reduced kernel differs from the oracle on ~0.08% of bf16 elements; an injected rsqrt
  defect was caught at dim 256 and 512 and **not** at 1024. Bitwise is legitimate only where
  there is no reduction — one thread per element.
- **FMA contraction is uncontrolled tree-wide.** Any host reference for a contracted
  expression uses `mul_add`, with a bound that fires if contraction ever stops.

---

# Wave 2 — folded in 2026-08-05

Measured after wave 1 was scoped. None of these was visible until the baseline was taken
properly; each is grounded below.

**A finding that reframes Track 3.** In-src `#[cfg(test)]` blocks hold **~4,000–4,500 code
lines** — `quant.rs` 418, `model.rs` 313, `format.rs` 279, `math.rs` 214, `hybrid.rs` 187,
`telemetry.rs` 185, `serve.rs` 162, `arena.rs` 162, and `backend/vk.rs` ~1,125 which Track 1
(**done 2026-08-06** — so re-derive this estimate WITHOUT the vk.rs term before starting)
deletes anyway. With `tests/` that is **~43% of the tree in test code.**

**This is not a place to hunt for lines.** Five separately-written copies of one harness
produced five instances of one guard bug in a month, so *consolidating scaffolding* is
justified and in scope. *Cutting coverage* is not, and the V4 decode is the argument:
correct-sounding English out of a loop that is 58% wrong. Any track below that reduces what
is asserted, rather than how many times it is spelled, is out of scope by construction.

## Track F — extend Track 3 into `src/` · ~1,000 · with Track 3

Track 3 as scoped touches only `tests/`. The in-src blocks carry the same fixture-building,
golden-loading and ULP-comparison duplication. Same harness, same **G4** gate: every recorded
deliberate break re-runs red with the same message naming the same subject.

## Track G — collapse the converters · ~700 · after Track 4

`convert.rs` 528 + `convert_v4` 226 + `fp8_to_i4` 158 + `add_indexer` 44 share one skeleton:
read safetensors → per-layer loop → pack → atomic `.part`+rename → manifest → copy tokenizer
→ optional `--verify`. Cheaper after Track 4, which extracts the safetensors half.

**Preserve verbatim:** `convert_v4 --verify` compares the re-read **file against the mmap'd
source**, deliberately *not* against the buffer it just wrote, because that comparison could
never fail. If the shared skeleton averages that away it has made the gate unfireable.

**Gate:** G2 on both artifact formats — 0 bytes differ over the real checkpoint.

## Track H — delete `i4_audit` · 697 · anytime, independent

A straight deletion, not a refactor. Both citations are **closed** investigations
(`codebook-rotation.md:172`, `int4-scales.md:186,190`), and `int4-scales.md` already records
*"2026-08-01: `i4_audit` no longer re-derives them"* — the capability those docs describe was
removed from the tool before this. The investigations keep the record.

**Preserved at tag `archive/i4-audit`** (annotated, pushed). If an int4 quantization question
reopens, restore rather than rewrite — recipe and its two traps in `int4-scales.md` §8, which
is the single source of truth for it. (A bare `git checkout … -- src/bin/i4_audit.rs` is
**not** it, and stops working once Track K lands.)

**Gate:** nothing references it after removal — `grep` `docs/`, `tests/`, `*.sh`, `Cargo.toml`
— and `architecture.md:711`'s `src/bin/` inventory is updated in the same commit.

> **DONE 2026-08-05. Two corrections that generalise to the other deletion tracks.**
>
> 1. **"no inbound refs" was false — there were seven files, one of them build-breaking.**
>    `.github/workflows/release.yml:104` packaged the binary into every release tarball from a
>    hand-written `for b in rivoli convert … i4_audit ppl replay` loop. It fails at
>    `install -m755 target/release/i4_audit`, at TAG time, and is invisible to `cargo build`,
>    `cargo clippy` and every test. **Any track deleting a `src/bin/` target must check that
>    loop.**
> 2. **The gate as written is unsatisfiable.** "Nothing references it" contradicts the
>    corrected-in-place convention — closing this out *added* mentions. What was actually met:
>    **no BUILD reference, and every surviving prose mention sits in a dated note naming the
>    tag.** Use that wording for Tracks G and K.
>
> Also deleted: `quant::vq_decode_proj`, whose only caller this was. **Cut a deletion's
> orphans in the same commit** — it was briefly kept for a bit-identity with `matvec_vq` that
> no test checked, which is the failure mode the track existed to remove.

## Track I — the shared layer-loop skeleton · ~600 · after Track 0

`gpu.rs` and `v4gpu.rs` share prefill / decode / argmax / EOS / profile-summary. The merge
already produced a jscpd clone between them that was **in neither branch alone**, which is
the mechanical evidence that the skeleton is common.

**The bodies stay apart.** Merging the per-layer work is how a GLM-shaped path ends up running
on a V4 config — the same argument that keeps `ModelConfig` and `V4Config` separate, which
caught two misplaced fields this month. Factor the *loop*, not the *layer*.

**Blocked on Track 0**, which owns `v4gpu.rs` until `attn_out` is diagnosed.

**Gate:** G1 on both a GLM and a V4 decode — byte-identical output, and GLM's expert hit
counters identical, the way the pool refactor was proven (36595 hit / 14405 miss, exactly
equal across arms).

### Two checks Track H bought for every later track — 2026-08-06

**1. `.github/workflows/release.yml` has a hand-written binary list.** Track H's deletion of
`src/bin/i4_audit.rs` would have broken the release job: the loop
`for b in rivoli convert fp8_to_i4 add_indexer i4_audit ppl replay` packaged it into every
tarball. **Nothing local catches this** — `cargo build`, union clippy and every test stay
green, and it fails at `install -m755 target/release/<bin>`, at TAG time, in a job that only
runs on a release. **Tracks G and K both touch `src/bin/`; check that loop.**

This also retires the gate I wrote for Track H. *"Nothing references it"* is unsatisfiable
here, because closing a track under the correct-in-place convention **adds** mentions. The
honest form, which Track H met and which G and K should use:

> no BUILD reference remains, and every surviving prose mention sits in a dated note naming
> the tag.

**2. Intra-doc links are unverified tree-wide.** `RUSTDOCFLAGS="-D warnings" cargo doc` fails
on this tree today — measured 2026-08-06, **45 error lines**: one ambiguous `[`format`]` plus
~20 unresolved paths (`crate::backend::gpustream::Signal` ×6, `launch_moe_acc_drain` ×3). All
pre-existing, and **rustdoc is in neither CI job**, so nothing catches a link a refactor
breaks. **This lands hardest on Track K**, which moves `quant`/`format` symbols that other
modules link to by path. Either add `cargo doc` to the track's own gate or accept that the
docs' cross-references rot silently — but decide it, do not inherit it.

## Track K — unify the format/offset families · ~350 · after Track 4

`format.rs` 1,110 + `quant.rs` 1,029 carry three parallel `*_proj_bytes` / `*_slot_offsets`
families for `.vq3` / `.i4` / `.f4`. **Measured 2026-08-05: `.f4` and `.i4` produce identical
offsets across a band of dimensions** — equal iff `ceil(i/32) == 4*ceil(i/128)`, which holds
for every `i` in `97..=128 (mod 128)`, and **`moe_inter` cannot separate them at all**,
because w2's scale span begins at `off[5]` and so its length appears in no offset.

So the arithmetic already says there are fewer families than there are spellings. The unified
form must keep the property that the block count and `shared_block`'s refusal read **one**
`RoutedFmt::has_shared`, so they cannot disagree — that is what closed the `n_experts + 1`
trap.

**Gate:** G2, plus the existing `v4_loading` format-separation tests must still go red under
the recorded break (restoring `n_experts + 1` rejects the shipped `L00.f4` at 3422556160 vs
3435925504 — exactly one stride).

## Declined, with reasons

- **`otlp`** — ~150 lines for three heavy dependencies. Worth doing for **build time**, not
  for the line count; different justification, decide separately.
- **`misa` / `dsa`** (48 / 52 refs) — retiring `misa` might reach 500 lines, but whether it is
  live science or the next thing to retire is a research call, not a measurement.
- **Dead code** — hunted and not found: 5 `allow(dead_code)`, ~13 single-file `pub fn`s (mostly
  FFI decls), no unused dependencies. The mass here is deliberate.

---

## Sequencing

```
now    Track 0  (attn_out)       ─┐ parallel, owns attn.rs + v4gpu.rs
       Track 0b (encoding)       ─┤ parallel, owns tokenizer + new module
       Track H  (rm i4_audit)    ─┘ independent of everything; tag already cut

s1     Track 1  (retire Vulkan)     ALONE — DONE 2026-08-06; was 79 cfg sites over 18 files

s2     Track 2  (ABI macro)      ─┐ parallel, disjoint file sets
       Track 4  (crates)         ─┘

s3     Track G  (converters)     ─┐ parallel; G and K both build on Track 4
       Track K  (format/offset)  ─┤
       Track I  (loop skeleton)  ─┘ needs Track 0 to have released v4gpu.rs

s4     Track 3  (tests/)         ─┐ ALONE as a pair — the safety net goes last
       Track F  (in-src tests)   ─┘
```

| track | lines | gate |
|---|---:|---|
| 1 retire Vulkan | ~~6,600~~ **12,420 — DONE 2026-08-06** | ~~G1 + featureless build + asserted cell count~~ **all held**; estimate was 2x low, see Track 1 |
| 3 tests/ harness | 2,200 | **G4** |
| 4 crates | 1,000 | G2 |
| 2 ABI macro | 1,000 | **G3** per launcher |
| F in-src tests | 1,000 | **G4** |
| G converters | 700 | G2 |
| H rm i4_audit | 697 | ~~no inbound refs~~ **DONE 2026-08-05** — there were seven files, one build-breaking; see Track H |
| I loop skeleton | 600 | G1 on GLM **and** V4 |
| K format/offset | 350 | G2 + recorded break still red |
| **total** | **~14,050** | **39%** |

**Projected:** 36,100 → ~22,050, a **39%** reduction, with every feature except the Vulkan
backend and the `i4_audit` forensic tool intact.

## What would be a successful negative result

Any track that reports "this cannot be done at acceptable cost, here is the measurement" is a
success. Specifically: if Track 2's macro cannot carry the per-launcher prose, it should stop
at the extern decls and say so. If `tiny_http` cannot preserve the no-runtime property, Track
4 should skip it. If the break corpus cannot be made runnable, Track 3 does not start.
