---
status: live
verdict: A staged 30% code reduction (36,100 -> ~25,400 lines) across four tracks. Retiring the Vulkan backend is 6,600 of it and must run ALONE because its cfg sites reach 21 files. 50% was the ask and is NOT reachable without deleting further features or merging ModelConfig/V4Config, which reintroduces a defect class that has bitten twice.
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

50% is 18,050 lines. The four selected tracks total **~10,700 (30%)**. The gap is not
recoverable by being cleverer:

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

## Track 1 — Retire the Vulkan backend · ~6,600 lines · RUNS ALONE

**Rationale.** 6 of 36 cells decode (`tests/mode-matrix.sh`), 16 of 29 kernels, ~1.9× slower,
refuses `int4`/`hybrid`/`dsa`/`misa` at startup, and cannot run V4 at all. Every V4 launcher
signature change this month cost a parallel edit to a backend that cannot use it. Classified
by the user as an **unfinished port, not a feature**.

**Vulkan-only files (6,210 measured):** `tests/vk.rs` 2550 · `src/backend/vk.rs` 2407 ·
`tests/glsl_numerics.rs` 303 · `src/backend/vkstream.rs` 38 · `kernels/vk/*` ~900.

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

## Sequencing

```
now      Track 0  (attn_out)      ─┐  parallel, own attn.rs + v4gpu.rs
         Track 0b (encoding)      ─┘  parallel, own tokenizer + new module

wave 1   Track 1  (retire Vulkan)     ALONE — cfg sites reach 21 files

wave 2   Track 2  (ABI macro)      ─┐  parallel, disjoint file sets
         Track 4  (crates)         ─┘

wave 3   Track 3  (test harness)       ALONE — needs 1, 2, 4 settled
```

**Projected:** 36,100 → ~25,400, a **30%** reduction, with every feature except the Vulkan
backend intact.

## What would be a successful negative result

Any track that reports "this cannot be done at acceptable cost, here is the measurement" is a
success. Specifically: if Track 2's macro cannot carry the per-launcher prose, it should stop
at the extern decls and say so. If `tiny_http` cannot preserve the no-runtime property, Track
4 should skip it. If the break corpus cannot be made runnable, Track 3 does not start.
