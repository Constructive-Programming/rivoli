---
status: live
scope: engine
verdict: THE ARENA IS NOT THE LOCUS AND NO LAYER IS NAMED. `--arena-refresh` (a full-width device read of the just-written bounce arena, enqueued before the copy) passes the acceptance protocol at 1-3% cost — 7,168 tokens, zero events, 95% residual bound 1 per 2,389 against a pre-fix 1 per 299-578 — but `--direct-vmm-dma`, which has NO arena and NO H2D copy, still diverges (91/512 @420, 2026-08-18), so any account that is purely "the copy read stale arena bytes" is REFUTED AS A COMPLETE ONE. Sixteen mechanisms are eliminated by measurement and five candidate layers remain, INCLUDING OUR OWN MISSING SYNCHRONISATION — "hardware bug" is a hypothesis, not a finding, and the micro-repro is clean at 1e6 reads, which points toward the engine's concurrency rather than away from it. The surviving rule that fits every cell is narrow: the repair must read THE REGION THE NVMe DMA'd INTO, at full density, BEFORE the next device agent consumes it — reading the destination afterwards is RED (`sc` @236, `se` @301), equal bulk from a decoy is RED (@12), the bare dispatch is RED (@292), a shader copy instead of SDMA is RED, and every arena memory type is RED with the allocation read back. That rule predicts the one untested cell, DIRECT + a full-width slot read, which unifies the evidence if clean and kills the theory if red. ONLY GLM IS ESTABLISHED AFFECTED — Glimmer is a genuine control (dense, byte-identical pinned AND at 21/52 streamed), while V4 (32 ids) and K3 (16 ids) are UNTESTED at any length where the defect is detectable, and since the rate is PER READ, K3 streaming ~92% of 1.3 TiB should be the worst arm of the four.
---

# GLM decode nondeterminism — closeout

**What it is.** Two runs of one binary with byte-identical arguments produce different token
ids. Measured on GLM-5.2 int3-vq and int4, in both the pre-rewrite and post-rewrite trees, at a
rate of roughly **1 event per 299–578 token-forwards** — and the rate is per **read**, not per
token. One event rewrites the whole tail: 496 of 512 ids differed after a divergence at
position 13.

**Where it stands.** Mitigated and gated by `--arena-refresh`; **root cause unknown**; the
structure everyone assumed was the locus has been eliminated. The working record with every
intermediate arm is `glm-bug.md`; this document is the closeout.

---

## 1. The conclusion, stated precisely

The natural summary — *"an unknown hardware bug that forces us to read through recently loaded
bytes to guarantee they are correct"* — contains three claims. **Two hold. One does not.**

### "Read through recently loaded bytes" — holds, but far more narrowly than it sounds

Reading recently-loaded bytes does **not** generically repair anything:

- `sc` reads the **entire destination slot** at full width immediately after the copy — **RED @236**.
- `se` reads **every slot** at end of layer, nine times the bulk — **RED @301**.

The only read that has ever suppressed the defect is of **the region the NVMe DMA'd into,
at full density, before any other device agent consumes it.** Everything else in the matrix
below is red.

### "Unknown" — holds

No mechanism is named. The one theory this investigation committed to in writing — that the
GPU-side reader holds stale lines of the arena and a kernel copy would refill them through the
normal vector path an SDMA engine bypasses — **was built and tested. `--copy-by-kernel` is
RED.** The prediction failed and the theory did not survive it.

### "Hardware bug" — NOT established, and it should be adopted last

Five layers could carry this defect: silicon (gfx1151, LPDDR5, IOMMU, SDMA), firmware/driver
(amdgpu, KFD), the HIP runtime, the kernel's io_uring/O_DIRECT completion-vs-visibility
contract, and **our own missing synchronisation**. Nothing has excluded the fifth.

Two facts pull against the hardware reading:

1. **The micro-repro is CLEAN at 1e6 reads** — 0 mismatches, 4–8× cleaner than the engine. The
   hazard needs the engine's own concurrency, which is evidence *toward* our code, not away.
2. **Only one of four arms shows it.** A general "freshly-DMA'd bytes are unreliable" property
   of the silicon would not spare a dense arm that streams 21 of 52 layers.

Calling it hardware is the least falsifiable and least actionable of the five and buys nothing
until a **firing micro-repro** exists. That repro is the only exhibit AMD would act on, and it
is the thing that would promote the hypothesis to a finding.

### The rule that fits every cell — and the one experiment that tests it

> A device-side reader — the SDMA copy engine **or** a compute kernel — can read **stale** bytes
> from a region the NVMe has just DMA'd into. A prior full-width read **by a compute kernel**
> repairs that region for the next consumer. Reading the destination afterwards is too late:
> the wrong bytes are already there.

In **bounce** mode the DMA target is the arena and the next consumer is the SDMA copy, so the
pre-copy arena read (`--arena-refresh`) repairs it. In **direct** mode the DMA target is the pool
slot and the next consumer is the MoE kernel — **and no such prior read exists**, which is
exactly why direct diverged.

**The untested cell that decides this: DIRECT + a full-width read of the pool slot**, enqueued
after the read completes and before the ticket signal. **Clean ⇒ the rule unifies every arm and
names the defect's shape. Red ⇒ the rule is dead.** Either way it is decisive.

---

## 2. Everything tried

Every arm below was **read back and confirmed to have applied**, after two rounds were lost to
interventions that never took effect (an intervention that did not apply and one that does not
work produce the same red).

### Fix candidates

| # | intervention | reads the DMA'd region? | density | result |
|---|---|---|---|---|
| 1 | **`bh` / `--arena-refresh`** — full-width arena read, pre-copy | **yes** | **bulk** | **CLEAN** |
| 2 | `bh-line:32` — one dword per 128 B across the whole region | yes | 1/32 | RED @704 — the **latest** red of the investigation, i.e. a partial mitigation |
| 3 | `bh-decoy` — same bytes, same duration, different buffer | no | bulk | RED @12 — the earliest |
| 4 | `bh-nop` — the launch and its dispatch acquire, no read | no | none | RED @292 |
| 5 | `sc` — the whole destination slot, after the copy | no | bulk | RED @236 |
| 6 | `sc-nop` — same launch, ~no work | no | none | RED @292 |
| 7 | `se` — every slot at end of layer | no | bulk×9 | RED @301 |
| 8 | `xa,ac` — compute-side folds | no | — | RED @292 |
| 9 | `--pinned-coherent` — fine-grained arena, readback `0x40000000` | — | — | RED 225/512 |
| 10 | write-combined arena, readback `0x4` | — | — | RED |
| 11 | `--copy-by-kernel` — shader copy instead of SDMA, counts which path ran | — | — | RED |
| 12 | v2 light probe (column hashes only) | no | — | RED @169 |
| 13 | **`--direct-vmm-dma`** — no arena, no copy at all | n/a | none | **RED 91/512 @420** |
| 14 | CPU-side store fence | — | — | refuted by argument: the CPU is neither producer nor consumer on this path |
| 15 | stride sweep 16 → 8 → 4 → 1 | yes | — | **started, never completed** |
| 16 | **DIRECT + full-width slot read** | **yes** | **bulk** | **the open cell** |

**What the shape demands of any theory**: bulk (stride-32 sampling insufficient),
region-specific (equal bulk elsewhere insufficient), not a dispatch effect, not a copy-engine
effect, and not curable by any arena memory type.

### Instruments built

`tests/determinism-glm.sh` (power-sized at a 512 floor, prompt pinned by length and md5, with a
deviceless `--self-test` red-proof) · `tests/nll-divergence.sh --power` (Poisson intervals,
rule of three) · `tests/divergence-columns.sh` · the per-layer divergence log with device-side
folds at three stream positions (`bh` pre-copy, `sc` post-copy, `se` end-of-layer) ·
`arena_repro` (isolated micro-repro; clean at 1e6) · `docs/measurement/probes/fetch_dest.hip`
(destination bandwidth, self-policing against page-cache fallback) · a guarded id comparator
that refuses unless both arms exist and carry the expected count · and the **`slot_stalls`**
counter, which is INV-9's own falsifier and **had never been read by anything** before this
investigation. It reports 0 on diverging pairs.

---

## 3. Ruled out, with the evidence that rules it out

| ruled out | by |
|---|---|
| Speculative decode / MTP | the outlier run had `--mtp` **off**; the `--mtp` run agreed with a clean run 0/32 |
| A rewrite regression | the pinned pre-rewrite binary diverges from itself **worse** — 247/512 @265 vs 61/512 @452 |
| GPU kernels, reduction order, the scoring harness | Glimmer teacher-forced twice fully pinned: byte-identical, PPL 7.008490 both runs, 761 positions |
| Layer streaming as a generic mechanism | Glimmer at 52/52 pinned vs **31 pinned / 21 streamed**: byte-identical |
| MoE accumulation | verified at the kernel: clamped terms, wrapping `u64` atomicAdd bounded at 2^62, fixed-stride drain — the lane split is exactly equivalent to one accumulator |
| Float non-associativity | **there is no float atomic in any kernel** |
| Arena relocation racing an in-flight read | by construction (unconditional end-of-layer sync, ticketed miss kernels, relocation only before any read resolves) **and** by observation: `relocs = 0` on the event row |
| Read-before-write | the unconditional detector has **never fired**, in any run, including every diverging one |
| Attention and anything upstream of the MoE | `xn` identical at the event |
| Residency selecting arithmetic | `gl`, `pk`, `sl` identical, and single-format `int3-vq` cannot vary format |
| A structural boundary at the first divergence | 236 sits mid-plateau; the MLA split plan cuts at nr=193 and nr=241 |
| A systematic `--max-mem` bias | budget CI [-0.00792,+0.00119] and control CI [-0.01017,+0.00167] both contain zero over 762 positions |
| Contention as a **cause** | it amplifies (452 quiet → 13 under load) but the quiet box still diverges |
| Anything vq3-specific | **`--mode int4` diverges too** — different stride (19.125 vs 14.625 MiB), different kernel, no codebooks. 95,351 misses vs 70,030, and it diverged at token **28** rather than 320–452, which is what established the **per-read** rate |
| The staging hand-out | `slot_stalls = 0` on diverging pairs |
| Host→device visibility of the arena | a fine-grained arena, **read back** as `coherent-bit true`, still RED |
| The dispatch acquire | `bh-nop` RED @292 |
| Bandwidth or delay | `bh-decoy` RED at equal bytes and equal duration |
| SDMA specifically | `--copy-by-kernel` RED, with a counter proving which path ran |
| **The arena hop being NECESSARY** | **`--direct-vmm-dma` RED, 2026-08-18** |
| Storage and transport | btrfs datasum verifies the direct-IO reads; the kernel log is clean of mce/kfd errors (owner-audited) |

### Explicitly NOT ruled out

- **Our own missing synchronisation.** Nothing has excluded it, and the clean micro-repro is
  consistent with it.
- **The toolchain.** HIP 7.14.60850 / clang 23.1 is the only installed toolchain, so both
  binaries in the old-tree comparison were built under it. That experiment varied **source** and
  held **toolchain fixed**: it shows the upgrade did not *retire* the defect and says nothing
  about whether it introduced one. Testing it needs the old source under the old toolchain,
  which is not installed here.
- **Driver, firmware, and silicon**, individually.
- **Which hop corrupts** — the NVMe write, the consumer's read, or something between.

---

## 4. Which models are affected

| model | status | basis |
|---|---|---|
| **GLM-5.2** (744B/40B MoE) | **AFFECTED — established** | 512-token pairs, both trees, **both formats**, rate 1 per 299–578 per read |
| **Muse Glimmer-30B** (dense) | **not observed — and a genuine control** | byte-identical fully pinned (0 streamed) **and** at 21/52 streamed. No routed pool, no per-expert admission, layer-wise deterministic reads |
| **DeepSeek-V4-Flash** | **UNTESTED — do not read as clean** | byte-identical over **32 ids only**, and **GLM passes at 32 too**. Its 97% hit rate means ~⅛ the reads: lower *exposure*, not immunity |
| **Kimi-K3** (2.8T MoE) | **UNTESTED** | the first decode was 16 tokens |

**The exposure model makes a prediction.** The rate is per **read**. That is what the int4 arm
established: +36% misses moved the first divergence from token 320–452 to token **28**.
**K3 streams ~92% of a 1.3 TiB model — the highest read volume of any arm — so if the defect is
in the shared fetch path, K3 should be the worst affected of the four.** It is also the only arm
with no determinism measurement at all.

**Do not restate "Glimmer and V4 are clean."** Glimmer is a control that never exercises the
routed-expert path; V4 has been measured only at a length where the defect is known to hide.
Neither is evidence of immunity, and treating a GLM-scoped closure as engine-wide has already
gone wrong once in this repo.

---

## 5. The mitigation's operating envelope

Someone will inherit `--arena-refresh` without reading this investigation. This section is for
them.

### What it does and what it costs

A full-width device read of the just-written bounce-arena window, enqueued on the fetch stream
**before** the copy, value discarded (`kernels/async.hip::touch_arena_kernel`). It is a plain
CLI flag and deliberately **not** behind a feature, so the acceptance protocol compares two
release binaries differing in exactly one argument.

**1–3% of decode throughput**: 2.65 tok/s at 1536 tokens against a 2.70–2.73 control, one
artifact and one prompt. The earlier "~10%" figure was the *probe* fold, which also hashed and
wrote a 19 MB log per run; the bare read is nearly free.

### Why it is nearly free today, and why that will change

The refresh adds **~2.25 GB/token** of device reads — a second pass over every byte fetched.
It costs 1–3% only because **the engine is disk-bound**: at 78% hit rate a token moves 2.25 GB
at ~12 GB/s, ~181 ms of transfer against ~117 ms of compute, so the extra device read hides
inside the NVMe wait.

**That is a property of the current bottleneck, not of the fix.** Anything that removes the disk
bottleneck — a faster array, a higher hit rate, a smaller expert format, the hybrid `FormatPlan`
— makes this cost *visible*. Re-measure it after any change that moves bytes/token, and do not
carry the 1–3% forward as a constant.

### The ceiling, and what watches it

The mechanism has no name, so **nothing guarantees the read stays sufficient.** A future
compiler, driver, cache geometry or scheduler change could stop making a full-width read repair
the region, and the fix would silently become a 1–3% tax that fixes nothing.

**The only thing that would notice is `tests/determinism-glm.sh`.** No unit test, no invariant,
no CI job covers this — CI has no `rocm` arm and no GPU arm at all. Concretely:

- **Do not delete or weaken `--arena-refresh` without re-running that gate**, at its 512 floor,
  with a same-day control.
- **A single green pair is not a live gate.** Two same-day stock controls came back GREEN on
  2026-08-18 on an unchanged machine — a 0.1–2.9% event under the measured rate. A **red** arm
  needs no control; a **green** one is uninterpretable without one.
- Any arm that claims the flag applied must show the log line, not the intent.

### Known limits of the claim

- Measured on **one artifact, one prompt, one budget** (`--max-mem 115`, int3-vq, dense).
- 7,168 tokens of exposure bounds the residual rate at **1 per 2,389** (95%, rule of three). That
  is a **≥4–8× reduction as a floor**, not a proof of zero.
- It is **not** a fix for `arena-relocation-vs-in-flight-reads`, a separate recorded defect
  (9/8452 corrupted reads at a tight budget) in the pool's compaction path.
- The other three arms have never been run under this gate.

### The open decision

The flag is **opt-in**. Correctness argues for on-by-default and the price to quote is 1–3%.
That is the owner's call and it is not made here. Whichever way it goes, the recorded command
line in `docs/measurement/` must carry the flag's state, because every quality and throughput
number in this repo is now conditional on it.
