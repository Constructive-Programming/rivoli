---
status: live
scope: glm
verdict: The frozen WORKING RECORD of the GLM nondeterminism hunt (was root-level `glm-bug.md`, vendored 2026-08-20) — every intermediate arm, the rate model, the priced fix candidates and the corrections ledger, current to 2026-08-18. SUPERSEDED for conclusions by glm-nondeterminism-closeout.md; read this only for how an arm was run.
---

# The GLM decode nondeterminism bug

> **VENDORED 2026-08-20.** This is the working record the closeout cites as `glm-bug.md` —
> it lived untracked at the primary repo root, one `rm` from gone, while committed docs cited
> it. Frozen at its 2026-08-18 state; §6's arms and the `--copy-via-cpu` fix candidate are in
> the closeout only.

**Status as of 2026-08-18. MITIGATED AND GATED; ROOT CAUSE UNKNOWN.** `--arena-refresh` passes
the full acceptance protocol at a ~1-3% cost (§14). Sixteen mechanisms are eliminated by
measurement and none of the survivors explains why the repair must be both *bulk* and
*arena-specific*, so the mitigation ships labelled as one. Owner-designated **hard blocking
requirement** for the M10–M19 feature wave.

Working branch: `wave/fix-glm-determinism` (6 commits). Evidence branch: `wave/m10-spine`
(`docs/investigations/glm-nondeterminism.md`). Vendored raw evidence:
`docs/measurement/glm-divergence-evidence/{a,a2,b,ref}.nll`.

---

## 1. The bug, in one paragraph

**GLM-5.2 running `--mode int3-vq` produces different text on repeated, byte-identical
invocations.** The engine is fully deterministic — same routing, same residency decisions,
same arithmetic — until a rare event delivers **wrong bytes for one expert's weights**. The
run does not crash, warn, or produce garbage: it computes a perfectly valid forward pass on
slightly wrong weights, one argmax flips, and because greedy decode feeds its own output
back, every subsequent token is rewritten. The event is a **rate**, not a threshold: roughly
one per 300–600 generated tokens, so a 32-token run usually looks clean and a 512-token run
usually does not. It affects both the current tree and the pre-rewrite tree, and it is
invisible to every existing gate because those gates were written short.

The failing quantity is the **payload hash of expert weights read out of a pool slot**, with
the slot identity, the routing that chose it, and the attention output feeding it all
provably identical between runs.

---

## 2. Symptom and reproduction

Artifact `/var/db/rivoli/glm52-vq3-full`, prompt pinned at 317 bytes
(md5 `18927a780b36b029d03450d2100e9242`, recorded verbatim in `tests/determinism-glm.sh`).

```
rivoli <artifact> --bench 512 --mode int3-vq --attn dense --max-mem 115 --prompt <P> --dump-ids <out>
```

Run twice, `cmp` the id bodies.

| pair | conditions | result |
|---|---|---|
| rewrite × 2 | box loaded (1.4 TiB convert running) | **496 of 512 ids differ**, first at position **13** |
| rewrite × 2 | quiet box | **61 of 512 differ**, first at position **452** |
| rewrite × 2 | quiet box, gate run | **RED**, first at position **320** |
| **old tree** × 2 (`ref-pin` @ 6b7f496, `--no-mtp`) | quiet box | **247 of 512 differ**, first at position **265** |
| rewrite × 2 | 32 tokens | **byte-identical** |
| rewrite, `--mtp` vs no-`--mtp` | 32 tokens | **byte-identical** |

Teacher-forced scoring over a 762-token corpus, identical flags, run twice:
**555 of 762 positions differ**; PPL 5.200080 vs 5.209284; hit rate differs run to run
(78.2643% vs 78.2352%).

**Two magnitudes, one defect.** Teacher forcing re-anchors every position to the committed
corpus, so a perturbation cannot change the input sequence and propagates only numerically
through KV — a small dNLL wobble. Greedy decode feeds output back, so one flipped argmax
rewrites the tail. Hence "0.0018 nats" and "496 of 512 ids" describe the same events.

---

## 3. The decisive coordinate

Both arms run with `--features corruption-probe --divergence-log`, quiet box, KFD witness 0
before and after. Compared with `tests/divergence-columns.sh`. First differing row:

```
row 12427   pos=164  nrow=1  layer=24                      misses=1  relocs=0
  A   xn 7e4e946650f24190   h 9c7e5614ddb784fc   x 5f7ac21d78086fba
      gl 18d6d2d88f7ef0f8   pk 7384d5fcc94ff963  sl ea11f614bd198904
  B   xn 7e4e946650f24190   h bb1a15e399e82fbf   x c4d30b6a9ae85370
      gl 18d6d2d88f7ef0f8   pk 7384d5fcc94ff963  sl ea11f614bd198904
```

| column | meaning | verdict |
|---|---|---|
| `xn` | attention output entering the MoE | **identical** |
| `gl` | router logits | **identical** |
| `pk` | expert picks | **identical** |
| `sl` | slot assignment | **identical** |
| `h` | **hash of the expert weight bytes read from those slots** | **DIFFERS** |
| `x` | MoE output | differs, as a consequence of `h` |
| `relocs` | arena relocations during the pass | **0** |
| `misses` | fetches landing on this layer | 1 |

**Reading**: the same experts were selected, by the same routing, from the same attention
input, and placed in the same slots — and the bytes read out of those slots differed. The
corruption is upstream of all engine logic and downstream of nothing the engine decides.

---

## 3b. The second coordinate — and why it reframes the first (2026-08-17, Phase 0)

**Phase 0 control is VALID.** On the same box, same day, same pinned prompt: the **unprobed**
pair **diverged at body line 66**, and the **v2 light-probe** pair **diverged at body line
169**. So the unmodified engine still diverges today, and the *light* probe does not suppress
— which isolates the suppressor to the v3 hop folds (§7).

The light pair's first differing row, all eleven columns compared:

```
row 11814   pos=186  nrow=1  layer=35   misses=3  relocs=1
  A   xn bd7368d84ea4b46f   h b7d6b00bf231b134   x 568ecb4fe448726f
  B   xn bd7368d84ea4b46f   h b7d6b00bf231b134   x 327c32c8d8c34960
      gl b545f4899bbd9bde   pk c2d211a6885ec420  sl 27e82e49106795e3   (all identical)
```

**Only `x` differs. `h` is IDENTICAL.** That is the opposite signature to §3, where `h`
differed and `x` followed.

**The parsimonious reading is that `h` measures at a different instant than the consumer
does.** The probe folds the slot's bytes at one moment; the kernel reads them at another. If
the corruption lands between the fold and the read — or if the kernel reads a different
address after a relocation — then `h` can match while the kernel consumed different bytes.
The alternative, that the MoE arithmetic is itself nondeterministic on identical inputs, is
refuted at the kernel (§4.5, §4.6: fixed-point accumulation, lane split exactly equivalent to
one accumulator, no float atomics anywhere).

**Consequence, and it is a correction to how §3 was read:** `h` **differing is evidence of
wrong bytes; `h` matching is NOT evidence of correct bytes.** A single mechanism — wrong bytes
at kernel-read time — produces both signatures depending on whether the probe's fold happens
to straddle the corruption. Any ablation cell that reports "`h` same" therefore **cannot
exonerate a hop**, and the matrix in §11b must be read with that asymmetry in mind.

**Relocation is neither necessary nor sufficient.** `relocs > 0` occurs in **14,338 of 42,666
rows (33.6%)** of an ordinary run — routine, not remarkable — and the §3 event fired at
`relocs = 0`. The §3b event fired at `relocs = 1`. Neither correlation carries weight on its
own.

## 4. Ruled out (with the evidence that rules it out)

1. **Speculative decode / MTP.** The outlier run in a three-way comparison had `--mtp` OFF;
   the `--mtp` run agreed with a clean no-MTP run at 0 of 32 positions. MTP was nearly
   convicted by a gate comparing against a baseline that itself moves.
2. **A rewrite regression.** The pinned pre-rewrite binary diverges from itself **worse**
   (247/512, first at 265) than the rewrite (61/512, first at 452) on the same box, artifact
   and flags.
3. **GPU kernels, reduction order, and the scoring harness.** Muse Glimmer teacher-forced
   twice, fully pinned (`52 of 52 layers pinned, 0 streamed`): **byte-identical**, PPL
   7.008490 both runs, 761 positions.
4. **Layer streaming as a generic mechanism.** Glimmer ids at `--max-mem 32` (52/52 pinned)
   vs `--max-mem 20` (**31 pinned, 21 streamed through 1 slot**): **byte-identical**. The
   partition, slot refill and write-after-read fence are output-neutral under real eviction.
5. **MoE accumulation.** `moe_fixed` clamps per term; the accumulator is a wrapping `u64`
   `atomicAdd` bounded at 2^62; the drain sums both lanes at a fixed stride — so the
   resident/miss lane split is *exactly equivalent* to a single accumulator. Verified at the
   kernel, not assumed from the design note.
6. **Float non-associativity.** There is **no float atomic in any kernel**. Identical input
   bytes give identical outputs.
7. **Arena relocation racing an in-flight read** — the standing hypothesis for this repo's
   #1 open defect, refuted twice over. By construction: `run_layer` ends every layer with an
   unconditional `hipDeviceSynchronize`, every miss kernel waits on its ticket, `launch_moe`
   host-awaits both lanes, and `submit` relocates only before it resolves any read — the
   device is idle when the arena compacts. By observation: **`relocs = 0` on the event row.**
8. **Read-before-write.** The unconditional `READ-BEFORE-WRITE` detector has **never fired**
   in any run, including every run that diverged.
9. **Attention, and anything upstream of the MoE.** `xn` identical at the event.
10. **Residency selecting arithmetic** (the old hybrid-mode defect class). `gl`, `pk` and
    `sl` all identical; and this is single-format `int3-vq`, where format cannot vary.
11. **A structural boundary at the first divergence.** Position 236 looked like one; the MLA
    split plan cuts at nr=193 and nr=241, so 236 sits mid-plateau. Split-KV exonerated.
12. **A systematic `--max-mem` effect.** The re-specified `p4` gate is GREEN: budget CI
    `[-0.00792, +0.00119]` and control CI `[-0.01017, +0.00167]` both contain zero over 762
    scored positions (noise half-width 0.00592 nats). The wobble is not a bias toward staler
    or wrong weights.
13. **Contention as a cause.** It amplifies — first divergence moves from position 452
    (quiet) to 13 (heavy CPU/NFS load) — but the quiet box still diverges.
14. **Anything format-specific to vq3** — the codebook indirection, the vq3 slab geometry,
    the 14.625 MiB stride, `launch_moe_expert_range`. **`--mode int4` diverges too**: a
    512-token unprobed pair on the same artifact, prompt and budget **diverged at body line
    28**. Different format, different stride (19.125 MiB), different kernel
    (`launch_moe_expert_range_i4`), no codebooks — same defect. Until this run, every arm in
    the investigation had been `int3-vq` against itself.

    **And it strengthens the per-read rate model.** int4 experts are ~31% larger, so at the
    same `--max-mem` fewer fit resident: **95,351 misses** against int3-vq's **70,030** over
    the same 512 tokens (+36%). It diverged at token **28** rather than 320–452. More reads
    per token, earlier event — consistent with a rate per *read*, not per token.

---

## 5. NOT ruled out

- **Which of three hops corrupts**: the NVMe read into the pinned bounce arena, the
  `hipMemcpyAsync` from bounce arena into the GTT pool slot, or something touching the slot
  after the copy. This is the open question and the hop-splitter was built to answer it.
- **The toolchain.** HIP 7.14.60850 / clang 23.1 is the only installed toolchain, so *both*
  binaries in the old-tree comparison were built under it. That experiment varied **source**
  and held **toolchain fixed**: it shows the upgrade did not *retire* the defect and says
  **nothing** about whether it introduced it. Testing that needs the old source under the old
  toolchain, which is not installed here.
- **Whether the instrument suppresses the event.** See §7.

---

## 6. The rate model

Derived from per-run first-divergence positions in the retained `.nll` arms: **236, 362, 375,
and one arm clean through 762**. All four arms — two budgets, two source trees — are
**bit-identical on positions 0..235**.

- **1 event per 299 tokens** (matched pair) / **1 per 578** (conservative, keeping the clean
  arm's exposure). Exact 95% Poisson interval **1 per [206, 2804]**.
- `1 − exp(−2n/rate)` predicts observations it was not fitted on: **10–19% divergence
  probability at 32 tokens, 83–97% at 512**.
- **There is no safe length.** The 32-vs-512 contrast needs no extra mechanism; a clean
  32-token run is a coin flip that landed, not a property.
- Magnitude *at* the event is **1.2e-2 nats** — far too large for a rounding difference — and
  the wake grows rather than decays (median |dNLL| 2.8e-3 → 3.8e-2).
- **Rank by first-diverging position, never by differing count.** 496 of 512 is *at least
  one* event plus a cascade; later events are invisible once the sequences part.

---

## 7. In flight: the hop-splitter

Three cross-run folds bracket the path: `bh` (bounce arena after the NVMe read), `sc` (pool
slot after the copy), `se` (all the layer's slots at end of layer, each at its own index
offset). Decision rule:

- **`bh` differs** → the bytes arrived wrong from storage.
- **`bh` equal, `sc` differs** → the copy corrupts, and **verify-on-read is the wrong fix** —
  it would detect corruption the engine itself caused and re-read through the same path.
- **All three equal, `h` still differs** → a ticket/timeline ordering failure; cheap to fix,
  but it invalidates INV-4/INV-5's claims.

**Result: the heavy probe suppresses the event.** Two instrumented pairs, **2,048 tokens of
exposure, zero events** — a 512-token pair (identical ids, identical hit/miss 236,570/70,030,
all 42,666 log rows equal) and a 1536-token pair (identical ids, identical hit/miss
714,436/206,564, all 122,538 rows equal). P(zero events in 2,048 tokens) = **0.11% (matched
rate) to 2.89% (conservative)**.

**The variable is the probe's weight, not the presence of a probe.** Four cells:

| prompt | probe | result |
|---|---|---|
| default | **light** (column hashes only) | **DIVERGED** — token 164, the §3 coordinate |
| essay | **none** | **DIVERGED** — 61/512 at 452; gate RED at 320 |
| essay | **heavy** (+ `bh`/`sc`/`se`) | **CLEAN** — 512 tokens |
| essay | **heavy** | **CLEAN** — 1536 tokens |

The light probe (which produced the coordinate) did not suppress it; the heavy probe does.
The difference is ~14.4 GB/token of extra reads, one of which **reads the pool slot
immediately after the copy**.

**Leading hypothesis, from the suppression itself: a device-side visibility problem on the
copy.** The compute kernel occasionally reads a slot whose `hipMemcpyAsync` has been issued
and signalled but whose bytes are not yet coherently visible; any additional read that forces
ordering or a flush masks it. This fits every other observation — host bookkeeping correct,
`READ-BEFORE-WRITE` never firing, tickets and waits all in place, `relocs = 0` at the event,
routing and slot identity identical while only the payload differs.

**Consequence: the hop-splitter as built cannot catch its own quarry.** That is a finding
about the instrument, not a failure of it. The next instrument must observe without adding a
read on the fetch path — e.g. hashing *after* the compute kernel has already consumed the
slot, or a passive host-side check of the bounce buffer only.

Measured probe cost: **1.65–1.70 tok/s against 2.42–2.60 uninstrumented ≈ 33%**, close to the
agent's ~27% estimate (~14.4 GB/token of extra hashing).

A trap avoided in review: the first fold kernel did one `atomicXor` per element against a
**single global u64** — 3.6e9 same-address atomics per token, serialising to **1–4 s/token
against a 388 ms budget**. It would have dilated its own subject ~10×. Now reduced in shared
memory, one atomic per block, **3744× fewer**, still exact (XOR is associative).

---

## 7b. The completed ablation matrix (2026-08-18) — what suppresses, and what that costs the hypotheses

Every cell: 1536-token pairs, quiet box, pinned prompt (md5 `18927a780b36b029d03450d2100e9242`),
one binary vintage per protocol run, verdicts through a comparator that refuses on missing or
short arms.

| intervention | touches arena? | how much | outcome |
|---|---|---|---|
| v2 light (no fetch folds) | no | — | **RED** @169 |
| `sc` — reads the whole slot after the copy | no | bulk | **RED** @236 |
| `sc-nop` — same launch, ~no work | no | none | **RED** @292 |
| `se` — reads every slot at end of layer | no | bulk×9 | **RED** @301 |
| `xa,ac` — compute-side folds | no | — | **RED** @292 |
| `bh-nop` — the launch and its dispatch acquire, no read | no | none | **RED** @292 |
| `bh-line` (= `bh-line:32`) — one dword every 128 B **across the whole region**, ~1/32 of the bytes | **yes** | 1/32 | **RED** @704 |
| `bh-decoy` — same bytes, same duration, NOT the arena | no | bulk | **RED** @12 |
| **`bh` — the full fold of the just-written arena region** | **yes** | **bulk** | **CLEAN** |
| `bh,sc,se` | yes | bulk | **CLEAN** (512 and 1536) |
| `--pinned-coherent` — fine-grained arena, **read-back verified** | — | — | **RED** (225/512) |

**Suppression requires the read to be of the arena AND at full density.** The same bulk moved
from a decoy buffer does not do it; the dispatch alone does not do it; reading the destination
slots does not do it; and **sampling the arena at stride 32 does not do it**.

**CORRECTED 2026-08-18 — a mislabel of mine, and the inference rested on it.** This section
first recorded `bh-line` as reading "one cache line" and concluded "one line insufficient, all
lines sufficient ⇒ a per-line effect". That is wrong: `bh-line` is `LINE_F32 = 32` f32, i.e.
one dword every **128 B across the entire region**, so it already touches *every* line at
~1/32 of the bytes. Both arms touch every line. What the data actually shows is a
**dose-response in bytes read** — stride-32 sampling insufficient, stride-1 sufficient — with
the granularity **unknown**. Consistent with that: `bh-line` is the **latest** red in the whole
investigation (704 against a ~292 median), i.e. a partial mitigation rather than a null.

A stride sweep (32 → 16 → 8 → 4 → 1, with `bh-line:1` as the known-clean positive control) is
the cheapest decisive experiment and is **running**. Its first clean N names both the
granularity and the price. One concrete prediction worth recording before the result: if a
single dword touch pulls only the containing **64 B sector** rather than both halves of a
128 B line, stride 32 repairs half of each line and **stride 16 should be clean at 1/16 of the
bytes** — which would be a genuine *fix*, where `bh` is only a probe at ~10% of throughput.

**`--pinned-coherent` is refuted, and this time the intervention was observed to apply.** The
engine now reads the allocation back: `requested hipHostMallocDefault | returned flags 0x0
coherent-bit false` versus `requested hipHostMallocCoherent | returned flags 0x40000000
coherent-bit true`. So `hipHostMallocCoherent` is not a no-op on ROCm 7.14, the arena really
was fine-grained, and the divergence was unchanged. **Host→device visibility of the arena is
not the mechanism.**

**Therefore `bh` is not a fix candidate — it is a Heisenberg probe.** It costs ~10% of
throughput and masks the defect; a mitigation that works by spending bandwidth on a hash
nobody checks is a coincidence with a price, not a repair.

**What the shape now demands of any theory**: a per-line effect (one line is not enough, all
lines are) that is specific to the arena's own pages (a decoy of equal size is not enough),
that fine-grained allocation does not cure, and that a dispatch acquire does not cure. A
plausible frame worth testing: the GPU-side reader holds **stale lines of the arena region
from a previous fetch through that same region**, and a full device-side read refills them,
while one line refills only itself and a decoy refills none. That predicts **Phase 3B
(copy by kernel instead of SDMA)** is the decisive next cell — a kernel copy reads through the
normal cache path that the SDMA engine may bypass — and it is a shippable fix at roughly equal
bandwidth if it holds.

## 8. Impact on the project

- **MTP's losslessness gate is unprovable at 512 tokens** — not failed, unprovable, because
  the baseline does not reproduce itself. It passes byte-identically at 32. MTP's economics
  are unaffected and good: **238/337 = 70.6% acceptance**, `p0.8+` bin 89% (n=195), draft cost
  3.4% of decode wall, against a 53% break-even.
- **Every GLM byte-identity gate needs a token count and an A-vs-A control arm** in the same
  cell. Without the control, the gate blames whatever feature is under test.
- **The old tree's byte-identity claims** — gated MTP at 1.108×, `parity-glm.sh`, the quality
  ladder's A/Bs — are **unproven at long lengths** rather than wrong.
- **Paired dNLL remains valid** on a fixed corpus, provided each comparison runs its own
  control arm. Noise floor: **0.035252 nats** derived from the retained arms. (An earlier
  "0.0018 nats" came from a *different* `p4` invocation — hit 78.2643/78.2352 vs
  78.2591/78.2597 — and was dropped for mismatched provenance.) Lag-1 autocorrelation is
  **negative** (−0.080, −0.009), so `bin/ppl`'s naive SE is conservative here and the feared
  ~3.9× interval inflation does not occur.
- **Unaffected arms**: Glimmer is byte-identical pinned *and* streaming; V4 was byte-identical
  over 32 ids at 97% hit. Their gates mean what they say.

---

## 9. Instruments built for this bug

| thing | where | what it does |
|---|---|---|
| `--divergence-log` | feature `corruption-probe` (`trace` deliberately does **not** imply it) | per-pass column hashes: `pos nrow layer xn h x gl pk sl misses relocs` |
| `hash_rows` + oracle, `xor_fold` | `crates/core` | the folds, with a deviceless test |
| `tests/determinism-glm.sh` | gate | two arms, identical args, ids compared. Prompt recorded **verbatim** and pinned by length+md5 on every invocation. Length floor 512, from the conservative rate needing 465 for 80% power |
| `tests/divergence-columns.sh` | comparator | takes column names from **each file's own header**, refuses logs whose headers disagree |
| `tests/nll-divergence.sh` | deviceless | position-wise `.nll` comparison, `--se`, `--power` |
| INV-9 | `architecture.md` §8b + test | renames the existing slot-reuse test rather than duplicating it |
| `RoutedGeom::check_reads_fit_their_slots` | `crates/engine` | replaces a `debug_assert!` that `--release` compiled out; the assert is **deleted**, not left alongside |

**Gate design lesson, generalised**: two arms that stop at *different* lengths have
diverged — that **is** the finding. An early version of the gate classified it as a setup
error. Lengths are now compared against each other first: differ → RED, equal-but-short →
setup error, equal-and-full → compare ids.

---

## 10. Candidate fixes, priced (none chosen)

Deliberately unchosen: picking now would price a mechanism not yet identified, and one branch
of §7 makes the obvious instrument the wrong one.

| rung | cost | note |
|---|---|---|
| verify-on-read, full width | **34% of throughput** with a SIMD hash | **impossible with scalar FNV**. 2.0 GB/token of expert bytes (130.4 misses × 15,335,424 B stride) |
| verify scale rows only | ~1.4% | arithmetic, not measured |
| 1-in-64 sampled verify | ~0.5% | arithmetic, not measured |
| global lock / barrier | — | **not on the ladder.** Fetch and compute are already ordered; a barrier would cost the streaming overlap and fix nothing |

Percentages are arithmetic against the only effective bandwidth this repo has measured
(~135 GB/s). What it actually costs depends on the reaper's io-wait slack, which `io_wait_ns`
already measures and nobody has read against this question.

---

## 11. What remains to be done

1. **Build an instrument that does not suppress its subject.** The hop-splitter is
   established as suppressing (§7). Options, cheapest first: drop the `se` fold (nine slots
   per layer, the expensive one) and re-test with `bh`/`sc` alone; hash the slot *after* the
   compute kernel has consumed it rather than before; or check the bounce buffer host-side
   only, adding nothing to the fetch path. Each needs its own suppression control — a clean
   run now proves nothing until the instrument is shown not to mask the event.
2. **Test the visibility hypothesis directly.** If the mechanism is DMA-completion
   visibility, an explicit acquire/flush between the copy's completion signal and the
   consuming kernel should fix it outright, at a cost far below the §10 ladder — and it is a
   cheap experiment even if it turns out to be wrong. This is the highest-value next step and
   it needs no new instrument.
3. **Split the corrupting hop** into storage / copy / post-copy, then choose a fix from §10
   accordingly.
3. **Test the toolchain hypothesis**, or record it as permanently untestable here.
4. **Re-run the determinism gate as a red proof** once a fix exists, and keep the current RED
   as the pre-fix baseline.
5. **Instrumented teacher-forcing.** The probe reaches greedy decode only on this branch, so
   an instrumented pair yields *one* coordinate. With `wave/m10-spine`'s scorer merged, both
   arms stay re-anchored and second and third events remain visible.
6. **Fix the K3 budget-floor accounting** found alongside this work: the engine states a floor
   ("needs 113,857,749,854 bytes before any weight is pinned"), then the pool sizes itself
   into everything above it, so a budget that clears the stated floor still OOMs on a later
   allocation (`--max-mem` 115 → OOM at 96 MiB; 110 → OOM at 144 MiB; **108 → decodes**).
7. **Re-run `parity-glm.sh` and `smoke-glm.sh`** for the GLM chain — both are owed, one from a
   coordinator error (an outer `flock` around a script that locks per cell: self-deadlock).

---

## 11b. The test plan, ordered by information per GPU-hour (owner, 2026-08-17)

**Phase 0 — free, do first.**
- **`dmesg`/journal audit** on this box (**needs root — owner only**): amdgpu/SDMA faults, KFD
  resets, NVMe AERs, btrfs checksum errors, MCE, around 08:5×, 13:0×, 15:0× and the int4 run.
  One command; might name the hop outright.
- **Re-establish today's rate (control)**: an unprobed 512 pair and a v2-light-probe pair.
  *Every suppression/ablation conclusion below rests on the unmodified engine still diverging
  today; a clean control voids all of it.* All binaries built **before any arm** — inter-arm
  builds poison page cache (measured 1.36 → 5.14 ms/miss). **[RUNNING]**

**Phase 1 — name the suppressing fold** (per-fold flags, feature+flag never env var; 1536-token pairs).
`F1: +bh only` — fires with `bh` differing ⇒ storage convicted with zero slot reads; clean at
power ⇒ the arena read is itself a suppressor, pointing at hop-1 timing.
`F2: +sc only` — the expected suppressor under the visibility hypothesis.
`F3: +se only` — the innocent control; runs after the consumer, so it should fire like F0.
**Whichever fold turns RED→GREEN is the mask, and its pipeline position names the mechanism.**

**Phase 2 — delay vs. read: what does `sc` physically repair?** Two variants at the same
pipeline position: (i) an **equal-duration spin kernel, no memory access** — suppression here
means the hazard is **pure time** (fixed-lag write visibility); (ii) a **one-cacheline read of
the slot** — suppression only here means a **cache/visibility repair by touching the bytes**.
(ii) costs ~0%, so it doubles as the first candidate fix.

**Phase 3 — stream/copy ablations.**
`A` miss kernels on the fetch stream (plain FIFO, no cross-stream wait) — clean at power
convicts the `hipStreamWaitValue64` edge, and is itself shippable if the overlap loss is
tolerable (measure it). `B` copy by kernel instead of SDMA (same stream, same timeline) —
clean convicts the SDMA path; still diverging exonerates the copy engine. `C` blocking
`hipMemcpy` (sledgehammer) — if even this diverges, the whole copy/wait model is exonerated
and suspicion returns to storage or at-rest corruption.

**Phase 4 — no-GPU, high-rate, run between GPU cells.**
- **Storage userspace repro**: same io_uring O_DIRECT path, same `hipHostMalloc` pinned arena,
  random (layer, expert) blocks read twice + `memcmp`. ~2–3M reads/hour against an
  engine-implied ~1/41k–82k per read ⇒ **20–80 events/hour if storage is the hop**; clean at
  1e8 bounds it below 3e-8. Zero GPU tenancy.
- **Micro-repro of the wait edge**: pinned src → GTT dst `hipMemcpyAsync` + `WriteValue64` on
  stream F, `WaitValue64` + verify on stream C, hot loop at pool-sized slots with L2 pressure.
  Turns a 20-min-per-arm hunt into millions of iterations/hour — and if it fires it is the
  **minimal driver repro to hand AMD**.
- **DRAM canary**: ~100 GiB pinned, splitmix pattern, continuous host re-verify. int4's
  per-read scaling already argues against flips; EDAC is empty on this box (no ECC telemetry).
- **Payload dump on event**: light probe plus a trigger that `copy_out_raw`s the union's slots
  on the first cross-run `h` difference. **The corruption's shape is decisive**: whole-slot
  plausible = old tenant / early read; prefix-new/suffix-old = partial copy; scattered
  cachelines = visibility mix; single word = flip.

**Resolution path (fixes already priced).** wait-edge/visibility → miss kernels on the fetch
stream (measured overlap loss) or the 1-line `sc`-lite read (~0%, mitigation only with the
micro-repro naming why). SDMA → copy kernel (≈ same bandwidth, measure). storage →
scale-row-only verify (~1.4%) plus the userspace exhibit. at-rest/flip → sampled verify
(~0.5%) as detection; the real fix is ECC hardware.

**Acceptance protocol for any "fixed" claim** — so nobody repeats the false green in §12:
same-day **RED control of the unpatched engine** at ≥512 with the pinned prompt; fix arm GREEN
at 512 ×2 **plus 4×1536-token pairs**; print the rate bound (`tests/nll-divergence.sh
--power`); record throughput cost; keep the current RED as the pre-fix baseline. Every cell
states **ngen, probe config and box load**, and no comparison ships without existence+length
guards.

**Highest-value next move**: Phase 0 control + Phase 1 F1/F2 (~2 GPU-hours). It either
convicts storage outright or names `sc` as the mask — which makes the visibility hazard the
working mechanism and unlocks Phase 2 as the decisive cheap experiment.

## 14. THE FIX: `--arena-refresh` (2026-08-18) — protocol PASSED

A full-width device read of the just-written bounce-arena window, enqueued on the fetch stream
**before** the copy, value discarded. `crates/backend/kernels/async.hip::touch_arena_kernel`,
plumbed as a plain CLI flag (deliberately **not** behind `corruption-probe`, so the protocol
compares two release binaries differing in exactly one argument). Commit `41f6f3d`.

### Acceptance protocol, met in full
| arm | result |
|---|---|
| same-day RED control, unpatched, 512 | **DIVERGED at id 491 of 512** |
| `--arena-refresh`, 512 (×2) | **IDENTICAL**, both |
| `--arena-refresh`, 1536 (×4) | **IDENTICAL**, all four |
| `slot_stalls` on every fix arm | **0** (hand-out was pure round-robin) |

**Exposure: 7,168 tokens, zero events.** 95% upper bound on the residual rate (rule of three)
is **1 per 2,389 tokens**, against a measured pre-fix rate of **1 per 299–578** — a **≥4–8×
reduction as a floor**, and P(all six pairs clean | pre-fix rate) = **3.9e-11 to 4.1e-6**.

**Cost: ~1–3%.** 2.65 tok/s at 1536 tokens against 2.70–2.73 for the control. An order of
magnitude cheaper than the ~10% the probe fold suggested, because that fold also hashed and
wrote a 19 MB log; the bare read is nearly free. **The 34% verify-on-read rung in §10 was never
needed.**

### It is a mitigation, and the ceiling is recorded
The mechanism has no name. What must be true of any theory: the repair is **bulk** (stride-32
sampling is insufficient) and **arena-specific** (the same bulk from a decoy buffer is
insufficient), while the arena's *memory type* is irrelevant (default, fine-grained and
write-combined all diverge, each read back and confirmed applied) and the *dispatch* is
irrelevant (`bh-nop` diverges). **A future compiler, driver or cache geometry could stop making
the read sufficient, and nothing would notice except `tests/determinism-glm.sh`.** Do not delete
it without re-running that gate.

### What remains open
1. **Root cause.** Phase 3A (misses FIFO on the fetch stream, convicting the
   `hipStreamWaitValue64` edge) and 3C (blocking copy) are unbuilt and are the last two
   ablations on the list.
2. **A firing micro-repro.** `arena_repro` is clean at 1e6 reads because it lacks the engine's
   concurrency; adding concurrent compute and arena reuse is what would make it fire, and a
   firing repro is the only exhibit AMD would act on. It now takes `--arena-refresh` so a
   candidate can be re-run there without the engine.
3. **Whether the default should flip.** The flag is opt-in. Correctness argues for on-by-default;
   that is the owner's call, and the cost to quote is 1-3%, measured on one artifact and one
   prompt.
4. **The other three arms.** V4, K3 and Glimmer share this fetch path. Glimmer was byte-identical
   pinned *and* streaming, V4 byte-identical over 32 ids — neither is proof at 512, and neither
   has been run under this gate.

## 12. Claims made and later corrected

Recorded because each was stated confidently and propagated before being checked.

| claim | correction |
|---|---|
| "Byte-identity holds at 32 and fails by 512" | It is a **rate**, not a threshold. 32 tokens carries a 10–19% divergence probability. There is no safe length |
| "Both trees have it, so the ROCm upgrade is exonerated" | Both binaries were built under **the same** toolchain. The experiment held toolchain fixed; the upgrade is **untested**, not exonerated |
| "One event, then a cascade" | **At least one.** Later events are invisible once the sequences part |
| "Symmetric noise, not a bias" | Overreach. The tighter block SE **excludes zero**; the `--max-mem` conclusion rests on control-vs-budget agreeing (−0.0042 vs −0.0034), which needs no interval |
| "Noise floor 0.0018 nats" | From a *different* `p4` invocation than the retained arms. Superseded by **0.035252** derived from these |
| "The conversion died / was OOM-killed" | It never died. It was **starved** by a 115 GiB pin (GTT is system RAM); a `ps` probe returned a false negative and an NFS listing was one file behind |
| "Place the indexer last so no address moves" | Rust evaluates the `let` before the struct literal, and `place` is a bump allocator — the indexer landed **first** and shifted nine weights. No output changed, so no gate could see it |
| **"INT4 IDENTICAL — the vq3 path is implicated, not the generic fetch path"** | **A false green produced by the coordinator.** The comparison ran `cmp` over two id files that **did not exist** — that run had just been killed — and two empty streams compare equal, so the check was structurally incapable of any other verdict. Re-run properly, int4 **diverges**, the opposite conclusion. Fixed with a comparator that refuses unless both files exist and carry the expected id count, proven able both to refuse and to pass. **A comparison with no existence or length guard is not a check** |

---

## 13. Where the evidence lives

- Retained `.nll` arms: `docs/measurement/glm-divergence-evidence/{a,a2,b,ref}.nll` on
  `wave/fix-glm-determinism`. (`ref.nll` records no corpus hash, no `attn`, no `max_mem` — a
  different writer — and carries no conclusion alone.)
- Gate battery evidence, preserved out of tmpfs:
  `scratchpad/ppl-evidence-3obcBe/` (`a.nll`, `a2.nll`, `b.nll`, `ref.nll`, per-arm witness files).
- Divergence logs with the §3 coordinate: `scratchpad/div-{a,b}.log`, ids `scratchpad/ids-{a,b}.txt`.
- Instrumented pairs: `scratchpad/hop2-*` (clean), `scratchpad/hop3-*` (1536-token, running).
- Narrative record: `docs/investigations/glm-nondeterminism.md` on `wave/m10-spine`, whose
  `verdict:` and INDEX row lead with the text-level result.

---

## 15. The direct-load alternative, re-priced (2026-08-18)

**Question (owner):** the first implementation loaded from disk **straight into the GPU**, was
measured slower, and was discarded. Now that the arena is where the defect lives, is recovering
it a simplification — and which is faster today, bounce+`--arena-refresh` or direct?

### The archeology: the mechanism is `--direct-vmm-dma`

It was the ORIGINAL destination. `e26ab09` made the pinned-host bounce the default and kept
direct as an opt-out flag; `3e2ed79` (2026-08-01) deleted it. The code is recoverable verbatim
at `git show 3e2ed79^:src/fetch/stream.rs` and the whole mechanism is **two branches** — `queue`
picks `into = dst` instead of an arena window, and `reap` has nothing to copy.

**What it deletes, and why it is attractive.** No arena means no `mod stage` (136 lines), no
eight staging entry points in `async.hip`, and **all three coherence flags go** —
`--arena-refresh`, `--copy-by-kernel`, `--pinned-coherent`, 36 sites. It deletes the *locus* of
this defect rather than repairing it, which is exactly the right instinct.

**The prerequisite never lapsed.** `routed::pool_budget` still rounds the budget down to
`ALIGN`, with a comment saying hot slots are anchored at the high end so an unaligned budget
would "make every hot-slot DMA destination violate the alignment `stream.rs` asserts" — and
`queue` still carries `debug_assert_eq!(dst % ALIGN, 0)`. **Both are vestigial under bounce**
(a memcpy destination needs no block alignment); they are maintained for a DMA destination that
was deleted three weeks ago. That is why the restore cost ~40 lines.

### Rung 0: the bandwidth gap has NOT closed

New probe `docs/measurement/probes/fetch_dest.hip` — two arms, one variable (where the O_DIRECT
read lands), everything else held to the engine's shape: whole-expert random reads across the
75 layer files, submit-K-drain-all-K, and a GPU-BUSY arm because the NVMe DMAs into the same
LPDDR5 the MoE kernels stream out of.

**Pre-registered threshold, stated before the run: ≥ 11.4 GB/s.** Direct starts ~4% ahead (it
deletes the 38 µs/layer hop, 0.74% of a token, plus the 1–3% `--arena-refresh` costs); a token
spends 181 of 386 ms on 146.65 misses, so direct may be 8.5% worse per miss and still tie —
12.4/1.085.

| GPU-BUSY, 3 trials | K1 | K2 | K4 | K8 |
|---|--:|--:|--:|--:|
| PINNED (bounce arena) | 12.0–12.4 | 12.6–13.2 | **12.9–13.5** | 13.3–13.7 |
| VMM (direct into pool) | 2.8–3.1 | 4.3–4.6 | **6.4–6.8** | 5.1–6.6 |

**6.4 vs 13.3 GB/s — a 2.08× gap that reproduces 2026-07-30's 2.19× (5.66 vs 12.4) on kernel
6.18.39 / ROCm 7.14.** Direct misses the bar by 44%. **The perf question is closed.**

In-engine confirmation, 512 tokens at `--max-mem 115`: direct decodes at **1.92 tok/s** against
the shipped path's ~2.70 — a **29% throughput cost**, against `--arena-refresh`'s 1–3%. (Better
than the historical ablation's 2.59 → 1.19; the gap is on the DMA leg, which is only part of a
token, and today's drive is faster.)

**Two side findings from the same run, both worth keeping:**

1. **The `get_user_pages` EFAULT has not returned.** Direct read, decoded, and produced 512
   in-range ids on 6.18.39. The 2026-07-17 amdgpu regression is still absent.
2. **The "coherent read tax" does not reproduce.** Over a 256 MiB buffer the device reads
   `hipHostMalloc` at **229.2 GB/s** and device-local VMM at **235.5** — within 3%. The ~9%
   figure that justified the VMM allocator was already flagged as having no live source
   (`kernels/vmm.hip`); under ROCm 7.14 it is not there at all. This does not change the
   allocator choice, which stands on the **write** side measured above, but a future reader
   should not re-derive a read-side argument from it.

### Two instrument defects of mine, caught before the numbers were used

**The first run of this probe reported 61 GB/s on the VMM arm — 4× the drive's own ceiling in
the PINNED control.** Two causes, both mine:

- The worker seed was reset per `run()`, so **every trial replayed the same offsets** and trials
  2–3 were served from page cache. O_DIRECT was silently not in force for that arm.
- The memory-type readback ran over a **15 MiB** buffer against gfx1151's **32 MiB MALL**, so it
  reported 456 vs 567 GB/s — *neither number was DRAM*, and it discriminated nothing.

Fixed: per-cell offset salt; `/proc/self/io` `read_bytes` accounting that flags any cell whose
block-layer traffic falls below 90% of what it asked for (`!CACHED disk=67%` — it fires on
exactly the cells that produced the bogus number); readback moved to 256 MiB. **The detector is
red-proofed at startup**: a deliberately buffered 64 MiB re-read must move `read_bytes` the
first time and not the second, and the probe prints `DETECTOR ARMED` or declares itself
decoration.

### Ruled out without a measurement: reading the arena in place

A third reading of "direct" — no copy, no DMA change, the MoE kernel reads the host arena where
it lies — loses on an argument. **The pool is a cache.** A streamed expert is read on its miss
and again on every subsequent hit (75% of accesses at `--max-mem 115`), so a system-domain read
tax would be paid forever to save one copy. Not built.

### The risk direct does NOT remove, and one it adds

It does **not** fix `arena-relocation-vs-in-flight-reads` (9/8452 corrupted reads at a tight
budget) — that is the *pool's* compaction `memcpy_dtod`, not the staging arena. And it **widens**
that window: a slot is being written for the whole ~2.4 ms read instead of the copy's ~0.18 ms.
Not a blocker at `--max-mem 115`; a reason never to run this arm at a tight budget.

### Rung 2: direct mode DIVERGES — the arena is not sufficient to explain the defect

`tests/determinism-glm.sh` with `DETERMINISM_FIX_FLAGS=--direct-vmm-dma`, pinned prompt
(317 bytes, md5 `18927a780b36b029d03450d2100e9242`), 512 × 2 at `--max-mem 115`:

```
RED: 91 of 512 ids differ, first at position 420 (0-based)
  pos 420: A=8317 B=14959
```

Both arms 1.91 / 1.84 tok/s, `slot_stalls 0`, no EFAULT.

**A decode with NO staging arena and NO H2D copy still fails to reproduce itself.** The bytes
go from the NVMe into the pool slot by DMA and the MoE kernel reads them there; nothing in
between. So:

> **Any account of this defect that is purely "the device-side copy read stale bytes out of the
> pinned arena" is refuted as a COMPLETE explanation.** The defect does not require the arena
> hop to exist.

**And that makes §14 worse, not better.** `--arena-refresh` is still the only intervention
measured to suppress the divergence, and it acts on a structure whose *removal* does not
suppress it. It was recorded there as a mitigation with an unexplained mechanism; it is now a
mitigation whose mechanism is not merely unnamed but **cannot be the one the code comments
imply**. The comments state what was observed (a bulk, arena-specific, pre-copy read suppresses
it) and that remains true; the framing that the arena hop is therefore the hazard does not.

**The confound, stated because it is not small.** Direct widens the window during which a pool
slot is being *written* from the copy's ~0.18 ms to the whole read's ~2.4 ms — roughly 13× —
and `RoutedPool::relocate` (`routed.rs:402`) memcpys a slot's bytes during compaction with no
argument that it waits for an in-flight DMA into that slot. That is the separate, already
recorded `arena-relocation-vs-in-flight-reads` defect (9/8452 at a tight budget), and direct
enlarges its exposure. **So direct's divergence may be its own hazard rather than the same
one.** Separating them needs an arm in which no relocation occurs; until that runs, the safe
reading is the one stated above — the arena hop is not *necessary* for divergence — and not the
stronger "the arena is innocent".

**Today's controls, and why the red arm still stands alone.** The same-day RED control on the
stock binary came back **GREEN twice** (2 × 512, i.e. 2048 token-forwards clean, a 0.1–2.9%
event under the 1-per-299 to 1-per-578 rate). Machine unchanged: same boot, ROCm 7.14, same
artifact, 0 KFD tenants. A *green* direct arm would therefore have been uninterpretable today —
which is exactly why the direct arm was run before more control exposure. A **red** arm needs no
control: it is a positive observation of divergence. The 0-events-in-2048 stock against
1-event-in-1024 direct is **not** enough to claim direct is worse; it is recorded as an
observation, not a finding.

### Verdict

**`--direct-vmm-dma` is dead on both counts and stays a diagnostic.** It is 29% slower in-engine
(1.92 vs ~2.70 tok/s) and 2.08× slower on the DMA leg, and it does **not** fix the defect. The
simplification it would have bought — deleting `mod stage`, eight staging entry points and three
coherence flags — is not available at that price.

It is kept behind the flag rather than deleted again, because it is the only arm with no arena
and it has now earned its keep once. The code is ~40 lines and refuses to combine with the three
staging knobs.

**What this points at next**, and it is a different question from the one §11 left open: the
missing guarantee is on the edge between *a pool slot having been written* and *the MoE kernel
reading it* — the ticket timeline, the dispatch acquire, or slot reuse and relocation — not on
the arena→slot copy. The cheapest next arm is direct at a budget large enough that
`relocate` never fires, which either removes the confound or convicts compaction outright.

### Rung 3: `--slot-refresh` — the rule's own prediction, REFUTED (2026-08-18)

§15 above proposed the one rule that fit every cell: *a device-side reader can read stale bytes
from the region the NVMe just DMA'd into, and a prior full-width read by a compute kernel repairs
it for the next consumer.* In bounce mode that read is `--arena-refresh` and it is clean; in
direct mode nothing reads the slot first, which was offered as the explanation for direct's
divergence.

`--slot-refresh` performs exactly that read of the pool slot, on the fetch stream, after the
completion and before the ticket signal the miss kernel waits on. Observed to apply, from inside
`reap` after the launch returned:

```
SLOT REFRESH applied: full-width read of 15335424 B of pool slot 2
RED: 210 of 512 ids differ, first at position 300 (0-based)
```

**The rule is dead.** The investigation has exactly one clean configuration and no rule.

**Three properties separate the clean cell from this red one, and none is isolated**: the region
read is *pinned host* memory rather than device-local VMM; the next consumer is the *SDMA copy
engine* rather than a compute kernel; and there is a copy at all. Naming which one matters is the
next question — and it is a strictly smaller one than "why does any of this happen".

### The compaction confound — closed, and one claim of mine retracted

§15 flagged that both direct arms are only evidence about the shared defect if their divergence is
not `relocate`'s own hazard. It is not:

1. **Frequency is matched.** Same command, one flag different, identical work (34,131 vs 34,118
   misses over 318 tokens): **relocs 8763 direct, 8755 bounce** — 0.1% apart. Compaction fires at
   the same rate in the configuration that is clean.
2. **Relocation races nothing, by construction.** `admit_misses` evicts, places and compacts
   strictly before `resolve` hands the batch's reads to the reaper; layer L's per-expert awaits and
   its unconditional end-of-layer `device_sync` complete before L+1 submits, so every byte of L has
   landed before L+1 relocates. That is §4's argument for bounce, and its barrier is
   mode-independent.

> **RETRACTED, same day, by its author.** §15 said direct "widens the window in which a pool slot
> is being written from the copy's ~0.18 ms to the whole read's ~2.4 ms — roughly 13x". **That is
> backwards.** In DIRECT the slot write is COMPLETE when `reap` returns; in BOUNCE `reap` returning
> means the copy was only ENQUEUED and the slot is written later on the stream. Which mode exposes
> a slot longer is **not established**. It was asserted as if measured, and it was not.

**So §15's inference stands**: the defect does not *require* the arena hop. Direct has no arena, no
copy, and a full-width read of the region it does write — and it still diverges.

**And `relocs` now prints every run**, beside `slot_stalls`. Both were counters that existed and
that nothing read: `slot_stalls` is INV-9's own falsifier and went unread for half this
investigation; `relocs` was visible only to a feature build nobody runs. A counter nothing reads
is not an instrument.

