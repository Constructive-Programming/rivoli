---
status: live
scope: glm
verdict: GLM int3-vq does not reproduce itself run to run — two teacher-forced runs at IDENTICAL flags moved 555 of ~761 scored positions and 0.0018 nats of mean NLL (PPL 5.200080 vs 5.209284). THREE controls bound it, and the third narrowed the scope: Muse Glimmer scored twice fully pinned is byte-identical, which excludes the kernels, reduction order, toolchain and this milestone's scoring harness; Glimmer at --max-mem 32 vs 20 is byte-identical in its ids while ACTIVELY STREAMING 21 of 52 layers through one slot, which excludes layer streaming as a generic cause; and V4 at 97% hit was byte-identical over 32 ids. So the wobble is confined to the ROUTED EXPERT POOL — per-expert admission, the two-ended arena's relocations and ticket lifetime, and the MoE accumulation — which GLM has and Glimmer does not; those two candidates are NOT separated. The exclusion is strong evidence rather than proof, because Glimmer's probe is greedy ids (argmax-robust) and GLM's is NLL floats; one Glimmer --ppl pair at the same budgets makes it airtight and is experiment 0. The usable result is the magnitude: a 0.0018-nat noise floor against the 0.0134-0.0172-nat gaps the wave's ladder must resolve, ~8x headroom, so paired dNLL stays valid provided every comparison runs its own A-vs-A control. Root cause deliberately NOT chased; named, bounded, handed off.
---

# GLM int3-vq does not repeat itself, and by how much

**Found by a gate that was wrong.** `tests/ppl-gates.sh`'s `p4` cell demanded that the
per-token NLLs be byte-identical across `--max-mem` 115 and 70. They were not — 553
positions moved — and the cell reported a P4 violation. Running the control the cell did
not have (**the same budget twice**) moved **555**. The budget was never the variable.

That is worth stating as a method lesson before any measurement: **an invariant gate over
a quantity that has not been shown to repeat is not a gate, it is a nondeterminism
detector with a misleading message.** The cell has been re-specified to calibrate itself
against a control arm it now always runs; the record of what it used to be is in its own
header and in `docs/measurement/gate-red-proofs.md` §5.

## What was measured

All rows: GLM-5.2 `/var/db/rivoli/glm52-vq3-full`, `--mode int3-vq --attn dense`,
`--cache-policy 2q` (default), release, teacher-forced over `tests/ppl-corpus.txt`
(762 tokens, so ~761 scored positions — the counts below are as reported and the
off-by-one against 762 is immaterial to every conclusion here), sole-tenant under the
flock, 2026-08-17.

| run pair | positions moved | PPL | hit % |
|---|---:|---|---|
| **115 vs 115** (the control) | **555** | 5.200080 / 5.209284 | 78.2643 / 78.2352 |
| 115 vs 70 (the original gate) | 553 | — | 78.2628 / 62.0900 |

**The differing-position COUNT has no discriminating power here** — 553 against 555 — and
that is the single most useful line in this table. The first version of the gate reported
the count as its evidence. A 16-point swing in hit rate changed how many positions moved
by less than the run-to-run variation did.

The magnitude, which does discriminate:

```
mean dNLL over the control pair = ln(5.209284 / 5.200080) = 0.0017684 nats
                                = 0.177% of PPL
                                = 0.18x the repo's 1% quality bar (ln 1.01 = 0.00995)
```

## The noise floor, as a number downstream gates must state

**0.0018 nats of mean |dNLL| on GLM int3-vq at `--max-mem 115`, ~761 scored positions.**

Against it, the quality gaps this wave's ladder has to resolve:

| gap | in nats | multiple of the floor |
|---|---:|---:|
| 0.07 PPL | 0.01337 | 7.6x |
| 0.09 PPL | 0.01716 | 9.7x |

So paired dNLL **remains the right instrument and remains usable** — the effects being
chased are an order of magnitude above the wobble. What changes is that a comparison of
two GLM runs may no longer be reported without its floor, and **a difference under ~0.002
nats on this arm is not a measurement**. `tests/ppl-gates.sh`'s `p4` cell now measures the
floor on every invocation rather than citing this page, because a floor that is quoted
rather than re-measured is an inherited number.

This applies to every downstream paired-dNLL gate in the wave — M11's fp8 ladder, M15's
boundary work, M17's quality comparisons. It does **not** apply to Glimmer (below).

## What is EXCLUDED

Two controls, both run before this page was written:

- **Muse Glimmer, teacher-forced twice, `52 of 52 layers pinned, 0 streamed`:
  BYTE-IDENTICAL, PPL 7.008490 both runs.** This is the load-bearing control. It excludes,
  as a class: the HIP kernels' reduction order, the fp32 accumulation path, the
  HIP 7.14 / clang 23.1 toolchain, the argmax fold, the logits D2H, `score::nll_of`'s
  arithmetic, the `.nll` writer, and the teacher-forced walk itself. **The scoring harness
  this milestone built is bit-reproducible.** Whatever moves, moves upstream of it.
- **Muse Glimmer fp8, `--dump-ids`, 64 tokens, TWO budgets — ids BYTE-IDENTICAL while
  actively streaming** (2026-08-17, from M11's own P4 gate):

  | budget | partition |
  |---|---|
  | `--max-mem 32` | 52 of 52 layers pinned, 0 streamed, 28.5 GiB tier |
  | `--max-mem 20` | 31 of 52 pinned, **21 streamed through 1 slot**, 19.4 GiB tier |

  **This is the control that changes the scope of this page.** The earlier Glimmer control
  streamed nothing, so it could only say "an arm that does no streaming is deterministic".
  This one says an arm doing *real* streaming — partition, slot refill under genuine
  eviction pressure through a SINGLE slot, the write-after-read fence — is deterministic
  AND budget-neutral.
- **DeepSeek-V4 at 97% hit: byte-identical over 32 decoded ids** (`--dump-ids`). Weakest of
  the three: 32 greedy tokens, and 97% hit means its streaming path is barely exercised.

## What is NOT split, and must not be claimed

**Layer streaming is NOT the cause** — the Glimmer 32-vs-20 control rules out the generic
mechanism. What GLM has and Glimmer does not is the **routed expert pool**, and that is
where the wobble is confined:

1. **Per-expert admission and the two-ended arena** — relocation, eviction, ticket
   lifetime racing an in-flight read. Glimmer's slot machinery is whole-layer and
   coarse-grained; GLM's is per-expert with compaction. The old tree names the intended
   invariant explicitly (`old:docs/reference/architecture.md` §6: *"Misses are allocated
   before any read is issued, so a relocation never races an in-flight fetch. This is the
   correctness spine of the cache"*) and separately recorded a live violation of it — a
   read outliving its layer and having its slot memcpy'd out from under it. Leading
   candidate. **A candidate, not a finding.**
2. **The MoE accumulation** — the fixed-point drain over two streams. Glimmer is dense and
   exercises none of it.

**Those two are not separated, and this page does not guess between them.**

### The honest caveat on the exclusion, which is inferential and not yet proof

**The three controls and the GLM measurement are not the same probe.** Glimmer's evidence
is 64 (and 512) greedy **ids**; GLM's is ~761 **NLL floats**. An id is an argmax and is
robust to any perturbation smaller than the gap between the top two logits; an NLL float
carries every last bit. So a Glimmer run could in principle wobble in the logits and still
emit identical ids — the two probes differ in sensitivity by orders of magnitude, and
`old:`'s own softcap finding is the precedent (7 of 1103 captures moved with `emitted.ids`
IDENTICAL, which is why greedy gates are provably blind to that path).

The exclusion above is therefore **strong evidence, not proof**. Making it commensurable is
cheap and is experiment 0 below.

## The cheap next discriminators, in order

Each is one device session, none needs new code:

0. **Glimmer `--ppl` at two budgets, the STREAMING one on the control arm** — the same
   comparison as the third control, but scored as NLL floats, so it is the same probe GLM
   failed. One invocation:

   ```
   PPL_MODE_FLAGS='--attn dense' PPL_MEM_A=20 PPL_MEM_B=32 \
     tests/ppl-gates.sh /swarm/storage/ai/rivoli/glimmer-30b-fp8 p4
   ```

   **The budget order matters and is the opposite of the obvious one.** `p4` runs its
   control pair at `MEM_A`, and the question here is whether STREAMING perturbs the
   output — so the streaming budget (20, which pins 31 of 52 and streams 21 through one
   slot) goes on `MEM_A`. Putting 32 there runs the control fully resident and re-answers
   the question the first Glimmer control already answered.

   Expected: the control comes back byte-identical, which puts the cell in its STRICT
   branch automatically, and the budget arm must then match byte-for-byte too. Green makes
   the layer-streaming exclusion airtight rather than inferential; red would be the most
   interesting result on this list. It also exercises `p4`'s strict branch on a device for
   the first time.

   **Needs a binary that reads fp8 Glimmer, which is M11's work and is NOT in the tree that
   wrote this page** (`grep -r Fp8 crates/engine/src/glimmer/` is empty at
   `wave/m10-spine`). Either run it with M11's binary, or run it after that merge, or
   substitute bf16 Glimmer with budgets chosen to straddle its own partition — the argument
   is about layer streaming, not about the weight format.

## The cheap next discriminators, in order

Each is one device session, none needs new code:

1. **GLM at two budgets, control-paired at each.** If the wobble is the routed pool,
   its magnitude should SCALE WITH STREAMED VOLUME: the control pair at `--max-mem 70`
   (62.1% hit, so ~1.7x the misses) should move further than the control pair at 115
   (78.3% hit). A flat magnitude across budgets points at the MoE accumulation instead.
   This is the highest-information run and it is already half-paid — `p4` runs a control
   pair at `MEM_A` every time, so it only needs a second invocation with the budgets
   swapped.
2. **GLM fully resident.** If a budget exists at which GLM's routed pool never evicts, its
   control pair should go byte-identical. That would separate candidate 1 from candidate 2
   outright: no admission, no relocation, but the MoE accumulation still running. Whether
   such a budget exists on this box is the open question — the routed experts not fitting
   is the whole reason this engine exists (P1), so it may not, and a shrunk shadow artifact
   would be the substitute.
3. **A dense-only GLM prompt** is NOT a discriminator and is listed so nobody tries it:
   GLM has 3 dense layers against 75 MoE, so no prompt avoids the MoE path.

## Two notes on the record itself

- **Every number on this page is reproducible from what is written here**, because the
  probe is a teacher-forced walk over a COMMITTED corpus (`tests/ppl-corpus.txt`, whose
  FNV-1a the engine logs and writes into the `.nll` header) rather than a prompt. That is
  deliberate. `docs/measurement/baseline-2026-08-16.md` records its command as
  `--prompt '<P>'` — a literal placeholder — and the text appears nowhere in that doc, its
  commit body, or the tree, so **that baseline's rows cannot be reproduced from their own
  record.** Any doc here records its input verbatim or names a committed file;
  `tests/ppl-gates.sh` pins its one prompt as a script constant for the same reason.
- **A bandwidth-bound arm is insensitive to prompt choice**: Glimmer bf16 measured 2.51
  tok/s against the baseline's recorded 2.56 — within 2% — on a *different, unrecoverable*
  prompt. Worth knowing before anyone treats a 2% move on that arm as a signal. It does not
  rescue the baseline's reproducibility; it just bounds what was lost.

## What is NOT being done

**The root cause is not being chased here** (coordinator's instruction, 2026-08-17, and
the right call): M10's job was the instruments, three other tracks are blocked on them,
and a residency-race hunt is its own milestone with its own gates. This page exists so
that work starts from a bounded statement rather than from a rediscovery.

What it hands off: a magnitude (0.0018 nats), a proven-clean half of the system (kernels,
toolchain, scoring), two unseparated candidates, and a ranked list of one-session
experiments that separates them.
