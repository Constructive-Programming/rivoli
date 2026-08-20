---
scope: engine
status: live
verdict: How to measure, and the four-out-of-five lesson that says why the ISA beats a profile.
---

# rivoli — how to measure, and how to read what you measured

> **This is the most reusable prose in the repo and it used to be 39 KB deep inside
> `PERF.md`.** It is method, not measurement: it does not go stale when a number does.

Status: **analysis + roadmap.** Leads with the **structural paths** (the higher-level
goals that move throughput by a multiplier or restructure a whole phase); the per-kernel
findings are **follow-up polish** to that structural work and to the existing improvement
proposals ([CACHE_ROUTE](CACHE_ROUTE.md), [CACHE_PILOT](CACHE_PILOT.md), the fp8-int4
work). Residency / cache-conditional routing is **not** covered here — it is owned in full
by those two proposals.

**A correctness defect found by this document's per-kernel work does NOT live in this
document.** The `route` tranche turned up an fp8 block scale mis-applied at `block < 4`,
affecting every fp8 block-scaled GEMV — it is written up in **docs/measurement/benchmarks.md, "Bugs found
and fixed"**, because someone auditing numerics reads that file and someone tuning kernels
reads this one. Two known-broken twins are recorded there too. Perf items cross-reference
the bug; they do not host it.

## Before any quality comparison: measure the noise floor

**Added 2026-08-17 (M10), and it is a precondition, not an aside.** A paired-dNLL number
means nothing until you know what the same run compared against ITSELF produces.

| arm | two runs, identical flags | floor |
|---|---|---|
| GLM int3-vq, `--max-mem 115`, 761 scored positions | 555 positions moved, PPL 5.200080 vs 5.209284 | **0.0018 nats** of mean dNLL |
| Muse Glimmer bf16, 52/52 layers pinned, 0 streamed | **byte-identical**, PPL 7.008490 both | **0** |

So the floor is **per arm, and it is a property of streaming, not of the engine as a
whole** — see `docs/investigations/glm-nondeterminism.md`, which is scoped `glm` for
exactly that reason. Do not carry GLM's number to another arm, and do not assume a new
arm has one until its control has been run.

Three rules follow, and each has already been paid for once:

1. **Run the control.** An A/B without an A-vs-A arm cannot tell "the knob moved the
   output" from "the engine does not repeat itself". `tests/ppl-gates.sh`'s `p4` cell was
   specified without one, reddened on a real device run, and was measuring nondeterminism
   the whole time.
2. **Never rank on a difference below the floor.** On GLM that is ~0.002 nats. The gaps
   the ladder chases are 0.07–0.09 PPL = 0.0134–0.0172 nats, ~8x the floor, so the
   instrument is sound — but state the floor alongside the number.
3. **Do not quote the floor from here; re-measure it.** A floor carried in prose is an
   inherited number. `p4` measures its own on every invocation, which is why that cell
   runs three arms instead of two.
4. **Treat every such interval as a LOWER BOUND on its true width until the independence
   assumption is checked.** `bin/ppl`'s `SE = sd/sqrt(n)` assumes per-position differences
   are independent, and on a streaming arm they are not — one event propagates through KV
   into every later position. The correction could be ~4x, which is the difference between
   ~8x headroom and none. It is a deviceless check on two retained `.nll` files and it is
   owed: `docs/investigations/glm-nondeterminism.md`, the caveat under the floor table.

**Counting differing POSITIONS is not a measurement of this.** GLM moved 553 positions
across a 16-point hit-rate swing and 555 across no change at all. The count saturates;
the mean and its interval do not.

## How to read — and write — this document

**A phase profile localises cost; it does not explain it.** Any mechanism attributed to a
hot kernel without reading its ISA is a **HYPOTHESIS**. Mark it as one when writing, and
confirm with `hipcc -S` before implementing — that costs a compile, not a device slot.

This is not a general caution; it is the measured result of the first per-kernel tranche.
**Four of the five per-kernel items below had the wrong mechanism.** The profile was right
about *where* the time went in all five cases and wrong about *why* in four:

| # | Mechanism in the plan | Mechanism in the ISA |
|---|---|---|
| 2 o_proj | too few blocks to fill the machine | signed 32-bit divide in the inner loop |
| 3 attend | LDS caps occupancy at 1 WG/CU | still 1 WG/CU after the fix; the real prize was an LDS read-modify-write, and a lever the item never listed (HB) |
| 4 absorb | transpose direction, single-byte loads, low ILP | 64-bit divide LLVM cannot strength-reduce |
| 5 lm_head | split-K, "same shape argument as o_proj" | not grid-starved at all (19,360 blocks); it is load width |

Every one of those corrections came from a compiler that was available the whole time,
free, while the GPU was the bottleneck. Both invocations are CPU-only and need no device:

```sh
# 1. The gfx1151 ISA for a kernel translation unit.
hipcc --offload-arch=gfx1151 -O3 --cuda-device-only -S kernels/linalg.hip -o /tmp/k.s
awk '/^gemv_fp8_splitk:/,/^\.Lfunc_end/' /tmp/k.s > /tmp/kernel.s   # isolate one kernel
awk '/Inner Loop Header/,/s_cbranch_execnz/' /tmp/kernel.s          # isolate its hot loop

# 2. Registers, scratch, spills, occupancy.
hipcc --offload-arch=gfx1151 -O3 -Rpass-analysis=kernel-resource-usage -c kernels/attn.hip -o /dev/null
```

**Three ways a static instruction count lies.** Each cost an afternoon:

- **Unroll factors differ between the versions being compared.** Normalize before quoting a
  ratio. `mla_absorb_fp8`'s loops were unrolled ×3 before a fix and ×2 after, so raw block
  sizes (498 vs 52) suggest ~10× where per iteration it is ~6×. Count a once-per-iteration
  op — `ds_load`, or the weight load — to recover the factor.
- **A guarded path inflates the count when the guard is not taken at real dims.** That same
  kernel's 498 instructions include a 64-bit Newton–Raphson division behind `v_cmpx_ne_u64`
  which is **dead** at GLM dims (`row` ≤ 24576, `block` = 128, both fit in 32 bits). When a
  count spans a branch, say which side runs.
- **Do not conclude "no divide in the loop" by grepping for `v_rcp_iflag_f32`.** LLVM
  strength-reduces a division by a loop-invariant runtime value into a magic multiply, so the
  reciprocal disappears while the cost does not. For a **signed** divide what survives is the
  quotient correction — that is the signature to grep for. `kernels/common.hpp` cites this.

Count the cost you may have added, not just the one you removed: one exec-mask sequence
(`s_and_saveexec_b32`) went 6 → **37** → 4 across a round where only the mask handling
changed.

**All four corrected mechanisms have now been implemented and measured, and the ISA was
right every time** — absorb's load width 1.40×, lm_head's load width 1.78×, attend's LDS
read-modify-write −12%, and o_proj's divide, which was real but not binding. So the
column that matters is not "the plan was wrong four times out of five"; it is that **the
ISA reading was right four times out of four.** One caveat carries forward, from o_proj:
the ISA tells you the mechanism reliably, and **the magnitude not at all** — a 34% VALU
cut bought 2.3% there, while the same class of fix bought 78% here. Use it to choose what
to implement, then still measure what it was worth.

**The same error has a magnitude form, and this document made it too.** Item #5's estimate
was revised from "a few ms" up to "~10 ms" by reasoning that `tail` ≈ 16 ms is "almost
entirely" lm_head, implying <100 GB/s. Measured: lm_head is **8.12 ms**, roughly *half*
the bucket, and the original "a few ms" was closer than the revision. **A bucket gives you
a total, not a decomposition** — inferring a component's cost from its phase budget is the
same mistake as inferring its mechanism from its phase budget, and both were made here on
the same afternoon. Decompose by measurement before you estimate from a bucket.

**The structural paths below have NOT been through this filter.** Path A and Path B were
written in the same style, from the same profile, and their mechanism claims are
localisation plus a reasoned guess rather than localisation plus a diagnosis. They may
well be right — but treat them as un-ISA'd until someone checks, and expect the per-kernel
hit rate to apply.

## What the time IS — the CLASS axis (always on, `class/tok`)

> **CORRECTED 2026-08-16 (M10): the `class/tok` line does not exist in this tree.** Every
> field behind it — `gpu_wait_ms`, `io_wait_ms`, the `cpu` split, the `route`/`tail`
> sub-splits, `moe_us_by_miss` — described instruments the OLD engine carried (reaper-ring
> stamps, HIP-event brackets, the DSA indexer's own timeline) and the rewrite carries none
> of them yet. They were removed from `ProfileSummary` rather than ported as zeroes,
> because a bucket that reads 0 because nothing filled it is worse than an absent one.
>
> What is always on here instead is the **PHASE** line, four disjoint decode-thread
> buckets plus a measured remainder:
>
> ```
> PROFILE/tok: wall <W>ms = attend <A> + ffn <F> + fetch-wait <X> + head <H> + other <O>
>              | named <P>% of wall
> ```
>
> Unlike the class spans below these DISJOINTLY partition wall, so they may be stacked and
> `other` is the honest unattributed remainder rather than a residual absorbing every
> error. **What each bucket covers is per-arm** — set by where that arm's existing sync
> points already sit, because no sync was added to sharpen them (a sync would change the
> thing being measured) — and the per-arm table lives on `ProfileSummary`'s doc in
> `crates/engine/src/telemetry.rs`, not here, so it cannot drift from the code that stamps
> it. `tests/ppl-gates.sh`'s `profile` cell bounds `other` and censuses the buckets for
> zero; its red proofs are in [gate-red-proofs.md](gate-red-proofs.md) §5.
>
> **The method below is still the method** — that a bracket mixing host compute with
> blocking waits localises cost without explaining it, and that a mechanism attributed to a
> hot kernel without reading its ISA is a hypothesis. Each removed split returns with the
> instrument that measures it, and the paragraphs below say what each one was for.

The phase buckets below (`route` / `moe-gpu` / `tail`) say **where** time goes. They are
*regions*, and each mixes host compute with blocking waits — which is why `tail` spent this
document's whole life with most of itself attributable to no kernel. The `class/tok` line
cuts the same work by **activity**, and every term is a stamped span:

```
class/tok [spans overlap; no residual]:
    gpu-wait 321.4ms (95% of wall) | io-wait 165.1ms (49%) | cpu 6.2ms (2%)
    cpu = launch 2.6ms + route 0.31ms + submit 0.87ms + tokio-poll 2.4ms
split/tok: route = 101.1ms gpu-wait + 0.3ms host-routing
           tail wait 5.5ms, of which 5.5ms is GPU
```
*(hybrid+lru, `-bench 128`, `--max-mem 100`, wall 338.2 ms/tok.)*

These are **spans, not just counters** — pass `--spans` and they export as real OTLP
spans with true start/end times across both threads, so a trace viewer draws the overlap
instead of implying it. See [traces.md](traces.md).

> **CORRECTED 2026-08-01: the `tokio-poll` term above no longer exists.** `cpu` is now
> exactly `launch + route + submit`. The term went when the expert launches were enqueued
> straight onto the compute stream, and tokio itself was removed from the dependency graph
> the same day — its entire use had been `Builder::new_current_thread().build()?.block_on(..)`
> at eight sites, replaced by a 15-line std park/unpark `block_on`. **The measurement is
> kept as recorded**: 2.4 ms of 338.2 ms/tok wall is 0.7%, so its removal does not move the
> conclusion below, and a run whose numbers are quoted elsewhere should not be silently
> re-tabulated. Expect four terms in a fresh `class/tok` line, not five.

**These spans OVERLAP and do not sum to wall — by design.** `io-wait` is the reaper thread
blocked in `io_uring` while the decode thread computes; it is 54% of wall precisely because
it is concurrent with it. *(An earlier version of this line added "and 95% of it is hidden",
which was the broken `fetch_hidden_pct`; the honest figure was ~22%, and the point being
made here — that these spans overlap and must not be forced into a partition — never
depended on it. See [architecture.md](../reference/architecture.md) §3.)* Forcing these
into a partition is what broke the first version: it made `io-wait` a *derived*
`moe_wall − compute_gpu` (a host
clock minus a GPU clock) which reported **8.4 ms**, and `cpu` a residual. The measured
io-wait is **183.7 ms — 20× larger**. The derived number was not measuring io-wait at all;
it was measuring the *unhidden remainder*, which is a different and much smaller thing.

> **`fetch_hidden_pct` and `exposed_fetch_ms` were deleted from the engine 2026-08-01** —
> the fields, the PROFILE line's `% hidden` / `ms exposed` terms, the OTLP gauge and the
> `split/exposed-fetch` series. (The Grafana stat tile that queried the gauge was
> **repointed** at `rivoli_ms_per_tok{class="io-wait"}`, not deleted.) They are gone on the
> authority of their own 27-line doc comment, which said the number was "SUBSTANTIALLY
> OVERSTATED — an upper bound, and not a tight one… prefer `io_wait_ms`, which is measured".
> The paragraph above is why: **`io-wait` is the measurement and the derived quotient was
> not**, and a metric that reports 96% where the true ceiling is ≤57% (and 99% for a
> configuration decoding at half speed) does not become useful by being labelled a gauge.
> The diagnosis is preserved in [`investigations/perf-evidence.md`](../investigations/perf-evidence.md)
> and `reference/architecture.md` §3; only the emitting code is gone.

**The price, accepted deliberately: no residual is reported.** Measured `cpu` is 6.5 ms
where the old residual said 8.8 — so ~2.3 ms/tok is genuinely unattributed. It is now
invisible rather than dressed up as host compute. A residual bucket absorbs every error in
every other term, which makes it the one number in a profile that cannot be wrong and
cannot be useful.

**Three things this settles.**

1. **We are GPU-bound, and host compute is negligible — 6.2 ms, under 2%.** Now measured
   in four named pieces rather than inferred: kernel launch 2.6 ms (~1.5k driver calls),
   tokio poll 2.4 ms *(term deleted 2026-08-01 — see the note above)*, `submit_layer`
   0.87 ms, `route_into` 0.31 ms. Any plan premised on
   host overhead is chasing 2%, and we know which 2%.
2. **`route` is not routing.** Of its 101 ms, **101.1 ms is a blocking D2H wait and 0.31 ms
   is the actual host routing** — top-k over 256 experts across 75 layers costs nothing.
   `route` is an attention-GPU phase wearing a host-phase name.
3. **`tail`'s missing half was never a kernel.** The argmax D2H is 5.5 ms and effectively
   all of it is GPU. The rest of the old `tail` bucket is decode-loop host work now
   itemised above. **docs/measurement/benchmarks.md's "half of `tail` is in none of its kernels" is
   answered, and the answer demotes it.**

**Measuring io-wait properly exposed a pre-existing bug in `fetch_wall_ms`.** `hits` and
`misses` are rebased when the profile resets after prefill; `fetch_ns` never was, so every
`fetch` number this project has published folded the prefill's cold, expensive fetch into
the decode average. It was invisible at `-bench 512` (5 prompt tokens amortize away) and
obvious the moment the new io-wait was read at `-bench 8`, where it reported **136% of
wall**. Both counters are now baselined: at `-bench 128` **`fetch` drops 184 → 165 ms/tok,
an 11% correction** to a long-standing figure. Older `fetch` numbers in docs/measurement/benchmarks.md are
high by roughly this much, and by more at small `-bench`.

### How far to trust each bucket

| bucket | confidence | why |
|---|---|---|
| `gpu-wait` | **high** as *host state* | Stamped at every blocking call; audit found none unwrapped on this arm. |
| `io-wait` | **high** | Stamped around `run_job`'s reap loop at the ring, excluding queue/submit. Independently ≈ `fetch_wall` (165.1 vs 165.0), which is the expected agreement. |
| `cpu` | **high, but a lower bound** | Four stamped regions. Anything host-side outside them is unattributed and *not shown* — hence the ~2.3 ms gap to the old residual. |

**`gpu-wait` is not utilisation, and the gap is measured.** Sampled through a decode,
`rocm-smi` reports **83.9%** GPU busy (n=65 steady-state, 77–91%) against 95% gpu-wait.
Both are right and measure different things: ~11 points (~38 ms/tok) is the host blocked on
a GPU that is *not* executing — launch gaps, driver and queue-drain overhead. **Reading
`gpu-wait` as utilisation overstates it by an eighth.**

**But put wide error bars on the 38 ms.** `gpu-wait` is stamped and trustworthy; the 83.9%
it is differenced against is an *instantaneous SMU register read*, not a time-integrated
counter, from a tool that in the same breath reports 512 MiB of VRAM (the real unified pool
is 116 GiB of GTT), a 0% fan, and throws `map::at` on every invocation. It is also coarse
and rolling-averaged — ~12 s to track a step.

**~6 ms of the 38 is now measured rather than guessed:** the device-side kernel dispatch
floor is 1.97 µs × ~1500 kernels ≈ **3.0 ms/tok**, and the host→GPU join tax is 11–20 µs ×
~180 joins ≈ **2.7 ms/tok**. A clean negative came with it — `hipDeviceSynchronize`,
`hipStreamSynchronize` and `hipEventSynchronize` all cost the same, and spinning on
`hipEventQuery` is *strictly worse* — so there is no cheap win in swapping sync primitives.
**The remaining ~30 ms has a suspect already named in `src/gpu.rs`: per-expert host-gated
launch bubbles, which fall inside `compute_gpu_ns` and are why it is an upper bound.**
Resolving this needs GPU-timeline spans on the host clock — see [GPU_TRACE.md](GPU_TRACE.md),
which also shows that is ~4 hours of work and that the clock problem is already solved.

**Audit — what is stamped.** Every blocking call in `src/gpu.rs` was enumerated. On this
arm (`--attn dense`, `topk=device`) the decode path blocks in exactly three places: the
gate-logits D2H, the argmax D2H, and the end-of-layer `device_sync`. `pin`/`hybrid`/
`stream`/`arena` contain **no** blocking calls; expert fetch is async throughout, and the
one place it really waits — the reap loop — is where `io-wait` is now taken. Two indexer
D2Hs were unwrapped and would have corrupted the `--attn dsa` arms (invisible under
`--attn dense`, where `dsa_select_layer` never runs); both are fixed. Left unwrapped
deliberately: the `TopkPath::Verify` debug arm, the `--ppl` logits D2H, and `--checksum-x`.

**One overclaim corrected.** The tail HIP-event span was documented as the tail kernels'
execution time; it is a *bracket*. It measures 5.50 ms against a 4.66 ms microbench sum, so
**~0.84 ms (15%) is inter-kernel gap** — the caveat `idx_gpu_ns` already carried. It is an
upper bound, and the reported "0% overhead" is 0.1 ms display rounding, not a measured zero.

Expensive per-kernel detail stays behind the `trace` feature; this axis is free — it rides
joins the forward pass already pays and adds no sync.

