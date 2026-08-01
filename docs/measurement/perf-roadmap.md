---
status: live
verdict: The ranked performance roadmap. Live rows: #2 VQ_K codebook, #5 the MLA HB sweep.
---

# rivoli — the ranked performance roadmap

> Evidence for every row is in
> [`investigations/perf-evidence.md`](../investigations/perf-evidence.md); method is in
> [`how-to-measure.md`](how-to-measure.md).

## Ranked roadmap

| # | Item | Path | Est. impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | `fp8_to_i4` | B | int4 PPL 73.43 → **5.12**, hybrid → **5.19** | low | **done** |
| 1b | ~~`--hot-pct` re-tune~~ | B | — | — | **struck — flag deleted, unrunnable** |
| 2 | VQ_K=2048 L1-resident codebook | B / follow-up #1 | med (lifts gather wall) + smaller experts | med (requant) | new |
| 3 | Batched-GEMV kernels | A | med now, unlocks MTP | med–high | **done, on `main`** — 6 kernels take `nrow`, bit-identical per row |
| 4 | MTP / speculative decode | A | **1.108× measured** with `--mtp-min-conf 0.8` (0.93–0.95× ungated) | high | **DONE and WON 2026-07-31** — gate on draft confidence; see the note below |

> **Item 4 was re-derived as this table asked, and reached the same verdict by a route that
> explains it.** Shipped end to end (`docs/reference/architecture.md` §13): 2.50 vs 2.69 tok/s at 128
> tokens, 2.49 vs 2.63 at 512, output byte-identical. The mechanism is arithmetic, not
> tuning: the MoE is 67% of the pass and a batched pass launches the **UNION** of both rows'
> routing — 14.5 experts against a single row's 9, so **1.61× the weight reads** — while the
> second row per expert is genuinely free (178 vs 176 µs on 0-miss layers). Attention
> behaved as designed (0.83× per token). Break-even is **1.53 tokens/pass ≈ 53% acceptance**
> and measured acceptance is 42–54%.
>
> So it is a coin flip landing slightly wrong, not a structural impossibility, and the ONLY
> lever is acceptance — skipping zero-weight rows inside the kernel would recover ~8%,
> because ~92% of an expert launch is the weight read. **Do not re-open without a draft head
> that clears 53%.** GLM-5.2 ships one MTP layer and depth-2 chains accept at 4.4%, so that
> head is not available in this checkpoint.
>
> **RE-OPENED AND WON, 2026-07-31 (same day). The "ONLY lever is acceptance" sentence above
> is the error.** The other lever is not spending the verify pass on drafts that will not
> pay for it. `--mtp-min-conf 0.8` gates on the draft head's own top-1 probability and
> measures **2.97 tok/s against 2.68 sequential = 1.108×**, byte-identical output, on the
> coherent (memory-systems) prompt. Two things made it work, neither of which this section
> had: the accept-vs-confidence calibration is **prompt-invariant** (the ≥0.8 bin lands at
> 91% across two prompts and two quantizations, while its share of drafts moves 25% → 52%),
> and acceptance tracks the **text** rather than the head — 65.7% on coherent generation
> versus 46.0% on the sample that trips the degeneration warning. Rebuilding the head at
> int4 moved acceptance by 3.4 pp ± 7.4, i.e. not at all, so "de-quantize the head" is
> refuted. Full table in `docs/reference/architecture.md` §13.
| 5 | `mla_latent_attend` occupancy | follow-up #3 | ~5–7 ms now, huge at long ctx | med | **partial** — `acc`→regs done (−12%); HB sweep not run |
| 6a | lm_head load width | follow-up #5 | kernel **1.78×**; `tail` **−3.2 ms** in-engine; wall **~+1%, not noticeable** | low | **done** |
| 6b | o_proj split-K / x-tiling | follow-up #2 | — | — | **refuted and reverted** |
| 7 | `mla_absorb` restructure | follow-up #4 | **−0.80 ms/tok, measured** | med | **done** |
| 8 | Faster demand fetch (deeper queues, split reads, unpinned arena) | B | — | — | **closed as negative 2026-08-01** — the drive is already giving what the queue depth buys; see below |

> **Item 8 is a closed door, and it is worth knowing which door.** The demand fetch runs at
> ~10 GB/s; `docs/measurement/probes/fetch_batch.hip` reproduces the engine's exact shape (pinned bounce
> buffers, submit-*m*-drain-*m*, random 15.3 MB reads, GPU busy beside it) and the drive
> gives **7.7 GB/s at QD1 and ~13 at QD4**. Weighted by the engine's own miss distribution
> that predicts 15.8 s against a measured 18.3 s of `io_wait` over 64 tokens — inside the
> probe's own run-to-run spread, which is itself ±25% at QD1. Splitting one expert read
> K ways does raise its queue depth (1.94 → 1.44 ms), but only the 18% of layers that miss
> exactly once benefit: **~2% overall**, for a real change to the ring. Measured, dropped.
>
> **The duty cycle looked open for a day; it is not.** The drive idles ~35% of every token,
> and filling it needs the routing known before that layer's attention. The predictor works
> — 82.7% recall on the misses (`--features pred-probe`, `--pred-probe`) — but the window
> is **1.13 ms against
> a ~2 ms expert read**, so it fits 0.74 of one read where a layer needs 2.9, and the 23%
> of a top-8 prefetch that goes unused costs +67 ms/token against a ≤85 ms/token ceiling.
> Closed 2026-08-01; full arithmetic in `CACHE_PILOT.md` §"Feasibility, settled".
>
> That leaves **#2 as the only live fetch lever**: it moves *fewer bytes*, shortening the
> busy 65% rather than trying to fill the idle 35% — and a smaller expert is also the one
> thing that would make the idle window worth filling.

**Suggested sequence, revised.** The original sequence led with #1 and treated #4 as the
big multiplier; #1 has landed and #4 has a measured loss against it, so:

1. **#5's HB × `MLA_MIN_TILES_PER_SPLIT` sweep** — the cheapest unclaimed win, and still
   mandatory before any long-context work. Note it is a **4-site change** (`kernels/attn.hip`,
   `kernels/vk/mla_latent_attend.comp`, and the two mirrored launcher constants in
   `src/backend/vk.rs`), which the item's text does not say.
2. **Path B (#2)** — the biggest remaining structural lever, on the 210 ms that is 60% of
   wall. Now the *only* live item in Path B, since 1b is struck.
3. ~~**`tail`'s missing ~62%**~~ — **ANSWERED and struck.** The CLASS axis shows it is
   decode-loop host CPU (~6 ms of the 8.9 ms `cpu` bucket), not a hidden kernel. Promoting
   it was the right call on the evidence available; measuring it cost one run and demoted
   it, because total host compute is under 3% of the token.
4. **Path A (#3 → #4)** only after the draft cost is re-derived against today's `moe-gpu`.
   Do not re-estimate it from accept rate: 84% accept has already been measured, and lost.

**Measurement discipline (learned the hard way):** rank format/numerics changes by (a) the
replay residency sim, (b) a fixed forced-token wall-clock bench, and (c) perplexity for
quality — **never** free-running greedy tok/s (confounded by output degeneration; a broken
run looks fastest). See [docs/reference/modes.md](docs/reference/modes.md) and [docs/measurement/benchmarks.md](../benchmarks.md).
