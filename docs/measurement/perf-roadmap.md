---
status: live
verdict: The ranked performance roadmap, re-scored 2026-08-04 on recurring cost rather than wall at today's bottleneck. Live rows: #2 VQ_K=2048 (1.189x MoE kernels at 12-bit, +18.7% relFrob, needs a real dNLL gate), #10 general-R MoE kernels. #5 DONE 2026-08-02 (HB 8→16, 2.08x kernel, −3.2 ms/tok, gated). #8 and #11 stay closed on complexity and quality — NOT on "bytes stop buying anything below the floor", which was the wrong axis.
---

# rivoli — the ranked performance roadmap

> Evidence for every row is in
> [`investigations/perf-evidence.md`](../investigations/perf-evidence.md); method is in
> [`how-to-measure.md`](how-to-measure.md).

## How rows are scored — CORRECTED 2026-08-04

> Rows #2, #8 and #11 were each closed or discounted with arithmetic of the form *"this win
> hides behind the other bottleneck, so it buys ~0 ms of wall."* **That is the wrong axis,
> and it was used three times.**
>
> A win that does not move `max(transfer, compute)` still removes work that runs **on every
> token, forever**: GPU busy time is energy and thermal headroom on a unified-memory part,
> and bytes not fetched are NVMe reads not performed. Meanwhile the costs those rows were
> weighed against — a requant, a rebuild, a format migration — are paid **once**, against
> millions of decodes. They amortize to ~0.
>
> So score a row on **recurring cost** (ms, joules, bytes/token) and put one-time costs on
> the amortized side. What stays on the cost side is what is *permanent*: **quality,
> complexity, maintenance.** Two consequences that bind:
>
> - A cheap quality **screen** is enough to reject a row whose benefit really is ~0. It is
>   **not** enough once the cost side amortizes away — then the screen must become a real
>   paired dNLL before the row can be closed.
> - *"Below the floor, where bytes stop buying anything"* (item 11) and *"~2% overall"*
>   (item 8) are both this error. Both closures happen to survive re-scoring, but on
>   different grounds, re-stated in their notes below.

## Ranked roadmap

| # | Item | Path | Est. impact | Effort | Status |
|---|---|---|---|---|---|
| 1 | `fp8_to_i4` | B | int4 PPL 73.43 → **5.12**, hybrid → **5.19** | low | **done** |
| 1b | ~~`--hot-pct` re-tune~~ | B | — | — | **struck — flag deleted, unrunnable** |
| 2 | VQ_K=2048 L1-resident codebook | B / follow-up #1 | **12-bit: 1.189× MoE kernels, compute floor 117→~99 ms. 11-bit: 7.7% fewer bytes, transfer 181→167 ms.** Both cost **+18.7% relFrob** | med (requant) | **PROBED 2026-08-04, NOT closed** — two variants at one quality price; needs a real dNLL gate, see below |
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
| 5 | `mla_latent_attend` occupancy | follow-up #3 | `acc`→regs −12%; **HB 8→16 = 2.08× kernel, −3.2 ms/tok** | med | **DONE 2026-08-02** — HB=16 shipped on both backends, gated at +0.00217 nats; `MLA_MIN_TILES_PER_SPLIT` measured INERT and stays at 4; Vulkan HB=16 too |
| 6a | lm_head load width | follow-up #5 | kernel **1.78×**; `tail` **−3.2 ms** in-engine; wall **~+1%, not noticeable** | low | **done** |
| 6b | o_proj split-K / x-tiling | follow-up #2 | — | — | **refuted and reverted** |
| 7 | `mla_absorb` restructure | follow-up #4 | **−0.80 ms/tok, measured** | med | **done** |
| 8 | Faster demand fetch (deeper queues, split reads, unpinned arena) | B | — | — | **closed as negative 2026-08-01** — the drive is already giving what the queue depth buys; see below |
| 9 | Layer-major prefill | A | **prefill 2.15×**; expert reads 159.56 → 28.20/token (the compulsory floor); output byte-identical | med | **done; DEFAULT since 2026-08-03** — the flag is deleted and the mode is derived (`--trace` falls back to token-major, since a v2 capture mis-segments under it). Decode pays a one-off ~2.7 s warm-up, 1.8% of the saving. `docs/reference/architecture.md` §14 |
| 10 | General-`R` MoE kernels (tiled GEMM) | follow-up #9 | the rest of #9: a 2-row pass still re-reads its experts from LPDDR5, and that is now the prefill bound | **high** | new — see below |
| 11 | Better cache policy (residency / hit rate) | B | ceiling **8.6 ms/tok (2.4%)** at 115 GiB — and that is Belady's, not a reachable one | — | **closed as negative 2026-08-02 AT 115 GiB — still live at 61**; see below |

> **Item 2 was probed 2026-08-04 and it is TWO products at one quality price, not one item.**
> `docs/measurement/probes/vq_codebook.hip` (real dims 6144/2048, 9 experts, R=1, real 11-bit
> unpack, median of 3) and `examples/vq_k_probe.rs` (real fp8 weights, 2^20-subvector sample,
> both codebooks from the same sample, scored on **held-out** experts).
>
> | variant | recurring win | format cost |
> |---|---|---|
> | **12-bit container, K=2048** | MoE kernels **1.189×** (`gateup` 934→741 µs, `down` 462→411 µs); compute floor 117 → ~99 ms | **none** — only the codebook and encoder move |
> | **11-bit container, K=2048** | 7.7% fewer bytes/token (2.25→2.08 GB); transfer 181 → 167 ms | index packing changes |
>
> **The two halves of the original item fight each other, which is the finding.** The 16 KiB
> codebook alone is worth 1.189×. Adding the 11-bit packing — the half that actually buys the
> bytes — collapses it to **1.022×**, *while* reading 7.7% fewer bytes. A 12-bit index never
> spans more than 2 bytes (`shift ∈ {0,4}`, strength-reducible); an 11-bit index starting at
> bit 7 spans 3. A dword-load variant behaves identically (1.029×), so it is the loss of the
> shift pattern, not the extra byte load. Calibration arms: `k4096_b12_f32` (64 KiB) is
> **0.739×**, `k256_b12_f16` is 1.178× — so the probe is size-sensitive and **K=2048 already
> captures the whole asymptote**. `shared_gu_k4096` (one codebook for gate+up, 64→32 KiB
> aggregate) buys only 1.6%, which refutes the aggregate-footprint story: the effect keys on
> the span each gather *stream* ranges over, knee between 16 and 32 KiB. No rocprof on this
> box, so that mechanism is inference from eight consistent arms, not a counter reading.
>
> **Quality is on the rate-distortion frontier, and that is the real cost.** Held-out mean
> relFrob 0.15509 (K=4096) → 0.18407 (K=2048) = **+18.68%**; high-rate VQ theory for `d=4`
> predicts `2^0.25` = **+18.92%** for halving K. Spread across 3 projections × 3 experts is
> under 0.2%. There is no structural slack to recover — an independent arrival at
> [`codebook-rotation.md`](../investigations/codebook-rotation.md)'s "int3-vq is rate-limited".
> int4 anchors at 0.11858. Extrapolated against the 5.120/5.275 ladder that is ~+0.024 nats,
> ~2.4× the 0.00995 bar, **on the rung that is already worst of three**.
>
> **Why this row is NOT closed.** The first recommendation was NO-GO, and half its reasoning
> was the retired axis above — "the 1.189× is spent on slack the fetch already covers." Under
> the corrected scoring the 12-bit variant costs *nothing but the requant and the quality*,
> and the requant amortizes. So the decision now rests **entirely on quality**, and
> +18.68% relFrob is a **screen** extrapolated across two quantizer families — adequate to
> reject a ~0-benefit row, not adequate to reject this one. **Settle it with a real paired
> dNLL after a requant**, and repeat every arm: ~40% of 5k-token runs are silently corrupted
> (see the warning under item 5).
>
> Also measured and worth recording: sharing one codebook between gate and up is
> **quality-free** (0.15511/0.15510 vs 0.15512/0.15501 separate), matching
> `codebook-rotation.md`'s cross-layer result — but the kernel pays only 1.016× for it, so
> there is no reason to do it.

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
>
> **RE-SCORED 2026-08-04. The closure survives, but "~2% overall" was never the reason.**
> 2% of every token forever is not a rounding error, and under the scoring correction above
> it is a genuine recurring saving. What actually closes this row is that its cost is
> **permanent, not one-time**: split reads mean "a real change to the ring", i.e. complexity
> and failure surface carried for the life of the engine, against a 2% that only 18% of
> layers see. The duty-cycle half closes on its own arithmetic and needs no re-scoring — a
> +67 ms/token prefetch cost against a ≤85 ms/token ceiling is a **loss**, not a hidden win.

> **Item 11 closes the OTHER byte lever, and it was never priced until 2026-08-02.** Hit rate
> is the second of the two things that move bytes, and every policy on record — LRU vs 2Q vs
> ARC, `TwoQSplit::default`'s Kin/Kout sweep — was ranked only against the others, so nobody
> could say whether 2Q's win was 2 pp or 20 pp short of what the trace allows. `bin/replay`
> now prints Belady's clairvoyant bound (`opt`) and a `headroom` line beside it.
>
> **At 115 GiB the answer is that the lever is spent.** Best online is 83.63%, OPT 90.00% —
> 6.37 pp, which looks like room and is not. Transfer at best-online is already **125.6 ms
> against §3's 117 ms compute floor**, and a clairvoyant policy takes it to 76.7 ms, *below*
> the floor, where bytes stop buying anything. Since wall follows `max(transfer, compute)`,
> the entire remaining value of cache-policy work is **125.6 → 117 = 8.6 ms/token, 2.4%** —
> for a policy that cannot exist. Break-even is 84.75% and LRU sits 1.1 pp under it. **Do not
> open another policy at this budget.**
>
> **At 61 GiB the same table says the opposite**: transfer 230.6 ms against the same floor,
> 13.74 pp of headroom, OPT worth ~105 ms/token. So residency has no budget-free verdict —
> which also means the cheapest 105 ms on a starved machine is `--max-mem`, not a policy.
> Full table, the transfer model and three optimism caveats in
> [`benchmarks.md`](benchmarks.md), "The Belady bound on residency".
>
> **RE-SCORED 2026-08-04. "Below the floor, where bytes stop buying anything" is FALSE as
> written** — it is the clearest instance of the retired axis, and this row is where the
> phrase came from. Bytes not fetched below the compute floor are still NVMe reads not
> performed: drive wear, power, and a fetch ring that idles instead of working. What is
> capped at 8.6 ms/token is **wall**, not value.
>
> The closure still holds, on the cost side rather than the benefit side: another policy is
> **permanent complexity** — a new implementation, a new tuning surface, and in `--mode
> hybrid` a residency change is an *output* change (the INV-1 defect,
> `docs/reference/architecture.md` §8b). That cost never amortizes. Weighed against ≤6.37 pp
> of hit rate that Belady says nobody can reach anyway, the answer is unchanged: **do not
> open another policy at 115 GiB.** At 61 GiB nothing about this row was ever
> bottleneck-relative, so it is untouched by the correction.

> **Item 5 is done, and it corrects three things this table said about it.** HB 8→16 halves
> the DRAM KV re-read multiplier: **226.5 → 108.8 µs at nr512 (2.08×)**, 769.8 → 421.2 at
> nr2048, **−3.2 ms/token of `route`** in engine. Free in registers, LDS unchanged.
>
> 1. **It is NOT a numerics-free change**, which is why this item recommended the HB route.
>    `by_grid` binds above **nr≈640** and doubling HB doubles it, moving the split plan and
>    the summation order. It took the full gate and passed: **+0.00217 nats, CI
>    [−0.00243, +0.00676]** against a 0.00995 bar.
> 2. **`MLA_MIN_TILES_PER_SPLIT` is INERT at both HB values.** This item predicted it would
>    "start to bite" once HB rose. It does not — `tps` rounds back. It stays at 4, so the
>    "two-parameter sweep" has one live parameter.
> 3. **"~5–7 ms now" overstates a short-context decode.** A 512-token run averages nr≈170,
>    where the kernel is small; the realized figure is −3.2 ms and it grows with context.
>
> **Vulkan is HB=16 too** (done the same day; the row's first version said it was still 8).
> `MLA_HB` is now single-sourced from build.rs into both `dims.rs` and the shader's `-DHB`,
> which also retired an assert that had only ever held by coincidence and a device-limit
> check that would have admitted hardware unable to run the 512-thread attend workgroup.
> Vulkan suite: 141 passed, 0 failed. Full write-up in [`benchmarks.md`](benchmarks.md),
> "The MLA HB sweep".
>
> **A warning that outranks the item.** The gate above is trustworthy only because every arm
> was REPEATED: ~40% of 5k-token runs are silently corrupted (`benchmarks.md`, "Long runs are
> NON-DETERMINISTIC"), which is ~0.5 PPL — 50× the bar. Four readings during this sweep were
> wrong before the arms were repeated. Do not take a single 5k-token quality number here.

**Suggested sequence, revised.** The original sequence led with #1 and treated #4 as the
big multiplier; #1 has landed and #4 has a measured loss against it, so:

1. ~~**#5's HB × `MLA_MIN_TILES_PER_SPLIT` sweep**~~ — **DONE 2026-08-02.** It was a 4-site
   change (`kernels/attn.hip`, `kernels/vk/mla_latent_attend.comp`, and the two mirrored
   launcher constants in `src/backend/vk.rs`), which the item's text did not say; and it had
   one live parameter, not two.
2. **Path B (#2), 12-bit variant** — probed 2026-08-04 and now blocked on **one measurement**:
   a requant at K=2048 and a real paired dNLL. 1.189× on the MoE kernels for no format
   change; the whole question is whether +18.68% relFrob lands inside the bar. Still the
   *only* live item in Path B, since 1b is struck. Do the requant before re-arguing the row —
   the screen has already given everything a screen can give.
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

> **#10, and why it is `high` effort rather than "template the kernel wider".** #9 removed
> the NVMe term from prefill and left the LPDDR5 one, which is now the bound: a pass is
> `MAXROW` = 2 rows, so each 2-row pass still streams its experts out of RAM and layer-major
> only halves how often. `moe_gateup_vq`/`moe_down_vq` and the int4 twins are templated on
> `R` and return guard 1004 above 2, so the obvious move is to raise the template.
>
> **That does not work, and the reason is the activation side, not the weight side.** The
> kernels hold `float acc[R]` in registers and read `x` from cache. At R=2 the activations
> are 48 KB and every wave re-reads them out of L2 for free. At R=769 they are 18.9 MB —
> larger than L2 — so `x` would stream from DRAM once per output-row block, and at 2048
> output rows per expert that costs far more than the 15.34 MB of weights it saved. A wide
> `R` therefore needs LDS tiling on BOTH operands, i.e. an actual tiled GEMM per expert per
> projection, plus its Vulkan twin. That is the work; the payoff is the gap between #9's
> 2.15× and the ~13× the read count alone would suggest.
>
> Intermediate `R` is the cheap probe and it is not free either: `moe_h` is
> `[slot][row][inter]` and grows as `union × R × moe_inter × 4`, i.e. 2.1 MB per row of `R`
> at a 257-expert union — 67 MB at R=32, 269 MB at R=128. Dispatching one expert at a time
> collapses that to `R × moe_inter × 4` and is the way in.
