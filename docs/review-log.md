# Review log — findings to walk through together

Every code review run this session, its findings, and how each was resolved.
All review findings were **fixed before their commit** (review-before-pause);
this log exists so we can sanity-check the fixes and the judgment calls, and so
the **open / deferred** items (bottom) don't get lost.

## Review 1 — attention modes (dense/streaming/dsa/misa + fp8 scalar)
Commit `f58d70e`. 8 finder angles → verify → 10 findings, all fixed.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | Indexer `k_norm` used `rms_norm_eps` (1e-5); HF hardcodes LayerNorm `eps=1e-6` | correctness | `K_NORM_EPS` const |
| 2 | BF16 became indexable → the unvalidated `require()`+`read_f32` path could reinterpret bf16 as f32 | correctness | `Snapshot::typed(name, dtype)`; every raw-decode site checks dtype |
| 3 | `kv.append` ran before the fallible indexer select → cache desync on error | correctness | reordered: select before append |
| 4 | `mem::take(rows)` dance unnecessary + zeroed `s.rows` on the select error path | simplification | direct disjoint field borrows |
| 5 | MISA `active_heads=0` selected first 2048 by tie-break, silently | correctness | `ensure!(h>0)` + `--misa-heads` validates |
| 6 | Shared-layer `pos==0` escape could return an empty selection | correctness | unconditional guard (layer 0 always full) |
| 7 | No `max_ctx` bound before device KV writes | correctness | `ensure!(pos<max_ctx)` in `forward` |
| 8 | Per-token `k_norm` allocs + 32× redundant key widening | efficiency | hoisted `k_norm`; widen once into scratch |
| 9 | MISA pool maintained in DSA mode for an impossible mid-run mode switch | efficiency | gated on `active_heads.is_some()` |
| 10 | MISA-vs-DSA overlap test near-vacuous (forced IoU≥0.99) | test | m=512, realistic inputs, floor 0.75 (healthy 0.84) |

## Review 2 — fp8 GPU wiring
Commit `b5c2ed8`. 3 findings, all fixed. (Load-bearing logic reviewed clean.)

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | e4m3 subnormal path clamped a rounded-up mantissa to 7 instead of promoting to the smallest normal (2^-6) | correctness (low) | promote `m>=8 → 0x08`, host + device |
| 2 | Host quantizer used `x*(1/scale)` (2 roundings) vs device `x/scale` → bit-exact invariant seed-fragile | correctness (low) | host divides too |
| 3 | `append_kv_fp8` assumed `kvl%128==0` & `>=128`; launcher guard incomplete | robustness | tightened kernel guard + `GpuEngine::new` ensure |

## Review 3 — MISA GPU device head-router
Clean. Zero findings. (Verified pool-warm-at-crossover, `m_blocks` rounding,
head routing `nact=h`, IndexShare, sync/borrow.)

## Review 4 — MTP M1 scalar oracle
Clean on code; 1 doc finding. Fixed in `6f81308`.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | `docs/mtp.md` described `enorm`/`hnorm` **backwards** from the (correct) code; would mislead an M2 kernel author | docs | doc corrected (e=embedding→enorm, h=hidden→hnorm) |

## Review 5 — MTP M2 device draft
Commit `c0a1b0e`. Numeric port confirmed exact; 2 defensive/resource findings, fixed.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | `mtp_draft` lacked the `pos<max_ctx` guard `forward()` has → M3 could OOB-write the MTP KV | correctness (latent) | added the guard |
| 2 | MTP scratch (esp. `mtp_lc`/`mtp_rc`) allocated unconditionally → non-MTP runs pay VRAM / risk OOM | resource | gated on `pin.mtp().is_some()` |

## Review 6 — MTP M3 batched-verify speculative loop
Two parallel finder passes (correctness/KV-bookkeeping + batched-MoE kernel).
Dense-bf16 path (the benchmark path) confirmed correct: KV/MTP-KV lockstep,
accept/reject rollback, trunk preservation, eos/ngen equivalence, union/weight
indexing, kernel stride/int4/reduce math. 5 findings, all fixed before commit.

| # | Finding | Severity | Resolution |
|---|---|---|---|
| 1 | `forward_batch` hardcoded bf16 KV path — `--kv-fp8` engine would reinterpret the fp8 slab as bf16 | correctness | `generate_spec` bails unless `!kv_fp8` (+ CLI fail-fast) |
| 2 | `forward_batch` hardcoded dense attention, ignoring `self.mode` — `--attn streaming/dsa/misa` would attend the wrong rows | correctness | `generate_spec` bails unless `mode==Dense` (+ CLI fail-fast) |
| 3 | Batched `sh` SwiGLU scratch omitted `.max(dense_inter)` moe_h has → latent OOB for models with large dense FFN (safe at GLM dims) | correctness (latent) | sized to `max(Emax*moe_inter, dense_inter)` |
| 4 | Comments over-claimed "bit-identical" output; batched MoE reduces experts in union order vs `forward`'s score-desc → ULP diffs can flip a near-tie | docs | reworded to "greedy-equivalent"; caveat in code + test |
| 5 | Near-capacity fallback (`pos+MAX_SPEC>max_ctx`) truncates instead of decoding to ngen | correctness (low) | left as a graceful stop (generate() itself bails there); documented |

**M3 measured result: gate NOT met.** spec 0.53 tok/s vs baseline 0.71 (64 tok);
43.2% accept; hit 69.6% vs 75.4%. Correct but slower — `forward_batch` gave up
cross-layer prefetch and baseline's prefetch dominates the batched-fetch win. Perf
follow-up (M4): restore prefetch in the batched path. See `docs/mtp.md` M3 status.

---

## Open / deferred — to discuss together

Not review findings, but decisions and loose ends worth a shared look:

- **512-benchmark mechanism (unconfirmed).** dsa/misa (1.28 tok/s) and fp8 (1.6)
  beat dense (0.80) at 512 tokens — entirely via the MoE expert **prefetch**
  path (hit 79→91→95%), not attention (attn flat ~204ms, routing identical). The
  control (dense re-run) proved it's **real, not warming**. The `--no-prefetch`
  A/B that would nail prefetch-timing as the cause was **interrupted** (~112/512).
  Rerun it, and understand *why* the indexer's extra compute helps prefetch — it
  implies an unclaimed **~1.6× prefetch-tuning win** dense could capture directly.
- **Sparse-regime (>2048) never run end-to-end.** dsa/misa's sparse **scoring**
  path is validated only at the kernel level + dense-fallback equivalence; the
  10k benchmark that would exercise it was stopped. No numerical cross-check vs
  the HF reference at long context yet.
- **M2 MTP experts are resident (~4.8 GB).** Deliberate for the M2 gate; M3 must
  switch them to **streaming** through the shared LRU pool (extend `moe_table`/
  `resolve_layer` to layer 78) — otherwise they waste VRAM and don't share the
  pool the batched verify needs.
- **MTP attention runs Dense.** Correct below 2048 (== the trained index_share
  top-k); wire **index_share reuse** for long-context drafts (`mtp.rs` DEFERRED).
- **fp8 uncalibrated.** vLLM measured MLA-fp8 accuracy drift without calibration;
  `--kv-fp8` is bf16-by-default and unvalidated at long context.
- **Standing pre-existing DEFERRED notes** (not new): attn.rs P1 (token-tiled KV
  re-read), P2 (per-layer KV slab), P3 (resolved-tensor table) — see PLAN.md.
- **Root install pending.** `out-idx`/`out-mtp` live in the `~/glm52-snap`
  overlay; `/var/db/llama-server/...` needs root (session key not on `root@rh-anine`).
