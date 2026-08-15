---
status: live
scope: engine
verdict: The gates-first rewrite is through M0, M1, and M2 — gates armed and red-proofed before any code, anchors vendored before the engine (GLM's generated fresh, 26/26 defect cells), substrate and artifact layers ported with both feature arms verified, converter round-trip proven byte-stable on a synthetic checkpoint, review round 1's 24 findings applied, and M3 closed on silicon — 38/38 kernel oracles green on gfx1151, floors measured at both draws, census armed with a shrink-only deferred table; M4's decode-loop design note awaits the owner before the loop is written.
---

# The rewrite, milestone by milestone

The approved plan (2026-08-15) rebuilds rivoli on this orphan branch: quality gates before
code, anchors before the engine, the six-crate DAG as the layering, GLM-5.2 first, then
serve → Glimmer → V4 → K3. The old tree at `wt/glimmer-s2` @ `6b7f496e` stays live as the
parity reference. This doc is the running record; each milestone gets a dated section
stating what shipped, what its exit gate showed, and what was deliberately not done.

## M0 — gates (DONE 2026-08-15)

Commits `b56ff21`, `e922a34`, `35132d6`. Workspace + jscpd + CodeScene 10/10 + docs
registry + derived exemption ledger + CI; anchors (Glimmer ×6, K3 ×2), frozen V4 oracle,
`bin/ppl` + corpora. Exit gate: `docs/measurement/gate-red-proofs.md` — every gate shown
red, two of them fired unplanted during the port itself (the jscpd fnv1a catch, the
ledger's 0→3 refusal). OWED: the CodeScene score-half red proof, blocked on
`CS_ACCESS_TOKEN`.

## M1 — substrate (DONE 2026-08-15, three items deferred)

Commits `4b5b3da`, `3c0f577`, `74bdb20`, `285e989`, `f071c52`.

- **Pure core:** `arena`/`cache`/`hybrid` ported verbatim; `residency::partition()` is P6
  as a signature (one author, prefix-shaped, monotone in `free`, refuses below floor with
  the arithmetic). INV-1 registered.
- **Backend waist:** `hip.rs` + `gpustream.rs` + `Signal`/`block_on`/`NULL_STREAM` + all
  11 kernels; both arms verified on the box (featureless = `abi` alone; `rocm` = 11
  kernels through hipcc, clippy clean). The three `repr(C)` mirrors moved to
  `backend::abi` with the layout assert. INV-4/INV-6 arrived with the port and the
  registry gate refused the tree until §8b documented them.
- **Engine shell begins:** `fetch/` (io_uring ticketed dataflow) + `telemetry.rs` ported;
  all five instrument features declared and forwarded; `feature-matrix.sh` (quick 9/9 on
  the box) + `matrix.rs` list-drift gate, red-proofed.

Exit gate: registry non-empty, both-direction check green, INV red-proofed
(`gate-red-proofs.md` §2b). All verification with explicit exit codes after two
false-green incidents (cwd reset to the main repo; `| tail` eating a red) — both recorded
in the session memory.

**Deferred from M1, deliberately — each to its first consumer, not dropped:**
1. **Legality table** (`decide(arch, flag)`) → M4, with the CLI whose flags it judges. A
   table over flags that do not exist yet would be invented rows.
2. **Gate-taxonomy / tolerance-with-provenance types** → M2/M3, with the GLM anchor and
   the first kernel oracle that load them. The shapes are sketched in the plan; building
   them unconsumed invites a redesign the first consumer would force anyway.
3. **proptest** → with the first property that earns a generator (partition monotonicity
   is currently swept deterministically over a range; proptest joins when the arena
   relocation properties port in M2+).

## M2 — GLM artifact + config + converter + the GLM anchor (DONE 2026-08-15)

Landed so far: `core::num` (conversions + `Scoring`), `artifact::{quant,format,schema,
arch,glm_config}` (commit `46f2153` — sniffing is identity-only, presentation policy
deliberately not ported), the three converters + `engine::device` (commit `73ecfa1`).
Still owed: the GLM anchor, artifact tests, tokenizer (deferred — coupled to
`dsv4_encoding`, arrives with the CLI at M4).

**Anchor scouting, settled 2026-08-15:**
- The fp8 source is `/swarm/storage/ai/openclaw/glm52-fp8` — `model_type: glm_moe_dsa`,
  **no `auto_map`**, so the first-party stack is transformers-native (no remote code).
- Fresh venv at `/home/rhansen/glm-anchor/venv`: torch 2.13.0+cpu, transformers 5.15.0,
  `glm_moe_dsa` in `CONFIG_MAPPING_NAMES`. CPU-only, so the anchor needs no GPU and no
  lock, same as Glimmer's.
- NOTE: the old `/home/rhansen/glimmer-anchor/venv` (pinned in the old tree's anchor.md)
  is GONE from the box — a Glimmer anchor regeneration would need its venv rebuilt at the
  pinned transformers commit fe747d88 first.
- Mechanisms the taps must capture, from reading `modeling_glm_moe_dsa.py`: MLA with
  q-LoRA (`q_a_proj→q_a_layernorm→q_b_proj`) and kv-LoRA (`kv_a_proj_with_mqa` split
  kv_lora_rank + qk_rope, `kv_a_layernorm` on the kv half only), **interleaved** RoPE
  (both attention and indexer — V3.2 uses half-split, this does not), `expand_kv`
  latent expansion, DSA indexer (own wq_b/wk/LayerNorm(eps 1e-6)/weights_proj; ReLU
  scores; head-weighted sum; `indexer_types` full/shared with `prev_topk_indices`
  cross-layer sharing), sigmoid router + `e_score_correction_bias` + group top-2 +
  norm_topk + `routed_scaling_factor` (router `weight` is ZERO-init — a tiny model must
  draw it, or every expert ties), MoE + shared expert, dense first `first_k_dense_replace`
  layers. Config `__post_init__` computes `indexer_types` from freq/offset and forces
  `head_dim = qk_rope_head_dim`.
- Tiny-config non-degeneracy plan (every width distinct, lesson 30): vocab 61, hidden 48,
  inter 96, moe_inter 24, layers 6 (2 dense + 4 sparse), heads 4, kv_heads 4, routed 10
  top-3, shared 1, kv_lora 20, q_lora 28, qk_rope 8, qk_nope 14 (sum 22 ≠ kv_lora),
  v_head 10, index_topk 4 (< PROMPT_LEN so the sparse path is exercised — the old dsa
  fast-path-below-topk lesson), index_head_dim 16, index_n_heads 2.

**Exit gate MET (2026-08-15):** anchor integrity green (`glm_anchor.rs`, 8 tests, byte
pins + derived census), defect matrix fully red-capable (13 × 2 = 26/26, regeneration
script green end-to-end), and converter round-trip byte-stable (`glm_convert.rs`: a
synthetic two-layer fp8 checkpoint — one dense + one MoE layer, every branch of the
tensor walk — converts twice to byte-identical artifacts, and a checkpoint without
`generation_config.json` REFUSES, closing the old tree's 56-run no-stop-tokens defect at
`finish_artifact` itself: aux copies are now errors, gated on the ARTIFACT after the
copy). Review round 1 (24 findings applied) sits between the anchor and this close.
Moved out of M2 with reasons: tokenizer (coupled to `dsv4_encoding` — M4 with the CLI);
`artifact_compat`/`arch_artifacts` byte-pin regressions against REAL artifacts (need the
real dirs and belong with M4's first real decode, not a synthetic fixture).

## M3 — GLM kernels via oracle TDD (DONE 2026-08-15)

- **Floors first** (commit `2121af0`): fp32 floors at both draws, 10 buckets; the DSA
  mask found exact-only; goldens re-pinned on eager experts.
- **Routing home** (`acfc2e1`): `core::routing` with INV-1 (inherited number and
  meaning); the P6 invariant renumbered to INV-8.
- **Oracle suites ported** (`18e3c61`, `92794e7`): kernel.rs + indexer_kernel +
  fwd_kernel + headtail + the generic device-test common half; engine::indexer.
- **DEVICE RUNS GREEN 2026-08-15 ~08:00: 38/38** — kernel 24/24 (184.65s, dev
  profile), indexer_kernel 5/5, fwd_kernel 4/4, headtail 5/5, each `--test-threads=1`
  under the flock, binaries built outside it. Run context recorded: an idle KFD tenant
  (`hiptest`, another session, 0.17 GB, 6h45m old, flock free) was present; these are
  correctness suites with no timing claims, so the runs stand — any red would have been
  treated as suspect-contention and re-run, none occurred.

- **Census armed** (`08af54b`): 53 launchers, 32 with oracles, 21 in a both-ends-checked
  DEFERRED table (6 → M7, 15 → M8) that can only shrink; red-proofed both directions.

**Exit gate MET:** every GLM-owned launcher passes its oracle on the device; census
green; floors recorded with provenance. One residual carried into M4, named rather than
dropped: cross-checking the ported oracle tolerances against the anchor-derived floors
(the ported suites carry the old tree's own measured tolerances, so the port is not
ungated — the floors are the second witness).

## M4 — the GLM decode loop (CLOSED 2026-08-15; design note below, evidence at the end)

The loop rewrite is the heart of this whole effort and the first piece with real design
freedom — everything until now was gates, anchors, and ports. Written down before any
code, per the plan's own discipline; the owner should react to this before `glm/loop.rs`
exists.

**What ports with confidence (the old loop's proven interior):**
- The per-layer kernel SEQUENCE (norm → MLA q/kv projections → rope → DSA select →
  attend → o_proj → router → expert descriptors → MoE accumulate → drain) — byte-level
  behaviour pinned by 38 device-green oracles and the anchor.
- The fixed-point u64 MoE accumulator (kernel-side; "no invariant to violate").
- The descriptor-array MoE launch and the staged-hop fetch destination.

**What re-architects (the new contracts, all already built and tested):**
1. Residency: `GlmPin` becomes a thin view over `core::residency::partition()` (INV-8) —
   the old `Pin` derived placement itself; the new one only EXECUTES a partition.
2. Fetch: the `hit`-mask-free ticketed dataflow ports, but `wait_on` consumes a typed
   `Pending` and slot refills carry the write-after-read fence per the s2 lesson.
3. Dispatch: `enum Engine` with `Engine::open` sniffing the artifact (schema::arch_of_named
   already refuses unknowns); `run_glm` is a real branch from day one — the old tree's
   500-line main-fallthrough shape is the named anti-goal.
4. Traversal: the loop body takes `Span{layers, x_off, tail}` + `Rows` from day one
   (layer-major prefill is a Span iteration, not a second engine; rows batching is not a
   retrofit — Glimmer's 72-minute prompt is the cautionary number).

**Design questions ANSWERED by the owner, 2026-08-15 ~17:30:**
- Q1: **Single format first.** int3-vq/int4 only at M4; hybrid returns as FormatPlan
  with its own INV re-armed.
- Q2: **`--attn dense` first.** Dense only at M4 — dsa follows it (the anchor pins dsa,
  so dsa is the first post-dense increment), streaming/misa after.
- Q3: **MTP deferred** past parity (M5+). The verify pass rides the Rows dimension,
  designed in from the start, so nothing is foreclosed.

M4's exit gate (unchanged from the plan): anchor decode gate green at the pre-measured
tolerance; INV-1 red-proofed live; release.yml on.

**M4 opened 2026-08-15 — the pool, redesigned rather than ported.** A first verbatim-shaped
port of `old:src/routed.rs` landed red (CodeScene 8.6: `submit` cc 18 / 110 LoC / 8 args,
`new` 73 LoC / 8 args) and was reset rather than patched, because the score was pointing
at a design fault, not a style one: `submit`'s eighth argument was `fmt: &mut
Vec<RoutedFmt>` — the exact channel of the old tree's #2 open defect, per-expert format
decided by which tier residency chose. The redo (`crates/engine/src/routed.rs`, 10.0)
makes the pool **single-format** per Q1: one `RoutedGeom`, `submit` fills a `ResolvedBatch` (né `SubmitOut`, renamed in the 2026-08-15 naming pass)
of slots + tickets only, and `RoutedPool::fmt()` is the only format answer — a per-expert
one is no longer expressible. Operands bundled as `PoolCfg`/`Selection`/`RankWindow`
(startup knobs vs per-layer selection vs trace-only inputs); the three arena phases are
named methods. INV-5 (ticket-per-descriptor, no residency mask) entered §8b with its
ported test. Suite green, device arm 60/0.

**GlmPin landed (a40e4c3): placement authored by `partition()`, executed by the pin.**
Units = routed experts layer-major; floor = resident footprint + slack + batch slots;
KV/scratch deliberately 0 (GLM's `--max-mem` has always budgeted weights only — folding
them in changes the flag's meaning and is owed its own measurement). `.f4` refused as a
V4 container at the one place GLM's format set is confronted.

**FIRST REAL DECODE, 2026-08-15 (`examples/glm_smoke.rs`).** The whole new stack —
GlmPin over `partition()` (5853 of 19200 experts resident at `--max-mem 100`), the
single-format pool, the dense loop — decoded `glm52-vq3-full` end-to-end on the DEV
profile with every `debug_assert!` live: prompt ids `[9707, 3837]` → greedy
`[17351, 198, 40, 2776]`, all finite, clean exit, 624 hits / 1176 misses. (0.02 tok/s is
a dev-profile NFS-cold number and is not a benchmark.) The tiny GLM anchor cannot drive
the device engine — its `kv_lora=20` violates the fp8-KV 128-block the real model's 512
satisfies — so M4's end-to-end evidence is this real decode plus the old-engine id
comparison (the M5 parity primary, pulled forward as a smoke): same ids through the
reference at the pinned SHA, compared with `--ids-out`.

**M4 EXIT EVIDENCE (2026-08-15), with the one substitution argued above** (the tiny
anchor cannot drive the fp8-KV device engine, so the end-to-end gate is the first real
decode against the pinned reference):

- **Token parity with the reference.** Old engine at the pin (built detached at
  `6b7f496e`, its own target dir), `--mode int3-vq --attn dense --no-mtp --max-mem 100
  --prompt "Hi"`: prompt ids `[154822, 154824, 154827, 13041, 154828, 154841, 154842]`
  → `[13041, 1052, 0, 358]` ("Hi there! I"). The rewrite on the SAME ids, dev profile,
  every `debug_assert!` live: **identical, token for token.**
- **INV-1 live (P4).** Same decode at `--max-mem 60` (3052 resident experts vs 5853,
  43.8 vs 83.8 GiB pool): **identical ids.** Residency moved bytes, never text. The id
  comparison can go red — the earlier different-prompt run's `[17351, 198, 40, 2776]`
  mismatches it — and the weight-perturbation red-proof is M5's, where the parity gate
  becomes a scripted test over longer runs rather than a hand-driven smoke.
- **release.yml** landed at M0 (self-hosted rocm+gfx1151 runner, tag-gated).

Deliberately OUT of M4, each with its recorded owner decision: MTP (deferred past
parity), dsa/misa (first post-dense increment), tokenizer + `enum Engine` + serve
(M6 with the CLI), hybrid (returns as `FormatPlan`), `Profile` (first benchmark).

**Loop staging (the M4 code as landed, in commit-sized steps, each green before the
next):** 1. `glm/desc.rs` — expert-descriptor builders, single format. 2. `glm/engine.rs`
— the engine struct + `new()` (KV slabs, scratch, streams; dense attention only).
3. `glm/forward.rs` — `Span` + the layer-major schedule + the dense forward pass over
the ticketed pool. 4. `glm/decode.rs` — argmax, greedy `generate`. DEFERRED with the
instruments: `Profile` (the old per-phase timer — M4's exit gate is anchor correctness,
not tok/s; it returns with the first benchmark), checksum-x/pred-probe/stale-sel
(feature-gated instruments, each behind feature AND flag when they land). The
old wedge watchdog existed to reap decodes hung by the gpustream bug class the rewrite
closes structurally (INV-4/INV-6: host-releasable waits — a dead producer cannot hang
the device). Its one obligation that survives is honesty about trace sinks:
`flush_trace`'s per-token flush now argues from `Drop` discarding errors, not from a
watchdog's `process::exit`. If a wedge class ever reappears, reopen this with the
evidence rather than resurrecting the module.


## The quality mandate (DONE 2026-08-15, owner-driven, same day as M0-M3)

Owner set three rules mid-stream: **CodeScene 10.0, whole tree, no exemptions** (then one
sanctioned: numerics.rs floor 9.6, measured unfixable without burying the transliteration);
**line caps** 800 soft / 1200 hard; scope extended to .hpp/.py. Executed as five parallel
agent waves (32 + 5 + 11 + 3 + 1 max-effort agents, one file or territory each, exclusive
territories, device/git forbidden, isolated target dirs) with coordinator integration
between waves. Final state: every source file at CodeScene 10.0 (one exemption row, three
deaths as its red-proof), longest file 785 lines, jscpd 0 with SIX ledgered exemptions,
39/39 device oracles green after every kernel-adjacent wave.

Things the waves surfaced beyond scores, each fixed and red-proofed in its commit:
- build.rs tracked only common.hpp after the header split — edits to sibling headers
  LINKED STALE KERNEL OBJECTS (the arch-staleness class again, one week after review
  round 1 caught its RIVOLI_OFFLOAD_ARCH sibling).
- CodeScene's measured mechanics, now written where they bind: Low Cohesion ~605
  non-comment LOC at LCOM4>=3; Primitive Obsession >=7 fns AND >=11 primitive args;
  arg counter skips tuple-typed params entirely (a false-green class, recorded, not
  exploited); comments are free.
- The old quant parameter-list jscpd exemption retired by the Fp8W/VqW/RowScaledW views
  — the exact hop its own note priced.
- Device-loop noalias strengthened, not lost, under the bundling the owner accepted:
  accumulators return by value (SROA to registers), read-only spans keep __restrict__ on
  members; rowview.hpp carries the table M5's benchmarks re-price.
