---
status: live
scope: engine
verdict: THE PLAN IS COMPLETE, M0-M9 — gates armed and red-proofed before any code, anchors before engines, and all four architectures decoding through one seam: GLM token-identical to the pinned reference (M4/M5's scripted parity gate), Glimmer reproducing its first-party goldens on silicon (M7), DeepSeek-V4 at forced-history parity 30/32 with both flips at near-ties (M8), and Kimi-K3 — the decode loop no tree ever had — closing M9 with the kernel census at 60/60/0, the KDA boundary at 2.265e-7 against the reference's own captures, and a five-row red-proof plan paid in full. The build is the rocm build; the deviceless arm is --no-default-features and a gate proves it.
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
`CS_ACCESS_TOKEN` — **PAID during the quality mandate** (the vendored `bad.rs.txt`
standing red-proof scores below ten on every armed run since; this line stayed "owed"
long after, corrected 2026-08-16).

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


## M5 — parity as a scripted gate (CLOSED 2026-08-15)

M4's hand-driven id comparison is now `tests/parity-glm.sh`: the reference binary
(release, the pin `6b7f496e`, prebuilt in its own target dir) against the rewrite's
`glm_smoke` (dev profile, every `debug_assert!` live), one invocation, both arms flock'd
with a contention witness each. Discipline encoded rather than trusted: the gate NEVER
builds (a build between arms evicts page cache); the witness is "KFD tenants not
descended from the arm's own pid" — never a binary-path whitelist, which would wave
through a peer running the identical pinned reference — plus a pre-arm GTT baseline
(KFD is blind to Vulkan tenants); prompt ids ride the reference's own tokenizer log with
the extracted count asserted against the advertised count (the log prints at most 12);
exit codes classify token mismatch (1) from setup (2) from witnessed contention (3,
discard and rerun). Parity means the reference's WHOLE sequence is a prefix of the
rewrite's: the reference chat-frames its prompt so its EOS is reachable, and the smoke
has no tokenizer until M6.

- **GREEN 2026-08-15.** `"The sky is blue"` → 10 chat-framed ids, ngen 32,
  `--max-mem 100`: **32/32 generated ids identical**, both witnesses empty.
  Supplementary: the arms' prefill counters are identical (3587 expert reads, 1492
  hits, 29.4%) — routing traffic matched, not just tokens.
- **RED-PROOF, a measured ladder** (each rung its own ~15-min GPU run): (1) a 1-ulp
  f32 flip in one codeword — **annihilated by the pin's f32→fp16 codebook narrowing**
  before any arithmetic ran, ids identical; (2) a sign flip of the same codeword
  (0.23 → −0.23, fp16-survivable, ~0.05% of one projection's weights) — **below
  greedy-argmax margins**, ids identical; (3) the whole gate-projection codebook
  sign-inverted — **diverges at token 1**, the output degenerates into a repetition
  loop, and the hit/miss split moves (3950/250 vs 3055/1145): wrong arithmetic
  cascades into ROUTING, because later layers route on a corrupted residual. The two
  sub-threshold rungs stay recorded as the gate's sensitivity floor: single-byte
  artifact corruption is invisible to an 8-token parity run — this gate detects wrong
  engines, not slightly-wrong artifacts.
- The loud-fail path fired live before the first green: run 1 failed prompt-id
  extraction (the reference logs on stdout; the script grepped stderr) with exit 2
  and a message naming the fix.
- Review round (correctness + simplicity, device use forbidden in both prompts):
  12 findings, 11 applied — descendant-pid witness, classified exit codes end to end,
  vacuous-red-proof guards (ngen ≥ reference count, non-empty prompt-ids, realpath'd
  shadow links), `decode_new` folded with an rc return, the locale-dependent
  cmp-parser dropped. Declined with the argument in place: a same-invocation pristine
  control — the green run IS the control, same binary and prompt, minutes earlier.
- **dNLL fallback not spent.** The primary token-identity gate passed at full length,
  so the paired-dNLL path (5000-token corpus) stays in reserve for the day a refactor
  makes exact identity too strict (a kernel reorder inside a documented tolerance).

Deliberately OUT of M5: multi-prompt sweeps (the gate takes `PARITY_PROMPT`; a corpus
sweep belongs to the first refactor that needs one), MTP (still deferred), dsa.


## M6 — tokenizer, enum Engine, serve, the thin CLI (CLOSED 2026-08-16)

Four commits, each agent-built in its own detached worktree and re-verified at
integration: the GLM tokenizer (chat framing token-identical to the reference, two
provenance legs — the reference binary's logged runs AND name-by-name resolution in the
artifact's own tokenizer.json; red-proven by one perturbed id reddening two tests), the
engine seam (`enum Engine<'a>`, main owns the borrow chain; the `(arch × flag)` legality
table in core with a four-way red proof, one rung unplanned: a flipped cell FAILS TO
COMPILE because the orphaned refusal message is dead code under warnings-deny), serve
(three files under the cap, zero cfg, zero Engine-arm matches; the port found clap
4.6.6's `requires` INERT — `--bench 4 --think` sailed through — replaced with
`conflicts_with` plus a parse test that reddens on the old spelling), and the exit gate.

**M6 EXIT EVIDENCE (2026-08-16).** `tests/smoke-glm.sh`, 12 cells green in one run:
six refusals firing with the legality table's own message fragments (hybrid, dsa,
streaming, mtp, and both clap exclusivities), bench decoding `[13041, 1052, 0, 358]` —
token-identical to M4's recorded reference run through the full
tokenizer→seam→engine chain with the same 1253/547 hit/miss split — and a live server:
port-opens-when-loaded readiness, /v1/models, a non-stream completion answering
`'Hi there! I'`, SSE frames with the `[DONE]` terminator, clean shutdown. Red-proofed:
a wrong message fragment reddens a refusal cell (run and shown), and the serve cell now
traps EXIT so a red cannot orphan a flock-holding server.

**Review round (correctness + simplicity on the whole M6 diff), applied:** the
featureless refusal moved to the actual door (`Engine::ensure_backend()` before the 19 MB
tokenizer parse — the seam doc claimed it, the call order had drifted; now proven live),
`--bench 0`/`max_tokens: 0` refused (the loop decides a token before checking the
budget, so zero generated ONE), `--trace` refuses under `--port` (one v2 trace across
many requests has no request delimiter), `encode_chat_continuation` deleted (its only
caller does not exist in this tree — the module's own port rule), `Flag::Prompt`/
`Flag::DumpIds` rows deleted by the table's own admission rule (`Ctx` stays: K3's
recurrent KDA state answers it differently in principle), the `--think` rationale folded
to its one home (`ChatOpts::thinking`), the eos test fixture salted by pid. Declined
with arguments in place: the `-bench` value-position rewrite (reference-faithful,
documented) and `room_for`'s conservative off-by-2 (reference arithmetic, errs safe).

**Known-inherited serve behaviors, recorded not fixed** (each verbatim from the
reference; fix on demand with its own gate): special tokens inside message content
inject real framing ids; a mid-request engine error closes the connection with no HTTP
error body; `chatcmpl-{created}` ids collide within one second; prose after a closed
`</tool_call>` reaches non-stream clients but not streaming ones.


## M7 — Muse Glimmer, the first dense model (CLOSED 2026-08-16)

Three agent-built pieces, each re-verified at integration. **Artifact side** (`74aa630`):
glimmer_config (found the reference's quantization_config guard checking the wrong
nesting level — K3's real checkpoint nests it inside text_config; caught by reading real
bytes, closed with both arms asserted), the string-rendering chat template with a 31-case
byte pin from the reference stack's own driver, convert_glimmer (eos_token_ids' second
caller arrives; json_truthy moves beside python_json — four port notes updated in place).
**Engine** (`03c4ba7`): the dense loop split by cohesion, P6 live (the floor asked twice,
slots=0 first, so a model that fits pays no phantom streaming slot; Glimmer's floor
charges KV+scratch where GLM's historical --max-mem contract passes 0 — the reference
shipped that KV uncounted as a live defect), kernel inventory CLEAN (12/12 launchers
already ported byte-identical; M7's real debt was the six deferred oracle suites — 25
tests, census 38 covered/15 deferred, all green on silicon), Engine::Glimmer with
ArchCfg/RoutedSpec (a dense arm filling a routed knob is the lie the table exists to
stop) and the Emit hoist (the emission protocol is not an architectural fact; loop
BODIES stay separate per the plan's measured decision). Legality row: --mode on a dense
model is FallbackLoudly, not Refuse, or the no-flags invocation dies — pinned by two
tests, red-proven.

**M7 EXIT EVIDENCE (2026-08-16).** `glimmer_anchor_decode.rs` + its deviceless widths
half: the rewrite's engine reproduces the first-party goldens — 14 logit cells + 16
per-layer branch cells + the P4 test (same prompt, roomy vs tight budget: bit-identical
logits, and the tight arm must actually stream) — **3/3 green on silicon in 0.59 s**,
both draws, 30-capture census asserted. Every tolerance carries provenance AND an
envelope gate: a bound outside 2–5× of its measured envelope reddens the deviceless
suite (catches the tolerance-picked-to-pass and the bound-under-its-floor both).
Red-proven on device on ALL FOUR recipe rows (observed magnitudes in
gate-red-proofs.md §4, each within 25% of old:'s prediction) — and the paying produced
two operator false-greens, both the same trap: a mutation that orphaned a variable,
warnings-deny failing the rebuild, and the build's exit eaten by a pipeline, so a STALE
binary ran. The tells are recorded: a red-proof that refuses to go red, and a red with
the WRONG failing test. Two mutations that provably cannot redden are recorded with
measurements in the module doc (eps_post swap: 0.2–0.6× the bf16 floor; softcap off:
TV unchanged) so nobody spends a GPU hour on them.

**Standing debts, recorded with owners:** `output_multiplier` has no value assertion
anywhere (argmax-invariant — no greedy gate can see it wrong; needs a logit-space gate);
`kernel_glimmer_attend.rs` at 796/800; the three oracle files each spell their own
fixture `draw` (shared home: tests/common — PAID at M8, `common::draws`/`fill`);
`old:tests/glimmer_tolerance.rs` never ported while `anchor.md` still claims its
enforcement — correct or port with M8's tolerance work; the real-checkpoint tensor-name
gate needs the shipped index.json (not on this machine).


## M8 — DeepSeek-V4-Flash, the second MoE arm (CLOSED 2026-08-16)

Commits `26a87b5` (artifact: `V4Config`, the dsv4 chat encoding gold-gated byte-for-byte,
`convert_v4`), `89799cf` (deviceless: geometry/rope/KV-compress selection), `5640d07`
(the arm: pin/engine/attn/kvcompress/moe/forward/decode, the third-arm factoring —
`crate::resident` as the one placement author, `seam::Emit`, pool-owned trace — and the
legality row atomic with the arm), `6e88f64` (nine kernel-oracle suites), `dad811c` (the
round's review batch). The bsz=1 scope cut is ONE `ensure!` at
`Extent::check_single_row_decode`, not 32 `debug_assert!`s.

**Census 15 → 0: 53 launchers, 53 with oracles.** The suites were authored deviceless
against the frozen oracle and paid first silicon contact here: 174 device tests green
across 24 binaries, after one real disagreement that the KERNEL won — the rope suite red
at err=3.886e-3 (one bf16 ulp at the fixture's scale) because the test's host reference
skipped the bf16 store rounding both the kernel and the oracle perform; the host rounds
now, and the round trip carries a format-derived 2^-8 bound instead of a ten-f32-ulp one
its bf16-quantized intermediate could never meet. The compress-defects registry's
two-message red proof was run as prescribed: a deleted `BELOW_RESOLUTION` row fails
`NOT RECORDED`, a number changed by one fails `the record is stale`, green restored at
the true record.

**Review round (correctness + simplicity, read-only, device forbidden): 9 findings, 9
applied.** The P1 was a defect-scope misclassification: `RopeBaseThetaEverywhere` was
excluded from the compressor sweep with a reason ("keys off a ratio-0 layer") true only
of its `if !compressed` twin — an excluded defect is never run, so the sweep could not
catch its own hole. Measured on device after reclassification: it separates above the
64-code floor on every cell. The two P2s were stale prose (the indexer suite disclaiming
an oracle defect fixed 2026-08-05) and a vacuous proof (fresh-state determinism claimed
as "reads no state" — now a NaN-poison bit-compare, plus the 256-token whole-multiple
path's first value comparison against the oracle). Simplicity: dead `placement` fields,
dead re-export, one dead defense, `tightest_ratio()` to one author, `a_kv_rows` carried
beside its allocation, compressor widths off `Dims`.

**Exit gate — parity against the pinned reference, by the reference's own standard.**
The anchor question closed as an M4-style substitution: V4's first-party stack is
tilelang + CUDA fast-hadamard (not runnable on this box), so the frozen oracle — itself
gated against the 167 GB checkpoint — plus end-to-end parity with the old tree stands in
for a first-party anchor. The rewrite's first V4 decode ran 2026-08-16
(`rivoli /var/db/rivoli/v4-f4-full --bench 32 --ctx 2048`, dev profile, 6.42 tok/s,
prefill 9 tokens — framing token-count-identical to the reference's). Free-run texts are
word-identical except a position-1 fork (`That's` vs `That is`), converging immediately.
Quantified with the reference's OWN drift instrument (a `teacher-forcing` build of the
pin, built into its own target dir so the pinned parity binary is untouched;
`-bench 32 --logit-dump --force-tokens <the rewrite's 32 ids>`): **30/32 positions the
reference's argmax IS the rewrite's token**; the two flips sit at top-2 gaps 0.21 and
1.40 against an agreeing-position median of 3.83, seven OTHER near-ties (< 1.5) did not
flip, and both flipped tokens are the reference's rank-1 alternative. Calibration is
`old:docs/investigations/v4-decode-decomposition.md` §M9's registered standard for
intra-engine numeric drift (17/512 flips, flip-gap median 0.099 vs 3.19 overall, max
|Δlogit| 8.14 — "the drift resamples ties, it does not degrade"): this cross-engine pair
sits well inside it. Exact token identity was never the promise across kernels that are
tolerance-gated rather than instruction-identical — M5 registered this exact fallback.

Deliberately OUT of M8, each named in place: MTP (still deferred everywhere), the scored
indexer selection (positional selection is the arm's declared deviation; the indexer
weights are counted but not placed), bsz > 1 (the `ensure!`), a rewrite-side
`--logit-dump` (the drift A/B above used the reference's; the rewrite grows one when a
refactor needs a same-engine A/B).


## M9 — Kimi-K3, the final milestone (CLOSED 2026-08-16). THE PLAN IS COMPLETE.

Commits `58aac6e` (backend: the KDA/SiTU/AttnRes/MHA kernels ported from `wt/k3-s1a`'s
S2 work; seven launchers into the census as live deferrals; the FP4 factoring's
bit-neutrality confirmed on THIS silicon by the V4 decode reproducing its exact 32 ids),
`9f61570` (artifact: `K3Config` with 20 red-watched gates, `convert_k3` grown a
trunk-shape confrontation the reference never had, the tensor-name census vendored — and
a stale gate claim found IN THE REWRITE and closed: `quant/naming.rs` citing a census
that existed only in another tree), `f29f922` (seven kernel-oracle suites, 43/43 on
first silicon, census **60/60/0**, tolerance table rule-gated at 10x-at-2sf over
measured floors, red proofs paid on two kernels plus a wrong-site plant that became its
own specificity evidence), `97b3de4` (the arm — see below), `e6e2773` + `1cc2a40` (the
round's two review batches). Mid-round the build itself changed shape: **the rocm fuse**
(`26423a0`, owner rule: `default = ["rocm"]`, a bare build IS the engine) and its leak
fix (`97b5116`: `--no-default-features` re-armed rocm through the dep edges until every
edge carried `default-features = false`; the matrix's new resolve cell — red-proven both
ways, then hardened again when review found its own false green).

**The arm is the first-authored piece: no tree ever had a K3 decode loop.** The old
branch stopped at anchor-scored kernels; the rewrite composed them — 69 KDA + 24 gated
MLA + latent MoE + AttnRes folds, token-sequential prefill, the seam's fourth variant,
the legality row atomic with it (4-arm agreement test). The twelve traps of
`k3:docs/reference/k3-architecture.md` §10 each carry a named owner in `k3/engine.rs`'s
header — a table review then improved: trap 11's claimed owner did not bind (two
`Vec<f32>` cannot argue), and the header now says the one call site is load-bearing text.

**Exit gate, first silicon 2026-08-16, red-proof plan paid in full.** The KDA recurrence
boundary against the reference's own captures: worst disagreement **2.265e-7** vs the
`kda_op` bound 6.3e-4, non-zero by the anti-vacuity assert. The composition,
structurally, on a synthetic F4-legal tiny model through the real parser and converter
primitives: P4 bit-identity across pool budgets that provably differ (147456 vs 24576 B,
16 tight-arm misses), second-generate identity, carried-state == replayed-prefix. First
contact found one real defect (the synthetic config lacked the top-level wrapper pair
the sniff requires — the parse sits in a binary its author could never run). All five
red-proof rows observed: freshness (source refuses; no stale green), transpose dropped
1.300e0, k/v swap 7.041e-1, the lb guard's 1006 — whose first plant orphaned its
variable and was itself refused by warnings-deny — and the budget discriminator.

**Reviews: 5 correctness + 15 simplicity findings, 20/20 applied.** The P1: the MLA
caches were allocated uninitialized while the mask argument held only for FINITE garbage
— NaN in recycled device memory poisons the softmax denominator; zeroed once at
allocation now. The rest: the lb door with a red-watched test, the resolve cell's
non-empty guard, `worst_rel`'s zero-elements refusal, and the simplification batch
(`legality.rs` 897 → 575 via the tests split; `score_all` closing its own stated
invariant; `to_key_major`'s lost length assert restored).

**The two anchor vendorings, reconciled by role.** S1b (226 tensors, `crates/oracles`)
backs the anchor-integrity suite, the tolerance table, and the decode gate's KDA
fixtures; the S2 recapture (290 tensors, `crates/engine/tests/k3/`) carries the 64
operator captures five kernel suites need and S1b never had. Both FNV-pinned, consumers
disjoint, the kernel-side table documenting where it diverges. **Standing debt, named:**
the oracles table's older rows are one-draw floors (its own doc says so); the both-draw
refresh pairs with its measurement doc and belongs to whoever next touches the anchor
suite. Also standing: `moe_fixed`'s ±2^19 range is inherited from V4's precedent with no
K3-specific measurement yet.

Deliberately OUT of M9, each named in place: a real-checkpoint decode (no K3 checkpoint
on this box; the pin reads ~1.3 TiB when one arrives, `--ctx ≤ 8192`), the three
declared performance costs (shared-MLP stream overlap, the strided attend that kills the
masked-full-width waste, chunked-KDA prefill with the UT-transform inverse), a chat
encoding (`--port` refuses), MTP.

> **CORRECTED 2026-08-16.** The chat-encoding item read "(`--port` refuses; none exists in any
> tree)". The checkpoint itself disproves the parenthetical: Kimi-K3 ships `encoding_k3.py`, a
> 647-line first-party XTML renderer. What it does not ship is a `chat_template`, which is the
> narrower true claim. `--port` still refuses and should — porting the encoder needs an
> id-pinned golden against its own output, and the tokenizer it would encode through does not
> exist here either (K3 is tiktoken; there is no `tokenizer.json`). Found on first contact with
> the real checkpoint, which also corrected the same sentence in `main.rs` twice over —
> `docs/investigations/k3-first-checkpoint.md` §3, §4.


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
