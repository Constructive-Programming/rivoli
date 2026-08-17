---
status: live
scope: glm
verdict: GLM int3-vq greedy decode does not reproduce its own generated TEXT — byte-identical at 32 tokens, 61 of 512 ids differing on a quiet box and 496 of 512 under load, first difference anywhere from position 13 to 452, and the pinned OLD tree is worse (247/512), so it is neither a rewrite regression nor the ROCm 7.14 upgrade. BOTH previously named candidates are now REFUTED BY CONSTRUCTION in this tree, which is the milestone's result. MoE accumulation cannot be the mechanism: `moe_fixed` saturates per term, the accumulator is a wrapping u64 `atomicAdd`, and the drain sums the two lane blocks at a fixed stride, so the resident/miss lane split is exactly equivalent to one accumulator — verified at the kernel, not inherited. Arena-relocation-vs-in-flight-read cannot be the mechanism either: `run_layer` ends EVERY layer with an unconditional `hipDeviceSynchronize`, every miss kernel sits behind `hipStreamWaitValue64` on its ticket, and `launch_moe` host-awaits both lanes, so no read outlives its layer and `relocate`'s blocking `hipMemcpy` cannot race a fetch. Three independent read-only audits of the sync substrate, the host residency path and the whole non-MoE GLM path found NO live ordering hole, no float atomic in any kernel, no hash-order or time-derived decision, and no read of unwritten memory — so the divergence must be a WRONG-BYTES READ, not a reordered sum. It is not statically visible, so the deliverables are instruments: `--divergence-log` (forward-ported and extended from archive commit 544fea7) localises a divergence to a (position, layer, QUANTITY) coordinate and splits the layer at the two seams the candidates sat either side of; `tests/determinism-glm.sh` is a length-aware gate with a 256-token floor, because a 32-token determinism gate is green on this engine. GPU time is now the binding constraint, and the first experiment is FREE — the pool's READ-BEFORE-WRITE detector is already live and unconditional in every build.
---

# GLM does not reproduce itself, and both named suspects are innocent

**Inherited from `wave/m10-spine`'s doc of the same name**, which measured the defect and
bounded it. That page's scope statement — "the wobble is confined to the routed expert pool
… per-expert admission, the two-ended arena's relocations and ticket lifetime, MoE
accumulation … those two candidates are NOT separated" — is what this page revisits, and the
answer is that **neither of them can be the mechanism in this tree.** The measurements on
that page stand unchanged; what changes is what they can be attributed to.

## The measurement, unchanged and restated so this page is self-contained

Artifact `/var/db/rivoli/glm52-vq3-full`, one prompt (`tests/bench-matrix.sh`'s default
essay prompt, ~71 prompt tokens), `--bench 512 --mode int3-vq --attn dense --max-mem 115`,
`--dump-ids` compared body-only, 2026-08-17:

| runs | result |
|---|---|
| rewrite, no-MTP × 2, box under heavy CPU/NFS load | **496 of 512 differ, first at 13** |
| rewrite, no-MTP × 2, quiet box (114 GiB free) | **61 of 512 differ, first at 452** |
| OLD pinned binary (`ref-pin` @ 6b7f496), `--no-mtp`, quiet box | **247 of 512 differ, first at 265** |
| rewrite, no-MTP × 2, **32 tokens** | **byte-identical** |
| rewrite, `--mtp` vs no-MTP, 32 tokens | **byte-identical** |
| Glimmer, teacher-forced × 2, fully pinned | **byte-identical** (PPL 7.008490 twice) |
| Glimmer ids, `--max-mem` 32 (52/52 pinned) vs 20 (21 of 52 streamed, 1 slot) | **byte-identical** |
| V4 ids × 2 at 97% hit | **byte-identical** |

Contention amplifies and quiet does not cure. **The length dependence is the single most
important row**: it is why every gate over this property has to state its token count, and
why a short gate is not a conservative gate but a vacuous one.

## What is now REFUTED, and by what

Both refutations are structural — read off the code and the kernels, not off a run — which
is why they are stated as refutations rather than as further exclusions.

### MoE accumulation order cannot do it

`docs/reference/architecture.md`'s standing claim is that fixed-point accumulation makes the
MoE order-independent. That claim was **checked in this tree rather than trusted**, because
the brief asked for exactly that:

- `common.hpp::moe_fixed` clamps **each term independently** (`llrintf(fmin(fmax(v, ±MAX)) ·
  2^44)`), so saturation cannot depend on the order terms arrive in.
- `moe.hip:124`/`:371` accumulate with `atomicAdd` on `unsigned long long`. Integer addition
  is associative and commutative, and the width argument in `common.hpp` bounds `Σ` over
  ≤16 clamped terms at `2^62` — a full binade of slack, so no wrap occurs either.
- `moe_acc_drain_impl` sums the `MOE_ACC_ROWS` lane blocks into a `long long` at a fixed
  stride and converts **once**, in `double`.

Therefore the resident-lane / miss-lane split (`glm/mlp.rs:281-290`, two streams, two
accumulator blocks) is *exactly equivalent* to one accumulator: which lane a contribution
lands in is a residency-dependent decision that **provably cannot** change the sum. The
launch batching of maximal resident runs is the same argument. Nothing in the MoE's
accumulation is schedule-sensitive.

Stronger, and this is the load-bearing generalisation: **there is no float atomic in any
kernel in the tree.** `grep atomic kernels/*.hip *.hpp` yields the u64 MoE `atomicAdd`, the
`atomicCAS` on the non-finite diagnostic flag, and a u32 histogram in the DSA indexer that
`--attn dense` never launches. The split-KV attend's cut is a pure function of `(H, nr)` and
both its kernels are on stream 0, so the combine cannot race the partials. **Given identical
input bytes, this engine's arithmetic is bit-reproducible.** A divergence therefore has to be
a wrong-bytes read.

### Arena relocation vs an in-flight read cannot do it *in this tree*

The prior hypothesis — a read outliving its layer and having its slot `memcpy`'d out from
under it, with pins not stopping compaction, seen as 9 corrupted reads in 8452 at
`--max-mem 30` — describes the OLD tree. In this one three barriers close it, and they
compose:

1. `glm/forward.rs::run_layer` ends **every** layer with `device_sync()`, which is
   `hipDeviceSynchronize` (`linalg.hip:404`) — the whole device, all streams.
2. Every miss kernel is enqueued behind `hipStreamWaitValue64` on its ticket
   (`asyncfetch.rs::wait`), and the reaper signals that timeline on the fetch stream only
   *after* enqueueing the bounce→slot copy. So a miss's bytes have landed before its kernel
   runs.
3. `launch_moe` host-awaits BOTH lanes with `hipLaunchHostFunc`-backed signals before
   returning, so all of layer L's copies have executed before layer L+1 submits anything.

Consequently, when `admit_misses` evicts, frees and relocates, **the device is idle**. And
`submit`'s phase order already forbids the remaining shape: every relocation happens in
`admit_misses`, and `resolve` computes final slots and issues reads only afterwards, so a
read never targets a slot that later moves. `relocate`'s `memcpy_dtod` is a blocking
`hipMemcpy` (`hip.rs:343`), which is enough precisely *because* nothing else is in flight —
note that it would NOT be enough otherwise, since rivoli's streams are `hipStreamNonBlocking`
and the null stream carries no implicit ordering against them.

## What three independent audits found, so the next reader does not repeat them

Three read-only audits ran on 2026-08-17 over (a) the sync and memory-visibility substrate,
(b) the host residency path, (c) the whole non-MoE GLM path. Their negative results are the
useful part:

- **No hash-order, time-derived, address-derived or unstable-sort decision** reaches an
  eviction, a tier, a relocation order or a read order. The one `retain` over a
  `HashMap` (`hybrid.rs:252`, the LFU halving) is order-INVARIANT: its closure reads and
  writes only the entry's own value. Recency ordering is a `BTreeMap` on a monotonic tick.
  `route_into`'s comparator carries an index tiebreak, so top-k ties resolve to lowest index.
- **The `Arena`'s relocation sequence is a pure function of its `(alloc, free)` sequence.**
  Its free lists are `Vec` with `pop`/`swap_remove`: history-dependent, fully determined.
- **Argmax is a single-block reduction** (`dim3(1), dim3(256)`) whose combine resolves an
  exact tie to `min(index)` explicitly, with `__syncthreads` between rounds and no atomics.
  A tie cannot resolve differently across runs. Do not revisit this.
- **No buffer in the GLM path is read before it is written**, padding included: every
  producer and consumer is parameterised on `nrow` (the row count is a *template* parameter,
  never a grid dimension), `moe_acc` and `argmax_dev` are explicitly zeroed at construction,
  and `moe_hidden`'s two lanes index by the ABSOLUTE descriptor index so they are disjoint.
  The one full-buffer D2H of a half-uninitialised staging buffer (`gate_logits_host` at
  `nrow == 1`) is consumed only in its written half.
- **The KV cache reads exactly `pos + r + 1` rows**, no padding, and the split plan is a pure
  function of `(H, nr)`.

So there is no live ordering hole and no unwritten read. **The defect is not statically
visible**, and that is why this milestone's deliverable is instruments rather than a fix.

## The instruments

### `--divergence-log` — a coordinate AND a mechanism

Forward-ported from archive commit `544fea7` (reachable from
`archive/belady-residency-bound`), not cherry-picked: that commit's `--checksum-x` and
`--checksum-route` were written against a single-file `gpu.rs`, and both are folded here into
one flag and one file format. Its two load-bearing design decisions are inherited verbatim
and are the reason it can be pointed at this bug at all:

- **The fold is XOR** (`kernels/fwd.hip::hash_rows`, splitmix64-finalized over
  `(index, exact bits)`), because XOR is commutative *and* associative and so is bit-identical
  whatever order the atomics land in. A float sum would report a difference from scheduling
  jitter alone.
- **Nothing touches the host or the disk mid-run.** The predecessor copied the residual to
  the host every layer and produced a CLEAN run on a configuration that reproduced without
  it — the tool built for the bug could not be used on it. Here the folds stay on the device,
  all slots drain in ONE D2H per pass at a point the end-of-layer sync has already idled, and
  the records are written after the last token.

What is **new** is that it is a discriminator rather than a localiser. Three quantities per
layer cut the layer at the two seams the refuted candidates sat either side of:

| column | quantity | a difference here, with the earlier columns equal, means |
|---|---|---|
| `xn` | the MoE's input (post-attention rmsnorm) | attention or its KV cache; the MLP has not run |
| `h` | `moe_hidden`, the SwiGLU intermediate | the gate/up expert BYTES, or that kernel |
| `x` | the residual at layer exit | the down projection, the accumulator or the drain |

plus six host columns that cost nothing because routing is already a host function of
host-resident data: `gl` (what the router saw, FNV-1a over exact bytes), `pk` (what it
picked), `wx` (the routing-weight matrix), `sl` (WHERE the pool put each expert — arena
**offsets**, never addresses, since the VMM base differs per run), and the layer's `misses`
and `relocs` deltas.

Diff two logs: the first differing LINE is the coordinate, the first differing COLUMN names
the mechanism. `crates/engine/tests/fwd_kernel.rs::hash_rows_matches_the_host_fold` scores
the kernel against `probe::fold_host` and asserts the three properties the instrument's
usefulness depends on — bit-exactness, one-ULP sensitivity, and permutation sensitivity —
because every conclusion below will be read off a pair of these hashes.

**Do not run it under `--trace`, and do not accept a green obtained with tracing enabled.**
`trace` adds a poison fill and a `device_sync` per layer-with-misses, which is the class of
perturbation that masked this fault before. `--divergence-log` is its own feature
(`corruption-probe`, which `trace` implies) for exactly that reason.

### `tests/determinism-glm.sh` — length-aware, with a floor

Two runs, one binary, byte-identical arguments, ids compared. **`ngen` has a hard floor of
256** and the gate refuses below it with the reason: the defect is byte-identical at 32
tokens on the very tree that fails at 512, so a short determinism gate is green on a broken
engine. The default is 512, the length every recorded measurement of this defect used. A
green **bounds the rate at that length; it does not prove determinism**, and the gate says so
in its own output.

It carries a contention witness per arm (KFD holders resolved by descent from the arm's pid,
GTT baseline for tenants KFD cannot see), never builds, and refuses a short arm — two short
arms of equal length would otherwise be a green over a decode that never happened.

Red proof, both halves:

- **Mechanical, deviceless:** `tests/determinism-glm.sh --self-test` feeds the gate's own
  comparator two id streams that differ by one token, and a truncated stream, and fails if
  either compares equal. Run 2026-08-17: reddens on both. This is a proof about the
  comparator, not about the engine, and is labelled as such.
- **Live:** the defect is present, so the 512-token arm on this tree IS the red proof, and
  the table at the top of this page is its record. The gate's green after a fix is believable
  because the same command was red on the same tree.

## The next experiments, in order, and the first one is free

1. **Grep the diverging runs' stderr for `READ-BEFORE-WRITE`.** `routed.rs::touch_hits`
   carries a live, **unconditional** detector for "the policy counted this expert a HIT but
   its bytes never landed since admission" — it is not feature-gated and it was active in
   every run in the table above. If it fired, the coordinate is already recorded. If it was
   silent, the premature-read half of wrong-bytes is excluded at zero cost. **Its blind spot
   must be stated with the result:** it detects bytes that never arrived, never bytes that
   arrived and were then clobbered.
2. **`--divergence-log` on both arms at 512 tokens**, on a `--features corruption-probe`
   build, quiet box, flock + witness. Diff the two logs. The first differing column decides
   between "the expert bytes were wrong" (`h` moves with `xn` equal) and "the arithmetic
   after them was" (`x` moves with both equal) — and if `xn` moves first, the whole
   routed-pool scope statement inherited from `wave/m10-spine` is wrong and attention is in
   frame.
3. **Log `slot_stalls()` per token on both arms.** It should be identically 0: the per-layer
   `device_sync` makes every staging slot landed at `submit`, which is what INV-9 rests on. A
   non-zero count means the two runs stopped being the same program, and it is the only
   host-side amplifier on the path — `scan_free` advances its cursor past every candidate it
   tests, so ONE skip offsets the hand-out for the rest of the run and the two arms never
   re-converge.
4. **`--max-mem` high enough that the routed pool never evicts**, if such a budget exists on
   this box. Its control pair going byte-identical would put the fault in the fetch path;
   still diverging would take the pool out of frame entirely. P1 says the routed experts do
   not fit, so this may need a shrunk shadow artifact.

## Two defects found on the way, neither able to explain a rare divergence

Recorded here rather than dropped, and both are latent-not-live:

- **`asyncfetch.rs`'s `debug_assert_eq!(sub, 0, "VQ expert read must be block-aligned")` is
  compiled out under `--release`, and it is the only thing "checking" a claim the fetch path
  depends on.** If a read's file offset were not `ALIGN`-aligned, `reap` would copy the
  aligned superset to the slot BASE and every projection pointer would be off by `sub` bytes.
  It is currently guaranteed by arithmetic rather than by the assert: expert offsets are
  `VQ_ALIGN + e·stride` with `VQ_ALIGN == ALIGN == 4096` and `stride` a multiple of it, so
  `sub == 0` and `nbytes == stride` exactly — the copy neither underruns nor spills into the
  neighbouring slot. A spill there would be precisely a timing-independent cross-expert
  corruption, so this is worth an `ensure!` at pool construction rather than a
  `debug_assert!` at every read. **This is the repo's most common review finding wearing a
  new hat** and it is in the fetch path of the defect under investigation.
- **Nothing asserts `lm_head.o_dim == cfg.vocab`.** `tail()` writes `logits` at width
  `o_dim`; `argmax_rows` reduces over `cfg.vocab`; `logits` is allocated `MAXROW · vocab`. A
  smaller `o_dim` makes argmax reduce over uninitialised device memory; a larger one writes
  past `logits` into the buffer `hipMalloc` handed out next, which is `argmax_dev`. Inert for
  the real artifact in either direction, and one `ensure!` to close.

Also corrected while reading: `glm/engine.rs`'s comment describes `moe_acc` as
`[MOE_ACC_ROWS][MAXROW][hidden]`, but the live lane stride is `nrow · hidden`
(`mlp.rs:279`, matching the drain's own `n`). Writer and reader agree and every drain zeroes
exactly what it reads, so it is correct — but a third consumer trusting the comment would
alias.

## What this costs the record, unchanged from the inherited page

The old tree's own byte-identity claims — gated MTP at `--mtp-min-conf 0.8`, the parity
gates, the quality ladder's A/Bs — were measured on an engine that does not reproduce itself
over long runs. They are not thereby wrong; they are **unproven at any length where a
divergence event is likely**. Any future byte-identity claim on GLM must state its token
count. `wave/m12-glm-chain`'s MTP losslessness gate is the immediate casualty: at 70.6%
acceptance it is well past break-even and cannot be closed until the baseline reproduces.
