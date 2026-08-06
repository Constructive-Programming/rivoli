---
status: closed-negative
verdict: RETIRED 2026-08-06 as an unfinished port, not a feature — 16 of 29 kernels, 6 of 36 mode-matrix cells decoding and 30 refusing, ~1.9x slower on --mode int3-vq --attn dense, and no DeepSeek-V4 path at all. Kept as the inventory of what was and was not ported, the numerics and index-width rules, and the mechanised-guard registry; code at tag archive/vulkan-backend-hb16.
---

# rivoli — Vulkan kernel inventory

> **RETIRED 2026-08-06. This file was `reference/vulkan-kernels.md` and `status: live`
> until then; it moved here the day the backend was deleted.**
>
> Nothing below describes code in the tree. `src/backend/vk.rs`, `src/backend/vkstream.rs`,
> `kernels/vk/*`, `tests/vk.rs` and `tests/glsl_numerics.rs` are preserved at the tag
> **`archive/vulkan-backend-hb16`** (which supersedes `archive/vulkan-backend`, cut one
> commit earlier and missing the HB=16 work).
>
> **Why it was retired, measured at retirement:** 16 of 29 kernels ported; `tests/mode-matrix.sh`
> ran 6 of 36 cells to a decode and 30 refused at startup; ~1.9x slower than ROCm on the one
> configuration it could run (`--mode int3-vq --attn dense`); `--mode int4`/`hybrid` and
> `--attn dsa`/`misa` refused at startup; and no DeepSeek-V4 decode path at all. Every V4
> launcher signature change cost a parallel edit to a backend that could not use it. The
> user's decision was explicit: *"we won't use vulkan moving forward."*
>
> **Kept, rather than deleted, for what it rules out.** Two things here outlived the code
> and are the reason to read it: the **numerics and index-width rules** (what a `.comp`
> shader must do to agree with a HIP kernel bit-for-bit) and the **two OPEN fp8-dot gaps**,
> which were never closed. Anyone porting these kernels to a third API starts here rather
> than rediscovering them.
>
> The port *journal* — four phases, and everything that was tried and rejected — is
> [`vulkan-port.md`](vulkan-port.md).

## Kernel inventory — port 16 of 29

v1 is **`--mode int3-vq`, `--attn dense`, decode only**. That is the smallest thing that
runs the model, and it cuts 13 kernels from the port.

| File | Port now | Port difficulty |
|---|---|---|
| `linalg.hip` | `gemv_fp8`, `gemv_fp8_splitk`, `gemv_f32`, `gemv_i8`, `swiglu`, `rmsnorm`, `rope_interleave` | `gemv_f32`/`swiglu`/`rmsnorm` trivial; the split-K LDS combine and the e4m3 LUT need care |
| `moe.hip` | `moe_gateup_vq`, `moe_down_vq`, `moe_acc_drain` | hardest — per-expert device addresses, fp16 codebook gather |
| `fwd.hip` | `embed_i8_row`, `append_kv`, `gather_rope`, `vadd`, `argmax_reduce` | easy |
| `mla.hip` | `mla_absorb_fp8`, `mla_value_fp8` | easy-moderate |
| `attn.hip` | `mla_latent_attend`, `mla_attend_combine` | moderate — dynamic shared memory sizing becomes a specialization constant |

> **CORRECTED 2026-08-01.** The `moe.hip` row read `moe_reduce`, which no longer exists on
> either backend — the HIP kernel, its C wrapper, both launchers, the Vulkan push struct and
> `kernels/vk/moe_reduce.comp` were all deleted that day. Fixed-point accumulation
> (`architecture.md` §12) replaced it on 2026-07-31 and it had been off the decode path ever
> since; the last two claims on it — "the f32 reference the oracle tests pin against" and
> "the shape `tests/vk.rs` probes cross-queue visibility with" — both moved to
> `moe_acc_drain`, which is the shader that is actually ported and dispatched. **One-for-one
> substitution, so the 16-of-29 count is unchanged.**

**Deferred, with the reason:**

- `vq_encode` (`linalg.hip`) — converter only. `convert --gpu` stays a ROCm-build tool;
  a Vulkan box converts on a HIP machine or on CPU.
- `gemv_vq`, `gemv_i4` (`linalg.hip`) — standalone microbench/oracle kernels. Decode
  goes through `moe_*_vq`; nothing in the forward pass calls these.
- `moe_gateup_i4`, `moe_down_i4` (`moe.hip`) — only needed for `--mode int4|hybrid`.
- `indexer.hip` ×5 plus `layernorm` (`linalg.hip`, the indexer's k_norm) — the DSA path.
- `vaxpy` (`fwd.hip`) — reached only by `--moe-gain != 1`, an experiment knob. `vadd`, the
  g = 1 case, IS ported and is what every normal decode uses.

**Ported but SINGLE-ROW, as of 2026-07-31 — a third category this inventory did not have.**
`gemv_fp8` (+split-K), `gemv_f32`, `gemv_i8`, `mla_absorb_fp8`, `mla_value_fp8` and
`moe_expert_range` gained an `nrow` argument on the HIP side for the speculative verify pass
(ARCHITECTURE §13). The `.comp` shaders have no row axis, so all six accept `nrow == 1` and
return `Err` above it. Consequence: **speculative decode is ROCm-only.**

That is NOT enforced by `validate_backend`, and the asymmetry is deliberate but thin. The
13 deferred kernels are gated at startup because `--mode`/`--attn` name them directly; the
row count is a property of the artifact (does it carry an MTP head?) rather than of a flag,
and it is only reachable at all because the Vulkan default `--mode int3-vq` is the mode the
head rides. A Vulkan run against a head-carrying artifact therefore fails at the first layer
of the first token rather than before the artifact is opened. If that combination becomes
expected rather than incidental, move it into `validate_backend` — the message already names
`--features rocm`.

**How the deferral is enforced, as of increment 1.** Two layers, and the order matters:

- `Config::validate_backend` refuses `--mode int4|hybrid` and `--attn dsa|misa` at
  STARTUP, before the artifact is opened. `main` refuses `--moe-gain != 1` beside it. Each
  message names the missing kernels and says `--features rocm`. This is what a user hits.
- The launchers themselves return `Err` — **not a no-op returning `Ok`**. That distinction
  is the whole reason they exist: a no-op `layernorm` leaves the DSA indexer selecting rows
  from unnormalised keys, and a no-op `moe_expert_range_i4` leaves the partial slab holding
  the previous token's experts. Both produce plausible numbers and no diagnostic.
  `tests/vk.rs::deferred_launchers_refuse_rather_than_no_op` is the gate on that.

`vmm.hip` and `async.hip` are runtime shims, not kernels — replaced by the Vulkan
memory/queue layer rather than ported. Two exceptions landed in increment 1 because the
integration path calls them: `vmm.hip::fill_u32` became `vkCmdFillBuffer` (that command IS
the kernel, so there is no GLSL twin to write), and `fwd.hip::flag_nonfinite` was ported as
a real shader — stubbing the NaN localizer would make it report tag 0, "no layer was
non-finite", on a run that went non-finite.

**Shader language: GLSL compiled by `glslc`.** The kernels are already C-like; GLSL
keeps the diff readable against the `.hip` originals, which matters because the two
must stay numerically identical. Slang would also work but adds a toolchain nobody
here has. `build.rs` gains a `vulkan` arm compiling `kernels/vk/*.comp` → SPIR-V,
embedded via `include_bytes!`. Keep `rerun-if-changed` on the shared header — the
`common.hpp` staleness bug (see git history) is easy to repeat with `#include`d GLSL.

## Device requirements

Moved here from `investigations/vulkan-port.md` on 2026-08-01: this is what the backend
demands of a device *today*, not a decision the port once made, and twelve shaders cite it.
Init fails fast with a clear message naming the missing one.

- `VK_KHR_buffer_device_address` (core 1.2) — lets `ExpertDescVq`'s six raw pointers stay
  six `uint64` addresses in a params buffer instead of six descriptor bindings. Without it
  the per-expert launch path needs a descriptor rewrite per expert.
- `shaderInt64` (core 1.0) — the other half of the above: the shader dereferences those
  addresses as `uint64_t` buffer references (`GL_EXT_buffer_reference`), a 64-bit integer
  operation.
- `VK_EXT_subgroup_size_control` with `requiredSubgroupSize = 32` — the kernels assume
  `WAVE 32` (gfx1151 native wave32; see `kernels/common.hpp`).
- `subgroupShuffleRelative` + `subgroupBasic` (core 1.1 subgroup ops).
- `VK_KHR_shader_float16_int8` + `VK_KHR_16bit_storage` — the fp16 VQ codebook.
- `VK_KHR_timeline_semaphore` (core 1.2).
- A `DEVICE_LOCAL | HOST_VISIBLE` heap covering GTT — on Strix Halo host RAM *is* GPU
  memory, and the pin path depends on writing resident weights directly.

**Deliberately NOT required: `VK_KHR_8bit_storage`.** Read packed u8 weights as `uint`
words and unpack bytes/nibbles in the shader. The hot loops already unpack manually, so the
extension buys nothing — which is why four shaders say "deliberately not required" rather
than leaving a reader to wonder whether it was an oversight.

Use a **dedicated compute queue**, not the universal graphics queue.

### Index width

GLSL `uint` row arithmetic wraps at 2^32 elements = 4.29e9. `gemv_f32` (router gate,
256x5120 = 1.3e6) is nowhere near it, but `lm_head` via `gemv_i8` is 151552x5120 = 7.76e8 —
only 5.5x of margin, and it would wrap SILENTLY into another allocation. Use `uint64_t` row
indices from `gemv_i8`/`gemv_fp8`/`moe_*` onward
(`GL_EXT_shader_explicit_arithmetic_types_int64` is already enabled and `shaderInt64`
already required) and **record the ceiling per kernel as you port** — `append_kv`,
`embed_i8_row`, `gather_rope`, `mla_absorb_fp8` and `mla_value_fp8` each carry that
computation in a comment, which is the whole mechanism: there is no check, only the habit.

## Numerics that must stay bit-exact

`bf16f`/`f2bf16`/`e4m3f` in `common.hpp` are bit-exact with `src/math.rs` **on the finite
domain** — the CPU oracles in `tests/kernel.rs` test that. They are NOT bit-exact for NaN:
`half::bf16::from_f32` forces the quiet bit (`| 0x0040`) where `common.hpp`'s `f2bf16`
returns the top 16 bits verbatim. That divergence predates the port; **the GLSL must mirror
HIP, not `math.rs`**, or the two backends disagree. GLSL has no `__builtin_memcpy`; use
`floatBitsToUint`/`uintBitsToFloat`. The e4m3 LUT (`e4m3_lut_build`) is a 256-float `shared`
array — prefer the LUT to a live `exp2`, which GLSL specifies only to 3 ULP.

### Same spelling, different semantics — the porting pre-flight

**Run this list against the HIP source BEFORE writing a shader, not after the oracle
fails.** Every entry below was discovered the expensive way, and each one was silent at
every stage before a numeric comparison. They are one class: a construct that is spelled
the same or nearly the same in HIP and GLSL, and means something different. That class is
where this port's real defects live — not in the algorithms, which transliterate fine.

| HIP | GLSL | Mechanism | Symptom |
|---|---|---|---|
| `float* lut` parameter | `float lut[256]` parameter | C decays arrays to pointers; GLSL copies value-result | Per-invocation private copy, writes lost, silent garbage. Rule 11 |
| `i / block`, signed | `i / block` | LLVM strength-reduces, but the signed-quotient correction survives | Silent cost — 44→29 VALU/iter. Use `findMSB` |
| `__shfl_down(v, o, 32)` | `subgroupShuffleDown` | Out-of-range returns the caller's own value in HIP; UNDEFINED in SPIR-V | Silent divergence on the last lanes. Rule 2 bans it |
| `1.0f/sqrtf(z)` | `inversesqrt(z)` | Different accuracy contracts under identical intent | Silent ULP divergence from the CPU oracle. Rule 9 |
| `exp2f(float(k))` | `exp2(float(k))` | Exact on an integer argument in libm; NOT REQUIRED to be in GLSL | Silent ULP error under a claimed bit-exactness contract |
| `(unsigned char)p` | *(no such cast)* | C truncates; GLSL has no 8-bit type, so it must be `p & 0xFFu` | A wrong mask is a wrong weight, not a compile error |
| `isfinite(x)` | *(does not exist)* | Must be composed from `isinf`/`isnan` | Compiles only if you notice; easy to get the polarity backwards |
| `size_t` index | `uint` index | 32-bit overflow at 4 GB in the shader, not on the host | Silent wrap on large tensors. See the Index width note |
| `OpFMul` + `OpFAdd` | *one* fused multiply-add | The DRIVER contracts below SPIR-V. The module still shows two opcodes | The disassembly does not tell you the arithmetic. See below |

**The FMA row is the one that breaks the instrument.** Every other entry is caught by
reading the source or the disassembly; this one is invisible in both. `glslc` emits
`OpFMul` then `OpFAdd`, the module says so, and the driver's backend fuses them anyway —
so the single-rounding FMA result differs from the two-rounding result, and nothing you
can inspect statically says which you got.

Measured, and it is not marginal: a CPU oracle modelling `acc += s * (a+b+c+d)` as written
differed from `mla_value_fp8` on **10 of 15 outputs**; modelling the fused chain
(`t = x0*l0; t = fma(x1,l1,t); …; acc = fma(s,t,acc)`) made it **exactly zero**.

Two consequences worth carrying:

- **"Disassemble anything whose bit-exactness matters" is necessary and not sufficient.**
  That advice appears twice in this document and it was written before this was known. The
  disassembly is ground truth for *which operations exist*, not for *how they are rounded*.
- **It is probably why the two backends agree.** hipcc defaults to contracting too
  (`-ffp-contract=fast` is on at `-O3` without `-ffast-math`), so HIP and Vulkan plausibly
  fuse the same expressions. That is a reason the port works, not a reason to relax: it
  means bit-identity between backends rests on two independent compilers making the same
  contraction choice, which is an assumption nobody has stated until now.

The generative rule, which is worth more than the table: **a HIP construct whose GLSL
analogue is "the obvious builtin" needs an explicit decision, never a transliteration.**
That is exactly how `inversesqrt` got in, and how `exp2` sat under a comment calling
bit-exactness a contract until this list was written.

### THE EXACTNESS STRATEGY: eliminate the degree of freedom, do not hope it is exercised alike

The single most consequential fact in this port, and it took until the FMA finding to state
plainly:

> **Cross-backend bit-identity rests on two independent compilers making the same choice
> wherever their specifications leave one open.** Careful porting does not achieve
> exactness — it only makes the two sides ELIGIBLE to agree. The agreement itself is
> contingent on decisions neither backend's author controls.

Every accuracy contract (`exp2` at 3 ULP, `inversesqrt` at 2 ULP), every latitude to
contract a multiply-add, every unspecified summation order is such a choice. Two conformant
compilers may resolve each differently and both be right.

**So the strategy is not to match the choices. It is to remove them.** Three fixes already
in this tree are instances of one move, and naming the move matters more than the list,
because the list cannot tell you what to do about a case it does not contain:

| degree of freedom | how it was removed |
|---|---|
| `exp2`'s accuracy contract | power of two **built from bits** — exact by construction, no contract left to differ on |
| `pow`/`cos`/`sin` in rope | evaluated **host-side in f64**, uploaded as a table — no shader transcendental left to differ on |
| which multiply fuses in `a*b + c*d` | a **named temporary** for one product — no contraction freedom left to differ on |

The pattern: find the point where the two toolchains are permitted to disagree, and
restructure so the permission never arises. That is stronger than testing for agreement,
because a test can only observe today's compilers.

**When you meet a new case, ask in this order:**

1. **Can the value be constructed exactly?** (`exp2` — integer argument, exact power of two.)
2. **Can the work move to a place with one implementation?** (rope — the host, in f64.)
3. **Can the expression be written so only one lowering is legal?** (the accumulator — split
   the products across statements.)
4. Only if none apply: **pre-register the divergence**, name the mechanism, and put it on
   the token-ID gate's suspect list. `exp` in softmax is the one genuine case so far, and it
   is genuine because its argument is an arbitrary runtime value.

**Why this hazard class appears in 2c and not earlier, which also predicts where it appears
next.** Contraction only has a choice to make when TWO multiplies feed one add. Every
kernel ported through 2b has at most one multiply per add — `acc += x*w`, `acc += s*(…)` —
so both compilers fused the only candidate and the backends agreed *by having no freedom*,
not by anyone's care. `attn.hip:185`'s `acc[k]*corr + p*Lt[..]` is the first expression in
the port with two, and the first place the choice is real. Grep for that shape when
approaching any remaining kernel.

### A WORK PARTITION IS A NUMERICS DECISION. `mla_plan_splits` IS NOT A PERF KNOB.

The one divergence class that lives BETWEEN kernels rather than inside one, which puts it
outside the reach of every instrument this port has built.

`attn.hip::mla_plan_splits` cuts the `nr` attended rows into `n_splits` chunks. It reads
like tuning — it is derived from CU count and a work threshold, and its own comment
describes occupancy. But the cut determines **which rows each split reduces**, therefore
the summation order, therefore the bits. The HIP source says so in passing ("same context
length → same cut → same summation order → bit-identical results"); the Vulkan port has to
reproduce the function EXACTLY, not merely plausibly.

**Why no existing instrument would catch a divergence here.** Each backend's per-kernel
oracle models that backend's own plan, so each agrees with its own reference and both go
green. The numbers differ only when the two are compared to each other — which nothing does
until the token-ID gate. So a "harmless" retuning of either copy — bumping
`MLA_TARGET_BLOCKS` for a new GPU, changing `MLA_MIN_TILES_PER_SPLIT` — is a **silent
numerics change** that presents as a decode divergence months later, in a different file
from the one that was edited.

Guarded by `tests/vk.rs`'s attention oracle asserting a specific split count for its shape,
so a drift in either copy fails loudly and locally. That is a weak guard for a strong
hazard; it is what is available without a cross-backend harness.

**The general shape, worth carrying to the remaining kernels:** *anything that partitions
work and feeds an order-sensitive reduction is a numerics decision wearing a scheduling
costume.* Candidates already in this codebase: `gemv_fp8`'s split-K threshold (a bare 4096
literal on the Vulkan side, uncoupled from the HIP constant — the same defect, unfixed),
`moe_reduce`'s expert batching, and any future change to `TILE` or `HB`. Ask of each: if I
retuned this for speed, would the OUTPUT change? If yes, it is single-sourced or it is a
bug waiting for a profiler.

> **2026-08-01: `moe_reduce` was the example, and it is now the counter-example.** It is
> deleted — replaced by `moe_acc_drain` over an i64 fixed-point accumulator, where every
> expert `atomicAdd`s at scale 2^-44 in any order. **Integer addition associates, so the
> partition carries no freedom and retuning it cannot move the output.** That is the only
> way an item leaves this list: not by being checked more carefully, but by the
> schedule-sensitivity being designed out. Kept here as the worked example, because the
> question — "if I retuned this for speed, would the OUTPUT change?" — is what the rest of
> the list still needs asked of it.

Two of these have mechanised guards (rules 2 and 9) and one is preventive (rule 11).
The rest are checked by reading, which is why the list exists.

### Rule 11 is the first PREVENTIVE rule

Every other mechanised rule is detective: it fires on a defect that already exists in the
tree. Rule 11 fires on a defect about to be introduced. `mla_absorb_fp8` and
`mla_value_fp8` both call `e4m3_lut_build`, so a mechanical transliteration of tranche 2b
reintroduces the copy-in/copy-out bug TWICE, in two kernels whose oracles do not exist
yet — and it would again be silent at every stage before the numeric check. The rule was
written an hour before the code it will stop.

That is the strongest available argument for mechanising a rule the moment it is
understood rather than after the next instance.

---

## The mechanised guards — the registry, written 2026-08-01

**Ten places in this document and in `kernels/vk/` cite "rule 2", "rule 9", "rule 11" as
though a numbered list existed. None did** — not in this file, not in the pre-split
`docs/VULKAN.md`, not in any commit. The rules are real and they are enforced; only the
list was missing, so a reader meeting "rule 11" in a shader had no way to learn what it
forbids. The table is that list. It is derived from the enforcing code, which is the only
authority: **`build.rs::vulkan()` runs the SPIR-V guards on every shader, every build.**

| # | Guard | Mechanism | What it forbids, and the defect that bought it |
|---|---|---|---|
| — | `spirv_val` | `build.rs`, `spirv-val --target-env` | Invalid modules. The `--target-env` is load-bearing: without it only the UNIVERSAL SPIR-V rules apply, not Vulkan's. |
| 2 | `no_subgroup_arithmetic` | `build.rs`, capability scan | `OpCapability GroupNonUniformArithmetic`/`Clustered` — i.e. `subgroupAdd` and friends. Their summation order is implementation-defined, which breaks greedy decode's reproducibility. Every reduction is the fixed `wave_sum` shuffle ladder instead. |
| — | `no_reciprocal_rewrite` | `build.rs`, `OpFMul` constant scan | `glslc -O` rewriting `a / K` into `a * fl(1/K)`. At K = 448 that differs from a true divide by 1 ULP on **55% of inputs, measured**, while hipcc keeps a real `fdiv`. No oracle tolerance is tight enough to see it. |
| 9 | `no_banned_builtins` | `build.rs`, `OpExtInst` denylist | `InverseSqrt`. Vulkan specifies it to 2 ULP as ONE operation; HIP does a correctly-rounded `sqrt` then a correctly-rounded divide. The risk is a reviewer "tidying" `1.0/sqrt(z)`. |
| 11 | `no_array_parameters` | `build.rs`, whole-array `OpLoad` scan | GLSL's copy-in/copy-out of array parameters. `e4m3_lut_build(inout float lut[256], …)` gave every invocation a private 1 KB copy; the fp8 table became noise (err = 8.6e37). It passed a clean compile, `spirv-val`, every other guard, and GPU-AV — the reads were IN BOUNDS. |
| 12 | `no_barrier_without_memory` | `build.rs`, barrier + storage scan | A bare `barrier()`, which orders SHARED memory only (0x108, no `UniformMemory` bit) where HIP's `__syncthreads()` orders global too. `rope_interleave` shipped this way and passed on today's RADV. The skip set (`BARRIER_EXEMPT`) is PINNED and a stale entry fails the build. |
| 8 | transcription lock | `tests/glsl_numerics.rs` | `f2e4m3`/`f2bf16` in `common.glsl` drifting from the CPU transcriptions that test them. It hashes the two function bodies and fails on a change — a stale transcription would keep passing while testing a function the shader no longer has. |
| 10 | launcher coverage | `tests/kernel_coverage.rs` | A `launch_*` in `src/backend/vk.rs` with no test. Tranche 2a ported six kernels and shipped `gemv_i8`/`gemv_fp8` never once executed, while the suite went 16 tests to 23 — coverage grew while the gap grew faster. **Re-keyed 2026-08-06 onto `src/backend/` and still live** — the only rule in this table that outlived the backend, because its subject was never Vulkan-specific, and the one row here that is not archived. It found 18 of 48 `hip.rs` launchers unexercised on arrival. |

**Numbers 1, 3–7 are not recoverable.** The surviving labels are the code's own — `build.rs`
says "ninth", "eleventh", "twelfth"; the tests say "EIGHTH" and "TENTH"; rule 2 is pinned by
this document's own pre-flight table naming it for the subgroup ban. **Row 10 is now the
exception to that sourcing:** the re-key deleted the "TENTH MECHANISED RULE" ordinal from
`tests/kernel_coverage.rs`, because it indexed this archived table and nothing maintains the
numbering. The row keeps its number here as history; the code no longer asserts one. The
rest were never
written down anywhere that survived, so the gaps above are honest rather than reserved. **If
you add a guard, put its number in its doc comment** — that is the only reason eight of
these are still identifiable.

Two of the guards are *detective* (they fire on a defect already in the tree) and rule 11 is
*preventive*, per the section above. Rule 12 was verified BOTH ways before adoption — quiet
on the four kernels with legitimate shared-memory barriers, loud on the one bug — which is
the standard rule 11's silence set.

---

## Known gaps in the fp8 dot — OPEN, both of them

Recorded here 2026-08-01. `kernels/common.hpp`'s `fp8_dot_strided` cited "docs/PERF.md #4"
for these; `PERF.md` was split three ways and its item numbering no longer exists, so for a
while the only record of two live defects was a comment pointing at a deleted file. Both
were re-verified in the tree before writing this.

**1. `kernels/vk/fp8.glsl` has the block-scale quad bug its HIP twin was fixed for.**
The dword path applies ONE block scale to a quad's four columns, so it is only the right
scale when the tile is at least a quad wide; at `block` 1 or 2 the columns past the tile
boundary silently take `i0`'s scale. The HIP side guards it — `int n4 = (block >= 4) ?
(i_dim >> 2) : 0`, handing those rows to the per-column tail — and the GLSL side still
computes `uint n4 = uint(i_dim) >> 2` unconditionally. **The oracle mirrors the bug**, so
the bit-exactness gate does not fire on it. The engine runs `block = 128`, so nothing
reaches it today; a smaller block would.

**2. `rivoli_gemv_fp8` does not guard the `i_dim % 4` its `w4` cast needs.**
Unlike the Vulkan twin and unlike `rivoli_mla_value_fp8`'s `kvl % 4`. The requirement is
now CONDITIONAL rather than absolute — at `block < 4` the cast is never reached — which is
why this survived as a comment rather than a fix. `kernels/linalg.hip` marks the site.

