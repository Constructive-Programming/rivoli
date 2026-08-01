---
status: live
verdict: Which kernels the Vulkan backend has: 16 of 29 ported, 6 more single-row. It decodes --mode int3-vq --attn dense at ~1.9x slower.
---

# rivoli — Vulkan kernel inventory

> The port *journal* — four phases, and everything that was tried and rejected — is
> [`investigations/vulkan-port.md`](../investigations/vulkan-port.md). This file is only
> what the backend can run today.

## Kernel inventory — port 16 of 29

v1 is **`--mode int3-vq`, `--attn dense`, decode only**. That is the smallest thing that
runs the model, and it cuts 13 kernels from the port.

| File | Port now | Port difficulty |
|---|---|---|
| `linalg.hip` | `gemv_fp8`, `gemv_fp8_splitk`, `gemv_f32`, `gemv_i8`, `swiglu`, `rmsnorm`, `rope_interleave` | `gemv_f32`/`swiglu`/`rmsnorm` trivial; the split-K LDS combine and the e4m3 LUT need care |
| `moe.hip` | `moe_gateup_vq`, `moe_down_vq`, `moe_reduce` | hardest — per-expert device addresses, fp16 codebook gather |
| `fwd.hip` | `embed_i8_row`, `append_kv`, `gather_rope`, `vadd`, `argmax_reduce` | easy |
| `mla.hip` | `mla_absorb_fp8`, `mla_value_fp8` | easy-moderate |
| `attn.hip` | `mla_latent_attend`, `mla_attend_combine` | moderate — dynamic shared memory sizing becomes a specialization constant |

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

### Numerics that must stay bit-exact

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

