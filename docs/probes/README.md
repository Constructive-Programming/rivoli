# docs/probes — standalone diagnostics

Instruments, not tests. They are never run by CI and are not part of the crate build —
either a single-file `hipcc` program or its own tiny cargo package — so none of them can
perturb `rivoli`'s dependency graph or lint surface. You fire one when you need to establish
a fact about the machine, then record the answer in the doc that depends on it.

Everything from "Operating in this repo" onward is method that outlives any one probe: it is
about writing guards and reading their silence, and it is why this file survived the crate it
originally documented.

| probe | question it answered | status |
|---|---|---|
| `waitvalue_visibility.hip` | Is a `hipStreamWriteValue64` on one queue visible to a `hipStreamWaitValue64` enqueued on another? | 0 mismatches over 8.4e8 checks; INV-4 rests on it |
| `fetch_batch.hip` | Is the demand fetch leaving drive bandwidth on the table? | **No** — ARCHITECTURE §3 |
| `fetch_stream_ops.hip` | What does the reaper pay per completed read, above the NVMe? | the per-read `hipLaunchHostFunc` was dead; deleted 2026-08-01 |
| `vk_validation/` | Do the Vulkan validation checkers actually fire on this driver + layer? | answered, deleted; source in git at `77b5500:docs/probes/vk_validation` |

**Run every one of these under the GPU lock** (`flock /tmp/rivoli-gpu.lock -c '…'`) — they
allocate device memory and take real bandwidth, so an unlocked probe corrupts whatever
benchmark is running beside it, and vice versa. The two fetch probes are also the reason to
say it twice: their whole point is to measure a drive under a *specific* concurrent load, so
a stray neighbour does not just add noise, it answers a different question.

## The two fetch probes, and why they are a pair

`fetch_batch` asks what the DEVICE can do in the engine's exact shape; `fetch_stream_ops`
asks what the ENGINE adds on top. Run apart they mislead in opposite directions — the first
against an idle GPU flatters the drive by ~12%, the second says nothing about whether the
drive was the limit. Together they bracket the answer, and the bracket is what showed the
demand fetch has no bandwidth left to recover (ARCHITECTURE §3) while still turning up one
piece of dead work on the reaper's critical path.

Both are honest about a real hazard: **QD1 varies 7.7–12.5 GB/s across runs of the same
probe.** Any conclusion that needs a single-digit percentage from them needs repeats, and the
"~26% unexplained" this pair was written to chase turned out to be mostly that spread.

## `vk_validation` — trust a silence, but verify it first

**The probe is deleted; this section is its finding.** Restoring it from the tree at
`77b5500` is the way to re-establish the matrix below — do that rather than re-deriving
the fault injections, which are the expensive part (each mode had to be tuned until the
layer actually complained, and a fault the layer ignores looks identical to a checker
that is not watching).

The Vulkan backend's whole safety argument rests on the validation layer reporting
nothing. That argument is worthless if the checker is loaded but inert, which is not a
hypothetical: **synchronisation validation and GPU-assisted validation are both OFF by
default**, so a clean run under the default configuration says nothing whatsoever about
the `Gpu::enqueue` barrier or about buffer-device-address accesses — and those are, in
order, the two things most likely to be wrong in this backend.

So before trusting silence, make each checker speak. Each mode injected one deliberate
fault and reported whether the expected diagnostic came back — `core` needs no env,
`sync` needs `VK_LAYER_VALIDATE_SYNC=1`, `gpuav` needs `VK_LAYER_GPUAV_ENABLE=1`.

| mode | fault injected | expected diagnostic |
|---|---|---|
| `core` | `vkCreateBuffer(size = 0)` | `VUID-VkBufferCreateInfo-size-00912` |
| `sync` | two overlapping `vkCmdFillBuffer`s with no barrier between them | `SYNC-HAZARD-WRITE-AFTER-WRITE` |
| `gpuav` | compute shader stores 4 MiB past a 256-byte allocation, through a `GL_EXT_buffer_reference` whose address arrives as a bare `uint64` push constant | `VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819`, "Out of bounds access" |
| `compute-compute` | two dispatches read-modify-writing one address, no barrier | `SYNC-HAZARD-*` |
| `compute-copy` | dispatch writes via buffer reference, `vkCmdCopyBuffer` reads it, no barrier | `SYNC-HAZARD-READ-AFTER-WRITE` |
| `compute-copy-desc` | as above but the shader writes through a DESCRIPTOR binding | `SYNC-HAZARD-READ-AFTER-WRITE` |

### The sync modes map a coverage MATRIX, not a pass/fail

**A checker's envelope has to be established per hazard class you intend to rely on,
not once for the checker.** "Synchronisation validation works" is not a fact about
synchronisation validation — it is a fact about one hazard class on one driver and one
layer version. The last three modes exist because that turned out to matter a great
deal here:

| hazard class | access model | fires? |
|---|---|---|
| transfer ↔ transfer | n/a | **yes** (`sync`) |
| compute → compute | buffer reference | **no** (`compute-compute`) |
| compute → transfer | buffer reference | **no** (`compute-copy`) |
| compute → transfer | descriptor | **no** (`compute-copy-desc`) |

**Synchronisation validation on this stack covers only transfer↔transfer.** Every
hazard class involving a compute dispatch is invisible, with or without a barrier, and
whether the shader reaches memory through a descriptor or a buffer reference. Since a
compute backend is dispatches almost exclusively, that is close to no coverage of the
thing it is for: `Gpu::enqueue`'s barrier — both its COMPUTE→COMPUTE and its
COMPUTE→TRANSFER scoping — is spec-derived, and a clean suite says nothing about it.

The descriptor row is what stops the easy explanation. This is not a
buffer-device-address blind spot; a descriptor-bound write is equally invisible, so
switching away from bare device addresses would not buy coverage back.

Anything the matrix marks "no" must be treated as UNVERIFIED in the code that depends
on it, however clean the suite looks. A row is only evidence if the corresponding probe
mode has been seen to fire on this exact stack.

Exit status is 0 only if the expected diagnostic was observed. A **non-zero exit is the
finding**: it means that checker is not watching, and anything you concluded from its
silence has to be withdrawn.

`gpuav` is the one that matters most. rivoli passes every buffer to every kernel as an
opaque device address in a push constant — there is no descriptor, no object, and
nothing for the CPU-side layer to bounds-check against. GPU-AV instrumenting the shader
is the only thing standing between a wrong address and plausible garbage, and the probe
injects that exact access shape rather than a toy.

## When to re-run

Whenever the answer could have changed and you are about to rely on it: a Mesa/RADV
update, a `vulkan-layers` update, a loader update, or a move to different hardware. This
box went from *no validation layer installed at all* to 1.4.341 inside a single working
session; "it fired last time" is not evidence about this time. **That is the standing
cost of having deleted the probe: the trigger still exists, and meeting it now takes a
`git show 77b5500:docs/probes/vk_validation/src/main.rs` first.** The alternative was
keeping 800 lines and a second crate resident to answer a question that fires on a
driver update, and the record below is what it produced.

Last established on RADV STRIX_HALO (AMD Radeon 8060S),
`VK_LAYER_KHRONOS_validation` 1.4.341, Vulkan 1.4.335 loader:
`core` **fires**, `sync` **fires**, `gpuav` **fires**,
`compute-compute` **silent**, `compute-copy` **silent**, `compute-copy-desc` **silent**.

## Operating in this repo: two hazards that bite silently

Neither is about Vulkan. Both cost real work in this session and neither announces
itself.

### Is the GPU in use? Ask the driver, not `pgrep`

```bash
ls /sys/class/kfd/kfd/proc/                 # PIDs holding a GPU context
cat /proc/<pid>/comm                        # who they are
```

`/sys/class/kfd/kfd/proc/` is the amdgpu driver's own list of processes with an open
GPU context. It is name-independent and authoritative.

**Every name-based check is structurally blind**, and the blindness is invisible —
it reports "free" rather than erroring. Confirmed live: kfd showed PID 2195696
holding a context while `pgrep -x rivoli` returned nothing, because the process was
named `rivoli_base`. `cargo test` binaries (`kernel-e0184386`, `vk-…`) are worse:
their names contain a content hash and match no fixed pattern at all, so a suite run
holds the device completely invisibly.

A second, separate failure of the same check: `pgrep -f "target/release/rivoli"`
matches **its own shell's command line**, so it reports BUSY forever. A name-based
check can therefore fail in both directions — blind to real users, and tripping on
itself. If you must use one, run it from a script file so the pattern is not in the
invoking command line, and know it still cannot see test binaries.

### Never `git stash` here — the stash stack is shared across worktrees

`git stash` is per-repository, not per-worktree, so a `stash push` in one worktree and
a `stash pop` in another operate on the same stack. A no-op `push` (nothing to stash)
followed by a `pop` will happily restore **someone else's** work into your tree, as
conflict markers in files you never touched.

There is more than one party working in this repository. To get a clean tree, use
`git checkout <ref> -- <paths>` or a separate build directory.

## Two kinds of false confidence, and the question that finds each

Everything below is one of two failures. They feel identical from inside — a green
suite — and you find them with different questions, so it is worth knowing which you
are hunting.

**An instrument reporting on the wrong thing.** It runs, it produces output, and the
output is about something other than what you believe. A validation layer whose
messages go to an empty sink. A `pgrep` matching its own shell. A probe that passes
because the thing it tests kept the process alive. A test whose oracle was copied from
the implementation.

> Ask: **would this fire if the thing were wrong?**
> Answer it by breaking the thing on purpose and watching.
>
> **Break however you like. Restore with `git checkout -- <file>`, never by reverse
> substitution.** Undoing a deliberate break with `sed 's|new|old|'` once rewrote a
> COMMENT in `append_kv.comp` that legitimately quoted the removed literal while
> explaining why it was removed. The build stayed clean and the documentation had been
> falsified. `git checkout` is exact by construction and cannot touch prose that happens
> to quote the code.
>
> Note the vector: this repo already had one "comment asserting something false" — that
> one was written by a person. This one was a RIGHT comment rewritten by a script. Same
> failure, and no author to catch it in review, so the technique has to be safe by
> construction rather than by careful reading at the end of each cycle.

#### What break-verification does NOT establish

This is the strongest technique in this document and it has a boundary, written down
because the practice is now good enough that its limit is the useful part.

**Breaking a thing and watching it go red proves the guard fires. It says nothing about
whether it fires CORRECTLY in every transition.** Three defects got through a full
break-verification pass of fifteen guards, and none was reachable by the technique:

- **Interactions between findings.** Each guard was broken alone, so nothing exercised
  two firing at once — which is exactly where one masked the other (see "A guard can mask
  another guard" below).
- **Deletion.** Every break ADDS or CHANGES something. A per-item check is driven by the
  items that exist, so the transition it structurally cannot observe is an item ceasing to
  exist — a renamed shader left an orphaned exemption that would silently pre-authorise
  any future shader taking the name back.
- **The guard's own new failure modes.** Adding an argument to `spirv-val` created a
  failure class that could not exist when it took none: its argument-parsing diagnostic
  goes to stdout, which the reporter discarded, so a wrong spelling would have failed
  every shader with an empty body and blamed the modules for a broken invocation.

All three were found by a correctness review reading the diff, not by breaking anything.
So: **break-verification is the positive case, and review is the only thing covering the
interaction, deletion, and self-inflicted cases.** Budget both. The same conclusion the
synchronisation-validation risk reached from the other direction — on this stack, review
is the primary defence and the mechanised layer is secondary.

**A guard that is never exercised.** It is correct, it is present, and no test
distinguishes it from its absence — the suite would be equally green with the guard
deleted. `n_blocks == kvl/128` was unreachable because an earlier arm rejected every
case that would have tested it. The odd-`ropn` refusal — the one place this backend is
deliberately stricter than HIP — existed only as a comment. `place`'s word padding was
invisible because every test length was already a multiple of four.

> Ask: **would the suite notice if I deleted this?**
> Answer it by deleting it and running the tests.

**Coverage that grew while a gap grew faster.** A number moved in the reassuring
direction while the thing it stands for moved the other way. Tranche 2a ported six
kernels and wrote three oracles: the suite went from 16 tests to 23, every one passed,
and the two hardest kernels in the batch had never executed. "We added tests" is exactly
the evidence someone would cite to argue the opposite.

**A check constitutionally blind to the defect the code is exposed to.** Not an
instrument pointed at the wrong thing, and not a guard that never runs — a check that
runs, passes honestly, and *cannot* see a whole class of defect, where that class is
exactly what the code is exposed to.

`tests/glsl_numerics.rs` transcribes shader functions into Rust and compares against
`math.rs` over ~1.2M values. It is strong evidence about statements. It said `e4m3f` was
bit-exact while the shader used GLSL's `exp2` — 3 ULP allowed, no integer exemption — and
the transcription used Rust's `f32::exp2`, which is exact. Both sides agreed perfectly
and the only property at issue differed between them. **A transcription mirrors
STATEMENTS, and an accuracy contract is precisely what a transcription cannot
transcribe.** The lock would have stayed green forever.

The blindness is co-located with the exposure, which is what makes it dangerous rather
than merely incomplete: the technique is weakest on precision contracts, and precision
contracts are the entire `inversesqrt`/`exp2` hazard class this port keeps hitting.

> Ask: **what kind of defect is this check constitutionally unable to see, and is the
> code exposed to that kind?**
> Not "would it fire" and not "would the suite notice if I deleted it" — both of those
> pass here. Answer it by naming the class, then checking whether the code contains one.

#### The measurable form: compare the size of the defect to the size of the bound

The version of this question you can answer with arithmetic instead of judgement, and the
one that stops you fixing the wrong thing.

A numeric test compares against a bound. If the defect class you care about perturbs the
result by *orders of magnitude less than that bound*, the test cannot detect it — not on
this input, not on a bigger one, not ever. **That is a property of the two magnitudes, not
of the test data**, so the usual instinct (find a better shape) is guaranteed to fail while
looking like progress.

The case. A kernel oracle compared at `1e-3·mx + 1e-3` ≈ `2e-3`. The property at issue was
SUMMATION ORDER, which perturbs an f32 reduction by ~`1e-7`. Four orders of magnitude
apart. It was first diagnosed as a shape problem — the test's dimensions genuinely did
leave half the lanes idle — and the shape was duly fixed. **A deliberately wrong summation
order still passed, at 27001× margin.** The shape diagnosis was correct and irrelevant.

The fix was a different KIND of assertion: compare bits. With bit-identity asserted, the
same wrong order failed 10 of 15 elements.

> Before enlarging a shape or tightening a tolerance, do the division: **how big is the
> defect, how big is the bound?** If the answer is "orders of magnitude smaller", stop
> looking for a better input and change what you are asserting.

The trap has a tell worth recognising: you can construct a *correct* argument that the
current input under-exercises the code — as was done here — and act on it, and be no better
off. A true diagnosis is not the same as the operative one.

Note the distinction from a broken tool: the lock **worked exactly as designed** — it
fired on the edit, named the three-step check, and refused to pass until the hash was set
deliberately, which is what forced all 256 values to be verified first. What was wrong was
the SCOPE OF THE CLAIM being read into it. A tool doing its job while the claim built on
it is too broad is a different failure from a tool that does not work, and the fix is to
the claim.

> Ask: **what is NOT in this suite?**
> Answer it by enumerating what SHOULD be covered and diffing — not by reading the
> count. `tests/kernel_coverage.rs` does this mechanically for kernels.

**A green suite is not a claim about what is in it.**

The first question is about instruments, the second about guards, and neither finds the
other's failures. A check can also be both at once: `assert_quantization_unambiguous`
fires correctly *and* its 8-ULP margin is exactly wide enough to hide a 1-ULP toolchain
divergence, so it protects a comparison and blinds it in the same stroke.

## Writing a guard: target what survives the compiler

A guard that inspects compiled output must match a construct that **survives
optimisation**, not the one you wrote in source. This is not a matter of picking a
broader or narrower signature — get it wrong and the guard is *structurally incapable of
firing*, while looking correct and passing review.

The worked example. `void e4m3_lut_build(inout float lut[256], uint tid)` copied a shared
array per invocation and produced garbage. The obvious guard is "reject an
`OpFunctionParameter` of array type" — it names exactly the thing that is wrong. It
cannot work: `glslc -O` inlines the helper, and the shipped module contains **zero**
function parameters. Measured, not assumed.

What survives is the copy itself — a whole-array `OpLoad` — because that is what
inlining leaves behind. That signature fires.

Check the existing rules against this and it is why they work, though nobody chose them
for it: capabilities survive (they are module-level declarations), `InverseSqrt` survives
(an opcode is not inlined away), a whole-array `OpLoad` survives. Anything that exists
only in the *source* — a function boundary, a parameter, a variable name, a type alias —
may not reach the artifact at all.

> Before writing rule twelve: compile a deliberate instance, disassemble it, and confirm
> the construct you plan to match is actually present. If it is not, you are matching
> source structure through a lens that has already discarded it.

## Traps worth knowing

### A check that stops at the first failure reports a floor, not a count

`build.rs` compiled shaders in sorted order and aborted on the first bad one. Break a
shared constant and four shaders fail; the build names one. Fix it, rebuild, get named
the next. Each rebuild teaches one fact the compiler already knew in full — and
"the build passes now" after fixing one error is a weaker statement than it sounds,
because nothing ever established that one was the only one.

It surfaced the good way: a deliberate-break arm reported DID NOT FIRE for a `#error`
that does fire, because an earlier-sorted shader's `#error` aborted the build first.
A *negative* result distrusted enough to chase, rather than a positive one accepted.
Fixed by compiling everything, accumulating, and failing once with the whole list.

The shape generalises past shaders — any first-failure-abort check reports a lower
bound on the damage. A test harness that stops on first failure does it. So does
`assert_validation_clean` in `tests/vk.rs`, per test: it asserts a count is zero, so the
first message aborts that test and any later ones in it go unseen. That is acceptable
there (one message is already a failure worth investigating) but it means the count in
the message is a floor, and the phrasing should never imply otherwise.

Ask of any check: **if there were three problems, would this tell me three?**

### A guard can mask another guard

The same disease one level up: not the first FAILURE aborting the rest, but the first
FINDING doing it. A guard that `return`s as soon as it has something to say cannot report
the second thing it knows.

The case, and it is worse than a lost message. `build.rs` has one rule that encourages
replacing an LDS reduction with the `wave_sum` shuffle ladder, and another that refuses to
judge barriers in modules holding shared memory. Perform the encouraged conversion and the
module drops its shared storage while keeping a bare `barrier()` over buffer traffic —
bit-for-bit the `rope_interleave` signature that the second rule exists to catch. The
bookkeeping finding ("this module is listed as exempt but no longer needs to be") fired
first and returned, swallowing "barrier that orders NOTHING". **A developer following the
stated remedy would have introduced a live ordering bug and been told the build was
clean** — and the substituted message reads as trivial admin, which is exactly when nobody
looks harder.

Two lessons, and the second is the general one:

- A guard should ACCUMULATE its findings and fall through, for the same reason the build
  accumulates across shaders.
- **Rules interact.** One rule's recommended fix can walk code into another rule's blind
  spot. When adding a rule, ask what the OTHER rules tell people to do, and whether any of
  those edits lands in the new rule's exempt case — a question no amount of breaking the
  new rule on its own will answer.

### A correct guard resting on a false rationale

`DeviceTier::place` pads its cursor to a word boundary, and the comment said this is
what stops one placement's 32-bit read reaching into the next placement's data. The
padding is harmless and the code is safe. The *reason* was wrong: `off` is already
rounded to 256, so `round_up_256(off + len) == round_up_256(off + span)` for every
length and the returned addresses are byte-identical with and without it.

Nobody is harmed by a guard that does nothing — until someone acts on the rationale.
Lowering that 256-byte alignment is a plausible, local-looking tightening, and anyone
doing it while believing `place` covers the gap would convert a benign read into a live
overrun. **The code was safe and the argument was load-bearing, and it is the argument
a maintainer reads.**

So: when a guard's justification is checked and found false, fix the justification even
if the code stays. State what the guard actually does, and why it is still worth
keeping if it is. The failure mode is not the dead code — it is the true-sounding
sentence next to it.

### A comment can camouflage the bug it describes

`Gpu::signal` submitted to the queue without holding the mutex that Vulkan's
external-synchronisation requirement needs — while carrying a SAFETY comment asserting
the queue *was* externally synchronised, citing the `Send`/`Sync` note whose invariant
is "the queue is only touched under the `cmd` mutex". The function bearing the citation
was the code breaking the cited rule.

That is worse than no comment. An uncommented `queue_submit` invites a reader to check;
a commented one invites them to move on. The same shape appears wherever documentation
asserts the property it violates — a doc comment promising a bound the function does not
check, a `# Safety` clause listing a precondition the body then assumes rather than
verifies.

The fix was not a better comment. `vk::Queue` now lives *inside* the mutex-guarded
`Cmd` struct rather than beside it on `Gpu`, so an unguarded submit is
`error[E0609]: no field 'queue' on type '&Gpu'`. Verified by trying it. Every rule in
this backend that has been violated twice is now a build failure instead — the
`subgroupAdd` capability scan, `push_struct!`'s padding and budget assertions,
`WAVE`/`ROWS_PER_BLOCK` single-sourcing, and this. Conventions in `src/vk.rs` have a
measured failure rate of two.

### A test built to fail needs its passing arm checked too

The strongest form of the trap, because it survives all the other checks.

`chained_dispatch_respects_the_barrier` was rebuilt to prove it could detect a deleted
barrier. At 2048×2048 with the barrier removed it failed 8 of 8 — the exact result the
experiment was looking for. It was wrong. Running the same test with the barrier
*restored* showed it failed 8 of 8 there as well: the ping/pong output selection was
inverted, so it read the wrong buffer and would have failed under any condition. A
broken instrument and a confirmed hypothesis produced identical output, and the
difference was only visible from the arm nobody thinks to run.

**When you construct a test expecting failure, verify it passes under the condition
where it should pass.** A negative result needs its positive control exactly as much as
a positive result needs its negative one — and the direction that gets skipped is
whichever one you already believe.

This is the same family as the two below, and worse: there the instrument was silent
when it should have spoken, which at least looks like nothing. Here it spoke, fluently,
and said what was expected.

### The subject of a test can prop up its own scaffolding

See the `ash::Entry` story below: a probe passed because the thing it was testing kept
the process alive. Any check whose subject can supply an incidental side effect the
check depends on can produce a pass that means nothing — and the more thoroughly the
subject is engaged, the likelier it props something up.

### Match on the ID as well as the body

The probes match on `pMessageIdName` as well as the message body. An earlier version
searched only the body for `SYNC-HAZARD` and printed "sync validation caught it: false"
while the layer's raw output, three lines above, showed the hazard — the label lives in
the ID field, not the text. If you extend these, print the raw message too, and never
let a match expression be the only thing you read.
