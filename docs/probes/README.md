# docs/probes — standalone diagnostics

Instruments, not tests. They are never run by CI and are not part of the crate build;
each is its own tiny cargo package so it cannot perturb `rivoli`'s dependency graph or
lint surface. You fire one when you need to establish a fact about the machine, then
record the answer in the doc that depends on it.

| probe | question it answers |
|---|---|
| `vk_validation/` | Do the Vulkan validation checkers actually fire on this driver + layer? |

## `vk_validation` — trust a silence, but verify it first

The Vulkan backend's whole safety argument rests on the validation layer reporting
nothing. That argument is worthless if the checker is loaded but inert, which is not a
hypothetical: **synchronisation validation and GPU-assisted validation are both OFF by
default**, so a clean run under the default configuration says nothing whatsoever about
the `Gpu::enqueue` barrier or about buffer-device-address accesses — and those are, in
order, the two things most likely to be wrong in this backend.

So before trusting silence, make each checker speak. Each mode injects one deliberate
fault and reports whether the expected diagnostic came back.

```
cd docs/probes/vk_validation

cargo run -- core                                  # no env needed
VK_LAYER_VALIDATE_SYNC=1 cargo run -- sync
VK_LAYER_GPUAV_ENABLE=1  cargo run -- gpuav
```

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
session; "it fired last time" is not evidence about this time.

Last established on RADV STRIX_HALO (AMD Radeon 8060S),
`VK_LAYER_KHRONOS_validation` 1.4.341, Vulkan 1.4.335 loader:
`core` **fires**, `sync` **fires**, `gpuav` **fires**,
`compute-compute` **silent**, `compute-copy` **silent**, `compute-copy-desc` **silent**.

## Traps worth knowing

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
