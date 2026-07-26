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

Last established: **all three fire.** RADV STRIX_HALO (AMD Radeon 8060S),
`VK_LAYER_KHRONOS_validation` 1.4.341, Vulkan 1.4.335 loader.

## A trap worth knowing

The probes match on `pMessageIdName` as well as the message body. An earlier version
searched only the body for `SYNC-HAZARD` and printed "sync validation caught it: false"
while the layer's raw output, three lines above, showed the hazard — the label lives in
the ID field, not the text. If you extend these, print the raw message too, and never
let a match expression be the only thing you read.
