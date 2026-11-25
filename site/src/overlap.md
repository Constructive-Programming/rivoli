# The overlap

A decode step walks the model one layer at a time. On a model bigger than memory, some of
each layer's weights live on NVMe. rivoli's job is to make sure that, by the time the GPU
reaches a layer, the bytes it needs have already arrived.

## Three tiers

| tier | holds | reached by |
|---|---|---|
| **Pin** | whatever the memory budget leaves room for | the GPU directly, through GTT |
| **Cache** | recently or frequently used experts and layers | the GPU directly, through GTT |
| **Stream** | everything else | NVMe reads into staging slots, then a copy the GPU waits on |

Strix Halo's LPDDR5 is unified. The GPU reads system memory through GTT, so "resident" just
means "already in RAM", with no separate VRAM copy. The NVMe-to-RAM leg is what has to hide
behind compute.

## The pin is a function of free memory

What stays resident is decided at run time from how much memory is actually free. That
depends on other tenants, on the KV cache at the configured context, and on scratch space.
It is never decided by model architecture
([P6](principles.md#p6--the-pin-is-a-function-of-free-memory)).

In the code, a single function, `rivoli-core::residency::partition(ordered, free, floor)`,
makes that decision. It has no architecture parameter to consult. Its result is a *prefix*
of the caller's priority order, so more memory only ever extends the pin. A dense model
that fits is the same code path with the prefix covering everything. A budget below the
floor is refused with the arithmetic in the message, so a run never silently degrades.

## Caching is a trade

The cache is not a transparency layer. Memory space, bandwidth and compute form a
triangle, and which format a layer or expert takes is a decision made against it
([P2](principles.md#p2--caching-is-a-key-principle)). For example, a 4-bit format can be
cheaper to compute but more expensive to store than a 3-bit vector-quantized one. The
4-bit format is used where there is bandwidth and space to load it, and the 3-bit one
where there is not.

## No residency branch in the loop

For mixture-of-experts layers, submitting the routed experts returns one *ticket* per
expert, not a "this one is resident" mask. The kernel launch waits on the ticket. A
resident expert's ticket is already satisfied, so resident, missing and in-flight experts
all take the same path. The loop has nothing to branch on, so it cannot branch wrong.

Waits can be enqueued on the device before their producer exists, and any wait can be
released from the host. So a fetch that fails returns an error instead of hanging the GPU.
