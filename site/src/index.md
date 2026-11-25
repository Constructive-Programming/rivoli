<div class="rv-hero">
<h1>rivoli<span>.</span></h1>
<p class="rv-tagline">Decode LLMs bigger than memory, on one box. Weights stream from NVMe <strong>overlapped with compute</strong>.</p>
<span class="rv-pill">Rust</span><span class="rv-pill">HIP / ROCm</span><span class="rv-pill">gfx1151 · Strix Halo</span><span class="rv-pill ember">NVMe streaming</span><span class="rv-pill ember">MoE</span>
</div>

rivoli is a decode engine for large language models that do not fit in memory. It targets
one machine: an AMD Strix Halo APU (gfx1151) with unified LPDDR5 that the GPU reaches
through GTT. Whatever does not fit stays on NVMe. While the GPU computes one layer, the
next layer's weights are already on their way.

<svg class="rv-overlap" viewBox="0 0 700 150" role="img" aria-label="Timeline: the NVMe stream fetches layer n+1 while the GPU computes layer n"><rect class="lane" x="100" y="20" width="590" height="40" rx="6"/><rect class="lane" x="100" y="76" width="590" height="40" rx="6"/><text x="0" y="45">NVMe → GTT</text><text x="0" y="101">GPU compute</text><rect class="stream" x="110" y="26" width="88" height="28" rx="4"/><text class="on-stream" x="118" y="45">fetch L1</text><rect class="compute" x="206" y="82" width="88" height="28" rx="4"/><text class="on-compute" x="214" y="101">layer L1</text><rect class="stream" x="206" y="26" width="88" height="28" rx="4"/><text class="on-stream" x="214" y="45">fetch L2</text><rect class="compute" x="302" y="82" width="88" height="28" rx="4"/><text class="on-compute" x="310" y="101">layer L2</text><rect class="stream" x="302" y="26" width="88" height="28" rx="4"/><text class="on-stream" x="310" y="45">fetch L3</text><rect class="compute" x="398" y="82" width="88" height="28" rx="4"/><text class="on-compute" x="406" y="101">layer L3</text><rect class="stream" x="398" y="26" width="88" height="28" rx="4"/><text class="on-stream" x="406" y="45">fetch L4</text><rect class="compute" x="494" y="82" width="88" height="28" rx="4"/><text class="on-compute" x="502" y="101">layer L4</text><rect class="stream" x="494" y="26" width="88" height="28" rx="4"/><text class="on-stream" x="502" y="45">fetch L5</text><rect class="compute" x="590" y="82" width="88" height="28" rx="4"/><text class="on-compute" x="598" y="101">layer L5</text><line class="axis" x1="100" y1="132" x2="690" y2="132"/><text x="620" y="146">time →</text></svg>

**The overlap is the whole design.** A model that fits in memory is not a separate mode.
It is the trivial case of the streaming path, where everything happens to be resident.

## What makes it different

- **One box, no portability tax.** rivoli is written for gfx1151 and its unified memory.
  When a hardware-specific path is faster, it wins over portability
  ([P3](principles.md#p3--maximize-hardware-features)).
- **Bytes per token is the currency.** Decode speed is traffic divided by bandwidth.
  Quantization, residency, prefetch and speculative decode are all ways to cut traffic.
  Each one's quality cost is measured, never assumed
  ([P5](principles.md#p5--bytes-per-token-are-the-currency)).
- **The memory knob trades speed, never text.** `--max-mem` changes tok/s and hit rate.
  It does not change the output ([P4](principles.md#p4--the-memory-knob-trades-speed-never-text)).
- **Every claim is a gate that can go red.** Each test is shown failing on purpose before
  its pass counts ([Quality gates](quality.md)).

## Where to go next

- [The overlap](overlap.md): how streaming, residency and caching fit together.
- [Principles](principles.md): the seven rules every change is checked against.
- [Architecture](architecture.md): six crates whose dependency graph *is* the layering.
- [Models](models.md): the model families in the tree.
- [Building](building.md): toolchain, build and test commands.
