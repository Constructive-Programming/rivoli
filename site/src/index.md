<div class="rv-hero">
<figure class="rv-hero-photo">
<img src="img/rivoli-hummingbird.webp" alt="Male Rivoli's hummingbird in flight" width="678" height="517">
<figcaption>Rivoli's hummingbird, male. <a href="https://commons.wikimedia.org/wiki/File:Rivoli%27s_Hummingbird_(male)_V%26R_SW_Research_Station_Portal_AZ_2019-04-14_11-08-37_(33931138468).jpg">Bettina Arrigoni</a>, <a href="https://creativecommons.org/licenses/by/2.0/">CC BY 2.0</a>; background removed</figcaption>
</figure>
<div class="rv-hero-text">
<h1>rivoli<span>.</span></h1>
<p class="rv-tagline">Decode LLMs bigger than memory, on one box. Weights stream from NVMe <strong>overlapped with compute</strong>.</p>
<span class="rv-pill">Rust</span><span class="rv-pill">HIP / ROCm</span><span class="rv-pill">unified memory systems</span><span class="rv-pill ember">NVMe streaming</span><span class="rv-pill ember">MoE</span>
</div>
</div>

rivoli is a decode engine for large language models that do not fit in memory. It is
designed for ROCm unified memory systems, where the CPU and GPU share one pool of RAM.
Whatever does not fit stays on NVMe. In each layer the GPU starts on the experts already in
memory while the missing ones stream in, so NVMe reads hide behind compute.

<svg class="rv-overlap" viewBox="0 0 890 212" role="img" aria-label="Timeline of two MoE layers showing the GPU nearly 100% booked: attention and gate run back-to-back with resident expert hits and arrived misses, while brief CPU route, submit, and reap steps orchestrate NVMe reads that stream in completely hidden behind resident compute."><rect class="lane" x="110" y="24" width="770" height="40" rx="6"/><text x="0" y="49">CPU</text><rect class="lane" x="110" y="74" width="770" height="40" rx="6"/><text x="0" y="99">NVMe → RAM</text><rect class="lane" x="110" y="124" width="770" height="40" rx="6"/><text x="0" y="149">GPU</text><text x="116" y="14">layer 1</text><rect class="compute" x="116" y="130" width="92" height="28" rx="4"/><text class="on-compute" x="162" y="148" text-anchor="middle">attn + gate</text><text class="cpu-label" x="203" y="48" text-anchor="end">route</text><rect class="cpu" x="208" y="30" width="8" height="28" rx="3"/><rect class="cpu" x="218" y="30" width="10" height="28" rx="3"/><text class="cpu-label" x="233" y="48">submit</text><rect class="compute" x="218" y="130" width="180" height="28" rx="4"/><text class="on-compute" x="308" y="148" text-anchor="middle">resident experts (hits)</text><rect class="stream" x="228" y="80" width="160" height="28" rx="4"/><text class="on-stream" x="308" y="98" text-anchor="middle">stream misses</text><text class="cpu-label" x="383" y="48" text-anchor="end">reap</text><rect class="cpu" x="388" y="30" width="10" height="28" rx="3"/><rect class="compute" x="401" y="130" width="91" height="28" rx="4"/><text class="on-compute" x="446" y="148" text-anchor="middle">misses</text><line class="divider" x1="494" y1="16" x2="494" y2="174"/><text x="500" y="14">layer 2</text><rect class="compute" x="498" y="130" width="92" height="28" rx="4"/><text class="on-compute" x="544" y="148" text-anchor="middle">attn + gate</text><text class="cpu-label" x="585" y="48" text-anchor="end">route</text><rect class="cpu" x="590" y="30" width="8" height="28" rx="3"/><rect class="cpu" x="600" y="30" width="10" height="28" rx="3"/><text class="cpu-label" x="615" y="48">submit</text><rect class="compute" x="600" y="130" width="180" height="28" rx="4"/><text class="on-compute" x="690" y="148" text-anchor="middle">resident experts (hits)</text><rect class="stream" x="610" y="80" width="160" height="28" rx="4"/><text class="on-stream" x="690" y="98" text-anchor="middle">stream misses</text><text class="cpu-label" x="765" y="48" text-anchor="end">reap</text><rect class="cpu" x="770" y="30" width="10" height="28" rx="3"/><rect class="compute" x="783" y="130" width="91" height="28" rx="4"/><text class="on-compute" x="828" y="148" text-anchor="middle">misses</text><line class="axis" x1="110" y1="180" x2="880" y2="180"/><text x="110" y="200">NVMe reads hide behind resident compute — GPU stays nearly fully booked</text><text x="822" y="200">time →</text></svg>

In each mixture-of-experts layer:

1. **GPU:** attention, then the router's gate logits.
2. **CPU, `route`:** reads the gate logits and picks the top-k experts for each token
   (sigmoid, bias, top-k), then merges the picks across tokens.
3. **CPU, `submit`:** looks up each chosen expert in the cache. A hit gets a ticket that is
   already satisfied. A miss gets a slot and an NVMe read, and a ticket that the read will
   satisfy.
4. **GPU, `hits`:** runs the resident experts immediately, while the misses stream in.
5. **CPU, `reap`:** collects completed reads and releases their tickets.
6. **GPU, `misses`:** runs the experts that just arrived. Every expert launch waits on its
   ticket, so there is no "is it resident?" branch.

**The overlap is the whole design.** A model that fits in memory is not a separate mode.
It is the trivial case of the streaming path, where everything happens to be resident.

## What makes it different

- **Built for unified memory.** rivoli targets ROCm systems where the GPU reads system
  RAM directly, so there is no separate VRAM copy to manage. When a hardware-specific
  path is faster, it wins over portability
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
