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

<svg class="rv-overlap" viewBox="0 0 890 200" role="img" aria-label="Timeline of two MoE layers: the GPU runs attention and the gate; the CPU routes, looks up the expert cache and submits NVMe reads for the misses; the GPU runs the resident experts while the misses stream in; the CPU reaps the completed reads; the GPU runs the experts that just arrived."><rect class="lane" x="110" y="20" width="770" height="40" rx="6"/><text x="0" y="45">CPU</text><rect class="lane" x="110" y="70" width="770" height="40" rx="6"/><text x="0" y="95">NVMe → RAM</text><rect class="lane" x="110" y="120" width="770" height="40" rx="6"/><text x="0" y="145">GPU</text><rect class="compute" x="120" y="126" width="52" height="28" rx="4"/><text class="on-compute" x="125" y="145">attn</text><rect class="cpu" x="176" y="26" width="48" height="28" rx="4"/><text class="on-cpu" x="181" y="45">route</text><rect class="cpu" x="228" y="26" width="56" height="28" rx="4"/><text class="on-cpu" x="233" y="45">submit</text><rect class="stream" x="288" y="76" width="110" height="28" rx="4"/><text class="on-stream" x="293" y="95">read misses</text><rect class="compute" x="288" y="126" width="60" height="28" rx="4"/><text class="on-compute" x="293" y="145">hits</text><rect class="cpu" x="402" y="26" width="38" height="28" rx="4"/><text class="on-cpu" x="407" y="45">reap</text><rect class="compute" x="444" y="126" width="52" height="28" rx="4"/><text class="on-compute" x="449" y="145">misses</text><text x="120" y="12">layer 1</text><rect class="compute" x="500" y="126" width="52" height="28" rx="4"/><text class="on-compute" x="505" y="145">attn</text><rect class="cpu" x="556" y="26" width="48" height="28" rx="4"/><text class="on-cpu" x="561" y="45">route</text><rect class="cpu" x="608" y="26" width="56" height="28" rx="4"/><text class="on-cpu" x="613" y="45">submit</text><rect class="stream" x="668" y="76" width="110" height="28" rx="4"/><text class="on-stream" x="673" y="95">read misses</text><rect class="compute" x="668" y="126" width="60" height="28" rx="4"/><text class="on-compute" x="673" y="145">hits</text><rect class="cpu" x="782" y="26" width="38" height="28" rx="4"/><text class="on-cpu" x="787" y="45">reap</text><rect class="compute" x="824" y="126" width="52" height="28" rx="4"/><text class="on-compute" x="829" y="145">misses</text><text x="500" y="12">layer 2</text><line class="axis" x1="110" y1="176" x2="880" y2="176"/><text x="810" y="194">time →</text></svg>

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
