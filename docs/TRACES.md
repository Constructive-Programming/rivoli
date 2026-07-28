# Viewing a decode run as a trace (Grafana)

The `class/tok` line in [PERF.md](PERF.md) reports **scalars**: `io-wait 183ms` tells you the
reaper waited that long, not *when*, so nothing in it can show that the wait happened
underneath the decode thread's GPU waits. That overlap is the bet the whole streaming
design is placed on, and a sum cannot settle it.

`RIVOLI_SPANS` turns the same measurements into **real OTLP spans with real start/end
times**, on one timeline across both threads. In a trace viewer the `io-wait/uring-reap`
bars sit *under* the `gpu-wait/*` bars when fetch is hidden, and *beside* them when it is
not — which is the picture, not a percentage.

## What rivoli emits

| signal | what | needs |
|---|---|---|
| **traces** | `rivoli.decode` → `token N` → `layer L` → leaf intervals (`gpu-wait/*`, `io-wait/uring-reap`, `cpu/*`), each leaf tagged `thread=decode\|reaper`, with true start/end times | `RIVOLI_SPANS` set |
| **metrics** | `rivoli_ms_per_tok{class,thread}` plus `rivoli_tok_per_s`, `rivoli_hit_pct`, `rivoli_gb_per_tok`, `rivoli_miss_per_tok`, `rivoli_fetch_hidden_pct`, `rivoli_tokens` | always, when OTLP is on |

Both go out over OTLP/HTTP at run end. Metrics exist because span attributes are
*searchable* but not *chartable* — Grafana cannot draw `gpu_wait_ms` over time from a span
tag, so a dashboard built on traces alone can only show one run at a time.

### Span attributes — identity, not measurements

Each level carries what identifies it, and **deliberately not** the numbers:

| span | attributes |
|---|---|
| `rivoli.decode` | `rivoli.{model,mode,cache_policy,attn,topk_path,moe_gain,max_mem_gib,bench_tokens,prompt,route_j,route_m,2q_kin_pct,2q_kout_pct,sinks,window,misa_heads,tokens_generated}` |
| `token N` | `rivoli.token_index`, `rivoli.token_id` |
| `layer L` | `experts.{cold,warm}.{int4,int3_vq}`, `experts.cold`, `experts.total` |
| leaf | `thread` |

**The root carries the run's arguments, not its metrics.** An earlier version put
`tok_per_s`, `wall_ms_per_tok` and friends on it; those went out as metrics too, which made
two sources of truth that can drift, and they are not what you search a trace *by*. You
search by "which mode / which policy / which budget", then read the numbers off a chart.
`route_j`/`route_m` appear only under `top-m` and `max_mem_gib` only when set — a `(4, 9)`
next to `lru`, or a `0 GiB` budget, would read as a setting that did something.

**`layer L`'s expert composition is the point of the layer level.** Eight cold int4 experts
and eight warm int3-vq experts are different animals, and residency × format is the pair
that explains the layer's cost. It has to be sampled between the residency check and the
format decision — residency *after* `submit_layer` reads as all-resident, because that is
where misses get their slots.

## Where to send it

**Grafana does not ingest anything.** It queries stores. `alloy-gateway.hr-home.xyz` is the
intended receiver — as of writing it **does not resolve** (no DNS record in the `hr-home.xyz`
zone) and no OTLP port is open on `grafana.hr-home.xyz` (192.168.2.1), so the gateway still
has to be stood up. Config below.

### Alloy gateway

```alloy
// otlp in ---------------------------------------------------------------
otelcol.receiver.otlp "in" {
  http { endpoint = "0.0.0.0:4318" }
  grpc { endpoint = "0.0.0.0:4317" }
  output {
    traces  = [otelcol.processor.batch.default.input]
    metrics = [otelcol.processor.batch.default.input]
  }
}

// Batching matters here: rivoli's exporter is a SimpleSpanProcessor, so it sends
// ONE span per HTTP request. Alloy coalescing them is what keeps a 5000-span run
// from becoming 5000 writes against Tempo.
otelcol.processor.batch "default" {
  output {
    traces  = [otelcol.exporter.otlp.tempo.input]
    metrics = [otelcol.exporter.prometheus.mimir.input]
  }
}

otelcol.exporter.otlp "tempo" {
  client { endpoint = "tempo:4317", tls { insecure = true } }
}

// OTLP gauges -> Prometheus remote-write. `rivoli.ms_per_tok` arrives as
// `rivoli_ms_per_tok`; the `class`/`thread` attributes become labels, which is what
// every panel in the dashboard groups by.
otelcol.exporter.prometheus "mimir" {
  // Keep the names as the dashboard expects them. With suffixes on, the exporter
  // appends unit/type to every series and `rivoli_ms_per_tok` silently becomes
  // something else — every panel then queries a metric that does not exist and draws
  // an empty graph rather than an error, which is the worst possible failure mode.
  add_metric_suffixes = false
  forward_to = [prometheus.remote_write.mimir.receiver]
}
prometheus.remote_write "mimir" {
  endpoint { url = "http://mimir:9009/api/v1/push" }
}
```

Then add two Grafana data sources: **Tempo** (`http://tempo:3200`) and **Prometheus**
(`http://mimir:9009/prometheus`).

### If you would rather not run Mimir

Point `prometheus.remote_write` at any Prometheus with `--web.enable-remote-write-receiver`,
or drop the metrics branch and let Alloy expose them for scrape instead. The dashboard only
needs *a* Prometheus-compatible source.

## Recording a run

```sh
cargo build --release --features rocm,otlp --bin rivoli

RIVOLI_SPANS=5000 \
OTEL_SERVICE_NAME=rivoli \
OTEL_EXPORTER_OTLP_ENDPOINT=http://alloy-gateway.hr-home.xyz:4318 \
./target/release/rivoli /var/db/rivoli/glm52-vq3-full \
    --mode hybrid --cache-policy lru --attn dense --max-mem 100 -bench 128
```

Verified end-to-end against a local collector: **1501 trace POSTs + 1 metrics POST** for a
16-token run at `RIVOLI_SPANS=1500`.

## The dashboard

Import [`grafana-rivoli-dashboard.json`](grafana-rivoli-dashboard.json) (Dashboards →
New → Import). It prompts for the metrics and trace data sources. Thirteen panels:
headline stats, the **CLASS** row, the **PHASE** row, a CPU breakdown, the splits,
residency, and an embedded trace view.

**The one thing the dashboard is opinionated about:** the CLASS panel is explicitly
*unstacked* and says so in its description, because those series overlap — stacking them
would draw a total that does not exist. The PHASE panel *is* stacked, because those do
partition wall. Keeping the two rows visually distinct is the whole point of having both.

## Things that will bite

- **`RIVOLI_SPANS` is a span budget, spent by SAMPLING whole tokens across the run**
  (default 5000 if set to a non-number). It used to record the first N intervals and stop,
  which meant the timeline only ever showed the **cold start** — the least representative
  part of a decode, while the cache is still filling — and silently presented it as the
  run. Now the stride is `ceil(ngen × per_tok / budget)`, prefill is discarded at the same
  point the profile counters are rebased, and the log states the plan:
  `budget 5000 / (~472 spans/tok x 128 tok) -> every 13th token sampled (10 of 128)`.
  Hitting the cap now means the stride under-estimated spans/token, and says so.
- **Whole tokens, not individual intervals.** Sampling intervals would leave half-built
  layers whose synthesised parent spans lie about their own children.
- **`per_tok` is derived, not measured.** Six leaf spans per MoE layer — `cpu/launch`,
  `gate-d2h`, `route-into`, `submit-layer`, `end-of-layer-sync`, and the reaper's
  `io-wait/uring-reap`. An earlier version discovered it at runtime by timing token 0; that
  calibration fired *during prefill*, before the token count was known, and planned the
  whole run off `ngen = 0`. It is a property of the model, so it is now computed from
  `n_layers`. Round the estimate UP: a long stride samples fewer tokens, a short one
  truncates the tail.
- **One HTTP POST per span.** The exporter is a `SimpleSpanProcessor` (deliberately: no
  async runtime of ours), so 5000 spans is 5000 blocking round-trips at run end. Fine to
  localhost, slow to a remote Tempo. It happens *after* the decode and after the numbers
  are taken, so it never perturbs a measurement — but if it is painful, run a local
  `otel-collector` with a `batch` processor and forward from there.
- **Measured cost: +0.15% on wall, and that is the FULLY-instrumented figure** (every
  token recorded, `RIVOLI_SPANS` sized past the whole run, no OTLP endpoint so only the
  recording is timed): 338.5 → 339.0 ms/tok, min-of-2 interleaved. With the default
  sampling stride it is **+0.00%** — min-of-3 interleaved, 338.2 vs 338.2, against a
  0.4 ms within-arm spread. Recording is a `Vec` push behind a mutex; no exporter touches
  the hot path.
  Unset `RIVOLI_SPANS` for A/B timing runs regardless — benchmarks.md's rule is that the
  instrument must not be in the arm, and "too small to measure" is not "zero".
- **The spans are the same intervals the scalars come from**, so the two views cannot
  disagree. If they do, that is a bug worth chasing.
- **`Span::end()` is a trap.** It is `end_with_timestamp(now())` and it silently discards
  the builder's `with_end_time`. The first version of this used it, so every child was
  stamped as ending at *export* time — they all overlapped and Tempo rendered ~4000 spans
  as one collapsed nest. Always `end_with_timestamp(rec.end)`. Same for the root, which
  additionally needs explicit bounds spanning its children: a parent whose window does not
  contain its children is the other half of that same broken waterfall.
- **A flat span list is not a waterfall.** Emitting the leaves as direct siblings of the
  root is technically a trace and practically unreadable. The export synthesises
  `token N` → `layer L` levels from the leaves' own bounds (so they cannot disagree with
  them); `spans::mark()` supplies the position with two relaxed atomic stores, and the
  reaper reads them too, so its io-wait lands under the layer whose batch it is servicing.
  Verified by decoding the exported protobuf: depth histogram `{0:1, 1:2, 2:106, 3:597}`,
  600 distinct end times, 0 orphaned parent references.

## What to actually look at

1. **Is fetch hidden?** `io-wait/uring-reap` bars should overlap `gpu-wait/end-of-layer-sync`
   and the MoE region. Where they stick out past a token's GPU work, that is exposed fetch
   — the only fetch that costs wall.
2. **The ~38 ms/tok of blocked-but-idle GPU.** `gpu-wait` is 95% of wall while `rocm-smi`
   reports ~84% busy. The gaps between adjacent `cpu/launch` and `gpu-wait/*` bars are
   where that lives, and the timeline is the only view that can localise it.
3. **`cpu/launch` per layer.** 78 bars per token; a long one is a layer paying more driver
   time than its neighbours.

---

# Pyroscope — researched, and the answer is no

**Decision: do not wire a Pyroscope SDK into rivoli.** Recorded here so it is not
re-litigated; the reasoning is about *this* workload, so it changes if the workload does.

**The blocking fact.** `pprof-rs` — the backend every Rust Pyroscope integration uses —
samples on `ITIMER_PROF`, which decrements **only while the process burns user+system CPU**.
rivoli spends 95% of wall blocked in HIP joins. At 6.2 ms CPU per 338 ms token and a nominal
100 Hz, that is **~0.6 samples per token, ~1.9 samples/second** for the whole process. A
15-second upload is ~30 samples. That is not a flame graph.

It is worse than sparse: `ITIMER_PROF` is **process-directed**, so the kernel delivers
SIGPROF to an arbitrary eligible thread and the sample is attributed wherever it lands, not
to the thread that burned the CPU (the documented source of Go's profile skew,
golang/go#14434). With decode / reaper / tokio threads, the `thread` attribution — the one
dimension we actually care about — is exactly what cannot be trusted.

**Span profiles are not available to us, twice over.** Grafana supports them for Go, Java,
Ruby, .NET and Python; **Rust is absent**, and `grafana/pyroscope-rs#172` has sat open since
2024-07 with no reply. Independently, they require the profiler label to be live *while the
sampler fires* — and our spans are replayed at run end from cheaply-recorded intervals
precisely so nothing touches the hot path. Adopting span profiles would mean inverting the
design this file exists to describe.

**OTLP profiles would be the clean answer and do not exist yet.** The Profiles signal went
public Alpha 2026-03-26, `opentelemetry-rust` has no Profiles API at any version (0.32 is
current), and `otelcol.receiver.otlp` in Alloy 1.18 exposes only `logs`/`metrics`/`traces` —
no `profiles` output. Revisit when Rust gets an SDK.

### What to do instead, in value order

1. **Profile the GPU — it is the 95%.** `rocprofv3` / `rocprof-sys` for per-kernel duration,
   occupancy and launch gaps. Better still, feed roctracer callbacks into the existing span
   replay so kernels land on the same timeline as the `decode`/`reaper` spans. This composes
   with the design instead of fighting it, and it is where the ~38 ms/tok of
   blocked-but-GPU-idle actually lives.
2. **Profile the 6.2 ms of host CPU ad hoc, not continuously.** `samply record` or
   `perf record -F 999 -g --per-thread` on one run: per-thread, no `ITIMER_PROF` skew, no
   signal handler in our process, no cargo feature. ~15 minutes. Add standing instrumentation
   only if it finds something.
3. **If off-CPU time in Grafana is genuinely wanted:** `pyroscope.ebpf` with
   `off_cpu_threshold = 0.5`, forwarded to `pyroscope.write`. Out-of-process (no SIGPROF risk,
   no cargo feature, unwinds Rust via `.eh_frame`), ~1 hour of Alloy config, needs root and
   the host PID namespace. Be honest that it will mostly restate what `class/tok` already
   reports: "blocked in `hipStreamSynchronize` → `ioctl` on the KFD fd."

### If it is adopted anyway

`pyroscope` **2.1.1** (2026-07-21) is alive and maintained by Grafana; `pyroscope_pprofrs` is
superseded — the backend is vendored into the main crate behind `backend-pprof-rs`. Push goes
to `<url>/push.v1.PusherService/Push`, received by Alloy's `pyroscope.receive_http`
(default port 8080). `BackendConfig { report_thread_name: true }` gives `decode`/`reaper`
labels without any tagging API. Budget 2–4 h behind the `trace` feature.

**One prerequisite, now already met.** SIGPROF interrupts `io_uring_enter`, which is *not* in
the `SA_RESTART`-able class. `Streamer::reap` always retried `EINTR`; `Streamer::submit` did
not, and would have turned a stray signal into a poisoned fetch. That inconsistency is fixed
(`src/stream.rs`) — it was a latent robustness gap regardless of profilers, since `reap`'s own
comment already grants that a signal can reach the reaper thread.

**Unverified, flagged:** whether `pprof-rs` unwinding from a signal handler through
`libamdhip64`/`libhsa-runtime64` is safe (no evidence either way), and whether
`pyroscope.ebpf` off-CPU cleanly attributes amdgpu ioctl waits.
