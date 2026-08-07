---
scope: engine
status: live
verdict: The --features otlp instrument: its three switches, what the engine emits, and how to read a trace. Verified end to end 2026-08-01, after it had stopped compiling.
---

# Viewing a decode run as a trace (Grafana)

The `class/tok` line in [how-to-measure.md](how-to-measure.md) reports **scalars**: `io-wait 183ms` tells you the
reaper waited that long, not *when*, so nothing in it can show that the wait happened
underneath the decode thread's GPU waits. That overlap is the bet the whole streaming
design is placed on, and a sum cannot settle it.

`--spans` turns the same measurements into **real OTLP spans with real start/end
times**, on one timeline across both threads. In a trace viewer the `io-wait/uring-reap`
bars sit *under* the `gpu-wait/*` bars when fetch is hidden, and *beside* them when it is
not — which is the picture, not a percentage.

## The feature, and the two switches that are not it

**Verified end to end 2026-08-01 — and it had been broken.** See "How it rotted" below
before trusting anything here.

| switch | when | effect if absent |
|---|---|---|
| `--features otlp` | build | `export()` is a no-op stub; the opentelemetry stack is not linked. Everything below is inert. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | run | log-only. No collector needed, nothing is sent, the run is unaffected. |
| `--spans [<BUDGET>]` | run | **metrics still export; traces do not.** The interval recorder never initialises, so there is nothing to build a timeline from. Bare `--spans` means 5000. |

The feature is off by default because the opentelemetry stack is heavy, and it is a
build-time gate for the same reason `teacher-forcing` and `pred-probe` are: an instrument
that can be linked into a stock binary eventually is.

**The span budget was `RIVOLI_SPANS` until 2026-08-01, and that broke the project's own
rule** — an instrument goes behind a feature *and a flag, never an env var*, because an env
var is invisible to `--help`, absent from the command line
`docs/measurement/benchmarks.md` records, and silently active in a build that looks stock.
It is `--spans` now, gated on the feature like `--ppl` and `--pred-probe` are on theirs, so
it does not appear in a build that cannot use it. **The env var is inert, not deprecated:**
a run with `RIVOLI_SPANS=5000` and no `--spans` exports metrics and a single empty root
span, verified 2026-08-01.

The one remaining env var, `OTEL_EXPORTER_OTLP_ENDPOINT`, keeps its exception by not being
ours to name — it is the OpenTelemetry standard variable that every collector, SDK and
deployment already speaks, and renaming it would cost more than the rule buys.

### How it rotted, and what that says

`--features otlp` **did not compile** on 2026-08-01. `src/telemetry.rs` recorded a
`cpu/tokio-poll` metric series from `ProfileSummary::launch_ms`, a field deleted when the
expert launches moved onto the compute stream — the removal swept the always-on code and
never reached the feature-gated code, because **nothing in CI builds this feature.** The
`rocm` and `vulkan` arms are what get built; `otlp` compiles only when someone asks for it,
and nobody had since the field went.

Two things followed from fixing it, both worth carrying:

- **The `cpu/tokio-poll` series is gone**, not renamed. `cpu` is now exactly
  `launch + route + submit`. The dashboard's CPU-breakdown panel described the old sum and
  now says so.
- **A false truncation warning was firing.** Prefill is one forward pass over 78 layers,
  ~390 intervals, so any budget under that tripped the recorder's cap *before generation
  started*. `plan()` discards the prefill records but did not clear the flag that describes
  them — so a run whose sampled timeline was complete still ended with "the exported
  timeline is missing the LATER sampled tokens" and a `spans_truncated` attribute on the
  root. Reproduced at `--spans 200`, fixed, and the fix is what makes the measured run
  below trustworthy.

### The measured run, 2026-08-01

`--mode int3-vq --cache-policy lru --attn dense --max-mem 100 -bench 8`, exporting to a
local sink that counts POSTs:

```
--spans: budget 5000 / (~472 spans/tok x 8 tok) -> every 1th token sampled (8 of 8)
/v1/traces:  2186 POSTs, 670197 bytes
/v1/metrics:    1 POST,    1670 bytes
```

No truncation warning; 3.52 tok/s, within the spread of the same config without it.

## What rivoli emits

| signal | what | needs |
|---|---|---|
| **traces** | `rivoli.decode` → `token N` → `layer L` → leaf intervals (`gpu-wait/*`, `io-wait/uring-reap`, `cpu/*`), each leaf tagged `thread=decode\|reaper`, with true start/end times | `--spans` given |
| **metrics** | `rivoli_ms_per_tok{class,thread}` plus `rivoli_tok_per_s`, `rivoli_hit_pct`, `rivoli_gb_per_tok`, `rivoli_miss_per_tok`, `rivoli_tokens`, `rivoli_degenerate` (+ `rivoli_loop_period`/`_repeats` when it fires) — **every one of them additionally labelled `{mode,cache_policy,attn,max_mem_gib,mtp}`**, see below | always, when OTLP is on |

> **CORRECTED 2026-08-01.** This row also listed `rivoli_fetch_hidden_pct`. That gauge is
> **deleted** — it and `exposed_fetch_ms` were removed on the authority of their own doc
> comment, which said the number was "SUBSTANTIALLY OVERSTATED — an upper bound, and not a
> tight one… prefer `io_wait_ms`, which is measured". It was derived from
> `moe_wall − compute_gpu`, and `compute_gpu` under-counted, so the quotient reported ~96%
> against a true overlap ceiling of ≤57% and printed 99% for a configuration that decoded at
> half speed. **The honest series for "did fetch cost wall" is `rivoli_ms_per_tok{class="io-wait"}`.**
> The Grafana stat tile that queried the gauge was repointed there the same day. Nothing was
> lost: the measurement that condemned the metric is in
> [`investigations/perf-evidence.md`](../investigations/perf-evidence.md) and
> [`how-to-measure.md`](how-to-measure.md).

Both go out over OTLP/HTTP at run end. Metrics exist because span attributes are
*searchable* but not *chartable* — Grafana cannot draw `gpu_wait_ms` over time from a span
tag, so a dashboard built on traces alone can only show one run at a time.

### Metric labels — which run this is

**Added 2026-08-01** (`investigations/otlp-modernization.md` §2a, checklist item 1). Every
exported datapoint carries the run's identity:

| label | source | values |
|---|---|---|
| `mode` | `--mode` | `int4` · `int3-vq` · `hybrid` |
| `cache_policy` | `--cache-policy` | `lru` · `2q` · `arc` |
| `attn` | `--attn` | `dense` · `dsa` · `streaming {…}` · `misa {…}` — the parameterised two carry their own arguments |
| `max_mem_gib` | `--max-mem` | the integer budget, **omitted entirely** when the budget auto-sizes |
| `mtp` | `--mtp-min-conf` | the gate to 2dp, or `off`. ALWAYS present, unlike the budget: speculative decode is on by default, ungated it is a 0.93-0.95x LOSS and at 0.8 a 1.108x win, so two runs differing only here are not comparable. "Off" is a state the run was in, not a missing measurement. |

They come from `RunInfo`, the same struct the root span's attributes come from, so a trace
and its metrics cannot disagree about what the run was.

**Before this they were not there at all, and the omission was worse than a missing
filter.** `rivoli_tok_per_s` from `--mode hybrid --max-mem 115` and from `--mode int3-vq
--max-mem 70` were the *same Prometheus series* — two configurations averaged into one line,
with nothing on the chart to say so. The paragraph below ("you search by which mode / which
policy / which budget, then read the numbers off a chart") described the spans and was
simply false of the chart.

**They are datapoint attributes, not resource attributes, and that is not an implementation
detail.** Alloy's `otelcol.exporter.prometheus` moves resource attributes into `target_info`
and leaves them off the series unless `resource_to_telemetry_conversion` is enabled — a
switch in a collector config file outside this repo. Run identity attached to the `Resource`
would therefore look right in the code, export cleanly, and produce a dashboard whose `mode`
picker is empty: the same "empty graph rather than an error" failure that
`add_metric_suffixes` gets its own warning for in the Alloy config below. Only
`service.name` and `service.version` ride on the Resource.

**Cardinality is bounded by the command line, and must stay that way.** One datapoint per
run, every value a flag argument from a closed set; their product is the benchmark matrix,
which is the thing worth charting. `model` and `prompt` are deliberately *not* labels — they
are unbounded, and both are already root-span attributes, where cardinality is free. Nothing
per-token, per-layer or per-expert may be added beside these; that view is the trace, which
costs a `Vec` push instead of a time series.

### Span attributes — identity, not measurements

Each level carries what identifies it, and **deliberately not** the numbers:

| span | attributes |
|---|---|
| `rivoli.decode` | `rivoli.{model,mode,cache_policy,attn,moe_gain,max_mem_gib,bench_tokens,prompt,sinks,window,misa_heads,tokens_generated,degenerate}` (+ `loop_{period,repeats,start}` when degenerate) |
| `token N` | `rivoli.token_index`, `rivoli.token_id` |
| `layer L` | `experts.{cold,warm}.{int4,int3_vq}`, `experts.cold`, `experts.total` |
| leaf | `thread` |

**The root carries the run's arguments, not its metrics.** An earlier version put
`tok_per_s`, `wall_ms_per_tok` and friends on it; those went out as metrics too, which made
two sources of truth that can drift, and they are not what you search a trace *by*. You
search by "which mode / which policy / which budget", then read the numbers off a chart.
`max_mem_gib`, `bench_tokens` and `prompt` are emitted **only when set** — a `0 GiB` budget
or `0` generated tokens would read as a setting that did something, rather than as absent.

**Since 2026-08-01 the chart carries the same three** as metric labels (previous section),
which is what makes that sentence true rather than aspirational: the trace search and the
dashboard filter now name identical values, and `max_mem_gib` is omitted-when-unset on both
sides for the same reason.

> **CORRECTED 2026-08-01.** The row above also listed `route_j`, `route_m`, `2q_kin_pct` and
> `2q_kout_pct`, and this paragraph said "`route_j`/`route_m` appear only under `top-m`".
> **The engine emits none of the four.** `route_j`/`route_m` went with `top-m` when that
> policy was retired 2026-07-30 (`investigations/cache-conditional-routing.md`);
> `2q_kin_pct`/`2q_kout_pct` went 2026-08-01 with the `--2q-kin`/`--2q-kout` flags, which
> were deleted because `TwoQSplit::default()` was the only value anything ever passed — a
> constant is not run identity, so labelling every span with it bought nothing. `topk_path`
> had already been corrected out of this table for the same class of reason. The
> conditional-emission rule the paragraph was illustrating is real and still holds; only its
> examples were dead.
>
> The attribute set is thin on **run identity** — no artifact hash, no git rev, no host —
> and that is a known gap with a proposal attached, not an oversight:
> [`investigations/otlp-modernization.md`](../investigations/otlp-modernization.md) §2a.
>
> **UPDATE 2026-08-01.** §2a's *metric* half is built — mode / cache_policy / attn /
> max_mem_gib are now labels on every datapoint (see "Metric labels" above). What is still
> missing is the artifact hash, the git rev and the host, and §2a proposes none of those:
> they are a gap on the **span** side with no proposal behind them, so do not read the
> shipped labels as having closed this.

**`layer L`'s expert composition is the point of the layer level.** Eight cold int4 experts
and eight warm int3-vq experts are different animals, and residency × format is the pair
that explains the layer's cost. It has to be sampled between the residency check and the
format decision — residency *after* `submit_layer` reads as all-resident, because that is
where misses get their slots.

## Where to send it

**Grafana does not ingest anything.** It queries stores. The receiver is an **Alloy gateway
at `192.168.2.62`**, and it is **up** — this section used to say the opposite, that the
gateway "still has to be stood up", which stopped being true and sat here as a standing
discouragement from using a pipeline that works.

Verified from the decode box, 2026-07-30:

| check | result |
|---|---|
| `POST http://192.168.2.62:4318/v1/traces` (empty `resourceSpans`) | **200** |
| `:4317` (OTLP/gRPC) | open |
| `http://192.168.2.62:12345/-/ready` (Alloy's own UI) | **"Alloy is ready."** |
| `https://grafana.hr-home.xyz/api/health` | **200** |

The table is what this box can probe; the end-to-end half — spans and metrics from a real
decode landing in this gateway and rendering in the Grafana dashboard — is confirmed by the
operator, who has the screenshots. Recorded with that attribution rather than as a
reproducible check, because a reader who reruns the table has NOT reproduced the pipeline.
**Do not point this at `grafana.hr-home.xyz`
(192.168.2.1)** — that host serves the Grafana UI on 443 and refuses 4317/4318, because
Grafana is the query layer, not the receiver. That confusion is the reason this paragraph
names an IP and a port rather than "the Grafana box".

Config below.

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
// every panel in the dashboard groups by, and so do the run-identity attributes
// (`mode`, `cache_policy`, `attn`, `max_mem_gib`) that tell two runs apart.
//
// That works because they are DATAPOINT attributes. Resource attributes take a
// different path: this exporter parks them in `target_info` and they never reach the
// series unless `resource_to_telemetry_conversion` is turned on here. Nothing rivoli
// filters a chart by is allowed to depend on that switch — which is why only
// `service.name`/`service.version` are on the Resource.
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

OTEL_SERVICE_NAME=rivoli \
OTEL_EXPORTER_OTLP_ENDPOINT=http://192.168.2.62:4318 \
./target/release/rivoli /var/db/rivoli/glm52-vq3-full \
    --mode hybrid --cache-policy lru --attn dense --max-mem 100 -bench 128 --spans 5000
```

`--spans` is what produces the timeline; without it the run still exports metrics, and the
trace is one empty `rivoli.decode` span.

Verified end-to-end against a local collector: **1501 trace POSTs + 1 metrics POST** for a
16-token run at a 1500-span budget, and against the live gateway at `192.168.2.62:4318`,
whose spans and metrics render in Grafana.

One POST per span, because the exporter is a `SimpleSpanProcessor` — which is why the Alloy
config above batches. At `--spans 5000` that is 5000 requests at run END, not during
decode, so it costs the measurement nothing.

## The dashboard

Import [`grafana-dashboard.json`](grafana-dashboard.json) (Dashboards →
New → Import). It prompts for the metrics and trace data sources. Thirteen panels:
headline stats, the **CLASS** row, the **PHASE** row, a CPU breakdown, the splits,
residency, and an embedded trace view.

**The one thing the dashboard is opinionated about:** the CLASS panel is explicitly
*unstacked* and says so in its description, because those series overlap — stacking them
would draw a total that does not exist. The PHASE panel *is* stacked, because those do
partition wall. Keeping the two rows visually distinct is the whole point of having both.

## Things that will bite

- **`--spans` is a span budget, spent by SAMPLING whole tokens across the run**
  (bare `--spans` means 5000). It used to record the first N intervals and stop,
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
  token recorded, `--spans` sized past the whole run, no OTLP endpoint so only the
  recording is timed): 338.5 → 339.0 ms/tok, min-of-2 interleaved. With the default
  sampling stride it is **+0.00%** — min-of-3 interleaved, 338.2 vs 338.2, against a
  0.4 ms within-arm spread. Recording is a `Vec` push behind a mutex; no exporter touches
  the hot path.
  Drop `--spans` for A/B timing runs regardless — docs/measurement/benchmarks.md's rule is that the
  instrument must not be in the arm, and "too small to measure" is not "zero".
- **The spans are the same intervals the scalars come from**, so the two views cannot
  disagree. If they do, that is a bug worth chasing.
- **Nothing in CI compiles `--features otlp`.** It broke once already for exactly that
  reason (see "How it rotted"). If you change `ProfileSummary` or `RunInfo`, build this arm
  before you believe you are done — the compiler is the only thing that will tell you, and
  only if you ask it:

  ```sh
  cargo build --release --features rocm,otlp    # the arm no other command covers
  ```

  The label set itself is covered without the feature: `RunInfo::labels` lives in
  always-compiled code and `telemetry::run_label_tests` runs under a plain
  `cargo test --release --features rocm`.
- **Series recorded before 2026-08-01 carry no run-identity labels**, so any query written
  as `rivoli_tok_per_s{mode="hybrid"}` silently excludes every older datapoint rather than
  erroring. That is the correct behaviour — those points genuinely could have been any mode
  — but if a panel looks empty for a period you know you measured, this is why. Do not
  paper over it with `or on() rivoli_tok_per_s`; that re-merges exactly what the labels were
  added to separate.
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
dimension we actually care about — is exactly what cannot be trusted. *(2026-08-01: tokio
is gone from the dependency graph — its entire use was `block_on` at eight sites — so the
thread set is decode / reaper plus the io_uring SQPOLL kernel thread. **Two eligible threads
instead of three does not fix process-directed delivery**, so the argument is unchanged.)*

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
(`src/fetch/stream.rs`) — it was a latent robustness gap regardless of profilers, since `reap`'s own
comment already grants that a signal can reach the reaper thread.

**Unverified, flagged:** whether `pprof-rs` unwinding from a signal handler through
`libamdhip64`/`libhsa-runtime64` is safe (no evidence either way), and whether
`pyroscope.ebpf` off-CPU cleanly attributes amdgpu ioctl waits.
