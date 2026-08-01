---
status: live
verdict: Why ROCm GPU profiling does not work on this part, and what to do instead. Only if you are attaching a profiler.
---

# GPU profiling into OTLP — what is possible on this box

Researched July 2026 for gfx1151 / Strix Halo. The goal that motivated it: localise the
**~38 ms/tok** that `class/tok` shows as host-blocked-but-GPU-idle (`gpu-wait` 95% of wall
against `rocm-smi`'s ~84% busy). Conclusions first, because two of them are surprising.

## 1. rocprofiler-sdk cannot attach here at all

Not "is awkward" — **cannot**. This is a Gentoo install with no `/opt/rocm`, and the
profiling hooks are compiled out of the runtime:

```
dev-util/hip/hip-7.2.0-r1.ebuild:171   -DHIP_ENABLE_ROCPROFILER_REGISTER=OFF
dev-util/hip/hip-7.2.0-r1.ebuild:182   -DUSE_PROF_API=OFF
dev-libs/rocr-runtime-7.2.0.ebuild:68  -DCMAKE_DISABLE_FIND_PACKAGE_rocprofiler-register=ON
```

Corroborated rather than assumed: `strings libamdhip64.so.7 | grep -ci rocprofiler` → **0**,
same for `libhsa-runtime64.so.1`. `rocprofv3`, `roctracer`, `librocprofiler-register` and
`amd-smi` are all absent, and **no `rocprofiler` ebuild exists in any configured repo**.

So rocprofiler-sdk means two patched ebuilds plus a from-source build **before** any rivoli
change — 1–3 days, mostly toolchain, with real risk of regressing a working GPU. The ~100
line C shim it would need afterwards is the easy part. Do not start here.

`roctracer` is not a fallback: `USE_PROF_API=OFF` removes exactly its HIP-side callbacks.
PC sampling is unsupported on gfx1151 regardless (gfx9/gfx12 only).

## 2. The clock-correlation problem does not exist on AMD

This is the good surprise, and it is what makes GPU spans cheap. Measured on this box:
`HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY` = 1e9, so ticks *are* nanoseconds, and

```
hsa_ns - CLOCK_MONOTONIC     = +311 ns   (read window 381 ns)
hsa_ns - CLOCK_MONOTONIC_RAW = -58.5 ms  (NOT this one)
```

**ROCm timestamps are `CLOCK_MONOTONIC` nanoseconds — the same clock Rust's `Instant` uses.**
`telemetry::spans`' existing `t0_mono`/`t0_wall` anchor converts them with the two lines
already in `record()`. No offset protocol, no drift correction.

**HIP events can be anchored to the host clock** without any new shim: record a reference
event, sync it, bracket with `Instant::now()`, then `hipEventElapsedTime(E_ref, E_i)` places
every later event on that timeline. Measured agreement **±1–2 µs with zero rate drift** over
2 s (necessarily — same counter), and it **works cross-stream**: a deliberate 2.000 ms host
gap between two streams reconstructed as 2.103 ms.

**Overhead, measured over 3000 kernels** — this is what decides the design:

| instrumentation | device timeline | vs none |
|---|---:|---:|
| none | 1.97 µs/kernel | — |
| 1 event / 16 kernels | 2.14 µs | **+8%** |
| 1 event / 4 kernels | 2.61 µs | +32% |
| 1 event / kernel | 4.72 µs | **+140%** |

**Per-kernel events are a Heisenberg instrument for this exact question**: one event costs
more device time than a small kernel, so it roughly doubles the inter-kernel interval we are
trying to measure. Per-phase (~1 in 16) is +8% and honest. `hipEventElapsedTime` readback is
0.485 µs, taken at a join we already pay.

**One precision trap:** `rivoli_event_elapsed` returns `f32`, so the quantum is
`delta × 2⁻²⁴` — 24 ns at a 400 ms delta, but **17.9 µs at 300 s**. Re-anchor once per
token, not once per run.

Also worth knowing: gfx1151 power-gates hard. After 200 ms idle a trivial marker round-trip
took **~1.6 ms** versus ~12 µs hot.

## 3. Nothing converts ROCm traces to OTLP, anywhere

Verified by enumeration, not by search. `rocprofv3` emits csv/json/pftrace/otf2/rocpd — no
OTLP, and no `otlp.py` in the converter. `rocprof-sys`: `grep -ril 'otlp\|opentelemetry'` →
**0 hits**. Perfetto→OTLP does not exist and Perfetto's own docs say it "is **not** a
distributed tracer in the vein of OpenTelemetry." There is no CUPTI→OTLP span bridge either,
and OTel semconv v1.43.0 has no `gpu.*` span conventions.

**Nobody emits GPU-side spans, which is why nobody solved the clock problem.** On AMD the
hardware already solved it (§2).

## 4. GPU metrics: one component works, and most of the numbers are lies

opentelemetry-collector-contrib v0.157.0 has **113 receivers and zero GPU receivers of any
vendor**; Alloy v1.18.0 has no GPU exporter. The one thing that works is **node_exporter's
`drm` collector** (AMD-only by implementation), which drops straight into the existing
gateway:

```alloy
prometheus.exporter.unix "gpu" { enable_collectors = ["drm"] }
prometheus.scrape "gpu" {
  targets    = prometheus.exporter.unix.gpu.targets
  forward_to = [otelcol.receiver.prometheus.gpu.receiver]
}
```

Skip AMD's Device Metrics Exporter: Prometheus-only, and **APU support was refused on the
record** (device-metrics-exporter#281, `wontfix`, "We only support MI2xx and MI3xx").

**What to publish, and what to refuse to publish.** Measured on this box:

- **VRAM is meaningless here.** `mem_info_vram_total` = 512 MiB (a BIOS UMA carve-out) while
  the real unified pool is `mem_info_gtt_total` = 116 GiB — **the same memory, 232×
  disagreement**. rivoli's weights live in GTT. Never sum the two.
- **Power is whole-SoC.** `power1_label` = `PPT` (package power tracking); on APUs that
  includes the CPU. `rocm-smi` calling it "Graphics Package Power" is misleading.
- **No memory bandwidth.** `mem_busy_percent` is absent. The only DRAM signal is
  `average_dram_reads/writes` inside `gpu_metrics`, which nothing exposes.
- **Fabricated values:** `rocm-smi` prints `Fan 0%` rather than N/A (reads as a dead fan);
  PCIe link speed is an on-die constant; vddgfx/vddnb read 0 loaded and idle.

Publishing 512 MiB of VRAM and a 0% fan is worse than publishing nothing — the same stance
TRACES.md takes on metric-name suffixes. Chart `node_drm_gpu_busy_percent`,
`node_drm_memory_gtt_used_bytes`, `node_hwmon_power_average_watt` (**labelled as SoC package
power incl. CPU**), temp and freq. Omit the rest.

## 5. The 38 ms target itself deserves a caveat

`gpu-wait` is stamped and trustworthy. The **~84% busy it is differenced against comes from
`rocm-smi`** — an instantaneous SMU register read, not a time-integrated counter, from a tool
that also reports 512 MiB of VRAM, a 0% fan, and throws `map::at` on every invocation. **The
error bars on 38 ms are wider than [PERF.md](PERF.md) implies.**

Two mechanisms are now measured and can be *subtracted* rather than guessed:

| mechanism | measured | per token |
|---|---:|---:|
| device-side kernel dispatch floor | 1.97 µs/kernel | ~1500 kernels → **~3.0 ms** |
| host→GPU join tax | 11–20 µs/join | ~180 joins → **~2.7 ms** |

That is **~6 ms of the ~38**. A clean negative result came with it: `hipDeviceSynchronize`,
`hipStreamSynchronize` and `hipEventSynchronize` all cost the same ~11–20 µs, and
busy-spinning on `hipEventQuery` is **strictly worse** (13.7 vs 11.2 µs). There is no cheap
win from swapping sync primitives or scheduling flags.

**The remaining ~30 ms already has a named suspect, in our own source.** `src/gpu.rs` says
it: *"each partial launches only after its per-expert `sig.await` resolves on the host, so
the compute stream sits idle between host-gated launches and those bubbles fall inside this
span"* — which is why `compute_gpu_ns` is documented as an upper bound. **Per-expert
host-gated launch bubbles are the leading hypothesis, and they are measurable with events we
already have.**

## Recommendation

**Tier 1 — extend the replay model with absolute GPU spans (2–4 h, no new dependencies).**
Record `(name, stream, start_ms, end_ms)` against a per-token anchor; replay through the
existing `SpanBuilder::with_start_time/with_end_time` path with a `stream` attribute beside
the current `thread`. Instrument the per-expert partial first — a host-gated bubble shows up
as a *gap between adjacent GPU spans*, which is the only view that can prove it. Then the
per-layer block, so `cpu/launch` bars can be differenced against the GPU work they issued.
The existing `moe_ev_*` / `tail_ev_*` / `idx_ev_*` pairs already exist and just are not
replayed as spans.

Its most valuable output is not the flamegraph: it is a **measured GPU-busy total on the same
clock as `gpu-wait`**, which either confirms the 38 ms or dissolves it. Same 2–4 hours.

**Tier 2 — the `drm` collector (~1 h of Alloy config, zero rivoli changes).** §4.

**Tier 3 — rocprofiler-sdk, only if Tier 1 localises but cannot diagnose.** Its real
advantage is per-kernel granularity and kernel names for free; note that per-kernel events
perturb by +140%, so rocprofiler is arguably the *only* way to get true per-kernel data. 1–3
days, most of it toolchain. §1.

**Unverified, flagged:** whether flipping the two Gentoo cmake flags yields a working stack
(nobody in the tree does it); `gpu_busy_percent` linearity under load on gfx1151; whether the
occasional +17 µs anchor outlier is purely sync-window noise.
