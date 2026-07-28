//! Decode-run telemetry: the always-on stdout PROFILE summary + an optional OTLP
//! span. Both are cheap — one emission at the end of a run, no per-token cost, no
//! GPU syncs (the underlying buckets ride the joins the forward pass already pays).
//! The expensive fine-grained audits + correctness probes live behind the `trace`
//! feature in the engine, not here.
//!
//! OTLP is opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT` (unset ⇒ log-only, no collector
//! needed) and exports a single `rivoli.decode` span synchronously at run end — no
//! async runtime.

/// Interval recorder — turns the class counters into **real spans** with real
/// start/end times, so a trace viewer (Tempo/Jaeger/Grafana) renders them on a timeline
/// and the OVERLAP is visible rather than inferred.
///
/// The counters elsewhere in this file are scalars: `io-wait 183.7ms` tells you the
/// reaper waited that long, but not *when*, so nothing can show that it happened
/// underneath the decode thread's GPU waits. That is the whole question the engine's
/// design turns on, and a sum cannot answer it. These records can.
///
/// **Cost is a `Vec` push behind a mutex — no exporter, no network, nothing per-span
/// during decode.** Intervals are replayed as spans with explicit timestamps only at
/// run end, which is why this can be on while the numbers it produces stay honest.
///
/// Off unless `RIVOLI_SPANS` is set (a span budget; default 5000 when empty).
///
/// The budget is spent by **sampling whole tokens spread across the run**, not by
/// recording the first N intervals and stopping. Taking the prefix meant the timeline
/// only ever showed the cold start — the least representative part of a decode, when the
/// cache is still filling — and silently claimed to be the run. The stride is
/// self-calibrating: token 0 is always recorded, its span count reveals the per-token
/// cost, and the stride for the rest falls out of `ngen x per_tok / budget`.
///
/// WHOLE tokens, deliberately: sampling individual intervals would leave half-built
/// layers whose synthesised parent spans lie about their own children.
pub mod spans {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
    use std::sync::Mutex;
    use std::time::{Instant, SystemTime};

    /// One closed interval on one thread, with the position it happened at so the
    /// export can rebuild a tree. A flat list of 1200 siblings is not a waterfall.
    pub struct Rec {
        pub name: &'static str,
        pub thread: &'static str,
        pub start: SystemTime,
        pub end: SystemTime,
        /// Decode-loop token index.
        pub tok: u32,
        /// The token id being processed at that index, for the token span's attribute.
        pub tok_id: u32,
        /// Layer index, or -1 for work outside the layer loop (the tail).
        pub layer: i32,
    }

    struct Log {
        /// Anchors the monotonic clock to a wall clock: `Instant` cannot be turned into
        /// the `SystemTime` OTLP needs, so both are sampled once and deltas are added.
        t0_mono: Instant,
        t0_wall: SystemTime,
        cap: usize,
        recs: Mutex<Vec<Rec>>,
        /// Set once the cap is hit, so the export can say so instead of presenting a
        /// truncated timeline as a complete one.
        truncated: AtomicBool,
    }

    static LOG: OnceLock<Option<Log>> = OnceLock::new();
    /// Where the decode loop currently is. Read by `record` on whatever thread calls it,
    /// so the reaper's io-wait inherits the token/layer it is servicing — approximate by
    /// construction (it is a different thread) but right in every case that matters,
    /// because the reaper only ever works on the batch the decode loop just submitted.
    /// Sampling plan: record every `STRIDE`-th token, whole.
    static STRIDE: AtomicU32 = AtomicU32::new(1);
    /// Whether the CURRENT token is being sampled. Computed once per token in `mark`, so
    /// `record` costs one atomic load rather than a division.
    static SAMPLED: AtomicBool = AtomicBool::new(true);
    static CUR_TOK: AtomicU32 = AtomicU32::new(0);
    static CUR_TOK_ID: AtomicU32 = AtomicU32::new(0);
    static CUR_LAYER: AtomicI32 = AtomicI32::new(-1);

    /// Called by the decode loop as it advances. Two relaxed stores; free when disabled.
    pub fn mark(tok: u32, tok_id: u32, layer: i32) {
        if log().is_none() {
            return;
        }
        if CUR_TOK.swap(tok, Ordering::Relaxed) != tok {
            let stride = STRIDE.load(Ordering::Relaxed).max(1);
            SAMPLED.store(tok % stride == 0, Ordering::Relaxed);
        }
        CUR_TOK_ID.store(tok_id, Ordering::Relaxed);
        CUR_LAYER.store(layer, Ordering::Relaxed);
    }

    /// Plan the sampling stride, and drop anything recorded so far.
    ///
    /// Called once, where the profile counters are rebased after prefill — which is the
    /// same reason: **prefill is warm-up and must not be in the sample.** An earlier
    /// version calibrated the stride by measuring token 0's span count at the first token
    /// boundary, which fired *during prefill*, before the token count was known, and
    /// planned the whole run off `ngen = 0`. `per_tok` is a property of the model
    /// (~5 spans per layer plus the tail), not something to discover at runtime, so the
    /// measurement step was both wrong and unnecessary.
    pub fn plan(ngen: usize, per_tok: usize) {
        let Some(l) = log() else { return };
        let stride = (ngen.max(1) as u64 * per_tok.max(1) as u64)
            .div_ceil(l.cap as u64)
            .clamp(1, u32::MAX as u64) as u32;
        STRIDE.store(stride, Ordering::Relaxed);
        SAMPLED.store(true, Ordering::Relaxed); // token 0 is always in the sample
        CUR_TOK.store(0, Ordering::Relaxed);
        if let Ok(mut v) = l.recs.lock() {
            v.clear();
        }
        if let Some(m) = LAYERS.get() {
            if let Ok(mut v) = m.lock() {
                v.clear();
            }
        }
        tracing::info!(
            "RIVOLI_SPANS: budget {} / (~{per_tok} spans/tok x {ngen} tok) -> every {stride}th \
             token sampled ({} of {ngen}), prefill discarded",
            l.cap,
            ngen.div_ceil(stride as usize),
        );
    }

    fn log() -> Option<&'static Log> {
        LOG.get_or_init(|| {
            let v = std::env::var("RIVOLI_SPANS").ok()?;
            let cap = v.trim().parse::<usize>().unwrap_or(5000).max(1);
            Some(Log {
                t0_mono: Instant::now(),
                t0_wall: SystemTime::now(),
                cap,
                recs: Mutex::new(Vec::with_capacity(cap.min(1 << 16))),
                truncated: AtomicBool::new(false),
            })
        })
        .as_ref()
    }

    /// True when recording is on. Callers use it to skip taking timestamps they would
    /// otherwise throw away.
    pub fn enabled() -> bool {
        log().is_some() && SAMPLED.load(Ordering::Relaxed)
    }

    /// Record a closed interval. `start`/`end` are monotonic instants from any thread;
    /// they are converted against the shared anchor so cross-thread spans land on one
    /// timeline — which is the point, since io-wait lives on the reaper thread.
    pub fn record(name: &'static str, thread: &'static str, start: Instant, end: Instant) {
        let Some(l) = log() else { return };
        if !SAMPLED.load(Ordering::Relaxed) {
            return;
        }
        // `saturating_duration_since` because an `Instant` taken before the anchor (a
        // span opened during construction) would otherwise panic on subtraction.
        let (s, e) = (
            l.t0_wall + start.saturating_duration_since(l.t0_mono),
            l.t0_wall + end.saturating_duration_since(l.t0_mono),
        );
        let Ok(mut v) = l.recs.lock() else { return };
        // Hard backstop. With sampling this should not trigger; if it does, the stride
        // under-estimated the per-token cost and the tail of the run is missing.
        if v.len() >= l.cap {
            l.truncated.store(true, Ordering::Relaxed);
            return;
        }
        v.push(Rec {
            name,
            thread,
            start: s,
            end: e,
            tok: CUR_TOK.load(Ordering::Relaxed),
            tok_id: CUR_TOK_ID.load(Ordering::Relaxed),
            layer: CUR_LAYER.load(Ordering::Relaxed),
        });
    }

    /// Per-(token, layer) expert composition, for the layer spans. Residency x format is
    /// the thing that explains a layer's cost — a layer of eight cold int4 experts is a
    /// different animal from eight warm vq3 ones — and it is knowable only here, between
    /// the residency check and the format decision.
    #[derive(Clone, Copy, Default)]
    pub struct LayerState {
        pub tok: u32,
        pub layer: i32,
        pub cold_i4: u16,
        pub warm_i4: u16,
        pub cold_vq3: u16,
        pub warm_vq3: u16,
    }

    static LAYERS: OnceLock<Mutex<Vec<LayerState>>> = OnceLock::new();

    /// Record a layer's expert composition. Same enable gate as `record`.
    pub fn record_layer(st: LayerState) {
        if log().is_none() || !SAMPLED.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(mut v) = LAYERS.get_or_init(|| Mutex::new(Vec::new())).lock() {
            // Same cap as the interval log, for the same reason.
            if v.len() < 200_000 {
                v.push(st);
            }
        }
    }

    /// Drain the per-layer expert composition.
    pub fn drain_layers() -> Vec<LayerState> {
        LAYERS
            .get()
            .and_then(|m| m.lock().ok().map(|mut v| std::mem::take(&mut *v)))
            .unwrap_or_default()
    }

    /// Drain the recorded intervals for export. Returns `(records, truncated)`.
    pub fn drain() -> (Vec<Rec>, bool) {
        let Some(l) = log() else {
            return (Vec::new(), false);
        };
        let truncated = l.truncated.load(Ordering::Relaxed);
        match l.recs.lock() {
            Ok(mut v) => (std::mem::take(&mut *v), truncated),
            Err(_) => (Vec::new(), truncated),
        }
    }
}

/// The run's arguments — what belongs on the root span. A span's attributes should say
/// WHAT THIS RUN WAS so traces can be found and compared; the numbers it produced are
/// metrics, and duplicating them here just makes two sources of truth that can drift.
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub model: String,
    pub mode: String,
    pub cache_policy: String,
    pub attn: String,
    pub topk_path: &'static str,
    /// `--max-mem <GiB>`, or None when the budget auto-sizes.
    pub max_mem_gib: Option<u64>,
    pub bench_tokens: Option<usize>,
    pub prompt: Option<String>,
    pub moe_gain: f32,
    /// `--cache-policy top-m`'s (J, M); None under every other policy, where a 0 would
    /// read as a measurement of something that did not happen.
    pub route_jm: Option<(usize, usize)>,
    pub two_q_kin: u32,
    pub two_q_kout: u32,
    pub sinks: usize,
    pub window: usize,
    pub misa_heads: usize,
}

/// End-of-run per-token performance summary — the PROFILE line and the OTLP span
/// fields. Built by the GPU engine from its always-on [`Profile`](crate::gpu) buckets.
#[derive(Debug, Clone, Copy)]
pub struct ProfileSummary {
    /// The `RIVOLI_TOPK` arm this run used (docs/NPU.md). Printed in the PROFILE line so
    /// a row pasted into benchmarks.md identifies its own arm — the engine names it once
    /// at construction, thousands of log lines earlier.
    pub topk_path: &'static str,
    pub tok_per_s: f64,
    pub hit_pct: f64,
    pub wall_ms: f64,
    pub route_ms: f64,
    /// The DSA indexer's GPU-timeline span; see [`Profile::idx_gpu_ns`](crate::gpu).
    pub idx_gpu_ms: f64,
    /// The host half of the selection (score D2H + CPU top-k + row upload) — GPU-idle time.
    /// `None` on the device arms, which never do it; a 0.0 there would read as a
    /// measurement of work that no longer exists (same reason as `swap_pct`).
    pub idx_host_ms: Option<f64>,
    /// Full layers per token that took the scoring path. Not `21` until context exceeds
    /// `index_topk`, and 0 whenever the indexer never scored — which is what gates the
    /// report line below.
    pub idx_layers_per_tok: f64,
    /// CPU wall of the overlapped MoE phase (the `block_on`).
    pub moe_wall_ms: f64,
    /// GPU-event span of the compute stream (partials + reduce).
    pub compute_gpu_ms: f64,
    /// Off-thread reaper fetch cost (queue→submit→reap all misses).
    pub fetch_wall_ms: f64,
    /// Aggregate expert-stream idle (tokio) — the SUM of per-expert load-waits across
    /// the ~9 concurrent tasks, so it over-counts the wall; a load-pressure gauge, not
    /// the exposed fetch (that's `moe_wall − compute_gpu`).
    pub load_wait_ms: f64,
    /// The stream's active launch cost (tokio poll).
    pub launch_ms: f64,
    /// Percent of `fetch_wall` buried behind compute: `1 − (moe_wall−compute_gpu)/fetch_wall`.
    pub fetch_hidden_pct: f64,
    pub miss_per_tok: f64,
    pub ms_per_miss: f64,
    pub gb_per_tok: f64,
    /// `--cache-policy top-m` only: the share of chosen expert slots that were NOT in
    /// the true top-K — the quality cost of cache-conditional routing, and per
    /// docs/CACHE_ROUTE.md "Counters" the one number you tune (J, M) against. `None`
    /// under lru/2q/arc, which never substitute; printing 0.0% there would read as a
    /// measurement of something that did not happen.
    pub swap_pct: Option<f64>,

    // ---- CLASS spans: what the machine was DOING. All directly measured ----
    // These OVERLAP and may sum to MORE than `wall_ms`. `io_wait_ms` runs on the reaper
    // thread concurrently with everything else, so it is not a share of wall. The
    // deliberate consequence: **no residual is reported.** An earlier `cpu` was
    // `wall − the waits`, which absorbed every error in the other terms and measured
    // nothing; unattributed time is now simply not shown rather than dressed up as a
    // number.
    /// Decode thread blocked in a device join — stamped at every blocking call.
    ///
    /// **NOT "the GPU was busy."** Against `rocm-smi`'s independent busy counter this
    /// reads ~95% of wall while the device reports ~84%: roughly 11 points is the host
    /// blocked on a GPU that is not executing (launch gaps, driver/queue-drain).
    pub gpu_wait_ms: f64,
    /// Reaper thread blocked in `io_uring` completions — measured at the ring, in
    /// `run_job`'s reap loop, excluding the queue/submit syscalls around it.
    /// Off-thread, so it overlaps the decode wall and can exceed it.
    pub io_wait_ms: f64,
    /// Host compute: the sum of `cpu_launch_ms + cpu_route_ms + cpu_submit_ms` plus the
    /// expert stream's tokio poll time. Every term stamped; none derived.
    pub cpu_ms: f64,
    /// Host time issuing kernel launches (per-layer attention/MLP block + the tail).
    pub cpu_launch_ms: f64,
    /// Host time in `route_into` — sigmoid/bias/top-k over 256 experts per MoE layer.
    pub cpu_route_ms: f64,
    /// Host time in `Pin::submit_layer` — residency, policy bookkeeping, read specs.
    pub cpu_submit_ms: f64,
    /// `moe_wall − compute_gpu`: fetch that could not hide behind compute. Kept for
    /// continuity with `fetch_hidden_pct`, but it is a host clock minus a GPU clock —
    /// prefer `io_wait_ms`, which is measured.
    pub exposed_fetch_ms: f64,
    /// The blocking half of `route_ms` (the gate-logits D2H).
    pub route_wait_ms: f64,
    /// The argmax D2H — the single call the entire tail phase hides behind.
    pub tail_wait_ms: f64,
    /// HIP-event BRACKET across final rmsnorm → lm_head → argmax. An upper bound on the
    /// tail's GPU work: measured 5.50 ms against a 4.66 ms microbench sum, so ~15% is
    /// inter-kernel gap.
    pub tail_gpu_ms: f64,
}

impl ProfileSummary {
    /// The always-on stdout PROFILE line. Under the async overlap, `moe_wall` is the
    /// real per-token MoE cost; `fetch_wall` is the fetch work, of which
    /// `fetch_hidden_pct` overlapped compute and only `load_wait` was exposed.
    pub fn report(&self) {
        let exposed = (self.moe_wall_ms - self.compute_gpu_ms).max(0.0);
        tracing::info!(
            // wall/route at 0.1 ms: the DSA selection A/B (docs/NPU.md) turns on deltas of
            // a few ms against a ~400 ms token, which 1 ms resolution rounds into noise.
            "PROFILE/tok [topk={}]: {:.1}ms wall | route {:.1}ms | moe {:.0}ms (gpu {:.0}ms) | fetch {:.0}ms ({:.0}% hidden, {:.0}ms exposed) | {:.2} miss, {:.2}ms/miss, {:.2} GB",
            self.topk_path,
            self.wall_ms,
            self.route_ms,
            self.moe_wall_ms,
            self.compute_gpu_ms,
            self.fetch_wall_ms,
            self.fetch_hidden_pct,
            exposed,
            self.miss_per_tok,
            self.ms_per_miss,
            self.gb_per_tok,
        );
        tracing::info!(
            "  stream/tok: load-wait {:.0}ms (Σ over ~9 concurrent tasks) | launch {:.1}ms (poll)",
            self.load_wait_ms,
            self.launch_ms,
        );
        // The CLASS view: the PROFILE line says WHERE the time is (phases); this says
        // WHAT it is. Every term is measured — none is a residual — so they OVERLAP and
        // need not sum to wall. `io-wait` is on the reaper thread and routinely exceeds
        // it. The `%` is therefore "of wall", not "share of wall", and unattributed time
        // is deliberately not shown.
        let pct = |ms: f64| 100.0 * ms / self.wall_ms.max(1e-9);
        tracing::info!(
            "  class/tok [spans overlap; no residual]: gpu-wait {:.1}ms ({:.0}% of wall) | \
             io-wait {:.1}ms ({:.0}%) | cpu {:.1}ms ({:.0}%)",
            self.gpu_wait_ms,
            pct(self.gpu_wait_ms),
            self.io_wait_ms,
            pct(self.io_wait_ms),
            self.cpu_ms,
            pct(self.cpu_ms),
        );
        tracing::info!(
            "    cpu = launch {:.1}ms + route {:.2}ms + submit {:.2}ms + tokio-poll {:.1}ms",
            self.cpu_launch_ms,
            self.cpu_route_ms,
            self.cpu_submit_ms,
            self.launch_ms,
        );
        // The two phase/class splits that motivated this view: `route` was a region
        // mixing a blocking D2H with host routing, and the whole `tail` phase was one
        // opaque wait with ~59% attributable to no kernel.
        tracing::info!(
            "  split/tok: route = {:.1}ms gpu-wait + {:.1}ms host-routing | tail wait {:.1}ms, of which {:.1}ms is GPU ({:.0}% overhead)",
            self.route_wait_ms,
            (self.route_ms - self.route_wait_ms).max(0.0),
            self.tail_wait_ms,
            self.tail_gpu_ms,
            100.0 * (self.tail_wait_ms - self.tail_gpu_ms).max(0.0) / self.tail_wait_ms.max(1e-9),
        );
        // DSA indexer decomposition (docs/NPU.md M0). Silent when the indexer never
        // scored — dense/streaming, or a context that stayed under `index_topk`, where a
        // row of zeros would read as a measurement of something that did not happen.
        if self.idx_layers_per_tok > 0.0 {
            let host = match self.idx_host_ms {
                Some(ms) => format!(
                    "host {ms:.1}ms (D2H+topk+upload) => {:.1}us per layer",
                    ms * 1e3 / self.idx_layers_per_tok,
                ),
                None => "host n/a (selection on device)".to_string(),
            };
            tracing::info!(
                "  indexer/tok: gpu {:.1}ms => {:.1}us per layer + {host} over {:.3} scoring layers",
                self.idx_gpu_ms,
                // Guarded non-zero by the `> 0.0` above, so this division is safe.
                self.idx_gpu_ms * 1e3 / self.idx_layers_per_tok,
                self.idx_layers_per_tok,
            );
        }
        if let Some(swap) = self.swap_pct {
            tracing::info!(
                "  route: swap {swap:.2}% of chosen slots were outside the true top-K \
                 (cache-conditional routing; hit% above is NOT comparable to a run \
                 without it — see docs/CACHE_ROUTE.md \"Risks\")"
            );
        }
    }
}

/// Export the decode summary to OTLP as one `rivoli.decode` span, when the `otlp`
/// feature is built AND `OTEL_EXPORTER_OTLP_ENDPOINT` is set. No-op otherwise (the
/// PROFILE line already logged the same numbers). Never fails the run — an export
/// error is warned and swallowed.
pub fn export_decode(summary: &ProfileSummary, tokens: usize, run: &RunInfo) {
    #[cfg(feature = "otlp")]
    if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some() {
        if let Err(e) = otlp::export(summary, tokens, run) {
            tracing::warn!("OTLP export failed ({e}); metrics logged only");
        }
    }
    #[cfg(not(feature = "otlp"))]
    let _ = (summary, tokens, run);
}

#[cfg(feature = "otlp")]
mod otlp {
    use super::{ProfileSummary, RunInfo};
    use anyhow::{Context, Result};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::{Span, TraceContextExt, Tracer, TracerProvider as _};
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;
    use std::collections::BTreeMap;
    use std::time::SystemTime;

    /// Build a one-shot tracer (blocking HTTP exporter + a SimpleSpanProcessor, so the
    /// span exports synchronously on `end()`/`shutdown()` — no async runtime of ours),
    /// emit the `rivoli.decode` span with the summary as attributes, and flush. The
    /// endpoint + protocol come from the standard `OTEL_*` env vars. Version is the
    /// build's `CARGO_PKG_VERSION`.
    pub fn export(summary: &ProfileSummary, tokens: usize, run: &RunInfo) -> Result<()> {
        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_http()
            .build()
            .context("build OTLP span exporter")?;
        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "rivoli".to_string());
        let provider = SdkTracerProvider::builder()
            .with_resource(
                Resource::builder()
                    .with_service_name(service_name)
                    .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
                    .build(),
            )
            .with_simple_exporter(exporter)
            .build();

        let tracer = provider.tracer("rivoli");
        // Drain BEFORE building the root: the root needs explicit start/end covering the
        // children, and `tracer.start()` would stamp it at export time — minutes after
        // the intervals it parents. A parent whose window does not contain its children
        // is what makes a waterfall render as one collapsed nest.
        let (recs, truncated) = super::spans::drain();
        let n = recs.len();
        let bounds = recs
            .iter()
            .fold(None::<(SystemTime, SystemTime)>, |acc, r| match acc {
                None => Some((r.start, r.end)),
                Some((lo, hi)) => Some((lo.min(r.start), hi.max(r.end))),
            });
        let mut span = match bounds {
            Some((lo, hi)) => tracer
                .span_builder("rivoli.decode")
                .with_start_time(lo)
                .with_end_time(hi)
                .start(&tracer),
            // No intervals recorded (RIVOLI_SPANS unset): a plain now-stamped span, which
            // is correct — there are no children for it to fail to contain.
            None => tracer.start("rivoli.decode"),
        };
        // WHAT THIS RUN WAS, not what it measured. The numbers went out as metrics —
        // repeating them here would make two sources of truth that drift, and they are
        // not what you search a trace by. These are.
        span.set_attribute(KeyValue::new("rivoli.model", run.model.clone()));
        span.set_attribute(KeyValue::new("rivoli.mode", run.mode.clone()));
        span.set_attribute(KeyValue::new("rivoli.cache_policy", run.cache_policy.clone()));
        span.set_attribute(KeyValue::new("rivoli.attn", run.attn.clone()));
        span.set_attribute(KeyValue::new("rivoli.topk_path", run.topk_path));
        span.set_attribute(KeyValue::new("rivoli.moe_gain", f64::from(run.moe_gain)));
        span.set_attribute(KeyValue::new("rivoli.2q_kin_pct", i64::from(run.two_q_kin)));
        span.set_attribute(KeyValue::new("rivoli.2q_kout_pct", i64::from(run.two_q_kout)));
        span.set_attribute(KeyValue::new("rivoli.sinks", run.sinks as i64));
        span.set_attribute(KeyValue::new("rivoli.window", run.window as i64));
        span.set_attribute(KeyValue::new("rivoli.misa_heads", run.misa_heads as i64));
        // Absent rather than zero where zero would read as a measurement: an auto-sized
        // budget is not "0 GiB", and (J, M) under lru is not "(0, 0)".
        if let Some(g) = run.max_mem_gib {
            span.set_attribute(KeyValue::new("rivoli.max_mem_gib", g as i64));
        }
        if let Some(n) = run.bench_tokens {
            span.set_attribute(KeyValue::new("rivoli.bench_tokens", n as i64));
        }
        if let Some(p) = &run.prompt {
            span.set_attribute(KeyValue::new("rivoli.prompt", p.clone()));
        }
        if let Some((j, m)) = run.route_jm {
            span.set_attribute(KeyValue::new("rivoli.route_j", j as i64));
            span.set_attribute(KeyValue::new("rivoli.route_m", m as i64));
        }
        // Tokens actually generated is run shape, not a measurement of speed.
        span.set_attribute(KeyValue::new("rivoli.tokens_generated", tokens as i64));

        // Rebuild the tree: decode -> token N -> layer L -> the leaf intervals. Emitting
        // the leaves as 1200 direct siblings of the root is technically a trace and
        // practically unreadable; the token/layer levels are what make a waterfall
        // navigable, and they are synthesised from the leaves' own bounds so they cannot
        // disagree with them.
        //
        // Every span here is closed with `end_with_timestamp`. `end()` would stamp
        // `now()` and silently discard the builder's `with_end_time`.
        let cx = opentelemetry::Context::current_with_span(span);
        let layer_states: BTreeMap<(u32, i32), super::spans::LayerState> = super::spans::drain_layers()
            .into_iter()
            .map(|st| ((st.tok, st.layer), st))
            .collect();
        let mut by_tok: BTreeMap<u32, Vec<super::spans::Rec>> = BTreeMap::new();
        for r in recs {
            by_tok.entry(r.tok).or_default().push(r);
        }
        for (tok_i, tok_recs) in by_tok {
            let Some((t_lo, t_hi)) = span_bounds(&tok_recs) else {
                continue;
            };
            // The token id, not just the index — the index says "the 7th step", the id
            // says which token the model was actually working on.
            let tok_id = tok_recs.first().map(|r| r.tok_id).unwrap_or(0);
            let mut tok_span = tracer
                .span_builder(format!("token {tok_i}"))
                .with_start_time(t_lo)
                .with_end_time(t_hi)
                .with_attributes([
                    KeyValue::new("rivoli.token_index", tok_i as i64),
                    KeyValue::new("rivoli.token_id", i64::from(tok_id)),
                ])
                .start_with_context(&tracer, &cx);
            let tok_cx = opentelemetry::Context::current_with_span(tok_span);

            let mut by_layer: BTreeMap<i32, Vec<super::spans::Rec>> = BTreeMap::new();
            for r in tok_recs {
                by_layer.entry(r.layer).or_default().push(r);
            }
            for (layer_i, layer_recs) in by_layer {
                // layer -1 is work outside the layer loop (the tail): hang it straight off
                // the token rather than inventing a "layer -1" level for it.
                let parent_cx = if layer_i < 0 {
                    tok_cx.clone()
                } else {
                    let Some((l_lo, l_hi)) = span_bounds(&layer_recs) else {
                        continue;
                    };
                    // Expert composition: residency x format. This is what explains a
                    // layer's cost — eight cold int4 experts and eight warm vq3 ones are
                    // different animals — and the counts are recorded at submit time
                    // because that is the only point where both are known.
                    let mut attrs = vec![KeyValue::new("rivoli.layer", layer_i as i64)];
                    if let Some(st) = layer_states.get(&(tok_i, layer_i)) {
                        attrs.push(KeyValue::new("experts.cold.int4", i64::from(st.cold_i4)));
                        attrs.push(KeyValue::new("experts.warm.int4", i64::from(st.warm_i4)));
                        attrs.push(KeyValue::new("experts.cold.int3_vq", i64::from(st.cold_vq3)));
                        attrs.push(KeyValue::new("experts.warm.int3_vq", i64::from(st.warm_vq3)));
                        attrs.push(KeyValue::new(
                            "experts.cold",
                            i64::from(st.cold_i4 + st.cold_vq3),
                        ));
                        attrs.push(KeyValue::new(
                            "experts.total",
                            i64::from(st.cold_i4 + st.warm_i4 + st.cold_vq3 + st.warm_vq3),
                        ));
                    }
                    let ls = tracer
                        .span_builder(format!("layer {layer_i}"))
                        .with_start_time(l_lo)
                        .with_end_time(l_hi)
                        .with_attributes(attrs)
                        .start_with_context(&tracer, &tok_cx);
                    let c = opentelemetry::Context::current_with_span(ls);
                    let mut ls = c.span();
                    ls.end_with_timestamp(l_hi);
                    c
                };
                for r in layer_recs {
                    let mut leaf = tracer
                        .span_builder(r.name)
                        .with_start_time(r.start)
                        .with_end_time(r.end)
                        .with_attributes([KeyValue::new("thread", r.thread)])
                        .start_with_context(&tracer, &parent_cx);
                    leaf.end_with_timestamp(r.end);
                }
            }
            let mut ts = tok_cx.span();
            ts.end_with_timestamp(t_hi);
        }
        let mut span = cx.span();
        span.set_attribute(KeyValue::new("spans_recorded", n as i64));
        // A truncated timeline that does not say so reads as a complete one.
        if truncated {
            span.set_attribute(KeyValue::new("spans_truncated", true));
            tracing::warn!(
                "RIVOLI_SPANS cap reached after {n} intervals — the sampling stride \
                 under-estimated spans/token, so the exported timeline is missing the \
                 LATER sampled tokens. Raise RIVOLI_SPANS or the per_tok estimate."
            );
        }
        match bounds {
            Some((_, hi)) => span.end_with_timestamp(hi),
            None => span.end(),
        }

        // Flush the simple processor's export before we return (the run ends here).
        provider.shutdown().context("flush OTLP spans")?;
        export_metrics(summary, tokens)?;
        Ok(())
    }

    /// Min start / max end over a set of records — the bounds a synthesised parent needs
    /// so its window actually contains its children.
    fn span_bounds(recs: &[super::spans::Rec]) -> Option<(SystemTime, SystemTime)> {
        recs.iter().fold(None, |acc, r| match acc {
            None => Some((r.start, r.end)),
            Some((lo, hi)) => Some((lo.min(r.start), hi.max(r.end))),
        })
    }

    /// Export the same summary as OTLP **metrics**, not just span attributes.
    ///
    /// Attributes on a span are searchable but not chartable: Grafana cannot draw
    /// `gpu_wait_ms` over time from them, so a dashboard built on traces alone can only
    /// show individual runs. These gauges give the time series. One export at run end —
    /// a `PeriodicReader` would need a background thread and there is nothing to sample
    /// between runs anyway, so the reader is flushed once and shut down.
    ///
    /// Every class gauge carries `class` and (where it applies) `thread`, so one panel
    /// can `sum by (class)` instead of hard-coding a query per metric.
    fn export_metrics(summary: &ProfileSummary, tokens: usize) -> Result<()> {
        use opentelemetry::metrics::MeterProvider as _;
        use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};

        let exporter = opentelemetry_otlp::MetricExporter::builder()
            .with_http()
            .build()
            .context("build OTLP metric exporter")?;
        let service_name =
            std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "rivoli".to_string());
        let provider = SdkMeterProvider::builder()
            .with_reader(PeriodicReader::builder(exporter).build())
            .with_resource(
                Resource::builder()
                    .with_service_name(service_name)
                    .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
                    .build(),
            )
            .build();
        let m = provider.meter("rivoli");

        // ms/token, the unit every number in the PROFILE line is already in.
        let per_tok = m.f64_gauge("rivoli.ms_per_tok").build();
        let mut g = |v: f64, class: &'static str, thread: &'static str| {
            per_tok.record(
                v,
                &[
                    KeyValue::new("class", class),
                    KeyValue::new("thread", thread),
                ],
            );
        };
        // The class axis — overlapping spans, so a stacked panel would LIE. The
        // dashboard draws these as separate lines for exactly that reason.
        g(summary.gpu_wait_ms, "gpu-wait", "decode");
        g(summary.io_wait_ms, "io-wait", "reaper");
        g(summary.cpu_ms, "cpu", "decode");
        g(summary.cpu_launch_ms, "cpu/launch", "decode");
        g(summary.cpu_route_ms, "cpu/route", "decode");
        g(summary.cpu_submit_ms, "cpu/submit", "decode");
        g(summary.launch_ms, "cpu/tokio-poll", "decode");
        // The phase axis — these DO partition wall, and may be stacked.
        g(summary.route_ms, "phase/route", "decode");
        g(summary.moe_wall_ms, "phase/moe", "decode");
        g(
            (summary.wall_ms - summary.route_ms - summary.moe_wall_ms).max(0.0),
            "phase/tail",
            "decode",
        );
        g(summary.wall_ms, "wall", "decode");
        // Sub-splits worth their own series.
        g(summary.route_wait_ms, "split/route-gpu-wait", "decode");
        g(summary.tail_wait_ms, "split/tail-wait", "decode");
        g(summary.tail_gpu_ms, "split/tail-gpu", "decode");
        g(summary.compute_gpu_ms, "split/moe-compute-gpu", "decode");
        g(summary.exposed_fetch_ms, "split/exposed-fetch", "reaper");

        m.f64_gauge("rivoli.tok_per_s")
            .build()
            .record(summary.tok_per_s, &[]);
        m.f64_gauge("rivoli.hit_pct")
            .build()
            .record(summary.hit_pct, &[]);
        m.f64_gauge("rivoli.gb_per_tok")
            .build()
            .record(summary.gb_per_tok, &[]);
        m.f64_gauge("rivoli.miss_per_tok")
            .build()
            .record(summary.miss_per_tok, &[]);
        m.f64_gauge("rivoli.fetch_hidden_pct")
            .build()
            .record(summary.fetch_hidden_pct, &[]);
        m.u64_gauge("rivoli.tokens").build().record(tokens as u64, &[]);

        provider.shutdown().context("flush OTLP metrics")?;
        Ok(())
    }

}
