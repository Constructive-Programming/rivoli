//! Decode-run telemetry: the always-on stdout PROFILE summary + an optional OTLP
//! span. Both are cheap — one emission at the end of a run, no per-token cost, no
//! GPU syncs (the underlying buckets ride the joins the forward pass already pays).
//! The expensive fine-grained audits + correctness probes live behind the `trace`
//! feature in the engine, not here.
//!
//! OTLP is opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT` (unset ⇒ log-only, no collector
//! needed) and exports a single `rivoli.decode` span synchronously at run end — no
//! async runtime.
//!
//! **Split into siblings on 2026-08-15** (CodeScene file-size cliff, ~880 lines; the
//! whole 8.81 was Low Cohesion). The cut is by cohesion, not by size: [`spans`] is the
//! live interval recorder, [`degeneracy`] reads the produced text after the fact, and
//! `otlp` is the exporter. What stayed is the end-of-run RECORD — what the run was
//! ([`RunInfo`]) and what it measured ([`ProfileSummary`]) — and the seam that hands
//! them to the exporter.

pub mod degeneracy;
pub mod spans;

#[cfg(feature = "otlp")]
mod otlp;

// Re-exported at the path they were defined at until the split, so no `use` outside
// this module had to move. `telemetry::degeneracy::detect_loop` also resolves.
pub use degeneracy::{
    LoopReport, RepetitionReport, detect_loop, has_repeated_block, is_degenerate, repetition_report,
};

/// The run's arguments — what belongs on the root span. A span's attributes should say
/// WHAT THIS RUN WAS so traces can be found and compared; the numbers it produced are
/// metrics, and duplicating them here just makes two sources of truth that can drift.
#[derive(Debug, Clone)]
pub struct RunInfo {
    pub model: String,
    pub mode: String,
    pub cache_policy: String,
    pub attn: String,
    /// `--max-mem <GiB>`, or None when the budget auto-sizes.
    pub max_mem_gib: Option<u64>,
    /// The `--mtp-min-conf` gate when speculative decode is ON, else None.
    ///
    /// Part of run identity because it changes THROUGHPUT, not just quality: ungated the
    /// verify pass is 0.93-0.95x (a loss), and at 0.8 it is 1.108x. Two runs that differ
    /// only here are not comparable, and until 2026-08-02 they shared a series.
    pub mtp_min_conf: Option<f32>,
    pub bench_tokens: Option<usize>,
    pub prompt: Option<String>,
    pub moe_gain: f32,
    pub sinks: usize,
    pub window: usize,
    pub misa_heads: usize,
    /// Set when the generation ended in a verbatim repetition loop. `None` is the
    /// measurement "it did not", which is why this is an Option and not a bool plus
    /// three zeroes.
    pub degenerate: Option<LoopReport>,
}

impl RunInfo {
    /// **Which run this is**, as metric labels: `(key, value)` in wire order.
    ///
    /// Every exported gauge carries these. Without them `--mode hybrid --max-mem 115` and
    /// `--mode int3-vq --max-mem 70` are the *same* Prometheus series and average together
    /// silently — the whole metrics half was uncomparable, and
    /// `measurement/traces.md` told the reader to "search by which mode / which policy /
    /// which budget, then read the numbers off a chart" against a chart with no such label
    /// (`investigations/otlp-modernization.md` §2a).
    ///
    /// **Cardinality, and what bounds it — read this before adding a label here.** One
    /// datapoint per RUN, and every value below is a command-line argument out of a small
    /// closed set: `mode` 3 (`int4|int3-vq|hybrid`), `cache_policy` 3 (`lru|2q|arc`),
    /// `attn` 4 shapes (`streaming`/`misa` carry their parameters in the `Debug` string, so
    /// they are bounded by what a flag can be given, not by anything the run computes),
    /// `max_mem_gib` ~10 budgets in practice. Their product is the benchmark matrix, which
    /// is the point. **Nothing that varies per token, per layer or per prompt may join
    /// them** — `model` and `prompt` are deliberately absent because they are unbounded,
    /// and both are already root-span attributes, where cardinality is free.
    ///
    /// This lives outside `mod otlp` on purpose: it is the only place `RunInfo`'s fields
    /// are named for the exporter, and named here the default `--features rocm` build
    /// compiles it. A field rename that reaches only the feature-gated exporter is exactly
    /// how `launch_ms` sat broken (`measurement/traces.md`, "How it rotted").
    pub fn labels(&self) -> Vec<(&'static str, String)> {
        [
            ("mode", self.mode.clone()),
            ("cache_policy", self.cache_policy.clone()),
            ("attn", self.attn.clone()),
        ]
        .into_iter()
        // Absent rather than zero, the same rule the root span keeps: an auto-sized budget
        // is not "0 GiB", and a series labelled `max_mem_gib="0"` would read as a run that
        // was given a budget of nothing.
        .chain(self.max_mem_gib.map(|g| ("max_mem_gib", g.to_string())))
        // ALWAYS present, unlike the budget: "speculative decode was off" is a state the
        // run was in, not a measurement that is missing, and it is the difference between
        // a 1.108x arm and a 0.93x one. Two decimals because the gate is swept in tenths.
        .chain(std::iter::once((
            "mtp",
            self.mtp_min_conf
                .map_or_else(|| "off".to_string(), |c| format!("{c:.2}")),
        )))
        .collect()
    }
}

/// End-of-run per-token performance summary — the PROFILE line and the OTLP span
/// fields. Built by the GPU engine from its always-on [`Profile`](crate::gpu) buckets.
#[derive(Debug, Clone, Copy)]
pub struct ProfileSummary {
    /// Mean MoE bracket (µs) and layer count, indexed by that layer's MISS count.
    ///
    /// `compute_gpu` is a bracket, so the aggregate cannot say whether it is large because
    /// the shaders are slow or because the compute stream idles waiting for bytes. This
    /// can: read the SHAPE. A flat profile across miss counts means the gaps are not fetch
    /// waits; a rising one means they are, and the slope is the per-miss cost.
    pub moe_us_by_miss: [Option<(f64, u32)>; 16],
    pub tok_per_s: f64,
    pub hit_pct: f64,
    pub wall_ms: f64,
    pub route_ms: f64,
    /// The DSA indexer's GPU-timeline span; see [`Profile::idx_gpu_ns`](crate::gpu).
    pub idx_gpu_ms: f64,
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
    pub miss_per_tok: f64,
    pub ms_per_miss: f64,
    pub gb_per_tok: f64,

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
    /// Host compute: the sum of `cpu_launch_ms + cpu_route_ms + cpu_submit_ms`. Every term
    /// stamped; none derived. (It also carried the expert stream's tokio poll time until
    /// 2026-08-01; enqueueing the launches straight onto the compute stream left no such
    /// work to attribute — see `gpu.rs`'s `cpu_ns`.)
    pub cpu_ms: f64,
    /// Host time issuing kernel launches (per-layer attention/MLP block + the tail).
    pub cpu_launch_ms: f64,
    /// Host time in `route_into` — sigmoid/bias/top-k over 256 experts per MoE layer.
    pub cpu_route_ms: f64,
    /// Host time in `RoutedPool::submit` — residency, policy bookkeeping, read specs.
    pub cpu_submit_ms: f64,
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
    /// The always-on stdout PROFILE line. Under the async overlap, `moe_wall` is the real
    /// per-token MoE cost and `fetch_wall` the reaper's work behind it.
    ///
    /// The `% hidden` and `ms exposed` terms were removed 2026-08-01 with the two fields
    /// behind them. Both derived from `moe_wall − compute_gpu`, and `compute_gpu` is a
    /// BRACKET that CONTAINS the compute stream's stall — so every millisecond spent
    /// waiting on NVMe was counted as compute, and therefore as fetch successfully hidden.
    /// It reported 96% where the arithmetic caps the true figure at 57%: layers with 0
    /// misses run the MoE bracket in 1563 us, so 75 layers of pure kernel work is 117
    /// ms/token (`examples/moe_bench.rs` independently floors at 113), against a measured
    /// `moe_wall` of 266 — 117 compute + 149 stall, and fetch can only hide behind the 117.
    /// Read `io_wait_ms` in the class line below, which is measured at the io_uring ring,
    /// and the `moe/layer by miss count` line, which measures the stall rather than
    /// inferring its absence.
    pub fn report(&self) {
        self.report_phases();
        self.report_moe_by_miss();
        self.report_classes();
        self.report_splits();
        self.report_indexer();
    }

    /// WHERE the time went, by phase. One line, always printed.
    fn report_phases(&self) {
        tracing::info!(
            // wall/route at 0.1 ms: the DSA selection A/B (docs/investigations/npu-offload.md) turns on deltas of
            // a few ms against a ~400 ms token, which 1 ms resolution rounds into noise.
            "PROFILE/tok: {:.1}ms wall | route {:.1}ms | moe {:.0}ms (gpu {:.0}ms) | fetch {:.0}ms | {:.2} miss, {:.2}ms/miss, {:.2} GB",
            self.wall_ms,
            self.route_ms,
            self.moe_wall_ms,
            self.compute_gpu_ms,
            self.fetch_wall_ms,
            self.miss_per_tok,
            self.ms_per_miss,
            self.gb_per_tok,
        );
    }

    /// The MoE bracket decomposed by miss count — printed only when there is more than
    /// one populated bucket, since a single bucket has no shape to read.
    fn report_moe_by_miss(&self) {
        let pop: Vec<(usize, f64, u32)> = self
            .moe_us_by_miss
            .iter()
            .enumerate()
            .filter_map(|(m, v)| v.map(|(us, n)| (m, us, n)))
            .collect();
        if pop.len() <= 1 {
            return;
        }
        let cells: Vec<String> = pop
            .iter()
            .map(|(m, us, n)| format!("{m}m:{us:.0}us(n={n})"))
            .collect();
        let lo = pop.first().map(|c| c.1).unwrap_or(0.0);
        let hi = pop.last().map(|c| c.1).unwrap_or(0.0);
        tracing::info!(
            "  moe/layer by miss count: {} | span {:.0}->{:.0}us ({:+.0}%)",
            cells.join(" "),
            lo,
            hi,
            if lo > 0.0 {
                100.0 * (hi - lo) / lo
            } else {
                0.0
            },
        );
    }

    /// The CLASS view: [`Self::report_phases`] says WHERE the time is (phases); this says
    /// WHAT it is. Every term is measured — none is a residual — so they OVERLAP and
    /// need not sum to wall. `io-wait` is on the reaper thread and routinely exceeds
    /// it. The `%` is therefore "of wall", not "share of wall", and unattributed time
    /// is deliberately not shown.
    fn report_classes(&self) {
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
            "    cpu = launch {:.1}ms + route {:.2}ms + submit {:.2}ms",
            self.cpu_launch_ms,
            self.cpu_route_ms,
            self.cpu_submit_ms,
        );
    }

    /// The two phase/class splits that motivated the class view: `route` was a region
    /// mixing a blocking D2H with host routing, and the whole `tail` phase was one
    /// opaque wait with ~59% attributable to no kernel.
    fn report_splits(&self) {
        tracing::info!(
            "  split/tok: route = {:.1}ms gpu-wait + {:.1}ms host-routing | tail wait {:.1}ms, of which {:.1}ms is GPU ({:.0}% overhead)",
            self.route_wait_ms,
            (self.route_ms - self.route_wait_ms).max(0.0),
            self.tail_wait_ms,
            self.tail_gpu_ms,
            100.0 * (self.tail_wait_ms - self.tail_gpu_ms).max(0.0) / self.tail_wait_ms.max(1e-9),
        );
    }

    /// DSA indexer decomposition (docs/investigations/npu-offload.md M0). Silent when the indexer never
    /// scored — dense/streaming, or a context that stayed under `index_topk`, where a
    /// row of zeros would read as a measurement of something that did not happen.
    fn report_indexer(&self) {
        if self.idx_layers_per_tok <= 0.0 {
            return;
        }
        tracing::info!(
            "  indexer/tok: gpu {:.1}ms => {:.1}us per layer (selection on device) over \
             {:.3} scoring layers",
            self.idx_gpu_ms,
            // Guarded non-zero by the `> 0.0` above, so this division is safe.
            self.idx_gpu_ms * 1e3 / self.idx_layers_per_tok,
            self.idx_layers_per_tok,
        );
    }
}

/// Export the decode summary to OTLP as one `rivoli.decode` span, when the `otlp`
/// feature is built AND `OTEL_EXPORTER_OTLP_ENDPOINT` is set. No-op otherwise (the
/// PROFILE line already logged the same numbers). Never fails the run — an export
/// error is warned and swallowed.
pub fn export_decode(summary: &ProfileSummary, tokens: usize, run: &RunInfo) {
    #[cfg(feature = "otlp")]
    if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some()
        && let Err(e) = otlp::export(summary, tokens, run)
    {
        tracing::warn!("OTLP export failed ({e}); metrics logged only");
    }
    #[cfg(not(feature = "otlp"))]
    let _ = (summary, tokens, run);
}

/// The metric label set — the run-identity half of the OTLP export, tested **without** the
/// `otlp` feature on purpose.
///
/// `cargo test` is the whole gate here (there is no CI, and nothing anyone routinely runs
/// compiles `--features otlp`), so an assertion that only holds inside `mod otlp` holds
/// nowhere. These tests reach `RunInfo::labels` instead, which is the one place the fields
/// are named for the exporter, and they run under the default build.
#[cfg(test)]
mod run_label_tests {
    use super::{LoopReport, RunInfo};

    /// Every field from a literal, deliberately. Adding a field to `RunInfo` breaks this
    /// function, which forces whoever adds it to decide whether it is run *identity* (a
    /// label) or a *measurement* (a gauge) — that mechanism is the point of the test, and
    /// the assertions below are secondary to it.
    fn run(max_mem_gib: Option<u64>) -> RunInfo {
        RunInfo {
            model: "/models/glm-5.2-int4".to_string(),
            mode: "hybrid".to_string(),
            cache_policy: "2q".to_string(),
            attn: "dsa".to_string(),
            max_mem_gib,
            mtp_min_conf: Some(0.8),
            bench_tokens: Some(256),
            prompt: Some("explain virtual memory".to_string()),
            moe_gain: 1.0,
            sinks: 4,
            window: 2048,
            misa_heads: 0,
            degenerate: None,
        }
    }

    #[test]
    fn two_runs_that_differ_in_config_are_two_series() {
        // These four key strings are what a dashboard template variable queries literally
        // (`label_values(rivoli_tok_per_s, mode)` and friends). A rename here is an empty
        // picker there, which reads as "no runs" rather than as an error — so a rename is
        // a two-file edit, and this assertion is the reminder.
        assert_eq!(
            run(Some(115)).labels(),
            vec![
                ("mode", "hybrid".to_string()),
                ("cache_policy", "2q".to_string()),
                ("attn", "dsa".to_string()),
                ("max_mem_gib", "115".to_string()),
                ("mtp", "0.80".to_string()),
            ],
        );

        // Speculative decode is on by DEFAULT and its gate decides whether the verify pass
        // is a 1.108x win or a 0.93x loss, so an ungated run and a gated one must not share
        // a series. Unlike the budget, "off" is emitted rather than omitted: it is a state
        // the run was in, and a missing key would make the two indistinguishable again.
        let ungated = RunInfo {
            mtp_min_conf: None,
            ..run(Some(115))
        };
        assert_ne!(ungated.labels(), run(Some(115)).labels());
        assert!(ungated.labels().contains(&("mtp", "off".to_string())));

        // The defect this whole change exists for: before it, these two landed in the same
        // Prometheus series and averaged together in silence.
        assert_ne!(run(Some(115)).labels(), run(Some(70)).labels());
        let other = RunInfo {
            mode: "int3-vq".to_string(),
            ..run(Some(115))
        };
        assert_ne!(run(Some(115)).labels(), other.labels());
    }

    #[test]
    fn an_auto_sized_budget_is_absent_rather_than_zero() {
        let keys: Vec<&str> = run(None).labels().into_iter().map(|(k, _)| k).collect();
        assert_eq!(keys, ["mode", "cache_policy", "attn", "mtp"]);
    }

    #[test]
    fn the_label_set_stays_bounded() {
        // Cardinality is the reason this is safe (see `RunInfo::labels`): every label is a
        // flag value from a closed set, one datapoint per run. `prompt` and `model` are
        // unbounded and belong on the root span, where cardinality is free; a per-token or
        // per-layer value must never appear here at all. If this fails, the question to
        // answer is not "raise the bound" but "how many series does that multiply into".
        let labels = run(Some(115)).labels();
        assert!(labels.len() <= 6, "label set grew to {labels:?}");
        for (k, _) in &labels {
            assert!(
                !["prompt", "model", "token", "layer", "expert"].contains(k),
                "{k} is unbounded or per-token; it does not belong on a metric label",
            );
        }
        // A degenerate run must be labelled the same way as a healthy one — the loop's
        // period/repeats are the gauge VALUES, not part of the series identity, or every
        // failing run would get a series of its own.
        let mut degen = run(Some(115));
        degen.degenerate = Some(LoopReport {
            period: 3,
            repeats: 40,
            start: 12,
        });
        assert_eq!(degen.labels(), run(Some(115)).labels());
    }
}
