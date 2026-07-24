//! Decode-run telemetry: the always-on stdout PROFILE summary + an optional OTLP
//! span. Both are cheap — one emission at the end of a run, no per-token cost, no
//! GPU syncs (the underlying buckets ride the joins the forward pass already pays).
//! The expensive fine-grained audits + correctness probes live behind the `trace`
//! feature in the engine, not here.
//!
//! OTLP is opt-in via `OTEL_EXPORTER_OTLP_ENDPOINT` (unset ⇒ log-only, no collector
//! needed) and exports a single `rivoli.decode` span synchronously at run end — no
//! async runtime.

/// End-of-run per-token performance summary — the PROFILE line and the OTLP span
/// fields. Built by the GPU engine from its always-on [`Profile`](crate::gpu) buckets.
#[derive(Debug, Clone, Copy)]
pub struct ProfileSummary {
    pub tok_per_s: f64,
    pub hit_pct: f64,
    pub wall_ms: f64,
    pub route_ms: f64,
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
}

impl ProfileSummary {
    /// The always-on stdout PROFILE line. Under the async overlap, `moe_wall` is the
    /// real per-token MoE cost; `fetch_wall` is the fetch work, of which
    /// `fetch_hidden_pct` overlapped compute and only `load_wait` was exposed.
    pub fn report(&self) {
        let exposed = (self.moe_wall_ms - self.compute_gpu_ms).max(0.0);
        tracing::info!(
            "PROFILE/tok: {:.0}ms wall | route {:.0}ms | moe {:.0}ms (gpu {:.0}ms) | fetch {:.0}ms ({:.0}% hidden, {:.0}ms exposed) | {:.2} miss, {:.2}ms/miss, {:.2} GB",
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
    }
}

/// Export the decode summary to OTLP as one `rivoli.decode` span, when the `otlp`
/// feature is built AND `OTEL_EXPORTER_OTLP_ENDPOINT` is set. No-op otherwise (the
/// PROFILE line already logged the same numbers). Never fails the run — an export
/// error is warned and swallowed.
pub fn export_decode(summary: &ProfileSummary, tokens: usize) {
    #[cfg(feature = "otlp")]
    if std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some() {
        if let Err(e) = otlp::export(summary, tokens) {
            tracing::warn!("OTLP export failed ({e}); metrics logged only");
        }
    }
    #[cfg(not(feature = "otlp"))]
    let _ = (summary, tokens);
}

#[cfg(feature = "otlp")]
mod otlp {
    use super::ProfileSummary;
    use anyhow::{Context, Result};
    use opentelemetry::KeyValue;
    use opentelemetry::trace::{Span, Tracer, TracerProvider as _};
    use opentelemetry_sdk::Resource;
    use opentelemetry_sdk::trace::SdkTracerProvider;

    /// Build a one-shot tracer (blocking HTTP exporter + a SimpleSpanProcessor, so the
    /// span exports synchronously on `end()`/`shutdown()` — no async runtime of ours),
    /// emit the `rivoli.decode` span with the summary as attributes, and flush. The
    /// endpoint + protocol come from the standard `OTEL_*` env vars. Version is the
    /// build's `CARGO_PKG_VERSION`.
    pub fn export(summary: &ProfileSummary, tokens: usize) -> Result<()> {
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
        let mut span = tracer.start("rivoli.decode");
        span.set_attribute(KeyValue::new("tokens", tokens as i64));
        span.set_attribute(KeyValue::new("tok_per_s", summary.tok_per_s));
        span.set_attribute(KeyValue::new("hit_pct", summary.hit_pct));
        span.set_attribute(KeyValue::new("wall_ms_per_tok", summary.wall_ms));
        span.set_attribute(KeyValue::new("route_ms_per_tok", summary.route_ms));
        span.set_attribute(KeyValue::new("moe_wall_ms_per_tok", summary.moe_wall_ms));
        span.set_attribute(KeyValue::new(
            "compute_gpu_ms_per_tok",
            summary.compute_gpu_ms,
        ));
        span.set_attribute(KeyValue::new(
            "fetch_wall_ms_per_tok",
            summary.fetch_wall_ms,
        ));
        span.set_attribute(KeyValue::new("load_wait_ms_per_tok", summary.load_wait_ms));
        span.set_attribute(KeyValue::new("launch_ms_per_tok", summary.launch_ms));
        span.set_attribute(KeyValue::new("fetch_hidden_pct", summary.fetch_hidden_pct));
        span.set_attribute(KeyValue::new("miss_per_tok", summary.miss_per_tok));
        span.set_attribute(KeyValue::new("gb_per_tok", summary.gb_per_tok));
        span.end();

        // Flush the simple processor's export before we return (the run ends here).
        provider.shutdown().context("flush OTLP spans")?;
        Ok(())
    }
}
