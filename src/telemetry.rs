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
    pub fetch_ms: f64,
    pub mlp_ms: f64,
    pub route_ms: f64,
    pub miss_per_tok: f64,
    pub ms_per_miss: f64,
    pub gb_per_tok: f64,
    pub nvme_read_ms: f64,
    pub bounce_copy_ms: f64,
}

impl ProfileSummary {
    /// The always-on stdout PROFILE line (per-token ms breakdown + disk traffic).
    pub fn report(&self) {
        tracing::info!(
            "PROFILE/tok: {:.0}ms wall | fetch {:.0}ms ({:.2} miss, {:.2}ms/miss, {:.2} GB) | mlp {:.0}ms | route {:.0}ms",
            self.wall_ms,
            self.fetch_ms,
            self.miss_per_tok,
            self.ms_per_miss,
            self.gb_per_tok,
            self.mlp_ms,
            self.route_ms,
        );
        tracing::info!(
            "  fetch split/tok: nvme-read {:.0}ms | bounce-copy {:.0}ms",
            self.nvme_read_ms,
            self.bounce_copy_ms,
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
        span.set_attribute(KeyValue::new("fetch_ms_per_tok", summary.fetch_ms));
        span.set_attribute(KeyValue::new("mlp_ms_per_tok", summary.mlp_ms));
        span.set_attribute(KeyValue::new("route_ms_per_tok", summary.route_ms));
        span.set_attribute(KeyValue::new("miss_per_tok", summary.miss_per_tok));
        span.set_attribute(KeyValue::new("gb_per_tok", summary.gb_per_tok));
        span.set_attribute(KeyValue::new("nvme_read_ms_per_tok", summary.nvme_read_ms));
        span.set_attribute(KeyValue::new(
            "bounce_copy_ms_per_tok",
            summary.bounce_copy_ms,
        ));
        span.end();

        // Flush the simple processor's export before we return (the run ends here).
        provider.shutdown().context("flush OTLP spans")?;
        Ok(())
    }
}
