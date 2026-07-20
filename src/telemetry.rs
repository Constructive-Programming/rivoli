//! OpenTelemetry → OTLP span export, for Tempo (via the cluster's Alloy collector
//! or Tempo's OTLP/HTTP endpoint directly). Mirrors the ollama-router telemetry
//! pattern so rivoli traces land in the same fleet pipeline.
//!
//! Export is **opt-in**: with `OTEL_EXPORTER_OTLP_ENDPOINT` unset, rivoli runs
//! log-only (the `fmt` layer) and needs no collector — the common local/bench
//! case. When set (e.g. `http://tempo.monitor.svc:4318` or a node-local Alloy),
//! the decode span batch-exports over OTLP/HTTP. Build this INSIDE the tokio
//! runtime (the batch processor spawns onto it); `main` does so from `run`.

use anyhow::Result;
use opentelemetry::KeyValue;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::TracerProvider;
use tracing_subscriber::Layer;
use tracing_subscriber::registry::LookupSpan;

/// True when OTLP export is configured via the standard env var.
fn enabled() -> bool {
    std::env::var_os("OTEL_EXPORTER_OTLP_ENDPOINT").is_some()
}

/// The OTLP tracing layer plus the provider handle to flush on shutdown.
type OtlpLayer<S> = (Box<dyn Layer<S> + Send + Sync + 'static>, TracerProvider);

/// Build the OTLP tracing layer plus the provider to hold for shutdown flushing,
/// or `None` when export is not configured (rivoli then runs log-only). Endpoint
/// and knobs come from the standard `OTEL_*` env vars, read by the exporter;
/// `OTEL_SERVICE_NAME` overrides the service name (default `rivoli`).
///
/// Must be called from within a tokio runtime context — the batch span processor
/// spawns its export task onto it.
pub fn otlp_layer<S>(version: &str) -> Result<Option<OtlpLayer<S>>>
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a> + Send + Sync,
{
    if !enabled() {
        return Ok(None);
    }

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()?;

    let service_name = std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "rivoli".to_string());

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
        .with_resource(Resource::new(vec![
            KeyValue::new("service.name", service_name),
            KeyValue::new("service.version", version.to_string()),
        ]))
        .build();

    let tracer = provider.tracer("rivoli");
    let layer = tracing_opentelemetry::layer().with_tracer(tracer);
    Ok(Some((Box::new(layer), provider)))
}
