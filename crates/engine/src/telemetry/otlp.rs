//! The OTLP half: one `rivoli.decode` span tree plus the gauges, exported
//! synchronously at run end. Built only under `--features otlp`, and reached only
//! through [`super::export_decode`], which additionally requires
//! `OTEL_EXPORTER_OTLP_ENDPOINT` — so an unconfigured run pays nothing.
//!
//! **Moved out of `telemetry.rs` verbatim on 2026-08-15**, which had grown past
//! CodeScene's file-size cliff (~880 lines) and scored 8.81 on Low Cohesion alone: the
//! one file held four jobs that never call each other. The cut is by COHESION, not by
//! line count — this is one whole LCOM4 component, moved intact with its comments and
//! the measurements they carry. `telemetry.rs` re-exports what was public, so every
//! path that resolved before still resolves.

use super::{ProfileSummary, RunInfo};
use anyhow::{Context, Result};
use opentelemetry::KeyValue;
use opentelemetry::trace::{Span, TraceContextExt, Tracer, TracerProvider as _};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::trace::SdkTracerProvider;
use std::collections::BTreeMap;
use std::time::SystemTime;

/// The `Resource` BOTH signals carry, built in one place because they must agree.
///
/// A trace and a metric that disagree on `service.name` land under two different
/// services in the backend and nothing correlates them — the span waterfall and the
/// gauges from the same run stop being the same run. `OTEL_SERVICE_NAME` overrides
/// the default so a side-by-side arm can be tagged; `service.version` is the build's
/// `CARGO_PKG_VERSION`, which is what tells one binary's numbers from another's.
fn resource() -> Resource {
    let service_name = std::env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| "rivoli".to_string());
    Resource::builder()
        .with_service_name(service_name)
        .with_attribute(KeyValue::new("service.version", env!("CARGO_PKG_VERSION")))
        .build()
}

/// Build a one-shot tracer (blocking HTTP exporter + a SimpleSpanProcessor, so the
/// span exports synchronously on `end()`/`shutdown()` — no async runtime of ours),
/// emit the `rivoli.decode` span with the summary as attributes, and flush. The
/// endpoint + protocol come from the standard `OTEL_*` env vars; the identity of the
/// run is [`resource`], shared with the metric export.
pub fn export(summary: &ProfileSummary, tokens: usize, run: &RunInfo) -> Result<()> {
    let provider = span_provider()?;
    let tracer = provider.tracer("rivoli");
    // Drain BEFORE building the root: the root needs explicit start/end covering the
    // children, and `tracer.start()` would stamp it at export time — minutes after
    // the intervals it parents. A parent whose window does not contain its children
    // is what makes a waterfall render as one collapsed nest.
    let (recs, truncated) = super::spans::drain();
    let n = recs.len();
    let bounds = span_bounds(&recs);
    let mut span = start_root(&tracer, bounds);
    set_run_attributes(&mut span, run);
    set_outcome_attributes(&mut span, tokens, run);
    let cx = opentelemetry::Context::current_with_span(span);
    emit_token_spans(&tracer, &cx, recs);
    close_root(&cx, n, truncated, bounds);
    // Flush the simple processor's export before we return (the run ends here).
    provider.shutdown().context("flush OTLP spans")?;
    export_metrics(summary, tokens, run)
}

/// The span half of the pipeline: a blocking HTTP exporter behind a
/// `SimpleSpanProcessor`, so spans export synchronously on `end()`/`shutdown()` — no
/// async runtime of ours. Endpoint + protocol come from the standard `OTEL_*` env
/// vars; the identity of the run is [`resource`], shared with the metric export.
fn span_provider() -> Result<SdkTracerProvider> {
    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_http()
        .build()
        .context("build OTLP span exporter")?;
    Ok(SdkTracerProvider::builder()
        .with_resource(resource())
        .with_simple_exporter(exporter)
        .build())
}

/// The `rivoli.decode` root, stamped to span the drained intervals so it contains its
/// own children.
fn start_root<T: Tracer>(tracer: &T, bounds: Option<(SystemTime, SystemTime)>) -> T::Span {
    match bounds {
        Some((lo, hi)) => tracer
            .span_builder("rivoli.decode")
            .with_start_time(lo)
            .with_end_time(hi)
            .start(tracer),
        // No intervals recorded (--spans not given): a plain now-stamped span, which
        // is correct — there are no children for it to fail to contain.
        None => tracer.start("rivoli.decode"),
    }
}

/// WHAT THIS RUN WAS, not what it measured. The numbers went out as metrics —
/// repeating them here would make two sources of truth that drift, and they are
/// not what you search a trace by. These are.
fn set_run_attributes<S: Span>(span: &mut S, run: &RunInfo) {
    span.set_attribute(KeyValue::new("rivoli.model", run.model.clone()));
    span.set_attribute(KeyValue::new("rivoli.mode", run.mode.clone()));
    span.set_attribute(KeyValue::new(
        "rivoli.cache_policy",
        run.cache_policy.clone(),
    ));
    span.set_attribute(KeyValue::new("rivoli.attn", run.attn.clone()));
    // `moe_gain`/`sinks`/`window`/`misa_heads` left with their `RunInfo` fields
    // (2026-08-16): no flag in this tree fills them, and an attribute that is always its
    // default is a knob the recorded run never had. Each returns with its flag.
    // Absent rather than zero where zero would read as a measurement: an auto-sized
    // budget is not "0 GiB", and a run with no `-bench` did not generate 0 tokens.
    if let Some(g) = run.max_mem_gib {
        span.set_attribute(KeyValue::new("rivoli.max_mem_gib", g as i64));
    }
    if let Some(n) = run.bench_tokens {
        span.set_attribute(KeyValue::new("rivoli.bench_tokens", n as i64));
    }
    if let Some(p) = &run.prompt {
        span.set_attribute(KeyValue::new("rivoli.prompt", p.clone()));
    }
}

/// What the run PRODUCED, as run shape rather than as speed: how many tokens, and
/// whether it degenerated.
fn set_outcome_attributes<S: Span>(span: &mut S, tokens: usize, run: &RunInfo) {
    span.set_attribute(KeyValue::new("rivoli.tokens_generated", tokens as i64));
    // Degeneration is a first-class outcome, not a footnote: a looped run's tok/s is
    // an artifact (few experts, inflated hit rate) and must never be ranked as if it
    // were a result. Present as an attribute so a query can exclude it.
    span.set_attribute(KeyValue::new("rivoli.degenerate", run.degenerate.is_some()));
    if let Some(d) = run.degenerate {
        span.set_attribute(KeyValue::new("rivoli.loop_period", d.period as i64));
        span.set_attribute(KeyValue::new("rivoli.loop_repeats", d.repeats as i64));
        span.set_attribute(KeyValue::new("rivoli.loop_start", d.start as i64));
    }
}

/// Stamp the root with what the timeline actually contains, then close it on the
/// children's own end rather than on `now()`.
fn close_root(
    cx: &opentelemetry::Context,
    recorded: usize,
    truncated: bool,
    bounds: Option<(SystemTime, SystemTime)>,
) {
    let span = cx.span();
    span.set_attribute(KeyValue::new("spans_recorded", recorded as i64));
    // A truncated timeline that does not say so reads as a complete one.
    if truncated {
        span.set_attribute(KeyValue::new("spans_truncated", true));
        tracing::warn!(
            "--spans cap reached after {recorded} intervals — the sampling stride \
             under-estimated spans/token, so the exported timeline is missing the \
             LATER sampled tokens. Raise --spans or the per_tok estimate."
        );
    }
    match bounds {
        Some((_, hi)) => span.end_with_timestamp(hi),
        None => span.end(),
    }
}

/// Rebuild the tree: decode -> token N -> layer L -> the leaf intervals. Emitting
/// the leaves as 1200 direct siblings of the root is technically a trace and
/// practically unreadable; the token/layer levels are what make a waterfall
/// navigable, and they are synthesised from the leaves' own bounds so they cannot
/// disagree with them.
///
/// Every span here is closed with `end_with_timestamp`. `end()` would stamp
/// `now()` and silently discard the builder's `with_end_time`.
fn emit_token_spans<T>(tracer: &T, cx: &opentelemetry::Context, recs: Vec<super::spans::Rec>)
where
    T: Tracer,
    T::Span: Send + Sync + 'static,
{
    let layer_states: BTreeMap<(u32, i32), super::spans::ExpertComposition> =
        super::spans::drain_layers()
            .into_iter()
            .map(|st| ((st.tok, st.layer), st))
            .collect();
    for (tok, tok_recs) in group_by(recs, |r| r.tok) {
        let Some((t_lo, t_hi)) = span_bounds(&tok_recs) else {
            continue;
        };
        // The token id, not just the index — the index says "the 7th step", the id
        // says which token the model was actually working on.
        let tok_id = tok_recs.first().map(|r| r.tok_id).unwrap_or(0);
        let tok_span = tracer
            .span_builder(format!("token {tok}"))
            .with_start_time(t_lo)
            .with_end_time(t_hi)
            .with_attributes([
                KeyValue::new("rivoli.token_index", i64::from(tok)),
                KeyValue::new("rivoli.token_id", i64::from(tok_id)),
            ])
            .start_with_context(tracer, cx);
        let tok_cx = opentelemetry::Context::current_with_span(tok_span);
        emit_layer_spans(
            tracer,
            &tok_cx,
            tok_recs,
            &TokenLayers {
                tok,
                states: &layer_states,
            },
        );
        tok_cx.span().end_with_timestamp(t_hi);
    }
}

/// The expert-composition table, narrowed to the token whose layers are being emitted.
/// One argument instead of two so the emitter stays inside the argument budget, and so
/// the pair cannot be passed out of step with each other.
struct TokenLayers<'a> {
    tok: u32,
    states: &'a BTreeMap<(u32, i32), super::spans::ExpertComposition>,
}

/// The layer level of the tree, plus its leaves.
fn emit_layer_spans<T>(
    tracer: &T,
    tok_cx: &opentelemetry::Context,
    tok_recs: Vec<super::spans::Rec>,
    layers: &TokenLayers<'_>,
) where
    T: Tracer,
    T::Span: Send + Sync + 'static,
{
    for (layer_i, layer_recs) in group_by(tok_recs, |r| r.layer) {
        // layer -1 is work outside the layer loop (the tail): hang it straight off
        // the token rather than inventing a "layer -1" level for it.
        let parent_cx = if layer_i < 0 {
            tok_cx.clone()
        } else {
            let Some((l_lo, l_hi)) = span_bounds(&layer_recs) else {
                continue;
            };
            let ls = tracer
                .span_builder(format!("layer {layer_i}"))
                .with_start_time(l_lo)
                .with_end_time(l_hi)
                .with_attributes(layer_attributes(
                    layer_i,
                    layers.states.get(&(layers.tok, layer_i)),
                ))
                .start_with_context(tracer, tok_cx);
            let c = opentelemetry::Context::current_with_span(ls);
            c.span().end_with_timestamp(l_hi);
            c
        };
        emit_leaf_spans(tracer, &parent_cx, layer_recs);
    }
}

/// Expert composition: residency x format. This is what explains a layer's cost —
/// eight cold int4 experts and eight warm vq3 ones are different animals — and the
/// counts are recorded at submit time because that is the only point where both are
/// known.
fn layer_attributes(layer_i: i32, st: Option<&super::spans::ExpertComposition>) -> Vec<KeyValue> {
    let mut attrs = vec![KeyValue::new("rivoli.layer", i64::from(layer_i))];
    if let Some(st) = st {
        attrs.push(KeyValue::new("experts.cold.int4", i64::from(st.cold_i4)));
        attrs.push(KeyValue::new("experts.warm.int4", i64::from(st.warm_i4)));
        attrs.push(KeyValue::new(
            "experts.cold.int3_vq",
            i64::from(st.cold_vq3),
        ));
        attrs.push(KeyValue::new(
            "experts.warm.int3_vq",
            i64::from(st.warm_vq3),
        ));
        attrs.push(KeyValue::new(
            "experts.cold",
            i64::from(st.cold_i4 + st.cold_vq3),
        ));
        attrs.push(KeyValue::new(
            "experts.total",
            i64::from(st.cold_i4 + st.warm_i4 + st.cold_vq3 + st.warm_vq3),
        ));
    }
    attrs
}

/// The measured intervals themselves, hung off whichever level is their parent.
fn emit_leaf_spans<T: Tracer>(
    tracer: &T,
    parent_cx: &opentelemetry::Context,
    recs: Vec<super::spans::Rec>,
) {
    for r in recs {
        let mut leaf = tracer
            .span_builder(r.name)
            .with_start_time(r.start)
            .with_end_time(r.end)
            .with_attributes([KeyValue::new("thread", r.thread)])
            .start_with_context(tracer, parent_cx);
        leaf.end_with_timestamp(r.end);
    }
}

/// Bucket records by one of their positional fields. `BTreeMap`, so the export walks
/// tokens and layers in index order — a waterfall sorted by hash is unreadable.
fn group_by<K: Ord>(
    recs: Vec<super::spans::Rec>,
    key: impl Fn(&super::spans::Rec) -> K,
) -> BTreeMap<K, Vec<super::spans::Rec>> {
    let mut out: BTreeMap<K, Vec<super::spans::Rec>> = BTreeMap::new();
    for r in recs {
        out.entry(key(&r)).or_default().push(r);
    }
    out
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
/// can `sum by (class)` instead of hard-coding a query per metric — and **every**
/// datapoint, class gauge or scalar, additionally carries [`RunInfo::labels`], which
/// is what makes two runs distinguishable at all.
fn export_metrics(summary: &ProfileSummary, tokens: usize, run: &RunInfo) -> Result<()> {
    use opentelemetry::metrics::MeterProvider as _;
    use opentelemetry_sdk::metrics::{PeriodicReader, SdkMeterProvider};

    let exporter = opentelemetry_otlp::MetricExporter::builder()
        .with_http()
        .build()
        .context("build OTLP metric exporter")?;
    let provider = SdkMeterProvider::builder()
        .with_reader(PeriodicReader::builder(exporter).build())
        .with_resource(resource())
        .build();
    let m = provider.meter("rivoli");

    // WHICH RUN THIS IS — see `RunInfo::labels` for the set and for what bounds its
    // cardinality. Appended to every `record()` below; a datapoint without them is a
    // datapoint that averages into every other run's.
    //
    // **DATAPOINT attributes, not RESOURCE attributes**, and the distinction is the
    // whole point rather than a detail: `record(v, attrs)` puts these in the
    // gauge's `NumberDataPoint.attributes` (they become Prometheus labels on the
    // series), while the `Resource` built above lands in `ResourceMetrics.resource`
    // and Alloy's `otelcol.exporter.prometheus` parks that in `target_info` unless
    // `resource_to_telemetry_conversion` is enabled — a collector setting in a config
    // file outside this repo. Putting run identity there would draw an empty graph
    // instead of an error, which `measurement/traces.md` ("Things that will bite")
    // already records as the worst possible failure mode. So `service.name` and
    // `service.version` stay on the Resource, and everything you filter a chart by
    // goes through `record()`.
    let run_labels: Vec<KeyValue> = run
        .labels()
        .into_iter()
        .map(|(k, v)| KeyValue::new(k, v))
        .collect();

    record_class_gauges(&m, summary, &run_labels);
    record_scalar_gauges(&m, summary, tokens, &run_labels);
    record_outcome_gauges(&m, run, &run_labels);

    provider.shutdown().context("flush OTLP metrics")?;
    Ok(())
}

/// The phase axis, all on one `rivoli.ms_per_tok` gauge separated by a `phase`
/// attribute. Unlike the old tree's class spans these DISJOINTLY partition wall
/// (`other` is the measured remainder and the bucket-sum gate bounds it), so a stacked
/// panel is honest here — the first time in this engine's telemetry that has been true.
///
/// The old `class`/`thread` axis (gpu-wait / io-wait / cpu splits) went with the
/// `ProfileSummary` fields behind it: those described instruments this tree's arms do
/// not carry yet, and a gauge fed a structural zero charts as a measurement of nothing.
/// Each series returns with the instrument that fills it.
fn record_class_gauges(
    m: &opentelemetry::metrics::Meter,
    summary: &ProfileSummary,
    run_labels: &[KeyValue],
) {
    // ms/token, the unit every number in the PROFILE line is already in.
    let per_tok = m.f64_gauge("rivoli.ms_per_tok").build();
    let g = |v: f64, phase: &'static str| {
        let attrs: Vec<KeyValue> = run_labels
            .iter()
            .cloned()
            .chain([KeyValue::new("phase", phase)])
            .collect();
        per_tok.record(v, &attrs);
    };
    g(summary.attend_ms, "attend");
    g(summary.ffn_ms, "ffn");
    g(summary.fetch_wait_ms, "fetch-wait");
    g(summary.head_ms, "head");
    g(summary.other_ms, "other");
    g(summary.wall_ms, "wall");
}

/// The scalars carry run identity and nothing else — `tok_per_s` is THE ranking
/// number, so it is the one series that must never merge two configurations.
/// (`gb_per_tok`/`miss_per_tok` left with their fields, 2026-08-16 — nothing measures
/// bytes per token in this tree yet; they return with the counter that does.)
fn record_scalar_gauges(
    m: &opentelemetry::metrics::Meter,
    summary: &ProfileSummary,
    tokens: usize,
    run_labels: &[KeyValue],
) {
    m.f64_gauge("rivoli.tok_per_s")
        .build()
        .record(summary.tok_per_s, run_labels);
    m.f64_gauge("rivoli.hit_pct")
        .build()
        .record(summary.hit_pct, run_labels);
    m.u64_gauge("rivoli.tokens")
        .build()
        .record(tokens as u64, run_labels);
}

/// Degeneration, chartable: a dashboard can show "how many cells degenerated" over a
/// matrix run rather than requiring someone to read 44 logs — which needs the labels
/// to say WHICH cells.
fn record_outcome_gauges(
    m: &opentelemetry::metrics::Meter,
    run: &RunInfo,
    run_labels: &[KeyValue],
) {
    m.u64_gauge("rivoli.degenerate")
        .build()
        .record(u64::from(run.degenerate.is_some()), run_labels);
    if let Some(d) = run.degenerate {
        m.u64_gauge("rivoli.loop_period")
            .build()
            .record(d.period as u64, run_labels);
        m.u64_gauge("rivoli.loop_repeats")
            .build()
            .record(d.repeats as u64, run_labels);
    }
}
