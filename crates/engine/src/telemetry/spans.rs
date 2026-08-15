//! Interval recorder — turns the class counters into **real spans** with real
//! start/end times, so a trace viewer (Tempo/Jaeger/Grafana) renders them on a timeline
//! and the OVERLAP is visible rather than inferred.
//!
//! The counters in the parent module are scalars: `io-wait 183.7ms` tells you the
//! reaper waited that long, but not *when*, so nothing can show that it happened
//! underneath the decode thread's GPU waits. That is the whole question the engine's
//! design turns on, and a sum cannot answer it. These records can.
//!
//! **Cost is a `Vec` push behind a mutex — no exporter, no network, nothing per-span
//! during decode.** Intervals are replayed as spans with explicit timestamps only at
//! run end, which is why this can be on while the numbers it produces stay honest.
//!
//! Off unless `--spans <BUDGET>` is given (see [`init`]); `--spans` alone means 5000.
//!
//! The budget is spent by **sampling whole tokens spread across the run**, not by
//! recording the first N intervals and stopping. Taking the prefix meant the timeline
//! only ever showed the cold start — the least representative part of a decode, when the
//! cache is still filling — and silently claimed to be the run. The stride is
//! self-calibrating: token 0 is always recorded, its span count reveals the per-token
//! cost, and the stride for the rest falls out of `ngen x per_tok / budget`.
//!
//! WHOLE tokens, deliberately: sampling individual intervals would leave half-built
//! layers whose synthesised parent spans lie about their own children.
//!
//! **Moved out of `telemetry.rs` verbatim on 2026-08-15**, which had grown past
//! CodeScene's file-size cliff (~880 lines) and scored 8.81 on Low Cohesion alone: the
//! one file held four jobs that never call each other. The cut is by COHESION, not by
//! line count — this is one whole LCOM4 component, moved intact with its comments and
//! the measurements they carry. `telemetry.rs` re-exports what was public, so every
//! path that resolved before still resolves.
//!
//! One word is not verbatim: the doc above said "the counters elsewhere in this
//! file", which the move made false — they are `super`'s now.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};
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
        SAMPLED.store(tok.is_multiple_of(stride), Ordering::Relaxed);
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
    // Clear the truncation flag with the records it describes. Prefill is one forward
    // pass over 78 layers — ~390 intervals — so any budget below that trips the cap
    // before generation starts, and the flag outlived the data: a run whose sampled
    // timeline is COMPLETE still ended with "the exported timeline is missing the LATER
    // sampled tokens" and a `spans_truncated` attribute on the root. Measured
    // 2026-08-01 at `--spans 200`. A truncation warning that fires on untruncated
    // output is worse than none — it teaches the reader to ignore the real one.
    l.truncated.store(false, Ordering::Relaxed);
    if let Some(Ok(mut v)) = LAYERS.get().map(|m| m.lock()) {
        v.clear();
    }
    tracing::info!(
        "--spans: budget {} / (~{per_tok} spans/tok x {ngen} tok) -> every {stride}th \
         token sampled ({} of {ngen}), prefill discarded",
        l.cap,
        ngen.div_ceil(stride as usize),
    );
}

/// Arm the recorder with a span budget, and anchor its clocks. `main` calls this once
/// for `--spans`; without that call every entry point in this module is a no-op, so a
/// run that does not ask for a timeline pays nothing for the possibility of one.
///
/// **Called from `main`, not read from the environment.** This was `RIVOLI_SPANS` until
/// 2026-08-01, which broke the project's own rule — an instrument goes behind a feature
/// AND a flag, never an env var, because an env var is invisible to `--help`, absent
/// from the command line a benchmark records, and silently active in a build that looks
/// stock. The two `OTEL_*` variables that remain are the OpenTelemetry standard's, not
/// ours to rename; this one was ours.
///
/// Idempotent by construction: a second call cannot re-anchor the clocks a first one
/// established, because the intervals already recorded against them would shift.
pub fn init(cap: usize) {
    let cap = cap.max(1);
    let _ = LOG.set(Some(Log {
        t0_mono: Instant::now(),
        t0_wall: SystemTime::now(),
        cap,
        recs: Mutex::new(Vec::with_capacity(cap.min(1 << 16))),
        truncated: AtomicBool::new(false),
    }));
}

fn log() -> Option<&'static Log> {
    LOG.get()?.as_ref()
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
pub struct ExpertComposition {
    pub tok: u32,
    pub layer: i32,
    pub cold_i4: u16,
    pub warm_i4: u16,
    pub cold_vq3: u16,
    pub warm_vq3: u16,
}

static LAYERS: OnceLock<Mutex<Vec<ExpertComposition>>> = OnceLock::new();

/// Record a layer's expert composition. Same enable gate as `record`.
pub fn record_layer(st: ExpertComposition) {
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
pub fn drain_layers() -> Vec<ExpertComposition> {
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
