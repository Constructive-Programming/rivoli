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
    /// Set when the generation ended in a verbatim repetition loop. `None` is the
    /// measurement "it did not", which is why this is an Option and not a bool plus
    /// three zeroes.
    pub degenerate: Option<LoopReport>,
}

// `moe_gain`, `sinks`, `window` and `misa_heads` left this struct on 2026-08-16: no flag
// in this tree fills them (`--moe-gain`, `--sinks`/`--window`, `--misa-heads` are all
// old-tree knobs that have not been ported), so a value here would be a constant nothing
// spent — exactly the "recorded command line carrying a knob that did nothing" lie
// `rivoli_core::legality` exists to stop. Each returns with the flag that gives it a
// value (M13 streaming knobs, M14 `--misa-heads`).

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

/// Which named phase a decode-thread span belongs to. The vocabulary is fixed across all
/// four arms so their profiles are comparable; what each name COVERS on a given arm is a
/// property of that arm's sync points and is written on [`ProfileSummary`]'s fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Attend,
    /// MoE on the routed arms, the dense MLP on Glimmer — one name because it is one
    /// slot in the comparison, not a claim that the work is the same.
    Ffn,
    FetchWait,
    Head,
}

/// Decode-thread phase accumulators, in nanoseconds — the raw material of
/// [`ProfileSummary`]. One per engine, stamped by that arm's own loop.
///
/// **Each bucket is an independent sum of directly-stamped spans; nothing here is
/// derived.** That is what makes the bucket-sum gate able to go red: `other` (computed
/// later as `wall − named`) is NOT a field, so a dropped or forgotten stamp cannot hide
/// in it — it surfaces as `other` growing past the gate's bound. The mirror design (a
/// clock that attributes every nanosecond to the current phase) was rejected because it
/// sums to wall BY CONSTRUCTION, which makes the gate vacuous: an arm that never
/// switched phases would still pass.
///
/// Cost: two `Instant::now()` per stamp, ~600 stamp pairs per GLM token (78 layers × ~4
/// sections), ≈30 µs against a ~390 ms token (<0.01%) — cheap enough to be always on.
///
/// `pub` rather than `pub(crate)` even though only this crate's arm modules stamp it: in
/// the deviceless build NO arm module compiles, so `pub(crate)` makes this and `Phase`
/// unconstructed dead code, and `warnings = deny` turns that into a build error. One
/// spelling that works in both arms beats a tighter one plus a cfg'd `allow` at four
/// sites (tried, 2026-08-16).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Phases {
    pub attend_ns: u64,
    pub ffn_ns: u64,
    pub fetch_wait_ns: u64,
    pub head_ns: u64,
}

impl Phases {
    /// Close a span opened at `since` into bucket `p`; returns the closing instant so
    /// consecutive sections can chain (`t = lap(A, t)`) without a gap between them.
    pub fn lap(&mut self, p: Phase, since: std::time::Instant) -> std::time::Instant {
        let now = std::time::Instant::now();
        // u128→u64: a span would need 584 years to overflow.
        let ns = now.duration_since(since).as_nanos() as u64;
        match p {
            Phase::Attend => self.attend_ns += ns,
            Phase::Ffn => self.ffn_ns += ns,
            Phase::FetchWait => self.fetch_wait_ns += ns,
            Phase::Head => self.head_ns += ns,
        }
        now
    }

    /// The counters since `start` — how a decode loop rebases past the prefill, exactly
    /// as every arm already rebases its hit/miss counters (stats describe steady-state
    /// decode, and the prefill is warm-up).
    #[must_use]
    pub fn since(&self, start: &Phases) -> Phases {
        Phases {
            attend_ns: self.attend_ns - start.attend_ns,
            ffn_ns: self.ffn_ns - start.ffn_ns,
            fetch_wait_ns: self.fetch_wait_ns - start.fetch_wait_ns,
            head_ns: self.head_ns - start.head_ns,
        }
    }

    /// Sum of the four named buckets.
    pub fn named_ns(&self) -> u64 {
        self.attend_ns + self.ffn_ns + self.fetch_wait_ns + self.head_ns
    }
}

/// End-of-run per-token phase summary — the always-on PROFILE line and the OTLP gauge
/// fields. Built by [`ProfileSummary::from_decode`] from the [`Phases`] each arm's loop
/// stamps (the one constructor, and until 2026-08-16 nothing outside `cfg(test)` built
/// this type at all — a profile nothing fills is this repo's named telemetry trap).
///
/// **The old-tree field set (gpu/io/cpu classes, route/tail splits, moe-by-miss,
/// indexer) did not survive the port, deliberately** — those fields described instruments
/// the old engine had and this tree's arms measure none of them yet, so each would read a
/// structural zero. `docs/measurement/how-to-measure.md` carries the dated correction and
/// the list; each split returns WITH the instrument that measures it.
///
/// **Bucket semantics are per-arm, set by where each arm's existing sync points sit** —
/// no device syncs were added to sharpen them (a sync would change the thing measured):
///
/// | | attend | ffn | fetch-wait | head |
/// |---|---|---|---|---|
/// | GLM | attn launches + the gate-logits D2H (the layer's one host join, which drains attention execution; it also contains the gate GEMV's own execution, a 6144×256 f32 GEMV priced small against MLA over a growing KV) | host routing + submit/stage + the compute-stream await + drain + the end-of-layer sync | the residual wait on the MISS stream after the compute stream's await returned — fetch cost NOT hidden by resident compute, the design's own number; 0.0 means fully hidden | tail launches + the argmax D2H that drains final-norm → lm_head → argmax |
/// | Glimmer | attn-half launches (µs — no per-layer join exists on this arm) | MLP-half launches (same) | the synchronous slot-fill memcpy of each streamed layer (967.942 MB apiece — real, and the number P6 turns on) | `sample`, whose `device_sync` drains the WHOLE layer stack's execution on this arm |
/// | V4 / K3 | attention-half launches only (µs) — the gate D2H that drains attention execution sits inside these arms' ffn half and its span lands THERE; splitting it out the way GLM does is deferred until a V4/K3 phase number is needed (V4 decodes at the old tree's speed, so nothing is being attributed there yet) | everything else in the layer incl. the gate D2H and the `device_sync`s (expert waits are device-side, so fetch exposure drains here undistinguished) | 0.0 — no host-visible fetch wait exists to stamp | head launches + the argmax join |
#[derive(Debug, Clone, Copy)]
pub struct ProfileSummary {
    pub tok_per_s: f64,
    pub hit_pct: f64,
    /// Mean decode wall per token, ms — same clock as `DecodeStats::decode_s`.
    pub wall_ms: f64,
    pub attend_ms: f64,
    pub ffn_ms: f64,
    pub fetch_wait_ms: f64,
    pub head_ms: f64,
    /// `wall − (attend + ffn + fetch-wait + head)`: loop glue, `Emit`, the sink, embed
    /// and flag launches — everything deliberately not stamped. DERIVED, and the only
    /// derived number here; the bucket-sum gate bounds it (measured ~1% on GLM), so a
    /// dropped stamp shows up as `other` exploding rather than as a silent hole.
    /// Negative means a span was double-counted, which the gate's upper bound catches.
    pub other_ms: f64,
}

impl ProfileSummary {
    /// The one constructor: per-token means over a decode of `ntok` tokens that took
    /// `decode` wall time, with `ph` already rebased past the prefill.
    pub fn from_decode(ph: &Phases, decode: std::time::Duration, ntok: usize, hp: f64) -> Self {
        let n = ntok.max(1) as f64;
        let ms = |ns: u64| ns as f64 / n / 1e6;
        let wall_ms = decode.as_secs_f64() * 1e3 / n;
        Self {
            tok_per_s: ntok as f64 / decode.as_secs_f64().max(1e-9),
            hit_pct: hp,
            wall_ms,
            attend_ms: ms(ph.attend_ns),
            ffn_ms: ms(ph.ffn_ns),
            fetch_wait_ms: ms(ph.fetch_wait_ns),
            head_ms: ms(ph.head_ns),
            other_ms: wall_ms - ms(ph.named_ns()),
        }
    }

    /// `other` as a share of wall — **the bucket-sum gate's number, and the only one
    /// it reads.** A dropped accumulation drives it up, a span stamped into two buckets
    /// drives it NEGATIVE, and `tests/ppl-gates.sh`'s `profile` cell bounds it on both
    /// sides; the host tests below judge the same quantity, so the deviceless and the
    /// on-device halves of the gate cannot come to disagree about what they measure.
    ///
    /// This was `named_pct` (= `100 − this`) until 2026-08-16, printed alongside `other`
    /// and documented as what the gate greps. It was not: the gate's regex stops at
    /// `other` and re-derives the share itself. Two spellings of one quantity, one of
    /// them consumed and the other only claimed to be — so the claimed one went.
    pub fn other_pct(&self) -> f64 {
        100.0 * self.other_ms / self.wall_ms.max(1e-9)
    }

    /// The always-on PROFILE line, one per run, on the log stream.
    ///
    /// **Microsecond precision, and it is load-bearing rather than tidy.** At `{:.1}` a
    /// bucket that is genuinely microseconds prints `0.0`, and the gate's per-bucket
    /// census — which reads "> 0" as "the accumulation was stamped" — would call that a
    /// dropped stamp. `ProfileSummary`'s own table says Glimmer's and V4/K3's attend
    /// buckets ARE launch-only microseconds, so a one-decimal line makes the census
    /// false the day the cell is pointed at a second arm (both reviews, 2026-08-16).
    ///
    /// The trailing percentage is [`Self::other_pct`], printed because "other 4.0"
    /// against "wall 390.0" is arithmetic a reader should not have to do — the SAME
    /// number the gate computes from the same line, not a second one that could drift.
    /// The four buckets are printed too, so the gate can re-derive `other` from them and
    /// check this line against itself: `other_ms` is the only derived field here, and a
    /// consumer that reads it without checking it trusts the arithmetic it is auditing.
    pub fn report(&self) {
        tracing::info!(
            "PROFILE/tok: wall {:.3}ms = attend {:.3} + ffn {:.3} + fetch-wait {:.3} + \
             head {:.3} + other {:.3} ({:.2}% of wall)",
            self.wall_ms,
            self.attend_ms,
            self.ffn_ms,
            self.fetch_wait_ms,
            self.head_ms,
            self.other_ms,
            self.other_pct(),
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

/// The phase arithmetic, host-only — every property the bucket-sum gate leans on that
/// does not need a device. The gate's own red-proof (drop one arm's accumulation, watch
/// `other` blow the band) needs a GPU and lives in `tests/ppl-gates.sh`; what is pinned
/// HERE is that the arithmetic those runs are judged by cannot silently change shape.
#[cfg(test)]
mod phase_tests {
    use super::{Phase, Phases, ProfileSummary};
    use std::time::Duration;

    /// A dropped bucket surfaces as `other`, never as a smaller wall — the structural
    /// property the whole design was chosen for (see [`Phases`]'s doc for the rejected
    /// alternative, a switching clock that summed to wall by construction).
    #[test]
    fn a_missing_bucket_lands_in_other_not_in_a_shrunken_wall() {
        let full = Phases {
            attend_ns: 40_000_000,
            ffn_ns: 320_000_000,
            fetch_wait_ns: 12_000_000,
            head_ns: 6_000_000,
        };
        let wall = Duration::from_millis(384); // ≈ the named sum + 6 ms of glue
        let s = ProfileSummary::from_decode(&full, wall, 1, 78.0);
        assert!(
            s.other_pct() < 3.0,
            "healthy profile: other {:.1}%",
            s.other_pct()
        );
        // The red-proof's arithmetic: same run, ffn stamps dropped.
        let dropped = Phases { ffn_ns: 0, ..full };
        let s = ProfileSummary::from_decode(&dropped, wall, 1, 78.0);
        assert!(
            s.other_pct() > 80.0,
            "a dropped ffn bucket must surface as `other`, got {:.1}%",
            s.other_pct()
        );
        assert!((s.other_ms - 326.0).abs() < 1.0, "other absorbs it visibly");
    }

    /// Double-counting (a span stamped into two buckets) drives the stamped sum PAST
    /// wall and `other` NEGATIVE — the gate's lower bound exists for exactly this defect,
    /// and it is tight (zero) rather than fitted, because disjoint spans on one thread
    /// cannot sum past the wall they sit inside.
    #[test]
    fn double_counting_goes_past_wall_rather_than_hiding() {
        let ph = Phases {
            attend_ns: 300_000_000,
            ffn_ns: 300_000_000,
            fetch_wait_ns: 0,
            head_ns: 0,
        };
        let s = ProfileSummary::from_decode(&ph, Duration::from_millis(400), 1, 0.0);
        assert!(s.other_ms < 0.0, "overlap must show as negative other");
        assert!(s.other_pct() < 0.0, "overlap drives the share negative");
    }

    /// `lap` chains without gaps and `since` rebases — the two mechanics every arm's
    /// loop leans on (rebasing past prefill is the same move the hit counters make).
    #[test]
    fn lap_accumulates_and_since_rebases() {
        let mut p = Phases::default();
        let t = std::time::Instant::now();
        let t = p.lap(Phase::Attend, t);
        let _ = p.lap(Phase::Ffn, t);
        assert!(p.attend_ns > 0 && p.ffn_ns > 0);
        let base = p;
        let t = std::time::Instant::now();
        let _ = p.lap(Phase::Head, t);
        let d = p.since(&base);
        assert_eq!(d.attend_ns, 0, "rebased attend");
        assert_eq!(d.ffn_ns, 0, "rebased ffn");
        assert!(d.head_ns > 0, "only post-rebase work remains");
        assert_eq!(d.named_ns(), d.head_ns);
    }

    /// Zero generated tokens must not divide by zero (a run whose first token was EOS
    /// still builds a summary).
    #[test]
    fn zero_tokens_is_a_summary_not_a_panic() {
        let s = ProfileSummary::from_decode(&Phases::default(), Duration::from_millis(5), 0, 0.0);
        assert!(s.wall_ms.is_finite() && s.tok_per_s.is_finite());
    }
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
