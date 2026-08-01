---
status: live
verdict: PARTLY BUILT. Keep OTLP — measured, no leaner path exists at 0.30 and it costs 64 crates. Run-identity labels {mode,cache_policy,attn,max_mem_gib,mtp} SHIPPED 2026-08-02, as did the §3 drops; MTP acceptance and moe-by-miss are still proposed.
---

# What `--features otlp` should export

> **This is a PLAN, not a description.** Almost nothing below is built. It is `live` and it
> sits in `investigations/` because it is an *open* question with a recommendation attached;
> when it is executed it becomes `closed-shipped` and the live half of it belongs in
> [`measurement/traces.md`](../measurement/traces.md), which is where the instrument is
> documented. Read `traces.md` for what the feature *does*; read this for what it *should*.

> **PARTIALLY EXECUTED 2026-08-01, by a different change than this plan.** The
> over-engineering audit (`3e2ed79`) landed **§3, "What to drop"**, almost entirely — it was
> the half that only removed things, so it needed none of the labelling work the rest of the
> plan is blocked on. Specifically:
>
> | §3 row | outcome |
> |---|---|
> | `rivoli.fetch_hidden_pct` gauge | **dropped** — the `ProfileSummary` field went too, so the PROFILE line's `% hidden` term is gone |
> | `split/exposed-fetch` series | **dropped** with `exposed_fetch_ms` |
> | dashboard panel 5, `fetch hidden %` | **repointed, not deleted** — see below |
> | both stale `top-m` doc comments | **gone**, along with the `two_q_kin`/`two_q_kout` fields they were misattached to |
> | `route_j`/`route_m` in `traces.md` | **struck**, with `2q_kin_pct`/`2q_kout_pct` beside them |
> | `loop_period`/`loop_repeats` gauges | **still there.** The one §3 row not executed |
>
> **Panel 5 was repointed at `rivoli_ms_per_tok{class="io-wait"}` rather than deleted.** §3
> asked for deletion because "a panel querying a removed metric draws a plausible zero" —
> that argument is about the *query*, not the *tile*, and the panel's own description already
> said "prefer the io-wait series". So the tile now shows the number it was always deferring
> to, `min` and `unit` corrected from percent to ms and the thresholds dropped (there is no
> good/bad band for io-wait; it overlaps wall by design). Deleting it would have left a hole
> at `x:12` in a five-tile row and cost a re-layout to say less.
>
> Also settled by the same change: the **DEFECT** noted below at `src/telemetry.rs:568`
> (`report` recomputing the retracted `moe_wall − compute_gpu` beside the corrected field) is
> **fixed by construction** — both terms were removed from the PROFILE line. The measured
> counterfactual survives as a local in `Profile::summary` feeding `gpu_wait_ms`.
>
> **Items 1–4 and 7–10 are untouched, and item 1 is still the one that matters**: nothing
> identifies the run, so every metric series is still uncomparable.

> **ITEM 1 SHIPPED 2026-08-01, later the same day.** Run identity is now on every exported
> datapoint: `mode`, `cache_policy`, `attn`, and `max_mem_gib` (omitted when the budget
> auto-sizes), built by `RunInfo::labels()` and appended in `export_metrics` to the
> `ms_per_tok` class gauge and to all six scalars. `--mode hybrid --max-mem 115` and
> `--mode int3-vq --max-mem 70` are no longer the same series. Documented in
> [`measurement/traces.md`](../measurement/traces.md) under "Metric labels — which run this
> is", which is where the live half belongs.
>
> **Two deliberate departures from §2a as written:**
>
> - **`mtp_min_conf` is NOT in the label set.** §2a lists it as "new field, see 2b", and
>   adding it means adding a `RunInfo` field *and setting it in `src/main.rs`* — which makes
>   it part of item 3, not item 1. It is the one row of §2a's table still outstanding, and
>   the gate setting therefore remains invisible on both the chart and the root span.
> - **The test that landed is the label half, not §5's `series()` test.** §5 step 1 (moving
>   the class table onto `ProfileSummary::series()`) is checklist item 2 and is untouched;
>   the same *structural* argument was applied to the labels instead — `RunInfo::labels()`
>   is deliberately outside `mod otlp`, so the default `--features rocm` build compiles the
>   only place `RunInfo`'s fields are named for the exporter, and
>   `telemetry::run_label_tests` (three tests, no feature, no GPU, no collector) asserts the
>   label keys, the omit-when-`None` rule, and that two budgets do not collide. Item 2 still
>   needs doing for the class table, which is where the `launch_ms` E0609 actually came from.

## STATE — the answer in fifteen lines

1. **The metrics half has a hole that makes it nearly useless, and it is not the crate
   count.** Every gauge is emitted with `class`/`thread` labels *only*
   (`src/telemetry.rs:923-986`). Nothing identifies the run. `rivoli_tok_per_s` from
   `--mode hybrid --max-mem 115` and from `--mode int3-vq --max-mem 70` are the **same
   Prometheus series**. `traces.md:106` tells the reader to "search by which mode / which
   policy / which budget, then read the numbers off a chart" — the chart has no such label
   to filter on. Fix this first; nothing else in this plan matters until it is done.
   **DONE 2026-08-01** — four of §2a's five labels are on every datapoint; see the note
   above for the fifth (`mtp_min_conf`, which belongs to item 3).
2. **Speculative decode is on by default and is invisible to the exporter.** `mtp_hit` /
   `mtp_seen` / `mtp_verify` (`src/gpu.rs:434-458`) go to stdout only
   (`src/gpu.rs:2610-2678`). Break-even is 53% acceptance and acceptance tracks the *text*
   (46.0% degenerate vs 65.7% coherent, `reference/architecture.md` §13) — so it varies per
   run, which is exactly what a time series is for.
3. **`moe_us_by_miss` is the honest stall measurement and is not exported.** It is already
   in `ProfileSummary` (`src/telemetry.rs:473`) and it is the number
   `reference/architecture.md` §3 weights the drive's QD table against.
4. **Drop** `fetch_hidden_pct` and `exposed_fetch_ms` (removed by a separate audit item),
   the two `top-m` doc-comment leftovers, and `traces.md`'s `route_j`/`route_m` claim.
5. **Dependency: keep full OTLP.** +64 crates, measured. There is no leaner path — feature
   trimming at 0.30 does not drop `tonic`, verified. A JSON/scrape replacement saves **zero**
   crates because the traces buy them, and the traces are the irreplaceable half.
6. **The test that pins it is not a CI job** (there is no CI). It is moving the metric table
   out of `#[cfg(feature = "otlp")]` into always-compiled code, so `cargo build --features
   rocm` is the gate. That alone would have caught the `launch_ms` E0609.

---

## 1. What is exported today

### Traces (only when `--spans <BUDGET>` is given; `src/main.rs:165-179`)

`rivoli.decode` → `token N` → `layer L` → leaf intervals, rebuilt from the recorder's flat
list at `src/telemetry.rs:769-859`. Twelve distinct leaf names exist, all measured, all from
intervals the forward pass already pays for:

| leaf | site |
|---|---|
| `gpu-wait/gate-d2h` | `src/gpu.rs:1708` |
| `gpu-wait/argmax-d2h` | `src/gpu.rs:2337` |
| `gpu-wait/end-of-layer-sync` | `src/gpu.rs:2144` |
| `gpu-wait/{idx-sync,idx-scores-d2h,misa-sync,misa-d2h}` | `src/gpu.rs:1084-1167` |
| `cpu/launch`, `cpu/launch-tail` | `src/gpu.rs:1686,1700,2253` |
| `cpu/route-into`, `cpu/submit-layer` | `src/gpu.rs:1767,1837` |
| `io-wait/uring-reap` | `src/fetch/asyncfetch.rs:418` |

Root attributes are run identity, deliberately not measurements
(`src/telemetry.rs:732-767`). `layer L` carries `experts.{cold,warm}.{int4,int3_vq}`
(`src/telemetry.rs:821-835`). Sampling is whole-token with a derived stride
(`src/telemetry.rs:104-132`, `src/gpu.rs:2461`).

**This half is in good shape.** Two gaps, both in §3 below.

### Metrics (always, when the endpoint is set; `src/telemetry.rs:901-990`)

One `rivoli.ms_per_tok{class,thread}` gauge carrying 15 class values, plus seven scalar
gauges (`tok_per_s`, `hit_pct`, `gb_per_tok`, `miss_per_tok`, `fetch_hidden_pct`, `tokens`,
`degenerate`) and two conditional ones (`loop_period`, `loop_repeats`).

> **As of 2026-08-01 this reads 14 class values and six scalar gauges** — `fetch_hidden_pct`
> and the `split/exposed-fetch` class were dropped by §3 (see the note at the top of this
> file). The inventory is left as it was surveyed, because §2 and §3 below both argue
> against it and a plan whose "before" picture has been quietly edited cannot be audited.

### The dashboard (`docs/measurement/grafana-dashboard.json`, uid `rivoli-decode`)

13 panels. Every one queries `rivoli_ms_per_tok{class=...}` or a scalar gauge; there is **no
template variable over run identity** — `templating.list` holds only the two datasource
pickers. Panel 5 is `fetch hidden %`; panel 11 (`Splits`) draws `class=~"split/.*"`, which
includes `split/exposed-fetch`.

> **2026-08-01:** panel 5 is now `io-wait / tok` and panel 11's regex matches four live
> splits, not five. **The run-identity gap is untouched and is still the finding of this
> file** — 13 panels, no way to tell two runs apart.

---

## 2. What is worth exporting, and why

The engine is NVMe expert streaming overlapped with resident compute. The drive is at
capacity (`reference/architecture.md` §3: ~10 GB/s achieved against a 7.7–13.0 GB/s probe
table), so the live levers are **fewer bytes** (roadmap #2) and **fewer passes**
(speculative decode, roadmap #4). A metric earns its place by moving one of those or by
telling you a run is not comparable.

### 2a. Run identity on every datapoint — **do this first, alone if nothing else**

> **BUILT 2026-08-01**, everything below except the `mtp_min_conf` row. The mechanism is
> exactly as proposed — one shared attribute set appended to every `record()` — with the
> set itself built by `RunInfo::labels()` in always-compiled code rather than inside
> `mod otlp`, so a `RunInfo` field rename is a compile error under plain `--features rocm`.

Without it the metrics half cannot answer any question it is drawn for. Add a shared
attribute set to every `record()` in `export_metrics`:

| label | source | cardinality |
|---|---|---|
| `mode` | `RunInfo::mode` | 3 |
| `cache_policy` | `RunInfo::cache_policy` | 3 |
| `attn` | `RunInfo::attn` | 4 |
| `max_mem_gib` | `RunInfo::max_mem_gib` (omit when `None`) | ~10 |
| `mtp_min_conf` | new field, see 2b | ~5 |

**Datapoint attributes, not resource attributes.** Alloy's
`otelcol.exporter.prometheus` leaves resource attributes in `target_info` unless
`resource_to_telemetry_conversion` is set, and that switch lives in a config file outside
this repo. `traces.md:174-179` already records what happens when a metric's shape depends on
an invisible collector setting: every panel draws an empty graph rather than an error, "the
worst possible failure mode". Do not build a second instance of that.

**Never label with `prompt` or `model`** — unbounded, and the prompt is already a span
attribute where cardinality is free.

Once this lands, the dashboard gets a `mode`/`policy` template variable and the whole
benchmark matrix becomes one chart instead of 44 logs.

### 2b. MTP acceptance under the `--mtp-min-conf 0.8` gate

On by default (`src/main.rs:218` — the flag is `--no-mtp`), worth 1.108×, and break-even is
a knife-edge 53% that moves with the text. The counters exist and are already computed at
`src/gpu.rs:2610-2678`; they are simply not on `ProfileSummary`. Add four fields and four
gauges:

| gauge | value | why |
|---|---|---|
| `rivoli.mtp_accept_pct` | `mtp_hit / mtp_seen` | the number break-even is measured against |
| `rivoli.mtp_speculated_pct` | `mtp_verify / mtp_seen` | what the gate actually did — `g` in the cost model |
| `rivoli.tokens_per_pass` | `generated / mtp_seen` | the measured speedup, not a projection |
| `rivoli.mtp_draft_ms` | `mtp_draft_ns / mtp_draft_n` | `d`; caps what a pre-draft gate could ever save |

Record nothing when `mtp_seen == 0`, following the rule `report()` already keeps for the
indexer (`src/telemetry.rs:644`): a zero would read as a measurement of something that did
not happen.

Also carry `mtp_min_conf` into `RunInfo` so the gate setting is on the root span and in the
label set. Today a trace cannot tell you what threshold produced it.

### 2c. `moe_us_by_miss` — the per-layer stall shape

Already on `ProfileSummary` (`src/telemetry.rs:473`), already printed
(`src/telemetry.rs:586-606`), exported nowhere. It is the one instrument that separates "the
shaders are slow" from "the stream is starved", it is what `reference/architecture.md`
§3's drive table is weighted by, and it is the honest thing to look at in place of a hiding
percentage. Two gauges with a `misses` label:

```
rivoli.moe_us_by_layer{misses="0".."15"}   # mean bracket, µs
rivoli.moe_layers{misses="0".."15"}        # n, so a thin bucket can be discounted
```

Skip empty buckets. Bounded cardinality: 16.

### 2d. The indexer, when it scored

`idx_gpu_ms` and `idx_layers_per_tok` are on the summary and printed
(`src/telemetry.rs:644-652`) but not exported. Under `--attn dsa|misa` this is the whole
attention story, and roadmap #5 lives next door. Two gauges, guarded by
`idx_layers_per_tok > 0.0` exactly as the print is.

### 2e. Degeneration: export the detector that actually fires

`RunInfo::degenerate` carries `detect_loop` only (`src/main.rs:710,777`). But `main` also
computes `longest_repeated_block` (`:715`) and `repetition_report` (`:731`), and warns
`STRUCTURALLY DEGENERATE` at `:738` — **and none of that reaches the span or the gauge.**
The 329-repeat `**Memory Product.**` failure that motivated `repetition_report`
(`src/telemetry.rs:361-382`) sets `rivoli.degenerate = 0` today.

That is the exported degeneration flag missing the most common failure shape there is. Fix:
add `top_line: usize` and `longest_repeated_block: usize` to `RunInfo`, and set the
attribute and the gauge from `detect_loop(..).is_some() || is_degenerate(&rep)`. A query
that excludes degenerate cells is only worth having if it excludes them all.

### 2f. One span attribute: which token spans were verify passes

MTP is on by default, so **every trace recorded today is of a speculative run and does not
say so.** Under the gate roughly half the passes are two-row and half are one-row
(`architecture.md` §13: tokens/pass 1.459), and the two shapes differ by ~1.53× in cost
because the MoE launches the union of both rows' routing. On the timeline they are just
`token N` spans of wildly different length with nothing to explain it.

Add `rivoli.mtp.speculated` (bool) and `rivoli.mtp.accepted` (bool) to the `token N` span.
Mechanically this is one more atomic beside `CUR_TOK_ID` (`src/telemetry.rs:79`) set by the
decode loop where it already calls `spans::mark` (`src/gpu.rs:1460`).

### 2g. What is deliberately NOT added

- **Eviction counters / arena occupancy.** `hit_pct` and `gb_per_tok` already carry the
  residency signal, and `reference/architecture.md` §3 closed the fetch-tuning door: only
  moving fewer bytes helps. A counter that answers no live question is the thing this repo
  keeps deleting.
- **`slot_stalls`.** Zero on every run measured, and `src/gpu.rs:2603` already warns
  loudly when it is not. A gauge that is always 0 trains people to ignore the panel.
- **Any per-token or per-layer metric stream.** The spans are that view, they cost a `Vec`
  push, and they already carry `experts.cold`/`experts.total` per layer.

---

## 3. What to drop

| drop | where | why |
|---|---|---|
| `rivoli.fetch_hidden_pct` gauge | `src/telemetry.rs:974-976` | the field is removed by the approved audit item |
| dashboard panel 5, `fetch hidden %` | `grafana-dashboard.json` | queries a metric that will not exist; an empty stat panel reads as "0% hidden" |
| `split/exposed-fetch` series | `src/telemetry.rs:960` | same, `exposed_fetch_ms` goes with it |
| `rivoli.loop_period` / `rivoli.loop_repeats` gauges | `src/telemetry.rs:983-986` | present only on failing runs, unlabelled, unchartable. They are already span attributes (`:763-767`), which is the right home for them |
| stale `top-m` doc comment | `src/telemetry.rs:451-453` | describes `(J, M)` under a retired policy; it is attached to `two_q_kin`/`two_q_kout`, which are 2Q's percentages |
| stale `top-m` doc comment | `src/telemetry.rs:516-520` | **worse: it has no field.** The block ends mid-sentence and rustdoc attaches all 18 lines to `gpu_wait_ms` (`:533`), so `gpu_wait_ms` is currently documented as a top-m substitution rate |
| `route_j` / `route_m` in the attribute table | `traces.md:98` and the sentence at `:108` | the engine emits neither; `topk_path` beside it was already corrected, these were missed |

Nothing else in the metric set is dead. The `cpu/tokio-poll` tombstone comment
(`src/telemetry.rs:941-945`) should stay — it explains a series that vanished from a live
dashboard.

### Two corrections the deletion of `fetch_hidden_pct` should carry with it

Recorded here because this repo corrects in place rather than deleting, and because both
were found while reading for this plan.

> **NOTE 2026-08-01.** The 27-line doc comment at `src/telemetry.rs:490-512` — the one that
> says `fetch_hidden_pct` is "SUBSTANTIALLY OVERSTATED" and reports 96% against a true ≤57%
> — **describes a formula the engine no longer uses.** Commit `ff3d51b` replaced
> `1 − (moe_wall − compute_gpu)/fetch_wall` with a measured counterfactual:
> `exposed = moe_wall − moe_ns_by_miss[0]/n × instances` (`src/gpu.rs:279-288`), which
> `reference/architecture.md:147-150` records as putting bounce at 22% and `--direct-vmm-dma`
> at 10% — the same ordering as throughput. The retraction is real; the doc comment was
> never updated to say the retraction had been *acted on*. This does not change the
> recommendation to remove the metric (`moe_us_by_miss` in §2c measures the stall instead of
> inferring its absence, and one honest number beats two), but whoever executes the
> deletion should know it is being justified by prose that is a commit behind.

> **DEFECT, unrelated to OTLP but on the same line.** `ProfileSummary::report` recomputes
> `let exposed = (self.moe_wall_ms - self.compute_gpu_ms).max(0.0)` at
> `src/telemetry.rs:568` — the **retracted** formula — and prints it beside
> `self.fetch_hidden_pct`, which is the **corrected** one. The stdout PROFILE line therefore
> mixes an honest percentage with a stale absolute, and `self.exposed_fetch_ms` (the honest
> field, `src/gpu.rs:354`) is never printed at all. Whichever way the audit item lands, line
> 568 must stop deriving what `summary()` already measured.

---

## 4. The dependency question: keep full OTLP

**Measured, this repo, 2026-08-01:**

```
cargo tree --offline --features rocm       --no-default-features   → 104 crates
cargo tree --offline --features rocm,otlp  --no-default-features   → 168 crates
Cargo.lock                                                          → 187 packages
```

**+64 crates**, and the list is what the audit said: `reqwest`, `hyper`, `tonic`, `tower`,
`tower-http`, `prost`, `mio`, and the entire ICU stack (`icu_normalizer`,
`icu_properties_data`, `zerovec`, …) arriving via `idna` → `url` → `reqwest`.

### There is no leaner OTLP path at 0.30 — this was tested, not assumed

A probe crate resolving `opentelemetry-otlp = "0.30"` three ways:

| config | crate set |
|---|---|
| current (`features = ["metrics"]`, defaults on) | baseline |
| `default-features = false` + `["trace","metrics","http-proto","reqwest-blocking-client"]` | **identical** |
| `default-features = false` + `["trace","metrics","http-json","reqwest-blocking-client"]` | +2 (`serde_json` path) |

`cargo tree -i tonic` explains it: `tonic ← opentelemetry-proto v0.30 ← opentelemetry-otlp`,
and `opentelemetry-proto` resolves with `FEATURES=…gen-tonic-messages,tonic,…` even under an
http-json-only configuration. **`tonic`/`tower`/`hyper` cannot be trimmed from the
manifest.** Do not spend an afternoon on `default-features = false`; it buys nothing.

### The three options, and the recommendation

| option | crates saved | what it costs |
|---|---:|---|
| **Keep full OTLP** ✅ | 0 | 64 opt-in crates, never in a shipped binary |
| Hand-rolled OTLP/HTTP JSON | ~64 | ~200 lines of spec-shaped encoder, against a wire format with no compiler to check it, re-creating the exact "no test, silently rots" failure this feature just recovered from |
| JSON/Prometheus text + scrape | **0** | an operator migration, for zero crates |

**Recommend: keep full OTLP, both signals, unchanged.**

The third option is the one that looks attractive and is not. The 64 crates are bought by
the **traces**, and the traces are the half that cannot be replaced: a Tempo waterfall
showing `io-wait/uring-reap` bars sitting *under* `gpu-wait/*` bars is the only view that
settles the overlap question this engine's whole design rests on, and no amount of scalar
export produces it. Moving the metrics half to a scrape file leaves every one of those 64
crates in place *and* costs an Alloy/Grafana migration on a pipeline that is verified working
(`traces.md:116-139`). Zero savings, real cost.

**State the price plainly rather than paying it down.** `otlp` is `default = []`
(`Cargo.toml:86`), it is not in `rocm` or `vulkan`, and `cargo build --release --features
rocm` never compiles a line of it. The lockfile carries the 64 entries either way, because
optional dependencies are always locked. So the true cost is: a slower build for whoever
asks for the feature, and 64 more crates in an audit. That is a fair price for the only
instrument in the repo that can *draw* the overlap.

**Not doing, and closing the question:** a stock-build machine-readable summary
(`--profile-json <path>`, ~15 lines, `serde_json` is already an unconditional dependency) is
a genuinely good idea for benchmark-matrix work and is **not part of this plan**. It is not
an OTLP alternative — it needs no feature and competes with nothing here. Propose it
separately if the matrix work wants it.

---

## 5. The test that pins it

The E0609 happened because `summary.launch_ms` was named **only** inside
`#[cfg(feature = "otlp")] mod otlp`, and nothing anyone runs compiles that. There is no CI;
`cargo test` is the whole gate; a featureless build must still compile. So the fix is
structural, not procedural.

**Step 1 — move the metric table into always-compiled code.** `src/telemetry.rs` is
unconditional (`src/lib.rs:20`), so put the class table on `ProfileSummary` itself:

```rust
impl ProfileSummary {
    /// (class, thread, ms). The ONE list of what the class axis contains — `report()`
    /// prints it and the OTLP exporter records it, so a deleted field is a compile
    /// error under plain `--features rocm` rather than a surprise for whoever next
    /// builds `otlp`. That is exactly how `launch_ms` got out (traces.md, "How it rotted").
    pub fn series(&self) -> Vec<(&'static str, &'static str, f64)> { … }
}
```

`export_metrics` then iterates it instead of naming fields. Every field access moves into
code the default build compiles. **This alone would have caught the E0609**, and it removes
a duplicated list rather than adding one.

**Step 2 — one test, no feature, no GPU, no collector.**

```rust
#[test]
fn the_metric_series_are_named_and_finite() {
    let s = ProfileSummary { /* every field, from literals */ };
    let names: Vec<_> = s.series().iter().map(|(c, _, _)| *c).collect();
    // The dashboard queries these strings literally. A rename here is an empty
    // graph there, which reads as zero rather than as an error.
    for want in ["wall", "gpu-wait", "io-wait", "cpu", "cpu/launch", "cpu/route",
                 "cpu/submit", "phase/route", "phase/moe", "phase/tail"] {
        assert!(names.contains(&want), "dashboard queries {want}; series are {names:?}");
    }
    // Retired series must not come back by copy-paste.
    for gone in ["cpu/tokio-poll", "split/exposed-fetch"] {
        assert!(!names.contains(&gone));
    }
    assert!(s.series().iter().all(|(_, _, v)| v.is_finite()));
}
```

Three properties, each of which has already failed in this repo once:

1. **Constructing `ProfileSummary` from literals** forces the test to be edited when a field
   is added or removed — which is the mechanism, not the assertions.
2. **Naming the dashboard's query strings** makes a rename a two-file edit instead of a
   silently empty panel (`traces.md:174-179` on `add_metric_suffixes`).
3. **`is_finite`** catches the division-by-zero shape that `report()` guards by hand at
   `:604` and `:650`.

**Step 3 — say so where someone will read it.** `traces.md:263-266` already carries the
"nothing in CI compiles `--features otlp`" warning. Add the one-line build to the repo's
standard block so it is a habit rather than a warning:

```bash
cargo build --release --features rocm,otlp    # the arm no other command covers
```

**Explicitly not proposed:** a CI job (there is no CI), a mock OTLP collector in-tree, or an
integration test that needs an endpoint. Each is more machinery than the failure justifies —
the failure was a field name, and a compiler catches field names for free once the field is
named in code that gets compiled.

---

## 6. Implementation checklist, in order

Each step is independently landable and independently useful. Steps 1–2 are the ones that
make the feature genuinely useful; the rest is hygiene.

- [x] **1. Run identity on every metric datapoint** (`src/telemetry.rs:923-986`). Build one
      `Vec<KeyValue>` from `RunInfo` (`mode`, `cache_policy`, `attn`, `max_mem_gib` when
      `Some`, `mtp_min_conf`) and append it in the `g()` closure and every scalar `record()`.
      Add `mtp_min_conf: f32` to `RunInfo` (`src/telemetry.rs:441`, set at
      `src/main.rs:763-778`) and to the root span. **Without this step nothing else on this
      list is worth doing.**
      **DONE 2026-08-01** for the four `RunInfo` already carries, via `RunInfo::labels()` +
      `telemetry::run_label_tests`. `mtp_min_conf` is **not** done: the field does not exist
      and populating it is a `src/main.rs` edit, so it moves to item 3, which is where the
      rest of the MTP work is. Until then no chart can tell a gated run from an ungated one.
- [ ] **2. Move the class table to `ProfileSummary::series()`** (§5 step 1) and rewrite
      `export_metrics` + `report()`'s class block to iterate it. Add the test (§5 step 2).
      Run `cargo test --release --features rocm` **and** `cargo build --release --features
      rocm,otlp`.
- [ ] **3. MTP gauges** (§2b). Four fields onto `ProfileSummary`, populated in
      `Profile::summary` — note the MTP counters live on `GpuEngine`, not `Profile`, so they
      pass as arguments the way `hits`/`misses`/`fetch_wall_ns` already do
      (`src/gpu.rs:255-262`). Guard on `mtp_seen > 0`.
      **Also inherits §2a's last row**: add `mtp_min_conf` to `RunInfo` and to
      `RunInfo::labels()` — one line each — so the gate setting becomes a label like the
      rest. Item 1 could not, because setting the field is a `src/main.rs` edit.
- [ ] **4. `moe_us_by_miss` gauges** (§2c). Data is already on the summary; this is a loop
      over 16 buckets, skipping `None`.
- [x] **5. Drop the dead metrics** (§3): `fetch_hidden_pct`, `split/exposed-fetch`,
      ~~`loop_period`/`loop_repeats` gauges~~. Delete dashboard panel 5 in the same commit —
      a panel querying a removed metric draws a plausible zero.
      **DONE 2026-08-01 except the loop gauges, which remain.** Panel 5 was **repointed** at
      `rivoli_ms_per_tok{class="io-wait"}` rather than deleted — see the note at the top.
- [x] **6. Fix the two `top-m` doc comments** (`src/telemetry.rs:451-453`, `:516-520`). The
      second is the load-bearing one: it currently documents `gpu_wait_ms`.
      **DONE 2026-08-01** — both comments and the `two_q_kin`/`two_q_kout` fields under them
      are gone with the `--2q-kin`/`--2q-kout` flags.
- [ ] **7. Degeneration flag** (§2e): `top_line` + `longest_repeated_block` onto `RunInfo`,
      OR'd into the attribute and the gauge.
- [ ] **8. Indexer gauges** (§2d), guarded by `idx_layers_per_tok > 0.0`.
- [ ] **9. `token N` span MTP attributes** (§2f) — one atomic beside `CUR_TOK_ID`.
- [ ] **10. Dashboard**: add a `mode`/`cache_policy` template variable over the new labels,
      an MTP row (accept %, speculated %, tokens/pass), and a `moe_us_by_layer` panel keyed
      on `misses`. Bump nothing else; the CLASS/PHASE stacking distinction
      (`traces.md:224-227`) is correct and must survive the edit.
- [ ] **11. `traces.md`**: ~~strike `route_j`/`route_m` (`:98`, `:108`)~~ **done 2026-08-01,
      with `2q_kin_pct`/`2q_kout_pct` and `rivoli_fetch_hidden_pct` beside them**; ~~document
      the new metric set and labels in the emit table (`:85-86`), and add the `rocm,otlp`
      build line~~ **done 2026-08-01 for the labels** — the emit row names them, a new
      "Metric labels — which run this is" section states the datapoint-not-resource rule and
      the cardinality bound, the Alloy config comment says why `target_info` is the trap, and
      the `rocm,otlp` build line is in "Things that will bite". Still to document: whatever
      items 3, 4, 7 and 8 add. Then move this file's live half into it and re-status this doc
      `closed-shipped`.

      **For whoever does that re-status:** this file's `verdict:` front matter still reads
      "add run-identity labels **without which every metric series is uncomparable**", which
      item 1 has now answered. It was left alone deliberately — the verdict is duplicated in
      [`00-orientation/INDEX.md`](../00-orientation/INDEX.md) and `tests/docs.rs` fails
      unless both change in the same edit.
- [ ] **12.** `cargo test --release --features rocm --test docs` and
      `cargo clippy --release --features rocm,otlp --all-targets`.

**Do not** re-verify the pipeline end to end as part of this. `traces.md:116-139` records the
gateway as up and confirmed by the operator; re-probing it costs a session and reproduces
nothing that is not already written down.
