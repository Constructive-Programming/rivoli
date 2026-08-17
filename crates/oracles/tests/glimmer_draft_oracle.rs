//! **The DFlash drafter's VALUE gate: `rivoli_oracles::dflash` scored against the vendored
//! draft goldens, capture by capture, at measured tolerances.**
//!
//! `glimmer_anchor_draft.rs` gates the SHAPES that make the drafter a drafter; this binary is
//! the arithmetic half those shapes pass over in silence — the M17a oracle. Three layers of
//! evidence, each proven able to fail:
//!
//! 1. **The parameter draws are bit-exact.** The draft goldens vendor no weights (the driver
//!    refuses `--dump-weights` off the clean text run), so the oracle REGENERATES them via
//!    `torchdraw`; that transliteration is proven against the two vendored TEXT weight sets —
//!    107 first-party tensors per salt reproduced bit for bit — before any forward is believed.
//! 2. **The clean forward sits inside floors measured from the reference itself.** Every
//!    tolerance is 10x the fp32-vs-fp64 distance of the SAME reference computation
//!    (`--dtype float64`), measured 2026-08-16 on both salts before this oracle existed —
//!    commands in `docs/measurement/glimmer-reference/anchor.md`.
//! 3. **Five planted defects redden where they reach and hold where they do not.** An oracle
//!    that disagrees everywhere proves nothing, so every defect test asserts BOTH halves.
//!
//! The comparison guards its own reference first: a capture that is non-finite or degenerate
//! fails before any tolerance is consulted — `f32::max` ignores NaN and `!=` is true on it,
//! and both have produced false greens in this repo's history.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#[path = "glimmer_anchor_common/mod.rs"]
mod anchor; // keep this preamble blank-line-free: spread out, these are a jscpd clone
#[path = "common/tolerance.rs"]
mod tolerance;
use anchor::{GoldenSet, cfg, draft_goldens, golden_read, ints, load, meta, meta_usize, num};
use rivoli_oracles::dflash::{self, DraftDefect, DraftDims, DraftInputs, DraftParams, DraftTrace};
use rivoli_oracles::torchdraw::{self, Family};
use serde_json::Value;
use tolerance::{Policy, Tol, rel_row as row};

/// The two vendored weight sets, by the same bytes `glimmer_anchor.rs` pins. Provenance is NOT
/// re-checked here — that file gates length and FNV of all six vendored FILES (four goldens plus
/// these two weight sets; read "six anchor binaries" until 2026-08-17, conflating them with the
/// five test binaries), and a second frozen copy of those numbers agreeing with the first is not
/// a check (`crates/engine/tests/glimmer_anchor/mod.rs` states the same rule).
const WEIGHT_SETS: [(&str, &[u8]); 2] = [
    ("weights-1", include_bytes!("glimmer-anchor-weights-1.bin")),
    ("weights-2", include_bytes!("glimmer-anchor-weights-2.bin")),
];

// ------------------------------------------------------------------------------------------
// Tolerances: measured floors, the 10x rule, and where each defect's signal was measured.
// ------------------------------------------------------------------------------------------

/// Per-capture tolerances for the oracle-vs-golden comparison, under `tolerance.rs`'s own
/// apparatus (`Policy::Rel`, [`tolerance::tolerances_leave_room`]).
///
/// **`floor`** is the reference's fp32 run against its own fp64 run (`--dtype float64`), worst
/// over both salts and all five layers, measured 2026-08-16 BEFORE this oracle existed. The
/// oracle computes in f64, so a correct oracle sits at ~the floor and a tolerance of 10x
/// admits it; the commands are in `glimmer-reference/anchor.md` §draft-tolerances.
///
/// **`weakest_defect`** is a measured LOWER BOUND on the smallest signal among this binary's
/// planted defects, **min over both salts** (the binding case: a tolerance must catch the defect
/// on every draw).
///
/// > **SWEPT AND CORRECTED 2026-08-17**, all four defects x ten operators x both salts; the 4x10
/// > matrix and the owed standing-sweep are `anchor.md` §`weakest_defect` SWEPT. Every row is at
/// > or under its measured weakest signal (8,132x-56,336x above tolerance), so no
/// > `tolerances_leave_room` verdict was wrong — but the rule stated here (targeting-defect for
/// > the five attend-side rows, min-over-all-four for the rest) does NOT describe them: it cannot
/// > hold for `encoder.out`, where three defects leave the capture at the clean 2.980e-7 floor
/// > (§11 step 4, Q never sees the context) and would put its bound under its own tolerance; and
/// > seven of ten declared values are this file's L0 figures, not worst-layer ones (`attend.q`
/// > 5.031e-1 declared against a 6.170e-1 worst-layer minimum). A bound is all this column is.
///
/// **CORRECTED 2026-08-16, same day.** This said the six non-attend operators "are upstream of
/// every trap at L0 BY DESIGN (they are the defect matrix's hold half), so their reach is
/// through the corrupted stream at L1+". Review traced `layer_forward` and that is false for
/// four of them: `post_attention_layernorm.out`, `mlp.out`, `final_norm.out` and `logits` are
/// all strictly DOWNSTREAM of the attend traps at L0 already (`x += o_proj(attend_out)` comes
/// before `ln_post`). Only `encoder.out` and `input_layernorm.out` are upstream, and only
/// `input_layernorm.out` ever appears in a `holds` list. The numbers are unaffected — every one
/// is ~4 decades above its tolerance either way — but the rule as written did not describe
/// where they came from, which is how a measured number becomes an inherited one.
///
/// Written to 3 s.f., not the K3/GLIMMER tables' 2. **CORRECTED 2026-08-16, same day:** this
/// said "two of these floors (2.0248e-6, 2.4320e-6) round at 2 s.f. to 9.88x/9.87x", and
/// recomputing it for review found FOUR outside `FLOOR_MULT`'s (9.9, 10.2) band — `encoder.out`
/// 9.879x, `input_layernorm.out` 9.878x, `final_norm.out` 9.868x and, worst and unnamed,
/// `logits` at **9.820x**. The decision was right and the count was inherited rather than
/// derived, which is the failure this repo names most often. At 3 s.f. every ratio lands in
/// 9.976x–10.012x, inside the band without loosening the rule.
///
/// **Ten rows for eleven captures**: the reference captures the SAME tensor as
/// `draft.final_norm.out` and `draft.last_hidden` (the hook on `norm`, and the model output),
/// so both score through the `final_norm.out` row — one row per OPERATOR, which is what a
/// tolerance is a property of, and the same merge `anchor.md`'s floor table makes.
const DRAFT_ORACLE: &[Tol] = &[
    row("encoder.out", 2.3281e-6, 9.832e-1, 2.33e-5),
    row("input_layernorm.out", 2.4993e-6, 2.457e-1, 2.50e-5),
    row("attend.q", 4.7961e-6, 5.031e-1, 4.80e-5),
    row("attend.k", 2.9202e-6, 4.632e-1, 2.92e-5),
    row("attend.v", 2.3172e-6, 1.307e0, 2.32e-5),
    row("attend.out", 4.6989e-6, 1.015e0, 4.70e-5),
    row("post_attention_layernorm.out", 2.7168e-6, 2.212e-1, 2.72e-5),
    row("mlp.out", 3.7631e-6, 3.936e-1, 3.76e-5),
    row("final_norm.out", 3.2815e-6, 2.850e-1, 3.28e-5),
    row("logits", 2.9504e-6, 4.015e-1, 2.95e-5),
];

/// The `Rel` threshold for one operator, with the two failures kept distinct for the reason
/// `k3/tolerance.rs::rel_tolerance` states: `None` means nothing is being scored, which an
/// `unwrap_or(default)` would turn into silence, and `ExactOnly` means someone decided no
/// threshold separates correct from defective and this comparison must not invent one.
fn tol_for(operator: &str) -> f32 {
    match tolerance::tolerance(DRAFT_ORACLE, operator) {
        Some(Policy::Rel(tol)) => *tol,
        Some(Policy::ExactOnly) => panic!("{operator}: ExactOnly has no Rel bound"),
        None => panic!("no tolerance registered for {operator}"),
    }
}

/// The bounds and the defect margins hold together, per the shared apparatus.
#[test]
fn the_draft_tolerances_leave_room() {
    tolerance::tolerances_leave_room(DRAFT_ORACLE);
}

// ------------------------------------------------------------------------------------------
// Reading one draft case out of the goldens.
// ------------------------------------------------------------------------------------------

/// One salt's full scoring input: the golden, its paired weight set, and the dims both agree on.
struct Case {
    name: &'static str,
    golden: GoldenSet,
    weights: GoldenSet,
    dims: DraftDims,
    salt: String,
}

fn dims_from(golden: &GoldenSet, c: &Value) -> DraftDims {
    let w = anchor::widths(c);
    DraftDims {
        hidden: w.hidden,
        heads: w.heads,
        kv_heads: w.kv,
        head_dim: w.head_dim,
        inter: num(c, "intermediate_size"),
        layers: num(c, "num_hidden_layers"),
        block: meta_usize(golden, "block_size"),
        window: num(c, "sliding_window"),
        targets: ints(golden, "target_layer_ids").len(),
        eps: c["rms_norm_eps"].as_f64().expect("rms_norm_eps"),
        rope_theta: c["rope_parameters"]["rope_theta"]
            .as_f64()
            .expect("rope_theta"),
    }
}

/// Both salts, paired with their weight sets. **The census is here, not in each test**: every
/// test in this binary is `for case in cases()`, so a re-vendor that renamed the entries out of
/// `draft_goldens()`'s prefix filter would take the whole file green with nothing compared.
fn cases() -> Vec<Case> {
    let out = draft_cases();
    assert_eq!(
        out.len(),
        2,
        "draft case census — both salts must be present"
    );
    out
}

fn draft_cases() -> Vec<Case> {
    draft_goldens()
        .zip(WEIGHT_SETS)
        .map(|(v, (wname, wbytes))| {
            let golden = load(v);
            let weights = GoldenSet::read_glimmer(&mut &wbytes[..])
                .unwrap_or_else(|e| panic!("{wname}: {e:#}"));
            // The drafter borrows the TARGET's embedding and lm_head, so the pairing must be
            // by SALT, not by list position happening to line up.
            let salt = meta(&golden, "salt").to_owned();
            assert_eq!(
                salt,
                meta(&weights, "salt"),
                "{}: paired with {wname}",
                v.name
            );
            let c = cfg(&golden);
            Case {
                name: v.name,
                dims: dims_from(&golden, &c),
                golden,
                weights,
                salt,
            }
        })
        .collect()
}

impl Case {
    /// A float capture's values, with the guard the module header promises: finite and
    /// non-degenerate BEFORE any comparison consults it.
    fn want(&self, name: &str) -> &[f32] {
        let (_, vals) = golden_read::float(&self.golden, name);
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "{}: {name} carries non-finite reference values",
            self.name
        );
        assert!(
            vals.iter().any(|v| *v != 0.0),
            "{}: {name} is all zero — a degenerate reference proves nothing",
            self.name
        );
        vals
    }

    fn weight(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let (shape, vals) = golden_read::float(&self.weights, name);
        (shape.to_vec(), vals.to_vec())
    }

    /// The draft block embedded from the target's vendored matrix, raw or with the planted
    /// norm — the shared prelude of the two embedding gates, whose difference IS the defect.
    fn embedded(&self, defect: DraftDefect) -> Vec<f32> {
        let (eshape, etable) = self.weight("model.language_model.embed_tokens.weight");
        assert_eq!(eshape[1], self.dims.hidden, "{}: embed width", self.name);
        let ids = ints(&self.golden, "draft.block_ids");
        dflash::embed_block(ids, &etable, &self.dims, defect)
    }

    /// The forward under one defect, from the golden's captured inputs and the borrowed lm_head.
    fn run(&self, params: &DraftParams, defect: DraftDefect) -> DraftTrace {
        self.run_dims(&self.dims, params, defect)
    }

    /// [`Self::run`] under a PERTURBED config — the only way a gate can separate two dims
    /// fields that the fixture happens to give the same value.
    fn run_dims(&self, dims: &DraftDims, params: &DraftParams, defect: DraftDefect) -> DraftTrace {
        let (lm_shape, lm_head) = self.weight("lm_head.weight");
        assert_eq!(
            lm_shape[1], self.dims.hidden,
            "{}: lm_head width",
            self.name
        );
        let ctx = golden_read::float(&self.golden, "draft.context_concat").1;
        let noise = golden_read::float(&self.golden, "draft.noise_embeds").1;
        let io = DraftInputs {
            ctx_concat: ctx,
            noise,
            lm_head: &lm_head,
            vocab: lm_shape[0],
        };
        dflash::forward(dims, params, &io, defect)
    }

    fn params(&self) -> DraftParams {
        dflash::draw_params(&self.dims, &self.salt)
    }
}

/// `max|got-want| / max|want|` — the same metric the tolerance tables are stated in.
///
/// The GOT side is guarded here, BEFORE the fold, because `f64::max` returns the other
/// argument when one side is NaN: a fold over all-NaN differences comes back 0.0, scoring a
/// capture that computed nothing as a perfect match. That is this repo's oldest false green
/// (a broken kernel passed 9 of 9 that way, 2026-08-05) and it has been reintroduced twice
/// since — `crates/engine/tests/common/scoring.rs` records both. `Case::want` guards the
/// REFERENCE side; without this, the module header's promise covered only half the comparison.
/// A panic rather than `INFINITY`: an infinite rel would satisfy the defect matrix's
/// `rel > REDDEN` asserts, turning a broken oracle into evidence that a plant reddened.
fn worst_rel(got: &[f64], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "shape drift inside a named capture");
    assert!(
        got.iter().all(|g| g.is_finite()),
        "the oracle produced a non-finite value; no tolerance can be consulted on it"
    );
    let denom = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(g, w)| (g - f64::from(*w)).abs())
        .fold(0.0f64, f64::max);
    (worst / f64::from(denom)) as f32
}

/// Every Rel-scored capture of one trace, as `(capture name, operator, values)`.
///
/// `last_hidden` and `final_norm.out` are the same tensor captured twice by the reference
/// (the hook on `norm` and the model output); BOTH captures are scored — the census below is
/// against the golden's own contents, not against an opinion about redundancy — and both map
/// to the one `final_norm.out` tolerance row, because a tolerance is a property of the
/// operator, not of how many hooks the reference hung on it.
fn scored(tr: &DraftTrace) -> Vec<(String, &'static str, Vec<f64>)> {
    let mut out: Vec<(String, &'static str, Vec<f64>)> = vec![(
        "draft.encoder.out".into(),
        "encoder.out",
        tr.encoder_out.clone(),
    )];
    for (l, lt) in tr.layers.iter().enumerate() {
        let mut cap = |what: &'static str, vals: &Vec<f64>| {
            out.push((format!("draft.L{l}.{what}"), what, vals.clone()));
        };
        cap("input_layernorm.out", &lt.ln_in);
        cap("attend.q", &lt.q);
        cap("attend.k", &lt.k);
        cap("attend.v", &lt.v);
        cap("attend.out", &lt.attend_out);
        cap("post_attention_layernorm.out", &lt.ln_post);
        cap("mlp.out", &lt.mlp_out);
    }
    out.push((
        "draft.final_norm.out".into(),
        "final_norm.out",
        tr.final_norm.clone(),
    ));
    out.push((
        "draft.last_hidden".into(),
        "final_norm.out",
        tr.final_norm.clone(),
    ));
    out.push(("draft.logits".into(), "logits", tr.logits.clone()));
    out
}

// ------------------------------------------------------------------------------------------
// 1. The draws.
// ------------------------------------------------------------------------------------------

/// **Every tensor of both vendored weight sets regenerates bit for bit** from
/// `sha256(salt/name)` alone: 99 draws plus the 8 documented `L{l}.attn.gate_proj.weight`
/// aliases (stale names for the same tensor object, kept for `glimmer_gate.rs`) per salt.
///
/// Exactly ONE of the three draw families may reproduce a tensor — a second match would mean
/// the families are not distinguishable on these bytes and the classification below it is
/// vacuous. The counts are absolutes, not `> 0`: a re-vendor that drops tensors must fail
/// here, not shrink the census silently.
#[test]
fn the_salted_draws_regenerate_both_vendored_weight_sets_bit_for_bit() {
    for (wname, bytes) in WEIGHT_SETS {
        let w = GoldenSet::read_glimmer(&mut &bytes[..]).expect("weights parse");
        let salt = meta(&w, "salt").to_owned();
        let names: Vec<&str> = w.floats.iter().map(|(n, _, _)| n.as_str()).collect();
        let vals = |name: &str| golden_read::float(&w, name).1;
        let (mut drawn, mut aliases) = (0usize, 0usize);
        for name in &names {
            if let Some(rest) = name.strip_prefix('L') {
                let (l, _) = rest.split_once('.').expect("alias layer index");
                let modern = format!("model.language_model.layers.{l}.self_attn.gate_proj.weight");
                assert_eq!(vals(name), vals(&modern), "{wname}: alias {name} drifted");
                aliases += 1;
                continue;
            }
            let want = vals(name);
            let matches: Vec<Family> = [Family::Projection, Family::Norm, Family::CenteredNorm]
                .into_iter()
                .filter(|f| torchdraw::draw(&salt, name, want.len(), *f) == want)
                .collect();
            assert_eq!(
                matches.len(),
                1,
                "{wname}: {name} reproduced by {matches:?} families, not exactly one"
            );
            drawn += 1;
        }
        assert_eq!((drawn, aliases), (99, 8), "{wname}: census moved");
    }
}

/// The draw gate can go red: a seed off by one bit reproduces nothing, so the NAME → seed
/// keying above is load-bearing and not an accident of every draw looking alike.
///
/// The family knob has no arm here on purpose — the census above already requires exactly one
/// of three families to reproduce each of 99 tensors on both salts, which is strictly stronger
/// than "the wrong family fails on `lm_head.weight`". This is the half that census cannot
/// supply: it turns `draw`'s two inputs one at a time, and only the seed is left.
#[test]
fn the_draw_gate_reds_on_a_wrong_seed() {
    let w = GoldenSet::read_glimmer(&mut &WEIGHT_SETS[0].1[..]).expect("weights parse");
    let salt = meta(&w, "salt");
    let name = "lm_head.weight";
    let want = golden_read::float(&w, name).1;
    let seed = torchdraw::seed_for(&format!("{salt}/{name}"));
    assert_ne!(
        torchdraw::uniform(seed ^ 1, want.len(), Family::Projection),
        want,
        "a flipped seed bit still reproduced {name}"
    );
}

// ------------------------------------------------------------------------------------------
// 2. The clean forward.
// ------------------------------------------------------------------------------------------

/// **The oracle reproduces every captured value of both draft goldens** inside the measured
/// tolerances: 39 Rel-scored captures per salt, the 5 masks and the block embedding bit-exact,
/// and the candidate ids identical. The census is an absolute against the golden's own float
/// table: every capture is scored except `draft.context_concat` and `draft.noise_embeds`,
/// which are the forward's INPUTS (the embedding is scored by its own gate below).
#[test]
fn the_oracle_reproduces_every_captured_value_of_both_goldens() {
    for case in cases() {
        let tr = case.run(&case.params(), DraftDefect::None);
        let pairs = scored(&tr);
        assert_eq!(pairs.len(), 39, "{}: scored-capture census", case.name);
        // BOTH directions of the table, by the shared apparatus. `tol_for` alone panics on a
        // capture with no row and notices nothing about a ROW WITH NO CAPTURE — an unconsulted
        // threshold being a number that arrived from somewhere other than this oracle's
        // measurement, which is the whole failure that file's census exists to prevent.
        let consulted: Vec<&str> = pairs.iter().map(|(_, op, _)| *op).collect();
        tolerance::table_covers_exactly(DRAFT_ORACLE, &consulted);
        for (cap_name, operator, got) in &pairs {
            let want = case.want(cap_name);
            let rel = worst_rel(got, want);
            assert!(
                rel <= tol_for(operator),
                "{}: {cap_name} diverges by {rel:e} (tolerance {:e})",
                case.name,
                tol_for(operator)
            );
        }
        // The float-table census: nothing in the golden goes unscored but the two inputs.
        let scored_names: Vec<&str> = pairs.iter().map(|(n, _, _)| n.as_str()).collect();
        let mut masks = 0;
        for (name, _, _) in &case.golden.floats {
            if name.ends_with(".attend.mask") {
                masks += 1;
            } else if name != "draft.context_concat" && name != "draft.noise_embeds" {
                assert!(scored_names.contains(&name.as_str()), "{name} is unscored");
            }
        }
        assert_eq!(masks, case.dims.layers, "{}: mask census", case.name);
        // The INT table too, by NAME, or the headline claim above is about the float table
        // only. `block_ids` and `target_layer_ids` are read as inputs (here and in `dims_from`),
        // `candidates` is scored below, and `prompt.ids` is the driver's record of what it fed
        // the target — the one int this oracle consumes nowhere, named so that saying so is a
        // decision rather than an omission. Written 3 and measured 4 on first run, 2026-08-16.
        let int_names: Vec<&str> = case
            .golden
            .ints
            .iter()
            .map(|(n, _, _)| n.as_str())
            .collect();
        assert_eq!(
            int_names,
            [
                "prompt.ids",
                "target_layer_ids",
                "draft.block_ids",
                "draft.candidates"
            ],
            "{}: int census",
            case.name
        );
        // All five captures against the ONE mask the oracle builds. That is the stronger read
        // of the reference (which also builds one, outside its layer loop): five captures that
        // are not all equal to it is a defect a per-layer oracle mask would reproduce silently.
        for l in 0..case.dims.layers {
            let want_mask = case.want(&format!("draft.L{l}.attend.mask"));
            assert_eq!(tr.mask, want_mask, "{}: L{l} mask (exact)", case.name);
        }
        let want_cand = ints(&case.golden, "draft.candidates");
        assert_eq!(tr.candidates, want_cand, "{}: candidates", case.name);
    }
}

/// **THE BLOCK ATTENDS ITSELF, and both classes of pair are present.** This is the property the
/// whole fixture exists to make scorable, and it is asserted rather than assumed.
///
/// Until 2026-08-16 it was FALSE. At `sliding_window 4`, `block_size 4`, ctx 12 the mask
/// `|q_row - kv| <= w` let the furthest query reach kv 7 of 16, and the block-vs-block submatrix
/// summed to exactly **0.0** on both salts — no query ever attended the block, so §11 step 5
/// (attention is bidirectional ACROSS THE BLOCK) was pinned by the mask pattern and by no value,
/// and `DraftDefect::CausalMask`'s red was the re-selection of CONTEXT rows rather than the
/// property under test. The goldens were re-vendored at `sliding_window 13` — the value that
/// exercises both classes — and every floor and every defect signal in this file was re-measured
/// under it — the defect matrix's own table below records what moved, next to the constants
/// that assert it. **Four of the ten `weakest_defect` values were re-derived** (`attend.q`,
/// `attend.v`, `attend.out`, `logits`); the other six were NOT re-measured and are carried from
/// the old geometry. Five of those six have floors that DID move, so their captures changed and
/// their signals are stale by an unmeasured amount. Left standing because every one sits ~4
/// decades above its tolerance and `tolerances_leave_room`'s verdict cannot turn on it — but
/// they are inherited numbers until swept, and this sentence is the record that they are.
///
/// 13 of the 16 block-vs-block pairs attend and 3 are masked, so the window still BINDS inside
/// the block: an all-ones mask (w >= 15) would score green against a port that had no mask.
///
/// **What it costs, so nobody has to rediscover it.** At w >= 12 the CONTEXT half of the mask is
/// all ones, so this fixture no longer exercises window-masking of the context — asserted below
/// so the blind spot is declared rather than latent. The trade is FORCED: swept at ctx 12 /
/// block 4, context columns are masked only for w <= 10 and a strictly-bidirectional pair
/// appears only at w >= 13, and the two ranges never meet. That follows from the reference's own
/// mask form (`q_offset = 0` indexes queries by ROW while K/V spans `ctx + block`), so no
/// geometry with this mask has both at any ctx. §11 step 5 is the property under test, so the
/// block wins.
#[test]
fn the_block_attends_itself_and_the_window_still_binds() {
    for case in cases() {
        let d = &case.dims;
        let m = case.want("draft.L0.attend.mask");
        let kv_len = m.len() / d.block;
        let ctx = kv_len - d.block;
        assert_eq!(
            (d.block, d.window, ctx),
            (4, 13, 12),
            "{}: fixture widths — the geometry moved, so every floor and signal here is stale",
            case.name
        );
        let attended: usize = (0..d.block)
            .flat_map(|q| (ctx..kv_len).map(move |kv| (q, kv)))
            .filter(|(q, kv)| m[q * kv_len + kv] > 0.5)
            .count();
        assert_eq!(
            (attended, d.block * d.block - attended),
            (13, 3),
            "{}: block-vs-block attended/masked split",
            case.name
        );
        // The half a causal mask forbids: row 0 attending row 1 is bidirectionality itself, and
        // no count of attended pairs alone would distinguish it from a causal block.
        assert!(
            m[ctx + 1] > 0.5,
            "{}: block row 0 does not attend block row 1 — the mask is not bidirectional",
            case.name
        );
        // The declared blind spot, asserted so it stays declared: every context column is
        // attended by every query, so a defect in how the window masks CONTEXT is invisible
        // here. If this ever fails, the fixture gained context masking and the doc above —
        // which says that is impossible with this mask form — is what needs re-deriving.
        assert!(
            (0..d.block).all(|q| (0..ctx).all(|kv| m[q * kv_len + kv] > 0.5)),
            "{}: a context column is masked — re-derive the sweep in the doc above",
            case.name
        );
    }
}

/// **The mask reads `window`, not `block`.** Until the 2026-08-16 re-vendor the two were BOTH 4
/// — one field apart in [`DraftDims`] — and substituting either for the other scored green on
/// every value in this file; only this gate reddened, which is how the collapse was found. They
/// are now 13 and 4, so it is gone from the fixture, and the gate stays because it is what
/// proves that: widening the window by one must move the mask, which it cannot do if the mask is
/// reading `block`.
#[test]
fn the_mask_reads_the_window_and_not_the_block_size() {
    for case in cases() {
        let params = case.params();
        let clean = case.run(&params, DraftDefect::None);
        let wider = DraftDims {
            window: case.dims.window + 1,
            ..case.dims
        };
        let bad = case.run_dims(&wider, &params, DraftDefect::None);
        assert_ne!(
            bad.mask, clean.mask,
            "{}: the mask ignored `window`",
            case.name
        );
    }
}

/// **The borrowed embedding is read RAW, bit for bit**: gathering `draft.block_ids` from the
/// target's vendored `embed_tokens` matrix reproduces `draft.noise_embeds` exactly. The
/// reference reaches past its normed wrapper on purpose; the gather is a copy, so the right
/// tolerance is none at all — and [`DraftDefect::EmbedNormApplied`]'s red below is what keeps
/// this exactness a gate rather than an assumption.
#[test]
fn the_borrowed_embedding_reads_raw_and_bit_exact() {
    for case in cases() {
        let got = case.embedded(DraftDefect::None);
        assert_eq!(
            got,
            case.want("draft.noise_embeds"),
            "{}: raw gather",
            case.name
        );
    }
}

// ------------------------------------------------------------------------------------------
// 3. The defect matrix. Each planted defect must redden the comparison at the captures it
//    reaches AND hold — bit-identically, since the untouched code path is the same f64
//    computation — at the captures it does not.
// ------------------------------------------------------------------------------------------

/// One defect's expectations at layer 0, where reach is separable (from L1 on, the corrupted
/// residual stream reaches everything, which localises nothing).
struct Reach {
    defect: DraftDefect,
    /// Captures that must equal the clean trace EXACTLY at L0.
    holds: &'static [&'static str],
    /// Captures that must redden at L0, each with the signal MEASURED for it — min over the
    /// two salts, so every salt has to reach it. Carried as data, not prose: `REDDEN` alone is
    /// 20x to 2000x below every one of these, so a plant that decayed to a tenth of its
    /// strength would still clear it and the recorded numbers would quietly become decoration.
    reds: &'static [(&'static str, f32)],
    /// The same for the logits, which every defect reaches by construction.
    logits: f32,
}

/// ~208x the widest tolerance (4.80e-5, `attend.q`) and 21x under the smallest measured L0 signal
/// (2.146e-1, `attend.k` under the encoder-norm plant, salt 2); a defect that moves a capture
/// less than this is not the defect it claims to be.
const REDDEN: f32 = 1e-2;

/// The stages upstream of every attend-side plant: the hold set the causal-mask and
/// wrong-grouping defects share, since both corrupt only what the attention DOES with q/k/v.
const PRE_ATTEND: &[&str] = &[
    "encoder.out",
    "input_layernorm.out",
    "attend.q",
    "attend.k",
    "attend.v",
];

fn l0_values<'t>(tr: &'t DraftTrace, what: &str) -> &'t [f64] {
    let lt = &tr.layers[0];
    match what {
        "input_layernorm.out" => &lt.ln_in,
        "attend.q" => &lt.q,
        "attend.k" => &lt.k,
        "attend.v" => &lt.v,
        "attend.out" => &lt.attend_out,
        "encoder.out" => &tr.encoder_out,
        other => panic!("no L0 accessor for {other}"),
    }
}

fn assert_reach(case: &Case, params: &DraftParams, clean: &DraftTrace, r: &Reach) {
    let bad = case.run(params, r.defect);
    for what in r.holds {
        assert_eq!(
            l0_values(&bad, what),
            l0_values(clean, what),
            "{}: {:?} moved {what}, which it must not reach",
            case.name,
            r.defect
        );
    }
    // Both bars, every time: above REDDEN says the plant did SOMETHING, at-or-above its
    // recorded signal says it still does what it was measured doing. The 0.1% slack is for the
    // 4-s.f. rounding of the recorded number itself and nothing else.
    let signal = |what: &str, rel: f32, recorded: f32| {
        assert!(
            rel > REDDEN,
            "{}: {:?} left {what} at rel {rel:e} — the plant did not redden it",
            case.name,
            r.defect
        );
        assert!(
            rel >= recorded * 0.999,
            "{}: {:?} {what} rel {rel:e} is under its recorded {recorded:e} — the plant weakened",
            case.name,
            r.defect
        );
    };
    for (what, recorded) in r.reds {
        let cap = if *what == "encoder.out" {
            "draft.encoder.out".to_owned()
        } else {
            format!("draft.L0.{what}")
        };
        signal(
            what,
            worst_rel(l0_values(&bad, what), case.want(&cap)),
            *recorded,
        );
    }
    // Reach is never total: the logits must red (everything is upstream of them) while the
    // held set proves the defect did not simply corrupt the world.
    signal(
        "logits",
        worst_rel(&bad.logits, case.want("draft.logits")),
        r.logits,
    );
}

/// Traps 1 (causal mask), 2 (RoPE untailed) and the encoder-norm mirror, on both salts; trap
/// 4 (grouping) follows with its per-head half, and trap 3 (embed-norm) plants upstream of
/// the forward and has its own gate below. Measured L0 red-set signals, MIN of the two salts
/// (recorded 2026-08-16; the binding case, since the assert runs per salt):
///
/// | defect | reds at L0 (rel) | holds at L0 |
/// |---|---|---|
/// | CausalMask | attend.out 2.018e0, logits 1.075e0, mask ≠ | encoder, ln_in, q, k, v |
/// | RopeUntailed | attend.q 5.031e-1, attend.out 2.173e-1, logits 4.015e-1 | encoder, ln_in, k, v, mask |
/// | EncoderNormSkipped | encoder 9.832e-1, k 2.146e-1, v 1.307e0, out 1.201e0, logits 7.023e-1 | ln_in, q, mask |
///
/// **Re-measured 2026-08-16 at the re-vendored geometry** (`sliding_window` 4 → 13), and the
/// movement is the point: `CausalMask` went 1.471e0 → 2.018e0 at `attend.out` and 6.231e-1 →
/// 1.075e0 at the logits, because at the old window no query reached the block and making the
/// block causal could only re-select CONTEXT rows. The trap now costs what it is supposed to
/// cost. `RopeUntailed`'s logits nearly doubled (2.213e-1 → 4.015e-1) for the same reason.
///
/// **These are L0 figures, and `DRAFT_ORACLE`'s `weakest_defect` column is not.** That column
/// is the worst LAYER — the same bucket-level aggregation `--by-operator` makes for the text
/// tables — so the two differ on purpose (`attend.k` 2.146e-1 here against 4.632e-1 there; on a
/// row whose worst layer IS L0 the two coincide, e.g. `attend.q` at 5.031e-1 — coincidence, not
/// transcription).
/// Since 2026-08-16 the numbers in THIS table are data, asserted per salt by [`assert_reach`].
/// ("Neither is a transcription of the other" stood here until 2026-08-17, when the sweep showed
/// seven of `DRAFT_ORACLE`'s ten rows ARE these L0 figures — see that table's own note.)
#[test]
fn each_planted_defect_reddens_its_reach_and_holds_the_rest() {
    let reaches = [
        Reach {
            defect: DraftDefect::CausalMask,
            holds: PRE_ATTEND,
            reds: &[("attend.out", 2.018e0)],
            logits: 1.075e0,
        },
        Reach {
            defect: DraftDefect::RopeUntailed,
            holds: &["encoder.out", "input_layernorm.out", "attend.k", "attend.v"],
            reds: &[("attend.q", 5.031e-1), ("attend.out", 2.173e-1)],
            logits: 4.015e-1,
        },
        Reach {
            defect: DraftDefect::EncoderNormSkipped,
            holds: &["input_layernorm.out", "attend.q"],
            reds: &[
                ("encoder.out", 9.832e-1),
                ("attend.k", 2.146e-1),
                ("attend.v", 1.307e0),
                ("attend.out", 1.201e0),
            ],
            logits: 7.023e-1,
        },
    ];
    for case in cases() {
        let params = case.params();
        let clean = case.run(&params, DraftDefect::None);
        for r in &reaches {
            assert_reach(&case, &params, &clean, r);
        }
        // The causal plant must also move the MASK itself — that is the tensor it lives in —
        // while the two other forward defects leave every mask untouched.
        let causal = case.run(&params, DraftDefect::CausalMask);
        assert_ne!(causal.mask, clean.mask, "{}: causal mask", case.name);
        for d in [DraftDefect::RopeUntailed, DraftDefect::EncoderNormSkipped] {
            let bad = case.run(&params, d);
            assert_eq!(bad.mask, clean.mask, "{}: {d:?} mask", case.name);
        }
    }
}

/// **Trap 4's value half, head by head.** Reusing the target's Q:KV ratio (3 here, 16 real —
/// read from the TARGET golden's config, not hardcoded) moves exactly the heads whose KV
/// assignment changes (`h/2 != h/3`: heads 2, 4, 5) and holds the rest bit-identically
/// (heads 0, 1, 3). The vendored `assert_ne!` group-count test guards the shape half; this is
/// the divergence a port with legal shapes still produces — measured at L0, min of the salts:
/// attend.out 1.217e0, logits 1.022e0. Projections, norms, RoPE and mask are untouched by
/// construction, asserted via the same reach machinery.
#[test]
fn reusing_the_targets_grouping_moves_exactly_the_reassigned_heads() {
    let target_cfg = cfg(&load(anchor::text_goldens().next().expect("a text golden")));
    let wrong = anchor::widths(&target_cfg).group();
    for case in cases() {
        let d = &case.dims;
        let params = case.params();
        let clean = case.run(&params, DraftDefect::None);
        let bad = case.run(&params, DraftDefect::TargetGrouping { group: wrong });
        assert_reach(
            &case,
            &params,
            &clean,
            &Reach {
                defect: DraftDefect::TargetGrouping { group: wrong },
                holds: PRE_ATTEND,
                reds: &[("attend.out", 1.015e0)],
                logits: 1.180e0,
            },
        );
        let (hd, block) = (d.head_dim, d.block);
        let head = |tr: &DraftTrace, h: usize| -> Vec<f64> {
            (0..block)
                .flat_map(|r| tr.layers[0].attend_out[(r * d.heads + h) * hd..][..hd].to_vec())
                .collect()
        };
        let (mut moved, mut held) = (0usize, 0usize);
        for h in 0..d.heads {
            if h / d.group() == h / wrong {
                assert_eq!(
                    head(&bad, h),
                    head(&clean, h),
                    "{}: head {h} moved",
                    case.name
                );
                held += 1;
            } else {
                assert_ne!(
                    head(&bad, h),
                    head(&clean, h),
                    "{}: head {h} held",
                    case.name
                );
                moved += 1;
            }
        }
        // Both classes must occur, or the per-head claim above is quantified over nothing —
        // and `moved == 0` is also how a fixture whose two group counts had COLLAPSED would
        // present, which is why no separate `assert_ne!` on them is needed (the sibling
        // `glimmer_anchor_draft.rs` asserts the shape half from these same goldens).
        assert!(
            moved > 0 && held > 0,
            "{}: {moved} moved, {held} held — with {} vs {wrong} groups",
            case.name,
            d.group()
        );
    }
}

/// **Trap 3: the embed-norm applied to the borrowed embedding reddens the bit-exact gate.**
/// The other half — that the raw read matches exactly — is
/// [`the_borrowed_embedding_reads_raw_and_bit_exact`]; together they pin §11 step 3 from both
/// sides. Measured signal: rel 2.051e1 (min of the salts) — the weightless norm DIVIDES each
/// row by its rms, ~0.046 for a ±0.08 uniform draw over 72, so every row is scaled ~21x.
#[test]
fn applying_the_embed_norm_reddens_the_embedding_gate() {
    for case in cases() {
        let got = case.embedded(DraftDefect::EmbedNormApplied);
        let widened: Vec<f64> = got.iter().copied().map(f64::from).collect();
        let rel = worst_rel(&widened, case.want("draft.noise_embeds"));
        assert!(
            rel > REDDEN,
            "{}: the planted embed-norm moved the block by only {rel:e}",
            case.name
        );
    }
}
