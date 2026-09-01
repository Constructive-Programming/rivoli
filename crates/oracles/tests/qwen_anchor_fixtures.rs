//! **The operator fixtures S3's kernels will be scored against, and the two censuses that keep
//! them from being satisfied by an empty set.**
//!
//! The other half of `qwen_anchor.rs`, split from it because one file carrying both came to 1290
//! lines against this repo's 1200-line hard cap. That file owns provenance and the byte pins; this
//! one owns what is IN the goldens — names, shapes, values, the captured-layer census, and the
//! defect matrix's declaration.
//!
//! Shapes are asserted, not just names. A `[1, 9, 20, 20]` recurrent state reshaped to
//! `[1, 9, 400]` carries the same numbers and a different meaning, which is exactly the fail-open
//! `golden::diff` refuses for the same reason. Every dimension is derived from the golden's own
//! recorded widths.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[path = "common/qwen_anchor_read.rs"]
mod read;

use read::{
    BUCKET_ORDER, DECODE, GoldenSet, bucket_of, capture_census, decay_rates_are_not_all_equal,
    float, int_shape, ints, load, logits_are_finite_and_spread, meta_json, names,
    one_row_of_logits, shape_of, top_k_is_in_range_and_distinct, width,
};

// ---------------------------------------------------------------------------------------------
// Shapes.

/// The operator fixtures S3 needs, named and shaped.
#[test]
fn the_operator_fixtures_s3_needs_are_present() {
    for v in DECODE {
        let g = load(v);
        let (c, w) = (meta_json(&g, "tiny_config"), meta_json(&g, "widths"));
        gdn_fixtures_are_shaped(&g, &w);
        indexer_fixtures_are_shaped(&g, &w);
        hyper_connection_fixtures_are_shaped(&g, &w);
        moe_fixtures_are_shaped(&g, &w);
        // **There is no `model.norm`.** The 4-to-1 hyper-connection collapse IS this model's only
        // final norm (MOD:1330/1430) and the checkpoint carries no such tensor, which is the one
        // structural fact about the tail a port is most likely to add by habit. Asserted as an
        // ABSENCE, because nothing else in this file can notice a tensor that should not exist.
        assert!(
            !names(&g).any(|n| n == "model.norm"),
            "this model has no final `model.norm`; the hc collapse is the norm"
        );
        one_row_of_logits(&g, &c);
        // The 4-to-1 collapse, which stands where a final norm would.
        let hidden = width(&w, "hidden_size");
        assert_eq!(
            shape_of(&g, "model.hyper_connection_mixer"),
            vec![1, 1, hidden]
        );
    }
}

/// The gated delta rule, on layer 0 — the first `linear_attention` layer and the first module the
/// model runs any real arithmetic in.
///
/// The state is `[1, heads_v, d_k, d_v]` and it is **square at both the tiny widths (20) and the
/// real ones (128)**, so no shape assertion can see which axis the buffer puts first. That is a
/// hazard the width audit's `equal` list deliberately CARRIES rather than hides, and
/// `gdn_state_axis_swapped` is what prices it: as with K3's KDA state, the layout has to be settled
/// by scoring both readings against the reference OUTPUT, never by a shape check.
fn gdn_fixtures_are_shaped(g: &GoldenSet, w: &Value) {
    let (nv, dk, dv) = (
        width(w, "linear_num_value_heads"),
        width(w, "linear_key_head_dim"),
        width(w, "linear_value_head_dim"),
    );
    let op = "model.layers.0.linear_attn.gated_delta_rule";
    for leg in ["query", "key", "value"] {
        assert_eq!(shape_of(g, &format!("{op}.in.{leg}")), vec![1, 1, nv, dk]);
    }
    for leg in ["g", "beta"] {
        assert_eq!(shape_of(g, &format!("{op}.in.{leg}")), vec![1, 1, nv]);
    }
    for leg in ["A_log", "dt_bias"] {
        assert_eq!(shape_of(g, &format!("{op}.in.{leg}")), vec![nv]);
    }
    let state = vec![1, nv, dk, dv];
    assert_eq!(shape_of(g, &format!("{op}.in.initial_state")), state);
    assert_eq!(shape_of(g, &format!("{op}.out.state")), state);
    assert_eq!(shape_of(g, &format!("{op}.out.o")), vec![1, 1, nv, dk]);
    gdn_projection_fixtures_are_shaped(g, w, nv, dv);
}

/// The fused input projections and the short convolution, which sit UPSTREAM of the recurrence.
///
/// The conv weight is captured because the reference hands `conv1d.weight.squeeze(1)` to the
/// module-level `causal_conv1d_fn` (MOD:477-496) and never calls the `nn.Conv1d` — so a forward
/// hook on that module fires ZERO times, which is the shape of the bug K3's anchor carried for a
/// day with three comments claiming otherwise.
fn gdn_projection_fixtures_are_shaped(g: &GoldenSet, w: &Value, nv: usize, dv: usize) {
    let at = "model.layers.0.linear_attn";
    // `conv_dim` is `2*key_dim + value_dim` and is NOT `value_dim` — a pair the audit separates,
    // and the reason is that the conv runs over q, k and v while `in_proj_z` covers only v.
    let conv = width(w, "conv_dim");
    assert_eq!(
        conv,
        2 * width(w, "key_dim") + width(w, "value_dim"),
        "conv_dim must be the fused q|k|v width"
    );
    assert_eq!(shape_of(g, &format!("{at}.in_proj_qkv")), vec![1, 1, conv]);
    assert_eq!(
        shape_of(g, &format!("{at}.in_proj_z")),
        vec![1, 1, width(w, "value_dim")]
    );
    for leg in ["in_proj_a", "in_proj_b"] {
        assert_eq!(shape_of(g, &format!("{at}.{leg}")), vec![1, 1, nv]);
    }
    let kernel = shape_of(g, &format!("{at}.conv.weight"));
    assert_eq!(kernel[0], conv, "the conv weight is [conv_dim, kernel]");
    assert_eq!(shape_of(g, &format!("{at}.conv.in")), vec![1, conv, 1]);
    assert_eq!(shape_of(g, &format!("{at}.conv.out")), vec![1, conv, 1]);
    // The output norm-gate is `[heads_v, d_v]` — per head, not flattened.
    assert_eq!(shape_of(g, &format!("{at}.norm")), vec![nv, dv]);
}

/// The QSA indexer on layer 3, the first `qwen_sparse_attention` layer.
///
/// At the decode position there are 17 visible tokens and `indexer_compress_ratio` 4, so there are
/// **4 complete blocks** against a `block_topk` of 2 — the position is in the SELECTIVE regime,
/// which is what makes every indexer defect row non-vacuous here. That is asserted rather than
/// described, from the recorded widths and the captured shapes.
fn indexer_fixtures_are_shaped(g: &GoldenSet, w: &Value) {
    let ix = "model.layers.3.self_attn.indexer";
    let (heads, dim) = (4usize, width(w, "indexer_head_dim"));
    assert_eq!(
        shape_of(g, &format!("{ix}.index_qk_proj")),
        vec![1, 1, width(w, "index_qk_width")]
    );
    assert_eq!(
        shape_of(g, &format!("{ix}.q_layernorm")),
        vec![1, 1, heads, dim]
    );
    let blocks = shape_of(g, &format!("{ix}.pooled_keys"))[0];
    assert_eq!(shape_of(g, &format!("{ix}.k_layernorm")), vec![blocks, dim]);
    assert_eq!(shape_of(g, &format!("{ix}.scores")), vec![blocks]);
    assert_eq!(int_shape(g, &format!("{ix}.block_starts")), [blocks]);
    // 16 warm tokens plus the decoded one, over `compress_ratio` — derived, so a window change
    // cannot leave this passing on a stale literal.
    let visible = 17;
    let ratio = 4;
    assert_eq!(
        blocks,
        visible / ratio,
        "complete blocks at the decode position"
    );
    assert!(
        blocks > width(w, "block_topk"),
        "{blocks} blocks against a budget of {} — the decode position must be SELECTIVE or every \
         indexer defect row is vacuous here",
        width(w, "block_topk")
    );
    // The mask the module returns, over the 17 visible positions.
    assert_eq!(shape_of(g, ix), vec![1, 1, 1, visible]);
    // The attention's own fused q/gate projection: `heads * head_dim * 2`, the per-head
    // `[query|gate]` interleave `attn_gate_block_split` prices.
    assert_eq!(
        shape_of(g, "model.layers.3.self_attn.q_proj"),
        vec![1, 1, width(w, "q_proj_width")]
    );
}

/// The two hyper-connection folds, on OPPOSITE sides of the attention and the MoE (MOD:1222 and
/// MOD:1238). Separate buckets and separate fixtures: a single `hc` bucket would put a post-MLP
/// tensor upstream of the MoE in the comparator's ordering, and every router row's localisation
/// claim would then fail on its own downstream fold.
fn hyper_connection_fixtures_are_shaped(g: &GoldenSet, w: &Value) {
    let (hidden, stack, low) = (
        width(w, "hidden_size"),
        width(w, "hc_width"),
        width(w, "hc_lowrank"),
    );
    assert_eq!(
        stack,
        4 * hidden,
        "the hc stream stack is hc_count * hidden"
    );
    for fold in ["attn_hyper_connection", "mlp_hyper_connection"] {
        let at = format!("model.layers.0.{fold}");
        assert_eq!(shape_of(g, &format!("{at}.hc_norm")), vec![1, 1, stack]);
        // The low-rank mix, down then up. `hc_lowrank` is hidden/8 in the real config and is kept
        // at hidden/8 here, while staying distinct from the GDN key head width.
        assert_eq!(
            shape_of(g, &format!("{at}.input_mix_weight_down")),
            vec![1, 1, low]
        );
        assert_eq!(
            shape_of(g, &format!("{at}.input_mix_weight_up")),
            vec![1, 1, stack]
        );
        assert_eq!(
            shape_of(g, &format!("{at}.block_inject_weight")),
            vec![1, 1, 4]
        );
        // Output 0 is the collapsed hidden the block consumes; 1 is the updated stream stack.
        assert_eq!(shape_of(g, &format!("{at}.0")), vec![1, 1, hidden]);
        assert_eq!(shape_of(g, &format!("{at}.1")), vec![1, 1, stack]);
    }
}

/// The MoE: the shared expert (which runs BEFORE the router, MOD:928-930), the router's three
/// outputs, and the routed result.
fn moe_fixtures_are_shaped(g: &GoldenSet, w: &Value) {
    let (hidden, inter) = (width(w, "hidden_size"), width(w, "moe_intermediate_size"));
    let at = "model.layers.0.mlp";
    for leg in ["gate_proj", "up_proj"] {
        assert_eq!(
            shape_of(g, &format!("{at}.shared_expert.{leg}")),
            vec![1, inter]
        );
    }
    assert_eq!(
        shape_of(g, &format!("{at}.shared_expert.down_proj")),
        vec![1, hidden]
    );
    assert_eq!(shape_of(g, &format!("{at}.shared_expert_gate")), vec![1, 1]);
    let (experts, topk) = (width(w, "num_experts"), width(w, "num_experts_per_tok"));
    assert_eq!(shape_of(g, &format!("{at}.gate.0")), vec![1, experts]);
    assert_eq!(shape_of(g, &format!("{at}.gate.1")), vec![1, topk]);
    assert_eq!(int_shape(g, &format!("{at}.gate.2")), [1, topk]);
    assert_eq!(shape_of(g, &format!("{at}.experts")), vec![1, hidden]);
    assert_eq!(shape_of(g, at), vec![1, 1, hidden]);
}

// ---------------------------------------------------------------------------------------------
// Values.

/// The numbers are numbers, not noise — **and not degenerate**.
///
/// A few hundred float tensors and no value-level assertion means a golden of zeros, of NaN, or
/// drawn on wrong scales passes every shape test above. The degeneracy half is the other reason two
/// salts exist: a fixture whose second routed expert carries a vanishing weight, or whose `beta`
/// saturates the delta-rule gate, masks any bug in the arithmetic it gates — and it would look like
/// a perfectly ordinary golden.
#[test]
fn the_captured_values_are_on_their_declared_scales_and_not_degenerate() {
    for v in DECODE {
        let g = load(v);
        let w = meta_json(&g, "widths");
        gdn_draws_are_on_scale(&g, v.name);
        logits_are_finite_and_spread(&g, v.name);
        routing_selected_distinct_experts(&g, &w, v.name);
        indexer_scores_separate_at_the_budget_cut(&g, &w, v.name);
    }
}

/// The GDN draws and the state, whose behaviour the recurrence depends on.
///
/// `A_log` is the per-head decay rate: a CONSTANT one makes every head decay identically and a
/// kernel that ignored the term entirely would still match, so "not all equal" is asserted on top
/// of finiteness. `beta` reaches the recurrence PRE-sigmoid, so a draw far out in either tail would
/// pin the delta-rule update at 0 or 1 and hide whatever the update does; ±8 is where `sigmoid` is
/// within 4e-4 of its limits.
fn gdn_draws_are_on_scale(g: &GoldenSet, name: &str) {
    let op = "model.layers.0.linear_attn.gated_delta_rule";
    let (_, a_log) = float(g, &format!("{op}.in.A_log"));
    assert!(
        a_log.iter().all(|x| x.is_finite()),
        "{name}: A_log must be finite"
    );
    decay_rates_are_not_all_equal(a_log, name);
    let (_, beta) = float(g, &format!("{op}.in.beta"));
    assert!(
        beta.iter().all(|x| x.abs() < 8.0),
        "{name}: a beta saturates the delta-rule gate: {beta:?}"
    );
    // The state goes in non-zero (it is the warm prefill's output) and comes out CHANGED. A kernel
    // that produced the right `o` from an unchanged state agrees for exactly one token, which is
    // precisely how long this fixture runs — so both halves are asserted.
    let (_, s_in) = float(g, &format!("{op}.in.initial_state"));
    let (_, s_out) = float(g, &format!("{op}.out.state"));
    assert!(
        s_in.iter().any(|x| *x != 0.0),
        "{name}: the warm prefill left an all-zero state; the recurrence is unscoreable"
    );
    assert!(
        s_in.iter()
            .zip(s_out)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "{name}: the decode step did not change the recurrent state"
    );
}

/// Routing, as ints and as weights.
///
/// Distinctness is the real check on the indices — `[0, 0]` is a pair top-k cannot produce, and a
/// bound of `0..num_experts` alone would accept it.
///
/// The WEIGHTS carry two claims. Their sum pins `norm_topk_prob`, which is a config-class DEFAULT
/// absent from the checkpoint's own file (trap T25) and therefore exactly the renormalisation a
/// port reading only that file would omit. And no weight may vanish: an expert weighted at ~0
/// contributes nothing, so a bug in that expert's arithmetic would be invisible here.
fn routing_selected_distinct_experts(g: &GoldenSet, w: &Value, name: &str) {
    let at = "model.layers.0.mlp.gate";
    let (topk, experts) = (width(w, "num_experts_per_tok"), width(w, "num_experts"));
    top_k_is_in_range_and_distinct(ints(g, &format!("{at}.2")), topk, experts, name);
    let (_, weights) = float(g, &format!("{at}.1"));
    let sum: f32 = weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "{name}: top-k weights sum to {sum}, not the renormalised 1.0 that `norm_topk_prob` — a \
         config-class default, absent from the checkpoint file — turns on"
    );
    let biggest = weights.iter().fold(0f32, |m, x| m.max(*x));
    assert!(
        weights.iter().all(|x| *x >= biggest * 0.05),
        "{name}: a routed weight is degenerate ({weights:?}) — that expert is unscoreable"
    );
}

/// **The indexer's score at the BUDGET CUT must separate, and that is a narrower claim than "the
/// scores are distinct" — because the reference's relu makes exact ties normal.**
///
/// Why the scores are captured at all: three defect rows perturbed them without moving the top-2
/// and left the whole bucket bit-identical when only the final MASK was recorded — argmax
/// invariance, the same shape as this tree's "ids moved by exactly 0.000e0" finding. Capturing the
/// scores is half the fix; the other half is that the selection must be attributable to the
/// arithmetic rather than to a sort, which is what this checks.
///
/// > **Measured 2026-08-31, and it CORRECTED this assertion.** The first version demanded all four
/// > block scores distinct. It went RED at `qwen-anchor-2`, whose scores are
/// > `[0.0, 0.0, 0.712, 1.244]` — and the tie is not a degenerate draw. The per-block score is
/// > `relu(score).sum(dim=-1) / sqrt(d)`, so **a block whose every raw score is negative scores
/// > EXACTLY 0.0**, and two such blocks tie exactly. At an untrained draw that is common; at the
/// > real `block_topk` of 512 over thousands of blocks it will be common too.
/// >
/// > **That leaves a trap this anchor DEMONSTRATES and does not price, recorded rather than
/// > closed.** When the budget cut falls INSIDE a tie, which blocks get selected is decided by
/// > `torch.topk`'s tie-break, not by the model — so a port with a different tie-break diverges on
/// > a fixture where every captured score is bit-identical. No defect row covers it (perturbing a
/// > tie-break is not a perturbation of the reference's arithmetic), and this fixture cannot,
/// > because its own cut is clean. `anchor.md` carries it as OPEN for S3.
///
/// So the claim is: the score at rank `block_topk - 1` is strictly above the one at rank
/// `block_topk`, by a margin that is a real gap rather than a rounding accident. Ties BELOW the cut
/// are among blocks nobody selects and change nothing.
fn indexer_scores_separate_at_the_budget_cut(g: &GoldenSet, w: &Value, name: &str) {
    let (_, scores) = float(g, "model.layers.3.self_attn.indexer.scores");
    assert!(
        scores.iter().all(|x| x.is_finite()),
        "{name}: indexer scores must be finite"
    );
    let topk = width(w, "block_topk");
    assert!(
        scores.len() > topk,
        "{name}: {} blocks against a budget of {topk} — the selection is not a choice",
        scores.len()
    );
    let mut sorted: Vec<f32> = scores.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).expect("finite"));
    // Not every block relu'd to zero: that would make the whole selection a tie-break.
    assert!(
        sorted[0] > 0.0,
        "{name}: every block scored 0.0 — the relu erased the whole comparison"
    );
    let span = sorted[0] - sorted[sorted.len() - 1];
    let cut = sorted[topk - 1] - sorted[topk];
    assert!(
        cut > span * 1e-3,
        "{name}: the score margin at the budget cut is {cut} against a total span of {span} \
         ({scores:?}) — too tight for the selection to be attributable to the arithmetic rather \
         than to `topk`'s tie-break"
    );
}

// ---------------------------------------------------------------------------------------------
// The capture census and the defect matrix's two columns.

/// Every captured layer is present in every golden, and no others.
///
/// The exact set matters twice: `capture_layers` is the metadata a reader trusts, and K3's driver
/// once matched `model.layers.1` as a prefix and captured layers 1 and 10-19 (and 3 with 30-39) —
/// 25 layers instead of 6, invisible in every per-tensor assertion and obvious here.
///
/// The seven are chosen for what each IS: 0 the first GDN layer and first MoE; 1 the PLE HOST
/// (`ple_layer_ids: [2]` is one-indexed); 2 the GDN layer immediately downstream of the injection,
/// which gives every PLE-side defect a captured green boundary at layer 0 and a captured red at
/// layer 1; 3 the first QSA layer; 27 a mid-pattern QSA layer; 46 and 47 the last GDN and last QSA,
/// the pair the real 48-layer pattern ends with.
#[test]
fn exactly_the_declared_layers_were_captured() {
    for v in DECODE {
        capture_census(&load(v), &[0, 1, 2, 3, 27, 46, 47], v.name);
    }
}

/// **The defect matrix's census, and the anti-vacuity check on its two COLUMNS.**
///
/// The matrix itself is the python gate (`qwen-anchor.sh`, one `--compare` per row per salt, whose
/// `_gate_first_touch` refuses a row that changed nothing, a row whose own declared bucket stayed
/// bit-identical, or a row that reddened anything UPSTREAM of it). Its verdicts live in
/// `anchor.md`. What is checkable deviceless, from the bytes, is that the DECLARATION the gate
/// scores is complete and that neither of its two columns can be satisfied by an empty set:
///
/// * **the census** — every class's rows have a `first_touch` entry and every `first_touch` key
///   belongs to exactly one class, so a row cannot be scored without being classified or classified
///   without being scored;
/// * **the RED column** — every declared first-touched bucket is actually CAPTURED in this golden.
///   An uncaptured bucket would score green through a default and the whole localisation claim
///   would rest on nothing, which is the fail-open K3's review found;
/// * **the HOLD column** — for every row, the buckets strictly upstream of its first-touched one
///   are counted. A row with an empty upstream set has no hold column at all, and the rows in that
///   position are asserted to be EXACTLY those whose declared bucket is the globally-first captured
///   bucket — an equivalence, so the exception is derived rather than listed and cannot quietly
///   grow. The count with a real hold column is pinned, so it cannot quietly shrink either.
#[test]
fn the_defect_census_covers_every_class_and_both_columns() {
    for v in DECODE {
        let g = load(v);
        let classes = meta_json(&g, "defect_classes");
        let touch = meta_json(&g, "first_touch");
        assert_eq!(
            census_is_a_partition(&classes, &touch),
            38,
            "{}: defect rows",
            v.name
        );
        assert_eq!(
            classes.as_object().expect("classes").len(),
            13,
            "{}: defect classes",
            v.name
        );
        the_nine_plan_classes_each_carry_two_rows(&classes);
        both_columns_are_non_empty(&g, &touch);
    }
}

/// **The port plan's actual exit rule: `>= 2` rows for each of the NINE named classes, not `>= 18`
/// rows total.**
///
/// The distinction is the whole point of the rule and it is recorded in `qwen-architecture.md`'s own
/// dated note: the trap table met the count and not the rule, with 5/5/5/2/2 across four classes and
/// one-or-zero across the rest, so a matrix built off it would have read complete while carrying
/// **zero** coverage of the GDN decay form, of where the epsilon lives, and of the router's gate/up
/// order. A total-only assertion here would reproduce exactly that hole.
///
/// The names are the plan's, spelled as the driver's `CLASSES` keys spell them, so a class silently
/// renamed on the generating side fails here rather than being counted under a name nobody checks.
/// The other four classes this matrix carries — norm offset, fused-QKV segmentation, attention gate
/// split, state axis order — are **not** on the plan's list, so their single rows are not a
/// violation; each is a defect this tree was bitten by on another port, which is why they exist.
fn the_nine_plan_classes_each_carry_two_rows(classes: &Value) {
    for class in [
        "gdn decay form",
        "conv tap order",
        "output-gate activation",
        "indexer relu/pool/budget",
        "hc transpose + lowrank order",
        "router bias/norm/w1w3",
        "n-gram seed/order/layer",
        "partial-rope width",
        "eps homes",
    ] {
        let rows = classes[class].as_array().unwrap_or_else(|| {
            panic!("no class named {class:?} — the plan names it, so it exists")
        });
        assert!(
            rows.len() >= 2,
            "{class} carries {} defect row(s); the plan's rule is >= 2 PER CLASS, and one row \
             cannot show that a defect's localisation is a property of the arithmetic",
            rows.len()
        );
    }
}

/// Every classified row is declared and every declared row is classified, exactly once. Returns the
/// row count so the caller can pin it.
fn census_is_a_partition(classes: &Value, touch: &Value) -> usize {
    let declared: BTreeSet<&str> = touch
        .as_object()
        .expect("first_touch")
        .keys()
        .map(String::as_str)
        .collect();
    let mut classified: BTreeMap<&str, usize> = BTreeMap::new();
    for (class, rows) in classes.as_object().expect("defect_classes") {
        for row in rows.as_array().expect("class rows") {
            let row = row.as_str().expect("a defect name");
            assert!(
                declared.contains(row),
                "{row} is in class {class} with no first_touch entry — it would be scored by the \
                 matrix without a localisation claim"
            );
            *classified.entry(row).or_default() += 1;
        }
    }
    for (row, n) in &classified {
        assert_eq!(*n, 1, "{row} is in {n} classes; the classes must partition");
    }
    let unclassified: Vec<&str> = declared
        .iter()
        .filter(|r| !classified.contains_key(*r))
        .copied()
        .collect();
    assert!(
        unclassified.is_empty(),
        "declared but in no class: {unclassified:?} — a perturbation nobody decided what it prices"
    );
    classified.len()
}

/// Both columns of every row, scored against the buckets this golden actually holds.
fn both_columns_are_non_empty(g: &GoldenSet, touch: &Value) {
    let held = buckets_held(g);
    let first = *held.first().expect("the golden holds captured buckets");
    let mut without_hold: Vec<(&str, (usize, usize))> = Vec::new();
    for (defect, at) in touch.as_object().expect("first_touch") {
        let layer = at[0].as_u64().expect("a layer index") as usize;
        let bucket = at[1].as_str().expect("a bucket name");
        let rank = BUCKET_ORDER
            .iter()
            .position(|b| *b == bucket)
            .unwrap_or_else(|| panic!("{defect} declares unknown bucket {bucket}"));
        assert!(
            held.contains(&(layer, rank)),
            "{defect} declares ({layer}, {bucket}) as its first touched bucket, and this golden \
             does not capture it — an uncaptured bucket is not evidence of localisation"
        );
        if held.iter().all(|b| *b >= (layer, rank)) {
            without_hold.push((defect, (layer, rank)));
        }
    }
    // The equivalence: a row has no hold column exactly when its bucket is the FIRST one captured.
    for (defect, at) in &without_hold {
        assert_eq!(
            *at, first,
            "{defect} has an empty hold column without being the first captured bucket — the \
             matrix's upstream-green half is vacuous for it and nothing says so"
        );
    }
    // The six are the hyper-connection and eps rows, which first touch layer 0's `hc_attn` — the
    // first module of the first layer, with `ngram_hash` and `ple_inject` absent there because the
    // PLE host is layer 1. Their reddening half still has content; only their hold half is empty,
    // and that is a property of where the model starts rather than of the matrix.
    assert_eq!(
        touch.as_object().expect("first_touch").len() - without_hold.len(),
        32,
        "rows with a non-empty hold column; the {} without are the ones at {}",
        without_hold.len(),
        BUCKET_ORDER[first.1]
    );
}

/// Every `(layer, bucket-rank)` this golden holds a tensor in, sorted into execution order.
fn buckets_held(g: &GoldenSet) -> Vec<(usize, usize)> {
    let mut held: Vec<(usize, usize)> = names(g).filter_map(bucket_of).collect();
    held.sort_unstable();
    held.dedup();
    held
}
