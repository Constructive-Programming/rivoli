//! **The DRAFT goldens: the shapes that make the DFlash drafter a drafter rather than a small
//! target.**
//!
//! One of the four binaries the Muse Glimmer S1b anchor gate is split across — `glimmer_anchor.rs`
//! carries the framing and the byte pins, `glimmer_anchor_common/mod.rs` the tables and accessors.
//! Every way a port goes wrong here is a way of REUSING the target's attention path, and each of
//! those is visible as a shape: Q over the block alone against K/V over `context + block`, a context
//! that enters through `encoder.fc` at `len(target_layer_ids) * hidden`, two norms per layer instead
//! of four, no vocabulary of its own, and a GQA group count that is not the target's.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#[path = "glimmer_anchor_common/mod.rs"]
mod anchor; // keep this preamble blank-line-free: spread out, the four are a jscpd clone
use anchor::{
    GoldenSet, Vendored, Widths, cfg, draft_goldens, ints, load, meta_usize, num, real, shape_is,
    shape_of, text_goldens, widths,
};
use serde_json::Value;

/// The drafter's own two lengths, alongside the four widths its config implies.
#[derive(Clone, Copy)]
struct Draft {
    w: Widths,
    block: usize,
    ctx: usize,
}

/// The captures taken once per draft file: the encoder path, the block, and the logits it borrows
/// the target's vocabulary for.
fn check_draft_toplevel_shapes(v: &Vendored, g: &GoldenSet, c: &Value, d: Draft) {
    // The context is one column block per `target_layer_id`.
    let targets = ints(g, "target_layer_ids").len();
    shape_is(g, "draft.context_concat", &[1, d.ctx, targets * d.w.hidden]);
    shape_is(g, "draft.encoder.out", &[1, d.ctx, d.w.hidden]);
    shape_is(g, "draft.noise_embeds", &[1, d.block, d.w.hidden]);
    assert_eq!(
        ints(g, "draft.block_ids").len(),
        d.block,
        "{}: block ids",
        v.name
    );
    // One anchor token plus masks in, `block - 1` candidates out: index 0 is sliced off.
    assert_eq!(
        ints(g, "draft.candidates").len(),
        d.block - 1,
        "{}: candidates",
        v.name
    );
    // **The drafter's own config has no `vocab_size`**, because it owns neither the embedding
    // nor the lm_head — it borrows the target's (section 11). So the logit width has to come
    // from the TARGET's config, and asserting the absence is asserting the borrow.
    assert!(
        c["vocab_size"].is_null(),
        "{}: the drafter has acquired a vocab of its own",
        v.name
    );
    let vocab = num(&cfg(&load(text_goldens().next().unwrap())), "vocab_size");
    shape_is(g, "draft.logits", &[1, d.block, vocab]);
}

/// One drafter layer. `glimmer-architecture.md` §11: Q comes from the block alone while K/V span
/// `context + block` inside one call — a way a port goes wrong by reusing the target's attention
/// path, and a shape.
fn check_draft_layer_shapes(g: &GoldenSet, p: &str, d: Draft) {
    let (w, block, ctx) = (d.w, d.block, d.ctx);
    shape_is(
        g,
        &format!("{p}.attend.q"),
        &[1, w.heads, block, w.head_dim],
    );
    for what in ["attend.k", "attend.v"] {
        let name = format!("{p}.{what}");
        assert_eq!(
            shape_of(g, &name),
            vec![1, w.kv, ctx + block, w.head_dim],
            "{name}: K/V must span context+block while Q spans block alone"
        );
    }
    shape_is(g, &format!("{p}.attend.mask"), &[1, 1, block, ctx + block]);
    // Two norms per layer, not four: the drafter is plain pre-norm and has no post-FFN norm.
    for what in ["input_layernorm", "post_attention_layernorm", "mlp"] {
        shape_is(g, &format!("{p}.{what}.out"), &[1, block, w.hidden]);
    }
    let post_ffn = format!("{p}.post_feedforward_layernorm.out");
    assert!(
        g.floats.iter().all(|(n, _, _)| n != &post_ffn),
        "{p}: the drafter has no post-FFN norm; a capture for one means it was built as a target \
         layer"
    );
}

/// **The DFlash golden has the shapes that make the drafter a drafter, not a small target.**
///
/// `glimmer-architecture.md` §11: Q comes from the block alone while K/V span `context + block`
/// inside one call, the context enters through `encoder.fc` at `len(target_layer_ids) * hidden`,
/// and the KV group count is not the target's. Each of those is a way a port goes wrong by reusing
/// the target's attention path, and each is a shape.
#[test]
fn the_draft_golden_has_the_shapes_that_make_it_a_drafter() {
    for v in draft_goldens() {
        let g = load(v);
        let c = cfg(&g);
        let d = Draft {
            w: widths(&c),
            block: meta_usize(&g, "block_size"),
            ctx: meta_usize(&g, "context_len"),
        };
        check_draft_toplevel_shapes(v, &g, &c, d);
        for l in 0..num(&c, "num_hidden_layers") {
            check_draft_layer_shapes(&g, &format!("draft.L{l}"), d);
        }
    }
}

/// **The drafter's attention shape differs from the target's**, which is the property that makes a
/// port reusing the target's path fail rather than silently pass.
#[test]
fn the_drafter_does_not_share_the_targets_attention_shape() {
    let target = widths(&cfg(&load(text_goldens().next().unwrap())));
    let drafter = widths(&cfg(&load(draft_goldens().next().unwrap())));
    assert_ne!(
        target.group(),
        drafter.group(),
        "the two GQA group counts are equal, so a port that reuses the target's shape passes here \
         and fails on the real 16:1 against 4:1"
    );
    assert_eq!(
        target.hidden, drafter.hidden,
        "the drafter borrows the target's embedding and lm_head, so the widths must match"
    );
    // The real pairing, from the vendored config, so the tiny ratio is not the only evidence.
    let real = real();
    assert_eq!(
        num(&real, "num_attention_heads") / num(&real, "num_key_value_heads"),
        16
    );
}
