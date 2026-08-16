//! Kimi-K3's tensor names, pinned against the checkpoint's own index — the census gate.
//!
//! Ported from `k3:tests/k3_names.rs`, bodies and comments travelling with it, and vendored
//! together with `docs/measurement/k3-reference/tensor-families.tsv` (2026-08-16): before
//! this file existed here, `quant/naming.rs`'s K3 block CITED this census while nothing in
//! this tree ran it — a stale claim of a gate, which is worse than no claim.
//!
//! **Why the census exists at all.** `K3Config`'s field spellings were taken from a prose
//! table and two of them were wrong in a way that would have refused every real checkpoint
//! on `missing field`. Tensor names are the same hazard with a worse failure mode: a config
//! key that does not exist refuses loudly, while a *tensor* name that does not exist looks
//! like a corrupt shard, and a name that exists but points at the wrong tensor repacks
//! silently. So every string in `quant/naming.rs`'s K3 block is checked against the vendored
//! reduction of the shipped `model.safetensors.index.json` (497,220 tensors, 96 shards,
//! revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`), vendored because the index itself
//! is 60 MB.
//!
//! **Where it lives.** The k3 tree keeps this as a top-level test binary; GLM/V4/Glimmer in
//! this tree have no index census yet ("the index-side gate arrives with the real-checkpoint
//! work" — `v4_convert.rs`'s header), so there is no convention to fold into. It sits beside
//! `k3_config.rs` in this crate's tests because the strings it pins are this crate's, and
//! K3 is the one model whose index reduction is already vendored.
//!
//! The TSV is data, not documentation: `#`-comments, then `count \t dtype \t shape \t
//! family`, where a family is a name with `.layers.<n>.`, `.experts.<n>.` and `.blocks.<n>.`
//! collapsed to `{L}`, `{E}`, `{B}`. Nothing here parses the 60 MB file; the header comment
//! carries the source sha256 to regenerate the reduction from.
//!
//! No GPU, no network — 7 KB of vendored TSV and 6.8 KB of vendored JSON.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::quant::{
    K3_PACKED, K3_PROJ, K3_SCALE, K3_TEXT_PREFIX, f4_expert_bytes, f4_groups, f4_row_bytes,
    k3_expert_base,
};

const FAMILIES: &str = include_str!("../../../docs/measurement/k3-reference/tensor-families.tsv");

/// One row of the vendored reduction.
struct TsvFamily {
    count: usize,
    dtype: String,
    /// Empty when the row's shape is `?` — a family none of the fetched shard headers
    /// covered, recorded as UNKNOWN rather than as absent, so callers must opt in.
    shape: Vec<usize>,
    name: String,
}

/// Parse the vendored `tensor-families.tsv`. Dimensions are `x`-separated.
fn families() -> Vec<TsvFamily> {
    FAMILIES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 4, "malformed row: {l:?}");
            TsvFamily {
                count: f[0].parse().expect("count"),
                dtype: f[1].to_string(),
                shape: match f[2] {
                    "?" => Vec::new(),
                    s => s.split('x').map(|d| d.parse().expect("dim")).collect(),
                },
                name: f[3].to_string(),
            }
        })
        .collect()
}

/// `(hidden, expert_in, moe_inter)` read from the **vendored `config.json`**, the same file
/// `k3_config.rs` pins the schema against.
///
/// So the assertions below relate three things — the config, the tensor shapes in the index,
/// and this engine's geometry functions — rather than relating literals to themselves. Two
/// assertions in the k3 tree's first draft were `assert_eq!(<literal>, <literal>)`, which
/// cannot fail; reading the dims from the file is what fixed them there and is kept here.
fn shipped_dims() -> (usize, usize, usize) {
    let v: serde_json::Value = serde_json::from_str(include_str!(
        "../../../docs/measurement/k3-reference/config.json"
    ))
    .expect("the vendored config must parse");
    let t = &v["text_config"];
    let get = |k: &str| {
        t[k].as_u64()
            .unwrap_or_else(|| panic!("text_config.{k} is not an integer")) as usize
    };
    (
        get("hidden_size"),
        get("routed_expert_hidden_size"),
        get("moe_intermediate_size"),
    )
}

fn find<'a>(fams: &'a [TsvFamily], name: &str) -> &'a TsvFamily {
    fams.iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("no such family in the shipped index: {name}"))
}

/// A count the TSV's own header declares, taken as the integer immediately before `phrase`.
///
/// **Why the header rather than a constant here.** These numbers used to live twice in the
/// k3 tree — as literals in the test and as prose in the file's header — two frozen copies
/// agreeing with each other, so a typo in either was invisible and the header was
/// decoration. Reading them out of the artifact leaves ONE copy and makes the header
/// load-bearing. Panics if the phrase moves rather than defaulting, because a header reword
/// must be a visible edit and not a check that quietly stops checking.
fn declared(phrase: &str) -> usize {
    let at = FAMILIES.find(phrase).unwrap_or_else(|| {
        panic!("the TSV header no longer says `{phrase}`; this test reads its counts from there")
    });
    let digits: String = FAMILIES[..at]
        .trim_end()
        .chars()
        .rev()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits
        .chars()
        .rev()
        .collect::<String>()
        .parse()
        .unwrap_or_else(|e| panic!("no number before `{phrase}` in the TSV header: {e}"))
}

/// The reduction itself must be the file we think it is. A truncated or hand-edited TSV
/// would make every assertion below vacuous, which is the shape a fixture-backed gate fails
/// in.
///
/// Each count is compared against what the header DECLARES, so the rows and the provenance
/// line have to agree. What this still cannot see is an upstream revision that ADDS a
/// family: both sides of every comparison here are frozen together, a limitation the TSV's
/// own header states under OPEN along with what closing it would take.
#[test]
fn the_vendored_reduction_is_intact() {
    let fams = families();
    let (want_fams, want_text) = (declared("families total"), declared("text-side"));
    assert_eq!(
        fams.len(),
        want_fams,
        "the header declares {want_fams} families; the rows are {}",
        fams.len()
    );
    // Tensors, from the counts alone — so a dropped row cannot pass unnoticed.
    let total: usize = fams.iter().map(|f| f.count).sum();
    assert_eq!(
        total,
        declared("tensors across"),
        "family counts must sum to the tensor count the header declares"
    );
    // The vision side is a SIBLING of `language_model`, so the split is a prefix test.
    let (text, other): (Vec<&TsvFamily>, Vec<&TsvFamily>) = fams
        .iter()
        .partition(|f| f.name.starts_with("language_model."));
    assert_eq!(
        (text.len(), other.len()),
        (want_text, want_fams - want_text),
        "text-side / vision-side families, against the header's {want_fams} total and \
         {want_text} text-side"
    );
    for f in other {
        assert!(
            f.name.starts_with("vision_tower.") || f.name.starts_with("mm_projector."),
            "unexpected non-text family {}, which the converter would silently include",
            f.name
        );
    }
}

/// **Exactly these families have no shape, and no others — so the blind spot cannot widen.**
///
/// A `?` shape means no fetched shard header covered that family (the index carries neither
/// dtype nor shape; those come from HTTP Range reads of three shards). Recorded as unknown
/// rather than as absent, which is right — but it is a HOLE in every shape assertion in
/// this file, and without this list an unbounded one: a re-reduction whose Range fetch
/// failed for a shard would turn more rows into `?` and quietly widen it, with every
/// remaining assertion still green. Enumerating means losing coverage is an edit.
///
/// **Five of the seventeen are IN SCOPE and therefore have no shape checked anywhere**: the
/// embedding, the head, the final norm, and the two halves of the model-level attn-res
/// fold. Their dims are derivable from the vendored `config.json`, and this deliberately
/// does NOT assert a derivation — the header's claim is that this file is "the checkpoint's
/// own index, not a transliteration", and asserting an expectation against no ground truth
/// is exactly the transliteration it rules out. Fetching those shard headers is what closes
/// it.
#[test]
fn exactly_the_declared_families_have_no_shape() {
    const NO_SHAPE: [&str; 17] = [
        "language_model.lm_head.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight",
        "mm_projector.post_norm.weight",
        "mm_projector.proj.0.weight",
        "mm_projector.proj.2.weight",
        "vision_tower.encoder.blocks.{B}.mlp.fc0.weight",
        "vision_tower.encoder.blocks.{B}.mlp.fc1.weight",
        "vision_tower.encoder.blocks.{B}.norm0.weight",
        "vision_tower.encoder.blocks.{B}.norm1.weight",
        "vision_tower.encoder.blocks.{B}.wo.weight",
        "vision_tower.encoder.blocks.{B}.wqkv.weight",
        "vision_tower.encoder.final_layernorm.weight",
        "vision_tower.patch_embed.pos_emb.weight",
        "vision_tower.patch_embed.proj.weight",
    ];
    let fams = families();
    let mut got: Vec<&str> = fams
        .iter()
        .filter(|f| f.shape.is_empty())
        .map(|f| f.name.as_str())
        .collect();
    got.sort_unstable();
    let mut want: Vec<&str> = NO_SHAPE.to_vec();
    want.sort_unstable();
    assert_eq!(
        got, want,
        "the set of shape-unknown families changed. MORE of them means a shard header went \
         unfetched and the shape checks below cover less than they did — re-fetch rather \
         than updating this list. FEWER means someone filled one in, which is progress: \
         update the list and add the shape assertion that is now possible."
    );
    // Derived from the same list, not typed twice, so this number cannot drift from it.
    let in_scope = want
        .iter()
        .filter(|n| n.starts_with("language_model."))
        .count();
    assert_eq!(
        in_scope, 5,
        "in-scope families with no shape checked anywhere — this is the residual gap, and \
         it shrinks only by fetching shard headers"
    );
}

/// Every string in `quant/naming.rs`'s K3 block, against the index.
///
/// This is the assertion that would have caught a name taken from the C reference's loader:
/// `layers.{L}.block_sparse_moe...` without the `language_model.model.` prefix matches
/// NOTHING, and a converter built on it finds zero tensors and blames the checkpoint.
#[test]
fn the_k3_names_are_the_checkpoints_own() {
    let fams = families();
    // The prefix, tested as a property of every text family rather than as a literal.
    for f in fams
        .iter()
        .filter(|f| f.name.starts_with("language_model."))
    {
        assert!(
            f.name.starts_with(K3_TEXT_PREFIX) || f.name == "language_model.lm_head.weight",
            "{} does not carry K3_TEXT_PREFIX — the only text-side exception is lm_head, \
             which sits beside `model.` rather than under it",
            f.name
        );
    }
    // The six expert tensors, composed **the way `F4Expert::spans` composes them** —
    // `{base}.{proj}` then the two suffixes — so this pins the string shape a conversion
    // actually runs against. (The k3 tree once asserted a helper's output against the
    // constants the helper had just concatenated: a guard unable to fire, deleted there.)
    for proj in K3_PROJ {
        let base = format!("{}.{proj}", k3_expert_base(7, 42));
        for n in [format!("{base}.{K3_PACKED}"), format!("{base}.{K3_SCALE}")] {
            let fam = n
                .replace("layers.7.", "layers.{L}.")
                .replace("experts.42.", "experts.{E}.");
            let got = find(&fams, &fam);
            // 92 MoE layers x 896 experts. Read off the TSV, and it is the one count that
            // proves these families cover the dense layer's ABSENCE as well as the MoE
            // layers' presence: 93 layers would be 83,328.
            assert_eq!(got.count, 92 * 896, "{fam}");
            assert_eq!(
                got.dtype, "U8",
                "{fam}: MXFP4 nibbles and e8m0 scales are both U8"
            );
        }
    }
    assert_eq!(
        k3_expert_base(3, 0),
        "language_model.model.layers.3.block_sparse_moe.experts.0"
    );

    // The trunk-side MoE tensors the converter must NOT put in `.f4`, each confirming a
    // fact the plan states: the latent sandwich's two projections, the aggregate norm, and
    // a router that scores on FULL width. Shapes are `[out, in]`.
    let moe = |t: &str| format!("language_model.model.layers.{{L}}.block_sparse_moe.{t}");
    for (tensor, want) in [
        ("routed_expert_down_proj.weight", vec![3584, 7168]),
        ("routed_expert_up_proj.weight", vec![7168, 3584]),
        ("routed_expert_norm.weight", vec![3584]),
        ("gate.weight", vec![896, 7168]),
    ] {
        let f = find(&fams, &moe(tensor));
        assert_eq!(f.shape, want, "{tensor}");
        assert_eq!(
            f.count, 92,
            "{tensor} is per MoE layer, and layer 0 is dense"
        );
        assert_eq!(f.dtype, "BF16", "{tensor} is trunk-side and NOT quantized");
    }
    // The selection bias — F32 where its neighbours are BF16, worth pinning because
    // `noaux_tc` reads it at selection only and a converter that widened or skipped it
    // would move which experts run.
    let bias = find(&fams, &moe("gate.e_score_correction_bias"));
    assert_eq!((bias.count, bias.dtype.as_str()), (92, "F32"));
    assert_eq!(bias.shape, vec![896]);
    // The shared expert: ONE fused MLP per layer at FULL width, which is why `.f4` has no
    // shared block. Its down projection is `[hidden, 2 x moe_inter]` — **against the
    // vendored CONFIG's dims, not against literals**: this form fails if the config and
    // the tensor shapes ever disagree, which is the fact worth holding.
    let (hidden, _, moe_inter) = shipped_dims();
    let sh = find(&fams, &moe("shared_experts.down_proj.weight"));
    assert_eq!(
        (sh.count, sh.shape.clone(), sh.dtype.as_str()),
        (92, vec![hidden, 2 * moe_inter], "BF16"),
        "the fused shared MLP is two experts' worth of intermediate at full width"
    );
}

/// **The repack is a copy, and this is the arithmetic that says so.**
///
/// rivoli's `.f4` projection is `o_dim` packed rows of `f4_row_bytes(i_dim)` followed by
/// `o_dim x f4_groups(i_dim)` e8m0 bytes. The checkpoint ships exactly that: `[o_dim,
/// i_dim/2]` U8 nibbles and `[o_dim, i_dim/32]` U8 scales, low-nibble-even
/// (`k3:docs/reference/k3-architecture.md` §9), group 32. So converting a K3 expert is two
/// `copy_from_slice`s per projection — no transposition, no re-blocking, no dequantise
/// step, and no arithmetic to get wrong.
///
/// Checked against the shard header's own shapes rather than against the plan's byte
/// figure, and it reproduces that figure (17,547,264 B/expert) as a consequence.
#[test]
fn the_shipped_expert_layout_is_already_rivolis() {
    let fams = families();
    // **From the vendored config, not hardcoded** — the widths under test are `expert_in`
    // (the 3584 latent, NOT `hidden_size`) and `moe_inter`, and reading them from the file
    // is what makes this test relate the config to the shipped shapes instead of restating
    // both.
    let (hidden, expert_in, moe_inter) = shipped_dims();
    assert_ne!(
        expert_in, hidden,
        "routed_expert_hidden_size == hidden_size: the latent this whole stage exists for \
         is gone"
    );
    // Slot order is gate, up, down — so the first two are entered at the latent and the
    // third at the intermediate. Getting this pairing wrong is the `w2`-in-the-wrong-slot
    // case.
    let widths = [
        (K3_PROJ[0], moe_inter, expert_in),
        (K3_PROJ[1], moe_inter, expert_in),
        (K3_PROJ[2], expert_in, moe_inter),
    ];
    let mut total = 0;
    for (proj, o_dim, i_dim) in widths {
        let fam = |t: &str| {
            format!("language_model.model.layers.{{L}}.block_sparse_moe.experts.{{E}}.{proj}.{t}")
        };
        // The nibble span: the checkpoint's second dim IS `f4_row_bytes`.
        assert_eq!(
            find(&fams, &fam(K3_PACKED)).shape,
            vec![o_dim, f4_row_bytes(i_dim)],
            "{proj}.{K3_PACKED} is not [o_dim, f4_row_bytes(i_dim)] — the nibbles are not \
             packed along the input dim and the repack is NOT a copy"
        );
        // The scale span: one e8m0 byte per group of `F4_GROUP` along the same dim.
        assert_eq!(
            find(&fams, &fam(K3_SCALE)).shape,
            vec![o_dim, f4_groups(i_dim)],
            "{proj}.{K3_SCALE} is not [o_dim, f4_groups(i_dim)]"
        );
        total += o_dim * f4_row_bytes(i_dim) + o_dim * f4_groups(i_dim);
    }
    // No `assert_eq!(F4_GROUP, 32)`: the scale-shape assertion above already pins it. The
    // TSV gives `w1.weight_scale` as `3072x112` against an input dim of 3584, and
    // `div_ceil(3584, g) = 112` holds for `g = 32` alone — so the shape assertion fires
    // first on any change to `F4_GROUP`, and a restated constant would sit behind it.
    //
    // ...and the whole expert, against the geometry the artifact writer uses.
    assert_eq!(total, f4_expert_bytes(expert_in, moe_inter));
    assert_eq!(
        total, 17_547_264,
        "the k3 plan's G0 item 5a byte figure, re-derived from the shapes"
    );
}
