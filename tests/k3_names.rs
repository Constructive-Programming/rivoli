//! Kimi-K3's tensor names, pinned against the checkpoint's own index.
//!
//! **Why this file exists.** `K3Config`'s field spellings were taken from a prose table and two
//! of them were wrong in a way that would have refused every real checkpoint on `missing field`
//! (`model.rs`, 2026-08-10). Tensor names are the same hazard with a worse failure mode: a config
//! key that does not exist refuses loudly, while a *tensor* name that does not exist can look
//! like a corrupt shard, and a name that exists but points at the wrong tensor repacks silently.
//!
//! So every string in `quant.rs`'s K3 naming block is checked here against
//! `docs/measurement/k3-reference/tensor-families.tsv` — a reduction of the shipped
//! `model.safetensors.index.json` (497,220 tensors, 96 shards, revision
//! `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`), vendored because the index itself is 60 MB.
//!
//! The TSV is data, not documentation: `#`-comments, then `count \t dtype \t shape \t family`,
//! where a family is a name with `.layers.<n>.`, `.experts.<n>.` and `.blocks.<n>.` collapsed to
//! `{L}`, `{E}`, `{B}`. Nothing here parses the 60 MB file; if the vendored reduction is ever
//! regenerated, the header comment carries the source sha256 to regenerate it from.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::artifact::quant::{
    K3_PACKED, K3_PROJ, K3_SCALE, K3_TEXT_PREFIX, f4_expert_bytes, f4_groups, f4_row_bytes,
    k3_expert_base,
};

const FAMILIES: &str = include_str!("../docs/measurement/k3-reference/tensor-families.tsv");

/// One row of the vendored reduction.
struct Family {
    count: usize,
    dtype: String,
    shape: Vec<usize>,
    name: String,
}

fn families() -> Vec<Family> {
    FAMILIES
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 4, "malformed row: {l:?}");
            Family {
                count: f[0].parse().expect("count"),
                dtype: f[1].to_string(),
                // `?` marks a family none of the three fetched shard headers covered — recorded
                // as unknown rather than as absent. An empty vec, so callers must opt in.
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
/// `model.rs`'s `k3_base_matches_the_shipped_config` pins the schema against.
///
/// Added 2026-08-11 so the assertions below relate three things — the config, the tensor shapes in
/// the index, and this engine's geometry functions — rather than relating literals to themselves.
/// Two assertions here were `assert_eq!(<literal>, <literal>)` before review, which cannot fail.
fn shipped_dims() -> (usize, usize, usize) {
    let v: serde_json::Value =
        serde_json::from_str(include_str!("../docs/measurement/k3-reference/config.json"))
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

fn find<'a>(fams: &'a [Family], name: &str) -> &'a Family {
    fams.iter()
        .find(|f| f.name == name)
        .unwrap_or_else(|| panic!("no such family in the shipped index: {name}"))
}

/// The reduction itself must be the file we think it is. A truncated or hand-edited TSV would
/// make every assertion below vacuous, which is the shape a fixture-backed gate fails in.
#[test]
fn the_vendored_reduction_is_intact() {
    let fams = families();
    assert_eq!(fams.len(), 60, "60 families were reduced from the index");
    // 497,220 tensors, from the counts alone — so a dropped row cannot pass unnoticed.
    let total: usize = fams.iter().map(|f| f.count).sum();
    assert_eq!(
        total, 497_220,
        "family counts must sum to the index's tensor count"
    );
    // The vision side is a SIBLING of `language_model`, so the split is a prefix test.
    let (text, other): (Vec<&Family>, Vec<&Family>) = fams
        .iter()
        .partition(|f| f.name.starts_with("language_model."));
    assert_eq!(
        (text.len(), other.len()),
        (48, 12),
        "text-side / vision-side families"
    );
    for f in other {
        assert!(
            f.name.starts_with("vision_tower.") || f.name.starts_with("mm_projector."),
            "unexpected non-text family {}, which the converter would silently include",
            f.name
        );
    }
}

/// Every string in `quant.rs`'s K3 naming block, against the index.
///
/// This is the assertion that would have caught a name taken from the C reference's loader:
/// `layers.{L}.block_sparse_moe...` without the `language_model.model.` prefix matches NOTHING,
/// and a converter built on it finds zero tensors and blames the checkpoint.
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
    // The six expert tensors, composed **the way `F4Expert::spans` composes them** — `{base}.{proj}`
    // then the two suffixes — so this pins the string shape a conversion actually runs against.
    // It used to call a `k3_expert_proj` helper that nothing in `src/` used, and assert that its
    // output ended with the same constants it had just concatenated: a guard unable to fire, on a
    // path the converter never takes. Helper deleted, review 2026-08-11.
    for proj in K3_PROJ {
        let base = format!("{}.{proj}", k3_expert_base(7, 42));
        // Substitute the two indices back out and the result must BE a shipped family.
        for n in [format!("{base}.{K3_PACKED}"), format!("{base}.{K3_SCALE}")] {
            let fam = n
                .replace("layers.7.", "layers.{L}.")
                .replace("experts.42.", "experts.{E}.");
            let got = find(&fams, &fam);
            // 92 MoE layers x 896 experts. Read off the TSV, and it is the one count that proves
            // these families cover the dense layer's ABSENCE as well as the MoE layers' presence:
            // 93 layers would be 83,328.
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

    // The trunk-side MoE tensors the converter must NOT put in `.f4`, each confirming a fact the
    // plan states: the latent sandwich's two projections, the aggregate norm, and a router that
    // scores on FULL width. Shapes are `[out, in]`.
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
    // The shared expert: ONE fused MLP per layer at FULL width, which is why `.f4` has no shared
    // block. Its down projection is `[hidden, 2 x moe_inter]` — **against the vendored CONFIG's
    // dims, not against literals**. `assert_eq!(6144, 2 * 3072)` stood here until review
    // 2026-08-11 pointed out it is a compile-time identity over constants that reads nothing:
    // this form fails if the config and the tensor shapes ever disagree, which is the fact worth
    // holding.
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
/// rivoli's `.f4` projection is `o_dim` packed rows of `f4_row_bytes(i_dim)` followed by `o_dim x
/// f4_groups(i_dim)` e8m0 bytes. The checkpoint ships exactly that: `[o_dim, i_dim/2]` U8 nibbles
/// and `[o_dim, i_dim/32]` U8 scales, low-nibble-even (`k3-architecture.md` §9), group 32. So
/// converting a K3 expert is two `copy_from_slice`s per projection — no transposition, no
/// re-blocking, no dequantise step, and no arithmetic to get wrong.
///
/// Checked against the shard header's own shapes rather than against the plan's byte figure, and
/// it reproduces that figure (17,547,264 B/expert) as a consequence.
#[test]
fn the_shipped_expert_layout_is_already_rivolis() {
    let fams = families();
    // **From the vendored config, not hardcoded** — the widths under test are `expert_in` (the
    // 3584 latent, NOT `hidden_size`) and `moe_inter`, and reading them from the file is what
    // makes this test relate the config to the shipped shapes instead of restating both.
    let (hidden, expert_in, moe_inter) = shipped_dims();
    assert_ne!(
        expert_in, hidden,
        "routed_expert_hidden_size == hidden_size: the latent this whole stage exists for is gone"
    );
    // Slot order is gate, up, down — so the first two are entered at the latent and the third at
    // the intermediate. Getting this pairing wrong is the `w2`-in-the-wrong-slot case.
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
    // No `assert_eq!(F4_GROUP, 32)`: the scale-shape assertion above already pins it. The TSV gives
    // `w1.weight_scale` as `3072x112` against an input dim of 3584, and `div_ceil(3584, g) = 112`
    // holds for `g = 32` alone (`g` must be `>= 32` and `< 32.29`). So the shape assertion fires
    // first on any change to `F4_GROUP`, and this one would be a constant restated behind it.
    //
    // ...and the whole expert, against the geometry the artifact writer uses.
    assert_eq!(total, f4_expert_bytes(expert_in, moe_inter));
    assert_eq!(
        total, 17_547_264,
        "G0 item 5a's byte figure, re-derived from the shapes"
    );
}
