//! Integration check: the DeepSeek-V4-Flash checkpoint's real per-layer tensor sets agree
//! with what `V4Config` says they should be.
//!
//! **This is the implication `convert_v4` bets on.** `write_layer_resident` decides from the
//! CONFIG — `layer_has_compressor`, `layer_has_indexer`, `layer_routes_by_hash` — which
//! tensor groups to copy, rather than probing the checkpoint with `has()`. That is the right
//! way round (a layer whose tensors disagree with `num_hash_layers` must fail, not silently
//! take whichever branch the file happens to satisfy) but it means an error in
//! `compress_ratios` becomes a converter that drops a whole tensor group for one layer and
//! says nothing. `--verify` cannot see it: it only checks the routed experts.
//!
//! The artifact produced during S1a covers layers 0-2 only — a ratio-0, a ratio-0 and a
//! ratio-4 layer. The ratio-128 shape and the non-hash `ffn.gate.bias` branch are therefore
//! exercised by NO artifact, which is exactly why this walks all 43 layers of the index
//! instead of the artifact.
//!
//! Skips (loudly) when the checkpoint is absent, like `tests/artifact.rs` does for GLM.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::artifact::model::{V4Config, load_config};
use std::collections::HashSet;

/// The shipped checkpoint. Overridable so this is not pinned to one machine's layout.
fn checkpoint() -> Option<String> {
    let dir = std::env::var("RIVOLI_V4_SRC")
        .unwrap_or_else(|_| "/var/db/rivoli/deepseek-v4-flash-0731".into());
    if std::fs::metadata(format!("{dir}/model.safetensors.index.json")).is_ok() {
        return Some(dir);
    }
    // Printed, not silent: a green line for a test that opened nothing is the failure mode
    // this repo has already been bitten by.
    eprintln!("SKIP v4_artifact: no checkpoint at {dir} (set RIVOLI_V4_SRC)");
    None
}

/// Every tensor name in the index, which needs no shard to be present — so this runs
/// against a partially-downloaded checkpoint too.
fn index_names(dir: &str) -> HashSet<String> {
    let v: serde_json::Value = serde_json::from_slice(
        &std::fs::read(format!("{dir}/model.safetensors.index.json")).unwrap(),
    )
    .unwrap();
    v["weight_map"]
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect()
}

/// For all 43 layers: the tensor groups present must be exactly the ones the config's
/// `compress_ratios` and `num_hash_layers` imply — in BOTH directions. Present-when-not-
/// expected matters as much as absent-when-expected: it would mean the converter is
/// silently dropping weights the model uses.
#[test]
fn per_layer_tensor_groups_match_the_config() {
    let Some(dir) = checkpoint() else { return };
    let cfg: V4Config = load_config(&dir).unwrap();
    let names = index_names(&dir);

    let (mut compressors, mut indexers, mut hashed) = (0, 0, 0);
    for l in 0..cfg.n_layers {
        let has = |suffix: &str| names.contains(&format!("layers.{l}.{suffix}"));

        // Present on every layer, whatever the ratio — if one of these is missing the
        // "expected" sets below would be trivially satisfiable.
        for t in [
            "attn.wq_a.weight",
            "attn.wq_b.weight",
            "attn.wkv.weight",
            "attn.wo_a.weight",
            "attn.wo_b.weight",
            "attn.attn_sink",
            "attn_norm.weight",
            "ffn_norm.weight",
            "ffn.gate.weight",
            "ffn.shared_experts.w1.weight",
            "hc_attn_fn",
            "hc_ffn_fn",
        ] {
            assert!(has(t), "layer {l}: {t} missing — every layer must have it");
        }

        let want_compressor = cfg.layer_has_compressor(l).unwrap();
        assert_eq!(
            has("attn.compressor.wkv.weight"),
            want_compressor,
            "layer {l} (ratio {}): compressor presence disagrees with the config",
            cfg.compress_ratio(l).unwrap()
        );
        let want_indexer = cfg.layer_has_indexer(l).unwrap();
        for t in [
            "attn.indexer.wq_b.weight",
            "attn.indexer.weights_proj.weight",
            "attn.indexer.compressor.wkv.weight",
        ] {
            assert_eq!(
                has(t),
                want_indexer,
                "layer {l} (ratio {}): {t} presence disagrees with the config",
                cfg.compress_ratio(l).unwrap()
            );
        }
        // A hash layer routes from `tid2eid` and has NO bias; a scored layer is the exact
        // reverse. Both halves asserted, because "has a table" and "has no bias" are
        // independently wrong-able and the converter's `if/else` assumes both.
        let want_hash = cfg.layer_routes_by_hash(l);
        assert_eq!(has("ffn.gate.tid2eid"), want_hash, "layer {l}: tid2eid");
        assert_eq!(has("ffn.gate.bias"), !want_hash, "layer {l}: gate bias");

        // Every routed expert exists, and expert `n_experts` does NOT — the boundary
        // `.f4` relies on when it writes exactly `n_experts` blocks and no shared one.
        assert!(has(&format!("ffn.experts.{}.w1.weight", cfg.n_experts - 1)));
        assert!(!has(&format!("ffn.experts.{}.w1.weight", cfg.n_experts)));

        compressors += usize::from(want_compressor);
        indexers += usize::from(want_indexer);
        hashed += usize::from(want_hash);
    }

    // The counts `other-models.md` and the port plan quote, now measured rather than
    // asserted from the config alone: 41 of 43 layers compress, 21 of those index.
    assert_eq!((compressors, indexers, hashed), (41, 21, 3));
}

/// `V4Config`'s dimensions against the tensors they describe, on one layer of each shape.
/// The converter checks these per layer at convert time, but only over the range it is
/// asked for — the S1a artifact is layers 0-2, so nothing has ever confronted the config
/// with a ratio-128 layer's tensors.
#[test]
fn config_dims_match_the_tensor_shapes() {
    let Some(dir) = checkpoint() else { return };
    let cfg: V4Config = load_config(&dir).unwrap();
    // Shapes need the shard headers, not just the index, so skip a layer whose shard has
    // not been downloaded rather than failing on it.
    let Ok(st) = rivoli::artifact::format::Safetensors::open_dir(&dir) else {
        eprintln!("SKIP config_dims: shards not all present");
        return;
    };
    // ratio 0 (no compressor), ratio 4 (compressor + indexer), ratio 128 (compressor only).
    for l in [0, 2, 3] {
        let want = [
            (
                format!("layers.{l}.attn.wq_a.weight"),
                vec![cfg.q_lora_rank, cfg.hidden],
            ),
            (
                format!("layers.{l}.attn.wq_b.weight"),
                vec![cfg.n_heads * cfg.head_dim, cfg.q_lora_rank],
            ),
            (
                format!("layers.{l}.attn.wkv.weight"),
                vec![cfg.head_dim, cfg.hidden],
            ),
            (format!("layers.{l}.attn.attn_sink"), vec![cfg.n_heads]),
            (
                format!("layers.{l}.ffn.gate.weight"),
                vec![cfg.n_experts, cfg.hidden],
            ),
            (
                format!("layers.{l}.hc_attn_fn"),
                vec![24, cfg.hc_mult * cfg.hidden],
            ),
        ];
        for (name, want) in want {
            assert_eq!(st.shape(&name).unwrap(), &want[..], "{name}");
        }
        if cfg.layer_has_indexer(l).unwrap() {
            assert_eq!(
                st.shape(&format!("layers.{l}.attn.indexer.wq_b.weight"))
                    .unwrap(),
                &[cfg.index_n_heads * cfg.index_head_dim, cfg.q_lora_rank]
            );
            assert_eq!(
                st.shape(&format!("layers.{l}.attn.indexer.weights_proj.weight"))
                    .unwrap(),
                &[cfg.index_n_heads, cfg.hidden]
            );
        }
    }
}
