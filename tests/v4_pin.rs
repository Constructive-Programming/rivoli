//! `V4Pin::build` against the shipped `.f4` artifact.
//!
//! **Separate from `tests/v4_loading.rs` because this one needs the GPU.** Placement goes
//! through `DeviceTier`, which allocates a real device slab and refuses to start if another
//! tenant holds GPU memory — so this file follows the repo's GPU-test idiom (`tests/vk.rs`)
//! and simply runs, failing when the device is busy. `v4_loading.rs` stays host-only.
//!
//! What it is for: `V4Pin::build` resolves ~40 tensors a layer by NAME into typed fields,
//! and a name resolved into the wrong field is silent — every one of these tensors exists,
//! so nothing fails to load. The dimensions are the discriminant. `wq_a` is `[1024, 4096]`
//! and `wkv` is `[512, 4096]`; `wo_a` is `[8192, 4096]` and `wo_b` its transpose. Swapping
//! any of those pairs changes an `o_dim` that the config predicts independently, which is
//! what the assertions below compare against.
//!
//! **What this CANNOT see, stated rather than implied:**
//!   * `w1` and `w3` (gate and up) have identical shapes, so a swap between them is
//!     dimensionally invisible here — as `quant::V4_PROJ`'s doc says, only a numerical
//!     oracle can catch it, which is what `src/v4oracle/` exists for.
//!   * The shipped fixture is layers 0-2 and `num_hash_layers` is 3, so **every layer it
//!     holds is hash-routed**: the `V4Route::Scored` arm, the `ffn.gate.bias` extent check,
//!     and the ratio-128 compressor-without-indexer shape are constructed by NO artifact on
//!     this machine. `tests/v4_artifact.rs` records the same gap for the converter and walks
//!     the checkpoint index instead. Closing it needs an artifact covering a layer >= 3.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(any(feature = "rocm", feature = "vulkan"))]

use rivoli::artifact::model::{V4Config, load_config};
use rivoli::memory::pin::{V4Pin, V4Route};

#[path = "common/v4_artifact_dir.rs"]
mod v4_artifact_dir;
use v4_artifact_dir::v4_artifact;

#[test]
fn v4_pin_places_every_tensor_into_the_field_its_dimensions_predict() {
    let Some(dir) = v4_artifact("resident.safetensors") else { return };
    let cfg: V4Config = load_config(&dir).unwrap();
    let pin = V4Pin::build(&dir, &cfg).expect("the shipped artifact must load");

    // The pin holds the ARTIFACT's layers, not the model's. The fixture is partial on
    // purpose; if it ever stops being, this stops testing the case it was chosen for.
    assert!(
        pin.layers.len() < cfg.n_layers,
        "this fixture must stay PARTIAL — it holds {} of {} layers",
        pin.layers.len(),
        cfg.n_layers
    );

    let (h, hd, qk) = (cfg.hidden, cfg.head_dim, cfg.q_lora_rank);
    let heads = cfg.n_heads * hd;
    assert_eq!((pin.embed.o_dim, pin.embed.i_dim), (cfg.vocab, h));
    assert_eq!((pin.head.o_dim, pin.head.i_dim), (cfg.vocab, h));

    for (l, p) in pin.layers.iter().enumerate() {
        let dims = |w: &rivoli::memory::pin::Fp8Weight| (w.o_dim, w.i_dim);
        // Every one of these is a different (o_dim, i_dim), so a tensor resolved into the
        // wrong field moves at least one of them.
        assert_eq!(dims(&p.wq_a), (qk, h), "layer {l} wq_a");
        assert_eq!(dims(&p.wq_b), (heads, qk), "layer {l} wq_b");
        // ONE kv entry, head_dim wide, serving as both K and V for all heads.
        assert_eq!(dims(&p.wkv), (hd, h), "layer {l} wkv");
        let o_rank = cfg.o_groups * cfg.o_lora_rank;
        assert_eq!(dims(&p.wo_a), (o_rank, heads / cfg.o_groups), "layer {l} wo_a");
        assert_eq!(dims(&p.wo_b), (h, o_rank), "layer {l} wo_b");
        // The shared expert: fp8 e4m3 and RESIDENT, so it must be here and not in the .f4.
        // `down` is the transposed one, which is what makes it distinguishable from the
        // other two — `gate` vs `up` is not (see the module doc).
        assert_eq!(dims(&p.shared.gate), (cfg.moe_inter, h), "layer {l} w1");
        assert_eq!(dims(&p.shared.up), (cfg.moe_inter, h), "layer {l} w3");
        assert_eq!(dims(&p.shared.down), (h, cfg.moe_inter), "layer {l} w2");

        // Routing: the CONFIG decides which branch, and the artifact carries only that
        // branch's tensor — so a disagreement is a failed load, not a silent fallback.
        match &p.route {
            V4Route::Hash { tid2eid } => {
                assert!(cfg.layer_routes_by_hash(l), "layer {l} hash-routed, config says no");
                assert_eq!(tid2eid.len(), cfg.vocab * cfg.top_k, "layer {l} tid2eid extent");
                // Already range-checked at load; confirm the invariant survived into the pin
                // rather than trusting that the parser ran.
                assert!(
                    tid2eid.iter().all(|&e| (e as usize) < cfg.n_experts),
                    "layer {l} tid2eid holds an id outside 0..{}",
                    cfg.n_experts
                );
            }
            V4Route::Scored { bias } => {
                assert!(!cfg.layer_routes_by_hash(l), "layer {l} scored, config says hash");
                assert_eq!(bias.len(), cfg.n_experts, "layer {l} gate bias");
            }
        }

        // The per-layer optional groups, against `compress_ratios` — 0 means neither, 4
        // means both, 128 means a compressor and no indexer.
        assert_eq!(
            p.compressor.is_some(),
            cfg.layer_has_compressor(l).unwrap(),
            "layer {l} compressor presence (ratio {})",
            cfg.compress_ratio(l).unwrap()
        );
        assert_eq!(
            p.indexer.is_some(),
            cfg.layer_has_indexer(l).unwrap(),
            "layer {l} indexer presence (ratio {})",
            cfg.compress_ratio(l).unwrap()
        );
        if let Some(ix) = &p.indexer {
            assert_eq!(
                dims(&ix.wq_b),
                (cfg.index_n_heads * cfg.index_head_dim, qk),
                "layer {l} indexer wq_b"
            );
        }
    }

    // The routed set is addressable and stops at the last ROUTED expert — the `.f4` has no
    // shared block, and `V4Pin` must not have opened it as though it did.
    assert!(pin.f4.read_spec(0, cfg.n_experts - 1).is_ok());
    assert!(pin.f4.read_spec(0, cfg.n_experts).is_err());
    assert!(pin.f4.shared_block(0).is_err());
}
