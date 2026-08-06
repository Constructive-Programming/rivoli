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
//! Both shipped fixtures are driven, and between them they cover every branch:
//!
//! | fixture | layers | ratios | routing | compressor / indexer |
//! |---|---|---|---|---|
//! | `v4-f4-l0-2` | 0-2 | 0, 0, 4 | all **hash** (`num_hash_layers` = 3) | none, none, both |
//! | `v4-f4-l3-5` | 3-5 | 128, 4, 128 | all **scored** | all three, layer 4 only |
//!
//! `l3-5` is also the only one whose range does not start at 0, so it is what proves
//! `V4Pin::layer`'s absolute-id mapping — against `l0-2` alone an off-by-`range.start` bug
//! is invisible, because the offset is zero.
//!
//! **What this CANNOT see, stated rather than implied:** `w1` and `w3` (gate and up) have
//! identical shapes, so a swap between them is dimensionally invisible here — as
//! `quant::V4_PROJ`'s doc says, only a numerical oracle can catch it, which is what
//! `src/v4oracle/` exists for.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::artifact::model::{V4Config, load_config};
use rivoli::memory::pin::{V4Pin, V4Route};

#[path = "common/v4_artifact_dir.rs"]
mod v4_artifact_dir;

#[test]
fn v4_pin_places_every_tensor_into_the_field_its_dimensions_predict() {
    let mut ran = 0;
    let mut all = Seen::default();
    for dir in [
        v4_artifact_dir::v4_artifact("resident.safetensors"),
        v4_artifact_dir::v4_artifact_l3_5("resident.safetensors"),
    ]
        .into_iter()
        .flatten()
    {
        let s = check_one(&dir);
        all.hash |= s.hash;
        all.scored |= s.scored;
        all.indexer |= s.indexer;
        all.compressor_only |= s.compressor_only;
        ran += 1;
    }
    assert!(ran > 0, "no V4 artifact on this machine — nothing was checked");
    // The union, asserted. Otherwise a machine with only `l0-2` would run all-green having
    // never constructed a `V4Route::Scored`, and this test would report coverage it does
    // not have — which is the failure mode this whole port keeps re-learning.
    assert!(
        all.hash && all.scored && all.indexer && all.compressor_only,
        "the fixtures present did not cover every branch: {all:?} — `l0-2` gives hash + \
         ratio-4, `l3-5` gives scored + ratio-128-without-indexer, and both are needed"
    );
}

/// Which branches a fixture actually constructed. Returned rather than asserted inside,
/// because no single artifact covers all of them and a per-fixture assertion would either
/// fail on the other one or assert nothing.
#[derive(Default, Debug)]
struct Seen {
    hash: bool,
    scored: bool,
    indexer: bool,
    compressor_only: bool,
}

fn check_one(dir: &str) -> Seen {
    let mut seen = Seen::default();
    let cfg: V4Config = load_config(dir).unwrap();
    // `V4Pin::build` now also builds the `.f4` streaming pool, so it takes a device budget.
    // 12 GiB against a 3-layer fixture's ~2.5 GiB resident leaves ~9.5 GiB of pool — the
    // pool's own behaviour is `tests/v4_pool.rs`'s subject; here it only has to construct.
    let pin = V4Pin::build(dir, &cfg, 12 << 30, "2q", Default::default(), None)
        .unwrap_or_else(|e| panic!("{dir} must load: {e:#}"));
    let range = pin.range();

    // The pin holds the ARTIFACT's layers, not the model's. Both fixtures are partial on
    // purpose; if one stops being, it stops testing the case it was chosen for.
    assert!(
        range.len() < cfg.n_layers,
        "{dir} must stay PARTIAL — it holds {} of {} layers",
        range.len(),
        cfg.n_layers
    );
    // Absolute ids only. Outside the range must be refused rather than wrapping into
    // another layer's weights — the failure `V4Pin::layer` exists to make impossible.
    assert!(pin.layer(range.end).is_err(), "{dir}: past the end");
    if range.start > 0 {
        assert!(pin.layer(range.start - 1).is_err(), "{dir}: before the start");
    }

    let (h, hd, qk) = (cfg.hidden, cfg.head_dim, cfg.q_lora_rank);
    let heads = cfg.n_heads * hd;
    assert_eq!((pin.embed.o_dim, pin.embed.i_dim), (cfg.vocab, h));
    assert_eq!((pin.head.o_dim, pin.head.i_dim), (cfg.vocab, h));

    for l in range.clone() {
        let p = pin.layer(l).unwrap();
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
                seen.hash = true;
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
                seen.scored = true;
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
        seen.indexer |= p.indexer.is_some();
        seen.compressor_only |= p.compressor.is_some() && p.indexer.is_none();
        if let Some(ix) = &p.indexer {
            assert_eq!(
                dims(&ix.wq_b),
                (cfg.index_n_heads * cfg.index_head_dim, qk),
                "layer {l} indexer wq_b"
            );
        }
    }

    // The routed set is addressable at the artifact's OWN first layer — not at 0, which is
    // not in `l3-5` at all — and stops at the last ROUTED expert: the `.f4` has no shared
    // block and `V4Pin` must not have opened it as though it did.
    let l0 = range.start;
    assert!(pin.f4.read_spec(l0, cfg.n_experts - 1).is_ok(), "{dir}");
    assert!(pin.f4.read_spec(l0, cfg.n_experts).is_err(), "{dir}");
    assert!(pin.f4.shared_block(l0).is_err(), "{dir}");
    seen
}
