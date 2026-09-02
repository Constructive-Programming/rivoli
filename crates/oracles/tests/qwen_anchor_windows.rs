//! **The two goldens that are not a decode step: the 16-position prefill window and the
//! real-parameter n-gram hash.**
//!
//! Each exists for exactly one claim the two decode goldens cannot make, and each earns its bytes
//! by that claim alone. Split from `qwen_anchor.rs` and `qwen_anchor_fixtures.rs` because one file
//! carrying all three subjects came to 1290 lines against this repo's 1200-line hard cap.
//!
//! * the **prefill window** is the dense-versus-selective boundary, which is only visible ACROSS
//!   queries — in decode there is one query and it is entirely selective;
//! * the **n-gram anchor** is the hash at the checkpoint's own 20-million-row base, which the tiny
//!   config shrinks to 1024 so the table fits in a fixture.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::census::qwen::ngram_hash;
use rivoli_artifact::qwen_config::QwenConfig;
use rivoli_artifact::schema::parse_config;
use serde_json::Value;
use std::collections::BTreeSet;

#[path = "common/qwen_anchor_read.rs"]
mod read;

use read::{
    GoldenSet, NGRAM_REAL, PREFILL, REAL_CONFIG, captured_layers, int_shape, ints, load, meta_json,
    shape_of, width,
};

// ---------------------------------------------------------------------------------------------
// The prefill window.

/// **The prefill golden's one claim: BOTH selection regimes in one forward pass.**
///
/// The dense-versus-selective boundary is the whole of trap T19 and half of T8, and it is only
/// visible across QUERIES. `block_topk = indexer_budget / compress_ratio`, and the reference's
/// `topk(min(block_topk, complete_blocks))` (MOD:695) makes QSA **dense by construction** whenever
/// the visible set is at most `indexer_budget + compress_ratio - 1` — 2051 tokens at the real
/// budget, which is why `qwen-architecture.md` records T19's `full_attention` alias as invisible to
/// every planned oracle. At this fixture's budget of 8 that boundary drops to 11, so a 16-position
/// window holds 11 dense queries and 5 selective ones.
///
/// **And the evidence is the ladder of thirteen `k_layernorm` captures, which exists only because a
/// duplicate-name defect was found and fixed on 2026-08-31.** The transcribed indexer body calls
/// that norm once per query position with at least one complete block, and all thirteen results
/// were being written under ONE name — `qwen_anchor_lib.read_golden` refuses duplicates and says of
/// the alternative "the Rust reader keeps both. Latent; loud", but nothing had ever read the prefill
/// golden back, so the latent half was what shipped. `golden_read::float` finds the first match, so
/// twelve of the thirteen were unreachable by name and this test could not have been written.
///
/// Their leading dimensions are the per-query complete-block count, `(query + 1) / compress_ratio`,
/// and comparing each against `block_topk` is what partitions the queries into the two regimes.
#[test]
fn the_prefill_window_holds_both_selection_regimes() {
    let g = load(&PREFILL);
    g.expect_defect("None").expect("an unperturbed run");
    assert_eq!(g.meta_get("mode"), Some("prefill"));
    assert_eq!(g.meta_get("seq"), Some("16"));
    assert_eq!(g.meta_get("salt"), Some("qwen-anchor-1"));
    // **The prefill path runs the CHUNKED recurrence and only that**, where a decode golden records
    // both bodies. That asymmetry is what `common/tolerance.rs` cites for `gdn_op`'s second floor:
    // the chunked path disagrees with the recurrent one by 9.420e-5 over 36 layers, 6.3x the fp64
    // rounding floor, and a chunked PREFILL kernel scored against a recurrent DECODE fixture would
    // have to be exact-only. Pinned here because it is the fact that makes that owed item real.
    assert_eq!(
        g.meta_get("gdn_entry"),
        Some("torch_chunk_gated_delta_rule")
    );
    assert_eq!(g.meta_get("conv_entry"), Some("causal_conv1d_fn"));
    assert_eq!(g.floats.len(), 57, "prefill float tensors");
    assert_eq!(g.ints.len(), 2, "prefill int tensors");
    // ONE layer — the first QSA layer. All seven layers at 16 positions is 2.7 MB of vendored bytes
    // for a claim that lives in one layer's indexer, and the full-depth structure is already pinned
    // by the two decode goldens.
    assert_eq!(
        captured_layers(&g),
        vec![3],
        "the prefill window is one layer"
    );
    let w = meta_json(&g, "widths");
    let (ratio, budget, topk) = (4usize, 8usize, width(&w, "block_topk"));
    assert_eq!(
        topk,
        budget / ratio,
        "block_topk is budget / compress_ratio"
    );
    assert_eq!(
        width(&w, "dense_below"),
        budget + ratio - 1,
        "the provably-dense ceiling is budget + ratio - 1"
    );
    // The repeat census says thirteen, and it is ASSERTED rather than counted from the names, so a
    // capture that silently stopped repeating fails here instead of shrinking a loop to nothing.
    let repeats = meta_json(&g, "repeated_captures");
    let key = "model.layers.3.self_attn.indexer.k_layernorm";
    assert_eq!(repeats[key].as_u64(), Some(13), "the k_layernorm ladder");
    assert_eq!(
        repeats.as_object().expect("repeated_captures").len(),
        1,
        "the indexer norm is the only module called more than once"
    );
    the_ladder_partitions_the_queries(&g, key, ratio, topk);
    // The mask over 16 queries and 16 keys, and the attention weights beside it.
    assert_eq!(
        shape_of(&g, "model.layers.3.self_attn.indexer"),
        vec![1, 1, 16, 16]
    );
    assert_eq!(
        shape_of(&g, "model.layers.3.self_attn.1"),
        vec![1, 4, 16, 16]
    );
    assert_eq!(int_shape(&g, "model.layers.3.mlp.gate.2"), [16, 2]);
}

/// The thirteen leading dimensions, in call order, against the two regimes.
///
/// Call order is the ordinal in the name (bare for the first, `#k` after), and the k-th call is the
/// k-th query position with a complete block — positions 0, 1 and 2 have none at `compress_ratio`
/// 4. So for one-based `k` the query index is `k + ratio - 2` and its block count must be
/// `(query + 1) / ratio`: derived from the ratio, not transcribed from the observed shapes, so a
/// window or ratio change fails here rather than passing on a stale ladder.
fn the_ladder_partitions_the_queries(g: &GoldenSet, key: &str, ratio: usize, topk: usize) {
    let (mut dense, mut selective) = (0usize, 0usize);
    for k in 1..=13 {
        let name = if k == 1 {
            key.to_string()
        } else {
            format!("{key}#{k}")
        };
        let query = k + ratio - 2;
        let blocks = shape_of(g, &name)[0];
        assert_eq!(
            blocks,
            (query + 1) / ratio,
            "{name}: query {query} must see {} complete blocks",
            (query + 1) / ratio
        );
        if blocks > topk {
            selective += 1;
        } else {
            dense += 1;
        }
    }
    // 8 of the thirteen calls are dense. The other three dense queries — 0, 1 and 2 — have no
    // complete block and never reach the norm at all, which is why the dense TOTAL is 11 and only
    // 8 appear in the ladder. Both numbers are asserted, because "11 dense and 5 selective" is the
    // claim and 8 + 5 is what the bytes show.
    assert_eq!((dense, selective), (8, 5), "the two regimes, by query");
    assert_eq!(
        dense + selective + (ratio - 1),
        16,
        "every query accounted for: {dense} dense with blocks, {selective} selective, {} dense \
         with no complete block",
        ratio - 1
    );
    assert_eq!(dense + (ratio - 1), 11, "the provably-dense queries");
}

// ---------------------------------------------------------------------------------------------
// The real-parameter micro-anchor.

/// **The n-gram hash at FULL width, which the tiny config cannot reach.**
///
/// `ngram_vocab_size_base` is 20,000,000 in the checkpoint and 1024 in the tiny config: the tiny
/// fixture keeps the RULE (16 distinct primes strictly above `base - 1`, exclusive-prefix-sum
/// offsets, a `make_ngram_vocab_size_divisible_by` pad) while the table fits in a golden. This
/// anchor is that rule at the real numbers, and its 6 int tensors are the only place in the tree
/// where the 320-million-row hash space appears.
///
/// **Everything here is INTEGER, so there is no tolerance and none is expressible** — the fp32 and
/// fp64 runs produce the same ids, which is why `common/tolerance.rs` gives `ngram_hash` no row and
/// says so in place. A hash is exact or it is wrong. The five n-gram defect rows in the tiny matrix
/// (weakest `ngram_seed_wrong` at 1.691e0) are what make the exact comparison non-vacuous.
///
/// It does not duplicate `crates/artifact/tests/qwen_names.rs`: that file pins which families the
/// fp8 block-128 scheme covers and excludes the n-gram table from it, and separately settles the
/// GDN output norm's ones-centred form from 256 real bytes. This is the hash ARITHMETIC, which
/// neither of those touches.
///
/// **The strongest gate over these bytes is the NEXT test**, which asserts them against
/// `rivoli_artifact::census::qwen::ngram_hash(&cfg, 0)` — 16 primes, 16 offsets, 3 multipliers,
/// byte for byte — turning this rule check into a comparison of the port's derivation against
/// first-party observation. It could not land on track B (2026-09-01): `crates/artifact/src/census/`
/// existed only on `track/qwen-artifact`, and the merge base (`fac53c6`) had neither the module nor
/// `QwenTextConfig`, so the call did not compile there and a test that does not compile is not a
/// gate. It landed 2026-09-02, the first commit in which the two share a tree.
///
/// **The assertion was settled numerically first, so what landed was a known-true one.**
/// Track A's derivation was transliterated and run against these bytes on 2026-09-01: at
/// `ple_layer_index = 0` the multipliers `[23703573157769, 20109073645365, 8052911324071]`, all
/// 16 vocabularies, all 16 offsets, `total_vocab_size` 320,001,446 and the 320,001,536 padded
/// rows all match **exactly**; at `ple_layer_index = 1` every one of them differs
/// (multipliers `[3352040966061, …]`, first prime 20000213 against 20000003). So the gate is
/// non-vacuous in the direction that matters — it distinguishes this checkpoint's PLE layer
/// index. See also the deliberate second `is_prime` below.
#[test]
fn the_real_parameter_ngram_anchor_pins_the_full_width_hash() {
    let g = load(&NGRAM_REAL);
    assert_eq!(g.meta_get("mode"), Some("ngram-real"));
    assert_eq!(g.floats.len(), 0, "this anchor is integer only");
    assert_eq!(g.ints.len(), 6, "int tensors");
    // The one deviation, stated IN the bytes: the embedding WIDTH is 16 instead of 2560, because
    // the ids do not depend on it. Everything else is the checkpoint's own value.
    assert_eq!(
        g.meta_get("embedding_dim_deviation"),
        Some("16 instead of 2560; ids do not depend on it")
    );
    let params = meta_json(&g, "real_parameters");
    let real: Value = serde_json::from_str(REAL_CONFIG).unwrap();
    let real = &real["text_config"];
    for key in [
        "vocab_size",
        "ngram_size",
        "heads_per_ngram",
        "ngram_vocab_size_base",
        "make_ngram_vocab_size_divisible_by",
    ] {
        assert_eq!(params[key], real[key], "{key} must be the checkpoint's own");
    }
    // `seed` is not in the checkpoint's file at all: it is the config class's default, 1234, and
    // `layer_multipliers` is a function of it. Trap T25 in the one place it changes an integer.
    assert_eq!(params["seed"].as_u64(), Some(1234), "the defaulted seed");
    assert!(
        real.get("seed").is_none(),
        "seed is now IN the checkpoint's config.json — the multipliers no longer come from a default"
    );
    the_window_exercises_the_document_boundary(&g, real);
    the_head_tables_follow_the_rule(&g, real);
    every_hashed_id_lands_in_its_own_head(&g);
}

/// **The pinned window contains the REAL eos id, twice.**
///
/// Not decoration: `eos_token_id` pads the n-gram history (MOD:1076) and RESETS the context at every
/// document boundary (MOD:1053-1067), so a window without it never exercises either path — and the
/// tiny config cannot carry the real id at all, because 248044 does not fit a 512-row vocab. This
/// golden is the only place the real id reaches the real hash.
fn the_window_exercises_the_document_boundary(g: &GoldenSet, real: &Value) {
    let eos = real["eos_token_id"].as_i64().expect("eos_token_id");
    let window = ints(g, "window");
    let boundaries = window.iter().filter(|&&t| t == eos).count();
    assert!(
        boundaries >= 2,
        "the window holds {boundaries} document boundaries ({window:?}) — one or none cannot show \
         that the reset happens per boundary rather than once"
    );
    let vocab = real["vocab_size"].as_i64().expect("vocab_size");
    assert!(
        window.iter().all(|&t| t >= 0 && t < vocab),
        "a window token is outside the real vocabulary: {window:?}"
    );
}

/// The 16 head vocabularies and their offsets, checked as the RULE rather than as 32 constants.
///
/// Constants would be the thing to verify; the rule is what a port has to implement. Each head's
/// vocabulary is the next prime at or above `ngram_vocab_size_base`, they are strictly increasing,
/// the offsets are the exclusive prefix sum, and the embedding is padded UP to a multiple of
/// `make_ngram_vocab_size_divisible_by`.
fn the_head_tables_follow_the_rule(g: &GoldenSet, real: &Value) {
    let ngram_size = real["ngram_size"].as_u64().expect("ngram_size");
    let heads = (ngram_size - 1) * real["heads_per_ngram"].as_u64().expect("heads_per_ngram");
    let vocabs = ints(g, "ngram_heads_vocab_sizes");
    let offsets = ints(g, "ngram_heads_offsets");
    let totals = ints(g, "totals");
    assert_eq!(vocabs.len() as u64, heads, "one vocabulary per n-gram head");
    assert_eq!(offsets.len(), vocabs.len(), "one offset per head");
    assert_eq!(
        ints(g, "layer_multipliers").len() as u64,
        ngram_size,
        "one multiplier per n-gram position"
    );
    let base = real["ngram_vocab_size_base"].as_i64().expect("base");
    let mut running = 0i64;
    for (h, (&v, &o)) in vocabs.iter().zip(offsets).enumerate() {
        assert!(
            v >= base,
            "head {h}'s vocabulary {v} is below the base {base}"
        );
        assert!(is_prime(v), "head {h}'s vocabulary {v} is not prime");
        if h > 0 {
            assert!(
                v > vocabs[h - 1],
                "head {h}'s prime {v} is not above head {}'s {}",
                h - 1,
                vocabs[h - 1]
            );
        }
        assert_eq!(
            o, running,
            "head {h}'s offset is not the exclusive prefix sum"
        );
        running += v;
    }
    assert_eq!(
        totals[0], running,
        "total_vocab_size is the sum of the bands"
    );
    let pad = real["make_ngram_vocab_size_divisible_by"]
        .as_i64()
        .expect("pad");
    assert_eq!(
        totals[1] % pad,
        0,
        "the embedding row count is not padded to {pad}"
    );
    assert!(
        totals[1] >= totals[0] && totals[1] - totals[0] < pad,
        "{} embedding rows for {} bands is not the next multiple of {pad}",
        totals[1],
        totals[0]
    );
}

/// **Every hashed id inside its OWN head's band.**
///
/// The check that each offset reached the head it belongs to, which a global range test cannot make:
/// a hash that applied head 5's offset to head 3's id stays inside `[0, total)` and would pass one.
/// `ngram_order_swapped` is the defect row that prices this in the tiny fixture; here it is checked
/// against the real 20-million-wide bands.
fn every_hashed_id_lands_in_its_own_head(g: &GoldenSet) {
    let shape = int_shape(g, "ngram_ids");
    let vocabs = ints(g, "ngram_heads_vocab_sizes");
    let offsets = ints(g, "ngram_heads_offsets");
    let ids = ints(g, "ngram_ids");
    let heads = *shape.last().expect("ngram_ids is [1, positions, heads]");
    assert_eq!(heads, vocabs.len(), "the last axis is the n-gram heads");
    let positions = ids.len() / heads;
    assert_eq!(
        positions,
        ints(g, "window").len(),
        "one id row per window token"
    );
    let mut distinct = BTreeSet::new();
    for (i, &id) in ids.iter().enumerate() {
        let h = i % heads;
        let (lo, hi) = (offsets[h], offsets[h] + vocabs[h]);
        assert!(
            id >= lo && id < hi,
            "id {id} at position {} head {h} is outside that head's band [{lo}, {hi})",
            i / heads
        );
        distinct.insert(id);
    }
    // A hash that collapsed would put every position on one row per head. `> heads` is the weakest
    // statement that rules that out, and the measured value is far above it.
    assert!(
        distinct.len() > heads,
        "only {} distinct ids across {positions} positions and {heads} heads — the hash collapsed",
        distinct.len()
    );
}

/// **The cross-gate the merge made possible: the port's own derivation against the checkpoint's own
/// bytes.** `census::qwen::ngram_hash(&cfg, 0)` must reproduce the golden's vendored int64 buffers —
/// 16 primes, 16 offsets, 3 multipliers IN ORDER, the band total and the padded row count — exactly,
/// and at `ple_layer_index = 1` every prime, every multiplier and every offset past the first must
/// differ (offset 0 is 0 at any ordinal, by definition of an exclusive prefix sum, so it is asserted
/// AS zero rather than counted as a difference). Two arms because the equality alone is not evidence
/// that the ordinal is 0: it is evidence only if 1 disagrees.
///
/// The config is `REAL_CONFIG` through `parse_config`, the converter's own resolution path — not a
/// hand-built `QwenTextConfig` — so a field the port reads wrong reddens here with the same value the
/// converter would spend. `crates/artifact/tests/qwen_names.rs` scores the same derivation against
/// the TSV header's transcription; this is the scoring against first-party bytes, and the two are
/// deliberately not one test (a header transcription scored against itself is the failure both avoid).
///
/// RED OBSERVED (plant: the equality arm handed ordinal 1 instead of 0, 2026-09-02):
/// `left: [20000213, 20000231, ...] right: [20000003, 20000023, ...]` at the primes — the first
/// assertion in the body, so nothing above it could have masked it. Reverted; 4 passed.
#[test]
fn the_ports_ngram_derivation_reproduces_the_goldens_int64_buffers_at_ordinal_zero_only() {
    let g = load(&NGRAM_REAL);
    let cfg: QwenConfig =
        parse_config(REAL_CONFIG).expect("the vendored real config must load and validate");
    let t = &cfg.text;
    let as_u64 = |xs: &[i64]| -> Vec<u64> {
        xs.iter()
            .map(|&x| u64::try_from(x).expect("a non-negative int64"))
            .collect()
    };
    let h0 = ngram_hash(t, 0).expect("the derivation at ordinal 0");
    assert_eq!(
        h0.vocab_sizes,
        as_u64(ints(&g, "ngram_heads_vocab_sizes")),
        "the 16 head vocabularies must be the checkpoint's own primes"
    );
    assert_eq!(
        h0.offsets,
        as_u64(ints(&g, "ngram_heads_offsets")),
        "the 16 offsets must be the checkpoint's own exclusive prefix sums"
    );
    assert_eq!(
        h0.multipliers,
        as_u64(ints(&g, "layer_multipliers")),
        "the 3 multipliers, IN ORDER — a set comparison passes on the shifted SplitMix64 form"
    );
    let totals = as_u64(ints(&g, "totals"));
    assert_eq!(
        h0.vocab_sizes.iter().sum::<u64>(),
        totals[0],
        "the band total"
    );
    assert_eq!(h0.padded_rows, totals[1], "the padded embedding row count");

    // The discriminant: the same derivation at the NEXT ordinal shares nothing with the bytes.
    let h1 = ngram_hash(t, 1).expect("the derivation at ordinal 1");
    let all_differ =
        |a: &[u64], b: &[u64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x != y);
    assert!(
        all_differ(&h1.vocab_sizes, &h0.vocab_sizes),
        "every prime must move at ordinal 1: {:?} vs {:?}",
        h1.vocab_sizes,
        h0.vocab_sizes
    );
    assert!(
        all_differ(&h1.multipliers, &h0.multipliers),
        "every multiplier must move at ordinal 1 (`base_seed = seed + 10007 * ordinal`)"
    );
    assert_eq!(h1.offsets[0], 0, "offset 0 is 0 at any ordinal");
    assert!(
        all_differ(&h1.offsets[1..], &h0.offsets[1..]),
        "every offset past the first must move at ordinal 1"
    );
}

/// Trial division. The vocabularies are around 2e7, so this is under 4500 divisions each — cheaper
/// than a dependency, and cheaper than pinning 16 constants that would themselves become the thing
/// nobody verified.
///
/// **A SECOND trial-division walk exists in the tree, and keeping it is deliberate** (2026-09-01).
/// `rivoli_artifact::census::qwen::ngram_hash` (track A, `track/qwen-artifact`) carries its own
/// `is_prime` over `u64`, and a reviewer proposed collapsing the pair. The old argument for this
/// copy — "cheaper than a dependency" — does not cover that, because A's is not a dependency;
/// this one does. **The check this file makes is a check ON that derivation.** The cross-gate
/// above asserts A's 16 primes equal the golden's vendored int64 bytes; the rule assertions
/// assert the golden's own primes ARE prime. Route both through one primality walk and a bug in
/// it — an overflowing `d * d`, a mishandled small case — makes the derivation and its check
/// agree by construction, which is the recorded "recovered-parameter gates live on an asymmetry"
/// failure and cancels exactly the defect the gate is hunting. Independence is the point, so the
/// duplication is priced rather than removed; jscpd does not report the pair (token floor, plus
/// `i64` against `u64`), so this note is what keeps the decision visible.
fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    if n % 2 == 0 {
        return n == 2;
    }
    let mut d = 3;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 2;
    }
    true
}
