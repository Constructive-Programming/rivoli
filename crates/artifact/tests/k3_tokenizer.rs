//! Kimi-K3's tiktoken loader, scored against the FIRST-PARTY reference.
//!
//! Goldens come from `k3_tokenizer_driver.py` — the checkpoint's own `tokenization_kimi.py`
//! constants driven through OpenAI's `tiktoken` over the shipped `tiktoken.model`. That is the
//! tokenizer the model was trained with, so scoring runs one way: `src/tiktoken.rs` is wrong
//! when it disagrees, never the goldens.
//!
//! **Deviceless.** No GPU, no flock, no python at read time — unlike everything else on the K3
//! arm, this runs in CI, on the featureless default `cargo test`.
//!
//! **`pat_str` equality needs no checkpoint at all** and therefore never skips: it compares this
//! tree's constant against the reference's copy in the vendored JSON. That is **one** of the nine
//! gates here; the other eight need `tiktoken.model` and skip without it. Four more gates live in
//! `src/tiktoken.rs`'s own `mod tests` and need no checkpoint either, so **five of thirteen assert
//! in CI** — the count that matters, and it is not nine.
//!
//! > **CORRECTED 2026-08-17, before this ever ran anywhere but here.** This said "every id gate
//! > also asserts a CENSUS. A run that examined nothing cannot report success." **False, and
//! > the second sentence was the dangerous half:** `load_or_skip` returning `None` leads to a
//! > bare `return`, which libtest scores as a PASS. The censuses defend a *partial* run after a
//! > successful load; they say nothing about a run that loaded nothing. On a machine with no
//! > checkpoint this suite reports 9/9 having checked one string constant.
//! >
//! > **Absent-checkpoint policy, copied from `crates/cli/tests/codescene.rs`:** warn-and-skip by
//! > default so a missing artifact cannot brick `cargo test` on a box that has no K3, and
//! > `RIVOLI_K3_REQUIRED=1` turns every skip into a panic. The honest limitation is that tree's
//! > too — libtest captures the skip's `eprintln!`, so a skipped run looks green in the
//! > one-line summary, and the enforcement point is whoever sets the flag. **Nothing sets it
//! > today** (there is no K3 smoke script), so that is owed and named in
//! > `docs/investigations/k3-first-checkpoint.md` §7 rather than implied here.

// tests: panic-on-failure is the idiom, and a broken gate should be loud.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_artifact::tiktoken::{MAX_SAME_CLASS_RUN, PAT_STR, SPECIAL_SLOTS, Vocab};
use serde_json::Value;

/// The converted K3 artifact — deliberately the ARTIFACT and not the 1.42 TiB source.
///
/// Pointing here makes this a gate on what `convert_k3` actually shipped: if the aux list ever
/// stops carrying `tiktoken.model` or `tokenizer_config.json`, these tests skip and say which
/// file is missing, rather than passing against a checkpoint the engine never opens.
/// `RIVOLI_K3_ARTIFACT` overrides it.
///
/// **`RIVOLI_`-prefixed, because these names are one flat namespace across every binary and
/// test in the tree** (`RIVOLI_CS_REQUIRED`, `RIVOLI_BACKENDS`, `RIVOLI_OFFLOAD_ARCH`,
/// `RIVOLI_GPU_LOCK`); a bare `K3_ARTIFACT` is a name collision waiting to happen. An env var
/// rather than a flag because a deviceless test has no command line to read a path from and
/// cannot discover a 1.42 TiB artifact's location — which is this file's own argument, not
/// `crates/oracles/tests/k3-anchor.sh`'s. An earlier draft cited that script; its carve-out
/// explicitly scopes itself to something that "is not a cargo run", and this is one.
const ART_DEFAULT: &str = "/swarm/storage/ai/rivoli/kimi-k3-f4";

fn art() -> String {
    std::env::var("RIVOLI_K3_ARTIFACT").unwrap_or_else(|_| ART_DEFAULT.to_string())
}

/// The reference's own output, vendored beside this file.
fn golden() -> Value {
    let raw = include_str!("k3-tokenizer-cases.json");
    serde_json::from_str(raw).expect("the vendored golden is JSON")
}

/// The reference ids of one case row.
///
/// A helper because four call sites walked `["ids"] -> as_array -> as_u64 -> u32` identically
/// and jscpd reported them (2026-08-17).
fn ids_of(row: &Value) -> Vec<u32> {
    row["ids"]
        .as_array()
        .expect("row has ids")
        .iter()
        .map(|i| i.as_u64().expect("id is a number") as u32)
        .collect()
}

/// Load the vocabulary out of the artifact, or announce the skip.
fn load_or_skip(what: &str) -> Option<Vocab> {
    let dir = art();
    for need in ["tiktoken.model", "tokenizer_config.json"] {
        if std::fs::metadata(format!("{dir}/{need}")).is_err() {
            absent(what, &format!("{dir}/{need} absent"));
            return None;
        }
    }
    Some(Vocab::load(&dir).expect("load the K3 tiktoken vocabulary"))
}

/// Skip-or-panic on a missing checkpoint, per the module header's policy. Returns only in skip
/// mode; every caller's next statement is its `return`.
fn absent(what: &str, detail: &str) {
    assert!(
        std::env::var_os("RIVOLI_K3_REQUIRED").is_none(),
        "K3 tokenizer gate REQUIRED but did not run ({what}): {detail}"
    );
    eprintln!("skip {what}: {detail} (set RIVOLI_K3_REQUIRED=1 to make this a failure)");
}

/// **The vocabulary the goldens were generated from is the one being scored against.**
///
/// Provenance that nothing recomputes is decoration — the ids below would otherwise be scored
/// against whatever `tiktoken.model` happens to sit at `RIVOLI_K3_ARTIFACT`, with the golden's
/// hash sitting inertly beside them. FNV-1a rather than the `sha256` the golden also carries:
/// `rivoli_core::hash::fnv1a` is in the tree and `sha2` is not, and adding a crypto dependency
/// to pin a fixture is the worse trade (that function's own doc says so). The sha256 stays in
/// the JSON for a human comparing against upstream.
#[test]
fn the_vocabulary_matches_the_one_the_goldens_came_from() {
    let dir = art();
    let path = format!("{dir}/tiktoken.model");
    let Ok(bytes) = std::fs::read(&path) else {
        absent("k3 vocabulary provenance", &format!("{path} absent"));
        return;
    };
    let g = golden();
    let prov = &g["provenance"];
    assert_eq!(
        bytes.len() as u64,
        prov["tiktoken_model_bytes"].as_u64().unwrap(),
        "{path} is a different length than the vocabulary the goldens were generated from"
    );
    assert_eq!(
        rivoli_core::hash::fnv1a(&bytes),
        prov["tiktoken_model_fnv1a"].as_u64().unwrap(),
        "{path} has the right length but different bytes than the goldens' vocabulary — every \
         id assertion in this file is scored against the wrong vocabulary"
    );
}

/// **The pattern, character for character.** No checkpoint, so this never skips.
///
/// This is the assertion that catches a hand-copied regex directly. Without it, a mistyped
/// alternative shows up only as an id difference in whichever case happens to cross it — and if
/// no case crosses it, not at all.
#[test]
fn the_pat_str_is_the_references_character_for_character() {
    let g = golden();
    let want = g["pat_str"].as_str().expect("golden carries pat_str");
    assert_eq!(
        PAT_STR, want,
        "src/tiktoken.rs's PAT_STR differs from the first-party pattern the goldens were \
         generated with. Fix the constant, never the golden."
    );
    // The two traps are present as written, asserted by substring so that "the pattern matches"
    // cannot be satisfied by a pattern which quietly dropped one of them.
    // These two `contains`/`matches` rows can only redden when the constant AND the golden moved
    // together — a checkpoint bump plus someone "fixing" the constant to match. That is exactly
    // their job, and it is worth saying so, because they look redundant beside the equality
    // above and the next reader will otherwise delete them.
    assert!(
        PAT_STR.contains(r"\s+(?!\S)"),
        "the negative-lookahead alternative is gone; the `regex` crate cannot express it and \
         dropping it changes whitespace tokenization silently"
    );
    assert_eq!(
        PAT_STR.matches(r"&&[^\p{Han}]]").count(),
        4,
        "the four Han class-intersection clauses keep Han out of Latin runs, and THIS is the \
         only gate that can see them — see the mixed-token test below"
    );
    // `MAX_SAME_CLASS_RUN` is a constant that CHANGES IDS once it trips, which is the property
    // that earned `PAT_STR` its string equality — so it is pinned the same way rather than
    // trusted. Nothing else here would notice a typo: the chunking gate drives the cap
    // explicitly and only uses this value as "a number far above the fixture".
    assert_eq!(
        MAX_SAME_CLASS_RUN as u64,
        golden()["max_no_whitespaces_chars"].as_u64().unwrap(),
        "MAX_SAME_CLASS_RUN differs from the reference's MAX_NO_WHITESPACES_CHARS"
    );
    assert_eq!(
        SPECIAL_SLOTS as u64,
        golden()["num_reserved_special_tokens"].as_u64().unwrap(),
        "SPECIAL_SLOTS differs from the reference's num_reserved_special_tokens"
    );
}

/// **Why the four `&&[^\p{Han}]]` clauses are invisible to id equality**, asserted rather than
/// explained.
///
/// Measured 2026-08-17 while red-proofing: removing one intersection changes the
/// pre-tokenizer — `"hello你好"` becomes one piece instead of two — and leaves every id
/// **identical**, at the reference level too (0 of 12 Han-boundary texts differed in python).
/// The reason is this test's subject: **no token in the vocabulary mixes Han with non-Han**, so
/// no byte-pair merge can cross the boundary and BPE reconstructs the very split the
/// intersection would have made.
///
/// **This is a TRIPWIRE, not the guard** — a correction to an earlier draft that called it the
/// load-bearing one. Trap 2 is already caught twice, by `pat_str` string equality and (better) by
/// `the_pat_str_compiles_under_fancy_regex`, which observes the split itself. What this adds is
/// the *premise*: if a future vocabulary ships a Han-mixing token, it reddens and tells the reader
/// that id equality has **started** covering the intersections — so the reasoning above stops
/// being quietly false rather than silently outliving its evidence.
#[test]
fn no_vocabulary_token_mixes_han_with_non_han() {
    let Some(v) = load_or_skip("k3 han-mixing census") else {
        return;
    };
    // `\p{Han}` and its complement from the same engine the pre-tokenizer uses, so this asks
    // the question in the pattern's OWN terms — `[^\p{Han}]`, exactly as the four clauses spell
    // it — rather than in a hand-rolled codepoint range or a variant that also excludes `\s`.
    let han = fancy_regex::Regex::new(r"\p{Han}").unwrap();
    let other = fancy_regex::Regex::new(r"[^\p{Han}]").unwrap();
    let (mut examined, mut first_mixed) = (0usize, None);
    for rank in 0..v.num_base() as u32 {
        let bytes = v.token_bytes(rank).expect("rank below num_base");
        // A token may be a partial UTF-8 sequence — byte-level BPE guarantees nothing else — and
        // a fragment cannot mix scripts as text, so it is skipped rather than counted as
        // examined.
        let Ok(s) = std::str::from_utf8(bytes) else {
            continue;
        };
        examined += 1;
        if han.is_match(s).unwrap() && other.is_match(s).unwrap() {
            first_mixed = first_mixed.or(Some((rank, s.to_string())));
        }
    }
    assert_eq!(
        first_mixed, None,
        "a token mixes Han with non-Han: byte-pair merges can now cross the script boundary, so \
         the PAT_STR intersections have become visible to id equality. Add Han-boundary cases \
         to the driver and update this test's argument."
    );
    // Non-vacuity, and the ONLY form of it that can fire here. An earlier draft also asserted
    // `examined + undecodable == num_base`, which review pointed out is a tautology: the loop
    // increments exactly one counter per rank over exactly `num_base` ranks, so no defect makes
    // it false. This one can — a `from_utf8` that rejected everything, or a `token_bytes` that
    // returned fragments, drops `examined` and reddens.
    assert!(
        examined > v.num_base() / 2,
        "only {examined} of {} tokens decoded as text — this census is not looking at the \
         vocabulary it claims to",
        v.num_base()
    );
}

/// **Id equality over every vendored case.** The gate.
#[test]
fn every_case_encodes_to_the_first_party_ids() {
    let Some(v) = load_or_skip("k3 id equality") else {
        return;
    };
    let g = golden();
    let cases = g["cases"].as_array().expect("cases");
    let mut seen = std::collections::BTreeMap::<String, usize>::new();
    for c in cases {
        let (name, text) = (c["name"].as_str().unwrap(), c["text"].as_str().unwrap());
        let want = ids_of(c);
        let got = v
            .encode(text)
            .unwrap_or_else(|e| panic!("encode {name}: {e}"));
        assert_eq!(
            got, want,
            "case {name} text={text:?}: ids differ from the first-party reference"
        );
        *seen
            .entry(c["group"].as_str().unwrap().to_string())
            .or_default() += 1;
    }
    // **The census, and it is not decoration.** Each group targets one `pat_str` clause; a
    // golden regenerated with a trimmed case list would otherwise shrink this gate silently
    // while still reporting green.
    let want_groups: &[(&str, usize)] = &[
        ("case", 7),
        ("contractions", 15),
        ("digits", 7),
        ("edge", 11),
        ("han", 10),
        ("lookahead", 13),
        ("newlines", 7),
        ("special_context", 5),
        ("special_named", 16),
        ("special_reserved", 4),
    ];
    let got: Vec<(&str, usize)> = seen.iter().map(|(k, n)| (k.as_str(), *n)).collect();
    assert_eq!(got, want_groups, "the case census moved");
    assert_eq!(cases.len(), 95, "expected 95 vendored cases");
}

/// The special block is POSITIONAL — the half a config-driven port gets wrong.
#[test]
fn the_special_block_is_positional_and_matches_the_reference() {
    let Some(v) = load_or_skip("k3 special block") else {
        return;
    };
    let g = golden();
    assert_eq!(
        v.num_base(),
        g["num_base_tokens"].as_u64().unwrap() as usize,
        "the base vocabulary size moved"
    );
    // Every NAMED id. **This row is NOT independent of the loader, and an earlier draft claimed
    // it was** (review, 2026-08-17): the golden's `named_specials` IS `added_tokens_decoder`'s
    // id→content, the same file `named_specials()` parses, so `special(name) == id` follows from
    // the positional rule plus that config. It is kept as a cheap statement of the mapping the
    // rest of the file depends on. What genuinely IS independent lives elsewhere: `num_base`
    // (python's `len(mergeable)` against Rust's `parse_ranks`, asserted just above), the 240
    // reserved spellings (two independent positional constructions), and the 16 `special_named`
    // CASE rows, whose ids come through python's encoder.
    // The COUNT only. The per-name loop that stood here is deleted (review, 2026-08-17): the
    // golden's `named_specials` IS `added_tokens_decoder`'s id→content, the same file
    // `named_specials()` parses, so `special(name) == id` followed from the positional rule plus
    // that config and asserted nothing the loader could get wrong. The same 16 ids ARE checked
    // independently — as `special_named` CASE rows, whose ids come through python's encoder.
    assert_eq!(
        g["named_specials"].as_object().unwrap().len(),
        16,
        "K3 names 16 of the 256 slots"
    );
    // ALL 240 UNNAMED slots — the rows a port trusting `added_tokens_decoder` alone fails.
    // Read out of the reference's own special map rather than spelled here: the driver used to
    // spell them and got 163839 wrong (it is `[PAD]`), this gate caught it, and deriving both
    // sides from the reference is what stops a fixture asserting its own invention.
    let reserved = g["reserved_examples"].as_object().unwrap();
    assert_eq!(
        reserved.len(),
        SPECIAL_SLOTS - 16,
        "the unnamed-slot census moved"
    );
    for (id, spelling) in reserved {
        let id: u32 = id.parse().unwrap();
        let spelling = spelling.as_str().unwrap();
        assert_eq!(
            v.special(spelling),
            Some(id),
            "reserved slot {id} must still be spellable as {spelling:?}"
        );
    }
}

/// **The boundary, and the eos disagreement pinned.**
///
/// A loader right about text and wrong about where the special block starts produces fluent
/// output that never stops, which no id-equality case over ordinary prose would catch.
#[test]
fn the_vocabulary_boundary_and_the_eos_disagreement_are_pinned() {
    let Some(v) = load_or_skip("k3 vocabulary boundary") else {
        return;
    };
    let dir = art();
    // From the golden, not a literal: the reference states `n_vocab` and this must equal it.
    assert_eq!(
        v.n_vocab() as u64,
        golden()["n_vocab"]
            .as_u64()
            .expect("golden carries n_vocab"),
        "n_vocab differs from the reference's"
    );
    assert_eq!(v.num_base() + SPECIAL_SLOTS, v.n_vocab());

    // `n_vocab == vocab_size`, read from the artifact's own manifest. The logits row count is
    // `vocab_size`, so a vocabulary larger than it can emit an id the head cannot produce and
    // one smaller leaves rows unreachable.
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/manifest.json")).unwrap())
            .unwrap();
    let vocab_size = manifest["text_config"]["vocab_size"].as_u64().unwrap() as usize;
    assert_eq!(
        v.n_vocab(),
        vocab_size,
        "n_vocab != the manifest's vocab_size"
    );

    // The stop token, by SPELLING and by id, against the file the engine reads it from.
    let eos = rivoli_artifact::tokenizer::eos_token_ids(&dir).unwrap();
    assert_eq!(eos, vec![163_586], "generation_config.json's eos_token_id");
    assert_eq!(
        v.special("<|end_of_msg|>"),
        Some(163_586),
        "the stop token's spelling must resolve to the id generation_config names"
    );

    // > **The checkpoint disagrees with itself, and 163586 is the one to decode against.**
    // > `tokenizer_config.json` declares `eos_token: "[EOS]"`, which is id 163585;
    // > `config.json` and `generation_config.json` both say 163586 (`<|end_of_msg|>`). The two
    // > generation-side files agree with each other and with the XTML framing, so they win.
    // > Pinned rather than described: if a future checkpoint reconciles them, this row fails
    // > and someone re-reads all three files instead of inheriting this paragraph.
    let tok_cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{dir}/tokenizer_config.json")).unwrap(),
    )
    .unwrap();
    let declared = tok_cfg["eos_token"].as_str().unwrap();
    assert_eq!(declared, "[EOS]", "tokenizer_config's eos_token spelling");
    assert_eq!(v.special("[EOS]"), Some(163_585), "[EOS] is 163585");
    assert_ne!(
        v.special(declared).unwrap(),
        eos[0],
        "tokenizer_config's eos_token and generation_config's eos_token_id are expected to \
         DISAGREE on this checkpoint; if they now agree, re-read all three files"
    );
}

/// The same-class run chunking, at a small cap so no 25,000-character fixture is needed.
#[test]
fn chunking_matches_the_reference_at_a_small_cap() {
    let Some(v) = load_or_skip("k3 run chunking") else {
        return;
    };
    let g = golden();
    let rows = g["chunked"].as_array().expect("chunked rows");
    let mut n = 0;
    for c in rows {
        let (name, text) = (c["name"].as_str().unwrap(), c["text"].as_str().unwrap());
        let cap = c["max_run"].as_u64().unwrap() as usize;
        let want = ids_of(c);
        assert_eq!(
            v.encode_capped(text, cap).unwrap(),
            want,
            "chunked case {name} at cap {cap}"
        );
        n += 1;
    }
    assert_eq!(n, 7, "expected 7 chunked cases");
    // **Anti-vacuity: chunking must actually CHANGE something.** Without this the rows above
    // would pass for a loader that ignored the cap entirely, since most texts are shorter than
    // any cap. `MAX_SAME_CLASS_RUN` is the shipped value and a 40-character run is far below
    // it, so the two calls must differ.
    let long = "x".repeat(40);
    assert_ne!(
        v.encode_capped(&long, 7).unwrap(),
        v.encode_capped(&long, MAX_SAME_CLASS_RUN).unwrap(),
        "a low cap produced the same ids as the shipped cap — the cap is being ignored"
    );
}

/// **A K3 artifact opens through the real door.** The gate for the defect this milestone found.
///
/// `Tokenizer::load` is what `main.rs` calls, unconditionally, before the architecture match.
/// Until 2026-08-17 it took `tokenizer.json` as the only possibility, so this call failed on a
/// complete and verified 1.42 TiB K3 artifact and `--bench` could not run. Everything else in
/// this file gates `tiktoken::Vocab` directly; this gates the SEAM, which is where the bug was.
#[test]
fn the_shared_tokenizer_door_opens_a_k3_artifact() {
    let dir = art();
    // `absent`, not a bare `eprintln!` + return. This test had its own inline skip and so was
    // the ONE gate that stayed green under `RIVOLI_K3_REQUIRED=1` against a bogus artifact path
    // — found by red-proofing the flag itself (2026-08-17), which is the whole reason to prove a
    // gate can fail rather than to reason that it can.
    if std::fs::metadata(format!("{dir}/tiktoken.model")).is_err() {
        absent("k3 tokenizer door", &format!("{dir}/tiktoken.model absent"));
        return;
    }
    let tok = rivoli_artifact::tokenizer::Tokenizer::load(&dir)
        .expect("Tokenizer::load must open a K3 artifact");
    // Stop tokens arrive through the same path every other arm uses.
    assert_eq!(tok.eos, vec![163_586], "eos ids via the shared door");
    // And it encodes through the shared `encode`, matching the vendored reference ids for a
    // case that is in the golden — so the seam cannot be right about opening and wrong about
    // which vocabulary it opened.
    let g = golden();
    let want = ids_of(
        g["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "latin_han_latin")
            .expect("case latin_han_latin"),
    );
    assert_eq!(tok.encode("hello你好world").unwrap(), want);
    assert_eq!(tok.decode_all(&want).unwrap(), "hello你好world");

    // **The chat door still REFUSES, and that is the correct behaviour, not an oversight.**
    // K3's framing is its first-party XTML encoder and is not ported. The hazard being gated is
    // the silent alternative: `encode_chat_turns` already falls back to RAW encoding with a
    // warning when a vocabulary lacks GLM's chat tokens, and a tiktoken vocabulary lacks all of
    // them, so without an explicit refusal K3 would have quietly taken that path.
    let err = tok
        .encode_chat("hi")
        .expect_err("K3 must refuse GLM chat framing rather than silently encoding raw");
    let msg = err.to_string();
    assert!(
        msg.contains("tiktoken") && msg.contains("XTML"),
        "the refusal should name the format and why: {msg}"
    );
}

/// Round-trip. **Necessary, and far too weak to be the gate** — a consistently wrong encoder
/// round-trips perfectly, which is why id equality above is the real check. This one catches
/// the decode direction, which no id case exercises at all.
#[test]
fn decode_round_trips_every_case() {
    let Some(v) = load_or_skip("k3 round trip") else {
        return;
    };
    let g = golden();
    let cases = g["cases"].as_array().unwrap();
    let mut n = 0;
    for c in cases {
        let text = c["text"].as_str().unwrap();
        let ids = v.encode(text).unwrap();
        let back = v.decode(&ids).unwrap();
        assert_eq!(
            back,
            text,
            "round trip lost {:?}",
            c["name"].as_str().unwrap()
        );
        n += 1;
    }
    assert_eq!(n, 95, "round-tripped fewer cases than the census expects");
    // An id above the vocabulary is an error, not silently empty text: a sampler bug that
    // emitted one would otherwise print nothing and look like a quiet model.
    assert!(v.decode(&[163_840]).is_err(), "163840 is past n_vocab");
}
