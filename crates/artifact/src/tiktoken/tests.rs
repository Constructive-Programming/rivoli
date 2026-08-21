//! `tiktoken`'s unit tests — the checkpoint-free half of that module's gates.
//!
//! Split out of `tiktoken.rs` when it crossed the 800-line soft cap, following
//! `v4_encoding`'s `mod tests;` precedent. **Every test here needs no artifact**, which is what
//! makes them the K3 arm's only gates that assert in CI rather than skipping — so they are also
//! where a tripwire's red proof belongs (see `the_han_mixing_tripwire_fires_on_a_token_that_mixes`
//! and its two siblings, each driven over a SYNTHETIC vocabulary).

// Crate-wide `unwrap`/`expect` are deny; in a unit test a firing one IS the report, and these
// fixtures carry no checkpoint that could be legitimately missing.
#![allow(clippy::expect_used, clippy::unwrap_used)]
use super::*;

/// The pattern must COMPILE under the engine that has to run it, and SPLIT as the reference
/// does. Separate from the id gate because a pattern that fails to compile makes every other
/// case fail for the same uninformative reason, and because this is the assertion that
/// `fancy-regex` — not `regex` — is the right dependency.
///
/// **Load-bearing for the Han intersections, which id equality cannot see** (see `PAT_STR`).
/// The `"AB你好c"` row observes the split BEHAVIOURALLY rather than comparing pattern text,
/// so it reddens on a broken intersection even though every id stays identical — measured
/// 2026-08-17. Checkpoint-free, so this is one of the few gates on this arm that asserts in
/// CI rather than skipping.
#[test]
fn the_pat_str_compiles_under_fancy_regex() {
    let re = fancy_regex::Regex::new(PAT_STR).unwrap();
    // The lookahead alternative actually takes effect: with `\s+(?!\S)` the run before a
    // word gives up its last space. This is the behaviour `regex` cannot express at all.
    let split = |s: &str| -> Vec<String> {
        re.find_iter(s)
            .map(|m| m.unwrap().as_str().to_string())
            .collect()
    };
    assert_eq!(
        split("a    b"),
        ["a", "   ", " b"],
        "the lookahead is not in effect"
    );
    // And the class intersections keep Han out of the Latin runs. **`AB你好c` rather than
    // `hello你好`, because there are FOUR intersection clauses and one probe does not see
    // them all** (measured 2026-08-17, after review pointed out the gap): `hello你好`
    // detects only alt-2's LOWERCASE clause, and `A你好` — the probe review itself proposed
    // — detects three of four, missing alt-2's UPPERCASE one. `AB你好c` exercises an
    // uppercase run, a Han run and a trailing lowercase run in one string, and removing ANY
    // of the four changes its split. Without it, two of the four clauses had no behavioural
    // gate at all and rested on string equality alone.
    assert_eq!(
        split("AB你好c"),
        ["AB", "你好", "c"],
        "Han was absorbed into a Latin run — one of the four &&[^Han] clauses is broken"
    );
}

#[test]
fn same_class_runs_split_at_the_cap() {
    // Under the cap: one chunk, unchanged.
    assert_eq!(split_same_class_runs("aaa", 5), vec!["aaa"]);
    // Over it: the run breaks, and only same-class characters count toward it.
    assert_eq!(split_same_class_runs("aaaa", 2), vec!["aa", "aa"]);
    assert_eq!(split_same_class_runs("aa  aa", 2), vec!["aa  aa"]);
    // Empty input must not index `s[0]` — the reference guards this explicitly.
    assert_eq!(split_same_class_runs("", 4), vec![""]);
}

#[test]
fn a_vocabulary_missing_a_single_byte_is_refused() {
    // Two-byte tokens only: BPE could not represent an arbitrary byte, so `assemble`
    // refuses rather than failing later on some particular input.
    let bytes_of: Vec<Vec<u8>> = (0..4u8).map(|b| vec![b, b]).collect();
    let ranks = ranks_from(&bytes_of);
    // Matched rather than `unwrap_err`, which would need `Debug` on `Vocab` — and a
    // derived `Debug` on a struct holding the 163,584-entry rank map is a 19 MB print
    // waiting for whoever first adds `{:?}` to a diagnostic.
    let err = match Vocab::assemble(ranks, bytes_of, HashMap::new()) {
        Ok(_) => panic!("a vocabulary with no single-byte tokens was accepted"),
        Err(e) => e,
    };
    assert!(
        err.to_string().contains("of 256 single-byte tokens"),
        "unexpected: {err}"
    );
}

/// A vocabulary with all 256 bytes, plus whatever extra tokens the caller wants. The
/// smallest thing `assemble` accepts, so a unit test can exercise a `Vocab` property with no
/// checkpoint on disk.
fn synthetic(extra: &[&[u8]]) -> Vocab {
    try_synthetic(extra, HashMap::new()).expect("synthetic vocab")
}

/// The fallible form, and the one that takes NAMED specials. One builder, because three
/// fixtures wanted a `Vocab` with no checkpoint and jscpd reported the copies.
fn try_synthetic(extra: &[&[u8]], named: HashMap<u32, String>) -> Result<Vocab> {
    let mut bytes_of: Vec<Vec<u8>> = (0..=u8::MAX).map(|b| vec![b]).collect();
    bytes_of.extend(extra.iter().map(|b| b.to_vec()));
    let ranks = ranks_from(&bytes_of);
    Vocab::assemble(ranks, bytes_of, named)
}

/// **The Han tripwire fires**, proven on a vocabulary that has a mixing token.
///
/// Without this the tripwire is a check that has only ever been green, which is the shape the
/// house rules refuse. It also lands the proof in CI: the real-vocabulary census needs a 2.7 MB
/// `tiktoken.model` and skips without it, while this needs nothing.
#[test]
fn the_han_mixing_tripwire_fires_on_a_token_that_mixes() {
    // Clean: single bytes only. A lone byte of a multi-byte codepoint is not valid UTF-8 and
    // is skipped, so no single-byte vocabulary can mix scripts.
    assert_eq!(synthetic(&[]).mixed_script_token(), None);
    // `a` + 你 in ONE token — exactly what would let a byte-pair merge cross the boundary.
    let v = synthetic(&["a你".as_bytes()]);
    let (rank, text) = v
        .mixed_script_token()
        .expect("the mixing token must be found");
    assert_eq!((rank, text.as_str()), (256, "a你"));
    // And Han alone is NOT mixing — otherwise the tripwire would fire on every CJK vocabulary.
    assert_eq!(synthetic(&["你好".as_bytes()]).mixed_script_token(), None);
}

/// **The prefix tripwire fires.** `find_special` is longest-match; tiktoken is
/// leftmost-first-alternative, and the two agree only while no spelling prefixes another.
#[test]
fn the_special_prefix_tripwire_fires_on_a_prefixed_spelling() {
    // K3's own 256 positional spellings differ only in an id, so none prefixes another.
    assert_eq!(synthetic(&[]).prefixed_special(), None);
    // Name two slots so that one spelling IS a strict prefix of the other.
    let mut bytes_of: Vec<Vec<u8>> = (0..=u8::MAX).map(|b| vec![b]).collect();
    bytes_of.push(b"zz".to_vec());
    let ranks = ranks_from(&bytes_of);
    let named = HashMap::from([
        (257u32, "<|s|>".to_string()),
        (258u32, "<|s|>x".to_string()),
    ]);
    let v = Vocab::assemble(ranks, bytes_of, named).expect("synthetic vocab");
    assert_eq!(
        v.prefixed_special(),
        Some(("<|s|>".to_string(), "<|s|>x".to_string()))
    );
}

/// **The distinctness invariant fires.** Two named slots given the SAME spelling.
///
/// Cannot be red-proofed by perturbing the real artifact — no shipped config collides — so the
/// proof has to be a synthetic one, and it lands in CI where the artifact never is.
#[test]
fn colliding_special_spellings_are_refused() {
    let named = HashMap::from([
        (257u32, "<|dup|>".to_string()),
        (258u32, "<|dup|>".to_string()),
    ]);
    let err = match try_synthetic(&[b"zz"], named) {
        Ok(_) => panic!("two ids sharing one spelling was accepted: one is unencodable"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("distinct spellings"), "unexpected: {err}");
}

#[test]
fn non_dense_ranks_are_refused() {
    // `AQ==` is byte 0x01. Rank 0 then rank 2: decode indexes by rank, so the gap would
    // shift every token above it.
    let err = parse_ranks("AQ== 0\nAg== 2\n").unwrap_err();
    assert!(err.to_string().contains("not dense"), "unexpected: {err}");
}
