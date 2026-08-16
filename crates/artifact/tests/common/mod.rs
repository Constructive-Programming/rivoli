//! Shared helpers for the per-architecture load-boundary gates (`glimmer_config.rs`,
//! `v4_config.rs`).
//!
//! **It exists because jscpd said so, which is this tree's rule for when a helper is shared.**
//! `v4_config.rs` arrived at M8 as the second gate built on "mutate the shipped config at a JSON
//! pointer and require the refusal to name its reason", and `crates/cli/build.rs` reported four
//! clones against `glimmer_config.rs` on the first compile. The response the repo prescribes is
//! to factor, never to exempt — see `crates/cli/tests/common/mod.rs`, whose own header records
//! the same growth rule and the same trigger.
//!
//! Generic over [`ArchConfig`] rather than per-model: the property under test is the *schema*
//! contract, which is one contract with three implementors, and a per-model copy is exactly the
//! thing that lets one gate acquire a fix the others silently do not.

// Compiled into EACH including binary; neither uses every helper. The engine tests' common and
// the cli's carry the same argument.
#![allow(dead_code)]
// Meta-gate: a mutation that is wrongly ACCEPTED must panic loudly and name itself.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_artifact::schema::{ArchConfig, parse_config};
use serde_json::Value;

/// The refusal message for `shipped` with the value at `pointer` replaced, or a panic naming
/// the mutation that was wrongly ACCEPTED.
///
/// **The panic is the point**: a refusal test whose subject silently parses is the false green
/// these gates exist to prevent. A missing `pointer` panics too, so a test row that stops
/// naming a real key in the document reddens rather than quietly testing nothing.
///
/// Goes through [`parse_config`] — the same entry every converter uses. A test that constructed
/// the config struct directly would skip both the architecture check and `validate`, which are
/// the two things under test.
pub fn refusal<T: ArchConfig>(shipped: &str, pointer: &str, value: Value) -> String {
    let mut doc: Value = serde_json::from_str(shipped).unwrap();
    let slot = doc
        .pointer_mut(pointer)
        .unwrap_or_else(|| panic!("{pointer} is not a path in the shipped config"));
    *slot = value;
    match parse_config::<T>(&doc.to_string()) {
        Ok(_) => panic!("{pointer} was mutated to a wrong value and the config still parsed"),
        Err(e) => format!("{e:#}"),
    }
}

/// Every `(pointer, value, want)` row: mutate `shipped` there, and require the refusal's
/// MESSAGE to contain `want`.
///
/// **The message check is the row's whole content.** A refusal that happens to fire for an
/// unrelated reason would satisfy a bare `is_err()`, which is how a guard gets deleted without a
/// red test — and the old tree records a case where two rows shared a `want` substring and
/// transposing an argument left both green. One function rather than a loop per test because
/// jscpd reported the two loop tails as a clone the moment the second table existed, and the
/// argument the tail carries is not one to have twice.
pub fn each_refusal<T: ArchConfig>(shipped: &str, rows: &[(&str, Value, &str)]) {
    for (pointer, value, want) in rows {
        let err = refusal::<T>(shipped, pointer, value.clone());
        assert!(
            err.contains(want),
            "{pointer} = {value} refused, but not for the reason under test\n  \
             wanted the message to contain: {want}\n  got: {err}"
        );
    }
}
