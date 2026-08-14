//! Reading vendored S1b anchor goldens, shared by every model that has one.
//!
//! Two ports now vendor python-produced goldens (`tests/k3_anchor.rs`, `tests/glimmer_anchor.rs`)
//! and both need the same three things: a tensor by name, its shape, and a hash of the file's
//! bytes. Written twice these are a `build.rs` jscpd failure at `--min-tokens 15` — `fnv1a` alone
//! is about twenty tokens of pure arithmetic — so they live here from the moment there is a second
//! caller, which is the rule this repo applies everywhere else.
//!
//! Included with `#[path]` rather than through `common/mod.rs`, like `k3_tolerance.rs`: a test
//! binary that wants these does not want the 46 KB of artifact helpers next door.

#![allow(dead_code)] // each consumer uses a subset; the unused half is not dead for the other one

/// Re-exported so a consumer of this facade needs one import, not two. Not cosmetic: with both,
/// every anchor test opens with the same four-line preamble and `build.rs`'s jscpd gate matches
/// them — an import list is the one duplication Rust gives you no way to factor, so the fix is to
/// have fewer imports rather than an exemption saying the copy is the point.
pub use rivoli_oracles::golden::GoldenSet;

/// One float tensor's shape and values, by name. Panics with the file's own contents, because "not
/// found" is almost always a renamed capture and the next question is always "then what IS in
/// there".
pub fn float<'g>(g: &'g GoldenSet, name: &str) -> (&'g [usize], &'g [f32]) {
    g.floats
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, s, v)| (s.as_slice(), v.as_slice()))
        .unwrap_or_else(|| {
            let some: Vec<&String> = g.floats.iter().take(3).map(|(n, _, _)| n).collect();
            panic!(
                "{name} is not in the golden; it holds {} float tensors, e.g. {some:?}",
                g.floats.len()
            )
        })
}

/// One int tensor's values, by name. Same panic-on-absent contract as [`float`], and for the same
/// reason: a check that locates its input by name has a third outcome, and silently defaulting on
/// it is a gate that reads as coverage and is zero.
pub fn ints<'g>(g: &'g GoldenSet, name: &str) -> &'g [i64] {
    g.ints
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, _, v)| v.as_slice())
        .unwrap_or_else(|| {
            let some: Vec<&String> = g.ints.iter().take(3).map(|(n, _, _)| n).collect();
            panic!(
                "{name} is not in the golden; it holds {} int tensors, e.g. {some:?}",
                g.ints.len()
            )
        })
}

pub fn shape_of(g: &GoldenSet, name: &str) -> Vec<usize> {
    float(g, name).0.to_vec()
}

/// FNV-1a byte pinning, re-exported from its one owner. The body lived here until the
/// CodeScene gate's cache needed the same hash and jscpd reported the pair (2026-08-15);
/// `rivoli_core::hash` now carries it and the published-vector test.
pub use rivoli_core::hash::fnv1a;

/// One vendored golden, with the two facts that pin its bytes.
///
/// Shared since 2026-08-11, when Muse Glimmer's anchor made this a second table and jscpd matched
/// the two declarations. **`name` is a label for failure messages, not a claim about the file** —
/// what binds an entry to its bytes is [`Vendored::check_bytes`]; anything the file says about
/// itself (its mode, its salt) is read out of its own metadata, never restated here.
pub struct Vendored {
    pub name: &'static str,
    pub bytes: &'static [u8],
    pub len: usize,
    pub fnv: u64,
}

impl Vendored {
    /// **When this fails after a deliberate regeneration, update the constants and say so in the
    /// port's `anchor.md`.** That is the intended workflow: re-vendoring is a reviewed change, not
    /// a side effect of running the driver.
    pub fn check_bytes(&self) {
        assert_eq!(self.bytes.len(), self.len, "{}: length", self.name);
        assert_eq!(fnv1a(self.bytes), self.fnv, "{}: FNV-1a", self.name);
    }
}
