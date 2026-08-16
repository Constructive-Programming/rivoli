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

/// Re-lay each head's square state from the K3 reference's `[value][key]` into the
/// kernel's `[key][value]` — `[heads][d][d]`, per head. Named for the POSTCONDITION, not
/// the mechanism: "transpose" is direction-free, and this is the one place in the port
/// where getting the direction backwards is invisible to every assertion.
///
/// **Measured, not chosen.** The state is square at both the tiny widths (32) and the real
/// ones (128), so no shape assertion can see which axis the reference's BUFFER puts first.
/// Scoring both interpretations of the anchor's own `initial_state` against its `out.o`
/// settled it: with the transpose the recurrence agrees to 2.5e-7, without it to 2.2e-1 to
/// 5.6e-1. rivoli's own state starts at zero and never leaves the device, and
/// `[key][value]` is the coalescing order — so the transpose is a FIXTURE boundary, and
/// the anchor's `KdaStateLayout` defect run prices getting it backwards
/// (`k3:tests/k3_kernels.rs:2556`). One owner for the kernel suite and the anchor gate,
/// on [`GoldenSet::k3_gate_lower_bound`]'s precedent — the anchor gate's own copy had
/// silently lost the length assert.
pub fn to_key_major(v: &[f32], heads: usize, dim: usize) -> Vec<f32> {
    assert_eq!(v.len(), heads * dim * dim, "not a per-head square");
    (0..heads)
        .flat_map(|h| (0..dim).flat_map(move |i| (0..dim).map(move |j| (h, i, j))))
        .map(|(h, i, j)| v[h * dim * dim + j * dim + i])
        .collect()
}

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

/// Check every vendored golden against its byte pins. One owner: the three anchor gates
/// each carried this loop until jscpd matched a pair of them (2026-08-15) — found only
/// when a THIRD anchor forced the build script to re-run, the stale-fingerprint hole
/// CLAUDE.md's "clippy-green is not duplication-green" note warns about.
pub fn check_pinned_bytes(goldens: &[Vendored]) {
    for v in goldens {
        v.check_bytes();
    }
}
