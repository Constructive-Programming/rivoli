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
    lookup(&g.floats, name).unwrap_or_else(|| absent(name, "float", &g.floats))
}

/// One int tensor's values, by name. Same panic-on-absent contract as [`float`], and for the same
/// reason: a check that locates its input by name has a third outcome, and silently defaulting on
/// it is a gate that reads as coverage and is zero.
pub fn ints<'g>(g: &'g GoldenSet, name: &str) -> &'g [i64] {
    lookup(&g.ints, name)
        .map(|(_, v)| v)
        .unwrap_or_else(|| absent(name, "int", &g.ints))
}

/// The lookup both readers are the same question of.
///
/// Written twice as a find-map-unwrap_or_else over the same `(name, shape, values)` tuple, the
/// code-health reviewer scored the pair as duplicated code and the file sank to 9.38; only the
/// element type and one noun in the message differ, so the type is generic and the noun is an
/// argument. Neither message moved, and neither did the contract: an absent name is a refusal
/// that names what IS present, never a default.
fn lookup<'g, X>(
    rows: &'g [(String, Vec<usize>, Vec<X>)],
    name: &str,
) -> Option<(&'g [usize], &'g [X])> {
    rows.iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, s, v)| (s.as_slice(), v.as_slice()))
}

/// The absent-name panic, shared for the reason above. A renamed capture is the event this
/// branch exists for, and "the golden holds 41 float tensors, e.g. […]" is what turns the next
/// question into a lookup rather than a re-run with a print.
fn absent<X>(name: &str, kind: &str, rows: &[(String, Vec<usize>, Vec<X>)]) -> ! {
    let some: Vec<&String> = rows.iter().take(3).map(|(n, _, _)| n).collect();
    panic!(
        "{name} is not in the golden; it holds {} {kind} tensors, e.g. {some:?}",
        rows.len()
    )
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

// -----------------------------------------------------------------------------------------
// **The checks every anchor gate makes, factored here 2026-08-31 when the Qwen anchor became
// the FOURTH caller and jscpd reported ELEVEN clones against `k3_anchor.rs` — 34 to 133
// tokens each, all of them real.**
//
// They were not written as copies on purpose: every port's anchor has to ask the same five
// questions (what produced this file, are the two draws two draws, did the head emit logits,
// did top-k select distinct experts, which layers were captured), and each port answered them
// in its own file until a fourth one made the gate speak up. That is the sequence
// `check_pinned_bytes` above already records for the byte pins, one round later.
//
// The alternative was an exemption, and it would have been wrong for the reason this file's
// own header gives: the matched tokens are not the point of anything here. Nothing about
// "assert the salt reached the draw" is per-model — only the numbers are, and those stay at
// the call sites as arguments.

use serde_json::Value;

/// A JSON integer field as a `usize`, panicking with the key.
///
/// Every anchor reads widths and counts out of a golden's recorded config this way. K3 and
/// Qwen each had their own `field`; jscpd matched them at 55 tokens.
pub fn json_usize(c: &Value, key: &str) -> usize {
    c[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is not an integer in the recorded config")) as usize
}

/// **What produced this golden, by value** — the unperturbed-run claim, the salt, and every
/// `(key, expected)` pair the caller pins.
///
/// The pairs stay at the call site because they are the per-model half; the loop and the two
/// checks around it are not. A golden regenerated under a different stack or a flipped
/// implementation selector is a golden of a different computation, with metadata truthfully
/// recording a configuration nothing compared — which is why this is by value and not a
/// presence check. Two of K3's were presence-only until review 2026-08-11 and could be
/// satisfied by the driver's own failure sentinels.
pub fn provenance(g: &GoldenSet, salt: &str, pairs: &[(&str, &str)]) {
    g.expect_defect("None")
        .expect("a vendored anchor golden is an unperturbed run");
    assert_eq!(g.meta_get("salt"), Some(salt), "{salt}: metadata salt");
    for (key, want) in pairs {
        assert_eq!(g.meta_get(key), Some(*want), "{salt}: metadata {key}");
    }
}

/// **The second salt must be a second DRAW, not the first one wearing a different label.**
///
/// Comparing the two files' hashes carries no information: each embeds its own `salt` string
/// in its metadata, so they are guaranteed to differ whatever the weights did. A driver
/// refactor that stopped mixing the salt into the parameter seed would put bit-identical
/// tensors in both files and pass every other assertion in an anchor gate — while the fixture
/// claimed the second draw whose whole purpose is that a bug degenerate at one draw's values
/// cannot hide.
///
/// So it is compared per tensor, over every float the two share, on `to_bits` — because
/// `-0.0 == 0.0` and `NaN != NaN` would each lie here in one direction. `expected_shared` is
/// asserted rather than merely lower-bounded: the two goldens must hold the SAME tensors and
/// differ in their VALUES, and a shrinking intersection is its own finding.
pub fn draws_are_independent(a: &GoldenSet, b: &GoldenSet, expected_shared: usize) {
    let mut shared = 0usize;
    let mut identical: Vec<&str> = Vec::new();
    for (name, _, va) in &a.floats {
        let Some((_, _, vb)) = b.floats.iter().find(|(n, _, _)| n == name) else {
            continue;
        };
        shared += 1;
        if va
            .iter()
            .map(|x| x.to_bits())
            .eq(vb.iter().map(|x| x.to_bits()))
        {
            identical.push(name);
        }
    }
    assert_eq!(
        shared, expected_shared,
        "the two draws must hold the same tensors and differ in their values"
    );
    assert!(
        identical.is_empty(),
        "{} of {shared} shared float tensors are bit-identical across the two salts, e.g. {:?}. \
         Weights come from `sha256(salt/parameter-name)`, so EVERY tensor must differ; any that \
         do not mean the salt stopped reaching the draw.",
        identical.len(),
        &identical[..identical.len().min(3)]
    );
}

/// The fields the GENERATING side recorded that it asserted, from `structural_asserted`.
///
/// The point of the metadata key: a gate here may only claim a field the driver actually
/// checked, so two lists with nothing tying them together — the drift K3's review found, where
/// two fields were documented as asserted on both sides and asserted on neither — becomes one
/// list plus a lookup.
pub fn declared_fields(g: &GoldenSet) -> Vec<&str> {
    g.meta_get("structural_asserted")
        .expect("structural_asserted")
        .split(',')
        .collect()
}

/// The forward pass produced numbers and it did not produce ONE number.
///
/// A collapsed head is the shape a golden of zeros takes at the far end of the model, and it
/// passes every shape assertion an anchor makes.
pub fn logits_are_finite_and_spread(g: &GoldenSet, name: &str) {
    let (_, logits) = float(g, "logits");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "{name}: logits must be finite"
    );
    assert!(
        logits.iter().any(|x| *x != logits[0]),
        "{name}: all {} logits are identical — the forward pass collapsed",
        logits.len()
    );
}

/// Top-k routing indices: in range, and **distinct**.
///
/// Distinctness is the real check. `[0, 0]` is a pair top-k cannot produce, and a bound of
/// `0..experts` alone would accept it — so an all-zero int section would pass the range half
/// on its own.
pub fn top_k_is_in_range_and_distinct(idx: &[i64], top_k: usize, experts: usize, name: &str) {
    assert!(
        idx.iter().all(|&e| (e as usize) < experts),
        "{name}: expert id out of range: {idx:?}"
    );
    assert_eq!(
        idx.iter().collect::<std::collections::BTreeSet<_>>().len(),
        top_k,
        "{name}: top-{top_k} selected the same expert twice: {idx:?}"
    );
}

/// Every captured tensor's name, floats then ints.
pub fn names(g: &GoldenSet) -> impl Iterator<Item = &str> {
    g.floats
        .iter()
        .map(|(n, _, _)| n.as_str())
        .chain(g.ints.iter().map(|(n, _, _)| n.as_str()))
}

/// **The capture census: three lists, because each PAIR of them alone has a hole.**
///
/// `expected` is what the port's gate says it wants, `capture_layers` is what the driver says
/// it hooked, and the tensor names are what is actually there. Comparing only the last two
/// lets a narrowed run agree with its own metadata; comparing only the first two lets a driver
/// claim layers it never captured. The reason is a measured bug: K3's driver once matched
/// `model.layers.1` as a PREFIX and captured layers 1 and 10-19 (and 3 with 30-39) — 25 layers
/// instead of 6, invisible in every per-tensor assertion and obvious here.
pub fn capture_census(g: &GoldenSet, expected: &[usize], name: &str) {
    let declared: Vec<usize> = g
        .meta_get("capture_layers")
        .expect("capture_layers")
        .split(',')
        .map(|s| s.parse().expect("a layer index"))
        .collect();
    assert_eq!(declared, expected, "{name}: capture_layers metadata");
    assert_eq!(
        captured_layers(g),
        declared,
        "{name}: captured layers must be exactly the declared ones"
    );
}

/// **The per-head decay rates are not ALL EQUAL.**
///
/// Both linear-attention families here carry one (`A_log` in K3's KDA and in Qwen's gated delta
/// rule), and a range check alone accepts an ALL-ZEROS draw because `log(1) = 0` is in range. A
/// constant decay makes every head decay identically, so a kernel that ignored the term
/// entirely would still match — on K3 only the FNV pin caught that.
pub fn decay_rates_are_not_all_equal(a_log: &[f32], name: &str) {
    assert!(
        a_log.iter().any(|x| *x != a_log[0]),
        "{name}: A_log is constant at {} — every head would decay identically",
        a_log[0]
    );
}

/// The head emitted exactly one row of logits, as wide as the recorded vocabulary.
pub fn one_row_of_logits(g: &GoldenSet, c: &Value) {
    assert_eq!(
        shape_of(g, "logits"),
        vec![1, 1, json_usize(c, "vocab_size")]
    );
}

/// The layer indices a golden actually holds tensors for, sorted and deduplicated.
pub fn captured_layers(g: &GoldenSet) -> Vec<usize> {
    let mut seen: Vec<usize> = names(g)
        .filter_map(|n| n.strip_prefix("model.layers."))
        .filter_map(|rest| rest.split('.').next())
        .map(|d| d.parse().expect("a layer index"))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen
}
