//! What the oracle records, and the reachability census the defect matrix reads beside it:
//! [`Capture`], [`Counters`], and the duplicate-refusing append that makes a golden name
//! mean exactly one tensor.
//!
//! **Split out of `forward.rs` on 2026-08-15, verbatim**, under the 800-line file gate
//! (`crates/cli/tests/line_limit.rs`) and the whole-tree CodeScene 10/10 gate
//! (`crates/cli/tests/codescene.rs`). The cut is by COHESION: this is the recorder, and it
//! knows nothing about the model — no config, no weight, no defect. `forward.rs` re-exports
//! both public types at their original paths, so `v4oracle::forward::{Capture, Counters}`
//! still resolves.

/// What the oracle records. Float tensors are the goldens proper; the integer tensors are
/// SELECTION goldens (indexer top-k, router choices), which no numeric tolerance can stand
/// in for — a wrong Hadamard basis or a wrong router tie-break changes *which* values are
/// combined while leaving every magnitude plausible.
#[derive(Default, Clone)]
pub struct Capture {
    pub floats: Vec<Named<f32>>,
    pub ints: Vec<Named<i64>>,
    pub counters: Counters,
}

/// Reachability counters. These exist so the defect matrix can assert magnitude-gated
/// defects BIDIRECTIONALLY without fitting the expectation to the observation: e.g.
/// `SwigluUnclamped` must perturb a case iff `swiglu_clamped > 0` in that case, which is
/// measured independently of whether the defect fired.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counters {
    /// Clamp EVENTS the `swiglu_limit = 10.0` bound caused -- not elements: one element can
    /// contribute twice when both its `up` and its `gate` are out of range. Only ever read
    /// as zero-vs-nonzero, so the distinction costs nothing, but the name should not lie.
    pub swiglu_clamp_events: usize,
    /// Router logits at which `ln(1 + e^x)` OVERFLOWS f32 -- i.e. where dropping
    /// `softplus`'s `threshold = 20` identity branch is observable at all.
    ///
    /// NOT "logits above 20", which is what this counted first and is the wrong instrument:
    /// for `20 < x < ~88` the two forms are bit-identical in f32 (`ln(1+e^x) = x +
    /// ln(1+e^-x)`, and at x = 21 the correction is 7.6e-10 against an ulp of 1.9e-6). The
    /// threshold only becomes load-bearing where `e^x` reaches infinity, near 88. A counter
    /// that fired at 20 would make `RouterNoSoftplusThreshold` look reachable in a range
    /// where it provably is not.
    pub softplus_overflows: usize,
    /// Compressed blocks emitted by ANY compressor in this call -- the attention one and,
    /// on a ratio-4 layer, the indexer's own. They always fire together, which is why one
    /// counter suffices; `reachable()` reads it as "compression happened at all".
    pub compressed_blocks: usize,
    /// Prefill positions that did NOT survive into the sliding-window ring —
    /// `seqlen.saturating_sub(window_size)`. Zero means the ring was never rotated, so
    /// `PrefillRingWritesFirstWindow` is inert by construction.
    pub prefill_evicted: usize,
    /// Blocks the INDEXER's own compressor emitted. Separate from `compressed_blocks`,
    /// which counted both and so read 2x on any ratio-4 layer -- harmless while only
    /// `> 0` was read, and silently wrong for the first predicate written on the count.
    pub indexer_compressed_blocks: usize,
    /// Query rows where `index_topk` actually CUT (`k < n_compressed`).
    ///
    /// Zero means the indexer selected every compressed block, so `.compress_idxs` records
    /// an invariant set and cannot distinguish a right ranking from a wrong one. Without
    /// this counter that vacuity is invisible -- the goldens still exist, still compare
    /// equal, and still look like coverage.
    pub indexer_truncated: usize,
    /// The indexer ran (this layer has `compress_ratio == 4`).
    pub indexer_ran: bool,
}

impl Capture {
    /// Record a float tensor. Public so a driver can add goldens the layer body does not
    /// produce -- the embedding, a head output -- under the same naming.
    ///
    /// A duplicate name is a hard error, not a second entry. `float()` returns the FIRST
    /// match, so a collision makes every later tensor of that name invisible to both the
    /// comparator and the golden file -- and the four-layer emit produced exactly that
    /// before `run_layer` started prefixing the layer id. Silent shadowing is the failure
    /// mode this whole oracle exists to not have.
    pub fn push(&mut self, name: &str, shape: &[usize], v: Vec<f32>) {
        push_unique(&mut self.floats, name, shape, v);
    }
    /// Record an integer (selection) tensor. Same uniqueness rule as [`Capture::push`].
    pub fn push_i(&mut self, name: &str, shape: &[usize], v: Vec<i64>) {
        push_unique(&mut self.ints, name, shape, v);
    }
    pub fn float(&self, name: &str) -> Option<&[f32]> {
        find_tensor(&self.floats, name)
    }
    pub fn int(&self, name: &str) -> Option<&[i64]> {
        find_tensor(&self.ints, name)
    }
}

/// One recorded tensor: `(name, shape, values)`.
type Named<T> = (String, Vec<usize>, Vec<T>);

/// The FIRST tensor of this name, which is what makes [`push_unique`]'s duplicate assertion
/// load-bearing rather than decorative. Generic so `floats` and `ints` cannot drift apart.
fn find_tensor<'a, T>(from: &'a [Named<T>], name: &str) -> Option<&'a [T]> {
    from.iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, _, v)| v.as_slice())
}

/// Append one named tensor, refusing a duplicate name and a shape that does not describe it.
///
/// Takes the destination list rather than `&mut Capture`: `floats` and `ints` are SEPARATE
/// namespaces, and a helper that searched both would refuse a legal `foo` recorded once as
/// each. So each call still fails for its own caller's reason, and names its own tensor.
fn push_unique<T>(into: &mut Vec<Named<T>>, name: &str, shape: &[usize], v: Vec<T>) {
    assert_eq!(
        shape.iter().product::<usize>(),
        v.len(),
        "{name}: shape/len mismatch"
    );
    assert!(
        find_tensor(into, name).is_none(),
        "duplicate golden name {name}"
    );
    into.push((name.to_string(), shape.to_vec(), v));
}
