//! Content pinning. One owner for the whole workspace — the CodeScene gate's cache and
//! the anchor fixtures' byte pins each carried a copy until jscpd reported the pair on
//! the day the second one was ported (2026-08-15), which is the gate doing its job.

/// FNV-1a's offset basis and prime, named once. They were literals in one function; they
/// became constants when a second and third consumer arrived, because a transcription typo
/// in a copy is exactly the failure the published-vectors test below can only catch in the
/// copy it tests.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold one byte into an FNV-1a state. The whole algorithm is this line; the two functions
/// below differ only in how they get their bytes, so this is where it is written down.
#[inline]
fn fnv1a_byte(h: u64, b: u8) -> u64 {
    (h ^ u64::from(b)).wrapping_mul(FNV_PRIME)
}

/// FNV-1a over a byte slice. Not a cryptographic claim — it exists so a vendored fixture
/// that changed by one byte cannot pass as the one a doc describes, and adding a sha2
/// dependency to hash test fixtures is the worse trade.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET, |h, &b| fnv1a_byte(h, b))
}

/// FNV-1a over a sequence of `u64`, each folded little-endian byte-wise — so this agrees
/// with [`fnv1a`] over the same values' `to_le_bytes()` concatenation, and a caller that has
/// integers rather than a buffer does not have to materialise one.
///
/// Takes an iterator so a caller can widen ids, offsets or bit patterns on the fly. Its
/// consumer is `rivoli_engine::probe`'s divergence log, whose columns are expert ids and
/// arena offsets rather than bytes.
pub fn fnv1a_u64s(vals: impl Iterator<Item = u64>) -> u64 {
    vals.fold(FNV_OFFSET, |h, v| {
        v.to_le_bytes().iter().fold(h, |h, &b| fnv1a_byte(h, b))
    })
}

/// splitmix64's finalizer: an avalanche step in which every input bit reaches every output
/// bit. Not a generator — the caller owns the state; this is only the mixing.
///
/// **One owner, and jscpd named the day it needed one** (2026-08-17): the V4 oracle's
/// synthetic-weight RNG and the divergence probe's per-element fold carried the same three
/// lines, and the oracle's own comment had already anticipated it ("chosen over xorshift64*
/// only because format.rs already has that one and the duplication gate is not budgeted").
///
/// The `hash_rows` KERNEL carries a fourth copy in HIP, which no Rust-side factoring can
/// remove — that one is pinned instead, by scoring the kernel against
/// `xor_fold` below in `crates/engine/tests/fwd_kernel.rs`.
pub fn splitmix_finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One element's contribution to an XOR fold over an f32 array's exact BITS: splitmix64's
/// avalanche over `(index, bits)`. PRIVATE — [`xor_fold`] is the only caller, and the device
/// twin corresponds to the whole fold rather than to this step.
///
/// **The index is mixed in, and it is load-bearing twice.** XOR is self-inverse, so without it
/// two elements holding the same bit pattern would cancel out of the fold; and a permutation
/// of the same values would hash identically. The avalanche is what stops two nearby one-ulp
/// differences cancelling instead.
///
/// Element 0 holding `+0.0` folds to 0 and contributes nothing, because 0 is a fixed point of
/// the finalizer (asserted below). Exactly one `(index, bits)` pair has that property, so it
/// costs one collision in 2^64 — recorded because it looks like a bug and is not.
fn xor_fold_step(i: usize, bits: u32) -> u64 {
    splitmix_finalize(((i as u64) << 32) ^ u64::from(bits))
}

/// The whole XOR fold of `x`, on the HOST.
///
/// **This is an ORACLE, and that is why it lives in core rather than beside its caller.** The
/// `hash_rows` HIP kernel computes the same fold on the device; `crates/engine/tests/
/// fwd_kernel.rs::hash_rows_matches_the_host_fold` scores the two against each other, because
/// every conclusion `rivoli_engine::probe`'s divergence log supports is read off a pair of
/// those device hashes, and an instrument nobody checked is a source of confident wrong
/// answers. A fold in HIP is one copy no Rust-side factoring can remove; pinning it is the
/// substitute.
///
/// **XOR, not a sum, and that is the property the whole instrument rests on.** XOR is
/// commutative AND associative, so the device fold is bit-identical whatever order its atomics
/// land in. A float sum would be neither and would report a difference from scheduling jitter
/// alone — an instrument noisier than its subject measures nothing.
pub fn xor_fold(x: &[f32]) -> u64 {
    xor_fold_from(x, 0)
}

/// The XOR fold with the index space OFFSET by `i_base` — how several disjoint buffers fold into
/// one accumulator while staying sensitive to which buffer each element came from.
///
/// Folding N buffers at `i_base = j * n` treats them as one virtual array of length `N*n`. Without
/// the offset the fold mixes only the within-buffer index, so the result is invariant under
/// swapping two buffers' contents — which for `rivoli_engine::probe`'s pool-slot fold would make it
/// blind to a crossed destination, the exact defect class it is aimed at.
pub fn xor_fold_from(x: &[f32], i_base: u64) -> u64 {
    x.iter().enumerate().fold(0u64, |h, (i, v)| {
        h ^ xor_fold_step((i_base + i as u64) as usize, v.to_bits())
    })
}

#[cfg(test)]
mod tests {
    /// The published FNV-1a test vectors: the offset basis for "", and the classic "a"
    /// and "foobar" values — a transcription typo in either constant moves these.
    #[test]
    fn fnv1a_matches_the_published_vectors() {
        assert_eq!(super::fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(super::fnv1a(b"foobar"), 0x85944171f73967e8);
    }

    /// The `u64` form must agree with the byte form over the same bytes — otherwise the two
    /// are separate hashes wearing one name, and a caller that switched between them would
    /// silently invalidate every recorded value.
    #[test]
    fn the_u64_form_agrees_with_the_byte_form() {
        let vals = [0u64, 1, 0xdead_beef_1234_5678, u64::MAX];
        let bytes: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(
            super::fnv1a_u64s(vals.iter().copied()),
            super::fnv1a(&bytes)
        );
        assert_eq!(super::fnv1a_u64s(std::iter::empty()), super::fnv1a(b""));
    }

    /// **0 is a FIXED POINT**, and it is asserted rather than treated as a defect: every term
    /// of the finalizer is a shift-xor or a multiply, so zero maps to zero. That is why
    /// splitmix64 the GENERATOR adds the golden gamma to its state *before* finalizing, and
    /// why `rivoli_engine::probe::fold_step` records that its element-0-is-+0.0 case
    /// contributes nothing to the XOR fold. Written down here because a future reader who
    /// "fixed" it would silently change every recorded hash in the tree.
    #[test]
    fn zero_is_a_fixed_point_of_the_finalizer() {
        assert_eq!(super::splitmix_finalize(0), 0);
    }

    /// The finalizer against splitmix64's **published** output sequence for seed 0: feeding
    /// the generator's first three states must give its first three documented outputs. A
    /// transcription typo in either 64-bit constant fails this.
    ///
    /// Published vectors rather than a self-measured avalanche statistic, which is what this
    /// was first written as (min-bits-flipped == 23 over four bases). That number had to be
    /// measured by this tree, carried by this tree, and re-derived by anyone who touched it —
    /// provenance the algorithm's own published values already have for free, exactly as
    /// `fnv1a_matches_the_published_vectors` above does for FNV (review finding, 2026-08-17).
    /// The XOR fold's three load-bearing properties, WITHOUT A DEVICE.
    ///
    /// `xor_fold` is the oracle the `hash_rows` kernel is scored against — and that scoring is a
    /// GPU test, while CI has no GPU arm, so until this existed the oracle itself was checked
    /// only by whoever remembered to run the device suite (review, 2026-08-17). An unchecked
    /// oracle is the confident-wrong-answer machine the probe's own doc warns about.
    ///
    /// Each assertion pins a property the divergence log's usefulness depends on:
    /// one-ulp sensitivity (or the probe is blind to what it hunts), permutation sensitivity
    /// (what mixing the index in buys, since XOR is self-inverse), and self-inverse cancellation
    /// (which is why `Probe::drain` must re-zero its slab).
    #[test]
    fn xor_fold_is_ulp_sensitive_permutation_sensitive_and_self_inverse() {
        let x: Vec<f32> = (0..1000).map(|i| (i as f32) * 0.5 - 250.0).collect();
        let base = super::xor_fold(&x);
        let mut ulp = x.clone();
        ulp[617] = f32::from_bits(x[617].to_bits() ^ 1);
        assert_ne!(
            super::xor_fold(&ulp),
            base,
            "a one-ULP change must move the fold"
        );
        let mut swapped = x.clone();
        swapped.swap(4, 900);
        assert_ne!(
            super::xor_fold(&swapped),
            base,
            "a permutation of the same values must move the fold"
        );
        // Folding an array into an accumulator twice cancels — the property the probe's re-zero
        // exists for. `xor_fold` starts from 0, so `base ^ base == 0` states it directly.
        assert_eq!(base ^ base, 0);
        // ...and the fold is not trivially zero, which the line above would also satisfy.
        assert_ne!(base, 0);

        // The index OFFSET must matter, or several buffers folding into one accumulator would be
        // invariant under swapping their contents — which is what it exists to prevent.
        assert_ne!(
            super::xor_fold_from(&x, 1),
            base,
            "i_base must change the fold"
        );
        // ...and offsetting is the same as prepending that many elements: folding buffer B at
        // `i_base = n` equals folding `A ++ B` and then XOR-ing out A's own fold. This is the
        // property `probe`'s multi-slot fold relies on, so it is asserted rather than assumed.
        let (a, b) = (&x[..400], &x[400..]);
        assert_eq!(
            super::xor_fold(a) ^ super::xor_fold_from(b, 400),
            base,
            "folding the halves at their true offsets must reconstruct the whole"
        );
    }

    #[test]
    fn splitmix_finalize_matches_the_published_sequence() {
        const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut state = 0u64;
        for want in [
            0xe220_a839_7b1d_cdaf_u64,
            0x6e78_9e6a_a1b9_65f4,
            0x06c4_5d18_8009_454f,
        ] {
            state = state.wrapping_add(GAMMA);
            assert_eq!(super::splitmix_finalize(state), want);
        }
    }
}
