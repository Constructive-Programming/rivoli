//! **The anchor fixtures' deterministic parameter draws: torch's CPU generator, transliterated.**
//!
//! The Muse Glimmer anchor kit (`tests/glimmer_anchor_lib.py::init_weights`) fills every
//! parameter from `torch.Generator().manual_seed(seed).uniform_(lo, hi)` with the seed keyed by
//! the parameter's NAME — `sha256(f"{salt}/{name}")[:8]` little-endian, masked to 63 bits. That
//! makes every tensor in the fixture a pure function of its name, which is what lets a Rust
//! consumer REGENERATE the reference's weights instead of vendoring them: the draft goldens ship
//! no weight dump at all (`--dump-weights` is text-mode-only by the driver's own refusal), and
//! this module is how the drafter oracle gets the exact f32 values the reference computed from.
//!
//! Three torch behaviours are load-bearing and each is pinned by a measurement:
//!
//! * `manual_seed` truncates the 64-bit seed to its low 32 bits before the standard MT19937
//!   state expansion. The sha256-derived seeds are 63-bit, so a wrong truncation reads a
//!   different stream — caught by the bit-exactness gate below.
//! * `uniform_` maps each raw draw through its LOW 24 bits: `x = (r & 0xFF_FFFF) / 2^24`,
//!   exact in f32. Verified 2026-08-16 against the pinned venv: MT19937(12345)'s first word is
//!   0xEDFB51E2 and torch's first uniform is `0xFB51E2 / 2^24`.
//! * The affine step is a FUSED `x * (hi - lo) + lo` — a single rounding. Measured 2026-08-16
//!   on seed 5486853808060098981 (`glimmer-anchor-1/draft/encoder.fc.weight`): draws 1 and 3
//!   of the first four disagree with the separately-rounded form and agree with `mul_add`.
//!
//! None of the three is trusted from this comment: the gate in
//! `tests/glimmer_draft_oracle.rs` regenerates every tensor of the vendored
//! `glimmer-anchor-weights-{1,2}.bin` (99 draws + 8 aliases per salt, first-party bytes) and
//! requires bit-identity, so any drift in seeding, stream, masking or rounding fails there
//! before any oracle comparison is believed.
//!
//! Frozen like the rest of this crate: it changes only when the reference's draw changes, and
//! the engine must never call into it on a decode path.

use sha2::{Digest, Sha256};

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The standard MT19937 stream, as `at::mt19937` runs it on the CPU.
///
/// Deliberately NOT a `rand` dependency: the crate ecosystem's `mt19937` wrappers differ in
/// seeding entry points, and the whole value of this type is that its two dozen lines are
/// checkable line-by-line against the reference and against the vendored bytes.
struct Mt19937 {
    state: [u32; N],
    next: usize,
}

impl Mt19937 {
    /// `torch.Generator().manual_seed(seed)`: low 32 bits into word 0, then the standard
    /// Knuth-multiplier expansion.
    fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..N {
            let prev = state[j - 1];
            state[j] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        Self { state, next: N }
    }

    fn refill(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
            self.state[i] = self.state[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.next = 0;
    }

    fn next_u32(&mut self) -> u32 {
        if self.next >= N {
            self.refill();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

/// The seed a parameter's name (already joined to its salt, `"{salt}/{name}"`) hashes to:
/// sha256, first 8 bytes little-endian, masked to 63 bits — `_gen` in the fixture kit.
///
/// `sha2` is a dependency here because the HASH IS THE REFERENCE'S CHOICE, not this crate's: a
/// hand-rolled compression function would put a transliteration risk at the single point every
/// regenerated weight flows through, to save one pure-Rust crate. (The artifact side's "FNV,
/// not a sha256 dep" rule chose the hash; this module has no choice to make.)
pub fn seed_for(salted_name: &str) -> u64 {
    let digest = Sha256::digest(salted_name.as_bytes());
    let mut le = [0u8; 8];
    le.copy_from_slice(&digest[..8]);
    u64::from_le_bytes(le) & ((1u64 << 63) - 1)
}

/// The three families `_draw_into` tells apart by how the owning module APPLIES its weight.
/// Bounds are the fixture kit's, verbatim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// Everything that is not a norm: projections, embeddings, heads.
    Projection,
    /// A plain `x * w` norm, filled near one.
    Norm,
    /// A `(1 + w)` norm, filled near zero — the target's centered family; the drafter has none.
    CenteredNorm,
}

impl Family {
    fn bounds(self) -> (f32, f32) {
        match self {
            Family::Projection => (-0.08, 0.08),
            Family::Norm => (0.8, 1.2),
            Family::CenteredNorm => (-0.2, 0.2),
        }
    }
}

/// `torch.empty(n).uniform_(lo, hi, generator=seeded)`, element for element.
///
/// `x` is exact (a 24-bit integer over a power of two); the affine step is `mul_add` because
/// the reference build fuses it — see the module header's measurement.
pub fn uniform(seed: u64, n: usize, family: Family) -> Vec<f32> {
    let (lo, hi) = family.bounds();
    let mut rng = Mt19937::new(seed);
    (0..n)
        .map(|_| {
            let r24 = rng.next_u32() & 0x00ff_ffff;
            let x = (r24 as f32) / 16_777_216.0;
            x.mul_add(hi - lo, lo)
        })
        .collect()
}

/// One named draw: `uniform` under the seed `"{salt}/{name}"` hashes to.
pub fn draw(salt: &str, name: &str, n: usize, family: Family) -> Vec<f32> {
    uniform(seed_for(&format!("{salt}/{name}")), n, family)
}

#[cfg(test)]
mod tests {
    //! The two torch behaviours the module header states as MEASUREMENTS, gated rather than
    //! asserted in prose. Both were read off the pinned venv 2026-08-16; a number that lives
    //! only in a comment is this repo's most-repeated failure, and the bit-exactness gate in
    //! `tests/glimmer_draft_oracle.rs` catches drift in all three behaviours TOGETHER without
    //! ever saying which one moved.
    use super::{Family, Mt19937, uniform};

    #[test]
    fn the_generator_and_the_low_24_bit_map_are_torchs() {
        assert_eq!(
            Mt19937::new(12345).next_u32(),
            0xEDFB_51E2,
            "MT19937(12345)'s first word"
        );
        // torch's first uniform on that seed, in the (0, 1) mapping `uniform_` applies before
        // the affine step: exact in f32, so `assert_eq!` and not a tolerance.
        let (lo, hi) = Family::Projection.bounds();
        let want = (0x00FB_51E2 as f32 / 16_777_216.0).mul_add(hi - lo, lo);
        assert_eq!(uniform(12345, 1, Family::Projection), vec![want]);
    }

    /// `manual_seed` truncates to the LOW 32 bits, so a 63-bit seed and its low half are the
    /// same stream. Stated in the header as load-bearing; nothing else in the tree pins it,
    /// because every real seed here is already 63-bit and agrees with itself.
    #[test]
    fn the_seed_is_truncated_to_32_bits() {
        let n = 4;
        let low = uniform(12345, n, Family::Norm);
        assert_eq!(uniform((1u64 << 40) | 12345, n, Family::Norm), low);
        assert_ne!(uniform(12346, n, Family::Norm), low);
    }
}
