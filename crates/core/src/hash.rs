//! Content pinning. One owner for the whole workspace — the CodeScene gate's cache and
//! the anchor fixtures' byte pins each carried a copy until jscpd reported the pair on
//! the day the second one was ported (2026-08-15), which is the gate doing its job.

/// FNV-1a over a byte slice. Not a cryptographic claim — it exists so a vendored fixture
/// that changed by one byte cannot pass as the one a doc describes, and adding a sha2
/// dependency to hash test fixtures is the worse trade.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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
}
