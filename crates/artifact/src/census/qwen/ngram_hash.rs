//! The n-gram table's hash parameters, re-derived from the config — the S4 gate
//! `docs/reference/qwen-architecture.md` §5 names as owed.
//!
//! **Why in Rust rather than as a golden.** The three multipliers do not depend on `head_idx` at
//! all, so a golden over them passes on a wrong `global_head_idx` — and the 16 primes are exactly
//! where that error shows. The whole derivation is ~40 lines of integer arithmetic over `u64`, so
//! it belongs beside the converter, which byte-compares its output against the checkpoint's own
//! three I64 buffers rather than trusting either.
//!
//! **The layer index is in BOTH halves of the per-PLE-layer indexing, and stating only one of them
//! is the trap.** `base_seed = seed + 10007 * ple_layer_index` carries it for the multipliers;
//! `global_head_idx = ple_layer_index * ngram_heads + head_idx` carries it for the vocabularies. At
//! index 0 this reproduces the checkpoint byte-exactly and at index 1 it does not — which is the
//! EVIDENCE that this checkpoint's PLE layer index is 0, rather than an assumption about it.

use anyhow::{Result, ensure};

use crate::qwen_config::QwenTextConfig;

/// SplitMix64's golden gamma.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// The reference's default n-gram `seed`, **1234** (CFG:156).
///
/// A CONSTANT and not a config field, on the module header's rule: the shipped `config.json`
/// does not set it, `qwen-architecture.md` §5 records the default without recording the JSON key
/// it would live under, and a serde field whose key this port has not verified against the file
/// is a guess wearing a schema. The mitigation is stronger than the field would be — the
/// converter byte-compares [`ngram_hash`]'s output against the checkpoint's own three I64
/// buffers, so a checkpoint seeded differently REFUSES loudly instead of hashing to the wrong
/// rows. Bind the key when a source states it.
pub const NGRAM_SEED: u64 = 1234;

/// The per-PLE-layer seed stride (`base_seed = seed + 10007 * ple_layer_index`).
const NGRAM_LAYER_STRIDE: u64 = 10007;

/// The n-gram hash's parameters, and the row geometry they imply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NgramHash {
    /// One per position of `ngram_size`, always odd.
    pub multipliers: Vec<u64>,
    /// Per head, the `(global_head_idx + 1)`-th prime strictly greater than
    /// `ngram_vocab_size_base - 1`.
    pub vocab_sizes: Vec<u64>,
    /// Exclusive prefix sums of [`Self::vocab_sizes`].
    pub offsets: Vec<u64>,
    /// `sum(vocab_sizes)` rounded up to `make_ngram_vocab_size_divisible_by`.
    pub padded_rows: u64,
    /// `padded_rows / split_ngram_parts` — one shard's row count.
    pub rows_per_shard: u64,
}

/// SplitMix64, in the standard increment-then-mix form.
///
/// **The increment is INSIDE, and that is measured rather than chosen.** With it, the three
/// multipliers derived at `ple_layer_index = 0` reproduce the checkpoint's I64 `layer_multipliers`
/// buffer byte-exactly. Without it the sequence SHIFTS BY ONE and still contains two of the three
/// correct values in the wrong slots — so a gate checking one multiplier, or checking them as a
/// set, would pass on the wrong form; that near-miss is why [`ngram_hash`] compares all three, in
/// order. The avalanche itself is `rivoli_core::hash::splitmix_finalize`, which owns those three
/// lines for the reason its own doc records; what is HERE is the increment.
fn splitmix64(x: u64) -> u64 {
    rivoli_core::hash::splitmix_finalize(x.wrapping_add(GOLDEN_GAMMA))
}

/// Trial division to `sqrt(n)`: ~2,200 divisions per ~2e7 candidate and a few hundred candidates
/// per call — nothing beside a 172.76 GiB conversion, and it keeps the derivation dependency-free.
fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n.is_multiple_of(2) {
        return n == 2;
    }
    let mut d = 3u64;
    while d.saturating_mul(d) <= n {
        if n.is_multiple_of(d) {
            return false;
        }
        d += 2;
    }
    true
}

/// **The 16 head vocabularies, their offsets and the 3 multipliers, re-derived from the config.**
///
/// `ple_layer_index` is in BOTH halves of the per-layer indexing, and stating only one of them is
/// the trap (`qwen-architecture.md` §5): `base_seed = seed + 10007 * ple_layer_index` carries it
/// for the multipliers, `global_head_idx = ple_layer_index * ngram_heads + head_idx` carries it
/// for the primes. At index 0 this reproduces the checkpoint's three I64 buffers byte-exactly and
/// at index 1 it does not — which is the EVIDENCE that this checkpoint's PLE layer index is 0,
/// rather than an assumption about it.
///
/// Deriving it in Rust rather than vendoring the three multipliers as a golden is the point: the
/// multipliers do not depend on `head_idx` at all, so a golden over them passes on a wrong
/// `global_head_idx` and the 16 primes are where that error shows.
pub fn ngram_hash(cfg: &QwenTextConfig, ple_layer_index: u64) -> Result<NgramHash> {
    let heads = u64::try_from(cfg.ngram_heads()).unwrap_or(0);
    ensure!(heads > 0, "ngram_heads is 0; nothing to derive");
    let vocab = u64::try_from(cfg.vocab).unwrap_or(0);
    ensure!(vocab > 1, "vocab_size {} cannot bound the hash", cfg.vocab);
    // `half_bound = ((2^63 - 1) // vocab_size) // 2`, so `2 * (mix % half_bound) + 1` is odd and
    // stays inside the range a token-id product cannot overflow.
    let half_bound = ((u64::MAX >> 1) / vocab) / 2;
    ensure!(
        half_bound > 0,
        "half_bound is 0 at vocab_size {}",
        cfg.vocab
    );
    let base_seed = NGRAM_SEED.wrapping_add(NGRAM_LAYER_STRIDE.wrapping_mul(ple_layer_index));
    let multipliers = (0..cfg.ngram_size)
        .map(|i| {
            let step = GOLDEN_GAMMA.wrapping_mul(u64::try_from(i + 1).unwrap_or(0));
            2 * (splitmix64(base_seed.wrapping_add(step)) % half_bound) + 1
        })
        .collect();

    // The primes: the `(global_head_idx + 1)`-th prime strictly greater than `base - 1`, i.e.
    // simply the next `heads` primes after `base - 1`, skipped past this layer's offset.
    let base = u64::try_from(cfg.ngram_vocab_size_base).unwrap_or(0);
    ensure!(
        base > 1,
        "ngram_vocab_size_base {} is too small",
        cfg.ngram_vocab_size_base
    );
    let skip = usize::try_from(ple_layer_index * heads).unwrap_or(usize::MAX);
    let vocab_sizes: Vec<u64> = (base..)
        .filter(|&n| is_prime(n))
        .skip(skip)
        .take(usize::try_from(heads).unwrap_or(0))
        .collect();
    ensure!(
        vocab_sizes.len() == usize::try_from(heads).unwrap_or(0),
        "found {} head vocabularies, wanted {heads}",
        vocab_sizes.len()
    );
    let offsets: Vec<u64> = vocab_sizes
        .iter()
        .scan(0u64, |acc, &v| {
            let here = *acc;
            *acc += v;
            Some(here)
        })
        .collect();

    let divisor = u64::try_from(cfg.make_ngram_vocab_size_divisible_by).unwrap_or(0);
    let parts = u64::try_from(cfg.split_ngram_parts).unwrap_or(0);
    ensure!(
        divisor > 0 && parts > 0,
        "make_ngram_vocab_size_divisible_by {divisor} / split_ngram_parts {parts} must both be \
         positive"
    );
    let padded_rows = vocab_sizes.iter().sum::<u64>().div_ceil(divisor) * divisor;
    ensure!(
        padded_rows.is_multiple_of(parts),
        "the padded table is {padded_rows} rows, which does not divide into {parts} shards — the \
         shards are ordered row RANGES and a ragged split makes the last one a different width"
    );
    Ok(NgramHash {
        multipliers,
        vocab_sizes,
        offsets,
        padded_rows,
        rows_per_shard: padded_rows / parts,
    })
}
