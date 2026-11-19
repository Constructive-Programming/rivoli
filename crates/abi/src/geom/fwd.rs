//! `kernels/fwd.hip`'s geometry: the single-block reductions.

use super::F32;

/// `argmax`'s one block: a 256-leaf tree over `(value, index)` pairs.
pub const ARGMAX_BLOCK: u32 = 256;
/// `argmax`'s dynamic LDS, carved `sval[ARGMAX_BLOCK] (f32) | sidx[ARGMAX_BLOCK] (i32)`.
pub const ARGMAX_LDS_BYTES: u32 = 2 * ARGMAX_BLOCK * F32;
const _: () = assert!(ARGMAX_BLOCK.is_power_of_two());

/// `hash_rows`'s block: a 256-leaf XOR tree, one atomic per block.
pub const HASH_BLOCK: u32 = 256;
/// `hash_rows`'s dynamic LDS: one u64 leaf per thread.
pub const HASH_LDS_BYTES: u32 = HASH_BLOCK * 8;
const _: () = assert!(HASH_BLOCK.is_power_of_two());

/// `append_kv`'s scale group: each `APPEND_KV_GROUP` lanes reduce one amax in LDS, so `kvl`
/// is a multiple of it (the launcher refuses otherwise, 1002).
pub const APPEND_KV_GROUP: u32 = 128;
/// `append_kv`'s dynamic LDS for a `kvl`-thread block: one f32 amax leaf per thread.
pub const fn append_kv_lds_bytes(kvl: u32) -> u32 {
    kvl * F32
}
const _: () = assert!(APPEND_KV_GROUP.is_power_of_two());
