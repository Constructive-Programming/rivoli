//! DSA lightning-indexer constants shared by the host selection path (gpu.rs
//! `dsa_select_layer`) and the device kernels (indexer.hip). The indexer itself
//! runs on device; this module only pins the two magic numbers both sides must
//! agree on.

/// MISA router block size (pooled tokens per running-mean key). MUST match
/// `MISA_BLOCK` in kernels/indexer.hip — the block pool the head router scores over.
pub const MISA_BLOCK: usize = 1024;

/// LayerNorm epsilon for the indexer's `k_norm` (which, unlike every model RMSNorm,
/// ships a bias). Matches the HF reference; distinct from `cfg.rms_norm_eps`.
pub const K_NORM_EPS: f32 = 1e-6;
