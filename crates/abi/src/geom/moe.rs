//! `kernels/moe.hip`'s and `moe_f4.hip`'s geometry: the passes launch `ROWS_PER_BLOCK` waves
//! per block (one wave per expert row) and the drains `ELEMENTWISE_BLOCK` threads, both in
//! `super`; what is specific to the family is the fp4 group a launcher refuses by.

/// Weights per e8m0 scale along an fp4 reduction dim (the `.f4` format's group) — K3's
/// launcher refuses a reduction dim that is not a multiple (1002); the device crate's
/// `dot_nib::F4_GROUP` is asserted equal to it.
pub const F4_GROUP: u32 = 32;
