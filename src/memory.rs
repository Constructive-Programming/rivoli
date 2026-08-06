//! Where weights LIVE on the device, and what decides which ones stay.
//!
//! [`device`] owns the two VMM allocations (the bump-allocated resident tier and the pool's
//! backing store); [`arena`] is the two-ended byte arena carved out of the second; [`cache`]
//! and [`hybrid`] are the eviction substrate and the three policies over it; [`routed`] is
//! the streaming pool built on all three, shared by both architectures' pins; [`pin`] ties
//! them to the model, holding the resident weight set and resolving a layer's experts to
//! device pointers. See docs/reference/architecture.md §1 and §6.

pub mod arena;
pub mod cache;
pub mod device;
pub mod hybrid;
#[cfg(feature = "rocm")]
pub mod pin;
#[cfg(feature = "rocm")]
pub mod routed;
