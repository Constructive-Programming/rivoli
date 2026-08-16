//! The Muse Glimmer-30B engine arm: pin (whole-layer placement + streaming slots), the
//! sandwich-normed layer loop, and greedy decode. One arm per architecture — four loop
//! bodies in four files is a measured decision, and a pin parameterised by an arch flag
//! is a GLM-shaped placement path one `if` away from running on the wrong artifact.
//!
//! **Glimmer is DENSE, and that makes it stream MORE than GLM, not less.** There is no
//! router, no expert union and no pool here; what there is instead is a 55.712 GB weight
//! set every token reads in full, partitioned into a resident prefix of whole layers and
//! a streamed suffix. P6 (`old:docs/reference/principles.md`): the pin is a function of
//! free memory, never of architecture — so this arm calls the same
//! [`rivoli_core::residency::partition`] GLM does, over units that happen to be layers
//! instead of experts.
//!
//! **What Glimmer does NOT have, and why nothing here fakes it.** No routed format
//! (`--mode` names how routed experts are stored, and there are none — the weights are
//! bf16), no cache policy (a cyclic dense scan makes every fixed subset of size `k`
//! equally good, so the axis collapses to one answer), no routed-expert trace, no
//! speculative head. `rivoli_core::legality`'s `MuseGlimmer` row is where each of those
//! is said out loud to the user rather than silently ignored.
//!
//! # Where the shape differs from [`crate::glm`], and why the two loops are separate
//!
//! | | glm | glimmer |
//! |---|---|---|
//! | attention | MLA + q-LoRA, fp8 latent KV | GQA 32Q/2KV over bf16, sigmoid output gate |
//! | norms | one rmsnorm form, one eps | centered `(1+w)` per layer + plain at the head, TWO eps |
//! | streamed unit | one routed expert (~2 MB) | one whole layer (967.942 MB) |
//! | fill | io_uring O_DIRECT, ticketed, async | host memcpy from the mmap, synchronous |
//! | prefill batching | `MAXROW` rows of real math | reorder only; every launch stays `m = 1` |
//!
//! Nothing in that table is a parameter of the other loop. What IS shared lives in
//! `rivoli-core` (the partition) and `rivoli-backend` (the launchers), which is the seam
//! the sharing was designed to happen at.

// **`geometry` is NOT gated on the backend, and the other four are.** It is arithmetic over
// a config that touches no device, and CI has no `rocm` job — so anything behind that gate is
// compiled as often as someone runs it here. Its own header carries the argument and the
// `old:` finding that produced it.
pub mod geometry;

#[cfg(feature = "rocm")]
pub mod decode;
#[cfg(feature = "rocm")]
pub mod engine;
#[cfg(feature = "rocm")]
mod forward;
#[cfg(feature = "rocm")]
pub mod pin;
