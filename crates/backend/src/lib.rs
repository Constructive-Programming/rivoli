//! # rivoli-backend — the waist
//!
//! One backend (HIP/ROCm, gfx1151), one seam: streams, events, signals,
//! `block_on`, the kernel launchers and their ABI wall, and the compiled
//! `kernels/*.hip`. Everything above imports this seam and never a backend
//! symbol; "bytes and spans, no format names" is the proven contract — the old
//! tree's memory substrate crossed the V4 port without one line of change
//! because this boundary held. Owns the `rocm` feature; without it the crate
//! compiles empty and the workspace still builds and tests (that featureless
//! build is a real, CI-tested configuration, not breakage).
