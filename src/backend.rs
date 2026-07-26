//! The compute-backend waist: one implementation, chosen at BUILD time.
//!
//! `gpu.rs` and `pin.rs` import `crate::backend::*` rather than `crate::hip::*`, so
//! swapping HIP for Vulkan is a cargo flag and nothing else. No trait, no dynamic
//! dispatch — the two backends are never live at once, so a vtable in front of every
//! `launch_*` on the decode hot path would buy nothing. See docs/VULKAN.md.

#[cfg(all(feature = "rocm", feature = "vulkan"))]
compile_error!("features `rocm` and `vulkan` are mutually exclusive");

#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
pub use crate::hip::*;

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
pub use crate::vk::*;
