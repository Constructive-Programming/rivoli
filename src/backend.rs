//! The compute-backend waist: one implementation, chosen at BUILD time.
//!
//! No trait, no dynamic dispatch — the two backends are never live at once, so a vtable
//! in front of every `launch_*` on the decode hot path would buy nothing.
//!
//! # NOTHING IMPORTS THIS YET. It is a plan, not a seam.
//!
//! An earlier version of this comment said `gpu.rs` and `pin.rs` import `crate::backend::*`
//! so that "swapping HIP for Vulkan is a cargo flag and nothing else." Both halves were
//! false, and the second is the one that would have misled:
//!
//! - **No consumer exists.** `gpu.rs` imports `crate::hip::{...}` and `pin.rs` imports
//!   `crate::hip::memcpy_dtod` directly. Grep for `crate::backend` and this file's own
//!   doc comment is the only hit. Redirecting them is Phase 4 work that has not been done.
//! - **The `launch_*` surface is not the whole boundary.** `gpu.rs` and `asyncfetch.rs`
//!   also import `crate::gpustream::{HipStream, HipEvent, Signal, stream_signal}`
//!   directly, and `HipStream` has NO Vulkan analogue — the engine's two-stream overlap
//!   is a concurrency structure, not a launcher. Re-exporting `vk::*` here would not
//!   supply it. See docs/VULKAN.md, "Phase 4 needs two queues, not one".
//!
//! Left in place because the design is still right and the file is where the switch will
//! land. But an unexercised waist reads as completed integration, which is exactly the
//! kind of false signal this codebase treats as worse than absent code — hence the
//! heading rather than a footnote.

#[cfg(all(feature = "rocm", feature = "vulkan"))]
compile_error!("features `rocm` and `vulkan` are mutually exclusive");

#[cfg(all(feature = "rocm", not(feature = "vulkan")))]
pub use crate::hip::*;

#[cfg(all(feature = "vulkan", not(feature = "rocm")))]
pub use crate::vk::*;
