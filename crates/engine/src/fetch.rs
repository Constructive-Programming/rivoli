//! Getting cold expert bytes off NVMe and onto the device without the GPU waiting.
//!
//! [`stream`] is the io_uring submit-all/join-once reader and the O_DIRECT alignment math;
//! [`asyncfetch`] is the ticketed dataflow that lets a kernel be enqueued behind bytes that
//! have not landed yet. See docs/reference/architecture.md §3 and §4.
#![cfg(feature = "rocm")]

pub mod asyncfetch;
pub mod stream;
