//! rivoli — GLM-5.2 MoE decode engine (int3-vq / int4 / hybrid routed experts).
//! See docs/ARCHITECTURE.md for the module map and MODES.md for the run modes.
pub mod arena;
pub mod attn;
/// The build-time backend switch (`rocm` XOR `vulkan`) — see docs/VULKAN.md.
pub mod backend;
pub mod cache;
pub mod config;
pub mod device;
pub mod format;
pub mod hybrid;
pub mod indexer;
pub mod math;
pub mod model;
pub mod quant;
pub mod telemetry;
pub mod tokenizer;
pub mod watchdog;

#[cfg(feature = "rocm")]
pub mod asyncfetch;
#[cfg(feature = "rocm")]
pub mod gpu;
#[cfg(feature = "rocm")]
pub mod gpustream;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "rocm")]
pub mod pin;
#[cfg(feature = "rocm")]
pub mod stream;

#[cfg(feature = "vulkan")]
pub mod vk;
