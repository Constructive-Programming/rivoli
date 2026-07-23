//! rivoli — int3-vq GLM-5.2 decode engine (rewrite). See docs/ARCHITECTURE.md.
pub mod attn;
pub mod cache;
pub mod config;
pub mod device;
pub mod format;
pub mod indexer;
pub mod math;
pub mod model;
pub mod quant;
pub mod telemetry;
pub mod tokenizer;
pub mod watchdog;

#[cfg(feature = "rocm")]
pub mod gpu;
#[cfg(feature = "rocm")]
pub mod hip;
#[cfg(feature = "rocm")]
pub mod pin;
#[cfg(feature = "rocm")]
pub mod stream;
