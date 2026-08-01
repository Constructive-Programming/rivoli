//! The artifact IS the model: reading it, and the formats it is written in.
//!
//! [`format`] is the on-disk layout and its manifest; [`quant`] the int3-vq / int4 / fp8
//! codecs shared by the engine and the `bin/` converters; [`model`] the hyperparameters;
//! [`tokenizer`] the vocabulary and turn framing; [`config`] the run configuration
//! discovered from the machine. See docs/reference/architecture.md §7.

pub mod config;
pub mod format;
pub mod model;
pub mod quant;
pub mod tokenizer;
