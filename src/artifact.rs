//! The artifact IS the model: reading it, and the formats it is written in.
//!
//! [`format`] is the on-disk layout and its manifest; [`quant`] the int3-vq / int4 / fp8
//! codecs shared by the engine and the `bin/` converters; [`model`] the hyperparameters;
//! [`tokenizer`] the vocabulary and turn framing; [`config`] the run configuration
//! discovered from the machine. See docs/reference/architecture.md §7.

pub mod config;
/// DeepSeek-V4's prompt encoding. Beside [`tokenizer`] rather than inside it because it is
/// a different job: this produces the STRING, `tokenizer` turns it into ids. It is also the
/// only turn framing in this crate that is a port of executable Python rather than of a
/// Jinja template — the checkpoint ships no `chat_template.jinja` on purpose.
pub mod dsv4_encoding;
pub mod format;
/// Muse Glimmer's prompt encoding, beside [`dsv4_encoding`] and for the same reason: it
/// produces the STRING and [`tokenizer`] turns it into ids. Unlike DeepSeek's this one IS a
/// port of a `chat_template.jinja`, and it is pinned byte-for-byte against that file's own
/// renderer rather than against a reading of it.
pub mod glimmer_encoding;
pub mod model;
pub mod quant;
pub mod tokenizer;
