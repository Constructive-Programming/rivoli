//! # rivoli-artifact — the artifact IS the model
//!
//! Format discriminators (`Fmt`), per-model config types sharing a validation
//! vocabulary (never a shared struct — that hazard caught two fields living in
//! the wrong struct in the old tree), architecture sniffing (from the weights,
//! never a `--arch` flag; unknown architectures refuse at startup), the
//! tokenizer, and the converters' sealed writer. Host-only and featureless:
//! every test here runs in the featureless CI job.
pub mod arch;
pub mod format;
pub mod glimmer;
pub mod glimmer_config;
/// Muse Glimmer's prompt encoding. A sibling of [`tokenizer`]'s GLM surface rather than a
/// member of it, and the split is the models': GLM's encoder builds a token-ID list, Glimmer's
/// builds a **string** that is tokenized afterwards. The old tree's own header says of that
/// pair that "the two must not converge", so they do not share a module.
pub mod glimmer_encoding;
pub mod glm_config;
pub mod quant;
pub mod schema;
pub mod tokenizer;
