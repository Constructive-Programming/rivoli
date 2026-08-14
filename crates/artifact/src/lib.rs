//! # rivoli-artifact — the artifact IS the model
//!
//! Format discriminators (`Fmt`), per-model config types sharing a validation
//! vocabulary (never a shared struct — that hazard caught two fields living in
//! the wrong struct in the old tree), architecture sniffing (from the weights,
//! never a `--arch` flag; unknown architectures refuse at startup), the
//! tokenizer, and the converters' sealed writer. Host-only and featureless:
//! every test here runs in the featureless CI job.
