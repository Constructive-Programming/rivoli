//! # rivoli (cli) — thin entry points
//!
//! `main` shrinks to parse → `Engine::open` → loop; `serve` takes `&mut Engine`
//! and nothing model-shaped. The workspace meta-gates live in this crate's
//! `tests/` (docs registry, CodeScene, and later the invariant registry and
//! kernel census) because cli is the leaf that sees every other crate, and its
//! `build.rs` arms the jscpd duplication gate on every workspace build.
