//! Shared scaffolding for this crate's examples.
//!
//! Cargo does not build `examples/common/mod.rs` as an example of its own (no `main`), so this is
//! the ordinary way for two examples to share code. It exists because `build.rs`'s duplication gate
//! reported the logging setup the moment a second example wanted it — the two were identical to the
//! token, which is what the gate is for even when the text is boilerplate.

/// Start an example: install the log sink and hand back the arguments.
///
/// Both halves together, not a `logging()` a caller follows with its own `args()`. That split left
/// every example opening with the same four tokens and jscpd reported it — correctly, since the
/// shared thing really was bigger than the shared function admitted. Widening the helper is the
/// honest direction; renaming around the gate is not.
///
/// **stderr, not stdout, and that is load-bearing**: `glm_smoke`'s stdout is the id stream the
/// parity gate diffs, and one log line interleaved into it reads as a token mismatch.
pub fn start() -> Vec<String> {
    logging();
    std::env::args().skip(1).collect()
}

fn logging() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
}
