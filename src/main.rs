use anyhow::{Context, Result, bail};
use rolibri::config::Config;
use tracing::info;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let snapshot = std::env::args()
        .nth(1)
        .context("usage: rolibri <snapshot-dir>")?;
    let cfg = Config::from_env(snapshot);

    // Rule 1 of the colibri campaign: the full config is the first line of
    // every run — a benchmark whose parameters aren't in its log never happened.
    info!("rolibri {} | {cfg}", env!("CARGO_PKG_VERSION"));

    // Decode is synchronous; tokio owns the feed side only. Worker count is
    // the CPU pool size — never the SMT-logical count (see PLAN.md evidence).
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(cfg.threads)
        .enable_all()
        .build()
        .context("tokio runtime")?;

    rt.block_on(run(cfg))
}

async fn run(cfg: Config) -> Result<()> {
    if !std::path::Path::new(&cfg.snapshot).is_dir() {
        bail!("snapshot dir not found: {}", cfg.snapshot);
    }
    // M0: snapshot mmap + tensor index land here (PLAN.md).
    info!("M0 skeleton: snapshot dir present; engine not yet implemented");
    Ok(())
}
