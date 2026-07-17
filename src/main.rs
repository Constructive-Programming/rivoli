use anyhow::{Context, Result, bail};
use rolibri::config::Config;
use tracing::info;

/// CLI: `rolibri <snapshot-dir> [-bench <tokens>]`. No environment variables,
/// no other flags — everything else is auto-discovered (see config.rs).
fn parse_args() -> Result<(String, Option<usize>)> {
    let mut snapshot = None;
    let mut bench = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-bench" => {
                let n = args.next().context("-bench requires a token count")?;
                bench = Some(n.parse().context("-bench takes an integer")?);
            }
            _ if snapshot.is_none() => snapshot = Some(a),
            _ => bail!("unexpected argument: {a}"),
        }
    }
    let snapshot = snapshot.context("usage: rolibri <snapshot-dir> [-bench <tokens>]")?;
    Ok((snapshot, bench))
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let (snapshot, bench) = parse_args()?;
    let cfg = Config::discover(snapshot, bench)?;

    // Rule 1: the full discovered config is the first line of every run.
    info!("rolibri {} | {cfg}", env!("CARGO_PKG_VERSION"));

    // Decode is synchronous; tokio owns the feed side only. Worker count is
    // the discovered CPU pool size — never the SMT-logical count.
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

    // M0 gate: mmap + index every shard, under 5s.
    let t0 = std::time::Instant::now();
    let snap = rolibri::snapshot::Snapshot::open(&cfg.snapshot)?;
    info!(
        "indexed {} tensors in {:.2}s",
        snap.len(),
        t0.elapsed().as_secs_f64()
    );

    // M0 gate: GPU toolchain is live end-to-end (real launch), or say why not.
    match rolibri::hip::probe() {
        Ok(()) => info!("HIP probe ok — gfx1151 engine live"),
        Err(e) => info!("HIP probe unavailable: {e}"),
    }

    match cfg.bench {
        Some(n) => info!("bench mode ({n} tokens) — decode engine lands in M1"),
        None => bail!("server mode not yet implemented; use -bench <tokens>"),
    }
    Ok(())
}
