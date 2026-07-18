use anyhow::{Context, Result, bail};
use rivoli::config::Config;
use tracing::info;

/// CLI: `rivoli <snapshot-dir> [-bench <tokens>]`. No environment variables,
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
    let snapshot = snapshot.context("usage: rivoli <snapshot-dir> [-bench <tokens>]")?;
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
    info!("rivoli {} | {cfg}", env!("CARGO_PKG_VERSION"));

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

    // Model dimensions from config.json.
    let mc = rivoli::model::ModelConfig::load(&cfg.snapshot)?;
    info!(
        "model: {} layers ({} dense) hidden={} heads={} experts={} top{} moe_inter={} vocab={}",
        mc.n_layers,
        mc.dense_layers,
        mc.hidden,
        mc.n_heads,
        mc.n_experts,
        mc.top_k,
        mc.moe_inter,
        mc.vocab
    );
    info!(
        "mla: q_lora={} kv_lora={} qk={}+{} v_head={} rope_theta={}",
        mc.q_lora_rank,
        mc.kv_lora_rank,
        mc.qk_nope_head_dim,
        mc.qk_rope_head_dim,
        mc.v_head_dim,
        mc.rope_theta()
    );

    // Tokenizer (tokenizer.json). Round-trip the fixed bench prompt as a
    // liveness check. Bench input is fixed by design — it's a benchmark, not a
    // knob; real prompts arrive via the server API (later).
    const BENCH_PROMPT: &str = "The sky is blue because";
    let tok = rivoli::tokenizer::Tokenizer::load(&cfg.snapshot)?;
    let prompt_ids = tok.encode(BENCH_PROMPT)?;
    info!(
        "tokenizer: prompt {BENCH_PROMPT:?} -> {} tokens {:?}; eos={:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(12)],
        tok.eos
    );

    // M0 gate: mmap + index every shard, under 5s.
    let t0 = std::time::Instant::now();
    let snap = rivoli::snapshot::Snapshot::open(&cfg.snapshot)?;
    info!(
        "indexed {} tensors in {:.2}s",
        snap.len(),
        t0.elapsed().as_secs_f64()
    );

    // Expert usage ranking (drives the pin) — a primed, READ-ONLY profiling
    // artifact (colibri's .coli_usage). Priming is a deliberate offline pass, not
    // a side effect of inference; the decode path never writes it.
    let usage = rivoli::usage::Usage::load(&cfg.snapshot)?;
    info!(
        "usage: {} selections over {} (layer,expert) pairs",
        usage.total_selections(),
        usage.counts.len()
    );

    // M0 gate: GPU toolchain is live end-to-end (real launch), or say why not.
    match rivoli::hip::probe() {
        Ok(()) => info!("HIP probe ok — gfx1151 engine live"),
        Err(e) => info!("HIP probe unavailable: {e}"),
    }

    let ngen = cfg
        .bench
        .context("server mode not yet implemented; use -bench <tokens>")?;

    // M3: resident GPU decode. Build the pin (auto-sized tier), then decode on
    // device. Falls back to the scalar reference path without the `rocm` feature.
    #[cfg(feature = "rocm")]
    {
        let (free, _total) = rivoli::device::mem_info()?;
        // Leave 6 GiB free; cap the tier (48 GiB proven safe for one-shot alloc).
        let cap = free.saturating_sub(6 << 30).min(48 << 30);
        let t = std::time::Instant::now();
        let pin = rivoli::pin::Pin::build(&snap, &mc, &usage, cap)?;
        info!(
            "pin: {:.1} GiB resident, {} routed experts, built in {:.1}s",
            pin.used() as f64 / (1u64 << 30) as f64,
            pin.pinned_experts(),
            t.elapsed().as_secs_f64()
        );
        let max_ctx = prompt_ids.len() + ngen + 1;
        let mut engine = rivoli::gpu::GpuEngine::new(pin, &mc, max_ctx)?;
        let t0 = std::time::Instant::now();
        let ids = engine.generate(&prompt_ids, ngen, &tok.eos)?;
        let dt = t0.elapsed().as_secs_f64();
        let (hits, misses) = (engine.hits(), engine.misses());
        info!(
            "GPU: {} tokens in {:.1}s ({:.2} tok/s) | expert hit {:.1}% ({hits} hit / {misses} miss)",
            ids.len(),
            dt,
            ids.len() as f64 / dt,
            100.0 * hits as f64 / (hits + misses).max(1) as f64,
        );
        info!("{BENCH_PROMPT}{}", tok.decode_all(&ids)?);
        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    {
        let mut engine = rivoli::engine::Engine::new(&snap, &mc);
        let t0 = std::time::Instant::now();
        let ids = engine.generate(&prompt_ids, ngen, &tok.eos)?;
        let dt = t0.elapsed().as_secs_f64();
        info!(
            "generated {} tokens in {:.1}s ({:.2} tok/s)",
            ids.len(),
            dt,
            ids.len() as f64 / dt
        );
        info!("{BENCH_PROMPT}{}", tok.decode_all(&ids)?);
        Ok(())
    }
}
