use anyhow::{Context, Result, bail};
use rivoli::config::Config;
use tracing::info;

/// CLI: `rivoli <snapshot-dir> [-bench <tokens>] [--pre-seed] [--direct-vmm-dma]`.
/// No environment variables — everything else is auto-discovered (see config.rs).
/// `--pre-seed` warms the routed-expert LRU from `.coli_usage` at build time
/// (+~6.8pt on the first tokens, ~23s slower build); default OFF gives a fast
/// build and the LRU self-warms within a few tokens.
/// `--direct-vmm-dma` opts out of the default pinned-host bounce and DMAs cold
/// reads straight into VMM. The bounce is default because it measures ~13% faster
/// (avoids the coherent tax on DMA into host-mapped device pages) and survives NFS
/// sources; use this only to force raw DMA. See stream.hip for the details.
/// `--trace <path>` dumps the routed-expert access trace for the offline cache-
/// policy sim (bin/replay). `--prompt <text>` overrides the fixed bench prompt so
/// diverse, request-like inputs can be traced.
/// DEFAULTS are the validated winning config: `--cache-policy arc` + prefetch on.
/// `--no-prefetch` and `--cache-policy lru|2q` opt out (for A/B benching). Prefetch
/// composes with `--direct-vmm-dma` (sound via the pool's disjointness floor — see
/// Pin::build's slot_floor + prefetch_layer's correctness note).
/// `--max-pool-size <GiB>` overrides the default device-pool cap (config::MAX_POOL,
/// 80 GiB); the resolved pool budget is `min(free − OS_RESERVE, max_pool_size)`.
/// `--direct-io` makes the cold-expert reads use O_DIRECT (bypass the OS page
/// cache); default is buffered reads through the page cache.
/// `--attn dense|streaming|dsa|misa` picks the attention row-selection
/// mechanism (see attn::AttnMode). `--sinks`/`--window` shape streaming mode
/// (defaults 4 / 8192, the StreamingLLM shape). dsa/misa need the `out-idx-*`
/// indexer shard in the snapshot dir.
struct Args {
    snapshot: String,
    bench: Option<usize>,
    pre_seed: bool,
    direct_vmm_dma: bool,
    trace: Option<String>,
    prompt: Option<String>,
    cache_policy: String,
    prefetch: bool,
    prefetch_depth: usize,
    max_pool_size: u64,
    direct_io: bool,
    attn: String,
    sinks: usize,
    window: usize,
}

fn parse_args() -> Result<Args> {
    const USAGE: &str = "usage: rivoli <snapshot-dir> [-bench <tokens>] [--pre-seed] \
         [--direct-vmm-dma] [--trace <path>] [--prompt <text>] [--cache-policy lru|2q|arc] \
         [--no-prefetch] [--prefetch-depth <n>] [--max-pool-size <GiB>] [--direct-io] \
         [--attn dense|streaming|dsa|misa] [--sinks <n>] [--window <n>]";
    let mut snapshot = None;
    // Defaults are the validated winning config: ARC eviction + cross-layer prefetch
    // (best hit% on realistic multi-request sessions + the ~+11% prefetch overlap).
    let mut a = Args {
        snapshot: String::new(),
        bench: None,
        pre_seed: false,
        direct_vmm_dma: false,
        trace: None,
        prompt: None,
        cache_policy: "arc".to_string(),
        prefetch: true,
        prefetch_depth: 2,
        max_pool_size: rivoli::config::MAX_POOL,
        direct_io: false,
        attn: "dense".to_string(),
        sinks: 4,
        window: 8192,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-bench" => {
                let n = args.next().context("-bench requires a token count")?;
                a.bench = Some(n.parse().context("-bench takes an integer")?);
            }
            "--pre-seed" => a.pre_seed = true,
            "--direct-vmm-dma" => a.direct_vmm_dma = true,
            "--trace" => a.trace = Some(args.next().context("--trace requires a path")?),
            "--prompt" => a.prompt = Some(args.next().context("--prompt requires text")?),
            "--cache-policy" => {
                a.cache_policy = args.next().context("--cache-policy requires lru|2q|arc")?;
            }
            "--prefetch" => a.prefetch = true, // default already on; explicit is fine
            "--no-prefetch" => a.prefetch = false,
            "--prefetch-depth" => {
                a.prefetch_depth = args
                    .next()
                    .context("--prefetch-depth requires an integer")?
                    .parse()
                    .context("--prefetch-depth takes an integer")?;
            }
            "--max-pool-size" => {
                let gib: u64 = args
                    .next()
                    .context("--max-pool-size requires a GiB integer")?
                    .parse()
                    .context("--max-pool-size takes an integer number of GiB")?;
                if gib == 0 {
                    bail!("--max-pool-size must be a positive integer number of GiB");
                }
                a.max_pool_size = gib << 30;
            }
            "--direct-io" => a.direct_io = true,
            "--attn" => {
                a.attn = args
                    .next()
                    .context("--attn requires dense|streaming|dsa|misa")?;
            }
            "--sinks" => {
                a.sinks = args
                    .next()
                    .context("--sinks requires an integer")?
                    .parse()
                    .context("--sinks takes an integer")?;
            }
            "--window" => {
                a.window = args
                    .next()
                    .context("--window requires an integer")?
                    .parse()
                    .context("--window takes an integer")?;
            }
            _ if snapshot.is_none() => snapshot = Some(arg),
            _ => bail!("unexpected argument: {arg}\n{USAGE}"),
        }
    }
    a.snapshot = snapshot.context(USAGE)?;
    Ok(a)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let a = parse_args()?;
    let attn = match a.attn.as_str() {
        "dense" => rivoli::attn::AttnMode::Dense,
        "streaming" => rivoli::attn::AttnMode::Streaming {
            sinks: a.sinks,
            window: a.window,
        },
        "dsa" => rivoli::attn::AttnMode::Dsa,
        // 8 active heads of 32 — the MISA paper's validated GLM-5 setting.
        "misa" => rivoli::attn::AttnMode::Misa { active_heads: 8 },
        other => bail!("unknown --attn mode {other:?} (dense|streaming|dsa|misa)"),
    };
    let cfg = Config::discover(
        a.snapshot,
        a.bench,
        a.pre_seed,
        a.direct_vmm_dma,
        a.trace,
        a.prompt,
        a.cache_policy,
        a.prefetch,
        a.prefetch_depth,
        a.max_pool_size,
        a.direct_io,
        attn,
    )?;

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
    let bench_prompt = cfg.prompt.as_deref().unwrap_or("The sky is blue because");
    let tok = rivoli::tokenizer::Tokenizer::load(&cfg.snapshot)?;
    let prompt_ids = tok.encode(bench_prompt)?;
    info!(
        "tokenizer: prompt {bench_prompt:?} -> {} tokens {:?}; eos={:?}",
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
        // The resident GPU path runs dense + streaming (sparse gather kernel);
        // dsa/misa still need the on-device indexer. Fail before the
        // multi-minute pin build, not after.
        if matches!(
            cfg.attn,
            rivoli::attn::AttnMode::Dsa | rivoli::attn::AttnMode::Misa { .. }
        ) {
            bail!(
                "--attn {:?} is scalar-reference only for now (GPU indexer pending); \
                 build without the rocm feature to use the reference engine",
                cfg.attn
            );
        }
        let (free, _total) = rivoli::device::mem_info()?;
        // Budget = the always-resident set (footprint computed from cfg in
        // Pin::build, ~9-10 GiB for GLM-5.2) + the routed-expert LRU. Fill most of
        // device memory: the LRU is online priming, so a bigger pool captures more
        // of this run's working set (sim: 3200 slots→72%, 4200→75% hit) — leave
        // OS_RESERVE headroom for scratch/KV, capped at cfg.max_pool_size
        // (--max-pool-size, default MAX_POOL). The pin splits this into resident tier
        // + LRU. This is the ACTUAL residency budget — the config log carries the two
        // bounds, this line carries the resolved value.
        use rivoli::config::OS_RESERVE;
        const GIB: f64 = (1u64 << 30) as f64;
        let cap = free
            .saturating_sub(OS_RESERVE as usize)
            .min(cfg.max_pool_size as usize);
        info!(
            "device pool budget {:.1} GiB (free {:.1} GiB − {:.0} GiB OS reserve, capped at {:.0} GiB)",
            cap as f64 / GIB,
            free as f64 / GIB,
            OS_RESERVE as f64 / GIB,
            cfg.max_pool_size as f64 / GIB,
        );
        let t = std::time::Instant::now();
        let pin = rivoli::pin::Pin::build(
            &snap,
            &mc,
            &usage,
            cap,
            cfg.pre_seed,
            !cfg.direct_vmm_dma,
            cfg.trace.as_deref(),
            &cfg.cache_policy,
            cfg.prefetch,
            cfg.prefetch_depth,
            cfg.direct_io,
        )?;
        info!("pin built in {:.1}s", t.elapsed().as_secs_f64());
        let max_ctx = prompt_ids.len() + ngen + 1;
        let mut engine = rivoli::gpu::GpuEngine::new(pin, &mc, max_ctx, cfg.attn.clone())?;
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
        if cfg.prefetch {
            let (correct, total) = engine.prefetch_recall();
            info!(
                "prefetch: recall {:.1}% ({correct} of {total} predicted experts selected) \
                 | drain-wait {:.0}ms total ({:.1}ms/tok not hidden)",
                100.0 * correct as f64 / total.max(1) as f64,
                engine.prefetch_wait_ms(),
                engine.prefetch_wait_ms() / ids.len().max(1) as f64,
            );
        }
        info!("{bench_prompt}{}", tok.decode_all(&ids)?);
        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    {
        let mut engine = rivoli::engine::Engine::new(&snap, &mc, cfg.attn.clone())?;
        let t0 = std::time::Instant::now();
        let ids = engine.generate(&prompt_ids, ngen, &tok.eos)?;
        let dt = t0.elapsed().as_secs_f64();
        info!(
            "generated {} tokens in {:.1}s ({:.2} tok/s)",
            ids.len(),
            dt,
            ids.len() as f64 / dt
        );
        info!("{bench_prompt}{}", tok.decode_all(&ids)?);
        Ok(())
    }
}
