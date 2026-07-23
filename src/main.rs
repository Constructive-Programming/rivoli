//! rivoli — int3-vq GLM-5.2 decode engine. The artifact IS the model: point
//! `rivoli` at a converted artifact directory (manifest.json + codebooks.f32 +
//! resident.safetensors + `L{ll}.vq3` + tokenizer) and it decodes on device.
//!
//! Zero-knob by design — the machine is auto-discovered (see config.rs); the flags
//! below are benchmark/diagnostic overrides, not required configuration.

use anyhow::{Context, Result, bail};
use rivoli::config::Config;
use tracing::info;

/// CLI: `rivoli <model-dir> [-bench <tokens>] [flags]`. Defaults are the winning
/// cell of the 512-token grid: `--cache-policy 2q` + prefetch on at depth 1 (a
/// sharp joint optimum, not two independent choices). `--no-prefetch` /
/// `--cache-policy lru|arc` opt out. `--direct-vmm-dma` forces raw DMA over the
/// default pinned-host bounce; `--trace <path>` dumps the routed-expert access
/// trace for the offline `replay` sim; `--prompt <text>` overrides the bench
/// prompt; `--max-mem <GiB>` caps the device pool budget.
struct Args {
    model: String,
    bench: Option<usize>,
    direct_vmm_dma: bool,
    trace: Option<String>,
    prompt: Option<String>,
    cache_policy: String,
    two_q_kin: u32,
    two_q_kout: u32,
    prefetch: bool,
    prefetch_depth: usize,
    max_mem: Option<u64>,
    direct_io: bool,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    #[cfg(feature = "trace")]
    checksum_x: bool,
}

fn parse_args() -> Result<Args> {
    const USAGE: &str = "usage: rivoli <model-dir> [-bench <tokens>] [--direct-vmm-dma] \
         [--trace <path>] [--prompt <text>] [--cache-policy lru|2q|arc] [--2q-kin <pct>] \
         [--2q-kout <pct>] [--no-prefetch] [--prefetch-depth <n>] [--max-mem <GiB>] \
         [--buffered-io] [--os-reserve <GiB>]";
    let mut model = None;
    let mut a = Args {
        model: String::new(),
        bench: None,
        direct_vmm_dma: false,
        trace: None,
        prompt: None,
        cache_policy: "2q".to_string(),
        two_q_kin: rivoli::cache::TwoQSplit::default().kin_pct(),
        two_q_kout: rivoli::cache::TwoQSplit::default().kout_pct(),
        prefetch: true,
        prefetch_depth: 1,
        max_mem: None,
        direct_io: true,
        #[cfg(feature = "trace")]
        checksum_x: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-bench" => {
                let n = args.next().context("-bench requires a token count")?;
                a.bench = Some(n.parse().context("-bench takes an integer")?);
            }
            "--direct-vmm-dma" => a.direct_vmm_dma = true,
            "--trace" => a.trace = Some(args.next().context("--trace requires a path")?),
            "--prompt" => a.prompt = Some(args.next().context("--prompt requires text")?),
            "--cache-policy" => {
                a.cache_policy = args.next().context("--cache-policy requires lru|2q|arc")?;
            }
            "--2q-kin" => {
                a.two_q_kin = args
                    .next()
                    .context("--2q-kin requires a percentage")?
                    .parse()
                    .context("--2q-kin takes an integer percentage")?;
            }
            "--2q-kout" => {
                a.two_q_kout = args
                    .next()
                    .context("--2q-kout requires a percentage")?
                    .parse()
                    .context("--2q-kout takes an integer percentage")?;
            }
            "--no-prefetch" => a.prefetch = false,
            "--prefetch-depth" => {
                a.prefetch_depth = args
                    .next()
                    .context("--prefetch-depth requires an integer")?
                    .parse()
                    .context("--prefetch-depth takes an integer")?;
            }
            "--max-mem" => {
                let gib: u64 = args
                    .next()
                    .context("--max-mem requires a GiB integer")?
                    .parse()
                    .context("--max-mem takes an integer number of GiB")?;
                if gib == 0 {
                    bail!("--max-mem must be a positive integer number of GiB");
                }
                a.max_mem = Some(gib << 30);
            }
            "--buffered-io" => a.direct_io = false,
            "--os-reserve" => {
                let gib: u64 = args
                    .next()
                    .context("--os-reserve requires a GiB integer")?
                    .parse()
                    .context("--os-reserve takes an integer number of GiB")?;
                if gib == 0 {
                    bail!("--os-reserve must be a positive integer number of GiB");
                }
                rivoli::config::OS_RESERVE_OVERRIDE
                    .store(gib << 30, std::sync::atomic::Ordering::Relaxed);
            }
            #[cfg(feature = "trace")]
            "--checksum-x" => a.checksum_x = true,
            _ if model.is_none() => model = Some(arg),
            _ => bail!("unexpected argument: {arg}\n{USAGE}"),
        }
    }
    a.model = model.context(USAGE)?;
    Ok(a)
}

fn main() -> Result<()> {
    let a = parse_args()?;
    #[cfg(feature = "trace")]
    let checksum_x = a.checksum_x;
    #[cfg_attr(not(feature = "trace"), allow(unused_mut))]
    let mut cfg = Config::discover(
        a.model,
        a.bench,
        a.direct_vmm_dma,
        a.trace,
        a.prompt,
        a.cache_policy,
        rivoli::cache::TwoQSplit::new(a.two_q_kin, a.two_q_kout)?,
        a.prefetch,
        a.prefetch_depth,
        a.max_mem,
        a.direct_io,
    )?;
    #[cfg(feature = "trace")]
    {
        cfg.checksum_x = checksum_x;
    }

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let version = env!("CARGO_PKG_VERSION");
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into());
    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
    // Rule 1: the full discovered config is the first line of every run.
    info!("rivoli {version} | {cfg}");

    if !std::path::Path::new(&cfg.model).is_dir() {
        bail!("model dir not found: {}", cfg.model);
    }

    // Model dimensions from the artifact's manifest.json.
    let mc = rivoli::model::ModelConfig::load(&cfg.model)?;
    info!(
        "model: {} layers ({} dense) hidden={} heads={} experts={} top{} moe_inter={} vocab={}",
        mc.n_layers, mc.dense_layers, mc.hidden, mc.n_heads, mc.n_experts, mc.top_k, mc.moe_inter, mc.vocab
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

    // Tokenizer (tokenizer.json + generation_config.json, copied into the artifact).
    let bench_prompt = cfg.prompt.as_deref().unwrap_or("The sky is blue because");
    let tok = rivoli::tokenizer::Tokenizer::load(&cfg.model)?;
    let prompt_ids = tok.encode(bench_prompt)?;
    info!(
        "tokenizer: prompt {bench_prompt:?} -> {} tokens {:?}; eos={:?}",
        prompt_ids.len(),
        &prompt_ids[..prompt_ids.len().min(12)],
        tok.eos
    );

    let ngen = cfg
        .bench
        .context("server mode not yet implemented; use -bench <tokens>")?;

    #[cfg(feature = "rocm")]
    {
        use rivoli::config::os_reserve;
        const GIB: f64 = (1u64 << 30) as f64;
        let (free, _total) = rivoli::device::mem_info()?;
        let reserve = os_reserve() as usize;
        // MAX_BUDGET is the load-bearing bound (not the reserve): the pool is sized
        // from MemAvailable, so on a memory-rich boot the reserve alone would size
        // past the driver's durable-backing limit and decode would read back NaN.
        let safe = free
            .saturating_sub(reserve)
            .min(rivoli::config::MAX_BUDGET as usize);
        let cap = match cfg.max_mem {
            Some(m) => safe.min(m as usize),
            None => safe,
        };
        info!(
            "device pool budget {:.1} GiB (free {:.1} GiB − {:.0} GiB OS reserve, capped at {:.0} GiB{})",
            cap as f64 / GIB,
            free as f64 / GIB,
            reserve as f64 / GIB,
            rivoli::config::MAX_BUDGET as f64 / GIB,
            match cfg.max_mem {
                Some(m) => format!(", capped at --max-mem {:.0} GiB", m as f64 / GIB),
                None => String::new(),
            },
        );
        let t = std::time::Instant::now();
        let pin = rivoli::pin::Pin::build(
            &cfg.model,
            &mc,
            cap,
            !cfg.direct_vmm_dma,
            cfg.trace.as_deref(),
            &cfg.cache_policy,
            cfg.two_q,
            cfg.prefetch,
            cfg.prefetch_depth,
        )?;
        info!("pin built in {:.1}s", t.elapsed().as_secs_f64());
        let max_ctx = prompt_ids.len() + ngen + 1;
        let mut engine = rivoli::gpu::GpuEngine::new(pin, &mc, max_ctx)?;
        #[cfg(feature = "trace")]
        engine.set_checksum_x(cfg.checksum_x);
        // Wedge watchdog: a hung GPU join can't be caught inside the decode loop, so
        // a background thread aborts the process if no token lands for `wd_secs`.
        let wd_secs = std::env::var("RIVOLI_WATCHDOG_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        engine.set_heartbeat(rivoli::watchdog::spawn(std::time::Duration::from_secs(
            wd_secs,
        ))?);
        let t0 = std::time::Instant::now();
        let ids = engine.generate(&prompt_ids, ngen, &tok.eos)?;
        let dt = t0.elapsed().as_secs_f64();
        let (hits, misses) = (engine.hits(), engine.misses());
        let tok_per_s = ids.len() as f64 / dt;
        let hit_pct = 100.0 * hits as f64 / (hits + misses).max(1) as f64;
        info!(
            "GPU: {} tokens in {dt:.1}s ({tok_per_s:.2} tok/s) | expert hit {hit_pct:.1}% ({hits} hit / {misses} miss)",
            ids.len(),
        );
        let coll = engine.slot_collisions();
        if coll > 0 {
            info!(
                "slot-reuse collisions caught: {coll} — each would have been a silently \
                 corrupted expert before the guard",
            );
        }
        info!("{bench_prompt}{}", tok.decode_all(&ids)?);
        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    {
        let _ = (prompt_ids, ngen, bench_prompt);
        bail!("rivoli was built without the `rocm` feature; rebuild with --features rocm to decode")
    }
}
