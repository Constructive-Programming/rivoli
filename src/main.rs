//! rivoli — GLM-5.2 MoE decode engine (routed experts int3-vq/int4/hybrid, default
//! hybrid; see MODES.md). The artifact IS the model: point
//! `rivoli` at a converted artifact directory (manifest.json + codebooks.f32 +
//! resident.safetensors + `L{ll}.vq3` + tokenizer) and it decodes on device.
//!
//! Zero-knob by design — the machine is auto-discovered (see config.rs); the flags
//! below are benchmark/diagnostic overrides, not required configuration.

use anyhow::{Context, Result, bail};
use rivoli::config::Config;
use tracing::info;

/// CLI: `rivoli <model-dir> [-bench <tokens>] [flags]`. `--cache-policy lru|2q|arc|top-m`
/// (default 2q) picks the cache policy — `top-m` additionally routes toward resident
/// experts (`--route-j`/`--route-m`, docs/CACHE_ROUTE.md), which is the one policy that
/// changes the model's output; `--direct-vmm-dma` forces raw DMA over the
/// default pinned-host bounce; `--trace <path>` dumps the routed-expert access trace
/// (v2: demand keys plus the ranked candidate window) for the offline `replay` sim;
/// `--prompt <text>` overrides the bench prompt;
/// `--max-mem <GiB>` sets the device budget literally (no OS reserve — may OOM);
/// without it the budget auto-sizes to `free − 16 GiB`.
struct Args {
    model: String,
    bench: Option<usize>,
    direct_vmm_dma: bool,
    trace: Option<String>,
    prompt: Option<String>,
    cache_policy: String,
    two_q_kin: u32,
    two_q_kout: u32,
    /// `--cache-policy top-m`'s (J, M); ignored by every other policy.
    route: rivoli::hybrid::RouteAdvice,
    /// `--ppl <text-file>`: score that file TEACHER-FORCED and write one NLL per
    /// predicted token to `--ppl-out`, instead of generating. The quality gate for
    /// `top-m` (docs/CACHE_ROUTE.md "Quality"); never a free-running decode.
    ppl: Option<String>,
    ppl_out: Option<String>,
    max_mem: Option<u64>,
    /// Routed-expert format (`--mode int3-vq|int4|hybrid`, default hybrid).
    mode: rivoli::config::Mode,
    /// Attention mode (`--attn auto|dense|streaming|dsa|misa`). `auto` picks `dsa`
    /// when the artifact carries indexer weights, else `dense`.
    attn: String,
    sinks: usize,
    window: usize,
    misa_heads: usize,
    /// `--moe-gain <g>`: scale the whole MoE branch by `g` before the residual add.
    /// An EXPERIMENT knob, not a tuning parameter — see kernels/fwd.hip::vaxpy.
    moe_gain: f32,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    #[cfg(feature = "trace")]
    checksum_x: bool,
}

fn parse_args() -> Result<Args> {
    const USAGE: &str = "usage: rivoli <model-dir> [-bench <tokens>] \
         [--mode int3-vq|int4|hybrid] [--direct-vmm-dma] \
         [--trace <path>] [--prompt <text>] [--cache-policy lru|2q|arc|top-m] [--2q-kin <pct>] \
         [--2q-kout <pct>] [--route-j <n>] [--route-m <n>] [--max-mem <GiB>] \
         [--ppl <text-file>] [--ppl-out <path>] \
         [--attn auto|dense|streaming|dsa|misa] [--sinks <n>] [--window <n>] [--misa-heads <n>] \
         [--moe-gain <g>]";
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
        route: rivoli::hybrid::RouteAdvice::default(),
        ppl: None,
        ppl_out: None,
        max_mem: None,
        mode: rivoli::config::Mode::default(),
        attn: "auto".to_string(),
        sinks: 4,
        window: 8192,
        misa_heads: 8, // the MISA paper's validated GLM setting
        moe_gain: 1.0, // 1.0 = the engine's normal arithmetic, bit-identical (vadd, not vaxpy)
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
            "--mode" => {
                a.mode = rivoli::config::Mode::parse(
                    &args.next().context("--mode requires int3-vq|int4|hybrid")?,
                )?;
            }
            "--trace" => a.trace = Some(args.next().context("--trace requires a path")?),
            "--prompt" => a.prompt = Some(args.next().context("--prompt requires text")?),
            "--cache-policy" => {
                a.cache_policy = args
                    .next()
                    .context("--cache-policy requires lru|2q|arc|top-m")?;
            }
            "--ppl" => a.ppl = Some(args.next().context("--ppl requires a text file")?),
            "--ppl-out" => a.ppl_out = Some(args.next().context("--ppl-out requires a path")?),
            "--route-j" => {
                a.route.j = args
                    .next()
                    .context("--route-j requires an integer")?
                    .parse()
                    .context("--route-j takes an integer")?;
            }
            "--route-m" => {
                a.route.m = args
                    .next()
                    .context("--route-m requires an integer")?
                    .parse()
                    .context("--route-m takes an integer")?;
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
            "--attn" => a.attn = args.next().context("--attn requires a mode")?,
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
            "--moe-gain" => {
                a.moe_gain = args
                    .next()
                    .context("--moe-gain requires a float")?
                    .parse()
                    .context("--moe-gain takes a float")?;
                // A sweep that silently ran at 0 or a negative gain would produce a
                // confidently degenerate arm; the band is generous but finite.
                if !(0.5..=1.5).contains(&a.moe_gain) {
                    bail!("--moe-gain {} outside [0.5, 1.5]", a.moe_gain);
                }
            }
            "--misa-heads" => {
                a.misa_heads = args
                    .next()
                    .context("--misa-heads requires an integer")?
                    .parse()
                    .context("--misa-heads takes an integer")?;
                if a.misa_heads == 0 {
                    bail!("--misa-heads must be >= 1");
                }
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

/// Does the artifact carry resident DSA indexer weights? (`auto` picks `dsa` iff so.)
/// Checks layer 0's `wk` — `indexer_layout` guarantees layer 0 is a full layer, so
/// its indexer is present whenever any is. `open_dir` merges every *.safetensors in
/// the artifact, so this sees `indexer.safetensors` (added post-hoc) too.
fn artifact_has_indexer(model_dir: &str) -> bool {
    rivoli::format::Safetensors::open_dir(model_dir)
        .map(|st| st.has("model.layers.0.self_attn.indexer.wk.weight"))
        .unwrap_or(false)
}

/// Resolve `--attn` (with `auto` → dsa/dense by artifact contents) into an `AttnMode`.
fn resolve_attn(a: &Args) -> Result<rivoli::attn::AttnMode> {
    use rivoli::attn::AttnMode;
    let mode = if a.attn == "auto" {
        if artifact_has_indexer(&a.model) {
            "dsa"
        } else {
            "dense"
        }
    } else {
        a.attn.as_str()
    };
    Ok(match mode {
        "dense" => AttnMode::Dense,
        "streaming" => AttnMode::Streaming {
            sinks: a.sinks,
            window: a.window,
        },
        "dsa" => AttnMode::Dsa,
        "misa" => AttnMode::Misa {
            active_heads: a.misa_heads,
        },
        other => bail!("unknown --attn mode {other:?} (auto|dense|streaming|dsa|misa)"),
    })
}

fn main() -> Result<()> {
    let a = parse_args()?;
    let attn = resolve_attn(&a)?;
    // Bound before `a` is partially moved into `discover` below.
    let (a_ppl, a_ppl_out) = (a.ppl.clone(), a.ppl_out.clone());
    let a_moe_gain = a.moe_gain;
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
        a.route,
        a.max_mem,
        attn,
    )?;
    #[cfg(feature = "trace")]
    {
        cfg.checksum_x = checksum_x;
    }
    cfg.mode = a.mode;
    // Mode-dependent, so it cannot live in `discover` — see Config::validate.
    cfg.validate()?;

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
        mc.n_layers,
        mc.dense_layers,
        mc.hidden,
        mc.n_heads,
        mc.n_experts,
        mc.top_k,
        mc.moe_inter,
        mc.vocab
    );
    // Which tool produced the `.i4` set, and from what — only for the modes that
    // actually read it. int4/hybrid quality numbers are interpretable only against
    // this line; unstamped means an artifact predating provenance, where a
    // `vq3_to_i4` set and an `fp8_to_i4` set are indistinguishable on disk.
    if cfg.mode.uses_int4() {
        match rivoli::format::I4Source::load(&cfg.model)? {
            Some(s) => info!(
                "i4 source: {} ({}, group {}) layers {}..{} from {}",
                s.tool,
                s.chain,
                s.group.map_or("per-row".to_string(), |g| g.to_string()),
                s.layers[0],
                s.layers[1],
                s.src
            ),
            None => info!("i4 source: unstamped (artifact predates i4 provenance)"),
        }
    }
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

    // `--ppl` scores a fixed text instead of generating, so `-bench` is meaningless there.
    let ppl_ids = match &a_ppl {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read --ppl text {path}"))?;
            let ids = tok.encode(&text)?;
            // The corpus is part of the measurement, so its identity goes in the log next
            // to the numbers — a quality claim whose text drifted is worth nothing later.
            info!(
                "ppl corpus {path:?}: {} bytes, {} tokens, fnv1a64 {}",
                text.len(),
                ids.len(),
                rivoli::math::fnv1a64_hex(text.as_bytes()),
            );
            Some(ids)
        }
        None => None,
    };
    let ngen = if ppl_ids.is_some() {
        0
    } else {
        cfg.bench
            .context("server mode not yet implemented; use -bench <tokens>")?
    };

    #[cfg(feature = "rocm")]
    {
        use rivoli::config::OS_RESERVE;
        const GIB: f64 = (1u64 << 30) as f64;
        let (free, _total) = rivoli::device::mem_info()?;
        // `--max-mem` is honoured LITERALLY — no OS reserve; the user asked for that
        // size, so it's allowed to OOM/fail at pin build. The auto path just leaves
        // OS_RESERVE free (there is no hard footprint ceiling — the old NaN cliff was
        // our own bug, since fixed).
        let cap = match cfg.max_mem {
            Some(m) => {
                info!(
                    "device pool budget {:.1} GiB (--max-mem, literal — no reserve; may OOM)",
                    m as f64 / GIB
                );
                m as usize
            }
            None => {
                let cap = free.saturating_sub(OS_RESERVE as usize);
                info!(
                    "device pool budget {:.1} GiB (auto: free {:.1} GiB − {:.0} GiB OS reserve)",
                    cap as f64 / GIB,
                    free as f64 / GIB,
                    OS_RESERVE as f64 / GIB,
                );
                cap
            }
        };
        // dsa/misa need the resident DSA indexer placed by Pin::build.
        let want_indexer = matches!(
            cfg.attn,
            rivoli::attn::AttnMode::Dsa | rivoli::attn::AttnMode::Misa { .. }
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
            cfg.route,
            want_indexer,
            cfg.mode,
        )?;
        info!("pin built in {:.1}s", t.elapsed().as_secs_f64());
        // `--ppl` walks the CORPUS, not the bench prompt, and its `ngen` is 0 — sizing the
        // KV cache from `prompt_ids` would allocate ~17 positions for an ~900-position
        // scoring pass. Take whichever run this actually is.
        let max_ctx = match &ppl_ids {
            Some(ids) => ids.len() + 1,
            None => prompt_ids.len() + ngen + 1,
        };
        let mut engine = rivoli::gpu::GpuEngine::new(pin, &mc, max_ctx, cfg.attn.clone())?;
        #[cfg(feature = "trace")]
        engine.set_checksum_x(cfg.checksum_x);
        engine.set_moe_gain(a_moe_gain);
        // Wedge watchdog: a hung GPU join can't be caught inside the decode loop, so
        // a background thread aborts the process if no token lands for `wd_secs`.
        let wd_secs = std::env::var("RIVOLI_WATCHDOG_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(60);
        engine.set_heartbeat(rivoli::watchdog::spawn(std::time::Duration::from_secs(
            wd_secs,
        ))?);
        // --- `--ppl`: teacher-forced quality gate, not a decode -----------------------
        // Returns before `generate` because the two are mutually exclusive by design:
        // a free-running generation's own trajectory is the confound this measurement
        // exists to remove.
        if let Some(ids) = ppl_ids {
            let t0 = std::time::Instant::now();
            let nlls = engine.nll_forced(&ids)?;
            let dt = t0.elapsed().as_secs_f64();
            let mean = nlls.iter().map(|&v| v as f64).sum::<f64>() / nlls.len().max(1) as f64;
            let (hits, misses) = (engine.hits(), engine.misses());
            let hit_pct = 100.0 * hits as f64 / (hits + misses).max(1) as f64;
            let swap = engine.swap_pct();
            // The (J, M) actually in force, or None when the policy ignores them.
            let top_m = (cfg.cache_policy == "top-m").then_some((cfg.route.j, cfg.route.m));
            info!(
                "PPL: {:.6} (mean NLL {mean:.6} over {} predicted tokens, {dt:.1}s) | \
                 hit {hit_pct:.2}% | swap {}",
                mean.exp(),
                nlls.len(),
                match swap {
                    Some(s) => format!("{s:.2}%"),
                    None => "n/a".to_string(),
                },
            );
            // Per-token NLLs are the actual deliverable: two runs over the same text are
            // PAIRED at every position, and differencing them detects a systematic shift
            // far smaller than the sampling noise in two independent perplexities.
            if let Some(path) = &a_ppl_out {
                use std::io::Write;
                let mut w = std::io::BufWriter::new(
                    std::fs::File::create(path).with_context(|| format!("create {path}"))?,
                );
                writeln!(
                    w,
                    "# rivoli-nll v1 mode={} policy={} moe_gain={a_moe_gain} j={} m={} tokens={} hit_pct={hit_pct:.4} swap_pct={}",
                    cfg.mode,
                    cfg.cache_policy,
                    // `--route-j`/`--route-m` default to (2, 12) whatever the policy, so a
                    // baseline run would otherwise record knobs it never consulted. This
                    // file is the measurement of record; a reader must be able to tell the
                    // baseline from a cell by its header alone — which is also why
                    // `moe_gain` is here: a gain sweep differs in NOTHING else, so six
                    // arms would otherwise write six identical headers and `bin/ppl`
                    // would label them identically.
                    top_m.map_or("na".into(), |(j, _)| j.to_string()),
                    top_m.map_or("na".into(), |(_, m)| m.to_string()),
                    nlls.len(),
                    swap.map_or("na".to_string(), |s| format!("{s:.4}")),
                )?;
                for v in &nlls {
                    writeln!(w, "{v:.8}")?;
                }
                w.flush().context("flush --ppl-out")?;
                info!("wrote {} per-token NLLs to {path}", nlls.len());
            }
            return Ok(());
        }

        let t0 = std::time::Instant::now();
        let (ids, summary) = engine.generate(&prompt_ids, ngen, &tok.eos)?;
        let dt = t0.elapsed().as_secs_f64();
        let (hits, misses) = (engine.hits(), engine.misses());
        info!(
            "GPU: {} tokens in {dt:.1}s ({:.2} tok/s) | expert hit {:.1}% ({hits} hit / {misses} miss)",
            ids.len(),
            summary.tok_per_s,
            summary.hit_pct,
        );
        // OTLP: one decode span carrying the always-on summary (opt-in via
        // OTEL_EXPORTER_OTLP_ENDPOINT; log-only otherwise). Exported synchronously on
        // drop — no async runtime.
        rivoli::telemetry::export_decode(&summary, ids.len());
        info!("{bench_prompt}{}", tok.decode_all(&ids)?);
        Ok(())
    }

    #[cfg(not(feature = "rocm"))]
    {
        let _ = (prompt_ids, ngen, bench_prompt, ppl_ids, a_ppl_out);
        bail!("rivoli was built without the `rocm` feature; rebuild with --features rocm to decode")
    }
}
