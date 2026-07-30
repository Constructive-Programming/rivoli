//! rivoli — GLM-5.2 MoE decode engine (routed experts int3-vq/int4/hybrid; the default is
//! hybrid on `rocm` and int3-vq on `vulkan`, whose int4 kernels are not ported — see
//! MODES.md and `config::Mode`). The artifact IS the model: point
//! `rivoli` at a converted artifact directory (manifest.json + codebooks.f32 +
//! resident.safetensors + `L{ll}.vq3` + tokenizer) and it decodes on device.
//!
//! Zero-knob by design — the machine is auto-discovered (see config.rs); the flags
//! below are benchmark/diagnostic overrides, not required configuration.

use anyhow::{Result, bail};

// The decode path's imports and helpers. A featureless build compiles this file down to
// the refusal stub in `main` (see there for why the binary still builds at all), so
// everything only the real `main` reaches is gated with it rather than left to warn.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
use anyhow::Context;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
use rivoli::config::Config;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
use tracing::info;

// NOTE: doc comments on this struct and its fields are USER-FACING — clap renders them as
// `--help`. Rationale for the code goes in `//` comments like this one, which clap ignores.
//
// `clap` derives the parse, the help text and the range checks from the struct. The
// hand-rolled loop it replaced was ~150 lines of
// `args.next().context(..)?.parse().context(..)?`, one block per flag, plus a `USAGE`
// const maintained separately from the flags it described — and by the end no longer
// matching them (it never mentioned `--misa-heads`, `--checksum-x` or `--2q-*`'s bounds).
//
// No `wrap_help` feature, so clap does not re-wrap: the terminal does. That also means
// `max_term_width` would be inert, which is why it is not set.
/// Zero-knob by design: every flag is a benchmark or diagnostic override, and a bare
/// `rivoli <model-dir>` is a complete invocation.
#[derive(clap::Parser, Debug)]
#[command(name = "rivoli", about = "GLM-5.2 int3-vq MoE decode engine (HIP/ROCm or Vulkan)")]
struct Args {
    /// The converted artifact directory (manifest.json + codebooks + resident.safetensors
    /// + L{ll}.vq3 + tokenizer). The artifact IS the model.
    model: String,

    /// Decode this many tokens, print PROFILE, exit. Omit for the interactive/server path.
    ///
    /// Spelled `-bench` in every recorded command line (benchmarks.md, MODES.md,
    /// tests/bench-matrix.sh); `main` rewrites that single-dash form to `--bench` before
    /// clap sees it, since clap has no single-dash-long concept. Both work.
    #[arg(long)]
    bench: Option<usize>,

    /// Routed-expert format. Default: hybrid on rocm, int3-vq on vulkan (whose int4
    /// kernels are not ported) — see MODES.md and `config::Mode`.
    #[arg(long, default_value_t, value_parser = rivoli::config::Mode::parse)]
    mode: rivoli::config::Mode,

    /// Routed-expert cache policy. All three are output-neutral: routing never consults
    /// residency (see math.rs `inv_1_routing_never_consults_the_cache`), so a policy change
    /// moves throughput and hit rate, never the tokens produced.
    #[arg(long, default_value = "2q", value_parser = ["lru", "2q", "arc"])]
    cache_policy: String,

    /// 2Q's A1in probation bound, percent of pool capacity. Ignored by lru/arc.
    #[arg(long = "2q-kin", value_name = "PCT", default_value_t = rivoli::cache::TwoQSplit::default().kin_pct())]
    two_q_kin: u32,

    /// 2Q's A1out ghost bound, percent of pool capacity. May exceed 100 (the ghost holds
    /// keys only). Ignored by lru/arc.
    #[arg(long = "2q-kout", value_name = "PCT", default_value_t = rivoli::cache::TwoQSplit::default().kout_pct())]
    two_q_kout: u32,



    /// Device budget in GiB, taken LITERALLY — no OS reserve, so it may OOM at build.
    /// Without it the budget auto-sizes to `free − 16 GiB`.
    #[arg(long, value_name = "GIB", value_parser = clap::value_parser!(u64).range(1..))]
    max_mem: Option<u64>,

    /// Attention row selection. `auto` picks `dsa` when the artifact carries indexer
    /// weights AND the backend has the indexer kernels, else `dense` — see `resolve_attn`.
    #[arg(long, default_value = "auto", value_parser = ["auto", "dense", "streaming", "dsa", "misa"])]
    attn: String,

    /// `--attn streaming`: the number of leading sink tokens kept.
    #[arg(long, value_name = "N", default_value_t = 4)]
    sinks: usize,

    /// `--attn streaming`: the trailing window kept.
    #[arg(long, value_name = "N", default_value_t = 8192)]
    window: usize,

    /// `--attn misa`: how many indexer heads score tokens (the MISA paper's validated GLM
    /// setting is 8).
    #[arg(long, value_name = "N", default_value_t = 8, value_parser = clap::value_parser!(u16).range(1..))]
    misa_heads: u16,

    /// Opt OUT of the pinned-host bounce: DMA cold reads straight into VMM device memory.
    /// The bounce is the default because it measures faster and survives kernels whose
    /// amdgpu path EFAULTs on direct io_uring DMA into VMM (src/stream.rs).
    #[arg(long)]
    direct_vmm_dma: bool,

    /// How many top-ranked LOOKA predictions per horizon (L+1 and L+2) to feed the cache
    /// policy as eviction vetoes. ALWAYS ON; pass 0 to disable. Runs the target layer's
    /// router against this layer's post-attention residual; measured precision by rank is
    /// 99/96/93/87/78/67/55/42% (docs/CACHE_PILOT.md). A hint only vetoes eviction — it
    /// never changes routing — so output is bit-identical at every value.
    #[arg(long, value_name = "K", default_value_t = rivoli::gpu::DEFAULT_HINT_K)]
    hint_k: usize,

    /// Dump the routed-expert access trace (v2: demand keys plus the ranked candidate
    /// window) for the offline `replay` sim.
    #[arg(long, value_name = "PATH")]
    trace: Option<String>,

    /// Override the fixed bench prompt, for capturing traces of diverse inputs.
    #[arg(long, value_name = "TEXT")]
    prompt: Option<String>,

    /// Score this file TEACHER-FORCED and write one NLL per predicted token to --ppl-out
    /// instead of generating. The quality instrument for any change that CAN move output
    /// (a format or a kernel); never a free-running decode. Cache changes no longer need it
    /// — routing is residency-blind, so they are output-neutral by construction.
    #[arg(long, value_name = "TEXT_FILE", requires = "ppl_out")]
    ppl: Option<String>,

    /// Where `--ppl` writes its per-token NLLs.
    #[arg(long, value_name = "PATH")]
    ppl_out: Option<String>,

    /// Scale the whole MoE branch by `g` before the residual add. An EXPERIMENT knob, not
    /// a tuning parameter (kernels/fwd.hip::vaxpy). The band is generous but finite: a
    /// sweep that silently ran at 0 or a negative gain would produce a confidently
    /// degenerate arm. g = 1.0 is bit-identical to the plain vadd.
    #[arg(long, value_name = "G", default_value_t = 1.0, value_parser = moe_gain_in_band)]
    moe_gain: f32,

    /// Write the generated token ids, one per line.
    ///
    /// For **gate A** of the Vulkan acceptance gate (docs/VULKAN.md): agreement on token
    /// IDs for K tokens. Comparing decoded TEXT is not a substitute — different id
    /// sequences can decode to identical text, so a text diff reports only a lower bound
    /// on divergence. Gate A is a standing obligation across commits, so it needs an
    /// instrument rather than an eyeball.
    #[arg(long, value_name = "PATH")]
    dump_ids: Option<String>,

    /// Encode the prompt RAW, with no GLM turn framing. Reproduces every benchmark number
    /// recorded before templating existed — and those runs could never stop, so this is
    /// also how you re-measure the degeneration it caused.
    #[arg(long)]
    raw_prompt: bool,

    /// DIAGNOSTIC: hash the residual stream after every layer.
    #[cfg(feature = "trace")]
    #[arg(long)]
    checksum_x: bool,
}

/// `--moe-gain`'s band. A hand-written parser because clap's `range` value_parser is
/// integers only, and an unchecked float here is the failure mode the band exists for.
fn moe_gain_in_band(s: &str) -> Result<f32, String> {
    let g: f32 = s.parse().map_err(|e| format!("{s:?} is not a float: {e}"))?;
    if (0.5..=1.5).contains(&g) {
        Ok(g)
    } else {
        Err(format!("{g} is outside [0.5, 1.5]"))
    }
}

/// Parse argv, accepting the legacy single-dash `-bench` alongside `--bench`.
///
/// clap has no single-dash-long form, and `-bench N` is what every recorded command line
/// in benchmarks.md, MODES.md and tests/bench-matrix.sh uses. Rewriting the token is three
/// lines; re-recording a year of benchmark provenance is not. Only an exact `-bench`
/// matches, so `-b`, `--bench` and a positional path are all untouched.
fn parse_args() -> Args {
    use clap::Parser;
    let argv = std::env::args().map(|a| if a == "-bench" { "--bench".into() } else { a });
    Args::parse_from(argv)
}

#[cfg(any(feature = "rocm", feature = "vulkan"))]
/// Does the artifact carry resident DSA indexer weights? (`auto` picks `dsa` iff so AND the
/// backend has the indexer kernels — see [`resolve_attn`].)
/// Checks layer 0's `wk` — `indexer_layout` guarantees layer 0 is a full layer, so
/// its indexer is present whenever any is. `open_dir` merges every *.safetensors in
/// the artifact, so this sees `indexer.safetensors` (added post-hoc) too.
fn artifact_has_indexer(model_dir: &str) -> bool {
    rivoli::format::Safetensors::open_dir(model_dir)
        .map(|st| st.has("model.layers.0.self_attn.indexer.wk.weight"))
        .unwrap_or(false)
}

/// Resolve `--attn` (with `auto` → dsa/dense by artifact contents) into an `AttnMode`.
///
/// `auto` ALSO asks what the backend can run. The Vulkan build has none of the five DSA
/// indexer kernels, so `auto` resolving to `dsa` on an artifact that carries indexer weights
/// made a bare `rivoli <model>` fail on its own default — the second of the two ways it did.
/// `cfg!` rather than `#[cfg]` so both arms keep compiling on both backends and
/// `artifact_has_indexer` is still called (and still reported) either way: the reason for
/// the choice is worth logging, since "auto picked dense" is otherwise indistinguishable
/// from "this artifact has no indexer".
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn resolve_attn(a: &Args) -> Result<rivoli::attn::AttnMode> {
    use rivoli::attn::AttnMode;
    let mode = if a.attn == "auto" {
        match (artifact_has_indexer(&a.model), cfg!(feature = "vulkan")) {
            (true, false) => "dsa",
            (true, true) => {
                eprintln!(
                    "--attn auto: the artifact carries DSA indexer weights, but this is a \
                     Vulkan build and the lightning-indexer kernels are not ported \
                     (docs/VULKAN.md) — resolving to `dense`. Use --features rocm for dsa."
                );
                "dense"
            }
            (false, _) => "dense",
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
            active_heads: a.misa_heads as usize,
        },
        other => bail!("unknown --attn mode {other:?} (auto|dense|streaming|dsa|misa)"),
    })
}

/// A build with NO compute backend refuses at the door.
///
/// The binary still compiles featureless on purpose — `cargo test` and the featureless
/// clippy pass build every target, and `backend.rs` records why breaking those is not an
/// option: they are what keeps the backend-independent half (config, math, quant, arena,
/// cache, telemetry) honest. So this is a runtime refusal rather than a `compile_error!`,
/// and it is the FIRST thing that happens: the old version discovered memory, loaded the
/// manifest, built the tokenizer and encoded the prompt before admitting it could not
/// decode.
///
/// `parse_args` still runs, so `--help` and flag validation work in a featureless build —
/// which is the one useful thing it can do.
#[cfg(not(any(feature = "rocm", feature = "vulkan")))]
fn main() -> Result<()> {
    let _ = parse_args();
    bail!(
        "rivoli was built with NO compute backend and cannot decode. Rebuild with \
         `--features rocm` (HIP/ROCm) or `--features vulkan` — exactly one; they are \
         mutually exclusive (src/backend.rs, docs/VULKAN.md)."
    )
}

/// A build with no compute backend cannot reach this: see the stub above.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn main() -> Result<()> {
    let a = parse_args();
    let attn = resolve_attn(&a)?;
    // Bound before `a` is partially moved into `discover` below.
    let (a_ppl, a_ppl_out) = (a.ppl.clone(), a.ppl_out.clone());
    let a_moe_gain = a.moe_gain;
    let a_raw_prompt = a.raw_prompt;
    let a_dump_ids = a.dump_ids.clone();
    // Bound here for the OTLP root span's run-identity attributes: `a` is partially
    // moved into `discover` below, and `cfg` does not keep all of these verbatim.
    let a_model = a.model.clone();
    let a_cache_policy = a.cache_policy.clone();
    let a_attn = format!("{attn:?}").to_lowercase();
    let (a_max_mem, a_bench) = (a.max_mem, a.bench);
    let (a_2q_kin, a_2q_kout) = (a.two_q_kin, a.two_q_kout);
    let (a_sinks, a_window, a_misa_heads) = (a.sinks, a.window, a.misa_heads as usize);
    #[cfg(feature = "trace")]
    let checksum_x = a.checksum_x;
    #[cfg_attr(not(feature = "trace"), allow(unused_mut))]
    let mut cfg = Config::discover(
        a.model,
        a.bench,
        a.direct_vmm_dma,
        a.hint_k,
        a.trace,
        a.prompt,
        a.cache_policy,
        rivoli::cache::TwoQSplit::new(a.two_q_kin, a.two_q_kout)?,
        // GiB -> BYTES. The flag is documented and value_named in GiB and every consumer
        // (`Config::max_mem`, the pool cap, the log line's `/ GIB`) is in bytes. The clap
        // migration dropped this multiply, so `--max-mem 115` asked for 115 BYTES and the
        // pool came out at "0.0 GiB (~0 slots)" before failing in `rivoli_vmm_alloc(0)`.
        // Loud rather than silent, but it invalidates any run that passed the flag.
        a.max_mem.map(|g| g.saturating_mul(1 << 30)),
        attn,
    )?;
    #[cfg(feature = "trace")]
    {
        cfg.checksum_x = checksum_x;
    }
    cfg.mode = a.mode;
    // Mode-dependent, so it cannot live in `discover` — see Config::validate.
    cfg.validate()?;
    // Two DIFFERENT questions, asked together: is this configuration coherent (above), and
    // does the backend this binary was built with have the kernels it needs (here)? The
    // second is a no-op under `rocm` and refuses `--mode int4|hybrid` / `--attn dsa|misa`
    // under `vulkan`. Kept separate so `validate`'s own tests stay backend-independent —
    // see Config::validate_backend.
    cfg.validate_backend()?;
    // `--moe-gain` is the one backend-gated knob that is NOT a `Config` field, so it is
    // checked here rather than in `validate_backend` with the others. g == 1 takes the
    // ported `vadd`; anything else takes `vaxpy`, which the Vulkan backend defers. Rejected
    // at startup for the same reason as the modes: the alternative is discovering it forty
    // layers into the first token.
    #[cfg(feature = "vulkan")]
    if a_moe_gain != 1.0 {
        bail!(
            "--moe-gain {a_moe_gain} needs the `vaxpy` kernel, which the Vulkan backend does \
             not have (docs/VULKAN.md defers it; `vadd`, the g = 1 case, is ported). Drop the \
             flag, or rebuild with --features rocm."
        );
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
    // The group size is not decoration: the engine indexes `scale[o·ngroups + i/G]`,
    // so reading a set quantized at a different G walks a differently-shaped array.
    // That does not fault — it silently yields garbage (a per-row set read as G=128
    // produced rel_l2=NaN, gain=NaN in the ground-truth oracle), which is the failure
    // mode a stamp exists to prevent. Refuse rather than decode nonsense.
    if cfg.mode.uses_int4() {
        match rivoli::format::I4Source::load(&cfg.model)? {
            Some(s) => {
                info!(
                    "i4 source: {} ({}, group {}) layers {}..{} from {}",
                    s.tool,
                    s.chain,
                    s.group.map_or("per-row".to_string(), |g| g.to_string()),
                    s.layers[0],
                    s.layers[1],
                    s.src
                );
                let want = rivoli::quant::I4_GROUP;
                match s.group {
                    Some(g) if g == want => {}
                    Some(g) => anyhow::bail!(
                        "`.i4` set was quantized at group {g}, this binary reads group {want}. \
                         Rebuild with `fp8_to_i4`, or run a binary built for group {g}."
                    ),
                    None => anyhow::bail!(
                        "`.i4` set is per-row (pre-group provenance), this binary reads group \
                         {want}. Rebuild with `fp8_to_i4`."
                    ),
                }
            }
            None => anyhow::bail!(
                "`.i4` set is unstamped so its group size is unknown, and this binary reads \
                 group {}. Rebuild with `fp8_to_i4`, or use `--mode int3-vq`.",
                rivoli::quant::I4_GROUP
            ),
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
    // Chat framing by default: raw text leaves the model outside any assistant turn, so
    // its EOS ids (two of which are turn boundaries) are unreachable and decode runs to
    // the token limit every time. `--ppl` deliberately does NOT go through here — it
    // scores a fixed corpus, not a turn.
    let prompt_ids = if a_raw_prompt {
        tok.encode(bench_prompt)?
    } else {
        tok.encode_chat(bench_prompt)?
    };
    info!(
        "tokenizer: prompt {bench_prompt:?} -> {} tokens{} {:?}; eos={:?}",
        prompt_ids.len(),
        if a_raw_prompt { " RAW (no chat framing — decode cannot stop on EOS)" } else { " chat-framed" },
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

    #[cfg(any(feature = "rocm", feature = "vulkan"))]
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
        engine.set_hint_k(cfg.hint_k);
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
            info!(
                "PPL: {:.6} (mean NLL {mean:.6} over {} predicted tokens, {dt:.1}s) | \
                 hit {hit_pct:.2}%",
                mean.exp(),
                nlls.len(),
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
                    "# rivoli-nll v1 mode={} policy={} moe_gain={a_moe_gain} tokens={} hit_pct={hit_pct:.4}",
                    cfg.mode,
                    cfg.cache_policy,
                    // `moe_gain` is in the header because a gain sweep differs in NOTHING
                    // else: six arms would otherwise write six identical headers and
                    // `bin/ppl` would label them identically. `bin/ppl` labels cells with
                    // `split_whitespace().take(3)`, so the first three fields are the label.
                    nlls.len(),
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
        // Degeneration check. A looped generation's tok/s is an ARTIFACT — it routes to
        // a handful of experts, inflates the hit rate, and therefore benchmarks FAST —
        // so it must be surfaced loudly, not left for someone to spot in the text. Three
        // verbatim repeats of a block up to 64 tokens long; prose does not do that.
        let degenerate = rivoli::telemetry::detect_loop(&ids, 3, 64);
        // The restart signal. A tail cycle is LATE degeneration; a run that answers and
        // then answers again is already broken and may have no cycle at all. Threshold:
        // an eighth of the generation, floor 32 tokens. Healthy prose repeats phrases,
        // not paragraphs.
        let lrb = rivoli::telemetry::longest_repeated_block(&ids);
        let lrb_bar = std::cmp::max(32, ids.len() / 8);
        if degenerate.is_none() && lrb >= lrb_bar {
            tracing::warn!(
                "SUSPECT OUTPUT: the longest block repeated in {} generated tokens is {lrb} \
                 tokens (bar {lrb_bar}) with no verbatim tail cycle — the shape of a \
                 RESTART rather than a loop. tok/s here is still suspect: re-answering \
                 re-routes to the same experts, which inflates the hit rate.",
                ids.len(),
            );
        }
        // Structural repetition, the signal the two exact-matching detectors above are
        // blind to. Both passed a run that was 329 repetitions of "**Memory Product.**"
        // with a varying label, so this is not a belt-and-braces addition — it is the
        // detector that actually works on the common failure shape.
        let text = tok.decode_all(&ids).unwrap_or_default();
        let rep = rivoli::telemetry::repetition_report(&text);
        tracing::info!(
            "generation: {} tokens, longest repeated block {lrb}, top-line x{}, distinct {:.3}",
            ids.len(),
            rep.top_line,
            rep.distinct,
        );
        if rivoli::telemetry::is_degenerate(&rep) {
            tracing::warn!(
                "STRUCTURALLY DEGENERATE: one line repeats {}x and the distinct-word ratio \
                 is {:.3} (healthy band 0.42-0.53). This is a near-miss loop — a varying \
                 slot in a repeated template — which the verbatim-cycle and \
                 longest-repeated-block checks CANNOT see. tok/s is not usable: the hit \
                 rate rises as the output collapses.",
                rep.top_line,
                rep.distinct,
            );
        }
        if let Some(d) = degenerate {
            tracing::warn!(
                "DEGENERATE OUTPUT: the last {} of {} generated tokens are {} verbatim \
                 repeats of a {}-token block (loop starts at token {}). This run's tok/s \
                 is NOT comparable — a looped decode reuses a few experts, so its hit \
                 rate and speed are both inflated. Investigate before ranking it.",
                d.period * d.repeats,
                ids.len(),
                d.repeats,
                d.period,
                d.start,
            );
        }
        // The run's identity for the root span.
        let run = rivoli::telemetry::RunInfo {
            model: a_model.clone(),
            mode: cfg.mode.to_string(),
            cache_policy: a_cache_policy.clone(),
            attn: a_attn.clone(),
            max_mem_gib: a_max_mem,
            bench_tokens: a_bench,
            prompt: cfg.prompt.clone(),
            moe_gain: a_moe_gain,
            two_q_kin: a_2q_kin,
            two_q_kout: a_2q_kout,
            sinks: a_sinks,
            window: a_window,
            misa_heads: a_misa_heads,
            degenerate,
        };
        rivoli::telemetry::export_decode(&summary, ids.len(), &run);
        if let Some(path) = &a_dump_ids {
            use std::io::Write;
            let mut w = std::io::BufWriter::new(
                std::fs::File::create(path).with_context(|| format!("create {path}"))?,
            );
            // Header names the arm, so two files cannot be silently compared across
            // different backends/modes — the same discipline `--ppl-out` uses.
            writeln!(
                w,
                "# rivoli-ids v1 backend={} mode={} policy={} attn={} tokens={}",
                if cfg!(feature = "vulkan") { "vulkan" } else { "rocm" },
                cfg.mode,
                a_cache_policy,
                a_attn,
                ids.len(),
            )?;
            for id in &ids {
                writeln!(w, "{id}")?;
            }
            w.flush().context("flush --dump-ids")?;
            info!("wrote {} token ids to {path}", ids.len());
        }
        info!("{bench_prompt}{}", tok.decode_all(&ids)?);
        Ok(())
    }
}
