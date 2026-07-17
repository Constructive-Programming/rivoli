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

    // Expert usage ranking (drives the pin). Missing file = cold start.
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

    // M1 smoke: run the real int4 weights through the new MLP/MoE path. Embed
    // the last prompt token and feed it (raw — attention isn't wired yet) into
    // the dense MLP (layer 0) and the first MoE block. This exercises embedding
    // + SwiGLU + sigmoid routing on real weights; magnitudes are indicative,
    // not the final forward (that needs attention + the per-layer norms).
    let last = prompt_ids.last().copied().context("empty prompt")?;
    // embed_tokens is int8 (one byte/weight), not int4 — a small distinct class.
    let embed_bytes = snap.require("model.embed_tokens.weight")?;
    let embed_scale = snap.require("model.embed_tokens.weight.qs")?;
    let mut x = vec![0.0f32; mc.hidden];
    rivoli::quant::dequant_int8_row(embed_bytes, embed_scale, last as usize, &mut x);
    let l2 = |v: &[f32]| v.iter().map(|&a| a * a).sum::<f32>().sqrt();
    info!("embed[{last}]: l2={:.3} x[0..3]={:?}", l2(&x), &x[..3]);

    let max_inter = mc.dense_inter.max(mc.moe_inter * mc.n_shared);
    let mut scratch = rivoli::moe::MlpScratch::new(mc.hidden, max_inter);
    let mut out = vec![0.0f32; mc.hidden];
    rivoli::moe::dense_mlp(&snap, &mc, 0, &x, &mut scratch, &mut out)?;
    info!("dense MLP L0: l2={:.3} out[0..3]={:?}", l2(&out), &out[..3]);
    rivoli::moe::moe_block(&snap, &mc, mc.dense_layers, &x, &mut scratch, &mut out)?;
    info!(
        "MoE L{}: l2={:.3} out[0..3]={:?}",
        mc.dense_layers,
        l2(&out),
        &out[..3]
    );

    // MLA attention smoke: input_layernorm → attention on layer 0, appending to
    // the KV cache. Exercises the full absorb path (q_a/q_b/kv_a/kv_b/o + RoPE +
    // bf16 latents) on real weights for the first two positions.
    let mut kv = rivoli::attn::KvCache::new(&mc);
    let mut ascr = rivoli::attn::AttnScratch::new(&mc);
    let in_ln = rivoli::quant::read_f32(snap.require("model.layers.0.input_layernorm.weight")?);
    let mut xn = vec![0.0f32; mc.hidden];
    let mut aout = vec![0.0f32; mc.hidden];
    for pos in 0..2 {
        xn.copy_from_slice(&x);
        rivoli::math::rmsnorm(&mut xn, &in_ln, mc.rms_norm_eps as f32);
        rivoli::attn::attention(&snap, &mc, 0, &xn, pos, &mut kv, &mut ascr, &mut aout)?;
        info!(
            "attn L0 pos{pos}: l2={:.3} out[0..3]={:?}",
            l2(&aout),
            &aout[..3]
        );
    }

    match cfg.bench {
        Some(n) => info!("bench mode ({n} tokens) — full decode loop wiring is next"),
        None => bail!("server mode not yet implemented; use -bench <tokens>"),
    }
    Ok(())
}
