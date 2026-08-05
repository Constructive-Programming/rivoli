//! Emit per-layer golden activations for DeepSeek-V4-Flash from the real checkpoint, and
//! re-run the defect matrix against real weights.
//!
//! S1b of `docs/investigations/v4-flash-port.md`. Needs no feature and touches no engine
//! code — like `bin/ppl`, it is host arithmetic that never sees a GPU, so there is no decode
//! path for it to cost. It reads safetensors directly rather than waiting on S1a's `.f4`.
//!
//! ```text
//! v4-oracle emit    --model /var/db/rivoli/deepseek-v4-flash-0731 --out goldens.bin
//! v4-oracle defects --model /var/db/rivoli/deepseek-v4-flash-0731 --layer 2
//! ```
//!
//! **Read `tests/v4_oracle.rs` before trusting a golden from this.** That file is where the
//! gate is *proved* — ~40 deliberate breakages, each asserted both to perturb the goldens it
//! claims to and to leave the rest bit-identical. This binary only produces numbers; it is
//! the test that establishes those numbers can reject a wrong implementation, and names the
//! two defects they measurably cannot.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use anyhow::{Context, Result, bail};
use rivoli::v4oracle::forward::{
    Capture, CompressorW, Defect, ExpertW, HeadTailW, IndexerW, LayerW, Oracle, Step,
};
use rivoli::v4oracle::golden::{GoldenSet, diff};
use rivoli::v4oracle::numerics::{bf16_decode, bf16_encode};
use rivoli::v4oracle::weights::{Checkpoint, V4Config, WMat};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// The short fixed prompt every golden is taken at.
///
/// Real tokenizer output, not hand-picked ids: the hash-routed layers index `tid2eid` by
/// token id, so a synthetic id sequence would exercise a routing pattern the model never
/// sees. Tokenized with the `tokenizers` crate against the checkpoint's own
/// `tokenizer.json` — deliberately not through `src/artifact/`, so the oracle shares no code
/// with the engine it judges.
const PROMPT: &str = "The capital of France is Paris, and the capital of Japan is";

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_default();
    let mut model = PathBuf::from("/var/db/rivoli/deepseek-v4-flash-0731");
    let mut out = PathBuf::from("v4-goldens.bin");
    let mut layers = 4usize;
    let mut decode_steps = 2usize;
    let mut layer = 2usize;
    let mut rest = args;
    let mut seen: Vec<String> = Vec::new();
    while let Some(flag) = rest.next() {
        seen.push(flag.clone());
        match flag.as_str() {
            "--model" => model = PathBuf::from(next(&mut rest, &flag)?),
            "--out" => out = PathBuf::from(next(&mut rest, &flag)?),
            "--layers" => layers = next(&mut rest, &flag)?.parse()?,
            "--decode-steps" => decode_steps = next(&mut rest, &flag)?.parse()?,
            "--layer" => layer = next(&mut rest, &flag)?.parse()?,
            other => bail!("unknown argument {other}"),
        }
    }
    // Reject flags the subcommand ignores rather than running against a default the caller
    // did not intend and saying nothing about it.
    match cmd.as_str() {
        "emit" if seen.iter().any(|f| f == "--layer") => bail!("emit takes --layers, not --layer"),
        "defects" if seen.iter().any(|f| f == "--out" || f == "--layers") => {
            bail!("defects takes --layer; it writes no file and drives one layer")
        }
        "emit" => emit(&model, &out, layers, decode_steps),
        "defects" => defects(&model, layer, decode_steps),
        _ => bail!(
            "usage:\n  v4-oracle emit    [--model DIR] [--out FILE] [--layers N] [--decode-steps K]\
             \n  v4-oracle defects [--model DIR] [--layer L] [--decode-steps K]"
        ),
    }
}

fn next(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("{flag} needs a value"))
}
/// The configuration every run uses: the shipped one, with `max_seq_len` cut to what the
/// oracle actually reaches.
///
/// `max_seq_len` sizes only the RoPE tables and the caches — it is not a trained constant
/// and does not enter any weight — so shrinking it changes nothing about the arithmetic
/// while keeping the ring and the compressed region small enough to compare by eye.
fn config(model: &Path) -> Result<V4Config> {
    let mut cfg = V4Config::v4_flash();
    // Before a single weight is read: if the hard-coded constants have drifted from the
    // checkpoint's own config, every golden below would be quietly wrong.
    cfg.assert_matches_reference_json(&model.join("inference/config.json"))?;
    cfg.max_seq_len = 512;
    Ok(cfg)
}

fn tokenize(model: &Path) -> Result<Vec<u32>> {
    let tk = tokenizers::Tokenizer::from_file(model.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer.json: {e}"))?;
    let enc = tk.encode(PROMPT, false).map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

// ---------------------------------------------------------------------------------------
// weight loading
// ---------------------------------------------------------------------------------------

fn dense_f32(ck: &Checkpoint, name: &str) -> Result<Vec<f32>> {
    ck.get(name)?.to_f32()
}

/// `wo_a` is `F8_E4M3` + a 128x128 `.scale` on disk but a plain bf16 parameter in the
/// reference, because `Attention.forward` consumes it raw in an einsum. `convert.py`'s
/// `wo_a` branch is exactly this: multiply by the block scale, then cast to bf16. There is
/// NO activation quantization on this projection.
fn dequantized_to_bf16(ck: &Checkpoint, name: &str) -> Result<WMat> {
    let q = ck.fp8(name)?;
    let (rows, cols) = (q.rows(), q.cols());
    let mut v = Vec::with_capacity(rows * cols);
    let mut row = Vec::with_capacity(cols);
    for r in 0..rows {
        q.row(r, &mut row);
        v.extend(row.iter().map(|x| bf16_decode(bf16_encode(*x))));
    }
    Ok(WMat::Dense { rows, cols, v })
}

fn load_compressor(ck: &Checkpoint, p: &str, ratio: usize, d: usize, rotate: bool) -> Result<CompressorW> {
    Ok(CompressorW {
        ratio,
        overlap: ratio == 4,
        d,
        rotate,
        ape: dense_f32(ck, &format!("{p}.ape"))?,
        wkv: ck.dense(&format!("{p}.wkv.weight"))?,
        wgate: ck.dense(&format!("{p}.wgate.weight"))?,
        norm: dense_f32(ck, &format!("{p}.norm.weight"))?,
    })
}

/// Load one layer. `experts` names exactly the routed experts to materialise — one is
/// 13.37 MB and the machine this runs on shares its memory with a live decode, so the whole
/// set is never loaded speculatively.
fn load_layer(ck: &Checkpoint, cfg: &V4Config, l: usize, experts: &[usize]) -> Result<LayerW> {
    let p = format!("layers.{l}");
    let a = format!("{p}.attn");
    let ratio = cfg.compress_ratio(l);
    let hash = l < cfg.n_hash_layers;
    let expert = |q: &str| -> Result<ExpertW> {
        Ok(ExpertW {
            w1: ck.fp4(&format!("{q}.w1.weight"))?,
            w2: ck.fp4(&format!("{q}.w2.weight"))?,
            w3: ck.fp4(&format!("{q}.w3.weight"))?,
        })
    };
    Ok(LayerW {
        attn_sink: dense_f32(ck, &format!("{a}.attn_sink"))?,
        wq_a: ck.fp8(&format!("{a}.wq_a.weight"))?,
        q_norm: dense_f32(ck, &format!("{a}.q_norm.weight"))?,
        wq_b: ck.fp8(&format!("{a}.wq_b.weight"))?,
        wkv: ck.fp8(&format!("{a}.wkv.weight"))?,
        kv_norm: dense_f32(ck, &format!("{a}.kv_norm.weight"))?,
        wo_a: dequantized_to_bf16(ck, &format!("{a}.wo_a.weight"))?,
        wo_b: ck.fp8(&format!("{a}.wo_b.weight"))?,
        attn_norm: dense_f32(ck, &format!("{p}.attn_norm.weight"))?,
        ffn_norm: dense_f32(ck, &format!("{p}.ffn_norm.weight"))?,
        hc_attn_fn: dense_f32(ck, &format!("{p}.hc_attn_fn"))?,
        hc_attn_base: dense_f32(ck, &format!("{p}.hc_attn_base"))?,
        hc_attn_scale: dense_f32(ck, &format!("{p}.hc_attn_scale"))?,
        hc_ffn_fn: dense_f32(ck, &format!("{p}.hc_ffn_fn"))?,
        hc_ffn_base: dense_f32(ck, &format!("{p}.hc_ffn_base"))?,
        hc_ffn_scale: dense_f32(ck, &format!("{p}.hc_ffn_scale"))?,
        gate_w: ck.dense(&format!("{p}.ffn.gate.weight"))?,
        gate_bias: (!hash).then(|| dense_f32(ck, &format!("{p}.ffn.gate.bias"))).transpose()?,
        tid2eid: hash.then(|| ck.get(&format!("{p}.ffn.gate.tid2eid"))?.to_i64()).transpose()?,
        compressor: (ratio != 0)
            .then(|| load_compressor(ck, &format!("{a}.compressor"), ratio, cfg.head_dim, false))
            .transpose()?,
        // `Indexer` exists ONLY where `compress_ratio == 4` -- 21 of the 43 layers. Checked
        // against the index rather than assumed, so a checkpoint that disagreed with
        // `model.py` would stop the run instead of silently losing the indexer.
        indexer: match (ratio == 4, ck.has_prefix(&format!("{a}.indexer."))) {
            (true, true) => Some(IndexerW {
                wq_b: ck.fp8(&format!("{a}.indexer.wq_b.weight"))?,
                weights_proj: ck.dense(&format!("{a}.indexer.weights_proj.weight"))?,
                compressor: load_compressor(
                    ck,
                    &format!("{a}.indexer.compressor"),
                    ratio,
                    cfg.index_head_dim,
                    true,
                )?,
            }),
            (false, false) => None,
            (r4, present) => bail!(
                "layer {l}: compress_ratio == 4 is {r4} but indexer tensors present is {present}"
            ),
        },
        experts: experts
            .iter()
            .map(|&e| Ok((e, expert(&format!("{p}.ffn.experts.{e}"))?)))
            .collect::<Result<HashMap<_, _>>>()?,
        shared: {
            // The SHARED expert is fp8 at 128x128, not fp4 at group 32: `MoE.__init__`
            // passes `expert_dtype` to the routed experts and nothing to the shared one, so
            // it inherits the default. The scale shapes on disk confirm it
            // (`[16, 32]` for a [2048, 4096] weight, not `[2048, 128]`).
            let q = format!("{p}.ffn.shared_experts");
            ExpertW {
                w1: ck.fp8(&format!("{q}.w1.weight"))?,
                w2: ck.fp8(&format!("{q}.w2.weight"))?,
                w3: ck.fp8(&format!("{q}.w3.weight"))?,
            }
        },
    })
}

/// The head tail's weights. Names verified against the checkpoint index, not guessed:
/// `hc_head_{fn,base,scale}` are F32 `[4, 16384]`/`[4]`/`[1]`, `norm.weight` is BF16 `[4096]`
/// and `head.weight` BF16 `[129280, 4096]` — the same six `bin/convert_v4`'s `MODEL_LEVEL`
/// emits.
///
/// `norm.weight` and `head.weight` go through the bf16 → f32 decode `RawTensor::to_f32`
/// does, which is what `RMSNorm`/`ParallelHead` mean by "stored bf16, held f32".
fn load_head_tail(ck: &Checkpoint) -> Result<HeadTailW> {
    Ok(HeadTailW {
        hc_head_fn: dense_f32(ck, "hc_head_fn")?,
        hc_head_base: dense_f32(ck, "hc_head_base")?,
        hc_head_scale: dense_f32(ck, "hc_head_scale")?,
        norm: dense_f32(ck, "norm.weight")?,
        lm_head: ck.dense("head.weight")?,
    })
}

/// A fixed, bf16-representable `[s, hc_mult, dim]` probe.
///
/// Used for TWO things in `defects`: the layer's residual input and the head tail's input.
/// They are the same buffer because both only need a fixed, reproducible stimulus, and one
/// generator is one place for the seed to live.
///
/// **Synthetic on purpose, and this is the whole reason the head tail can be emitted at all.**
/// Composing it with the layer chain at `--layers 4` of 43 would produce a logits vector that
/// is not any quantity the model computes, and a tensor named `logits` sitting next to real
/// per-layer goldens is the most misusable thing this file could write. A declared probe
/// cannot be read as a residual stream; the head tail is a pure function of its input, so the
/// golden gates exactly the same arithmetic either way. See `HeadTailW`'s doc.
fn fixed_probe(cfg: &V4Config, s: usize) -> Vec<f32> {
    let mut r = rivoli::v4oracle::weights::NamedRng::new("v4-head-probe");
    (0..s * cfg.hc_mult * cfg.dim).map(|_| bf16_decode(bf16_encode(r.unit()))).collect()
}

/// Run the head tail on `probe` and record it, input included.
///
/// The input is pushed HERE rather than inside `Oracle::head_tail`, so that the golden file
/// is self-contained: a device-side comparison needs the `h` that produced these logits, and
/// a golden whose input lives only in a function it cannot call is not a gate.
///
/// `s > 1` matters -- at one row `x[:, -1]` and `x[:, 0]` coincide and the final norm's
/// per-token statistic is its joint one, so two of the head defects become inert. Both
/// callers pass the prompt length, which `open()` has already refused below 4.
fn head_goldens(o: &Oracle, tw: &HeadTailW, cfg: &V4Config, probe: &[f32], cap: &mut Capture) {
    let row = cfg.hc_mult * cfg.dim;
    assert_eq!(probe.len() % row, 0, "the probe is not a whole number of [hc_mult, dim] rows");
    let s = probe.len() / row;
    cap.push("head.probe.in", &[s, cfg.hc_mult, cfg.dim], probe.to_vec());
    o.head_tail(tw, probe, s, "probe", cap);
}

/// Which routed experts a layer will actually use.
///
/// Hash layers are free: `tid2eid[input_id]` needs no activations at all. Score-routed
/// layers are not, so they get the whole set — 3.4 GB, loaded for one layer at a time and
/// dropped before the next.
fn experts_for(ck: &Checkpoint, cfg: &V4Config, l: usize, ids: &[u32]) -> Result<Vec<usize>> {
    if l >= cfg.n_hash_layers {
        return Ok((0..cfg.n_routed_experts).collect());
    }
    let map = ck.get(&format!("layers.{l}.ffn.gate.tid2eid"))?.to_i64()?;
    let k = cfg.n_activated_experts;
    let mut set: Vec<usize> =
        ids.iter().flat_map(|&t| map[t as usize * k..(t as usize + 1) * k].to_vec()).map(|e| e as usize).collect();
    set.sort_unstable();
    set.dedup();
    Ok(set)
}

// ---------------------------------------------------------------------------------------
// the shared driver
// ---------------------------------------------------------------------------------------

/// Drive `lws` through a prefill and `decode_steps` decode steps on one set of layer states.
///
/// `h_for(phase, ids)` supplies the residual stream each phase starts from. `emit` embeds
/// the fed token; `defects` hands back a fixed probe so a defect's effect is isolated to the
/// layer rather than inherited from the layers before it. One driver, because two copies of
/// a prefill-then-decode loop is two chances to get `start_pos` wrong -- and a wrong
/// `start_pos` is a silent-wrong of exactly the kind this oracle exists to catch.
fn drive(
    o: &Oracle,
    lws: &[LayerW],
    ids: &[u32],
    decode_steps: usize,
    cap: &mut Capture,
    mut h_for: impl FnMut(usize, &[u32]) -> Vec<f32>,
) {
    let s = ids.len();
    let mut states: Vec<_> = (0..lws.len()).map(|l| o.fresh_state(l)).collect();
    for phase in 0..=decode_steps {
        // Each decode step re-feeds the prompt's LAST token. Without an LM head there is no
        // sampled continuation, and a golden that depended on a sampler would not be a gate.
        // This is NOT a claim about what the model would generate; it makes the decode
        // captures a well-defined function of the prompt and of the cached state, which is
        // all S2 needs to compare against.
        let (n, start_pos, here, tag) = if phase == 0 {
            (s, 0usize, ids.to_vec(), "pre".to_string())
        } else {
            (1usize, s + phase - 1, vec![ids[s - 1]], format!("dec{}", phase - 1))
        };
        let mut h = h_for(phase, &here);
        for (l, lw) in lws.iter().enumerate() {
            let step = Step { lw, layer: l, s: n, start_pos, input_ids: &here, phase: &tag };
            o.run_layer(&step, &mut states[l], &mut h, cap);
        }
        eprintln!("{tag}: start_pos {start_pos}, {n} row(s), {} layers", lws.len());
    }
}

/// `(config, checkpoint, prompt ids)` -- the setup both commands need.
fn open(model: &Path) -> Result<(V4Config, Checkpoint, Vec<u32>)> {
    let cfg = config(model)?;
    let ck = Checkpoint::open(model)?;
    let ids = tokenize(model)?;
    eprintln!("prompt: {} tokens {ids:?}", ids.len());
    if ids.len() < 4 {
        bail!("the prompt must be at least `compress_ratio` (4) tokens to reach the compressor");
    }
    Ok((cfg, ck, ids))
}

// ---------------------------------------------------------------------------------------
// emit
// ---------------------------------------------------------------------------------------

fn emit(model: &Path, out: &Path, layers: usize, decode_steps: usize) -> Result<()> {
    let (cfg, ck, ids) = open(model)?;
    let s = ids.len();
    let o = Oracle::new(cfg.clone(), Defect::None);
    let hw = ck.dense("embed.weight")?;
    // The head tail runs on its OWN probe, never on the layer chain's residual -- see
    // `fixed_probe`. That is why it is loaded here and not threaded through `drive`.
    let tw = load_head_tail(&ck)?;

    // Loaded ONCE and held across every phase: reloading layer 3's 256 experts per decode
    // step is 3.4 GB of reads each time, and this machine's memory is shared with a live
    // decode.
    let mut lws = Vec::with_capacity(layers);
    for l in 0..layers {
        let experts = experts_for(&ck, &cfg, l, &ids)?;
        eprintln!("layer {l} (ratio {}): {} routed experts", cfg.compress_ratio(l), experts.len());
        lws.push(load_layer(&ck, &cfg, l, &experts)?);
    }

    let mut cap = Capture::default();
    cap.push("embed", &[s, cfg.hc_mult, cfg.dim], o.embed(&hw, &ids));
    drive(&o, &lws, &ids, decode_steps, &mut cap, |_, here| o.embed(&hw, here));
    head_goldens(&o, &tw, &cfg, &fixed_probe(&cfg, s), &mut cap);

    let meta = vec![
        ("model".into(), model.display().to_string()),
        ("prompt".into(), PROMPT.to_string()),
        ("prompt_ids".into(), format!("{ids:?}")),
        ("layers".into(), layers.to_string()),
        ("decode_steps".into(), decode_steps.to_string()),
        (
            "layer_classes".into(),
            (0..layers)
                .map(|l| {
                    let r = cfg.compress_ratio(l);
                    format!("L{l}=ratio{r}{}", if r == 4 { "+indexer" } else { "" })
                })
                .collect::<Vec<_>>()
                .join(" "),
        ),
        (
            "caveats".into(),
            "bsz=1; the head.probe.* tensors are the head tail run on a SYNTHETIC probe \
             (head.probe.in), NOT on the layer chain -- they are not the model's logits and \
             say nothing about what it would generate; ratio-128 compression is NOT exercised \
             at this prompt length -- it needs >=128 tokens; summation order is not pinned \
             (see v4oracle/forward.rs)"
                .into(),
        ),
    ];
    let g = GoldenSet::from_capture(meta, cap);
    let mut f = std::io::BufWriter::new(std::fs::File::create(out)?);
    g.write(&mut f)?;
    eprintln!("wrote {} float and {} int goldens to {}", g.floats.len(), g.ints.len(), out.display());
    for (n, shape, v) in &g.floats {
        let (lo, hi) =
            v.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
        eprintln!("  {n:<30} {shape:?} range [{lo:.4e}, {hi:.4e}]");
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------
// defects, on real weights
// ---------------------------------------------------------------------------------------

/// Re-run the defect set against ONE real layer and print what each breakage moved.
///
/// **This CHECKS NOTHING and always exits 0.** It is a print-out for a human to read beside
/// the expectation table in `tests/v4_oracle.rs`, not a gate: the table and the reachability
/// predicates live in the test crate and are not importable here. A `-- NOTHING --` row is
/// therefore a finding you have to notice, not a failure. The verdicts are established on
/// the toy; this is how you spot the toy having stopped standing in for the model.
fn defects(model: &Path, layer: usize, decode_steps: usize) -> Result<()> {
    let (cfg, ck, ids) = open(model)?;
    // ALL of them, even on a hash layer where `experts_for` would name only ~50: the point
    // of this command is to run `HashRoutingIgnored`, which routes by top-k score and
    // therefore reaches experts `tid2eid` never names. Loading the hash set would panic in
    // `moe` on the first such expert -- a crash, not a silent wrong, but still an
    // instrument that cannot measure the thing it was built to measure.
    let experts: Vec<usize> = (0..cfg.n_routed_experts).collect();
    eprintln!("layer {layer}, ratio {}, all {} experts", cfg.compress_ratio(layer), experts.len());
    // Only the target layer is loaded, and it is driven from a FIXED probe rather than from
    // the real chain: running layers 0..L first would make every defect's effect depend on
    // the layers before it, which is the opposite of an isolation test.
    let lws = vec![load_layer(&ck, &cfg, layer, &experts)?];
    // The head tail rides along on the same probe. Without it the six `Head*` defects would
    // print `-- NOTHING --` here for want of a golden to move, and this table has no
    // expectations of its own -- a reader would have to know, unaided, that the row was a
    // fixture gap and not a finding. Costs `head.weight`, ~2.1 GB widened; `embed.weight` is
    // NOT loaded here at all, which is why the head tail's weights are their own struct.
    let tw = load_head_tail(&ck)?;
    // NOTE: the seed moved from "real-defect-probe" to "v4-head-probe" on 2026-08-05 when the
    // layer probe and the head probe became one generator. Every number this command prints
    // therefore moved with it, for that reason and not because any arithmetic changed. An
    // older printout is not comparable against a newer one.
    let probe = fixed_probe(&cfg, ids.len());
    let row = cfg.hc_mult * cfg.dim;

    let capture = |d: Defect| {
        let o = Oracle::new(cfg.clone(), d);
        let mut cap = Capture::default();
        drive(&o, &lws, &ids, decode_steps, &mut cap, |phase, _| {
            if phase == 0 { probe.clone() } else { probe[..row].to_vec() }
        });
        head_goldens(&o, &tw, &cfg, &probe, &mut cap);
        cap
    };
    let base = capture(Defect::None);
    println!("{:<32} goldens moved (first 4 of each)", "defect");
    for d in Defect::breakages() {
        let moved: Vec<String> = diff(&base, &capture(d))
            .iter()
            .filter(|x| x.changed > 0)
            .map(|x| format!("{}({}/{})", x.name, x.changed, x.total))
            .collect();
        println!(
            "{:<32} {}",
            format!("{d:?}"),
            if moved.is_empty() {
                "-- NOTHING --".to_string()
            } else {
                moved.iter().take(4).cloned().collect::<Vec<_>>().join(" ")
            }
        );
    }
    Ok(())
}
