//! convert_v4 — the DeepSeek-V4-Flash-0731 checkpoint → a rivoli `.f4` artifact.
//!
//! Separate from `bin/convert` (GLM-5.2 → `.vq3`) because the two share almost nothing:
//! there is no codebook to learn, no VQ encode, and no fp8 dequant on the routed path.
//! **Every routed expert is copied, not quantized.** V4 ships them already at 4 bits —
//! e2m1 nibble pairs with e8m0 block scales — and re-quantizing a 4-bit source into
//! 3.25-bit int3-vq is the lossy-on-lossy chain `docs/investigations/int4-scales.md`
//! records at PPL 73.43. The attention half is likewise copied: it is already F8_E4M3 at
//! the 128×128 block size the resident path uses, and only its e8m0 scale byte is widened
//! to the f32 that path reads (exactly — see `SafeWriter::copy_fp8_e8m0`).
//!
//! ```text
//! manifest.json          # the V4 config + a `format` section
//! resident.safetensors   # attention fp8 (verbatim) + norms/gate (f32) + embed/head (bf16)
//!                        #   + the SHARED expert, which is fp8 e4m3 and NOT FP4
//! L{ll}.f4               # per layer: header + n_experts routed blocks, no shared block
//! ```
//!
//! `--verify` re-reads what it wrote and byte-compares it against the source. `--help` is
//! the flag reference.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use rivoli::artifact::format::{
    Dtype, EXPERT_HEADER_BYTES, ExpertHeader, F4_MAGIC, FormatMeta, SafeWriter, Safetensors,
    F4Expert, finish_artifact,
};
use rivoli::artifact::model::{V4Config, load_config};
use rivoli::artifact::quant::{
    FP8_BLOCK, V4_PROJ, VQ_ALIGN, f4_expert_bytes, f4_expert_stride, fill_expert_blocks,
    v4_expert_base,
};

// NOTE: doc comments on the FIELDS below are USER-FACING — clap renders them as `--help`.
#[derive(Parser)]
#[command(
    name = "convert_v4",
    about = "DeepSeek-V4-Flash checkpoint → the rivoli .f4 artifact (FP4 experts repacked, \
             fp8 attention copied)"
)]
struct Args {
    /// The V4-Flash checkpoint directory: config.json, model.safetensors.index.json, the
    /// `*.safetensors` shards, and the tokenizer files copied into the artifact.
    src_dir: String,

    /// Artifact directory to write. Created if absent; an existing `L{ll}.f4` is REUSED
    /// rather than rewritten, so a killed run resumes on the same command line. Layer files
    /// are written to `.part` and renamed, so a run killed mid-write leaves no short file
    /// for the next run to trust.
    out_dir: String,

    /// First layer to convert (default 0).
    #[arg(long, value_name = "L", default_value_t = 0)]
    from: usize,

    /// One past the last layer to convert (default: the whole model). The range is
    /// recorded in the manifest's `f4_source`, so a partial artifact can never claim to be
    /// whole — and only the shards holding these layers are opened, which is what lets a
    /// convert run against a checkpoint that is still downloading.
    #[arg(long, value_name = "L")]
    to: Option<usize>,

    /// After writing each layer, read the FILE back and byte-compare every expert against
    /// the source tensors. The repack is a copy, so the only correct answer is zero
    /// differing bytes; anything else is a defect in the writer, not a quantization error.
    #[arg(long)]
    verify: bool,

    /// Verify and write NOTHING. Implies `--verify`.
    ///
    /// `--verify` alone is a flag on a converter: it still rewrites `resident.safetensors`
    /// and `manifest.json` for `--from/--to`, so aiming a narrow `--verify` at a whole
    /// artifact SILENTLY TRUNCATES it. Use this whenever the target is an artifact you want
    /// to keep — which, for a verification pass, is always.
    #[arg(long)]
    verify_only: bool,
}

/// The resident (non-routed-expert) tensors of one layer, in the order they are written.
///
/// Names are V4's own — `layers.{l}.attn.wq_a`, not `model.layers.{l}.self_attn.q_a_proj`.
/// The artifact describes a V4 model; translating into GLM's scheme would be a second
/// naming convention to get wrong, and the decode path has to learn V4's either way.
fn write_layer_resident(
    w: &mut SafeWriter,
    src: &Safetensors,
    cfg: &V4Config,
    l: usize,
) -> Result<()> {
    let lb = format!("layers.{l}");
    // Confront the config with the tensors ONCE per layer, before copying any of them.
    //
    // Everything below is copied verbatim, and the manifest carries the config verbatim —
    // so without this, `n_heads`, `head_dim`, `q_lora_rank`, `o_groups`, `o_lora_rank`,
    // `vocab` and `top_k` reach S2's kernel launches having never been compared to the
    // weights they describe. A disagreement would size every attention launch wrongly at
    // decode time with no error at convert time, which is this port's whole hazard class.
    // (`hidden`, `moe_inter` and `n_experts` are already confronted by `F4Expert::spans`.)
    // The reference's own `convert.py --model-parallel` narrows exactly these tensors
    // while leaving `config.json` untouched, so a shard-narrowed checkpoint is a concrete
    // way to reach the mismatch rather than a hypothetical one.
    let (h, hd) = (cfg.hidden, cfg.head_dim);
    for (name, want) in [
        (format!("{lb}.attn.wq_a.weight"), vec![cfg.q_lora_rank, h]),
        (
            format!("{lb}.attn.wq_b.weight"),
            vec![cfg.n_heads * hd, cfg.q_lora_rank],
        ),
        // ONE kv entry, `head_dim` wide, serving as both K and V for all heads.
        (format!("{lb}.attn.wkv.weight"), vec![hd, h]),
        (
            format!("{lb}.attn.wo_a.weight"),
            vec![cfg.o_groups * cfg.o_lora_rank, cfg.n_heads * hd / cfg.o_groups],
        ),
        (
            format!("{lb}.attn.wo_b.weight"),
            vec![h, cfg.o_groups * cfg.o_lora_rank],
        ),
        (format!("{lb}.attn.attn_sink"), vec![cfg.n_heads]),
        (format!("{lb}.ffn.gate.weight"), vec![cfg.n_experts, h]),
        (format!("{lb}.attn_norm.weight"), vec![h]),
    ] {
        let got = src.shape(&name)?;
        ensure!(got == want, "{name}: shape {got:?}, config implies {want:?}");
    }
    for t in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
        w.add_widened(src, &format!("{lb}.{t}.weight"))?;
    }
    // A per-head f32 logit added to the softmax DENOMINATOR only. Already f32.
    w.copy_verbatim(src, &format!("{lb}.attn.attn_sink"), Dtype::F32)?;
    // wo_a is fp8 in the checkpoint and the reference DEQUANTIZES it to bf16 before use
    // (`convert.py`, and `Attention.__init__` declares it `dtype=torch.bfloat16`), then
    // does a bf16 einsum. It is carried here as fp8 + f32 scale like its neighbours; S2
    // must decide whether to match the reference's bf16 arithmetic or use an fp8 GEMV,
    // because the two do not agree bit-for-bit and the oracle is written against bf16.
    for p in ["wq_a", "wq_b", "wkv", "wo_a", "wo_b"] {
        w.copy_fp8_e8m0(src, &format!("{lb}.attn.{p}"))?;
    }
    // The shared expert is F8_E4M3 at 128×128 — NOT FP4. It is always-on and resident, so
    // it rides the fp8 path here rather than the `.f4` stream. Its base comes from
    // `v4_expert_base` (index == n_experts) so that the routed/shared name boundary has
    // one definition, not one in quant.rs and an inline copy here.
    let shared = v4_expert_base(l, cfg.n_experts, cfg.n_experts);
    for p in V4_PROJ {
        w.copy_fp8_e8m0(src, &format!("{shared}.{p}"))?;
    }
    w.add_widened(src, &format!("{lb}.ffn.gate.weight"))?;
    // A hash layer selects experts from `tid2eid` and has NO bias; a scored layer has a
    // bias and no table. Driven off the config rather than off `src.has(…)`, so a layer
    // whose tensors disagree with `num_hash_layers` fails instead of silently taking
    // whichever branch the checkpoint happens to satisfy.
    if cfg.layer_routes_by_hash(l) {
        let name = format!("{lb}.ffn.gate.tid2eid");
        let got = src.shape(&name)?;
        ensure!(
            got == [cfg.vocab, cfg.top_k],
            "{name}: shape {got:?}, config implies [{}, {}]",
            cfg.vocab,
            cfg.top_k
        );
        w.copy_verbatim(src, &name, Dtype::I64)?;
    } else {
        w.copy_verbatim(src, &format!("{lb}.ffn.gate.bias"), Dtype::F32)?;
    }
    for t in ["hc_attn", "hc_ffn"] {
        for s in ["base", "fn", "scale"] {
            w.copy_verbatim(src, &format!("{lb}.{t}_{s}"), Dtype::F32)?;
        }
    }
    if cfg.layer_has_compressor(l)? {
        write_compressor(w, src, &format!("{lb}.attn.compressor"))?;
    }
    if cfg.layer_has_indexer(l)? {
        // The lightning indexer is NOT shaped like GLM's. It has no `wk`/`k_norm` of its
        // own: `Indexer.__init__` gives it a second `Compressor` (with the Hadamard
        // rotation) to build the keys it scores against, plus `wq_b` (fp8) and
        // `weights_proj` (bf16). Read from the shipped checkpoint, 2026-08-05 — an earlier
        // guess at GLM's names failed on layer 2 during the first real convert.
        w.copy_fp8_e8m0(src, &format!("{lb}.attn.indexer.wq_b"))?;
        w.add_widened(src, &format!("{lb}.attn.indexer.weights_proj.weight"))?;
        write_compressor(w, src, &format!("{lb}.attn.indexer.compressor"))?;
    }
    Ok(())
}

/// One `Compressor`'s four tensors. Both the attention's and the indexer's own, which
/// differ only in width — the shapes are taken from the source, never assumed, because
/// they vary with `compress_ratio`: `ape` is `[ratio, coff·head_dim]` and `wkv`/`wgate`
/// are `[coff·head_dim, dim]`, where `coff = 1 + (ratio == 4)` (the overlapping-window
/// case stores two halves in one tensor). Layer 2 at ratio 4 has `ape[4, 1024]`; layer 3
/// at ratio 128 has `ape[128, 512]`.
///
/// `wkv`/`wgate` are bf16 on disk but `Compressor.__init__` declares them fp32 and its
/// forward runs `x.float()` — "compression need fp32", per the reference's own comment.
/// Widening them here is therefore the reference's arithmetic, not a choice.
fn write_compressor(w: &mut SafeWriter, src: &Safetensors, base: &str) -> Result<()> {
    w.copy_verbatim(src, &format!("{base}.ape"), Dtype::F32)?;
    for t in ["norm.weight", "wgate.weight", "wkv.weight"] {
        w.add_widened(src, &format!("{base}.{t}"))?;
    }
    Ok(())
}

/// What the resident writer does with a tensor. Spelled out rather than inferred from the
/// source dtype, because bf16 alone does not say which: `norm.weight` is widened (the
/// loader reads norms as f32) while `embed`/`head` are kept bf16 on purpose.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Emit {
    /// Same dtype, same bytes.
    Verbatim(Dtype),
    /// bf16 → f32.
    WidenToF32,
}

/// Tensors that belong to no layer. ONE list, because it drives two things that have to
/// agree: which shards `open_indexed` opens, and what the resident writer emits. As two
/// lists, adding a tensor to the writer alone meant its shard was never opened — failing as
/// "tensor X not found", which points at the checkpoint rather than at the filter.
const MODEL_LEVEL: &[(&str, Emit)] = &[
    ("embed.weight", Emit::Verbatim(Dtype::Bf16)),
    ("head.weight", Emit::Verbatim(Dtype::Bf16)),
    ("norm.weight", Emit::WidenToF32),
    ("hc_head_base", Emit::Verbatim(Dtype::F32)),
    ("hc_head_fn", Emit::Verbatim(Dtype::F32)),
    ("hc_head_scale", Emit::Verbatim(Dtype::F32)),
];

fn main() -> Result<()> {
    let args = Args::parse();
    let args = Args { verify: args.verify || args.verify_only, ..args };
    let cfg: V4Config = load_config(&args.src_dir)?;
    let to = args.to.unwrap_or(cfg.n_layers);
    // Refused, not clamped: `--to 999` on a 43-layer model silently converting 43 layers
    // would look like it did what was asked.
    ensure!(
        args.from < to && to <= cfg.n_layers,
        "layer range [{}, {to}) is not inside [0, {})",
        args.from,
        cfg.n_layers
    );
    let (hidden, moe_inter, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    std::fs::create_dir_all(&args.out_dir)?;

    let layers: Vec<usize> = (args.from..to).collect();
    // Only the shards holding what we are about to read: the tensors of these layers, plus
    // the model-level ones. A checkpoint still downloading has truncated shards, and
    // opening one would fail the whole run over a layer we are not converting.
    let wanted: Vec<String> = layers.iter().map(|l| format!("layers.{l}.")).collect();
    let src = Safetensors::open_indexed(&args.src_dir, |n| {
        MODEL_LEVEL.iter().any(|&(t, _)| t == n)
            || wanted.iter().any(|p| n.starts_with(p.as_str()))
    })?;
    eprintln!(
        "convert_v4: hidden={hidden} moe_inter={moe_inter} experts={ne} layers {}..{to} \
         (of {})",
        args.from, cfg.n_layers
    );

    // 1. Routed experts → one `.f4` per layer. Pure repack.
    let stride = f4_expert_stride(hidden, moe_inter);
    let ebytes = f4_expert_bytes(hidden, moe_inter);
    for &l in &layers {
        let path = format!("{}/L{l:02}.f4", args.out_dir);
        let reused = std::fs::metadata(&path).is_ok();
        if reused {
            eprintln!("convert_v4: {path} exists, reusing");
        }
        let expert = |e| F4Expert {
            src: &src,
            base: v4_expert_base(l, e, ne),
            hidden,
            moe_inter,
        };
        if !reused {
            // One aligned block for the header, then `ne` routed blocks — and NO shared block,
            // unlike `.vq3`/`.i4`. V4's shared expert is fp8 e4m3, a different format entirely;
            // a block written past `ne` would be the wrong ARITHMETIC, not just wrong weights.
            let mut buf = vec![0u8; VQ_ALIGN + ne * stride];
            buf[..EXPERT_HEADER_BYTES].copy_from_slice(
                &ExpertHeader::new(F4_MAGIC, l, ne, hidden, moe_inter, stride).to_bytes(),
            );
            fill_expert_blocks(&mut buf[VQ_ALIGN..], stride, ebytes, ne, |e, slot| {
                expert(e).pack(slot)
            })
            .with_context(|| format!("repack layer {l}"))?;
            // tmp→rename: `std::fs::write` is not atomic and a layer file is 3.4 GB, so a run
            // killed mid-write would otherwise leave a short `L{ll}.f4` that the next run
            // reuses and never re-reads. Rename within one directory is atomic.
            let part = format!("{path}.part");
            std::fs::write(&part, &buf).with_context(|| format!("write {part}"))?;
            std::fs::rename(&part, &path).with_context(|| format!("rename {part} -> {path}"))?;
            eprintln!("convert_v4: wrote {path} ({} bytes)", buf.len());
        }

        // Verification reads the FILE, and runs on a REUSED layer too. Two reasons it is
        // not `back == buf`: that comparison could only ever pass (the buffer came from
        // `pack`, and `diff` reads the same source spans, so `differing` could never be
        // non-zero behind it) — a guard unable to fire dressed as a verification; and a
        // reused layer has no `buf` at all, which is exactly the layer whose bytes nobody
        // has ever checked. Comparing the file against the mmap'd source tests the writer's
        // offsets, the block stride, the write, and whatever a previous run left behind.
        if args.verify {
            let back = std::fs::read(&path).with_context(|| format!("re-read {path}"))?;
            ensure!(
                back.len() == VQ_ALIGN + ne * stride,
                "{path}: {} bytes on disk, expected {}",
                back.len(),
                VQ_ALIGN + ne * stride
            );
            let mut differing = 0usize;
            for e in 0..ne {
                let off = VQ_ALIGN + e * stride;
                differing += expert(e).diff(&back[off..off + ebytes])?.len();
            }
            ensure!(
                differing == 0,
                "layer {l}: {differing} bytes differ from the source — the repack is \
                 supposed to be a COPY"
            );
            eprintln!("convert_v4: verified L{l:02}.f4 — {ne} experts, 0 bytes differ");
        }
    }

    // A verification pass STOPS HERE. `--verify` on its own is a flag on a converter, so it
    // still runs steps 2 and 3 below — and both are unconditional writes scoped to
    // `--from/--to`. Pointing `--verify --from 0 --to 3` at a whole artifact therefore
    // truncates its resident set to three layers and rewrites the manifest to claim `[0, 3]`,
    // leaving 40 orphaned `.f4` files whose tensors are gone. Found by review 2026-08-05
    // before it ran: `tests/refactor-gates/capture.sh` was about to do exactly that to the
    // 146 GB `/var/db/rivoli/v4-f4-full`, and then decode the artifact it had just broken.
    //
    // With `--verify-only` the run is READ-ONLY, which is also what makes verifying all 43
    // layers affordable — the expensive half of a convert is the writing, not the reading.
    if args.verify_only {
        eprintln!("convert_v4: verify-only — {} layer(s) checked, nothing written", layers.len());
        return Ok(());
    }

    // 2. Resident set. ALWAYS rewritten, never reused — unlike a `.f4`, which is one
    // self-contained layer, `resident.safetensors` is a single file spanning every layer
    // the run covered. Reusing it after a run with a different `--from/--to` produced an
    // artifact whose `.f4` set and resident set covered different ranges, and the missing
    // layers surfaced later as "tensor layers.3.attn_norm.weight not found", which reads
    // like a corrupt checkpoint rather than a stale artifact. It is the cheap half of the
    // convert (16 s for the experts, ~2 s for this), so there is nothing to protect.
    let rpath = format!("{}/resident.safetensors", args.out_dir);
    let mut w = SafeWriter::new();
    // embed/head stay BF16. Whether to requantize them (GLM's converter int8s both) is
    // a quality question with a paired-dNLL measurement attached, and making it silently
    // here — in the one step that is otherwise lossless end to end — would put an
    // unmeasured approximation inside a "repack" and leave nothing to attribute a
    // regression to. 2.1 GB of a ~10 GB resident set.
    // ponytail: int8 embed/head once there is a decode to measure the cost against.
    for &(name, emit) in MODEL_LEVEL {
        match emit {
            Emit::Verbatim(d) => w.copy_verbatim(&src, name, d)?,
            Emit::WidenToF32 => w.add_widened(&src, name)?,
        }
    }
    for &l in &layers {
        write_layer_resident(&mut w, &src, &cfg, l)
            .with_context(|| format!("resident layer {l}"))?;
    }
    w.write(&rpath)?;
    eprintln!("convert_v4: wrote {rpath}");

    // 3. manifest.json = the source config + a `format` section + provenance.
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{}/config.json", args.src_dir))?)?;
    manifest["format"] = serde_json::to_value(FormatMeta::current(FP8_BLOCK))?;
    // Provenance, shaped like `I4Source` and for the reason its doc gives: two `.f4` sets
    // built from different checkpoints are byte-indistinguishable on disk, so an artifact
    // that records neither its source nor its range is one whose contents cannot be
    // attributed. `layers` is the range this artifact HOLDS — deliberately not
    // `num_hidden_layers`, because rewriting that would make a 3-layer artifact claim to
    // be a 3-layer MODEL, and every per-layer table in the config (`compress_ratios`,
    // `num_hash_layers`) is indexed by the REAL layer id. A partial artifact of layers
    // 20..23 has no layer 0 at all.
    //
    // Overwritten, not merged: the resident set above is rewritten for exactly this range
    // on every run, so the artifact really does cover only [from, to) afterwards.
    manifest["f4_source"] = serde_json::json!({
        "tool": "convert_v4",
        "chain": "fp4->fp4 (repack)",
        "src": args.src_dir,
        "layers": [args.from, to],
    });
    finish_artifact(
        "convert_v4",
        &args.out_dir,
        &args.src_dir,
        &manifest,
        &[
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
        ],
    )?;
    eprintln!("convert_v4: done — {} → {}", args.src_dir, args.out_dir);
    Ok(())
}
