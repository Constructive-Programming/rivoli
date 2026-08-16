//! `convert_v4` — the DeepSeek-V4-Flash-0731 checkpoint → a rivoli `.f4` artifact.
//! Ported from `old:src/bin/convert_v4.rs` (`wt/glimmer-s2` @ 6b7f496), comments travelling
//! with their code.
//!
//! The binary keeps its model name (2026-08-09 rename pass): a checkpoint converter is
//! about its source checkpoint — the tensor names, the fp4 copy path and the shared-expert
//! boundary below are all THAT model's — and the installed executable name is user-visible,
//! recorded in command lines in the old tree's `docs/investigations/v4-flash-port.md` and
//! `docs/reference/architecture.md` §12.
//!
//! Separate from `bin/convert` (GLM-5.2 → `.vq3`) because the two share almost nothing:
//! there is no codebook to learn, no VQ encode, and no fp8 dequant on the routed path.
//! **Every routed expert is copied, not quantized.** V4 ships them already at 4 bits —
//! e2m1 nibble pairs with e8m0 block scales — and re-quantizing a 4-bit source into
//! 3.25-bit int3-vq is the lossy-on-lossy chain the old tree's `int4-scales.md`
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
use rivoli_artifact::format::{
    ArtifactDirs, Dtype, F4_NAMING_V4, FormatMeta, RoutedRepack, SafeWriter, Safetensors,
    f4_source, finish_artifact,
};
use rivoli_artifact::quant::{FP8_BLOCK, V4_PROJ, v4_expert_base};
use rivoli_artifact::v4_config::V4Config;

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
fn write_layer_resident<'a>(
    w: &mut SafeWriter<'a>,
    src: &'a Safetensors,
    cfg: &V4Config,
    l: usize,
) -> Result<()> {
    let lb = format!("layers.{l}");
    confront_config_with_tensors(src, cfg, &lb)?;
    for t in ["attn_norm", "ffn_norm", "attn.q_norm", "attn.kv_norm"] {
        w.add_widened(src, &format!("{lb}.{t}.weight"))?;
    }
    // A per-head f32 logit added to the softmax DENOMINATOR only. Already f32.
    w.copy_verbatim(src, &format!("{lb}.attn.attn_sink"), Dtype::F32)?;
    // wo_a is fp8 in the checkpoint and the reference DEQUANTIZES it to bf16 before use
    // (`convert.py`, and `Attention.__init__` declares it `dtype=torch.bfloat16`), then
    // does a bf16 einsum. It is carried here as fp8 + f32 scale like its neighbours; the
    // engine must decide whether to match the reference's bf16 arithmetic or use an fp8 GEMV,
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
    write_router(w, src, cfg, &lb, l)?;
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

/// Confront the config with the tensors ONCE per layer, before copying any of them.
///
/// Everything `write_layer_resident` writes is copied verbatim, and the manifest carries the
/// config verbatim — so without this, `n_heads`, `head_dim`, `q_lora_rank`, `o_groups`,
/// `o_lora_rank`, `vocab` and `top_k` reach the engine's kernel launches having never been
/// compared to the weights they describe. A disagreement would size every attention launch
/// wrongly at decode time with no error at convert time, which is this port's whole hazard
/// class. (`hidden`, `moe_inter` and `n_experts` are already confronted by `F4Expert::spans`.)
/// The reference's own `convert.py --model-parallel` narrows exactly these tensors while
/// leaving `config.json` untouched, so a shard-narrowed checkpoint is a concrete way to reach
/// the mismatch rather than a hypothetical one.
///
/// Split out of `write_layer_resident` in the port: the reference has the table, the copy walk
/// and the router branch in one body, which is the shape the code-health gate refuses. The
/// split is also the argument made structural — this function is exactly "what the config
/// claims, checked against the file".
fn confront_config_with_tensors(src: &Safetensors, cfg: &V4Config, lb: &str) -> Result<()> {
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
            vec![
                cfg.o_groups * cfg.o_lora_rank,
                cfg.n_heads * hd / cfg.o_groups,
            ],
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
        ensure!(
            got == want,
            "{name}: shape {got:?}, config implies {want:?}"
        );
    }
    Ok(())
}

/// The one tensor that differs between a hash-routed layer and a scored one.
///
/// A hash layer selects experts from `tid2eid` and has NO bias; a scored layer has a
/// bias and no table. Driven off the config rather than off `src.has(…)`, so a layer
/// whose tensors disagree with `num_hash_layers` fails instead of silently taking
/// whichever branch the checkpoint happens to satisfy.
fn write_router<'a>(
    w: &mut SafeWriter<'a>,
    src: &'a Safetensors,
    cfg: &V4Config,
    lb: &str,
    l: usize,
) -> Result<()> {
    if cfg.layer_routes_by_hash(l) {
        let name = format!("{lb}.ffn.gate.tid2eid");
        let got = src.shape(&name)?;
        ensure!(
            got == [cfg.vocab, cfg.top_k],
            "{name}: shape {got:?}, config implies [{}, {}]",
            cfg.vocab,
            cfg.top_k
        );
        w.copy_verbatim(src, &name, Dtype::I64)
    } else {
        w.copy_verbatim(src, &format!("{lb}.ffn.gate.bias"), Dtype::F32)
    }
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
fn write_compressor<'a>(w: &mut SafeWriter<'a>, src: &'a Safetensors, base: &str) -> Result<()> {
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

/// The layer range this run covers, and the config that decided it — everything the guards
/// below establish before a single tensor is read.
///
/// Split out of `main` on the precedent `convert_glimmer::refuse_before_writing` set: the
/// reference's `main` runs the checks, the repack loop, the resident walk and the manifest in
/// one body, which is the shape the code-health gate refuses. Returning the config is what
/// keeps this to ONE parse — a second `load_config` later would be the changed-during-the-run
/// class M7's review closed.
fn refuse_before_writing(
    src_dir: &str,
    out_dir: &str,
    from: usize,
    to: Option<usize>,
) -> Result<(V4Config, std::ops::Range<usize>)> {
    // **Not in the reference, and added deliberately.** `convert_v4` writes
    // `resident.safetensors` AND every `L{ll}.f4` into `out_dir` while `Safetensors` holds the
    // source shards mapped — so an out_dir that is the src_dir is a SIGBUS, a fatal signal
    // rather than an error, with the output half-formed. The guard and its argument live with
    // `SafeWriter`, whose hazard it is; `convert_glimmer` calls the same function.
    SafeWriter::refuse_writing_into_source(&artifact_dirs(src_dir, out_dir))?;
    let cfg: V4Config = V4Config::load(src_dir)?;
    let to = to.unwrap_or(cfg.n_layers);
    // Refused, not clamped: `--to 999` on a 43-layer model silently converting 43 layers
    // would look like it did what was asked.
    ensure!(
        from < to && to <= cfg.n_layers,
        "layer range [{from}, {to}) is not inside [0, {})",
        cfg.n_layers
    );
    Ok((cfg, from..to))
}

/// Step 2: the resident set. ALWAYS rewritten, never reused — unlike a `.f4`, which is one
/// self-contained layer, `resident.safetensors` is a single file spanning every layer
/// the run covered. Reusing it after a run with a different `--from/--to` produced an
/// artifact whose `.f4` set and resident set covered different ranges, and the missing
/// layers surfaced later as "tensor layers.3.attn_norm.weight not found", which reads
/// like a corrupt checkpoint rather than a stale artifact. It is the cheap half of the
/// convert (16 s for the experts, ~2 s for this), so there is nothing to protect.
fn write_resident(
    src: &Safetensors,
    cfg: &V4Config,
    layers: &[usize],
    out_dir: &str,
) -> Result<()> {
    let rpath = format!("{out_dir}/resident.safetensors");
    let mut w = SafeWriter::new();
    // embed/head stay BF16. Whether to requantize them (GLM's converter int8s both) is
    // a quality question with a paired-dNLL measurement attached, and making it silently
    // here — in the one step that is otherwise lossless end to end — would put an
    // unmeasured approximation inside a "repack" and leave nothing to attribute a
    // regression to. 2.1 GB of a ~10 GB resident set.
    // ponytail: int8 embed/head once there is a decode to measure the cost against.
    for &(name, emit) in MODEL_LEVEL {
        match emit {
            Emit::Verbatim(d) => w.copy_verbatim(src, name, d)?,
            Emit::WidenToF32 => w.add_widened(src, name)?,
        }
    }
    for &l in layers {
        write_layer_resident(&mut w, src, cfg, l).with_context(|| format!("resident layer {l}"))?;
    }
    w.write(&rpath)?;
    eprintln!("convert_v4: wrote {rpath}");
    Ok(())
}

/// Auxiliary files copied beside the weights, so the artifact is self-contained: the engine
/// loads its tokenizer and its stop tokens from the model directory, not from the checkpoint.
///
/// **Three, not four** — unlike `convert_glimmer`'s. This checkpoint ships NO
/// `chat_template.jinja`, deliberately and by its README's own statement, which is the whole
/// reason `rivoli_artifact::v4_encoding` exists as a hand-port with no file to load.
const AUX: [&str; 3] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
];

/// Step 3: `manifest.json` = the source config + a `format` section + provenance.
///
/// Takes the two directories as `&str` rather than `&Args`, which is also what keeps the
/// `ArtifactDirs` literal below from being a jscpd clone of `convert_glimmer`'s — a 16-token
/// `ArtifactDirs { out: &args.out_dir, src: &args.src_dir }` is over the 15-token floor all by
/// itself, and two converters that both have an `args` cannot both spell it that way.
fn artifact_dirs<'a>(src_dir: &'a str, out_dir: &'a str) -> ArtifactDirs<'a> {
    ArtifactDirs {
        out: out_dir,
        src: src_dir,
    }
}

fn write_manifest(src_dir: &str, out_dir: &str, layers: std::ops::Range<usize>) -> Result<()> {
    let mut manifest = FormatMeta::manifest_from_config(src_dir, FP8_BLOCK)?;
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
    manifest["f4_source"] = f4_source("convert_v4", src_dir, layers);
    finish_artifact(
        "convert_v4",
        artifact_dirs(src_dir, out_dir),
        &manifest,
        &AUX,
    )
}

fn main() -> Result<()> {
    // DESTRUCTURED, where the reference re-binds `Args { verify: verify || verify_only, ..args }`
    // and threads `&args` onward. Two reasons, and the second is mechanical. The first: that
    // re-bind makes `args.verify` mean something different from what the user typed, three
    // screens from where it was decided, and `let verify = verify || verify_only;` says it once.
    // The second: `} <blank> fn main() -> Result<()> { let args = Args::parse();` is 20 tokens
    // over 5 lines, which is a jscpd clone of `convert_glimmer`'s opening — `add_indexer.rs` and
    // `fp8_to_i4.rs` in this directory already destructure, so this follows them rather than
    // inventing a shape.
    let Args {
        src_dir,
        out_dir,
        from,
        to,
        verify,
        verify_only,
    } = Args::parse();
    // `--verify-only` implies `--verify` — one intent, resolved once, exactly as `--help` says.
    let verify = verify || verify_only;
    let (cfg, range) = refuse_before_writing(&src_dir, &out_dir, from, to)?;
    let (hidden, moe_inter, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    std::fs::create_dir_all(&out_dir)?;

    let layers: Vec<usize> = range.clone().collect();
    // Only the shards holding what we are about to read: the tensors of these layers, plus
    // the model-level ones. A checkpoint still downloading has truncated shards, and
    // opening one would fail the whole run over a layer we are not converting.
    let wanted: Vec<String> = layers.iter().map(|l| format!("layers.{l}.")).collect();
    let src = Safetensors::open_indexed(&src_dir, |n| {
        MODEL_LEVEL.iter().any(|&(t, _)| t == n) || wanted.iter().any(|p| n.starts_with(p.as_str()))
    })?;
    eprintln!(
        "convert_v4: hidden={hidden} moe_inter={moe_inter} experts={ne} layers {}..{} \
         (of {})",
        range.start, range.end, cfg.n_layers
    );

    // 1. Routed experts → one `.f4` per layer. Pure repack. The loop itself lives in
    // `RoutedRepack`, shared with `convert_k3` — this converter's only contribution to it is
    // `v4_expert_base` and the fact that V4's experts are entered at `hidden`.
    let repack = RoutedRepack {
        tool: "convert_v4",
        out_dir: &out_dir,
        src: &src,
        naming: &F4_NAMING_V4,
        expert_in: hidden,
        moe_inter,
        n_experts: ne,
        verify,
        write: !verify_only,
    };
    for &l in &layers {
        repack.layer(l, |e| v4_expert_base(l, e, ne))?;
    }

    // A verification pass STOPS HERE. `--verify` on its own is a flag on a converter, so it
    // still runs steps 2 and 3 below — and both are unconditional writes scoped to
    // `--from/--to`. Pointing `--verify --from 0 --to 3` at a whole artifact therefore
    // truncates its resident set to three layers and rewrites the manifest to claim `[0, 3]`,
    // leaving 40 orphaned `.f4` files whose tensors are gone. Found by review 2026-08-05
    // before it ran: the old tree's `tests/refactor-gates/capture.sh` was about to do exactly
    // that to the 146 GB `/var/db/rivoli/v4-f4-full`, and then decode the artifact it had just
    // broken.
    //
    // With `--verify-only` the run is READ-ONLY, which is also what makes verifying all 43
    // layers affordable — the expensive half of a convert is the writing, not the reading.
    if verify_only {
        eprintln!(
            "convert_v4: verify-only — {} layer(s) checked, nothing written",
            layers.len()
        );
        return Ok(());
    }

    write_resident(&src, &cfg, &layers, &out_dir)?;
    write_manifest(&src_dir, &out_dir, range)?;
    eprintln!("convert_v4: done — {src_dir} → {out_dir}");
    Ok(())
}
