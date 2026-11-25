//! The synthetic Inkling-2-Small artifact `inkling_small_mxfp8.rs` scores: a TINY, VALID Small
//! wrapper whose dense tensors are deterministic LCG draws and whose routed experts are real
//! `.mxfp8` files produced by [`Mxfp8Layer`] from a source in the checkpoint's own interleaved
//! `w13` spelling.
//!
//! **The dimensions are Small's SHAPE, not its sizes** — 6 layers (5 sliding + 1 block end),
//! hidden 64, inter 64, 8 experts top-2 — so every code path the 48-layer model takes (static
//! and dynamic short convolutions, block residual, host router, O_DIRECT `.mxfp8` fill, the fused
//! SiTU-GLU MoE, `logit_tail`) runs at a size a gate can afford on every invocation. The fixture
//! is ~2 MB and builds in milliseconds, which is what lets this gate exist on a box that has
//! never seen the 619 GB checkpoint.
//!
//! **Everything here is `pub` and used once**, from the gate file beside it: this module exists
//! so the assertions fit under the 450-line cap, not so a second consumer can share it. It is
//! compiled into that ONE test binary (cargo does not treat a `tests/` subdirectory as a target).

// Tests: panic-on-failure is the idiom, and every fixture value is generated rather than threaded
// out as a `Result`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_artifact::format::{
    Dtype, FormatMeta, Mxfp8Layer, SafeWriter, Safetensors, TensorData, mxfp8_expert_prefix,
    mxfp8_source,
};
use rivoli_artifact::inkling_config::{Inkling2Config, InklingTextConfig};
use rivoli_artifact::quant::{mxfp8_expert_bytes, mxfp8_groups, mxfp8_slot_offsets};
use rivoli_core::hash::splitmix_finalize;
use rivoli_core::num::f32_to_bf16;
use rivoli_engine::inkling::geometry::is_block_end;
use rivoli_engine::inkling::names;
use std::path::{Path, PathBuf};

/// House scratch, never `/tmp` (63 GB tmpfs in RAM; O_DIRECT does not work there either).
const SCRATCH_ROOT: &str = "/var/cache/rivoli/scratch";
const HIDDEN: usize = 64;
const INTER: usize = 64;
const LAYERS: usize = 6;
const BLOCK: usize = 6;
const N_EXPERTS: usize = 8;

/// Every dimension the fixture's tensor shapes need, read ONCE from the config so the writer and
/// the byte layout cannot disagree.
#[derive(Clone, Copy)]
struct Dims {
    h: usize,
    nh: usize,
    kv: usize,
    hd: usize,
    dr: usize,
    ext: usize,
    k: usize,
    e: usize,
    inter: usize,
    layers: usize,
    block: usize,
    vocab: usize,
    unpadded: usize,
}

fn dims(t: &InklingTextConfig) -> Dims {
    Dims {
        h: t.hidden_size,
        nh: t.num_attention_heads,
        kv: t.num_key_value_heads,
        hd: t.head_dim,
        dr: t.d_rel,
        ext: t.rel_extent,
        k: t.sconv_kernel_size,
        e: t.n_routed_experts,
        inter: t.intermediate_size,
        layers: t.num_hidden_layers,
        block: t.n_layers_per_attn_res_block,
        vocab: t.vocab_size,
        unpadded: t.unpadded_vocab_size,
    }
}

/// The routed set's size in EXPERTS — what both residency arms are counted against.
pub fn expert_total(cfg: &InklingTextConfig) -> usize {
    let d = dims(cfg);
    d.layers * d.e
}

/// One routed expert's identity — the seed of its bytes, so the repack's source and the bf16
/// sibling's slab read the SAME expert without a shared buffer or an ordering convention.
#[derive(Clone, Copy)]
struct ExpertId {
    layer: usize,
    expert: usize,
}

impl ExpertId {
    fn seed(self) -> u64 {
        (self.layer as u64) << 8 | self.expert as u64
    }
}

/// A deterministic draw. Seeded per tensor (or per expert), so writing the two artifacts in
/// different orders cannot move a value.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// SplitMix64's avalanche, from [`splitmix_finalize`] — ONE owner for the mixing, shared with
    /// every other fixture in the tree that needs a draw; this struct owns only the counter.
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        splitmix_finalize(self.0)
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u32 << 24) as f32
    }

    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }

    /// A finite, non-NaN e4m3 byte: exponent field 6 (so the value is `±(1+m/8)·0.5`), both
    /// signs. e4m3's NaN codes are `0x7F`/`0xFF` and neither is reachable from this construction.
    fn e4m3(&mut self) -> u8 {
        let hi = if self.next() & 1 == 1 { 0xB0 } else { 0x30 };
        hi | (self.next() & 7) as u8
    }
}

// ---------------------------------------------------------------------------------------
// the directory
// ---------------------------------------------------------------------------------------

/// A scratch directory that removes itself. Named for the process and the caller's tag, so two
/// runs on this shared machine cannot collide the way `/tmp/<fixed>` would.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(tag: &str) -> Self {
        let root = Path::new(SCRATCH_ROOT);
        assert!(
            root.is_dir(),
            "{SCRATCH_ROOT} is missing — this gate's fixture is O_DIRECT-read, so it cannot live \
             on a tmpfs; create the scratch subvolume instead of moving the test"
        );
        let dir = root.join(format!(
            "test-inkling-small-mxfp8-{}-{tag}",
            std::process::id()
        ));
        if dir.exists() {
            std::fs::remove_dir_all(&dir).expect("clear a stale fixture");
        }
        std::fs::create_dir_all(&dir).expect("create the fixture directory");
        Self(dir)
    }

    pub fn path(&self) -> String {
        self.0.to_str().expect("scratch is UTF-8").to_owned()
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        // A destructor must not panic, and it must not hide a leak either: a file still mapped
        // by the pool on a hard abort makes the removal fail, so the failure is reported and
        // the next run's `Scratch::new` clears the directory instead.
        if let Err(e) = std::fs::remove_dir_all(&self.0) {
            eprintln!("scratch not removed ({}): {e}", self.0.display());
        }
    }
}

// ---------------------------------------------------------------------------------------
// the config
// ---------------------------------------------------------------------------------------

/// A tiny, VALID Small wrapper: the shipped member's root pair and quant header, with the
/// dimensions narrowed to what a gate can afford. [`write_config`] loads it through
/// `Inkling2Config`, whose validator runs the member split, the GQA divisibility, the DERIVED
/// `local_layer_ids` complement and the MXFP8 block alignment — so a fixture that drifted out of
/// shape fails here rather than mid-decode.
fn config_json() -> serde_json::Value {
    serde_json::json!({
        "model_type": "inkling2",
        "architectures": ["Inkling2ForConditionalGeneration"],
        "text_config": {
            "model_max_length": 128,
            "torch_dtype": "bfloat16",
            "hidden_size": HIDDEN,
            "num_hidden_layers": LAYERS,
            "vocab_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "d_rel": 16,
            "rel_extent": 32,
            "q_bias": false,
            "o_bias": false,
            "log_scaling_n_floor": 128,
            "log_scaling_alpha": 0.1,
            "rms_norm_eps": 0.000001,
            "pre_norm_pre_affine": false,
            "final_norm_pre_affine": false,
            "use_value_norm_no_gamma": true,
            "qk_l2_clip_threshold": 22.627416997969522,
            "rel_l2_clip_threshold": 8.0,
            "local_layer_ids": [0, 1, 2, 3, 4],
            "sconv_kernel_size": 4,
            "unpadded_vocab_size": 60,
            "final_logit_softcapping": 30.0,
            "swa_head_dim": 32,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "sliding_window_size": 32,
            "n_layers_per_attn_res_block": BLOCK,
            "attn_res_normalize_values": true,
            "n_routed_experts": N_EXPERTS,
            "num_experts_per_tok": 2,
            "n_shared_experts": 0,
            "intermediate_size": INTER,
            "mlp_gate_softcap": 4.0,
            "mlp_up_softcap": 25.0,
            "route_scale": 8.0,
            "use_gate_bias": true
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [1, 32],
            "scale_fmt": "ue8m0",
            "modules_to_not_convert": [
                "model.llm.unembed",
                "model.visual",
                "model.llm.layers.*.attn.*"
            ]
        }
    })
}

/// Write `config.json` into `dir` and load it back — the same door the CLI walks.
pub fn write_config(dir: &str) -> Inkling2Config {
    let doc = config_json();
    std::fs::write(
        format!("{dir}/config.json"),
        serde_json::to_vec_pretty(&doc).expect("the fixture config is JSON"),
    )
    .expect("write config.json");
    Inkling2Config::load(dir).expect("the fixture config must pass the Small validator")
}

// ---------------------------------------------------------------------------------------
// the resident tensors
// ---------------------------------------------------------------------------------------

/// One tensor's identity: what `SafeWriter::add` needs besides the payload.
struct Tensor {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
}

fn add_tensor(w: &mut SafeWriter<'_>, t: Tensor, vals: &[f32]) {
    let bytes: Vec<u8> = match t.dtype {
        Dtype::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
        _ => vals
            .iter()
            .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
            .collect(),
    };
    w.add(t.name, t.dtype, TensorData::new(t.shape, bytes));
}

/// The shape and storage of one layer tensor, by its suffix — the storage spelling
/// [`names::layer_tensor_names`] expands. `None` for the routed experts, which are not resident.
fn shape_of(suffix: &str, d: &Dims, full: bool) -> Option<(Dtype, Vec<usize>)> {
    let (nh, kv, hd) = (d.nh, d.kv, d.hd);
    Some(match suffix {
        names::Q_LINEAR => (Dtype::Bf16, vec![nh * hd, d.h]),
        names::Q_NORM => (Dtype::F32, vec![nh * hd]),
        names::K_LINEAR | names::V_LINEAR => (Dtype::Bf16, vec![kv * hd, d.h]),
        names::K_SCONV_STATIC | names::V_SCONV_STATIC => (Dtype::F32, vec![kv * hd, 1, d.k]),
        names::K_SCONV_TAPS | names::V_SCONV_TAPS => (Dtype::F32, vec![d.k, kv * hd]),
        names::K_SCONV_PROJ | names::V_SCONV_PROJ => (Dtype::Bf16, vec![kv * d.k, d.h]),
        names::O_GATE => (Dtype::Bf16, vec![nh * hd, d.h]),
        names::O_LINEAR => (Dtype::Bf16, vec![d.h, nh * hd]),
        names::SINK => (Dtype::Bf16, vec![nh, d.h]),
        names::REL_PROJ => (Dtype::Bf16, vec![d.ext, d.dr]),
        names::REL_LINEAR => (Dtype::Bf16, vec![nh * d.dr, d.h]),
        names::REL_NORM => (Dtype::F32, vec![nh * d.dr]),
        names::POST_ATTN_NORM | names::SLIDING_INPUT_NORM => (Dtype::F32, vec![d.h]),
        names::POST_ATTN_CONV | names::POST_MLP_CONV => (Dtype::F32, vec![d.h, 1, d.k]),
        names::MLP_INPUT_QUERY => (Dtype::F32, vec![d.h]),
        names::ATTENTION_INPUT_QUERY if full => (Dtype::F32, vec![3, d.h]),
        names::ATTENTION_INPUT_QUERY => (Dtype::F32, vec![d.h]),
        names::ROUTER => (Dtype::Bf16, vec![d.e, d.h]),
        names::ROUTER_BIAS => (Dtype::F32, vec![d.e]),
        _ if names::FULL_INPUT_NORMS.contains(&suffix) => (Dtype::F32, vec![d.h]),
        _ => return None,
    })
}

/// Values for one tensor: norms at 1.0 (a scaled draw would pass the placement checks and then
/// feed the residual stream zeros), convolutions and projections small, so the decode is finite
/// and no two logits are born tied.
fn values(suffix: &str, n: usize, rng: &mut Lcg) -> Vec<f32> {
    if suffix.contains("norm") {
        return vec![1.0; n];
    }
    let a = if suffix.contains("conv") { 0.02 } else { 0.05 };
    (0..n).map(|_| rng.sym(a)).collect()
}

fn globals(w: &mut SafeWriter<'_>, d: &Dims) {
    let mut rng = Lcg::new(0x600D);
    let g = names::GLOBAL_SUFFIXES;
    for (name, shape, dtype) in [
        (g[0], vec![d.vocab, d.h], Dtype::Bf16),
        (g[1], vec![d.h], Dtype::F32),
        (g[2], vec![d.h], Dtype::F32),
        (g[3], vec![d.h], Dtype::F32),
        (g[5], vec![d.unpadded, d.h], Dtype::Bf16),
    ] {
        let n = shape.iter().product();
        let vals = values(name, n, &mut rng);
        add_tensor(
            w,
            Tensor {
                name: name.to_owned(),
                dtype,
                shape,
            },
            &vals,
        );
    }
}

fn layer_dense(w: &mut SafeWriter<'_>, d: &Dims, layer: usize) {
    let full = is_block_end(layer, d.block);
    let prefix = names::layer_prefix(layer);
    let mut rng = Lcg::new(layer as u64 + 1);
    for name in names::layer_tensor_names(layer, full, false) {
        let Some(suffix) = name.strip_prefix(&prefix).and_then(|s| s.strip_prefix('.')) else {
            continue;
        };
        let Some((dtype, shape)) = shape_of(suffix, d, full) else {
            continue;
        };
        let n = shape.iter().product();
        let vals = values(suffix, n, &mut rng);
        add_tensor(w, Tensor { name, dtype, shape }, &vals);
    }
}

// ---------------------------------------------------------------------------------------
// the experts — one author for the bytes, read by the repack and by the bf16 sibling
// ---------------------------------------------------------------------------------------

/// The three projections' `(o_dim, i_dim)` in slot order: gate, up, down.
fn proj_dims(d: &Dims) -> [(usize, usize); 3] {
    [(d.inter, d.h), (d.inter, d.h), (d.h, d.inter)]
}

/// The scale byte the fixture writes: `0x7F` is `2^0`, so an expert's value IS its e4m3 decode
/// and the cross-check compares representations rather than a power of two chosen here. It is a
/// legal ue8m0 (`0xFF`, the NaN, is never written — the repack refuses it).
const UNIT_SCALE: u8 = 0x7F;

/// One expert's on-disk block, laid out by `mxfp8_slot_offsets` — packed e4m3 rows then one
/// ue8m0 byte per 32 weights, per projection, in `[gate, gate_scale, up, up_scale, down,
/// down_scale]` order. This is the layout the repack writes, and the byte sequence the bf16
/// sibling dequantizes.
fn expert_block(d: &Dims, id: ExpertId) -> Vec<u8> {
    let off = mxfp8_slot_offsets(d.h, d.inter);
    let mut out = vec![0u8; mxfp8_expert_bytes(d.h, d.inter)];
    let mut rng = Lcg::new(id.seed());
    for (p, (o, i)) in proj_dims(d).into_iter().enumerate() {
        let packed = &mut out[off[2 * p]..off[2 * p] + o * i];
        for b in packed.iter_mut() {
            *b = rng.e4m3();
        }
        let scale = &mut out[off[2 * p + 1]..off[2 * p + 1] + o * mxfp8_groups(i)];
        scale.fill(UNIT_SCALE);
    }
    out
}

/// The block's values, `[gate, up, down]`, row-major — the test's OWN e4m3×ue8m0 decoder, so the
/// bf16 sibling is not built by calling the kernel's LUT the MXFP8 arm is then scored against.
fn dequant_block(d: &Dims, block: &[u8]) -> [Vec<f32>; 3] {
    let off = mxfp8_slot_offsets(d.h, d.inter);
    let mut out: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for (p, (o, i)) in proj_dims(d).into_iter().enumerate() {
        let packed = &block[off[2 * p]..off[2 * p] + o * i];
        let scale = &block[off[2 * p + 1]..off[2 * p + 1] + o * mxfp8_groups(i)];
        let groups = mxfp8_groups(i);
        let mut vals = Vec::with_capacity(o * i);
        for row in 0..o {
            for col in 0..i {
                let s = e8m0(scale[row * groups + col / 32]);
                vals.push(e4m3(packed[row * i + col]) * s);
            }
        }
        out[p] = vals;
    }
    out
}

fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((b >> 3) & 0x0F);
    let man = f32::from(b & 0x07) / 8.0;
    if exp == 0 {
        sign * man * 2f32.powi(-6)
    } else {
        sign * (1.0 + man) * 2f32.powi(exp - 7)
    }
}

fn e8m0(s: u8) -> f32 {
    2f32.powi(i32::from(s) - 127)
}

/// One layer's four SOURCE tensors, in the checkpoint's own spelling: `w13` is
/// `[E, 2·inter, hidden]` with gate/up rows INTERLEAVED (`[gate_0, up_0, gate_1, …]`) and its
/// scale grid interleaved the same way, `w2` is `[E, hidden, inter]`. That interleaving is the
/// fact `Mxfp8Layer` exists to undo, and building the source in it is what makes this fixture
/// exercise the real repack rather than a layout the test invented.
struct LayerSource {
    w13: Vec<u8>,
    w13_scale: Vec<u8>,
    w2: Vec<u8>,
    w2_scale: Vec<u8>,
}

impl LayerSource {
    fn new(d: &Dims) -> Self {
        Self {
            w13: vec![0u8; d.e * 2 * d.inter * d.h],
            w13_scale: vec![0u8; d.e * 2 * d.inter * mxfp8_groups(d.h)],
            w2: vec![0u8; d.e * d.h * d.inter],
            w2_scale: vec![0u8; d.e * d.h * mxfp8_groups(d.inter)],
        }
    }

    /// Scatter expert `id`'s flat block into the interleaved rows, using the SAME offsets the
    /// repack reads back: `off[0]`/`off[1]` are gate's packed and scale spans, `off[2]`/`off[3]`
    /// up's, `off[4]`/`off[5]` down's.
    fn place(&mut self, d: &Dims, id: ExpertId, off: [usize; 6]) {
        let block = expert_block(d, id);
        let (rows, g) = (2 * d.inter, mxfp8_groups(d.h));
        for j in 0..d.inter {
            let row = id.expert * rows + 2 * j;
            // gate occupies even rows `2j`, up odd rows `2j+1` — the source's own interleave.
            for (slot, dst) in [(0usize, row), (2, row + 1)] {
                let src = off[slot] + j * d.h;
                self.w13[dst * d.h..][..d.h].copy_from_slice(&block[src..src + d.h]);
                let src = off[slot + 1] + j * g;
                self.w13_scale[dst * g..][..g].copy_from_slice(&block[src..src + g]);
            }
        }
        let (dn, dg) = (d.h * d.inter, mxfp8_groups(d.inter));
        self.w2[id.expert * dn..][..dn].copy_from_slice(&block[off[4]..off[4] + dn]);
        self.w2_scale[id.expert * d.h * dg..][..d.h * dg]
            .copy_from_slice(&block[off[5]..off[5] + d.h * dg]);
    }

    fn add(self, w: &mut SafeWriter<'_>, prefix: &str, d: &Dims) {
        let (rows, g, dg) = (2 * d.inter, mxfp8_groups(d.h), mxfp8_groups(d.inter));
        for (suffix, dtype, shape, bytes) in [
            ("w13_weight", Dtype::F8E4M3, vec![d.e, rows, d.h], self.w13),
            (
                "w13_weight_scale",
                Dtype::U8,
                vec![d.e, rows, g],
                self.w13_scale,
            ),
            ("w2_weight", Dtype::F8E4M3, vec![d.e, d.h, d.inter], self.w2),
            (
                "w2_weight_scale",
                Dtype::U8,
                vec![d.e, d.h, dg],
                self.w2_scale,
            ),
        ] {
            let name = format!("{prefix}{suffix}");
            w.add(name, dtype, TensorData::new(shape, bytes));
        }
    }
}

/// The resident expert slabs the bf16 sibling reads: `[E, 2·inter, hidden]` (gate rows then up
/// rows — the order `place_experts` and the forward's split assume) and `[E, hidden, inter]`.
fn expert_slabs(w: &mut SafeWriter<'_>, d: &Dims) {
    for layer in 0..d.layers {
        let mut gu = Vec::with_capacity(d.e * 2 * d.inter * d.h);
        let mut dn = Vec::with_capacity(d.e * d.h * d.inter);
        for expert in 0..d.e {
            let [gate, up, down] = dequant_block(d, &expert_block(d, ExpertId { layer, expert }));
            gu.extend_from_slice(&gate);
            gu.extend_from_slice(&up);
            dn.extend_from_slice(&down);
        }
        let p = names::layer_prefix(layer);
        for (suffix, shape, vals) in [
            (names::EXPERTS_GATE_UP, vec![d.e, 2 * d.inter, d.h], gu),
            (names::EXPERTS_DOWN, vec![d.e, d.h, d.inter], dn),
        ] {
            let name = format!("{p}.{suffix}");
            add_tensor(
                w,
                Tensor {
                    name,
                    dtype: Dtype::Bf16,
                    shape,
                },
                &vals,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------
// the artifact
// ---------------------------------------------------------------------------------------

/// Which container the experts live in: Small's real `.mxfp8` files, or the cross-check's
/// resident bf16 slabs.
pub enum Storage {
    MxFp8,
    Bf16,
}

/// Build `dir` as a complete artifact. `MxFp8` writes a source file in the checkpoint's spelling,
/// runs the real repack over it (with `verify`, so each file is read back and byte-compared), and
/// stamps the manifest's `mxfp8_source`; `Bf16` is resident-only (Mini's container, Small's
/// values).
pub fn build(scratch: &Scratch, storage: Storage) -> (String, Inkling2Config) {
    let dir = scratch.path();
    let cfg = write_config(&dir);
    let d = dims(&cfg.text);
    let mut resident = SafeWriter::new();
    globals(&mut resident, &d);
    for layer in 0..d.layers {
        layer_dense(&mut resident, &d, layer);
    }
    if matches!(storage, Storage::Bf16) {
        expert_slabs(&mut resident, &d);
    }
    resident
        .write(&format!("{dir}/resident.safetensors"))
        .expect("write resident.safetensors");
    if matches!(storage, Storage::MxFp8) {
        repack(&dir, &d);
    }
    (dir, cfg)
}

/// Write the source tensors, then let [`Mxfp8Layer`] produce `L{ll}.mxfp8` for each layer and
/// byte-verify what it wrote. The source lives BESIDE the output and is left in place: it is the
/// fixture's provenance.
fn repack(dir: &str, d: &Dims) {
    let mut src = SafeWriter::new();
    let off = mxfp8_slot_offsets(d.h, d.inter);
    for layer in 0..d.layers {
        let mut source = LayerSource::new(d);
        for expert in 0..d.e {
            source.place(d, ExpertId { layer, expert }, off);
        }
        source.add(&mut src, &mxfp8_expert_prefix(layer), d);
    }
    let src_path = format!("{dir}/source.safetensors");
    src.write(&src_path).expect("write the repack source");
    let st = Safetensors::open_file(&src_path).expect("open the repack source");
    for layer in 0..d.layers {
        // Fields grouped as (which layer's geometry, then the container, then the policies)
        // rather than in the type's declaration order: a nine-field literal is name-keyed, so
        // the order carries nothing, and an identical one in a second file is a clone jscpd
        // reports with no reader's benefit.
        Mxfp8Layer {
            layer,
            n_experts: d.e,
            expert_in: d.h,
            moe_inter: d.inter,
            tool: "synthetic",
            out_dir: dir,
            src: &st,
            write: true,
            verify: true,
        }
        .run()
        .expect("the repack writes and verifies one layer");
    }
    let mut manifest = FormatMeta::manifest_from_config(dir, 128).expect("stamp the format");
    manifest["mxfp8_source"] = mxfp8_source("synthetic", dir, 0..d.layers);
    std::fs::write(
        format!("{dir}/manifest.json"),
        serde_json::to_vec(&manifest).expect("manifest is JSON"),
    )
    .expect("write manifest.json");
}
