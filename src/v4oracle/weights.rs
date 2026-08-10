//! Configuration, weight storage, and the two weight sources the oracle runs against:
//! the real safetensors checkpoint, and a small synthetic model with the same *structure*.
//!
//! **Why there are two sources.** The defect matrix in `tests/v4_oracle.rs` has to run ~30
//! deliberate breakages across several layer classes and both phases. On the real
//! checkpoint that is hours and 167 GB; on a structurally-identical toy it is under a
//! second and needs nothing on disk. The question the matrix answers — "can this gate
//! fire, and does it stay silent where it should" — is about STRUCTURE, not about the
//! trained values, so the toy answers it exactly as well. The real checkpoint is what
//! `bin/v4-oracle` emits goldens from, and it re-runs the same matrix as a cross-check so
//! the toy's verdict is not taken on trust.

use crate::v4oracle::numerics::{bf16_decode, bf16_encode, e2m1_decode, e4m3_decode, e8m0_decode};
use anyhow::{Context, Result, anyhow, bail};
use std::collections::HashMap;
use std::path::Path;

/// `model.py::ModelArgs`, restricted to the main path. DSpark/MTP fields are deliberately
/// absent — `forward_spec` is out of scope (v4-flash-port.md §"Scope cut").
///
/// Same name as `artifact::model::V4Config` and deliberately NOT the same type — the oracle
/// must not share code with the engine it judges (`mod.rs` states the rule). Kept
/// model-named through the 2026-08-09 rename pass: a transliteration's value is fidelity to
/// one reference file, not generality.
#[derive(Clone, Debug, PartialEq)]
pub struct V4Config {
    pub vocab_size: usize,
    pub dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_hash_layers: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_activated_experts: usize,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub q_lora_rank: usize,
    pub head_dim: usize,
    pub rope_head_dim: usize,
    pub norm_eps: f32,
    pub o_groups: usize,
    pub o_lora_rank: usize,
    pub window_size: usize,
    pub compress_ratios: Vec<usize>,
    pub compress_rope_theta: f32,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub index_n_heads: usize,
    pub index_head_dim: usize,
    pub index_topk: usize,
    pub hc_mult: usize,
    pub hc_sinkhorn_iters: usize,
    pub hc_eps: f32,
    /// `max_seq_len` sizes the ring cache and the compressed region. The reference defaults
    /// to 4096; the oracle only ever runs a few dozen positions.
    pub max_seq_len: usize,
}

impl V4Config {
    /// The shipped DeepSeek-V4-Flash-0731 configuration, from `inference/config.json`.
    ///
    /// Hard-coded rather than parsed so the oracle has no dependency on S1a's config
    /// loader — the instrument must not share code with the thing it judges. It is not
    /// left to drift: `bin/v4-oracle` calls [`V4Config::assert_matches_reference_json`]
    /// against the on-disk file before it reads a single weight, so any divergence stops
    /// the run rather than producing a quietly-wrong golden.
    pub fn v4_flash() -> Self {
        Self {
            vocab_size: 129280,
            dim: 4096,
            moe_inter_dim: 2048,
            n_layers: 43,
            n_hash_layers: 3,
            n_heads: 64,
            n_routed_experts: 256,
            n_activated_experts: 6,
            route_scale: 1.5,
            swiglu_limit: 10.0,
            q_lora_rank: 1024,
            head_dim: 512,
            rope_head_dim: 64,
            norm_eps: 1e-6,
            o_groups: 8,
            o_lora_rank: 1024,
            window_size: 128,
            compress_ratios: {
                // `[0, 0]`, then 41 entries alternating `4, 128` and ENDING on a 4 (so
                // layer 42, the last one, is ratio 4 and has an indexer), then three 0s that
                // the 43-layer model never reaches. 46 entries in the config.
                //
                // The obvious `for i in 0..40` writes 40 alternating entries and leaves
                // layer 42 at 0 -- which silently deletes that layer's compressor AND its
                // indexer. Caught by `assert_matches_reference_json`, which is exactly why
                // that check runs before any weight is read rather than being a comment.
                let mut v = vec![0, 0];
                for i in 0..41 {
                    v.push(if i % 2 == 0 { 4 } else { 128 });
                }
                v.extend_from_slice(&[0, 0, 0]);
                v
            },
            compress_rope_theta: 160000.0,
            original_seq_len: 65536,
            rope_theta: 10000.0,
            rope_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            index_n_heads: 64,
            index_head_dim: 128,
            index_topk: 512,
            hc_mult: 4,
            hc_sinkhorn_iters: 20,
            hc_eps: 1e-6,
            max_seq_len: 4096,
        }
    }

    /// A structurally-identical miniature: every discriminant the main path branches on is
    /// preserved, every extent is shrunk.
    ///
    /// Preserved on purpose, because a defect that only shows up through one of these
    /// would otherwise be untestable here:
    /// - three layer classes in the same order (`ratio 0, 0, 4, r`), so layer 2 has an
    ///   indexer and layer 3 does not — `Indexer` exists only where `ratio == 4`;
    /// - `n_hash_layers = 3`, so layers 0..2 route by `tid2eid` and layer 3 by score;
    /// - `head_dim - rope_head_dim` divisible by 64, so the PARTIAL kv `act_quant` has a
    ///   whole number of blocks and a non-empty un-quantized tail;
    /// - every `Linear` input dimension divisible by 128 (`act_quant`'s block) and every
    ///   expert input dimension divisible by 32 (fp4's group);
    /// - `hc_mult = 4` exactly, since `hc_split_sinkhorn`'s `mix_hc = (2 + hc) * hc`
    ///   layout is what the mHC weights are packed in.
    ///
    /// Shrunk on purpose: `window_size` 8 and the second compress ratio 8 (not 128), so a
    /// 12-token prompt reaches BOTH the ring-cache wrap and the ratio-r compressor. At the
    /// real 128/128 neither is reachable without a 129-token prompt.
    pub fn toy() -> Self {
        Self {
            vocab_size: 64,
            dim: 256,
            moe_inter_dim: 128,
            n_layers: 4,
            // 8 heads over 4 groups, so `n_heads / o_groups == 2`. NOT 4-over-4: at one head
            // per group `WoGroupsInterleaved` degenerates into the correct grouping and the
            // defect becomes structurally unable to fire -- which is exactly the "guard that
            // cannot fail" this whole exercise exists to avoid.
            n_heads: 8,
            n_routed_experts: 8,
            n_activated_experts: 2,
            q_lora_rank: 128,
            // 256 gives an un-roped span of 192 = 3 x 64, so the PARTIAL kv `act_quant`
            // has several whole blocks and a non-empty tail rather than a single block.
            //
            // It was chosen to make `KvActQuantBlock128` fire; it does NOT, and no head_dim
            // would --
            // `act_quant_block_size_is_almost_invisible_under_ue8m0_scales` shows why: a
            // ue8m0 scale is a power of two and e4m3 is scale-invariant under those, so
            // re-blocking changes nothing until an in-block dynamic range of ~2^13. Keep the
            // value for the reason above, not for that one.
            head_dim: 256,
            o_groups: 4,
            o_lora_rank: 128,
            window_size: 8,
            compress_ratios: vec![0, 0, 4, 8],
            original_seq_len: 512,
            index_n_heads: 4,
            // A power of two, because the indexer Hadamard-rotates rows of this width.
            index_head_dim: 128,
            // 2, so `index_topk.min(n_comp)` actually TRUNCATES at the long prompt
            // (n_comp reaches 4) and not at the short one. At 4 the top-k selected every
            // compressed block in every case: the set was invariant, `.compress_idxs` could
            // not distinguish a right ranking from a wrong one, and `sparse_attn` never ran
            // sparsely. That is the trap MEMORY.md records as "a dsa A/B under 2048 tokens
            // covers nothing" -- a selector that passes vacuously below its threshold.
            index_topk: 2,
            max_seq_len: 64,
            // Everything not named above is INHERITED from the real config, not re-typed:
            // `n_hash_layers`, `hc_mult`, `hc_sinkhorn_iters`, `swiglu_limit`, `route_scale`,
            // `rope_theta`/`compress_rope_theta`, the betas and both epsilons. A toy that
            // re-declared them could drift from the model it is supposed to stand in for,
            // and the drift would show up as a defect-matrix result that means nothing.
            ..Self::v4_flash()
        }
    }

    pub fn mix_hc(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }
    pub fn hc_dim(&self) -> usize {
        self.hc_mult * self.dim
    }
    pub fn compress_ratio(&self, layer: usize) -> usize {
        self.compress_ratios.get(layer).copied().unwrap_or(0)
    }

    /// Fail loudly if the hard-coded constants above have drifted from the reference's own
    /// `inference/config.json`. Called by `bin/v4-oracle` before any weight is read.
    ///
    /// `dspark_*` is deliberately ignored -- out of scope, and its presence must not fail.
    ///
    /// Three main-path fields are NOT checked and cannot be: `norm_eps` and `hc_eps` do not
    /// appear in `inference/config.json` at all (they are `ModelArgs` defaults, 1e-6 both),
    /// and `max_seq_len` is a runtime sizing knob this oracle overrides. Everything else the
    /// main path reads is compared.
    pub fn assert_matches_reference_json(&self, json_path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(json_path)
            .with_context(|| format!("reading {}", json_path.display()))?;
        let j: serde_json::Value = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", json_path.display()))?;
        let mut bad = Vec::new();
        {
            let mut num =
                |key: &str, mine: f64| match j.get(key).and_then(serde_json::Value::as_f64) {
                    Some(theirs) if theirs == mine => {}
                    Some(theirs) => bad.push(format!("{key}: oracle {mine} vs config {theirs}")),
                    None => bad.push(format!("{key}: absent from config, oracle assumes {mine}")),
                };
            num("vocab_size", self.vocab_size as f64);
            num("dim", self.dim as f64);
            num("moe_inter_dim", self.moe_inter_dim as f64);
            num("n_layers", self.n_layers as f64);
            num("n_hash_layers", self.n_hash_layers as f64);
            num("n_heads", self.n_heads as f64);
            num("n_routed_experts", self.n_routed_experts as f64);
            num("n_activated_experts", self.n_activated_experts as f64);
            num("route_scale", self.route_scale as f64);
            num("swiglu_limit", self.swiglu_limit as f64);
            num("q_lora_rank", self.q_lora_rank as f64);
            num("head_dim", self.head_dim as f64);
            num("rope_head_dim", self.rope_head_dim as f64);
            num("o_groups", self.o_groups as f64);
            num("o_lora_rank", self.o_lora_rank as f64);
            num("window_size", self.window_size as f64);
            num("compress_rope_theta", self.compress_rope_theta as f64);
            num("original_seq_len", self.original_seq_len as f64);
            num("rope_theta", self.rope_theta as f64);
            num("rope_factor", self.rope_factor as f64);
            num("beta_fast", self.beta_fast as f64);
            num("beta_slow", self.beta_slow as f64);
            num("index_n_heads", self.index_n_heads as f64);
            num("index_head_dim", self.index_head_dim as f64);
            num("index_topk", self.index_topk as f64);
            num("hc_mult", self.hc_mult as f64);
            num("hc_sinkhorn_iters", self.hc_sinkhorn_iters as f64);
        }
        match j.get("score_func").and_then(serde_json::Value::as_str) {
            // The oracle implements exactly one scoring function. A checkpoint that asked
            // for another would make every routing golden wrong, silently.
            Some("sqrtsoftplus") => {}
            other => bad.push(format!(
                "score_func: oracle implements sqrtsoftplus, config says {other:?}"
            )),
        }
        let theirs: Vec<usize> = j
            .get("compress_ratios")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|u| u as usize))
                    .collect()
            })
            .unwrap_or_default();
        if theirs.len() < self.n_layers
            || theirs[..self.n_layers] != self.compress_ratios[..self.n_layers]
        {
            bad.push(format!(
                "compress_ratios: oracle {:?} vs config {:?}",
                &self.compress_ratios[..self.n_layers.min(self.compress_ratios.len())],
                theirs
            ));
        }
        if bad.is_empty() {
            Ok(())
        } else {
            bail!(
                "V4Config::v4_flash() has drifted from {}:\n  {}",
                json_path.display(),
                bad.join("\n  ")
            )
        }
    }
}

/// A weight matrix in the storage format the checkpoint actually uses, kept quantized so
/// the oracle's GEMM applies the scales the way the reference's kernels do.
#[derive(Clone)]
pub enum WMat {
    /// bf16 or f32 in the checkpoint. `linear()`'s `else` branch: **no activation
    /// quantization**. Used for `compressor.{wkv,wgate}`, `indexer.weights_proj`,
    /// `gate.weight`, and `attn.wo_a` (which is fp8 on disk but is consumed raw by the
    /// einsum — see `forward.rs`).
    Dense {
        rows: usize,
        cols: usize,
        v: Vec<f32>,
    },
    /// `float8_e4m3fn` weights with `float8_e8m0fnu` scales on a 128x128 grid.
    /// `linear()` quantizes the activation to fp8 at block 128 before the GEMM.
    Fp8 {
        rows: usize,
        cols: usize,
        w: Vec<u8>,
        s: Vec<u8>,
    },
    // jscpd:ignore-start
    //
    // THESE FOUR FIELDS ARE THE ON-DISK SCHEMA, NOT A COPY-PASTE, so the gate is switched
    // off over this variant. Blanking one side of a pair is enough for jscpd, so the
    // `Fp8` declaration above and the `Checkpoint::fp8` construction below carry no marker;
    // the same argument covers all three sites. The companion region is at
    // `Checkpoint::fp8`.
    //
    // The checkpoint stores both quantized formats as (rows, cols, weight bytes, scale
    // bytes). Only the scale GRID differs -- 128x128 blocks for fp8, 32 elements of K per
    // output row for fp4 -- and nothing in the bytes says which, so the variant has to carry
    // it: decoding an fp4 tensor against the fp8 grid reads real scales at wrong strides and
    // yields finite, plausible, wrong weights on every expert.
    //
    // The factoring that would remove the text is `Fp8(Q)`/`Fp4(Q)` over one payload struct.
    // It does keep the distinction -- the variant still discriminates every `match` in
    // `WMat` -- so this is not a safety argument; it is that the hop costs a `.0` in every
    // arm of `rows`/`cols`/`row` and at every construction and pattern site in `tests/`, to
    // satisfy a 15-token window that cannot tell a schema from a duplication. Two payload
    // STRUCTS (`Blocked`/`Grouped`) would not even do that -- same clone one level down.
    /// `float4_e2m1fn_x2` weights (two nibbles per byte along K) with `float8_e8m0fnu`
    /// scales per 32 elements of K. `linear()` quantizes the activation to fp8 at block 128.
    Fp4 {
        rows: usize,
        cols: usize,
        w: Vec<u8>,
        s: Vec<u8>,
    },
    // jscpd:ignore-end
}

impl WMat {
    pub fn rows(&self) -> usize {
        match self {
            WMat::Dense { rows, .. } | WMat::Fp8 { rows, .. } | WMat::Fp4 { rows, .. } => *rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            WMat::Dense { cols, .. } | WMat::Fp8 { cols, .. } | WMat::Fp4 { cols, .. } => *cols,
        }
    }

    /// One dequantized row of the weight, `cols` values. The scale grid differs per format;
    /// this is the only place that knows how.
    pub fn row(&self, r: usize, out: &mut Vec<f32>) {
        out.clear();
        match self {
            WMat::Dense { cols, v, .. } => out.extend_from_slice(&v[r * cols..(r + 1) * cols]),
            WMat::Fp8 { cols, w, s, .. } => {
                // 128x128 blocks: scale row r/128, scale column k/128.
                let sb_cols = cols.div_ceil(128);
                let base = (r / 128) * sb_cols;
                for k in 0..*cols {
                    out.push(e4m3_decode(w[r * cols + k]) * e8m0_decode(s[base + k / 128]));
                }
            }
            WMat::Fp4 { cols, w, s, .. } => {
                // group-32 along K only; every output row has its own scale row.
                let sb_cols = cols / 32;
                let bytes = cols / 2;
                for k in 0..*cols {
                    let byte = w[r * bytes + k / 2];
                    // LOW nibble carries the EVEN (lower) K index. Not a convention we get
                    // to pick: `inference/convert.py::cast_e2m1fn_to_e4m3fn` unpacks
                    // `stack([TABLE[low], TABLE[high]]).flatten()`. Getting it backwards is
                    // a permutation INSIDE each 32-element scale group, so group
                    // boundaries, the amax/scale relation and the code histogram are all
                    // invariant under it -- no statistic can find it, only end-to-end
                    // quality. `Defect::Fp4NibbleSwap` keeps that A/B-able.
                    let nib = if k % 2 == 1 { byte >> 4 } else { byte & 0x0f };
                    out.push(e2m1_decode(nib) * e8m0_decode(s[r * sb_cols + k / 32]));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------
// safetensors
// ---------------------------------------------------------------------------------------

/// The subset of dtypes this checkpoint uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    Bf16,
    F32,
    F8E4M3,
    F8E8M0,
    I8,
    I64,
}

impl Dtype {
    /// Narrow the `safetensors` crate's dtype to the six this checkpoint uses. Kept as its
    /// own closed enum rather than a re-export for the reason given on the engine side: the
    /// crate's is `#[non_exhaustive]`, so every match would need a `_` arm — and 0.8.0's
    /// `F8_E4M3FNUZ` is a live example of a dtype whose bytes are indistinguishable from
    /// `F8_E4M3` but whose exponent bias is not. The oracle refuses it here for the same
    /// reason the engine does, and refusing it SEPARATELY is the point: a shared narrowing
    /// would make one misreading agree with itself across both implementations.
    fn narrow(d: safetensors::Dtype) -> Result<Self> {
        Ok(match d {
            safetensors::Dtype::BF16 => Dtype::Bf16,
            safetensors::Dtype::F32 => Dtype::F32,
            safetensors::Dtype::F8_E4M3 => Dtype::F8E4M3,
            safetensors::Dtype::F8_E8M0 => Dtype::F8E8M0,
            safetensors::Dtype::I8 => Dtype::I8,
            safetensors::Dtype::I64 => Dtype::I64,
            other => bail!("unhandled safetensors dtype {other:?}"),
        })
    }
}

pub struct RawTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl RawTensor {
    /// Decode to f32. Only meaningful for the unscaled dtypes; a scaled fp8/fp4 tensor
    /// must go through [`WMat`] instead, which is why this refuses them rather than
    /// returning plausible-looking garbage.
    pub fn to_f32(&self) -> Result<Vec<f32>> {
        Ok(match self.dtype {
            Dtype::Bf16 => self
                .bytes
                .chunks_exact(2)
                .map(|c| bf16_decode(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F32 => self
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            d => bail!(
                "{d:?} carries a separate scale tensor; decode it as a WMat, not an f32 array"
            ),
        })
    }

    pub fn to_i64(&self) -> Result<Vec<i64>> {
        if self.dtype != Dtype::I64 {
            bail!("expected I64, got {:?}", self.dtype);
        }
        Ok(self
            .bytes
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }
}

/// A minimal, independent safetensors reader over a sharded checkpoint.
///
/// Deliberately not `src/artifact/`'s loader: the oracle judges the engine, so sharing a
/// reader with it would make any misreading of the checkpoint invisible to both.
///
/// **NARROWED 2026-08-06.** Both this and `artifact::format::Safetensors` now parse the
/// header through the `safetensors` crate, so one third-party parser does sit on both sides
/// of that boundary. The independence claim above is unchanged and this is why: what the
/// oracle exists to catch is a disagreement about *arithmetic* — dequant, scale direction,
/// block tiling, reduction order — and none of that is in the framing. A framing bug is not
/// a quiet disagreement the two could share; it reads the wrong bytes and both sides produce
/// garbage loudly. The shard SELECTION, the dtype narrowing, and every decode below stay
/// separately written, which is where a shared misreading could actually hide.
pub struct Checkpoint {
    root: std::path::PathBuf,
    index: HashMap<String, String>,
}

impl Checkpoint {
    pub fn open(root: &Path) -> Result<Self> {
        let idx_path = root.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&idx_path)
            .with_context(|| format!("reading {}", idx_path.display()))?;
        let j: serde_json::Value = serde_json::from_str(&text)?;
        let map = j
            .get("weight_map")
            .and_then(serde_json::Value::as_object)
            .ok_or_else(|| anyhow!("{} has no weight_map", idx_path.display()))?;
        let index = map
            .iter()
            .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
            .collect();
        Ok(Self {
            root: root.to_path_buf(),
            index,
        })
    }

    /// Does the index carry any tensor under `prefix`? Used to *discover* what a layer
    /// carries rather than assuming it — that is how the loader cross-checks that
    /// `attn.indexer.*` exists exactly where `compress_ratio == 4`.
    pub fn has_prefix(&self, prefix: &str) -> bool {
        self.index.keys().any(|k| k.starts_with(prefix))
    }

    pub fn get(&self, name: &str) -> Result<RawTensor> {
        let shard = self
            .index
            .get(name)
            .ok_or_else(|| anyhow!("tensor {name} is not in the index"))?;
        let path = self.root.join(shard);
        let file = std::fs::File::open(&path).with_context(|| {
            format!(
                "opening {} for tensor {name} — if the download is still running this shard \
                 may not have landed yet",
                path.display()
            )
        })?;
        // SAFETY: the mapping is read-only and dropped before this function returns. The
        // usual mmap hazard — the file being truncated under us — is the same hazard the
        // rest of this repo accepts for the artifact, and a partially-written shard is
        // caught by the length check below rather than by a fault.
        let map = unsafe { memmap2::Mmap::map(&file) }
            .with_context(|| format!("mmapping {}", path.display()))?;
        // Everything this used to check by hand, `read_metadata` now checks for every tensor
        // in the shard: `8 + header + data == file length` exactly, offsets contiguous from
        // zero, and `end - begin == shape.product() * dtype.size()`. That last one was the
        // `want` comparison here, and it needed a `checked_sub` because `b - a` would WRAP in
        // release on a malformed header with `b < a`; the crate does the same subtraction
        // under a `e < s` guard. `Dtype::width` existed only to compute `want` and went with
        // it. The "shard is incomplete (still downloading?)" diagnosis does not survive in
        // the crate's error, so it is restored in the context below — the check is what
        // matters, but an unreadable message from a correct check gets misread as corruption.
        // It is offered as one cause among several, not as the diagnosis: `read_metadata`
        // rejects a malformed header too, and naming only truncation would send a reader
        // hunting a download that finished fine. anyhow appends the crate's own reason, which
        // is the half that says which.
        let (hdr_len, meta) = safetensors::SafeTensors::read_metadata(&map)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| {
                format!(
                    "{}: not a readable safetensors — the shard may be incomplete (still \
                     downloading), or its header malformed",
                    path.display()
                )
            })?;
        let info = meta.info(name).ok_or_else(|| {
            anyhow!("index says {name} is in {shard}, but its header has no such tensor")
        })?;
        let (a, b) = info.data_offsets;
        let hdr_end = 8 + hdr_len;
        Ok(RawTensor {
            dtype: Dtype::narrow(info.dtype).with_context(|| format!("tensor {name}"))?,
            shape: info.shape.clone(),
            bytes: map[hdr_end + a..hdr_end + b].to_vec(),
        })
    }

    /// A dense (bf16/f32) matrix as a [`WMat`].
    pub fn dense(&self, name: &str) -> Result<WMat> {
        let t = self.get(name)?;
        let (rows, cols) = two_dims(name, &t.shape)?;
        Ok(WMat::Dense {
            rows,
            cols,
            v: t.to_f32()?,
        })
    }

    /// An fp8 weight plus its `.scale`.
    pub fn fp8(&self, name: &str) -> Result<WMat> {
        let w = self.get(name)?;
        let s = self.get(&scale_name(name)?)?;
        let (rows, cols) = two_dims(name, &w.shape)?;
        if w.dtype != Dtype::F8E4M3 || s.dtype != Dtype::F8E8M0 {
            bail!(
                "{name}: expected F8_E4M3 + F8_E8M0 scale, got {:?} + {:?}",
                w.dtype,
                s.dtype
            );
        }
        let want = [rows.div_ceil(128), cols.div_ceil(128)];
        if s.shape != want {
            bail!(
                "{name}.scale: expected {want:?} for a 128x128 grid over {rows}x{cols}, got {:?}",
                s.shape
            );
        }
        // jscpd:ignore-start -- see the argument at `WMat::Fp4`. Identical payloads are
        // constructed identically; what differs is the shape check ABOVE each of these, and
        // it stays put so each names its own format in its own message. Blanking this side
        // is enough, so `Checkpoint::fp4`'s tail carries no marker.
        Ok(WMat::Fp8 {
            rows,
            cols,
            w: w.bytes,
            s: s.bytes,
        })
        // jscpd:ignore-end
    }

    /// An fp4 expert weight plus its `.scale`. The checkpoint stores the nibbles as `I8`.
    pub fn fp4(&self, name: &str) -> Result<WMat> {
        let w = self.get(name)?;
        let s = self.get(&scale_name(name)?)?;
        let (rows, packed) = two_dims(name, &w.shape)?;
        let cols = packed * 2;
        if s.shape != [rows, cols / 32] {
            bail!(
                "{name}.scale: expected [{rows}, {}] for group-32 over K={cols}, got {:?}",
                cols / 32,
                s.shape
            );
        }
        Ok(WMat::Fp4 {
            rows,
            cols,
            w: w.bytes,
            s: s.bytes,
        })
    }
}

/// `foo.weight` -> `foo.scale`. The checkpoint REPLACES the `.weight` suffix rather than
/// appending, which `inference/convert.py` produces via
/// `name.replace("weight_scale_inv", "scale")`. Appending instead yields
/// `foo.weight.scale`, which is simply absent -- a loud failure, but only because this
/// refuses names that do not end in `.weight` rather than guessing.
fn scale_name(name: &str) -> Result<String> {
    match name.strip_suffix(".weight") {
        Some(stem) => Ok(format!("{stem}.scale")),
        None => bail!("{name}: a quantized tensor's name must end in `.weight` to have a `.scale`"),
    }
}

fn two_dims(name: &str, shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [r, c] => Ok((*r, *c)),
        other => bail!("{name}: expected a 2-D tensor, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------
// the synthetic source
// ---------------------------------------------------------------------------------------

/// A deterministic, name-seeded PRNG. Seeding by tensor NAME rather than by draw order
/// means adding a tensor to the toy model does not shift every other tensor's values, so a
/// defect-matrix result recorded today still means the same thing tomorrow.
pub struct NamedRng(u64);

impl NamedRng {
    pub fn new(name: &str) -> Self {
        // FNV-1a over the name, then a splitmix step so short names differ in high bits.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in name.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        Self(h | 1)
    }
    fn next_u64(&mut self) -> u64 {
        // splitmix64. Chosen over xorshift64* only because `src/artifact/format.rs` already
        // has that one and the duplication gate is not budgeted; statistically either is
        // more than adequate for synthetic weights.
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// `n` values in `[-scale, scale)` from the named seed. Reproducible from the name alone.
pub fn draw(name: &str, n: usize, scale: f32) -> Vec<f32> {
    let mut r = NamedRng::new(name);
    (0..n).map(|_| r.unit() * scale).collect()
}

/// [`draw`], rounded to what bf16 can hold exactly.
///
/// The round-trip is the load-bearing part: a stimulus the checkpoint's dtype cannot
/// represent gets rounded by the engine and not by the oracle, and that difference reads as
/// a defect at every golden downstream of it.
pub fn fixed_bf16(name: &str, n: usize, scale: f32) -> Vec<f32> {
    draw(name, n, scale)
        .into_iter()
        .map(|x| bf16_decode(bf16_encode(x)))
        .collect()
}
