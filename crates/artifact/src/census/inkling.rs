//! Inkling 2 Mini's 85-family census, and the two things the converter does with it: classify
//! every tensor the single shard ships, and refuse one it does not know.
//!
//! **The census is the exclusion list.** As in [`super::qwen`], excluded tensors are excluded BY
//! NAME, never by prefix glob and never by "everything else": the 26 `EXCLUDED-VISION` rows name
//! every `model.visual.*` family, and the closure is stated both ways — every one of the shard's
//! 1244 tensors is converted or named-excluded, and a name matching none of the 85 patterns is a
//! hard error. `mtp.safetensors` is a SEPARATE file the text converter never opens (refused by
//! `Flag::Mtp`), so it is not in this census at all.
//!
//! **The family key is (pattern, dtype, shape), not the pattern alone.** Unlike qwen's, this
//! census splits several patterns into two rows by storage format: `attn_res.attn_queries.{I}` is
//! `2304` F32 on the 19 sliding layers and `3x2304` F32 on the 4 full ones; `mlp.experts.w13_weight`
//! is U8 `[128,2304,1152]` (NVFP4) on layers 0–19 and F8_E4M3 `[128,2304,2304]` (MXFP8) on 20–23.
//! A pattern-only lookup would collide those rows, so the dtype and shape — both read from the
//! shard header, never trusted from the name — are part of the key.
//!
//! **One copy of each number.** Counts, shapes, dtypes and byte totals live in
//! `docs/measurement/inkling2-reference/mini/tensor-families.tsv`, derived from the live shard
//! header and hash-gated by `crates/artifact/tests/inkling_pins.rs`. This module adds what a TSV
//! cannot hold: the name→pattern collapse and the exhaustive per-family confrontation.

use super::{Row, Tsv};
use crate::format::Dtype;
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;

/// The vendored census. 8 columns, `#`-prefixed header, no column-header row.
pub const FAMILIES: Tsv = Tsv::new(
    "docs/measurement/inkling2-reference/mini/tensor-families.tsv",
    include_str!("../../../../docs/measurement/inkling2-reference/mini/tensor-families.tsv"),
);

use super::{
    COL_BYTES_PER, COL_COUNT, COL_DTYPE, COL_PATTERN, COL_SHAPE, COL_STATUS, COL_TOTAL, COLS,
};

/// The two legal `status` values, as the TSV's COLUMNS block spells them.
const STATUS_TEXT: &str = "TEXT";
const STATUS_VISION: &str = "EXCLUDED-VISION";

/// What the converter does with a family: consume it into the text artifact, or exclude it by
/// name. The vision tower is pinned and counted but never spent (text-only v1, owner scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Text,
    ExcludedVision,
}

impl Status {
    /// The exclusion test the converter's `in_scope` reads.
    pub fn is_excluded(self) -> bool {
        matches!(self, Status::ExcludedVision)
    }
}

fn status_of(s: &str) -> Result<Status> {
    match s {
        STATUS_TEXT => Ok(Status::Text),
        STATUS_VISION => Ok(Status::ExcludedVision),
        other => {
            anyhow::bail!("unknown census status {other:?} — not {STATUS_TEXT}/{STATUS_VISION}")
        }
    }
}

/// The four dtypes the Mini shard declares, and only those. A census row naming any other dtype
/// is a family this checkpoint does not have, refused rather than admitted.
///
/// `pub` so `tests/inkling_census.rs` reads the census dtype column through the SAME map the
/// loader uses rather than a duplicate `match` that could drift from it.
pub fn dtype_of(s: &str) -> Result<Dtype> {
    match s {
        "F32" => Ok(Dtype::F32),
        "BF16" => Ok(Dtype::Bf16),
        "F8_E4M3" => Ok(Dtype::F8E4M3),
        "U8" => Ok(Dtype::U8),
        other => anyhow::bail!("census dtype {other:?} is not one of F32/BF16/F8_E4M3/U8"),
    }
}

/// The inverse, for the family key. `Dtype::name` is private to `format`, and the narrow map is
/// the tighter check anyway: a tensor whose dtype is not one of the four cannot match a row.
fn dtype_name(d: Dtype) -> Result<&'static str> {
    match d {
        Dtype::F32 => Ok("F32"),
        Dtype::Bf16 => Ok("BF16"),
        Dtype::F8E4M3 => Ok("F8_E4M3"),
        Dtype::U8 => Ok("U8"),
        other => anyhow::bail!("tensor dtype {other:?} is not one the Mini census declares"),
    }
}

/// One family row of the vendored census.
pub struct Family {
    pub pattern: String,
    pub status: Status,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
    pub count: u64,
    pub bytes_per: u64,
    pub total: u64,
}

/// A name with its indices collapsed to the census's placeholders.
///
/// `layers.<N>` → `layers.{L}` (text and vision both), and the two query-bank indices
/// `attn_res.attn_queries.<N>` / `attn_res.mlp_queries.<N>` → `{I}`. No other segment carries an
/// index in this census, so a numeric segment after any other keyword is left alone — and would
/// then fail to match a pattern, which is the loud outcome for an unexpected name.
pub fn collapse(name: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut prev = "";
    for seg in name.split('.') {
        let placeholder = if is_numeric(seg) && prev == "layers" {
            "{L}"
        } else if is_numeric(seg) && is_query_bank(prev) {
            "{I}"
        } else {
            seg
        };
        out.push(placeholder);
        prev = seg;
    }
    out.join(".")
}

/// A census index segment: non-empty and all ASCII digits.
fn is_numeric(seg: &str) -> bool {
    !seg.is_empty() && seg.bytes().all(|b| b.is_ascii_digit())
}

/// The two query-bank keywords whose numeric index collapses to `{I}`.
fn is_query_bank(prev: &str) -> bool {
    matches!(prev, "attn_queries" | "mlp_queries")
}

/// Per-status tallies over one confrontation — the census line the converter prints.
pub struct Summary {
    pub text_tensors: u64,
    pub text_bytes: u64,
    pub vision_tensors: u64,
    pub vision_bytes: u64,
}

impl Summary {
    pub fn total_tensors(&self) -> u64 {
        self.text_tensors + self.vision_tensors
    }

    /// One readable line, printed before the first byte is converted.
    pub fn line(&self) -> String {
        format!(
            "census: {} TEXT tensors ({} B) consumed, {} EXCLUDED-VISION tensors ({} B) by name",
            self.text_tensors, self.text_bytes, self.vision_tensors, self.vision_bytes
        )
    }
}

/// The parsed census, indexed by (pattern, dtype, shape).
pub struct Census {
    families: Vec<Family>,
    by_key: HashMap<(String, &'static str, Vec<usize>), usize>,
}

impl Census {
    /// Parse the vendored TSV, asserting each row's `count * bytes_per == total` and that no two
    /// rows share a (pattern, dtype, shape) key.
    pub fn load() -> Result<Self> {
        let rows = FAMILIES.rows(COLS)?;
        let mut families: Vec<Family> = Vec::with_capacity(rows.len());
        let mut by_key = HashMap::new();
        for r in &rows {
            let f = family_from_row(r)?;
            let key = (f.pattern.clone(), dtype_name(f.dtype)?, f.shape.clone());
            ensure!(
                by_key.insert(key, families.len()).is_none(),
                "family key {:?} is not unique (pattern, dtype, shape) in {}",
                f.pattern,
                FAMILIES.path()
            );
            families.push(f);
        }
        Ok(Self { families, by_key })
    }

    pub fn families(&self) -> &[Family] {
        self.families.as_slice()
    }

    /// The family a tensor belongs to, from its name, dtype and shape — all three read from the
    /// shard header. A name matching no row is a hard error: an upstream revision that ADDS a
    /// family becomes loud instead of silently dropped.
    pub fn family_of(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&Family> {
        let pattern = collapse(name);
        let dn = dtype_name(dtype)?;
        let key = (pattern.clone(), dn, shape.to_vec());
        self.by_key
            .get(&key)
            .map(|&i| &self.families[i])
            .with_context(|| {
                format!(
                    "tensor {name:?} (pattern {pattern:?}, {dn}, shape {shape:?}) matches no \
                     census family — the checkpoint is not the pinned revision, or adds a family"
                )
            })
    }

    /// The exhaustive closure over the shard header: classify every tensor, confront each
    /// family's observed count against the census, and check each tensor's byte length against
    /// its row's `bytes_per`. `tensors` is `(name, dtype, shape, byte_len)` for every tensor the
    /// shard holds. Both ends are checked — a family the shard lacks reddens here, and a tensor
    /// the census lacks reddens in [`Self::family_of`].
    pub fn confront(&self, tensors: &[(&str, Dtype, &[usize], u64)]) -> Result<Summary> {
        let mut observed = vec![0u64; self.families.len()];
        let mut sum = Summary {
            text_tensors: 0,
            text_bytes: 0,
            vision_tensors: 0,
            vision_bytes: 0,
        };
        for &(name, dtype, shape, blen) in tensors {
            let pattern = collapse(name);
            let dn = dtype_name(dtype)?;
            let key = (pattern, dn, shape.to_vec());
            let &i = self.by_key.get(&key).with_context(|| {
                format!("tensor {name:?} (shape {shape:?}, {dn}) matches no census family")
            })?;
            let f = &self.families[i];
            ensure!(
                f.bytes_per == blen,
                "{name}: {blen} bytes where the census records {} per tensor — a shape or dtype \
                 drift the count alone would not catch",
                f.bytes_per
            );
            observed[i] += 1;
            if f.status.is_excluded() {
                sum.vision_tensors += 1;
                sum.vision_bytes += blen;
            } else {
                sum.text_tensors += 1;
                sum.text_bytes += blen;
            }
        }
        for (i, f) in self.families.iter().enumerate() {
            ensure!(
                observed[i] == f.count,
                "family {:?} ({:?}): observed {} tensors, census records {} — every name \
                 classified, so this is a count disagreement rather than a missing family",
                f.pattern,
                f.dtype,
                observed[i],
                f.count
            );
        }
        Ok(sum)
    }
}

/// Parse one census row into a [`Family`], asserting its byte closure.
fn family_from_row(r: &Row<'_>) -> Result<Family> {
    let status = status_of(r.str_at(COL_STATUS)?)?;
    let pattern = r.str_at(COL_PATTERN)?.to_string();
    let shape: Vec<usize> = r
        .dims_at(COL_SHAPE)?
        .into_iter()
        .map(|d| usize::try_from(d).context("census dimension exceeds usize"))
        .collect::<Result<_>>()?;
    let dtype = dtype_of(r.str_at(COL_DTYPE)?)?;
    let count = r.u64_at(COL_COUNT, "count")?;
    let bytes_per = r.u64_at(COL_BYTES_PER, "bytes-per-tensor")?;
    let total = r.u64_at(COL_TOTAL, "total-bytes")?;
    ensure!(
        count * bytes_per == total,
        "{}: family {pattern:?} count {count} x {bytes_per} B != total {total}",
        FAMILIES.path()
    );
    Ok(Family {
        pattern,
        status,
        shape,
        dtype,
        count,
        bytes_per,
        total,
    })
}
