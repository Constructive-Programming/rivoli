//! Qwen3.8-Flash-Next's 109-family census, and the two things a converter does with it:
//! classify every tensor the checkpoint ships, and refuse one it does not know.
//!
//! **The census is the exclusion list.** The track rule is that excluded tensors are excluded
//! BY NAME, never by prefix glob and never by "everything else" — because a census that sums to
//! the right grand total while one class silently vanished is the failure mode. So MTP's 35
//! families and vision's 21 are named rows here, and the closure is stated the other way round
//! as well: **every tensor in the source index is converted, passed through, or named-excluded,
//! and a name that matches none of the 109 patterns is a hard error.** That is what makes an
//! upstream revision which ADDS a family loud instead of silently dropped, which the vendored
//! TSV's own header calls the one thing it cannot see on its own.
//!
//! **One copy of each number.** Counts, shapes, dtypes and byte totals live in
//! `docs/measurement/qwen-reference/tensor-families.tsv` — derived from all 131 shard headers at
//! the pinned revision, hash-gated by `crates/artifact/tests/qwen_names.rs` — and are read from
//! there rather than retyped. What this module adds is what a TSV cannot hold: each family's ROLE
//! in the artifact, which layer indices each pattern may legally carry (from the config, not the
//! census), and the two derived checks that catch what a count cannot — the `weight_scale_inv`
//! grid ORIENTATION and the n-gram hash parameters.

pub mod ngram_hash;

/// The n-gram hash derivation, re-exported at the path its callers already spell. A sibling module
/// under the 800-line cap and a split by JOB: everything above is "what tensors this checkpoint
/// ships"; that file is "what the 51.2 G-element table's row space IS", which is integer
/// arithmetic over `u64` and shares nothing with a TSV.
pub use ngram_hash::{NGRAM_SEED, NgramHash, ngram_hash};

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};

use super::Tsv;
use crate::format::Dtype;
use crate::quant::FP8_BLOCK;
use crate::qwen_config::QwenTextConfig;

/// The vendored census. 8 columns, `#`-prefixed header AND column-header rows.
pub const FAMILIES: Tsv = Tsv::new(
    "docs/measurement/qwen-reference/tensor-families.tsv",
    include_str!("../../../../docs/measurement/qwen-reference/tensor-families.tsv"),
);

/// The TSV's column contract, from its own COLUMNS block.
const COLS: usize = 8;
const COL_STATUS: usize = 0;
const COL_PATTERN: usize = 1;
const COL_EXAMPLE: usize = 2;
const COL_SHAPE: usize = 3;
const COL_DTYPE: usize = 4;
const COL_COUNT: usize = 5;
const COL_BYTES_PER: usize = 6;
const COL_TOTAL: usize = 7;

/// The three legal `v1-status` values, as the TSV's COLUMNS block spells them.
const STATUS_V1: &str = "v1";
const STATUS_MTP: &str = "EXCLUDED-v1-MTP";
const STATUS_VISION: &str = "EXCLUDED-v1-VISION";

/// What the converter does with a family — the classification the TSV cannot carry. Ordered by
/// the artifact layout, so [`Summary::line`] reads as the output directory's file list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// An fp8-e4m3 routed expert projection, block-128 scaled. Streamed per layer.
    RoutedWeight,
    /// Its `weight_scale_inv` grid — **BF16, not F32**, and per projection
    /// `[out/128, in/128]`, which is [20, 5] for `down_proj` and [5, 20] for `gate`/`up`. The
    /// orientation is checked by [`Census::confront_scale_grids`]; reading a [5, 20] grid as
    /// [20, 5] gives each tile a different tile's scale over 80.5 GB of expert weight, and no
    /// count or dtype check sees it.
    RoutedScale,
    /// One of the 128 ordered row ranges of the 51.2 G-element n-gram table, fp8-e4m3.
    NgramShard,
    /// The n-gram table's SINGLE per-tensor BF16 scale. `Fp8W`'s block path does not apply to
    /// the table; a per-tensor-scale dequant does.
    NgramScale,
    /// One of the three I64 hash-parameter buffers — head vocabularies, offsets, multipliers.
    /// Passed through, and byte-compared against [`ngram_hash`]'s re-derivation rather than
    /// trusted.
    HashParam,
    /// A BF16 tensor the artifact keeps resident and copies verbatim.
    Resident,
    /// The MTP draft layer — excluded by name for v1.
    ExcludedMtp,
    /// The vision tower and its merger — excluded by name for v1.
    ExcludedVision,
}

impl Role {
    /// Every role, in layout order — so a tally cannot omit one by forgetting it.
    pub const ALL: [Role; 8] = [
        Role::RoutedWeight,
        Role::RoutedScale,
        Role::NgramShard,
        Role::NgramScale,
        Role::HashParam,
        Role::Resident,
        Role::ExcludedMtp,
        Role::ExcludedVision,
    ];

    /// The label the census line prints. Kebab-case, so a log line greps as one token.
    pub const fn label(self) -> &'static str {
        match self {
            Role::RoutedWeight => "routed-weight",
            Role::RoutedScale => "routed-scale",
            Role::NgramShard => "ngram-shard",
            Role::NgramScale => "ngram-scale",
            Role::HashParam => "hash-param",
            Role::Resident => "resident",
            Role::ExcludedMtp => "excluded-mtp",
            Role::ExcludedVision => "excluded-vision",
        }
    }

    /// Is this a family the v1 artifact must NOT contain?
    pub const fn is_excluded(self) -> bool {
        matches!(self, Role::ExcludedMtp | Role::ExcludedVision)
    }
}

/// Which `{L}` values a pattern may carry, derived from the CONFIG rather than from the census.
///
/// Where trap T12 is caught: the PLE tensors sit under `layers.1.` because `ple_layer_ids` is
/// one-indexed, so a converter emitting `layers.2.ple.*` — or reading a `self_attn` tensor on a
/// GatedDeltaNet layer — has a name whose PATTERN is in the census and whose INDEX is not legal,
/// which a per-pattern count check cannot tell apart.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayerDomain {
    /// Every layer, `0..num_hidden_layers` — MoE and hyper-connection families.
    All,
    /// The 36 `linear_attention` layers.
    LinearAttention,
    /// The 12 `full_attention`-spelled, indexer-running layers.
    SparseAttention,
    /// The single PLE host, `ple_layer_ids[0] - 1`.
    PleHost,
    /// The MTP draft layers, `0..mtp_num_hidden_layers`.
    Mtp,
}

/// One family row of the vendored census, plus its role and its legal layer domain.
pub struct Family {
    pub status: String,
    /// The name with indices collapsed: `{L}` layer, `{E}` expert, `{B}` vision block,
    /// `{S}` n-gram shard.
    pub pattern: String,
    /// A real name from the shipped index, so a refusal can quote one.
    pub example: String,
    pub shape: Vec<u64>,
    pub dtype: Dtype,
    pub count: u64,
    pub bytes_per_tensor: u64,
    pub total_bytes: u64,
    pub role: Role,
}

impl Family {
    /// Parameters in this family: the shape's product times the tensor count.
    pub fn params(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.count
    }

    fn layer_domain(&self) -> Option<LayerDomain> {
        if !self.pattern.contains(".layers.{L}.") {
            return None;
        }
        Some(if self.pattern.starts_with("mtp.") {
            LayerDomain::Mtp
        } else if self.pattern.contains(".linear_attn.") {
            LayerDomain::LinearAttention
        } else if self.pattern.contains(".self_attn.") {
            LayerDomain::SparseAttention
        } else if self.pattern.contains(".ple.") {
            LayerDomain::PleHost
        } else {
            LayerDomain::All
        })
    }
}

/// The three dtypes this checkpoint declares, and only those.
///
/// **Not the inverse of `format::Dtype::name`**, which is private to that module — and the
/// narrow map is also the tighter check: a census row declaring `F8_E8M0` or `U8` names a family
/// this checkpoint does not have, and the refusal says so rather than silently admitting a
/// format the qwen converter has no path for. FNUZ never reaches here at all: `Dtype::narrow`
/// refuses `F8_E4M3FNUZ`/`F8_E5M2FNUZ` at the safetensors boundary, which matters on AMD, where
/// those are the native encodings and the bytes do not say which one they are.
fn dtype_of(census: &str) -> Result<Dtype> {
    Ok(match census {
        "F8_E4M3" => Dtype::F8E4M3,
        "BF16" => Dtype::Bf16,
        "I64" => Dtype::I64,
        other => bail!(
            "{}: dtype {other:?} is not one of the three this checkpoint declares (F8_E4M3, \
             BF16, I64)",
            FAMILIES.path()
        ),
    })
}

/// The role a v1 pattern plays, from the pattern alone.
///
/// Suffix and segment tests over the census's OWN patterns, not over arbitrary names: every string
/// tested here is one of 53 rows in a hash-gated file, so this is a lookup table written as
/// predicates rather than a heuristic. `Resident` is the fallthrough for v1 ONLY — the excluded
/// arms are decided by `status` first, so "everything else" never reaches an exclusion.
fn v1_role(pattern: &str) -> Role {
    let expert = pattern.contains(".mlp.experts.{E}.");
    match () {
        () if expert && pattern.ends_with("_proj.weight") => Role::RoutedWeight,
        () if expert && pattern.ends_with("_proj.weight_scale_inv") => Role::RoutedScale,
        () if pattern.ends_with(".ngram_embedding.shard_{S}.weight") => Role::NgramShard,
        () if pattern.ends_with(".ngram_embedding.weight_scale") => Role::NgramScale,
        () if pattern.contains(".ple_embedding.ngram_heads_")
            || pattern.ends_with(".ple_embedding.layer_multipliers") =>
        {
            Role::HashParam
        }
        () => Role::Resident,
    }
}

/// The dtype and rank each role must have, so a census row cannot claim a role its shape
/// contradicts. Byte counts are gated by `qwen_names.rs`; what is gated here is ROLE-to-layout,
/// which the converter's arithmetic rests on — a `RoutedScale` with a 1-D shape would be a
/// per-tensor scale wearing a grid's name, the exact distinction the n-gram table exists to make.
fn check_role_layout(f: &Family) -> Result<()> {
    let (want_dtype, want_rank): (Dtype, &[usize]) = match f.role {
        Role::RoutedWeight | Role::NgramShard => (Dtype::F8E4M3, &[2]),
        Role::RoutedScale => (Dtype::Bf16, &[2]),
        Role::NgramScale => (Dtype::Bf16, &[1]),
        Role::HashParam => (Dtype::I64, &[1]),
        // Resident carries norms [n], projections [o, i] and the two conv1d [c, 1, k].
        Role::Resident => (Dtype::Bf16, &[1, 2, 3]),
        // The excluded arms are not this port's arithmetic, and vision ships biases and a
        // 5-D patch embedding; nothing here reads their layout, so nothing here constrains it.
        Role::ExcludedMtp | Role::ExcludedVision => return Ok(()),
    };
    ensure!(
        f.dtype == want_dtype && want_rank.contains(&f.shape.len()),
        "{}: role {} wants {want_dtype:?} at rank {want_rank:?}, but the census row is {:?} \
         {:?}",
        f.pattern,
        f.role.label(),
        f.dtype,
        f.shape
    );
    Ok(())
}

/// A name with its indices collapsed to the census's placeholders, and the indices themselves.
///
/// The direction matters: the converter collapses each of the source index's 152,089 NAMES and
/// looks the pattern up, rather than expanding names from patterns and comparing sets. Same
/// closure, one hash lookup per tensor, and — the reason it is written this way — the INDICES fall
/// out of the same pass, so they can be bounds-checked against the config.
pub fn collapse(name: &str) -> (String, Vec<(char, usize)>) {
    let mut pattern = String::with_capacity(name.len());
    let mut indices = Vec::new();
    let mut previous = "";
    for (n, segment) in name.split('.').enumerate() {
        if n > 0 {
            pattern.push('.');
        }
        let placeholder = match previous {
            "layers" => Some('L'),
            "experts" => Some('E'),
            "blocks" => Some('B'),
            _ => None,
        };
        match (placeholder, segment.parse::<usize>()) {
            (Some(kind), Ok(idx)) => {
                pattern.push('{');
                pattern.push(kind);
                pattern.push('}');
                indices.push((kind, idx));
            }
            _ => match segment.strip_prefix("shard_").map(str::parse::<usize>) {
                Some(Ok(idx)) => {
                    pattern.push_str("shard_{S}");
                    indices.push(('S', idx));
                }
                _ => pattern.push_str(segment),
            },
        }
        previous = segment;
    }
    (pattern, indices)
}

/// Per-role tallies over one confrontation — the census line a converter prints.
pub struct Summary {
    per_role: Vec<(Role, u64, u64)>,
}

impl Summary {
    /// `(tensors, bytes)` for one role.
    pub fn of(&self, role: Role) -> (u64, u64) {
        self.per_role
            .iter()
            .find(|(r, _, _)| *r == role)
            .map_or((0, 0), |&(_, t, b)| (t, b))
    }

    /// `(tensors, bytes)` over every role that is NOT excluded — what the artifact holds.
    pub fn v1(&self) -> (u64, u64) {
        self.fold(|r| !r.is_excluded())
    }

    /// `(tensors, bytes)` over every role — the source checkpoint's own totals.
    pub fn total(&self) -> (u64, u64) {
        self.fold(|_| true)
    }

    fn fold(&self, keep: impl Fn(Role) -> bool) -> (u64, u64) {
        self.per_role
            .iter()
            .filter(|(r, _, _)| keep(*r))
            .fold((0, 0), |(t, b), &(_, rt, rb)| (t + rt, b + rb))
    }

    /// **Every arm named, with byte totals** — the one line a conversion is cited by.
    ///
    /// A grand total alone is what the track rule refuses: it balances while a class vanishes.
    /// Each role is printed even at zero, because a zero is the interesting reading.
    pub fn line(&self) -> String {
        let arms: Vec<String> = self
            .per_role
            .iter()
            .map(|(r, t, b)| format!("{} {t}/{b}", r.label()))
            .collect();
        let ((vt, vb), (tt, tb)) = (self.v1(), self.total());
        format!(
            "census {tt} tensors / {tb} B, of which v1 {vt}/{vb}: {}",
            arms.join(", ")
        )
    }
}

/// The parsed census, indexed by pattern.
pub struct Census {
    families: Vec<Family>,
    by_pattern: BTreeMap<String, usize>,
}

impl Census {
    /// Parse the vendored TSV, assign every row a role, and check each row's role against its
    /// own layout. Cheap enough to call per process: 109 rows.
    pub fn load() -> Result<Self> {
        let mut families = Vec::new();
        let mut by_pattern = BTreeMap::new();
        for row in FAMILIES.rows(COLS)? {
            let status = row.str_at(COL_STATUS)?.to_string();
            let pattern = row.str_at(COL_PATTERN)?.to_string();
            let role = match status.as_str() {
                STATUS_V1 => v1_role(&pattern),
                STATUS_MTP => Role::ExcludedMtp,
                STATUS_VISION => Role::ExcludedVision,
                other => bail!(
                    "{}: v1-status {other:?} is not one of {STATUS_V1:?} / {STATUS_MTP:?} / \
                     {STATUS_VISION:?}. If this is the COLUMN-HEADER row then its `#` prefix \
                     was dropped, and every tally below is off by one row",
                    FAMILIES.path()
                ),
            };
            let f = Family {
                status,
                example: row.str_at(COL_EXAMPLE)?.to_string(),
                shape: row.dims_at(COL_SHAPE)?,
                dtype: dtype_of(row.str_at(COL_DTYPE)?)?,
                count: row.u64_at(COL_COUNT, "count-of-tensors-in-family")?,
                bytes_per_tensor: row.u64_at(COL_BYTES_PER, "bytes-per-tensor")?,
                total_bytes: row.u64_at(COL_TOTAL, "total-bytes")?,
                role,
                pattern,
            };
            check_role_layout(&f)?;
            // Patterns are the converter's lookup keys, so a duplicate would leave one of two
            // rows unreachable while every sum still balanced.
            ensure!(
                by_pattern
                    .insert(f.pattern.clone(), families.len())
                    .is_none(),
                "{}: duplicate name-pattern {}",
                FAMILIES.path(),
                f.pattern
            );
            families.push(f);
        }
        ensure!(
            !families.is_empty(),
            "{}: no data rows parsed — the file or the comment filter moved",
            FAMILIES.path()
        );
        Ok(Census {
            families,
            by_pattern,
        })
    }

    pub fn families(&self) -> &[Family] {
        &self.families
    }

    /// `(tensors, bytes)` the census records for one role — the arm totals a converter prints
    /// and a gate scores. Here because jscpd said so: `qwen_names.rs` and
    /// `crates/cli/tests/qwen_convert.rs` each grew the same `filter(role).map(bytes).sum()` run.
    /// Unlike [`Summary::of`] it reads the CENSUS rather than one confrontation's tally — what the
    /// pinned revision holds, against what a run observed.
    pub fn role_totals(&self, role: Role) -> (u64, u64) {
        self.families
            .iter()
            .filter(|f| f.role == role)
            .fold((0, 0), |(t, b), f| (t + f.count, b + f.total_bytes))
    }

    /// The family one tensor NAME belongs to, or a refusal naming the tensor.
    ///
    /// This is the "a new family is a hard error" edge. The refusal quotes both the name and the
    /// pattern it collapsed to, because the actionable question is whether the checkpoint grew a
    /// family or whether [`collapse`] stopped recognising an index.
    pub fn family_of(&self, name: &str) -> Result<&Family> {
        let (pattern, _) = collapse(name);
        let i = self.by_pattern.get(&pattern).with_context(|| {
            format!(
                "{name} collapses to {pattern:?}, which is not one of the {} families in {}. \
                 Either this checkpoint is not the pinned revision — a family this census has \
                 never seen is the one thing it cannot detect on its own — or an index spelling \
                 changed and `collapse` no longer recognises it. Refusing rather than guessing \
                 a role",
                self.families.len(),
                FAMILIES.path()
            )
        })?;
        self.families
            .get(*i)
            .context("census index is out of step with its rows")
    }

    /// One tensor, confronted with its own census row: dtype and shape both.
    ///
    /// Called per tensor from the converter's shard walk, where the header carries the real
    /// dtype and shape — so this is where a re-quantized or re-shaped upstream tensor reddens.
    /// A shape check alone would not do: two tensors of the same LENGTH and different shapes are
    /// byte-indistinguishable, which is what `assert_verbatim`'s own helper had to learn.
    pub fn confront_tensor(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<&Family> {
        let f = self.family_of(name)?;
        // `usize::MAX` on an impossible conversion rather than 0: a 0 would compare equal to a
        // zero-length axis, and this comparison must fail loudly rather than pass narrowly.
        let want: Vec<usize> = f
            .shape
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
            .collect();
        ensure!(
            f.dtype == dtype && want == shape,
            "{name} is {dtype:?} {shape:?} where the census family {} is {:?} {want:?} — this \
             tensor is {} verbatim, so no later step compares the two",
            f.pattern,
            f.dtype,
            if f.role.is_excluded() {
                "excluded"
            } else {
                "carried"
            }
        );
        Ok(f)
    }

    /// **Every routed expert's `weight_scale_inv` grid is `[out/128, in/128]` — per projection.**
    ///
    /// `down_proj`'s weight is [2560, 640] so its grid is [20, 5]; `gate_proj` and `up_proj` are
    /// [640, 2560] so theirs are [5, 20]. Building the grid at one orientation for all three
    /// indexes two of them transposed, giving each 128x128 tile a different tile's scale across
    /// 80.5 GB — two thirds of the expert weight — with no crash, no length error and no dtype
    /// mismatch. The census's own rows are what this reads, so the check is that the vendored
    /// pair is self-consistent BEFORE 172.76 GiB moves.
    ///
    /// Derived from the weight row rather than tabulated: a table of expected grids would be a
    /// second copy of the thing under test.
    pub fn confront_scale_grids(&self) -> Result<()> {
        let block = u64::try_from(FP8_BLOCK).unwrap_or(u64::MAX);
        let mut checked = 0usize;
        for w in self
            .families
            .iter()
            .filter(|f| f.role == Role::RoutedWeight)
        {
            let pattern = w
                .pattern
                .strip_suffix(".weight")
                .map(|stem| format!("{stem}.weight_scale_inv"))
                .with_context(|| format!("{}: a routed weight not ending `.weight`", w.pattern))?;
            let s = self
                .by_pattern
                .get(&pattern)
                .and_then(|i| self.families.get(*i))
                .with_context(|| format!("no scale family {pattern} for {}", w.pattern))?;
            let want: Vec<u64> = w.shape.iter().map(|d| d.div_ceil(block)).collect();
            ensure!(
                s.shape == want,
                "{pattern} is {:?} but its weight {:?} implies a [out/{FP8_BLOCK}, \
                 in/{FP8_BLOCK}] grid of {want:?}. A transposed grid hands every tile a \
                 different tile's scale over {} B of expert weight, and no count, dtype or \
                 length check sees it",
                s.shape,
                w.shape,
                w.total_bytes
            );
            ensure!(
                s.count == w.count,
                "{pattern} has {} tensors against its weight's {} — every routed projection \
                 carries exactly one grid",
                s.count,
                w.count
            );
            checked += 1;
        }
        // The examined count, asserted rather than assumed: a role predicate that stopped
        // matching would leave this loop empty and the check vacuously green (P7).
        ensure!(
            checked == 3,
            "{checked} routed projections were confronted, not the 3 this checkpoint ships \
             (gate_proj, up_proj, down_proj) — the RoutedWeight role predicate stopped matching"
        );
        Ok(())
    }

    /// **The exhaustive confrontation: every name in the source index, against this census.**
    ///
    /// Three closures at once, which is what makes it exhaustive rather than a spot check:
    ///
    /// 1. every name collapses to one of the 109 patterns (a new family is a hard error);
    /// 2. every index it carries is legal for that pattern *per the config* — so
    ///    `layers.2.ple.*` is refused even though `layers.{L}.ple.*` is a real family;
    /// 3. every pattern is observed exactly `count` times (the both-ends check an exemption
    ///    list cannot make: a class that vanished shows up here, not in a grand total).
    ///
    /// Runs against the INDEX, before a shard is opened, so a checkpoint that is not the pinned
    /// revision costs milliseconds rather than 172.76 GiB of reads.
    pub fn confront_index(&self, cfg: &QwenTextConfig, names: &[&str]) -> Result<Summary> {
        let mut seen = vec![0u64; self.families.len()];
        for name in names {
            let (pattern, indices) = collapse(name);
            let i = *self.by_pattern.get(&pattern).with_context(|| {
                format!(
                    "{name} collapses to {pattern:?}, which is no family in {}",
                    FAMILIES.path()
                )
            })?;
            let f = self
                .families
                .get(i)
                .context("census index is out of step with its rows")?;
            self.check_indices(cfg, name, f, &indices)?;
            *seen.get_mut(i).context("tally is out of step")? += 1;
        }
        for (f, &got) in self.families.iter().zip(&seen) {
            ensure!(
                got == f.count,
                "{} appears {got} times in the source index, but the census records {} — a \
                 family that partly vanished leaves every grand total balanced",
                f.pattern,
                f.count
            );
        }
        self.confront_scale_grids()?;
        Ok(self.summarise(&seen))
    }

    /// Every index a name carries, bounded by the config — never by the census's own count,
    /// except for vision blocks, whose depth this port does not bind (see `qwen_config`'s
    /// header: `vision_config` is deliberately unbound, so its 27 is read from the census row
    /// that will be excluded anyway).
    fn check_indices(
        &self,
        cfg: &QwenTextConfig,
        name: &str,
        f: &Family,
        indices: &[(char, usize)],
    ) -> Result<()> {
        for &(kind, idx) in indices {
            let (bound, what) = match kind {
                'L' => (self.layer_bound(cfg, f, idx)?, "layer"),
                'E' => (cfg.num_experts, "expert"),
                'S' => (cfg.split_ngram_parts, "n-gram shard"),
                'B' => (usize::try_from(f.count).unwrap_or(0), "vision block"),
                other => bail!("{name}: unknown index placeholder {other:?}"),
            };
            ensure!(
                idx < bound,
                "{name}: {what} index {idx} is not below {bound} for family {}",
                f.pattern
            );
        }
        Ok(())
    }

    /// The layer bound for one observed `{L}`, and — the sharp half — the FAMILY check, which is
    /// not a bound at all.
    ///
    /// A `self_attn` tensor on a GatedDeltaNet layer, or a `ple` tensor on any layer but the
    /// host, is a name whose pattern and count are both fine. Only the config knows it is wrong.
    fn layer_bound(&self, cfg: &QwenTextConfig, f: &Family, idx: usize) -> Result<usize> {
        let domain = f.layer_domain().with_context(|| {
            format!(
                "{}: an {{L}} index in a family with no layer domain",
                f.pattern
            )
        })?;
        let (ok, wanted) = match domain {
            LayerDomain::All => (idx < cfg.n_layers, "any layer"),
            LayerDomain::Mtp => (idx < cfg.mtp_num_hidden_layers, "an MTP draft layer"),
            LayerDomain::LinearAttention => (
                !cfg.layer_is_sparse_attention(idx)?,
                "a linear_attention (GatedDeltaNet) layer",
            ),
            LayerDomain::SparseAttention => (
                cfg.layer_is_sparse_attention(idx)?,
                "a full_attention-spelled (indexer-running) layer",
            ),
            LayerDomain::PleHost => (
                cfg.layer_hosts_ple(idx),
                "the ONE-INDEXED ple_layer_ids host, i.e. layer_idx = ple_layer_ids[0] - 1",
            ),
        };
        ensure!(
            ok,
            "{} at layer {idx}, which is not {wanted}. `layer_types` calls it {:?}; the PLE host \
             is layer_idx {:?}. This is trap T12's shape: the pattern is a real family and the \
             count would still balance",
            f.pattern,
            cfg.layer_types.get(idx),
            cfg.ple_host_layer().ok()
        );
        Ok(if domain == LayerDomain::Mtp {
            cfg.mtp_num_hidden_layers
        } else {
            cfg.n_layers
        })
    }

    fn summarise(&self, seen: &[u64]) -> Summary {
        let per_role = Role::ALL
            .iter()
            .map(|&role| {
                let (t, b) = self
                    .families
                    .iter()
                    .zip(seen)
                    .filter(|(f, _)| f.role == role)
                    .fold((0, 0), |(t, b), (f, &got)| {
                        (t + got, b + got * f.bytes_per_tensor)
                    });
                (role, t, b)
            })
            .collect();
        Summary { per_role }
    }
}
