//! Qwen3.8-Flash-Next's vendored S0 evidence, recomputed — the census gate.
//!
//! **Why this file exists: census item 1's own closing condition says "the gate RECOMPUTES
//! FNV-1a from the live file and byte-compares", and until this landed nothing did.**
//! `docs/measurement/qwen-reference/tensor-families.tsv` closed its census in prose
//! (`EQUAL: True`, twice) and recorded FNV-1a pins its author had computed by hand. A pin
//! checked only against a frozen copy of itself is decoration — the scar is on record
//! (`artifacts-carry-their-own-provenance`) — and the review that found this one also found
//! that the S0 hand-off report had already retyped `config.json`'s fnv1a WRONG
//! (`84a8e18f0f2b02b5` for `6e35d0820f1ec218`) in the same message that said the values were
//! deliberately not being retyped. The tree was right and the prose was wrong, which is
//! exactly the asymmetry a recompute closes and a second frozen copy does not.
//!
//! Ported in shape from `k3_names.rs`, its direct precedent: `include_str!` a vendored TSV
//! reduction plus the vendored `config.json`, read every expected value out of the TSV's OWN
//! header so there is one copy of each number rather than two agreeing frozen ones, and panic
//! rather than default when a header phrase moves — a header reword must be a visible edit,
//! not a check that quietly stops checking.
//!
//! **What is different here, and why the two parsers are not yet one function.** K3's TSV is
//! 4 columns, count first, no header row; qwen's is 8 columns with a `#`-prefixed header row
//! and count in column 6. So `k3_names.rs::families` cannot read this file and
//! [`families`] cannot read K3's. The `#` prefix on the column-header row exists so K3's
//! `filter(|l| !l.starts_with('#'))` idiom at least *survives* contact with this file instead
//! of panicking inside `"count-of-tensors-in-family".parse()`; lifting one shared helper is
//! owed at S4, when the converter census gives the second reader a reason to exist. Naming
//! the debt beats forking a third parser to hide it.
//!
//! No GPU, no network: 22 KB of vendored TSV, 71 KB of vendored JSON, 256 B of vendored
//! weight payload.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_core::hash::fnv1a;

const DIR: &str = "docs/measurement/qwen-reference";
const FAMILIES: &str = include_str!("../../../docs/measurement/qwen-reference/tensor-families.tsv");
// Bytes, not `&str`, for all three: the pins below are over the FILE, and a `&str` would
// normalise nothing today but makes "these are the bytes that hash" a claim about UTF-8
// rather than about the file. `config.json` is also parsed as JSON below, from this same
// constant, so the parse and the pin cannot be reading different content.
const CONFIG: &[u8] = include_bytes!("../../../docs/measurement/qwen-reference/config.json");
const GENERATION: &[u8] =
    include_bytes!("../../../docs/measurement/qwen-reference/generation_config.json");
const NORM_L0: &[u8] =
    include_bytes!("../../../docs/measurement/qwen-reference/linear_attn_norm_l0.bin");

/// The three legal `v1-status` values, as the TSV's own COLUMNS block spells them.
const STATUSES: [&str; 3] = ["v1", "EXCLUDED-v1-MTP", "EXCLUDED-v1-VISION"];

/// One data row of the vendored census.
struct Family {
    status: String,
    pattern: String,
    shape: Vec<u64>,
    dtype: String,
    count: u64,
    bytes_per_tensor: u64,
    total_bytes: u64,
}

impl Family {
    /// Parameters in this family: the shape's product, times the number of tensors.
    fn params(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.count
    }
}

/// One data row, parsed. Dimensions are `x`-separated.
///
/// Every field is parsed, including ones no assertion below reads: a row whose
/// `bytes-per-tensor` is not an integer is a corrupt row whether or not this run looks at
/// that column, and the parse is the cheapest place to say so. `Result`-free because a
/// malformed census is a broken fixture and every caller is a test.
///
/// **A named function rather than a closure inside [`families`], and jscpd is why.** Ported
/// as a closure it reproduced `k3_names.rs::families`'s first six lines token for token (89
/// tokens, reported on this file's first build), which is the duplication gate making the
/// same argument this module's header makes in prose: the two censuses want one parser. The
/// lift is S4's — it needs the converter's reader to exist before there is a third caller to
/// design for — so the shape here changed instead of the gate.
fn parse_row(line: &str) -> Family {
    let f: Vec<&str> = line.split('\t').collect();
    assert_eq!(f.len(), 8, "malformed row: {line:?}");
    assert!(
        STATUSES.contains(&f[0]),
        "field 0 is {:?}, not one of {STATUSES:?} — if this is the COLUMN-HEADER row then its \
         `#` prefix was dropped, and every sum below is off by one row",
        f[0]
    );
    Family {
        status: f[0].to_string(),
        pattern: f[1].to_string(),
        shape: f[3].split('x').map(|d| d.parse().expect("dim")).collect(),
        dtype: f[4].to_string(),
        count: f[5].parse().expect("count"),
        bytes_per_tensor: f[6].parse().expect("bytes-per-tensor"),
        total_bytes: f[7].parse().expect("total-bytes"),
    }
}

/// The vendored census's data rows: everything that is neither a `#` comment nor blank.
fn families() -> Vec<Family> {
    let is_data = |l: &&str| !(l.starts_with('#') || l.trim().is_empty());
    FAMILIES.lines().filter(is_data).map(parse_row).collect()
}

/// The whitespace token after `keyword`, on the one header line that starts `# <anchor>`.
///
/// **One accessor for every number this test expects**, because the alternative is a literal
/// in the test AND prose in the header — two frozen copies agreeing with each other, where a
/// typo in either is invisible and the header is decoration. Reading them out of the artifact
/// leaves one copy and makes the header load-bearing (`k3_names.rs::declared`'s argument,
/// which this generalises: that one reads the integer BEFORE its phrase, because K3's header
/// is written the other way round).
///
/// **Anchored at a line start rather than searched loose.** Several of these labels are
/// substrings of each other — `family rows` occurs inside `sum of \`count\` over all family
/// rows`, and `v1` inside `v1-status` — so a loose `find` would read a number off the wrong
/// line and still pass. Panics when a label moves, rather than defaulting: a reworded header
/// must be a visible edit here, not a check that silently stops checking.
fn header_token(anchor: &str, keyword: &str) -> &'static str {
    let at = FAMILIES.find(&format!("\n# {anchor}")).unwrap_or_else(|| {
        panic!(
            "no line in {DIR}/tensor-families.tsv's header starts `# {anchor}`; this test reads \
             its expected values from there"
        )
    });
    let line = FAMILIES[at + 1..].lines().next().expect("a line follows");
    let after = line
        .find(keyword)
        .unwrap_or_else(|| panic!("`# {anchor}`'s line carries no `{keyword}`: {line:?}"))
        + keyword.len();
    line[after..]
        .split_whitespace()
        .next()
        .unwrap_or_else(|| panic!("`# {anchor}`'s `{keyword}` has no value: {line:?}"))
}

/// A count the header declares as `# <label>   <number>`.
fn declared(label: &str) -> u64 {
    header_token(label, label)
        .parse()
        .unwrap_or_else(|e| panic!("`# {label}` is not followed by a number: {e}"))
}

/// A `params`/`bytes` total the header declares for one `v1-status`.
fn declared_for(status: &str, keyword: &str) -> u64 {
    header_token(&format!("STATUS {status} "), &format!("{keyword} "))
        .parse()
        .unwrap_or_else(|e| panic!("STATUS {status}'s {keyword} is not a number: {e}"))
}

/// **The census closes, recomputed from the rows rather than asserted in prose.**
///
/// This is the assertion `EQUAL: True` was standing in for. Both totals come out of the
/// header, so the rows and the provenance lines have to agree; what neither this nor the
/// header can see is an upstream revision that ADDS a tensor family, which is why the
/// revision pin two blocks above them is the other half of the evidence.
#[test]
fn the_vendored_census_closes_against_its_own_header() {
    let fams = families();
    assert_eq!(
        fams.len() as u64,
        declared("family rows"),
        "the header declares its family-row count; the rows are {}",
        fams.len()
    );
    let tensors = declared("tensors in model.safetensors.index.json");
    assert_eq!(
        fams.iter().map(|f| f.count).sum::<u64>(),
        tensors,
        "family counts must sum to the index's tensor count — the `EQUAL: True` this replaces"
    );
    assert_eq!(
        declared("sum of `count` over all family rows"),
        tensors,
        "the header's own two tensor counts disagree, so one of them was retyped"
    );
    let bytes = declared("index metadata.total_size");
    assert_eq!(
        fams.iter().map(|f| f.total_bytes).sum::<u64>(),
        bytes,
        "family byte totals must sum to the index's `metadata.total_size`"
    );
    assert_eq!(
        declared("sum of `total-bytes` over family rows"),
        bytes,
        "the header's own two byte totals disagree"
    );
    // Patterns are the converter's lookup keys at S4, so a duplicate would leave one of two
    // rows unreachable while every sum above still balanced.
    let mut pats: Vec<&str> = fams.iter().map(|f| f.pattern.as_str()).collect();
    pats.sort_unstable();
    let before = pats.len();
    pats.dedup();
    assert_eq!(before, pats.len(), "duplicate name-pattern in the census");
}

/// **No row is ragged**: `bytes-per-tensor × count == total-bytes`, on every row, and the
/// per-tensor size is the shape's product times the dtype's width.
///
/// The header's claim is that "every pattern has exactly one (dtype, shape) row, so
/// bytes-per-tensor is exact, not an average". This is that claim as arithmetic. Without it
/// the totals above could balance while individual rows were averages — which a per-family
/// byte figure must not be, since S4's converter sizes allocations from it. The second
/// assertion is the one a transposed or truncated SHAPE fails: it passes the product test
/// and cannot pass this one.
#[test]
fn every_family_row_is_exact_rather_than_an_average() {
    let fams = families();
    assert!(
        !fams.is_empty(),
        "no rows parsed — the file or the comment filter moved"
    );
    for f in &fams {
        assert_eq!(
            f.bytes_per_tensor * f.count,
            f.total_bytes,
            "{}: bytes-per-tensor {} x count {} != total-bytes {}",
            f.pattern,
            f.bytes_per_tensor,
            f.count,
            f.total_bytes
        );
        let width: u64 = match f.dtype.as_str() {
            "F8_E4M3" => 1,
            "BF16" => 2,
            "I64" => 8,
            d => panic!("{}: unknown dtype {d}", f.pattern),
        };
        assert_eq!(
            f.shape.iter().product::<u64>() * width,
            f.bytes_per_tensor,
            "{}: shape {:?} at {width} B/element is not {} B",
            f.pattern,
            f.shape,
            f.bytes_per_tensor
        );
    }
}

/// **The parameter counts the plan of record cites, re-derived from the rows.**
///
/// `~180 GB on disk` was read as a byte figure for a month (it is the PARAMETER count:
/// 180,007,507,860 params occupy 185,502,232,570 B = 172.76 GiB), and `4B MTP` was
/// transcribed from a model card where the excluded rows sum to 2,607,304,448. Both
/// corrections now live in ONE place — this TSV's header — and this test recomputes them from
/// the rows, so the plan doc cites the file instead of carrying a fourth copy of a number
/// nobody re-derives.
#[test]
fn the_parameter_counts_split_by_v1_status_are_re_derivable() {
    let fams = families();
    for status in STATUSES {
        let rows: Vec<&Family> = fams.iter().filter(|f| f.status == status).collect();
        assert!(
            !rows.is_empty(),
            "no {status} rows — the status spelling moved and this cell went vacuous"
        );
        assert_eq!(
            rows.iter().map(|f| f.params()).sum::<u64>(),
            declared_for(status, "params"),
            "{status}: parameter count, summed over {} rows",
            rows.len()
        );
        assert_eq!(
            rows.iter().map(|f| f.total_bytes).sum::<u64>(),
            declared_for(status, "bytes"),
            "{status}: byte total, summed over {} rows",
            rows.len()
        );
    }
}

/// **The vendored bytes hash as the header records — recomputed from the live files.**
///
/// This is census item 1's closing condition, verbatim. Length is checked too: a truncation
/// changes both, and reporting only the hash makes a truncated fixture read as an edited one.
///
/// The sha256 beside each pin is deliberately NOT checked: adding a `sha2` dependency to
/// compare a vendored fixture is the trade this workspace already declined
/// (`core/src/hash.rs`'s header — "FNV, not a sha256 dep"), and the sha is there for a human
/// re-fetching the file with `curl` and `sha256sum`. FNV-1a over the live bytes is what this
/// tree can recompute, so it is what is gated.
#[test]
fn the_vendored_files_hash_as_the_header_records() {
    for (name, bytes) in [
        ("config.json", CONFIG),
        ("generation_config.json", GENERATION),
        ("linear_attn_norm_l0.bin", NORM_L0),
    ] {
        let want_len: usize = header_token(name, "bytes ")
            .parse()
            .expect("declared bytes");
        assert_eq!(bytes.len(), want_len, "{name}: length");
        assert_eq!(
            format!("{:016x}", fnv1a(bytes)),
            header_token(name, "fnv1a64 "),
            "{name}: FNV-1a over the live bytes does not match the pin the header records. \
             Recompute it — do NOT retype a hash out of a report, which is how the S0 \
             hand-off came to carry `84a8e18f0f2b02b5` for this file"
        );
    }
}

/// **T1's measured weight, as bytes in tree rather than as a statistic in prose.**
///
/// `qwen-architecture.md`'s T1 row is the one place that doc OVERRIDES a primary source with
/// a measurement: the tech report calls every RMSNorm zero-centred, and the GDN output norm
/// is not — the reference applies a bare `w ⊙ x̂` there. The evidence is these 256 bytes. It
/// drives the `gdn_norm_unit_offset` defect row a kernel will be scored against, so it is
/// decoded and re-verdicted here instead of trusted.
///
/// **The DISCRIMINANT is the offset, not the digits.** A zero-centred weight is centred on 0
/// and read through `(1.0 + w)`; a ones-centred weight is centred on 1 and read bare.
/// `min > 0.5` separates them by a margin no fp16 narrowing can close — this tensor's own
/// minimum is 0.875, and its zero-centred sibling `hc_norm` (not vendored; its provenance is
/// tabulated in the doc) runs −5.94 to +6.63. The digits are pinned transitively by the hash
/// test above, so nothing here restates one.
#[test]
fn the_gdn_output_norm_weight_is_ones_centred() {
    assert_eq!(NORM_L0.len(), 256, "BF16 [128] is 256 bytes");
    let vals: Vec<f32> = NORM_L0
        .chunks_exact(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect();
    // The examined count, asserted rather than assumed: a `chunks_exact` yielding nothing
    // would leave every assertion below vacuously true.
    assert_eq!(vals.len(), 128, "scored elements");
    let (min, max) = vals
        .iter()
        .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    assert!(
        min > 0.5,
        "layers.0.linear_attn.norm.weight runs [{min}, {max}] with mean {mean}. A minimum at \
         or below 0.5 would mean these weights are ZERO-centred — the tech report's reading — \
         and the port would then owe `(1.0 + w)` here, changing every GDN layer's output. T1 \
         and the `gdn_norm_unit_offset` defect row both rest on this"
    );
    assert!(
        (0.9..1.1).contains(&mean),
        "mean {mean} is not centred on one; T1's verdict is that it is"
    );
}

/// **Which families the block-128 fp8 scheme actually covers, from the config's own
/// quantization block.**
///
/// The plan of record transcribed the source artifact as "fine-grained block-128 e4m3 (the
/// scheme `Fp8W` already ingests)", full stop. The vendored config says something narrower and
/// this test is why the correction is not a fourth copy of a number nobody re-derives: the
/// block scheme covers everything NOT in `modules_to_not_convert` — i.e. the routed experts —
/// and the 51.2 G-element n-gram table is named in `modules_to_convert` and carries a single
/// per-TENSOR BF16 `weight_scale` instead of a grid. `Fp8W`'s block path does not apply to it,
/// which is a converter fact and not a footnote.
#[test]
fn the_fp8_block_scheme_covers_the_routed_experts_and_not_the_ngram_table() {
    let v: serde_json::Value =
        serde_json::from_slice(CONFIG).expect("the vendored config must parse");
    let q = &v["quantization_config"];
    assert_eq!(q["quant_method"], "fp8");
    assert_eq!(q["weight_block_size"][0], 128);
    assert_eq!(q["weight_block_size"][1], 128);
    assert_eq!(
        q["weight_per_tensor"], false,
        "the EXPERTS are block-scaled; a per-tensor flag here would mean one scale for all of          them, which is the other reading of `weight_scale` vs `weight_scale_inv`"
    );
    assert_eq!(
        q["modules_to_convert"].as_array().map(Vec::len),
        Some(1),
        "exactly one family is opted IN by name"
    );
    assert_eq!(
        q["modules_to_convert"][0], "ple.ple_embedding.ngram_embedding",
        "the n-gram table is the opted-in family, and its scale is per-tensor"
    );
    let excluded = q["modules_to_not_convert"]
        .as_array()
        .expect("modules_to_not_convert");
    // Not a magic number: it is the count the plan doc cites when it says the routed experts
    // are "the only families not in the config's own exclusion list", and the two must agree.
    assert_eq!(excluded.len(), 943, "excluded module count");
    for name in ["lm_head", "model.language_model.embed_tokens"] {
        assert!(
            excluded.iter().any(|m| m == name),
            "{name} must be excluded from fp8 — a quantized head or embedding would change              every logit"
        );
    }
    assert!(
        !excluded
            .iter()
            .any(|m| m.as_str().is_some_and(|m| m.ends_with(".mlp.experts"))),
        "no routed-expert family may appear in the exclusion list: they are exactly what the          block-128 scheme quantizes"
    );
}

/// The vendored `config.json` is the document the ROOT-ONLY sniff reads, and the near-miss
/// this port's identity rests on is IN it.
///
/// `crates/artifact/src/schema.rs` gates the resolution itself. What is gated here is that
/// the near-miss is REAL rather than hypothetical — `vision_config.model_type` really is the
/// root string verbatim in the shipped file — because if upstream renamed it, the root-only
/// rule would still be right while its argument was gone, and the next reader would find a
/// rule with no reason and relax it.
#[test]
fn the_vendored_config_carries_the_root_only_near_miss() {
    let v: serde_json::Value =
        serde_json::from_slice(CONFIG).expect("the vendored config must parse");
    assert_eq!(v["model_type"], "qwen4_exp");
    assert_eq!(v["architectures"][0], "Qwen4ExpForConditionalGeneration");
    assert_eq!(
        v["vision_config"]["model_type"], v["model_type"],
        "the vision block's model_type is no longer the root string verbatim — the root-only \
         reading in schema.rs and arch.rs cites this as its whole reason"
    );
    assert_eq!(
        v["text_config"]["model_type"], "qwen4_exp_text",
        "the text half's own spelling, which the sniff must NOT accept"
    );
    // The nesting is mandatory and present: every architecture dimension lives under
    // `text_config`, which is what makes the root-only reading a decision rather than an
    // accident of where the fields happen to sit.
    let t = &v["text_config"];
    assert_eq!(t["num_hidden_layers"], 48);
    assert_eq!(
        t["ple_layer_ids"][0], 2,
        "ONE-indexed, so the host is layer_idx 1"
    );
    assert!(
        t["layer_types"]
            .as_array()
            .expect("layer_types")
            .iter()
            .any(|l| l == "full_attention"),
        "the checkpoint's own spelling is `full_attention`, which the reference REWRITES to \
         `qwen_sparse_attention` because those layers run the indexer. A port reading this \
         field at face value implements dense attention on 12 of 48 layers — and is \
         bit-identical below 2051 cached tokens, so no planned oracle sees it"
    );
}
