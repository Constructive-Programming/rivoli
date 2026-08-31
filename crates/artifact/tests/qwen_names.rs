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
//! Every expected value is read out of the TSV's OWN header, so there is one copy of each number
//! rather than two agreeing frozen ones, and a moved header phrase panics rather than defaulting —
//! a reword must be a visible edit, not a check that quietly stops checking.
//!
//! > **S4, 2026-08-31: the parser debt is PAID and the census gained three cells.** This file
//! > carried its own 8-column parser and its own header accessors, and its header named the debt:
//! > "lifting one shared helper is owed at S4, when the converter census gives the second reader a
//! > reason to exist". S4 gave it three — `convert_qwen`, `crates/cli/tests/qwen_convert.rs` and
//! > this file — so the reader moved into `rivoli_artifact::census` (library code, because a
//! > BINARY reads it: the converter's exclusion list IS the census) and `k3_names.rs::families`
//! > moved onto it in the same commit. What this file now reads is `census::qwen::Census`, the
//! > parsed-and-role-assigned form, which is also what the converter drives off — so the gate and
//! > the tool cannot disagree about what a family is. The three new cells are the per-ROLE tally
//! > (every arm named, with byte totals), the `weight_scale_inv` grid orientation, and the n-gram
//! > hash derivation, plus — since the derived-widths cell moved here from the converter gate,
//! > where it needed no converter binary — the CONFIG-versus-CENSUS agreement: every v1 family's
//! > shape recomputed from `text_config`'s head counts and confronted with the shard headers'.
//!
//! No GPU, no network: 22 KB of vendored TSV, 71 KB of vendored JSON, 256 B of vendored
//! weight payload.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::census::qwen::{Census, FAMILIES, Role, collapse, ngram_hash};
use rivoli_artifact::format::Dtype;
use rivoli_artifact::qwen_config::{QwenConfig, QwenTextConfig};
use rivoli_artifact::schema::parse_config;
use rivoli_core::hash::fnv1a;

const DIR: &str = "docs/measurement/qwen-reference";
// Bytes, not `&str`, for all three: the pins below are over the FILE, and a `&str` would
// normalise nothing today but makes "these are the bytes that hash" a claim about UTF-8
// rather than about the file. `config.json` is also parsed as JSON below, from this same
// constant, so the parse and the pin cannot be reading different content.
const CONFIG: &[u8] = include_bytes!("../../../docs/measurement/qwen-reference/config.json");
const GENERATION: &[u8] =
    include_bytes!("../../../docs/measurement/qwen-reference/generation_config.json");
const NORM_L0: &[u8] =
    include_bytes!("../../../docs/measurement/qwen-reference/linear_attn_norm_l0.bin");

fn census() -> Census {
    Census::load().expect("the vendored qwen census must parse")
}

/// The shipped config, through the real resolution path — the same call the converter makes.
fn shipped() -> QwenConfig {
    parse_config(std::str::from_utf8(CONFIG).expect("the vendored config is UTF-8"))
        .expect("the shipped Qwen3.8-Flash-Next config must load and validate")
}

/// A count the header declares as `# <label>   <number>`.
fn declared(label: &str) -> u64 {
    FAMILIES.declared(label).unwrap_or_else(|e| panic!("{e:#}"))
}

/// A `params`/`bytes` total the header declares for one `v1-status`.
fn declared_for(status: &str, keyword: &str) -> u64 {
    FAMILIES
        .header_token(&format!("STATUS {status} "), &format!("{keyword} "))
        .unwrap_or_else(|e| panic!("{e:#}"))
        .parse()
        .unwrap_or_else(|e| panic!("STATUS {status}'s {keyword} is not a number: {e}"))
}

/// The header line that starts `#   <anchor>`.
///
/// Factored out of the two accessors below because they shared it token for token and jscpd
/// reported the pair — which is also the honest shape: finding the line and reading numbers off it
/// are two steps, and only the second differs between the two layouts.
fn header_line(anchor: &str) -> &'static str {
    let text = FAMILIES.text();
    let at = text
        .find(&format!("\n#   {anchor}"))
        .unwrap_or_else(|| panic!("{DIR}/tensor-families.tsv has no `#   {anchor}` line"));
    text[at + 1..].lines().next().expect("a line follows")
}

/// The `want` integers on the lines FOLLOWING `#   <anchor>` — how the header lays out the two
/// 16-element I64 buffers, which do not fit on one line.
///
/// Two accessors rather than one loose scan, because the header states the two shapes differently
/// and a loose scan is how a number gets read off the wrong line: the `ngram_heads_vocab_sizes`
/// anchor LINE carries the literal `16` twice ("16 distinct PRIMES", "all 16 verified prime"), so
/// including it would collect two spurious values.
fn header_list(anchor: &str, want: usize) -> Vec<u64> {
    let anchored = header_line(anchor);
    let after = FAMILIES.text().find(anchored).expect("its own line") + anchored.len();
    let mut out: Vec<u64> = Vec::new();
    for line in FAMILIES.text()[after..].lines() {
        out.extend(
            line.trim_start_matches('#')
                .split_whitespace()
                .filter_map(|t| t.parse::<u64>().ok()),
        );
        if out.len() >= want {
            break;
        }
    }
    assert_eq!(out.len(), want, "`{anchor}` should list {want} integers");
    out
}

/// The integers ON the `#   <anchor>` line, after its `=` — how the header states the three
/// multipliers.
fn header_inline(anchor: &str, want: usize) -> Vec<u64> {
    let (_, rhs) = header_line(anchor)
        .split_once('=')
        .expect("the anchor line carries an `=`");
    let out: Vec<u64> = rhs
        .split_whitespace()
        .filter_map(|t| t.parse::<u64>().ok())
        .collect();
    assert_eq!(out.len(), want, "`{anchor}` should state {want} integers");
    out
}

/// **The census closes, recomputed from the rows rather than asserted in prose.**
///
/// This is the assertion `EQUAL: True` was standing in for. Both totals come out of the
/// header, so the rows and the provenance lines have to agree; what neither this nor the
/// header can see is an upstream revision that ADDS a tensor family, which is why the
/// revision pin two blocks above them is the other half of the evidence.
#[test]
fn the_vendored_census_closes_against_its_own_header() {
    let c = census();
    let fams = c.families();
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
    // Duplicate patterns are refused by `Census::load` itself — they are the converter's lookup
    // keys, so a duplicate would leave one of two rows unreachable while every sum balanced.
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
    let c = census();
    let fams = c.families();
    assert!(
        !fams.is_empty(),
        "no rows parsed — the file or the comment filter moved"
    );
    for f in fams {
        assert_eq!(
            f.bytes_per_tensor * f.count,
            f.total_bytes,
            "{}: bytes-per-tensor {} x count {} != total-bytes {}",
            f.pattern,
            f.bytes_per_tensor,
            f.count,
            f.total_bytes
        );
        let width: u64 = match f.dtype {
            Dtype::F8E4M3 => 1,
            Dtype::Bf16 => 2,
            Dtype::I64 => 8,
            d => panic!("{}: unexpected dtype {d:?}", f.pattern),
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
///
/// The status list is DERIVED from the rows rather than written down: `Census::load` refuses a
/// status it does not know, so the set is already gated, and a frozen list here would be one more
/// copy that can disagree.
#[test]
fn the_parameter_counts_split_by_v1_status_are_re_derivable() {
    let c = census();
    let mut statuses: Vec<&str> = c.families().iter().map(|f| f.status.as_str()).collect();
    statuses.sort_unstable();
    statuses.dedup();
    assert_eq!(
        statuses.len(),
        3,
        "the census should carry exactly three v1-status values, got {statuses:?}"
    );
    for status in statuses {
        let rows: Vec<_> = c.families().iter().filter(|f| f.status == status).collect();
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

/// **Every arm named, with byte totals** — the converter's own census line, checked against the
/// TSV's status totals.
///
/// The track rule this cell exists for: a census that sums to the right grand total while one
/// class silently vanished is the failure mode. So the eight ROLES are tallied independently and
/// each is required non-empty, and the two excluded roles are required to reproduce the header's
/// own `EXCLUDED-v1-*` byte figures — which is the statement "MTP and vision are excluded BY NAME,
/// and here is how many bytes that names".
///
/// RED OBSERVED (plant P11, the `NgramShard` role predicate spelled `shard_{Z}` so it stops
/// matching): the census refused to LOAD — `…ngram_embedding.shard_{S}.weight: role resident wants
/// Bf16 at rank [1, 2, 3], but the census row is F8E4M3 [2500012, 160]`. A role predicate that
/// stops matching is therefore caught by `check_role_layout` first, which is the stronger place for
/// it; this cell stays the backstop for a role whose fallthrough would be layout-compatible.
#[test]
fn every_role_arm_is_populated_and_the_excluded_arms_are_the_headers_own() {
    let c = census();
    let mut total = 0u64;
    for role in Role::ALL {
        let rows: Vec<_> = c.families().iter().filter(|f| f.role == role).collect();
        assert!(
            !rows.is_empty(),
            "role {} matched no family — a role predicate that stops matching turns every \
             assertion about that arm vacuous, and the converter would then write nothing for it",
            role.label()
        );
        total += c.role_totals(role).1;
    }
    assert_eq!(
        total,
        declared("index metadata.total_size"),
        "the eight roles must partition the census — every family in exactly one arm"
    );
    for (role, status) in [
        (Role::ExcludedMtp, "EXCLUDED-v1-MTP"),
        (Role::ExcludedVision, "EXCLUDED-v1-VISION"),
    ] {
        let (_, arm) = c.role_totals(role);
        assert_eq!(
            arm,
            declared_for(status, "bytes"),
            "{}: the role tally and the header's status total must be the same bytes",
            role.label()
        );
    }
}

/// **The `weight_scale_inv` grid orientation, per projection — the 80.5 GB trap.**
///
/// `Census::confront_scale_grids` derives each grid from its own weight's shape, so this cell is
/// the vendored pair being confronted BEFORE 172.76 GiB moves. The two orientations are also
/// asserted here explicitly, because "derived correctly" and "the two projections really do
/// differ" are different claims and only the second one says the check has anything to catch.
///
/// RED OBSERVED (plant P9, `down_proj.weight_scale_inv`'s TSV shape transposed to `5x20`):
/// `down_proj.weight_scale_inv is [5, 20] but its weight [2560, 640] implies a [out/128, in/128]
/// grid of [20, 5]. A transposed grid hands every tile a different tile's scale over 40265318400 B
/// of expert weight` — from the DERIVED check, not from a tabulated expectation.
#[test]
fn the_routed_scale_grids_are_per_projection_and_the_two_orientations_differ() {
    let c = census();
    c.confront_scale_grids()
        .expect("the vendored census's grids must agree with their weights");
    let grid = |pattern: &str| {
        c.families()
            .iter()
            .find(|f| f.pattern.ends_with(pattern) && f.status == "v1")
            .map(|f| f.shape.clone())
            .unwrap_or_else(|| panic!("no v1 family ending {pattern}"))
    };
    assert_eq!(
        grid("mlp.experts.{E}.down_proj.weight_scale_inv"),
        vec![20, 5],
        "down_proj's weight is [2560, 640], so its grid is [20, 5]"
    );
    for proj in ["gate_proj", "up_proj"] {
        assert_eq!(
            grid(&format!("mlp.experts.{{E}}.{proj}.weight_scale_inv")),
            vec![5, 20],
            "{proj}'s weight is [640, 2560], so its grid is [5, 20] — the TRANSPOSE of \
             down_proj's, which is why a single tabulated orientation silently mis-scales two \
             thirds of the expert weight"
        );
    }
}

/// **The n-gram hash parameters, re-derived in Rust and byte-compared against the checkpoint's own
/// I64 buffers — at index 0, which must match, AND at index 1, which must not.**
///
/// `qwen-architecture.md` §5 names this gate as owed at S4 and says why a golden over the three
/// multipliers alone would not do: they do not depend on `head_idx` at all, so such a golden
/// passes on a wrong `global_head_idx` and the 16 primes are where that error shows.
///
/// RED OBSERVED (plant P10, `splitmix64` without the golden-gamma increment):
/// `left: [16547087256759, 23703573157769, 20109073645365]` against
/// `right: [23703573157769, 20109073645365, 8052911324071]` — the sequence SHIFTED BY ONE, carrying
/// two of the three correct multipliers in the wrong slots. A gate comparing them as a set, or
/// checking only `m_1`, would have passed on it.
///
/// The index-1
/// arm is the other half: the byte-exact match at index 0 is what makes "this checkpoint's PLE
/// layer index is 0" evidence rather than an assumption, and it is only evidence if index 1
/// disagrees.
#[test]
fn the_ngram_hash_parameters_are_re_derivable_at_index_zero_and_not_at_one() {
    let cfg = shipped();
    let t = &cfg.text;
    // The host layer, one-indexed in the file and zero-based here — the trap T12 conversion.
    assert_eq!(
        t.ple_host_layer().expect("a PLE host"),
        1,
        "ple_layer_ids [2] is ONE-INDEXED, so the host is layer_idx 1"
    );
    let h0 = ngram_hash(t, 0).expect("the derivation must run at index 0");
    assert_eq!(
        h0.vocab_sizes,
        header_list("ngram_heads_vocab_sizes", t.ngram_heads()),
        "the 16 head vocabularies must be the checkpoint's own I64 buffer"
    );
    assert_eq!(
        h0.offsets,
        header_list("ngram_heads_offsets", t.ngram_heads()),
        "the offsets are the exclusive prefix sums, and the checkpoint ships them"
    );
    assert_eq!(
        h0.multipliers,
        header_inline("layer_multipliers", t.ngram_size),
        "the three multipliers, IN ORDER — the wrong SplitMix64 form yields a sequence \
         containing two of these three in the wrong slots, so a set comparison or a single-value \
         check would pass on it"
    );
    // Geometry: the padded table divides into the shards, and the shard shape the census records
    // is what the derivation implies. This is the other end of the same claim.
    let shard = census()
        .families()
        .iter()
        .find(|f| f.role == Role::NgramShard)
        .map(|f| f.shape.clone())
        .expect("an n-gram shard family");
    assert_eq!(
        shard,
        vec![
            h0.rows_per_shard,
            t.ngram_head_dim().expect("a head dim") as u64
        ],
        "the census's shard shape must be the derived rows-per-shard by the gathered row width"
    );
    assert_eq!(
        h0.padded_rows,
        h0.rows_per_shard * t.split_ngram_parts as u64,
        "the shards are ordered row RANGES and must tile the padded table exactly"
    );

    // Index 1 — the discriminant. BOTH halves of the per-layer indexing move, and if only one of
    // them did, two PLE layers would collide onto the same rows with plausible output and no
    // crash (`qwen-architecture.md` §5).
    let h1 = ngram_hash(t, 1).expect("the derivation must run at index 1 too");
    assert_ne!(
        h1.vocab_sizes, h0.vocab_sizes,
        "the head vocabularies must CONTINUE past the first layer's primes; if they restarted, \
         the byte-exact match above would not be evidence that this checkpoint's index is 0"
    );
    assert_ne!(
        h1.multipliers, h0.multipliers,
        "the multipliers must reseed per PLE layer (`base_seed = seed + 10007 * index`)"
    );
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
        let want_len: usize = FAMILIES
            .header_token(name, "bytes ")
            .unwrap_or_else(|e| panic!("{e:#}"))
            .parse()
            .expect("declared bytes");
        assert_eq!(bytes.len(), want_len, "{name}: length");
        assert_eq!(
            format!("{:016x}", fnv1a(bytes)),
            FAMILIES
                .header_token(name, "fnv1a64 ")
                .unwrap_or_else(|e| panic!("{e:#}")),
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
        "the EXPERTS are block-scaled; a per-tensor flag here would mean one scale for all of \
         them, which is the other reading of `weight_scale` vs `weight_scale_inv`"
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
            "{name} must be excluded from fp8 — a quantized head or embedding would change \
             every logit"
        );
    }
    assert!(
        !excluded
            .iter()
            .any(|m| m.as_str().is_some_and(|m| m.ends_with(".mlp.experts"))),
        "no routed-expert family may appear in the exclusion list: they are exactly what the \
         block-128 scheme quantizes"
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

// ------------------------------------------------------------------------------------------
// The census against the CONFIG — two independent derivations of the same shapes.
// ------------------------------------------------------------------------------------------

/// Every v1 family's shape, derived from `text_config`'s HEAD COUNTS rather than read off the
/// census — `(census-pattern suffix, shape)`. The suffix match lets one row cover the three
/// hyper-connection sites, which genuinely share a width; every row must match at least one family
/// and every family exactly one row, so neither list can drift out from under the other.
fn derived_shapes(t: &QwenTextConfig) -> Vec<(&'static str, Vec<u64>)> {
    let w = |v: &[usize]| v.iter().map(|&d| d as u64).collect::<Vec<u64>>();
    let (h, hc, lo) = (t.hidden, t.hc_stream(), t.hc_lowrank);
    let hash = ngram_hash(t, 0).expect("the n-gram derivation");
    let rows = hash.rows_per_shard as usize;
    vec![
        ("lm_head.weight", w(&[t.vocab, h])),
        ("embed_tokens.weight", w(&[t.vocab, h])),
        // GDN, and the asymmetry is the whole point: qkv is 2 x 16 x 128 + 48 x 128.
        ("linear_attn.in_proj_qkv.weight", w(&[t.gdn_qkv_width(), h])),
        ("linear_attn.in_proj_z.weight", w(&[t.gdn_value_width(), h])),
        ("linear_attn.out_proj.weight", w(&[h, t.gdn_value_width()])),
        (
            "linear_attn.in_proj_a.weight",
            w(&[t.linear_num_value_heads, h]),
        ),
        (
            "linear_attn.in_proj_b.weight",
            w(&[t.linear_num_value_heads, h]),
        ),
        (
            "linear_attn.conv1d.weight",
            w(&[t.gdn_qkv_width(), 1, t.linear_conv_kernel_dim]),
        ),
        ("linear_attn.A_log", w(&[t.linear_num_value_heads])),
        ("linear_attn.dt_bias", w(&[t.linear_num_value_heads])),
        ("linear_attn.norm.weight", w(&[t.linear_value_head_dim])),
        // QSA. `q_proj` is TWICE the output width — see `qsa_q_width`'s doc for the reading.
        ("self_attn.q_proj.weight", w(&[t.qsa_q_width(), h])),
        ("self_attn.o_proj.weight", w(&[h, t.qsa_out_width()])),
        ("self_attn.k_proj.weight", w(&[t.qsa_kv_width(), h])),
        ("self_attn.v_proj.weight", w(&[t.qsa_kv_width(), h])),
        ("self_attn.q_norm.weight", w(&[t.head_dim])),
        ("self_attn.k_norm.weight", w(&[t.head_dim])),
        (
            "self_attn.indexer.index_qk_proj.weight",
            w(&[t.indexer_qk_width(), h]),
        ),
        (
            "self_attn.indexer.q_layernorm.weight",
            w(&[t.indexer_head_dim]),
        ),
        (
            "self_attn.indexer.k_layernorm.weight",
            w(&[t.indexer_head_dim]),
        ),
        // MoE. Note gate/up are [inter, hidden] and down is [hidden, inter] — the pairing whose
        // transposition is what `confront_scale_grids` catches on the routed side.
        ("mlp.gate.weight", w(&[t.num_experts, h])),
        ("mlp.shared_expert.gate_proj.weight", w(&[t.moe_inter, h])),
        ("mlp.shared_expert.up_proj.weight", w(&[t.moe_inter, h])),
        ("mlp.shared_expert.down_proj.weight", w(&[h, t.moe_inter])),
        ("mlp.shared_expert_gate.weight", w(&[1, h])),
        ("mlp.experts.{E}.gate_proj.weight", w(&[t.moe_inter, h])),
        ("mlp.experts.{E}.up_proj.weight", w(&[t.moe_inter, h])),
        ("mlp.experts.{E}.down_proj.weight", w(&[h, t.moe_inter])),
        // Hyper-connections: a 4-wide stream mixed through a 320 rank, at three sites.
        ("input_mix_weight_down.weight", w(&[lo, hc])),
        ("input_mix_weight_up.weight", w(&[hc, lo])),
        ("block_inject_weight.weight", w(&[t.hc_count, hc])),
        ("hc_norm.weight", w(&[hc])),
        // PLE. `key_proj` maps hidden into the FULL hc stream, which is what makes the injected
        // value 10240 wide rather than 2560.
        ("ple.key_proj.weight", w(&[hc, h])),
        ("ple.value_proj.weight", w(&[h, h])),
        ("ple.norm_query.weight", w(&[hc])),
        ("ple.norm_key.weight", w(&[hc])),
        ("ple.norm_conv.weight", w(&[hc])),
        ("ple.conv1d.weight", w(&[hc, 1, t.ple_conv_kernel_size])),
        // The n-gram table and its parameters.
        (
            "ngram_embedding.shard_{S}.weight",
            w(&[rows, t.ngram_head_dim().expect("a head dim")]),
        ),
        ("ngram_embedding.weight_scale", w(&[1])),
        ("ple_embedding.ngram_heads_offsets", w(&[t.ngram_heads()])),
        (
            "ple_embedding.ngram_heads_vocab_sizes",
            w(&[t.ngram_heads()]),
        ),
        ("ple_embedding.layer_multipliers", w(&[t.ngram_size])),
    ]
}

/// **The config's head counts and the census's shard headers agree on every v1 shape.**
///
/// The three `weight_scale_inv` grids are excluded from the table and covered by
/// `Census::confront_scale_grids`, which DERIVES each grid from its own weight rather than
/// tabulating it — a table of expected grids would be a second copy of the thing under test.
///
/// RED OBSERVED (plant P6a, `qsa_q_width` returning `qsa_out_width` — the factor of 2 dropped):
/// `left: [12288, 2560] right: [6144, 2560]` on `self_attn.q_proj.weight`, and nothing else moved.
///
/// **A second plant reddened somewhere BETTER, and that is worth recording.** P6b halved
/// `gdn_qkv_width` (`2 * key + value` → `key + value`), and this test never reached its comparison:
/// the config refused to LOAD, with `the GDN qkv width does not decompose into q + k + v` from
/// `validate_linear_attention`. So that mis-derivation is caught one layer earlier, by the config's
/// own internal consistency check, and this cell is the backstop for the widths that have no such
/// check — `q_proj`'s doubling among them.
#[test]
fn the_config_derived_widths_are_the_censuses_own_shapes() {
    let (cfg, c) = (shipped(), census());
    let table = derived_shapes(&cfg.text);
    let mut matched_rows = vec![0usize; table.len()];
    let mut covered = 0usize;
    for f in c
        .families()
        .iter()
        .filter(|f| f.status == "v1" && f.role != Role::RoutedScale)
    {
        let hits: Vec<usize> = table
            .iter()
            .enumerate()
            .filter(|(_, (suffix, _))| f.pattern.ends_with(suffix))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            hits.len(),
            1,
            "{} matched {} rows of the derived table; every family must match exactly one, or \
             one of them is being scored against another's width",
            f.pattern,
            hits.len()
        );
        let i = hits[0];
        matched_rows[i] += 1;
        covered += 1;
        assert_eq!(
            f.shape, table[i].1,
            "{}: the census says {:?}, the config's head counts imply {:?}",
            f.pattern, f.shape, table[i].1
        );
    }
    // 50 = the 53 v1 families minus the three `weight_scale_inv` grids. Asserted rather than
    // assumed: this loop's whole power is that it examined every one of them.
    assert_eq!(covered, 50, "v1 families confronted with a derived width");
    for (i, (suffix, _)) in table.iter().enumerate() {
        assert!(
            matched_rows[i] > 0,
            "the derived row {suffix:?} matched no census family — a row nothing scores is a \
             width nobody is checking"
        );
    }
    c.confront_scale_grids()
        .expect("the routed grids must follow from their own weights");
}

/// Every census family is reachable from a real name — `collapse` is the converter's only lookup,
/// and a pattern nothing collapses to is a row the converter can never reach.
#[test]
fn every_census_pattern_is_reachable_by_collapsing_a_real_name() {
    let c = census();
    let mut seen = 0usize;
    for f in c.families() {
        let (pattern, _) = collapse(&f.example);
        assert_eq!(
            pattern, f.pattern,
            "the census's own example name for {} collapses to {pattern:?}",
            f.pattern
        );
        assert_eq!(
            c.family_of(&f.example).expect("must resolve").pattern,
            f.pattern
        );
        seen += 1;
    }
    assert_eq!(seen, 109, "every family row must be reachable");
}
