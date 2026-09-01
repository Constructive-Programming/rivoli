//! **The vendored Qwen3.8-Flash-Next anchor goldens, and the four accessors every gate over them
//! needs.** One owner for the byte pins and one owner for the loader.
//!
//! Three test binaries read these files — `qwen_anchor.rs` (provenance, bytes, structure, the
//! width audit, the tolerance table), `qwen_anchor_fixtures.rs` (the operator fixtures, the value
//! checks, the capture and defect censuses) and `qwen_anchor_windows.rs` (the prefill window and
//! the real-parameter n-gram anchor). They are separate binaries because one file carrying all of
//! it came to 1290 lines against this repo's 1200-line hard cap, which is the same split Muse
//! Glimmer's anchor already took across four files.
//!
//! Splitting them is exactly what makes this module necessary: `load` plus `meta_json` plus
//! `width` is about thirty tokens of body, well over `build.rs`'s jscpd floor of fifteen, so a
//! second copy is a build error — and, the reason that gate exists, the copies would drift on
//! which `Arch` they pass or which key they call the widths. The byte pins matter more still: two
//! files each pinning `qwen-anchor-decode-qwen-anchor-1.bin`'s FNV is two constants to update on a
//! re-vendor, and the one that gets missed is the one that was protecting something.
//!
//! Included with `#[path]` rather than through a `common/mod.rs`, like `tolerance.rs` and
//! `golden_read.rs`: a test binary that wants these does not want the artifact helpers next door.
//! It re-exports `golden_read` rather than making each consumer include both, so a consumer names
//! one module.

#![allow(dead_code)] // each of the three consumers uses a subset; the rest is not dead for them

#[path = "golden_read.rs"]
pub mod golden_read;

use rivoli_core::legality::Arch;
use serde_json::Value;

// `#[allow(unused_imports)]` on a `pub use`, and the argument is the same one `golden_read.rs`
// writes for its `#![allow(dead_code)]`: this module is included with `#[path]` into three PRIVATE
// module positions, so a re-export no consumer reaches through is an unused import in that binary
// rather than a public API. Each of the three uses a different subset — the windows file never
// calls `check_pinned_bytes`, the provenance file never calls `float` — and with
// `[workspace.lints.rust] warnings = deny` every one of those is a compile error. The alternative
// is three divergent import lists in this file, which is the drift it exists to prevent.
#[allow(unused_imports)]
pub use golden_read::{
    GoldenSet, Vendored, capture_census, captured_layers, check_pinned_bytes,
    decay_rates_are_not_all_equal, declared_fields, draws_are_independent, float, ints,
    json_usize as field, logits_are_finite_and_spread, names, one_row_of_logits, provenance,
    shape_of, top_k_is_in_range_and_distinct,
};

/// **The two decode draws.** Everything that says "both draws" means these.
///
/// Two rather than one because a single draw cannot show that a property is a fact about the
/// arithmetic rather than about the numbers it landed on, and a kernel bug degenerate at one
/// draw's values hides completely — so the second salt is coverage, not redundancy.
///
/// **When these constants fail after a deliberate regeneration, update them and say so in
/// `docs/measurement/qwen-reference/anchor.md`.** Re-vendoring is a reviewed change, not a side
/// effect of running the driver — and a re-vendor invalidates every red proof measured against the
/// old bytes, so those have to be re-run too.
pub const DECODE: &[Vendored] = &[
    Vendored {
        name: "qwen-anchor-1",
        bytes: include_bytes!("../qwen-anchor-decode-qwen-anchor-1.bin"),
        len: 338_050,
        fnv: 0xee98_0262_f256_e9b5,
    },
    Vendored {
        name: "qwen-anchor-2",
        bytes: include_bytes!("../qwen-anchor-decode-qwen-anchor-2.bin"),
        len: 338_050,
        fnv: 0x013a_8394_17a2_a7f1,
    },
];

/// The 16-position prefill window, salt 1 only, layer 3 only. It exists for one claim the decode
/// goldens cannot make — the dense-versus-selective boundary, which is only visible across
/// QUERIES.
pub const PREFILL: Vendored = Vendored {
    name: "qwen-anchor-prefill",
    bytes: include_bytes!("../qwen-anchor-prefill-qwen-anchor-1.bin"),
    len: 399_035,
    fnv: 0x7cf7_e4d4_f811_a461,
};

/// The n-gram hash at FULL width. The only golden here whose parameters are the checkpoint's own
/// rather than a shrunk copy of them.
pub const NGRAM_REAL: Vendored = Vendored {
    name: "qwen-anchor-ngram-real",
    bytes: include_bytes!("../qwen-anchor-ngram-real.bin"),
    len: 2_891,
    fnv: 0x6400_56bb_5509_61bf,
};

/// The vendored real config, the same file `crates/artifact/tests/qwen_names.rs` pins the bytes of.
pub const REAL_CONFIG: &str =
    include_str!("../../../../docs/measurement/qwen-reference/config.json");

/// Every vendored golden, for the gates that hold over all four.
pub fn all() -> Vec<&'static Vendored> {
    DECODE.iter().chain([&PREFILL, &NGRAM_REAL]).collect()
}

pub fn load(v: &Vendored) -> GoldenSet {
    GoldenSet::read_anchor_for(Arch::QwenFlashNext, &mut &v.bytes[..])
        .unwrap_or_else(|e| panic!("the vendored {} golden must load: {e:#}", v.name))
}

/// One metadata value parsed as JSON. Panics naming the key, because every caller's next question
/// on absence is "then what does the file say".
pub fn meta_json(g: &GoldenSet, key: &str) -> Value {
    serde_json::from_str(g.meta_get(key).unwrap_or_else(|| panic!("{key} metadata")))
        .unwrap_or_else(|e| panic!("{key} is not valid json: {e}"))
}

/// A width out of a golden's recorded `widths` or `real_widths` map. Derived at generation time
/// from the config, so this is the number the reference was actually built with rather than one
/// restated here — the rule K3's anchor learned from a review that found four accidentally-equal
/// quantities behind assertions that looked like they pinned a coupling. A derived width fails when
/// the config drifts; a literal agrees with it.
pub fn width(w: &Value, key: &str) -> usize {
    w[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is not in the recorded widths")) as usize
}

/// An int tensor's SHAPE, which [`ints`] does not return.
pub fn int_shape<'g>(g: &'g GoldenSet, name: &str) -> &'g [usize] {
    g.ints
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, s, _)| s.as_slice())
        .unwrap_or_else(|| panic!("{name} is not among the golden's int tensors"))
}

/// Within-layer execution order, the linear order the python comparator's `_bucket_rank` uses.
/// Layer-major above it: a defect first touching layer 3's indexer must leave every tensor of
/// layers 0-2 identical, whatever bucket it is in.
///
/// Every position is read off the reference: `DecoderLayer.forward` is PLE, attention fold,
/// attention family, MLP fold, MoE (MOD:1207-1243); `Attention.forward` is indexer, projections,
/// q/k norms, rope, attend, gate, `o_proj` (MOD:790-833); `SparseMoeBlock.forward` runs the SHARED
/// expert before the router (MOD:928-930).
///
/// **It is a second copy of the python side's tuple, and that is deliberate.** The census's job is
/// to check the declaration from OUTSIDE the tool that made it; importing the order from the same
/// place would make the check circular. A drift between the two is caught by the golden's own
/// `first_touch` map, every bucket of which must be found here by name.
pub const BUCKET_ORDER: [&str; 17] = [
    "ngram_hash",
    "ple_inject",
    "hc_attn",
    "gdn_proj",
    "gdn_conv",
    "gdn_op",
    "gdn_out_norm",
    "gdn_out",
    "indexer",
    "qsa_proj",
    "qk_norm",
    "qsa",
    "hc_mlp",
    "moe_shared",
    "moe_route",
    "moe",
    "residual",
];

/// `(layer, rank)` for a captured tensor, or `None` for the model-level tail. **Order is
/// semantic** — `self_attn.indexer` must be tested before the `self_attn` it starts with, and
/// `linear_attn.norm` before `linear_attn`.
pub fn bucket_of(name: &str) -> Option<(usize, usize)> {
    let rest = name.strip_prefix("model.layers.")?;
    let mut parts = rest.splitn(2, '.');
    let layer: usize = parts.next()?.parse().ok()?;
    let tail = parts.next()?;
    let operator = [
        ("ple.ple_embedding", "ngram_hash"),
        ("ple", "ple_inject"),
        ("attn_hyper_connection", "hc_attn"),
        ("mlp_hyper_connection", "hc_mlp"),
        ("linear_attn.gated_delta_rule", "gdn_op"),
        ("linear_attn.conv", "gdn_conv"),
        ("linear_attn.norm", "gdn_out_norm"),
        ("linear_attn.in_proj", "gdn_proj"),
        ("linear_attn", "gdn_out"),
        ("self_attn.indexer", "indexer"),
        ("self_attn.q_proj", "qsa_proj"),
        ("self_attn.k_proj", "qsa_proj"),
        ("self_attn.v_proj", "qsa_proj"),
        ("self_attn.q_norm", "qk_norm"),
        ("self_attn.k_norm", "qk_norm"),
        ("self_attn", "qsa"),
        ("mlp.gate", "moe_route"),
        ("mlp.shared_expert", "moe_shared"),
        ("mlp", "moe"),
    ]
    .iter()
    .find(|(prefix, _)| tail.starts_with(prefix))
    .map_or("residual", |(_, op)| op);
    Some((layer, BUCKET_ORDER.iter().position(|b| *b == operator)?))
}
