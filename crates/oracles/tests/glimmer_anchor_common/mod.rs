//! The tables and accessors the four Muse Glimmer S1b anchor binaries read the vendored goldens
//! through.
//!
//! `glimmer_anchor.rs` carries the framing and holds what is true of a FILE; `glimmer_anchor_text.rs`,
//! `glimmer_anchor_draft.rs` and `glimmer_anchor_arithmetic.rs` hold the three groups of property.
//! All four open the same files and derive the same widths from them, so the tables and the readers
//! live here once — written four times they are a `build.rs` jscpd failure at `--min-tokens 15`, and
//! four copies of a byte pin is four places for a re-vendor to be recorded in three of.
//!
//! **A directory module rather than a `tests/glimmer_anchor_common.rs` sibling**, because cargo
//! turns every `tests/*.rs` into a test binary: flat, this would embed the goldens a fifth time and
//! run zero tests. `tests/common/` next door is the same shape for the same reason.
//!
//! Widths are **derived from each golden's own `tiny_config`**, never written as literals, and the
//! fields that are supposed to be REAL are compared against the vendored `config.json` rather than
//! against constants. A literal agrees with drift; a derived value fails on it.

#![allow(dead_code)] // each binary uses a subset; the rest is live for its siblings
#![allow(unused_imports)] // ...and a re-export the including binary never names is that same fact
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use serde_json::Value;

#[path = "../common/golden_read.rs"]
pub mod golden_read;

/// Re-exported so a binary that reads goldens needs one import rather than two. Same argument
/// `golden_read` itself makes for re-exporting `GoldenSet`: an import list is the one duplication
/// Rust gives you no way to factor, so the fix is to have fewer imports.
pub use golden_read::{GoldenSet, Vendored, ints, shape_of};

/// Which mode a vendored file holds, read off the entry's own name.
///
/// **The mode is not a field here**, deliberately: the file already says what it is, the byte pin
/// binds this entry to that file, and a second copy of a fact the file carries is the shape this
/// port has already been bitten by. `the_anchor_goldens_record_what_produced_them` checks the two
/// agree.
pub fn is_mode(v: &Vendored, mode: &str) -> bool {
    v.name.starts_with(mode)
}

pub const GOLDENS: &[Vendored] = &[
    Vendored {
        name: "text-1",
        bytes: include_bytes!("../glimmer-anchor-text-1.bin"),
        len: 643_957,
        fnv: 0xc765_6dea_dd50_3c51,
    },
    Vendored {
        name: "text-2",
        bytes: include_bytes!("../glimmer-anchor-text-2.bin"),
        len: 643_957,
        fnv: 0xe778_0679_924e_cd5f,
    },
    Vendored {
        name: "draft-1",
        bytes: include_bytes!("../glimmer-anchor-draft-1.bin"),
        len: 72_145,
        fnv: 0x3dcf_a1ed_6536_a6f0,
    },
    Vendored {
        name: "draft-2",
        bytes: include_bytes!("../glimmer-anchor-draft-2.bin"),
        len: 72_145,
        fnv: 0xd15d_109a_9a72_f7ab,
    },
];

/// The vendored real config, the same file `glimmer-architecture.md` was extracted from.
pub const REAL_CONFIG: &str =
    include_str!("../../../../docs/measurement/glimmer-reference/config.json");

pub fn load(v: &Vendored) -> GoldenSet {
    GoldenSet::read_glimmer(&mut &v.bytes[..]).unwrap_or_else(|e| panic!("{}: {e:#}", v.name))
}

pub fn text_goldens() -> impl Iterator<Item = &'static Vendored> {
    GOLDENS.iter().filter(|v| is_mode(v, "text"))
}

pub fn draft_goldens() -> impl Iterator<Item = &'static Vendored> {
    GOLDENS.iter().filter(|v| is_mode(v, "draft"))
}

/// Run `f` over every text golden, already loaded, with its config and the four widths.
///
/// Three checks started with the same four lines — load, parse the config, read the widths, name
/// the file — and `build.rs`'s jscpd gate rejected them at 73 tokens. The duplication was real;
/// this is the factoring, not an exemption.
pub fn each_text(mut f: impl FnMut(&Vendored, &GoldenSet, &Value, Widths)) {
    for v in text_goldens() {
        let g = load(v);
        let c = cfg(&g);
        let w = widths(&c);
        f(v, &g, &c, w);
    }
}

/// The four widths every shape is built from, so that a config drift fails the gate instead of
/// being agreed with. Read together because they are only meaningful together — and carried as
/// one value so that no shape check needs four more arguments to say what it expects.
#[derive(Clone, Copy)]
pub struct Widths {
    pub hidden: usize,
    pub heads: usize,
    pub kv: usize,
    pub head_dim: usize,
}

impl Widths {
    /// The concatenated head width the output gate and `o_proj` work at. **Not `hidden`** — that
    /// collision is what `the_tiny_widths_did_not_collapse_a_distinction` exists to refuse.
    pub fn concat(self) -> usize {
        self.heads * self.head_dim
    }

    /// The GQA broadcast factor: how many query heads share one KV head.
    pub fn group(self) -> usize {
        self.heads / self.kv
    }
}

pub fn widths(c: &Value) -> Widths {
    Widths {
        hidden: num(c, "hidden_size"),
        heads: num(c, "num_attention_heads"),
        kv: num(c, "num_key_value_heads"),
        head_dim: num(c, "head_dim"),
    }
}

/// The tiny config a golden was produced under, parsed out of its own metadata.
pub fn cfg(g: &GoldenSet) -> Value {
    serde_json::from_str(meta(g, "tiny_config")).expect("tiny_config is JSON")
}

pub fn meta<'g>(g: &'g GoldenSet, key: &str) -> &'g str {
    g.meta_get(key)
        .unwrap_or_else(|| panic!("the golden carries no {key:?} in its metadata"))
}

pub fn num(c: &Value, key: &str) -> usize {
    c[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{key} is not an integer in the config: {}", c[key])) as usize
}

pub fn meta_usize(g: &GoldenSet, key: &str) -> usize {
    meta(g, key).parse().expect("a numeric metadata value")
}

pub fn real() -> Value {
    let top: Value = serde_json::from_str(REAL_CONFIG).expect("the vendored config.json parses");
    top["text_config"].clone()
}

/// One capture's shape, by name — the assertion these binaries make most often, said once so that
/// a call site is the tensor's name next to the widths it is derived from rather than five lines of
/// macro. Where a mismatch means something a shape diff cannot say, the caller keeps its own
/// `assert_eq!` and its message.
pub fn shape_is(g: &GoldenSet, name: &str, want: &[usize]) {
    assert_eq!(shape_of(g, name), want, "{name}");
}
