//! Readers shared by the tests that consume the K3 anchor goldens.
//!
//! `k3_anchor.rs` asks whether the vendored bytes are the ones the doc describes; `k3_attn_res.rs`
//! runs a kernel over them. Different questions, same three lines of lookup — and `build.rs`'s
//! jscpd gate rejected the second copy of `float` at 114 tokens the moment S2 item 1 wrote one.
//! Factored rather than exempted, which is this repo's standing answer.

#![allow(dead_code)] // each includer uses a subset; deadness here is a per-binary accident

use rivoli::v4oracle::golden::GoldenSet;

/// One float tensor's shape and values, by name.
///
/// Panics with the file's own contents, because "not found" is almost always a renamed capture and
/// the next question is always "then what IS in there".
pub fn float<'g>(g: &'g GoldenSet, name: &str) -> (&'g [usize], &'g [f32]) {
    g.floats
        .iter()
        .find(|(n, _, _)| n == name)
        .map(|(_, s, v)| (s.as_slice(), v.as_slice()))
        .unwrap_or_else(|| {
            let some: Vec<&String> = g.floats.iter().take(3).map(|(n, _, _)| n).collect();
            panic!(
                "{name} is not in the golden; it holds {} float tensors, e.g. {some:?}",
                g.floats.len()
            )
        })
}

pub fn shape_of(g: &GoldenSet, name: &str) -> Vec<usize> {
    float(g, name).0.to_vec()
}
