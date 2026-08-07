//! The on-disk golden format, and the comparison S2/S3 gate on.
//!
//! Deliberately trivial: a magic, a metadata block, then length-prefixed named tensors.
//! There is no version negotiation and no schema — a golden file is produced and consumed
//! by the same commit, and anything cleverer is a place for the gate to fail open.

use crate::v4oracle::forward::Capture;
use anyhow::{Context, Result, bail};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"RIVV4GLD";

/// The metadata key `v4-oracle emit` records its `--defect` under -- one constant, because
/// the writer (the bin) and the readers below must agree on the spelling or the check
/// silently degrades to "every file is legacy".
pub const DEFECT_KEY: &str = "defect";

pub struct GoldenSet {
    /// Free-form provenance: the config, the prompt, the commit. Written into the file so a
    /// golden can never be separated from what produced it.
    pub meta: Vec<(String, String)>,
    pub floats: Vec<(String, Vec<usize>, Vec<f32>)>,
    pub ints: Vec<(String, Vec<usize>, Vec<i64>)>,
}

impl GoldenSet {
    pub fn from_capture(meta: Vec<(String, String)>, cap: Capture) -> Self {
        Self {
            meta,
            floats: cap.floats,
            ints: cap.ints,
        }
    }

    pub fn write(&self, w: &mut impl Write) -> Result<()> {
        w.write_all(MAGIC)?;
        put_u64(w, self.meta.len() as u64)?;
        for (k, v) in &self.meta {
            put_str(w, k)?;
            put_str(w, v)?;
        }
        put_tensors(w, &self.floats, |x: &f32| x.to_le_bytes())?;
        put_tensors(w, &self.ints, |x: &i64| x.to_le_bytes())
    }

    pub fn read(r: &mut impl Read) -> Result<Self> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).context("reading golden magic")?;
        if &magic != MAGIC {
            bail!("not a rivoli V4 golden file");
        }
        let mut meta = Vec::new();
        for _ in 0..get_u64(r)? {
            meta.push((get_str(r)?, get_str(r)?));
        }
        let floats = get_tensors(r, f32::from_le_bytes)?;
        let ints = get_tensors(r, i64::from_le_bytes)?;
        Ok(Self { meta, floats, ints })
    }

    /// One metadata value by key. The single lookup every consumer goes through --
    /// [`GoldenSet::defect`], `tests/v4_loop.rs::meta`, `cmp`'s provenance check -- because
    /// three hand-rolled `.iter().find(...)` chains over the same field are a jscpd clone
    /// waiting to be typed (review, 2026-08-07).
    pub fn meta_get(&self, key: &str) -> Option<&str> {
        self.meta
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// The `Defect` name this set was emitted under.
    ///
    /// A file with no [`DEFECT_KEY`] predates the `--defect` flag (2026-08-07) and was
    /// therefore produced by a binary that could only run `Defect::None` -- absence IS
    /// `"None"`, not an unknown. Every emit since writes the key unconditionally,
    /// `"None"` included, so treating absence as `None` is not a fail-open going forward.
    pub fn defect(&self) -> &str {
        self.meta_get(DEFECT_KEY).unwrap_or("None")
    }

    /// Refuse a set emitted under any defect other than `expected`.
    ///
    /// The loader-side half of the `--defect` contract
    /// (`docs/investigations/real-weights-defect-goldens.md`): a perturbed golden that can
    /// be mistaken for a `Defect::None` one is worse than no perturbation feature at all.
    /// A gate that calls this before comparing fails AT LOAD on a mismatched file, instead
    /// of going numerically red in a way that reads as an engine defect.
    pub fn expect_defect(&self, expected: &str) -> Result<()> {
        let got = self.defect();
        if got == expected {
            Ok(())
        } else {
            bail!(
                "this golden set was emitted under --defect {got}, but the comparison \
                 expects {expected} -- refusing to score against a perturbed reference"
            )
        }
    }
}

/// Named, shaped tensors of one element type. Generic over the element width so the f32 and
/// i64 sections are one piece of code: two copies of a serializer is two chances to write a
/// length and read a different one.
type Tensors<T> = Vec<(String, Vec<usize>, Vec<T>)>;

fn put_tensors<const N: usize, T>(
    w: &mut impl Write,
    items: &Tensors<T>,
    enc: fn(&T) -> [u8; N],
) -> Result<()> {
    put_u64(w, items.len() as u64)?;
    for (n, shape, v) in items {
        put_str(w, n)?;
        put_u64(w, shape.len() as u64)?;
        for d in shape {
            put_u64(w, *d as u64)?;
        }
        put_u64(w, v.len() as u64)?;
        for x in v {
            w.write_all(&enc(x))?;
        }
    }
    Ok(())
}

fn get_tensors<const N: usize, T>(r: &mut impl Read, dec: fn([u8; N]) -> T) -> Result<Tensors<T>> {
    let mut out = Vec::new();
    for _ in 0..get_u64(r)? {
        let n = get_str(r)?;
        let shape: Vec<usize> = (0..get_u64(r)?)
            .map(|_| get_u64(r).map(|v| v as usize))
            .collect::<Result<_>>()?;
        let len = get_u64(r)? as usize;
        let mut v = Vec::with_capacity(len);
        let mut b = [0u8; N];
        for _ in 0..len {
            r.read_exact(&mut b)?;
            v.push(dec(b));
        }
        out.push((n, shape, v));
    }
    Ok(out)
}

/// How two captures differ, per tensor.
pub struct Diff {
    pub name: String,
    /// `max |a - b| / (max |b| + tiny)` over the tensor. Relative to the tensor's own scale
    /// because the activations here span several orders of magnitude between layers.
    pub rel: f32,
    /// Elements that are not bit-identical.
    pub changed: usize,
    pub total: usize,
}

/// The union of `a`'s and `b`'s tensor names, in `a`'s order then `b`'s extras.
fn union_names<T>(a: &Tensors<T>, b: &Tensors<T>) -> Vec<String> {
    let mut names: Vec<String> = a.iter().map(|(n, _, _)| n.clone()).collect();
    for (n, _, _) in b {
        if !names.contains(n) {
            names.push(n.clone());
        }
    }
    names
}

/// One tensor of `v` by name, with its declared shape.
fn find<'t, T>(v: &'t Tensors<T>, n: &str) -> Option<(&'t [usize], &'t [T])> {
    v.iter()
        .find(|(m, _, _)| m == n)
        .map(|(_, s, x)| (s.as_slice(), x.as_slice()))
}

/// Compare one section (all floats, or all ints), given how to score a matched pair.
///
/// A tensor present on one side only, or of a different shape, scores an INFINITE
/// difference rather than being skipped. A comparison that silently ignores what it cannot
/// line up fails OPEN, and a defect that deletes a golden would then read as agreement —
/// which is the failure this whole oracle exists to not have.
fn diff_section<T>(
    a: &Tensors<T>,
    b: &Tensors<T>,
    score: fn(&[T], &[T]) -> (f32, usize),
) -> Vec<Diff> {
    union_names(a, b)
        .into_iter()
        // SHAPE is compared, not just flat length: a golden reshaped from [s, h, d] to
        // [s, h*d] carries the same numbers and a different meaning, and letting that read
        // as agreement is the same fail-open as ignoring a missing tensor.
        .map(|n| match (find(a, &n), find(b, &n)) {
            (Some((sa, x)), Some((sb, y))) if sa == sb && x.len() == y.len() => {
                let (rel, changed) = score(x, y);
                Diff {
                    name: n,
                    rel,
                    changed,
                    total: x.len(),
                }
            }
            _ => Diff {
                name: n,
                rel: f32::INFINITY,
                changed: usize::MAX,
                total: 0,
            },
        })
        .collect()
}

/// Compare two captures tensor by tensor.
pub fn diff(a: &Capture, b: &Capture) -> Vec<Diff> {
    let mut out = diff_section(&a.floats, &b.floats, |x, y| {
        // Relative to the tensor's own scale: activations span several orders of magnitude
        // between layers, so one absolute epsilon would be both too tight and too loose.
        let scale = y.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-30);
        let changed = x
            .iter()
            .zip(y)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        // Non-finite pairs are skipped rather than folded: the indexer's masked scores are
        // `-inf` on both sides and `(-inf) - (-inf)` is NaN, which `f32::max` would silently
        // swallow. `changed` still counts them bit-for-bit, so nothing is lost.
        let rel = x
            .iter()
            .zip(y)
            .filter(|(p, q)| p.is_finite() && q.is_finite())
            .fold(0.0f32, |m, (p, q)| m.max((p - q).abs() / scale));
        (rel, changed)
    });
    // Selection goldens are exact-or-not; there is no "close" index.
    out.extend(diff_section(&a.ints, &b.ints, |x, y| {
        let changed = x.iter().zip(y).filter(|(p, q)| p != q).count();
        (if changed > 0 { f32::INFINITY } else { 0.0 }, changed)
    }));
    out
}

/// True iff every tensor is bit-identical AND the two captures name exactly the same
/// tensors with the same lengths.
///
/// A tensor missing on one side, or of a different length, arrives here as
/// `changed == usize::MAX` from [`diff`], so this single condition covers all three
/// failure modes. It deliberately does NOT also require a non-empty tensor: an empty
/// `.compress_idxs` is the correct golden for a layer whose compressor has not yet
/// produced a block, and rejecting it would make legitimate cases look like disagreement.
pub fn identical(a: &Capture, b: &Capture) -> bool {
    diff(a, b).iter().all(|d| d.changed == 0)
}

fn put_u64(w: &mut impl Write, v: u64) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
fn get_u64(r: &mut impl Read) -> Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn put_str(w: &mut impl Write, s: &str) -> Result<()> {
    put_u64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}
fn get_str(r: &mut impl Read) -> Result<String> {
    let n = get_u64(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8(b)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
mod tests {
    use super::*;

    /// A minimal set, pushed through the SAME write/read the gate uses -- an in-memory
    /// struct would not prove the header survives the file format.
    fn roundtrip(meta: Vec<(String, String)>) -> GoldenSet {
        let g = GoldenSet {
            meta,
            floats: vec![("t".into(), vec![2], vec![1.0, 2.0])],
            ints: vec![],
        };
        let mut buf = Vec::new();
        g.write(&mut buf).expect("write");
        GoldenSet::read(&mut buf.as_slice()).expect("read")
    }

    #[test]
    fn a_perturbed_golden_cannot_pass_as_none() {
        let g = roundtrip(vec![(DEFECT_KEY.into(), "RopeHalfSplit".into())]);
        assert_eq!(g.defect(), "RopeHalfSplit");
        let err = g
            .expect_defect("None")
            .expect_err("the mismatch must refuse");
        // Both names must appear: the message is the whole diagnosis of a stale env var
        // OR a stale file, and it cannot say which side is wrong.
        let msg = err.to_string();
        assert!(
            msg.contains("RopeHalfSplit") && msg.contains("None"),
            "{msg}"
        );
        g.expect_defect("RopeHalfSplit")
            .expect("the declared match must pass");
    }

    #[test]
    fn a_legacy_file_without_the_key_is_none() {
        // Pre-flag files were emitted by a binary that could only run `Defect::None`,
        // so absence is a safe `None` -- but only for files that really lack the key.
        let g = roundtrip(vec![("model".into(), "/x".into())]);
        assert_eq!(g.defect(), "None");
        g.expect_defect("None")
            .expect("legacy files must stay loadable");
        g.expect_defect("RopeHalfSplit")
            .expect_err("a legacy file must not satisfy a perturbed expectation");
    }
}
