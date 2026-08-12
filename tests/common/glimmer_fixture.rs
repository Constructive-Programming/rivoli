//! **The two vendored Muse Glimmer text goldens, and the shape every S2 fixture reads them in.**
//!
//! Shared by `glimmer_attend.rs` and `glimmer_rope.rs` rather than copied: `build.rs`'s jscpd gate
//! rejects a second copy at 15 tokens, and — the reason that gate exists — two copies of "which
//! step carries how many query rows" would drift, and the one that drifted would still pass.
//!
//! Provenance is NOT re-checked here. `glimmer_anchor.rs` gates the bytes, their length and their
//! FNV; a second frozen copy of those numbers agreeing with the first is not a check.

// Included by `#[path]` into several test binaries, each of which uses PART of this module. Both
// lints below are per-binary accidents rather than statements about the module: a helper no
// binary happens to call is dead in that binary, and a re-export no binary happens to name is
// unused in it.
#![allow(dead_code, unused_imports)]

#[path = "golden_read.rs"]
mod golden_read;

// The device helpers, re-exported so a fixture needs ONE `#[path]` include and not three. Rust
// gives test binaries no way to share a module except by each spelling an include, so every
// spelling a fixture must repeat is a jscpd clone waiting — this preamble has now been rejected
// twice, and folding the includes into one is the fix that keeps working as fixtures are added.
#[path = "mod.rs"]
mod device;
pub use device::{back, dev, f32b, f32v, zeros};

#[path = "tolerance.rs"]
pub mod tolerance;
pub use golden_read::{float, shape_of};

use serde_json::Value;

/// One golden's int tensor, by name. Re-exported through here so a fixture needs one include.
pub fn ints_of<'g>(gold: &'g Golden, name: &str) -> &'g [i64] {
    golden_read::ints(&gold.g, name)
}

/// The same two text goldens `glimmer_anchor.rs` vendors, by the same bytes. Their provenance,
/// length and hash are gated there; this file only reads them, so re-checking would be two
/// frozen copies agreeing with each other rather than a check.
pub const TEXT: [(&str, &[u8]); 2] = [
    ("text-1", include_bytes!("../glimmer-anchor-text-1.bin")),
    ("text-2", include_bytes!("../glimmer-anchor-text-2.bin")),
];
pub struct Golden {
    pub name: &'static str,
    pub g: golden_read::GoldenSet,
    pub c: Value,
}
impl Golden {
    pub fn n(&self, key: &str) -> usize {
        self.c[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: {key} is not an integer in tiny_config", self.name))
            as usize
    }

    /// `(hq, hkv, head_dim)`, read together because no one of them means anything alone.
    pub fn dims(&self) -> (usize, usize, usize) {
        (
            self.n("num_attention_heads"),
            self.n("num_key_value_heads"),
            self.n("head_dim"),
        )
    }

    /// `(prompt_len, decode_steps)` — how many query rows each step carries.
    pub fn steps(&self) -> (usize, usize) {
        let get = |k: &str| {
            self.g
                .meta_get(k)
                .unwrap_or_else(|| panic!("{}: no {k} in metadata", self.name))
                .parse()
                .expect("a numeric metadata value")
        };
        (get("prompt_len"), get("decode_steps"))
    }

    /// Query rows and the absolute position of the first one, at step `t`.
    pub fn geometry(&self, t: usize) -> (usize, usize) {
        let (prompt, _) = self.steps();
        if t == 0 {
            (prompt, 0)
        } else {
            (1, prompt + t - 1)
        }
    }
}
pub fn goldens() -> Vec<Golden> {
    TEXT.iter()
        .map(|(name, bytes)| {
            let g = golden_read::GoldenSet::read_glimmer(&mut &bytes[..])
                .unwrap_or_else(|e| panic!("{name}: {e:#}"));
            let raw = g.meta_get("tiny_config").expect("tiny_config");
            let c = serde_json::from_str(raw).expect("tiny_config is JSON");
            Golden { name, g, c }
        })
        .collect()
}
/// `[1, heads, rows, dim]` as the reference lays it out -> `[rows][heads][dim]` as every kernel
/// in this tree does. The transpose IS the deviation, and `attn.rs` will name it at the call
/// site when S3 wires this up; doing it here keeps the kernel's layout the engine's.
pub fn to_engine(v: &[f32], heads: usize, rows: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * heads * dim];
    for h in 0..heads {
        for r in 0..rows {
            let src = (h * rows + r) * dim;
            let dst = (r * heads + h) * dim;
            out[dst..dst + dim].copy_from_slice(&v[src..src + dim]);
        }
    }
    out
}
/// One capture, in engine layout, with its shape asserted rather than assumed.
///
/// `transpose` says which of the two layouts the capture is in, and BOTH occur inside one
/// `eager_attention_forward`: `q`/`k_cache`/`v_cache` arrive heads-first `[1, heads, rows, dim]`
/// as the reference holds them, while `out` is captured after transformers' own
/// `attn_output.transpose(1, 2)` and is therefore already `[1, rows, heads, dim]` — the engine's
/// layout. Reading `out` as heads-first transposes a square-ish tensor into fluent nonsense, and
/// at the tiny widths (6 heads, 12 rows) it does not even fail a shape check by accident.
pub fn cap(gold: &Golden, name: &str, want: &[usize], transpose: bool) -> Vec<f32> {
    // `&[usize]` and not `[usize; 4]`: the captures are NOT all four-dimensional. `attend.*` is
    // `[1, heads, rows, dim]`, but `attn.gate_proj.out` and `attn.o_proj.in_gated` are
    // `[1, rows, heads*dim]` — three. The fixed-width version made a caller pad with a trailing 1
    // to typecheck, which asserted a shape the golden never had.
    assert!(
        !transpose || want.len() == 4,
        "{name}: only a 4-D capture has a layout to transpose"
    );
    let (shape, vals) = float(&gold.g, name);
    assert_eq!(shape, want, "{}: {name} shape", gold.name);
    if transpose {
        to_engine(vals, want[1], want[2], want[3])
    } else {
        vals.to_vec()
    }
}
/// How many rows a cache capture actually holds.
///
/// **Not the sequence length.** transformers' `DynamicSlidingWindowLayer` keeps only the last
/// `sliding_window - 1` rows and returns `cat(kept, new)`, so on a sliding layer at decode this
/// is `sliding_window` rows covering absolute positions `[pos - window + 1, pos]`, while a
/// global layer's is the whole prefix. The offset between the two is derived, never assumed:
/// `kv_offset = total - rows`, which is `get_mask_sizes`'s
/// `max(cumulative_length - sliding_window + 1, 0)` arrived at from the shape.
pub fn cap_rows(gold: &Golden, name: &str) -> usize {
    let s = shape_of(&gold.g, name);
    assert_eq!(
        s.len(),
        4,
        "{}: {name} is not a 4-D capture: {s:?}",
        gold.name
    );
    s[2]
}
/// Whether a capture is present at all.
///
/// `shape_of` cannot answer this: `float` PANICS on an absent name, deliberately — a missing
/// capture is nearly always a rename, and the next question is always "then what IS in there".
/// So the `shape_of(..).is_empty()` this file used to branch on was dead code twice over: an
/// absent mask would have aborted the test instead of skipping it, and a present one is never
/// empty.
pub fn present(gold: &Golden, name: &str) -> bool {
    gold.g.floats.iter().any(|(n, _, _)| n == name)
}
/// Every (golden, step, layer) the goldens carry, with the window of that layer.
pub fn each_case(mut f: impl FnMut(&Golden, usize, usize, usize)) {
    for gold in &goldens() {
        let (_, steps) = gold.steps();
        let sliding = golden_read::ints(&gold.g, "layer_is_sliding").to_vec();
        let win = gold.n("sliding_window");
        for t in 0..=steps {
            for (l, is_sliding) in sliding.iter().enumerate() {
                f(gold, t, l, if *is_sliding != 0 { win } else { 0 });
            }
        }
    }
}
/// How many cases `each_case` must produce and how many of those the ring test keeps — computed
/// from the goldens' metadata, because the failure this guards against is a loop that silently
/// covers less than it claims. Returns `(all, sliding-at-decode)`.
///
/// **The layer count and the step count are cross-checked against an INDEPENDENT source, and
/// that is the whole point.** The first version derived `all` from `layer_is_sliding.len()` and
/// compared it to a loop over the same tensor: an empty `layer_is_sliding` gave zero cases and
/// an expectation of zero, so `assert_eq!(0, 0)` passed while the suite tested nothing and
/// printed "worst abs over 0 cases". `num_hidden_layers` comes out of `tiny_config`, which the
/// driver writes separately, so the two can disagree — and a zero of either is now refused
/// outright rather than multiplied through.
pub fn expected() -> (usize, usize) {
    let (mut all, mut ring) = (0, 0);
    for gold in &goldens() {
        let (_, steps) = gold.steps();
        let sliding = golden_read::ints(&gold.g, "layer_is_sliding").to_vec();
        let layers = gold.n("num_hidden_layers");
        assert_eq!(
            sliding.len(),
            layers,
            "{}: layer_is_sliding has {} entries against num_hidden_layers {layers}",
            gold.name,
            sliding.len()
        );
        assert!(
            layers > 0 && steps > 0,
            "{}: {layers} layers, {steps} decode steps",
            gold.name
        );
        all += (steps + 1) * layers;
        ring += steps * sliding.iter().filter(|s| **s != 0).count();
    }
    (all, ring)
}

/// `max|got - want| / max|want|` — **the metric every Glimmer tolerance is stated in**, and the one
/// `glimmer_anchor_driver.py::by_operator` computes to produce the floors. Stated once, here,
/// because a fixture that scores against a row in a different metric is comparing two numbers that
/// are not the same quantity.
///
/// Scaled by the reference side's own magnitude, once per tensor, not per element: a per-element
/// ratio divides one rounding error by another wherever the reference is near zero.
pub fn worst_rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    let scale = want.iter().copied().fold(0.0f32, |m, w| m.max(w.abs()));
    // An all-zero reference has no scale to divide by; any difference is then infinitely relative,
    // and reporting infinity is more honest than dividing by an epsilon.
    if scale == 0.0 {
        return if got.iter().all(|g| *g == 0.0) {
            0.0
        } else {
            f32::INFINITY
        };
    }
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
        / scale
}

/// The `Rel` tolerance for one operator, or a panic saying which of the two ways it was absent.
///
/// Every S2 fixture asks for its row through here rather than matching on the policy itself:
/// three copies of that `match` was a jscpd clone, and — the reason the gate rejects it — three
/// copies would drift on what `ExactOnly` and `None` mean. They are NOT the same failure.
/// `ExactOnly` is a decision that the operator cannot be scored under a tolerance at all, and
/// none of these kernels can honour it (each re-associates or derives a transcendental, so none
/// will ever be bit-equal with torch). `None` means the row was renamed and the fixture is
/// scoring against nothing.
pub fn rel_tolerance(operator: &str) -> f32 {
    match tolerance::tolerance(tolerance::GLIMMER, operator) {
        Some(tolerance::Policy::Rel(t)) => *t,
        Some(tolerance::Policy::ExactOnly) => panic!(
            "{operator} is ExactOnly, which no S2 kernel can honour — it would have to be \
             bit-equal with torch. That is a decision about this operator and has to be read."
        ),
        None => panic!("tolerance::GLIMMER has no `{operator}` row, so nothing here is scored"),
    }
}

/// Join the device and read a buffer back as f32.
///
/// The three-line epilogue every launch helper in this port had ended with. jscpd rejected the
/// third copy, and it was right in the way that matters: `device_sync().unwrap()` is the line that
/// makes the read valid, and a helper that forgot it would return whatever was in the buffer
/// before the launch — which, in a test that seeds its output with zeros, reads as a kernel that
/// wrote nothing rather than as a missing barrier.
pub fn sync_read(b: &rivoli::memory::device::DeviceBuf) -> Vec<f32> {
    rivoli::backend::hip::device_sync().unwrap();
    f32v(&back(b))
}

// ---- device-op wrappers -------------------------------------------------------------------
//
// The S2 fixtures kept spelling these, and jscpd kept rejecting the next copy. They live here for
// the same reason `sync_read` does: a launch wrapper is three lines of pointer casts where a
// mistake is a wrong answer rather than a compile error, and one copy is one place to be right.

/// A cheap deterministic fill in `[-scale, scale)`. The period must not divide any width these
/// fixtures use, or a transposed or strided read lands on an equal value and passes.
pub fn fill(n: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let u = ((i.wrapping_mul(2_654_435_761).wrapping_add(salt * 40_503)) % 65_536) as f32;
            (u / 32_768.0 - 1.0) * scale
        })
        .collect()
}

/// bf16 truncation, matching what a bf16 checkpoint holds. A host reference MUST see these values
/// and not the f32 originals, or the comparison measures the fixture's rounding instead of the
/// kernel's arithmetic.
pub fn to_bf16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
}

pub fn from_bf16(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// `out[j] = Σ_i x[i] · w[j][i]` for one activation row, via `gemm_bf16`.
pub fn gemv_bf16(x: &[f32], w: &[u16], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(x.len(), k, "activation width");
    assert_eq!(w.len(), n * k, "weight elements");
    let xb = dev(&f32b(x));
    let wb = dev(&w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let ob = zeros(n * 4);
    // SAFETY: `x` is `k` live f32, `w` is `n*k` live u16, `out` is `n` writable f32, none aliasing
    // another, all live until the sync in `sync_read`. m = 1: one activation row.
    unsafe {
        rivoli::backend::hip::launch_gemm_bf16(
            xb.ptr() as *const f32,
            wb.ptr() as *const u16,
            ob.ptr() as *mut f32,
            1,
            n,
            k,
            std::ptr::null_mut(),
        )
    }
    .expect("gemm_bf16 launch");
    sync_read(&ob)
}

/// `h = silu(g) * u` via `swiglu`.
pub fn swiglu(g: &[f32], u: &[f32]) -> Vec<f32> {
    assert_eq!(g.len(), u.len(), "swiglu operands");
    let (gb, ub) = (dev(&f32b(g)), dev(&f32b(u)));
    let hb = zeros(g.len() * 4);
    // SAFETY: three distinct live buffers of `g.len()` f32 each, all outliving the sync below.
    unsafe {
        rivoli::backend::hip::launch_swiglu(
            gb.ptr() as *const f32,
            ub.ptr() as *const f32,
            g.len(),
            hb.ptr() as *mut f32,
            std::ptr::null_mut(),
        )
    }
    .expect("swiglu launch");
    sync_read(&hb)
}
