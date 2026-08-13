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
pub use device::{
    FixtureTensor, GLIMMER_SHIPPED_CONFIG, TempRoot, back, decode_one, dev, f32b, f32v,
    gemm_bf16_launch, rms_inv, run_convert_glimmer, u16b, weightless, window_lo, worst_rel,
    write_glimmer_aux, write_glimmer_config, write_index, write_safetensors, zeros,
};

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

/// The tiny model's `attn.gate_proj` weight per layer, one file per salt, in the SAME container
/// format as the goldens (`--dump-weights`, `glimmer_anchor_driver.py::weights_capture`).
///
/// **Why this exists at all, since the anchor's whole design is activations-only.** S3 item 3
/// scores the gate OPERAND against `attn.gate_proj.out`, and computing `gate_proj` needs the
/// projection. It is NOT recoverable from the captures: `gate_proj` is 72 -> 48 and a layer sees
/// 18 rows, so 18 equations against 72 unknowns per output element is underdetermined by 4x and
/// EVERY candidate operand admits a weight that fits the captures exactly. The recover-and-predict
/// shape the sandwich norms use works there only because a norm is elementwise; here it would be
/// vacuous rather than weak, which is the harder failure to notice.
///
/// **Separate files, so the goldens did not move.** Adding these to the goldens would change their
/// bytes and all four FNVs pinned in `glimmer_anchor.rs`. Instead `--dump-weights` adds nothing to
/// the capture set, and that was verified the only way it can be: both goldens were regenerated
/// with the flag on and came back BYTE-IDENTICAL to the vendored ones (2026-08-13).
pub const WEIGHTS: [(&str, &[u8]); 2] = [
    ("text-1", include_bytes!("../glimmer-anchor-weights-1.bin")),
    ("text-2", include_bytes!("../glimmer-anchor-weights-2.bin")),
];

/// The weight sets, in the same order and under the same names as [`goldens`], so a caller can zip
/// them. Panics rather than skipping if a file is unreadable: a missing weight set must not
/// degrade item 3's gate into a pass over nothing.
pub fn weight_sets() -> Vec<(&'static str, golden_read::GoldenSet)> {
    WEIGHTS
        .iter()
        .map(|(name, bytes)| {
            let g = golden_read::GoldenSet::read_glimmer(&mut &bytes[..])
                .unwrap_or_else(|e| panic!("{name} weights: {e:#}"));
            (*name, g)
        })
        .collect()
}
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
/// `(layers, steps)` for one golden, with the two refusals that stop a census multiplying zero.
///
/// Both matter and they are different failures. A zero of either makes an expected count of zero
/// against a loop that scored nothing, so `assert_eq!(0, 0)` passes while the suite reports a worst
/// over nothing — this repo hit that once. And `num_hidden_layers` comes from `tiny_config` while
/// `layer_is_sliding` is written separately by the driver, so the two can DISAGREE; cross-checking
/// them is what makes a layer count independent of the loop that consumes it, which arithmetic over
/// the same value never is.
///
/// Extracted 2026-08-12: the chain census in `glimmer_norm.rs` needed the same two refusals, wrote
/// them out, and jscpd rejected the copy — the gate landing on the exact lines a review had just
/// asked to be shared.
pub fn census_dims(gold: &Golden) -> (usize, usize) {
    let (_, steps) = gold.steps();
    let layers = gold.n("num_hidden_layers");
    let sliding = ints_of(gold, "layer_is_sliding").len();
    assert!(
        layers > 0 && steps > 0,
        "{}: {layers} layers, {steps} decode steps",
        gold.name
    );
    assert_eq!(
        sliding, layers,
        "{}: layer_is_sliding has {sliding} entries against num_hidden_layers {layers}",
        gold.name
    );
    (layers, steps)
}

pub fn expected() -> (usize, usize) {
    let (mut all, mut ring) = (0, 0);
    for gold in &goldens() {
        let (_, steps) = gold.steps();
        let (layers, _) = census_dims(gold);
        all += (steps + 1) * layers;
        ring += steps
            * ints_of(gold, "layer_is_sliding")
                .iter()
                .filter(|s| **s != 0)
                .count();
    }
    (all, ring)
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

/// A running score against one tolerance row: the bar, the worst seen, **which case produced it**,
/// and how many there were.
///
/// Two fixtures spelled the same four-line epilogue and jscpd rejected the second copy
/// (2026-08-12); the count lives with the score because a loop that forgets to increment prints a
/// worst over nothing, the vacuity `expected()` next door exists to catch.
///
/// **`at` is the reason this is a struct and not a free function.** Both callers finish with a bar
/// an order of magnitude tighter than the row (the row is a whole-bucket fp32 floor and cannot see a
/// two-decade regression), and that tighter bar is checked on the FOLD — so without a remembered
/// locator its failure names none of the 612 cases, and finding out which one meant re-instrumenting
/// a test that needs a shared GPU under a flock. Review finding, 2026-08-12.
///
/// `glimmer_attend.rs` and `glimmer_rope.rs` carry the same epilogue and were NOT migrated: jscpd
/// does not match them, so nothing forces it, and rewriting two green GPU fixtures to share a
/// three-field accumulator is not worth the run. Said here rather than left silent, because the next
/// reader will otherwise take two call sites for the tree's only two.
pub struct Scored {
    pub tol: f32,
    pub worst: f32,
    pub at: String,
    pub cases: usize,
    operator: &'static str,
}

impl Scored {
    pub fn new(operator: &'static str) -> Self {
        Self {
            tol: rel_tolerance(operator),
            worst: 0.0,
            at: "<nothing scored>".to_string(),
            cases: 0,
            operator,
        }
    }

    /// Score one case under the row's metric and refuse it over the row's tolerance.
    ///
    /// `what` names the case and runs only when this case ties or beats the worst so far — which is
    /// also the only state the assert below can fire in, since a strictly worse earlier case would
    /// already have failed. So the label is available where it is needed without formatting every
    /// green case, and `at` survives the loop for the caller's tighter bar.
    pub fn case(&mut self, got: &[f32], want: &[f32], what: impl FnOnce() -> String) {
        let r = worst_rel(got, want);
        if r >= self.worst {
            self.worst = r;
            self.at = what();
        }
        assert!(
            r <= self.tol,
            "{}: {} scores {r:e} > {:e}",
            self.operator,
            self.at,
            self.tol
        );
        self.cases += 1;
    }
}

/// One row of a launcher-guard table, judged: `Some(code)` must be exactly that guard code —
/// a launch failing for any OTHER reason would satisfy a bare `is_err` and leave the guard it
/// claims to exercise untested — and `None` must be accepted.
///
/// Here because the second guard-table fixture (`glimmer_head.rs`) copied the first's match
/// verbatim and jscpd rejected the build; a third table is inevitable and this is its one copy.
pub fn expect_guard(got: anyhow::Result<()>, want: Option<i32>, what: &str) {
    match want {
        Some(code) => {
            let e = got.expect_err(what).to_string();
            assert!(
                e.contains(&format!("argument guard rejected ({code})")),
                "{what}: expected guard {code}, got {e:?}"
            );
        }
        None => got.unwrap_or_else(|e| panic!("{what}: {e:#}")),
    }
}

/// Read a buffer back as f32 — one spelling of `f32v(&back(b))`, kept because jscpd rejected
/// the third copy of the epilogue.
///
/// The barrier lives in `back` itself, which opens with `device_sync`. This helper's first
/// draft added a second sync and justified the whole helper with it ("the line that makes the
/// read valid") — a comment claiming to enforce something the callee already enforced, this
/// repo's most-cited finding shape, caught by review 2026-08-12.
pub fn sync_read(b: &rivoli::memory::device::DeviceBuf) -> Vec<f32> {
    f32v(&back(b))
}

// ---- device-op wrappers -------------------------------------------------------------------
//
// The S2 fixtures kept spelling these, and jscpd kept rejecting the next copy. They live here for
// the same reason `sync_read` does: a launch wrapper is three lines of pointer casts where a
// mistake is a wrong answer rather than a compile error, and one copy is one place to be right.

/// The one place `launch_rmsnorm_centered_single` is spelled, returning its `Result`.
///
/// Two spellings — the wrapper below and `glimmer_norm.rs`'s guard table — were a jscpd clone,
/// which is the gate being right about the substance too: a guard test and a scoring test must
/// drive the SAME call, or the guard is proving something about a second launch nobody uses.
///
/// # Safety
/// `x` and `w` must be `n` readable f32 and `y` `n` writable f32, all live until the next
/// `device_sync` — except when the call is expected to be REFUSED, where the guard returns
/// before any launch and nothing is dereferenced.
pub unsafe fn rmsnorm_launch(
    x: *const f32,
    w: *const f32,
    n: usize,
    eps: f32,
    y: *mut f32,
) -> anyhow::Result<()> {
    // SAFETY: the caller's contract above. Null stream: every caller launches one kernel and
    // then joins, so there is nothing to order against.
    unsafe {
        rivoli::backend::hip::launch_rmsnorm_centered_single(x, w, n, eps, y, std::ptr::null_mut())
    }
}

/// One row through Muse Glimmer's CENTERED RMSNorm: `y = x · rsqrt(mean(x²)+eps) · (1 + w)`.
///
/// Here rather than in `glimmer_norm.rs` because jscpd matched the launch block against the
/// wrappers already in this file. It briefly took a `centered: bool` and dispatched to
/// `rmsnorm_single` for the plain form — deleted 2026-08-12 when review pointed out the test's
/// only plain comparison belongs on the HOST anyway, so there was never a second device form to
/// select and the bool was pure surface.
pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let (xb, wb) = (dev(&f32b(x)), dev(&f32b(w)));
    let y = zeros(x.len() * 4);
    // SAFETY: three distinct allocations of exactly `x.len()` live f32, `y` writable, all three
    // outliving the sync inside `sync_read`.
    unsafe {
        rmsnorm_launch(
            xb.ptr() as *const f32,
            wb.ptr() as *const f32,
            x.len(),
            eps,
            y.ptr() as *mut f32,
        )
    }
    .expect("rmsnorm_centered_single launch");
    sync_read(&y)
}

/// A cheap deterministic fill in `[-scale, scale)`. The period must not divide any width these
/// fixtures use, or a transposed or strided read lands on an equal value and passes.
///
/// **The mixing xor is load-bearing — the linear form failed its own contract (found
/// 2026-08-12).** As `(31153·i + salt·40503) mod 65536` the flat period was 65536, which
/// divides no width here, but a 2-D read cares about `gcd(row_stride, 65536)`: at stride 6656
/// (= 2^9·13) rows repeat every 65536/512 = **128 rows**, so a [19968 x 6656] weight held 128
/// distinct rows and a kernel reading row c+128 for row c passed bit-identically. Folding bits
/// 13..28 into the low 16 makes row equality need `(j-j')·stride ≡ 0 (mod 2^29)` — a period of
/// 2^20 rows at these strides, past any width in the tree.
pub fn fill(n: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = i
                .wrapping_mul(2_654_435_761)
                .wrapping_add(salt.wrapping_mul(40_503));
            let u = ((h ^ (h >> 13)) % 65_536) as f32;
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

/// `out[t][j] = Σ_i x[t][i] · w[j][i]` over `m` activation rows, via `gemm_bf16`.
pub fn gemm_bf16(x: &[f32], w: &[u16], m: usize, n: usize, k: usize) -> Vec<f32> {
    assert_eq!(x.len(), m * k, "activation elements");
    assert_eq!(w.len(), n * k, "weight elements");
    let xb = dev(&f32b(x));
    let wb = dev(&w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
    let ob = zeros(m * n * 4);
    // SAFETY: `x` is `m*k` live f32, `w` is `n*k` live u16, `out` is `m*n` writable f32, none
    // aliasing another, all live until the sync in `sync_read`.
    unsafe {
        device::gemm_bf16_launch(
            xb.ptr() as *const f32,
            wb.ptr() as *const u16,
            ob.ptr() as *mut f32,
            m,
            n,
            k,
            std::ptr::null_mut(),
        )
    };
    sync_read(&ob)
}

/// One activation row. Item 3 needs `m` = 12 (a whole prefill), so the body moved to [`gemm_bf16`]
/// and this delegates — a second copy of those twelve lines is what jscpd would have refused, and
/// the shared body is also the only way the two stay the same call.
pub fn gemv_bf16(x: &[f32], w: &[u16], n: usize, k: usize) -> Vec<f32> {
    gemm_bf16(x, w, 1, n, k)
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
