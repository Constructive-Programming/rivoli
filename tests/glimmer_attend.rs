//! **`gqa_attend` on the device, scored against the S1b anchor.** S2 item 1 of
//! `docs/investigations/glimmer-port.md`.
//!
//! Grouped-query attention is the one new kernel family Glimmer needs
//! (`glimmer-architecture.md` §4): 32 Q heads over 2 KV heads, a 2048-row window on three of
//! every four layers, the whole causal prefix on the fourth. rivoli had no kernel with a
//! distinct V, no kernel with more than one KV head, and no kernel that derives its own causal
//! bound.
//!
//! # What this suite can and cannot promise
//!
//! Every comparison here is against `t*.L*.attend.out`, captured from the first-party stack
//! **after** `repeat_kv` and **before** the sigmoid gate and `o_proj`, with `attend.q` /
//! `attend.k_cache` / `attend.v_cache` captured as the kernel receives them — Q post-`qk_norm`,
//! post-`qk_scale_factor`, post-rope, and K/V pre-`repeat_kv`. So this gate covers the
//! broadcast, the causal bound, the window and the softmax, and covers **nothing** about how q
//! and k got that way. `qk_norm`, the 3.87 scale and rope are gated by the anchor's own
//! defect runs, not here.
//!
//! **The mask is not fed to the kernel, it is compared against it.** At Glimmer's 131072
//! context a `[tq][s]` mask array is larger than the model, so the kernel derives the bound
//! from `(start_pos, win)`; `the_derived_bound_reproduces_the_captured_mask` is what makes that
//! derivation a checked claim rather than a comment. Trap 14's off-by-one lives entirely in
//! that one test.
//!
//! **The tolerance is not bit-exactness and cannot be.** The kernel reduces the score with
//! `__shfl_down` in a ladder; torch sums sequentially. `MAX_ABS` is 10x the measured worst case
//! over every layer and step of both goldens, at the tiny widths — see its comment.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
#![cfg(feature = "rocm")]

use rivoli::backend::hip::{device_sync, launch_gqa_attend};
use serde_json::Value;

mod common;
use common::{back, dev, f32b, f32v, zeros};

// `GoldenSet` is spelled `golden_read::GoldenSet` below rather than imported: with it in this
// list, the include preamble matched `glimmer_anchor.rs`'s for 18 tokens and jscpd rejected the
// build. Two test binaries have no way to share a module except by each spelling the `#[path]`
// include, so the fix is to keep this list short, not to exempt the region.
#[path = "common/golden_read.rs"]
mod golden_read;
use golden_read::{float, ints, shape_of};

/// The same two text goldens `glimmer_anchor.rs` vendors, by the same bytes. Their provenance,
/// length and hash are gated there; this file only reads them, so re-checking would be two
/// frozen copies agreeing with each other rather than a check.
const TEXT: [(&str, &[u8]); 2] = [
    ("text-1", include_bytes!("glimmer-anchor-text-1.bin")),
    ("text-2", include_bytes!("glimmer-anchor-text-2.bin")),
];

/// The re-association floor, measured 2026-08-11 over both goldens, all 8 layers, all 7 steps:
///
/// | | worst absolute difference from the reference |
/// |---|---|
/// | this kernel, cache indexed by position | **6.56e-7** |
/// | this kernel, ring-indexed | 3.95e-7 |
/// | the host oracle in this file | 4.47e-7 |
///
/// The bar is 10x the largest of them. It is a floor, not a quality budget: the kernel reduces
/// each score with `__shfl_down` in a ladder while torch sums sequentially, so a difference of
/// this size is the summation order and nothing else. Every defect this gate exists to catch
/// moves the output by O(1) instead — see `WRONG_SIGNAL`.
const MAX_ABS: f32 = 6.6e-6;

/// What a wrong answer actually looks like here, so the two are never confused.
///
/// The smallest signal the interleaved KV broadcast produced over those same 112 cases was
/// **0.335** — five orders of magnitude above the floor above. This constant sits between them
/// at 1e-2: far enough above `MAX_ABS` that rounding can never reach it, far enough below the
/// measured signal that a defect cannot slip under it. A gate whose pass and fail differ by
/// 5 decades is worth more than a tight one, and this records that the gap was measured rather
/// than hoped for.
const WRONG_SIGNAL: f32 = 1e-2;

struct Golden {
    name: &'static str,
    g: golden_read::GoldenSet,
    c: Value,
}

impl Golden {
    fn n(&self, key: &str) -> usize {
        self.c[key]
            .as_u64()
            .unwrap_or_else(|| panic!("{}: {key} is not an integer in tiny_config", self.name))
            as usize
    }

    /// `(hq, hkv, head_dim)`, read together because no one of them means anything alone.
    fn dims(&self) -> (usize, usize, usize) {
        (
            self.n("num_attention_heads"),
            self.n("num_key_value_heads"),
            self.n("head_dim"),
        )
    }

    /// `(prompt_len, decode_steps)` — how many query rows each step carries.
    fn steps(&self) -> (usize, usize) {
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
    fn geometry(&self, t: usize) -> (usize, usize) {
        let (prompt, _) = self.steps();
        if t == 0 {
            (prompt, 0)
        } else {
            (1, prompt + t - 1)
        }
    }
}

fn goldens() -> Vec<Golden> {
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
fn to_engine(v: &[f32], heads: usize, rows: usize, dim: usize) -> Vec<f32> {
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
fn cap(gold: &Golden, name: &str, want: [usize; 4], transpose: bool) -> Vec<f32> {
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
fn cap_rows(gold: &Golden, name: &str) -> usize {
    let s = shape_of(&gold.g, name);
    assert_eq!(
        s.len(),
        4,
        "{}: {name} is not a 4-D capture: {s:?}",
        gold.name
    );
    s[2]
}

/// One attention call's inputs, in engine layout.
///
/// A struct because the device path and the host oracle take exactly the same six values, and
/// `build.rs`'s jscpd gate rejected the two signatures as a 79-token clone — correctly: five of
/// the six are `usize` or `&[f32]`, so any permutation type-checks and the failure is not a
/// panic but attention over the wrong rows. `v4compress.rs`'s `CompCase` is the same fix for the
/// same reason.
struct Case<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    /// `(hq, hkv, head_dim)`.
    dims: (usize, usize, usize),
    /// `(query rows, absolute position of row 0, window — 0 for a global layer)`.
    geom: (usize, usize, usize),
}

/// Launch and read back. `ring_cap` 0 is a cache indexed by position.
fn run(c: &Case, ring_cap: usize) -> Vec<f32> {
    let (hq, hkv, d) = c.dims;
    let (tq, start_pos, win) = c.geom;
    let (qb, kb, vb) = (dev(&f32b(c.q)), dev(&f32b(c.k)), dev(&f32b(c.v)));
    let ob = zeros(tq * hq * d * 4);
    // SAFETY: the three inputs are live device buffers of exactly the sizes the launcher's
    // safety comment requires, `ob` is writable and distinct from all three, and all four
    // outlive the `device_sync` below.
    unsafe {
        launch_gqa_attend(
            qb.ptr() as *const f32,
            kb.ptr() as *const f32,
            vb.ptr() as *const f32,
            hq,
            hkv,
            d,
            tq,
            start_pos,
            win,
            ring_cap,
            1.0 / (d as f32).sqrt(),
            ob.ptr() as *mut f32,
            std::ptr::null_mut(),
        )
    }
    .expect("gqa_attend launch");
    device_sync().unwrap();
    f32v(&back(&ob))
}

/// The window's lower bound for a query at absolute position `pos`: `[pos - win + 1, pos]`,
/// INCLUSIVE of `pos` itself, and 0 on a global layer. Trap 14 is the `+ 1`.
///
/// Shared by the host oracle and the mask comparison — jscpd caught them restating it, and it is
/// right to: two copies of a bound is how one of them drifts. The kernel has its own, in HIP,
/// which is the implementation this file exists to check.
fn window_lo(pos: usize, win: usize) -> usize {
    if win > 0 && pos >= win {
        pos - win + 1
    } else {
        0
    }
}

/// The reference attention, on the host, with the KV broadcast selectable.
///
/// `block` is the reference's own `repeat_kv` (`expand(b, hkv, group, s, d).reshape(...)`, so Q
/// head `i` reads KV head `i / group`); `false` is the interleave `i % hkv`, which is trap 10
/// and the only reason this function takes an argument at all. Both are attention, both are
/// fluent, and one of them is the model.
fn attend_host(c: &Case, block: bool) -> Vec<f32> {
    let (q, k, v) = (c.q, c.k, c.v);
    let (hq, hkv, d) = c.dims;
    let (tq, start_pos, win) = c.geom;
    let group = hq / hkv;
    let scale = 1.0 / (d as f32).sqrt();
    let mut out = vec![0.0; tq * hq * d];
    for row in 0..tq {
        let pos = start_pos + row;
        let lo = window_lo(pos, win);
        for h in 0..hq {
            let kvh = if block { h / group } else { h % hkv };
            let qrow = &q[(row * hq + h) * d..][..d];
            let logits: Vec<f32> = (lo..=pos)
                .map(|j| {
                    let kr = &k[(j * hkv + kvh) * d..][..d];
                    scale * (0..d).map(|i| qrow[i] * kr[i]).sum::<f32>()
                })
                .collect();
            let mx = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let w: Vec<f32> = logits.iter().map(|s| (s - mx).exp()).collect();
            let denom: f32 = w.iter().sum();
            let dst = &mut out[(row * hq + h) * d..][..d];
            for (j, wj) in (lo..=pos).zip(&w) {
                let vr = &v[(j * hkv + kvh) * d..][..d];
                for i in 0..d {
                    dst[i] += wj / denom * vr[i];
                }
            }
        }
    }
    out
}

/// Largest ABSOLUTE difference.
///
/// Not relative, and that is a decision rather than laziness: this output is a convex
/// combination of V rows, so every component is bounded by `max|V|` and an absolute bar means
/// the same thing everywhere in the tensor. A relative bar does not — a component of the
/// average can legitimately be ~0, and the ratio then divides one rounding error by another.
/// The first version of this file used `|g-w| / max(|w|, 1e-3)`, which is an absolute measure
/// wearing a relative name: every "3.7e-5 relative" it reported was a 3.7e-8 difference against
/// the 1e-3 floor.
fn worst_abs(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

/// Every (golden, step, layer) the goldens carry, with the window of that layer.
fn each_case(mut f: impl FnMut(&Golden, usize, usize, usize)) {
    for gold in &goldens() {
        let (_, steps) = gold.steps();
        let sliding = ints(&gold.g, "layer_is_sliding").to_vec();
        let win = gold.n("sliding_window");
        for t in 0..=steps {
            for (l, is_sliding) in sliding.iter().enumerate() {
                f(gold, t, l, if *is_sliding != 0 { win } else { 0 });
            }
        }
    }
}

/// Everything one (step, layer) comparison needs, with the cache's geometry resolved from the
/// capture instead of predicted.
struct Fixture {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    want: Vec<f32>,
    dims: (usize, usize, usize),
    /// `(tq, start_pos RELATIVE to the cache's first row, win)` — the kernel's own numbering.
    /// The kernel indexes rows from 0, so handing it the absolute position of a sliding
    /// layer's query would send it past the end of a `window`-row cache.
    geom: (usize, usize, usize),
    /// Absolute position held by cache row 0.
    kv_offset: usize,
}

fn fixture(gold: &Golden, t: usize, l: usize, win: usize) -> Fixture {
    let (hq, hkv, d) = gold.dims();
    let (tq, start_pos) = gold.geometry(t);
    let p = format!("t{t}.L{l}");
    let rows = cap_rows(gold, &format!("{p}.attend.k_cache"));
    let kv_offset = (start_pos + tq).checked_sub(rows).unwrap_or_else(|| {
        panic!(
            "{}: {p} cache holds {rows} rows past the sequence",
            gold.name
        )
    });
    Fixture {
        q: cap(gold, &format!("{p}.attend.q"), [1, hq, tq, d], true),
        k: cap(
            gold,
            &format!("{p}.attend.k_cache"),
            [1, hkv, rows, d],
            true,
        ),
        v: cap(
            gold,
            &format!("{p}.attend.v_cache"),
            [1, hkv, rows, d],
            true,
        ),
        want: cap(gold, &format!("{p}.attend.out"), [1, tq, hq, d], false),
        dims: (hq, hkv, d),
        geom: (tq, start_pos - kv_offset, win),
        kv_offset,
    }
}

impl Fixture {
    fn case(&self) -> Case<'_> {
        Case {
            q: &self.q,
            k: &self.k,
            v: &self.v,
            dims: self.dims,
            geom: self.geom,
        }
    }
}

/// How many cases `each_case` must produce, and how many of those the ring test keeps —
/// computed from the goldens' own metadata rather than written down, because the failure this
/// guards against is a loop that silently covers less than it claims. Returns
/// `(all, sliding-at-decode)`.
fn expected() -> (usize, usize) {
    let (mut all, mut ring) = (0, 0);
    for gold in &goldens() {
        let (_, steps) = gold.steps();
        let sliding = ints(&gold.g, "layer_is_sliding").to_vec();
        all += (steps + 1) * sliding.len();
        ring += steps * sliding.iter().filter(|s| **s != 0).count();
    }
    (all, ring)
}

// ------------------------------------------------------------------------------------------

/// The kernel against the reference, over every layer and every step of both goldens.
///
/// This is 8 layers x 19 steps x 2 goldens = 304 comparisons, and it is deliberately not
/// narrowed to a representative few: a local layer, a global layer, layer 0, the last layer and
/// every window-boundary crossing are all in here by construction, which is exactly the
/// coverage G1b names. The tiny `sliding_window` of 4 against 18 decoded positions is what
/// makes the crossings dense rather than incidental.
#[test]
fn the_kernel_matches_the_anchor_at_every_layer_and_step() {
    let mut worst: f32 = 0.0;
    let mut cases = 0;
    each_case(|gold, t, l, win| {
        let f = fixture(gold, t, l, win);
        // `ring_cap` 0: the reference's cache is contiguous and indexed from its own row 0,
        // which `fixture` has already accounted for in `geom`.
        let got = run(&f.case(), 0);

        let r = worst_abs(&got, &f.want);
        assert!(
            r <= MAX_ABS,
            "{}: t{t}.L{l} (win {win}, geom {:?}, kv_offset {}) worst abs {r:e} > {MAX_ABS:e}",
            gold.name,
            f.geom,
            f.kv_offset
        );
        worst = worst.max(r);
        cases += 1;
    });
    // Anti-vacuity: an empty `layer_is_sliding` or a metadata rename would make the loop above
    // pass over nothing at all.
    println!("kernel: worst abs over {cases} cases: {worst:e}");
    assert_eq!(
        cases,
        expected().0,
        "the comparison loop did not cover every step and layer"
    );
}

/// The bound the kernel DERIVES is the mask the reference BUILT.
///
/// This is the reason the kernel takes no mask. The reference's mask is captured clamped to
/// 1.0/0.0 (its additive form carries the dtype minimum, which survives no round trip); what is
/// compared is which positions are attended.
///
/// **It does not guard the kernel, and the pair is what covers trap 14.** The bound is restated
/// here in Rust, so breaking `lo` in the kernel leaves this test green — measured, by doing
/// exactly that: `pos - win` in place of `pos - win + 1` reddens
/// `the_kernel_matches_the_anchor_at_every_layer_and_step` at 1.07 absolute and this test not at
/// all. This half proves the RULE is the reference's; that half proves the kernel implements
/// this rule. Neither alone is trap 14.
#[test]
fn the_derived_bound_reproduces_the_captured_mask() {
    let mut checked = 0;
    each_case(|gold, t, l, win| {
        let (tq, start_pos) = gold.geometry(t);
        let name = format!("t{t}.L{l}.attend.mask");
        // A global layer at t=0 may carry no mask at all — a full causal prefill needs none.
        if shape_of(&gold.g, &name).is_empty() {
            return;
        }
        let (ms, mv) = float(&gold.g, &name);
        // The mask spans the CACHE, not the sequence, and on a sliding layer at decode those
        // differ — `get_mask_sizes` returns `kv_length = window - 1 + query_length`. Column `j`
        // is therefore absolute position `kv_offset + j`, and the kernel's own numbering starts
        // at the same place.
        let cols = ms[3];
        let kv_offset = (start_pos + tq) - cols;
        assert_eq!(ms, [1, 1, tq, cols], "{}: {name} shape", gold.name);
        for row in 0..tq {
            let pos = start_pos + row - kv_offset;
            let lo = window_lo(pos, win);
            for j in 0..cols {
                let want = mv[row * cols + j] > 0.5;
                let got = j >= lo && j <= pos;
                assert_eq!(
                    got, want,
                    "{}: {name} row {row} (pos {pos}, win {win}) column {j}: derived {got}, \
                     reference {want}",
                    gold.name
                );
            }
        }
        checked += 1;
    });
    assert!(checked >= 100, "only {checked} masks compared");
}

/// **The red proof for trap 10.** `i / group` and `i % hkv` are both attention over the same
/// tensors; if the goldens could not tell them apart, the test above would be green on either.
///
/// Asserted on the HOST oracle rather than by breaking the kernel, because the kernel has one
/// mapping compiled into it — and the claim being proved is about the FIXTURE's power, not the
/// kernel's. The oracle matching the reference is what earns it the right to make the second
/// claim.
#[test]
fn the_goldens_separate_the_two_kv_broadcast_mappings() {
    let (mut separated, mut worst, mut sep_min) = (0, 0.0f32, f32::INFINITY);
    each_case(|gold, t, l, win| {
        let (hq, hkv, _) = gold.dims();
        assert!(
            hq / hkv != hkv,
            "the widths cannot separate the two mappings"
        );
        let f = fixture(gold, t, l, win);
        let right = attend_host(&f.case(), true);
        let wrong = attend_host(&f.case(), false);
        assert!(
            worst_abs(&right, &f.want) <= MAX_ABS,
            "{}: t{t}.L{l} the host oracle does not reproduce the reference, so it cannot prove \
             anything about the wrong mapping",
            gold.name
        );
        assert!(
            worst_abs(&wrong, &f.want) > WRONG_SIGNAL,
            "{}: t{t}.L{l} the interleaved broadcast produced the SAME output as the block one \
             — this fixture is blind to trap 10",
            gold.name
        );
        worst = worst.max(worst_abs(&right, &f.want));
        sep_min = sep_min.min(worst_abs(&wrong, &f.want));
        separated += 1;
    });
    println!("oracle: worst abs {worst:e}; smallest wrong-mapping signal {sep_min:e}");
    assert_eq!(
        separated,
        expected().0,
        "the separation loop did not cover every step and layer"
    );
}

/// A ring cache and a linear one attend the same rows.
///
/// The second of the kernel's "two row-sources". At real scale a local layer holds 2048 slots
/// and position `p` lives at `p % 2048`, so the ring is what makes a 131072-token context cost
/// 2048 rows — and an off-by-one in the mapping reads a row that is 2048 positions stale while
/// every shape stays right. Built by rotating the reference's own cache into ring order, so the
/// two runs differ in nothing but the indexing.
#[test]
fn a_ring_cache_attends_the_same_rows_as_a_linear_one() {
    let (mut ran, mut worst) = (0, 0.0f32);
    each_case(|gold, t, l, win| {
        // Only meaningful on a sliding layer, and only once the ring has wrapped.
        let (tq, start_pos) = gold.geometry(t);
        let s_len = start_pos + tq;
        if win == 0 || s_len <= win || tq != 1 {
            return;
        }
        let (_, hkv, d) = gold.dims();
        let f = fixture(gold, t, l, win);
        let rows = f.k.len() / (hkv * d);

        // Scatter the reference's contiguous window into ring slots: absolute position `j` lives
        // at `j % win`, which is where the engine's cache writer will put it. The linear run in
        // the test above reads the same values in cache order, so a difference between the two
        // is the modulo mapping and nothing else.
        let stride = hkv * d;
        let (mut kr, mut vr) = (vec![0.0; win * stride], vec![0.0; win * stride]);
        for i in 0..rows {
            let slot = (f.kv_offset + i) % win;
            kr[slot * stride..][..stride].copy_from_slice(&f.k[i * stride..][..stride]);
            vr[slot * stride..][..stride].copy_from_slice(&f.v[i * stride..][..stride]);
        }

        // Absolute positions this time — that is what a ring is indexed by.
        let geom = (tq, f.geom.1 + f.kv_offset, win);
        let case = Case {
            q: &f.q,
            k: &kr,
            v: &vr,
            dims: f.dims,
            geom,
        };
        let got = run(&case, win);
        let r = worst_abs(&got, &f.want);
        assert!(
            r <= MAX_ABS,
            "{}: t{t}.L{l} ring worst abs {r:e}",
            gold.name
        );
        worst = worst.max(r);
        ran += 1;
    });
    assert_eq!(
        ran,
        expected().1,
        "the ring loop did not cover every sliding layer at decode"
    );
    println!("ring: worst abs over {ran} cases: {worst:e}");
}

/// One row of the guard table: `(hq, hkv, head_dim, win, ring_cap, expected code, what)`, where
/// the code is `None` for a call that must be ACCEPTED.
///
/// An alias rather than a bare tuple because clippy rejects the literal type as too complex, and
/// rather than a struct because a struct literal per row made six near-identical field lists that
/// jscpd then rejected as clones. Both gates are right; a named tuple is what satisfies them.
type GuardCase = (usize, usize, usize, usize, usize, Option<i32>, &'static str);

/// Each argument guard rejects before any launch, so a bad call is an error code and not a
/// fault in someone else's kernel three launches later.
///
/// `win == 0` with a ring is the one worth spelling out: a global layer holding a ring would
/// attend the last `ring_cap` positions and silently drop everything older, which is fluent,
/// wrong, and permanent. The launcher refuses rather than choosing one of the two meanings.
#[test]
fn the_launcher_refuses_what_it_cannot_compute() {
    let b = zeros(4096);
    // Named `scratch_out` rather than the `pm` every other suite uses: after rustfmt broke the
    // call below into one argument per line, its tail matched `f4_attn.rs`'s launcher-guard tail
    // token for token and jscpd rejected the build. The names are the only thing that differed.
    let (p, scratch_out) = (b.ptr() as *const f32, b.ptr() as *mut f32);
    // `want` is asserted as a CODE, not merely as "an error": a launch that failed for some
    // other reason would satisfy `is_err` and leave the guard it claims to exercise untested.
    let cases: [GuardCase; 6] = [
        (0, 2, 8, 4, 0, Some(1001), "zero Q heads"),
        (7, 2, 8, 4, 0, Some(1003), "hq not a multiple of hkv"),
        (6, 2, 512, 4, 0, Some(1002), "head_dim past the accumulator"),
        (6, 2, 8, 0, 4, Some(1005), "a ring on a global layer"),
        (6, 2, 8, 8, 4, Some(1005), "a ring shorter than the window"),
        (
            6,
            2,
            8,
            4,
            4,
            None,
            "a ring exactly the window, which is legal",
        ),
    ];
    for (hq, hkv, d, win, ring_cap, want, what) in cases {
        // SAFETY: the five rejected calls return before `hipLaunchKernelGGL`, so no pointer is
        // dereferenced. The sixth does launch, at 6 heads x 1 row x head_dim 8 — 48 f32 in and
        // 48 out, well inside the 4096-byte buffer every argument points into. It is nonsense
        // arithmetic over aliased inputs, which is fine: this test reads the return code.
        let got = unsafe {
            launch_gqa_attend(
                p,
                p,
                p,
                hq,
                hkv,
                d,
                1,
                0,
                win,
                ring_cap,
                1.0,
                scratch_out,
                std::ptr::null_mut(),
            )
        };
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
    device_sync().unwrap();
}
