//! inkling's single-query sink attention on the device, scored against the CPU oracle
//! `rivoli_oracles::inkling::sink_attend` — **no C twin** (owner direction 2026-11-24). The QK dots
//! and the weight·V sum are reductions whose accumulation order differs from the oracle's naive
//! loops, so the comparison is RELATIVE (the clip family's convention), not bitwise.
//!
//! **Tolerance provenance.** The anchor carves the fp32 floor per operator
//! (`docs/measurement/inkling2-reference/anchor.md` §Tolerances) and names attention softmax as one
//! of the floors the S4 kernel suite still owes there. `1e-4` relative is the allowance for the
//! block reductions re-associating the oracle's sequential folds (the max tree and the LDS sum both
//! reorder a `n`-term, then a `d`-term sum).
//!
//! The census waiver `ORACLE_VALIDATED` (`kernel_coverage.rs`) names this suite as the kernel's oracle.
//!
//! Device arm: `flock /var/run/sys-gpu.lock` with `--test-threads=1` (see CLAUDE.md).
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{assert_guard, assert_rel, attn_scale, dev, f32b, fill, launch_to_f32, ok};
use rivoli_backend::hip::launch_sink_attend;
use rivoli_oracles::inkling::sink_attend as cpu_sink_attend;

/// The launch geometry: `heads` query heads over `kv_heads` cached heads, each of `n` keys inside
/// a stride of `kv_stride` rows of width `d` (`kv_stride >= n`; the window is the head's first `n`
/// rows).
#[derive(Clone, Copy)]
struct Geom {
    n: usize,
    d: usize,
    heads: usize,
    kv_heads: usize,
    kv_stride: usize,
}

impl Geom {
    /// A single KV head (`kv_heads == 1`) over a window that is the whole stride — the degenerate
    /// layout the engine uses for a non-GQA layer.
    fn flat(n: usize, d: usize, heads: usize) -> Self {
        Self {
            n,
            d,
            heads,
            kv_heads: 1,
            kv_stride: n,
        }
    }
}

/// One batched case: `q` is `[heads, d]`, `bias` `[heads, n]`, `sink` `[heads]` (a DEVICE buffer in
/// the launcher's API), and `k`/`v` the `[kv_heads, kv_stride, d]` per-head cache.
#[derive(Clone, Copy)]
struct SinkAttendCase<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    bias: &'a [f32],
    sink: &'a [f32],
    scaling: f32,
    geom: Geom,
}

impl<'a> SinkAttendCase<'a> {
    /// The four staged operands arrive as one tuple so the constructor stays at four arguments.
    fn new(
        qkvb: (&'a [f32], &'a [f32], &'a [f32], &'a [f32]),
        sink: &'a [f32],
        geom: Geom,
        scaling: f32,
    ) -> Self {
        let (q, k, v, bias) = qkvb;
        Self {
            q,
            k,
            v,
            bias,
            sink,
            scaling,
            geom,
        }
    }

    /// The `kvh` a query head maps to — `h / (heads / kv_heads)`, the kernel's one division.
    fn kvh(&self, h: usize) -> usize {
        h / (self.geom.heads / self.geom.kv_heads)
    }

    /// Head `h`'s cached slice, entered at the head's own stride: `n` rows of `d`.
    fn cache(&self, h: usize) -> (usize, usize) {
        let g = self.geom;
        let base = self.kvh(h) * g.kv_stride * g.d;
        (base, base + g.n * g.d)
    }
}

/// The device draw for one geometry: `q`, `k`, `v`, `bias`, salted so the four do not coincide.
fn operands(g: Geom, seed: u64) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        fill(g.heads * g.d, seed, 1.0),
        fill(g.kv_heads * g.kv_stride * g.d, seed + 1, 1.0),
        fill(g.kv_heads * g.kv_stride * g.d, seed + 2, 1.0),
        fill(g.heads * g.n, seed + 3, 1.0),
    )
}

/// One geometry's device launch, returning the launcher's own `Result` so the guard tests exercise
/// the same entry point.
fn sink_attend_launch(c: &SinkAttendCase) -> anyhow::Result<Vec<f32>> {
    let g = c.geom;
    let qb = dev(&f32b(c.q));
    let kb = dev(&f32b(c.k));
    let vb = dev(&f32b(c.v));
    let bb = dev(&f32b(c.bias));
    let sb = dev(&f32b(c.sink));
    // `.max(1)` so a `heads = 0` guard arm still names a real destination.
    launch_to_f32((g.heads * g.d).max(1), |op, st| unsafe {
        launch_sink_attend(
            qb.ptr() as *const f32,
            kb.ptr() as *const f32,
            vb.ptr() as *const f32,
            bb.ptr() as *const f32,
            sb.ptr() as *const f32,
            c.scaling,
            g.n,
            g.d,
            g.heads,
            g.kv_heads,
            g.kv_stride,
            op,
            st,
        )
    })
}

/// The per-head oracle of a batched case, head `h` reading its own KV head's window — so a
/// transposed head map (every query head on KV head 0) cannot pass.
fn oracle(c: &SinkAttendCase) -> Vec<f32> {
    let g = c.geom;
    (0..g.heads)
        .flat_map(|h| {
            let (a, b) = c.cache(h);
            cpu_sink_attend(
                (
                    &c.q[h * g.d..(h + 1) * g.d],
                    &c.k[a..b],
                    &c.v[a..b],
                    &c.bias[h * g.n..(h + 1) * g.n],
                ),
                c.sink[h],
                c.scaling,
            )
        })
        .collect()
}

/// The SINK-LESS softmax for head `h`, spelled here rather than borrowed from the sink oracle — so a
/// shared mistake in the `−∞` sink path cannot make both sides agree on the wrong answer.
fn sinkless(c: &SinkAttendCase, h: usize) -> Vec<f32> {
    let g = c.geom;
    let (a, b) = c.cache(h);
    let q = &c.q[h * g.d..(h + 1) * g.d];
    let (k, v) = (&c.k[a..b], &c.v[a..b]);
    let bias = &c.bias[h * g.n..(h + 1) * g.n];
    let logits: Vec<f32> = (0..g.n)
        .map(|j| {
            let dot: f32 = (0..g.d).map(|i| q[i] * k[j * g.d + i]).sum();
            dot * c.scaling + bias[j]
        })
        .collect();
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let z: f32 = logits.iter().map(|&l| (l - m).exp()).sum();
    let w: Vec<f32> = logits.iter().map(|&l| (l - m).exp() / z).collect();
    (0..g.d)
        .map(|i| (0..g.n).map(|j| w[j] * v[j * g.d + i]).sum())
        .collect()
}

/// The decode step reproduces the CPU oracle across ragged key counts and head widths, single and
/// batched.
#[test]
fn sink_attend_matches_the_cpu_oracle() {
    let mut geoms = vec![
        Geom::flat(1, 4, 1),
        Geom::flat(33, 8, 1),
        Geom::flat(128, 16, 1),
        Geom::flat(257, 128, 1),
        Geom::flat(64, 3, 3),
    ];
    // GQA: 6 query heads over 2 cached KV heads, a stride wider than the window.
    geoms.push(Geom {
        n: 16,
        d: 128,
        heads: 6,
        kv_heads: 2,
        kv_stride: 20,
    });
    // A long context (the LDS budget's neighbourhood) over 8 cached heads.
    geoms.push(Geom {
        n: 1024,
        d: 128,
        heads: 8,
        kv_heads: 8,
        kv_stride: 1024,
    });
    for (i, g) in geoms.into_iter().enumerate() {
        let (q, k, v, bias) = operands(g, 0x1234 + i as u64);
        let sink = fill(g.heads, 0x9abc + i as u64, 0.5);
        let c = SinkAttendCase::new((&q, &k, &v, &bias), &sink, g, attn_scale(g.d));
        let got = ok(sink_attend_launch(&c), "gpu sink_attend");
        assert_rel(
            &oracle(&c),
            &got,
            &format!("sink_attend n={} d={} h={}", g.n, g.d, g.heads),
            1e-4,
        );
    }
}

/// T10: the sink absorbs probability mass. One key, logit 1, sink logit 0 → the kept weight is
/// `1/(1+e^-1) ≈ 0.731`, strictly below the sink-less softmax's 1.0.
#[test]
fn sink_attend_sink_absorbs_probability_mass() {
    let g = Geom::flat(1, 1, 1);
    let q = vec![1.0];
    let k = vec![1.0];
    let v = vec![1.0];
    let bias = vec![0.0];
    let sink = vec![0.0];
    let c = SinkAttendCase::new((&q, &k, &v, &bias), &sink, g, 1.0);
    let got = ok(sink_attend_launch(&c), "sink_attend T10");
    let want = 1.0 / (1.0 + (-1.0f32).exp());
    assert!(
        (got[0] - want).abs() < 1e-6,
        "T10: want {want}, got {}",
        got[0]
    );
    assert!(got[0] < 1.0, "T10: the sink must absorb mass");
}

/// The block-residual reuse (MOD:843–858): a `−∞` sink absorbs no mass, so `sink_attend` reduces to
/// the ordinary sink-less softmax — the block bank's attend. The GPU must match BOTH the CPU sink
/// oracle at the `−∞` edge (ocml `exp(−∞)=0`, `maxnum(−∞, x)=x`) AND, directly, a sink-less softmax
/// written here: agreeing with the oracle alone would leave a shared `−∞` defect invisible.
#[test]
fn sink_attend_negative_infinity_sink_is_sinkless() {
    let g = Geom {
        n: 16,
        d: 8,
        heads: 2,
        kv_heads: 1,
        kv_stride: 16,
    };
    let (q, k, v, bias) = operands(g, 0x2468);
    let sink = vec![f32::NEG_INFINITY; g.heads];
    let c = SinkAttendCase::new((&q, &k, &v, &bias), &sink, g, attn_scale(g.d));
    let got = ok(sink_attend_launch(&c), "sink_attend -inf");
    assert_rel(&oracle(&c), &got, "sink_attend sink=-inf is sinkless", 1e-4);
    for h in 0..g.heads {
        let want = sinkless(&c, h);
        assert_rel(
            &want,
            &got[h * g.d..(h + 1) * g.d],
            &format!("sink=-inf head {h} == an ordinary softmax"),
            1e-4,
        );
    }
}

/// The geometry refusals: a zero extent (1001), a non-divisible head map (1001), a window wider
/// than its stride (1001), a bad `scaling` (1006), and the LDS budget (1001).
#[test]
fn sink_attend_geometry_is_guarded() {
    let g = Geom::flat(4, 4, 1);
    let (q, k, v, bias) = operands(g, 1);
    let sink = vec![0.5f32; 2];
    let base = SinkAttendCase::new((&q, &k, &v, &bias), &sink, g, 1.0);
    let bad_geoms = [
        ("n = 0", Geom { n: 0, ..g }),
        ("d = 0", Geom { d: 0, ..g }),
        ("heads = 0", Geom { heads: 0, ..g }),
        ("kv_heads = 0", Geom { kv_heads: 0, ..g }),
        (
            "heads % kv_heads != 0",
            Geom {
                heads: 3,
                kv_heads: 2,
                ..g
            },
        ),
        ("kv_stride < n", Geom { kv_stride: 3, ..g }),
        // The budget is (n + ROW_BLOCK)·4 <= 65536 → n <= 16128.
        (
            "(n + ROW_BLOCK)·4 > 65536",
            Geom {
                n: 16129,
                kv_stride: 16129,
                ..g
            },
        ),
    ];
    for (why, geom) in bad_geoms {
        assert_guard(
            sink_attend_launch(&SinkAttendCase { geom, ..base }),
            Some(1001),
            why,
        );
    }
    for bad in [f32::NAN, -1.0, f32::INFINITY] {
        assert_guard(
            sink_attend_launch(&SinkAttendCase {
                scaling: bad,
                ..base
            }),
            Some(1006),
            &format!("scaling = {bad}"),
        );
    }
    assert_guard(sink_attend_launch(&base), None, "valid geometry");
}
