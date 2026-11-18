//! The codec family: every scalar codec and per-element helper EXHAUSTIVELY over its input
//! domain (2^32 f32 bit patterns where the input is an f32), in chunks, compared on the device.
//! Groups, inputs and parameter lists are 0b's.

use super::{Args, BLOCK, Dev, Group, Out, Outs, Probes, Table, blocks};

/// Inputs per launch: 0b's chunk. The two output buffers hold one chunk of the widest output
/// (u64), 512 MiB each — 1 GiB of device memory for the family, against this box's ~100 GiB
/// GTT, in exchange for 64 launches per exhaustive sweep instead of thousands.
const CHUNK: u64 = 1 << 26;
const ALL: u64 = 1 << 32;

/// One sweep: the kernel, its output type, and how many inputs (`0..n`).
type Spec<'a> = (&'a str, Out, u64);

/// The family's context: the probes, the table, and the two sweep buffers, allocated once.
struct Sweeper<'a> {
    p: &'a Probes,
    t: &'a mut Table,
    bufs: (Dev, Dev),
}

impl Sweeper<'_> {
    /// Sweep `k` over inputs `0..n` as row `k` + `label`; `extra` appends the kernel's
    /// trailing scalar parameters.
    fn sweep(&mut self, (k, out, n): Spec, label: &str, extra: impl Fn(Args) -> Args) {
        let mut g = Group::default();
        let esz = out.size();
        let (a, b) = (&self.bufs.0, &self.bufs.1);
        let mut base = 0u64;
        while base < n {
            let cnt = CHUNK.min(n - base) as u32;
            let bytes = cnt as usize * esz;
            let geom = blocks(cnt.div_ceil(BLOCK));
            let args = |o| extra(Args::default().p(base as u32).p(cnt).p(o));
            let o = Outs {
                a,
                b,
                bytes,
                ty: out,
            };
            let c = self.p.cell(((k, k), geom), args, o);
            g.cell(&c, bytes, || {
                format!("input=0x{:08x}", base + c.first / esz as u64)
            });
            base += CHUNK;
        }
        self.t.push(format!("{k}{label}"), g);
    }
}

pub fn codecs(p: &Probes, t: &mut Table) {
    let bufs = (
        Dev::zeroed(CHUNK as usize * 8),
        Dev::zeroed(CHUNK as usize * 8),
    );
    let mut s = Sweeper { p, t, bufs };
    plain(&mut s);
    parameterised(&mut s);
    situ(&mut s);
    lut(&mut s);
}

/// The one-argument codecs.
fn plain(s: &mut Sweeper) {
    let specs: [Spec; 11] = [
        ("ex_bf16f", Out::F32, 1 << 16),
        ("ex_e4m3f", Out::F32, 256),
        ("ex_e8m0f", Out::F32, 256),
        ("ex_e2m1f", Out::F32, ALL),
        ("ex_rbf16", Out::F32, ALL),
        ("ex_f2bf16", Out::U16, ALL),
        ("ex_f2e4m3", Out::U8, ALL),
        ("ex_f2e2m1", Out::U32, ALL),
        ("ex_moe_fixed", Out::U64, ALL),
        ("ex_blk_shift", Out::U32, ALL),
        ("ex_nquad", Out::U32, ALL),
    ];
    for spec in specs {
        s.sweep(spec, "", |a| a);
    }
}

/// The helpers with one scalar parameter, at each value 0b chose.
fn parameterised(s: &mut Sweeper) {
    for k in 0..8i32 {
        s.sweep(("ex_nib", Out::F32, ALL), &format!(" k={k}"), |a| a.p(k));
    }
    for mi in [1.0f32 / 448.0, 1.0 / 6.0, 1.0] {
        s.sweep(("ex_frs", Out::F32, ALL), &format!(" max_inv={mi}"), |a| {
            a.p(mi)
        });
    }
    // Scales: the power-of-two domain fast_round_scale produces, then adversarial ones.
    let scales = [
        1.0f32,
        0.000_976_562_5,
        1_048_576.0,
        0.3,
        1e-40,
        0.0,
        -0.0,
        f32::INFINITY,
        f32::NAN,
        -2.0,
    ];
    for sc in scales {
        for k in ["ex_aq8", "ex_aq4"] {
            s.sweep((k, Out::F32, ALL), &format!(" s={sc}"), |a| a.p(sc));
        }
    }
    // bf16 x bf16 is swiglu_clamped's whole domain for a given limit; 10 is V4's.
    for lim in [10.0f32, 7.0, 0.3, 1e30] {
        s.sweep(
            ("ex_swiglu", Out::F32, ALL),
            &format!(" limit={lim}"),
            |a| a.p(lim),
        );
    }
}

/// SiTU-GLU over each operand's whole domain, against fixed values of the other operand and
/// K3's betas, then a pair that is not a power of two.
fn situ(s: &mut Sweeper) {
    let others = [
        0.0f32,
        -0.0,
        1.0,
        -3.7,
        25.0,
        100.0,
        1e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        1e-40,
    ];
    for (b1, b2) in [(4.0f32, 25.0f32), (1.0, 1.0)] {
        for o in others {
            let label = format!(" other={o} b={b1},{b2}");
            for k in ["ex_situ_g", "ex_situ_u"] {
                s.sweep((k, Out::F32, ALL), &label, |a| a.p(o).p(b1).p(b2));
            }
        }
    }
}

/// The e4m3 LUT the fp8 GEMVs build — 256 threads writing straight to global. Judged as
/// `u32` words, strictly, as 0b did: the table's NaN entries are codes to reproduce exactly.
fn lut(s: &mut Sweeper) {
    let mut g = Group::default();
    let (a, b) = (&s.bufs.0, &s.bufs.1);
    let o = Outs {
        a,
        b,
        bytes: 1024,
        ty: Out::U32,
    };
    let c =
        s.p.cell((("k_lut", "k_lut"), blocks(1)), |o| Args::default().p(o), o);
    g.cell(&c, 1024, || "lut".into());
    s.t.push("k_lut".into(), g);
}
