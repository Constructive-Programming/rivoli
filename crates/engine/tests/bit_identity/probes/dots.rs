//! The seeded families: the wave/block reductions and the four dot formats, every lane's
//! result stored. Shapes, seed classes and launch geometry are 0b's; the draws come from this gate's `Lcg`, so the cells are new and the CLASSES are the same.

use super::{Args, BLOCK, Dev, Geom, Group, Out, Outs, Probes, Table, blocks};
use crate::common::Lcg;
use crate::inputs::SEEDS;
use rivoli_backend::LdsBytes;

/// Seed classes: 0–5 of 8 normal draws with fixed edges, 6 wide magnitudes (2^±60, rounding
/// and overflow), 7 nasty (inf, NaN payloads, −0, subnormals, NaN codes in the weights).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Cls {
    Normal,
    Wide,
    Nasty,
}

fn cls(seed: u64) -> Cls {
    match seed % 8 {
        7 => Cls::Nasty,
        6 => Cls::Wide,
        _ => Cls::Normal,
    }
}

const NASTY: [u32; 10] = [
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0xffc0_0001,
    0x7f80_0001,
    0x7fbf_ffff,
    0x8000_0000,
    1,
    0x807f_ffff,
    0x0040_0000,
];

/// `n` f32 for `seed` in its class, at normal scale `scale`.
fn floats(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut r = Lcg(seed.wrapping_mul(7919).wrapping_add(13));
    let c = cls(seed);
    let mut v: Vec<f32> = (0..n)
        .map(|_| {
            let (u1, u2) = ((r.f() + 1.0) * 0.5, (r.f() + 1.0) * 0.5);
            let x = match c {
                // Box–Muller: a normal draw from two uniforms.
                Cls::Normal | Cls::Nasty => {
                    scale * (-2.0 * u1.max(1e-12).ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
                }
                Cls::Wide => {
                    (1.0 + u1)
                        * 2f32.powi((u2 * 120.0) as i32 - 60)
                        * if r.f() < 0.0 { -1.0 } else { 1.0 }
                }
            };
            if c == Cls::Nasty && r.f() > 0.97 {
                f32::from_bits(NASTY[(u1 * 9.99) as usize])
            } else {
                x
            }
        })
        .collect();
    if n > 4 && c == Cls::Normal {
        v[..4].copy_from_slice(&[-0.0, 1e-40, 1e-30, -1e-38]);
    }
    v
}

/// `n` random bytes for `seed`, salted per operand.
fn bytes(n: usize, seed: u64, salt: u64) -> Vec<u8> {
    let mut r = Lcg(seed.wrapping_mul(104_729).wrapping_add(salt));
    (0..n).map(|_| ((r.f() + 1.0) * 127.99) as u8).collect()
}

/// One seeded launch pair: kernels `(hipcc, Rust)`, the geometry both launch at, the f32
/// output size, the input buffers (kept alive here until the launch is compared), and the
/// argument list as a function of the output pointer.
struct Launch {
    k: (&'static str, &'static str),
    geom: Geom,
    out_bytes: usize,
    _ins: Vec<Dev>,
    args: Box<dyn Fn(*mut std::ffi::c_void) -> Args>,
}

/// One row: `cell(seed)` for every seed, each pair into fresh, differently filled outputs and
/// compared on the device. Every seeded probe writes f32, so its cells are NaN-classified.
fn group(p: &Probes, t: &mut Table, name: String, cell: impl Fn(u64) -> Launch) {
    let mut g = Group::default();
    for s in 1..=SEEDS {
        let l = cell(s);
        let (a, b) = (&Dev::zeroed(l.out_bytes), &Dev::zeroed(l.out_bytes));
        let o = Outs {
            a,
            b,
            bytes: l.out_bytes,
            ty: Out::F32,
        };
        let c = p.cell((l.k, l.geom), &l.args, o);
        g.cell(&c, l.out_bytes, || format!("seed={s} byte={}", c.first));
    }
    t.push(name, g);
}

pub fn reductions(p: &Probes, t: &mut Table) {
    const N: usize = 1 << 16;
    for k in ["k_wave_sum", "k_wave_max"] {
        group(p, t, k.into(), |s| {
            let x = Dev::from(&floats(N, s, 1.0));
            let xp = x.ptr();
            Launch {
                k: (k, k),
                geom: blocks((N / BLOCK as usize) as u32),
                out_bytes: N * 4,
                _ins: vec![x],
                args: Box::new(move |o| Args::default().p(xp).p(o)),
            }
        });
    }
    for bd in [32u32, 64, 128, 256, 512, 1024] {
        const BLOCKS: u32 = 64;
        let n = (bd * BLOCKS) as usize;
        group(p, t, format!("k_block_sum2 bd={bd}"), |s| {
            let (x, y) = (
                Dev::from(&floats(n, s, 1.0)),
                Dev::from(&floats(n, s + 1000, 3.0)),
            );
            let (xp, yp) = (x.ptr(), y.ptr());
            let args = Box::new(move |o| Args::default().p(xp).p(yp).p(o).p(bd));
            let k = ("k_block_sum2", "k_block_sum2");
            Launch {
                k,
                geom: Geom {
                    blocks: BLOCKS,
                    threads: bd,
                    lds: LdsBytes(bd * 4),
                },
                out_bytes: n * 8,
                _ins: vec![x, y],
                args,
            }
        });
    }
}

/// Rows per dot launch: 3 blocks of 8 waves, the last partial.
const O_DIM: i32 = 20;
const WAVE_GRID: u32 = (O_DIM as u32).div_ceil(8);

/// Rows a dot kernel reduces at once: 2 for `_r2`, else 1.
fn rows_of(k: &str) -> usize {
    if k.contains("_r2") { 2 } else { 1 }
}

pub fn dot_families(p: &Probes, t: &mut Table) {
    if t.wants("fp8") {
        fp8(p, t);
    }
    if t.wants("i4") {
        nib(
            p,
            t,
            "k_i4_wave",
            &[6144, 2048, 1280, 1024, 4224, 1028, 1000, 999, 130, 7],
        );
    }
    if t.wants("f4") {
        nib(
            p,
            t,
            "k_f4_wave",
            &[4096, 2048, 7168, 1280, 1152, 3712, 1028, 136, 2050, 64],
        );
    }
    if t.wants("vq") {
        vq(p, t);
    }
}

fn fp8(p: &Probes, t: &mut Table) {
    let shapes = [
        (6144, 128),
        (2048, 128),
        (7168, 128),
        (4100, 128),
        (1028, 4),
        (36, 4),
        (2048, 2),
        (1001, 1),
        (999, 2),
        (4, 128),
    ];
    for k in [
        "k_fp8_wave",
        "k_fp8_wave_r1",
        "k_fp8_wave_r2",
        "k_fp8_splitk",
        "k_fp8_splitk_r1",
        "k_fp8_splitk_r2",
    ] {
        let (r, splitk) = (rows_of(k), k.contains("splitk"));
        for (i_dim, block) in shapes {
            let (sc_cols, sc_rows) = ((i_dim + block - 1) / block, (O_DIM + block - 1) / block);
            group(p, t, format!("{k} i_dim={i_dim} block={block}"), |s| {
                let x = Dev::from(&floats(r * i_dim as usize, s, 1.0));
                let mut w = bytes((O_DIM * i_dim) as usize, s, 1);
                if cls(s) != Cls::Nasty {
                    // e4m3's NaN codes (S.1111.111) only in the nasty class.
                    w.iter_mut()
                        .filter(|b| **b & 0x7f == 0x7f)
                        .for_each(|b| *b ^= 1);
                }
                let (w, sc) = (
                    Dev::from(&w),
                    Dev::from(&floats((sc_rows * sc_cols) as usize, s + 500, 0.01)),
                );
                let (xp, wp, sp) = (x.ptr(), w.ptr(), sc.ptr());
                let (width, grid) = if splitk {
                    (256, O_DIM as u32)
                } else {
                    (32, WAVE_GRID)
                };
                Launch {
                    k: (k, k),
                    // The Rust twin builds its LUT in 1 KiB of dynamic LDS; hipcc's is a
                    // static array and leaves the dynamic bytes unread.
                    geom: Geom {
                        lds: LdsBytes(1024),
                        ..blocks(grid)
                    },
                    out_bytes: O_DIM as usize * r * width * 4,
                    _ins: vec![x, w, sc],
                    args: Box::new(move |o| {
                        Args::default()
                            .p(xp)
                            .p(wp)
                            .p(sp)
                            .p(O_DIM)
                            .p(i_dim)
                            .p(block)
                            .p(sc_cols)
                            .p(o)
                    }),
                }
            });
        }
    }
}

/// A wave-per-row dot's launch: `ins[..3]` are its three leading operands, then `O_DIM`,
/// the width, and the output — the `I4_ARGS`/`F4_ARGS` order (VQ adds a fourth operand).
fn wave_dot(k: (&'static str, &'static str), r: usize, width: i32, ins: Vec<Dev>) -> Launch {
    let ptrs: Vec<_> = ins.iter().map(Dev::ptr).collect();
    Launch {
        k,
        geom: blocks(WAVE_GRID),
        out_bytes: O_DIM as usize * r * 32 * 4,
        _ins: ins,
        args: Box::new(move |o| {
            ptrs.iter()
                .fold(Args::default(), |a, &p| a.p(p))
                .p(O_DIM)
                .p(width)
                .p(o)
        }),
    }
}

/// Whether an `r`-row nibble launch at `dim` is skipped: R=2 needs every dword-path
/// activation row 16-byte aligned (`dim % 4 == 0`); int4 rows under 256 never take that path.
fn misaligned(r: usize, dim: i32, f4: bool) -> bool {
    let long_enough = f4 || dim >= 256;
    r == 2 && dim % 4 != 0 && long_enough
}

/// fp4 operands: packed e2m1 weights, e8m0 scales in a realistic band unless the seed is
/// nasty (which keeps 0, 0xff and 254 in play).
fn f4_operands(d: usize, s: u64) -> (Dev, Dev) {
    let mut sc = bytes(O_DIM as usize * d.div_ceil(32), s, 4);
    if cls(s) != Cls::Nasty {
        sc.iter_mut().for_each(|b| *b = 120 + *b % 15);
    }
    (
        Dev::from(&bytes(O_DIM as usize * d / 2, s, 3)),
        Dev::from(&sc),
    )
}

/// int4 operands: packed nibbles, one f32 scale per 128 columns.
fn i4_operands(d: usize, s: u64) -> (Dev, Dev) {
    let scales = floats(O_DIM as usize * d.div_ceil(128), s + 600, 0.02);
    (
        Dev::from(&bytes(O_DIM as usize * d.div_ceil(2), s, 2)),
        Dev::from(&scales),
    )
}

/// A format's weight and scale operands for a dim and a seed.
type Operands = fn(usize, u64) -> (Dev, Dev);

/// int4 (`k_i4_wave*`, f32 group scales) and fp4 (`k_f4_wave_r*`, e8m0 scales): one loop
/// family in the prelude, so one row builder here.
fn nib(p: &Probes, t: &mut Table, stem: &str, dims: &[i32]) {
    let f4 = stem.starts_with("k_f4");
    let (kernels, operands): (&[&'static str], Operands) = if f4 {
        (&["k_f4_wave_r1", "k_f4_wave_r2"], f4_operands)
    } else {
        (&["k_i4_wave", "k_i4_wave_r1", "k_i4_wave_r2"], i4_operands)
    };
    for &k in kernels {
        let r = rows_of(k);
        for &dim in dims.iter().filter(|&&d| !misaligned(r, d, f4)) {
            group(p, t, format!("{k} dim={dim}"), |s| {
                let d = dim as usize;
                let (w, sc) = operands(d, s);
                wave_dot(
                    (k, k),
                    r,
                    dim,
                    vec![Dev::from(&floats(r * d, s, 1.0)), w, sc],
                )
            });
        }
    }
}

/// bf16 scale `±[2^-3, 2^3)` and fp16 codebook entries `±[2^-5, 2^5)`, with special codes
/// sprinkled into the nasty class.
fn vq_operands(r: &mut Lcg, (n_sc, nasty): (usize, bool)) -> (Vec<u16>, Vec<u16>) {
    let mut u = || ((r.f() + 1.0) * 32767.99) as u16;
    let mut sc: Vec<u16> = (0..n_sc)
        .map(|_| 0x3c00 + (u() % 0x400) - 0x200 + ((u() & 1) << 15))
        .collect();
    let mut cb: Vec<u16> = (0..4096 * 4)
        .map(|_| (u() & 0x83ff) | ((10 + u() % 10) << 10))
        .collect();
    if nasty {
        for _ in 0..64 {
            let i = u() as usize % cb.len();
            cb[i] = [0x7c00, 0xfc00, 0x7e00, 0x7c01, 0x8000, 0x0001, 0x03ff][u() as usize % 7];
        }
        for _ in 0..4 {
            let i = u() as usize % sc.len();
            sc[i] = [0x7f80, 0xffc0, 0x8000, 0x0001][u() as usize % 4];
        }
    }
    (sc, cb)
}

fn vq(p: &Probes, t: &mut Table) {
    // `k_vq_wave_r1` is the same function as `k_vq_wave` on the Rust side, which LTO merges
    // away (no `.kd`), so hipcc's `_r1` is scored against Rust's `k_vq_wave` (0b's note).
    for (hk, rk) in [
        ("k_vq_wave", "k_vq_wave"),
        ("k_vq_wave_r1", "k_vq_wave"),
        ("k_vq_wave_r2", "k_vq_wave_r2"),
    ] {
        let r = rows_of(hk);
        for i_dim in [6144i32, 2048, 1032, 4104, 8] {
            let nsub = i_dim as usize / 4;
            group(p, t, format!("{hk} i_dim={i_dim}"), |s| {
                let x = Dev::from(&floats(r * i_dim as usize, s, 1.0));
                let idx = Dev::from(&bytes(O_DIM as usize * nsub * 12 / 8 + 1, s, 5));
                let mut rng = Lcg(s.wrapping_mul(31).wrapping_add(7));
                let n_sc = O_DIM as usize * nsub.div_ceil(16);
                let (sc, cb) = vq_operands(&mut rng, (n_sc, cls(s) == Cls::Nasty));
                wave_dot(
                    (hk, rk),
                    r,
                    i_dim,
                    vec![x, idx, Dev::from(&sc), Dev::from(&cb)],
                )
            });
        }
    }
}
