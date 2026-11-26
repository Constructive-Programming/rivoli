//! The OCP MX FP8 dot with a ROW-WISE ue8m0 scale — Inkling-2's `[1, 32]` grid, which is
//! `dot_fp8`'s tile grid at one row. No C twin: Inkling-2 has no hipcc kernel
//! (owner direction 2026-11-24), so this helper is scored against the first-party
//! `rivoli_artifact::inkling_dequant::Mxfp8` through an oracle suite rather than against
//! the bit-identity gate.
//!
//! **The scale is a raw U8 exponent, not an e4m3 code (trap T1).** `Mxfp8::decode_row`
//! multiplies the decoded e4m3 value by `2^(scale − 127)`; reading the byte as an e4m3 code
//! instead — `e4m3_byte` here, the fp8 path's LUT elsewhere — would decode it as an e4m3
//! number (`0x80` → `-0`, `0x70` → `224`) and produce finite, plausible, wrong output.
//! Hence [`e8m0f`], the same decode the fp4 path uses for its group scales.
//!
//! **The e4m3 weights decode arithmetically, in registers — no LUT, no shared-memory table,
//! no barrier.** Placing a byte's magnitude at f32 bits `26..20` (`m7 << 20`) puts the e4m3
//! exponent field where f32's lives and its 3-bit mantissa under it, so the bits read as the
//! code's value at bias 127 — e4m3 subnormals become f32 subnormals, `±m·2^-129` — and ONE
//! multiply by `2^120` re-biases to bias 7, lifting those subnormals to the exact `±m·2^-9`.
//! All 254 finite codes land on exactly the values `formats::e4m3f` gives — both paths are
//! exact in f32, so the LUT's 256 entries were this arithmetic written down — `±0.0`
//! included. `0x7F`/`0xFF` decode as `±480` (the pattern's normal reading) instead of NaN,
//! so this decoder is FINITE-ONLY: Inkling-2's MXFP8 planes hold no NaN, and the oracle
//! suite draws `0x00..=0x3E`. `formats::e4m3f` stays the general decoder for `mxfp8_dequant`.
//!
//! The 256-entry table this replaces cost more than its 1 KB: every `ds_load_b32` indexed it
//! by a DATA byte, so a wave's 32 lanes scattered over the 32 banks by weight VALUE and each
//! of the four loads paid whatever conflict count the data drew (32-way, a full serialise, is
//! reachable); and the table lived in dynamic LDS, so each kernel built it at block entry and
//! `sync_threads`ed before its first dot, and the launch had to grant `GEMV_FP8_LDS_BYTES`.
//! About a dozen integer ops and four multiplies per dword need no shared memory at all.
//! Subnormal codes depend on the device's IEEE mode — `gfx1151` runs it (no
//! `oclc_daz_opt_on`), and the oracle suite's `0x01..=0x07` codes redden if that ever stops
//! being true.
//!
//! **Fused sites** are `dot_fp8.rs`'s: `rowview::quad`, then `fma(s, quad, acc)`; a tail
//! column is `fma(x·e4m3(w), s, acc)`.
//!
//! **The tile is one row, so a chunk never straddles two scale groups — and the group is
//! 32 columns wide, twice the sixteen a lane's `global_load_b128` brings in.** A lane starts
//! at `col = step + lane·16`, a multiple of 16, so its sixteen weights lie inside ONE group
//! and share ONE scale byte: the 16-byte pass decodes the scale once per sixteen weights
//! where the dword pass needs one per four, i.e. 4× fewer scale loads and 4× fewer
//! `e8m0f` (six instructions each) for the same bytes, and one `fma(s, ·, acc)` instead of
//! four.
//!
//! Three tiers, a PARTITION of the columns: 16 bytes per lane (512 per wave-trip), then a
//! dword per lane (128 per trip) for what the 512-column floor cannot take, then one column
//! per lane. Every Inkling-2 width lands with no scalar tail at all — Mini's
//! `2304 = 4·512 + 2·128` and `1152 = 2·512 + 128`, Small's `4608 = 9·512` and
//! `2304 = 4·512 + 2·128` — and the oracle suite's `(hidden, inter)` of `(512, 256)`,
//! `(128, 32)` and `(64, 32)` reaches all three tiers on one pass or the other.

use core::intrinsics::fmuladd;

use super::dot_nib::e8m0f;
use super::reduce::{WAVE, wave_sum};
use super::rowview::{F32x4, RowsView, U32x4, aligned_base, for_strided, quad};

/// log2 of the columns one ue8m0 scale byte governs: the `[1, 32]` grid of
/// `inkling_dequant::Mxfp8` (`LOAD:808–810`).
pub const MXFP8_GROUP_SHIFT: u32 = 5;
/// Columns per scale byte — 32, the group `rivoli_abi::geom::F4_GROUP` also names.
pub const MXFP8_GROUP: usize = 1 << MXFP8_GROUP_SHIFT;

/// One MXFP8 weight row: `w` = `packed[o*i_dim..]` (one e4m3 byte per column) and `scale` =
/// `scales[o*ceil(i_dim/32)..]` (one ue8m0 exponent per 32 columns). Column `i` is
/// `e4m3_byte(w[i]) · 2^(scale[i >> 5] − 127)` — decoded in registers, so a row holds no
/// shared-memory table and the launch grants the block no dynamic LDS.
#[derive(Clone, Copy)]
pub struct MxFp8Row {
    pub w: *const u8,
    pub scale: *const u8,
    pub dim: usize,
}

/// Columns one 16-byte-pass step covers across the wave: 16 per lane, 32 lanes.
const CHUNK16_STEP: usize = WAVE * 16;

/// Columns one dword-pass step covers across the wave: 4 per lane, 32 lanes.
const DWORD_STEP: usize = WAVE * 4;

/// Where the 16-byte pass stops: [`aligned_base`]'s floor when BOTH the weight row and the
/// activation row are 16-byte aligned, else 0 so the dword pass takes every column.
///
/// The activation half is row 0's base (`x.x`): every tier's `col` is a multiple of 4, so a
/// 16-byte-aligned row base makes every `x.at(0, col)` a [`F32x4`]-aligned address. The
/// launcher's contract fixes `x` at 16 bytes and `h` at 16 bytes, but pass 2's row base is
/// `h + e·inter` — 16-byte only while `inter % 4 == 0` — so the check is made, not assumed.
#[inline(always)]
fn base16(x: RowsView, w: MxFp8Row, step: usize) -> usize {
    if (w.w as usize | x.x as usize) & 15 != 0 {
        return 0;
    }
    aligned_base(w.w, w.dim, step)
}

/// `2^120`: re-biases an e4m3 byte placed at f32 bits `26..20` (exponent bias 7 → 127) and
/// lifts e4m3 subnormals (`±m·2^-129`) to the exact `±m·2^-9`.
const E4M3_REBIAS: f32 = f32::from_bits(0x7B80_0000);

/// One e4m3 byte decoded in registers: sign to bit 31, magnitude to bits `26..20`, then
/// [`E4M3_REBIAS`]. `b` is masked here, so a caller may hand over a byte read as a `u32` or
/// a dword still holding its neighbours above bit 7.
///
/// Exact for every finite code and `±0.0`; `0x7F`/`0xFF` read as `±480`, not NaN (the module
/// note's finite-only argument).
#[inline(always)]
fn e4m3_byte(b: u32) -> f32 {
    f32::from_bits(((b & 0x80) << 24) | ((b & 0x7F) << 20)) * E4M3_REBIAS
}

/// The four e4m3 weights of dword `p` (little-endian byte order, matching the row), decoded
/// in registers: zero LDS traffic, zero shared-memory offsets to hold.
#[inline(always)]
fn decode_quad(p: u32) -> [f32; 4] {
    [
        e4m3_byte(p),
        e4m3_byte(p >> 8),
        e4m3_byte(p >> 16),
        e4m3_byte(p >> 24),
    ]
}

/// The sixteen activation floats at `col` — four `global_load_b128`.
///
/// # Safety
/// `col` must be a multiple of 4 with `x.at(0, col)` 16-byte aligned and 16 floats readable.
#[inline(always)]
unsafe fn load_x16(x: RowsView, col: usize) -> [F32x4; 4] {
    // SAFETY: the caller's obligation.
    unsafe {
        let p = x.at(0, col);
        [
            F32x4::load(p),
            F32x4::load(p.add(4)),
            F32x4::load(p.add(8)),
            F32x4::load(p.add(12)),
        ]
    }
}

/// The four activation floats at `col` — the dword pass's `x`, loaded once for a pair.
///
/// # Safety
/// 4 floats must be readable at `x.at(0, col)`.
#[inline(always)]
unsafe fn load_x4(x: RowsView, col: usize) -> [f32; 4] {
    // SAFETY: the caller's obligation.
    unsafe {
        let p = x.at(0, col);
        [*p, *p.add(1), *p.add(2), *p.add(3)]
    }
}

/// The sixteen products `x[k]·e4m3(w[k])` of one 16-byte chunk, summed before its ONE scale.
///
/// A balanced tree of the four [`quad`]s: `quad` keeps hipcc's within-dword fold order, the
/// tree only shortens the chunk's dependency chain (three adds, not sixteen), and the bound
/// the oracle suite states — `(inter + hidden)·ε` relative — is decades above a
/// reassociation of sixteen terms.
#[inline(always)]
fn sum16(x: [F32x4; 4], p: [u32; 4]) -> f32 {
    (quad(x[0].0, decode_quad(p[0])) + quad(x[1].0, decode_quad(p[1])))
        + (quad(x[2].0, decode_quad(p[2])) + quad(x[3].0, decode_quad(p[3])))
}

/// `acc + s·Σ` over the sixteen columns at `col` of row `w`, `x` already in registers: the
/// 16-byte pass's inner step — one scale byte and one `e8m0f` for sixteen weights.
///
/// Hoisting `s` out of the sixteen multiplies moves no value: `e8m0f` is exactly `2^(b−127)`,
/// and scaling by a power of two is exact, so `s·(Σ q)` and `Σ s·q` differ only where the
/// sixteen terms' own additions are reassociated (see [`sum16`]).
///
/// # Safety
/// `col` must be a multiple of 16 inside `w.dim` with `w.w.add(col)` and `x`'s `col` both
/// 16-byte aligned.
#[inline(always)]
unsafe fn dot16(x: [F32x4; 4], w: MxFp8Row, col: usize, acc: f32) -> f32 {
    // SAFETY: the caller's obligation.
    unsafe {
        let s = e8m0f(*w.scale.add(col >> MXFP8_GROUP_SHIFT));
        fmuladd(s, sum16(x, U32x4::load(w.w.add(col)).0), acc)
    }
}

/// `acc + s·quad` over the four columns at `col`: the dword pass's inner step, one scale
/// byte per dword (the four bytes share it: `col % 4 == 0` and the group is 32 wide).
///
/// # Safety
/// `col` must be a multiple of 4 inside `w.dim`.
#[inline(always)]
unsafe fn dot4(xq: [f32; 4], w: MxFp8Row, col: usize, acc: f32) -> f32 {
    // SAFETY: the caller's obligation.
    unsafe {
        let p = *w.w.cast::<u32>().add(col >> 2);
        let s = e8m0f(*w.scale.add(col >> MXFP8_GROUP_SHIFT));
        fmuladd(s, quad(xq, decode_quad(p)), acc)
    }
}

/// `acc + x·e4m3(w)·s` for one column: the tail's step, one scale byte per column.
///
/// # Safety
/// `i` must be inside `w.dim`.
#[inline(always)]
unsafe fn dot1(xi: f32, w: MxFp8Row, i: usize, acc: f32) -> f32 {
    // SAFETY: the caller's obligation.
    unsafe {
        let v = e4m3_byte(u32::from(*w.w.add(i)));
        let s = e8m0f(*w.scale.add(i >> MXFP8_GROUP_SHIFT));
        fmuladd(xi * v, s, acc)
    }
}

/// One WAVE reduces one weight row: Σ over the wave's columns of `x·e4m3(w)·2^(s−127)`, the
/// result on lane 0 after [`wave_sum`]. Sixteen columns per lane per trip while both the row
/// and the activation are 16-byte aligned, a dword per lane after that while the row is
/// 4-byte aligned, a column per lane last — the tiers of the module note.
///
/// # Safety
/// `w.w`, `w.scale` and row 0 of `x` must be readable over `w.dim` columns. Every lane of the
/// wave must call it.
#[inline(always)]
pub unsafe fn dot_mxfp8_wave<const U: usize>(x: RowsView, w: MxFp8Row, lane: usize) -> f32 {
    let mut acc = 0.0f32;
    // The tiers PARTITION the columns (`b16 <= b4 <= dim`): an unaligned row or activation
    // moves a boundary down to 0 rather than loading past it.
    let b16 = base16(x, w, CHUNK16_STEP);
    let b4 = aligned_base(w.w, w.dim, DWORD_STEP);
    // SAFETY (all three tiers): the caller's obligation — tier 1 reads columns
    // `col..col+16 <= b16`, tier 2 `col..col+4 <= b4`, the tail `i < dim`.
    unsafe {
        for_strided::<U>(0, b16, CHUNK16_STEP, |step| {
            let col = step + lane * 16;
            acc = dot16(load_x16(x, col), w, col, acc);
        });
        for_strided::<1>(b16, b4, DWORD_STEP, |step| {
            let col = step + lane * 4;
            acc = dot4(load_x4(x, col), w, col, acc);
        });
        for_strided::<1>(b4 + lane, w.dim, WAVE, |i| {
            acc = dot1(*x.at(0, i), w, i, acc);
        });
    }
    wave_sum(acc)
}

/// The FUSED twin for a gate/up pair: ONE read of the activation row dotted against BOTH
/// rows, `(gate, up)` on lane 0 after their [`wave_sum`]s.
///
/// `moe_gateup_mxfp8_situ`'s two rows are the same `[inter, hidden]` shape over the same `x`
/// — the expert's hidden row — so dotting them one after the other loaded every activation
/// float twice for nothing, and paid the tier boundaries, the column arithmetic and the
/// scale decode twice as well. Here each tier loads `x` once and spends it on both rows;
/// the two weight streams are what the trip's bytes are (1 KB at `U = 1`).
///
/// # Safety
/// Both rows' `w`/`scale` must be readable over their own `dim` columns, and row 0 of `x`
/// readable over `max(g.dim, u.dim)`; every lane of the wave must call it.
#[inline(always)]
pub unsafe fn dot_mxfp8_pair_wave<const U: usize>(
    x: RowsView,
    g: MxFp8Row,
    u: MxFp8Row,
    lane: usize,
) -> (f32, f32) {
    let (mut acc_g, mut acc_u) = (0.0f32, 0.0f32);
    // Each boundary is the LOWER of the two rows': a tier may only run where BOTH rows take
    // it, and the tail carries whatever is left of either. The rows are the same width in
    // every real call, where this is exactly the single dot's ladder.
    let b16 = base16(x, g, CHUNK16_STEP).min(base16(x, u, CHUNK16_STEP));
    let b4 = aligned_base(g.w, g.dim, DWORD_STEP).min(aligned_base(u.w, u.dim, DWORD_STEP));
    let dim = g.dim.max(u.dim);
    // SAFETY (all three tiers): the caller's obligation — tier 1 reads columns
    // `col..col+16 <= b16` of both rows, tier 2 `col..col+4 <= b4`, the tail `i < dim` with
    // a per-row bound.
    unsafe {
        for_strided::<U>(0, b16, CHUNK16_STEP, |step| {
            let col = step + lane * 16;
            let xc = load_x16(x, col);
            acc_g = dot16(xc, g, col, acc_g);
            acc_u = dot16(xc, u, col, acc_u);
        });
        for_strided::<1>(b16, b4, DWORD_STEP, |step| {
            let col = step + lane * 4;
            let xq = load_x4(x, col);
            acc_g = dot4(xq, g, col, acc_g);
            acc_u = dot4(xq, u, col, acc_u);
        });
        for_strided::<1>(b4 + lane, dim, WAVE, |i| {
            let xi = *x.at(0, i);
            if i < g.dim {
                acc_g = dot1(xi, g, i, acc_g);
            }
            if i < u.dim {
                acc_u = dot1(xi, u, i, acc_u);
            }
        });
    }
    (wave_sum(acc_g), wave_sum(acc_u))
}
