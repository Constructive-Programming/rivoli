//! The prelude PROBE kernels — compiled only with the `probes` feature, into the SECOND code
//! object `tests/bit-identity.sh --build-probes` builds; never into the shipped one.
//!
//! Each kernel exposes ONE prelude helper with plain arguments, the twin of the same-named
//! kernel in `crates/engine/tests/bit_identity/probes/probe.hip` (built by hipcc against the
//! `archive/hip-kernels` headers), so `tests/bit-identity.sh --probes` launches both with
//! one argument buffer and compares the outputs. Ported from Phase 0b's scratch harness
//! (`docs/investigations/rust-kernels.md` §"Phase 0b — the prelude", 2026-11-18), which
//! measured them on the device. A red-proof plant is a patch to the real prelude
//! (`tests/bit-identity/plants/<family>.patch`), applied to a scratch copy, never a cfg here.

use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
use core::intrinsics::gpu::gpu_launch_sized_workgroup_mem;

use crate::prelude::{self as p, dot_fp8, dot_nib, dot_vq, formats as f, reduce, rowview};

// Every probe launches 256-thread blocks; the block size is a constant, never read from
// the dispatch packet (Phase 0c rule).
const BLOCK: u32 = 256;

#[inline(always)]
fn gid() -> u32 {
    workgroup_id_x() * BLOCK + workitem_id_x()
}

macro_rules! sweep {
    ($name:ident, $t:ty, |$b:ident $(, $a:ident : $at:ty)*| $e:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(base: u32, n: u32, out: *mut $t $(, $a: $at)*) {
            let g = gid();
            if g >= n { return; }
            let $b = base.wrapping_add(g);
            unsafe { *out.add(g as usize) = $e };
        }
    };
}

sweep!(ex_bf16f, f32, |b| f::bf16f(b as u16));
sweep!(ex_rbf16, f32, |b| f::rbf16(f32::from_bits(b)));
sweep!(ex_e4m3f, f32, |b| f::e4m3f(b as u8));
sweep!(ex_e2m1f, f32, |b| f::e2m1f(b));
sweep!(ex_e8m0f, f32, |b| dot_nib::e8m0f(b as u8));
sweep!(
    ex_frs,
    f32,
    |b, max_inv: f32| p::actquant::fast_round_scale(f32::from_bits(b), max_inv)
);
sweep!(ex_aq8, f32, |b, s: f32| p::actquant::act_quant_roundtrip(
    f32::from_bits(b),
    s
));
sweep!(ex_aq4, f32, |b, s: f32| p::actquant::fp4_quant_roundtrip(
    f32::from_bits(b),
    s
));
sweep!(ex_nib, f32, |b, k: i32| {
    <dot_nib::I4 as dot_nib::NibFormat>::decode(b >> (4 * k))
});
sweep!(ex_swiglu, f32, |b, limit: f32| p::swiglu_clamped(
    f::bf16f((b >> 16) as u16),
    f::bf16f(b as u16),
    limit
));
sweep!(ex_situ_g, f32, |b, u: f32, b1: f32, b2: f32| p::situ_glu(
    f32::from_bits(b),
    u,
    b1,
    b2
));
sweep!(ex_situ_u, f32, |b, g: f32, b1: f32, b2: f32| p::situ_glu(
    g,
    f32::from_bits(b),
    b1,
    b2
));
sweep!(ex_f2bf16, u16, |b| f::f2bf16(f32::from_bits(b)));
sweep!(ex_f2e4m3, u8, |b| f::f2e4m3_rne(f32::from_bits(b)));
sweep!(ex_f2e2m1, u32, |b| f::f2e2m1(f32::from_bits(b)));
sweep!(ex_blk_shift, u32, |b| rowview::blk_shift(b as i32) as u32);
sweep!(ex_nquad, u32, |b| p::absorb_nquad(b as i32) as u32);
sweep!(ex_moe_fixed, u64, |b| p::moe_fixed(f32::from_bits(b)));

#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn k_lut(out: *mut f32) {
    unsafe { dot_fp8::e4m3_lut_build(out, workitem_id_x() as usize) }
}

// ── reductions ──
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn k_wave_sum(input: *const f32, out: *mut f32) {
    let g = gid() as usize;
    unsafe { *out.add(g) = reduce::wave_sum(*input.add(g)) }
}
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn k_wave_max(input: *const f32, out: *mut f32) {
    let g = gid() as usize;
    unsafe { *out.add(g) = reduce::wave_max(*input.add(g)) }
}
#[unsafe(no_mangle)]
pub unsafe extern "gpu-kernel" fn k_block_sum2(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    bd: u32,
) {
    let red = gpu_launch_sized_workgroup_mem::<f32>();
    let t = workitem_id_x() as usize;
    let g = workgroup_id_x() as usize * bd as usize + t;
    unsafe {
        let s = reduce::block_sum_lds(*a.add(g), red, t, bd as usize);
        let u = reduce::block_sum_lds(*b.add(g), red, t, bd as usize);
        *out.add(2 * g) = s;
        *out.add(2 * g + 1) = u;
    }
}

// ── dots ──
const W: usize = reduce::WAVE;

/// One wave per weight row: (row, lane), or None past `o_dim`.
#[inline(always)]
fn wave_row(o_dim: i32) -> Option<(usize, usize)> {
    let t = workitem_id_x() as usize;
    let o = workgroup_id_x() as usize * reduce::ROWS_PER_BLOCK + (t >> 5);
    (o < o_dim as usize).then_some((o, t & 31))
}

#[inline(always)]
unsafe fn store_rows<const R: usize>(
    out: *mut f32,
    (o, lane, width): (usize, usize, usize),
    a: [f32; R],
) {
    for (r, v) in a.into_iter().enumerate() {
        unsafe { *out.add((o * R + r) * width + lane) = v };
    }
}

/// One `#[no_mangle]` probe kernel taking its argument block by value and handing it to a
/// generic body. The blocks below are `repr(C)` in the C kernel's PARAMETER ORDER, so their
/// bytes are exactly the positional kernarg layout hipcc gives `probe.hip`'s twin — the
/// harness launches both from one packed buffer.
macro_rules! probe {
    ($name:ident($args:ty) = $body:expr) => {
        #[unsafe(no_mangle)]
        pub unsafe extern "gpu-kernel" fn $name(a: $args) {
            unsafe { $body(a) }
        }
    };
}

/// `FP8_ARGS` of `probe.hip`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Fp8Args {
    x: *const f32,
    w: *const u8,
    scale: *const f32,
    o_dim: i32,
    i_dim: i32,
    block: i32,
    sc_cols: i32,
    out: *mut f32,
}

/// The LUT in dynamic LDS, then row `o`'s view.
#[inline(always)]
unsafe fn fp8_row(a: &Fp8Args, o: usize) -> dot_fp8::Fp8Row {
    let lut = gpu_launch_sized_workgroup_mem::<f32>();
    unsafe {
        dot_fp8::e4m3_lut_build(lut, workitem_id_x() as usize);
        reduce::sync_threads();
        dot_fp8::Fp8Row {
            w: a.w.add(o * a.i_dim as usize),
            scale: a
                .scale
                .add((o >> rowview::blk_shift(a.block)) * a.sc_cols as usize),
            lut,
            i_dim: a.i_dim as usize,
            block: a.block,
        }
    }
}

/// Store a wave-per-row dot's `R` results for every lane; `None` = this wave has no row.
#[inline(always)]
unsafe fn store_wave<const R: usize>(
    out: *mut f32,
    wr: Option<(usize, usize)>,
    dot: impl FnOnce(usize, usize) -> [f32; R],
) {
    if let Some((o, lane)) = wr {
        unsafe { store_rows(out, (o, lane, W), dot(o, lane)) };
    }
}

impl Fp8Args {
    /// The activation rows, `i_dim` apart.
    fn x(&self) -> rowview::RowsView {
        rowview::RowsView {
            x: self.x,
            stride: self.i_dim as usize,
        }
    }
}

/// One wave per weight row, every lane's result stored.
unsafe fn fp8_wave<const R: usize, const U: usize>(a: Fp8Args) {
    let wr = wave_row(a.o_dim);
    // The LUT build barriers, so every thread builds before any returns.
    let row = unsafe { fp8_row(&a, wr.map_or(0, |(o, _)| o)) };
    unsafe {
        store_wave(a.out, wr, |_, lane| {
            dot_fp8::dot_fp8_wave_r::<R, U>(a.x(), row, lane)
        })
    };
}

/// Split-K: one BLOCK per row, all threads stride, every thread's partial stored.
unsafe fn fp8_splitk<const R: usize, const U: usize>(a: Fp8Args) {
    let (o, t) = (workgroup_id_x() as usize, workitem_id_x() as usize);
    let row = unsafe { fp8_row(&a, o) };
    let span = rowview::LaneSpan {
        start: t,
        stride: BLOCK as usize,
    };
    let acc = unsafe { dot_fp8::fp8_dot_strided_r::<R, U>(a.x(), row, span) };
    unsafe { store_rows(a.out, (o, t, BLOCK as usize), acc) };
}

probe!(k_fp8_wave(Fp8Args) = fp8_wave::<1, { dot_fp8::FP8_SINGLE_ROW_UNROLL }>);
probe!(k_fp8_wave_r1(Fp8Args) = fp8_wave::<1, 1>);
probe!(k_fp8_wave_r2(Fp8Args) = fp8_wave::<2, 1>);
probe!(k_fp8_splitk(Fp8Args) = fp8_splitk::<1, { dot_fp8::FP8_SINGLE_ROW_UNROLL }>);
probe!(k_fp8_splitk_r1(Fp8Args) = fp8_splitk::<1, 1>);
probe!(k_fp8_splitk_r2(Fp8Args) = fp8_splitk::<2, 1>);

/// `I4_ARGS` (int4, `S = f32` group scales) and `F4_ARGS` (fp4, `S = u8` e8m0 scales) of
/// `probe.hip` — one block, typed by the scale element.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NibArgs<S> {
    v: *const f32,
    w: *const u8,
    scale: *const S,
    o_dim: i32,
    dim: i32,
    out: *mut f32,
}

/// Both nibble formats: weight row stride `ceil(dim/2)` bytes, one scale per `2^GROUP_SHIFT`
/// columns (int4's 128, fp4's 32). For fp4 `dim` is even, so `ceil(dim/2)` is `dim/2` —
/// the stride `probe.hip`'s `f4_wave_r` uses.
unsafe fn nib_wave<F: dot_nib::NibFormat, const R: usize, const U: usize>(a: NibArgs<F::Scale>) {
    let d = a.dim as usize;
    unsafe {
        store_wave(a.out, wave_row(a.o_dim), |o, lane| {
            let groups = d.div_ceil(1 << F::GROUP_SHIFT);
            let row = dot_nib::NibRow::<F> {
                w: a.w.add(o * d.div_ceil(2)),
                scale: a.scale.add(o * groups),
                dim: d,
            };
            dot_nib::dot_nib_wave_r::<R, U, F>(rowview::RowsView { x: a.v, stride: d }, row, lane)
        })
    };
}

use dot_nib::{F4, I4, NIBBLE_BATCHED_UNROLL as NBU};
probe!(k_i4_wave(NibArgs<f32>) = nib_wave::<I4, 1, 1>);
probe!(k_i4_wave_r1(NibArgs<f32>) = nib_wave::<I4, 1, NBU>);
probe!(k_i4_wave_r2(NibArgs<f32>) = nib_wave::<I4, 2, NBU>);
probe!(k_f4_wave_r1(NibArgs<u8>) = nib_wave::<F4, 1, NBU>);
probe!(k_f4_wave_r2(NibArgs<u8>) = nib_wave::<F4, 2, NBU>);

/// `VQ_ARGS` of `probe.hip`: the activations, packed 12-bit indices, bf16 group scales and
/// the shared fp16 codebook, then the extents and the output.
#[derive(Clone, Copy)]
#[repr(C)]
pub struct VqArgs {
    x: *const f32,
    idx: *const u8,
    scale: *const u16,
    cb: *const f16,
    o_dim: i32,
    i_dim: i32,
    out: *mut f32,
}

unsafe fn vq_wave<const R: usize>(a: VqArgs) {
    let d = a.i_dim as usize;
    let (nsub, xs) = (d / dot_vq::VQ_DIM, rowview::RowsView { x: a.x, stride: d });
    unsafe {
        store_wave(a.out, wave_row(a.o_dim), |o, lane| {
            let row = dot_vq::VqRow {
                idx: a.idx.add(o * (nsub * dot_vq::VQ_INDEX_BITS / 8)),
                scale: a.scale.add(o * nsub.div_ceil(dot_vq::VQ_SUBS_PER_GROUP)),
                cb: a.cb,
                i_dim: d,
            };
            dot_vq::dot_vq_wave_r::<R>(xs, row, lane)
        })
    };
}

probe!(k_vq_wave(VqArgs) = vq_wave::<1>);
// k_vq_wave_r1 would be the identical function, which LTO merges away (no .kd); the harness
// scores hip k_vq_wave_r1 against rust k_vq_wave.
probe!(k_vq_wave_r2(VqArgs) = vq_wave::<2>);
