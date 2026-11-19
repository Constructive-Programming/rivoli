//! `kvcompress.hip`'s launchers (tag `archive/hip-kernels`): the compressor's bf16 GEMM, the state deposit and the two pooling
//! phases. The C took `CompGeom`/`CompFinish` by POINTER across the wall and refused a null one
//! (1000); the wrappers take references, so that refusal has no Rust spelling and the guards start
//! at the geometry.

use super::{Rows, any_nonpositive, elementwise, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::CompGeom;
use rivoli_abi::geom::{COMPRESS_BLOCK, COMPRESS_ENTS_MAX, COMPRESS_LDS_BYTES, WAVE};
use rivoli_abi::kernarg::{CompressDecode, CompressPrefill, Deposit, GemmBf16, Strided};
use std::ffi::c_void;

static GEMM_BF16: Kernel<GemmBf16> = Kernel::new(c"gemm_bf16");
static KV_COMPRESS_DEPOSIT: Kernel<Strided<Deposit>> = Kernel::new(c"kv_compress_deposit");
static KV_COMPRESS_PREFILL: Kernel<CompressPrefill> = Kernel::new(c"kv_compress_prefill");
static KV_COMPRESS_DECODE: Kernel<CompressDecode> = Kernel::new(c"kv_compress_decode");

/// `m · n` waves of `WAVE` lanes, in elementwise blocks (`GEMM_BF16_BLOCK` = `ELEMENTWISE_BLOCK`).
///
/// # Safety
/// The `launch_gemm_bf16` contract.
pub(crate) unsafe fn rivoli_gemm_bf16(a: GemmBf16, stream: *mut c_void) -> i32 {
    let checked = if any_nonpositive(&[a.m, a.n, a.k]) {
        Err(1001)
    } else {
        elementwise(i64::from(a.m) * i64::from(a.n) * i64::from(WAVE))
    };
    unsafe { run(&GEMM_BF16, checked, stream, &a) }
}

/// The compressor's shared refusals (the C's `compress_guard`), in its order: extents and
/// `coff ∈ {1, 2}` (1001), `rd` within `d` and even (1002), the DERIVED `cd`/`ents` (1003), the LDS
/// tables (1004), and `overlap ⇔ ratio == 4` (1010). `ratio <= 1` because at 1 the deposit writes
/// nothing and the pool reads an unfilled slot. The products and sums WRAP, as hipcc's `int`
/// arithmetic did in practice: a debug build's overflow check would panic where the C returned a
/// code (correctness review, 2026-11-19).
fn compress_guard(p: &CompGeom) -> Result<(), i32> {
    let checks = [
        (
            p.ratio <= 1 || any_nonpositive(&[p.d, p.rd]) || !(1..=2).contains(&p.coff),
            1001,
        ),
        (p.rd > p.d || p.rd & 1 != 0, 1002),
        (
            p.cd != p.coff.wrapping_mul(p.d) || p.ents != p.coff.wrapping_mul(p.ratio),
            1003,
        ),
        (p.ents > COMPRESS_ENTS_MAX as i32, 1004),
        ((p.coff == 2) != (p.ratio == 4), 1010),
    ];
    checks
        .into_iter()
        .find_map(|(bad, code)| bad.then_some(code))
        .map_or(Ok(()), Err)
}

/// One pooling block per row of `COMPRESS_BLOCK` threads with the window's LDS.
fn pool(Rows(n): Rows) -> Geometry {
    Geometry {
        blocks: n as u32,
        threads: COMPRESS_BLOCK,
        lds: LdsBytes(COMPRESS_LDS_BYTES),
    }
}

/// The geometry (above), `s > 0` (1005), then a run that stays inside the `[ratio, cd]` `ape` table
/// (1008). Grid-strided, so any grid is correct; this one covers the larger loop in one pass, and
/// the kernel strides by exactly the threads launched.
///
/// # Safety
/// The `launch_kv_compress_deposit` contract.
pub(crate) unsafe fn rivoli_kv_compress_deposit(a: Deposit, stream: *mut c_void) -> i32 {
    let p = a.p;
    let checked = compress_guard(&p).and_then(|()| {
        if a.s <= 0 {
            return Err(1005);
        }
        let rem = a.s % p.ratio;
        if a.slot0 < 0 || a.slot0.wrapping_add(rem) > p.ratio {
            return Err(1008);
        }
        elementwise(i64::from(p.ratio.max(rem)) * i64::from(p.cd))
    });
    let threads = checked.map_or(0, |g| u64::from(g.blocks) * u64::from(g.threads));
    unsafe {
        run(
            &KV_COMPRESS_DEPOSIT,
            checked,
            stream,
            &Strided { a, threads },
        )
    }
}

/// The geometry, then `nblk > 0` (1006 — NOT a no-op success: the caller believed a block would
/// be emitted). One block per compressed output block.
///
/// # Safety
/// The `launch_kv_compress_prefill` contract.
pub(crate) unsafe fn rivoli_kv_compress_prefill(
    nblk: Rows,
    a: CompressPrefill,
    stream: *mut c_void,
) -> i32 {
    let checked = compress_guard(&a.p).and_then(|()| {
        if nblk.0 <= 0 {
            Err(1006)
        } else {
            Ok(pool(nblk))
        }
    });
    unsafe { run(&KV_COMPRESS_PREFILL, checked, stream, &a) }
}

/// The geometry, `start_pos > 0` (1007; 0 is the prefill path), then a position that COMPLETES a
/// window (1009). ONE block, always: its barriers order the pool against the window slide.
///
/// # Safety
/// The `launch_kv_compress_decode` contract.
pub(crate) unsafe fn rivoli_kv_compress_decode(a: CompressDecode, stream: *mut c_void) -> i32 {
    let checked = compress_guard(&a.p).and_then(|()| {
        if a.start_pos <= 0 {
            Err(1007)
        } else if a.start_pos.wrapping_add(1) % a.p.ratio != 0 {
            Err(1009)
        } else {
            Ok(pool(Rows(1)))
        }
    });
    unsafe { run(&KV_COMPRESS_DECODE, checked, stream, &a) }
}
