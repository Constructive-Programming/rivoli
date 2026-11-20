//! The device tripwire for `DeviceBuf::copy_in_at`'s promise: a blocking pageable H2D
//! `hipMemcpy` has landed by the time it returns, so a kernel enqueued next on a
//! `hipStreamNonBlocking` stream reads the new bytes. `stage_batch` (`glm/mlp.rs`) relies on this
//! with no event. The citation is in `copy_in_at`'s doc; the full argument is in
//! `docs/investigations/rust-kernels.md` §5 "stage_batch ordering".
//!
//! **Check:** each iteration uploads fresh data, then (with no other sync) folds the destination
//! with `hash_rows` on a non-blocking stream. If the device fold differs from the host fold, the
//! kernel read bytes that had not landed yet.
//!
//! **Standing red proof, run every time:** the same loop, with the upload replaced by an
//! asynchronous copy on the null stream that waits behind a long null-stream kernel and is never
//! waited on, has to see stale reads at every size. Otherwise this check could not detect a
//! missing ordering, and its green would mean nothing.
#![cfg(feature = "rocm")]
// A failed alloc, readback or stream creation in a device test should abort the test loudly;
// there is no caller to propagate it to.
#![allow(clippy::expect_used)]

use std::ffi::c_void;

use rivoli_backend::{gpustream::HipStream, hiprt};
use rivoli_engine::device::{DeviceBuf, as_le_bytes};

mod common;
use common::{Lcg, back, hash_fold, zeros};

/// Bytes per upload. The first three are the uploads `stage_batch` actually makes on GLM:
/// `wexpert` at one row (9 f32), and `descs` at nrow 1 and nrow 2 (9·48 and 17·48 bytes). All
/// three take the CLR's shader-copy path. After them come the edges of the CLR's other copy
/// paths, plus the size just past each edge (`rocsettings.cpp`, `rocblit.cpp`): up to 16 KiB a
/// staged shader copy, up to 64 KiB a staged SDMA copy, above that SDMA from the source pinned
/// in place.
const SIZES: [usize; 9] = [
    36,
    432,
    816,
    16 << 10,
    (16 << 10) + 4,
    64 << 10,
    (64 << 10) + 4,
    1 << 20,
    16 << 20,
];

/// Roughly how many bytes each size uploads in total, so small sizes get many iterations and
/// 16 MiB gets 16. The point is to keep the host fold affordable on the dev profile.
const BYTES_PER_SIZE: usize = 256 << 20;

/// How many iterations at one size read stale bytes, out of how many ran.
#[derive(Debug)]
struct SizeRow {
    bytes: usize,
    stale: usize,
    examined: usize,
}

/// Run `upload` then a `hash_rows` read on `stream`, over and over, and count the reads whose
/// device fold disagrees with the host fold.
fn stale_reads(
    bytes: usize,
    stream: &HipStream,
    mut upload: impl FnMut(&mut DeviceBuf, &[f32]),
) -> SizeRow {
    let n = bytes / 4;
    let examined = (BYTES_PER_SIZE / bytes).clamp(16, 256);
    let mut dst = DeviceBuf::new(bytes).expect("alloc dst");
    let mut out = zeros(8);
    let mut r = Lcg(0x5EED ^ bytes as u64);
    let base: Vec<u32> = (0..n).map(|_| r.f().to_bits()).collect();
    let stale = (0..examined as u32)
        .filter(|&it| {
            // Every word differs from the previous iteration's, so a stale read moves the fold.
            let src: Vec<f32> = base.iter().map(|&b| f32::from_bits(b ^ (it + 1))).collect();
            out.copy_in_at(0, &[0u8; 8]).expect("zero the fold");
            upload(&mut dst, &src);
            // SAFETY: `dst` holds `n` f32 and `out` one zeroed u64; `back` joins the device before
            // either is touched again; `stream` is live.
            unsafe {
                hash_fold(
                    dst.ptr() as *const f32,
                    n,
                    out.ptr_mut() as *mut u64,
                    stream.raw(),
                )
            };
            let mut w = [0u8; 8];
            w.copy_from_slice(&back(&out)[..8]);
            u64::from_le_bytes(w) != rivoli_core::hash::xor_fold(&src)
        })
        .count();
    let row = SizeRow {
        bytes,
        stale,
        examined,
    };
    println!(
        "{} B: {}/{} stale reads",
        row.bytes, row.stale, row.examined
    );
    row
}

/// The engine's MoE streams come from this same constructor: non-blocking, so they are not
/// ordered against the null stream.
fn reader_stream() -> HipStream {
    HipStream::miss().expect("non-blocking stream")
}

/// The tripwire: a blocking pageable `copy_in_at` is always visible to the next kernel on a
/// non-blocking stream. If this goes red, `stage_batch`'s uploads need an event that both MoE
/// streams wait on.
#[test]
fn blocking_pageable_upload_lands_before_a_nonblocking_reader() {
    let stream = reader_stream();
    let stale: Vec<SizeRow> = SIZES
        .iter()
        .map(|&b| {
            stale_reads(b, &stream, |dst, src| {
                dst.copy_in_at(0, as_le_bytes(src)).expect("copy_in_at")
            })
        })
        .filter(|row| row.stale > 0)
        .collect();
    assert!(
        stale.is_empty(),
        "a blocking pageable hipMemcpy was NOT visible to the next non-blocking-stream kernel: \
         {stale:?}"
    );
}

/// A `hipHostMalloc` buffer, freed on drop. The red arm copies from it because only a copy
/// from pinned memory is guaranteed to still be in flight when `hipMemcpyAsync` returns.
struct Pinned(*mut c_void);

impl Drop for Pinned {
    fn drop(&mut self) {
        // SAFETY: the pointer came from `rivoli_pinned_alloc` and is freed once, here.
        unsafe { hiprt::rivoli_pinned_free(self.0) };
    }
}

/// Bytes the delay kernel folds on the null stream before each red-arm copy. This holds the
/// copy back long enough that even a 36 B copy is still queued when the reader runs on the other
/// stream, so the red proof has power at `stage_batch`'s real sizes too, not only at sizes where
/// the copy is slow by itself.
const DELAY_BYTES: usize = 64 << 20;

/// The standing red proof: remove the ordering and this check MUST see stale reads, at every
/// size. The planted upload is a delay kernel and then a `hipMemcpyAsync` from pinned memory,
/// both on the null stream, with no wait. The null stream does not order against the reader's
/// non-blocking stream.
#[test]
fn async_upload_without_a_wait_is_seen_stale() {
    let stream = reader_stream();
    let delay = zeros(DELAY_BYTES);
    let mut delay_out = zeros(8);
    let blind: Vec<SizeRow> = SIZES
        .iter()
        .map(|&bytes| {
            let pinned = Pinned(hiprt::rivoli_pinned_alloc(bytes as u64));
            assert!(!pinned.0.is_null(), "hipHostMalloc({bytes}) failed");
            stale_reads(bytes, &stream, |dst, src| {
                // SAFETY: `pinned` holds `bytes` host bytes and `src` is exactly `bytes` long;
                // the device is joined (`back`) before `pinned` is written again.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr().cast::<u8>(), pinned.0.cast(), bytes)
                };
                // SAFETY: `delay` is DELAY_BYTES/4 live f32 and `delay_out` one live u64; the
                // fold's value is never read, so it does not need zeroing.
                unsafe {
                    hash_fold(
                        delay.ptr() as *const f32,
                        DELAY_BYTES / 4,
                        delay_out.ptr_mut() as *mut u64,
                        rivoli_backend::NULL_STREAM,
                    )
                };
                // SAFETY: `dst` owns `bytes` device bytes and `pinned` `bytes` pinned bytes;
                // both stay alive past the join in `back`.
                let rc = unsafe {
                    hiprt::rivoli_memcpy_h2d_async(
                        dst.ptr_mut().cast(),
                        pinned.0,
                        bytes as u64,
                        rivoli_backend::NULL_STREAM,
                    )
                };
                assert_eq!(rc, 0, "hipMemcpyAsync");
            })
        })
        .filter(|row| row.stale == 0)
        .collect();
    assert!(
        blind.is_empty(),
        "with no ordering, these sizes saw NO stale read, so the tripwire beside this test cannot \
         detect a missing ordering at them: {blind:?}"
    );
}
