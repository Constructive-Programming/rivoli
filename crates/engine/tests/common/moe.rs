//! MoE expert-range dispatch scaffolding, shared by `kernel_moe.rs` and
//! `kernel_moe_artifact.rs`.
//!
//! **MOVED HERE from `kernel.rs` (via `kernel_moe.rs`) 2026-08-15**, when the MoE oracles
//! split into `kernel_moe.rs` (synthetic quantized fixtures, runs on any GPU) and
//! `kernel_moe_artifact.rs` (the shipped `.i4` set and its fp8 checkpoint, skips loudly
//! without them). `i4_launch_drain` is what both drive, and it drags in the whole chain
//! below — a second copy of any of it is what `build.rs`'s duplication gate is for.
//!
//! It sits beside [`MoeRange`], which was already in the parent for the same reason, and
//! follows [`super::GemmBf16`]/[`super::gemm_bf16_launch`]: a device-typed launch operand and
//! its wrapper live here GATED, so the featureless registry binaries (`docs`, `invariants`)
//! never see them. **One `#[cfg]` on the module rather than nine on the items** — nine copies
//! of the attribute is nine identical token runs, and jscpd reported two of the structs as a
//! clone of each other on the strength of it.
//!
//! **Split out of `common/mod.rs` 2026-08-15** under the file-size gate — the module became a
//! file, so the single `#[cfg]` the paragraph above argues for is now on the `pub mod moe;`
//! line in `mod.rs`. Bodies and comments verbatim; `use common::moe::{…}` is untouched.

use super::{MoeRange, dev, f32b, f32v};
use rivoli_backend::gpustream::HipStream;
use rivoli_backend::hip::ExpertDesc;
use rivoli_engine::device::DeviceBuf;

/// The four per-dispatch buffers of one MoE expert range: the token rows, the gate weights,
/// the per-expert `h` staging, and the fixed-point accumulator.
///
/// Descriptors, codebooks and geometry are fixed for a whole test; these four are what a
/// batched arm swaps. Bundled so the two batching tests can each drive their range through
/// a closure taking ONE operand rather than the same five-parameter list written twice.
pub struct MoeIo<'a> {
    x: &'a DeviceBuf,
    w: &'a DeviceBuf,
    h: &'a mut DeviceBuf,
    acc: &'a mut DeviceBuf,
}

impl<'a> MoeIo<'a> {
    pub fn new(
        x: &'a DeviceBuf,
        w: &'a DeviceBuf,
        h: &'a mut DeviceBuf,
        a: &'a mut DeviceBuf,
    ) -> Self {
        Self { x, w, h, acc: a }
    }

    /// The four device addresses, in launcher order. Consuming, because two of them are
    /// unique borrows and the address outlives the reborrow that produced it.
    pub fn ptrs(self) -> (*const f32, *const f32, *mut f32, *mut u64) {
        let (x, w) = (self.x.ptr() as *const f32, self.w.ptr() as *const f32);
        let acc = self.acc.ptr_mut() as *mut u64;
        (x, w, self.h.ptr_mut() as *mut f32, acc)
    }
}

/// What every int4 expert-range dispatch holds fixed for a whole test: the uploaded
/// descriptor array and the stream it runs on. Only [`MoeIo`] and `nrow` change between arms.
pub struct MoeCtx<'a> {
    descs: &'a DeviceBuf,
    stream: &'a HipStream,
}

impl<'a> MoeCtx<'a> {
    pub fn new(descs: &'a DeviceBuf, stream: &'a HipStream) -> Self {
        Self { descs, stream }
    }
}

/// `moe_expert_range_i4` over experts `[g.e_start, g.e_end())`.
pub fn expert_range_i4(io: MoeIo<'_>, cx: &MoeCtx<'_>, g: MoeRange, nrow: usize) {
    let (x, w, h, acc) = io.ptrs();
    let d = cx.descs.ptr() as *const ExpertDesc;
    let (hidden, inter, st) = (g.hidden, g.inter, cx.stream.raw());
    // SAFETY: `x` is `nrow` rows of [hidden], `w` is [e_count·nrow], `h` [e_count·nrow·inter]
    // and `acc` `nrow` rows of [hidden] u64; the stream is live for the call.
    unsafe {
        rivoli_backend::hip::launch_moe_expert_range_i4(
            x, hidden, inter, g.e_start, g.e_count, d, w, h, acc, nrow, st,
        )
    }
    .expect("moe_expert_range_i4");
}

/// The drain's two buffers: the fixed-point accumulator it consumes and the f32 destination
/// it writes. Always allocated and drained as a pair, and both are unique borrows.
pub struct Drain<'a> {
    pub out: &'a mut DeviceBuf,
    pub acc: &'a mut DeviceBuf,
}

impl<'a> Drain<'a> {
    pub fn new(out: &'a mut DeviceBuf, acc: &'a mut DeviceBuf) -> Self {
        Self { out, acc }
    }
}

/// One `moe_acc_drain` over row `row` of the accumulator.
///
/// `row` is what lets a batched arm drain its rows with the same launch the single-row arms
/// use — the drain itself is always single-row.
pub fn drain(d: Drain<'_>, row: usize, hidden: usize, stream: &HipStream) {
    // SAFETY: `row` is inside both buffers, which every caller sizes for it; the stream is
    // live for the call.
    unsafe {
        rivoli_backend::hip::launch_moe_acc_drain(
            d.out.ptr_mut().add(row * hidden * 4) as *mut f32,
            d.acc.ptr_mut().add(row * hidden * 8) as *mut u64,
            hidden,
            1,
            1.0,
            stream.raw(),
        )
    }
    .expect("moe_acc_drain");
}

/// The descriptor ARRAY on device — the addresses themselves, uploaded verbatim.
///
/// **Generic over the descriptor since 2026-08-16**, when M8's `kernel_v4_moe.rs` needed the
/// same upload for `ExpertDescF4` — six `*const u8` where [`ExpertDesc`] is four wider
/// pointers. A second body would have been a second `size_of_val`/`from_raw_parts` pair, which
/// is what `build.rs`'s duplication gate is for, and the arithmetic reads no field: it is the
/// slice's own byte span either way.
///
/// It does NOT weaken the type separation `ExpertDescF4`'s doc argues for. That separation
/// lives at the CONSTRUCTION sites, which stay type-checked; this is already downstream of the
/// `buf.ptr() as *const _` cast that doc records as compiling either way.
pub fn desc_buf<T: Copy>(descs: &[T]) -> DeviceBuf {
    // SAFETY: both descriptor types are plain pointers, and the span is exactly the slice's
    // own bytes.
    dev(unsafe {
        std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(descs))
    })
}

/// One MoE expert's two matrix dims. Both are bare `usize` and each is plausible in the
/// other's position at every launcher, oracle and buffer size that takes the pair.
#[derive(Clone, Copy)]
pub struct Dims {
    pub hidden: usize,
    pub inter: usize,
}

impl Dims {
    pub fn new(hidden: usize, inter: usize) -> Self {
        Self { hidden, inter }
    }
}

/// The three MoE destination buffers for `nrow` token rows: per-expert `h` staging, the
/// fixed-point accumulator, and the f32 output.
///
/// ONE u64 accumulator row per token, not `e` partial rows; the output starts at zero
/// because the drain ADDS into it — it is the residual add.
pub fn moe_bufs(e: usize, nrow: usize, d: Dims) -> (DeviceBuf, DeviceBuf, DeviceBuf) {
    let z = |n: usize| dev(&vec![0u8; n]);
    (
        z(e * nrow * d.inter * 4),
        z(nrow * d.hidden * 8),
        z(nrow * d.hidden * 4),
    )
}

/// Launch `[0, descs.len())` int4 experts ONE AT A TIME and drain — the tail every int4 MoE
/// test shares, extracted because `gpu_i4_moe` and `gpu_i4_expert` had it verbatim and the
/// duplication gate said so. Per expert rather than one range: bit-identical by
/// `moe_expert_range`'s own argument (`e = e_start + row / inter`, every row independent).
///
/// The caller must keep the buffers the descriptors point INTO alive across this call.
pub fn i4_launch_drain(descs: &[ExpertDesc], x: &[f32], w: &[f32], d: Dims) -> Vec<f32> {
    let e = descs.len();
    let (descb, xb, wb) = (desc_buf(descs), dev(&f32b(x)), dev(&f32b(w)));
    let (mut hbuf, mut pbuf, mut obuf) = moe_bufs(e, 1, d);
    let stream = HipStream::new().expect("stream");
    let cx = MoeCtx::new(&descb, &stream);
    for k in 0..e {
        let io = MoeIo::new(&xb, &wb, &mut hbuf, &mut pbuf);
        expert_range_i4(io, &cx, MoeRange::new(d.hidden, d.inter, k, 1), 1);
    }
    drain(Drain::new(&mut obuf, &mut pbuf), 0, d.hidden, &stream);
    rivoli_backend::hip::device_sync().expect("sync");
    f32v(&obuf.copy_out().expect("out"))
}
