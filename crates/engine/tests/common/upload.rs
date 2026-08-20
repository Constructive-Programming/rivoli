//! Bytes in, bytes out, and the device buffers they travel through: the little-endian codecs
//! every upload and readback takes, and the `rocm`-gated `dev`/`zeros`/`back`/`ok` that spend
//! them.
//!
//! Grouped by the trip a fixture makes, not by type: `f32b` exists because a device upload
//! takes bytes, and `f32v` exists to read the same bytes back — separating the two halves is
//! how a suite ends up decoding its own uploads differently from the code under test.
//!
//! **The gate is what makes the device half safe to share, not the location** — the argument
//! and the two same-day corrections that settled it are in `mod.rs`'s header, which is where a
//! reader lands. Move a device-typed helper here UNGATED and the featureless registry binaries
//! break; that is the failure the note predicts.
//!
//! **Split out of `common/mod.rs` 2026-08-15** under the file-size gate. Bodies and their
//! comments travelled verbatim, and `mod.rs` re-exports this module with a glob, so every
//! `use common::{DeviceBuf, back, dev, f32b, f32v, ok, u16b, zeros, …}` is untouched.

/// f32 slice → little-endian bytes, the form every device upload takes.
pub fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// u16 slice → little-endian bytes (bf16 scales, fp16 codebooks, roped keys).
pub fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// f32 → fp16 bytes — the VQ codebook is uploaded fp16 (the kernel decodes `__half`),
/// while the CPU reference keeps the f32 codebook, so these oracles measure exactly the
/// fp16 codebook-rounding error against the tol.
pub fn f16b(v: &[f32]) -> Vec<u8> {
    u16b(
        &v.iter()
            .map(|&x| rivoli_core::num::f32_to_f16(x))
            .collect::<Vec<_>>(),
    )
}

/// One `WMat::Dense` weight as the bf16 codes a kernel decodes with `bf16f`.
///
/// Asserts the round-trip is EXACT rather than assuming it. The checkpoint stores these in
/// bf16 and `Checkpoint::dense` widens them to f32, so re-encoding must be lossless — if it
/// ever is not, the kernel is being fed a different matrix from the oracle and every
/// comparison downstream silently measures that instead of the arithmetic.
///
/// Here rather than in one test file because two suites now upload the compressor's
/// `wkv`/`wgate` — `kvcompress_kernel.rs` at the real checkpoint and `f4_attn.rs` at the toy
/// — and `build.rs`'s duplication gate sees a second copy.
pub fn bf16_rows(w: &rivoli_oracles::v4oracle::weights::WMat) -> Vec<u16> {
    let (rows, cols) = (w.rows(), w.cols());
    let mut out = Vec::with_capacity(rows * cols);
    let mut buf = Vec::new();
    for r in 0..rows {
        w.row(r, &mut buf);
        for &v in &buf {
            let code = rivoli_core::num::f32_to_bf16(v);
            assert_eq!(
                rivoli_core::num::bf16_to_f32(code),
                v,
                "compressor weight row {r} is not bf16-exact: the oracle and the kernel \
                 would be reading different numbers"
            );
            out.push(code);
        }
    }
    out
}

/// `(cos, sin)` pairs flattened to the `[pos][2*i], [pos][2*i+1]` layout every V4 rotary
/// consumer indexes — `compress_finish_row` on the device and `Io::freqs` in `attn::v4`.
pub fn flat_freqs(t: &[(f32, f32)]) -> Vec<f32> {
    t.iter().flat_map(|&(c, s)| [c, s]).collect()
}

/// Little-endian bytes → f32 vec, the inverse of [`f32b`] for readback.
///
/// Delegates to the engine's own decoder rather than repeating it: an oracle that read
/// bytes back differently from the code under test could agree with itself while both were
/// wrong about the file format.
pub fn f32v(b: &[u8]) -> Vec<f32> {
    rivoli_artifact::quant::read_f32(b)
}

/// Little-endian bytes → fixed-width words, `f` being the `from_le_bytes` that decodes one.
///
/// ONE body for both widths, and it is load-bearing rather than tidy. `chunks_exact(N)
/// .map(from_le_bytes)` is already spelled in `quant.rs::read_f32` and in `bin/convert.rs`'s
/// VQ encoder; writing [`u16v`] and [`u32v`] out as two more bodies takes the tree from
/// **0 clones to 2** under `build.rs`'s gate. Measured both ways on 2026-08-06, because a
/// reviewer proposed exactly that simplification.
///
/// What makes it worth a comment is where jscpd points: the two clones it reports are
/// `quant.rs`<->`v4oracle/weights.rs` and `quant.rs`<->`bin/convert.rs` — **neither names
/// this file.** The copies here are the members that tip an existing pair over the
/// threshold, so the gate sends you to `src/` for a duplicate you introduced in `tests/`.
fn le_words<const N: usize, T>(b: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
    b.chunks_exact(N)
        .map(|c| f(c.try_into().expect("chunks_exact yields exactly N")))
        .collect()
}

/// Little-endian bytes → u16 vec — bf16 key caches, fp16 codebooks, VQ indices.
pub fn u16v(b: &[u8]) -> Vec<u16> {
    le_words(b, u16::from_le_bytes)
}

/// Little-endian bytes → u32 vec — the non-finite flag and `index_topk`'s row set.
pub fn u32v(b: &[u8]) -> Vec<u32> {
    le_words(b, u32::from_le_bytes)
}

// ---------------------------------------------------------------------------------------
// Device scaffolding. See the CORRECTED note in the PARENT module's header (`mod.rs`, which
// is where a reader lands) for why it is no longer kept out of here. Re-anchored 2026-08-15
// with the split; the note itself did not move, because the rule it settles is the whole
// module's, not this file's.
// ---------------------------------------------------------------------------------------

// (The Glimmer-config-typed fixture helper that lived here arrives at M7 with its
// consumer tests; its parameter type is not ported yet.)

#[cfg(feature = "rocm")]
/// The four device addresses a GQA-family attend launch takes that are not shapes: three read
/// operands and one destination.
///
/// **HOISTED 2026-08-17.** `kernel_glimmer_attend.rs` spelled it as `GqaIo` and M17c's
/// `kernel_glimmer_block_attend.rs` spelled the same four fields, which `build.rs`'s duplication
/// gate reported as a 45-token cross-file clone the moment the second existed. The argument for
/// bundling is the one `GqaIo` carried and it survives the move: the launchers take thirteen
/// arguments, **four of them interchangeable raw addresses**, and every oracle spells the list
/// twice — once for the value arms and once for the guard table — so a transposed pair would move
/// both copies together while both stayed green.
///
/// `kernel_attend.rs::AttIo` is deliberately NOT this type: it carries five `&DeviceBuf` for MLA's
/// latent operands, a different set with a different lifetime story.
#[derive(Clone, Copy)]
pub struct AttendIo {
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub out: *mut f32,
}

impl AttendIo {
    /// The addresses of four live buffers.
    ///
    /// Taking the buffers rather than the pointers is what removes the SECOND clone: every call
    /// site was spelling the same four `ptr() as *const f32` casts, and jscpd reported that too.
    /// `out` is `&mut` so the non-aliasing the kernels' `__restrict__` assumes is checked by the
    /// borrow checker at the construction site rather than argued in a comment. **A guard table
    /// that deliberately aliases all four addresses at one buffer constructs the fields directly
    /// instead** — it never reaches a launch, so aliasing there is unobservable and is the point;
    /// `kernel_glimmer_attend.rs`'s refusal table is that case and names it.
    pub fn new(
        q: &rivoli_engine::device::DeviceBuf,
        k: &rivoli_engine::device::DeviceBuf,
        v: &rivoli_engine::device::DeviceBuf,
        out: &mut rivoli_engine::device::DeviceBuf,
    ) -> Self {
        Self {
            q: q.ptr() as *const f32,
            k: k.ptr() as *const f32,
            v: v.ptr() as *const f32,
            out: out.ptr_mut() as *mut f32,
        }
    }
}

/// The ABI every GQA-family attend launcher in this tree shares.
///
/// `gqa_attend` and `gqa_block_attend` take the **same thirteen arguments in the same order** and
/// differ only in what two of them MEAN — `start_pos`/`ring_cap` against `q_offset`/`kv_len`. That
/// is worth a named type rather than two wrappers: jscpd reported the two wrappers as a clone
/// (2026-08-17) and it was right, because a shared ABI spelled twice is one place for a transposed
/// pair to hide.
pub type AttendLauncher = unsafe fn(
    *const f32,
    *const f32,
    *const f32,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    f32,
    *mut f32,
    *mut std::ffi::c_void,
) -> anyhow::Result<()>;

/// One attend launch: `dims` is `[hq, hkv, d, tq, a, win, b]` in launcher order, where `a` and `b`
/// are the two position/extent arguments whose MEANING is the launcher's.
///
/// **The dims order is spelled here and nowhere else.** Seven bare `usize` in a row is the
/// excess-argument defect at full size; the array keeps them together and leaves every call site a
/// literal a reader can check against this line. Returns the launcher's own `Result` so a guard
/// table can read a CODE while the value arms demand success.
///
/// # Safety
/// `io.q` and `io.out` are `tq * hq * d` live f32; `io.k` and `io.v` are the launcher's documented
/// extent (`ring_cap`/`start_pos + tq` slots for the causal attend, `kv_len` rows for the block
/// one) of `hkv * d` f32 each; none alias another, and all live until the next device sync. A
/// REJECTED call returns before any kernel launch and dereferences nothing, which is what lets a
/// guard table pass aliased addresses.
pub unsafe fn attend_launch(
    f: AttendLauncher,
    io: AttendIo,
    dims: [usize; 7],
    scale: f32,
) -> anyhow::Result<()> {
    let [hq, hkv, d, tq, a, win, b] = dims;
    // SAFETY: the caller's contract above. Null stream: every call site launches once and joins.
    unsafe {
        f(
            io.q,
            io.k,
            io.v,
            hq,
            hkv,
            d,
            tq,
            a,
            win,
            b,
            scale,
            io.out,
            std::ptr::null_mut(),
        )
    }
}

pub fn dev(b: &[u8]) -> rivoli_engine::device::DeviceBuf {
    let mut d = rivoli_engine::device::DeviceBuf::new(b.len().max(1)).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}

/// A fresh non-blocking stream, for the oracles that launch onto one rather than onto null.
///
/// **A real stream and not `null_mut()` is a deliberate choice each caller inherits**: the
/// launchers took a stream parameter before any oracle passed one, and a suite that only ever
/// passed null would score the arithmetic without exercising the argument. It says nothing about
/// WHICH stream an operation landed on — a launch on the null stream produces identical bytes —
/// only that the parameter is threaded.
///
/// Here rather than per file since M8: five V4 suites opened one with the same line, which is five
/// places for the failure message to drift and one clone `build.rs`'s gate reported. Guard tests
/// that reject before any launch keep passing `null_mut()` at the call, for the reason they each
/// state: there is no launch for a stream to order.
#[cfg(feature = "rocm")]
pub fn stream() -> rivoli_backend::gpustream::HipStream {
    rivoli_backend::gpustream::HipStream::new().expect("hip stream")
}

/// The device buffer type every helper here returns and every oracle file names.
///
/// Re-exported rather than imported per file: `use rivoli_engine::device::DeviceBuf;` sat
/// directly above `mod common;` in three of the four oracle files, and that five-line run of
/// boilerplate is a jscpd clone with nothing in it worth sharing. Named through the module
/// that hands you `dev()` instead, it joins the `use common::{…}` list those files already have.
// `#[allow(unused_imports)]` for the reason the module header gives `dead_code`: this compiles
// into EVERY test binary and most of them never name the buffer type. A re-export nobody in a
// given binary uses is that binary's business, not a defect — and the allow is on this ONE item
// rather than the file, so a genuinely dead `use` elsewhere still reports.
#[allow(unused_imports)]
#[cfg(feature = "rocm")]
pub use rivoli_engine::device::DeviceBuf;

/// A zeroed device buffer of `n` bytes — a kernel destination.
///
/// ZEROED rather than uninitialised, and load-bearing for the oracles that compare a
/// destination the kernel only PARTLY writes — `append_kv` fills one row of a five-row slab,
/// `index_pool_push` one block of a three-block pool. The untouched remainder is asserted
/// too, so a wrote-the-wrong-row defect shows up as a mismatch against zero rather than as
/// noise.
#[cfg(feature = "rocm")]
pub fn zeros(n: usize) -> rivoli_engine::device::DeviceBuf {
    dev(&vec![0u8; n])
}

/// One `gemm_bf16` launch's operands and dims, and the one place they are spelled.
///
/// `glimmer_fixture.rs`'s `gemv_bf16` and `glimmer_residency.rs`'s fence gate both drive this
/// kernel and jscpd matched their call blocks (2026-08-12). The gate was right about the substance
/// too: seven positional arguments where `n` and `k` are both bare `usize` is a place a mistake is
/// a wrong answer rather than a compile error — so the six that describe the operands are named
/// fields here, and [`gemm_bf16_launch`] takes this and the stream.
#[cfg(feature = "rocm")]
#[derive(Clone, Copy)]
pub struct GemmBf16 {
    pub x: *const f32,
    pub w: *const u16,
    pub out: *mut f32,
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// One `gemm_bf16` launch.
///
/// > **Two things were tried first and are recorded so they are not tried again.** Hoisting it here
/// > UNGATED is an `E0433` on the featureless build — this module compiles into `docs` and
/// > `invariants`, which are GPU-free, and `rivoli::backend` is `rocm`-gated; that is the exact
/// > failure the module header above predicts, and it reached a review rather than a run.
/// > Hoisting the CASTS at the call site instead, to make the two blocks structurally unalike and
/// > delete this helper, does NOT satisfy jscpd — measured, still 29 tokens matched.
///
/// # Safety
/// `g.x` is `g.m * g.k` live f32, `g.w` is `g.n * g.k` live u16, `g.out` is `g.m * g.n` writable
/// f32, none aliasing another, all live until the caller's next `device_sync`.
#[cfg(feature = "rocm")]
pub unsafe fn gemm_bf16_launch(g: GemmBf16, stream: *mut std::ffi::c_void) {
    // SAFETY: the caller's contract above.
    unsafe { rivoli_backend::hip::launch_gemm_bf16(g.x, g.w, g.out, g.m, g.n, g.k, stream) }
        .expect("gemm_bf16 launch");
}

/// A launch that must succeed, with the launcher named.
///
/// Every oracle here passes dims the kernel's own guards accept, so an `Err` is a guard that
/// MOVED, not a case to handle.
///
/// It exists for `fwd_kernel.rs` and `indexer_kernel.rs` specifically, which is why the
/// older oracle files still use `.expect` under their own blanket allow. Those two gained
/// the same allow with the same three-line preamble and `build.rs`'s duplication gate
/// rejected it — a clone produced by suppressing a lint instead of removing its cause. This
/// removes the cause, and `expect_used = "deny"` stays live in both files.
///
/// `{e:#}` so an `anyhow` chain prints its causes; a launcher's guard code is in the
/// innermost one and `{e}` would drop it.
#[cfg(feature = "rocm")]
pub fn ok<T>(r: anyhow::Result<T>, what: &str) -> T {
    r.unwrap_or_else(|e| panic!("{what} refused the launch: {e:#}"))
}

/// Join the device, then read a buffer back. The join is HERE rather than at each call
/// site because forgetting it reads the destination before the kernel has written it,
/// which fails as a wrong ANSWER rather than as a missing sync — the most expensive
/// possible spelling of the mistake.
#[cfg(feature = "rocm")]
pub fn back(d: &rivoli_engine::device::DeviceBuf) -> Vec<u8> {
    rivoli_backend::hip::device_sync().expect("device_sync");
    d.copy_out().expect("copy_out")
}

// **MOVED HERE from `kernel.rs` 2026-08-16**, when the MLA/attend suites left it for
// `kernel_attend.rs`: both that file and `kernel.rs` (GEMV) read a device destination back
// against a CPU oracle, and a second copy of the join is what `build.rs`'s duplication gate is
// for. It did NOT come with the MoE split a day earlier — that split's oracles read their
// destination through the fixed-point drain's own `f32v(.copy_out())` instead, so nothing there
// called this helper yet.
/// [`super::assert_close`] against a device destination, read back. The join belongs to the
/// caller — several oracles enqueue two kernels and sync once, which is how the engine runs
/// them.
#[cfg(feature = "rocm")]
pub fn assert_out(want: &[f32], got: &rivoli_engine::device::DeviceBuf, label: &str) {
    super::assert_close(want, &f32v(&got.copy_out().expect("out")), label);
}
