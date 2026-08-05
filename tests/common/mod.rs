//! The backend-neutral half of the kernel-oracle scaffolding, shared by `tests/vk.rs`,
//! `tests/kernel.rs`, `tests/docs.rs` and `tests/invariants.rs`.
//!
//! It was copy-pasted per file until 2026-07-30, and the copies had already started to
//! drift: two spellings of the same `Lcg` bug note, two `assert_close` bodies with the
//! same tolerance, and `f16b`/`u16b` present in one file and re-derived in the other.
//! Anything that touches a device TYPE stays in the test file that owns it — `dev` is
//! `DeviceBuf` under HIP and `Buf` under Vulkan, and that difference is the point of
//! having two files.
//!
//! `dead_code` is allowed because this module is compiled into EACH test binary, and
//! neither uses every helper. The alternative is per-consumer cfg gates on a test
//! utility, which is more machinery than the warning is worth.
#![allow(dead_code)]
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

use rivoli::v4compress::LayerKind;
use rivoli::v4oracle::forward::{CompressorW, IndexerW};
use rivoli::v4oracle::numerics::{bf16_decode, bf16_encode};
use rivoli::v4oracle::weights::{Checkpoint, NamedRng, V4Config};
use std::path::Path;

/// Every file under `root` with extension `ext`, recursively. Unsorted.
///
/// WALK, do not list files. The two registry checks that call this had each grown their
/// own copy — `docs.rs` recursive, `invariants.rs` an explicit stack — and both exist for
/// the same reason: the hand-maintained path list `invariants.rs` replaced named five
/// files, and moving `hybrid`/`gpustream`/`pin` into subsystem folders on 2026-07-31
/// silently emptied it, after which the registry reported every INV-n as untested. A
/// coverage check keyed on a remembered list fails in the direction that looks like a real
/// regression, which costs more than the walk.
pub fn walk(root: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let p = e.path();
            match p.is_dir() {
                true => stack.push(p),
                false if p.extension().is_some_and(|x| x == ext) => out.push(p),
                false => {}
            }
        }
    }
    out
}

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
            .map(|&x| rivoli::math::f32_to_f16(x))
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
/// `wkv`/`wgate` — `v4_compress_kernel.rs` at the real checkpoint and `v4_attn.rs` at the toy
/// — and `build.rs`'s duplication gate sees a second copy.
pub fn bf16_rows(w: &rivoli::v4oracle::weights::WMat) -> Vec<u16> {
    let (rows, cols) = (w.rows(), w.cols());
    let mut out = Vec::with_capacity(rows * cols);
    let mut buf = Vec::new();
    for r in 0..rows {
        w.row(r, &mut buf);
        for &v in &buf {
            let code = rivoli::math::f32_to_bf16(v);
            assert_eq!(
                rivoli::math::bf16_to_f32(code),
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
/// consumer indexes — `v4c_finish_row` on the device and `Io::freqs` in `attn::v4`.
pub fn flat_freqs(t: &[(f32, f32)]) -> Vec<f32> {
    t.iter().flat_map(|&(c, s)| [c, s]).collect()
}

/// Little-endian bytes → f32 vec, the inverse of [`f32b`] for readback.
///
/// Delegates to the engine's own decoder rather than repeating it: an oracle that read
/// bytes back differently from the code under test could agree with itself while both were
/// wrong about the file format.
pub fn f32v(b: &[u8]) -> Vec<f32> {
    rivoli::artifact::quant::read_f32(b)
}

/// Report the max error AND the threshold it was compared against. Printing BOTH is the
/// point: a green oracle that passed on 100x of headroom looks exactly like one that passed
/// on 2x, and only one of them is evidence of anything.
pub fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let (err, tol) = report(want, got, label);
    assert!(
        err <= tol,
        "{label}: err={err:.3e} > tol={tol:.3e} max={:.3e}",
        max_abs(want)
    );
}

/// The largest magnitude in a slice — the scale every tolerance in this suite is stated
/// against.
///
/// Extracted because a second tolerance FORMULA now exists: `tests/v4_kernel.rs` bounds
/// relative to the bf16 quantum instead of [`err_tol`]'s `1e-3·max + 1e-3`, whose absolute
/// floor is 5% of the signal at that fixture's scale. The formulas differ on purpose; the
/// SCALE they are stated against must not, and three copies of this fold were the duplicate
/// the gate found.
pub fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

/// [`err_tol`] plus the comparison line, returning the pair so the caller decides what a
/// failure means: [`assert_close`] panics, `vk.rs`'s `Shapes::close` records and keeps
/// going. The PRINT is what they share, and a second copy of the format string is a second
/// format.
pub fn report(want: &[f32], got: &[f32], label: &str) -> (f32, f32) {
    let (err, tol) = err_tol(want, got);
    report_line(label, err, tol, max_abs(want))
}

/// [`report`] against a tolerance RELATIVE to the largest expected element, for callers
/// whose signal is too small for [`err_tol`]'s `1e-3` absolute floor to mean anything —
/// `tests/v4_kernel.rs`, where one routed MoE layer's output is ~2e-2 and that floor would
/// be 5% of it.
///
/// Takes the ratio and computes the metric itself. An earlier version took `(err, tol, mx)`
/// — three interchangeable `f32`s, where swapping the first two turns the caller's
/// `err <= tol` into `tol <= err`: a gate that goes green on every failure. That is this module's
/// own argument about six bare `usize` in a row, made about
/// `f32`.
pub fn report_rel(want: &[f32], got: &[f32], label: &str, rel: f32) -> (f32, f32) {
    let mx = max_abs(want);
    report_line(label, max_err(want, got), rel * mx, mx)
}

/// The comparison LINE, given an error and whatever bound the caller holds it to. Named for
/// what it emits: it was `report_margin` until 2026-08-05 and the margin is gone. Private:
/// [`report`] and [`report_rel`] are the two ways in, and a third caller would be a third
/// tolerance with no argument attached to it.
///
/// **Prints `err` and `tol` side by side, not a ratio.** It printed `margin = tol/err`
/// until 2026-08-05, and that number is pathological at both ends of its range: a bit-exact
/// result rendered as `margin=532543503195029799199619132512272384.0x`, which reads as
/// corruption rather than as the best possible outcome, and a deliberate-break test — where
/// passing means err EXCEEDS tol — rendered as `margin=0.0x`, which reads as failure beside
/// a green test. Two numbers the reader compares themselves have neither pathology, and the
/// distance is still on the page.
fn report_line(label: &str, err: f32, tol: f32, mx: f32) -> (f32, f32) {
    println!("{label}: err={err:.3e} tol={tol:.3e} max={mx:.3e}");
    (err, tol)
}

/// `(max abs error, tolerance)` for a want/got pair — the shared arithmetic behind
/// [`assert_close`] and `vk.rs`'s multi-shape `Shapes::close`, which records instead of
/// panicking. Two copies of a tolerance formula is two tolerances.
pub fn err_tol(want: &[f32], got: &[f32]) -> (f32, f32) {
    (max_err(want, got), 1e-3 * max_abs(want) + 1e-3)
}

/// The largest absolute disagreement between two slices — the error metric every
/// comparison in this suite uses, whatever tolerance it is held to.
fn max_err(want: &[f32], got: &[f32]) -> f32 {
    want.iter().zip(got).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
}

/// `n` positive per-block scales, `|f·0.1| + 0.01`. Every fp8 oracle in both backend files
/// draws them this way, and the 0.01 floor is load-bearing rather than tidy: a tile whose
/// scale rounds to zero makes the comparison for that tile vacuous, and `assert_close`'s
/// relative tolerance would not show it.
pub fn block_scales(r: &mut Lcg, n: usize) -> Vec<f32> {
    (0..n).map(|_| (r.f() * 0.1).abs() + 0.01).collect()
}

/// An fp8 GEMV case: e4m3 weights, `n_scales` block scales, the input, and the host result.
///
/// `n_scales` is the caller's, not computed here, because the two backends spell the scale
/// grid differently (`i_dim / block` against `i_dim.div_ceil(block)`) and unifying that
/// would change one of them for a shape neither currently tries. The DRAW ORDER — weights,
/// scales, x — is the part that has to be shared: it is what makes a seed mean the same
/// data on both sides.
pub fn gemv_fp8_case(
    r: &mut Lcg,
    o_dim: usize,
    i_dim: usize,
    block: usize,
    n_scales: usize,
) -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let packed: Vec<u8> = (0..o_dim * i_dim)
        .map(|_| rivoli::math::f32_to_e4m3(r.f()))
        .collect();
    let scale = block_scales(r, n_scales);
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();
    let mut want = vec![0.0f32; o_dim];
    rivoli::artifact::quant::matvec_fp8(&mut want, &x, &packed, &scale, i_dim, block);
    (packed, scale, x, want)
}

/// Random int8 weights and their per-row scales, drawn the way every int8 oracle here
/// draws them.
///
/// The `1e-4` floor is load-bearing for the same reason [`block_scales`]'s `0.01` is: a row
/// whose scale rounds to zero makes that row's comparison vacuous, and a relative tolerance
/// would not show it. The DRAW ORDER — weights, then scales — is what makes a seed mean the
/// same data at both call sites.
pub fn i8_weights(r: &mut Lcg, o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    let packed: Vec<u8> = (0..o_dim * i_dim)
        .map(|_| (r.f() * 127.0) as i8 as u8)
        .collect();
    let scale: Vec<f32> = (0..o_dim).map(|_| (r.f() * 0.01).abs() + 1e-4).collect();
    (packed, scale)
}

/// `matvec_i8` into a fresh `o_dim` vector. Returned rather than written through an
/// out-param so the caller binds it in one line; the two int8 oracles generate their
/// weights differently and share only this step.
pub fn want_i8(x: &[f32], packed: &[u8], scale: &[f32], o_dim: usize, i_dim: usize) -> Vec<f32> {
    let mut want = vec![0.0f32; o_dim];
    rivoli::artifact::quant::matvec_i8(&mut want, x, packed, scale, o_dim, i_dim);
    want
}

/// The kv_b geometry both MLA launchers take.
///
/// Six bare `usize` in a row, every one of them plausible in any other's position, spelled
/// in an oracle, a launch wrapper and a guard closure PER BACKEND — five copies of the same
/// order, and a transposed pair would have moved the oracle and the kernel together. Pure
/// dimensions, so it belongs here rather than beside either backend's buffer type.
#[derive(Clone, Copy)]
pub struct Mla {
    pub h: usize,
    /// The q head stride. `mla_value_fp8` never reads q, so [`Mla::value_dims`] leaves this
    /// zero — cheaper than a second five-field shape whose only difference from this one is
    /// a field nothing reads.
    pub qh: usize,
    pub nope: usize,
    pub vh: usize,
    pub kvl: usize,
    pub block: usize,
}

impl Mla {
    pub fn new(h: usize, qh: usize, nope: usize, vh: usize, kvl: usize, block: usize) -> Self {
        Self {
            h,
            qh,
            nope,
            vh,
            kvl,
            block,
        }
    }

    /// `mla_value_fp8`'s shape: it takes no `qh`.
    pub fn value_dims(h: usize, nope: usize, vh: usize, kvl: usize, block: usize) -> Self {
        Self::new(h, 0, nope, vh, kvl, block)
    }

    /// kv_b's full row count, `h·(nope + vh)`.
    pub fn rows(self) -> usize {
        self.h * (self.nope + self.vh)
    }

    /// The two launcher guards both CPU oracles restate. An oracle that accepted a shape
    /// the launcher rejects would be checking the kernel against a case it can never run.
    ///
    /// `qh` is deliberately absent: `value_dims` leaves it zero and `mla_value_fp8` never
    /// reads it, so the absorb oracle asserts it separately.
    pub fn assert_guarded(self) {
        let (h, nope, vh) = (self.h, self.nope, self.vh);
        let (kvl, block) = (self.kvl, self.block);
        assert!(
            h > 0 && nope > 0 && vh > 0 && kvl > 0 && block > 0,
            "guard 1001"
        );
        assert!(
            block.is_power_of_two(),
            "guard 1003: blk_shift needs a power-of-two tile"
        );
    }
}

/// The MLA attention's shape.
///
/// Five `usize` and an f32 that travel together through the split planner, the tile
/// widener, the CPU reference and the dispatch — and the reference and the dispatch take
/// the SAME six, so every test spelled them twice. A transposed pair would have moved both
/// sides identically and the comparison would still have agreed.
#[derive(Clone, Copy)]
pub struct Att {
    pub h: usize,
    pub nr: usize,
    pub kvl: usize,
    pub rope: usize,
    pub n_blocks: usize,
    pub scale: f32,
}

impl Att {
    /// `n_blocks` is not a free parameter — the fp8 latent cache carries one block scale per
    /// 128 latent dims, so it FOLLOWS from `kvl`, and every test derived it the same way.
    /// Deriving it once removes the only way a reference and a launcher could have been
    /// handed different block-scale strides for the same cache.
    pub fn new(h: usize, nr: usize, kvl: usize, rope: usize, scale: f32) -> Self {
        Self {
            h,
            nr,
            kvl,
            rope,
            n_blocks: kvl / 128,
            scale,
        }
    }
}

/// One MoE dispatch's geometry: the two matrix dims and the half-open expert range
/// `[e_start, e_start + e_count)`.
///
/// The same four, in the same order, in `moe_expert_range`'s wrapper and in both of the
/// VQ oracles that check it — three copies per backend of a list whose middle two entries
/// are interchangeable to the type checker.
#[derive(Clone, Copy)]
pub struct MoeRange {
    pub hidden: usize,
    pub inter: usize,
    pub e_start: usize,
    pub e_count: usize,
}

impl MoeRange {
    pub fn new(hidden: usize, inter: usize, e_start: usize, e_count: usize) -> Self {
        Self {
            hidden,
            inter,
            e_start,
            e_count,
        }
    }

    /// One past the last expert this range writes — the oracles size their staging by it.
    pub fn e_end(self) -> usize {
        self.e_start + self.e_count
    }
}

/// The oracles' deterministic input source.
pub struct Lcg(pub u64);

impl Lcg {
    /// Uniform in [-1, 1).
    ///
    /// `>> 32`, not `>> 33`. The old shift kept only 31 bits, so dividing by `u32::MAX`
    /// gave [0, 0.5) and `*2 - 1` gave [-1, 0) — **every sample negative**, for the whole
    /// life of both test files. In a matvec oracle that makes every `x[i]*w[i]` product
    /// positive, so the partial sums GROW instead of cancelling: `mx` inflates, the
    /// `1e-3 * mx` relative tolerance inflates with it, and the oracles were passing on
    /// roughly two orders of magnitude of headroom. It also meant no oracle here had ever
    /// exercised floating-point cancellation — the only regime where summation order
    /// matters, and the entire reason the kernels reduce with a fixed shuffle ladder
    /// (`__shfl_down` / `wave_sum`) instead of an atomic.
    pub fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------------------
// V4-Flash checkpoint scaffolding, shared by `v4_compress_probe.rs` and
// `v4_compress_kernel.rs`
// ---------------------------------------------------------------------------------------
//
// These moved here from `v4_compress_probe.rs` when the kernel test needed the same loader:
// both drive the SAME two compressors (layer 2 at ratio 4, layer 3 at ratio 128), one
// against the oracle alone and one against the GPU, and a second copy of `compressor_w`
// would be a second set of shape assertions that could drift apart while both stayed green.
// `build.rs`'s duplication gate watches `tests/`, so it would also be a build error.
//
// No device type appears here, which is this module's rule: `DeviceBuf` lives in the rocm
// test file that owns it.

pub const CKPT: &str = "/var/db/rivoli/deepseek-v4-flash-0731";
/// `bin/v4-oracle`'s `PROMPT` tokenizes to 13 ids — the length every hole is keyed to.
pub const EMIT_LEN: usize = 13;
/// Two whole ratio-128 blocks.
///
/// It does NOT exercise a block-to-block state carry, which an earlier version of this
/// comment claimed: at ratio 128 `overlap` is false and `256 % 128 == 0`, so both the
/// `overlap && cutoff >= ratio` and the `remainder > 0` state writes are skipped and
/// prefill pools every block independently. Two reviewers disproved the claim the same way
/// — substitute zero-length `kv_state`/`score_state` and the output is bit-identical.
///
/// Two blocks still earn their keep, for the reason that survives: the blocks are RoPE'd at
/// `freqs_cis[0:256:128]`, i.e. positions 0 and 128, so a wrong per-block rope position or
/// unflatten stride is observable here and would be hidden by a single block (position 0,
/// where the rotation is the identity).
pub const PROBE_LEN: usize = 256;
/// A ratio-128 prefill with a REMAINDER, which is the only prefill path that writes the
/// compressor state — and the state the decode branch then reads.
pub const PROBE_REMAINDER_LEN: usize = 300;
/// Ratio-128 decode completes its first block here: `(start_pos + 1) % 128 == 0`.
pub const RATIO_128_FIRST_DECODE_BLOCK: usize = 127;

pub fn checkpoint() -> Option<Checkpoint> {
    if !Path::new(CKPT).join("model.safetensors.index.json").exists() {
        eprintln!("SKIP: no checkpoint at {CKPT}");
        return None;
    }
    Some(Checkpoint::open(Path::new(CKPT)).expect("opening checkpoint"))
}


/// One layer's `attn.compressor.*`, at `head_dim` and `rotate` set by which compressor it is.
///
/// Loading these directly rather than through `bin/v4-oracle`'s `load_layer` is the whole
/// point: `load_layer` also pulls the layer's routed experts, which is 3.4 GB per layer, and
/// none of it is read by `Oracle::compressor`.
pub fn compressor_w(ck: &Checkpoint, prefix: &str, ratio: usize, d: usize, rotate: bool) -> CompressorW {
    let kind = LayerKind::from_ratio(ratio);
    let cw = CompressorW {
        ratio,
        overlap: kind.overlap(),
        d,
        rotate,
        ape: ck.get(&format!("{prefix}.ape")).unwrap().to_f32().unwrap(),
        wkv: ck.dense(&format!("{prefix}.wkv.weight")).unwrap(),
        wgate: ck.dense(&format!("{prefix}.wgate.weight")).unwrap(),
        norm: ck.get(&format!("{prefix}.norm.weight")).unwrap().to_f32().unwrap(),
    };
    // The shape trap from the S2c brief, asserted rather than assumed: `ape` is
    // [ratio, coff*d], so [4, 1024] at ratio 4 (coff 2) and [128, 512] at ratio 128 (coff 1).
    // A loader that inferred the width from `d` alone gets 512, which is WRONG on the ratio-4
    // attention compressor and right on ratio 128 -- an earlier version of this comment had
    // that backwards. The error is a silent misindex, not a length mismatch, because both
    // widths are 512-multiples.
    assert_eq!(
        cw.ape.len(),
        ratio * kind.coff() * d,
        "{prefix}: ape is [ratio, coff*d] = [{ratio}, {}]",
        kind.coff() * d
    );
    // `[out, in]`, the torch `Linear` convention `Oracle::linear` reads: rows are the
    // projection width, cols the model dim. Asserting `cols` here instead passed on L2 by
    // coincidence of both being 4096-adjacent and is the axis mix-up worth pinning.
    assert_eq!(cw.wkv.rows(), kind.coff() * d, "{prefix}: wkv projects TO coff*d");
    assert_eq!(cw.wgate.rows(), kind.coff() * d, "{prefix}: wgate matches wkv");
    assert_eq!(cw.wkv.cols(), cw.wgate.cols(), "{prefix}: both read the same model dim");
    assert_eq!(cw.norm.len(), d, "{prefix}: norm is over head_dim, not coff*head_dim");
    cw
}

/// A deterministic bf16 activation block, `[n, dim]`.
/// One layer's `attn.indexer.*`, as [`IndexerW`].
///
/// Lifted here from `v4_compress_probe.rs` when `v4_indexer_kernel.rs` became a second
/// consumer and `build.rs`'s duplication gate found the copy. The comment that moved with it
/// is the load-bearing part: `wq_b` is fp8 on disk (it ships a `.scale`), unlike
/// `weights_proj`, which is bare bf16 — and V4's `Indexer` has **no `wk` and no `k_norm`**.
/// Guessing GLM's names here is what broke S1a's first convert.
///
/// `rotate = true` on the nested compressor: the indexer's own Hadamard-and-fp4 finish where
/// the attention compressor partially fp8-quantizes. Same class, different arithmetic.
pub fn indexer_w(ck: &Checkpoint, layer: usize, c: &V4Config) -> IndexerW {
    IndexerW {
        wq_b: ck.fp8(&format!("layers.{layer}.attn.indexer.wq_b.weight")).unwrap(),
        weights_proj: ck
            .dense(&format!("layers.{layer}.attn.indexer.weights_proj.weight"))
            .unwrap(),
        compressor: compressor_w(
            ck,
            &format!("layers.{layer}.attn.indexer.compressor"),
            4,
            c.index_head_dim,
            true,
        ),
    }
}

pub fn probe(name: &str, n: usize, dim: usize) -> Vec<f32> {
    let mut r = NamedRng::new(name);
    (0..n * dim).map(|_| bf16_decode(bf16_encode(r.unit()))).collect()
}


/// Names from `names` for which `present` is false — the "coverage census" shape shared by
/// every source-scanning test here.
///
/// Factored 2026-08-05 because `jscpd` refused the second copy, and it was right: this idiom
/// had reached `tests/kernel_coverage.rs` and `tests/v4_oracle.rs` independently, which is
/// the same drift this module's header records for `assert_close` and `f16b`.
///
/// The caller keeps its own `assert!` and its own message. That is deliberate — the message
/// is the whole value of a census failure (*which* names, and what the reader should do
/// about them), and a shared message would have to be generic enough to be useless. Only the
/// set arithmetic is common.
pub fn absent<S: AsRef<str>>(names: &[S], present: impl Fn(&str) -> bool) -> Vec<&str> {
    names
        .iter()
        .map(AsRef::as_ref)
        .filter(|n| !present(n))
        .collect()
}
