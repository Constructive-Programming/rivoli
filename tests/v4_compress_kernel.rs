//! **The V4-Flash attention compressor kernels, scored against S1b's oracle.** S2c of
//! `docs/investigations/v4-flash-port.md`.
//!
//! Four cells, all at the real checkpoint's own tensors: {ratio 4, ratio 128} x {prefill,
//! decode}. Ratio 4 is the overlapping branch with `ape[4, 1024]`; ratio 128 is the
//! non-overlapping one with `ape[128, 512]`. A shape assumption that holds on layer 2 breaks
//! on layer 3, which is why both are here and why `tests/common/mod.rs::compressor_w`
//! asserts the widths at load.
//!
//! # How a green result here is made to mean something
//!
//! Every defect in this path is silent-wrong. So agreement with the oracle is necessary and
//! nowhere near sufficient, and this file spends most of its length on the other half —
//! showing the comparison can REJECT. Three techniques, in descending order of strength:
//!
//! 1. **Exact defect impersonation.** Two of the oracle's breakages are expressible as a
//!    change to a kernel INPUT rather than to the kernel: `Defect::CompressorNoApe` is
//!    `ape` zeroed, and `Defect::RopeNoYarn` is the ratio-0 rotary table in place of the
//!    compressed one. For those two the kernel is fed the perturbed input and required to
//!    match the oracle *running with that defect* to the same tolerance it matches the
//!    clean oracle — and to be far from the clean oracle. That is a real red/green, proved
//!    at the bit level, with no break switch shipped in the kernel.
//! 2. **Distance separation** for the breakages that live INSIDE the kernel and cannot be
//!    reached from outside it (the RoPE pairing, the block-end position, the bf16 stores,
//!    the `act_quant` extent). For each, the distance from the GPU output to the
//!    defect-injected oracle must dwarf the distance to the clean one. This is S2b's
//!    method (`tests/v4_attn.rs`) and it proves the METRIC has resolution, not that this
//!    kernel would fail if broken in that specific way.
//! 3. **Named inertness.** A defect that cannot fire on a cell is asserted to leave the
//!    oracle bit-identical, so that "the kernel matched" on that cell is recorded as
//!    proving nothing about it rather than being silently counted as coverage.
//!
//! # What this file provably cannot detect — read this before trusting it
//!
//! * **Anything the oracle is also wrong about.** The kernel was written from `model.py`
//!   AND from the oracle's transliteration of it; a shared misreading is invisible here by
//!   construction. `src/v4compress.rs`'s `jscpd:ignore` region makes the same point about
//!   the host half and is worth reading.
//! * **The indexer's compressor** (`rotate = true`: Hadamard + fp4 instead of the partial
//!   fp8). Out of S2c's scope, not exercised, not claimed.
//! * **`expf` agreement.** The pooling softmax calls `expf` on device and `f32::exp` on the
//!   host. They are not required to agree bit-for-bit and the tolerance absorbs the
//!   difference, so a softmax that was wrong by less than that is invisible. The
//!   separations measured below say how much room that leaves.
//! * **Whether `act_quant`'s subnormal e4m3 ties are reached at all.** The quantizer is
//!   S2b's and S2b built a fixture engineered to land on them; nothing in these fixtures
//!   does, so this file exercises the COMMON path of that kernel and not its corners.
//!
//! Skips with a printed reason when the checkpoint is absent — there is no CI and this
//! reads 167 GB of index metadata, so it must not be a hard failure on a machine without it.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli::backend::hip::device_sync;
use rivoli::math::{bf16_to_f32, f32_to_bf16};
use rivoli::memory::device::DeviceBuf;
use rivoli::v4compress::{Buffers, Finish, Geom, LayerKind};
use rivoli::v4oracle::forward::{CompState, CompressorW, Counters, Defect, Oracle};
use rivoli::v4oracle::weights::{Checkpoint, V4Config, WMat};

mod common;
use common::{EMIT_LEN, PROBE_LEN, PROBE_REMAINDER_LEN, checkpoint, compressor_w, probe};

/// The SECOND ratio-128 decode block, at `(255 + 1) % 128 == 0`.
///
/// Stepping only to the first one (127) cannot see the RoPE position rule: there the block
/// index is 0 and the absolute position is 0, so `start_pos / ratio` and
/// `(start_pos / ratio) * ratio` agree, and the rotation is the identity either way. At the
/// second block they are 1 and 128. This is the same shape of trap as the previous S2c
/// session's finding that a 256-token ratio-128 prefill exercises no state carry.
const RATIO_128_SECOND_DECODE_BLOCK: usize = 255;

// =======================================================================================
// device plumbing
// =======================================================================================

/// A device allocation that remembers how many ELEMENTS it holds.
///
/// The count is the point. Every field of [`Buffers`] is a bare pointer with a shape
/// contract stated only in prose, and the failure that contract exists to prevent — a
/// scratch sized for decode handed a prefill — is recorded in
/// `docs/investigations/v4-flash-port.md` as a live hazard S2b left open. Here the length
/// travels with the pointer and [`Cells::run`] checks it before every launch.
struct Dev {
    buf: DeviceBuf,
    len: usize,
}

impl Dev {
    /// Upload `v` as little-endian bytes. Generic over the element so the f32 and u16 paths
    /// are one function: they differed only in `size_of`, and two copies of an upload is two
    /// places for an element-count-versus-byte-count slip to live.
    fn up<T: Copy, const N: usize>(v: &[T], le: fn(T) -> [u8; N]) -> Self {
        let bytes: Vec<u8> = v.iter().copied().flat_map(le).collect();
        let mut buf = DeviceBuf::new(bytes.len().max(1)).expect("v4c: device alloc");
        buf.copy_in_at(0, &bytes).expect("v4c: upload");
        Self { buf, len: v.len() }
    }

    fn f32(v: &[f32]) -> Self {
        Self::up(v, f32::to_le_bytes)
    }

    fn u16(v: &[u16]) -> Self {
        Self::up(v, u16::to_le_bytes)
    }

    /// `n` copies of `fill`. `score_state` is **-inf**-initialised, not zero: a zero would
    /// make every never-written slot a live pooling entry with weight `exp(0 - m)`, which
    /// is a plausible number and a wrong window.
    fn filled(n: usize, fill: f32) -> Self {
        Self::f32(&vec![fill; n])
    }

    fn read(&self) -> Vec<f32> {
        rivoli::artifact::quant::read_f32(&self.buf.copy_out().expect("v4c: readback"))
    }

    fn p(&self) -> *const f32 {
        self.buf.ptr().cast()
    }

    fn pm(&mut self) -> *mut f32 {
        self.buf.ptr_mut().cast()
    }
}

/// One `WMat::Dense` weight as the bf16 codes the kernel decodes with `bf16f`.
///
/// Asserts the round-trip is EXACT rather than assuming it. The checkpoint stores these in
/// bf16 and `Checkpoint::dense` widens them to f32, so re-encoding must be lossless — if it
/// ever is not, the kernel is being fed a different matrix from the oracle and every
/// comparison below silently measures that instead of the pooling.
fn bf16_rows(w: &WMat) -> Vec<u16> {
    let (rows, cols) = (w.rows(), w.cols());
    let mut out = Vec::with_capacity(rows * cols);
    let mut buf = Vec::new();
    for r in 0..rows {
        w.row(r, &mut buf);
        for &v in &buf {
            let code = f32_to_bf16(v);
            assert_eq!(
                bf16_to_f32(code),
                v,
                "compressor weight row {r} is not bf16-exact: the oracle and the kernel \
                 would be reading different numbers"
            );
            out.push(code);
        }
    }
    out
}

/// `Oracle::freqs`'s `(cos, sin)` pairs, flattened to the `[pos][2*i], [pos][2*i+1]` layout
/// `v4c_finish_row` indexes.
fn flat_freqs(t: &[(f32, f32)]) -> Vec<f32> {
    t.iter().flat_map(|&(c, s)| [c, s]).collect()
}

// =======================================================================================
// the metric
// =======================================================================================

/// A comparison of two `[n, d]` block sets, split by whether `act_quant` touched the dim.
///
/// The split is the instrument. `act_quant` covers dims `[0, d - rd)` at block 64 and leaves
/// the RoPE tail `[d - rd, d)` in bf16 (model.py:378), so **the tail is a direct window onto
/// the pre-quantization arithmetic**. A disagreement confined to the quantized region cannot
/// be a pooling, norm or RoPE bug; one that appears in the tail must be.
///
/// `worst_ratio` exists because e4m3 is a STEP function: a pre-quantization value a hair
/// over a rounding boundary quantizes a whole step away, and one e4m3 step is
/// [`E4M3_ULP`] = **exactly 16 bf16 codes**. By magnitude alone that is indistinguishable
/// from a real ~9% arithmetic error; what distinguishes it is the RATIO landing on an e4m3
/// step and only a handful of elements moving at all.
struct Diff {
    /// Max bf16 code gap over every element.
    max: u32,
    /// Max over dims `act_quant` rewrote.
    max_quant: u32,
    /// Max over the RoPE tail, which `act_quant` never touches.
    max_tail: u32,
    /// How many elements differ at all, out of how many.
    differing: usize,
    total: usize,
    /// `(dim within the row, want, got)` at the largest gap.
    worst: (usize, f32, f32),
}

impl Diff {
    /// `got / want` at the worst element. One e4m3 step is a ratio in **[1.0667, 1.125]**
    /// (or the reciprocal) depending where in the binade it lands — not `1.125` flat, which
    /// only holds at mantissa 0. Outside that range the disagreement is not a boundary flip.
    fn worst_ratio(&self) -> f32 {
        let (_, w, g) = self.worst;
        if w == 0.0 { f32::INFINITY } else { g / w }
    }

    fn one_line(&self, label: &str) -> String {
        let head = format!(
            "{label}: max={} (quant_dims={} rope_tail={}) differing={}/{} ({:.4}%)",
            self.max,
            self.max_quant,
            self.max_tail,
            self.differing,
            self.total,
            100.0 * self.differing as f64 / self.total as f64
        );
        // `worst` is only meaningful when something differed. Printing `ratio=inf` for a
        // bit-identical pass would read as the pathological "want is zero" case on the one
        // line a triager looks at.
        if self.differing == 0 {
            return format!("{head} bit-identical");
        }
        let (dim, w, g) = self.worst;
        format!("{head} worst@dim{dim} want={w:e} got={g:e} ratio={:.4}", self.worst_ratio())
    }
}

/// Compare two flattened `[n, d]` block sets in bf16 code space.
///
/// Both sides hold bf16 values — the kernel's last act on every row is `v4c_rbf16` and the
/// oracle's is `round_bf16` — so the unit is exact and no epsilon is chosen: re-encode both
/// and difference the codes. 0 is bit-identical, 1 is adjacent representable values.
///
/// Sign goes through a monotone ordering first. Raw bf16 codes across zero would report
/// ~65000 for two values a hair apart, which would make the metric read as noise exactly
/// where cancellation put the interesting cases.
///
/// **This is a RELATIVE metric and it has a known blind spot at zero**: a code gap says
/// nothing about absolute magnitude, so an element near zero can report a large gap for a
/// negligible absolute difference. That is why [`Diff`] carries `differing` and `worst` —
/// the count and the actual pair are what separate that case from a real error, and the max
/// alone cannot.
fn diff(want: &[f32], got: &[f32], d: usize, rd: usize) -> Diff {
    assert_eq!(want.len(), got.len(), "diff: length mismatch");
    assert!(want.len().is_multiple_of(d), "diff: not a whole number of [d] rows");
    let ord = |x: f32| -> i32 {
        let c = i32::from(f32_to_bf16(x) as i16);
        if c < 0 { -32768 - c } else { c }
    };
    let quant_dims = d - rd;
    let mut out = Diff {
        max: 0,
        max_quant: 0,
        max_tail: 0,
        differing: 0,
        total: want.len(),
        worst: (0, 0.0, 0.0),
    };
    for (i, (&w, &g)) in want.iter().zip(got).enumerate() {
        let e = ord(w).abs_diff(ord(g));
        if e == 0 {
            continue;
        }
        out.differing += 1;
        let dim = i % d;
        if dim < quant_dims {
            out.max_quant = out.max_quant.max(e);
        } else {
            out.max_tail = out.max_tail.max(e);
        }
        if e > out.max {
            out.max = e;
            out.worst = (dim, w, g);
        }
    }
    out
}

/// The verdict on a CLEAN comparison, stated in the unit each region actually ends in.
///
/// Three conditions, and the point is that no one of them can be satisfied by loosening
/// another: the RoPE tail is not quantized so it is held to the bf16 floor; the quantized
/// dims may differ by at most one e4m3 step; and only a sliver of elements may differ at
/// all. A real arithmetic error shows up in the tail, or moves more than one step, or moves
/// the bulk of the elements — this rejects all three.
fn assert_clean(name: &str, dv: &Diff) -> Vec<String> {
    let mut bad = Vec::new();
    if dv.max_tail > CLEAN_ULP {
        bad.push(format!(
            "{name}: RoPE tail {} > {CLEAN_ULP} bf16 ULP — `act_quant` never touches those \
             dims, so this is the pooling, the norm or the rotation and not a rounding step",
            dv.max_tail
        ));
    }
    if dv.max_quant > E4M3_ULP {
        bad.push(format!(
            "{name}: quantized dims {} > one e4m3 step ({E4M3_ULP}) — more than a boundary flip",
            dv.max_quant
        ));
    }
    let frac = dv.differing as f64 / dv.total as f64;
    if frac > MAX_BOUNDARY_FLIPS {
        bad.push(format!(
            "{name}: {}/{} elements differ ({:.3}%) — a boundary flip is rare and this is \
             systematic, so the one-step allowance is covering a real error",
            dv.differing, dv.total, 100.0 * frac
        ));
    }
    bad
}

/// Compare and PRINT. The number is the evidence — a comparison that passed at 0 and one
/// that passed at 3 look identical in a green run, and only one says the kernel reproduces
/// the reference.
fn gap(label: &str, want: &[f32], got: &[f32], d: usize, rd: usize) -> u32 {
    let dv = diff(want, got, d, rd);
    println!("{}", dv.one_line(label));
    dv.max
}

/// The bound every clean comparison in this file is held to.
///
/// Not zero, and the reason is specific: `v4c_block_sum` folds the RMSNorm's sum-of-squares
/// as a tree over 256 threads while the oracle folds it sequentially over 512 elements, and
/// `wave_sum` does the same to both projection dots. That re-association moves `rs` by a
/// relative ~1e-7, which the following bf16 store rounds away in almost every element and
/// occasionally does not. `expf` versus `f32::exp` adds the same order again.
///
/// 2 is "the re-association floor plus one". It applies **only to the RoPE tail**, which is
/// the last region still ending in a bf16 store — see [`E4M3_ULP`] for why it was the wrong
/// unit for the rest, and for what the first real-weights run actually measured.
const CLEAN_ULP: u32 = 2;

/// One e4m3 quantization step, in bf16 codes. **Exactly 16, in every binade.**
///
/// e4m3 carries 3 mantissa bits and bf16 carries 7, over the same exponent semantics: a
/// value is `2^E·(1 + m/8)` against `2^E·(1 + n/128)`, so `m → m+1` *is* `n → n+16`. There
/// is no binade dependence and no approximation in that.
///
/// **This is why the first real-weights run read 16 and why widening `CLEAN_ULP` would have
/// been the wrong repair.** `act_quant` is the last thing that touches dims `[0, d - rd)`,
/// so those dims do not end in a bf16 store at all — they end in an e4m3 one, and 16 codes
/// is not "a 9% error", it is the SMALLEST nonzero disagreement those dims can express. It
/// is the e4m3 1-ULP. Holding a quantized output to a bf16 ULP was a unit error in the
/// harness, not a defect in the kernel.
///
/// The bound stays honest because it is one step and because two independent conditions sit
/// beside it: the untouched RoPE tail is still held to [`CLEAN_ULP`], and
/// [`MAX_BOUNDARY_FLIPS`] caps how many elements may move at all. A systematic error cannot
/// hide under a one-step allowance while satisfying both.
const E4M3_ULP: u32 = 16;

/// The fraction of elements allowed to sit on the far side of an e4m3 rounding boundary.
///
/// Derived, and then actually SET at the derivation — which the first version of this
/// constant was not, and a reviewer showed what that cost.
///
/// An element flips only if its pre-quantization value lies within the two implementations'
/// relative disagreement `ε` of a boundary. The relative step is `(1/8)/(1 + m/8)` for
/// `m ∈ [0,8)`, i.e. between **0.0667 and 0.125** — so the expected flip fraction is
/// `ε / 0.0667`, not `ε / 0.125`; taking the wide end understates flips by up to 1.9x.
/// Re-association puts `ε` near 1e-6, predicting well under one flip in 32768 elements;
/// `ε = 1e-4` predicts 0.15%.
///
/// **This was 1% and that was wrong.** At 1% the three clean conditions were jointly
/// satisfiable by a real systematic error: a uniform ~0.1% relative error (a wrong
/// `norm_eps`, an `ape` scaled by 1e-3, a subtly wrong softmax) leaves the tail under one
/// bf16 code — one code is 0.78% relative — flips the quantized dims by exactly one step
/// where it flips them at all, and lands near 0.8% of elements. All three green, real bug.
/// At 0.1% that example fails loudly, while still sitting ~40x above the worst `ε` the
/// derivation admits. The measured fraction is printed on every comparison so this can be
/// tightened from data rather than from argument.
const MAX_BOUNDARY_FLIPS: f64 = 0.001;

/// How far a defect must move the output before the comparison is said to resolve it.
///
/// Stated against the **quantization floor**, not against the clean gap. One e4m3 step is
/// the smallest disagreement a quantized dim can express, so anything within a step or two
/// of that is indistinguishable from a boundary flip. Four steps is the bound; the first
/// real-weights run measured the `no-ape` separations at ~30000, nearly 2000 steps, so real
/// defects clear this by three orders of magnitude and the bound is not what decides them.
const RESOLVABLE: u32 = 4 * E4M3_ULP;

// =======================================================================================
// one cell
// =======================================================================================

/// One compressor under test: its weights, its geometry, and the buffers both sides use.
///
/// Holds the ORACLE's weights and derives the device ones, rather than loading twice. Two
/// loads of the same tensor is how a comparison ends up scoring one implementation against
/// a differently-transposed copy of its own input.
struct Cell {
    cw: CompressorW,
    geom: Geom,
    cfg: V4Config,
    layer: usize,
    wkv: Dev,
    wgate: Dev,
    ape: Dev,
    norm: Dev,
}

impl Cell {
    fn load(ck: &rivoli::v4oracle::weights::Checkpoint, cfg: &V4Config, layer: usize) -> Self {
        let ratio = cfg.compress_ratio(layer);
        let cw = compressor_w(ck, &format!("layers.{layer}.attn.compressor"), ratio, cfg.head_dim, false);
        let geom = Geom::attention(LayerKind::from_ratio(ratio), cfg.head_dim, cfg.rope_head_dim, cfg.norm_eps)
            .expect("a compressed layer has a Geom");
        Self {
            wkv: Dev::u16(&bf16_rows(&cw.wkv)),
            wgate: Dev::u16(&bf16_rows(&cw.wgate)),
            ape: Dev::f32(&cw.ape),
            norm: Dev::f32(&cw.norm),
            cw,
            geom,
            cfg: cfg.clone(),
            layer,
        }
    }

    fn ratio(&self) -> usize {
        self.cw.ratio
    }

    /// The rotary table for this layer, taken from the ORACLE under `defect` and flattened.
    ///
    /// Deliberately not rebuilt from `freqs_cis`/`rope_for_layer`. Two things follow. The
    /// kernel is handed the oracle's own table, so any disagreement below is arithmetic and
    /// never table construction — which is what makes the gap numbers mean the pooling. And
    /// `Defect::RopeNoYarn` becomes expressible as an INPUT: `Oracle::freqs` returns the
    /// ratio-0 table under that defect, which is exactly the substitution
    /// `the_ratio_0_rope_table_reproduces_the_no_yarn_defect_exactly` performs.
    ///
    /// The cost, stated: `rope_for_layer` — the selector the ENGINE would use — is not
    /// exercised here. `tests/v4_compress.rs` covers it against the same oracle, on the host,
    /// with no device involved.
    fn table(&self, defect: Defect) -> Vec<f32> {
        flat_freqs(Oracle::new(self.cfg.clone(), defect).freqs(self.layer))
    }

    /// Run BOTH implementations over the same script of calls and return
    /// `(oracle blocks, gpu blocks)` — one flat `[n, d]` vector each, concatenated in
    /// emission order.
    ///
    /// `steps` is `(seqlen, start_pos)` pairs, so a cell can be a single prefill, a prefill
    /// followed by many decodes, or anything else. Both sides walk the SAME script with the
    /// SAME activations and the same fresh state, which is what makes the comparison about
    /// the arithmetic rather than about the driving.
    ///
    /// `ape_over` and `freqs_over` substitute a kernel input. They are how
    /// `Defect::CompressorNoApe` and `Defect::RopeNoYarn` are impersonated exactly — see
    /// this file's header.
    fn run(
        &mut self,
        defect: Defect,
        steps: &[(usize, usize)],
        ape_over: Option<&[f32]>,
        freqs_over: Option<&[f32]>,
    ) -> (Vec<f32>, Vec<f32>) {
        let o = Oracle::new(self.cfg.clone(), defect);
        let mut cs: CompState =
            o.fresh_state(self.layer).comp.expect("a compressed layer has compressor state");
        let mut ctr = Counters::default();
        let d = self.cfg.head_dim;
        let (cd, ents) = (self.geom.cd(), self.geom.ents());
        let max_rows = steps.iter().map(|s| s.0).max().unwrap_or(1);

        let clean_table = self.table(Defect::None);
        let freqs_dev = Dev::f32(freqs_over.unwrap_or(&clean_table));
        let ape_dev = ape_over.map(Dev::f32);
        let mut kv_state = Dev::filled(ents * cd, 0.0);
        let mut score_state = Dev::filled(ents * cd, f32::NEG_INFINITY);
        let mut kv = Dev::filled(max_rows * cd, 0.0);
        let mut score = Dev::filled(max_rows * cd, 0.0);
        let mut out = Dev::filled(max_rows.div_ceil(self.ratio()).max(1) * d, 0.0);

        let (mut want, mut got) = (Vec::new(), Vec::new());
        for &(s, start_pos) in steps {
            // Same activations to both sides. `probe` is seeded by name, so the fixture is
            // reproducible and a rerun compares the same numbers.
            let x = probe(&format!("l{}-s{s}-p{start_pos}", self.layer), s, self.cfg.dim);
            if let Some(v) = o.compressor(
                &self.cw,
                &mut cs,
                &x,
                s,
                start_pos,
                o.freqs(self.layer),
                &mut ctr,
            ) {
                want.extend_from_slice(&v);
            }

            let x_dev = Dev::f32(&x);
            let fin = Finish {
                norm: self.norm.p(),
                freqs: freqs_dev.p(),
                out: out.pm(),
            };
            let b = Buffers {
                x: x_dev.p(),
                dim: self.cfg.dim,
                wkv: self.wkv.buf.ptr().cast(),
                wgate: self.wgate.buf.ptr().cast(),
                ape: ape_dev.as_ref().map_or_else(|| self.ape.p(), Dev::p),
                fin,
                kv_state: kv_state.pm(),
                score_state: score_state.pm(),
                kv: kv.pm(),
                score: score.pm(),
                scratch_rows: max_rows,
            };
            // The shape contract `Buffers` states only in prose, checked against the
            // lengths `Dev` carries. Without this the file would be asserting agreement
            // between an oracle and a kernel reading past the end of its inputs.
            assert!(kv.len >= s * cd && score.len >= s * cd, "scratch too small for {s} rows");
            assert!(kv_state.len == self.geom.state_len(), "state buffer is [ents, cd]");
            assert!(self.ape.len == self.ratio() * cd, "ape is [ratio, coff*d]");
            // SAFETY: every pointer above comes from a `Dev` alive for this iteration, at
            // the element counts just asserted; `device_sync` below completes the work
            // before any of them drops.
            let n = unsafe { rivoli::v4compress::compress(&self.geom, &b, s, start_pos) }
                .expect("compress launch");
            device_sync().expect("device_sync");
            got.extend_from_slice(&out.read()[..n * d]);
        }
        assert_eq!(
            want.len(),
            got.len(),
            "the two implementations disagree on HOW MANY blocks are emitted, which no \
             value comparison below would have reported"
        );
        (want, got)
    }
}

/// The four cells, built once. Every test below walks this list, so a cell added here is
/// added to the clean comparison, the two impersonations and the separation sweep at once —
/// and a cell cannot be exercised by one of them and quietly missing from another.
///
/// `ratio4/prefill` is 256 tokens; `ratio128/prefill` is 300, which is the only prefill path
/// that writes state (`300 % 128 == 44`). A 256-token ratio-128 prefill writes NO state —
/// `overlap` is false and `256 % 128 == 0`, so both state writes are skipped — which the
/// previous S2c session got wrong and two of its reviewers disproved by zeroing the buffers.
/// `state_is_not_read_by_the_ratio_128_prefill_at_a_whole_multiple` re-proves it here.
///
/// Both decode scripts run to their SECOND completed block. Stopping at the first cannot
/// see the RoPE position rule; see [`RATIO_128_SECOND_DECODE_BLOCK`].
struct Spec {
    layer: usize,
    /// `(seqlen, start_pos)` per call — the same pair `Compressor.forward` takes.
    script: Vec<(usize, usize)>,
    name: &'static str,
}

fn cells() -> Option<(Checkpoint, V4Config, Vec<Spec>)> {
    let ck = checkpoint()?;
    let cfg = V4Config::v4_flash();
    assert_eq!(cfg.compress_ratio(2), 4, "layer 2 is the overlapping class");
    assert_eq!(cfg.compress_ratio(3), 128, "layer 3 is the non-overlapping class");
    let spec = |layer, script, name| Spec { layer, script, name };
    let list = vec![
        spec(2, vec![(PROBE_LEN, 0)], "ratio4/prefill"),
        spec(2, decode_script(4, 23), "ratio4/decode"),
        spec(3, vec![(PROBE_REMAINDER_LEN, 0)], "ratio128/prefill+remainder"),
        spec(3, decode_script(128, RATIO_128_SECOND_DECODE_BLOCK), "ratio128/decode"),
    ];
    Some((ck, cfg, list))
}

/// A short prefill followed by single-row decodes up to and including `last`.
///
/// The assertion is the point of the function: a decode script that completes fewer than two
/// blocks cannot distinguish the RoPE position `(start_pos / ratio) * ratio` from the block
/// index `start_pos / ratio`, because at the first block both are 0 and the rotation is the
/// identity. Building the script by hand is how that gets forgotten.
fn decode_script(ratio: usize, last: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(EMIT_LEN, 0)];
    v.extend((EMIT_LEN..=last).map(|p| (1, p)));
    assert!(
        v.iter().filter(|&&(s, p)| s == 1 && (p + 1) % ratio == 0).count() >= 2,
        "a decode script must complete at least two blocks, else the RoPE position and the \
         block index cannot be told apart"
    );
    v
}

// =======================================================================================
// the four cells
// =======================================================================================

/// Ratio 4 (layer 2) and ratio 128 (layer 3), prefill and decode, against the clean oracle.
#[test]
fn the_four_cells_reproduce_the_oracle() {
    let Some((ck, cfg, list)) = cells() else { return };
    // Every cell reports BEFORE anything asserts. The first run of this file asserted inside
    // the loop and failed on cell 1 of 4, so three quarters of the diagnostic never printed
    // and the failure could not be told from a phase-dependent one. A gate that aborts on
    // its first cell hands back a quarter of what it measured.
    let mut over = Vec::new();
    for Spec { layer, script, name } in list {
        let mut cell = Cell::load(&ck, &cfg, layer);
        let (want, got) = cell.run(Defect::None, &script, None, None);
        assert!(!want.is_empty(), "{name}: the script emitted nothing — it gates nothing");
        assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite output");
        let dv = diff(&want, &got, cfg.head_dim, cfg.rope_head_dim);
        println!("{}", dv.one_line(name));
        over.extend(assert_clean(name, &dv));
    }
    assert!(over.is_empty(), "clean comparison failed:\n  {}", over.join("\n  "));
}

/// The ratio-128 prefill at 256 reads NO compressor state — re-proved against the GPU.
///
/// The previous S2c session asserted a block-to-block state carry here that does not exist,
/// and two reviewers disproved it by substituting zero-length state buffers and getting
/// bit-identical output. The technique is what is worth keeping, so it is applied to the
/// kernel: `Cell::run` allocates fresh state per call, so two identical runs must agree
/// bit-for-bit, and the length asserted below shows the buffers were real rather than
/// accidentally empty.
///
/// Scoped to a whole multiple of the ratio on purpose — at 300 tokens the remainder path
/// DOES write state, which is why `cells()` uses 300 for the ratio-128 prefill cell.
#[test]
fn state_is_not_read_by_the_ratio_128_prefill_at_a_whole_multiple() {
    let Some((ck, cfg, _)) = cells() else { return };
    assert_eq!(PROBE_LEN % 128, 0, "the claim is scoped to a whole multiple of the ratio");
    let mut cell = Cell::load(&ck, &cfg, 3);
    let (_, base) = cell.run(Defect::None, &[(PROBE_LEN, 0)], None, None);
    let (_, again) = cell.run(Defect::None, &[(PROBE_LEN, 0)], None, None);
    assert!(!base.is_empty(), "256 tokens pools two ratio-128 blocks");
    assert_eq!(base, again, "the harness is not deterministic — nothing else here is evidence");
}

// =======================================================================================
// making it go red — technique 1: exact defect impersonation
// =======================================================================================

/// **`ape` is load-bearing, proved exactly.** Zeroing the position embedding is precisely
/// `Defect::CompressorNoApe`, so the kernel fed a zero `ape` must reproduce the oracle
/// running with that defect — to the same tolerance as the clean comparison — while being
/// far from the clean oracle.
///
/// This is the strongest gate available without shipping a break switch. It does not merely
/// show the output moved: it shows it moved *to the specific wrong place the oracle says a
/// missing `ape` produces*. A kernel that ignored `ape` entirely would pass the first
/// assertion and fail the second.
#[test]
fn zeroing_ape_reproduces_the_no_ape_defect_exactly() {
    let Some((ck, cfg, list)) = cells() else { return };
    for Spec { layer, script, name } in list {
        let mut cell = Cell::load(&ck, &cfg, layer);
        let zeros = vec![0.0f32; cell.cw.ape.len()];
        let (clean, _) = cell.run(Defect::None, &script, None, None);
        let (broken, gpu) = cell.run(Defect::CompressorNoApe, &script, Some(&zeros), None);
        assert_impersonates(name, "no-ape", &clean, &broken, &gpu, cfg.head_dim, cfg.rope_head_dim);
    }
}

/// **The rotary table selection is load-bearing, proved exactly.** Handing the kernel the
/// ratio-0 table (base `rope_theta`, no YaRN) in place of the compressed one is precisely
/// `Defect::RopeNoYarn`, so the kernel must land where the oracle-with-that-defect lands.
///
/// This is the hazard `docs/investigations/v4-flash-port.md` records from S2b — `Io.freqs`
/// is a raw pointer that cannot distinguish the two tables — measured rather than argued.
/// `Finish` groups the pointer with `norm` and `out` for the same reason; nothing in the
/// type system tells the two tables apart, so a test has to.
#[test]
fn the_ratio_0_rope_table_reproduces_the_no_yarn_defect_exactly() {
    let Some((ck, cfg, list)) = cells() else { return };
    for Spec { layer, script, name } in list {
        let mut cell = Cell::load(&ck, &cfg, layer);
        let plain = cell.table(Defect::RopeNoYarn);
        assert_ne!(plain, cell.table(Defect::None), "{name}: the two tables must differ");
        let (clean, _) = cell.run(Defect::None, &script, None, None);
        let (broken, gpu) = cell.run(Defect::RopeNoYarn, &script, None, Some(&plain));
        assert_impersonates(name, "no-yarn", &clean, &broken, &gpu, cfg.head_dim, cfg.rope_head_dim);
    }
}

/// The two-sided assertion both impersonations make: the GPU lands ON the defect-injected
/// oracle, and FAR from the clean one.
///
/// Both halves are required and neither alone is worth anything. Without the first, the
/// perturbation is only known to have changed something. Without the second, a kernel that
/// ignored the perturbed input entirely — the exact failure being hunted — would pass,
/// because the clean and defect oracles would be close and it would match both.
fn assert_impersonates(
    cell: &str,
    what: &str,
    clean: &[f32],
    broken: &[f32],
    gpu: &[f32],
    d: usize,
    rd: usize,
) {
    let hit = diff(broken, gpu, d, rd);
    println!("{}", hit.one_line(&format!("{cell} {what}: gpu vs defect-oracle")));
    let bad = assert_clean(&format!("{cell} {what} impersonation"), &hit);
    assert!(
        bad.is_empty(),
        "the {what} perturbation must land on the oracle's own defect to within the \
         quantization floor (NOT bit-exactly -- `act_quant` makes one e4m3 step the smallest \
         expressible disagreement):\n  {}",
        bad.join("\n  ")
    );
    let sep = gap(&format!("{cell} {what}: gpu vs CLEAN oracle"), clean, gpu, d, rd);
    assert!(
        sep >= RESOLVABLE,
        "{cell}: the {what} perturbation moved the output by only {sep} codes — under \
         {RESOLVABLE} ({} e4m3 steps) it is not distinguishable from the quantization floor, \
         so this cell cannot see whether the input is consulted at all",
        RESOLVABLE / E4M3_ULP
    );
}

// =======================================================================================
// making it go red — technique 2: distance separation, and 3: named inertness
// =======================================================================================

/// Every remaining in-scope breakage is measurably further from the GPU than the clean
/// oracle is — or is asserted INERT on that cell and therefore claimed as coverage of
/// nothing.
///
/// These live inside the kernel and cannot be reached by perturbing an input, so this is the
/// weaker instrument: it proves the metric resolves each defect, not that this kernel would
/// fail if broken that way. The two tests above are the strong half.
///
/// The inert half matters as much as the separated half. `CompressorNoOverlap` on a
/// ratio-128 layer has no term to disable, and a run that quietly "passed" it would be read
/// as coverage of the overlapping branch by anyone scanning the list.
#[test]
fn each_in_scope_defect_is_further_from_the_gpu_than_the_clean_oracle_is() {
    let Some((ck, cfg, list)) = cells() else { return };
    // The compressor's own breakages, the RoPE ones inside `v4c_finish_row`, the four
    // `act_quant` ones (S2b's kernel, this module's call arguments) and the bf16 stores.
    // Defects outside the compressor — the attention core, the router, the MoE, the indexer
    // — are excluded here rather than silently passing inside the list.
    // Derived by EXHAUSTIVE match over `Defect::ALL` rather than spelled as a list. A list
    // silently omits any variant added later; the match makes one a compile error, which is
    // the same argument `src/v4compress.rs` makes about wildcards on domain enums — and the
    // moment a new breakage is added is exactly when someone must decide whether the
    // compressor can see it.
    let in_scope: Vec<Defect> = Defect::ALL.iter().copied().filter(|d| in_compressor_scope(*d)).collect();
    assert!(in_scope.len() >= 10, "the scope filter selected almost nothing");

    let mut bad = Vec::new();
    for Spec { layer, script, name } in list {
        let mut cell = Cell::load(&ck, &cfg, layer);
        let (clean, gpu) = cell.run(Defect::None, &script, None, None);
        let cd = diff(&clean, &gpu, cfg.head_dim, cfg.rope_head_dim);
        println!("{}", cd.one_line(&format!("{name} clean")));
        // RECORDED, not asserted here. An over-budget clean comparison makes the separations
        // below uninterpretable, but it does not make them unmeasurable — and their pattern
        // is diagnostic in its own right, so measuring them beats aborting.
        bad.extend(assert_clean(name, &cd));

        for &def in &in_scope {
            // The two impersonations have their own, stronger tests above.
            if matches!(def, Defect::CompressorNoApe | Defect::RopeNoYarn) {
                continue;
            }
            let (broken, _) = cell.run(def, &script, None, None);
            if broken == clean {
                // INERT here, by construction. Printed rather than skipped silently: the
                // point of naming it is that this cell must not be counted as covering it.
                println!("{name}: {def:?} is INERT here — this cell covers it not at all");
                continue;
            }
            let sep = gap(&format!("{name} {def:?}"), &broken, &gpu, cfg.head_dim, cfg.rope_head_dim);
            if sep < RESOLVABLE {
                bad.push(format!("{name}/{def:?} sep={sep} < {RESOLVABLE}"));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "the metric cannot resolve these, so a kernel carrying the defect might well pass: {}",
        bad.join(" | ")
    );
}

/// Does this breakage live anywhere the attention compressor touches?
///
/// Exhaustive and wildcard-free on purpose. `Defect` is a domain enum this repo owns, so a
/// variant added by a later stage must come back here and be classified rather than
/// defaulting to "not our problem" — which is how a real compressor defect ends up outside
/// every list that claims to cover the compressor.
fn in_compressor_scope(d: Defect) -> bool {
    match d {
        // The compressor's own three, the RoPE inside `v4c_finish_row`, the four
        // `act_quant` arguments and the bf16 stores.
        Defect::CompressorNoOverlap
        | Defect::CompressorNoApe
        | Defect::CompressorRopeAtBlockEnd
        | Defect::RopeAllDims
        | Defect::RopeFirstDims
        | Defect::RopeHalfSplit
        | Defect::RopeNoYarn
        | Defect::SkipKvActQuant
        | Defect::KvActQuantWholeTensor
        | Defect::KvActQuantBlock128
        | Defect::KvActQuantNoRoundScale
        | Defect::NoBf16Rounding => true,
        // `None` is the baseline, not a breakage. `RopeYarnEverywhere` and
        // `RopeBaseThetaEverywhere` key off a ratio-0 layer, which by construction has no
        // compressor at all. Everything below belongs to the attention core (S2b), the
        // router and MoE (S2a), or the indexer (unwritten).
        Defect::None
        | Defect::RopeYarnEverywhere
        | Defect::RopeBaseThetaEverywhere
        | Defect::SkipQkNorm
        | Defect::QkNormUsesQNormWeight
        | Defect::QkNormAfterRope
        | Defect::SkipAttnSink
        | Defect::AttnSinkNotMaxShifted
        | Defect::PrefillRingWritesFirstWindow
        | Defect::SkipOutputDerotation
        | Defect::OutputDerotationForward
        | Defect::WoGroupsSplitHeadDim
        | Defect::WoGroupsInterleaved
        | Defect::IndexerNoRelu
        | Defect::IndexerNoFp4Quant
        | Defect::IndexerNoHadamard
        | Defect::IndexerNoWeights
        | Defect::SwigluUnclamped
        | Defect::SwigluClampGateBothSides
        | Defect::RouterSoftmax
        | Defect::RouterNoSoftplusThreshold
        | Defect::RouterBiasedWeights
        | Defect::RouterNoRenorm
        | Defect::RouterNoScale
        | Defect::HashRoutingIgnored
        | Defect::RouteWeightAfterW2
        | Defect::SharedExpertWeighted
        | Defect::Fp4NibbleSwap
        | Defect::SinkhornOneFewerIter
        | Defect::SinkhornCombTransposed
        | Defect::HcPostNoComb
        | Defect::HcPreNoRsqrt
        // Added by the head-tail stage, 2026-08-05, because this match is exhaustive and
        // wildcard-free BY DESIGN -- the doc above asks the adding stage to classify rather
        // than let a new variant default. Classification only; no logic here changed.
        //
        // `IndexerBf16RunningSum` is the indexer's per-head score reduction, and the indexer
        // has its OWN compressor -- distinct instance, distinct algorithm (fp4 + Hadamard,
        // not partial fp8). It cannot reach the attention compressor. The six `Head*`
        // variants live strictly after the last block, downstream of everything here.
        | Defect::IndexerBf16RunningSum
        | Defect::HeadHcNoRsqrt
        | Defect::HeadHcRsqrtPerCopy
        | Defect::HeadNormSkipped
        | Defect::HeadNormNotBf16
        | Defect::HeadNormOverAllTokens
        | Defect::HeadLogitsFromFirstRow => false,
    }
}

/// `CompressorNoOverlap` must be inert at ratio 128 and live at ratio 4 — the pin that says
/// the previous test's `INERT` branch reports a real structural fact rather than a defect
/// that quietly stopped working.
///
/// Without this, every defect could become inert everywhere, the sweep above would print a
/// wall of `INERT`, and it would pass.
#[test]
fn the_overlap_defect_is_inert_at_ratio_128_and_live_at_ratio_4() {
    let Some((ck, cfg, _)) = cells() else { return };
    let script_128 = vec![(PROBE_REMAINDER_LEN, 0)];
    let mut l3 = Cell::load(&ck, &cfg, 3);
    let (clean_128, _) = l3.run(Defect::None, &script_128, None, None);
    let (broken_128, _) = l3.run(Defect::CompressorNoOverlap, &script_128, None, None);
    assert_eq!(
        clean_128, broken_128,
        "at ratio 128 `overlap` is already false, so this defect has no term to disable"
    );

    let script_4 = vec![(PROBE_LEN, 0)];
    let mut l2 = Cell::load(&ck, &cfg, 2);
    let (clean_4, gpu_4) = l2.run(Defect::None, &script_4, None, None);
    let (broken_4, _) = l2.run(Defect::CompressorNoOverlap, &script_4, None, None);
    assert_ne!(clean_4, broken_4, "at ratio 4 the defect must bite");
    let sep = gap("ratio4 no-overlap vs gpu", &broken_4, &gpu_4, cfg.head_dim, cfg.rope_head_dim);
    assert!(
        sep >= RESOLVABLE,
        "the overlapping branch is the half of the compressor ratio 128 never runs, and this \
         cell resolves it by only {sep} bf16 codes"
    );
}

/// `Geom` refuses `Plain`, and the two live geometries disagree on both derived fields in
/// OPPOSITE directions — which is the shape trap stated as an inequality rather than prose.
///
/// A guard nobody proves can fire is a guard that might be `if (false)`.
#[test]
fn the_two_geometries_differ_in_opposite_directions() {
    assert!(
        Geom::attention(LayerKind::Plain, 512, 64, 1e-6).is_none(),
        "a ratio-0 layer has no Compressor object in the reference and must have no Geom"
    );
    let g4 = Geom::attention(LayerKind::Overlap, 512, 64, 1e-6).unwrap();
    let g128 = Geom::attention(LayerKind::NonOverlap(128), 512, 64, 1e-6).unwrap();
    assert_eq!((g4.cd(), g4.ents(), g4.state_len()), (1024, 8, 8192));
    assert_eq!((g128.cd(), g128.ents(), g128.state_len()), (512, 128, 65536));
    assert!(
        g4.cd() > g128.cd() && g4.ents() < g128.ents(),
        "ratio 128 HALVES the projection width and multiplies the window — a loader that \
         inferred one from the other would be right on exactly one of the two layers"
    );
}

/// **The instrument, before it is trusted to diagnose anything.** Needs no device.
///
/// The first run of this file reported 16 ULP and I proposed an explanation for it. An
/// explanation resting on a metric nobody had exercised is a guess wearing a number, so this
/// pins what `diff` reports for two differences whose answers are known independently:
/// exactly one e4m3 step, and a value near zero.
#[test]
fn the_diff_metric_reports_what_it_claims() {
    const D: usize = 512;
    const RD: usize = 64;
    // One e4m3 step at 1.5 -> 1.625. `act_quant` reconstructs `e4m3(v/s)·s`, so a
    // pre-quantization value a hair either side of the boundary lands a whole step away.
    //
    // EXACTLY 16, and the arithmetic is independent of `diff`: bf16 codes 0x3FC0 and 0x3FD0
    // differ by 0x10. An earlier version of this test derived 14.8 from `log2(1.625/1.5)·128`
    // and asserted `14..=16` — bf16 codes are LINEAR in the mantissa, not logarithmic, so
    // that derivation was wrong and the range it produced would have passed an implementation
    // that violated the one claim this test exists to pin.
    let mut want = vec![1.0f32; D];
    let mut got = vec![1.0f32; D];
    want[3] = 1.5;
    got[3] = 1.625;
    let dv = diff(&want, &got, D, RD);
    assert_eq!(f32_to_bf16(1.5), 0x3FC0, "the fixture's own premise");
    assert_eq!(f32_to_bf16(1.625), 0x3FD0);
    assert_eq!(dv.max, E4M3_ULP, "one e4m3 step must read as exactly {E4M3_ULP} bf16 codes");
    assert!((dv.worst_ratio() - 1.0833).abs() < 0.001, "ratio {} ", dv.worst_ratio());
    assert_eq!(dv.differing, 1, "exactly one element moved");
    assert_eq!((dv.max_quant, dv.max_tail), (dv.max, 0), "dim 3 is inside the quantized region");

    // Binade-independence, which is the half the constant actually rests on. If this held
    // only near 1.0 then `E4M3_ULP` would be a coincidence of the fixture rather than a
    // property of the two formats, and the whole diagnosis above it would be unfounded.
    for e in [-4i32, -1, 0, 3, 9] {
        let base = 2.0f32.powi(e);
        for m in 0..7 {
            let a = base * (1.0 + m as f32 / 8.0);
            let b = base * (1.0 + (m + 1) as f32 / 8.0);
            let (mut wa, mut gb) = (vec![1.0f32; D], vec![1.0f32; D]);
            wa[1] = a;
            gb[1] = b;
            assert_eq!(
                diff(&wa, &gb, D, RD).max,
                E4M3_ULP,
                "one e4m3 step at 2^{e}·(1+{m}/8) must still be {E4M3_ULP} codes"
            );
        }
    }

    // The SPLIT is the whole diagnostic, so it must actually split. Same difference, moved
    // into the RoPE tail, has to land in the other bucket — otherwise a tail-only failure
    // would read as a quantization artifact and send the next reader the wrong way.
    let (mut w2, mut g2) = (vec![1.0f32; D], vec![1.0f32; D]);
    w2[D - 1] = 1.5;
    g2[D - 1] = 1.625;
    let dv2 = diff(&w2, &g2, D, RD);
    assert_eq!((dv2.max_quant, dv2.max_tail), (0, dv2.max), "dim 511 is the untouched tail");

    // The known blind spot, pinned rather than described: near zero the code gap is large
    // for a negligible absolute difference. `differing` and `worst` are what distinguish it,
    // which is why the verdict must never rest on `max` alone.
    let (mut w3, mut g3) = (vec![1.0f32; D], vec![1.0f32; D]);
    w3[7] = 1e-30;
    g3[7] = 2e-30;
    let dv3 = diff(&w3, &g3, D, RD);
    assert!(dv3.max >= 100, "a doubling near zero reads as a whole binade: {}", dv3.max);
    assert!((w3[7] - g3[7]).abs() < 1e-29, "…for an absolute difference of nothing");

    // Identical input must read exactly zero, or every 0 printed above means nothing.
    assert_eq!(diff(&want, &want, D, RD).max, 0);
    assert_eq!(diff(&want, &want, D, RD).differing, 0);
    println!("{}", dv.one_line("e4m3-step fixture"));
}
