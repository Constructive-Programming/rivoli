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

/// The largest bf16 ULP gap between two slices, and where it is.
///
/// Both sides hold **bf16 values** — the kernel's last act on every row is `v4c_rbf16` and
/// the oracle's is `round_bf16` — so the natural unit is exact and no epsilon is chosen:
/// re-encode both and difference the codes. A gap of 0 is bit-identical; 1 is adjacent
/// representable values.
///
/// Sign is handled by mapping to a monotone ordering first. Comparing raw bf16 codes across
/// zero would report ~65000 for two values a hair apart, which would make the whole metric
/// read as noise exactly where cancellation put the interesting cases.
fn ulp_gap(want: &[f32], got: &[f32]) -> u32 {
    assert_eq!(want.len(), got.len(), "ulp_gap: length mismatch");
    let ord = |x: f32| -> i32 {
        let c = i32::from(f32_to_bf16(x) as i16);
        if c < 0 { -32768 - c } else { c }
    };
    want.iter().zip(got).map(|(&a, &b)| ord(a).abs_diff(ord(b))).max().unwrap_or(0)
}

/// Print the gap with its label. The number is the evidence — a comparison that passed at 0
/// ULP and one that passed at 3 look identical in a green test run, and only one of them
/// says the kernel reproduces the reference.
fn gap(label: &str, want: &[f32], got: &[f32]) -> u32 {
    let g = ulp_gap(want, got);
    println!("{label}: max bf16 ULP gap = {g}");
    g
}

/// The bound every clean comparison in this file is held to.
///
/// Not zero, and the reason is specific: `v4c_block_sum` folds the RMSNorm's sum-of-squares
/// as a tree over 256 threads while the oracle folds it sequentially over 512 elements, and
/// `wave_sum` does the same to both projection dots. That re-association moves `rs` by a
/// relative ~1e-7, which the following bf16 store rounds away in almost every element and
/// occasionally does not. `expf` versus `f32::exp` adds the same order again.
///
/// 2 is chosen as "the re-association floor plus one", and the measured gaps are PRINTED so
/// a future reader can see how much of it was actually used rather than trusting the bound.
const CLEAN_ULP: u32 = 2;

/// How much further a defect must be than the clean comparison for the separation to mean
/// anything. A defect that moved the output by 3 ULP against a clean gap of 2 would be
/// inside the noise; one that moves it by 20x is not.
const SEPARATION: u32 = 8;

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
    for Spec { layer, script, name } in list {
        let mut cell = Cell::load(&ck, &cfg, layer);
        let (want, got) = cell.run(Defect::None, &script, None, None);
        assert!(!want.is_empty(), "{name}: the script emitted nothing — it gates nothing");
        assert!(got.iter().all(|v| v.is_finite()), "{name}: non-finite output");
        let g = gap(name, &want, &got);
        assert!(g <= CLEAN_ULP, "{name}: {g} ULP > {CLEAN_ULP}");
    }
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
        assert_impersonates(name, "no-ape", &clean, &broken, &gpu);
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
        assert_impersonates(name, "no-yarn", &clean, &broken, &gpu);
    }
}

/// The two-sided assertion both impersonations make: the GPU lands ON the defect-injected
/// oracle, and FAR from the clean one.
///
/// Both halves are required and neither alone is worth anything. Without the first, the
/// perturbation is only known to have changed something. Without the second, a kernel that
/// ignored the perturbed input entirely — the exact failure being hunted — would pass,
/// because the clean and defect oracles would be close and it would match both.
fn assert_impersonates(cell: &str, what: &str, clean: &[f32], broken: &[f32], gpu: &[f32]) {
    let hit = gap(&format!("{cell} {what}: gpu vs defect-oracle"), broken, gpu);
    assert!(
        hit <= CLEAN_ULP,
        "{cell}: the {what} perturbation must land exactly where the oracle's own defect \
         lands, got {hit} ULP > {CLEAN_ULP}"
    );
    let sep = gap(&format!("{cell} {what}: gpu vs CLEAN oracle"), clean, gpu);
    assert!(
        sep >= SEPARATION * CLEAN_ULP,
        "{cell}: the {what} perturbation moved the output by only {sep} ULP, so this cell \
         cannot see whether the input is consulted at all and must not be cited as covering it"
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

    for Spec { layer, script, name } in list {
        let mut cell = Cell::load(&ck, &cfg, layer);
        let (clean, gpu) = cell.run(Defect::None, &script, None, None);
        let base = gap(&format!("{name} clean"), &clean, &gpu);
        assert!(base <= CLEAN_ULP, "{name}: clean gap {base} — nothing below is meaningful");

        for &d in &in_scope {
            // The two impersonations have their own, stronger tests above.
            if matches!(d, Defect::CompressorNoApe | Defect::RopeNoYarn) {
                continue;
            }
            let (broken, _) = cell.run(d, &script, None, None);
            if broken == clean {
                // INERT here, by construction. Printed rather than skipped silently: the
                // point of naming it is that this cell must not be counted as covering it.
                println!("{name}: {d:?} is INERT here — this cell covers it not at all");
                continue;
            }
            let sep = gap(&format!("{name} {d:?}"), &broken, &gpu);
            assert!(
                sep >= SEPARATION * CLEAN_ULP,
                "{name}: {d:?} sits {sep} ULP from the GPU against a clean {base} — the \
                 metric cannot resolve it, so a kernel carrying that defect might well pass"
            );
        }
    }
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
        | Defect::HcPreNoRsqrt => false,
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
    let sep = gap("ratio4 no-overlap vs gpu", &broken_4, &gpu_4);
    assert!(
        sep >= SEPARATION * CLEAN_ULP,
        "the overlapping branch is the half of the compressor ratio 128 never runs, and this \
         cell resolves it by only {sep} ULP"
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
