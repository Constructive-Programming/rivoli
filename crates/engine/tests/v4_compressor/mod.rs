//! The V4 attention-compressor harness, shared by `kernel_v4_compress.rs` (the clean comparison
//! and the two exact impersonations) and `kernel_v4_compress_defects.rs` (the separation sweep).
//!
//! One module because the two binaries must measure the SAME thing: every distance either of
//! them reports is `diff` in bf16 code space over the same four cells, from the same baseline.
//! A second copy of [`Cell::run`] would be a second launch sequence that could drift from the
//! engine's while both stayed green, and a second copy of [`Widths`] would be a second verdict.
//! `glimmer_anchor/mod.rs` beside this is the precedent for the shape.
//!
//! Ported from `old:tests/kvcompress_kernel.rs`, whose bodies and arguments travelled. **What
//! changed, and why**: that file drove `rivoli::kvcompress::compress`, a public entry point
//! taking a `Buffers` struct of raw pointers. This tree has no such function — the compressor is
//! `super::kvcompress::LayerCompressor`, private to the engine's V4 arm — so [`Cell::step`]
//! spells the launch sequence itself, out of the four launchers the engine calls in the same
//! order. That is a strictly larger claim than the old file made: it scores the LAUNCHERS, which
//! is what `crates/cli/tests/kernel_coverage.rs` asks for, and it puts the emission count on
//! `select::compress_dst` — the engine's own — rather than on a second `seqlen / ratio` here.
//!
//! It is also a smaller claim in one place, named rather than left to be discovered: the engine's
//! `LayerCompressor::run` also bounds `at.seqlen` against its scratch and calls
//! `Extent::check_single_row_decode`. Those two guards are `select.rs`'s own unit tests', not
//! this harness's, and a defect in either is invisible here.

#![allow(dead_code)] // this module compiles into BOTH binaries and neither names every item

use rivoli_backend::abi::CompFinish;
use rivoli_backend::hip::{
    device_sync, launch_act_quant_f8_prefix, launch_kv_compress_decode, launch_kv_compress_deposit,
    launch_kv_compress_prefill,
};
use rivoli_engine::device::DeviceBuf;
use rivoli_engine::v4::geometry::{Geom, KV_QUANT_BLOCK, LayerKind, Quantize};
use rivoli_engine::v4::select::{Extent, compress_dst};
use rivoli_oracles::v4oracle::forward::{CompState, CompressorW, Counters, Oracle};
use rivoli_oracles::v4oracle::weights::{Checkpoint, V4Config as OracleV4Config, fixed_bf16};
use std::ffi::c_void;

use super::common::{
    CompSpec, Configs, GemmBf16, bf16_rows, compressor_w, flat_freqs, gemm_bf16_launch, ok, stream,
};

/// `bin/v4-oracle`'s `PROMPT` tokenizes to 13 ids — the length every hole is keyed to.
pub const EMIT_LEN: usize = 13;

/// Two whole ratio-128 blocks.
///
/// It does NOT exercise a block-to-block state carry, which an earlier version of this
/// comment claimed: at ratio 128 `overlap` is false and `256 % 128 == 0`, so both the
/// `overlap && cutoff >= ratio` and the `remainder > 0` state writes are skipped and prefill
/// pools every block independently. Two reviewers disproved the claim the same way —
/// substitute zero-length state buffers and the output is bit-identical.
///
/// Two blocks still earn their keep, for the reason that survives: the blocks are RoPE'd at
/// `freqs[0:256:128]`, i.e. positions 0 and 128, so a wrong per-block rope position or
/// unflatten stride is observable here and would be hidden by a single block (position 0,
/// where the rotation is the identity).
pub const PROBE_LEN: usize = 256;

/// A ratio-128 prefill with a REMAINDER, which is the only prefill path that writes the
/// compressor state — and the state the decode branch then reads.
pub const PROBE_REMAINDER_LEN: usize = 300;

/// The SECOND ratio-128 decode block, at `(255 + 1) % 128 == 0`.
///
/// Stepping only to the first one (127) cannot see the RoPE position rule: there the block index
/// is 0 and the absolute position is 0, so `start_pos / ratio` and `(start_pos / ratio) * ratio`
/// agree, and the rotation is the identity either way. At the second block they are 1 and 128.
pub const RATIO_128_SECOND_DECODE_BLOCK: usize = 255;

/// The metric, split out under the file-size gate. A glob so the two suites keep naming it
/// through `v4_compressor::`, which is where a reader looking for the harness lands.
mod metric;
pub use metric::*;

/// The oracle's breakage enum, re-exported so a consumer names the defect through the harness that
/// takes one. `Run::defect` is this type and nothing else here is parameterised by it, so a second
/// import path would be two ways to spell one vocabulary.
pub use rivoli_oracles::v4oracle::forward::Defect;

// =======================================================================================
// device plumbing
// =======================================================================================

/// A device allocation that remembers how many ELEMENTS it holds.
///
/// The count is the point. Every pointer the four compressor launchers take carries its shape
/// contract in prose only, and the failure that contract exists to prevent — a scratch sized for
/// decode handed a prefill — is a live hazard on this path. Here the length travels with the
/// pointer and [`Cell::step`] checks it before every launch.
pub struct Dev {
    buf: DeviceBuf,
    len: usize,
}

impl Dev {
    /// Upload `v` as little-endian bytes. Generic over the element so the f32 and u16 paths are
    /// one function: they differ only in `size_of`, and two copies of an upload is two places
    /// for an element-count-versus-byte-count slip to live.
    fn up<T: Copy, const N: usize>(v: &[T], le: fn(T) -> [u8; N]) -> Self {
        let bytes: Vec<u8> = v.iter().copied().flat_map(le).collect();
        let mut buf = DeviceBuf::new(bytes.len().max(1)).expect("v4c: device alloc");
        buf.copy_in_at(0, &bytes).expect("v4c: upload");
        Self { buf, len: v.len() }
    }

    pub fn f32(v: &[f32]) -> Self {
        Self::up(v, f32::to_le_bytes)
    }

    pub fn u16(v: &[u16]) -> Self {
        Self::up(v, u16::to_le_bytes)
    }

    /// `n` copies of `fill`. `score_state` is **-inf**-initialised, not zero: a zero makes every
    /// never-written slot a live pooling entry with weight `exp(0 - m)`, which is a plausible
    /// number and a wrong window.
    fn filled(n: usize, fill: f32) -> Self {
        Self::f32(&vec![fill; n])
    }

    fn read(&self) -> Vec<f32> {
        rivoli_artifact::quant::read_f32(&self.buf.copy_out().expect("v4c: readback"))
    }

    fn p(&self) -> *const f32 {
        self.buf.ptr().cast()
    }

    fn pm(&mut self) -> *mut f32 {
        self.buf.ptr_mut().cast()
    }
}

// =======================================================================================
// one cell
// =======================================================================================

/// One compressor under test: its weights, its geometry, and the buffers both sides use.
///
/// Holds the ORACLE's weights and DERIVES the device ones, rather than loading twice. Two loads
/// of the same tensor is how a comparison ends up scoring one implementation against a
/// differently-transposed copy of its own input.
pub struct Cell {
    pub cw: CompressorW,
    geom: Geom,
    cfg: OracleV4Config,
    dim: usize,
    layer: usize,
    wkv: Dev,
    wgate: Dev,
    ape: Dev,
    norm: Dev,
}

impl Cell {
    pub fn load(ck: &Checkpoint, c: &Configs, layer: usize) -> Self {
        let ratio = c.ratio(layer);
        let cw = compressor_w(
            ck,
            &format!("layers.{layer}.attn.compressor"),
            CompSpec {
                ratio,
                d: c.engine.head_dim,
                rotate: false,
            },
        );
        let geom = Geom::attention(
            c.kind(layer),
            c.engine.head_dim,
            c.engine.qk_rope_head_dim,
            c.engine.rms_norm_eps as f32,
        )
        .expect("a compressed layer has a Geom");
        assert_eq!(
            geom.quantize(),
            Quantize::PartialFp8,
            "the ATTENTION compressor's finish is the partial fp8 one; the Hadamard-and-fp4 \
             finish belongs to the indexer's nested compressor and is scored by \
             kernel_v4_indexer.rs"
        );
        Self {
            wkv: Dev::u16(&bf16_rows(&cw.wkv)),
            wgate: Dev::u16(&bf16_rows(&cw.wgate)),
            ape: Dev::f32(&cw.ape),
            norm: Dev::f32(&cw.norm),
            cw,
            geom,
            cfg: c.oracle.clone(),
            dim: c.engine.hidden,
            layer,
        }
    }

    pub fn ratio(&self) -> usize {
        self.geom.ratio()
    }

    /// The rotary table for this layer, taken from the ORACLE under `defect` and flattened.
    ///
    /// Deliberately not rebuilt from `rope::table`. Two things follow. The kernel is handed the
    /// oracle's own table, so any disagreement below is arithmetic and never table construction
    /// — which is what makes the gap numbers mean the pooling. And `Defect::RopeNoYarn` becomes
    /// expressible as an INPUT: `Oracle::freqs` returns the ratio-0 table under that defect,
    /// which is exactly the substitution the no-yarn impersonation performs.
    ///
    /// The cost, stated: `rope::for_layer` — the selector the ENGINE would use — is not
    /// exercised here. Its own unit tests cover it, on the host, with no device involved.
    pub fn table(&self, defect: Defect) -> Vec<f32> {
        flat_freqs(Oracle::new(self.cfg.clone(), defect).freqs(self.layer))
    }

    /// Run BOTH implementations over the same script of calls and return
    /// `(oracle blocks, gpu blocks)` — one flat `[n, d]` vector each, in emission order.
    ///
    /// `steps` is `(seqlen, start_pos)` pairs, so a cell can be a single prefill, a prefill
    /// followed by many decodes, or anything else. Both sides walk the SAME script with the SAME
    /// activations and the same fresh state, which is what makes the comparison about the
    /// arithmetic rather than about the driving.
    ///
    /// `ape_over` and `freqs_over` substitute a kernel INPUT. They are how
    /// `Defect::CompressorNoApe` and `Defect::RopeNoYarn` are impersonated exactly.
    pub fn run(&mut self, r: Run<'_>) -> (Vec<f32>, Vec<f32>) {
        let o = Oracle::new(self.cfg.clone(), r.defect);
        let mut cs: CompState = o
            .fresh_state(self.layer)
            .comp
            .expect("a compressed layer has compressor state");
        let mut ctr = Counters::default();
        let d = self.geom.d();
        let (cd, ents) = (self.geom.cd(), self.geom.ents());
        let max_rows = r.steps.iter().map(|s| s.0).max().unwrap_or(1);

        let clean_table = self.table(Defect::None);
        let freqs = Dev::f32(r.freqs_over.unwrap_or(&clean_table));
        let ape_over = r.ape_over.map(Dev::f32);
        let mut bufs = Scratch {
            kv_state: Dev::filled(ents * cd, 0.0),
            score_state: Dev::filled(ents * cd, f32::NEG_INFINITY),
            kv: Dev::filled(max_rows * cd, 0.0),
            score: Dev::filled(max_rows * cd, 0.0),
            out: Dev::filled(max_rows.div_ceil(self.ratio()).max(1) * d, 0.0),
        };

        let (mut want, mut got) = (Vec::new(), Vec::new());
        for &(seqlen, start_pos) in r.steps {
            // Same activations to both sides. `fixed_bf16` is seeded by NAME, so the fixture is
            // reproducible and a rerun compares the same numbers.
            let tag = format!("l{}-s{seqlen}-p{start_pos}", self.layer);
            let x = fixed_bf16(&tag, seqlen * self.dim, 1.0);
            if let Some(v) = o.compressor(
                &self.cw,
                &mut cs,
                &x,
                seqlen,
                start_pos,
                o.freqs(self.layer),
                &mut ctr,
            ) {
                want.extend_from_slice(&v);
            }
            // ONE draw, handed to both sides. Re-drawing it for the device from the same tag
            // would be a second spelling of the seed, and a fixture the two implementations
            // disagreed about is the one failure no value comparison below could report.
            let at = Extent { seqlen, start_pos };
            let ape = ape_over.as_ref().unwrap_or(&self.ape);
            let inputs = Inputs {
                x: &Dev::f32(&x),
                ape,
                freqs: &freqs,
            };
            let n = self.step(at, &mut bufs, inputs);
            got.extend_from_slice(&bufs.out.read()[..n * d]);
        }
        assert_eq!(
            want.len(),
            got.len(),
            "the two implementations disagree on HOW MANY blocks are emitted, which no value \
             comparison would have reported"
        );
        (want, got)
    }

    /// One call of `Compressor.forward` on the device — the four launchers the engine's
    /// `LayerCompressor` runs, in its order.
    ///
    /// **The deposit runs on EVERY call, including one that emits nothing.** The reference writes
    /// `kv_state`/`score_state` in both phases and only THEN decides whether to emit, so a step
    /// that emits nothing still deposits. At ratio 128 that is every prompt under 128 tokens and
    /// 127 of every 128 decode steps, and skipping the call on a non-emitting step would build
    /// the pooling window out of every 128th token.
    ///
    /// The emission COUNT comes from `select::compress_dst` at region base 0, not from a second
    /// `seqlen / ratio` here: the engine bounds its two destinations with the same function, so a
    /// divergence between the count this harness reads back and the count the engine reserves is
    /// not expressible.
    /// `&self`, not `&mut self`: every mutation of this call belongs to `b`, and taking a unique
    /// borrow of the cell would forbid the caller from holding `&self.ape` as `t.ape` — which is
    /// exactly what the un-substituted path does.
    fn step(&self, at: Extent, b: &mut Scratch, t: Inputs<'_>) -> usize {
        let g = self.geom;
        let (cd, d, rd) = (g.cd(), g.d(), g.rd());
        // The shape contracts the launchers state only in prose, checked against the lengths
        // `Dev` carries. Without these this module would be asserting agreement between an oracle
        // and a kernel reading past the end of its inputs.
        assert!(
            b.kv.len >= at.seqlen * cd && b.score.len >= at.seqlen * cd,
            "scratch too small for {} rows",
            at.seqlen
        );
        assert_eq!(b.kv_state.len, g.state_len(), "state buffer is [ents, cd]");
        assert_eq!(t.ape.len, g.ratio() * cd, "ape is [ratio, coff*d]");
        assert_eq!(t.x.len, at.seqlen * self.dim, "x is [seqlen, dim]");
        let stream = stream();
        let st = stream.raw();
        // Every raw address taken ONCE, before the launches. Two `pm()` calls on one `Dev` inside
        // an argument list is a double unique borrow; hoisting also puts the aliasing question
        // where the SAFETY note can answer it, which is the reason rather than the borrow rule.
        let (kv, score) = (b.kv.pm(), b.score.pm());
        let (kv_state, score_state) = (b.kv_state.pm(), b.score_state.pm());
        let out = b.out.pm();
        let emitted = compress_dst(LayerKind::from_ratio(g.ratio()), 0, at).map_or(0, |(_, n)| n);
        // SAFETY: `x` is `seqlen * dim` live f32 (asserted) and both weights are `[cd, dim]` bf16
        // by `bf16_rows`; `kv`/`score` are `seqlen * cd` writable f32 (asserted); the two state
        // buffers are `state_len()` (asserted) and `ape` is `[ratio, cd]` (asserted); `out` is
        // `emitted * d`, since `Cell::run` sized it by `max_rows.div_ceil(ratio)` and
        // `compress_dst` divides by the same ratio. Every buffer is a distinct `Dev` allocation,
        // so none aliases another, and `device_sync` below completes every launch before any of
        // them is read or dropped.
        unsafe {
            for (w, dst) in [(&self.wkv, kv), (&self.wgate, score)] {
                gemm_bf16_launch(
                    GemmBf16 {
                        x: t.x.p(),
                        w: w.buf.ptr().cast(),
                        out: dst,
                        m: at.seqlen,
                        n: cd,
                        k: self.dim,
                    },
                    st,
                );
            }
            // On EVERY call, emitting or not — see this function's doc.
            launch_kv_compress_deposit(
                kv,
                score,
                t.ape.p(),
                kv_state,
                score_state,
                g.abi(),
                at.seqlen,
                at.start_pos % g.ratio(),
                st,
            )
            .expect("kv_compress_deposit");
            if emitted > 0 {
                let fin = CompFinish {
                    norm: self.norm.p(),
                    freqs: t.freqs.p(),
                    out,
                };
                // The two pool kernels bound to ONE result before it is checked, rather than
                // `.expect(...)` on each: the failure means the same thing in both arms, and the
                // exploded call rustfmt produced for the longer of them was a token-run clone of
                // the engine's own exploded finish launch — `build.rs`'s duplication gate reported
                // it, and collapsing both to one line each is the fix rather than an exemption.
                let pos = at.start_pos;
                let r = if at.is_prefill() {
                    launch_kv_compress_prefill(kv, score, t.ape.p(), &fin, g.abi(), emitted, st)
                } else {
                    launch_kv_compress_decode(kv_state, score_state, &fin, g.abi(), pos, st)
                };
                ok(r, "the pool kernel");
                finish_fp8(out, emitted, Widths::checked(d, rd), st);
            }
        }
        device_sync().expect("device_sync");
        emitted
    }
}

/// The compressor's PARTIAL fp8 finish: dims `[0, d - rd)` at block 64, over `rows` emitted
/// blocks — the extent and the order `model.py:373-378` uses.
///
/// Takes [`Widths`] rather than the two extents loose, and that is the same argument the metric
/// makes about its own buckets: `d` and `rd` are interchangeable to the type checker, and passing
/// the WHOLE row here is the INDEXER's finish wearing this one's clothes — finite, plausible and
/// wrong. `Cell::load` asserts the `Quantize` so the choice is made once, at load; this is where
/// the extent that choice implies is spelled.
///
/// A named function and not the launch inline, for a mechanical reason worth recording: inline,
/// the seven-argument call is a token run `build.rs`'s duplication gate matched against the
/// ENGINE's own finish launch in `v4/kvcompress.rs`. The gate was right that the two are the same
/// call; the fix is to give this side a shape of its own rather than an exemption, and routing the
/// extent through `Widths` is the shape that also improves it.
///
/// # Safety
/// `out` is `rows * w.d` writable, device-resident f32, read and written in place, outliving
/// `st`'s completion. `st` is a live `hipStream_t` or null.
unsafe fn finish_fp8(out: *mut f32, rows: usize, w: Widths, st: *mut c_void) {
    let span = w.d - w.rd;
    // SAFETY: the caller's contract above; `span < w.d` by `Widths::checked`.
    ok(
        unsafe { launch_act_quant_f8_prefix(out, out, rows, w.d, span, KV_QUANT_BLOCK, st) },
        "act_quant_f8_prefix",
    );
}

/// The five per-call device buffers: the pooling state, this call's projections, and the blocks.
///
/// Named `state_*`/`proj_*`-style rather than after the roles the kernels give them, on the
/// engine's own argument: all four `[.., cd]` buffers are f32 and two are the pooling STATE while
/// two are this call's PROJECTIONS, so those are the pair most worth being unable to confuse.
struct Scratch {
    kv_state: Dev,
    score_state: Dev,
    kv: Dev,
    score: Dev,
    out: Dev,
}

/// The three read-only device inputs one call is handed: this step's activation and the two
/// tables the finish reads.
///
/// A struct rather than three `&Dev` arguments, and the reason is that nothing downstream can tell
/// them apart: `ape` is `[ratio, cd]`, `freqs` is the rotary table and `x` is `[seqlen, dim]`, all
/// f32 and all reached through a `*const f32`. `x` joined the pair 2026-08-16 when CodeScene's
/// excess-argument rule priced [`Cell::step`]'s fifth parameter — which is the same argument this
/// struct was already making, applied to the one input that had escaped it.
#[derive(Clone, Copy)]
struct Inputs<'a> {
    x: &'a Dev,
    ape: &'a Dev,
    freqs: &'a Dev,
}

/// One [`Cell::run`]: the defect the ORACLE runs under, the script both sides walk, and the two
/// optional KERNEL-input substitutions that make two of those defects exactly impersonable.
///
/// A struct because `ape_over` and `freqs_over` are both `Option<&[f32]>` and swapping them
/// compiles — and the swap is silent: the kernel would be handed a rotary table as its positional
/// embedding and vice versa, producing finite numbers from the wrong tensors.
#[derive(Clone, Copy)]
pub struct Run<'a> {
    pub defect: Defect,
    /// `(seqlen, start_pos)` per call — the same pair `Compressor.forward` takes.
    pub steps: &'a [(usize, usize)],
    pub ape_over: Option<&'a [f32]>,
    pub freqs_over: Option<&'a [f32]>,
}

impl<'a> Run<'a> {
    /// The clean run over `steps` — the baseline every distance is measured from.
    pub fn clean(steps: &'a [(usize, usize)]) -> Self {
        Self {
            defect: Defect::None,
            steps,
            ape_over: None,
            freqs_over: None,
        }
    }
}

// =======================================================================================
// the four cells
// =======================================================================================

/// One cell: which layer, what script, and the name every measurement is filed under.
///
/// `ratio4/prefill` is 256 tokens; `ratio128/prefill` is 300, which is the only prefill path that
/// writes state (`300 % 128 == 44`). A 256-token ratio-128 prefill writes NO state — `overlap` is
/// false and `256 % 128 == 0`, so both state writes are skipped.
///
/// Both decode scripts run to their SECOND completed block. Stopping at the first cannot see the
/// RoPE position rule; see [`RATIO_128_SECOND_DECODE_BLOCK`].
pub struct Spec {
    pub layer: usize,
    pub script: Vec<(usize, usize)>,
    pub name: &'static str,
}

/// A spec's cell, loaded, together with its CLEAN `(oracle, gpu)` pair.
///
/// The opening move of every sweep, and the pairing that must not drift: every distance any sweep
/// reports is measured from this baseline, and a loop that loaded one spec's cell and baselined
/// another's script would produce numbers that are meaningless and entirely plausible.
///
/// The cell comes back LIVE because most callers go on to `run` it under a defect, and
/// re-`load`ing would re-read the layer's weights.
pub fn load_and_baseline(ck: &Checkpoint, c: &Configs, spec: &Spec) -> (Cell, Vec<f32>, Vec<f32>) {
    let mut cell = Cell::load(ck, c, spec.layer);
    let (want, got) = cell.run(Run::clean(&spec.script));
    (cell, want, got)
}

/// The checkpoint, the config pair, and the four cells — or `None` when the checkpoint is absent.
///
/// Every test walks this list, so a cell added here is added to the clean comparison, the two
/// impersonations and the separation sweep at once — and a cell cannot be exercised by one of
/// them and quietly missing from another.
pub fn cells() -> Option<(Checkpoint, Configs, Vec<Spec>)> {
    let c = Configs::new()?;
    let ck = Checkpoint::open(std::path::Path::new(super::common::CKPT)).expect("checkpoint");
    assert_eq!(c.ratio(2), 4, "layer 2 is the overlapping class");
    assert_eq!(c.ratio(3), 128, "layer 3 is the non-overlapping class");
    let spec = |layer, script, name| Spec {
        layer,
        script,
        name,
    };
    let list = vec![
        spec(2, vec![(PROBE_LEN, 0)], "ratio4/prefill"),
        spec(2, decode_script(4, 23), "ratio4/decode"),
        spec(
            3,
            vec![(PROBE_REMAINDER_LEN, 0)],
            "ratio128/prefill+remainder",
        ),
        spec(
            3,
            decode_script(128, RATIO_128_SECOND_DECODE_BLOCK),
            "ratio128/decode",
        ),
    ];
    Some((ck, c, list))
}

/// A short prefill followed by single-row decodes up to and including `last`.
///
/// The assertion is the point of the function: a decode script that completes fewer than two
/// blocks cannot distinguish the RoPE position `(start_pos / ratio) * ratio` from the block index
/// `start_pos / ratio`, because at the first block both are 0 and the rotation is the identity.
/// Building the script by hand is how that gets forgotten.
pub fn decode_script(ratio: usize, last: usize) -> Vec<(usize, usize)> {
    let mut v = vec![(EMIT_LEN, 0)];
    v.extend((EMIT_LEN..=last).map(|p| (1, p)));
    assert!(
        v.iter()
            .filter(|&&(s, p)| s == 1 && (p + 1).is_multiple_of(ratio))
            .count()
            >= 2,
        "a decode script must complete at least two blocks, else the RoPE position and the block \
         index cannot be told apart"
    );
    v
}
