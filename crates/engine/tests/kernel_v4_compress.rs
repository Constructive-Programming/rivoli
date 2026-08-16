//! **The V4-Flash attention compressor's three kernels, scored against the frozen host oracle**
//! at the real checkpoint's own tensors — `kv_compress_deposit`, `kv_compress_prefill`,
//! `kv_compress_decode`, plus the `act_quant_f8_prefix` finish they owe.
//!
//! Four cells: `{ratio 4, ratio 128} x {prefill, decode}`. Ratio 4 is the overlapping branch with
//! `ape[4, 1024]`; ratio 128 is the non-overlapping one with `ape[128, 512]`. A shape assumption
//! that holds on layer 2 breaks on layer 3, which is why both are here and why
//! `common::compressor_w` asserts the widths at load.
//!
//! Ported from `old:tests/kvcompress_kernel.rs`. The harness — the metric, the cell, the launch
//! sequence — is `v4_compressor/mod.rs`, shared with `kernel_v4_compress_defects.rs`; its header
//! carries what changed in the port and why. This file holds the CLEAN comparison and the two
//! exact impersonations; the separation sweep and its coverage registry are next door.
//!
//! # How a green result here is made to mean something
//!
//! Every defect in this path is silent-wrong. So agreement with the oracle is necessary and
//! nowhere near sufficient, and most of this suite's length is the other half — showing the
//! comparison can REJECT. The strongest technique available without shipping a break switch in a
//! kernel is **exact defect impersonation**: two of the oracle's breakages are expressible as a
//! change to a kernel INPUT rather than to the kernel. `Defect::CompressorNoApe` is `ape` zeroed,
//! and `Defect::RopeNoYarn` is the ratio-0 rotary table in place of the compressed one. For those
//! two the kernel is fed the perturbed input and required to match the oracle *running with that
//! defect*, to the same tolerance it matches the clean oracle — and to be FAR from the clean
//! oracle. That is a real red/green, proved at the bit level.
//!
//! The weaker techniques — distance separation for the breakages that live inside the kernel, and
//! named inertness — are `kernel_v4_compress_defects.rs`'s.
//!
//! # What this file provably cannot detect — read this before trusting it
//!
//! * **Anything the oracle is also wrong about.** The kernels were written from `model.py` AND
//!   from the oracle's transliteration of it; a shared misreading is invisible here by
//!   construction.
//! * **The INDEXER's compressor** (`rotate = true`: Hadamard + fp4 instead of the partial fp8).
//!   Not exercised here; `kernel_v4_indexer.rs` scores it.
//! * **`expf` agreement.** The pooling softmax calls `expf` on device and `f32::exp` on the host.
//!   They are not required to agree bit-for-bit and the tolerance absorbs the difference, so a
//!   softmax wrong by less than that is invisible. The separations the sibling file measures say
//!   how much room that leaves.
//! * **Whether `act_quant`'s subnormal e4m3 ties are reached at all.** Nothing in these fixtures
//!   lands on them, so this file exercises the COMMON path of that kernel and not its corners —
//!   `kernel_v4_quant.rs` owns the corners.
//! * **`Defect::RopeNoYarn` at `ratio4/decode`.** The impersonation is perfect there and the
//!   separation from clean is 8 codes, so that cell cannot tell "wrong table" from "no table".
//!   Stated plainly because `RopeNoYarn` is a named port requirement: **this suite cannot see it
//!   at RATIO-4 decode.** Not "at decode" — `ratio128/decode` is absent from
//!   [`NO_YARN_BELOW_RESOLUTION`] and is still required to separate. `ratio4/prefill` separates at
//!   31,215 and is what gates the requirement.
//!
//! Skips with a printed reason when the checkpoint is absent — there is no CI here and this reads
//! 5.3 MB of index metadata off a 167 GB checkpoint.
//!
//! # RED-PROOF PLAN — for the integrator's first device run
//!
//! Never executed: no `rocm` CI arm, and no checkpoint or GPU for this port. Two mutations, in
//! `kernels/kvcompress.hip`:
//!
//! * In `kv_compress_prefill`'s finish, RoPE each block at its block INDEX (`blk`) instead of at
//!   its first absolute position (`blk * ratio`). [`the_four_cells_reproduce_the_oracle`] must go
//!   RED on `ratio128/prefill+remainder` and on `ratio4/prefill`, in the **RoPE TAIL** —
//!   `max_tail` far over `CLEAN_ULP = 2`, which is the diagnostic split doing its job: the tail
//!   is the region `act_quant` never touches, so a failure confined to it is the rotation and
//!   cannot be a quantization artefact. The two decode cells must stay green, which is what
//!   attributes the failure to the prefill pool.
//! * Delete the `ape` add from `kv_compress_deposit`. [`the_four_cells_reproduce_the_oracle`]
//!   must go RED on all four cells, and
//!   [`zeroing_ape_reproduces_the_no_ape_defect_exactly`] must go **GREEN on its first assertion
//!   and RED on its second** — the kernel now IS the no-ape kernel, so it lands on the
//!   defect-injected oracle and no longer separates from the clean one. That asymmetry is the
//!   whole point of the two-sided assertion, and a mutation that reddens both halves has broken
//!   something other than `ape`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use rivoli_engine::v4::geometry::{Geom, LayerKind, Quantize};

mod common;
mod v4_compressor;

use v4_compressor::{
    Cell, Defect, Diff, E4M3_ULP, PROBE_LEN, RESOLVABLE, Run, Widths, assert_clean, cells, diff,
    gap, load_and_baseline,
};

/// The `RopeNoYarn` impersonation cells that land inside the quantization floor, MEASURED.
///
/// **This is not bookkeeping.** `Io.freqs` is a raw pointer that cannot tell the ratio-0 table
/// from the YaRN one, and mixing them is fluent wrong output. At `ratio4/decode` the
/// impersonation is *perfect* (max=0 against the defect-injected oracle, bit-identical) and the
/// separation from the clean oracle is only **8 codes**, half an e4m3 step: the cell cannot
/// distinguish "consulted the wrong table" from "did not consult the table at all".
///
/// The mechanism: `ratio4/decode` is `decode_script(4, 23)`, so the run concatenates the 3
/// prefill blocks with the ones completing at positions 15, 19 and 23 — six blocks spanning
/// 0..23, and the separation is the max over all of them. Every one of those positions is small
/// enough that the ratio-0 and YaRN tables barely diverge.
///
/// **An EXPECTED VALUE, not a skip.** The entry is asserted to reproduce its recorded separation
/// exactly, so a cell that stops being unresolvable still fires; and it must be REACHED, so a
/// stale entry cannot silently swallow a case that no longer occurs.
const NO_YARN_BELOW_RESOLUTION: &[(&str, u32)] = &[("ratio4/decode", 8)];

/// Every recorded non-separation must actually sit BELOW the floor, and must name a cell this run
/// visits.
///
/// Without the first half the assertion below is only `sep == want`, so an entry of 31215 would
/// pass — and the failure message would still print "(inside the quantization floor)", a false
/// claim emitted by the assertion itself. That is the exclusion list absorbing a SEPARATING cell,
/// which is the exact failure a coverage registry exists to prevent.
///
/// The second half is the anti-vacuity: a single entry is not a reason to skip it. If
/// `ratio4/decode` ever leaves the cell list, this record would sit there looking like considered
/// non-coverage while naming nothing.
fn assert_no_yarn_records_are_well_formed(cells: &[&str]) {
    for (c, s) in NO_YARN_BELOW_RESOLUTION {
        assert!(
            *s < RESOLVABLE,
            "NO_YARN_BELOW_RESOLUTION {c} records {s} >= {RESOLVABLE} — a separating cell must \
             not be recorded as non-coverage"
        );
        assert!(
            cells.contains(c),
            "NO_YARN_BELOW_RESOLUTION records {c} at sep={s}, but this run has no such cell"
        );
    }
}

/// Ratio 4 (layer 2) and ratio 128 (layer 3), prefill and decode, against the clean oracle.
#[test]
fn the_four_cells_reproduce_the_oracle() {
    let Some((ck, c, list)) = cells() else {
        return;
    };
    // Every cell reports BEFORE anything asserts. The reference tree's first run asserted inside
    // the loop and failed on cell 1 of 4, so three quarters of the diagnostic never printed and
    // the failure could not be told from a phase-dependent one. A gate that aborts on its first
    // cell hands back a quarter of what it measured.
    let w = Widths::of(&c.engine);
    let mut over = Vec::new();
    for spec in &list {
        let name = spec.name;
        let (_, want, got) = load_and_baseline(&ck, &c, spec);
        assert!(
            !want.is_empty(),
            "{name}: the script emitted nothing — it gates nothing"
        );
        assert!(
            got.iter().all(|v| v.is_finite()),
            "{name}: non-finite output"
        );
        let dv = diff(&want, &got, w);
        println!("{}", dv.one_line(name));
        over.extend(assert_clean(name, &dv));
    }
    assert!(
        over.is_empty(),
        "clean comparison failed:\n  {}",
        over.join("\n  ")
    );
}

/// The ratio-128 prefill at 256 reads NO compressor state — re-proved against the GPU.
///
/// A previous session of this port asserted a block-to-block state carry here that does not
/// exist, and two reviewers disproved it by substituting zero-length state buffers and getting
/// bit-identical output. The technique is what is worth keeping, so it is applied to the kernel:
/// `Cell::run` allocates fresh state per call, so two identical runs must agree bit-for-bit.
///
/// Scoped to a whole multiple of the ratio on purpose — at 300 tokens the remainder path DOES
/// write state, which is why the cell list uses 300 for the ratio-128 prefill cell.
#[test]
fn state_is_not_read_by_the_ratio_128_prefill_at_a_whole_multiple() {
    let Some((ck, c, _)) = cells() else { return };
    assert_eq!(
        PROBE_LEN % 128,
        0,
        "the claim is scoped to a whole multiple of the ratio"
    );
    let script = vec![(PROBE_LEN, 0)];
    let mut cell = Cell::load(&ck, &c, 3);
    let (_, base) = cell.run(Run::clean(&script));
    let (_, again) = cell.run(Run::clean(&script));
    assert!(!base.is_empty(), "256 tokens pools two ratio-128 blocks");
    assert_eq!(
        base, again,
        "the harness is not deterministic — nothing else here is evidence"
    );
}

/// The two-sided assertion both impersonations make: the GPU lands ON the defect-injected oracle,
/// and FAR from the clean one.
///
/// Both halves are required and neither alone is worth anything. Without the first, the
/// perturbation is only known to have changed something. Without the second, a kernel that
/// IGNORED the perturbed input entirely — the exact failure being hunted — would pass, because
/// the clean and defect oracles would be close and it would match both.
///
/// [`Impersonation`] names WHICH perturbation on WHICH cell, and what the separation is expected
/// to be.
fn assert_impersonates(i: Impersonation<'_>, m: Measured<'_>, w: Widths) {
    let Impersonation {
        cell,
        what,
        expect_sep,
    } = i;
    let hit = diff(m.broken, m.gpu, w);
    println!(
        "{}",
        hit.one_line(&format!("{cell} {what}: gpu vs defect-oracle"))
    );
    let bad = assert_clean(&format!("{cell} {what} impersonation"), &hit);
    assert!(
        bad.is_empty(),
        "the {what} perturbation must land on the oracle's own defect to within the quantization \
         floor (NOT bit-exactly — `act_quant` makes one e4m3 step the smallest expressible \
         disagreement):\n  {}",
        bad.join("\n  ")
    );
    let sep = gap(
        &format!("{cell} {what}: gpu vs CLEAN oracle"),
        m.clean,
        m.gpu,
        w,
    );
    match expect_sep {
        // Recorded as non-separating. Asserted as an EXACT expected value, not skipped: if this
        // cell ever gains resolution the record is stale and must be revisited, and a silent skip
        // would let that pass as coverage it is not.
        Some(want) => assert_eq!(
            sep, want,
            "{cell}: the {what} separation is RECORDED as {want} codes (inside the \
             {RESOLVABLE}-code quantization floor, so this cell cannot see the defect). It \
             measured {sep}. Either the kernel changed or the fixture did — update the {what} \
             registry and say why."
        ),
        None => assert!(
            sep >= RESOLVABLE,
            "{cell}: the {what} perturbation moved the output by only {sep} codes — under \
             {RESOLVABLE} ({} e4m3 steps) it is not distinguishable from the quantization floor, \
             so this cell cannot see whether the input is consulted at all",
            RESOLVABLE / E4M3_ULP
        ),
    }
}

/// The three block sets one impersonation compares: the clean oracle, the defect-injected oracle,
/// and the GPU run under the perturbed input.
///
/// A struct because all three are `&[f32]` of the same length and any permutation compiles —
/// and the permutations are not equally wrong in an obvious way: `(broken, clean, gpu)` would
/// assert that the two ORACLES agree, which they do to within the floor on the cell this file
/// records as unresolvable, so that swap goes green on exactly the case the registry exists for.
/// Which perturbation, on which cell, and what its separation from the clean oracle is expected
/// to be.
///
/// A struct because `cell` and `what` are two `&str` in a row — swapped, the failure message names
/// the perturbation where a reader expects the cell and sends them to the wrong fixture — and
/// because `expect_sep` belongs WITH them: `None` means this cell must separate and `Some(n)` means
/// it is recorded as landing inside the quantization floor at exactly `n` codes, which is a fact
/// about the (cell, perturbation) pair and about nothing else.
#[derive(Clone, Copy)]
struct Impersonation<'a> {
    cell: &'a str,
    what: &'a str,
    expect_sep: Option<u32>,
}

impl<'a> Impersonation<'a> {
    /// A constructor and not a literal per call site: rustfmt's `struct_lit_width` turns every
    /// `Impersonation { .. }` into one line per field, and the two impersonations' four-line runs
    /// are then a clone `build.rs`'s duplication gate reports. Positional is safe here in a way it
    /// is not at a launch — the three arguments read left to right as the sentence the failure
    /// message prints.
    fn new(cell: &'a str, what: &'a str, expect_sep: Option<u32>) -> Self {
        Self {
            cell,
            what,
            expect_sep,
        }
    }
}

#[derive(Clone, Copy)]
struct Measured<'a> {
    clean: &'a [f32],
    broken: &'a [f32],
    gpu: &'a [f32],
}

impl<'a> Measured<'a> {
    /// A constructor and not a literal per call site: rustfmt's `struct_lit_width` turns every
    /// `Measured { .. }` into one line per field, and the two impersonations' literals then differ
    /// in nothing at all — which `build.rs`'s duplication gate reports, correctly. Positional here
    /// is safe in a way it is not at the launch: the three arguments are bound on the three lines
    /// above every call, in this order, from names that say which is which.
    fn new(clean: &'a [f32], broken: &'a [f32], gpu: &'a [f32]) -> Self {
        Self { clean, broken, gpu }
    }
}

/// **`ape` is load-bearing, proved exactly.** Zeroing the position embedding is precisely
/// `Defect::CompressorNoApe`, so the kernel fed a zero `ape` must reproduce the oracle running
/// with that defect — to the same tolerance as the clean comparison — while being far from the
/// clean oracle.
///
/// It does not merely show the output moved: it shows it moved *to the specific wrong place the
/// oracle says a missing `ape` produces*. A kernel that ignored `ape` entirely would pass the
/// first assertion and fail the second.
#[test]
fn zeroing_ape_reproduces_the_no_ape_defect_exactly() {
    let Some((ck, c, list)) = cells() else {
        return;
    };
    let w = Widths::of(&c.engine);
    for spec in &list {
        let (mut cell, clean, _) = load_and_baseline(&ck, &c, spec);
        let zeros = vec![0.0f32; cell.cw.ape.len()];
        let (broken, gpu) = cell.run(Run {
            defect: Defect::CompressorNoApe,
            ape_over: Some(&zeros),
            ..Run::clean(&spec.script)
        });
        // `None` — every cell must separate on `no-ape`, and every cell does. Spelled inline,
        // where the no-yarn arm binds its `Impersonation` first: the two tails are otherwise the
        // same token run and `build.rs`'s duplication gate reports them, correctly, since the
        // difference between the two impersonations is entirely in what precedes this line.
        assert_impersonates(
            Impersonation::new(spec.name, "no-ape", None),
            Measured::new(&clean, &broken, &gpu),
            w,
        );
    }
}

/// **The rotary table SELECTION is load-bearing, proved exactly.** Handing the kernel the ratio-0
/// table (base `rope_theta`, no YaRN) in place of the compressed one is precisely
/// `Defect::RopeNoYarn`, so the kernel must land where the oracle-with-that-defect lands.
///
/// This is the hazard `CompFinish` exists to name — `freqs` is a raw pointer that cannot
/// distinguish the two tables, which have the same type, stride and shape — measured rather than
/// argued.
#[test]
fn the_ratio_0_rope_table_reproduces_the_no_yarn_defect_exactly() {
    let Some((ck, c, list)) = cells() else {
        return;
    };
    assert_no_yarn_records_are_well_formed(&list.iter().map(|s| s.name).collect::<Vec<_>>());
    let w = Widths::of(&c.engine);
    for spec in &list {
        let name = spec.name;
        let (mut cell, clean, _) = load_and_baseline(&ck, &c, spec);
        let plain = cell.table(Defect::RopeNoYarn);
        assert_ne!(
            plain,
            cell.table(Defect::None),
            "{name}: the two tables must differ, else this substitutes nothing"
        );
        // The substitution bound before the run rather than built in the argument list: the
        // inline form is token-identical to the `no-ape` one next door once rustfmt explodes it,
        // which `build.rs`'s duplication gate reports — and naming it also puts the ONE field that
        // differs between the two impersonations on a line of its own.
        let run = Run {
            defect: Defect::RopeNoYarn,
            freqs_over: Some(&plain),
            ..Run::clean(&spec.script)
        };
        let (broken, gpu) = cell.run(run);
        let expect = NO_YARN_BELOW_RESOLUTION
            .iter()
            .find(|(cell_name, _)| *cell_name == name)
            .map(|(_, s)| *s);
        let imp = Impersonation::new(name, "no-yarn", expect);
        assert_impersonates(imp, Measured::new(&clean, &broken, &gpu), w);
    }
}

/// `Geom` refuses `Plain`, and the two live geometries disagree on both derived fields in
/// OPPOSITE directions — the shape trap stated as an inequality rather than as prose.
///
/// A guard nobody proves can fire is a guard that might be `if (false)`. Needs no device and no
/// checkpoint: it is arithmetic over a layer class.
#[test]
fn the_two_geometries_differ_in_opposite_directions() {
    assert!(
        Geom::attention(LayerKind::Plain, 512, 64, 1e-6).is_none(),
        "a ratio-0 layer has no Compressor object in the reference and must have no Geom"
    );
    let g4 = Geom::attention(LayerKind::Overlap, 512, 64, 1e-6).unwrap();
    let g128 = Geom::attention(LayerKind::NonOverlap(128), 512, 64, 1e-6).unwrap();
    assert_eq!((g4.cd(), g4.ents(), g4.state_len()), (1024, 8, 8192));
    assert_eq!(
        (g128.cd(), g128.ents(), g128.state_len()),
        (512, 128, 65536)
    );
    assert!(
        g4.cd() > g128.cd() && g4.ents() < g128.ents(),
        "ratio 128 HALVES the projection width and multiplies the window — a loader that inferred \
         one from the other would be right on exactly one of the two layers"
    );
    // And both owe the PARTIAL fp8 finish, which is what separates them from the indexer's nested
    // compressor at the identical dimensions. `kernel_v4_indexer.rs` asserts the other side.
    for g in [g4, g128] {
        assert_eq!(g.quantize(), Quantize::PartialFp8);
    }
}

/// **`Widths::checked` rejects a transposed pair — the proof that guard can fire.**
///
/// Why an `assert!` and not the subtraction is argued once, at `Widths`. This is the other half:
/// the replacement must itself be exercised, or it is the same shape of claim one level up.
///
/// LOOSENING is the failure mode the pair closes, and it needs a case per conjunct. Every other
/// test here passes a legitimate `(512, 64)`, so a bound relaxed in either direction stays green
/// for all of them; an INVERTED condition would go red everywhere at once.
#[test]
#[should_panic(expected = "transposed")]
fn widths_rejects_a_transposed_pair() {
    // The shipped shape with its two fields swapped: `rope_head_dim` 64 is the tail INSIDE
    // `head_dim` 512, so (64, 512) is exactly the mistake.
    let _ = Widths::checked(64, 512);
}

/// ...and a zero tail is rejected too — the half a transposition cannot produce, which is why it
/// matches its OWN message rather than the transposition one.
#[test]
#[should_panic(expected = "empty RoPE tail")]
fn widths_rejects_an_empty_rope_tail() {
    let _ = Widths::checked(512, 0);
}

/// **The instrument, before it is trusted to diagnose anything.** Needs no device and no
/// checkpoint.
///
/// The reference tree's first run reported 16 ULP and an explanation was proposed for it. An
/// explanation resting on a metric nobody had exercised is a guess wearing a number, so this pins
/// what `diff` reports for differences whose answers are known independently: exactly one e4m3
/// step, the same step in the tail, and a value near zero.
#[test]
fn the_diff_metric_reports_what_it_claims() {
    // Synthetic widths, not the config's: this pins what the METRIC reports, so the shape only
    // has to be a legal one.
    let w = Widths::checked(512, 64);
    // One e4m3 step at 1.5 -> 1.625. `act_quant` reconstructs `e4m3(v/s)·s`, so a
    // pre-quantization value a hair either side of the boundary lands a whole step away.
    //
    // EXACTLY 16, and the arithmetic is independent of `diff`: bf16 codes 0x3FC0 and 0x3FD0
    // differ by 0x10. An earlier version of this test derived 14.8 from `log2(1.625/1.5)·128` and
    // asserted `14..=16` — bf16 codes are LINEAR in the mantissa, not logarithmic, so that
    // derivation was wrong and the range it produced would have passed an implementation that
    // violated the one claim this test exists to pin.
    assert_eq!(
        rivoli_core::num::f32_to_bf16(1.5),
        0x3FC0,
        "the fixture's own premise"
    );
    assert_eq!(rivoli_core::num::f32_to_bf16(1.625), 0x3FD0);
    let at = |i: usize, a: f32, b: f32| -> Diff {
        let (mut want, mut got) = (vec![1.0f32; w.d], vec![1.0f32; w.d]);
        want[i] = a;
        got[i] = b;
        diff(&want, &got, w)
    };
    let dv = at(3, 1.5, 1.625);
    assert_eq!(
        dv.max, E4M3_ULP,
        "one e4m3 step must read as exactly {E4M3_ULP} bf16 codes"
    );
    assert!(
        (dv.worst_ratio() - 1.0833).abs() < 0.001,
        "ratio {}",
        dv.worst_ratio()
    );
    assert_eq!(dv.differing, 1, "exactly one element moved");
    assert_eq!(
        (dv.max_quant, dv.max_tail),
        (dv.max, 0),
        "dim 3 is inside the quantized region"
    );

    // Binade-independence, which is the half `E4M3_ULP` actually rests on. If this held only near
    // 1.0 then the constant would be a coincidence of the fixture rather than a property of the
    // two formats, and the whole diagnosis above it would be unfounded.
    for e in [-4i32, -1, 0, 3, 9] {
        let base = 2.0f32.powi(e);
        for m in 0..7 {
            let (a, b) = (
                base * (1.0 + m as f32 / 8.0),
                base * (1.0 + (m + 1) as f32 / 8.0),
            );
            assert_eq!(
                at(1, a, b).max,
                E4M3_ULP,
                "one e4m3 step at 2^{e}·(1+{m}/8) must still be {E4M3_ULP} codes"
            );
        }
    }

    // The SPLIT is the whole diagnostic, so it must actually split. The same difference, moved
    // into the RoPE tail, has to land in the other bucket — otherwise a tail-only failure would
    // read as a quantization artifact and send the next reader the wrong way.
    let dv2 = at(w.d - 1, 1.5, 1.625);
    assert_eq!(
        (dv2.max_quant, dv2.max_tail),
        (0, dv2.max),
        "dim 511 is the untouched tail"
    );

    // The known blind spot, pinned rather than described: near zero the code gap is large for a
    // negligible absolute difference. `differing` and `worst` are what distinguish it, which is
    // why the verdict must never rest on `max` alone.
    let dv3 = at(7, 1e-30, 2e-30);
    assert!(
        dv3.max >= 100,
        "a doubling near zero reads as a whole binade: {}",
        dv3.max
    );

    // Identical input must read exactly zero, or every 0 printed anywhere above means nothing.
    let same = diff(&vec![1.0f32; w.d], &vec![1.0f32; w.d], w);
    assert_eq!((same.max, same.differing), (0, 0));
    println!("{}", dv.one_line("e4m3-step fixture"));
}
