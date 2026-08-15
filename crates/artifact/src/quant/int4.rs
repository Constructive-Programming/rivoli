//! The **group-scaled int4** routed-expert format: its geometry, its quantizer, its dense
//! inverse, the scalar oracle the HIP kernels are validated against, and the single writer
//! of its slot layout.
//!
//! **Split out of `quant.rs` on 2026-08-15, by FORMAT** — see [`super::vq`] for the measured
//! reason (CodeScene 8.54, LCOM4 over a file that had grown four formats with no call edge
//! between them). Bodies and comments travelled verbatim; the module comment below carries
//! the PPL 73.4 measurement that rejected the per-row scale, which is the whole reason this
//! format is shaped the way it is.
//!
//! The block geometry it shares with `.vq3` and `.f4` (`expert_bytes`, `expert_stride`,
//! `slot_offsets`) stays in the parent — one walk, three formats. Every public name is
//! re-exported by `quant.rs`, so `bin/fp8_to_i4` and the engine's kernel tests are
//! unchanged.

use super::{
    RowScaledW, debug_check_gemv, expert_bytes, expert_stride, slot_offsets, write_le_scales,
};

// ── int4 (group-scaled): the "warm expert" format ───────────────────────────
// Symmetric int4 with one f32 scale per [`I4_GROUP`] weights ALONG THE INPUT DIM:
// `W[o,i] = (nibble(o,i) − 8) · scale[o·ngroups + i/I4_GROUP]`. `packed` = o_dim rows
// of `i4_row_bytes(i_dim)` bytes (LOW nibble = col 2j, HIGH = col 2j+1).
//
// This replaced a PER-ROW scale (`scale[o]`, i.e. one step for all 6144 weights of a
// gate/up row), which is a known-bad design point and measured like one: a single
// outlier set the step for the whole row and rounded the bulk toward zero — 603 of
// 6144 rows past 50% zeros on L03 e0 down_proj, and `--mode int4` PPL 73.4 against
// int3-vq's 5.28. `.vq3` already carries one scale per `VQ_GROUP` = 64 weights, which
// is why it does not suffer this. Group scales are what the int4 literature and every
// published int4 GLM checkpoint (AWQ/GPTQ, `group_size=128`) actually use.
//
// The scale therefore lives INSIDE the dot (each group's partial is scaled before it
// is accumulated), not outside it — see `dot_i4_wave` in kernels/common.hpp.

/// Weights per f32 int4 group scale, along the input dimension. 128 is the
/// AWQ/GPTQ/Marlin default; 64 (what `.vq3` uses) is the other point worth sweeping.
/// A multiple of 8, so the GPU dot's 8-nibble dword fast path never straddles a group.
pub const I4_GROUP: usize = 128;

/// Number of f32 group scales in one int4 row (`ceil(i_dim / I4_GROUP)`).
pub fn i4_groups(i_dim: usize) -> usize {
    i_dim.div_ceil(I4_GROUP)
}

/// Row stride in bytes for an int4 matrix (2 nibbles/byte).
pub fn i4_row_bytes(i_dim: usize) -> usize {
    i_dim.div_ceil(2)
}

// (The parameter-list jscpd exemption that stood here died 2026-08-15: the `Fp8W`/`VqW`/
// `I4W`/`I8W` views above ARE the fix its own note named, and with the lists gone there
// is nothing left to exempt.)
//
// CORRECTED 2026-08-15, same day, by the split that moved this function out of `quant.rs`:
// "above" now means the PARENT module, not this file, and there is no `I4W` or `I8W` — the
// two twin field lists were folded into one `RowScaledW`, which is what the sentence was
// about. Kept and corrected rather than reworded away, because a note whose landmarks have
// moved is the same hazard as an exemption whose argument names a deleted file: the reader
// goes looking for something that is not there and concludes the note is stale in some
// other way too.
/// Reference int4 GEMV `y[o] = Σ_i x[i]·(nibble(o,i) − 8)·scale[o, i/I4_GROUP]` — the
/// CPU oracle the `moe_gateup_i4`/`moe_down_i4` kernels validate against. `scale` is
/// `o_dim · i4_groups(i_dim)` f32, row-major.
pub fn matvec_i4(y: &mut [f32], x: &[f32], w: RowScaledW<'_>, shape: [usize; 2]) {
    let RowScaledW { packed, scale } = w;
    let [o_dim, i_dim] = shape;
    let (rb, ng) = (i4_row_bytes(i_dim), i4_groups(i_dim));
    debug_check_gemv(y, x, o_dim, i_dim);
    debug_assert_eq!(scale.len(), o_dim * ng);
    for (o, yo) in y.iter_mut().enumerate() {
        let row = &packed[o * rb..(o + 1) * rb];
        let srow = &scale[o * ng..(o + 1) * ng];
        let mut acc = 0.0f32;
        for (i, &xi) in x.iter().enumerate() {
            let byte = row[i >> 1];
            let n = (if i & 1 == 0 { byte & 0x0F } else { byte >> 4 }) as i32 - 8;
            acc += xi * n as f32 * srow[i / I4_GROUP];
        }
        *yo = acc;
    }
}

/// Quantize `w[o_dim·i_dim]` (row-major) → group-scaled symmetric int4 (packed bytes
/// plus `o_dim · i4_groups(i_dim)` f32 scales). Per group of [`I4_GROUP`] weights along
/// the input dim, `s = max|group|/7` so the group's extreme maps to nibble 15 (value +7);
/// nibbles clamp to `[0,15]`. Round-trips through [`matvec_i4`].
//  (Rewrapped so no line STARTS with `+`: rustdoc reads a leading `+ ` as a list bullet,
//  which is what `clippy::doc_lazy_continuation` was flagging.)
///
/// The scale is per GROUP and not per ROW because an outlier only ever coarsens its
/// own 128 weights instead of the whole 6144-wide row — see the module comment above
/// for the measurement that forced the change.
pub fn quant_i4(w: &[f32], o_dim: usize, i_dim: usize) -> (Vec<u8>, Vec<f32>) {
    debug_assert_eq!(w.len(), o_dim * i_dim);
    let (rb, ng) = (i4_row_bytes(i_dim), i4_groups(i_dim));
    let mut packed = vec![0u8; o_dim * rb];
    let mut scale = vec![0.0f32; o_dim * ng];
    for ((row, prow), srow) in w
        .chunks_exact(i_dim)
        .zip(packed.chunks_exact_mut(rb))
        .zip(scale.chunks_exact_mut(ng))
    {
        quant_i4_row(row, prow, srow);
    }
    (packed, scale)
}

/// One row: per group of [`I4_GROUP`] weights, `s = max|group|/7` so the group's extreme
/// lands on nibble 15 (+7), then round-to-nearest into the packed nibbles.
fn quant_i4_row(row: &[f32], packed: &mut [u8], scale: &mut [f32]) {
    for (g, (seg, s)) in row.chunks(I4_GROUP).zip(scale.iter_mut()).enumerate() {
        let amax = seg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        *s = if amax > 0.0 { amax / 7.0 } else { 1.0 };
        for (t, &wi) in seg.iter().enumerate() {
            let q = ((wi / *s).round() as i32 + 8).clamp(0, 15) as u8;
            set_nibble(packed, g * I4_GROUP + t, q);
        }
    }
}

/// Write nibble `q` at column `i` of a packed int4 row: LOW nibble = even column. The WRITE
/// half of the packing only — the two decoders keep their own spellings on purpose, see
/// [`dequant_i4`].
#[inline]
fn set_nibble(row: &mut [u8], i: usize, q: u8) {
    let b = &mut row[i >> 1];
    *b = if i & 1 == 0 {
        (*b & 0xF0) | q
    } else {
        (*b & 0x0F) | (q << 4)
    };
}

/// Reconstruct the dense `W[o_dim, i_dim]` a `(packed, scale)` int4 pair represents —
/// the inverse of [`quant_i4`], and the SINGLE int4 reader for offline use.
///
/// Deliberately spells out the nibble convention rather than calling [`matvec_i4`]
/// with basis vectors: an audit that reconstructs weights through the very routine it
/// is auditing cannot detect a decode bug. The round trip `dequant_i4(quant_i4(w))` is
/// unit-tested below, which is what keeps the two spellings honest.
///
/// No non-test caller since `i4_audit` was retired 2026-08-05 (tag `archive/i4-audit`) —
/// "offline use" above is now only that tool, restored. It stays where the VQ equivalent
/// went, because the round trip below actually pins the second spelling to the first.
pub fn dequant_i4(packed: &[u8], scale: &[f32], o_dim: usize, i_dim: usize) -> Vec<f32> {
    let (rb, ng) = (i4_row_bytes(i_dim), i4_groups(i_dim));
    debug_assert_eq!(scale.len(), o_dim * ng);
    let mut w = vec![0f32; o_dim * i_dim];
    for o in 0..o_dim {
        let row = &packed[o * rb..(o + 1) * rb];
        let srow = &scale[o * ng..(o + 1) * ng];
        for i in 0..i_dim {
            let b = row[i >> 1];
            let n = (if i & 1 == 0 { b & 0x0F } else { b >> 4 }) as i32 - 8;
            w[o * i_dim + i] = n as f32 * srow[i / I4_GROUP];
        }
    }
    w
}

/// On-disk bytes of one int4 projection `W[o_dim, i_dim]`: `o_dim` packed rows then
/// `o_dim · i4_groups(i_dim)` f32 group scales, back-to-back (one projection = one
/// contiguous span).
pub fn i4_proj_bytes(o_dim: usize, i_dim: usize) -> usize {
    o_dim * i4_row_bytes(i_dim) + o_dim * i4_groups(i_dim) * 4
}

/// Unpadded on-disk bytes of one int4 expert (gate‖gate_scale‖up‖up_scale‖down‖
/// down_scale). Reuses [`super::vq_expert_layout`] — the (o,i) dims are format-independent.
pub fn i4_expert_bytes(expert_in: usize, moe_inter: usize) -> usize {
    expert_bytes(expert_in, moe_inter, i4_proj_bytes)
}

/// Per-expert on-disk stride: [`i4_expert_bytes`] padded up to [`super::VQ_ALIGN`], so one
/// int4 expert is a single block-aligned read (mirrors the `.vq3` stride).
pub fn i4_expert_stride(expert_in: usize, moe_inter: usize) -> usize {
    expert_stride(i4_expert_bytes(expert_in, moe_inter))
}

/// The six byte offsets within one int4 expert block, in expert-descriptor field
/// order (moe.hip's int4 interpretation of the shared `ExpertDesc` six-pointer layout).
/// Every packed span starts 4-byte aligned (rows are `i_dim/2`, i_dim a multiple of
/// 8; scale spans are whole f32), so `dot_i4_wave`'s dword fast path stays valid.
pub fn i4_slot_offsets(expert_in: usize, moe_inter: usize) -> [usize; 6] {
    slot_offsets(expert_in, moe_inter, |o, i| {
        (o * i4_row_bytes(i), o * i4_groups(i) * 4)
    })
}

/// Write projection `k`'s packed nibbles + f32 group scales into an expert block at
/// the offsets [`i4_slot_offsets`] defines. The SINGLE writer of the `.i4` slot layout
/// — `bin/fp8_to_i4` goes through it, as did the retired `vq3_to_i4`, so no two
/// producers can disagree on where a projection's bytes land (the same rule
/// `vq_slot_offsets` states for `.vq3`). `w` is exactly what [`quant_i4`] returned for this
/// projection; a short `scale` would leave the tail of the span holding the PREVIOUS
/// projection's bytes, so the lengths are checked rather than trusted.
///
/// **Takes the pair as a [`RowScaledW`] rather than as two slices, since 2026-08-15.** This
/// was the last five-argument function in the module and the one CodeScene's arg-count rule
/// (threshold 4 for Rust) still named after the wave that introduced the three view structs.
/// The view is the fix that note predicted, and it is the right one here for a second
/// reason: `quant_i4` returns the pair, so the two now travel from producer to writer as one
/// value and cannot be re-paired with another projection's scales in between.
pub fn write_i4_proj(slot: &mut [u8], off: &[usize; 6], k: usize, w: RowScaledW<'_>) {
    let RowScaledW { packed, scale } = w;
    debug_assert!(
        off[k * 2] + packed.len() <= off[k * 2 + 1],
        "packed overruns its span"
    );
    // The scale span must END where the next projection begins (and the last one
    // inside the slot): a `scale` sized for a different `I4_GROUP` would otherwise
    // run into — or leave stale bytes in — the neighbouring projection.
    let scale_end = off[k * 2 + 1] + scale.len() * 4;
    match off.get(k * 2 + 2) {
        Some(&next) => debug_assert_eq!(scale_end, next, "scale span != i4_slot_offsets"),
        None => debug_assert!(scale_end <= slot.len(), "scale span overruns the slot"),
    }
    let po = off[k * 2];
    slot[po..po + packed.len()].copy_from_slice(packed);
    write_le_scales(
        &mut slot[off[k * 2 + 1]..],
        scale.iter().map(|s| s.to_le_bytes()),
    );
}

#[cfg(test)]
mod tests {
    // Every bound below is derived from the group quantiser's own step (`amax/7`, round to
    // nearest) rather than from a previous run, so it can fail on the code. Crate-wide
    // `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Deterministic uniform `[-1, 1)` stream. One spelling of the LCG the tests below
    /// draw weights from, seeded per test so no two share a state and a failure
    /// reproduces from its seed alone.
    fn uniform(seed: u64) -> impl FnMut() -> f32 {
        let mut st = seed;
        move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((st >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
        }
    }

    /// `dequant_i4` is a SECOND spelling of the nibble convention `matvec_i4` and
    /// `quant_i4` carry (deliberately — an audit that reconstructs weights through the
    /// routine it audits cannot see a decode bug). This is what keeps the spellings
    /// honest: the round trip must land inside one quantiser step, and `matvec_i4` on
    /// the packed bytes must equal a plain dot against the reconstruction.
    ///
    /// The step bound is the assertion that matters, and it is now PER GROUP:
    /// `amax(group)/7` with round-to-nearest puts every weight within `s_g/2` of a grid
    /// point, so `max|w - ŵ| ≤ s_g/2` holds inside each group. An implementation that
    /// dropped the −8 zero point, swapped the nibble halves, or — the regression this
    /// test exists to catch — kept ONE scale for the whole row would blow past it,
    /// because the row's groups differ in magnitude by 10^g here. An error of exactly
    /// zero would mean no rounding happened at all and the test is measuring nothing.
    #[test]
    fn dequant_i4_inverts_quant_i4_within_one_step_per_group() {
        // 3 full groups + a partial 4th, so the div_ceil group count is exercised too.
        let (o_dim, i_dim) = (5usize, I4_GROUP * 3 + 16);
        let ng = i4_groups(i_dim);
        assert_eq!(ng, 4);
        let mut rnd = uniform(0x1234_5678);
        let w = decade_scaled_rows(o_dim, i_dim, &mut rnd);
        let (packed, scale) = quant_i4(&w, o_dim, i_dim);
        assert_eq!(scale.len(), o_dim * ng);
        let back = dequant_i4(&packed, &scale, o_dim, i_dim);
        assert_half_step_per_group(&w, &back, &scale, i_dim);
        // The GEMV oracle and the dense reconstruction must agree on the same bytes.
        let x: Vec<f32> = (0..i_dim).map(|_| rnd()).collect();
        let mut y = vec![0f32; o_dim];
        matvec_i4(&mut y, &x, RowScaledW::new(&packed, &scale), [o_dim, i_dim]);
        assert_gemv_matches_dense(&y, &back, &x, i_dim);
    }

    /// Rows whose group `g` is scaled by 10^g, with each group's amax pinned. A single
    /// per-row scale could not stay within half a step of the SMALL groups, so this is what
    /// separates the two formats.
    fn decade_scaled_rows(o_dim: usize, i_dim: usize, rnd: &mut impl FnMut() -> f32) -> Vec<f32> {
        let mut w: Vec<f32> = (0..o_dim * i_dim)
            .map(|n| rnd() * 10f32.powi(((n % i_dim) / I4_GROUP) as i32))
            .collect();
        for row in w.chunks_exact_mut(i_dim) {
            for g in 0..i4_groups(i_dim) {
                row[g * I4_GROUP] = 9.0 * 10f32.powi(g as i32); // the amax setter
            }
        }
        w
    }

    /// Every group's round trip lands inside half of ITS OWN step — and not exactly on
    /// zero, which would mean no rounding happened and the bound is measuring nothing.
    fn assert_half_step_per_group(w: &[f32], back: &[f32], scale: &[f32], i_dim: usize) {
        let ng = i4_groups(i_dim);
        for (o, (wr, br)) in w
            .chunks_exact(i_dim)
            .zip(back.chunks_exact(i_dim))
            .enumerate()
        {
            for (g, &s) in scale[o * ng..(o + 1) * ng].iter().enumerate() {
                let cols = g * I4_GROUP..((g + 1) * I4_GROUP).min(i_dim);
                let err = cols.map(|i| (wr[i] - br[i]).abs()).fold(0.0f32, f32::max);
                assert!(
                    err <= s * 0.5 + 1e-6,
                    "row {o} group {g}: max round-trip error {err:.6e} exceeds half a step ({:.6e})",
                    s * 0.5
                );
                assert!(
                    err > 0.0,
                    "row {o} group {g}: zero error means no rounding happened"
                );
            }
        }
    }

    /// `matvec_i4` on the packed bytes equals a plain dot against the dense reconstruction.
    fn assert_gemv_matches_dense(y: &[f32], back: &[f32], x: &[f32], i_dim: usize) {
        for (o, (&yo, br)) in y.iter().zip(back.chunks_exact(i_dim)).enumerate() {
            let want: f32 = br.iter().zip(x).map(|(&b, &xi)| b * xi).sum();
            assert!(
                (yo - want).abs() <= 1e-4 * want.abs().max(1.0),
                "row {o}: {yo} != {want}"
            );
        }
    }

    /// Group scales must beat a per-row scale on weights whose magnitude varies ALONG
    /// the row — the whole reason for the format change. Reconstruction error is
    /// measured against a per-row `amax/7` quantiser spelled out here, so the
    /// comparison does not depend on the old implementation still existing.
    ///
    /// Scored on the BULK — every column outside the row's one outlier group — and not
    /// on the whole row. Whole-row rel-L2 is dominated by the outlier group, which both
    /// quantisers represent about equally well; it comes out ~0.071 either way and
    /// hides the entire effect. The bulk is where decode quality lives and where the
    /// per-row scale rounds weights to zero, so that is what the assertion prices.
    #[test]
    fn group_scales_beat_a_per_row_scale_on_the_bulk() {
        let (o_dim, i_dim) = (4usize, I4_GROUP * 8);
        let mut rnd = uniform(0xACE1);
        // One outlier group per row, 1000× the rest — the pathology a per-row scale
        // cannot absorb (it rounds the other 7/8 of the row toward zero).
        let outlier = |o: usize| o % 8;
        let w: Vec<f32> = (0..o_dim * i_dim)
            .map(|n| {
                let big = (n % i_dim) / I4_GROUP == outlier(n / i_dim);
                rnd() * if big { 1000.0 } else { 1.0 }
            })
            .collect();
        // rel-L2 and zero-fraction over the non-outlier columns only.
        let bulk = |rec: &[f32]| -> (f64, f64) {
            let (mut n, mut d, mut z, mut c) = (0f64, 0f64, 0usize, 0usize);
            for o in 0..o_dim {
                for i in 0..i_dim {
                    if i / I4_GROUP == outlier(o) {
                        continue;
                    }
                    let (a, b) = (w[o * i_dim + i] as f64, rec[o * i_dim + i] as f64);
                    (n, d, c) = (n + (b - a) * (b - a), d + a * a, c + 1);
                    z += usize::from(b == 0.0);
                }
            }
            ((n / d).sqrt(), z as f64 / c as f64)
        };
        let (packed, scale) = quant_i4(&w, o_dim, i_dim);
        let (g_rel, g_zero) = bulk(&dequant_i4(&packed, &scale, o_dim, i_dim));
        // Per-row reference: s = max|row|/7, round-to-nearest, clamp to [-8, 7].
        let per_row: Vec<f32> = w
            .chunks_exact(i_dim)
            .flat_map(|row| {
                let s = row.iter().fold(0f32, |m, v| m.max(v.abs())) / 7.0;
                row.iter()
                    .map(move |&v| ((v / s).round() as i32).clamp(-8, 7) as f32 * s)
            })
            .collect();
        let (r_rel, r_zero) = bulk(&per_row);
        assert!(
            g_rel * 4.0 < r_rel,
            "bulk relL2: group-{I4_GROUP} {g_rel:.4} is not decisively better than per-row {r_rel:.4}"
        );
        // The mechanism, not just the score: the per-row scale rounds nearly the whole
        // bulk to zero, the group scale rounds almost none of it.
        assert!(
            r_zero > 0.9,
            "per-row bulk should be almost all zeros, got {r_zero:.3}"
        );
        assert!(
            g_zero < 0.1,
            "group bulk should barely round to zero, got {g_zero:.3}"
        );
    }

    #[test]
    fn i4_slot_offsets_are_contiguous_and_aligned() {
        let (expert_in, inter) = (6144usize, 2048usize);
        let off = i4_slot_offsets(expert_in, inter);
        assert_eq!(off[0], 0);
        for &o in &off {
            assert_eq!(o % 4, 0, "packed/scale span {o} not 4-byte aligned");
        }
        // last span (down_scale) ends exactly at i4_expert_bytes.
        assert_eq!(
            off[5] + expert_in * i4_groups(inter) * 4,
            i4_expert_bytes(expert_in, inter)
        );
    }

    #[test]
    fn i4_quant_matvec_roundtrip() {
        // quant_i4 → matvec_i4 approximates the true GEMV within int4 error.
        // i spans several groups so the group indexing is part of what is checked.
        let (o, i) = (16usize, I4_GROUP * 3);
        let mut w = vec![0.0f32; o * i];
        let mut s = 0x2468u64;
        let mut rf = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        for v in w.iter_mut() {
            *v = rf();
        }
        let x: Vec<f32> = (0..i).map(|_| rf()).collect();
        let mut want = vec![0.0f32; o];
        for oo in 0..o {
            want[oo] = (0..i).map(|ii| w[oo * i + ii] * x[ii]).sum();
        }
        let (packed, scale) = quant_i4(&w, o, i);
        let mut got = vec![0.0f32; o];
        matvec_i4(&mut got, &x, RowScaledW::new(&packed, &scale), [o, i]);
        // int4 group quant: err bounded by ~scale·Σ|x| worst case; check it tracks.
        let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let err = want
            .iter()
            .zip(&got)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            err < 0.15 * mx + 0.1,
            "i4 roundtrip err={err:.3} max={mx:.3}"
        );
    }
}
