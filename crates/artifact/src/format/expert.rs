//! One routed FP4 expert as it sits in a SOURCE checkpoint: which six tensors it is made
//! of, what shape they must have, and the byte offsets they occupy in an `.f4` block.
//!
//! The per-expert half of the repack; [`super::repack`] is the per-layer driver that calls
//! it. Split by cardinality, which is also where the checks split: everything here is about
//! one expert's six spans — the shape guards that catch a transposition, the naming table
//! that lets two checkpoints ship the identical layout under different names, and the e8m0
//! NaN refusal that only a path reading every scale byte can make.

use anyhow::{Context, Result, bail, ensure};

use super::{Dtype, tensors::Safetensors};

// ── `.f4` repack (DeepSeek-V4-Flash's native FP4 routed experts) ────────────────
//
// V4 ships its routed experts ALREADY at 4 bits: `<proj>.weight` is `I8[o, i/2]` holding
// e2m1 nibble pairs, `<proj>.scale` is `F8_E8M0[o, i/32]`. rivoli's `.f4` block is the
// same nibbles and the same exponent bytes at O_DIRECT-aligned offsets — a REPACK, with
// nothing fit, nothing re-rounded, and no error introduced.
//
// Two facts make the copy a plain `memcpy` rather than a transcode. Both are read from the
// reference; neither is CHECKED here, and saying so matters because a copy cannot detect
// either being wrong:
//   * NIBBLE ORDER. torch's `float4_e2m1fn_x2` and `inference/convert.py:31-33`
//     (`low = x & 0x0F; high = (x >> 4) & 0x0F; stack([TABLE[low], TABLE[high]])`) put
//     logical element 2j in the LOW nibble of byte j — the convention `quant::matvec_i4`
//     already reads. A repack under the opposite convention is the same byte copy, so this
//     becomes checkable only when a `matvec_f4` exists to decode one. That is S3's, and it
//     is the check to add there.
//   * The SCALE GRID. `inference/kernel.py:468` declares
//     `scales_b: T.Tensor[(N, ceildiv(K, 32))]` indexed `[n, k]` — `[o_dim, f4_groups]`
//     row-major, which IS checked, by the shape guard in `F4Expert::spans`.

/// One routed expert — V4's or K3's — located in a source checkpoint: which tensors it is made
/// of and what shape they must have.
///
/// It exists to pair the six spans that `spans`, [`Self::pack`] and [`Self::diff`] all walk in
/// the same order, so the writer and the verifier cannot disagree about the layout.
///
/// One construction site outside the tests ([`super::repack::RoutedRepack::layer`]) — which is why folding it
/// into that struct was considered and declined: the split by cardinality (per-layer state there,
/// per-expert layout here) still reads correctly, and this type is `pub` with three methods. The
/// hazard the shape checks in [`Self::spans`] exist for is transposition: the three projections
/// are `(moe_inter, expert_in)·2 + (expert_in, moe_inter)`, and a swap keeps the byte count.
pub struct F4Expert<'a> {
    pub src: &'a Safetensors,
    /// `layers.{l}.ffn.experts.{e}` — see `quant::v4_expert_base`. On K3,
    /// `quant::k3_expert_base`, which is a longer path under `language_model.model.`.
    pub base: String,
    pub expert_in: usize,
    pub moe_inter: usize,
    /// How THIS checkpoint spells the two tensors of a projection, and what dtypes they carry.
    /// The `.f4` container is identical either way — only the source names differ.
    pub naming: &'static F4Naming,
}

/// How a checkpoint spells one FP4 projection's two tensors, and what dtypes they declare.
///
/// **Two checkpoints ship the identical MXFP4 layout under different names and different dtype
/// strings**, verified against both files: V4 writes `<proj>.weight` as `I8` with `<proj>.scale`
/// as `F8_E8M0`, and K3 writes compressed-tensors' `<proj>.weight_packed` / `<proj>.weight_scale`
/// with **both as `U8`**. The bytes are the same thing — e2m1 nibble pairs and bare e8m0
/// exponents — so this is a naming table, not a format abstraction.
///
/// It is a struct rather than four more `F4Expert` fields because the four values are only
/// correct as a SET: pairing K3's `weight_packed` with V4's `F8_E8M0` scale dtype would refuse
/// the checkpoint, and pairing V4's `.scale` name with K3's `U8` would read the wrong tensor if
/// one ever existed under both names.
pub struct F4Naming {
    pub projs: [&'static str; 3],
    pub packed: &'static str,
    pub scale: &'static str,
    pub packed_dtype: Dtype,
    pub scale_dtype: Dtype,
}

/// DeepSeek-V4-Flash: `w1.weight` (`I8`) + `w1.scale` (`F8_E8M0`).
pub const F4_NAMING_V4: F4Naming = F4Naming {
    projs: crate::quant::V4_PROJ,
    packed: "weight",
    scale: "scale",
    packed_dtype: Dtype::I8,
    scale_dtype: Dtype::F8E8M0,
};

/// Kimi-K3: `w1.weight_packed` + `w1.weight_scale`, **both `U8`**.
///
/// The dtype is the part worth naming. `compressed-tensors` writes the scale grid as plain `U8`
/// rather than declaring an e8m0 type, so the checkpoint's own metadata does not say these bytes
/// are exponents — `quantization_config.config_groups.group_0.weights.scale_dtype` does, and it
/// says `torch.uint8`. That is why [`F4Expert::spans`] checks the SHAPE against `f4_groups`: for
/// K3 the shape is the only evidence in the file that a byte per 32 weights is a scale.
pub const F4_NAMING_K3: F4Naming = F4Naming {
    projs: crate::quant::K3_PROJ,
    packed: crate::quant::K3_PACKED,
    scale: crate::quant::K3_SCALE,
    packed_dtype: Dtype::U8,
    scale_dtype: Dtype::U8,
};

impl F4Expert<'_> {
    /// This expert's six source spans, paired with their byte offset inside an `.f4`
    /// block: `(w1, w1.scale, w3, w3.scale, w2, w2.scale)` — see
    /// [`crate::quant::V4_PROJ`] for why that is gate/up/down order.
    ///
    /// ONE definition of the layout, used by both [`Self::pack`] and [`Self::diff`]. That
    /// is deliberate: what the verifier must prove is that the VALUES pass through
    /// untouched, and re-deriving the offsets in both places would only test that a copy
    /// of the arithmetic agrees with itself. The layout is pinned separately, by
    /// `f4_expert_bytes` (size) and by [`super::header::ExpertHeader`] (dims), and a wrong layout changes
    /// the file's length.
    fn spans(&self) -> Result<Vec<(usize, &[u8])>> {
        let (expert_in, moe_inter, base) = (self.expert_in, self.moe_inter, &self.base);
        // The offsets come from `f4_slot_offsets` — the SAME function the streaming pool
        // points its `ExpertDescF4` at (`memory::routed::TierFmt`). This used to walk the
        // spans and accumulate `off` itself, which made the writer and the reader two
        // implementations of one layout: a shifted scale span would have been written and
        // read consistently by rivoli and disagreed with nothing until the kernel decoded a
        // projection against another one's exponents. Now a change to the layout moves both
        // ends or neither.
        let off = crate::quant::f4_slot_offsets(expert_in, moe_inter);
        let nm = self.naming;
        let mut out = Vec::with_capacity(6);
        for (p, (proj, (o_dim, i_dim))) in nm
            .projs
            .into_iter()
            .zip(crate::quant::vq_expert_layout(expert_in, moe_inter))
            .enumerate()
        {
            // Shapes are checked, not trusted: `[o, i/2]` and `[o, i/32]` are the only pair
            // the byte counts below are correct for, and a transposed or mis-blocked source
            // would otherwise copy the right NUMBER of bytes in the wrong order — a file
            // that passes every length check and decodes to noise.
            let (w, wsh) = self
                .src
                .typed(&format!("{base}.{proj}.{}", nm.packed), nm.packed_dtype)?;
            ensure!(
                wsh == [o_dim, i_dim / 2],
                "{base}.{proj}.{}: shape {wsh:?} != [{o_dim},{}] (FP4 nibble pairs)",
                nm.packed,
                i_dim / 2
            );
            let (sc, ssh) = self
                .src
                .typed(&format!("{base}.{proj}.{}", nm.scale), nm.scale_dtype)?;
            let groups = crate::quant::f4_groups(i_dim);
            // **On K3 this shape check is the only evidence in the file that these bytes are
            // scales at all** — `compressed-tensors` declares them plain `U8`, not an e8m0 type.
            // See `F4_NAMING_K3`.
            ensure!(
                ssh == [o_dim, groups],
                "{base}.{proj}.{}: shape {ssh:?} != [{o_dim},{groups}] (one e8m0 per {} \
                 weights along the input dim)",
                nm.scale,
                crate::quant::F4_GROUP
            );
            let wb = o_dim * crate::quant::f4_row_bytes(i_dim);
            ensure!(
                w.len() == wb && sc.len() == o_dim * groups,
                "{base}.{proj}: source spans are shorter than their shapes"
            );
            // **The e8m0 NaN check, and this is the one place in the engine that can make
            // it.** `0xff` is the format's NaN. The kernel decodes it correctly
            // (`common.hpp::e8m0f` returns a quiet NaN rather than `2^128`) but cannot
            // REFUSE it, and `moe_fixed`'s saturating clamp then launders the NaN into a
            // finite ±2^14 — so one bad byte is 32 weights of plausible garbage with no
            // error anywhere. `docs/investigations/v4-flash-port.md` §S3 requirement 10.
            //
            // It runs HERE, at repack, because this is the only path that reads every
            // routed scale byte: at decode the bytes DMA from NVMe straight into the pool
            // slot and the host never sees them. Measured on the shipped 43-layer set
            // (9,261,023,232 scale bytes): 9 distinct codes, all in `0x76..=0x7e`
            // (2^-9..2^-1), zero `0x00` and zero `0xff` — so this guard is green on every
            // artifact that exists, and the only thing that has made it speak is the
            // injection in this file's own
            // `an_e8m0_nan_scale_byte_is_refused_at_repack_and_a_subnormal_one_is_not`.
            //
            // `0x00` is deliberately NOT refused. It is `2^-127`, a legal encoding that
            // `e8m0f` and `quant::e8m0` both decode exactly (f32 carries it as a
            // subnormal); refusing it would be inventing a rule the format does not have.
            // The reason it is worth a sentence is that `b << 23` WOULD hand back +0, which
            // is why both decoders special-case it.
            //
            // `nm.scale`, not a literal `.scale`: this message's whole value is that it names
            // the exact tensor, row and group that fails, and on K3 the tensor is
            // `…w1.weight_scale`. It WAS a literal, so a K3 refusal would have named a tensor no
            // K3 checkpoint contains — caught by review 2026-08-11, and only visible because the
            // fixture below now runs both naming tables.
            if let Some(k) = sc.iter().position(|&b| b == 0xff) {
                bail!(
                    "{base}.{proj}.{}[{}][{}] is 0xff — the e8m0 NaN. The FP4 kernels \
                     cannot reject it and `moe_fixed`'s clamp turns it into a finite \
                     ±2^14, so a whole {}-weight group would decode to plausible garbage.",
                    nm.scale,
                    k / groups,
                    k % groups,
                    crate::quant::F4_GROUP,
                );
            }
            out.push((off[p * 2], w));
            out.push((off[p * 2 + 1], sc));
        }
        // No `off[5] + …== f4_expert_bytes(…)` assertion here: `slot_offsets` derives both
        // from the same per-projection byte counts, so it can never fire. The check that
        // CAN is `pack`'s, on the buffer it was handed.
        Ok(out)
    }

    /// Repack into `dst` (`f4_expert_bytes` long). A byte copy — nothing is fit, nothing
    /// is re-rounded, and no error is introduced.
    pub fn pack(&self, dst: &mut [u8]) -> Result<()> {
        let want = crate::quant::f4_expert_bytes(self.expert_in, self.moe_inter);
        ensure!(
            dst.len() == want,
            "{}: destination is {} bytes, an expert block is {want}",
            self.base,
            dst.len()
        );
        for (off, bytes) in self.spans()? {
            dst[off..off + bytes.len()].copy_from_slice(bytes);
        }
        Ok(())
    }

    /// Byte offsets within `block` that disagree with the source tensors. **Empty means
    /// the repack was bit-exact**; anything else names where it was not.
    ///
    /// Returns the offsets rather than a bool so a caller can say WHICH bytes moved —
    /// which is what lets `convert_v4 --verify` name where a 3.4 GB layer file went wrong.
    /// Note that `diff` shares `spans()` with [`Self::pack`], so against a block derived
    /// from `pack` it is a tautology; its value is against a block read back from DISK,
    /// which is the only way `--verify` uses it.
    ///
    /// **Sharing `spans()` also means `diff` inherits its e8m0 `0xff` refusal, so on a source
    /// carrying one it returns `Err` instead of a byte list.** Deliberate: a source with an
    /// e8m0 NaN is unusable whether or not the bytes round-tripped, and "your source has a
    /// NaN scale at w3[7][2]" is the more actionable of the two reports. `convert_v4
    /// --verify` propagates it and never reaches its "N bytes differ" summary. Nothing
    /// observable changes on the shipped artifact — measured zero `0xff` across all
    /// 9,261,023,232 of its scale bytes.
    pub fn diff(&self, block: &[u8]) -> Result<Vec<usize>> {
        let mut bad = Vec::new();
        for (off, want) in self.spans()? {
            let got = block
                .get(off..off + want.len())
                .with_context(|| format!("{}: block shorter than the source spans", self.base))?;
            if got != want {
                bad.extend(
                    (0..want.len())
                        .filter(|&k| got[k] != want[k])
                        .map(|k| off + k),
                );
            }
        }
        Ok(bad)
    }
}

#[cfg(test)]
mod tests {
    // Both tests build the expected block from the tensor NAMES spelled out literally rather
    // than from `spans`, because `pack` and `diff` share it and comparing them is `A == A`.
    // Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::super::fixtures::F4Fixture;
    use super::*;

    /// **An e8m0 `0xff` scale byte is refused at repack, and `0x00` is not.**
    ///
    /// `0xff` is the format's NaN. `common.hpp::e8m0f` decodes it correctly — to a quiet
    /// NaN, which is the right answer — and then `moe_fixed`'s saturating clamp launders it
    /// into a finite ±2^14, so a single bad byte becomes 32 weights of plausible garbage
    /// with no error anywhere downstream. The kernels cannot refuse it; this is the only
    /// path in the engine that reads every routed scale byte, because at decode they DMA
    /// from NVMe straight into the pool slot and the host never sees them.
    /// `docs/investigations/v4-flash-port.md` §S3 requirement 10.
    ///
    /// **Both directions, and the second is the one that needed measuring.** The guard must
    /// fire on `0xff` in any of the three projections — proved by injecting one, per
    /// projection, and requiring the message to name it. And it must leave everything else
    /// bit-identical: the requirement was handed over as "reject `0x00`/`0xff`", and `0x00`
    /// is a LEGAL encoding (`2^-127`, which f32 carries exactly as a subnormal and which
    /// both `quant::e8m0` and `e8m0f` special-case for that reason). Refusing it would be
    /// inventing a rule the format does not have, so a `0x00` fixture must pack unchanged —
    /// and it must pack to the SAME bytes the clean control does apart from that one, which
    /// is what stops this from being a test that any accept-everything packer passes.
    ///
    /// Measured before writing either half: over the shipped 43-layer set, 9,261,023,232
    /// scale bytes, 9 distinct codes, all in `0x76..=0x7e`, **zero `0x00` and zero `0xff`**.
    /// So this guard is green on every artifact that exists and the injection above is the
    /// only thing that has ever made it speak.
    #[test]
    fn an_e8m0_nan_scale_byte_is_refused_at_repack_and_a_subnormal_one_is_not() {
        use crate::quant::f4_expert_bytes;
        let clean = F4Fixture::new("e8m0_ok");
        let st = clean.open();
        let n = f4_expert_bytes(clean.expert_in, clean.moe_inter);
        let mut base = vec![0u8; n];
        clean
            .expert(&st)
            .pack(&mut base)
            .expect("a fixture with no 0xff must pack");

        // One `0xff` per projection, at a byte that is not the first — a guard that only
        // looked at scale[0] would pass a first-byte-only test. And under BOTH naming tables,
        // because the value of this refusal is that it names the tensor the checkpoint AT HAND
        // contains: with the literal `.scale` it once had, the V4 half stays green and the K3
        // half fails on the name. That asymmetry is the bug, so the loop has to be the outer one.
        for nm in [&F4_NAMING_V4, &F4_NAMING_K3] {
            for slot in 0..3 {
                let tag = format!("e8m0_nan_{}_{slot}", nm.scale);
                let fx = F4Fixture::named(&tag, Some((slot, 5, 0xff)), nm);
                let st = fx.open();
                let e = format!(
                    "{:#}",
                    fx.expert(&st)
                        .pack(&mut vec![0u8; n])
                        .err()
                        .unwrap_or_else(|| panic!("{tag}: a 0xff scale byte must be refused"))
                );
                assert!(
                    e.contains(&format!("{}.{}[", nm.projs[slot], nm.scale)) && e.contains("0xff"),
                    "{tag}: the refusal must name the projection and the byte, got: {e}"
                );
            }
        }

        // `0x00` passes, and changes exactly the byte it was written into. Two assertions,
        // because "it packed" alone would also hold for a packer that dropped the scales.
        let fx = F4Fixture::with_scale_byte("e8m0_zero", Some((1, 5, 0x00)));
        let st = fx.open();
        let mut got = vec![0u8; n];
        fx.expert(&st)
            .pack(&mut got)
            .expect("0x00 is 2^-127, a legal e8m0 encoding — it must NOT be refused");
        let diff: Vec<usize> = (0..n).filter(|&k| got[k] != base[k]).collect();
        let off = crate::quant::f4_slot_offsets(fx.expert_in, fx.moe_inter);
        assert_eq!(
            diff,
            vec![off[3] + 5],
            "a 0x00 in w3's scales must move exactly that byte of the block"
        );
    }

    /// **A packed `.f4` block is the six source tensors concatenated, in this order.**
    ///
    /// The expected block is built here from the tensor NAMES spelled out literally, not
    /// from `F4Expert::spans` — that independence is the whole point. `pack` and `diff`
    /// share `spans`, so `diff` can only ever report `block != pack's output`; asking it
    /// about a block derived from `pack` is `A == A` and cannot fail. Comparing against a
    /// literal order and a literal concatenation CAN fail, and does: verified by mutation
    /// (2026-08-05) against a packer that swapped nibbles, a `V4_PROJ` with w1/w3
    /// transposed, and `F4_GROUP` changed 32 → 64.
    ///
    /// What this does NOT establish is that the order is the RIGHT one — `w1` really being
    /// gate and `w3` really being up is pinned separately, against the reference source, by
    /// `quant::tests::v4_proj_order_matches_the_reference_expert_forward`. And nibble
    /// ORDER within a byte is unchecked here by construction: a repack with the opposite
    /// convention is the same byte copy. That becomes checkable only when a `matvec_f4`
    /// exists to decode one, which is S3.
    #[test]
    fn f4_pack_concatenates_the_source_tensors_in_w1_w3_w2_order() {
        use crate::quant::f4_expert_bytes;
        let fx = F4Fixture::new("pack");
        let st = fx.open();

        let mut want = Vec::new();
        for name in [
            "e.w1.weight",
            "e.w1.scale",
            "e.w3.weight",
            "e.w3.scale",
            "e.w2.weight",
            "e.w2.scale",
        ] {
            let dt = if name.ends_with(".scale") {
                Dtype::F8E8M0
            } else {
                Dtype::I8
            };
            want.extend_from_slice(st.typed(name, dt).unwrap().0);
        }

        let mut got = vec![0u8; f4_expert_bytes(fx.expert_in, fx.moe_inter)];
        fx.expert(&st).pack(&mut got).unwrap();
        assert_eq!(
            got.len(),
            want.len(),
            "block size disagrees with the source spans"
        );
        assert_eq!(got, want, "the repack is not a straight concatenation");

        // `diff` agrees with the same independently-built block, and reports the exact
        // offset of a single changed byte — which is what makes `convert_v4 --verify`
        // able to name where a 3.4 GB layer file went wrong.
        let e = fx.expert(&st);
        assert_eq!(e.diff(&want).unwrap(), Vec::<usize>::new());
        for k in [0, want.len() / 3, want.len() - 1] {
            let mut bad = want.clone();
            bad[k] ^= 0xFF;
            assert_eq!(e.diff(&bad).unwrap(), vec![k], "diff missed a flip at {k}");
        }
    }
}
