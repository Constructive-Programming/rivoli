//! Test fixtures more than one submodule needs, so the 2026-08-15 split did not turn one
//! helper into six copies — which `jscpd` would refuse anyway, and rightly: a scratch
//! directory that forgets its remove-then-create in ONE copy is a test silently inheriting
//! the last run's files.
//!
//! `pub(super)` rather than `pub`: reachable from every `format::*` test module and from
//! nothing else. Both items moved here verbatim from the single file's `mod tests`; the
//! visibility is the only edit.
#![allow(clippy::unwrap_used, clippy::expect_used)] // fixtures: a panic here IS the report

use super::expert::{F4_NAMING_V4, F4Expert, F4Naming};
use super::tensors::{SafeWriter, Safetensors};

/// A fresh, empty scratch directory for one test. Every test here needs one and each
/// had spelled out the same four lines; a shared helper also guarantees the
/// remove-then-create, which a test that only created would inherit stale files from.
pub(super) fn tmpdir(tag: &str) -> String {
    let dir = std::env::temp_dir()
        .join(format!("rivoli_{tag}"))
        .to_string_lossy()
        .to_string();
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A tiny V4-shaped FP4 expert on disk. Dims are the smallest multiples of `F4_GROUP`
/// that give a projection more than one group, so a group-index error has somewhere to
/// show; `w1` and `w3` are given different content so a slot swap is not invisible.
pub(super) struct F4Fixture {
    pub(super) dir: String,
    pub(super) expert_in: usize,
    pub(super) moe_inter: usize,
    pub(super) naming: &'static F4Naming,
}

impl F4Fixture {
    pub(super) fn new(tag: &str) -> Self {
        Self::with_scale_byte(tag, None)
    }

    /// `poison = Some((slot, k, b))` overwrites projection `slot`'s scale byte `k` with
    /// `b` — the one-field perturbation the e8m0 cases below need, so the control and
    /// the break differ in exactly one byte and nothing else.
    pub(super) fn with_scale_byte(tag: &str, poison: Option<(usize, usize, u8)>) -> Self {
        Self::named(tag, poison, &F4_NAMING_V4)
    }

    /// **Parameterised over the naming table, because that is the only way `F4_NAMING_K3`
    /// is exercised through [`F4Expert::spans`] at all.** Added 2026-08-11 after review
    /// observed that every fixture here was V4-named, so K3's four names and two dtypes —
    /// the set `F4_NAMING_K3`'s doc says is "only correct as a SET" — would first be tested
    /// against the 1.42 TiB checkpoint. It immediately paid: the e8m0 refusal was naming a
    /// literal `.scale`, a tensor no K3 shard contains.
    pub(super) fn named(
        tag: &str,
        poison: Option<(usize, usize, u8)>,
        nm: &'static F4Naming,
    ) -> Self {
        use crate::quant::{F4_GROUP, f4_groups, f4_row_bytes, vq_expert_layout};
        let dir = tmpdir(&format!("f4_{tag}"));
        let (expert_in, moe_inter) = (64, 32);
        let mut w = SafeWriter::new();
        for (slot, (proj, (o_dim, i_dim))) in nm
            .projs
            .into_iter()
            .zip(vq_expert_layout(expert_in, moe_inter))
            .enumerate()
        {
            // `tag` is '1' | '3' | '2', which keeps the three projections distinct —
            // in particular w1 != w3, which have identical shapes.
            let t = usize::from(proj.as_bytes()[1]);
            let weight: Vec<u8> = (0..o_dim * f4_row_bytes(i_dim))
                .map(|k| ((k * 7 + t) % 251) as u8)
                .collect();
            // 100..=149 — inside the band the SHIPPED artifact actually uses
            // (measured 2026-08-05 over all 9,261,023,232 of its scale bytes: 9 distinct
            // codes, 0x76..=0x7e), and in particular never 0xff. So the clean fixture
            // exercises the accept path rather than the reject one.
            let mut scale: Vec<u8> = (0..o_dim * f4_groups(i_dim))
                .map(|k| (100 + (k + t) % 50) as u8)
                .collect();
            if let Some((s, k, b)) = poison
                && s == slot
            {
                scale[k] = b;
            }
            w.add(
                format!("e.{proj}.{}", nm.packed),
                nm.packed_dtype,
                vec![o_dim, i_dim / 2],
                weight,
            );
            w.add(
                format!("e.{proj}.{}", nm.scale),
                nm.scale_dtype,
                vec![o_dim, i_dim / F4_GROUP],
                scale,
            );
        }
        w.write(&format!("{dir}/e.safetensors")).unwrap();
        Self {
            dir,
            expert_in,
            moe_inter,
            naming: nm,
        }
    }

    pub(super) fn open(&self) -> Safetensors {
        Safetensors::open_file(&format!("{}/e.safetensors", self.dir)).unwrap()
    }

    pub(super) fn expert<'a>(&self, src: &'a Safetensors) -> F4Expert<'a> {
        F4Expert {
            src,
            base: "e".into(),
            expert_in: self.expert_in,
            moe_inter: self.moe_inter,
            // Whichever table WROTE the fixture — reading it back under the other one is a
            // different test (and would fail on the tensor name, not the dtype).
            naming: self.naming,
        }
    }
}

impl Drop for F4Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
