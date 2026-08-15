//! The per-layer repack driver: write an `.f4` layer if it is absent, then optionally read
//! it back and prove it byte-for-byte against the source.
//!
//! Sits above [`super::expert`] (one expert's spans) and [`super::layer`] (the bounded,
//! atomic write), and exists because `convert_v4` and `convert_k3` differ in exactly one
//! thing — how an expert's tensor prefix is spelled — and a copied verify loop is the shape
//! where one model's converter gets a fix the other's silently does not.

use anyhow::{Context, Result, ensure};

use super::expert::{F4Expert, F4Naming};
use super::header::{ExpertHeader, F4_MAGIC, LayerDims};
use super::layer::{LAYER_WINDOW, write_expert_layer};
use super::tensors::Safetensors;
use crate::quant::{VQ_ALIGN, f4_expert_bytes, f4_expert_stride};

/// One `.f4` layer file's worth of work: write it if absent, then optionally prove it.
///
/// **Factored out when `convert_k3` became the second caller and the duplication gate refused
/// it** — 209 tokens of the verify block alone. That is the right outcome and not a formality:
/// the two converters differ only in how an expert's tensor prefix is spelled, and a copied
/// verify loop is the shape where one model's converter gets a fix the other's silently does not.
/// `docs/reference/architecture.md`'s rule holds here — everything that is not the difference
/// between the models belongs in one place.
pub struct RoutedRepack<'a> {
    /// Program name for the progress lines, so the log says which converter is running.
    pub tool: &'a str,
    pub out_dir: &'a str,
    pub src: &'a Safetensors,
    pub naming: &'static F4Naming,
    /// The width the ROUTED BLOCK is entered at — `hidden` on V4, the 3584 latent on K3. See
    /// [`crate::quant::vq_expert_layout`].
    pub expert_in: usize,
    pub moe_inter: usize,
    pub n_experts: usize,
    /// Re-read the file and compare against the source spans.
    pub verify: bool,
    /// False for a verify-only pass, which must not write.
    pub write: bool,
}

/// Where one layer's file is, how its blocks are laid out, and whether it is already on disk.
///
/// Resolved once per layer and handed to both halves, so the writer and the verifier cannot
/// disagree about the stride or the block size — the pair that decides where every expert
/// starts, and the pair a wrong answer to would be written and read back consistently.
struct LayerPlan {
    layer: usize,
    path: String,
    stride: usize,
    ebytes: usize,
    /// Both converters resume by SKIPPING an output path that already exists, without reading
    /// it — [`write_expert_layer`]'s doc carries why that is safe and what it costs.
    reused: bool,
}

impl RoutedRepack<'_> {
    /// Convert layer `l`. `base(e)` gives expert `e`'s source tensor prefix — the ONLY thing
    /// that differs between the two models, which is why it is a closure and everything else
    /// is a field.
    ///
    /// `Sync` on the closure because `fill_expert_blocks` packs the window's experts in parallel
    /// and each thread calls `base` for its own expert. Both callers pass a plain `format!`, so
    /// the bound costs them nothing. (An earlier draft of this line said omitting it would
    /// "silently" make the loop single-threaded — it would not: `write_expert_layer` requires
    /// `Sync`, so leaving it off is a compile error, which is the good outcome.)
    pub fn layer(&self, l: usize, base: impl Fn(usize) -> String + Sync) -> Result<()> {
        let plan = self.plan(l);
        // A run either produces the file or it does not touch it; there is no third mode, and
        // the read-only branch is the one `--verify-only` promised in its `--help`.
        if self.write {
            self.write_layer(&plan, &base)?;
        } else {
            self.require_converted(&plan)?;
        }
        if self.verify {
            self.verify_layer(&plan, &base)?;
        }
        Ok(())
    }

    /// Locate layer `l`'s file and its block geometry, and stat it — the one place the path
    /// spelling and the `f4_*` geometry calls live, so the two halves below share them.
    fn plan(&self, layer: usize) -> LayerPlan {
        let path = format!("{}/L{layer:02}.f4", self.out_dir);
        LayerPlan {
            layer,
            stride: f4_expert_stride(self.expert_in, self.moe_inter),
            ebytes: f4_expert_bytes(self.expert_in, self.moe_inter),
            reused: std::fs::metadata(&path).is_ok(),
            path,
        }
    }

    /// Expert `e` of this layer, located in the source checkpoint — the one construction site
    /// outside the tests, shared by the writer and the verifier so both read the same spans.
    fn expert(&self, base: &impl Fn(usize) -> String, e: usize) -> F4Expert<'_> {
        F4Expert {
            src: self.src,
            base: base(e),
            expert_in: self.expert_in,
            moe_inter: self.moe_inter,
            naming: self.naming,
        }
    }

    /// Write the layer, or report that there was nothing to write.
    ///
    /// "reusing" is said only here, where there was a write to skip. On a `--verify-only` run
    /// the file is the thing being checked, not something being reused, and saying so 92 times
    /// above 92 "verified" lines is noise a reader has to learn to ignore.
    fn write_layer(
        &self,
        plan: &LayerPlan,
        base: &(impl Fn(usize) -> String + Sync),
    ) -> Result<()> {
        let (ne, tool) = (self.n_experts, self.tool);
        if plan.reused {
            eprintln!("{tool}: {} exists, reusing", plan.path);
            return Ok(());
        }
        // One aligned block for the header, then `ne` routed blocks — and NO shared block,
        // unlike `.vq3`/`.i4`. Neither model's shared expert is FP4 (V4's is fp8 e4m3, K3's
        // is BF16 at full width), so a block written past `ne` would be the wrong
        // ARITHMETIC, not just the wrong weights. Streamed in `LAYER_WINDOW` windows: a K3
        // layer is 15.72 GB and buffering one would be host RAM the GPU shares.
        let n = write_expert_layer(
            &plan.path,
            &ExpertHeader::new(
                F4_MAGIC,
                LayerDims {
                    layer: plan.layer,
                    n_experts: ne,
                    expert_in: self.expert_in,
                    moe_inter: self.moe_inter,
                    stride: plan.stride,
                },
            )
            .to_bytes(),
            plan.stride,
            plan.ebytes,
            ne,
            LAYER_WINDOW,
            |e, slot| self.expert(base, e).pack(slot),
        )
        .with_context(|| format!("write layer {} (repack or I/O)", plan.layer))?;
        eprintln!("{tool}: wrote {} ({n} bytes)", plan.path);
        Ok(())
    }

    /// A read-only pass needs the layer to exist already.
    ///
    /// **`--verify-only` writes NOTHING, and that is a deliberate behaviour change** made when
    /// this loop was factored out of `convert_v4` (2026-08-11) and spotted by review rather
    /// than by me. The old inline form was `if !reused { write… }` with no verify-only guard,
    /// so `convert_v4 --verify-only` over a range containing an unconverted layer silently
    /// CONVERTED it — against that flag's own `--help`, which promises the run is read-only.
    /// Refusing here is what the flag says; the message has to be explicit, because the
    /// alternative is `std::fs::read` failing with a bare ENOENT and a reader concluding the
    /// artifact is damaged rather than incomplete.
    fn require_converted(&self, plan: &LayerPlan) -> Result<()> {
        if plan.reused {
            return Ok(());
        }
        ensure!(
            !self.verify,
            "{} does not exist, so there is nothing to verify — layer {} has never been \
             converted. `--verify-only` is read-only by contract and will not create it; drop \
             the flag to convert, or narrow the layer range to what this artifact holds.",
            plan.path,
            plan.layer
        );
        Ok(())
    }

    /// Read the FILE back and compare it against the mmap'd source, so this tests the writer's
    /// offsets, the block stride, the write, and whatever a previous run left behind. It runs
    /// on a REUSED layer too — that is precisely the layer whose bytes nobody has ever checked.
    /// It is deliberately NOT `back == buf`: that comparison could only ever pass, since the
    /// buffer came from `pack` and `diff` reads the same source spans, so it would be a guard
    /// unable to fire dressed as a verification.
    ///
    /// It reads ONE expert's window at a time rather than the file. This was `fs::read`,
    /// inherited from `convert_v4` where the whole layer is 3.42 GB and survivable; against
    /// K3's 896 experts the same line is a **15.72 GB allocation per layer** on a box whose
    /// 128 GB LPDDR5 is shared with the GPU — the exact figure [`write_expert_layer`]'s doc
    /// uses to justify bounding the WRITE side, promoted to the read side by sharing this loop
    /// (review 2026-08-11). The one-expert measurement in
    /// `docs/measurement/k3-reference/repack-one-expert.md` could not have caught it: at
    /// `ne = 1` the buffer is 17.5 MB. `diff` only ever looks at `ebytes` bytes at `off`.
    fn verify_layer(&self, plan: &LayerPlan, base: &impl Fn(usize) -> String) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let (ne, tool) = (self.n_experts, self.tool);
        let f =
            std::fs::File::open(&plan.path).with_context(|| format!("re-read {}", plan.path))?;
        let len = f.metadata()?.len();
        ensure!(
            len == (VQ_ALIGN + ne * plan.stride) as u64,
            "{}: {len} bytes on disk, expected {}",
            plan.path,
            VQ_ALIGN + ne * plan.stride
        );
        let mut differing = 0usize;
        let mut win = vec![0u8; plan.ebytes];
        for e in 0..ne {
            let off = VQ_ALIGN + e * plan.stride;
            f.read_exact_at(&mut win, off as u64)
                .with_context(|| format!("{}: expert {e}'s block at byte {off}", plan.path))?;
            differing += self.expert(base, e).diff(&win)?.len();
        }
        ensure!(
            differing == 0,
            "layer {}: {differing} bytes differ from the source — the repack is supposed \
             to be a COPY",
            plan.layer
        );
        eprintln!(
            "{tool}: verified L{:02}.f4 — {ne} experts, 0 bytes differ",
            plan.layer
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // One test, and its first half is the only automated exercise of `RoutedRepack::layer` at
    // all — including the per-expert windowed read that replaced a whole-file `fs::read`.
    // Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::super::expert::F4_NAMING_V4;
    use super::super::fixtures::{F4Fixture, tmpdir};
    use super::*;

    /// **A written layer verifies, and `--verify-only` over a layer nobody converted REFUSES
    /// instead of writing it.**
    ///
    /// The refusal is a deliberate behaviour change made when this loop was factored out of
    /// `convert_v4` (2026-08-11): the old inline form was `if !reused { write… }` with no
    /// verify-only guard, so a `--verify-only` run over an unconverted layer silently CONVERTED
    /// it, against that flag's own `--help`. Nothing asserted it until review asked, and a
    /// behaviour change with no test is indistinguishable from a regression later.
    ///
    /// The first half is not filler. It is the only automated exercise of [`RoutedRepack::layer`]
    /// at all, and it walks the per-expert windowed read that replaced a whole-file `fs::read` —
    /// the path whose absence of a test is why that allocation reached K3's 15.72 GB scale before
    /// anyone noticed.
    #[test]
    fn a_verify_only_pass_refuses_a_layer_that_was_never_converted() {
        let fx = F4Fixture::new("repack_layer");
        let st = fx.open();
        let out = tmpdir("f4_repack_out");
        let repack = |write: bool| RoutedRepack {
            tool: "test",
            out_dir: &out,
            src: &st,
            naming: &F4_NAMING_V4,
            expert_in: fx.expert_in,
            moe_inter: fx.moe_inter,
            n_experts: 1,
            verify: true,
            write,
        };
        // Layer 7, not 0: the path is `L{l:02}.f4`, and a zero would hide a formatting slip.
        repack(true)
            .layer(7, |_| "e".into())
            .expect("write, then verify what was written");
        // Same directory, a layer that was never written — so `reused` is false and `write` is off.
        let e = format!("{:#}", repack(false).layer(9, |_| "e".into()).unwrap_err());
        assert!(
            e.contains("L09.f4") && e.contains("never been converted"),
            "verify-only must refuse by name, got: {e}"
        );
        assert!(
            std::fs::metadata(format!("{out}/L09.f4")).is_err(),
            "verify-only WROTE the layer it was supposed to refuse"
        );
        let _ = std::fs::remove_dir_all(&out);
    }
}
