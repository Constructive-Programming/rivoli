//! The per-layer repack driver: write an `.f4` layer if it is absent, then optionally read
//! it back and prove it byte-for-byte against the source — and `LayerRepack`, the resume /
//! verify / report policy BOTH routed formats share, which is why this file is the home of the
//! windowed verify read rather than of one model's converter.
//!
//! Sits above [`super::expert`] (one expert's spans) and [`super::layer`] (the bounded,
//! atomic write), and exists because `convert_v4` and `convert_k3` differ in exactly one
//! thing — how an expert's tensor prefix is spelled — and a copied verify loop is the shape
//! where one model's converter gets a fix the other's silently does not. `.mxfp8` is the third
//! caller of that lesson, and the first one whose FORMAT differs as well as its names.

use anyhow::{Context, Result, ensure};

use super::expert::{F4Expert, F4Naming};
use super::header::{EXPERT_HEADER_BYTES, ExpertHeader, F4_MAGIC, LayerDims};
use super::layer::{ExpertFile, LAYER_WINDOW, write_expert_layer};
use super::tensors::Safetensors;
use crate::quant::{ExpertBlocks, VQ_ALIGN, f4_expert_bytes, f4_expert_stride};

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

/// The blocks of one layer file as a VERIFIER walks them: where block 0 starts, how far apart
/// blocks are, how many payload bytes each carries, and how many there are.
///
/// Bundled rather than passed as four `usize`s beside a path, for the reason
/// [`crate::quant::ExpertBlocks`] gives about `stride`/`bytes`/`count`: a swap here is a
/// correct-length comparison against the wrong pair of numbers, which is exactly the failure a
/// verifier exists to catch and would instead commit.
#[derive(Clone, Copy)]
pub(super) struct LayerBlocks<'a> {
    pub(super) path: &'a str,
    /// Where block 0 starts — [`crate::quant::VQ_ALIGN`] for every header-carrying format.
    pub(super) hbytes: usize,
    pub(super) stride: usize,
    /// Useful bytes at the start of a block; the padding to `stride` is never compared.
    pub(super) bytes: usize,
    pub(super) count: usize,
}

impl<'a> LayerBlocks<'a> {
    /// One layer file's block geometry, from the `(stride, useful bytes, block count)` a
    /// format's own plan resolved.
    ///
    /// `hbytes` is [`VQ_ALIGN`] here rather than a parameter because the two formats that
    /// reach this — `.f4` and `.mxfp8` — both carry a 40-byte header in an aligned first
    /// block ([`RoutedFmt::hbytes`](super::header::RoutedFmt::hbytes) derives it from the
    /// magic, and both have one).
    pub(super) fn new(path: &'a str, geom: (usize, usize, usize)) -> Self {
        let (stride, bytes, count) = geom;
        Self {
            path,
            hbytes: VQ_ALIGN,
            stride,
            bytes,
            count,
        }
    }
}

/// Re-read `blocks` from disk, ONE BLOCK AT A TIME, and return how many bytes `compare`
/// reported as differing across all of them.
///
/// Bounded by one block, never by the file, and that is the point: the first version of this
/// loop in `convert_v4` read a whole 3.42 GB layer, which was survivable, and the same line
/// against K3's 896 experts is a 15.72 GB allocation per layer on a box whose LPDDR5 is shared
/// with the GPU. Small's `.mxfp8` layer is 12.61 GB for the same reason. The length check
/// comes first so a short or long file is named as such rather than read past its end.
///
/// `compare` gets the block INDEX as well as its bytes, because a message that names expert 91
/// is the difference between a five-minute fix and a bisect.
fn differing_bytes(
    blocks: LayerBlocks<'_>,
    mut compare: impl FnMut(usize, &[u8]) -> Result<usize>,
) -> Result<usize> {
    let LayerBlocks {
        path,
        hbytes,
        stride,
        bytes,
        count,
    } = blocks;
    let f = std::fs::File::open(path).with_context(|| format!("re-read {path}"))?;
    let len = f.metadata()?.len();
    ensure!(
        len == (hbytes + count * stride) as u64,
        "{path}: {len} bytes on disk, expected {}",
        hbytes + count * stride
    );
    let mut win = vec![0u8; bytes];
    let mut differing = 0usize;
    for e in 0..count {
        let off = hbytes + e * stride;
        std::os::unix::fs::FileExt::read_exact_at(&f, &mut win, off as u64)
            .with_context(|| format!("{path}: block {e} at byte {off}"))?;
        differing += compare(e, &win)?;
    }
    Ok(differing)
}

/// One routed layer file's identity, geometry and the status of the run that asked for it.
///
/// The bundle both format writers hand to [`Self::run`]: paths and geometry belong to each
/// format (`.f4` and `.mxfp8` differ in stride, in magic and in how an expert is packed), but
/// the POLICIES — resume by reusing an existing file, never write on a verify-only pass, judge
/// every verify by the same threshold and say it in the same words — belong to the container.
/// Two hand-written copies of those is what this struct exists to prevent, and the duplication
/// gate found the second copy the day `.mxfp8` landed.
pub(super) struct LayerRepack<'a> {
    tool: &'a str,
    ext: &'a str,
    layer: usize,
    reused: bool,
    write: bool,
    verify: bool,
    blocks: LayerBlocks<'a>,
}

impl<'a> LayerRepack<'a> {
    /// `ident` is `(tool, extension, layer)` and `flags` is `(write, verify, reused)` — tuples
    /// rather than six positionals, and rather than a seven-field literal at each call site,
    /// which is the shape the gate refused.
    pub(super) fn new(
        ident: (&'a str, &'a str, usize),
        flags: (bool, bool, bool),
        blocks: LayerBlocks<'a>,
    ) -> Self {
        let (tool, ext, layer) = ident;
        let (write, verify, reused) = flags;
        Self {
            tool,
            ext,
            layer,
            reused,
            write,
            verify,
            blocks,
        }
    }

    /// The whole of a layer's work: produce the file when asked, refuse to verify what was
    /// never produced, then prove whatever is on disk.
    ///
    /// **`--verify-only` writes NOTHING, and that is a deliberate behaviour change** made when
    /// this loop was factored out of `convert_v4` (2026-08-11) and spotted by review rather
    /// than by me. The old inline form was `if !reused { write… }` with no verify-only guard,
    /// so `convert_v4 --verify-only` over a range containing an unconverted layer silently
    /// CONVERTED it — against that flag's own `--help`, which promises the run is read-only.
    /// Refusing here is what the flag says; the message has to be explicit, because the
    /// alternative is `std::fs::read` failing with a bare ENOENT and a reader concluding the
    /// artifact is damaged rather than incomplete.
    pub(super) fn run(
        &self,
        produce: impl FnOnce() -> Result<()>,
        prove: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if self.write {
            produce()?;
        } else {
            ensure!(
                self.reused,
                "{} does not exist, so there is nothing to verify — layer {} has never been \
                 converted. `--verify-only` is read-only by contract and will not create it; \
                 drop the flag to convert, or narrow the layer range to what this artifact \
                 holds.",
                self.blocks.path,
                self.layer
            );
        }
        if self.verify {
            prove()?;
        }
        Ok(())
    }

    /// Say "reusing" and report whether the write was skipped. Said only where there WAS a
    /// write to skip: on a `--verify-only` run the file is the thing being checked, not
    /// something being reused, and saying so 92 times above 92 "verified" lines is noise a
    /// reader has to learn to ignore.
    pub(super) fn skip_reused(&self) -> bool {
        if !self.reused {
            return false;
        }
        eprintln!("{}: {} exists, reusing", self.tool, self.blocks.path);
        true
    }

    /// The 40-byte header block 0 carries. `dims` is `(expert_in, moe_inter)` — the two widths
    /// the caller's config holds and this geometry does not.
    pub(super) fn header(&self, magic: [u8; 4], dims: (usize, usize)) -> [u8; EXPERT_HEADER_BYTES] {
        ExpertHeader::new(
            magic,
            LayerDims {
                layer: self.layer,
                n_experts: self.blocks.count,
                expert_in: dims.0,
                moe_inter: dims.1,
                stride: self.blocks.stride,
            },
        )
        .to_bytes()
    }

    /// Write the file: one aligned header block, then `count` blocks `stride` apart, streamed
    /// in [`LAYER_WINDOW`] windows.
    pub(super) fn write(
        &self,
        header: [u8; EXPERT_HEADER_BYTES],
        fill: impl Fn(usize, &mut [u8]) -> Result<()> + Sync,
    ) -> Result<()> {
        let b = self.blocks;
        let file = ExpertFile {
            path: b.path,
            header: &header,
        };
        let blocks = ExpertBlocks {
            stride: b.stride,
            bytes: b.bytes,
            count: b.count,
        };
        let n = write_expert_layer(file, blocks, LAYER_WINDOW, fill)
            .with_context(|| format!("write L{:02}.{} (repack or I/O)", self.layer, self.ext))?;
        eprintln!("{}: wrote {} ({n} bytes)", self.tool, b.path);
        Ok(())
    }

    /// How many of this file's bytes disagree with a source read by `compare`, ONE block at a
    /// time — a routed layer is up to 15.7 GB and this box's LPDDR5 is shared with the GPU.
    pub(super) fn differing(
        &self,
        compare: impl FnMut(usize, &[u8]) -> Result<usize>,
    ) -> Result<usize> {
        differing_bytes(self.blocks, compare)
    }

    /// Judge and announce a verify pass. ONE wording and one threshold for every format: the
    /// repack is a COPY, so any differing byte is a defect and not a tolerance.
    ///
    /// A verify runs on a REUSED layer too — that is precisely the layer whose bytes nobody has
    /// ever checked.
    pub(super) fn report(&self, differing: usize) -> Result<()> {
        ensure!(
            differing == 0,
            "L{:02}.{}: {differing} bytes differ from the source — the repack is supposed to \
             be a COPY",
            self.layer,
            self.ext
        );
        eprintln!(
            "{}: verified L{:02}.{} — {} experts, 0 bytes differ",
            self.tool, self.layer, self.ext, self.blocks.count
        );
        Ok(())
    }
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
        let file = self.file(&plan);
        // The policy (write, or refuse; then verify) is [`LayerRepack::run`]'s, shared with
        // the `.mxfp8` writer: a run either produces the file or it does not touch it, and the
        // read-only branch is the one `--verify-only` promised in its `--help`.
        file.run(
            || self.write_layer(&file, &base),
            || self.verify_layer(&file, &base),
        )
    }

    /// This layer's file identity: what [`Self::plan`] resolved, plus the format name the log
    /// lines and the header magic are read from.
    fn file<'b>(&'b self, plan: &'b LayerPlan) -> LayerRepack<'b> {
        let blocks = LayerBlocks::new(&plan.path, (plan.stride, plan.ebytes, self.n_experts));
        LayerRepack::new(
            (self.tool, "f4", plan.layer),
            (self.write, self.verify, plan.reused),
            blocks,
        )
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
    /// Everything after the format's own two facts — the magic and how one expert is packed —
    /// is [`LayerRepack`]'s: the header, the windowed write and the "reusing" line are the
    /// same for `.mxfp8` and must not drift apart.
    fn write_layer(
        &self,
        file: &LayerRepack<'_>,
        base: &(impl Fn(usize) -> String + Sync),
    ) -> Result<()> {
        if file.skip_reused() {
            return Ok(());
        }
        // One aligned block for the header, then `n_experts` routed blocks — and NO shared
        // block, unlike `.vq3`/`.i4`. Neither model's shared expert is FP4 (V4's is fp8 e4m3,
        // K3's is BF16 at full width), so a block written past `n_experts` would be the wrong
        // ARITHMETIC, not just the wrong weights. Streamed in `LAYER_WINDOW` windows: a K3
        // layer is 15.72 GB and buffering one would be host RAM the GPU shares.
        let header = file.header(F4_MAGIC, (self.expert_in, self.moe_inter));
        file.write(header, |e, slot| self.expert(base, e).pack(slot))
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
    /// `ne = 1` the buffer is 17.5 MB. `diff` only ever looks at the block's usable bytes.
    ///
    /// The walk, the threshold and the closing line are [`LayerRepack`]'s; only the
    /// comparison — `F4Expert::diff`'s source spans — is this format's.
    fn verify_layer(&self, file: &LayerRepack<'_>, base: &impl Fn(usize) -> String) -> Result<()> {
        let differing = file.differing(|e, win| Ok(self.expert(base, e).diff(win)?.len()))?;
        file.report(differing)
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
