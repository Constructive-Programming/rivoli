//! The per-layer **MXFP8** repack: Inkling-2-Small's four interleaved expert tensors become
//! one `L{ll}.mxfp8` file, which is then optionally proved byte-for-byte against the source.
//!
//! Inkling-2-Small ships its routed experts ALREADY at 8 bits — `w13_weight` is
//! `F8_E4M3[E, 2·inter, hidden]` with a `U8[E, 2·inter, hidden/32]` ue8m0 scale grid, and
//! `w2_weight`/`w2_weight_scale` the same pair at `[E, hidden, inter]` — so producing a
//! `.mxfp8` artifact is a REPACK into the O_DIRECT-aligned block layout, not a quantization.
//! Nothing is fit, nothing is re-rounded, and no error is introduced; `--verify` is therefore
//! allowed to demand byte identity, and does.
//!
//! **The one transform is a DE-INTERLEAVE, and it is the source's own layout.** Within
//! `w13_weight[e]`, rows are `[gate_0, up_0, gate_1, up_1, …]` — transformers' `Interleave`
//! on dim 1 (CONV:365–372), applied to the WEIGHT and to the SCALE grid alike. So the
//! `.mxfp8` block's `gate` slot takes even rows `2j` and its `up` slot odd rows `2j+1`, from
//! both tensors. Treating `w13` as a `[gate; up]` concatenation is trap T2 and produces a
//! file that opens, streams and decodes the gate against the up weights.
//!
//! **One layer at a time, from ONE shard at a time.** [`mxfp8_expert_prefix`] is the sole spelling
//! of the checkpoint's tensor names, and the converter selects shards with it
//! (`Safetensors::open_indexed`) — so an mmap covers the layer in hand rather than the 619 GB
//! checkpoint, and a layer whose shards have arrived converts while the rest still downloads.

use anyhow::{Result, bail, ensure};

use super::Dtype;
use super::header::MXFP8_MAGIC;
use super::repack::{LayerBlocks, LayerRepack};
use super::tensors::Safetensors;
use crate::quant::{
    MXFP8_GROUP, MXFP8_NAN_SCALE, mxfp8_expert_bytes, mxfp8_expert_stride, mxfp8_groups,
    mxfp8_slot_offsets,
};

/// The checkpoint's tensor-name prefix for one layer's routed experts.
///
/// ONE spelling, because two would be a shard-selection predicate that disagrees with the
/// reads it selected for. Verified against the vendored `hf_quant_config.json`, whose MXFP8
/// parameter group names exactly `model.llm.layers.{l}.mlp.experts.w13_weight` and
/// `.w2_weight` for all 48 layers.
pub fn mxfp8_expert_prefix(layer: usize) -> String {
    format!("model.llm.layers.{layer}.mlp.experts.")
}

/// One `.mxfp8` layer file's worth of work: write it if absent, then optionally prove it.
///
/// Mirrors [`super::repack::RoutedRepack`] in shape and differs in exactly the two things
/// that belong to the format — the interleaved source spans
/// ([`Mxfp8Expert::for_each`]) and the `.mxfp8` geometry — because the two formats' files are
/// built by two different writers but the RESUME and VERIFY policies must not be two.
pub struct Mxfp8Layer<'a> {
    /// Program name for the progress lines, so the log says which converter is running.
    pub tool: &'a str,
    pub out_dir: &'a str,
    /// The shard(s) holding THIS layer's four expert tensors — one or two files, never the
    /// whole checkpoint.
    pub src: &'a Safetensors,
    pub layer: usize,
    pub n_experts: usize,
    /// The width the routed block is entered at: Small's `hidden_size` (4608), which is what
    /// `[1, 32]` scales along the input dim of gate/up, and the OUTPUT of `w2`.
    pub expert_in: usize,
    /// `intermediate_size` (2304) — the routed experts' intermediate width.
    pub moe_inter: usize,
    /// Write the layer file when it is absent. False for a `--verify-only` pass.
    pub write: bool,
    /// Re-read whatever is on disk and compare it against the source spans.
    pub verify: bool,
}

/// The layer file's path and block geometry, plus whether it is already on disk.
///
/// Resolved once and handed to both halves, so the writer and the verifier cannot disagree
/// about the stride or the block size — the pair that decides where every expert starts, and
/// the pair a wrong answer to would be written and read back consistently.
struct Plan {
    path: String,
    stride: usize,
    ebytes: usize,
    reused: bool,
}

/// The three numbers one expert block is written with — what a config holds, in the order a
/// caller should read them: the expert count, the width the block is ENTERED at, and the
/// routed intermediate width.
#[derive(Clone, Copy)]
pub struct LayerShape {
    pub n_experts: usize,
    /// Small's `hidden_size` (4608), not `intermediate_size`: the block is entered at `hidden`
    /// and `[1, 32]` scales run along that dim.
    pub expert_in: usize,
    pub moe_inter: usize,
}

/// Where one layer's work comes from: the program name for the progress lines, the artifact
/// directory it writes into, and the shard set holding its four expert tensors.
pub struct LayerInputs<'a> {
    pub tool: &'a str,
    pub out_dir: &'a str,
    pub src: &'a Safetensors,
}

impl<'a> Mxfp8Layer<'a> {
    /// One layer's work, in the two-flag form every caller needs: `write` and `verify` start
    /// false, and [`Self::mode`] sets them.
    ///
    /// A constructor rather than a ten-field literal at each call site — the literal is how a
    /// caller comes to pass `moe_inter` where `expert_in` belongs, and how four call sites
    /// come to drift apart.
    pub fn new(inputs: LayerInputs<'a>, layer: usize, shape: LayerShape) -> Self {
        Self {
            tool: inputs.tool,
            out_dir: inputs.out_dir,
            src: inputs.src,
            layer,
            n_experts: shape.n_experts,
            expert_in: shape.expert_in,
            moe_inter: shape.moe_inter,
            write: false,
            verify: false,
        }
    }

    /// The two flags ARE the three modes: write (and verify if asked), verify-only (neither
    /// writes), and read nothing (both false, which only a caller checking existence wants).
    pub fn mode(mut self, write: bool, verify: bool) -> Self {
        self.write = write;
        self.verify = verify;
        self
    }

    /// Write the layer if asked to, then verify it if asked to.
    ///
    /// A run either produces the file or it does not touch it; the read-only branch is the one
    /// `--verify-only` promised in its `--help`. The policy is `LayerRepack::run`'s, shared
    /// with the `.f4` writer.
    pub fn run(&self) -> Result<()> {
        let plan = self.plan();
        let blocks = LayerBlocks::new(&plan.path, (plan.stride, plan.ebytes, self.n_experts));
        let file = LayerRepack::new(
            (self.tool, "mxfp8", self.layer),
            (self.write, self.verify, plan.reused),
            blocks,
        );
        file.run(
            || self.write_layer(&file),
            || {
                let diff = file.differing(|e, blk| Ok(self.expert(e).diff(blk)?.len()))?;
                file.report(diff)
            },
        )
    }

    fn plan(&self) -> Plan {
        let path = format!("{}/L{:02}.mxfp8", self.out_dir, self.layer);
        Plan {
            reused: std::fs::metadata(&path).is_ok(),
            stride: mxfp8_expert_stride(self.expert_in, self.moe_inter),
            ebytes: mxfp8_expert_bytes(self.expert_in, self.moe_inter),
            path,
        }
    }

    /// Expert `e` of this layer, located in the source — one construction site, shared by the
    /// writer and the verifier so both read the same spans.
    fn expert(&self, e: usize) -> Mxfp8Expert<'_> {
        Mxfp8Expert {
            src: self.src,
            prefix: mxfp8_expert_prefix(self.layer),
            e,
            n_experts: self.n_experts,
            expert_in: self.expert_in,
            moe_inter: self.moe_inter,
        }
    }

    fn write_layer(&self, file: &LayerRepack<'_>) -> Result<()> {
        if file.skip_reused() {
            return Ok(());
        }
        // Header block, then exactly `n_experts` routed blocks and NO shared one: Small ships
        // `n_shared_experts 0`, so a block written past `n_experts` would be geometry the
        // loader can never address. Streamed in windows by [`LayerRepack::write`] — a Small
        // layer is 12.61 GB, and buffering it would be host RAM the GPU shares.
        let header = file.header(MXFP8_MAGIC, (self.expert_in, self.moe_inter));
        file.write(header, |e, slot| self.expert(e).pack(slot))
    }
}

/// One routed MXFP8 expert as it sits in the SOURCE checkpoint: `w13` and its scale grid
/// (gate/up interleaved), `w2` and its scale grid.
///
/// The shape checks and the ue8m0 refusal live here because this is the only path that reads
/// every routed scale byte — at decode the bytes DMA from NVMe straight into a pool slot and
/// the host never sees them. Every check names the tensor, expert, row and group, so a bad
/// byte is a five-minute fix rather than a day of "the output looks wrong".
pub struct Mxfp8Expert<'a> {
    src: &'a Safetensors,
    /// [`mxfp8_expert_prefix`]'s value for this expert's layer.
    prefix: String,
    e: usize,
    n_experts: usize,
    expert_in: usize,
    moe_inter: usize,
}

impl Mxfp8Expert<'_> {
    /// Walk this expert's sources in `.mxfp8` block order, calling `put` with each
    /// `(destination offset, source bytes)` pair.
    ///
    /// ONE definition of the layout, used by both [`Self::pack`] and [`Self::diff`]. The
    /// offsets come from [`mxfp8_slot_offsets`] — the SAME function the streaming pool points
    /// its descriptor at — so a change to the layout moves both ends or neither.
    ///
    /// Grouped as one callback rather than returning a `Vec` of slices because the gate and up
    /// slots are STRIDED in the source (every other row) and contiguous in the destination:
    /// 4608 per-row pairs for one expert, which a `Vec<(&[u8], usize)>` would allocate per
    /// expert per pass for no reader's benefit.
    fn for_each(&self, mut put: impl FnMut(usize, &[u8])) -> Result<()> {
        let (e, h, i) = (self.e, self.expert_in, self.moe_inter);
        let (groups_in, groups_out) = (mxfp8_groups(h), mxfp8_groups(i));
        let name = |suffix: &str| format!("{}{suffix}", self.prefix);
        let (w13_name, w2_name) = (name("w13_weight"), name("w2_weight"));
        let (s13_name, s2_name) = (name("w13_weight_scale"), name("w2_weight_scale"));
        let (wb, wsh) = self.src.typed(&w13_name, Dtype::F8E4M3)?;
        let (sb, ssh) = self.src.typed(&s13_name, Dtype::U8)?;
        let (db, dsh) = self.src.typed(&w2_name, Dtype::F8E4M3)?;
        let (tb, tsh) = self.src.typed(&s2_name, Dtype::U8)?;
        // Shapes are checked, not trusted: a transposed or mis-blocked source copies the right
        // NUMBER of bytes in the wrong order, which is a file that passes every length check
        // and decodes to noise.
        ensure!(
            wsh == [self.n_experts, 2 * i, h] && ssh == [self.n_experts, 2 * i, groups_in],
            "{w13_name}: {wsh:?} / {s13_name}: {ssh:?} != [{}, {}, {h}] and its {groups_in} \
             ue8m0 groups (gate/up interleaved, one e4m3 value per weight)",
            self.n_experts,
            2 * i
        );
        ensure!(
            dsh == [self.n_experts, h, i] && tsh == [self.n_experts, h, groups_out],
            "{w2_name}: {dsh:?} / {s2_name}: {tsh:?} != [{}, {h}, {i}] and its {groups_out} \
             ue8m0 groups",
            self.n_experts
        );
        let off = mxfp8_slot_offsets(h, i);
        let (row, srow) = (2 * i * h, 2 * i * groups_in);
        let we = &wb[e * row..][..row];
        let se = &sb[e * srow..][..srow];
        for j in 0..i {
            put(off[0] + j * h, &we[2 * j * h..][..h]);
            let gs = &se[2 * j * groups_in..][..groups_in];
            refuse_nan_scale(&s13_name, e, 2 * j, gs)?;
            put(off[1] + j * groups_in, gs);
            put(off[2] + j * h, &we[(2 * j + 1) * h..][..h]);
            let us = &se[(2 * j + 1) * groups_in..][..groups_in];
            refuse_nan_scale(&s13_name, e, 2 * j + 1, us)?;
            put(off[3] + j * groups_in, us);
        }
        let de = &db[e * h * i..][..h * i];
        let te = &tb[e * h * groups_out..][..h * groups_out];
        // Down's scale grid is the whole expert's, one row per output row — walked by chunks
        // so the row index and the bytes it names cannot drift apart.
        for (r, line) in te.chunks_exact(groups_out).enumerate() {
            refuse_nan_scale(&s2_name, e, r, line)?;
        }
        put(off[4], de);
        put(off[5], te);
        Ok(())
    }

    /// Repack into `dst` (`mxfp8_expert_bytes` long). A byte copy plus the de-interleave —
    /// nothing is fit, nothing is re-rounded.
    pub fn pack(&self, dst: &mut [u8]) -> Result<()> {
        let want = mxfp8_expert_bytes(self.expert_in, self.moe_inter);
        ensure!(
            dst.len() == want,
            "{}expert {}: destination is {} bytes, an MXFP8 expert block is {want}",
            self.prefix,
            self.e,
            dst.len()
        );
        self.for_each(|offset, bytes| dst[offset..offset + bytes.len()].copy_from_slice(bytes))
    }

    /// Byte offsets within `block` that disagree with the source tensors. **Empty means the
    /// repack was bit-exact.**
    ///
    /// Returns the offsets rather than a bool so a caller can say WHICH bytes moved, which is
    /// what lets `convert_inkling_small --verify` name where a 12.61 GB layer file went wrong.
    /// Shares [`Self::for_each`] with [`Self::pack`], so it also inherits the ue8m0 refusal: a
    /// source carrying `0xFF` is unusable whether or not its bytes round-tripped, and "your
    /// source has a NaN scale at w13_weight_scale[7][46][3]" is the actionable report.
    pub fn diff(&self, block: &[u8]) -> Result<Vec<usize>> {
        let want = mxfp8_expert_bytes(self.expert_in, self.moe_inter);
        ensure!(
            block.len() >= want,
            "{}expert {}: block is {} bytes, an MXFP8 expert is {want}",
            self.prefix,
            self.e,
            block.len()
        );
        let mut bad = Vec::new();
        self.for_each(|offset, bytes| {
            for (k, &b) in bytes.iter().enumerate() {
                if block[offset + k] != b {
                    bad.push(offset + k);
                }
            }
        })?;
        Ok(bad)
    }
}

/// Refuse a ue8m0 NaN (`0xFF`) in one row of a scale grid, naming the tensor, expert, row and
/// group.
///
/// The kernels decode `0xFF` correctly (a quiet NaN rather than `2^128`) but cannot REFUSE it,
/// and `moe_fixed`'s saturating clamp then launders the NaN into a finite ±2^14 — one bad byte
/// is 32 weights of plausible garbage with no error anywhere. `0x00` is deliberately NOT
/// refused: it is `2^-127`, a legal encoding the decoders handle as an f32 subnormal.
fn refuse_nan_scale(tensor: &str, e: usize, row: usize, row_scales: &[u8]) -> Result<()> {
    if let Some(g) = row_scales.iter().position(|&b| b == MXFP8_NAN_SCALE) {
        bail!(
            "{tensor}[{e}][{row}][{g}] is 0x{:02x} — the ue8m0 NaN. The MXFP8 kernels cannot \
             reject it and `moe_fixed`'s clamp turns it into a finite ±2^14, so a whole \
             {}-weight group would decode to plausible garbage.",
            MXFP8_NAN_SCALE,
            MXFP8_GROUP
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // The subject is a byte walk over an in-memory source, so nothing here touches a file.
    // Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// The group helper this file's doc argues about, pinned at both of Small's dims: the
    /// gate/up input dim is `hidden` (4608 → 144 groups) and down's is `inter` (2304 → 72).
    #[test]
    fn smalls_scale_grids_are_the_t1_shape_and_not_the_declared_32x32_tile() {
        assert_eq!(mxfp8_groups(4608), 144);
        assert_eq!(mxfp8_groups(2304), 72);
        // A [32, 32] tile grid would store 4608/32 x 2304/32 per expert — 32x fewer bytes.
        assert_ne!(mxfp8_groups(4608) / 32, 144);
    }
}
