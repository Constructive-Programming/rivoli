//! MXFP8 → MXFP4 rows: the `--f4` expert half both Inkling converters share.
//!
//! `convert_inkling` (Mini) and `convert_inkling_small` (Small-MXFP8) each grew `--f4` on
//! 2026-11-25, and the work under it is the same three things: decode a source row, quantize it
//! into an `.f4` expert block, and publish that block atomically in windows. Written twice it is
//! a duplication-gate clone AND a second definition of the block layout — the failure each of
//! this tree's routed formats has already paid for once. So it lives here, and each binary
//! includes the file with `#[path]`: cargo autobin-discovers every `src/bin/*.rs` as its own
//! executable, so a shared module cannot sit beside the two roots as a bare `.rs` (the
//! `convert/gpu.rs` precedent, same reason).
//!
//! **The destination geometry is never re-derived.** Every offset comes from
//! [`f4_slot_offsets`], [`f4_row_bytes`] and [`f4_groups`] — the functions the streaming pool
//! points its descriptor at and `F4Expert`'s spans tile — so a row quantized here is a row the
//! kernel reads.
//!
//! **`--f4` is the one LOSSY path in this tree's routed experts.** Every other `.f4` producer
//! repacks a source that already ships e2m1 (see [`quantize_f4_row`]); Inkling 2's experts ship
//! NVFP4 or MXFP8, so this quantizes and the converters' paired-dNLL gate is what scores it.

use anyhow::{Result, ensure};
use rivoli_artifact::format::{
    Dtype, ExpertFile, ExpertHeader, F4_MAGIC, LAYER_WINDOW, LayerDims, Safetensors,
    write_expert_layer,
};
use rivoli_artifact::quant::{
    ExpertBlocks, MXFP8_GROUP, e8m0, f4_expert_bytes, f4_expert_stride, f4_groups, f4_row_bytes,
    f4_slot_offsets, mxfp8_groups, quantize_f4_row,
};
use rivoli_core::num::e4m3_to_f32;
use std::sync::LazyLock;

/// A row-addressable weight plane — the one thing the three `--f4` sources have in common.
///
/// Mini's NVFP4 planes (`decode_expert_row`), a checkpoint's MXFP8 planes and an `.mxfp8` layer
/// file's already-deinterleaved spans are three different geometries, and the expert loop below
/// must not learn any of them: it asks for row `r` and gets `cols` f32 values back.
pub trait RowSource {
    /// Decode `row` into `out`, which must be exactly as long as the row is wide.
    fn row(&self, row: usize, out: &mut [f32]) -> Result<()>;
}

/// One MXFP8 plane: `rows` rows of `cols` e4m3 weights, each row with `ceil(cols / 32)` ue8m0
/// bytes — ROW-WISE `[1, 32]`, which is the grid the checkpoint's scale shapes and the artifact's
/// header both carry (trap T1: the `[32, 32]` in `hf_quant_config.json` is refuted by the shapes).
pub struct Mxfp8Rows<'a> {
    weights: &'a [u8],
    scales: &'a [u8],
    rows: usize,
    cols: usize,
}

impl<'a> Mxfp8Rows<'a> {
    /// Bind one plane's two spans and check them against `(rows, cols)`.
    ///
    /// Checked at construction rather than per row: a transposed or mis-blocked plane copies the
    /// right NUMBER of bytes in the wrong order, which is a file that passes every length check
    /// downstream and decodes to noise.
    pub fn new(weights: &'a [u8], scales: &'a [u8], (rows, cols): (usize, usize)) -> Result<Self> {
        let (want_w, want_s) = (rows * cols, rows * mxfp8_groups(cols));
        ensure!(
            cols.is_multiple_of(MXFP8_GROUP) && weights.len() == want_w && scales.len() == want_s,
            "MXFP8 plane: {} weight bytes / {} scale bytes for {rows} rows of {cols} — expected \
             {want_w} / {want_s} at a {MXFP8_GROUP}-weight group",
            weights.len(),
            scales.len()
        );
        Ok(Self {
            weights,
            scales,
            rows,
            cols,
        })
    }
}

impl RowSource for Mxfp8Rows<'_> {
    fn row(&self, row: usize, out: &mut [f32]) -> Result<()> {
        ensure!(
            row < self.rows && out.len() == self.cols,
            "MXFP8 plane: row {row} of {} into {} values for {} columns",
            self.rows,
            out.len(),
            self.cols
        );
        let groups = mxfp8_groups(self.cols);
        let w = &self.weights[row * self.cols..][..self.cols];
        let s = &self.scales[row * groups..][..groups];
        let table = e4m3_table();
        for (g, chunk) in w.chunks_exact(MXFP8_GROUP).enumerate() {
            let scale = e8m0(s[g])?;
            let dst = &mut out[g * MXFP8_GROUP..][..MXFP8_GROUP];
            for (o, &byte) in dst.iter_mut().zip(chunk) {
                *o = table[byte as usize] * scale;
            }
        }
        Ok(())
    }
}

/// The 256 e4m3 values, decoded once per process.
///
/// A Small layer is 6912 rows of up to 4608 weights and there are 18,432 experts, so the decode
/// runs billions of times; this is [`e4m3_to_f32`] tabulated over every byte and hoisted out of
/// the inner loop, so it cannot drift from the function it tabulates. The scale byte goes through
/// [`e8m0`] rather than a shift because `0xFF` is the format's NaN and must be REFUSED on the one
/// path that reads every routed scale byte — 32 weights of plausible garbage otherwise.
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: LazyLock<[f32; 256]> = LazyLock::new(|| {
        let mut table = [0f32; 256];
        for (byte, slot) in table.iter_mut().enumerate() {
            *slot = e4m3_to_f32(byte as u8);
        }
        table
    });
    &TABLE
}

/// One expert's two interleaved sources and the width its block is entered at.
#[derive(Clone, Copy)]
pub struct ExpertSrc<'a> {
    /// `w13`: rows `2j` are gate and `2j + 1` up — the checkpoint's own interleave (transformers'
    /// `Interleave` on dim 1, CONV:365–372). Treating it as a `[gate; up]` concatenation is trap
    /// T2: a file that opens, streams, and decodes the gate against the up weights.
    pub w13: &'a dyn RowSource,
    /// `w2`: `hidden` rows of `inter` — down, in place.
    pub w2: &'a dyn RowSource,
    /// `(hidden, inter)` — the two widths every offset below is derived from.
    pub dims: (usize, usize),
}

/// Quantize one expert into its `.f4` slot (`f4_expert_bytes` long), writing every byte of it.
///
/// The write obligation is [`write_expert_layer`]'s: a window buffer is reused across experts, so
/// a partial write would leave the previous expert's bytes in this one's span.
pub fn quantize_expert_f4(slot: &mut [u8], src: ExpertSrc<'_>) -> Result<()> {
    let (h, i) = src.dims;
    let bytes = f4_expert_bytes(h, i);
    ensure!(
        slot.len() == bytes,
        "{}-byte slot for an {bytes}-byte MXFP4 block ({h}×{i})",
        slot.len()
    );
    let off = f4_slot_offsets(h, i);
    let (g_in, g_out) = (f4_groups(h), f4_groups(i));
    let (gate_w, down_w) = (f4_row_bytes(h), f4_row_bytes(i));
    let mut row = vec![0f32; h.max(i)];
    for j in 0..i {
        src.w13.row(2 * j, &mut row[..h])?;
        put_f4_row(slot, (off[0] + j * gate_w, off[1] + j * g_in), &row[..h])?;
        src.w13.row(2 * j + 1, &mut row[..h])?;
        put_f4_row(slot, (off[2] + j * gate_w, off[3] + j * g_in), &row[..h])?;
    }
    for r in 0..h {
        src.w2.row(r, &mut row[..i])?;
        put_f4_row(slot, (off[4] + r * down_w, off[5] + r * g_out), &row[..i])?;
    }
    Ok(())
}

/// Quantize one row into its slot: `at.0` is the packed span's start, `at.1` the ue8m0 grid's.
/// The two offsets come from [`f4_slot_offsets`], so a row written here is a row the streaming
/// pool's descriptor points at.
///
/// `split_at_mut` rather than two index borrows, because a projection's scale grid FOLLOWS its
/// packed rows (`off[1] = off[0] + rows × cols`) and the two spans are therefore disjoint with
/// the packed one first — the one fact `split_at_mut(at.1)` relies on.
fn put_f4_row(slot: &mut [u8], at: (usize, usize), row: &[f32]) -> Result<()> {
    let (packed_at, scales_at) = at;
    let (head, tail) = slot.split_at_mut(scales_at);
    let width = f4_row_bytes(row.len());
    quantize_f4_row(
        row,
        &mut head[packed_at..packed_at + width],
        &mut tail[..f4_groups(row.len())],
    )
}

/// One `L{ll}.f4` file's identity and geometry.
#[derive(Clone, Copy)]
pub struct F4LayerSpec<'a> {
    /// The artifact directory the file is written into.
    pub out_dir: &'a str,
    /// The layer index, which is both the file's name and its header's.
    pub layer: usize,
    /// ROUTED experts — an `.f4` block holds exactly this many (Inkling 2 has no shared expert).
    pub n_experts: usize,
    /// `(hidden, inter)` — the width the block is entered at, and the routed intermediate width.
    pub dims: (usize, usize),
}

/// Write `L{ll}.f4` in windows, or keep the one already there.
///
/// Returns whether this call wrote, so a caller can say "written" or "kept" without stat-ing the
/// path again. Resume is by SKIPPING an existing file, the policy every routed writer here shares:
/// [`write_expert_layer`] publishes only by rename, so an existing file is complete.
///
/// `fill` is called for each expert with its block, fanned across the threads the window is
/// split over (`fill_expert_blocks`), which is the parallelism a per-layer `thread::scope` would
/// have to reproduce — and the window is what bounds host memory, not the layer.
pub fn write_f4_layer(
    spec: F4LayerSpec<'_>,
    fill: impl Fn(usize, &mut [u8]) -> Result<()> + Sync,
) -> Result<bool> {
    let (h, i) = spec.dims;
    let path = format!("{}/L{:02}.f4", spec.out_dir, spec.layer);
    if std::fs::metadata(&path).is_ok() {
        return Ok(false);
    }
    let stride = f4_expert_stride(h, i);
    let header = ExpertHeader::new(
        F4_MAGIC,
        LayerDims {
            layer: spec.layer,
            n_experts: spec.n_experts,
            expert_in: h,
            moe_inter: i,
            stride,
        },
    )
    .to_bytes();
    let file = ExpertFile {
        path: &path,
        header: &header,
    };
    let blocks = ExpertBlocks {
        stride,
        bytes: f4_expert_bytes(h, i),
        count: spec.n_experts,
    };
    write_expert_layer(file, blocks, LAYER_WINDOW, fill)?;
    Ok(true)
}

/// Quantize one checkpoint layer's four interleaved MXFP8 expert tensors into `L{ll}.f4`.
///
/// The shape check is the whole defence against a mis-blocked source: `w13` is
/// `[E, 2·inter, hidden]` with `[E, 2·inter, hidden/32]` scales and `w2` the same pair at
/// `[E, hidden, inter]`, and a transposed pair keeps every byte count and decodes to noise.
/// Named after the tensors it reads (`<prefix>w13_weight` …), so the message points at one.
pub fn quantize_layer_mxfp8(
    spec: F4LayerSpec<'_>,
    src: &Safetensors,
    prefix: &str,
) -> Result<bool> {
    let (n, (h, i)) = (spec.n_experts, spec.dims);
    let (g_in, g_out) = (mxfp8_groups(h), mxfp8_groups(i));
    let name = |suffix: &str| format!("{prefix}{suffix}");
    let (wb, wsh) = src.typed(&name("w13_weight"), Dtype::F8E4M3)?;
    let (sb, ssh) = src.typed(&name("w13_weight_scale"), Dtype::U8)?;
    let (db, dsh) = src.typed(&name("w2_weight"), Dtype::F8E4M3)?;
    let (tb, tsh) = src.typed(&name("w2_weight_scale"), Dtype::U8)?;
    ensure!(
        wsh == [n, 2 * i, h] && ssh == [n, 2 * i, g_in] && dsh == [n, h, i] && tsh == [n, h, g_out],
        "{prefix}: MXFP8 expert shapes {wsh:?} / {ssh:?} / {dsh:?} / {tsh:?} are not [{n}, {}, \
         {h}] / [{n}, {}, {g_in}] / [{n}, {h}, {i}] / [{n}, {h}, {g_out}]",
        2 * i,
        2 * i
    );
    let (w13_row, w2_row) = (2 * i * h, h * i);
    let (w13_scale, w2_scale) = (2 * i * g_in, h * g_out);
    write_f4_layer(spec, |e, slot| {
        let w13 = Mxfp8Rows::new(
            &wb[e * w13_row..][..w13_row],
            &sb[e * w13_scale..][..w13_scale],
            (2 * i, h),
        )?;
        let w2 = Mxfp8Rows::new(
            &db[e * w2_row..][..w2_row],
            &tb[e * w2_scale..][..w2_scale],
            (h, i),
        )?;
        quantize_expert_f4(
            slot,
            ExpertSrc {
                w13: &w13,
                w2: &w2,
                dims: (h, i),
            },
        )
    })
}

/// Convenience wrapper for callers that already hold two [`RowSource`] references.
pub fn quantize_pair_f4(
    slot: &mut [u8],
    w13: &dyn RowSource,
    w2: &dyn RowSource,
    dims: (usize, usize),
) -> Result<()> {
    quantize_expert_f4(slot, ExpertSrc { w13, w2, dims })
}
