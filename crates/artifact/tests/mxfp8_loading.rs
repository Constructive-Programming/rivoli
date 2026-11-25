//! The `.mxfp8` container end to end, deviceless: a synthetic Small-shaped checkpoint goes
//! through the real repacker and comes back out through the real `ExpertSet`.
//!
//! **What this pins, and why each is worth a gate.** The de-interleave (trap T2) is invisible to
//! every length check — a `[gate; up]` concatenation produces the same number of bytes in the
//! wrong order — so the block is compared ROW BY ROW against the source's even/odd rows. The
//! magic is the only thing separating a 12.6 GB `.mxfp8` layer from a mis-read `.f4`, so a
//! well-formed header of another format must be refused. The ue8m0 NaN is refused where the
//! kernels cannot refuse it. And `mxfp8_layer_range` is the loader's whole notion of which
//! layers exist, so its round trip and its three refusals are exercised against a real
//! `manifest.json`.
//!
//! Scratch lives under Cargo's target-dir tmp (`CARGO_TARGET_TMPDIR`), never `/tmp`: this box's
//! `/tmp` is a 63 GB tmpfs inside the RAM the GPU shares.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: a panic here IS the report

use std::os::unix::fs::FileExt;

use rivoli_artifact::format::{
    Dtype, EXPERT_HEADER_BYTES, ExpertSet, F4_MAGIC, LayerInputs, LayerShape, MXFP8_MAGIC,
    Mxfp8Layer, RoutedFmt, SafeWriter, Safetensors, SetDims, TensorData, VQ3_MAGIC,
    mxfp8_expert_prefix, mxfp8_layer_range, mxfp8_source,
};
use rivoli_artifact::quant::{
    VQ_ALIGN, mxfp8_expert_bytes, mxfp8_expert_stride, mxfp8_groups, mxfp8_slot_offsets,
};

/// The synthetic expert-block dims. Small's real ones are 4608x2304 with 384 experts
/// (`quant::mxfp8`'s unit test pins those numbers exactly); these are the smallest multiples of
/// the format's 32-weight group that give a projection more than one group, so a group-index
/// error has somewhere to show.
#[derive(Clone, Copy)]
struct Dims {
    experts: usize,
    hidden: usize,
    inter: usize,
}

const DIMS: Dims = Dims {
    experts: 2,
    hidden: 64,
    inter: 32,
};

impl Dims {
    /// The block geometry as the repacker takes it — same three numbers, one name each.
    fn shape(self) -> LayerShape {
        LayerShape {
            n_experts: self.experts,
            expert_in: self.hidden,
            moe_inter: self.inter,
        }
    }
}

/// One synthetic checkpoint's four expert tensors, in the checkpoint's own layout:
/// `w13` gate/up INTERLEAVED along dim 1, and its scale grid interleaved identically.
struct Synth {
    w13: Vec<u8>,
    s13: Vec<u8>,
    w2: Vec<u8>,
    s2: Vec<u8>,
}

impl Synth {
    /// Deterministic, and with the two properties the assertions need: expert `e`'s content
    /// differs from every other expert's, and a gate row differs from the up row beside it (so
    /// a de-interleave that swapped them could not pass). Scale bytes stay in the ue8m0 range
    /// real checkpoints use (`0x70..=0x7a`), and `poison` plants one `0xFF` at
    /// `(expert, row, group)` of the `w13` grid.
    fn build(d: Dims, poison: Option<(usize, usize, usize)>) -> Self {
        let (e, h, i) = (d.experts, d.hidden, d.inter);
        let (w13, s13) = fill_plane((e, 2 * i, h), (97, 13, 7), |ex, r, g| {
            poison
                .filter(|p| *p == (ex, r, g))
                .map_or(0x70 + ((ex + r + g) % 11) as u8, |_| 0xFF)
        });
        let (w2, s2) = fill_plane((e, h, i), (31, 5, 11), |ex, r, g| {
            0x72 + ((ex + r + g) % 5) as u8
        });
        Self { w13, s13, w2, s2 }
    }
}

/// Fill one `[E, rows, cols]` MXFP8 weight plane and its `[E, rows, ceil(cols/32)]` scale grid.
fn fill_plane(
    (e, rows, cols): (usize, usize, usize),
    (ke, kr, kc): (usize, usize, usize),
    scale: impl Fn(usize, usize, usize) -> u8,
) -> (Vec<u8>, Vec<u8>) {
    let ng = mxfp8_groups(cols);
    let mut w = Vec::with_capacity(e * rows * cols);
    let mut s = Vec::with_capacity(e * rows * ng);
    for ex in 0..e {
        for r in 0..rows {
            w.extend((0..cols).map(|c| ((ex * ke + r * kr + c * kc) % 200 + 1) as u8));
            s.extend((0..ng).map(|g| scale(ex, r, g)));
        }
    }
    (w, s)
}
/// A fresh scratch directory under Cargo's own tmp — never `/tmp`, which on this box is a
/// 63 GB tmpfs inside the RAM the GPU shares.
///
/// The REMOVE is load-bearing rather than tidiness: the repacker resumes by REUSING an
/// existing `L00.mxfp8` without reading it, so a stale file from a previous run would make
/// every assertion below pass against bytes this run never wrote.
fn scratch(tag: &str) -> String {
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("mxfp8_{tag}"));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).unwrap();
    dir.to_string_lossy().into_owned()
}

/// Write the synthetic checkpoint as one safetensors file named the way the repacker reads it.
fn write_source(dir: &str, d: Dims, s: &Synth) -> String {
    let (e, h, i) = (d.experts, d.hidden, d.inter);
    let (gi, go) = (mxfp8_groups(h), mxfp8_groups(i));
    let base = mxfp8_expert_prefix(0);
    let mut w = SafeWriter::new();
    w.add(
        format!("{base}w13_weight"),
        Dtype::F8E4M3,
        TensorData::new(vec![e, 2 * i, h], s.w13.clone()),
    );
    w.add(
        format!("{base}w13_weight_scale"),
        Dtype::U8,
        TensorData::new(vec![e, 2 * i, gi], s.s13.clone()),
    );
    w.add(
        format!("{base}w2_weight"),
        Dtype::F8E4M3,
        TensorData::new(vec![e, h, i], s.w2.clone()),
    );
    w.add(
        format!("{base}w2_weight_scale"),
        Dtype::U8,
        TensorData::new(vec![e, h, go], s.s2.clone()),
    );
    let path = format!("{dir}/source.safetensors");
    w.write(&path).unwrap();
    path
}

/// Repack one layer into `out`. `verify` is always on: the writer's offsets are only
/// meaningful against bytes read back from disk.
fn convert(dir: &str, out: &str, d: Dims, s: &Synth) {
    let src_path = write_source(dir, d, s);
    let src = Safetensors::open_file(&src_path).unwrap();
    std::fs::create_dir_all(out).unwrap();
    let inputs = LayerInputs {
        tool: "mxfp8_loading_test",
        out_dir: out,
        src: &src,
    };
    Mxfp8Layer::new(inputs, 0, d.shape())
        .mode(true, true)
        .run()
        .unwrap();
}

/// Expert `e`'s useful block bytes, read back through the loader's own offsets.
fn read_block(out: &str, e: usize, d: Dims) -> Vec<u8> {
    let stride = mxfp8_expert_stride(d.hidden, d.inter);
    let ebytes = mxfp8_expert_bytes(d.hidden, d.inter);
    let f = std::fs::File::open(format!("{out}/L00.mxfp8")).unwrap();
    let mut b = vec![0u8; ebytes];
    f.read_exact_at(&mut b, (VQ_ALIGN + e * stride) as u64)
        .unwrap();
    b
}

fn set_dims(d: Dims) -> SetDims {
    SetDims::new(0..1, d.experts, d.hidden, d.inter)
}

#[test]
fn a_repacked_layer_opens_and_its_gate_and_up_slots_are_the_deinterleaved_source() {
    let dir = scratch("roundtrip");
    let out = format!("{dir}/art");
    let s = Synth::build(DIMS, None);
    convert(&dir, &out, DIMS, &s);
    let (h, i) = (DIMS.hidden, DIMS.inter);
    let g = mxfp8_groups(h);

    // The file the loader will open: VQ_ALIGN header block + exactly `n_experts` blocks, and
    // an MXFP8 header in the first 40 bytes.
    let stride = mxfp8_expert_stride(h, i);
    let len = std::fs::metadata(format!("{out}/L00.mxfp8")).unwrap().len();
    assert_eq!(len, (VQ_ALIGN + DIMS.experts * stride) as u64);
    let head = std::fs::read(format!("{out}/L00.mxfp8")).unwrap();
    assert_eq!(&head[0..4], &MXFP8_MAGIC);
    assert_eq!(head.len(), VQ_ALIGN + DIMS.experts * stride);
    assert_eq!(EXPERT_HEADER_BYTES, 40);

    // The reader: geometry, slots and the aligned read spec the streamer consumes.
    let set = ExpertSet::open_routed(&out, RoutedFmt::MxFp8, set_dims(DIMS)).unwrap();
    assert_eq!(set.fmt(), RoutedFmt::MxFp8);
    assert_eq!(set.slot_offsets(), mxfp8_slot_offsets(h, i));
    assert_eq!(set.n_experts(), DIMS.experts);
    let (_fd, begin, useful) = set.read_spec(0, 1).unwrap();
    assert_eq!(begin, VQ_ALIGN + stride);
    assert_eq!(useful, mxfp8_expert_bytes(h, i));
    assert_eq!(begin % VQ_ALIGN, 0);

    // The de-interleave, row by row: gate slot = even source rows, up slot = odd ones, and the
    // scale grids follow their weights.
    let off = mxfp8_slot_offsets(h, i);
    let block = read_block(&out, 0, DIMS);
    for j in 0..i {
        let gate = &block[off[0] + j * h..off[0] + (j + 1) * h];
        let up = &block[off[2] + j * h..off[2] + (j + 1) * h];
        assert_eq!(gate, &s.w13[2 * j * h..(2 * j + 1) * h], "gate row {j}");
        assert_eq!(up, &s.w13[(2 * j + 1) * h..(2 * j + 2) * h], "up row {j}");
        assert_ne!(gate, up, "the fixture must tell gate and up apart");
        assert_eq!(
            &block[off[1] + j * g..off[1] + (j + 1) * g],
            &s.s13[2 * j * g..(2 * j + 1) * g],
            "gate scale row {j}"
        );
        assert_eq!(
            &block[off[3] + j * g..off[3] + (j + 1) * g],
            &s.s13[(2 * j + 1) * g..(2 * j + 2) * g],
            "up scale row {j}"
        );
    }
    // `w2` is a plain copy: `[hidden, inter]` weights then `[hidden, inter/32]` scales.
    let go = mxfp8_groups(i);
    assert_eq!(&block[off[4]..off[4] + h * i], &s.w2[..h * i]);
    assert_eq!(&block[off[5]..off[5] + h * go], &s.s2[..h * go]);
    // 6336 useful bytes, padded to 8192 — the toy dims' stride, asserted once so the offsets
    // above cannot be silently re-based by a geometry change.
    assert_eq!(mxfp8_expert_bytes(h, i), 6336);
    assert_eq!(stride, 8192);
}

#[test]
fn a_layer_file_of_another_format_is_refused_by_its_magic_not_its_length() {
    let dir = scratch("magic");
    let good = format!("{dir}/good");
    let s = Synth::build(DIMS, None);
    convert(&dir, &good, DIMS, &s);
    let bytes = std::fs::read(format!("{good}/L00.mxfp8")).unwrap();
    for (tag, magic) in [("f4", F4_MAGIC), ("vq3", VQ3_MAGIC)] {
        let bad = format!("{dir}/{tag}");
        std::fs::create_dir_all(&bad).unwrap();
        let mut patched = bytes.clone();
        patched[0..4].copy_from_slice(&magic);
        std::fs::write(format!("{bad}/L00.mxfp8"), &patched).unwrap();
        // Same LENGTH, so the refusal is the magic: this is the check the length cannot make.
        assert_eq!(
            std::fs::metadata(format!("{bad}/L00.mxfp8")).unwrap().len(),
            bytes.len() as u64
        );
        // `let … else`, not `expect_err`: that would demand `ExpertSet: Debug`, and an
        // `ExpertSet` holds open O_DIRECT fds — the whole point is that it cannot be printed.
        let Err(err) = ExpertSet::open_routed(&bad, RoutedFmt::MxFp8, set_dims(DIMS)) else {
            panic!("a {tag} magic must not open as .mxfp8");
        };
        let text = format!("{err:#}");
        assert!(text.contains("mxfp8 magic"), "{tag}: {text}");
    }
}

#[test]
fn an_e8m0_nan_scale_byte_is_refused_at_repack_and_writes_no_file() {
    let dir = scratch("nan");
    let out = format!("{dir}/art");
    // Expert 0, w13 row 4, group 1 (of 2 groups for hidden=64) — a gate row, which is the slot a transposed read would hit.
    let s = Synth::build(DIMS, Some((0, 4, 1)));
    let src_path = write_source(&dir, DIMS, &s);
    let src = Safetensors::open_file(&src_path).unwrap();
    std::fs::create_dir_all(&out).unwrap();
    let inputs = LayerInputs {
        tool: "mxfp8_loading_test",
        out_dir: &out,
        src: &src,
    };
    let err = Mxfp8Layer::new(inputs, 0, DIMS.shape())
        .mode(true, false)
        .run()
        .expect_err("0xFF is the ue8m0 NaN and must not reach an artifact");
    let text = format!("{err:#}");
    assert!(text.contains("ue8m0 NaN"), "{text}");
    assert!(
        text.contains("[0][4][1]"),
        "the refusal must name the byte: {text}"
    );
    // Nothing was published: `write_expert_layer` renames a tmp file into place, so a refusal
    // in the fill leaves no `L00.mxfp8` for a later run to "reuse" unverified.
    assert!(std::fs::metadata(format!("{out}/L00.mxfp8")).is_err());
}

#[test]
fn the_manifest_layer_range_round_trips_and_refuses_what_it_cannot_describe() {
    let dir = scratch("range");
    let art = format!("{dir}/art");
    std::fs::create_dir_all(&art).unwrap();
    let write = |layers: std::ops::Range<usize>| {
        let m = serde_json::json!({
            "mxfp8_source": mxfp8_source("convert_inkling_small", "/checkpoints/small", layers),
        });
        std::fs::write(
            format!("{art}/manifest.json"),
            serde_json::to_vec(&m).unwrap(),
        )
        .unwrap();
    };
    write(3..7);
    // Read back, and bounded by the model's own layer count.
    assert_eq!(mxfp8_layer_range(&art, 48).unwrap(), 3..7);
    // `to > num_hidden_layers` and an empty range are both refused rather than clipped.
    write(3..49);
    assert!(mxfp8_layer_range(&art, 48).is_err());
    write(3..3);
    assert!(mxfp8_layer_range(&art, 48).is_err());
    // Absent block: an ERROR (the loader's only source for which layers exist), and so is a
    // manifest that is not there at all.
    std::fs::write(format!("{art}/manifest.json"), b"{}").unwrap();
    assert!(mxfp8_layer_range(&art, 48).is_err());
    let empty = scratch("range_empty");
    assert!(mxfp8_layer_range(&empty, 48).is_err());
}
