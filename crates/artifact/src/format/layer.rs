//! Writing one routed-expert layer file, in bounded host memory and atomically.
//!
//! The one function every converter's inner loop ends at — `bin/convert`'s `.vq3`,
//! `bin/fp8_to_i4`'s `.i4` and [`super::repack`]'s `.f4` — and it is here alone rather than
//! beside any of them because the two properties it carries were each learned by ONE of them
//! and are owed to all three: a window ceiling (K3's 15.72 GB layer) and a pid-suffixed
//! tmp+rename (a killed `bin/convert` leaving an unrepairable short file). Two hand-written
//! copies of this loop is how the second was missing from one of them for months.

use anyhow::{Context, Result, ensure};

use super::header::EXPERT_HEADER_BYTES;
use crate::quant::VQ_ALIGN;

/// Host memory one call to [`write_expert_layer`] may hold, regardless of the layer's size.
///
/// 1 GiB, chosen against the two things it trades off. It must stay large RELATIVE TO one
/// expert block (not a multiple of one — it is a multiple of none of the four strides) so
/// [`crate::quant::fill_expert_blocks`] still has more blocks per window than it has
/// threads. Blocks per 1 GiB window, against ~32 threads:
///
/// | format | expert bytes = stride | per window |
/// |---|---:|---:|
/// | GLM `.vq3` 6144/2048 | 15,335,424 | 70 |
/// | GLM `.i4` 6144/2048 | 20,054,016 | 53 |
/// | V4 `.f4` 4096/2048 | 13,369,344 | 80 |
/// | K3 `.f4` 3584/3072 | 17,547,264 | 61 |
///
/// And it must stay small next to the machine: this box is 128 GB of LPDDR5 *shared with the
/// GPU*, `/tmp` is a 63 GB tmpfs living in that same RAM, and a convert runs alongside whatever
/// else holds the arena.
pub const LAYER_WINDOW: usize = 1 << 30;

/// The header has to fit the block-0 pad it is written into. Both are compile-time constants,
/// so this is a compile-time check — it was briefly an `ensure!` inside
/// [`write_expert_layer`], which is a runtime test of `40 <= 4096` on every layer.
const _: () = assert!(EXPERT_HEADER_BYTES <= VQ_ALIGN);

/// Write one layer's expert file — header block, then `blocks` expert blocks — in bounded host
/// memory, published to `path` by a rename from `<path>.<pid>.part`.
///
/// **Bounded, because the buffered form does not scale past GLM.** Both converters used to
/// allocate the whole file (`vec![0u8; VQ_ALIGN + blocks * stride]`), fill it, and write it in
/// one call: 3.42 GB per layer for V4 (256 blocks) and 3.94 GB for GLM (257 — its shared expert
/// rides the same slab), survivable, but **15.72 GB for Kimi-K3** (896 x 17,547,264) on a box
/// whose entire LPDDR5 is 128 GB and shared with the GPU. The buffer was pure waste at any
/// size — every byte was written out once and never re-read.
/// Parallelism survives because the window is the unit, not the expert:
/// `fill_expert_blocks` still packs each window across all threads over disjoint slices, so
/// serialising the pack to save memory was never on the table.
///
/// **Atomic, because both converters resume by SKIPPING an output path that already exists,
/// without reading it** (`bin/convert` `continue`s, `convert_v4` sets `reused`). A non-atomic
/// write plus a kill would leave a short multi-GB `L{ll}.{ext}` that re-running the tool can
/// never repair: the artifact fails at load on `open_routed`'s `ensure!(len == want)`, and the
/// fix is a manual `rm` of a file nobody would suspect. Found 2026-08-06 because `convert_v4`
/// had tmp+rename from the day it was written and `bin/convert` did not — two hand-written
/// copies of one loop, and the single step where they diverged was the one carrying the defect.
///
/// **The temp name carries the pid, and that is not decoration** — self-review 2026-08-07, the
/// same lesson `bin/ppl`, `f4_loading`, `v4_encoding` and `v4_oracle` each learned on their own
/// scratch paths. Agents share this machine and a convert takes no lock (it is CPU only; the
/// GPU flock does not serialise it), so two runs into one `out_dir` are reachable. On a FIXED
/// `<path>.part` both would `File::create` + truncate + write concurrently, and the rename
/// would publish interleaved bytes. Interleaving two writes **of equal length** yields a file
/// of exactly the right length, so `open_routed`'s length check passes it — the one corruption
/// shape that gets past the loader. The cost is that a killed run leaves
/// `L{ll}.{ext}.<pid>.part` behind rather than a name the next run overwrites, which is the
/// better failure: multi-GB debris under an obviously non-artifact name is visible.
///
/// **No `fsync` before the rename**, deliberately: the guarantee is against process death, not
/// power loss, and one fsync per 3.5 GB layer buys a property no converter has ever claimed.
/// `I4Source::stamp` DOES fsync its manifest, and that asymmetry is right — a torn manifest is
/// unrecoverable, a torn layer file is regenerable.
///
/// **`fill` must write all `bytes` of the slot it is handed.** This is a real obligation, not a
/// formality, and it is stronger than what the buffered form required. The whole-layer `vec!`
/// gave every block fresh zeros, so a closure that wrote only part of its slot left the rest
/// `0x00`; the reused window hands it **the previous expert's payload**. Both current closures
/// are total — `encode_expert` advances `off` by `vq_proj_bytes` across exactly three
/// projections, and `F4Expert::pack` copies six spans that tile `[0, f4_expert_bytes)` — but
/// the tree already contains a helper built to tolerate a short write:
/// [`crate::quant::write_le_scales`] "stop[s] at whichever of the two runs out".
///
/// Concretely, the shape to avoid: a scale iterator one group short (`f4_groups` is
/// `div_ceil`, or a format that reads scales from the source instead of computing them) leaves
/// the tail of that projection's scale span holding **another expert's e8m0 exponents**. Right
/// file length, every length check passes, and `--verify` compares through the same `spans()`
/// so it never looks there. Under the buffered writer the identical bug wrote `0x00` = a dead
/// group, which is visible. `bin/fp8_to_i4` states this same requirement correctly for its own
/// reused buffer; the debug-only clear below keeps dev builds behaving like the old writer.
///
/// The `bytes..stride` padding is a weaker and separate matter: nothing writes it at all, so it
/// stays zero from the single allocation — see the comment there.
///
/// `window` is the host-memory ceiling and both converters pass [`LAYER_WINDOW`]. It is a
/// parameter rather than a constant read from inside so a test can reach the window BOUNDARY
/// without allocating a gigabyte; a thin wrapper supplying the constant would be a second copy
/// of this parameter list, which `jscpd` refuses (measured: 6 lines, 45 tokens).
pub fn write_expert_layer(
    path: &str,
    header: &[u8; EXPERT_HEADER_BYTES],
    stride: usize,
    bytes: usize,
    blocks: usize,
    window: usize,
    fill: impl Fn(usize, &mut [u8]) -> Result<()> + Sync,
) -> Result<u64> {
    use std::io::Write;
    // A `stride` of 0 would reach `chunks_exact_mut(0)` and PANIC rather than return, and
    // `window / stride` would divide by zero one line further down. Refused, not clamped: the
    // `.max(1)` this replaced turned a caller's bad dimension into a panic in someone else's
    // function. `bytes <= stride` is `fill_expert_blocks`'s own check and is left to it.
    ensure!(
        stride > 0,
        "expert stride is 0 — no block geometry to write"
    );
    let part = format!("{path}.{}.part", std::process::id());
    // Not a `BufWriter`. Every write below is either the one `VQ_ALIGN` header block or a whole
    // window, and `BufWriter` passes any write at or above its capacity straight through, so it
    // would buffer nothing and its flush would guard nothing.
    let mut f = std::fs::File::create(&part).with_context(|| format!("create {part}"))?;
    // Block 0 is the header, padded to `VQ_ALIGN` so expert 0 starts block-aligned for the
    // loader's O_DIRECT reads. Same layout the buffered writer produced.
    let mut pad = [0u8; VQ_ALIGN];
    pad[..EXPERT_HEADER_BYTES].copy_from_slice(header);
    f.write_all(&pad).with_context(|| format!("write {part}"))?;

    let per = (window / stride).clamp(1, blocks.max(1));
    // Zeroed ONCE, not per window, and the reuse is safe for a specific reason:
    // `fill_expert_blocks` hands each closure `&mut slot[..bytes]`, so the `bytes..stride`
    // padding is never written by anybody and stays zero for the buffer's whole life — the
    // same way it did in the whole-layer `vec!` this replaced. A per-window `fill(0)` was here
    // first, justified as stopping one window's tail leaking into the next window's padding;
    // the red-proof for that showed the test stayed GREEN without it, because no path dirties
    // padding at all. It was a memset of up to 1 GiB per window (~16 GiB per K3 layer)
    // defending against nothing.
    let mut win = vec![0u8; per * stride];
    for start in (0..blocks).step_by(per) {
        let span = &mut win[..per.min(blocks - start) * stride];
        // Dev-profile only, and NOT for the padding — for the DATA region. It costs nothing in
        // release and makes a `fill` that writes less than `bytes` degrade the way it did under
        // the whole-layer buffer: zeros, a visibly dead group, instead of the previous expert's
        // payload read as this one's. See the `fill` obligation in the doc comment. This is
        // insurance, not a check — it cannot report the short write, only defuse it.
        #[cfg(debug_assertions)]
        span.fill(0);
        crate::quant::fill_expert_blocks(span, stride, bytes, span.len() / stride, |j, slot| {
            fill(start + j, slot)
        })?;
        f.write_all(span).with_context(|| format!("write {part}"))?;
    }
    // `File` has no user-space buffer to flush; the bytes are in the kernel by here, which is
    // all the rename needs (see the fsync paragraph above).
    drop(f);
    std::fs::rename(&part, path).with_context(|| format!("rename {part} -> {path}"))?;
    Ok((VQ_ALIGN + blocks * stride) as u64)
}

#[cfg(test)]
mod tests {
    // The reference arm is the whole-layer buffer this replaced, spelled out rather than
    // derived from the thing under test — that independence is what makes the byte-identity
    // claim mean something. Crate-wide `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::super::fixtures::tmpdir;
    use super::super::header::{ExpertHeader, F4_MAGIC, LayerDims};
    use super::*;

    /// Every `*.part` currently in `dir`, as bare file names.
    fn walk_parts(dir: &str) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".part"))
            .collect();
        v.sort();
        v
    }

    /// Streaming a layer in windows produces the same file the whole-layer buffer did, and
    /// keeps doing so when the block count is not a multiple of the window.
    ///
    /// The reference arm here is the code both converters used to run inline — allocate
    /// `VQ_ALIGN + blocks * stride`, stamp the header, `fill_expert_blocks` over the rest,
    /// write once. That is what G1a's byte-identity claim is against, so it is spelled out
    /// rather than derived from the thing under test.
    ///
    /// `stride > bytes` and every payload non-zero, so a block landing in the wrong slot moves
    /// bytes rather than merely repeating them, and any padding that got written shows up.
    ///
    /// `blocks = 7` against `per = 2` deliberately leaves a short final window; an off-by-one
    /// there writes 8 blocks or 6, and the length assertion catches it.
    #[test]
    fn a_windowed_expert_layer_is_byte_identical_to_the_buffered_one() {
        let dir = tmpdir("expert_layer");
        let (stride, bytes, blocks) = (64usize, 40usize, 7usize);
        let header = ExpertHeader::new(
            F4_MAGIC,
            LayerDims {
                layer: 3,
                n_experts: blocks,
                expert_in: 128,
                moe_inter: 64,
                stride,
            },
        )
        .to_bytes();
        // Distinct non-zero payload per expert, and the LAST byte of each differs, so a
        // block written into the wrong slot moves bytes rather than merely repeating them.
        let fill = |e: usize, slot: &mut [u8]| -> Result<()> {
            slot.fill(0xA0 | (e as u8 & 0x0f));
            slot[bytes - 1] = 0xE0 | (e as u8 & 0x0f);
            Ok(())
        };

        // Reference: the whole-layer buffer, as both converters wrote it before 2026-08-10.
        let mut buf = vec![0u8; VQ_ALIGN + blocks * stride];
        buf[..EXPERT_HEADER_BYTES].copy_from_slice(&header);
        crate::quant::fill_expert_blocks(&mut buf[VQ_ALIGN..], stride, bytes, blocks, fill)
            .unwrap();

        // One window per two blocks: four windows, the last holding one block.
        let path = format!("{dir}/L03.f4");
        let n =
            write_expert_layer(&path, &header, stride, bytes, blocks, 2 * stride, fill).unwrap();

        // A zero stride is REFUSED, not clamped. It used to be `stride.max(1)`, which turned a
        // caller's bad geometry into a panic inside `chunks_exact_mut(0)` two functions away.
        let zero = format!("{dir}/L04.f4");
        assert!(
            write_expert_layer(&zero, &header, 0, 0, blocks, 2 * stride, fill).is_err(),
            "stride 0 accepted"
        );
        assert!(!std::path::Path::new(&zero).exists(), "refused but created");

        // Another process's debris is NOT adopted. This arm moved here from `write_atomic`'s
        // test when that function was deleted: the pid-suffixed `.part` is the defence against
        // two concurrent converts into one `out_dir`, whose interleaved writes OF EQUAL LENGTH
        // yield a file of exactly the right length — the one corruption shape `open_routed`'s
        // length check cannot see. Seeded longer than the payload so adopting it would fail on
        // length rather than on content, and left in place afterwards to prove it was untouched.
        let foreign = format!("{path}.999999.part");
        std::fs::write(&foreign, vec![0xFDu8; buf.len() + 64]).unwrap();
        let again = format!("{dir}/L05.f4");
        write_expert_layer(&again, &header, stride, bytes, blocks, 2 * stride, fill).unwrap();
        assert_eq!(
            std::fs::read(&again).unwrap().len(),
            buf.len(),
            "adopted it"
        );
        assert_eq!(
            std::fs::read(&foreign).unwrap().len(),
            buf.len() + 64,
            "consumed another process's .part"
        );
        std::fs::remove_file(&foreign).unwrap();

        let got = std::fs::read(&path).unwrap();

        assert_eq!(n as usize, buf.len(), "reported length");
        assert_eq!(got.len(), buf.len(), "file length");
        assert_eq!(
            got,
            buf,
            "windowed output differs from the buffered form at byte {:?}",
            got.iter().zip(&buf).position(|(a, b)| a != b)
        );
        // Independent of the reference arm: every block's padding is zero. Stated separately
        // because if BOTH arms grew the same padding bug the comparison above would pass.
        for e in 0..blocks {
            let pad = &got[VQ_ALIGN + e * stride + bytes..VQ_ALIGN + (e + 1) * stride];
            assert!(pad.iter().all(|&b| b == 0), "block {e} padding not zero");
        }
        assert_eq!(walk_parts(&dir), Vec::<String>::new(), "left a temp file");
    }
}
