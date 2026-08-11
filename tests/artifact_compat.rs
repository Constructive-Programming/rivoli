//! **The artifacts this machine already holds must still open, byte for byte.**
//!
//! G1a's second and fourth bullets (`docs/investigations/k3-port.md`). Adding a third model to this
//! engine meant renaming the expert-geometry parameter (`hidden` → `expert_in`, 2026-08-09),
//! re-typing `F4Expert` around an `F4Naming`, and factoring both converters' layer loop into
//! `RoutedRepack` — and none of that is allowed to move a byte or an offset in artifacts that cost
//! **805 GiB and many hours** to produce and that nobody wants to rebuild to find out.
//!
//! Worth its own file rather than a case in `f4_loading.rs`: that suite's shipped fixtures are the
//! 12 GiB three-layer sets and its questions are about the loader's arithmetic. These two ask
//! whether the FULL artifact still opens, and they cover **GLM as well as V4**, which no test did
//! before. GLM is the more informative half — `.vq3` and `.i4` are the formats WITH a shared block
//! and `.i4` has no header at all, so between them they exercise every branch of
//! `RoutedFmt::{hbytes, has_shared}` that `.f4` leaves untested.
//!
//! **What this actually reads**, corrected 2026-08-11 after review called the earlier claim
//! ("nothing here reads a weight … the largest read is one 4 KiB header") false on both counts. It
//! stats the length of each of 195 layer files — twice each, once in `open_routed` and once in
//! `check_set`, so ~390 calls — and asks `read_spec` for offsets, no weight in either. But
//! `set.shared_block(l)` does a full-stride O_DIRECT read, so the GLM half pulls
//! `2·15,335,424 + 2·20,054,016 = 70,778,880 B` (67.5 MiB) of real expert bytes, uncached. The
//! header read is 40 bytes (`EXPERT_HEADER_BYTES`), not 4 KiB. Still cheap and still no GPU; the
//! numbers matter because this is the paragraph someone will cite when deciding to run it in a loop.
//!
//! Absent artifacts SKIP; an explicitly-pointed env var that does not resolve FAILS — the rule
//! `common/f4_artifact_dir.rs` states, for the reason it gives: libtest hides stderr on a passing
//! test, so a skip that looked like a pass would be invisible.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::artifact::format::{ExpertSet, RoutedFmt, SetDims, f4_layer_range};
use rivoli::artifact::model::{ModelConfig, V4Config};
// The geometry module by alias, not by item imports. `f4_loading.rs` needs the same four names and
// imports them item-by-item; the two import lines were then character-identical and jscpd counted
// the preamble as a clone. `geom::` also reads better in a file that puts all three formats' strides
// side by side, which is the only place in the tree that does.
use rivoli::artifact::quant as geom;

#[path = "common/f4_artifact_dir.rs"]
mod f4_artifact_dir;

/// This format's per-expert stride, re-derived from the public geometry functions.
///
/// `RoutedFmt::geometry` — what `open_routed` uses — is private, and that is convenient rather than
/// awkward: the test computes the stride a second way, so a change that moved a stride would have
/// to be made in both places to stay green.
fn stride_of(fmt: RoutedFmt, expert_in: usize, moe_inter: usize) -> usize {
    match fmt {
        RoutedFmt::Vq3 => geom::vq_expert_stride(expert_in, moe_inter),
        RoutedFmt::I4 => geom::i4_expert_stride(expert_in, moe_inter),
        RoutedFmt::F4 => geom::f4_expert_stride(expert_in, moe_inter),
    }
}

/// `(header bytes, shared blocks)` for a format — the two facts that differ between the three, and
/// the two `RoutedFmt` predicates these tests exist to pin.
///
/// `.f4`: 4 KiB header, no shared block. `.vq3`: 4 KiB header, one shared block. `.i4`: **no
/// header**, one shared block. Stated as data so a format that changed category has to change this
/// line to stay green, rather than three copies of an `if`.
fn shape_of(fmt: RoutedFmt) -> (usize, usize) {
    match fmt {
        RoutedFmt::F4 => (geom::VQ_ALIGN, 0),
        RoutedFmt::Vq3 => (geom::VQ_ALIGN, 1),
        RoutedFmt::I4 => (0, 1),
    }
}

/// Open one routed set, confront every layer file with the geometry and the ends with the block
/// layout, and return the bytes accounted for.
///
/// That return value is G1a's fourth bullet: the artifact's size **derived from the config**, not
/// read off `du`.
fn check_set(
    dir: &str,
    fmt: RoutedFmt,
    layers: std::ops::Range<usize>,
    cfg: (usize, usize, usize),
) -> usize {
    let (n_experts, expert_in, moe_inter) = cfg;
    let stride = stride_of(fmt, expert_in, moe_inter);
    let (header, shared) = shape_of(fmt);
    let want = header + (n_experts + shared) * stride;
    let set = ExpertSet::open_routed(
        dir,
        fmt,
        SetDims::new(layers.clone(), n_experts, expert_in, moe_inter),
    )
    .unwrap_or_else(|e| panic!("{dir}: .{} set failed to open: {e:#}", fmt.ext()));

    let mut total = 0;
    for l in layers.clone() {
        let path = format!("{dir}/L{l:02}.{}", fmt.ext());
        let len = std::fs::metadata(&path)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .len() as usize;
        assert_eq!(
            len, want,
            "{path}: {len} bytes on disk; {n_experts} experts + {shared} shared at stride \
             {stride}, after a {header}-byte header, is {want}"
        );
        total += len;
    }
    // Offsets on the FIRST and LAST layer only: `read_spec` derives them from `(layer, expert)`
    // with no per-layer state, so the layers in between would restate the same arithmetic. The ends
    // are what catch an off-by-one in `first_layer`.
    for l in [layers.start, layers.end - 1] {
        for e in [0, n_experts - 1] {
            let (_, begin, useful) = set.read_spec(l, e).unwrap();
            assert_eq!(
                begin,
                header + e * stride,
                "{dir} L{l:02} expert {e} offset"
            );
            // Not an EOF check — `begin` is pinned above, so this reduces to
            // `expert_bytes <= stride`, i.e. that the geometry pads UP to `VQ_ALIGN`. It binds the
            // two functions to each other, which is the only thing left to bind here.
            assert!(
                begin + useful <= want,
                "{dir} L{l:02} expert {e}: {useful} useful bytes do not fit the {stride} stride"
            );
        }
        // The `has_shared` fork, asserted rather than assumed: block `n_experts` is the shared
        // expert on `.vq3`/`.i4` and does not exist on `.f4`.
        assert_eq!(
            set.shared_block(l).is_ok(),
            shared == 1,
            "{dir} L{l:02}: shared-block availability must follow the format"
        );
    }
    total
}

/// The full V4 artifact: 43 layers of `.f4`, header + 256 blocks, no shared block.
#[test]
fn the_full_v4_artifact_still_opens_byte_and_offset_identically() {
    let Some(dir) = f4_artifact_dir::v4_artifact_full("L00.f4") else {
        return;
    };
    let cfg: V4Config = rivoli::artifact::model::load_config(&dir).unwrap();
    let range = f4_layer_range(&dir, cfg.n_layers).unwrap();
    // The WHOLE model, unlike the l0-2/l3-5 fixtures — which is also why this artifact could not
    // catch a `first_layer` bug and those two still exist.
    assert_eq!(
        range,
        0..cfg.n_layers,
        "the full artifact holds every layer"
    );

    let routed = check_set(
        &dir,
        RoutedFmt::F4,
        range,
        (cfg.n_experts, cfg.hidden, cfg.moe_inter),
    );
    // Byte accounting, derived and confronted with the disk: 43 x (4096 + 256 x 13,369,344).
    // Measured 2026-08-11 — 147,169,914,880 B of `.f4` plus a 9,557,453,182 B resident set is
    // **156,727,368,062 B**; the whole directory is 156,733,738,580, the extra 6,370,518 being
    // `tokenizer.json` and three small json files. Either way 145.97 GiB, which is the "~146 GiB"
    // the plan carries. (The two-term sum was written as the four-term total until review; it
    // rounded the same, which is how it hid.) Correct — and I nearly reported the plan wrong by
    // summing only the `.f4` files (137.06 GiB) and reading the difference as a GB/GiB mislabel.
    assert_eq!(routed, 147_169_914_880, "total .f4 bytes");
    let resident = std::fs::metadata(format!("{dir}/resident.safetensors"))
        .unwrap()
        .len() as usize;
    let gib = (routed + resident) as f64 / 1024f64.powi(3);
    assert!(
        (145.0..147.0).contains(&gib),
        "V4 artifact is {gib:.2} GiB, expected ~146 — routed {routed} + resident {resident}"
    );
}

/// The full GLM artifact: 76 layers, `.vq3` AND `.i4`, both with a shared block, `.i4` headerless.
///
/// The `.i4` half is what would catch a header appearing where there is none — its length is an
/// exact multiple of the stride, so a 4 KiB header would put every expert offset in the artifact
/// out by 4096, and the length check is what says so.
#[test]
fn the_full_glm_artifact_still_opens_byte_and_offset_identically() {
    let Some(dir) = f4_artifact_dir::glm_artifact_full("L03.vq3") else {
        return;
    };
    let cfg = ModelConfig::load(&dir).unwrap();
    // The range the ENGINE uses, taken from `pin.rs` rather than invented here: the dense prefix is
    // skipped, and the MTP head is one layer PAST `n_layers` and carries experts like any other. On
    // this artifact that is 3..79 — 76 files, which is what is on disk.
    // `L{:02}`, the one naming rule this file and `open_routed` both depend on. It was `L{}` until
    // review: harmless at 78, but for any `n_layers < 10` it probes `L3.vq3`, never finds `L03.vq3`,
    // silently concludes there is no MTP head, and then fails on the layer count instead.
    let mtp = std::fs::metadata(format!("{dir}/L{:02}.vq3", cfg.n_layers)).is_ok();
    let layers = cfg.dense_layers..(cfg.n_layers + usize::from(mtp));
    // A property of the CONFIG, not of the disk — GLM artifacts carry no range to read, unlike the
    // V4 half's `f4_source`, so the honest fix is the diagnosis in the message below.
    assert_eq!(
        layers.len(),
        76,
        "the config implies 76 layer files ({}..{}); if this artifact holds a narrower range the \
         failure below will be a missing file, not a wrong length",
        layers.start,
        layers.end
    );

    let dims = (cfg.n_experts, cfg.hidden, cfg.moe_inter);
    let vq = check_set(&dir, RoutedFmt::Vq3, layers.clone(), dims);
    let i4 = check_set(&dir, RoutedFmt::I4, layers, dims);

    // Byte accounting, both halves. Measured 2026-08-11 from this artifact: 299,531,812,864
    // (`.vq3`) + 391,695,040,512 (`.i4`) + 16,618,233,984 (two resident files) + 20,245,356 (json)
    // + 196,608 (codebooks) = 707,865,529,324 B = **659.25 GiB**.
    //
    // **The plan said 675 GiB, which is 15.75 GiB high.** No directory on this machine measures
    // 675, and this artifact is complete — layers 3..78 contiguous, the same set in both formats.
    // Corrected in `k3-port.md` and `other-models.md`; this is what will notice if it drifts again.
    // No GiB range check here, unlike the V4 half: `vq + i4` is a function of two values pinned to
    // exact literals on the lines above, so a range over it is entailed and could never fire. V4's
    // twin is not dead — it is the only bound on `resident`, which nothing else constrains. That
    // asymmetry is the tell, and review found it.
    assert_eq!(vq, 299_531_812_864, "total .vq3 bytes");
    assert_eq!(i4, 391_695_040_512, "total .i4 bytes");
}
