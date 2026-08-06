//! The `.f4` loading path: what `ExpertSet` must refuse, and what it must not.
//!
//! **Every refusal here is proved by injecting the defect it names and watching it fire.**
//! That is not ceremony in this port — `docs/investigations/v4-flash-port.md` records a
//! tautological anti-vacuity assertion shipped twice, and a `head_dim == qk_rope_head_dim`
//! guard that admitted everything because `(512-512).is_multiple_of(64)` is true. So each
//! case below pairs the break with the intact control, and asserts on the error TEXT: a
//! `.f4` opened as `.vq3` fails several ways at once, and "it errored" would not say the
//! guard under test was the one that spoke.
//!
//! The central fact, from the shipped artifact rather than from the design:
//! `L00.f4` is `4096 + 256 × 13369344 = 3422556160` bytes **exactly**, so an `.f4` has room
//! for `n_experts` blocks and not one more. `.vq3`/`.i4` carry `n_experts + 1` (the last is
//! the shared expert); V4's shared expert is fp8 e4m3 and lives in `resident.safetensors`.
//!
//! Host-side only — no GPU, no `DeviceTier`. The synthetic fixtures are toy-dimension and a
//! few KB; the real-artifact cases skip loudly when it is absent.
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::artifact::format::{
    EXPERT_HEADER_BYTES, ExpertHeader, ExpertSet, F4_MAGIC, RoutedFmt, SetDims, VQ3_MAGIC,
    f4_layer_range,
};
use rivoli::artifact::model::V4Config;
use rivoli::artifact::quant::{VQ_ALIGN, f4_expert_stride, i4_expert_stride, vq_expert_stride};

#[path = "common/v4_artifact_dir.rs"]
mod v4_artifact_dir;

/// Toy dims: the smallest that keep the three formats' strides distinct and give each
/// projection more than one FP4 group. `f4_expert_stride(64, 32)` is one `VQ_ALIGN` block.
const HIDDEN: usize = 64;
const MOE_INTER: usize = 32;
const N_EXPERTS: usize = 4;
const N_LAYERS: usize = 2;

/// A scratch directory that removes itself. `ExpertSet` addresses `L{ll}.{ext}` inside a
/// directory, so each fixture needs its own; `Drop` means a failing assertion does not leave
/// one behind for the next run to inherit and trust.
struct Scratch(std::path::PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        // The pid is not decoration: agents share this machine, and two concurrent runs
        // on a fixed path would `remove_dir_all` each other's fixtures mid-test.
        let d = std::env::temp_dir().join(format!("rivoli_v4load_{tag}_{}", std::process::id()));
        drop(std::fs::remove_dir_all(&d));
        std::fs::create_dir_all(&d).unwrap();
        Self(d)
    }

    fn dir(&self) -> String {
        self.0.to_string_lossy().into_owned()
    }

    fn len_of(&self, name: &str) -> usize {
        std::fs::metadata(self.0.join(name)).unwrap().len() as usize
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        drop(std::fs::remove_dir_all(&self.0));
    }
}

fn dims(n_layers: usize) -> SetDims {
    SetDims::new(0..n_layers, N_EXPERTS, HIDDEN, MOE_INTER)
}

/// Assert that `r` failed, and that the message names the guard under test.
///
/// On the text, not on `is_err()`: several of these fixtures are wrong in a way that more
/// than one check could catch, and a bare `is_err()` would pass while the guard being tested
/// never ran.
fn refuses<T>(r: anyhow::Result<T>, tag: &str, needle: &str) {
    let e = format!(
        "{:#}",
        r.err()
            .unwrap_or_else(|| panic!("{tag}: this must be refused"))
    );
    assert!(e.contains(needle), "{tag}: expected {needle:?}, got: {e}");
}

/// How a synthetic layer file departs from a correct one. Named fields rather than raw
/// numbers so each case says which single property it perturbed.
#[derive(Clone, Copy)]
struct Defect {
    magic: [u8; 4],
    /// Blocks written after the header. A correct `.f4` is `N_EXPERTS`.
    blocks: usize,
    /// Added to the header's `layer` field.
    layer_skew: usize,
    /// Swap the header's `hidden`/`moe_inter` — a file of exactly the same length.
    transpose: bool,
    /// Written into the header's `stride`, which is the value the WRITER indexed blocks
    /// with. `None` means "the stride this build derives", i.e. agreement.
    stride: Option<usize>,
}

impl Defect {
    /// A correct `.f4`. Every case starts here and perturbs one field, so the control and
    /// the break differ in exactly one thing.
    const OK: Self = Self {
        magic: F4_MAGIC,
        blocks: N_EXPERTS,
        layer_skew: 0,
        transpose: false,
        stride: None,
    };
}

/// Write `n_layers` synthetic `.f4` files. Bodies are zeros — the reader validates the
/// header and the LENGTH and never looks at a block here.
///
/// A header-only defect is checked to leave the file at the length the reader ACCEPTS —
/// against `N_EXPERTS`, not against `d.blocks`, or the assertion would restate the
/// expression that sized the buffer and could never fail. That distinction is the whole
/// point: without it, `transposed` and `layer_skew` could quietly be length tests.
fn write_f4_set(tag: &str, n_layers: usize, d: Defect) -> Scratch {
    let s = Scratch::new(tag);
    let stride = f4_expert_stride(HIDDEN, MOE_INTER);
    let (hidden, moe_inter) = if d.transpose {
        (MOE_INTER, HIDDEN)
    } else {
        (HIDDEN, MOE_INTER)
    };
    for l in 0..n_layers {
        // The header's stride can disagree with the one the file is actually laid out at —
        // that is the whole reason `ExpertHeader::new` takes it rather than re-deriving it.
        let h = ExpertHeader::new(
            d.magic,
            l + d.layer_skew,
            N_EXPERTS,
            hidden,
            moe_inter,
            d.stride.unwrap_or(stride),
        );
        let mut buf = vec![0u8; VQ_ALIGN + d.blocks * stride];
        buf[..EXPERT_HEADER_BYTES].copy_from_slice(&h.to_bytes());
        std::fs::write(s.0.join(format!("L{l:02}.f4")), &buf).unwrap();
    }
    if d.blocks == Defect::OK.blocks {
        assert_eq!(
            s.len_of("L00.f4"),
            VQ_ALIGN + N_EXPERTS * stride,
            "{tag}: a header-only defect must leave the length the reader accepts"
        );
    }
    s
}

fn open_f4(s: &Scratch, d: SetDims) -> anyhow::Result<ExpertSet> {
    ExpertSet::open_routed(&s.dir(), RoutedFmt::F4, d)
}

/// Build a defective `.f4` set, open it, and require the named guard to speak.
fn f4_refuses(tag: &str, n_layers: usize, d: Defect, needle: &str) {
    let s = write_f4_set(tag, n_layers, d);
    refuses(open_f4(&s, dims(n_layers)), tag, needle);
}

/// The control. A well-formed `.f4` opens, and its blocks address exactly the file — no room
/// for a shared one. Without this every refusal below could be a reader that rejects
/// everything.
#[test]
fn a_well_formed_f4_opens_and_its_last_expert_ends_at_eof() {
    let s = write_f4_set("ok", N_LAYERS, Defect::OK);
    let set = open_f4(&s, dims(N_LAYERS)).unwrap();
    let stride = f4_expert_stride(HIDDEN, MOE_INTER);
    let len = s.len_of("L00.f4");

    for e in 0..N_EXPERTS {
        let (_, begin, useful) = set.read_spec(0, e).unwrap();
        assert_eq!(begin, VQ_ALIGN + e * stride, "expert {e} block start");
        assert!(begin + useful <= len, "expert {e} reads past EOF");
    }
    // The last ROUTED expert's block ends the file. Note what this does and does not
    // prove: `open_routed` already enforced `len == hbytes + n_experts*stride` using the
    // same `has_shared`, so against a fixture THIS function wrote it largely restates the
    // fixture. It pins `read_spec`'s offsets against that length; the ground-truth version
    // is `the_shipped_f4_artifact_opens_at_its_own_layer_range`, on a file rivoli did not
    // write in this process.
    let (_, last, _) = set.read_spec(0, N_EXPERTS - 1).unwrap();
    assert_eq!(
        last + stride,
        len,
        "an .f4 must end at the last routed expert — a shared block would need {stride} \
         more bytes than the file has"
    );
    // Out-of-range expert and layer are still refused.
    assert!(set.read_spec(0, N_EXPERTS).is_err());
    assert!(set.read_spec(N_LAYERS, 0).is_err());
}

/// **The `n_experts + 1` the old reader hard-coded**, and its mirror. A file carrying the
/// extra shared block is refused as `.f4`; so is one a block short.
#[test]
fn f4_refuses_a_file_sized_for_a_shared_block() {
    let stride = f4_expert_stride(HIDDEN, MOE_INTER);
    let want = VQ_ALIGN + N_EXPERTS * stride;
    for (tag, blocks) in [
        ("blocks_plus_one", N_EXPERTS + 1),
        ("blocks_minus_one", N_EXPERTS - 1),
    ] {
        let got = VQ_ALIGN + blocks * stride;
        f4_refuses(
            tag,
            1,
            Defect {
                blocks,
                ..Defect::OK
            },
            &format!("{got} bytes, expected {want}"),
        );
    }
}

/// The magic is the discriminant the length check cannot be, so it is tested where length is
/// NOT a tell: the file is exactly the right length for the format it is opened as, and
/// carries the other format's magic.
#[test]
fn magic_separates_the_formats_when_the_length_cannot() {
    f4_refuses(
        "f4_len_vq3_magic",
        1,
        Defect {
            magic: VQ3_MAGIC,
            ..Defect::OK
        },
        "expected .f4 magic",
    );

    // The mirror: a `.vq3`-length file (n_experts + 1 blocks at the vq3 stride) carrying FP4
    // magic, opened as `.vq3`. Proves the check is per-format and not "reject anything that
    // is not VQ3".
    let s = Scratch::new("vq3_len_f4_magic");
    let stride = vq_expert_stride(HIDDEN, MOE_INTER);
    let mut buf = vec![0u8; VQ_ALIGN + (N_EXPERTS + 1) * stride];
    let hdr = |m| ExpertHeader::new(m, 0, N_EXPERTS, HIDDEN, MOE_INTER, stride).to_bytes();
    buf[..EXPERT_HEADER_BYTES].copy_from_slice(&hdr(F4_MAGIC));
    let path = s.0.join("L00.vq3");
    std::fs::write(&path, &buf).unwrap();
    refuses(
        ExpertSet::open_routed(&s.dir(), RoutedFmt::Vq3, dims(1)),
        "vq3_len_f4_magic",
        "expected .vq3 magic",
    );
    // Bidirectional control: the SAME bytes with the right magic open. So the refusal above
    // is the magic, and not the length, the stride or the dims.
    buf[..EXPERT_HEADER_BYTES].copy_from_slice(&hdr(VQ3_MAGIC));
    std::fs::write(&path, &buf).unwrap();
    ExpertSet::open_routed(&s.dir(), RoutedFmt::Vq3, dims(1))
        .expect("the same bytes with the right magic must open");
}

/// The header's dims and layer id. A transposed `(hidden, moe_inter)` gives a file of exactly
/// the same length — the reason the header carries dims at all — and a shuffled layer id one
/// that is byte-identical apart from four. `write_f4_set` asserts the length is unchanged, so
/// neither case can degrade into the length check.
#[test]
fn f4_header_dims_and_layer_id_are_confronted_with_the_config() {
    // Needles name the FIELD, not just "header dims": that ensure! spans layer, n_experts,
    // hidden and moe_inter, so a substring matching all four would not say which fired.
    f4_refuses(
        "transposed",
        N_LAYERS,
        Defect {
            transpose: true,
            ..Defect::OK
        },
        &format!("hidden {MOE_INTER} moe_inter {HIDDEN}"),
    );
    f4_refuses(
        "layer_skew",
        N_LAYERS,
        Defect {
            layer_skew: 1,
            ..Defect::OK
        },
        "(layer 1 experts",
    );
    // The header's `stride` — the one field that is not re-derivable from the config, and
    // the one the converter writes from the value it INDEXED BLOCKS WITH. A file whose
    // header claims a different stride is exactly the right total length, so only this
    // conjunct can see it.
    f4_refuses(
        "stride_skew",
        1,
        Defect {
            stride: Some(f4_expert_stride(HIDDEN, MOE_INTER) + VQ_ALIGN),
            ..Defect::OK
        },
        &format!("stride {}", f4_expert_stride(HIDDEN, MOE_INTER)),
    );
}

/// An inverted layer range. `open_routed` computes `n_layers - first_layer` for a
/// `Vec::with_capacity`, so before the guard this was a `usize` underflow panic naming
/// nothing — and `open_vq3` takes the two bounds as loose arguments.
#[test]
fn an_inverted_layer_range_is_refused_rather_than_underflowing() {
    let s = write_f4_set("inverted", 2, Defect::OK);
    // Built from bindings, not the literal `2..1`: clippy's `reversed_empty_ranges` fires
    // on the literal, and this range being reversed is the fixture, not a mistake.
    let (from, to) = (2, 1);
    refuses(
        open_f4(&s, SetDims::new(from..to, N_EXPERTS, HIDDEN, MOE_INTER)),
        "inverted",
        "is inverted",
    );
    // The control: the same set over a well-ordered range opens.
    open_f4(&s, dims(2)).expect("0..2 must open");
}

/// `.i4` stays headerless and keeps its shared block: the `.f4` work must not have moved
/// either. `.i4` is the case with no magic at all, so a reader that started demanding one
/// everywhere would show up here.
#[test]
fn i4_stays_headerless_and_keeps_its_shared_block() {
    let s = Scratch::new("i4_shared");
    let stride = i4_expert_stride(HIDDEN, MOE_INTER);
    // Headerless, n_experts + 1 blocks from offset 0, and not a valid header at byte 0.
    std::fs::write(s.0.join("L00.i4"), vec![0xABu8; (N_EXPERTS + 1) * stride]).unwrap();
    let set = ExpertSet::open_routed(&s.dir(), RoutedFmt::I4, dims(1)).unwrap();
    let (_, begin, _) = set.read_spec(0, 0).unwrap();
    assert_eq!(begin, 0, ".i4 block 0 starts at offset 0 — no header block");
    let shared = set.shared_block(0).expect(".i4 has a shared block");
    assert!(
        shared.iter().all(|&b| b == 0xAB),
        "the shared block must be read from block n_experts"
    );
}

/// An `.f4` has no shared expert, and that is the FORMAT's property — not the accident that
/// the file happens to stop there. Proved by the message: a reader relying on EOF would
/// report an I/O error, and would hand back garbage from any `.f4` with trailing bytes.
#[test]
fn f4_refuses_a_shared_block_by_format_not_by_eof() {
    let s = write_f4_set("no_shared", 1, Defect::OK);
    let set = open_f4(&s, dims(1)).unwrap();
    refuses(set.shared_block(0), "no_shared", "no shared block");
    refuses(set.shared_block(0), "no_shared", "resident.safetensors");
}

// ── the artifact's own layer range ──────────────────────────────────────────────

/// `f4_source.layers` is the loader's only source for which layers exist, so absent and
/// malformed must both be errors rather than a `0..num_hidden_layers` guess — precisely the
/// wrong guess for the partial artifact the field exists to describe.
#[test]
fn f4_source_range_is_read_and_confronted_with_the_layer_count() {
    let s = Scratch::new("f4src");
    let mf = s.0.join("manifest.json");
    let write = |body: &str| std::fs::write(&mf, body).unwrap();

    // A 3-layer artifact of a 43-layer model is normal, must load, and must NOT be widened
    // to the model's own count by anything.
    write(r#"{"f4_source":{"tool":"convert_v4","layers":[0,3]}}"#);
    assert_eq!(f4_layer_range(&s.dir(), 43).unwrap(), 0..3);

    for (tag, body, needle) in [
        ("absent", r#"{"num_hidden_layers":43}"#, "no `f4_source`"),
        (
            "no_layers",
            r#"{"f4_source":{"tool":"convert_v4"}}"#,
            "malformed",
        ),
        (
            "not_a_pair",
            r#"{"f4_source":{"layers":[0,1,2]}}"#,
            "malformed",
        ),
        // Ranges describing more than the model has, nothing at all, or backwards.
        (
            "past_the_end",
            r#"{"f4_source":{"layers":[0,44]}}"#,
            "not a non-empty range",
        ),
        (
            "empty",
            r#"{"f4_source":{"layers":[3,3]}}"#,
            "not a non-empty range",
        ),
        (
            "backwards",
            r#"{"f4_source":{"layers":[5,2]}}"#,
            "not a non-empty range",
        ),
    ] {
        write(body);
        refuses(f4_layer_range(&s.dir(), 43), tag, needle);
    }
}

// ── against the shipped artifact ────────────────────────────────────────────────

/// Open a shipped artifact's `.f4` set at the range its own manifest declares.
fn open_shipped(dir: &str) -> (V4Config, std::ops::Range<usize>, ExpertSet) {
    let cfg: V4Config = rivoli::artifact::model::load_config(dir).unwrap();
    let range = f4_layer_range(dir, cfg.n_layers).unwrap();
    let d = SetDims::new(range.clone(), cfg.n_experts, cfg.hidden, cfg.moe_inter);
    let set = ExpertSet::open_routed(dir, RoutedFmt::F4, d).unwrap();
    (cfg, range, set)
}

/// The synthetic fixtures above are toy-dimension. This is the arithmetic that actually
/// broke, at the real 256 × 13369344, read from the file `convert_v4 --verify` produced.
#[test]
fn the_shipped_f4_artifact_opens_at_its_own_layer_range() {
    let Some(dir) = v4_artifact_dir::v4_artifact("L00.f4") else {
        return;
    };
    let (cfg, range, set) = open_shipped(&dir);
    assert_eq!(range.start, 0, "this fixture is the starts-at-zero case");
    assert!(
        range.end < cfg.n_layers,
        "this fixture is deliberately PARTIAL — if it ever covers all {} layers it has \
         stopped exercising the `num_hidden_layers` != artifact-range case",
        cfg.n_layers
    );

    let stride = f4_expert_stride(cfg.hidden, cfg.moe_inter);
    let len = std::fs::metadata(format!("{dir}/L00.f4")).unwrap().len() as usize;
    assert_eq!(
        len,
        VQ_ALIGN + cfg.n_experts * stride,
        "the shipped L00.f4 must be header + n_experts blocks and nothing more"
    );
    // At the REAL stride, not the toy one: this is where the `+ 1` cost 13,369,344 bytes.
    let (_, last, _) = set.read_spec(0, cfg.n_experts - 1).unwrap();
    assert_eq!(last + stride, len);
    // Nothing else is asserted here. `shared_block(0).is_err()` and a `.vq3` open of this
    // directory both pass for reasons that are not the guard — an EOF and a missing file
    // respectively — so they are proved in the synthetic cases, where the message can be
    // checked.
}

/// The layers-3-5 artifact: a `.f4` set whose range does **not** start at 0.
///
/// `l0-2` cannot test this at all — `layer - first_layer` is the identity there, so an
/// `ExpertSet` that ignored `first_layer` entirely would pass every other case in this
/// file. Here layer 3 is `files[0]`, and asking for layer 0 must be refused rather than
/// silently returning layer 3's descriptor.
#[test]
fn an_f4_set_that_does_not_start_at_layer_zero_is_addressed_by_absolute_id() {
    let Some(dir) = v4_artifact_dir::v4_artifact_l3_5("L03.f4") else {
        return;
    };
    let (_cfg, range, set) = open_shipped(&dir);
    assert_eq!(range, 3..6, "this fixture is the non-zero-start case");

    // **Each layer's descriptor must point at that layer's FILE.** Resolved through
    // `/proc/self/fd`, because every weaker phrasing is satisfiable by a wrong mapping that
    // happens to coincide on this fixture: `layer % files.len()` equals `layer -
    // first_layer` for exactly 3..6 over 3 files, and an injected version of it passed a
    // distinct-fds check and an offset check. The filename cannot coincide.
    for l in range.clone() {
        let (fd, begin, _) = set.read_spec(l, 0).unwrap();
        let path = std::fs::read_link(format!("/proc/self/fd/{fd}")).unwrap();
        assert!(
            path.ends_with(format!("L{l:02}.f4")),
            "layer {l} resolved to {path:?}"
        );
        // Expert 0 sits at the header offset in EVERY layer file — the block offset is a
        // function of the expert, not the layer, so this is what would alias if the layer
        // mapping were the only thing distinguishing them.
        assert_eq!(begin, VQ_ALIGN, "layer {l} expert 0");
    }
    // Below and above the range are refused, not wrapped.
    for l in [0, 1, 2, 6] {
        assert!(set.read_spec(l, 0).is_err(), "layer {l} is not in [3, 6)");
    }
    assert!(
        set.shared_block(3).is_err(),
        "still an .f4: no shared block"
    );
}

/// **The set answers for its own layout, so nothing can be paired with the wrong one.**
///
/// `TierFmt::new` used to be handed `(fmt, off, layers, n_experts)` and check that the
/// offsets fitted the stride. That check could not fire: every routed block is padded to
/// `VQ_ALIGN`, so another format's shorter layout sits inside it, and `.f4`/`.i4` tile
/// identically at every real dimension anyway. The set now derives all four, and this is
/// what pins the derivation — including that it is the `.f4` layout and not `.i4`'s, which at
/// these toy dims are distinguishable and at V4's are not (the two collide for 25% of all
/// `i_dim`; `quant::f4_slot_offsets` has the identity).
#[test]
fn an_f4_set_reports_the_format_range_and_slot_layout_it_was_opened_with() {
    use rivoli::artifact::quant::{f4_slot_offsets, i4_slot_offsets};
    let s = write_f4_set("selfdesc", N_LAYERS, Defect::OK);
    let set = open_f4(&s, dims(N_LAYERS)).unwrap();
    assert_eq!(set.fmt(), RoutedFmt::F4);
    assert_eq!(set.layers(), 0..N_LAYERS);
    assert_eq!(set.n_experts(), N_EXPERTS);
    assert_eq!(set.slot_offsets(), f4_slot_offsets(HIDDEN, MOE_INTER));
    assert_ne!(
        set.slot_offsets(),
        i4_slot_offsets(HIDDEN, MOE_INTER),
        "at (64, 32) the two nibble layouts differ — inside the collision band (which \
         includes 4096/2048) they do NOT, which is why this is derived rather than passed in"
    );
    // NOT asserted here: "every offset is inside the block". Both sides would come from
    // `f4_slot_offsets`/`f4_expert_bytes` over the same dims, so it cannot fail for any
    // input — it is the walk agreeing with itself. `quant`'s
    // `every_routed_format_places_each_projection_at_the_sum_of_the_ones_before_it` confronts
    // the walk with the independent `*_proj_bytes` formulas instead, which can.
}
