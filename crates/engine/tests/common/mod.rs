//! The device-free half of the kernel-oracle scaffolding — the GENERIC slice of the old
//! tree's tests/common/mod.rs (lines 1-737 there). The V4-checkpoint helpers that shared
//! the file (Checkpoint/Oracle wiring, `/var/db/rivoli/...` paths) arrive with M8, their
//! first consumer in this tree.
//!
//! Originally shared by ten test binaries:
//! `docs`, `invariants`, `kernel`, `f4_attn`, `kvcompress_kernel`, `kvcompress_probe`,
//! `headtail`, `blockindex_kernel`, `f4_kernel` and `v4_oracle`.
//!
//! It was copy-pasted per file until 2026-07-30, and the copies had already started to
//! drift: two spellings of the same `Lcg` bug note, two `assert_close` bodies with the
//! same tolerance, and `f16b`/`u16b` present in one file and re-derived in the other.
//!
//! **A helper that touches a device TYPE is `#[cfg(feature = "rocm")]`-gated here, not
//! banished to the file that owns it.**
//!
//! > **CORRECTED 2026-08-06, reconciling two same-day corrections that disagreed.** The old
//! > rule — device types stay in the owning test file — was argued from the two backends:
//! > `dev` was `DeviceBuf` under HIP and `Buf` under Vulkan, "and that difference is the
//! > point of having two files". Vulkan was retired 2026-08-06, so that argument names
//! > something that no longer exists.
//! >
//! > Two agents corrected this independently within hours and landed on opposite answers.
//! > One deleted the rule and moved `dev`/`zeros`/`back`/`ok` here. The other kept it,
//! > re-derived on stronger ground: this module compiles into EVERY binary listed above, and
//! > `docs` and `invariants` are GPU-free registry checks, so a device type here would put
//! > both behind a device.
//! >
//! > **The second argument is right and the first is safe, because of a fact neither stated:
//! > the moved helpers are `#[cfg(feature = "rocm")]`.** Verified — `cargo check --test docs`
//! > featureless, with them present, is 0 errors. So they can live here, and the constraint
//! > that survives is the gate, not the location. Move a device helper here ungated and you
//! > break `docs` and `invariants`, which is the failure the second agent predicted.
//!
//! **Five older oracle files still spell their own uploader** (`f4_kernel`, `f4_attn`,
//! `headtail`, `kvcompress_kernel`, `blockindex_kernel`) and survive the duplication
//! gate only because their `.expect` strings differ — a half-migration, not a design.
//!
//! `dead_code` is allowed because this module is compiled into EACH test binary and none
//! uses every helper. The alternative is per-consumer cfg gates on a test utility, which is
//! more machinery than the warning is worth.
//!
//! **SPLIT 2026-08-15** under the file-size gate: this file is the umbrella, and the six
//! submodules below hold the code, grouped by what a reader is looking for — `reference` (the
//! host side of a comparison: the draw source, the fixture draws, the formulas), `scoring` (the
//! metric, the tolerance and the line they print), `asserts` (those promoted to a panic),
//! `upload` (the byte codecs and the `rocm`-gated device buffers), `geometry` (the shape
//! bundles) and `moe` (the expert-range launch chain). Bodies and their comments moved verbatim.
//!
//! The `scoring`/`asserts` seam is the one place the grouping was decided by a MEASUREMENT
//! rather than by reading: together they score 9.68 on CodeScene's Primitive Obsession rule —
//! the `assert_*` surface is `&[f32], &[f32], &str` by design and the ratio is a whole-file
//! property — and 10.0 apiece once separated. `scoring.rs`'s header carries it.
//!
//! **The re-export is a glob per submodule, and that is the point**: every `use common::{…}`
//! in the oracle files above resolves exactly as it did, so the split is invisible from
//! outside it — a move, not a rewrite. A helper that finds a second consumer still arrives at
//! `common`; it just lands in the submodule whose argument it shares.
#![allow(dead_code)]
#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]

mod asserts;
mod geometry;
mod reference;
mod scoring;
mod upload;
mod v4;

// `#[allow(unused_imports)]` for the reason the header gives `dead_code`, one lint further on:
// this compiles into EVERY test binary and none of them names an item from every submodule —
// `headtail.rs` reaches `scoring`, `asserts` and `upload` only, and rustc reported
// `reference::*` as an unused import there the first time this file was a glob umbrella
// (measured 2026-08-15; `-D warnings` makes that a build failure, not a warning).
//
// ONE grouped `use` rather than an attribute per line, and the grouping is what keeps the
// suppression honest: the allow covers the re-export block and nothing else, so a genuinely
// dead `use` inside a submodule still reports. A file-level `#![allow(unused_imports)]` here
// would propagate down the module tree into all six files and suppress exactly that — the
// same argument the `DeviceBuf` re-export in `upload.rs` makes about attributing ONE item.
#[allow(unused_imports)]
pub use {asserts::*, geometry::*, reference::*, scoring::*, upload::*, v4::*};

#[cfg(feature = "rocm")]
pub mod moe;

// (`walk` stayed with its consumers — the cli crate's registry meta-gates own the one
// copy; the kernel-oracle scaffolding here never walks the tree.)

// ---------------------------------------------------------------------------------------
// V4-Flash checkpoint scaffolding — `v4.rs`, ARRIVED 2026-08-16 with M8's four oracle suites
// ---------------------------------------------------------------------------------------
//
// The old tree kept this in `common/mod.rs` because two suites drove the SAME two compressors
// (layer 2 at ratio 4, layer 3 at ratio 128), one against the oracle alone and one against the
// GPU, and a second copy of `compressor_w` would be a second set of shape assertions that could
// drift apart while both stayed green. Three suites need it here.
//
// **The split note this replaces predicted exactly this**: "when that loader arrives it lands
// in its own submodule beside the six above, not in this file — the umbrella holds module
// declarations and re-exports only." It did, and `v4.rs` carries one thing the old copy could
// not: the CONFIG PAIR, `Configs`, which holds the checkpoint's parsed `config.json` against
// the oracle's hard-coded transliteration of it. Every dimension a launcher is handed comes
// from the former; the latter builds the `Oracle` and nothing else.
