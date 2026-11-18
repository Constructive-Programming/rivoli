//! # rivoli-abi — the one definition of everything that crosses the kernel wall
//!
//! `no_std` and dependency-free so that BOTH sides compile it: the host (stable, as a
//! workspace member under `rivoli-backend`) and `crates/kernels` (the pinned nightly, for
//! `amdgcn-amd-amdhsa`). Before the Rust-kernel migration (`docs/investigations/
//! rust-kernels.md` §2) every `repr(C)` type here had a C twin in `kernels/*.hip` and a
//! compile-time size assert standing in for the agreement nothing else could check; a type
//! whose kernel has moved to Rust needs neither, because there is only one type.
//!
//! **Launch geometry lives here too, and that is the Phase 0c rule made structural.** A Rust
//! kernel never reads blockDim from the dispatch packet (§5, Phase 0c: ~1.1-1.2 µs/launch at
//! 24-48 workgroups), so a kernel whose block size the launcher FIXES compiles that size in
//! as a constant — and a constant the launcher can disagree with is silent wrong output
//! (0c's red arm). Both sides naming the SAME `const` is the tie: the launcher launches with
//! [`geom`]'s value and the kernel strides by it, so there is no second spelling to drift.
#![no_std]

pub mod geom;
pub mod kernarg;

/// The `repr(C)` mirror of `kernels/kvcompress.hip`'s `CompGeom`, and the only part of the
/// compressor's geometry that crosses the ABI wall.
///
/// Fields are PUBLIC, and that is a reviewed retreat, not an oversight: the old tree
/// enforced "only `Geom` constructs this" with module privacy, which a crate boundary
/// cannot express — a 7-arg field-order constructor is exactly as transposable as the
/// fields themselves (review 2026-08-15), so the ~45 lines of getters bought nothing.
/// What actually guards the layout is the compile-time assert below plus the kernel's
/// run-time `compress_guard`; what guards DERIVATION is the M8 rule that the engine's
/// `Geom` is the sole producer.
/// `PartialEq` and not `Eq`, because `eps` is a float — and it is derived at all for one
/// reason: it is what makes the M8 rule above CHECKABLE. The engine builds two of these on an
/// indexed layer, for the attention compressor and for the indexer's nested one, and the
/// claim that they differ only in a finish the kernel never sees is a claim about these
/// seven fields being equal. Without this derive that claim is a comment.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompGeom {
    pub ratio: i32,
    pub coff: i32,
    pub d: i32,
    pub cd: i32,
    pub ents: i32,
    pub rd: i32,
    pub eps: f32,
}

// The `repr(C)` mirror, pinned at compile time. The run-time `compress_guard` already
// catches a REORDER of the six `int`s — swap `ratio`/`coff` or `cd`/`ents` and guard 1003
// fires — which a size check cannot do, since a reorder is size-invariant. What the guard
// cannot catch is a field being ADDED, REMOVED, or changed WIDTH on one side only, because
// then the two structs are different objects and the guard is reading whichever bytes
// happen to line up. This catches that, at the file where the edit was made.
const _: () = assert!(
    size_of::<CompGeom>() == 28 && align_of::<CompGeom>() == 4,
    "CompGeom must stay six i32 and one f32 — the layout kernels/kvcompress.hip's CompGeom declares"
);

/// What the compressor's finish stage reads and writes — the same three pointers for both
/// pool kernels, in `kernels/kvcompress.hip`'s `CompFinish` layout.
///
/// One struct rather than three parameters because the three must AGREE with each other
/// and nothing in a flat argument list makes them: `freqs` must be the layer's
/// **compressed** RoPE table (the ratio-0 table has the same type, stride and shape, and
/// substituting it is fluent wrong text); `norm` is the **compressor's own** RMSNorm
/// weight, not the layer's `kv_norm`, which is also `[d]` and also f32; `out` is `[nblk,
/// d]` at prefill and one row at decode. The grouping was forced by the old tree's
/// duplication gate, which found the three repeated across the two pool call sites — the
/// gate was right about the shape of the problem, so they are chosen once, together.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CompFinish {
    pub norm: *const f32,
    pub freqs: *const f32,
    pub out: *mut f32,
}
