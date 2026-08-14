//! The `repr(C)` types that cross the kernel ABI wall, available featureless.
//!
//! Moved here from the old tree's `kvcompress.rs` when the backend became its own crate:
//! `hip.rs`'s launchers take these by pointer, and the backend depends on nothing above
//! libc, so the mirror structs live WITH the wall they mirror. The semantic constructors
//! (deriving geometry from a `LayerKind`) stay with the engine's compressor module, which
//! remains the sole intended producer — see `CompGeom::new`'s note.

/// The `repr(C)` mirror of `kernels/kvcompress.hip`'s `CompGeom`, and the only part of the
/// compressor's geometry that crosses the ABI wall.
///
/// Fields are PRIVATE and [`CompGeom::new`] is the only way in, because three of the six
/// integers are derived and a caller that computed one from a stale `coff` gets a kernel
/// that reads the right shape at the wrong stride — `ape`'s row stride is `cd`, and at
/// ratio 4 that is 1024 while at ratio 128 it is 512. Handing the kernel six loose `i32`s
/// would make every one of those positions plausible in the others.
///
/// In the old tree the engine's `Geom` was the one construction site, enforced by module
/// privacy; across the crate boundary that enforcement is down to `new` being awkward
/// enough that nobody calls it casually, plus the run-time `compress_guard` (catches a
/// REORDER of the six ints) and the layout assert below (catches a width/count drift).
/// When the compressor module arrives (M8), it re-earns sole-producer status there.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CompGeom {
    ratio: i32,
    coff: i32,
    d: i32,
    cd: i32,
    ents: i32,
    rd: i32,
    eps: f32,
}

impl CompGeom {
    /// The one way in. Argument order is the C struct's field order on purpose — a caller
    /// building this is transcribing the kernel's declaration, not composing free values.
    #[allow(clippy::too_many_arguments)] // it mirrors a seven-field C struct, one arg per field
    pub const fn new(ratio: i32, coff: i32, d: i32, cd: i32, ents: i32, rd: i32, eps: f32) -> Self {
        CompGeom {
            ratio,
            coff,
            d,
            cd,
            ents,
            rd,
            eps,
        }
    }

    pub const fn ratio(&self) -> i32 {
        self.ratio
    }
    pub const fn coff(&self) -> i32 {
        self.coff
    }
    pub const fn d(&self) -> i32 {
        self.d
    }
    pub const fn cd(&self) -> i32 {
        self.cd
    }
    pub const fn ents(&self) -> i32 {
        self.ents
    }
    pub const fn rd(&self) -> i32 {
        self.rd
    }
    pub const fn eps(&self) -> f32 {
        self.eps
    }
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

/// The four extents the sparse indexer's scoring contracts over.
///
/// A struct because they are four bare `usize` in a row and every one is plausible in any
/// other's position — `heads` and `hd` in particular are 64 and 128 on every layer that
/// has an indexer, so a transposed pair indexes a real row of `q` and produces a finite
/// score. What it does NOT buy, stated because the neighbouring types do enforce
/// something: there are no derived fields here and no relation between them to check, so
/// this is naming and nothing else.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScoreDims {
    /// Query rows: the prompt length at prefill, 1 at decode.
    pub s: usize,
    /// Compressed blocks visible to this call, `(start_pos + seqlen) / ratio`.
    pub n_comp: usize,
    /// The indexer's OWN head count — NOT the attention's, which can coincide.
    pub heads: usize,
    /// The indexer's OWN head dim — NOT the attention's.
    pub hd: usize,
}
