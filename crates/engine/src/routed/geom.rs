//! The routed set's STATIC geometry: the pool key packing, the six projection offsets
//! resolved to device addresses, and the per-`(layer,expert)` O_DIRECT read-spec table.
//!
//! Split out of `routed.rs` when that file crossed the 800-line soft cap
//! (`crates/cli/build.rs`); the cut is by cohesion, not by size. Nothing here holds
//! residency state or mutates: every value is fixed at `open()` from the artifact and
//! read-only for the rest of the run, which is exactly the half of the pool that has no
//! interaction with eviction, relocation or the fetch ring. The comments carrying the
//! measurements that justified each choice travel with their code.

use anyhow::{Context, Result, ensure};
use rivoli_artifact::format::{ExpertSet, RoutedFmt};
use std::os::fd::RawFd;

/// One projection's two addresses inside an expert's pool slot.
///
/// **Both are `*const u8` and that is the point.** This carried a `*const u16` `scales`
/// while there were two formats, which was already a half-truth (`.i4`'s scales are f32,
/// "reinterpreted at the launch site") and becomes a wrong one at `.f4`, whose e8m0
/// scales are ONE byte. A slot is six byte addresses; what they mean is the descriptor's
/// business, said once at the descriptor type.
#[derive(Clone, Copy)]
pub struct ProjSlot {
    pub packed: *const u8,
    pub scale: *const u8,
}

/// One expert's three projections resolved to device addresses — what a launch
/// descriptor is built from. Field order is slot order everywhere in this engine:
/// gate, up, down.
#[derive(Clone, Copy)]
pub struct ExpertSlot {
    pub gate: ProjSlot,
    pub up: ProjSlot,
    pub down: ProjSlot,
}

/// Resolve one projection's two pointers at slot-relative offsets `(poff, soff)` from an
/// expert-block base — the single builder shared by resident shared experts and the
/// streamed routed ones.
///
/// # Safety
/// Both offsets must lie within the expert block at `base`.
#[inline]
unsafe fn proj_at(base: *const u8, poff: usize, soff: usize) -> ProjSlot {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        ProjSlot {
            packed: base.add(poff),
            scale: base.add(soff),
        }
    }
}

/// Resolve all six offsets against one block base.
///
/// # Safety
/// Every offset in `off` must lie within the expert block at `base`.
#[inline]
pub unsafe fn slot_at(base: *const u8, off: &[usize; 6]) -> ExpertSlot {
    // SAFETY: forwarded from this function's own contract.
    unsafe {
        ExpertSlot {
            gate: proj_at(base, off[0], off[1]),
            up: proj_at(base, off[2], off[3]),
            down: proj_at(base, off[4], off[5]),
        }
    }
}

/// Pack `(layer, expert)` into the pool key. Both must fit in 16 bits — GLM is
/// ≤92 layers × 256 routed experts, comfortably under 2^16.
pub fn expert_key(layer: usize, expert: usize) -> u32 {
    debug_assert!(
        layer < (1 << 16) && expert < (1 << 16),
        "layer {layer}/expert {expert} exceed the 16-bit pool key packing"
    );
    ((layer as u32) << 16) | expert as u32
}

/// The routed set's geometry: the projection offsets, which format decodes it, the slot
/// stride, and the per-`(layer,expert)` O_DIRECT read-spec table. ONE per pool — the
/// arena still has two ends (that is 2Q's shape), but both ends hold this same layout,
/// so compaction is always a same-size move and no check has to keep two copies agreeing.
#[derive(Clone)]
pub struct RoutedGeom {
    pub(super) off: [usize; 6],
    pub(super) fmt: RoutedFmt,
    pub(super) stride: usize,
    /// `(fd, begin, len)` per `(layer - first_layer) * n_experts + expert`.
    table: Vec<(RawFd, usize, usize)>,
    /// The row basis of `table`, kept WITH the table rather than beside it in
    /// [`RoutedPool`](super::RoutedPool): `(layer - first_layer) * n_experts + expert`
    /// indexed with a `first_layer` from somewhere else reads a different layer's expert
    /// and fails no check. [`RoutedGeom::spec`] is the only reader, so there is no second
    /// copy to disagree — which is why the pool has no guard comparing bases, and why it
    /// should not grow one.
    first_layer: usize,
    n_experts: usize,
}

impl RoutedGeom {
    /// Tabulate one set's read specs, taking EVERYTHING from the set: the format, the six
    /// projection offsets, the slot stride, the layer range and the expert count.
    ///
    /// **One argument, and that is the design.** This took `fmt` and `off` and a `layers`
    /// range, with an `ensure!` that the offsets were ascending and inside the stride.
    /// That guard was written, and then asked what would have to be true for it to fire:
    /// nothing realistic. Every routed block is padded up to `VQ_ALIGN`, so `.vq3`'s
    /// layout on an `.f4` slot (9,961,472 against a 13,369,344 stride) sits comfortably
    /// inside it and passes — and `.f4` and `.i4` tile identically for 25% of all
    /// `i_dim`, both models' dimensions included, so the pairing that actually costs
    /// correctness is invisible to any check at all. A guard that cannot fire is worse
    /// than none; the fix is that the set knows its own format, so there is nothing left
    /// to pair wrongly.
    pub fn new(src: &ExpertSet) -> Result<Self> {
        let layers = src.layers();
        let n_experts = src.n_experts();
        let first_layer = layers.start;
        let mut table = Vec::with_capacity(layers.len() * n_experts);
        for l in layers {
            for e in 0..n_experts {
                table.push(src.read_spec(l, e)?);
            }
        }
        Ok(Self {
            off: src.slot_offsets(),
            fmt: src.fmt(),
            stride: src.expert_slot(),
            table,
            first_layer,
            n_experts,
        })
    }

    /// The cold-read spec for `(layer, expert)`, by ABSOLUTE layer id.
    ///
    /// The `layer - first_layer` subtraction lives here, with the table it indexes:
    /// indexing the table with its own basis removes the disagreement instead of
    /// checking for it.
    pub(super) fn spec(&self, layer: usize, expert: usize) -> Result<(RawFd, usize, usize)> {
        self.row(layer)?;
        self.table
            .get(self.row(layer)? * self.n_experts + expert)
            .copied()
            .context("unreachable: `row` bounds both indices")
    }

    /// `layer`'s row in [`Self::table`], both ends checked.
    ///
    /// **`expert` is bounded here too, and a `table.get()` alone would not do it.** The
    /// index is `row * n_experts + expert`, so on any row but the LAST an
    /// `expert >= n_experts` lands inside the table on a later layer's row and comes back
    /// `Ok` with that layer's fd and offset — a silently wrong cold read, not an error.
    /// Concretely on a 3-layer fixture, `(0, 256)` would return layer 1 expert 0.
    /// `ExpertSet::read_spec` bounds `expert`, but that ran at table-BUILD time; nothing
    /// re-checked it at lookup.
    ///
    /// Split out from [`Self::spec`] so [`RoutedPool::submit`](super::RoutedPool::submit)
    /// can run it BEFORE it mutates anything — see the range check at the top of `submit`
    /// for why that ordering is not a style preference.
    fn row(&self, layer: usize) -> Result<usize> {
        let rows = self.table.len() / self.n_experts;
        let row = layer.checked_sub(self.first_layer).filter(|&r| r < rows);
        row.with_context(|| {
            format!(
                "layer {layer} is outside a .{} set over {rows} layers from layer {}",
                self.fmt.ext(),
                self.first_layer,
            )
        })
    }

    /// Is `(layer, expert)` addressable? The pre-flight half of [`Self::spec`].
    pub(super) fn addressable(&self, layer: usize, expert: usize) -> Result<()> {
        self.row(layer)?;
        ensure!(
            expert < self.n_experts,
            "expert {expert} >= {} in a .{} set",
            self.n_experts,
            self.fmt.ext(),
        );
        Ok(())
    }
}
