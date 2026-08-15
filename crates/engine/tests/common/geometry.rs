//! The shape bundles: dimension lists that travel together through an oracle, a launch
//! wrapper and a guard, passed as ONE value so the order is spelled in one place.
//!
//! Each of the three below makes the same argument in place and earned it separately — a row
//! of bare `usize` where every entry is plausible in any other's position moves the reference
//! and the kernel TOGETHER when it is transposed, so the comparison still agrees and the gate
//! is blind. Pure dimensions, so they sit here rather than beside a buffer type.
//!
//! **Split out of `common/mod.rs` 2026-08-15** under the file-size gate. Bodies and their
//! comments travelled verbatim, and `mod.rs` re-exports this module with a glob, so every
//! `use common::{Att, Mla, MoeRange}` in the oracle files is untouched.

/// The kv_b geometry both MLA launchers take.
///
/// Six bare `usize` in a row, every one of them plausible in any other's position, spelled
/// in an oracle, a launch wrapper and a guard closure PER BACKEND — five copies of the same
/// order, and a transposed pair would have moved the oracle and the kernel together. Pure
/// dimensions, so it belongs here rather than beside either backend's buffer type.
#[derive(Clone, Copy)]
pub struct Mla {
    pub h: usize,
    /// The q head stride. `mla_value_fp8` never reads q, so callers for that kernel leave this
    /// zero — cheaper than a second five-field shape whose only difference from this one is
    /// a field nothing reads.
    ///
    /// A `value_dims(h, nope, vh, kvl, block)` constructor did that zeroing until 2026-08-15.
    /// It was deleted rather than reshaped: five positional `usize` is the same excess-argument
    /// defect this struct exists to answer, and with the fields public `Mla { qh: 0, .. }` says
    /// it without an order to get wrong. [`Mla::new`] took six the same way and was reshaped to
    /// take ONE the same day — see its own note for why the fix was an array and not six named
    /// fields at every call site.
    pub qh: usize,
    pub nope: usize,
    pub vh: usize,
    pub kvl: usize,
    pub block: usize,
}

impl Mla {
    /// The six dims in kv_b order: `[h, qh, nope, vh, kvl, block]`.
    ///
    /// **One array argument, and the order is spelled HERE and nowhere else.** It took the six
    /// as six parameters until 2026-08-15, which is CodeScene's excess-argument rule at full
    /// size. The obvious fix — delete the constructor, write `Mla { h: 4, qh: 128, .. }` at each
    /// call site — was tried and REVERTED the same day: rustfmt's `struct_lit_width` is 18, so
    /// every literal wider than that becomes one line per field, and `kernel.rs`'s six MLA
    /// shapes (which differ in one dim at a time, deliberately) then shared four identical lines
    /// three ways. `build.rs`'s duplication gate reported it, correctly.
    ///
    /// So the array is not "positional again by accident". It keeps the six TOGETHER, names
    /// their order in one place, and leaves the public fields for a caller who wants
    /// `Mla { qh: 0, .. }` — which the value-kernel callers do.
    pub fn new(dims: [usize; 6]) -> Self {
        let [h, qh, nope, vh, kvl, block] = dims;
        Self {
            h,
            qh,
            nope,
            vh,
            kvl,
            block,
        }
    }

    /// kv_b's full row count, `h·(nope + vh)`.
    pub fn rows(self) -> usize {
        self.h * (self.nope + self.vh)
    }

    /// The two launcher guards both CPU oracles restate. An oracle that accepted a shape
    /// the launcher rejects would be checking the kernel against a case it can never run.
    ///
    /// `qh` is deliberately absent: the value kernel's callers leave it zero and
    /// `mla_value_fp8` never reads it, so the absorb oracle asserts it separately.
    pub fn assert_guarded(self) {
        let (h, nope, vh) = (self.h, self.nope, self.vh);
        let (kvl, block) = (self.kvl, self.block);
        assert!(
            h > 0 && nope > 0 && vh > 0 && kvl > 0 && block > 0,
            "guard 1001"
        );
        assert!(
            block.is_power_of_two(),
            "guard 1003: blk_shift needs a power-of-two tile"
        );
    }
}

/// The MLA attention's shape.
///
/// Five `usize` and an f32 that travel together through the split planner, the tile
/// widener, the CPU reference and the dispatch — and the reference and the dispatch take
/// the SAME six, so every test spelled them twice. A transposed pair would have moved both
/// sides identically and the comparison would still have agreed.
#[derive(Clone, Copy)]
pub struct Att {
    pub h: usize,
    pub nr: usize,
    pub kvl: usize,
    pub rope: usize,
    pub n_blocks: usize,
    pub scale: f32,
}

impl Att {
    /// Neither `n_blocks` nor `scale` is a free parameter, and both are derived here for the
    /// same reason.
    ///
    /// The fp8 latent cache carries one block scale per 128 latent dims, so `n_blocks`
    /// FOLLOWS from `kvl` and every test derived it the same way; deriving it once removes the
    /// only way a reference and a launcher could have been handed different block-scale
    /// strides for the same cache.
    ///
    /// The softmax `scale` follows from `kvl + rope` the same way. `kernel.rs` carried an
    /// `att(h, nt, kvl, rope)` wrapper that did nothing else, arguing that "five call sites
    /// spelling `1.0 / ((kvl + rope) as f32).sqrt()` is five places for it to drift from the
    /// kernel's" — true, and the wrapper left a fifth argument here that could still be handed
    /// a wrong number. **Folded in 2026-08-15** with the `kernel.rs` split, which deleted the
    /// wrapper: there is now no way to state a scale that does not follow from the shape. The
    /// guard test, which only asks whether a `kvl` is accepted, previously passed a literal
    /// `1.0` and now takes the derived value — it never reads the result.
    pub fn new(h: usize, nr: usize, kvl: usize, rope: usize) -> Self {
        Self {
            h,
            nr,
            kvl,
            rope,
            n_blocks: kvl / 128,
            scale: 1.0 / ((kvl + rope) as f32).sqrt(),
        }
    }
}

/// One MoE dispatch's geometry: the two matrix dims and the half-open expert range
/// `[e_start, e_start + e_count)`.
///
/// The same four, in the same order, in `moe_expert_range`'s wrapper and in both of the
/// VQ oracles that check it — three copies per backend of a list whose middle two entries
/// are interchangeable to the type checker.
#[derive(Clone, Copy)]
pub struct MoeRange {
    pub hidden: usize,
    pub inter: usize,
    pub e_start: usize,
    pub e_count: usize,
}

impl MoeRange {
    pub fn new(hidden: usize, inter: usize, e_start: usize, e_count: usize) -> Self {
        Self {
            hidden,
            inter,
            e_start,
            e_count,
        }
    }

    /// One past the last expert this range writes — the oracles size their staging by it.
    pub fn e_end(self) -> usize {
        self.e_start + self.e_count
    }
}
