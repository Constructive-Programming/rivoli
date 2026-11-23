//! The one bar every CPU plant model clears before its device arm is worth a lease.
//!
//! A plant is shown ADVERSARIAL on the gate's own inputs by counting how many of the cells (or
//! seeds) it examines it actually moves. The spike's first plant passed at 8 seeds (§1 of
//! `rust-kernels.md`), so a plant that moves too few cells proves nothing: the device arm would
//! go green for want of power, not for want of a defect. Until 2026-11-23 this check was written
//! four times (`plants.rs`, `linalg_plants.rs`, `attn_plants.rs`, `plant_models.rs`) with three
//! spellings of the same ratio test; it lives here once.

/// How many of the examined cells (or seeds) a plant moved.
#[derive(Clone, Copy, Debug)]
pub(super) struct Rate {
    pub moved: u64,
    pub of: u64,
}

/// The fraction of examined cells a plant must move: `moved / of >= num / den`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Bar {
    pub num: u64,
    pub den: u64,
}

/// The default demand: the gate reddens on one changed word, so a plant must move at least
/// nine cells in ten. A model that argues a lower bar for its own plant (a sigmoid that swallows
/// a few-ulp move, a defect only some shapes can express) passes its own `Bar` and says why.
pub(super) const MOST: Bar = Bar { num: 9, den: 10 };

/// Print the plant's rate, and refuse it below `bar`.
pub(super) fn demand(plant: &str, rate: Rate, bar: Bar) {
    let Rate { moved, of } = rate;
    println!("{plant}: {moved}/{of} moved (bar {}/{})", bar.num, bar.den);
    assert!(
        of > 0,
        "{plant}: examined no cells — the plant model is vacuous"
    );
    assert!(
        moved * bar.den >= of * bar.num,
        "{plant}: moved only {moved}/{of} — below the {}/{} bar, sub-threshold",
        bar.num,
        bar.den
    );
}
