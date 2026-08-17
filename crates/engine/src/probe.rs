//! `--divergence-log`: localise a run-to-run divergence to a (position, layer, QUANTITY)
//! coordinate, without perturbing what it measures.
//!
//! ## The question it answers
//!
//! GLM int3-vq does not reproduce itself. Teacher-forced, two runs at identical flags first
//! disagree at position 236 and 362 of 762 — about one event per 300 token-forwards — and
//! greedy decode turns one such event into a rewritten tail. **Neither the output text nor
//! the `.nll` file can say WHERE the event was**, only that everything after it differs:
//! past the first divergence the two runs are computing over different state, so a differing
//! COUNT is one event plus its wake and not a severity.
//!
//! `tests/nll-divergence.sh` extracts the first position from a pair of `.nll` files. This
//! goes one level down and names the (layer, quantity) inside it.
//!
//! ## What makes it a DISCRIMINATOR and not just a localiser
//!
//! Three quantities per layer, cutting the layer at the two seams the two standing
//! hypotheses sit either side of:
//!
//! | slot | quantity | folded for | a difference HERE, with the earlier slots equal, means |
//! |---|---|---|---|
//! | [`Q::Xn`] | `xn`, the post-attention rmsnorm output | every layer | attention or its KV cache; the MLP has not run yet |
//! | [`Q::H`] | `moe_hidden`, the SwiGLU intermediate | MoE layers | `xn` was equal, so the GATE/UP expert BYTES, or that kernel |
//! | [`Q::X`] | the residual at layer exit | every layer | both were equal, so the down projection, the accumulator or the drain |
//!
//! `Xn` covers GLM's 3 dense layers as well as its 75 MoE ones, so "attention agreed here" is
//! a claim the log can make about any layer. `H` cannot exist on a dense layer and prints `-`
//! there — never 0, because a 0 would let two runs "agree" about a quantity neither measured.
//!
//! Beside them, five host columns ([`Cols`]) that cost no device traffic at all because
//! routing is already a host function of host-resident data: what the router SAW, what it
//! PICKED, WHERE the pool put each expert, and the layer's miss and relocation deltas.
//!
//! Diff two logs: the first differing LINE is the coordinate, the first differing COLUMN
//! names the mechanism.
//!
//! ## The two properties that let it be pointed at THIS bug
//!
//! **The fold is XOR** — `rivoli_core::hash::xor_fold` states the argument and
//! `kernels/fwd.hip::hash_rows` is its device twin. XOR is commutative *and* associative, so
//! the device fold is bit-identical whatever order its atomics land in; a float sum would
//! report a difference from scheduling jitter alone, and an instrument noisier than its
//! subject measures nothing.
//!
//! **Nothing touches the host or the disk between the first fold and the last token.** The
//! archived predecessor (`--checksum-x`, archive 544fea7) copied the residual to the host
//! every layer and produced a CLEAN run on a configuration that reproduced without it — the
//! tool built for the bug could not be used on it. So: folds stay on the device, every slot
//! drains in ONE D2H per pass at a point the end-of-layer `device_sync` has already idled,
//! and the records are written after the last token.
//!
//! That is also why this is its own feature rather than part of `trace`, which adds a poison
//! fill and a `device_sync` per layer-with-misses. **Never debug this defect under `--trace`,
//! and never accept a green obtained with tracing enabled.**

use crate::device::DeviceBuf;
use crate::fetch::asyncfetch::ScProbe;
use anyhow::{Context, Result, ensure};
use rivoli_backend::{fill_u32, launch_hash_rows};

/// The device-folded quantities. The ORDER is the cut described in this module's header —
/// "before the MLP", "after gate/up", "after down + drain" — so a reader of a log line sees
/// the layer's causal order left to right.
/// `Copy` so a fold site can name a variant without borrowing; nothing clones one.
#[derive(Clone, Copy)]
pub enum Q {
    /// `xn`: the MoE's input, after the post-attention rmsnorm.
    Xn = 0,
    /// `moe_hidden`: the SwiGLU intermediate, after gate/up and before down.
    H = 1,
    /// The residual stream at layer exit.
    X = 2,
    /// **`bh`** — the BOUNCE ARENA slot, folded on the fetch stream the moment the NVMe read
    /// completes and BEFORE the bounce->slot copy. "What the drive delivered."
    Bh = 3,
    /// **`sc`** — the POOL slot, folded on the fetch stream immediately AFTER the copy.
    /// "What arrived in the pool." `bh != sc` isolates the copy.
    Sc = 4,
    /// **`se`** — the POOL slots of the WHOLE BATCH, folded at END OF LAYER on the null stream
    /// after both MoE lanes have been awaited, each at its own `i_base`. "What every expert this
    /// layer used holds, at rest."
    ///
    /// **Its value is COVERAGE, and the rule for reading it is CROSS-RUN.** Two earlier versions of
    /// this comment got that wrong, and both errors are worth recording. The first claimed the
    /// point was detecting a write to the slot after the copy — review answered that `bh` equal
    /// *and* `sc` equal with `h` differing already implies the kernel read something else, which is
    /// true for the expert that was READ. `bh`/`sc` bracket ONE copy, so they see ~1.7 of the
    /// batch's 9 slots at the measured 78% hit rate; `se` sees all 9, and that is what separates "a
    /// RESIDENT expert's bytes went wrong on an earlier token" from "the kernel read a fresh slot
    /// too early". The second version then invited `sc == se` as a WITHIN-RUN test, which cannot
    /// hold: a fold over one slot and a fold over nine are different quantities and would differ on
    /// every row, sending an operator after an innocent hop. **Compare A's `se` against B's `se`.**
    /// The full rule is in the log's own header, where it travels with the data.
    Se = 5,
}

/// Which of the three FETCH-PATH folds a run enables. `xn`/`h`/`x` are always folded.
///
/// **The default is NONE, and that is the whole point of this type.** The light probe — the
/// per-layer columns with no fetch-path folds — is the configuration that PRODUCED the token-164
/// coordinate. Adding all three at once produced 2,048 instrumented tokens with ZERO events
/// against a rate predicting ~4-7 (P = 0.11% matched / 2.89% conservative): **the heavy probe
/// suppresses the defect it was built to localise.** So the folds are now opt-in one at a time,
/// and the suppressing configuration cannot be reached by accident.
///
/// Which fold turns RED→GREEN names the mask, and its position in the pipeline names where the
/// mechanism lives. `se` is the control: it runs AFTER the consumer has read the slot, so it
/// should not be able to suppress anything — if it does, the hypothesis is wrong.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Folds {
    /// Fold the bounce arena after the NVMe read, before the copy.
    pub bh: bool,
    /// What, if anything, runs at the post-copy position.
    pub sc: crate::fetch::asyncfetch::ScProbe,
    /// Fold every slot the layer used, at end of layer.
    pub se: bool,
}

impl Folds {
    /// Parse `--divergence-folds`: a comma-separated subset of
    /// `bh,sc,se,sc-spin,sc-line`. Empty or absent = the light probe.
    ///
    /// Unknown names are REFUSED rather than ignored, and the three `sc` forms are mutually
    /// exclusive: a typo that silently disabled the one fold a cell exists to test would make the
    /// cell's green mean the opposite of what it appears to.
    pub fn parse(spec: &str) -> Result<Self> {
        let mut f = Self::default();
        for name in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let sc = |f: &Folds| -> Result<()> {
                ensure!(
                    f.sc == ScProbe::Off,
                    "--divergence-folds names more than one `sc` variant; they occupy the same \
                     pipeline position and are alternatives, not additions"
                );
                Ok(())
            };
            match name {
                "bh" => f.bh = true,
                "se" => f.se = true,
                "sc" => {
                    sc(&f)?;
                    f.sc = ScProbe::Full;
                }
                "sc-spin" => {
                    sc(&f)?;
                    f.sc = ScProbe::Spin;
                }
                "sc-line" => {
                    sc(&f)?;
                    f.sc = ScProbe::Line;
                }
                other => anyhow::bail!(
                    "unknown --divergence-folds entry {other:?} (bh, sc, sc-spin, sc-line, se)"
                ),
            }
        }
        Ok(f)
    }

    /// The config as it appears in the log header — every log states what produced it, because a
    /// divergence log without its probe configuration cannot be attributed, and two logs from
    /// different configurations must never be compared.
    pub fn label(&self) -> String {
        let mut v: Vec<&str> = Vec::new();
        if self.bh {
            v.push("bh");
        }
        match self.sc {
            ScProbe::Off => {}
            ScProbe::Full => v.push("sc"),
            ScProbe::Spin => v.push("sc-spin"),
            ScProbe::Line => v.push("sc-line"),
        }
        if self.se {
            v.push("se");
        }
        match v.is_empty() {
            true => "light".to_string(),
            false => v.join(","),
        }
    }
}

#[cfg(test)]
mod fold_tests {
    // A panic in a test is loud and correct, which is the workspace's stated reason for opting back
    // in per file; a panic on the decode path would be a crash in a server, which is why it is
    // deny-level everywhere else.
    #![allow(clippy::expect_used)]
    use super::Folds;
    use crate::fetch::asyncfetch::ScProbe;

    /// `--divergence-folds` parsing, with the two refusals that matter.
    ///
    /// A silently-misparsed fold set is the worst outcome this flag has: every Phase 1 cell is
    /// "enable exactly one fold and see whether the pair still diverges", so a typo that quietly
    /// enabled NOTHING would make the cell green and be read as "this fold is the mask" — the
    /// precise inversion of the truth. Hence unknown names are refused rather than ignored, and the
    /// three `sc` forms refuse each other rather than the last one winning.
    #[test]
    fn folds_parse_refuses_what_it_cannot_honour() {
        // The default is the LIGHT probe — the configuration that produced the coordinate.
        assert_eq!(Folds::parse("").expect("empty").label(), "light");
        assert_eq!(Folds::default().label(), "light");

        let f = Folds::parse("bh").expect("bh");
        assert!(f.bh && !f.se && f.sc == ScProbe::Off, "bh must not arm sc");
        assert_eq!(f.label(), "bh");

        let f = Folds::parse("sc-spin").expect("sc-spin");
        assert_eq!(f.sc, ScProbe::Spin);
        assert!(!f.bh && !f.se, "an sc variant must not arm the others");
        assert_eq!(f.label(), "sc-spin");

        // Order-independent, and the label is canonical so two logs of the same config compare
        // equal however the operator typed it.
        assert_eq!(
            Folds::parse("se, bh ,sc-line").expect("spaced").label(),
            Folds::parse("bh,sc-line,se").expect("plain").label()
        );

        assert!(
            Folds::parse("bh,nope").is_err(),
            "unknown names are refused"
        );
        assert!(
            Folds::parse("sc,sc-line").is_err(),
            "two sc variants occupy one pipeline position and must refuse"
        );
        assert!(Folds::parse("sc-spin,sc").is_err(), "in either order");
    }
}

/// Device u64 fold slots per layer — one per [`Q`].
const NQ: usize = 6;

/// `NQ` must cover every [`Q`], checked AT COMPILE TIME.
///
/// A `#[test]` was the first form and was the wrong one: `mod probe` is gated on
/// `corruption-probe`, the prescribed battery is `cargo test --workspace` (which does not set it),
/// CI has no rocm arm, and `tests/feature-matrix.sh` runs `cargo check` plus two integration
/// targets — so nothing executed it (review, 2026-08-17). A `const` assertion is checked by every
/// build of every feature cell, which is the coverage the claim needs.
const _: () = assert!(Q::Se as usize + 1 == NQ, "NQ must cover every Q variant");

/// One MoE layer's host-side columns. Every field is a fold of data the decode thread already
/// holds, so recording them costs no device traffic and no I/O.
///
/// **There were six; `wexpert`'s fold was cut.** Review, 2026-08-17: the routing weights are a
/// pure host function of `gl` (`weigh_row` is a sum, a divide and a multiply over a `Vec`), so
/// with `gl` equal they cannot differ, and the union build that `wexpert`'s layout also covered
/// is covered by `sl` — which folds slot offsets IN UNION ORDER. A column that cannot differ
/// widens every line and the diff that reads it.
#[derive(Clone, Copy)]
pub struct Cols {
    /// FNV-1a over row 0's gate logits' exact BYTES — bit pattern, never value, so a one-ulp
    /// move is visible. This is what the router SAW.
    pub gl: u64,
    /// FNV-1a over every row of the pass's picks, in row-then-rank order. This is what it DID with what it saw, and
    /// with `gl` beside it the pair answers INV-1 directly: equal `gl` and unequal `pk` would
    /// mean routing consulted something outside its inputs.
    ///
    /// **The reason first given for keeping this was WRONG and is corrected here** (review,
    /// 2026-08-17): it said the picks' determinism rests on `select_nth_unstable_by`, a std
    /// implementation detail. It does not — `core::routing::topk_into` follows the partition
    /// with `out.sort_by(cmp)` under a TOTAL comparator (value-desc, then index-asc), so std's
    /// instability is invisible and the picks in rank order are a pure function of the logits.
    /// By that argument alone this column would be as derivable as `wexpert` was, and would go.
    ///
    /// It stays for a different and real reason: it DISAMBIGUATES the line above it from the
    /// line below. A record with `gl` equal and `sl` different is ambiguous between "routing
    /// chose different experts" and "the pool placed the same experts in different slots" —
    /// two entirely different subsystems. With `pk` between them the log answers that itself.
    pub pk: u64,
    /// FNV-1a over each selected expert's SLOT OFFSET in the pool, in union order — a
    /// pool-relative number, never an address, because the VMM base differs between runs and
    /// hashing it would report a difference on every line.
    pub sl: u64,
    /// Cold reads this layer submitted.
    pub miss: u64,
    /// Arena compaction relocations this layer executed.
    ///
    /// **Kept although relocation is the hypothesis this milestone REFUTED**, which sounds
    /// backwards and is not: the refutation is structural reasoning about barriers, and this
    /// probe exists because structural reasoning about this defect has not yet been confirmed
    /// by measurement. A column that lets the first device run CHECK the refutation at zero
    /// cost is worth more than one that assumes it — and the archived probe's load-bearing
    /// finding was exactly "identical (misses, relocs), different answer", which this column
    /// is what made sayable.
    pub reloc: u64,
}

/// The `--divergence-log` state: the device fold slab, and the records so far.
pub struct Probe {
    /// `[n_layers][NQ]` device u64. Zeroed at construction and re-zeroed by every
    /// [`Probe::drain`] — the fold is an XOR and `hipMalloc` does not zero, so an unzeroed slab
    /// would corrupt exactly the first token's hashes, the one token an investigation into
    /// onset POSITION least wants to distrust.
    dev: DeviceBuf,
    host: Vec<u8>,
    /// Per-layer host columns for the pass in flight; `None` on a dense layer, which has no
    /// router and no pool.
    cols: Vec<Option<Cols>>,
    /// Which fetch-path folds this run enables; see [`Folds`].
    folds: Folds,
    /// The whole log, appended a line at a time. Held in memory: see the header for why the
    /// write cannot happen during the run. ONE `String` rather than a `Vec<String>` — at
    /// 512 tokens x 78 layers that would be 40k heap allocations on a path whose entire design
    /// goal is to add nothing to the run.
    recs: String,
}

impl Probe {
    pub fn new(n_layers: usize, folds: Folds) -> Result<Self> {
        ensure!(n_layers > 0, "divergence probe over a 0-layer model");
        let bytes = n_layers * NQ * 8;
        let mut dev = DeviceBuf::new(bytes)?;
        // SAFETY: `dev` owns `bytes`, just allocated, and `bytes` is a multiple of 4.
        unsafe { fill_u32(dev.ptr_mut(), 0, bytes)? };
        Ok(Self {
            dev,
            host: vec![0u8; bytes],
            cols: vec![None; n_layers],
            folds,
            recs: String::new(),
        })
    }

    /// The device address of `(layer, q)`'s fold slot, bounds-checked.
    ///
    /// One place, because a layer outside the slab would fold into ANOTHER LAYER's slot and report
    /// a difference that never happened — the worst failure mode an instrument has — and two
    /// copies of that check are one edit away from disagreeing. (jscpd said so on the day the
    /// second caller landed.) Refused with both numbers rather than clamped.
    fn slot_ptr(&mut self, layer: usize, q: Q) -> Result<*mut u64> {
        ensure!(
            layer < self.cols.len(),
            "divergence probe: layer {layer} outside a {}-layer slab",
            self.cols.len()
        );
        // `q` is bounded too, and not for symmetry: `layer * NQ + q` with `q >= NQ` lands in the
        // NEXT layer's block on every layer but the last, and past the `DeviceBuf` on the last —
        // a silent device OOB write, and on the earlier layers a fold that corrupts another
        // layer's slot while looking like it worked. Adding a `Q` variant without bumping `NQ`
        // is all it takes, so this is total rather than trusting the enum (review, 2026-08-17).
        ensure!(
            (q as usize) < NQ,
            "divergence probe: Q index {} >= NQ {NQ} — a variant was added without bumping NQ",
            q as usize
        );
        // SAFETY: `layer * NQ + (q as usize) < cols.len() * NQ` by the check above and `q < NQ`,
        // so the offset is inside the slab this `DeviceBuf` owns.
        Ok(unsafe { (self.dev.ptr_mut() as *mut u64).add(layer * NQ + q as usize) })
    }

    /// Which fetch-path folds are enabled.
    pub fn folds(&self) -> Folds {
        self.folds
    }

    /// The device address of `(layer, q)`'s fold slot — for a caller that folds into it itself
    /// (the reaper thread, via [`crate::fetch::asyncfetch::FetchFolds`]) rather than through
    /// [`Self::fold`].
    ///
    /// Each slot is named by its [`Q`], never reached by offset from another. An earlier version
    /// handed out `Q::Bh`'s address and let callers `.add(1)`/`.add(2)`, which was correct only
    /// because of the order the variants are declared in and needed a test to catch a reordering.
    pub fn fold_slot(&mut self, layer: usize, q: Q) -> Result<*mut u64> {
        self.slot_ptr(layer, q)
    }

    /// XOR-fold `n` device f32 at `x` into `(layer, q)`'s slot, on the null stream. No sync, no
    /// D2H.
    ///
    /// # Safety
    /// `x` must be `n` live device f32 whose writers have retired.
    pub unsafe fn fold(&mut self, q: Q, layer: usize, x: *const f32, n: usize) -> Result<()> {
        let out = self.slot_ptr(layer, q)?;
        // SAFETY: `out` is one live device u64 inside the slab; `x`/`n` are the caller's
        // contract, forwarded.
        unsafe { launch_hash_rows(x, n, 0, out, rivoli_backend::NULL_STREAM) }
    }

    /// Record a MoE layer's host columns for the pass in flight.
    ///
    /// Refuses an out-of-range layer rather than dropping it, for the same reason [`Self::fold`]
    /// does: a silently dropped column reads in the log exactly like a dense layer, and a diff
    /// would then report two runs agreeing about a layer neither recorded.
    pub fn set_cols(&mut self, layer: usize, c: Cols) -> Result<()> {
        *self
            .cols
            .get_mut(layer)
            .with_context(|| format!("divergence probe: layer {layer} outside the slab"))? =
            Some(c);
        Ok(())
    }

    /// Drain `layers`' folds in ONE D2H, emit one record per layer, and re-zero the slab.
    ///
    /// Called once per pass — once per token in decode, once per (layer, rows) pass under
    /// layer-major prefill — at a point the end-of-layer `device_sync` has already idled the
    /// device, so it adds no barrier of its own.
    ///
    /// **Re-zeroes the WHOLE slab, not just `layers`.** Under layer-major prefill a pass covers
    /// one layer, so the other slots are already zero and clearing them is free; in decode the
    /// pass covers every layer. The alternative — clearing only the drained range — would
    /// differ from this only in a shape neither caller produces, and would leave a stale slot
    /// live if one ever did. XOR is self-inverse, so a stale slot is not a loud failure but a
    /// wrong hash.
    pub fn drain(&mut self, pos: usize, nrow: usize, layers: std::ops::Range<usize>) -> Result<()> {
        self.dev.copy_out_into(&mut self.host)?;
        let bytes = self.host.len();
        // SAFETY: `dev` owns `bytes` (allocated with exactly this length) and it is a multiple of 4.
        unsafe { fill_u32(self.dev.ptr_mut(), 0, bytes)? };
        // THE SYNC IS LOAD-BEARING AND WAS A REAL BUG WITHOUT IT.
        //
        // `fill_u32` is an ASYNC launch on the null stream. The three fetch-path folds (`Bh`,
        // `Sc`, `Se`) are launched by the REAPER THREAD on the FETCH stream, which is
        // `hipStreamNonBlocking` — so the null stream carries no implicit ordering against them.
        // Without this, the next pass's arena fold could land BEFORE this clear executed and then
        // be wiped: two runs would read 0 and "agree" about bytes neither hashed (a false
        // exclusion), or one would read 0 and the other a hash (a false positive naming the wrong
        // hop). Either is the instrument lying, which is the one thing it may not do.
        //
        // It is closed today by accident as well — `route_layer`'s gate-logits D2H is a blocking
        // null-stream sync and sits between this clear and any later fold — but an instrument must
        // not rest on a barrier that belongs to someone else's code and could go async.
        //
        // The cost is nil where it stands: `drain` runs at the end of a pass, immediately after
        // its own blocking D2H, at a point `run_layer`'s per-layer `device_sync` has already
        // idled the device. It adds no barrier that was not already there — which is the test any
        // sync in this file has to pass, because added syncs are what MASKED this defect's
        // predecessor.
        rivoli_backend::device_sync()?;
        let words: Vec<u64> = self
            .host
            .chunks_exact(8)
            .map(|c| c.try_into().map(u64::from_le_bytes).unwrap_or(0))
            .collect();
        for l in layers {
            let at = |q: usize| -> Result<u64> {
                words
                    .get(l * NQ + q)
                    .copied()
                    .context("divergence probe: fold slot outside the drained slab")
            };
            // A dense layer has no router, no pool and no `moe_hidden`, so those columns print
            // `-` rather than 0. NOT cosmetic: a 0 would read as a measured hash that happened
            // to be zero, so two runs would "agree" about a quantity neither of them measured
            // — a false EXCLUSION, which is the one failure mode an instrument must not have.
            // `xn` and `x` ARE folded on every layer, so they are always numbers.
            let moe = self.cols.get(l).copied().flatten();
            let h = match moe {
                Some(_) => format!("{:016x}", at(Q::H as usize)?),
                None => "-".to_string(),
            };
            // `-` MEANS NOT MEASURED and is never 0, for two reasons that compound. A fold is
            // absent when the run did not ENABLE it (`--divergence-folds`) and when the layer gave
            // it nothing to do — `bh`/`sc` bracket a copy, so with no miss there was no copy; `se`
            // runs on every MoE layer. Printing 0 for either would let two runs "agree" about bytes
            // neither hashed, which is the false-exclusion trap this instrument already fell into
            // once. `sc-spin` also prints `-`: it writes a word so the loop is not optimised away,
            // but that word is a function of the launch geometry and carries nothing about any
            // payload, so it must not look like a hash somebody can compare.
            let hex = |on: bool, q: Q| -> Result<String> {
                match on {
                    true => Ok(format!("{:016x}", at(q as usize)?)),
                    false => Ok("-".to_string()),
                }
            };
            let trio = match moe {
                None => "- - -".to_string(),
                Some(c) => format!(
                    "{} {} {}",
                    hex(self.folds.bh && c.miss > 0, Q::Bh)?,
                    hex(
                        matches!(self.folds.sc, ScProbe::Full | ScProbe::Line) && c.miss > 0,
                        Q::Sc
                    )?,
                    hex(self.folds.se, Q::Se)?,
                ),
            };
            // A dense layer has no router and no pool, so its five host columns are `-` too.
            let host = match moe {
                None => "- - - - -".to_string(),
                Some(c) => format!(
                    "{:016x} {:016x} {:016x} {} {}",
                    c.gl, c.pk, c.sl, c.miss, c.reloc
                ),
            };
            use std::fmt::Write as _;
            writeln!(
                self.recs,
                "{pos} {nrow} {l} {:016x} {h} {:016x} {host} {trio}",
                at(Q::Xn as usize)?,
                at(Q::X as usize)?,
            )
            .context("divergence probe: formatting a record into the in-memory log")?;
            if let Some(slot) = self.cols.get_mut(l) {
                *slot = None;
            }
        }
        Ok(())
    }

    /// Write the log. Built whole and written once — the run is over by the time this is
    /// called, which is the point (see the header).
    pub fn write(&self, path: &str) -> Result<()> {
        let mut out = format!("# rivoli-divergence-folds {}\n", self.folds.label());
        // The fold config goes FIRST and on its own line, because two logs from different configs
        // must never be compared — `tests/divergence-columns.sh` refuses on it — and because the
        // heavy config is now known to SUPPRESS the defect, so a log that does not say which folds
        // produced it cannot be attributed at all.
        out.push_str(
            "# rivoli-divergence v4 pos nrow layer xn h x gl pk sl misses relocs bh sc se\n\
             # pos is the PASS's FIRST ROW and nrow is how many it carried: (pos=k, nrow=1) is \
             token k in decode, (pos=k, nrow=2) is a layer-major prefill row-block. v2 added \
             nrow because pos alone made those two indistinguishable\n\
             # `-` means NOT MEASURED, never zero: a dense layer has no h and no router, and a \
             layer with no miss has no bh/sc (se is folded on every MoE layer)\n\
             # bh = bounce arena after the NVMe read; sc = the pool slot just copied; se = ALL \
             the slots this layer used, at end of layer\n\
             # DECISION RULE, CROSS-RUN and never within-run (sc folds 1 slot, se folds ~9, so \
             they are different quantities): h_A!=h_B is the trigger; then bh differs -> the \
             DRIVE; bh equal and sc differs -> the COPY; bh and sc equal and se differs -> a slot \
             was wrong AT REST and not the one just copied (a resident expert, or a write after \
             the copy); all three equal -> the kernel read a slot BEFORE it landed, a \
             ticket/timeline ordering failure\n\
             # diff two runs: first differing LINE is the coordinate, first differing \
             COLUMN names the mechanism\n",
        );
        out.push_str(&self.recs);
        std::fs::write(path, out).with_context(|| format!("write {path}"))?;
        tracing::info!(
            "wrote {} divergence records to {path}",
            self.recs.lines().count()
        );
        Ok(())
    }
}
