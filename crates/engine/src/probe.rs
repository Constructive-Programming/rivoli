//! `--divergence-log`: localise a run-to-run divergence to a (position, layer, QUANTITY)
//! coordinate, without perturbing what it measures.
//!
//! ## Why an instrument was needed at all
//!
//! GLM int3-vq greedy decode does not reproduce itself: two runs at identical flags on the
//! same artifact are byte-identical over 32 tokens and differ over 512 (61 of 512 ids on a
//! quiet box, 496 of 512 under CPU/NFS load — the onset moves with machine load, the defect
//! does not). The generated text cannot localise it, because ONE changed token rewrites the
//! whole tail: an output diff says only "somewhere at or before the first difference".
//!
//! ## What makes it a DISCRIMINATOR rather than a detector
//!
//! Each layer folds three quantities into their own slots, and they cut the layer at the two
//! seams the two standing hypotheses sit on either side of:
//!
//! | slot | quantity | what a difference HERE, with the earlier slots equal, means |
//! |---|---|---|
//! | [`Q::Xn`] | `xn`, the post-attention rmsnorm output | attention (or its KV cache) diverged; the MLP has not run yet |
//! | [`Q::H`] | `moe_hidden`, the SwiGLU intermediate | `xn` was equal, so the GATE/UP expert BYTES (or that kernel) diverged — the routed-pool hypothesis |
//! | [`Q::X`] | the residual at layer exit | `xn` and `h` were equal, so the fault is in the DOWN projection, the fixed-point accumulator or the drain — the MoE-accumulation hypothesis |
//!
//! Beside them, per MoE layer, six host-side columns that cost no device traffic at all
//! because routing is already a host function of already-host data ([`Cols`]): what the
//! router SAW, what it PICKED, the weights it produced, WHERE the pool put each expert, and
//! the layer's miss and relocation counts. So a divergence coordinate arrives next to what
//! the fetch path did at that coordinate, which is the only form in which it is actionable.
//!
//! Read a pair of logs by diffing them: the first differing LINE is the coordinate, and the
//! first differing COLUMN on that line names the mechanism.
//!
//! ## Two properties this file exists to preserve
//!
//! **The fold is XOR, and that is load-bearing, not a style choice.** XOR is commutative and
//! associative, so `hash_rows`' atomics give a bit-identical result whatever order they land
//! in. A float sum would be neither and would report a difference from scheduling jitter
//! alone — an instrument noisier than its subject measures nothing. (`hash_rows` mixes the
//! element INDEX in before splitmix64's finalizer for the other half of the same argument:
//! XOR is self-inverse, so two elements holding the same bit pattern would otherwise cancel
//! out of the fold, and a permutation would hash identically.)
//!
//! **Nothing here touches the device or the disk between the first fold and the last token.**
//! The old tree's predecessor copied the residual to the HOST every layer, and a
//! configuration that reproduced its fault without the probe produced a clean run with it —
//! the tool built for the bug could not be pointed at it. So: the folds stay on the device,
//! all `n_layers * NQ` slots come back in ONE D2H at a point the end-of-layer `device_sync`
//! has already idled, and the records are held in memory and written after the run.
//!
//! That is also why this is its own feature rather than part of `trace`: `trace` adds a
//! poison-fill and a `device_sync` per layer-with-misses, which is exactly the class of
//! perturbation recorded to hide this fault. **Never debug this defect under `--trace`, and
//! never accept a green obtained with tracing enabled as evidence of anything.**

#![cfg(feature = "rocm")]

// Gated INSIDE rather than at the module, on the pattern `k3.rs` and `v4.rs` use and for the
// same reason: the three folds at the bottom of this file are pure arithmetic over plain
// data, and they are the half where a wrong answer is silent. `fold_host` is the oracle
// `crates/engine/tests/fwd_kernel.rs` scores the `hash_rows` KERNEL against, so it must exist
// in a build that runs the default device suite — otherwise the instrument the whole
// investigation rests on is checked only by whoever remembers to pass its feature.
#[cfg(feature = "corruption-probe")]
use crate::device::DeviceBuf;
#[cfg(feature = "corruption-probe")]
use anyhow::{Context, Result};
#[cfg(feature = "corruption-probe")]
use rivoli_backend::launch_hash_rows;

/// The device-folded quantities. The ORDER is the cut described in this module's header —
/// `Xn` before `H` before `X` is "before the MLP", "after gate/up", "after down + drain" —
/// so a reader of a log line sees the layer's causal order left to right.
#[cfg(feature = "corruption-probe")]
#[derive(Clone, Copy)]
pub enum Q {
    /// `xn`: the MoE's input, after the post-attention rmsnorm.
    Xn = 0,
    /// `moe_hidden`: the SwiGLU intermediate, after gate/up and before down.
    H = 1,
    /// The residual stream at layer exit.
    X = 2,
}

/// Device u64 fold slots per layer — one per [`Q`].
#[cfg(feature = "corruption-probe")]
const NQ: usize = 3;

/// One MoE layer's host-side columns. Every field is a fold of data the decode thread
/// already holds, so recording them costs no device traffic and no I/O.
#[cfg(feature = "corruption-probe")]
#[derive(Clone, Copy, Default)]
pub struct Cols {
    /// FNV-1a over the gate logits' exact BYTES — bit pattern, never value, so a one-ulp
    /// move is visible. This is what the router SAW.
    pub gl: u64,
    /// FNV-1a over row 0's picks, in rank order. With `gl` beside it this answers INV-1
    /// directly: equal `gl` and unequal `pk` would mean routing consulted something outside
    /// its inputs.
    pub pk: u64,
    /// FNV-1a over the `[descriptor][row]` routing-weight matrix's bytes.
    pub wx: u64,
    /// FNV-1a over each selected expert's SLOT OFFSET in the pool — a pool-relative
    /// number, never an address, because the VMM base differs between runs and hashing it
    /// would report a difference on every line.
    pub sl: u64,
    /// Cold reads this layer submitted.
    pub miss: u64,
    /// Arena compaction relocations this layer executed.
    pub reloc: u64,
}

/// The `--divergence-log` state: the device fold slab, and the records so far.
#[cfg(feature = "corruption-probe")]
pub struct Probe {
    /// `[n_layers][NQ]` device u64. Zeroed at construction and re-zeroed by every
    /// [`Probe::drain`] — `hipMalloc` does not zero and the fold is an XOR, so an unzeroed
    /// slab would corrupt exactly the first token's hashes, the one token an investigation
    /// into onset position least wants to distrust.
    dev: DeviceBuf,
    host: Vec<u8>,
    zeros: Vec<u8>,
    n_layers: usize,
    /// Per-layer host columns for the pass in flight; `None` on a dense layer, which has no
    /// router and no pool.
    cols: Vec<Option<Cols>>,
    /// One formatted line per (pos, layer). Held in memory: see the header for why the write
    /// cannot happen during the run.
    recs: Vec<String>,
}

#[cfg(feature = "corruption-probe")]
impl Probe {
    pub fn new(n_layers: usize) -> Result<Self> {
        let bytes = n_layers * NQ * 8;
        let mut dev = DeviceBuf::new(bytes)?;
        dev.copy_in_at(0, &vec![0u8; bytes])?;
        Ok(Self {
            dev,
            host: vec![0u8; bytes],
            zeros: vec![0u8; bytes],
            n_layers,
            cols: vec![None; n_layers],
            recs: Vec::new(),
        })
    }

    /// XOR-fold `n` device f32 at `x` into `(layer, q)`'s slot. No sync, no D2H.
    ///
    /// # Safety
    /// `x` must be `n` live device f32 whose writers have retired.
    pub unsafe fn fold(&mut self, q: Q, layer: usize, x: *const f32, n: usize) -> Result<()> {
        // A layer outside the slab would fold into another layer's slot and report a
        // difference that never happened — the worst failure mode an instrument has. Refused
        // with both numbers rather than clamped.
        anyhow::ensure!(
            layer < self.n_layers,
            "divergence probe: layer {layer} outside a {}-layer slab",
            self.n_layers
        );
        let slot = layer * NQ + q as usize;
        // SAFETY: `slot < n_layers * NQ` by the check above, so the offset is inside the
        // slab; `x`/`n` are the caller's contract, forwarded.
        unsafe { launch_hash_rows(x, n, (self.dev.ptr_mut() as *mut u64).add(slot)) }
    }

    /// Record a MoE layer's host columns for the pass in flight.
    pub fn set_cols(&mut self, layer: usize, c: Cols) {
        if let Some(slot) = self.cols.get_mut(layer) {
            *slot = Some(c);
        }
    }

    /// Drain `layers`' folds in ONE D2H, emit one record per layer, and re-zero the slab.
    ///
    /// Called once per pass — which is once per token in decode and once per (layer, rows)
    /// pass under layer-major prefill — at a point the end-of-layer `device_sync` has
    /// already idled the device, so it adds no barrier of its own.
    pub fn drain(&mut self, pos: usize, layers: std::ops::Range<usize>) -> Result<()> {
        self.dev.copy_out_into(&mut self.host)?;
        self.dev.copy_in_at(0, &self.zeros)?;
        for l in layers {
            let at = |q: usize| -> u64 {
                let o = (l * NQ + q) * 8;
                self.host
                    .get(o..o + 8)
                    .and_then(|b| b.try_into().ok())
                    .map_or(0, u64::from_le_bytes)
            };
            let c = self.cols.get(l).copied().flatten();
            // A dense layer has no router and no pool, so its six host columns are `-`
            // rather than 0: a zero would read as "the router picked nothing", which is a
            // different claim and one a diff cannot tell from the truth.
            let host = c.map_or_else(
                || "- - - - - -".to_string(),
                |c| {
                    format!(
                        "{:016x} {:016x} {:016x} {:016x} {} {}",
                        c.gl, c.pk, c.wx, c.sl, c.miss, c.reloc
                    )
                },
            );
            self.recs.push(format!(
                "{pos} {l} {:016x} {:016x} {:016x} {host}",
                at(Q::Xn as usize),
                at(Q::H as usize),
                at(Q::X as usize),
            ));
            if let Some(slot) = self.cols.get_mut(l) {
                *slot = None;
            }
        }
        Ok(())
    }

    /// Write the log. Built whole and written once — the run is over by the time this is
    /// called, which is the point (see the header).
    pub fn write(&self, path: &str) -> Result<()> {
        let mut out = String::from(
            "# rivoli-divergence v1 pos layer xn h x gl pk wx sl misses relocs\n\
             # diff two runs: first differing LINE is the coordinate, first differing \
             COLUMN names the mechanism\n",
        );
        for r in &self.recs {
            out.push_str(r);
            out.push('\n');
        }
        std::fs::write(path, out).with_context(|| format!("write {path}"))?;
        tracing::info!("wrote {} divergence records to {path}", self.recs.len());
        Ok(())
    }
}

/// The HOST twin of the `hash_rows` kernel — one element's contribution to the XOR fold.
///
/// Public, and the reason is P7: `crates/engine/tests/fwd_kernel.rs` scores the kernel
/// against a fold built from this, so the instrument the whole investigation rests on has an
/// oracle of its own. An instrument nobody checked is a source of confident wrong answers.
pub fn fold_step(i: usize, bits: u32) -> u64 {
    // Element 0 holding +0.0 folds to 0 and so contributes nothing, because 0 is a fixed
    // point of the finalizer (`rivoli_core::hash`, asserted there). Exactly one (index, bits)
    // pair has that property, so it costs one collision out of 2^64 and no accuracy the
    // instrument depends on — recorded because it looks like a bug and is not.
    rivoli_core::hash::splitmix_finalize(((i as u64) << 32) ^ u64::from(bits))
}

/// The whole host-side fold of `x` — what a correct `hash_rows` over the same array must
/// produce, bit for bit.
pub fn fold_host(x: &[f32]) -> u64 {
    x.iter()
        .enumerate()
        .fold(0u64, |h, (i, v)| h ^ fold_step(i, v.to_bits()))
}
