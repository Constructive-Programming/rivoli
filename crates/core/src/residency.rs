//! The residency contract — P6 as a function signature.
//!
//! **The pin is a function of free memory, never of architecture.** The old tree learned
//! this the expensive way: a dense model's pin was built unconditional on the ground that
//! "a dense model has nothing to stream", and the owner caught it as a category error —
//! dense is MORE streaming-bound than MoE (every weight is read every token; resident
//! fraction × bandwidth IS the tok/s; the pin measured worth 5.1×). So there is exactly
//! one author for the placement decision, and its signature has **no architecture
//! parameter to consult**. Per-model pins hold different tensors; which of them are
//! resident is decided here and only here.
//!
//! Everything is plain data in and plain data out — the engine executes the returned
//! [`Partition`]; this module never sees a pointer, a stream, or a weight format.

use std::num::NonZeroU64;

/// Opaque handle to one streamable unit (an expert, a layer, a tensor — the caller
/// decides the granularity and keeps the mapping). Core never learns what it names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct UnitId(pub u32);

/// A byte count. A newtype because the old tree passed bare `usize` byte counts beside
/// element counts and slot indices, and one transposition survived review.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Bytes(pub u64);

/// One placeable unit: an identity and a size. Sizes are nonzero by construction — a
/// zero-byte unit is a bookkeeping error upstream, not a placement decision.
#[derive(Debug, Clone, Copy)]
pub struct Unit {
    pub id: UnitId,
    pub bytes: NonZeroU64,
}

/// What the budget must cover before any weight is pinned. All four are charges the run
/// pays whether or not a single unit is resident: the always-resident set (embeddings,
/// norms, whatever the model cannot stream), KV at the configured max context, scratch,
/// and the streaming slots themselves.
#[derive(Debug, Clone, Copy, Default)]
pub struct Floor {
    pub always_resident: Bytes,
    pub kv_at_max_ctx: Bytes,
    pub scratch: Bytes,
    pub slot_bytes: Bytes,
}

impl Floor {
    pub fn total(&self) -> Bytes {
        Bytes(self.always_resident.0 + self.kv_at_max_ctx.0 + self.scratch.0 + self.slot_bytes.0)
    }
}

/// Why a budget was refused. The numbers ride in the type so the message at the boundary
/// can name them — "refused" without the arithmetic is the unhelpful half of a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// `free` cannot even cover the floor: nothing can be pinned and the streaming path
    /// itself cannot run. The caller refuses the run; it does not degrade.
    BelowFloor { need: Bytes, have: Bytes },
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Refusal::BelowFloor { need, have } => write!(
                f,
                "budget below floor: the run needs {} bytes before any weight is pinned \
                 and has {}",
                need.0, have.0
            ),
        }
    }
}

/// The placement decision: `pinned` is a PREFIX of the caller's `ordered` list, `streamed`
/// is the rest. Prefix-ness is the load-bearing shape — it is what makes the decision
/// monotone in `free` (more memory can only extend the pin, never reshuffle it), and it
/// is what lets a dense cyclic model's optimal policy (a static prefix partition — the
/// Belady degenerate) fall out as the same code path rather than a bolted-on special case.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Partition {
    pub pinned: Vec<UnitId>,
    pub streamed: Vec<UnitId>,
}

/// THE P6 function. `ordered` is the caller's priority order (highest value first — for a
/// cyclic dense model, layer order; for MoE, whatever the residency evidence says);
/// `free` is what the machine has at run time; `floor` is what must be paid first.
///
/// All-resident is the degenerate happy case: when everything fits, `streamed` is empty
/// and the streaming path idles — never a separate resident-only design (P1).
pub fn partition(ordered: &[Unit], free: Bytes, floor: Floor) -> Result<Partition, Refusal> {
    let need = floor.total();
    let Some(mut headroom) = free.0.checked_sub(need.0) else {
        return Err(Refusal::BelowFloor { need, have: free });
    };
    let mut pinned = Vec::new();
    let mut streamed = Vec::new();
    let mut still_pinning = true;
    for u in ordered {
        // First unit that does not fit ends the pin: a GAP in the pin (skipping a big
        // unit to pin a later small one) would break prefix-ness and with it monotonicity
        // — the old tree's eviction hints died of exactly that kind of second authority.
        if still_pinning && u.bytes.get() <= headroom {
            headroom -= u.bytes.get();
            pinned.push(u.id);
        } else {
            still_pinning = false;
            streamed.push(u.id);
        }
    }
    Ok(Partition { pinned, streamed })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)] // tests: panic-on-failure is the idiom

    use super::*;

    fn units(sizes: &[u64]) -> Vec<Unit> {
        sizes
            .iter()
            .enumerate()
            .map(|(i, &b)| Unit {
                id: UnitId(u32::try_from(i).unwrap()),
                bytes: NonZeroU64::new(b).unwrap(),
            })
            .collect()
    }

    const FLOOR: Floor = Floor {
        always_resident: Bytes(10),
        kv_at_max_ctx: Bytes(20),
        scratch: Bytes(5),
        slot_bytes: Bytes(15),
    };

    /// INV-1: the pin is a function of `(ordered, free, floor)` ONLY — P6 as a gate.
    /// The signature admits no architecture parameter, so the property provable here is
    /// the surviving one: same inputs, same partition, and the partition is monotone in
    /// `free` — more memory only ever EXTENDS the pinned prefix. This is the invariant
    /// whose violation was the old tree's dense-pin category error.
    #[test]
    fn inv_1_the_pin_is_monotone_in_free_memory_and_nothing_else() {
        let us = units(&[100, 50, 200, 25]);
        let mut prev_len = 0;
        for free in (50..=500).step_by(7) {
            let p = partition(&us, Bytes(free), FLOOR).unwrap();
            // Prefix-ness: pinned ++ streamed is exactly the input order.
            let mut all = p.pinned.clone();
            all.extend(&p.streamed);
            assert_eq!(all, us.iter().map(|u| u.id).collect::<Vec<_>>());
            // Monotone: a bigger budget never pins fewer.
            assert!(p.pinned.len() >= prev_len, "pin shrank as free grew");
            prev_len = p.pinned.len();
        }
        // Degenerate top (P1): everything fits ⇒ nothing streams.
        let top = partition(&us, Bytes(1_000_000), FLOOR).unwrap();
        assert!(top.streamed.is_empty());
        assert_eq!(top.pinned.len(), 4);
    }

    #[test]
    fn below_floor_refuses_with_the_arithmetic_in_the_message() {
        let err = partition(&units(&[10]), Bytes(49), FLOOR).unwrap_err();
        assert_eq!(
            err,
            Refusal::BelowFloor {
                need: Bytes(50),
                have: Bytes(49)
            }
        );
        let msg = err.to_string();
        assert!(msg.contains("50") && msg.contains("49"), "message: {msg}");
    }

    #[test]
    fn the_first_unit_that_does_not_fit_ends_the_pin() {
        // 60 headroom after the 50-byte floor at free=110: unit0 (40) pins, unit1 (30)
        // does NOT fit the remaining 20 — and unit2 (10), which would fit, streams
        // anyway, because a gapped pin is a reshuffled pin.
        let p = partition(&units(&[40, 30, 10]), Bytes(110), FLOOR).unwrap();
        assert_eq!(p.pinned, vec![UnitId(0)]);
        assert_eq!(p.streamed, vec![UnitId(1), UnitId(2)]);
    }
}
