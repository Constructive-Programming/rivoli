//! DSA lightning-indexer constants shared by the host selection path (gpu.rs
//! `dsa_select_layer`) and the device kernels (indexer.hip). The indexer itself
//! runs on device; this module only pins the two magic numbers both sides must
//! agree on.

/// MISA router block size (pooled tokens per running-mean key). MUST match
/// `MISA_BLOCK` in kernels/indexer.hip — the block pool the head router scores over.
pub const MISA_BLOCK: usize = 1024;

/// LayerNorm epsilon for the indexer's `k_norm` (which, unlike every model RMSNorm,
/// ships a bias). Matches the HF reference; distinct from `cfg.rms_norm_eps`.
pub const K_NORM_EPS: f32 = 1e-6;

/// Per-full-layer bookkeeping for the 1-step-stale selection (`--stale-sel`, M1a of
/// docs/investigations/npu-offload.md). The selections themselves are device buffers in
/// `gpu.rs`; this holds the one host-side fact per indexer slab — how many rows the
/// selection stored at the previous token has, or `None` when the previous token stored
/// nothing (it was at or below `index_topk`, where the indexer computes no selection and
/// there is nothing to be stale about).
///
/// Split out of `DeviceIndexer` so the lag-by-one contract — what token `t` SERVES is what
/// token `t-1` COMPUTED — is a pure state machine a host test can walk without a device.
/// The decisions this type encodes, and their reasons:
///
/// - **`None` serves dense over the whole prefix.** The first token past `index_topk` has
///   no stale selection to consume. The real decoupled design (npu-offload.md, window 2)
///   would attend dense there too, and can afford to: dense needs no selection at all, and
///   at the crossing token it costs one row more than a top-k attend. Dense is also EXACT,
///   so this choice cannot flatter the stale arm past that single token.
/// - **`Some(nr)` serves exactly `nr` stored rows.** No union with the current position:
///   a stale selection was scored before the current token's key existed, so serving it
///   verbatim (the diagonal possibly absent) is precisely the approximation M1a exists to
///   price. Patching the diagonal in is a *different, better* variant — measure it only if
///   the verbatim form fails the quality gate, and as its own arm.
///
/// UNGATED on purpose, unlike every other piece of `stale-sel`: pure host code whose test
/// must run in the featureless CI job — a host module gated on the feature is compiled
/// exactly as often as someone names it, which is the recorded `otlp` rot (CLAUDE.md).
pub struct StaleShare {
    nr: Vec<Option<usize>>,
}

impl StaleShare {
    pub fn new(n_slabs: usize) -> Self {
        Self {
            nr: vec![None; n_slabs],
        }
    }

    /// Row count of the selection stored for `slab` at the previous token, or `None` if
    /// that token stored nothing. Read BEFORE `store` overwrites it — same token order as
    /// the device side, where the D2D copy that serves the old selection is enqueued
    /// before `index_topk` overwrites the buffer it reads from.
    pub fn stored(&self, slab: usize) -> Option<usize> {
        self.nr[slab]
    }

    /// Record that this token's fresh selection (`nr` rows) is now in `slab`'s buffer,
    /// to be served at the next token.
    pub fn store(&mut self, slab: usize, nr: usize) {
        self.nr[slab] = Some(nr);
    }
}

#[cfg(test)]
mod tests {
    use super::StaleShare;

    /// The lag-by-one contract, walked in the same read-then-store order as
    /// `dsa_select_layer` drives it: the crossing token finds nothing stored, every later
    /// token sees exactly what the previous one stored. The stored value is `nt` itself —
    /// NOT a realistic row count (the engine always stores `topk`) — because a constant
    /// would pass equally for serve-current, serve-two-back or a stuck value, and an
    /// assertion that cannot go red gates nothing.
    #[test]
    fn stale_share_serves_the_previous_tokens_selection() {
        let mut s = StaleShare::new(2);
        // Slab 1 stays untouched throughout — per-layer state must not bleed across slabs.
        for nt in 5..=8 {
            let served = s.stored(0);
            match nt {
                5 => assert_eq!(served, None, "crossing token must find nothing stored"),
                _ => assert_eq!(served, Some(nt - 1), "token nt={nt} must see nt-1's store"),
            }
            s.store(0, nt);
            assert_eq!(s.stored(1), None, "slab 1 was never stored to");
        }
    }
}
