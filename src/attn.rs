//! Attention row-selection modes. The MLA absorb + flash-attend core (gpu.rs) is
//! row-set-agnostic; only which cached tokens a step attends over differs by
//! [`AttnMode`]. The DSA/MISA lightning-indexer selection itself runs on device
//! (gpu.rs `dsa_select_layer` + indexer.hip); this module holds the mode enum and
//! the position-based StreamingLLM row set.

/// Which tokens each decode step attends over. Selected once per layer per token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttnMode {
    /// Full softmax over every cached token. Exactly the trained model at
    /// ≤ index_topk context; mildly out-of-distribution beyond.
    Dense,
    /// StreamingLLM: first `sinks` tokens + last `window` tokens, position based,
    /// no weights. Bounds attention BANDWIDTH, not cache memory.
    Streaming { sinks: usize, window: usize },
    /// Native DSA: the trained lightning indexer picks top-index_topk tokens per
    /// full layer; shared layers reuse the nearest preceding full layer's selection
    /// (IndexShare). Needs the resident indexer weights.
    Dsa,
    /// DSA with MISA head routing (arXiv 2605.07363): only `active_heads` of the
    /// indexer heads score tokens (routed by a block-pool estimate per full layer).
    Misa { active_heads: usize },
}

/// StreamingLLM row set over `nt` cached tokens: the first `sinks` tokens plus the
/// last `window` tokens, ascending, overlap-free. Never empty for `nt ≥ 1` (a
/// zero-sink zero-window config still attends the current token — the window floor
/// is the row that was just appended).
pub fn streaming_rows(nt: usize, sinks: usize, window: usize, rows: &mut Vec<u32>) {
    rows.clear();
    let sink_end = sinks.min(nt);
    let win_start = nt.saturating_sub(window.max(1)).max(sink_end);
    rows.extend(0..sink_end as u32);
    rows.extend(win_start as u32..nt as u32);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streaming_rows_shapes() {
        let mut r = Vec::new();
        // Fewer tokens than sinks+window → everything (dense-equivalent).
        streaming_rows(5, 4, 100, &mut r);
        assert_eq!(r, vec![0, 1, 2, 3, 4]);
        // Disjoint sinks + window.
        streaming_rows(100, 4, 10, &mut r);
        assert_eq!(&r[..4], &[0, 1, 2, 3]);
        assert_eq!(&r[4..], (90u32..100).collect::<Vec<_>>().as_slice());
        // Window overlapping the sinks clips, no duplicates.
        streaming_rows(10, 8, 5, &mut r);
        assert_eq!(r, (0u32..10).collect::<Vec<_>>());
        // Degenerate zero-sink zero-window still attends the current token.
        streaming_rows(50, 0, 0, &mut r);
        assert_eq!(r, vec![49]);
    }
}
