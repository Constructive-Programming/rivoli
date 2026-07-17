//! Expert usage ranking. `<snapshot>/.coli_usage` is plain text, one line per
//! observed (layer, expert): `<layer> <expert> <count>`. The counts drive the
//! pin: hottest experts become resident, so ~95% of activations hit RAM/device
//! instead of NVMe. Format is shared with colibri — the file transfers directly.

use anyhow::{Context, Result};
use std::collections::HashMap;

/// Accumulated selection counts, keyed by (layer, expert).
#[derive(Debug, Default)]
pub struct Usage {
    pub counts: HashMap<(u16, u16), u64>,
}

impl Usage {
    /// Load `<dir>/.coli_usage`. A missing file is not an error — it means no
    /// history yet (cold start), so the pin falls back to natural order.
    pub fn load(dir: &str) -> Result<Self> {
        let path = format!("{dir}/.coli_usage");
        let text = match std::fs::read_to_string(&path) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Self::default()),
            Err(e) => return Err(e).with_context(|| format!("read {path}")),
        };
        let mut counts = HashMap::new();
        for line in text.lines() {
            let mut it = line.split_ascii_whitespace();
            let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) else {
                continue; // tolerate blank/short lines
            };
            if let (Ok(l), Ok(e), Ok(c)) = (l.parse::<u16>(), e.parse::<u16>(), c.parse::<u64>()) {
                *counts.entry((l, e)).or_insert(0) += c;
            }
        }
        Ok(Self { counts })
    }

    /// (layer, expert) pairs ranked hottest-first — the pin fill order. Returns
    /// the counts too; the pin-fill loop ignores them. One allocation, no
    /// second collect just to drop the count.
    pub fn ranked(&self) -> Vec<((u16, u16), u64)> {
        let mut v: Vec<_> = self.counts.iter().map(|(&k, &c)| (k, c)).collect();
        // Descending by count; (layer, expert) as a stable tiebreak so the
        // ranking is deterministic across runs (no Math.random-style drift).
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v
    }

    pub fn total_selections(&self) -> u64 {
        self.counts.values().sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_hottest_first_deterministically() {
        let mut u = Usage::default();
        u.counts.insert((3, 5), 100);
        u.counts.insert((3, 1), 500);
        u.counts.insert((4, 0), 500); // tie with (3,1) → lower key first
        let r = u.ranked();
        assert_eq!(r[0].0, (3, 1));
        assert_eq!(r[1].0, (4, 0));
        assert_eq!(r[2].0, (3, 5));
        assert_eq!(u.total_selections(), 1100);
    }
}
