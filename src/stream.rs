//! The streaming spine: expert slabs flow NVMe → host slab pool → decode,
//! with bounded channels (backpressure by construction) and prefetch driven
//! by the pilot predictor. Decode itself stays synchronous; async owns the
//! FEED side only — that's where colibri serialized and starved.

use anyhow::Result;
use tokio::sync::mpsc;

/// Identifies one expert's weights: (layer, expert id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertId {
    pub layer: u16,
    pub expert: u16,
}

/// A decode-ready expert slab (int4 weights + scales, colibri packing).
#[derive(Debug)]
pub struct ExpertSlab {
    pub id: ExpertId,
    pub bytes: bytes::Bytes,
}

/// Where an expert's weights currently live. The engine asks; the feed answers
/// without blocking — `Cold` means a fetch was enqueued and the caller should
/// overlap other work (GPU launches, resident experts) before polling again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// In the device tier — computed by the GPU path.
    Device,
    /// Pinned or cached in host RAM — computed by the CPU pool.
    Host,
    /// Not resident; fetch enqueued to the pread pool.
    Cold,
}

/// Bounded feed of cold experts. `cap` slabs in flight caps memory and gives
/// backpressure for free; no unbounded queue can OOM the box.
pub struct ExpertFeed {
    pub tx: mpsc::Sender<ExpertId>,
    pub rx: mpsc::Receiver<ExpertSlab>,
}

impl ExpertFeed {
    /// Wire a request channel to a slab-delivery channel through `workers`
    /// blocking pread tasks. Skeleton: the pread pool lands in M4 (PLAN.md);
    /// until then this is the shape the engine codes against.
    pub fn new(cap: usize) -> (Self, mpsc::Receiver<ExpertId>, mpsc::Sender<ExpertSlab>) {
        let (req_tx, req_rx) = mpsc::channel(cap);
        let (slab_tx, slab_rx) = mpsc::channel(cap);
        (
            Self {
                tx: req_tx,
                rx: slab_rx,
            },
            req_rx,
            slab_tx,
        )
    }

    /// Enqueue a prefetch; ok to drop on a full queue (prefetch is advisory —
    /// the demand path re-requests on actual miss).
    pub fn prefetch(&self, id: ExpertId) -> Result<()> {
        let _ = self.tx.try_send(id);
        Ok(())
    }
}
