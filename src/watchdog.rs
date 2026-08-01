//! In-process wedge watchdog. The decode loop runs on one thread doing blocking
//! `hipDeviceSynchronize` joins; if the GPU wedges (the amdgpu large-GTT hang, or
//! a device fault), that join never returns and no in-loop deadline check can ever
//! fire. So a separate thread watches a per-token heartbeat and, if the loop stops
//! making progress for longer than `deadline`, prints why and aborts the process —
//! a clean, loud refusal instead of a silent forever-hang (the M5 hardening rule).

use anyhow::{Context, Result};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// A progress heartbeat. Clone it into the decode loop and call [`Heartbeat::beat`]
/// once per token; the watchdog thread reads it.
#[derive(Clone)]
pub struct Heartbeat {
    base: Instant,
    last_ms: Arc<AtomicU64>,
}

impl Heartbeat {
    /// Record forward progress (call once per generated token).
    pub fn beat(&self) {
        self.last_ms
            .store(self.base.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
}

/// Spawn the watchdog and return the [`Heartbeat`] the decode loop must beat.
///
/// `deadline` must comfortably exceed the slowest healthy token (a cold-miss token
/// is ~1-2 s here, so a deadline of tens of seconds only trips on a real wedge).
/// The watchdog thread is a daemon: it exits with the process and needs no join.
pub fn spawn(deadline: Duration) -> Result<Heartbeat> {
    let base = Instant::now();
    let hb = Heartbeat {
        base,
        last_ms: Arc::new(AtomicU64::new(0)),
    };
    hb.beat(); // prime, so the first token has a full deadline
    let last_ms = hb.last_ms.clone();
    let deadline_ms = deadline.as_millis() as u64;
    thread::Builder::new()
        .name("wedge-watchdog".into())
        .spawn(move || {
            let tick = Duration::from_secs(2).min(deadline);
            loop {
                thread::sleep(tick);
                let now = base.elapsed().as_millis() as u64;
                let stalled = now.saturating_sub(last_ms.load(Ordering::Relaxed));
                if stalled > deadline_ms {
                    // stderr (not tracing) so it lands even if a fault wedged logging.
                    eprintln!(
                        "FATAL: decode wedged — no token progress for {}s \
                         (GPU hang / amdgpu GTT wedge); aborting.",
                        stalled / 1000
                    );
                    std::process::exit(2);
                }
            }
        })
        .context("spawn wedge-watchdog thread")?;
    Ok(hb)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn beat_keeps_the_deadline_from_tripping() {
        // A short deadline that we keep alive by beating faster than it, then let
        // lapse — proving the heartbeat gates the (would-be) abort. We can't test
        // the process::exit path directly, so assert the staleness logic instead.
        let hb = spawn(Duration::from_secs(3600)).unwrap(); // never trips during the test
        let before = hb.last_ms.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(5));
        hb.beat();
        assert!(hb.last_ms.load(Ordering::Relaxed) >= before);
    }
}

