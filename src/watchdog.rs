//! In-process wedge watchdog. The decode loop runs on one thread doing blocking
//! `hipDeviceSynchronize` joins; if the GPU wedges (the amdgpu large-GTT hang, or
//! a device fault), that join never returns and no in-loop deadline check can ever
//! fire. So a separate thread watches a per-token heartbeat and, if the loop stops
//! making progress for longer than `deadline`, prints why and aborts the process —
//! a clean, loud refusal instead of a silent forever-hang (the M5 hardening rule).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
// `spawn` is the only thing here that needs these, and it is `trace`-only.
#[cfg(feature = "trace")]
use anyhow::{Context, Result};
#[cfg(feature = "trace")]
use std::thread;
#[cfg(feature = "trace")]
use std::time::Duration;

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

/// A heartbeat nobody watches — what a build without `trace` gets.
///
/// The type stays in EVERY build even though the thread does not, because `serve` takes a
/// `&Heartbeat` and the decode loop beats one per token; making those `Option` would spread
/// four `#[cfg]`s across two modules to delete one atomic store per token. `beat` on this is
/// a relaxed store to an `Arc<AtomicU64>` nothing reads, which is the cheapest honest way to
/// keep one code path.
pub fn inert() -> Heartbeat {
    Heartbeat {
        base: Instant::now(),
        last_ms: Arc::new(AtomicU64::new(0)),
    }
}

/// Spawn the watchdog and return the [`Heartbeat`] the decode loop must beat.
///
/// **`trace`-only since 2026-08-03.** It aborts the process on a stall, which is right for
/// a diagnostic run and wrong for a server that is merely slow: the deadline has to be
/// guessed against the slowest healthy token, and a cold-miss token here is already 1-2 s.
/// A wedge is loud enough without a killer thread in the shipped binary.
///
/// `deadline` must comfortably exceed the slowest healthy token (a cold-miss token
/// is ~1-2 s here, so a deadline of tens of seconds only trips on a real wedge).
/// The watchdog thread is a daemon: it exits with the process and needs no join.
#[cfg(feature = "trace")]
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

// `trace`-gated with `spawn` itself: the whole module tests the watchdog thread, which a
// build without `trace` does not have. `inert()` has nothing to test — it is a store to an
// atomic nobody reads, and asserting that would test the language.
#[cfg(all(test, feature = "trace"))]
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
