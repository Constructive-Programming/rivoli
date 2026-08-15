//! The `--trace` v2 access-trace sink: its window width, the caller-side inputs, the
//! header the file opens with, and the three [`RoutedPool`] methods that drive it.
//!
//! Split out of `routed.rs` when that file crossed the 800-line soft cap
//! (`crates/cli/build.rs`); the cut is by cohesion, not by size. Everything a capture
//! touches is here and nothing else is, so the format — which `bin/replay` parses and
//! which no longer exists once a trace is written — can be read in one place. The pool's
//! `trace` field stays with the pool because its lifetime is the pool's.
//!
//! Not to be confused with the `trace` *feature*, which gates the slot poisoning and the
//! read-before-write localiser in `routed.rs`. This sink is a runtime flag; that is a
//! build-time instrument, and the two are deliberately independent.

use super::{RoutedPool, Selection, expert_key};
use anyhow::{Context, Result};

/// Width of the trace-v2 candidate window: the top-W router candidates recorded per
/// routing decision, on top of the `top_k` that actually ran. W bounds the largest M
/// the offline (J, M) substitution grid in the old tree's cache-conditional-routing
/// investigation can explore — an M wider than this cannot be evaluated from a captured
/// trace without recapturing. 32 is 4× `top_k` (8) and an eighth of `n_experts` (256):
/// far past any M where promoting a resident-but-lower-ranked expert is still
/// defensible, and only ~380 bytes a line.
pub const TRACE_WINDOW: usize = 32;

/// The trace sink's inputs: the ranked top-[`TRACE_WINDOW`] candidate expert ids and the
/// full per-expert `choice` array they index into. Pass [`RankWindow::OFF`] when not
/// tracing — nothing else reads them.
#[derive(Clone, Copy)]
pub struct RankWindow<'a> {
    pub window: &'a [usize],
    pub choice: &'a [f32],
}

impl RankWindow<'_> {
    /// The not-tracing value: both slices empty, so the sink writes nothing.
    pub const OFF: RankWindow<'static> = RankWindow {
        window: &[],
        choice: &[],
    };
}

/// Open the `--trace` sink and write the version header. The header is deliberately
/// unparseable as data: `replay` reads each line for whitespace-separated u32s and drops
/// the empty ones, so this line contributes nothing and a v2 trace replays through a v1
/// reader byte-identically.
pub(super) fn open_trace(path: &str, top_k: usize) -> Result<std::io::BufWriter<std::fs::File>> {
    use std::io::Write;
    let mut w = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("open trace {path}"))?,
    );
    writeln!(w, "# rivoli-trace v2 top_k={top_k} window={TRACE_WINDOW}").context("write trace")?;
    Ok(w)
}

impl RoutedPool {
    /// Is the `--trace` sink on? The layer loop gates the candidate-window `topk_into`
    /// on this so a non-tracing decode pays literally nothing for trace v2.
    pub fn tracing(&self) -> bool {
        self.trace.is_some()
    }

    /// Flush the trace sink. Called per token, because the trace CANNOT rely on
    /// `BufWriter`'s `Drop`: `Drop` discards flush errors, so a wedged or ENOSPC run
    /// would leave a silently short capture with a clean exit code. A trace is ~30
    /// minutes of sole-tenant GPU time; losing it quietly is far worse than one `write`
    /// per token. Errors propagate here, unlike in `Drop`.
    pub fn flush_trace(&mut self) -> Result<()> {
        if let Some(w) = &mut self.trace {
            use std::io::Write;
            w.flush().context("flush trace")?;
        }
        Ok(())
    }

    /// The `--trace` v2 sink: the demand keys this layer looks up, then `|`, then the
    /// top-[`TRACE_WINDOW`] candidates as `key:choice`.
    ///
    /// BOTH lists are in router RANK order, and that is LOAD-BEARING, not incidental.
    /// `sel` and `window` both come out of `topk_into` over the same `choice` buffer
    /// with the same comparator (value-desc, index-asc), and `topk_into` finishes with
    /// a full sort — so `window[..sel.len()] == sel` element for element, and
    /// `bin/replay` hard-fails a trace where that prefix does not hold. Reordering
    /// `sel` for any local reason (coalescing reads by expert id, say) would silently
    /// change the meaning of every captured trace. The debug_assert is the tripwire.
    pub(super) fn write_trace(&mut self, sel: Selection<'_>, win: RankWindow<'_>) -> Result<()> {
        debug_assert!(
            win.window.is_empty() || win.window.starts_with(sel.experts),
            "trace v2: the candidate window must be the ranking that produced `sel`"
        );
        let Some(w) = &mut self.trace else {
            return Ok(());
        };
        use std::io::Write;
        for (j, &e) in sel.experts.iter().enumerate() {
            if j > 0 {
                write!(w, " ").context("write trace")?;
            }
            write!(w, "{}", expert_key(sel.layer, e)).context("write trace")?;
        }
        // ponytail: the `choice` values have no consumer yet — the (J, M) grid needs
        // only the RANK order, which the list already carries. Written anyway because a
        // capture is GPU-gated, sole-tenant and ~30 minutes, so these few bytes are
        // cheap now and unrecoverable later without another capture; and the deferred
        // route-KL counter needs the mass distribution.
        write!(w, " |").context("write trace")?;
        for &e in win.window {
            write!(w, " {}:{:.6}", expert_key(sel.layer, e), win.choice[e])
                .context("write trace")?;
        }
        writeln!(w).context("write trace")
    }
}
