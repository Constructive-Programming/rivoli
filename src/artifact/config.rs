//! Zero-knob auto-discovery. No environment variables, no config files: the
//! machine is measured at startup and the resolved numbers are printed as the first
//! line of every run — a benchmark whose parameters aren't in its log never
//! happened.

use anyhow::{Context, Result, bail};
use std::fmt;

/// Headroom the AUTO budget leaves free for the OS + the pinned io_uring arena,
/// bytes. With no `--max-mem`, the total device budget is `free − OS_RESERVE`; an
/// explicit `--max-mem` ignores this entirely.
///
/// 16 GiB keeps the auto-sized pool from starving the OS + the io_uring bounce arena
/// on this 124 GiB box. (The old ~92 GiB "driver durable-backing NaN cliff" that a
/// separate `MAX_BUDGET` ceiling guarded turned out to be a bug in our own code, now
/// fixed — so there is no hard footprint ceiling any more.)
pub const OS_RESERVE: u64 = 16 << 30;

/// Routed-expert format mode. The always-resident set (attention, dense MLPs, shared
/// expert) is unaffected; this only picks how the 256 routed experts/layer decode.
/// See docs/reference/modes.md for the tradeoffs.
///
/// # The default is `hybrid`, unconditionally, since 2026-08-06
///
/// It used to be backend-dependent — `hybrid` on `rocm`, `int3-vq` on `vulkan` — because the
/// Vulkan backend had no int4 expert kernels, so `validate_backend` rejected `hybrid` and a
/// bare `rivoli <model>` on a Vulkan build failed on its own default before reading a byte
/// of the artifact. With that backend retired the divergence has no second side, so the
/// `cfg_attr` pair is gone and `hybrid` is simply the default.
///
/// The reasoning is kept because it is the general rule, not a Vulkan detail: **a default
/// must be a configuration the build can actually run.** If a backend is ever added that
/// cannot do int4, this is the knob, and the note above records what the choice cost —
/// `--mode` is echoed in `Config`'s Display and in the OTLP run attributes precisely so no
/// measurement can silently attribute one mode's numbers to another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Every routed expert is int3-VQ (`.vq3`): smallest, most slots, gather-bound.
    Int3Vq,
    /// Every routed expert is int4 (`.i4`): ~1.8× faster compute, bigger, fewer slots.
    Int4,
    /// Frequent experts int4 (HOT), the rest int3-VQ (COLD). Needs both file sets; the
    /// byte-aware policy floats the split.
    #[default]
    Hybrid,
}

impl Mode {
    /// Parse the `--mode` value.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "int3-vq" => Ok(Mode::Int3Vq),
            "int4" => Ok(Mode::Int4),
            "hybrid" => Ok(Mode::Hybrid),
            other => bail!("unknown --mode {other:?} (int3-vq|int4|hybrid)"),
        }
    }
    /// The routed experts decode from int4 (int4 mode, or the hybrid HOT slab + shared).
    pub fn uses_int4(self) -> bool {
        self != Mode::Int3Vq
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Mode::Int3Vq => "int3-vq",
            Mode::Int4 => "int4",
            Mode::Hybrid => "hybrid",
        })
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the self-contained VQ artifact directory (manifest.json + codebooks
    /// + resident.safetensors + `L{ll}.vq3`). The artifact IS the model.
    pub model: String,
    /// Benchmark mode: decode this many tokens then print PROFILE and exit. None =
    /// server mode (later milestone).
    pub bench: Option<usize>,
    /// Dump the routed-expert access trace to this path (`--trace`), format v2: a
    /// `# rivoli-trace v2 top_k=<k> window=<w>` header, then one line per MoE layer —
    /// the keys it looked up, then `|`, then the top-`w` router candidates as
    /// `key:choice`. Feeds the offline `replay` cache-policy sim; the header and the
    /// `|` tail are both invisible to a v1 reader.
    pub trace: Option<String>,
    /// Override the fixed bench prompt (`--prompt`), for capturing routing traces of
    /// diverse inputs. None = the default prompt.
    pub prompt: Option<String>,
    /// Routed-expert cache policy (`--cache-policy` lru|2q|arc). Default "2q".
    /// All three are output-neutral: routing never consults residency — see
    /// docs/investigations/cache-conditional-routing.md.
    pub cache_policy: String,
    /// 2Q's A1in/A1out split, percentages of pool capacity. Ignored by `lru`/`arc`.
    ///
    /// Always [`crate::memory::cache::TwoQSplit::default`] in the engine: the `--2q-kin` /
    /// `--2q-kout` flags that fed it were deleted 2026-08-01, having never appeared in a
    /// recorded command line, a test or a script. The split is still swept — offline, by
    /// `bin/replay`'s own `--kin`/`--kout` against a trace, which is where a policy grid
    /// belongs and where it costs no GPU time.
    pub two_q: crate::memory::cache::TwoQSplit,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    pub checksum_x: bool,
    /// Routed-expert format mode (`--mode int3-vq|int4|hybrid`; default `hybrid`).
    /// docs/reference/modes.md has the tradeoffs. Set by `main`, like the checksum
    /// diagnostics.
    pub mode: Mode,
    /// Device budget override, bytes (`--max-mem <GiB>`). None (default) auto-sizes to
    /// `free − OS_RESERVE`. `Some(n)` uses exactly `n` — no OS reserve; the user asked
    /// for it, so it's allowed to OOM/fail at build.
    pub max_mem: Option<u64>,
    /// Attention row-selection mode (`--attn auto|dense|streaming|dsa|misa`, resolved
    /// in `main`). `auto` picks `dsa` when the artifact carries indexer weights, else
    /// `dense`. dsa/misa need the resident DSA indexer. (`auto` also used to consider
    /// whether the backend HAD the indexer kernels — the Vulkan build never did, and was
    /// retired 2026-08-06.)
    pub attn: crate::attn::AttnMode,
}

fn mem_available() -> Result<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").context("read /proc/meminfo")?;
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .context("parse MemAvailable")?;
            return Ok(kb * 1024);
        }
    }
    bail!("MemAvailable not found in /proc/meminfo")
}

/// Refuse a machine that cannot host the auto-sized expert pool, before anything is
/// allocated. Only the AUTO path needs headroom: an explicit `--max-mem` is honoured
/// literally — the user asked for that size, so it is allowed to OOM at pin build.
///
/// A free function taking the one argument it reads, because measuring the machine is all
/// it does. It was also all `Config::discover` did: nine passthrough arguments (with an
/// `#[allow(clippy::too_many_arguments)]` to say so), a struct literal, and then `main`
/// immediately overwriting two of the fields it had just defaulted. `main` builds the
/// literal itself now. Sole-tenant GPU enforcement stayed in `device::DeviceTier::new`,
/// the single owner of the GTT guard, for the same reason: next to what it protects.
pub fn check_budget(max_mem: Option<u64>) -> Result<()> {
    if max_mem.is_some() {
        return Ok(());
    }
    let avail = mem_available()?;
    if avail <= OS_RESERVE {
        bail!(
            "only {:.1} GB available; need more than the {:.0} GB OS reserve",
            avail as f64 / 1e9,
            OS_RESERVE as f64 / 1e9
        );
    }
    Ok(())
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "model={} bench={:?} mode={} attn={:?} cache_policy={} 2q_kin={}% 2q_kout={}% trace={:?} prompt={:?} os_reserve={:.0}GiB max_mem={}",
            self.model,
            self.bench,
            self.mode,
            self.attn,
            self.cache_policy,
            self.two_q.kin_pct(),
            self.two_q.kout_pct(),
            self.trace,
            self.prompt,
            OS_RESERVE as f64 / GIB,
            match self.max_mem {
                Some(n) => format!("{:.0}GiB", n as f64 / GIB),
                None => "auto(all free)".to_string(),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// THE DEFAULT MODE SELECTS WHICH ROUTED-EXPERT ARITHMETIC RUNS, so it changes decode
    /// output — pin it.
    ///
    /// This replaces `the_default_mode_passes_the_backend_gate`, which was deleted on
    /// 2026-08-06 together with `Config::validate_backend`. That test called the gate and,
    /// by its own doc, "on `rocm` … asserts nothing"; with one backend it would have been
    /// vacuous in every configuration. The rule it was defending survives in [`Mode`]'s
    /// header — *a default must be a configuration the build can actually run* — and this
    /// is the half of it that can still fail: the default is `Hybrid`, and a stray
    /// `#[default]` moved to another variant would otherwise be caught by nothing.
    ///
    /// Deliberately backend-free so it runs in the featureless build, which since the
    /// Vulkan job was deleted is the only configuration CI compiles.
    #[test]
    fn the_default_mode_is_hybrid() {
        assert_eq!(
            Mode::default(),
            Mode::Hybrid,
            "changing the default --mode changes decode output; see docs/reference/modes.md"
        );
    }
}
