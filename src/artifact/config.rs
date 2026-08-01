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
/// # The DEFAULT is backend-dependent, and that is the lesser of two wrongs
///
/// `hybrid` on `rocm`, `int3-vq` on `vulkan`. The Vulkan backend has no int4 expert kernels
/// (docs/investigations/vulkan-port.md ports 16 of 29), so `validate_backend` rejects `hybrid` — which meant a
/// bare `rivoli <model>` on a Vulkan build failed on its own default, before it read a
/// single byte of the artifact. A first command that errors out is a bad seam, and "pick a
/// mode this build can actually run" is a better default than "pick the mode the other
/// build prefers and then refuse".
///
/// It IS a divergence, and the alternative — one default, refused half the time — was
/// worse. `--mode` is echoed in `Config`'s Display and in the OTLP run attributes, so no
/// measurement can silently attribute one mode's numbers to the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Mode {
    /// Every routed expert is int3-VQ (`.vq3`): smallest, most slots, gather-bound.
    #[cfg_attr(feature = "vulkan", default)]
    Int3Vq,
    /// Every routed expert is int4 (`.i4`): ~1.8× faster compute, bigger, fewer slots.
    Int4,
    /// Frequent experts int4 (HOT), the rest int3-VQ (COLD). Needs both file sets; the
    /// byte-aware policy floats the split.
    #[cfg_attr(not(feature = "vulkan"), default)]
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
    /// Routed-expert format mode (`--mode int3-vq|int4|hybrid`; default `hybrid` on `rocm`,
    /// `int3-vq` on `vulkan` — see [`Mode`] for why the default differs). docs/reference/modes.md has the
    /// tradeoffs. Set by `main`, like the checksum diagnostics.
    pub mode: Mode,
    /// Device budget override, bytes (`--max-mem <GiB>`). None (default) auto-sizes to
    /// `free − OS_RESERVE`. `Some(n)` uses exactly `n` — no OS reserve; the user asked
    /// for it, so it's allowed to OOM/fail at build.
    pub max_mem: Option<u64>,
    /// Attention row-selection mode (`--attn auto|dense|streaming|dsa|misa`, resolved
    /// in `main`). `auto` picks `dsa` when the artifact carries indexer weights AND the
    /// backend has the indexer kernels, else `dense` — on a Vulkan build it is always
    /// `dense`, and says so. dsa/misa need the resident DSA indexer.
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

impl Config {
    /// Reject, AT STARTUP, the configurations whose kernels the selected backend does not
    /// have. Nothing to reject on `rocm`, which is the reference backend.
    ///
    /// The Vulkan launchers for these paths return `Err` too, but that fires mid-decode —
    /// after the artifact is mmapped, the tier is filled and forty layers have run. Same
    /// information, an order of magnitude more expensive to receive. See docs/investigations/vulkan-port.md,
    /// "Kernel inventory — port 16 of 29".
    ///
    /// The ONLY startup gate that reads the build's features, and it is kept that way on
    /// purpose. A sibling `validate` policed flag COMBINATIONS — a property of the
    /// configuration, the same on every machine — until 2026-08-01, when it was deleted for
    /// having had an `Ok(())` body since `top-m` was retired. It was ever separate because
    /// folding a build-time capability gate into it made its tests pass or fail depending on
    /// which feature the suite was compiled with; the symptom was
    /// `top_m_in_hybrid_mode_fails_loudly` failing under `--features vulkan` on the wrong
    /// error entirely. Any future combination gate belongs in its own function for the same
    /// reason, not in here. `main` also asks the `--moe-gain` gate, which has no `Config`
    /// field to hang off.
    #[cfg(feature = "vulkan")]
    pub fn validate_backend(&self) -> Result<()> {
        use crate::attn::AttnMode;
        if self.mode != Mode::Int3Vq {
            bail!(
                "--mode {} needs the int4 expert kernels, which the Vulkan backend does not \
                 have (docs/investigations/vulkan-port.md defers them). Use --mode int3-vq, or rebuild with \
                 --features rocm.",
                self.mode
            );
        }
        // `Dense` and `Streaming` share the ported attention path; DSA and MISA need the
        // five indexer kernels plus `layernorm`, none of which is ported. `auto` is
        // resolved to a concrete mode before this runs, so it cannot slip through.
        if matches!(self.attn, AttnMode::Dsa | AttnMode::Misa { .. }) {
            bail!(
                "--attn {:?} needs the DSA lightning-indexer kernels (index_append/score/\
                 topk/pool_push/head_route and layernorm), which the Vulkan backend does not \
                 have (docs/investigations/vulkan-port.md defers them). Use --attn dense or --attn streaming, or \
                 rebuild with --features rocm.",
                self.attn
            );
        }
        Ok(())
    }

    /// The non-Vulkan arm of `validate_backend`: `rocm` has every kernel, so there is
    /// nothing to refuse.
    #[cfg(not(feature = "vulkan"))]
    pub fn validate_backend(&self) -> Result<()> {
        Ok(())
    }
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
    #![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom
    use super::*;

    fn cfg(mode: Mode) -> Config {
        Config {
            model: "/nonexistent".into(),
            bench: Some(1),
            trace: None,
            prompt: None,
            cache_policy: "lru".into(),
            two_q: crate::memory::cache::TwoQSplit::default(),
            checksum_x: false,
            mode,
            max_mem: None,
            attn: crate::attn::AttnMode::Dense,
        }
    }

    /// THE DEFAULT CONFIGURATION MUST RUN ON THE BUILD THAT HAS IT.
    ///
    /// A bare `rivoli <model>` on a Vulkan build used to fail twice before reading a byte of
    /// the artifact: `--mode` defaulted to `hybrid`, which `validate_backend` refuses, and
    /// `--attn auto` resolved to `dsa` on any artifact carrying indexer weights, which it
    /// also refuses. This half of the fix is testable here; the `auto` half lives in
    /// `main::resolve_attn` and needs a real artifact to exercise, so it is verified by
    /// running the binary rather than by a unit test.
    ///
    /// Backend-independent by construction — on `rocm`, `validate_backend` is a no-op and
    /// this asserts nothing; on `vulkan` it is the whole point.
    #[test]
    fn the_default_mode_passes_the_backend_gate() {
        cfg(Mode::default())
            .validate_backend()
            .expect("the default --mode must be one this build's backend can run");
    }

    /// The Vulkan capability gate: int3-vq + dense/streaming pass; int4, hybrid, dsa and
    /// misa are refused AT STARTUP with a message naming both the missing kernels and the
    /// way out. See docs/investigations/vulkan-port.md, "Kernel inventory — port 16 of 29".
    #[cfg(feature = "vulkan")]
    #[test]
    fn vulkan_refuses_the_unported_modes() {
        use crate::attn::AttnMode;
        let with_attn = |m: Mode, a: AttnMode| {
            let mut c = cfg(m);
            c.attn = a;
            c
        };
        for a in [AttnMode::Dense, AttnMode::Streaming { sinks: 4, window: 512 }] {
            with_attn(Mode::Int3Vq, a)
                .validate_backend()
                .expect("int3-vq + a ported attention mode is the supported configuration");
        }
        for m in [Mode::Int4, Mode::Hybrid] {
            let e = with_attn(m, AttnMode::Dense)
                .validate_backend()
                .expect_err("int4 expert kernels are not ported");
            let msg = e.to_string();
            assert!(msg.contains("int4 expert kernels"), "must name what is missing: {msg}");
            assert!(msg.contains("--features rocm"), "must name the way out: {msg}");
        }
        for a in [AttnMode::Dsa, AttnMode::Misa { active_heads: 4 }] {
            let e = with_attn(Mode::Int3Vq, a)
                .validate_backend()
                .expect_err("the DSA indexer kernels are not ported");
            let msg = e.to_string();
            assert!(msg.contains("index_append"), "must name what is missing: {msg}");
            assert!(msg.contains("--features rocm"), "must name the way out: {msg}");
        }
    }
}
