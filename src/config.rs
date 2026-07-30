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
/// See MODES.md for the tradeoffs.
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
    /// Opt OUT of the pinned-host bounce and DMA cold reads straight into VMM device
    /// memory (`--direct-vmm-dma`). Default false = bounce (read into pinned host,
    /// then `hipMemcpy` into VMM) — measures faster (sidesteps the coherent/snoop tax
    /// on DMA into host-mapped device pages) and survives kernels whose amdgpu path
    /// EFAULTs on direct io_uring DMA into VMM (see src/stream.rs). Set only to force the
    /// raw-DMA path.
    pub direct_vmm_dma: bool,
    /// Dump the routed-expert access trace to this path (`--trace`), format v2: a
    /// `# rivoli-trace v2 top_k=<k> window=<w>` header, then one line per MoE layer —
    /// the keys it looked up, then `|`, then the top-`w` router candidates as
    /// `key:choice`. Feeds the offline `replay` cache-policy sim; the header and the
    /// `|` tail are both invisible to a v1 reader.
    pub trace: Option<String>,
    /// Override the fixed bench prompt (`--prompt`), for capturing routing traces of
    /// diverse inputs. None = the default prompt.
    pub prompt: Option<String>,
    /// Routed-expert cache policy (`--cache-policy` lru|2q|arc|top-m). Default "2q".
    /// `top-m` is the only one that also changes WHICH experts run — see
    /// docs/CACHE_ROUTE.md and [`Config::validate`].
    pub cache_policy: String,
    /// 2Q's A1in/A1out split (`--2q-kin` / `--2q-kout`, percentages of pool capacity).
    /// Ignored by `lru`/`arc`. Unset = [`crate::cache::TwoQSplit::default`].
    pub two_q: crate::cache::TwoQSplit,
    /// `top-m`'s (J, M) (`--route-j` / `--route-m`). Ignored by every other policy.
    pub route: crate::hybrid::RouteAdvice,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    pub checksum_x: bool,
    /// Routed-expert format mode (`--mode int3-vq|int4|hybrid`, default `hybrid`). See
    /// [`Mode`] and MODES.md. Set by `main`, like the checksum diagnostics.
    pub mode: Mode,
    /// Device budget override, bytes (`--max-mem <GiB>`). None (default) auto-sizes to
    /// `free − OS_RESERVE`. `Some(n)` uses exactly `n` — no OS reserve; the user asked
    /// for it, so it's allowed to OOM/fail at build.
    pub max_mem: Option<u64>,
    /// Attention row-selection mode (`--attn auto|dense|streaming|dsa|misa`, resolved
    /// in `main`). `auto` picks `dsa` when the artifact carries indexer weights, else
    /// `dense`. dsa/misa need the resident DSA indexer.
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

impl Config {
    /// Discover the machine. The expert-pool budget is derived here from
    /// `MemAvailable`; sole-tenant GPU enforcement lives in `device::DeviceTier::new`
    /// (the single owner of the GTT guard), closer to the allocation it protects.
    // Each arg is a distinct runtime knob threaded from the CLI; bundling them into a
    // struct used at one call site is churn.
    #[allow(clippy::too_many_arguments)]
    pub fn discover(
        model: String,
        bench: Option<usize>,
        direct_vmm_dma: bool,
        trace: Option<String>,
        prompt: Option<String>,
        cache_policy: String,
        two_q: crate::cache::TwoQSplit,
        route: crate::hybrid::RouteAdvice,
        max_mem: Option<u64>,
        attn: crate::attn::AttnMode,
    ) -> Result<Self> {
        let avail = mem_available()?;
        // Only the auto path needs headroom; an explicit --max-mem sizes itself.
        if max_mem.is_none() && avail <= OS_RESERVE {
            bail!(
                "only {:.1} GB available; need more than the {:.0} GB OS reserve",
                avail as f64 / 1e9,
                OS_RESERVE as f64 / 1e9
            );
        }
        Ok(Self {
            model,
            bench,
            direct_vmm_dma,
            trace,
            prompt,
            cache_policy,
            two_q,
            route,
            checksum_x: false,
            mode: Mode::default(),
            max_mem,
            attn,
        })
    }

    /// Reject the flag combinations the engine cannot honour. Called by `main` AFTER
    /// `mode` is set (which `discover` cannot see).
    ///
    /// `top-m` is the first policy that is neither mode-agnostic nor output-neutral, so
    /// both of these are hard errors rather than quiet fallbacks: a fallback would
    /// attribute some other mechanism's behaviour to `top-m` in a measurement that is
    /// specifically trying to price `top-m`.
    pub fn validate(&self) -> Result<()> {
        self.validate_backend()?;
        if self.cache_policy != "top-m" {
            return Ok(());
        }
        if self.mode == Mode::Hybrid {
            bail!(
                "--cache-policy top-m is implemented for the SINGLE-FORMAT modes only \
                 (--mode int3-vq or --mode int4). The hybrid rank-driven tier rule \
                 (docs/CACHE_ROUTE.md, \"Mode integration\") is not built yet; falling back \
                 to the frequency threshold would credit its behaviour to top-m."
            );
        }
        if self.trace.is_some() {
            bail!(
                "--cache-policy top-m cannot be combined with --trace: the v2 trace format \
                 promises the candidate window's first top_k entries ARE the selection \
                 (bin/replay hard-fails otherwise), and substitution is precisely what \
                 breaks that prefix. Capture traces under lru|2q|arc."
            );
        }
        Ok(())
    }

    /// Reject, AT STARTUP, the configurations whose kernels the selected backend does not
    /// have. Nothing to reject on `rocm`, which is the reference backend.
    ///
    /// The Vulkan launchers for these paths return `Err` too, but that fires mid-decode —
    /// after the artifact is mmapped, the tier is filled and forty layers have run. Same
    /// information, an order of magnitude more expensive to receive. See docs/VULKAN.md,
    /// "Kernel inventory — port 16 of 29".
    #[cfg(feature = "vulkan")]
    fn validate_backend(&self) -> Result<()> {
        use crate::attn::AttnMode;
        if self.mode != Mode::Int3Vq {
            bail!(
                "--mode {} needs the int4 expert kernels, which the Vulkan backend does not \
                 have (docs/VULKAN.md defers them). Use --mode int3-vq, or rebuild with \
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
                 have (docs/VULKAN.md defers them). Use --attn dense or --attn streaming, or \
                 rebuild with --features rocm.",
                self.attn
            );
        }
        Ok(())
    }

    /// The `rocm` arm of [`Config::validate_backend`]: every kernel is present, so there is
    /// nothing to refuse.
    #[cfg(not(feature = "vulkan"))]
    fn validate_backend(&self) -> Result<()> {
        Ok(())
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "model={} bench={:?} mode={} attn={:?} direct_vmm_dma={} cache_policy={} 2q_kin={}% 2q_kout={}% route_j={} route_m={} trace={:?} prompt={:?} os_reserve={:.0}GiB max_mem={}",
            self.model,
            self.bench,
            self.mode,
            self.attn,
            self.direct_vmm_dma,
            self.cache_policy,
            self.two_q.kin_pct(),
            self.two_q.kout_pct(),
            self.route.j,
            self.route.m,
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

    fn cfg(policy: &str, mode: Mode, trace: Option<&str>) -> Config {
        Config {
            model: "/nonexistent".into(),
            bench: Some(1),
            direct_vmm_dma: false,
            trace: trace.map(String::from),
            prompt: None,
            cache_policy: policy.into(),
            two_q: crate::cache::TwoQSplit::default(),
            route: crate::hybrid::RouteAdvice::default(),
            checksum_x: false,
            mode,
            max_mem: None,
            attn: crate::attn::AttnMode::Dense,
        }
    }

    /// `top-m` in `--mode hybrid` must FAIL, not quietly fall back to the frequency
    /// threshold: the hybrid rank-driven tier rule (docs/CACHE_ROUTE.md "Mode
    /// integration") is a later step, and a silent fallback would let a hybrid run
    /// report `top-m` numbers that `top-m` did not produce.
    #[test]
    fn top_m_in_hybrid_mode_fails_loudly() {
        let e = cfg("top-m", Mode::Hybrid, None).validate().expect_err("must reject");
        let msg = e.to_string();
        assert!(msg.contains("SINGLE-FORMAT"), "unhelpful message: {msg}");
        assert!(msg.contains("not built yet"), "must say what is missing: {msg}");
        for m in [Mode::Int3Vq, Mode::Int4] {
            cfg("top-m", m, None).validate().expect("single-format modes are supported");
        }
    }

    /// Substitution breaks the v2 trace's "window prefix == selection" promise, so the
    /// two cannot be captured together — bin/replay would either bail or, worse, screen
    /// a future (J, M) grid against a trace already distorted by an earlier one.
    #[test]
    fn top_m_with_trace_fails_loudly() {
        let e = cfg("top-m", Mode::Int4, Some("/tmp/t"))
            .validate()
            .expect_err("must reject");
        assert!(e.to_string().contains("--trace"), "{e}");
    }

    /// ...and none of it touches the other policies: validate is a no-op for them.
    #[test]
    fn the_other_policies_are_unconstrained() {
        for p in ["lru", "2q", "arc"] {
            for m in [Mode::Int3Vq, Mode::Int4, Mode::Hybrid] {
                cfg(p, m, Some("/tmp/t")).validate().expect("{p} must stay unconstrained");
            }
        }
    }
}
