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
    /// EFAULTs on direct io_uring DMA into VMM (see stream.hip). Set only to force the
    /// raw-DMA path.
    pub direct_vmm_dma: bool,
    /// Dump the routed-expert access trace to this path (`--trace`): one line per MoE
    /// layer, the keys it looked up. Feeds the offline `replay` cache-policy sim.
    pub trace: Option<String>,
    /// Override the fixed bench prompt (`--prompt`), for capturing routing traces of
    /// diverse inputs. None = the default prompt.
    pub prompt: Option<String>,
    /// Routed-expert eviction policy (`--cache-policy` lru|2q|arc). Default "2q".
    pub cache_policy: String,
    /// 2Q's A1in/A1out split (`--2q-kin` / `--2q-kout`, percentages of pool capacity).
    /// Ignored by `lru`/`arc`. Unset = [`crate::cache::TwoQSplit::default`].
    pub two_q: crate::cache::TwoQSplit,
    /// DIAGNOSTIC (`--checksum-layer <l>`): hash every routed expert's weights on MoE
    /// layer `l` after they land, keyed by `(layer, expert)` — the corruption probe
    /// that caught the 2Q eviction bug. Set by `main`, not discovered.
    pub checksum_layer: Option<usize>,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    pub checksum_x: bool,
    /// Feed pool (tokio workers + pread tasks). Physical cores ÷ 2 — the measured
    /// optimum; the SMT-logical default is the proven pathology. The CPU never
    /// computes experts — it routes, samples, and keeps the GPU fed.
    pub threads: usize,
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
        // available_parallelism() is the LOGICAL count (SMT included): /2 gives
        // physical cores, /2 again is the measured feed-pool optimum.
        let threads =
            std::thread::available_parallelism().map_or(8, |n| (n.get() / 4).clamp(4, 16));
        Ok(Self {
            model,
            bench,
            direct_vmm_dma,
            trace,
            prompt,
            cache_policy,
            two_q,
            checksum_layer: None,
            checksum_x: false,
            threads,
            max_mem,
            attn,
        })
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "model={} bench={:?} attn={:?} direct_vmm_dma={} cache_policy={} 2q_kin={}% 2q_kout={}% trace={:?} prompt={:?} os_reserve={:.0}GiB max_mem={} threads={}",
            self.model,
            self.bench,
            self.attn,
            self.direct_vmm_dma,
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
            self.threads,
        )
    }
}
