//! Zero-knob auto-discovery. No environment variables, no config files: the
//! machine is measured at startup and the resolved numbers are printed as the first
//! line of every run — a benchmark whose parameters aren't in its log never
//! happened.

use anyhow::{Context, Result, bail};
use std::fmt;

/// Safety headroom left free for the OS + the pinned io_uring arena, bytes. The
/// expert pool takes `free − OS_RESERVE` by default (`--max-mem` caps below that).
///
/// 12 GiB: `hipMemGetInfo` on this 124 GiB Strix Halo APU reports ~100 GiB free, but
/// the driver will not durably back a device footprint that large under live decode.
/// Too small a reserve grows the pool until a long run over-subscribes physical
/// memory, high VMM slots get reclaimed, and an expert reads back as NaN —
/// deterministically killing decode (invisible below ~256 tokens). The separate
/// [`MAX_BUDGET`] hard ceiling is the real guard; this keeps the auto-sized default
/// clear of it.
pub const OS_RESERVE: u64 = 12 << 30;

/// Hard ceiling on the TOTAL device budget — resident tier plus routed-expert pool.
/// The budget derives from `MemAvailable`, so on a memory-rich boot the OS reserve
/// alone would size past the point where the driver can no longer durably back the
/// VMM allocation, at which point decode silently reads back NaN (~92 GiB total
/// footprint on this box, around token 290). 88 GiB leaves ~4 GiB of margin under
/// that cliff; do NOT raise it without re-running the corruption check.
pub const MAX_BUDGET: u64 = 88 << 30;

/// Runtime override for [`OS_RESERVE`] (`--os-reserve <GiB>`), for re-testing the
/// pool-size ceiling. The hard [`MAX_BUDGET`] limit still stands.
pub static OS_RESERVE_OVERRIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The effective OS reserve: the override if set, else [`OS_RESERVE`].
pub fn os_reserve() -> u64 {
    let v = OS_RESERVE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if v == 0 { OS_RESERVE } else { v }
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
    /// EFAULTs on direct io_uring DMA into VMM (see stream.hip). Set only to force the
    /// raw-DMA path.
    pub direct_vmm_dma: bool,
    /// Dump the routed-expert access trace to this path (`--trace`): one line per MoE
    /// layer, the keys it looked up. Feeds the offline `replay` cache-policy sim.
    pub trace: Option<String>,
    /// Override the fixed bench prompt (`--prompt`), for capturing routing traces of
    /// diverse inputs. None = the default prompt.
    pub prompt: Option<String>,
    /// Routed-expert eviction policy (`--cache-policy` lru|2q|arc). Default
    /// "2q". 2Q/ARC add the scan resistance that matters once prefetch injects a
    /// misprediction stream.
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
    /// Cross-layer expert prefetch (`--prefetch`). Default false. When on, each MoE
    /// layer predicts the NEXT MoE layer's routed experts from its post-attn residual
    /// and submits their cold reads on a second io_uring ring, overlapping the fetch
    /// with this layer's GPU compute.
    pub prefetch: bool,
    /// Max predicted experts prefetched per layer (`--prefetch-depth`, top-N by router
    /// score). NVMe is bandwidth-bound, so only the idle-during-compute window is
    /// exploitable — a small N (default 2). Ignored unless `prefetch`.
    pub prefetch_depth: usize,
    /// Feed pool (tokio workers + pread tasks). Physical cores ÷ 2 — the measured
    /// optimum; the SMT-logical default is the proven pathology. The CPU never
    /// computes experts — it routes, samples, and keeps the GPU fed.
    pub threads: usize,
    /// Device expert-pool budget cap, bytes (`--max-mem <GiB>`). None (default) takes
    /// all safe free memory (`free − OS_RESERVE`); `Some(n)` caps lower. Bigger = more
    /// resident experts = higher hit rate on this cold-miss-fetch-bound decode.
    pub max_mem: Option<u64>,
    /// Cold-expert read path (`--direct-io`). true = O_DIRECT (bypass the page cache,
    /// DMA straight from NVMe), false (default) = buffered. Only selects the fd; the
    /// queue/drain/bounce path is byte-identical, so decode is bit-identical either way.
    pub direct_io: bool,
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
        prefetch: bool,
        prefetch_depth: usize,
        max_mem: Option<u64>,
        direct_io: bool,
    ) -> Result<Self> {
        let avail = mem_available()?;
        let reserve = os_reserve();
        if avail <= reserve {
            bail!(
                "only {:.1} GB available; need more than the {:.0} GB OS reserve",
                avail as f64 / 1e9,
                reserve as f64 / 1e9
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
            prefetch,
            prefetch_depth,
            threads,
            max_mem,
            direct_io,
        })
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "model={} bench={:?} direct_vmm_dma={} direct_io={} cache_policy={} 2q_kin={}% 2q_kout={}% prefetch={} prefetch_depth={} trace={:?} prompt={:?} os_reserve={:.0}GiB max_mem={} threads={}",
            self.model,
            self.bench,
            self.direct_vmm_dma,
            self.direct_io,
            self.cache_policy,
            self.two_q.kin_pct(),
            self.two_q.kout_pct(),
            self.prefetch,
            self.prefetch_depth,
            self.trace,
            self.prompt,
            os_reserve() as f64 / GIB,
            match self.max_mem {
                Some(n) => format!("{:.0}GiB", n as f64 / GIB),
                None => "auto(all free)".to_string(),
            },
            self.threads,
        )
    }
}
