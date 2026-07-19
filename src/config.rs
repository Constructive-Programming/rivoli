//! Zero-knob auto-discovery. No environment variables, no config files: the
//! machine is measured at startup and the resolved numbers are printed as the
//! first line of every run — a benchmark whose parameters aren't in its log
//! never happened (lesson from the colibri campaign).

use anyhow::{Context, Result, bail};
use std::fmt;

/// Reserved for the OS and other on-system processes, bytes.
pub const OS_RESERVE: u64 = 16 << 30;

/// Upper bound on the device expert pool (tier + routed slab), bytes. Fill most
/// of device memory but cap here so scratch/KV keep headroom; the pool is online
/// priming, so a bigger cap only captures more of this run's working set.
pub const MAX_POOL: u64 = 80 << 30;

#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the GLM-5.2 int4 snapshot directory (colibri-compatible layout).
    pub snapshot: String,
    /// Benchmark mode: decode this many tokens then print PROFILE and exit.
    /// None = server mode (later milestone).
    pub bench: Option<usize>,
    /// Warm-start the routed-expert LRU from `.coli_usage` at build time
    /// (`--pre-seed`). Default false: fast build, LRU self-warms in a few tokens.
    pub pre_seed: bool,
    /// Opt OUT of the pinned-host bounce and DMA cold reads straight into VMM
    /// device memory (`--direct-vmm-dma`). Default false = bounce (read into pinned
    /// host, then `hipMemcpy` into VMM) — which measures ~13% FASTER than direct
    /// (it sidesteps the coherent/snoop tax on DMA into host-mapped device pages)
    /// AND survives NFS sources, where direct io_uring DMA into VMM EFAULTs on some
    /// kernels (e.g. 6.18.38-gentoo; see stream.hip). Set this only to force the
    /// raw-DMA path (local source, and a kernel where it's actually faster).
    pub direct_vmm_dma: bool,
    /// Dump the routed-expert access trace to this path (`--trace`): one line per
    /// MoE layer, the LRU keys it looked up. Feeds the offline cache-policy sim.
    /// None = no trace (the normal decode path).
    pub trace: Option<String>,
    /// Override the fixed bench prompt (`--prompt`), for capturing routing traces of
    /// diverse, request-like inputs. None = the default "The sky is blue because".
    pub prompt: Option<String>,
    /// Routed-expert eviction policy (`--cache-policy` lru|2q|arc). Default "lru".
    /// 2Q/ARC add scan resistance that matters once prefetch injects a
    /// misprediction stream; without prefetch they measure ≈ LRU.
    pub cache_policy: String,
    /// Cross-layer expert prefetch (`--prefetch`). Default false (baseline). When on,
    /// each MoE layer predicts the NEXT MoE layer's routed experts from its post-attn
    /// residual and submits their cold reads on a second io_uring ring, so the fetch
    /// overlaps this layer's GPU compute instead of stalling the next layer.
    pub prefetch: bool,
    /// Max predicted experts prefetched per layer (`--prefetch-depth`, top-N by
    /// router score). The NVMe is bandwidth-bound, so only the ~idle-during-compute
    /// window is exploitable — a small N (default 2). Ignored unless `prefetch`.
    pub prefetch_depth: usize,
    /// Feed pool (tokio workers + pread tasks). Physical cores ÷ 2 — the
    /// measured optimum; the SMT-logical default is the proven pathology
    /// (0.35 vs 0.86 tok/s). The CPU never computes experts — it routes,
    /// samples, and keeps the GPU fed.
    pub threads: usize,
    /// Upper bound on the device expert pool (tier + routed slab), bytes
    /// (`--max-pool-size <GiB>`). Default [`MAX_POOL`]. `main` caps the resolved
    /// pool budget at `min(free − OS_RESERVE, max_pool_size)`.
    pub max_pool_size: u64,
    /// Cold-expert read path (`--direct-io`). `true` = O_DIRECT (bypass the OS page
    /// cache, DMA straight from NVMe), `false` (default) = buffered reads through the
    /// page cache. Only selects which fd the io_uring cold reads use — the
    /// queue/drain/bounce/`hipMemcpy` path is byte-identical either way, so decode is
    /// bit-identical between modes; only the cache-vs-no-cache mechanism differs.
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
    /// Discover the machine. Fails loudly if another GPU tenant is active —
    /// sole tenancy is a startup invariant, not a runtime hope.
    // Each arg is a distinct discovered/passed runtime knob threaded from the CLI;
    // bundling them into a struct used at one call site is churn.
    #[allow(clippy::too_many_arguments)]
    pub fn discover(
        snapshot: String,
        bench: Option<usize>,
        pre_seed: bool,
        direct_vmm_dma: bool,
        trace: Option<String>,
        prompt: Option<String>,
        cache_policy: String,
        prefetch: bool,
        prefetch_depth: usize,
        max_pool_size: u64,
        direct_io: bool,
    ) -> Result<Self> {
        let avail = mem_available()?;
        if avail <= OS_RESERVE {
            bail!(
                "only {:.1} GB available; need more than the {:.0} GB OS reserve",
                avail as f64 / 1e9,
                OS_RESERVE as f64 / 1e9
            );
        }
        // Sole-tenant enforcement lives in `device::DeviceTier::new` (the single
        // owner of the GTT guard) — it reads the whole-device counter right before
        // the one big allocation, closer to the failure it prevents.
        // Prefetch + --direct-vmm-dma are SOUND together: the pin floors the pool at
        // `top_k + prefetch_depth`, so prefetch's evictions are provably disjoint from
        // the current layer's live experts — the async DMA into VMM (direct mode)
        // never overwrites a slot the running MoE reads. So no guard here; the two
        // compose. (See Pin::build's slot_floor + prefetch_layer's correctness note.)
        // available_parallelism() is the LOGICAL count (SMT included): /2 gives
        // physical cores, /2 again is the measured feed-pool optimum.
        let threads =
            std::thread::available_parallelism().map_or(8, |n| (n.get() / 4).clamp(4, 16));
        Ok(Self {
            snapshot,
            bench,
            pre_seed,
            direct_vmm_dma,
            trace,
            prompt,
            cache_policy,
            prefetch,
            prefetch_depth,
            threads,
            max_pool_size,
            direct_io,
        })
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "snap={} bench={:?} pre_seed={} direct_vmm_dma={} direct_io={} cache_policy={} prefetch={} prefetch_depth={} trace={:?} prompt={:?} os_reserve={:.0}GiB max_pool_size={:.0}GiB threads={}",
            self.snapshot,
            self.bench,
            self.pre_seed,
            self.direct_vmm_dma,
            self.direct_io,
            self.cache_policy,
            self.prefetch,
            self.prefetch_depth,
            self.trace,
            self.prompt,
            OS_RESERVE as f64 / GIB,
            self.max_pool_size as f64 / GIB,
            self.threads
        )
    }
}
