//! Zero-knob auto-discovery. No environment variables, no config files: the
//! machine is measured at startup and the resolved numbers are printed as the
//! first line of every run — a benchmark whose parameters aren't in its log
//! never happened (lesson from the colibri campaign).

use anyhow::{Context, Result, bail};
use std::fmt;

/// Reserved for the OS and other on-system processes, bytes.
pub const OS_RESERVE: u64 = 16 << 30;

/// Foreign GTT usage above this means another GPU tenant is active → refuse
/// to start (a foreign allocation landing mid-run is the proven wedge path).
pub const SOLE_TENANT_MAX_GTT: u64 = 1 << 30;

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
    /// Total budget for expert residency (pin + slab pool), bytes:
    /// MemAvailable − OS_RESERVE − engine overhead (refined at snapshot load).
    pub mem_budget: u64,
    /// Free unified/GTT memory observed at startup, bytes. The device tier is
    /// carved from this in one allocation (stability ladder may cap it from
    /// observed device-loss data — never from configuration).
    pub gtt_free: u64,
    /// Feed pool (tokio workers + pread tasks). Physical cores ÷ 2 — the
    /// measured optimum; the SMT-logical default is the proven pathology
    /// (0.35 vs 0.86 tok/s). The CPU never computes experts — it routes,
    /// samples, and keeps the GPU fed.
    pub threads: usize,
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

/// (total, used) GTT bytes from the amdgpu sysfs node, if present.
fn gtt_info() -> Option<(u64, u64)> {
    let read = |name: &str| -> Option<u64> {
        for card in ["card0", "card1"] {
            let p = format!("/sys/class/drm/{card}/device/{name}");
            if let Ok(s) = std::fs::read_to_string(&p) {
                return s.trim().parse().ok();
            }
        }
        None
    };
    Some((read("mem_info_gtt_total")?, read("mem_info_gtt_used")?))
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
    ) -> Result<Self> {
        let avail = mem_available()?;
        if avail <= OS_RESERVE {
            bail!(
                "only {:.1} GB available; need more than the {:.0} GB OS reserve",
                avail as f64 / 1e9,
                OS_RESERVE as f64 / 1e9
            );
        }
        let (gtt_total, gtt_used) = gtt_info().unwrap_or((0, 0));
        if gtt_used > SOLE_TENANT_MAX_GTT {
            bail!(
                "another GPU tenant holds {:.1} GB of GTT — refusing to start \
                 (sole tenancy required; free the GPU and retry)",
                gtt_used as f64 / 1e9
            );
        }
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
            mem_budget: avail - OS_RESERVE,
            gtt_free: gtt_total.saturating_sub(gtt_used),
            threads,
        })
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "snap={} bench={:?} pre_seed={} direct_vmm_dma={} cache_policy={} prefetch={} prefetch_depth={} trace={:?} prompt={:?} mem_budget={:.1}GiB gtt_free={:.1}GiB os_reserve={:.0}GiB threads={}",
            self.snapshot,
            self.bench,
            self.pre_seed,
            self.direct_vmm_dma,
            self.cache_policy,
            self.prefetch,
            self.prefetch_depth,
            self.trace,
            self.prompt,
            self.mem_budget as f64 / GIB,
            self.gtt_free as f64 / GIB,
            OS_RESERVE as f64 / GIB,
            self.threads
        )
    }
}
