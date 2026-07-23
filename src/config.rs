//! Zero-knob auto-discovery. No environment variables, no config files: the
//! machine is measured at startup and the resolved numbers are printed as the
//! first line of every run — a benchmark whose parameters aren't in its log
//! never happened (lesson from the colibri campaign).

use crate::attn::AttnMode;
use anyhow::{Context, Result, bail};
use std::fmt;

/// Safety headroom left free for the OS + the pinned io_uring arena, bytes. The
/// expert pool takes `free − OS_RESERVE` by default (`--max-mem` caps below that).
///
/// 26 GiB, NOT 8: `hipMemGetInfo` on this 124 GiB Strix Halo APU reports ~100 GiB
/// free, but the driver will not durably back a device footprint that large under
/// live decode. With the old 8 GiB reserve the pool grew to ~82 GiB (~92 GiB with
/// the resident tier); a long run then streams enough experts to over-subscribe
/// physical memory, high VMM slots get reclaimed, and an expert reads back as NaN
/// — deterministically killing decode around token ~290 (invisible below ~256
/// tokens, which is why 64/128/256-token benches never caught it). Measured on
/// this box: ~74 GiB total footprint is clean to 512 tokens, ~92 GiB corrupts. A
/// 26 GiB reserve keeps the default in the verified-safe zone; raise `--max-mem`
/// only after confirming a longer safe ceiling on the hardware in hand.
pub const OS_RESERVE: u64 = 12 << 30;

/// Hard ceiling on the TOTAL device budget — the always-resident tier plus the
/// routed-expert pool. This is the safety bound, and it is the one that matters:
/// the budget is derived from `MemAvailable`, so on a memory-rich boot the OS
/// reserve alone would size past the point where the driver can no longer durably
/// back the VMM allocation, at which point decode silently reads back NaN (PLAN.md
/// records that failure at ~92 GiB total footprint, around token 290).
///
/// 88 GiB = ~10 GiB resident tier + ~78 GiB pool (4415 slots on this box), verified
/// at 512 tokens with a routed-expert workload byte-identical to a small-pool run —
/// the same corruption check that caught the 2Q eviction bug. That leaves ~4 GiB of
/// margin under the recorded cliff. Do NOT raise it without re-running that check.
pub const MAX_BUDGET: u64 = 88 << 30;

/// Runtime override for [`OS_RESERVE`] (`--os-reserve <GiB>`), for re-testing the
/// pool-size ceiling. The 26 GiB default was set because a ~84 GiB total footprint
/// measured SLOWER — but the recorded mechanism was OS page reclaim, i.e. the pool
/// competing with the page cache, and cold-expert reads are O_DIRECT now and never
/// touch the page cache. That confound is gone and the ceiling deserves re-testing.
/// The separate hard limit stands: at >= ~92 GiB total the driver cannot durably
/// back the VMM pool and decode NaNs (see PLAN.md), so stay well under it.
pub static OS_RESERVE_OVERRIDE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The effective OS reserve: the override if set, else [`OS_RESERVE`].
pub fn os_reserve() -> u64 {
    let v = OS_RESERVE_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if v == 0 { OS_RESERVE } else { v }
}

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
    /// 2Q's A1in/A1out split (`--2q-kin` / `--2q-kout`, percentages of pool
    /// capacity). Ignored by `lru`/`arc`. Unset = [`cache::TwoQSplit::default`],
    /// which reproduces the historical hardcoded `cap/4` / `cap/2` exactly.
    pub two_q: crate::cache::TwoQSplit,
    /// DIAGNOSTIC (`--checksum-layer <l>`): hash every routed expert's weights on
    /// MoE layer `l` after they land, keyed by `(layer, expert)`. Set directly by
    /// `main` rather than discovered — it is a probe, not a tuned knob.
    pub checksum_layer: Option<usize>,
    /// DIAGNOSTIC (`--checksum-x`): hash the residual stream after every layer.
    pub checksum_x: bool,
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
    /// Device expert-pool budget cap, bytes (`--max-mem <GiB>`). `None` (the
    /// default) means take all safe free memory: `main` sets the pool to
    /// `free − OS_RESERVE`. `Some(n)` caps it lower — the resolved budget is
    /// `min(free − OS_RESERVE, n)`. Bigger = more resident experts = higher hit
    /// rate on this cold-miss-fetch-bound decode.
    pub max_mem: Option<u64>,
    /// Cold-expert read path (`--direct-io`). `true` = O_DIRECT (bypass the OS page
    /// cache, DMA straight from NVMe), `false` (default) = buffered reads through the
    /// page cache. Only selects which fd the io_uring cold reads use — the
    /// queue/drain/bounce/`hipMemcpy` path is byte-identical either way, so decode is
    /// bit-identical between modes; only the cache-vs-no-cache mechanism differs.
    pub direct_io: bool,
    /// Attention row-selection mechanism (`--attn dense|streaming|dsa|misa`,
    /// with `--sinks`/`--window` shaping streaming). See `attn::AttnMode`.
    pub attn: AttnMode,
    /// Store the MLA latent cache as fp8-e4m3 + per-128 block scales instead of
    /// bf16 (`--kv-fp8`). Halves KV bandwidth/capacity; ~e4m3 precision loss on
    /// the latent (the rope half stays bf16). Default false (bf16).
    pub kv_fp8: bool,
    /// Directory of VQ-int3 routed-expert files (`--vq-dir`): `L{ll}.i3` per MoE
    /// layer + `codebook.f32`. When set, routed experts stream as VQ-int3 instead of
    /// int4; everything else stays int4 from the snapshot. Default `None` (int4).
    pub vq_dir: Option<String>,
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
        two_q: crate::cache::TwoQSplit,
        prefetch: bool,
        prefetch_depth: usize,
        max_mem: Option<u64>,
        direct_io: bool,
        attn: AttnMode,
        kv_fp8: bool,
        vq_dir: Option<String>,
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
            two_q,
            checksum_layer: None,
            checksum_x: false,
            prefetch,
            prefetch_depth,
            threads,
            max_mem,
            direct_io,
            attn,
            kv_fp8,
            vq_dir,
        })
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const GIB: f64 = (1u64 << 30) as f64;
        write!(
            f,
            "snap={} bench={:?} pre_seed={} direct_vmm_dma={} direct_io={} cache_policy={} 2q_kin={}% 2q_kout={}% prefetch={} prefetch_depth={} trace={:?} prompt={:?} os_reserve={:.0}GiB max_mem={} threads={} attn={:?} kv_fp8={} vq_dir={:?}",
            self.snapshot,
            self.bench,
            self.pre_seed,
            self.direct_vmm_dma,
            self.direct_io,
            self.cache_policy,
            self.two_q.kin_pct(),
            self.two_q.kout_pct(),
            self.prefetch,
            self.prefetch_depth,
            self.trace,
            self.prompt,
            OS_RESERVE as f64 / GIB,
            match self.max_mem {
                Some(n) => format!("{:.0}GiB", n as f64 / GIB),
                None => "auto(all free)".to_string(),
            },
            self.threads,
            self.attn,
            self.kv_fp8,
            self.vq_dir
        )
    }
}
