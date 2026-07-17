//! Env-first configuration. Every run prints the FULL resolved config before
//! doing anything — a benchmark whose parameters aren't in its log never
//! happened (lesson from the colibri campaign).

use std::fmt;

#[derive(Debug, Clone)]
pub struct Config {
    /// Path to the GLM-5.2 int4 snapshot directory (colibri-compatible layout).
    pub snapshot: String,
    /// Total expert memory budget in GB (host pins + device tier + cache).
    pub ram_gb: f64,
    /// Usage-ranked pinned experts, GB. Deterministic: exactly this many bytes.
    pub pin_gb: f64,
    /// Device-resident tier, GB. 32 is the only size that never device-lost;
    /// grow only with stability data (PLAN.md M3/M6).
    pub dev_gb: f64,
    /// CPU worker pool size. Physical cores / 2 measured optimal (8 on rh-anine);
    /// NEVER default to logical-core count (SMT contention: 0.35 vs 0.86 tok/s).
    pub threads: usize,
    /// Tokens to generate.
    pub ngen: usize,
    /// Prompt text.
    pub prompt: String,
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

impl Config {
    pub fn from_env(snapshot: String) -> Self {
        let phys_cores = std::thread::available_parallelism().map_or(8, |n| n.get() / 2);
        Self {
            snapshot,
            ram_gb: env_f64("RAM_GB", 112.0),
            pin_gb: env_f64("PIN_GB", 64.0),
            dev_gb: env_f64("DEV_GB", 32.0),
            threads: env_usize("THREADS", phys_cores.clamp(4, 8)),
            ngen: env_usize("NGEN", 128),
            prompt: std::env::var("PROMPT").unwrap_or_else(|_| "The sky is blue because".into()),
        }
    }
}

impl fmt::Display for Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SNAP={} RAM_GB={} PIN_GB={} DEV_GB={} THREADS={} NGEN={}",
            self.snapshot, self.ram_gb, self.pin_gb, self.dev_gb, self.threads, self.ngen
        )
    }
}
