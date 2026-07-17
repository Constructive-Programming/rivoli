//! Per-token profile buckets, colibri-PROFILE-style. Two hard rules from the
//! campaign: (1) every report begins with the full config line; (2) a "GPU
//! run" with zero kernel launches is reported as CPU FALLBACK, loudly — the
//! silent-fallback runs cost us a day of wrong conclusions.

use std::time::Duration;

#[derive(Debug, Default)]
pub struct Profile {
    pub expert_disk: Duration,
    pub expert_matmul: Duration,
    pub attention: Duration,
    pub router: Duration,
    pub sample: Duration,
    pub other: Duration,
    /// GPU kernel launches actually submitted. Zero + rocm feature on = fallback.
    pub gpu_launches: u64,
    /// Expert cache/pin hits and total activations, for the hit-rate line.
    pub hits: u64,
    pub activations: u64,
}

impl Profile {
    pub fn hit_rate(&self) -> f64 {
        if self.activations == 0 {
            return 0.0;
        }
        self.hits as f64 / self.activations as f64
    }

    pub fn report(&self, tokens: usize, wall: Duration) -> String {
        let toks = tokens as f64 / wall.as_secs_f64().max(1e-9);
        let engine = if self.gpu_launches > 0 {
            "gpu"
        } else {
            "CPU-FALLBACK"
        };
        format!(
            "decode {tokens} tokens in {:.2}s ({toks:.2} tok/s) | hit {:.1}% | engine={engine} launches={} \
             | disk {:.1}s matmul {:.1}s attn {:.1}s router {:.1}s sample {:.1}s other {:.1}s",
            wall.as_secs_f64(),
            self.hit_rate() * 100.0,
            self.gpu_launches,
            self.expert_disk.as_secs_f64(),
            self.expert_matmul.as_secs_f64(),
            self.attention.as_secs_f64(),
            self.router.as_secs_f64(),
            self.sample.as_secs_f64(),
            self.other.as_secs_f64(),
        )
    }
}
