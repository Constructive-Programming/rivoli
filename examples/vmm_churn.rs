//! A standalone reproducer for `glimmer-open-items.md` §4b: SIGSEGV inside `libamdhip64`'s VMM
//! path, seen from `tests/glimmer_reference.rs` at roughly 3 runs in 52.
//!
//! **Why a separate binary.** That test takes ~40 s per trial and builds a whole engine — a
//! tokenizer, a converted fixture, a KV cache and 52 layers — around the two calls under
//! suspicion. Three cores put the fault in `rivoli_vmm_alloc` (twice) and `rivoli_vmm_free`
//! (once) with nothing of Glimmer's in the frame below, so the hypothesis this exists to test is
//! that **VMM allocate/free churn alone is enough**, with no model at all.
//!
//! ```text
//! cargo run --release --features rocm --example vmm_churn -- [iters] [bytes]
//! ```
//!
//! Exits 0 if every cycle completed, so a crash is a signal rather than something to read out of
//! a log. Sizes default to what the fixture actually asks for (1.4-1.7 MB tiers), because the
//! whole point is to reproduce THAT, not to allocate something convenient.

fn main() {
    #[cfg(not(feature = "rocm"))]
    {
        eprintln!("vmm_churn needs --features rocm");
        std::process::exit(2);
    }
    #[cfg(feature = "rocm")]
    {
        let a: Vec<String> = std::env::args().collect();
        let iters: usize = a.get(1).and_then(|s| s.parse().ok()).unwrap_or(500);
        // 1_425_856 and 1_665_472 are the two sizes the captured cores died on.
        let bytes: usize = a.get(2).and_then(|s| s.parse().ok()).unwrap_or(1_425_856);
        eprintln!("vmm_churn: {iters} cycles of DeviceTier::new({bytes})");
        for i in 0..iters {
            // Built and dropped inside the loop body: one allocate, one free, nothing else
            // alive across the boundary.
            match rivoli::memory::device::DeviceTier::new(bytes) {
                Ok(t) => drop(t),
                Err(e) => {
                    eprintln!("vmm_churn: cycle {i} FAILED to allocate: {e:#}");
                    std::process::exit(1);
                }
            }
            if i % 100 == 99 {
                eprintln!("  {} cycles ok", i + 1);
            }
        }
        eprintln!("vmm_churn: {iters} cycles completed, no crash");
    }
}
