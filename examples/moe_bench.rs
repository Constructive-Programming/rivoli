//! Matched MoE-kernel microbench — the SAME source, both backends.
//!
//! `examples/dot_bench.rs` cannot answer "how much slower are the Vulkan shaders?"
//! because it measures `gemv_vq`/`gemv_i4`, which the Vulkan backend does not have. This
//! measures `moe_expert_range` + `moe_acc_drain`, which both backends do have, at the
//! engine's real dims — so the number is a like-for-like kernel comparison.
//!
//! Why it is needed: the in-engine `moe` bucket reads 675 ms on Vulkan against 276 ms on
//! ROCm, but that bucket contains launch overhead, host-gated launch bubbles between
//! per-expert `sig.await`s, and the drain. Isolating the kernels separates
//! "the shaders are slow" from "the orchestration is slow", and only the first is a
//! shader problem.
//!
//! Run: `cargo run --release --features rocm   --example moe_bench`
//!      `cargo run --release --features vulkan --example moe_bench`
//!
//! Every item below is gated on a backend feature INDIVIDUALLY rather than by one
//! `#![cfg(any(rocm, vulkan))]` at file scope. That inner attribute blanks the whole file,
//! and cargo then rejects the example for having no `main` — so a featureless
//! `cargo test` / `cargo check --all-targets` failed on THIS FILE with `E0601`, which
//! reads like a missing function rather than a missing feature. The stub `main` at the
//! bottom is what the gating buys: a build with no backend produces a binary that says so.
#![allow(clippy::expect_used)]
#[cfg(any(feature = "rocm", feature = "vulkan"))]
use rivoli::backend::{Event, ExpertDesc, Stream, device_sync, launch_moe_acc_drain, launch_moe_expert_range};
#[cfg(any(feature = "rocm", feature = "vulkan"))]
use rivoli::device::DeviceBuf;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
use rivoli::quant::{VQ_DIM, VQ_K, vq_expert_bytes, vq_slot_offsets};

/// The engine's shapes (GLM-5.2): hidden 6144, moe_inter 2048, top-8 routed + 1 shared.
#[cfg(any(feature = "rocm", feature = "vulkan"))]
const HIDDEN: usize = 6144;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
const INTER: usize = 2048;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
const NDESC: usize = 9;
#[cfg(any(feature = "rocm", feature = "vulkan"))]
const ITERS: usize = 40;
/// Widest token-row batch measured. Buffers are sized for this and the `nrow=1` arm just
/// uses the first row of each, so both arms read the SAME weight slab — which is the whole
/// comparison: how much does a second token row cost when the weights are already being read?
#[cfg(any(feature = "rocm", feature = "vulkan"))]
const MAXROW: usize = 2;

#[cfg(any(feature = "rocm", feature = "vulkan"))]
struct Rng(u64);
#[cfg(any(feature = "rocm", feature = "vulkan"))]
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
}

/// Bytes from a repeated 4 KiB pattern. VARIED, not a constant fill: the VQ dot gathers
/// through a codebook, and a constant index would make every lane hit one entry — turning
/// a scattered read into a broadcast and timing something the real kernel never does.
/// (`dot_bench` has the same note for the same reason.)
#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn pattern(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let p: Vec<u8> = (0..4096).map(|_| r.next() as u8).collect();
    let mut v = p.repeat(n.div_ceil(p.len()));
    v.truncate(n);
    v
}

#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn dev(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len().max(4)).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}

#[cfg(any(feature = "rocm", feature = "vulkan"))]
fn main() {
    let backend = if cfg!(feature = "vulkan") { "vulkan" } else { "rocm" };
    let slot = vq_expert_bytes(HIDDEN, INTER);
    let off = vq_slot_offsets(HIDDEN, INTER);
    println!("backend={backend}  hidden={HIDDEN} inter={INTER} ndesc={NDESC}");
    println!("expert slot = {:.2} MB, {NDESC} experts = {:.2} MB read per call",
             slot as f64 / 1e6, (slot * NDESC) as f64 / 1e6);

    // One contiguous slab holding NDESC experts, exactly as the pool does.
    let weights = dev(&pattern(slot * NDESC, 0xA11CE));
    // fp16 codebooks: VQ_K entries of VQ_DIM halves, one per projection.
    let cb_bytes = VQ_K * VQ_DIM * 2;
    let (cb0, cb1, cb2) = (
        dev(&pattern(cb_bytes, 1)),
        dev(&pattern(cb_bytes, 2)),
        dev(&pattern(cb_bytes, 3)),
    );
    let mut r = Rng(7);
    let x = dev(&f32b(
        &(0..MAXROW * HIDDEN).map(|_| (r.next() >> 8) as f32 / 1e7).collect::<Vec<_>>(),
    ));
    // `wexpert[e·nrow + t]` — token row fastest. All-ones means every row routed to every
    // expert, i.e. the WORST case for batching: no row skips any expert's atomic.
    let w = dev(&f32b(&[1.0f32; NDESC * MAXROW]));
    let mut h = dev(&vec![0u8; NDESC * MAXROW * INTER * 4]);
    // ONE accumulator row per token row, not NDESC partial rows — every expert atomicAdds
    // into it.
    let mut acc = dev(&vec![0u8; MAXROW * HIDDEN * 8]);
    let mut out = dev(&vec![0u8; MAXROW * HIDDEN * 4]);

    // Descriptors point into the slab at each expert's six projection offsets.
    let base = weights.ptr();
    let descs: Vec<ExpertDesc> = (0..NDESC)
        .map(|e| {
            // SAFETY: `e*slot + off[i]` is inside the slab by construction of
            // vq_slot_offsets, whose last offset + its projection's bytes == slot.
            let at = |i: usize| unsafe { base.add(e * slot + off[i]) };
            ExpertDesc {
                gate_indices: at(0),
                gate_scales: at(1) as *const u16,
                up_indices: at(2),
                up_scales: at(3) as *const u16,
                down_indices: at(4),
                down_scales: at(5) as *const u16,
            }
        })
        .collect();
    let dbuf = dev(&descs
        .iter()
        .flat_map(|d| {
            [
                d.gate_indices as usize, d.gate_scales as usize,
                d.up_indices as usize, d.up_scales as usize,
                d.down_indices as usize, d.down_scales as usize,
            ]
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<u8>>()
        })
        .collect::<Vec<u8>>());

    let stream = Stream::compute().expect("stream");
    let (ev0, ev1) = (Event::new().expect("ev"), Event::new().expect("ev"));

    // Two shapes, because they answer different questions.
    //
    // `batched` is one dispatch over all NDESC experts — the cleanest measure of shader
    // throughput. `per-expert` is NDESC dispatches of one expert each, which is what the
    // ENGINE actually does: every expert is gated on its own ticket, so the launches
    // cannot be batched. The difference between the two isolates PER-LAUNCH cost, and a
    // Vulkan launch is a record + submit where a HIP launch is just an enqueue.
    let mut run = |label: &str, iters: usize, per_expert: bool, nrow: usize| -> f64 {
        // One pass of the shape under test. Hoisted so the warm-up and the timed loop
        // cannot drift apart — they were duplicated, and a bench whose two copies disagree
        // measures the difference between them.
        // SAFETY: every pointer is a live device buffer sized for MAXROW rows above;
        // `stream` is live; `nrow` is 1 or 2, which the launcher validates.
        let mut pass = || unsafe {
            for e in 0..(if per_expert { NDESC } else { 1 }) {
                launch_moe_expert_range(
                    x.ptr() as *const f32, HIDDEN, INTER,
                    if per_expert { e } else { 0 },
                    if per_expert { 1 } else { NDESC },
                    dbuf.ptr() as *const ExpertDesc,
                    cb0.ptr() as *const u16, cb1.ptr() as *const u16, cb2.ptr() as *const u16,
                    w.ptr() as *const f32, h.ptr_mut() as *mut f32,
                    acc.ptr_mut() as *mut u64, nrow, stream.raw(),
                )
                .expect("moe");
            }
            // One drain per token row: the drain's `rows` axis is the STREAM split, not
            // the token batch, so the rows are separate calls at separate bases.
            for t in 0..nrow {
                launch_moe_acc_drain(
                    out.ptr_mut().add(t * HIDDEN * 4) as *mut f32,
                    acc.ptr_mut().add(t * HIDDEN * 8) as *mut u64,
                    HIDDEN, 1, 1.0, stream.raw(),
                )
                .expect("drain");
            }
        };
        // Warm up: first-touch page faults and pipeline creation must not be timed.
        for _ in 0..3 {
            pass();
        }
        device_sync().expect("sync");

        let t = std::time::Instant::now();
        ev0.record(stream.raw()).expect("ev0");
        for _ in 0..iters {
            pass();
        }
        ev1.record(stream.raw()).expect("ev1");
        device_sync().expect("sync");
        let wall_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
        let gpu_us = f64::from(Event::elapsed_ms(&ev0, &ev1).expect("elapsed")) * 1000.0
            / iters as f64;
        let gb = (slot * NDESC) as f64 / 1e9;
        // GPU time is the honest denominator where it is available; wall includes launch
        // cost, which is exactly the thing this bench is trying NOT to attribute to the
        // shaders. Both are printed so the gap between them is visible.
        println!(
            "{label:<14} wall {wall_us:8.1} us/call   gpu {gpu_us:8.1} us/call   \
             {:6.1} GB/s (gpu)   {:6.1} GB/s (wall)",
            gb / (gpu_us * 1e-6) , gb / (wall_us * 1e-6),
        );
        gpu_us
    };
    let b1 = run("batched", ITERS, false, 1);
    let p1 = run("per-expert", ITERS, true, 1);
    let b2 = run("batched r2", ITERS, false, 2);
    let p2 = run("per-expert r2", ITERS, true, 2);
    // `c` in the speculative-decode throughput model: a verify pass covering 2 token rows
    // costs `c` single-row passes. Speculating pays whenever (1 + p_accept)/c > 1, so this
    // ratio is what decides whether MTP is worth wiring into the decode loop at all.
    println!(
        "\n2-row cost ratio c (gpu):  batched {:.3}x   per-expert {:.3}x\n\
         at the measured 53.5% draft acceptance -> {:.3}x / {:.3}x tokens per unit work",
        b2 / b1,
        p2 / p1,
        1.535 / (b2 / b1),
        1.535 / (p2 / p1),
    );
    println!("\nNOTE: the pool is re-read from the same {:.0} MB every iteration, so this \
              is cache-warm relative to the engine, which streams 78 distinct experts. \
              Compare backends against each other, not against the in-engine moe bucket.",
             (slot * NDESC) as f64 / 1e6);
}

/// A build with no compute backend has no kernels to time.
///
/// Exists so this file yields a `main` in EVERY configuration — see the note at the top.
/// Mirrors `src/main.rs`'s refusal: exit non-zero and name the fix.
#[cfg(not(any(feature = "rocm", feature = "vulkan")))]
fn main() {
    eprintln!(
        "moe_bench measures the MoE kernels and this build has no compute backend. \
         Re-run with `--features rocm` or `--features vulkan` (exactly one)."
    );
    std::process::exit(1);
}
