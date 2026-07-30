//! Matched MoE-kernel microbench — the SAME source, both backends.
//!
//! `examples/dot_bench.rs` cannot answer "how much slower are the Vulkan shaders?"
//! because it measures `gemv_vq`/`gemv_i4`, which the Vulkan backend does not have. This
//! measures `moe_expert_range` + `moe_reduce`, which both backends do have, at the
//! engine's real dims — so the number is a like-for-like kernel comparison.
//!
//! Why it is needed: the in-engine `moe` bucket reads 675 ms on Vulkan against 276 ms on
//! ROCm, but that bucket contains launch overhead, host-gated launch bubbles between
//! per-expert `sig.await`s, and the reduce. Isolating the kernels separates
//! "the shaders are slow" from "the orchestration is slow", and only the first is a
//! shader problem.
//!
//! Run: `cargo run --release --features rocm   --example moe_bench`
//!      `cargo run --release --features vulkan --example moe_bench`
#![cfg(any(feature = "rocm", feature = "vulkan"))]
#![allow(clippy::expect_used)]
use rivoli::backend::{Event, ExpertDesc, Stream, device_sync, launch_moe_expert_range, launch_moe_reduce};
use rivoli::device::DeviceBuf;
use rivoli::quant::{VQ_DIM, VQ_K, vq_expert_bytes, vq_slot_offsets};

/// The engine's shapes (GLM-5.2): hidden 6144, moe_inter 2048, top-8 routed + 1 shared.
const HIDDEN: usize = 6144;
const INTER: usize = 2048;
const NDESC: usize = 9;
const ITERS: usize = 40;

struct Rng(u64);
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
fn pattern(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let p: Vec<u8> = (0..4096).map(|_| r.next() as u8).collect();
    let mut v = p.repeat(n.div_ceil(p.len()));
    v.truncate(n);
    v
}

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dev(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len().max(4)).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}

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
    let x = dev(&f32b(&(0..HIDDEN).map(|_| (r.next() >> 8) as f32 / 1e7).collect::<Vec<_>>()));
    let w = dev(&f32b(&vec![1.0f32; NDESC]));
    let mut h = dev(&vec![0u8; NDESC * INTER * 4]);
    let mut partial = dev(&vec![0u8; NDESC * HIDDEN * 4]);
    let mut out = dev(&vec![0u8; HIDDEN * 4]);

    // Descriptors point into the slab at each expert's six projection offsets.
    let base = weights.ptr() as *const u8;
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
    // ENGINE actually does: every partial is gated on its own `sig.await`, so the launches
    // cannot be batched. The difference between the two isolates PER-LAUNCH cost, and a
    // Vulkan launch is a record + submit where a HIP launch is just an enqueue.
    let mut run = |label: &str, iters: usize, per_expert: bool| {
        // Warm up: first-touch page faults and pipeline creation must not be timed.
        for _ in 0..3 {
            // SAFETY: every pointer is a live device buffer sized above; `stream` is live.
            unsafe {
                let (start, count) = if per_expert { (0, 1) } else { (0, NDESC) };
                let _ = (start, count);
                for e in 0..(if per_expert { NDESC } else { 1 }) {
                    launch_moe_expert_range(
                        x.ptr() as *const f32, HIDDEN, INTER,
                        if per_expert { e } else { 0 },
                        if per_expert { 1 } else { NDESC },
                        dbuf.ptr() as *const ExpertDesc,
                        cb0.ptr() as *const u16, cb1.ptr() as *const u16, cb2.ptr() as *const u16,
                        w.ptr() as *const f32, h.ptr_mut() as *mut f32,
                        partial.ptr_mut() as *mut f32, stream.raw(),
                    )
                    .expect("moe");
                }
                launch_moe_reduce(
                    partial.ptr() as *const f32, NDESC, HIDDEN,
                    out.ptr_mut() as *mut f32, stream.raw(),
                )
                .expect("reduce");
            }
        }
        device_sync().expect("sync");

        let t = std::time::Instant::now();
        ev0.record(stream.raw()).expect("ev0");
        for _ in 0..iters {
            // SAFETY: as above.
            unsafe {
                let (start, count) = if per_expert { (0, 1) } else { (0, NDESC) };
                let _ = (start, count);
                for e in 0..(if per_expert { NDESC } else { 1 }) {
                    launch_moe_expert_range(
                        x.ptr() as *const f32, HIDDEN, INTER,
                        if per_expert { e } else { 0 },
                        if per_expert { 1 } else { NDESC },
                        dbuf.ptr() as *const ExpertDesc,
                        cb0.ptr() as *const u16, cb1.ptr() as *const u16, cb2.ptr() as *const u16,
                        w.ptr() as *const f32, h.ptr_mut() as *mut f32,
                        partial.ptr_mut() as *mut f32, stream.raw(),
                    )
                    .expect("moe");
                }
                launch_moe_reduce(
                    partial.ptr() as *const f32, NDESC, HIDDEN,
                    out.ptr_mut() as *mut f32, stream.raw(),
                )
                .expect("reduce");
            }
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
            "{label:<10} wall {wall_us:8.1} us/call   gpu {gpu_us:8.1} us/call   \
             {:6.1} GB/s (gpu)   {:6.1} GB/s (wall)",
            gb / (gpu_us * 1e-6) , gb / (wall_us * 1e-6),
        );
    };
    run("batched", ITERS, false);
    run("per-expert", ITERS, true);
    println!("\nNOTE: the pool is re-read from the same {:.0} MB every iteration, so this \
              is cache-warm relative to the engine, which streams 78 distinct experts. \
              Compare backends against each other, not against the in-engine moe bucket.",
             (slot * NDESC) as f64 / 1e6);
}
