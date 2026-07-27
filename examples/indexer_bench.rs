//! DSA lightning-indexer cost + hiding-window microbench — the M0/M1 instrument for
//! [`docs/NPU.md`]. Answers two questions, both NO-NPU:
//!
//! * **M0 (the prize):** what does the indexer cost the GPU, per token, at each
//!   context? Decomposed per kernel, then scaled by the 21 FULL indexer layers
//!   GLM-5.2 actually runs (`indexer_types`: 21 full / 57 shared — NOT 78).
//! * **M1 (the window):** how long is the GPU busy with work that does not depend on
//!   the indexer's selection? Two candidates: the rest of attention phase 1 (exact,
//!   same token) and one layer's MLP (needs the stale-selection approximation).
//!
//! **What this rig does NOT measure**, stated up front because the plan's central risk
//! lives here: hideability is a BANDWIDTH claim, and an NPU sharing the 256 GB/s
//! LPDDR5 bus with the GPU work it overlaps cannot be simulated by a second GPU
//! consumer. A GPU∥GPU probe was built, run, and removed: `index_score` at nt=32768
//! launches 32768 workgroups and the MoE batch ~9000, so each alone over-subscribes
//! all 40 CUs and neither can finish sooner concurrently regardless of spare
//! bandwidth. It measured CU contention, not the bus, and the outcome was determined
//! before it ran. Its three arms are recorded in benchmarks.md, "DSA indexer round".
//! The bus question is answered instead by the GB/s figures this rig prints, against
//! which the indexer's demand is arithmetic — see docs/NPU.md.
//! Run: `cargo run --release --features rocm --example indexer_bench`
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]
use rivoli::device::DeviceBuf;
use rivoli::hip::{
    ExpertDesc, device_sync, launch_append_kv, launch_gather_rope, launch_gemv_f32,
    launch_gemv_fp8, launch_index_append, launch_index_score, launch_index_topk, launch_layernorm,
    launch_mla_absorb_fp8, launch_moe_expert_range, launch_moe_reduce, launch_rmsnorm, launch_rope,
    launch_swiglu,
};
use rivoli::indexer::K_NORM_EPS;
use rivoli::math::{f32_to_e4m3, f32_to_f16};
use rivoli::quant::{VQ_DIM, VQ_K, vq_groups, vq_row_bytes};

// --- GLM-5.2, every value from /var/db/rivoli/glm52-vq3-full/manifest.json. Getting
// these wrong silently measures a different model, so each names its manifest key. ---
const HIDDEN: usize = 6144; // hidden_size
const Q_LORA: usize = 2048; // q_lora_rank
const KVL: usize = 512; // kv_lora_rank
const ROPE: usize = 64; // qk_rope_head_dim
const NOPE: usize = 192; // qk_nope_head_dim
const VH: usize = 256; // v_head_dim
const N_HEADS: usize = 64; // num_attention_heads
const QK: usize = NOPE + ROPE; // per-head q/k width (ModelConfig::qk_head_dim)
const ROPE_THETA: f64 = 8_000_000.0; // rope_parameters.rope_theta (NOT the 1e4 default)
const IDX_NH: usize = 32; // index_n_heads
const IDX_HD: usize = 128; // index_head_dim
/// `index_topk`. The engine's guard is `nt <= topk` (gpu.rs `dsa_select_layer`), so it
/// returns dense AT 2048 as well as below it — hence the sweep includes 2048 and every
/// test here is `nt > IDX_TOPK`.
const IDX_TOPK: usize = 2048;
const MOE_INTER: usize = 2048; // moe_intermediate_size
const DENSE_INTER: usize = 12288; // intermediate_size (the 3 dense layers)
const SLOTS: usize = 9; // num_experts_per_tok (8) + n_shared_experts (1)
const FP8_BLOCK: usize = 128;
/// FULL indexer layers per token — `indexer_types` is 21 "full" / 57 "shared", and
/// shared layers reuse the preceding full layer's selection without running a single
/// indexer kernel. Every per-token total below is ×21, not ×78.
const N_FULL: usize = 21;
/// Of those 21, layers 0/1/2 are dense-MLP (`first_k_dense_replace` = 3) and the other
/// 18 are MoE. Only matters for which MLP window they sit in front of.
const N_FULL_MOE: usize = 18;
const N_FULL_DENSE: usize = 3;
const _: () = assert!(N_FULL_MOE + N_FULL_DENSE == N_FULL);
/// KV slab rows for the window rigs; only has to exceed [`BENCH_POS`].
const KV_ROWS: usize = 4096;
/// Arbitrary in-range cache row. `append_kv`/`index_append`/`rope` costs are all
/// position-independent, so the value is free — it just has to be inside the slabs.
const BENCH_POS: usize = 1000;
/// `benchmarks.md`, "Per-kernel round", fix arm: `o_proj` fp8 [6144×16384].
const O_PROJ_REF_US: f64 = 528.95;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

fn dev(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len()).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}
fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn zeros(n: usize) -> DeviceBuf {
    dev(&vec![0u8; n])
}
fn fill_f32(n: usize, v: f32) -> DeviceBuf {
    dev(&f32b(&vec![v; n]))
}
/// Raw output pointer for a buffer this rig writes through. Taken once from a `mut`
/// binding, so no launch site has to launder a `*const` into a `*mut`.
fn out_ptr(b: &mut DeviceBuf) -> *mut f32 {
    b.ptr_mut() as *mut f32
}

/// `n` bytes from a repeated 4 KiB varying pattern (dot_bench's approach, same reason):
/// these are bandwidth/gather measurements, so the VALUES are irrelevant but a constant
/// fill is not — the fp8 LUT and the VQ codebook gather would both collapse to a
/// broadcast and time something the real kernel never does. `seed` varies the block so
/// distinct buffers are not byte-identical.
fn pattern(n: usize, seed: u64, mut byte: impl FnMut(f32) -> u8) -> Vec<u8> {
    let mut r = Rng(seed);
    let p: Vec<u8> = (0..4096).map(|_| byte(r.f())).collect();
    let mut v = p.repeat(n.div_ceil(p.len()));
    v.truncate(n);
    v
}
fn fp8_weight(o: usize, i: usize, seed: u64) -> (DeviceBuf, DeviceBuf) {
    // |v| <= 0.1 never encodes to the e4m3 NaN pattern, so no fixup is needed.
    let packed = dev(&pattern(o * i, seed, |v| f32_to_e4m3(v * 0.1)));
    let scale = fill_f32(o.div_ceil(FP8_BLOCK) * i.div_ceil(FP8_BLOCK), 1.0);
    (packed, scale)
}
/// bf16 key bytes in a sane range. Arbitrary byte pairs reinterpreted as bf16 span
/// ±1e38, which makes the 128-element dot saturate to ±inf. `index_score`'s own timing
/// would not care (every lane reads every element regardless of value), but inf/NaN
/// scores would poison the top-k this rig also exercises, and `topk_into`'s
/// `partial_cmp(..).unwrap_or(Equal)` comparator is not a total order over NaN.
fn bf16_keys(n: usize, seed: u64) -> Vec<u8> {
    let mut r = Rng(seed);
    let block: Vec<u8> = (0..2048)
        .map(|_| ((r.f() * 0.5).to_bits() >> 16) as u16)
        .flat_map(|h| h.to_le_bytes())
        .collect();
    let mut v = block.repeat(n.div_ceil(block.len()));
    v.truncate(n);
    v
}

/// Mean µs per call of `f`, which receives the iteration index so a caller can rotate
/// over per-layer slabs. One warm-up call, then `iters` behind a SINGLE sync — this is
/// launch *throughput*, and for kernels of a few µs it is bounded below by the host
/// launch cost, not by the GPU. See [`time_serial`].
fn time(iters: u32, f: &dyn Fn(usize)) -> f64 {
    f(0);
    device_sync().expect("sync");
    let t = std::time::Instant::now();
    for i in 0..iters {
        f(i as usize);
    }
    device_sync().expect("sync");
    t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0
}

/// As [`time`] but with a sync PER iteration — the engine's shape, since
/// `dsa_select_layer` joins once per full layer and cannot pipeline across its D2H.
/// The gap between the two is per-launch/sync overhead, which `benchmarks.md` names as
/// the leading candidate for the ~7.8 ms of the `tail` bucket that is in none of its
/// kernels. It dominates the few-µs key-path kernels, so both are reported.
fn time_serial(iters: u32, f: &dyn Fn(usize)) -> f64 {
    f(0);
    device_sync().expect("sync");
    let t = std::time::Instant::now();
    for i in 0..iters {
        f(i as usize);
        device_sync().expect("sync");
    }
    t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0
}

/// One row: µs per call, and what that is per token once scaled by the number of full
/// layers that run it (always [`N_FULL`] — every indexer kernel runs on all of them).
fn row(name: &str, us: f64) {
    println!(
        "  {name:<28} {us:9.2} us/call   x{N_FULL} = {:7.3} ms/tok",
        us * N_FULL as f64 / 1000.0
    );
}

/// The `index_score` rig: a roped query, the gate weights, [`N_FULL`] rotating key
/// slabs, and the score output. The rotation matters — one 8.4 MB slab replayed in a
/// loop is served by Strix Halo's 32 MB MALL, while the engine cycles all 21 per token.
struct ScoreInputs {
    q: DeviceBuf,
    w: DeviceBuf,
    kc: Vec<DeviceBuf>,
    scores: DeviceBuf,
    scores_ptr: *mut f32,
}

impl ScoreInputs {
    fn new(max_nt: usize) -> Self {
        let mut scores = zeros(max_nt * 4);
        let scores_ptr = out_ptr(&mut scores);
        ScoreInputs {
            q: fill_f32(IDX_NH * IDX_HD, 0.02),
            w: fill_f32(IDX_NH, 0.5),
            kc: (0..N_FULL)
                .map(|l| dev(&bf16_keys(max_nt * IDX_HD * 2, 0xB16 + l as u64)))
                .collect(),
            scores,
            scores_ptr,
        }
    }

    /// Launch `index_score` for slab `i` at context `nt` (DSA: null heads = all 32).
    fn launch(&self, i: usize, nt: usize) {
        let wscale = 1.0 / (IDX_NH as f32).sqrt();
        let dscale = 1.0 / (IDX_HD as f32).sqrt();
        // SAFETY: every pointer is a live device buffer of the documented size; `heads`
        // is null (the DSA path, all 32 heads); `scores` holds max_nt >= nt floats.
        unsafe {
            launch_index_score(
                self.q.ptr() as *const f32,
                self.w.ptr() as *const f32,
                self.kc[i % N_FULL].ptr() as *const u16,
                std::ptr::null(),
                nt,
                IDX_NH,
                IDX_NH,
                IDX_HD,
                wscale,
                dscale,
                self.scores_ptr,
            )
            .expect("index_score");
        }
    }
}

/// M0 — the indexer's GPU cost, decomposed, across context.
///
/// `dsa_select_layer` runs two groups per FULL layer: the **key path** every token
/// (`gemv_fp8(wk)` → `layernorm(k_norm)` → `rope` → append, so the cache is ready when
/// the threshold is crossed), and the **score path** only once `nt > index_topk`
/// (`gemv_fp8(wq_b)` → `rope` ×nh → `gemv_f32(weights_proj)` → `index_score`). Both are
/// the prize — offloading the indexer removes all of it.
///
/// Every per-layer weight is allocated [`N_FULL`] times and rotated, because the engine
/// holds a distinct `IndexerPin` per full layer and touches all 21 per token.
fn m0(nts: &[usize], si: &ScoreInputs) -> (f64, f64) {
    let max_nt = *nts.iter().max().expect("non-empty context sweep");

    // --- key path (context-independent) ---
    let xn = fill_f32(HIDDEN, 0.02);
    let wk: Vec<(DeviceBuf, DeviceBuf)> = (0..N_FULL)
        .map(|l| fp8_weight(IDX_HD, HIDDEN, 0x3A0 + l as u64))
        .collect();
    let kn_w = fill_f32(IDX_HD, 1.0);
    let kn_b = zeros(IDX_HD * 4);
    let mut k = zeros(IDX_HD * 4);
    let mut kc_app: Vec<DeviceBuf> = (0..N_FULL).map(|_| zeros(max_nt * IDX_HD * 2)).collect();
    let kc_app_ptr: Vec<*mut u16> = kc_app.iter_mut().map(|b| b.ptr_mut() as *mut u16).collect();
    let xnp = xn.ptr() as *const f32;
    let kp = out_ptr(&mut k);

    // The four key-path launches, in the engine's order. SAFETY: device scratch of the
    // documented dims; `wk[i]`/`kc_app_ptr[i]` are live per-layer buffers.
    let key_path = |i: usize| unsafe {
        let (p, s) = &wk[i % N_FULL];
        launch_gemv_fp8(
            xnp,
            p.ptr(),
            s.ptr() as *const f32,
            IDX_HD,
            HIDDEN,
            FP8_BLOCK,
            kp,
        )
        .expect("wk");
        launch_layernorm(
            kp as *const f32,
            kn_w.ptr() as *const f32,
            kn_b.ptr() as *const f32,
            IDX_HD,
            K_NORM_EPS,
            kp,
        )
        .expect("layernorm");
        launch_rope(kp, 1, ROPE, ROPE, BENCH_POS, ROPE_THETA).expect("rope k");
        launch_index_append(kp as *const f32, kc_app_ptr[i % N_FULL], BENCH_POS, IDX_HD)
            .expect("index_append");
    };

    println!("\nM0 — indexer key path (EVERY token, every full layer, context-independent):");
    let key_us = time(200, &key_path);
    let key_serial = time_serial(200, &key_path);
    row("key path (4 kernels)", key_us);
    println!(
        "    with a sync PER call (the engine's shape, one join per full layer): \
         {key_serial:.2} us => {:.3} ms/tok. The {:.2} us gap is launch/sync overhead, not GPU work.",
        key_serial * N_FULL as f64 / 1000.0,
        key_serial - key_us
    );
    drop(kc_app); // 21 slabs at max_nt — free before the score path allocates its own.

    // --- score path (only when nt > index_topk) ---
    let qr = fill_f32(Q_LORA, 0.02);
    let wqb: Vec<(DeviceBuf, DeviceBuf)> = (0..N_FULL)
        .map(|l| fp8_weight(IDX_NH * IDX_HD, Q_LORA, 0x9B0 + l as u64))
        .collect();
    let wproj: Vec<DeviceBuf> = (0..N_FULL)
        .map(|_| fill_f32(IDX_NH * HIDDEN, 0.001))
        .collect();
    let mut iq = zeros(IDX_NH * IDX_HD * 4);
    let mut iw = zeros(IDX_NH * 4);
    let (qrp, iqp, iwp) = (qr.ptr() as *const f32, out_ptr(&mut iq), out_ptr(&mut iw));

    println!("\nM0 — indexer score path (ONLY when context > index_topk = {IDX_TOPK}):");
    // SAFETY: as above; `wqb[i]` is a live [4096x2048] fp8 weight + its block scales.
    let us_wqb = time(60, &|i| unsafe {
        let (p, s) = &wqb[i % N_FULL];
        launch_gemv_fp8(
            qrp,
            p.ptr(),
            s.ptr() as *const f32,
            IDX_NH * IDX_HD,
            Q_LORA,
            FP8_BLOCK,
            iqp,
        )
        .expect("wq_b");
    });
    row("gemv_fp8 wq_b[4096x2048]", us_wqb);
    let us_ropeq = time(200, &|_| unsafe {
        launch_rope(iqp, IDX_NH, IDX_HD, ROPE, BENCH_POS, ROPE_THETA).expect("rope q");
    });
    row("rope query[32x64]", us_ropeq);
    let us_wp = time(200, &|i| unsafe {
        launch_gemv_f32(
            xnp,
            wproj[i % N_FULL].ptr() as *const f32,
            IDX_NH,
            HIDDEN,
            iwp,
        )
        .expect("weights_proj");
    });
    row("gemv_f32 weights_proj", us_wp);
    let fixed_us = us_wqb + us_ropeq + us_wp;
    row("== score-path fixed subtotal", fixed_us);

    println!("\nM0 — index_score vs context (the only part that grows):");
    println!("      nt      us/call     GB/s    x21 ms/tok   indexer/layer us   total ms/tok");
    for &nt in nts {
        let us = time(if nt >= 16384 { 30 } else { 120 }, &|i| si.launch(i, nt));
        let gbs = (nt * IDX_HD * 2) as f64 / (us * 1e-6) / 1e9;
        // At or below the threshold the engine returns dense BEFORE the score path, so
        // the only cost that token is the key path. Report what the engine really pays.
        let per_layer = if nt > IDX_TOPK {
            key_us + fixed_us + us
        } else {
            key_us
        };
        let note = if nt > IDX_TOPK {
            ""
        } else {
            "  (score path NOT run: dense)"
        };
        println!(
            "  {nt:7}  {us:9.2}  {gbs:7.1}  {:9.3}   {per_layer:12.2}   {:9.3}{note}",
            us * N_FULL as f64 / 1000.0,
            per_layer * N_FULL as f64 / 1000.0,
        );
    }

    // Cheap oracle: a kernel that early-returned would have produced fast, clean,
    // entirely publishable rows above. Prove index_score actually wrote varying finite
    // scores before any of those numbers is quoted.
    si.launch(0, 4096);
    device_sync().expect("sync");
    let mut host = Vec::new();
    si.scores.copy_out_prefix(&mut host, 4096 * 4).expect("d2h");
    let v: Vec<f32> = host
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let finite = v.iter().all(|x| x.is_finite());
    let distinct = v.iter().any(|x| *x != v[0]);
    println!(
        "  oracle: index_score wrote {} scores, finite={finite} varying={distinct}{}",
        v.len(),
        if finite && distinct {
            "  ok"
        } else {
            "  <-- DEGENERATE: the rows above are not measuring the kernel"
        }
    );
    (key_us, fixed_us)
}

/// A plausible stand-in for the indexer's output: ReLU'd weighted head sums, so
/// non-negative with a long tail and mostly distinct. Shared by the host-cost row and the
/// host-vs-device comparison so the two are seeded identically.
fn synth_scores(n: usize) -> Vec<f32> {
    let mut r = Rng(0x5C0);
    (0..n)
        .map(|_| {
            let a = r.f();
            a.abs() * a.abs() * 10.0
        })
        .collect()
}

/// M0b — the host half of the selection: the score D2H and the top-k. Not GPU time,
/// but it is time the GPU spends idle inside `dsa_select_layer` (the engine syncs,
/// reads `nt` floats, top-k's them on the CPU, uploads 2048 rows), so an NPU that
/// produced the selection itself would remove it from the wall too. Measured
/// separately because it is a different kind of cost and must not be folded into a
/// "GPU prize".
///
/// `topk_into` is comparison-driven (`select_nth_unstable_by` + `sort_by`), so this row
/// is DISTRIBUTION-DEPENDENT. The scores are therefore synthesised here to resemble the
/// kernel's output — non-negative, heavy-tailed, mostly distinct — rather than reusing
/// whatever the timing sweep left in the buffer, which is periodic and tie-heavy and
/// would not exercise quickselect the way real scores do.
fn m0_host(nts: &[usize], scores: &mut DeviceBuf) {
    println!("\nM0b — host selection round-trip (D2H + top-k + row upload), per full layer:");
    println!("      nt     D2H us   topk us    total    x21 ms/tok");
    let max_nt = *nts.iter().max().expect("non-empty context sweep");
    let synth = synth_scores(max_nt);
    scores.copy_in_at(0, &f32b(&synth)).expect("seed scores");

    let mut host = Vec::new();
    let mut scores_f: Vec<f32> = Vec::new();
    let mut sel = Vec::new();
    let mut rows: Vec<u32> = Vec::new();
    let mut rows_buf = zeros(max_nt * 4);
    for &nt in nts {
        let iters = 20;
        let (mut d2h, mut topk) = (0u128, 0u128);
        for _ in 0..iters {
            let t = std::time::Instant::now();
            scores.copy_out_prefix(&mut host, nt * 4).expect("d2h");
            scores_f.clear();
            scores_f.extend(
                host.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
            d2h += t.elapsed().as_nanos();
            let t = std::time::Instant::now();
            rivoli::math::topk_into(&scores_f, IDX_TOPK, &mut sel);
            sel.sort_unstable();
            rows.clear();
            rows.extend(sel.iter().map(|&i| i as u32));
            let bytes: Vec<u8> = rows.iter().flat_map(|x| x.to_le_bytes()).collect();
            rows_buf.copy_in_at(0, &bytes).expect("h2d");
            topk += t.elapsed().as_nanos();
        }
        // Oracle: the D2H really moved the seeded bytes (a no-op copy would time fast).
        assert_eq!(
            scores_f[..nt.min(8)],
            synth[..nt.min(8)],
            "score D2H did not round-trip"
        );
        let (d, k) = (
            d2h as f64 / iters as f64 / 1000.0,
            topk as f64 / iters as f64 / 1000.0,
        );
        println!(
            "  {nt:7}  {d:8.2}  {k:8.2}  {:8.2}   {:9.3}",
            d + k,
            (d + k) * N_FULL as f64 / 1000.0
        );
    }
}

/// The engine's OWN score array, from a `RIVOLI_DUMP_SCORES` file if one is present
/// (`(u32 layer, u32 nt, f32[nt])` records, native LE). Tiled or truncated to `n`, which
/// preserves the value distribution exactly and is only approximate about ordering across
/// a tile boundary. `None` when no file is given, and the column is then omitted rather
/// than silently substituted — this is the one column that needs no fixture-matching
/// argument, so it must never be confused with one that does.
fn real_scores(n: usize) -> Option<Vec<f32>> {
    let path = std::env::var("RIVOLI_DUMP_SCORES").ok()?;
    let b = std::fs::read(path).ok()?;
    let mut out: Vec<f32> = Vec::new();
    let mut off = 0usize;
    while off + 8 <= b.len() && out.len() < n {
        let nt = u32::from_le_bytes([b[off + 4], b[off + 5], b[off + 6], b[off + 7]]) as usize;
        off += 8;
        if off + 4 * nt > b.len() {
            break;
        }
        out.extend(
            b[off..off + 4 * nt]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        );
        off += 4 * nt;
    }
    if out.is_empty() {
        return None;
    }
    while out.len() < n {
        let take = (n - out.len()).min(out.len());
        out.extend_from_within(..take);
    }
    out.truncate(n);
    Some(out)
}

/// M0c — device `index_topk` against the host round-trip it replaces, MATCHED.
///
/// Both implementations are timed in the same rig, on the same buffer, on the same two
/// distributions. That matters more than it sounds: the obvious comparison — this
/// kernel against the 214.2 / 334.1 µs/layer the engine was measured spending — mixes
/// instruments, and in the direction that flatters the kernel, because the in-engine
/// host figure carries a ~2× in-situ penalty (docs/NPU.md) that a microbench device
/// figure does not. Only the columns below may be divided by one another.
///
/// THREE fixtures, because two effects were confounded in an earlier revision.
/// `dense` has few ties and random order. `scattered` has the same heavy tie structure
/// as the engine is assumed to produce but random order. `sorted-sparse` has that tie
/// structure pre-sorted into `topk_into`'s comparator order, which is its best case and
/// nothing to do with ties — comparing it against `scattered` is what separates the two.
///
/// Which of these the engine actually produces has NEVER been measured; the claim that
/// its scores are tie-dominated is an assumption, and docs/NPU.md records evidence
/// against it.
///
/// Contexts start at 2456 — the shorter in-engine run's mean — because at `nt <= 2048`
/// the engine returns dense before scoring and neither implementation runs at all.
fn m0c_topk_matched(nts: &[usize], scores: &mut DeviceBuf) {
    let max_nt = *nts.iter().max().expect("non-empty context sweep");
    let mut rows = zeros(IDX_TOPK * 4);
    let rows_ptr = rows.ptr_mut() as *mut u32;
    let sp = scores.ptr() as *const f32;

    let dense = synth_scores(max_nt);
    // Tie-heavy AND randomly ordered, built PER nt: an earlier revision generated the
    // non-zeros across max_nt and then sliced to nt, so at nt < max_nt almost all of them
    // were sliced away (23 survivors at nt=2456, i.e. 99% zeros) and the fixture did not
    // share `sorted_sparse`'s tie fraction the way its comment claimed. Generating inside
    // the sweep keeps the tie structure identical and varies ONLY the order, which is the
    // whole point of the pair.
    let scattered_at = |n: usize| {
        let mut v = vec![0.0f32; n];
        for j in 0..300.min(n) {
            v[(j * 7919) % n] = (300 - j) as f32 * 0.25;
        }
        v
    };
    // Tie-heavy AND pre-sorted: the same sparsity with the non-zeros descending from
    // index 0. Kept because it is a TRAP, and the trap is the finding. `topk_into` seeds
    // its workspace with the identity permutation and orders by (score desc, index asc),
    // so for this fixture the identity IS the sorted order: quickselect and the trailing
    // sort both get an already-sorted slice, which is their best case. The device kernel
    // gets no such benefit. An earlier revision measured only this shape and concluded
    // the kernel was 1.6x slower than the CPU — an artifact of the fixture, not of ties.
    let mut sorted_sparse = vec![0.0f32; max_nt];
    for (i, x) in sorted_sparse.iter_mut().enumerate().take(300) {
        *x = (300 - i) as f32 * 0.25;
    }

    let mut host_buf = Vec::new();
    let mut scores_f: Vec<f32> = Vec::new();
    let mut sel = Vec::new();
    let mut rows_host: Vec<u32> = Vec::new();
    let mut rows_up = zeros(max_nt.max(IDX_TOPK) * 4);

    println!("\nM0c — host vs device selection, MATCHED rig and data (µs per full layer):");
    println!(
        "       nt |  dense (few ties)   |  scattered (ties, random)       |  sorted-sparse (PRE-SORTED)     |  REAL engine scores"
    );
    println!(
        "          |  host   dev   ratio |  host    dev    ratio           |  host    dev    ratio"
    );
    for &nt in nts {
        let mut cell = [(0.0f64, 0.0f64); 4];
        let scattered = scattered_at(nt);
        // The engine's own scores, if a `RIVOLI_DUMP_SCORES` file is present: the only
        // column that needs no fixture-matching argument at all.
        let real = real_scores(nt);
        let mut sets: Vec<&Vec<f32>> = vec![&dense, &scattered, &sorted_sparse];
        if let Some(r) = real.as_ref() {
            sets.push(r);
        }
        for (j, data) in sets.iter().enumerate() {
            scores
                .copy_in_at(0, &f32b(&data[..nt]))
                .expect("seed scores");
            // SAFETY: `scores` holds nt f32; `rows` holds IDX_TOPK u32, which is exactly
            // what the kernel writes here since k == IDX_TOPK <= nt for every nt below.
            let dev_us = time(60, &|_| unsafe {
                launch_index_topk(sp, nt, IDX_TOPK, rows_ptr).expect("index_topk");
            });
            let iters = 20;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                scores.copy_out_prefix(&mut host_buf, nt * 4).expect("d2h");
                scores_f.clear();
                scores_f.extend(
                    host_buf
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                );
                rivoli::math::topk_into(&scores_f, IDX_TOPK, &mut sel);
                sel.sort_unstable();
                rows_host.clear();
                rows_host.extend(sel.iter().map(|&i| i as u32));
                let bytes: Vec<u8> = rows_host.iter().flat_map(|x| x.to_le_bytes()).collect();
                rows_up.copy_in_at(0, &bytes).expect("h2d");
            }
            let host_us = t.elapsed().as_nanos() as f64 / iters as f64 / 1000.0;
            cell[j] = (host_us, dev_us);
        }
        let real_col = if real.is_some() {
            format!(
                " | {:6.1} {:6.1} {:5.2}x",
                cell[3].0,
                cell[3].1,
                cell[3].0 / cell[3].1
            )
        } else {
            String::new()
        };
        println!(
            "  {nt:7} | {:6.1} {:5.1} {:5.2}x | {:6.1} {:6.1} {:5.2}x | {:6.1} {:6.1} {:5.2}x{real_col}",
            cell[0].0,
            cell[0].1,
            cell[0].0 / cell[0].1,
            cell[1].0,
            cell[1].1,
            cell[1].0 / cell[1].1,
            cell[2].0,
            cell[2].1,
            cell[2].0 / cell[2].1,
        );
    }
}

/// M1 window 1 — the EXACT window: every launch in attention phase 1 that does NOT
/// depend on the indexer's selection.
///
/// This is the correction that matters most in this file. An earlier revision scoped
/// window 1 to "kv_proj + KV-append", which is ~22 µs. But in `forward` the indexer's
/// own inputs — `xn` (post input_layernorm) and `qr` (post q_a_ln) — are both ready
/// after the *second* rmsnorm, and the first consumer of the selection is
/// `launch_attend`. Everything the engine launches in between is selection-independent
/// and is therefore the real window:
///
/// ```text
/// gemv_fp8(q_b) → gemv_fp8(kv_a) → rmsnorm(kv_a_ln) → rope(key) → rope(query, 64 heads)
///   → append_kv → mla_absorb_fp8(kv_b) → gather_rope     [then attend, which needs it]
/// ```
///
/// `q_b` alone is [16384×2048] fp8 = 33.5 MB and `kv_b` is 14.7 MB, so the omitted work
/// was an order of magnitude larger than the part that was measured. Scoping a window
/// to a subset of the independent work does not make the answer conservative — it makes
/// it wrong in the direction of refuting a window that exists.
fn m1_window_exact() -> f64 {
    let xn = fill_f32(HIDDEN, 0.02);
    let qr = fill_f32(Q_LORA, 0.02);
    // ROT copies of the two big weights, rotated like the indexer's. Replaying ONE
    // 33.5 MB `q_b` measures 372 GB/s — above the 256 GB/s bus, i.e. the 32 MB MALL
    // serving it — while the engine holds a distinct `q_b`/`kv_b` per layer and pays a
    // cold read. Window 1 is the number that decides the exact-overlap design, so it
    // does not get to be the one row measured out of cache. Rotation depth is whatever
    // exceeds the 32 MB MALL: 21 for the small per-layer weights (free — it is also the
    // layer count), 4 here, since 4 x 33.5 + 4 x 14.7 MB = 193 MB already clears it 6x.
    // NOT saturation-tested: nothing here shows ROT=8 would not move the number further.
    const ROT: usize = 4;
    let qb: Vec<(DeviceBuf, DeviceBuf)> = (0..ROT)
        .map(|l| fp8_weight(N_HEADS * QK, Q_LORA, 0x11 + l as u64))
        .collect();
    let kvb: Vec<(DeviceBuf, DeviceBuf)> = (0..ROT)
        .map(|l| fp8_weight(N_HEADS * (NOPE + VH), KVL, 0x33 + l as u64))
        .collect();
    let (kva_p, kva_s) = fp8_weight(KVL + ROPE, HIDDEN, 0x22);
    let kva_ln = fill_f32(KVL, 1.0);
    let mut comp = zeros((KVL + ROPE) * 4);
    let mut q = zeros(N_HEADS * QK * 4);
    let mut qabs = zeros(N_HEADS * KVL * 4);
    let mut qrope = zeros(N_HEADS * ROPE * 4);
    let nb = KVL / FP8_BLOCK;
    let mut lc8 = zeros(KV_ROWS * KVL);
    let mut lscale = zeros(KV_ROWS * nb * 4);
    let mut rc = zeros(KV_ROWS * ROPE * 2);
    let (xnp, qrp) = (xn.ptr() as *const f32, qr.ptr() as *const f32);
    let (cp, qp) = (out_ptr(&mut comp), out_ptr(&mut q));
    let (qabsp, qropep) = (out_ptr(&mut qabs), out_ptr(&mut qrope));
    let (lc8p, lscalep) = (lc8.ptr_mut(), lscale.ptr_mut() as *mut f32);
    let rcp = rc.ptr_mut() as *mut u16;

    println!("\nM1 window 1 — selection-independent attention phase 1 (EXACT; no approximation):");
    // SAFETY: the engine's own phase-1 launches at the engine's dims, over live device
    // scratch; BENCH_POS is inside the KV_ROWS slabs allocated above.
    let step = |sel: usize, i: usize| unsafe {
        match sel {
            0 => {
                let (p, s) = &qb[i % ROT];
                launch_gemv_fp8(
                    qrp,
                    p.ptr(),
                    s.ptr() as *const f32,
                    N_HEADS * QK,
                    Q_LORA,
                    FP8_BLOCK,
                    qp,
                )
                .expect("q_b")
            }
            1 => {
                launch_gemv_fp8(
                    xnp,
                    kva_p.ptr(),
                    kva_s.ptr() as *const f32,
                    KVL + ROPE,
                    HIDDEN,
                    FP8_BLOCK,
                    cp,
                )
                .expect("kv_a");
                launch_rmsnorm(cp as *const f32, kva_ln.ptr() as *const f32, KVL, 1e-5, cp)
                    .expect("kv_ln");
                launch_rope(cp.add(KVL), 1, ROPE, ROPE, BENCH_POS, ROPE_THETA).expect("rope kv");
                launch_append_kv(
                    cp as *const f32,
                    cp.add(KVL) as *const f32,
                    lc8p,
                    lscalep,
                    rcp,
                    BENCH_POS,
                    KVL,
                    ROPE,
                    nb,
                )
                .expect("append_kv");
            }
            2 => launch_rope(qp.add(NOPE), N_HEADS, QK, ROPE, BENCH_POS, ROPE_THETA)
                .expect("rope query"),
            3 => {
                let (p, s) = &kvb[i % ROT];
                launch_mla_absorb_fp8(
                    qp as *const f32,
                    p.ptr(),
                    s.ptr() as *const f32,
                    N_HEADS,
                    QK,
                    NOPE,
                    VH,
                    KVL,
                    FP8_BLOCK,
                    qabsp,
                )
                .expect("mla_absorb")
            }
            _ => launch_gather_rope(qp as *const f32, qropep, N_HEADS, QK, NOPE, ROPE)
                .expect("gather_rope"),
        }
    };
    let names = [
        "gemv_fp8 q_b[16384x2048]",
        "kv_a+rmsnorm+rope+append",
        "rope query[64x256]",
        "mla_absorb_fp8",
        "gather_rope",
    ];
    let mut total = 0.0;
    for (s, name) in names.iter().enumerate() {
        let us = time(60, &|i| step(s, i));
        let note = if s == 0 || s == 3 {
            format!("  ({ROT} rotating weight copies)")
        } else {
            String::new()
        };
        println!("  {name:<28} {us:9.2} us{note}");
        total += us;
    }
    println!("  == WINDOW 1 total            {total:9.2} us / full layer");
    total
}

/// One MoE layer's expert batch: 9 VQ-int3 experts + the fixed-order reduce. This is
/// the COMPUTE floor of the decoupled window — the engine's real MoE wall is larger (it
/// launches the experts one at a time, each gated on a fetch `Signal`, so it also
/// absorbs host-gated launch bubbles), so the floor keeps the hideability gate
/// conservative.
struct MoeRig {
    _idx: Vec<DeviceBuf>,
    _sc: Vec<DeviceBuf>,
    _cb: Vec<DeviceBuf>,
    descs: DeviceBuf,
    wexpert: DeviceBuf,
    x: DeviceBuf,
    partial: DeviceBuf,
    _out: DeviceBuf,
    cb_ptr: [*const u16; 3],
    h_ptr: *mut f32,
    partial_ptr: *mut f32,
    out_ptr: *mut f32,
    bytes: usize,
    _h: DeviceBuf,
}

impl MoeRig {
    fn new() -> Self {
        let layout = [
            (MOE_INTER, HIDDEN), // gate
            (MOE_INTER, HIDDEN), // up
            (HIDDEN, MOE_INTER), // down
        ];
        let mut idx = Vec::new();
        let mut sc = Vec::new();
        let mut descs: Vec<ExpertDesc> = Vec::new();
        // Expert bytes the batch reads: the same layout formula `quant::vq_expert_bytes`
        // uses, so the GB/s figure is denominated in the engine's own accounting.
        let bytes = SLOTS
            * layout
                .iter()
                .map(|&(o, i)| o * vq_row_bytes(i) + o * vq_groups(i) * 2)
                .sum::<usize>();
        for e in 0..SLOTS {
            let (mut p, mut s) = (Vec::new(), Vec::new());
            for (j, &(o, i)) in layout.iter().enumerate() {
                let seed = 0x4000 + (e * 3 + j) as u64;
                // 12-bit VQ indices drawn from a repeating 4 KiB block, so the gathered
                // codebook entries are periodic rather than uniform over VQ_K=4096 —
                // friendlier to L1 than the engine's gather. Direction: shortens the
                // window (conservative for the hideability gate) and understates the
                // share of bus the window commits (optimistic for the headroom figure).
                let b = dev(&pattern(o * vq_row_bytes(i), seed, |v| {
                    ((v * 127.0) as i8) as u8
                }));
                let g = dev(&pattern(o * vq_groups(i) * 2, seed ^ 0xF, |v| {
                    ((v * 31.0) as i8) as u8
                }));
                p.push(b.ptr());
                s.push(g.ptr() as *const u16);
                idx.push(b);
                sc.push(g);
            }
            descs.push(ExpertDesc {
                gate_indices: p[0],
                gate_scales: s[0],
                up_indices: p[1],
                up_scales: s[1],
                down_indices: p[2],
                down_scales: s[2],
            });
        }
        // Three DISTINCT codebooks at three distinct addresses, as the engine holds
        // (gate/up/down). PERF.md follow-up #1 turns on gate+up = 64 KB not fitting L1,
        // so collapsing them into one buffer would measure a machine we do not have.
        let mut r = Rng(0xC0DE);
        let cb: Vec<DeviceBuf> = (0..3)
            .map(|_| {
                let v: Vec<u16> = (0..VQ_K * VQ_DIM).map(|_| f32_to_f16(r.f())).collect();
                dev(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>())
            })
            .collect();
        let cb_ptr = [
            cb[0].ptr() as *const u16,
            cb[1].ptr() as *const u16,
            cb[2].ptr() as *const u16,
        ];
        // ExpertDesc is repr(C) and the host is LE, so the kernel reads these bytes
        // verbatim — the same upload gpu.rs does through its `as_le_bytes` helper.
        // SAFETY: `descs` is a live Vec<ExpertDesc>; the slice covers exactly its bytes.
        let dbytes = unsafe {
            std::slice::from_raw_parts(
                descs.as_ptr() as *const u8,
                std::mem::size_of_val(&descs[..]),
            )
        };
        let (mut h, mut partial, mut out) = (
            zeros(SLOTS * MOE_INTER * 4),
            zeros(SLOTS * HIDDEN * 4),
            zeros(HIDDEN * 4),
        );
        let (h_ptr, partial_ptr, o_ptr) =
            (out_ptr(&mut h), out_ptr(&mut partial), out_ptr(&mut out));
        MoeRig {
            _idx: idx,
            _sc: sc,
            _cb: cb,
            descs: dev(dbytes),
            wexpert: fill_f32(SLOTS, 0.1),
            x: fill_f32(HIDDEN, 0.02),
            partial,
            _out: out,
            cb_ptr,
            h_ptr,
            partial_ptr,
            out_ptr: o_ptr,
            bytes,
            _h: h,
        }
    }

    fn launch(&self) {
        // SAFETY: descriptors and codebooks are resident for the rig's lifetime; h and
        // partial are sized SLOTS*inter / SLOTS*hidden as the kernels require.
        unsafe {
            launch_moe_expert_range(
                self.x.ptr() as *const f32,
                HIDDEN,
                MOE_INTER,
                0,
                SLOTS,
                self.descs.ptr() as *const ExpertDesc,
                self.cb_ptr[0],
                self.cb_ptr[1],
                self.cb_ptr[2],
                self.wexpert.ptr() as *const f32,
                self.h_ptr,
                self.partial_ptr,
                std::ptr::null_mut(),
            )
            .expect("moe batch");
            launch_moe_reduce(
                self.partial.ptr() as *const f32,
                SLOTS,
                HIDDEN,
                self.out_ptr,
                std::ptr::null_mut(),
            )
            .expect("moe reduce");
        }
    }
}

/// M1 window 2 — the DECOUPLED window: one layer's MLP. Reachable only with the
/// stale-selection approximation, because the indexer of layer `l` for token `t+1`
/// depends on data the MLP of layer `l` at token `t` has not produced yet.
///
/// Measured per LAYER, not per token. An earlier revision quoted the MoE at 210 ms,
/// which is the whole token across 75 MoE layers; the handoff is per full layer, so the
/// window one handoff pair has to fit inside is a single layer's MLP.
///
/// Reported in GB/s as well as µs, because the hideability question is a BANDWIDTH
/// question: what fraction of the bus is already committed during the window we want to
/// hide the indexer inside. Both the 256 GB/s theoretical peak and the ~194.5 GB/s this
/// rig actually reaches on `o_proj` are quoted — the second is the honest denominator
/// for a headroom argument.
fn m1_window_mlp(moe: &MoeRig) -> (f64, f64) {
    let us_moe = time(60, &|_| moe.launch());
    let moe_gbs = moe.bytes as f64 / (us_moe * 1e-6) / 1e9;
    println!("\nM1 window 2 — one layer's MLP (DECOUPLED; needs the stale-selection approx):");
    println!(
        "  MoE batch: {SLOTS} vq3 experts + reduce   {us_moe:9.2} us  <-- x{N_FULL_MOE} full layers"
    );
    println!(
        "    reads {:.1} MB -> {moe_gbs:6.1} GB/s = {:.0}% of the 256 GB/s peak, {:.0}% of the \
         194.5 GB/s achieved",
        moe.bytes as f64 / 1e6,
        moe_gbs / 2.56,
        100.0 * moe_gbs / 194.5,
    );

    // Layers 0/1/2 are FULL indexer layers AND dense-MLP layers, so their window is the
    // dense SwiGLU, not the MoE batch. Measured so all 21 full layers are accounted for
    // rather than assumed comparable.
    let xn = fill_f32(HIDDEN, 0.02);
    let (gate_w, gate_s) = fp8_weight(DENSE_INTER, HIDDEN, 0x51);
    let (up_w, up_s) = fp8_weight(DENSE_INTER, HIDDEN, 0x52);
    let (down_w, down_s) = fp8_weight(HIDDEN, DENSE_INTER, 0x53);
    let (mut g, mut u, mut o) = (
        zeros(DENSE_INTER * 4),
        zeros(DENSE_INTER * 4),
        zeros(HIDDEN * 4),
    );
    let xnp = xn.ptr() as *const f32;
    let (g_ptr, u_ptr, o_ptr) = (out_ptr(&mut g), out_ptr(&mut u), out_ptr(&mut o));
    // SAFETY: the dense-MLP sublayer's four launches at GLM's dense dims, live scratch.
    let us_dense = time(30, &|_| unsafe {
        launch_gemv_fp8(
            xnp,
            gate_w.ptr(),
            gate_s.ptr() as *const f32,
            DENSE_INTER,
            HIDDEN,
            FP8_BLOCK,
            g_ptr,
        )
        .expect("gate");
        launch_gemv_fp8(
            xnp,
            up_w.ptr(),
            up_s.ptr() as *const f32,
            DENSE_INTER,
            HIDDEN,
            FP8_BLOCK,
            u_ptr,
        )
        .expect("up");
        launch_swiglu(g_ptr as *const f32, u_ptr as *const f32, DENSE_INTER, g_ptr)
            .expect("swiglu");
        launch_gemv_fp8(
            g_ptr as *const f32,
            down_w.ptr(),
            down_s.ptr() as *const f32,
            HIDDEN,
            DENSE_INTER,
            FP8_BLOCK,
            o_ptr,
        )
        .expect("down");
    });
    println!(
        "  dense fp8 SwiGLU MLP             {us_dense:9.2} us  <-- x{N_FULL_DENSE} full layers"
    );
    (us_moe, us_dense)
}

/// Validity controls. The numbers in this run are only worth reading if both pass, so
/// both print a verdict rather than leaving the comparison to the reader's eye.
///
/// 1. **Cross-instrument agreement** against `o_proj` fp8 [6144×16384], which
///    benchmarks.md records at [`O_PROJ_REF_US`] / 190.6 GB/s via `examples/dot_bench`.
///    What it proves is narrower than "cross-instrument": `time`, `pattern` and `dev`
///    here are copies of dot_bench's, so this catches environment drift, driver change
///    and rig-construction error — NOT a shared defect in the timing method or fill.
/// 2. **The MALL rotation does something.** `index_score` at nt=32768 touches 8.39 MB
///    per call. Replayed against ONE slab that fits Strix Halo's 32 MB MALL it reads
///    cache, not DRAM. If rotated and un-rotated time the same, the rotation is not
///    buying what it claims and every long-context row is suspect.
fn controls(si: &ScoreInputs, nt: usize) {
    println!("\nCONTROLS — the rig checked against a recorded number and against itself:");
    let (p, s) = fp8_weight(HIDDEN, 16384, 0x0B1);
    let x = fill_f32(16384, 0.5);
    let mut y = zeros(HIDDEN * 4);
    let yp = out_ptr(&mut y);
    // SAFETY: [6144x16384] fp8 weight + its block scales, live device scratch.
    let us = time(60, &|_| unsafe {
        launch_gemv_fp8(
            x.ptr() as *const f32,
            p.ptr(),
            s.ptr() as *const f32,
            HIDDEN,
            16384,
            FP8_BLOCK,
            yp,
        )
        .expect("o_proj");
    });
    let gbs = (HIDDEN * 16384) as f64 / (us * 1e-6) / 1e9;
    let drift = (us - O_PROJ_REF_US).abs() / O_PROJ_REF_US;
    println!(
        "  o_proj fp8[6144x16384]  {us:8.2} us  {gbs:6.1} GB/s  vs benchmarks.md {O_PROJ_REF_US} us  [{}]",
        if drift < 0.05 {
            "ok"
        } else {
            "DRIFT >5% — reconcile before quoting anything below"
        }
    );

    let rot = time(30, &|i| si.launch(i, nt));
    let fixed = time(30, &|_| si.launch(0, nt));
    println!(
        "  index_score nt={nt}: rotating {N_FULL} slabs {rot:8.2} us | ONE slab {fixed:8.2} us  ({:.2}x) [{}]",
        rot / fixed,
        if rot / fixed > 1.05 {
            "ok"
        } else {
            "ROTATION INERT — long-context rows are cache-resident and understate the engine"
        }
    );
}

fn main() {
    // The context sweep. 128 and 2048 sit at or below index_topk, where the engine
    // returns dense before scoring — included so the threshold shows up in the data
    // rather than being asserted.
    let nts = [128usize, 2048, 4096, 8192, 16384, 32768];
    let max_nt = *nts.iter().max().expect("non-empty context sweep");
    println!("GLM-5.2 DSA indexer / NPU-offload M0+M1 microbench");
    println!(
        "dims: hidden {HIDDEN} q_lora {Q_LORA} | index nh {IDX_NH} hd {IDX_HD} topk {IDX_TOPK}"
    );
    println!(
        "layers: 78 total, {N_FULL} FULL indexer ({N_FULL_MOE} MoE + {N_FULL_DENSE} dense), 57 shared"
    );
    println!(
        "rotation: {N_FULL} distinct copies of every per-layer indexer weight + key slab (defeats the 32 MB MALL)"
    );

    let mut si = ScoreInputs::new(max_nt);
    controls(&si, max_nt);
    let (key_us, fixed_us) = m0(&nts, &si);
    m0_host(&nts, &mut si.scores);
    m0c_topk_matched(&[2456, 4096, 5209, 8192, 16384, 32768], &mut si.scores);
    let win1 = m1_window_exact();
    let moe = MoeRig::new();
    let (moe_us, dense_us) = m1_window_mlp(&moe);
    println!("\nM1 summary — window sizes vs the indexer they would have to hide:");
    println!("  window 1 (exact)   phase-1 independent  {win1:9.2} us / full layer (x{N_FULL})");
    println!(
        "  window 2 (stale)   MoE batch            {moe_us:9.2} us / full layer (x{N_FULL_MOE})"
    );
    println!(
        "  window 2 (stale)   dense MLP            {dense_us:9.2} us / full layer (x{N_FULL_DENSE})"
    );
    println!(
        "  indexer/full layer: {:.2} us at ctx<=topk (key path only), {:.2} us + index_score above it",
        key_us,
        key_us + fixed_us
    );
}
