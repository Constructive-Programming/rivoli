//! The FUSED-BLOCK half of the HIP ABI wall: one launch per model sub-block — a streaming
//! MoE expert range and the fixed-point accumulator it lands in, the mHC multi-copy
//! residual, MLA and GQA attention over the KV slabs, the sparse lightning indexer, and the
//! KV compressor.
//!
//! Split out of `hip.rs` 2026-08-15 under the 800-line file ceiling; a move, not a rewrite —
//! every declaration is byte-identical to the one it replaced and `hip.rs` re-exports this
//! module, so `rivoli_backend::hip::launch_attend` resolves as before.
//!
//! What separates these from `hip_linalg.rs`'s primitives is a contract, not a size: a block
//! launcher owns intermediate device buffers the caller must size but never reads (`h`,
//! `post`, `comb`, `pre`, the partial scratch), it is stream-ordered as a SET with its
//! neighbours rather than individually, and its `# Safety` block is where the sizing relation
//! between those buffers is written down. A primitive has operands; these have a layout.

use crate::abi::{CompFinish, CompGeom};
use crate::hip::{ExpertDesc, ExpertDescF4, abi_ty, ensure_hip_status, launchers};
use anyhow::Result;
use std::ffi::c_void;

// Doc links only — see the same block in `hip_linalg.rs` for why these are imports and not
// twenty rewritten comments.
#[allow(unused_imports)]
use crate::{
    hip::{attend_scratch_floats, device_sync},
    hip_linalg::{launch_act_quant_f8, launch_act_quant_f8_prefix, launch_vadd},
};

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — the declarations below, and nothing else in this file.
//
// The other half of the wall, and the same exemption for the same reason. `hip_linalg.rs`
// carries the argument in full and it is not restated here; what IS specific to this half is
// which declarations collide, because that is what a reviewer has to weigh against deleting
// the marker:
//
//   `moe_expert_range` / `_i4` / `_f4`  share `x, hidden, inter, e_start, e_count` and then
//                                       diverge — three weight formats, three kernels, one
//                                       range convention. Measured 2026-08-15 with the
//                                       markers deleted: four of the region's clones are
//                                       runs inside this trio.
//   `mla_absorb_fp8` / `mla_value_fp8`  share the fp8 block-scaled `kv_b` operand list. They
//                                       are the two halves of one absorb-attend-value chain
//                                       and read as a pair or not at all.
//
// Factoring either group behind a shared prefix is the thing this wall exists to prevent:
// you could no longer read the C signature off the Rust declaration, and for the MoE trio it
// would move exactly the three declarations that [`ExpertDescF4`](crate::hip::ExpertDescF4)'s
// own note says have no downstream check between them — an `.f4` block dispatched through
// the i4 path decodes e2m1 nibbles at the wrong group size, finite and plausible and wrong.
//
// The region is anchored to the macro invocation, so anything added to this file outside it
// is gated. That is deliberate and is the lesson of the old tree's markers, which bracketed
// nearly a whole file, stayed put while the code moved out from under them, and silently
// exempted whatever drifted in.
//
// **Inside the region, review is the only duplication gate.** jscpd is excluded here by
// construction and clippy only asks whether a `# Safety` section EXISTS, not whether it
// belongs to the item under it — the old tree shipped a launcher carrying THREE concatenated
// docs describing two other kernels' buffers that way. When adding a launcher, put the item
// immediately under its own doc and re-read the launcher above it, because that is the one an
// insertion breaks.
launchers! {
    // ── streaming MoE and its fixed-point accumulator ───────────────────────────────────────

    /// Streaming MoE: gate/up + down for the absolute expert range `[e_start,
    /// e_start+e_count)` on `stream`, atomically accumulating into the shared fixed-point
    /// `acc` row. Drain it with [`launch_moe_acc_drain`] once every range has landed.
    ///
    /// `acc` is `hidden` u64 per token row — ONE row per row, not `e·hidden`. Ranges on
    /// DIFFERENT streams may accumulate concurrently and the result is unchanged, because
    /// integer addition is associative; that is the whole reason this is not an f32 slab plus
    /// a reduce.
    ///
    /// `nrow` token rows (1 or 2) share ONE read of the expert weights — the batched verify
    /// pass a speculative decode needs. Every buffer puts the token row FASTEST:
    /// `x[t·hidden + i]`, `h[(e·nrow + t)·inter + j]`, `wexpert[e·nrow + t]`,
    /// `acc[t·hidden + o]`. `wexpert[e·nrow + t] == 0` means row `t` did not route to expert
    /// `e`, which is how the caller passes the UNION of two tokens' picks with no mask.
    ///
    /// At `nrow == 1` every one of those indices collapses to the single-row form, so the
    /// shipping decode path's layout and arithmetic are unchanged.
    ///
    /// # Safety
    /// Every device pointer (`descs`/codebooks/`wexpert`/`x`/`h`/`acc`) must outlive
    /// `stream`'s completion — await its [`Signal`](crate::gpustream::Signal), and each must
    /// own `nrow` rows in the layout above.
    launch_moe_expert_range -> rivoli_moe_expert_range, "moe_expert_range" (
        x: *const f32,
        hidden: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        descs: *const ExpertDesc,
        gate_cb: *const u16,
        up_cb: *const u16,
        down_cb: *const u16,
        wexpert: *const f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: usize as i32,
        stream: *mut c_void,
    );

    /// int4 counterpart of [`launch_moe_expert_range`]: gate/up + down for the absolute
    /// range `[e_start, e_start+e_count)` on `stream`, decoding int4 (f32 group scales).
    /// `descs` are [`ExpertDesc`]; contributions land in the same fixed-point `acc` row,
    /// so int4 and VQ experts of one layer mix freely within a batch.
    ///
    /// # Safety
    /// Every device pointer (`descs`/packed weights/`wexpert`/`x`/`h`/`acc`) must
    /// outlive `stream`'s completion — await its [`Signal`](crate::gpustream::Signal).
    launch_moe_expert_range_i4 -> rivoli_moe_expert_range_i4, "moe_expert_range_i4" (
        x: *const f32,
        hidden: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        descs: *const ExpertDesc,
        wexpert: *const f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: usize as i32,
        stream: *mut c_void,
    );

    /// DeepSeek-V4 counterpart of [`launch_moe_expert_range_i4`]: FP4 experts (e2m1 nibbles,
    /// one e8m0 scale per 32 weights along the reduction dim) for the descriptor range
    /// `[e_start, e_start+e_count)` on `stream`. Contributions land in the same fixed-point
    /// `acc` row, so this shares [`launch_moe_acc_drain`] with the other two formats.
    ///
    /// **`x` must already be fp8-quantized** by [`launch_act_quant_f8`] — V4 quantizes the
    /// activation in front of every quantized `Linear` and this path cannot do it per output
    /// row (see `linalg.hip::act_quant_f8`). The `h` re-quantization between the two passes IS
    /// done here, because forgetting it is silent.
    ///
    /// `n_desc` is the length of the `descs` array. It exists because `.f4` holds
    /// `n_experts` blocks and **no shared block** — V4's shared expert is fp8 e4m3 at 128x128
    /// and stays resident — unlike `.vq3`/`.i4`, which hold `n_experts + 1` with the shared
    /// expert last. An index one past the end there reads the wrong expert; here it reads
    /// something that is not e2m1 nibbles at all, i.e. the wrong ARITHMETIC.
    ///
    /// `swiglu_limit` comes from the config (`10.0`); the launcher refuses every value that
    /// would disable the clamp — `<= 0`, NaN and `+/-inf` — because an
    /// unclamped SwiGLU on this path is a known silent defect, not a configuration.
    ///
    /// # Safety
    /// Every device pointer (`descs`/packed weights/`wexpert`/`x`/`h`/`acc`) must outlive
    /// `stream`'s completion — await its [`Signal`](crate::gpustream::Signal).
    ///
    /// **`wexpert` and `h` are indexed by the DESCRIPTOR index — whatever placement the
    /// caller chose for `descs` — not by position within `[e_start, e_start+e_count)`.**
    /// (`f4gpu::routed_experts` writes launch order since 2026-08-07; the GLM twins'
    /// callers write absolute ids.) So both must be sized for `n_desc`, not for `e_count`:
    /// `wexpert` is `n_desc·nrow` f32 and `h` is `n_desc·nrow·inter` f32. A caller that read
    /// these as range-relative and allocated `e_count` of them would run off the end the first
    /// time it passed `e_start > 0`, which is the first thing a two-stream pipeline does.
    /// `x` is `nrow` rows of `hidden` f32 and `acc` is `nrow` rows of `hidden` u64.
    ///
    /// `x` and `h` must be **16-byte aligned**, and this is UNCHECKED: `dot_f4_wave_r`'s fast
    /// path gates on the WEIGHT row's 4-byte alignment and then issues `float4` loads on the
    /// activation regardless, so a misaligned `x` faults rather than falling back to the scalar
    /// tail. Every `DeviceBuf` allocation satisfies it (`hipMalloc`); a pointer into the middle
    /// of one need not.
    ///
    /// `x` and `h` must not ALIAS: both are `__restrict__` in the kernel.
    launch_moe_expert_range_f4 -> rivoli_moe_expert_range_f4, "moe_expert_range_f4" (
        x: *const f32,
        hidden: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        n_desc: usize as i32,
        descs: *const ExpertDescF4,
        wexpert: *const f32,
        swiglu_limit: f32,
        h: *mut f32,
        acc: *mut u64,
        nrow: usize as i32,
        stream: *mut c_void,
    );

    /// Drain the fixed-point MoE accumulator into the residual:
    /// `x[o] += gain·(Σ_r acc[r][o])·2⁻⁴⁴`, resetting `acc` to zero for the next layer.
    ///
    /// `rows` is ONE ROW PER STREAM, not per expert. Every expert on a given stream shares a
    /// row; separate streams get separate rows because sharing one measured +825 µs on a
    /// 6-miss layer — same atomic count as a 0-miss layer, so the cost was cache lines
    /// bouncing between queues, not the atomics themselves.
    ///
    /// This IS the residual add on a MoE layer — it replaces [`launch_vadd`] there rather
    /// than running before it, so the convert costs no extra pass and needs no barrier of
    /// its own: the end-of-layer `device_sync` already stands between this and the next
    /// layer's first atomic.
    ///
    /// # Safety
    /// `x` and `acc` hold `n` f32 / `n` u64; EVERY stream that accumulated into `acc` must
    /// already have completed.
    launch_moe_acc_drain -> rivoli_moe_acc_drain_s, "moe_acc_drain" (
        x: *mut f32,
        acc: *mut u64,
        n: usize as i32,
        rows: usize as i32,
        gain: f32,
        stream: *mut c_void,
    );

    /// Drain the fixed-point MoE accumulator into a SEPARATE buffer:
    /// `out[o] = (Σ_r acc[r][o])·2⁻⁴⁴`, resetting `acc` for the next layer.
    ///
    /// **For Kimi-K3, whose MoE block does not end at the residual.** Its routed sum lives in a
    /// 3584-wide latent that must be RMSNormed as an AGGREGATE and up-projected to 7168 before it
    /// can be added to anything — so the sum has to be intercepted, not folded in.
    ///
    /// [`launch_moe_acc_drain`] is the right kernel for GLM and V4 and the wrong one here. The ONE
    /// difference the code cannot show — `=` against `+=` — is argued at `kernels/moe.hip`; the two
    /// kernels now share a templated body, so the rest is not a difference at all.
    ///
    /// **No `gain`, and the sibling's is not an oversight to copy**: a positive scalar applied to
    /// this buffer is erased by the RMSNorm that immediately follows it, so the parameter could not
    /// be used correctly — `kernels/moe.hip` carries the arithmetic. `routed_scaling_factor` is not
    /// it either; that multiplies the router weights inside the sum.
    ///
    /// `n` is the accumulator's row width — `nrow · latent`, and K3's `nrow` is 1. Passing `hidden`
    /// overruns on the LAST row, not the first: with `MOE_ACC_ROWS = 2` and a 3584-wide latent, a
    /// `n = 7168` reads `[0, 7168)` for `r = 0`, which is exactly the whole buffer and in bounds,
    /// then `[7168, 14336)` for `r = 1`, which is entirely outside it. That 2x coincidence between
    /// `hidden/latent` and `rows` is what makes the bug survive the first row.
    ///
    /// # Safety
    /// `out` holds `n` f32 and `acc` holds `rows·n` u64; EVERY stream that accumulated into `acc`
    /// must already have completed. `out` must not alias `acc`, and — unlike the sibling — it does
    /// NOT need to be zeroed first, because this assigns.
    launch_moe_acc_drain_to -> rivoli_moe_acc_drain_to_s, "moe_acc_drain_to" (
        out: *mut f32,
        acc: *mut u64,
        n: usize as i32,
        rows: usize as i32,
        stream: *mut c_void,
    );

    // ── mHC — the multi-copy residual ───────────────────────────────────────────────────────

    /// `model.py::Block.hc_pre` over `s` tokens: reduce the `hc` residual copies to one with
    /// Sinkhorn-normalised learned weights, and emit the `post`/`comb` the matching
    /// [`launch_hc_post`] consumes.
    ///
    /// `iters` is `hc_sinkhorn_iters` from the config. Passing it from `V4Config` rather than
    /// baking it in is what keeps the count from drifting from `config.json`; what the tests
    /// here prove is that the parameter is LIVE (2 and 20 disagree).
    ///
    /// > **CORRECTED 2026-08-07.** This said a numerical comparison *cannot* gate the exact
    /// > value, "at 20 passes a 4x4 positive matrix is far past convergence, so 19 and 20
    /// > agree bit-for-bit". True of the toy fixture, false of the checkpoint — 19 vs 20
    /// > moves 39,893/53,248 of `L0.pre.ffn_norm_out` there. A real-weights golden would
    /// > gate the count; the toy fixture these kernel tests run on cannot. Measurement in
    /// > `tests/v4_oracle.rs::sinkhorn_has_converged_long_before_iteration_20`.
    ///
    /// `hc` is checked against the kernel's `HC_MULT`, not merely passed: `mix_hc = (2+hc)·hc`
    /// is how the mHC weights are packed on disk, so a mismatch is a different checkpoint.
    ///
    /// # Safety
    /// `h` is `s · hc · dim` f32, `fnw` is `(2+hc)·hc` rows of `hc·dim`, `scale` is 3 and
    /// `base` is `(2+hc)·hc`. Outputs: `y` `s·dim`, `post` `s·hc`, `comb` `s·hc·hc`. All must
    /// outlive `stream`'s completion — await its
    /// [`Signal`](crate::gpustream::Signal) — and no output may alias `h`, which is
    /// `__restrict__` in the kernel.
    launch_hc_pre -> rivoli_hc_pre, "hc_pre" (
        h: *const f32,
        fnw: *const f32,
        scale: *const f32,
        base: *const f32,
        s: usize as i32,
        hc: usize as i32,
        dim: usize as i32,
        iters: usize as i32,
        norm_eps: f32,
        hc_eps: f32,
        y: *mut f32,
        post: *mut f32,
        comb: *mut f32,
        stream: *mut c_void,
    );

    /// `model.py::Block.hc_post`: expand the sublayer output `x` back to `hc` residual copies,
    /// mixing the pre-sublayer `residual` through `comb`.
    ///
    /// `comb` is indexed `[source, dest]`. Transposing it leaves every output row a
    /// combination of the same vectors, so no magnitude or norm check can see it.
    ///
    /// # Safety
    /// `x` is `s·dim` f32, `residual` and `y` are `s·hc·dim`, `post` is `s·hc`, `comb` is
    /// `s·hc·hc`. All must outlive `stream`'s completion — await its
    /// [`Signal`](crate::gpustream::Signal).
    ///
    /// **`y` must not alias `residual`.** An in-place residual expansion is the obvious thing to
    /// want and it is wrong twice over: the two are `__restrict__`, and thread `i` writes
    /// `y[i]` while other threads are still reading every source copy of `residual`, with no
    /// barrier between them.
    launch_hc_post -> rivoli_hc_post, "hc_post" (
        x: *const f32,
        residual: *const f32,
        post: *const f32,
        comb: *const f32,
        s: usize as i32,
        hc: usize as i32,
        dim: usize as i32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// `Block.hc_head` (model.py:709-716): collapse `[s, hc, dim]` to `[s, dim]`, bf16-rounded.
    ///
    /// Two kernels on one stream, so no host sync sits between them. `pre` is `s * hc` f32 of
    /// SCRATCH — the gate vector — and is written before it is read.
    ///
    /// # Safety
    /// `h` is `s * hc * dim` live f32; `fnw` is `hc * hc * dim`; `base` is `hc`; `scale` is 1;
    /// `pre` is `s * hc` writable; `y` is `s * dim` writable. None aliases another (every kernel
    /// parameter is `__restrict__`) and all outlive `stream`'s completion. `stream` is a live
    /// `hipStream_t`, or null for the default stream.
    launch_hc_head_collapse -> rivoli_hc_head_collapse, "hc_head_collapse" (
        h: *const f32,
        fnw: *const f32,
        base: *const f32,
        scale: *const f32,
        pre: *mut f32,
        y: *mut f32,
        s: usize as i32,
        hc: usize as i32,
        dim: usize as i32,
        eps: f32,
        hc_eps: f32,
        stream: *mut c_void,
    );

    // ── attention: MLA, GQA, and the KV slabs they read ─────────────────────────────────────

    /// MLA absorb: `qabs[head][i] = Σ_d q[head·qh+d]·kv_b[rbase+d][i]` over kv_b's `nope`
    /// absorb rows (rbase = head·(nope+vh)), head-batched. kv_b fp8-e4m3 block-scaled.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `q` (`h·qh` f32),
    /// `kvb` (`h·(nope+vh)·kvl` bytes), `kvb_scale` (block-scale f32), `qabs` (`h·kvl` f32).
    launch_mla_absorb_fp8 -> rivoli_mla_absorb_fp8, "mla_absorb_fp8" (
        q: *const f32,
        kvb: *const u8,
        kvb_scale: *const f32,
        h: usize as i32,
        qh: usize as i32,
        nope: usize as i32,
        vh: usize as i32,
        kvl: usize as i32,
        block: usize as i32,
        nrow: usize as i32,
        qabs: *mut f32,
    );

    /// MLA value: `ctx[head][j] = Σ_i clat[head][i]·kv_b[rbase+nope+j][i]` over kv_b's `vh`
    /// value rows, head-batched. kv_b fp8-e4m3 block-scaled.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `clat` (`h·kvl` f32),
    /// `kvb` (`h·(nope+vh)·kvl` bytes), `kvb_scale` (block-scale f32), `ctx` (`h·vh` f32).
    launch_mla_value_fp8 -> rivoli_mla_value_fp8, "mla_value_fp8" (
        clat: *const f32,
        kvb: *const u8,
        kvb_scale: *const f32,
        h: usize as i32,
        nope: usize as i32,
        vh: usize as i32,
        kvl: usize as i32,
        block: usize as i32,
        nrow: usize as i32,
        ctx: *mut f32,
    );

    /// MLA flash attention `clat = Σ_t softmax((qabs·L_t + qrope·R_t)·scale)·L_t` over
    /// the fp8-e4m3 latent cache (per-128 block scales) + bf16 roped key, head-batched,
    /// split-KV when `partial` is non-null.
    ///
    /// `rows` (nullable) lists the `nr` attended token indices for DSA sparse attention;
    /// null = dense over the whole `0..nr` causal prefix.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `qabs` (`h·kvl` f32),
    /// `qrope` (`h·rope` f32), `lc8`/`lscale`/`rc` the KV cache (indexed by token — up to
    /// `pos+1` rows; `n_blocks = kvl/128`), `rows` (`nr` u32 or null), `clat` (`h·kvl`
    /// f32), `partial` ([`attend_scratch_floats`] f32 or null = single split).
    launch_attend -> rivoli_mla_attend, "mla_attend" (
        qabs: *const f32,
        qrope: *const f32,
        lc8: *const u8,
        lscale: *const f32,
        rc: *const u16,
        rows: *const u32,
        h: usize as i32,
        nr: usize as i32,
        kvl: usize as i32,
        rope: usize as i32,
        n_blocks: usize as i32,
        scale: f32,
        clat: *mut f32,
        partial: *mut f32,
    );

    /// Append one token's latent (fp8-e4m3 + per-128 block scale) + roped key (bf16) to
    /// the KV slabs at row `pos`. `kvl` must be a multiple of 128 in `[128, 1024]`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `latent` (`kvl` f32), `rope`
    /// (`ropn` f32), `lc8`/`lscale`/`rc` the KV slabs (row `pos` in-bounds; `n_blocks =
    /// kvl/128`).
    launch_append_kv -> rivoli_append_kv, "append_kv" (
        latent: *const f32,
        rope: *const f32,
        lc8: *mut u8,
        lscale: *mut f32,
        rc: *mut u16,
        pos: usize as i32,
        kvl: usize as i32,
        ropn: usize as i32,
        n_blocks: usize as i32,
    );

    /// Gather each head's roped query segment: `qrope[head·ropn+d] = q[head·qh+nope+d]`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `q` (`h·qh` f32), `qrope`
    /// (`h·ropn` f32).
    launch_gather_rope -> rivoli_gather_rope, "gather_rope" (
        q: *const f32,
        qrope: *mut f32,
        h: usize as i32,
        qh: usize as i32,
        nope: usize as i32,
        ropn: usize as i32,
    );

    /// `kernel.py::sparse_attn` — MQA over one `d`-wide entry that is both key and value for
    /// all `h` heads, gathered by `idxs` (`-1` masks a slot), with `sink` entering the
    /// softmax DENOMINATOR only.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `q` (`m * h * d` f32), `kv` (`d` f32
    /// per row, indexed by `idxs`, so at least `max(idxs) + 1` rows), `sink` (`h` f32), `idxs`
    /// (`m * topk` i32), `o` (`m * h * d` f32). `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_gather_attn_shared_kv -> rivoli_gather_attn_shared_kv, "gather_attn_shared_kv" (
        q: *const f32,
        kv: *const f32,
        sink: *const f32,
        idxs: *const i32,
        m: usize as i32,
        h: usize as i32,
        d: usize as i32,
        topk: usize as i32,
        scale: f32,
        o: *mut f32,
        stream: *mut c_void,
    );

    /// Grouped-query attention with a derived causal bound — Muse Glimmer's 32Q/2KV layers.
    ///
    /// Q head `i` reads KV head `i / (hq / hkv)`, which is a per-head BLOCK and not an
    /// interleave; `win > 0` bounds each query to `[pos - win + 1, pos]` INCLUSIVE of its own
    /// position; `win == 0` is a global layer and attends the whole causal prefix. No mask is
    /// taken — the bound is derived, because Glimmer's 131072 context makes a `[tq][s]` mask
    /// larger than the model. The kernel comment carries the four traps.
    ///
    /// **`start_pos` is the absolute position of query row 0, and the cache must be indexed to
    /// match — the two modes index it differently.** With `ring_cap != 0`, slot is
    /// `position % ring_cap`, so `start_pos` stays absolute and the ring may hold any window of
    /// history. With `ring_cap == 0` the slot IS the position, so the cache must run from
    /// position 0: a caller that trims a linear cache to its last `win` rows and leaves
    /// `start_pos` absolute reads past the end, and one that trims without also shifting
    /// `start_pos` attends the wrong rows fluently. `tests/glimmer_attend.rs` does exactly that
    /// shift, deliberately, because the reference hands it a trimmed cache — see `Fixture`.
    /// Both engine paths avoid the question: a global layer holds the whole prefix, a sliding
    /// layer uses the ring.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `q` (`tq * hq * d` f32), `k` and `v`
    /// (each `hkv * d` f32 per slot, so at least `ring_cap` slots with a ring and
    /// `start_pos + tq` without), `out` (`tq * hq * d` f32), none aliasing another (every
    /// kernel parameter is `__restrict__`). `stream` is a live `hipStream_t`, or null for the
    /// default stream.
    ///
    /// A ring must be at least `win + tq - 1` slots, which the launcher enforces: one launch
    /// dereferences the UNION of its rows' windows, so `tq` query rows need `win + tq - 1`
    /// positions live at once and a `win`-slot ring overwrites its own oldest row mid-launch.
    /// Decode (`tq == 1`) is the case where `ring_cap == win` suffices, and it is the only case
    /// the goldens can reach.
    launch_gqa_attend -> rivoli_gqa_attend, "gqa_attend" (
        q: *const f32,
        k: *const f32,
        v: *const f32,
        hq: usize as i32,
        hkv: usize as i32,
        d: usize as i32,
        tq: usize as i32,
        start_pos: usize as i32,
        win: usize as i32,
        ring_cap: usize as i32,
        scale: f32,
        out: *mut f32,
        stream: *mut c_void,
    );

    // ── the sparse lightning indexer (DSA / MISA) ───────────────────────────────────────────

    /// Append one indexer key row (bf16) at `pos`: `kcache[pos·hd+i] = bf16(k[i])`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `k` (`hd` f32), `kcache`
    /// (row `pos` in-bounds).
    launch_index_append -> rivoli_index_append, "index_append" (
        k: *const f32,
        kcache: *mut u16,
        pos: usize as i32,
        hd: usize as i32,
    );

    /// Score every cached token against the indexer query heads:
    /// `scores[t] = Σ_{h∈active} w[h]·wscale·ReLU((q_h·k_t)·dscale)`. `heads` (nullable)
    /// lists the `nact` active heads (MISA); null = all `nh` heads (DSA).
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `q` (`nh·hd` f32), `w` (`nh`
    /// f32), `kcache` (`nt·hd` bf16), `heads` (`nact` u32 or null), `scores` (`nt` f32).
    launch_index_score -> rivoli_index_score, "index_score" (
        q: *const f32,
        w: *const f32,
        kcache: *const u16,
        heads: *const u32,
        nt: usize as i32,
        nh: usize as i32,
        nact: usize as i32,
        hd: usize as i32,
        wscale: f32,
        dscale: f32,
        scores: *mut f32,
    );

    /// Select the DSA attend row set on device: `rows[0..min(k,nt))`, ASCENDING by index.
    ///
    /// Writes device-side only — no D2H, no host top-k, and no `device_sync`: the attend
    /// consumes `rows` on the same stream, so program order is the whole requirement.
    ///
    /// **Intended** to be bit-identical to the `topk_into(..) + sort_unstable()` it
    /// replaces; `tests/kernel.rs::index_topk_matches_host_selection` is the gate for that
    /// claim. The tiebreak rule and its rationale live at the kernel, once.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `scores` (`nt` f32), `rows`
    /// (at least `min(k, nt)` u32 — the kernel writes exactly that many).
    launch_index_topk -> rivoli_index_topk, "index_topk" (
        scores: *const f32,
        nt: usize as i32,
        k: usize as i32,
        rows: *mut u32,
    );

    /// Fold token `t`'s indexer key into its MISA block pool running mean.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `k` (`hd` f32), `pool`
    /// (block `t/MISA_BLOCK` in-bounds).
    launch_index_pool_push -> rivoli_index_pool_push, "index_pool_push" (
        k: *const f32,
        pool: *mut f32,
        t: usize as i32,
        hd: usize as i32,
    );

    /// MISA head-router estimate `e[j] = mean_b |w[j]·ReLU(q_j·k̄_b)|` over the block pool.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `q` (`nh·hd` f32), `w` (`nh`
    /// f32), `pool` (`m_blocks·hd` f32), `e` (`nh` f32).
    launch_index_head_route -> rivoli_index_head_route, "index_head_route" (
        q: *const f32,
        w: *const f32,
        pool: *const f32,
        m_blocks: usize as i32,
        nh: usize as i32,
        hd: usize as i32,
        e: *mut f32,
    );

    // ── the KV compressor ───────────────────────────────────────────────────────────────────

    /// The state deposit of `Compressor.forward` — **both phases**, which are one operation
    /// distinguished only by `slot0`.
    ///
    /// A prefill of `s` tokens deposits its `s % ratio` trailing rows starting at slot 0; a
    /// decode deposits its single row at slot `start_pos % ratio`. See
    /// `kernels/kvcompress.hip::kv_compress_deposit` for why that is a unification and not a
    /// coincidence.
    ///
    /// Must be launched on **every** call, including one that emits no block: the reference
    /// writes the state and only then returns `None`. At ratio 128 that is every prompt under
    /// 128 tokens and 127 of every 128 decode steps.
    ///
    /// Refuses `s <= 0` (guard 1005) and a `slot0` whose run would leave the `[ratio, cd]`
    /// `ape` table (guard 1008).
    ///
    /// # Safety
    /// `kv`/`score` are `s · p.cd()` live f32; `ape` is `p.ratio() · p.cd()`; the two state
    /// buffers are `p.state_len()` writable f32. None may alias another — every kernel
    /// parameter is `__restrict__`. `p` is read host-side before the launch; the device buffers
    /// must outlive `stream`'s completion. `stream` is a live `hipStream_t`, or null for the
    /// default stream.
    launch_kv_compress_deposit -> rivoli_kv_compress_deposit, "kv_compress_deposit" (
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        kv_state: *mut f32,
        score_state: *mut f32,
        p: &CompGeom as *const CompGeom,
        s: usize as i32,
        slot0: usize as i32,
        stream: *mut c_void,
    );

    /// Prefill pooling for `nblk` compressed blocks — `overlap_transform`, the per-feature
    /// softmax over the pooling window, the bf16 store, `RMSNorm`, and the RoPE at each block's
    /// FIRST absolute position.
    ///
    /// Does **not** run `act_quant`; call [`launch_act_quant_f8_prefix`] over dims `[0, d - rd)` at
    /// block 64 afterwards, which is the order and the partial extent model.py:373-378 uses.
    ///
    /// Refuses `nblk <= 0` (guard 1006) rather than launching nothing and returning success,
    /// which would hand the caller an unwritten `out`.
    ///
    /// # Safety
    /// `kv`/`score` are at least `nblk · p.ratio() · p.cd()` live f32; `ape` is
    /// `p.ratio() · p.cd()`; `f` satisfies [`CompFinish`]'s field contract with `out` sized
    /// `nblk · p.d()` and `freqs` covering position `(nblk - 1) · p.ratio()`. None may alias
    /// another. All must outlive `stream`'s completion; `stream` is a live `hipStream_t`, or
    /// null for the default stream.
    launch_kv_compress_prefill -> rivoli_kv_compress_prefill, "kv_compress_prefill" (
        kv: *const f32,
        score: *const f32,
        ape: *const f32,
        f: &CompFinish as *const CompFinish,
        p: &CompGeom as *const CompGeom,
        nblk: usize as i32,
        stream: *mut c_void,
    );

    /// Pool one COMPLETED decode window out of the compressor state into a single block, and
    /// slide the window.
    ///
    /// Reads no activation: this step's row was already deposited by
    /// [`launch_kv_compress_deposit`], `ape` included. Call **only** when
    /// `(start_pos + 1) % ratio == 0`; the launcher refuses otherwise (guard 1009) rather than
    /// pooling a half-filled window into finite, plausible, wrong numbers.
    ///
    /// # Safety
    /// The two state buffers are `p.state_len()` f32 and are read-modify-written; `f` satisfies
    /// [`CompFinish`]'s field contract with `out` sized one row of `p.d()` and `freqs` covering
    /// position `(start_pos / ratio) * ratio`. None may alias another. All must outlive
    /// `stream`'s completion; `stream` is a live `hipStream_t`, or null for the default stream.
    launch_kv_compress_decode -> rivoli_kv_compress_decode, "kv_compress_decode" (
        kv_state: *mut f32,
        score_state: *mut f32,
        f: &CompFinish as *const CompFinish,
        p: &CompGeom as *const CompGeom,
        start_pos: usize as i32,
        stream: *mut c_void,
    );
}
// jscpd:ignore-end
