//! The RESIDUAL-STREAM third of the HIP ABI wall's fused-block half: one launch per model
//! sub-block on the token's spine — a streaming MoE expert range and the fixed-point
//! accumulator it lands in, and the residual mixers (V4's mHC multi-copy pair, K3's
//! `attn_res`).
//!
//! Split out of `hip.rs` 2026-08-15 under the 800-line file ceiling; a move, not a rewrite —
//! every declaration is byte-identical to the one it replaced and `hip.rs` re-exports this
//! module, so `rivoli_backend::hip::launch_moe_expert_range` resolves as before. Split AGAIN
//! 2026-08-16 under the same ceiling when the M9 (Kimi-K3) launchers landed: the attention
//! blocks — MLA/GQA/MHA, the sparse indexer, the KV compressor, and the KDA recurrent-state
//! family — moved whole to `hip_attn.rs`, on the cut "what mixes the residual stream" against
//! "what reads or writes per-position context state".
//!
//! What separates these from `hip_linalg.rs`'s primitives is a contract, not a size: a block
//! launcher owns intermediate device buffers the caller must size but never reads (`h`,
//! `post`, `comb`, `pre`, the partial scratch), it is stream-ordered as a SET with its
//! neighbours rather than individually, and its `# Safety` block is where the sizing relation
//! between those buffers is written down. A primitive has operands; these have a layout.

use crate::hip::{ExpertDesc, ExpertDescF4, abi_ty, ensure_hip_status, launchers};
use anyhow::Result;
use std::ffi::c_void;

// Doc links only — see the same block in `hip_linalg.rs` for why these are imports and not
// twenty rewritten comments. (`CompFinish`/`CompGeom` and the attention-side links left with
// their launchers in the 2026-08-16 split.)
#[allow(unused_imports)]
use crate::{
    hip::device_sync,
    hip_linalg::{launch_act_quant_f8, launch_vadd},
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
//   `moe_expert_range_f4` / `_f4_situ`  (M9) share everything down to `wexpert` and then
//                                       diverge — `swiglu_limit` against the two betas. The
//                                       separation IS the two models' arithmetic and the
//                                       declarations carry it; the shared prefix is the
//                                       collision. (`mla_absorb_fp8`/`mla_value_fp8`, listed
//                                       here until 2026-08-16, moved to `hip_attn.rs` with
//                                       their chain — that file's marker carries them now.)
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

    /// **Kimi-K3's routed expert range** — the fp4 pair with SiTU-GLU fused and the routing weight
    /// applied AFTER `w2`. `k3-architecture.md` §6, plan §3b; ported from `k3:src/backend/hip.rs`
    /// (M9).
    ///
    /// Four differences from [`launch_moe_expert_range_f4`], each of which passes every length
    /// check if it is got wrong, and each argued at the kernel it lives in (`moe_f4.hip`):
    ///
    /// 1. **`situ_glu`, not `swiglu_clamped`** — and no `swiglu_limit`, because SiTU-GLU's
    ///    saturation is `tanh` inside the function rather than a parameter. The two **betas** are
    ///    refused instead (1006): `<= 0`, NaN and `+/-inf` all fail quietly, `+inf` most quietly of
    ///    all, since `b·tanh(x/b) -> x`.
    /// 2. **The routing weight is applied in pass 2**, to the bf16 `w2` output — `rbf16(dv)·w`, not
    ///    `rbf16(dv·w)`. The reference is `moe_infer`'s
    ///    `.type(topk_weight.dtype).mul_(topk_weight).sum(dim=1)` on the expert's full output.
    ///    `w2` is linear, so this looks like a free reassociation of V4's fold; it is not, because a
    ///    bf16 store sits between the passes and pass 2 sums in fixed point.
    /// 3. **TWO launches, not three.** No `act_quant_f8` between the passes: K3's `w2` takes a plain
    ///    fp32 activation (§6's `k3_matmul_mxfp4(edn, act, ...)`), where V4's fp8 `Linear`
    ///    quantizes its own input. Quantizing here would add an error the reference does not have.
    /// 4. **`expert_in` is the LATENT width** (3584), not `cfg.hidden` (7168) — the experts run in
    ///    latent space, and it is both `w1`/`w3`'s reduction dim and `w2`'s output dim. S1a item 3
    ///    renamed the k3 tree's Rust expert-geometry layer to `expert_in` for this distinction; a
    ///    caller binding `cfg.hidden` positionally here computes a wrong row and is refused by
    ///    nothing.
    ///
    /// `acc` is the latent-wide accumulator [`launch_moe_acc_drain_to`] drains, not the residual.
    ///
    /// # Safety
    /// `x` is `expert_in` live f32 and **16-byte aligned** (an unchecked obligation of the `float4`
    /// loads in `dot_f4_wave_r`, exactly as the fp8 twin requires). `descs` is `n_desc`
    /// `ExpertDescF4`, whose spans must cover `inter x expert_in` fp4 weights and their group-32
    /// scales. `wexpert` is `>= e_start + e_count` f32, `h` is `>= (e_start + e_count) * inter` f32,
    /// `acc` is `expert_in` u64. All outlive `stream`'s completion; `stream` is a live
    /// `hipStream_t`, or null for the default stream.
    ///
    /// **`h` must be 16-byte aligned too, and it is the one a caller is likely to get wrong.** Pass
    /// 2 issues the same `float4` loads on `h_in + e * inter`, so a caller that sub-slices `h` out
    /// of a shared per-layer scratch arena at an unaligned offset faults inside `dot_f4_wave_r`
    /// rather than falling back to the scalar tail — and only at the real widths, where
    /// `inter >= 256` enters the fast path. `x` and `h` must also **not alias**: both are
    /// `__restrict__` in the kernel. Both sentences are in the fp8 twin's doc and were dropped from
    /// this one; restored 2026-08-12 by review, and no fixture allocates `h` unaligned, so neither
    /// has a negative test.
    ///
    /// `nrow` must be 1 (guard 1003) — K3 has no speculative decode (`num_nextn_predict_layers` is
    /// 0), and pass 1 is the one kernel of the pair that is NOT row-templated, so `nrow > 1` is a
    /// two-kernel change rather than a parameter.
    launch_moe_expert_range_f4_situ -> rivoli_moe_expert_range_f4_situ, "moe_expert_range_f4_situ" (
        x: *const f32,
        expert_in: usize as i32,
        inter: usize as i32,
        e_start: usize as i32,
        e_count: usize as i32,
        n_desc: usize as i32,
        descs: *const ExpertDescF4,
        wexpert: *const f32,
        b1: f32,
        b2: f32,
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
    /// > `tests/v4_oracle_targeted.rs::sinkhorn_has_converged_long_before_iteration_20`.
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

    // ── Block Attention Residuals — Kimi-K3's residual mixer (M9) ───────────────────────────

    /// Kimi-K3's Block Attention Residual fold: `out = softmax(<RMSNorm(src_s), fold>) @ src`.
    /// Ported from `k3:src/backend/hip.rs` (M9).
    ///
    /// `src` is `[tokens][nsrc][n]` and `out` is `[tokens][n]`. The softmax mixes the sources
    /// **unnormalised** — `kernels/residual.hip` carries the argument and the defect that prices
    /// it.
    ///
    /// **`src_stride` is the element distance between tokens in `src`, and it is NOT `nsrc·n`.**
    /// `k3-architecture.md` §3 sizes the arena at `[T][9][hidden]` — nine slots per token whatever
    /// the current depth — so a caller with that layout passes `9·n` while the live stack is
    /// `nsrc·n`. A stride below `nsrc·n` is refused (1005): it would overlap consecutive tokens.
    ///
    /// `nsrc` outside `1..=16` is refused (1003) rather than clamped, because a stack larger than
    /// one snapshot per `attn_res_block_size` layers plus the prefix sum means the caller's block
    /// bookkeeping is wrong, and an EMPTY stack means §3's layer-level emptiness guard went
    /// missing. Neither is a case this kernel should quietly define.
    ///
    /// # Safety
    /// `src` is `tokens·nsrc·n` f32, `fold` is `n` f32 and `out` is `tokens·n` f32, all live until
    /// the next [`device_sync`]. `out` must NOT alias `src`: every thread reads all `nsrc` sources
    /// for its column after the block has already written scores, and a caller aliasing them would
    /// have the mixing loop read values it has itself overwritten.
    launch_attn_res -> rivoli_attn_res, "attn_res" (
        src: *const f32,
        fold: *const f32,
        tokens: usize as i32,
        nsrc: usize as i32,
        n: usize as i32,
        src_stride: usize,
        eps: f32,
        out: *mut f32,
    );
}
// jscpd:ignore-end
