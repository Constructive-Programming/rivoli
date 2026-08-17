//! The ATTENTION third of the HIP ABI wall's fused-block half: the attention operators and
//! the machinery that builds and selects what they read — MLA's absorb/attend/value chain
//! over the fp8 latent cache, GQA over the ring, K3's dense MHA over expanded per-head K/V,
//! the KV slabs' append, the sparse lightning indexer (DSA/MISA), the KV compressor, and the
//! gated-delta-rule (KDA) recurrent-state family.
//!
//! Split out of `hip_blocks.rs` 2026-08-16 under the 800-line file ceiling, when the M9
//! (Kimi-K3) launchers landed; a move, not a rewrite — every pre-existing declaration is
//! byte-identical to the one it replaced, and `hip.rs` re-exports this module, so
//! `rivoli_backend::hip::launch_attend` resolves as before. The cut against `hip_blocks.rs`
//! is "what reads or writes per-position context state" (a KV row, an indexer pool, a
//! conv window, a recurrent S matrix) against "what mixes the residual stream"; the cut
//! against `hip_linalg.rs` is unchanged — a primitive has operands, these have a layout.

// External imports first, crate imports second — the REVERSE of the sibling files' order,
// deliberately: the jscpd gate tokenizes an identical import prologue as a clone across the
// invocation files, and the prologue sits OUTSIDE the exempt region below, which is where it
// must stay (the region is anchored to the macro invocation so additions outside it are
// gated).
use anyhow::Result;
use std::ffi::c_void;

use crate::abi::{CompFinish, CompGeom};
use crate::hip::{abi_ty, ensure_hip_status, launchers};

// Doc links only — see the same block in `hip_linalg.rs` for why these are imports and not
// twenty rewritten comments.
#[allow(unused_imports)]
use crate::{
    hip::{attend_scratch_floats, device_sync},
    hip_linalg::{launch_act_quant_f8_prefix, launch_sigmoid_gate},
};

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — the declarations below, and nothing else in this file.
//
// The third invocation file of the wall, and the same exemption for the same reason:
// `hip_linalg.rs` carries the argument in full and it is not restated here. What IS specific
// to this file is which declarations collide, because that is what a reviewer has to weigh
// against deleting the marker:
//
//   `mla_absorb_fp8` / `mla_value_fp8`  share the fp8 block-scaled `kv_b` operand list. They
//                                       are the two halves of one absorb-attend-value chain
//                                       and read as a pair or not at all. (Listed at
//                                       `hip_blocks.rs` until the 2026-08-16 split moved
//                                       them here with their chain.)
//   the KDA family (M9)                 `short_conv_silu_f32`, `rmsnorm_gate_heads_f32` and
//                                       `gated_delta_recurrent_f32` all end in
//                                       `... , *mut f32, *mut c_void` tails, and the last
//                                       two share `heads`/`head_dim`/`eps` runs — one layer
//                                       family's shapes, three different operators, and the
//                                       C signatures they mirror fix every list.
//
// The region is anchored to the macro invocation, so anything added to this file outside it
// is gated. Inside it, review is the only duplication gate — when adding a launcher, put the
// item immediately under its own doc and re-read the launcher above it, because that is the
// one an insertion breaks (`hip_linalg.rs`'s marker carries the measured incident).
launchers! {
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

    /// **Bidirectional block attention: the DFlash drafter's attend** (M17c). The same GQA
    /// operator as [`launch_gqa_attend`] with one difference that is the whole point — the
    /// drafter denoises a 16-row block at once, so a query attends LATER rows of its own block,
    /// which a causal bound forbids.
    ///
    /// `q`/`out` are `[tq][hq][d]`, `k`/`v` are `[kv_len][hkv][d]` LINEAR — no ring, because the
    /// drafter rebuilds `ctx + block` rows for every drafted block rather than rolling a cache.
    /// That absence is why there is no `ring_cap`: `launch_gqa_attend`'s
    /// `ring_cap >= win + tq - 1` guard exists for a multi-row batch reading a slot it
    /// overwrote, and with no ring the failure mode is unspellable.
    ///
    /// **`q_offset` is the CONTEXT LENGTH, not a decode position, and it is not optional.** The
    /// reference's overlay is `abs(q_idx - kv_idx) <= win` with `q_idx = row + q_offset`, and
    /// `masking_utils.py` takes `q_offset` from the cache when one is present and **0 when it is
    /// not**. Pass `ctx` in decode; pass `0` only to reproduce a vendored S1b golden, which was
    /// captured with `use_cache=False`. At the shipped ctx 4096 / block 16 / win 2048, passing 0
    /// gives **0 of 256** block-vs-block pairs — the block never attends itself, the drafter is
    /// a context-reader, and every shape and byte count still checks out.
    ///
    /// **`kv_len` is taken, not derived**, because this bound runs PAST `pos` and the buffer's
    /// extent is the only thing that stops it.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `q` (`tq·hq·d` f32), `k` and
    /// `v` (`kv_len·hkv·d` f32 each), `out` (`tq·hq·d` f32). `k`/`v` must hold **`kv_len` rows**,
    /// not `q_offset + tq` — the bidirectional upper bound reads up to `q_offset + tq - 1 + win`,
    /// clamped to `kv_len - 1` by the kernel, so a `kv_len` larger than the allocation is an
    /// out-of-bounds read that no argument guard can see.
    ///
    /// Three defects here are invisible to the vendored goldens — the `q_offset` branch, the
    /// inclusive-vs-strict lower edge, and the target's `qk_scale_factor` leaking in. They are
    /// gated deviceless from the shipped config by `crates/cli/tests/drafter_convert.rs`;
    /// `glimmer-reference/drafter-checkpoint.md` carries the measurements.
    launch_gqa_block_attend -> rivoli_gqa_block_attend, "gqa_block_attend" (
        q: *const f32,
        k: *const f32,
        v: *const f32,
        hq: usize as i32,
        hkv: usize as i32,
        d: usize as i32,
        tq: usize as i32,
        q_offset: usize as i32,
        win: usize as i32,
        kv_len: usize as i32,
        scale: f32,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// Dense multi-head attention over explicit per-head K and V: Kimi-K3's gated MLA core.
    /// Ported from `k3:src/backend/hip.rs` (M9).
    ///
    /// `q` is `[heads][d]`, `k` is `[heads][kv][d]`, `v` is `[heads][kv][dv]`, `out` is
    /// `[heads][dv]`. `mask` is an additive `[kv]` row or null. **Not** [`launch_attend`], which is
    /// DeepSeek-V4-Flash's absorbed latent form against an fp8 cache — different cache, different
    /// arithmetic, and K3 caches the expanded k/v on purpose (`k3-architecture.md` §5).
    ///
    /// `scale` is the caller's, and it is a trap: §5 takes it over the **full** head width (192),
    /// not over `qk_nope`. The 64 rope dims are unrotated but still scored, so they are part of
    /// `d` — this signature has no rope argument because the caller concatenates.
    ///
    /// K3's MLA output gate — `attn *= sigmoid(gate)`, between this attend and `o_proj` — is the
    /// EXISTING [`launch_sigmoid_gate`], not a new launcher; `kernels/fwd.hip` records the M9
    /// decision. The trap-10 twin it must not be confused with is
    /// [`launch_rmsnorm_gate_heads_f32`] below: MLA gates with no norm, KDA norms then gates.
    ///
    /// `kv` above 8192 is refused (1004) rather than truncated: scores are staged in LDS. That
    /// bound is the kernel's ceiling and its upgrade path is written at the definition.
    ///
    /// # Safety
    /// `q` is `heads·d` f32, `k` is `heads·kv·d`, `v` is `heads·kv·dv`, `out` is `heads·dv`, and
    /// `mask` — if non-null — is `kv` f32. All live until the next [`device_sync`], and `out` must
    /// not alias any input.
    launch_mha_attend -> rivoli_mha_attend, "mha_attend" (
        q: *const f32,
        k: *const f32,
        v: *const f32,
        mask: *const f32,
        heads: usize as i32,
        kv: usize as i32,
        d: usize as i32,
        dv: usize as i32,
        scale: f32,
        out: *mut f32,
    );

    // ── the gated-delta-rule (KDA) recurrent-state family — Kimi-K3, M9 ─────────────────────
    //
    // Three launchers, one layer family, launched back to back on one stream in S3's loop
    // (`k3-architecture.md` §4): the short convolution feeds the recurrence feeds the gated
    // head norm. Each mutates per-layer decode STATE in place — the conv window, the S
    // matrix — which is what puts them in this file rather than `hip_linalg.rs`. All three
    // ported from `k3:src/backend/hip.rs`.

    /// **One decode step of Kimi-K3's KDA short convolution**, `k3-architecture.md` §4 step 2 —
    /// depthwise over time, SiLU fused into the output, the window advanced in place.
    ///
    /// `cur` and `out` are `channels` f32; `w` is `[channels][taps]`; `win` is `[channels][taps]` —
    /// **shifted left and appended to in place**, so on return it is the window the NEXT token
    /// convolves and is directly comparable with the reference's returned conv cache.
    ///
    /// **Three things a caller can get wrong and the shape cannot express.** The taps run
    /// oldest→newest, so the LAST one multiplies the current token; the window stores pre-conv,
    /// pre-SiLU inputs, so it is not the previous output; and the SiLU is on the accumulated value,
    /// so there is no separate activation to run afterwards. All three are §4 step 2's and all three
    /// are red-proved in `k3:tests/k3_kernels.rs`.
    ///
    /// **`win` is `taps` wide, not `taps - 1`, and that is measured rather than inferred.** The
    /// reference's cache already contains the current token — its last slot is bit-identical to this
    /// token's projection — so a caller sized from §4's C snippet, which keeps `hist` and `cur` in
    /// separate buffers, would allocate one slot per channel too few and run off the end of every
    /// window. This kernel's first draft did exactly that.
    ///
    /// `taps` outside `2..=16` is refused (1002): below 2 the window is one element and the operator
    /// is a per-channel scale, which means the caller read `short_conv_kernel_size` wrong.
    ///
    /// # Safety
    /// Every pointer is a device buffer of the size above, live until `stream` completes. `win` is
    /// written and must not alias `cur`, `w` or `out`; `out` may not alias `cur` (each thread reads
    /// its own channel and writes it, so aliasing is safe by index, but both are `__restrict__`).
    launch_short_conv_silu_f32 -> rivoli_short_conv_silu_f32, "short_conv_silu_f32" (
        cur: *const f32,
        w: *const f32,
        channels: usize as i32,
        taps: usize as i32,
        win: *mut f32,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// **One decode step of the gated delta rule** — the recurrence inside Kimi-K3's 69 KDA layers,
    /// `k3-architecture.md` §4 steps 3-7. `kernels/recurrent.hip` carries the arithmetic and the
    /// argument for its two-pass shape.
    ///
    /// `q`, `k`, `v` and `g` are each `[heads][head_dim]`, `beta_pre` and `a_log` are `[heads]`,
    /// `dt_bias` is `[heads][head_dim]`, `state` is `[heads][head_dim][head_dim]` **updated in
    /// place**, and `out` is `[heads][head_dim]`.
    ///
    /// **The inputs are RAW, and each one is a trap this signature cannot express.** `q` and `k`
    /// arrive pre-L2-norm (`v` is never normed), `beta_pre` arrives pre-sigmoid, and `g` is the bare
    /// gate projection with `a_log`, `dt_bias`, the sigmoid and `lower_bound` all unapplied — the
    /// kernel does every one of those, because fla does and the anchor's captures are taken at fla's
    /// boundary. A caller that pre-normalises anything here gets a silently different recurrence.
    ///
    /// `state` is `[key][value]`, and the axis order is NOT visible in the shape — `head_k_dim ==
    /// head_dim` in the tiny anchor (32) and in the real model (128), so a transposed state is
    /// square and every dimension check passes. It is pinned by measurement instead:
    /// `k3:tests/k3_kernels.rs` scores both interpretations of the anchor's own `initial_state` and
    /// only this one reproduces the reference's output.
    ///
    /// `head_dim` is the block width, so it must be a power of two (guard 1003 — the L2-norm
    /// reduction halves it) and at most 1024 (guard 1002). `lower_bound` outside fla's own
    /// `-5 <= lb < 0` range is refused (1006), NaN included: a NaN bound makes every decay NaN and
    /// nothing downstream would attribute it here.
    ///
    /// # Safety
    /// Every pointer is a device buffer of the size above and must outlive `stream`'s completion.
    /// **Every one is `__restrict__` in the kernel, so none may alias another** — that is the whole
    /// requirement and it is why the reason matters less than it looks. (This said `out` must not
    /// alias `q`, `k`, `v` or `g` "which are read after the state passes begin". Only `v` is: `q`
    /// and `k` are read at the top and `g` before the barrier, all before the first state pass.
    /// Corrected 2026-08-12 by review — the requirement was right and stricter than the index
    /// arithmetic needs, and the reason given for it was wrong, which invites relaxing it.)
    /// `stream` is a live `hipStream_t`, or null for the default stream.
    launch_gated_delta_recurrent_f32 -> rivoli_gated_delta_recurrent_f32,
    "gated_delta_recurrent_f32" (
        q: *const f32,
        k: *const f32,
        v: *const f32,
        g: *const f32,
        beta_pre: *const f32,
        a_log: *const f32,
        dt_bias: *const f32,
        heads: usize as i32,
        head_dim: usize as i32,
        lower_bound: f32,
        state: *mut f32,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// **Kimi-K3's fused gated head norm**, `k3-architecture.md` §4 steps 8-9:
    /// `out = o · rsqrt(mean(o²) + eps) · weight · sigmoid(gate)`, per head.
    ///
    /// `o`, `gate` and `out` are `[heads][head_dim]`; `weight` is `[head_dim]`, shared across heads.
    ///
    /// **NORM THEN GATE, which is trap 10.** MLA gates with no norm — [`launch_sigmoid_gate`] after
    /// [`launch_mha_attend`] — and KDA norms first. The two families must not share a kernel on the
    /// strength of both having a gate. Fused because the reference fuses it
    /// (`FusedRMSNormGated(head_dim, activation='sigmoid')` called as `o_norm(o, g)`), so the
    /// intermediate is unobservable and the anchor can only score the composition.
    ///
    /// `eps` is the caller's, read from the config rather than hardcoded, and it goes on the MEAN —
    /// unlike [`launch_gated_delta_recurrent_f32`]'s L2 norm, which adds its own eps to the SUM.
    /// Negative, NaN and infinite `eps` are refused (1006); zero is legal and exact.
    ///
    /// # Safety
    /// Every pointer is a device buffer of the size above, live until `stream` completes, and none
    /// may alias another — all four are `__restrict__` in the kernel.
    launch_rmsnorm_gate_heads_f32 -> rivoli_rmsnorm_gate_heads_f32, "rmsnorm_gate_heads_f32" (
        o: *const f32,
        gate: *const f32,
        weight: *const f32,
        heads: usize as i32,
        head_dim: usize as i32,
        eps: f32,
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
