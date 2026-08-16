//! The PRIMITIVE half of the HIP ABI wall: one launch per matvec, matmul, embedding row,
//! elementwise op, activation quantizer, normalization or RoPE — each one an operator the
//! model graph names, taking buffers and extents and nothing model-shaped.
//!
//! Split out of `hip.rs` 2026-08-15 under the 800-line file ceiling. It is a move: every
//! declaration below is byte-identical to the one it replaced, in the same `launchers!` DSL,
//! and `hip.rs` re-exports this module, so `rivoli_backend::hip::launch_gemv_fp8` and
//! `waist.rs`'s glob resolve exactly as before. The cut is by cohesion — the other half,
//! `hip_blocks.rs`, holds the launchers that fuse a whole model sub-block, which is a
//! different contract: those take a stream and own their own intermediate buffers, these
//! take one operator's operands.
//!
//! Read `hip.rs` first for the DSL, the descriptor structs and the return-code check; the
//! exemption argument for both invocation files is under this file's opening marker.

use crate::hip::{abi_ty, ensure_hip_status, launchers};
use anyhow::Result;
use std::ffi::c_void;

// Doc links only. An intra-doc link resolves in the module it is WRITTEN in, so the moment
// the wall became three files every `[`device_sync`]` in these `# Safety` blocks — sixteen of
// them — became an unresolved link, silently: rustdoc warns, and nothing in this workspace
// runs rustdoc. Importing the names is the one-line fix that leaves the prose byte-identical;
// spelling `crate::hip::` into sixteen comments is the same edit made twenty times, and it is
// the safety contract that would carry the noise.
//
// `NULL_STREAM` is in the list for a different reason: `launch_gemv_f32`'s doc has linked to
// it since before the split and the link has never resolved, because the constant lives above
// the backend in `waist.rs` and `hip.rs` had no reason to import it. Naming it here costs
// nothing and closes it. The five `crate::gpustream::Signal` links in `hip_blocks.rs` are the
// same vintage and are NOT closed this way — that path is simply wrong (`Signal` is defined in
// `waist.rs`), so fixing them means editing five `# Safety` blocks, which is a claim-touching
// edit and does not belong in a file move.
#[allow(unused_imports)]
use crate::{
    NULL_STREAM,
    hip::{device_sync, memcpy_dtod_async},
    hip_blocks::launch_moe_acc_drain,
};

// jscpd:ignore-start
//
// EXEMPT FROM THE DUPLICATION GATE — the declarations below, and nothing else in this file.
//
// > **SPLIT 2026-08-15, and the region count went from two to three.** The wall outgrew the
// > 800-line file ceiling, so its one macro invocation became two — the primitives here and
// > the fused blocks in `hip_blocks.rs` — and a marker pair cannot span files. Nothing about
// > the argument below changed: it is the argument for BOTH halves, and `hip_blocks.rs`'s
// > marker carries the collisions specific to that half rather than a second copy of this
// > text. The alternative was to partition the wall by which declarations jscpd happens to
// > see, keeping one half unexempted — that is a gate authoring the architecture, and it
// > would have split `swiglu`/`swiglu_clamped_bf16` and `rmsnorm_single`/`_centered_single`
// > across files precisely because they collide, which is the opposite of why they are
// > documented as pairs.
//
// This is the surviving half of an argument made before Track 2 and re-tested after it. The
// original note exempted two regions, the `extern` block and the wrappers, on two grounds. One
// of them is now obsolete: "every kernel is written out twice, once as a decl and once as a
// wrapper" was true and is what `launchers!` above removed — 1307 code lines to 685, with the
// comment count going UP.
//
// The other ground is unchanged and is why this marker is still here, though the number that
// came with it was wrong and is corrected below.
//
// > **MEASURED 2026-08-06, correcting the inherited claim.** The original note said "roughly 25
// > are DIFFERENT kernels that merely take the same shape — `gemv_fp8`/`i8`/`i4`/`vq` all take
// > `x, packed, scale, o_dim, i_dim, y`". Counted rather than asserted: the declarations below have
// > zero exact duplicates, and exactly ONE PAIR shares even a type sequence under different names
// > (`act_quant_f8` and `act_quant_f4_rotated`, both `*mut f32, i32, i32, *mut c_void`). The `gemv`
// > family does not actually agree: `gemv_vq` takes seven parameters (`indices`, `scales`,
// > `codebook`), `gemv_i4` takes six.
// >
// > **No count of the declarations here, deliberately** (it said "46, 41 when this was written").
// > It went four stale in silence, a fifth the day `launch_moe_acc_drain_to` landed, and a sixth
// > when that kernel's `gain` was deleted — while the argument this marker rests on, below, never
// > depended on it. A hand-maintained tally of the file it sits in bills a line of upkeep to every
// > commit that touches the file and catches nothing.
//
// What jscpd matches is a shared PREFIX, not a shared list. `moe_expert_range` and
// `moe_expert_range_i4` agree on `x, hidden, inter, e_start, e_count, descs` and then diverge.
// Re-measured with the markers deleted: **11 clones, every one a prefix run between two
// adjacent declarations**, none of them logic.
//
// > **RE-MEASURED 2026-08-15, because an inherited count is an unverified one.** Markers
// > deleted, jscpd at `minTokens 15`, twice: **19 clones** over the single file as it stood
// > (18 across these declarations, one inside the `extern` block region `hip.rs` keeps), and
// > **20** over the three files it became — the extra one is a run that was internal before
// > and now crosses the two halves, which is the split showing up in the measurement and
// > nothing else. The shape of the finding is unchanged and is the thing to carry forward:
// > every one is a parameter-list prefix run between two declarations, and not one is logic.
// > The count is the wall growing, not the argument weakening. That 20 → 0 with the markers
// > restored is what says all three regions are load-bearing rather than decorative.
//
// **A named-prefix DSL would work and is still refused.** `@moe_head, gate_cb: …` is a
// tt-muncher away and would save perhaps 40-60 lines across the ~10 launchers sharing a prefix.
// It costs the one property this wall exists for — you could no longer read the C signature off
// the Rust declaration, and "the mirroring IS the contract" is the whole argument for the file.
// The clearest candidate is also the worst one: `moe_expert_range` takes `*const ExpertDesc` and
// `_f4` takes `*const ExpertDescF4` plus an extra `n_desc`, and [`ExpertDescF4`]'s own note says
// dispatching an `.f4` block through the i4 path decodes e2m1 nibbles at the wrong group size
// with **no downstream check that could find it**. Factoring their common prefix behind one name
// moves those two toward each other, which is the opposite of what that separation is for.
//
// **The region is narrower than what it replaced, and deliberately so.** The old markers
// bracketed lines 67-524 and 602-2061 — nearly the whole file. Track 2 moved the code out from
// under them and they stayed where they were, silently exempting whatever had drifted into
// those line ranges. That is the "stale exemption is a hole in the gate" failure happening
// inside the change that was supposed to prevent it. This one is anchored to the macro
// invocation, so anything added outside it is gated.
//
// **What it costs, measured the hard way 2026-08-12.** A live exemption is a hole in the gate
// too, and the doc comments are inside this one. The Glimmer port added `logit_softcap` and
// `sigmoid_gate` here by inserting each item ABOVE an existing one, which detached that item's
// doc; clippy's `missing_safety_doc` caught the orphan both times, and both times it was
// "fixed" by pasting a fresh copy of the doc below rather than by noticing the old one was
// still sitting above the wrong item. Net result, until a review found it: **25 duplicated
// comment lines**, `launch_rope_split_half`'s doc present twice differing by one word, and
// `launch_logit_softcap` carrying THREE concatenated docs with THREE `# Safety` sections — the
// first two describing a `base`/`count`/`stride` buffer and a non-aliasing `g` operand that
// function does not have. A launcher whose stated safety contract belongs to two other kernels
// is worse than one with no comment, because it reads as authority.
//
// Neither gate could see it. jscpd is excluded by this marker; clippy only asks whether a
// `# Safety` section EXISTS. **So inside this region, review is the only duplication gate** —
// and when adding a launcher, put the item immediately under its own doc and re-read the
// launcher above it, because that is the one an insertion breaks.
//
// ── WHICH MODEL USES WHICH KERNEL ───────────────────────────────────────────────────────
//
// Kernels are named for what they DO, not for the model that introduced them. Until
// 2026-08-09 fifteen of them carried a `v4_` prefix and were renamed for their mechanism.
// The model affiliation those prefixes carried is now `tests/kernel_coverage.rs::OWNERS`,
// which maps every launcher to the engine source files that call it and is CHECKED.
//
// It is a test and not a comment here for a measured reason: the first draft of this change
// put the lists in this file as prose, and a review found SIX of them wrong on the day they
// were written — `swiglu` claimed GLM-only while `f4gpu.rs` calls it, `swiglu_clamped_bf16`
// claimed a V4 caller it does not have, and `act_quant_f8`/`vadd`/`flag_nonfinite` were each
// filed under the wrong engine. A hand-maintained ownership list is exactly the artefact
// `tests/kernel_coverage.rs` already refuses to carry ("an exemption asserts nothing and
// rots silently"), and trading a name that was inaccurate-but-self-maintaining for a comment
// that is inaccurate and hand-maintained would have made this refactor a net loss.
//
// The pairs worth knowing before reaching for either are documented AT the kernels, because
// that is where the arithmetic that separates them is: `rmsnorm_single`/`rmsnorm_batch` (one
// statistic vs one per row), `gemv_fp8`/`gemv_fp8_bf16` (input slicing and store dtype),
// `swiglu`/`swiglu_clamped_bf16` (not one parameter apart at any `L`), and
// `rope_interleave`/`rope_adjacent` (half-split vs adjacent pair convention).
launchers! {
    // ── matvec and matmul ───────────────────────────────────────────────────────────────────

    /// VQ-int3 GEMV `y = W·x` (group scales applied inside the decode).
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`].
    launch_gemv_vq -> rivoli_gemv_vq, "gemv_vq" (
        x: *const f32,
        indices: *const u8,
        scales: *const u16,
        codebook: *const u16,
        o_dim: usize as i32,
        i_dim: usize as i32,
        y: *mut f32,
    );

    /// fp8-e4m3 block-scaled GEMV `y = W·x` (attention/dense projections).
    ///
    /// `nrow` token rows (1 or 2) share ONE read of the weights: `x[r·i_dim + i]` →
    /// `y[r·o_dim + o]`. That read is the cost — the attention projections are 165 MB of fp8
    /// per layer against a 24 KB `x` — so a batched verify pass is where this earns its
    /// keep. At `nrow == 1` both indices are the single-row ones and nothing changes.
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `x` (`nrow·i_dim` f32),
    /// `packed` (`o_dim·i_dim` bytes), `scale` (block-scale f32), `y` (`nrow·o_dim` f32).
    launch_gemv_fp8 -> rivoli_gemv_fp8, "gemv_fp8" (
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        block: usize as i32,
        nrow: usize as i32,
        y: *mut f32,
    );

    /// fp8-e4m3 GEMV with 128x128 block scales and a bf16-rounded output.
    ///
    /// `x` is `m` rows of `groups` consecutive `k`-wide slices; output row `j` reads slice
    /// `j / (n_out / groups)` of its row. `groups = 1` is a plain `Linear` (every output row
    /// sees the whole activation); `groups = o_groups` is the grouped `wo_a` einsum, whose
    /// input groups are contiguous runs of heads and so need no gather.
    ///
    /// Does NOT quantize the activation — [`launch_act_quant_f8_prefix`] is a separate launch where
    /// the reference performs one, and `wo_a` gets none at all.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`m * groups * k` f32), `w`
    /// (`n_out * k` bytes), `wscale` (`ceil(n_out/block) * ceil(k/block)` f32), `out`
    /// (`m * n_out` f32). The `x` bound is exact and follows from the arguments; an earlier
    /// signature took the row and group strides separately, where the in-bounds relation was
    /// a three-way inequality nothing checked. `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_gemv_fp8_bf16 -> rivoli_gemv_fp8_bf16, "gemv_fp8_bf16" (
        x: *const f32,
        w: *const u8,
        wscale: *const f32,
        m: usize as i32,
        n_out: usize as i32,
        k: usize as i32,
        block: usize as i32,
        groups: usize as i32,
        out: *mut f32,
        stream: *mut c_void,
    );

    /// Group-scaled int4 GEMV `y[o] = Σ_i x·(nibble-8)·scale[o, i/I4_GROUP]` — the MoE
    /// `dot_i4_wave` wave-per-row, for the dot-throughput microbench. `scale` is
    /// `o_dim · i4_groups(i_dim)` f32.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`].
    launch_gemv_i4 -> rivoli_gemv_i4, "gemv_i4" (
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        y: *mut f32,
    );

    /// Per-row int8 GEMV `y = W·x` (lm_head → logits).
    ///
    /// # Safety
    /// Async device pointers live until the next [`device_sync`]: `x` (`i_dim` f32),
    /// `packed` (`o_dim·i_dim` bytes), `scale` (`o_dim` f32), `y` (`o_dim` f32).
    launch_gemv_i8 -> rivoli_gemv_i8, "gemv_i8" (
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        nrow: usize as i32,
        y: *mut f32,
    );

    /// f32 GEMV `y = W·x` (the MoE router gate).
    ///
    /// Takes a `stream`, and it is the only one of GLM's GEMVs that does. The reason is V4: this
    /// is the launcher `Gate.forward` maps to (`linear(x.float(), weight.float())`), so on the V4
    /// layer path it sits between `launch_rmsnorm_batch` and `launch_moe_expert_range_f4`, which both
    /// take one. Left on the null stream it would read a norm that no stream had been waited on —
    /// rivoli's streams are `hipStreamNonBlocking` — giving wrong logits and therefore a wrong
    /// SELECTION, which is the one V4 defect class no downstream numeric comparison can attribute.
    ///
    /// Not a V4-only twin launcher, and the contrast with [`memcpy_dtod_async`] earlier in this
    /// file is the reason: those two are separate entry points because they differ in whether the
    /// HOST blocks, which is a contract split the arena relocation depends on. Two spellings of
    /// one dispatch differing only in a stream handle has no such justification.
    ///
    /// All seven existing call sites pass [`NULL_STREAM`] and are unchanged in behaviour, on both
    /// backends: `vk::launch_gemv_f32` routes the same argument through `Q::parse`, which maps 0
    /// and 1 alike to `Q::Main` for exactly this case.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion. `stream` is a live `hipStream_t`, or
    /// null for the default stream.
    launch_gemv_f32 -> rivoli_gemv_f32, "gemv_f32" (
        x: *const f32,
        w: *const f32,
        o_dim: usize as i32,
        i_dim: usize as i32,
        nrow: usize as i32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// `out[m, n] = x[m, k] · w[n, k]^T` with `w` in **bf16** — the un-quantized `F.linear`
    /// path, which is the one `Compressor.wkv`/`wgate` take (`Linear(..., dtype=float32)`,
    /// model.py:302).
    ///
    /// Deliberately NOT [`launch_gemv_fp8_bf16`]: that one quantizes the activation to fp8 at
    /// block 128 in front of the GEMM, which the reference does only for quantized `Linear`s.
    /// Sending the compressor through it would introduce a quantization the reference never
    /// applies, and the resulting error would be indistinguishable from a pooling bug.
    ///
    /// # Safety
    /// `x` is `m · k` live f32, `w` is `n · k` live u16, `out` is `m · n` writable f32, none
    /// aliasing another (every kernel parameter is `__restrict__`), all live until `stream`
    /// completes. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_gemm_bf16 -> rivoli_gemm_bf16, "gemm_bf16" (
        x: *const f32,
        w: *const u16,
        out: *mut f32,
        m: usize as i32,
        n: usize as i32,
        k: usize as i32,
        stream: *mut c_void,
    );

    // ── embedding rows ──────────────────────────────────────────────────────────────────────

    /// int8 embedding row lookup: `x[i] = embed[token][i]·scale[token]`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `packed` (`≥(token+1)·hidden`
    /// bytes), `scale` (`≥token+1` f32), `x` (`hidden` f32).
    launch_embed_i8_row -> rivoli_embed_i8_row, "embed_i8_row" (
        packed: *const u8,
        scale: *const f32,
        token: usize as i32,
        hidden: usize as i32,
        x: *mut f32,
    );

    /// `Transformer.forward` 914-916: gather token `token`'s bf16 embedding row and broadcast it
    /// into `hc` copies. `x` receives `hc * hidden` f32.
    ///
    /// # Safety
    /// `w` is `>= (token + 1) * hidden` live u16, `x` is `hc * hidden` writable f32, they do not
    /// alias (both are `__restrict__`), and both outlive `stream`'s completion. `stream` is a
    /// live `hipStream_t`, or null for the default stream.
    launch_embed_bf16_row_bcast -> rivoli_embed_bf16_row_bcast, "embed_bf16_row_bcast" (
        w: *const u16,
        token: usize as i32,
        hidden: usize as i32,
        hc: usize as i32,
        x: *mut f32,
        stream: *mut c_void,
    );

    // ── elementwise and reductions ──────────────────────────────────────────────────────────

    /// `x += y` — the residual add on a dense-MLP layer.
    ///
    /// It said "`--moe-gain != 1` takes `launch_vaxpy` instead" until 2026-08-06, and that had
    /// been false for some time: `--moe-gain` folds into [`launch_moe_acc_drain`]'s `gain`
    /// multiply, which is the MoE layer's residual add, and the 3 dense layers must NOT be
    /// attenuated with it. `vaxpy` was deleted — this comment is why it survived, since a
    /// launcher that a doc comment says is on a live path reads as reachable to every grep.
    ///
    /// # Safety
    /// `x` and `y` must be device pointers to at least `n` f32.
    launch_vadd -> rivoli_vadd, "vadd" (
        x: *mut f32,
        y: *const f32,
        n: usize as i32,
    );

    /// Greedy argmax over `logits[0..n]` → (`out_idx`, `out_val`); lowest index on a
    /// tie, NaN never wins (matches the host fold).
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`]: `logits` (`n` f32),
    /// `out_idx` (one i32), `out_val` (one f32).
    launch_argmax -> rivoli_argmax, "argmax" (
        logits: *const f32,
        n: usize as i32,
        out_idx: *mut i32,
        out_val: *mut f32,
    );

    /// Record `tag` in `*flag` if any of `x[0..n]` is non-finite (first writer wins).
    ///
    /// The localiser for the intermittent non-finite-logits bug. Adds no sync — the caller
    /// reads `flag` on the argmax D2H the tail already pays — because the host-copy
    /// alternative (`--checksum-x`) perturbs timing enough to hide the fault entirely.
    ///
    /// # Safety
    /// `x` must be `n` device f32; `flag` one device u32, zeroed before the run.
    launch_flag_nonfinite -> rivoli_flag_nonfinite, "flag_nonfinite" (
        x: *const f32,
        n: usize as i32,
        tag: u32,
        flag: *mut u32,
    );

    /// Batched VQ encode (offline converter accelerator): `idx[i] = argmin_k …`.
    ///
    /// # Safety
    /// Device pointers live until the next [`device_sync`].
    launch_vq_encode -> rivoli_vq_encode, "vq_encode" (
        sub: *const f32,
        codebook: *const f32,
        cbnorm: *const f32,
        n: usize as i32,
        idx: *mut u16,
    );

    // ── activations and gates ───────────────────────────────────────────────────────────────

    /// SwiGLU combine `h = silu(g)·u` (GLM's dense fp8 MLP, and V4's shared-expert chain;
    /// safe in place, `h` may alias `g`).
    ///
    /// `stream` is trailing and null is the null stream, per the `mla.hip` V4-launcher
    /// contract: V4's shared-expert chain is stream-ordered as a SET (§M7), and GLM's one
    /// call site passes null — unchanged to the byte.
    ///
    /// # Safety
    /// Device pointers (`g`, `u`, `h` each `n` f32) live until the next [`device_sync`];
    /// `stream` is null or a live stream ordering every producer of `g`/`u`.
    launch_swiglu -> rivoli_swiglu, "swiglu" (
        g: *const f32,
        u: *const f32,
        n: usize as i32,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// DeepSeek-V4's `Expert.forward` combine, for the resident fp8 **shared** expert:
    /// `h = bf16( silu(min(bf16(g), limit)) · clamp(bf16(u), ±limit) )`. Safe in place.
    ///
    /// `MoE.__init__` hands `swiglu_limit` to `shared_experts` as well as to the routed ones
    /// (model.py:632) and `Expert.forward` clamps both, so the shared expert needs the same
    /// clamped arithmetic `launch_moe_expert_range_f4` already runs on the fp4 routed experts.
    /// **NOT YET WIRED, as of 2026-08-05, and this launcher existing does not fix anything by
    /// itself.** *[CORRECTED 2026-08-08: the V4 loop exists now and calls the unclamped
    /// [`launch_swiglu`] from its shared-expert chain — `f4gpu.rs::shared_expert` names the
    /// deviation (`Defect::SwigluUnclamped`) at the call. So this launcher's caller-to-be
    /// exists and still does not call it; the sentence below about what the wiring must do
    /// stands.]* The clamped combine is available and gated (`tests/f4_kernel.rs` §7), and
    /// the shared expert's fix is to call THIS and not [`launch_swiglu`] — because that one
    /// gives `v4oracle::Defect::SwigluUnclamped` on one contribution in seven of all 43
    /// layers, fluent and wrong.
    ///
    /// # Why this is not [`launch_swiglu`] with a `limit`
    ///
    /// **At no value of `limit` would the two agree**, so a parameter could not have expressed
    /// it. [`launch_swiglu`] is `(g/(1+e^-g))·u` and rounds nothing. Three differences besides
    /// the clamp, each of them a defect the oracle names:
    ///
    /// - both operands are bf16-rounded **before** the clamp (`Linear` stores bf16 and
    ///   `Expert.forward` reads it back with `.float()`) — `Defect::NoBf16Rounding`;
    /// - the product is bf16-rounded, the reference's `x.to(dtype)` in front of `w2`;
    /// - `F.silu`'s multiply form `g·sigmoid(g)`, not the division form, which `moe.hip` records
    ///   as one rounding apart that "would normally vanish under the bf16 store ... except
    ///   exactly at a rounding boundary". This one is **true by construction and unexercised**:
    ///   swapping the forms was measured bit-identical over all 512 outputs of the fp4 fixture
    ///   (2026-08-05), which is what that comment predicts when the boundary is not reached.
    ///   The decision below does not rest on it — the two bf16 roundings and the clamp are each
    ///   demonstrated by a break that goes red.
    ///
    /// GLM has no `swiglu_limit` and should never acquire one, so passing an "unclamped"
    /// sentinel from its four call sites would put a value in the tree that no config can
    /// produce. This refuses `limit <= 0`, **NaN and `+/-inf`** (guard 1006, the same code `moe.hip`
    /// returns for the same check on the same argument), so unclamped is not spellable here.
    ///
    /// The clamp itself is one `kernels/common.hpp::swiglu_clamped`, shared with
    /// `moe_gateup_f4_impl`: the routed and shared paths must agree bit for bit, and `kernels/`
    /// is not scanned by `build.rs`'s jscpd gate, so a second copy would have drifted unseen.
    ///
    /// # Safety
    /// `g`, `u` and `h` are each `n` f32 and must outlive `stream`'s completion. `h` may alias
    /// `g` or `u` — every thread reads both, then writes once, and that write depends on both
    /// reads. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_swiglu_clamped_bf16 -> rivoli_swiglu_clamped_bf16, "swiglu_clamped_bf16" (
        g: *const f32,
        u: *const f32,
        n: usize as i32,
        limit: f32,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// **SiTU-GLU over a gate/up pair — Kimi-K3's activation**, `k3-architecture.md` §8:
    /// `y = (b1·tanh(g/b1)·sigmoid(g)) · (b2·tanh(u/b2))`, so `|y| <= b1·b2` (100 at the shipped
    /// 4 and 25). The arithmetic is `kernels/common.hpp::situ_glu` and is argued there; ported
    /// from `k3:src/backend/hip.rs` (M9).
    ///
    /// **Not [`launch_swiglu_clamped_bf16`] with different constants.** The sigmoid takes the
    /// UNCAPPED gate, `up` is transformed rather than clamped, neither operand is bf16-rounded,
    /// and the store is f32. Merging the two behind a parameter is the refactor `common.hpp`
    /// warns against for `swiglu`/`swiglu_clamped` — the same hazard, one model over.
    ///
    /// **This is the DENSE path only** — layer 0's MLP and the shared MLP. The routed experts
    /// fuse the same helper inside `moe_f4.hip`'s fp4 expert kernel and never come through here,
    /// so a change made here alone leaves every routed expert wrong (plan §3b).
    ///
    /// Both betas must be finite and positive; `<= 0`, NaN and `+/-inf` are refused (1006, the
    /// same code the `swiglu_limit` guards return for the same shape of check). A zero or NaN
    /// beta does not degrade gracefully — it makes `tanh(x/b)` saturate or go NaN for every
    /// element, which no magnitude check on the output would catch.
    ///
    /// # Safety
    /// `g`, `u` and `h` are each `n` f32 and must outlive `stream`'s completion. `h` may alias
    /// `g` or `u` — every thread reads both, then writes once, and that write depends on both
    /// reads. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_situ_glu_f32 -> rivoli_situ_glu_f32, "situ_glu_f32" (
        g: *const f32,
        u: *const f32,
        n: usize as i32,
        b1: f32,
        b2: f32,
        h: *mut f32,
        stream: *mut c_void,
    );

    /// `x[i] *= sigmoid(g[i])` — Muse Glimmer's attention output gate, applied between the attend
    /// and `o_proj`.
    ///
    /// **`g` must be `gate_proj(LAYER INPUT)`, not anything derived from `x`.** The reference
    /// computes the gate from the post-`input_layernorm` activation (`glimmer-architecture.md`
    /// §4 item 3); gating on the attention output has the right shapes and the wrong model, and no
    /// signature can prevent it — `tests/glimmer_gate.rs` is what holds a caller to it.
    ///
    /// Not [`launch_swiglu`]: that is `silu(g)*u`, which carries an extra factor of `g`.
    ///
    /// USED BY Kimi-K3's gated MLA too (M9): §5's output gate is this same arithmetic at the
    /// same seam — between [`launch_mha_attend`](crate::hip_attn::launch_mha_attend) and
    /// `o_proj` — so the k3 tree's out-of-place copy was deliberately NOT ported
    /// (`kernels/fwd.hip` records the decision, and trap 10's twin is
    /// [`launch_rmsnorm_gate_heads_f32`](crate::hip_attn::launch_rmsnorm_gate_heads_f32),
    /// which norms first and must not be reached for here).
    ///
    /// A `g` of ±Inf gates by the LIMIT (exactly 1 or 0) — finite output from non-finite input,
    /// a reviewed decision recorded on the kernel.
    ///
    /// # Safety
    /// `x` is `n` writable f32 and `g` is `n` readable f32, both live until the next
    /// [`device_sync`], and they must not alias — both are `__restrict__`. `stream` is null or a
    /// live stream ordering every producer of `x` and `g` — this kernel runs BETWEEN the attend
    /// and the o_proj GEMV, so at any stream-ordered call site null is the unordered-read bug,
    /// not a default.
    launch_sigmoid_gate -> rivoli_sigmoid_gate, "sigmoid_gate" (
        x: *mut f32,
        g: *const f32,
        n: usize as i32,
        stream: *mut c_void,
    );

    /// `x[i] = cap * tanh(x[i] * mult / cap)` — Muse Glimmer's logit path, applied to the head's
    /// output. `mult` is `output_multiplier`, `cap` is `final_logit_softcapping`; the launcher
    /// refuses a transposed pair (`mult >= cap`, code 1002) because the swap is a silent
    /// sign-quantiser, and refuses non-finite or non-positive values of either (1001).
    ///
    /// **Every greedy gate in this repo is provably blind to omitting this.** `mult > 0` and `tanh`
    /// is strictly increasing, so it cannot move an argmax — the anchor measured `softcap_off`
    /// leaving `emitted.ids` bit-identical while the logits moved. It changes every probability,
    /// so its evidence must come from logits and never from what was decoded.
    ///
    /// **A non-finite logit passes through unchanged.** The naive form maps ±Inf to exactly ±cap,
    /// which would launder an overflowed head output into a finite argmax winner one launch
    /// before the engine's ONLY post-final-layer fault detector (`argmax`'s non-finite bail).
    ///
    /// # Safety
    /// `x` is `n` writable f32, live until the next [`device_sync`]; `stream` is null or a live
    /// stream ordering the head GEMV that produced `x`.
    launch_logit_softcap -> rivoli_logit_softcap, "logit_softcap" (
        x: *mut f32,
        n: usize as i32,
        mult: f32,
        cap: f32,
        stream: *mut c_void,
    );

    // ── activation quantizers ───────────────────────────────────────────────────────────────

    /// `kernel.py::act_quant(v, 128, "ue8m0", inplace=True)` over `n_rows x row_len` f32, in
    /// place — the fp8 activation quantization V4 performs in front of every quantized
    /// `Linear`, fp4-weight ones included.
    ///
    /// Fused quantize-then-dequantize: the buffer stays f32 and holds `e4m3(v/s)·s`. That is
    /// what the reference's `inplace=True` does and what the oracle models, so the values a
    /// following GEMV consumes are the reference's own.
    ///
    /// # Safety
    /// `v` is `n_rows · row_len` live f32 for `stream`'s duration.
    launch_act_quant_f8 -> rivoli_act_quant_f8, "act_quant_f8" (
        v: *mut f32,
        n_rows: usize as i32,
        row_len: usize as i32,
        stream: *mut c_void,
    );

    /// `kernel.py::act_quant(x, block, "ue8m0", inplace=True)` over `rows` rows of
    /// `row_stride` floats, quantizing the first `n` of each — reading `src`, writing
    /// `dst` at the same offsets. `src == dst` is the reference's in-place form;
    /// `n < row_stride` is then the KV entry's PARTIAL quantization (model.py:512, dims
    /// `[0, head_dim - rope_head_dim)` at block 64). `src != dst` is the M10 fused
    /// quantize-copy — one launch where the qkv chain ran `memcpy_dtod_async` + this,
    /// leaving `src` untouched for its other readers — and the launcher REFUSES it at
    /// `n != row_stride` (code 1002), because a partial-width quant-from-source would
    /// leave `dst`'s row tails stale where the copy it replaces filled them.
    /// `n == row_stride` at block 128 is what every quantized `Linear` does to its
    /// activation before the GEMM.
    ///
    /// # Safety
    /// `src` and `dst` are device buffers of at least `rows * row_stride` f32 (the same
    /// buffer, or non-overlapping ones), and must outlive `stream`'s completion. `stream`
    /// is a live `hipStream_t`, or null for the default stream.
    launch_act_quant_f8_prefix -> rivoli_act_quant_f8_prefix, "act_quant_f8_prefix" (
        src: *const f32,
        dst: *mut f32,
        rows: usize as i32,
        row_stride: usize as i32,
        n: usize as i32,
        block: usize as i32,
        stream: *mut c_void,
    );

    /// `rotate_activation` then `fp4_act_quant(·, 32, inplace=True)` over `rows` rows of `d`
    /// floats, in place — `Oracle::indexer_spread` (forward.rs:1130-1138) and the finish
    /// `Compressor.forward` performs when `rotate = true` (model.py:374-376).
    ///
    /// Applied to BOTH the indexer's `q` rows and its nested compressor's pooled rows, which is
    /// why it is one launcher rather than a step inside either. [`launch_act_quant_f8_prefix`] is the
    /// *other* compressor's finish and takes a partial extent; this one has none — the Hadamard
    /// covers the whole row, RoPE tail included. Handing either the other's extent is finite,
    /// plausible and wrong, so `kvcompress::Geom` carries which is due and
    /// `kvcompress::compress` matches on it.
    ///
    /// `d` must be a power of two no greater than 256 and a multiple of 32; the launcher
    /// refuses otherwise (guards 1002/1003/1004) rather than transforming a length the
    /// reference would have zero-padded, or quantizing a ragged tail against its own amax.
    ///
    /// # Safety
    /// `x` is `rows · d` writable, 4-byte-aligned, device-resident f32, read and written in
    /// place, and outlives `stream`'s completion. `stream` is a live `hipStream_t`, or null for
    /// the default stream.
    launch_act_quant_f4_rotated -> rivoli_act_quant_f4_rotated, "act_quant_f4_rotated" (
        x: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        stream: *mut c_void,
    );

    // ── normalization ───────────────────────────────────────────────────────────────────────

    /// RMSNorm `y = x·rsqrt(mean(x²)+eps)·w`.
    ///
    /// # Safety
    /// Device pointers (`x`, `w`, `y` each `n` f32) live until the next [`device_sync`].
    launch_rmsnorm_single -> rivoli_rmsnorm_single, "rmsnorm_single" (
        x: *const f32,
        w: *const f32,
        n: usize as i32,
        eps: f32,
        y: *mut f32,
    );

    /// `y[i] = x[i]·(1/sqrt(mean(x²)+eps))·(1 + w[i])` — Muse Glimmer's CENTERED RMSNorm, one row.
    ///
    /// **The `(1 + w)` is the whole difference from [`launch_rmsnorm_single`], and this model uses
    /// BOTH forms.** Glimmer's four per-layer sandwich norms are centered with `w` initialised to
    /// ZEROS; its final norm, weightless qk_norm and embedding norm are the plain form with `w`
    /// ones (`glimmer-architecture.md` §5). NEITHER substitution announces itself — the anchor's
    /// `norm_not_centered` run leaves zero non-finite values and decodes on, so §5's "crashes into
    /// garbage" is wrong and §9 trap 5's "runs clean" is right. Two entry points rather than a flag, the
    /// `launch_rope_split_half` argument: the wrong form is silent in one direction (a plain weight
    /// through here scales by ≈2 and stays fluent) and a bool would put it one argument from every
    /// GLM and V4 call site.
    ///
    /// `eps` is refused unless positive and finite (code 1002). Glimmer's four norms carry TWO eps
    /// three orders of magnitude apart, and passing one where the other belongs is the anchor
    /// defect that sets this operator's tolerance row.
    ///
    /// # Safety
    /// `x`, `w` and `y` are each `n` live f32 — `x`/`w` readable, `y` writable — until the next
    /// [`device_sync`]; `y` may alias `x` (each thread reads and writes only index `i` after the
    /// block-wide reduction has completed). `stream` is null or a live stream ordering every
    /// producer of `x` and `w`.
    launch_rmsnorm_centered_single -> rivoli_rmsnorm_centered_single, "rmsnorm_centered_single" (
        x: *const f32,
        w: *const f32,
        n: usize as i32,
        eps: f32,
        y: *mut f32,
        stream: *mut c_void,
    );

    /// `RMSNorm.forward` over `rows` rows of `d` floats, in place: f32 statistic, learned
    /// weight, bf16-rounded store.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`rows * d` f32), `w` (`d` f32).
    /// `stream` is a live `hipStream_t`, or null for the default stream.
    launch_rmsnorm_batch -> rivoli_rmsnorm_batch, "rmsnorm_batch" (
        x: *mut f32,
        w: *const f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        stream: *mut c_void,
    );

    /// Muse Glimmer's weightless QK-norm over `rows` heads of `d`, in place, times `scale`.
    ///
    /// Q passes `qk_scale_factor` (3.87) and K passes 1.0 — the scale is Q's alone
    /// (`glimmer-architecture.md` trap 3), and it is folded here rather than given a second pass
    /// over the tensor. **Not `rmsnorm_batch`** (which multiplies a learned weight and stores bf16)
    /// **and not `mla.hip::qk_norm`** (whose statistic is bf16 by DeepSeek-V4's reference); the
    /// kernel's own comment carries both arguments.
    ///
    /// # Safety
    /// `x` is a device buffer of `rows · d` live f32, written in place, live until the next
    /// [`device_sync`]. `stream` is null or a live stream ordering every producer of `x`.
    ///
    /// **`x` is DESTROYED — the pre-norm values do not survive this call**, so no consumer of them
    /// may be enqueued after this launch on any stream. The clause above constrains what may WRITE
    /// `x` before; this one constrains what may READ it after, and only the first was written down
    /// (review, 2026-08-13). The realistic violation is a `--trace` or `--pred-probe` readback of q
    /// expecting `q_proj`'s output and getting post-norm, post-3.87 bytes; no fixture can see it,
    /// which is the same shape as the null-stream finding this port already carries.
    launch_rmsnorm_weightless_batch -> rivoli_rmsnorm_weightless_batch, "rmsnorm_weightless_batch" (
        x: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        scale: f32,
        stream: *mut c_void,
    );

    /// LayerNorm with bias `y = (x-mean)/sqrt(var+eps)·w + b` (the indexer k_norm).
    ///
    /// # Safety
    /// Device pointers (`x`, `w`, `b`, `y` each `n` f32) live until the next [`device_sync`].
    launch_layernorm -> rivoli_layernorm, "layernorm" (
        x: *const f32,
        w: *const f32,
        b: *const f32,
        n: usize as i32,
        eps: f32,
        y: *mut f32,
    );

    /// The weightless per-head QK-norm of model.py:504, in place over `rows = s * n_heads`
    /// rows of `head_dim`. Must be launched BEFORE the RoPE — see the kernel's note: the
    /// oracle provably cannot see the order, so it comes from the reference.
    ///
    /// # Safety
    /// `q` is a device buffer of at least `rows * d` f32, and must outlive `stream`'s
    /// completion. `stream` is a live `hipStream_t`, or null for the default stream.
    launch_qk_norm -> rivoli_qk_norm, "qk_norm" (
        q: *mut f32,
        rows: usize as i32,
        d: usize as i32,
        eps: f32,
        stream: *mut c_void,
    );

    // ── RoPE — three pairing conventions, never interchangeable ─────────────────────────────

    /// Interleaved RoPE in place over `count` segments of `seg` at `stride`.
    ///
    /// # Safety
    /// `base` is a device buffer of `count·stride` f32, live until the next [`device_sync`].
    launch_rope_interleave -> rivoli_rope_interleave, "rope_interleave" (
        base: *mut f32,
        count: usize as i32,
        stride: usize as i32,
        seg: usize as i32,
        pos: usize as i32,
        theta: f64,
    );

    /// Split-half RoPE in place — transformers' `rotate_half`, where the pair is
    /// `(x[j], x[j+seg/2])` rather than two adjacent elements. Muse Glimmer's convention.
    ///
    /// **Same arithmetic as [`launch_rope_interleave`], different pairing, and the two are NOT
    /// interchangeable.** Applying one where the other is meant produces fluent wrong text and no
    /// error — `glimmer-architecture.md` §9 trap 9. They are separate entry points rather than one
    /// with a flag precisely so a GLM or V4 call site cannot reach this convention by changing an
    /// argument; `kernels/linalg.hip` carries the argument, and `swiglu`/`swiglu_clamped_bf16` is
    /// the precedent.
    ///
    /// # Safety
    /// `base` is a device buffer of `count·stride` f32, live until the next [`device_sync`].
    launch_rope_split_half -> rivoli_rope_split_half, "rope_split_half" (
        base: *mut f32,
        count: usize as i32,
        stride: usize as i32,
        seg: usize as i32,
        pos: usize as i32,
        theta: f64,
    );

    /// `apply_rotary_emb` over the last `rd` dims of each of `rows` rows, ADJACENT-PAIR
    /// (`view_as_complex`), from a precomputed `(cos, sin)` table. Row `r` takes position
    /// `pos0 + r / rows_per_pos`. `inverse` conjugates it — the output de-rotation.
    ///
    /// # Safety
    /// Device pointers must outlive `stream`'s completion: `x` (`rows * row_len` f32), `tbl`
    /// (at least `(pos0 + rows / rows_per_pos) * rd` f32, interleaved cos/sin). `stream` is a
    /// live `hipStream_t`, or null for the default stream.
    launch_rope_adjacent -> rivoli_rope_adjacent, "rope_adjacent" (
        x: *mut f32,
        tbl: *const f32,
        rows: usize as i32,
        row_len: usize as i32,
        rd: usize as i32,
        pos0: usize as i32,
        rows_per_pos: usize as i32,
        inverse: bool as i32,
        stream: *mut c_void,
    );
}
// jscpd:ignore-end
