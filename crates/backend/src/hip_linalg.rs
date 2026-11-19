//! The PRIMITIVE half of the HIP ABI wall: one launch per matvec, matmul, embedding row or
//! elementwise op — each one an operator the model graph names, taking buffers and extents
//! and nothing model-shaped. The activation family (gates, quantizers, norms, RoPE) is
//! `hip_activation.rs` since 2026-11-18, split by kernel family to match `src/launch/`.
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
use crate::launch::rivoli_gemm_bf16;
use anyhow::Result;
use rivoli_abi::kernarg::GemmBf16;
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
// > **SPLIT AGAIN 2026-11-18, by kernel family.** The activation family (gates, quantizers,
// > norms, RoPE) moved to `hip_activation.rs`, and with it most of the EXAMPLES below — the
// > `swiglu`/`rmsnorm_single` pairs, the `act_quant_f8`/`act_quant_f4_rotated` collision, the
// > `logit_softcap`/`sigmoid_gate` doc incident. The argument is unchanged and is the
// > argument for every invocation file; read its examples as being about the wall, wherever
// > they now sit. RE-MEASURED that day with this file's and `hip_activation.rs`'s markers
// > deleted (`npx jscpd -c .jscpd.json --exitCode 7 crates` over a scratch copy): **11
// > clones** — 5 inside this file, 5 inside `hip_activation.rs`, 1 across the two — every
// > one a parameter-list prefix run between two declarations, none of them logic. So both
// > markers are load-bearing; the counts below are the single file's, kept as history.
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
    launch_gemm_bf16 => rivoli_gemm_bf16(GemmBf16 { x, w, out, m: m as i32, n: n as i32, k: k as i32 }, stream), "gemm_bf16" (
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

}
// jscpd:ignore-end
