//! Minimal HIP surface: under the `rocm` feature this binds the hipcc-built
//! kernel launchers. Without the feature the whole module compiles away — a
//! CPU-only dev build takes the scalar `engine::Engine` path in main.rs instead,
//! so there is no GPU surface to stub out.

#[cfg(feature = "rocm")]
use anyhow::{Result, bail};

/// One MoE expert's six weight tensors, as raw DEVICE pointers into the resident
/// tier (or a cold-fetch slab). `repr(C)` matching `struct ExpertDesc` in
/// moe_fused.hip — the kernel reads a `[E]` array of these so experts may live
/// at arbitrary device offsets (no contiguous-stride assumption; PLAN.md D1).
#[cfg(feature = "rocm")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDesc {
    pub gate_packed: *const u8,
    pub gate_scale: *const f32,
    pub up_packed: *const u8,
    pub up_scale: *const f32,
    pub down_packed: *const u8,
    pub down_scale: *const f32,
}

/// VQ-int3 expert descriptor (mirrors `ExpertDescVq` in `moe_fused.hip`): per
/// projection the packed 12-bit indices + bf16 group scales. The codebook is
/// shared across all experts and passed to [`launch_moe_vq`] separately.
#[cfg(feature = "rocm")]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDescVq {
    pub gate_indices: *const u8,
    pub gate_scales: *const u16,
    pub up_indices: *const u8,
    pub up_scales: *const u16,
    pub down_indices: *const u8,
    pub down_scales: *const u16,
}

#[cfg(feature = "rocm")]
unsafe extern "C" {
    fn rivoli_moe_experts(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e: i32,
        descs: *const ExpertDesc,
        wexpert: *const f32,
        h: *mut f32,
        partial: *mut f32,
        out: *mut f32,
    ) -> i32;

    fn rivoli_moe_experts_vq(
        x: *const f32,
        hidden: i32,
        inter: i32,
        e: i32,
        descs: *const ExpertDescVq,
        codebook: *const f32,
        wexpert: *const f32,
        h: *mut f32,
        partial: *mut f32,
        out: *mut f32,
    ) -> i32;

    fn rivoli_device_sync() -> i32;

    fn rivoli_embed_i8_row(
        packed: *const u8,
        scale: *const f32,
        token: i32,
        hidden: i32,
        x: *mut f32,
    ) -> i32;

    fn rivoli_append_kv(
        latent: *const f32,
        rope: *const f32,
        lc: *mut u16,
        rc: *mut u16,
        pos: i32,
        kvl: i32,
        ropn: i32,
    ) -> i32;

    fn rivoli_gather_rope(
        q: *const f32,
        qrope: *mut f32,
        h: i32,
        qh: i32,
        nope: i32,
        ropn: i32,
    ) -> i32;

    fn rivoli_vadd(x: *mut f32, y: *const f32, n: i32) -> i32;

    fn rivoli_argmax(logits: *const f32, n: i32, out_idx: *mut i32, out_val: *mut f32) -> i32;

    fn rivoli_gemv_i4(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;

    fn rivoli_gemv_vq(
        x: *const f32,
        indices: *const u8,
        scales: *const u16,
        codebook: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;

    fn rivoli_vq_encode(
        sub: *const f32,
        codebook: *const f32,
        cbnorm: *const f32,
        n: i32,
        idx: *mut u16,
    ) -> i32;

    fn rivoli_gemv_i8(
        x: *const f32,
        packed: *const u8,
        scale: *const f32,
        o_dim: i32,
        i_dim: i32,
        y: *mut f32,
    ) -> i32;

    fn rivoli_gemv_f32(x: *const f32, w: *const f32, o_dim: i32, i_dim: i32, y: *mut f32) -> i32;

    fn rivoli_rmsnorm(x: *const f32, w: *const f32, n: i32, eps: f32, y: *mut f32) -> i32;

    fn rivoli_rope(base: *mut f32, count: i32, stride: i32, seg: i32, pos: i32, theta: f64) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_absorb(
        q: *const f32,
        kvb_packed: *const u8,
        kvb_scale: *const f32,
        h: i32,
        qh: i32,
        nope: i32,
        vh: i32,
        kvl: i32,
        qabs: *mut f32,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_value(
        clat: *const f32,
        kvb_packed: *const u8,
        kvb_scale: *const f32,
        h: i32,
        nope: i32,
        vh: i32,
        kvl: i32,
        ctx: *mut f32,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_attend(
        qabs: *const f32,
        qrope: *const f32,
        lc: *const u16,
        rc: *const u16,
        rows: *const u32,
        h: i32,
        nr: i32,
        kvl: i32,
        rope: i32,
        scale: f32,
        clat: *mut f32,
        partial: *mut f32,
    ) -> i32;

    /// f32 count the split-KV partial scratch needs for these dims (worst-case
    /// split count). Pure arithmetic — no HIP calls, safe to call before init.
    fn rivoli_mla_attend_scratch_floats(h: i32, kvl: i32) -> usize;

    fn rivoli_gemv_bf16(x: *const f32, w: *const u16, o_dim: i32, i_dim: i32, y: *mut f32) -> i32;
    fn rivoli_layernorm(
        x: *const f32,
        w: *const f32,
        b: *const f32,
        n: i32,
        eps: f32,
        y: *mut f32,
    ) -> i32;
    fn rivoli_index_append(k: *const f32, kcache: *mut u16, pos: i32, hd: i32) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn rivoli_index_score(
        q: *const f32,
        w: *const f32,
        kcache: *const u16,
        heads: *const u32,
        nt: i32,
        nh: i32,
        nact: i32,
        hd: i32,
        wscale: f32,
        dscale: f32,
        scores: *mut f32,
    ) -> i32;
    fn rivoli_index_pool_push(k: *const f32, pool: *mut f32, t: i32, hd: i32) -> i32;
    fn rivoli_index_head_route(
        q: *const f32,
        w: *const f32,
        pool: *const f32,
        m_blocks: i32,
        nh: i32,
        hd: i32,
        e: *mut f32,
    ) -> i32;

    fn rivoli_append_kv_fp8(
        latent: *const f32,
        rope: *const f32,
        lc8: *mut u8,
        lscale: *mut f32,
        rc: *mut u16,
        pos: i32,
        kvl: i32,
        ropn: i32,
        n_blocks: i32,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    fn rivoli_mla_attend_fp8(
        qabs: *const f32,
        qrope: *const f32,
        lc8: *const u8,
        lscale: *const f32,
        rc: *const u16,
        rows: *const u32,
        h: i32,
        nr: i32,
        kvl: i32,
        rope: i32,
        n_blocks: i32,
        scale: f32,
        clat: *mut f32,
        partial: *mut f32,
    ) -> i32;
}

/// Number of `f32`s the caller must own for [`launch_attend`]'s split-KV partial
/// scratch at these dims. Sized for the kernel's maximum split count, so ONE
/// allocation per session covers every context length — mid-session `hipMalloc`
/// is the amdgpu GTT wedge trigger (PLAN.md M3 wiring contract).
#[cfg(feature = "rocm")]
#[must_use]
pub fn attend_scratch_floats(h: usize, kvl: usize) -> usize {
    if h > i32::MAX as usize || kvl > i32::MAX as usize {
        return 0; // the launchers reject these dims anyway
    }
    // SAFETY: pure integer arithmetic in the kernel TU; touches no device state.
    unsafe { rivoli_mla_attend_scratch_floats(h as i32, kvl as i32) }
}

/// Shared launcher return-code check. The kernels' convention (see moe_fused.hip):
/// 0 = success, POSITIVE = the launcher's own arg guard, NEGATIVE = -(hipError_t).
#[cfg(feature = "rocm")]
fn check(r: i32, name: &str) -> Result<()> {
    if r != 0 {
        let kind = if r > 0 { "arg guard" } else { "HIP runtime" };
        bail!("{name} failed ({kind}, code {r})");
    }
    Ok(())
}

/// Launch MLA flash attention over the resident KV cache: for each head, writes
/// the attention-weighted latent `clat_h = Σ_t softmax((qabs_h·L_t + qrope_h·R_t)
/// ·scale)·L_t` as `H` contiguous `kvl`-length rows. All arguments are DEVICE
/// pointers — `qabs`/`qrope` (`h*kvl`/`h*rope` f32), the bf16 cache `lc`/`rc`,
/// and `clat` (`h*kvl` f32, fully written). `rows` selects which cache rows are
/// attended (sparse modes): null = dense over rows `0..nr`; non-null = gather
/// of the `nr` listed rows (ascending). Does NOT synchronize — call
/// [`device_sync`] once per token.
///
/// `partial` is the caller-owned split-KV scratch: when it is non-null the rows
/// are partitioned across extra workgroups (filling the GPU instead of the
/// 8-block head grid) and a second kernel combines the partials into `clat`. A
/// null `partial` forces the single-split path — correct, just slower.
///
/// # Safety
/// The launch is ASYNCHRONOUS: the kernel reads/writes the pointers below AFTER
/// this call returns, so all must stay valid until the next [`device_sync`]
/// RETURNS. Shapes (all device pointers in the current HIP context): `qabs`
/// `h*kvl` f32, `qrope` `h*rope` f32, `lc`/`rc` at least `(max row)+1` cache
/// rows, `rows` `nr` u32 (when non-null) with EVERY entry a valid cache row —
/// the kernel cannot bounds-check the slab — `clat` `h*kvl` f32, and `partial`
/// either null or at least [`attend_scratch_floats(h, kvl)`](attend_scratch_floats)
/// f32 (the kernel writes it without bounds-checking).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_attend(
    qabs: *const f32,
    qrope: *const f32,
    lc: *const u16,
    rc: *const u16,
    rows: *const u32,
    h: usize,
    nr: usize,
    kvl: usize,
    rope: usize,
    scale: f32,
    clat: *mut f32,
    partial: *mut f32,
) -> Result<()> {
    debug_assert!(
        h <= i32::MAX as usize
            && nr <= i32::MAX as usize
            && kvl <= i32::MAX as usize
            && rope <= i32::MAX as usize,
        "attn dims exceed i32 (h={h} nr={nr} kvl={kvl} rope={rope})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_mla_attend(
            qabs,
            qrope,
            lc,
            rc,
            rows,
            h as i32,
            nr as i32,
            kvl as i32,
            rope as i32,
            scale,
            clat,
            partial,
        )
    };
    check(r, "launch_attend")
}

/// fp8 variant of [`launch_attend`]: the latent cache is fp8-e4m3 (`lc8`,
/// `nr*kvl` u8) + per-128 block scales (`lscale`, `nr*n_blocks` f32) instead of
/// bf16; the rope cache `rc` stays bf16. Launch-only.
///
/// # Safety
/// Same async contract as [`launch_attend`]; `lc8`/`lscale` (and `rc`/`rows`/
/// `qabs`/`qrope`/`clat`/`partial`) must stay valid device pointers until the
/// next [`device_sync`] returns, every `rows` entry a valid cache row, and
/// `partial` null or ≥ [`attend_scratch_floats(h, kvl)`](attend_scratch_floats)
/// f32.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_attend_fp8(
    qabs: *const f32,
    qrope: *const f32,
    lc8: *const u8,
    lscale: *const f32,
    rc: *const u16,
    rows: *const u32,
    h: usize,
    nr: usize,
    kvl: usize,
    rope: usize,
    n_blocks: usize,
    scale: f32,
    clat: *mut f32,
    partial: *mut f32,
) -> Result<()> {
    debug_assert!(
        h <= i32::MAX as usize
            && nr <= i32::MAX as usize
            && kvl <= i32::MAX as usize
            && rope <= i32::MAX as usize
            && n_blocks <= i32::MAX as usize,
        "attn_fp8 dims exceed i32 (h={h} nr={nr} kvl={kvl} rope={rope} n_blocks={n_blocks})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_mla_attend_fp8(
            qabs,
            qrope,
            lc8,
            lscale,
            rc,
            rows,
            h as i32,
            nr as i32,
            kvl as i32,
            rope as i32,
            n_blocks as i32,
            scale,
            clat,
            partial,
        )
    };
    check(r, "launch_attend_fp8")
}

/// fp8 variant of [`launch_append_kv`]: quantizes `latent` to fp8-e4m3 into
/// `lc8` (row `pos`) with per-128 block scales into `lscale`, and appends the
/// bf16 rope key to `rc`. Launch-only.
///
/// # Safety
/// Async — `latent`/`rope` inputs and `lc8`/`lscale`/`rc` slabs (with room for
/// row `pos`) must stay valid device pointers until the next [`device_sync`]
/// returns.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_append_kv_fp8(
    latent: *const f32,
    rope: *const f32,
    lc8: *mut u8,
    lscale: *mut f32,
    rc: *mut u16,
    pos: usize,
    kvl: usize,
    ropn: usize,
    n_blocks: usize,
) -> Result<()> {
    debug_assert!(
        pos <= i32::MAX as usize
            && kvl <= i32::MAX as usize
            && ropn <= i32::MAX as usize
            && n_blocks <= i32::MAX as usize,
        "append_kv_fp8 dims exceed i32"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_append_kv_fp8(
            latent,
            rope,
            lc8,
            lscale,
            rc,
            pos as i32,
            kvl as i32,
            ropn as i32,
            n_blocks as i32,
        )
    };
    check(r, "launch_append_kv_fp8")
}

/// Launch one fused MoE expert-batch over resident device memory: every expert
/// `e` shares `x` and the `(hidden, inter)` shape, accumulating
/// `out += Σ_e w[e] · down_e(silu(gate_e·x) ⊙ up_e·x)`. All arguments are DEVICE
/// pointers — `x`, `out`, the `[e]` weight array `wexpert`, the `[e]` descriptor
/// array `descs`, and the per-expert weights the descriptors point at. `out`
/// must be pre-zeroed (the kernel accumulates). Does NOT synchronize — call
/// [`device_sync`] once per token (≤1 join/token).
///
/// # Safety
/// The launch is ASYNCHRONOUS: the kernel dereferences every pointer below
/// AFTER this call returns, so all of them — and the device buffers the
/// `ExpertDesc` weight pointers address — must stay valid until the next
/// [`device_sync`] RETURNS. Dropping any of them before that join is a GPU
/// use-after-free. Shapes the kernel assumes (all device pointers in the current
/// HIP context):
///   - `x`: `hidden` f32; `wexpert`: `e` f32
///   - `partial`: `e * hidden` f32 scratch (each expert writes its own row; the
///     fixed-order reduce sums them → `out`, so both are fully written — no
///     pre-zero needed)
///   - `out`: `hidden` f32
///   - `descs`: ≥ `e` contiguous `ExpertDesc`, each pointing at
///     - `gate_packed`, `up_packed`: `inter * ((hidden+1)/2)` bytes
///     - `gate_scale`, `up_scale`: `inter` f32
///     - `down_packed`: `hidden * ((inter+1)/2)` bytes
///     - `down_scale`: `hidden` f32
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e: usize,
    descs: *const ExpertDesc,
    wexpert: *const f32,
    h: *mut f32,
    partial: *mut f32,
    out: *mut f32,
) -> Result<()> {
    debug_assert!(
        hidden <= i32::MAX as usize && inter <= i32::MAX as usize && e <= i32::MAX as usize,
        "moe dims exceed i32 (hidden={hidden} inter={inter} e={e})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity.
    let r = unsafe {
        rivoli_moe_experts(
            x,
            hidden as i32,
            inter as i32,
            e as i32,
            descs,
            wexpert,
            h,
            partial,
            out,
        )
    };
    check(r, "launch_moe")
}

/// Batched VQ encode: `idx[i] = argmin_k (‖c_k‖² − 2·sub_i·c_k)` over the codebook,
/// for offline conversion. Bit-identical to `quant::quant_vq`'s nearest, so
/// GPU-encoded output validates against the CPU oracle. `cbnorm` = per-entry `‖c‖²`
/// precomputed host-side the same way.
///
/// # Safety
/// Async — `sub` (`n·VQ_DIM` f32), `codebook` (`VQ_K·VQ_DIM` f32), `cbnorm` (`VQ_K`
/// f32), `idx` (`n` u16) must be valid device pointers live until the next
/// [`device_sync`].
#[cfg(feature = "rocm")]
pub unsafe fn launch_vq_encode(
    sub: *const f32,
    codebook: *const f32,
    cbnorm: *const f32,
    n: usize,
    idx: *mut u16,
) -> Result<()> {
    debug_assert!(n <= i32::MAX as usize, "vq_encode n exceeds i32 ({n})");
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_vq_encode(sub, codebook, cbnorm, n as i32, idx) };
    check(r, "launch_vq_encode")
}

/// Launch a fused **VQ-int3** MoE expert batch — the VQ analog of [`launch_moe`].
/// Group scales are applied inside the decode, so there is no per-row post-scale;
/// `codebook` is the single shared `VQ_K·VQ_DIM` f32 buffer.
///
/// # Safety
/// Async, device pointers live until the next [`device_sync`]. Per expert
/// descriptor: `gate_indices`/`up_indices` = `inter·vq_row_bytes(hidden)` bytes,
/// `gate_scales`/`up_scales` = `inter·vq_groups(hidden)` u16; `down_indices` =
/// `hidden·vq_row_bytes(inter)` bytes, `down_scales` = `hidden·vq_groups(inter)`
/// u16. `x`/`wexpert`/`h`/`partial`/`out` sized as in [`launch_moe`].
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_moe_vq(
    x: *const f32,
    hidden: usize,
    inter: usize,
    e: usize,
    descs: *const ExpertDescVq,
    codebook: *const f32,
    wexpert: *const f32,
    h: *mut f32,
    partial: *mut f32,
    out: *mut f32,
) -> Result<()> {
    debug_assert!(
        hidden <= i32::MAX as usize && inter <= i32::MAX as usize && e <= i32::MAX as usize,
        "moe_vq dims exceed i32 (hidden={hidden} inter={inter} e={e})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity.
    let r = unsafe {
        rivoli_moe_experts_vq(
            x,
            hidden as i32,
            inter as i32,
            e as i32,
            descs,
            codebook,
            wexpert,
            h,
            partial,
            out,
        )
    };
    check(r, "launch_moe_vq")
}

/// Launch a batch-1 int4 GEMV `y = scale ⊙ (W·x)` over resident device memory
/// (`W` = `packed` per-row nibbles + per-row `scale`, `o_dim × i_dim`). Device
/// pointers, launch-only. The workhorse for the attention projections
/// (q_a/q_b/kv_a/o_proj) that must stay on-device to avoid per-layer joins.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `packed` (`o_dim·⌈i_dim/2⌉` bytes), `scale`
/// (`o_dim` f32), `y` (`o_dim` f32) must be valid device pointers that stay live
/// until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_i4(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_i4 dims exceed i32 (o={o_dim} i={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_i4(x, packed, scale, o_dim as i32, i_dim as i32, y) };
    check(r, "launch_gemv_i4")
}

/// Launch a batch-1 VQ-int3 GEMV `y = W·x` where `W` is `indices` (packed 12-bit
/// codebook indices) + `scales` (bf16 per-group) decoded against the shared
/// `codebook`, `o_dim × i_dim`. Group scales are applied inside the dot, so unlike
/// int4 there is no per-row post-scale. Device pointers, launch-only.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `indices` (`o_dim·vq_row_bytes(i_dim)` bytes),
/// `scales` (`o_dim·vq_groups(i_dim)` u16), `codebook` (`VQ_K·VQ_DIM` f32), `y`
/// (`o_dim` f32) must be valid device pointers live until the next [`device_sync`].
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_vq(
    x: *const f32,
    indices: *const u8,
    scales: *const u16,
    codebook: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_vq dims exceed i32 (o={o_dim} i={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_vq(x, indices, scales, codebook, o_dim as i32, i_dim as i32, y) };
    check(r, "launch_gemv_vq")
}

/// Launch a batch-1 int8 GEMV `y = scale ⊙ (W·x)` (`W` = `packed` signed bytes +
/// per-row `scale`, `o_dim × i_dim`) — the lm_head projection. Device pointers,
/// launch-only.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `packed` (`o_dim·i_dim` bytes), `scale` (`o_dim`
/// f32), `y` (`o_dim` f32) must be valid device pointers live until the next
/// [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_i8(
    x: *const f32,
    packed: *const u8,
    scale: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_i8 dims exceed i32 (o={o_dim} i={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_i8(x, packed, scale, o_dim as i32, i_dim as i32, y) };
    check(r, "launch_gemv_i8")
}

/// Launch a batch-1 f32 GEMV `y = W·x` (`W` = `w`, `o_dim × i_dim`, no scale) —
/// the MoE router gate. Device pointers, launch-only.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `w` (`o_dim·i_dim` f32), `y` (`o_dim` f32) must be
/// valid device pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_f32(
    x: *const f32,
    w: *const f32,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_f32 dims exceed i32 (o={o_dim} i={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_f32(x, w, o_dim as i32, i_dim as i32, y) };
    check(r, "launch_gemv_f32")
}

/// Launch RMSNorm `y = x·(1/√(mean(x²)+eps))·w` over a device vector of `n` f32.
///
/// # Safety
/// Async — `x`/`w`/`y` (each `n` f32 device pointers) must stay valid until the
/// next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_rmsnorm(
    x: *const f32,
    w: *const f32,
    n: usize,
    eps: f32,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(n <= i32::MAX as usize, "rmsnorm n exceeds i32 ({n})");
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_rmsnorm(x, w, n as i32, eps, y) };
    check(r, "launch_rmsnorm")
}

/// Launch interleaved RoPE at position `pos` on `count` device segments of `seg`
/// f32 each, segment `s` at `base[s*stride ..]` (matching attn.rs). Used for the
/// KV key (count=1) and the per-head query rope segments (count=H, stride=qh).
///
/// # Safety
/// Async — `base` must cover `(count-1)*stride + seg` f32 (the kernel's max
/// access) and stay valid until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_rope(
    base: *mut f32,
    count: usize,
    stride: usize,
    seg: usize,
    pos: usize,
    theta: f64,
) -> Result<()> {
    debug_assert!(
        count <= i32::MAX as usize
            && stride <= i32::MAX as usize
            && seg <= i32::MAX as usize
            && pos <= i32::MAX as usize,
        "rope dims exceed i32 (count={count} stride={stride} seg={seg} pos={pos})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_rope(
            base,
            count as i32,
            stride as i32,
            seg as i32,
            pos as i32,
            theta,
        )
    };
    check(r, "launch_rope")
}

/// Launch the MLA absorb: `qabs[head][i] = Σ_d q[head·qh+d]·kv_b[rbase+d][i]·
/// scale[rbase+d]` (rbase = head·(nope+vh)), folding each head's q_nope through
/// kv_b's `nope` rows into `H·kvl` latent-space queries. Device pointers,
/// launch-only.
///
/// # Safety
/// Async — `q` (`H·qh` f32), `kvb_packed`/`kvb_scale` (kv_b int4: `H·(nope+vh)`
/// rows × `⌈kvl/2⌉` bytes / `H·(nope+vh)` f32), `qabs` (`H·kvl` f32) must be
/// valid device pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_mla_absorb(
    q: *const f32,
    kvb_packed: *const u8,
    kvb_scale: *const f32,
    h: usize,
    qh: usize,
    nope: usize,
    vh: usize,
    kvl: usize,
    qabs: *mut f32,
) -> Result<()> {
    debug_assert!(
        h <= i32::MAX as usize
            && qh <= i32::MAX as usize
            && nope <= i32::MAX as usize
            && vh <= i32::MAX as usize
            && kvl <= i32::MAX as usize,
        "mla_absorb dims exceed i32"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_mla_absorb(
            q,
            kvb_packed,
            kvb_scale,
            h as i32,
            qh as i32,
            nope as i32,
            vh as i32,
            kvl as i32,
            qabs,
        )
    };
    check(r, "launch_mla_absorb")
}

/// Launch the MLA value projection: `ctx[head][j] = scale[rbase+nope+j]·Σ_i
/// clat[head][i]·kv_b[rbase+nope+j][i]`, projecting each head's attention-
/// weighted latent through kv_b's `vh` value rows to `H·vh`. Device pointers,
/// launch-only.
///
/// # Safety
/// Async — `clat` (`H·kvl` f32), `kvb_packed`/`kvb_scale` (as in
/// [`launch_mla_absorb`]), `ctx` (`H·vh` f32) must be valid device pointers live
/// until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_mla_value(
    clat: *const f32,
    kvb_packed: *const u8,
    kvb_scale: *const f32,
    h: usize,
    nope: usize,
    vh: usize,
    kvl: usize,
    ctx: *mut f32,
) -> Result<()> {
    debug_assert!(
        h <= i32::MAX as usize
            && nope <= i32::MAX as usize
            && vh <= i32::MAX as usize
            && kvl <= i32::MAX as usize,
        "mla_value dims exceed i32"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_mla_value(
            clat,
            kvb_packed,
            kvb_scale,
            h as i32,
            nope as i32,
            vh as i32,
            kvl as i32,
            ctx,
        )
    };
    check(r, "launch_mla_value")
}

/// Launch the embedding row lookup: `x[i] = (int8)embed[token·hidden+i]·
/// scale[token]`. Device pointers, launch-only.
///
/// # Safety
/// Async — `packed`/`scale` (the int8 embed table + row scales), `x` (`hidden`
/// f32) valid device pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_embed_i8_row(
    packed: *const u8,
    scale: *const f32,
    token: usize,
    hidden: usize,
    x: *mut f32,
) -> Result<()> {
    debug_assert!(
        token <= i32::MAX as usize && hidden <= i32::MAX as usize,
        "embed dims exceed i32"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_embed_i8_row(packed, scale, token as i32, hidden as i32, x) };
    check(r, "launch_embed_i8_row")
}

/// Launch the KV append: bf16-quantize `latent` (`kvl` f32) and `rope` (`ropn`
/// f32) into the per-layer bf16 slabs `lc`/`rc` at row `pos`. Launch-only.
///
/// # Safety
/// Async — `latent`/`rope` inputs and `lc`/`rc` slabs (with room for row `pos`)
/// must be valid device pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_append_kv(
    latent: *const f32,
    rope: *const f32,
    lc: *mut u16,
    rc: *mut u16,
    pos: usize,
    kvl: usize,
    ropn: usize,
) -> Result<()> {
    debug_assert!(
        pos <= i32::MAX as usize && kvl <= i32::MAX as usize && ropn <= i32::MAX as usize,
        "append_kv dims exceed i32"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_append_kv(latent, rope, lc, rc, pos as i32, kvl as i32, ropn as i32) };
    check(r, "launch_append_kv")
}

/// Launch the roped-query gather: `qrope[head·ropn+d] = q[head·qh+nope+d]`,
/// collecting each head's rope segment into contiguous `qrope[H·ropn]`.
/// Launch-only.
///
/// # Safety
/// Async — `q` (`h·qh` f32) and `qrope` (`h·ropn` f32) valid device pointers live
/// until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gather_rope(
    q: *const f32,
    qrope: *mut f32,
    h: usize,
    qh: usize,
    nope: usize,
    ropn: usize,
) -> Result<()> {
    debug_assert!(
        h <= i32::MAX as usize
            && qh <= i32::MAX as usize
            && nope <= i32::MAX as usize
            && ropn <= i32::MAX as usize,
        "gather_rope dims exceed i32"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gather_rope(q, qrope, h as i32, qh as i32, nope as i32, ropn as i32) };
    check(r, "launch_gather_rope")
}

/// Launch the residual add `x[i] += y[i]` over `n` device f32. Launch-only.
///
/// # Safety
/// Async — `x`/`y` (`n` f32) valid device pointers live until the next
/// [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_vadd(x: *mut f32, y: *const f32, n: usize) -> Result<()> {
    debug_assert!(n <= i32::MAX as usize, "vadd n exceeds i32 ({n})");
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_vadd(x, y, n as i32) };
    check(r, "launch_vadd")
}

/// Launch the greedy argmax reduction over `logits[0..n]` (one block, LDS tree
/// reduce), writing the winning index to `out_idx` and its value to `out_val`
/// (the caller's small device result buffer). Matches the host fold's tie-break
/// (first/lowest index) and NaN handling exactly. Device pointers, launch-only.
///
/// # Safety
/// Async — `logits` (`n` f32), `out_idx` (one i32), `out_val` (one f32) must be
/// valid device pointers live until the next join (here: the caller's D2H of the
/// result, which serializes after this launch on the null stream).
#[cfg(feature = "rocm")]
pub unsafe fn launch_argmax(
    logits: *const f32,
    n: usize,
    out_idx: *mut i32,
    out_val: *mut f32,
) -> Result<()> {
    debug_assert!(n <= i32::MAX as usize, "argmax n exceeds i32 ({n})");
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_argmax(logits, n as i32, out_idx, out_val) };
    check(r, "launch_argmax")
}

/// Launch a bf16 GEMV `y[o] = Σ_i x[i]·bf16(w[o·i_dim+i])` — the DSA indexer's
/// `wk`/`wq_b`/`weights_proj`. `x` f32, `w` bf16, `y` f32. Launch-only.
///
/// # Safety
/// Async — `x` (`i_dim` f32), `w` (`o_dim·i_dim` u16), `y` (`o_dim` f32) valid
/// device pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_gemv_bf16(
    x: *const f32,
    w: *const u16,
    o_dim: usize,
    i_dim: usize,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(
        o_dim <= i32::MAX as usize && i_dim <= i32::MAX as usize,
        "gemv_bf16 dims exceed i32 (o_dim={o_dim} i_dim={i_dim})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_gemv_bf16(x, w, o_dim as i32, i_dim as i32, y) };
    check(r, "launch_gemv_bf16")
}

/// Launch LayerNorm `y = (x-mean)/sqrt(var+eps)·w + b` over `n` device f32 — the
/// indexer `k_norm` (the one norm in the model with a bias). Launch-only.
///
/// # Safety
/// Async — `x`/`w`/`b`/`y` (each `n` f32) valid device pointers live until the
/// next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_layernorm(
    x: *const f32,
    w: *const f32,
    b: *const f32,
    n: usize,
    eps: f32,
    y: *mut f32,
) -> Result<()> {
    debug_assert!(n <= i32::MAX as usize, "layernorm n exceeds i32 ({n})");
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_layernorm(x, w, b, n as i32, eps, y) };
    check(r, "launch_layernorm")
}

/// Launch the indexer key append: bf16-quantize `k` (`hd` f32) into the
/// per-full-layer key slab `kcache` at row `pos`. Launch-only.
///
/// # Safety
/// Async — `k` (`hd` f32) and `kcache` (room for row `pos`) valid device
/// pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_index_append(
    k: *const f32,
    kcache: *mut u16,
    pos: usize,
    hd: usize,
) -> Result<()> {
    debug_assert!(
        pos <= i32::MAX as usize && hd <= i32::MAX as usize,
        "index_append dims exceed i32 (pos={pos} hd={hd})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_index_append(k, kcache, pos as i32, hd as i32) };
    check(r, "launch_index_append")
}

/// Launch the indexer scoring: `scores[t] = Σ_{h∈active} w[h]·wscale·
/// ReLU((q_h·k_t)·dscale)` over `nt` cached tokens. `heads` (null = all `nh`)
/// lists the `nact` active heads (MISA). Launch-only.
///
/// # Safety
/// Async — `q` (`nh·hd` f32), `w` (`nh` f32), `kcache` (`nt·hd` u16), `heads`
/// (`nact` u32 or null), `scores` (`nt` f32) valid device pointers live until
/// the next join (the caller's D2H of `scores`).
#[cfg(feature = "rocm")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_score(
    q: *const f32,
    w: *const f32,
    kcache: *const u16,
    heads: *const u32,
    nt: usize,
    nh: usize,
    nact: usize,
    hd: usize,
    wscale: f32,
    dscale: f32,
    scores: *mut f32,
) -> Result<()> {
    debug_assert!(
        nt <= i32::MAX as usize
            && nh <= i32::MAX as usize
            && nact <= i32::MAX as usize
            && hd <= i32::MAX as usize,
        "index_score dims exceed i32 (nt={nt} nh={nh} nact={nact} hd={hd})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe {
        rivoli_index_score(
            q,
            w,
            kcache,
            heads,
            nt as i32,
            nh as i32,
            nact as i32,
            hd as i32,
            wscale,
            dscale,
            scores,
        )
    };
    check(r, "launch_index_score")
}

/// Launch the MISA block-pool update: fold indexer key `k` (`hd` f32) into its
/// running-mean block at token `t` (block `t / MISA_BLOCK`). Runs every token so
/// the pool is ready when context crosses `index_topk`. Launch-only.
///
/// # Safety
/// Async — `k` (`hd` f32) and `pool` (room for row `t / MISA_BLOCK`, `hd` f32)
/// valid device pointers live until the next [`device_sync`] returns.
#[cfg(feature = "rocm")]
pub unsafe fn launch_index_pool_push(
    k: *const f32,
    pool: *mut f32,
    t: usize,
    hd: usize,
) -> Result<()> {
    debug_assert!(
        t <= i32::MAX as usize && hd <= i32::MAX as usize,
        "index_pool_push dims exceed i32 (t={t} hd={hd})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r = unsafe { rivoli_index_pool_push(k, pool, t as i32, hd as i32) };
    check(r, "launch_index_pool_push")
}

/// Launch the MISA head router: `e[j] = mean_b |w[j]·ReLU(q_j·pool_b)|` over the
/// `m_blocks` pooled blocks and `nh` heads (paper Eq. 7-8; raw ranking, no
/// wscale/dscale). The caller D2Hs `e` and picks the top-`active_heads` host-side.
/// Launch-only.
///
/// # Safety
/// Async — `q` (`nh·hd` f32), `w` (`nh` f32), `pool` (`m_blocks·hd` f32), and
/// `e` (`nh` f32) valid device pointers live until the next join (the caller's
/// D2H of `e`).
#[cfg(feature = "rocm")]
pub unsafe fn launch_index_head_route(
    q: *const f32,
    w: *const f32,
    pool: *const f32,
    m_blocks: usize,
    nh: usize,
    hd: usize,
    e: *mut f32,
) -> Result<()> {
    debug_assert!(
        m_blocks <= i32::MAX as usize && nh <= i32::MAX as usize && hd <= i32::MAX as usize,
        "index_head_route dims exceed i32 (m_blocks={m_blocks} nh={nh} hd={hd})"
    );
    // SAFETY: caller's contract (see # Safety) covers pointer validity/lifetime.
    let r =
        unsafe { rivoli_index_head_route(q, w, pool, m_blocks as i32, nh as i32, hd as i32, e) };
    check(r, "launch_index_head_route")
}

/// Block until all launched kernels retire (one join point per token), surfacing
/// any async execution fault.
#[cfg(feature = "rocm")]
pub fn device_sync() -> Result<()> {
    // SAFETY: no arguments; hipDeviceSynchronize is always safe to call.
    let r = unsafe { rivoli_device_sync() };
    if r != 0 {
        bail!("device_sync failed (HIP runtime, code {r})");
    }
    Ok(())
}
