//! `kernels/attn.hip`'s kernels: MLA's latent flash attend and its split combine, V4's
//! shared-KV gather attend, the GQA pair (causal ring, bidirectional block) and K3's dense MHA.

/// What an MLA attend reads and writes, as its launcher's caller hands it over — the
/// [`MlaAttend`] block less the two fields the launcher's split plan decides.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlaIo {
    /// `[h * kvl]` absorbed queries.
    pub qabs: *const f32,
    /// `[h * rope]` roped queries.
    pub qrope: *const f32,
    /// `[tokens * kvl]` fp8-e4m3 latent cache.
    pub lc8: *const u8,
    /// `[tokens * n_blocks]` per-128 block scales.
    pub lscale: *const f32,
    /// `[tokens * rope]` bf16 roped keys.
    pub rc: *const u16,
    /// `[nr]` attended token indices, or null = the dense prefix `0..nr`.
    pub rows: *const u32,
    /// `[h * kvl]` output, written when the plan is one split.
    pub clat: *mut f32,
    pub h: i32,
    pub nr: i32,
    pub kvl: i32,
    pub rope: i32,
    pub n_blocks: i32,
    pub scale: f32,
}
const _: () = assert!(size_of::<MlaIo>() == 80);

/// `mla_latent_attend` — [`MlaIo`] plus the split plan: `tps` tile steps per split, and
/// `partial` (the caller's scratch when the plan splits, null when it does not).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlaAttend {
    pub io: MlaIo,
    pub tps: i32,
    pub partial: *mut f32,
}
const _: () = assert!(size_of::<MlaAttend>() == 96);

/// `mla_attend_combine` — merge `n_splits` partials of `h` heads × `kvl` into `clat`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MlaCombine {
    pub partial: *const f32,
    pub clat: *mut f32,
    pub h: i32,
    pub kvl: i32,
    pub n_splits: i32,
}
const _: () = assert!(size_of::<MlaCombine>() == 32);

/// `gather_attn_shared_kv` — per (query row, head), softmax over the `topk` `kv` rows `idxs`
/// names (`< 0` masks), `sink[head]` in the denominator only; `o` is `[m][h][d]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GatherAttn {
    pub q: *const f32,
    pub kv: *const f32,
    pub sink: *const f32,
    pub idxs: *const i32,
    pub o: *mut f32,
    pub h: i32,
    pub d: i32,
    pub topk: i32,
    pub scale: f32,
}
const _: () = assert!(size_of::<GatherAttn>() == 56);

/// The operands both GQA attends share: `out`/`q` `[rows][hq][d]`, `k`/`v` `[slots][hkv][d]`,
/// window `win`, softmax scale `scale`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Gqa {
    pub out: *mut f32,
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub hq: i32,
    pub hkv: i32,
    pub d: i32,
    pub win: i32,
    pub scale: f32,
}
const _: () = assert!(size_of::<Gqa>() == 56);

/// `gqa_attend` — causal, row 0 at absolute `start_pos`; `ring_cap` maps a position to its
/// slot (`0` = slot is position).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GqaRing {
    pub gqa: Gqa,
    pub start_pos: i32,
    pub ring_cap: i32,
}
const _: () = assert!(size_of::<GqaRing>() == 64);

/// `gqa_block_attend` — bidirectional, row 0 at `q_offset`, over `kv_len` linear rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct GqaBlock {
    pub gqa: Gqa,
    pub q_offset: i32,
    pub kv_len: i32,
}
const _: () = assert!(size_of::<GqaBlock>() == 64);

/// `mha_attend` — per head, `q` `[d]` against `k` `[kv][d]` and `v` `[kv][dv]`, `mask` an
/// additive `[kv]` row or null; `out` `[heads][dv]`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MhaAttend {
    pub q: *const f32,
    pub k: *const f32,
    pub v: *const f32,
    pub mask: *const f32,
    pub out: *mut f32,
    pub kv: i32,
    pub d: i32,
    pub dv: i32,
    pub scale: f32,
}
const _: () = assert!(size_of::<MhaAttend>() == 56);
