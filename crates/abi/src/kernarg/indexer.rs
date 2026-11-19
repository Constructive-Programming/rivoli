//! `kernels/indexer.hip`'s kernels: the DSA/MISA lightning indexer (key LayerNorm, cache append,
//! scoring, the MISA pool and head router, the device top-k) and Qwen's QSA block-key stage — plus
//! the block scorers both `indexer.hip` and `blockindex.hip` launch over one operand set.
//!
//! The `.hip` file named here is deleted from the tree; cite it at tag `archive/hip-kernels`.

/// `layernorm` — `y = (x - mean)/sqrt(var + eps)·w + b` over one row of `n`: a `RowNorm`
/// plus the bias the indexer's `k_norm` ships and no model RMSNorm does.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct LayerNorm {
    pub row: super::RowNorm,
    pub b: *const f32,
}
const _: () = assert!(size_of::<LayerNorm>() == 40);

/// `index_append` — `kcache[pos·hd + i] = bf16(k[i])`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IndexAppend {
    pub k: *const f32,
    pub kcache: *mut u16,
    pub pos: i32,
    pub hd: i32,
}
const _: () = assert!(size_of::<IndexAppend>() == 24);

/// `index_score` — every cached token against the `nact` active query heads (`heads` null = the
/// first `nact`). The launcher's `nh` is a guard only; the kernel never reads it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IndexScore {
    pub q: *const f32,
    pub w: *const f32,
    pub kcache: *const u16,
    pub heads: *const u32,
    pub scores: *mut f32,
    pub nt: i32,
    pub nact: i32,
    pub hd: i32,
    pub wscale: f32,
    pub dscale: f32,
}
const _: () = assert!(size_of::<IndexScore>() == 64);
const _: () = assert!(size_of::<super::Blocked<IndexScore>>() == 72);

/// `index_pool_push` — fold token `t`'s key into its MISA block's running mean.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PoolPush {
    pub k: *const f32,
    pub pool: *mut f32,
    pub t: i32,
    pub hd: i32,
}
const _: () = assert!(size_of::<PoolPush>() == 24);

/// `index_head_route` — the MISA router estimate per head over `m_blocks` pooled keys.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HeadRoute {
    pub q: *const f32,
    pub w: *const f32,
    pub pool: *const f32,
    pub e: *mut f32,
    pub m_blocks: i32,
    pub nh: i32,
    pub hd: i32,
}
const _: () = assert!(size_of::<HeadRoute>() == 48);
const _: () = assert!(size_of::<super::Blocked<HeadRoute>>() == 56);

/// `index_topk` — `rows[0..k)` the top-`k` of `s[0..nt)`, ascending by index; `k <= nt` (the
/// launcher clamps).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TopK {
    pub s: *const f32,
    pub rows: *mut u32,
    pub nt: i32,
    pub k: i32,
}
const _: () = assert!(size_of::<TopK>() == 24);

/// `index_pool_norm_f32` — mean-pool `ratio` keys per block row, then RMSNorm it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PoolNorm {
    pub keys: *const f32,
    pub weight: *const f32,
    pub out: *mut f32,
    pub ratio: i32,
    pub hd: i32,
    pub eps: f32,
}
const _: () = assert!(size_of::<PoolNorm>() == 40);
const _: () = assert!(size_of::<super::Blocked<PoolNorm>>() == 48);

/// The block scorers' four buffers — `q` `[s, heads, hd]`, `kv` `[n_comp, hd]`, `w` `[s, heads]`,
/// `score` `[s, n_comp]`. ONE definition for the engine's call sites and both kernels: three
/// indistinguishable `*const f32` in a row are what the grouping keeps in order (the host's
/// destructure-before-the-call wrapper this replaced said so at length).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScoreBufs {
    pub q: *const f32,
    pub kv: *const f32,
    pub w: *const f32,
    pub score: *mut f32,
}
const _: () = assert!(size_of::<ScoreBufs>() == 32);

/// `index_score_blocks` (bf16, `blockindex.hip`) and `index_score_blocks_f32` — the buffers plus
/// the four extents, narrowed once by the launcher row from the host's `ScoreDims`.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ScoreBlocks {
    pub bufs: ScoreBufs,
    pub s: i32,
    pub n_comp: i32,
    pub heads: i32,
    pub hd: i32,
}
const _: () = assert!(size_of::<ScoreBlocks>() == 48);
