//! The routed-expert descriptors: one expert's six device pointers, which the engine writes
//! into a device array and the MoE kernels read by expert index. Shared by host and device —
//! before the Rust-kernel migration each had a C twin in `kernels/moe.hip` / `moe_f4.hip`
//! (`ExpertDescVq`/`ExpertDescI4`, `ExpertDescF4`) that only "same order" tied to it.

/// One routed expert's six device pointers (per projection: a data ptr + a scale
/// ptr). ONE layout for both formats — byte-identical six-pointer `repr(C)` structs,
/// the kernel picks the interpretation: for int3-VQ (`launch_moe_expert_range`) the
/// pairs are packed 12-bit indices + bf16 group scales (`moe.hip`'s `ExpertDescVq` at tag
/// `archive/hip-kernels`); for
/// int4 (`launch_moe_expert_range_i4`) they are packed 4-bit weights + f32 group
/// scales, one per `I4_GROUP` weights (its `ExpertDescI4`). Both formats are
/// group-scaled; only the scale WIDTH and the weight coding differ. The scale pointer
/// is typed `*const u16` here
/// (built from the VQ carrier) but its VALUE just addresses whatever the kernel reads.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDesc {
    pub gate_indices: *const u8,
    pub gate_scales: *const u16,
    pub up_indices: *const u8,
    pub up_scales: *const u16,
    pub down_indices: *const u8,
    pub down_scales: *const u16,
}

/// One DeepSeek-V4 routed expert's six device pointers — `moe_f4.hip`'s `ExpertDescF4` (tag
/// `archive/hip-kernels`).
///
/// Separate from [`ExpertDesc`] rather than a third interpretation of it. Dispatching a
/// `.f4` block through `launch_moe_expert_range_i4` would decode e2m1 nibbles as
/// `nibble − 8` at group 128 instead of group 32 — plausible magnitudes from the wrong
/// codebook, and there is no shape, size or scale check downstream that could find it. The
/// scales are `*const u8` because e8m0 IS one byte, a third width beside VQ's bf16 and
/// int4's f32, which is where [`ExpertDesc`]'s "one layout, kernel picks the
/// interpretation" stops being honest.
///
/// **The separation is a signpost, not a proof.** Every real dispatch reaches its
/// descriptor array through `buf.ptr() as *const _`, and that cast compiles either way —
/// only construction sites are type-checked. Making it a proof needs the keep-alive buffer
/// and the typed address to be ONE value (a `DescArray<T>` owning the `DeviceBuf` and
/// handing out only `*const T`), which reaches the engine's own
/// `self.descs_buf.ptr() as *const ExpertDesc` and so belongs to the engine's wiring rather
/// than to this ABI type.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ExpertDescF4 {
    /// `w1` — the GATE projection, `[inter, hidden]` as e2m1 nibble pairs.
    pub gate_packed: *const u8,
    /// `[inter, ceil(hidden / F4_GROUP)]` e8m0 bytes, row-major, one scale ROW per output
    /// row — not the 128x128 tile grid fp8 uses.
    pub gate_scale: *const u8,
    /// `w3` — the UP projection. Same shape as `w1`, which is exactly why a swap of the
    /// two is invisible to every structural check (`quant.rs::V4_PROJ`).
    pub up_packed: *const u8,
    /// `[inter, ceil(hidden / F4_GROUP)]` — same grid as `gate_scale`.
    pub up_scale: *const u8,
    /// `w2` — the DOWN projection, `[hidden, inter]`.
    pub down_packed: *const u8,
    /// `[hidden, ceil(inter / F4_GROUP)]` — the reduction dim is `inter` here, not `hidden`.
    pub down_scale: *const u8,
}

impl ExpertDescF4 {
    /// A descriptor that faults if it is ever read.
    ///
    /// Launch-order descriptor tables are sized `n_experts`, so most entries sit past any
    /// one token's selection and no launch names them. Filling those with nulls rather than
    /// with a copy of some resolved expert is the difference between a fault and a
    /// plausible wrong weight the day a range is computed wrongly. On the type because
    /// every arm with an F4 table wants the same defence — the V4 and K3 engines each
    /// carried a verbatim copy under its own name.
    pub fn null() -> Self {
        let n = core::ptr::null();
        Self {
            gate_packed: n,
            gate_scale: n,
            up_packed: n,
            up_scale: n,
            down_packed: n,
            down_scale: n,
        }
    }
}

const _: () = assert!(size_of::<ExpertDesc>() == 48);
const _: () = assert!(size_of::<ExpertDescF4>() == 48);
