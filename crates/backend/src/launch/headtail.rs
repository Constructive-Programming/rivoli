//! `headtail.hip`'s launchers: V4's bf16 embedding broadcast and `Block.hc_head`.

use super::{elementwise, row_gather, run};
use crate::module::{Geometry, Kernel, LdsBytes};
use rivoli_abi::geom::WAVE;
use rivoli_abi::kernarg::{EmbedBf16, HcHeadBlend, HcHeadGate};
use std::ffi::c_void;

static EMBED_BF16_ROW_BCAST: Kernel<EmbedBf16> = Kernel::new(c"embed_bf16_row_bcast");
static HC_HEAD_COLLAPSE_GATE: Kernel<HcHeadGate> = Kernel::new(c"hc_head_collapse_gate");
static HC_HEAD_COLLAPSE_BLEND: Kernel<HcHeadBlend> = Kernel::new(c"hc_head_collapse_blend");

/// # Safety
/// The `launch_embed_bf16_row_bcast` contract.
pub(crate) unsafe fn rivoli_embed_bf16_row_bcast(a: EmbedBf16, stream: *mut c_void) -> i32 {
    // Every refusal is 1001, so the C's order (`hidden`, `token`, `hc`) is moot.
    let checked = if a.hc <= 0 {
        Err(1001)
    } else {
        row_gather(a.token, a.hidden)
    };
    unsafe { run(&EMBED_BF16_ROW_BCAST, checked, stream, &a) }
}

/// The gate (one full wave per token), then the blend on the SAME stream, so the gate's
/// writes precede the blend's reads without a host sync. A failed gate launch returns before
/// the blend, as the C.
///
/// # Safety
/// The `launch_hc_head_collapse` contract.
pub(crate) unsafe fn rivoli_hc_head_collapse(
    g: HcHeadGate,
    y: *mut f32,
    stream: *mut c_void,
) -> i32 {
    if [g.s, g.hc, g.dim].iter().any(|&v| v <= 0) {
        return 1001;
    }
    let gate = Geometry {
        blocks: g.s as u32,
        // One full wave per token: `wave_sum` needs every lane.
        threads: WAVE,
        lds: LdsBytes(0),
    };
    let rc = unsafe { run(&HC_HEAD_COLLAPSE_GATE, Ok(gate), stream, &g) };
    if rc != 0 {
        return rc;
    }
    let b = HcHeadBlend {
        h: g.h,
        pre: g.pre,
        y,
        hc: g.hc,
        dim: g.dim,
        s: g.s,
    };
    let n = i64::from(g.s) * i64::from(g.dim);
    unsafe { run(&HC_HEAD_COLLAPSE_BLEND, elementwise(n), stream, &b) }
}
