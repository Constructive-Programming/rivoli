//! The dense trunk's storage WIDTH, and the byte census the floor is stated from: which of the
//! five large attention projections an artifact holds as fp8-e4m3 with a per-tile scale grid, the
//! sniff that decides it, the placer that resolves either width, and the totals
//! [`measure_resident`] hands the partition.
//!
//! A child module of [`super`] for the 800-line cap, and the cohesion line is "what the dense
//! weights are stored in and what that costs" — the parent places them, this file says in which
//! container and for how many bytes. `pin/mxfp8.rs` is the other half of that split and asks the
//! sibling question about the ROUTED experts; the two never overlap.

use super::*;
use crate::inkling::geometry::global_weight_bytes;
use crate::resident::{Fp8Weight, place_fp8};
use rivoli_artifact::format::FormatMeta;

/// One dense-trunk projection: bf16, or fp8-e4m3 block-scaled.
///
/// **The five large attention projections are the only tensors a dense fp8 converter rewrites**
/// (`self_attn.{q,k,v}_proj.linear`, `o_proj.gate_proj`, `o_proj.linear`). Everything else dense —
/// the norms, the sink, the relative-bias triple, the router, the message gates, the embedding and
/// the head — is bf16 on every artifact, so those stay [`Bf16Weight`] and pay no dispatch at all.
///
/// Two arms rather than a nullable `Fp8Weight` beside a `Bf16Weight`: a bf16 artifact would then
/// carry four dead pointers per layer, and the forward would have to test one of them per launch.
#[derive(Clone, Copy)]
pub enum DenseWeight {
    Bf16(Bf16Weight),
    Fp8(Fp8Weight),
}

impl DenseWeight {
    /// The reduction width the activation must match — the one question both arms answer.
    pub fn i_dim(&self) -> usize {
        match self {
            Self::Bf16(w) => w.i_dim,
            Self::Fp8(w) => w.i_dim,
        }
    }
}

/// The artifact's dense fp8 block size, or `None` on a bf16 trunk.
///
/// **Sniffed from ONE tensor — layer 0's `q_proj.linear` — and only its DTYPE.** A converter that
/// rewrites the trunk rewrites all five projections on every layer or none, so asking per tensor
/// would only move a mixed artifact's failure from load to the first launch that reads the odd
/// one out. What the dtype cannot say is the block size, and that is the manifest's
/// ([`FormatMeta::load`]) — read ONLY when the trunk really is fp8, so a bf16 artifact never needs
/// a `manifest.json` at all, which is the state every pre-fp8 Mini artifact is in.
pub(super) fn sniff(dir: &str, st: &Safetensors) -> Result<Option<usize>> {
    let name = format!("{}.{}", names::layer_prefix(0), names::Q_LINEAR);
    let (_, dt, _) = st.raw(&name)?;
    match dt {
        Dtype::Bf16 => Ok(None),
        Dtype::F8E4M3 => Ok(Some(FormatMeta::load(dir)?.fp8_block)),
        other => bail!(
            "{name} is {other:?}: a dense projection is bf16 or fp8-e4m3, and a third width has \
             no launcher on this arm"
        ),
    }
}

/// Place one dense projection in whichever container the artifact stores it in.
///
/// `place_fp8` takes the BASE name (`…linear`) and appends `.weight`/`.weight_scale_inv` itself,
/// while the artifact's tensor census spells the weight (`…linear.weight`) — so the suffix is
/// stripped here, once, rather than at each of the five call sites. The strip is also the check
/// that a name reaching an fp8 placer is one this module understands.
pub(super) fn place_dense_proj(
    tier: &mut DeviceTier,
    st: &Safetensors,
    name: &str,
    fp8_block: Option<usize>,
) -> Result<DenseWeight> {
    let Some(block) = fp8_block else {
        return Ok(DenseWeight::Bf16(place_bf16(tier, st, name)?));
    };
    let base = name
        .strip_suffix(".weight")
        .with_context(|| format!("{name} is not a `.weight` tensor"))?;
    Ok(DenseWeight::Fp8(place_fp8(tier, st, base, block)?))
}

/// The dense (non-expert) byte total over every layer, read off the artifact's own headers.
///
/// **The artifact's lengths, not `dense_layer_elements` × a dtype.** The per-layer census formula
/// counts ELEMENTS across two storage widths (bf16 projections, f32 norms), so multiplying it by
/// either width is wrong by the other one's share; the header is exact by construction and is the
/// same tensor list [`place_layer`](super::place_layer) places.
///
/// **An fp8 projection costs TWO tensors, and the second is not in the census list.** The name
/// walk lists `…linear.weight` only, so the `…weight_scale_inv` grid would be missing from the
/// floor and [`crate::device::DeviceTier`] would run out of slab part-way through the placement.
/// The grids are small — a `[4096, 4608]` weight at block 128 carries a 32×36 f32 grid, 4.6 KB
/// against the 18.9 MB weight — and that is exactly why the omission would survive review: the
/// tier's bump cursor is what notices, at the LAST placement, as a refusal that names no tensor.
/// The width is read PER TENSOR, so an artifact with fp8 on some layers and bf16 on others is
/// counted correctly without a second rule.
fn dense_total(st: &Safetensors, flags: &[(bool, bool)]) -> Result<usize> {
    let mut n = 0;
    for (l, &(full, looped)) in flags.iter().enumerate() {
        for name in names::layer_tensor_names(l, full, looped) {
            if name.contains(".mlp.experts.") {
                continue; // the routed unit: `expert_layer_bytes`
            }
            let (raw, dt, _) = st.raw(&name)?;
            n += raw.len();
            if dt == Dtype::F8E4M3 {
                let base = name
                    .strip_suffix(".weight")
                    .with_context(|| format!("fp8 tensor {name} is not a `.weight` tensor"))?;
                n += st.raw(&format!("{base}.weight_scale_inv"))?.0.len();
            }
        }
    }
    Ok(n)
}

/// The global tensors' byte total, read off the artifact — equal to
/// [`global_weight_bytes`]'s derived figure, which is what the floor uses.
fn globals_bytes(st: &Safetensors, text: &InklingTextConfig) -> Result<usize> {
    let mut n = 0;
    for (i, name) in names::GLOBAL_SUFFIXES.iter().enumerate() {
        if i == 4 && text.loop_config.is_none() {
            continue; // message_input.query is the looped member's alone
        }
        n += st.raw(name)?.0.len();
    }
    Ok(n)
}

/// Measure the artifact's global + dense bytes and verify the globals agree with geometry.
pub(super) fn measure_resident(
    st: &Safetensors,
    text: &InklingTextConfig,
    flags: &[(bool, bool)],
) -> Result<(usize, usize)> {
    let dense = dense_total(st, flags)?;
    let globals = global_weight_bytes(text);
    let carried = globals_bytes(st, text)?;
    ensure!(
        carried == globals,
        "the artifact's global tensors total {carried} bytes against geometry's {globals} — \
         the floor is stated from a derivation that no longer matches the artifact"
    );
    Ok((globals, dense))
}
