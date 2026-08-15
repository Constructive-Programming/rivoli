//! GPU-accelerated encode: the nearest-codebook argmin (the run-time ceiling, ~VQ_K×
//! the other work) offloaded to `rivoli_vq_encode`. Host still does fp8 dequant,
//! group normalization, the closed-form scale refit, and packing — all O(o·i), a
//! fraction of the argmin. Produces bytes BIT-IDENTICAL to `quant_vq` (same
//! metric/tie-break in the kernel, same host refit here). Ported from `fp82vq`'s
//! validated GPU path; one encoder per projection codebook, shared behind a `Mutex`
//! so host dequant/refit/pack run parallel while the GPU argmins serialize.
//!
//! MOVED out of `bin/convert.rs` on 2026-08-15 for the CodeScene file-size/cohesion
//! cliff: convert.rs sat at 882 lines and scored **8.54** on Low Cohesion (LCOM4 = 3) —
//! the argmin-offload cluster here shares no data and no call edge with the conversion
//! pipeline that surrounded it, which is exactly what LCOM4 counts. Split by that
//! cohesion boundary rather than by line count: this module is the only `rocm`-gated code
//! in the binary, the only code that names a device buffer or a kernel launch, and its
//! whole interface to the rest of `convert` is `Encoder`, `DenseW` and `quant_vq_gpu`.
//! Cutting anywhere else would have moved lines without moving a responsibility.
//!
//! Every body and comment travelled verbatim. ONE deliberate change came with the move,
//! and only because the move exposed it: the `(w, shape)` parameter pair is now the
//! [`DenseW`] view — see its doc for the measurement. Diluted by the pipeline's
//! `SafeWriter`/`Safetensors` signatures, the pair scored clean in the old file; alone in
//! a file of slices and dims it is 89% primitive arguments, and CodeScene reports
//! Primitive Obsession at 9.68. The margin above the threshold is ONE argument — a
//! future helper here that takes bare `&[f32]` where a named view would do will re-break
//! this gate, which is the smell telling the truth rather than the gate being brittle.

use super::*;
use rivoli_artifact::quant::{set_idx, vq_groups};
use rivoli_backend::hip::{device_sync, launch_vq_encode};
use rivoli_engine::device::DeviceBuf;

/// Subvectors per group — the span one bf16 scale covers, and therefore the slice of
/// pass-1 indices the refit is fitted against.
const SUBS: usize = VQ_GROUP / VQ_DIM;

fn f32_as_bytes(v: &[f32]) -> &[u8] {
    // SAFETY: f32 is POD; the view is `4·len` read-only bytes (LE host == LE device).
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// Resident codebook + `‖c‖²` on device, plus reusable subvector/index buffers
/// sized to the largest projection.
pub struct Encoder {
    codebook: DeviceBuf,
    cbnorm: DeviceBuf,
    sub: DeviceBuf,
    idx: DeviceBuf,
    max_sub: usize,
}

// SAFETY: DeviceBuf holds process-global device pointers; every access goes
// through `encode` behind the caller's Mutex, so device ops never run concurrent.
unsafe impl Send for Encoder {}

impl Encoder {
    pub fn new(codebook: &[f32], max_sub: usize) -> Result<Self> {
        let cbnorm = rivoli_artifact::quant::codebook_norms(codebook);
        let mut cb = DeviceBuf::new(codebook.len() * 4)?;
        cb.copy_in_at(0, f32_as_bytes(codebook))?;
        let mut nb = DeviceBuf::new(cbnorm.len() * 4)?;
        nb.copy_in_at(0, f32_as_bytes(&cbnorm))?;
        Ok(Self {
            codebook: cb,
            cbnorm: nb,
            sub: DeviceBuf::new(max_sub * VQ_DIM * 4)?,
            idx: DeviceBuf::new(max_sub * 2)?,
            max_sub,
        })
    }

    pub fn encode(&mut self, sub: &[f32]) -> Result<Vec<u16>> {
        let n = sub.len() / VQ_DIM;
        ensure!(
            n <= self.max_sub,
            "encode batch {n} > max_sub {}",
            self.max_sub
        );
        self.sub.copy_in_at(0, f32_as_bytes(sub))?;
        // SAFETY: buffers sized ≥ n; live until the sync below.
        unsafe {
            launch_vq_encode(
                self.sub.ptr() as *const f32,
                self.codebook.ptr() as *const f32,
                self.cbnorm.ptr() as *const f32,
                n,
                self.idx.ptr_mut() as *mut u16,
            )?;
        }
        device_sync()?;
        let mut bytes = Vec::new();
        self.idx.copy_out_prefix(&mut bytes, n * 2)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect())
    }
}

/// One dequantized projection: row-major `W[o_dim, i_dim]` f32 with the dims that index
/// it. Every walk below strides `w` by `i_dim` and every geometry here is derived from
/// that pair, so they travel as one value — the same pairing `Fp8W`/`VqW`/`RowScaledW`
/// make for the on-disk formats in `artifact::quant`.
///
/// Split out of the parameter lists when this module moved to its own file (2026-08-15).
/// Isolated from the conversion pipeline that had diluted it, the bare
/// `(&[f32], [usize; 2])` pair read as CodeScene Primitive Obsession — measured 9.68
/// with the pair spelled out at four call sites, 10.0 with it named once. The smell was
/// pointing at something real: nothing but argument order said the shape described those
/// weights, and `quant_vq_gpu` is the one entry point where a shape from a DIFFERENT
/// projection would still index in bounds.
#[derive(Clone, Copy)]
pub struct DenseW<'a> {
    pub w: &'a [f32],
    pub shape: [usize; 2],
}

/// One projection's geometry, derived once. Both passes and the packer index the same
/// three arrays off it, so a recomputation is a chance for them to disagree.
#[derive(Clone, Copy)]
struct Shape {
    o_dim: usize,
    i_dim: usize,
    ngroups: usize,
    nsub_row: usize,
}

impl Shape {
    /// Every group of `dw` as `(weights, staging subvector base, scale index)`, in
    /// row-major order. Stated once because both passes walk it and a refit is only
    /// sound if it renormalizes the exact span pass 1 normalized.
    ///
    /// By value, not by reference: the caller reads it out of the `Staging` it is about
    /// to write, so a borrow here would collide with that write.
    ///
    /// Takes the whole [`DenseW`] and reads only `w` from it: `self` is the geometry
    /// DERIVED from that same view in `Staging::new`, so the pair cannot disagree unless
    /// a caller mixes two projections — which is the hazard the view exists to close.
    fn groups<'w>(self, dw: DenseW<'w>) -> impl Iterator<Item = (&'w [f32], usize, usize)> {
        let Shape {
            o_dim,
            i_dim,
            ngroups,
            nsub_row,
        } = self;
        (0..o_dim).flat_map(move |o| {
            (0..ngroups).map(move |grp| {
                (
                    &dw.w[o * i_dim + grp * VQ_GROUP..][..VQ_GROUP],
                    o * nsub_row + grp * SUBS,
                    o * ngroups + grp,
                )
            })
        })
    }
}

/// What the host hands the GPU and what it keeps: `sub` is every group's weights
/// divided by its scale (the argmin input), `scales` the bf16 scale each was divided
/// by. One value because they are only meaningful together — a scale that moves
/// invalidates exactly its own subvectors.
struct Staging {
    sh: Shape,
    sub: Vec<f32>,
    scales: Vec<u16>,
}

impl Staging {
    /// Derives the geometry as well as the arrays: `Shape` has no other construction
    /// site, and one constructor is one place for the two to disagree about the dims.
    fn new(dw: DenseW<'_>) -> Self {
        let [o_dim, i_dim] = dw.shape;
        let _ = vq_groups(i_dim); // dim sanity (asserts i_dim % VQ_GROUP == 0)
        let sh = Shape {
            o_dim,
            i_dim,
            ngroups: i_dim / VQ_GROUP,
            nsub_row: i_dim / VQ_DIM,
        };
        Staging {
            sh,
            sub: vec![0.0f32; sh.o_dim * sh.nsub_row * VQ_DIM],
            scales: vec![0u16; sh.o_dim * sh.ngroups],
        }
    }

    /// Divide one group by `scale` into its subvector span — the only write into `sub`,
    /// so pass 1 and the refit cannot normalize differently.
    fn renorm(&mut self, seg: &[f32], sbase: usize, scale: u16) {
        let inv = 1.0 / bf16_to_f32(scale);
        let base = sbase * VQ_DIM;
        for (t, &v) in seg.iter().enumerate() {
            self.sub[base + t] = v * inv;
        }
    }

    /// Pass 1: normalize every group by its amax-derived bf16 scale.
    fn normalize(&mut self, dw: DenseW<'_>) {
        for (seg, sbase, si) in self.sh.groups(dw) {
            let amax = seg.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
            let sb = f32_to_bf16(if amax > 0.0 { amax } else { 1.0 });
            self.scales[si] = sb;
            self.renorm(seg, sbase, sb);
        }
    }

    /// Refit each group against its pass-1 entries; renormalize only changed groups.
    /// Reports whether any changed — if none did, pass 1's indices are already final
    /// and the second GPU argmin is skipped entirely.
    fn refit(&mut self, dw: DenseW<'_>, codebook: &[f32], idx1: &[u16]) -> bool {
        let mut changed = false;
        for (seg, sbase, si) in self.sh.groups(dw) {
            // The refit itself is `quant_vq`'s, called rather than restated: the two
            // encoders are asserted bit-identical under `--validate`, and this
            // accumulation is where a divergence would come from.
            let refit = rivoli_artifact::quant::vq_refit(seg, &idx1[sbase..sbase + SUBS], codebook);
            if let Some(refit) = refit
                && refit != self.scales[si]
            {
                self.scales[si] = refit;
                self.renorm(seg, sbase, refit);
                changed = true;
            }
        }
        changed
    }

    /// Flat per-subvector indices → the on-disk row layout (`set_idx` owns the 12-bit
    /// packing, so this only walks it), paired with the scales they were fitted under.
    fn pack(self, idx: &[u16]) -> (Vec<u8>, Vec<u16>) {
        let (rb, nsub_row) = (vq_row_bytes(self.sh.i_dim), self.sh.nsub_row);
        let mut indices = vec![0u8; self.sh.o_dim * rb];
        for (o, ir) in indices.chunks_exact_mut(rb).enumerate() {
            for t in 0..nsub_row {
                set_idx(ir, t, idx[o * nsub_row + t]);
            }
        }
        (indices, self.scales)
    }
}

/// One locked argmin batch. Both call sites go through it so the poison mapping — a
/// worker that died mid-launch, from which there is no recovery here — is written once.
fn argmin(enc: &std::sync::Mutex<Encoder>, sub: &[f32]) -> Result<Vec<u16>> {
    enc.lock()
        .map_err(|_| anyhow::anyhow!("encoder poisoned"))?
        .encode(sub)
}

/// GPU analog of `quant_vq`: two batched argmin passes (amax-scale, then the
/// refit scale) around the same host refit. Bit-identical output to `quant_vq`.
pub fn quant_vq_gpu(
    dw: DenseW<'_>,
    codebook: &[f32],
    enc: &std::sync::Mutex<Encoder>,
) -> Result<(Vec<u8>, Vec<u16>)> {
    let mut st = Staging::new(dw);
    st.normalize(dw);
    let idx1 = argmin(enc, &st.sub)?;
    let idx = if st.refit(dw, codebook, &idx1) {
        argmin(enc, &st.sub)?
    } else {
        idx1
    };
    Ok(st.pack(&idx))
}
