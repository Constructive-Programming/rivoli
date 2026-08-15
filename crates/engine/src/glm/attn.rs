//! The GLM MLA attention sublayer, dense rows only at M4 — row selection is
//! `(null, pos + r + 1)`, the attend kernel's fast path, and the differing row count
//! IS the causal mask (row `r` sees the rows earlier rows of the same pass appended,
//! because append and attend sit on the null stream in order).
//!
//! Four phases in launch order, split where the old `gpu.rs` monolith drew its own
//! comment headers: q/kv projections → cache append + absorb → attend → output
//! projection back into the residual. The fp8 GEMVs and the absorb carry every row
//! through ONE read of their weights; the norms/ropes/appends launch per row because
//! their scalar arguments (`pos`) or strides differ per row and each is a microsecond
//! kernel over ≤6144 floats.

use super::engine::GlmEngine;
use super::forward::{ResidualBase, RowNorm, Rows};
use anyhow::Result;
use rivoli_backend::{
    launch_append_kv, launch_attend, launch_gather_rope, launch_gemv_fp8, launch_mla_absorb_fp8,
    launch_mla_value_fp8, launch_rmsnorm_single, launch_rope_interleave, launch_vadd,
};

/// One layer's attention pointers, resolved once per layer so the four phases share a
/// single derivation (a second copy of the pointer arithmetic is a second place for it
/// to be wrong). All Copy raw pointers — holding them across `&mut self` calls borrows
/// nothing.
#[derive(Clone, Copy)]
struct AttnPtrs {
    x: *mut f32,
    xn: *mut f32,
    q_lora: *mut f32,
    q: *mut f32,
    compressed_kv: *mut f32,
    qabsp: *mut f32,
    qropep: *mut f32,
    clatp: *mut f32,
    partial: *mut f32,
    attn_ctx: *mut f32,
    subp: *mut f32,
    lc8p: *mut u8,
    kv_scale: *mut f32,
    rcp: *mut u16,
}

impl GlmEngine<'_> {
    /// The whole attention sublayer for layer `l`: projections, cache append, absorb,
    /// attend, output projection, residual add, pre-MLP norm into `xn`.
    pub(super) fn attention(&mut self, l: usize, rows: Rows, x: ResidualBase) -> Result<()> {
        let p = self.attn_ptrs(l, x);
        self.qkv_project(l, rows, p)?;
        self.cache_and_absorb(l, rows, p)?;
        self.attend_rows(rows, p)?;
        self.output_project(l, rows, p)
    }

    /// Resolve the layer's scratch and KV-slab pointers once.
    fn attn_ptrs(&mut self, l: usize, x: ResidualBase) -> AttnPtrs {
        AttnPtrs {
            x: x.0,
            xn: self.xn.ptr_mut() as *mut f32,
            q_lora: self.q_lora.ptr_mut() as *mut f32,
            q: self.q.ptr_mut() as *mut f32,
            compressed_kv: self.compressed_kv.ptr_mut() as *mut f32,
            qabsp: self.q_absorbed.ptr_mut() as *mut f32,
            qropep: self.q_rope.ptr_mut() as *mut f32,
            clatp: self.ctx_latent.ptr_mut() as *mut f32,
            partial: self.attn_partial.ptr_mut() as *mut f32,
            attn_ctx: self.attn_ctx.ptr_mut() as *mut f32,
            subp: self.attn_out.ptr_mut() as *mut f32,
            lc8p: self.kv_latent[l].ptr_mut(),
            kv_scale: self.kv_latent_scale[l].ptr_mut() as *mut f32,
            rcp: self.kv_rope[l].ptr_mut() as *mut u16,
        }
    }

    /// Phase 1: input norm, then the q-LoRA and kv-LoRA projections.
    fn qkv_project(&mut self, l: usize, rows: Rows, p: AttnPtrs) -> Result<()> {
        let lw = &self.pin.layers[l];
        let (input_ln, q_a, q_a_ln, q_b, kv_a) = (lw.input_ln, lw.q_a, lw.q_a_ln, lw.q_b, lw.kv_a);
        let (qlr, eps) = (self.cfg.q_lora_rank, self.cfg.rms_norm_eps as f32);
        let nrow = rows.nrow;
        // SAFETY: every pointer is live device scratch or a resident weight for its
        // dims; each launch's inputs are produced by a prior launch on the same
        // (default) stream, so ordering holds; every `.add(r * …)` is inside the
        // MAXROW-wide allocation since r < nrow.
        unsafe {
            self.norm_rows(
                RowNorm {
                    src: p.x,
                    w: input_ln,
                    dst: p.xn,
                },
                nrow,
            )?;
            launch_gemv_fp8(
                p.xn, q_a.packed, q_a.scale, q_a.o_dim, q_a.i_dim, q_a.block, nrow, p.q_lora,
            )?;
            for r in 0..nrow {
                let q = p.q_lora.add(r * qlr);
                launch_rmsnorm_single(q, q_a_ln, qlr, eps, q)?; // in place
            }
            launch_gemv_fp8(
                p.q_lora, q_b.packed, q_b.scale, q_b.o_dim, q_b.i_dim, q_b.block, nrow, p.q,
            )?;
            launch_gemv_fp8(
                p.xn,
                kv_a.packed,
                kv_a.scale,
                kv_a.o_dim,
                kv_a.i_dim,
                kv_a.block,
                nrow,
                p.compressed_kv,
            )?;
        }
        Ok(())
    }

    /// Phase 2: normalize the latent, rope key and query, append to the KV slabs,
    /// absorb, gather the roped query halves.
    fn cache_and_absorb(&mut self, l: usize, rows: Rows, p: AttnPtrs) -> Result<()> {
        let cfg = self.cfg;
        let lw = &self.pin.layers[l];
        let (kv_a_ln, kv_b) = (lw.kv_a_ln, lw.kv_b);
        let (h, qh, nope, rope, kvl) = (
            cfg.n_heads,
            cfg.qk_head_dim(),
            cfg.qk_nope_head_dim,
            cfg.qk_rope_head_dim,
            cfg.kv_lora_rank,
        );
        let (eps, theta, nb) = (cfg.rms_norm_eps as f32, cfg.rope_theta(), self.n_kv_blocks);
        // SAFETY: as in `qkv_project` — live scratch, resident weights, null-stream
        // ordering, r < nrow; row `pos + r` is inside the KV slabs by `forward_inner`'s
        // max_ctx check.
        unsafe {
            for r in 0..rows.nrow {
                // `compressed_kv`'s row stride is kvl+rope but the norm covers only the first
                // kvl, so this cannot ride a single-stride batched norm.
                let c = p.compressed_kv.add(r * (kvl + rope));
                launch_rmsnorm_single(c, kv_a_ln, kvl, eps, c)?; // normalize latent
                launch_rope_interleave(c.add(kvl), 1, rope, rope, rows.pos + r, theta)?;
                launch_rope_interleave(
                    p.q.add(r * h * qh + nope),
                    h,
                    qh,
                    rope,
                    rows.pos + r,
                    theta,
                )?;
                launch_append_kv(
                    c,
                    c.add(kvl),
                    p.lc8p,
                    p.kv_scale,
                    p.rcp,
                    rows.pos + r,
                    kvl,
                    rope,
                    nb,
                )?;
            }
            launch_mla_absorb_fp8(
                p.q,
                kv_b.packed,
                kv_b.scale,
                h,
                qh,
                nope,
                cfg.v_head_dim,
                kvl,
                kv_b.block,
                rows.nrow,
                p.qabsp,
            )?;
            for r in 0..rows.nrow {
                launch_gather_rope(
                    p.q.add(r * h * qh),
                    p.qropep.add(r * h * rope),
                    h,
                    qh,
                    nope,
                    rope,
                )?;
            }
        }
        Ok(())
    }

    /// Phase 3: dense flash attend, one launch per row over its own `pos + r + 1` rows.
    fn attend_rows(&mut self, rows: Rows, p: AttnPtrs) -> Result<()> {
        let cfg = self.cfg;
        let (h, kvl, rope, nb) = (
            cfg.n_heads,
            cfg.kv_lora_rank,
            cfg.qk_rope_head_dim,
            self.n_kv_blocks,
        );
        let scale = 1.0 / (cfg.qk_head_dim() as f32).sqrt();
        let scratch = rivoli_backend::attend_scratch_floats(h, kvl);
        // SAFETY: as in `qkv_project`; the null rows pointer is the kernel's dense
        // fast path.
        unsafe {
            for r in 0..rows.nrow {
                launch_attend(
                    p.qabsp.add(r * h * kvl),
                    p.qropep.add(r * h * rope),
                    p.lc8p,
                    p.kv_scale,
                    p.rcp,
                    std::ptr::null(),
                    h,
                    rows.pos + r + 1,
                    kvl,
                    rope,
                    nb,
                    scale,
                    p.clatp.add(r * h * kvl),
                    p.partial.add(r * scratch),
                )?;
            }
        }
        Ok(())
    }

    /// Phase 4: value projection, output projection, residual add, pre-MLP norm.
    fn output_project(&mut self, l: usize, rows: Rows, p: AttnPtrs) -> Result<()> {
        let cfg = self.cfg;
        let lw = &self.pin.layers[l];
        let (kv_b, o_proj, post_ln) = (lw.kv_b, lw.o_proj, lw.post_ln);
        // SAFETY: as in `qkv_project`.
        unsafe {
            launch_mla_value_fp8(
                p.clatp,
                kv_b.packed,
                kv_b.scale,
                cfg.n_heads,
                cfg.qk_nope_head_dim,
                cfg.v_head_dim,
                cfg.kv_lora_rank,
                kv_b.block,
                rows.nrow,
                p.attn_ctx,
            )?;
            launch_gemv_fp8(
                p.attn_ctx,
                o_proj.packed,
                o_proj.scale,
                o_proj.o_dim,
                o_proj.i_dim,
                o_proj.block,
                rows.nrow,
                p.subp,
            )?;
            // Both rows in one launch: `x` and `attn_out` are contiguous row-minor.
            launch_vadd(p.x, p.subp, rows.nrow * cfg.hidden)?; // residual
            self.norm_rows(
                RowNorm {
                    src: p.x,
                    w: post_ln,
                    dst: p.xn,
                },
                rows.nrow,
            )?; // pre-MLP norm → xn
        }
        Ok(())
    }
}
