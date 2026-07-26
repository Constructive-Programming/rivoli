//! GPU kernels vs their CPU oracles in quant.rs. Compiles to nothing without rocm.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli::device::DeviceBuf;
use rivoli::gpustream::HipStream;
use rivoli::hip::{
    ExpertDesc, attend_scratch_floats, device_sync, launch_argmax, launch_attend, launch_gemv_fp8,
    launch_gemv_i8, launch_gemv_vq, launch_index_topk, launch_mla_absorb_fp8, launch_mla_value_fp8,
    launch_moe_expert_range, launch_moe_expert_range_i4, launch_moe_reduce, launch_vadd,
};
use rivoli::math::{bf16_to_f32, e4m3_to_f32, f32_to_bf16, f32_to_e4m3, silu, softmax};
use rivoli::quant::{VQ_DIM, VQ_K, matvec_fp8, matvec_i8, matvec_vq, quant_vq};

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
/// f32 → fp16 bytes — the VQ codebook is uploaded fp16 (the kernel decodes __half),
/// while the CPU reference keeps the f32 codebook, so these oracles measure exactly
/// the fp16 codebook-rounding error against the tol.
fn f16b(v: &[f32]) -> Vec<u8> {
    u16b(&v.iter().map(|&x| rivoli::math::f32_to_f16(x)).collect::<Vec<_>>())
}
fn f32v(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn dev(b: &[u8]) -> DeviceBuf {
    let mut d = DeviceBuf::new(b.len()).expect("alloc");
    d.copy_in_at(0, b).expect("fill");
    d
}
fn assert_close(want: &[f32], got: &[f32], label: &str) {
    let mx = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let err = want
        .iter()
        .zip(got)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let tol = 1e-3 * mx + 1e-3;
    // Print the MARGIN, not just pass/fail. An oracle clearing its threshold by 100x
    // and one clearing it by 2x look identical in a green test run, and only the
    // second is evidence of anything.
    println!(
        "{label}: err={err:.3e} tol={tol:.3e} margin={:.1}x",
        tol / err.max(f32::MIN_POSITIVE)
    );
    assert!(err <= tol, "{label}: err={err:.3e} > tol={tol:.3e} max={mx:.3e}");
}

struct Lcg(u64);
impl Lcg {
    /// Uniform in [-1, 1).
    ///
    /// `>> 32`, not `>> 33`. The old shift kept only 31 bits, so dividing by
    /// `u32::MAX` gave [0, 0.5) and `*2 - 1` gave [-1, 0) — **every sample negative**,
    /// for the whole life of this file. In a matvec oracle that makes every
    /// x[i]*w[i] product positive, so the partial sums GROW instead of cancelling:
    /// `mx` inflates, the `1e-3 * mx` relative tolerance inflates with it, and the
    /// oracles have been passing on roughly two orders of magnitude of headroom. It
    /// also means no oracle here has ever exercised floating-point cancellation —
    /// the only regime where summation order matters, and the entire reason the
    /// kernels reduce with a fixed `__shfl_down` ladder instead of an atomic.
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn gemv_fp8_matches_oracle() {
    // block-scaled fp8 GEMV vs matvec_fp8. Two shapes: a short reduction (plain
    // wave-per-row path) and a long one (i_dim ≥ 4096 → the split-K path
    // launch_gemv_fp8 dispatches to for the o_proj-class projections).
    let block = 128usize;
    for (o_dim, i_dim, label) in [(256usize, 512usize, "gemv_fp8"), (128, 16384, "gemv_fp8_splitk")] {
        let mut r = Lcg(0xF8 ^ i_dim as u64);
        let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| f32_to_e4m3(r.f())).collect();
        let sc_cols = i_dim / block;
        let scale: Vec<f32> = (0..(o_dim / block) * sc_cols)
            .map(|_| (r.f() * 0.1).abs() + 0.01)
            .collect();
        let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();

        let mut want = vec![0.0f32; o_dim];
        matvec_fp8(&mut want, &x, &packed, &scale, i_dim, block);

        let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
        let mut yb = dev(&vec![0u8; o_dim * 4]);
        unsafe {
            launch_gemv_fp8(
                xb.ptr() as *const f32,
                pb.ptr(),
                sb.ptr() as *const f32,
                o_dim,
                i_dim,
                block,
                yb.ptr_mut() as *mut f32,
            )
            .expect("launch");
        }
        device_sync().expect("sync");
        assert_close(&want, &f32v(&yb.copy_out().expect("out")), label);
    }
}

#[test]
fn gemv_fp8_rejects_non_power_of_two_block() {
    // The fp8 dot indexes the block scale with a SHIFT (`blk_shift`), which is only
    // the same index as `i / block` for a power-of-two tile. The launcher must reject
    // anything else rather than compute a silently wrong scale index.
    //
    // Asserts the guard CODE, not just is_err(): a test that accepted any error would
    // still pass if someone replaced the power-of-two test with `block != 128`, or if
    // the dim guard (1001) started swallowing these first. Both i_dim arms are covered
    // because `rivoli_gemv_fp8` dispatches to the split-K launcher at i_dim >= 4096
    // (GEMV_SPLITK_MIN_IDIM) and that launcher carries its own copy of the guard.
    let o_dim = 64usize;
    for i_dim in [512usize, 8192] {
        let packed = dev(&vec![0u8; o_dim * i_dim]);
        let scale = dev(&f32b(&vec![1.0f32; o_dim * i_dim]));
        let x = dev(&f32b(&vec![0.0f32; i_dim]));
        let mut y = dev(&vec![0u8; o_dim * 4]);
        // 1 is a power of two (bsh = 0, `i >> 0` == `i / 1`), so it must be ACCEPTED.
        for (block, want) in [(128usize, None), (96, Some(1003)), (127, Some(1003)), (1, None)] {
            // SAFETY: all four buffers are device-resident and amply sized for these dims.
            let r = unsafe {
                launch_gemv_fp8(
                    x.ptr() as *const f32,
                    packed.ptr(),
                    scale.ptr() as *const f32,
                    o_dim,
                    i_dim,
                    block,
                    y.ptr_mut() as *mut f32,
                )
            };
            match want {
                None => assert!(r.is_ok(), "i_dim={i_dim} block={block}: {r:?}"),
                Some(code) => {
                    let msg = format!("{:#}", r.expect_err("expected a guard rejection"));
                    assert!(
                        msg.contains(&code.to_string()),
                        "i_dim={i_dim} block={block}: want guard {code}, got {msg:?}"
                    );
                }
            }
        }
        device_sync().expect("sync");
    }
}

#[test]
fn mla_attend_rejects_unsupported_kvl() {
    // `mla_latent_attend` keeps its online accumulator in MLA_ACC_REGS*SUBW = 512
    // registers per lane and indexes it by k = (i - lane)/SUBW, so a kvl over that cap
    // — or one that is not a multiple of 128 — must be REJECTED (guard 1004), never run
    // with columns silently dropped or the wrong per-128 block scale applied.
    let (h, kvl, rope) = (8usize, 512usize, 64usize);
    let mut scratch = dev(&vec![0u8; attend_scratch_floats(h, kvl) * 4]);
    let big = dev(&vec![0u8; h * 1024 * 4]);
    let mut clat = dev(&vec![0u8; h * kvl * 4]);
    for (bad_kvl, want) in [(kvl, None), (640usize, Some(1004)), (160, Some(1004))] {
        // SAFETY: every buffer is sized for the largest kvl tried (1024 floats/head);
        // the rejected cases never dereference them at all.
        let r = unsafe {
            launch_attend(
                big.ptr() as *const f32,
                big.ptr() as *const f32,
                big.ptr(),
                big.ptr() as *const f32,
                big.ptr() as *const u16,
                std::ptr::null(),
                h,
                16,
                bad_kvl,
                rope,
                bad_kvl / 128,
                1.0,
                clat.ptr_mut() as *mut f32,
                scratch.ptr_mut() as *mut f32,
            )
        };
        match want {
            None => assert!(r.is_ok(), "kvl={bad_kvl}: {r:?}"),
            Some(code) => {
                let msg = format!("{:#}", r.expect_err("expected a guard rejection"));
                assert!(msg.contains(&code.to_string()), "kvl={bad_kvl}: got {msg:?}");
            }
        }
    }
    device_sync().expect("sync");
}

#[test]
fn gemv_i8_matches_oracle() {
    // lm_head's GEMV — the last op before argmax, and until now the only quantized
    // kernel in the engine with NO oracle at all.
    let (o_dim, i_dim) = (512usize, 6144usize);
    let mut r = Lcg(0x18);
    let packed: Vec<u8> = (0..o_dim * i_dim).map(|_| (r.f() * 127.0) as i8 as u8).collect();
    let scale: Vec<f32> = (0..o_dim).map(|_| (r.f() * 0.01).abs() + 1e-4).collect();
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();

    let mut want = vec![0.0f32; o_dim];
    matvec_i8(&mut want, &x, &packed, &scale, o_dim, i_dim);

    let (xb, pb, sb) = (dev(&f32b(&x)), dev(&packed), dev(&f32b(&scale)));
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    // SAFETY: device buffers sized [i_dim], [o_dim*i_dim], [o_dim], [o_dim] f32.
    unsafe {
        launch_gemv_i8(
            xb.ptr() as *const f32,
            pb.ptr(),
            sb.ptr() as *const f32,
            o_dim,
            i_dim,
            yb.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&yb.copy_out().expect("out")), "gemv_i8");
}

#[test]
fn gemv_vq_matches_oracle() {
    let mut r = Lcg(0x53);
    let (o_dim, i_dim) = (2048usize, 512usize);
    let codebook: Vec<f32> = (0..VQ_K * VQ_DIM).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
    let (indices, scales) = quant_vq(&w, o_dim, i_dim, &codebook);
    let x: Vec<f32> = (0..i_dim).map(|_| r.f()).collect();

    let mut want = vec![0.0f32; o_dim];
    matvec_vq(&mut want, &x, &indices, &scales, &codebook, o_dim, i_dim);

    let (xb, ib, sb, cb) = (
        dev(&f32b(&x)),
        dev(&indices),
        dev(&u16b(&scales)),
        dev(&f16b(&codebook)),
    );
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    unsafe {
        launch_gemv_vq(
            xb.ptr() as *const f32,
            ib.ptr(),
            sb.ptr() as *const u16,
            cb.ptr() as *const u16,
            o_dim,
            i_dim,
            yb.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&yb.copy_out().expect("out")), "gemv_vq");
}

/// argmax_reduce: max value wins, ties → lowest index, NaN never wins. Plus vadd
/// as a residual-add smoke check (both fwd.hip glue).
#[test]
fn fwd_argmax_and_vadd() {
    // argmax: a plateau (tie → lowest index) with a NaN that must lose.
    let mut logits = vec![0.1f32, 0.5, 0.5, f32::NAN, 0.3, 0.5, -1.0];
    let want_idx = 1i32; // first 0.5
    let lb = dev(&f32b(&logits));
    let mut ib = dev(&[0u8; 4]);
    let mut vb = dev(&[0u8; 4]);
    unsafe {
        launch_argmax(
            lb.ptr() as *const f32,
            logits.len(),
            ib.ptr_mut() as *mut i32,
            vb.ptr_mut() as *mut f32,
        )
        .expect("launch argmax");
    }
    device_sync().expect("sync");
    let got_idx = i32::from_le_bytes(
        ib.copy_out().expect("out")[..4]
            .try_into()
            .expect("4 bytes"),
    );
    let got_val = f32::from_le_bytes(
        vb.copy_out().expect("out")[..4]
            .try_into()
            .expect("4 bytes"),
    );
    assert_eq!(got_idx, want_idx, "argmax idx");
    assert_eq!(got_val, 0.5, "argmax val");

    // vadd: x += y, elementwise.
    let y: Vec<f32> = logits
        .iter()
        .map(|v| if v.is_nan() { 0.0 } else { *v })
        .collect();
    for l in logits.iter_mut() {
        if l.is_nan() {
            *l = 0.0;
        }
    }
    let mut xb = dev(&f32b(&logits));
    let yb = dev(&f32b(&y));
    unsafe {
        launch_vadd(
            xb.ptr_mut() as *mut f32,
            yb.ptr() as *const f32,
            logits.len(),
        )
        .expect("vadd");
    }
    device_sync().expect("sync");
    let got = f32v(&xb.copy_out().expect("out"));
    let want: Vec<f32> = logits.iter().zip(&y).map(|(a, b)| a + b).collect();
    assert_close(&want, &got, "vadd");
}

/// Dense two-pass scalar softmax over the whole cache — the oracle the flash
/// (online) kernel must match. Latent is fp8-e4m3 × per-128 block scale (the exact
/// dequant the kernel does); the roped key is bf16.
#[allow(clippy::too_many_arguments)]
fn attend_reference(
    qabs: &[f32],
    qrope: &[f32],
    lc8: &[u8],
    lscale: &[f32],
    rc: &[u16],
    h: usize,
    nt: usize,
    kvl: usize,
    rope: usize,
    n_blocks: usize,
    scale: f32,
) -> Vec<f32> {
    let lat = |t: usize, i: usize| e4m3_to_f32(lc8[t * kvl + i]) * lscale[t * n_blocks + i / 128];
    let mut clat = vec![0.0f32; h * kvl];
    let mut scores = vec![0.0f32; nt];
    for head in 0..h {
        let qa = &qabs[head * kvl..(head + 1) * kvl];
        let qr = &qrope[head * rope..(head + 1) * rope];
        for (t, sc) in scores.iter_mut().enumerate() {
            let rrow = &rc[t * rope..(t + 1) * rope];
            let mut a = 0.0f32;
            // `lat(t, i)` needs the index, so a range loop is the clear form here.
            #[allow(clippy::needless_range_loop)]
            for i in 0..kvl {
                a += qa[i] * lat(t, i);
            }
            for (d, &rb) in rrow.iter().enumerate() {
                a += qr[d] * bf16_to_f32(rb);
            }
            *sc = a * scale;
        }
        softmax(&mut scores);
        let out = &mut clat[head * kvl..(head + 1) * kvl];
        for (t, &sc) in scores.iter().enumerate() {
            for (i, o) in out.iter_mut().enumerate() {
                *o += sc * lat(t, i);
            }
        }
    }
    clat
}

fn check_attend(seed: u64, h: usize, nt: usize, kvl: usize, rope: usize) {
    assert_eq!(kvl % 128, 0, "fp8 latent cache needs kvl a multiple of 128");
    let n_blocks = kvl / 128;
    let mut r = Lcg(seed);
    let qabs: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();
    let qrope: Vec<f32> = (0..h * rope).map(|_| r.f()).collect();
    // Latent as fp8 bytes + positive per-128 block scales; key round-tripped through
    // bf16. Kernel and reference decode identical bits — the only difference under
    // test is flash (online) vs two-pass softmax.
    let lc8: Vec<u8> = (0..nt * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let lscale: Vec<f32> = (0..nt * n_blocks)
        .map(|_| (r.f() * 0.1).abs() + 0.01)
        .collect();
    let rc: Vec<u16> = (0..nt * rope).map(|_| f32_to_bf16(r.f())).collect();
    let scale = 1.0 / ((kvl + rope) as f32).sqrt();
    let want = attend_reference(
        &qabs, &qrope, &lc8, &lscale, &rc, h, nt, kvl, rope, n_blocks, scale,
    );

    let (qab, qrb) = (dev(&f32b(&qabs)), dev(&f32b(&qrope)));
    let (lcb, lsb, rcb) = (dev(&lc8), dev(&f32b(&lscale)), dev(&u16b(&rc)));
    let mut clatb = dev(&vec![0u8; h * kvl * 4]);
    // Non-null worst-case scratch → the multi-split + combine path is what's checked.
    let mut partb = dev(&vec![0u8; attend_scratch_floats(h, kvl) * 4]);
    unsafe {
        launch_attend(
            qab.ptr() as *const f32,
            qrb.ptr() as *const f32,
            lcb.ptr(),
            lsb.ptr() as *const f32,
            rcb.ptr() as *const u16,
            std::ptr::null(), // dense (no row gather)
            h,
            nt,
            kvl,
            rope,
            n_blocks,
            scale,
            clatb.ptr_mut() as *mut f32,
            partb.ptr_mut() as *mut f32,
        )
        .expect("launch attend");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&clatb.copy_out().expect("out")), "mla_attend");
}

#[test]
fn mla_attend_glm_dims() {
    // GLM MLA (kv_lora=512, qk_rope=64); H=64 = 8 full HB blocks, nt spans many
    // TILE steps → the split-KV planner picks n_splits>1.
    check_attend(0x0a5e_1102, 64, 300, 512, 64);
}

#[test]
fn mla_attend_edges() {
    // Single cached token stresses the online init (m=-inf on the first token);
    // H=20 → a partial second HB block (4 active + 4 inactive lanes). kvl=128 = the
    // smallest legal fp8-block latent (n_blocks=1).
    check_attend(0x00c0_ffee, 4, 1, 128, 8);
    check_attend(0xb10c_c0de, 20, 130, 512, 64);
}

/// MLA absorb + value kernels vs an f32 reference on the dequantized kv_b. kv_b is
/// fp8-e4m3 block-scaled: [H·(nope+vh) rows, kvl cols], block 128 on both axes.
#[test]
fn mla_fp8_matches_reference() {
    let mut r = Lcg(0x77);
    let (h, qh, nope, vh, kvl, block) = (4usize, 128usize, 128usize, 64usize, 256usize, 128usize);
    let rows = h * (nope + vh);
    let sc_cols = kvl / block;
    let packed: Vec<u8> = (0..rows * kvl).map(|_| f32_to_e4m3(r.f())).collect();
    let scale: Vec<f32> = (0..(rows / block).max(1) * sc_cols)
        .map(|_| (r.f() * 0.1).abs() + 0.01)
        .collect();
    // reference reads the exact dequant the kernels see: kvb[row][i] = e4m3·block-scale.
    let sc_rows = rows.div_ceil(block);
    let kvbf = |row: usize, i: usize| -> f32 {
        e4m3_to_f32(packed[row * kvl + i])
            * scale[(row / block).min(sc_rows - 1) * sc_cols + i / block]
    };
    let q: Vec<f32> = (0..h * qh).map(|_| r.f()).collect();
    let clat: Vec<f32> = (0..h * kvl).map(|_| r.f()).collect();

    // absorb: qabs[head][i] = Σ_d q[head·qh+d]·kvb[rbase+d][i]
    let mut want_abs = vec![0.0f32; h * kvl];
    for head in 0..h {
        let rbase = head * (nope + vh);
        for i in 0..kvl {
            let mut acc = 0.0f32;
            for d in 0..nope {
                acc += q[head * qh + d] * kvbf(rbase + d, i);
            }
            want_abs[head * kvl + i] = acc;
        }
    }
    // value: ctx[head][j] = Σ_i clat[head][i]·kvb[rbase+nope+j][i]
    let mut want_val = vec![0.0f32; h * vh];
    for head in 0..h {
        let rbase = head * (nope + vh);
        for j in 0..vh {
            let mut acc = 0.0f32;
            for i in 0..kvl {
                acc += clat[head * kvl + i] * kvbf(rbase + nope + j, i);
            }
            want_val[head * vh + j] = acc;
        }
    }

    let (kb, sb) = (dev(&packed), dev(&f32b(&scale)));
    let (qb, clb) = (dev(&f32b(&q)), dev(&f32b(&clat)));
    let mut absb = dev(&vec![0u8; h * kvl * 4]);
    let mut valb = dev(&vec![0u8; h * vh * 4]);
    unsafe {
        launch_mla_absorb_fp8(
            qb.ptr() as *const f32,
            kb.ptr(),
            sb.ptr() as *const f32,
            h,
            qh,
            nope,
            vh,
            kvl,
            block,
            absb.ptr_mut() as *mut f32,
        )
        .expect("launch absorb");
        launch_mla_value_fp8(
            clb.ptr() as *const f32,
            kb.ptr(),
            sb.ptr() as *const f32,
            h,
            nope,
            vh,
            kvl,
            block,
            valb.ptr_mut() as *mut f32,
        )
        .expect("launch value");
    }
    device_sync().expect("sync");
    assert_close(
        &want_abs,
        &f32v(&absb.copy_out().expect("out")),
        "mla_absorb",
    );
    assert_close(
        &want_val,
        &f32v(&valb.copy_out().expect("out")),
        "mla_value",
    );
}

/// Fused VQ MoE (moe_gateup_vq → moe_down_vq → moe_reduce), 3 per-projection
/// codebooks, vs a matvec_vq+silu reference on the same quantized bytes.
#[test]
fn moe_vq_matches_reference() {
    use rivoli::quant::{vq_expert_layout, vq_groups, vq_row_bytes};
    let mut r = Lcg(0x33);
    let (hidden, inter, e) = (128usize, 64usize, 3usize); // multi-group hidden, one-group inter
    // 3 codebooks
    let cbs: [Vec<f32>; 3] = std::array::from_fn(|_| (0..VQ_K * VQ_DIM).map(|_| r.f()).collect());
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..e).map(|_| r.f()).collect();

    // per expert per projection: quant to (indices, scales) against the right codebook
    let dims = vq_expert_layout(hidden, inter); // [(gate o,i),(up o,i),(down o,i)]
    let mut enc: Vec<[(Vec<u8>, Vec<u16>); 3]> = Vec::new();
    for _ in 0..e {
        let mut per = std::array::from_fn(|_| (Vec::new(), Vec::new()));
        for (p, &(o_dim, i_dim)) in dims.iter().enumerate() {
            let wv: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
            per[p] = quant_vq(&wv, o_dim, i_dim, &cbs[p]);
        }
        enc.push(per);
    }

    // reference: Σ_e w[e]·down(silu(gate·x)⊙up·x)
    let mut want = vec![0.0f32; hidden];
    for ex in 0..e {
        let mut g = vec![0.0f32; inter];
        let mut u = vec![0.0f32; inter];
        matvec_vq(
            &mut g,
            &x,
            &enc[ex][0].0,
            &enc[ex][0].1,
            &cbs[0],
            inter,
            hidden,
        );
        matvec_vq(
            &mut u,
            &x,
            &enc[ex][1].0,
            &enc[ex][1].1,
            &cbs[1],
            inter,
            hidden,
        );
        let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
        let mut down = vec![0.0f32; hidden];
        matvec_vq(
            &mut down,
            &h,
            &enc[ex][2].0,
            &enc[ex][2].1,
            &cbs[2],
            hidden,
            inter,
        );
        for o in 0..hidden {
            want[o] += w[ex] * down[o];
        }
    }

    // device: hold bufs alive; descriptors point into them
    let mut bufs: Vec<DeviceBuf> = Vec::new();
    let push = |b: Vec<u8>, bufs: &mut Vec<DeviceBuf>| -> *const u8 {
        bufs.push(dev(&b));
        bufs.last().expect("just pushed").ptr()
    };
    let descs: Vec<ExpertDesc> = (0..e)
        .map(|ex| ExpertDesc {
            gate_indices: push(enc[ex][0].0.clone(), &mut bufs),
            gate_scales: push(u16b(&enc[ex][0].1), &mut bufs) as *const u16,
            up_indices: push(enc[ex][1].0.clone(), &mut bufs),
            up_scales: push(u16b(&enc[ex][1].1), &mut bufs) as *const u16,
            down_indices: push(enc[ex][2].0.clone(), &mut bufs),
            down_scales: push(u16b(&enc[ex][2].1), &mut bufs) as *const u16,
        })
        .collect();
    let descb = dev(unsafe {
        std::slice::from_raw_parts(
            descs.as_ptr() as *const u8,
            std::mem::size_of_val(&descs[..]),
        )
    });
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let (g0, g1, g2) = (
        dev(&f16b(&cbs[0])),
        dev(&f16b(&cbs[1])),
        dev(&f16b(&cbs[2])),
    );
    let mut hbuf = dev(&vec![0u8; e * inter * 4]);
    let mut pbuf = dev(&vec![0u8; e * hidden * 4]);
    let mut obuf = dev(&vec![0u8; hidden * 4]);
    let _ = (vq_groups(hidden), vq_row_bytes(hidden)); // (layout used by quant/vq_expert)
    // The production path: each expert computes its own partial via a single-expert
    // range on a compute stream (exercising e_start indexing), then a fixed-order
    // reduce — same as the async expert stream, minus the load overlap.
    let stream = HipStream::new().expect("stream");
    unsafe {
        for k in 0..e {
            launch_moe_expert_range(
                xb.ptr() as *const f32,
                hidden,
                inter,
                k,
                1,
                descb.ptr() as *const ExpertDesc,
                g0.ptr() as *const u16,
                g1.ptr() as *const u16,
                g2.ptr() as *const u16,
                wb.ptr() as *const f32,
                hbuf.ptr_mut() as *mut f32,
                pbuf.ptr_mut() as *mut f32,
                stream.raw(),
            )
            .expect("launch moe_expert_range");
        }
        launch_moe_reduce(
            pbuf.ptr() as *const f32,
            e,
            hidden,
            obuf.ptr_mut() as *mut f32,
            stream.raw(),
        )
        .expect("launch moe_reduce");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&obuf.copy_out().expect("out")), "moe_vq");
}

/// int4 MoE (moe_gateup_i4 → moe_down_i4 → moe_reduce), per-row scale, vs a
/// matvec_i4+silu reference on the same quantized bytes. hidden ≥ 256 exercises
/// dot_i4_wave's dword fast path; inter < 256 its scalar tail.
#[test]
fn moe_i4_matches_reference() {
    use rivoli::quant::{matvec_i4, quant_i4};
    let mut r = Lcg(0x14);
    let (hidden, inter, e) = (256usize, 128usize, 3usize);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let w: Vec<f32> = (0..e).map(|_| r.f()).collect();
    let dims = [(inter, hidden), (inter, hidden), (hidden, inter)]; // gate, up, down (o,i)
    let mut enc: Vec<[(Vec<u8>, Vec<f32>); 3]> = Vec::new();
    for _ in 0..e {
        let mut per: [(Vec<u8>, Vec<f32>); 3] = std::array::from_fn(|_| (Vec::new(), Vec::new()));
        for (p, &(o_dim, i_dim)) in dims.iter().enumerate() {
            let wv: Vec<f32> = (0..o_dim * i_dim).map(|_| r.f()).collect();
            per[p] = quant_i4(&wv, o_dim, i_dim);
        }
        enc.push(per);
    }

    // reference: Σ_e w[e]·down(silu(gate·x)⊙up·x)
    let mut want = vec![0.0f32; hidden];
    for ex in 0..e {
        let mut g = vec![0.0f32; inter];
        let mut u = vec![0.0f32; inter];
        matvec_i4(&mut g, &x, &enc[ex][0].0, &enc[ex][0].1, inter, hidden);
        matvec_i4(&mut u, &x, &enc[ex][1].0, &enc[ex][1].1, inter, hidden);
        let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
        let mut down = vec![0.0f32; hidden];
        matvec_i4(&mut down, &h, &enc[ex][2].0, &enc[ex][2].1, hidden, inter);
        for o in 0..hidden {
            want[o] += w[ex] * down[o];
        }
    }

    let mut bufs: Vec<DeviceBuf> = Vec::new();
    let push = |b: Vec<u8>, bufs: &mut Vec<DeviceBuf>| -> *const u8 {
        bufs.push(dev(&b));
        bufs.last().expect("just pushed").ptr()
    };
    // One ExpertDesc for both formats: the int4 kernel reads the six pointers as
    // packed weights + f32 row-scale (the scale ptr is typed *const u16 but only its
    // ADDRESS matters — cast the f32 buffer ptr through it).
    let descs: Vec<ExpertDesc> = (0..e)
        .map(|ex| ExpertDesc {
            gate_indices: push(enc[ex][0].0.clone(), &mut bufs),
            gate_scales: push(f32b(&enc[ex][0].1), &mut bufs) as *const u16,
            up_indices: push(enc[ex][1].0.clone(), &mut bufs),
            up_scales: push(f32b(&enc[ex][1].1), &mut bufs) as *const u16,
            down_indices: push(enc[ex][2].0.clone(), &mut bufs),
            down_scales: push(f32b(&enc[ex][2].1), &mut bufs) as *const u16,
        })
        .collect();
    let descb = dev(unsafe {
        std::slice::from_raw_parts(descs.as_ptr() as *const u8, std::mem::size_of_val(&descs[..]))
    });
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&w)));
    let mut hbuf = dev(&vec![0u8; e * inter * 4]);
    let mut pbuf = dev(&vec![0u8; e * hidden * 4]);
    let mut obuf = dev(&vec![0u8; hidden * 4]);
    let stream = HipStream::new().expect("stream");
    unsafe {
        for k in 0..e {
            launch_moe_expert_range_i4(
                xb.ptr() as *const f32,
                hidden,
                inter,
                k,
                1,
                descb.ptr() as *const ExpertDesc,
                wb.ptr() as *const f32,
                hbuf.ptr_mut() as *mut f32,
                pbuf.ptr_mut() as *mut f32,
                stream.raw(),
            )
            .expect("launch moe_expert_range_i4");
        }
        launch_moe_reduce(
            pbuf.ptr() as *const f32,
            e,
            hidden,
            obuf.ptr_mut() as *mut f32,
            stream.raw(),
        )
        .expect("launch moe_reduce");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&obuf.copy_out().expect("out")), "moe_i4");
}

/// GPU int4 MoE on REAL colibri `.i4` bytes in the actual slot layout
/// (`i4_slot_offsets`) vs `matvec_i4` on the same bytes. The gap neither
/// `moe_i4_matches_reference` (synthetic `quant_i4`, separate buffers) nor the host
/// probe (CPU only) covers. Skips if the artifact is absent.
#[test]
fn moe_i4_real_data_matches_cpu() {
    use rivoli::quant::{
        i4_expert_bytes, i4_row_bytes, i4_slot_offsets, matvec_i4, vq_expert_layout,
    };
    use std::os::unix::fs::FileExt;
    let path = "/var/db/rivoli/glm52-vq3-full/L03.i4";
    let Ok(f) = std::fs::File::open(path) else {
        eprintln!("skip moe_i4_real_data: {path} absent");
        return;
    };
    let (hidden, inter) = (6144usize, 2048usize);
    let mut blk = vec![0u8; i4_expert_bytes(hidden, inter)];
    f.read_exact_at(&mut blk, 0).expect("read expert 0"); // routed expert 0, layer 3 (block 0)
    let off = i4_slot_offsets(hidden, inter);
    let dims = vq_expert_layout(hidden, inter); // [(gate o,i),(up),(down)]

    let mut r = Lcg(0x99);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let proj = |k: usize| -> (Vec<u8>, Vec<f32>) {
        let (o, i) = dims[k];
        let p = blk[off[k * 2]..off[k * 2] + o * i4_row_bytes(i)].to_vec();
        let sc = blk[off[k * 2 + 1]..off[k * 2 + 1] + o * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (p, sc)
    };
    let (gp, gs) = proj(0);
    let (up, us) = proj(1);
    let (dp, ds) = proj(2);
    let mut g = vec![0f32; inter];
    matvec_i4(&mut g, &x, &gp, &gs, inter, hidden);
    let mut u = vec![0f32; inter];
    matvec_i4(&mut u, &x, &up, &us, inter, hidden);
    let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
    let mut want = vec![0f32; hidden];
    matvec_i4(&mut want, &h, &dp, &ds, hidden, inter);

    let slot = dev(&blk);
    let base = slot.ptr();
    let desc = ExpertDesc {
        gate_indices: unsafe { base.add(off[0]) },
        gate_scales: unsafe { base.add(off[1]) } as *const u16,
        up_indices: unsafe { base.add(off[2]) },
        up_scales: unsafe { base.add(off[3]) } as *const u16,
        down_indices: unsafe { base.add(off[4]) },
        down_scales: unsafe { base.add(off[5]) } as *const u16,
    };
    let descb = dev(unsafe {
        std::slice::from_raw_parts(
            (&desc as *const ExpertDesc) as *const u8,
            std::mem::size_of::<ExpertDesc>(),
        )
    });
    let (xb, wb) = (dev(&f32b(&x)), dev(&f32b(&[1.0f32])));
    let mut hbuf = dev(&vec![0u8; inter * 4]);
    let mut pbuf = dev(&vec![0u8; hidden * 4]);
    let mut obuf = dev(&vec![0u8; hidden * 4]);
    let stream = HipStream::new().expect("stream");
    unsafe {
        launch_moe_expert_range_i4(
            xb.ptr() as *const f32,
            hidden,
            inter,
            0,
            1,
            descb.ptr() as *const ExpertDesc,
            wb.ptr() as *const f32,
            hbuf.ptr_mut() as *mut f32,
            pbuf.ptr_mut() as *mut f32,
            stream.raw(),
        )
        .expect("launch i4");
        launch_moe_reduce(pbuf.ptr() as *const f32, 1, hidden, obuf.ptr_mut() as *mut f32, stream.raw())
            .expect("reduce");
    }
    device_sync().expect("sync");
    let got = f32v(&obuf.copy_out().expect("out"));
    let dot: f64 = want.iter().zip(&got).map(|(a, b)| *a as f64 * *b as f64).sum();
    let (na, nb): (f64, f64) = (
        want.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt(),
        got.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt(),
    );
    eprintln!(
        "moe_i4_real: cosine(GPU,CPU)={:.4} want[0..3]={:?} got[0..3]={:?}",
        dot / (na * nb + 1e-30),
        &want[..3],
        &got[..3]
    );
    assert_close(&want, &got, "moe_i4_real");
}

/// `index_topk` vs the host selection it replaces, on the shapes that actually occur.
///
/// The oracle is the engine's own two lines — `topk_into(scores, k, &mut sel)` then
/// `sel.sort_unstable()` — not a reimplementation, so this pins the kernel to what the
/// attend has always consumed rather than to my reading of it.
///
/// **The row buffer is sentinel-filled and the tail is asserted untouched.** Without
/// that, over-selection is invisible: the readback would be truncated to `want.len()`,
/// and whenever the correct answer is an index *prefix* — which it is for the
/// `ReLU-sparse` case, since its non-zero scores sit at indices 0..300 — a kernel that
/// emitted every tied row would still match on the first k. Measured against a serial
/// simulation of this kernel: of three mutations (drop the cross-chunk tie carry; use
/// `<= need` for the tie budget; drop the -0.0 canonicalisation), the tie carry is
/// caught by seven cases on selection alone, the canonicalisation only by
/// `mixed +0.0/-0.0`, and **`<= need` by nothing except the tail check**.
///
/// The `ReLU-sparse` and `scattered zeros` cases are the engine's real shape:
/// `index_score` ReLUs every head contribution, so most tokens score exactly 0.0 and
/// the k-th boundary falls INSIDE that tie group, leaving the index-ascending rule to
/// decide the bulk of the selection. `scattered zeros` additionally makes the answer
/// non-prefix, which is the combination nothing else here covers. nt = 5185 and
/// k = 2048 are the longer in-engine context and GLM-5.2's `index_topk`.
#[test]
fn index_topk_matches_host_selection() {
    const SENTINEL: u32 = 0xFFFF_FFFF;
    fn host(scores: &[f32], k: usize) -> Vec<u32> {
        let mut sel = Vec::new();
        rivoli::math::topk_into(scores, k, &mut sel);
        sel.sort_unstable();
        sel.iter().map(|&i| i as u32).collect()
    }
    let mut rng = Lcg(0x7071_C0DE);
    let nt = 5185usize;
    // Realistic shape, answer is a prefix: real scores at the front, rest ReLU'd to 0.0.
    let mut relu_sparse = vec![0.0f32; nt];
    for (i, x) in relu_sparse.iter_mut().enumerate().take(300) {
        *x = (300 - i) as f32 * 0.25;
    }
    // Realistic shape, answer is NOT a prefix: the same sparsity, scattered.
    let mut scattered = vec![0.0f32; nt];
    for j in 0..300 {
        scattered[(j * 7919) % nt] = (300 - j) as f32 * 0.25;
    }
    let dense: Vec<f32> = (0..nt).map(|_| rng.f() * 8.0).collect();
    let heavy_ties: Vec<f32> = (0..nt).map(|_| (rng.f() * 4.0).floor()).collect();
    let cases: Vec<(&str, Vec<f32>, usize)> = vec![
        (
            "mixed +0.0/-0.0",
            (0..4096)
                .map(|i| if i % 3 == 0 { -0.0 } else { 0.0 })
                .collect(),
            2048,
        ),
        (
            "negatives only",
            (0..4096).map(|i| -((i % 11) as f32)).collect(),
            2048,
        ),
        ("k == nt", (0..2048).map(|i| (i % 7) as f32).collect(), 2048),
        ("k == nt - 1", (0..2049).map(|i| (i % 7) as f32).collect(), 2048),
        ("k > nt (wrapper clamp)", (0..500).map(|i| (i % 7) as f32).collect(), 2048),
        ("single block", (0..200).map(|i| (i % 5) as f32).collect(), 64),
        ("ReLU-sparse (engine shape, prefix answer)", relu_sparse, 2048),
        ("scattered zeros (engine shape, non-prefix answer)", scattered, 2048),
        ("dense random", dense, 2048),
        ("heavy ties", heavy_ties, 2048),
    ];
    for (name, scores, k) in cases {
        let n = scores.len();
        let want = host(&scores, k);
        let written = k.min(n);
        assert_eq!(
            want.len(),
            written,
            "oracle wrote {} rows, expected {written} on {name}",
            want.len()
        );
        let sb = dev(&f32b(&scores));
        // Sentinel fill: anything the kernel writes past `written` shows up below.
        let slots = n.max(k);
        let mut rb = dev(&vec![0xFFu8; slots * 4]);
        // SAFETY: scores holds n f32; rows holds >= min(k,n) u32.
        unsafe {
            launch_index_topk(sb.ptr() as *const f32, n, k, rb.ptr_mut() as *mut u32)
                .expect("index_topk");
        }
        device_sync().expect("sync");
        let raw = rb.copy_out().expect("rows out");
        let got: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(
            &got[..written],
            &want[..],
            "index_topk selection differs on {name}"
        );
        assert!(
            got[written..].iter().all(|&v| v == SENTINEL),
            "index_topk wrote past min(k,nt)={written} on {name} — over-selection"
        );
        assert!(
            got[..written].windows(2).all(|w| w[0] < w[1]),
            "index_topk output not strictly ascending on {name}"
        );
    }
}
