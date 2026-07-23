//! GPU kernels vs their CPU oracles in quant.rs. Compiles to nothing without rocm.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli::device::DeviceBuf;
use rivoli::hip::{ExpertDescVq, device_sync, launch_gemv_fp8, launch_gemv_vq, launch_moe_vq};
use rivoli::math::{f32_to_e4m3, silu};
use rivoli::quant::{VQ_DIM, VQ_K, matvec_fp8, matvec_vq, quant_vq};

fn f32b(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u16b(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
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
    assert!(
        err <= 1e-3 * mx + 1e-3,
        "{label}: err={err:.3e} max={mx:.3e}"
    );
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }
}

#[test]
fn gemv_fp8_matches_oracle() {
    // block-scaled fp8 GEMV vs matvec_fp8. Dims a multiple of block on both axes.
    let mut r = Lcg(0xF8);
    let (o_dim, i_dim, block) = (256usize, 512usize, 128usize);
    // build fp8 packed + block scales, then the exact dequant the oracle sees.
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
    assert_close(&want, &f32v(&yb.copy_out().expect("out")), "gemv_fp8");
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
        dev(&f32b(&codebook)),
    );
    let mut yb = dev(&vec![0u8; o_dim * 4]);
    unsafe {
        launch_gemv_vq(
            xb.ptr() as *const f32,
            ib.ptr(),
            sb.ptr() as *const u16,
            cb.ptr() as *const f32,
            o_dim,
            i_dim,
            yb.ptr_mut() as *mut f32,
        )
        .expect("launch");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&yb.copy_out().expect("out")), "gemv_vq");
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
    let mut push = |b: Vec<u8>, bufs: &mut Vec<DeviceBuf>| -> *const u8 {
        bufs.push(dev(&b));
        bufs.last().unwrap().ptr()
    };
    let descs: Vec<ExpertDescVq> = (0..e)
        .map(|ex| ExpertDescVq {
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
        dev(&f32b(&cbs[0])),
        dev(&f32b(&cbs[1])),
        dev(&f32b(&cbs[2])),
    );
    let mut hbuf = dev(&vec![0u8; e * inter * 4]);
    let mut pbuf = dev(&vec![0u8; e * hidden * 4]);
    let mut obuf = dev(&vec![0u8; hidden * 4]);
    let _ = (vq_groups(hidden), vq_row_bytes(hidden)); // (layout used by quant/vq_expert)
    unsafe {
        launch_moe_vq(
            xb.ptr() as *const f32,
            hidden,
            inter,
            e,
            descb.ptr() as *const ExpertDescVq,
            g0.ptr() as *const f32,
            g1.ptr() as *const f32,
            g2.ptr() as *const f32,
            wb.ptr() as *const f32,
            hbuf.ptr_mut() as *mut f32,
            pbuf.ptr_mut() as *mut f32,
            obuf.ptr_mut() as *mut f32,
        )
        .expect("launch moe_vq");
    }
    device_sync().expect("sync");
    assert_close(&want, &f32v(&obuf.copy_out().expect("out")), "moe_vq");
}
