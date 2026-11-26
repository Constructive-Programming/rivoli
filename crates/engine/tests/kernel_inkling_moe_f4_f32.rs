//! **Inkling-2's fp32-latent MXFP4 routed-expert range on the device** —
//! `launch_moe_expert_range_f4_situ_f32` scored against an f64 reference that keeps both the
//! between-pass `h` activation and the pass-2 `dv` output in fp32 (no K3 `rbf16` store).
//!
//! No C twin (owner direction 2026-11-24): validated through `ORACLE_VALIDATED` in
//! `crates/cli/tests/kernel_coverage.rs`.
//!
//! Device tests: `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
mod k3;

use common::{DeviceBuf, assert_rel, moe::desc_buf};
use k3::{Lcg, back, dev, f32b, ok, zeros};
use rivoli_artifact::quant::{F4_LUT, e8m0};
use rivoli_backend::hip::{ExpertDescF4, launch_moe_expert_range_f4_situ_f32};
use rivoli_oracles::inkling::softcap_swiglu;

const FIXED_SCALE: f32 = 1.0 / (1u64 << 44) as f32;
const BETAS: (f32, f32) = (4.0, 25.0);

struct F4Mat {
    packed: Vec<u8>,
    scales: Vec<u8>,
}

struct F4Case {
    experts: Vec<[F4Mat; 3]>,
    x: Vec<f32>,
    weights: Vec<f32>,
    hidden: usize,
    inter: usize,
}

fn draw_f4(r: &mut Lcg, rows: usize, cols: usize) -> F4Mat {
    let packed = (0..rows * (cols / 2))
        .map(|_| ((r.f() * 0.5 + 0.5) * 255.0) as u8)
        .collect();
    let scales = (0..rows * (cols / 32))
        .map(|_| 0x70u8 + (((r.f() * 0.5 + 0.5) * 10.99) as u8).min(10))
        .collect();
    F4Mat { packed, scales }
}

fn dot_f4_row(m: &F4Mat, row: usize, x: &[f32]) -> f32 {
    let cols = x.len();
    let (pbase, sbase) = (row * (cols / 2), row * (cols / 32));
    let sum: f64 = x
        .iter()
        .enumerate()
        .map(|(c, &xv)| {
            let nib = ((m.packed[pbase + (c >> 1)] >> (4 * (c & 1))) & 0xF) as usize;
            let s = e8m0(m.scales[sbase + (c >> 5)]).unwrap();
            f64::from(F4_LUT[nib]) * f64::from(s) * f64::from(xv)
        })
        .sum();
    sum as f32
}

fn fixed44_f32(v: f32) -> i64 {
    const CAP: f32 = (1u64 << 14) as f32;
    (f64::from(v.clamp(-CAP, CAP)) * ((1u64 << 44) as f64)).round() as i64
}

fn host_f4_f32(case: &F4Case) -> Vec<f32> {
    let mut acc = vec![0i64; case.hidden];
    for ([gate, up, down], &w) in case.experts.iter().zip(&case.weights) {
        if w == 0.0 {
            continue;
        }
        let mid: Vec<f32> = (0..case.inter)
            .map(|j| {
                let g = dot_f4_row(gate, j, &case.x);
                let u = dot_f4_row(up, j, &case.x);
                softcap_swiglu(g, u, BETAS.0, BETAS.1)
            })
            .collect();
        for (o, slot) in acc.iter_mut().enumerate() {
            let dv = dot_f4_row(down, o, &mid);
            *slot = slot.wrapping_add(fixed44_f32(dv * w));
        }
    }
    acc.into_iter().map(|v| v as f32 * FIXED_SCALE).collect()
}

fn upload_f4(experts: &[[F4Mat; 3]]) -> (Vec<DeviceBuf>, DeviceBuf) {
    let mut bufs = Vec::with_capacity(experts.len() * 6);
    let mut host = Vec::with_capacity(experts.len());
    for [gate, up, down] in experts {
        let base = bufs.len();
        for m in [gate, up, down] {
            bufs.push(dev(&m.packed));
            bufs.push(dev(&m.scales));
        }
        host.push(ExpertDescF4 {
            gate_packed: bufs[base].ptr(),
            gate_scale: bufs[base + 1].ptr(),
            up_packed: bufs[base + 2].ptr(),
            up_scale: bufs[base + 3].ptr(),
            down_packed: bufs[base + 4].ptr(),
            down_scale: bufs[base + 5].ptr(),
        });
    }
    (bufs, desc_buf(&host))
}

fn run_device(case: &F4Case) -> Vec<f32> {
    let (_keep, descs) = upload_f4(&case.experts);
    let xb = dev(&f32b(&case.x));
    let wb = dev(&f32b(&case.weights));
    let count = case.experts.len();
    let mut hb = zeros(count * case.inter * 4);
    let mut ab = zeros(case.hidden * 8);
    let stream = ok(rivoli_backend::Stream::compute(), "stream");
    // SAFETY: buffers are sized for `(count, hidden, inter)` and outlive `device_sync`.
    let rc = unsafe {
        launch_moe_expert_range_f4_situ_f32(
            xb.ptr() as *const f32,
            case.hidden,
            case.inter,
            0,
            count,
            count,
            descs.ptr() as *const ExpertDescF4,
            wb.ptr() as *const f32,
            BETAS.0,
            BETAS.1,
            hb.ptr_mut() as *mut f32,
            ab.ptr_mut() as *mut u64,
            1,
            stream.raw(),
        )
    };
    ok(rc, "moe_expert_range_f4_situ_f32");
    ok(rivoli_backend::device_sync(), "sync");
    back(&ab)
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32 * FIXED_SCALE)
        .collect()
}

#[test]
fn f4_situ_f32_matches_fp32_host_oracle_without_bf16_rounding() {
    let mut r = Lcg(0x44f32);
    for &(ne, hidden, inter) in &[(2usize, 512usize, 256usize), (3, 64, 32)] {
        let experts = (0..ne)
            .map(|_| {
                [
                    draw_f4(&mut r, inter, hidden),
                    draw_f4(&mut r, inter, hidden),
                    draw_f4(&mut r, hidden, inter),
                ]
            })
            .collect();
        let x = (0..hidden).map(|_| r.f()).collect();
        let mut weights: Vec<f32> = (0..ne).map(|i| 0.25 + 0.15 * i as f32).collect();
        if ne >= 3 {
            weights[1] = 0.0;
        }
        let case = F4Case {
            experts,
            x,
            weights,
            hidden,
            inter,
        };
        let got = run_device(&case);
        let want = host_f4_f32(&case);
        let tol = (inter + hidden) as f32 * f32::EPSILON;
        assert_rel(
            &want,
            &got,
            &format!("f4_f32 ne={ne} h={hidden} i={inter}"),
            tol,
        );
    }
}
