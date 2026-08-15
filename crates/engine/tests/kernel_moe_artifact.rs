//! The int4 MoE path on the SHIPPED artifact: `.i4` bytes in their real slot layout, scored
//! against `matvec_i4` on the same bytes and against the fp8 checkpoint they were derived
//! from, with the artifact's own provenance stamp gating both.
//!
//! **Split out of `kernel_moe.rs` on 2026-08-15 — by COHESION, not by size alone.** The
//! `kernel.rs` these came from had reached 2263 lines and 79 functions over nine kernel
//! families and scored 8.03 on CodeScene's "Low Cohesion" and function-count rules; the MoE
//! group it left in was still over the same cliff on its own. The cut here is the one the
//! tests already make for themselves: **everything in this file `return`s early when
//! `/var/db/rivoli/glm52-vq3-full` or the fp8 checkpoint is absent**, and nothing in
//! `kernel_moe.rs` does — those fixtures are synthetic and run wherever there is a GPU. Two
//! files, two preconditions, one skip line each.
//!
//! The dispatch chain both halves drive — `MoeIo`, `MoeCtx`, `Drain`, `expert_range_i4`,
//! `drain`, `desc_buf`, `moe_bufs`, `Dims` and `i4_launch_drain` — moved to `common` with the
//! split, because a second copy of it is exactly what `build.rs`'s duplication gate refuses.
//!
//! Every body below travelled VERBATIM with its comments; in this repo a comment carries the
//! measurement that justified the choice, so a re-worded one loses evidence.
#![cfg(feature = "rocm")]
#![allow(clippy::expect_used)]

use rivoli_artifact::format::{FormatMeta, I4Source, Safetensors};
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::quant::{
    I4_GROUP, RowScaledW, i4_expert_bytes, i4_groups, i4_row_bytes, i4_slot_offsets, matvec_i4,
    vq_expert_layout,
};
use rivoli_backend::hip::ExpertDesc;
use rivoli_core::num::silu;
use std::os::unix::fs::FileExt;

mod common;
use common::moe::{Dims, i4_launch_drain};
use common::{Lcg, assert_close, dev};

/// Run ONE int4 expert block through the real GPU path — `moe_gateup_i4` →
/// `moe_down_i4` → `moe_acc_drain`, weight 1.0 — and return `down(silu(gate·x) ⊙ up·x)`.
/// `blk` is an expert's on-disk bytes; `off` its `i4_slot_offsets`. Shared by the two
/// real-data tests so they exercise byte-for-byte the same launch, and a change to the
/// descriptor layout cannot be fixed in one test and left stale in the other.
fn gpu_i4_expert(blk: &[u8], off: &[usize; 6], x: &[f32], d: Dims) -> Vec<f32> {
    let slot = dev(blk);
    let base = slot.ptr();
    // SAFETY: every offset lies within `blk`, which `slot` holds device-resident.
    let desc = ExpertDesc {
        gate_indices: unsafe { base.add(off[0]) },
        gate_scales: unsafe { base.add(off[1]) } as *const u16,
        up_indices: unsafe { base.add(off[2]) },
        up_scales: unsafe { base.add(off[3]) } as *const u16,
        down_indices: unsafe { base.add(off[4]) },
        down_scales: unsafe { base.add(off[5]) } as *const u16,
    };
    let out = i4_launch_drain(std::slice::from_ref(&desc), x, &[1.0f32], d);
    drop(slot); // the descriptor pointed into it; keep it alive across the launch above
    out
}

/// GPU int4 MoE on REAL colibri `.i4` bytes in the actual slot layout
/// (`i4_slot_offsets`) vs `matvec_i4` on the same bytes. The gap neither
/// `moe_i4_matches_reference` (synthetic `quant_i4`, separate buffers) nor the host
/// probe (CPU only) covers. Skips if the artifact is absent.
#[test]
fn moe_i4_real_data_matches_cpu() {
    let path = "/var/db/rivoli/glm52-vq3-full/L03.i4";
    let Ok(f) = std::fs::File::open(path) else {
        eprintln!("skip moe_i4_real_data: {path} absent");
        return;
    };
    let d = Dims::new(6144, 2048);
    let (hidden, inter) = (d.hidden, d.inter);
    let mut blk = vec![0u8; i4_expert_bytes(hidden, inter)];
    f.read_exact_at(&mut blk, 0).expect("read expert 0"); // routed expert 0, layer 3 (block 0)
    let off = i4_slot_offsets(hidden, inter);
    let dims = vq_expert_layout(hidden, inter); // [(gate o,i),(up),(down)]

    let mut r = Lcg(0x99);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let proj = |k: usize| -> (Vec<u8>, Vec<f32>) {
        let (o, i) = dims[k];
        let p = blk[off[k * 2]..off[k * 2] + o * i4_row_bytes(i)].to_vec();
        let sc = blk[off[k * 2 + 1]..off[k * 2 + 1] + o * i4_groups(i) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        (p, sc)
    };
    let (w, dm) = (proj(0), [inter, hidden]);
    let mut g = vec![0f32; inter];
    matvec_i4(&mut g, &x, RowScaledW::new(&w.0, &w.1), dm);
    let w = proj(1);
    let mut u = vec![0f32; inter];
    matvec_i4(&mut u, &x, RowScaledW::new(&w.0, &w.1), dm);
    let h: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
    let w = proj(2);
    let mut want = vec![0f32; hidden];
    matvec_i4(&mut want, &h, RowScaledW::new(&w.0, &w.1), [hidden, inter]);

    let got = gpu_i4_expert(&blk, &off, &x, d);
    let dot: f64 = want
        .iter()
        .zip(&got)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let n2 = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    eprintln!(
        "moe_i4_real: cosine(GPU,CPU)={:.4} want[0..3]={:?} got[0..3]={:?}",
        dot / (n2(&want) * n2(&got) + 1e-30),
        &want[..3],
        &got[..3]
    );
    assert_close(&want, &got, "moe_i4_real");
}

/// The artifact's own `i4_source` stamp, or `None` after a loud skip line.
///
/// The checkpoint comes from the artifact's OWN stamp, and the bands below only
/// describe the `fp8->int4` chain. The retired `vq3_to_i4` rewrote `L{l}.i4` IN PLACE
/// in this very directory with the other chain, and artifacts it produced are still
/// on disk; without this the test would keep running and quietly certify the
/// derivation it exists to distinguish.
fn i4_provenance(art: &str, layer: usize) -> Option<I4Source> {
    let Some(prov) = I4Source::load(art).expect("read i4_source") else {
        eprintln!("skip moe_i4_real_data_vs_fp8: artifact carries no i4_source stamp");
        return None;
    };
    assert_eq!(
        prov.chain, "fp8->int4",
        "the assertion bands characterise the fp8->int4 chain; this artifact is {}",
        prov.chain
    );
    // A set quantized at a different group size is a differently-shaped scale array,
    // not a slightly-different one: reading it walks the wrong strides and yields
    // rel_l2=NaN rather than a large error. Skip loudly instead of asserting on NaN,
    // which reports "systematic gain error" and sends the reader hunting a numerics
    // bug that is really a stale artifact.
    if prov.group != Some(I4_GROUP) {
        eprintln!(
            "skip moe_i4_real_data_vs_fp8: artifact is group {:?}, this build reads group {} \
             — rebuild with fp8_to_i4",
            prov.group, I4_GROUP
        );
        return None;
    }
    assert!(
        prov.layers[0] <= layer && layer < prov.layers[1],
        "layer {layer} outside the stamped range {:?}",
        prov.layers
    );
    Some(prov)
}

/// One expert's gate/up/down weights dequantized out of the fp8 checkpoint, in `dims` order.
fn fp8_expert_weights(
    src: &Safetensors,
    layer: usize,
    dims: &[(usize, usize)],
    block: usize,
) -> Vec<Vec<f32>> {
    let base = format!("model.layers.{layer}.mlp.experts.0");
    ["gate_proj", "up_proj", "down_proj"]
        .iter()
        .zip(dims)
        .map(|(p, &(o, i))| {
            src.dequant_fp8(&format!("{base}.{p}"), [o, i], block)
                .expect("dequant fp8")
        })
        .collect()
}

/// `(rel_l2, gain, max_err/max|ref|)` over a got/want pair, all in f64 — the three aggregate
/// statistics the bands below are stated in, computed in one pass over the same elements.
fn aggregate_scores(got: &[f32], want: &[f32]) -> (f64, f64, f64) {
    let (mut num, mut den, mut dot) = (0f64, 0f64, 0f64);
    for (&a, &b) in got.iter().zip(want) {
        let (a, b) = (a as f64, b as f64);
        num += (a - b) * (a - b);
        den += b * b;
        dot += a * b;
    }
    let (mx_ref, mx_err) = got.iter().zip(want).fold((0f64, 0f64), |(r, e), (&a, &b)| {
        (r.max(b.abs() as f64), e.max((a - b).abs() as f64))
    });
    ((num / den).sqrt(), dot / den, mx_err / mx_ref)
}

/// The int4 path against INDEPENDENT ground truth: the original fp8 checkpoint the
/// `.i4` set was derived from, dequantized and dotted in **f64** through the same
/// `down(silu(gate·x) ⊙ up·x)` chain. `moe_i4_real_data_matches_cpu` compares the GPU
/// kernel to our own `matvec_i4`; nothing in this reference touches `matvec_i4`,
/// `quant_i4`, or a nibble.
///
/// **What it catches, stated honestly.** Errors present in the DERIVATION — wrong
/// tensor, wrong dims, wrong `weight_scale_inv` tiling, an `.i4` set rebuilt from a
/// different checkpoint or through the old `fp8→vq3→int4` chain. Those move `gain`
/// toward 0 and `rel_l2` past 1. It does **not** catch a nibble order or zero point
/// that `quant_i4` and the kernel SHARE: `quant_i4` wrote these bytes, so a shared
/// convention cancels end to end. Nothing in this repo can pin that — both the writer
/// and the reader are ours; the anchor is colibri's `.qs` format, off-tree.
///
/// **It is also a COARSE gate, deliberately.** Two aggregate statistics over 6144
/// outputs cannot see corruption confined to a few percent of rows — a simulated 2%
/// K-tail truncation sits inside any band wide enough to hold the real spread. The
/// tight per-element gate is the sibling test's `assert_close` (max-abs at
/// `1e-3·max`), which catches that trivially. `max_err/max|ref|` here is the cheap
/// complement, not a substitute.
///
/// **The bands are tight because the measurement is deterministic.** `x` is a fixed
/// seed, the artifact is fixed, and the kernels reduce with a fixed `__shfl_down`
/// ladder — so this is not a sample from a distribution, it is one number. `bin/i4_audit`
/// (same generator, same `CHAIN_SEED`) reproduces it on the CPU to four decimals:
/// rel_l2 **0.2951**, gain **1.0009**, max_err/max|ref| **0.1603** for L3 expert 0.
/// Bands sit ~12% around those, which is roughly 3× more sensitive than a band sized
/// for x-draw spread would be — and rel-L2 through `silu` really does vary ~15% across
/// draws, which is exactly why quoting a same-distribution-but-different-seed anchor
/// (as an earlier revision did) is not good enough.
///
/// Retired 2026-08-05, tag `archive/i4-audit`; restore per `docs/investigations/int4-scales.md`
/// §8, then:
///
///     cargo run --bin i4_audit -- /var/db/rivoli/glm52-vq3-full \
///         /swarm/storage/ai/openclaw/glm52-fp8 --layer 3 --experts 0,7,256
#[test]
fn moe_i4_real_data_vs_fp8_ground_truth() {
    const ART: &str = "/var/db/rivoli/glm52-vq3-full";
    const LAYER: usize = 3; // first MoE layer; block 0 = routed expert 0
    let path = format!("{ART}/L{LAYER:02}.i4");
    let Ok(f) = std::fs::File::open(&path) else {
        eprintln!("skip moe_i4_real_data_vs_fp8: {path} absent");
        return;
    };
    let Some(prov) = i4_provenance(ART, LAYER) else {
        return;
    };
    let ckpt = prov.src.clone();
    let Ok(src) = Safetensors::open_dir(&prov.src) else {
        eprintln!("skip moe_i4_real_data_vs_fp8: checkpoint {ckpt} absent");
        return;
    };
    let block = FormatMeta::load(ART).expect("format meta").fp8_block;
    // From the manifest, not hardcoded: the slot offsets below are functions of these
    // dims, so a shape the constants disagreed with would read the wrong bytes and
    // still produce a plausible-looking number.
    let cfg = ModelConfig::load(ART).expect("artifact manifest");
    let d = Dims::new(cfg.hidden, cfg.moe_inter);
    let (hidden, inter) = (d.hidden, d.inter);
    let mut blk = vec![0u8; i4_expert_bytes(hidden, inter)];
    f.read_exact_at(&mut blk, 0).expect("read expert 0");
    let off = i4_slot_offsets(hidden, inter);
    let dims = vq_expert_layout(hidden, inter);

    // ── the reference: fp8 → f64, no quantized code in the path ─────────────────
    let wref = fp8_expert_weights(&src, LAYER, &dims, block);
    let mv64 = |w: &[f32], x: &[f32], o_dim: usize, i_dim: usize| -> Vec<f32> {
        (0..o_dim)
            .map(|o| {
                w[o * i_dim..(o + 1) * i_dim]
                    .iter()
                    .zip(x)
                    .map(|(&a, &b)| a as f64 * b as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    };
    let mut r = Lcg(0x5A17);
    let x: Vec<f32> = (0..hidden).map(|_| r.f()).collect();
    let g = mv64(&wref[0], &x, inter, hidden);
    let u = mv64(&wref[1], &x, inter, hidden);
    let hv: Vec<f32> = (0..inter).map(|j| silu(g[j]) * u[j]).collect();
    let want = mv64(&wref[2], &hv, hidden, inter);

    // ── what the shipped int4 path produces, on the GPU, from the real bytes ────
    let got = gpu_i4_expert(&blk, &off, &x, d);

    let (rel_l2, gain, rel_max) = aggregate_scores(&got, &want);
    // Margins, not just pass/fail — a run drifting from 0.24 toward 0.32 stays green
    // and unremarked right up until it crosses (the reason `assert_close` prints them).
    println!(
        "moe_i4_vs_fp8: rel_l2={rel_l2:.4} (band 0.14..0.20) gain={gain:.4} (band 1.01..1.09) \
         max_err/max|ref|={rel_max:.4} (bound 0.16)"
    );
    // These bands describe GROUP-128 (`quant::I4_GROUP`). The per-row format they replaced
    // anchored at rel_l2 0.2951 / gain 1.0009; group-128 measures 0.1698 / 1.0493 on the
    // same artifact coordinates and seed — a 42% error reduction, and the reason `--mode
    // int4` went from PPL 73.43 to 5.12. Changing `I4_GROUP` moves these ON PURPOSE:
    // re-anchor with the retired `bin/i4_audit` (see the doc comment above), do not widen.
    // (A mismatched artifact never reaches
    // here — the group gate above skips it, because reading one yields rel_l2=NaN and this
    // assertion would then blame a "systematic gain error" for a stale file.)
    assert!(
        (1.01..=1.09).contains(&gain),
        "SYSTEMATIC gain error vs fp8 ground truth: gain={gain:.4}. Cosine scores 1.0000 \
         for any uniform or per-group scale error, so nothing else here would see this. \
         Group-128 at an amax/7 step runs slightly hot (~1.05) because the step is set by \
         each group's own extreme; a value near 1.00 means the loading factor changed."
    );
    assert!(
        (0.14..=0.20).contains(&rel_l2),
        "rel_l2={rel_l2:.4} != the deterministic 0.1698 this artifact and seed produce. \
         Below 0.14 the reference has collapsed into the thing it checks (a 128-wide group \
         at an amax/7 step still loses ~0.12, ~0.17 through the silu chain); above 0.20 the \
         derivation is wrong, not merely coarse — and 0.26+ is the per-row format, i.e. an \
         artifact that predates group scales."
    );
    assert!(
        rel_max <= 0.16,
        "max_err/max|ref|={rel_max:.4} > 0.16 — a few corrupted output rows move this \
         while leaving rel_l2 inside its band"
    );
}
