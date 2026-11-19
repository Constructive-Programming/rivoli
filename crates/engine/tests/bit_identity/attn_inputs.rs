//! The attention rows' input builders — deviceless, so the CPU plant models
//! (`attn_plants.rs`) read the very cells the device rows launch.
//!
//! **Adversarial inputs.** Every seed draws `inputs::draw`'s sixteen binades; edge seeds put
//! its `EDGES` (±100, ±0, subnormals, bf16 ties) at the front; and every [`nasty`] seed (one in 8)
//! plants ±inf and NaN in the queries (−inf and NaN scores, a NaN running max), and, for
//! MLA, e4m3's NaN codes in the cache. NaN is compared as a class (`device.rs`).

use super::common::{Lcg, f32v, u16b};
use super::inputs::{draw, words32};
use rivoli_core::num::f32_to_bf16;

/// Every `NASTY`-th seed (from seed 7) carries non-finite inputs.
const NASTY: u64 = 8;

/// Whether `seed` is a nasty one.
pub fn nasty(seed: u64) -> bool {
    seed % NASTY == NASTY - 1
}

/// Plant `+inf`, `−inf` and NaN at three seeded positions of `v` past the `EDGES` slots, on
/// nasty seeds only.
pub fn plant_nonfinite(v: &mut [f32], (seed, stream): (u64, u64)) {
    if !nasty(seed) || v.len() <= 16 {
        return;
    }
    let mut r = Lcg(seed ^ (stream << 32) ^ 0x5eed);
    for x in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        let at = 16 + ((r.f() + 1.0) * 0.5 * (v.len() - 16) as f32) as usize;
        v[at.min(v.len() - 1)] = x;
    }
}

/// `n` seeded indices in `0..below` as f32 words (the cell's inputs are f32 buffers).
pub fn index_words(n: usize, below: usize, r: &mut Lcg) -> Vec<u32> {
    (0..n)
        .map(|_| (((r.f() + 1.0) * 0.5 * below as f32) as usize).min(below - 1) as u32)
        .collect()
}

/// One MLA cell's shape: `h` heads, `nr` attended rows, latent width `kvl` (rope 64), `sparse`
/// rows (an index list into a cache of `2·nr + 5` tokens) or the dense prefix, and whether
/// the caller hands the split scratch at all.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Mla {
    pub h: usize,
    pub nr: usize,
    pub kvl: usize,
    pub sparse: bool,
    pub scratch: bool,
}

pub const fn mla([h, nr, kvl]: [usize; 3], sparse: bool, scratch: bool) -> Mla {
    Mla {
        h,
        nr,
        kvl,
        sparse,
        scratch,
    }
}

pub const ROPE: usize = 64;

/// One MLA cell's inputs: `[qabs, qrope, lc8, lscale, rc, rows]` (rows empty when dense).
pub fn mla_inputs(s: Mla, seed: u64) -> Vec<Vec<f32>> {
    let tokens = if s.sparse { 2 * s.nr + 5 } else { s.nr };
    let nb = s.kvl / 128;
    let mut r = Lcg(seed.wrapping_mul(0x2545_f491_4f6c_dd1d));
    // e4m3 codes, NaN (0x7f/0xff) only on nasty seeds: one NaN latent per token would make
    // every score NaN and the row a NaN-class comparison of nothing.
    let lc8: Vec<u8> = (0..tokens * s.kvl)
        .map(|_| {
            let b = ((r.f() + 1.0) * 128.0) as u8;
            if b & 0x7f == 0x7f && !nasty(seed) {
                b - 1
            } else {
                b
            }
        })
        .collect();
    let mut qabs = draw(s.h * s.kvl, (seed, 1), 0.05, seed % 4 == 1);
    plant_nonfinite(&mut qabs, (seed, 1));
    let rc: Vec<u16> = draw(tokens * ROPE, (seed, 5), 1.0, true)
        .into_iter()
        .map(f32_to_bf16)
        .collect();
    let rows = if s.sparse {
        words32(&index_words(s.nr, tokens, &mut r))
    } else {
        Vec::new()
    };
    vec![
        qabs,
        draw(s.h * ROPE, (seed, 2), 0.05, false),
        f32v(&lc8),
        draw(tokens * nb, (seed, 3), 0.004, seed.is_multiple_of(4)),
        f32v(&u16b(&rc)),
        rows,
    ]
}
