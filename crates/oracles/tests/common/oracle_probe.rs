//! Oracle-probe helpers for `v4_oracle.rs` — the V4-checkpoint-free slice of the old
//! tests/common tail (its Checkpoint/LayerKind-coupled half stays deferred to M8).

use rivoli_oracles::v4oracle::forward::{Capture, LayerCtx, LayerW, Oracle};
use rivoli_oracles::v4oracle::numerics::{bf16_decode, bf16_encode};
use rivoli_oracles::v4oracle::weights::{NamedRng, V4Config};

/// The layer a probe drives: its weights and its index, together — the pair every probe
/// call threads and a bare `(usize, &LayerW)` invites swapping against other indices.
pub struct ProbeLayer<'a> {
    pub idx: usize,
    pub w: &'a LayerW,
}

pub fn prefill_capture(o: &Oracle, at: ProbeLayer<'_>, ids: &[u32], h: &mut Vec<f32>) -> Capture {
    let ProbeLayer { idx: layer, w: lw } = at;
    let mut st = o.fresh_state(layer);
    let mut cap = Capture::default();
    let step = LayerCtx {
        lw,
        layer,
        s: ids.len(),
        start_pos: 0,
        input_ids: ids,
        step_tag: "pre",
    };
    o.run_layer(&step, &mut st, h, &mut cap);
    cap
}

/// A deterministic RESIDUAL-STREAM block, `[s, hc_mult * dim]`, seeded by `tag`.
///
/// [`probe`] with the one row width that is not arbitrary. `hc_mult * dim` is what the mHC
/// residual is, and it was spelled at three call sites in two files under two different
/// treatments — `v4_oracle`'s `fixed_h` wrapped it and argued the wrapper was worth it, while
/// `f4_kernel` inlined the identical product twice with no comment. jscpd sees none of that
/// (each site was a single expression, far under its default `minLines: 5`), which makes it a
/// "known, not merely unseen" case rather than a licence to leave it.
///
/// Fixed per `tag` so a defect at prefill cannot change a later step's INPUT: only the
/// layer's own cached state carries a defect forward, which is what makes "this case is
/// unaffected" a statement about the defect rather than about propagation.
pub fn residual_probe(cfg: &V4Config, tag: &str, s: usize) -> Vec<f32> {
    probe(tag, s, cfg.hc_mult * cfg.dim)
}

/// A deterministic bf16 activation block, `[n, dim]`, seeded by `name`.
///
/// **Changing the draw or the `NamedRng` sequence re-bases goldens in five suites at once** —
/// `v4_oracle`, `f4_kernel`, `blockindex_kernel`, `kvcompress_kernel` and
/// `kvcompress_probe`. `v4_oracle` and `f4_kernel` reach it only through
/// [`residual_probe`], so neither file can see its own exposure from its own source.
///
/// This doc line was orphaned onto `indexer_w` until 2026-08-06, which is how a shared
/// fixture source ended up with nothing at its definition saying what it is shared by.
pub fn probe(name: &str, n: usize, dim: usize) -> Vec<f32> {
    let mut r = NamedRng::new(name);
    (0..n * dim)
        .map(|_| bf16_decode(bf16_encode(r.unit())))
        .collect()
}
