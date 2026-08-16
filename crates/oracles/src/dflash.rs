//! **The DFlash block drafter's CPU forward — a value-scoring transliteration of
//! `modeling_muse_glimmer_assistant.py` at transformers `fe747d88`, the commit the vendored
//! draft goldens pin.**
//!
//! DFlash is the block-diffusion drafting SCHEME (one bidirectional forward denoises a
//! `block_size` window against a projected target context); the model that uses it here is
//! `meta-models/Muse-Glimmer-30B-assistant`, and that membership is data — the scheme name is
//! the behaviour name. `glimmer-architecture.md` §11 is the in-repo spec; the goldens are
//! `tests/glimmer-anchor-draft-{1,2}.bin`, whose shapes `tests/glimmer_anchor_draft.rs`
//! already gates. What THIS module adds is the arithmetic those shapes pass over in silence,
//! and `tests/glimmer_draft_oracle.rs` scores every captured tensor against it.
//!
//! The drafter shares almost nothing with its target, and each difference below is one a port
//! goes wrong by REUSING the target's path (the four are this oracle's plantable defects):
//!
//! * attention is **bidirectional** across the block, never causal ([`DraftDefect::CausalMask`]);
//! * Q spans the block alone while K/V span `context + block` in the same call, so RoPE's
//!   tables cover the full range and **Q takes the tail slice**
//!   ([`DraftDefect::RopeUntailed`]);
//! * the borrowed embedding is read **raw** — the target's weightless embed-norm is
//!   deliberately skipped, and the reference carries a comment saying the norm can never be
//!   folded into the matrix for exactly this reason ([`DraftDefect::EmbedNormApplied`]);
//! * the GQA group count is the drafter's own, **not** the target's
//!   ([`DraftDefect::TargetGrouping`]).
//!
//! Arithmetic is f64 throughout: the vendored captures are the reference's f32 run, whose
//! distance to its own f64 run is the measured floor the oracle's tolerances are 10x of, so an
//! f64 oracle sits inside the band by construction if and only if it computes the same thing.
//!
//! Frozen like `v4oracle`: written from the reference, changed only when the reference
//! changes, never called on a decode path.

use crate::torchdraw::{self, Family};

/// Everything the forward is shaped by, read from a golden's own `tiny_config` (or the real
/// checkpoint's config), never written as literals — a literal agrees with drift.
#[derive(Clone, Copy)]
pub struct DraftDims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub inter: usize,
    pub layers: usize,
    pub block: usize,
    /// The bidirectional sliding window — `sliding_attention` on every drafter layer.
    pub window: usize,
    /// `len(target_layer_ids)`: how many hidden-state slabs the encoder concatenates.
    pub targets: usize,
    pub eps: f64,
    pub rope_theta: f64,
}

impl DraftDims {
    /// The drafter's own GQA broadcast factor. The defect matrix proves that substituting the
    /// TARGET's here moves values, which is the half the vendored `assert_ne!` shape test
    /// cannot see.
    pub fn group(&self) -> usize {
        self.heads / self.kv_heads
    }

    fn kv_len(&self, ctx: usize) -> usize {
        ctx + self.block
    }
}

/// One drafter layer's parameters, in the reference's own storage layout
/// (`nn.Linear` weights are `[out, in]`).
pub struct LayerParams {
    pub q_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    /// Weighted per-head QK-norms — `[head_dim]`, a tensor the TARGET's weightless norms do
    /// not even ship. Defaulting these from the target is the S6 item-1 mistake.
    pub q_norm: Vec<f32>,
    pub k_norm: Vec<f32>,
    pub gate_proj: Vec<f32>,
    pub up_proj: Vec<f32>,
    pub down_proj: Vec<f32>,
    pub ln_in: Vec<f32>,
    pub ln_post: Vec<f32>,
}

/// The whole drafter: 11 tensors per layer plus the three globals. It owns no embedding and
/// no lm_head — the census in the scoring test asserts that absence rather than assuming it.
pub struct DraftParams {
    pub layers: Vec<LayerParams>,
    pub final_norm: Vec<f32>,
    pub enc_fc: Vec<f32>,
    pub enc_norm: Vec<f32>,
}

/// The fixture kit's family rule, specialised to the drafter: no Assistant module is a
/// centered norm (the type-based scan in `init_weights` finds none), so the rule collapses to
/// "the owner's last path component contains `norm`" — which is exactly `_draw_into`'s elif.
fn family_for(name: &str) -> Family {
    // Every drawn name is `<path>.<owner>.weight`, so the owner's last path component is
    // `rsplit('.').nth(1)`. `None` is the dotless case, which no parameter name reaches; it
    // falls to `Projection` exactly as the reference's else does.
    if name.rsplit('.').nth(1).is_some_and(|o| o.contains("norm")) {
        Family::Norm
    } else {
        Family::Projection
    }
}

/// Regenerate the drafter's parameters for one salt, under the reference's own
/// `named_parameters()` names (enumerated from the pinned venv 2026-08-16; the scoring test
/// pins them transitively, since a wrong name draws a different stream and no capture agrees).
///
/// The draft run seeds with `"{salt}/draft"` — `init_weights(draft, f"{salt}/draft")`.
pub fn draw_params(dims: &DraftDims, salt: &str) -> DraftParams {
    let d = dims;
    let draft_salt = format!("{salt}/draft");
    let tensor = |name: &str, n: usize| torchdraw::draw(&draft_salt, name, n, family_for(name));
    let layers = (0..d.layers)
        .map(|l| {
            let attn = |p: &str, n: usize| tensor(&format!("layers.{l}.self_attn.{p}.weight"), n);
            let mlp = |p: &str, n: usize| tensor(&format!("layers.{l}.mlp.{p}.weight"), n);
            LayerParams {
                q_proj: attn("q_proj", d.heads * d.head_dim * d.hidden),
                k_proj: attn("k_proj", d.kv_heads * d.head_dim * d.hidden),
                v_proj: attn("v_proj", d.kv_heads * d.head_dim * d.hidden),
                o_proj: attn("o_proj", d.hidden * d.heads * d.head_dim),
                q_norm: attn("q_norm", d.head_dim),
                k_norm: attn("k_norm", d.head_dim),
                gate_proj: mlp("gate_proj", d.inter * d.hidden),
                up_proj: mlp("up_proj", d.inter * d.hidden),
                down_proj: mlp("down_proj", d.hidden * d.inter),
                ln_in: tensor(&format!("layers.{l}.input_layernorm.weight"), d.hidden),
                ln_post: tensor(
                    &format!("layers.{l}.post_attention_layernorm.weight"),
                    d.hidden,
                ),
            }
        })
        .collect();
    DraftParams {
        layers,
        final_norm: tensor("norm.weight", d.hidden),
        enc_fc: tensor("encoder.fc.weight", d.hidden * d.targets * d.hidden),
        enc_norm: tensor("encoder.output_norm_enc.weight", d.hidden),
    }
}

/// The four ways a port reuses the target's path, each plantable so the scoring test can show
/// the comparison redden at the tensors the defect reaches AND hold at the ones it does not.
/// An oracle that disagrees everywhere proves nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DraftDefect {
    None,
    /// The block kept causal: the reference's own sliding-causal mask (strict lower window
    /// edge, `kv > q - w && kv <= q`) in place of the bidirectional `|q - kv| <= w`.
    CausalMask,
    /// RoPE's Q slice taken from the HEAD of the table (positions `0..block`) instead of the
    /// tail (`ctx..ctx+block`) — the spec's "off by `ctx_len` is a silent quality loss, not a
    /// crash": every shape survives it.
    RopeUntailed,
    /// The target's weightless embed-norm applied to the borrowed embedding rows. Plants in
    /// [`embed_block`] — the step it corrupts — and [`forward`] refuses it.
    EmbedNormApplied,
    /// K/V heads addressed through the TARGET's Q:KV ratio instead of the drafter's own.
    /// Carries the wrong ratio so the test can pass the target's actual number in.
    TargetGrouping {
        group: usize,
    },
    /// `output_norm_enc` skipped: `H_t = fc(concat)` fed to every layer unnormed. Mirrors the
    /// reference's own `defect_draft_context_unprojected` (the anchor kit proves the PYTHON
    /// comparison reddens on it); here it is the one defect that reaches `encoder.out`, whose
    /// red evidence no block-side trap can supply — Q never sees the context (§11 step 4).
    EncoderNormSkipped,
}

/// The forward's inputs, bundled: the two captured tensors the drafter consumes, and the
/// borrowed lm_head (the target's — the drafter has no vocabulary of its own).
pub struct DraftInputs<'a> {
    /// `draft.context_concat`, `[ctx][targets * hidden]`.
    pub ctx_concat: &'a [f32],
    /// `draft.noise_embeds`, `[block][hidden]` — see [`embed_block`] for how it is produced.
    pub noise: &'a [f32],
    /// `lm_head.weight`, `[vocab][hidden]`, from the target's vendored weight set.
    pub lm_head: &'a [f32],
    pub vocab: usize,
}

/// One layer's observable stages, named as the golden captures them.
pub struct LayerTrace {
    pub ln_in: Vec<f64>,
    /// `[heads][block][head_dim]` — post-norm, post-RoPE, as `eager_attention_forward` sees it.
    pub q: Vec<f64>,
    /// `[kv_heads][ctx + block][head_dim]`.
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    /// `[block][heads][head_dim]` — the reference transposes back before `o_proj`.
    pub attend_out: Vec<f64>,
    pub ln_post: Vec<f64>,
    pub mlp_out: Vec<f64>,
}

/// Every stage the golden captures, in one forward.
pub struct DraftTrace {
    pub encoder_out: Vec<f64>,
    /// `[block][ctx + block]`, 1.0 = attend — the golden stores the boolean form.
    ///
    /// **One mask, not one per layer.** The golden captures `draft.L{l}.attend.mask` five
    /// times, and the scoring test still compares all five — against this one value, which is
    /// the stronger claim: the reference builds the mask once outside its layer loop, so five
    /// captures that are not all equal to it is a defect either side would otherwise hide.
    pub mask: Vec<f32>,
    pub layers: Vec<LayerTrace>,
    pub final_norm: Vec<f64>,
    pub logits: Vec<f64>,
    /// Argmax of rows `1..block` — row 0 is the anchor and is sliced off (§11 step 6).
    pub candidates: Vec<i64>,
}

/// `weight * (x * rsqrt(mean(x^2) + eps))`, per `width`-row — the Assistant RMSNorm.
fn rmsnorm(rows: &[f64], weight: &[f32], width: usize, eps: f64) -> Vec<f64> {
    // `zip` below would SILENTLY emit `min(width, weight.len())` values per row, i.e. a short
    // vector for a transposed width rather than a failure at the call that caused it.
    assert_eq!(weight.len(), width, "rmsnorm weight is not the row width");
    rows.chunks(width)
        .flat_map(|row| {
            let mean = row.iter().map(|x| x * x).sum::<f64>() / width as f64;
            let scale = 1.0 / (mean + eps).sqrt();
            row.iter()
                .zip(weight)
                .map(move |(x, w)| f64::from(*w) * (x * scale))
        })
        .collect()
}

/// `x @ W^T` for an `nn.Linear` weight `[out][in]`, rows of `x` in `[.., in]`.
///
/// `out_dim` is asserted, not consumed: `w.chunks(in_dim)` already yields exactly the weight's
/// row count, so a `.take(out_dim)` would only TRUNCATE a wrong-sized weight into a short
/// output that surfaces later as a shape mismatch inside a named capture. Stating the expected
/// row count at each of the nine call sites is worth keeping — as a check, which is what it now
/// is (it was the silent `take` until 2026-08-16).
fn linear(x: &[f64], w: &[f32], out_dim: usize, in_dim: usize) -> Vec<f64> {
    assert_eq!(w.len(), out_dim * in_dim, "linear: weight is not [out][in]");
    x.chunks(in_dim)
        .flat_map(|row| {
            w.chunks(in_dim).map(|wrow| {
                row.iter()
                    .zip(wrow)
                    .map(|(a, b)| a * f64::from(*b))
                    .sum::<f64>()
            })
        })
        .collect()
}

/// The rotary tables over the FULL `kv_len` range: `cos/sin[pos][j]` with the half-and-half
/// `emb = cat(freqs, freqs)` layout. Built once per forward, exactly as `rotary_emb` computes
/// them from `arange(ctx + block)` — Q's tail slice is taken by the caller.
///
/// **The one place this oracle is NOT f64, and it has to be.** The reference pins its rotary
/// path to float32 no matter what dtype the model runs in: `inv_freq` is built with
/// `torch.arange(..., dtype=torch.float)` (`modeling_muse_glimmer_assistant.py:352`), and
/// `forward` re-casts with `.float()` inside `maybe_autocast(enabled=False)  # Force float32`
/// (:358-363). So the `--dtype float64` run that MEASURED the tolerance floors used the same
/// f32 tables as the f32 run, and the floor has exactly zero contribution from this table —
/// it cannot price a table computed any other way. Found by review 2026-08-16 and confirmed by
/// reading the pinned reference; an f64 table sat ~2e-7 relative from the reference's, which is
/// inside `attend.q`'s 3.51e-5 but is a deviation the floor was never evidence about, and one
/// that grows linearly with position (the fixture's largest angle is 15 rad; the real model's
/// are in the thousands).
struct Rope {
    cos: Vec<f64>,
    sin: Vec<f64>,
    dim: usize,
}

impl Rope {
    fn new(dims: &DraftDims, kv_len: usize) -> Self {
        let d = dims.head_dim;
        let half = d / 2;
        // f32 throughout, in the reference's own order: `inv_freq = 1/(theta**(2f/dim))` once,
        // then `inv_freq @ position` — NOT `position / theta**(2f/dim)`, which rounds
        // differently.
        let inv_freq: Vec<f32> = (0..half)
            .map(|f| 1.0f32 / (dims.rope_theta as f32).powf(2.0 * f as f32 / d as f32))
            .collect();
        let mut cos = vec![0.0; kv_len * d];
        let mut sin = vec![0.0; kv_len * d];
        for p in 0..kv_len {
            for j in 0..d {
                let ang = p as f32 * inv_freq[j % half];
                cos[p * d + j] = f64::from(ang.cos());
                sin[p * d + j] = f64::from(ang.sin());
            }
        }
        Self { cos, sin, dim: d }
    }

    /// `x*cos + rotate_half(x)*sin` on one head-row sitting at absolute position `pos`.
    fn rotate(&self, row: &mut [f64], pos: usize) {
        let d = self.dim;
        let half = d / 2;
        let (c, s) = (&self.cos[pos * d..][..d], &self.sin[pos * d..][..d]);
        let x: Vec<f64> = row.to_vec();
        for j in 0..d {
            let rot = if j < half { -x[j + half] } else { x[j - half] };
            row[j] = x[j].mul_add(c[j], rot * s[j]);
        }
    }
}

/// The mask the reference builds for this CACHELESS call, and the quirk that makes it worth a
/// paragraph: `create_bidirectional_sliding_window_mask` indexes queries by their row in the
/// query tensor (`q_offset = 0` with no cache), while RoPE places the same rows at positions
/// `ctx..ctx+block`. So the vendored pattern is `|q_row - kv_idx| <= window` — q0 of the tiny
/// fixture attends kv 0..=4 and none of the block, both salts, all five layers (read off the
/// captures 2026-08-16). The oracle transliterates the reference AS PINNED; whether that
/// off-window indexing is desirable at real context lengths is a serving-path question the
/// cache answers, not this fixture.
fn mask(dims: &DraftDims, ctx: usize, defect: DraftDefect) -> Vec<f32> {
    let kv_len = dims.kv_len(ctx);
    let w = dims.window as i64;
    let allowed = |q: i64, kv: i64| match defect {
        // The reference's own sliding-causal pair: inclusive upper edge, STRICT lower —
        // `masking_utils.py`'s `sliding_window_overlay` (`kv > q - w`) AND causal (`kv <= q`).
        DraftDefect::CausalMask => kv <= q && kv > q - w,
        // Bidirectional: inclusive on BOTH sides (`abs(q - kv) <= w`) — the asymmetry against
        // the causal form is the reference's, not this file's.
        _ => (q - kv).abs() <= w,
    };
    (0..dims.block as i64)
        .flat_map(|q| (0..kv_len as i64).map(move |kv| f32::from(u8::from(allowed(q, kv)))))
        .collect()
}

/// Per-head attention over the concatenated K/V, `head_dim^-0.5` scaling, softmax in the
/// masked support only — additively masking with the dtype minimum and exponentiating, as the
/// reference does, zeroes the same entries.
///
/// Two free functions rather than a struct: everything a struct would have held is already in
/// [`LayerCtx`] or derivable from the slices, so `self` bought only `self.`.
fn attend(lc: &LayerCtx<'_>, tr: &LayerTrace) -> Vec<f64> {
    let d = lc.dims;
    let (hd, kv_len) = (d.head_dim, d.kv_len(lc.ctx));
    let scale = 1.0 / (hd as f64).sqrt();
    let mut out = vec![0.0; d.block * d.heads * hd];
    for h in 0..d.heads {
        // Q-heads per KV head: the drafter's own group, or the planted target ratio.
        let kv_h = h / lc.group;
        assert!(kv_h < d.kv_heads, "grouping walked off the KV heads");
        let (k, v) = (
            &tr.k[kv_h * kv_len * hd..][..kv_len * hd],
            &tr.v[kv_h * kv_len * hd..][..kv_len * hd],
        );
        for row in 0..d.block {
            let q = &tr.q[(h * d.block + row) * hd..][..hd];
            let probs = softmax_row(q, k, &lc.mask[row * kv_len..][..kv_len], scale);
            let o = &mut out[(row * d.heads + h) * hd..][..hd];
            for (j, p) in probs.iter().enumerate() {
                for (oi, vj) in o.iter_mut().zip(&v[j * hd..][..hd]) {
                    *oi += p * vj;
                }
            }
        }
    }
    out
}

/// One query row's softmax over its masked support. `head_dim` is `q.len()` and the KV length
/// is `mask.len()` — the slices ARE the two widths, so neither is worth a parameter that could
/// disagree with them.
fn softmax_row(q: &[f64], k: &[f64], mask: &[f32], scale: f64) -> Vec<f64> {
    let hd = q.len();
    let scores: Vec<f64> = (0..mask.len())
        .map(|j| {
            let kj = &k[j * hd..][..hd];
            q.iter().zip(kj).map(|(a, b)| a * b).sum::<f64>() * scale
        })
        .collect();
    let m = scores
        .iter()
        .zip(mask)
        .filter(|(_, ok)| **ok > 0.5)
        .map(|(s, _)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = scores
        .iter()
        .zip(mask)
        .map(|(s, ok)| if *ok > 0.5 { (s - m).exp() } else { 0.0 })
        .collect();
    let z: f64 = e.iter().sum();
    // Every row of every mask this model builds has support (the window always covers the
    // query's own diagonal band); a zero here is a harness bug, not a model state — and it
    // would divide the whole row to NaN, which the scoring test's got-side guard then reports
    // as a broken oracle three stages downstream instead of here.
    assert!(z > 0.0, "softmax over an empty support");
    e.into_iter().map(|x| x / z).collect()
}

fn widen(x: &[f32]) -> Vec<f64> {
    x.iter().copied().map(f64::from).collect()
}

/// One DFlash cycle's forward — `MuseGlimmerAssistantModel.forward` plus the borrowed-lm_head
/// logits the driver computes around it, with every captured stage recorded.
pub fn forward(
    dims: &DraftDims,
    p: &DraftParams,
    io: &DraftInputs<'_>,
    defect: DraftDefect,
) -> DraftTrace {
    assert!(
        defect != DraftDefect::EmbedNormApplied,
        "the embed-norm defect plants at embed_block, upstream of this forward"
    );
    let d = dims;
    let ctx = io.ctx_concat.len() / (d.targets * d.hidden);
    let kv_len = d.kv_len(ctx);

    // Encoder, once, shared by every layer: H_t = output_norm_enc(fc(concat)).
    let fc = linear(
        &widen(io.ctx_concat),
        &p.enc_fc,
        d.hidden,
        d.targets * d.hidden,
    );
    let encoder_out = match defect {
        DraftDefect::EncoderNormSkipped => fc,
        _ => rmsnorm(&fc, &p.enc_norm, d.hidden, d.eps),
    };

    let rope = Rope::new(d, kv_len);
    // Q's positions: the tail of the table — or its head, under the planted offset defect.
    let q_base = match defect {
        DraftDefect::RopeUntailed => 0,
        _ => ctx,
    };
    // The GQA broadcast factor, named as `DraftDims::group` and the sibling shape gate's
    // `Widths::group` name it — one quantity, one name along the whole path.
    let group = match defect {
        DraftDefect::TargetGrouping { group } => group,
        _ => d.group(),
    };

    // Once per forward, not once per layer: the mask is a function of `ctx` and the window
    // alone, and the reference builds it once outside the layer loop for exactly that reason.
    let block_mask = mask(d, ctx, defect);

    let mut x = widen(io.noise);
    let mut layers = Vec::with_capacity(d.layers);
    for lp in &p.layers {
        let tr = layer_forward(
            &LayerCtx {
                dims: d,
                ctx,
                q_base,
                group,
                mask: &block_mask,
            },
            lp,
            &encoder_out,
            &rope,
            &mut x,
        );
        layers.push(tr);
    }

    let final_norm = rmsnorm(&x, &p.final_norm, d.hidden, d.eps);
    let logits = linear(&final_norm, io.lm_head, io.vocab, d.hidden);
    let candidates = logits.chunks(io.vocab).skip(1).map(argmax_first).collect();
    DraftTrace {
        encoder_out,
        mask: block_mask,
        layers,
        final_norm,
        logits,
        candidates,
    }
}

/// What one layer's forward is parameterised by beyond its own weights.
struct LayerCtx<'a> {
    dims: &'a DraftDims,
    ctx: usize,
    q_base: usize,
    group: usize,
    /// The forward's one block mask, borrowed — built upstream, where its layer-independence
    /// shows, and kept on [`DraftTrace`] rather than copied into each [`LayerTrace`].
    mask: &'a [f32],
}

/// One decoder layer, mutating the residual stream in place and returning its trace.
fn layer_forward(
    lc: &LayerCtx<'_>,
    lp: &LayerParams,
    encoder_out: &[f64],
    rope: &Rope,
    x: &mut [f64],
) -> LayerTrace {
    let d = lc.dims;
    let (hd, kv_len) = (d.head_dim, d.kv_len(lc.ctx));
    let ln_in = rmsnorm(x, &lp.ln_in, d.hidden, d.eps);

    // K/V's source is `cat(context, normed block)` — the context bypasses Q, o_proj and the
    // FFN entirely, entering as extra K/V rows only (§11 step 4).
    let mut kv_src = encoder_out.to_vec();
    kv_src.extend_from_slice(&ln_in);

    let to_heads = |flat: Vec<f64>, heads: usize, rows: usize| -> Vec<f64> {
        // [rows][heads*hd] -> [heads][rows][hd], the transpose(1, 2) the reference applies.
        let mut out = vec![0.0; heads * rows * hd];
        for r in 0..rows {
            for h in 0..heads {
                let src = &flat[(r * heads + h) * hd..][..hd];
                out[(h * rows + r) * hd..][..hd].copy_from_slice(src);
            }
        }
        out
    };
    // K and V project the SAME concatenated source through the same shape — said once, so the
    // asymmetry that matters (Q projects the block alone) stays visible.
    let kv_proj = |w: &[f32]| {
        to_heads(
            linear(&kv_src, w, d.kv_heads * hd, d.hidden),
            d.kv_heads,
            kv_len,
        )
    };
    let mut q = to_heads(
        linear(&ln_in, &lp.q_proj, d.heads * hd, d.hidden),
        d.heads,
        d.block,
    );
    let mut k = kv_proj(&lp.k_proj);
    let v = kv_proj(&lp.v_proj);

    // Weighted QK-norm per head-row, THEN RoPE — the reference's order.
    q = rmsnorm(&q, &lp.q_norm, hd, d.eps);
    k = rmsnorm(&k, &lp.k_norm, hd, d.eps);
    for (i, row) in q.chunks_mut(hd).enumerate() {
        rope.rotate(row, lc.q_base + i % d.block);
    }
    for (i, row) in k.chunks_mut(hd).enumerate() {
        rope.rotate(row, i % kv_len);
    }

    let mut tr = LayerTrace {
        ln_in,
        q,
        k,
        v,
        attend_out: Vec::new(),
        ln_post: Vec::new(),
        mlp_out: Vec::new(),
    };
    tr.attend_out = attend(lc, &tr);

    let attn = linear(&tr.attend_out, &lp.o_proj, d.hidden, d.heads * hd);
    for (xi, ai) in x.iter_mut().zip(&attn) {
        *xi += ai;
    }
    tr.ln_post = rmsnorm(x, &lp.ln_post, d.hidden, d.eps);
    let gate = linear(&tr.ln_post, &lp.gate_proj, d.inter, d.hidden);
    let up = linear(&tr.ln_post, &lp.up_proj, d.inter, d.hidden);
    let act: Vec<f64> = gate
        .iter()
        .zip(&up)
        .map(|(g, u)| g / (1.0 + (-g).exp()) * u)
        .collect();
    tr.mlp_out = linear(&act, &lp.down_proj, d.hidden, d.inter);
    for (xi, mi) in x.iter_mut().zip(&tr.mlp_out) {
        *xi += mi;
    }
    tr
}

/// torch `argmax` semantics: the FIRST index of the maximum, so a tie cannot flap between the
/// oracle and the reference.
fn argmax_first(row: &[f64]) -> i64 {
    let mut best = 0;
    for (i, x) in row.iter().enumerate() {
        if *x > row[best] {
            best = i;
        }
    }
    best as i64
}

/// The draft block's embedding: `[anchor] + masks`, gathered RAW from the target's matrix —
/// bit-exact rows, because the reference reaches past the normed wrapper to `nn.Embedding` on
/// purpose (its comment: the norm cannot be merged into the matrix, DFlash needs it unnormed).
///
/// [`DraftDefect::EmbedNormApplied`] plants the honest wrong turn: the target's WEIGHTLESS
/// RMS norm (`with_scale=False`, eps = `rms_norm_eps`) applied on top, which is what a port
/// gets by embedding through the target's normal path.
pub fn embed_block(ids: &[i64], table: &[f32], dims: &DraftDims, defect: DraftDefect) -> Vec<f32> {
    let rows: Vec<f32> = ids
        .iter()
        .flat_map(|id| {
            let i = usize::try_from(*id).unwrap_or_else(|_| panic!("negative token id {id}"));
            table[i * dims.hidden..][..dims.hidden].iter().copied()
        })
        .collect();
    if defect != DraftDefect::EmbedNormApplied {
        return rows;
    }
    let ones = vec![1.0f32; dims.hidden];
    rmsnorm(&widen(&rows), &ones, dims.hidden, dims.eps)
        .into_iter()
        .map(|x| x as f32)
        .collect()
}
