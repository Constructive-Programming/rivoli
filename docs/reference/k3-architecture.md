---
scope: k3
status: live
verdict: Kimi-K3's forward pass, extracted from the C reference at ff11dce (2026-08-07) and precise enough to write kernels from. 93 layers: 69 KDA (linear attention, delta rule, per-key-channel decay, short causal conv k=4 with fused SiLU) + 24 gated MLA at one-based 4,8,...,88,92,93 — note 92 and 93 are ADJACENT, breaking the every-fourth pattern. NoPE throughout: the 64 rope dims exist, are cached, are SCORED, and are never rotated. Block Attention Residuals are a multi-residual-stream scheme snapshotting at zero-based layers 0,12,...,84 and mixing <=9 sources by softmax twice per layer plus once model-level. MoE routes on FULL hidden width, then down-projects 7168->3584, runs top-16 of 896 MXFP4 experts in latent space, RMSNorms the AGGREGATE, up-projects back, and adds ONE fused shared MLP (intermediate 6144, bf16, trunk-side) unweighted — though that fusing is a load-time transform, not necessarily the on-disk layout. Router bias steers selection only; weights come from the UNBIASED sigmoid, renormalised over the 16. Trunk is bf16 at 108.81 GB; 113.49 GB is trunk plus embed and lm_head, a conflation k3.h:14 is explicitly headed to prevent. Twelve named order-of-operations traps, each of which runs cleanly and produces a wrong model.
---

# Kimi-K3 — the forward pass

Extracted 2026-08-09 from the C reference `github.com/FareedKhan-dev/kimi-k3-in-c` pinned at
**`ff11dce858a2eb8a781224facdffd33a1fa48d25`** (2026-08-07, "Release v1.0.0: verified end to
end"), cross-checked against the shipped `config.json`.

This is written to be implementable, not readable. `docs/investigations/k3-port.md` is the
plan; this is the specification it builds against.

**Two separate caveats, and only one of them is now discharged.**

*Transcription:* this was first extracted through a summarizing fetch layer, which is the wrong
thing to have between a source and a spec. **It has since been re-verified against the raw
pinned source** (downloaded and read directly, not summarized): every quoted C block matches
real code, and six defects found in that pass are fixed here. Where a block is abridged it is
by elision only — `k3_attn_res`'s `score[]`/`z` declarations and its max-subtracted softmax are
elided behind a comment, and various `(size_t)` casts and guards are dropped. **No block below
is a compilable transcript; each is faithful to the arithmetic.**

*Single-sourcing:* not discharged. Everything here is what **one** C reimplementation asserts.
It claims per-layer conformance against the released model
(`tests/fixtures/gates/conform_all_93.log`) but that claim is not independently verified here —
see the plan's G1b, and its item 10 (the first-party tensor index), which is the cheap
structural check that this extraction is *complete*.

## 1. Shapes

```c
hidden 7168 · n_layers 93 · vocab 163840 · rms_eps 1e-5 · no tied embeddings
KDA:  kda_heads 96, kda_head_dim 128 (d_k == d_v), conv_k 4, gate_lb -5.0
MLA:  n_heads 96, q_lora 1536, kv_lora 512, qk_nope 128, qk_rope 64, v_head 128,
      mla_out_gate 1
MoE:  n_experts 896, topk 16, n_shared 2, latent 3584, moe_inter 3072,
      routed_scale 1.0, moe_renorm 1, latent_norm 1
dense: first_dense 1, dense_inter 33792
AttnRes: attn_res_block 12          SiTU: situ_b1 4.0, situ_b2 25.0
```

## 2. The layer map — explicit, not inferred

`linear_attn_config` carries an **explicit `full_attn_layers` array** of 24 one-based indices.
The reference reads only that one and derives KDA as its complement (`k3_is_kda` is
`!k3_is_mla`), giving 69. A `kda_layers` array also appears in the config, but nothing in the
reference consumes it — **do not rely on it**; derive the complement and assert 69 and 24.

**The one-based indexing is the off-by-one that matters**: the reference calls it out as the
mistake that "silently swaps KDA and MLA layers".

```
full_attn_layers (ONE-BASED):
  4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93
```

```c
int k3_is_mla(const K3Cfg *c, int layer) {
    for (int i = 0; i < c->n_full_attn; i++)
        if (c->full_attn[i] == layer + 1) return 1;
    return 0;
}
int k3_is_kda(const K3Cfg *c, int layer)   { return !k3_is_mla(c, layer); }
int k3_is_dense(const K3Cfg *c, int layer) { return layer < c->first_dense; }
```

**Zero-based MLA layers are 3, 7, 11, …, 87, 91, 92 — 91 and 92 are ADJACENT.** The
every-fourth pattern breaks at the end. Assert the count and the last two explicitly.

**Layer 0 is simultaneously** a KDA layer, the dense-FFN layer, and an AttnRes block boundary.
It is the least representative layer in the model and the one everyone tests first.

## 3. Block Attention Residuals (AttnRes)

A multi-residual-stream scheme, not a plain residual. The network keeps a **stack of
snapshots** of the residual taken at block boundaries plus a **running prefix sum**, and forms
each module's input as a softmax-weighted mixture over that stack.

**Boundaries** fire at zero-based `layer_idx % 12 == 0` → layers 0, 12, 24, 36, 48, 60, 72, 84
— **8 snapshots**. 93 is not a multiple of 12; the last block is 9 layers deep (84…92). That
asymmetry is real, and the reference never validates divisibility.

**Tensors**, per layer, all `[hidden]` fp32: `attn_res_norm`, `attn_res_proj`, `mlp_res_norm`,
`mlp_res_proj`. Plus **one model-level pair**, `output_attn_res_norm` / `output_attn_res_proj`.
`_proj` ships as `[1][hidden]` — a **single scoring vector**, not a matrix. Norm gain and
scoring vector collapse: `fold[i] = norm[i] * proj[i]`, foldable at load time.

```c
void k3_attn_res(float *out, const float *src, const float *fold,
                 int nsrc, int n, float eps)
{
    for (int s = 0; s < nsrc; s++) {
        const float *v = src + (size_t)s * n;
        double ss = 0.0;
        for (int i = 0; i < n; i++) ss += (double)v[i] * (double)v[i];
        const float inv = (float)(1.0 / sqrt(ss / (double)n + (double)eps));
        /* key is the NORMALISED source; fold already carries norm.weight*proj.weight */
        double acc = 0.0;
        for (int i = 0; i < n; i++) acc += (double)(v[i] * inv) * (double)fold[i];
        score[s] = (float)acc;
    }
    /* max-subtracted softmax over nsrc, then: */
    for (int i = 0; i < n; i++) out[i] = 0.0f;
    for (int s = 0; s < nsrc; s++) {
        const float p = (float)(score[s] / z);
        const float *v = src + (size_t)s * n;   /* the RAW source, not the key */
        for (int i = 0; i < n; i++) out[i] += p * v[i];
    }
}
```

`score_s = <RMSNorm(src_s), fold>`, `out = softmax(score) @ src_raw`. **The softmax mixes the
UNNORMALISED sources.** No temperature, no per-source matrix, no head split.

Cost is negligible (~24 M MAC/token total). The cost is **layout**: `block_residual` must stay
live across the whole layer loop, because the model-level aggregation at the end reads the
snapshot taken at layer 0. It is `[T][9][hidden]` fp32 = **T × 258 KB**.

**Size it at prefill, not at decode.** rivoli's layer-major prefill is the default and keeps
the whole prompt live, so all T are concurrent: 1.06 GB at a 4,096-token prompt, 2.11 GB at
8,192, 4.2 GB at 16k — against a residual budget the plan sizes at ~5.7 GB. Whether snapshots
are fp32 or bf16, and whether prefill is chunked to bound this, is a sizing decision the port
must take rather than inherit.

### The layer loop

```
prefix_sum = h
if block_residual NON-EMPTY:
    h = attn_res([blocks..., prefix_sum], attn_fold)    <- REPLACES h
if layer_idx % 12 == 0:
    push prefix_sum onto block_residual
    prefix_sum = NONE                                   <- the reset
h = input_layernorm(h)
h = attention(h)                                        <- KDA or MLA
prefix_sum = (NONE) ? h : prefix_sum + h
h = attn_res([blocks..., prefix_sum], mlp_fold)         <- UNCONDITIONAL, no empty guard
h = post_attention_layernorm(h)
h = moe(h) or dense_mlp(h)
prefix_sum = (NONE) ? h : prefix_sum + h
return prefix_sum
```

`h` on entry and exit **is** `prefix_sum`. The two `attn_res` calls overwrite `h` only to
produce the module *input*; the residual chain is `pref`, re-seeded from `h` each layer. Norms
apply to the **aggregated** `h`, never to `pref`.

## 4. KDA — 69 layers

One function serves prefill and decode; the only difference is `T`. **No chunked kernel exists
in the reference** — the released model has one (`fused_recurrent` and `chunked`, per
`tools/verify_kda.py`, matched at 2e-4), but porting it means porting from `fla`.

**Two correctness conditions belong to the chunked form and are absent here**, flagged by
`k3.h:51-60` under "NOT INVARIANTS OF THIS IMPLEMENTATION": the UT-transform inverse
`(I + A_kk)^-1`, and the retention of `A_qk`'s diagonal but **not** `A_kk`'s. Anyone adding a
chunked KDA path must reinstate both **together with gating fixtures**. Since a chunked prefill
is the obvious throughput win, this is the highest-value paragraph in the reference's headers
for this port.

### Weights

```c
const void  *q, *k, *v;                /* [H*D][hidden] each              */
const float *q_conv, *k_conv, *v_conv; /* [H*D][conv_k] depthwise, fp32   */
const void  *f_a, *f_b;                /* [D][hidden], [H*D][D]           */
const float *A_log;                    /* [H] PER HEAD (tensor is D long) */
const float *dt_bias;                  /* [H*D] per (head, channel)       */
const void  *b;                        /* [H][hidden] -> per-head scalar  */
const void  *g;                        /* [H*D][hidden] full-rank gate    */
const float *o_norm;                   /* [D] shared across heads         */
const void  *o;                        /* [hidden][H*D]                   */
```

### Order of operations

1. **Projections** from the normed layer input `x`: q, k, v `[H*D]`; `beta_pre` `[H]`;
   and `z = f_b(f_a(x))` — **one shared rank-128 pair feeds all 96 heads**.
2. **ShortConv, k=4, depthwise over time, SiLU FUSED into the output.** Applied to q, k, v
   separately, each with its own weights. Taps are oldest→newest; `w[k-1]` multiplies the
   current token. History holds **pre-conv, pre-SiLU** inputs.
   ```c
   float acc = w[c*k + hist] * cur;
   for (int j = 0; j < hist; j++) acc += w[c*k + j] * buf[j];
   y[t*channels + c] = acc * sigmoidf_(acc);   /* SiLU, fused */
   ```
3. **L2Norm on q and k ONLY**, per head over D=128. `v` is left alone. `eps = 1e-6` added to
   the **sum** of squares (not the mean), double accumulator — a different convention from
   `k3_rmsnorm`, in the same function.
4. **beta** `= sigmoid(b_proj(x))`, per-head scalar.
5. **Decay:**
   ```c
   const float a = expf(A_log[h]);          /* PER HEAD */
   const float u  = a * (z[i] + dt_bias[i]);/* bias BEFORE the scale */
   const float gi = lb * sigmoidf_(u);      /* lb = -5.0, so gi in (lb, 0]  */
   alpha[i] = expf(gi);                     /* in (e^lb, 1] -- 1.0 is valid */
   ```
   `gate_lower_bound` **multiplies the sigmoid**; it is not a clamp. **The intervals are
   closed at the top** — the reference writes `(lb, 0]` and `(e^lb, 1]`, because in fp32
   sigmoid underflows to 0 below about -87.3, so **`alpha == 1.0` exactly is legitimate
   saturation** meaning perfect retention. A kernel asserting `alpha < 1.0` fires on valid
   input.
6. **q scaled by `d_k^-0.5`** after L2Norm, before the recurrence. `k` and `v` unscaled.
7. **The delta rule**, per head, `S[i][j]` with `i` = key channel, `j` = value channel:
   ```
   S[i][j] ← alpha[i] * S[i][j]                     decay rows by the per-key-channel gate
   u[j]     = Σ_i k[i] * S[i][j]                    u = Sᵀ k, from the ALREADY-DECAYED S
   S[i][j] ← S[i][j] + k[i] * beta * (v[j] - u[j])  rank-one; (v-u) is the prediction error
   o[j]     = Σ_i q[i] * S[i][j]                    read the UPDATED S
   ```
8. **Head-wise RMSNorm** with learnable `o_norm` `[D]`, shared across heads.
9. **Then** gate: `o *= sigmoid(g_proj(x))`, full-rank, from the layer input `x`.
10. **Then** `o_proj`. Order is norm → gate → project. (MLA gates *without* a norm — see §5.)

### State

| piece | shape | dtype | bytes |
|---|---|---|---|
| `S` | `[96][128][128]` | fp32 | 6,291,456 |
| conv history, q\|k\|v | 3 × `[12288][3]` | fp32 | 442,368 |

**6,733,824 B per layer.** The reference allocates for all 93 layers (626 MB) and wastes the
MLA slabs; KDA-only need is 69 × 6.73 MB = **464.6 MB**. Zeroed once per sequence, never reset
between decode steps.

Heads are independent and the reference asserts head-parallel is bit-identical to serial. On
gfx1151 each head's `S` is 64 KB, so it will not sit in LDS alongside anything else — expect
it resident with four streaming passes. **Fusing those four passes is the obvious HIP win and
is not what the C does.**

## 5. Gated MLA — 24 layers, NoPE

```c
const int qh  = qn + qr;              /* 192: FULL head width  */
const int kvw = c->kv_lora + qr;      /* 576: latent + rope slot */
const int kvd = qn + vh;              /* 256: cached per head  */
const float scale = 1.0f / sqrtf((float)qh);   /* over 192, NOT over qk_nope */
```

- `q_a → q_a_norm → q_b`. **One** projection `kv_a` emits the compressed latent **and** the
  shared rope slot; `kv_a_norm` covers **the latent only**, never the rope slot.
- **NoPE, proven:** there is no cos/sin, no position term, and no `rope_theta` anywhere in the
  engine. The 64 rope dims are cached and **still scored**:
  ```c
  for (int i = 0; i < qn; i++) d += (double)qt[i] * (double)ks[i];
  /* the rope slot is UNROTATED but still scored, and the SAME 64
   * values serve every head. Dropping this term is the silent bug. */
  for (int i = 0; i < qr; i++) d += (double)qt[qn + i] * (double)kr[i];
  ```
  The shared 64-dim key slot is **one head broadcast to all 96**.
- Causality is unconditional (`s <= p`).
- **Output gate before `o_proj`, with NO norm** — the opposite of KDA's norm-then-gate:
  ```c
  k3_mmw(gbuf, x + t*E, w->g, w->wdt, E, H * vh);
  for (int i = 0; i < H * vh; i++) acc[i] *= 1.0f / (1.0f + expf(-gbuf[i]));
  k3_mmw(out + t*E, acc, w->o, w->wdt, H * vh, E);
  ```
- The KV cache stores the **expanded** per-head k/v (96 × 256 floats per position per layer =
  2.37 MB/pos across 24 layers), not the 576-float latent — deliberate, because re-expanding
  through `kv_b` per cached position is far slower.

## 6. MoE

Documented order, and every step matters:

```
1. route on the FULL hidden width, BEFORE any projection
2. down-project 7168 -> 3584
3. run the selected experts IN LATENT SPACE and sum, weighted
4. RMSNorm the AGGREGATE, never per expert
5. up-project 3584 -> 7168
6. add the shared expert computed on the ORIGINAL full-width input,
   with NO routing weight and NO scaling
```

```c
k3_router(idx, wt, xt, w->gate, w->bias, E, c->n_experts, c->topk,
          c->moe_renorm, c->routed_scale);
k3_mmw(z, xt, w->down, w->wdt, E, L);                 /* 7168 -> 3584 */
for (j...) {                                          /* top-16, in latent space */
    k3_matmul_mxfp4(gu,     z, q.p1, q.s1, L, I, K3_MXFP4_GROUP);
    k3_matmul_mxfp4(gu + I, z, q.p3, q.s3, L, I, K3_MXFP4_GROUP);
    k3_situ_glu(act, gu, I, c->situ_b1, c->situ_b2);
    k3_matmul_mxfp4(edn, act, q.p2, q.s2, I, L, K3_MXFP4_GROUP);
    for (int i = 0; i < L; i++) accL[i] += wt[j] * edn[i];
}
if (c->latent_norm) k3_rmsnorm(accL, accL, w->latent_norm, L, c->rms_eps);
k3_mmw(ot, accL, w->up, w->wdt, L, E);                /* 3584 -> 7168 */
/* shared expert, on xt (FULL width), added UNWEIGHTED, AFTER the up-projection */
k3_mmw(sgu,      xt, w->sh1, w->wdt, E, SI);          /* SI = 3072*2 = 6144 */
k3_mmw(sgu + SI, xt, w->sh3, w->wdt, E, SI);
k3_situ_glu(sact, sgu, SI, c->situ_b1, c->situ_b2);
k3_mmw(sdn, sact, w->sh2, w->wdt, SI, E);
for (int i = 0; i < E; i++) ot[i] += sdn[i];
```

**The two "shared experts" are ONE fused wider MLP**, `sh1`/`sh3` `[6144][7168]`, `sh2`
`[7168][6144]`, bf16, trunk-side. Not MXFP4 and not in the routed-expert cache — but it does
stream with the trunk, so "resident" is the wrong word for it.

**The fusing is a load-time transform, not an architectural fact.** Because the down
projection sums over the intermediate axis and SiTU-GLU is elementwise, two `[3072]` shared
experts concatenated into one `[6144]` intermediate is *exactly equivalent* to summing two
MLPs. So this tells you what the reference does internally, **not what the checkpoint ships**.
The converter must match the checkpoint — accept both layouts and fuse if it finds two.

Routed expert shapes per expert: `w1`/`w3` `[3072][3584]`, `w2` `[3584][3072]`, MXFP4.
Per expert `3 × 3584 × 3072 = 33,030,144` params → **17,547,264 B** at 0.53125 B/weight.

Trunk-side: `routed_expert_down_proj [3584][7168]`, `routed_expert_up_proj [7168][3584]`,
`routed_expert_norm [3584]`.

### Router

```c
score[e]  = sigmoid(<W_e, x>);                    /* independent, do NOT sum to 1 */
choice[e] = score[e] + bias[e];                   /* bias affects SELECTION ONLY */
/* top-k on choice, then: */
w[j] = score[best];                               /* the UNBIASED score */
/* renorm over the 16 selected only: */
w[j] *= 1.0 / (Σ_selected w + 1e-20);
w[j] *= routed_scale;                             /* 1.0 today; keep the multiply */
```

No softmax. No grouped routing. Ties break first-index-first; outputs come back in descending
score order.

## 7. Layer 0, final norm, head

Layer 0 is dense: `dense_inter 33792`, **SiTU-GLU with the same betas**, tensors
`mlp.{gate,up,down}_proj`.

After the 93 layers there is **a third, model-level AttnRes aggregation** over the 8 snapshots
plus the final prefix (9 sources) — skipping it is silent. Then `RMSNorm(·, model.norm.weight)`
and `lm_head [163840][7168]`. No bias, no logit scaling, no tied embeddings. Embedding lookup
is a plain bf16→fp32 gather with **no scale factor**.

## 8. SiTU-GLU

```c
const float a = b1 * tanhf(g / b1) * sigmoidf_(g);   /* b1 = 4  */
const float u = b2 * tanhf(up[i] / b2);              /* b2 = 25 */
y[i] = a * u;
```

The sigmoid takes the **uncapped** gate. `|y| ≤ b1*b2 = 100`. Layout is `[gate | up]`
concatenated.

## 9. MXFP4

```c
const unsigned char nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
orow[i] = K3_E2M1[nib] * mult;
```

**Low nibble is the even element** — matches rivoli's `.f4`. Scale is
`ldexpf(1.0f, (int)sb - 127)`, group 32.

**The accuracy contract, which is the load-bearing tolerance fact in this file.**
`k3_matmul_mxfp4` uses **8 accumulators per 32-element group** and applies the group scale
**before** accumulating (`acc += sub * K3_E8M0[sb]`). It is **deliberately NOT bit-identical**
to dequantise-then-matmul; the reference requires agreement to **1e-6** and gates it in
`test_expert.c`. Unchecked preconditions: `group <= 64`, `in` even, `scales` is
`rows × ceil(in/group)`.

**`sb == 255` maps to ZERO in the reference.** rivoli's `e8m0f` returns a quiet NaN and
`quant.rs::e8m0` *bails*. See the plan's S1a — this must be settled against K3's real scale
tensors before conversion.

## 10. Twelve traps

Each runs cleanly and produces a wrong model.

1. **`A_log` is per HEAD (96), `dt_bias` is per channel (12288).** The `A_log` tensor is 128
   long and only the first 96 entries matter. `A_log[h*D+d]` compiles and is wrong.
2. **Recurrence order**: decay → read `u` → delta write → read `o` from the **updated** state.
   Any permutation runs.
3. **`a*(z + dt_bias)`**, not `a*z + dt_bias`.
4. **`gate_lb` multiplies the sigmoid**; it is not a clamp or an additive floor.
5. **SiLU is fused into the conv output** (`acc * sigmoid(acc)`), not applied before the conv.
6. **L2Norm is after conv+SiLU, on q and k only**, `eps` on the **sum** of squares, `1e-6` —
   while `k3_rmsnorm` divides by `n` first and uses `1e-5`.
7. **`q * d_k^-0.5` after L2Norm**, not folded into weights, not applied to `k`.
8. **MLA's softmax scale is `192^-0.5`** (the full head width), not `qk_nope^-0.5`.
9. **MLA's unrotated 64 rope dims are still scored.** Dropping the term is the silent bug.
10. **MLA gates without a norm; KDA norms then gates.** Same word, opposite order.
11. **The router's combining weights come from the UNBIASED sigmoid**, renormalised over the
    16 selected. Using `choice` is wrong.
12. **The MoE RMSNorms the aggregate, not each expert**, and the shared expert adds **after**
    the up-projection at full width, unweighted.

Reference-side numerics for anyone chasing a tolerance: the C accumulates every long reduction
in **double**, while the recurrence and the conv are fp32 throughout. But only **two** of them
use the 16-wide pairwise tree — `k3_matmul` and `k3_matmul_bf16`. `l2norm_`, `k3_rmsnorm`,
`k3_attn_res` and `k3_router` are plain **sequential** double loops. A HIP port
accumulating projections in fp32 will not match bitwise; whether that matters is a tolerance
question the goldens settle.
