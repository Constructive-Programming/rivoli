//! What the three checkpoints CALL things: the projection-name triples, the per-model
//! expert prefixes, and the routed/shared boundary between them.
//!
//! **Split out of `quant.rs` on 2026-08-15.** It was the cleanest of the disconnected
//! components CodeScene's LCOM4 was pricing at 8.54 — not one of these functions shares a
//! call edge with a byte, a scale or a GEMV; they build strings that must match bytes on
//! someone else's disk. Bodies and comments travelled verbatim, and the comments are the
//! point: each records the checkpoint index it was read off and the date, because every
//! string here is a fact about a file this repo does not own.
//!
//! The one exception to "no call edge" is [`expert_projs`]/[`v4_expert_projs`], which zip a
//! name triple against [`super::vq_expert_layout`] — the ZIP is as order-bearing as the
//! list, which is why it lives beside the names rather than at each caller.
//!
//! Every public name is re-exported by `quant.rs`.

use super::vq_expert_layout;

/// The checkpoint tensor-name suffixes of those same three projections, in the SAME
/// order — every offline tool that walks an expert zips this against
/// [`vq_expert_layout`], so one index means one projection in both. It lives beside the
/// layout rather than in each `src/bin` (where it had been declared, identically, four
/// times) because a second copy of an ORDER-BEARING list is exactly the kind that goes
/// wrong silently: a reordered copy still compiles and still runs, and scores gate's
/// weights against up's.
pub const PROJ: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];

/// One expert's three projections: the checkpoint tensor suffix paired with the
/// `(o_dim, i_dim)` it has, in slot order. See [`expert_projs`].
pub type ExpertProjs = [(&'static str, (usize, usize)); 3];

/// [`PROJ`] already zipped against [`vq_expert_layout`] — the name/shape pairs an expert
/// encoder walks, in slot order. Here for the same reason `PROJ` is: the ZIP is as
/// order-bearing as the list, and `bin/convert` and `bin/fp8_to_i4` had spelled it out
/// identically (down to recomputing the layout once per expert inside the worker loop).
pub fn expert_projs(expert_in: usize, moe_inter: usize) -> ExpertProjs {
    let [g, u, d] = vq_expert_layout(expert_in, moe_inter);
    [(PROJ[0], g), (PROJ[1], u), (PROJ[2], d)]
}

/// The checkpoint tensor prefix of expert `e` in `layer`. Routed experts are numbered;
/// `e == n_experts` is the SHARED expert, which lives under an entirely different name.
///
/// Beside `PROJ` for the same reason: two tools (`convert`, `fp8_to_i4`) walk a layer's
/// `n_experts + 1` blocks in this order, and a copy that got the boundary wrong would
/// quantize the shared expert's weights into a routed slot — producing a file of exactly
/// the right size that every length check passes.
pub fn expert_base(layer: usize, e: usize, n_experts: usize) -> String {
    expert_slot_name(&format!("model.layers.{layer}.mlp"), e, n_experts)
}

/// `{trunk}.experts.{e}` — the ROUTED slot's name. All three checkpoints spell this half the
/// same way; the trunk stays at each caller precisely because a shared constant is how one
/// model's rename would retarget another model's converter, the argument [`K3_PROJ`] makes
/// against sharing [`V4_PROJ`]'s strings.
fn routed_expert_name(trunk: &str, e: usize) -> String {
    format!("{trunk}.experts.{e}")
}

/// [`routed_expert_name`], or `{trunk}.shared_experts` past the routed count. The BOUNDARY,
/// which is the half a copy gets wrong silently — and which is why K3 calls the routed half
/// directly: it has no shared block past the routed ones to cross into.
fn expert_slot_name(trunk: &str, e: usize, n_experts: usize) -> String {
    if e < n_experts {
        routed_expert_name(trunk, e)
    } else {
        format!("{trunk}.shared_experts")
    }
}

// --- DeepSeek-V4-Flash tensor naming ------------------------------------------------
//
// V4's checkpoint uses its reference implementation's names, NOT HuggingFace's: no
// `model.` prefix, `attn`/`ffn` rather than `self_attn`/`mlp`, `.scale` rather than
// `.weight_scale_inv`, and `w1`/`w3`/`w2` rather than `gate_proj`/`up_proj`/`down_proj`.
// Verified against the shipped `model.safetensors.index.json` (72,317 entries), 2026-08-04.

/// V4's three expert projections in the SAME slot order as [`PROJ`] — i.e. gate, up, down.
///
/// **The order is `w1, w3, w2` and that is not a typo.** `inference/model.py`'s
/// `Expert.forward` is `gate = self.w1(x)`, `up = self.w3(x)`, then `return self.w2(…)`,
/// so w3 is the UP projection and w2 is down. Storing them in gate/up/down order keeps one
/// slot index meaning one projection across all three of this engine's formats.
///
/// A `w2` in the wrong slot is caught by its shape (`[expert_in, moe_inter]`, transposed from
/// the other two). A `w1`/`w3` SWAP is not: they are the same shape, and a repack that
/// swapped them would be internally consistent and byte-clean. Only a numerical oracle
/// against the reference can see that, which is what S1b exists for.
///
/// Model-bound (kept through the 2026-08-09 rename pass): these are the checkpoint's own
/// tensor names, and their gate/up/down slot ORDER is a fact about `Expert.forward`, pinned
/// against the reference by the tests below. A behaviour name would hide that these three
/// strings must match bytes on someone else's disk.
pub const V4_PROJ: [&str; 3] = ["w1", "w3", "w2"];

/// [`V4_PROJ`] zipped against [`vq_expert_layout`] — the V4 analogue of [`expert_projs`].
pub fn v4_expert_projs(expert_in: usize, moe_inter: usize) -> ExpertProjs {
    let [g, u, d] = vq_expert_layout(expert_in, moe_inter);
    [(V4_PROJ[0], g), (V4_PROJ[1], u), (V4_PROJ[2], d)]
}

/// V4's tensor prefix for expert `e` in `layer`; `e == n_experts` is the SHARED expert.
/// The V4 analogue of [`expert_base`], and the boundary matters for the same reason —
/// except that in V4 the two are not even the same *format*: routed experts are FP4
/// (`I8` nibble pairs + `F8_E8M0` scales) and the shared expert is `F8_E4M3` at 128×128,
/// so a block written past the boundary is not merely the wrong weights, it is the wrong
/// arithmetic. `.f4` therefore holds routed experts ONLY; the shared expert rides the
/// resident fp8 path.
pub fn v4_expert_base(layer: usize, e: usize, n_experts: usize) -> String {
    expert_slot_name(&format!("layers.{layer}.ffn"), e, n_experts)
}

// --- Kimi-K3 tensor naming ---------------------------------------------------------------
//
// Read off the checkpoint's own `model.safetensors.index.json` (497,220 tensors, 96 shards,
// revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`) on 2026-08-10, reduced to families and
// vendored at `docs/measurement/k3-reference/tensor-families.tsv`.
// `crates/artifact/tests/k3_names.rs` pins every string below against that file — until
// 2026-08-16 this line cited the k3 tree's census while nothing in THIS tree ran it, which
// is a stale claim of a gate; the census and the TSV were vendored together to close it.
//
// **Nothing here was inferred from the reference implementation's variable names**, which is a
// rule this port learned the expensive way in `model.rs`: the C reference calls K3's config
// scalars `kda_heads`/`conv_k`/`rms_eps` and the JSON calls them `num_heads`/
// `short_conv_kernel_size`/`rms_norm_eps`, and two guessed spellings became defects that would
// have refused every real checkpoint.

/// The `language_model.model.` prefix every text-side K3 tensor carries.
///
/// **No document in this repo mentioned it before 2026-08-10** — the plan and the architecture
/// doc both quote names from `layers.` onward, because the C reference's own loader does. A
/// converter built from those docs finds zero tensors and reports a corrupt checkpoint.
/// (`vision_tower.*` and `mm_projector.*` are siblings of `language_model`, not children, which
/// is why skipping the vision side is a prefix test rather than a substring search.)
pub const K3_TEXT_PREFIX: &str = "language_model.model.";

/// K3's three expert projections in [`PROJ`]'s slot order — gate, up, down.
///
/// **The names are `w1`/`w3`/`w2`, the same three strings as [`V4_PROJ`], and the coincidence is
/// worth stating rather than sharing a constant**: they are two different checkpoints that happen
/// to agree, and `docs/reference/k3-architecture.md` §6 fixes K3's slot order from its own
/// forward pass (`w1` gate, `w3` up, `w2` down) rather than from V4's. A shared constant would
/// make one model's rename silently retarget the other's converter.
///
/// Shapes, from the shard header: `w1`/`w3` are `[moe_inter, expert_in]` and `w2` is
/// `[expert_in, moe_inter]`, so a `w2` in the wrong slot is caught by its shape and a `w1`/`w3`
/// swap is not — the same asymmetry [`V4_PROJ`] documents, and the same answer: only S1b's
/// numerical oracle can see it.
pub const K3_PROJ: [&str; 3] = ["w1", "w3", "w2"];

/// The two tensors per projection: MXFP4 nibbles and their e8m0 group scales.
///
/// `compressed-tensors` names, **not** HuggingFace's `.weight`/`.weight_scale_inv`. Both are `U8`
/// on disk — the scale tensor is raw e8m0 exponent bytes, which is why it is `weight_scale` and
/// not `weight_scale_inv`: there is no reciprocal and no f32 anywhere.
pub const K3_PACKED: &str = "weight_packed";
pub const K3_SCALE: &str = "weight_scale";

/// K3's tensor prefix for routed expert `e` in `layer`.
///
/// **There is no shared-expert arm, unlike [`v4_expert_base`]**, and that is not an omission:
/// K3's shared expert is one fused BF16 MLP at FULL width (`shared_experts.down_proj` is
/// `[7168, 6144]`), trunk-side, in a different dtype and a different layout from the routed
/// experts. `.f4` holds routed experts only, and `has_shared()` is already false for F4.
pub fn k3_expert_base(layer: usize, e: usize) -> String {
    routed_expert_name(
        &format!("{K3_TEXT_PREFIX}layers.{layer}.block_sparse_moe"),
        e,
    )
}

// There is deliberately no `k3_expert_proj(layer, e, p) -> (packed, scale)` helper. One existed
// for a day and was DELETED 2026-08-11 after review: nothing in `src/` called it — `F4Expert::spans`
// composes those six names itself from `F4Naming`, which is the path a conversion actually runs —
// and its only test asserted that the helper's output ended with the same three constants the
// helper had just concatenated. A second constructor for a string the engine builds elsewhere is
// a second thing to keep in step, and a test of it is a guard unable to fire.

#[cfg(test)]
mod tests {
    // The one test here reads a reference checkpoint with `expect`, so a restructured
    // reference names itself in the panic instead of silently passing. Crate-wide
    // `unwrap`/`expect` are `deny`; a firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// **`V4_PROJ`'s ORDER, derived from the reference rather than restated.**
    ///
    /// This exists because a mutation test found the hole: swapping `w1` and `w3` in the
    /// constant is invisible to everything else in S1a. The `.f4` repack maps source name →
    /// block slot through this one constant, so the writer and the byte-exactness verifier
    /// both move — the artifact is self-consistently wrong, byte-clean, and only a
    /// numerical oracle against the reference could see it. The two tensors even have
    /// identical shapes, so no dimension check helps.
    ///
    /// So the order is read back out of `inference/model.py`'s `Expert.forward`
    /// (`gate = self.w1(x)`, `up = self.w3(x)`, `return self.w2(…)`) and compared. That
    /// turns the doc comment's citation from decoration into a check. Skipped when the
    /// checkpoint is absent — and S1b's oracle remains the real gate, since this pins only
    /// what the reference SAYS, not what rivoli then computes.
    #[test]
    fn v4_proj_order_matches_the_reference_expert_forward() {
        const REF: &str = "/var/db/rivoli/deepseek-v4-flash-0731/inference/model.py";
        let Ok(src) = std::fs::read_to_string(REF) else {
            eprintln!("SKIP v4_proj_order: no reference at {REF} — V4_PROJ is UNPINNED");
            return;
        };
        // `Expert.forward` only — `MoE.forward` and the mtp blocks mention `w1`/`w2` too.
        let body = src
            .split_once("class Expert(nn.Module)")
            .map(|(_, rest)| rest)
            .and_then(|rest| rest.split_once("class MoE(nn.Module)"))
            .map(|(body, _)| body)
            .expect("Expert class not found — the reference has been restructured");
        let pick = |lhs: &str| -> String {
            let at = body
                .find(lhs)
                .unwrap_or_else(|| panic!("{lhs:?} not in Expert.forward"));
            let rest = &body[at + lhs.len()..];
            let w = rest
                .split_once('(')
                .map(|(w, _)| w.trim())
                .expect("no call after the projection");
            assert!(
                w.starts_with('w') && w.len() == 2,
                "unexpected projection {w:?}"
            );
            w.to_string()
        };
        let got = [
            pick("gate = self."),
            pick("up = self."),
            pick("return self."),
        ];
        assert_eq!(
            got,
            V4_PROJ.map(String::from),
            "V4_PROJ is [gate, up, down]; the reference says {got:?}"
        );
    }
}
