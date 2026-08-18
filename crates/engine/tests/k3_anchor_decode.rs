//! **M9's device exit gate: the K3 engine's decode composition, scored where the anchor can
//! score it and proved structurally where it cannot.**
//!
//! `k3_anchor/mod.rs`'s header carries the division of labour and its two reasons (the
//! anchor vendors no weights, and its tiny widths refuse the engine's own `.f4` group
//! rule). What remains for a device gate, and what each half claims:
//!
//! 1. **The KDA recurrence boundary against the reference itself.** The anchor captures the
//!    complete decode-step interface of the one operator no document attests
//!    (`k3:docs/reference/k3-architecture.md` §4: fla owns the inner arithmetic), at three
//!    layers, both salts. This gate uploads those inputs EXACTLY as `k3/forward.rs`
//!    composes the launch — same argument set, same raw-input contract, the fixture-side
//!    `[value][key] → [key][value]` transpose that the engine's state layout demands — and
//!    scores `out.o` and `out.state` under the tolerance table's `kda_op` row, the one
//!    anchor-derived bound with provenance (`crates/oracles/tests/common/tolerance.rs`,
//!    gated by `k3_anchor.rs::the_tolerance_table_is_supported_by_its_measurements`).
//! 2. **The composition, structurally, on a synthetic F4-legal tiny model**: residency
//!    moves bytes and never text (P4 — bit-identical logits across budgets that provably
//!    differ, `RoutedPool::budget()` being the discriminator); a second `generate` on one
//!    engine is bit-identical to the first (which is also what proves `reset` clears every
//!    recurrent state and conv ring); and a CARRIED KDA state equals a REPLAYED prefix
//!    bit-for-bit — the property that lets the per-operator kernel gates compose into a
//!    correct decode, and the decode-level fact no kernel gate can see.
//!
//! What is deliberately NOT here: per-operator numerics (`kernel_k3_*.rs` against their own
//! fixtures), and an engine-vs-reference logits comparison, which needs a weight dump the
//! anchor does not carry — `the_anchor_widths_are_not_engine_runnable` in the widths gate
//! goes red the day that changes.
//!
//! # RED-PROOF PLAN — the integrator pays these on first device run (P7)
//!
//! Each row is a one-line edit, reverted after. Rows 1-3 redden test 1; row 4 reddens the
//! residency test; row 5 is the binary-freshness proof the others depend on.
//!
//! | # | edit | reddens |
//! |---|---|---|
//! | 1 | `k3_anchor::to_key_major` → return `v.to_vec()` (drop the transpose) | `out.state` past `kda_op`'s 6.3e-4 — the anchor's own `KdaStateLayout` defect, priced at 1.75e0 |
//! | 2 | swap the `k` and `v` uploads at this file's recurrence launch | `out.o` and `out.state` — the anchor's weakest KDA defect is 1.75e0 against tol 6.3e-4 |
//! | 3 | pass `0.0` for `lower_bound` at the launch | launcher guard 1006 refuses (`-5 <= lb < 0`), the test errors — proves the config value REACHES the launch |
//! | 4 | in [`residency_moves_bytes_and_never_text`], pass `roomy` for both budgets | the `budget_tight < budget_roomy` discriminator — the glimmer gate's row-4 precedent, and what makes rows 1-3 evidence about both residency states |
//! | 5 | in [`kda_step`], `drop(got)` before the comparison | the SOURCE no longer compiles — so a run that still prints green executed a STALE binary. The eaten-exit false green was paid twice on 2026-08-16 (`docs/measurement/gate-red-proofs.md` §4); run this row FIRST |
//!
//! **PAID 2026-08-16, all five rows, in row order 5-1-3-4-2** (each planted, observed,
//! reverted; green 3/3 re-observed after): row 5 — build exit 101, zero green lines from
//! the run attempt; row 1 — `out.o` red at **1.300e0**; row 3 — guard **1006** refused,
//! after a first plant that orphaned `lb` and was itself refused by warnings-deny (the
//! consuming form is `lb * 0.0`); row 4 — "the two budgets resolved to one pool
//! (147456 B)"; row 2 — `out.o` red at **7.041e-1**. The clean run's recorded worst:
//! 2.265e-7 at k3-anchor-1 L12 `out.o`, three orders under the bound and non-zero (the
//! anti-vacuity assert's territory); pools 147456 vs 24576 B, 16 tight-arm misses.
//!
//! # Running it
//!
//! Device tests: under the GPU flock, `-- --test-threads=1`, on the DEV profile (the pin
//! and the loop carry live `debug_assert!`s there and this fixture is far too small for
//! timing to matter), `--nocapture` for the worst-case numbers.
//!
//! ```text
//! flock /var/run/sys-gpu.lock -c 'cargo test -p rivoli-engine --test k3_anchor_decode \
//!     -- --test-threads=1 --nocapture'
//! ```

// The whole binary, like every kernel_* sibling: a featureless build has no engine, and
// the deviceless claims live in `k3_anchor_widths.rs` so gating this off costs CI nothing.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;
mod k3_anchor;

use common::{Lcg, back, dev, f32b, f32v, ok, rel};
use k3_anchor::{KDA_LAYERS, anchors, float, kda_tag, kda_tol, to_key_major};
use rivoli_artifact::format::{
    Dtype, EXPERT_HEADER_BYTES, ExpertHeader, F4_MAGIC, LAYER_WINDOW, LayerDims, SafeWriter,
    write_expert_layer,
};
use rivoli_artifact::k3_config::K3Config;
use rivoli_artifact::quant::f4::{f4_expert_bytes, f4_expert_stride, f4_slot_offsets};
use rivoli_artifact::schema::parse_config;
use rivoli_backend::{NULL_STREAM, device_sync, launch_gated_delta_recurrent_f32};
use rivoli_core::num::f32_to_bf16;
use rivoli_engine::k3::engine::K3Engine;
use rivoli_engine::k3::pin::K3Pin;
use rivoli_engine::resident::{PinCfg, safetensors_bytes};
use rivoli_engine::seam::GenSpec;

/// **The engine's KDA launch composition against the reference's own recurrence.** Per
/// salt, per captured KDA layer: the anchor's raw inputs through the exact launch
/// `k3/forward.rs` makes, `out.o` and `out.state` scored under `kda_op`'s measured bound.
#[test]
fn the_kda_recurrence_reproduces_the_anchor() {
    let tol = kda_tol();
    let mut worst = (0.0f32, String::new());
    for a in &anchors() {
        for l in KDA_LAYERS {
            let (o_rel, s_rel) = kda_step(a, l);
            for (what, r) in [("out.o", o_rel), ("out.state", s_rel)] {
                assert!(
                    r < tol,
                    "{}: L{l} {what} disagrees with the reference by {r:.3e}, past the \
                     kda_op tolerance {tol:e} (10x its measured chunk-vs-recurrent floor; \
                     the weakest KDA defect the anchor priced is 1.75e0)",
                    a.name
                );
                if r > worst.0 {
                    worst = (r, format!("{} L{l} {what}", a.name));
                }
            }
        }
    }
    println!("  worst KDA disagreement {:.3e} at {}", worst.0, worst.1);
    // Anti-vacuity: two independent fp32 implementations of the recurrence cannot agree
    // to 0.0 over six fixtures — bitwise agreement here means something compared itself
    // to itself.
    assert!(worst.0 > 0.0, "every comparison was bitwise-identical");
}

/// One captured KDA step through the device recurrence. Returns `(rel out.o, rel out.state)`,
/// both compared in the ENGINE's key-major state layout. Every width and the gate bound
/// come from the anchor's OWN tiny config, so the launch cannot disagree with the fixture.
fn kda_step(a: &k3_anchor::Anchor, l: usize) -> (f32, f32) {
    let (nh, hd) = (a.attn_field("num_heads"), a.attn_field("head_dim"));
    let lb = a.lower_bound();
    let tag = kda_tag(l);
    let get = |n: &str| float(&a.caps, &format!("{tag}.{n}")).1;
    // fla's state is [value][key]; the engine's is [key][value] — the transpose is the
    // FIXTURE's job, and `to_key_major`'s doc carries the layout argument.
    let state0 = to_key_major(get("in.initial_state"), nh, hd);
    let mut state = dev(&f32b(&state0));
    let mut out = dev(&f32b(&vec![0.0f32; nh * hd]));
    // The first seven fixture names ARE the launch's input operands, in argument order —
    // ONE list (`k3_anchor::KDA_FIXTURE`), so the census and this upload cannot cover
    // different names.
    let bufs: Vec<_> = k3_anchor::KDA_FIXTURE[..7]
        .iter()
        .map(|n| dev(&f32b(get(n))))
        .collect();
    // SAFETY: every buffer above was sized from the capture the widths gate census pins;
    // the launch is `k3/forward.rs::kda_attention`'s, argument for argument.
    unsafe {
        ok(
            launch_gated_delta_recurrent_f32(
                bufs[0].ptr().cast(),
                bufs[1].ptr().cast(),
                bufs[2].ptr().cast(),
                bufs[3].ptr().cast(),
                bufs[4].ptr().cast(),
                bufs[5].ptr().cast(),
                bufs[6].ptr().cast(),
                nh,
                hd,
                lb,
                state.ptr_mut().cast(),
                out.ptr_mut().cast(),
                NULL_STREAM,
            ),
            "the recurrence launch",
        );
    }
    ok(device_sync(), "join");
    let want_state = to_key_major(get("out.state"), nh, hd);
    (
        rel(&f32v(&back(&out)), get("out.o")),
        rel(&f32v(&back(&state)), &want_state),
    )
}

// ---------------------------------------------------------------------------------------
// The synthetic F4-legal tiny model — engine-runnable where the anchor's is not.
// ---------------------------------------------------------------------------------------

/// The tiny config, built as JSON and parsed through the REAL parser so `validate` (the
/// partition assert, the group rule, the flag set) vouches for the fixture the same way it
/// vouches for the shipped checkpoint.
///
/// The widths exercise every path: two attention families (KDA at 0,1,2,4; MLA at 3), the
/// dense layer 0, four MoE layers, a boundary mid-run (`attn_res_block_size` 2 → pushes at
/// 0, 2, 4 and a final 4-source fold), and top-2-of-4 routed experts at a 64-wide latent.
fn tiny_config() -> K3Config {
    // The WRAPPER pair is the recognised one — `arch.rs` deliberately excludes the nested
    // `kimi_linear` spelling (the real config.json is a multimodal wrapper; descending
    // first would look for `kimi_k3` where the file says `kimi_linear`). Authored without
    // these two keys originally: the refusal fired on this gate's first device run,
    // 2026-08-16 — the parse sits in the rocm binary its author could never execute.
    let doc = serde_json::json!({
        "model_type": "kimi_k3",
        "architectures": ["KimiK3ForConditionalGeneration"],
        "text_config": {
        "model_type": "kimi_linear",
        "architectures": ["KimiLinearForCausalLM"],
        "num_hidden_layers": 5, "hidden_size": 64, "vocab_size": 32,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "rms_norm_eps": 1e-5, "dtype": "bfloat16",
        "linear_attn_config": {
            "full_attn_layers": [4], "kda_layers": [1, 2, 3, 5],
            "num_heads": 2, "head_dim": 32, "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0, "use_full_rank_gate": true,
        },
        "q_lora_rank": 16, "kv_lora_rank": 128, "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8, "v_head_dim": 16,
        "mla_use_nope": true, "mla_use_output_gate": true,
        "attn_res_block_size": 2,
        "num_experts": 4, "num_experts_per_token": 2, "num_shared_experts": 2,
        "routed_expert_hidden_size": 64, "moe_intermediate_size": 64,
        "latent_moe_use_norm": true, "moe_renormalize": true,
        "num_expert_group": 1, "topk_group": 1, "topk_method": "noaux_tc",
        "moe_router_activation_func": "sigmoid", "routed_scaling_factor": 1.0,
        "moe_layer_freq": 1,
        "activation_situ_beta": 4.0, "activation_situ_linear_beta": 25.0,
        "first_k_dense_replace": 1, "intermediate_size": 32, "hidden_act": "situ",
        "num_nextn_predict_layers": 0, "tie_word_embeddings": false,
    }});
    parse_config::<K3Config>(&doc.to_string()).expect("the tiny config is a legal K3 one")
}

/// An artifact directory that removes itself on drop — panic-safe, `tag`-keyed so two
/// tests cannot collide, and under the per-process id so two RUNS cannot either. A
/// `String`, not the glimmer gate's `PathBuf`: every consumer here wants `&str` and the
/// difference is also what keeps the two fixtures' identical PURPOSE from becoming an
/// identical token stream under the duplication gate.
struct TinyModel(String);

impl TinyModel {
    fn path(&self) -> &str {
        &self.0
    }
}

impl Drop for TinyModel {
    fn drop(&mut self) {
        // Ignore the error: a fixture that failed before writing has nothing to remove,
        // and a cleanup failure must not mask the assertion that got the test here.
        drop(std::fs::remove_dir_all(&self.0));
    }
}

fn bf16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| f32_to_bf16(*v).to_le_bytes())
        .collect()
}

/// The driver's weight families (`k3_anchor_driver.py::init_weights`), as data: gains near
/// 1 (a norm drawn near zero makes every downstream activation a denormal), `dt_bias` on
/// its own (-4, 1) scale, everything else near 0. `A_log`'s log-uniform draw stays at its
/// one call site — it is a transform of a draw, not a fourth family.
#[derive(Clone, Copy)]
enum Fam {
    Proj,
    Gain,
    DtBias,
}

impl Fam {
    fn span(self) -> (f32, f32) {
        match self {
            Fam::Proj => (-0.08, 0.08),
            Fam::Gain => (0.8, 1.2),
            Fam::DtBias => (-4.0, 1.0),
        }
    }
}

/// Deterministic weights into the one `SafeWriter` — holding the writer is what keeps each
/// tensor site to one short line against the census.
struct Draw {
    r: Lcg,
    w: SafeWriter<'static>,
}

impl Draw {
    /// `n` uniform draws in `[lo, hi)` — `Lcg::f` is `[-1, 1)`, rescaled here once.
    fn vals(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..n)
            .map(|_| lo + (hi - lo) * (self.r.f() + 1.0) * 0.5)
            .collect()
    }

    fn proj(&mut self, name: &str, o: usize, i: usize) {
        let v = self.vals(o * i, -0.08, 0.08);
        self.w
            .add(name.to_string(), Dtype::Bf16, vec![o, i], bf16_bytes(&v));
    }

    fn norm(&mut self, name: &str, shape: Vec<usize>) {
        let n = shape.iter().product();
        let v = self.vals(n, 0.8, 1.2);
        self.w
            .add(name.to_string(), Dtype::Bf16, shape, bf16_bytes(&v));
    }

    fn f32t(&mut self, name: &str, shape: Vec<usize>, fam: Fam) {
        let n = shape.iter().product();
        let (lo, hi) = fam.span();
        let v = self.vals(n, lo, hi);
        self.w.add(name.to_string(), Dtype::F32, shape, f32b(&v));
    }
}

/// Write the whole tiny artifact: resident.safetensors, four `.f4` expert layers, and the
/// manifest whose `f4_source.layers` is the loader's only source for which layers exist.
fn write_artifact(tag: &str) -> (TinyModel, K3Config) {
    let cfg = tiny_config();
    let t = &cfg.text;
    let root = std::env::temp_dir().join(format!("rivoli-k3-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("create the artifact dir");
    let art = TinyModel(root.to_str().expect("utf-8 temp path").to_string());

    let mut d = Draw {
        r: Lcg(0x4b33_5eed),
        w: SafeWriter::new(),
    };
    write_globals(&mut d, t);
    for l in 0..t.n_layers {
        write_layer_tensors(&mut d, t, l);
    }
    d.w.write(&format!("{}/resident.safetensors", art.path()))
        .expect("write the resident tensors");
    write_experts(art.path(), t);
    std::fs::write(
        format!("{}/manifest.json", art.path()),
        serde_json::json!({ "f4_source": { "layers": [t.first_k_dense_replace, t.n_layers] } })
            .to_string(),
    )
    .expect("write the manifest");
    (art, cfg)
}

fn write_globals(d: &mut Draw, t: &rivoli_artifact::k3_config::K3TextConfig) {
    d.proj(
        "language_model.model.embed_tokens.weight",
        t.vocab,
        t.hidden,
    );
    d.proj("language_model.lm_head.weight", t.vocab, t.hidden);
    d.norm("language_model.model.norm.weight", vec![t.hidden]);
    d.norm(
        "language_model.model.output_attn_res_norm.weight",
        vec![t.hidden],
    );
    d.proj(
        "language_model.model.output_attn_res_proj.weight",
        1,
        t.hidden,
    );
}

/// One layer's sandwich norms and folds, then its two family halves — split like the pin's
/// own placement, so each half stays readable against the census it mirrors.
fn write_layer_tensors(d: &mut Draw, t: &rivoli_artifact::k3_config::K3TextConfig, l: usize) {
    let at = format!("language_model.model.layers.{l}");
    for name in ["input_layernorm", "post_attention_layernorm"] {
        d.norm(&format!("{at}.{name}.weight"), vec![t.hidden]);
    }
    for base in ["self_attention_res", "mlp_res"] {
        d.norm(&format!("{at}.{base}_norm.weight"), vec![t.hidden]);
        d.proj(&format!("{at}.{base}_proj.weight"), 1, t.hidden);
    }
    write_attn_tensors(d, t, l, &format!("{at}.self_attn"));
    write_ffn_tensors(d, t, l, &at);
}

fn write_attn_tensors(
    d: &mut Draw,
    t: &rivoli_artifact::k3_config::K3TextConfig,
    l: usize,
    sa: &str,
) {
    let (hid, la) = (t.hidden, &t.linear_attn_config);
    match t.layer_is_mla(l).expect("a mapped layer") {
        false => {
            let ch = la.num_heads * la.head_dim;
            for name in ["q_proj", "k_proj", "v_proj", "g_proj"] {
                d.proj(&format!("{sa}.{name}.weight"), ch, hid);
            }
            for c in ["q_conv1d", "k_conv1d", "v_conv1d"] {
                let shape = vec![ch, 1, la.short_conv_kernel_size];
                d.f32t(&format!("{sa}.{c}.weight"), shape, Fam::Proj);
            }
            d.proj(&format!("{sa}.f_a_proj.weight"), la.head_dim, hid);
            d.proj(&format!("{sa}.f_b_proj.weight"), ch, la.head_dim);
            d.proj(&format!("{sa}.b_proj.weight"), la.num_heads, hid);
            // The driver's own scale: a wrong A_log freezes or erases the state.
            let a_log: Vec<f32> = d
                .vals(la.num_heads, 1.0, 16.0)
                .iter()
                .map(|x| x.ln())
                .collect();
            d.w.add(
                format!("{sa}.A_log"),
                Dtype::F32,
                vec![la.num_heads],
                f32b(&a_log),
            );
            d.f32t(&format!("{sa}.dt_bias"), vec![ch], Fam::DtBias);
            d.f32t(&format!("{sa}.o_norm.weight"), vec![la.head_dim], Fam::Gain);
            d.proj(&format!("{sa}.o_proj.weight"), hid, ch);
        }
        true => {
            let (nh, ov) = (t.n_heads, t.n_heads * t.v_head_dim);
            d.proj(&format!("{sa}.q_a_proj.weight"), t.q_lora_rank, hid);
            d.norm(&format!("{sa}.q_a_layernorm.weight"), vec![t.q_lora_rank]);
            let qh = t.qk_nope_head_dim + t.qk_rope_head_dim;
            d.proj(&format!("{sa}.q_b_proj.weight"), nh * qh, t.q_lora_rank);
            let kva = t.kv_lora_rank + t.qk_rope_head_dim;
            d.proj(&format!("{sa}.kv_a_proj_with_mqa.weight"), kva, hid);
            d.norm(&format!("{sa}.kv_a_layernorm.weight"), vec![t.kv_lora_rank]);
            let kvb = t.qk_nope_head_dim + t.v_head_dim;
            d.proj(&format!("{sa}.kv_b_proj.weight"), nh * kvb, t.kv_lora_rank);
            d.proj(&format!("{sa}.g_proj.weight"), ov, hid);
            d.proj(&format!("{sa}.o_proj.weight"), hid, ov);
        }
    }
}

fn write_ffn_tensors(
    d: &mut Draw,
    t: &rivoli_artifact::k3_config::K3TextConfig,
    l: usize,
    at: &str,
) {
    let hid = t.hidden;
    match t.layer_is_dense(l) {
        true => {
            let m = format!("{at}.mlp");
            d.proj(&format!("{m}.gate_proj.weight"), t.dense_inter, hid);
            d.proj(&format!("{m}.up_proj.weight"), t.dense_inter, hid);
            d.proj(&format!("{m}.down_proj.weight"), hid, t.dense_inter);
        }
        false => {
            let m = format!("{at}.block_sparse_moe");
            d.proj(&format!("{m}.gate.weight"), t.n_experts, hid);
            d.f32t(
                &format!("{m}.gate.e_score_correction_bias"),
                vec![t.n_experts],
                Fam::Proj,
            );
            d.proj(
                &format!("{m}.routed_expert_down_proj.weight"),
                t.expert_in,
                hid,
            );
            d.norm(&format!("{m}.routed_expert_norm.weight"), vec![t.expert_in]);
            d.proj(
                &format!("{m}.routed_expert_up_proj.weight"),
                hid,
                t.expert_in,
            );
            let sh = t.n_shared * t.moe_inter;
            d.proj(&format!("{m}.shared_experts.gate_proj.weight"), sh, hid);
            d.proj(&format!("{m}.shared_experts.up_proj.weight"), sh, hid);
            d.proj(&format!("{m}.shared_experts.down_proj.weight"), hid, sh);
        }
    }
}

/// The `.f4` layers: random e2m1 nibbles under e8m0 scales drawn from the byte range the
/// shipped V4 set measured (`0x76..=0x7e`) — never `0xff`, the NaN the repack path refuses.
fn write_experts(dir: &str, t: &rivoli_artifact::k3_config::K3TextConfig) {
    let (lat, inter, ne) = (t.expert_in, t.moe_inter, t.n_experts);
    let (bytes, stride) = (f4_expert_bytes(lat, inter), f4_expert_stride(lat, inter));
    let off = f4_slot_offsets(lat, inter);
    for l in t.first_k_dense_replace..t.n_layers {
        let dims = LayerDims {
            layer: l,
            n_experts: ne,
            expert_in: lat,
            moe_inter: inter,
            stride,
        };
        let header: [u8; EXPERT_HEADER_BYTES] = ExpertHeader::new(F4_MAGIC, dims).to_bytes();
        // One deterministic byte stream per (layer, expert), scales re-drawn into range.
        let seeds: Vec<u64> = (0..ne).map(|e| 0x9e37 + (l * ne + e) as u64).collect();
        write_expert_layer(
            &format!("{dir}/L{l:02}.f4"),
            &header,
            stride,
            bytes,
            ne,
            LAYER_WINDOW,
            |e, slot| {
                let mut r = Lcg(seeds[e]);
                for b in slot.iter_mut() {
                    *b = ((r.f() + 1.0) * 127.5) as u8;
                }
                for span in [off[1]..off[2], off[3]..off[4], off[5]..bytes] {
                    for b in &mut slot[span] {
                        *b = 0x76 + (*b % 9);
                    }
                }
                Ok(())
            },
        )
        .expect("write an expert layer");
    }
    // The scale spans above are derived from `f4_slot_offsets`; nothing else checks them
    // here because `ExpertSet::open_routed` re-derives the same geometry at load.
}

/// Open the engine at one budget. The capacity arithmetic mirrors the pin's floor (tier =
/// file bytes + its 64 MiB slack, plus the top-2 batch slots) — a MIRROR, not an import,
/// so if `K3Pin`'s slack ever grows past it the TIGHT arm refuses loudly at build rather
/// than silently pinning a different count.
fn open_engine<'c>(
    dir: &str,
    cfg: &'c K3Config,
    extra_units: usize,
    max_ctx: usize,
) -> K3Engine<'c> {
    let unit = f4_expert_stride(cfg.text.expert_in, cfg.text.moe_inter);
    let floor = ok(safetensors_bytes(dir, None), "file bytes") + (64 << 20) + 2 * unit;
    let pin = PinCfg {
        // Stock allocation: a fixture must not silently test the candidate fix.
        pinned_coherent: false,
        copy_by_kernel: false,
        capacity: floor + extra_units * unit + 512,
        cache_policy: "2q",
        two_q: rivoli_core::cache::TwoQSplit::default(),
        trace_path: None,
    };
    let pin = ok(K3Pin::build(dir, &cfg.text, pin), "build the tiny pin");
    ok(K3Engine::new(pin, &cfg.text, max_ctx), "build the engine")
}

/// One greedy decode; returns `(ids, logits, misses, pool_budget)`.
fn run(e: &mut K3Engine<'_>, prompt: &[u32], ngen: usize) -> (Vec<u32>, Vec<f32>, u64, usize) {
    let spec = GenSpec {
        prompt,
        ngen,
        eos: &[],
    };
    let out = ok(e.decode(spec, &mut |_| true), "a tiny decode");
    let logits = ok(e.logits(), "the last logits");
    (out.ids, logits, out.stats.misses, e.pool_budget())
}

/// **P4 on the synthetic model: the budget moves bytes and never text — and the two arms
/// really are two residency states.** Plus determinism: a second `generate` on the SAME
/// engine is bit-identical, which is `reset` proving it cleared every KDA state, conv ring
/// and mask.
#[test]
fn residency_moves_bytes_and_never_text() {
    let (art, cfg) = write_artifact("p4");
    // 16 streamable units exist (4 MoE layers x 4 experts); +32 pins everything with room
    // to spare, +1 pins one and streams fifteen.
    let mut roomy = open_engine(art.path(), &cfg, 32, 8);
    let (ids_a, logits_a, _, budget_a) = run(&mut roomy, &[1, 2, 3], 3);
    let (ids_a2, logits_a2, _, _) = run(&mut roomy, &[1, 2, 3], 3);
    assert_eq!(
        ids_a, ids_a2,
        "a second generate on one engine changed the text"
    );
    assert!(
        logits_a == logits_a2,
        "a second generate changed the logits — reset left state behind"
    );
    drop(roomy);
    let mut tight = open_engine(art.path(), &cfg, 1, 8);
    let (ids_b, logits_b, misses_b, budget_b) = run(&mut tight, &[1, 2, 3], 3);
    println!("  pools: roomy {budget_a} B, tight {budget_b} B, tight misses {misses_b}");
    // The discriminator: the arms differ in PLACEMENT INPUT, provably, deviceless of any
    // cache semantics — this is what the red-proof plan's row 4 turns off.
    assert!(
        budget_b < budget_a,
        "the two budgets resolved to one pool ({budget_a} B) — the arms are the same arm"
    );
    assert!(misses_b > 0, "the tight arm never streamed an expert");
    assert_eq!(ids_a, ids_b, "residency changed the TEXT (P4)");
    assert!(
        logits_a == logits_b,
        "residency changed the LOGITS while the ids agreed — P4 is violated below argmax \
         resolution, the worst version of the defect"
    );
}

/// **A carried KDA state equals a replayed prefix, bit for bit.** Decode two tokens in one
/// run; then hand the first emitted token back as prompt tail to a FRESH engine and decode
/// one. Same launch sequence by construction (prefill IS the decode path on this arm), so
/// anything but equality is state the carry lost or invented.
#[test]
fn carrying_the_state_equals_replaying_the_prefix() {
    let (art, cfg) = write_artifact("carry");
    let mut e = open_engine(art.path(), &cfg, 32, 8);
    let (ids, logits_carried, _, _) = run(&mut e, &[1, 2, 3], 2);
    assert_eq!(ids.len(), 2, "the carry run must emit twice");
    let replay: Vec<u32> = [1, 2, 3, ids[0]].to_vec();
    let (ids2, logits_replayed, _, _) = run(&mut e, &replay, 1);
    assert_eq!(
        ids2,
        vec![ids[1]],
        "the replayed prefix picked a different token"
    );
    assert!(
        logits_carried == logits_replayed,
        "carried state and replayed prefix disagree at the bit level — the recurrence \
         carried something a replay does not reproduce"
    );
}
