//! Integration check: an artifact produced by `convert` reads back through the
//! loader (format.rs) — the converter↔loader byte contract, end to end. Runs only
//! when RIVOLI_ARTIFACT points to an artifact dir (a `--layers N` conversion), else
//! it skips, so CI without a checkpoint stays green.
#![allow(clippy::unwrap_used)]

use rivoli::artifact::format::{Dtype, ExpertSet, FormatMeta, Safetensors, load_codebooks};
use rivoli::artifact::model::ModelConfig;
use rivoli::artifact::quant::{VQ_ALIGN, VQ_DIM, VQ_K, vq_expert_bytes};
use rivoli::artifact::tokenizer::Tokenizer;

/// The artifact dir, or `None` after printing the skip line. Every test in this file is
/// gated on it, and the gate is a function so the four copies of the message — the thing a
/// reader greps for when a CI run reports four passes and did nothing — cannot drift.
fn artifact_dir() -> Option<String> {
    match std::env::var("RIVOLI_ARTIFACT") {
        Ok(d) => Some(d),
        Err(_) => {
            eprintln!("skip: set RIVOLI_ARTIFACT to an artifact dir");
            None
        }
    }
}

/// Run `body` with the artifact's tokenizer, or skip.
///
/// A combinator rather than an `Option`-returning getter because the two chat-framing
/// tests are gated identically, and `let Some(tok) = … else { return }` written twice is
/// two places for the skip path to stop matching the message [`artifact_dir`] prints.
fn with_tokenizer(body: impl FnOnce(&Tokenizer)) {
    if let Some(dir) = artifact_dir() {
        body(&Tokenizer::load(&dir).unwrap());
    }
}

#[test]
fn artifact_reads_back() {
    let Some(dir) = artifact_dir() else { return };
    // manifest: format section + config fields
    let meta = FormatMeta::load(&dir).unwrap();
    assert_eq!((meta.vq_dim, meta.vq_k, meta.vq_group), (VQ_DIM, VQ_K, 64));
    let cfg = ModelConfig::load(&dir).unwrap();

    // 3 per-projection codebooks
    let cbs = load_codebooks(&dir).unwrap();
    for cb in &cbs {
        assert_eq!(cb.len(), VQ_K * VQ_DIM);
    }

    // resident.safetensors: embed int8 with the right shape, attention fp8 present
    let res = Safetensors::open_file(&format!("{dir}/resident.safetensors")).unwrap();
    let (_, sh) = res.typed("model.embed_tokens.weight", Dtype::I8).unwrap();
    assert_eq!(sh, &[cfg.vocab, cfg.hidden]);
    assert!(res.has("model.embed_tokens.weight.scale"));
    res.typed("model.layers.3.self_attn.q_a_proj.weight", Dtype::F8E4M3)
        .unwrap();
    res.typed("model.layers.3.mlp.gate.weight", Dtype::F32)
        .unwrap(); // widened

    // .vq3: headers validate against config; routed reads aligned; shared block sized
    let n_present = std::fs::read_dir(&dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .path()
                .extension()
                .is_some_and(|x| x == "vq3")
        })
        .count();
    let last = cfg.dense_layers + n_present;
    let vq = ExpertSet::open_vq3(
        &dir,
        cfg.dense_layers,
        last,
        cfg.n_experts,
        cfg.hidden,
        cfg.moe_inter,
    )
    .unwrap();
    let (_, begin, len) = vq.read_spec(cfg.dense_layers, 0).unwrap();
    assert_eq!(begin % VQ_ALIGN, 0, "routed read must be O_DIRECT aligned");
    assert_eq!(len, vq_expert_bytes(cfg.hidden, cfg.moe_inter));
    let shared = vq.shared_block(cfg.dense_layers).unwrap();
    assert_eq!(shared.len(), vq_expert_bytes(cfg.hidden, cfg.moe_inter));
    eprintln!(
        "artifact OK: {n_present} layers, {} experts + shared",
        cfg.n_experts
    );
}

/// The shipped `.i4` bytes ARE `quant_i4(dequant_fp8(checkpoint))` — bit for bit.
///
/// Every other check on this format is a statistic with a tolerance. This one is exact,
/// and it is the only thing that can tell one `.i4` GENERATION from another: a
/// a `vq3_to_i4` set and an `fp8_to_i4` set are byte-indistinguishable
/// on disk by shape alone (`format.rs::I4Source`), which is precisely how a bad `.i4`
/// set stayed invisible once already. It also catches a torn or partially-resumed
/// conversion, a checkpoint revision swap, and a wrong `weight_scale_inv` tiling.
///
/// CPU only — no `rocm`, no GPU — so it runs wherever the two directories are mounted.
/// Needs `RIVOLI_ARTIFACT`; the checkpoint path comes from the artifact's own
/// `i4_source` stamp, so the test cannot be pointed at the wrong ground truth.
#[test]
fn i4_bytes_are_what_the_checkpoint_quantizes_to() {
    use rivoli::artifact::format::I4Source;
    use rivoli::artifact::quant::{
        i4_expert_stride, i4_groups, i4_row_bytes, i4_slot_offsets, quant_i4, read_f32,
        vq_expert_layout,
    };
    use std::os::unix::fs::FileExt;
    let Some(dir) = artifact_dir() else { return };
    let Some(prov) = I4Source::load(&dir).unwrap() else {
        eprintln!("skip i4_bytes: {dir} has no i4_source stamp");
        return;
    };
    if prov.chain != "fp8->int4" {
        eprintln!(
            "skip i4_bytes: chain is {} — only fp8->int4 is byte-checkable here",
            prov.chain
        );
        return;
    }
    let Ok(src) = Safetensors::open_dir(&prov.src) else {
        eprintln!("skip i4_bytes: checkpoint {} absent", prov.src);
        return;
    };
    let cfg = ModelConfig::load(&dir).unwrap();
    let block = FormatMeta::load(&dir).unwrap().fp8_block;
    let (h, m, ne) = (cfg.hidden, cfg.moe_inter, cfg.n_experts);
    let (stride, off) = (i4_expert_stride(h, m), i4_slot_offsets(h, m));
    let layer = prov.layers[0];
    let Ok(f) = std::fs::File::open(format!("{dir}/L{layer:02}.i4")) else {
        eprintln!("skip i4_bytes: L{layer:02}.i4 absent");
        return;
    };
    // A routed expert and the SHARED expert (block `ne`) — the shared one is the case
    // whose tensor path differs (`mlp.shared_experts` vs `mlp.experts.{e}`), i.e. the
    // one place a naming or offset mismatch could pair the wrong weights with a block
    // and still produce a plausible-looking matrix.
    for e in [0usize, ne] {
        let mut blk = vec![0u8; stride];
        f.read_exact_at(&mut blk, (e * stride) as u64).unwrap();
        let base = if e < ne {
            format!("model.layers.{layer}.mlp.experts.{e}")
        } else {
            format!("model.layers.{layer}.mlp.shared_experts")
        };
        for (k, (&(o_dim, i_dim), proj)) in vq_expert_layout(h, m)
            .iter()
            .zip(["gate_proj", "up_proj", "down_proj"])
            .enumerate()
        {
            let w = src
                .dequant_fp8(&format!("{base}.{proj}"), o_dim, i_dim, block)
                .unwrap();
            let (want_packed, want_scale) = quant_i4(&w, o_dim, i_dim);
            let po = off[k * 2];
            let got_packed = &blk[po..po + o_dim * i4_row_bytes(i_dim)];
            let so = off[k * 2 + 1];
            let got_scale = read_f32(&blk[so..so + o_dim * i4_groups(i_dim) * 4]);
            // Report WHERE, not just that: a whole-projection mismatch (wrong tensor)
            // and three bad rows (a torn write) need different next actions.
            let bad = want_packed
                .chunks_exact(i4_row_bytes(i_dim))
                .zip(got_packed.chunks_exact(i4_row_bytes(i_dim)))
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                bad, 0,
                "expert {e} {proj}: {bad}/{o_dim} packed rows differ from quant_i4(fp8)"
            );
            assert_eq!(
                want_scale, got_scale,
                "expert {e} {proj}: group scales differ from quant_i4(fp8)"
            );
        }
    }
    eprintln!(
        "i4 bytes OK: L{layer:02} experts 0 and {ne} (shared) are bit-exact vs {}",
        prov.src
    );
}

/// The chat framing must match the CHECKPOINT's `chat_template.jinja` byte for byte.
/// `Tokenizer::encode_chat_turns` hand-ports that template — there is no Jinja engine here —
/// and the converted artifact does **not** carry the template, so nothing but this test
/// couples the two. It has drifted once already: the framing was GLM-4's `<|role|>\n{...}`
/// with no thinking prefill until 2026-08-01, one token off per turn, which is why the
/// expected strings below are written out in full rather than built from the code's own
/// pieces. Source of truth: `chat_template.jinja` in `manifest.json`'s `i4_source.src`.
#[test]
fn chat_framing_matches_the_checkpoint_template() {
    use rivoli::artifact::tokenizer::ChatOpts;
    with_tokenizer(|tok| {
        // Decoding with specials rendered is what makes this readable as the template's output.
        let render = |turns: &[(&str, &str)], o: &ChatOpts| {
            tok.decode_all(&tok.encode_chat_turns(turns, o).unwrap())
                .unwrap()
        };
        // The three thinking cases differ ONLY in `reasoning_effort`, so that is the only
        // thing spelled per case: three copies of the surrounding `ChatOpts` would let one
        // of them silently stop setting `thinking`, and the expected strings would still
        // look like they were asserting the effort word.
        let thinking = |effort: Option<&str>| {
            render(
                &[("user", "hi")],
                &ChatOpts {
                    thinking: true,
                    reasoning_effort: effort,
                    ..Default::default()
                },
            )
        };

        // enable_thinking=false: no Reasoning Effort turn, and the prompt closes <think> at once.
        assert_eq!(
            render(&[("user", "hi")], &ChatOpts::default()),
            "[gMASK]<sop><|user|>hi<|assistant|><think></think>"
        );
        // The template's own default: <think> left OPEN, preceded by the effort system turn.
        assert_eq!(
            thinking(None),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>"
        );
        // Only "high" is High. The template capitalizes its default for everything else, so
        // "low" and "medium" render Max — that is the template's behaviour, not a shortcut.
        assert_eq!(
            thinking(Some("high")),
            "[gMASK]<sop><|system|>Reasoning Effort: High<|user|>hi<|assistant|><think>"
        );
        assert_eq!(
            thinking(Some("low")),
            "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>"
        );
        // Multi-turn: NO separator after any role token, and history's reasoning is cleared.
        assert_eq!(
            render(
                &[
                    ("system", "S"),
                    ("user", "a"),
                    ("assistant", "b"),
                    ("user", "c")
                ],
                &ChatOpts::default()
            ),
            "[gMASK]<sop><|system|>S<|user|>a<|assistant|><think></think>b<|user|>c\
             <|assistant|><think></think>"
        );
    });
}

/// The TOOL half of the same template, pinned the same way and for the same reason. The
/// `# Tools` system turn is what teaches the model this checkpoint's `<tool_call>` syntax —
/// invent a preamble instead and it reaches for other frameworks' conventions — so its
/// wording, its blank lines and the `", "` / `": "` spacing of the JSON all matter. That
/// spacing is Python's `json.dumps`, which is what Jinja's `tojson` emits; serde_json's
/// compact form is a different token sequence.
#[test]
fn tool_framing_matches_the_checkpoint_template() {
    use rivoli::artifact::tokenizer::ChatOpts;
    use serde_json::json;
    // `strict` is dropped by the template's own macro, as is `defer_loading`.
    let tools = json!([{"type": "function", "function": {
        "name": "wx",
        "description": "Weather",
        "strict": true,
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }}]);
    let turns = [
        ("user", "weather in Paris?"),
        (
            "assistant",
            "<tool_call>wx<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        ),
        ("observation", "<tool_response>18C</tool_response>"),
    ];
    let want = concat!(
        "[gMASK]<sop>",
        "<|system|>\n# Tools\n\n",
        "You may call one or more functions to assist with the user query.\n\n",
        "You are provided with function signatures within <tools></tools> XML tags:\n",
        "<tools>\n",
        r#"{"name": "wx", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}"#,
        "\n</tools>\n\n",
        "For each function call, output the function name and arguments within the following XML format:\n",
        "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>",
        "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>",
        "<|user|>weather in Paris?",
        "<|assistant|><think></think><tool_call>wx<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        "<|observation|><tool_response>18C</tool_response>",
        "<|assistant|><think></think>",
    );
    with_tokenizer(|tok| {
        let got = tok
            .decode_all(
                &tok.encode_chat_turns(
                    &turns,
                    &ChatOpts {
                        tools: Some(&tools),
                        ..Default::default()
                    },
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(got, want);
    });
}
