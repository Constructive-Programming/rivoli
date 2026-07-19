//! MTP M1 gate: the scalar draft-head oracle drafts the token the main model
//! actually emits next, at a sane accept rate. Slow (reference scalar decode),
//! so `#[ignore]`d — run explicitly:
//!   cargo test --features rocm --test mtp -- --ignored --nocapture
//! Needs the snapshot WITH the out-mtp shard (RIVOLI_SNAPSHOT or ~/glm52-snap).

use rivoli::attn::AttnMode;
use rivoli::engine::Engine;
use rivoli::model::ModelConfig;
use rivoli::mtp::Mtp;
use rivoli::snapshot::Snapshot;
use rivoli::tokenizer::Tokenizer;

fn snapshot_dir() -> Option<String> {
    let dir = std::env::var("RIVOLI_SNAPSHOT")
        .unwrap_or_else(|_| format!("{}/glm52-snap", std::env::var("HOME").unwrap_or_default()));
    std::path::Path::new(&dir).is_dir().then_some(dir)
}

#[test]
#[ignore = "slow: reference scalar decode + MTP draft per step"]
fn mtp_draft_accepts_main_model_tokens() -> anyhow::Result<()> {
    let Some(dir) = snapshot_dir() else {
        eprintln!("skipping: no snapshot");
        return Ok(());
    };
    let snap = Snapshot::open(&dir)?;
    let cfg = ModelConfig::load(&dir)?;
    if snap
        .bf16("model.layers.78.eh_proj.weight", 2 * cfg.hidden)
        .is_err()
    {
        eprintln!("skipping: snapshot has no out-mtp shard");
        return Ok(());
    }
    let tok = Tokenizer::load(&dir)?;

    let mut engine = Engine::new(&snap, &cfg, AttnMode::Dense, false)?;
    let mut mtp = Mtp::new(&snap, &cfg)?;

    // Short run — the reference path is slow. `toks` accumulates the main
    // model's own greedy continuation, so drafts are scored against the model's
    // real next-next tokens.
    let ngen = 6usize;
    let mut toks = tok.encode("The capital of France is")?;
    let prompt_len = toks.len();

    // draft[i] = MTP's prediction for the token at position i+2.
    let mut drafts: Vec<(usize, u32)> = Vec::new();
    let steps = prompt_len + ngen;
    for i in 0..steps {
        if i >= toks.len() {
            break;
        }
        let pred = engine.step(toks[i], i)?; // predicts token i+1
        // the token at position i+1 (teacher-forced in the prompt, else greedy)
        let next = if i + 1 < toks.len() {
            toks[i + 1]
        } else {
            toks.push(pred);
            pred
        };
        let d = mtp.draft(&snap, &cfg, engine.trunk(), next, i)?; // drafts token i+2
        drafts.push((i + 2, d));
    }

    let scored: Vec<(u32, u32)> = drafts
        .iter()
        .filter(|(t, _)| *t < toks.len())
        .map(|(t, d)| (toks[*t], *d))
        .collect();
    let hits = scored.iter().filter(|(want, got)| want == got).count();
    let total = scored.len();
    let rate = hits as f64 / total.max(1) as f64;
    eprintln!(
        "MTP draft accept rate: {hits}/{total} = {rate:.2}  (drafts vs main-model tokens)\n\
         continuation: {:?}",
        tok.decode_all(&toks[prompt_len..]).unwrap_or_default()
    );
    assert!(total >= 4, "too few scored drafts ({total})");
    assert!(
        rate > 0.5,
        "MTP draft accept rate {rate:.2} below the 0.5 M1 gate"
    );
    Ok(())
}
