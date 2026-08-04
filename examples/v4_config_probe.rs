//! One-off: hand the shipped DeepSeek-V4-Flash-0731 `config.json` to the engine's own
//! loader and report exactly where it stops. Scratch diagnostic for the multi-model
//! port — delete once the port has a real gate.
//!
//! usage: cargo run --release --features rocm --example v4_config_probe -- <model-dir>

fn main() {
    let dir = std::env::args().nth(1).expect("usage: v4_config_probe <model-dir>");
    let path = format!("{dir}/config.json");
    let text = std::fs::read_to_string(&path).expect("read config");

    match rivoli::artifact::model::ModelConfig::load(&dir) {
        Ok(c) => println!("ACCEPTED: {c:#?}"),
        Err(e) => println!("REFUSED at the config boundary:\n  {e:#}"),
    }

    // Which required fields are actually absent, so the report is the whole list rather
    // than whichever one serde happened to name first.
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let required = [
        "num_hidden_layers", "hidden_size", "num_attention_heads",
        "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
        "n_routed_experts", "num_experts_per_tok", "moe_intermediate_size",
        "intermediate_size", "n_shared_experts", "first_k_dense_replace",
        "routed_scaling_factor", "norm_topk_prob", "vocab_size",
    ];
    let missing: Vec<_> = required.iter().filter(|k| v.get(*k).is_none()).collect();
    println!("\nrequired-by-ModelConfig fields absent from this config ({}):", missing.len());
    for m in &missing {
        println!("  {m}");
    }
}
