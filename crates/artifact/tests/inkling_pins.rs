//! Inkling 2's vendored evidence, recomputed — the pin and census-closure gate (S1).
//!
//! **Why this file exists.** Census item 1's closing condition is that a gate RECOMPUTES FNV-1a
//! from the live vendored bytes and byte-compares, rather than trusting a frozen copy of a hash
//! its author typed. `qwen_names.rs` records the scar: a hand-off retyped a config's fnv1a WRONG
//! while claiming the values were not being retyped. This file is the same recompute for both
//! Inkling members, plus the census closure the TSV exists to make checkable — every tensor in
//! exactly one family row, the byte totals accounting for the shard payload, and the text/vision
//! split the converter's by-name exclusion will be held to at S2.
//!
//! Every expected number is read out of the TSV's OWN header (`census::Tsv`, the shared reader —
//! not a fourth parser), so there is one copy of each rather than two agreeing frozen ones, and a
//! moved header phrase panics rather than defaulting.
//!
//! The conditional cell reads ONLY the safetensors header of the local Mini shard (8-byte LE
//! length, then that many JSON bytes) — never the 19.5 GB payload — and skips unless
//! `RIVOLI_INKLING_REQUIRED` is set, on `k3_tokenizer.rs`'s policy. No GPU, no network.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli_artifact::census::Tsv;
use rivoli_core::hash::fnv1a;

const FAMILIES: Tsv = Tsv::new(
    "docs/measurement/inkling2-reference/mini/tensor-families.tsv",
    include_str!("../../../docs/measurement/inkling2-reference/mini/tensor-families.tsv"),
);

// Bytes, not `&str`: the pins are over the FILE. The three Small tokenizer files are asserted
// byte-identical to Mini's below rather than pinned twice over — one fact, one place.
const MINI_CONFIG: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/config.json");
const MINI_GENERATION: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/generation_config.json");
const MINI_QUANT: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/hf_quant_config.json");
const MINI_PROCESSOR: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/processor_config.json");
const MINI_CHAT: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/chat_template.jinja");
const MINI_SPECIALS: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/special_tokens_map.json");
const MINI_TOKENIZER: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/mini/tokenizer_config.json");
const SMALL_CONFIG: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/small-mxfp8/config.json");
const SMALL_GENERATION: &[u8] = include_bytes!(
    "../../../docs/measurement/inkling2-reference/small-mxfp8/generation_config.json"
);
const SMALL_QUANT: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/small-mxfp8/hf_quant_config.json");
const SMALL_PROCESSOR: &[u8] = include_bytes!(
    "../../../docs/measurement/inkling2-reference/small-mxfp8/processor_config.json"
);
const SMALL_CHAT: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/small-mxfp8/chat_template.jinja");
const SMALL_SPECIALS: &[u8] = include_bytes!(
    "../../../docs/measurement/inkling2-reference/small-mxfp8/special_tokens_map.json"
);
const SMALL_TOKENIZER: &[u8] = include_bytes!(
    "../../../docs/measurement/inkling2-reference/small-mxfp8/tokenizer_config.json"
);
const TIKTOKEN: &[u8] =
    include_bytes!("../../../docs/measurement/inkling2-reference/tiktoken.model");

/// (header anchor, live bytes) for every vendored file the TSV pins. The anchor is the relative
/// path exactly as the header spells it, so `header_token` finds the one line that starts it.
const PINS: &[(&str, &[u8])] = &[
    ("mini/config.json", MINI_CONFIG),
    ("mini/generation_config.json", MINI_GENERATION),
    ("mini/hf_quant_config.json", MINI_QUANT),
    ("mini/processor_config.json", MINI_PROCESSOR),
    ("mini/chat_template.jinja", MINI_CHAT),
    ("mini/special_tokens_map.json", MINI_SPECIALS),
    ("mini/tokenizer_config.json", MINI_TOKENIZER),
    ("small-mxfp8/config.json", SMALL_CONFIG),
    ("small-mxfp8/generation_config.json", SMALL_GENERATION),
    ("small-mxfp8/hf_quant_config.json", SMALL_QUANT),
    ("small-mxfp8/processor_config.json", SMALL_PROCESSOR),
    ("small-mxfp8/chat_template.jinja", SMALL_CHAT),
    ("small-mxfp8/special_tokens_map.json", SMALL_SPECIALS),
    ("small-mxfp8/tokenizer_config.json", SMALL_TOKENIZER),
    ("tiktoken.model", TIKTOKEN),
];

/// Length and FNV-1a over the live bytes, both read from the header. The sha256 beside each pin
/// is deliberately NOT recomputed: `core/src/hash.rs` declined a `sha2` dependency to pin a
/// fixture, and the sha is there for a human re-fetching with `curl` and `sha256sum`.
fn assert_pin(anchor: &str, bytes: &[u8]) {
    let want_len: usize = FAMILIES
        .header_token(anchor, "bytes ")
        .expect("header bytes pin")
        .parse()
        .expect("declared bytes is an integer");
    assert_eq!(bytes.len(), want_len, "{anchor}: length");
    let want_hash = FAMILIES
        .header_token(anchor, "fnv1a64 ")
        .expect("header fnv1a64 pin");
    assert_eq!(
        format!("{:016x}", fnv1a(bytes)),
        want_hash,
        "{anchor}: FNV-1a over the live bytes does not match the pin. Recompute it — do NOT \
         retype a hash out of a report"
    );
}

#[test]
fn the_vendored_files_hash_as_the_header_records() {
    // The examined count, asserted: an empty PINS would leave the loop vacuously green.
    assert_eq!(PINS.len(), 15, "vendored files pinned");
    for (anchor, bytes) in PINS {
        assert_pin(anchor, bytes);
    }
}

#[test]
fn the_small_member_shares_mini_s_tokenizer_bytes() {
    // One tokenizer for the port (the rank file is vendored once at the parent); the members'
    // chat template, specials map and tokenizer config are the same bytes. Asserted, not assumed
    // from equal hashes — a hash collision is not identity, and these are small enough to compare.
    assert_eq!(SMALL_CHAT, MINI_CHAT, "chat_template.jinja");
    assert_eq!(SMALL_SPECIALS, MINI_SPECIALS, "special_tokens_map.json");
    assert_eq!(SMALL_TOKENIZER, MINI_TOKENIZER, "tokenizer_config.json");
}

/// A count the header declares as `# <label>  <number>`, unwrapped for the gate.
fn declared(label: &str) -> u64 {
    FAMILIES.declared(label).unwrap()
}

/// A header token parsed as `u64` — one place for the annotation the asserts need, so the four
/// status tallies are four one-line calls rather than four cloned five-line blocks (jscpd).
fn token(anchor: &str, keyword: &str) -> u64 {
    FAMILIES
        .header_token(anchor, keyword)
        .unwrap()
        .parse::<u64>()
        .unwrap()
}

/// The census closure, recomputed from the rows rather than read off the prose `EQUAL: True`.
/// Columns: status family example shape dtype count bytes-per-tensor total-bytes.
#[test]
fn the_census_closure_recomputes_from_its_own_rows() {
    let rows = FAMILIES.rows(8).expect("the census parses");
    assert!(!rows.is_empty(), "examined rows");
    assert_eq!(
        rows.len() as u64,
        declared("family rows"),
        "family row count"
    );

    let mut tensors = 0u64;
    let mut payload = 0u64;
    let mut text_tensors = 0u64;
    let mut text_bytes = 0u64;
    let mut vision_tensors = 0u64;
    let mut vision_bytes = 0u64;
    for r in &rows {
        let status = r.str_at(0).unwrap();
        let count = r.u64_at(5, "count").unwrap();
        let per = r.u64_at(6, "bytes-per-tensor").unwrap();
        let total = r.u64_at(7, "total-bytes").unwrap();
        // Homogeneity: every row's total is its count times its per-tensor bytes. A row that
        // mixes shapes would break this, which is the split-by-(dtype,shape) contract.
        assert_eq!(count * per, total, "row total: {}", r.str_at(1).unwrap());
        tensors += count;
        payload += total;
        match status {
            "TEXT" => {
                text_tensors += count;
                text_bytes += total;
            }
            "EXCLUDED-VISION" => {
                vision_tensors += count;
                vision_bytes += total;
            }
            other => panic!("unknown census status {other:?}"),
        }
    }
    // Nonzero examined counts per status, so a filter that matched nothing cannot pass.
    assert!(
        text_tensors > 0 && vision_tensors > 0,
        "both statuses examined"
    );

    assert_eq!(tensors, declared("tensors in the shard header"));
    assert_eq!(payload, declared("payload bytes (header data_offsets)"));
    assert_eq!(text_tensors, token("STATUS TEXT ", "tensors "));
    assert_eq!(text_bytes, token("STATUS TEXT ", "bytes "));
    assert_eq!(vision_tensors, token("STATUS EXCLUDED-VISION ", "tensors "));
    assert_eq!(vision_bytes, token("STATUS EXCLUDED-VISION ", "bytes "));
    // The text + vision split accounts for the whole shard — no tensor unclassified.
    assert_eq!(text_tensors + vision_tensors, tensors);
    assert_eq!(text_bytes + vision_bytes, payload);
}

/// The live Mini shard's header agrees with the census: tensor count, payload span, and the
/// file-size identity `8 + header_len + payload == file size`. Reads ONLY the header.
#[test]
fn the_live_shard_header_matches_the_census() {
    use std::io::Read;
    let path = "/swarm/storage/ai/rivoli/inkling-2-mini-src/model-00001-of-00001.safetensors";
    let Ok(meta) = std::fs::metadata(path) else {
        assert!(
            std::env::var_os("RIVOLI_INKLING_REQUIRED").is_none(),
            "Inkling shard gate REQUIRED but did not run: {path} absent"
        );
        eprintln!("skip live shard header: {path} absent (set RIVOLI_INKLING_REQUIRED=1 to force)");
        return;
    };
    let mut f = std::fs::File::open(path).expect("open shard");
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).expect("8-byte header length");
    let header_len = u64::from_le_bytes(len_buf);
    let mut header = vec![0u8; header_len as usize];
    f.read_exact(&mut header).expect("header JSON");
    let json: serde_json::Value = serde_json::from_slice(&header).expect("header is JSON");
    let obj = json.as_object().expect("header is an object");

    let mut tensors = 0u64;
    let mut payload = 0u64;
    for (name, v) in obj {
        if name == "__metadata__" {
            continue;
        }
        tensors += 1;
        let offsets = v["data_offsets"].as_array().expect("data_offsets");
        payload = payload.max(offsets[1].as_u64().expect("offset end"));
    }
    assert!(tensors > 0, "examined tensors");
    assert_eq!(tensors, declared("tensors in the shard header"));
    assert_eq!(payload, declared("payload bytes (header data_offsets)"));
    assert_eq!(
        8 + header_len + payload,
        meta.len(),
        "8 + header_len + payload == file size"
    );
    assert_eq!(
        meta.len(),
        token("model-00001-of-00001.safetensors", "bytes "),
        "file size matches the shard pin"
    );
}
