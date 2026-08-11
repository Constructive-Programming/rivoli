//! Every flag `run_glimmer` must refuse, refused by the SHIPPED binary.
//!
//! **Why this exists at all.** `Arch::hidden_flags` is consumed only for clap's `.hide(true)`,
//! so the parser still accepts every flag it hides — hiding a flag from `--help` is not the
//! same as rejecting it. A branch that omits a refusal compiles clean, passes clippy, and
//! silently takes a knob it cannot honour. Nothing but a test that runs the binary can tell
//! the difference.
//!
//! **Deviceless, and that is a property of where the bail sits**, not luck: the architecture
//! dispatch runs before any `DeviceTier` is built, so the whole file completes without
//! touching the GPU. If a future change moves the dispatch after engine construction these
//! tests start needing the flock, and they will say so by hanging on a busy machine.
//!
//! The artifact is `glimmer_convert`'s fixture, converted — the same one `glimmer_pin` uses.
//! A refusal test needs a checkpoint the binary accepts far enough to reach the refusal, and
//! a synthetic one is the only kind available (the real model is 59.553 GB and is not here).

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::glimmer_convert_fixture;

const DIM: usize = 8;

/// Run the shipped binary against the converted fixture with `extra` appended, and return
/// **stdout and stderr together**. `tracing`'s fmt layer writes to stdout and `anyhow`'s
/// top-level report to stderr, so the layer-map log and the bail that follows it land on
/// different streams — reading one of them gets half the run.
///
/// No `-bench`: `run_glimmer` never reads it, since the dispatch happens before anything
/// computes a token count. Passing it anyway would have been harmless for eight of the nine
/// rows below and wrong for the ninth — clap declares `--port` and `--bench` mutually
/// exclusive, so the run would die in the parser and the `--port` refusal would never be
/// reached. It did, on the first run of this test.
fn rivoli(artifact: &std::path::Path, extra: &[&str]) -> String {
    let o = std::process::Command::new(env!("CARGO_BIN_EXE_rivoli"))
        .arg(artifact)
        .args(extra)
        .output()
        .expect("run rivoli");
    assert!(
        !o.status.success(),
        "a Glimmer artifact must not decode yet"
    );
    String::from_utf8_lossy(&o.stdout).to_string() + &String::from_utf8_lossy(&o.stderr)
}

/// **Each inapplicable flag is refused, by name, with a reason.**
///
/// Asserted on the MESSAGE and not on `is_err`: the run fails for many reasons on this
/// artifact — it fails unconditionally, in fact, since Glimmer does not decode — so an
/// exit-code check would pass on every row without any refusal existing. That is the same
/// trap `glimmer_pin`'s refusal test hit for real on a busy GPU.
#[test]
fn every_flag_that_does_not_apply_is_refused_by_name() {
    let root = std::env::temp_dir().join(format!("glimmer-flags-{}", std::process::id()));
    let _ = glimmer_convert_fixture(&root, DIM);
    let art = root.join("out");

    // The flag, how a user sets it, and a distinctive fragment of the reason — so a message
    // rewritten into a different claim reddens rather than passing on the flag name alone.
    for (flag, args, reason) in [
        ("--attn", vec!["--attn", "dense"], "fixed in the weights"),
        ("--sinks", vec!["--sinks", "8"], "there is no --attn here"),
        ("--window", vec!["--window", "4096"], "2048 rows"),
        ("--misa-heads", vec!["--misa-heads", "4"], "no analogue of"),
        ("--mode", vec!["--mode", "int4"], "no second format to pick"),
        (
            "--cache-policy",
            vec!["--cache-policy", "lru"],
            "nothing to evict",
        ),
        (
            "--max-mem",
            vec!["--max-mem", "70"],
            "cannot run this artifact at any setting",
        ),
        (
            "--trace",
            vec!["--trace", "/dev/null"],
            "no routed experts to access",
        ),
        (
            "--port",
            vec!["--port", "8080"],
            "a signature, not an architectural fact",
        ),
    ] {
        let err = rivoli(&art, &args);
        assert!(
            err.contains(flag) && err.contains(reason),
            "{flag} must be refused with its reason, got:\n{err}"
        );
    }
    let _ = std::fs::remove_dir_all(&root);
}

/// **A bare run gets past every refusal and stops at the one honest bail.**
///
/// The other half of the gate, and the half that catches a refusal fired on the DEFAULT
/// value. A table of eight `bail!`s where one compares against the wrong default refuses a
/// user who passed nothing — which reads as "this model is broken" rather than "you passed a
/// flag it cannot honour", and no test above can see it.
#[test]
fn a_run_with_no_flags_reaches_the_decode_bail() {
    let root = std::env::temp_dir().join(format!("glimmer-bare-{}", std::process::id()));
    let _ = glimmer_convert_fixture(&root, DIM);
    let err = rivoli(&root.join("out"), &[]);
    assert!(
        err.contains("do not decode yet") && err.contains("S2"),
        "a bare run must reach the decode bail, got:\n{err}"
    );
    assert!(
        !err.contains("does not apply"),
        "no flag refusal may fire on a run that passed no flags:\n{err}"
    );
    // The layer map is logged BEFORE the bail — the evidence that the config parsed rather
    // than that the binary recognised a directory name. 3 of the fixture's 4 layers slide,
    // one period of the shipped `[sliding, sliding, sliding, full]` pattern.
    assert!(
        err.contains("4 layers (3 sliding at window 2 / 1 full)") && err.contains("DENSE"),
        "the layer map must be reported before the bail:\n{err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}

/// **A manifest contradicting a load-bearing field refuses AT STARTUP, from the binary.**
///
/// G1a asks for this "proven by feeding it one", and the 26-row defect run in
/// `artifact/model.rs` proves it about `GlimmerConfig` — a unit test on a type. This proves
/// it about the *shipped binary*, which is a different claim: it is what fails if the
/// dispatch ever bails before parsing, and that ordering is a comment today rather than a
/// gate.
///
/// The defect is the pairing invariant, because it is the one no shape check anywhere
/// downstream could catch: `layer_types` and `layer_rope_theta` are independent arrays, a
/// layer is rotated IFF it slides, and a disagreement rotates a NoPE layer with plausible
/// frequencies and decodes fluently.
#[test]
fn a_manifest_that_contradicts_itself_refuses_at_startup() {
    let root = std::env::temp_dir().join(format!("glimmer-badcfg-{}", std::process::id()));
    let _ = glimmer_convert_fixture(&root, DIM);
    let manifest = root.join("out").join("manifest.json");

    // Layer 0 is `sliding_attention` and therefore rotated. Zero its theta and it claims to
    // slide without rotating — which is a state no Muse Glimmer layer is in.
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest).unwrap()).unwrap();
    cfg["text_config"]["layer_rope_theta"][0] = serde_json::json!(0.0);
    std::fs::write(&manifest, serde_json::to_vec_pretty(&cfg).unwrap()).unwrap();

    let err = rivoli(&root.join("out"), &[]);
    assert!(
        err.contains("layer 0 is SlidingAttention") && err.contains("rotated IFF it slides"),
        "the refusal must name the layer and the invariant, got:\n{err}"
    );
    assert!(
        !err.contains("do not decode yet"),
        "the config refusal must come BEFORE the decode bail:\n{err}"
    );
    let _ = std::fs::remove_dir_all(&root);
}
