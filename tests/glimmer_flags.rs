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

// **Gated on the backend, and that is not because it needs one.** Nothing here touches the
// GPU. But it runs the shipped `rivoli` binary, and a featureless build of that binary is a
// refusal stub — `main` bails with "built with NO compute backend" before it ever reaches the
// architecture dispatch. So without this gate every test below fails under CI's featureless
// `cargo test --release --locked`, which is the ONE job this repo has. Caught by review
// 2026-08-11, after the file shipped green on a `--features rocm` machine.
#![cfg(feature = "rocm")]
#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

mod common;
use common::{GLIMMER_FIXTURE_DIM as DIM, TempRoot, glimmer_convert_fixture};

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

/// **`--max-mem` is ACCEPTED, and the partition it implies is REPORTED.**
///
/// The opposite assertion to the table below, and a separate test because "is refused, naming
/// its reason" and "is accepted, and the run then reports and reaches the honest bail" are
/// different shapes.
///
/// The `residency:` assertion is the half review found missing: the first version checked only
/// that the flag parsed, so nothing in the tree asserted that `run_glimmer` prints the split —
/// which is the one thing R1 gives an operator.
#[test]
fn max_mem_is_accepted_and_reports_the_partition() {
    // **`--max-mem` moved from refused to ACCEPTED at R1**, so this asserts the OPPOSITE of
    // what the table below asserts for every other residency flag. It is a separate test
    // rather than a table row because "is refused, naming its reason" and "is accepted, and
    // the run then reaches the honest bail" are different shapes.
    //
    // 8 GB against a 4-layer fixture is far above the whole model, so the partition is
    // all-resident — the point here is that the flag PARSES and the run proceeds, not what
    // the split is. `glimmer_residency.rs` gates the arithmetic at every boundary.
    let root = TempRoot::new("glimmer-maxmem");
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let err = rivoli(&root.join("out"), &["--max-mem", "8"]);
    assert!(
        !err.contains("--max-mem does not apply"),
        "--max-mem must be accepted for a Glimmer artifact since R1, got:\n{err}"
    );
    assert!(
        err.contains("do not decode yet"),
        "with --max-mem accepted the run must reach the honest decode bail, got:\n{err}"
    );
    assert!(
        err.contains("residency:") && err.contains("layers pinned"),
        "the run must report the partition the budget implies, got:\n{err}"
    );
}

/// **Each inapplicable flag is refused, by name, with a reason.**
///
/// Asserted on the MESSAGE and not on `is_err`: the run fails for many reasons on this
/// artifact — it fails unconditionally, in fact, since Glimmer does not decode — so an
/// exit-code check would pass on every row without any refusal existing. That is the same
/// trap `glimmer_pin`'s refusal test hit for real on a busy GPU.
#[test]
fn every_flag_that_does_not_apply_is_refused_by_name() {
    let root = TempRoot::new("glimmer-flags");
    let _ = glimmer_convert_fixture(root.path(), DIM);
    let art = root.join("out");

    // The flag, how a user sets it, and a distinctive fragment of the reason — so a message
    // rewritten into a different claim reddens rather than passing on the flag name alone.
    for (flag, args, reason) in [
        ("--attn", vec!["--attn", "dense"], "fixed in the weights"),
        ("--sinks", vec!["--sinks", "8"], "there is no --attn here"),
        (
            "--window",
            vec!["--window", "4096"],
            // Not the row COUNT: the message reports this artifact's own `sliding_window`,
            // which is 2 in the fixture and 2048 in the shipped model.
            "property of how the weights were trained",
        ),
        ("--misa-heads", vec!["--misa-heads", "4"], "no analogue of"),
        ("--mode", vec!["--mode", "int4"], "no second format to pick"),
        (
            // The reason CHANGED at R1 and the fragment is chosen to hold the new claim:
            // "nothing to evict" was false once the budget could leave layers streaming, and
            // a test matching only the flag name would have passed the rewrite either way.
            "--cache-policy",
            vec!["--cache-policy", "lru"],
            "no policy left to choose",
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

    // **The same flags again, at clap's own default value.** This is the row that could not
    // pass before 2026-08-11: the refusals compared VALUES, so `--cache-policy 2q` on a model
    // with no cache and `--mode hybrid` on a model with no experts were accepted — and
    // `tests/mode-matrix.sh` passes exactly those explicitly, so it is the ordinary case, not
    // an exotic one. "Was this flag typed" and "does it hold a non-default value" are
    // different questions, and only the first is the one a refusal wants to ask. Found by
    // review, in the gap between the loop above (non-default values only) and the bare run
    // below (no flags at all).
    for (flag, args) in [
        ("--attn", vec!["--attn", "auto"]),
        ("--sinks", vec!["--sinks", "4"]),
        ("--window", vec!["--window", "8192"]),
        ("--misa-heads", vec!["--misa-heads", "8"]),
        ("--mode", vec!["--mode", "hybrid"]),
        ("--cache-policy", vec!["--cache-policy", "2q"]),
        ("--moe-gain", vec!["--moe-gain", "1.0"]),
    ] {
        let err = rivoli(&art, &args);
        assert!(
            err.contains(flag) && err.contains("does not apply"),
            "{flag} at its DEFAULT value must still be refused — it was typed, and that is \
             what the recorded command line will show. Got:\n{err}"
        );
    }
}

/// **A bare run gets past every refusal and stops at the one honest bail.**
///
/// The other half of the gate, and the half that catches a refusal fired on the DEFAULT
/// value. A table of nine `bail!`s where one asks the wrong question refuses a
/// user who passed nothing — which reads as "this model is broken" rather than "you passed a
/// flag it cannot honour", and no test above can see it.
#[test]
fn a_run_with_no_flags_reaches_the_decode_bail() {
    let root = TempRoot::new("glimmer-bare");
    let _ = glimmer_convert_fixture(root.path(), DIM);
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
    let root = TempRoot::new("glimmer-badcfg");
    let _ = glimmer_convert_fixture(root.path(), DIM);
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
}
