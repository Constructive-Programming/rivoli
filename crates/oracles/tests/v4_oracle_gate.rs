//! **The gate's own plumbing, proved able to go red.**
//!
//! Everything the defect matrix is BUILT ON but does not itself test: the comparator, the
//! golden file's writer/reader pair, `Capture::push`'s duplicate-name guard, the
//! safetensors reader that opens the 167 GB checkpoint, `window_topk`, and the `--defect`
//! flag parser. Each of these can fail open, and a matrix run on a comparator that never
//! says "different" is 33 green tests about nothing.
//!
//! Split out of `v4_oracle.rs` on 2026-08-15 for the 800-line ceiling; the family's
//! orientation stays in that file's header and the shared toy driver in
//! `common/oracle_probe.rs`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "common/oracle_probe.rs"]
mod oracle_probe;

use oracle_probe::run;
use rivoli_oracles::golden::{GoldenSet, identical};
use rivoli_oracles::v4oracle::forward::{Capture, Defect};

#[test]
fn the_comparator_itself_can_go_red() {
    // Proving a gate green is worthless until you have seen it red. Three ways a naive
    // comparator fails open: identical inputs, a one-ulp change, and a missing tensor.
    let a = run(2, 12, Defect::None);
    assert!(identical(&a.pre, &a.pre), "a capture must equal itself");

    let mut b = a.pre.clone();
    // By NAME: an index into `floats` is an ordering detail, and a reorder would silently
    // move this probe onto a different tensor (or an empty one, which `v[0]` would panic on).
    let (_, _, v) = b
        .floats
        .iter_mut()
        .find(|(n, _, _)| n.ends_with(".q"))
        .expect("a .q golden");
    v[0] = f32::from_bits(v[0].to_bits() ^ 1);
    assert!(!identical(&a.pre, &b), "a one-ulp change went undetected");

    let mut c = a.pre.clone();
    c.floats.remove(0);
    assert!(!identical(&a.pre, &c), "a DELETED golden read as agreement");

    let mut e = a.pre.clone();
    e.floats[0].2.push(0.0);
    assert!(!identical(&a.pre, &e), "a length change read as agreement");

    let mut f = a.pre.clone();
    let (_, shape, _) = &mut f.floats[0];
    *shape = vec![shape.iter().product()];
    assert!(
        !identical(&a.pre, &f),
        "a RESHAPE with identical values read as agreement"
    );
}

#[test]
fn the_safetensors_reader_rejects_malformed_headers() {
    // The one component that reads the 167 GB checkpoint, and the only thing that exercised
    // it was a binary that needs the checkpoint present. These are synthetic files.
    let dir = std::env::temp_dir().join(format!("v4-oracle-st-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let write = |name: &str, hdr: &str, body: &[u8]| {
        let mut v = (hdr.len() as u64).to_le_bytes().to_vec();
        v.extend_from_slice(hdr.as_bytes());
        v.extend_from_slice(body);
        std::fs::write(dir.join(name), v).unwrap();
    };
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        r#"{"weight_map":{"ok":"a.st","short":"b.st","backwards":"c.st","past_end":"d.st"}}"#,
    )
    .unwrap();
    write(
        "a.st",
        r#"{"ok":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &1.0f32
            .to_le_bytes()
            .iter()
            .chain(2.0f32.to_le_bytes().iter())
            .copied()
            .collect::<Vec<u8>>(),
    );
    // shape [2] F32 needs 8 bytes; the header claims 4.
    write(
        "b.st",
        r#"{"short":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#,
        &[0u8; 4],
    );
    // data_offsets reversed -- `b - a` would WRAP in release.
    write(
        "c.st",
        r#"{"backwards":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
        &[0u8; 8],
    );
    // well-formed header, truncated body: the shard is still downloading.
    write(
        "d.st",
        r#"{"past_end":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0u8; 4],
    );

    let ck = rivoli_oracles::v4oracle::weights::Checkpoint::open(&dir).unwrap();
    assert_eq!(ck.get("ok").unwrap().to_f32().unwrap(), vec![1.0, 2.0]);
    assert!(ck.has_prefix("o") && !ck.has_prefix("zz"));
    for bad in ["short", "backwards", "past_end"] {
        assert!(ck.get(bad).is_err(), "{bad} was accepted");
    }
    assert!(ck.get("absent").is_err());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn window_topk_matches_the_reference_by_hand() {
    // `model.py::get_window_topk_idxs` (lines 260-271), transcribed by hand from the Python
    // rather than from the Rust. The middle branch (0 < start_pos < window_size - 1, which
    // pads with -1) is the one a port forgets, and the grid reaches it only incidentally.
    let w = rivoli_oracles::v4oracle::forward::window_topk;
    // prefill, seqlen 5, window 8: causal, row t attends [max(0,t-7), t], -1 beyond.
    let short = w(8, 5, 0);
    assert_eq!(short[0], vec![0, -1, -1, -1, -1]);
    assert_eq!(short[4], vec![0, 1, 2, 3, 4]);
    // prefill, seqlen 12, window 8: row 11 sees positions 4..=11, and row 3 -- the middle
    // branch, 0 < start < window - 1 -- pads.
    let long = w(8, 12, 0);
    assert_eq!(long[11], vec![4, 5, 6, 7, 8, 9, 10, 11]);
    assert_eq!(long[3], vec![0, 1, 2, 3, -1, -1, -1, -1]);
    // decode inside the first window: F.pad(arange(sp+1), (0, win-sp-1), value=-1); and past
    // it: cat([arange(sp%win+1, win), arange(0, sp%win+1)]) -- oldest first.
    let (inside, past) = (w(8, 1, 2), w(8, 1, 9));
    assert_eq!(inside, vec![vec![0, 1, 2, -1, -1, -1, -1, -1]]);
    assert_eq!(past, vec![vec![2, 3, 4, 5, 6, 7, 0, 1]]);
}

#[test]
fn the_golden_file_survives_a_round_trip() {
    // The writer and the reader are the only two halves of the format, and nothing else in
    // the tree exercises the reader -- so without this they could disagree on a length
    // prefix forever and every other test would still pass. S2 loads goldens through
    // `GoldenSet::read`.
    let cap = run(2, 12, Defect::None).pre;
    let want = GoldenSet::from_capture(vec![("k".to_string(), "v".to_string())], cap.clone());
    let mut buf = Vec::new();
    want.write(&mut buf).unwrap();
    let got = GoldenSet::read(&mut buf.as_slice()).unwrap();
    assert_eq!(got.meta, want.meta);
    assert_eq!(got.floats, want.floats);
    // Non-empty on both sides, or the three equalities above are truths about empty vectors.
    let carried = !want.floats.is_empty() && !want.ints.is_empty();
    assert_eq!(got.ints, want.ints);
    assert!(carried, "the round trip carried nothing");
    // ...and it must reject something that is not a golden file, or the magic is decoration.
    let junk = GoldenSet::read(&mut b"not a golden".as_slice());
    assert!(junk.is_err());
}

#[test]
fn a_duplicate_golden_name_is_rejected() {
    // `Capture::float` returns the FIRST match, so a duplicate name silently shadows every
    // later tensor of that name -- which is what a four-layer emit did before `run_layer`
    // namespaced by layer. Proving the guard fires is the point; a guard nobody has seen go
    // red is not a guard.
    let mut c = Capture::default();
    c.push("x", &[2], vec![1.0, 2.0]);
    assert!(std::panic::catch_unwind(move || c.push("x", &[2], vec![3.0, 4.0])).is_err());
    let mut c = Capture::default();
    assert!(
        std::panic::catch_unwind(move || c.push("y", &[3], vec![1.0])).is_err(),
        "shape/len"
    );
}

/// The `--defect` flag's parser, both directions -- `v4-oracle emit` trusts it to make a
/// typo IMPOSSIBLE to mistake for `None`, because a silent fallback would emit two
/// identical goldens and an A/B that cannot fail.
#[test]
fn defect_from_flag_roundtrips_and_refuses_loudly() {
    // `ALL` includes `None` first, so this also proves `--defect None` is spellable --
    // the base arm of an A/B goes through the same code path as omitting the flag.
    for &d in Defect::ALL {
        assert_eq!(Defect::from_flag(&format!("{d:?}")), Ok(d));
    }
    // The refusal must LIST every variant -- the message is how a caller discovers the
    // spelling, and a list that silently dropped one would make that variant unreachable
    // from the command line in practice.
    let err = Defect::from_flag("RopeHalfSplt").expect_err("a typo must refuse");
    for &d in Defect::ALL {
        assert!(
            err.contains(&format!("{d:?}")),
            "the refusal must list {d:?}: {err}"
        );
    }
    // Exact match only. Forgiving case would put every name one typo away from another,
    // and the cost of strictness is one re-run with the spelling the error just printed.
    Defect::from_flag("ropehalfsplit").expect_err("case must not be forgiven");
}
