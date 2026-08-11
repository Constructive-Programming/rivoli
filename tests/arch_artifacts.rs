//! **Every architecture's real artifact still opens.** The regression half of each port's
//! G1a: adding a fourth architecture must not change how the other three load.
//!
//! `src/artifact/model.rs` already asserts that the four schemas do not cross-parse, on
//! vendored config text. This is the other direction and the one that needs the filesystem:
//! `arch_of_artifact` resolves the discriminant from a real manifest, and `load_config`
//! re-runs that architecture's whole `validate` against it. Between them they cover the two
//! ways a new arch breaks an old one — a recogniser that now claims a manifest it should not,
//! and a `validate` that got stricter for everybody.
//!
//! **What it does when the artifacts are not there.** These are checkpoints on one machine,
//! not fixtures — CI's runner has none, and neither does a fresh clone. So:
//!
//! - **No artifact at all** → this machine cannot run the check. Print and return.
//! - **Some present** → every one that IS present must open, *and* the count must equal
//!   [`EXPECTED_PRESENT`].
//!
//! > **CORRECTED 2026-08-11**, by review, on two counts. The first version asserted
//! > `opened > 0` unconditionally, which **reddens CI** — the only job this repo has runs
//! > featureless on a runner where `/var/db/rivoli` does not exist. And its "skips loudly"
//! > claim was false: libtest captures stdout and stderr of PASSING tests, so the `SKIP` line
//! > is visible only under `--nocapture` and a run that silently degraded from two
//! > architectures to one looked identical to a full one. `EXPECTED_PRESENT` is what makes
//! > that degradation red, and it is a literal rather than a count of what happened to be
//! > there, because a number derived from the run cannot disagree with the run.
//!
//! Paths are overridable so this is not pinned to one layout.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::arch::Arch;
use rivoli::artifact::model as gm;

/// `(env override, default path, the architecture it must resolve to)`.
///
/// **The names and defaults are `tests/common/f4_artifact_dir.rs`'s, not new ones.** This file
/// first invented `RIVOLI_GLM_ARTIFACT` and `RIVOLI_V4_ARTIFACT` for the same two directories —
/// which were already taken, by that module, for DIFFERENT artifacts (`RIVOLI_V4_ARTIFACT` is
/// the 3-layer `v4-f4-l0-2`, not the 43-layer full set). Two tests reading one variable with
/// two meanings is a trap that fires only for whoever sets it, and it fired here on the rebase
/// onto `wt/k3-s1a`: `f4_loading` treats an explicitly-set-but-unresolvable value as a FAILURE
/// rather than a skip — deliberately, for the same libtest-captures-stderr reason this file
/// records — so pointing the name at nothing to simulate CI broke a test that was not mine.
///
/// K3's row is the only new name, and there is no artifact for it yet.
const ARTIFACTS: [(&str, &str, Arch); 3] = [
    (
        "RIVOLI_GLM_ARTIFACT_FULL",
        "/var/db/rivoli/glm52-vq3-full",
        Arch::GlmMoeDsa,
    ),
    (
        "RIVOLI_V4_ARTIFACT_FULL",
        "/var/db/rivoli/v4-f4-full",
        Arch::DeepseekV4,
    ),
    ("RIVOLI_K3_ARTIFACT", "/var/db/rivoli/k3-full", Arch::KimiK3),
];

/// How many of [`ARTIFACTS`] exist on a machine that has any. **Two on this one: GLM and V4.**
/// K3's converter landed at its own S1a and no `.k3` artifact has been built yet — so its row
/// is coverage this test does not have, and the number says so rather than the prose.
///
/// When a K3 artifact appears, this becomes 3 and the test goes red until it does — which is
/// the point: an architecture gaining coverage should require a deliberate edit, in the same
/// way losing it does.
const EXPECTED_PRESENT: usize = 2;

#[test]
fn every_architecture_still_resolves_and_validates_its_own_artifact() {
    let mut opened = 0usize;
    let mut absent: Vec<&str> = Vec::new();
    for (var, default, want) in ARTIFACTS {
        let dir = std::env::var(var).unwrap_or_else(|_| default.to_string());
        if !std::path::Path::new(&dir).is_dir() {
            absent.push(want.name());
            continue;
        }
        opened += 1;
        let got = gm::arch_of_artifact(&dir)
            .unwrap_or_else(|e| panic!("{dir}: arch_of_artifact refused it: {e:#}"));
        assert_eq!(got, want, "{dir} resolved to the wrong architecture");
        // Not just the discriminant — the whole config, so a `validate` that tightened for
        // one architecture and accidentally for all of them fails here. Each arm names its
        // own type because that IS the thing under test; a generic helper would need the
        // type as a parameter and prove nothing extra.
        let err = match want {
            Arch::GlmMoeDsa => gm::load_config::<gm::ModelConfig>(&dir).err(),
            Arch::DeepseekV4 => gm::load_config::<gm::V4Config>(&dir).err(),
            Arch::KimiK3 => gm::load_config::<gm::K3Config>(&dir).err(),
            Arch::MuseGlimmer => gm::load_config::<gm::GlimmerConfig>(&dir).err(),
        };
        assert!(
            err.is_none(),
            "{dir} no longer loads as {}: {:#}",
            want.name(),
            err.unwrap()
        );
    }
    if opened == 0 {
        // Not a failure: a machine with no checkpoints (CI, a fresh clone) cannot run this
        // check at all, and reddening there would make a permanently-red gate nobody reads.
        eprintln!(
            "SKIP arch_artifacts: none of {:?} present — this run asserted NOTHING. Set one \
             of {:?} to point at a real artifact.",
            ARTIFACTS.map(|(_, d, _)| d),
            ARTIFACTS.map(|(v, _, _)| v)
        );
        return;
    }
    // Some were present, so this machine DOES have checkpoints — and then the count is a
    // claim about coverage. `opened > 0` would have been satisfied by one, and a run covering
    // one architecture is indistinguishable from a run covering all of them in libtest's
    // default output.
    assert_eq!(
        opened,
        EXPECTED_PRESENT,
        "{opened} of {} architectures' artifacts were present ({absent:?} missing). Either \
         coverage was lost, or one arrived and EXPECTED_PRESENT needs updating — both want a \
         deliberate edit.",
        ARTIFACTS.len()
    );
}
