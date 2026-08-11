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
//! **Skips loudly.** These are checkpoints on one machine, not fixtures; a test that opened
//! nothing and printed a green line is the failure mode this repo has been bitten by, so an
//! absent artifact prints SKIP and the presence of at least one is asserted at the end.
//! Paths are overridable so this is not pinned to one layout.

#![allow(clippy::unwrap_used, clippy::expect_used)] // tests: panic-on-failure is the idiom

use rivoli::arch::Arch;
use rivoli::artifact::model as gm;

/// `(env override, default path, the architecture it must resolve to)`.
const ARTIFACTS: [(&str, &str, Arch); 3] = [
    (
        "RIVOLI_GLM_ARTIFACT",
        "/var/db/rivoli/glm52-vq3-full",
        Arch::GlmMoeDsa,
    ),
    (
        "RIVOLI_V4_ARTIFACT",
        "/var/db/rivoli/v4-f4-full",
        Arch::DeepseekV4,
    ),
    ("RIVOLI_K3_ARTIFACT", "/var/db/rivoli/k3-full", Arch::KimiK3),
];

#[test]
fn every_architecture_still_resolves_and_validates_its_own_artifact() {
    let mut opened = 0usize;
    for (var, default, want) in ARTIFACTS {
        let dir = std::env::var(var).unwrap_or_else(|_| default.to_string());
        if !std::path::Path::new(&dir).is_dir() {
            eprintln!("SKIP {}: no artifact at {dir} (set {var})", want.name());
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
    assert!(
        opened > 0,
        "no architecture's artifact was present — this test asserted nothing. Set one of {:?}",
        ARTIFACTS.map(|(v, _, _)| v)
    );
}
