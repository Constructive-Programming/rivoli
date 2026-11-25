//! **Inkling 2 Mini's LOOPED schedule over the `.mxfp8` pin — the container-independence gate.**
//!
//! `inkling_engine_anchor.rs` scores the loop against the first-party anchor on a bf16 artifact,
//! and `inkling_small_mxfp8.rs` scores the pool on the NON-looped member. Neither can see the
//! composition this file exists for: Mini's own 24-layer schedule (a 6-layer prefix, the core
//! `6..18` run twice, a 6-layer suffix) running over `RoutedPool`, where the second pass looks up
//! the SAME stored `(layer, expert)` keys the first pass did.
//!
//! Three facts only this composition shows:
//!
//! 1. **The looped config opens on the `.mxfp8` pin at all.** `InklingPin::build_mxfp8` used to
//!    refuse a looped config; the pool's dispatch reads the STORED layer, so the refusal was the
//!    only thing standing between Mini and the streaming backend.
//! 2. **Residency invariance (P4) on the looped schedule** — `inkling_gate::p4`, the same protocol
//!    the non-looped gate runs: one prompt, two budgets, bit-identical ids and logits. A second
//!    pass that read a slot the pool had already evicted would move a token, and no non-looped
//!    arm can see that.
//! 3. **The seam's sniff.** `ArchCfg::InklingLoop` carries no container flag, so `rivoli` has to
//!    find `mxfp8_source` in the manifest and open the pool pin, while the bf16 artifact of the
//!    same member keeps the resident one.
//!
//! **Run it as the workspace runs every device suite:** `-- --test-threads=1`.
//!
//! **RED-PROOF RECIPE (P7 — the device half is OWED until it runs on the box).**
//! * loop: restore the `text.loop_config.is_none()` refusal in `build_mxfp8` and
//!   [`looped_mxfp8_decodes_and_repeats_itself`] panics on `.expect(...)`.
//! * P4: make the tight arm's pool hand back a stale slot (or skip the `wait` before reuse) and
//!   [`residency_moves_no_bit_on_the_looped_schedule`] reddens inside `inkling_gate::p4` on the
//!   token.
//! * sniff: make `has_mxfp8_source` return `false` unconditionally and
//!   [`the_seam_sniffs_the_container`] panics in `Engine::open` — the bf16 pin has no expert slabs
//!   to place, so it cannot open the `.mxfp8` directory.

#![cfg(feature = "rocm")]
// Tests: panic-on-failure is the idiom, and every fixture value is generated rather than threaded
// out as a `Result`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod inkling_gate;
mod inkling_small;

use inkling_gate::{N_CTX, NGEN, SPEC, bits, decode, p4};
use inkling_small::{Scratch, Storage, build_mini, expert_total};
use rivoli_engine::inkling::engine::InklingEngine;
use rivoli_engine::resident::PinCfg;
use rivoli_engine::seam::{ArchCfg, Engine, OpenSpec, PoolKnobs};

/// The budget every SEAM arm opens with. 1 GiB pins the looped fixture's routed set outright,
/// which is what the sniff test needs: its subject is which pin the manifest chose, not how the
/// pool behaves at a boundary (that is [`residency_moves_no_bit_on_the_looped_schedule`]'s).
const ONE_GIB: OpenSpec = OpenSpec {
    max_mem_gib: Some(1),
    max_ctx: N_CTX,
};

/// One greedy run through the SEAM — the same request the direct arms make, so the only variable
/// left between two seam opens is which pin the manifest's sniff chose.
fn seam_ids(engine: &mut Engine<'_>) -> Vec<u32> {
    engine
        .generate(SPEC, &mut |_| true)
        .expect("the seam must decode the looped fixture")
        .ids
}

/// **The loop over the pool.** The config loops, the experts stream from `.mxfp8` files, and the
/// second pass over `6..18` reads the same stored `(layer, expert)` keys the first pass did. A
/// repeat of the SAME request is the A-vs-A floor: it separates "the pool is not state-free" from
/// "this arm does not repeat itself", which is the question any later mismatch raises.
#[test]
fn looped_mxfp8_decodes_and_repeats_itself() {
    let scratch = Scratch::new("mini-mx");
    let (dir, cfg) = build_mini(&scratch, Storage::MxFp8);
    let total = expert_total(&cfg.text);
    let mut engine = InklingEngine::open_mxfp8(&dir, &cfg.text, PinCfg::stock(64 << 20), N_CTX)
        .expect("a LOOPED config must open on the .mxfp8 pin");
    assert_eq!(
        engine.residency(),
        (total, 0),
        "64 MiB must pin every one of the {total} experts and stream none"
    );
    let (ids, row, _) = decode(&mut engine);
    assert_eq!(ids.len(), NGEN, "one id per requested token");
    let (ids_again, row_again, _) = decode(&mut engine);
    assert_eq!(ids, ids_again, "the looped arm must repeat its own ids");
    assert_eq!(
        bits(&row),
        bits(&row_again),
        "the looped arm must repeat its own logits bit-for-bit — a message gate reading the \
         wrong pass's bank is not a rounding difference"
    );
    eprintln!("looped .mxfp8: {total} experts pinned, ids {ids:?}");
}

/// **P4 on the looped schedule**, through the protocol both members share
/// ([`inkling_gate::p4`]): everything pinned vs the smallest capacity that opens, ids AND logits
/// bit-identical. The A-vs-A rerun inside it is the floor — a loop is where "the pool is not
/// state-free" would first show, because the same keys are looked up a second time.
#[test]
fn residency_moves_no_bit_on_the_looped_schedule() {
    let scratch = Scratch::new("mini-p4");
    let (dir, cfg) = build_mini(&scratch, Storage::MxFp8);
    p4(&dir, &cfg.text);
}

/// **The seam's sniff.** `ArchCfg::InklingLoop` carries no container flag, so `Engine::open` has
/// to read `mxfp8_source` out of the manifest: the `.mxfp8` artifact must open through the pool
/// pin, and the bf16 artifact of the same member through the resident one.
#[test]
fn the_seam_sniffs_the_container() {
    let mx = Scratch::new("mini-seam-mx");
    let (mx_dir, mx_cfg) = build_mini(&mx, Storage::MxFp8);
    let bf = Scratch::new("mini-seam-bf16");
    let (bf_dir, bf_cfg) = build_mini(&bf, Storage::Bf16);

    // The RESIDENT pin cannot open an `.mxfp8` artifact — the expert slabs it places are not in
    // one — so a successful seam open of that directory is itself proof the sniff fired. Asserted
    // rather than assumed: a fixture that grew slabs into both containers would make the next
    // assertion pass for the wrong reason.
    assert!(
        InklingEngine::open(&mx_dir, &mx_cfg.text, 64 << 20, N_CTX).is_err(),
        "the bf16 pin opened an .mxfp8 artifact — the containers are no longer distinguishable"
    );

    let mut pooled = Engine::open(
        &mx_dir,
        ArchCfg::InklingLoop(&mx_cfg, PoolKnobs::stock()),
        ONE_GIB,
    )
    .expect("the seam must open a looped .mxfp8 artifact through the pool");
    assert_eq!(pooled.max_ctx(), N_CTX);
    let ids_pooled = seam_ids(&mut pooled);

    let mut resident = Engine::open(
        &bf_dir,
        ArchCfg::InklingLoop(&bf_cfg, PoolKnobs::stock()),
        ONE_GIB,
    )
    .expect("the seam must keep opening a looped bf16 artifact");
    let ids_resident = seam_ids(&mut resident);
    assert_eq!(ids_pooled.len(), NGEN, "one id per requested token");
    assert_eq!(ids_resident.len(), NGEN, "one id per requested token");
    eprintln!("seam: .mxfp8 {ids_pooled:?}, bf16 {ids_resident:?}");
}
