//! The request and the measurements EVERY Inkling composition gate makes, in one place.
//!
//! `inkling_small_mxfp8.rs` (the non-looped member) and `inkling_mini_mxfp8.rs` (the looped one)
//! ask the same question of two artifacts: the same greedy prompt, decoded at a budget that pins
//! every expert and at the smallest capacity that opens, with ids and logits compared
//! **bit-for-bit** (P4: the memory knob trades speed, never text). The protocol is one author
//! here rather than two copies in the gates — the two copies had already reached 106 identical
//! tokens before `build.rs`'s duplication gate reported them.
//!
//! **The fixture itself is `inkling_small`'s** (the writer for both members' artifacts); this
//! module only opens what that module wrote, so the two cannot drift on a tensor name.
//!
//! **Run every gate as the workspace runs a device suite:** `-- --test-threads=1`.

// Two gate binaries compile this module and each uses a subset (the looped gate never asks for
// `p4`'s non-looped arm, the non-looped one never calls `seam_ids`), so an unused item is dead in
// that binary by construction — `tests/common/mod.rs`'s argument, one module over.
#![allow(dead_code)]
// Tests: panic-on-failure is the idiom, and every fixture value is generated rather than threaded
// out as a `Result`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::inkling_small::expert_total;
use rivoli_artifact::inkling_config::InklingTextConfig;
use rivoli_engine::inkling::engine::InklingEngine;
use rivoli_engine::resident::PinCfg;
use rivoli_engine::seam::GenSpec;

pub const N_CTX: usize = 32;
pub const NGEN: usize = 4;
pub static PROMPT: [u32; 4] = [1, 7, 13, 29];

/// The one greedy request both gates make: the fixture prompt, `NGEN` new tokens, and no stop
/// token — the ids are compared bit-for-bit across budgets, so an EOS the fixture happened to
/// emit would make the comparison's width depend on which arm ran.
pub const SPEC: GenSpec<'static> = GenSpec {
    prompt: &PROMPT,
    ngen: NGEN,
    eos: &[],
};

/// One greedy decode plus the two things every gate reads off it: the ids, the sampled logits
/// row, and the fills this run spent (the A-vs-A arm asserts the last is zero once warm).
pub fn decode(engine: &mut InklingEngine<'_>) -> (Vec<u32>, Vec<f32>, u64) {
    let decoded = engine
        .decode(SPEC, &mut |_| true)
        .expect("the synthetic Inkling fixture must decode");
    let logits = engine.logits().expect("a sampled row is available");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "a decode row must be finite — an infinite or NaN logit is a broken composition, not a \
         tolerance question"
    );
    (decoded.ids, logits, decoded.stats.misses)
}

/// The row's bit patterns. `==` on f32 hides a `-0.0`/`0.0` move and says nothing about NaN, and
/// every claim this protocol makes is BIT-identity, so the comparison is made where the claim
/// lives.
pub fn bits(row: &[f32]) -> Vec<u32> {
    row.iter().map(|v| v.to_bits()).collect()
}

/// The smallest capacity that opens at all: every routed expert must stream from it. Found by
/// BISECTION on observed behaviour rather than by re-deriving the pin's floor in a gate — the
/// pin's slack, the tier's 256-byte rounding and the pool's staging arena are the pin's
/// arithmetic, and a copy here would be a second author to drift.
pub fn open_tight<'a>(dir: &str, text: &'a InklingTextConfig, n_ctx: usize) -> InklingEngine<'a> {
    let (mut lo, mut hi) = (0usize, 64 << 20);
    let opens =
        |cap: usize| InklingEngine::open_mxfp8(dir, text, PinCfg::stock(cap), n_ctx).is_ok();
    assert!(opens(hi), "a 64 MiB budget must open the fixture");
    // `PinCfg::stock(0)` is a LITERAL zero-byte budget, not "auto-size": the pin's floor must
    // refuse it. That is the invariant the bisection rests on, and it is asserted rather than
    // assumed because an auto-sizing capacity would silently make every later arm "pinned".
    assert!(
        !opens(0),
        "a zero-byte capacity must be refused by the pin's floor"
    );
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if opens(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    InklingEngine::open_mxfp8(dir, text, PinCfg::stock(hi), n_ctx)
        .expect("the minimal successful capacity opens by construction")
}

/// **P4, at the seam, for either member.** One prompt, two budgets: everything pinned, and the
/// smallest capacity that opens at all (so the routed set streams through `RoutedPool`'s O_DIRECT
/// fill). Ids AND logits must be bit-identical — a token that moves when only residency changed
/// is a pool defect.
///
/// **The pinned arm's A-vs-A rerun is the floor**, and it is checked here rather than in each
/// gate: it separates "the pool is not state-free" from "this arm does not repeat itself", which
/// is the question any later mismatch raises.
///
/// The messages name the member from the CONFIG (`loop_config`'s presence), not from a parameter:
/// the two arms' failures must not be indistinguishable in a log, and a label a caller passes is
/// a label a caller can pass wrongly. **The passing summary is printed here for the same reason**
/// — a gate that re-printed the same three numbers would be a five-line copy `build.rs`'s
/// duplication gate reports, and a number printed by the arm that measured it is the one that
/// cannot drift from the arm that measured it.
pub fn p4(dir: &str, text: &InklingTextConfig) {
    let label = match text.loop_config.is_some() {
        true => "looped",
        false => "non-looped",
    };
    let total = expert_total(text);
    let mut pinned = InklingEngine::open_mxfp8(dir, text, PinCfg::stock(64 << 20), N_CTX)
        .expect("the fixture opens with room for every expert");
    assert_eq!(
        pinned.residency(),
        (total, 0),
        "64 MiB must pin every one of the {total} {label} experts and stream none"
    );
    let (ids, logits, first_misses) = decode(&mut pinned);
    let (ids_again, logits_again, warm_misses) = decode(&mut pinned);
    assert_eq!(
        warm_misses, 0,
        "the pinned {label} arm must not miss once warmed (first decode had {first_misses} \
         cold-start fills, repeat had {warm_misses} — a budget that can hold every expert must \
         never evict)"
    );
    assert_eq!(ids, ids_again, "the pinned {label} arm must repeat its ids");
    assert_eq!(
        bits(&logits),
        bits(&logits_again),
        "the pinned {label} arm must repeat its own logits bit-for-bit"
    );

    let mut streamed = open_tight(dir, text, N_CTX);
    let (tight_pinned, tight_streamed) = streamed.residency();
    assert!(
        tight_streamed > 0,
        "the tight {label} arm pinned all {tight_pinned} experts — a budget at the open boundary \
         must force the routed set through the pool"
    );
    let (ids_streamed, logits_streamed, streamed_misses) = decode(&mut streamed);
    assert!(
        streamed_misses > 0,
        "the tight {label} arm recorded no miss — it never left the resident tier, so this run is \
         not the streaming arm it claims to be"
    );
    assert_eq!(
        ids, ids_streamed,
        "ids moved when only residency changed: pinned {ids:?} vs streamed {ids_streamed:?} — P4 \
         lets bytes/token move and forbids text from moving"
    );
    assert_eq!(
        bits(&logits),
        bits(&logits_streamed),
        "logits are not bit-identical across budgets on the {label} schedule — the pool is not \
         state-free (a slot reused under a live launch, or decoded from a stale block)"
    );
    eprintln!(
        "P4 ({label}): {total} experts, tight arm pinned {tight_pinned} and streamed \
         {tight_streamed}, ids and logits bit-identical"
    );
}
