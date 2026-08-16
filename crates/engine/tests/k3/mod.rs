//! **The Kimi-K3 kernel-oracle harness** — the vendored S2 anchor pair, the config readers,
//! and the scoring spine (`rel`, `tripwire`, `score_all`, `priced`) shared by the seven
//! `kernel_k3_*.rs` suites. Ported from `k3:tests/k3_kernels.rs` (M9), whose 39 tests those
//! suites carry; the port splits one 3,718-line file along its `// ====` item banners because
//! this tree's 800-line soft cap says to, and this module is the share the banners had in
//! common.
//!
//! # Why a SECOND vendored anchor pair exists, one directory over from the first
//!
//! `crates/oracles/tests/k3-anchor-decode-k3-anchor-{1,2}.bin` is the **S1b capture set** —
//! 226 tensors, and none of `.fold`, `self_attn.attend.*`, `o_proj.in_gated`,
//! `routed_expert_norm.{in,weight}`, `*_conv1d.weight` or `o_norm.{in,weight}` are in it.
//! Five of the seven M9 kernel suites are unwriteable against those bytes: the fold has an
//! output and no scoring vector, the convolution has a window and no taps. The k3 tree
//! regenerated its anchors during S2 for exactly this reason (its item 1 header records
//! "found by walking up to write this file and finding nothing to launch with",
//! `k3:tests/k3_kernels.rs:40`), and the pair vendored HERE is that S2 recapture — 290
//! tensors, same `RIVK3GLD` container, read by the same [`GoldenSet::read_k3`]. The S1b pair
//! next door stays untouched because `crates/oracles/tests/k3_anchor.rs` pins its bytes;
//! reconciling the two vendorings is that gate's owner's call, not this port's — flagged in
//! the port report rather than done quietly.
//!
//! Device tests: run with `-- --test-threads=1` under `flock /var/run/sys-gpu.lock`.

// Compiled into seven binaries; none names every item, and deadness per binary is an
// accident of which item that binary ports (`common/mod.rs` makes the same argument).
#![allow(dead_code)]

use super::common::{Got, Want, worst_rel};

/// `Policy`/`Tol` — the tolerance-table SHAPE, shared with the anchor gates by `#[path]`
/// exactly as `glimmer_anchor/mod.rs` includes `golden_read.rs`: one file on disk, so the
/// kernel table below and the anchor tables next door cannot drift on what a row IS.
#[path = "../../../oracles/tests/common/tolerance.rs"]
pub mod shape;

#[path = "../../../oracles/tests/common/golden_read.rs"]
pub mod golden_read;

pub mod tolerance;

// `#[allow(unused_imports)]` for the reason the module header gives `dead_code`: this
// compiles into seven binaries and the fp4-expert one, whose fixtures are synthetic, never
// reads a golden tensor by name. The allow is on this ONE re-export, so a genuinely dead
// `use` elsewhere still reports.
#[allow(unused_imports)]
pub use golden_read::{GoldenSet, Vendored, float};

// **The suites import this harness with `use k3::*;`, and the glob is a jscpd decision, not
// laziness.** An import LIST is the one duplication Rust gives you no way to factor
// (`golden_read.rs`'s own header makes the argument) — and seven suites each spelling
// `use common::{Lcg, back, dev, f32b, f32v, ok, ...}; use k3::{GOLDENS, GoldenSet, ...}`
// produced two of the first build's nine clones, between the pairs whose lists happened to
// coincide. One glob per suite has no list to coincide; the common items each binary needs
// ride through this re-export, and the per-file `use rivoli_backend::hip::...` line stays
// explicit because the launcher names are what a reader greps a suite for.
#[allow(unused_imports)]
pub use super::common::{
    GemmBf16, Lcg, assert_guard, assert_guards, back, dev, f32b, f32v, gemm_bf16_launch, ok,
    stream, u16b, zeros,
};

/// Both vendored draws. A kernel bug degenerate at one draw's values hides completely, and
/// the softmax is exactly the sort of arithmetic that has a degenerate case — a stack whose
/// scores happen to be far apart collapses the mixture onto one source and stops testing the
/// mixing at all. Running both is the anchor's own argument for vendoring two, applied to
/// the kernel (`k3:tests/k3_kernels.rs:73`).
pub const GOLDENS: [(&str, &[u8]); 2] = [
    (
        "k3-anchor-1",
        include_bytes!("k3-anchor-decode-k3-anchor-1.bin"),
    ),
    (
        "k3-anchor-2",
        include_bytes!("k3-anchor-decode-k3-anchor-2.bin"),
    ),
];

/// The byte pins for [`GOLDENS`], in the `Vendored` form the anchor gates use. A pin checked
/// only against a frozen copy of itself is decoration, so the FNV is recomputed from the
/// live bytes by `kernel_k3_attn_res.rs::the_vendored_anchors_match_their_pins` — the one
/// suite that owns the pin test, since seven copies of it would be seven identical loops.
pub fn vendored() -> [Vendored; 2] {
    [
        Vendored {
            name: "k3-anchor-1 (S2 recapture)",
            bytes: GOLDENS[0].1,
            len: 353_638,
            fnv: 11_418_977_334_977_154_342,
        },
        Vendored {
            name: "k3-anchor-2 (S2 recapture)",
            bytes: GOLDENS[1].1,
            len: 353_638,
            fnv: 927_488_350_216_143_535,
        },
    ]
}

pub fn load(bytes: &[u8]) -> GoldenSet {
    GoldenSet::read_k3(&mut &bytes[..]).expect("the vendored golden must load")
}

/// The config the reference was built from, as the golden itself carries it.
///
/// One parse with several readers ([`eps`], [`betas`], [`lower_bound`]), because every one
/// of them is making the same argument: a constant a fixture hardcoded would agree with
/// itself if the reference's value ever moved. `crates/oracles/tests/k3_anchor.rs` pins the
/// tiny config's structural fields against the real checkpoint's, so reading them here says
/// "the model's value", not "the file's" (`k3:tests/k3_kernels.rs:124`).
pub fn tiny(g: &GoldenSet) -> serde_json::Value {
    serde_json::from_str(g.meta_get("tiny_config").expect("tiny_config")).expect("valid json")
}

/// The eps the reference's RMSNorm used, read off the golden's own `tiny_config` — not a
/// literal `1e-5`. The reference reads `norm.variance_epsilon`, which is
/// `config.rms_norm_eps` (`k3:tests/k3_kernels.rs:114`).
pub fn eps(g: &GoldenSet) -> f32 {
    tiny(g)["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32
}

/// The two SiTU betas, read off the golden's own `tiny_config` — they are STRUCTURAL, so
/// the tiny config cannot have shrunk them. The second key is `activation_situ_linear_beta`;
/// abbreviating it to `activation_linear_beta` is a mistake the k3 port made once, in S1a,
/// where it would have refused every real checkpoint (`k3:tests/k3_kernels.rs:1520`).
pub fn betas(g: &GoldenSet) -> (f32, f32) {
    let c = tiny(g);
    let f = |k: &str| c[k].as_f64().unwrap_or_else(|| panic!("{k} missing")) as f32;
    (f("activation_situ_beta"), f("activation_situ_linear_beta"))
}

/// The pair the config ships, for the SYNTHETIC cases only.
///
/// The distinction matters: [`betas`] reads the golden's own config because a golden-scored
/// fixture that hardcoded the pair would agree with itself if the reference's value moved.
/// The synthetic cases — the SiTU width/magnitude sweep and the whole fp4-expert oracle —
/// take the betas as ARGUMENTS to a function under test rather than as a property of a
/// captured output, so pinning the shipped pair is the right thing there. One symbol so the
/// two conventions are visible instead of a literal at three sites (`k3:tests/k3_kernels.rs:1527`).
pub const SHIPPED_BETAS: (f32, f32) = (4.0, 25.0);

/// `-5.0`, read off the golden's own config rather than written down — through the reader's
/// own accessor, where the bound's inclusive-end argument now lives (it was spelled here AND
/// in the anchor gate's harness, which was the 2026-08-16 integration's one jscpd finding).
pub fn lower_bound(g: &GoldenSet) -> f32 {
    GoldenSet::k3_gate_lower_bound(&tiny(g))
}

/// Relative difference the way the anchor's `--by-operator` measures it, so the number
/// compared here is the number the tolerance was measured in.
///
/// The house metric, not a fourth spelling of it: `common::worst_rel` already returns
/// INFINITY on a non-finite `got` — the `f32::max`-skips-NaN trap that let an all-NaN kernel
/// output score 0.0 in both trees' histories — and PANICS on a length mismatch or a
/// non-finite `want`, which is strictly louder than the reference `rel`'s INFINITY for the
/// same two conditions (`k3:tests/k3_kernels.rs:370`). Every scoring site in the seven K3
/// suites goes through here, so the device result, the host oracles and every defect variant
/// are scored identically; two call sites computing their own denominator is how a defect
/// "fails" against a slightly different number.
pub fn rel(got: &[f32], want: &[f32]) -> f32 {
    worst_rel(Got(got), Want(want))
}

/// **The regression tripwire every K3 suite carries, in one place.**
///
/// `tolerance`'s operator rows are WHOLE-MODEL floors: they were measured on fp32-vs-fp64
/// runs carrying upstream drift, while these fixtures hand each kernel the reference's OWN
/// inputs. Every kernel lands one to three orders under its tolerance — and against the
/// tolerance alone, a change that degraded a kernel by two orders would pass in silence. So
/// each site also pins the worst it actually measures and gets 10x of room: close enough to
/// catch a regression, far enough not to fire on a reassociated sum.
///
/// **The tolerance is still the contract.** This is not a second one; it is a smoke alarm on
/// a number that has no business moving. Moving a constant is allowed and re-measuring is
/// how — what is not allowed is loosening one to make a red run green without knowing why it
/// moved. (`k3:tests/k3_kernels.rs:281`; one function rather than a copy per site because
/// jscpd rejected the fourth copy there at 135 tokens.)
/// The two bars every K3 scoring site holds a result to: the operator tolerance (the outer
/// envelope, from the table) and the fixture's own measured worst (what the tripwire binds
/// at). One type because they only ever travel together — [`tripwire`], [`score_all`] and
/// [`priced`] each took them as two bare `f32`s whose swap compiles and inverts the gate,
/// which is `common/scoring.rs`'s `Want`/`Got` argument made about bounds instead of data.
#[derive(Clone, Copy)]
pub struct Bars {
    pub tol: f32,
    pub observed: f32,
}

pub fn tripwire(r: f32, b: Bars, at: &str) {
    let Bars { tol, observed } = b;
    let observed_worst = observed;
    // **The tripwire must be TIGHTER than the operator tolerance, or it is decoration.**
    // Nothing related the two before 2026-08-12, and the message below invites moving the
    // constant — so a hand-widened `observed_worst` could sail past `tol` and leave the site
    // with no bar at all. Measured payoff, from the review that found it: a 10x-wrong KDA
    // L2-norm eps moves the anchor comparison by 1.775e-4 to 4.184e-4, which this tripwire
    // catches with 100x to spare and the 6.3e-4 `kda_op` tolerance does not see at any of
    // the six sites (`k3:tests/k3_kernels.rs:299`).
    //
    // Only the LOWER direction can be checked uniformly: the gap between the two bars runs
    // from 6.5x (`dense_mlp`) to 4,800x (`moe_latent`), because an operator tolerance is a
    // whole-model floor and a tripwire is this fixture's own measurement — an upper backstop
    // in the table's 30x style would reject the tightest row.
    assert!(
        observed_worst * 10.0 <= tol,
        "{at}: the {observed_worst:e} tripwire admits {:e}, which is ABOVE the {tol:e} operator \
         tolerance — so this site has no effective bar. Re-measure the kernel rather than the \
         constant.",
        observed_worst * 10.0
    );
    assert!(
        r <= observed_worst * 10.0,
        "{at}: {r:e} is far above the {observed_worst:e} this kernel achieves. Still inside the \
         {tol:e} operator tolerance, so this is a REGRESSION tripwire — re-measure and move the \
         constant only if the new value is defensible."
    );
}

/// Score every output of one operator against its golden: the operator tolerance, then the
/// tripwire.
///
/// Factoring it is what stops the sites disagreeing about the ORDER of the two bars, which
/// matters: the tolerance is the outer envelope and the tripwire is what binds, and a site
/// that checked only one would be checking the wrong one (`k3:tests/k3_kernels.rs:338`).
pub fn score_all(at: &str, b: Bars, pairs: &[(&str, &[f32], &[f32])]) {
    for (what, got, want) in pairs {
        let r = rel(got, want);
        let at = format!("{at} {what}");
        assert!(r <= b.tol, "{at}: {r:e} exceeds {:e}", b.tol);
        tripwire(r, b, &at);
    }
}

/// One defect variant has to move the result clear of the tripwire by the table's own margin.
///
/// The table's 30x `DEFECT_MARGIN`, applied to the bar this fixture ENFORCES rather than to
/// the operator tolerance — which is the distinction the k3 tree's
/// `the_situ_sigmoid_takes_the_uncapped_gate` found the hard way, where the bucket tolerance
/// turned out unable to catch a defect the tripwire catches by 2,800x
/// (`k3:tests/k3_kernels.rs:353`).
pub fn priced(at: &str, what: &str, moved: f32, b: Bars) {
    let bar = b.observed * 10.0 * 30.0;
    assert!(
        moved > bar,
        "{at} {what}: moved the operator by only {moved:e}, under the {bar:e} this fixture's \
         tripwire needs cleared by the table's 30x — so it does not price the difference it \
         names, and the agreement elsewhere says nothing about which form the kernel \
         implements. (The operator tolerance is {:e}.)",
        b.tol
    );
}

/// Max-subtracted softmax in `f64`. Both f64 host oracles that need one (the AttnRes fold's
/// and the MLA attend's) end with these four lines; the k3 tree's jscpd caught the copy
/// (`k3:tests/k3_kernels.rs:193`).
pub fn softmax64(score: &[f64]) -> Vec<f64> {
    let m = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = score.iter().map(|s| (s - m).exp()).collect();
    let z: f64 = ex.iter().sum();
    ex.into_iter().map(|e| e / z).collect()
}

/// `n` independent f64 sums, narrowed to f32 on the way out.
///
/// The narrowing is the part worth having in one place: it is the only rounding the f64
/// oracles perform, so a stray `as f32` on an intermediate would be a floor that is not one
/// (`k3:tests/k3_kernels.rs:938`).
pub fn sums64(n: usize, mut term: impl FnMut(usize) -> f64) -> Vec<f32> {
    (0..n).map(|j| term(j) as f32).collect()
}

/// `SituAndMul.forward` in f64 — `(b1·tanh(g/b1)·sigmoid(g)) · (b2·tanh(u/b2))`.
///
/// The sigmoid takes `g`, NOT `b1·tanh(g/b1)`: the two factors saturate at different rates
/// and feeding the capped value to the sigmoid is the smooth, plausible, wrong version. The
/// ONE flag is the defect — a defect run is the correct oracle with one thing changed, and a
/// second function is how the two drift into differing by something nobody intended
/// (`k3:tests/k3_kernels.rs:1580`). Here rather than in the SiTU suite because the fp4
/// expert oracle composes the same arithmetic between its two passes.
pub fn situ1(g: f32, u: f32, (b1, b2): (f32, f32), capped_sigmoid: bool) -> f32 {
    let (b1, b2) = (f64::from(b1), f64::from(b2));
    let (g, u) = (f64::from(g), f64::from(u));
    let t = b1 * (g / b1).tanh();
    let sig = if capped_sigmoid { t } else { g };
    (t * (1.0 / (1.0 + (-sig).exp())) * (b2 * (u / b2).tanh())) as f32
}

/// The same arithmetic over two slices — the SiTU scoring oracle. Elementwise primitive,
/// map over it: the arithmetic stays in exactly one place (`k3:tests/k3_kernels.rs:1604`).
pub fn host_situ(gate: &[f32], up: &[f32], betas: (f32, f32), capped_sigmoid: bool) -> Vec<f32> {
    gate.iter()
        .zip(up)
        .map(|(&g, &u)| situ1(g, u, betas, capped_sigmoid))
        .collect()
}

/// The beta pairs every SiTU launcher must refuse, and each is quiet in its own way — the
/// argument is at `rivoli_situ_glu_f32`. `NaN` makes `tanh(x/b)` NaN for every element; `0`
/// saturates to ±1 except exactly at `x == 0`, where it is NaN; `+inf` is the silent
/// spelling of "no saturation", since `b·tanh(x/b) -> x`; negative flips the saturating
/// branch (`k3:tests/k3_kernels.rs:1756`).
pub const BAD_BETAS: [(f32, f32, &str); 7] = [
    (0.0, 25.0, "b1 = 0"),
    (4.0, 0.0, "b2 = 0"),
    (-4.0, 25.0, "b1 negative"),
    (f32::NAN, 25.0, "b1 NaN"),
    (4.0, f32::NAN, "b2 NaN"),
    (f32::INFINITY, 25.0, "b1 +inf"),
    (4.0, f32::INFINITY, "b2 +inf"),
];

/// Hold one launcher to [`BAD_BETAS`], and to accepting the shipped pair.
///
/// **One function for both beta-guarded launchers** (`situ_glu_f32` and
/// `moe_expert_range_f4_situ`), which is stronger than two copies as well as shorter: their
/// kernels each claim to use "the same code AND the same expression" as the other, and this
/// is what makes that claim checkable rather than aspirational (`k3:tests/k3_kernels.rs:1770`).
pub fn assert_betas_guarded(launcher: &str, mut refused: impl FnMut(f32, f32) -> bool) {
    for (b1, b2, case) in BAD_BETAS {
        assert!(refused(b1, b2), "{launcher}: {case} was accepted");
    }
    // Not refusing everything, which is how a refusal test passes vacuously.
    assert!(
        !refused(SHIPPED_BETAS.0, SHIPPED_BETAS.1),
        "{launcher}: the shipped betas were refused, so the guard rejects everything and the \
         seven assertions above carry no information"
    );
}
