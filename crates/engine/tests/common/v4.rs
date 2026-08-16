//! The DeepSeek-V4-Flash **checkpoint** scaffolding: where the checkpoint lives, how a suite
//! skips when it is absent, the compressor weights three suites load out of it, and the two
//! `V4Config` types a checkpoint-backed comparison holds against each other.
//!
//! **Ported from `old:tests/common/mod.rs`'s "V4-Flash checkpoint scaffolding" region**, which
//! `mod.rs`'s header already promised would "land in its own submodule beside the six above,
//! not in this file" when its first consumer arrived. This is that submodule and M8 is that
//! consumer. Bodies and their arguments travelled; what changed is named in place.
//!
//! Nothing here touches a device type, so it is featureless like `reference.rs` and
//! `geometry.rs` beside it — a checkpoint reader is host work, and the suites that need a GPU
//! carry their own gate.
//!
//! # What did NOT come, and why
//!
//! `probe(name, n, dim)` and `residual_probe(cfg, tag, s)`. Both were two-line wrappers over a
//! bf16-rounded `NamedRng` draw, and the oracle crate now exports exactly that as
//! `v4oracle::weights::fixed_bf16(name, n, scale)` — the same FNV-1a name seed, the same
//! `unit()` sequence, the same round-trip. Re-spelling either here would be a THIRD copy of a
//! draw whose whole value is that one seed means one byte stream; the suites call `fixed_bf16`
//! directly and multiply out the row width at the call, where the shape is visible.

use rivoli_artifact::v4_config::V4Config as EngineV4Config;
use rivoli_engine::v4::geometry::LayerKind;
use rivoli_oracles::v4oracle::forward::{Capture, CompressorW, Defect, LayerCtx, Oracle, Sinks};
use rivoli_oracles::v4oracle::toy::{self, ToyModel};
use rivoli_oracles::v4oracle::weights::{
    Checkpoint, NamedRng, V4Config as OracleV4Config, fixed_bf16,
};
use std::path::Path;
use std::sync::OnceLock;

/// The unconverted DeepSeek-V4-Flash-0731 checkpoint — safetensors shards, `config.json`, and
/// the 5.3 MB index the suites read.
pub const CKPT: &str = "/var/db/rivoli/deepseek-v4-flash-0731";

/// [`CKPT`] if the checkpoint is there, or `None` after printing WHY.
///
/// A skip and not a failure: this reads 5.3 MB of index metadata off a 167 GB checkpoint that no
/// CI job has, and there is no `rocm` CI arm to have it. **The print is the whole contract** —
/// libtest captures stdout on a PASSING test, so a suite that skips silently is indistinguishable
/// from one that ran, which is why every caller returns immediately after this and asserts nothing
/// on the empty path.
///
/// One body for both entry points below, because two copies of a presence check is two places for
/// one of them to start testing a different file — and `build.rs`'s duplication gate said so on
/// the first compile.
fn present() -> Option<&'static str> {
    if Path::new(CKPT)
        .join("model.safetensors.index.json")
        .exists()
    {
        return Some(CKPT);
    }
    eprintln!("SKIP: no checkpoint at {CKPT}");
    None
}

/// Open the checkpoint, or skip loudly. See [`present`].
pub fn checkpoint() -> Option<Checkpoint> {
    Some(Checkpoint::open(Path::new(present()?)).expect("opening checkpoint"))
}

/// The two V4 configurations a checkpoint-backed comparison needs, and the assertion that
/// they agree.
///
/// **They are deliberately different types and the difference is the instrument.**
/// [`OracleV4Config::v4_flash`] HARD-CODES the shipped numbers so the oracle depends on no
/// loader of the engine's — that independence is its whole value, and reading a dimension out
/// of it to size a kernel launch would spend it. So every dimension a KERNEL is handed below
/// comes from [`EngineV4Config::load`], which parses the checkpoint's own `config.json`
/// through the engine's schema, and the oracle's copy is used for nothing but constructing
/// the `Oracle`.
///
/// [`Configs::new`] then asserts the pair agrees on the six dimensions both declare. That is
/// not redundant with either: it is the ONLY executable check that the transliteration's
/// hard-coded constants still describe the checkpoint the kernels are being scored on. The
/// oracle's own `assert_matches_reference_json` covers the same ground and runs only inside
/// `bin/v4-oracle`, which is not on any `cargo test` path in this tree.
pub struct Configs {
    /// Parsed from the checkpoint. **The source of every dimension a launcher is handed.**
    pub engine: EngineV4Config,
    /// The transliteration's own constants. Used to build an `Oracle` and for nothing else.
    pub oracle: OracleV4Config,
}

impl Configs {
    /// Load and cross-check, or `None` when the checkpoint is absent (see [`present`]).
    pub fn new() -> Option<Self> {
        let engine =
            EngineV4Config::load(present()?).expect("parsing the checkpoint's config.json");
        let oracle = OracleV4Config::v4_flash();
        // Pairs rather than one conjunction: the message must name WHICH dimension drifted,
        // because the repair differs — a changed checkpoint is a re-measurement and a changed
        // hard-code is a bug in the instrument.
        //
        // `qk_rope_head_dim` against `rope_head_dim` is the one rename in the list, and it is
        // why this cannot be a derive: the two types spell the same field differently because
        // one is named for the JSON key and the other for the reference's `ModelArgs`.
        for (what, a, b) in [
            ("head_dim", engine.head_dim, oracle.head_dim),
            (
                "rope_head_dim",
                engine.qk_rope_head_dim,
                oracle.rope_head_dim,
            ),
            (
                "index_head_dim",
                engine.index_head_dim,
                oracle.index_head_dim,
            ),
            ("index_n_heads", engine.index_n_heads, oracle.index_n_heads),
            ("hidden/dim", engine.hidden, oracle.dim),
            ("n_heads", engine.n_heads, oracle.n_heads),
            ("hc_mult", engine.hc_mult, oracle.hc_mult),
            ("sliding_window", engine.sliding_window, oracle.window_size),
        ] {
            assert_eq!(
                a, b,
                "{what}: the checkpoint's config.json says {a} and the oracle's hard-coded \
                 V4Config says {b}. The transliteration no longer describes the checkpoint the \
                 kernels below are scored against, so every comparison in this suite is between \
                 two different models."
            );
        }
        // The two epsilons are f64 in the JSON and f32 in the oracle, so they are compared in
        // the narrower domain — which is also the domain both reach a kernel in.
        assert_eq!(
            engine.rms_norm_eps as f32, oracle.norm_eps,
            "norm_eps disagreement: the RMSNorm inside every pooled block would be the \
             oracle's and not the checkpoint's"
        );
        Some(Self { engine, oracle })
    }

    /// This layer's compression ratio, **from the checkpoint** — the classifier every layer
    /// class in these suites is derived through.
    ///
    /// `expect` and not `unwrap_or(0)`: `EngineV4Config::compress_ratio` bounds-checks against
    /// `n_layers`, and a layer index past the end is a fixture bug rather than a plain layer.
    pub fn ratio(&self, layer: usize) -> usize {
        self.engine
            .compress_ratio(layer)
            .expect("layer index inside n_layers")
    }

    /// This layer's class, through [`Self::ratio`], so no suite spells `from_ratio` on a
    /// number it read somewhere else.
    pub fn kind(&self, layer: usize) -> LayerKind {
        LayerKind::from_ratio(self.ratio(layer))
    }
}

// ---------------------------------------------------------------------------------------
// the TOY model — the fixture every checkpoint-free V4 oracle runs on
// ---------------------------------------------------------------------------------------

/// A toy configuration, the model built from it, and a defect-free oracle over it.
///
/// A tuple rather than a struct because every consumer destructures it in one line
/// (`let (cfg, m, o) = toy_fixture();`) and none of the three is ever passed without the others.
pub type Toy = (OracleV4Config, ToyModel, Oracle);

/// Build a toy model and a CLEAN oracle over `cfg`.
///
/// Separate from [`toy_fixture`] because one suite needs the same construction at OTHER dims —
/// the fp4 dword-trip test resizes `dim` and `moe_inter_dim` — and two copies of these two lines
/// is what `build.rs`'s duplication gate reported the first time they existed in four files.
///
/// `Defect::None` and nothing else: every deliberate break in the V4 kernel suites is a change to
/// a KERNEL argument or a second oracle constructed per defect, so the cached one only ever wants
/// the clean arithmetic.
pub fn build_toy(cfg: OracleV4Config) -> Toy {
    let m = toy::build(&cfg);
    let o = Oracle::new(cfg.clone(), Defect::None);
    (cfg, m, o)
}

/// The standard toy fixture, built once per test binary.
///
/// `V4Config::toy` keeps `n_hash_layers = 3` over 4 layers, so layers 0-2 route by `tid2eid` and
/// layer 3 by score — both router modes reachable from one fixture, which is the only reason a
/// 4-layer toy has 4 layers. Layer 0 is ratio-0 (`Plain`), layer 2 is ratio-4 with an indexer and
/// layer 3 is ratio-8: three layer classes, every discriminant preserved and every extent shrunk.
///
/// Cached, because four suites build it and `toy::build` quantizes every weight.
pub fn toy_fixture() -> &'static Toy {
    static M: OnceLock<Toy> = OnceLock::new();
    M.get_or_init(|| build_toy(OracleV4Config::toy()))
}

/// One prefill of one layer: which oracle drives it, which layer, and the seed and shape of the
/// residual it is driven from.
///
/// A struct because `layer` and `s` are both bare `usize` and swapping them type-checks — a
/// 3-token prompt through layer 12 and a 12-token prompt through layer 3 are both legal and only
/// one is the fixture. It carries the ORACLE too, because the interesting runs are the defect
/// ones: a caller that reached for the cached clean oracle while asking for a defect capture would
/// get a green comparison against nothing.
#[derive(Clone, Copy)]
pub struct Prefill<'a> {
    /// The oracle to drive. `toy_fixture().2` for a clean run; a freshly constructed one under a
    /// `Defect` for a break.
    pub o: &'a Oracle,
    pub layer: usize,
    /// Seeds BOTH the residual draw and the prompt ids, so one name means one whole fixture.
    pub tag: &'a str,
    /// Prompt length. `LayerCtx::s` and `input_ids.len()` are both derived from it here and never
    /// passed separately — a `LayerCtx` whose `s` disagreed with its id slice is a fixture no
    /// comparison could have caught, because `run_layer` walks `s` positions through whatever
    /// slice it was handed and every golden downstream becomes a capture of a prompt nobody wrote.
    pub s: usize,
    /// The residual draw's amplitude.
    pub scale: f32,
}

/// Drive one PREFILL `run_layer` and return `(what it captured, the residual it started from)`.
///
/// The residual is handed back because both consumers compare against it: it is `L{layer}.pre.in`,
/// and a driver whose `h` is not what the oracle recorded has re-fixtured the whole run.
///
/// The state is dropped on the way out because both callers drop it. A caller that needs to drive
/// a second step against the same state wants `run_layer` directly — this is the one-shot prefill,
/// and pretending otherwise would hand back a state whose `start_pos` the caller would have to
/// reconstruct.
pub fn prefill(fx: &Toy, p: Prefill<'_>) -> (Capture, Vec<f32>) {
    let (cfg, m, _) = fx;
    let mut h = fixed_bf16(p.tag, p.s * cfg.hc_mult * cfg.dim, p.scale);
    let mut ids_rng = NamedRng::new(&format!("{}-ids", p.tag));
    let ids: Vec<u32> = (0..p.s)
        .map(|_| ids_rng.below(cfg.vocab_size) as u32)
        .collect();
    let started_from = h.clone();
    let step = LayerCtx {
        lw: &m.layers[p.layer],
        layer: p.layer,
        // `ids.len()` and never `p.s`, even though they are equal by construction two lines up:
        // one expression, so the pair the `Prefill::s` doc warns about cannot come apart here.
        s: ids.len(),
        start_pos: 0,
        input_ids: &ids,
        step_tag: "pre",
    };
    let mut st = p.o.fresh_state(p.layer);
    let mut cap = Capture::default();
    // The sinks bound before the call rather than built in the argument list — the inline form is
    // a token run `build.rs`'s duplication gate matched against `rivoli-oracles`' own test driver,
    // which spells the identical `run_layer` invocation across a crate boundary that no factoring
    // can cross. Naming the pair is the honest fix: it is one value in the reference's own API
    // (`Sinks`), and hoisting it says so.
    let sinks = Sinks {
        st: &mut st,
        cap: &mut cap,
    };
    p.o.run_layer(&step, &mut h, sinks);
    (cap, started_from)
}

/// WHICH compressor a layer's `attn.compressor.*` load is for: its ratio, its head width, and the
/// finish it owes.
///
/// A struct because there are TWO compressors on an indexed layer and only `rotate` separates
/// them — the attention one at `head_dim` with the partial fp8 finish, the indexer's nested one at
/// `index_head_dim` with the Hadamard-and-fp4 one. `ratio` and `d` are both bare `usize`, so a
/// transposed pair loads real tensors at a real width and the shape assertions below would pass on
/// exactly one of the two layer classes. Naming the three moves that mistake to the call site.
#[derive(Clone, Copy)]
pub struct CompSpec {
    pub ratio: usize,
    /// The COMPRESSOR's head width, which is `head_dim` for the attention one and the much
    /// narrower `index_head_dim` for the indexer's.
    pub d: usize,
    /// `true` only for the indexer's nested compressor.
    pub rotate: bool,
}

/// One layer's `attn.compressor.*` at `spec.d` wide, with `rotate` set by WHICH compressor it is.
///
/// Loading these directly rather than through a whole-layer loader is the point: a layer's
/// routed experts are 3.4 GB and none of it is read by `Oracle::compressor`.
///
/// The four shape assertions are the load-bearing part and each pins a trap the S2c brief
/// names:
///
/// * `ape` is `[ratio, coff*d]`, so `[4, 1024]` at ratio 4 (coff 2) and `[128, 512]` at ratio
///   128 (coff 1). A loader that inferred the width from `d` alone gets 512, which is WRONG on
///   the ratio-4 attention compressor and right on ratio 128. The error is a silent misindex
///   and not a length mismatch, because both widths are 512-multiples.
/// * `wkv`/`wgate` are `[out, in]`, the torch `Linear` convention `Oracle::linear` reads: rows
///   are the projection width, cols the model dim. Asserting `cols` here instead passed on
///   layer 2 by coincidence of both being 4096-adjacent, and is the axis mix-up worth pinning.
/// * `norm` is over `head_dim`, not `coff * head_dim`.
pub fn compressor_w(ck: &Checkpoint, prefix: &str, spec: CompSpec) -> CompressorW {
    let CompSpec { ratio, d, rotate } = spec;
    let kind = LayerKind::from_ratio(ratio);
    let get = |suffix: &str| {
        ck.get(&format!("{prefix}.{suffix}"))
            .expect("compressor tensor")
            .to_f32()
            .expect("compressor tensor as f32")
    };
    let dense = |suffix: &str| {
        ck.dense(&format!("{prefix}.{suffix}"))
            .expect("compressor projection")
    };
    let cw = CompressorW {
        ratio,
        overlap: kind.overlap(),
        d,
        rotate,
        ape: get("ape"),
        wkv: dense("wkv.weight"),
        wgate: dense("wgate.weight"),
        norm: get("norm.weight"),
    };
    assert_eq!(
        cw.ape.len(),
        ratio * kind.coff() * d,
        "{prefix}: ape is [ratio, coff*d] = [{ratio}, {}]",
        kind.coff() * d
    );
    assert_eq!(
        cw.wkv.rows(),
        kind.coff() * d,
        "{prefix}: wkv projects TO coff*d"
    );
    assert_eq!(
        cw.wgate.rows(),
        kind.coff() * d,
        "{prefix}: wgate matches wkv"
    );
    assert_eq!(
        cw.wkv.cols(),
        cw.wgate.cols(),
        "{prefix}: both read the same model dim"
    );
    assert_eq!(
        cw.norm.len(),
        d,
        "{prefix}: norm is over head_dim, not coff*head_dim"
    );
    cw
}
