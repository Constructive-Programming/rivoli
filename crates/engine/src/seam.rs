//! The one engine seam: [`Engine`], the closed set of per-architecture decode loops that
//! `main` and (next) `serve` program against.
//!
//! **A closed enum, not a trait object.** Every architecture's loop is a different shape —
//! MLA vs MQA vs KDA, streamed experts vs none — so a `dyn Engine` would either grow a
//! method per architecture or hide the differences behind a lowest common denominator.
//! An enum keeps the dispatch exhaustive (a new arm breaks every match until it is
//! handled), costs no vtable on a path taken once per token, and lets each arm keep its
//! own state type verbatim.
//!
//! **The lifetime is deliberate.** `Engine<'a>` borrows the `ModelConfig` its caller owns
//! — exactly as `GlmPin` and `GlmEngine` already do. `main` keeps the config on its stack
//! and the engine borrows it for the run; that is why there is no self-referential
//! construction here and no crate to make one.
//!
//! **[`Engine::open`] is the only place that knows the featureless build cannot decode.**
//! A backendless `rivoli` still has to compile — `cargo test --workspace` and the
//! featureless clippy run build every target, and that is what keeps the
//! backend-independent half honest — so the refusal is a runtime one, taken at the door,
//! before a tokenizer is loaded or a prompt is encoded. Consumers therefore carry no
//! `#[cfg]` of their own: `main.rs` is the same source in both builds.

use anyhow::Result;
use rivoli_artifact::format::RoutedFmt;
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_core::cache::TwoQSplit;

/// One generation request. Bundled (like the pool's `PoolCfg`) so `generate`'s signature
/// stays readable and a new knob cannot push it past the argument budget.
pub struct GenSpec<'a> {
    pub prompt: &'a [u32],
    /// Token budget for the GENERATED tail (the prompt is not counted).
    pub ngen: usize,
    /// Any of these ends the run (and is not emitted).
    pub eos: &'a [u32],
}

/// What a decode measured about itself. Minimal on purpose: the old engine's per-phase
/// `Profile` is deferred to the first benchmark — but a run still reports the two numbers
/// that make its cost citeable.
pub struct DecodeStats {
    pub decode_s: f64,
    pub tok_s: f64,
    pub hits: u64,
    pub misses: u64,
}

/// What a decode produced: the ids, and what the run measured about itself.
///
/// A named pair rather than the arm's own `(Vec<u32>, DecodeStats)` tuple, because this is
/// the seam's public result and it is about to have two consumers in two crates. `.0` and
/// `.1` read the same whichever way round they are, which is how a swapped destructuring
/// survives review; `ids` and `stats` do not.
pub struct Decoded {
    pub ids: Vec<u32>,
    pub stats: DecodeStats,
}

/// What [`Engine::open`] needs beyond the artifact and its config: the run's device
/// budget, the routed format it decodes with, the cache knobs, and the context it will
/// allocate KV for. Bundled for the same reason as [`GenSpec`].
pub struct OpenSpec<'a> {
    /// `--max-mem` in GiB, taken LITERALLY when present. `None` auto-sizes; see
    /// [`OS_RESERVE`].
    pub max_mem_gib: Option<u64>,
    /// The run's ONE routed format. `--mode hybrid` has no value here at all, which is
    /// why `rivoli_core::legality` refuses it rather than resolving it to one of these.
    pub fmt: RoutedFmt,
    pub cache_policy: &'a str,
    pub two_q: TwoQSplit,
    pub trace_path: Option<&'a str>,
    /// KV-slab capacity in tokens — a hard per-run ceiling, allocated once at startup.
    pub max_ctx: usize,
}

/// The decode engines, one variant per architecture that has a loop.
///
/// GLM-5.2 is the only arm today; `rivoli_core::legality::decide` is what tells a user of
/// another artifact that, in the same breath as it tells them why.
#[cfg(feature = "rocm")]
pub enum Engine<'a> {
    Glm(crate::glm::engine::GlmEngine<'a>),
}

/// A backendless build has no arm at all, and says so in the type: `Infallible` makes the
/// variant unconstructible, so [`Engine::open`]'s refusal is the ONLY way this type can be
/// reached — `generate` is unreachable by construction rather than by a runtime check.
/// The `PhantomData` keeps `'a` used, which a variantless enum cannot do.
#[cfg(not(feature = "rocm"))]
pub enum Engine<'a> {
    Never(std::convert::Infallible, std::marker::PhantomData<&'a ()>),
}

/// What the auto budget leaves for everyone else when `--max-mem` is absent.
///
/// Memory here is unified LPDDR5 through GTT, so device bytes are host bytes: the
/// compositor, the page cache the expert stream reads through, and the process itself all
/// come out of the same pool. `--max-mem` is honoured literally — the user asked for that
/// size and is allowed to OOM at pin build — and the auto path just leaves this much free.
#[cfg(feature = "rocm")]
pub const OS_RESERVE: u64 = 16 << 30;

impl<'a> Engine<'a> {
    /// Open `dir` as a decode engine.
    ///
    /// `cfg` is borrowed for the engine's whole life, so the caller must keep it — that
    /// borrow is what makes "the config this engine was built for" un-swappable.
    ///
    /// It is the GLM config type because GLM is the only arm, and `ModelConfig::load`
    /// refuses every other architecture before serde reads a dimension — so holding one is
    /// already evidence of which arm this is. A second architecture brings its own config
    /// type and turns this into a dispatch on the sniffed [`rivoli_core::legality::Arch`];
    /// nothing here blocks that, because the caller sniffs before it opens.
    #[cfg(feature = "rocm")]
    pub fn open(dir: &str, cfg: &'a ModelConfig, spec: OpenSpec<'_>) -> Result<Engine<'a>> {
        let pin = crate::glm::pin::GlmPin::build(
            dir,
            cfg,
            crate::glm::pin::GlmPinCfg {
                capacity: device_budget(spec.max_mem_gib)?,
                fmt: spec.fmt,
                cache_policy: spec.cache_policy,
                two_q: spec.two_q,
                trace_path: spec.trace_path,
            },
        )?;
        Ok(Engine::Glm(crate::glm::engine::GlmEngine::new(
            pin,
            cfg,
            spec.max_ctx,
        )?))
    }

    /// The refusal a build with no compute backend gives, at the door.
    ///
    /// FIRST, before anything expensive: the old tree's version discovered memory, loaded
    /// the manifest, built the tokenizer and encoded the prompt before admitting it could
    /// not decode.
    #[cfg(not(feature = "rocm"))]
    pub fn open(dir: &str, cfg: &'a ModelConfig, spec: OpenSpec<'_>) -> Result<Engine<'a>> {
        let _ = (dir, cfg, spec);
        anyhow::bail!(
            "rivoli was built with NO compute backend and cannot decode. Rebuild with \
             `--features rocm` (HIP/ROCm), the only backend."
        )
    }

    /// The KV ceiling this engine was BUILT with, in tokens.
    ///
    /// Exists so a long-lived caller can refuse a request before decoding it. `serve`
    /// answers "does this conversation fit?" on every request, and the honest source of
    /// that number is the engine that allocated the slabs: [`OpenSpec::max_ctx`] passed a
    /// SECOND time alongside the engine would be a copy free to drift the moment `open`
    /// ever clamps or rounds it, and the drift would surface as a `forward` refusal minutes
    /// into a decode instead of as a 400 at the door.
    pub fn max_ctx(&self) -> usize {
        match self {
            #[cfg(feature = "rocm")]
            Engine::Glm(e) => e.max_ctx(),
            #[cfg(not(feature = "rocm"))]
            Engine::Never(never, _) => match *never {},
        }
    }

    /// Greedy-decode `req`, streaming each token to `sink` the moment it lands; return
    /// false from it to stop early. The delegating match is the whole seam — everything
    /// architecture-shaped is on the other side of it.
    pub fn generate(
        &mut self,
        req: GenSpec<'_>,
        sink: &mut dyn FnMut(u32) -> bool,
    ) -> Result<Decoded> {
        match self {
            #[cfg(feature = "rocm")]
            Engine::Glm(e) => {
                let (ids, stats) = e.generate(req, sink)?;
                Ok(Decoded { ids, stats })
            }
            #[cfg(not(feature = "rocm"))]
            Engine::Never(never, _) => {
                let _ = (req, sink);
                match *never {}
            }
        }
    }
}

/// The device bytes a pin may use.
///
/// Lives here rather than in the CLI so that every consumer of the seam — `main` today,
/// `serve` next — spends the same budget: a budget that differed between two entry points
/// would look like a residency difference in every log line it produced.
#[cfg(feature = "rocm")]
fn device_budget(max_mem_gib: Option<u64>) -> Result<usize> {
    const GIB: f64 = (1u64 << 30) as f64;
    let (free, _total) = crate::device::mem_info()?;
    let Some(gib) = max_mem_gib else {
        let cap = free.saturating_sub(OS_RESERVE as usize);
        tracing::info!(
            "device pool budget {:.1} GiB (auto: free {:.1} GiB − {:.0} GiB OS reserve)",
            cap as f64 / GIB,
            free as f64 / GIB,
            OS_RESERVE as f64 / GIB,
        );
        return Ok(cap);
    };
    let bytes = (gib as usize) << 30;
    tracing::info!(
        "device pool budget {:.1} GiB (--max-mem, literal — no reserve; may OOM)",
        bytes as f64 / GIB
    );
    Ok(bytes)
}
