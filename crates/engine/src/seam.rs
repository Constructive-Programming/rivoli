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
use rivoli_artifact::glimmer_config::GlimmerConfig;
use rivoli_artifact::glm_config::ModelConfig;
use rivoli_artifact::k3_config::K3Config;
use rivoli_artifact::v4_config::V4Config;
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
    /// Weight fetches that found what they needed already resident, and those that did not —
    /// **at the ARM's own streaming granularity**, which is the one thing these two fields do
    /// not carry. GLM counts routed experts (~2 MB apiece); Glimmer counts whole layer slots
    /// (967.942 MB apiece). The RATIO is comparable across arms and the counts are not, so no
    /// report may put them in one column.
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

/// The token-emission protocol every decode loop shares — **one author for "is this token
/// output?", "may the run continue?" and "what did the run measure?".**
///
/// The arms' LOOPS are deliberately separate and stay so: GLM's is one async flow wrapped
/// around a streamed expert pool, Glimmer's is synchronous because its fill is a host memcpy
/// with nothing to await. What is not an architectural fact is the protocol around them, and
/// it must not be reimplemented per arm.
///
/// **`old:` has the receipt.** Its Glimmer driver pushed the stop token and then tested for
/// it, so an EOS-terminated run rendered the marker into the printed completion and reported
/// one token more than it produced — two decode drivers in one binary disagreeing about
/// whether the terminator is output, which a golden comparison inherits silently. That is a
/// second-authority bug, and the fix is one authority rather than a second careful copy.
/// `build.rs`'s duplication gate reported the second copy the moment this arm landed, which
/// is the gate arriving at the same conclusion.
///
/// `#[cfg(feature = "rocm")]` because its only consumers are the arms, which are — exactly
/// like [`OS_RESERVE`] below. A backendless build has no loop to run this protocol for.
#[cfg(feature = "rocm")]
pub(crate) struct Emit<'a> {
    eos: &'a [u32],
    ngen: usize,
    ids: Vec<u32>,
}

#[cfg(feature = "rocm")]
impl<'a> Emit<'a> {
    pub(crate) fn new(spec: &GenSpec<'a>) -> Self {
        Self {
            eos: spec.eos,
            ngen: spec.ngen,
            ids: Vec::with_capacity(spec.ngen),
        }
    }

    /// Offer the token decided at the current position. **False means stop.**
    ///
    /// A stop token ends the run WITHOUT being pushed or handed to `sink` — it is not part of
    /// the output. `sink` sees every emitted token before the next forward starts: `serve`
    /// streams from it and returns false when the client hangs up, otherwise a closed
    /// connection would keep the sole-tenant GPU busy for the rest of the budget.
    pub(crate) fn offer(&mut self, tok: u32, sink: &mut dyn FnMut(u32) -> bool) -> bool {
        // The zero-budget check lives HERE, at the protocol's one author, because the
        // push-then-check shape below emits one token at ngen 0 (review 2026-08-16). Both
        // CLI doors already refuse zero; this is for the library caller that computes
        // `budget - prompt.len()` and lands on it.
        if self.ngen == 0 || self.eos.contains(&tok) {
            return false;
        }
        self.ids.push(tok);
        // `sink` is called before the budget is consulted, exactly as the two hand-written
        // loops did: the caller is told about the token it just got even when it is the last.
        sink(tok) && self.ids.len() < self.ngen
    }

    /// Close the run: the ids, and the stats over `decode`.
    ///
    /// The elapsed time is the ARM's, not a clock this owns, and that is deliberate: stats
    /// describe steady-state decode and must EXCLUDE the cold prefill, so the instant they
    /// start from is a point inside the arm's own flow (GLM's is inside its `block_on`). A
    /// clock here would have to be told when to start, which is the same fact spelled twice.
    pub(crate) fn finish(
        self,
        decode: std::time::Duration,
        hits: u64,
        misses: u64,
    ) -> (Vec<u32>, DecodeStats) {
        let decode_s = decode.as_secs_f64();
        let stats = DecodeStats {
            decode_s,
            tok_s: self.ids.len() as f64 / decode_s.max(1e-9),
            hits,
            misses,
        };
        tracing::info!(
            "DECODE: {} tokens in {:.1} s = {:.2} tok/s | {} hits / {} misses",
            self.ids.len(),
            stats.decode_s,
            stats.tok_s,
            stats.hits,
            stats.misses,
        );
        (self.ids, stats)
    }
}

/// What [`Engine::open`] needs REGARDLESS of architecture: the run's device budget and the
/// context it will allocate KV for. Bundled for the same reason as [`GenSpec`].
///
/// **The routed knobs left this struct at M7**, when the second arm arrived. They were four
/// of its six fields and all four are GLM-shaped — a format, a cache policy, a 2Q split and
/// a routed-expert trace path — so a dense arm made to fill them in would be filling in
/// fields nothing spends, which is the exact lie `rivoli_core::legality` exists to stop.
/// They now ride with the config that makes them meaningful, in [`ArchCfg::Glm`].
pub struct OpenSpec {
    /// `--max-mem` in GiB, taken LITERALLY when present. `None` auto-sizes; see
    /// [`OS_RESERVE`].
    pub max_mem_gib: Option<u64>,
    /// KV-slab capacity in tokens — a hard per-run ceiling, allocated once at startup.
    pub max_ctx: usize,
}

/// The routed-expert pool's startup knobs — the three that describe a POOL, which both routed
/// arms have and a dense architecture has none of.
///
/// **The routed FORMAT is deliberately not here**, and its absence is what makes this one type
/// rather than two. GLM chooses between `.vq3` and `.i4` with `--mode`; V4's experts are `.f4`
/// because that is what the checkpoint stores, so it has nothing to choose and a field it
/// always filled the same way would be exactly the "recorded command line carrying a knob
/// nothing spent" `rivoli_core::legality` exists to stop. GLM carries its format as its own
/// arm of [`ArchCfg`] — one extra value on the architecture that has the choice, rather than
/// an `Option` two arms fill differently and a third could forget.
pub struct PoolKnobs<'a> {
    pub cache_policy: &'a str,
    pub two_q: TwoQSplit,
    pub trace_path: Option<&'a str>,
}

/// A caller's token sink: called with each token the moment it lands, BEFORE the next forward.
/// **Returning false stops the run** — `serve` returns false when the client hangs up,
/// otherwise a closed connection would keep the sole-tenant GPU busy for the rest of the
/// budget.
///
/// An alias and not a spelled-out type at each of the four `decode`/`generate` signatures: at
/// full width it pushes every one of them past this workspace's line limit, and rustfmt then
/// breaks them into a five-line block that is byte-identical across two arms — which is what
/// `build.rs`'s duplication gate reported the moment the third arm landed.
pub type TokenSink<'a> = &'a mut dyn FnMut(u32) -> bool;

/// The sniffed architecture together with its config and its own startup knobs — the one
/// value [`Engine::open`] dispatches on.
///
/// **A closed enum for the same reason [`Engine`] is one**, and it is the other half of the
/// same seam: `Engine` is what a decode loop looks like from outside, this is what opening
/// one looks like. Adding an architecture breaks both matches until the new arm is handled.
///
/// **Each config type is already evidence of which architecture this is** — `ModelConfig`
/// and `GlimmerConfig` both refuse every other architecture by name before serde reads a
/// dimension (`rivoli_artifact::schema::parse_config`), so holding one is a proof rather
/// than a claim, and there is deliberately no `--arch` flag that could disagree with it.
/// That is why this carries the config and not an `Arch` beside one.
///
/// `GlimmerConfig`, not its `text_config`: a bare text dict is not evidence that the
/// wrapper around it was Glimmer's, and the wrapper is where `dtype` and
/// `quantization_config` are asserted. The engine reads `.text` on the other side of this.
pub enum ArchCfg<'a> {
    /// The routed format rides HERE and not in [`PoolKnobs`] — see that type for why.
    /// `Mode::Hybrid` has no [`RoutedFmt`] at all, which is why `rivoli_core::legality`
    /// refuses it rather than resolving it to one of the two real formats.
    Glm(&'a ModelConfig, RoutedFmt, PoolKnobs<'a>),
    Glimmer(&'a GlimmerConfig),
    V4(&'a V4Config, PoolKnobs<'a>),
    /// Routed like V4 — a pool, no format choice (the checkpoint ships MXFP4) — so the
    /// same `PoolKnobs` and the same absence of a `RoutedFmt`.
    K3(&'a K3Config, PoolKnobs<'a>),
}

/// The decode engines, one variant per architecture that has a loop.
///
/// GLM-5.2 (MoE, streamed experts) and Muse Glimmer-30B (dense, streamed whole layers);
/// `rivoli_core::legality::decide` is what tells a user of a third artifact that there is
/// no arm for it, in the same breath as it tells them why.
///
/// **`large_enum_variant` is allowed, and boxing would be the wrong fix here.** The arms are
/// 1792 and 632 bytes (clippy, 2026-08-16), so the enum is the larger of the two — and that
/// lint is about values that get copied, collected or passed by value on a hot path. This one
/// is constructed exactly ONCE per process, moved once out of [`Engine::open`] onto `main`'s
/// stack, and reached through `&mut` for the rest of the run; `serve` holds the same single
/// value across every request. A `Box` would buy a 1.8 KB startup memcpy back and pay for it
/// with a heap allocation and an indirection on each of the two methods below.
///
/// What would change the answer: an `Engine` stored per request, per session or in a
/// collection. If one ever is, box the arms rather than widening this allowance.
#[cfg(feature = "rocm")]
#[allow(clippy::large_enum_variant)]
pub enum Engine<'a> {
    Glm(crate::glm::engine::GlmEngine<'a>),
    Glimmer(crate::glimmer::engine::GlimmerEngine<'a>),
    V4(crate::v4::engine::V4Engine<'a>),
    K3(crate::k3::engine::K3Engine<'a>),
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
pub(crate) const OS_RESERVE: u64 = 16 << 30;

impl<'a> Engine<'a> {
    /// Open `dir` as a decode engine, on the arm `cfg` names.
    ///
    /// The config inside `cfg` is borrowed for the engine's whole life, so the caller must
    /// keep it — that borrow is what makes "the config this engine was built for"
    /// un-swappable. See [`ArchCfg`] for why the dispatch is on the config's TYPE rather
    /// than on an `Arch` handed alongside it.
    #[cfg(feature = "rocm")]
    pub fn open(dir: &str, cfg: ArchCfg<'a>, spec: OpenSpec) -> Result<Engine<'a>> {
        // One budget for every arm, computed before the match: a budget that differed by
        // architecture would look like a residency difference in every log line it produced.
        let capacity = device_budget(spec.max_mem_gib)?;
        match cfg {
            ArchCfg::Glm(cfg, fmt, knobs) => {
                let pin = crate::glm::pin::GlmPin::build(dir, cfg, fmt, pin_cfg(capacity, knobs))?;
                let e = crate::glm::engine::GlmEngine::new(pin, cfg, spec.max_ctx)?;
                Ok(Engine::Glm(e))
            }
            // `open` rather than a pin-then-new pair: Glimmer's floor CHARGES for its KV
            // cache and activation scratch, so the two footprints must be computed once and
            // used by both the partition and the allocation. Splitting the call here would
            // hand the pin one number and the constructor another.
            ArchCfg::Glimmer(cfg) => {
                let e = crate::glimmer::engine::GlimmerEngine::open(
                    dir,
                    &cfg.text,
                    capacity,
                    spec.max_ctx,
                )?;
                Ok(Engine::Glimmer(e))
            }
            ArchCfg::V4(cfg, knobs) => {
                // BEFORE the pin, which reads nine gigabytes: this arm's positional block
                // selection has a context ceiling, and a run that exceeds it should learn so
                // at the door rather than after the load. `check_context` carries why the
                // answer is a refusal and not a clamp.
                crate::v4::engine::check_context(cfg, spec.max_ctx)?;
                let pin = crate::v4::pin::V4Pin::build(dir, cfg, pin_cfg(capacity, knobs))?;
                let e = crate::v4::engine::V4Engine::new(pin, cfg, spec.max_ctx)?;
                Ok(Engine::V4(e))
            }
            ArchCfg::K3(cfg, knobs) => {
                // Same door discipline as V4's, different ceiling: K3's is the attend
                // kernel's staging bound, and the pin behind it reads a terabyte-class
                // artifact — the most expensive load in the tree to refuse late.
                crate::k3::geometry::check_context(spec.max_ctx)?;
                let pin = crate::k3::pin::K3Pin::build(dir, &cfg.text, pin_cfg(capacity, knobs))?;
                let e = crate::k3::engine::K3Engine::new(pin, &cfg.text, spec.max_ctx)?;
                Ok(Engine::K3(e))
            }
        }
    }

    /// The refusal a build with no compute backend gives.
    #[cfg(not(feature = "rocm"))]
    pub fn open(dir: &str, cfg: ArchCfg<'a>, spec: OpenSpec) -> Result<Engine<'a>> {
        let _ = (dir, cfg, spec);
        Self::ensure_backend()?;
        unreachable!("ensure_backend errored above in a backendless build")
    }

    /// Whether this build can decode AT ALL — the door check.
    ///
    /// The old tree's backendless binary discovered memory, loaded the manifest, built
    /// the tokenizer and encoded the prompt before admitting it could not decode; then
    /// this tree's first CLI did the tokenizer-and-prompt half of that again (review
    /// 2026-08-16 — the seam doc claimed the fix, the call order had drifted). main calls
    /// THIS before touching the tokenizer, so the claim is now a call site, not prose.
    pub fn ensure_backend() -> Result<()> {
        #[cfg(not(feature = "rocm"))]
        anyhow::bail!(
            "rivoli was built with NO compute backend and cannot decode. Rebuild with \
             `--features rocm` (HIP/ROCm), the only backend."
        );
        #[cfg(feature = "rocm")]
        Ok(())
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
            #[cfg(feature = "rocm")]
            Engine::Glimmer(e) => e.max_ctx(),
            #[cfg(feature = "rocm")]
            Engine::V4(e) => e.max_ctx(),
            #[cfg(feature = "rocm")]
            Engine::K3(e) => e.max_ctx(),
            #[cfg(not(feature = "rocm"))]
            Engine::Never(never, _) => match *never {},
        }
    }

    /// The GLM arm, or the refusal `--divergence-log` gives everywhere else.
    ///
    /// **A refusal, not a silent no-op.** GLM is the arm that does not reproduce itself, so
    /// it is the only arm the folds are placed for; an arm that accepted the flag and wrote
    /// an empty log would be a recorded command line carrying a knob nothing spent, which is
    /// what [`rivoli_core::legality`] exists to stop. Delegated here rather than plumbed
    /// through [`OpenSpec`] for the same reason — a shared startup struct would gain a field
    /// three arms fill and never read.
    ///
    /// One accessor rather than the same `#[cfg]`-split match in both callers below, because
    /// that is exactly the pair `build.rs`'s duplication gate reported when they were two
    /// (2026-08-17). It also removes the fudge the split version needed: the writer used to
    /// return `Ok(())` on a non-GLM arm on the grounds that the door had already refused,
    /// which is a second, weaker copy of the same decision.
    /// Gated on `rocm` as well as its own feature, unlike every other method here: the probe
    /// IS part of the device path, so a deviceless build has no arm to point it at and
    /// [`Engine::open`] has already refused to decode. That also keeps the two callers below
    /// single-bodied — a `#[cfg]`-split body in each is the shape the duplication gate
    /// rejected in the first place.
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    fn glm_for_probe(&mut self) -> Result<&mut crate::glm::engine::GlmEngine<'a>> {
        match self {
            Engine::Glm(e) => Ok(e),
            // A bare `bail!` rather than a `rivoli_core::legality` row, unlike `--trace`'s
            // `DENSE_HAS_NO_EXPERT_TRACE`, and the reason is the DAG: the legality table lives in
            // `rivoli-core`, which cannot name a feature this crate declares — a row for a flag
            // that does not exist in most builds would have to be unconditional, and then the
            // table would advertise a flag `--help` does not show. If `corruption-probe` ever
            // becomes unconditional, this belongs in the table, where the smoke gate would
            // assert its message (nothing asserts this one).
            _ => anyhow::bail!(
                "--divergence-log is implemented for GLM only: it folds the routed-MoE \
                 quantities that split GLM's own run-to-run divergence, and no other arm has \
                 been shown not to reproduce itself"
            ),
        }
    }

    /// Arm `--divergence-log`: fold three per-layer quantities on the device so two runs of
    /// one input can be diffed to a (position, layer, quantity) coordinate.
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    pub fn arm_divergence_log(&mut self, folds: crate::probe::Folds) -> Result<()> {
        self.glm_for_probe()?.arm_divergence_log(folds)
    }

    /// Write the `--divergence-log`. After the run, never during it — the records are held in
    /// memory precisely so the measurement is not perturbed by its own instrument.
    #[cfg(all(feature = "rocm", feature = "corruption-probe"))]
    pub fn write_divergence_log(&mut self, path: &str) -> Result<()> {
        self.glm_for_probe()?.write_divergence_log(path)
    }

    /// Greedy-decode `req`, streaming each token to `sink` the moment it lands; return
    /// false from it to stop early. The delegating match is the whole seam — everything
    /// architecture-shaped is on the other side of it.
    ///
    /// **The match yields the arm's own `(ids, stats)` and the `Decoded` is built ONCE
    /// below it.** Wrapping inside each arm reads more naturally and is exactly the shape
    /// `build.rs`'s jscpd gate reported the moment the second arm landed: two arms whose
    /// bodies were the same twenty tokens. Hoisting the construction is also the honest
    /// factoring — the arms differ in which loop runs, not in what a decode result is.
    ///
    /// **The two builds are separate blocks rather than one `#[cfg]`-attributed match**,
    /// which is the shape every other method here uses. Hoisting forced it: featureless,
    /// the only surviving arm is the uninhabited one, so the match diverges and the `Ok`
    /// after it is `unreachable_code` — an error under this workspace's `-D warnings`. The
    /// split says the same thing without asking the compiler to reason about a match that
    /// has no reachable arms.
    pub fn generate(&mut self, req: GenSpec<'_>, sink: TokenSink<'_>) -> Result<Decoded> {
        #[cfg(not(feature = "rocm"))]
        {
            let Engine::Never(never, _) = self;
            let _ = (req, sink);
            match *never {}
        }
        #[cfg(feature = "rocm")]
        {
            match self {
                // **The two arms return different shapes, and that is history rather than
                // design.** [`Decoded`]'s own doc argues for the named pair over the tuple —
                // `.0`/`.1` read the same whichever way round they are, which is how a
                // swapped destructuring survives review. Glimmer's loop was written after
                // that type existed and returns it; GLM's predates it. GLM should follow
                // when that file is next opened, at which point this match collapses to two
                // identical forwarding arms — which the duplication gate will then have an
                // opinion about, and the answer will be to hoist the whole match.
                Engine::Glm(e) => {
                    let (ids, stats) = e.generate(req, sink)?;
                    Ok(Decoded { ids, stats })
                }
                Engine::Glimmer(e) => e.decode(req, sink),
                Engine::V4(e) => e.decode(req, sink),
                Engine::K3(e) => e.decode(req, sink),
            }
        }
    }
}

/// The run's budget joined to the pool knobs — what a routed pin takes.
///
/// One function because both routed arms build the identical four-field value out of the same
/// two inputs, and the literal is five lines under rustfmt's `struct_lit_width`.
#[cfg(feature = "rocm")]
fn pin_cfg(capacity: usize, knobs: PoolKnobs<'_>) -> crate::resident::PinCfg<'_> {
    crate::resident::PinCfg {
        capacity,
        cache_policy: knobs.cache_policy,
        two_q: knobs.two_q,
        trace_path: knobs.trace_path,
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
