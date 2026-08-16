//! The DeepSeek-V4-Flash-0731 engine arm: pin (resident attention core + streamed `.f4`
//! experts), the hyper-connected layer loop, and greedy decode. One arm per architecture —
//! four loop bodies in four files is a measured decision, and a pin parameterised by an arch
//! flag is a GLM-shaped placement path one `if` away from running on the wrong artifact.
//!
//! # Why the module is named for the model, when the house rule says otherwise
//!
//! "Name code for behaviour, not the model" is this repo's rule and it holds for kernels,
//! traits and formats — `kvcompress.hip`, `.f4`, `swiglu_clamped_bf16`. It does not hold for
//! an ARM, because an arm's whole content is "which model this is": [`crate::glm`] and
//! [`crate::glimmer`] are already named that way, `lib.rs` names this one before it exists,
//! and a behaviour name here would have to describe an architecture no other checkpoint
//! shares. The reference engine tried the other spelling — it renamed its loop to the routed
//! FORMAT (`f4gpu`) — and its own header then had to warn that the name over-promises,
//! because a different `.f4` model would not drop in. That warning is the cost of the rename;
//! this tree pays the rule's cost instead, once, here.
//!
//! # The shape, and where each fact is enforced
//!
//! | | glm | v4 |
//! |---|---|---|
//! | attention | MLA + q-LoRA, fp8 latent KV | shared-K=V MQA, one `head_dim` entry for every head |
//! | rows attended | the whole causal prefix | a `sliding_window` ring **plus** pooled blocks |
//! | KV per layer | latent + roped key, appended | `[ring ‖ compressed]`, one contiguous buffer |
//! | residual | one `[hidden]` stream | `[hc_mult, hidden]` — four streams and a learned mix |
//! | routed format | `.vq3` or `.i4`, chosen by `--mode` | `.f4`, chosen by the checkpoint |
//! | shared expert | routed-format, folded into the batch | **fp8 e4m3, resident**, its own dispatch |
//! | rows per pass | up to [`crate::glm::MAXROW`] | exactly [`ROWS`] — the expert kernel has no other |
//!
//! Nothing in that table is a parameter of the other loop. What IS shared lives in
//! `rivoli-core` (the partition, the router), `rivoli-backend` (the launchers) and
//! [`crate::routed`] (the streaming pool), which is the seam the sharing was designed for.
//!
//! # The three deviations from the reference, declared rather than discovered
//!
//! Each is inherited from the reference engine, which measured it; none is a defect this port
//! introduced, and each names the oracle defect it corresponds to so a future scoring run
//! recognises it instead of chasing it.
//!
//! 1. **The shared expert's SwiGLU is clamped**, where the reference engine's was not
//!    (`v4oracle::Defect::SwigluUnclamped`). This arm calls `launch_swiglu_clamped_bf16` with
//!    the config's `swiglu_limit`, which is the kernel written for exactly this and the reason
//!    it exists. That is a divergence FROM the reference ENGINE and a convergence WITH the
//!    reference MODEL, which is the direction that matters.
//! 2. **The compressed-block selection is positional, not scored.** The trained-in indexer
//!    ranks blocks; this arm takes every causally-legal one. The two agree on the SET below
//!    [`select::positional_context_limit`] and [`engine::check_context`] refuses a context at
//!    or above it — at the door, before the pin reads nine gigabytes — so the shortcut is
//!    bounded by a startup check rather than by a comment. They never agree on the score
//!    ORDER, which the online softmax folds in: a deliberate difference and a floor under any
//!    tolerance a scoring run may claim.
//!
//!    **This is the one deviation a USER meets.** At the shipped `index_topk` the ceiling is
//!    2052, so the CLI's 4096 `--ctx` default is refused and a V4 run must name a smaller one.
//!    Clamping instead would contradict `rivoli_core::legality`'s `Support` on that flag and
//!    silently truncate a conversation the user sized deliberately; `check_context` carries
//!    the argument, and the ceiling goes away when the indexer does.
//! 3. **The MoE output is not bf16-rounded** between the accumulator drain and the residual.
//!    The fixed-point accumulator is the arithmetic here; rounding it would be a second
//!    quantization the kernel does not perform.

// **Not gated on the backend, and the other seven are.** These three are arithmetic over a
// config that touches no device: layer classes, rotary tables, and which rows a query may
// read. CI has no `rocm` job, so anything behind that gate is compiled as often as someone
// runs it here — and this is the half where a wrong answer is silent. `crate::glimmer`'s
// `geometry` carries the same argument and the `old:` finding that produced it.
pub mod geometry;
pub mod rope;
pub mod select;

#[cfg(feature = "rocm")]
pub mod attn;
#[cfg(feature = "rocm")]
pub mod decode;
/// The device half. Gated, unlike the three above, for the reason this module's own comment
/// gives: these need a backend, and a featureless build has none to give them.
#[cfg(feature = "rocm")]
pub mod engine;
#[cfg(feature = "rocm")]
pub mod forward;
#[cfg(feature = "rocm")]
pub mod kvcompress;
#[cfg(feature = "rocm")]
pub mod moe;
#[cfg(feature = "rocm")]
pub mod pin;

// **The deviceless half landed FIRST, alone, and deliberately** — it is where a wrong answer
// is silent (a rotary table with the wrong base, a block selected one step early, a pooled
// row placed by a counter instead of by its position), and the device half is a caller of all
// three.
//
// > **UPDATED: the device half landed and this arm is REACHABLE.** [`pin`] places the
// > resident set and opens the `.f4` pool; [`engine`], [`kvcompress`], [`attn`], [`moe`],
// > [`forward`] and [`decode`] are the loop; `Engine::V4` and
// > `rivoli_core::legality::deepseek_v4` landed in the SAME commit, which they had to —
// > `main`'s `the_arms_and_the_legality_table_agree_about_who_can_start` asserts
// > `!refused == has_arm` over every architecture, so a row without a loop and a loop without
// > a row are each a red tree.
//
// **What is still NOT here, and each is a decision rather than an omission:** the trained-in
// indexer (deviation 2 — the pin does not place its weights, and the positional selection is
// bounded by [`select::positional_context_limit`] at startup); the reference's per-phase
// `Profile` and its four split instruments; and the probe API its per-layer oracle comparison
// drove. The last two are the same narrowing [`crate::glm`] took, for the reason [`engine`]'s
// header gives: a bucket that reads zero because nothing filled it is this repo's named
// telemetry trap.

/// Token rows one routed-expert dispatch carries. **One, and it is the kernel's number
/// rather than this loop's preference.**
///
/// `moe.hip` instantiates `rivoli_moe_expert_range_f4` at `R = 1` only and its guard 1003
/// refuses anything else, so a V4 decode is structurally single-row: there is no verify pass
/// to batch and no speculative arm to build, whatever a draft head might later offer.
/// `rivoli_core::legality`'s `--mtp` cell says exactly that to the user, and says it as a
/// missing KERNEL rather than a missing head, because that is which of the two would have to
/// arrive first.
///
/// It is a named constant and not a bare `1` for the reason [`crate::glm::MAXROW`] is one:
/// every buffer that is `ROWS`-shaped should say so, and the day an `R = 2` kernel exists
/// with a measurement behind it, the grep that finds every affected site is this name.
///
/// **What this does NOT bound is the PREFILL.** Attention is the only operation here with a
/// cross-token dependency, so the prompt goes through `attention` in one call of `m` rows
/// while the MoE runs `m` times — see [`forward`]. So `ROWS` is the MoE's batch, not the
/// pass's.
pub const ROWS: usize = 1;
