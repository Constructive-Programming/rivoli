//! The Kimi-K3 engine arm: pin (resident bf16 trunk + streamed MXFP4 latent-space experts),
//! the KDA/MLA interleaved layer loop under multi-residual AttnRes folds, and greedy decode.
//! One arm per architecture — four loop bodies in four files is a measured decision, and
//! `crate::v4`'s header carries the argument for why an ARM is named for its model when the
//! house rule says behaviour.
//!
//! # The shape, and where each fact is enforced
//!
//! | | v4 | k3 |
//! |---|---|---|
//! | attention | shared-K=V MQA, window + pooled blocks | 69 KDA (delta-rule recurrence) + 24 gated MLA, NoPE — the map is `layer_is_mla`, never modulo |
//! | per-token cross-token op | attention only, whole-prompt prefill | the KDA recurrence — every token DEPENDS on the last, so prefill is token-sequential (`decode.rs`) |
//! | KV / state | `[ring ‖ compressed]` per layer | KDA: `[96][128][128]` state + three conv rings, context-free; MLA: expanded `[heads][ctx][192/128]` caches |
//! | residual | `[hc_mult, hidden]` learned mix | a SNAPSHOT STACK + prefix sum, softmax-folded twice per layer plus once model-level (`state.rs`) |
//! | routed format | `.f4`, checkpoint-chosen | `.f4` MXFP4, checkpoint-chosen — and run in the 3584 LATENT, not at hidden 7168 |
//! | experts | top-8 of 256 at hidden width | top-16 of 896, down-projected 7168→3584, aggregate-normed, up-projected |
//! | shared expert | fp8, resident, own dispatch | ONE fused bf16 `[7168,6144]` MLP, trunk-side, added unweighted AFTER the up-projection |
//! | rows per pass | exactly `ROWS` | exactly [`ROWS`] — the situ expert kernel refuses `nrow != 1` (guard 1003) |
//!
//! Nothing in that table is a parameter of another loop. What IS shared lives in
//! `rivoli-core` (the partition, the router scoring), `rivoli-backend` (the launchers) and
//! [`crate::routed`] (the streaming pool) — the seam the sharing was designed for.
//!
//! # The one deviation from the reference, declared rather than discovered
//!
//! **Prefill is token-sequential and the AttnRes arena is one token wide.** The C reference
//! runs T tokens through one function; first-party prefills through fla's CHUNKED KDA kernel,
//! which this tree does not have — the reference's own headers say a chunked port must
//! reinstate the UT-transform inverse and the `A_qk`-diagonal retention together with gating
//! fixtures (`k3:docs/reference/k3-architecture.md` §4), and that is a measured increment,
//! not a default. Sequential prefill also collapses §3's `[T][9][hidden]` arena sizing
//! decision to `[9][hidden]`: no token's snapshots outlive its own pass. The cost is prefill
//! wall time, priced when a benchmark exists; the correctness is identical by construction —
//! decode and prefill are the SAME code path here, which is also what makes `--trace`'s
//! token-major recovery exact on this arm.
//!
//! Twelve order-of-operations traps run cleanly and produce a wrong model
//! (`k3:docs/reference/k3-architecture.md` §10); `engine.rs`'s header maps each to the type,
//! call site or test that owns it.

// Not gated on the backend, and the device half is — `crate::v4`'s convention and its
// argument: these are arithmetic over a config that touches no device, CI compiles only the
// featureless build, and this is the half where a wrong answer is silent.
pub mod geometry;
pub mod state;

#[cfg(feature = "rocm")]
pub mod decode;
/// The device half, module by module, for the reason `crate::v4` gives: a featureless build
/// has no backend to give them.
#[cfg(feature = "rocm")]
pub mod engine;
#[cfg(feature = "rocm")]
pub mod forward;
#[cfg(feature = "rocm")]
pub mod pin;

/// Token rows one routed-expert dispatch carries. **One, and it is the kernel's number**:
/// `moe_f4.hip` instantiates the situ expert range at `R = 1` only and guard 1003 refuses
/// anything else. A named constant for the reason `crate::v4::ROWS` is one — the day a
/// multi-row kernel exists with a measurement behind it, this name is the grep that finds
/// every affected site. `rivoli_core::legality`'s K3 `--mtp` cell names the OTHER blocker
/// too: a verify pass would also need a multi-token KDA recurrence, which is the chunked
/// kernel the module header defers.
pub const ROWS: usize = 1;
