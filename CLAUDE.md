# rivoli (rewrite) — orientation for agents

Decode engine for LLMs bigger than memory: AMD Strix Halo gfx1151, unified LPDDR5 via GTT,
weights streamed from NVMe **overlapped with compute** — the overlap is the whole design.
Rust workspace + HIP/ROCm, one backend. This is the ground-up rewrite; the old tree stays
live as the parity reference at **`wt/glimmer-s2` @ `6b7f496e`** — port from there, cite
it as `old:<path>`, and treat its `docs/` as the archive of closed investigations.

Start with `docs/00-orientation/TOUR.md` (two pages), then `INDEX.md` — decide what NOT to
open from the verdict column.

## Quality is paramount

**Every claim is a gate that can go red (P7), and a gate is proven able to fail before its
green is believed.** TDD is the ordering rule: the anchor, oracle, or measured tolerance
exists *before* the code it scores. A check that has never been red is not evidence; a
check whose examined-count can silently reach zero is not a check.

The seven principles are `docs/reference/principles.md` (owner-confirmed 2026-08-12) — a
change that violates one is wrong even if every test passes. Index: **P1** bigger-than-
memory on this one box · **P2** caching trades space/bandwidth/compute · **P3** hardware
over portability · **P4** the memory knob trades speed, never text · **P5** bytes/token is
the currency · **P6** the pin is a function of free memory, never architecture · **P7**
every claim is a gate that can go red.

## TDD workflow

- Anchor/golden first: goldens come from the **first-party reference stack** (never a
  transliteration), vendored with self-describing pins, gated by a defect matrix in which
  every defect reddens where it should and holds where it should.
- Every new gate ships with a **recorded red proof** (run, shown red, reverted or kept as
  a standing fixture). A proof that refuses to go red is itself evidence — debug the tree,
  not the proof.
- **Tolerances carry provenance**: measured from operator fp32 floors on **≥ 2 weight
  draws** *before* the kernel exists, citing the measurement. A tolerance picked to make a
  kernel pass is not a tolerance; a round number is a confession.

## FP conventions

Pure core / imperative shell: `rivoli-core` is total functions over plain data emitting
values (`Directive`s, spans, verdicts); `rivoli-engine` is the interpreter that spends
them on the device. Nothing in core can name a stream, a pointer, or a weight format —
the workspace DAG enforces it. `unwrap`/`expect` are deny-level workspace-wide (tests and
build scripts opt back in per-file, with the argument). Newtypes for units: bytes, tokens,
layers, expert ids are never bare `usize`. Typestate for lifecycles (a ticket is consumed
by `wait_on`, a writer is sealed, an unloaded artifact is a different type). FP crates are
admitted where they earn their dependency weight, argued at the `use` site; the standing
list is `proptest` and `thiserror`, and an HKT crate joins only when a concrete hole
appears that enum dispatch cannot fill.

## Gates

- **jscpd** — duplication is a build error, zero budget (`crates/cli/build.rs`, every
  build, both feature arms). Precondition: rustfmt-clean, or the result is a lower bound.
  **6** regions are exempt via ignore markers (three in the ported frozen V4 oracle,
  where verbatim transcription of the reference is the point; two in `backend/hip.rs`,
  the HIP ABI wall's extern declarations and its one hand-written launcher; one in
  `artifact/quant.rs`, the `matvec_*` oracle parameter lists — each argues in place); the
  count is derived by `crates/cli/tests/docs.rs`, and marker text may appear only on a
  bare marker line.
- **CodeScene 10/10** — `crates/cli/tests/codescene.rs`, whole tree, hard threshold;
  standing red-proof fixture must score < 10 every run; exemptions argued in place and
  checked at both ends. Warn-and-skip without a license locally; CI hard-fails via
  `RIVOLI_CS_REQUIRED=1`. Needs `CS_ACCESS_TOKEN` in the environment.
- **clippy** `-D warnings`, `--all-targets`, plus the deviceless feature-union run once
  instrument features exist.
- **docs registry** — status/scope/verdict front matter + INDEX agreement, test-enforced.
- Landing later, each with the thing that makes it non-vacuous: INV-n registry (first
  invariant, M1), feature matrix (first feature, M1), kernel census (first launcher, M3),
  refactor/parity gates (first capture, M5).

## Build and test

```bash
cargo test --workspace            # dev profile: debug_assert! is LIVE. Default to this.
cargo clippy --workspace --all-targets
cargo build --release             # benchmarks and performance evaluation ONLY
```

**A run that is not timing something is a dev-profile run.** `[profile.release]` compiles
out every `debug_assert!`; that is what benchmarks are measured under, and it is also why
a check that must hold in a shipped binary is an `assert!` and pays its cost.

## Measurement discipline (all of these drew blood in the old tree)

- **The GPU is sole-tenant.** Wrap every GPU command in `flock /var/run/sys-gpu.lock -c`,
  build OUTSIDE the lock. The flock is advisory and other agents skip it: sample a
  **contention witness** per arm (`/dev/kfd` holders + `mem_info_gtt_used` — KFD is blind
  to Vulkan tenants) and **discard any arm with a non-empty witness**.
- **Always `-- --test-threads=1` on any suite that touches the device.** Parallel device
  tests build parallel io_uring rings and wedge; `cargo test --lib` counts as a GPU arm.
- **Never `cargo build` between the two arms of a benchmark** (page-cache eviction moved
  ms/miss 1.36 → 5.14 once).
- **Rank quality on paired dNLL from `bin/ppl`, never the PPL column.** An interval
  straddling zero is *inconclusive*, not a pass. `distinct`/longest-repeated-block measure
  nothing — read the text.
- **Instruments go behind a feature AND a flag, never an env var** (invisible to `--help`,
  absent from recorded command lines, silently active in a stock-looking build). Scripts
  that are not cargo runs may use env vars with the argument written down in place.
- An unwitnessed number is unciteable. Prefer measured spans to derived percentages; a
  bracket timer that contains what it is used to rule out is a defect class.

## Conventions

- Comments explain *why* and carry the measurement that justified the choice; a bare
  restatement of the code is noise here.
- Name code for **behaviour + ABI**, never for the model that introduced it; model
  membership is data in a checked census table.
- Docs: front matter on every file, corrections **in place with a dated note**, superseded
  docs move directory rather than being deleted, closed verdicts rule out only their scope.
- Verify sync with `git rev-parse HEAD origin/rw/main`, not log counting.
- `rtk proxy <cmd>` shows unfiltered cargo/git output.
