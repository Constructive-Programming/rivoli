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
  **8** regions are exempt via ignore markers (three in the ported frozen V4 oracle,
  where verbatim transcription of the reference is the point; four across the HIP ABI
  wall, which the 800-line ceiling split into `backend/hip.rs` — its extern declarations
  and its one hand-written launcher — plus one per macro-invocation file,
  `backend/hip_linalg.rs`, `backend/hip_blocks.rs` and `backend/hip_attn.rs` (the third
  invocation file, split out 2026-08-16 with M9's launchers), since a marker pair cannot
  span files; one in `core/routing.rs`, the frozen `route_into_pre` photograph — each argues
  in place;
  `quant.rs`'s parameter-list exemption died 2026-08-15 when the `Fp8W`/`VqW`/
  `RowScaledW` views paid the hop its note had priced); the
  count is derived by `crates/cli/tests/docs.rs`, and marker text may appear only on a
  bare marker line.
- **CodeScene 10/10** — `crates/cli/tests/codescene.rs`, whole tree, hard threshold;
  standing red-proof fixture must score < 10 every run; exemptions argued in place and
  checked at both ends. Warn-and-skip without a license locally; CI hard-fails via
  `RIVOLI_CS_REQUIRED=1`. Needs `CS_ACCESS_TOKEN` in the environment.
- **line caps** — 1200 hard (`crates/cli/tests/line_limit.rs`, red-proofed), 800 soft
  (`cargo:warning` from the cli build script on every build: the next edit to a warned
  file should shrink it). CodeScene binds independently below both.
- **warnings are errors, structurally** — `[workspace.lints.rust] warnings = deny` and
  `[workspace.lints.clippy] all = deny` in the manifest, so a local `cargo check`
  enforces what CI enforces (owner rule 2026-08-15; red-proofed with a planted unused
  variable). Per-file `#![allow]` needs its argument written beside it.
- **clippy** `--all-targets` plus the deviceless feature-union run.
- **docs registry** — status/scope/verdict front matter + INDEX agreement, test-enforced.
- **parity** — `tests/parity-glm.sh`: the rewrite's greedy ids token-identical to the
  pinned reference (prefix rule: the reference can stop at EOS, the smoke cannot yet),
  flock + descendant-pid witness per arm, never builds. On-demand (GPU, ~1 h), not CI.
  Red-proofed 2026-08-15 by a gate-codebook inversion, after two measured sub-threshold
  rungs (1-ulp: erased by fp16 narrowing; one sign flip: under argmax margins).
- **ppl gates** — `tests/ppl-gates.sh`: three cells over the M10 instruments — `profile`
  (the stamped phase buckets account for the decode wall, per-bucket census + a re-derived
  remainder), `p4` (P4 at NLL, THREE arms — A, a same-budget control A', and B — whose
  strictness CALIBRATES ITSELF: byte-identity is demanded of an arm whose control repeats
  byte-for-byte, and elsewhere the floor is measured and the budget's interval must contain
  zero where the control's does; every verdict carries its scored-position count and a
  strict-branch difference runs a second BUDGET arm before convicting — not a second control
  pair, which was the rejected first attempt and carries no information about the budget —
  because a one-off divergence does not recur while a real budget effect is stable. Re-specified 2026-08-17 after the byte-identity form
  reddened on GLM's own nondeterminism, not on the budget. **On a non-reproducing arm it
  reports UNCALIBRATED (exit 1), so it is a diagnostic there, not a merge gate**), `tf`
  (paired dNLL against
  the pinned reference inside a pre-registered ±ln(1.01) equivalence band; INCONCLUSIVE is
  never a pass). `--expect-red[=FRAGMENT]` inverts the classification so a red proof is
  judged by the gate's own code. On-demand (GPU, ~28 min, or ~34 min if the strict branch takes its fourth arm; + ~6 min per red-proof), not CI.
  Shares `tests/gpu-witness.sh` with the parity gate. Classifier half red-proofed
  2026-08-16 deviceless; the engine half is OWED (`docs/measurement/gate-red-proofs.md` §5).
- **smoke** — `tests/smoke-glm.sh`: the CLI end to end — every legality refusal asserted
  against the table's own message fragments, the bench cell pinned to the recorded
  reference ids, a live serve round-trip (readiness, /v1/models, non-stream, SSE).
  On-demand (GPU, ~45 min), not CI. Red-proofed 2026-08-16 (wrong fragment reddens).
- **kernel census** — `crates/cli/tests/kernel_coverage.rs`: every launcher has an oracle
  suite or a live deferral, checked both ends; 60/60/0 since M9. **INV-n registry** —
  `crates/cli/tests/invariants.rs`, doc-and-test must move together. **feature matrix** —
  `tests/feature-matrix.sh` + `crates/cli/tests/matrix.rs` (lists derived from the
  manifests; the resolve cell proves `--no-default-features` is genuinely deviceless).
  All landed with what made each non-vacuous (M1–M3); this line said "landing later"
  until M9 closed.

## Build and test

**The build IS the rocm build** (owner rule 2026-08-16): `default = ["rocm"]` at every
level of the DAG, so a bare `cargo build` is the real engine and a bare
`cargo test --workspace` is a GPU arm — flock it and serialize it like any device suite.
The deviceless refusal-stub arm is `--no-default-features`; it is what CI's hipcc-less
runner builds, and the feature matrix pins every cell's exact set with it.

```bash
flock /var/run/sys-gpu.lock -c 'cargo test --workspace -- --test-threads=1'
                                  # dev profile, host + device in one battery.
cargo test --workspace --no-default-features   # the deviceless arm (CI's build); safe anywhere
cargo clippy --workspace --all-targets                        # rocm arm
cargo clippy --workspace --all-targets --no-default-features  # stub arm
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
  nothing — read the text. **And run the A-vs-A control**: GLM int3-vq does not repeat
  itself (0.0018 nats of mean dNLL between two runs at identical flags, 2026-08-17), while
  Glimmer fully pinned is byte-identical — so the floor is per arm, it is a property of
  streaming, and a comparison below it is not a measurement.
  `docs/investigations/glm-nondeterminism.md` bounds it;
  `docs/measurement/how-to-measure.md` opens with the three rules.
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
