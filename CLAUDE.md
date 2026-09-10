# rivoli (rewrite) — orientation for agents

Decode engine for LLMs bigger than memory: AMD Strix Halo gfx1151, unified LPDDR5 via GTT,
weights streamed from NVMe **overlapped with compute** — the overlap is the whole design.
Rust workspace + HIP/ROCm, one backend. This is the ground-up rewrite; the old tree stays
live as the parity reference at **tag `archive/glimmer-s2` @ `6b7f496e`** (was branch `wt/glimmer-s2`) — port from there, cite
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
  build, both feature arms). The gate belongs to the `crates/cli` unit, so a verification
  chain that never builds that package has not run it: measured 2026-09-04, an
  `-p rivoli-artifact`-only chain plus one `cargo test` that died at argument parsing let a
  38-token clone reach a commit. Cargo DOES re-run the script for an edit anywhere under
  `crates/` (measured — an appended newline advanced the unit's `output` file), so the hole
  was the chain, not the fingerprint; the last check before committing a `.rs` change is
  `cargo build -p rivoli` or a workspace arm. Precondition: rustfmt-clean, or the result is
  a lower bound.
  Its real floor is `minTokens: 15` **and** jscpd's unset `minLines` default of 5: a 4-line
  verbatim copy of a live function passed the gate (measured 2026-09-04, planted and
  removed), while 5-line/41-token and 13-line/182-token copies both redden it. So "zero
  budget" is zero clones of ≥ 5 lines, and a deliberate small copy is invisible — which
  matters when a refactor is meant to be proven caught as well as when one is being avoided.
  **9** regions are exempt via ignore markers (three in the ported frozen V4 oracle,
  where verbatim transcription of the reference is the point; four across the HIP ABI
  wall, which the 800-line ceiling split into `backend/hip.rs` — its extern declarations
  and its one hand-written launcher — plus one per macro-invocation file,
  `backend/hip_linalg.rs`, `backend/hip_blocks.rs` and `backend/hip_attn.rs` (the third
  invocation file, split out 2026-08-16 with M9's launchers), since a marker pair cannot
  span files; one in `core/routing.rs`, the frozen `route_into_pre` photograph — each argues
  in place;
  `quant.rs`'s parameter-list exemption died 2026-08-15 when the `Fp8W`/`VqW`/
  `RowScaledW` views paid the hop its note had priced; and one in
  `engine/examples/arena_repro.rs`, added 2026-08-18, where two examples sharing a module
  cannot avoid emitting `mod common; fn main() { … common::start() }` — the matched tokens
  ARE the sharing, and the only way to unmatch them is to stop sharing, which is strictly
  more duplication); the
  count is derived by `crates/cli/tests/docs.rs`, and marker text may appear only on a
  bare marker line.
- **CodeScene 10/10** — `crates/cli/tests/codescene.rs`, whole tree, hard threshold;
  standing red-proof fixture must score < 10 every run; exemptions argued in place and
  checked at both ends. Warn-and-skip without a license locally; CI hard-fails via
  `RIVOLI_CS_REQUIRED=1`. Needs `CS_ACCESS_TOKEN` in the environment.
- **line caps** — 1200 hard (`crates/cli/tests/line_limit.rs`, red-proofed), 800 soft
  (`cargo:warning` from the cli build script on every build: the next edit to a warned
  file should shrink it). CodeScene binds independently below both.
- **stale build-script binary tripwire** — `crates/cli/build.rs` asserts its compile-time
  `CARGO_MANIFEST_DIR` against the run-time one, so a build-script binary compiled for one
  checkout and handed to another by a shared target dir stops the build instead of scanning
  a sibling's tree; red-proofed 2026-08-30 (`docs/measurement/gate-red-proofs.md` §12, W2
  stage 2). It catches only the EXECUTED-stale-binary half: when both cargo units are fresh
  the script is never run and its cached output is replayed silently (W2 stage 1), and no
  assert can speak from inside a binary that does not execute — that half is closed only by
  real target-dir isolation, i.e. the explicit per-invocation `CARGO_TARGET_DIR`.
- **harness bash-guard** — `.claude/hooks/bash-guard.sh`, a `PreToolUse` hook on the `Bash`
  matcher in the versioned `.claude/settings.json`: it reads the tool payload as JSON on stdin
  and answers with an exit code (0 allows; 2 blocks and hands the reason back on stderr).
  Three rules, each evaluated **per segment** of the command line (`;`, `&&`, `||`, `|`,
  newline) and on the segment's **executable position**, never on a substring of an argument:
  **R1** reaches the device without `flock /var/run/sys-gpu.lock` *in that same segment*;
  **R2** a cargo build verb with no explicit `CARGO_TARGET_DIR` — and the lock is NOT an
  exemption, because device scheduling and build isolation are independent; **R3** a mutating
  git verb (`stash|checkout|restore|switch|clean|reset|merge|rebase`) on this shared tree,
  with `add` and `commit` deliberately allowed. It is the THIRD layer and not load-bearing:
  only Bash calls made through the harness are scanned, so `flock` stays the cross-tenant
  guard and the per-arm contention witness stays the post-hoc detector. Standing red proof:
  `crates/cli/tests/hook_guard.rs` — **76** rows driven straight into the hook, deviceless, on
  every `cargo test`, the count gated twice rather than remembered (`ROWS.len()` against the
  population in `tests/common`, and `docs.rs` against this sentence: CORRECTED 2026-09-04 from
  78, a hand-count that stood in four files for four days beside a 76-row table, §13); the same
  table scored **42 rows red** (33 false-allow, 9 false-block) against the pre-fix matcher,
  whose rules asked their questions of the whole command string
  (`docs/measurement/gate-red-proofs.md` §13).
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
- **device halves** — `tests/device-halves.sh [attend|fp8|all]`: M17c's block-attend kernel and
  M11's fp8 anti-fallback gate as witnessed arms. It sources `tests/gpu-witness.sh` so the flock
  and the contention witness are not optional, and it **never builds** (§12: a build inside the
  lock holds the device for compile time; a build between arms evicts page cache). Requires
  `CARGO_TARGET_DIR` naming this checkout's own dir, and refuses a `cargo`-built default.
  Both arms green 2026-09-04 on rh-anine (7/7 and 1/1, empty witness, GTT 17 MiB). Standing
  red proof: the classifier reddens (`RED … no scored tests`, exit 1) on a stub binary that
  exits 0 reporting `0 passed` — planted on the box, observed, removed. On-demand (GPU, ~1 min),
  not CI.
- **kernel census** — `crates/cli/tests/kernel_coverage.rs`: every launcher has an oracle
  suite or a live deferral, checked both ends; **66/66/0** as measured 2026-09-08 by `cargo test -p rivoli --no-default-features --test kernel_coverage -- --nocapture` (the figure stood at 61 for four milestones and nobody re-counted it — the same prose-drift class §13 caught in its own row count; `docs/reference/kernel-inventory.md` enumerates the same population row by row), and it rose past 61 when M17c's
  `gqa_block_attend` landed with its launcher and `kernel_glimmer_block_attend.rs`. Its
  DEFERRED row opened and closed the table's **third** turn within one commit — the census
  REFUSED the stale row rather than letting it stand, which is the both-ends check doing what
  an exemption list cannot. Its `covered` scanner also grew a second form that day: a launcher
  handed to a shared launch helper as its first argument counts, matched against a
  whitespace-stripped corpus. Both halves were forced — jscpd refused the second per-file
  launch wrapper, and without the new form a kernel covered since M7 read as uncovered. A
  second census in the same file (2026-09-10) asks the liveness question the oracle census
  cannot: every launcher has a production caller under `crates/engine/src`/`crates/cli/src` or
  a classified row in the inventory's uncalled table — both ends checked, the heading's row
  count derived from the table itself; red-proofed the same day by four planted-and-restored
  registry edits plus two kept failed plants (`docs/measurement/gate-red-proofs.md` §17). **INV-n registry** —
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

Every line below carries its own `CARGO_TARGET_DIR` because on this box that is the only
thing that isolates a build (`## Build cache and worktrees`); `rivoli` is the PRIMARY
checkout's name, a worktree substitutes its own. Spelled out per invocation rather than
`export`ed or held in a shell variable on purpose: a bash env prefix binds exactly the one
command it prefixes, and the guard reads the prefix that is actually there.

```bash
flock /var/run/sys-gpu.lock -c 'CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace -- --test-threads=1'
                                  # dev profile, host + device in one battery.
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo test --workspace --no-default-features   # the deviceless arm (CI's build); safe anywhere
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo clippy --workspace --all-targets                        # rocm arm
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo clippy --workspace --all-targets --no-default-features  # stub arm
CARGO_TARGET_DIR=/var/cache/rivoli/target/rivoli cargo build --release             # benchmarks and performance evaluation ONLY
```

**All five lines are payload rows in `crates/cli/tests/hook_guard.rs`, not prose.** The version of this block without the prefixes was refused by this file's own guard
(R2) — a quick-start a reader cannot run is a defect in the file, and it is the reason the
guard's rows include the canonical arms explicitly.

**A run that is not timing something is a dev-profile run.** `[profile.release]` compiles
out every `debug_assert!`; that is what benchmarks are measured under, and it is also why
a check that must hold in a shipped binary is an `assert!` and pays its cost.

## Build cache and worktrees

**Build output belongs on `/var/cache/rivoli`, not in the worktree — `/home` is NFS.** Every
node on this box's network mounts the same tree, so every artefact cargo writes and every
freshness `stat` it makes crosses the network. Measured 2026-08-13: `du -sh` on an
in-worktree `target/` **timed out at 120 s**, while the same command on the cache dir
returned in **0.006 s**. That per-stat cost is paid constantly, not once.

`/var/cache/rivoli` is a **btrfs subvolume** on the local NVMe with **`chattr +C`**
(nodatacow): no copy-on-write fragmentation and no checksum cost for output that is
rewritten constantly, and nested subvolumes fall out of any snapshot of `/` for free. **The
flag is inherited only by files created after it is set**, so it goes on the subvolume while
it is empty — it cannot be applied retroactively to a tree that already has content.

**Worktrees are created by `tests/new-worktree.sh <name> <branch> <base-sha>`, never by a
raw `git worktree add`.** They live under `.claude/worktrees/<name>`. The script is the one
place that gets all three of these right at once, and each is something that has already
gone wrong here silently:

- **the base sha is REQUIRED** — `worktree add -b` defaults it to the current checkout's
  HEAD, which is how a worktree ends up based off the primary tree instead of the sibling
  branch it was meant to continue;
- **one target dir per worktree — and the MECHANISM is an explicit environment variable,
  not the config file.** `/etc/security/pam_env.conf` line 6 has set
  `CARGO_TARGET_DIR DEFAULT=/var/cache/users/@{PAM_USER}/cargo-target` on every node since
  2026-08-13; it is deliberate machine config, so we adapt to it rather than remove it.
  Cargo's precedence is `--target-dir` > `CARGO_TARGET_DIR` > `build.target-dir`, so that
  login variable beats a `.cargo/config.toml` in every interactive and ssh shell — which is
  every shell an agent runs in. **Therefore every cargo invocation in every rivoli checkout
  carries its own target dir on the command line:**
  `CARGO_TARGET_DIR=/var/cache/rivoli/target/<name> cargo …`, where `<name>` is the
  worktree's name and is `rivoli` for the primary checkout. `env -u CARGO_TARGET_DIR cargo
  …` is the other working form: it strips the variable so the config binds. The script
  still writes `<wt>/.cargo/config.toml` holding
  `[build] target-dir = "/var/cache/rivoli/target/<name>"` and still refuses (exit 66) if
  another checkout's config already claims that directory, because that file is three
  things — declared intent, the claim registry the refusal reads, and the binding mechanism
  wherever the variable is absent (CI, `env -u`). It is not, on this box, what redirects a
  build. `docs/measurement/gate-red-proofs.md` §12 is where the config was OBSERVED losing
  to the variable silently — W3's first attempt returned two greens while certifying an
  isolation that had never happened — and §7e-bis is the defect the isolation exists to
  stop;
- **`/.cargo/` must be ignored by the COMMON gitdir** (`git rev-parse --git-common-dir`) —
  the per-worktree gitdir's `info/exclude` is NOT read, which costs a confused minute if you
  put it there. The script *verifies* this with `git check-ignore`, and on failure removes
  the worktree and branch it just made — via a cleanup trap armed the moment `worktree add`
  succeeds, so no failure path between there and the final print can leave a half-made one.

**A shared target dir is a correctness defect, not a slow build.** Cargo reuses the
build-script BINARY it finds there; `crates/cli/build.rs` bakes `env!("CARGO_MANIFEST_DIR")`
at its own compile time, so a sibling worktree's binary scans a `crates` directory this
checkout does not have. **The duplication gate did not run at all for that invocation, and a
commit landed on the false green** — `docs/measurement/gate-red-proofs.md` §7e-bis. The
build script now compares its compile-time and run-time `CARGO_MANIFEST_DIR` and panics
naming the defect class, so a stale binary is loud; a per-invocation
`CARGO_TARGET_DIR=/var/cache/rivoli/target/<name>` is what stops it arising, and it only
stops it while it is actually on the invocation.

**The tooling exception to "worktrees only via `tests/new-worktree.sh`".** Plugin and agent
tooling — `isolation: "worktree"`, the worktree skills — creates scratch worktrees under
`.claude/worktrees/` with a raw `git worktree add` and no `.cargo/config.toml`, and the
script cannot intercept that. Such a worktree must NEVER run cargo without an explicit
`CARGO_TARGET_DIR` on the invocation; the harness guard (`.claude/hooks/bash-guard.sh`, rule
R2 — see `## Gates`) is what enforces it, and `crates/cli/tests/hook_guard.rs` is what keeps
the guard honest.

**Read exit codes UNPIPED.** The same false green was a `cargo check … | grep … | head`
followed by an unconditional `echo "CHECK OK"`: the pipeline reported its last stage and the
echo printed a green that nothing had established. Redirect to a file, capture `$?` from the
command itself, then read the file.

The worktree-local `.cargo/config.toml` sets **`build.target-dir` only** — no profile, no
flags — and it is machine state in `info/exclude`, not a repo file, so CI never sees it. It
cannot change what `--release` compiles out; **read what such a file contains rather than
inferring from its presence or absence.**

**Scratch files go to `/var/cache/rivoli/scratch`**, not `/tmp`. `/tmp` is a **63 GB tmpfs
in RAM** that competes with the GPU's unified-memory budget, and it is session-scoped: a
sweep script written there is gone next session, and reading a census from a *previous*
session's scratch directory is how a stale result gets mistaken for a fresh one.

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
- Verify sync with `git rev-parse HEAD origin/main`, not log counting.
- `rtk proxy <cmd>` shows unfiltered cargo/git output.
