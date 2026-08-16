---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). M11's fp8 gates are PAID deviceless — layer_bytes stripped of its scale grid, sniff falling back to the compiled-in block, the converter at the wrong block and with one projection class skipped, and the parity script over six runs including both refusals and a green baseline — while its DEVICE half is OWED with recipes written down: the anti-fallback assert in glimmer_fp8_decode.rs. Slot::fill's third-zip-leg guard was RETIRED rather than proven - the parameter it checked was deleted, so the truncation has no shape.
---

# M0 gate red proofs

Run 2026-08-15 on `rw/main` at commit `e922a34`, per P7: a gate that has never been red is
not evidence. Each proof was planted, observed red, reverted, and observed green again in
the same session. The exact reddening output is quoted; everything else is reproduction.

## 1. jscpd (duplication, build error)

Planted: the body of `rivoli_core::hash::fnv1a` copied into `crates/artifact/src/glimmer.rs`
as `planted_clone` (26 tokens — above the 15-token floor).

```
npx --no -- jscpd -c .jscpd.json --exitCode 7 crates   → exit 7   (red)
git checkout crates/artifact/src/glimmer.rs
npx --no -- jscpd -c .jscpd.json --exitCode 7 crates   → exit 0   (green again)
```

This gate also fired **unplanted, twice, during the M0 port itself** — the CodeScene
gate's private `fnv1a` against the ported `golden_read.rs` copy — which is a stronger
proof than the synthetic one: it caught duplication nobody believed they had written.
The fix (one owner in `rivoli-core::hash`) is commit `e922a34`.

## 2. Docs registry (front matter ↔ INDEX agreement)

Planted: `TAMPERED ` prepended to `how-to-measure.md`'s front-matter verdict, INDEX row
left untouched — the exact "corrected in one place only" drift the gate exists for.

```
test the_index_lists_every_doc_with_a_matching_verdict ... FAILED   (red)
git checkout docs/measurement/how-to-measure.md
cargo test -p rivoli --test docs                        → 3 passed  (green again)
```

The derived exemption ledger (third test in the same file) also fired unplanted during
the port: the frozen V4 oracle brought 3 ignore-marker regions and the test refused
CLAUDE.md's stale `0` until the ledger line was updated.

## 2b. Invariant registry (added 2026-08-15, M1)

Planted: the §8b table's `INV-1` renamed to `INV-9`, which breaks BOTH directions at
once — INV-9 documented with no test, `inv_1_*` tested with no row.

```
INV-[9] documented in architecture.md §8b with no `inv_<n>_*` test. ...   FAILED  (red)
revert → 1 passed                                                          (green again)
```

## 3. CodeScene (10/10 code health)

Two halves, one still owed:

- **Required-mode half, proven:** with no `CS_ACCESS_TOKEN` in the environment,
  `RIVOLI_CS_REQUIRED=1 cargo test --test codescene` panics —
  `CodeScene gate REQUIRED but cs did not run (crates/artifact/src/glimmer.rs): not JSON
  (unlicensed cs prints PAT prose here)` — naming the file and the cause. So CI (which
  sets the variable) cannot silently skip; classification is by output, and exit codes
  were measured to be uninformative (unlicensed `cs review` exits 1 like any failure).
- **Score half, OWED:** the standing fixture (`codescene-redproof/bad.rs.txt` must score
  < 10) and a planted-unhealthy-function run both require a licensed `cs`. The dev box
  has a cached cloud license JWT (exp 2026-08-31) but `cs review` demands the
  `CS_ACCESS_TOKEN` PAT, which is not in this environment. **First licensed run must
  execute both proofs and correct this doc in place with the observed scores.** Until
  then every local `cargo test` runs this gate in warn-and-skip mode, and the skip is
  invisible under libtest capture — stated here so nobody reads local green as health.

## 4. Muse Glimmer anchor decode (added 2026-08-16, M7's exit gate)

`crates/engine/tests/glimmer_anchor_decode.rs` scores the engine against the reference's own
logits; `crates/engine/tests/glimmer_anchor_widths.rs` is its deviceless half. **Both halves
are proven** — the deviceless half in its authoring session, the device half the same day
(below), on the box, under the flock.

**Proven, no GPU.** Both planted, observed red, reverted, observed green again:

```
# a renamed capture — the way a golden gate silently stops scoring
BRANCH_CAP → "post_feedforward_layernorm2.out"
  t0.L0.post_feedforward_layernorm2.out is not in the golden; it holds 1099 float
  tensors, e.g. ["t0.embed_norm.out", "t0.rope.cos", "t0.rope.sin"]     FAILED   (red)
revert → 2 passed                                                                (green again)

# a bound dropped under the envelope it was measured from
TOL 2e-1 → 1e-1
  TOL is 1.47x its measured envelope 6.8e-2. Under ~2x it reddens on a correct
  engine …                                                              FAILED   (red)
revert → 2 passed                                                                (green again)
```

**PAID 2026-08-16, same day, first GPU session.** All four rows executed on gfx1151 under the
flock, each reverted byte-identical, green after (3/3 in 0.38–0.68 s per run):

| # | mutation | observed red | `old:` said |
|---|---|---|---|
| 4 | both budgets roomy | `tight budget pinned 8 and streamed 0 with 0 fills` FAILED | `streamed 0` |
| 1 | `qk_scales` → q clamped to 1.0 | logits **6.779e-1**, branch 3.182e-1; the P4 test rightly stayed green (both arms wrong identically is still bit-identical) | 6.7e-1 |
| 2 | `rotated` predicate inverted | logits **1.113e0**, branch 5.094e-1 | 1.1e0 |
| 3 | k/v handles swapped at the attend launch | **1.023e0** / 8.364e-1 | 1.3e0 |

This tree's own green worsts (`--nocapture` — libtest eats a passing test's prints): logits
**5.742e-2** (text-1) / **6.954e-2** (text-2) vs `TOL` 2e-1; branch **2.993e-2** at L6 vs
`TOL_BRANCH` 1e-1. Note 6.954e-2 sits 2% ABOVE `old:`'s truncation-measured envelope top of
6.8e-2 — the round-to-nearest prediction ("slightly under") was wrong in direction and right
in magnitude; the margin to the bound is 2.9×, and this line is now the envelope's owner.

**Two operator false-greens during the paying, both the same trap, both recorded:** rows 4 and
1 first "ran" against a STALE binary because the mutation orphaned a variable, `warnings =
deny` failed the rebuild, and the build's exit code had been eaten by `| tail` /
`echo $?`-after-pipeline. The tell was a red-proof that refused to go red (row 4) and a red
with the WRONG failure (row 1 reddened the partition test, not the logits). Debug the
harness before the tree: check the build's exit UNPIPED, and read WHICH test failed, not
just that one did.

## 5. Muse Glimmer fp8 (added 2026-08-16, M11)

Five gates arrive with `--fp8`: three deviceless (two in `glimmer/geometry.rs`'s test
module, one in `crates/cli/tests/glimmer_convert.rs`), one shell (`tests/convert-parity-
glimmer-fp8.sh`), one on device (`crates/engine/tests/glimmer_fp8_decode.rs`). **The four
deviceless ones are PAID below; the device one is OWED and its recipe is written down.**

**Reverts were hand-edits, verified by `sha256sum -c`, not `git checkout`** — this branch's
work is uncommitted, so the restore command the sections above use would have discarded it.
The hash file is the evidence the tree came back byte-identical; every proof below was
followed by `sha256sum -c` reporting `OK` for the mutated file.

### 5a. `layer_bytes` charges the fp8 scale grid

Planted: the `ProjFmt::Fp8` arm reduced to `n` — the packed byte per weight with the f32
grid dropped, which is the size a reader who thought "fp8 is one byte" would write and
which under-reserves the tier by exactly what `DeviceTier::place` then bails on.

```
test glimmer::geometry::geometry_tests::layer_bytes_charges_each_format_its_own_size ... FAILED
assertion `left == right` failed
  left: 33792     (1_024 + 32_768,        the grid omitted)
 right: 33824     (1_024 + 32_768 + 8*4,  eight [1,1] grids)     (red)
revert → 8 passed                                                (green again)
```

### 5b. `ProjFmt::sniff` reads the stamp, never a compiled-in constant

Planted: `FormatMeta::load(dir).map(|m| m.fp8_block).unwrap_or(quant::FP8_BLOCK)` — the
plausible defect, since the default IS what the converter stamps today and the fallback is
invisible until someone converts at another block.

```
test glimmer::geometry::geometry_tests::sniff_reads_the_dtype_and_consults_the_manifest_only_for_fp8 ... FAILED
assertion failed: ProjFmt::sniff(&dir, &st).is_err()               (red — the stampless
                                                                    fp8 fixture was accepted)
revert → 8 passed                                                  (green again)
```

### 5c. The converter quantizes the projections, at the shipped block, and nothing else

Two mutations, each reddening a different assertion of the one test.

```
# (i) the wrong block constant — `add_quantized_fp8(&src, base, 64)`
assertion `left == right` failed: model.language_model.layers.0.mlp.gate_proj.weight_scale_inv: grid shape
  left: [2, 1]                                                     (red)
 right: [1, 1]
```

That is the fixture's discrimination claim paying out on exactly the tensor it names: `INTER`
= 96 against a 64 block gives `div_ceil(96, 64) == 2`. It also fixes the boundary the test's
own doc used to get wrong — the second grid row appears at any block **strictly below** 96,
not at 96, where `div_ceil(96, 96) == 1`.

```
# (ii) one projection class skipped — `&& !name.ends_with("mlp.down_proj.weight")`
Error: --fp8 quantized 28 tensors; this checkpoint's 4 layers imply 32   (red — the
                                                                          converter's own
                                                                          count `ensure!`)
revert (both) → 4 passed                                                 (green again)
```

### 5d. `tests/convert-parity-glimmer-fp8.sh`

Its header prescribed a red proof and claimed "the M11 record logs both runs" before any
record existed. Five runs, on six-file scratch directories holding one line each (the script
is a byte comparator; artifact-shaped content would not exercise anything more of it):

| run | input | result |
|---|---|---|
| green baseline | two distinct, identical dirs | `PARITY: every file byte-identical`, exit **0** |
| one byte flipped | `resident.safetensors` last byte `s`→`S` | `DIFFER` + `PARITY FAILED`, exit **1** |
| self-comparison | same dir twice | `REFUSED: both arguments resolve to …`, exit **66** |
| self-comparison, respelled | `dir` vs `dir/.` | same refusal, exit **66** — `realpath` earns its place |
| symlinked member | distinct dirs, `resident.safetensors` a symlink to the ref's | `REFUSED: resident.safetensors resolves to the same file in both directories`, exit **66** |
| missing member | `chat_template.jinja` absent from cand | `chat_template.jinja: MISSING (cand)` + `PARITY FAILED`, exit **1** |

The green baseline is listed deliberately: a gate proven only red is a gate that might be
unconditionally red, which is the same false evidence in the other direction.

### 5e. `glimmer_fp8_decode.rs` — OWED, device

The gate makes two claims and each needs its own mutation. Run under the flock,
`-- --test-threads=1 --nocapture`, dev profile; revert by hand-edit and `sha256sum -c`.

| # | mutation | must redden |
|---|---|---|
| 1 | `proj`'s `ProjPin::Fp8` arm dispatched to `launch_gemm_bf16` on `w.packed` cast to `*const u16` — the named silent-fallback failure, made real | the anti-fallback assert (`differing > 0`) |
| 2 | `Slot::fill`'s destination walk shortened by one (`.take(self.tensors.len() - 1)`) so the last tensor keeps layer 0's bytes | the P4 assert (`split.logits == all.logits`) |
| 3 | `quantize_artifact`'s grid census asserted against a fixture whose `inter` is under the block | the census assert — this one is a standing claim about coverage, not a defect |

Row 1 is the one that matters: a green anti-fallback assert is the ONLY structural evidence
that fp8 arithmetic ran, and it has never been red. Until it is, read a green on this file
as "the two artifacts differ", not as "the fp8 kernel is dispatched".

### 5e-bis. An operator false-green during the paying, and it is a NEW combination

Recorded on the same rule as §4's pair. Between two of the runs above, `cargo check
--workspace --all-targets` **failed** with

```
error: failed to run custom build command for `rivoli v0.1.0 (…/wave-m11/crates/cli)`
  process didn't exit successfully: … build-script-build (exit status: 101)
  --- stdout
  cargo:rerun-if-changed=…/wave-m19/crates
```

and the failure was invisible, because the command was `cargo check … | grep -E
'^(error|warning)' -A 5 | head -20; echo "CHECK OK"` — the grep matched, `head` swallowed the
rest, and the unconditional `echo` printed a green that nothing had established. **A commit
was made on it.** (It was sound; that is luck, not evidence.)

The cause is the fourth shared resource: `CARGO_TARGET_DIR` is one directory for every
worktree on this box, and `crates/cli/build.rs` bakes `env!("CARGO_MANIFEST_DIR")` at its own
compile time. A sibling worktree (`wave-m19`) compiled the build script last, so cargo reused
ITS binary, which scanned a `crates` directory this checkout does not have and panicked. The
jscpd gate did not run for that invocation at all.

**Two rules, both already in this repo's notes, that only bite together:** read the exit code
UNPIPED, and isolate `CARGO_TARGET_DIR` for any run whose green you intend to cite. Every gate
in §5 was re-run afterwards under `CARGO_TARGET_DIR=/var/db/rivoli/m11/verify-target` with the
exit code captured directly — deviceless suite exit 0 (**298 passed, 0 failed, 81 binaries**),
clippy exit 0 on both arms, `cargo fmt --check` exit 0, jscpd exit 0 (0 clones).

### 5f. `Slot::fill`'s third zip leg — RETIRED rather than proven

A gate that cannot exist needs no red proof, and this one stopped existing the same day it
was written. Kept as a record because the sequence is the lesson.

Correctness review found `Slot::fill`'s `.zip(&self.lens).zip(tails)` unguarded: `tails`
arrived as a parameter, a short one truncates a zip **silently**, and `Slot::new` asserted
only `addrs == lens`. The first fix was a runtime `ensure!` comparing the two lengths — a
real check, and one whose red proof needs a `DeviceTier` and therefore a contended GPU.

The shipped fix instead **deleted the parameter**: `Slot` now holds `Vec<(String, usize)>`,
name beside length, built by `Slot::new` from `layer_tails` and never handed in from
outside. `fill` zips two sequences that are both the slot's own and were built together, and
there is no third leg to be short. `GlimmerPin` lost its `tails` field with it.

**A guard you can delete beats a guard you must prove**, and on this box "must prove" meant
booking sole-tenant device time to demonstrate a failure the type system can refuse instead.
`Slot::new`'s `ensure!(addrs().len() == tensors.len())` **stays** and is not in the same
class: it compares two genuinely independent walks — the pin's field-by-field placement
against the config-driven tail list — which can diverge with nobody noticing.
