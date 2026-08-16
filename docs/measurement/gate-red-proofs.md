---
status: data
scope: engine
verdict: Every M0 gate and the M1 invariant registry were shown red before its green was believed — jscpd exit 7 on a planted 26-token clone, the docs registry FAILED on a one-sided verdict edit, the exemption ledger fired twice for real during the port, and RIVOLI_CS_REQUIRED turned CodeScene tool-absence into a panic naming the file; the CodeScene score-below-10 half is owed and standing, blocked only on CS_ACCESS_TOKEN. M7's anchor-decode gate is proven red in BOTH halves — deviceless (an absent capture name, a tolerance under its envelope) and on device (all four recipe rows executed 2026-08-16 with observed magnitudes matching old:'s, plus two recorded operator false-greens whose lesson is part of the record). M11's fp8 gates are PAID deviceless — layer_bytes stripped of its scale grid, sniff falling back to the compiled-in block, the converter at the wrong block and with one projection class skipped, and the parity script over six runs including both refusals and a green baseline — while its DEVICE half is OWED with recipes written down: the anti-fallback assert in glimmer_fp8_decode.rs. Slot::fill's third-zip-leg guard was RETIRED rather than proven - the parameter it checked was deleted, so the truncation has no shape. M11b's id pin is PAID on the real 27 MB tokenizer, 31 of 31 cases identical to apply_chat_template and red-proofed by closing a system turn with the non-stop token; its serve door ships BOTH halves (request framing and reply channel-splitting) after the request-half-only version was written and reverted as a regression, behind SIX red-proofed pure gates including the arch dispatch itself; a prefix-monotonicity property caught two streaming P0s that no non-streaming gate could see - a raw turn header streamed then the channel wedged forever, and a partial <|eot|> at the prefix boundary. Only the live SSE round-trip is OWED on the GPU.
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

## 6. Muse Glimmer chat framing (added 2026-08-17, M11b)

Until M11b every Glimmer prompt — bench and serve — was framed with **GLM's** chat template.
`glimmer_encoding.rs` had existed since the port and was wired to nothing; `main.rs`'s own
comment called it "a KNOWN GAP, not a decision" and said the change was owed an id-pinned
comparison first. This is that comparison.

### 6a. The id pin — PAID, deviceless

`crates/artifact/tests/glimmer_template.rs::rendered_prompts_tokenize_to_the_vendored_ids`
runs `render` → `Tokenizer::encode` → `case["ids"]` over all 31 vendored cases, on the
checkpoint's own 27 MB `tokenizer.json`.

```
RIVOLI_GLIMMER_ARTIFACT=/swarm/storage/ai/rivoli/glimmer-30b-fp8   cargo test -p rivoli-artifact --test glimmer_template --no-default-features -- --nocapture
  id pin: 31 cases tokenized identically to apply_chat_template
test result: ok. 5 passed                                                        (green)
```

**This closes a gap that had been recorded as owed and unclosable.** The sibling census test's dated
correction states the property — every special resolves to ONE id — is "true and UNVERIFIED",
and names the exact run that would close it, adding "the tiny fixture has none". A real
tokenizer is on this box now, and the run is above.

**Red proof.** Planted: `system_tail` closes with `<|eom|>` instead of `<|eot|>` — a one-token
change, and the one that decides whether a decode can STOP (only `<|eot|>` is a stop id).

```
case `plain_user` diverges at id 49 (got 57 ids, want 57)
  got  ...[392, 2540, 706, 392, 1556, 4205, 200007, 200022, …]
  want ...[392, 2540, 706, 392, 1556, 4205, 200008, 200022, …]      FAILED   (red)
revert (sha256 verified) → 5 passed                                          (green again)
```

**What the id half adds over the byte pin beside it**, since any `render` mutation reddens
both: the byte pin never calls a tokenizer, so it cannot see whether `<|start|>` became one
id or five ordinary pieces — that is decided by the `tokenizers` crate reading the
checkpoint's added-token table, outside this crate entirely. The reddening above shows ids
200007/200008 as SINGLE tokens, which is the property being pinned.

**Without `RIVOLI_GLIMMER_ARTIFACT` the test asserts its own reason** rather than returning
green having compared nothing: the 31 cases must still be present. An `eprintln!` skip would
be invisible under libtest capture, which is why "it skips loudly" is not a thing here.

### 6b. The serve door's six pure gates — PAID, deviceless

**The request half alone was written and then REVERTED.** Review found that framing a Glimmer
request with Glimmer's template while reading its reply back with GLM's would produce, for
every served reply, either an empty `content` (the GLM split hunts a `</think>` that is not
there and calls everything reasoning) or one leaking raw `to=user<|message|>` onto the user's
screen. That is a REGRESSION on a door that worked, and worse than the defect it was fixing.
**Both halves land together or neither does**, and `serve::split_channels`'s doc says so where
the next reader will be.

Both halves are pure functions of the request body and the generated text, so they live in
`serve/oai.rs` — the module whose header says it is where the pure functions and the tests are
— rather than beside their callers. (The first draft of this line justified that with "`frame_prompt`
takes a `&Ctx` holding `&mut Engine`" — false, and believing it is what left the dispatch
ungated; the real reason is that `frame_prompt` needs a 27 MB tokenizer.) Six gates, `cargo test -p rivoli --bin rivoli` — five
in `serve/glimmer.rs`, one in `serve/mod.rs`. **Each was planted, observed red, reverted, and
`sha256sum`-verified**, per this document's own standard for the word PAID:

| gate | planted defect → observed red |
|---|---|
| `a_developer_turn_is_framed_as_a_system_turn_and_not_dropped` | rewrite disabled → FAILED. Glimmer's role chain has no `else`, so an unmapped `developer` turn renders as NOTHING — instructions dropped behind a 200 |
| `the_reasoning_strength_maps_the_request_over_the_servers_default` | `if think { effort }` → `effort.or(…)` → FAILED. All five cells; **two are rivoli's invention** on a template with no thinking boolean, and an invented mapping with no gate is prose |
| `a_body_without_messages_is_refused` | `.context(…)?` → `.unwrap_or_default()` → FAILED. Otherwise `render` emits a system block and a bare generation prompt, and the model answers nobody |
| `split_glimmer_reads_the_recipient_and_never_swallows_a_reply` | two hardenings, quoted below. Eight cases, including the one that matters: text with no markers comes back WHOLE as content, because an empty `content` reads as a working server returning nothing |
| `every_prefix_of_a_generation_grows_both_channels_monotonically` | **needed no plant — it went red on the tree it was written for, twice.** See §6b-P0 |
| `the_reply_is_read_back_with_the_same_template_that_framed_it` (`serve/mod.rs`) | arms crossed → FAILED with `left: (" to=user<\|message\|>hi<\|eot\|>", "")` — content empty, the whole raw turn in reasoning. This is the DISPATCH, and until it existed both leaves were gated and the `match` pairing them was not |

#### 6b-P0. The streaming defect the monotonicity gate caught, and the one it caught next

**A P0 found by review before it shipped, then a second one found by the gate written for the
first.** Both were in `split_glimmer`, both streaming-only, and the non-streaming path was
correct throughout — so §6c's owed device recipe, a plain `curl`, would have come back green
over a broken door.

1. **The whole-text fallback fired on PREFIXES.** `render` ends the prompt at
   `<|start|>assistant`, so the first tokens a model emits are the turn header ` to=user` —
   several tokens before `<|message|>`. With no `<|message|>` yet, the fallback returned the
   raw header as content and `stream_decode` sent it. Then `<|message|>` arrived, content
   collapsed to the body, `strip_prefix(" to=user")` failed on it and on every token after, and
   `sent_c` froze. **Every streamed Glimmer reply would have been the literal text ` to=user`
   and nothing else, `finish_reason: "stop"`.** Fixed by `complete: bool`.
2. **A partial marker at the prefix boundary.** With (1) fixed, the new gate went red anyway:
   `prefix 36 reported content "The sky is blue.<", which the finished "The sky is blue." does
   not start with`. The `<` is three tokens short of `<|eot|>`; emitting it wedges the channel
   exactly as the header did. Fixed by `trim_partial_marker`, which is `delta`'s U+FFFD rule
   for markers instead of codepoints.

The gate is therefore its own red proof twice over, which is stronger evidence than a plant:
it caught duplication of a failure mode nobody had thought of yet. **Write the property, not
the cases** — a hand-written case list starts at a `<|message|>`, and both defects lived
strictly before one.

**Two operator false-greens while paying the dispatch proof, both the §4 trap again.** The
first mutation (`Arch::MuseGlimmer if false`) made the match non-exhaustive; the second
orphaned `complete`. Both were BUILD failures under `warnings = deny`, and both classifiers
read "no FAILED in the output" as green. The tell was a red-proof that refused to go red — the
signal §4 already names. A mutation for a red proof has to keep the tree COMPILING, which on a
`-D warnings` crate means keeping every binding live.

**`split_glimmer`'s two hardenings are red-proofed**, both planted, observed red, reverted,
sha256 verified identical:

```
# "earliest terminator wins" -> "try <|eot|>, else <|eom|>"
  left: ("", "A<|eom|>B")     right: ("", "A")                            FAILED  (red)
# `header.trim() == "to=self"` -> `header.contains("to=self")`
  left: ("CALL", "")          right: ("", "CALL")                         FAILED  (red)
revert both -> 1 passed                                                          (green again)
```

Read the left-hand sides: the first LEAKS `<|eom|>` into the text a user sees, which is the
whole class this function exists to stop; the second files a `to=selfcheck.run` tool call as
reasoning and drops it out of `content` entirely. Neither is hypothetical — both were the
naive spelling, and both were written that way first.

**Two scope lines are drawn in code and stated, not left to be discovered:**

- **`tools` is withheld from Glimmer's template.** It would render the ATEM
  `<atem:function_calls>` preamble while `oai::parse_tool_calls` reads GLM's `<tool_call>`
  markup and nothing else — so every tool use would return as prose with
  `finish_reason: "stop"`, a confidently wrong answer. A Glimmer tool request gets an honest
  untooled reply until an ATEM parser is ported.
- **DeepSeek-V4 still takes GLM's framing on the serve door**, and Kimi-K3's arm is
  unreachable (`main` refuses `--port` for it first). Both matches are EXHAUSTIVE, which is
  what makes the V4 gap visible instead of hidden behind an `if arch == MuseGlimmer` — the
  shape the first draft had, and the reason it was rejected.

### 6c. The serve round-trip — OWED, device

What §6b cannot reach: that a real decode through the real template TERMINATES. Needs a live
`Engine`, hence the GPU. Recipe, under the flock, dev profile:

```
rivoli /swarm/storage/ai/rivoli/glimmer-30b-fp8 --port 18174 &
curl -s localhost:18174/v1/chat/completions -d '{"model":"g","messages":[{"role":"user","content":"Say hello."}],"max_tokens":16}'
```

Green = a reply that TERMINATES on `<|eot|>` rather than running to `max_tokens`. Red proof:
point the arm at `encode_chat_turns` (the pre-M11b behaviour) and watch the reply run to the
limit — the old tree's 56-run retraction is that failure at scale.

**The second cell is the reply's SHAPE**: `content` non-empty, carrying no `<|message|>`,
`to=user` or `<|eot|>` fragment. That is the half §6b's `split_glimmer` gates on synthetic
text and only a real decode can gate on real output.
