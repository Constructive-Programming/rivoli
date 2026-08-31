---
name: model-port
description: The checklist for bringing a new model into rivoli — config pins, tokenizer, converter, first-party anchors, kernel census, docs registration, and the exit smoke. Invoke at the START of any model port or artifact-format port, and again before calling one done; each item below is a scar from one of the four ports already on record, so treat it as a census to answer, not advice to weigh.
---

# Porting a model

Four ports are on record (GLM, V4, Glimmer, K3). Every line here cost one of them time.
Work the list in order — the ordering is the TDD rule: **the thing that scores the code
exists before the code.**

## 1. Vendored config pins — byte-compared, never eyeballed

Vendor the upstream `config.json` (and any tokenizer/generation config the runtime reads)
under `docs/measurement/<model>-reference/`, and make the gate **recompute the pin from
the live artifact file** and compare bytes.

**Scar:** a pin checked only against a frozen *copy of itself* is decoration — it agrees
with itself forever while the artifact drifts. A structural test catches a field that
moved; it never catches one that was **added**. Recompute (FNV-1a over the bytes — do not
add a sha256 dependency for this), compare, and let the diff name the key.

## 2. Tokenizer TYPE first — before any weights move

Establish the tokenizer *type* and its exact vocab/merges source before downloading
anything large. **Scar: K3 re-downloaded 1.42 TiB** because the type was settled after the
weights instead of before them.

Also check where the **chat template** actually lives: `chat_template.jinja` has shipped
only in the fp8 *source* repo (`manifest i4_source.src`), not in the converted artifact,
and rivoli's hand-port drifted to another family's role framing for months without a gate
noticing. Pin the template's bytes the same way as §1.

## 3. Converter — `ensure!` the exact tensor counts, exclusions BY NAME

The converter asserts the **exact** number of tensors it consumed and the exact number it
emitted, and every tensor it deliberately skips is listed **by name** in the exclusion set.

Not "roughly all of them", not a `>=`, not a family prefix that silently absorbs a tensor
the next revision adds. A count with a named exclusion list is the only form where a new
upstream tensor is a loud failure instead of a quiet drop.

## 4. First-party anchors + a defect matrix — before the kernels

Goldens come from the **first-party reference stack**, never a transliteration of it, and
they are vendored with self-describing pins. Each anchor ships with a **defect matrix**:
a set of planted defects where each one is shown to **redden where it should** and to
**hold where it should**.

- A check that has never been red is not evidence. A plant must be OBSERVED to modify the
  tree *and* redden the assertion you added — of 17 plants once run, 3 lied (rustfmt ate
  one; earlier asserts masked two).
- **Tolerances carry provenance**: measured from operator fp32 floors on ≥ 2 weight draws,
  *before* the kernel exists, citing the measurement. A tolerance picked to make a kernel
  pass is not a tolerance; a round number is a confession.
- If the reference is triton-only or otherwise device-bound, generate the goldens on the
  GPU **once**, vendor the bytes, and let the gate read them deviceless.
- Score both plausible axis orders of any square tensor against the reference output — a
  `[D][D]` state passes every shape check under either interpretation.

## 5. Kernel census — both ends, every launcher

`crates/cli/tests/kernel_coverage.rs` requires every launcher to have either an oracle
suite or a **live** `DEFERRED` row, and it checks **both ends**: a launcher with no row is
a failure, and a row naming a launcher that is now covered is *also* a failure.

**Scar:** a DEFERRED row opened and closed within a single commit because the census
REFUSED the stale row rather than letting it stand — which is the both-ends check doing
what an exemption list cannot. Never park a row "until the oracle lands".

## 6. Registration lands in the SAME change

Three things move together with the port, not after it:

- the model's **`SCOPES` entry** in `crates/cli/tests/docs.rs`;
- the **plan/investigation doc** under `docs/`, with `status:`/`scope:`/`verdict:` front
  matter (see the `docs-registry` skill);
- its **one INDEX row**, whose verdict cell matches the front-matter verdict.

A port whose doc is registered "next commit" is a red gate for whoever pulls next.

## 7. The thin end-to-end smoke lands WITH the exit gate

The port's exit gate is not the last kernel oracle — it is a **thin end-to-end smoke**
(`tests/smoke-<model>.sh`): the CLI door to door, every legality refusal asserted against
the refusal table's own message fragments, a bench cell pinned to recorded reference ids.
It is red-proofed like any gate (a wrong fragment must redden it).

**Scar: `tests/smoke-k3.sh` never existed.** It was scheduled for "after the port", and
later means never. Land the smoke in the same change as the exit gate, thin if it must be
thin.

## 8. Name code for behaviour + ABI, never for the model

No `v4_*`, no `k3_*` in kernel, trait or struct names. Name what the code **does** and the
ABI it does it over; **model membership is data in a checked census table**. Format names
(`fp8`, `int4`, `vq`) describe the ABI and are fine.

## Before calling it done

Answer all eight out loud, each with the file or run that closes it. An item closed by
memory rather than by an artifact is open.
