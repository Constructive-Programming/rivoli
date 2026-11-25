# Quality gates

<div class="rv-card">

**Every claim is a gate that can go red, and a gate is proven able to fail before its
pass is believed.** A check that has never failed is not evidence. A check that can
silently examine zero items is not a check.

</div>

## Test-first, against first-party references

- **Anchors come first.** Golden outputs come from the model's own reference stack, never
  a transliteration. They are vendored with self-describing pins and gated by a defect
  matrix: every planted defect must turn the gate red where it should.
- **Tolerances carry provenance.** They are measured from fp32 floors on at least two
  weight draws *before* the kernel exists. A tolerance picked to make a kernel pass is not
  a tolerance.
- **Red proofs are recorded.** Every new gate ships with a run that shows it failing.

## What runs

| gate | checks |
|---|---|
| duplication (jscpd) | zero clones, on every build |
| CodeScene | 10/10 across the whole tree |
| line caps | 1200 lines hard, 800 soft |
| clippy, warnings | denied as errors workspace-wide |
| parity | greedy token ids identical to the pinned reference |
| perplexity gates | paired per-token NLL inside a pre-registered equivalence band |
| bit-identity | every Rust kernel byte-identical to its HIP C++ predecessor |
| kernel census | every launcher has an oracle and a production caller |
| invariant registry | every documented invariant has a test, and the reverse |
| docs registry | every doc carries status, scope and verdict |

## Measurement discipline

- **The GPU is sole-tenant.** Every device run takes a lock and samples a contention
  witness. Any arm with another tenant present is discarded.
- **Quality is ranked on paired dNLL**, never on a perplexity column. An interval that
  straddles zero is inconclusive, not a pass.
- **Run the A-vs-A control.** Some streaming configurations are not bit-reproducible, so
  the noise floor is measured per arm. A difference below it is not a measurement.
- **An unwitnessed number is unciteable.**
