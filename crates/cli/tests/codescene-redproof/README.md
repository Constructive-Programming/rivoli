---
status: data
scope: engine
verdict: A deliberately unhealthy Rust file (deep nesting, six arguments, duplicated conditional blocks) vendored so the CodeScene gate can prove it goes red on every run; if cs ever scores it 10/10 the gate is blind and says so.
---

# codescene-redproof

`bad.rs.txt` is scored via stdin by `tests/codescene.rs::the_red_proof_fixture_scores_below_ten`.
`.txt` keeps it out of rustc, jscpd, and the walk; `--file-name bad.rs` makes cs score it as Rust.
Do not "fix" it — its unhealthiness is the payload.
