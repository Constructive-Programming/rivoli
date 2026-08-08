#!/usr/bin/env python3
"""Score two `--logit-dump` traces against each other — §M9's drift gates (b) and (d).

The dump format is defined at `eval::LogitTrace::for_dump` since 2026-08-08 (it was
`V4Engine::arm_logit_trace`'s, and BOTH architectures now write it — the `V4LT` magic is a
kept misnomer, see that doc): `b"V4LT"`, vocab as u32 LE, then per decode position a u32 LE
(the engine's OWN argmax at that position, given its — possibly forced — history) followed
by `vocab` f32 LE logits. Two arms teacher-forced on the SAME `--force-tokens` file are
positionally comparable at every position, which is what lets this count argmax flips over
all of them instead of up to the first divergence.

    python3 docs/measurement/probes/v4_logit_drift.py A.lt B.lt          # gate (b)
    python3 docs/measurement/probes/v4_logit_drift.py --tokens A.lt      # emit force file
    python3 docs/measurement/probes/v4_logit_drift.py --identical A.lt B.lt   # gate (d)

Gate (b) prints, per the §M9 registration: positions compared, argmax flips (count AND the
position list — a flip changes text, so each one is read, not summarized), max |Δlogit|
overall, and the per-position |Δ| profile's head/tail so growth with position is visible.
Gate (d) is byte equality of two runs of the SAME arm, checked as files — determinism has
no tolerance. `--tokens` prints one token id per line, the argmax column of a dump: arm
S's recorded stream becomes both arms' `--force-tokens` input.

Touches no GPU; numpy only.
"""

import sys

import numpy as np


def load(path):
    raw = open(path, "rb").read()
    if raw[:4] != b"V4LT":
        sys.exit(f"{path}: not a --logit-dump file (bad magic {raw[:4]!r})")
    vocab = int(np.frombuffer(raw, "<u4", 1, 4)[0])
    rec = 4 + 4 * vocab
    body = raw[8:]
    if len(body) % rec:
        sys.exit(
            f"{path}: truncated — {len(body)} body bytes is not a multiple of the "
            f"{rec}-byte record (vocab {vocab}); a partial dump proves nothing"
        )
    n = len(body) // rec
    arg = np.empty(n, dtype=np.uint32)
    logits = np.empty((n, vocab), dtype=np.float32)
    for i in range(n):
        off = i * rec
        arg[i] = np.frombuffer(body, "<u4", 1, off)[0]
        logits[i] = np.frombuffer(body, "<f4", vocab, off + 4)
    return arg, logits


def main():
    args = sys.argv[1:]
    if args and args[0] == "--tokens":
        arg, _ = load(args[1])
        print("\n".join(str(int(t)) for t in arg))
        return
    if args and args[0] == "--identical":
        a, b = open(args[1], "rb").read(), open(args[2], "rb").read()
        if a == b:
            print(f"IDENTICAL: {len(a)} bytes")
        else:
            # A strict prefix has no differing byte inside zip's range — the shorter
            # length IS the first divergence, and a prefix is exactly the truncation
            # shape this message exists for.
            n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            sys.exit(
                f"NOT identical: first differing byte at offset {n} "
                f"(lengths {len(a)} / {len(b)}) — the determinism gate FAILS"
            )
        return
    if len(args) != 2:
        sys.exit(__doc__)
    (arg_a, log_a), (arg_b, log_b) = load(args[0]), load(args[1])
    if log_a.shape != log_b.shape:
        sys.exit(
            f"shape mismatch: {log_a.shape} vs {log_b.shape} — the arms recorded "
            "different position counts and are not positionally comparable"
        )
    n = len(arg_a)
    d = np.abs(log_a - log_b)
    per_pos = d.max(axis=1)
    flips = np.nonzero(arg_a != arg_b)[0]
    print(f"positions compared : {n}")
    print(f"argmax flips       : {len(flips)}" + (f" at {flips.tolist()}" if len(flips) else ""))
    print(f"max |dlogit|       : {per_pos.max():.6g} at position {int(per_pos.argmax())}")
    print(f"mean |dlogit| max  : {per_pos.mean():.6g}")
    print(f"first 8 positions  : {[f'{v:.3g}' for v in per_pos[:8]]}")
    print(f"last 8 positions   : {[f'{v:.3g}' for v in per_pos[-8:]]}")
    if len(flips):
        print(
            "\nFLIPS CHANGE TEXT. Per the §M9 gate (c): decode both arms' texts and READ "
            "them — the repo's rule is that no metric substitutes for the text."
        )


if __name__ == "__main__":
    main()
