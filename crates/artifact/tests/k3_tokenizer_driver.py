"""Generate the id-pinned goldens for Kimi-K3's tiktoken vocabulary.

**The reference is the FIRST-PARTY stack, never a transliteration** — the checkpoint's own
`tokenization_kimi.py` construction (its `pat_str`, its positional special-token block) driven
through OpenAI's `tiktoken` over the shipped `tiktoken.model`. That is the tokenizer the model
was trained with; `crates/artifact/src/tiktoken.rs` is scored against what this writes and never
the other way round.

**`encoding_k3.py` is deliberately NOT involved.** It renders chat messages into a *string*
(K3's XTML framing) and is a later milestone that `--port` still refuses; this file is about
string -> ids. A loader gated against the framing file would be gated against the wrong
reference — see `docs/investigations/k3-first-checkpoint.md` section 7.

Usage (no GPU, no lock, seconds — its venv holds `tiktoken` and NOTHING else, so it cannot
disturb the anchor venvs whose pinned versions are other models' provenance):

    K3_SRC=/swarm/storage/ai/rivoli/kimi-k3-src \\
    /home/rhansen/k3-tokenizer/venv/bin/python \\
        crates/artifact/tests/k3_tokenizer_driver.py

Writes `crates/artifact/tests/k3-tokenizer-cases.json` beside itself and prints a census.
Needs only `tiktoken.model` + `tokenizer_config.json` from the checkpoint (2.8 MB of a
1.42 TiB repo), so regenerating this touches none of the weights.

Configured by env var for the reason `tests/k3-anchor.sh` states at length: CLAUDE.md's
"instruments go behind a feature AND a flag, never an env var" is about the engine binary
(invisible to `--help`, absent from a recorded command line, silently active in a stock
build). None of that applies to a driver script that is not a cargo run and whose invocation
is recorded on the line above.
"""

import hashlib


import json
import os
import pathlib
import sys

import tiktoken

def fnv1a(data: bytes) -> int:
    """FNV-1a/64, matching `rivoli_core::hash::fnv1a` exactly.

    Emitted so the Rust gate can RECOMPUTE this from the `tiktoken.model` it is about to score
    against, rather than trusting a number in a file beside it. `sha256` is kept too — it is
    what a human compares against the upstream repo — but nothing in the tree can recompute it
    without a `sha2` dependency, which is the wrong trade for pinning a fixture.
    """
    h = 0xCBF29CE484222325
    for b in data:
        h = ((h ^ b) * 0x00000100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h

# The checkpoint's own import (`from tiktoken.load import load_tiktoken_bpe`). Spelled as an
# explicit submodule import because `import tiktoken` alone does not bind `tiktoken.load` in
# 0.13.0 -- an `AttributeError`, which is the loud direction.
from tiktoken.load import load_tiktoken_bpe

# --- the first-party construction, copied from the checkpoint's tokenization_kimi.py ---------
#
# Copied VERBATIM rather than imported, and that is the point: importing would need
# `transformers` (a 5.x dependency this driver has no other use for) and would let a future
# transformers refactor silently change what "the reference" means. The two constants below are
# the whole of what `TikTokenTokenizer` contributes, and the Rust side pins both -- PAT_STR by
# string equality against what this file emits, so a transcription typo in either copy is a
# red test rather than a tokenization difference nobody can see.
#
# Source: /swarm/storage/ai/rivoli/kimi-k3-src/tokenization_kimi.py, revision
# 9f62e4e9fffbd0a83ddd60e1c209d828994b3569, read 2026-08-17.
NUM_RESERVED_SPECIAL_TOKENS = 256
# `_encode_text_piece`'s two caps. The INNER one (whitespace-class runs) changes ids and is
# ported; the OUTER one is declared-not-ported on the Rust side — see `tiktoken.rs::encode`.
MAX_NO_WHITESPACES_CHARS = 25_000
TIKTOKEN_MAX_ENCODE_CHARS = 400_000
PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


def split_whitespaces_or_nonwhitespaces(s: str, max_consecutive_slice_len: int):
    """`TikTokenTokenizer._split_whitespaces_or_nonwhitespaces`, copied verbatim.

    Copied for `PAT_STR`'s reason — importing would pull `transformers` in. The body below is
    diffable against the checkpoint's line for line.
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
        is_now_space = s[i].isspace()

        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


def build_encoding(src: pathlib.Path):
    """`TikTokenTokenizer.__init__`'s model, with its special block named from the config."""
    vocab_file = src / "tiktoken.model"
    mergeable = load_tiktoken_bpe(str(vocab_file))
    num_base = len(mergeable)
    named = {
        int(i): e["content"]
        for i, e in json.loads((src / "tokenizer_config.json").read_text())[
            "added_tokens_decoder"
        ].items()
    }
    # **Positional, not listed.** Every id in the 256-slot block gets a name whether the
    # config names it or not; the reserved spelling is part of the vocabulary, so a text
    # containing `<|reserved_token_163592|>` encodes to that single id.
    special = {
        named.get(i, f"<|reserved_token_{i}|>"): i
        for i in range(num_base, num_base + NUM_RESERVED_SPECIAL_TOKENS)
    }
    enc = tiktoken.Encoding(
        name=vocab_file.name,
        pat_str=PAT_STR,
        mergeable_ranks=mergeable,
        special_tokens=special,
    )
    return enc, mergeable, named, special


def cases(named: dict) -> list:
    """Texts chosen to break the two `pat_str` traps, plus the special-token block.

    Every group names the clause it targets. A case list that merely looks varied is the
    failure mode here: the traps are single alternatives inside one long pattern, and prose
    exercises maybe three of the eight.
    """
    out = []

    def add(group, name, text):
        out.append({"group": group, "name": name, "text": text})

    # TRAP 1 -- `\s+(?!\S)`, the negative lookahead. A whitespace run FOLLOWED BY
    # non-whitespace is the discriminator: the lookahead makes the run give its last character
    # to the following piece, so `a    b` splits `a` / `   ` / ` b`. Without lookaround the run
    # is taken whole and `b` stands alone -- different ids, no error. Runs of length 1 and 2 are
    # included because the boundary is where an off-by-one lives.
    add("lookahead", "one_space_between", "a b")
    add("lookahead", "two_spaces_between", "a  b")
    add("lookahead", "four_spaces_between", "a    b")
    add("lookahead", "many_spaces_between", "a" + " " * 9 + "b")
    add("lookahead", "trailing_spaces", "hello   ")
    add("lookahead", "trailing_single_space", "hello ")
    add("lookahead", "leading_spaces", "   hello")
    add("lookahead", "only_spaces", "     ")
    add("lookahead", "tabs_between", "a\t\t\tb")
    add("lookahead", "mixed_ws_between", "a \t \t b")
    add("lookahead", "ws_before_punct", "a    !")
    add("lookahead", "ws_before_digit", "a    7")
    add("lookahead", "ws_before_han", "a    你好")

    # `\s*[\r\n]+` sits directly before the lookahead clause, so newline handling is where a
    # wrong alternative ORDER shows up.
    add("newlines", "lf", "line1\nline2")
    add("newlines", "crlf", "line1\r\nline2")
    add("newlines", "blank_line_crlf", "line1\r\n\r\nline3")
    add("newlines", "trailing_lf", "text\n")
    add("newlines", "trailing_crlf", "text\r\n")
    add("newlines", "spaces_then_lf", "text   \n   more")
    add("newlines", "many_lf", "a\n\n\n\nb")

    # TRAP 2 -- `&&[^\p{Han}]]` class intersection. These clauses exist so that Han is taken by
    # alternative 1 and never absorbed into a Latin run, which only shows at the BOUNDARY
    # between the two scripts.
    add("han", "han_only", "你好世界")
    add("han", "han_then_latin", "你好hello")
    add("han", "latin_then_han", "hello你好")
    add("han", "latin_han_latin", "hello你好world")
    add("han", "han_latin_alternating", "你a好b世c")
    add("han", "han_then_digits", "你好123")
    add("han", "han_with_punct", "你好。世界！")
    add("han", "han_spaced", "你好 世界")
    add("han", "japanese_kana_not_han", "こんにちは")
    add("han", "kana_and_han", "日本語のテキスト")

    # `\p{N}{1,3}` -- capped at three, so runs longer than three SPLIT. A port that drops the
    # bound tokenizes long numbers as one piece.
    add("digits", "one", "7")
    add("digits", "three", "123")
    add("digits", "four", "1234")
    add("digits", "ten", "1234567890")
    add("digits", "grouped", "1,234,567")
    add("digits", "year_in_prose", "in 2026 the model shipped")
    add("digits", "decimal", "3.14159")

    # The `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` tail, in BOTH cases -- `(?i:` is the only
    # case-insensitive island in the pattern, and dropping the flag changes only uppercase text.
    for w in ["it's", "IT'S", "don't", "DON'T", "we're", "WE'RE",
              "you've", "YOU'VE", "I'm", "I'M", "I'll", "I'LL", "he'd", "HE'D"]:
        add("contractions", f"contraction_{w.replace(chr(39), '_')}", w)
    add("contractions", "curly_apostrophe_is_not_the_clause", "it’s")

    # Alternatives 2 and 3 differ only in whether the uppercase run is `*` or `+`, so mixed
    # case is what tells them apart.
    add("case", "lower", "hello")
    add("case", "upper", "HELLO")
    add("case", "title", "Hello")
    add("case", "camel", "helloWorld")
    add("case", "internal_caps", "McDonald")
    add("case", "upper_then_lower", "ABCdef")
    add("case", "shouty_sentence", "THIS IS FINE")

    # The SPECIAL BLOCK. Every named id, individually -- a loop over the 16 rather than a
    # sample, because the failure this catches is one entry misplaced, and a sample is exactly
    # what misses it. Then a RESERVED id, which is the half a config-driven port gets wrong:
    # the names are positional, so an unnamed slot still has a spelling.
    for i, content in sorted(named.items()):
        add("special_named", f"special_{i}", content)
    # Genuinely unnamed slots: 163592 sits between two named ones, 163837 is the HIGHEST
    # reserved id (163838/163839 are [UNK]/[PAD]).
    add("special_reserved", "reserved_163592", "<|reserved_token_163592|>")
    add("special_reserved", "reserved_163700", "<|reserved_token_163700|>")
    add("special_reserved", "reserved_highest_163837", "<|reserved_token_163837|>")
    # And the NEGATIVE row that makes the three above mean something: 163839 IS named ([PAD]),
    # so `<|reserved_token_163839|>` is NOT a special spelling and must fall through to ordinary
    # byte-pair encoding. A port that generates reserved names for every slot without checking
    # the config would collapse this to one id. The driver's own first draft asserted the wrong
    # thing here and the Rust gate caught it.
    add("special_reserved", "named_slot_is_not_reserved_163839",
        "<|reserved_token_163839|>")

    # Specials in context -- K3's actual XTML framing, which is what a chat port will emit.
    add("special_context", "xtml_turn",
        '<|open|>message role="user"<|sep|>hello<|close|>message<|sep|><|end_of_msg|>')
    add("special_context", "special_touching_text", "before<|sep|>after")
    add("special_context", "specials_adjacent", "<|open|><|close|>")
    add("special_context", "special_like_but_not", "<|not_a_real_token|>")
    add("special_context", "bare_angle_pipe", "<| |>")

    # Bytes and boundaries. Multi-byte codepoints split across BPE tokens are what
    # `decode_all` exists for, and an empty string is where an unchecked `s[0]` panics.
    add("edge", "empty", "")
    add("edge", "single_space", " ")
    add("edge", "single_newline", "\n")
    add("edge", "nul_and_controls", "a\x00b\x01c")
    add("edge", "emoji", "\U0001f680\U0001f525")
    add("edge", "emoji_zwj_family", "\U0001f468‍\U0001f469‍\U0001f467")
    add("edge", "accented_latin", "naïve café über")
    add("edge", "cyrillic", "Привет мир")
    add("edge", "arabic_rtl", "مرحبا بالعالم")
    add("edge", "combining_marks", "éà")
    add("edge", "long_prose",
        "The routed experts do not fit in memory, so they stream from NVMe while the "
        "resident ones compute -- that overlap is the whole design.")

    return out


def main() -> int:
    src = pathlib.Path(os.environ.get("K3_SRC", "/swarm/storage/ai/rivoli/kimi-k3-src"))
    if not (src / "tiktoken.model").is_file():
        print(f"K3_SRC={src} has no tiktoken.model", file=sys.stderr)
        return 1

    enc, mergeable, named, special = build_encoding(src)
    vocab_bytes = (src / "tiktoken.model").read_bytes()

    rows = cases(named)
    for c in rows:
        # `allowed_special="all"` is `TikTokenTokenizer.encode`'s default path
        # (`allow_special_tokens=True` -> `_encode_text_piece`), so it is the behaviour the
        # Rust `encode` is scored against.
        c["ids"] = enc.encode(c["text"], allowed_special="all")

    # The same-class run chunking (`MAX_NO_WHITESPACES_CHARS`), gated at a SMALL cap.
    #
    # The real cap is 25,000 characters, and a fixture that reached it would be a 25 KB string
    # in a vendored JSON to test one boundary. So the cap is a PARAMETER on the Rust side and
    # these rows drive it low — the same trick `write_expert_layer`'s `window` uses to reach its
    # boundary without allocating a gigabyte. What is being gated is that chunking happens at
    # all and at the right place: encoding a long run whole gives DIFFERENT ids from encoding it
    # in chunks, which is exactly why the reference does this and why dropping it is silent.
    chunked = []
    for name, text, cap in [
        ("run_at_cap", "abcdefgh", 8),
        ("run_over_cap", "abcdefghij", 4),
        ("long_nonspace_run", "x" * 40, 7),
        ("digits_over_cap", "1234567890" * 3, 8),
        ("space_run_over_cap", " " * 20, 6),
        ("alternating_classes", ("ab   " * 8), 5),
        ("han_run_over_cap", "你好世界" * 5, 6),
    ]:
        ids = []
        for piece in split_whitespaces_or_nonwhitespaces(text, cap):
            ids.extend(enc.encode(piece, allowed_special="all"))
        chunked.append({"name": name, "text": text, "max_run": cap, "ids": ids})

    doc = {
        "_comment": (
            "GENERATED by crates/artifact/tests/k3_tokenizer_driver.py from the FIRST-PARTY "
            "stack. Do not hand-edit: crates/artifact/tests/k3_tokenizer.rs scores the Rust "
            "loader against these ids, so an edit here moves the reference rather than "
            "fixing anything."
        ),
        "provenance": {
            "python": sys.version.split()[0],
            "tiktoken": tiktoken.__version__,
            "checkpoint_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
            "tiktoken_model_sha256": hashlib.sha256(vocab_bytes).hexdigest(),
            "tiktoken_model_fnv1a": fnv1a(vocab_bytes),
            "tiktoken_model_bytes": len(vocab_bytes),
        },
        # Pinned so the Rust side asserts its OWN pattern equals the reference's. This is the
        # check that catches a transcription typo directly, rather than as a puzzling id
        # difference in whichever case happens to cross the broken alternative.
        "pat_str": PAT_STR,
        "num_reserved_special_tokens": NUM_RESERVED_SPECIAL_TOKENS,
        # Pinned for `pat_str`'s reason: it changes ids once it trips, so a typo in the Rust
        # copy must be a red test rather than a tokenization difference nobody can see.
        "max_no_whitespaces_chars": MAX_NO_WHITESPACES_CHARS,
        "tiktoken_max_encode_chars": TIKTOKEN_MAX_ENCODE_CHARS,
        "num_base_tokens": len(mergeable),
        "n_vocab": enc.n_vocab,
        # The full named map, so the Rust side's positional construction is checked against
        # the reference's rather than against the config it also read.
        "named_specials": {str(i): c for i, c in sorted(named.items())},
        # **Read out of the reference's own special map, never spelled by this file.**
        # An earlier draft wrote `f"<|reserved_token_{i}|>"` for a hand-picked id list and put
        # 163839 in it — which is NAMED `[PAD]`, so the golden asserted a spelling the reference
        # does not have. The Rust gate caught it on its first run (2026-08-17). That is the same
        # defect this whole milestone came from, reproduced inside the fixture meant to gate it:
        # a driver that invents its expectations tests the driver. Inverting `special` and
        # subtracting `named` makes the golden state what the reference states.
        "reserved_examples": {
            str(i): n for n, i in sorted(special.items(), key=lambda kv: kv[1])
            if i not in named
        },
        "cases": rows,
        "chunked": chunked,
    }
    out = pathlib.Path(__file__).with_name("k3-tokenizer-cases.json")
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")

    groups: dict = {}
    for c in rows:
        groups[c["group"]] = groups.get(c["group"], 0) + 1
    print(f"wrote {out} ({out.stat().st_size} B)")
    print(f"  tiktoken {tiktoken.__version__}  python {sys.version.split()[0]}")
    print(f"  num_base_tokens {len(mergeable)}  n_vocab {enc.n_vocab}")
    print(f"  named specials {len(named)}  cases {len(rows)}")
    for g, n in sorted(groups.items()):
        print(f"    {g:18s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
