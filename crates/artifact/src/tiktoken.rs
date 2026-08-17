//! A tiktoken byte-pair vocabulary: `tiktoken.model` ranks plus a positional special block.
//!
//! Named for the FORMAT, not the model that brought it — Kimi-K3 is the only checkpoint here
//! that ships one today, and the census of which models use it belongs in
//! [`crate::arch`]-adjacent data rather than in this file's name.
//!
//! **Why this exists at all.** [`crate::tokenizer::Tokenizer`] reads HuggingFace
//! `tokenizer.json` through the `tokenizers` crate, and **Kimi-K3 ships no such file**: its
//! `tokenizer_config.json` declares `tokenizer_class: "TikTokenTokenizer"` and the vocabulary
//! is `tiktoken.model`, 163,584 lines of `<base64> <rank>`. Until this landed, a K3 artifact
//! could not be opened by the CLI **at all** — `Tokenizer::load` runs before the architecture
//! match, so `--bench` failed too, and the 1.42 TiB artifact existed with nothing able to read
//! it. `docs/investigations/k3-first-checkpoint.md` §4 is the diagnosis, §7 the scope.
//!
//! **The reference is first-party and the scoring is one-directional.** The construction below
//! is the checkpoint's own `tokenization_kimi.py` (its `pat_str`, its 256-slot positional
//! special block); the goldens in `tests/k3-tokenizer-cases.json` come from that file's
//! constants driven through OpenAI's `tiktoken`, and `tests/k3_tokenizer.rs` scores this code
//! against them and never the reverse.
//!
//! **Not the chat framing.** `encoding_k3.py` renders messages into a *string* (K3's XTML
//! `<|open|>`/`<|sep|>`/`<|close|>` framing) and is a separate, still-refused milestone. This
//! module is text ↔ ids only.

use anyhow::{Context, Result, bail, ensure};
use base64::Engine as _;
use std::collections::{HashMap, HashSet};

/// Ids reserved for special tokens above the base vocabulary — `num_reserved_special_tokens`.
///
/// **The block is positional, and that is the part a config-driven port gets wrong.** Every id
/// in `num_base .. num_base + 256` has a spelling whether `added_tokens_decoder` names it or
/// not; unnamed slots are `<|reserved_token_{id}|>`, so that literal in a prompt encodes to
/// that single id. K3 names 16 of the 256.
pub const SPECIAL_SLOTS: usize = 256;

/// The reference's cap on consecutive same-class (whitespace vs not) characters per encode
/// call, from `tokenization_kimi.py`'s `MAX_NO_WHITESPACES_CHARS`.
///
/// It exists upstream to dodge a panic in tiktoken's Rust core
/// (`openai/tiktoken` issue 195), and it is carried because it **changes ids** once it trips:
/// the reference encodes each chunk separately, so a 30,000-character unbroken run tokenizes
/// differently from the same run encoded whole. Nothing in ordinary prose reaches it; a
/// base64 blob pasted into a prompt does.
pub const MAX_SAME_CLASS_RUN: usize = 25_000;

/// The pre-tokenizer pattern, character for character from `tokenization_kimi.py`'s `pat_str`.
///
/// Pinned by string equality against the driver's copy (`tests/k3-tokenizer-cases.json` carries
/// `pat_str`), because the failure mode of a hand-copied regex is not a compile error — it is a
/// tokenization difference on whichever input happens to cross the broken alternative, which no
/// round-trip and no smoke test would see.
///
/// **Two alternatives are traps and neither fails loudly:**
///
/// * `\s+(?!\S)` is a **negative lookahead**, and the `regex` crate supports no lookaround at
///   all. That is why this module uses `fancy-regex`. The clause makes a whitespace run give
///   its last character to the following piece — measured: `"a    b"` is `["a", "   ", " b"]`
///   = `[64, 274, 291]`, where taking the run whole would yield different ids with no error.
/// * `[…&&[^\p{Han}]]` is **character-class intersection**. `regex-syntax` does support `&&`,
///   which makes it look safe under a change of engine; it is pinned rather than trusted.
///
/// > **MEASURED 2026-08-17, and it corrects this file's first draft.** The intersection clauses
/// > were expected to be catchable by id equality at a script boundary. They are **not**, and
/// > not because the case set is thin: **no token in this vocabulary mixes Han with non-Han**
/// > (checked, 0 of 163,584 — `tests/k3_tokenizer.rs` asserts it), so no byte-pair merge can
/// > cross the boundary and BPE *reconstructs* exactly the split the intersection would have
/// > produced. Removing one clause changes the pre-tokenizer (`"hello你好"` becomes one piece
/// > instead of two) and leaves the ids **identical** — confirmed at the reference level in
/// > python, 0 of 12 Han-boundary texts differing.
/// >
/// > So the string equality above is not belt-and-braces for this trap, it is the ONLY guard,
/// > and the mixed-token assertion is what will tell a future reader when that stops being
/// > true. Same shape as Glimmer's `qk_scale_on_k`: invisible by algebra, not by resolution.
pub const PAT_STR: &str = concat!(
    r"[\p{Han}]+",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+",
    r"[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+(?!\S)",
    "|",
    r"\s+",
);

/// A loaded tiktoken vocabulary: byte-pair ranks, the special block, and the pre-tokenizer.
pub struct Vocab {
    /// Token bytes → rank. Lookups are by `&[u8]` slice, which `Vec<u8>: Borrow<[u8]>` allows
    /// without allocating a key per probe — and [`Self::bpe`] probes once per adjacent pair
    /// per merge round.
    ranks: HashMap<Vec<u8>, u32>,
    /// Rank → token bytes, for decode. A `Vec` rather than a second map because the ranks are
    /// dense `0..num_base` and [`Self::load`] proves it.
    bytes_of: Vec<Vec<u8>>,
    special_id: HashMap<String, u32>,
    special_name: HashMap<u32, String>,
    /// First bytes and lengths of every special spelling, so [`Self::encode`] only attempts a
    /// special match where one could start. Derived from the names, never hardcoded to `<`/`[`:
    /// a checkpoint that spells a special differently would silently stop being scanned for.
    special_heads: HashSet<u8>,
    /// Distinct special lengths, DESCENDING — longest match wins, which is what makes the scan
    /// independent of any prefix relation between two spellings.
    special_lens: Vec<usize>,
    splitter: fancy_regex::Regex,
}

impl Vocab {
    /// Load `<dir>/tiktoken.model` and name the special block from
    /// `<dir>/tokenizer_config.json`'s `added_tokens_decoder`.
    ///
    /// Both files are needed and neither substitutes for the other: the vocabulary stops at
    /// `num_base - 1`, so every special id — including `<|end_of_msg|>`, the actual stop token
    /// — exists only in the config. That is why `convert_k3` copies both.
    pub fn load(dir: &str) -> Result<Self> {
        let vpath = format!("{dir}/tiktoken.model");
        let text = std::fs::read_to_string(&vpath).with_context(|| format!("read {vpath}"))?;
        let bytes_of = parse_ranks(&text).with_context(|| format!("parse {vpath}"))?;
        let ranks = ranks_from(&bytes_of);
        let named = named_specials(dir)?;
        Self::assemble(ranks, bytes_of, named)
    }

    /// Build from parts. Separate from [`Self::load`] so the gate can construct a vocabulary
    /// without a checkpoint on disk, and so `load` stays a file-reading function only.
    fn assemble(
        ranks: HashMap<Vec<u8>, u32>,
        bytes_of: Vec<Vec<u8>>,
        named: HashMap<u32, String>,
    ) -> Result<Self> {
        let num_base = bytes_of.len();
        // **Every single byte must be a token, and this is a real precondition rather than a
        // sanity check.** `bpe` starts from one part per byte and only ever merges a pair whose
        // concatenation is in `ranks`, so every surviving part is either a merged pair (in
        // ranks by construction) or a lone byte. If some byte were absent, a piece containing
        // it would have no rank and encoding would fail on ordinary input. Measured on the
        // shipped file 2026-08-17: 256 of 256 present.
        let singles = (0..=u8::MAX)
            .filter(|b| ranks.contains_key(&vec![*b]))
            .count();
        ensure!(
            singles == 256,
            "the vocabulary has only {singles} of 256 single-byte tokens; byte-pair encoding \
             cannot represent arbitrary input without them"
        );
        let mut special_id = HashMap::with_capacity(SPECIAL_SLOTS);
        let mut special_name = HashMap::with_capacity(SPECIAL_SLOTS);
        for slot in 0..SPECIAL_SLOTS {
            let id = u32::try_from(num_base + slot).context("special id overflows u32")?;
            // The positional rule, and the `unwrap_or_else` IS the rule — not a fallback for a
            // missing entry. See `SPECIAL_SLOTS`.
            let name = named
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<|reserved_token_{id}|>"));
            special_id.insert(name.clone(), id);
            special_name.insert(id, name);
        }
        let named_outside: Vec<&u32> = named
            .keys()
            .filter(|id| (**id as usize) < num_base || (**id as usize) >= num_base + SPECIAL_SLOTS)
            .collect();
        ensure!(
            named_outside.is_empty(),
            "added_tokens_decoder names ids outside the special block \
             {num_base}..{}: {named_outside:?} — the block is positional, so a name outside it \
             would be silently unreachable",
            num_base + SPECIAL_SLOTS
        );
        let special_heads = special_id
            .keys()
            .filter_map(|n| n.as_bytes().first().copied())
            .collect();
        let mut special_lens: Vec<usize> = special_id.keys().map(|n| n.len()).collect();
        special_lens.sort_unstable();
        special_lens.dedup();
        special_lens.reverse();
        Ok(Self {
            ranks,
            bytes_of,
            special_id,
            special_name,
            special_heads,
            special_lens,
            splitter: fancy_regex::Regex::new(PAT_STR).context("compile the K3 pat_str")?,
        })
    }

    /// Base tokens plus the special block — `n_vocab`, which must equal the config's
    /// `vocab_size`. The caller asserts that; this only reports it.
    pub fn n_vocab(&self) -> usize {
        self.bytes_of.len() + SPECIAL_SLOTS
    }

    /// Where the special block starts.
    pub fn num_base(&self) -> usize {
        self.bytes_of.len()
    }

    /// One base token's bytes, by rank. `None` above [`Self::num_base`].
    ///
    /// Exists for the gate that checks WHY `PAT_STR`'s Han intersections are invisible to id
    /// equality (no token mixes Han with non-Han). A read-only accessor rather than exposing
    /// `bytes_of`, so the dense-rank invariant stays this module's to keep.
    pub fn token_bytes(&self, rank: u32) -> Option<&[u8]> {
        self.bytes_of.get(rank as usize).map(Vec::as_slice)
    }

    /// The id of a special spelling, e.g. `<|end_of_msg|>`.
    pub fn special(&self, name: &str) -> Option<u32> {
        self.special_id.get(name).copied()
    }

    /// Encode, with special spellings recognised — the reference's `encode` default
    /// (`allow_special_tokens=True` → `allowed_special="all"`).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_capped(text, MAX_SAME_CLASS_RUN)
    }

    /// [`Self::encode`] with the same-class run cap as a parameter.
    ///
    /// A parameter rather than a constant read from inside, for the reason
    /// `format::layer::write_expert_layer`'s `window` is one: it lets the gate reach the
    /// chunking BOUNDARY without a 25,000-character fixture. Nothing but a test should pass
    /// anything other than [`MAX_SAME_CLASS_RUN`].
    pub fn encode_capped(&self, text: &str, max_run: usize) -> Result<Vec<u32>> {
        ensure!(max_run > 0, "max_run must be positive");
        let mut out = Vec::new();
        for chunk in split_same_class_runs(text, max_run) {
            self.encode_chunk(chunk, &mut out)?;
        }
        Ok(out)
    }

    /// One chunk: cut it at special spellings, byte-pair the text between them.
    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) -> Result<()> {
        let mut rest = chunk;
        while !rest.is_empty() {
            match self.find_special(rest) {
                Some((at, len, id)) => {
                    self.encode_text(&rest[..at], out)?;
                    out.push(id);
                    rest = &rest[at + len..];
                }
                None => {
                    self.encode_text(rest, out)?;
                    break;
                }
            }
        }
        Ok(())
    }

    /// The first special spelling in `s` as `(byte offset, byte length, id)`.
    ///
    /// Scans only at positions whose byte is some special's first byte, and tries lengths
    /// longest-first so one spelling being a prefix of another cannot matter.
    fn find_special(&self, s: &str) -> Option<(usize, usize, u32)> {
        for (at, b) in s.as_bytes().iter().enumerate() {
            if !self.special_heads.contains(b) {
                continue;
            }
            for &len in &self.special_lens {
                let Some(cand) = s.get(at..at + len) else {
                    continue; // past the end, or not a char boundary
                };
                if let Some(&id) = self.special_id.get(cand) {
                    return Some((at, len, id));
                }
            }
        }
        None
    }

    /// Pre-tokenize with [`PAT_STR`] and byte-pair each piece.
    fn encode_text(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for m in self.splitter.find_iter(text) {
            let m = m.context("pat_str match failed")?;
            self.bpe(m.as_str().as_bytes(), out)?;
        }
        Ok(())
    }

    /// Byte-pair encode one pre-tokenized piece, appending its ids.
    ///
    /// Merges the lowest-rank adjacent pair until none is in the vocabulary — the standard
    /// tiktoken merge, written straightforwardly rather than with the reference's
    /// index-threading optimisation. Pieces are single words (the pre-tokenizer guarantees it),
    /// so the quadratic scan is over a handful of bytes; the vocabulary lookup dominates.
    ///
    /// **The equivalence to the reference is MEASURED, not reasoned.** "Repeatedly merge the
    /// globally lowest-rank adjacent pair" is the same function as tiktoken's threaded loop only
    /// if merge order cannot change the outcome, which is an argument about ties that is easy to
    /// get wrong on paper. So it was fuzzed differentially against the first-party encoder
    /// (2026-08-17): **13,400 texts, 473,049 ids, zero mismatches.** Two corpora — 5,000 random
    /// strings over twelve alphabets (Latin both cases, digits, whitespace, punctuation, Han,
    /// accented Latin, Cyrillic, Arabic, emoji with ZWJ, special-token spellings, control bytes)
    /// plus 1,000 targeted whitespace/script-boundary storms; then 8,400 more weighted toward
    /// DEEP merges — single pieces of 8..200 characters with no whitespace, so the loop runs to
    /// its full depth — plus long prose and 600 lossily-decoded random byte strings. `decode`
    /// round-tripped every row.
    ///
    /// That corpus is not vendored: the 95 curated cases in `tests/k3-tokenizer-cases.json` are
    /// the standing gate because each targets a named `pat_str` clause, and a megabyte of random
    /// strings would gate the same algorithm with far less to say about which clause broke. The
    /// numbers above are the one-time evidence that the merge itself is right; regenerate with
    /// the driver's constants against the shipped vocabulary if `bpe` is ever rewritten.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        // The whole piece is usually a token already — the common case for any word the
        // vocabulary carries, and it skips the merge loop entirely.
        if let Some(&rank) = self.ranks.get(piece) {
            out.push(rank);
            return Ok(());
        }
        let mut cuts: Vec<usize> = (0..=piece.len()).collect();
        while let Some(at) = self.lowest_pair(piece, &cuts) {
            cuts.remove(at + 1);
        }
        for w in cuts.windows(2) {
            let part = &piece[w[0]..w[1]];
            let &rank = self.ranks.get(part).with_context(|| {
                format!(
                    "no rank for {part:?} — every part is a merged pair or a lone byte, and all \
                     256 bytes are checked present at load, so this is unreachable"
                )
            })?;
            out.push(rank);
        }
        Ok(())
    }

    /// Index in `cuts` of the adjacent pair with the lowest rank, if any pair is a token.
    ///
    /// Split out from [`Self::bpe`] so the merge loop reads as one statement; `bpe` was a
    /// single function with three nested blocks and the cohesion gate prefers this shape.
    fn lowest_pair(&self, piece: &[u8], cuts: &[usize]) -> Option<usize> {
        let mut best: Option<(u32, usize)> = None;
        for at in 0..cuts.len().saturating_sub(2) {
            let pair = &piece[cuts[at]..cuts[at + 2]];
            let Some(&rank) = self.ranks.get(pair) else {
                continue;
            };
            if best.is_none_or(|(seen, _)| rank < seen) {
                best = Some((rank, at));
            }
        }
        best.map(|(_, at)| at)
    }

    /// Decode a whole id sequence, special spellings included.
    ///
    /// **Lossy UTF-8, matching the reference**, and that is required rather than tolerated:
    /// byte-level BPE splits one codepoint across several tokens, so any prefix of a sequence
    /// can end mid-character. Decoding the whole sequence at once is what keeps multi-token
    /// characters intact — the same argument `Tokenizer::decode_all` carries.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            match self.bytes_of.get(id as usize) {
                Some(b) => bytes.extend_from_slice(b),
                None => match self.special_name.get(&id) {
                    Some(name) => bytes.extend_from_slice(name.as_bytes()),
                    None => bail!(
                        "id {id} is outside this vocabulary (base {}, n_vocab {})",
                        self.num_base(),
                        self.n_vocab()
                    ),
                },
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}

/// Parse `<base64> <rank>` lines into the dense `rank -> bytes` table.
///
/// Returns only that table, not the pair: [`ranks_from`] derives the map from it, so handing
/// back both would be two representations of one fact for a caller to keep in step. (clippy's
/// `type_complexity` flagged the tuple, which was the right nudge for the wrong reason.)
fn parse_ranks(text: &str) -> Result<Vec<Vec<u8>>> {
    let mut pairs: Vec<(u32, Vec<u8>)> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (b64, rank) = line
            .split_once(' ')
            .with_context(|| format!("line {}: expected `<base64> <rank>`, got {line:?}", n + 1))?;
        let token = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .with_context(|| format!("line {}: {b64:?} is not base64", n + 1))?;
        let rank: u32 = rank
            .trim()
            .parse()
            .with_context(|| format!("line {}: {rank:?} is not a rank", n + 1))?;
        pairs.push((rank, token));
    }
    ensure!(!pairs.is_empty(), "the vocabulary file has no tokens");
    pairs.sort_unstable();
    // **Dense and gap-free, asserted.** `bytes_of` is indexed by rank, so a gap would make
    // decode return the wrong token for every id above it — and a duplicate rank would make
    // the vocabulary smaller than its own line count while still looking well-formed.
    let mut bytes_of = Vec::with_capacity(pairs.len());
    for (want, (rank, token)) in pairs.into_iter().enumerate() {
        let want = u32::try_from(want).context("rank overflows u32")?;
        ensure!(
            rank == want,
            "ranks are not dense: expected {want}, found {rank} — decode indexes by rank"
        );
        bytes_of.push(token);
    }
    Ok(bytes_of)
}

/// The rank map as the inverse of a dense `rank -> bytes` table.
///
/// One definition, because `parse_ranks` and the unit fixtures both build it and jscpd reported
/// the pair (2026-08-17). It is also the only place the two directions are tied together, so a
/// change to either cannot leave them disagreeing.
fn ranks_from(bytes_of: &[Vec<u8>]) -> HashMap<Vec<u8>, u32> {
    bytes_of
        .iter()
        .enumerate()
        .map(|(r, b)| (b.clone(), r as u32))
        .collect()
}

/// `added_tokens_decoder` as id → spelling.
fn named_specials(dir: &str) -> Result<HashMap<u32, String>> {
    let path = format!("{dir}/tokenizer_config.json");
    let raw = std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?;
    let doc: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("{path} is not JSON"))?;
    // Absent is not an error: the block is positional, so a config naming nothing yields 256
    // reserved spellings and a vocabulary that still encodes text correctly. It would have no
    // stop token, which is `Tokenizer::load`'s business and not this function's.
    let Some(map) = doc.get("added_tokens_decoder").and_then(|v| v.as_object()) else {
        return Ok(HashMap::new());
    };
    let mut out = HashMap::with_capacity(map.len());
    for (id, entry) in map {
        let id: u32 = id
            .parse()
            .with_context(|| format!("{path}: added_tokens_decoder key {id:?} is not an id"))?;
        let content = entry
            .get("content")
            .and_then(|v| v.as_str())
            .with_context(|| format!("{path}: added_tokens_decoder[{id}] has no `content`"))?;
        out.insert(id, content.to_string());
    }
    Ok(out)
}

/// Split so no chunk holds more than `max_run` consecutive whitespace or non-whitespace
/// characters — `tokenization_kimi.py`'s `_split_whitespaces_or_nonwhitespaces`.
///
/// **Declared deviation:** the class test is Rust's `char::is_whitespace` (Unicode
/// `White_Space`) where the reference uses Python's `str.isspace()`, which additionally treats
/// the file/group/record/unit separators `\x1c..\x1f` as space. The two disagree only on where
/// a chunk boundary falls, and only for a run already longer than `max_run`, so no realistic
/// input reaches the difference; it is written down rather than papered over.
fn split_same_class_runs(s: &str, max_run: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut run, mut start) = (0usize, 0usize);
    let mut in_space = s.chars().next().is_some_and(char::is_whitespace);
    for (at, c) in s.char_indices() {
        let now_space = c.is_whitespace();
        if in_space != now_space {
            run = 1;
            in_space = now_space;
        } else {
            run += 1;
            if run > max_run {
                out.push(&s[start..at]);
                start = at;
                run = 1;
            }
        }
    }
    out.push(&s[start..]);
    out
}

#[cfg(test)]
mod tests {
    // Crate-wide `unwrap`/`expect` are deny; a firing one in a unit test IS the report, and
    // these cases carry no checkpoint to be missing.
    #![allow(clippy::unwrap_used)]
    use super::*;

    /// The pattern must COMPILE under the engine that has to run it. A separate test from the
    /// id gate because a pattern that fails to compile makes every other case fail for the
    /// same uninformative reason, and because this is the assertion that `fancy-regex` — not
    /// `regex` — is the right dependency.
    #[test]
    fn the_pat_str_compiles_under_fancy_regex() {
        let re = fancy_regex::Regex::new(PAT_STR).unwrap();
        // The lookahead alternative actually takes effect: with `\s+(?!\S)` the run before a
        // word gives up its last space. This is the behaviour `regex` cannot express at all.
        let split = |s: &str| -> Vec<String> {
            re.find_iter(s)
                .map(|m| m.unwrap().as_str().to_string())
                .collect()
        };
        assert_eq!(
            split("a    b"),
            ["a", "   ", " b"],
            "the lookahead is not in effect"
        );
        // And the class intersection keeps Han out of the Latin run.
        assert_eq!(
            split("hello你好"),
            ["hello", "你好"],
            "Han was absorbed into a Latin run"
        );
    }

    #[test]
    fn same_class_runs_split_at_the_cap() {
        // Under the cap: one chunk, unchanged.
        assert_eq!(split_same_class_runs("aaa", 5), vec!["aaa"]);
        // Over it: the run breaks, and only same-class characters count toward it.
        assert_eq!(split_same_class_runs("aaaa", 2), vec!["aa", "aa"]);
        assert_eq!(split_same_class_runs("aa  aa", 2), vec!["aa  aa"]);
        // Empty input must not index `s[0]` — the reference guards this explicitly.
        assert_eq!(split_same_class_runs("", 4), vec![""]);
    }

    #[test]
    fn a_vocabulary_missing_a_single_byte_is_refused() {
        // Two-byte tokens only: BPE could not represent an arbitrary byte, so `assemble`
        // refuses rather than failing later on some particular input.
        let bytes_of: Vec<Vec<u8>> = (0..4u8).map(|b| vec![b, b]).collect();
        let ranks = ranks_from(&bytes_of);
        // Matched rather than `unwrap_err`, which would need `Debug` on `Vocab` — and a
        // derived `Debug` on a struct holding the 163,584-entry rank map is a 19 MB print
        // waiting for whoever first adds `{:?}` to a diagnostic.
        let err = match Vocab::assemble(ranks, bytes_of, HashMap::new()) {
            Ok(_) => panic!("a vocabulary with no single-byte tokens was accepted"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("of 256 single-byte tokens"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn non_dense_ranks_are_refused() {
        // `AQ==` is byte 0x01. Rank 0 then rank 2: decode indexes by rank, so the gap would
        // shift every token above it.
        let err = parse_ranks("AQ== 0\nAg== 2\n").unwrap_err();
        assert!(err.to_string().contains("not dense"), "unexpected: {err}");
    }
}
