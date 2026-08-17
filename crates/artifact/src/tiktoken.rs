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

/// The reference's OUTER cap, `TIKTOKEN_MAX_ENCODE_CHARS` — characters per encode call.
///
/// > **PORTED 2026-08-17, after first being declared unreachable and left out. That was wrong,
/// > and review produced the counterexample.** The claim was that `ATTEND_MAX_KV` 8192 tokens is
/// > ~32 KB of text, so 400,000 characters cannot be reached — which divides characters by
/// > TOKENS. The longest token in this vocabulary is **256 bytes**, so 8192 tokens can carry
/// > ~2.1 M characters and the cap sits *below* the reachable budget rather than far above it.
/// > Measured: `" " + "*".repeat(420_000)` is **1,664 ids** — comfortably inside `--ctx 8192` —
/// > and the reference returns **1,666** for it.
/// >
/// > The mechanism is that the reference restarts the inner run counter at each outer boundary
/// > and the unported version did not, so the two diverge whenever character 400,000 falls
/// > mid-run and off the inner grid. `"*".repeat(400_001)` is NOT a counterexample
/// > (400,000 = 16 x 25,000 lands on the grid), which is how spot-checking missed it.
pub const MAX_ENCODE_CHARS: usize = 400_000;

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
/// > **MEASURED 2026-08-17.** The intersections are invisible to *id* equality — not for want of
/// > cases, but because **no token in this vocabulary mixes Han with non-Han**, so no merge can
/// > cross the boundary and BPE reconstructs the split the clauses would have made. Removing one
/// > changes the pre-tokenizer (`"hello你好"` becomes one piece) and leaves ids identical,
/// > confirmed in python. Two gates still catch it: the string equality above, and
/// > `the_pat_str_compiles_under_fancy_regex` below, which observes the *split* and is the
/// > stronger of the pair. `docs/investigations/k3-first-checkpoint.md` §7 carries the record.
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
        // **Distinct, and last-write-wins would hide it.** `special_id` is a name→id map, so a
        // config whose `added_tokens_decoder` content collides with another entry — or with a
        // `<|reserved_token_N|>` spelling — silently yields fewer than 256 entries: one id becomes
        // unencodable while `special_name` still decodes to a spelling that re-encodes to a
        // DIFFERENT id. 256/256 distinct on the shipped config (2026-08-17); asserted because the
        // failure is a silent id substitution, not an error.
        ensure!(
            special_id.len() == SPECIAL_SLOTS,
            "the special block has {} distinct spellings for {SPECIAL_SLOTS} ids — two entries \
             collide, so one id cannot be encoded and decode/encode are no longer inverse",
            special_id.len()
        );
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

    /// The first base token mixing Han with non-Han, as `(rank, text)`.
    ///
    /// **A tripwire on a PREMISE, not a correctness check.** The `PAT_STR` Han intersections are
    /// invisible to id equality precisely because no such token exists (see `PAT_STR`), so BPE
    /// reconstructs the split the clauses would have made. If a future vocabulary ships one, the
    /// reasoning changes — id equality starts covering the intersections — and the gate that
    /// calls this reddens to say so rather than letting a stale argument outlive its evidence.
    ///
    /// A method rather than a loop inside the test, so the same definition serves the real
    /// vocabulary and the synthetic one a unit test can build (which is what proves the tripwire
    /// fires at all, and lands that proof in CI where no checkpoint exists).
    pub fn mixed_script_token(&self) -> Option<(u32, String)> {
        // `[^\p{Han}]` in the pattern's OWN terms, not a hand-rolled codepoint range.
        let han = fancy_regex::Regex::new(r"\p{Han}").ok()?;
        let other = fancy_regex::Regex::new(r"[^\p{Han}]").ok()?;
        (0..self.bytes_of.len() as u32).find_map(|rank| {
            // A partial UTF-8 sequence cannot mix scripts as text; byte-level BPE guarantees
            // nothing else, so those are skipped rather than counted.
            let s = std::str::from_utf8(self.token_bytes(rank)?).ok()?;
            let mixed = han.is_match(s).unwrap_or(false) && other.is_match(s).unwrap_or(false);
            mixed.then(|| (rank, s.to_string()))
        })
    }

    /// A special spelling that is a strict prefix of another, as `(shorter, longer)`.
    ///
    /// **[`Self::find_special`] takes the LONGEST match at the earliest position; tiktoken matches
    /// by regex alternation, which is leftmost-FIRST-alternative.** Those two rules agree only
    /// while no spelling is a prefix of another — verified over all 256 of K3's (zero prefix
    /// pairs, 2026-08-17), and asserted rather than assumed because a checkpoint that added, say,
    /// `<|sep|>` beside `<|sep|>2` would make the two encoders disagree with nothing to notice.
    pub fn prefixed_special(&self) -> Option<(String, String)> {
        let mut names: Vec<&String> = self.special_id.keys().collect();
        names.sort();
        names.iter().enumerate().find_map(|(i, a)| {
            names[i + 1..]
                .iter()
                .find(|b| b.starts_with(a.as_str()) && b.len() > a.len())
                .map(|b| ((*a).clone(), (*b).clone()))
        })
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
    ///
    /// **Two declared deviations from `_encode_text_piece`, both stated because the driver's
    /// golden pins the constants and something has to say what happens to them here:**
    ///
    /// * `TIKTOKEN_MAX_ENCODE_CHARS = 400_000`, the reference's OUTER chunk, is **not ported**.
    ///   It exists to dodge a `pyo3_runtime.PanicException` in tiktoken's Rust core, which this
    ///   implementation is not, and 400,000 characters is ~50x the whole `ATTEND_MAX_KV` 8192
    ///   context this engine can decode — so no reachable prompt crosses it. If a caller ever
    ///   encodes text that long for some non-decode purpose, the ids would diverge from the
    ///   reference at that boundary and this is the note that says so.
    /// * `allow_special_tokens=False` is **not ported** either. The reference offers it for
    ///   *user* text, where `<|end_of_msg|>` in a prompt must become ordinary bytes rather than
    ///   a control token; this arm only ever encodes raw bench prompts, where the distinction
    ///   cannot arise because nothing frames turns. **A chat port must add it before the first
    ///   user-supplied string reaches here** — otherwise a prompt containing that spelling
    ///   injects the model's stop token and the decode ends early on the user's own text.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_capped(text, MAX_ENCODE_CHARS, MAX_SAME_CLASS_RUN)
    }

    /// [`Self::encode`] with both caps as parameters.
    ///
    /// Parameters rather than constants read from inside, for the reason
    /// `format::layer::write_expert_layer`'s `window` is one: it lets a gate reach either
    /// BOUNDARY without a 400,000-character fixture. Nothing but a test should pass anything
    /// other than [`MAX_ENCODE_CHARS`] and [`MAX_SAME_CLASS_RUN`].
    pub fn encode_capped(&self, text: &str, max_chars: usize, max_run: usize) -> Result<Vec<u32>> {
        ensure!(max_chars > 0 && max_run > 0, "both caps must be positive");
        let mut out = Vec::new();
        // BOTH caps, nested as `_encode_text_piece` nests them: the outer one slices characters
        // and the inner run counter RESTARTS inside each slice. That restart is the entire
        // observable difference — see [`MAX_ENCODE_CHARS`].
        for outer in char_chunks(text, max_chars) {
            for chunk in split_same_class_runs(outer, max_run) {
                self.encode_chunk(chunk, &mut out)?;
            }
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
    /// Merges the **leftmost lowest-rank** adjacent pair until none is in the vocabulary — exactly
    /// tiktoken's `_byte_pair_merge` rule, tie-break included: both scan for the minimum with a
    /// strict `<` left to right, so equal ranks resolve to the earlier position and merge order
    /// cannot diverge.
    ///
    /// **The equivalence is MEASURED, not reasoned.** Differentially fuzzed against the
    /// first-party encoder (2026-08-17): **8,372 texts, 3,523,794 ids, zero mismatches**, twelve
    /// alphabets plus deep-merge pieces, 800 lossily-decoded random byte strings, and 57 rows
    /// that straddle BOTH caps (24,999 / 25,000 / 25,001 / 26,000 / 399,999 / 400,000 / 400,001 /
    /// 420,000 characters over six repeated characters). `decode` round-tripped every row. Review
    /// reproduced the earlier round independently at 13,000 / 490,925 / 0.
    ///
    /// > **The reference in that harness has to be `_encode_text_piece`, not `Encoding.encode`.**
    /// > An earlier corpus called tiktoken directly and so applied NEITHER cap; it reported three
    /// > mismatches on 26,000-character runs, and the ids it called correct were the unchunked
    /// > ones. The loader was right and the harness was wrong — which would have "fixed" working
    /// > code had the three failures been taken at face value instead of localised. Any future
    /// > re-fuzz must go through both caps.
    ///
    /// **Threaded rather than rescanned, and the old justification for rescanning was false.**
    /// The first version re-probed every adjacent pair in the `HashMap` each merge round — O(n²)
    /// hashed byte-slice lookups — on the argument that "pieces are single words". `PAT_STR` does
    /// not guarantee that: `\s+`, ` ?[^\s\p{L}\p{N}]+` and `[\p{Han}]+` are unbounded, so a pasted
    /// separator line or an unbroken CJK run arrives as ONE piece at the inner cap — 25,000
    /// characters, which for Han is **75,000 bytes**, about 2.8x10⁹ hashed probes and tens of
    /// seconds. That contradicted this file's own [`MAX_SAME_CLASS_RUN`] doc, which says such runs
    /// are reachable. Each merge now updates only the two neighbouring pair ranks: probes are
    /// O(n) and the remaining O(n²) is an integer scan, the same shape tiktoken has.
    ///
    /// `u32::MAX` is the "not a token" sentinel, safe because [`parse_ranks`] proves the ranks are
    /// dense `0..num_base` and no vocabulary approaches 2³²−1.
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        // The whole piece is usually a token already. Also required: the merge below assumes at
        // least two bytes, exactly as tiktoken's `byte_pair_encode` asserts `len > 1`.
        if let Some(&rank) = self.ranks.get(piece) {
            out.push(rank);
            return Ok(());
        }
        // `parts[i] = (start offset, rank of the pair starting there)`, plus two sentinels so the
        // final window walk and the neighbour update need no bounds special-casing.
        let mut parts: Vec<(usize, u32)> = (0..piece.len() - 1)
            .map(|i| (i, self.rank_of(&piece[i..i + 2])))
            .collect();
        parts.push((piece.len() - 1, u32::MAX));
        parts.push((piece.len(), u32::MAX));
        while let Some(at) = lowest(&parts) {
            if at > 0 {
                parts[at - 1].1 = self.pair_rank(piece, &parts, at - 1);
            }
            parts[at].1 = self.pair_rank(piece, &parts, at);
            parts.remove(at + 1);
        }
        for w in parts.windows(2) {
            let part = &piece[w[0].0..w[1].0];
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

    /// The rank of the pair that would result from merging at `at`, or `u32::MAX`.
    fn pair_rank(&self, piece: &[u8], parts: &[(usize, u32)], at: usize) -> u32 {
        parts
            .get(at + 3)
            .map_or(u32::MAX, |&(end, _)| self.rank_of(&piece[parts[at].0..end]))
    }

    /// A pair's rank, or `u32::MAX` when it is not a token.
    fn rank_of(&self, pair: &[u8]) -> u32 {
        self.ranks.get(pair).copied().unwrap_or(u32::MAX)
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

/// Index of the leftmost lowest-rank mergeable pair, or `None` when none is a token.
///
/// Free rather than a method because it reads only the threaded rank column. The final entry is a
/// sentinel and never a pair, hence the `len() - 1`.
fn lowest(parts: &[(usize, u32)]) -> Option<usize> {
    let mut best: Option<(u32, usize)> = None;
    for (at, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
        // Strict `<` keeps the LEFTMOST of equal ranks — tiktoken's tie-break.
        if rank != u32::MAX && best.is_none_or(|(seen, _)| rank < seen) {
            best = Some((rank, at));
        }
    }
    best.map(|(_, at)| at)
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

/// Slice `s` into runs of at most `max_chars` CHARACTERS — the reference's outer chunk.
///
/// Characters, not bytes: the reference indexes a Python `str`, so its boundary falls on
/// codepoints. A byte slice would both land elsewhere and risk splitting one.
fn char_chunks(s: &str, max_chars: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut n) = (0usize, 0usize);
    for (at, _) in s.char_indices() {
        if n == max_chars {
            out.push(&s[start..at]);
            start = at;
            n = 0;
        }
        n += 1;
    }
    // Always pushes a final slice, so `""` yields `[""]` — `split_same_class_runs` is written for
    // that and the empty case must still reach it.
    out.push(&s[start..]);
    out
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
    // Crate-wide `unwrap`/`expect` are deny; in a unit test a firing one IS the report, and
    // these fixtures carry no checkpoint that could be legitimately missing.
    #![allow(clippy::expect_used, clippy::unwrap_used)]
    use super::*;

    /// The pattern must COMPILE under the engine that has to run it, and SPLIT as the reference
    /// does. Separate from the id gate because a pattern that fails to compile makes every other
    /// case fail for the same uninformative reason, and because this is the assertion that
    /// `fancy-regex` — not `regex` — is the right dependency.
    ///
    /// **Load-bearing for the Han intersections, which id equality cannot see** (see `PAT_STR`).
    /// The `"AB你好c"` row observes the split BEHAVIOURALLY rather than comparing pattern text,
    /// so it reddens on a broken intersection even though every id stays identical — measured
    /// 2026-08-17. Checkpoint-free, so this is one of the few gates on this arm that asserts in
    /// CI rather than skipping.
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
        // And the class intersections keep Han out of the Latin runs. **`AB你好c` rather than
        // `hello你好`, because there are FOUR intersection clauses and one probe does not see
        // them all** (measured 2026-08-17, after review pointed out the gap): `hello你好`
        // detects only alt-2's LOWERCASE clause, and `A你好` — the probe review itself proposed
        // — detects three of four, missing alt-2's UPPERCASE one. `AB你好c` exercises an
        // uppercase run, a Han run and a trailing lowercase run in one string, and removing ANY
        // of the four changes its split. Without it, two of the four clauses had no behavioural
        // gate at all and rested on string equality alone.
        assert_eq!(
            split("AB你好c"),
            ["AB", "你好", "c"],
            "Han was absorbed into a Latin run — one of the four &&[^Han] clauses is broken"
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

    /// A vocabulary with all 256 bytes, plus whatever extra tokens the caller wants. The
    /// smallest thing `assemble` accepts, so a unit test can exercise a `Vocab` property with no
    /// checkpoint on disk.
    fn synthetic(extra: &[&[u8]]) -> Vocab {
        try_synthetic(extra, HashMap::new()).expect("synthetic vocab")
    }

    /// The fallible form, and the one that takes NAMED specials. One builder, because three
    /// fixtures wanted a `Vocab` with no checkpoint and jscpd reported the copies.
    fn try_synthetic(extra: &[&[u8]], named: HashMap<u32, String>) -> Result<Vocab> {
        let mut bytes_of: Vec<Vec<u8>> = (0..=u8::MAX).map(|b| vec![b]).collect();
        bytes_of.extend(extra.iter().map(|b| b.to_vec()));
        let ranks = ranks_from(&bytes_of);
        Vocab::assemble(ranks, bytes_of, named)
    }

    /// **The Han tripwire fires**, proven on a vocabulary that has a mixing token.
    ///
    /// Without this the tripwire is a check that has only ever been green, which is the shape the
    /// house rules refuse. It also lands the proof in CI: the real-vocabulary census needs a 2.7 MB
    /// `tiktoken.model` and skips without it, while this needs nothing.
    #[test]
    fn the_han_mixing_tripwire_fires_on_a_token_that_mixes() {
        // Clean: single bytes only. A lone byte of a multi-byte codepoint is not valid UTF-8 and
        // is skipped, so no single-byte vocabulary can mix scripts.
        assert_eq!(synthetic(&[]).mixed_script_token(), None);
        // `a` + 你 in ONE token — exactly what would let a byte-pair merge cross the boundary.
        let v = synthetic(&["a你".as_bytes()]);
        let (rank, text) = v
            .mixed_script_token()
            .expect("the mixing token must be found");
        assert_eq!((rank, text.as_str()), (256, "a你"));
        // And Han alone is NOT mixing — otherwise the tripwire would fire on every CJK vocabulary.
        assert_eq!(synthetic(&["你好".as_bytes()]).mixed_script_token(), None);
    }

    /// **The prefix tripwire fires.** `find_special` is longest-match; tiktoken is
    /// leftmost-first-alternative, and the two agree only while no spelling prefixes another.
    #[test]
    fn the_special_prefix_tripwire_fires_on_a_prefixed_spelling() {
        // K3's own 256 positional spellings differ only in an id, so none prefixes another.
        assert_eq!(synthetic(&[]).prefixed_special(), None);
        // Name two slots so that one spelling IS a strict prefix of the other.
        let mut bytes_of: Vec<Vec<u8>> = (0..=u8::MAX).map(|b| vec![b]).collect();
        bytes_of.push(b"zz".to_vec());
        let ranks = ranks_from(&bytes_of);
        let named = HashMap::from([
            (257u32, "<|s|>".to_string()),
            (258u32, "<|s|>x".to_string()),
        ]);
        let v = Vocab::assemble(ranks, bytes_of, named).expect("synthetic vocab");
        assert_eq!(
            v.prefixed_special(),
            Some(("<|s|>".to_string(), "<|s|>x".to_string()))
        );
    }

    /// **The distinctness invariant fires.** Two named slots given the SAME spelling.
    ///
    /// Cannot be red-proofed by perturbing the real artifact — no shipped config collides — so the
    /// proof has to be a synthetic one, and it lands in CI where the artifact never is.
    #[test]
    fn colliding_special_spellings_are_refused() {
        let named = HashMap::from([
            (257u32, "<|dup|>".to_string()),
            (258u32, "<|dup|>".to_string()),
        ]);
        let err = match try_synthetic(&[b"zz"], named) {
            Ok(_) => panic!("two ids sharing one spelling was accepted: one is unencodable"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("distinct spellings"), "unexpected: {err}");
    }

    #[test]
    fn non_dense_ranks_are_refused() {
        // `AQ==` is byte 0x01. Rank 0 then rank 2: decode indexes by rank, so the gap would
        // shift every token above it.
        let err = parse_ranks("AQ== 0\nAg== 2\n").unwrap_err();
        assert!(err.to_string().contains("not dense"), "unexpected: {err}");
    }
}
