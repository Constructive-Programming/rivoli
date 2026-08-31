//! The vendored tensor-family censuses, and the reader they share.
//!
//! A census is a reduction of one checkpoint's `model.safetensors.index.json` plus its shard
//! headers, vendored as a TSV under `docs/measurement/<model>-reference/` because the indexes
//! themselves are 17–60 MB. It carries, per family, the collapsed name pattern, the shape, the
//! dtype, the tensor count and the byte total — which is every fact a converter needs to refuse
//! a checkpoint that is not the one it was written against, and none of the weights.
//!
//! **Why this is library code and not a test helper.** It arrived as one: `k3_names.rs` and
//! `qwen_names.rs` each grew a parser, and the second one's header recorded the debt in place —
//! "lifting one shared helper is owed at S4, when the converter census gives the second reader a
//! reason to exist". S4 is here, and the reader it needed turned out to be `convert_qwen`
//! itself: the converter's exclusion list IS the census (MTP and vision excluded by NAME,
//! exhaustively, so a family the census does not know is a hard error), which makes the TSV a
//! runtime input to a binary rather than a fixture. A test-only module in `crates/artifact/tests`
//! could not have served it, and a copy in `crates/cli` would have been the third parser.
//!
//! **Two column contracts, one reader.** K3's TSV is 4 columns, count first, no header row;
//! qwen's is 8 columns with a `#`-prefixed header row and the count in column 6. Neither
//! contract is the other, so what is shared is the *mechanics* — comment filtering, tab
//! splitting, an asserted column count, `x`-separated dimensions, and the two header-token
//! idioms — while each census's meaning stays with its own module. That split is what the debt
//! note asked for; a single `Family` type over both would have had to special-case every column.
//!
//! **Everything here returns `Result`.** A malformed census is a broken fixture, and the callers
//! are a converter (which must refuse, not panic) as well as three gates (which may `expect`
//! under their own file allow).

pub mod qwen;

use anyhow::{Context, Result, ensure};

/// One vendored census TSV: its repo-relative path, for messages, and its bytes.
///
/// `&'static str` rather than a file read, on the precedent `k3_config.rs` states: a path read
/// would SKIP on a machine without the checkpoint rather than run, and these files are tens of
/// kilobytes. The path is carried alongside so a refusal names the file a reader must open,
/// which `include_str!` alone throws away.
pub struct Tsv {
    path: &'static str,
    text: &'static str,
}

/// One data row: its fields, and enough context for a refusal to name itself.
pub struct Row<'a> {
    path: &'static str,
    line: &'a str,
    fields: Vec<&'a str>,
}

impl Tsv {
    pub const fn new(path: &'static str, text: &'static str) -> Self {
        Tsv { path, text }
    }

    /// The whole file, for a caller that hashes it or greps it.
    pub const fn text(&self) -> &'static str {
        self.text
    }

    pub const fn path(&self) -> &'static str {
        self.path
    }

    /// Every data row — everything that is neither a `#` comment nor blank — each asserted to
    /// have exactly `cols` tab-separated fields.
    ///
    /// The column count is a parameter rather than a property of the file because it is the ONE
    /// thing the two contracts disagree about that a reader can check cheaply, and checking it
    /// is what turns "the file moved" into a named refusal instead of a parse error three
    /// columns later. A `#`-prefixed column-header row (qwen's) is filtered out here, which is
    /// exactly why that prefix is on it.
    pub fn rows(&self, cols: usize) -> Result<Vec<Row<'_>>> {
        self.text
            .lines()
            .filter(|l| !(l.starts_with('#') || l.trim().is_empty()))
            .map(|line| {
                let fields: Vec<&str> = line.split('\t').collect();
                ensure!(
                    fields.len() == cols,
                    "{}: malformed row, {} fields where the contract is {cols}: {line:?}",
                    self.path,
                    fields.len()
                );
                Ok(Row {
                    path: self.path,
                    line,
                    fields,
                })
            })
            .collect()
    }

    /// The whitespace token after `keyword`, on the one header line that starts `# <anchor>`.
    ///
    /// **Anchored at a line start rather than searched loose**, because several of these labels
    /// are substrings of each other — `family rows` occurs inside "sum of `count` over all
    /// family rows", and `v1` inside `v1-status` — so a loose `find` would read a number off the
    /// wrong line and still pass. Refuses when a label moves rather than defaulting: a reworded
    /// header must be a visible edit, not a check that quietly stops checking.
    ///
    /// This is the qwen contract's idiom (`# <label>   <number>`, number after the phrase); see
    /// [`Self::declared_before`] for K3's, which is written the other way round.
    pub fn header_token(&self, anchor: &str, keyword: &str) -> Result<&'static str> {
        let at = self.text.find(&format!("\n# {anchor}")).with_context(|| {
            format!(
                "no line in {}'s header starts `# {anchor}`; a caller reads its expected values \
                 from there",
                self.path
            )
        })?;
        let line = self.text[at + 1..]
            .lines()
            .next()
            .with_context(|| format!("{}: nothing follows `# {anchor}`", self.path))?;
        let after = line
            .find(keyword)
            .with_context(|| format!("`# {anchor}`'s line carries no `{keyword}`: {line:?}"))?
            + keyword.len();
        line[after..]
            .split_whitespace()
            .next()
            .with_context(|| format!("`# {anchor}`'s `{keyword}` has no value: {line:?}"))
    }

    /// A count the header declares as `# <label>   <number>`, where the label is its own anchor.
    pub fn declared(&self, label: &str) -> Result<u64> {
        self.header_token(label, label)?
            .parse()
            .with_context(|| format!("`# {label}` in {} is not followed by a number", self.path))
    }

    /// A count the header declares as `<number> <phrase>` — the integer immediately BEFORE
    /// `phrase`, which is K3's header contract.
    ///
    /// Not anchored at a line start, because K3's header states these mid-sentence. That is the
    /// weaker of the two idioms and it is kept rather than "improved": the file it reads is
    /// vendored and its wording is what the gate cites, so changing the contract would be an
    /// edit to a fixture in order to tidy a reader.
    pub fn declared_before(&self, phrase: &str) -> Result<u64> {
        let at = self.text.find(phrase).with_context(|| {
            format!(
                "{} no longer says `{phrase}`; a caller reads its counts from there",
                self.path
            )
        })?;
        let digits: String = self.text[..at]
            .trim_end()
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect();
        digits
            .chars()
            .rev()
            .collect::<String>()
            .parse()
            .with_context(|| format!("no number before `{phrase}` in {}", self.path))
    }
}

impl Row<'_> {
    /// Field `i`, by position.
    pub fn str_at(&self, i: usize) -> Result<&str> {
        self.fields
            .get(i)
            .copied()
            .with_context(|| format!("{}: no field {i} in {:?}", self.path, self.line))
    }

    /// Field `i` as an integer. Named for the column so a refusal says which one.
    pub fn u64_at(&self, i: usize, what: &str) -> Result<u64> {
        self.str_at(i)?
            .parse()
            .with_context(|| format!("{}: {what} is not an integer in {:?}", self.path, self.line))
    }

    /// Field `i` as `x`-separated dimensions.
    ///
    /// A lone `?` yields the EMPTY vector — K3's census records an unknown shape that way, for a
    /// family none of its fetched shard headers covered, so callers must opt in to using it
    /// rather than reading a zero. Qwen's census has no `?` rows and its own gate asserts every
    /// row's shape against its byte count, which is what makes that absence checkable.
    pub fn dims_at(&self, i: usize) -> Result<Vec<u64>> {
        let raw = self.str_at(i)?;
        if raw == "?" {
            return Ok(Vec::new());
        }
        raw.split('x')
            .map(|d| {
                d.parse().with_context(|| {
                    format!("{}: {d:?} is not a dimension in {:?}", self.path, self.line)
                })
            })
            .collect()
    }
}
