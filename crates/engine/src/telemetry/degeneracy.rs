//! Did the generation come out broken? Three exact detectors over the produced
//! tokens and text — a tail cycle ([`detect_loop`]), a long-range restart
//! ([`has_repeated_block`]) and a local template loop ([`repetition_report`] +
//! [`is_degenerate`]) — and the record of why each of the tempting cheap gates
//! (distinct-word ratio, MATTR-200, longest-repeated-block-as-a-length) was tried
//! and rejected. Nothing here times anything or touches the device: it reads the
//! output after the run, which is why it is its own module and not part of the
//! performance record.
//!
//! **Moved out of `telemetry.rs` verbatim on 2026-08-15**, which had grown past
//! CodeScene's file-size cliff (~880 lines) and scored 8.81 on Low Cohesion alone: the
//! one file held four jobs that never call each other. The cut is by COHESION, not by
//! line count — this is one whole LCOM4 component, moved intact with its comments and
//! the measurements they carry. `telemetry.rs` re-exports what was public, so every
//! path that resolved before still resolves.

/// A verbatim repetition loop found at the tail of a generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopReport {
    /// Length of the repeating block, in tokens.
    pub period: usize,
    /// How many consecutive verbatim copies of it end the generation.
    pub repeats: usize,
    /// Index in the generated sequence where the repetition began.
    pub start: usize,
}

/// Does some block of `k` tokens occur at least twice anywhere in `ids`?
///
/// The companion to [`detect_loop`], and needed because a tail cycle is the *late* stage
/// of degeneration. The early stage is a RESTART: the model answers, then answers again
/// in slightly different words. Observed on a real 128-token run — two near-identical
/// Rayleigh-scattering paragraphs — where `detect_loop` correctly found no verbatim
/// cycle (the paragraphs differed by a word) and could not have found one anyway, since
/// three repeats of a ~60-token block do not fit in 128 tokens. A run can be obviously
/// broken and still have no cycle, so both signals are needed.
///
/// A PREDICATE against a caller-supplied `k`, not a length. This was
/// `longest_repeated_block`, which binary-searched the length over this same rolling-hash
/// probe — O(n log n) rather than O(n) — and its one consumer then compared the answer to
/// a single bar, `max(32, n/8)`. log2(n) probes to compute a number whose only use was one
/// comparison; one probe at the bar answers the question. The exact length survives in the
/// warning only as "a block this big repeats", which is what the warning ever acted on.
///
/// Dropping the search also dropped its `hi = n/2` cap, and that is a fix rather than a
/// side effect: for a generation under 64 tokens the bar (floor 32) exceeded the cap, so
/// the restart warning could never fire however broken the output was. A `k`-block CAN
/// occur twice in fewer than `2k` tokens — that is exactly what a short periodic restart
/// looks like — and now it is seen.
///
/// Healthy prose repeats short phrases; a restart repeats whole sentences.
pub fn has_repeated_block(ids: &[u32], k: usize) -> bool {
    use std::collections::HashSet;
    // Two independent moduli: a single 64-bit rolling hash would make a false positive
    // (and thus a false "degenerate") possible on adversarial input, and this decides
    // whether a benchmark cell gets thrown away.
    const M1: u64 = 1_000_000_007;
    const M2: u64 = 998_244_353;
    const B1: u64 = 131;
    const B2: u64 = 137;
    let n = ids.len();
    if k == 0 || k > n {
        return false;
    }
    let (mut p1, mut p2) = (1u64, 1u64);
    for _ in 0..k {
        p1 = p1 * B1 % M1;
        p2 = p2 * B2 % M2;
    }
    let (mut h1, mut h2) = (0u64, 0u64);
    for &t in &ids[..k] {
        h1 = (h1 * B1 + u64::from(t) + 1) % M1;
        h2 = (h2 * B2 + u64::from(t) + 1) % M2;
    }
    let mut seen = HashSet::with_capacity(n);
    seen.insert((h1, h2));
    for i in k..n {
        h1 = (h1 * B1 + u64::from(ids[i]) + 1 + M1 * p1 - p1 * (u64::from(ids[i - k]) + 1) % M1)
            % M1;
        h2 = (h2 * B2 + u64::from(ids[i]) + 1 + M2 * p2 - p2 * (u64::from(ids[i - k]) + 1) % M2)
            % M2;
        if !seen.insert((h1, h2)) {
            return true;
        }
    }
    false
}

/// Detect a verbatim repetition loop at the **tail** of a generation.
///
/// **Deliberately not a distinct-token ratio.** [INT4.md](../docs/investigations/int4-scales.md) showed that
/// gate inverts: hybrid has the worst distinct ratio in the engine (0.138) and the
/// second-best perplexity, so a diversity threshold would reject the best config we
/// have. Repetitiveness is not the signal — a *cycle* is. The tail being literally N
/// verbatim copies of one block is something prose does not do and a wedged decode
/// always does, so it is a hard classifier rather than a soft one.
///
/// Returns the SMALLEST period that qualifies, so an ABABAB loop reports period 2 rather
/// than 4 or 6, and then walks backwards to count every copy and locate the onset.
pub fn detect_loop(ids: &[u32], min_repeats: usize, max_period: usize) -> Option<LoopReport> {
    if min_repeats < 2 {
        return None;
    }
    let n = ids.len();
    for period in 1..=max_period.min(n / min_repeats) {
        let block = &ids[n - period..];
        // Does the tail end in `min_repeats` copies of `block`?
        if !(1..min_repeats).all(|k| ids[n - period * (k + 1)..n - period * k] == *block) {
            continue;
        }
        // Qualifies. Walk back for the true count — the onset is the interesting part,
        // because "looped for the last 12 tokens" and "looped for the last 400" are very
        // different failures.
        let mut repeats = min_repeats;
        while n >= period * (repeats + 1)
            && ids[n - period * (repeats + 1)..n - period * repeats] == *block
        {
            repeats += 1;
        }
        return Some(LoopReport {
            period,
            repeats,
            start: n - period * repeats,
        });
    }
    None
}

/// Structural repetition — the signal both exact-matching detectors are blind to.
///
/// **Added because [`detect_loop`] and [`has_repeated_block`] BOTH passed a run
/// whose output was 329 repetitions of `**Memory Product.**`.** The loop had a varying
/// slot — `**Memory Phase:**`, `**Memory State:**`, `**Memory Status:**`, … — so there
/// was no verbatim cycle and the longest exact block was only 142 tokens. A near-miss
/// loop with one changing token is the most common real degeneration shape there is, and
/// exact matching cannot see it.
///
/// Two cheap signals that can:
/// - `top_line`: how many times the single most repeated line occurs. 1 on healthy
///   output; 38 / 53 / 329 as one run degenerated over 2048 / 4096 / 10000 tokens.
/// - `distinct`: distinct-word ratio. 0.43–0.53 healthy, 0.12–0.29 degenerate, and it
///   fell monotonically (0.474 → 0.366 → 0.288 → 0.244) across that same run.
///
/// On `distinct`: [INT4.md](../docs/investigations/int4-scales.md) warns that a distinct-token gate INVERTS —
/// hybrid has the worst ratio in the engine and the second-best perplexity. That warning
/// is about ranking *healthy* configs against each other, where the ratio does not track
/// quality. Reading it as "never use distinct ratio" was an over-generalisation, and it
/// cost four rounds of a benchmark matrix: in the 0.24 regime the output is visibly
/// broken, and this was the one instrument that would have said so. It is an ALARM, not
/// a ranking metric.
#[derive(Debug, Clone, Copy)]
pub struct RepetitionReport {
    pub top_line: usize,
    pub distinct: f64,
}

/// Structural-repetition signals over generated TEXT (not tokens — the varying slot is a
/// token-level difference, which is exactly why token-level exact matching misses it).
pub fn repetition_report(text: &str) -> RepetitionReport {
    RepetitionReport {
        top_line: top_line_count(text),
        distinct: distinct_word_ratio(text),
    }
}

/// Occurrences of the single most repeated line. Lines of 3 characters or fewer do not
/// count: blank lines and stray punctuation repeat in healthy prose too, and the signal
/// being read is a repeated TEMPLATE.
fn top_line_count(text: &str) -> usize {
    let mut lines: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for l in text.lines().map(str::trim).filter(|l| l.len() > 3) {
        *lines.entry(l).or_default() += 1;
    }
    lines.values().copied().max().unwrap_or(0)
}

/// Distinct-word ratio, case-folded, over alphabetic runs. Empty input scores 1.0 — no
/// words is the absence of a measurement, not zero diversity, and 0.0 would read as the
/// most degenerate output possible.
fn distinct_word_ratio(text: &str) -> f64 {
    let mut words = 0usize;
    let mut uniq = std::collections::HashSet::new();
    for w in text
        .split(|c: char| !c.is_alphabetic())
        .filter(|w| !w.is_empty())
    {
        words += 1;
        uniq.insert(w.to_ascii_lowercase());
    }
    if words == 0 {
        1.0
    } else {
        uniq.len() as f64 / words as f64
    }
}

/// Does this generation look structurally degenerate? **A repeated line only.**
///
/// The distinct-word ratio was in this test and has been REMOVED: type-token ratio falls
/// with length in perfectly healthy text, so a flat threshold is length-confounded.
/// Measured on real prose from `tests/ppl-corpus-5000.txt`: 0.593 at 300 words, 0.517 at
/// 800, 0.455 at 1500, **0.376 at 3000** — already under a 0.42 "healthy band" that was
/// derived from 512-token samples. It flagged a 2048-token run whose line repetition
/// (x4) and longest repeated block (175 of 2048) were both healthy.
///
/// A window-based variant (MATTR-200) fixes the length confound and is still not usable
/// alone: the 10k DSA run, 45% of whose output was a verbatim duplicate, scored **0.701 —
/// higher than real prose** — because a long-range restart looks diverse inside every
/// 200-word window. Local and long-range repetition need different instruments.
///
/// So the working set is two complementary EXACT signals, and neither is a diversity
/// measure: `top_line` here for local template loops (healthy 1–4, broken 25–329), and
/// [`has_repeated_block`] for long-range restarts (the longest repeat measured 6–18 tokens
/// on healthy runs against 4544 on the broken one, so a bar in the tens separates them by
/// two orders of magnitude). Both observed failures are caught by one or the other.
pub fn is_degenerate(r: &RepetitionReport) -> bool {
    r.top_line > 20
}

#[cfg(test)]
mod loop_tests {
    use super::{LoopReport, detect_loop};

    #[test]
    fn detects_cycles_and_leaves_prose_alone() {
        // Healthy: no verbatim cycle at the tail.
        assert_eq!(detect_loop(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 32), None);
        // Repetitive but NOT cyclic — the exact case a distinct-ratio gate would fail on
        // and this must not: only 3 distinct tokens, yet no repeating period.
        assert_eq!(detect_loop(&[1, 1, 2, 1, 1, 1, 2, 2, 1], 3, 32), None);

        // Period 1: the same token over and over.
        assert_eq!(
            detect_loop(&[9, 8, 7, 7, 7, 7], 3, 32),
            Some(LoopReport {
                period: 1,
                repeats: 4,
                start: 2
            })
        );
        // Period 3, and the SMALLEST period wins (this also matches period 6).
        assert_eq!(
            detect_loop(&[5, 1, 2, 3, 1, 2, 3, 1, 2, 3], 3, 32),
            Some(LoopReport {
                period: 3,
                repeats: 3,
                start: 1
            })
        );
        // Below the repeat threshold: two copies is a couplet, not a wedge.
        assert_eq!(detect_loop(&[4, 1, 2, 3, 1, 2, 3], 3, 32), None);
        assert_eq!(
            detect_loop(&[4, 1, 2, 3, 1, 2, 3], 2, 32).map(|r| r.period),
            Some(3)
        );
        // max_period must bound the search.
        assert_eq!(
            detect_loop(&[1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4], 3, 3),
            None
        );
        // Degenerate inputs must not panic or divide by zero.
        assert_eq!(detect_loop(&[], 3, 32), None);
        assert_eq!(detect_loop(&[7], 3, 32), None);
        assert_eq!(detect_loop(&[7, 7, 7], 1, 32), None); // min_repeats < 2 is meaningless
    }
}

#[cfg(test)]
mod lrb_tests {
    use super::has_repeated_block;

    /// The predicate is exact at the bar in BOTH directions — the length it replaced was
    /// only ever compared to one, so a test that pins the answer just below and just above
    /// `k` covers everything the search used to.
    #[test]
    fn answers_exactly_at_the_bar() {
        assert!(!has_repeated_block(&[], 1));
        assert!(!has_repeated_block(&[1, 2, 3, 4, 5], 1));
        // One token repeated, and no pair.
        assert!(has_repeated_block(&[1, 2, 3, 2, 9], 1));
        assert!(!has_repeated_block(&[1, 2, 3, 2, 9], 2));
        // A 3-block that recurs non-adjacently — the RESTART shape, which detect_loop
        // deliberately does not flag.
        assert!(has_repeated_block(&[7, 1, 2, 3, 8, 9, 1, 2, 3], 3));
        assert!(!has_repeated_block(&[7, 1, 2, 3, 8, 9, 1, 2, 3], 4));
        // A full cycle: half the sequence repeats.
        assert!(has_repeated_block(&[1, 2, 3, 4, 1, 2, 3, 4], 4));
        // A block of more than half the sequence cannot occur twice — but one of exactly
        // half can, and the deleted binary search's `hi = n/2` cap is why that is worth
        // stating: the predicate is asked at bars the search could never return.
        assert!(!has_repeated_block(&[1, 2, 3, 4, 1, 2, 3, 4], 5));
        let long: Vec<u32> = (0..100).collect();
        assert!(!has_repeated_block(&long, 1));
        // Degenerate arguments must answer, not panic: a 0-token block is not a block,
        // and no block longer than the input can occur in it at all.
        assert!(!has_repeated_block(&[1, 2, 3], 0));
        assert!(!has_repeated_block(&[1, 2, 3], 4));
    }
}

#[cfg(test)]
mod rep_tests {
    use super::{is_degenerate, repetition_report};

    #[test]
    fn catches_the_varying_slot_loop_that_exact_matching_missed() {
        // The real failure: structure repeats, one slot varies, so no verbatim cycle
        // exists and the longest exact block is short. Both other detectors pass this.
        let labels = [
            "Phase",
            "State",
            "Status",
            "Mode",
            "Form",
            "Shape",
            "Size",
            "Scale",
            "Scope",
            "Range",
            "Navigating",
            "Conducting",
            "Managing",
            "Administering",
            "Organizing",
            "Coordinating",
            "Arranging",
            "Ordering",
            "Systematizing",
            "Structuring",
            "Sequencing",
            "Aligning",
        ];
        let mut loopy = String::new();
        for l in labels.iter().cycle().take(60) {
            loopy.push_str(&format!("**Memory {l}:**\n**Memory Product.**\n\n"));
        }
        let r = repetition_report(&loopy);
        assert!(r.top_line > 20, "top_line was {}", r.top_line);
        assert!(is_degenerate(&r));

        // Length must not by itself trip the alarm. Long healthy text has a LOW
        // distinct-word ratio (real prose is 0.376 at 3000 words) and must still pass,
        // which is why `is_degenerate` gates on line repetition alone.
        let mut long_ok = String::new();
        for i in 0..400 {
            long_ok.push_str(&format!(
                "Page {i} is mapped lazily so untouched allocations cost nothing at all.\n"
            ));
        }
        let r = repetition_report(&long_ok);
        assert!(
            r.distinct < 0.30,
            "distinct was {} — pick a lower-entropy filler",
            r.distinct
        );
        assert!(r.top_line <= 20, "top_line was {}", r.top_line);
        assert!(
            !is_degenerate(&r),
            "a flat distinct-ratio threshold would false-positive here"
        );

        // Healthy prose: no line repeats, high distinct ratio.
        let ok = "Virtual memory gives each process a private address space. \
                  The kernel maps pages lazily, so untouched allocations cost nothing. \
                  A TLB caches recent translations; invalidating it is expensive, which \
                  is why context switches try to avoid full flushes on tagged hardware.";
        let r = repetition_report(ok);
        assert_eq!(r.top_line, 1);
        assert!(r.distinct > 0.4, "distinct was {}", r.distinct);
        assert!(!is_degenerate(&r));

        assert!(!is_degenerate(&repetition_report("")));
    }
}
