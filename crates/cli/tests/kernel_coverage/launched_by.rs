//! [`LAUNCHED_BY`]: the census's mapping for kernels a launcher of another name runs, and the
//! checks that keep it honest — split out of `kernel_coverage.rs` (2026-11-19) to keep that
//! file under the 800-line soft cap. Red proofs: `docs/measurement/gate-red-proofs.md` §19.

use super::{Text, UNROWED, gate_rows, shipped_kernels};
use std::path::Path;

/// `(kernel, launcher)`: kernels run by a launcher of ANOTHER name, as one phase of a
/// multi-kernel launch (a split-softmax pass and its combine). The census otherwise pairs each
/// kernel with a `rivoli_<kernel>` of its own name, which one launcher cannot have twice; a
/// listed kernel is rowed under `rivoli_<launcher>`, whose gate row and `=>` row score and
/// reach it. Checked at both ends ([`stale_launched_by`] names each refusal); an unlisted
/// phase reads as missing from both populations, like any unrowed kernel. That a row reaches
/// EVERY phase (a split needs a shape that splits) is argued by the row's shapes, at the row.
pub(super) const LAUNCHED_BY: [(&str, &str); 2] = [
    // `rivoli_mla_attend`: the split-KV attend, then (when the plan splits) the combine. Its
    // gate row's shapes reach both: `nr` from 1 (one split) past 2048 (16).
    ("mla_latent_attend", "mla_attend"),
    ("mla_attend_combine", "mla_attend"),
];

/// The names the shipped kernels are rowed under — less [`UNROWED`], each kernel's
/// [`LAUNCHED_BY`] launcher where it has one and its own name otherwise — sorted and
/// deduplicated (two phases of one launcher are one row).
pub(super) fn rowed_names() -> Vec<String> {
    let rowed_as = |kernel: &str| {
        LAUNCHED_BY
            .iter()
            .find(|(k, _)| *k == kernel)
            .map_or(kernel, |&(_, l)| l)
            .to_string()
    };
    let mut rowed: Vec<String> = shipped_kernels()
        .iter()
        .filter(|k| !UNROWED.contains(&k.as_str()))
        .map(|k| rowed_as(k))
        .collect();
    rowed.sort();
    rowed.dedup();
    rowed
}

/// The [`LAUNCHED_BY`] entries to refuse, each with its reason: the kernel is not shipped or
/// is `UNROWED`, is listed twice, has a row of its own, or is not LAUNCHED (`run(&STATIC, …)`
/// or `STATIC.launch…`, comment lines dropped) in `rivoli_<launcher>`'s own top-level body —
/// so a launcher must name every phase's static itself, not reach it through a helper.
pub(super) fn stale_launched_by() -> Vec<String> {
    let device = shipped_kernels();
    let rows: Vec<String> = gate_rows().into_iter().map(|(n, _)| n).collect();
    let launch = Text::corpus(Path::new("crates/backend/src/launch"));
    let dense: String = launch.0.chars().filter(|c| !c.is_whitespace()).collect();
    // `k`'s handle: the `NAME` of `static NAME: Kernel<A> = Kernel::new(c"k");`, with no other
    // item between `static` and the match (a `const` or `let` handle is refused, not misread).
    let handle = |k: &str| {
        let (before, _) = dense.split_once(&format!("=Kernel::new(c\"{k}\");"))?;
        let (name, ty) = before.rsplit_once("static")?.1.split_once(":Kernel<")?;
        let ident = name.chars().all(|c| c.is_alphanumeric() || c == '_');
        (ident && !ty.contains(';')).then(|| name.to_string())
    };
    // `rivoli_<l>`'s top-level body (to its column-0 `}`, refused if there is none), comment
    // lines dropped and whitespace stripped.
    let body = |l: &str| {
        let (_, rest) = launch.0.split_once(&format!("fn rivoli_{l}("))?;
        let (body, _) = rest.split_once("\n}\n")?;
        let code = body.lines().filter(|x| !x.trim_start().starts_with("//"));
        Some(
            code.flat_map(str::chars)
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
        )
    };
    // Whether `body` LAUNCHES the static `h` — `run(&h, …)` or `h.launch…(` — rather than
    // merely naming it.
    let launches = |body: &str, h: &str| {
        let via_run = body.contains(&format!("run(&{h},"));
        let via_method = body.match_indices(&format!("{h}.launch")).any(|(at, _)| {
            let left = body[..at].chars().next_back();
            !left.is_some_and(|c| c.is_alphanumeric() || c == '_')
        });
        via_run || via_method
    };
    let runs = |k: &str, l: &str| {
        handle(k)
            .zip(body(l))
            .is_some_and(|(h, b)| launches(&b, &h))
    };
    let reason = |i: usize, k: &str, l: &str| -> Option<&'static str> {
        let checks = [
            (!device.iter().any(|d| d == k), "not a shipped gpu-kernel"),
            (UNROWED.contains(&k), "an UNROWED kernel"),
            (
                LAUNCHED_BY[..i].iter().any(|(e, _)| *e == k),
                "listed twice",
            ),
            (
                rows.iter().any(|r| r == k),
                "the kernel has a row of its own",
            ),
            (
                !runs(k, l),
                "`rivoli_<launcher>` does not launch the kernel's `static` handle",
            ),
        ];
        checks
            .iter()
            .find(|(failed, _)| *failed)
            .map(|(_, why)| *why)
    };
    LAUNCHED_BY
        .iter()
        .enumerate()
        .filter_map(|(i, &(k, l))| Some(format!("({k}, {l}): {}", reason(i, k, l)?)))
        .collect()
}
