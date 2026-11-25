//! [`LAUNCHED_BY`]: the census's mapping for kernels a launcher of another name runs, and the
//! checks that keep it honest — split out of `kernel_coverage.rs` (2026-11-19) to keep that
//! file under the 800-line soft cap. Red proofs: `docs/measurement/gate-red-proofs.md` §19.

use super::{Text, UNROWED, gate_rows, shipped_kernels};
use std::path::Path;

/// `(kernel, launcher)`: kernels run by a launcher of ANOTHER name — one phase of a
/// multi-kernel launch (a split-softmax pass and its combine), one of the instantiations a
/// launcher selects per call (`nrow`'s R = 2, fp8's split-K route), or a single kernel under a
/// launcher whose name the tag fixed (the drains' `_s`). The census otherwise pairs each
/// kernel with a `rivoli_<kernel>` of its own name, which one launcher cannot have twice; a
/// listed kernel is rowed under `rivoli_<launcher>`, whose gate row and `=>` row score and
/// reach it. Checked at both ends ([`stale_launched_by`] names each refusal); an unlisted
/// phase reads as missing from both populations, like any unrowed kernel. That a row reaches
/// EVERY phase or selectable instantiation (a split needs a shape that splits) is argued by
/// the row's shapes, at the row.
pub(super) const LAUNCHED_BY: [(&str, &str); 27] = [
    // `rivoli_mla_attend`: the split-KV attend, then (when the plan splits) the combine. Its
    // gate row's shapes reach both: `nr` from 1 (one split) past 2048 (16).
    ("mla_latent_attend", "mla_attend"),
    ("mla_attend_combine", "mla_attend"),
    // Phase 1 L3 (2026-11-19): the MLA pair's two-token-row kernels, which the launcher picks
    // by `nrow` — the rows sweep `nrow ∈ {1, 2}`, so each reaches both kernels.
    ("mla_absorb_fp8_r2", "mla_absorb_fp8"),
    ("mla_value_fp8_r2", "mla_value_fp8"),
    // Phase 1 L1 (2026-11-19): `nrow` picks R = 2, `i_dim >= GEMV_SPLITK_MIN_IDIM` the split-K
    // route (linalg's gemv rows sweep both).
    ("gemv_fp8_r2", "gemv_fp8"),
    ("gemv_fp8_splitk", "gemv_fp8"),
    ("gemv_fp8_splitk_r2", "gemv_fp8"),
    ("gemv_i8_r2", "gemv_i8"),
    ("gemv_f32_r2", "gemv_f32"),
    // The gate, then the blend, on one stream.
    ("hc_head_collapse_gate", "hc_head_collapse"),
    ("hc_head_collapse_blend", "hc_head_collapse"),
    // Phase 1 L2 (2026-11-19): each GLM range launcher runs a gate/up and a down pass at R = 1
    // or 2; its row's shapes cover both widths (`bit_identity/moe.rs`, `VQ`/`I4`).
    ("moe_gateup_vq", "moe_expert_range"),
    ("moe_gateup_vq_r2", "moe_expert_range"),
    ("moe_down_vq", "moe_expert_range"),
    ("moe_down_vq_r2", "moe_expert_range"),
    ("moe_gateup_i4", "moe_expert_range_i4"),
    ("moe_gateup_i4_r2", "moe_expert_range_i4"),
    ("moe_down_i4", "moe_expert_range_i4"),
    ("moe_down_i4_r2", "moe_expert_range_i4"),
    // The fp4 pairs: two passes each, R = 1 only.
    ("moe_gateup_f4", "moe_expert_range_f4"),
    ("moe_down_f4", "moe_expert_range_f4"),
    ("moe_gateup_f4_situ", "moe_expert_range_f4_situ"),
    ("moe_down_f4_weighted", "moe_expert_range_f4_situ"),
    // Inkling-2 Small's MXFP8 pair (S6, 2026-11-25): the same two-pass shape as the fp4
    // twins, and no C twin, so it is `ORACLE_VALIDATED` in `kernel_coverage.rs` rather than
    // rowed by a bit-identity `ROWS` entry — the two census questions stay separate.
    ("moe_gateup_mxfp8_situ", "moe_expert_range_mxfp8_situ"),
    ("moe_down_mxfp8_weighted", "moe_expert_range_mxfp8_situ"),
    // The drains: one kernel each, under the tag's `_s` launcher names.
    ("moe_acc_drain", "moe_acc_drain_s"),
    ("moe_acc_drain_to", "moe_acc_drain_to_s"),
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
/// is `UNROWED`, is listed twice, has a row of its own, or is not launched by
/// `rivoli_<launcher>`'s own top-level body (comments dropped): the body must name the
/// kernel's `STATIC` as an identifier and contain a launch ([`has_launch`]) — so a launcher must
/// name every phase's static itself, not reach it through a helper. LOOSENED 2026-11-19 (Phase 1
/// integration, coordinator decision) from "`run(&STATIC,` or `STATIC.launch…` written
/// literally": that form forced L2's int4 range launcher into an `if` chain where its VQ twin
/// used a `match`, purely so jscpd would not pair them — code bent to satisfy a gate. A launcher
/// may now pick a static into a local and launch the local. The class this gives up, accepted:
/// a body that names a static in code it never launches while it launches another
/// (`gate-red-proofs.md` §19, P5).
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
        (name.chars().all(ident) && !ty.contains(';')).then(|| name.to_string())
    };
    // `rivoli_<l>`'s top-level body (to its column-0 `}`, refused if there is none), comments
    // dropped and whitespace stripped.
    let body = |l: &str| {
        let (_, rest) = launch.0.split_once(&format!("fn rivoli_{l}("))?;
        let (body, _) = rest.split_once("\n}\n")?;
        // Trailing comments too, since 2026-11-19: once naming the static is what counts, a
        // `// MOE_X` after code would otherwise stand in for it.
        let code = body
            .lines()
            .map(|x| x.split_once("//").map_or(x, |(code, _)| code));
        Some(
            code.flat_map(str::chars)
                .filter(|c| !c.is_whitespace())
                .collect::<String>(),
        )
    };
    let runs = |k: &str, l: &str| {
        handle(k)
            .zip(body(l))
            .is_some_and(|(h, b)| names_ident(&b, &h) && has_launch(&b))
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

fn ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Whether `body` names `h` as a whole identifier — not as a prefix or suffix of another name.
/// Comments never count: [`stale_launched_by`] drops them before asking.
fn names_ident(body: &str, h: &str) -> bool {
    body.match_indices(h).any(|(at, _)| {
        let left = body[..at].chars().next_back();
        let right = body[at + h.len()..].chars().next();
        !left.is_some_and(ident) && !right.is_some_and(ident)
    })
}

/// Whether `body` launches anything: `run(`, `run_2d(`, `.launch(` or `.launch_2d(`.
fn has_launch(body: &str) -> bool {
    let free_call = |l: &str| {
        body.match_indices(l)
            .any(|(at, _)| !body[..at].chars().next_back().is_some_and(ident))
    };
    free_call("run(")
        || free_call("run_2d(")
        || body.contains(".launch(")
        || body.contains(".launch_2d(")
}
