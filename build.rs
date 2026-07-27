//! Build script: panics are the correct failure mode for a broken toolchain.
#![allow(clippy::expect_used)]

//! Compile the HIP kernels with hipcc and link libamdhip64 — only under the `rocm`
//! feature, so `cargo check` / CPU-side dev needs no GPU toolchain. (The cold-expert
//! io_uring streamer is the `io-uring` crate now, talking to the syscalls directly —
//! no liburing system lib.)

use std::path::Path;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_FEATURE_VULKAN").is_ok() {
        return vulkan(); // second backend: glslc instead of hipcc, no HIP at all.
    }
    if std::env::var("CARGO_FEATURE_ROCM").is_err() {
        return; // CPU-only build: no HIP toolchain required.
    }

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let kernels = [
        "linalg", "moe", "mla", "attn", "fwd", "vmm", "indexer", "async",
    ];
    let hipcc = std::env::var("HIPCC").unwrap_or_else(|_| "hipcc".into());
    // ponytail: default gfx1151 (Strix Halo); override for other ROCm nodes (e.g.
    // gfx1010 on rh-desktop's RX 5700) with RIVOLI_OFFLOAD_ARCH.
    println!("cargo:rerun-if-env-changed=RIVOLI_OFFLOAD_ARCH");
    let arch = std::env::var("RIVOLI_OFFLOAD_ARCH").unwrap_or_else(|_| "gfx1151".into());
    let offload = format!("--offload-arch={arch}");

    // Headers are #included, not compiled units — track them so an edit forces a rebuild.
    println!("cargo:rerun-if-changed=kernels/common.hpp");
    let mut objs = Vec::new();
    for k in kernels {
        let src = format!("kernels/{k}.hip");
        let obj = format!("{out_dir}/{k}.o");
        println!("cargo:rerun-if-changed={src}");
        let mut cmd = Command::new(&hipcc);
        cmd.args([&offload, "-O3", "-fPIC"]);
        let status = cmd
            .args(["-c", &src, "-o", &obj])
            .status()
            .expect("run hipcc");
        assert!(status.success(), "hipcc failed on {src}");
        objs.push(obj);
    }

    let lib = format!("{out_dir}/librivolikernels.a");
    // `ar crs` only adds/replaces members — it never prunes ones dropped from the
    // list, so a since-deleted kernel's stale .o lingers and duplicate-symbol-clashes.
    // Start from a clean archive every build.
    let _ = std::fs::remove_file(&lib);
    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&lib);
    for o in &objs {
        ar.arg(o);
    }
    assert!(ar.status().expect("run ar").success(), "ar failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=rivolikernels");
    for dir in ["/usr/lib64", "/opt/rocm/lib", "/usr/lib"] {
        if Path::new(dir).join("libamdhip64.so").exists() {
            println!("cargo:rustc-link-search=native={dir}");
            break;
        }
    }
    println!("cargo:rustc-link-lib=dylib=amdhip64");
}

/// The `vulkan` arm: compile every `kernels/vk/*.comp` to SPIR-V in OUT_DIR, which
/// `src/vk.rs` picks up with `include_bytes!`. Nothing is linked — the loader is
/// opened at runtime by `ash`, so a Vulkan build needs no GPU library at link time.
/// Wave width and rows-per-workgroup, defined ONCE here and injected into both
/// languages: `-D` on the glslc command line, and a generated `dims.rs` the Rust side
/// includes. They used to be declared twice with a "must match" comment and nothing
/// checking, which is a regression against HIP — there `ROW_GRID`/`ROW_BLOCK` derive
/// from the single `common.hpp` define, so the launcher geometry and the shader cannot
/// disagree. If they did disagree here the launcher would dispatch too few workgroups
/// and the missing output rows would simply keep their previous contents: no fault, no
/// validation diagnostic, wrong numbers.
const VK_WAVE: u32 = 32;
const VK_ROWS_PER_BLOCK: u32 = 8;

/// The env the shaders are compiled FOR and validated AGAINST. The two tools spell the
/// flag differently: `glslc --target-env=X`, but `spirv-val --target-env X` — the `=`
/// form makes spirv-val print its usage and exit non-zero.
const VK_TARGET_ENV: &str = "vulkan1.3";

/// Modules where a barrier exists but [`no_barrier_without_memory`] declines to judge it,
/// with the LDS the barrier is presumed to be ordering. See that function for why the set
/// is pinned rather than merely reported. Module scope so `vulkan()` can also check that
/// every entry still names a real shader — the deletion case the per-shader comparison
/// cannot observe.
const BARRIER_EXEMPT: &[(&str, &str)] = &[
    ("append_kv", "block-max reduction over the latent row"),
    ("argmax_reduce", "LDS halving tree over the vocabulary partials"),
    ("gemv_fp8", "the e4m3 LUT, plus split-K's per-wave partials"),
    ("rmsnorm", "LDS halving tree over the sum of squares"),
    // Both MLA kernels carry ONE barrier, between E4M3_LUT_BUILD's writes to the shared
    // e4m3 table and the first read of it. That is shared-memory ordering only, which a
    // bare `barrier()` covers — checked by hand, as the diagnostic demands. Neither
    // writes a buffer any other thread reads (each thread owns one output element), so
    // unlike rope_interleave there is no buffer traffic for the barrier to order.
    ("mla_value_fp8", "the e4m3 LUT"),
    ("mla_absorb_fp8", "the e4m3 LUT"),
    // Every barrier in both attention kernels separates a shared WRITE from a shared
    // READ — the staged L/R token tile, and the combine's per-split weights — so bare
    // WorkgroupMemory semantics are what they need. Hand-checked, as the diagnostic
    // demands. Buffer traffic needs no ordering across them: the only writes are the
    // final per-thread stores to disjoint clat/partial slots, after the last barrier.
    ("mla_latent_attend", "the staged L/R token tile"),
    ("mla_attend_combine", "the per-split softmax weights and inv_l"),
];

fn vulkan() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR set by cargo");
    let glslc = std::env::var("GLSLC").unwrap_or_else(|_| "glslc".into());
    std::fs::write(
        format!("{out_dir}/dims.rs"),
        format!(
            "// @generated by build.rs — the single source is VK_WAVE/VK_ROWS_PER_BLOCK there.\n\
             pub const WAVE: u32 = {VK_WAVE};\n\
             pub const ROWS_PER_BLOCK: u32 = {VK_ROWS_PER_BLOCK};\n"
        ),
    )
    .expect("write dims.rs");
    println!("cargo:rerun-if-env-changed=GLSLC");
    // The DIRECTORY entry catches an added/removed shader; the per-file entries below
    // catch edits. Both are needed — and `common.glsl` is #included, not compiled, so
    // without its own entry an edit to it silently ships stale SPIR-V (exactly the
    // common.hpp staleness bug this repo already hit once).
    println!("cargo:rerun-if-changed=kernels/vk");

    let mut shaders: Vec<_> = std::fs::read_dir("kernels/vk")
        .expect("read kernels/vk")
        .map(|e| e.expect("dir entry").path())
        .collect();
    shaders.sort(); // deterministic build order
    for path in &shaders {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    // EVERY shader is compiled and checked before anything fails, and the failures are
    // reported together.
    //
    // Aborting at the first bad shader reports a FLOOR, not a count — and the situation
    // where that bites is the common one: change a shared constant, break four shaders,
    // get told about one. Fix it, rebuild, get told about the next. Each rebuild teaches
    // one fact the compiler already knew in full. It also hides breakage during review:
    // a `#error` in an early-sorted shader masked a second one in a later shader, and
    // that only surfaced because a deliberate-break check reported "did not fire" when
    // it should have.
    let mut failures: Vec<String> = Vec::new();
    let comps: Vec<_> = shaders
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "comp"))
        .collect();
    let stems: Vec<String> = comps
        .iter()
        .map(|p| p.file_stem().expect("shader stem").to_string_lossy().into_owned())
        .collect();

    // PRUNE stale SPIR-V before compiling. `include_bytes!` in src/vk.rs reads
    // `{OUT_DIR}/{name}.spv` directly, and OUT_DIR is never cleaned between incremental
    // builds — so deleting a `.comp` leaves its module on disk, still linked, still
    // dispatched, with every guard below skipped for it because the loop is driven by
    // the SOURCE directory. It would only fail on a clean build.
    //
    // Exactly the bug the HIP arm already fixed above ("`ar crs` never prunes ... start
    // from a clean archive every build"); the vulkan arm had no equivalent.
    if let Ok(entries) = std::fs::read_dir(&out_dir) {
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_some_and(|x| x == "spv")
                && !p
                    .file_stem()
                    .is_some_and(|s| stems.iter().any(|k| k.as_str() == s))
            {
                let _ = std::fs::remove_file(&p);
            }
        }
    }

    for src in comps.iter() {
        let stem = src.file_stem().expect("shader stem").to_string_lossy();
        let spv = format!("{out_dir}/{stem}.spv");
        let out = Command::new(&glslc)
            .args([&format!("--target-env={VK_TARGET_ENV}"), "-O", "-Ikernels/vk"])
            .arg(format!("-DWAVE={VK_WAVE}"))
            .arg(format!("-DROWS_PER_BLOCK={VK_ROWS_PER_BLOCK}"))
            .arg(src)
            .args(["-o", &spv])
            .output()
            .expect("run glslc");
        if !out.status.success() {
            failures.push(format!(
                "{}: glslc failed\n{}",
                src.display(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
            // The later checks read the .spv, which does not exist or is stale. Skipping
            // them is not "stopping early": every OTHER shader is still checked.
            continue;
        }
        failures.extend(spirv_val(&spv));
        // ONE disassembly, shared by every SPIR-V guard. Each used to shell out for
        // itself: four spirv-dis runs per shader, 44 per build, all producing the same
        // text.
        if let Some(text) = disassemble(&spv) {
            failures.extend(no_subgroup_arithmetic(&spv, &text));
            failures.extend(no_reciprocal_rewrite(&spv, &text));
            failures.extend(no_banned_builtins(&spv, &text));
            failures.extend(no_array_parameters(&spv, &text));
            failures.extend(no_barrier_without_memory(&spv, &stem, &text));
        }
    }
    // The one transition the per-shader EXEMPT check structurally CANNOT see: a shader
    // that no longer exists is never visited, so its entry is never re-examined by
    // anything. Rename `rmsnorm.comp` and the orphaned ("rmsnorm", ...) sits there
    // silently pre-authorising any FUTURE shader that takes the name back — which would
    // then be waived from barrier checking with no diagnostic and no human ever having
    // confirmed its barriers order what they must. That is the exact silent skip this
    // mechanism exists to abolish, re-entering through deletion.
    failures.extend(BARRIER_EXEMPT.iter().filter(|(m, _)| !stems.iter().any(|s| s == m)).map(
        |(m, _)| {
            format!(
                "BARRIER_EXEMPT lists `{m}`, but kernels/vk/{m}.comp does not exist.\n  \
                 A stale entry pre-authorises a future shader of that name to skip the \
                 barrier rule silently. Remove it from BARRIER_EXEMPT in build.rs."
            )
        },
    ));
    assert!(
        failures.is_empty(),
        "\n\n{} shader problem(s) — ALL of them, not just the first:\n\n{}\n",
        failures.len(),
        failures.join("\n\n")
    );
}

/// Fail the build if a module declares the `GroupNonUniformArithmetic` capability —
/// i.e. someone reached for `subgroupAdd`/`subgroupMul`/`subgroupMin`... in a
/// reduction.
///
/// Greedy decode must be bit-reproducible, and those ops have an
/// IMPLEMENTATION-DEFINED summation order. Every reduction here is a fixed
/// `subgroupShuffleDown` halving ladder instead (kernels/vk/common.glsl::wave_sum,
/// matching common.hpp). That rule was a comment in docs/VULKAN.md; this makes it a
/// build error, because a comment does not survive sixteen kernel ports.
///
/// Skipped with a warning if `spirv-dis` is absent, like `spirv-val` above.
///
/// Returns findings rather than panicking, so the caller can report every shader's
/// problems at once.
fn no_subgroup_arithmetic(spv: &str, text: &str) -> Vec<String> {
    ["GroupNonUniformArithmetic", "GroupNonUniformClustered"]
        .iter()
        .filter(|banned| text.contains(*banned))
        .map(|banned| {
            format!(
                "{spv}: declares OpCapability {banned} — a subgroup reduce (subgroupAdd \
                 and friends) has an implementation-defined summation order and breaks \
                 greedy decode's reproducibility. Use the wave_sum shuffle ladder."
            )
        })
        .collect()
}

/// Fail the build if `-O` turned a division into a multiply by a reciprocal.
///
/// `glslc -O` rewrites `a / K` for a literal K into `a * fl(1/K)`. For K = 448 that
/// differs from a true IEEE divide by 1 ULP on 55% of inputs — measured, not feared —
/// and hipcc (`-O3`, no `-ffast-math`) keeps a real `fdiv`, so the two backends'
/// numbers silently diverged. That defeats the premise the whole port rests on, and
/// nothing else catches it: the numeric oracles' tolerances are orders of magnitude
/// looser than 1 ULP, and even the byte-exact comparisons are protected by an
/// unambiguity margin that is precisely wide enough to hide it.
///
/// The determinism story here already assumes the toolchain does not rewrite float
/// arithmetic — that is what the fixed shuffle ladder and the `subgroupAdd` ban are
/// for. This is proof it does, so the assumption becomes a check.
///
/// THE RULE: no `OpFMul` by a float constant whose mantissa is non-zero. Multiplying by
/// a power of two (0.125, 512.0, 2^-6) is exact and common in the e4m3/bf16 paths;
/// multiplying by anything else is either an optimizer-invented reciprocal or an
/// author-written scale that deserves an argument. Pass the divisor in at runtime — an
/// operand the optimizer cannot see is an operand it cannot fold.
fn no_reciprocal_rewrite(spv: &str, text: &str) -> Vec<String> {
    // Legitimate non-power-of-two constant multiplies, if one is ever justified: add
    // the exact printed value here with a comment saying why it is safe. Empty on
    // purpose — every current kernel either divides at runtime or scales by 2^n.
    const ALLOWED: &[&str] = &[];

    let mut found = Vec::new();
    // name -> printed value, from `%float_x = OpConstant %float 0.125`
    let mut consts = std::collections::HashMap::new();
    for line in text.lines() {
        if let Some((name, val)) = line.split_once(" = OpConstant %float ") {
            consts.insert(name.trim(), val.trim());
        }
    }
    for line in text.lines() {
        if !line.contains(" = OpFMul ") {
            continue;
        }
        for tok in line.split_whitespace() {
            let Some(val) = consts.get(tok) else { continue };
            if ALLOWED.contains(val) {
                continue;
            }
            let Ok(v) = val.parse::<f32>() else { continue };
            // Exact power of two (or zero): mantissa bits all clear.
            if v == 0.0 || (v.to_bits() & 0x007f_ffff) == 0 {
                continue;
            }
            found.push(format!(
                "{spv}: multiplies by the non-power-of-two constant {val}\n  {}\n  \
                 This is how `glslc -O` spells `a / {:.0}` — and a reciprocal multiply \
                 is NOT bit-identical to the divide hipcc emits, so the two backends \
                 diverge silently. Pass the divisor in as a push constant (see \
                 append_kv.comp), or if this multiply is deliberate and safe, add \
                 \"{val}\" to ALLOWED in build.rs with the reason.",
                line.trim(),
                1.0 / v
            ));
        }
    }
    found
}

/// Fail the build if a module calls a GLSL.std.450 builtin whose accuracy contract
/// differs from the HIP expression it would replace.
///
/// The ninth mechanised rule, and it covers a source of divergence the reciprocal guard
/// structurally cannot: that one inspects an `OpFMul` constant, and these are FUNCTION
/// SUBSTITUTIONS with no constant to look at.
///
/// The risk is not the optimiser here — it is the porter reaching for the idiomatic
/// spelling. `inversesqrt(z)` is what a GLSL author writes, and what a reviewer would
/// "tidy" `1.0/sqrt(z)` into; Vulkan specifies it to 2 ULP as a SINGLE operation, where
/// HIP computes a correctly-rounded `sqrtf` and then a correctly-rounded divide. Two
/// different numbers, from a change that reads as cleanup.
///
/// A denylist of one entry that fires exactly is worth more than no guard. It grows as
/// more are identified; each needs the divergence it prevents named, or the next person
/// cannot judge whether an exemption is safe.
fn no_banned_builtins(spv: &str, text: &str) -> Vec<String> {
    /// `(instruction, what to write instead, why)`.
    const BANNED: &[(&str, &str, &str)] = &[(
        "InverseSqrt",
        "1.0 / sqrt(z)",
        "Vulkan specifies inversesqrt to 2 ULP as one operation; HIP does a \
         correctly-rounded sqrt then a correctly-rounded divide, so the results differ",
    )];

    let mut found = Vec::new();
    for line in text.lines() {
        // Match the operand position exactly: `%n = OpExtInst %type %set <Instr> ...`.
        // A substring search would also hit a name or a debug string.
        let Some((_, rest)) = line.split_once(" = OpExtInst ") else { continue };
        let Some(instr) = rest.split_whitespace().nth(2) else { continue };
        for (banned, instead, why) in BANNED {
            if instr == *banned {
                found.push(format!(
                    "{spv}: calls GLSL.std.450 `{banned}`\n  {}\n  Use `{instead}` \
                     instead — {why}. If this call is deliberate and the divergence is \
                     acceptable, remove it from BANNED in build.rs and say why.",
                    line.trim()
                ));
            }
        }
    }
    found
}

/// Fail the build if a shader copies a WHOLE ARRAY in one load.
///
/// The eleventh mechanised rule. GLSL passes parameters by VALUE-RESULT — copy-in/
/// copy-out — including arrays, and including `shared` ones. `void
/// e4m3_lut_build(inout float lut[256], uint tid)` therefore compiled to a
/// per-invocation 1 KB private copy of the whole shared array: 256 threads each copied
/// all 256 entries in, wrote one, and copied all 256 back, so every thread clobbered the
/// others with the uninitialised values it had loaded. The e4m3 table was noise and
/// every fp8 weight decoded to noise (err = 8.6e37).
///
/// It got past ALL of: a clean compile, spirv-val, the capability scan, the reciprocal
/// guard, the InverseSqrt guard, and GPU-assisted validation — the last because the
/// reads were IN BOUNDS. The addresses were right and the contents were garbage. Only
/// the numeric oracle caught it, and only because rule ten forced that oracle to exist.
///
/// WHY THIS SIGNATURE AND NOT `OpFunctionParameter` OF ARRAY TYPE. That is the obvious
/// spelling and it does not work: `glslc -O` INLINES the helper, so the shipped module
/// contains no function parameter at all. Measured — the buggy build has zero
/// `OpFunctionParameter` and two `OpLoad` of array type. A guard has to match what
/// survives optimisation, not what the source looks like.
///
/// WHY IT DOES NOT BLOCK A LOCAL ACCUMULATOR. `mla_latent_attend` needs
/// `float acc[MLA_ACC_REGS]`, and indexing an array element compiles to `OpAccessChain`
/// then `OpLoad %float` — never a whole-array load. Verified across every kernel here:
/// `rope_interleave` carries two local arrays and has zero whole-array loads; only the
/// buggy `gemv_fp8` had any.
///
/// Whether such an array lands in registers or spills to scratch is a DIFFERENT question
/// with a different instrument — `ScratchSize` from the compiler's resource usage, not
/// anything visible in SPIR-V. One rule attempting both would block legitimate code and
/// still miss the spill.
///
/// Write a macro instead, so the caller's variable is written directly.
fn no_array_parameters(spv: &str, text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| {
            l.split_once(" = OpLoad ")
                .is_some_and(|(_, rest)| rest.trim_start().starts_with("%_arr_"))
        })
        .map(|l| {
            format!(
                "{spv}: copies a WHOLE ARRAY in one load\n  {}\n  GLSL copies array \
                 arguments IN AND OUT per invocation, so passing a `shared` array to a \
                 function gives every thread a private copy and the writes are lost. Use \
                 a macro so the caller's variable is written directly (see \
                 E4M3_LUT_BUILD in kernels/vk/common.glsl). Indexing a local array is \
                 fine — that compiles to OpAccessChain + a scalar load, not this.",
                l.trim()
            )
        })
        .collect()
}

/// Fail the build if a shader has a barrier that orders NOTHING.
///
/// The twelfth rule. HIP's `__syncthreads()` orders LDS *and* global memory. GLSL's bare
/// `barrier()` compiles to `OpControlBarrier Workgroup Workgroup
/// AcquireRelease|WorkgroupMemory` — semantics 0x108, which orders SHARED memory only,
/// with the UniformMemory bit 0x40 absent. So the one-to-one transliteration silently
/// drops half of what it is porting, and this is the only member of the
/// same-spelling-different-semantics class so far where the GLSL is WEAKER than the HIP
/// rather than merely different.
///
/// `rope_interleave` shipped this way. Its barrier separates reads of `v[2j], v[2j+1]`
/// from writes to `v[j], v[half+j]` over the SAME buffer-reference range, and carried no
/// buffer semantics — a pure execution barrier with respect to the memory it existed to
/// order. Nothing could see it: spirv-val is silent, GPU-AV checks addresses, and sync
/// validation on this stack covers only transfer-to-transfer. It passed on today's RADV
/// and a driver or compiler change turns it into a subtly wrong rotation, not a fault.
///
/// SIGNATURE, and it is narrow on purpose: a module that has a control barrier, has NO
/// Workgroup-storage variables, and has NO memory barrier. With nothing in shared
/// storage, a WorkgroupMemory-only barrier is definitionally ordering nothing, so the
/// only thing it can have been written for is buffer traffic — and that is exactly the
/// bug. It does NOT try to decide whether a barrier in a module that also uses shared
/// memory needs buffer semantics; that is a dataflow question SPIR-V cannot answer
/// cheaply, and a guess there would false-positive on correct code.
///
/// VERIFIED BOTH WAYS before adoption, per the lesson from rule 11: it is quiet on all
/// four kernels with legitimate shared-memory barriers (append_kv, argmax_reduce,
/// gemv_fp8, rmsnorm — each has Workgroup variables), and it fires on `rope_interleave`
/// with the fix reverted.
///
/// Fix: `memoryBarrierBuffer(); barrier();`
///
/// ## The skip set is PINNED, because a silent skip is how this rule dies
///
/// The `has_shared` skip above is deliberate and correct — but it is also INVISIBLE, and
/// that is a defect independent of the rule being right. A module that gains a `shared`
/// variable drops out of barrier checking with no diagnostic, and the change that does it
/// reads as ordinary work: add an LDS tile, and every barrier in that kernel silently
/// stops being judged.
///
/// The trigger is narrower than "a module gains a `shared` declaration", and the
/// difference was MEASURED after a first version of this comment got it wrong. An unused
/// Workgroup variable is dead-code-eliminated: a global `shared float lut[256]` that a
/// module never reads leaves ZERO `OpVariable %_ptr_Workgroup` under `-O` (1 without it,
/// and this build ships `-O`). So hoisting a shared LUT into `common.glsl` would NOT
/// disarm the rule across every includer, as claimed here originally — only in the
/// modules that actually read it. The real trigger is **a barrier plus a Workgroup
/// variable the module genuinely uses**.
///
/// Left standing because a guard resting on a false rationale is this repo's own
/// documented trap: the code was right, the argument was wrong, and the argument is what
/// a maintainer reads when deciding whether the guard still earns its place.
///
/// Enforced in both directions so the list cannot rot into describing a past tree, plus a
/// whole-set check in `vulkan()` for the deletion case neither direction can see. An
/// entry costs an argument in a reviewable diff, matching `ALLOWED` in
/// `tests/kernel_coverage.rs` and `LOCKED` in `tests/glsl_numerics.rs`.
///
/// **Read the list as a coverage statement, because it is a blunt one:** of eleven
/// shaders, six have no barrier at all, FOUR are exempt here, and exactly ONE —
/// `rope_interleave`, the kernel whose bug created this rule — is actually judged. The
/// anti-vacuity principle this repo already applies ("prove the check looked at
/// something") extended one step: also say what it did NOT look at.
///
/// A `cargo:warning` on every build was considered and rejected. Cargo hides plain build
/// script output unless `-vv`, so it would land in a sink; and a warning that fires on
/// every healthy build gets filtered out by the reader, which is this repo's stated
/// reason for not counting PERFORMANCE validation messages. A build ERROR the moment the
/// set CHANGES is the event worth interrupting someone for.
fn no_barrier_without_memory(spv: &str, stem: &str, text: &str) -> Vec<String> {
    let has_barrier = text.contains("OpControlBarrier");
    let has_shared = text.contains("OpVariable %_ptr_Workgroup");
    let has_memory_barrier = text.contains("OpMemoryBarrier");

    // BOOKKEEPING AND THE REAL RULE BOTH REPORT, and the fall-through is the point.
    // Returning on the EXEMPT mismatch substituted one message for the other in the case
    // that matters most: convert `rmsnorm`'s halving tree to the `wave_sum` shuffle
    // ladder — which the capability rule above actively encourages — and it loses its
    // Workgroup variable while keeping a bare `barrier()` over buffer traffic. That is
    // bit-for-bit the `rope_interleave` signature, but `listed && !skipped` fired first
    // and the build said "remove a stale const entry", naming bookkeeping where the news
    // was a live ordering bug. Same first-failure-floor this file rejects at build scope,
    // reintroduced per shader.
    let mut found = Vec::new();
    let skipped = has_barrier && has_shared;
    let listed = BARRIER_EXEMPT.iter().any(|(m, _)| *m == stem);
    if skipped != listed {
        found.push(if skipped {
            format!(
                "{spv}: has a barrier AND shared memory, so the barrier rule SKIPS it — \
                 but `{stem}` is not in BARRIER_EXEMPT.\n  This kernel's barriers are no \
                 longer checked for the missing buffer semantics that `rope_interleave` \
                 shipped, and nothing would have said so. Add (\"{stem}\", \"<what the LDS \
                 is>\") to BARRIER_EXEMPT in build.rs, having first confirmed by hand that \
                 every barrier in it orders the memory it needs to."
            )
        } else {
            format!(
                "{spv}: `{stem}` is in BARRIER_EXEMPT, but the rule is NOT skipping it \
                 (barrier={has_barrier}, shared={has_shared}).\n  The exemption no longer \
                 describes this shader, and a stale entry reads as coverage that was \
                 waived. Remove it from BARRIER_EXEMPT in build.rs — and read any finding \
                 below this one first, it is the substantive one."
            )
        });
    }

    if !has_barrier || has_shared || has_memory_barrier {
        return found;
    }
    found.push(format!(
        "{spv}: has a barrier that orders NOTHING\n  OpControlBarrier with \
         WorkgroupMemory-only semantics, in a module with no Workgroup storage.\n  \
         GLSL's bare `barrier()` orders SHARED memory only, unlike HIP's \
         `__syncthreads()` which also orders global. If this barrier is protecting \
         buffer traffic, write `memoryBarrierBuffer(); barrier();` — otherwise the \
         barrier is dead and should be deleted."
    ));
    found
}

/// `spirv-dis` output, or `None` with a warning if the tool is absent — same optional
/// treatment as `spirv-val`, since it ships in a different package from glslc.
fn disassemble(spv: &str) -> Option<String> {
    match Command::new("spirv-dis").arg("--no-color").arg(spv).output() {
        Ok(o) if o.status.success() => Some(String::from_utf8_lossy(&o.stdout).into_owned()),
        Ok(_) => panic!("spirv-dis failed on {spv}"),
        Err(e) => {
            println!("cargo:warning=spirv-dis not run on {spv} ({e}); SPIR-V unchecked");
            None
        }
    }
}

/// Validate a compiled module with `spirv-val`, **under Vulkan's rules**.
///
/// This is STATIC module validation — it catches malformed SPIR-V, bad decorations,
/// and capability/extension mismatches. It does NOT see synchronisation, descriptor,
/// or buffer-device-address misuse; only the VK_LAYER_KHRONOS_validation runtime layer
/// does, and that layer is a separate install (see docs/VULKAN.md "Risks").
///
/// `--target-env` IS LOAD-BEARING, and its absence was a real hole rather than an
/// untidiness. Without it `spirv-val` applies only the UNIVERSAL SPIR-V 1.6 rules, and
/// the entire `VUID-StandaloneSpirv-*` family — every requirement Vulkan adds on top of
/// the bare specification — goes unchecked. Measured, not inferred: strip the `Block`
/// decoration off a push-constant struct and the old invocation exits **0** on the
/// identical module, while `--target-env vulkan1.3` reports
/// `[VUID-StandaloneSpirv-PushConstant-06675] PushConstant id '8' is missing Block
/// decoration`.
///
/// Note the paragraph above it: this comment claimed "bad decorations" as covered, and a
/// missing `Block` decoration is exactly the example it could not see.
///
/// A missing `spirv-val` warns rather than fails: it ships in a different package from
/// glslc, and a box that can build shaders should not be blocked on the checker.
fn spirv_val(spv: &str) -> Vec<String> {
    match Command::new("spirv-val")
        .args(["--target-env", VK_TARGET_ENV])
        .arg(spv)
        .output()
    {
        Ok(o) if o.status.success() => Vec::new(),
        // BOTH streams. Passing a flag introduced a failure class that did not exist when
        // this took no arguments: spirv-val sends VALIDATION errors to stderr but its
        // ARGUMENT-PARSING diagnostic — the whole usage block — to stdout, measured at
        // 4046 bytes on stdout and 0 on stderr for the `--target-env=X` spelling. Reading
        // stderr alone would fail all eleven shaders with an empty body, pointing at the
        // modules while the invocation was at fault.
        Ok(o) => vec![format!(
            "{spv}: spirv-val rejected the module\n{}\n{}",
            String::from_utf8_lossy(&o.stderr).trim(),
            String::from_utf8_lossy(&o.stdout).trim()
        )],
        Err(e) => {
            println!("cargo:warning=spirv-val not run on {spv} ({e}); SPIR-V unchecked");
            Vec::new()
        }
    }
}
