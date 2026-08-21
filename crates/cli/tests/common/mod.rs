//! Shared helpers for the workspace meta-gates. Grows only when a second test needs the
//! same helper — the old tree's `tests/common/mod.rs` reached 1050 lines one forced
//! factoring at a time, and each move was jscpd telling it a copy existed.

// Compiled into EACH meta-gate binary; none uses every helper (matrix.rs needs only
// repo_root) — the engine tests' common carries the same argument.
#![allow(dead_code)]

/// Every file under `root` with extension `ext`, recursively. Unsorted.
///
/// WALK, do not list files. The old tree's registry checks each grew their own copy of
/// this, and the hand-maintained path list one of them replaced named five files — moving
/// three of them into subsystem folders silently emptied it, after which the registry
/// reported every invariant as untested. A coverage check keyed on a remembered list fails
/// in the direction that looks like a real regression, which costs more than the walk.
pub fn walk(root: &std::path::Path, ext: &str) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        // Split each directory's entries in one pass instead of branching per entry inside the
        // loop: `partition` keeps the descend/collect decision to a single flat expression, and
        // the `filter` drops the third case (a file with the wrong extension) before it.
        // The double `flatten` swallows an unreadable directory — `walk` feeds coverage checks
        // that must not go red because a path vanished under them, and a directory that cannot
        // be read contributes nothing either way.
        let (subdirs, matches): (Vec<_>, Vec<_>) = std::fs::read_dir(&dir)
            .into_iter()
            .flatten()
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir() || p.extension().is_some_and(|x| x == ext))
            .partition(|p| p.is_dir());
        stack.extend(subdirs);
        out.extend(matches);
    }
    out
}

/// The workspace root: two levels above this crate's manifest.
pub fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(std::path::Path::to_path_buf)
        .unwrap_or_default()
}

// ── converter-gate fixtures ─────────────────────────────────────────────────────────────
//
// The three below arrived here on 2026-08-16, when `glimmer_convert.rs` became the second
// converter gate and `build.rs`'s jscpd reported all three as clones of `glm_convert.rs`'s
// copies. That is this file's stated growth rule working exactly as written — it grows when a
// second test needs the same helper, and each move is the duplication gate saying a copy
// exists. They are fixture plumbing, shared by construction rather than by coincidence: a
// converter gate needs deterministic weights, their bf16 encoding, and a scratch directory.

/// Deterministic pseudo-weights: a cheap hash of (name, index) — no RNG dependency, and values
/// in a plausible ±0.1 range so fp8 block scales stay finite.
///
/// **Keyed on the NAME**, which is what makes a byte comparison mean something: a converter
/// that wrote the right tensor's bytes under the wrong name, or the wrong tensor's under the
/// right one, fails rather than passing on identical content.
pub fn weights(name: &str, n: usize) -> Vec<f32> {
    let seed = rivoli_core::hash::fnv1a(name.as_bytes());
    (0..n)
        .map(|i| {
            let h = rivoli_core::hash::fnv1a(&(seed ^ i as u64).to_le_bytes());
            ((h % 2001) as f32 / 1000.0 - 1.0) * 0.1
        })
        .collect()
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| rivoli_core::num::f32_to_bf16(x).to_le_bytes())
        .collect()
}

/// A scratch root under `$TMPDIR`, **removed first**.
///
/// The `remove_dir_all` is load-bearing rather than tidiness: a stale `out1`/`out2` from a
/// killed run would satisfy a determinism compare vacuously, and a stale artifact directory
/// would satisfy a refusal test's "the output must not exist" in reverse.
///
/// `tag` carries the caller's own model and arm (`"glm-convert-rt"`), so one helper serves every
/// gate without the names colliding. Shaped unlike `ppl.rs`'s `tmp()` on purpose — jscpd matched
/// those two temp-dir helpers at 27 tokens once already.
///
/// `#[expect(clippy::expect_used)]` rather than a file-level allow: this module is compiled into
/// the meta-gates too, and a scratch directory that cannot be created is a broken harness that
/// should die loudly rather than a test failure to report.
#[expect(
    clippy::expect_used,
    reason = "a harness that cannot make a temp dir must die loudly"
)]
pub fn scratch(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("rivoli-{tag}-{}", std::process::id()));
    assert!(!d.exists() || std::fs::remove_dir_all(&d).is_ok());
    std::fs::create_dir_all(&d).expect("create scratch dir");
    d
}

// ── converter-gate plumbing ──────────────────────────────────────────────────────────────
//
// The five below arrived on 2026-08-16, when `v4_convert.rs` became the THIRD converter gate
// and `build.rs`'s jscpd reported each of them as a clone against `glimmer_convert.rs`. Same
// rule as the three above, and the same trigger: this file grows when a second test needs the
// same helper, and each move is the duplication gate saying a copy exists.
//
// They are plumbing, not judgement. Every one of them writes bytes or reads bytes back; not one
// of them decides what a converter should have done, which stays in each gate where its
// argument is.

/// One tensor of a synthetic checkpoint: name, dtype, shape, bytes.
///
/// **Carries its DTYPE**, where the Glimmer gate's local type did not — that model's fixture is
/// bf16 throughout, V4's is five dtypes, and a shared type that assumed one would make the V4
/// gate spell its own writer again. The name is first because it is the key every byte
/// comparison is made under.
pub type Tensor = (String, rivoli_artifact::format::Dtype, Vec<usize>, Vec<u8>);

/// Write one `*.safetensors` shard through the same `SafeWriter` the converters use.
///
/// Through the real writer rather than by hand: a fixture built by a second serializer would
/// test that serializer, and the header layout is exactly the thing a converter round-trip must
/// not have to re-derive.
pub fn write_shard(path: &std::path::Path, tensors: &[Tensor]) {
    let mut w = rivoli_artifact::format::SafeWriter::new();
    for (name, dtype, shape, bytes) in tensors {
        w.add(name.clone(), *dtype, shape.clone(), &bytes[..]);
    }
    // `to_string_lossy` rather than `to_str().unwrap()`: this module is compiled into the
    // META-GATES too, which carry no file-level `unwrap` allow, and a scratch path is ASCII by
    // construction so the lossy branch is unreachable.
    w.write(&path.to_string_lossy())
        .expect("write a fixture shard");
}

/// `model.safetensors.index.json` from `(tensor, shard)` pairs.
///
/// **Written FROM the tensor list rather than alongside it**, and re-written whenever that list
/// changes, because the index is what `open_indexed` selects shards by — a refusal test that
/// dropped a tensor from the shard and left it in the index would be testing a truncated-file
/// error instead of the guard it meant to.
pub fn write_weight_map(dir: &std::path::Path, entries: &[(String, &str)]) {
    let map: serde_json::Map<String, serde_json::Value> = entries
        .iter()
        .map(|(n, shard)| (n.clone(), serde_json::json!(shard)))
        .collect();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::json!({ "weight_map": map }).to_string(),
    )
    .expect("write the fixture index");
}

// ── converter-gate plumbing, second growth ───────────────────────────────────────────────
//
// The five below arrived on 2026-08-16, when `k3_convert.rs` became the FOURTH converter
// gate and `build.rs`'s jscpd reported each of them as a clone against `v4_convert.rs` —
// eleven regions in one report, all fixture plumbing. Same growth rule, same trigger.

/// Little-endian f32 bytes — the encoding of an F32 fixture tensor.
pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Deterministic filler bytes for a tensor a converter copies and never interprets.
///
/// **Keyed on the NAME**, like [`weights`], which is what makes a byte comparison mean
/// something: a converter that wrote the right tensor's bytes under the wrong name, or the
/// wrong tensor's under the right one, fails rather than passing on identical content.
pub fn opaque_bytes(name: &str, n: usize) -> Vec<u8> {
    weights(name, n)
        .iter()
        .map(|x| ((x * 1000.0) as i32).unsigned_abs() as u8)
        .collect()
}

/// e8m0 exponent bytes, kept inside the range the shipped V4 checkpoint actually uses.
///
/// Measured on the real 43-layer set: 9 distinct codes, all in `0x76..=0x7e`, with **zero**
/// `0x00` and zero `0xff`. `0xff` is the format's NaN and `F4Expert::spans` refuses it —
/// which both models' NaN-refusal arms rely on, so a good fixture must not contain one by
/// accident. K3 declares its scale bytes plain `U8`, so there the range is a property of
/// the VALUES, not of the dtype.
pub fn e8m0_bytes(name: &str, n: usize) -> Vec<u8> {
    opaque_bytes(name, n)
        .iter()
        .map(|b| 0x76 + (b % 9))
        .collect()
}

/// One fixture tensor with the byte policy every converter gate shares: bf16 and f32 get
/// deterministic VALUES, e8m0 scales stay in the measured range, and anything else is
/// opaque filler. A gate with a model-specific dtype (V4's `tid2eid` is `I64` with values
/// bounded by its expert count) intercepts that dtype and delegates the rest here.
pub fn tensor(name: &str, dtype: rivoli_artifact::format::Dtype, shape: Vec<usize>) -> Tensor {
    use rivoli_artifact::format::Dtype;
    let n: usize = shape.iter().product();
    let bytes = match dtype {
        Dtype::Bf16 => bf16_bytes(&weights(name, n)),
        Dtype::F32 => f32_bytes(&weights(name, n)),
        Dtype::F8E8M0 => e8m0_bytes(name, n),
        // F8E4M3, I8, U8 — bytes the converters copy and never decode.
        _ => opaque_bytes(name, n),
    };
    (name.to_string(), dtype, shape, bytes)
}

/// The shard and its index, always written together — the index is what `open_indexed`
/// selects shards by, so a refusal arm that dropped a tensor from one and left it in the
/// other would be testing a truncated-file error instead of the guard it meant to.
pub fn write_shard_and_index(src: &std::path::Path, shard: &str, tensors: &[Tensor]) {
    write_shard(&src.join(shard), tensors);
    let entries: Vec<(String, &str)> = tensors
        .iter()
        .map(|(n, _, _, _)| (n.clone(), shard))
        .collect();
    write_weight_map(src, &entries);
}

/// One converter binary under test: its `CARGO_BIN_EXE_*` path and the name its log lines
/// and refusals speak as. As free functions each gate had its own copy of the
/// run/convert/refuse triple, differing only in these two strings, and jscpd reported all
/// of them.
pub struct ConvertBin {
    pub exe: &'static str,
    pub tool: &'static str,
}

impl ConvertBin {
    /// This binary aimed at one `src`/`out` pair — the pair is fixed for a whole test arm
    /// while the flags vary per invocation, which is the abstraction the flat
    /// five-argument `refuses(src, out, extra, want)` was missing (CodeScene priced that
    /// form at arity 5, and it was right: the pair and the flags are different KINDS of
    /// argument).
    pub fn at<'a>(&'a self, src: &'a std::path::Path, out: &'a std::path::Path) -> ConvertRun<'a> {
        ConvertRun {
            bin: self,
            src,
            out,
        }
    }
}

/// [`ConvertBin::at`]'s result: one converter, one directory pair, many invocations.
pub struct ConvertRun<'a> {
    bin: &'a ConvertBin,
    src: &'a std::path::Path,
    out: &'a std::path::Path,
}

impl ConvertRun<'_> {
    /// The two positional dirs plus whatever flags this arm is about.
    pub fn invoke(&self, extra: &[&str]) -> std::process::Output {
        let mut cmd = std::process::Command::new(self.bin.exe);
        cmd.arg(self.src).arg(self.out).args(extra);
        cmd.output()
            .unwrap_or_else(|e| panic!("running {} failed outright: {e}", self.bin.tool))
    }

    /// A run that must SUCCEED; hands back stderr for the log assertions.
    #[track_caller]
    pub fn convert(&self, extra: &[&str]) -> String {
        expect_success(&self.invoke(extra), self.bin.tool)
    }

    /// A run that must REFUSE, for the reason named — a refusal firing for an unrelated
    /// cause must fail, which is `expect_refusal`'s contract.
    #[track_caller]
    pub fn refuses(&self, extra: &[&str], want: &str) {
        expect_refusal(&self.invoke(extra), want);
    }
}

/// A scratch root plus the `src`/`out` pair every converter arm starts from — factored
/// because the three-line spelling recurred at every `#[test]` head of the converter gate
/// files, and the runs between the differing tag literals were themselves over jscpd's
/// floor.
pub fn scratch_src_out(tag: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let root = scratch(tag);
    let (src, out) = (root.join("src"), root.join("out"));
    (root, src, out)
}

/// Little-endian f32 bytes as values — what an artifact's widened tensor decodes to.
pub fn as_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Little-endian bf16 bytes as values — what the SOURCE tensor a widened one came from holds.
///
/// The pair with [`as_f32`] is what makes a widening assertion mean something: a byte-length
/// check alone passes on a zeroed tensor, and `add_widened` is where a converter can be
/// plausibly wrong.
pub fn as_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| rivoli_core::num::bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect()
}

/// Assert a converter run REFUSED, and refused for the reason named.
///
/// The `want` check is the point: a refusal test that only asserts non-zero exit passes when the
/// binary fails for an unrelated reason, which is how a guard gets deleted without a red test.
#[track_caller]
pub fn expect_refusal(o: &std::process::Output, want: &str) {
    let err = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(
        !o.status.success() && err.contains(want),
        "expected a refusal naming {want:?}, got status {:?}:\n{err}",
        o.status.code()
    );
}

/// Assert a converter run SUCCEEDED, and hand back its stderr for the log assertions.
///
/// The log is returned rather than discarded because a converter's counted lines ("3 vision
/// tensors skipped", "experts=4 layers 0..4") are what turn an exclusion from an assumption
/// into an observation.
#[track_caller]
pub fn expect_success(o: &std::process::Output, what: &str) -> String {
    let log = String::from_utf8_lossy(&o.stderr).to_string();
    assert!(o.status.success(), "{what} failed:\n{log}");
    log
}

/// Write the fixture's `config.json`, pretty — the file every converter's `load_config` reads.
///
/// Pretty rather than compact so a failing fixture can be diffed by eye, and shared because both
/// converter gates do exactly this and jscpd reported the pair.
pub fn write_config(src: &std::path::Path, config: &serde_json::Value) {
    std::fs::create_dir_all(src).expect("create the fixture directory");
    std::fs::write(
        src.join("config.json"),
        serde_json::to_vec_pretty(config).expect("serialize the fixture config"),
    )
    .expect("write the fixture config");
}

/// Write the fixture's aux sidecars — the files a converter COPIES into the artifact rather
/// than reads, as `(name, body)` pairs.
///
/// Shared for `write_config`'s reason: jscpd reported the identical `for (name, body) in [...]
/// { fs::write }` loop the moment K3's list stopped being a different length from V4's
/// (2026-08-16, when K3's was corrected against the real checkpoint). **THREE gates carried
/// that loop and all three now call this** — the gate only ever reports a pair, so migrating
/// the two it named would have left the third to re-create the clone against whichever
/// converter gate is written next.
///
/// The LIST stays at each call site and the walk lives here, which is the split that matters:
/// which sidecars a checkpoint ships is the fact each gate asserts, and a fixture that invents
/// one the real source lacks is the defect this correction came from.
#[track_caller]
pub fn write_aux(src: &std::path::Path, files: &[(&str, &str)]) {
    for (name, body) in files {
        std::fs::write(src.join(name), body).expect("write the fixture aux file");
    }
}

/// The artifact's bytes for `name`, with its shape already confronted with the source's.
///
/// The shape check is here rather than at each caller because that is the half a comparison
/// cannot make: two tensors of the same LENGTH and different shapes compare byte-equal.
#[track_caller]
fn typed_with_shape<'a>(
    art: &'a rivoli_artifact::format::Safetensors,
    name: &str,
    dtype: rivoli_artifact::format::Dtype,
    shape: &[usize],
) -> &'a [u8] {
    let (got, got_shape) = art
        .typed(name, dtype)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    assert_eq!(got_shape, shape, "{name} shape");
    got
}

/// One artifact tensor that must be the source's bytes, unchanged, under the same dtype.
///
/// Takes the whole [`Tensor`] rather than its four parts: five arguments is over the
/// code-health gate's threshold, and the tuple already IS the value — a call that passed a
/// dtype other than the source tensor's own would be asking a different question than "did this
/// arrive unchanged".
#[track_caller]
pub fn assert_verbatim(art: &rivoli_artifact::format::Safetensors, t: &Tensor) {
    let (name, dtype, shape, bytes) = t;
    let got = typed_with_shape(art, name, *dtype, shape);
    assert_eq!(got, &bytes[..], "{name} is not byte-identical");
}

/// One artifact tensor that must be the source's bf16 tensor widened to f32 — **by VALUE**.
///
/// A byte-length check alone passes on a zeroed tensor, and `SafeWriter::add_widened` is where a
/// converter can be plausibly wrong, so the comparison is `as_f32(artifact)` against
/// `as_bf16(source)` rather than anything about sizes.
#[track_caller]
pub fn assert_widened(art: &rivoli_artifact::format::Safetensors, t: &Tensor) {
    let (name, _, shape, bytes) = t;
    let got = typed_with_shape(art, name, rivoli_artifact::format::Dtype::F32, shape);
    assert_eq!(
        as_f32(got),
        as_bf16(bytes),
        "{name} widened to the wrong values"
    );
}

/// Remove a [`scratch`] root at the end of a passing test.
///
/// Best-effort and infallible on purpose: a failed test should LEAVE its fixture behind to be
/// looked at, which is what happens automatically since the panic skips this call. Named rather
/// than spelled inline because the three-line `let _ = remove_dir_all(&root);` tail was itself
/// what jscpd matched between two gates (2026-08-16).
pub fn clean(root: &std::path::Path) {
    let _ = std::fs::remove_dir_all(root);
}
