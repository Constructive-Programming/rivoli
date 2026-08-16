//! `manifest.json`: the artifact's self-description, and the three things that read it.
//!
//! [`FormatMeta`] is the build's compiled-in parameters, refused at load if the artifact
//! disagrees; [`I4Source`] is provenance for an expert set that is otherwise
//! byte-indistinguishable on disk from one derived a different way; [`finish_artifact`] is
//! what publishes the file and the tokenizer beside it. Three sections of one document, and
//! the two atomic writers that publish it — [`write_json_atomic`] fsyncs where
//! [`super::layer`] deliberately does not, because a torn manifest bricks an artifact whose
//! expert set alone is ~365 GB while a torn layer file is regenerable.
//!
//! The `f4_source` section is NOT here: it is an input to opening the expert set rather than
//! a fact about the manifest, so it lives with its consumer in [`super::set`].

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

use crate::quant::{VQ_DIM, VQ_GROUP, VQ_INDEX_BITS, VQ_K};

/// The `format` section of `manifest.json` — everything the loader needs beyond
/// the HF config fields (which `ModelConfig` reads from the same file).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatMeta {
    pub version: u32,
    pub vq_dim: usize,
    pub vq_k: usize,
    pub vq_index_bits: usize,
    pub vq_group: usize,
    /// fp8 `weight_scale_inv` tile size (128 for the GLM-5.2 checkpoint).
    pub fp8_block: usize,
}

impl FormatMeta {
    pub const VERSION: u32 = 1;

    /// The current build's parameters — what the converter stamps into the manifest.
    pub fn current(fp8_block: usize) -> Self {
        Self {
            version: Self::VERSION,
            vq_dim: VQ_DIM,
            vq_k: VQ_K,
            vq_index_bits: VQ_INDEX_BITS,
            vq_group: VQ_GROUP,
            fp8_block,
        }
    }

    /// The checkpoint's own `config.json`, with this build's `format` section stamped in —
    /// the manifest a converter publishes, before any per-tool provenance is added.
    ///
    /// **Shared because jscpd said so, which is this tree's rule for when a helper is shared.**
    /// `convert_v4` arrived at M8 as the second converter that READS `config.json` (rather than
    /// re-serializing an already-parsed value, which is what `convert` does), and `build.rs`
    /// reported this exact pair of statements as a clone against `convert_glimmer` on the first
    /// compile. The contract it names is the one all three converters honour: *the manifest is
    /// the source config plus `format`*, so `<Arch>Config::load` reads the artifact exactly as
    /// it read the source and `arch::from_manifest_str` finds the same discriminant.
    ///
    /// `FormatMeta::current` stamps the compiled-in VQ parameters even on an artifact that has
    /// no VQ tensors. That is inert rather than a lie: [`Self::load`] compares them against the
    /// same constants, so they always agree, and a "nullable VQ section" turned out to be work
    /// nothing needed. `fp8_block` likewise describes a format a bf16 artifact does not use yet.
    pub fn manifest_from_config(src_dir: &str, fp8_block: usize) -> Result<serde_json::Value> {
        let path = format!("{src_dir}/config.json");
        let mut manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("read {path}"))?)
                .with_context(|| format!("parse {path}"))?;
        manifest["format"] = serde_json::to_value(Self::current(fp8_block))?;
        Ok(manifest)
    }

    /// Read `<dir>/manifest.json`'s `format` section and check it matches this build
    /// (VQ params are compiled into the kernels, so a mismatch is unrunnable).
    pub fn load(dir: &str) -> Result<Self> {
        let v: serde_json::Value =
            serde_json::from_slice(&std::fs::read(format!("{dir}/manifest.json"))?)
                .with_context(|| format!("parse {dir}/manifest.json"))?;
        let m: FormatMeta = serde_json::from_value(v["format"].clone())
            .context("manifest.json missing/invalid `format` section")?;
        ensure!(
            m.version == Self::VERSION,
            "artifact format v{} != build v{}",
            m.version,
            Self::VERSION
        );
        ensure!(
            m.vq_dim == VQ_DIM
                && m.vq_k == VQ_K
                && m.vq_index_bits == VQ_INDEX_BITS
                && m.vq_group == VQ_GROUP,
            "artifact VQ params differ from the compiled-in kernel params"
        );
        // The fp8 GEMV kernels index the block scale with a SHIFT (`blk_shift`), so a
        // non-power-of-two tile is unrunnable. The kernel launchers reject it too (arg
        // guard 1003); catching it here turns a mid-decode HIP error into a startup
        // message that names the offending value.
        ensure!(
            m.fp8_block > 0 && m.fp8_block.is_power_of_two(),
            "artifact fp8_block ({}) must be a power of two",
            m.fp8_block
        );
        Ok(m)
    }
}

/// The two directories an artifact finish reads and writes — one value because the pair
/// travels together and swapping them is a plausible, silent call-site error.
pub struct ArtifactDirs<'a> {
    pub out: &'a str,
    pub src: &'a str,
}

/// Refuse a converter's `aux` list against the SOURCE before any weight is read.
///
/// [`finish_artifact`] already refuses a missing aux file — this buys **when**. That function
/// is the last step of a convert, so without this the refusal lands after the whole weight
/// write, on a checkpoint that could have been rejected in milliseconds.
///
/// **That is not hypothetical, and it is why this moved out of `convert_glimmer`.** Kimi-K3's
/// list named `tokenizer.json`, which that checkpoint does not ship (it is tiktoken —
/// `docs/investigations/k3-first-checkpoint.md` §3). The bad name survived every gate because
/// the fixture wrote the file it named, and the real 1.3 TiB run would have refused at its
/// final step. `convert_glimmer` had carried this guard since 2026-08-16 and `convert_k3` had
/// not; a guard that lives in one converter protects one converter.
///
/// **Pass the SAME slice you pass [`finish_artifact`]** — a pre-check over a hand-maintained
/// subset is a second list to keep in step, and the drift it admits is precisely a name that
/// is checked but not copied, or copied but not checked.
pub fn require_aux(src_dir: &str, aux: &[&str]) -> Result<()> {
    for name in aux {
        anyhow::ensure!(
            std::path::Path::new(src_dir).join(name).is_file(),
            "{name} is missing from {src_dir}. The artifact is not self-contained without it, \
             and finish_artifact would refuse it only at the END of the convert"
        );
    }
    Ok(())
}

/// Write `<out_dir>/manifest.json` and copy `aux` (tokenizer and friends) beside it, so the
/// artifact is self-contained. The last step of every converter.
///
/// > **CORRECTED 2026-08-16.** This doc block said "a missing aux file is a WARNING rather
/// > than an error … failing a multi-hour convert on its absence at the very end would be the
/// > worse trade", and it was wrong twice over. The code below `?`s the copy and `ensure!`s
/// > the result — absence has been a hard error in this tree since the shared function was
/// > fixed — and the block was attached to [`ArtifactDirs`] rather than to this function, so
/// > the false claim was invisible from here. `convert_glimmer`'s `REQUIRED_AUX` note already
/// > recorded the true behaviour, which is how the two came to disagree in writing.
/// >
/// > The trade it describes is real and is answered by [`require_aux`], not by a warning: run
/// > the same list against the source FIRST, and the multi-hour convert never starts.
///
/// A missing manifest is not survivable either, so that one propagates too.
pub fn finish_artifact(
    tool: &str,
    dirs: ArtifactDirs<'_>,
    manifest: &serde_json::Value,
    aux: &[&str],
) -> Result<()> {
    let ArtifactDirs {
        out: out_dir,
        src: src_dir,
    } = dirs;
    let path = format!("{out_dir}/manifest.json");
    std::fs::write(&path, serde_json::to_vec_pretty(manifest)?)
        .with_context(|| format!("write {path}"))?;
    for name in aux {
        std::fs::copy(format!("{src_dir}/{name}"), format!("{out_dir}/{name}"))
            .with_context(|| format!("{tool}: copy {name} into the artifact"))?;
        // Gate the ARTIFACT, not the input: the copy above could succeed against a
        // just-deleted source or a full disk in ways whose error surfaces elsewhere, and
        // the old tree shipped a converter that exited 0 with no generation_config.json —
        // leaving the engine ZERO stop tokens, announced by one warn!, behind a 56-run
        // retraction. A structural absence is an error here, never a warning.
        anyhow::ensure!(
            std::fs::metadata(format!("{out_dir}/{name}")).is_ok_and(|m| m.len() > 0),
            "{tool}: {name} is missing or empty in the artifact after the copy"
        );
        eprintln!("{tool}: copied {name}");
    }
    Ok(())
}

/// Provenance of the artifact's `.i4` expert set: which tool produced it, from what,
/// and over which layers. Absent on artifacts built before this field existed — and
/// that absence is itself the signal, since a `vq3_to_i4` set and an `fp8_to_i4` set are
/// otherwise byte-indistinguishable on disk, which is exactly how a bad `.i4` set
/// stayed invisible.
///
/// EVERY writer of `L{l}.i4` must call [`I4Source::stamp`]. A stale stamp is worse
/// than no stamp: the engine reports it as fact, so a tool that rewrites the set
/// without restamping turns an honest ambiguity into a confident lie.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct I4Source {
    /// Binary that wrote the set, e.g. `"fp8_to_i4"`.
    pub tool: String,
    /// Derivation chain, e.g. `"fp8->int4"` vs the older `"fp8->vq3->int4"`.
    pub chain: String,
    /// Path of the source the weights were derived from.
    pub src: String,
    /// Half-open layer range this provenance covers, `[from, to)`. A run that
    /// rebuilds only part of the set records only that part, so a mixed artifact
    /// never claims to be uniform.
    pub layers: [usize; 2],
    /// [`crate::quant::I4_GROUP`] the set was quantized at — weights per f32 scale
    /// along the input dim. Without it a G=128 set and a G=64 set are the same
    /// `tool`/`chain`/`src` triple, and the quality difference between them would be
    /// unattributable. `None` is an artifact predating group scales (per-row); such a
    /// set has a different `.i4` file size and is rejected by `ExpertSet::open`, so
    /// this is a diagnosis, not a load-time guard.
    #[serde(default)]
    pub group: Option<usize>,
}

impl I4Source {
    /// Read `<dir>/manifest.json`'s `i4_source` section. `Ok(None)` means "no
    /// manifest, or no such field" — an unstamped artifact, which is a reportable
    /// fact rather than an error. A field that is PRESENT but unparseable is an
    /// error: silently reporting it as "unstamped" would hide a real corruption.
    pub fn load(dir: &str) -> Result<Option<Self>> {
        let Ok(text) = std::fs::read(format!("{dir}/manifest.json")) else {
            return Ok(None);
        };
        let v: serde_json::Value =
            serde_json::from_slice(&text).with_context(|| format!("parse {dir}/manifest.json"))?;
        let Some(f) = v.get("i4_source") else {
            return Ok(None);
        };
        Ok(Some(
            serde_json::from_value(f.clone()).context("manifest i4_source is malformed")?,
        ))
    }

    /// Everything a merge must match — this stamp minus its layer range. `group` belongs in
    /// it because merging a G=64 run into a G=128 run would claim one uniform set for two
    /// incompatible formats.
    fn derivation(&self) -> (&str, &str, &str, Option<usize>) {
        (&self.tool, &self.chain, &self.src, self.group)
    }

    /// Whether `prior` is the same derivation as this one over a range that touches it — the
    /// two facts a merge needs, asked one at a time so neither can be mistaken for the other.
    fn continues(&self, prior: &Self) -> bool {
        if prior.derivation() != self.derivation() {
            return false;
        }
        if prior.layers[0] > self.layers[1] {
            return false;
        }
        self.layers[0] <= prior.layers[1]
    }

    /// This stamp, widened to cover `prior` when the two are one contiguous claim — so a run
    /// resumed with `--from` still ends up claiming the whole set it rebuilt, rather than only
    /// its own final leg. Anything else REPLACES, which is what stops a rebuilt set from
    /// carrying a stale claim.
    fn merged_with(&self, prior: Option<Self>) -> Self {
        let Some(p) = prior.filter(|p| self.continues(p)) else {
            return self.clone();
        };
        Self {
            layers: [
                p.layers[0].min(self.layers[0]),
                p.layers[1].max(self.layers[1]),
            ],
            ..self.clone()
        }
    }

    /// Record this provenance in `<dir>/manifest.json`, merging with an existing stamp per
    /// [`Self::merged_with`] and publishing it per [`write_json_atomic`].
    pub fn stamp(&self, dir: &str) -> Result<()> {
        let path = format!("{dir}/manifest.json");
        let mut m: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&path).with_context(|| format!("read {path}"))?)
                .with_context(|| format!("parse {path}"))?;
        // An unreadable prior stamp is not an error HERE — we are replacing it. Only
        // the reader is strict, so a corrupt field can never be misreported as fact.
        m["i4_source"] = serde_json::to_value(self.merged_with(Self::load(dir).ok().flatten()))?;
        write_json_atomic(&path, &m)
    }
}

/// Publish a JSON document at `path` by tmp→fsync→rename.
///
/// The fsync is what separates this from [`super::layer::write_expert_layer`], which deliberately does not
/// pay one: a torn `manifest.json` bricks an artifact whose `.i4` set alone is ~365 GB, while
/// a torn layer file is regenerable.
fn write_json_atomic(path: &str, doc: &serde_json::Value) -> Result<()> {
    use std::io::Write;
    let tmp = format!("{path}.tmp");
    let mut f = std::fs::File::create(&tmp).with_context(|| format!("create {tmp}"))?;
    f.write_all(&serde_json::to_vec_pretty(doc)?)?;
    f.sync_all().with_context(|| format!("fsync {tmp}"))?;
    drop(f);
    std::fs::rename(&tmp, path).with_context(|| format!("rename {tmp} -> {path}"))
}

#[cfg(test)]
mod tests {
    // The subject is a manifest a previous run wrote, so every arm here round-trips through a
    // real file rather than through the struct. Crate-wide `unwrap`/`expect` are `deny`; a
    // firing one IS the report.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::super::fixtures::tmpdir;
    use super::*;

    /// Provenance round-trips, a missing field reads as unstamped, a malformed one is
    /// an error (not silently "unstamped"), and a resumed run merges into one range.
    #[test]
    fn i4_source_round_trips_and_merges_adjoining_runs() {
        let dir = tmpdir("i4src_test");
        let mf = format!("{dir}/manifest.json");
        std::fs::write(&mf, br#"{"hidden_size":6144}"#).unwrap();
        assert!(I4Source::load(&dir).unwrap().is_none()); // no field yet

        let a = I4Source {
            tool: "fp8_to_i4".into(),
            chain: "fp8->int4".into(),
            src: "/src".into(),
            layers: [3, 40],
            group: Some(crate::quant::I4_GROUP),
        };
        a.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().as_ref(), Some(&a));
        // The stamp must not clobber the rest of the manifest.
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&mf).unwrap()).unwrap();
        assert_eq!(v["hidden_size"], 6144);

        // A resume picking up where the first run stopped claims the whole range.
        I4Source {
            layers: [40, 78],
            ..a.clone()
        }
        .stamp(&dir)
        .unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().unwrap().layers, [3, 78]);

        // A different derivation replaces rather than merges — no stale claims.
        let b = I4Source {
            chain: "fp8->vq3->int4".into(),
            layers: [3, 78],
            ..a.clone()
        };
        b.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap(), Some(b.clone()));

        // A different GROUP is a different derivation too: rebuilding the tail at
        // G=64 must not merge into a G=128 claim over the head, or the manifest would
        // describe one uniform set where two incompatible formats sit side by side.
        let c = I4Source {
            group: Some(64),
            layers: [40, 78],
            ..b.clone()
        };
        c.stamp(&dir).unwrap();
        assert_eq!(I4Source::load(&dir).unwrap(), Some(c));

        // An artifact stamped before group scales existed reads as group: None.
        std::fs::write(
            &mf,
            br#"{"i4_source":{"tool":"t","chain":"c","src":"/s","layers":[0,1]}}"#,
        )
        .unwrap();
        assert_eq!(I4Source::load(&dir).unwrap().unwrap().group, None);

        // Present-but-malformed is an error, never a silent "unstamped".
        std::fs::write(&mf, br#"{"i4_source":{"tool":"x"}}"#).unwrap();
        assert!(I4Source::load(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
