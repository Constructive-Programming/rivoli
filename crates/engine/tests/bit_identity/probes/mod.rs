//! The prelude probe rows — Phase 0b's families (codec sweeps, reductions, the four dot
//! formats) ported from its scratch harness (recorded in `docs/investigations/rust-kernels.md`
//! §"Phase 0b — the prelude", 2026-11-18) onto this gate's machinery.
//!
//! **A module-load reference beside the shared-object one.** The launcher rows compare
//! `launch_*` with the tag's C launchers; a prelude helper has no launcher, so each is
//! exposed by a PROBE kernel — `probe.hip` here, built by hipcc against the tag's headers,
//! and its twin in `crates/kernels/src/probes`, built into a second code object by
//! `tests/bit-identity.sh --build-probes` — and both are launched from one argument buffer.
//!
//! The verdict rules carry over from 0b's harness: every group prints its examined count and
//! one with none is VACUOUS; outputs start as different fills per side; f32 outputs are
//! judged with NaN as a class, every other output type ([`Out`]) strictly; and the compare
//! runs ON THE DEVICE in hipcc-built kernels (`probe.hip`'s `cmp_*`), because the codec
//! sweeps are exhaustive over 2^32 inputs and compare ~1.5 TB.

mod codec;
mod dots;

use crate::common::{DeviceBuf, dev};
use rivoli_backend::{CodeObject, Geometry as Geom, fill_u32};
use std::ffi::{CString, c_void};

/// A device buffer of known length, as the probe rows use one.
pub struct Dev(DeviceBuf, usize);

impl Dev {
    pub fn zeroed(bytes: usize) -> Self {
        Self(DeviceBuf::new(bytes.max(1)).expect("alloc"), bytes)
    }

    /// An input, uploaded with 64 zero bytes of tail slack (0b's `up()`): a helper that reads
    /// past a row's end then reads the same zeros on both sides, not whatever the allocator
    /// placed next — which differs between the two launches, since an output is between them.
    pub fn from<T: Copy>(v: &[T]) -> Self {
        // SAFETY: plain `Copy` data, viewed as its bytes for the upload.
        let b = unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), size_of_val(v)) };
        let mut padded = b.to_vec();
        padded.resize(b.len() + 64, 0);
        Self(dev(&padded), b.len())
    }

    pub fn ptr(&self) -> *mut c_void {
        self.0.ptr().cast_mut().cast()
    }

    /// The first `bytes` (a multiple of 4: every probe output is) set to `byte`.
    fn fill(&self, byte: u8, bytes: usize) {
        assert!(bytes <= self.1, "fill past the allocation");
        // SAFETY: within the allocation.
        unsafe { fill_u32(self.ptr().cast(), u32::from_ne_bytes([byte; 4]), bytes) }.expect("fill");
    }
}

/// A kernel argument buffer at natural alignment — the kernarg layout both compilers use
/// for plain positional parameters.
#[derive(Default, Clone)]
pub struct Args(Vec<u8>);

impl Args {
    pub fn p<T: Copy>(mut self, v: T) -> Self {
        let a = align_of::<T>();
        self.0.resize(self.0.len().div_ceil(a) * a, 0);
        // SAFETY: `T: Copy` plain data; its bytes are read, not interpreted.
        let bytes = unsafe {
            std::slice::from_raw_parts(std::ptr::from_ref(&v).cast::<u8>(), size_of::<T>())
        };
        self.0.extend_from_slice(bytes);
        self
    }
}

/// One loaded code object, launching by name.
struct Module(CodeObject);

impl Module {
    fn load(path: &str) -> Self {
        let c = CString::new(path).expect("path has no NUL");
        Self(CodeObject::load(&c).unwrap_or_else(|e| panic!("hipModuleLoad {path}: hipError {e}")))
    }

    fn launch(&self, name: &str, geom: Geom, mut args: Args) {
        let c = CString::new(name).expect("name has no NUL");
        // SAFETY: the packed buffer is the probe kernel's own argument list (probe ABI).
        let e = unsafe { self.0.dispatch_packed(&c, geom, &mut args.0) };
        assert_eq!(e, 0, "launch {name}: rc {e}");
    }
}

/// An output's element type, stated at every call site: only `F32` is NaN-classified, so a
/// new probe that forgets the question cannot get the lenient rule by accident (0b's rule).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Out {
    F32,
    U8,
    U16,
    U32,
    U64,
}

impl Out {
    pub fn size(self) -> usize {
        match self {
            Out::U8 => 1,
            Out::U16 => 2,
            Out::F32 | Out::U32 => 4,
            Out::U64 => 8,
        }
    }
}

/// Kernels `(hipcc, Rust)` — the same name but for `k_vq_wave_r1` — and the geometry both
/// launch at.
pub type Pair<'a> = ((&'a str, &'a str), Geom);

/// The two output buffers of one comparison, how many bytes of each are compared, and their
/// element type.
pub struct Outs<'a> {
    pub a: &'a Dev,
    pub b: &'a Dev,
    pub bytes: usize,
    pub ty: Out,
}

/// One comparison: mismatching bytes, the lowest mismatching byte, and whether the mismatch
/// is NaN-payload-only (`ty` f32 and every differing word NaN on both sides).
pub struct Cmp {
    pub bad: u64,
    pub first: u64,
    pub nan_only: bool,
}

/// The two probe code objects, and the device compare.
pub struct Probes {
    hip: Module,
    rust: Module,
}

impl Probes {
    /// `RIVOLI_BITIDENT_PROBES_HIP` names the hipcc object `tests/bit-identity.sh` builds from
    /// `probe.hip` against the tag's headers; `RIVOLI_BITIDENT_PROBES_RUST` the Rust object
    /// (clean or a plant) its `--build-probes` built. Fixture paths a script sets, never
    /// instruments in a shipped binary.
    pub fn open() -> Self {
        let var = |v: &str| {
            std::env::var(v).unwrap_or_else(|_| {
                panic!("{v} unset — run through tests/bit-identity.sh --probes")
            })
        };
        let (hip, rust) = (
            var("RIVOLI_BITIDENT_PROBES_HIP"),
            var("RIVOLI_BITIDENT_PROBES_RUST"),
        );
        println!("probes: hip {hip}\nprobes: rust {rust}");
        Self {
            hip: Module::load(&hip),
            rust: Module::load(&rust),
        }
    }

    /// One comparison: fill the outputs differently per side (so a byte one side never
    /// writes cannot match stale data), launch `hk` (hipcc) into `a` and `rk` (Rust) into `b`,
    /// each from `args(out)`, and compare the first `bytes` on the device.
    pub fn cell(&self, ((hk, rk), geom): Pair, args: impl Fn(*mut c_void) -> Args, o: Outs) -> Cmp {
        o.a.fill(0xA5, o.bytes);
        o.b.fill(0x5A, o.bytes);
        self.hip.launch(hk, geom, args(o.a.ptr()));
        self.rust.launch(rk, geom, args(o.b.ptr()));
        self.compare(&o)
    }

    fn compare(&self, o: &Outs) -> Cmp {
        let run = |k: &str, n: usize, init: [u64; 2]| {
            let bytes: Vec<u8> = init.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut c = dev(&bytes);
            let args = Args::default()
                .p(o.a.ptr())
                .p(o.b.ptr())
                .p(n)
                .p(c.ptr_mut());
            self.hip.launch(k, (n.div_ceil(256) as u32, 256, 0), args);
            let v = crate::common::back(&c);
            (
                u64::from_le_bytes(v[..8].try_into().expect("8")),
                u64::from_le_bytes(v[8..16].try_into().expect("8")),
            )
        };
        let (bad, first) = run("cmp_bytes", o.bytes, [0, u64::MAX]);
        let nan_only = bad > 0 && o.ty == Out::F32 && {
            let (nan_both, value) = run("cmp_nan_words", o.bytes / 4, [0, 0]);
            nan_both > 0 && value == 0
        };
        Cmp {
            bad,
            first,
            nan_only,
        }
    }
}

/// One named comparison group — a row of the verdict table.
#[derive(Default)]
pub struct Group {
    pub cells: u64,
    pub bytes: u64,
    pub bad: u64,
    pub nan_only: u64,
    pub first: String,
}

impl Group {
    pub fn cell(&mut self, c: &Cmp, bytes: usize, where_: impl FnOnce() -> String) {
        let bad = c.bad > 0;
        self.cells += 1;
        self.bytes += bytes as u64;
        self.nan_only += u64::from(bad && c.nan_only);
        if bad && self.first.is_empty() {
            self.first = where_();
        }
        self.bad += u64::from(bad);
    }

    /// The verdict line in the launcher rows' token form; a value difference (a mismatched
    /// cell that is not NaN-payload-only) is RED, nothing examined is VACUOUS.
    pub fn verdict(&self, name: &str) -> (String, bool) {
        let token = match (self.cells, self.bad > self.nan_only) {
            (0, _) => "VACUOUS",
            (_, true) => "RED",
            (_, false) => "BIT-IDENTICAL",
        };
        let first = if self.first.is_empty() {
            String::new()
        } else {
            format!(" first {}", self.first)
        };
        let line = format!(
            "{token} {name}: {} cells, {} bytes; mismatched {} cells; nan-payload-only {}{first}",
            self.cells, self.bytes, self.bad, self.nan_only
        );
        (line, token == "BIT-IDENTICAL")
    }
}

/// Rows emitted so far, with the family filter.
pub struct Table {
    pub rows: Vec<(String, Group)>,
    families: Option<String>,
}

impl Table {
    /// `RIVOLI_BITIDENT_FAMILIES` (comma list: codec, reduce, fp8, i4, f4, vq) narrows a plant
    /// arm to the families its plant can reach — a fixture the script sets per arm.
    pub fn new() -> Self {
        Self {
            rows: Vec::new(),
            families: std::env::var("RIVOLI_BITIDENT_FAMILIES").ok(),
        }
    }

    pub fn wants(&self, family: &str) -> bool {
        self.families
            .as_ref()
            .is_none_or(|f| f.split(',').any(|x| x == family))
    }

    pub fn push(&mut self, name: String, g: Group) {
        let (line, _) = g.verdict(&name);
        println!("{line}");
        self.rows.push((name, g));
    }
}

/// Every probe family, in 0b's order.
pub fn run(p: &Probes) -> Table {
    let mut t = Table::new();
    if t.wants("codec") {
        codec::codecs(p, &mut t);
    }
    if t.wants("reduce") {
        dots::reductions(p, &mut t);
    }
    dots::dot_families(p, &mut t);
    t
}
