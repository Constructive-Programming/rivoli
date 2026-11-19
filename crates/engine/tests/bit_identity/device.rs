//! The gate's machinery: the tagged hipcc reference loaded from its own shared object, one
//! cell's two launches over shared inputs and per-side poisoned outputs, the byte compare,
//! the guard-code compare, and the per-kernel timing.

use super::common::{DeviceBuf, back, dev, f32b};
use rivoli_backend::gpustream::HipEvent;
use std::ffi::{CStr, CString, c_char, c_void};

unsafe extern "C" {
    fn dlopen(file: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}
const RTLD_NOW: i32 = 2;
/// glibc's default, spelled so the `-Bsymbolic` argument below is visible at the call.
const RTLD_LOCAL: i32 = 0;

/// The reference: the tagged `kernels/*.hip` the script's `REF_SOURCES` lists, built by
/// `tests/bit-identity.sh` with hipcc into a shared object whose `rivoli_*` C launchers are
/// called here exactly as the engine called them before the migration — guards, grid
/// geometry and kernel together. Loaded `RTLD_LOCAL` and linked `-Bsymbolic`, so its
/// launchers bind to its OWN kernels and never to a same-named symbol in this process
/// (`rivoli_act_quant_f8` is one: the Rust launcher exports it for `moe_f4.hip`).
pub struct Ref(*mut c_void);

impl Ref {
    /// `RIVOLI_BITIDENT_REF` names the shared object. An environment variable and not a flag
    /// because libtest owns this binary's argv; it is a FIXTURE path the script sets, never
    /// an instrument in a shipped binary (CLAUDE.md's carve-out for non-cargo scripts).
    pub fn open() -> Self {
        let path = std::env::var("RIVOLI_BITIDENT_REF")
            .expect("RIVOLI_BITIDENT_REF unset — run this through tests/bit-identity.sh");
        let c = CString::new(path.clone()).expect("path has no NUL");
        // SAFETY: a NUL-terminated path; the handle lives for the process.
        let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW | RTLD_LOCAL) };
        assert!(!h.is_null(), "dlopen {path}: {}", last_dl_error());
        Self(h)
    }

    /// The reference's C launcher `name`, as the fn-pointer type `F`.
    ///
    /// # Safety
    /// `F` is an `unsafe extern "C" fn` with `name`'s exact C signature.
    pub unsafe fn sym<F: Copy>(&self, name: &CStr) -> F {
        assert_eq!(
            size_of::<F>(),
            size_of::<*mut c_void>(),
            "F must be a fn pointer"
        );
        // SAFETY: a live handle and a NUL-terminated name.
        let p = unsafe { dlsym(self.0, name.as_ptr()) };
        assert!(
            !p.is_null(),
            "reference lacks {name:?}: {}",
            last_dl_error()
        );
        // SAFETY: the caller's signature contract; sizes checked above.
        unsafe { std::mem::transmute_copy(&p) }
    }
}

fn last_dl_error() -> String {
    // SAFETY: dlerror returns null or a NUL-terminated string valid until the next dl call.
    let e = unsafe { dlerror() };
    if e.is_null() {
        "no dlerror".into()
    } else {
        // SAFETY: as above.
        unsafe { CStr::from_ptr(e) }.to_string_lossy().into_owned()
    }
}

// The reference launchers' C signatures use these spellings, so each reads as the one line
// it is in the C: `I` an input buffer, `O` an output, `S` a stream.
pub type I = *const f32;
pub type O = *mut f32;
pub type S = *mut c_void;

/// Which twin a launch closure is driving.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Hip,
    Rust,
}

/// An output's starting contents. `Poison` outputs start as a DIFFERENT NaN pattern per
/// side, so an element either side failed to write is a mismatch by construction — two
/// launches that both did nothing cannot agree. `InPlace` outputs start as the same input
/// on both sides (the kernel reads them), and the row instead demands that at least one
/// cell changed them (see [`Tally::in_place`]).
pub enum Out {
    Poison(usize),
    InPlace(Vec<f32>),
}

/// Two DIFFERENT finite sentinels (±3.39e38 with distinct mantissas), not NaNs: under the
/// NaN-as-a-class compare two NaN poisons would compare equal, and an element both twins
/// failed to write would pass. A kernel that genuinely outputs one of these exact values on
/// one side only would read as a mismatch — the safe direction.
const POISON: [u32; 2] = [0x7f7f_1234, 0xff7f_5678];

impl Out {
    /// A second output with the same starting contents (the timing cell's own).
    fn fresh(&self) -> Self {
        match self {
            Out::Poison(n) => Out::Poison(*n),
            Out::InPlace(v) => Out::InPlace(v.clone()),
        }
    }

    fn upload(&self, side: Side) -> DeviceBuf {
        match self {
            Out::Poison(n) => dev(&f32b(&vec![f32::from_bits(POISON[side as usize]); *n])),
            Out::InPlace(v) => dev(&f32b(v)),
        }
    }
}

/// One cell's device inputs (read-only, shared by both sides), its two output sets, and
/// the outputs' starting contents (which the compare needs to tell in-place outputs apart).
pub struct Cell {
    ins: Vec<DeviceBuf>,
    outs: [Vec<DeviceBuf>; 2],
    init: Vec<Out>,
}

impl Cell {
    pub fn new(ins: &[Vec<f32>], init: Vec<Out>) -> Self {
        let set = |side| init.iter().map(|o| o.upload(side)).collect();
        Self {
            ins: ins.iter().map(|v| dev(&f32b(v))).collect(),
            outs: [set(Side::Hip), set(Side::Rust)],
            init,
        }
    }

    /// Input `i`'s device address.
    pub fn i(&self, i: usize) -> *const f32 {
        self.ins[i].ptr().cast()
    }

    /// Output `o`'s device address on `side`.
    pub fn o(&mut self, side: Side, o: usize) -> *mut f32 {
        self.outs[side as usize][o].ptr_mut().cast()
    }
}

/// One row's counts. Every count is printed, and the gate refuses a row where any is zero:
/// a comparison that examined nothing is not a pass.
#[derive(Default)]
pub struct Tally {
    pub cells: u64,
    pub words: u64,
    pub bad_cells: u64,
    pub bad_words: u64,
    /// Words where BOTH twins produced a NaN with different payloads — equal by the owner's
    /// NaN-as-a-class rule (2026-11-18), counted and printed so the class stays visible.
    pub nan_payload_only: u64,
    pub guards: u64,
    pub bad_guards: u64,
    /// In-place cells whose output moved from its input — must be > 0 for a row with any.
    pub moved: u64,
    pub in_place_rows: bool,
    pub notes: Vec<String>,
    pub timing: Vec<String>,
}

impl Tally {
    /// Launch both twins over `cell`, then compare every output byte for byte.
    pub fn run(&mut self, what: &str, mut cell: Cell, launch: impl Fn(Side, &mut Cell)) {
        launch(Side::Hip, &mut cell);
        launch(Side::Rust, &mut cell);
        self.cells += 1;
        let mut bad = 0u64;
        for (k, init) in cell.init.iter().enumerate() {
            let (hip, rust) = (back(&cell.outs[0][k]), back(&cell.outs[1][k]));
            let words = hip.len() as u64 / 4;
            let (diff, nan_only) = compare_f32(&hip, &rust);
            self.nan_payload_only += nan_only;
            self.words += words;
            bad += diff;
            if let Out::InPlace(v) = init {
                self.in_place_rows = true;
                self.moved += u64::from(hip != f32b(v));
            }
            if diff > 0 && self.notes.len() < 6 {
                self.notes
                    .push(format!("{what} out{k}: {diff}/{words} words differ"));
            }
        }
        self.bad_words += bad;
        self.bad_cells += u64::from(bad > 0);
    }

    /// One refusal, as `[hipcc, rust]` codes (see [`code`]): the reference's must be
    /// POSITIVE (the case must be a guard, or it would launch on the null pointers these
    /// cases pass), and the Rust launcher must refuse with the same code.
    pub fn guard(&mut self, what: &str, [hip_rc, rust_rc]: [i32; 2]) {
        assert!(
            hip_rc > 0,
            "{what}: the reference accepted (rc {hip_rc}) — not a guard case"
        );
        self.guards += 1;
        if rust_rc != hip_rc {
            self.bad_guards += 1;
            self.notes
                .push(format!("{what}: hipcc refused {hip_rc}, Rust {rust_rc}"));
        }
    }

    /// Per-launch µs of each twin at one shape — the gate cannot see a slowdown, this does.
    /// Three alternating batches of 200 after a warm-up; the minimum batch per side.
    pub fn time(&mut self, what: &str, cell: &mut Cell, launch: impl Fn(Side, &mut Cell)) {
        let mut best = [f64::MAX; 2];
        for side in [Side::Hip, Side::Rust] {
            (0..20).for_each(|_| launch(side, cell));
        }
        for _ in 0..3 {
            for side in [Side::Hip, Side::Rust] {
                let (a, b) = (
                    HipEvent::new().expect("event"),
                    HipEvent::new().expect("event"),
                );
                a.record(std::ptr::null_mut()).expect("record");
                (0..200).for_each(|_| launch(side, cell));
                b.record(std::ptr::null_mut()).expect("record");
                rivoli_backend::hip::device_sync().expect("sync");
                let ms = HipEvent::elapsed_ms(&a, &b).expect("elapsed");
                best[side as usize] = best[side as usize].min(f64::from(ms) * 1000.0 / 200.0);
            }
        }
        let [h, r] = best;
        self.timing.push(format!(
            "{what}: hipcc {h:.2} µs, rust {r:.2} µs, ratio {:.3}",
            r / h
        ));
    }

    /// The row's verdict line, and whether it is green. The first word is one of three
    /// tokens the script classifies on: `BIT-IDENTICAL` (every word equal, NaN as a class —
    /// see [`compare_f32`]), `RED` (a mismatch), or `VACUOUS` —
    /// a row that examined nothing (zero cells, words or guards, or in-place outputs that
    /// never moved), which is neither verdict and fails both the gate and its red proof.
    pub fn verdict(&self, row: &str) -> (String, bool) {
        let examined = self.cells > 0 && self.words > 0 && self.guards > 0;
        let wrote = !self.in_place_rows || self.moved > 0;
        let clean = self.bad_words == 0 && self.bad_guards == 0;
        let token = match (examined && wrote, clean) {
            (false, _) => "VACUOUS",
            (true, false) => "RED",
            (true, true) => "BIT-IDENTICAL",
        };
        let line = format!(
            "{token} {row}: {} cells, {} words, {} guards; mismatched {} cells / {} words, {} guards; \
             in-place moved {}; nan-payload-only {}",
            self.cells,
            self.words,
            self.guards,
            self.bad_cells,
            self.bad_words,
            self.bad_guards,
            self.moved,
            self.nan_payload_only,
        );
        (line, token == "BIT-IDENTICAL")
    }
}

/// A Rust launch's status in the C launchers' convention: `0` ok, the POSITIVE code from
/// `ensure_hip_status`'s "argument guard rejected (N)", or `-1` for any other error (a HIP
/// error is not a guard, and must not compare equal to one).
pub fn code(r: anyhow::Result<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => format!("{e:#}")
            .split_once("argument guard rejected (")
            .and_then(|(_, rest)| rest.split(')').next()?.parse().ok())
            .unwrap_or(-1),
    }
}

/// Both twins' status for one call; `call(side)` launches (or refuses) on that side.
pub fn both(call: impl Fn(Side) -> i32) -> [i32; 2] {
    [call(Side::Hip), call(Side::Rust)]
}

/// A row's shapes: every one is swept, and `hot` — the engine's hot shape for the kernel —
/// is also timed, at its seed-1 cell.
#[derive(Clone, Copy)]
pub struct Shapes<'a, S> {
    pub all: &'a [S],
    pub hot: S,
}

/// Every `shape × seed` cell of one row: `make` builds the cell's inputs, outputs and
/// scalar parameters from the shape and seed; `launch` drives one side and returns its
/// status, which must be acceptance.
pub fn sweep<S: Copy + PartialEq + std::fmt::Debug, P>(
    t: &mut Tally,
    shapes: Shapes<S>,
    make: impl Fn(S, u64) -> (Vec<Vec<f32>>, Vec<Out>, P),
    launch: impl Fn(Side, &mut Cell, S, &P) -> i32,
) {
    for &shape in shapes.all {
        for seed in 1..=super::inputs::SEEDS {
            let what = format!("{shape:?} seed {seed}");
            let (ins, outs, p) = make(shape, seed);
            let go = |side, c: &mut Cell| accepted(launch(side, c, shape, &p), side, &what);
            if shape == shapes.hot && seed == 1 {
                let timed = outs.iter().map(Out::fresh).collect();
                t.time(&format!("{shape:?}"), &mut Cell::new(&ins, timed), go);
            }
            t.run(&what, Cell::new(&ins, outs), go);
        }
    }
}

/// A value launch on `side`, which must have been accepted.
pub fn accepted(rc: i32, side: Side, what: &str) {
    let who = if side == Side::Hip {
        "the tagged hipcc"
    } else {
        "the Rust"
    };
    assert_eq!(
        rc, 0,
        "{who} launcher refused {what} (rc {rc}) — a value cell must be valid"
    );
}

/// The device addresses every guard case passes, as `(input, output)`: one live buffer of
/// 2^17 poisoned floats, larger than any guard case's extent. A guard case must be refused
/// before any launch — but if a regressed Rust guard ACCEPTS one, the launch writes this
/// buffer and the case reports rc 0 against the reference's positive code, a clean RED
/// line, instead of faulting the shared GPU on a null pointer (correctness review,
/// 2026-11-18). Allocated once and leaked: it lives as long as the process.
pub fn guard_ptrs() -> (I, O) {
    static ADDR: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let a = *ADDR.get_or_init(|| {
        let b = Box::leak(Box::new(dev(&f32b(&vec![
            f32::from_bits(POISON[0]);
            1 << 17
        ]))));
        b.ptr_mut() as usize
    });
    (a as I, a as O)
}

/// `(differing, nan_payload_only)` over two f32 buffers. **NaN is compared as a class**
/// (owner decision 2026-11-18): a word where both twins produced NaN counts as equal however
/// the payloads differ — 0b's GPU run measured 0 value differences across 13,956 cells but
/// 442 NaN-vs-NaN payload differences, from the operand order of commutative ops, which no
/// source spelling controls. Every other word must match bit for bit.
fn compare_f32(hip: &[u8], rust: &[u8]) -> (u64, u64) {
    let word = |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    hip.chunks_exact(4)
        .zip(rust.chunks_exact(4))
        .fold((0, 0), |(d, n), (a, b)| {
            let (x, y) = (word(a), word(b));
            match (a == b, x.is_nan() && y.is_nan()) {
                (true, _) => (d, n),
                (false, true) => (d, n + 1),
                (false, false) => (d + 1, n),
            }
        })
}
