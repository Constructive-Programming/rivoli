//! The Rust kernels' code object, loaded once, and the one launch path every Rust launcher
//! takes.
//!
//! `build.rs` builds `crates/kernels` with the pinned nightly and hands the code object over
//! as `RIVOLI_KERNELS_HSACO`; it is embedded here and loaded with `hipModuleLoadData` the
//! first time any Rust kernel launches. A [`Kernel`] is a name plus its lazily resolved
//! `hipFunction_t`; [`Kernel::launch`] is `hipModuleLaunchKernel` with the kernel's one
//! `rivoli_abi::kernarg` block — the struct the kernel itself takes, so no host-side list
//! restates a kernel's parameters.
//!
//! Return convention: `0`, or `-(hipError_t)` — the C launchers' convention, so a Rust
//! `rivoli_*` launcher drops into `launchers!` exactly where the extern one stood.

#![cfg(feature = "rocm")]

use crate::hiprt::{HipError, hipModuleGetFunction, hipModuleLaunchKernel, hipModuleLoadData};
use std::ffi::{CStr, c_void};
use std::marker::PhantomData;
use std::sync::OnceLock;

/// The code object, 8-aligned: `hipModuleLoadData` reads it as an ELF image in place.
#[repr(C, align(8))]
struct Image<const N: usize>([u8; N]);
static CODE: &Image<{ include_bytes!(env!("RIVOLI_KERNELS_HSACO")).len() }> =
    &Image(*include_bytes!(env!("RIVOLI_KERNELS_HSACO")));

/// A `hipModule_t`/`hipFunction_t` — opaque runtime handles, immutable once created and
/// usable from any host thread (HIP's contract), which is the whole case for `Send + Sync`.
#[derive(Clone, Copy)]
struct Handle(*mut c_void);
// SAFETY: see the type's doc — HIP module and function handles are thread-agnostic.
unsafe impl Send for Handle {}
// SAFETY: as above; nothing mutates through the handle on the host.
unsafe impl Sync for Handle {}

/// Load the module once per process. A failure is remembered, not retried: every later
/// launch reports the same `hipError_t`, which is the honest answer.
fn module() -> Result<Handle, HipError> {
    static MODULE: OnceLock<Result<Handle, HipError>> = OnceLock::new();
    *MODULE.get_or_init(|| {
        let mut m = std::ptr::null_mut();
        // SAFETY: `CODE` is a complete, 'static, aligned code object.
        let e = unsafe { hipModuleLoadData(&mut m, CODE.0.as_ptr().cast()) };
        if e == 0 { Ok(Handle(m)) } else { Err(e) }
    })
}

/// One kernel in the embedded code object, typed by the `rivoli_abi::kernarg` block it
/// takes — so a launch with another kernel's block is a compile error, not HIP copying the
/// metadata's byte count from past the end of the wrong struct. (`fn(A)`, not `A`, so the
/// handle stays `Sync` although the blocks hold raw pointers.)
pub(crate) struct Kernel<A> {
    name: &'static CStr,
    func: OnceLock<Result<Handle, HipError>>,
    args: PhantomData<fn(A)>,
}

/// `(grid blocks, block threads, dynamic LDS bytes)` — 1-D, the only shape any ported
/// launcher uses.
pub(crate) type Geometry = (u32, u32, u32);

impl<A: Copy> Kernel<A> {
    pub(crate) const fn new(name: &'static CStr) -> Self {
        Self {
            name,
            func: OnceLock::new(),
            args: PhantomData,
        }
    }

    fn func(&self) -> Result<Handle, HipError> {
        *self.func.get_or_init(|| {
            let m = module()?;
            let mut f = std::ptr::null_mut();
            // SAFETY: a live module and a NUL-terminated name.
            let e = unsafe { hipModuleGetFunction(&mut f, m.0, self.name.as_ptr()) };
            if e == 0 { Ok(Handle(f)) } else { Err(e) }
        })
    }

    /// Launch on `stream` (null = the null stream) with the kernel's argument block —
    /// the `rivoli_abi::kernarg` struct the kernel takes by value, handed to HIP as the
    /// single `kernelParams` entry (the code object's metadata sizes the copy).
    ///
    /// # Safety
    /// Every pointer in `args` satisfies the kernel's buffer contract, and `stream` is live
    /// or null.
    #[must_use = "the launch status is the only report of a failed launch"]
    pub(crate) unsafe fn launch(
        &self,
        (grid, block, lds): Geometry,
        stream: *mut c_void,
        args: &A,
    ) -> i32 {
        let f = match self.func() {
            Ok(f) => f,
            Err(e) => return -e,
        };
        let mut params = [std::ptr::from_ref(args).cast_mut().cast::<c_void>()];
        // SAFETY: the caller's contract; HIP copies `size_of::<A>()` bytes per the metadata.
        let e = unsafe {
            hipModuleLaunchKernel(
                f.0,
                grid,
                1,
                1,
                block,
                1,
                1,
                lds,
                stream,
                params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        -e
    }
}
