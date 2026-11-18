//! `kernels/async.hip`'s `touch_region`, the fetch path's page-touch.

use core::ffi::c_void;

/// `touch_region` — load every 16-byte unit of `src[0..n4]`, grid-stride by `stride`
/// (`gridDim · blockDim`, the launcher's choice); `sink` is written only on an impossible
/// value.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Touch {
    pub src: *const c_void,
    pub sink: *mut u64,
    pub n4: u64,
    pub stride: u64,
}
const _: () = assert!(size_of::<Touch>() == 32);
