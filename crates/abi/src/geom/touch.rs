//! `kernels/async.hip`'s `touch_region`, the fetch path's page-touch.

/// `touch_region`'s block (the kernel's grid-stride start index is `bid * TOUCH_BLOCK`).
pub const TOUCH_BLOCK: u32 = 256;
