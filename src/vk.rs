//! The Vulkan compute backend — the `rocm`-free twin of `hip.rs`. Instance/device
//! bring-up, memory, queue/timeline sync, and the `launch_*` surface over the SPIR-V
//! kernels in `kernels/vk/`. See docs/VULKAN.md.

#![cfg(feature = "vulkan")]
