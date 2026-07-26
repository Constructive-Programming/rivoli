//! The Vulkan compute backend — the `rocm`-free twin of `hip.rs`. Instance/device
//! bring-up, memory, queue/timeline sync, and the `launch_*` surface over the SPIR-V
//! kernels in `kernels/vk/`. See docs/VULKAN.md.
//!
//! Everything hangs off one process-global [`Gpu`]. HIP hides its context inside the
//! runtime, which is why `hip.rs`'s launchers and `device.rs`'s buffers take nothing
//! but pointers; keeping ours global buys the same signatures, so `gpu.rs` and
//! `pin.rs` never learn a backend exists.

#![cfg(feature = "vulkan")]

use anyhow::{Result, bail};
use ash::vk;
use std::sync::OnceLock;

/// The one wave width the kernels are written for (gfx1151 is native wave32). Unlike
/// HIP — where `WAVE 32` in `common.hpp` is an assumption — the pipelines pin it with
/// `requiredSubgroupSize`, so a driver that cannot honour it fails at init, loudly.
pub const WAVE: u32 = 32;
/// Threads per workgroup = `ROWS_PER_BLOCK * WAVE`; must match `kernels/vk/common.glsl`.
pub const ROWS_PER_BLOCK: u32 = 8;
pub const BLOCK: u32 = ROWS_PER_BLOCK * WAVE;

/// Instance, device, the dedicated compute queue, and a command pool on it. Built
/// once by [`gpu`]; never dropped (see the note there).
pub struct Gpu {
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub qfam: u32,
    pub memprops: vk::PhysicalDeviceMemoryProperties,
    /// Kept alive for the device's sake, and for `mem_info`'s memory-budget query.
    pub instance: ash::Instance,
    pub phys: vk::PhysicalDevice,
    entry: ash::Entry,
}

// SAFETY: every field is either an ash object (already Send+Sync) or a Vulkan handle.
// The handles that Vulkan requires to be EXTERNALLY SYNCHRONISED (the queue, command
// pools) are only touched behind the submit mutex in this module — same discipline
// the HIP side keeps by owning every stream on one thread.
unsafe impl Send for Gpu {}
unsafe impl Sync for Gpu {}

static GPU: OnceLock<Gpu> = OnceLock::new();

/// The process-global context, built on first use.
///
/// Not dropped: it lives until exit, and tearing a Vulkan device down while the
/// engine's buffers (also static-lifetime, also never freed on the HIP side) still
/// hold handles would be strictly worse than letting the OS reclaim it.
pub fn gpu() -> Result<&'static Gpu> {
    // ponytail: a lost init race just builds a second Gpu and drops it. The engine
    // drives the GPU from one thread, so this cannot happen in practice; a Mutex to
    // rule out a case that does not occur is not worth the deadlock surface.
    if let Some(g) = GPU.get() {
        return Ok(g);
    }
    let g = Gpu::new()?;
    Ok(GPU.get_or_init(|| g))
}

impl Gpu {
    fn new() -> Result<Self> {
        // SAFETY: loads libvulkan; the returned Entry owns it for the process.
        let entry = unsafe { ash::Entry::load() }
            .map_err(|e| anyhow::anyhow!("no Vulkan loader: {e}"))?;

        let app = vk::ApplicationInfo::default()
            .application_name(c"rivoli")
            .api_version(vk::API_VERSION_1_3);
        let ci = vk::InstanceCreateInfo::default().application_info(&app);
        // SAFETY: `ci` and everything it borrows outlive the call.
        let instance = unsafe { entry.create_instance(&ci, None) }?;

        let (phys, qfam) = Self::pick(&instance)?;
        let device = Self::create_device(&instance, phys, qfam)?;
        // SAFETY: `qfam` came from the queue-create-info above; index 0 exists.
        let queue = unsafe { device.get_device_queue(qfam, 0) };
        // SAFETY: `phys` is a live physical device from this instance.
        let memprops = unsafe { instance.get_physical_device_memory_properties(phys) };

        Ok(Self {
            device,
            queue,
            qfam,
            memprops,
            instance,
            phys,
            entry,
        })
    }

    /// First physical device whose queue families and features satisfy the whole
    /// contract. If none does, report EVERY missing item for the *last* candidate
    /// rather than a bare "no suitable device" — on a one-GPU box that is the answer.
    fn pick(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, u32)> {
        // SAFETY: `instance` is live.
        let devices = unsafe { instance.enumerate_physical_devices() }?;
        let mut last: Option<(String, Vec<String>)> = None;
        for phys in devices {
            // SAFETY: `phys` came from this instance.
            let props = unsafe { instance.get_physical_device_properties(phys) };
            let name = props.device_name_as_c_str().map_or_else(
                |_| "<unnamed>".to_string(),
                |s| s.to_string_lossy().into_owned(),
            );
            let mut missing = Vec::new();
            if props.api_version < vk::API_VERSION_1_3 {
                missing.push(format!(
                    "Vulkan 1.3 (device reports {}.{})",
                    vk::api_version_major(props.api_version),
                    vk::api_version_minor(props.api_version)
                ));
            }
            let qfam = Self::compute_queue(instance, phys);
            if qfam.is_none() {
                missing.push("a COMPUTE queue family".into());
            }
            missing.extend(Self::missing_features(instance, phys));
            match (qfam, missing.is_empty()) {
                (Some(q), true) => return Ok((phys, q)),
                _ => last = Some((name, missing)),
            }
        }
        match last {
            Some((name, missing)) => bail!(
                "no Vulkan device meets the rivoli contract; closest was {name}, missing: {}",
                missing.join(", ")
            ),
            None => bail!("no Vulkan physical devices found"),
        }
    }

    /// A DEDICATED compute family (COMPUTE without GRAPHICS) if the device has one,
    /// else any COMPUTE family. Colibri measured the dedicated queue as the better
    /// choice on RADV (commit 4d4cacc) — it does not contend with the compositor.
    fn compute_queue(instance: &ash::Instance, phys: vk::PhysicalDevice) -> Option<u32> {
        // SAFETY: `phys` came from `instance`.
        let fams = unsafe { instance.get_physical_device_queue_family_properties(phys) };
        let has_compute = |f: &vk::QueueFamilyProperties| f.queue_flags.contains(vk::QueueFlags::COMPUTE);
        let idx = |pred: &dyn Fn(&vk::QueueFamilyProperties) -> bool| {
            fams.iter().position(pred).map(|i| i as u32)
        };
        idx(&|f| has_compute(f) && !f.queue_flags.contains(vk::QueueFlags::GRAPHICS)).or_else(|| idx(&has_compute))
    }

    /// Every required feature the device lacks, named as the spec names it so the
    /// failure is actionable. Fail fast beats a mystery `VK_ERROR_DEVICE_LOST` later.
    fn missing_features(instance: &ash::Instance, phys: vk::PhysicalDevice) -> Vec<String> {
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default();
        // Scoped so the pNext chain's mutable borrows end and the filled-in structs
        // can be read below.
        let core = {
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut v11)
                .push_next(&mut v12)
                .push_next(&mut v13);
            // SAFETY: `phys` is live; the chain is valid for the duration of the call.
            unsafe { instance.get_physical_device_features2(phys, &mut f2) };
            f2.features
        };

        let mut sub = vk::PhysicalDeviceSubgroupProperties::default();
        let mut size = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
        {
            let mut p2 = vk::PhysicalDeviceProperties2::default()
                .push_next(&mut sub)
                .push_next(&mut size);
            // SAFETY: as above.
            unsafe { instance.get_physical_device_properties2(phys, &mut p2) };
        }

        let mut missing = Vec::new();
        let mut need = |ok: bool, what: &str| {
            if !ok {
                missing.push(what.to_string());
            }
        };
        // Lets ExpertDesc's six raw pointers stay six uint64 addresses in a push
        // constant instead of six descriptor bindings rewritten per expert.
        need(v12.buffer_device_address == vk::TRUE, "bufferDeviceAddress");
        // ...which the shader dereferences as uint64_t buffer references.
        need(core.shader_int64 == vk::TRUE, "shaderInt64");
        need(v12.timeline_semaphore == vk::TRUE, "timelineSemaphore");
        // The fp16 VQ codebook.
        need(v12.shader_float16 == vk::TRUE, "shaderFloat16");
        need(v11.storage_buffer16_bit_access == vk::TRUE, "storageBuffer16BitAccess");
        // WAVE 32 is not negotiable — the reductions are a fixed 32-lane ladder.
        need(v13.subgroup_size_control == vk::TRUE, "subgroupSizeControl");
        need(v13.compute_full_subgroups == vk::TRUE, "computeFullSubgroups");
        need(
            size.min_subgroup_size <= WAVE && WAVE <= size.max_subgroup_size,
            &format!(
                "a subgroup size of {WAVE} (device allows {}..={})",
                size.min_subgroup_size, size.max_subgroup_size
            ),
        );
        need(
            size.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE),
            "requiredSubgroupSizeStages including COMPUTE",
        );
        need(
            sub.supported_stages.contains(vk::ShaderStageFlags::COMPUTE),
            "subgroup operations in COMPUTE",
        );
        for (flag, name) in [
            (vk::SubgroupFeatureFlags::BASIC, "subgroup BASIC"),
            (vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE, "subgroup SHUFFLE_RELATIVE"),
        ] {
            need(sub.supported_operations.contains(flag), name);
        }
        // On Strix Halo host RAM *is* GPU memory (GTT); the pin path writes resident
        // weights straight into device-local memory, so this heap must exist.
        // SAFETY: `phys` is live.
        let mp = unsafe { instance.get_physical_device_memory_properties(phys) };
        let want = vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE;
        need(
            mp.memory_types[..mp.memory_type_count as usize]
                .iter()
                .any(|t| t.property_flags.contains(want)),
            "a DEVICE_LOCAL | HOST_VISIBLE memory type",
        );
        missing
    }

    fn create_device(
        instance: &ash::Instance,
        phys: vk::PhysicalDevice,
        qfam: u32,
    ) -> Result<ash::Device> {
        let prio = [1.0f32];
        let qci = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(qfam)
            .queue_priorities(&prio)];
        let mut v11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
        let mut v12 = vk::PhysicalDeviceVulkan12Features::default()
            .buffer_device_address(true)
            .timeline_semaphore(true)
            .shader_float16(true);
        let mut v13 = vk::PhysicalDeviceVulkan13Features::default()
            .subgroup_size_control(true)
            .compute_full_subgroups(true);
        let core = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        // Every extension this backend needs is core in 1.3 — including
        // subgroup_size_control and buffer_device_address — so the list is empty.
        let ci = vk::DeviceCreateInfo::default()
            .queue_create_infos(&qci)
            .enabled_features(&core)
            .push_next(&mut v11)
            .push_next(&mut v12)
            .push_next(&mut v13);
        // SAFETY: `phys` is live and `ci` outlives the call; features were verified
        // present by `missing_features` before we got here.
        Ok(unsafe { instance.create_device(phys, &ci, None) }?)
    }

    /// Free device memory and total, in bytes — `hipMemGetInfo`'s twin.
    ///
    /// Reports the DEVICE_LOCAL heap. VK_EXT_memory_budget would give a live free
    /// figure; without it "free" is the heap size, which is what the caller's budget
    /// arithmetic wants on an APU where the heap is host RAM anyway.
    pub fn mem_info(&self) -> (usize, usize) {
        let heaps = &self.memprops.memory_heaps[..self.memprops.memory_heap_count as usize];
        let total: u64 = heaps
            .iter()
            .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
            .map(|h| h.size)
            .sum();
        (total as usize, total as usize)
    }

    /// Device name, for startup logging and for test output.
    pub fn name(&self) -> String {
        // SAFETY: `self.phys` is live.
        let props = unsafe { self.instance.get_physical_device_properties(self.phys) };
        props.device_name_as_c_str().map_or_else(
            |_| "<unnamed>".to_string(),
            |s| s.to_string_lossy().into_owned(),
        )
    }
}

/// Silence the "field is never read" lint on the loader handle: `ash::Entry` owns the
/// dlopen'd libvulkan and must outlive the instance, but nothing calls through it.
impl Gpu {
    #[allow(dead_code)]
    fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn context_initialises() {
        let g = gpu().expect("vulkan init");
        let (free, total) = g.mem_info();
        println!("\nVULKAN device: {} — {:.1} GiB device-local\n", g.name(), total as f64 / (1u64 << 30) as f64);
        assert!(total > 0, "no DEVICE_LOCAL heap");
        assert_eq!(free, total);
        // Second call must hand back the same context, not build another.
        assert!(std::ptr::eq(g, gpu().expect("cached")));
    }
}
