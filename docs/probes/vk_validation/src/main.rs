//! Do the Vulkan validation checkers actually fire on this machine?
//!
//! rivoli's Vulkan backend argues its own correctness largely from the validation layer
//! reporting nothing. That argument is void if a checker is loaded but inert — and two
//! of the three are OFF BY DEFAULT, so a clean run under the default configuration says
//! nothing at all about the inter-dispatch barrier or about device-address accesses.
//!
//! Each mode injects one deliberate fault and reports whether the expected diagnostic
//! came back. Exit 0 only if it did; a NON-ZERO EXIT IS THE FINDING. See ../README.md.
//!
//!     cargo run -- core
//!     VK_LAYER_VALIDATE_SYNC=1 cargo run -- sync
//!     VK_LAYER_GPUAV_ENABLE=1  cargo run -- gpuav
//!
//! Hand-run diagnostic: panicking on setup failure is the correct behaviour, and the
//! lint bans that apply to the engine deliberately do not apply here.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ash::vk;
use std::ffi::CStr;
use std::sync::Mutex;

const LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

/// Every message the layer produced, id and body. A Vec rather than a counter because
/// the ID and the text live in different fields and an earlier version of this probe
/// reported the OPPOSITE OF THE TRUTH by matching only the body — `SYNC-HAZARD` is in
/// `pMessageIdName`. Keep the raw text; never let the match be the only thing you read.
static MESSAGES: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

unsafe extern "system" fn cb(
    _sev: vk::DebugUtilsMessageSeverityFlagsEXT,
    _ty: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let d = unsafe { &*data };
    let id = unsafe { d.message_id_name_as_c_str() }
        .unwrap_or(c"")
        .to_string_lossy()
        .into_owned();
    let msg = unsafe { d.message_as_c_str() }
        .unwrap_or(c"<none>")
        .to_string_lossy()
        .into_owned();
    eprintln!("  [layer] {id}\n          {}", msg.replace('\n', "\n          "));
    MESSAGES.lock().unwrap().push((id, msg));
    vk::FALSE
}

/// Did anything the layer said mention `needle`, in either the ID or the body?
fn saw(needle: &str) -> bool {
    let n = needle.to_lowercase();
    MESSAGES
        .lock()
        .unwrap()
        .iter()
        .any(|(id, msg)| id.to_lowercase().contains(&n) || msg.to_lowercase().contains(&n))
}

struct Ctx {
    /// Owns the dlopen'd libvulkan, and MUST outlive every handle below: dropping it
    /// dlcloses the library, so the next call through a function pointer into it
    /// segfaults. Found the hard way — an earlier version of this probe let `setup`
    /// drop the Entry, and `sync` died with SIGSEGV while `core` survived on luck,
    /// because the loaded layer happened to be holding the refcount up.
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    phys: vk::PhysicalDevice,
    queue: vk::Queue,
    qfam: u32,
}

fn setup(bda: bool) -> Ctx {
    let entry = unsafe { ash::Entry::load() }.expect("no Vulkan loader");
    let layers = unsafe { entry.enumerate_instance_layer_properties() }.expect("layers");
    assert!(
        layers.iter().any(|l| l.layer_name_as_c_str() == Ok(LAYER)),
        "VK_LAYER_KHRONOS_validation is not installed — nothing to probe"
    );
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
    let lnames = [LAYER.as_ptr()];
    let enames = [ash::ext::debug_utils::NAME.as_ptr()];
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_layer_names(&lnames)
                .enabled_extension_names(&enames),
            None,
        )
    }
    .expect("instance");

    use vk::DebugUtilsMessageSeverityFlagsEXT as Sev;
    use vk::DebugUtilsMessageTypeFlagsEXT as Ty;
    let dbg = ash::ext::debug_utils::Instance::new(&entry, &instance);
    unsafe {
        dbg.create_debug_utils_messenger(
            &vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(Sev::ERROR | Sev::WARNING)
                .message_type(Ty::GENERAL | Ty::VALIDATION | Ty::PERFORMANCE)
                .pfn_user_callback(Some(cb)),
            None,
        )
    }
    .expect("messenger");
    // Leaked on purpose: it must outlive every other object, and the process is about
    // to exit anyway.

    let phys = unsafe { instance.enumerate_physical_devices() }.expect("phys")[0];
    let fams = unsafe { instance.get_physical_device_queue_family_properties(phys) };
    let qfam = fams
        .iter()
        .position(|f| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .expect("a COMPUTE queue family") as u32;
    let prio = [1.0f32];
    let qci = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(qfam)
        .queue_priorities(&prio)];
    let mut v12 = vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(bda);
    let core = vk::PhysicalDeviceFeatures::default().shader_int64(bda);
    let mut ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(&qci)
        .enabled_features(&core);
    if bda {
        ci = ci.push_next(&mut v12);
    }
    let device = unsafe { instance.create_device(phys, &ci, None) }.expect("device");
    let queue = unsafe { device.get_device_queue(qfam, 0) };
    Ctx {
        _entry: entry,
        instance,
        device,
        phys,
        queue,
        qfam,
    }
}

fn memtype(c: &Ctx, bits: u32, want: vk::MemoryPropertyFlags) -> u32 {
    let mp = unsafe { c.instance.get_physical_device_memory_properties(c.phys) };
    (0..mp.memory_type_count)
        .find(|&i| {
            bits & (1 << i) != 0 && mp.memory_types[i as usize].property_flags.contains(want)
        })
        .expect("a suitable memory type")
}

/// One-shot command buffer, recorded by `f`, submitted and waited.
fn submit(c: &Ctx, f: impl FnOnce(vk::CommandBuffer)) {
    unsafe {
        let pool = c
            .device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(c.qfam),
                None,
            )
            .expect("pool");
        let cmd = c
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("cmd")[0];
        c.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin");
        f(cmd);
        c.device.end_command_buffer(cmd).expect("end");
        let bufs = [cmd];
        let fence = c
            .device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("fence");
        c.device
            .queue_submit(
                c.queue,
                &[vk::SubmitInfo::default().command_buffers(&bufs)],
                fence,
            )
            .expect("submit");
        let _ = c.device.wait_for_fences(&[fence], true, 5_000_000_000);
        let _ = c.device.device_wait_idle();
    }
}

/// CORE validation, on by default. A zero-sized buffer is the cheapest unambiguous VUID.
fn probe_core() -> &'static str {
    let c = setup(false);
    println!("fault: vkCreateBuffer(size = 0)");
    let _ = unsafe {
        c.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(0)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
            None,
        )
    };
    "VUID-VkBufferCreateInfo-size-00912"
}

/// SYNCHRONISATION validation, OPT-IN. Two overlapping writes with nothing between
/// them — the same class of mistake as omitting rivoli's inter-dispatch barrier, and
/// the reason a default-config clean run says nothing about that barrier.
fn probe_sync() -> &'static str {
    let c = setup(false);
    unsafe {
        let buf = c
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(4096)
                    .usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .expect("buf");
        let req = c.device.get_buffer_memory_requirements(buf);
        let mem = c
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memtype(
                        &c,
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    )),
                None,
            )
            .expect("mem");
        c.device.bind_buffer_memory(buf, mem, 0).expect("bind");
        println!("fault: two overlapping vkCmdFillBuffer, no barrier between them");
        submit(&c, |cmd| {
            c.device.cmd_fill_buffer(cmd, buf, 0, 4096, 0xAAAA_AAAA);
            c.device.cmd_fill_buffer(cmd, buf, 0, 4096, 0x5555_5555);
        });
    }
    "SYNC-HAZARD-WRITE-AFTER-WRITE"
}

/// GPU-ASSISTED validation, OPT-IN, and the one that matters most here: it is the only
/// checker that can see a bad DEVICE ADDRESS, because the address is an opaque uint64
/// with no object behind it. Injects rivoli's exact access shape.
fn probe_gpuav() -> &'static str {
    const BUF_BYTES: u64 = 256;
    const BAD_INDEX: u32 = 1 << 20;
    let c = setup(true);
    unsafe {
        let buf = c
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(BUF_BYTES).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
                None,
            )
            .expect("buf");
        let req = c.device.get_buffer_memory_requirements(buf);
        let mut mf =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mem = c
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memtype(
                        &c,
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ))
                    .push_next(&mut mf),
                None,
            )
            .expect("mem");
        c.device.bind_buffer_memory(buf, mem, 0).expect("bind");
        let addr = c
            .device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));

        let spv = include_bytes!(concat!(env!("OUT_DIR"), "/oob.spv"));
        let words = ash::util::read_spv(&mut std::io::Cursor::new(&spv[..])).expect("spv");
        let module = c
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("module");
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let layout = c
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                None,
            )
            .expect("layout");
        let pipe = c
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| e)
            .expect("pipeline")[0];

        #[repr(C)]
        struct Push {
            addr: u64,
            idx: u32,
            _pad: u32,
        }
        let push = Push {
            addr,
            idx: BAD_INDEX,
            _pad: 0,
        };
        let bytes =
            std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>());
        println!(
            "fault: store at float index {BAD_INDEX} of a {BUF_BYTES} B allocation \
             ({} B past the end), through a buffer reference",
            (BAD_INDEX as u64) * 4 - BUF_BYTES
        );
        submit(&c, |cmd| {
            c.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            c.device
                .cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            c.device.cmd_dispatch(cmd, 1, 1, 1);
        });
    }
    "VUID-RuntimeSpirv-PhysicalStorageBuffer64-11819"
}

/// SYNCHRONISATION validation, COMPUTE -> TRANSFER specifically.
///
/// `sync` proves the checker sees transfer-vs-transfer hazards. That does NOT imply it
/// sees a compute write followed by a transfer read, which is the dependency
/// `memcpy_dtod` (arena compaction reading a slot a kernel just wrote) actually needs.
/// A different hazard class can have different coverage, and rivoli's suite came back
/// clean on this pattern with a barrier that scopes only COMPUTE -> COMPUTE — either
/// the barrier is enough or the checker is blind here, and those are very different
/// facts. This mode settles it by removing the barrier ENTIRELY: if that is silent too,
/// the checker cannot speak to this pattern and its silence proves nothing.
fn probe_compute_copy() -> &'static str {
    let c = setup(true);
    unsafe {
        let mk = |size: u64, usage: vk::BufferUsageFlags| {
            let b = c
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .expect("buf");
            let req = c.device.get_buffer_memory_requirements(b);
            let mut mf = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let m = c
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(memtype(
                            &c,
                            req.memory_type_bits,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        ))
                        .push_next(&mut mf),
                    None,
                )
                .expect("mem");
            c.device.bind_buffer_memory(b, m, 0).expect("bind");
            b
        };
        let src = mk(
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        let dst = mk(256, vk::BufferUsageFlags::TRANSFER_DST);
        let addr = c
            .device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(src));

        let spv = include_bytes!(concat!(env!("OUT_DIR"), "/oob.spv"));
        let words = ash::util::read_spv(&mut std::io::Cursor::new(&spv[..])).expect("spv");
        let module = c
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("module");
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let layout = c
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                None,
            )
            .expect("layout");
        let pipe = c
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| e)
            .expect("pipeline")[0];

        #[repr(C)]
        struct Push {
            addr: u64,
            idx: u32,
            _pad: u32,
        }
        // idx = 0: an ordinary IN-BOUNDS store. This mode is about ordering, not OOB.
        let push = Push {
            addr,
            idx: 0,
            _pad: 0,
        };
        let bytes = std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>());
        println!("fault: compute stores to a buffer, then vkCmdCopyBuffer READS it, NO barrier");
        submit(&c, |cmd| {
            c.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            c.device
                .cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            c.device.cmd_dispatch(cmd, 1, 1, 1);
            let region = [vk::BufferCopy::default().size(256)];
            c.device.cmd_copy_buffer(cmd, src, dst, &region);
        });
    }
    "SYNC-HAZARD-READ-AFTER-WRITE"
}

/// SYNCHRONISATION validation, COMPUTE -> COMPUTE — the cell `Gpu::enqueue`'s barrier
/// actually lives in, and the one `chained_dispatch_respects_the_barrier` claims a
/// clean result for.
///
/// That claim is only worth something if the checker can see this hazard class. It
/// fires on transfer↔transfer and is silent on compute→transfer, so its behaviour here
/// cannot be inferred from either: **a checker's envelope has to be established per
/// hazard class you intend to rely on, not once for the checker.**
///
/// Two dispatches of a read-modify-write on the same address, no barrier between them.
fn probe_compute_compute() -> &'static str {
    let c = setup(true);
    unsafe {
        let buf = c
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(256).usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                ),
                None,
            )
            .expect("buf");
        let req = c.device.get_buffer_memory_requirements(buf);
        let mut mf =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mem = c
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memtype(
                        &c,
                        req.memory_type_bits,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    ))
                    .push_next(&mut mf),
                None,
            )
            .expect("mem");
        c.device.bind_buffer_memory(buf, mem, 0).expect("bind");
        let addr = c
            .device
            .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buf));

        let spv = include_bytes!(concat!(env!("OUT_DIR"), "/oob.spv"));
        let words = ash::util::read_spv(&mut std::io::Cursor::new(&spv[..])).expect("spv");
        let module = c
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("module");
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(16)];
        let layout = c
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges),
                None,
            )
            .expect("layout");
        let pipe = c
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| e)
            .expect("pipeline")[0];

        #[repr(C)]
        struct Push {
            addr: u64,
            idx: u32,
            _pad: u32,
        }
        let push = Push {
            addr,
            idx: 0,
            _pad: 0,
        };
        let bytes = std::slice::from_raw_parts((&raw const push).cast::<u8>(), size_of::<Push>());
        println!("fault: two dispatches read-modify-writing the same address, NO barrier");
        submit(&c, |cmd| {
            c.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            c.device
                .cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            c.device.cmd_dispatch(cmd, 1, 1, 1);
            // Nothing between them. The second dispatch reads what the first wrote.
            c.device.cmd_dispatch(cmd, 1, 1, 1);
        });
    }
    // Either RAW or WAW is a valid catch for a read-modify-write pair, so match the
    // family rather than one label.
    "SYNC-HAZARD"
}

/// The discriminator for [`probe_compute_copy`]: the SAME compute-write-then-copy
/// hazard, but the shader reaches the buffer through an ordinary DESCRIPTOR BINDING.
///
/// If this reports and the buffer-reference version does not, then synchronisation
/// validation's blind spot is buffer device addresses specifically — not
/// compute-to-transfer ordering — and since bare device addresses are how rivoli
/// passes every buffer to every kernel, sync validation can see almost nothing this
/// backend does.
fn probe_compute_copy_descriptor() -> &'static str {
    let c = setup(false);
    unsafe {
        let mk = |size: u64, usage: vk::BufferUsageFlags| {
            let b = c
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(size).usage(usage),
                    None,
                )
                .expect("buf");
            let req = c.device.get_buffer_memory_requirements(b);
            let m = c
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(req.size)
                        .memory_type_index(memtype(
                            &c,
                            req.memory_type_bits,
                            vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        )),
                    None,
                )
                .expect("mem");
            c.device.bind_buffer_memory(b, m, 0).expect("bind");
            b
        };
        let src = mk(
            256,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
        );
        let dst = mk(256, vk::BufferUsageFlags::TRANSFER_DST);

        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        let dsl = c
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                None,
            )
            .expect("dsl");
        let sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)];
        let dpool = c
            .device
            .create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
            .expect("dpool");
        let dsls = [dsl];
        let set = c
            .device
            .allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(dpool)
                    .set_layouts(&dsls),
            )
            .expect("set")[0];
        let binfo = [vk::DescriptorBufferInfo::default()
            .buffer(src)
            .offset(0)
            .range(vk::WHOLE_SIZE)];
        c.device.update_descriptor_sets(
            &[vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&binfo)],
            &[],
        );

        let spv = include_bytes!(concat!(env!("OUT_DIR"), "/descwrite.spv"));
        let words = ash::util::read_spv(&mut std::io::Cursor::new(&spv[..])).expect("spv");
        let module = c
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
            .expect("module");
        let layout = c
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&dsls),
                None,
            )
            .expect("layout");
        let pipe = c
            .device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(
                        vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(module)
                            .name(c"main"),
                    )
                    .layout(layout)],
                None,
            )
            .map_err(|(_, e)| e)
            .expect("pipeline")[0];

        println!(
            "fault: compute stores via a DESCRIPTOR, then vkCmdCopyBuffer READS it, NO barrier"
        );
        submit(&c, |cmd| {
            c.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipe);
            c.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[set],
                &[],
            );
            c.device.cmd_dispatch(cmd, 1, 1, 1);
            let region = [vk::BufferCopy::default().size(256)];
            c.device.cmd_copy_buffer(cmd, src, dst, &region);
        });
    }
    "SYNC-HAZARD-READ-AFTER-WRITE"
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    // Warn rather than fail: the point of the probe is to find out whether the checker
    // is on, and silently requiring the env var would just move the guesswork.
    let env_hint = |var: &str| {
        if std::env::var(var).is_err() {
            eprintln!("NOTE: {var} is not set — this checker is off by default, so a \
                       'not observed' result below is expected and means nothing.");
        }
    };
    let expect = match mode.as_str() {
        "core" => probe_core(),
        "sync" => {
            env_hint("VK_LAYER_VALIDATE_SYNC");
            probe_sync()
        }
        "gpuav" => {
            env_hint("VK_LAYER_GPUAV_ENABLE");
            probe_gpuav()
        }
        "compute-compute" => {
            env_hint("VK_LAYER_VALIDATE_SYNC");
            probe_compute_compute()
        }
        "compute-copy" => {
            env_hint("VK_LAYER_VALIDATE_SYNC");
            probe_compute_copy()
        }
        "compute-copy-desc" => {
            env_hint("VK_LAYER_VALIDATE_SYNC");
            probe_compute_copy_descriptor()
        }
        _ => {
            eprintln!(
                "usage: vk_validation_probe \
                 core|sync|gpuav|compute-compute|compute-copy|compute-copy-desc\n\
                 (see ../README.md — the sync modes map a COVERAGE MATRIX, not a pass/fail)"
            );
            std::process::exit(2);
        }
    };

    let n = MESSAGES.lock().unwrap().len();
    let hit = saw(expect);
    println!("\nexpected : {expect}");
    println!("messages : {n}");
    println!("observed : {hit}");
    if hit {
        println!("VERDICT  : {mode} validation FIRES. Its silence elsewhere is evidence.");
    } else {
        println!(
            "VERDICT  : {mode} validation did NOT report the injected fault. It is not \
             watching — withdraw anything concluded from its silence."
        );
        std::process::exit(1);
    }
}
