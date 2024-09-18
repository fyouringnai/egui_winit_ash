#![allow(deprecated)]
use ash::ext::debug_utils;
use ash::khr::{surface, swapchain};
use ash::vk::*;
use ash::{Device, Entry, Instance};
use egui::{containers, include_image, Vec2};
use egui_winit_ash::integration::EguiAsh;
use egui_winit_ash::utils::EguiAshCreateInfoBuilder;
use log::{error, info, trace, warn};
use std::borrow::Cow;
use std::ffi::CStr;
use std::os::raw::c_void;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalSize, Size};
use winit::event::{StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use winit::window::{Window, WindowAttributes, WindowId};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: DebugUtilsMessageSeverityFlagsEXT,
    message_type: DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut c_void,
) -> Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    let type_name = match message_type {
        DebugUtilsMessageTypeFlagsEXT::GENERAL => "GENERAL",
        DebugUtilsMessageTypeFlagsEXT::VALIDATION => "VALIDATION",
        DebugUtilsMessageTypeFlagsEXT::PERFORMANCE => "PERFORMANCE",
        _ => "Unknown",
    };

    match message_severity {
        DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            trace!(
                "{}",
                format!(
                    "[VERBOSE] [{}] {} ({}): {}",
                    type_name, message_id_name, message_id_number, message
                )
            );
        }
        DebugUtilsMessageSeverityFlagsEXT::INFO => {
            info!(
                "{}",
                format!(
                    "[INFO] [{}] {} ({}): {}",
                    type_name, message_id_name, message_id_number, message
                )
            );
        }
        DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!(
                "{}",
                format!(
                    "[WARNING] [{}] {} ({}): {}",
                    type_name, message_id_name, message_id_number, message
                )
            );
        }
        DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!(
                "{}",
                format!(
                    "[ERROR] [{}] {} ({}): {}",
                    type_name, message_id_name, message_id_number, message
                )
            );
        }
        _ => {}
    }

    FALSE
}

#[derive(Debug)]
enum UserEvent {
    Redraw(Duration),
}

#[derive(Default)]
struct App {
    window: Option<Arc<Window>>,
    gui: Option<Arc<RwLock<EguiAsh>>>,
    event_loop_proxy: Option<Arc<Mutex<EventLoopProxy<UserEvent>>>>,

    entry: Option<Arc<Entry>>,
    instance: Option<Arc<Instance>>,
    device: Option<Arc<Device>>,
    physical_device: PhysicalDevice,
    queues: Vec<Queue>,
    graphics_queue: Queue,
    present_queue: Queue,

    queue_family_indices: Vec<u32>,
    graphics_family_index: u32,
    present_family_index: u32,

    debug_utils_loader: Option<Arc<debug_utils::Instance>>,
    debug_utils_messenger: Option<DebugUtilsMessengerEXT>,

    swapchain_loader: Option<Arc<swapchain::Device>>,
    swapchain: SwapchainKHR,
    images: Vec<Image>,
    image_views: Vec<ImageView>,
    framebuffers: Arc<RwLock<Vec<Framebuffer>>>,
    format: Format,
    extent: Extent2D,
    pipeline_layout: PipelineLayout,
    graphics_pipeline: Pipeline,
    render_pass: RenderPass,

    command_pools: Vec<CommandPool>,
    command_buffers: Vec<CommandBuffer>,

    images_available: Vec<Semaphore>,
    render_finished: Vec<Semaphore>,
    in_flight_fences: Vec<Fence>,
    images_in_flight: Vec<Fence>,

    surface: SurfaceKHR,
    surface_loader: Option<Arc<surface::Instance>>,

    resized: bool,
    frame: usize,
    max_images_in_flight: usize,
    repaint_delay: Duration,
}

impl App {
    fn create_swapchain(&mut self) {
        let surface_loader = self.surface_loader.as_ref().unwrap();
        let swapchain_loader = self.swapchain_loader.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();

        let capabilities = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(self.physical_device, self.surface)
                .unwrap()
        };

        let present_modes = unsafe {
            surface_loader
                .get_physical_device_surface_present_modes(self.physical_device, self.surface)
                .unwrap()
        };

        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(self.physical_device, self.surface)
                .unwrap()
        };

        let image_count = if capabilities.max_image_count == 0 {
            capabilities.min_image_count + 1
        } else {
            capabilities
                .max_image_count
                .min(capabilities.min_image_count + 1)
        };

        let mut queue_family_indices = vec![];

        let sharing_mode = if self.present_family_index != self.graphics_family_index {
            queue_family_indices.push(self.present_family_index);
            queue_family_indices.push(self.graphics_family_index);
            SharingMode::CONCURRENT
        } else {
            SharingMode::EXCLUSIVE
        };

        let format = formats
            .iter()
            .find(|format| {
                format.format == Format::B8G8R8A8_SRGB && format.color_space == Default::default()
            })
            .unwrap_or(&formats[0])
            .clone();

        let extent = if capabilities.current_extent.width == u32::MAX {
            let size = capabilities.current_extent;
            Extent2D {
                width: size.width.clamp(
                    capabilities.min_image_extent.width,
                    capabilities.max_image_extent.width,
                ),
                height: size.height.clamp(
                    capabilities.min_image_extent.height,
                    capabilities.max_image_extent.height,
                ),
            }
        } else {
            capabilities.current_extent
        };

        let present_mode = if present_modes.contains(&PresentModeKHR::MAILBOX) {
            PresentModeKHR::MAILBOX
        } else {
            PresentModeKHR::FIFO
        };

        let swapchain_info = SwapchainCreateInfoKHR {
            surface: self.surface,
            min_image_count: image_count,
            image_format: format.format,
            image_color_space: format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: ImageUsageFlags::COLOR_ATTACHMENT,
            image_sharing_mode: sharing_mode,
            queue_family_index_count: queue_family_indices.len() as u32,
            p_queue_family_indices: queue_family_indices.as_ptr(),
            pre_transform: capabilities.current_transform,
            composite_alpha: CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: TRUE,
            ..Default::default()
        };

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_info, None)
                .unwrap()
        };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain).unwrap() };

        let image_views = images
            .iter()
            .map(|image| {
                let components = ComponentMapping {
                    r: ComponentSwizzle::IDENTITY,
                    g: ComponentSwizzle::IDENTITY,
                    b: ComponentSwizzle::IDENTITY,
                    a: ComponentSwizzle::IDENTITY,
                };

                let subresource = ImageSubresourceRange {
                    aspect_mask: ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };

                let image_view_info = ImageViewCreateInfo {
                    image: *image,
                    view_type: ImageViewType::TYPE_2D,
                    format: format.format,
                    components,
                    subresource_range: subresource,
                    ..Default::default()
                };

                unsafe { device.create_image_view(&image_view_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        self.swapchain = swapchain;
        self.images = images;
        self.image_views = image_views;
        self.format = format.format;
        self.extent = extent;
    }

    fn create_render_pass(&mut self) {
        let device = self.device.as_ref().unwrap();
        let description = AttachmentDescription {
            format: self.format,
            samples: SampleCountFlags::TYPE_1,
            load_op: AttachmentLoadOp::CLEAR,
            store_op: AttachmentStoreOp::STORE,
            stencil_load_op: AttachmentLoadOp::DONT_CARE,
            stencil_store_op: AttachmentStoreOp::DONT_CARE,
            initial_layout: ImageLayout::UNDEFINED,
            final_layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let reference = AttachmentReference {
            attachment: 0,
            layout: ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        };

        let subpass_description = SubpassDescription {
            pipeline_bind_point: PipelineBindPoint::GRAPHICS,
            color_attachment_count: 1,
            p_color_attachments: [reference].as_ptr(),
            ..Default::default()
        };

        let dependency = SubpassDependency {
            src_subpass: SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            src_access_mask: AccessFlags::empty(),
            dst_stage_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        };

        let render_pass_info = RenderPassCreateInfo {
            attachment_count: 1,
            p_attachments: [description].as_ptr(),
            subpass_count: 1,
            p_subpasses: [subpass_description].as_ptr(),
            dependency_count: 1,
            p_dependencies: [dependency].as_ptr(),
            ..Default::default()
        };

        let render_pass = unsafe { device.create_render_pass(&render_pass_info, None).unwrap() };

        self.render_pass = render_pass;
    }

    fn create_pipeline(&mut self) {
        let device = self.device.as_ref().unwrap();

        let vert = include_bytes!("shader/image_vert.spv");
        let frag = include_bytes!("shader/image_frag.spv");

        let vert_info = ShaderModuleCreateInfo {
            code_size: vert.len(),
            p_code: vert.as_ptr() as *const _,
            ..Default::default()
        };

        let frag_info = ShaderModuleCreateInfo {
            code_size: frag.len(),
            p_code: frag.as_ptr() as *const _,
            ..Default::default()
        };

        let vert_module = unsafe { device.create_shader_module(&vert_info, None).unwrap() };
        let frag_module = unsafe { device.create_shader_module(&frag_info, None).unwrap() };

        let vert_stage_info = PipelineShaderStageCreateInfo {
            stage: ShaderStageFlags::VERTEX,
            module: vert_module,
            p_name: CStr::from_bytes_with_nul(b"main\0").unwrap().as_ptr(),
            ..Default::default()
        };

        let frag_stage_info = PipelineShaderStageCreateInfo {
            stage: ShaderStageFlags::FRAGMENT,
            module: frag_module,
            p_name: CStr::from_bytes_with_nul(b"main\0").unwrap().as_ptr(),
            ..Default::default()
        };

        let vertex_input_info = PipelineVertexInputStateCreateInfo::default();

        let input_assembly_info = PipelineInputAssemblyStateCreateInfo {
            topology: PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: FALSE,
            ..Default::default()
        };

        let viewport = Viewport::default();

        let scissor = Rect2D::default();

        let viewport_info = PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: [viewport].as_ptr(),
            scissor_count: 1,
            p_scissors: [scissor].as_ptr(),
            ..Default::default()
        };

        let rasterization_info = PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: FALSE,
            rasterizer_discard_enable: FALSE,
            polygon_mode: PolygonMode::FILL,
            line_width: 1.0,
            cull_mode: CullModeFlags::NONE,
            front_face: FrontFace::COUNTER_CLOCKWISE,
            depth_bias_enable: FALSE,
            ..Default::default()
        };

        let multisample_info = PipelineMultisampleStateCreateInfo {
            sample_shading_enable: FALSE,
            rasterization_samples: SampleCountFlags::TYPE_1,
            ..Default::default()
        };

        let attachment_state = PipelineColorBlendAttachmentState {
            color_write_mask: ColorComponentFlags::RGBA,
            blend_enable: FALSE,
            ..Default::default()
        };

        let color_blend_info = PipelineColorBlendStateCreateInfo {
            logic_op_enable: FALSE,
            logic_op: LogicOp::COPY,
            attachment_count: 1,
            p_attachments: [attachment_state].as_ptr(),
            blend_constants: [0.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };

        let dynamic_states = [DynamicState::VIEWPORT, DynamicState::SCISSOR];

        let dynamic_state_info = PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout_info = PipelineLayoutCreateInfo::default();

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let pipeline_info = GraphicsPipelineCreateInfo {
            stage_count: 2,
            p_stages: [vert_stage_info, frag_stage_info].as_ptr(),
            p_vertex_input_state: [vertex_input_info].as_ptr(),
            p_input_assembly_state: [input_assembly_info].as_ptr(),
            p_viewport_state: [viewport_info].as_ptr(),
            p_rasterization_state: [rasterization_info].as_ptr(),
            p_multisample_state: [multisample_info].as_ptr(),
            p_color_blend_state: [color_blend_info].as_ptr(),
            layout: pipeline_layout,
            render_pass: self.render_pass,
            subpass: 0,
            base_pipeline_index: -1,
            p_dynamic_state: [dynamic_state_info].as_ptr(),
            ..Default::default()
        };

        let graphics_pipeline = unsafe {
            device
                .create_graphics_pipelines(PipelineCache::null(), &[pipeline_info], None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        self.pipeline_layout = pipeline_layout;
        self.graphics_pipeline = graphics_pipeline;
    }

    fn create_framebuffers(&mut self) {
        let device = self.device.as_ref().unwrap();
        let framebuffers = self
            .image_views
            .iter()
            .map(|image_view| {
                let framebuffer_info = FramebufferCreateInfo {
                    render_pass: self.render_pass,
                    attachment_count: 1,
                    p_attachments: [*image_view].as_ptr(),
                    width: self.extent.width,
                    height: self.extent.height,
                    layers: 1,
                    ..Default::default()
                };

                unsafe { device.create_framebuffer(&framebuffer_info, None).unwrap() }
            })
            .collect::<Vec<_>>();

        let mut f = self.framebuffers.write().unwrap();
        f.clear();
        f.extend(framebuffers);
        self.max_images_in_flight = f.len() - 1;
    }

    fn create_command_pools(&mut self) {
        let image_count = self.framebuffers.read().unwrap().len();
        self.command_pools.clear();

        let command_pool_info = CommandPoolCreateInfo {
            queue_family_index: self.queue_family_indices[0],
            flags: CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };

        (0..image_count).for_each(|_| {
            let command_pool = unsafe {
                self.device
                    .as_ref()
                    .unwrap()
                    .create_command_pool(&command_pool_info, None)
                    .unwrap()
            };

            self.command_pools.push(command_pool);
        });
    }

    fn create_command_buffers(&mut self) {
        let device = self.device.clone().unwrap();
        let image_count = self.framebuffers.read().unwrap().len();
        self.command_buffers.clear();

        (0..image_count).for_each(|index| {
            let command_buffer_info = CommandBufferAllocateInfo {
                command_pool: self.command_pools[index],
                level: CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };

            let command_buffer = unsafe {
                device
                    .allocate_command_buffers(&command_buffer_info)
                    .unwrap()[0]
            };

            self.command_buffers.push(command_buffer);

            self.record_command_buffer(index);
        });
    }

    fn record_command_buffer(&mut self, index: usize) {
        let device = self.device.as_ref().unwrap();
        let command_buffer = self.command_buffers[index];

        let begin_info = CommandBufferBeginInfo {
            flags: CommandBufferUsageFlags::SIMULTANEOUS_USE,
            ..Default::default()
        };

        let clear_values = [ClearValue {
            color: ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];

        let render_pass_info = RenderPassBeginInfo {
            render_pass: self.render_pass,
            framebuffer: self.framebuffers.read().unwrap()[index],
            render_area: Rect2D {
                offset: Offset2D { x: 0, y: 0 },
                extent: self.extent,
            },
            clear_value_count: clear_values.len() as u32,
            p_clear_values: clear_values.as_ptr(),
            ..Default::default()
        };

        let viewport = Viewport {
            x: 0.0,
            y: 0.0,
            width: self.extent.width as f32,
            height: self.extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };

        let scissor = Rect2D {
            offset: Offset2D { x: 0, y: 0 },
            extent: self.extent,
        };

        unsafe {
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin command buffer");

            device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(
                command_buffer,
                PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline,
            );

            device.cmd_set_viewport(command_buffer, 0, &[viewport]);

            device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            device.cmd_end_render_pass(command_buffer);

            device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    fn create_sync_objects(&mut self) {
        let device = self.device.as_ref().unwrap();
        let image_count = self.framebuffers.read().unwrap().len();
        self.images_available.clear();
        self.render_finished.clear();
        self.in_flight_fences.clear();
        self.images_in_flight.clear();

        (0..self.max_images_in_flight).for_each(|_| {
            let images_available = unsafe {
                device
                    .create_semaphore(&SemaphoreCreateInfo::default(), None)
                    .unwrap()
            };

            let render_finished = unsafe {
                device
                    .create_semaphore(&SemaphoreCreateInfo::default(), None)
                    .unwrap()
            };

            let in_flight_fence = unsafe {
                device
                    .create_fence(
                        &FenceCreateInfo {
                            flags: FenceCreateFlags::SIGNALED,
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap()
            };

            self.images_available.push(images_available);
            self.render_finished.push(render_finished);
            self.in_flight_fences.push(in_flight_fence);
        });
        self.images_in_flight = vec![Fence::null(); image_count];
    }

    fn recreate_swapchain(&mut self) {
        let device = self.device.as_ref().unwrap();

        unsafe {
            device.device_wait_idle().unwrap();
        }

        self.destroy_swapchain();

        self.create_swapchain();
        self.create_framebuffers();
        self.create_command_buffers();
        self.create_sync_objects();
    }

    fn destroy_swapchain(&mut self) {
        let device = self.device.as_ref().unwrap();
        let swapchain_loader = self.swapchain_loader.as_ref().unwrap();

        self.framebuffers
            .read()
            .unwrap()
            .iter()
            .for_each(|framebuffer| unsafe {
                device.destroy_framebuffer(*framebuffer, None);
            });

        self.images_available
            .iter()
            .for_each(|semaphore| unsafe { device.destroy_semaphore(*semaphore, None) });

        self.render_finished.iter().for_each(|semaphore| unsafe {
            device.destroy_semaphore(*semaphore, None);
        });

        self.in_flight_fences.iter().for_each(|fence| unsafe {
            device.destroy_fence(*fence, None);
        });

        self.image_views.iter().for_each(|image_view| unsafe {
            device.destroy_image_view(*image_view, None);
        });

        unsafe {
            swapchain_loader.destroy_swapchain(self.swapchain, None);
        }
    }

    fn destroy(&mut self) {
        let device = self.device.clone().unwrap();
        let surface_loader = self.surface_loader.clone().unwrap();
        let gui = self.gui.clone().unwrap();

        unsafe {
            device.device_wait_idle().unwrap();
        }

        gui.write().unwrap().destroy();

        self.destroy_swapchain();

        unsafe {
            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_render_pass(self.render_pass, None);

            self.command_pools
                .iter()
                .enumerate()
                .for_each(|(i, command_pool)| {
                    device.free_command_buffers(*command_pool, &[self.command_buffers[i]]);
                    device.destroy_command_pool(*command_pool, None);
                });
        }

        if VALIDATION_ENABLED {
            let debug_utils_loader = self.debug_utils_loader.as_ref().unwrap();
            let debug_utils_messenger = self.debug_utils_messenger.as_ref().unwrap();

            unsafe {
                debug_utils_loader.destroy_debug_utils_messenger(*debug_utils_messenger, None);
            }
        }

        unsafe {
            device.destroy_device(None);
        }

        let instance = self.instance.as_ref().unwrap();
        let entry = self.entry.clone().unwrap();

        unsafe {
            surface_loader.destroy_surface(self.surface, None);
            instance.destroy_instance(None);
            drop(entry);
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: StartCause) {
        if let Some(window) = self.window.as_ref() {
            match cause {
                StartCause::Init => {}
                StartCause::ResumeTimeReached { .. } => {
                    window.request_redraw();
                }
                StartCause::WaitCancelled { .. } => {}
                StartCause::Poll => {}
            }
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attributes = WindowAttributes::default()
            .with_title("Triangle")
            .with_inner_size(Size::Physical(PhysicalSize::new(1920, 1080)));

        let window = Arc::new(event_loop.create_window(attributes).unwrap());

        let entry = Arc::new(Entry::linked());

        let application_info = ApplicationInfo {
            p_application_name: CStr::from_bytes_until_nul(b"Triangle\0").unwrap().as_ptr(),
            application_version: make_api_version(0, 0, 1, 0),
            api_version: API_VERSION_1_3,
            ..Default::default()
        };

        let mut extensions =
            ash_window::enumerate_required_extensions(window.raw_display_handle().unwrap())
                .unwrap()
                .to_vec();

        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .unwrap()
            .iter()
            .map(|layer| unsafe {
                CStr::from_ptr(layer.layer_name.as_ptr())
                    .to_string_lossy()
                    .to_string()
            })
            .collect::<Vec<_>>();

        if VALIDATION_ENABLED
            && !available_layers.contains(&"VK_LAYER_KHRONOS_validation".to_string())
        {
            error!("Validation layer requested but not available. ");
        }

        if VALIDATION_ENABLED {
            extensions.push(EXT_DEBUG_UTILS_NAME.as_ptr());
        }

        let layers = if VALIDATION_ENABLED {
            vec![CStr::from_bytes_with_nul(b"VK_LAYER_KHRONOS_validation\0")
                .unwrap()
                .as_ptr()]
        } else {
            vec![]
        };

        let flags = if cfg!(target_os = "macos") {
            info!("macOS detected. Enabling portability extension.");
            extensions.push(KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_NAME.as_ptr());
            extensions.push(KHR_PORTABILITY_ENUMERATION_NAME.as_ptr());
            InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            InstanceCreateFlags::empty()
        };

        let instance_info = InstanceCreateInfo {
            flags,
            p_application_info: [application_info].as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            ..Default::default()
        };

        let instance = Arc::new(unsafe { entry.create_instance(&instance_info, None) }.unwrap());

        let (debug_utils_loader, debug_utils_messenger) = if VALIDATION_ENABLED {
            let debug_utils_create_info = DebugUtilsMessengerCreateInfoEXT {
                message_severity: DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | DebugUtilsMessageSeverityFlagsEXT::INFO
                    | DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
                message_type: DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                    | DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                pfn_user_callback: Some(vulkan_debug_callback),
                ..Default::default()
            };

            let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
            let debug_utils_messenger = unsafe {
                debug_utils_loader.create_debug_utils_messenger(&debug_utils_create_info, None)
            }
            .unwrap();
            (
                Some(Arc::new(debug_utils_loader)),
                Some(debug_utils_messenger),
            )
        } else {
            (None, None)
        };

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.raw_display_handle().unwrap(),
                window.raw_window_handle().unwrap(),
                None,
            )
        }
        .unwrap();

        let surface_loader = Arc::new(surface::Instance::new(&entry, &instance));

        let suitable_devices = unsafe {
            instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .filter(|physical_device| {
                    let queue_families =
                        instance.get_physical_device_queue_family_properties(*physical_device);

                    queue_families.iter().any(|queue_family| {
                        queue_family.queue_flags.contains(QueueFlags::GRAPHICS)
                            && surface_loader
                                .get_physical_device_surface_support(*physical_device, 0, surface)
                                .unwrap()
                    })
                })
                .collect::<Vec<_>>()
        };

        let physical_device = if let Some(device) = suitable_devices.iter().find(|&device| {
            let properties = unsafe { instance.get_physical_device_properties(*device) };
            properties.device_type == PhysicalDeviceType::DISCRETE_GPU
        }) {
            *device
        } else if let Some(device) = suitable_devices.iter().find(|&device| {
            let properties = unsafe { instance.get_physical_device_properties(*device) };
            properties.device_type == PhysicalDeviceType::INTEGRATED_GPU
        }) {
            *device
        } else {
            suitable_devices[0]
        };

        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) };

        let queue_family_indices = queue_families
            .iter()
            .enumerate()
            .filter_map(|(i, queue_family)| {
                if queue_family.queue_flags.contains(QueueFlags::GRAPHICS)
                    && unsafe {
                        surface_loader
                            .get_physical_device_surface_support(physical_device, i as u32, surface)
                            .unwrap()
                    }
                {
                    Some(i as u32)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let queue_infos = queue_family_indices
            .iter()
            .map(|index| {
                let queue_info = DeviceQueueCreateInfo {
                    queue_family_index: *index,
                    queue_count: 1,
                    p_queue_priorities: [1.0].as_ptr(),
                    ..Default::default()
                };
                queue_info
            })
            .collect::<Vec<_>>();

        let mut extensions = vec![KHR_SWAPCHAIN_NAME.as_ptr()];

        if cfg!(target_os = "macos") {
            extensions.push(KHR_PORTABILITY_SUBSET_NAME.as_ptr());
        }

        let device_info = DeviceCreateInfo {
            flags: Default::default(),
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            enabled_layer_count: layers.len() as u32,
            pp_enabled_layer_names: layers.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
            pp_enabled_extension_names: extensions.as_ptr(),
            ..Default::default()
        };

        let device = Arc::new(
            unsafe { instance.create_device(physical_device, &device_info, None) }.unwrap(),
        );

        let queues = queue_family_indices
            .iter()
            .map(|&index| unsafe { device.get_device_queue(index, 0) })
            .collect::<Vec<_>>();

        let graphics_queue = queues[0];
        let present_queue = queues[0];

        let swapchain_loader = Arc::new(swapchain::Device::new(&instance, &device));

        self.window = Some(window.clone());
        self.entry = Some(entry);
        self.instance = Some(instance.clone());
        self.device = Some(device.clone());
        self.queues = queues;
        self.graphics_queue = graphics_queue;
        self.present_queue = present_queue;
        self.debug_utils_loader = debug_utils_loader;
        self.debug_utils_messenger = debug_utils_messenger;
        self.surface = surface;
        self.surface_loader = Some(surface_loader);
        self.physical_device = physical_device;
        self.queue_family_indices = queue_family_indices.clone();
        self.graphics_family_index = queue_family_indices[0];
        self.present_family_index = queue_family_indices[0];
        self.swapchain_loader = Some(swapchain_loader);

        self.create_swapchain();
        self.create_render_pass();
        self.create_pipeline();
        self.create_framebuffers();
        self.create_command_pools();
        self.create_command_buffers();
        self.create_sync_objects();

        let gui_info = EguiAshCreateInfoBuilder::default()
            .instance(instance.clone())
            .physical_device(physical_device)
            .device(device.clone())
            .window(window.clone())
            .graphics_queue_index(0)
            .graphics_family_index(queue_family_indices[0])
            .framebuffers(self.framebuffers.clone())
            .format(self.format)
            .build();

        let gui = Arc::new(RwLock::new(EguiAsh::new(event_loop, gui_info)));

        let proxy = self.event_loop_proxy.clone().unwrap();
        {
            gui.write()
                .unwrap()
                .egui_ctx
                .set_request_repaint_callback(move |info| {
                    proxy
                        .lock()
                        .unwrap()
                        .send_event(UserEvent::Redraw(info.delay))
                        .unwrap();
                });
        }

        egui_extras::install_image_loaders(&gui.read().unwrap().egui_ctx);

        self.gui = Some(gui);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Redraw(delay) => self.repaint_delay = delay,
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let device = self.device.clone().unwrap();
        let swapchain_loader = self.swapchain_loader.clone().unwrap();
        let window = self.window.clone().unwrap();

        match event {
            WindowEvent::RedrawRequested => {
                if event_loop.exiting() {
                    return;
                }
                unsafe {
                    device
                        .wait_for_fences(&[self.in_flight_fences[self.frame]], true, u64::MAX)
                        .unwrap();
                }

                let result = unsafe {
                    swapchain_loader.acquire_next_image(
                        self.swapchain,
                        u64::MAX,
                        self.images_available[self.frame],
                        Fence::null(),
                    )
                };

                let image_index = match result {
                    Ok((index, _)) => index as usize,
                    Err(Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.recreate_swapchain();
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

                if !self.images_in_flight[image_index].is_null() {
                    unsafe {
                        device
                            .wait_for_fences(&[self.images_in_flight[image_index]], true, u64::MAX)
                            .unwrap();
                    }
                }

                self.images_in_flight[image_index] = self.in_flight_fences[self.frame];

                let wait_semaphores = [self.images_available[self.frame]];
                let signal_semaphores = [self.render_finished[self.frame]];
                let wait_stages = [PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
                let command_buffers = [self.command_buffers[image_index]];

                let submit_info = SubmitInfo {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: wait_semaphores.as_ptr(),
                    p_wait_dst_stage_mask: wait_stages.as_ptr(),
                    signal_semaphore_count: 1,
                    p_signal_semaphores: signal_semaphores.as_ptr(),
                    command_buffer_count: 1,
                    p_command_buffers: command_buffers.as_ptr(),
                    ..Default::default()
                };

                unsafe {
                    device
                        .reset_fences(&[self.in_flight_fences[self.frame]])
                        .unwrap();

                    device
                        .queue_submit(
                            self.graphics_queue,
                            &[submit_info],
                            self.in_flight_fences[self.frame],
                        )
                        .unwrap()
                }

                let gui_clone = self.gui.clone().unwrap();

                let mut gui = gui_clone.write().unwrap();

                gui.run(|egui_ctx| {
                    containers::Window::new("Hello Ferris").show(egui_ctx, |ui| {
                        ui.image(include_image!("images/ferris.gif"));
                        ui.add(
                            egui::Image::new("https://picsum.photos/seed/1.759706314/1024")
                                .rounding(10.0)
                                .max_size(Vec2::new(200.0, 200.0)),
                        );
                        ui.image(include_image!("images/ferris.png"));
                    });
                });

                let render_finished = gui.paint(signal_semaphores[0], self.extent, image_index);

                event_loop.set_control_flow(if self.repaint_delay.is_zero() {
                    window.request_redraw();
                    ControlFlow::Poll
                } else if let Some(repaint_delayed_instance) =
                    Instant::now().checked_add(self.repaint_delay)
                {
                    ControlFlow::WaitUntil(repaint_delayed_instance)
                } else {
                    ControlFlow::Wait
                });

                let signal_semaphores = [render_finished];

                let present_info = PresentInfoKHR {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: signal_semaphores.as_ptr(),
                    swapchain_count: 1,
                    p_swapchains: [self.swapchain].as_ptr(),
                    p_image_indices: [image_index as u32].as_ptr(),
                    ..Default::default()
                };

                let result =
                    unsafe { swapchain_loader.queue_present(self.present_queue, &present_info) };

                let changed = result == Err(Result::ERROR_OUT_OF_DATE_KHR) || result.unwrap();

                if changed || self.resized {
                    self.recreate_swapchain();
                    self.resized = false;
                }

                self.frame = (self.frame + 1) % self.max_images_in_flight;
            }
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    self.resized = true;
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            _ => {}
        }
        let event_response = self
            .gui
            .as_ref()
            .unwrap()
            .write()
            .unwrap()
            .on_window_event(&window, &event);

        if event_response.repaint {
            window.request_redraw();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        };
    }
    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.destroy();
    }
}

fn main() {
    pretty_env_logger::init();

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    let event_loop_proxy = Arc::new(Mutex::new(event_loop.create_proxy()));
    let mut app = App::default();
    app.event_loop_proxy = Some(event_loop_proxy);

    event_loop.run_app(&mut app).expect("Failed to run app");
}
