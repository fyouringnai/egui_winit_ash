use ash::vk::{Format, Framebuffer, PhysicalDevice};
use ash::{Device, Instance};
use std::sync::{Arc, RwLock};
use winit::window::Window;

#[derive(Default)]
pub struct EguiAshCreateInfoBuilder {
    pub instance: Option<Arc<Instance>>,
    pub physical_device: PhysicalDevice,
    pub graphics_family_index: u32,
    pub graphics_queue_index: u32,
    pub device: Option<Arc<Device>>,
    pub window: Option<Arc<Window>>,
    pub framebuffers: Option<Arc<RwLock<Vec<Framebuffer>>>>,
    pub format: Format,
    // the unit of the capacity is 20 kilobytes
    // default vertex is 1_024 * 20 = 20_480 bytes
    pub vertex_capacity: Option<usize>,
    // the unit of the capacity is 4 kilobyte
    // default index is 1_024 * 4 = 4_096 bytes
    pub index_capacity: Option<usize>,
    // default texture capacity is 1_024, and it cannot lower than 1
    pub texture_capacity: Option<usize>,
    pub pixels_per_point: Option<f32>,
}

impl EguiAshCreateInfoBuilder {
    pub fn instance(mut self, instance: Arc<Instance>) -> Self {
        self.instance = Some(instance);
        self
    }

    pub fn physical_device(mut self, physical_device: PhysicalDevice) -> Self {
        self.physical_device = physical_device;
        self
    }

    pub fn device(mut self, device: Arc<Device>) -> Self {
        self.device = Some(device);
        self
    }

    pub fn window(mut self, window: Arc<Window>) -> Self {
        self.window = Some(window);
        self
    }

    pub fn graphics_family_index(mut self, graphics_family_index: u32) -> Self {
        self.graphics_family_index = graphics_family_index;
        self
    }

    pub fn graphics_queue_index(mut self, graphics_queue_index: u32) -> Self {
        self.graphics_queue_index = graphics_queue_index;
        self
    }

    pub fn format(mut self, format: Format) -> Self {
        self.format = format;
        self
    }

    pub fn framebuffers(mut self, framebuffers: Arc<RwLock<Vec<Framebuffer>>>) -> Self {
        self.framebuffers = Some(framebuffers);
        self
    }

    pub fn vertex_capacity(mut self, vertex_capacity: usize) -> Self {
        self.vertex_capacity = Some(vertex_capacity);
        self
    }

    pub fn index_capacity(mut self, index_capacity: usize) -> Self {
        self.index_capacity = Some(index_capacity);
        self
    }

    pub fn texture_capacity(mut self, texture_capacity: usize) -> Self {
        self.texture_capacity = Some(texture_capacity);
        self
    }

    pub fn pixels_per_point(mut self, pixels_per_point: Option<f32>) -> Self {
        self.pixels_per_point = pixels_per_point;
        self
    }

    pub fn build(self) -> EguiAshCreateInfo {
        EguiAshCreateInfo {
            instance: self.instance.expect("Instance not provided"),
            physical_device: self.physical_device,
            graphics_family_index: self.graphics_family_index,
            graphics_queue_index: self.graphics_queue_index,
            device: self.device.expect("Device not provided"),
            window: self.window.expect("Window not provided"),
            framebuffers: self.framebuffers.expect("Framebuffers not provided"),
            format: self.format,
            vertex_capacity: self.vertex_capacity.unwrap_or(1_024),
            index_capacity: self.index_capacity.unwrap_or(1_024),
            texture_capacity: self.texture_capacity.unwrap_or(1_024).max(1),
            pixels_per_point: self.pixels_per_point,
            the_pixels_per_point: Arc::new(RwLock::new(self.pixels_per_point.unwrap_or(1.0))),
        }
    }
}

pub struct EguiAshCreateInfo {
    pub(crate) instance: Arc<Instance>,
    pub(crate) physical_device: PhysicalDevice,
    pub(crate) graphics_family_index: u32,
    pub(crate) graphics_queue_index: u32,
    pub(crate) device: Arc<Device>,
    pub(crate) window: Arc<Window>,
    pub(crate) framebuffers: Arc<RwLock<Vec<Framebuffer>>>,
    pub(crate) format: Format,
    pub(crate) vertex_capacity: usize,
    pub(crate) index_capacity: usize,
    pub(crate) texture_capacity: usize,
    pub(crate) pixels_per_point: Option<f32>,
    pub(crate) the_pixels_per_point: Arc<RwLock<f32>>,
}
