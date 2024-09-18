use crate::renderer::Renderer;
use crate::utils::EguiAshCreateInfo;
use ash::vk::{Extent2D, Semaphore};
use egui::ahash::HashSet;
use egui::epaint::ClippedShape;
use egui::{Context, FullOutput, TexturesDelta, ViewportId, ViewportInfo, ViewportOutput};
use egui_winit::{EventResponse, State};
use log::trace;
use std::mem::take;
use std::sync::{Arc, RwLock};
use winit::window::Window;

pub struct EguiAsh {
    pub egui_ctx: Context,
    pub egui_winit: State,
    renderer: Renderer,

    shapes: Vec<ClippedShape>,
    pixels_per_point: Arc<RwLock<f32>>,
    textures_delta: TexturesDelta,
    viewport_info: ViewportInfo,
}

impl EguiAsh {
    pub fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        create_info: EguiAshCreateInfo,
    ) -> Self {
        let pixels_per_point = create_info.pixels_per_point;
        let the_pixels_per_point = create_info.the_pixels_per_point.clone();
        let renderer = Renderer::new(create_info);
        let egui_ctx = Context::default();

        let egui_winit = State::new(
            egui_ctx.clone(),
            ViewportId::ROOT,
            event_loop,
            pixels_per_point,
            event_loop.system_theme(),
            Some(renderer.max_texture_side),
        );

        Self {
            egui_ctx,
            egui_winit,
            renderer,
            shapes: Default::default(),
            pixels_per_point: the_pixels_per_point,
            textures_delta: Default::default(),
            viewport_info: Default::default(),
        }
    }

    pub fn on_window_event(
        &mut self,
        window: &Window,
        event: &winit::event::WindowEvent,
    ) -> EventResponse {
        self.egui_winit.on_window_event(window, event)
    }

    pub fn run(&mut self, run_ui: impl FnMut(&Context)) {
        let raw_input = self.egui_winit.take_egui_input(&self.renderer.window);

        let FullOutput {
            platform_output,
            textures_delta,
            shapes,
            pixels_per_point,
            viewport_output,
        } = self.egui_ctx.run(raw_input, run_ui);

        if viewport_output.len() > 1 {
            log::warn!("Multiple viewports not yet supported by EguiAsh");
        }
        for (_, ViewportOutput { commands, .. }) in viewport_output {
            let mut actions_requested: HashSet<egui_winit::ActionRequested> = Default::default();
            egui_winit::process_viewport_commands(
                &self.egui_ctx,
                &mut self.viewport_info,
                commands,
                &self.renderer.window,
                &mut actions_requested,
            );
            for action in actions_requested {
                log::warn!("{:?} not yet supported by EguiAsh", action);
            }
        }

        self.egui_winit
            .handle_platform_output(&self.renderer.window, platform_output);

        self.shapes = shapes;
        *self.pixels_per_point.write().unwrap() = pixels_per_point;
        self.textures_delta.append(textures_delta);
    }

    pub fn paint(
        &mut self,
        images_available: Semaphore,
        extent: Extent2D,
        image_index: usize,
    ) -> Semaphore {
        let shapes = take(&mut self.shapes);
        let mut textures_delta = take(&mut self.textures_delta);

        textures_delta.set.iter().for_each(|(id, texture)| {
            self.renderer.register_texture(*id, texture);
            trace!("Registered texture {:?}", id);
        });

        let pixels_per_point = *self.pixels_per_point.read().unwrap();

        let clipped_primitives = self.egui_ctx.tessellate(shapes, pixels_per_point);

        let render_finished = self.renderer.draw_primitive(
            &clipped_primitives,
            images_available,
            extent,
            image_index,
        );

        for id in textures_delta.free.drain(..) {
            self.renderer.unregister_texture(id);
            trace!("Unregistered texture {:?}", id);
        }

        render_finished
    }

    pub fn destroy(&mut self) {
        self.renderer.destroy();
    }
}
