//! Video effects from plugins (`[effect]` in plugin.toml): a fragment shader run over every
//! frame of the video, entirely on the GPU. Does nothing unless an effect plugin is on.
//!
//! With an effect on, mpv draws into our own texture instead of the screen; the plugin's
//! shader then draws from it (and, with `feedback`, from its own previous output) into
//! another texture, which is copied to the screen. The shader file is checked for changes
//! twice a second and reloaded, so it can be edited while a video plays.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use eframe::egui_glow;
use eframe::glow::{self, HasContext};
use flex_core::plugins::ActiveEffect;

/// How often the shader file is checked for changes.
const RECHECK: Duration = Duration::from_millis(500);

struct Target {
    texture: glow::Texture,
    fbo: glow::Framebuffer,
}

struct Objects {
    size: (i32, i32),
    /// What mpv draws.
    video: Target,
    /// The effect's output, this frame's and last frame's (swapped each frame).
    out: [Target; 2],
}

/// What's compiled: the file, when it was last changed, and the setting uniforms (which
/// are declared in the shader's header).
#[derive(PartialEq)]
struct Source {
    path: PathBuf,
    modified: Option<SystemTime>,
    uniforms: Vec<String>,
}

/// Lives on the GL thread (inside the player's paint callback); the UI thread only reads
/// the status.
#[derive(Default)]
pub struct Effects {
    objects: Option<Objects>,
    vao: Option<glow::VertexArray>,
    copy: Option<glow::Program>,
    /// The effect's program; `None` if it didn't compile.
    program: Option<glow::Program>,
    source: Option<Source>,
    checked: Option<Instant>,
    started: Option<Instant>,
    frame: i32,
    /// Which of `out` is drawn this frame.
    current: usize,
    /// The shader's state for the plugin's status line, and which file it's about.
    status: Option<(PathBuf, (bool, String))>,
    /// No GL 3 (or GLES 3): effects can't run at all.
    unsupported: bool,
}

/// Uniforms every effect gets.
const HEADER: &str = "\
uniform sampler2D u_video;
uniform sampler2D u_prev;
uniform vec2 u_resolution;
uniform float u_time;
uniform int u_frame;
uniform vec4 u_video_rect;
";

/// The effect's fragment shader: Flex's header, the plugin's file (which defines
/// `vec4 effect(vec2 uv)`) and a `main` calling it. `#line 1` keeps error line numbers
/// matching the plugin's file.
pub fn fragment_source(version: &str, embedded: bool, uniforms: &[String], body: &str) -> String {
    let precision = if embedded { "precision highp float;\nprecision highp int;\n" } else { "" };
    let settings: String = uniforms.iter().map(|u| format!("uniform float {u};\n")).collect();
    format!(
        "{version}\n{precision}{HEADER}{settings}in vec2 v_uv;\nout vec4 flex_color;\n#line 1\n{body}\n\
         void main() {{ flex_color = effect(v_uv); }}\n"
    )
}

fn vertex_source(version: &str) -> String {
    // One triangle covering the screen, with no vertex buffer.
    format!(
        "{version}\nout vec2 v_uv;\nvoid main() {{\n    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n\
         v_uv = p;\n    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);\n}}\n"
    )
}

fn copy_source(version: &str, embedded: bool) -> String {
    let precision = if embedded { "precision highp float;\n" } else { "" };
    format!(
        "{version}\n{precision}uniform sampler2D u_tex;\nuniform float u_flip;\nin vec2 v_uv;\nout vec4 flex_color;\n\
         void main() {{\n    vec2 uv = vec2(v_uv.x, u_flip > 0.5 ? 1.0 - v_uv.y : v_uv.y);\n\
         flex_color = vec4(texture(u_tex, uv).rgb, 1.0);\n}}\n"
    )
}

/// The first line of a compiler log, for the status line (the whole log goes to the log).
fn first_line(log: &str) -> &str {
    log.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("unknown error")
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl Effects {
    /// The status line for this effect's plugin, once its shader has been tried.
    pub fn status(&self, effect: &ActiveEffect) -> Option<&(bool, String)> {
        self.status.as_ref().filter(|(path, _)| *path == effect.shader).map(|(_, s)| s)
    }

    /// Draws the video through `effect` into `target` (`w`×`h` pixels). `render(fbo, w, h)`
    /// draws mpv's frame (bottom-up rows). `rect` is where the picture is, without black
    /// bars, in 0–1 coordinates from the bottom left. Returns false, having drawn nothing,
    /// when the effect can't run: then the caller draws the video as usual.
    ///
    /// # Safety
    /// Must be called on the GL thread with `gl` current. Leaves GL state (program,
    /// textures, viewport, blending) for egui to reset.
    pub unsafe fn paint(
        &mut self,
        gl: &glow::Context,
        effect: &ActiveEffect,
        target: Option<glow::Framebuffer>,
        (w, h): (i32, i32),
        rect: [f32; 4],
        render: impl FnOnce(i32, i32, i32),
    ) -> bool {
        unsafe {
            if self.unsupported || w <= 0 || h <= 0 {
                return false;
            }
            let version = egui_glow::ShaderVersion::get(gl);
            if !version.is_new_shader_interface() {
                log::warn!("video effects need OpenGL 3 or OpenGL ES 3");
                self.unsupported = true;
                self.status = Some((effect.shader.clone(), (false, "Needs OpenGL 3".into())));
                return false;
            }
            if self.vao.is_none() {
                match self.create_shared(gl, version) {
                    Ok(()) => {}
                    Err(e) => {
                        log::warn!("video effects: {e}");
                        self.unsupported = true;
                        return false;
                    }
                }
            }
            self.reload(gl, effect, version);
            let (Some(program), Some(copy), Some(vao)) = (self.program, self.copy, self.vao) else { return false };

            let scale = effect.scale.clamp(0.25, 1.0);
            let size = (((w as f32 * scale).round() as i32).max(1), ((h as f32 * scale).round() as i32).max(1));
            if self.objects.as_ref().is_none_or(|o| o.size != size) {
                self.free_objects(gl);
                match create_objects(gl, size) {
                    Ok(o) => self.objects = Some(o),
                    Err(e) => {
                        log::warn!("video effects: {e}");
                        self.status = Some((effect.shader.clone(), (false, format!("Can't run: {e}"))));
                        return false;
                    }
                }
            }
            let Some(o) = &self.objects else { return false };

            render(o.video.fbo.0.get() as i32, size.0, size.1);

            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(vao));
            let (out, prev) = (&o.out[self.current], &o.out[1 - self.current]);

            // The effect, at the working size.
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(out.fbo));
            gl.viewport(0, 0, size.0, size.1);
            gl.use_program(Some(program));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(o.video.texture));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(prev.texture));
            let at = |name: &str| gl.get_uniform_location(program, name);
            gl.uniform_1_i32(at("u_video").as_ref(), 0);
            gl.uniform_1_i32(at("u_prev").as_ref(), 1);
            gl.uniform_2_f32(at("u_resolution").as_ref(), size.0 as f32, size.1 as f32);
            let started = *self.started.get_or_insert_with(Instant::now);
            gl.uniform_1_f32(at("u_time").as_ref(), started.elapsed().as_secs_f32());
            gl.uniform_1_i32(at("u_frame").as_ref(), self.frame);
            gl.uniform_4_f32(at("u_video_rect").as_ref(), rect[0], rect[1], rect[2], rect[3]);
            for (name, value) in &effect.uniforms {
                gl.uniform_1_f32(at(name).as_ref(), *value);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            // Onto the screen, scaled up if the effect works smaller.
            gl.bind_framebuffer(glow::FRAMEBUFFER, target);
            gl.viewport(0, 0, w, h);
            gl.use_program(Some(copy));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(out.texture));
            gl.uniform_1_i32(gl.get_uniform_location(copy, "u_tex").as_ref(), 0);
            // The window's rows run bottom-up like ours; an offscreen target's top-down.
            let flip = if target.is_some() { 1.0 } else { 0.0 };
            gl.uniform_1_f32(gl.get_uniform_location(copy, "u_flip").as_ref(), flip);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_texture(glow::TEXTURE_2D, None);
            gl.bind_vertex_array(None);
            gl.use_program(None);
            if effect.feedback {
                self.current = 1 - self.current;
            }
            self.frame = self.frame.wrapping_add(1);
            true
        }
    }

    /// Frees everything (no effect is on).
    ///
    /// # Safety
    /// GL thread, `gl` current.
    pub unsafe fn release(&mut self, gl: &glow::Context) {
        unsafe {
            self.free_objects(gl);
            if let Some(p) = self.program.take() {
                gl.delete_program(p);
            }
            if let Some(p) = self.copy.take() {
                gl.delete_program(p);
            }
            if let Some(vao) = self.vao.take() {
                gl.delete_vertex_array(vao);
            }
            self.source = None;
            self.checked = None;
            self.started = None;
        }
    }

    pub fn holds_gl(&self) -> bool {
        self.vao.is_some() || self.objects.is_some()
    }

    unsafe fn create_shared(&mut self, gl: &glow::Context, version: egui_glow::ShaderVersion) -> Result<(), String> {
        unsafe {
            let decl = version.version_declaration();
            let copy = compile(gl, &vertex_source(decl), &copy_source(decl, version.is_embedded()))?;
            match gl.create_vertex_array() {
                Ok(vao) => {
                    self.vao = Some(vao);
                    self.copy = Some(copy);
                    Ok(())
                }
                Err(e) => {
                    gl.delete_program(copy);
                    Err(e)
                }
            }
        }
    }

    /// Compiles the effect when it's new or its file changed.
    unsafe fn reload(&mut self, gl: &glow::Context, effect: &ActiveEffect, version: egui_glow::ShaderVersion) {
        unsafe {
            let same_file = self.source.as_ref().is_some_and(|s| s.path == effect.shader);
            if same_file && self.checked.is_some_and(|t| t.elapsed() < RECHECK) {
                return;
            }
            self.checked = Some(Instant::now());
            let uniforms: Vec<String> = effect.uniforms.iter().map(|(n, _)| n.clone()).collect();
            let source = Source { path: effect.shader.clone(), modified: modified(&effect.shader), uniforms };
            if self.source.as_ref() == Some(&source) {
                return;
            }
            if !same_file {
                // A different effect: start afresh.
                self.started = None;
                self.frame = 0;
                self.free_objects(gl);
            }
            let status = match std::fs::read_to_string(&source.path) {
                Err(e) => {
                    self.delete_program(gl);
                    let name = source.path.file_name().unwrap_or_default().to_string_lossy();
                    (false, format!("Can't read {name}: {e}"))
                }
                Ok(body) => {
                    let decl = version.version_declaration();
                    let fragment = fragment_source(decl, version.is_embedded(), &source.uniforms, &body);
                    match compile(gl, &vertex_source(decl), &fragment) {
                        Ok(program) => {
                            self.delete_program(gl);
                            self.program = Some(program);
                            log::info!("video effect {}: loaded {}", effect.name, source.path.display());
                            (true, "Running".to_owned())
                        }
                        Err(e) => {
                            // Keep showing the plain video until it's fixed.
                            self.delete_program(gl);
                            log::warn!("video effect {}: {}:\n{e}", effect.name, source.path.display());
                            (false, format!("Shader error: {}", first_line(&e)))
                        }
                    }
                }
            };
            self.status = Some((source.path.clone(), status));
            self.source = Some(source);
        }
    }

    unsafe fn delete_program(&mut self, gl: &glow::Context) {
        if let Some(p) = self.program.take() {
            unsafe { gl.delete_program(p) };
        }
    }

    unsafe fn free_objects(&mut self, gl: &glow::Context) {
        if let Some(o) = self.objects.take() {
            for t in [o.video].into_iter().chain(o.out) {
                unsafe {
                    gl.delete_framebuffer(t.fbo);
                    gl.delete_texture(t.texture);
                }
            }
        }
    }
}

unsafe fn compile(gl: &glow::Context, vertex: &str, fragment: &str) -> Result<glow::Program, String> {
    unsafe {
        let program = gl.create_program()?;
        let mut shaders = Vec::new();
        let mut error = None;
        for (kind, source) in [(glow::VERTEX_SHADER, vertex), (glow::FRAGMENT_SHADER, fragment)] {
            let shader = gl.create_shader(kind)?;
            gl.shader_source(shader, source);
            gl.compile_shader(shader);
            if !gl.get_shader_compile_status(shader) {
                error.get_or_insert_with(|| gl.get_shader_info_log(shader));
            }
            gl.attach_shader(program, shader);
            shaders.push(shader);
        }
        if error.is_none() {
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                error = Some(gl.get_program_info_log(program));
            }
        }
        for shader in shaders {
            gl.detach_shader(program, shader);
            gl.delete_shader(shader);
        }
        match error {
            None => Ok(program),
            Some(e) => {
                gl.delete_program(program);
                Err(e)
            }
        }
    }
}

unsafe fn create_target(gl: &glow::Context, (w, h): (i32, i32), formats: &[(u32, u32)]) -> Result<Target, String> {
    unsafe {
        for &(internal, kind) in formats {
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            let none = glow::PixelUnpackData::Slice(None);
            gl.tex_image_2d(glow::TEXTURE_2D, 0, internal as i32, w, h, 0, glow::RGBA, kind, none);
            for (param, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, param, value as i32);
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
            let fbo = gl.create_framebuffer()?;
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(fbo));
            gl.framebuffer_texture_2d(glow::FRAMEBUFFER, glow::COLOR_ATTACHMENT0, glow::TEXTURE_2D, Some(texture), 0);
            if gl.check_framebuffer_status(glow::FRAMEBUFFER) == glow::FRAMEBUFFER_COMPLETE {
                gl.disable(glow::SCISSOR_TEST);
                gl.clear_color(0.0, 0.0, 0.0, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
                return Ok(Target { texture, fbo });
            }
            gl.delete_framebuffer(fbo);
            gl.delete_texture(texture);
        }
        Err("framebuffers incomplete".into())
    }
}

unsafe fn create_objects(gl: &glow::Context, size: (i32, i32)) -> Result<Objects, String> {
    unsafe {
        let rgba8 = (glow::RGBA8, glow::UNSIGNED_BYTE);
        // Feedback fades over many frames: 8 bits a channel would leave trails stuck at
        // faint colours, so half floats where the GPU can draw into them.
        let float = [(glow::RGBA16F, glow::HALF_FLOAT), rgba8];
        let video = create_target(gl, size, &[rgba8])?;
        let a = create_target(gl, size, &float);
        let b = create_target(gl, size, &float);
        match (a, b) {
            (Ok(a), Ok(b)) => Ok(Objects { size, video, out: [a, b] }),
            (a, b) => {
                for t in [Ok(video), a, b].into_iter().flatten() {
                    gl.delete_framebuffer(t.fbo);
                    gl.delete_texture(t.texture);
                }
                Err("framebuffers incomplete".into())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn effect_source_declares_settings_and_keeps_line_numbers() {
        let body = "vec4 effect(vec2 uv) {\n    return texture(u_video, uv) * s_gain;\n}";
        let src = fragment_source("#version 330", false, &["s_gain".into()], body);
        assert!(src.starts_with("#version 330\nuniform sampler2D u_video;"));
        assert!(src.contains("uniform float s_gain;\n"));
        let after = src.split("#line 1\n").nth(1).unwrap();
        assert!(after.starts_with("vec4 effect"), "the plugin's file starts at line 1");
        assert!(src.trim_end().ends_with("void main() { flex_color = effect(v_uv); }"));
        let es = fragment_source("#version 300 es", true, &[], body);
        assert!(es.starts_with("#version 300 es\nprecision highp float;"));
    }

    #[test]
    fn status_takes_the_first_line_of_the_log() {
        assert_eq!(first_line("\n  0:3(12): error: syntax error\n0:4: more\n"), "0:3(12): error: syntax error");
        assert_eq!(first_line(""), "unknown error");
    }
}
