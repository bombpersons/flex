//! Video effects (`flex_core::effects`): a fragment shader run over every frame of the
//! video, entirely on the GPU. Does nothing unless an effect is chosen.
//!
//! With an effect on, mpv draws into our own texture instead of the screen; the plugin's
//! shader then draws from it (and, with `feedback`, from its own previous output) into
//! another texture, which is copied to the screen. The shader file is checked for changes
//! twice a second and reloaded, so it can be edited while a video plays. During a shuffle's
//! fade two effects run at once from the same video, and their outputs are mixed.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use eframe::egui;
use eframe::egui_glow;
use eframe::glow::{self, HasContext};
use flex_core::audio::{self, Analyser, Features, Levels};
use flex_core::effects::ActiveEffect;
use flex_core::plugins::{SettingKind, SettingSpec};
use flex_plugin::serde_json;

use crate::mpv::Mpv;
use crate::picture::{Picture, Upscale};
use crate::shared::Shared;

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
}

/// What's compiled: the file, when it was last changed, and the setting uniforms (which
/// are declared in the shader's header).
#[derive(PartialEq)]
struct Source {
    path: PathBuf,
    modified: Option<SystemTime>,
    uniforms: Vec<String>,
}

/// One effect being drawn.
#[derive(Default)]
struct Slot {
    /// The effect's program; `None` if it didn't compile.
    program: Option<glow::Program>,
    source: Option<Source>,
    checked: Option<Instant>,
    started: Option<Instant>,
    frame: i32,
    size: (i32, i32),
    /// The effect's output, this frame's and last frame's (swapped each frame).
    out: Option<[Target; 2]>,
    /// Which of `out` is drawn this frame.
    current: usize,
    /// Which of `out` was drawn last.
    drawn: usize,
    /// The shader's state for the plugin's status line, and which file it's about.
    status: Option<(PathBuf, (bool, String))>,
}

/// Lives on the GL thread (inside the player's paint callback); the UI thread only reads
/// the status.
#[derive(Default)]
pub struct Renderer {
    objects: Option<Objects>,
    vao: Option<glow::VertexArray>,
    /// `u_spectrum`: BANDS × HISTORY.
    spectrum: Option<glow::Texture>,
    depth: DepthGl,
    held: HeldFrames,
    faces: FaceGl,
    cuts: CutGl,
    copy: Option<glow::Program>,
    slots: [Slot; 2],
    /// Which of `slots` has the chosen effect; the other has the one fading out, if any.
    front: usize,
    /// No GL 3 (or GLES 3): effects can't run at all.
    unsupported: bool,
}

/// What an effect is drawn with this frame, worked out on the UI thread.
#[derive(Clone)]
pub struct Frame {
    pub effect: ActiveEffect,
    /// The effect fading out, and how far `effect` has faded in over it (0–1).
    pub fading: Option<(ActiveEffect, f32)>,
    /// The size mpv draws at, relative to the screen.
    pub video_scale: f32,
    /// Shuffle wants to know where the shot changes.
    pub find_cuts: bool,
    /// Where the picture is, without black bars (0–1 from the bottom left).
    pub rect: [f32; 4],
    pub video_time: f32,
    /// mpv's position: unlike `video_time`, the same for all of one video frame.
    pub pts: f32,
    /// The sound, when the effect asked for it and there is some.
    pub audio: Option<Arc<Features>>,
    /// The depth worker, when the effect asked for depth.
    pub depth: Option<crate::depth::Link>,
    /// The face worker, when the effect asked for faces.
    pub faces: Option<crate::faces::Link>,
}

/// The player's side: a smooth playback clock, and the sound for effects that want it.
#[derive(Default)]
pub struct Inputs {
    /// Video time, moving smoothly between mpv's position updates (which only change
    /// once per video frame).
    clock: f64,
    last: Option<Instant>,
    /// The channels and layout mpv's audio filter was added for.
    filter: Option<(usize, String)>,
    /// The filter's last report as mpv gave it, to spot new ones.
    metadata: Option<String>,
    /// Reports waiting for their sound to be heard, and the latest one that has been.
    waiting: VecDeque<(Instant, Levels)>,
    levels: Option<Levels>,
    analyser: Option<Analyser>,
}

impl Inputs {
    /// Adds mpv's audio filter, or replaces it if the sound's channels changed.
    fn keep_filter(&mut self, mpv: &Mpv) {
        let (Some(channels), Some(layout)) = (mpv.get_i64("audio-params/channel-count"), mpv.get_string("audio-params/channels")) else { return };
        let wanted = (channels.max(1) as usize, layout);
        if self.filter.as_ref() == Some(&wanted) {
            return;
        }
        self.stop(mpv);
        let filter = format!("@{}:lavfi=[{}]", audio::FILTER_LABEL, audio::filter_graph(wanted.0, &wanted.1));
        if let Err(e) = mpv.command(&["af", "add", &filter]) {
            log::warn!("video effects: no sound for effects: {e:#}");
        }
        self.filter = Some(wanted);
    }

    /// Takes mpv's audio filter out, when nothing wants the sound any more.
    pub fn stop(&mut self, mpv: &Mpv) {
        if self.filter.take().is_some() {
            let _ = mpv.command(&["af", "remove", &format!("@{}", audio::FILTER_LABEL)]);
        }
        self.metadata = None;
        self.waiting.clear();
        self.levels = None;
    }

    /// Call once per frame while an effect is on.
    pub fn tick(&mut self, mpv: &Mpv, wants_audio: bool) -> (f32, Option<Arc<Features>>) {
        let now = Instant::now();
        let dt = self.last.map_or(0.0, |t| now.duration_since(t).as_secs_f64());
        self.last = Some(now);
        let paused = mpv.get_flag("pause").unwrap_or(false) || mpv.get_flag("paused-for-cache").unwrap_or(false);
        match mpv.get_f64("time-pos") {
            Some(pos) if paused || (pos - self.clock).abs() > 0.25 => self.clock = pos,
            Some(pos) => {
                // Run at playback speed, gently pulled towards mpv's position.
                self.clock += dt * mpv.get_f64("speed").unwrap_or(1.0);
                self.clock += (pos - self.clock) * (dt * 4.0).min(1.0);
            }
            None => {}
        }

        if !wants_audio {
            self.stop(mpv);
            self.analyser = None;
            return (self.clock as f32, None);
        }
        self.keep_filter(mpv);
        if let Some((channels, _)) = self.filter {
            let metadata = mpv.get_string(&format!("af-metadata/{}", audio::FILTER_LABEL));
            if metadata.is_some() && metadata != self.metadata {
                if let Some(levels) = metadata.as_deref().and_then(|m| audio::read_levels(m, channels)) {
                    self.waiting.push_back((now, levels));
                }
                self.metadata = metadata;
            }
        }
        while self.waiting.front().is_some_and(|(at, _)| now.duration_since(*at) >= HEARD_AFTER) {
            self.levels = self.waiting.pop_front().map(|(_, levels)| levels);
        }
        let analyser = self.analyser.get_or_insert_with(Analyser::new);
        let features = analyser.update(self.levels.as_ref(), self.clock, dt as f32);
        (self.clock as f32, Some(Arc::new(features.clone())))
    }
}

/// How long after mpv's audio filter sees some sound it comes out of the speakers.
const HEARD_AFTER: Duration = Duration::from_millis(400);

/// F3 in the player: a floating window for choosing the effect and changing its
/// settings (and Anime4K and SVP) while the video plays. Changes are kept, as in
/// Settings. Returns this video's new picture options when Anime4K or SVP changed.
pub fn window(ctx: &egui::Context, open: &mut bool, sh: &mut Shared, frame: Option<&Frame>, picture: Picture) -> Option<Picture> {
    let mut new_picture = None;
    let right = ctx.content_rect().right();
    egui::Window::new("Video effects")
        .open(open)
        .default_pos(egui::pos2(right - 560.0, 60.0))
        .default_width(520.0)
        .resizable(false)
        .show(ctx, |ui| {
            ui.spacing_mut().slider_width = 260.0;
            ui.weak("F3 to hide · changes are kept");

            // Which effect.
            let names: Vec<(String, String)> = sh.effects.list().iter().map(|e| (e.id.clone(), e.manifest.name.clone())).collect();
            let current = sh.effects.chosen().map(|e| e.id.clone());
            let current_name = sh.effects.chosen().map_or("Off".to_owned(), |e| e.manifest.name.clone());
            let mut pick: Option<Option<String>> = None;
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("effect").selected_text(current_name).show_ui(ui, |ui| {
                    if ui.selectable_label(current.is_none(), "Off").clicked() {
                        pick = Some(None);
                    }
                    for (id, name) in &names {
                        if ui.selectable_label(current.as_deref() == Some(id), name).clicked() {
                            pick = Some(Some(id.clone()));
                        }
                    }
                });
                if ui.button("Look for new").on_hover_text("New or removed files in the effects folders").clicked() {
                    sh.refresh_effects();
                }
            });
            if let Some(pick) = pick {
                let (effects, db) = sh.effects_mut();
                effects.choose(pick.as_deref(), db);
            }

            if let Some(effect) = sh.effects.chosen().cloned() {
                let warn = egui::Color32::from_rgb(255, 140, 110);
                if let Some((false, error)) = &sh.effects.status {
                    ui.colored_label(warn, error);
                }
                if let Some(problem) = &effect.problem {
                    ui.colored_label(warn, problem);
                }
                if !effect.manifest.description.is_empty() {
                    ui.weak(&effect.manifest.description);
                }
                let specs: Vec<SettingSpec> = effect.settings().cloned().collect();
                egui::Grid::new("effect-settings").num_columns(2).spacing([16.0, 8.0]).show(ui, |ui| {
                    for spec in &specs {
                        setting(ui, sh, &effect.id, spec);
                        ui.end_row();
                    }
                });
                if !specs.is_empty() && ui.button("Back to the defaults").clicked() {
                    let (effects, db) = sh.effects_mut();
                    for spec in specs.iter().filter(|s| s.default.is_some()) {
                        effects.set_value(&effect.id, &spec.key, spec.default.clone().unwrap_or_default());
                    }
                    effects.save(&effect.id, db);
                }
            }

            if let Some(depth) = frame.and_then(|f| f.depth.as_ref()) {
                ui.separator();
                ui.label(format!("Depth: {}", depth.status().describe()));
            }
            if let Some(faces) = frame.and_then(|f| f.faces.as_ref()) {
                ui.label(format!("Faces: {}", faces.status().describe()));
            }
            if let Some(a) = frame.and_then(|f| f.audio.as_deref()) {
                ui.separator();
                ui.label("Sound");
                for (name, v) in [("level", a.level), ("bass", a.bass), ("mid", a.mid), ("treble", a.treble), ("beat", a.beat)] {
                    let height = ui.text_style_height(&egui::TextStyle::Body) + 6.0;
                    ui.add(egui::ProgressBar::new(v.clamp(0.0, 1.0)).desired_height(height).text(format!("{name} {v:.2}")));
                }
            }

            ui.separator();
            ui.label("Picture (as in Settings → Picture)");
            let upscale = picture.upscale.unwrap_or(Upscale::Off);
            let mut chosen = upscale;
            ui.horizontal(|ui| {
                ui.label("Anime4K");
                egui::ComboBox::from_id_salt("anime4k").selected_text(upscale.label()).show_ui(ui, |ui| {
                    for u in Upscale::ALL {
                        ui.selectable_value(&mut chosen, u, u.label());
                    }
                });
            });
            if chosen != upscale {
                sh.set_picture_upscale(chosen);
                new_picture = Some(Picture { upscale: (chosen != Upscale::Off).then_some(chosen), ..picture });
            }
            let mut smooth = picture.smooth;
            if ui.checkbox(&mut smooth, "60fps smoothing (SVP)").changed() {
                sh.set_picture_smooth(smooth);
                let base = new_picture.unwrap_or(picture);
                new_picture = Some(Picture { smooth: smooth && sh.picture_smooth(), ..base });
            }
        });
    new_picture
}

/// One setting's label and control. Sliders save when let go.
fn setting(ui: &mut egui::Ui, sh: &mut Shared, id: &str, spec: &SettingSpec) {
    let label = ui.label(&spec.label);
    if !spec.detail.is_empty() {
        label.on_hover_text(&spec.detail);
    }
    let value = sh.effects.value(id, &spec.key).cloned();
    let (effects, db) = sh.effects_mut();
    match spec.kind {
        SettingKind::Number => {
            let (min, max) = (spec.min.unwrap_or(0.0), spec.max.unwrap_or(100.0));
            let mut v = value.as_ref().and_then(serde_json::Value::as_f64).unwrap_or(min);
            let mut slider = egui::Slider::new(&mut v, min..=max).suffix(&spec.unit);
            if let Some(step) = spec.step.filter(|s| *s > 0.0) {
                slider = slider.step_by(step);
            }
            let r = ui.add(slider);
            if r.changed() {
                let v = if v.fract() == 0.0 { serde_json::json!(v as i64) } else { serde_json::json!(v) };
                effects.set_value(id, &spec.key, v);
            }
            if r.drag_stopped() || (r.changed() && !r.dragged()) {
                effects.save(id, db);
            }
        }
        SettingKind::Bool => {
            let mut on = value.as_ref().and_then(serde_json::Value::as_bool).unwrap_or(false);
            if ui.checkbox(&mut on, "").changed() {
                effects.set_setting(id, &spec.key, serde_json::json!(on), db);
            }
        }
        SettingKind::Choice => {
            let current = value.as_ref().and_then(serde_json::Value::as_str).unwrap_or("").to_owned();
            let shown = spec.options.iter().find(|o| o.id == current).map_or(current.clone(), |o| o.label.clone());
            let mut pick = None;
            egui::ComboBox::from_id_salt(("effect-choice", &spec.key)).selected_text(shown).show_ui(ui, |ui| {
                for o in &spec.options {
                    if ui.selectable_label(o.id == current, &o.label).clicked() {
                        pick = Some(o.id.clone());
                    }
                }
            });
            if let Some(pick) = pick {
                effects.set_setting(id, &spec.key, serde_json::json!(pick), db);
            }
        }
        SettingKind::Text | SettingKind::Button => {
            ui.label("");
        }
    }
}

/// Uniforms every effect gets.
const HEADER: &str = "\
uniform sampler2D u_video;
uniform sampler2D u_prev;
uniform vec2 u_resolution;
uniform float u_time;
uniform int u_frame;
uniform vec4 u_video_rect;
uniform float u_video_time;
uniform float u_level;
uniform float u_bass;
uniform float u_mid;
uniform float u_treble;
uniform float u_beat;
uniform sampler2D u_spectrum;
uniform sampler2D u_depth;
uniform float u_has_depth;
// Depth at a point on screen, from 0 (far) to 1 (near); 0 until there is some.
float depth_at(vec2 uv) {
    vec2 p = (uv - u_video_rect.xy) / max(u_video_rect.zw - u_video_rect.xy, vec2(1e-6));
    return texture(u_depth, p).r * u_has_depth;
}
uniform sampler2D u_masks;
uniform float u_has_masks;
// How much of a point on screen is a person, a wall or sky, each from 0 to 1.
vec4 masks_at(vec2 uv) {
    vec2 p = (uv - u_video_rect.xy) / max(u_video_rect.zw - u_video_rect.xy, vec2(1e-6));
    return texture(u_masks, p) * u_has_masks;
}
float person_at(vec2 uv) {
    return masks_at(uv).r;
}
float wall_at(vec2 uv) {
    return masks_at(uv).g;
}
float sky_at(vec2 uv) {
    return masks_at(uv).b;
}
const int MAX_FACES = 8;
uniform int u_face_count;
// Each face: x0, y0, x1, y1 on screen. Its five points: right eye, left eye, nose, right and
// left corners of the mouth. u_face_seen fades from 0 to 1 and back as faces come and go.
uniform vec4 u_faces[MAX_FACES];
uniform vec2 u_face_points[MAX_FACES * 5];
uniform float u_face_seen[MAX_FACES];
vec2 face_point(int face, int point) {
    return u_face_points[face * 5 + point];
}
// How much of a face is at a point on screen, from 0 to 1, soft at the edges.
float face_at(vec2 uv) {
    float amount = 0.0;
    for (int i = 0; i < u_face_count; i++) {
        vec2 centre = 0.5 * (u_faces[i].xy + u_faces[i].zw);
        vec2 radius = max(0.5 * (u_faces[i].zw - u_faces[i].xy), vec2(1e-5));
        amount = max(amount, u_face_seen[i] * (1.0 - smoothstep(0.8, 1.15, length((uv - centre) / radius))));
    }
    return amount;
}
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
        "{version}\n{precision}uniform sampler2D u_tex;\nuniform sampler2D u_from;\nuniform float u_faded_in;\nuniform float u_flip;\n\
         in vec2 v_uv;\nout vec4 flex_color;\n\
         void main() {{\n    vec2 uv = vec2(v_uv.x, u_flip > 0.5 ? 1.0 - v_uv.y : v_uv.y);\n\
         flex_color = vec4(mix(texture(u_from, uv).rgb, texture(u_tex, uv).rgb, u_faded_in), 1.0);\n}}\n"
    )
}

/// The first line of a compiler log, for the status line (the whole log goes to the log).
fn first_line(log: &str) -> &str {
    log.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("unknown error")
}

fn modified(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl Renderer {
    /// The status line for this effect, once its shader has been tried.
    pub fn status(&self, effect: &ActiveEffect) -> Option<&(bool, String)> {
        self.slots.iter().filter_map(|s| s.status.as_ref()).find(|(path, _)| *path == effect.shader).map(|(_, s)| s)
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
        frame: &Frame,
        target: Option<glow::Framebuffer>,
        (w, h): (i32, i32),
        render: impl FnOnce(i32, i32, i32),
    ) -> bool {
        let (effect, rect) = (&frame.effect, frame.rect);
        unsafe {
            if self.unsupported || w <= 0 || h <= 0 {
                return false;
            }
            let version = egui_glow::ShaderVersion::get(gl);
            if !version.is_new_shader_interface() {
                log::warn!("video effects need OpenGL 3 or OpenGL ES 3");
                self.unsupported = true;
                self.slots[self.front].status = Some((effect.shader.clone(), (false, "Needs OpenGL 3".into())));
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
            // As a fade starts, the effect leaving keeps its slot and the new one takes the other.
            let holds = |slot: &Slot, shader: &Path| slot.source.as_ref().is_some_and(|s| s.path == shader);
            if !holds(&self.slots[self.front], &effect.shader)
                && (holds(&self.slots[1 - self.front], &effect.shader) || frame.fading.as_ref().is_some_and(|(from, _)| holds(&self.slots[self.front], &from.shader)))
            {
                self.front = 1 - self.front;
            }
            let (front, back) = (self.front, 1 - self.front);
            self.slots[front].reload(gl, effect, version);
            match &frame.fading {
                Some((from, _)) => self.slots[back].reload(gl, from, version),
                None => self.slots[back].free(gl),
            }
            if self.slots[front].program.is_none() && self.slots[back].program.is_none() {
                return false;
            }
            let (Some(copy), Some(vao)) = (self.copy, self.vao) else { return false };

            let at_scale = |scale: f32| (((w as f32 * scale).round() as i32).max(1), ((h as f32 * scale).round() as i32).max(1));
            let size = at_scale(frame.video_scale.max(effect.scale).clamp(0.25, 1.0));
            if self.objects.as_ref().is_none_or(|o| o.size != size) {
                self.free_objects(gl);
                match create_target(gl, size, &[(glow::RGBA8, glow::UNSIGNED_BYTE)]) {
                    Ok(video) => self.objects = Some(Objects { size, video }),
                    Err(e) => {
                        log::warn!("video effects: {e}");
                        self.slots[front].status = Some((effect.shader.clone(), (false, format!("Can't run: {e}"))));
                        return false;
                    }
                }
            }
            let Some(o) = &self.objects else { return false };
            let scales = [Some(effect), frame.fading.as_ref().map(|(from, _)| from)];
            for (slot, drawn) in [front, back].into_iter().zip(scales) {
                let Some(drawn) = drawn else { continue };
                if let Err(e) = self.slots[slot].prepare(gl, at_scale(drawn.scale.clamp(0.25, 1.0))) {
                    log::warn!("video effects: {e}");
                    self.slots[slot].status = Some((drawn.shader.clone(), (false, format!("Can't run: {e}"))));
                    return false;
                }
            }
            // A feedback effect fading in starts from the one fading out, instead of from black.
            if frame.fading.is_some() && effect.feedback && self.slots[front].frame == 0 {
                let [a, b] = &self.slots;
                let (new, old) = if front == 0 { (a, b) } else { (b, a) };
                if let (Some(new_out), Some(old_out)) = (&new.out, &old.out) {
                    gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(old_out[old.drawn].fbo));
                    gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(new_out[1 - new.current].fbo));
                    gl.blit_framebuffer(0, 0, old.size.0, old.size.1, 0, 0, new.size.0, new.size.1, glow::COLOR_BUFFER_BIT, glow::LINEAR);
                    gl.bind_framebuffer(glow::FRAMEBUFFER, None);
                }
            }

            render(o.video.fbo.0.get() as i32, size.0, size.1);
            if frame.depth.is_some() || frame.faces.is_some() {
                self.held.push(gl, &o.video, size, frame.pts);
            } else {
                self.held.free(gl);
            }
            // The networks look at the kept copy of a frame, however many paints later they get
            // to it, so what they find is about exactly the picture that will be drawn.
            let (source, source_time) = self.held.newest().unwrap_or((o.video.fbo, frame.pts));
            match &frame.depth {
                Some(link) => self.depth.update(gl, link, source, size, rect, source_time, frame.video_time),
                None => self.depth.delay = 0.0,
            }
            match &frame.faces {
                Some(link) => self.faces.update(gl, link, source, size, rect, source_time, frame.video_time),
                None => self.faces.delay = 0.0,
            }
            if frame.find_cuts {
                self.cuts.update(gl, source, size, rect, source_time);
            }
            // The picture is held back to the frame depth and faces have caught up with.
            let (video_texture, pts) = if frame.depth.is_some() || frame.faces.is_some() {
                self.held.pick(frame.video_time - self.depth.delay.max(self.faces.delay)).unwrap_or((o.video.texture, frame.pts))
            } else {
                (o.video.texture, frame.pts)
            };
            let (depth_texture, mask_texture) = if frame.depth.is_some() { self.depth.map_for(pts) } else { (None, None) };
            let faces = if frame.faces.is_some() { self.faces.uniforms(pts, rect) } else { FaceUniforms::default() };

            gl.disable(glow::SCISSOR_TEST);
            gl.disable(glow::BLEND);
            gl.bind_vertex_array(Some(vao));

            // What every effect is given.
            let audio = frame.audio.as_deref();
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(video_texture));
            if let Some(spectrum) = self.spectrum {
                gl.active_texture(glow::TEXTURE2);
                gl.bind_texture(glow::TEXTURE_2D, Some(spectrum));
                if let Some(a) = audio {
                    let bytes: &[u8] = std::slice::from_raw_parts(a.spectrum.as_ptr().cast(), a.spectrum.len() * 4);
                    gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
                    let (w, h) = (audio::BANDS as i32, audio::HISTORY as i32);
                    gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, w, h, glow::RED, glow::FLOAT, glow::PixelUnpackData::Slice(Some(bytes)));
                }
            }
            gl.active_texture(glow::TEXTURE3);
            gl.bind_texture(glow::TEXTURE_2D, depth_texture);
            gl.active_texture(glow::TEXTURE4);
            gl.bind_texture(glow::TEXTURE_2D, mask_texture);
            let given = Given {
                rect,
                video_time: frame.video_time,
                audio,
                has_depth: depth_texture.is_some() && self.depth.serial > 0,
                has_masks: mask_texture.is_some() && self.depth.has_masks,
                faces: &faces,
            };

            // The effects, each at its working size.
            self.slots[front].draw(gl, effect, &given);
            if let Some((from, _)) = &frame.fading {
                self.slots[back].draw(gl, from, &given);
            }

            // Onto the screen, scaled up if the effect works smaller. One that didn't compile
            // fades as the plain video.
            let shown = |slot: &Slot| slot.program.and(slot.out.as_ref()).map_or(video_texture, |out| out[slot.drawn].texture);
            let new = shown(&self.slots[front]);
            let (old, faded_in) = match &frame.fading {
                Some((_, amount)) => (shown(&self.slots[back]), *amount),
                None => (new, 1.0),
            };
            gl.bind_framebuffer(glow::FRAMEBUFFER, target);
            gl.viewport(0, 0, w, h);
            gl.use_program(Some(copy));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(old));
            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_2D, Some(new));
            gl.uniform_1_i32(gl.get_uniform_location(copy, "u_tex").as_ref(), 0);
            gl.uniform_1_i32(gl.get_uniform_location(copy, "u_from").as_ref(), 1);
            gl.uniform_1_f32(gl.get_uniform_location(copy, "u_faded_in").as_ref(), faded_in);
            // The window's rows run bottom-up like ours; an offscreen target's top-down.
            let flip = if target.is_some() { 1.0 } else { 0.0 };
            gl.uniform_1_f32(gl.get_uniform_location(copy, "u_flip").as_ref(), flip);
            gl.draw_arrays(glow::TRIANGLES, 0, 3);

            gl.bind_texture(glow::TEXTURE_2D, None);
            for unit in [glow::TEXTURE1, glow::TEXTURE2, glow::TEXTURE3, glow::TEXTURE4] {
                gl.active_texture(unit);
                gl.bind_texture(glow::TEXTURE_2D, None);
            }
            gl.active_texture(glow::TEXTURE0);
            gl.bind_vertex_array(None);
            gl.use_program(None);
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
            for slot in &mut self.slots {
                slot.free(gl);
            }
            if let Some(p) = self.copy.take() {
                gl.delete_program(p);
            }
            if let Some(t) = self.spectrum.take() {
                gl.delete_texture(t);
            }
            self.depth.release(gl);
            self.held.free(gl);
            self.faces.release(gl);
            self.cuts.release(gl);
            if let Some(vao) = self.vao.take() {
                gl.delete_vertex_array(vao);
            }
        }
    }

    /// How far behind the picture is being held (seconds), which the sound has to match.
    pub fn held_back(&self) -> f32 {
        self.depth.delay.max(self.faces.delay)
    }

    /// Whether a scene cut has reached `shown` in the video since this was last asked.
    pub fn cut_shown(&mut self, shown: f32) -> bool {
        self.cuts.due.take_if(|at| *at <= shown).is_some()
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
                    self.spectrum = create_spectrum(gl).map_err(|e| log::warn!("video effects: no spectrum texture: {e}")).ok();
                    Ok(())
                }
                Err(e) => {
                    gl.delete_program(copy);
                    Err(e)
                }
            }
        }
    }

    unsafe fn free_objects(&mut self, gl: &glow::Context) {
        unsafe { self.held.free(gl) };
        if let Some(o) = self.objects.take() {
            unsafe {
                gl.delete_framebuffer(o.video.fbo);
                gl.delete_texture(o.video.texture);
            }
        }
    }
}

/// What every effect drawn this frame is given, besides the textures already bound.
struct Given<'a> {
    rect: [f32; 4],
    video_time: f32,
    audio: Option<&'a Features>,
    has_depth: bool,
    has_masks: bool,
    faces: &'a FaceUniforms,
}

impl Slot {
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
                self.free_targets(gl);
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
                    let started = Instant::now();
                    match compile(gl, &vertex_source(decl), &fragment) {
                        Ok(program) => {
                            self.delete_program(gl);
                            self.program = Some(program);
                            log::info!("video effect {}: loaded {} in {} ms", effect.name, source.path.display(), started.elapsed().as_millis());
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

    /// Makes sure its output is `size`.
    unsafe fn prepare(&mut self, gl: &glow::Context, size: (i32, i32)) -> Result<(), String> {
        if self.out.is_some() && self.size == size {
            return Ok(());
        }
        unsafe {
            self.free_targets(gl);
            let rgba8 = (glow::RGBA8, glow::UNSIGNED_BYTE);
            // Feedback fades over many frames: 8 bits a channel would leave trails stuck at
            // faint colours, so half floats where the GPU can draw into them.
            let float = [(glow::RGBA16F, glow::HALF_FLOAT), rgba8];
            let a = create_target(gl, size, &float);
            let b = create_target(gl, size, &float);
            match (a, b) {
                (Ok(a), Ok(b)) => {
                    self.out = Some([a, b]);
                    self.size = size;
                    Ok(())
                }
                (a, b) => {
                    for t in [a, b].into_iter().flatten() {
                        gl.delete_framebuffer(t.fbo);
                        gl.delete_texture(t.texture);
                    }
                    Err("framebuffers incomplete".into())
                }
            }
        }
    }

    /// Runs the effect once. Texture units 0 and 2–4 are already bound.
    unsafe fn draw(&mut self, gl: &glow::Context, effect: &ActiveEffect, given: &Given) {
        let (Some(program), Some(targets)) = (self.program, &self.out) else { return };
        unsafe {
            let (out, prev) = (&targets[self.current], &targets[1 - self.current]);
            gl.bind_framebuffer(glow::FRAMEBUFFER, Some(out.fbo));
            gl.viewport(0, 0, self.size.0, self.size.1);
            gl.use_program(Some(program));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(prev.texture));
            let at = |name: &str| gl.get_uniform_location(program, name);
            gl.uniform_1_i32(at("u_video").as_ref(), 0);
            gl.uniform_1_i32(at("u_prev").as_ref(), 1);
            gl.uniform_2_f32(at("u_resolution").as_ref(), self.size.0 as f32, self.size.1 as f32);
            let started = *self.started.get_or_insert_with(Instant::now);
            gl.uniform_1_f32(at("u_time").as_ref(), started.elapsed().as_secs_f32());
            gl.uniform_1_i32(at("u_frame").as_ref(), self.frame);
            let rect = given.rect;
            gl.uniform_4_f32(at("u_video_rect").as_ref(), rect[0], rect[1], rect[2], rect[3]);
            gl.uniform_1_f32(at("u_video_time").as_ref(), given.video_time);
            let level = |f: fn(&Features) -> f32| given.audio.map_or(0.0, f);
            gl.uniform_1_f32(at("u_level").as_ref(), level(|a| a.level));
            gl.uniform_1_f32(at("u_bass").as_ref(), level(|a| a.bass));
            gl.uniform_1_f32(at("u_mid").as_ref(), level(|a| a.mid));
            gl.uniform_1_f32(at("u_treble").as_ref(), level(|a| a.treble));
            gl.uniform_1_f32(at("u_beat").as_ref(), level(|a| a.beat));
            gl.uniform_1_i32(at("u_spectrum").as_ref(), 2);
            gl.uniform_1_i32(at("u_depth").as_ref(), 3);
            gl.uniform_1_f32(at("u_has_depth").as_ref(), if given.has_depth { 1.0 } else { 0.0 });
            gl.uniform_1_i32(at("u_masks").as_ref(), 4);
            gl.uniform_1_f32(at("u_has_masks").as_ref(), if given.has_masks { 1.0 } else { 0.0 });
            let faces = given.faces;
            gl.uniform_1_i32(at("u_face_count").as_ref(), faces.count);
            gl.uniform_4_f32_slice(at("u_faces").as_ref(), &faces.boxes);
            gl.uniform_2_f32_slice(at("u_face_points").as_ref(), &faces.points);
            gl.uniform_1_f32_slice(at("u_face_seen").as_ref(), &faces.seen);
            for (name, value) in &effect.uniforms {
                gl.uniform_1_f32(at(name).as_ref(), *value);
            }
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
        }
        self.drawn = self.current;
        if effect.feedback {
            self.current = 1 - self.current;
        }
        self.frame = self.frame.wrapping_add(1);
    }

    unsafe fn delete_program(&mut self, gl: &glow::Context) {
        if let Some(p) = self.program.take() {
            unsafe { gl.delete_program(p) };
        }
    }

    unsafe fn free_targets(&mut self, gl: &glow::Context) {
        for t in self.out.take().into_iter().flatten() {
            unsafe {
                gl.delete_framebuffer(t.fbo);
                gl.delete_texture(t.texture);
            }
        }
    }

    /// Back to holding nothing.
    unsafe fn free(&mut self, gl: &glow::Context) {
        unsafe {
            self.delete_program(gl);
            self.free_targets(gl);
        }
        *self = Self::default();
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

/// A GL fence, only touched on the GL thread (the renderer sits in a Mutex shared with the
/// UI thread, which only reads its status).
#[derive(Clone, Copy)]
struct Fence(glow::Fence);

unsafe impl Send for Fence {}

/// Faces as the shader's uniforms.
#[derive(Clone, Copy)]
struct FaceUniforms {
    count: i32,
    boxes: [f32; crate::faces::MAX_FACES * 4],
    points: [f32; crate::faces::MAX_FACES * 10],
    seen: [f32; crate::faces::MAX_FACES],
}

impl Default for FaceUniforms {
    fn default() -> Self {
        Self { count: 0, boxes: [0.0; crate::faces::MAX_FACES * 4], points: [0.0; crate::faces::MAX_FACES * 10], seen: [0.0; crate::faces::MAX_FACES] }
    }
}

struct DepthObjects {
    /// The picture shrunk to the model's size, and a buffer to read it back through.
    capture: Target,
    pbo: glow::Buffer,
    /// The latest depth maps and the video frames they're of, for `u_depth`.
    maps: Vec<(glow::Texture, Option<f32>)>,
    /// `u_masks`, each to go with the depth map in the same place in `maps`.
    masks: Vec<glow::Texture>,
}

/// Depth maps kept, so a held-back frame can have the one made from it.
const DEPTH_MAPS: usize = 6;
/// The most the picture is held back (seconds).
const MOST_HELD: f32 = 0.25;
/// How much longer to hold than depth takes, and how far that has to drift before the hold
/// changes (moving it skips the picture and the sound a little).
const HOLD_MARGIN: f32 = 0.01;
const HOLD_STEP: f32 = 0.015;
/// Depth maps to see before trusting how long they take.
const HOLD_AFTER_MAPS: u32 = 20;
/// The most video frames kept, and the memory they can use.
const HELD_FRAMES: usize = 8;
const HELD_BYTES: usize = 256 << 20;

/// How long to hold the picture, given how late depth maps are arriving.
fn next_delay(current: f32, late: f32) -> f32 {
    let want = (late + HOLD_MARGIN).clamp(0.0, MOST_HELD);
    if (want - current).abs() > HOLD_STEP { want } else { current }
}

/// Of frames at these positions, newest first, the one to show at `at`: the newest that's
/// due, or the oldest if none are yet.
fn pick_frame(times: &[f32], at: f32) -> Option<usize> {
    times.iter().position(|t| *t <= at).or(times.len().checked_sub(1))
}

/// The last few video frames, so the picture can wait for depth.
#[derive(Default)]
struct HeldFrames {
    slots: Vec<Target>,
    /// Newest first: which slot, and the frame's position.
    frames: std::collections::VecDeque<(usize, f32)>,
}

impl HeldFrames {
    /// Keeps a copy of this frame, if it's a new one.
    unsafe fn push(&mut self, gl: &glow::Context, video: &Target, size: (i32, i32), pts: f32) {
        unsafe {
            let newest = self.frames.front().map(|f| f.1);
            if newest == Some(pts) {
                return;
            }
            // A seek: what's kept is from somewhere else.
            if newest.is_some_and(|n| pts < n || pts > n + 1.0) {
                self.frames.clear();
            }
            if self.slots.is_empty() {
                let most = (HELD_BYTES / (size.0.max(1) as usize * size.1.max(1) as usize * 4)).clamp(2, HELD_FRAMES);
                for _ in 0..most {
                    match create_target(gl, size, &[(glow::RGBA8, glow::UNSIGNED_BYTE)]) {
                        Ok(t) => self.slots.push(t),
                        Err(e) => {
                            log::warn!("video effects: holding frames back: {e}");
                            break;
                        }
                    }
                }
            }
            if self.slots.is_empty() {
                return;
            }
            let slot = if self.frames.len() == self.slots.len() { self.frames.pop_back().map_or(0, |f| f.0) } else { self.frames.len() };
            gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(video.fbo));
            gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(self.slots[slot].fbo));
            gl.blit_framebuffer(0, 0, size.0, size.1, 0, 0, size.0, size.1, glow::COLOR_BUFFER_BIT, glow::NEAREST);
            gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            self.frames.push_front((slot, pts));
        }
    }

    /// The newest frame kept, to read from, and its position.
    fn newest(&self) -> Option<(glow::Framebuffer, f32)> {
        self.frames.front().map(|&(slot, time)| (self.slots[slot].fbo, time))
    }

    /// The frame to show at `at`, and its position.
    fn pick(&self, at: f32) -> Option<(glow::Texture, f32)> {
        let times: Vec<f32> = self.frames.iter().map(|f| f.1).collect();
        pick_frame(&times, at).map(|i| (self.slots[self.frames[i].0].texture, self.frames[i].1))
    }

    unsafe fn free(&mut self, gl: &glow::Context) {
        self.frames.clear();
        for t in self.slots.drain(..) {
            unsafe {
                gl.delete_framebuffer(t.fbo);
                gl.delete_texture(t.texture);
            }
        }
    }
}

/// Small frames read back for the depth worker (a frame late, so the CPU never waits for
/// the GPU), and the depth maps it sends back as a texture.
#[derive(Default)]
struct DepthGl {
    gl: Option<DepthObjects>,
    failed: bool,
    pending: Option<Fence>,
    /// The video time of the last frame sent, so a paused frame isn't sent again and again.
    last_time: Option<f32>,
    /// Whether masks have arrived.
    has_masks: bool,
    /// The newest depth map, and how many have arrived.
    serial: u64,
    arrived: u32,
    /// Which of `maps` the next depth map goes in.
    next_map: usize,
    /// The position of the frame being read back.
    pending_time: f32,
    /// How late depth maps arrive after their frame (seconds), on average.
    late: f32,
    /// How far the picture is being held back to match (seconds).
    delay: f32,
}

impl DepthGl {
    /// The depth map and masks made from the frame nearest `time`.
    fn map_for(&self, time: f32) -> (Option<glow::Texture>, Option<glow::Texture>) {
        let Some(o) = self.gl.as_ref() else { return (None, None) };
        let nearest = o.maps.iter().enumerate().filter_map(|(i, (_, at))| at.map(|at| (i, (at - time).abs()))).min_by(|a, b| a.1.total_cmp(&b.1)).map_or(0, |(i, _)| i);
        (o.maps.get(nearest).map(|m| m.0), o.masks.get(nearest).copied())
    }

    /// Picks up a frame read back last time and hands it over, stores a new depth map, and
    /// reads back this frame if the worker's free. `time` is the frame's position and `clock`
    /// the smooth playback time.
    #[allow(clippy::too_many_arguments)]
    unsafe fn update(&mut self, gl: &glow::Context, link: &crate::depth::Link, video: glow::Framebuffer, size: (i32, i32), rect: [f32; 4], time: f32, clock: f32) {
        use crate::depth::{HEIGHT, WIDTH};
        unsafe {
            if self.failed {
                return;
            }
            if self.gl.is_none() {
                match create_depth(gl) {
                    Ok(o) => self.gl = Some(o),
                    Err(e) => {
                        log::warn!("depth: {e}");
                        self.failed = true;
                        return;
                    }
                }
            }
            let Some(o) = self.gl.as_mut() else { return };
            let bytes = WIDTH * HEIGHT * 4;
            if let Some(Fence(fence)) = self.pending {
                match gl.client_wait_sync(fence, 0, 0) {
                    glow::TIMEOUT_EXPIRED => {}
                    status => {
                        gl.delete_sync(fence);
                        self.pending = None;
                        if status != glow::WAIT_FAILED {
                            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(o.pbo));
                            let ptr = gl.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, bytes as i32, glow::MAP_READ_BIT);
                            if !ptr.is_null() {
                                link.submit(std::slice::from_raw_parts(ptr, bytes).to_vec(), self.pending_time);
                                gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
                            }
                            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                        }
                    }
                }
            }
            if let Some(map) = link.latest().filter(|m| m.serial != self.serial) {
                let slot = self.next_map;
                self.next_map = (slot + 1) % o.maps.len();
                gl.bind_texture(glow::TEXTURE_2D, Some(o.maps[slot].0));
                gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
                let data: &[u8] = std::slice::from_raw_parts(map.values.as_ptr().cast(), map.values.len() * 4);
                let (w, h) = (WIDTH as i32, HEIGHT as i32);
                gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, w, h, glow::RED, glow::FLOAT, glow::PixelUnpackData::Slice(Some(data)));
                if let Some(masks) = &map.masks {
                    gl.bind_texture(glow::TEXTURE_2D, Some(o.masks[slot]));
                    gl.tex_sub_image_2d(glow::TEXTURE_2D, 0, 0, 0, w, h, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(masks)));
                    self.has_masks = true;
                }
                gl.bind_texture(glow::TEXTURE_2D, None);
                self.serial = map.serial;
                o.maps[slot].1 = Some(map.time);
                let late = clock - map.time;
                // Outside this, it's a seek and says nothing about how long depth takes.
                if (0.0..0.5).contains(&late) {
                    self.arrived += 1;
                    self.late = if self.arrived == 1 { late } else { self.late + (late - self.late) * 0.1 };
                    if self.arrived >= HOLD_AFTER_MAPS {
                        self.delay = next_delay(self.delay, self.late);
                    }
                }
            }
            if self.pending.is_none() && link.ready_for_frame() && self.last_time != Some(time) {
                self.last_time = Some(time);
                self.pending_time = time;
                // Just the picture, without black bars.
                let (x0, y0) = ((rect[0] * size.0 as f32) as i32, (rect[1] * size.1 as f32) as i32);
                let (x1, y1) = ((rect[2] * size.0 as f32) as i32, (rect[3] * size.1 as f32) as i32);
                gl.disable(glow::SCISSOR_TEST);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(video));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(o.capture.fbo));
                gl.blit_framebuffer(x0, y0, x1.max(x0 + 1), y1.max(y0 + 1), 0, 0, WIDTH as i32, HEIGHT as i32, glow::COLOR_BUFFER_BIT, glow::LINEAR);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(o.capture.fbo));
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(o.pbo));
                gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
                gl.read_pixels(0, 0, WIDTH as i32, HEIGHT as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::BufferOffset(0));
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                if let Ok(fence) = gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) {
                    self.pending = Some(Fence(fence));
                }
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
        }
    }

    unsafe fn release(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some(Fence(fence)) = self.pending.take() {
                gl.delete_sync(fence);
            }
            if let Some(o) = self.gl.take() {
                gl.delete_framebuffer(o.capture.fbo);
                gl.delete_texture(o.capture.texture);
                gl.delete_buffer(o.pbo);
                for (texture, _) in o.maps {
                    gl.delete_texture(texture);
                }
                for texture in o.masks {
                    gl.delete_texture(texture);
                }
            }
            self.has_masks = false;
            self.serial = 0;
            self.arrived = 0;
            self.next_map = 0;
            self.delay = 0.0;
            self.last_time = None;
        }
    }
}

/// Small frames read back for the face worker, and what it finds, eased from one report to
/// the next so faces glide at the screen's rate.
#[derive(Default)]
struct FaceGl {
    objects: Option<(Target, glow::Buffer)>,
    failed: bool,
    pending: Option<(Fence, (f32, f32))>,
    /// The position of the last frame sent, and of the one being read back.
    last_time: Option<f32>,
    pending_time: f32,
    /// What's come in lately, oldest first.
    reports: VecDeque<crate::faces::Report>,
    arrived: u32,
    /// How late reports arrive after their frame (seconds), on average.
    late: f32,
    /// How far the picture is being held back to match (seconds).
    delay: f32,
}

/// The tiny picture compared from frame to frame to find cuts.
const CUT_SIZE: (i32, i32) = (32, 18);
/// A halfway size, as one big step down would only look at a few pixels of the frame.
const CUT_HALFWAY: (i32, i32) = (128, 72);
/// A cut changes the picture this much on average (0–255 a channel), and this many times as much
/// as the frames before it changed, which rules out fast action.
const CUT_CHANGE: f32 = 20.0;
const CUT_OVER_USUAL: f32 = 4.0;

/// Spots a sudden change in the picture.
#[derive(Default)]
struct CutFinder {
    last: Vec<u8>,
    last_time: f32,
    usual: f32,
    was_cut: bool,
}

impl CutFinder {
    /// `rgba` is this frame's tiny picture, from `time` in the video.
    fn is_cut(&mut self, rgba: &[u8], time: f32) -> bool {
        // A seek: not the film cutting.
        let continues = self.last.len() == rgba.len() && time >= self.last_time && time < self.last_time + 1.0;
        let mut cut = false;
        if continues {
            let change: u32 = self.last.as_chunks::<4>().0.iter().zip(rgba.as_chunks::<4>().0).map(|(a, b)| (0..3).map(|c| u32::from(a[c].abs_diff(b[c]))).sum::<u32>()).sum();
            let change = change as f32 / (rgba.len() / 4 * 3) as f32;
            cut = !self.was_cut && change > CUT_CHANGE && change > self.usual * CUT_OVER_USUAL;
            if !cut {
                self.usual += (change - self.usual) * 0.25;
            }
        } else {
            self.usual = 0.0;
        }
        self.last.clear();
        self.last.extend_from_slice(rgba);
        self.last_time = time;
        self.was_cut = cut;
        cut
    }
}

/// Reads back a tiny copy of each new frame (a frame late, so the CPU never waits for the GPU).
#[derive(Default)]
struct CutGl {
    objects: Option<(Target, Target, glow::Buffer)>,
    failed: bool,
    pending: Option<(Fence, f32)>,
    last_time: Option<f32>,
    finder: CutFinder,
    /// The newest cut found that hasn't reached the screen yet.
    due: Option<f32>,
}

impl CutGl {
    unsafe fn update(&mut self, gl: &glow::Context, video: glow::Framebuffer, size: (i32, i32), rect: [f32; 4], time: f32) {
        let bytes = (CUT_SIZE.0 * CUT_SIZE.1 * 4) as usize;
        let rgba8 = [(glow::RGBA8, glow::UNSIGNED_BYTE)];
        unsafe {
            if self.failed {
                return;
            }
            if self.objects.is_none() {
                let made = create_target(gl, CUT_HALFWAY, &rgba8).and_then(|halfway| {
                    let small = create_target(gl, CUT_SIZE, &rgba8)?;
                    let pbo = gl.create_buffer()?;
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
                    gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, bytes as i32, glow::STREAM_READ);
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                    Ok((halfway, small, pbo))
                });
                match made {
                    Ok(o) => self.objects = Some(o),
                    Err(e) => {
                        log::warn!("scene cuts: {e}");
                        self.failed = true;
                        return;
                    }
                }
            }
            let Some((halfway, small, pbo)) = self.objects.as_ref() else { return };
            if let Some((Fence(fence), at)) = self.pending {
                match gl.client_wait_sync(fence, 0, 0) {
                    glow::TIMEOUT_EXPIRED => {}
                    status => {
                        gl.delete_sync(fence);
                        self.pending = None;
                        if status != glow::WAIT_FAILED {
                            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(*pbo));
                            let ptr = gl.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, bytes as i32, glow::MAP_READ_BIT);
                            if !ptr.is_null() {
                                if self.finder.is_cut(std::slice::from_raw_parts(ptr, bytes), at) {
                                    log::debug!("scene cut at {at:.3}");
                                    self.due = Some(at);
                                }
                                gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
                            }
                            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                        }
                    }
                }
            }
            if self.pending.is_none() && self.last_time != Some(time) {
                self.last_time = Some(time);
                let (x0, y0) = ((rect[0] * size.0 as f32) as i32, (rect[1] * size.1 as f32) as i32);
                let (x1, y1) = (((rect[2] * size.0 as f32) as i32).max(x0 + 1), ((rect[3] * size.1 as f32) as i32).max(y0 + 1));
                gl.disable(glow::SCISSOR_TEST);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(video));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(halfway.fbo));
                gl.blit_framebuffer(x0, y0, x1, y1, 0, 0, CUT_HALFWAY.0, CUT_HALFWAY.1, glow::COLOR_BUFFER_BIT, glow::LINEAR);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(halfway.fbo));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(small.fbo));
                gl.blit_framebuffer(0, 0, CUT_HALFWAY.0, CUT_HALFWAY.1, 0, 0, CUT_SIZE.0, CUT_SIZE.1, glow::COLOR_BUFFER_BIT, glow::LINEAR);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(small.fbo));
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(*pbo));
                gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
                gl.read_pixels(0, 0, CUT_SIZE.0, CUT_SIZE.1, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::BufferOffset(0));
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                if let Ok(fence) = gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) {
                    self.pending = Some((Fence(fence), time));
                }
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
        }
    }

    unsafe fn release(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some((Fence(fence), _)) = self.pending.take() {
                gl.delete_sync(fence);
            }
            if let Some((halfway, small, pbo)) = self.objects.take() {
                for t in [halfway, small] {
                    gl.delete_framebuffer(t.fbo);
                    gl.delete_texture(t.texture);
                }
                gl.delete_buffer(pbo);
            }
        }
        self.last_time = None;
        self.finder = CutFinder::default();
        self.due = None;
    }
}

/// Extra hold for faces, as `late` is an average and a frame shown before its report has to
/// make do with the last frame's.
const FACE_SPARE: f32 = 0.025;

/// How many reports are kept to draw from: over a second's worth.
const FACE_REPORTS: usize = 32;

/// The faces at `time`, between the reports either side of it.
fn faces_between(reports: &VecDeque<crate::faces::Report>, time: f32) -> crate::faces::Slots {
    let none = [None; crate::faces::MAX_FACES];
    let (before, after) = match reports.iter().position(|r| r.time > time) {
        None => return reports.back().map_or(none, |r| r.slots),
        Some(0) => return reports[0].slots,
        Some(i) => (&reports[i - 1], &reports[i]),
    };
    if after.after_cut {
        return before.slots;
    }
    let along = (time - before.time) / (after.time - before.time).max(1e-6);
    let mut out = none;
    for (out, (a, b)) in out.iter_mut().zip(before.slots.iter().zip(&after.slots)) {
        *out = match (a, b) {
            (Some(a), Some(b)) => {
                let mut mixed = *a;
                for (v, b) in mixed.coords.iter_mut().zip(b.coords) {
                    *v += (b - *v) * along;
                }
                mixed.seen += (b.seen - a.seen) * along;
                Some(mixed)
            }
            (a, b) => a.or(*b),
        };
    }
    out
}

impl FaceGl {
    /// Picks up a frame read back last time and hands it over, collects new reports, and reads
    /// back this frame if the worker's free. `time` is the frame's position and `clock` the
    /// smooth playback time.
    #[allow(clippy::too_many_arguments)]
    unsafe fn update(&mut self, gl: &glow::Context, link: &crate::faces::Link, video: glow::Framebuffer, size: (i32, i32), rect: [f32; 4], time: f32, clock: f32) {
        use crate::faces::{HEIGHT, WIDTH};
        unsafe {
            if self.failed {
                return;
            }
            if self.objects.is_none() {
                let made = create_target(gl, (WIDTH as i32, HEIGHT as i32), &[(glow::RGBA8, glow::UNSIGNED_BYTE)]).and_then(|capture| {
                    let pbo = gl.create_buffer()?;
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
                    gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, (WIDTH * HEIGHT * 4) as i32, glow::STREAM_READ);
                    gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                    Ok((capture, pbo))
                });
                match made {
                    Ok(o) => self.objects = Some(o),
                    Err(e) => {
                        log::warn!("faces: {e}");
                        self.failed = true;
                        return;
                    }
                }
            }
            let Some((capture, pbo)) = self.objects.as_ref() else { return };
            let bytes = WIDTH * HEIGHT * 4;
            if let Some((Fence(fence), filled)) = self.pending {
                match gl.client_wait_sync(fence, 0, 0) {
                    glow::TIMEOUT_EXPIRED => {}
                    status => {
                        gl.delete_sync(fence);
                        self.pending = None;
                        if status != glow::WAIT_FAILED {
                            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(*pbo));
                            let ptr = gl.map_buffer_range(glow::PIXEL_PACK_BUFFER, 0, bytes as i32, glow::MAP_READ_BIT);
                            if !ptr.is_null() {
                                link.submit(std::slice::from_raw_parts(ptr, bytes).to_vec(), filled, self.pending_time);
                                gl.unmap_buffer(glow::PIXEL_PACK_BUFFER);
                            }
                            gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                        }
                    }
                }
            }
            for report in link.take_reports() {
                // Going backwards: what's kept is from somewhere else.
                if self.reports.back().is_some_and(|r| report.time < r.time) {
                    self.reports.clear();
                }
                let late = clock - report.time;
                // Outside this, it's a seek and says nothing about how long faces take.
                if (0.0..0.5).contains(&late) {
                    self.arrived += 1;
                    self.late = if self.arrived == 1 { late } else { self.late + (late - self.late) * 0.1 };
                    if self.arrived >= HOLD_AFTER_MAPS {
                        self.delay = next_delay(self.delay, self.late + FACE_SPARE);
                    }
                }
                self.reports.push_back(report);
                if self.reports.len() > FACE_REPORTS {
                    self.reports.pop_front();
                }
            }
            if self.pending.is_none() && link.ready_for_frame() && self.last_time != Some(time) {
                self.last_time = Some(time);
                self.pending_time = time;
                // The picture without black bars, its shape kept and the rest left black: a
                // squashed face is harder to find. In GL's rows, the top of the picture is
                // at the top of the capture.
                let (x0, y0) = ((rect[0] * size.0 as f32) as i32, (rect[1] * size.1 as f32) as i32);
                let (x1, y1) = (((rect[2] * size.0 as f32) as i32).max(x0 + 1), ((rect[3] * size.1 as f32) as i32).max(y0 + 1));
                let scale = (WIDTH as f32 / (x1 - x0) as f32).min(HEIGHT as f32 / (y1 - y0) as f32);
                let (w, h) = ((((x1 - x0) as f32 * scale) as i32).clamp(1, WIDTH as i32), (((y1 - y0) as f32 * scale) as i32).clamp(1, HEIGHT as i32));
                gl.disable(glow::SCISSOR_TEST);
                gl.bind_framebuffer(glow::FRAMEBUFFER, Some(capture.fbo));
                gl.clear_color(0.0, 0.0, 0.0, 1.0);
                gl.clear(glow::COLOR_BUFFER_BIT);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(video));
                gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, Some(capture.fbo));
                gl.blit_framebuffer(x0, y0, x1, y1, 0, HEIGHT as i32 - h, w, HEIGHT as i32, glow::COLOR_BUFFER_BIT, glow::LINEAR);
                gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(capture.fbo));
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(*pbo));
                gl.pixel_store_i32(glow::PACK_ALIGNMENT, 4);
                gl.read_pixels(0, 0, WIDTH as i32, HEIGHT as i32, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelPackData::BufferOffset(0));
                gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
                if let Ok(fence) = gl.fence_sync(glow::SYNC_GPU_COMMANDS_COMPLETE, 0) {
                    self.pending = Some((Fence(fence), (w as f32 / WIDTH as f32, h as f32 / HEIGHT as f32)));
                }
                gl.bind_framebuffer(glow::FRAMEBUFFER, None);
            }
        }
    }

    /// The faces in the frame at `time`, as the shader wants them.
    fn uniforms(&self, time: f32, rect: [f32; 4]) -> FaceUniforms {
        let mut out = FaceUniforms::default();
        for (i, face) in faces_between(&self.reports, time).iter().enumerate() {
            let Some(face) = face else { continue };
            let (width, height) = (rect[2] - rect[0], rect[3] - rect[1]);
            let x = |v: f32| rect[0] + v * width;
            let y = |v: f32| rect[3] - v * height;
            out.boxes[i * 4..i * 4 + 4].copy_from_slice(&[x(face.coords[0]), y(face.coords[3]), x(face.coords[2]), y(face.coords[1])]);
            for p in 0..5 {
                out.points[(i * 5 + p) * 2] = x(face.coords[4 + p * 2]);
                out.points[(i * 5 + p) * 2 + 1] = y(face.coords[5 + p * 2]);
            }
            out.seen[i] = face.seen;
            out.count = i as i32 + 1;
        }
        out
    }

    unsafe fn release(&mut self, gl: &glow::Context) {
        unsafe {
            if let Some((Fence(fence), _)) = self.pending.take() {
                gl.delete_sync(fence);
            }
            if let Some((capture, pbo)) = self.objects.take() {
                gl.delete_framebuffer(capture.fbo);
                gl.delete_texture(capture.texture);
                gl.delete_buffer(pbo);
            }
            self.last_time = None;
            self.reports.clear();
            self.arrived = 0;
            self.late = 0.0;
            self.delay = 0.0;
        }
    }
}

unsafe fn create_depth(gl: &glow::Context) -> Result<DepthObjects, String> {
    use crate::depth::{HEIGHT, WIDTH};
    unsafe {
        let size = (WIDTH as i32, HEIGHT as i32);
        let capture = create_target(gl, size, &[(glow::RGBA8, glow::UNSIGNED_BYTE)])?;
        let pbo = gl.create_buffer()?;
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, Some(pbo));
        gl.buffer_data_size(glow::PIXEL_PACK_BUFFER, (WIDTH * HEIGHT * 4) as i32, glow::STREAM_READ);
        gl.bind_buffer(glow::PIXEL_PACK_BUFFER, None);
        let zeros = vec![0u8; WIDTH * HEIGHT * 4];
        let mut maps = Vec::with_capacity(DEPTH_MAPS);
        let mut masks = Vec::with_capacity(DEPTH_MAPS);
        for _ in 0..DEPTH_MAPS {
            let mask = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(mask));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::RGBA8 as i32, size.0, size.1, 0, glow::RGBA, glow::UNSIGNED_BYTE, glow::PixelUnpackData::Slice(Some(&zeros)));
            for (param, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, param, value as i32);
            }
            masks.push(mask);
            let texture = gl.create_texture()?;
            gl.bind_texture(glow::TEXTURE_2D, Some(texture));
            gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R16F as i32, size.0, size.1, 0, glow::RED, glow::FLOAT, glow::PixelUnpackData::Slice(Some(&zeros)));
            for (param, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, param, value as i32);
            }
            maps.push((texture, None));
        }
        gl.bind_texture(glow::TEXTURE_2D, None);
        Ok(DepthObjects { capture, pbo, maps, masks })
    }
}

/// The spectrum: one float per band and history row, starting silent.
unsafe fn create_spectrum(gl: &glow::Context) -> Result<glow::Texture, String> {
    unsafe {
        let texture = gl.create_texture()?;
        gl.bind_texture(glow::TEXTURE_2D, Some(texture));
        let zeros = vec![0u8; audio::BANDS * audio::HISTORY * 4];
        let data = glow::PixelUnpackData::Slice(Some(&zeros));
        // Half floats: filterable everywhere (plain floats aren't on GLES).
        gl.tex_image_2d(glow::TEXTURE_2D, 0, glow::R16F as i32, audio::BANDS as i32, audio::HISTORY as i32, 0, glow::RED, glow::FLOAT, data);
        for (param, value) in [
            (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
            (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
            (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
            (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
        ] {
            gl.tex_parameter_i32(glow::TEXTURE_2D, param, value as i32);
        }
        gl.bind_texture(glow::TEXTURE_2D, None);
        Ok(texture)
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

#[cfg(test)]
mod hold_tests {
    use super::*;

    fn report(time: f32, x: f32, after_cut: bool) -> crate::faces::Report {
        let mut slots = [None; crate::faces::MAX_FACES];
        slots[0] = Some(crate::faces::Shown { coords: [x; crate::faces::COORDS], seen: 1.0 });
        crate::faces::Report { time, after_cut, slots }
    }

    #[test]
    fn a_sudden_change_is_a_cut_but_steady_action_and_seeks_are_not() {
        let frame = |v: u8| vec![v; 64];
        let mut f = CutFinder::default();
        assert!(!f.is_cut(&frame(50), 0.0), "nothing to compare with");
        assert!(!f.is_cut(&frame(52), 0.04));
        assert!(f.is_cut(&frame(150), 0.08));
        assert!(!f.is_cut(&frame(151), 0.12));
        assert!(!f.is_cut(&frame(250), 5.0), "a seek");
        // Everything changing a lot, frame after frame.
        let mut f = CutFinder::default();
        let mut cuts = 0;
        for i in 0..60 {
            cuts += u32::from(f.is_cut(&frame(if i % 2 == 0 { 40 } else { 80 }), i as f32 * 0.04));
        }
        assert!(cuts <= 1, "{cuts}");
    }

    #[test]
    fn faces_are_drawn_where_they_were_at_that_frame() {
        let reports = VecDeque::from([report(1.0, 0.2, false), report(2.0, 0.4, false), report(3.0, 0.9, true)]);
        let x = |t: f32| faces_between(&reports, t)[0].unwrap().coords[0];
        assert!((x(1.5) - 0.3).abs() < 1e-5, "between two reports");
        assert_eq!(x(0.5), 0.2, "before them all");
        assert_eq!(x(4.0), 0.9, "after them all");
        assert_eq!(x(2.5), 0.4, "not across a cut");
        assert!(faces_between(&VecDeque::new(), 1.0)[0].is_none());
    }

    #[test]
    fn the_newest_frame_that_is_due_is_shown() {
        let newest_first = [1.25, 1.2, 1.15, 1.1];
        assert_eq!(pick_frame(&newest_first, 1.22), Some(1));
        assert_eq!(pick_frame(&newest_first, 2.0), Some(0));
        // Nothing's due yet: the oldest there is.
        assert_eq!(pick_frame(&newest_first, 1.0), Some(3));
        assert_eq!(pick_frame(&[], 1.0), None);
    }

    #[test]
    fn the_hold_only_moves_when_depth_drifts_enough() {
        assert_eq!(next_delay(0.0, 0.07), 0.07 + HOLD_MARGIN);
        let settled = 0.08;
        assert_eq!(next_delay(settled, 0.075), settled);
        assert_eq!(next_delay(settled, 0.11), 0.11 + HOLD_MARGIN);
        assert_eq!(next_delay(0.0, 2.0), MOST_HELD);
    }
}
