//! Video effects: shaders Flex runs over the video (the drawing is in the app). Each is
//! a `<id>.frag` in an `effects` folder, with an optional `<id>.toml` for its name,
//! options and settings. One effect is chosen at a time, and they can all be switched off
//! without forgetting it; its settings are kept per effect. Shuffle swaps the chosen effect
//! at random, fading from one to the next.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use flex_plugin::Settings;
use serde::{Deserialize, Serialize};

use crate::Db;
use crate::plugins::{SettingDrift, SettingKind, SettingSpec};

/// `<id>.toml`.
#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    /// Shown in Settings; defaults to the file's name.
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Give the shader its own previous output (`u_prev`), for trails and feedback.
    #[serde(default)]
    pub feedback: bool,
    /// Give the shader the sound, as measured by a filter inside mpv.
    #[serde(default)]
    pub audio: bool,
    /// Give the shader depth (`u_depth`), from a neural network on the GPU. Costs GPU time
    /// and a 99 MB download the first time.
    #[serde(default)]
    pub depth: bool,
    /// Give the shader where people, walls and sky are (`person_at`, `wall_at`, `sky_at`), from
    /// a small network that runs on top of depth's.
    #[serde(default)]
    pub segment: bool,
    /// Give the shader where faces are (`u_faces`, `face_at`), from a small neural network on
    /// the GPU.
    #[serde(default)]
    pub faces: bool,
    /// Keep the last second or so of frames for `history_at`.
    #[serde(default)]
    pub history: bool,
    /// Work out how things are moving, for `flow_at`.
    #[serde(default)]
    pub flow: bool,
    /// Build up the room behind people, for `plate_at`. Runs the masks.
    #[serde(default)]
    pub plate: bool,
    /// Find scene cuts, for `u_cut` and `u_since_cut`.
    #[serde(default)]
    pub cuts: bool,
    /// Measure the picture's palette and brightness.
    #[serde(default)]
    pub stats: bool,
    /// Size to work at, relative to the screen (0.25–1): lower is lighter on the GPU.
    #[serde(default = "full")]
    pub scale: f32,
    /// Whether shuffle picks this effect unless told otherwise: false for test views.
    #[serde(default = "yes")]
    pub shuffle: bool,
    #[serde(default)]
    pub settings: Vec<SettingSpec>,
}

fn full() -> f32 {
    1.0
}

fn yes() -> bool {
    true
}

impl Default for Manifest {
    fn default() -> Self {
        Self { name: String::new(), description: String::new(), feedback: false, audio: false, depth: false, segment: false, faces: false, history: false, flow: false, plate: false, cuts: false, stats: false, scale: 1.0, shuffle: true, settings: Vec::new() }
    }
}

/// An effect found on disk.
#[derive(Clone, Debug)]
pub struct Effect {
    /// The file's name without `.frag`.
    pub id: String,
    pub shader: PathBuf,
    pub manifest: Manifest,
    /// Its `.toml` couldn't be read (it runs with the defaults).
    pub problem: Option<String>,
}

impl Effect {
    /// Settings Flex can show and pass to the shader (not text or buttons).
    pub fn settings(&self) -> impl Iterator<Item = &SettingSpec> {
        self.manifest.settings.iter().filter(|s| matches!(s.kind, SettingKind::Number | SettingKind::Bool | SettingKind::Choice))
    }
}

/// Where effects are looked for: `effects` next to Flex's program (the ones a release
/// ships), the source folder in debug builds (so `cargo run` sees edits as they're
/// made), then `effects` in the data folder, which wins when two have the same name.
pub fn effect_dirs() -> Vec<PathBuf> {
    let beside = std::env::current_exe().ok().and_then(|exe| exe.parent().map(|d| d.join("effects")));
    let source = cfg!(debug_assertions).then(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../effects"));
    beside.into_iter().chain(source).chain([user_effect_dir()]).collect()
}

pub fn user_effect_dir() -> PathBuf {
    crate::data_dir().join("effects")
}

/// "my-cool_effect" → "My cool effect".
fn name_from_id(id: &str) -> String {
    let words = id.replace(['-', '_'], " ");
    let mut chars = words.chars();
    chars.next().map(|c| c.to_uppercase().chain(chars).collect()).unwrap_or_default()
}

/// Every effect in `dirs`, sorted by name; later folders replace earlier ones.
pub fn discover(dirs: &[PathBuf]) -> Vec<Effect> {
    let mut found: Vec<Effect> = Vec::new();
    for dir in dirs {
        let Ok(entries) = std::fs::read_dir(dir) else { continue };
        let mut shaders: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "frag") && p.is_file()).collect();
        shaders.sort();
        for shader in shaders {
            let Some(id) = shader.file_stem().map(|s| s.to_string_lossy().into_owned()) else { continue };
            let toml_path = shader.with_extension("toml");
            let (mut manifest, problem) = match std::fs::read_to_string(&toml_path) {
                Err(_) => (Manifest::default(), None),
                Ok(text) => match toml::from_str::<Manifest>(&text) {
                    Ok(m) => (m, None),
                    Err(e) => {
                        log::warn!("video effect {}: {e}", toml_path.display());
                        (Manifest::default(), Some(format!("{id}.toml: {}", e.message())))
                    }
                },
            };
            if manifest.name.trim().is_empty() {
                manifest.name = name_from_id(&id);
            }
            found.retain(|e| e.id != id);
            found.push(Effect { id, shader, manifest, problem });
        }
    }
    found.sort_by_key(|e| e.manifest.name.to_lowercase());
    found
}

/// The effect to draw, with its settings as shader uniforms.
#[derive(Clone, Debug, PartialEq)]
pub struct ActiveEffect {
    pub id: String,
    pub name: String,
    pub shader: PathBuf,
    pub feedback: bool,
    pub audio: bool,
    pub depth: bool,
    pub segment: bool,
    pub faces: bool,
    pub scale: f32,
    /// `s_<key>` = number, bool (0/1) or choice (its index).
    pub uniforms: Vec<(String, f32)>,
}

/// What drawing the video needs, across every effect that's on or about to be.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Needs {
    pub audio: bool,
    /// Depth, which masks are made on top of.
    pub depth: bool,
    pub segment: bool,
    pub faces: bool,
    pub history: bool,
    pub flow: bool,
    pub plate: bool,
    pub cuts: bool,
    pub stats: bool,
    /// The largest size any of them works at (0.25–1).
    pub scale: f32,
}

/// Effects taking turns at random.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Shuffle {
    pub on: bool,
    /// How long each effect stays, in seconds, not counting the fade: somewhere between these.
    pub shortest: f32,
    pub longest: f32,
    /// How long one effect takes to fade into the next, in seconds: 0 is straight away.
    pub fade: f32,
    /// Change where the shot changes, once the shortest turn is up, instead of at a random
    /// moment. After the longest turn it changes anyway.
    pub at_cuts: bool,
    /// How far each effect's settings are moved at random whenever it comes up, in percent:
    /// 0 leaves them alone, 100 can land anywhere in their range.
    pub randomise: f32,
}

impl Default for Shuffle {
    fn default() -> Self {
        Self { on: false, shortest: 20.0, longest: 60.0, fade: 4.0, at_cuts: false, randomise: 0.0 }
    }
}

/// Settings wandering slowly on their own, for whichever effect is on.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Drift {
    /// How far they go, in percent: 100 can reach anywhere in their range.
    pub amount: f32,
    /// How fast, in percent of the usual pace.
    pub speed: f32,
}

impl Default for Drift {
    fn default() -> Self {
        Self { amount: 0.0, speed: 100.0 }
    }
}

/// How long a setting takes to wander there and back at the usual pace, in seconds: from this
/// long, up to a bit under twice as quick.
const DRIFT_PERIOD: f64 = 90.0;

/// An effect on its way out.
struct Fading {
    from: String,
    elapsed: f32,
}

const CHOSEN_KEY: &str = "video_effect";
/// "0" when effects are switched off. Before this existed, off was having none chosen.
const ON_KEY: &str = "video_effects_on";
const SHUFFLE_KEY: &str = "video_effect_shuffle";
const DRIFT_KEY: &str = "video_effect_drift";
/// Kept with an effect's settings: whether shuffle may pick it.
const IN_SHUFFLE_SETTING: &str = "_in_shuffle";
/// Kept with an effect's settings: whether it's in the favourites.
const FAVOURITE_SETTING: &str = "_favourite";

fn settings_key(id: &str) -> String {
    format!("video_effect.{id}")
}

/// The effects found, which one is on, and their settings.
#[derive(Default)]
pub struct Effects {
    effects: Vec<Effect>,
    /// The chosen effect's id (kept even if its file has gone, or effects are off).
    chosen: Option<String>,
    /// Whether effects are switched on at all.
    on: bool,
    settings: HashMap<String, Settings>,
    /// Random settings for this turn of shuffle, in place of `settings`, and never saved.
    rolled: HashMap<String, Settings>,
    /// The chosen effect's shader state, from the app: (ok, text).
    pub status: Option<(bool, String)>,
    shuffle: Shuffle,
    /// Seconds of playing until shuffle moves on, and since it last did.
    until_next: f32,
    turn: f32,
    fading: Option<Fading>,
    /// What shuffle hasn't played yet this time round.
    turns: Vec<String>,
    random: u64,
    drift: Drift,
    /// How far along drifting is, which only moves while playing.
    drift_clock: f64,
    /// `drift.amount` (as 0–1), eased, so turning it on doesn't jolt the picture.
    drift_now: f64,
}

impl Effects {
    pub fn new(dirs: &[PathBuf], db: &Db) -> Self {
        let chosen = db.setting(CHOSEN_KEY).ok().flatten().filter(|s| !s.is_empty());
        let shuffle = db.setting(SHUFFLE_KEY).ok().flatten().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let drift = db.setting(DRIFT_KEY).ok().flatten().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
        let on = db.setting(ON_KEY).ok().flatten().map_or(chosen.is_some(), |s| s != "0");
        let mut effects = Self { chosen, on, shuffle, drift, ..Self::default() };
        effects.refresh(dirs, db);
        effects.wait();
        effects
    }

    /// Looks for effects again (new, changed or removed files).
    pub fn refresh(&mut self, dirs: &[PathBuf], db: &Db) {
        self.effects = discover(dirs);
        for e in &self.effects {
            let mut settings: Settings = e.manifest.settings.iter().filter_map(|s| Some((s.key.clone(), s.default.clone()?))).collect();
            let saved = db.setting(&settings_key(&e.id)).ok().flatten();
            if let Some(Ok(serde_json::Value::Object(saved))) = saved.map(|s| serde_json::from_str(&s)) {
                settings.extend(saved);
            }
            self.settings.insert(e.id.clone(), settings);
        }
        log::info!("video effects: {}", self.effects.iter().map(|e| e.id.as_str()).collect::<Vec<_>>().join(", "));
    }

    pub fn list(&self) -> &[Effect] {
        &self.effects
    }

    /// The effect that's on: none while they're switched off.
    pub fn chosen(&self) -> Option<&Effect> {
        self.remembered().filter(|_| self.on)
    }

    /// The effect that's on, or would be if they were switched back on.
    pub fn remembered(&self) -> Option<&Effect> {
        let id = self.chosen.as_deref()?;
        self.effects.iter().find(|e| e.id == id)
    }

    pub fn is_on(&self) -> bool {
        self.on
    }

    /// Switches effects off, keeping which one, shuffle and drift for when they come back.
    pub fn set_on(&mut self, on: bool, db: &Db) {
        self.on = on;
        self.status = None;
        self.fading = None;
        if on && self.remembered().is_none() {
            self.chosen = self.next_turn();
            self.save_chosen(db);
        }
        self.wait();
        self.save_on(db);
    }

    fn save_on(&self, db: &Db) {
        if let Err(e) = db.set_setting(ON_KEY, if self.on { "1" } else { "0" }) {
            log::warn!("saving whether video effects are on: {e:#}");
        }
    }

    /// Someone chose an effect: it shows at once, and with shuffle on gets a full turn.
    /// Choosing none switches effects off.
    pub fn choose(&mut self, id: Option<&str>, db: &Db) {
        let Some(id) = id else {
            self.set_on(false, db);
            return;
        };
        self.chosen = Some(id.to_owned());
        self.on = true;
        self.status = None;
        self.fading = None;
        self.rolled.clear();
        self.wait();
        self.save_chosen(db);
        self.save_on(db);
    }

    fn save_chosen(&self, db: &Db) {
        if let Err(e) = db.set_setting(CHOSEN_KEY, self.chosen.as_deref().unwrap_or("")) {
            log::warn!("saving the video effect: {e:#}");
        }
    }

    pub fn shuffle(&self) -> Shuffle {
        self.shuffle
    }

    pub fn set_shuffle(&mut self, shuffle: Shuffle, db: &Db) {
        let (started, stopped) = (shuffle.on && !self.shuffle.on, !shuffle.on && self.shuffle.on);
        let restart = started || shuffle.at_cuts != self.shuffle.at_cuts;
        self.shuffle = shuffle;
        self.save_shuffle(db);
        if started && self.chosen().is_none() {
            self.on = true;
            self.save_on(db);
            if self.remembered().is_none() {
                self.chosen = self.next_turn();
            }
            self.status = None;
        }
        if restart {
            self.wait();
        }
        if !shuffle.on || shuffle.randomise <= 0.0 {
            self.rolled.clear();
        }
        if stopped {
            // Stay on whichever came up.
            self.save_chosen(db);
        }
    }

    fn save_shuffle(&self, db: &Db) {
        let saved = serde_json::to_string(&self.shuffle).map_err(anyhow::Error::from).and_then(|json| db.set_setting(SHUFFLE_KEY, &json));
        if let Err(e) = saved {
            log::warn!("saving the video effect shuffle: {e:#}");
        }
    }

    pub fn drift(&self) -> Drift {
        self.drift
    }

    pub fn set_drift(&mut self, drift: Drift, db: &Db) {
        self.drift = drift;
        let saved = serde_json::to_string(&drift).map_err(anyhow::Error::from).and_then(|json| db.set_setting(DRIFT_KEY, &json));
        if let Err(e) = saved {
            log::warn!("saving the video effect drift: {e:#}");
        }
    }

    /// A number setting as it is right now, having drifted from `base`.
    fn drifted(&self, id: &str, spec: &SettingSpec, base: f64) -> f64 {
        if self.drift_now <= 0.0 || spec.drift == SettingDrift::Off {
            return base;
        }
        // Each setting of each effect goes its own way.
        let mut seed = 0xcbf29ce484222325u64;
        for byte in id.bytes().chain(spec.key.bytes()) {
            seed = (seed ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
        let part = |shift: u32| ((seed >> shift) & 0xffff) as f64 / 65536.0;
        let tau = std::f64::consts::TAU;
        let turns = self.drift_clock / DRIFT_PERIOD * (1.0 + 0.8 * part(0));
        let wave = 0.65 * (tau * (turns + part(16))).sin() + 0.35 * (tau * (turns * 2.37 + part(32))).sin();
        let (min, max) = (spec.min.unwrap_or(0.0), spec.max.unwrap_or(100.0));
        let moved = base + (min + (0.5 + 0.5 * wave) * (max - min) - base) * self.drift_now;
        let steps = match spec.drift {
            SettingDrift::Steps => true,
            SettingDrift::Auto => spec.unit.is_empty() && spec.step.is_some_and(|s| s >= 1.0),
            _ => false,
        };
        match spec.step.filter(|_| steps) {
            Some(step) => (min + ((moved - min) / step).round() * step).clamp(min, max),
            None => moved,
        }
    }

    /// A number setting of the chosen effect as it is right now.
    pub fn live_number(&self, spec: &SettingSpec) -> Option<f64> {
        let id = &self.chosen()?.id;
        Some(self.drifted(id, spec, self.value(id, &spec.key)?.as_f64()?))
    }

    pub fn in_shuffle(&self, id: &str) -> bool {
        let default = self.effects.iter().find(|e| e.id == id).is_some_and(|e| e.manifest.shuffle);
        self.value(id, IN_SHUFFLE_SETTING).and_then(serde_json::Value::as_bool).unwrap_or(default)
    }

    pub fn set_in_shuffle(&mut self, id: &str, included: bool, db: &Db) {
        self.set_setting(id, IN_SHUFFLE_SETTING, serde_json::json!(included), db);
    }

    pub fn is_favourite(&self, id: &str) -> bool {
        self.value(id, FAVOURITE_SETTING).and_then(serde_json::Value::as_bool).unwrap_or(false)
    }

    pub fn set_favourite(&mut self, id: &str, favourite: bool, db: &Db) {
        self.set_setting(id, FAVOURITE_SETTING, serde_json::json!(favourite), db);
    }

    /// Call once per frame, with the seconds since the last. The fade keeps going while
    /// paused, but only playing counts towards the next change.
    pub fn step(&mut self, dt: f32, playing: bool) {
        if !self.on {
            return;
        }
        if playing {
            self.turn += dt;
            self.drift_clock += f64::from(dt * self.drift.speed / 100.0);
        }
        let aim = f64::from(self.drift.amount / 100.0).clamp(0.0, 1.0);
        self.drift_now += (aim - self.drift_now) * f64::from(dt * 0.4).min(1.0);
        if let Some(fading) = &mut self.fading {
            fading.elapsed += dt;
            if fading.elapsed < self.shuffle.fade {
                return;
            }
            self.fading = None;
        }
        if !self.shuffle.on || !playing {
            return;
        }
        self.until_next -= dt;
        // Don't sit on one that doesn't compile, or whose file has gone.
        let broken = matches!(self.status, Some((false, _))) || self.chosen().is_none();
        if self.until_next > 0.0 && !broken {
            return;
        }
        self.change_turn(!broken);
    }

    /// Picks this turn's settings for an effect, each somewhere between how it's been set and
    /// a random point in its range.
    fn roll_settings(&mut self, id: &str) {
        let amount = f64::from(self.shuffle.randomise / 100.0).clamp(0.0, 1.0);
        let Some(effect) = self.effects.iter().find(|e| e.id == id).cloned().filter(|_| amount > 0.0) else { return };
        let mut rolled = Settings::default();
        for spec in effect.settings() {
            let current = self.settings.get(id).and_then(|s| s.get(&spec.key));
            let value = match spec.kind {
                SettingKind::Number => {
                    let (min, max) = (spec.min.unwrap_or(0.0), spec.max.unwrap_or(100.0));
                    let base = current.and_then(serde_json::Value::as_f64).unwrap_or(min);
                    let anywhere = min + f64::from(self.roll()) * (max - min);
                    let moved = base + (anywhere - base) * amount;
                    let snapped = spec.step.filter(|s| *s > 0.0).map_or(moved, |step| min + ((moved - min) / step).round() * step);
                    let value = ((snapped * 1e4).round() / 1e4).clamp(min, max);
                    if value.fract() == 0.0 { serde_json::json!(value as i64) } else { serde_json::json!(value) }
                }
                // Flipping half the time at most, so all the way up isn't always opposite.
                SettingKind::Bool => {
                    let on = current.and_then(serde_json::Value::as_bool).unwrap_or(false);
                    serde_json::json!(on != (f64::from(self.roll()) < amount * 0.5))
                }
                SettingKind::Choice if !spec.options.is_empty() && f64::from(self.roll()) < amount => {
                    let at = (self.roll() * spec.options.len() as f32) as usize;
                    serde_json::json!(spec.options[at.min(spec.options.len() - 1)].id)
                }
                _ => continue,
            };
            rolled.insert(spec.key.clone(), value);
        }
        self.rolled.insert(id.to_owned(), rolled);
    }

    fn change_turn(&mut self, may_fade: bool) {
        self.wait();
        let Some(next) = self.next_turn() else { return };
        self.rolled.retain(|id, _| Some(id) == self.chosen.as_ref());
        self.roll_settings(&next);
        let from = self.chosen.replace(next);
        self.status = None;
        self.fading = from.filter(|_| may_fade && self.shuffle.fade > 0.0).map(|from| Fading { from, elapsed: 0.0 });
    }

    /// Whether the player should look for scene cuts.
    pub fn wants_cuts(&self) -> bool {
        self.on && self.shuffle.on && self.shuffle.at_cuts
    }

    /// The shot has just changed: moves to the next effect if this one has had its turn.
    pub fn scene_changed(&mut self) {
        let shortest = self.shuffle.shortest.min(self.shuffle.longest);
        if !self.wants_cuts() || self.turn < shortest {
            return;
        }
        self.change_turn(true);
    }

    /// The effect fading out, and how far `active` has faded in over it (0–1), during a fade.
    pub fn fading(&self) -> Option<(ActiveEffect, f32)> {
        let fading = self.fading.as_ref()?;
        let from = self.effects.iter().find(|e| e.id == fading.from)?;
        let t = (fading.elapsed / self.shuffle.fade.max(0.01)).clamp(0.0, 1.0);
        Some((self.to_draw(from), t * t * (3.0 - 2.0 * t)))
    }

    /// Everything the chosen effect needs and, with shuffle on, everything any effect it
    /// might pick needs. That keeps the sound filter and the depth model running between
    /// effects, so sound doesn't drop out and the picture's hold doesn't change at each one.
    pub fn needs(&self) -> Needs {
        let fading = self.fading.as_ref().map(|f| f.from.as_str());
        let counted = self.effects.iter().filter(|e| {
            let id = Some(e.id.as_str());
            self.on && (id == self.chosen.as_deref() || id == fading || (self.shuffle.on && self.in_shuffle(&e.id)))
        });
        let mut needs = Needs { audio: false, depth: false, segment: false, faces: false, history: false, flow: false, plate: false, cuts: false, stats: false, scale: 0.25 };
        for e in counted {
            let m = &e.manifest;
            needs.audio |= m.audio;
            needs.depth |= m.depth || m.segment || m.plate;
            needs.segment |= m.segment || m.plate;
            needs.faces |= m.faces;
            needs.history |= m.history;
            needs.flow |= m.flow;
            needs.plate |= m.plate;
            needs.cuts |= m.cuts;
            needs.stats |= m.stats;
            needs.scale = needs.scale.max(usable_scale(m.scale));
        }
        needs
    }

    fn roll(&mut self) -> f32 {
        if self.random == 0 {
            let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64);
            self.random = now | 1;
        }
        // xorshift: the order only has to look random.
        self.random ^= self.random << 13;
        self.random ^= self.random >> 7;
        self.random ^= self.random << 17;
        (self.random >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Starts counting down to the next change.
    fn wait(&mut self) {
        let (a, b) = (self.shuffle.shortest, self.shuffle.longest);
        let (shortest, longest) = (a.min(b).max(0.0), a.max(b));
        self.until_next = if self.shuffle.at_cuts { longest } else { shortest + self.roll() * (longest - shortest) };
        self.turn = 0.0;
    }

    /// The next effect, going through them all in a random order before any comes round
    /// again. `None` if there's no other to go to.
    fn next_turn(&mut self) -> Option<String> {
        let pool: Vec<String> = self.effects.iter().filter(|e| self.in_shuffle(&e.id)).map(|e| e.id.clone()).collect();
        self.turns.retain(|id| pool.contains(id));
        for _ in 0..2 {
            while let Some(id) = self.turns.pop() {
                if Some(&id) != self.chosen.as_ref() {
                    return Some(id);
                }
            }
            self.turns = pool.clone();
            for i in (1..self.turns.len()).rev() {
                let j = (self.roll() * (i + 1) as f32) as usize;
                self.turns.swap(i, j.min(i));
            }
        }
        None
    }

    /// Off → each effect in turn → off.
    pub fn cycle(&mut self, db: &Db) {
        let at = self.chosen().and_then(|c| self.effects.iter().position(|e| e.id == c.id));
        let next = match at {
            None => self.effects.first(),
            Some(i) => self.effects.get(i + 1),
        };
        let id = next.map(|e| e.id.clone());
        self.choose(id.as_deref(), db);
    }

    pub fn value(&self, id: &str, key: &str) -> Option<&serde_json::Value> {
        self.rolled.get(id).and_then(|s| s.get(key)).or_else(|| self.settings.get(id)?.get(key))
    }

    pub fn set_setting(&mut self, id: &str, key: &str, value: serde_json::Value, db: &Db) {
        self.set_value(id, key, value);
        self.save(id, db);
    }

    /// Changes a setting without saving it yet (e.g. while a slider is dragged).
    pub fn set_value(&mut self, id: &str, key: &str, value: serde_json::Value) {
        if let Some(rolled) = self.rolled.get_mut(id) {
            rolled.insert(key.to_owned(), value.clone());
        }
        self.settings.entry(id.to_owned()).or_default().insert(key.to_owned(), value);
    }

    /// Saves an effect's settings.
    pub fn save(&self, id: &str, db: &Db) {
        let Some(settings) = self.settings.get(id) else { return };
        let json = serde_json::Value::Object(settings.clone()).to_string();
        if let Err(e) = db.set_setting(&settings_key(id), &json) {
            log::warn!("saving video effect settings: {e:#}");
        }
    }

    /// The chosen effect, ready to draw.
    pub fn active(&self) -> Option<ActiveEffect> {
        self.chosen().map(|e| self.to_draw(e))
    }

    fn to_draw(&self, e: &Effect) -> ActiveEffect {
        let uniforms = e
            .settings()
            .filter(|s| !s.key.is_empty() && s.key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'))
            .map(|s| {
                let value = self.value(&e.id, &s.key);
                let v = match s.kind {
                    SettingKind::Bool => f32::from(u8::from(value.and_then(serde_json::Value::as_bool).unwrap_or(false))),
                    SettingKind::Choice => {
                        let id = value.and_then(serde_json::Value::as_str).unwrap_or("");
                        s.options.iter().position(|o| o.id == id).unwrap_or(0) as f32
                    }
                    _ => self.drifted(&e.id, s, value.and_then(serde_json::Value::as_f64).unwrap_or(s.min.unwrap_or(0.0))) as f32,
                };
                (format!("s_{}", s.key), v)
            })
            .collect();
        ActiveEffect {
            id: e.id.clone(),
            name: e.manifest.name.clone(),
            shader: e.shader.clone(),
            feedback: e.manifest.feedback,
            audio: e.manifest.audio,
            depth: e.manifest.depth,
            segment: e.manifest.segment,
            faces: e.manifest.faces,
            scale: usable_scale(e.manifest.scale),
            uniforms,
        }
    }
}

fn usable_scale(scale: f32) -> f32 {
    if scale.is_finite() { scale.clamp(0.25, 1.0) } else { 1.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, text: &str) {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), text).unwrap();
    }

    #[test]
    fn finds_effects_and_keeps_their_settings() {
        let root = std::env::temp_dir().join(format!("flex-effects-{}", std::process::id()));
        let (shipped, mine) = (root.join("shipped"), root.join("mine"));
        write(&shipped, "melt.frag", "");
        write(
            &shipped,
            "melt.toml",
            "name = \"Melt\"\nfeedback = true\nscale = 0.1\naudio = true\n\
             [[settings]]\nkey = \"speed\"\nlabel = \"Speed\"\ntype = \"number\"\ndefault = 1.5\n\
             [[settings]]\nkey = \"mirror\"\nlabel = \"Mirror\"\ntype = \"bool\"\ndefault = true\n\
             [[settings]]\nkey = \"mode\"\nlabel = \"Mode\"\ntype = \"choice\"\ndefault = \"b\"\n\
             options = [{ id = \"a\", label = \"A\" }, { id = \"b\", label = \"B\" }]\n\
             [[settings]]\nkey = \"bad key\"\nlabel = \"Bad\"\ntype = \"number\"\n\
             [[settings]]\nkey = \"note\"\nlabel = \"Note\"\ntype = \"text\"\n",
        );
        write(&shipped, "dream.frag", "");
        write(&mine, "dream.frag", "");
        write(&mine, "dream.toml", "this isn't toml [");
        write(&mine, "my-test_one.frag", "");
        write(&mine, "readme.txt", "");

        let found = discover(&[shipped.clone(), mine.clone(), root.join("missing")]);
        let names: Vec<&str> = found.iter().map(|e| e.manifest.name.as_str()).collect();
        assert_eq!(names, ["Dream", "Melt", "My test one"]);
        assert_eq!(found[0].shader, mine.join("dream.frag"), "the data folder wins");
        assert!(found[0].problem.is_some());

        let db = Db::open_in_memory().unwrap();
        let dirs = [shipped, mine];
        let mut effects = Effects::new(&dirs, &db);
        assert_eq!(effects.active(), None, "off to start with");
        effects.choose(Some("melt"), &db);
        let active = effects.active().unwrap();
        assert_eq!((active.feedback, active.audio, active.scale), (true, true, 0.25));
        let uniforms = [("s_speed".to_owned(), 1.5), ("s_mirror".to_owned(), 1.0), ("s_mode".to_owned(), 1.0)];
        assert_eq!(active.uniforms, uniforms);

        effects.set_setting("melt", "speed", serde_json::json!(0.5), &db);
        // Choices and settings survive a restart.
        let mut effects = Effects::new(&dirs, &db);
        assert_eq!(effects.active().unwrap().uniforms[0], ("s_speed".to_owned(), 0.5));
        effects.cycle(&db);
        assert_eq!(effects.chosen().map(|e| e.id.as_str()), Some("my-test_one"));
        effects.cycle(&db);
        assert!(effects.chosen().is_none(), "after the last comes off");
        effects.cycle(&db);
        assert_eq!(effects.chosen().map(|e| e.id.as_str()), Some("dream"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn shuffle_takes_turns_and_fades() {
        let root = std::env::temp_dir().join(format!("flex-effects-shuffle-{}", std::process::id()));
        write(&root, "a.frag", "");
        write(&root, "a.toml", "audio = true\nscale = 0.5\n");
        write(&root, "b.frag", "");
        write(&root, "b.toml", "depth = true\n");
        write(&root, "c.frag", "");
        write(&root, "test.frag", "");
        write(&root, "test.toml", "faces = true\nshuffle = false\n");
        let db = Db::open_in_memory().unwrap();
        let dirs = [root.clone()];
        let mut effects = Effects::new(&dirs, &db);
        assert!(!effects.needs().audio, "nothing's on");

        effects.set_shuffle(Shuffle { on: true, shortest: 10.0, longest: 10.0, fade: 2.0, at_cuts: false, randomise: 0.0 }, &db);
        let first = effects.chosen().expect("shuffle starts one").id.clone();
        let needs = effects.needs();
        assert_eq!((needs.audio, needs.depth, needs.faces, needs.scale), (true, true, false, 1.0), "all that it might pick");

        // Stays put while paused, and until its time's up.
        effects.step(100.0, false);
        effects.step(9.0, true);
        assert_eq!(effects.chosen().unwrap().id, first);
        assert!(effects.fading().is_none());

        effects.step(2.0, true);
        let second = effects.chosen().unwrap().id.clone();
        assert_ne!(second, first);
        let (from, amount) = effects.fading().unwrap();
        assert_eq!((from.id.as_str(), amount), (first.as_str(), 0.0));
        effects.step(1.0, false);
        assert_eq!(effects.fading().unwrap().1, 0.5, "the fade carries on while paused");
        effects.step(1.0, true);
        assert!(effects.fading().is_none());

        // Each once before any comes round again, and never the test view.
        let mut seen = vec![first, second];
        effects.step(10.5, true);
        effects.step(2.0, true);
        seen.push(effects.chosen().unwrap().id.clone());
        seen.sort();
        assert_eq!(seen, ["a", "b", "c"]);

        // One that won't compile is left at once, without a fade.
        effects.status = Some((false, "Shader error".into()));
        let broken = effects.chosen().unwrap().id.clone();
        effects.step(0.1, true);
        assert_ne!(effects.chosen().unwrap().id, broken);
        assert!(effects.fading().is_none());

        // At cuts: not before the shortest turn, and only when asked to.
        effects.set_shuffle(Shuffle { on: true, shortest: 10.0, longest: 30.0, fade: 2.0, at_cuts: true, randomise: 0.0 }, &db);
        let before = effects.chosen().unwrap().id.clone();
        effects.step(9.0, true);
        effects.scene_changed();
        assert_eq!(effects.chosen().unwrap().id, before, "too soon");
        effects.step(2.0, true);
        effects.scene_changed();
        assert_ne!(effects.chosen().unwrap().id, before);
        assert_eq!(effects.fading().unwrap().0.id, before, "fades there too");
        let at_cut = effects.chosen().unwrap().id.clone();
        effects.step(29.0, true);
        assert_eq!(effects.chosen().unwrap().id, at_cut, "only the longest turn moves it without a cut");
        effects.step(2.0, true);
        assert_ne!(effects.chosen().unwrap().id, at_cut);
        // A fade of 0 switches at once, whichever way it's changing.
        effects.set_shuffle(Shuffle { on: true, shortest: 1.0, longest: 30.0, fade: 0.0, at_cuts: true, randomise: 0.0 }, &db);
        effects.step(2.0, true);
        let sharp = effects.chosen().unwrap().id.clone();
        effects.scene_changed();
        assert_ne!(effects.chosen().unwrap().id, sharp);
        assert!(effects.fading().is_none());
        effects.set_shuffle(Shuffle { on: true, shortest: 5.0, longest: 5.0, fade: 0.0, at_cuts: false, randomise: 0.0 }, &db);
        let sharp = effects.chosen().unwrap().id.clone();
        effects.step(6.0, true);
        assert_ne!(effects.chosen().unwrap().id, sharp);
        assert!(effects.fading().is_none());
        effects.set_shuffle(Shuffle { on: true, shortest: 10.0, longest: 10.0, fade: 2.0, at_cuts: false, randomise: 0.0 }, &db);
        effects.step(2.0, false);
        effects.step(11.0, true);
        let moved = effects.chosen().unwrap().id.clone();
        effects.step(2.5, true);
        effects.scene_changed();
        assert_eq!(effects.chosen().unwrap().id, moved, "cuts are ignored unless asked for");

        // Choosing one stops any fade; both are kept.
        effects.set_in_shuffle("c", false, &db);
        effects.choose(Some("a"), &db);
        assert!(Effects::new(&dirs, &db).shuffle().on);
        assert!(!Effects::new(&dirs, &db).in_shuffle("c"));

        // Switched off, nothing runs or changes, and it all comes back where it was.
        effects.set_on(false, &db);
        assert!(effects.chosen().is_none() && effects.active().is_none());
        assert!(!effects.needs().depth && !effects.wants_cuts());
        for _ in 0..200 {
            effects.step(1.0, true);
        }
        assert_eq!(effects.remembered().unwrap().id, "a");
        let reopened = Effects::new(&dirs, &db);
        assert!(!reopened.is_on() && reopened.chosen().is_none() && reopened.shuffle().on);
        effects.set_on(true, &db);
        assert_eq!(effects.chosen().unwrap().id, "a");
        assert!(effects.shuffle().on);

        // Choosing none is switching off, and choosing one switches back on.
        effects.choose(None, &db);
        assert!(!effects.is_on() && effects.shuffle().on);
        effects.choose(Some("b"), &db);
        assert!(effects.is_on() && Effects::new(&dirs, &db).is_on());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn favourites_are_kept_and_nothing_chosen_starts_off() {
        let root = std::env::temp_dir().join(format!("flex-effects-fav-{}", std::process::id()));
        write(&root, "one.frag", "");
        let dirs = [root.clone()];
        let db = Db::open_in_memory().unwrap();
        let mut effects = Effects::new(&dirs, &db);
        assert!(!effects.is_on());
        assert!(!effects.is_favourite("one"));
        effects.set_favourite("one", true, &db);
        assert!(Effects::new(&dirs, &db).is_favourite("one"));
        // Nothing to go back to, so switching on finds one.
        effects.set_on(true, &db);
        assert_eq!(effects.chosen().unwrap().id, "one");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn shuffle_can_roll_settings_without_touching_the_saved_ones() {
        let root = std::env::temp_dir().join(format!("flex-effects-roll-{}", std::process::id()));
        let toml = "[[settings]]\nkey = \"size\"\nlabel = \"Size\"\ntype = \"number\"\nmin = 0\nmax = 10\nstep = 2\ndefault = 4\n\
                    [[settings]]\nkey = \"mode\"\nlabel = \"Mode\"\ntype = \"choice\"\ndefault = \"a\"\n\
                    options = [{ id = \"a\", label = \"A\" }, { id = \"b\", label = \"B\" }]\n";
        for name in ["one", "two"] {
            write(&root, &format!("{name}.frag"), "");
            write(&root, &format!("{name}.toml"), toml);
        }
        let db = Db::open_in_memory().unwrap();
        let dirs = [root.clone()];
        let mut effects = Effects::new(&dirs, &db);
        effects.choose(Some("one"), &db);
        let on = Shuffle { on: true, shortest: 1.0, longest: 1.0, fade: 0.0, at_cuts: false, randomise: 100.0 };
        effects.set_shuffle(on, &db);

        let mut seen = std::collections::HashSet::new();
        for _ in 0..200 {
            effects.step(2.0, true);
            let id = effects.chosen().unwrap().id.clone();
            let size = effects.value(&id, "size").unwrap().as_i64().expect("on the grid");
            assert!((0..=10).contains(&size) && size % 2 == 0, "{size}");
            seen.insert((size, effects.value(&id, "mode").unwrap().as_str().unwrap().to_owned()));
        }
        assert!(seen.len() >= 8, "{seen:?}");

        // A change made now shows now, and is what's saved; the rolled values never are.
        let id = effects.chosen().unwrap().id.clone();
        effects.set_setting(&id, "size", serde_json::json!(6), &db);
        assert_eq!(effects.value(&id, "size").unwrap(), 6);
        let other = if id == "one" { "two" } else { "one" };
        let restarted = Effects::new(&dirs, &db);
        assert_eq!(restarted.value(&id, "size").unwrap(), 6);
        assert_eq!(restarted.value(other, "size").unwrap(), 4);

        // Half way: never further than half the way to the ends of the range.
        effects.set_shuffle(Shuffle { randomise: 50.0, ..on }, &db);
        for _ in 0..100 {
            effects.step(2.0, true);
            let id = effects.chosen().unwrap().id.clone();
            let base = if id == "one" || other == "one" { Effects::new(&dirs, &db).value(&id, "size").unwrap().as_i64().unwrap() } else { 4 };
            let size = effects.value(&id, "size").unwrap().as_i64().unwrap();
            assert!((size - base).abs() <= 4, "{size} from {base}");
        }

        // Turned off, it's back to the saved ones.
        effects.set_shuffle(Shuffle { randomise: 0.0, ..on }, &db);
        assert_eq!(effects.value(&id, "size").unwrap(), 6);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn settings_drift_slowly_and_stay_in_range() {
        let root = std::env::temp_dir().join(format!("flex-effects-drift-{}", std::process::id()));
        write(&root, "a.frag", "");
        write(
            &root,
            "a.toml",
            "[[settings]]\nkey = \"wide\"\nlabel = \"W\"\ntype = \"number\"\nmin = 0\nmax = 100\nstep = 5\nunit = \"%\"\ndefault = 50\n\
             [[settings]]\nkey = \"count\"\nlabel = \"C\"\ntype = \"number\"\nmin = 2\nmax = 12\nstep = 1\ndefault = 6\n\
             [[settings]]\nkey = \"rate\"\nlabel = \"R\"\ntype = \"number\"\nmin = 0\nmax = 10\ndrift = \"off\"\ndefault = 3\n",
        );
        let db = Db::open_in_memory().unwrap();
        let dirs = [root.clone()];
        let mut effects = Effects::new(&dirs, &db);
        effects.choose(Some("a"), &db);
        let uniform = |e: &Effects, i: usize| e.active().unwrap().uniforms[i].1;
        assert_eq!(uniform(&effects, 0), 50.0, "off to start with");

        effects.set_drift(Drift { amount: 100.0, speed: 100.0 }, &db);
        effects.step(0.016, true);
        assert!((uniform(&effects, 0) - 50.0).abs() < 2.0, "eases in rather than jumping");

        let (mut low, mut high, mut last) = (f32::MAX, f32::MIN, None);
        for _ in 0..30000 {
            effects.step(0.033, true);
            let wide = uniform(&effects, 0);
            assert!((0.0..=100.0).contains(&wide));
            if let Some(last) = last {
                assert!((wide - last as f32).abs() < 0.5, "slowly");
            }
            last = Some(wide);
            (low, high) = (low.min(wide), high.max(wide));
            let count = uniform(&effects, 1);
            assert!((2.0..=12.0).contains(&count) && count.fract() == 0.0, "{count}");
            assert_eq!(uniform(&effects, 2), 3.0, "rates stay put");
        }
        assert!(high - low > 40.0, "{low} to {high}");

        // Paused, nothing moves.
        let held = uniform(&effects, 0);
        let eased = (0..200).fold((), |_, _| effects.step(0.033, false));
        let _ = eased;
        assert!((uniform(&effects, 0) - held).abs() < 0.01);
        assert_eq!(Effects::new(&dirs, &db).drift().amount, 100.0, "kept");
        let _ = std::fs::remove_dir_all(&root);
    }
}
