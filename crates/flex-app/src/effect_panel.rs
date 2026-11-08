//! The effects panel: hold OK (or right-click) while a video plays to switch effect, change
//! its settings and set up shuffle, watching the result beside it.
//!
//! Up/Down: pick a row · Left/Right: change it · OK: next value · Back: close. With a mouse,
//! click a row, or click or drag along one with a bar.

use eframe::egui::{self, Align2, Pos2, Rect, TextStyle, pos2, vec2};
use flex_core::effects::{Drift, Shuffle};
use flex_core::plugins::{SettingKind, SettingSpec};
use flex_plugin::serde_json::{Value, json};

use crate::input::{Action, Press};
use crate::shared::Shared;
use crate::theme;
use crate::widgets;

const WIDTH: f32 = 660.0;
const ROW_HEIGHT: f32 = 60.0;
const ROW_GAP: f32 = 4.0;
const PADDING: f32 = 16.0;
const HEADER: f32 = 56.0;
const FOOTER: f32 = 100.0;
/// (lowest, highest, step), in seconds.
const TURN: (f32, f32, f32) = (5.0, 600.0, 5.0);
const FADE: (f32, f32, f32) = (0.0, 15.0, 0.5);
const RANDOMISE: (f32, f32, f32) = (0.0, 100.0, 10.0);
const DRIFT: (f32, f32, f32) = (0.0, 100.0, 10.0);
const DRIFT_SPEED: (f32, f32, f32) = (10.0, 400.0, 10.0);

#[derive(Clone)]
enum Line {
    Effect,
    Drift,
    DriftSpeed,
    Shuffle,
    ChangeAt,
    Shortest,
    Longest,
    Fade,
    Randomise,
    Setting(Box<SettingSpec>),
    InShuffle,
    Defaults,
}

/// A value somewhere along a bar.
struct Bar {
    min: f32,
    max: f32,
    value: f32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Stay,
    Close,
    /// For the player to deal with, like pausing.
    NotForMe,
}

#[derive(Default)]
pub struct EffectPanel {
    selected: usize,
    first: usize,
    rect: Option<Rect>,
    /// The effect with changes not saved yet, from holding a key down.
    unsaved: Option<String>,
}

fn lines(sh: &Shared) -> Vec<Line> {
    let mut lines = vec![Line::Effect, Line::Drift];
    if sh.effects.drift().amount > 0.0 {
        lines.push(Line::DriftSpeed);
    }
    lines.push(Line::Shuffle);
    if sh.effects.shuffle().on {
        lines.extend([Line::ChangeAt, Line::Shortest, Line::Longest, Line::Fade, Line::Randomise]);
    }
    if let Some(effect) = sh.effects.chosen() {
        lines.extend(effect.settings().cloned().map(Box::new).map(Line::Setting));
        if sh.effects.shuffle().on {
            lines.push(Line::InShuffle);
        }
        if effect.settings().next().is_some() {
            lines.push(Line::Defaults);
        }
    }
    lines
}

fn seconds(value: f32) -> String {
    if value >= 90.0 {
        format!("{}:{:02}", (value / 60.0) as u32, value as u32 % 60)
    } else {
        format!("{value} s")
    }
}

/// `value` moved `steps` along, on a whole number of steps from `min` and within range.
fn along(value: f64, steps: f64, min: f64, max: f64, step: f64) -> f64 {
    let at = ((value - min) / step).round() + steps;
    // Rounded so 0.1 steps don't show as 0.30000000000000004.
    (((min + at * step) * 1e4).round() / 1e4).clamp(min, max)
}

fn number(value: f64) -> Value {
    if value.fract() == 0.0 { json!(value as i64) } else { json!(value) }
}

fn step_of(spec: &SettingSpec, min: f64, max: f64) -> f64 {
    spec.step.filter(|s| *s > 0.0).unwrap_or((max - min) / 20.0).max(1e-4)
}

/// The setting moved one place, wrapping round for lists.
fn changed(spec: &SettingSpec, current: Option<&Value>, direction: f64) -> Option<Value> {
    match spec.kind {
        SettingKind::Bool => Some(json!(!current.and_then(Value::as_bool).unwrap_or(false))),
        SettingKind::Choice => {
            let count = spec.options.len() as i64;
            if count == 0 {
                return None;
            }
            let at = current.and_then(Value::as_str).and_then(|c| spec.options.iter().position(|o| o.id == c)).unwrap_or(0) as i64;
            Some(json!(spec.options[(at + direction as i64).rem_euclid(count) as usize].id))
        }
        SettingKind::Number => {
            let (min, max) = (spec.min.unwrap_or(0.0), spec.max.unwrap_or(100.0));
            Some(number(along(current.and_then(Value::as_f64).unwrap_or(min), direction, min, max, step_of(spec, min, max))))
        }
        SettingKind::Text | SettingKind::Button => None,
    }
}

impl EffectPanel {
    pub fn contains(&self, pos: Pos2) -> bool {
        self.rect.is_some_and(|r| r.contains(pos))
    }

    /// Saves anything left from holding a key down.
    pub fn save(&mut self, sh: &mut Shared) {
        if let Some(id) = self.unsaved.take() {
            let (effects, db) = sh.effects_mut();
            effects.save(&id, db);
        }
    }

    pub fn handle(&mut self, sh: &mut Shared, press: Press) -> Outcome {
        let lines = lines(sh);
        let last = lines.len().saturating_sub(1);
        self.selected = self.selected.min(last);
        if !press.repeat {
            self.save(sh);
        }
        match press.action {
            Action::Back | Action::Menu if !press.repeat => return Outcome::Close,
            Action::Up => self.selected = self.selected.saturating_sub(1),
            Action::Down => self.selected = (self.selected + 1).min(last),
            Action::Left | Action::Right => {
                // Holding a key down shouldn't race through whole shaders.
                if !press.repeat || !matches!(lines[self.selected], Line::Effect | Line::Shuffle | Line::ChangeAt | Line::InShuffle | Line::Defaults) {
                    self.change(sh, &lines[self.selected], if press.action == Action::Left { -1.0 } else { 1.0 }, press.repeat);
                }
            }
            Action::Ok if !press.repeat => self.activate(sh, &lines[self.selected]),
            Action::Ok => {}
            _ => return Outcome::NotForMe,
        }
        Outcome::Stay
    }

    fn activate(&mut self, sh: &mut Shared, line: &Line) {
        match line {
            Line::Defaults => self.defaults(sh),
            // There's no "next" for a value along a bar.
            Line::Drift | Line::DriftSpeed | Line::Shortest | Line::Longest | Line::Fade | Line::Randomise => {}
            Line::Setting(spec) if spec.kind == SettingKind::Number => {}
            _ => self.change(sh, line, 1.0, false),
        }
    }

    fn defaults(&mut self, sh: &mut Shared) {
        let Some(effect) = sh.effects.chosen().cloned() else { return };
        let (effects, db) = sh.effects_mut();
        for spec in effect.settings() {
            if let Some(default) = &spec.default {
                effects.set_value(&effect.id, &spec.key, default.clone());
            }
        }
        effects.save(&effect.id, db);
    }

    fn change(&mut self, sh: &mut Shared, line: &Line, direction: f32, held: bool) {
        let (shuffle, drift) = (sh.effects.shuffle(), sh.effects.drift());
        let timed = |value: f32, (min, max, step): (f32, f32, f32)| along(value.into(), direction.into(), min.into(), max.into(), step.into()) as f32;
        match line {
            Line::Effect => {
                // Off, then each effect in turn, either way round.
                let mut ids: Vec<Option<String>> = vec![None];
                ids.extend(sh.effects.list().iter().map(|e| Some(e.id.clone())));
                let chosen = sh.effects.chosen().map(|e| e.id.clone());
                let at = ids.iter().position(|id| *id == chosen).unwrap_or(0) as i64;
                let next = ids[(at + direction as i64).rem_euclid(ids.len() as i64) as usize].clone();
                let (effects, db) = sh.effects_mut();
                effects.choose(next.as_deref(), db);
            }
            Line::Drift => set_drift(sh, Drift { amount: timed(drift.amount, DRIFT), ..drift }),
            Line::DriftSpeed => set_drift(sh, Drift { speed: timed(drift.speed, DRIFT_SPEED), ..drift }),
            Line::Shuffle => set_shuffle(sh, Shuffle { on: !shuffle.on, ..shuffle }),
            Line::ChangeAt => set_shuffle(sh, Shuffle { at_cuts: !shuffle.at_cuts, ..shuffle }),
            Line::Shortest => {
                let shortest = timed(shuffle.shortest, TURN);
                set_shuffle(sh, Shuffle { shortest, longest: shuffle.longest.max(shortest), ..shuffle });
            }
            Line::Longest => {
                let longest = timed(shuffle.longest, TURN);
                set_shuffle(sh, Shuffle { longest, shortest: shuffle.shortest.min(longest), ..shuffle });
            }
            Line::Fade => set_shuffle(sh, Shuffle { fade: timed(shuffle.fade, FADE), ..shuffle }),
            Line::Randomise => set_shuffle(sh, Shuffle { randomise: timed(shuffle.randomise, RANDOMISE), ..shuffle }),
            Line::Setting(spec) => {
                let Some(id) = sh.effects.chosen().map(|e| e.id.clone()) else { return };
                let Some(value) = changed(spec, sh.effects.value(&id, &spec.key), direction.into()) else { return };
                self.set(sh, &id, spec, value, held);
            }
            Line::InShuffle => {
                let Some(id) = sh.effects.chosen().map(|e| e.id.clone()) else { return };
                let included = sh.effects.in_shuffle(&id);
                let (effects, db) = sh.effects_mut();
                effects.set_in_shuffle(&id, !included, db);
            }
            Line::Defaults => {}
        }
    }

    fn set(&mut self, sh: &mut Shared, id: &str, spec: &SettingSpec, value: Value, held: bool) {
        let (effects, db) = sh.effects_mut();
        if held {
            effects.set_value(id, &spec.key, value);
            self.unsaved = Some(id.to_owned());
        } else {
            effects.set_setting(id, &spec.key, value, db);
        }
    }

    /// The mouse put a bar at `fraction` of the way along.
    fn set_along(&mut self, sh: &mut Shared, line: &Line, fraction: f32, released: bool) {
        let (shuffle, drift) = (sh.effects.shuffle(), sh.effects.drift());
        let timed = |(min, max, step): (f32, f32, f32)| along((min + fraction * (max - min)).into(), 0.0, min.into(), max.into(), step.into()) as f32;
        match line {
            Line::Drift => set_drift(sh, Drift { amount: timed(DRIFT), ..drift }),
            Line::DriftSpeed => set_drift(sh, Drift { speed: timed(DRIFT_SPEED), ..drift }),
            Line::Shortest => {
                let shortest = timed(TURN);
                set_shuffle(sh, Shuffle { shortest, longest: shuffle.longest.max(shortest), ..shuffle });
            }
            Line::Longest => {
                let longest = timed(TURN);
                set_shuffle(sh, Shuffle { longest, shortest: shuffle.shortest.min(longest), ..shuffle });
            }
            Line::Fade => set_shuffle(sh, Shuffle { fade: timed(FADE), ..shuffle }),
            Line::Randomise => set_shuffle(sh, Shuffle { randomise: timed(RANDOMISE), ..shuffle }),
            Line::Setting(spec) => {
                let Some(id) = sh.effects.chosen().map(|e| e.id.clone()) else { return };
                let (min, max) = (spec.min.unwrap_or(0.0), spec.max.unwrap_or(100.0));
                let value = number(along(min + f64::from(fraction) * (max - min), 0.0, min, max, step_of(spec, min, max)));
                self.set(sh, &id, spec, value, !released);
            }
            _ => {}
        }
    }

    /// What a row says: its name, its value, and a bar if it has one.
    fn describe(sh: &Shared, line: &Line) -> (String, String, Option<Bar>) {
        let (shuffle, drift) = (sh.effects.shuffle(), sh.effects.drift());
        let timed = |value: f32, (min, max, _): (f32, f32, f32)| Some(Bar { min, max, value });
        let chosen = sh.effects.chosen();
        let on_off = |on: bool| (if on { "on" } else { "off" }).to_owned();
        match line {
            Line::Effect => ("Effect".into(), chosen.map_or("Off".into(), |e| e.manifest.name.clone()), None),
            Line::Drift => ("Drift".into(), if drift.amount > 0.0 { format!("{}%", drift.amount) } else { "Off".into() }, timed(drift.amount, DRIFT)),
            Line::DriftSpeed => ("Drift speed".into(), format!("{}%", drift.speed), timed(drift.speed, DRIFT_SPEED)),
            Line::Shuffle => ("Shuffle".into(), on_off(shuffle.on), None),
            Line::ChangeAt => ("Change at".into(), (if shuffle.at_cuts { "Scene cuts" } else { "Any time" }).into(), None),
            Line::Shortest => ("Shortest turn".into(), seconds(shuffle.shortest), timed(shuffle.shortest, TURN)),
            Line::Longest => ("Longest turn".into(), seconds(shuffle.longest), timed(shuffle.longest, TURN)),
            Line::Fade => ("Fade".into(), if shuffle.fade > 0.0 { seconds(shuffle.fade) } else { "Off".into() }, timed(shuffle.fade, FADE)),
            Line::Randomise => ("Random settings".into(), if shuffle.randomise > 0.0 { format!("{}%", shuffle.randomise) } else { "Off".into() }, timed(shuffle.randomise, RANDOMISE)),
            Line::InShuffle => ("In the shuffle".into(), (if chosen.is_some_and(|e| sh.effects.in_shuffle(&e.id)) { "yes" } else { "no" }).into(), None),
            Line::Defaults => ("Back to the defaults".into(), String::new(), None),
            Line::Setting(spec) => {
                let value = chosen.and_then(|e| sh.effects.value(&e.id, &spec.key));
                match spec.kind {
                    SettingKind::Bool => (spec.label.clone(), on_off(value.and_then(Value::as_bool).unwrap_or(false)), None),
                    SettingKind::Choice => {
                        let id = value.and_then(Value::as_str).unwrap_or("");
                        (spec.label.clone(), spec.options.iter().find(|o| o.id == id).map_or(id.to_owned(), |o| o.label.clone()), None)
                    }
                    _ => {
                        let (min, max) = (spec.min.unwrap_or(0.0), spec.max.unwrap_or(100.0));
                        let n = value.and_then(Value::as_f64).unwrap_or(min);
                        // The bar shows where it's drifted to, the number what it's set to.
                        let live = sh.effects.live_number(spec).unwrap_or(n);
                        (spec.label.clone(), format!("{n}{}", spec.unit), Some(Bar { min: min as f32, max: max as f32, value: live as f32 }))
                    }
                }
            }
        }
    }

    /// The help for the selected row.
    fn help(sh: &Shared, line: &Line) -> String {
        match line {
            Line::Effect => {
                let chosen = sh.effects.chosen();
                match (&sh.effects.status, chosen) {
                    (Some((false, error)), _) => format!("⚠ {error}"),
                    (_, Some(e)) => e.manifest.description.clone(),
                    (_, None) => "No effect".into(),
                }
            }
            Line::Drift => "Every effect's settings wander slowly on their own: this is how far from where they're set.".into(),
            Line::DriftSpeed => "How quickly they wander.".into(),
            Line::Shuffle => "Effects take turns at random, fading from one to the next.".into(),
            Line::ChangeAt if sh.effects.shuffle().at_cuts => "The next effect starts on the frame the shot changes.".into(),
            Line::ChangeAt => "The next effect starts at a random moment.".into(),
            Line::Shortest | Line::Longest if sh.effects.shuffle().at_cuts => "Cuts are ignored until the shortest turn is up. After the longest it changes anyway.".into(),
            Line::Shortest | Line::Longest => "Each effect stays for a random time between these.".into(),
            Line::Fade => "How long one effect takes to fade into the next. Off switches instantly.".into(),
            Line::Randomise => "Each time an effect comes up, its settings move this far at random from yours, which aren't changed.".into(),
            Line::InShuffle => "Whether shuffle picks this effect.".into(),
            Line::Defaults => String::new(),
            Line::Setting(spec) => spec.detail.clone(),
        }
    }

    /// `safe` is the part of the screen it's safe to draw on.
    pub fn ui(&mut self, ui: &egui::Ui, sh: &mut Shared, safe: Rect) {
        let lines = lines(sh);
        self.selected = self.selected.min(lines.len().saturating_sub(1));
        let panel = Rect::from_min_max(pos2(safe.right() - WIDTH, safe.top()), safe.right_bottom());
        self.rect = Some(panel);
        let painter = ui.painter();
        theme::paint_overlay(painter, panel, 14.0);
        let inner = panel.shrink(PADDING);
        let (body, small) = (TextStyle::Body.resolve(ui.style()), TextStyle::Small.resolve(ui.style()));
        let text = theme::overlay_text();
        painter.text(inner.left_top() + vec2(8.0, 4.0), Align2::LEFT_TOP, "Video effects", body.clone(), text);
        painter.text(inner.right_top() + vec2(-8.0, 10.0), Align2::RIGHT_TOP, "Back to close", small.clone(), text.gamma_multiply(0.6));

        let visible = (((inner.height() - HEADER - FOOTER + ROW_GAP) / (ROW_HEIGHT + ROW_GAP)).floor() as usize).max(1);
        let moved = ui.input(|i| i.pointer.delta() != egui::Vec2::ZERO);
        let wheel = ui.input(|i| i.events.iter().map(|e| if let egui::Event::MouseWheel { delta, .. } = e { delta.y } else { 0.0 }).sum::<f32>());
        if wheel != 0.0 && ui.input(|i| i.pointer.hover_pos()).is_some_and(|p| panel.contains(p)) {
            self.selected = if wheel > 0.0 { self.selected.saturating_sub(1) } else { (self.selected + 1).min(lines.len().saturating_sub(1)) };
        }
        self.first = self.first.min(self.selected).max((self.selected + 1).saturating_sub(visible)).min(lines.len().saturating_sub(visible));

        let mut clicked = None;
        let mut dragged = None;
        for (n, line) in lines.iter().enumerate().skip(self.first).take(visible) {
            let top = inner.top() + HEADER + (n - self.first) as f32 * (ROW_HEIGHT + ROW_GAP);
            let row = Rect::from_min_size(pos2(inner.left(), top), vec2(inner.width(), ROW_HEIGHT));
            let (name, value, bar) = Self::describe(sh, line);

            let response = ui.interact(row, egui::Id::new(("flex-effect-panel", n)), egui::Sense::click_and_drag());
            if response.hovered() && moved {
                self.selected = n;
            }
            if let Some(p) = response.interact_pointer_pos() {
                self.selected = n;
                if bar.is_some() {
                    if response.is_pointer_button_down_on() || response.drag_stopped() {
                        dragged = Some((line.clone(), ((p.x - row.left()) / row.width()).clamp(0.0, 1.0), response.drag_stopped()));
                    }
                } else if response.clicked() {
                    clicked = Some(line.clone());
                }
            }

            let selected = n == self.selected;
            let colour = widgets::paint_focusable(painter, row, selected, selected);
            let value_galley = painter.layout_no_wrap(value, body.clone(), colour);
            let value_x = row.right() - 20.0 - value_galley.size().x;
            painter.galley(pos2(value_x, row.center().y - value_galley.size().y / 2.0), value_galley, colour);
            let label = Rect::from_min_max(row.min, pos2(value_x - 16.0, row.bottom()));
            painter.with_clip_rect(label).text(row.left_center() + vec2(20.0, 0.0), Align2::LEFT_CENTER, name, body.clone(), colour);
            if let Some(bar) = bar {
                let span = (bar.max - bar.min).max(1e-4);
                widgets::paint_progress(painter, row, (bar.value - bar.min) / span, theme::progress_on_panel(selected));
            }
        }

        // More above or below.
        let more = |at: Pos2, up: bool| {
            let d = if up { -1.0 } else { 1.0 };
            let points = vec![at + vec2(-14.0, -5.0 * d), at + vec2(0.0, 7.0 * d), at + vec2(14.0, -5.0 * d)];
            painter.add(egui::Shape::line(points, egui::Stroke::new(3.0, text)));
        };
        if self.first > 0 {
            more(pos2(inner.center().x, inner.top() + HEADER - 12.0), true);
        }
        let list_bottom = inner.top() + HEADER + visible as f32 * (ROW_HEIGHT + ROW_GAP);
        if self.first + visible < lines.len() {
            more(pos2(inner.center().x, list_bottom + 2.0), false);
        }

        if let Some(line) = lines.get(self.selected) {
            let help = Self::help(sh, line);
            let galley = painter.layout(help, small, text, inner.width() - 16.0);
            let area = Rect::from_min_max(pos2(inner.left() + 8.0, inner.bottom() - FOOTER + 16.0), inner.right_bottom());
            painter.with_clip_rect(area).galley(area.min, galley, text);
        }

        if let Some(line) = clicked {
            match line {
                Line::Defaults => self.defaults(sh),
                other => self.change(sh, &other, 1.0, false),
            }
        }
        if let Some((line, fraction, released)) = dragged {
            self.set_along(sh, &line, fraction, released);
        }
    }
}

fn set_drift(sh: &mut Shared, drift: Drift) {
    let (effects, db) = sh.effects_mut();
    effects.set_drift(drift, db);
}

fn set_shuffle(sh: &mut Shared, shuffle: Shuffle) {
    let (effects, db) = sh.effects_mut();
    effects.set_shuffle(shuffle, db);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_stay_on_the_grid_and_in_range() {
        assert_eq!(along(0.2, 1.0, 0.0, 1.0, 0.1), 0.3);
        assert_eq!(along(0.95, 1.0, 0.0, 1.0, 0.1), 1.0);
        assert_eq!(along(0.0, -1.0, 0.0, 1.0, 0.1), 0.0);
        assert_eq!(along(0.37, 0.0, 0.0, 1.0, 0.25), 0.25);
    }
}
