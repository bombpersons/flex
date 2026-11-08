//! Faces for video effects that ask for them (`faces = true`): YuNet on the GPU through
//! flex-depth, in a thread of its own. The player's paint callback hands it small frames
//! (see `effects::Renderer`) whenever it's free, and picks up the faces as they come.
//!
//! A detector on its own jitters and misses frames, which would look bad in a shader, so
//! `Tracker` keeps each face in one slot, smooths it, and fades it in and out. Smoothing looks
//! at the frames either side of each one, so it adds no lag, and reports say which frame
//! they're about, so the picture can be held back until the faces for it are known.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use flex_depth::face::{Detection, Face};

pub use flex_depth::face::{HEIGHT, WIDTH};

pub const MAX_FACES: usize = 8;
/// Coordinates of a face: its box (x0, y0, x1, y1), then five points (x, y), from 0 to 1 over
/// the picture, top down.
pub const COORDS: usize = 14;
/// Nothing wanted faces for this long: stop.
const IDLE: Duration = Duration::from_secs(20);
/// Least time between detections, and the least it can be as a multiple of how long one takes,
/// so a slow GPU isn't kept busy with this.
const MIN_INTERVAL: Duration = Duration::from_millis(20);
const INTERVAL_PER_RUN: f32 = 3.0;
/// Reports waiting for the player to collect them.
const MOST_WAITING: usize = 16;
/// How much a box must overlap a face already being tracked to be that face.
const SAME_FACE: f32 = 0.2;
/// Detections in a row before a new face shows, which filters out one-off false ones.
const CONFIRM: u32 = 2;
/// How long a face that's gone missing is kept, and how long fading in or out takes (seconds).
const HOLD: f64 = 0.3;
const FADE: f64 = 0.15;
/// How much the detections around a report count, from the report's own outwards: reports
/// are this many detections behind the newest, less one.
const WEIGHTS: [f32; 3] = [3.0, 2.0, 1.0];
/// The average change in brightness (0–255) between frames that means a new shot.
const CUT: f32 = 32.0;

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Off,
    Starting,
    Ready {
        adapter: String,
        /// How long the last detection took, and how many faces are showing.
        ms: f32,
        faces: usize,
    },
    Failed(String),
}

impl Status {
    pub fn describe(&self) -> String {
        match self {
            Status::Off => "off".into(),
            Status::Starting => "starting…".into(),
            Status::Ready { adapter, ms, faces } => format!("{faces} found, {ms:.0} ms a frame on {adapter}"),
            Status::Failed(e) => format!("⚠ {e}"),
        }
    }
}

/// A face as effects get it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shown {
    pub coords: [f32; COORDS],
    /// 0 to 1, fading as the face comes and goes.
    pub seen: f32,
}

pub type Slots = [Option<Shown>; MAX_FACES];

/// The faces in one video frame.
#[derive(Clone, Debug)]
pub struct Report {
    /// The frame's position in the video (seconds).
    pub time: f32,
    /// The first after a cut or seek: don't blend it with the one before.
    pub after_cut: bool,
    pub slots: Slots,
}

struct Input {
    /// RGBA, bottom row first (as read back from GL), `WIDTH` × `HEIGHT`.
    rgba: Vec<u8>,
    /// How much of that the picture fills, from the top left.
    filled: (f32, f32),
    /// The frame's position in the video (seconds).
    time: f32,
}

struct State {
    status: Status,
    running: bool,
    input: Option<Input>,
    busy: bool,
    reports: VecDeque<Report>,
    last_wanted: Instant,
}

/// Shared between the UI, the paint callback and the worker.
#[derive(Clone)]
pub struct Link(Arc<(Mutex<State>, Condvar)>);

impl Default for Link {
    fn default() -> Self {
        let state = State { status: Status::Off, running: false, input: None, busy: false, reports: VecDeque::new(), last_wanted: Instant::now() };
        Self(Arc::new((Mutex::new(state), Condvar::new())))
    }
}

impl Link {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.0.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// An effect wants faces: starts the worker if it isn't running. Call every frame.
    pub fn wanted(&self) {
        let mut state = self.lock();
        state.last_wanted = Instant::now();
        if state.running || matches!(state.status, Status::Failed(_)) {
            return;
        }
        state.running = true;
        state.status = Status::Starting;
        drop(state);
        let link = self.clone();
        if let Err(e) = std::thread::Builder::new().name("faces".into()).spawn(move || link.work()) {
            let mut state = self.lock();
            state.running = false;
            state.status = Status::Failed(format!("couldn't start: {e}"));
        }
    }

    pub fn status(&self) -> Status {
        self.lock().status.clone()
    }

    /// Whether the worker would take a frame now.
    pub fn ready_for_frame(&self) -> bool {
        let state = self.lock();
        matches!(state.status, Status::Ready { .. }) && state.input.is_none() && !state.busy
    }

    /// Hands over a frame: RGBA, bottom row first, `WIDTH` × `HEIGHT`, with the picture filling
    /// `filled` of it from the top left. `time` is its position in the video.
    pub fn submit(&self, rgba: Vec<u8>, filled: (f32, f32), time: f32) {
        self.lock().input = Some(Input { rgba, filled, time });
        self.0.1.notify_all();
    }

    /// The reports that have come in since the last call, oldest first.
    pub fn take_reports(&self) -> Vec<Report> {
        self.lock().reports.drain(..).collect()
    }

    fn publish(&self, reports: impl IntoIterator<Item = Report>) {
        let mut state = self.lock();
        state.reports.extend(reports);
        let extra = state.reports.len().saturating_sub(MOST_WAITING);
        state.reports.drain(..extra);
    }

    fn work(&self) {
        let mut face = match Face::new() {
            Ok(face) => face,
            Err(e) => {
                log::warn!("faces: {e:#}");
                let mut state = self.lock();
                state.status = Status::Failed(format!("{e:#}"));
                state.running = false;
                return;
            }
        };
        log::info!("faces: running on {}", face.adapter());
        self.lock().status = Status::Ready { adapter: face.adapter().to_owned(), ms: 0.0, faces: 0 };

        let start = Instant::now();
        let mut tracker = Tracker::default();
        let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
        let mut last_gray: Vec<u8> = Vec::new();
        let mut last_time: Option<f32> = None;
        let mut after_cut = true;
        loop {
            let input = {
                let mut state = self.lock();
                loop {
                    if let Some(input) = state.input.take() {
                        state.busy = true;
                        break Some(input);
                    }
                    if state.last_wanted.elapsed() > IDLE {
                        break None;
                    }
                    state = self.0.1.wait_timeout(state, Duration::from_millis(500)).unwrap_or_else(|e| e.into_inner()).0;
                }
            };
            let Some(input) = input else { break };
            // GL's rows run bottom-up; the model's top-down.
            for y in 0..HEIGHT {
                let src = &input.rgba[(HEIGHT - 1 - y) * WIDTH * 4..][..WIDTH * 4];
                for (x, px) in src.as_chunks::<4>().0.iter().enumerate() {
                    rgb[(y * WIDTH + x) * 3..][..3].copy_from_slice(&px[..3]);
                }
            }
            let gray: Vec<u8> = rgb.as_chunks::<3>().0.iter().step_by(16).map(|p| p[1]).collect();
            let seeked = last_time.is_some_and(|t| input.time < t || input.time > t + 1.0);
            if seeked || is_cut(&last_gray, &gray) {
                self.publish(tracker.finish());
                tracker = Tracker::default();
                after_cut = true;
            }
            last_gray = gray;
            last_time = Some(input.time);

            let started = Instant::now();
            let found = match face.run(&rgb) {
                Ok(found) => found,
                Err(e) => {
                    log::warn!("faces: {e:#}");
                    let mut state = self.lock();
                    state.status = Status::Failed(format!("{e:#}"));
                    state.busy = false;
                    break;
                }
            };
            let ms = started.elapsed().as_secs_f32() * 1000.0;
            let found: Vec<[f32; COORDS]> = found.iter().map(|d| coords(d, input.filled)).collect();
            let report = tracker.update(&found, start.elapsed().as_secs_f64(), input.time).map(|mut report| {
                report.after_cut = std::mem::take(&mut after_cut);
                report
            });
            self.publish(report);
            self.lock().status = Status::Ready { adapter: face.adapter().to_owned(), ms, faces: tracker.showing() };
            std::thread::sleep(MIN_INTERVAL.max(Duration::from_secs_f32(ms * INTERVAL_PER_RUN / 1000.0)).saturating_sub(started.elapsed()));
            self.lock().busy = false;
        }
        drop(face);
        log::info!("faces: stopped");
        let mut state = self.lock();
        state.running = false;
        state.reports.clear();
        if matches!(state.status, Status::Ready { .. }) {
            state.status = Status::Off;
        }
    }
}

fn is_cut(last: &[u8], now: &[u8]) -> bool {
    if last.len() != now.len() || now.is_empty() {
        return false;
    }
    let total: u32 = last.iter().zip(now).map(|(a, b)| u32::from(a.abs_diff(*b))).sum();
    total as f32 / now.len() as f32 > CUT
}

/// A detection's coordinates over the picture, which fills `filled` of what the model saw.
fn coords(d: &Detection, filled: (f32, f32)) -> [f32; COORDS] {
    let (sx, sy) = (1.0 / (filled.0 * WIDTH as f32).max(1.0), 1.0 / (filled.1 * HEIGHT as f32).max(1.0));
    let mut c = [0.0; COORDS];
    c[..4].copy_from_slice(&[d.x * sx, d.y * sy, (d.x + d.width) * sx, (d.y + d.height) * sy]);
    for (n, [x, y]) in d.points.iter().enumerate() {
        c[4 + n * 2] = x * sx;
        c[5 + n * 2] = y * sy;
    }
    c
}

fn overlap(a: &[f32; COORDS], b: &[f32; COORDS]) -> f32 {
    let w = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
    let h = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let inter = w * h;
    inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter).max(1e-9)
}

/// A face being followed.
struct Track {
    /// What's been detected lately: which detection, and where.
    recent: VecDeque<(u64, [f32; COORDS])>,
    updated: f64,
    hits: u32,
    seen: f32,
}

impl Track {
    fn new(detection: u64, face: [f32; COORDS], now: f64) -> Self {
        Self { recent: VecDeque::from([(detection, face)]), updated: now, hits: 1, seen: 0.0 }
    }

    fn newest(&self) -> &[f32; COORDS] {
        &self.recent[self.recent.len() - 1].1
    }

    fn follow(&mut self, detection: u64, target: &[f32; COORDS], now: f64) {
        self.recent.push_back((detection, *target));
        if self.recent.len() > 2 * WEIGHTS.len() - 1 {
            self.recent.pop_front();
        }
        self.updated = now;
        self.hits += 1;
    }

    /// Where it was at `detection`: an average of the detections either side, which leaves
    /// movement where it was and only takes out the jitter.
    fn at(&self, detection: u64) -> [f32; COORDS] {
        let (mut sum, mut total) = ([0.0; COORDS], 0.0);
        for (at, face) in &self.recent {
            let Some(weight) = WEIGHTS.get(at.abs_diff(detection) as usize) else { continue };
            total += weight;
            for (sum, v) in sum.iter_mut().zip(face) {
                *sum += v * weight;
            }
        }
        if total == 0.0 {
            return *self.newest(); // being held after going missing
        }
        sum.map(|v| v / total)
    }
}

/// Keeps faces in their slots from one detection to the next.
#[derive(Default)]
struct Tracker {
    slots: [Option<Track>; MAX_FACES],
    last: Option<f64>,
    detections: u64,
    /// Detections not reported yet: which, and the frame's position.
    unreported: VecDeque<(u64, f32)>,
}

impl Tracker {
    /// `found` is what's been detected at `now` (seconds) in the frame at `time` in the video.
    /// Returns the report for an earlier frame, now that the frames after it are known.
    fn update(&mut self, found: &[[f32; COORDS]], now: f64, time: f32) -> Option<Report> {
        let dt = now - self.last.unwrap_or(now);
        self.last = Some(now);
        let detection = self.detections;
        self.detections += 1;

        // The most overlapping pairs first, so a face can't be taken by a worse match.
        let mut pairs: Vec<(f32, usize, usize)> = Vec::new();
        for (s, track) in self.slots.iter().enumerate() {
            let Some(track) = track else { continue };
            for (f, face) in found.iter().enumerate() {
                let amount = overlap(track.newest(), face);
                if amount >= SAME_FACE {
                    pairs.push((amount, s, f));
                }
            }
        }
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));
        let (mut slot_taken, mut face_taken) = ([false; MAX_FACES], vec![false; found.len()]);
        for (_, s, f) in pairs {
            if !slot_taken[s] && !face_taken[f] {
                slot_taken[s] = true;
                face_taken[f] = true;
                if let Some(track) = &mut self.slots[s] {
                    track.follow(detection, &found[f], now);
                }
            }
        }
        for (face, _) in found.iter().zip(&face_taken).filter(|(_, taken)| !**taken) {
            if let Some(free) = self.slots.iter_mut().find(|s| s.is_none()) {
                *free = Some(Track::new(detection, *face, now));
            }
        }

        let step = (dt / FADE) as f32;
        for slot in &mut self.slots {
            let Some(track) = slot else { continue };
            let fresh = now - track.updated <= HOLD;
            let target = if fresh && track.hits >= CONFIRM { 1.0 } else { 0.0 };
            track.seen = if target > 0.5 { (track.seen + step).min(1.0) } else { (track.seen - step).max(0.0) };
            if !fresh && track.seen <= 0.0 {
                *slot = None;
            }
        }

        self.unreported.push_back((detection, time));
        (self.unreported.len() >= WEIGHTS.len()).then(|| self.report()).flatten()
    }

    fn report(&mut self) -> Option<Report> {
        let (detection, time) = self.unreported.pop_front()?;
        let mut slots: Slots = [None; MAX_FACES];
        for (slot, out) in self.slots.iter().zip(&mut slots) {
            *out = slot.as_ref().map(|track| Shown { coords: track.at(detection), seen: track.seen });
        }
        Some(Report { time, after_cut: false, slots })
    }

    /// The frames not reported yet, as there won't be any more after them.
    fn finish(&mut self) -> Vec<Report> {
        std::iter::from_fn(|| self.report()).collect()
    }

    fn showing(&self) -> usize {
        self.slots.iter().flatten().filter(|t| t.seen > 0.0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face(x: f32, y: f32) -> [f32; COORDS] {
        let mut c = [0.0; COORDS];
        c[..4].copy_from_slice(&[x, y, x + 0.1, y + 0.2]);
        c
    }

    const STEP: f64 = 1.0 / 24.0;

    /// The tracker fed detections `from..to`, returning its reports in order.
    fn feed(t: &mut Tracker, from: usize, to: usize, found: impl Fn(usize) -> Vec<[f32; COORDS]>) -> Vec<Report> {
        (from..to).filter_map(|i| t.update(&found(i), i as f64 * STEP, i as f32 * STEP as f32)).collect()
    }

    #[test]
    fn a_face_shows_after_being_confirmed_and_fades_in() {
        let mut t = Tracker::default();
        let reports = feed(&mut t, 0, 16, |_| vec![face(0.2, 0.2)]);
        assert_eq!(reports[0].time, 0.0, "reports start at the first frame");
        assert!(reports[0].slots[0].unwrap().seen > 0.0, "known to be real by the time it's reported");
        assert_eq!(reports.last().unwrap().slots[0].unwrap().seen, 1.0);
    }

    #[test]
    fn a_one_off_false_detection_never_shows_and_goes_away() {
        let mut t = Tracker::default();
        let reports = feed(&mut t, 0, 24, |i| if i == 0 { vec![face(0.7, 0.7)] } else { vec![] });
        assert!(reports.iter().all(|r| r.slots[0].is_none_or(|s| s.seen == 0.0)));
        assert!(t.slots[0].is_none());
    }

    #[test]
    fn faces_keep_their_slots_whatever_order_they_come_in() {
        let mut t = Tracker::default();
        let reports = feed(&mut t, 0, 6, |i| if i % 2 == 0 { vec![face(0.1, 0.1), face(0.6, 0.6)] } else { vec![face(0.6, 0.6), face(0.1, 0.1)] });
        let last = &reports.last().unwrap().slots;
        assert!((last[0].unwrap().coords[0] - 0.1).abs() < 0.01);
        assert!((last[1].unwrap().coords[0] - 0.6).abs() < 0.01);
    }

    #[test]
    fn a_face_missed_briefly_is_held_then_fades() {
        let mut t = Tracker::default();
        feed(&mut t, 0, 12, |_| vec![face(0.2, 0.2)]);
        let held = feed(&mut t, 12, 13, |_| vec![]);
        assert_eq!(held[0].slots[0].unwrap().seen, 1.0, "just one miss");
        let back = feed(&mut t, 13, 14, |_| vec![face(0.2, 0.2)]);
        assert_eq!(back[0].slots[0].unwrap().seen, 1.0);
        feed(&mut t, 14, 40, |_| vec![]);
        assert!(t.slots[0].is_none(), "should have gone");
    }

    #[test]
    fn jitter_is_steadied_and_movement_has_no_lag() {
        let mut t = Tracker::default();
        let reports = feed(&mut t, 0, 60, |i| vec![face(0.3 + if i % 2 == 0 { 0.004 } else { -0.004 }, 0.3)]);
        let x = reports.last().unwrap().slots[0].unwrap().coords[0];
        assert!((x - 0.3).abs() < 0.0015, "{x}");
        // Moving steadily: each report is exactly where the face was at that frame.
        let mut t = Tracker::default();
        let at = |i: usize| 0.2 + i as f32 * 0.004;
        let reports = feed(&mut t, 0, 30, |i| vec![face(at(i), 0.3)]);
        for (i, report) in reports.iter().enumerate().skip(2) {
            assert!((report.slots[0].unwrap().coords[0] - at(i)).abs() < 1e-4, "frame {i}");
            assert!((report.time - i as f32 * STEP as f32).abs() < 1e-4);
        }
    }

    #[test]
    fn nothing_is_left_unreported_at_a_cut() {
        let mut t = Tracker::default();
        let reports = feed(&mut t, 0, 6, |_| vec![face(0.2, 0.2)]);
        assert_eq!(reports.len(), 4);
        assert_eq!(t.finish().len(), 2);
    }

    #[test]
    fn coordinates_are_over_the_picture_not_the_padding() {
        let d = Detection { x: 160.0, y: 44.0, width: 32.0, height: 44.0, score: 1.0, points: [[160.0, 44.0]; 5] };
        // A picture filling half the width and all the height.
        let c = coords(&d, (0.5, 1.0));
        assert!((c[0] - 0.5).abs() < 1e-5 && (c[1] - 0.125).abs() < 1e-5, "{c:?}");
        assert!((c[2] - 0.6).abs() < 1e-5, "{c:?}");
    }

    #[test]
    fn a_big_change_is_a_cut() {
        assert!(is_cut(&[0, 0, 0], &[200, 200, 200]));
        assert!(!is_cut(&[100, 100, 100], &[110, 95, 105]));
        assert!(!is_cut(&[], &[200]));
    }
}
