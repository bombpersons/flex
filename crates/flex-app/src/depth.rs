//! Depth for video effects that ask for it (`depth = true`): Depth Anything V2 Small on the
//! GPU through flex-depth, in a thread of its own. The player's paint callback hands it small
//! frames (see `effects::Renderer`) whenever it's free, and picks up depth maps as they come.
//!
//! The model (99 MB) is downloaded into the data folder the first time it's needed. The
//! thread stops, freeing the GPU memory, once nothing has asked for depth for a while.

use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use flex_depth::Depth;

use crate::ground::Ground;

/// The sizes the model can run at, in patches of 14 pixels, biggest first: all about 16:9. Below
/// the last, it gets little faster.
const LEVELS: [(usize, usize); 5] = [(37, 21), (32, 18), (28, 16), (24, 14), (21, 12)];
/// The frames in and depth maps out are always the biggest size: other levels scale them.
pub const WIDTH: usize = LEVELS[0].0 * flex_depth::PATCH;
pub const HEIGHT: usize = LEVELS[0].1 * flex_depth::PATCH;
/// A depth map taking longer than this, on average (ms), moves to a smaller size.
const TOO_SLOW: f32 = 35.0;
/// A bigger size is tried if it should take no longer than this (ms).
const ROOM: f32 = 28.0;
/// How quickly that average follows.
const PACE_FOLLOW: f32 = 0.15;
/// How long to wait after changing size before judging the new one.
const SETTLE: Duration = Duration::from_secs(3);
/// After a size turned out too slow, how long before going back up to it.
const RETRY: Duration = Duration::from_secs(60);
/// Nothing wanted depth for this long: stop.
const IDLE: Duration = Duration::from_secs(20);
/// How much of each new depth map goes into what effects see where it's hardly changed (the
/// rest is the last one, which hides the model's flicker).
const BLEND: f32 = 0.25;
/// How much depth (0–1) has to change, on average around a point, to count as more than
/// flicker, and how much is movement, which is taken whole so depth keeps up with it.
const FLICKER: f32 = 0.01;
const MOVEMENT: f32 = 0.04;
/// How far around a point that average goes, in pixels of the depth map.
const AROUND: usize = 7;
/// How quickly the range used to scale depth to 0–1 follows the picture.
const RANGE_FOLLOW: f32 = 0.2;
/// The average change between depth maps, each scaled by its own range, that means a new shot.
const CUT: f32 = 0.18;
/// At most this often, to leave the GPU time for the video (~30 depth maps a second, so a
/// film's every frame gets one if the GPU can manage it).
const MIN_INTERVAL: Duration = Duration::from_millis(33);

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Off,
    Starting,
    /// Downloading the model: percent done.
    Downloading(u8),
    Ready {
        adapter: String,
        /// How long the last depth map took.
        ms: f32,
        /// The size the model is running at, in pixels.
        size: (usize, usize),
        /// Whether people, walls and sky are being found too.
        masks: bool,
    },
    Failed(String),
}

impl Status {
    pub fn describe(&self) -> String {
        match self {
            Status::Off => "off".into(),
            Status::Starting => "starting…".into(),
            Status::Downloading(p) => format!("downloading the depth model ({p}%)…"),
            Status::Ready { adapter, ms, size, masks } => format!("{ms:.0} ms a frame at {}×{}{} on {adapter}", size.0, size.1, if *masks { " with masks" } else { "" }),
            Status::Failed(e) => format!("⚠ {e}"),
        }
    }
}

/// A depth map for effects: `values` from 0 (far) to 1 (near), bottom row first (as GL
/// textures go), `WIDTH` × `HEIGHT`.
pub struct DepthMap {
    pub values: Vec<f32>,
    pub serial: u64,
    /// The video frame it's of, by mpv's position.
    pub time: f32,
    /// How much of each pixel is a person (red), wall (green) and sky (blue), RGBA, bottom row
    /// first, `WIDTH` × `HEIGHT`, when an effect asked for them.
    pub masks: Option<Vec<u8>>,
    /// Where the ground is, as `ground::Ground::update` gives it.
    pub ground: [f32; 4],
}

struct State {
    status: Status,
    running: bool,
    /// A frame waiting: RGBA, bottom row first (as read back from GL), `WIDTH` × `HEIGHT`.
    input: Option<(Vec<u8>, f32)>,
    busy: bool,
    output: Option<Arc<DepthMap>>,
    last_wanted: Instant,
    /// Whether an effect wants people, walls and sky as well as depth.
    masks_wanted: bool,
    /// Into `LEVELS`, kept so the next video starts at the size that worked for this one.
    level: usize,
}

fn tokens(level: usize) -> f32 {
    (LEVELS[level].0 * LEVELS[level].1 + 1) as f32
}

/// Chooses the size the model runs at from how long depth maps are taking. Nearly all of a
/// map's time goes with the number of patches, so that says how long another size would take.
struct Pace {
    level: usize,
    average: Option<f32>,
    changed: Instant,
    retry: Option<Instant>,
}

impl Pace {
    fn new(level: usize, now: Instant) -> Self {
        Self { level, average: None, changed: now, retry: None }
    }

    /// A map took `ms`. Returns the level to move to, if it's time to move.
    fn update(&mut self, ms: f32, now: Instant) -> Option<usize> {
        let average = self.average.map_or(ms, |a| a + (ms - a) * PACE_FOLLOW);
        self.average = Some(average);
        if now.duration_since(self.changed) < SETTLE {
            return None;
        }
        let at = |level| average * tokens(level) / tokens(self.level);
        let level = if average > TOO_SLOW && self.level + 1 < LEVELS.len() {
            // Straight to the size that should be fast enough, or the smallest.
            self.retry = Some(now + RETRY);
            (self.level + 1..LEVELS.len()).find(|l| at(*l) <= TOO_SLOW).unwrap_or(LEVELS.len() - 1)
        } else if self.level > 0 && at(self.level - 1) <= ROOM && self.retry.is_none_or(|t| now >= t) {
            self.level - 1
        } else {
            return None;
        };
        self.level = level;
        self.average = None;
        self.changed = now;
        Some(level)
    }
}

/// Mask classes effects get: person, wall and sky.
const MASKS: usize = 3;

/// The model's [other, person, wall, sky] planes as person, wall and sky at `WIDTH` × `HEIGHT`.
fn mask_planes(planes: &[f32], size: (usize, usize)) -> [Vec<f32>; MASKS] {
    let pixels = size.0 * size.1;
    std::array::from_fn(|c| {
        let plane = &planes[(c + 1) * pixels..][..pixels];
        if size == (WIDTH, HEIGHT) { plane.to_vec() } else { scaled(plane, size, (WIDTH, HEIGHT)) }
    })
}

/// Masks (rows from the top) as a texture's bytes: RGBA, bottom row first.
fn mask_bytes(planes: &[Vec<f32>; MASKS]) -> Vec<u8> {
    let mut out = vec![255u8; WIDTH * HEIGHT * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let px = &mut out[((HEIGHT - 1 - y) * WIDTH + x) * 4..][..MASKS];
            for (byte, plane) in px.iter_mut().zip(planes) {
                *byte = (plane[y * WIDTH + x].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    out
}

/// For each pixel along a side of `to`: the two pixels of `from` it falls between, and how far.
fn spans(from: usize, to: usize) -> Vec<(usize, usize, f32)> {
    (0..to)
        .map(|i| {
            let at = ((i as f32 + 0.5) * from as f32 / to as f32 - 0.5).clamp(0.0, (from - 1) as f32);
            let a = at as usize;
            (a, (a + 1).min(from - 1), at - a as f32)
        })
        .collect()
}

/// A depth map resized, bilinearly.
fn scaled(src: &[f32], (sw, sh): (usize, usize), (dw, dh): (usize, usize)) -> Vec<f32> {
    let (xs, ys) = (spans(sw, dw), spans(sh, dh));
    let mut out = Vec::with_capacity(dw * dh);
    for &(y0, y1, ty) in &ys {
        let (r0, r1) = (&src[y0 * sw..][..sw], &src[y1 * sw..][..sw]);
        for &(x0, x1, tx) in &xs {
            let top = r0[x0] + (r0[x1] - r0[x0]) * tx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * tx;
            out.push(top + (bottom - top) * ty);
        }
    }
    out
}

/// An RGB picture resized, bilinearly.
fn scaled_rgb(src: &[u8], (sw, sh): (usize, usize), (dw, dh): (usize, usize)) -> Vec<u8> {
    let (xs, ys) = (spans(sw, dw), spans(sh, dh));
    let mut out = Vec::with_capacity(dw * dh * 3);
    for &(y0, y1, ty) in &ys {
        let (r0, r1) = (&src[y0 * sw * 3..][..sw * 3], &src[y1 * sw * 3..][..sw * 3]);
        for &(x0, x1, tx) in &xs {
            for c in 0..3 {
                let (a, b) = (f32::from(r0[x0 * 3 + c]), f32::from(r0[x1 * 3 + c]));
                let (d, e) = (f32::from(r1[x0 * 3 + c]), f32::from(r1[x1 * 3 + c]));
                let top = a + (b - a) * tx;
                let bottom = d + (e - d) * tx;
                out.push((top + (bottom - top) * ty + 0.5) as u8);
            }
        }
    }
    out
}

fn mean_change(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f32>() / a.len().max(1) as f32
}

/// `values` averaged over a square reaching `radius` pixels each way (fewer at the edges).
fn averaged(values: &[f32], (w, h): (usize, usize), radius: usize) -> Vec<f32> {
    let line = |src: &[f32], dst: &mut [f32], len: usize, step: usize| {
        let mut sum: f32 = (0..=radius.min(len - 1)).map(|i| src[i * step]).sum();
        let mut count = radius.min(len - 1) + 1;
        for i in 0..len {
            dst[i * step] = sum / count as f32;
            if i + radius + 1 < len {
                sum += src[(i + radius + 1) * step];
                count += 1;
            }
            if i >= radius {
                sum -= src[(i - radius) * step];
                count -= 1;
            }
        }
    };
    let mut across = vec![0.0; values.len()];
    for (src, dst) in values.chunks(w).zip(across.chunks_mut(w)) {
        line(src, dst, w, 1);
    }
    let mut out = vec![0.0; values.len()];
    for x in 0..w {
        line(&across[x..], &mut out[x..], h, w);
    }
    out
}

/// Moves `previous` towards `values`, and leaves the result in `values`: gently where only
/// flicker has changed them, and all the way where something moved. Going by the change
/// around each point keeps a moving edge from breaking up into speckles.
fn settle(values: &mut [f32], previous: &[f32], size: (usize, usize)) {
    let change: Vec<f32> = values.iter().zip(previous).map(|(v, p)| (v - p).abs()).collect();
    for ((v, p), c) in values.iter_mut().zip(previous).zip(averaged(&change, size, AROUND)) {
        let t = ((c - FLICKER) / (MOVEMENT - FLICKER)).clamp(0.0, 1.0);
        let blend = BLEND + (1.0 - BLEND) * t * t * (3.0 - 2.0 * t);
        *v = p + (*v - p) * blend;
    }
}

/// Shared between the UI, the paint callback and the worker.
#[derive(Clone)]
pub struct Link(Arc<(Mutex<State>, Condvar)>);

impl Default for Link {
    fn default() -> Self {
        let state = State { status: Status::Off, running: false, input: None, busy: false, output: None, last_wanted: Instant::now(), masks_wanted: false, level: 0 };
        Self(Arc::new((Mutex::new(state), Condvar::new())))
    }
}

fn model_path() -> PathBuf {
    flex_core::data_dir().join("models").join("depth-anything-v2-small.safetensors")
}

impl Link {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.0.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// An effect wants depth, and maybe `masks` too: starts the worker if it isn't running.
    /// Call every frame.
    pub fn wanted(&self, masks: bool) {
        let mut state = self.lock();
        state.last_wanted = Instant::now();
        state.masks_wanted = masks;
        if state.running || matches!(state.status, Status::Failed(_)) {
            return;
        }
        state.running = true;
        state.status = Status::Starting;
        drop(state);
        let link = self.clone();
        if let Err(e) = std::thread::Builder::new().name("depth".into()).spawn(move || link.work()) {
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
        // Even while busy: read back now, the next frame is ready the moment this one's done.
        matches!(state.status, Status::Ready { .. }) && state.input.is_none()
    }

    /// Hands over a frame (RGBA, bottom row first, `WIDTH` × `HEIGHT`) and its position.
    pub fn submit(&self, rgba: Vec<u8>, time: f32) {
        self.lock().input = Some((rgba, time));
        self.0.1.notify_all();
    }

    pub fn latest(&self) -> Option<Arc<DepthMap>> {
        self.lock().output.clone()
    }

    fn work(&self) {
        let path = model_path();
        let (level, mut with_masks) = {
            let state = self.lock();
            (state.level, state.masks_wanted)
        };
        let mut pace = Pace::new(level, Instant::now());
        let result = (|| -> anyhow::Result<Depth> {
            if let Some(dir) = path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let total = flex_depth::WEIGHTS_BYTES;
            let progress = |done: u64| {
                let pct = (done * 100 / total.max(1)).min(100) as u8;
                let mut state = self.lock();
                if state.status != Status::Downloading(pct) && done < total {
                    state.status = Status::Downloading(pct);
                }
            };
            flex_core::update::download_verified(flex_depth::WEIGHTS_URL, "the depth model", total, flex_depth::WEIGHTS_SHA256, &path, progress)?;
            Depth::load(&path, LEVELS[pace.level], with_masks)
        })();
        let mut depth = match result {
            Ok(depth) => depth,
            Err(e) => {
                log::warn!("depth: {e:#}");
                let mut state = self.lock();
                state.status = Status::Failed(format!("{e:#}"));
                state.running = false;
                return;
            }
        };
        log::info!("depth: running on {}", depth.adapter());
        self.lock().status = Status::Ready { adapter: depth.adapter().to_owned(), ms: 0.0, size: depth.size(), masks: with_masks };

        let mut previous: Option<Vec<f32>> = None;
        let mut previous_masks: Option<[Vec<f32>; MASKS]> = None;
        let mut last_own: Option<Vec<f32>> = None;
        let mut range: Option<(f32, f32)> = None;
        let mut ground = Ground::default();
        let mut serial = 0;
        let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
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
            let Some((rgba, time)) = input else { break };
            // GL's rows run bottom-up; the model's top-down.
            for y in 0..HEIGHT {
                let src = &rgba[(HEIGHT - 1 - y) * WIDTH * 4..][..WIDTH * 4];
                for (x, px) in src.as_chunks::<4>().0.iter().enumerate() {
                    rgb[(y * WIDTH + x) * 3..][..3].copy_from_slice(&px[..3]);
                }
            }
            let started = Instant::now();
            let size = depth.size();
            let result = if size == (WIDTH, HEIGHT) { depth.run(&rgb) } else { depth.run(&scaled_rgb(&rgb, (WIDTH, HEIGHT), size)) };
            let result = result.and_then(|raw| Ok((raw, if with_masks { Some(depth.masks()?) } else { None })));
            let ms = started.elapsed().as_secs_f32() * 1000.0;
            let (raw, planes) = match result {
                Ok((raw, planes)) if size == (WIDTH, HEIGHT) => (raw, planes),
                Ok((raw, planes)) => (scaled(&raw, size, (WIDTH, HEIGHT)), planes),
                Err(e) => {
                    log::warn!("depth: {e:#}");
                    let mut state = self.lock();
                    state.status = Status::Failed(format!("{e:#}"));
                    state.busy = false;
                    break;
                }
            };
            // Scaled to 0–1 by a range that follows the picture's smoothly (so it doesn't
            // pump), blended with the last map, and flipped back to bottom-up.
            let (lo, hi) = raw.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            let own_scale = 1.0 / (hi - lo).max(1e-6);
            let own: Vec<f32> = raw.iter().map(|v| (v - lo) * own_scale).collect();
            // A new shot has nothing to do with the last one: start again from this picture.
            if last_own.as_ref().is_some_and(|last| mean_change(&own, last) > CUT) {
                previous = None;
                previous_masks = None;
                range = None;
                ground.forget();
            }
            last_own = Some(own);
            let (lo, hi) = match range {
                Some((l, h)) => (l + (lo - l) * RANGE_FOLLOW, h + (hi - h) * RANGE_FOLLOW),
                None => (lo, hi),
            };
            range = Some((lo, hi));
            let scale = 1.0 / (hi - lo).max(1e-6);
            let mut values = vec![0.0; WIDTH * HEIGHT];
            for y in 0..HEIGHT {
                for x in 0..WIDTH {
                    values[(HEIGHT - 1 - y) * WIDTH + x] = ((raw[y * WIDTH + x] - lo) * scale).clamp(0.0, 1.0);
                }
            }
            if let Some(prev) = &previous {
                settle(&mut values, prev, (WIDTH, HEIGHT));
            }
            serial += 1;
            previous = Some(values.clone());
            let masks = planes.map(|planes| {
                let mut now = mask_planes(&planes, size);
                if let Some(before) = &previous_masks {
                    for (plane, before) in now.iter_mut().zip(before) {
                        settle(plane, before, (WIDTH, HEIGHT));
                    }
                }
                let rgba = mask_bytes(&now);
                previous_masks = Some(now);
                rgba
            });
            let ground = ground.update(&values, masks.as_deref(), (WIDTH, HEIGHT));
            {
                let mut state = self.lock();
                state.output = Some(Arc::new(DepthMap { values, serial, time, masks, ground }));
                state.status = Status::Ready { adapter: depth.adapter().to_owned(), ms, size, masks: with_masks };
            }
            // Still "busy" while waiting, so no frame is read back in the meantime.
            std::thread::sleep(MIN_INTERVAL.saturating_sub(started.elapsed()));
            let moved = pace.update(ms, Instant::now());
            let masks_wanted = self.lock().masks_wanted;
            if moved.is_some() || masks_wanted != with_masks {
                let level = pace.level;
                log::info!("depth: {ms:.0} ms a frame, running at {} × {} patches, masks {masks_wanted}", LEVELS[level].0, LEVELS[level].1);
                self.lock().level = level;
                with_masks = masks_wanted;
                previous_masks = None;
                // The old one goes first: a slow GPU may not have room for both.
                drop(depth);
                depth = match Depth::load(&path, LEVELS[level], with_masks) {
                    Ok(depth) => depth,
                    Err(e) => {
                        log::warn!("depth: {e:#}");
                        let mut state = self.lock();
                        state.status = Status::Failed(format!("{e:#}"));
                        state.running = false;
                        state.busy = false;
                        state.output = None;
                        return;
                    }
                };
            }
            self.lock().busy = false;
        }
        drop(depth);
        log::info!("depth: stopped");
        let mut state = self.lock();
        state.running = false;
        state.output = None;
        if matches!(state.status, Status::Ready { .. }) {
            state.status = Status::Off;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: (usize, usize) = (64, 40);

    #[test]
    fn averaged_handles_edges_and_keeps_a_flat_field() {
        let flat = vec![0.5; SIZE.0 * SIZE.1];
        assert!(averaged(&flat, SIZE, AROUND).iter().all(|v| (v - 0.5).abs() < 1e-4));
        let mut dot = vec![0.0; 25];
        dot[0] = 1.0;
        let out = averaged(&dot, (5, 5), 1);
        // The corner's square has 2 × 2 pixels in it.
        assert!((out[0] - 0.25).abs() < 1e-6);
        assert!((out[6] - 1.0 / 9.0).abs() < 1e-6);
        assert_eq!(out[12], 0.0);
    }

    #[test]
    fn flicker_is_smoothed_and_movement_is_taken_whole() {
        let previous = vec![0.5; SIZE.0 * SIZE.1];
        let mut flicker = vec![0.505; SIZE.0 * SIZE.1];
        settle(&mut flicker, &previous, SIZE);
        assert!((flicker[500] - (0.5 + 0.005 * BLEND)).abs() < 1e-5);
        let mut moved = vec![0.8; SIZE.0 * SIZE.1];
        settle(&mut moved, &previous, SIZE);
        assert!((moved[500] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn scaling_keeps_flat_pictures_and_sizes() {
        let out = scaled(&vec![0.25; 28 * 14], (28, 14), (56, 21));
        assert_eq!(out.len(), 56 * 21);
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-5));
        let out = scaled_rgb(&[7, 80, 200].repeat(56 * 28), (56, 28), (28, 14));
        assert_eq!(out.len(), 28 * 14 * 3);
        assert!(out.chunks(3).all(|p| p == [7, 80, 200]));
    }

    #[test]
    fn pace_moves_down_to_a_size_that_fits_and_settles_first() {
        let now = Instant::now();
        let mut pace = Pace::new(0, now);
        assert_eq!(pace.update(200.0, now + Duration::from_secs(1)), None);
        // 200 ms at the biggest size: only the smallest should come in under the limit.
        assert_eq!(pace.update(200.0, now + SETTLE), Some(LEVELS.len() - 1));
        assert_eq!(pace.level, LEVELS.len() - 1);
    }

    #[test]
    fn pace_waits_before_retrying_a_size_that_was_too_slow() {
        let now = Instant::now();
        let mut pace = Pace::new(0, now);
        assert_eq!(pace.update(TOO_SLOW + 1.0, now + SETTLE), Some(1));
        // Fast enough to go back up, by the estimate, but it only just failed.
        let t = now + SETTLE * 3;
        assert_eq!(pace.update(10.0, t), None);
        assert_eq!(pace.update(10.0, now + SETTLE + RETRY), Some(0));
    }

    #[test]
    fn pace_stays_put_between_too_slow_and_room() {
        let now = Instant::now();
        let mut pace = Pace::new(1, now);
        // Fine here; the size above would be slower than ROOM.
        let ms = ROOM * tokens(1) / tokens(0) + 5.0;
        assert_eq!(pace.update(ms, now + SETTLE), None);
    }

    #[test]
    fn masks_skip_other_and_go_bottom_row_first() {
        // 4 classes of 2 × 1 pictures would need scaling; use the full size for the layout.
        let pixels = WIDTH * HEIGHT;
        let mut planes = vec![0.0; 4 * pixels];
        planes[pixels] = 1.0; // person, at the top left
        let bytes = mask_bytes(&mask_planes(&planes, (WIDTH, HEIGHT)));
        assert_eq!(&bytes[(HEIGHT - 1) * WIDTH * 4..][..4], &[255, 0, 0, 255]);
        assert_eq!(&bytes[..4], &[0, 0, 0, 255]);
    }

    #[test]
    fn a_new_shot_is_a_big_average_change() {
        assert!(mean_change(&[0.1, 0.9], &[0.9, 0.1]) > CUT);
        assert!(mean_change(&[0.5, 0.52], &[0.5, 0.5]) < CUT);
    }
}
