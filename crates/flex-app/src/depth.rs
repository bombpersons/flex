//! Depth for video effects that ask for it (`depth = true`): MoGe-2 (depth with the shape of
//! things in it, and which way each surface faces), or Depth Anything V2 Small, on the GPU through
//! flex-depth, in a thread of its own. The player's paint callback hands it small frames (see
//! `effects::Renderer`) whenever it's free, and picks up depth maps as they come.
//!
//! The model (141 or 99 MB) is downloaded into the data folder the first time it's needed. The
//! thread stops, freeing the GPU memory, once nothing has asked for depth for a while.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use flex_depth::Depth;
use flex_depth::moge::Moge;

use crate::flow::{FLOW_HEIGHT, FLOW_WIDTH, Report};
use crate::ground::Ground;

/// The sizes the model can run at, in patches of 14 pixels, biggest first: all about 16:9. Below
/// the last, it gets little faster.
const LEVELS: [(usize, usize); 5] = [(37, 21), (32, 18), (28, 16), (24, 14), (21, 12)];
/// The frames in and depth maps out are always the biggest size: other levels scale them.
pub const WIDTH: usize = LEVELS[0].0 * flex_depth::PATCH;
pub const HEIGHT: usize = LEVELS[0].1 * flex_depth::PATCH;
/// The same for MoGe-2, which costs more per patch.
const MOGE_LEVELS: [(usize, usize); 4] = [(28, 16), (25, 14), (21, 12), (18, 10)];
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
/// At most this often, to leave the GPU time for the video (~30 depth maps a second, so a
/// film's every frame gets one if the GPU can manage it).
const MIN_INTERVAL: Duration = Duration::from_millis(33);
/// With movement to carry them along, depth and masks come from the model less often: starting
/// at this size, this often (every fourth frame of a film), or at once after a cut. That leaves
/// each run longer, so the limits for moving between sizes are higher.
const CARRIED_LEVEL: [usize; 2] = [1, 1];
const CARRIED_TOO_SLOW: f32 = 75.0;
const CARRIED_ROOM: f32 = 55.0;
const CARRIED_INTERVAL: Duration = Duration::from_millis(150);
/// Carrying is worst when things move fast, or the last run found it had gone wrong, and then
/// the model runs this often (every second frame). Fast is the speed nine tenths of the picture
/// is under, in fractions of the picture a second; gone wrong is how much depth (0–1) differed
/// on average. Both were chosen on film: about the worst tenth of moments.
const CARRIED_SOON: Duration = Duration::from_millis(70);
const CARRIED_FAST: f32 = 0.86;
const CARRIED_WRONG: f32 = 0.118;
/// The most (seconds) a map is carried by one report of movement.
const CARRIED_REACH: f32 = 0.5;
/// Maps waiting for the player to collect them.
const MOST_MAPS: usize = 6;

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Off,
    Starting,
    /// Downloading the model: percent done.
    Downloading(u8),
    Ready {
        /// Which network.
        model: &'static str,
        adapter: String,
        /// How long the last depth map took.
        ms: f32,
        /// The size the model is running at, in pixels.
        size: (usize, usize),
        /// Whether people, walls and sky are being found too.
        masks: bool,
        /// Whether maps are being carried along the movement between runs.
        carried: bool,
    },
    Failed(String),
}

impl Status {
    pub fn describe(&self) -> String {
        match self {
            Status::Off => "off".into(),
            Status::Starting => "starting…".into(),
            Status::Downloading(p) => format!("downloading the depth model ({p}%)…"),
            Status::Ready { model, adapter, ms, size, masks, carried } => format!("{model}: {ms:.0} ms a frame at {}×{}{}{} on {adapter}", size.0, size.1, if *masks { " with masks" } else { "" }, if *carried { ", carried by movement" } else { "" }),
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
    /// Ground, ceiling, water and plant; then window, skin, hair and clothes (from MoGe-2's head),
    /// likewise.
    pub more_masks: Option<[Vec<u8>; 2]>,
    /// Which way surfaces face (see `normal_bytes`), RGBA, bottom row first, `WIDTH` × `HEIGHT`,
    /// when the network gives them.
    pub normals: Option<Vec<u8>>,
    /// Where the ground is, as `ground::Ground::update` gives it.
    pub ground: [f32; 4],
    /// A guess at a frame that hasn't been seen yet, carried on along the last movement.
    pub ahead: bool,
    /// What can be told about the shot, when the model's just looked at the picture.
    pub clues: Option<crate::shot_type::Clues>,
}

struct State {
    status: Status,
    running: bool,
    /// A frame waiting: RGBA, bottom row first (as read back from GL), `WIDTH` × `HEIGHT`, its
    /// position, and whether there's been a scene cut since the last one handed over.
    input: Option<(Vec<u8>, f32, bool)>,
    busy: bool,
    /// Maps not collected yet, oldest first.
    output: VecDeque<Arc<DepthMap>>,
    /// The latest map of a frame that's really been seen, for the camera.
    newest: Option<Arc<DepthMap>>,
    last_wanted: Instant,
    /// Whether an effect wants people, walls and sky as well as depth.
    masks_wanted: bool,
    /// Whether to carry maps along the movement, and what's come in of it.
    carried: bool,
    moves: VecDeque<Report>,
    /// When carrying: when the model last started, and no frame before this unless there's been
    /// a cut.
    last_run: Option<Instant>,
    not_before: Option<Instant>,
    /// Into each `Kind`'s levels, kept so the next video starts at the size that worked for
    /// this one.
    level: [usize; 2],
}

/// Which network makes depth.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// MoGe-2: depth with the shape of things in it, and which way surfaces face.
    Moge = 0,
    /// Depth Anything V2 Small: quicker, and what people, walls and sky come from until
    /// MoGe-2 has its own.
    Small = 1,
}

impl Kind {
    /// MoGe-2, unless masks are wanted and it has no head for them; FLEX_DEPTH_MODEL=small or
    /// moge says otherwise.
    fn choose(masks: bool) -> Kind {
        match std::env::var("FLEX_DEPTH_MODEL").as_deref() {
            Ok("small") => Kind::Small,
            Ok("moge") => Kind::Moge,
            _ if masks && flex_depth::moge::MASK_HEAD.is_none() => Kind::Small,
            _ => Kind::Moge,
        }
    }

    fn levels(self) -> &'static [(usize, usize)] {
        match self {
            Kind::Moge => &MOGE_LEVELS,
            Kind::Small => &LEVELS,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Moge => "MoGe-2",
            Kind::Small => "Depth Anything V2",
        }
    }

    fn path(self) -> PathBuf {
        let file = match self {
            Kind::Moge => "moge-2-vits-normal.pt",
            Kind::Small => "depth-anything-v2-small.safetensors",
        };
        flex_core::data_dir().join("models").join(file)
    }

    /// Where it comes from: the URL, its SHA-256 and its size.
    fn source(self) -> (&'static str, &'static str, u64) {
        match self {
            Kind::Moge => (flex_depth::moge::WEIGHTS_URL, flex_depth::moge::WEIGHTS_SHA256, flex_depth::moge::WEIGHTS_BYTES),
            Kind::Small => (flex_depth::WEIGHTS_URL, flex_depth::WEIGHTS_SHA256, flex_depth::WEIGHTS_BYTES),
        }
    }
}

/// The network, loaded.
enum Net {
    Small(Depth),
    Moge(Moge),
}

/// What the network makes of a frame, at `WIDTH` × `HEIGHT`, rows from the top.
struct Found {
    /// Bigger is nearer, in no fixed units.
    disparity: Vec<f32>,
    masks: Option<Vec<Vec<f32>>>,
    /// Which way surfaces face: x to the right, y up, z towards the viewer.
    normals: Option<[Vec<f32>; 3]>,
}

impl Net {
    fn load(kind: Kind, grid: (usize, usize), masks: bool) -> anyhow::Result<Net> {
        Ok(match kind {
            Kind::Small => Net::Small(Depth::load(&kind.path(), grid, masks)?),
            Kind::Moge => Net::Moge(Moge::load(&kind.path(), grid, flex_depth::moge::MASK_HEAD.filter(|_| masks))?),
        })
    }

    fn size(&self) -> (usize, usize) {
        match self {
            Net::Small(d) => d.size(),
            Net::Moge(m) => m.size(),
        }
    }

    fn adapter(&self) -> &str {
        match self {
            Net::Small(d) => d.adapter(),
            Net::Moge(m) => m.adapter(),
        }
    }

    /// `rgb` is `WIDTH` × `HEIGHT`, rows from the top.
    fn run(&mut self, rgb: &[u8], masks: bool) -> anyhow::Result<Found> {
        let size = self.size();
        let input = if size == (WIDTH, HEIGHT) { std::borrow::Cow::Borrowed(rgb) } else { std::borrow::Cow::Owned(scaled_rgb(rgb, (WIDTH, HEIGHT), size)) };
        let full = |plane: &[f32], from: (usize, usize)| if from == (WIDTH, HEIGHT) { plane.to_vec() } else { scaled(plane, from, (WIDTH, HEIGHT)) };
        match self {
            Net::Small(d) => {
                let raw = d.run(&input)?;
                let masks = if masks { Some(mask_planes(&d.masks()?, None, size)) } else { None };
                Ok(Found { disparity: full(&raw, size), masks, normals: None })
            }
            Net::Moge(m) => {
                m.submit(&input)?;
                let g = m.geometry()?;
                let from = (g.width, g.height);
                let disparity: Vec<f32> = g.depth.iter().map(|d| 1.0 / d.max(1e-3)).collect();
                // MoGe's camera has y down and z away; effects have y up and z towards them.
                let normals = std::array::from_fn(|c| {
                    let sign = if c == 0 { 1.0 } else { -1.0 };
                    full(&g.normals.iter().map(|n| n[c] * sign).collect::<Vec<f32>>(), from)
                });
                let masks = if masks && m.has_masks() {
                    let found = m.masks()?;
                    Some(mask_planes(&found.scene, found.parts.as_deref(), from))
                } else {
                    None
                };
                Ok(Found { disparity: full(&disparity, from), masks, normals: Some(normals) })
            }
        }
    }
}

fn tokens(levels: &[(usize, usize)], level: usize) -> f32 {
    (levels[level].0 * levels[level].1 + 1) as f32
}

/// Chooses the size the model runs at from how long depth maps are taking. Nearly all of a
/// map's time goes with the number of patches, so that says how long another size would take.
struct Pace {
    levels: &'static [(usize, usize)],
    level: usize,
    average: Option<f32>,
    changed: Instant,
    retry: Option<Instant>,
}

impl Pace {
    fn new(levels: &'static [(usize, usize)], level: usize, now: Instant) -> Self {
        Self { levels, level: level.min(levels.len() - 1), average: None, changed: now, retry: None }
    }

    /// A map took `ms`, with maps `carried` along the movement or not. Returns the level to move
    /// to, if it's time to move.
    fn update(&mut self, ms: f32, now: Instant, carried: bool) -> Option<usize> {
        let (too_slow, room) = if carried { (CARRIED_TOO_SLOW, CARRIED_ROOM) } else { (TOO_SLOW, ROOM) };
        let average = self.average.map_or(ms, |a| a + (ms - a) * PACE_FOLLOW);
        self.average = Some(average);
        if now.duration_since(self.changed) < SETTLE {
            return None;
        }
        let levels = self.levels;
        let at = |level| average * tokens(levels, level) / tokens(levels, self.level);
        let level = if average > too_slow && self.level + 1 < levels.len() {
            // Straight to the size that should be fast enough, or the smallest.
            self.retry = Some(now + RETRY);
            (self.level + 1..levels.len()).find(|l| at(*l) <= too_slow).unwrap_or(levels.len() - 1)
        } else if self.level > 0 && at(self.level - 1) <= room && self.retry.is_none_or(|t| now >= t) {
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

/// Mask classes effects get, in this order: person, wall and sky (from either network), then
/// (from MoGe-2's head) ground, ceiling, water, plant, window, and a person's skin, hair and
/// clothes.
const MASKS: usize = 3;
const ALL_MASKS: usize = 11;

/// A network's planes (`size`, rows from the top, the first being "other") as the masks effects
/// get, at `WIDTH` × `HEIGHT`. `parts` are of a person, so they're scaled by how much of a
/// person each pixel is.
fn mask_planes(planes: &[f32], parts: Option<&[f32]>, size: (usize, usize)) -> Vec<Vec<f32>> {
    let pixels = size.0 * size.1;
    let full = |plane: &[f32]| if size == (WIDTH, HEIGHT) { plane.to_vec() } else { scaled(plane, size, (WIDTH, HEIGHT)) };
    let classes = planes.len() / pixels;
    let mut out: Vec<Vec<f32>> = (1..classes).map(|c| full(&planes[c * pixels..][..pixels])).collect();
    if let Some(parts) = parts {
        let person = &planes[pixels..][..pixels];
        for part in parts.chunks(pixels) {
            out.push(full(&part.iter().zip(person).map(|(a, b)| a * b).collect::<Vec<f32>>()));
        }
    }
    out
}

/// Four masks (rows from the top) as a texture's bytes: RGBA, bottom row first, with 1 for
/// any that aren't there.
fn mask_bytes(planes: &[Vec<f32>]) -> Vec<u8> {
    let mut out = vec![255u8; WIDTH * HEIGHT * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let px = &mut out[((HEIGHT - 1 - y) * WIDTH + x) * 4..][..4];
            for (byte, plane) in px.iter_mut().zip(planes) {
                *byte = (plane[y * WIDTH + x].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    out
}

/// Normals (rows from the top) as a texture's bytes: x, y, z from -1 to 1 as 0 to 255, RGBA,
/// bottom row first.
fn normal_bytes(planes: &[Vec<f32>; 3]) -> Vec<u8> {
    let mut out = vec![255u8; WIDTH * HEIGHT * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = y * WIDTH + x;
            let v = [planes[0][at], planes[1][at], planes[2][at]];
            let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
            let px = &mut out[((HEIGHT - 1 - y) * WIDTH + x) * 4..][..3];
            for (byte, c) in px.iter_mut().zip(v) {
                *byte = ((c / len * 0.5 + 0.5).clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
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

/// `plane` (`WIDTH` × `HEIGHT`) after `seconds` of movement, the movement being x and y in
/// fractions of the picture a second at that size, all rows from the top.
fn carried(plane: &[f32], flow: &[Vec<f32>; 2], seconds: f32) -> Vec<f32> {
    let (w, h) = (WIDTH as f32, HEIGHT as f32);
    let mut out = Vec::with_capacity(plane.len());
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let at = y * WIDTH + x;
            // What's here now was back along the movement.
            let from_x = (x as f32 - flow[0][at] * seconds * w).clamp(0.0, w - 1.0);
            let from_y = (y as f32 - flow[1][at] * seconds * h).clamp(0.0, h - 1.0);
            let (x0, y0) = (from_x as usize, from_y as usize);
            let (x1, y1) = ((x0 + 1).min(WIDTH - 1), (y0 + 1).min(HEIGHT - 1));
            let (tx, ty) = (from_x - x0 as f32, from_y - y0 as f32);
            let (r0, r1) = (&plane[y0 * WIDTH..][..WIDTH], &plane[y1 * WIDTH..][..WIDTH]);
            let top = r0[x0] + (r0[x1] - r0[x0]) * tx;
            let bottom = r1[x0] + (r1[x1] - r1[x0]) * tx;
            out.push(top + (bottom - top) * ty);
        }
    }
    out
}

/// What's known of the picture at one moment, rows from the top.
struct Picture {
    depth: Vec<f32>,
    masks: Option<Vec<Vec<f32>>>,
    normals: Option<[Vec<f32>; 3]>,
    time: f32,
}

impl Picture {
    fn moved(&self, flow: &[Vec<f32>; 2], seconds: f32) -> Picture {
        Picture {
            depth: carried(&self.depth, flow, seconds),
            masks: self.masks.as_ref().map(|masks| masks.iter().map(|m| carried(m, flow, seconds)).collect()),
            normals: self.normals.as_ref().map(|normals| std::array::from_fn(|c| carried(&normals[c], flow, seconds))),
            time: self.time + seconds,
        }
    }
}

/// The worker's latest picture, and how it's kept up with the movement between runs of the model.
#[derive(Default)]
struct Carrier {
    picture: Option<Picture>,
    /// The latest movement, at `WIDTH` × `HEIGHT`, while it follows on from what came before.
    flow: Option<[Vec<f32>; 2]>,
    last_report: Option<f32>,
    /// How fast things are moving, and how wrong the carried map was at the last run, this shot.
    fast: f32,
    wrong: f32,
    serial: u64,
    ground: [f32; 4],
}

/// The speed nine tenths of the picture is under, from planes of x then y.
pub fn fast_speed(flow: &[f32]) -> f32 {
    let (x, y) = flow.split_at(flow.len() / 2);
    let mut speeds: Vec<f32> = x.iter().zip(y).map(|(x, y)| x.hypot(*y)).collect();
    let at = speeds.len() * 9 / 10;
    *speeds.select_nth_unstable_by(at, f32::total_cmp).1
}

impl Carrier {
    /// How long there should be between runs of the model, by how carrying is going.
    fn interval(&self) -> Duration {
        if self.fast > CARRIED_FAST || self.wrong > CARRIED_WRONG { CARRIED_SOON } else { CARRIED_INTERVAL }
    }

    /// A new shot: nothing about the last one applies.
    fn cut(&mut self) {
        self.picture = None;
        self.fast = 0.0;
        self.wrong = 0.0;
    }

    fn map(&mut self, picture: &Picture, ahead: bool) -> DepthMap {
        self.serial += 1;
        DepthMap {
            // GL's rows run bottom-up.
            values: picture.depth.chunks(WIDTH).rev().flatten().copied().collect(),
            serial: self.serial,
            time: picture.time,
            masks: picture.masks.as_ref().map(|m| mask_bytes(&m[..MASKS])),
            more_masks: picture.masks.as_ref().filter(|m| m.len() == ALL_MASKS).map(|m| [mask_bytes(&m[3..7]), mask_bytes(&m[7..11])]),
            normals: picture.normals.as_ref().map(normal_bytes),
            ground: self.ground,
            ahead,
            clues: None,
        }
    }

    /// Takes in the movement in a new frame, and returns the maps that makes: that frame's, and
    /// a guess at the one after it, which won't have movement of its own.
    fn follow(&mut self, report: &Report) -> Vec<DepthMap> {
        let gap = self.last_report.map(|t| report.time - t).filter(|gap| (1e-4..CARRIED_REACH).contains(gap));
        self.last_report = Some(report.time);
        let Some(flow) = &report.flow else {
            self.flow = None;
            self.fast = 0.0;
            return Vec::new();
        };
        self.fast = fast_speed(flow);
        let (from, size) = ((FLOW_WIDTH, FLOW_HEIGHT), FLOW_WIDTH * FLOW_HEIGHT);
        let flow = [scaled(&flow[..size], from, (WIDTH, HEIGHT)), scaled(&flow[size..], from, (WIDTH, HEIGHT))];
        let mut maps = Vec::new();
        if let Some(picture) = self.picture.take() {
            let seconds = report.time - picture.time;
            if seconds <= 0.0 && seconds > -CARRIED_REACH {
                self.picture = Some(picture);
            } else if (0.0..CARRIED_REACH).contains(&seconds) {
                let now = picture.moved(&flow, seconds);
                maps.push(self.map(&now, false));
                if let Some(gap) = gap {
                    let next = now.moved(&flow, gap / 2.0);
                    maps.push(self.map(&next, true));
                }
                self.picture = Some(now);
            }
        }
        self.flow = Some(flow);
        maps
    }

    /// Takes in what the model found: brought to the same moment as what's known already, and
    /// settled against it. Returns the map that makes.
    fn found(&mut self, mut now: Picture) -> DepthMap {
        if let Some(mut before) = self.picture.take() {
            let ahead = before.time - now.time;
            if let Some(flow) = self.flow.as_ref().filter(|_| ahead.abs() < CARRIED_REACH) {
                if ahead > 0.0 {
                    now = now.moved(flow, ahead);
                } else if ahead < 0.0 {
                    before = before.moved(flow, -ahead);
                }
            }
            self.wrong = now.depth.iter().zip(&before.depth).map(|(a, b)| (a - b).abs()).sum::<f32>() / now.depth.len() as f32;
            settle(&mut now.depth, &before.depth, (WIDTH, HEIGHT));
            if let (Some(masks), Some(old)) = (&mut now.masks, &before.masks) {
                for (plane, old) in masks.iter_mut().zip(old) {
                    settle(plane, old, (WIDTH, HEIGHT));
                }
            }
            if let (Some(normals), Some(old)) = (&mut now.normals, &before.normals) {
                for (plane, old) in normals.iter_mut().zip(old) {
                    settle(plane, old, (WIDTH, HEIGHT));
                }
            }
        }
        let map = self.map(&now, false);
        self.picture = Some(now);
        map
    }
}

/// Shared between the UI, the paint callback and the worker.
#[derive(Clone)]
pub struct Link(Arc<(Mutex<State>, Condvar)>);

impl Default for Link {
    fn default() -> Self {
        let state = State { status: Status::Off, running: false, input: None, busy: false, output: VecDeque::new(), newest: None, last_wanted: Instant::now(), masks_wanted: false, carried: false, moves: VecDeque::new(), last_run: None, not_before: None, level: [0; 2] };
        Self(Arc::new((Mutex::new(state), Condvar::new())))
    }
}

impl Link {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.0.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// An effect wants depth, and maybe `masks` too: starts the worker if it isn't running.
    /// `carried` says the movement worker is to feed `carry`. Call every frame.
    pub fn wanted(&self, masks: bool, carried: bool) {
        let mut state = self.lock();
        state.last_wanted = Instant::now();
        state.masks_wanted = masks;
        state.carried = carried;
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

    /// Whether the worker would take a frame now. `cut`: there's been a scene cut since the last.
    pub fn ready_for_frame(&self, cut: bool) -> bool {
        let state = self.lock();
        // Even while busy: read back now, the next frame is ready the moment this one's done.
        matches!(state.status, Status::Ready { .. }) && state.input.is_none() && (cut || state.not_before.is_none_or(|t| Instant::now() >= t))
    }

    /// Hands over a frame (RGBA, bottom row first, `WIDTH` × `HEIGHT`), its position, and whether
    /// there's been a scene cut since the last frame handed over.
    pub fn submit(&self, rgba: Vec<u8>, time: f32, cut: bool) {
        self.lock().input = Some((rgba, time, cut));
        self.0.1.notify_all();
    }

    /// The movement in another frame, to carry the maps along.
    pub fn carry(&self, report: Report) {
        let mut state = self.lock();
        state.moves.push_back(report);
        let extra = state.moves.len().saturating_sub(MOST_MAPS);
        state.moves.drain(..extra);
        self.0.1.notify_all();
    }

    /// The latest map of a frame that's really been seen.
    pub fn newest_map(&self) -> Option<Arc<DepthMap>> {
        self.lock().newest.clone()
    }

    /// The maps that have come in since the last call, oldest first.
    pub fn take_maps(&self) -> Vec<Arc<DepthMap>> {
        self.lock().output.drain(..).collect()
    }

    fn publish(&self, maps: impl IntoIterator<Item = DepthMap>) {
        let mut state = self.lock();
        for map in maps {
            let map = Arc::new(map);
            if !map.ahead {
                state.newest = Some(map.clone());
            }
            state.output.push_back(map);
        }
        let extra = state.output.len().saturating_sub(MOST_MAPS);
        state.output.drain(..extra);
    }

    /// Carries the picture along whatever movement has come in.
    fn catch_up(&self, carrier: &mut Carrier) {
        let moves: Vec<Report> = self.lock().moves.drain(..).collect();
        for report in &moves {
            self.publish(carrier.follow(report));
        }
        if !moves.is_empty() {
            self.plan_next_run(carrier);
        }
    }

    /// When carrying, works out when the model should next run, from the last time it did.
    fn plan_next_run(&self, carrier: &Carrier) {
        let mut state = self.lock();
        state.not_before = state.last_run.filter(|_| state.carried).map(|last| last + crate::gpu::rest(crate::gpu::Net::Depth, carrier.interval()));
    }

    /// Downloads `kind` if it isn't here yet (showing how far it's got), and loads it.
    fn load(&self, kind: Kind, grid: (usize, usize), masks: bool) -> anyhow::Result<Net> {
        let path = kind.path();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let (url, sha, total) = kind.source();
        let progress = |done: u64| {
            let pct = (done * 100 / total.max(1)).min(100) as u8;
            let mut state = self.lock();
            if state.status != Status::Downloading(pct) && done < total {
                state.status = Status::Downloading(pct);
            }
        };
        flex_core::update::download_verified(url, "the depth model", total, sha, &path, progress)?;
        Net::load(kind, grid, masks)
    }

    fn work(&self) {
        let (mut with_masks, carried) = {
            let state = self.lock();
            (state.masks_wanted, state.carried)
        };
        let mut kind = Kind::choose(with_masks);
        let level = {
            let state = self.lock();
            if carried { state.level[kind as usize].max(CARRIED_LEVEL[kind as usize]) } else { state.level[kind as usize] }
        };
        let mut pace = Pace::new(kind.levels(), level, Instant::now());
        let mut net = match self.load(kind, kind.levels()[pace.level], with_masks) {
            Ok(net) => net,
            Err(e) => {
                log::warn!("depth: {e:#}");
                let mut state = self.lock();
                state.status = Status::Failed(format!("{e:#}"));
                state.running = false;
                return;
            }
        };
        log::info!("depth: {} running on {}", kind.name(), net.adapter());
        self.lock().status = Status::Ready { model: kind.name(), adapter: net.adapter().to_owned(), ms: 0.0, size: net.size(), masks: with_masks, carried: false };

        let mut carrier = Carrier::default();
        let mut range: Option<(f32, f32)> = None;
        let mut ground = Ground::default();
        let mut rgb = vec![0u8; WIDTH * HEIGHT * 3];
        loop {
            let input = loop {
                self.catch_up(&mut carrier);
                let mut state = self.lock();
                if !state.moves.is_empty() {
                    continue;
                }
                if let Some(input) = state.input.take() {
                    state.busy = true;
                    state.last_run = Some(Instant::now());
                    drop(state);
                    self.plan_next_run(&carrier);
                    break Some(input);
                }
                if state.last_wanted.elapsed() > IDLE {
                    break None;
                }
                let _ = self.0.1.wait_timeout(state, Duration::from_millis(500)).unwrap_or_else(|e| e.into_inner());
            };
            let Some((rgba, time, cut)) = input else { break };
            // GL's rows run bottom-up; the model's top-down.
            for y in 0..HEIGHT {
                let src = &rgba[(HEIGHT - 1 - y) * WIDTH * 4..][..WIDTH * 4];
                for (x, px) in src.as_chunks::<4>().0.iter().enumerate() {
                    rgb[(y * WIDTH + x) * 3..][..3].copy_from_slice(&px[..3]);
                }
            }
            let started = Instant::now();
            let size = net.size();
            let turn = crate::gpu::turn(crate::gpu::Net::Depth);
            let result = net.run(&rgb, with_masks);
            let ms = turn.ms();
            drop(turn);
            let found = match result {
                Ok(found) => found,
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
            let raw = found.disparity;
            let (lo, hi) = raw.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            // A new shot has nothing to do with the last one: start again from this picture.
            if cut {
                carrier.cut();
                range = None;
                ground.forget();
            }
            let (lo, hi) = match range {
                Some((l, h)) => (l + (lo - l) * RANGE_FOLLOW, h + (hi - h) * RANGE_FOLLOW),
                None => (lo, hi),
            };
            range = Some((lo, hi));
            let scale = 1.0 / (hi - lo).max(1e-6);
            let values: Vec<f32> = raw.iter().map(|v| ((v - lo) * scale).clamp(0.0, 1.0)).collect();
            // What moved while the model was running, so its answer goes in beside the latest.
            self.catch_up(&mut carrier);
            let mut map = carrier.found(Picture { depth: values, masks: found.masks, normals: found.normals, time });
            map.ground = ground.update(&map.values, map.masks.as_deref(), (WIDTH, HEIGHT));
            carrier.ground = map.ground;
            map.clues = carrier.picture.as_ref().and_then(|p| p.masks.as_ref().map(|masks| crate::shot_type::Clues::of(&p.depth, &masks[0], &masks[2], &rgb, (WIDTH, HEIGHT))));
            self.plan_next_run(&carrier);
            let carried = {
                let mut state = self.lock();
                let carried = state.carried;
                state.status = Status::Ready { model: kind.name(), adapter: net.adapter().to_owned(), ms, size, masks: with_masks, carried };
                carried
            };
            self.publish([map]);
            // Still "busy" while waiting, so no frame is read back in the meantime. When carrying,
            // the wait is `not_before`, as the movement needs following in the meantime.
            if !carried {
                std::thread::sleep(crate::gpu::rest(crate::gpu::Net::Depth, MIN_INTERVAL).saturating_sub(started.elapsed()));
            }
            let moved = pace.update(ms, Instant::now(), carried);
            let masks_wanted = self.lock().masks_wanted;
            let wanted_kind = Kind::choose(masks_wanted);
            if moved.is_some() || masks_wanted != with_masks || wanted_kind != kind {
                if wanted_kind != kind {
                    kind = wanted_kind;
                    let level = self.lock().level[kind as usize];
                    pace = Pace::new(kind.levels(), if carried { level.max(CARRIED_LEVEL[kind as usize]) } else { level }, Instant::now());
                    // Different networks' depth and normals don't follow on from each other.
                    carrier.cut();
                    range = None;
                }
                let level = pace.level;
                let grid = kind.levels()[level];
                log::info!("depth: {ms:.0} ms a frame; now {} at {} × {} patches, masks {masks_wanted}", kind.name(), grid.0, grid.1);
                self.lock().level[kind as usize] = level;
                with_masks = masks_wanted;
                if let Some(picture) = &mut carrier.picture {
                    picture.masks = None;
                }
                // The old one goes first: a slow GPU may not have room for both.
                drop(net);
                net = match self.load(kind, grid, with_masks) {
                    Ok(net) => net,
                    Err(e) => {
                        log::warn!("depth: {e:#}");
                        let mut state = self.lock();
                        state.status = Status::Failed(format!("{e:#}"));
                        state.running = false;
                        state.busy = false;
                        state.output.clear();
                        state.newest = None;
                        return;
                    }
                };
            }
            self.lock().busy = false;
        }
        drop(net);
        log::info!("depth: stopped");
        let mut state = self.lock();
        state.running = false;
        state.output.clear();
        state.newest = None;
        state.moves.clear();
        state.last_run = None;
        state.not_before = None;
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

    /// A picture dark but for one bright pixel, and a report moving everything evenly.
    fn dot(x: usize, y: usize, time: f32) -> Picture {
        let mut depth = vec![0.0; WIDTH * HEIGHT];
        depth[y * WIDTH + x] = 1.0;
        Picture { depth, masks: None, normals: None, time }
    }

    fn report(time: f32, vx: f32, vy: f32) -> Report {
        let size = FLOW_WIDTH * FLOW_HEIGHT;
        Report { time, flow: Some([vec![vx; size], vec![vy; size]].concat()) }
    }

    fn brightest(values: &[f32]) -> (usize, usize) {
        let at = values.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        (at % WIDTH, at / WIDTH)
    }

    #[test]
    fn maps_are_carried_the_way_things_move() {
        let mut carrier = Carrier { picture: Some(dot(100, 200, 1.0)), last_report: Some(0.9), ..Default::default() };
        // Across the picture in a second and down it twice, for a tenth of a second.
        let (vx, vy) = (1.0, 2.0);
        let maps = carrier.follow(&report(1.1, vx, vy));
        let (w, h) = (WIDTH as f32, HEIGHT as f32);
        let moved = ((100.0 + vx * 0.1 * w).round() as usize, (200.0 + vy * 0.1 * h).round() as usize);
        assert_eq!(maps.len(), 2);
        assert_eq!(brightest(&carrier.picture.as_ref().unwrap().depth), moved);
        // What's published has GL's rows, bottom first.
        assert_eq!(brightest(&maps[0].values), (moved.0, HEIGHT - 1 - moved.1));
        assert!((maps[0].time - 1.1).abs() < 1e-4);
        assert!(!maps[0].ahead);
        // And half way on to the next one, as a guess.
        assert!(maps[1].ahead);
        assert!((maps[1].time - 1.2).abs() < 1e-4);
    }

    #[test]
    fn the_model_runs_sooner_when_things_move_fast() {
        let mut carrier = Carrier { last_report: Some(0.9), ..Default::default() };
        carrier.follow(&report(1.0, 0.1, 0.0));
        assert_eq!(carrier.interval(), CARRIED_INTERVAL);
        carrier.follow(&report(1.1, CARRIED_FAST + 0.1, 0.0));
        assert_eq!(carrier.interval(), CARRIED_SOON);
        carrier.cut();
        assert_eq!(carrier.interval(), CARRIED_INTERVAL);
    }

    #[test]
    fn the_model_runs_sooner_when_the_carried_map_was_wrong() {
        let mut carrier = Carrier { picture: Some(Picture { depth: vec![0.2; WIDTH * HEIGHT], masks: None, normals: None, time: 1.0 }), ..Default::default() };
        carrier.found(Picture { depth: vec![0.2 + CARRIED_WRONG + 0.05; WIDTH * HEIGHT], masks: None, normals: None, time: 1.0 });
        assert_eq!(carrier.interval(), CARRIED_SOON);
    }

    #[test]
    fn a_new_shot_isnt_carried_but_the_picture_is_kept_for_its_cut() {
        let mut carrier = Carrier { picture: Some(dot(10, 10, 1.0)), last_report: Some(0.9), ..Default::default() };
        assert!(carrier.follow(&Report { time: 1.1, flow: None }).is_empty());
        assert!(carrier.flow.is_none());
        assert_eq!(brightest(&carrier.picture.unwrap().depth), (10, 10));
    }

    #[test]
    fn a_found_map_is_brought_to_the_moment_of_the_carried_one() {
        let mut carrier = Carrier::default();
        carrier.follow(&report(1.0, 1.0, 0.0));
        carrier.picture = Some(dot(100, 100, 1.1));
        // Found for 0.1 s earlier: it should land on top of the dot, not behind it.
        let x = 100 - (0.1 * WIDTH as f32).round() as usize;
        let map = carrier.found(dot(x, 100, 1.0));
        assert_eq!(brightest(&map.values), (100, HEIGHT - 1 - 100));
        assert!((map.time - 1.1).abs() < 1e-4);
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
        let mut pace = Pace::new(&LEVELS, 0, now);
        assert_eq!(pace.update(200.0, now + Duration::from_secs(1), false), None);
        // 200 ms at the biggest size: only the smallest should come in under the limit.
        assert_eq!(pace.update(200.0, now + SETTLE, false), Some(LEVELS.len() - 1));
        assert_eq!(pace.level, LEVELS.len() - 1);
    }

    #[test]
    fn pace_waits_before_retrying_a_size_that_was_too_slow() {
        let now = Instant::now();
        let mut pace = Pace::new(&LEVELS, 0, now);
        assert_eq!(pace.update(TOO_SLOW + 1.0, now + SETTLE, false), Some(1));
        // Fast enough to go back up, by the estimate, but it only just failed.
        let t = now + SETTLE * 3;
        assert_eq!(pace.update(10.0, t, false), None);
        assert_eq!(pace.update(10.0, now + SETTLE + RETRY, false), Some(0));
    }

    #[test]
    fn carried_maps_can_take_longer_as_they_come_less_often() {
        let now = Instant::now();
        let mut pace = Pace::new(&LEVELS, CARRIED_LEVEL[1], now);
        assert_eq!(pace.update(TOO_SLOW + 10.0, now + SETTLE, true), None);
        let mut pace = Pace::new(&LEVELS, CARRIED_LEVEL[1], now);
        assert!(pace.update(CARRIED_TOO_SLOW + 10.0, now + SETTLE, true).is_some_and(|l| l > CARRIED_LEVEL[1]));
    }

    #[test]
    fn pace_stays_put_between_too_slow_and_room() {
        let now = Instant::now();
        let mut pace = Pace::new(&LEVELS, 1, now);
        // Fine here; the size above would be slower than ROOM.
        let ms = ROOM * tokens(&LEVELS, 1) / tokens(&LEVELS, 0) + 5.0;
        assert_eq!(pace.update(ms, now + SETTLE, false), None);
    }

    #[test]
    fn masks_skip_other_and_go_bottom_row_first() {
        // 4 classes of 2 × 1 pictures would need scaling; use the full size for the layout.
        let pixels = WIDTH * HEIGHT;
        let mut planes = vec![0.0; 4 * pixels];
        planes[pixels] = 1.0; // person, at the top left
        let bytes = mask_bytes(&mask_planes(&planes, None, (WIDTH, HEIGHT)));
        assert_eq!(&bytes[(HEIGHT - 1) * WIDTH * 4..][..4], &[255, 0, 0, 255]);
        assert_eq!(&bytes[..4], &[0, 0, 0, 255]);
    }
}
