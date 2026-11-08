# Video effects

Flex can draw a shader over the video while it plays: kaleidoscopes, melting trails,
dreamy swirls, anything you can write as a GLSL fragment shader. Effects run on the GPU,
can react to the sound, and reload as soon as you save the file, so you can write one
while watching it.

Hold OK (or right-click) while a video plays to open the effects panel beside the picture.
Up and Down pick a row, Left and Right change it, and Back closes the panel: switch effect
and adjust its settings while you watch. They're also in Settings → Picture → Video effects.

Turn on Shuffle in the panel and effects take turns at random, each staying for a random time
between the shortest and longest you set, then fading into the next. Every effect plays once
before any comes round again, and the clock stops while the video's paused. "In the shuffle"
leaves out the effect you're on. Shuffle keeps sound, depth, masks and faces running for
whatever it might pick next, so that nothing stutters at the changes; leave out the effects
that use them if the GPU struggles.

Flex comes with these to start from, in [`effects`](effects):

- **Melt**: a turning kaleidoscope with glowing trails that change colour; beats kick it.
- **Dream**: a DeepDream-like feedback loop where edges curl into tendrils, bright
  patches blossom and details sprout texture.
- **Depth**: uses a neural network's guess at how far away everything is, for rainbow
  contours rippling through the scene, or fog.
- **Distance**: an effect that grows with distance, or closeness: fog, prisms, heat haze,
  blocks, colour or a rush of speed.
- **Relight**: new coloured lights in the film's world, casting shadows, using a surface
  worked out from depth. Also clay and an oil slick.
- **Diorama**: moves the camera inside the shot: floating, a dolly-zoom, or red and cyan 3D.
- **Rubber Sheet**: stretches the film over a wobbling mesh shaped by depth: waves, ripples,
  facets or wireframe.
- **Undertow**: the background melts and pours away while whoever's up close stays solid.
- **Ego Death**: people leave rainbow ghosts of themselves that spiral away.
- **Sky Eater**: the sky opens into a spiral that endlessly swallows the rest of the film.
- **Breathing Walls**: walls swell and sink and crawl with colour that follows the room.
- **Third Eye**: everyone grows a real eye in the middle of their forehead.
- **Wax Museum**: faces soften like candle wax and run down people's necks.
- **Hall of Faces**: each face sprouts a turning ring of copies of itself.
- **Liquid Mercury**: the whole world turns to rippling liquid metal.
- **Time Smear**: time runs slower the further away something is.
- **Mind Bloom**: a kaleidoscope that grows more arms with the bass and blooms into its own
  feedback.
- **Masks**: shows what Flex can tell apart in the picture: people, walls and sky.
- **Faces**: shows what Flex can see of the faces in the picture, for checking and for
  making effects.
- **Spotlight**: faces stay as they are while everything around them melts and dims.
- **Laser Eyes**: a beam from each eye, going the way the face is pointing and flaring on the
  beat.
- **Big Heads**: everyone's head swells up on top of a normal-sized body and pulses with the
  sound.
- **Soundstage**: sounds light up what's closest and roll back through the scene.

## Making one

An effect is a `.frag` file in an `effects` folder, plus an optional `.toml` file with
the same name:

```
effects/
  wobble.frag
  wobble.toml     (optional)
```

Flex looks in `effects` next to `flex.exe` (the ones it ships with), then in `effects`
in Flex's data folder (`%APPDATA%\Flex\data\effects` on Windows,
`~/Library/Application Support/Flex/effects` on a Mac; "Open effects folder" in Settings
takes you there). A file in the data folder wins over one with the same name next to
Flex, so copy `melt.frag` and `melt.toml` there to change them. After adding one, use
"Look for new effects".

Development builds (`cargo run`) also look in the source folder's `effects`, so edits
there show up straight away.

### The shader

Write one function, which gives the colour at `uv`:

```glsl
vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    return vec4(c.gbr, 1.0);
}
```

It's GLSL 1.40 / 3.30 (or ES 3.00 where that's all there is). Flex adds these, so
don't declare them yourself:

| Uniform | What |
|---|---|
| `sampler2D u_video` | the video frame as mpv drew it (with subtitles, and Anime4K or SVP if they're on) |
| `sampler2D u_prev` | the effect's own previous output (black unless `feedback = true`) |
| `vec2 u_resolution` | the size being drawn, in pixels |
| `float u_time` | seconds since the effect started; keeps going while paused |
| `float u_video_time` | the playback position in seconds: stops when paused, jumps when you seek |
| `int u_frame` | frames drawn since the effect started |
| `vec4 u_video_rect` | where the picture is without its black bars: `x0, y0, x1, y1` |
| `float s_<key>` | each `number`, `bool` (0 or 1) or `choice` (the option's index) setting |
| `float u_level`, `u_bass`, `u_mid`, `u_treble` | how loud the sound is overall, below 150 Hz, 150 Hz–2 kHz and above 2 kHz (0–1, with `audio = true`) |
| `float u_beat` | 1 on a beat (a jump in the bass), fading to 0 in about half a second |
| `sampler2D u_spectrum` | the sound's spectrum: x is frequency (30 Hz to 11 kHz, log-spaced, 64 bands), y is age (0 is now, 1 is about two seconds ago); values 0–1 in `.r` |
| `float depth_at(vec2 uv)` | a function: how near that point of the screen is, from 0 (far) to 1 (near) (with `depth = true`; 0 until the first depth map arrives) |
| `sampler2D u_depth`, `float u_has_depth` | the depth map itself, covering the picture rather than the screen, and 1 once there is one |
| `float person_at(vec2 uv)`, `wall_at`, `sky_at` | functions: how much of that point of the screen is a person, a wall or sky, from 0 to 1 (with `segment = true`; 0 until the first masks arrive). `vec4 masks_at(vec2 uv)` gives all three in r, g and b |
| `int u_face_count`, `vec4 u_faces[MAX_FACES]` | how far through the arrays to look, and each face's box on screen: x0, y0, x1, y1 (with `faces = true`) |
| `float u_face_seen[MAX_FACES]` | 0 to 1, fading as a face comes and goes; multiply by it so faces don't pop. It's 0 for empty slots |
| `vec2 face_point(int face, int point)` | where on the screen a face's point is: 0 right eye, 1 left eye, 2 nose, 3 right corner of the mouth, 4 left (the person's own right and left) |
| `float face_at(vec2 uv)` | a function: how much of any face is at that point of the screen, from 0 to 1, soft at the edges |

`uv` and `u_video_rect` run from 0 to 1, starting at the bottom left. Effects cover the
whole screen, black bars included, so they can draw into them.

Use `u_video_time` for anything that should freeze when the video's paused, and `u_time`
for things that should keep drifting. Go easy on big sudden changes driven by `u_beat`:
whole-screen flashes are hard on the eyes.

### The .toml

```toml
name = "Wobble"                # shown in Settings; defaults to the file's name
description = "Makes it wobble"
feedback = true                # give the shader its own last frame (u_prev); default false
audio = true                   # the sound uniforms; default false
depth = true                   # how far away things are; default false
segment = true                 # where people, walls and sky are; default false
faces = true                   # where faces are; default false
scale = 1.0                    # work at this fraction of the screen's size (0.25–1), then
                               # scale up: lower is lighter on the GPU; default 1
shuffle = false                # leave it out of shuffle unless someone adds it; default true

[[settings]]
key = "speed"                  # becomes `uniform float s_speed;`
label = "Speed"
type = "number"                # number, bool or choice
detail = "How fast it wobbles" # help under the setting
min = 0
max = 2
step = 0.25
unit = "×"
default = 1
```

Settings show under the effect in the panel and in Settings → Picture → Video effects, and
are kept per effect. A `choice` setting takes `options = [{ id = "soft", label = "Soft" }, ...]`, and
the shader gets the chosen option's position (0, 1, ...).

### Sound

With `audio = true`, a filter inside mpv measures how loud each of 32 frequency bands is, so
it works for anything Flex can play, YouTube included, and nothing reads the file a second
time. It reports about 5 to 10 times a second and values ease between reports, so sound
moves things a little less sharply than the picture changes, and very short sounds can be
missed. `u_spectrum` is stretched to its 64 bands from those 32. Levels are against the
loudest sound of the last few seconds, so quiet scenes still move things and explosions
don't max everything out for long.

### Depth

With `depth = true`, Flex works out how far away everything in the picture is, using
[Depth Anything V2 Small](https://github.com/DepthAnything/Depth-Anything-V2) (Apache-2.0)
running on the GPU as compute shaders, built on the kernels from Maisy's
[neurofield](https://github.com/bombpersons/neurofield). The first time an effect asks for
it Flex downloads the model (99 MB) into its data folder.

Depth is relative rather than in real distances, and is worked out up to 30 times a second
at 518 × 294, dropping to as little as 294 × 168 to keep each one under about 35 ms. It's
softer than the picture, and smoothed over time so it doesn't flicker, except where
something moved. The model takes a few frames to answer, so while an effect uses depth Flex
holds the picture and the sound back by that much, usually under a tenth of a second, and
each frame is drawn with the depth map made from it. It uses a fair bit of the GPU, so leave
it off unless the effect uses it; the model is unloaded a little while after the last effect
that wanted it. F2 and F3 show how long it's taking and at what size.

### People, walls and sky

With `segment = true`, a small network that runs on top of the depth model tells people,
walls and sky apart, so an effect can use them alongside `depth_at` without paying for depth
twice (it costs depth's time plus about a fifth). It was trained on COCO and ADE20K. People
and sky are solid, including in animation; walls are the weakest, so anything flat and dark
behind someone can be called a wall, and their edges are worth treating as soft.

### Faces

With `faces = true`, Flex finds up to eight faces with
[YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT), a
small network that runs on the GPU like depth does and comes with Flex. It's cheap, and looks
at up to about 30 frames a second. Each frame is drawn with the faces found in it: as with
depth, the picture and sound are held back a moment, usually 0.15 of a second or so, which
also lets Flex steady each face using the frames either side. A face keeps its slot in the arrays while it's on screen. It works best on
live action; very small faces, faces in profile or turned away, and cartoon characters are
missed some of the time, so effects need to look fine with no faces at all.

## Testing

- While a video plays, Flex checks the shader twice a second and reloads it when it
  changes. If it doesn't compile, the video plays without the effect, Settings → Picture
  → Video effects shows the first error, and the whole compiler output goes to Flex's
  log (Settings → "Open logs folder").
- F3 while a video plays opens a window for choosing the effect and changing its
  settings with sliders while you watch (plus Anime4K and SVP), with meters for the
  sound levels. Changes are kept, the same as in Settings.
- F2 shows which effect is on, the time it's given, and the sound levels.
- `flex --windowed some-video.mkv` starts Flex straight into a video.
