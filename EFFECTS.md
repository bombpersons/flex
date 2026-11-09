# Video effects

Flex can draw a shader over the video while it plays: kaleidoscopes, melting trails,
dreamy swirls, anything you can write as a GLSL fragment shader. Effects run on the GPU,
can react to the sound, and reload as soon as you save the file, so you can write one
while watching it.

Hold OK (or right-click) while a video plays to open the effects panel beside the picture.
Up and Down pick a row, Left and Right change it, OK opens the rows with an arrow, and Back goes
out a level and closes the panel. The cursor starts on the Effects row, so holding OK and then
pressing OK turns effects off, and doing it again brings back the same effect, shuffle and drift.
They're also in Settings → Picture → Video effects.

OK on the Effect row opens the list of effects, and whichever one you rest on shows behind it.
OK keeps it and Back returns to the one you had. Left and Right change section: All, Favourites,
and the effects that use people, faces, depth or the sound. Holding OK (or right-clicking) stars
an effect as a favourite, or with shuffle on ticks it in or out of the shuffle. Left and Right on
the Effect row move to the previous or next effect without opening the list.

Turn up Drift in the panel and every effect's settings wander slowly around where they're set.
The bars move as they go. Things that change at a rate, like Speed, stay put. OK on the row sets
how quickly.

Set Shuffle to Timed in the panel and effects take turns at random, each staying for a random time
between the shortest and longest you set, then fading into the next. Every effect plays once
before any comes round again, and the clock stops while the video's paused or the list of effects
is open. Set it to Scene cuts and the next effect starts on the frame the shot changes instead,
once the shortest turn is up. OK on the row opens the rest: Fade sets how long the change takes,
and Off makes it instant. Random settings moves an effect's settings at random each time it comes
up, without changing the ones you've set. Shuffle keeps sound, depth, masks and faces running for
whatever it might pick next, so that nothing stutters at the changes; untick the effects that use
them if the GPU struggles.

Flex comes with these to start from, in [`effects`](effects):

- **Kaleidoscope**: every part of a kaleidoscope on a slider: arms, nested folds, tiling, tunnels,
  which part of the film it sees, colour and trails.
- **Melt**: a turning kaleidoscope with glowing trails that change colour; beats kick it.
- **Dream**: a DeepDream-like feedback loop where edges curl into tendrils, bright
  patches blossom and details sprout texture.
- **Depth**: uses a neural network's guess at how far away everything is, for rainbow
  contours rippling through the scene, or fog.
- **Distance**: an effect that grows with distance, or closeness: fog (in any two colours), prisms, heat haze,
  blocks, colour or a rush of speed.
- **Relight**: new coloured lights in the film's world, casting shadows, using a surface
  worked out from depth. Also clay and an oil slick.
- **Diorama**: moves the camera inside the shot: floating, a dolly-zoom, or red and cyan 3D.
- **Rubber Sheet**: stretches the film over a wobbling mesh shaped by depth: waves, ripples,
  facets or wireframe.
- **Undertow**: the background melts and pours away while whoever's up close stays solid.
- **Ego Death**: people leave rainbow ghosts of themselves that spiral away.
- **Sky Eater**: the sky opens into a spiral that endlessly swallows the rest of the film.
- **Breathing Walls**: the room is inside something alive: walls swell and sink with each breath, buckling into wet folds and throbbing veins.
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
- **Face Mesh**: shows what Flex can see of a face up close: its mesh, which way the head points,
  and how open the eyes and mouth are, for checking and for working on an effect.
- **Mouth Portal**: open mouths are holes into a swirling galaxy.
- **War Paint**: glowing tribal markings painted on every face, flaring on the beat.
- **Cyborg**: half of every face is torn away to show steel plates, running circuits and a red eye.
- **Porcelain Dolls**: faces are glazed china with painted lips, and they crack as a shot goes on.
- **X-Ray**: a scanner sweeps across the picture, turning faces into skulls that talk when they do.
- **Drive-In**: everyone's skin is a screen showing the film, on and on forever.
- **Follow Your Nose**: a beam of light comes out of everyone's nose, wherever their head is
  pointing.
- **Spotlight**: faces stay as they are while everything around them melts and dims.
- **Laser Eyes**: a beam from each eye, going the way the face is pointing and flaring on the
  beat.
- **Big Heads**: everyone's head swells up on top of a normal-sized body and pulses with the
  sound.
- **Upside-Down Face**: eyes and mouth turned over inside each face, or the whole face.
- **Mouth Eyes**: eyes and mouth swap places.
- **Cyclops**: everyone's eyes merge into one big one.
- **Pinocchio**: noses grow for as long as a shot lasts and snap back at the cut.
- **Silhouette Cosmos**: people become person-shaped windows into a drifting galaxy, or stand
  in front of one.
- **Invisible Man**: people vanish and the room behind them is filled in from what's been seen.
- **Clone Army**: rows of the same people stretch away behind them.
- **Wallpaper Crawl**: walls writhe, and eyes open in them and watch the people.
- **Aquarium**: the room really floods: a water surface lying level with the floor or table, that wraps around people and furniture,
  with reflections, wakes, fish, and bubbles from people's mouths.
- **Snow Globe**: snow falls behind whatever's near and settles on things; a beat shakes it off.
- **Thermal Vision**: heat colours, with people and faces hottest.
- **Night of the Living**: grey-green people with hollow eyes, under a flickering light.
- **Black Hole**: a real black hole hanging in the room: light bends round it into a ring, people pass in front of it, and its disc lights the walls.
- **Melting Clocks**: everything but people sags and drips over its edges.
- **Fractal Zoom**: an endless fall through copies of the live picture, centred on a face.
- **Oil on Water**: rainbow bands like petrol on a puddle, curling around people.
- **Reaction-Diffusion Skin**: animal markings that grow by themselves over every surface, wrapping round shapes and lining up along the depth of the room.
- **Datamosh**: the wrong frames get dragged around by the right movement; the beat cleans it.
- **Pixel Sort Waterfall**: bright pixels pour down the screen and pool at the bottom.
- **Colour Time Split**: red, green and blue run at different moments.
- **Inkblot**: the picture folds into a mirrored Rorschach blot that keeps changing.
- **Strobe Sculpture**: people leave rising, glowing statues of themselves on the beat.
- **Aura**: a coloured energy field streams off each person, pulsing with the bass.
- **Living Static**: people are made of TV noise, with their faces flickering through.
- **Melting People**: people sag and drip, running off in streams that pool at the bottom.
- **Smoke People**: people come apart at the edges and blow away as smoke.
- **Pixel People**: people turn to chunky pixels, which change size on the beat.
- **Ink Drawing People**: people become a living pen sketch in a full-colour world.
- **Comic Book People**: people are printed in halftone dots with thick outlines, and get speech bubbles while they talk.
- **Fire People**: people burn from the feet up, with flames that follow them.
- **Tiny People**: people shrink where they are, leaving the room huge.
- **Time-Lagged Body**: the tops of people run behind their bottoms, so they ripple as they move.
- **Colour Thief**: the world is grey and only people have colour, or the other way round.
- **Glitch Bodies**: people tear into sliding strips on each beat.
- **Swarm**: people dissolve into buzzing specks that scatter with the bass.
- **Tiny Heads**: everyone's head shrinks to a pea on a full-sized body.
- **Googly Eyes**: plastic eyes whose pupils really roll and bounce as heads move.
- **Face Swap Roulette**: everyone's faces swap places, and swap again every few seconds.
- **Everyone Is Them**: the biggest face in the shot gets pasted onto everybody else.
- **Long Face**: faces stretch like toffee, chins sagging with the bass.
- **Clown Car**: greasepaint, a big red smile and a red nose that swells on the beat.
- **Deal With It**: pixel sunglasses drop out of the sky onto every face.
- **Blush Meter**: faces go redder the louder it gets, until steam comes out of their ears.
- **Witness Protection**: a black bar over the eyes, faces pixelated or blurred, or no features at all.
- **Moustache Everyone**: a curly handlebar moustache under every nose.
- **Love Beams**: hearts stream out of people's eyes the way they're facing.
- **Kaleidoface**: a kaleidoscope of someone's face, centred on their nose as they move.
- **Anime Eyes**: big drawn anime eyes, inked and shining, that blink when the real ones do, look where they look and sparkle on the beat.
- **Hypno Eyes**: eyes turn into spinning spirals.
- **Perfect Symmetry**: each face is one half mirrored, which is more unsettling than you'd think.
- **Spinning Features**: eyes and mouths turn round and round where they are.
- **Puppet Jaw**: jaws drop open and flap with the sound or with their own mouths, like a cut-out puppet.
- **Giraffe Necks**: heads rise on long stretchy necks that sway to the music.
- **Focus Pull**: a shallow film lens drifting between near and far, or held on a face.
- **Tiny World**: a thin slice in focus and toy-bright colours make everything a model railway.
- **Lidar Sweep**: a dark scene revealed by sweeps of laser light travelling away from the camera.
- **Point Cloud**: the scene as millions of floating dots, with the camera drifting around them.
- **Topographic Map**: the scene as a contour map, with hills where things are close.
- **Torchlight**: a dark room with a torch beam sweeping around it.
- **Magic Eye**: the film as a working stereogram: relax your eyes and it stands out in 3D.
- **Paper Theatre**: flat cardboard layers sway on strings and cast shadows on each other.
- **Thunderstorm**: rain falls behind whatever's close, lit by lightning.
- **Warp Speed**: the distance streaks into hyperspace while whoever's close stays still.
- **Beyond the Veil**: past a shimmering line in the distance, the world turns into something else.
- **Voxel World**: the scene rebuilt from blocks, standing taller the closer they are.
- **Neon Grid**: the world goes dark and everything's edges light up in neon.
- **God Rays**: shafts of light through the air, from the brightest thing in the shot and in its colour, with people cutting dark shadows through them and dust that swirls as things move.
- **Film Mountains**: the picture as a landscape: close things are peaks, far things are valleys.
- **Signals**: shows what Flex can work out from the picture, for checking and making effects.
- **Motion Paint**: only what's moving shows, painted in a colour for each direction.
- **Echoes**: everything leaves a trail of coloured copies of where it was a moment ago.
- **Everyone's Colour**: each person is washed in their own colour, which follows them about.
- **Film Palette**: the film redrawn in just its own four main colours.
- **Slow Burn**: the picture gets stranger the longer a shot goes on, and every cut snaps it back.
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
| `vec4 u_ground` | with `depth = true`: the floor, a table or whatever else is flat and level, found from depth. At a point of the picture (0 to 1, y up) its depth is `dot(vec3(x, y, 1.0), u_ground.xyz)`, and `u_ground.w` is how sure that is, from 0 to 1 |
| `sampler2D u_depth`, `float u_has_depth` | the depth map itself, covering the picture rather than the screen, and 1 once there is one |
| `float person_at(vec2 uv)`, `wall_at`, `sky_at` | functions: how much of that point of the screen is a person, a wall or sky, from 0 to 1 (with `segment = true`; 0 until the first masks arrive). `vec4 masks_at(vec2 uv)` gives all three in r, g and b |
| `int u_face_count`, `vec4 u_faces[MAX_FACES]` | how far through the arrays to look, and each face's box on screen: x0, y0, x1, y1 (with `faces = true`) |
| `float u_face_seen[MAX_FACES]` | 0 to 1, fading as a face comes and goes; multiply by it so faces don't pop. It's 0 for empty slots |
| `vec2 face_point(int face, int point)` | where on the screen a face's point is: 0 right eye, 1 left eye, 2 nose, 3 right corner of the mouth, 4 left (the person's own right and left) |
| `float face_at(vec2 uv)` | a function: how much of any face is at that point of the screen, from 0 to 1, soft at the edges |
| `vec3 mesh_point(int face, int n)`, `float u_face_mesh[MAX_FACES]` | a function: the mesh's point `n` (0 to 477, MediaPipe's numbering): x, y on screen and z, smaller nearer, in picture widths (with `mesh = true`). `u_face_mesh` says how much of a mesh that face has, from 0 for faces side-on |
| `vec3 u_face_open[MAX_FACES]` | how open the person's right eye, left eye and mouth are, from 0 to 1 (with `mesh = true`, and only where `u_face_mesh` is above 0) |
| `vec4 u_face_turn[MAX_FACES]` | which way a head points: x to the right of the picture, y up, z towards the viewer, and w 1 if that's known (with `mesh = true`). Side-on faces have this even with no mesh |
| `float on_face(vec2 uv)`, `vec2 face_uv(vec2 uv)`, `int face_there(vec2 uv)` | functions: how much of a face a point on screen is, from 0 to 1; where it is on the face (x 0 to 1 across it, y from the chin up, the same spot on every face whichever way it's turned); and which face it is, or -1 (with `mesh = true`) |
| `float skin_at(vec2 uv)`, `lips_at`, `nose_at`, `brows_at`, `eye_at`, `iris_at`, `mouth_at` | functions: how much of a point on screen is that part of a face, from 0 to 1 (with `mesh = true`). `mouth_at` is the gap between the lips, `eye_at` the whole opening of the eye and `iris_at` the coloured part of it |
| `vec4 face_texture(int face, vec2 at)` | a function: a face's pixels at a position on it (as `face_uv` gives), for drawing it somewhere else; alpha is 0 off the face (with `mesh = true`) |
| `float hair_at(vec2 uv)` | a function: roughly where hair is: the person above and around each face (with `mesh = true` and `segment = true`). It knows nothing of individual strands |
| `float face_roll(int face)` | a function: how far a head tilts to the side, anticlockwise, in radians (with `faces = true`) |
| `int who_at(vec2 uv)` | a function: which face a point of the picture belongs with, as an index into `u_faces`, or -1 (with `faces = true`). Use it with `person_at` to treat people separately |
| `float u_dt` | seconds since the last frame was drawn |
| `float u_cut`, `u_since_cut` | 1 on a scene cut, fading to 0 in about half a second, and seconds since the last one, for the frame on screen (with `cuts = true`) |
| `vec3 history_at(vec2 uv, float seconds)` | a function: the picture that many seconds ago, up to about one, at half size (with `history = true`) |
| `vec2 flow_at(vec2 uv)` | a function: how a point is moving, in screen widths and heights a second (with `flow = true`) |
| `vec3 plate_at(vec2 uv)`, `float plate_known(vec2 uv)` | functions: the room with people taken out, and how much of that point has really been seen since the last cut rather than filled in from around it (with `plate = true`) |
| `float u_brightness`, `u_colourfulness`, `u_warmth`, `vec3 u_palette[4]`, `float u_palette_share[4]` | the look of the picture: 0–1, 0–1, -1 (blue) to 1 (orange), its four main colours, and how much of the picture each covers (with `stats = true`). Colours keep their slots as they change |
| `vec3 u_light` | with `stats = true`: where the brightest part of the picture is, x and y from 0 to 1 across the picture, and in z how much it stands out as one place |

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
mesh = true                    # a closer look at faces, as well as where they are; default false
scale = 1.0                    # work at this fraction of the screen's size (0.25–1), then
                               # scale up: lower is lighter on the GPU; default 1
history = true                 # older frames; default false
flow = true                    # how things are moving; default false
plate = true                   # the room behind people (runs the masks); default false
cuts = true                    # scene cuts; default false
stats = true                   # the picture's palette and brightness; default false
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
are kept per effect. A `number` setting can say how it drifts with `drift = "smooth"`,
`"steps"` or `"off"`: use `"off"` for anything multiplied by the time, like a speed, or the
picture will jump as it changes. Left out, whole-numbered counts go in steps and the rest
smoothly. A `choice` setting takes `options = [{ id = "soft", label = "Soft" }, ...]`, and
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
depth, the picture and sound are held back a moment, usually under a fifth of a second, which
also lets Flex steady each face using the frames either side. A face keeps its slot in the arrays while it's on screen. It works best on
live action; very small faces, faces in profile or turned away, and cartoon characters are
missed some of the time, so effects need to look fine with no faces at all.

With `mesh = true`, Flex also looks closer at each frame's faces, on that same frame: up to
three of the largest get a 478-point mesh from MediaPipe's
[Face Mesh V2](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (Apache
2.0), which gives how open the eyes and mouth are too, and a
[head pose network](https://github.com/yakhyo/head-pose-estimation) (MIT) says which way heads
point, taking turns if there are several. Mesh costs more GPU time than `faces` alone, so only
ask for it if you'll use it. The mesh gives up on faces turned far to the side, so check
`u_face_mesh`; the direction carries on working there, and is left out for the backs of heads.

The mesh is also drawn into maps of the picture for every frame, so a shader can ask about any
pixel: `on_face`, the parts of the face (`skin_at`, `lips_at` and so on), and `face_uv`, which is
the same place on every face however it's turned. Painting in `face_uv` keeps things stuck to the
skin. `face_texture` gives a face's own pixels at any such position, for moving a face to
somewhere else. They only cover the face itself, from hairline to chin, and only faces with a
mesh; `meshed(face)` says whether a slot has one, so an effect can fall back on `u_faces`.

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
