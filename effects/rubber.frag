// Rubber Sheet: a mesh made in pixel space. There's a grid of vertices over the picture, each
// raised by the depth under it and moved by its own wobble; between them heights blend, as
// across a mesh's faces. Each pixel walks along its view ray until it goes under that surface,
// then shows the part of the film pinned there. Where depth jumps, the sheet stretches.
//
// s_style: 0 jelly, 1 ripples, 2 facets, 3 wireframe.

const int STEPS = 40;

vec2 cells() {
    vec2 size = u_video_rect.zw - u_video_rect.xy;
    float across = floor(s_cells);
    return vec2(across, max(1.0, floor(across * size.y * u_resolution.y / (size.x * u_resolution.x))));
}

vec2 to_picture(vec2 q) {
    return (q - u_video_rect.xy) / (u_video_rect.zw - u_video_rect.xy);
}

float hash(vec2 v, float k) {
    return fract(sin(dot(v, vec2(127.1 + k, 311.7 - k))) * 43758.5453);
}

// A vertex's height (.x) and sideways movement (.yz).
vec3 vertex(vec2 v, vec2 grid, float t, float amount) {
    vec2 at = u_video_rect.xy + (v / grid) * (u_video_rect.zw - u_video_rect.xy);
    float wave;
    vec2 side;
    if (s_style > 1.5 && s_style < 2.5) {
        float phase = 6.2831853 * hash(v, 0.0);
        wave = 0.6 * sin(t * (1.5 + 2.0 * hash(v, 3.0)) + phase);
        side = vec2(cos(t * 1.7 + phase), sin(t * 1.3 + 2.0 * phase));
    } else if (s_style > 0.5 && s_style < 1.5) {
        vec2 p = (v / grid - 0.5) * vec2(grid.x / grid.y, 1.0);
        float r = length(p);
        wave = sin(r * 26.0 - t * 4.0);
        side = p / max(r, 0.001) * cos(r * 26.0 - t * 4.0);
    } else {
        vec2 p = v / grid.x;
        wave = (sin(p.x * 17.0 + t * 2.0) + sin(p.y * 23.0 - t * 1.4) + sin((p.x + p.y) * 11.0 + t)) / 3.0;
        side = vec2(cos(p.y * 19.0 + t * 1.6), sin(p.x * 15.0 - t * 1.2));
    }
    return vec3(depth_at(at) + wave * 0.35 * amount, side * 0.018 * amount);
}

vec3 surface(vec2 q, vec2 grid, float t, float amount) {
    vec2 g = to_picture(q) * grid;
    vec2 v = floor(g);
    vec2 f = g - v;
    return mix(
        mix(vertex(v, grid, t, amount), vertex(v + vec2(1.0, 0.0), grid, t, amount), f.x),
        mix(vertex(v + vec2(0.0, 1.0), grid, t, amount), vertex(v + vec2(1.0, 1.0), grid, t, amount), f.x),
        f.y
    );
}

// Mirrors past the picture's edge, so a tilted view never shows a hole.
vec2 fit(vec2 q) {
    vec2 size = u_video_rect.zw - u_video_rect.xy;
    vec2 a = mod((q - u_video_rect.xy) / size, 2.0);
    return u_video_rect.xy + size * (1.0 - abs(1.0 - a));
}

vec4 effect(vec2 uv) {
    // The black bars stay black.
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }

    float t = u_video_time * s_speed;
    float amount = s_wobble / 100.0 * (1.0 + (u_beat + u_bass) * s_react / 100.0);
    vec2 grid = cells();
    vec2 centre = 0.5 * (u_video_rect.xy + u_video_rect.zw);

    // The view circles, so there's always some angle on the mesh, and leans in a little.
    float angle = 0.45 * t;
    vec2 tilt = s_sway / 100.0 * vec2(0.16 * cos(angle), 0.11 * sin(angle));
    float lean = 0.22 * s_sway / 100.0;
    float top = 1.0 + 0.35 * amount;
    float bottom = -0.35 * amount;
    float span = top - bottom;

    float z = top;
    vec2 q = uv;
    bool hit = false;
    float above = top;
    for (int i = 0; i <= STEPS; i++) {
        z = top - span * float(i) / float(STEPS);
        vec2 c = fit(uv + (tilt + (uv - centre) * lean) * (z - 0.5));
        if (surface(c, grid, t, amount).x >= z) {
            hit = true;
            break;
        }
        above = z;
    }
    if (!hit) {
        z = bottom;
        above = bottom;
    }
    for (int j = 0; j < 5; j++) {
        float middle = 0.5 * (z + above);
        vec2 c = fit(uv + (tilt + (uv - centre) * lean) * (middle - 0.5));
        if (surface(c, grid, t, amount).x >= middle) {
            z = middle;
        } else {
            above = middle;
        }
    }
    q = fit(uv + (tilt + (uv - centre) * lean) * (z - 0.5));

    vec3 s = surface(q, grid, t, amount);
    vec3 video = texture(u_video, fit(q + s.yz)).rgb;

    // Light from the upper left, so the wobble reads on flat walls too.
    vec2 e = (u_video_rect.zw - u_video_rect.xy) / grid;
    float slope = surface(q + vec2(e.x, 0.0), grid, t, amount).x - s.x + surface(q + vec2(0.0, e.y), grid, t, amount).x - s.x;
    vec3 c = video * (1.0 + clamp(slope * 2.5, -0.3, 0.35));

    if (s_style > 2.5) {
        vec2 g = to_picture(q) * grid;
        vec2 w = fwidth(g) * 1.2;
        vec2 near_line = min(fract(g), 1.0 - fract(g));
        float line = 1.0 - min(smoothstep(0.0, w.x, near_line.x), smoothstep(0.0, w.y, near_line.y));
        vec3 glow = 0.5 + 0.5 * cos(6.2831853 * (z * 1.5 + 0.05 * t + vec3(0.0, 0.33, 0.67)));
        c = mix(c * 0.45, glow * 1.3, line);
    }
    return vec4(c, 1.0);
}
