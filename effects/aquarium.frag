// Aquarium: below a wavy waterline the picture bends, picks up caustics and fades to blue with
// distance, and bubbles rise behind whatever's nearest.

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float sum = 0.0, amount = 0.5;
    for (int i = 0; i < 5; i++) {
        sum += amount * noise(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        amount *= 0.5;
    }
    return sum;
}

// The usual web of light on the bottom of a pool, from a few rounds of folding space.
float caustics(vec2 p, float t) {
    vec2 q = mod(p * 14.0, 6.2831853) - 250.0;
    vec2 i = q;
    float sum = 1.0;
    for (int n = 0; n < 4; n++) {
        float at = t * 0.6 * (1.0 - 3.5 / float(n + 1));
        i = q + vec2(cos(at - i.x) + sin(at + i.y), sin(at - i.y) + cos(at + i.x));
        sum += 1.0 / length(vec2(q.x / (sin(i.x + at) / 0.005), q.y / (cos(i.y + at) / 0.005)));
    }
    sum = 1.17 - pow(sum / 4.0, 1.4);
    return pow(abs(sum), 8.0);
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 1.0);
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_time;
    float filled = s_rise < 0.5 ? 1.0 : smoothstep(0.0, 1.0, t / s_rise);
    float bottom = u_video_rect.y, top = u_video_rect.w;
    float level = mix(bottom, top, s_level / 100.0 * filled);
    level += (sin(p.x * 11.0 + t * 1.4) * 0.007 + sin(p.x * 23.0 - t * 2.1) * 0.003) * (1.0 + 2.0 * u_bass);
    float under = (1.0 - smoothstep(-0.003, 0.003, uv.y - level));
    if (under <= 0.0) return vec4(c, 1.0);

    vec2 bend = vec2(sin(p.y * 28.0 + t * 1.7) + sin(p.y * 13.0 - t), cos(p.x * 24.0 - t * 1.4)) * 0.0028;
    vec2 from = clamp(uv + bend, u_video_rect.xy, u_video_rect.zw);
    float near = depth_at(from);
    vec3 w = texture(u_video, from).rgb;
    float deep = clamp((level - uv.y) / (top - bottom), 0.0, 1.0);
    float far = (1.0 - near) * s_murk / 100.0;
    w *= vec3(0.55, 0.85, 1.0);
    w += vec3(0.75, 0.95, 1.0) * min(caustics(p + near * 0.25, t), 1.5) * (1.0 - deep * 0.5) * s_light / 100.0 * (0.25 + 0.5 * near);
    vec3 blue = mix(vec3(0.05, 0.42, 0.56), vec3(0.01, 0.1, 0.3), deep);
    w = mix(w, blue, clamp(far * 1.2 + deep * 0.15, 0.0, 0.92));

    // Rays down from the surface.
    float rays = fbm(vec2(p.x * 7.0 + (level - uv.y) * 2.0, t * 0.15));
    w += vec3(0.3, 0.45, 0.5) * smoothstep(0.5, 0.8, rays) * (1.0 - deep) * s_light / 100.0 * 0.5;

    for (int i = 0; i < 3; i++) {
        float f = float(i);
        float size = 0.03 + f * 0.018;
        vec2 q = vec2(p.x, p.y - t * (0.12 + 0.05 * f)) / size + f * 13.0;
        vec2 cell = floor(q);
        float keep = step(0.85, hash(cell));
        vec2 centre = vec2(0.5 + 0.28 * sin(t * 1.5 + 6.0 * hash(cell + 3.0)), 0.5);
        float d = length(fract(q) - centre) / (0.15 + 0.2 * hash(cell + 1.0));
        float layer_near = 0.25 + 0.3 * f;
        float bubble = keep * (1.0 - smoothstep(0.85, 1.0, d)) * (0.25 + 0.75 * smoothstep(0.6, 1.0, d)) * step(near, layer_near);
        w += vec3(0.85, 0.95, 1.0) * bubble * 0.5;
    }
    w += vec3(0.8, 0.95, 1.0) * (1.0 - smoothstep(0.0, 0.012, abs(uv.y - level))) * 0.6;
    return vec4(mix(c, w, under), 1.0);
}
