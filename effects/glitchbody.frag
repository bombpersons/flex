// Glitch Bodies: the picture is cut into strips, each pushed sideways a random amount that changes
// on the beat, but only what's a person gets moved. Where someone has moved off, the signal drops out.

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

// What's probably behind someone: the picture around them, averaged.
vec3 room_behind(vec2 uv) {
    vec3 sum = vec3(0.0);
    float total = 1e-3;
    for (int r = 1; r <= 3; r++) {
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.7853982 + float(r);
            vec2 q = uv + vec2(cos(a), sin(a)) * vec2(u_resolution.y / u_resolution.x, 1.0) * float(r) * 0.11;
            float clear = 1.0 - smoothstep(0.1, 0.4, person_at(q));
            sum += texture(u_video, clamp(q, u_video_rect.xy, u_video_rect.zw)).rgb * clear;
            total += clear;
        }
    }
    return sum / total;
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 1.0);
    float t = u_video_time;
    float strip = floor(uv.y * u_resolution.y / (s_strips * u_resolution.y / 720.0));
    float moment = floor(t * 9.0);
    float strength = (s_always / 100.0 * step(0.55, hash(vec2(strip, moment + 3.0))) + u_beat) * s_tear / 100.0;
    float shift = (hash(vec2(strip, moment)) - 0.5) * 0.28 * strength;
    float person_here = smoothstep(0.4, 0.65, person_at(uv));
    vec2 from = uv + vec2(shift, 0.0);
    float there = from == clamp(from, u_video_rect.xy, u_video_rect.zw) ? smoothstep(0.4, 0.65, person_at(from)) : 0.0;
    if (person_here + there <= 0.0) return vec4(c, 1.0);
    vec3 shown = c;
    if (person_here > 0.0 && abs(shift) > 0.002) {
        // Dropped out where they used to be.
        float n = hash(vec2(floor(uv.x * u_resolution.x / 3.0), strip + moment));
        shown = mix(c, mix(room_behind(uv) * 0.8, vec3(n) * vec3(0.3, 0.9, 1.0), 0.35), person_here);
    }
    if (there > 0.0) {
        float apart = (0.004 + abs(shift) * 0.2) * s_split / 50.0;
        vec3 body = vec3(texture(u_video, from + vec2(apart, 0.0)).r, texture(u_video, from).g, texture(u_video, from - vec2(apart, 0.0)).b);
        shown = mix(shown, body, there);
    }
    return vec4(shown, 1.0);
}
