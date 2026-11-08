// Black Hole: each point shows what's further out from the hole than it is, so things look
// squeezed against it, by less the nearer they are. Then a black disc and a glowing, turning ring.

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

vec4 effect(vec2 uv) {
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_time * s_drift / 100.0;
    vec2 centre = (vec2(0.5, 0.52) + vec2(0.28 * sin(t * 0.31), 0.17 * sin(t * 0.43 + 1.0))) * aspect;
    float radius = 0.075 * s_size / 100.0 * (1.0 + 0.3 * u_bass);
    vec2 rel = p - centre;
    float r = max(length(rel), 1e-4);
    float near = depth_at(uv);
    float give = s_pull / 100.0 * (1.0 - near * s_resist / 100.0);

    float bend = radius * radius / r * 2.4 * give;
    float turn = radius / r * 1.1 * give;
    vec2 d = rel / r;
    d = vec2(d.x * cos(turn) - d.y * sin(turn), d.x * sin(turn) + d.y * cos(turn));
    vec2 from = (centre + d * (r + bend)) / aspect;
    vec3 c = texture(u_video, clamp(from, u_video_rect.xy, u_video_rect.zw)).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) c = vec3(0.0);

    // Red-shifted towards the edge of the disc.
    c *= mix(vec3(1.0, 0.45, 0.35), vec3(1.0), smoothstep(radius, radius * 3.2, r));

    // The disc of matter, seen at a slant.
    vec2 tilted = vec2(rel.x * 0.94 + rel.y * 0.34, (rel.y * 0.94 - rel.x * 0.34) * 3.2);
    float ring = length(tilted);
    float angle = atan(tilted.y, tilted.x);
    float streaks = fbm(vec2(angle * 3.0 + u_time * 1.2, ring * 40.0 - u_time * 0.6));
    float glow = smoothstep(radius * 1.25, radius * 1.5, ring) * (1.0 - smoothstep(radius * 1.7, radius * 3.4, ring));
    float side = 1.0 + 0.8 * sin(angle + 0.4);
    c += vec3(1.0, 0.55, 0.2) * glow * streaks * streaks * 3.2 * side * (1.0 + u_beat);

    c = mix(c, vec3(0.0), (1.0 - smoothstep(radius * 0.97, radius * 1.03, r)));
    c += vec3(1.0, 0.8, 0.55) * exp(-pow((r - radius * 1.06) / (radius * 0.05), 2.0)) * 1.1;
    return vec4(c, 1.0);
}
