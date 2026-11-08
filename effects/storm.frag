// Thunderstorm: layers of falling streaks each at their own distance, hidden by anything nearer,
// and splashing on the top of things. Lightning is kept soft and brief, as flashes are hard on eyes.

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

vec2 depth_slope(vec2 uv, float reach) {
    vec2 e = reach / u_resolution;
    return vec2(depth_at(uv + vec2(e.x, 0.0)) - depth_at(uv - vec2(e.x, 0.0)), depth_at(uv + vec2(0.0, e.y)) - depth_at(uv - vec2(0.0, e.y)));
}

float streaks(vec2 p, float scale, float t) {
    vec2 q = vec2(p.x * scale - p.y * s_wind / 100.0 * scale * 0.3, p.y * scale * 0.11 + t * (1.4 + scale * 0.02));
    vec2 cell = floor(q);
    if (hash(cell) > s_rain / 100.0) return 0.0;
    vec2 f = fract(q);
    return (1.0 - smoothstep(0.02, 0.08, abs(f.x - 0.5 - 0.3 * (hash(cell + 3.0) - 0.5)))) * smoothstep(0.0, 0.3, f.y) * (1.0 - smoothstep(0.7, 1.0, f.y));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 p = uv * vec2(u_resolution.x / u_resolution.y, 1.0);
    float t = u_time;
    float near = depth_at(uv);
    float gloom = s_dark / 100.0;
    vec3 graded = mix(c, c * vec3(0.5, 0.6, 0.8) * 0.8, gloom);

    float strike = pow(max(noise(vec2(t * 2.3, 7.0)) - 0.62, 0.0) / 0.38, 2.0) + u_beat * 0.25 * s_react / 100.0;
    strike *= s_lightning / 100.0;
    graded += c * vec3(0.8, 0.85, 1.0) * strike * (0.4 + 1.6 * (1.0 - near));

    float fall = 0.0;
    for (int i = 0; i < 4; i++) {
        float f = float(i);
        fall += streaks(p + f * 3.1, 90.0 - f * 20.0, t) * step(near, 0.2 + f * 0.25) * (0.22 + f * 0.09);
    }
    graded += vec3(0.75, 0.82, 0.95) * fall * (1.0 + 2.0 * strike);

    // Splashes on the top of things.
    float ledge = smoothstep(0.02, 0.06, near - depth_at(uv + vec2(0.0, 8.0 / u_resolution.y)));
    if (ledge > 0.0) {
        vec2 q = p * 90.0;
        vec2 cell = floor(q);
        float time_slot = floor(t * 7.0 + hash(cell) * 7.0);
        if (hash(cell + time_slot) > 0.93) graded += vec3(0.8, 0.86, 1.0) * ledge * (1.0 - smoothstep(0.15, 0.4, length(fract(q) - 0.5))) * s_rain / 100.0;
    }
    return vec4(graded, 1.0);
}
