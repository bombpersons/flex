// Pixel Sort Waterfall: anything bright above a point falls into it, a little further each frame,
// at a speed that differs from column to column. Near the bottom it stops falling and piles up.

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

float luma(vec3 c) {
    return dot(c, vec3(0.3, 0.59, 0.11));
}

vec4 effect(vec2 uv) {
    vec3 video = texture(u_video, uv).rgb;
    if (u_frame < 2 || uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(video, 1.0);
    float column = floor(uv.x * u_resolution.x / 2.0);
    float height = (uv.y - u_video_rect.y) / max(u_video_rect.w - u_video_rect.y, 1e-3);
    float rate = (1.0 + 9.0 * pow(hash(vec2(column, 1.0)), 2.0)) * s_speed / 100.0 * (1.0 + 1.5 * u_bass);
    float pooling = (1.0 - smoothstep(s_pool / 100.0 - 0.01, s_pool / 100.0 + 0.01, height + (noise(vec2(uv.x * 40.0, u_time)) - 0.5) * 0.015));
    vec2 above = uv + vec2(0.0, mix(rate, 0.0, pooling) * 2.0 / u_resolution.y);
    vec3 falling = texture(u_prev, above).rgb;
    float level = s_level / 100.0 + (hash(vec2(column, 3.0)) - 0.5) * 0.1;
    float bright = smoothstep(level, level + 0.1, luma(falling));
    vec3 c = mix(video, falling, bright * s_stay / 100.0);
    return vec4(c, 1.0);
}
