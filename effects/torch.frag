// Torchlight: each point gets a position in space from depth and a surface direction from how
// depth changes around it, and the torch lights it like a real scene.

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

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(0.0, 0.0, 0.0, 1.0);
    vec3 tints[4] = vec3[4](vec3(1.0, 0.82, 0.55), vec3(1.0), vec3(0.55, 0.75, 1.0), vec3(1.0, 0.2, 0.15));
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    float t = u_time * s_speed / 100.0;
    vec2 aim = vec2(0.5 + 0.38 * sin(t * 0.6) * cos(t * 0.23), 0.5 + 0.3 * sin(t * 0.47 + 1.0));
    float near = depth_at(uv);
    vec2 rel = (uv - aim) * aspect;
    float cone = 0.36 * s_size / 100.0;
    float beam = (1.0 - smoothstep(cone * 0.45, cone, length(rel))) + 0.14 * (1.0 - smoothstep(cone, cone * 2.0, length(rel)));
    vec2 slope = depth_slope(uv, 4.0) * 16.0;
    vec3 normal = normalize(vec3(-slope, 1.0));
    float facing = clamp(dot(normal, normalize(vec3(-rel * 1.5, 1.0))), 0.0, 1.0);
    float falloff = 0.65 + 0.9 * near * near;
    float shake = 1.0 - s_flicker / 100.0 * 0.5 * noise(vec2(u_time * 11.0, 2.0));
    vec3 light = tints[int(s_colour + 0.5)] * beam * (0.3 + 0.9 * facing) * falloff * shake * (1.0 + 0.5 * u_bass * s_react / 100.0) * 1.7;
    float floor_level = 1.0 - s_dark / 100.0;
    return vec4(c * (vec3(floor_level) + light), 1.0);
}
