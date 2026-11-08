// God Rays: walks from each point towards the light, adding up how much of the way is clear.
// Near things stand in the way, so they throw dark shafts through the beams.

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

vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(0.0, 0.0, 0.0, 1.0);
    float t = u_time * s_speed / 100.0;
    vec2 light = vec2(0.5 + 0.36 * sin(t * 0.5), u_video_rect.w + 0.05);
    vec2 step_by = (light - uv) / 28.0;
    vec2 at = uv + step_by * hash(uv * 311.0);
    float clear = 0.0;
    float fade = 1.0;
    for (int i = 0; i < 28; i++) {
        at += step_by;
        float there = depth_at(clamp(at, u_video_rect.xy, u_video_rect.zw));
        clear += (1.0 - smoothstep(0.35, 0.6, there)) * fade;
        fade *= 0.965;
    }
    clear /= 28.0;
    float dust = 0.7 + 0.6 * fbm(uv * 6.0 + vec2(u_time * 0.05, -u_time * 0.03));
    vec3 tint = mix(hue(s_colour / 360.0) * 0.5 + 0.55, hue(uv.x * 2.0 - t * 0.1), s_rainbow / 100.0);
    // Separate shafts, fanning out from the light.
    vec2 from_light = uv - light;
    float angle = atan(from_light.x * u_resolution.x / u_resolution.y, -from_light.y);
    float shafts = smoothstep(0.42, 0.72, noise(vec2(angle * 11.0 + t * 0.15, t * 0.1)));
    float strength = clear * clear * dust * shafts * s_strength / 100.0 * (1.0 + 0.9 * u_bass * s_react / 100.0);
    return vec4(c * 0.72 + tint * strength * 0.85, 1.0);
}
