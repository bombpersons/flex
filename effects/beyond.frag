// Beyond the Veil: whatever's further than the veil is replaced by the film folded into a slowly
// flowing kaleidoscope, with a burning edge where the two meet.

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
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_time * s_flow / 100.0;
    float far = 1.0 - depth_at(uv);
    float veil = s_distance / 100.0 + (fbm(p * 5.0 + t * 0.2) - 0.5) * 0.3 * s_shimmer / 100.0 - u_bass * 0.06 * s_react / 100.0;
    float beyond = smoothstep(veil - 0.02, veil + 0.02, far);
    if (beyond > 0.0) {
        vec2 rel = (uv - 0.5) * aspect;
        float wedge = 6.2831853 / max(s_arms, 1.0);
        float a = abs(mod(atan(rel.y, rel.x) + t * 0.12, wedge) - wedge * 0.5);
        vec2 k = length(rel) * vec2(cos(a), sin(a));
        k += vec2(fbm(k * 4.0 + t * 0.2), fbm(k * 4.0 - t * 0.17)) * 0.2 - 0.1;
        vec2 size = u_video_rect.zw - u_video_rect.xy;
        vec2 from = 1.0 - abs(1.0 - mod((k - vec2(0.3, 0.0)) / aspect / size + 0.5, 2.0));
        vec3 other = texture(u_video, u_video_rect.xy + from * size).rgb;
        other = mix(other, other * hue(length(rel) * 1.5 - t * 0.2) * 2.2, 0.75);
        c = mix(c, other, beyond);
    }
    float seam = 1.0 - smoothstep(0.0, 0.03, abs(far - veil));
    c += mix(vec3(1.0, 0.6, 0.2), vec3(0.6, 0.8, 1.0), 0.5 + 0.5 * sin(t + p.x * 4.0)) * seam * (1.2 + u_beat * s_react / 100.0);
    return vec4(c, 1.0);
}
