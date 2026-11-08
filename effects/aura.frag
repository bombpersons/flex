// Aura: light around people, kept in the alpha of u_prev and carried up and outwards each frame,
// so it flows off them. People themselves are left alone.

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
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 0.0);
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_time;
    float person = person_at(uv);
    float reach = 0.03 * s_size / 100.0 * (1.0 + u_bass * s_react / 50.0);
    float near = 0.0;
    for (int i = 0; i < 12; i++) {
        float a = float(i) * 0.5235988;
        near += person_at(uv + vec2(cos(a), sin(a)) * reach / aspect);
    }
    near /= 12.0;
    vec2 wander = vec2(fbm(p * 5.0 + vec2(0.0, -t * 0.7)), fbm(p * 5.0 + vec2(6.0, -t * 0.6))) - 0.47;
    vec2 from = uv - (vec2(0.0, 0.003) + wander * 0.008) * s_size / 100.0;
    float kept = u_frame < 2 ? 0.0 : texture(u_prev, from).a;
    float field = max(near * 1.1, kept * (0.955 + 0.03 * u_bass * s_react / 100.0));
    vec3 colour = hue(s_colours / 360.0 + s_spread / 100.0 * (field * 0.6 + wander.x * 1.5 + t * 0.05));
    float glow = smoothstep(0.05, 0.9, field) * (1.0 + 1.2 * u_beat * s_react / 100.0);
    float solid = smoothstep(0.4, 0.7, person);
    return vec4(c + colour * glow * (1.0 - solid) * 0.95, field);
}
