// Smoke People: noise eats holes in people from the outside in, showing what's probably behind
// them, and what's eaten goes into a smoke layer in the alpha of u_prev that rises and curls.

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
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 0.0);
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_time;
    float mask = person_at(uv);
    float person = smoothstep(0.35, 0.7, mask);
    // How far inside someone this is, roughly.
    float inner = 0.25 * (person_at(uv + vec2(0.05, 0.0) / aspect) + person_at(uv - vec2(0.05, 0.0) / aspect) + person_at(uv + vec2(0.0, 0.05)) + person_at(uv - vec2(0.0, 0.05)));
    float grain = fbm(p * 9.0 + vec2(0.0, -t * 0.5));
    float eaten = person * smoothstep(0.32, 0.62, grain + (1.0 - inner) * 0.5 * s_holes / 50.0 - 0.28 + u_bass * s_react / 400.0) * step(0.001, s_holes);

    vec2 curl = vec2(fbm(p * 6.0 + vec2(0.0, -t * 0.4)), fbm(p * 6.0 + vec2(5.2, -t * 0.3))) - 0.47;
    vec2 from = uv - vec2(s_wind / 100.0 * 0.0025, 0.0028) - curl * 0.006;
    float smoke = u_frame < 2 ? 0.0 : texture(u_prev, from).a * 0.975;
    smoke = max(smoke, (eaten + person * (1.0 - inner) * 0.8) * s_smoke / 100.0);

    vec3 shown = c;
    if (eaten > 0.0) shown = mix(c, room_behind(uv), eaten);
    float solid = person * (1.0 - eaten);
    vec3 ash = mix(vec3(0.62, 0.6, 0.58), vec3(0.9, 0.86, 0.8), grain);
    shown = mix(shown, ash, smoothstep(0.03, 0.8, smoke) * (1.0 - solid) * 0.85);
    return vec4(shown, smoke);
}
