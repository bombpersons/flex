// Silhouette Cosmos: inside people's outlines there's a nebula and three layers of stars.

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

vec3 stars(vec2 p, float scale, float t) {
    vec2 q = p * scale;
    vec2 cell = floor(q);
    vec2 place = vec2(hash(cell), hash(cell + 7.3)) * 0.7 + 0.15;
    float d = length(fract(q) - place);
    float twinkle = 0.6 + 0.4 * sin(t * (2.0 + 5.0 * hash(cell + 3.1)) + 40.0 * hash(cell));
    float there = step(0.72, hash(cell + 1.7));
    vec3 tint = mix(vec3(0.7, 0.8, 1.0), vec3(1.0, 0.8, 0.6), hash(cell + 5.5));
    return tint * there * twinkle * (1.0 - smoothstep(0.0, 0.09, d));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    float mask = person_at(uv);
    float inside = smoothstep(0.35, 0.65, mask);
    vec2 p = uv * vec2(u_resolution.x / u_resolution.y, 1.0);
    float t = u_time * s_speed / 100.0;
    if (inside > 0.0) {
        vec2 q = p * 2.2 + vec2(t * 0.05, t * 0.02);
        vec2 warp = vec2(fbm(q + t * 0.04), fbm(q + 5.2 - t * 0.03));
        float cloud = fbm(q + 3.0 * warp);
        vec3 nebula = mix(vec3(0.02, 0.0, 0.1), vec3(0.45, 0.05, 0.55), smoothstep(0.3, 0.7, cloud));
        nebula = mix(nebula, vec3(0.1, 0.5, 0.9), smoothstep(0.5, 0.85, warp.x));
        nebula = mix(nebula, vec3(1.0, 0.65, 0.4), pow(cloud, 4.0) * 2.5);
        vec3 sky = nebula * (1.0 + u_bass * s_react / 100.0);
        vec3 lights = stars(p + t * 0.02, 18.0, u_time) + stars(p + t * 0.04, 34.0, u_time) * 0.7 + stars(p + t * 0.08, 60.0, u_time) * 0.5;
        sky += lights * s_stars / 50.0 * (1.0 + 2.0 * u_treble * s_react / 100.0);
        c = mix(c, sky, inside);
    }
    float edge = smoothstep(0.15, 0.5, mask) * (1.0 - smoothstep(0.5, 0.85, mask));
    c += edge * vec3(0.5, 0.7, 1.0) * s_rim / 60.0;
    return vec4(c, 1.0);
}
