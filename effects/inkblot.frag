// Inkblot: mirrors the picture about its middle, wobbles it with noise, and turns it into ink on
// card, with the edges soaking outwards.

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
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 middle = 0.5 * (u_video_rect.xy + u_video_rect.zw);
    vec2 q = uv;
    q.x = middle.x - abs(q.x - middle.x);
    if (s_fold > 0.5) q.y = middle.y - abs(q.y - middle.y);
    float t = u_video_time * s_wander / 100.0;
    vec2 p = (q - middle) * vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 warp = vec2(fbm(p * 3.0 + vec2(t * 0.15, 1.0)), fbm(p * 3.0 + vec2(3.3, t * 0.12))) - 0.47;
    vec3 c = texture(u_video, clamp(q + warp * 0.11, u_video_rect.xy, u_video_rect.zw)).rgb;
    float luma = dot(c, vec3(0.3, 0.59, 0.11));
    float blot = luma + (fbm(p * 9.0 + warp * 3.0 - t * 0.1) - 0.5) * 0.45 - u_bass * 0.1;
    float edge = s_ink / 100.0 * 0.85;
    float ink = (1.0 - smoothstep(edge - 0.012, edge + 0.012, blot));
    float soak = (1.0 - smoothstep(edge, edge + 0.09, blot)) * 0.2;
    vec3 card = vec3(0.93, 0.89, 0.8) * (0.96 + 0.08 * noise(p * 300.0));
    vec3 black = mix(vec3(0.03, 0.03, 0.06), c * 1.1, s_film / 100.0);
    vec3 out_colour = mix(card * (1.0 - soak), black, ink);
    return vec4(out_colour, 1.0);
}
