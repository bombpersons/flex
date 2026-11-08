// Snow Globe: flakes in layers at different distances, hidden by anything nearer. Where the top
// of something is, snow builds up in the alpha of u_prev, and a beat knocks it off again.

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

float flakes(vec2 p, float scale, float t, float shake) {
    vec2 q = p * scale;
    q.y += t * (1.2 + 0.6 * hash(vec2(scale)));
    q.x += sin(t * 0.5 + q.y * 0.25) * 0.6 + shake * sin(t * 9.0 + scale) * 4.0;
    vec2 cell = floor(q);
    if (hash(cell + 2.0) > s_snow / 100.0) return 0.0;
    vec2 place = vec2(hash(cell), hash(cell + 5.0)) * 0.6 + 0.2;
    return (1.0 - smoothstep(0.02, 0.1, length(fract(q) - place)));
}

vec4 effect(vec2 uv) {
    vec2 at = uv;
    float glass = 0.0;
    if (s_glass < 0.5) {
        vec2 off = uv - 0.5;
        at = 0.5 + off * (0.93 + 0.12 * dot(off, off));
        glass = dot(off, off);
    }
    vec3 c = texture(u_video, at).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 1.0);
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = at * aspect;
    float t = u_time;
    float near = depth_at(at);
    float kick = u_beat * s_shake / 100.0;

    // The top of something: nearer here than just above.
    float step_up = 3.0 / u_resolution.y;
    float ledge = smoothstep(0.03, 0.07, near - depth_at(at + vec2(0.0, step_up * 3.0)));
    vec4 before = texture(u_prev, uv);
    float lying = u_frame < 2 ? 0.0 : before.a;
    if (lying > 1.0) lying = 0.0;
    float rate = s_settle / 100.0 * 0.025 * smoothstep(0.35, 0.75, fbm(p * 26.0));
    lying = clamp(lying + ledge * rate - 0.0012 - kick * 0.25, 0.0, 1.0);
    // Falling away from places that aren't ledges any more.
    lying *= mix(0.97, 1.0, ledge);
    c = mix(c, vec3(0.95, 0.97, 1.0), smoothstep(0.1, 0.8, lying));

    float fall = 0.0;
    for (int i = 0; i < 4; i++) {
        float f = float(i);
        float layer = 0.2 + f * 0.25;
        fall += flakes(p, 46.0 - f * 11.0, t, kick) * step(near, layer + 0.1) * (0.5 + f * 0.17);
    }
    c += vec3(0.95, 0.97, 1.0) * fall;
    if (s_glass < 0.5) {
        c *= 1.0 - smoothstep(0.2, 0.55, glass) * 0.55;
        vec2 shine = (uv - vec2(0.27, 0.74)) * vec2(1.0, 1.7);
        c += vec3(0.8, 0.9, 1.0) * (1.0 - smoothstep(0.0, 0.17, length(shine))) * 0.13;
    }
    return vec4(c, lying);
}
