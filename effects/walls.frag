// Breathing Walls: inside the wall mask the picture swells and sinks, and bands of colour
// run along the room's depth, bent by a slow swirl.

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

float swirl(vec2 p, float t) {
    float v = 0.0;
    float a = 1.0;
    for (int i = 0; i < 4; i++) {
        p += vec2(sin(p.y * 1.7 + t), cos(p.x * 1.5 - t * 0.8)) * 0.6;
        v += a * sin(p.x * 2.1 + p.y * 1.3);
        p *= 1.8;
        a *= 0.55;
    }
    return v;
}

vec4 effect(vec2 uv) {
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float wall = smoothstep(0.3, 0.65, wall_at(uv));
    float t = u_video_time;
    vec2 centre = 0.5 * (u_video_rect.xy + u_video_rect.zw);
    float breath = sin(t * 1.1 + 3.0 * length(uv - centre)) * (1.0 + u_bass);
    vec2 q = uv + (centre - uv) * breath * 0.05 * s_breath / 100.0 * wall;
    vec3 c = texture(u_video, clamp(q, u_video_rect.xy, u_video_rect.zw)).rgb;
    float depth = depth_at(uv);
    float s = swirl(uv * vec2(u_resolution.x / u_resolution.y, 1.0) * 5.0, t * 0.3);
    float bands = fract(depth * 9.0 + s * 0.35 + t * 0.12);
    vec3 pattern = palette(bands + 0.25 * s + u_mid * 0.3) * smoothstep(0.0, 0.1, bands) * smoothstep(1.0, 0.85, bands);
    float light = dot(c, vec3(0.3, 0.59, 0.11));
    vec3 lit = mix(c, pattern * (0.35 + 1.4 * light), s_crawl / 100.0);
    return vec4(mix(c, lit, wall), 1.0);
}
