// Ego Death: wherever the last frame differs from this one, it's kept, turned and recoloured,
// so anything that moves leaves ghosts. People are lit up from inside and feed the ghosts.

vec3 palette(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec3 hue(vec3 c, float a) {
    vec3 k = vec3(0.57735);
    return c * cos(a) + cross(k, c) * sin(a) + k * dot(k, c) * (1.0 - cos(a));
}

vec4 effect(vec2 uv) {
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec2 centre = 0.5 * (u_video_rect.xy + u_video_rect.zw);
    float kick = 1.0 + (u_bass + u_beat) * s_react / 100.0;
    float turn = 0.006 * kick;
    vec2 d = uv - centre;
    vec2 q = centre + mat2(cos(turn), -sin(turn), sin(turn), cos(turn)) * d * (1.0 - 0.008 * kick);
    vec3 past = hue(texture(u_prev, q).rgb, 0.09 * kick);
    vec3 now = texture(u_video, uv).rgb;
    float person = smoothstep(0.35, 0.7, person_at(uv));
    float differs = smoothstep(0.1, 0.35, length(past - now));
    float keep = differs * (0.75 + 0.245 * s_trails / 100.0) * step(0.001, s_trails);
    vec3 lit = mix(now, palette(dot(now, vec3(0.33)) * 2.0 + u_time * 0.25) * 1.3, 0.6);
    vec3 world = mix(now, past, keep);
    return vec4(mix(world, lit, person), 1.0);
}
