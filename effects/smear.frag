// Time Smear: each pixel keeps more of the last frame the further away it is, and the colours
// drift apart as they fall behind.

vec4 effect(vec2 uv) {
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    float far = 1.0 - smoothstep(0.15, 0.75, depth_at(uv));
    vec2 centre = 0.5 * (u_video_rect.xy + u_video_rect.zw);
    vec2 drift = (uv - centre) * 0.004 * s_split / 100.0 * far;
    vec3 past = vec3(texture(u_prev, uv - drift).r, texture(u_prev, uv).g, texture(u_prev, uv + drift).b);
    vec3 now = texture(u_video, uv).rgb;
    float keep = far * (0.6 + 0.395 * s_past / 100.0);
    return vec4(mix(now, past, keep), 1.0);
}
