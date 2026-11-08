// Masks: tints the picture by what's where, for checking it and for working on an effect.

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec4 m = masks_at(uv);
    vec3 tint = m.r * vec3(1.0, 0.2, 0.2) + m.g * vec3(0.25, 0.45, 1.0) + m.b * vec3(0.25, 0.9, 0.9);
    float amount = clamp(m.r + m.g + m.b, 0.0, 1.0) * 0.55;
    return vec4(mix(c, tint, amount), 1.0);
}
