// Masks: tints the picture by what's where, for checking it and for working on an effect.
//
// s_show: 0 what's where, 1 people's skin, hair and clothes.

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec3 tint;
    float amount;
    if (s_show > 0.5) {
        float skin = body_skin_at(uv), hair = hair_at(uv), clothes = clothes_at(uv);
        tint = skin * vec3(1.0, 0.75, 0.6) + hair * vec3(0.55, 0.3, 0.1) + clothes * vec3(0.7, 0.25, 0.9);
        amount = clamp(skin + hair + clothes, 0.0, 1.0) * 0.7;
    } else {
        vec4 m = masks_at(uv);
        vec4 more = more_masks_at(uv, 0);
        float window = window_at(uv);
        tint = m.r * vec3(1.0, 0.2, 0.2) + m.g * vec3(0.25, 0.45, 1.0) + m.b * vec3(0.25, 0.9, 0.9)
            + more.r * vec3(0.6, 0.45, 0.25) + more.g * vec3(0.8, 0.8, 0.5) + more.b * vec3(0.1, 0.3, 1.0)
            + more.a * vec3(0.25, 0.85, 0.25) + window * vec3(1.0, 0.85, 0.3);
        amount = clamp(m.r + m.g + m.b + more.r + more.g + more.b + more.a + window, 0.0, 1.0) * 0.55;
    }
    return vec4(mix(c, tint, amount), 1.0);
}
