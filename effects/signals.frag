// Signals: draws each of the inputs on its own. See EFFECTS.md for what they are.

vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (s_view < 0.5) {
        vec2 v = flow_at(uv) * vec2(u_resolution.x / u_resolution.y, 1.0);
        float speed = length(v);
        return vec4(hue(atan(v.y, v.x) / 6.2831853) * smoothstep(0.0, 0.35, speed), 1.0);
    }
    if (s_view < 1.5) return vec4(history_at(uv, 0.5), 1.0);
    if (s_view < 2.5) return vec4(plate_at(uv), 1.0);
    if (s_view < 3.5) return vec4(vec3(plate_known(uv)), 1.0);
    if (s_view < 4.5) {
        int who = who_at(uv);
        float person = smoothstep(0.4, 0.7, person_at(uv));
        vec3 tint = who < 0 ? vec3(0.5) : hue(float(who) * 0.17);
        return vec4(mix(c * 0.4, tint * (0.3 + dot(c, vec3(0.8))), person), 1.0);
    }
    if (uv.y < 0.12) {
        int k = int(clamp(uv.x * 4.0, 0.0, 3.0));
        return vec4(u_palette[k], 1.0);
    }
    if (uv.x > 0.92) {
        float bar = uv.x > 0.973 ? u_brightness : (uv.x > 0.947 ? u_colourfulness : 0.5 + 0.5 * u_warmth);
        return vec4(vec3(step(uv.y, bar)), 1.0);
    }
    return vec4(mix(c, vec3(1.0, 0.1, 0.1), u_cut), 1.0);
}
