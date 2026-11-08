// Laser Eyes: a beam from each eye, drawn as light added to the picture. Beams start at the
// eye and head outwards and down from the middle of the face.

vec3 tint() {
    if (s_colour < 0.5) return vec3(1.0, 0.12, 0.08);
    if (s_colour < 1.5) return vec3(0.15, 1.0, 0.2);
    return vec3(0.15, 0.4, 1.0);
}

// How close `p` is to the ray from `a` along `d`, as a bright core and a soft glow.
float beam(vec2 p, vec2 a, vec2 d, float width) {
    float along = max(dot(p - a, d), 0.0);
    float away = length(p - a - d * along);
    return 1.0 / (1.0 + pow(away / width, 2.0));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) {
        return vec4(0.0, 0.0, 0.0, 1.0);
    }
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_video_time;
    float flare = 1.0 + 1.5 * u_beat * s_react / 100.0;
    float power = s_power / 100.0;
    vec3 light = vec3(0.0);
    for (int i = 0; i < u_face_count; i++) {
        vec2 right = face_point(i, 0) * aspect;
        vec2 left = face_point(i, 1) * aspect;
        vec2 middle = 0.5 * (right + left);
        float spread = max(length(left - right), 0.004);
        float wander = s_sweep / 100.0 * sin(t * 1.3 + float(i) * 2.1);
        float width = spread * 0.05 * (0.6 + power) * flare;
        for (int e = 0; e < 2; e++) {
            vec2 eye = e == 0 ? right : left;
            vec2 out_of_face = normalize(eye - middle + vec2(0.0, 1e-5));
            vec2 d = normalize(out_of_face * (0.6 + 0.4 * wander) + vec2(0.3 * wander, -0.75));
            float b = beam(p, eye, d, width);
            float flash = 1.0 / (1.0 + pow(length(p - eye) / (spread * 0.14 * flare), 2.0));
            light += (b * power + flash * 0.9 * power) * u_face_seen[i];
        }
    }
    vec3 colour = tint();
    return vec4(c + colour * light * 1.6 * flare + vec3(pow(min(light.r, 1.0), 4.0)) * 0.9, 1.0);
}
