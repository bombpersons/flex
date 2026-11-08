// Spinning Features: each eye and mouth is a disc of picture turning on the spot.

struct Face {
    vec2 right, left, eyes, nose, mouth, up, across;
    float spread, height;
};

// A face's points with the screen's shape taken out, and its own up and across.
Face face_frame(int i, vec2 aspect) {
    Face f;
    f.right = face_point(i, 0) * aspect;
    f.left = face_point(i, 1) * aspect;
    f.eyes = 0.5 * (f.right + f.left);
    f.nose = face_point(i, 2) * aspect;
    f.mouth = 0.5 * (face_point(i, 3) + face_point(i, 4)) * aspect;
    f.height = max(length(f.eyes - f.mouth), 0.004);
    f.up = (f.eyes - f.mouth) / f.height;
    f.across = vec2(f.up.y, -f.up.x);
    f.spread = max(length(f.left - f.right), f.height * 0.55);
    return f;
}

// How much of a soft oval `p` is in.
float oval(vec2 p, vec2 centre, Face f, vec2 half_size) {
    vec2 rel = p - centre;
    return 1.0 - smoothstep(0.7, 1.0, length(vec2(dot(rel, f.across), dot(rel, f.up)) / half_size));
}
vec2 rotate(vec2 p, float a) {
    float c = cos(a), s = sin(a);
    return vec2(c * p.x - s * p.y, s * p.x + c * p.y);
}

vec3 spun(vec3 c, vec2 p, vec2 centre, Face f, vec2 half_size, float angle, float seen, vec2 aspect) {
    float amount = oval(p, centre, f, vec2(max(half_size.x, half_size.y))) * seen;
    if (amount <= 0.0) return c;
    return mix(c, texture(u_video, (centre + rotate(p - centre, angle)) / aspect).rgb, amount);
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_video_time * s_speed / 100.0 + u_beat * s_react / 100.0;
    float scale = s_size / 100.0;
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        c = spun(c, p, f.right, f, vec2(f.spread * 0.42) * scale, t * 2.0, seen, aspect);
        c = spun(c, p, f.left, f, vec2(f.spread * 0.42) * scale, -t * 2.0, seen, aspect);
        c = spun(c, p, f.mouth, f, vec2(f.spread * 0.62) * scale, t * 1.3, seen, aspect);
    }
    return vec4(c, 1.0);
}
