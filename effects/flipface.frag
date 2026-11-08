// Upside-Down Face: flips the eyes and mouth over where they are, or the whole face.

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

vec3 flipped(vec3 c, vec2 p, vec2 centre, Face f, vec2 half_size, float seen, vec2 aspect) {
    float amount = oval(p, centre, f, half_size) * seen;
    if (amount <= 0.0) return c;
    vec2 rel = p - centre;
    vec2 from = centre + f.across * dot(rel, f.across) - f.up * dot(rel, f.up);
    return mix(c, texture(u_video, from / aspect).rgb, amount);
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    for (int i = 0; i < u_face_count; i++) {
        if (u_face_seen[i] <= 0.0) continue;
        Face f = face_frame(i, aspect);
        if (s_style > 0.5) {
            c = flipped(c, p, 0.5 * (f.eyes + f.mouth), f, vec2(f.spread * 1.35, f.height * 1.35), u_face_seen[i], aspect);
        } else {
            vec2 eye = vec2(f.spread * 0.5, f.spread * 0.36);
            c = flipped(c, p, f.right, f, eye, u_face_seen[i], aspect);
            c = flipped(c, p, f.left, f, eye, u_face_seen[i], aspect);
            c = flipped(c, p, f.mouth, f, vec2(f.spread * 0.75, f.height * 0.5), u_face_seen[i], aspect);
        }
    }
    return vec4(c, 1.0);
}
