// Perfect Symmetry: folds the head over the line down the middle of the face.

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

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        vec2 mid = 0.5 * (f.eyes + f.mouth);
        float amount = oval(p, mid + f.up * f.height * 0.2, f, vec2(f.spread * 1.45, f.height * 1.9)) * seen;
        if (amount <= 0.0) continue;
        vec2 rel = p - mid;
        float x = dot(rel, f.across);
        // Use the half nearer the camera: the nose points away from it.
        float keep = s_side > 1.5 ? 1.0 : (s_side > 0.5 ? -1.0 : (dot(f.nose - mid, f.across) > 0.0 ? -1.0 : 1.0));
        vec2 from = p + f.across * (keep * abs(x) - x);
        c = mix(c, texture(u_video, from / aspect).rgb, amount);
        // With a mesh the fold follows the middle of the face even when it's turned.
        if (meshed(i) && face_there(uv) == i) {
            vec2 a = face_uv(uv);
            float kept_x = keep < 0.0 ? min(a.x, 1.0 - a.x) : max(a.x, 1.0 - a.x);
            c = mix(c, face_texture(i, vec2(kept_x, a.y)).rgb, on_face(uv) * seen);
        }
    }
    return vec4(c, 1.0);
}
