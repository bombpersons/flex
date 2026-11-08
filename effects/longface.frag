// Long Face: everything below the eyes is pulled downwards, and the chest gives way underneath.

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
    vec2 q = p;
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        vec2 rel = p - f.eyes;
        float across = abs(dot(rel, f.across));
        float down = -dot(rel, f.up);
        float wide = f.spread * 1.25;
        float side = 1.0 - smoothstep(0.75, 1.25, across / wide);
        if (side <= 0.0 || down < -f.height * 0.2) continue;
        float stretch = 1.0 + (s_length / 100.0 - 1.0 + u_bass * s_react / 100.0) * seen;
        float face_length = f.height * 1.75;
        float drawn = face_length * stretch;
        float give = f.height * 1.5;
        float from = down < drawn ? down / stretch : mix(face_length, down, smoothstep(0.0, 1.0, (down - drawn) / give));
        if (down >= drawn + give) continue;
        if (down < 0.0) from = down;
        q += f.up * (down - from) * side;
    }
    return vec4(texture(u_video, q / aspect).rgb, 1.0);
}
