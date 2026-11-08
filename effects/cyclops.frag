// Cyclops: paints skin from under each eye over it, then puts one big eye between them, made
// from whichever eye is facing the camera.

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
        vec2 hole = vec2(f.spread * 0.42, f.spread * 0.3);
        vec2 cheek = -f.up * f.spread * 0.6;
        float covered = max(oval(p, f.right, f, hole), oval(p, f.left, f, hole)) * seen;
        if (covered > 0.0) c = mix(c, texture(u_video, (p + cheek) / aspect).rgb, covered);

        float zoom = s_size / 100.0 * (1.0 + 0.4 * u_bass * s_react / 100.0);
        vec2 size = vec2(f.spread * 0.42, f.spread * 0.32) * zoom;
        float amount = oval(p, f.eyes, f, size) * seen;
        if (amount > 0.0) {
            vec2 rel = (p - f.eyes) / zoom;
            float nearer = smoothstep(-0.1, 0.1, dot(f.nose - f.eyes, f.across) / f.spread);
            vec3 right = texture(u_video, (f.right + rel) / aspect).rgb;
            vec3 left = texture(u_video, (f.left + rel) / aspect).rgb;
            c = mix(c, mix(left, right, nearer), amount);
        }
    }
    return vec4(c, 1.0);
}
