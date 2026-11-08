// Anime Eyes: magnifies each eye like a lens and puts shining highlights on it.

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
    vec3 shine = vec3(0.0);
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        float zoom = 1.0 + (s_size / 100.0 - 1.0) * seen;
        float radius = f.spread * 0.42 * zoom;
        for (int e = 0; e < 2; e++) {
            vec2 eye = e == 0 ? f.right : f.left;
            vec2 rel = p - eye;
            float r = length(vec2(dot(rel, f.across), dot(rel, f.up) * 0.85)) / radius;
            if (r >= 1.0) continue;
            q = eye + rel * mix(1.0 / zoom, 1.0, smoothstep(0.55, 1.0, r));
            float pulse = 1.0 + 0.8 * u_beat * s_react / 100.0;
            vec2 big = eye + (f.up * 0.3 - f.across * 0.2) * radius;
            vec2 small = eye + (f.across * 0.25 - f.up * 0.25) * radius;
            shine += vec3(1.0) * seen * s_sparkle / 100.0 * ((1.0 - smoothstep(0.17, 0.21, length(p - big) / (radius * pulse))) + 0.8 * (1.0 - smoothstep(0.07, 0.1, length(p - small) / (radius * pulse))));
        }
    }
    return vec4(texture(u_video, q / aspect).rgb + shine, 1.0);
}
