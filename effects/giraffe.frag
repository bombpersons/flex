// Giraffe Necks: each head is lifted off and the bit of neck under the chin is stretched to reach it.

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
    float t = u_video_time;
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        vec2 size = (u_faces[i].zw - u_faces[i].xy) * aspect;
        vec2 chin = 0.5 * (u_faces[i].xy + u_faces[i].zw) * aspect - f.up * size.y * 0.5;
        float lift = size.y * s_lift / 100.0 * 1.4 * (1.0 + 0.3 * u_bass * s_react / 100.0) * seen;
        float swing = sin(t * 1.3 + float(i) * 2.0) * s_sway / 100.0 * size.x * 0.9;
        vec2 rel = p - chin;
        float up = dot(rel, f.up);
        float along = clamp(up / max(lift, 1e-4), 0.0, 1.0);
        float bent = swing * along * along;
        float x = dot(rel, f.across) - bent;
        float neck = size.x * 0.27;
        // The neck.
        float low = size.y * 0.3;
        if (up > -low && up < lift + size.y * 0.1 && abs(x) < neck) {
            float reach = (up + low) / (lift + low);
            vec2 from = chin + f.across * x + f.up * (-low + reach * low * 0.95);
            float shade = 0.8 + 0.25 * cos(x / neck * 1.57);
            c = mix(c, texture(u_video, from / aspect).rgb * shade, (1.0 - smoothstep(neck * 0.9, neck, abs(x))) * seen);
        }
        // The head, on top of it.
        vec2 lifted = p - f.up * lift - f.across * swing;
        vec2 middle = chin + f.up * size.y * 0.62;
        vec2 l = vec2(dot(lifted - middle, f.across) / (size.x * 0.8), dot(lifted - middle, f.up) / (size.y * 0.78));
        float head = 1.0 - smoothstep(0.86, 1.0, length(l));
        if (head > 0.0 && lift > 0.0) c = mix(c, texture(u_video, lifted / aspect).rgb, head * seen);
    }
    return vec4(c, 1.0);
}
