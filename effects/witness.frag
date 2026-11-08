// Witness Protection: hides who everyone is.

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
    float scale = s_size / 100.0;
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        if (s_style < 0.5) {
            vec2 rel = p - f.eyes;
            vec2 l = abs(vec2(dot(rel, f.across) / (f.spread * 1.15 * scale), dot(rel, f.up) / (f.spread * 0.34 * scale)));
            if (max(l.x, l.y) < 1.0) c = mix(c, vec3(0.0), seen);
            continue;
        }
        vec2 mid = 0.5 * (f.eyes + f.mouth);
        float amount = oval(p, mid, f, vec2(f.spread * 1.35, f.height * 1.5) * scale) * seen;
        if (amount <= 0.0) continue;
        if (s_style < 1.5) {
            float block = f.height / 5.0;
            vec2 at = (floor(p / block) + 0.5) * block;
            c = mix(c, texture(u_video, at / aspect).rgb, step(0.05, amount));
        } else {
            vec3 sum = vec3(0.0);
            for (int k = 0; k < 16; k++) {
                float a = float(k) * 2.3999632, r = sqrt(float(k) / 16.0) * f.height * 0.35;
                sum += texture(u_video, (p + vec2(cos(a), sin(a)) * r) / aspect).rgb;
            }
            c = mix(c, sum / 16.0, amount);
        }
    }
    return vec4(c, 1.0);
}
