// Hypno Eyes: a spiral in each eye, with rings of light spreading from it on the beat.

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
vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float t = u_time * s_spin / 100.0;
    vec3 light = vec3(0.0);
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        float radius = f.spread * 0.33 * s_size / 100.0;
        for (int e = 0; e < 2; e++) {
            vec2 eye = e == 0 ? f.right : f.left;
            vec2 rel = (p - eye) / radius;
            float r = length(rel);
            float a = atan(rel.y, rel.x);
            float sp = 0.5 + 0.5 * sin(a * 2.0 + r * 16.0 - t * 5.0);
            vec3 spiral = mix(vec3(0.02), mix(vec3(0.95), hue(r * 0.5 - t * 0.1), s_colour / 100.0), smoothstep(0.4, 0.6, sp));
            c = mix(c, spiral, (1.0 - smoothstep(0.95, 1.05, r)) * seen);
            float rings = pow(0.5 + 0.5 * sin(r * 2.5 - t * 3.0), 6.0) * exp(-r * 0.3);
            light += hue(r * 0.12 + t * 0.05) * rings * seen * smoothstep(1.0, 1.4, r) * (0.05 + 0.5 * u_beat * s_react / 100.0);
        }
    }
    return vec4(c + light, 1.0);
}
