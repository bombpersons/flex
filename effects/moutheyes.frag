// Mouth Eyes: the band across the eyes goes where the mouth was, and the mouth where the eyes were.
// On the mesh where there is one, so they stay on the face however it's turned.

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

// Where things are on a face, as face_uv has it.
const vec2 RIGHT_EYE = vec2(0.35, 0.62);
const vec2 LEFT_EYE = vec2(0.65, 0.62);
const vec2 MOUTH = vec2(0.5, 0.305);

// How much of a soft oval on the face `a` is in.
float blob(vec2 a, vec2 centre, vec2 half_size) {
    return 1.0 - smoothstep(0.7, 1.0, length((a - centre) / half_size));
}

// `c` with the face's own pixels at `from` laid over it, by `amount`.
vec3 from_face(vec3 c, int face, vec2 from, float amount) {
    if (amount <= 0.0) return c;
    vec4 skin = face_texture(face, from);
    return mix(c, skin.rgb, skin.a * amount);
}

vec3 moved(vec3 c, vec2 p, vec2 to, vec2 from, Face f, vec2 half_size, float seen, vec2 aspect) {
    float amount = oval(p, to, f, half_size) * seen;
    if (amount <= 0.0) return c;
    return mix(c, texture(u_video, (from + p - to) / aspect).rgb, amount);
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    int there = face_there(uv);
    if (there >= 0 && meshed(there)) {
        vec2 a = face_uv(uv);
        float on = on_face(uv) * u_face_seen[there];
        vec2 eyes = 0.5 * (RIGHT_EYE + LEFT_EYE);
        c = from_face(c, there, a + eyes - MOUTH, blob(a, MOUTH, vec2(0.315, 0.126)) * on);
        c = from_face(c, there, a + MOUTH - eyes, blob(a, eyes, vec2(0.24, 0.15)) * on);
    }
    for (int i = 0; i < u_face_count; i++) {
        if (u_face_seen[i] <= 0.0 || meshed(i)) continue;
        Face f = face_frame(i, aspect);
        vec2 eyes = vec2(f.spread * 1.05, f.spread * 0.42);
        vec2 mouth = vec2(f.spread * 0.8, f.height * 0.48);
        c = moved(c, p, f.mouth, f.eyes, f, eyes, u_face_seen[i], aspect);
        c = moved(c, p, f.eyes, f.mouth, f, mouth, u_face_seen[i], aspect);
    }
    return vec4(c, 1.0);
}
