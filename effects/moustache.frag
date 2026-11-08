// Moustache Everyone: a shape worked out in the face's own coordinates, with a curl at each end.

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
float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}

float fbm(vec2 p) {
    float sum = 0.0, amount = 0.5;
    for (int i = 0; i < 5; i++) {
        sum += amount * noise(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        amount *= 0.5;
    }
    return sum;
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    vec3 tones[4] = vec3[4](vec3(0.03), vec3(0.2, 0.1, 0.04), vec3(0.62, 0.27, 0.05), vec3(0.6));
    vec3 tone = tones[int(s_colour + 0.5)];
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        vec2 at = mix(f.nose, f.mouth, 0.52);
        float w = f.spread * 0.72 * s_size / 100.0;
        vec2 rel = p - at;
        float x = dot(rel, f.across) / w, y = dot(rel, f.up) / w;
        float side = abs(x);
        float middle = -0.14 + 0.42 * side * side;
        float thick = 0.24 * pow(max(1.0 - side, 0.0), 0.75) + 0.02;
        float hair = (1.0 - smoothstep(thick - 0.02, thick + 0.02, abs(y - middle))) * (1.0 - smoothstep(0.98, 1.02, side));
        // The curl, turning in on the beat.
        float twirl = 0.8 * u_beat * s_react / 100.0;
        vec2 from_curl = vec2(side - 1.0, y - 0.28 - 0.14);
        float r = length(from_curl);
        float a = atan(from_curl.y, from_curl.x) + twirl;
        float arc = (1.0 - smoothstep(0.03, 0.05, abs(r - (0.09 + 0.05 * a / 3.14159)))) * (1.0 - smoothstep(0.26, 0.3, r));
        float amount = max(hair, arc) * seen;
        if (amount > 0.0) {
            float strands = 0.75 + 0.5 * noise(vec2(x * 60.0, (y - middle) * 12.0));
            c = mix(c, tone * strands + 0.02, amount);
        }
    }
    return vec4(c, 1.0);
}
