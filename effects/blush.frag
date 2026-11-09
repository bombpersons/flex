// Blush Meter: the loudness drives how red faces get and how much steam comes off them.

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
    float heat = smoothstep(0.1, 0.9, u_level);
    float over = smoothstep(s_boil / 100.0, s_boil / 100.0 + 0.3, u_level);
    vec3 steam_light = vec3(0.0);
    // With a mesh, only skin goes red, and the cheeks stay on the cheeks as the head turns.
    float skin = skin_at(uv) + nose_at(uv);
    if (skin > 0.0) {
        vec2 a = face_uv(uv);
        vec2 cheek = vec2(abs(a.x - 0.5) - 0.225, a.y - 0.46) / vec2(0.14, 0.1);
        float amount = s_blush / 100.0 * skin;
        c = mix(c, c * vec3(1.35, 0.55, 0.5), heat * amount);
        c = mix(c, vec3(0.95, 0.15, 0.2) * (0.5 + dot(c, vec3(0.33))), (1.0 - smoothstep(0.3, 1.0, length(cheek))) * (0.3 + 0.7 * heat) * amount * 0.8);
    }
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        vec2 mid = 0.5 * (f.eyes + f.mouth);
        float amount = s_blush / 100.0 * seen;
        if (!meshed(i)) c = mix(c, c * vec3(1.35, 0.55, 0.5), oval(p, mid, f, vec2(f.spread * 1.2, f.height * 1.45)) * heat * amount);
        for (int e = -1; e <= 1; e += 2) {
            vec2 cheek = f.eyes - f.up * f.height * 0.55 + f.across * float(e) * f.spread * 0.62;
            if (!meshed(i)) c = mix(c, vec3(0.95, 0.15, 0.2), oval(p, cheek, f, vec2(f.spread * 0.36, f.spread * 0.24)) * (0.3 + 0.7 * heat) * amount * 0.8);
            if (over <= 0.0) continue;
            vec2 ear = f.eyes + f.across * float(e) * f.spread * 1.1;
            vec2 rel = p - ear;
            float up = dot(rel, f.up) / f.height;
            if (up < 0.0 || up > 3.0) continue;
            float along = dot(rel, f.across) / f.height - float(e) * up * 0.25;
            float wander = (fbm(vec2(along * 3.0, up * 2.0 - u_time * 1.5) + float(i)) - 0.47) * 0.5;
            float width = 0.09 + up * 0.18;
            float puff = (1.0 - smoothstep(0.0, width, abs(along + wander))) * (1.0 - smoothstep(0.5, 3.0, up)) * smoothstep(0.0, 0.15, up);
            steam_light += vec3(0.85) * puff * over * seen * s_steam / 60.0;
        }
    }
    return vec4(c + steam_light, 1.0);
}
