// Love Beams: hearts leave each eye along the way the face is pointing, growing and fading as they
// go. Straight at the camera, they float upwards instead.

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
// Which way a face points on screen: length 0 straight at the camera, nearing 1 side on.
vec2 facing(Face f) {
    vec2 aim = vec2(dot(f.nose - 0.5 * (f.eyes + f.mouth), f.across) / f.height, 0.53 - dot(f.eyes - f.nose, f.up) / f.height) / 0.5;
    aim /= max(length(aim), 1.0);
    return f.across * aim.x + f.up * aim.y;
}

float heart(vec2 q) {
    q *= 0.85;
    q.y += 0.1;
    float a = q.x * q.x + q.y * q.y - 1.0;
    return a * a * a - q.x * q.x * q.y * q.y * q.y;
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    vec3 light = vec3(0.0);
    float t = u_time * s_speed / 100.0;
    float count = max(s_hearts, 1.0);
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        Face f = face_frame(i, aspect);
        vec2 aim = u_face_turn[i].w > 0.5 ? u_face_turn[i].xy : facing(f);
        vec2 d = normalize(mix(f.up, normalize(aim + vec2(0.0, 1e-5)), smoothstep(0.1, 0.3, length(aim))));
        for (int e = 0; e < 2; e++) {
            vec2 eye = meshed(i) ? iris_of(i, e) * aspect : (e == 0 ? f.right : f.left);
            for (int h = 0; h < int(count); h++) {
                float along = fract(t * 0.3 + float(h) / count + float(i) * 0.37 + float(e) * 0.5 / count);
                vec2 side = vec2(-d.y, d.x);
                vec2 at = eye + d * along * 1.1 + side * sin(along * 14.0 + float(h) * 2.0) * 0.03;
                float size = f.spread * (0.14 + 0.5 * along) * s_size / 100.0 * (1.0 + 0.4 * u_beat * s_react / 100.0);
                vec2 q = (p - at) / size;
                if (dot(q, q) > 3.0) continue;
                float inside = (1.0 - smoothstep(-0.03, 0.0, heart(q))) * (1.0 - smoothstep(0.04, 0.12, depth_at(uv) - depth_at(eye / aspect)));
                light = max(light, vec3(1.0, 0.16, 0.42) * inside * seen * (1.0 - smoothstep(0.55, 1.0, along)));
            }
        }
    }
    float amount = max(light.r, max(light.g, light.b));
    return vec4(mix(c, light / max(amount, 1e-3), amount), 1.0);
}
