// Anime Eyes: draws an eye over each real one, the way they're inked: a tall iris shaded dark at
// the top and glowing at the bottom, a heavy winged upper lash and big painted highlights. They
// narrow as a head turns, look where the head's facing, and blink now and then.

float hash(float n) {
    return fract(sin(n * 91.345) * 47453.5453);
}

vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

// The colour of a face's eyes.
vec3 iris_colour(int face) {
    if (s_colour > 0.5) {
        if (s_colour < 1.5) return vec3(0.12, 0.42, 1.0);
        if (s_colour < 2.5) return vec3(0.62, 0.2, 1.0);
        if (s_colour < 3.5) return vec3(0.08, 0.8, 0.45);
        if (s_colour < 4.5) return vec3(1.0, 0.62, 0.08);
        return vec3(1.0, 0.13, 0.25);
    }
    return mix(hue(hash(float(face) + 2.0)), vec3(0.2, 0.45, 1.0), 0.25);
}

float inside_of(float distance_in, float px) {
    return clamp(distance_in / px + 0.5, 0.0, 1.0);
}

// A four-pointed glint.
float glint(vec2 q, float size) {
    q = abs(q) / max(size, 1e-4);
    return 1.0 - smoothstep(0.85, 1.0, sqrt(q.x) + sqrt(q.y));
}

// One eye, with x running outwards from the nose and y up, in units of the eye. `side` is which
// way outwards is on screen, so the highlights land on the same side of both eyes.
vec4 draw_eye(vec2 l, vec2 look, float open, vec3 colour, float side, float pulse, float px) {
    vec2 tall = vec2(1.0, l.y > 0.0 ? max(1.08 * open, 0.03) : 0.8);
    float round_d = length(l / tall);
    float inside = inside_of(1.0 - round_d, px);

    vec3 white = vec3(0.98, 0.97, 1.0);
    white = mix(white, vec3(0.62, 0.66, 0.88), smoothstep(0.35, 1.0, l.y / max(tall.y, 0.05)) * step(0.0, l.y));

    vec2 centre = vec2(look.x, look.y + 0.06);
    vec2 ir = (l - centre) / vec2(0.74, 1.1);
    float d = length(ir);
    vec3 dark = colour * 0.2, mid = colour * 0.8, bright = mix(colour, vec3(1.0), 0.4) * 1.25;
    vec3 iris = mix(bright, mid, smoothstep(-0.95, -0.1, ir.y));
    iris = mix(iris, dark, smoothstep(-0.2, 0.7, ir.y));
    float angle = atan(ir.y, ir.x);
    iris *= 0.86 + 0.28 * (0.5 + 0.5 * sin(angle * 23.0 + 1.3 * sin(angle * 5.0))) * smoothstep(0.25, 0.9, d);
    iris += bright * 0.55 * smoothstep(0.55, 0.92, d) * (1.0 - smoothstep(-0.8, -0.1, ir.y));
    iris = mix(iris, dark * 0.45, smoothstep(0.84, 0.98, d));
    float pupil = length((l - centre - vec2(0.0, 0.1)) / (vec2(0.27, 0.44) * (1.0 + 0.12 * (pulse - 1.0))));
    iris = mix(iris, vec3(0.02, 0.01, 0.05), 1.0 - smoothstep(0.92, 1.08, pupil));

    vec3 eye = mix(white, iris, inside_of(1.0 - d, px / 0.74));

    float amount = s_sparkle / 100.0;
    vec2 q1 = l - centre - vec2(-0.26 * side, 0.4);
    float highlight = inside_of(1.0 - length(q1 / (vec2(0.27, 0.34) * pulse)), px / 0.27);
    vec2 q2 = l - centre - vec2(0.3 * side, -0.5);
    highlight = max(highlight, inside_of(1.0 - length(q2) / (0.14 * pulse), px / 0.14));
    vec2 q3 = l - centre - vec2(-0.05 * side, -0.68);
    highlight = max(highlight, 0.8 * inside_of(1.0 - length(q3) / 0.07, px / 0.07));
    highlight = max(highlight, glint(q1 + vec2(0.1 * side, -0.1), 0.25 + 0.6 * (pulse - 1.0)));
    eye = mix(eye, vec3(1.0), highlight * amount);

    vec4 out_colour = vec4(eye, inside);

    // Inked lines, darkest at the outside corner.
    vec3 ink = vec3(0.07, 0.04, 0.07);
    float thick = mix(0.14, 0.42, smoothstep(-0.7, 0.85, l.x));
    float upper = l.y > -0.06 ? inside_of(1.0 + thick - round_d, px) * inside_of(round_d - 0.98, px) : 0.0;
    float t = (l.x - 0.82) / 0.72;
    float wing = 0.0;
    if (t > 0.0 && t < 1.0) wing = inside_of(mix(0.17, 0.0, t) - abs(l.y - mix(0.12, 0.42, t)), px);
    float t2 = (l.x - 0.66) / 0.62;
    if (t2 > 0.0 && t2 < 1.0) wing = max(wing, inside_of(mix(0.08, 0.0, t2) - abs(l.y - mix(0.55, 0.92, t2)), px));
    float lower = l.y < 0.0 ? inside_of(1.06 - round_d, px) * inside_of(round_d - 0.985, px) * smoothstep(-0.6, 0.2, l.x) : 0.0;
    float crease = l.y > 0.0 ? 0.55 * inside_of(0.03 - abs(length(l / vec2(1.05, 1.5 * max(open, 0.05))) - 1.0), px) * smoothstep(-0.6, -0.1, l.x) * (1.0 - smoothstep(0.65, 1.0, l.x)) : 0.0;
    float lines = max(max(upper, wing), lower);
    out_colour.rgb = mix(out_colour.rgb, ink, lines);
    out_colour.a = max(out_colour.a, lines);
    out_colour.rgb = mix(out_colour.rgb, vec3(0.3, 0.17, 0.16), crease * (1.0 - out_colour.a));
    out_colour.a = max(out_colour.a, crease);
    return out_colour;
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 p = uv * aspect;
    float pulse = 1.0 + 0.45 * u_beat * s_react / 100.0;
    for (int i = 0; i < u_face_count; i++) {
        float seen = u_face_seen[i];
        if (seen <= 0.0) continue;
        vec2 right = face_point(i, 0) * aspect, left = face_point(i, 1) * aspect;
        vec2 nose = face_point(i, 2) * aspect;
        vec2 eyes = 0.5 * (right + left);
        vec2 mouth = 0.5 * (face_point(i, 3) + face_point(i, 4)) * aspect;
        float height = max(length(eyes - mouth), 0.004);
        vec2 up = (eyes - mouth) / height;
        vec2 across = vec2(up.y, -up.x);
        float frontal = max(length(left - right), height * 0.95);
        float size = frontal * 0.17 * s_size / 100.0;
        float look = clamp(dot(nose - eyes, across) / frontal * 1.3, -0.32, 0.32);

        float open = 1.0;
        if (s_blink > 0.5) {
            float period = 3.2 + 2.6 * hash(float(i) + 7.0);
            float at = fract(u_time / period + hash(float(i))) * period;
            open = max(smoothstep(0.0, 0.07, abs(at - 0.09) - 0.015), 0.03);
        }

        for (int e = 0; e < 2; e++) {
            vec2 eye = e == 0 ? right : left;
            // Side on, only the near eye shows.
            if (length(left - right) < 0.55 * height && abs(dot(eye - nose, across)) < abs(dot((e == 0 ? left : right) - nose, across))) continue;
            float out_x = sign(dot(eye - eyes, across) + (e == 0 ? -1e-5 : 1e-5));
            vec2 outward = across * out_x;
            float squash = clamp(abs(dot(eye - nose, across)) / (0.5 * frontal), 0.35, 1.25);
            vec2 centre = eye + outward * frontal * 0.04 - up * frontal * 0.02;

            // Blush under the eye, hatched the way it's drawn.
            vec2 cheek = p - (centre - up * size * 1.7 + outward * size * 0.25);
            float blush = 1.0 - smoothstep(0.5, 1.0, length(vec2(dot(cheek, across), dot(cheek, up) * 2.1)) / (size * 1.05));
            if (blush > 0.0 && s_blush > 0.0) {
                float hatch = 0.75 + 0.25 * smoothstep(-0.2, 0.2, sin((dot(cheek, across) - dot(cheek, up) * 0.6) / size * 18.0));
                c = mix(c, c * vec3(1.05, 0.72, 0.72) + vec3(0.2, 0.03, 0.08), blush * hatch * seen * s_blush / 100.0);
            }

            // With a mesh, these eyes do what the real ones are doing.
            float eye_open = open;
            vec2 gaze = vec2(look * out_x, 0.0);
            if (meshed(i)) {
                if (s_blink > 0.5) eye_open = max(smoothstep(0.0, 0.7, e == 0 ? u_face_open[i].x : u_face_open[i].y), 0.03);
                vec2 inner = mesh_point(i, e == 0 ? 133 : 362).xy * aspect, outer = mesh_point(i, e == 0 ? 33 : 263).xy * aspect;
                float wide = max(length(outer - inner), 1e-4);
                vec2 from_middle = iris_of(i, e) * aspect - 0.5 * (inner + outer);
                gaze = vec2(clamp(dot(from_middle, outer - inner) / (wide * wide) * 2.4, -0.32, 0.32), clamp(dot(from_middle, up) / wide * 2.0, -0.2, 0.2));
            }

            vec2 rel = p - centre;
            vec2 l = vec2(dot(rel, outward) / (size * squash), dot(rel, up) / size);
            if (dot(l, l) > 6.0) continue;
            float px = 1.6 / (u_resolution.y * size);
            vec4 drawn = draw_eye(l, gaze, eye_open, iris_colour(i), out_x * sign(across.x + 1e-5), pulse, px);
            c = mix(c, drawn.rgb, drawn.a * seen);
        }
    }
    return vec4(c, 1.0);
}
