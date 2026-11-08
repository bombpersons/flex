// Fractal Zoom: the picture holds a smaller copy of itself at a face (or a wandering point), which
// holds another, and so on, all live. Zooming by one whole copy lands back where it started, so
// the fall never ends.

vec3 turned(vec3 c, float a) {
    float s = sin(a), k = cos(a);
    return c * k + cross(vec3(0.57735), c) * s + vec3(0.57735) * dot(vec3(0.57735), c) * (1.0 - k);
}

vec4 effect(vec2 uv) {
    vec2 low = u_video_rect.xy, high = u_video_rect.zw;
    if (uv != clamp(uv, low, high)) return vec4(0.0, 0.0, 0.0, 1.0);
    vec2 centre = 0.5 * (low + high) + vec2(0.08 * sin(u_time * 0.37), 0.06 * cos(u_time * 0.29));
    if (u_face_count > 0) centre = mix(centre, face_point(0, 2), u_face_seen[0]);

    float size = s_window / 100.0;
    float fall = fract(u_time * 0.28 * s_speed / 100.0);
    vec2 q = centre + (uv - centre) * pow(size, fall);
    float level = fall;
    for (int i = 0; i < 16; i++) {
        vec2 deeper = centre + (q - centre) / size;
        if (deeper != clamp(deeper, low, high)) break;
        q = deeper;
        level -= 1.0;
    }
    vec3 c = texture(u_video, q).rgb;

    // A glowing frame round each copy, from how near its edge this point is.
    vec2 room = min(q - low, high - q) / (high - low);
    float near_edge = (1.0 - smoothstep(0.0, 0.03, min(room.x, room.y)));
    c += vec3(1.0, 0.8, 0.45) * near_edge * (0.25 + u_beat * 0.6);
    c = turned(c, level * 0.9 * s_hue / 100.0);
    return vec4(c, 1.0);
}
