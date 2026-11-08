// Invisible Man: keeps what's behind people in its own previous output, updating it only where
// nobody is. Alpha says how recently each point was seen clear; once that's run out, as it has
// after a cut, the room is guessed from the picture around the person.

float someone(vec2 uv) {
    float most = person_at(uv);
    for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.7853982;
        most = max(most, person_at(uv + vec2(cos(a), sin(a)) * vec2(0.012 * u_resolution.y / u_resolution.x, 0.012)));
    }
    return most;
}

vec3 guess(vec2 uv) {
    vec3 sum = vec3(0.0);
    float total = 1e-3;
    for (int r = 1; r <= 4; r++) {
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.7853982 + float(r);
            vec2 q = uv + vec2(cos(a), sin(a)) * vec2(u_resolution.y / u_resolution.x, 1.0) * float(r) * 0.09;
            float clear = 1.0 - smoothstep(0.1, 0.4, person_at(q));
            sum += texture(u_video, clamp(q, u_video_rect.xy, u_video_rect.zw)).rgb * clear;
            total += clear;
        }
    }
    return sum / total;
}

vec4 effect(vec2 uv) {
    vec3 video = texture(u_video, uv).rgb;
    float cover = someone(uv);
    float clear = 1.0 - smoothstep(0.08, 0.3, cover);
    // Until there are masks, nothing is known to be clear of people. What shuffle leaves in
    // u_prev for the first frame is another effect's picture, so that doesn't count either.
    if (u_has_masks < 0.5 || u_frame < 2) return vec4(video, 0.0);
    if (clear >= 1.0) return vec4(video, 1.0);

    float wave = s_shimmer / 100.0 * 0.0025;
    vec2 ripple = vec2(sin(uv.y * 70.0 + u_time * 5.0), cos(uv.x * 60.0 - u_time * 4.0)) * wave;
    float rim = smoothstep(0.08, 0.3, cover) * (1.0 - smoothstep(0.3, 0.8, cover));
    vec4 kept = texture(u_prev, uv + ripple * rim);
    float fresh = max(kept.a - 1.0 / (60.0 * s_memory), 0.0);
    vec3 room = kept.rgb;
    float known = smoothstep(0.0, 0.25, fresh);
    if (known < 1.0) room = mix(guess(uv), room, known);
    return vec4(mix(room, video, clear), mix(fresh, 1.0, clear));
}
