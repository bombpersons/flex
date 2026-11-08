// Tiny People: the room behind people is remembered, or guessed, as in Invisible Man. Then each
// person with a face is drawn smaller, standing on the lowest point of their own body.

// What's probably behind someone: the picture around them, averaged.
vec3 room_behind(vec2 uv) {
    vec3 sum = vec3(0.0);
    float total = 1e-3;
    for (int r = 1; r <= 3; r++) {
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.7853982 + float(r);
            vec2 q = uv + vec2(cos(a), sin(a)) * vec2(u_resolution.y / u_resolution.x, 1.0) * float(r) * 0.11;
            float clear = 1.0 - smoothstep(0.1, 0.4, person_at(q));
            sum += texture(u_video, clamp(q, u_video_rect.xy, u_video_rect.zw)).rgb * clear;
            total += clear;
        }
    }
    return sum / total;
}

float someone(vec2 uv) {
    float most = person_at(uv);
    for (int i = 0; i < 6; i++) {
        float a = float(i) * 1.0471976;
        most = max(most, person_at(uv + vec2(cos(a), sin(a)) * vec2(0.014 * u_resolution.y / u_resolution.x, 0.014)));
    }
    return most;
}

vec4 effect(vec2 uv) {
    vec3 video = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(video, 1.0);
    float cover = someone(uv);
    float clear = 1.0 - smoothstep(0.08, 0.3, cover);
    float fresh = 0.0;
    vec3 c = video;
    if (u_has_masks < 0.5 || u_frame < 2) {
        fresh = 0.0;
    } else if (clear >= 1.0) {
        fresh = 1.0;
    } else {
        vec4 kept = texture(u_prev, uv);
        fresh = max(kept.a - 1.0 / 600.0, 0.0);
        float known = smoothstep(0.0, 0.25, fresh);
        vec3 room = known < 1.0 ? mix(room_behind(uv), kept.rgb, known) : kept.rgb;
        c = mix(room, video, clear);
        fresh = mix(fresh, 1.0, clear);
    }
    if (u_has_masks < 0.5) return vec4(video, 0.0);
    // Only the room goes back into u_prev, so people are drawn into the colour here and the
    // room is left in alpha's care: whatever isn't a tiny person stays as it was.
    float scale = s_size / 100.0 * (1.0 + 0.12 * u_beat * s_react / 100.0);
    for (int i = 0; i < u_face_count; i++) {
        if (u_face_seen[i] <= 0.0) continue;
        vec2 chin = vec2(0.5 * (u_faces[i].x + u_faces[i].z), u_faces[i].y);
        // Down from the chin to where this person ends.
        float bottom = chin.y;
        for (int s = 1; s <= 10; s++) {
            float y = chin.y - float(s) * 0.07;
            if (y < u_video_rect.y || person_at(vec2(chin.x, y)) < 0.4) break;
            bottom = y;
        }
        vec2 stand = vec2(chin.x, bottom);
        vec2 from = stand + (uv - stand) / mix(1.0, scale, u_face_seen[i]);
        if (from != clamp(from, u_video_rect.xy, u_video_rect.zw)) continue;
        // Keep to this person, when there are several.
        bool theirs = true;
        for (int j = 0; j < u_face_count; j++) {
            if (j != i && abs(from.x - 0.5 * (u_faces[j].x + u_faces[j].z)) < abs(from.x - chin.x)) theirs = false;
        }
        float body = theirs ? smoothstep(0.4, 0.7, person_at(from)) : 0.0;
        if (body > 0.0) c = mix(c, texture(u_video, from).rgb, body);
    }
    return vec4(c, fresh);
}
