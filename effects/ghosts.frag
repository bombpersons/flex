// Ghosts: hung through the air of the shot's world, a few to each cell of a grid around where the
// camera is looking, each drifting on a slow loop of its own. Whatever's nearer stands in front.

float hash(vec3 p) {
    p = fract(p * vec3(0.1031, 0.1137, 0.1279));
    p += dot(p, p.yzx + 33.33);
    return fract((p.x + p.y) * p.z);
}

float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(vec3(i, 0.0)), b = hash(vec3(i + vec2(1.0, 0.0), 0.0));
    float c = hash(vec3(i + vec2(0.0, 1.0), 0.0)), d = hash(vec3(i + vec2(1.0, 1.0), 0.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float fbm(vec2 p) {
    float sum = 0.0, amount = 0.5;
    for (int i = 0; i < 4; i++) {
        sum += amount * noise(p);
        p = p * 2.03 + vec2(17.1, 9.2);
        amount *= 0.5;
    }
    return sum;
}

vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

// How tall something looks on screen `away` from the camera, if it's `size` of the picture's height
// 1 away. A shot's world has no real scale, so sizes go by the picture instead.
float screen_size(float size, float away) {
    return size / max(away, 1e-3) * (u_video_rect.w - u_video_rect.y);
}

// Most of a room is further away than this, in a shot's world.
const float LOOKING = 0.5;

// A ghost's shape at `q` (its height is 1, from -0.5 at the hem to 0.5 at the top of the head).
float ghost(vec2 q, float t, float seed) {
    float head = 1.0 - smoothstep(0.15, 0.19, length(q - vec2(0.0, 0.3)));
    float width = 0.17 + 0.16 * clamp(0.3 - q.y, 0.0, 1.0);
    float hem = -0.5 + 0.05 * sin(q.x * 26.0 + t * 3.0 + seed * 6.0);
    float body = (1.0 - smoothstep(width - 0.03, width, abs(q.x))) * smoothstep(hem, hem + 0.06, q.y) * step(q.y, 0.32);
    float shape = max(head, body);
    // Thinning out towards the hem.
    shape *= mix(0.35, 1.0, smoothstep(-0.5, 0.0, q.y)) * (0.75 + 0.25 * noise(q * 9.0 + vec2(0.0, t)));
    // Hollow eyes and an open mouth.
    float eyes = 1.0 - smoothstep(0.03, 0.045, length(vec2(abs(q.x) - 0.065, (q.y - 0.33) * 0.8)));
    float mouth = 1.0 - smoothstep(0.025, 0.04, length(vec2(q.x, (q.y - 0.21) * 0.6)));
    return shape * (1.0 - 0.85 * max(eyes, mouth));
}

vec4 effect(vec2 uv) {
    vec3 film = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw) || u_world_ready <= 0.0) return vec4(film, 1.0);
    vec2 aspect = vec2(u_resolution.x / u_resolution.y, 1.0);
    float shot = mod(u_shot_start * 100.0, 971.0);
    float t = u_video_time;
    float scene_distance = world_distance(depth_at(uv));
    vec3 origin, ahead;
    world_ray(vec2(0.5 * (u_video_rect.xy + u_video_rect.zw)), origin, ahead);
    // In front of the room: never further than most of it, nor than what's in the middle of the
    // picture. Only which cells are near changes with this; where they are doesn't.
    vec2 size = u_video_rect.zw - u_video_rect.xy;
    float room = 0.0;
    for (int i = 0; i < 5; i++) {
        vec2 at = vec2(0.5) + 0.25 * vec2(i == 1 ? -1.0 : i == 2 ? 1.0 : 0.0, i == 3 ? -1.0 : i == 4 ? 1.0 : 0.0);
        room += world_distance(depth_at(u_video_rect.xy + at * size)) / 5.0;
    }
    vec3 looking = origin + normalize(ahead) * min(LOOKING, 0.6 * room);
    float spacing = LOOKING * u_world_lens.y / u_world_lens.x / 1.6;
    vec3 middle = floor(looking / spacing);
    float pale = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int z = -1; z <= 1; z++) {
            vec3 cell = middle + vec3(float(x), 0.0, float(z));
            vec3 seed = cell + vec3(shot, 0.0, shot * 0.5);
            if (hash(seed) > s_count / 100.0) continue;
            float phase = 6.2831853 * hash(seed + 3.1);
            vec3 world = (cell + vec3(0.5, 0.5, 0.5)) * spacing + spacing * vec3(0.35 * sin(t * 0.13 + phase), 0.1 * sin(t * 0.4 + phase), 0.35 * cos(t * 0.11 + phase));
            vec3 from_middle = abs(world - looking) / spacing;
            float kept = 1.0 - smoothstep(0.9, 1.5, max(from_middle.x, from_middle.z));
            if (kept <= 0.0) continue;
            vec3 place = world_to_screen(world);
            if (place.z < 0.15) continue;
            float tall = screen_size(0.22 * s_size / 100.0, place.z);
            vec2 q = (uv - place.xy) * aspect / tall;
            if (abs(q.x) > 0.6 || abs(q.y) > 0.6) continue;
            float seen = smoothstep(place.z * 0.9, place.z * 1.1, scene_distance);
            float flicker = 0.8 + 0.2 * sin(t * 2.3 + phase * 3.0);
            pale = max(pale, ghost(q, t, hash(seed + 7.0)) * seen * kept * flicker);
        }
    }
    float grey = dot(film, vec3(0.3, 0.59, 0.11));
    vec3 cold = mix(film, vec3(grey) * vec3(0.85, 0.95, 1.05), s_chill / 100.0 * u_world_ready);
    vec3 c = mix(cold, vec3(0.82, 0.95, 1.0) * (0.75 + 0.5 * grey), pale * 0.6 * u_world_ready);
    c += vec3(0.5, 0.75, 0.9) * pale * pale * 0.25 * u_world_ready;
    return vec4(c, 1.0);
}
