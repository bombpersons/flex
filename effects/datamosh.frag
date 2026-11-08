// Datamosh: works out how each block of the picture moved, as a video codec does, then moves last
// frame's output that way instead of showing the new frame, so the wrong picture gets dragged
// about by the right movement. Alpha keeps last frame's real brightness to compare with.

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

float luma(vec3 c) {
    return dot(c, vec3(0.3, 0.59, 0.11));
}

vec4 effect(vec2 uv) {
    vec3 video = texture(u_video, uv).rgb;
    float now = luma(video);
    if (u_frame < 2 || uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(video, now);

    vec2 pixel = 1.0 / u_resolution;
    float block = s_block * u_resolution.y / 720.0;
    vec2 cell = floor(uv * u_resolution / block);
    vec2 middle = (cell + 0.5) * block * pixel;

    // Lucas-Kanade over a few points of the block.
    float xx = 1e-3, yy = 1e-3, xy = 0.0, xt = 0.0, yt = 0.0;
    float reach = block * 0.33;
    for (int i = -1; i <= 1; i += 2) {
        for (int j = -1; j <= 1; j += 2) {
            vec2 q = middle + vec2(float(i), float(j)) * reach * pixel;
            float dx = luma(texture(u_video, q + vec2(reach, 0.0) * pixel).rgb) - luma(texture(u_video, q - vec2(reach, 0.0) * pixel).rgb);
            float dy = luma(texture(u_video, q + vec2(0.0, reach) * pixel).rgb) - luma(texture(u_video, q - vec2(0.0, reach) * pixel).rgb);
            float dt = luma(texture(u_video, q).rgb) - texture(u_prev, q).a;
            xx += dx * dx; yy += dy * dy; xy += dx * dy; xt += dx * dt; yt += dy * dt;
        }
    }
    float det = xx * yy - xy * xy + 1e-5;
    vec2 moved = -vec2(yy * xt - xy * yt, xx * yt - xy * xt) / det * reach * 2.0;
    moved = clamp(moved, vec2(-block * 2.0), vec2(block * 2.0)) * s_smear / 100.0;

    vec3 c = texture(u_prev, uv - moved * pixel).rgb;
    float seed = hash(cell + floor(u_video_time * 8.0));
    float clean = step(seed, u_beat * s_clean / 100.0);
    float back = mix(0.04, 0.002, s_rot / 100.0);
    c = mix(c, video, max(clean, back));
    return vec4(c, now);
}
