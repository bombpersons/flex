// Magic Eye: a pattern repeats across the screen, each repeat shifted by how near that point is.
// Each pixel walks back across the repeats to the first strip to find what it should copy.

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
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(0.0, 0.0, 0.0, 1.0);
    float width = u_resolution.x;
    float strip = floor(width / max(s_strips, 2.0));
    float x = floor(uv.x * width);
    for (int i = 0; i < 40 && x >= strip; i++) {
        float d = depth_at(vec2(x / width, uv.y));
        x -= strip - floor(d * strip * 0.28 * s_depth / 100.0);
    }
    if (s_pattern < 0.5) {
        vec2 cell = vec2(max(x, 0.0), floor(uv.y * u_resolution.y));
        float shimmer = floor(u_time * s_speed / 10.0);
        vec3 c = vec3(hash(cell + shimmer), hash(cell + 3.1 + shimmer), hash(cell + 7.7 + shimmer));
        return vec4(mix(c, vec3(dot(c, vec3(0.33))), 0.3), 1.0);
    }
    return vec4(texture(u_video, vec2(u_video_rect.x + x / strip * (u_video_rect.z - u_video_rect.x), uv.y)).rgb, 1.0);
}
