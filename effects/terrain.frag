// Film Mountains: the depth map as a height field, with a camera ray marched over it. Where the
// ray meets the ground, the colour comes from the film at that spot.

vec4 effect(vec2 uv) {
    vec2 low = u_video_rect.xy, size = u_video_rect.zw - low;
    float t = u_video_time * s_fly / 100.0;
    float aspect = size.x * u_resolution.x / (size.y * u_resolution.y);
    vec2 screen = (uv - 0.5) * vec2(u_resolution.x / u_resolution.y, 1.0);
    float eye = 0.25 + 0.75 * s_tilt / 100.0;
    float tilt = 0.25 + 0.6 * s_tilt / 100.0;
    vec3 from = vec3(0.13 * sin(t * 0.6), eye, -0.9 + 0.15 * sin(t * 0.4));
    vec3 dir = normalize(vec3(screen.x, screen.y - tilt, 1.0));
    float peak = 0.45 * s_height / 100.0 * (1.0 + 0.3 * u_bass * s_react / 100.0);
    float dist = 0.0, last_h = 0.0, last_d = 0.0;
    for (int i = 0; i < 90; i++) {
        vec3 at = from + dir * dist;
        vec2 ground = vec2(at.x / aspect + 0.5, at.z * 0.5 + 0.5);
        if (at.z > 1.05 || at.y < -0.05) break;
        if (ground == clamp(ground, vec2(0.0), vec2(1.0))) {
            float h = depth_at(low + ground * size) * peak;
            if (at.y <= h) {
                // Between this step and the last, for a smooth surface.
                float off_now = h - at.y, off_then = last_d;
                float back = off_now / max(off_now + off_then, 1e-4);
                vec3 hit = from + dir * (dist - back * max(0.012, dist * 0.03));
                vec2 g = vec2(hit.x / aspect + 0.5, hit.z * 0.5 + 0.5);
                vec3 c = texture(u_video, low + g * size).rgb;
                float e = 0.006;
                float slope = depth_at(low + (g + vec2(e, 0.0)) * size) - depth_at(low + (g - vec2(e, 0.0)) * size) + depth_at(low + (g + vec2(0.0, e)) * size) - depth_at(low + (g - vec2(0.0, e)) * size);
                c *= clamp(0.82 - slope * peak * 9.0, 0.35, 1.5);
                float haze = 1.0 - exp(-(hit.z + 0.9) * 0.8 * s_fog / 40.0 + 0.5);
                c = mix(c, vec3(0.6, 0.7, 0.82), clamp(haze, 0.0, 0.85) * s_fog / 100.0);
                return vec4(c, 1.0);
            }
            last_d = at.y - h;
        }
        dist += max(0.012, dist * 0.03);
    }
    float up = clamp(dir.y * 2.0 + 0.3, 0.0, 1.0);
    return vec4(mix(vec3(0.75, 0.72, 0.68), vec3(0.2, 0.35, 0.65), up), 1.0);
}
