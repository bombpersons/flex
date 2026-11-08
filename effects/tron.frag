// Neon Grid: outlines from jumps in depth and in the picture, coloured by distance, and a grid
// laid out in perspective using depth as how far away each point is.

vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec2 depth_slope(vec2 uv, float reach) {
    vec2 e = reach / u_resolution;
    return vec2(depth_at(uv + vec2(e.x, 0.0)) - depth_at(uv - vec2(e.x, 0.0)), depth_at(uv + vec2(0.0, e.y)) - depth_at(uv - vec2(0.0, e.y)));
}

float luma(vec2 uv) {
    return dot(texture(u_video, uv).rgb, vec3(0.3, 0.59, 0.11));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(0.0, 0.0, 0.0, 1.0);
    float near = depth_at(uv);
    vec2 e = 1.5 / u_resolution * u_resolution.y / 720.0;
    float picture = length(vec2(luma(uv + vec2(e.x, 0.0)) - luma(uv - vec2(e.x, 0.0)), luma(uv + vec2(0.0, e.y)) - luma(uv - vec2(0.0, e.y))));
    float shape = length(depth_slope(uv, 2.0));
    float edge = smoothstep(0.08, 0.25, picture) * 0.8 + smoothstep(0.02, 0.09, shape);
    vec3 tint = hue(s_colour / 360.0 + near * 0.5);
    vec3 out_colour = c * s_film / 100.0;
    out_colour += tint * edge * s_lines / 100.0 * (1.0 + u_beat * s_react / 100.0) * 1.4;

    float z = 1.0 / (0.08 + near);
    float x = (uv.x - 0.5) * z * 2.0;
    vec2 cells = abs(fract(vec2(x, z - u_video_time * 0.5) * 2.0) - 0.5);
    float grid_line = 1.0 - smoothstep(0.0, 0.04 + 0.02 * z, min(cells.x, cells.y));
    out_colour += hue(s_colour / 360.0 + 0.5) * grid_line * s_grid / 100.0 * 0.5 * (1.0 - smoothstep(0.1, 0.75, near)) * (0.5 + u_bass * s_react / 100.0);
    return vec4(out_colour, 1.0);
}
