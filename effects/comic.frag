// Comic Book People: flat, punchy colour printed as dots of cyan, magenta, yellow and black at the
// usual screen angles, with a heavy black line round people and their features.

float dots(vec2 p, float amount, float angle, float size) {
    float c = cos(angle), s = sin(angle);
    vec2 q = mat2(c, -s, s, c) * p / size;
    float d = length(fract(q) - 0.5);
    return 1.0 - smoothstep(-0.08, 0.08, d - 0.72 * sqrt(clamp(amount, 0.0, 1.0)));
}

float luma(vec2 uv) {
    return dot(texture(u_video, uv).rgb, vec3(0.3, 0.59, 0.11));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 1.0);
    float mask = person_at(uv);
    float person = smoothstep(0.42, 0.6, mask);
    vec2 e = 2.0 / u_resolution * u_resolution.y / 720.0;
    float rim = smoothstep(0.32, 0.42, mask) * (1.0 - smoothstep(0.42, 0.56, mask));
    if (person + rim <= 0.0) return vec4(c, 1.0);
    // Film is dark for print, so it's lifted a long way first.
    vec3 lifted = pow(clamp(c, 0.0, 1.0), vec3(0.5));
    vec3 flat_colour = clamp((floor(lifted * 5.0 + 0.5) / 5.0 - 0.45) * s_punch / 100.0 + 0.45, 0.0, 1.0);
    float size = s_dots * u_resolution.y / 720.0 * (1.0 + 0.25 * u_beat * s_react / 100.0);
    vec2 p = uv * u_resolution;
    float black = 1.0 - max(flat_colour.r, max(flat_colour.g, flat_colour.b));
    vec3 ink = (1.0 - flat_colour - black) / max(1.0 - black, 0.01);
    vec3 paper = vec3(0.98, 0.95, 0.88);
    paper *= 1.0 - vec3(0.95, 0.1, 0.05) * dots(p, ink.r, 0.2618, size);
    paper *= 1.0 - vec3(0.05, 0.9, 0.1) * dots(p, ink.g, 1.309, size);
    paper *= 1.0 - vec3(0.02, 0.08, 0.9) * dots(p, ink.b, 0.0, size);
    paper *= 1.0 - dots(p, max(black - 0.35, 0.0) * 1.3, 0.7854, size);
    float dx = luma(uv + vec2(e.x, 0.0)) - luma(uv - vec2(e.x, 0.0));
    float dy = luma(uv + vec2(0.0, e.y)) - luma(uv - vec2(0.0, e.y));
    float line = smoothstep(0.1, 0.22, length(vec2(dx, dy))) * s_outline / 100.0;
    vec3 printed = mix(paper, vec3(0.02), line);
    vec3 shown = mix(c, printed, person);
    return vec4(mix(shown, vec3(0.0), rim * s_outline / 70.0), 1.0);
}
