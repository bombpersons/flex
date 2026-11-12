// Film Palette: each point takes the nearest of the picture's four main colours, like a
// screen print made specially for this shot.

vec4 effect(vec2 uv) {
    vec2 p = uv * vec2(u_resolution.x / u_resolution.y, 1.0);
    vec2 bend = (vec2(noise(p * 40.0), noise(p * 40.0 + 9.0)) - 0.5) * 0.008 * s_wobble / 30.0;
    vec3 c = texture(u_video, uv + bend).rgb;
    int best = 0;
    float least = 1e9;
    for (int k = 0; k < 4; k++) {
        vec3 d = c - u_palette[k];
        float away = dot(d, d);
        if (away < least) { least = away; best = k; }
    }
    return vec4(mix(c, u_palette[best], s_flat / 100.0) * s_bright / 100.0, 1.0);
}
