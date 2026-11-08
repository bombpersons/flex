// Motion Paint: colour from the direction things are moving in, strength from how fast, carried
// along with the movement frame after frame.

vec3 hue(float t) {
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    vec2 v = flow_at(uv);
    vec2 square = v * vec2(u_resolution.x / u_resolution.y, 1.0);
    float speed = smoothstep(0.03, 0.5 / (s_gain / 100.0), length(square));
    vec3 paint = hue(atan(square.y, square.x) / 6.2831853 + u_time * 0.03) * speed * (1.0 + u_bass * s_react / 50.0);
    vec3 past = u_frame < 2 ? vec3(0.0) : texture(u_prev, uv - v * u_dt).rgb * s_trail / 100.0;
    vec3 shown = max(paint + c * speed * s_film_moving / 100.0, past);
    return vec4(max(shown, c * s_film / 100.0), 1.0);
}
