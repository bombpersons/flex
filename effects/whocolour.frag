// Everyone's Colour: who_at says which face each bit of person belongs with.

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    float person = smoothstep(0.4, 0.65, person_at(uv));
    int who = who_at(uv);
    vec3 shown = c * (1.0 - s_world / 100.0);
    if (person > 0.0 && who >= 0) {
        float luma = dot(c, vec3(0.3, 0.59, 0.11));
        vec3 tint = hue(float(who) * 0.37 + u_time * 0.1 * s_turn / 100.0) * (0.25 + 1.6 * luma);
        tint *= 1.0 + 0.5 * u_beat * s_react / 100.0;
        shown = mix(shown, mix(c, tint, s_wash / 100.0), person);
    }
    return vec4(shown, 1.0);
}
