// Colour Thief: takes the colour out of everything on one side of people's outlines.

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) return vec4(c, 1.0);
    float person = smoothstep(0.4, 0.65, person_at(uv));
    float colourful = s_who < 0.5 ? person : 1.0 - person;
    float luma = dot(c, vec3(0.3, 0.59, 0.11));
    float boost = s_boost / 100.0 * (1.0 + u_bass * s_react / 100.0);
    vec3 vivid = clamp(vec3(luma) + (c - luma) * boost, 0.0, 1.0);
    return vec4(mix(mix(c, vec3(luma), s_grey / 100.0), vivid, colourful), 1.0);
}
