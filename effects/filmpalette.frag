// Film Palette: each point takes the nearest of the picture's four main colours, like a
// screen print made specially for this shot.

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
