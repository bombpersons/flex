// Reaction-Diffusion Skin: a one-number pattern kept in the alpha of u_prev. Each point is pushed
// up by what's close around it and down by what's further out, which settles into stripes and
// spots. Bright parts of the picture favour one, dark parts the other.

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

float around(vec2 uv, float radius, float turn) {
    float sum = 0.0;
    for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.7853982 + turn;
        sum += texture(u_prev, uv + vec2(cos(a), sin(a)) * radius / u_resolution).a;
    }
    return sum / 8.0;
}

vec4 effect(vec2 uv) {
    vec3 c = texture(u_video, uv).rgb;
    float pattern;
    if (u_frame < 3) {
        pattern = noise(uv * u_resolution / (s_size * u_resolution.y / 720.0 * 1.7));
    } else {
        float size = s_size * u_resolution.y / 720.0;
        float close = 0.5 * (texture(u_prev, uv).a + around(uv, size * 0.6, 0.3));
        float far = 0.5 * (around(uv, size * 1.5, 0.0) + around(uv, size * 2.4, 0.4));
        float luma = dot(c, vec3(0.3, 0.59, 0.11));
        float push = (close - far) * 7.0 + (luma - 0.45) * 0.45 * s_follow / 100.0 + (u_bass - 0.3) * 0.03;
        pattern = clamp(texture(u_prev, uv).a + clamp(push, -1.0, 1.0) * 0.07, 0.0, 1.0);
    }
    vec3 ink;
    float a = smoothstep(0.35, 0.65, pattern);
    if (s_colours < 0.5) {
        ink = mix(c * 0.2, c * 1.5 + 0.05, a);
    } else if (s_colours < 1.5) {
        ink = mix(vec3(0.7, 0.12, 0.25), vec3(1.0, 0.72, 0.45), a) * (0.5 + 0.8 * c);
    } else {
        ink = vec3(a) * (0.75 + 0.3 * c);
    }
    vec3 shown = mix(c, ink, s_strength / 100.0);
    if (uv != clamp(uv, u_video_rect.xy, u_video_rect.zw)) shown = vec3(0.0);
    return vec4(shown, pattern);
}
