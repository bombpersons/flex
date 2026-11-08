// YuNet's layers as compute shaders (see face.rs). Pictures are channels first, [c][h][w], in
// `mem`; weights are in `w`, all f32. One thread per output value.

struct Params {
    h: u32, w: u32, cin: u32, cout: u32,
    a: u32, b: u32, c: u32, weights: u32,
    bias: u32, kernel: u32, stride: u32, groups: u32,
    relu: u32, oh: u32, ow: u32, pad: u32,
}

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read_write> mem: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;

// A convolution, padded by half its kernel, either full (groups 1) or depthwise (groups =
// channels). Weights are [cout][cin / groups][kernel][kernel].
@compute @workgroup_size(16, 16)
fn conv(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.ow || id.y >= p.oh) { return; }
    let taps = p.kernel * p.kernel;
    let radius = i32(p.kernel / 2u);
    let plane = p.h * p.w;
    let inputs = p.cin / p.groups;
    let first = select(0u, id.z, p.groups > 1u);
    let cy = i32(id.y * p.stride);
    let cx = i32(id.x * p.stride);
    var acc = w[p.bias + id.z];
    for (var ci = 0u; ci < inputs; ci++) {
        let src = p.a + (first + ci) * plane;
        let weights = p.weights + (id.z * inputs + ci) * taps;
        for (var t = 0u; t < taps; t++) {
            let yy = cy + i32(t / p.kernel) - radius;
            let xx = cx + i32(t % p.kernel) - radius;
            let inside = yy >= 0 && yy < i32(p.h) && xx >= 0 && xx < i32(p.w);
            let at = u32(clamp(yy, 0, i32(p.h) - 1)) * p.w + u32(clamp(xx, 0, i32(p.w) - 1));
            acc += select(0.0, mem[src + at], inside) * w[weights + t];
        }
    }
    if (p.relu != 0u) { acc = max(acc, 0.0); }
    mem[p.c + id.z * p.oh * p.ow + id.y * p.ow + id.x] = acc;
}

// 2 x 2 max pooling.
@compute @workgroup_size(16, 16)
fn pool(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.ow || id.y >= p.oh) { return; }
    let src = p.a + id.z * p.h * p.w + id.y * 2u * p.w + id.x * 2u;
    let v = max(max(mem[src], mem[src + 1u]), max(mem[src + p.w], mem[src + p.w + 1u]));
    mem[p.c + id.z * p.oh * p.ow + id.y * p.ow + id.x] = v;
}

// A plus B doubled in size by repeating its pixels: B is [cin][h][w], A and the result
// twice that size.
@compute @workgroup_size(16, 16)
fn add_up(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= p.ow || id.y >= p.oh) { return; }
    let at = id.z * p.oh * p.ow + id.y * p.ow + id.x;
    let up = p.b + id.z * p.h * p.w + (id.y / 2u) * p.w + id.x / 2u;
    mem[p.c + at] = mem[p.a + at] + mem[up];
}
