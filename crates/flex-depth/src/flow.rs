//! How everything in the picture is moving: NeuFlow v2 (Apache 2.0, github.com/neufieldrobotics/
//! NeuFlow_v2, the "mixed" weights) as compute shaders, next to the depth network's. Its last
//! layers, which upscale the result, are left out: it stops at one value per 8 pixels, which is
//! finer than effects use. `tools/flow/convert.py` makes the weights, `tools/flow/reference.py`
//! the reference the tests compare against.

use anyhow::{Context, Result};

use super::{Params, Tensors, read_safetensors, wino_u};

const WEIGHTS: &str = "flow.safetensors";
/// The picture it takes.
pub const WIDTH: usize = 512;
pub const HEIGHT: usize = 288;
/// What comes out, which is one value per 8 pixels.
pub const FLOW_WIDTH: usize = WIDTH / 8;
pub const FLOW_HEIGHT: usize = HEIGHT / 8;
/// Refinement steps. On film two come out about the same as the eight it was trained with.
const STEPS: usize = 2;
/// Channels of the features at each scale.
const FEATURES: usize = 128;
const CONTEXT: usize = 64;
/// In the block a refinement step reads: correlations, context, memory, flow and radius.
const BLOCK: usize = 212;
const AT_CONTEXT: usize = 81;
const AT_MEMORY: usize = 145;
const AT_FLOW: usize = 209;
const AT_RADIUS: usize = 211;
const RADIUS: f32 = 4.0;
const LN_EPS: f32 = 1e-5;
const SLOT: u64 = 256;
/// Dispatches a submission: about 4 ms of work on an RX 580, so that what else is drawing can
/// get in between, while each submission's own cost stays small.
const OPS_PER_SUBMIT: usize = 40;
const NONE: u32 = u32::MAX;
const GELU: u32 = 1;
const ACCUMULATE: u32 = 2;
const LEAKY: u32 = 64;

#[derive(Clone, Copy, Debug)]
enum Kernel {
    Matmul,
    MatmulMem,
    LayerNorm,
    Softmax,
    Conv3,
    Conv,
    Half,
    Double,
    Map,
    Transpose,
    Channels,
    Correlate,
    Expect,
    WinoIn,
    WinoOut,
    Im2col,
    Sum,
}

/// In `Kernel`'s order: entry point, and whether the matrix multiply's B is in `mem`.
const KERNELS: [(&str, bool); 17] = [
    ("matmul", false),
    ("matmul", true),
    ("layer_norm", false),
    ("softmax", false),
    ("conv3x3_16", false),
    ("flow_conv", false),
    ("flow_half", false),
    ("flow_double", false),
    ("flow_map", false),
    ("flow_transpose", false),
    ("flow_channels", false),
    ("flow_correlate", false),
    ("flow_expect", false),
    ("wino_in", false),
    ("wino_out", false),
    ("flow_im2col", false),
    ("flow_sum", false),
];

fn groups(n: usize, of: usize) -> u32 {
    n.div_ceil(of) as u32
}

/// Runs of floats in the weights, as they're pushed.
struct Weights {
    t: Tensors,
    data: Vec<f32>,
}

impl Weights {
    fn raw(&self, name: &str) -> Result<&(Vec<usize>, Vec<f32>)> {
        self.t.get(name).with_context(|| format!("flow weights: no {name}"))
    }

    fn push(&mut self, v: &[f32]) -> u32 {
        let at = self.data.len() as u32;
        self.data.extend_from_slice(v);
        at
    }

    fn tensor(&mut self, name: &str) -> Result<u32> {
        let v = self.raw(name)?.1.clone();
        Ok(self.push(&v))
    }

    fn bias(&mut self, name: &str) -> Result<u32> {
        if self.t.contains_key(name) { self.tensor(name) } else { Ok(NONE) }
    }

    fn conv(&mut self, name: &str) -> Result<(u32, u32)> {
        Ok((self.tensor(&format!("{name}.weight"))?, self.bias(&format!("{name}.bias"))?))
    }

    /// A linear layer's weight as [in][out], for the matrix multiply, and its bias.
    fn linear(&mut self, name: &str) -> Result<(u32, u32, usize, usize)> {
        let (shape, v) = self.raw(&format!("{name}.weight"))?;
        let (out, inp) = (shape[0], shape[1]);
        let mut t = vec![0.0; inp * out];
        for o in 0..out {
            for i in 0..inp {
                t[i * out + o] = v[o * inp + i];
            }
        }
        let w = self.push(&t);
        let b = self.bias(&format!("{name}.bias"))?;
        Ok((w, b, inp, out))
    }
}

/// The dispatches for one job, and the memory they use.
#[derive(Default)]
struct Ops {
    list: Vec<(Kernel, Params, [u32; 3])>,
}

impl Ops {
    fn push(&mut self, kernel: Kernel, p: Params, g: [u32; 3]) {
        self.list.push((kernel, p, g));
    }

    /// 3x3, stride 1, padding 1: depth.wgsl's conv3x3_16, though `winograd` replaces them all.
    #[allow(clippy::too_many_arguments)]
    fn conv3(&mut self, a: u32, cin: usize, (h, w): (usize, usize), (weight, bias): (u32, u32), cout: usize, c: u32, flags: u32) {
        let p = Params { m: h as u32, k: cin as u32, n: w as u32, a_off: a, b_off: weight, bias_off: bias, c_off: c, x0: cout as u32, x1: 1, flags, ..Default::default() };
        self.push(Kernel::Conv3, p, [groups(w, 32), groups(h, 16), groups(cout, 16)]);
    }

    /// The larger convolutions, always with leaky ReLU after.
    #[allow(clippy::too_many_arguments)]
    fn conv(&mut self, a: u32, cin: usize, (h, w): (usize, usize), (weight, bias): (u32, u32), cout: usize, c: u32, (size, stride, pad): (usize, usize, usize)) {
        let (oh, ow) = ((h + 2 * pad - size) / stride + 1, (w + 2 * pad - size) / stride + 1);
        let p = Params { m: h as u32, k: cin as u32, n: w as u32, a_off: a, b_off: weight, bias_off: bias, c_off: c, x0: cout as u32, x1: stride as u32, a_bs: size as u32, a_rs: pad as u32, b_bs: oh as u32, b_rs: ow as u32, flags: LEAKY, ..Default::default() };
        self.push(Kernel::Conv, p, [groups(ow, 16), groups(oh, 16), groups(cout, 8)]);
    }

    fn half(&mut self, a: u32, channels: usize, (h, w): (usize, usize), c: u32) {
        let p = Params { m: h as u32, n: w as u32, a_off: a, c_off: c, ..Default::default() };
        self.push(Kernel::Half, p, [groups(w / 2, 16), groups(h / 2, 16), channels as u32]);
    }

    fn double(&mut self, a: u32, channels: usize, (h, w): (usize, usize), c: u32, scale: f32) {
        let p = Params { m: h as u32, n: w as u32, a_off: a, c_off: c, f0: scale, ..Default::default() };
        self.push(Kernel::Double, p, [groups(w * 2, 16), groups(h * 2, 16), channels as u32]);
    }

    /// `count` values: clamp(v * scale + shift, range), then `flags`.
    #[allow(clippy::too_many_arguments)]
    fn map(&mut self, a: u32, count: usize, c: u32, scale: f32, shift: f32, range: (f32, f32), flags: u32) {
        let p = Params { m: count as u32, a_off: a, c_off: c, f0: scale, f1: shift, f2: range.0, f3: range.1, flags, ..Default::default() };
        self.push(Kernel::Map, p, [groups(count, 64), 1, 1]);
    }

    fn copy(&mut self, a: u32, count: usize, c: u32) {
        self.map(a, count, c, 1.0, 0.0, (f32::MIN, f32::MAX), 0);
    }

    fn zero(&mut self, count: usize, c: u32) {
        self.map(c, count, c, 0.0, 0.0, (f32::MIN, f32::MAX), 0);
    }

    fn relu(&mut self, a: u32, count: usize, c: u32) {
        self.map(a, count, c, 1.0, 0.0, (0.0, f32::MAX), 0);
    }

    fn add(&mut self, a: u32, count: usize, c: u32) {
        self.map(a, count, c, 1.0, 0.0, (f32::MIN, f32::MAX), ACCUMULATE);
    }

    fn transpose(&mut self, a: u32, (m, n): (usize, usize), c: u32) {
        let p = Params { m: m as u32, n: n as u32, a_off: a, c_off: c, ..Default::default() };
        self.push(Kernel::Transpose, p, [groups(m * n, 64), 1, 1]);
    }

    /// C = A x W + bias for rows [rows][k] at `a`, with the first `k` rows of an [in][out] weight
    /// at `w`.
    #[allow(clippy::too_many_arguments)]
    fn linear(&mut self, a: (u32, u32, u32), rows: usize, k: usize, (w, bias): (u32, u32), out: usize, c: u32, flags: u32) {
        let p = Params { m: rows as u32, k: k as u32, n: out as u32, batch: 1, a_off: a.0, a_rs: a.1, a_cs: a.2, b_off: w, b_rs: out as u32, b_cs: 1, c_off: c, c_rs: out as u32, c_cs: 1, bias_off: bias, flags, ..Default::default() };
        self.push(Kernel::Matmul, p, [groups(out, 32), groups(rows, 32), 1]);
    }

    /// C[m][n] = A[m][k] x B[k][n], both in `mem`.
    #[allow(clippy::too_many_arguments)]
    fn multiply(&mut self, a: (u32, u32, u32), b: (u32, u32, u32), (m, k, n): (usize, usize, usize), c: (u32, u32)) {
        let p = Params { m: m as u32, k: k as u32, n: n as u32, batch: 1, a_off: a.0, a_rs: a.1, a_cs: a.2, b_off: b.0, b_rs: b.1, b_cs: b.2, c_off: c.0, c_rs: c.1, c_cs: 1, bias_off: NONE, ..Default::default() };
        self.push(Kernel::MatmulMem, p, [groups(n, 32), groups(m, 32), 1]);
    }

    fn layer_norm(&mut self, a: u32, (rows, n): (usize, usize), (weight, bias): (u32, u32), c: u32) {
        let p = Params { m: rows as u32, n: n as u32, a_off: a, a_rs: n as u32, b_off: weight, bias_off: bias, c_off: c, c_rs: n as u32, f0: LN_EPS, ..Default::default() };
        self.push(Kernel::LayerNorm, p, [rows as u32, 1, 1]);
    }

    fn softmax(&mut self, a: u32, size: usize, scale: f32) {
        let p = Params { m: size as u32, n: size as u32, a_off: a, f0: scale, ..Default::default() };
        self.push(Kernel::Softmax, p, [size as u32, 1, 1]);
    }
}

/// For a large convolution with many input channels (only one in the backbone): how many parts
/// to split its sum into as a matrix multiply, so that there are enough workgroups. The direct
/// kernel has too few on its small picture, each with a long sum.
fn matmul_parts(kernel: Kernel, p: &Params) -> Option<usize> {
    if !matches!(kernel, Kernel::Conv) || p.k < 32 {
        return None;
    }
    let rows = (p.k * p.a_bs * p.a_bs) as usize;
    let tiles = (p.b_bs * p.b_rs).div_ceil(32) as usize * (p.x0 as usize).div_ceil(32);
    let wanted = 144usize.div_ceil(tiles).clamp(1, rows / 256);
    // Parts that divide the sum evenly, each a whole number of the multiply's steps of 16.
    (1..=wanted).rev().find(|n| rows.is_multiple_of(n * 16))
}

/// Floats of scratch a convolution done as a matrix multiply needs.
fn matmul_scratch(p: &Params, parts: usize) -> usize {
    let pixels = (p.b_bs * p.b_rs) as usize;
    (p.k * p.a_bs * p.a_bs) as usize * pixels + parts * p.x0 as usize * pixels
}

/// Floats of scratch Winograd needs for a cin to cout convolution on an h x w picture.
fn wino_scratch(cin: usize, cout: usize, h: usize, w: usize) -> usize {
    16 * (cin + cout) * h.div_ceil(2) * w.div_ceil(2)
}

/// `ops` with its 3x3 convolutions done by Winograd (as depth's are, see `conv3x3_wino` in
/// lib.rs) and its large convolution as a matrix multiply. NeuFlow's pictures are small, so the
/// direct convolutions have too few workgroups to keep the GPU busy, where the matrix
/// multiplies have plenty. Weights are transformed into `wt` (once each, remembered in `done`);
/// the working space is at `scratch`.
fn winograd(ops: Ops, wt: &mut Weights, done: &mut std::collections::HashMap<u32, u32>, scratch: u32) -> Ops {
    let mut out = Ops::default();
    for (kernel, p, g) in ops.list {
        if let Some(parts) = matmul_parts(kernel, &p) {
            // A convolution with many input channels as a matrix multiply, its sum split into
            // `parts` side by side: C [pixel][cout] = cols^T [pixel][cin * taps] W^T.
            let (cin, cout, taps, pixels) = (p.k as usize, p.x0 as usize, (p.a_bs * p.a_bs) as usize, (p.b_bs * p.b_rs) as usize);
            let (rows, each) = (cin * taps, cin * taps / parts);
            let cols = scratch;
            let sums = scratch + (rows * pixels) as u32;
            let n = rows * pixels;
            out.push(Kernel::Im2col, Params { c_off: cols, ..p }, [groups(n, 64).min(65535), groups(n, 64 * 65535), 1]);
            let mul = Params {
                m: pixels as u32,
                k: each as u32,
                n: cout as u32,
                batch: parts as u32,
                a_off: cols,
                a_bs: (each * pixels) as u32,
                a_rs: 1,
                a_cs: pixels as u32,
                b_off: p.b_off,
                b_bs: each as u32,
                b_rs: 1,
                b_cs: rows as u32,
                c_off: sums,
                c_bs: (cout * pixels) as u32,
                c_rs: 1,
                c_cs: pixels as u32,
                bias_off: NONE,
                ..Default::default()
            };
            out.push(Kernel::Matmul, mul, [groups(cout, 32), groups(pixels, 32), parts as u32]);
            let sum = Params { k: cout as u32, m: pixels as u32, x0: parts as u32, a_off: sums, c_off: p.c_off, bias_off: p.bias_off, flags: p.flags, ..Default::default() };
            out.push(Kernel::Sum, sum, [groups(cout * pixels, 64), 1, 1]);
            continue;
        }
        let (h, w, cin, cout) = (p.m as usize, p.n as usize, p.k as usize, p.x0 as usize);
        if !matches!(kernel, Kernel::Conv3) {
            out.push(kernel, p, g);
            continue;
        }
        let u = *done.entry(p.b_off).or_insert_with(|| {
            let raw = wt.data[p.b_off as usize..][..cout * cin * 9].to_vec();
            wt.push(&wino_u(&raw, cout, cin))
        });
        let tiles = h.div_ceil(2) * w.div_ceil(2);
        let (v, m) = (scratch, scratch + (16 * cin * tiles) as u32);
        let input = Params { m: h as u32, k: cin as u32, n: w as u32, a_off: p.a_off, c_off: v, ..Default::default() };
        out.push(Kernel::WinoIn, input, [groups(tiles, 64), cin as u32, 1]);
        // For each of the 16: M^T [tiles][cout] = V^T [tiles][cin] U [cin][cout].
        let mul = Params {
            m: tiles as u32,
            k: cin as u32,
            n: cout as u32,
            batch: 16,
            a_off: v,
            a_bs: (cin * tiles) as u32,
            a_rs: 1,
            a_cs: tiles as u32,
            b_off: u,
            b_bs: (cin * cout) as u32,
            b_rs: cout as u32,
            b_cs: 1,
            c_off: m,
            c_bs: (cout * tiles) as u32,
            c_rs: 1,
            c_cs: tiles as u32,
            bias_off: NONE,
            ..Default::default()
        };
        out.push(Kernel::Matmul, mul, [groups(cout, 32), groups(tiles, 32), 16]);
        let output = Params { m: h as u32, n: w as u32, x0: cout as u32, a_off: m, c_off: p.c_off, bias_off: p.bias_off, flags: p.flags, ..Default::default() };
        out.push(Kernel::WinoOut, output, [groups(tiles, 64), cout as u32, 1]);
    }
    out
}

/// The memory `mem` has, handed out in order.
#[derive(Default)]
struct Memory {
    end: u32,
}

impl Memory {
    fn take(&mut self, floats: usize) -> u32 {
        let at = self.end;
        self.end += floats as u32;
        at
    }
}

/// One picture's input and backbone outputs: everything worked out from it alone.
#[derive(Clone, Copy)]
struct Picture {
    input: u32,
    s16: u32,
    s8: u32,
}

/// A layer of the attention: q, k, v, the merge, its norm, the MLP's two parts and its norm.
struct Layer {
    q: (u32, u32),
    k: (u32, u32),
    v: (u32, u32),
    merge: (u32, u32),
    norm1: (u32, u32),
    mlp0: u32,
    mlp1: u32,
    norm2: (u32, u32),
}

struct Refine {
    conv1: (u32, u32),
    conv2: (u32, u32),
    layers: [(u32, u32); 5],
    conv3: (u32, u32),
    inner: usize,
}

fn refine_weights(wt: &mut Weights, name: &str, inner: usize) -> Result<Refine> {
    let conv1 = wt.conv(&format!("{name}.conv1.conv"))?;
    let conv2 = wt.conv(&format!("{name}.conv2.conv"))?;
    let mut layers = [(0, NONE); 5];
    for (i, l) in layers.iter_mut().enumerate() {
        *l = wt.conv(&format!("{name}.conv_layers.{i}.conv"))?;
    }
    let conv3 = wt.conv(&format!("{name}.conv3"))?;
    Ok(Refine { conv1, conv2, layers, conv3, inner })
}

/// A step of refinement on `block` ([BLOCK][h][w]), using working space `work`.
fn refine(ops: &mut Ops, r: &Refine, corr: u32, block: u32, (h, w): (usize, usize), work: &[u32], scale: f32) {
    let plane = h * w;
    let flow = block + (AT_FLOW * plane) as u32;
    let p = Params { m: h as u32, n: w as u32, a_off: corr, b_off: flow, c_off: block, f0: scale, ..Default::default() };
    ops.push(Kernel::Correlate, p, [groups(w, 16), groups(h, 16), 1]);
    ops.conv3(block, BLOCK, (h, w), r.conv1, 128, work[0], LEAKY);
    ops.conv3(work[0], 128, (h, w), r.conv2, r.inner, work[1], LEAKY);
    for (i, layer) in r.layers.iter().enumerate() {
        ops.conv3(work[1 + i], r.inner, (h, w), *layer, r.inner, work[2 + i], LEAKY);
    }
    ops.conv3(work[6], r.inner, (h, w), r.conv3, 66, work[7], 0);
    ops.map(work[7] + (2 * plane) as u32, 64 * plane, block + (AT_MEMORY * plane) as u32, 1.0, 0.0, (-4.0, 4.0), 0);
    ops.add(work[7], 2 * plane, flow);
}

fn backbone(ops: &mut Ops, wt: &mut Weights, memory: &mut Memory, picture: Picture) -> Result<()> {
    let (w4, h4) = (WIDTH / 2, HEIGHT / 2);
    let (w8, h8) = (WIDTH / 4, HEIGHT / 4);
    let (w16, h16) = (WIDTH / 8, HEIGHT / 8);
    let s8 = (HEIGHT / 8, WIDTH / 8);
    let s16 = (HEIGHT / 16, WIDTH / 16);
    let half1 = memory.take(3 * h4 * w4);
    let half2 = memory.take(3 * h8 * w8);
    let half3 = memory.take(3 * h16 * w16);
    ops.half(picture.input, 3, (HEIGHT, WIDTH), half1);
    ops.half(half1, 3, (h4, w4), half2);
    ops.half(half2, 3, (h8, w8), half3);

    let plane8 = s8.0 * s8.1;
    let plane16 = s16.0 * s16.1;
    let block = |wt: &mut Weights, name: &str| -> Result<((u32, u32), (u32, u32))> { Ok((wt.conv(&format!("backbone.{name}.conv1"))?, wt.conv(&format!("backbone.{name}.conv2"))?)) };

    let (one, two) = block(wt, "block_8_1")?;
    let joined = memory.take(384 * plane8);
    let first = memory.take(256 * plane8);
    ops.conv(half1, 3, (h4, w4), one, 256, first, (8, 4, 2));
    ops.conv3(first, 256, s8, two, 256, joined, LEAKY);

    let (one, two) = block(wt, "block_8_2")?;
    let first = memory.take(128 * plane8);
    ops.conv(half2, 3, (h8, w8), one, 128, first, (6, 2, 2));
    ops.conv3(first, 128, s8, two, 128, joined + (256 * plane8) as u32, LEAKY);

    let (one, two) = block(wt, "block_cat_8")?;
    let first = memory.take(192 * plane8);
    ops.conv3(joined, 384, s8, one, 192, first, LEAKY);
    ops.conv3(first, 192, s8, two, 192, picture.s8, LEAKY);

    let (one, two) = block(wt, "block_16_1")?;
    let joined = memory.take(256 * plane16);
    let first = memory.take(128 * plane16);
    ops.conv(half3, 3, (h16, w16), one, 128, first, (6, 2, 2));
    ops.conv3(first, 128, s16, two, 128, joined, LEAKY);

    let (one, two) = block(wt, "block_8_16")?;
    let first = memory.take(128 * plane16);
    ops.conv(picture.s8, 192, s8, one, 128, first, (6, 2, 2));
    ops.conv3(first, 128, s16, two, 128, joined + (128 * plane16) as u32, LEAKY);

    let (one, two) = block(wt, "block_cat_16")?;
    let first = memory.take(190 * plane16);
    ops.conv3(joined, 256, s16, one, 190, first, LEAKY);
    ops.conv3(first, 190, s16, two, 190, picture.s16, LEAKY);
    Ok(())
}

/// Where the rest of the network keeps things, for a pair of pictures. The same places whichever
/// picture is the current one.
struct Shared {
    tokens: [u32; 2],
    messages: [u32; 2],
    q: u32,
    k: u32,
    v: u32,
    scores: u32,
    merged: u32,
    wide: u32,
    attended: [u32; 2],
    corr16: u32,
    soft: u32,
    block16: u32,
    work16: Vec<u32>,
    merge_in: [u32; 2],
    merge_mid: u32,
    features8: [u32; 2],
    context_in: u32,
    context_mid: u32,
    corr8: u32,
    block8: u32,
    work8: Vec<u32>,
}

struct Tails {
    layers: [Layer; 2],
    channels: (u32, u32),
    merge0: (u32, u32),
    merge2: (u32, u32),
    context0: (u32, u32),
    context2: (u32, u32),
    refine16: Refine,
    refine8: Refine,
}

const D: usize = 192;
const S16: (usize, usize) = (HEIGHT / 16, WIDTH / 16);
const S8: (usize, usize) = (HEIGHT / 8, WIDTH / 8);
const TOKENS: usize = S16.0 * S16.1;
const PIXELS8: usize = S8.0 * S8.1;

/// Everything after the backbone: `now` is the picture being worked out, `last` the one before.
fn rest(ops: &mut Ops, s: &Shared, t: &Tails, now: Picture, last: Picture) {
    let root = (D as f32).sqrt().recip();
    let (m, n) = (TOKENS, D);
    ops.transpose(now.s16, (D, TOKENS), s.tokens[0]);
    ops.transpose(last.s16, (D, TOKENS), s.tokens[1]);
    for layer in &t.layers {
        for side in 0..2 {
            let (src, target) = (s.tokens[side], s.tokens[1 - side]);
            let flat = (src, n as u32, 1);
            let other = (target, n as u32, 1);
            ops.linear(flat, m, n, layer.q, n, s.q, 0);
            ops.linear(other, m, n, layer.k, n, s.k, 0);
            ops.linear(other, m, n, layer.v, n, s.v, 0);
            ops.multiply((s.q, n as u32, 1), (s.k, 1, n as u32), (m, n, m), (s.scores, m as u32));
            ops.softmax(s.scores, m, root);
            ops.multiply((s.scores, m as u32, 1), (s.v, n as u32, 1), (m, m, n), (s.merged, n as u32));
            ops.linear((s.merged, n as u32, 1), m, n, layer.merge, n, s.q, 0);
            ops.layer_norm(s.q, (m, n), layer.norm1, s.merged);
            ops.linear(flat, m, n, (layer.mlp0, NONE), 2 * n, s.wide, 0);
            ops.linear((s.merged, n as u32, 1), m, n, (layer.mlp0 + (n * 2 * n) as u32, NONE), 2 * n, s.wide, ACCUMULATE);
            ops.map(s.wide, m * 2 * n, s.wide, 1.0, 0.0, (f32::MIN, f32::MAX), GELU);
            ops.linear((s.wide, 2 * n as u32, 1), m, 2 * n, (layer.mlp1, NONE), n, s.q, 0);
            ops.layer_norm(s.q, (m, n), layer.norm2, s.messages[side]);
        }
        ops.add(s.messages[0], m * n, s.tokens[0]);
        ops.add(s.messages[1], m * n, s.tokens[1]);
    }
    for side in 0..2 {
        ops.transpose(s.tokens[side], (TOKENS, D), s.attended[side]);
        let p = Params { m: TOKENS as u32, k: D as u32, a_off: s.attended[side], b_off: t.channels.0, bias_off: t.channels.1, c_off: s.attended[side], ..Default::default() };
        ops.push(Kernel::Channels, p, [groups(D * TOKENS, 64), 1, 1]);
    }

    // Matching, and a step of refinement, at the coarser scale.
    let feature = |a: u32, plane: usize| a + (CONTEXT * plane) as u32;
    let f16 = [feature(s.attended[0], TOKENS), feature(s.attended[1], TOKENS)];
    let root16 = (FEATURES as f32).sqrt().recip();
    ops.multiply((f16[0], 1, TOKENS as u32), (f16[1], TOKENS as u32, 1), (TOKENS, FEATURES, TOKENS), (s.corr16, TOKENS as u32));
    ops.copy(s.corr16, TOKENS * TOKENS, s.soft);
    ops.softmax(s.soft, TOKENS, root16);
    let flow16 = s.block16 + (AT_FLOW * TOKENS) as u32;
    let p = Params { m: S16.0 as u32, n: S16.1 as u32, a_off: s.soft, c_off: flow16, ..Default::default() };
    ops.push(Kernel::Expect, p, [groups(TOKENS, 64), 1, 1]);
    let context16 = s.block16 + (AT_CONTEXT * TOKENS) as u32;
    ops.relu(s.attended[0], CONTEXT * TOKENS, context16);
    ops.zero(CONTEXT * TOKENS, s.block16 + (AT_MEMORY * TOKENS) as u32);
    refine(ops, &t.refine16, s.corr16, s.block16, S16, &s.work16, root16);

    // Up to the finer scale.
    let flow8 = s.block8 + (AT_FLOW * PIXELS8) as u32;
    ops.double(flow16, 2, S16, flow8, 2.0);
    for (side, picture) in [now, last].into_iter().enumerate() {
        let joined = s.merge_in[side];
        ops.copy(feature(picture.s8, PIXELS8), FEATURES * PIXELS8, joined);
        ops.double(f16[side], FEATURES, S16, joined + (FEATURES * PIXELS8) as u32, 1.0);
        ops.conv3(joined, 2 * FEATURES, S8, t.merge0, FEATURES, s.merge_mid, GELU);
        ops.conv3(s.merge_mid, FEATURES, S8, t.merge2, FEATURES, s.features8[side], 0);
    }
    ops.relu(now.s8, CONTEXT * PIXELS8, s.context_in);
    ops.double(context16, CONTEXT, S16, s.context_in + (CONTEXT * PIXELS8) as u32, 1.0);
    ops.conv3(s.context_in, 2 * CONTEXT, S8, t.context0, CONTEXT, s.context_mid, GELU);
    ops.conv3(s.context_mid, CONTEXT, S8, t.context2, CONTEXT, s.block8 + (AT_CONTEXT * PIXELS8) as u32, 0);
    ops.multiply((s.features8[0], 1, PIXELS8 as u32), (s.features8[1], PIXELS8 as u32, 1), (PIXELS8, FEATURES, PIXELS8), (s.corr8, PIXELS8 as u32));
    ops.zero(CONTEXT * PIXELS8, s.block8 + (AT_MEMORY * PIXELS8) as u32);
    for _ in 0..STEPS {
        refine(ops, &t.refine8, s.corr8, s.block8, S8, &s.work8, root16);
    }
}

/// A run of dispatches: where its parameters start, and each one's kernel and size.
type Job = (u32, Vec<(Kernel, [u32; 3])>);

/// Works out the movement between each picture and the one before it.
pub struct Flow {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    mem: wgpu::Buffer,
    bind: wgpu::BindGroup,
    readback: wgpu::Buffer,
    /// Each picture's backbone, and what's left with each picture as the current one.
    first: [Job; 2],
    second: [Job; 2],
    pictures: [Picture; 2],
    flow: u32,
    /// Which picture was last filled in, if the next can be compared with it.
    last: Option<usize>,
    adapter: String,
    _gpu: super::Gpu,
    #[cfg(test)]
    shared: Shared,
    #[cfg(test)]
    params: Vec<Params>,
}

impl Flow {
    pub fn new() -> Result<Self> {
        let mut wt = Weights { t: read_safetensors(&super::models::load(WEIGHTS)?)?, data: Vec::new() };
        let mut memory = Memory::default();
        let plane16 = TOKENS;
        let pictures = [0, 1].map(|_| Picture { input: memory.take(3 * HEIGHT * WIDTH), s16: memory.take(D * plane16), s8: memory.take(D * PIXELS8) });

        // The parts that don't change get written once.
        let mut constants: Vec<(u32, Vec<f32>)> = Vec::new();
        for picture in pictures {
            // Position, y then x, measured from the middle.
            let mut pos = vec![0.0; 2 * plane16];
            for y in 0..S16.0 {
                for x in 0..S16.1 {
                    let at = y * S16.1 + x;
                    pos[at] = y as f32 - (S16.0 / 2) as f32;
                    pos[plane16 + at] = x as f32 - (S16.1 / 2) as f32;
                }
            }
            constants.push((picture.s16 + (190 * plane16) as u32, pos));
        }

        let mut before = [Ops::default(), Ops::default()];
        for (ops, picture) in before.iter_mut().zip(pictures) {
            backbone(ops, &mut wt, &mut memory, picture)?;
        }

        let mut layer = |i: usize| -> Result<Layer> {
            let name = format!("cross_attn_s16.layers.{i}");
            let two = |wt: &mut Weights, n: &str| -> Result<(u32, u32)> {
                let (w, b, _, _) = wt.linear(&format!("{name}.{n}"))?;
                Ok((w, b))
            };
            let normal = |wt: &mut Weights, n: &str| -> Result<(u32, u32)> { Ok((wt.tensor(&format!("{name}.{n}.weight"))?, wt.tensor(&format!("{name}.{n}.bias"))?)) };
            Ok(Layer {
                q: two(&mut wt, "q_proj")?,
                k: two(&mut wt, "k_proj")?,
                v: two(&mut wt, "v_proj")?,
                merge: two(&mut wt, "merge")?,
                norm1: normal(&mut wt, "norm1")?,
                mlp0: wt.linear(&format!("{name}.mlp.0"))?.0,
                mlp1: wt.linear(&format!("{name}.mlp.2"))?.0,
                norm2: normal(&mut wt, "norm2")?,
            })
        };
        let layers = [layer(0)?, layer(1)?];
        let tails = Tails {
            layers,
            channels: (wt.tensor("cross_attn_s16.norm.scale")?, wt.tensor("cross_attn_s16.norm.shift")?),
            merge0: wt.conv("merge_s8.0")?,
            merge2: wt.conv("merge_s8.2")?,
            context0: wt.conv("context_merge_s8.0")?,
            context2: wt.conv("context_merge_s8.2")?,
            refine16: refine_weights(&mut wt, "refine_s16", 128)?,
            refine8: refine_weights(&mut wt, "refine_s8", 96)?,
        };

        let work = |memory: &mut Memory, inner: usize, plane: usize| -> Vec<u32> {
            let mut v = vec![memory.take(128 * plane)];
            v.extend((0..6).map(|_| memory.take(inner * plane)));
            v.push(memory.take(66 * plane));
            v
        };
        let row = TOKENS * D;
        let shared = Shared {
            tokens: [memory.take(row), memory.take(row)],
            messages: [memory.take(row), memory.take(row)],
            q: memory.take(row),
            k: memory.take(row),
            v: memory.take(row),
            scores: memory.take(TOKENS * TOKENS),
            merged: memory.take(row),
            wide: memory.take(2 * row),
            attended: [memory.take(row), memory.take(row)],
            corr16: memory.take(TOKENS * TOKENS),
            soft: memory.take(TOKENS * TOKENS),
            block16: memory.take(BLOCK * TOKENS),
            work16: work(&mut memory, 128, TOKENS),
            merge_in: [memory.take(2 * FEATURES * PIXELS8), memory.take(2 * FEATURES * PIXELS8)],
            merge_mid: memory.take(FEATURES * PIXELS8),
            features8: [memory.take(FEATURES * PIXELS8), memory.take(FEATURES * PIXELS8)],
            context_in: memory.take(2 * CONTEXT * PIXELS8),
            context_mid: memory.take(CONTEXT * PIXELS8),
            corr8: memory.take(PIXELS8 * PIXELS8),
            block8: memory.take(BLOCK * PIXELS8),
            work8: work(&mut memory, 96, PIXELS8),
        };
        constants.push((shared.block16 + (AT_RADIUS * TOKENS) as u32, vec![RADIUS; TOKENS]));
        constants.push((shared.block8 + (AT_RADIUS * PIXELS8) as u32, vec![RADIUS; PIXELS8]));
        let mut after = [Ops::default(), Ops::default()];
        for (i, ops) in after.iter_mut().enumerate() {
            rest(ops, &shared, &tails, pictures[i], pictures[1 - i]);
        }
        let scratch_len = before
            .iter()
            .chain(&after)
            .flat_map(|o| &o.list)
            .map(|(k, p, _)| match matmul_parts(*k, p) {
                Some(parts) => matmul_scratch(p, parts),
                None if matches!(k, Kernel::Conv3) => wino_scratch(p.k as usize, p.x0 as usize, p.m as usize, p.n as usize),
                None => 0,
            })
            .max()
            .unwrap_or(0);
        let scratch = memory.take(scratch_len);
        let mut done = std::collections::HashMap::new();
        let before = before.map(|ops| winograd(ops, &mut wt, &mut done, scratch));
        let after = after.map(|ops| winograd(ops, &mut wt, &mut done, scratch));

        let gpu = super::Gpu::shared()?;
        let (device, queue) = (gpu.0.device.clone(), gpu.0.queue.clone());
        let bytes = u64::from(memory.end) * 4;
        let weight_bytes = wt.data.len() as u64 * 4;
        let needed = bytes.max(weight_bytes);
        if needed > gpu.0.most_buffer {
            anyhow::bail!("the GPU allows {} MB buffers; flow needs {} MB", gpu.0.most_buffer >> 20, needed >> 20);
        }
        let storage = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let w = storage("weights", weight_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&w, 0, bytemuck::cast_slice(&wt.data));
        let mem = storage("mem", bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        for (at, values) in &constants {
            queue.write_buffer(&mem, u64::from(*at) * 4, bytemuck::cast_slice(values));
        }
        let total: usize = before.iter().chain(&after).map(|o| o.list.len()).sum();
        let params = storage("params", SLOT * total as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let readback = storage("readback", bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flex-flow"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }),
                entry(1, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }),
                entry(2, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }),
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flex-flow"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &params, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: mem.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("flex-flow"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let code = concat!(include_str!("depth.wgsl"), include_str!("flow.wgsl"));
        let source = wgpu::ShaderModuleDescriptor { label: Some("flow.wgsl"), source: wgpu::ShaderSource::Wgsl(code.into()) };
        // SAFETY: our own shaders, and every offset and size they use comes from the plan above.
        let module = unsafe { device.create_shader_module_trusted(source, wgpu::ShaderRuntimeChecks::unchecked()) };
        let pipelines = KERNELS
            .iter()
            .map(|(entry, b_in_mem)| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions { constants: &[("B_IN_MEM", f64::from(u8::from(*b_in_mem)))], ..Default::default() },
                    cache: None,
                })
            })
            .collect();

        let mut slots = vec![0u8; total * SLOT as usize];
        let mut next = 0u32;
        let mut lay_out = |ops: Ops| -> Job {
            let first = next;
            let list = ops.list.into_iter().map(|(kernel, p, g)| {
                slots[next as usize * SLOT as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(&p));
                next += 1;
                (kernel, g)
            }).collect();
            (first, list)
        };
        let [a, b] = before;
        let (a, b) = (lay_out(a), lay_out(b));
        let [c, d] = after;
        let (c, d) = (lay_out(c), lay_out(d));
        queue.write_buffer(&params, 0, &slots);
        #[cfg(test)]
        let params: Vec<Params> = slots.chunks(SLOT as usize).map(|c| *bytemuck::from_bytes(&c[..std::mem::size_of::<Params>()])).collect();
        let flow = shared.block8 + (AT_FLOW * PIXELS8) as u32;
        Ok(Self { device, queue, pipelines, mem, bind, readback, first: [a, b], second: [c, d], pictures, flow, last: None, adapter: gpu.0.adapter.clone(), _gpu: gpu, #[cfg(test)] shared, #[cfg(test)] params })
    }

    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// The next picture won't follow on from the last: don't compare them.
    pub fn reset(&mut self) {
        self.last = None;
    }

    /// Takes a picture of `WIDTH` × `HEIGHT`, as RGB bytes row by row from the top. Returns how far
    /// each part of it has moved since the last picture, or nothing if there wasn't one: two
    /// planes, x then y, each `FLOW_WIDTH` × `FLOW_HEIGHT` from the top. They're in fractions of
    /// the picture, y downwards, and each place says where it's come from, so what's at
    /// (x, y) now was at (x - dx, y - dy).
    pub fn run(&mut self, rgb: &[u8]) -> Result<Option<Vec<f32>>> {
        self.submit(rgb)?;
        if self.last.is_none() {
            self.last = Some(0);
            return Ok(None);
        }
        let mut flow = self.read(self.flow, 2 * PIXELS8)?;
        for (i, v) in flow.iter_mut().enumerate() {
            // The network says where things go from this picture to the last.
            *v = -*v / if i < PIXELS8 { FLOW_WIDTH } else { FLOW_HEIGHT } as f32;
        }
        Ok(Some(flow))
    }

    fn submit(&mut self, rgb: &[u8]) -> Result<()> {
        anyhow::ensure!(rgb.len() == WIDTH * HEIGHT * 3, "expected {WIDTH}x{HEIGHT} RGB, got {} bytes", rgb.len());
        let now = self.last.map_or(0, |last| 1 - last);
        let pixels = WIDTH * HEIGHT;
        let mut planes = vec![0.0f32; 3 * pixels];
        // Its weights expect BGR, from 0 to 1 at this point.
        for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
            planes[i] = f32::from(p[2]) / 255.0;
            planes[pixels + i] = f32::from(p[1]) / 255.0;
            planes[2 * pixels + i] = f32::from(p[0]) / 255.0;
        }
        self.queue.write_buffer(&self.mem, u64::from(self.pictures[now].input) * 4, bytemuck::cast_slice(&planes));
        self.dispatch(&self.first[now])?;
        if self.last.is_some() {
            self.dispatch(&self.second[now])?;
            self.last = Some(now);
        }
        Ok(())
    }

    fn dispatch(&self, (first, list): &Job) -> Result<()> {
        for (part, ops) in list.chunks(OPS_PER_SUBMIT).enumerate() {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (i, (kernel, g)) in ops.iter().enumerate() {
                    pass.set_pipeline(&self.pipelines[*kernel as usize]);
                    let slot = u64::from(*first + (part * OPS_PER_SUBMIT + i) as u32) * SLOT;
                    pass.set_bind_group(0, &self.bind, &[slot as u32]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                }
            }
            // In parts, so that what else is drawing can get in between, but without waiting
            // for each: the GPU takes them in order.
            self.queue.submit([enc.finish()]);
        }
        Ok(())
    }

    /// For tuning: each dispatch's time (ms) in the backbone and the rest, grouped by what it
    /// is.
    #[cfg(test)]
    fn report(&self, params: &[Params]) {
        let time = |slot: u32, (kernel, g): &(Kernel, [u32; 3]), times: usize| {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for _ in 0..times {
                    pass.set_pipeline(&self.pipelines[*kernel as usize]);
                    pass.set_bind_group(0, &self.bind, &[(u64::from(slot) * SLOT) as u32]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                }
            }
            let started = std::time::Instant::now();
            self.queue.submit([enc.finish()]);
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            started.elapsed().as_secs_f64() * 1000.0
        };
        for (what, (first, list)) in [("backbone", &self.first[0]), ("the rest", &self.second[0])] {
            let mut groups: Vec<(String, f64, usize)> = Vec::new();
            let mut total = 0.0;
            for (i, op) in list.iter().enumerate() {
                let slot = first + i as u32;
                time(slot, op, 1);
                let t = (time(slot, op, 11) - time(slot, op, 1)) / 10.0;
                total += t;
                let p = &params[slot as usize];
                let name = match op.0 {
                    Kernel::Conv3 => format!("conv3 {}->{} at {}x{}", p.k, p.x0, p.n, p.m),
                    Kernel::Conv => format!("conv {}x{} {}->{} at {}x{}", p.a_bs, p.a_bs, p.k, p.x0, p.b_rs, p.b_bs),
                    Kernel::Matmul | Kernel::MatmulMem => format!("{:?} {}x{}x{} batch {}", op.0, p.m, p.k, p.n, p.batch),
                    Kernel::Im2col => format!("Im2col {}x{} {} at {}x{}", p.a_bs, p.a_bs, p.k, p.b_rs, p.b_bs),
                    Kernel::WinoIn => format!("WinoIn {} at {}x{}", p.k, p.n, p.m),
                    Kernel::WinoOut => format!("WinoOut {} at {}x{}", p.x0, p.n, p.m),
                    k => format!("{k:?}"),
                };
                match groups.iter_mut().find(|g| g.0 == name) {
                    Some(g) => {
                        g.1 += t;
                        g.2 += 1;
                    }
                    None => groups.push((name, t, 1)),
                }
            }
            groups.sort_by(|a, b| b.1.total_cmp(&a.1));
            eprintln!("{what}: {} dispatches, {total:.2} ms one by one", list.len());
            for (name, t, n) in groups.iter().take(30) {
                eprintln!("{t:7.3} ms  {n:3}x  {name}");
            }
        }
    }

    fn read(&self, at: u32, len: usize) -> Result<Vec<f32>> {
        let size = len as u64 * 4;
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.mem, u64::from(at) * 4, &self.readback, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..size);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out = bytemuck::cast_slice(&slice.get_mapped_range().map_err(|e| anyhow::anyhow!("{e}"))?).to_vec();
        self.readback.unmap();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compares each stage with PyTorch's (needs a reference file from tools/flow/reference.py, and
    /// a GPU): `FLEX_FLOW_REF=ref.safetensors cargo test -p flex-depth flow_matches_pytorch -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn flow_matches_pytorch() {
        let reference = read_safetensors(&std::fs::read(std::env::var("FLEX_FLOW_REF").expect("FLEX_FLOW_REF")).unwrap()).unwrap();
        let bytes = |name: &str| -> Vec<u8> { reference[name].1.iter().map(|v| *v as u8).collect() };
        let started = std::time::Instant::now();
        let mut net = Flow::new().unwrap();
        eprintln!("loaded in {:?} on {}", started.elapsed(), net.adapter());
        net.run(&bytes("rgb1")).unwrap();
        let flow = net.run(&bytes("rgb0")).unwrap().unwrap();

        let now = net.last.unwrap();
        let s = &net.shared;
        let stages: Vec<(&str, u32)> = vec![
            ("backbone_16", net.pictures[now].s16),
            ("backbone_8", net.pictures[now].s8),
            ("attention", s.attended[0]),
            ("flow_16", s.block16 + (AT_FLOW * TOKENS) as u32),
            ("corr_16", s.block16),
            ("merged", s.features8[0]),
            ("context_8", s.block8 + (AT_CONTEXT * PIXELS8) as u32),
            ("step_1", s.block8 + (AT_FLOW * PIXELS8) as u32),
        ];
        let mut worst = 0.0f32;
        for (name, at) in stages {
            let expected = &reference[name].1;
            let got = net.read(at, expected.len()).unwrap();
            let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let err = got.iter().zip(expected).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            eprintln!("{name:14} max |ref| {scale:10.4}  max error {err:.2e}  relative {:.2e}", err / scale);
            worst = worst.max(err / scale);
        }
        assert!(worst < 2e-3, "worst relative error {worst:.2e}");
        let expected: Vec<f32> = reference["step_1"].1.iter().enumerate().map(|(i, v)| -v / if i < PIXELS8 { FLOW_WIDTH } else { FLOW_HEIGHT } as f32).collect();
        let err = flow.iter().zip(&expected).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 1e-3, "flow out by {err:.2e}");

        // Speed, once warmed up, with the backbone reused from the picture before.
        let runs = 30;
        let (a, b) = (bytes("rgb0"), bytes("rgb1"));
        let started = std::time::Instant::now();
        for i in 0..runs {
            net.run(if i % 2 == 0 { &a } else { &b }).unwrap();
        }
        eprintln!("{:.1} ms a picture ({WIDTH}x{HEIGHT}, {STEPS} steps)", started.elapsed().as_secs_f64() * 1000.0 / f64::from(runs));
        let gpu_only = |net: &Flow| {
            let started = std::time::Instant::now();
            for _ in 0..runs {
                net.dispatch(&net.first[0]).unwrap();
                net.dispatch(&net.second[0]).unwrap();
                net.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            }
            started.elapsed().as_secs_f64() * 1000.0 / f64::from(runs)
        };
        eprintln!("{:.1} ms just the dispatches", gpu_only(&net));
        net.report(&net.params);
    }
}
