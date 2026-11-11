//! Pictures redrawn in a style: AnimeGANv3 (Asher Chan's, github.com/TachibanaYoshino/AnimeGANv3),
//! the styles of it that can be downloaded freely, and five paintings' styles from fast neural
//! style (`painting`), as compute shaders (animegan.wgsl, with depth.wgsl's matrix multiply and
//! 3x3 convolution). Their weights are read straight from the ONNX files they're published as,
//! downloaded when first wanted: AnimeGANv3's are free for non-commercial use only, so they
//! aren't shipped with Flex. `tools/animegan/reference.py` makes the reference the tests compare
//! against.
//!
//! The network is an encoder and decoder of the same block over and over: a convolution, then a
//! 1x1 convolution of that, and the first normalised over the whole picture and given the
//! second's mean and spread, then a leaky ReLU. In the middle, at an eighth of the size, is an
//! "external attention" block, and the decoder adds the encoder's blocks back in on the way up.
//! Every convolution mirrors the picture past its edges; none has a bias. Styles differ a little
//! around the attention (see `small`). Ghibli's is another network (`large`).
//!
//! The painting styles are Johnson, Alahi and Fei-Fei's network for fast style transfer, as
//! PyTorch's examples train it (with instance norms), from the ONNX Model Zoo (BSD licence).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail, ensure};

use crate::{ACCUMULATE, Conv, Kernel, LEAKY2, LEAKY2_LAST, Linear, NONE, Ops, Params, REFLECT, Runner, Upload, groups};

/// One of the styles: the file for it, and where it comes from.
#[derive(Debug, PartialEq)]
pub struct Style {
    /// What effects' settings call it.
    pub id: &'static str,
    pub label: &'static str,
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    /// Who made it, and on what terms, for showing wherever it's used.
    pub credit: &'static str,
}

const ANIMEGAN: &str = "AnimeGANv3 by Asher Chan (TachibanaYoshino), free for non-commercial use only";
const PAINTING: &str = "fast neural style (Johnson et al.) from PyTorch's examples and the ONNX Model Zoo, BSD licence";

/// One of the ONNX Model Zoo's fast neural styles, from its Hugging Face copy at `revision`.
macro_rules! painting {
    ($id:literal, $label:literal, $revision:literal, $sha256:literal) => {
        Style {
            id: $id,
            label: $label,
            file: concat!($id, "-9.onnx"),
            url: concat!("https://huggingface.co/onnxmodelzoo/", $id, "-9/resolve/", $revision, "/", $id, "-9.onnx"),
            sha256: $sha256,
            bytes: 6_728_029,
            credit: PAINTING,
        }
    };
}

/// AnimeGANv3's styles that can be downloaded freely (the rest are sold): all but Ghibli are
/// the same network, Ghibli a larger one (see `LARGE`); then the paintings' (see `painting`).
pub const STYLES: &[Style] = &[
    Style {
        id: "hayao",
        label: "Hayao",
        file: "AnimeGANv3_Hayao_36.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Hayao_36.onnx",
        sha256: "95ba7b219073fd5b12f569bc38056ffd3019cf4caf15b1feb9f73d1286c9f69d",
        bytes: 4_232_723,
        credit: ANIMEGAN,
    },
    Style {
        id: "hayao-warm",
        label: "Hayao, warmer",
        file: "animeganv3_H40_model.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.0.3/animeganv3_H40_model.onnx",
        sha256: "21dfdf6344fee2de2a18225ee8750bbd95f6740237bcf413142d8aafa7fa32f1",
        bytes: 4_232_883,
        credit: ANIMEGAN,
    },
    Style {
        id: "hayao-vivid",
        label: "Hayao, more vivid",
        file: "animeganv3_H64_model0.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.0.3/animeganv3_H64_model0.onnx",
        sha256: "420d75c3a990769261a5632fd16ded2d2d71c6933409d25380213c742ff90176",
        bytes: 4_232_978,
        credit: ANIMEGAN,
    },
    Style {
        id: "shinkai",
        label: "Shinkai",
        file: "AnimeGANv3_Shinkai_37.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.1.0/AnimeGANv3_Shinkai_37.onnx",
        sha256: "8dc4d789c44db472447806583c56809e155280225b991bd79f9e2cb20793dcb9",
        bytes: 4_233_084,
        credit: ANIMEGAN,
    },
    Style {
        id: "shinkai-bold",
        label: "Shinkai, bolder",
        file: "AnimeGANv3_Shinkai_40.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3/releases/download/v1.0.3/AnimeGANv3_Shinkai_40.onnx",
        sha256: "7a347fecb4e6c4da6f29bf5128ece909692dad26badec45da892f9ee6f8f9997",
        bytes: 4_232_891,
        credit: ANIMEGAN,
    },
    Style {
        id: "ghibli",
        label: "Ghibli",
        file: "AnimeGANv3_large_Ghibli_c1_e299.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/download/1.0/AnimeGANv3_large_Ghibli_c1_e299.onnx",
        sha256: "ad950eabe2a556424b1cacf0f4b7847777ea44033482e673771da878054c1e37",
        bytes: 6_971_598,
        credit: ANIMEGAN,
    },
    Style {
        id: "sketch",
        label: "Ink sketch",
        file: "AnimeGANv3_PortraitSketch_25.onnx",
        url: "https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference/releases/download/1.0/AnimeGANv3_PortraitSketch_25.onnx",
        sha256: "86de643d216a387ae94d6881fc4197e3427e9dca31a030386a1526f8077c55d8",
        bytes: 4_235_352,
        credit: ANIMEGAN,
    },
    painting!("mosaic", "Mosaic", "9a267ba8540fba8fb12bc78f3770e0fd9c923c56", "fa646dedade881243f8d5a2ceb7de2b93675b21fc24f7482894ac4851a9a0a47"),
    painting!("candy", "Candy", "f24fcf558bd4564360f41cf3b5321975a33f962f", "9d11a3529d1e547da6ae07201d93484dbab2ec0a3614535752c8f40f0fe2968a"),
    painting!("rain-princess", "Rain princess", "205625e4c00565b6c0946814de6165116b271e58", "4162912e6f75fedef6f810ae989b9e10d3d5d43308dab34b027c850cf255e152"),
    painting!("udnie", "Udnie", "ab74109f694dc8117f5e39a8571ac53bf83618f6", "8656b6ce7dec8f22ee13c2d557d6b67bd6f550dde88d0f2e7c9972aeb765cc0d"),
    painting!("pointilism", "Pointillism", "d9c3c80c36d238db4155f3029f996ce62b3c101f", "5ee2b8d4d6bc60a777f54e0fe96a1b717360a004b79d56c67390d4a975b14d98"),
];

pub fn style(id: &str) -> Option<&'static Style> {
    STYLES.iter().find(|s| s.id == id)
}

/// Its sizes must be multiples of this.
pub const MULTIPLE: usize = 8;

/// The size nearest to `pixels` in all with the picture's shape (`aspect`, width over height),
/// in multiples of `MULTIPLE`.
pub fn size_for(aspect: f64, pixels: usize) -> (usize, usize) {
    let height = (pixels as f64 / aspect.max(0.1)).sqrt();
    let round = |v: f64| ((v / MULTIPLE as f64).round() as usize).max(4) * MULTIPLE;
    (round(height * aspect), round(height))
}

// ---- Reading ONNX ---------------------------------------------------------------------------

/// Just enough of protobuf's wire format to read an ONNX file's graph.
struct Proto<'a> {
    b: &'a [u8],
}

enum Field<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Fixed,
}

impl<'a> Proto<'a> {
    fn varint(&mut self) -> Result<u64> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let (&byte, rest) = self.b.split_first().context("ONNX: cut short")?;
            self.b = rest;
            v |= u64::from(byte & 0x7f) << shift;
            if byte < 0x80 {
                return Ok(v);
            }
        }
        bail!("ONNX: a number too long")
    }

    fn next(&mut self) -> Result<Option<(u64, Field<'a>)>> {
        if self.b.is_empty() {
            return Ok(None);
        }
        let key = self.varint()?;
        let field = match key & 7 {
            0 => Field::Varint(self.varint()?),
            1 | 5 => {
                let len = if key & 7 == 1 { 8 } else { 4 };
                ensure!(self.b.len() >= len, "ONNX: cut short");
                self.b = &self.b[len..];
                Field::Fixed
            }
            2 => {
                let len = self.varint()? as usize;
                ensure!(self.b.len() >= len, "ONNX: cut short");
                let (bytes, rest) = self.b.split_at(len);
                self.b = rest;
                Field::Bytes(bytes)
            }
            other => bail!("ONNX: unexpected wire type {other}"),
        };
        Ok(Some((key >> 3, field)))
    }

    /// Each field `number` that holds bytes.
    fn all(b: &'a [u8], number: u64) -> Result<Vec<&'a [u8]>> {
        let mut out = Vec::new();
        let mut p = Proto { b };
        while let Some((n, f)) = p.next()? {
            if let (true, Field::Bytes(bytes)) = (n == number, f) {
                out.push(bytes);
            }
        }
        Ok(out)
    }
}

/// An ONNX graph's float tensors (name → shape, values) and its convolutions' weights (and
/// biases) in order.
struct Graph {
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
    convs: Vec<(String, Option<String>)>,
    /// Each ReduceSum's axes.
    sums: Vec<Vec<u64>>,
    /// Each instance norm's scales and shifts.
    norms: Vec<(String, String)>,
}

fn read_onnx(bytes: &[u8]) -> Result<Graph> {
    let graph = *Proto::all(bytes, 7)?.first().context("ONNX: no graph")?;
    let mut tensors = HashMap::new();
    let mut ints: HashMap<String, Vec<i64>> = HashMap::new();
    for t in Proto::all(graph, 5)? {
        let (mut dims, mut kind, mut name, mut raw, mut floats) = (Vec::new(), 0, String::new(), None, Vec::new());
        let mut p = Proto { b: t };
        while let Some((n, f)) = p.next()? {
            match (n, f) {
                (1, Field::Varint(d)) => dims.push(d as usize),
                (1, Field::Bytes(packed)) => {
                    let mut q = Proto { b: packed };
                    while !q.b.is_empty() {
                        dims.push(q.varint()? as usize);
                    }
                }
                (2, Field::Varint(k)) => kind = k,
                (4, Field::Bytes(packed)) => floats.extend(packed.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b))),
                (8, Field::Bytes(s)) => name = String::from_utf8_lossy(s).into_owned(),
                (9, Field::Bytes(r)) => raw = Some(r),
                _ => {}
            }
        }
        // Whole numbers (type 7): the axes of later versions' sums.
        if kind == 7 {
            if let Some(r) = raw {
                ints.insert(name, r.as_chunks::<8>().0.iter().map(|b| i64::from_le_bytes(*b)).collect());
            }
            continue;
        }
        // Only floats (type 1) are weights.
        if kind != 1 {
            continue;
        }
        let values = match raw {
            Some(r) => r.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            None => floats,
        };
        ensure!(values.len() == dims.iter().product::<usize>(), "ONNX: {name} is the wrong size");
        tensors.insert(name, (dims, values));
    }
    let (mut convs, mut sums, mut norms) = (Vec::new(), Vec::new(), Vec::new());
    for node in Proto::all(graph, 1)? {
        let (mut inputs, mut op, mut axes) = (Vec::new(), String::new(), Vec::new());
        let mut p = Proto { b: node };
        while let Some((n, f)) = p.next()? {
            match (n, f) {
                (1, Field::Bytes(s)) => inputs.push(String::from_utf8_lossy(s).into_owned()),
                (4, Field::Bytes(s)) => op = String::from_utf8_lossy(s).into_owned(),
                // An attribute: its ints (field 8, one at a time or packed).
                (5, Field::Bytes(a)) => {
                    let mut q = Proto { b: a };
                    while let Some((n, f)) = q.next()? {
                        match (n, f) {
                            (8, Field::Varint(v)) => axes.push(v),
                            (8, Field::Bytes(packed)) => {
                                let mut r = Proto { b: packed };
                                while !r.b.is_empty() {
                                    axes.push(r.varint()?);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
        match op.as_str() {
            "Conv" => convs.push((inputs.get(1).context("ONNX: a convolution with no weights")?.clone(), inputs.get(2).cloned())),
            "InstanceNormalization" => match inputs.as_slice() {
                [_, scale, shift] => norms.push((scale.clone(), shift.clone())),
                _ => bail!("ONNX: an instance norm with no scales and shifts"),
            },
            // Since ONNX's opset 13 the axes are an input.
            "ReduceSum" => sums.push(match inputs.get(1).and_then(|a| ints.get(a)) {
                Some(given) if axes.is_empty() => given.iter().map(|&a| a as u64).collect(),
                _ => axes,
            }),
            _ => {}
        }
    }
    Ok(Graph { tensors, convs, sums, norms })
}

// ---- The network ----------------------------------------------------------------------------

/// The network's convolutions in the graph's order: (size, channels in, channels out). Each
/// block is a convolution and the 1x1 after it; four 1x1s in the middle are the attention
/// block's.
const CONVS: [(usize, usize, usize); 31] = [
    (7, 3, 32),
    (1, 32, 32),
    (3, 32, 32),
    (1, 32, 32),
    (3, 32, 64),
    (1, 64, 64),
    (3, 64, 64),
    (1, 64, 64),
    (3, 64, 128),
    (1, 128, 128),
    (3, 128, 128),
    (1, 128, 128),
    (3, 128, 128),
    (1, 128, 128),
    (1, 128, 128),
    (1, 128, 128),
    (1, 128, 128),
    (1, 128, 128),
    (3, 128, 128),
    (1, 128, 128),
    (3, 128, 128),
    (1, 128, 128),
    (3, 128, 64),
    (1, 64, 64),
    (3, 64, 64),
    (1, 64, 64),
    (3, 64, 32),
    (1, 32, 32),
    (3, 32, 32),
    (1, 32, 32),
    (7, 32, 3),
];
/// The attention block's: in, keys, back, and out.
const ATTENTION: usize = 14;
/// The attention's normalising adds this to each sum.
const ATTENTION_EPS: f32 = 1e-9;

/// The "large" network's (Ghibli-c1's), a different one: plain convolutions (with biases, batch
/// norms folded in) and leaky ReLUs down to an eighth of the size, six residual blocks there
/// (1x1, 3x3, 1x1 added back), and up again with the way down's added back in; then one block
/// like the other network's norms, and a 3x3 to the picture.
const LARGE: [(usize, usize, usize); 39] = [
    (5, 3, 32),
    (3, 32, 32),
    (3, 32, 64),
    (3, 64, 64),
    (3, 64, 64),
    (3, 64, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (1, 128, 64),
    (3, 64, 64),
    (1, 64, 128),
    (1, 128, 64),
    (3, 64, 64),
    (1, 64, 128),
    (1, 128, 64),
    (3, 64, 64),
    (1, 64, 128),
    (1, 128, 64),
    (3, 64, 64),
    (1, 64, 128),
    (1, 128, 64),
    (3, 64, 64),
    (1, 64, 128),
    (1, 128, 64),
    (3, 64, 64),
    (1, 64, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 64),
    (3, 64, 64),
    (3, 64, 64),
    (3, 64, 32),
    (3, 32, 32),
    (5, 32, 32),
    (1, 32, 32),
    (3, 32, 3),
];
/// The large network's residual blocks start here.
const RESIDUAL: usize = 10;

/// The painting network's (fast neural style's): a 9x9 and two 3x3s of stride 2 down to a
/// quarter of the size, five residual blocks of two 3x3s there, then twice the size (nearest)
/// and a 3x3 twice, and a 9x9 to the picture. Every convolution but the last has an instance
/// norm after it (and a ReLU, but for each residual block's second). Pictures in and out are 0
/// to 255.
const PAINT: [(usize, usize, usize); 16] = [
    (9, 3, 32),
    (3, 32, 64),
    (3, 64, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 128),
    (3, 128, 64),
    (3, 64, 32),
    (9, 32, 3),
];
/// Its instance norms' epsilon (PyTorch's).
const PAINT_EPS: f32 = 1e-5;
// ag_norm's, ag_join's, ag_double's and ag_pack's (animegan.wgsl).
const NORM_RELU: u32 = 4096;
const NORM_LINEAR: u32 = 8192;
const AFFINE: u32 = 16384;
const NEAREST: u32 = 32768;
const BYTES: u32 = 65536;
const TANH: u32 = 256;
const ADD_BEFORE: u32 = 512;
/// ag_softmax divides each row by f0 + its sum instead.
const SUMS: u32 = 1;
/// ag_columns makes each column a softmax instead.
const SOFTMAX: u32 = 1;

/// A convolution's weights (and bias, or NONE) in `w`, as the kernels want them.
#[derive(Clone, Copy)]
struct Layer {
    w: u32,
    b: u32,
    size: usize,
    cin: usize,
    cout: usize,
}

impl Layer {
    fn linear(&self) -> Linear {
        Linear { w: self.w, b: self.b, inp: self.cin, out: self.cout }
    }

    fn conv(&self) -> Conv {
        Conv { w: self.w, b: self.b }
    }
}

/// Where a stage of the network ends up, for the tests: after which dispatch, at what offset,
/// and how many floats.
#[cfg(test)]
#[derive(Clone, Debug)]
struct Stage {
    name: &'static str,
    after: usize,
    at: u32,
    len: usize,
}

/// Floats in `mem`, handed out in order.
#[derive(Default)]
struct Mem {
    end: u32,
}

impl Mem {
    fn take(&mut self, len: usize) -> u32 {
        let at = self.end;
        self.end += len as u32;
        at
    }
}

/// What both networks' plans make.
struct Plan {
    ops: Ops,
    mem: Mem,
    input: u32,
    packed: u32,
    ease: u32,
    #[cfg(test)]
    stages: Vec<Stage>,
}

impl Plan {
    fn new(n: usize) -> Self {
        let mut mem = Mem::default();
        let input = mem.take(3 * n);
        let ease = mem.take(1);
        let packed = mem.take(n);
        Plan {
            ops: Ops::default(),
            mem,
            input,
            packed,
            ease,
            #[cfg(test)]
            stages: Vec::new(),
        }
    }

    #[cfg(test)]
    fn stage(&mut self, name: &'static str, at: u32, len: usize) {
        self.stages.push(Stage { name, after: self.ops.list.len(), at, len });
    }

    #[cfg(not(test))]
    fn stage(&mut self, _: &'static str, _: u32, _: usize) {}

    /// A 1x1 convolution of [cin][pixels] at `a` to `out`.
    fn conv1(&mut self, l: Layer, a: u32, pixels: usize, out: u32, flags: u32) {
        self.ops.conv1x1(a, pixels, l.linear(), out, flags);
    }

    /// A 3x3 convolution of [cin][h][w] at `a` to `out`, mirroring the edges.
    #[allow(clippy::too_many_arguments)]
    fn conv3(&mut self, l: Layer, a: u32, size: (usize, usize), out: u32, stride: usize, flags: u32) {
        self.ops.conv3x3(a, l.cin, size, l.conv(), l.cout, out, stride, flags | REFLECT);
    }

    /// A k x k convolution (3, 5, 7 or 9; stride 1) of [cin][h][w] at `x` to `out`.
    fn conv_k(&mut self, l: Layer, x: u32, (h, w): (usize, usize), out: u32, flags: u32) {
        let kernel = match l.size {
            9 => Kernel::Conv9,
            7 => Kernel::Conv7,
            5 => Kernel::Conv5,
            _ => Kernel::ConvK3,
        };
        let p = Params { m: h as u32, n: w as u32, k: l.cin as u32, x0: l.cout as u32, a_off: x, b_off: l.w, bias_off: l.b, c_off: out, flags, ..Default::default() };
        self.ops.push(kernel, p, [groups(w, 64), groups(h, 16), groups(l.cout, 8)]);
    }

    /// Normalised over the whole picture, `a` (cout channels of `pixels`) and the cout after it
    /// in `mem` (`a`'s 1x1 convolution): the scales and shifts that give the first the
    /// second's mean and spread, to `stats` [cout][2], eased by `ease`.
    fn adapt(&mut self, a: u32, cout: usize, pixels: usize, parts: u32, stats: u32, eps: f32) {
        let p = (pixels / 4096).clamp(1, 64);
        self.ops.push(Kernel::Stats, Params { m: pixels as u32, x0: p as u32, a_off: a, c_off: parts, ..Default::default() }, [p as u32, 2 * cout as u32, 1]);
        self.ops.push(
            Kernel::Join,
            Params { m: pixels as u32, x0: p as u32, x1: cout as u32, a_off: parts, c_off: stats, bias_off: self.ease, f0: eps, ..Default::default() },
            [groups(cout, 64), 1, 1],
        );
    }

    /// An instance norm of `a` (cout channels of `pixels`) to `out`, with the scales and then
    /// shifts at `gamma` in `w`, its own statistics at `stats` [cout][2] (eased by `ease`), and
    /// `flags` for ag_norm (NORM_RELU, or NORM_LINEAR with `add` added, or NONE).
    #[allow(clippy::too_many_arguments)]
    fn instance(&mut self, a: u32, cout: usize, pixels: usize, gamma: u32, parts: u32, stats: u32, add: u32, out: u32, flags: u32) {
        let p = (pixels / 4096).clamp(1, 64);
        self.ops.push(Kernel::Stats, Params { m: pixels as u32, x0: p as u32, a_off: a, c_off: parts, ..Default::default() }, [p as u32, cout as u32, 1]);
        self.ops.push(
            Kernel::Join,
            Params { m: pixels as u32, x0: p as u32, x1: cout as u32, a_off: parts, b_off: gamma, c_off: stats, bias_off: self.ease, f0: PAINT_EPS, flags: AFFINE, ..Default::default() },
            [groups(cout, 64), 1, 1],
        );
        self.ops.push(Kernel::AgNorm, Params { m: pixels as u32, k: cout as u32, a_off: a, x0: stats, b_off: add, c_off: out, flags, ..Default::default() }, [groups(pixels, 256), cout as u32, 1]);
    }

    /// Twice the size: [channels][h][w] at `x` to `out` (bilinear, or with NEAREST, nearest).
    fn double(&mut self, x: u32, channels: usize, (h, w): (usize, usize), out: u32, flags: u32) -> (usize, usize) {
        self.ops.push(Kernel::Double, Params { m: h as u32, n: w as u32, k: channels as u32, a_off: x, c_off: out, flags, ..Default::default() }, [groups(2 * w, 16), groups(2 * h, 16), channels as u32]);
        (2 * h, 2 * w)
    }

    /// [3][n] from -1 to 1 (or 0 to 255, with BYTES) at `planes` to RGB bytes at `packed`.
    fn pack(&mut self, planes: u32, n: usize, flags: u32) {
        let blocks = n.div_ceil(256);
        self.ops.push(Kernel::Pack, Params { m: n as u32, a_off: planes, c_off: self.packed, flags, ..Default::default() }, [blocks.min(65535) as u32, blocks.div_ceil(65535) as u32, 1]);
    }
}

pub struct AnimeGan {
    run: Runner,
    size: (usize, usize),
    /// Pictures go in as 0 to 255 (else -1 to 1).
    bytes_in: bool,
    input: u32,
    packed: u32,
    ease: u32,
    #[cfg(test)]
    stages: Vec<Stage>,
}

impl AnimeGan {
    /// A style's network for pictures `size` (width, height; multiples of `MULTIPLE`).
    pub fn load(path: &Path, size: (usize, usize)) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Self::from_bytes(&bytes, size)
    }

    pub fn from_bytes(bytes: &[u8], (width, height): (usize, usize)) -> Result<Self> {
        ensure!(width % MULTIPLE == 0 && height % MULTIPLE == 0 && width >= 32 && height >= 32, "restyling needs sizes in multiples of {MULTIPLE}, not {width}x{height}");
        let graph = read_onnx(bytes)?;
        let table: &[(usize, usize, usize)] = match (graph.convs.len(), graph.norms.len()) {
            (31, 0) => &CONVS,
            (39, 0) => &LARGE,
            (16, 15) => &PAINT,
            (n, m) => bail!("not a network for restyling: {n} convolutions, {m} instance norms"),
        };
        let mut up = Upload::default();
        let mut layers = Vec::new();
        for ((name, bias), &(size, cin, cout)) in graph.convs.iter().zip(table) {
            let (dims, v) = graph.tensors.get(name).with_context(|| format!("ONNX: no {name}"))?;
            ensure!(dims == &[cout, cin, size, size], "not an AnimeGANv3 network: {name} is {dims:?}");
            let w = if size == 1 {
                // [out][in] to [in][out] for the matrix multiply.
                let mut t = vec![0.0; cin * cout];
                for o in 0..cout {
                    for i in 0..cin {
                        t[i * cout + o] = v[o * cin + i];
                    }
                }
                up.push(&t)
            } else {
                up.push(v)
            };
            let b = match bias {
                Some(name) => {
                    let (dims, v) = graph.tensors.get(name).with_context(|| format!("ONNX: no {name}"))?;
                    ensure!(dims.iter().product::<usize>() == cout, "not an AnimeGANv3 network: {name} is {dims:?}");
                    up.push(v)
                }
                None => NONE,
            };
            layers.push(Layer { w, b, size, cin, cout });
        }
        let plan = if table.len() == PAINT.len() {
            // Each norm's scales, then its shifts, after the convolution it follows.
            let mut norms = Vec::new();
            for ((scale, shift), layer) in graph.norms.iter().zip(&layers) {
                let mut first = None;
                for name in [scale, shift] {
                    let (dims, v) = graph.tensors.get(name).with_context(|| format!("ONNX: no {name}"))?;
                    ensure!(dims == &[layer.cout], "not a fast neural style network: {name} is {dims:?}");
                    let at = up.push(v);
                    first.get_or_insert(at);
                }
                norms.extend(first);
            }
            painting(&layers, &norms, (width, height))
        } else if table.len() == CONVS.len() {
            // The norms' epsilon: the one 1x1x1x1 constant (it differs between styles).
            let eps = graph.tensors.values().find(|(d, _)| d == &[1, 1, 1, 1]).map(|(_, v)| v[0]).context("not an AnimeGANv3 network: no epsilon")?;
            // Some styles (Sketch) normalise the attention's output over the whole picture, with
            // a scale and shift for each channel.
            let gamma = graph.tensors.iter().find(|(k, (d, _))| k.ends_with("LayerNorm/gamma/read:0") && d.iter().product::<usize>() == 128);
            let layer_norm = match gamma {
                Some((name, (_, g))) => {
                    let mut beta = graph.tensors.iter().filter(|(k, (d, _))| *k != name && d == &[1, 128, 1, 1]);
                    let (_, (_, b)) = beta.next().context("ONNX: a layer norm with no shift")?;
                    ensure!(beta.next().is_none(), "ONNX: which is the layer norm's shift?");
                    let at = up.push(g);
                    up.push(b);
                    Some(at)
                }
                None => None,
            };
            // The attention's softmax is over the picture for each key and then each place's
            // are normalised over the keys (the sum's over axis 2), or the other way round (1).
            let swapped = match graph.sums.as_slice() {
                [axes] if axes == &[2] => false,
                [axes] if axes == &[1] => true,
                other => bail!("not an AnimeGANv3 network: attention sums over {other:?}"),
            };
            small(&layers, (width, height), eps, layer_norm, swapped)
        } else {
            let eps = graph.tensors.values().find(|(d, _)| d == &[1, 1, 1, 1]).map(|(_, v)| v[0]).context("not an AnimeGANv3 network: no epsilon")?;
            large(&layers, (width, height), eps)
        };
        let bytes_in = table.len() == PAINT.len();
        let run = Runner::new(&up, plan.ops, plan.mem.end, if bytes_in { "fast neural style" } else { "AnimeGANv3" })?;
        Ok(Self {
            run,
            size: (width, height),
            bytes_in,
            input: plan.input,
            packed: plan.packed,
            ease: plan.ease,
            #[cfg(test)]
            stages: plan.stages,
        })
    }

    pub fn size(&self) -> (usize, usize) {
        self.size
    }

    pub fn adapter(&self) -> &str {
        &self.run.adapter
    }

    /// Redraws a picture of `size()`, RGB bytes row by row from the top, as RGBA bytes the
    /// same way. `ease` (0 to 1) is how much each norm keeps of the last picture's: steadier
    /// from one picture to the next. 0 for the first picture of a shot.
    pub fn run(&mut self, rgb: &[u8], ease: f32) -> Result<Vec<u8>> {
        let (w, h) = self.size;
        let n = w * h;
        ensure!(rgb.len() == 3 * n, "expected {w}x{h} RGB, got {} bytes", rgb.len());
        let mut planes = vec![0.0f32; 3 * n];
        for (i, p) in rgb.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                planes[c * n + i] = if self.bytes_in { f32::from(p[c]) } else { f32::from(p[c]) / 127.5 - 1.0 };
            }
        }
        self.run.write(self.input, &planes);
        self.run.write(self.ease, &[ease.clamp(0.0, 1.0)]);
        self.run.run()?;
        let words = self.run.read(self.packed, n)?;
        let mut rgba: Vec<u8> = bytemuck::cast_slice(&words).to_vec();
        for px in rgba.as_chunks_mut::<4>().0 {
            px[3] = 255;
        }
        Ok(rgba)
    }
}

/// The network most styles use, for pictures (w, h). `layer_norm`: where the attention's
/// output's norm's scales and shifts are in `w`, if it has one. `swapped`: the attention's
/// softmax is over the keys and its normalising over the picture.
fn small(layers: &[Layer], (w, h): (usize, usize), eps: f32, layer_norm: Option<u32>, swapped: bool) -> Plan {
    let n = w * h;
    let mut plan = Plan::new(n);
    let m = &mut plan.mem;
    let e0 = m.take(32 * n);
    let e2 = m.take(64 * n / 4);
    let e4 = m.take(128 * n / 16);
    let e6 = m.take(128 * n / 64);
    // A convolution's output and its 1x1's, side by side for the statistics.
    let ab = m.take(64 * n);
    let big = m.take(64 * n);
    let small = m.take(32 * n);
    let parts = m.take(2 * 128 * 64 * 2);
    let stats: Vec<u32> = (0..14).map(|_| m.take(128 * 2)).collect();
    let t1 = m.take(128 * n / 64);
    let t2 = m.take(128 * n / 64);
    let planes = ab;
    let input = plan.input;

    // The last for the layer norm.
    let mut next_stats = stats[..13].iter().copied();
    let mut block = |plan: &mut Plan, i: usize, x: u32, (h, w): (usize, usize), stride: usize, out: u32, add: u32| -> (usize, usize) {
        let (conv, one) = (layers[i], layers[i + 1]);
        let (oh, ow) = (h / stride, w / stride);
        let pixels = oh * ow;
        if conv.size == 7 {
            plan.conv_k(conv, x, (h, w), ab, 0);
        } else {
            plan.conv3(conv, x, (h, w), ab, stride, 0);
        }
        let b = ab + (conv.cout * pixels) as u32;
        plan.conv1(one, ab, pixels, b, 0);
        let s = next_stats.next().expect("a block too many");
        plan.adapt(ab, conv.cout, pixels, parts, s, eps);
        plan.ops.push(Kernel::AgNorm, Params { m: pixels as u32, k: conv.cout as u32, a_off: ab, x0: s, b_off: add, c_off: out, ..Default::default() }, [groups(pixels, 256), conv.cout as u32, 1]);
        (oh, ow)
    };

    // Down.
    let s = block(&mut plan, 0, input, (h, w), 1, e0, NONE);
    plan.stage("down.0", e0, 32 * n);
    let s = block(&mut plan, 2, e0, s, 2, small, NONE);
    plan.stage("down.1", small, 32 * n / 4);
    let s = block(&mut plan, 4, small, s, 1, e2, NONE);
    plan.stage("down.2", e2, 64 * n / 4);
    let s = block(&mut plan, 6, e2, s, 2, small, NONE);
    plan.stage("down.3", small, 64 * n / 16);
    let s = block(&mut plan, 8, small, s, 1, e4, NONE);
    plan.stage("down.4", e4, 128 * n / 16);
    let s = block(&mut plan, 10, e4, s, 2, small, NONE);
    plan.stage("down.5", small, 128 * n / 64);
    let s = block(&mut plan, 12, small, s, 1, e6, NONE);
    plan.stage("down.6", e6, 128 * n / 64);

    // The attention: keys from the features, a softmax over the whole picture for each key
    // and then over the keys for each place, and back.
    let n8 = n / 64;
    plan.conv1(layers[ATTENTION], e6, n8, t1, 0);
    plan.conv1(layers[ATTENTION + 1], t1, n8, t2, 0);
    // Keys [128][n8]: along the rows, then the columns (or the other way round).
    let (rows, columns) = if swapped { (SUMS, SOFTMAX) } else { (0, 0) };
    let row = Params { m: 128, n: n8 as u32, a_off: t2, flags: rows, f0: ATTENTION_EPS, ..Default::default() };
    let column = Params { m: 128, n: n8 as u32, a_off: t2, flags: columns, f0: ATTENTION_EPS, ..Default::default() };
    let steps = [(Kernel::RowSoftmax, row, [128, 1, 1]), (Kernel::Columns, column, [groups(n8, 64), 1, 1])];
    for i in if swapped { [1, 0] } else { [0, 1] } {
        let (k, p, g) = steps[i];
        plan.ops.push(k, p, g);
    }
    plan.conv1(layers[ATTENTION + 2], t2, n8, t1, 0);
    plan.conv1(layers[ATTENTION + 3], t1, n8, t2, 0);
    let norm = match layer_norm {
        Some(gb) => {
            let s = stats[13];
            let p = (n8 / 4096).clamp(1, 64);
            plan.ops.push(Kernel::Stats, Params { m: n8 as u32, x0: p as u32, a_off: t2, c_off: parts, ..Default::default() }, [p as u32, 128, 1]);
            plan.ops.push(Kernel::Layer, Params { m: n8 as u32, x0: p as u32, x1: 128, a_off: parts, b_off: gb, c_off: s, f0: eps, ..Default::default() }, [1, 1, 1]);
            s
        }
        None => NONE,
    };
    plan.ops.push(Kernel::AgNorm, Params { m: n8 as u32, k: 128, a_off: t2, x0: norm, b_off: e6, flags: ADD_BEFORE, c_off: small, ..Default::default() }, [groups(n8, 256), 128, 1]);
    plan.stage("attention", small, 128 * n8);

    // Up.
    let s = plan.double(small, 128, s, big, 0);
    let s = block(&mut plan, 18, big, s, 1, small, e4);
    plan.stage("up.0", small, 128 * n / 16);
    let s = block(&mut plan, 20, small, s, 1, big, NONE);
    plan.stage("up.1", big, 128 * n / 16);
    let s = plan.double(big, 128, s, small, 0);
    let s = block(&mut plan, 22, small, s, 1, big, e2);
    plan.stage("up.2", big, 64 * n / 4);
    let s = block(&mut plan, 24, big, s, 1, small, NONE);
    plan.stage("up.3", small, 64 * n / 4);
    let s = plan.double(small, 64, s, big, 0);
    let s = block(&mut plan, 26, big, s, 1, small, e0);
    plan.stage("up.4", small, 32 * n);
    let s = block(&mut plan, 28, small, s, 1, big, NONE);
    plan.stage("up.5", big, 32 * n);
    debug_assert_eq!(s, (h, w));
    plan.conv_k(layers[30], big, (h, w), planes, TANH);
    plan.stage("out", planes, 3 * n);
    plan.pack(planes, n, 0);
    plan
}

/// The painting network, for pictures (w, h). `norms`: where each instance norm's scales (and
/// then shifts) are in `w`.
fn painting(layers: &[Layer], norms: &[u32], (w, h): (usize, usize)) -> Plan {
    let n = w * h;
    let mut plan = Plan::new(n);
    let m = &mut plan.mem;
    // Full size: 64 channels (the half's doubled), then 32; a half: 64, and 128 doubled; a
    // quarter: the residual blocks' stream and their insides.
    let full = m.take(64 * n);
    let out = m.take(32 * n);
    let half = m.take(64 * n / 4);
    let wide = m.take(128 * n / 4);
    let x = m.take(128 * n / 16);
    let r = m.take(128 * n / 16);
    let r2 = m.take(128 * n / 16);
    let planes = m.take(3 * n);
    let parts = m.take(128 * 64 * 2);
    let stats: Vec<u32> = (0..norms.len()).map(|_| m.take(128 * 2)).collect();
    let input = plan.input;
    let (hh, qq) = ((h / 2, w / 2), (h / 4, w / 4));
    let (nh, nq) = (n / 4, n / 16);
    // Layer i's instance norm of `a` (in place), with a ReLU.
    let norm = |plan: &mut Plan, i: usize, a: u32, pixels: usize| plan.instance(a, layers[i].cout, pixels, norms[i], parts, stats[i], NONE, a, NORM_RELU);

    // Down.
    plan.conv_k(layers[0], input, (h, w), full, 0);
    norm(&mut plan, 0, full, n);
    plan.stage("down.0", full, 32 * n);
    plan.conv3(layers[1], full, (h, w), half, 2, 0);
    norm(&mut plan, 1, half, nh);
    plan.stage("down.1", half, 64 * nh);
    plan.conv3(layers[2], half, hh, x, 2, 0);
    norm(&mut plan, 2, x, nq);
    plan.stage("down.2", x, 128 * nq);

    // Five residual blocks, each added back into x (the second's norm has no ReLU).
    for (b, name) in ["res.0", "res.1", "res.2", "res.3", "res.4"].into_iter().enumerate() {
        let i = 3 + 2 * b;
        plan.conv3(layers[i], x, qq, r, 1, 0);
        norm(&mut plan, i, r, nq);
        plan.conv3(layers[i + 1], r, qq, r2, 1, 0);
        plan.instance(r2, 128, nq, norms[i + 1], parts, stats[i + 1], x, x, NORM_LINEAR);
        plan.stage(name, x, 128 * nq);
    }

    // Up: twice the size (nearest) and a 3x3, twice; then the 9x9 to the picture.
    let s = plan.double(x, 128, qq, wide, NEAREST);
    plan.conv3(layers[13], wide, s, half, 1, 0);
    norm(&mut plan, 13, half, nh);
    plan.stage("up.0", half, 64 * nh);
    let s = plan.double(half, 64, s, full, NEAREST);
    debug_assert_eq!(s, (h, w));
    plan.conv3(layers[14], full, s, out, 1, 0);
    norm(&mut plan, 14, out, n);
    plan.stage("up.1", out, 32 * n);
    plan.conv_k(layers[15], out, (h, w), planes, 0);
    plan.stage("out", planes, 3 * n);
    plan.pack(planes, n, BYTES);
    plan
}

/// The large network (Ghibli-c1's), for pictures (w, h).
fn large(layers: &[Layer], (w, h): (usize, usize), eps: f32) -> Plan {
    let n = w * h;
    let mut plan = Plan::new(n);
    let m = &mut plan.mem;
    // The way down's, added back on the way up (and then free): full size, a half, a quarter.
    // s0, x and y follow each other: two or three of them are one bigger picture at times.
    let s0 = m.take(32 * n);
    let x = m.take(32 * n);
    let y = m.take(32 * n);
    let s3 = m.take(64 * n / 4);
    let s6 = m.take(128 * n / 16);
    // An eighth: the residual blocks' stream, and their insides.
    let r = m.take(128 * n / 64);
    let r1 = m.take(64 * n / 64);
    let r2 = m.take(64 * n / 64);
    let parts = m.take(2 * 32 * 64 * 2);
    let stats = m.take(32 * 2);
    let input = plan.input;
    let (half, quarter, eighth) = ((h / 2, w / 2), (h / 4, w / 4), (h / 8, w / 8));
    let leaky = LEAKY2;
    let conv = |plan: &mut Plan, i: usize, a: u32, size: (usize, usize), out: u32, stride: usize, flags: u32| plan.conv3(layers[i], a, size, out, stride, flags);

    // Down.
    plan.conv_k(layers[0], input, (h, w), s0, LEAKY2);
    plan.stage("base.0", s0, 32 * n);
    conv(&mut plan, 1, s0, (h, w), x, 2, leaky);
    conv(&mut plan, 2, x, half, y, 1, leaky);
    conv(&mut plan, 3, y, half, s3, 1, leaky);
    plan.stage("base.3", s3, 64 * n / 4);
    conv(&mut plan, 4, s3, half, x, 2, leaky);
    conv(&mut plan, 5, x, quarter, y, 1, leaky);
    conv(&mut plan, 6, y, quarter, s6, 1, leaky);
    plan.stage("base.6", s6, 128 * n / 16);
    conv(&mut plan, 7, s6, quarter, x, 2, leaky);
    conv(&mut plan, 8, x, eighth, y, 1, leaky);
    conv(&mut plan, 9, y, eighth, r, 1, leaky);
    plan.stage("base.9", r, 128 * n / 64);
    let n8 = n / 64;
    for b in 0..6 {
        let i = RESIDUAL + 3 * b;
        plan.conv1(layers[i], r, n8, r1, LEAKY2);
        conv(&mut plan, i + 1, r1, eighth, r2, 1, leaky);
        plan.conv1(layers[i + 2], r2, n8, r, ACCUMULATE | LEAKY2_LAST);
    }
    plan.stage("base.27", r, 128 * n / 64);

    // Up, each time adding the way down's back in after the first convolution.
    plan.double(r, 128, eighth, x, 0);
    conv(&mut plan, 28, x, quarter, s6, 1, leaky | ACCUMULATE);
    plan.stage("main.add", s6, 128 * n / 16);
    conv(&mut plan, 29, s6, quarter, x, 1, leaky);
    conv(&mut plan, 30, x, quarter, y, 1, leaky);
    plan.double(y, 128, quarter, x, 0);
    conv(&mut plan, 31, x, half, s3, 1, leaky | ACCUMULATE);
    plan.stage("main.add_1", s3, 64 * n / 4);
    conv(&mut plan, 32, s3, half, y, 1, leaky);
    conv(&mut plan, 33, y, half, s3, 1, leaky);
    plan.double(s3, 64, half, x, 0);
    conv(&mut plan, 34, x, (h, w), s0, 1, leaky | ACCUMULATE);
    plan.stage("main.add_2", s0, 32 * n);
    conv(&mut plan, 35, s0, (h, w), y, 1, leaky);
    plan.stage("main.7", y, 32 * n);
    // The one norm: the 5x5's normalised and given its 1x1's mean and spread.
    plan.conv_k(layers[36], y, (h, w), s0, 0);
    plan.conv1(layers[37], s0, n, x, 0);
    plan.adapt(s0, 32, n, parts, stats, eps);
    plan.ops.push(Kernel::AgNorm, Params { m: n as u32, k: 32, a_off: s0, x0: stats, b_off: NONE, c_off: y, ..Default::default() }, [groups(n, 256), 32, 1]);
    plan.stage("main.8", y, 32 * n);
    plan.conv_k(layers[38], y, (h, w), x, TANH);
    plan.stage("out", x, 3 * n);
    plan.pack(x, n, 0);
    plan
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_keep_the_shape() {
        assert_eq!(size_for(16.0 / 9.0, 640 * 360), (640, 360));
        let (w, h) = size_for(2.39, 200_000);
        assert!(w % MULTIPLE == 0 && h % MULTIPLE == 0);
        assert!((w as f64 / h as f64 - 2.39).abs() < 0.05, "{w}x{h}");
    }

    /// How much the pictures change from one to the next, against how much the frames do, with
    /// the norms eased by different amounts: `FLEX_ANIMEGAN=style.onnx
    /// FLEX_ANIMEGAN_FRAMES=frames.rgb FLEX_ANIMEGAN_SIZE=640x272 cargo test -p flex-depth --release
    /// animegan_flicker -- --ignored --nocapture`, the frames raw RGB one after another.
    /// With FLEX_ANIMEGAN_OUT, each run's pictures are written there as raw RGBA.
    #[test]
    #[ignore]
    fn animegan_flicker() {
        let model = std::env::var("FLEX_ANIMEGAN").expect("FLEX_ANIMEGAN");
        let frames = std::fs::read(std::env::var("FLEX_ANIMEGAN_FRAMES").expect("FLEX_ANIMEGAN_FRAMES")).unwrap();
        let size = std::env::var("FLEX_ANIMEGAN_SIZE").expect("FLEX_ANIMEGAN_SIZE");
        let (w, h) = size.split_once('x').map(|(w, h)| (w.parse().unwrap(), h.parse().unwrap())).unwrap();
        let frames: Vec<&[u8]> = frames.chunks_exact(w * h * 3).collect();
        let change = |a: &[u8], b: &[u8], step: usize| a.iter().zip(b).enumerate().filter(|(i, _)| i % step < 3).map(|(_, (x, y))| f64::from(x.abs_diff(*y))).sum::<f64>() / (w * h * 3) as f64;
        let input = frames.windows(2).map(|f| change(f[0], f[1], 3)).sum::<f64>() / (frames.len() - 1) as f64;
        let mut net = AnimeGan::load(Path::new(&model), (w, h)).unwrap();
        for ease in [0.0, 0.5, 0.7, 0.85] {
            let outs: Vec<Vec<u8>> = frames.iter().enumerate().map(|(i, f)| net.run(f, if i == 0 { 0.0 } else { ease }).unwrap()).collect();
            let output = outs.windows(2).map(|o| change(&o[0], &o[1], 4)).sum::<f64>() / (outs.len() - 1) as f64;
            eprintln!("ease {ease:.2}: pictures change {output:.2} a step, frames {input:.2}: {:.2}x", output / input);
            if let Ok(dir) = std::env::var("FLEX_ANIMEGAN_OUT") {
                std::fs::write(format!("{dir}/ease-{ease:.2}.rgba"), outs.concat()).unwrap();
            }
        }
    }

    /// Compares each block with ONNX Runtime's (needs a style's ONNX file, a reference from
    /// tools/animegan/reference.py made with it, and a GPU):
    /// `FLEX_ANIMEGAN=style.onnx FLEX_ANIMEGAN_REF=ref.safetensors cargo test -p flex-depth animegan_matches_onnx -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn animegan_matches_onnx() {
        let model = std::env::var("FLEX_ANIMEGAN").expect("FLEX_ANIMEGAN");
        let reference = crate::read_safetensors(&std::fs::read(std::env::var("FLEX_ANIMEGAN_REF").expect("FLEX_ANIMEGAN_REF")).unwrap()).unwrap();
        let (shape, rgb) = &reference["rgb"];
        let (h, w) = (shape[0], shape[1]);
        let rgb: Vec<u8> = rgb.iter().map(|v| *v as u8).collect();
        let mut net = AnimeGan::load(Path::new(&model), (w, h)).unwrap();
        let started = std::time::Instant::now();
        let out = net.run(&rgb, 0.0).unwrap();
        eprintln!("{w}x{h}: first run {:.1} ms", started.elapsed().as_secs_f64() * 1000.0);
        // AnimeGANv3's decoder's blocks with the encoder's added back.
        let painting = net.bytes_in;
        let added = |name: &str| match name {
            _ if painting => None,
            "up.0" => Some("down.4"),
            "up.2" => Some("down.2"),
            "up.4" => Some("down.0"),
            _ => None,
        };
        let mut worst = 0.0f32;
        for s in net.stages.clone() {
            net.run.time_first(s.after, 0);
            let got = net.run.read(s.at, s.len).unwrap();
            let mut expected = reference[s.name].1.clone();
            if let Some(other) = added(s.name) {
                expected.iter_mut().zip(&reference[other].1).for_each(|(e, o)| *e += o);
            }
            assert_eq!(got.len(), expected.len(), "{}", s.name);
            let scale = expected.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            let err = got.iter().zip(&expected).fold(0.0f32, |m, (g, e)| m.max((g - e).abs())) / scale;
            eprintln!("{:10} worst error {err:.2e} of {scale:.2}", s.name);
            worst = worst.max(err);
        }
        let expected = &reference["out"].1;
        let n = w * h;
        let bytes = (0..n).flat_map(|i| (0..3).map(move |c| (i, c))).fold(0i32, |m, (i, c)| {
            let v = expected[c * n + i];
            let e = if painting { v } else { (v + 1.0) * 127.5 }.clamp(0.0, 255.0).round() as i32;
            m.max((i32::from(out[i * 4 + c]) - e).abs())
        });
        eprintln!("bytes out at most {bytes} from ONNX Runtime's");
        assert!(worst < 2e-3, "worst relative error {worst}");
        assert!(bytes <= 2);

        let runs = 10;
        let started = std::time::Instant::now();
        for _ in 0..runs {
            net.run.run().unwrap();
        }
        eprintln!("{:.1} ms a picture, in {} submissions", started.elapsed().as_secs_f64() * 1000.0 / runs as f64, net.run.parts.len());
        eprintln!("{:.1} ms as one submission", net.run.time_first(net.run.ops.list.len(), runs));
        if std::env::var("FLEX_PROFILE").is_ok_and(|v| v == "context") {
            // Each dispatch's time where it is in the plan: the plan up to it, less the plan
            // before it, grouped by kernel.
            let mut before = 0.0;
            let mut by: Vec<(String, f64)> = Vec::new();
            for i in 1..=net.run.ops.list.len() {
                let t = net.run.time_first(i, 8);
                let (k, p, _) = &net.run.ops.list[i - 1];
                let name = match k {
                    Kernel::Conv3x3Wide => format!("conv3x3 {}->{} at {}x{} stride {}", p.k, p.x0, p.n, p.m, p.x1),
                    k => format!("{k:?}"),
                };
                match by.iter_mut().find(|b| b.0 == name) {
                    Some(b) => b.1 += t - before,
                    None => by.push((name, t - before)),
                }
                before = t;
            }
            by.sort_by(|a, b| b.1.total_cmp(&a.1));
            for (name, t) in by {
                eprintln!("{t:7.2} ms in place  {name}");
            }
        } else if std::env::var("FLEX_PROFILE").is_ok() {
            net.run.report(|_, k, p| match k {
                Kernel::Conv3x3Wide => format!("conv3x3 {}->{} at {}x{} stride {}", p.k, p.x0, p.n, p.m, p.x1),
                Kernel::Conv7 | Kernel::Conv9 => format!("{k:?} {}->{} at {}x{}", p.k, p.x0, p.n, p.m),
                Kernel::Matmul => format!("1x1 {}->{} over {}", p.k, p.n, p.m),
                k => format!("{k:?}"),
            });
        }
    }
}
