//! Finding faces: YuNet (MIT, from OpenCV's model zoo) as wgpu compute shaders. It gives each
//! face a box and five points. The plan and weights are in `yunet.bin`, made from the ONNX
//! file by `tools/face/convert.py`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};

const MODEL: &[u8] = include_bytes!("yunet.bin");
/// The picture it takes.
pub const WIDTH: usize = 640;
pub const HEIGHT: usize = 352;
const STRIDES: [usize; 3] = [8, 16, 32];
/// Below this it's probably not a face. Lower than YuNet's usual 0.9, which misses too many
/// faces in films.
const MIN_SCORE: f32 = 0.6;
/// Boxes overlapping this much are the same face.
const SAME_FACE: f32 = 0.3;
const SLOT: u64 = 256;
const OPS_PER_SUBMIT: usize = 12;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    h: u32,
    w: u32,
    cin: u32,
    cout: u32,
    a: u32,
    b: u32,
    c: u32,
    weights: u32,
    bias: u32,
    kernel: u32,
    stride: u32,
    groups: u32,
    relu: u32,
    oh: u32,
    ow: u32,
    pad: u32,
}

const KERNELS: [&str; 3] = ["conv", "pool", "add_up"];

/// One face, in pixels of the picture given. Points are the right eye, left eye, nose and the
/// right and left corners of the mouth (the person's own right and left).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Detection {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub score: f32,
    pub points: [[f32; 2]; 5],
}

#[derive(Clone, Copy)]
struct Tensor {
    at: u32,
    channels: usize,
    h: usize,
    w: usize,
}

fn field(op: &serde_json::Value, name: &str) -> Result<usize> {
    op[name].as_u64().map(|v| v as usize).with_context(|| format!("the plan's op has no {name}"))
}

fn name<'a>(op: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    op[key].as_str().with_context(|| format!("the plan's op has no {key}"))
}

pub struct Face {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Vec<wgpu::ComputePipeline>,
    mem: wgpu::Buffer,
    bind: wgpu::BindGroup,
    readback: wgpu::Buffer,
    ops: Vec<(usize, [u32; 3])>,
    /// Where each scale's outputs are: cls, obj, bbox, kps.
    outputs: [[Tensor; 4]; 3],
    adapter: String,
}

impl Face {
    pub fn new() -> Result<Self> {
        if MODEL.len() < 4 {
            bail!("the face model is empty");
        }
        let plan_len = u32::from_le_bytes([MODEL[0], MODEL[1], MODEL[2], MODEL[3]]) as usize;
        let plan: serde_json::Value = serde_json::from_slice(&MODEL[4..4 + plan_len]).context("reading the face model's plan")?;
        let weights: Vec<f32> = MODEL[4 + plan_len..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        let mut end = (3 * HEIGHT * WIDTH) as u32;
        tensors.insert("input".into(), Tensor { at: 0, channels: 3, h: HEIGHT, w: WIDTH });
        let groups = |n: usize| n.div_ceil(16) as u32;
        let mut ops = Vec::new();
        let mut params = Vec::new();
        for op in plan["ops"].as_array().context("the plan has no ops")? {
            let input = *tensors.get(name(op, "in")?).context("an op's input comes before it's made")?;
            let (kernel, mut p, out) = match name(op, "op")? {
                "conv" => {
                    let (stride, cout) = (field(op, "stride")?, field(op, "cout")?);
                    let (oh, ow) = ((input.h - 1) / stride + 1, (input.w - 1) / stride + 1);
                    let p = Params {
                        cin: field(op, "cin")? as u32,
                        cout: cout as u32,
                        weights: field(op, "w")? as u32,
                        bias: field(op, "b")? as u32,
                        kernel: field(op, "kernel")? as u32,
                        stride: stride as u32,
                        groups: field(op, "groups")? as u32,
                        relu: u32::from(op["relu"].as_bool().unwrap_or(false)),
                        ..Default::default()
                    };
                    (0, p, Tensor { at: 0, channels: cout, h: oh, w: ow })
                }
                "pool" => (1, Params::default(), Tensor { at: 0, channels: input.channels, h: input.h / 2, w: input.w / 2 }),
                "add_up" => {
                    let up = *tensors.get(name(op, "up")?).context("an op's input comes before it's made")?;
                    if up.h * 2 != input.h || up.w * 2 != input.w {
                        bail!("the picture's size has to be a multiple of 32");
                    }
                    (2, Params { b: up.at, ..Default::default() }, Tensor { at: 0, channels: input.channels, h: input.h, w: input.w })
                }
                other => bail!("unknown op {other}"),
            };
            let out = Tensor { at: end, ..out };
            end += (out.channels * out.h * out.w) as u32;
            // add_up's h and w are B's; the others' are the input's.
            let (h, w) = if kernel == 2 { (input.h / 2, input.w / 2) } else { (input.h, input.w) };
            p = Params { h: h as u32, w: w as u32, a: input.at, c: out.at, oh: out.h as u32, ow: out.w as u32, ..p };
            params.push(p);
            ops.push((kernel, [groups(out.w), groups(out.h), out.channels as u32]));
            tensors.insert(name(op, "out")?.to_owned(), out);
        }
        let output = |kind: &str, stride: usize| -> Result<Tensor> {
            let name = plan["outputs"][format!("{kind}_{stride}")].as_str().context("the plan has no output")?;
            tensors.get(name).copied().context("the plan's output isn't made")
        };
        let mut outputs = Vec::new();
        for stride in STRIDES {
            outputs.push([output("cls", stride)?, output("obj", stride)?, output("bbox", stride)?, output("kps", stride)?]);
        }
        let outputs: [[Tensor; 4]; 3] = outputs.try_into().map_err(|_| anyhow::anyhow!("three scales expected"))?;

        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
            .map_err(|e| anyhow::anyhow!("no GPU adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("flex-face"), ..Default::default() }))
            .map_err(|e| anyhow::anyhow!("GPU device: {e}"))?;
        let storage = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let w = storage("weights", weights.len() as u64 * 4, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        queue.write_buffer(&w, 0, bytemuck::cast_slice(&weights));
        let mem_bytes = end as u64 * 4;
        let mem = storage("mem", mem_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC);
        let readback = storage("readback", mem_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let slots = storage("params", SLOT * params.len() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let mut bytes = vec![0u8; params.len() * SLOT as usize];
        for (i, p) in params.iter().enumerate() {
            bytes[i * SLOT as usize..][..std::mem::size_of::<Params>()].copy_from_slice(bytemuck::bytes_of(p));
        }
        queue.write_buffer(&slots, 0, &bytes);

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage_entry = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("flex-face"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }),
                entry(1, storage_entry(false)),
                entry(2, storage_entry(true)),
            ],
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flex-face"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &slots, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64) }) },
                wgpu::BindGroupEntry { binding: 1, resource: mem.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: w.as_entire_binding() },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("flex-face"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let source = wgpu::ShaderModuleDescriptor { label: Some("face.wgsl"), source: wgpu::ShaderSource::Wgsl(include_str!("face.wgsl").into()) };
        // SAFETY: our own shaders, and every offset and size comes from the plan above, which
        // `mem` and the weights are made to fit.
        let module = unsafe { device.create_shader_module_trusted(source, wgpu::ShaderRuntimeChecks::unchecked()) };
        let pipelines = KERNELS
            .iter()
            .map(|entry| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        drop(w); // the bind group keeps it
        Ok(Self { device, queue, pipelines, mem, bind, readback, ops, outputs, adapter: adapter.get_info().name })
    }

    /// The GPU it runs on.
    pub fn adapter(&self) -> &str {
        &self.adapter
    }

    /// Faces in a picture of `WIDTH` × `HEIGHT`, as RGB bytes row by row from the top.
    pub fn run(&mut self, rgb: &[u8]) -> Result<Vec<Detection>> {
        Ok(decode(&self.outputs_for(rgb)?))
    }

    /// Each scale's cls, obj, bbox and kps cell by cell, as the last convolutions leave them
    /// (the scores before their sigmoid).
    fn outputs_for(&mut self, rgb: &[u8]) -> Result<[[Vec<f32>; 4]; 3]> {
        if rgb.len() != WIDTH * HEIGHT * 3 {
            bail!("expected {WIDTH}x{HEIGHT} RGB, got {} bytes", rgb.len());
        }
        // The model was trained on BGR, 0 to 255, channels first.
        let mut input = vec![0.0f32; 3 * HEIGHT * WIDTH];
        for (i, px) in rgb.as_chunks::<3>().0.iter().enumerate() {
            for c in 0..3 {
                input[c * HEIGHT * WIDTH + i] = f32::from(px[2 - c]);
            }
        }
        self.queue.write_buffer(&self.mem, 0, bytemuck::cast_slice(&input));
        for (part, ops) in self.ops.chunks(OPS_PER_SUBMIT).enumerate() {
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for (i, (kernel, g)) in ops.iter().enumerate() {
                    pass.set_pipeline(&self.pipelines[*kernel]);
                    let slot = (part * OPS_PER_SUBMIT + i) as u64 * SLOT;
                    pass.set_bind_group(0, &self.bind, &[slot as u32]);
                    pass.dispatch_workgroups(g[0], g[1], g[2]);
                }
            }
            self.queue.submit([enc.finish()]);
            self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("{e}"))?;
        }
        let mut enc = self.device.create_command_encoder(&Default::default());
        let tensors = self.outputs;
        let first = tensors.iter().flatten().map(|t| t.at).min().unwrap_or(0) as u64 * 4;
        let last = tensors.iter().flatten().map(|t| t.at as u64 + (t.channels * t.h * t.w) as u64).max().unwrap_or(0) * 4;
        enc.copy_buffer_to_buffer(&self.mem, first, &self.readback, 0, last - first);
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..last - first);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::PollType::wait_indefinitely()).map_err(|e| anyhow::anyhow!("{e}"))?;
        let out = {
            let view = slice.get_mapped_range().map_err(|e| anyhow::anyhow!("{e}"))?;
            let floats: &[f32] = bytemuck::cast_slice(&view);
            tensors.map(|scale| scale.map(|t| cells(&floats[(t.at as usize - first as usize / 4)..][..t.channels * t.h * t.w], t)))
        };
        self.readback.unmap();
        Ok(out)
    }
}

/// Channels first to cell by cell.
fn cells(src: &[f32], t: Tensor) -> Vec<f32> {
    let plane = t.h * t.w;
    let mut out = vec![0.0; src.len()];
    for cell in 0..plane {
        for c in 0..t.channels {
            out[cell * t.channels + c] = src[c * plane + cell];
        }
    }
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn overlap(a: &Detection, b: &Detection) -> f32 {
    let (x0, y0) = (a.x.max(b.x), a.y.max(b.y));
    let (x1, y1) = ((a.x + a.width).min(b.x + b.width), (a.y + a.height).min(b.y + b.height));
    let inter = (x1 - x0).max(0.0) * (y1 - y0).max(0.0);
    inter / (a.width * a.height + b.width * b.height - inter).max(1e-6)
}

/// From the network's outputs to faces, the way OpenCV's FaceDetectorYN does it.
fn decode(outputs: &[[Vec<f32>; 4]; 3]) -> Vec<Detection> {
    let mut found = Vec::new();
    for (scale, stride) in outputs.iter().zip(STRIDES) {
        let (cols, rows) = (WIDTH / stride, HEIGHT / stride);
        let [cls, obj, bbox, kps] = scale;
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                let score = (sigmoid(cls[i]) * sigmoid(obj[i])).sqrt();
                if score < MIN_SCORE {
                    continue;
                }
                let s = stride as f32;
                let (cf, rf) = (c as f32, r as f32);
                let (width, height) = (bbox[i * 4 + 2].exp() * s, bbox[i * 4 + 3].exp() * s);
                let mut points = [[0.0; 2]; 5];
                for (n, point) in points.iter_mut().enumerate() {
                    *point = [(kps[i * 10 + 2 * n] + cf) * s, (kps[i * 10 + 2 * n + 1] + rf) * s];
                }
                found.push(Detection { x: (cf + bbox[i * 4]) * s - width / 2.0, y: (rf + bbox[i * 4 + 1]) * s - height / 2.0, width, height, score, points });
            }
        }
    }
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for face in found {
        if kept.iter().all(|k| overlap(k, &face) < SAME_FACE) {
            kept.push(face);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    /// All the outputs for an empty picture, with one strong cell at `cell` of the first scale.
    fn outputs(cell: Option<(usize, usize)>) -> [[Vec<f32>; 4]; 3] {
        let mut out = STRIDES.map(|s| {
            let n = (WIDTH / s) * (HEIGHT / s);
            [vec![-20.0; n], vec![-20.0; n], vec![0.0; n * 4], vec![0.0; n * 10]]
        });
        if let Some((r, c)) = cell {
            let i = r * WIDTH / 8 + c;
            out[0][0][i] = 10.0;
            out[0][1][i] = 10.0;
            out[0][2][i * 4] = 0.5;
            out[0][2][i * 4 + 2] = 1.0;
            out[0][3][i * 10 + 2] = 0.25;
        }
        out
    }

    #[test]
    fn decodes_a_cell_into_pixels() {
        let faces = decode(&outputs(Some((10, 20))));
        assert_eq!(faces.len(), 1);
        let f = faces[0];
        let width = 1f32.exp() * 8.0;
        assert!((f.width - width).abs() < 1e-3);
        assert!((f.x - (20.5 * 8.0 - width / 2.0)).abs() < 1e-3, "{f:?}");
        assert!((f.y - (10.0 * 8.0 - 4.0)).abs() < 1e-3, "{f:?}");
        assert!((f.points[1][0] - 20.25 * 8.0).abs() < 1e-3, "{f:?}");
        assert!(f.score > 0.99);
    }

    #[test]
    fn nothing_is_found_in_nothing() {
        assert!(decode(&outputs(None)).is_empty());
    }

    #[test]
    fn overlapping_boxes_are_one_face() {
        let mut out = outputs(Some((10, 20)));
        let i = 10 * WIDTH / 8 + 21;
        out[0][0][i] = 2.0;
        out[0][1][i] = 2.0;
        out[0][2][i * 4 + 2] = 1.0;
        let faces = decode(&out);
        assert_eq!(faces.len(), 1, "{faces:?}");
        assert!(faces[0].score > 0.99, "the stronger one stays");
    }

    /// Compares the outputs with ONNX Runtime's (needs a GPU, and the files from
    /// tools/face/convert.py --reference):
    /// `FLEX_FACE_RGB=out.rgb FLEX_FACE_REF=out.bin cargo test -p flex-depth matches_onnx -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn matches_onnx() {
        let rgb = std::fs::read(std::env::var("FLEX_FACE_RGB").expect("FLEX_FACE_RGB")).unwrap();
        let reference = std::fs::read(std::env::var("FLEX_FACE_REF").expect("FLEX_FACE_REF")).unwrap();
        let reference: Vec<f32> = reference.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let started = std::time::Instant::now();
        let mut face = Face::new().unwrap();
        eprintln!("loaded in {:?} on {}", started.elapsed(), face.adapter());
        let outputs = face.outputs_for(&rgb).unwrap();
        let mut at = 0;
        let mut worst = 0.0f32;
        for (scale, stride) in outputs.iter().zip(STRIDES) {
            for (kind, values) in ["cls", "obj", "bbox", "kps"].iter().zip(scale) {
                // ONNX applies the sigmoid to the scores.
                let got: Vec<f32> = if *kind == "cls" || *kind == "obj" { values.iter().map(|v| sigmoid(*v)).collect() } else { values.clone() };
                let expected = &reference[at..at + got.len()];
                at += got.len();
                let err = got.iter().zip(expected).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
                eprintln!("{kind}_{stride:<3} max error {err:.2e}");
                worst = worst.max(err);
            }
        }
        assert_eq!(at, reference.len());
        assert!(worst < 5e-3, "worst error {worst:.2e}");
        let runs = 30;
        let started = std::time::Instant::now();
        for _ in 0..runs {
            face.run(&rgb).unwrap();
        }
        eprintln!("{:.1} ms a picture", started.elapsed().as_secs_f64() * 1000.0 / runs as f64);
        eprintln!("{:#?}", face.run(&rgb).unwrap());
    }
}
