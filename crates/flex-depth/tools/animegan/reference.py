"""Reference outputs for flex-depth's AnimeGANv3 tests: runs one of AnimeGANv3's ONNX files
(github.com/TachibanaYoshino/AnimeGANv3 releases) with ONNX Runtime on an image, and saves the
input with the output of each block, so the compute shaders can be checked stage by stage.

    python reference.py <style.onnx> <image> <out.safetensors> [width height]

Needs onnx, onnxruntime, safetensors, pillow and numpy. The picture is resized to width x height
(default 256 x 160; both multiples of 8) and saved as RGB bytes with the rest. Stages are
channels first, [channels][height][width], as flex-depth keeps them.
"""

import sys

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
from safetensors.numpy import save_file

# What flex-depth calls each stage, and the graph's name for it after "generator_1/" (or
# "generator/" in some styles).
STAGES = {
    **{f"down.{i}": f"base/LeakyRelu{'' if i == 0 else f'_{i}'}:0" for i in range(7)},
    "attention": "main/External_attention/LeakyRelu:0",
    **{f"up.{i}": f"main/LeakyRelu{'' if i == 0 else f'_{i}'}:0" for i in range(6)},
    "out": "main/out_layer_raw_output___4:0",
}
# The large network's (Ghibli-c1's), after "generator/".
LARGE = {
    **{f"base.{i}": f"base/LeakyRelu{'' if i == 0 else f'_{i}'}:0" for i in (0, 3, 6, 9, 27)},
    "main.add": "main/add:0",
    "main.add_1": "main/add_1:0",
    "main.add_2": "main/add_2:0",
    "main.7": "main/LeakyRelu_7:0",
    "main.8": "main/LeakyRelu_8:0",
    "out": "main/out_layer_raw_output___4:0",
}


def main():
    model, image, out = sys.argv[1], sys.argv[2], sys.argv[3]
    width, height = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (256, 160)
    assert width % 8 == 0 and height % 8 == 0, "sizes must be multiples of 8"
    graph = onnx.load(model)
    known = {o for n in graph.graph.node for o in n.output}
    prefix = next(p for p in ("generator_1/", "generator/") if p + STAGES["out"] in known)
    names = LARGE if prefix + LARGE["base.27"] in known else STAGES
    stages = {stage: prefix + name for stage, name in names.items()}
    for name in stages.values():
        assert name in known, f"no {name} in the graph"
        graph.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    session = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"])
    rgb = np.asarray(Image.open(image).convert("RGB").resize((width, height), Image.BICUBIC))
    x = rgb.astype(np.float32)[None] / 127.5 - 1.0
    names = [o.name for o in session.get_outputs()]
    results = dict(zip(names, session.run(names, {session.get_inputs()[0].name: x})))
    saved = {"rgb": np.ascontiguousarray(rgb)}
    for stage, name in stages.items():
        v = results[name][0].astype(np.float32)
        if v.shape[:2] == (height, width):
            v = v.transpose(2, 0, 1)
        saved[stage] = np.ascontiguousarray(v)
    save_file(saved, out)
    for k, v in saved.items():
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
