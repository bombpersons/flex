"""Trains the scene and body-part head for flex-depth's MoGe-2 as a Kaggle script kernel (or,
with SEG_FETCH=1, anywhere: it was run on Colab).

    kaggle kernels push -p crates/flex-depth/tools/moge/segment     # start
    kaggle kernels status bombpersons/flex-moge-segment-train
    kaggle kernels output bombpersons/flex-moge-segment-train -p runs/moge-segment

The same head as tools/segment's, on MoGe-2 ViT-S instead of Depth Anything V2, which stays
frozen: it reads the two transformer layers' tokens the neck reads (normed) and the neck's
64-channel features at 8x the patch grid, which the depth pass has already worked out. It's
built only from what depth.wgsl can run (1x1 and 3x3 convolutions, ReLU, align_corners
bilinear resizes, additions).

It gives two sets of answers. What each pixel is, of SCENE: from COCO panoptic and ADE20K's
150-class Scene Parsing set (two fifths of the pictures each), with their classes mapped by
name. And, for a person's pixels, which of PARTS: from ATR's human parsing labels (the other
fifth: one person each, full length, in fashion photos), where only the person counts for SCENE.
COCO's panoptic labels come from cocodataset.org, ATR from Hugging Face (mattmdjaga/
human_parsing_dataset), the depth model from Hugging Face at the revision Flex uses.
Everything in /kaggle/working/out comes back as output: head.safetensors (the best so far),
metrics.json, and samples_*.jpg (validation pictures with their masks).

Scored separately on each dataset's validation set, as IoU (ATR's: its last 500 pictures). SEG_ROOT, SEG_OUT, SEG_PANOPTIC,
SEG_MINUTES, SEG_BATCH and SEG_EVAL_MINUTES are for trying it out away from Kaggle; with
SEG_FETCH=1 it downloads the datasets itself (as on Colab).
"""

import glob
import io
import json
import math
import os
import random
import subprocess
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset, Sampler

START = time.time()
MINUTES = float(os.environ.get("SEG_MINUTES", 5 * 60))
EVAL_MINUTES = float(os.environ.get("SEG_EVAL_MINUTES", 30))
BATCH = int(os.environ.get("SEG_BATCH", 32))
LR, WEIGHT_DECAY, WARMUP = 1e-3, 1e-4, 200
ROOT = os.environ.get("SEG_ROOT", "/kaggle/input")
OUT = os.environ.get("SEG_OUT", "/kaggle/working/out")
PANOPTIC = os.environ.get("SEG_PANOPTIC", "/tmp/panoptic")
PANOPTIC_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"

MODEL = "Ruicheng/moge-2-vits-normal"
REVISION = "26b477f41595707c5db6770294c0d1721e8ed4ed"
MOGE_CODE = "https://github.com/microsoft/MoGe.git"
MOGE_COMMIT = "74fbce0"
MOGE_DIR = os.environ.get("SEG_MOGE", "/tmp/MoGe")
# The grid crates/flex-app/src/depth.rs runs the model at.
PATCH, GW, GH = 14, 25, 14
WIDTH, HEIGHT = GW * PATCH, GH * PATCH
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

SCENE = ["other", "person", "wall", "sky", "ground", "ceiling", "water", "plant", "window"]
OTHER, PERSON, WALL, SKY, GROUND, CEILING, WATER, PLANT, WINDOW = range(len(SCENE))
PARTS = ["skin", "hair", "clothes"]
SKIN, HAIR, CLOTHES = range(len(PARTS))
CLASSES = SCENE + PARTS
IGNORE = 255
SCENE_WEIGHTS = torch.tensor([0.5] + [1.0] * (len(SCENE) - 1))
COLOURS = np.array([[0, 0, 0], [255, 60, 60], [60, 110, 255], [60, 230, 230], [150, 110, 60], [200, 200, 120], [40, 90, 255], [60, 200, 60], [255, 220, 80]], dtype=np.float32)
PART_COLOURS = np.array([[255, 190, 150], [120, 60, 20], [180, 60, 220]], dtype=np.float32)
ATR = "mattmdjaga/human_parsing_dataset"
ATR_REVISION = os.environ.get("SEG_ATR_REVISION", "main")
# ATR's labels: 0 background, 1 hat, 2 hair, 3 sunglasses, 4 upper clothes, 5 skirt, 6 pants,
# 7 dress, 8 belt, 9 and 10 shoes, 11 face, 12 and 13 legs, 14 and 15 arms, 16 bag, 17 scarf.
ATR_PART = np.full(256, IGNORE, dtype=np.uint8)
ATR_PART[[11, 12, 13, 14, 15]] = SKIN
ATR_PART[2] = HAIR
ATR_PART[[1, 4, 5, 6, 7, 8, 9, 10, 17]] = CLOTHES
ATR_SCENE = np.full(256, PERSON, dtype=np.uint8)
ATR_SCENE[[0, 16]] = IGNORE  # The background could be anything; a bag isn't the person.


def scene_of(name):
    """SCENE's class for a COCO or ADE20K class name (ADE20K's are lists of synonyms)."""
    n = name.split(",")[0].strip().lower()
    if n in ("person",):
        return PERSON
    if n.startswith("wall"):
        return WALL
    if n == "sky" or n.startswith("sky-"):
        return SKY
    if n.startswith("floor") or n in ("rug-merged", "rug", "pavement-merged", "sidewalk", "road", "dirt-merged", "earth", "sand", "gravel", "snow",
                                      "playingfield", "field", "railroad", "platform", "grass-merged", "grass", "path", "runway", "dirt track", "land"):
        return GROUND
    if n.startswith("ceiling"):
        return CEILING
    if n in ("water-other", "water", "sea", "river", "lake", "swimming pool", "waterfall"):
        return WATER
    if n in ("tree-merged", "tree", "flower", "potted plant", "plant", "palm"):
        return PLANT
    if n.startswith("window"):
        return WINDOW
    return OTHER
SAMPLES = 8

device = "cuda" if torch.cuda.is_available() else "cpu"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(r.stdout[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"failed ({r.returncode}): {cmd}")


def find(tail):
    """A mounted dataset's folder, however deep Kaggle put it (without walking the pictures)."""
    for depth in range(1, 5):
        found = glob.glob(os.path.join(ROOT, *["*"] * depth, tail))
        if found:
            return found[0]
    raise FileNotFoundError(f"{tail} isn't under {ROOT}: {os.listdir(ROOT)}")


COCO_ZIPS = ["http://images.cocodataset.org/zips/train2017.zip", "http://images.cocodataset.org/zips/val2017.zip"]
ADE_ZIP = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"


def fetch_data():
    """Away from Kaggle (SEG_FETCH=1): COCO's pictures and ADE20K from where they're published,
    laid out under ROOT as Kaggle's datasets are."""
    coco = f"{ROOT}/coco/coco2017"
    os.makedirs(coco, exist_ok=True)
    for url in COCO_ZIPS:
        split = url.rsplit("/", 1)[1][:-4]
        if not os.path.exists(f"{coco}/{split}"):
            sh(f"curl -sSfL -o /tmp/{split}.zip {url} && unzip -q /tmp/{split}.zip -d {coco} && rm /tmp/{split}.zip")
    if not os.path.exists(f"{ROOT}/ade/ADEChallengeData2016"):
        os.makedirs(f"{ROOT}/ade", exist_ok=True)
        sh(f"curl -sSfL -o /tmp/ade.zip {ADE_ZIP} && unzip -q /tmp/ade.zip -d {ROOT}/ade && rm /tmp/ade.zip")


def get_panoptic():
    """COCO's panoptic annotations, unpacked: a zip of two JSON files and two more zips."""
    if os.path.exists(f"{PANOPTIC}/annotations/panoptic_val2017"):
        return
    os.makedirs(PANOPTIC, exist_ok=True)
    sh(f"curl -sSfL -o {PANOPTIC}/panoptic.zip {PANOPTIC_URL}")
    with zipfile.ZipFile(f"{PANOPTIC}/panoptic.zip") as z:
        z.extractall(PANOPTIC)
    for split in ("train2017", "val2017"):
        with zipfile.ZipFile(f"{PANOPTIC}/annotations/panoptic_{split}.zip") as z:
            z.extractall(f"{PANOPTIC}/annotations")
        os.remove(f"{PANOPTIC}/annotations/panoptic_{split}.zip")
    os.remove(f"{PANOPTIC}/panoptic.zip")


# ---- Data ------------------------------------------------------------------------------------


def coco_items(split, images):
    """(picture, label image, sorted segment ids, their classes) for each picture."""
    with open(f"{PANOPTIC}/annotations/panoptic_{split}.json") as f:
        data = json.load(f)
    classes = {c["id"]: scene_of(c["name"]) for c in data["categories"]}
    print("COCO:", {SCENE[k]: sorted(c["name"] for c in data["categories"] if classes[c["id"]] == k) for k in range(1, len(SCENE))}, flush=True)
    counts = [sum(v == c for v in classes.values()) for c in (PERSON, WALL, SKY)]
    assert counts[0] == 1 and counts[1] >= 5 and counts[2] == 1, counts
    items = []
    for a in data["annotations"]:
        # Crowds are drawn as one rough blob, so they don't count either way.
        segments = sorted((s["id"], IGNORE if s.get("iscrowd") else classes[s["category_id"]]) for s in a["segments_info"])
        if segments:
            ids, cls = zip(*segments)
            items.append((
                f"{images}/{a['file_name'].replace('.png', '.jpg')}",
                f"{PANOPTIC}/annotations/panoptic_{split}/{a['file_name']}",
                np.array(ids, dtype=np.int64),
                np.array(cls, dtype=np.uint8),
            ))
    return items


def ade_items(split, root):
    labels = sorted(glob.glob(f"{root}/annotations/{split}/*.png"))
    return [(p.replace("/annotations/", "/images/").replace(".png", ".jpg"), p, None, None) for p in labels]


ADE_LUT = np.full(256, OTHER, dtype=np.uint8)
ADE_LUT[0] = IGNORE


def ade_classes(root):
    """ADE20K's label values to SCENE, by the names in its objectInfo150.txt."""
    names = {}
    with open(f"{root}/objectInfo150.txt") as f:
        for line in f.readlines()[1:]:
            parts = line.rstrip("\n").split("\t")
            names[int(parts[0])] = parts[-1]
    for i, n in names.items():
        ADE_LUT[i] = scene_of(n)
    print("ADE20K:", {SCENE[k]: [names[i].split(",")[0] for i in names if ADE_LUT[i] == k] for k in range(1, len(SCENE))}, flush=True)
    assert ADE_LUT[1] == WALL and ADE_LUT[3] == SKY and ADE_LUT[13] == PERSON and ADE_LUT[4] == GROUND, "ADE20K's label numbers aren't what this expects"


# ATR's pictures and labels, as encoded bytes: [(picture, label)].
ATR_DATA = []


def atr_items():
    """ATR, all of it in memory (about 0.8 GB), as items for Pictures."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(f for f in HfApi().list_repo_files(ATR, repo_type="dataset", revision=ATR_REVISION) if f.endswith(".parquet"))
    for name in files:
        table = pq.read_table(hf_hub_download(ATR, name, repo_type="dataset", revision=ATR_REVISION)).to_pydict()
        ATR_DATA.extend((i["bytes"], m["bytes"]) for i, m in zip(table["image"], table["mask"]))
    items = [("atr", i, None, None) for i in range(len(ATR_DATA))]
    return items[:-500], items[-500:]


def to_classes(label, ids, cls):
    """SCENE and PARTS labels for a label image: [2][h][w]."""
    a = np.asarray(label)
    if isinstance(ids, str):
        return np.stack([ATR_SCENE[a], ATR_PART[a]])
    unknown = np.full(a.shape[:2], IGNORE, dtype=np.uint8)
    if ids is None:
        return np.stack([ADE_LUT[a], unknown])
    # COCO: each pixel's colour is a segment's id. Id 0 and anything unlisted is unlabelled.
    v = a[..., 0].astype(np.int64) + a[..., 1].astype(np.int64) * 256 + a[..., 2].astype(np.int64) * 65536
    at = np.minimum(np.searchsorted(ids, v), len(ids) - 1)
    out = cls[at].copy()
    out[ids[at] != v] = IGNORE
    return np.stack([out, unknown])


def grade(img):
    """Differences between snapshots and what films look like: exposure, colour grading,
    softness and compression."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.6, 1.3))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.5, 1.4))
    gamma = random.uniform(0.8, 1.25)
    lut = []
    for _ in range(3):
        gain = random.uniform(0.9, 1.1)
        lut += [min(255, int(255 * (i / 255) ** gamma * gain)) for i in range(256)]
    img = img.point(lut)
    if random.random() < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.5)))
    if random.random() < 0.3:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(20, 70))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img


class Pictures(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label_path, ids, cls = self.items[i]
        if path == "atr":
            picture, mask = ATR_DATA[label_path]
            img = Image.open(io.BytesIO(picture)).convert("RGB")
            label = Image.open(io.BytesIO(mask))
            ids = "atr"
        else:
            img = Image.open(path).convert("RGB")
            label = Image.open(label_path)
        w, h = img.size
        # The largest crop of the model's shape; smaller at random when training.
        cw = min(w, h * WIDTH / HEIGHT)
        ch = cw * HEIGHT / WIDTH
        if self.train:
            s = random.uniform(0.5, 1.0)
            cw, ch = cw * s, ch * s
            x, y = random.uniform(0, w - cw), random.uniform(0, h - ch)
        else:
            x, y = (w - cw) / 2, (h - ch) / 2
        # cw * HEIGHT / WIDTH can round to a hair over the picture's size.
        x, y = max(x, 0.0), max(y, 0.0)
        box = (x, y, min(x + cw, w), min(y + ch, h))
        img = img.resize((WIDTH, HEIGHT), Image.BICUBIC, box=box)
        label = label.resize((WIDTH, HEIGHT), Image.NEAREST, box=box)
        if self.train:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            img = grade(img)
        rgb = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        return rgb, torch.from_numpy(to_classes(label, ids, cls).copy()).long()


class Mix(Sampler):
    """Forever: items from consecutive datasets of `sizes`, each picked with its share of
    `weights` however big it is."""

    def __init__(self, sizes, weights):
        self.sizes, self.weights = sizes, weights

    def __iter__(self):
        r = random.Random(0)
        starts = np.cumsum([0] + list(self.sizes))
        while True:
            k = r.choices(range(len(self.sizes)), self.weights)[0]
            yield int(starts[k]) + r.randrange(self.sizes[k])

    def __len__(self):
        return 1 << 40


# ---- Model -----------------------------------------------------------------------------------


class Head(nn.Module):
    """Every step here has a kernel in depth.wgsl, with bilinear resizes as align_corners."""

    def __init__(self):
        super().__init__()
        self.token9 = nn.Conv2d(384, 128, 1)
        self.token12 = nn.Conv2d(384, 128, 1)
        self.context = nn.Conv2d(128, 128, 3, padding=1)
        self.mid = nn.Conv2d(128, 64, 3, padding=1)
        self.fused = nn.Conv2d(64, 64, 1)
        # Wider at the end than tools/segment's (48), for twelve answers instead of four.
        self.fine1 = nn.Conv2d(64, 64, 3, padding=1)
        self.fine2 = nn.Conv2d(64, 64, 3, padding=1)
        self.out = nn.Conv2d(64, len(CLASSES), 1)

    def forward(self, t9, t12, fused):
        x = F.relu(self.context(F.relu(self.token9(t9) + self.token12(t12))))
        x = F.interpolate(x, size=(GH * 4, GW * 4), mode="bilinear", align_corners=True)
        x = F.relu(self.mid(x))
        x = F.interpolate(x, size=fused.shape[2:], mode="bilinear", align_corners=True)
        x = F.relu(self.fine1(F.relu(x + self.fused(fused))))
        return self.out(F.relu(self.fine2(x)))


def grid(tokens):
    """Tokens without the class token, as an image."""
    b = tokens.shape[0]
    return tokens.transpose(1, 2).reshape(b, 384, GH, GW).float()


def load_moge():
    """MoGe-2 from its own code (with the libraries it needs only for other things stood in
    for) and its weights from Hugging Face, at the versions flex-depth was checked against."""
    import importlib.abc
    import importlib.machinery
    import sys
    import types

    class Absent(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        roots = ("scipy", "cv2", "utils3d", "utils3d_moge", "pipeline", "flex_gemm", "trimesh", "gradio")

        def find_spec(self, name, path, target=None):
            if name.split(".")[0] in self.roots:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)

        def create_module(self, spec):
            m = types.ModuleType(spec.name)
            m.__path__ = []
            m.__getattr__ = lambda attr: None
            return m

        def exec_module(self, module):
            pass

    if not os.path.exists(f"{MOGE_DIR}/moge"):
        sh(f"git clone -q {MOGE_CODE} {MOGE_DIR} && git -C {MOGE_DIR} checkout -q {MOGE_COMMIT}")
    sys.meta_path.insert(0, Absent())
    sys.path.insert(0, MOGE_DIR)
    from huggingface_hub import hf_hub_download
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained(hf_hub_download(MODEL, "model.pt", revision=REVISION))


@torch.no_grad()
def features(depth, rgb):
    """What flex-depth's MoGe pass has: the two taps' normed tokens, and the neck's level 3."""
    x = rgb.float() / 255
    enc = depth.encoder
    with torch.autocast(device, enabled=device == "cuda"):
        x14 = (x - enc.image_mean) / enc.image_std
        taps = enc.backbone.get_intermediate_layers(x14, n=enc.intermediate_layers, return_class_token=True)
        projected = sum(proj(t.permute(0, 2, 1).unflatten(2, (GH, GW)).contiguous()) for proj, (t, _) in zip(enc.output_projections, taps))
        neck_in = [projected, None, None, None, None]
        for level in range(5):
            uv = normalized_view_plane_uv(width=GW * 2**level, height=GH * 2**level, aspect_ratio=GW / GH, dtype=x.dtype, device=x.device)
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            neck_in[level] = uv if neck_in[level] is None else torch.cat([neck_in[level], uv], 1)
        fine = depth.neck(neck_in)[3]
    assert fine.shape[1:] == (64, GH * 8, GW * 8), fine.shape
    return grid(taps[0][0]), grid(taps[1][0]), fine.float()


def logits_of(head, feats):
    return F.interpolate(head(*feats), size=(HEIGHT, WIDTH), mode="bilinear", align_corners=True)


def part_of(logits, label, classes, weights):
    """Cross-entropy keeps soft edges honest; Dice on every class but the first keeps small
    things from being lost in the big ones. Nothing where nothing is labelled."""
    known = label != IGNORE
    if not known.any():
        return logits.sum() * 0.0
    ce = F.cross_entropy(logits, label, weight=weights, ignore_index=IGNORE)
    p = logits.softmax(1)
    dice = 0.0
    for c in classes:
        want = (label == c) & known
        inter = (p[:, c] * want).sum()
        dice = dice + 1 - (2 * inter + 1) / ((p[:, c] * known).sum() + want.sum() + 1)
    return ce + dice / len(classes)


def loss_of(logits, y):
    """`y` is [batch][2][h][w]: SCENE's labels, then PARTS' (on a person's pixels)."""
    scene, parts = logits[:, : len(SCENE)], logits[:, len(SCENE) :]
    return part_of(scene, y[:, 0], range(1, len(SCENE)), SCENE_WEIGHTS.to(y.device)) + part_of(parts, y[:, 1], range(len(PARTS)), None)


# ---- Checking --------------------------------------------------------------------------------


def check_labels(name, items):
    """How much of the pictures each class covers: a wrong mapping shows up here, cheaply."""
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    data = Pictures(items[:: max(1, len(items) // 300)], False)
    for _, y in DataLoader(data, batch_size=16, num_workers=os.cpu_count()):
        for c in range(len(SCENE)):
            counts[c] += int((y[:, 0] == c).sum())
        for c in range(len(PARTS)):
            counts[len(SCENE) + c] += int((y[:, 1] == c).sum())
    share = counts / max(1, counts[: len(SCENE)].sum())
    print(name, {c: round(float(s), 3) for c, s in zip(CLASSES, share)}, flush=True)
    return share


@torch.no_grad()
def evaluate(depth, head, items, parts):
    """IoU of each of SCENE's classes but "other", or (with `parts`) of each of PARTS."""
    head.eval()
    names = PARTS if parts else SCENE
    inter, union = np.zeros(len(names)), np.zeros(len(names))
    for rgb, y in DataLoader(Pictures(items, False), batch_size=BATCH, num_workers=os.cpu_count()):
        rgb, y = rgb.to(device), y.to(device)
        logits = logits_of(head, features(depth, rgb))
        if parts:
            p, y = logits[:, len(SCENE) :].argmax(1), y[:, 1]
        else:
            p, y = logits[:, : len(SCENE)].argmax(1), y[:, 0]
        known = y != IGNORE
        for c in range(len(names)):
            a, b = (p == c) & known, (y == c) & known
            inter[c] += int((a & b).sum())
            union[c] += int((a | b).sum())
    head.train()
    return {c: round(float(i / max(u, 1)), 4) for c, i, u in zip(names, inter, union) if c != "other"}


@torch.no_grad()
def save_samples(depth, head, items, path):
    """Each picture beside its classes, with a person's parts shown over their person."""
    head.eval()
    data = Pictures(items, False)
    rows = []
    for i in range(0, len(items), max(1, len(items) // SAMPLES)):
        rgb, _ = data[i]
        logits = logits_of(head, features(depth, rgb[None].to(device)))[0]
        scene = logits[: len(SCENE)].softmax(0).cpu().numpy()
        parts = logits[len(SCENE) :].softmax(0).cpu().numpy() * scene[PERSON]
        picture = rgb.permute(1, 2, 0).numpy().astype(np.float32)
        tint = (COLOURS[:, None, None] * scene[:, :, :, None]).sum(0)
        alpha = 1.0 - scene[OTHER]
        over = picture * (1 - 0.6 * alpha[:, :, None]) + tint * 0.6
        body = (PART_COLOURS[:, None, None] * parts[:, :, :, None]).sum(0)
        on = parts.sum(0)[:, :, None]
        people = picture * (1 - 0.7 * on) + body * 0.7
        rows.append(np.concatenate([picture, over, people], 1))
        if len(rows) == SAMPLES:
            break
    head.train()
    Image.fromarray(np.concatenate(rows, 0).clip(0, 255).astype(np.uint8)).save(path, quality=85)


def main():
    os.makedirs(OUT, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true; nproc; df -h /tmp | tail -1")
    if os.environ.get("SEG_FETCH"):
        fetch_data()
    get_panoptic()
    coco = find("coco2017")
    ade = find("ADEChallengeData2016")
    coco_train, coco_val = coco_items("train2017", f"{coco}/train2017"), coco_items("val2017", f"{coco}/val2017")
    ade_train, ade_val = ade_items("training", ade), ade_items("validation", ade)
    ade_classes(ade)
    atr_train, atr_val = atr_items()
    print(f"COCO {len(coco_train)} + {len(coco_val)}, ADE20K {len(ade_train)} + {len(ade_val)}, ATR {len(atr_train)} + {len(atr_val)}", flush=True)
    assert coco_train and coco_val and ade_train and ade_val and atr_train

    check_labels("COCO", coco_train)
    share = check_labels("ADE20K", ade_train)
    assert share[WALL] > share[SKY] > share[PERSON] > 0, "ADE20K's label numbers aren't what this expects"
    share = check_labels("ATR", atr_train)
    assert share[len(SCENE) + CLOTHES] > share[len(SCENE) + SKIN] > share[len(SCENE) + HAIR] > 0, "ATR's labels aren't what this expects"

    depth = load_moge().eval().to(device).requires_grad_(False)
    global normalized_view_plane_uv
    from moge.utils.geometry_torch import normalized_view_plane_uv
    head = Head().to(device)
    params = sum(p.numel() for p in head.parameters())
    print(f"head: {params} weights, {params * 4 / 1e6:.2f} MB", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    data = torch.utils.data.ConcatDataset([Pictures(coco_train, True), Pictures(ade_train, True), Pictures(atr_train, True)])
    loader = DataLoader(
        data, batch_size=BATCH, sampler=Mix([len(coco_train), len(ade_train), len(atr_train)], [0.4, 0.4, 0.2]), num_workers=os.cpu_count(),
        pin_memory=device == "cuda", persistent_workers=True, prefetch_factor=4,
    )

    began = time.time()
    total = MINUTES * 60 - (began - START)
    print(f"setup took {(began - START) / 60:.1f} min, training for {total / 60:.0f} min", flush=True)
    history, best, step, running, shown = [], -1.0, 0, 0.0, time.time()
    next_eval = began + EVAL_MINUTES * 60

    def checkpoint():
        nonlocal best
        # Half of COCO's validation pictures, to keep this quick.
        scores = {"COCO": evaluate(depth, head, coco_val[::2], False), "ADE20K": evaluate(depth, head, ade_val, False), "ATR": evaluate(depth, head, atr_val, True)}
        score = float(np.mean([np.mean(list(s.values())) for s in scores.values()]))
        minutes = (time.time() - began) / 60
        history.append({"step": step, "minutes": round(minutes, 1), "score": round(score, 4), **scores})
        print(f"step {step} ({minutes:.0f} min): score {score:.4f} {scores}", flush=True)
        with open(f"{OUT}/metrics.json", "w") as f:
            json.dump(history, f, indent=1)
        save_samples(depth, head, coco_val, f"{OUT}/samples_{step:07d}_coco.jpg")
        save_samples(depth, head, ade_val, f"{OUT}/samples_{step:07d}_ade.jpg")
        save_samples(depth, head, atr_val, f"{OUT}/samples_{step:07d}_atr.jpg")
        if score > best:
            best = score
            save_file({k: v.cpu().contiguous() for k, v in head.state_dict().items()}, f"{OUT}/head.safetensors", {"scene": ",".join(SCENE), "parts": ",".join(PARTS), "model": MODEL, "revision": REVISION})
            print("best so far: saved", flush=True)

    for rgb, y in loader:
        now = time.time()
        frac = (now - began) / max(total, 1)
        if frac >= 1:
            break
        lr = LR * 0.5 * (1 + math.cos(math.pi * frac)) * min(1.0, (step + 1) / WARMUP)
        for g in opt.param_groups:
            g["lr"] = lr
        rgb, y = rgb.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = loss_of(logits_of(head, features(depth, rgb)), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        running += loss.item()
        if step % 100 == 0:
            print(f"step {step} loss {running / 100:.4f} lr {lr:.2e} {100 * BATCH / (time.time() - shown):.0f} pictures/s", flush=True)
            running, shown = 0.0, time.time()
        if now >= next_eval:
            checkpoint()
            next_eval = time.time() + EVAL_MINUTES * 60
    checkpoint()
    print(f"done: best score {best:.4f}", flush=True)
    # The workers wait forever for the endless sampler otherwise.
    os._exit(0)


if __name__ == "__main__":
    main()
