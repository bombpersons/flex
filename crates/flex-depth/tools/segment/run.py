"""Trains the person / wall / sky head for flex-depth as a Kaggle script kernel.

    kaggle kernels push -p crates/flex-depth/tools/segment     # start
    kaggle kernels status bombpersons/flex-segment-train
    kaggle kernels output bombpersons/flex-segment-train -p runs/segment

The head sits on Depth Anything V2 Small, which stays frozen: it reads the last two
transformer layers' tokens and the finest fused feature map, which the depth pass has
already worked out. It's built only from what depth.wgsl can run (1x1 and 3x3 convolutions,
ReLU, align_corners bilinear resizes, additions).

Half of each batch's pictures come from COCO panoptic and half from ADE20K's 150-class
Scene Parsing set, all mapped to other / person / wall / sky. COCO's panoptic labels come
from cocodataset.org, the depth model from Hugging Face at the revision Flex uses.
Everything in /kaggle/working/out comes back as output: head.safetensors (the best so far),
metrics.json, and samples_*.jpg (validation pictures with their masks).

Scored separately on each dataset's validation set, as IoU. SEG_ROOT, SEG_OUT, SEG_PANOPTIC,
SEG_MINUTES, SEG_BATCH and SEG_EVAL_MINUTES are for trying it out away from Kaggle.
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
from transformers import AutoModelForDepthEstimation

START = time.time()
MINUTES = float(os.environ.get("SEG_MINUTES", 5 * 60))
EVAL_MINUTES = float(os.environ.get("SEG_EVAL_MINUTES", 30))
BATCH = int(os.environ.get("SEG_BATCH", 32))
LR, WEIGHT_DECAY, WARMUP = 1e-3, 1e-4, 200
ROOT = os.environ.get("SEG_ROOT", "/kaggle/input")
OUT = os.environ.get("SEG_OUT", "/kaggle/working/out")
PANOPTIC = os.environ.get("SEG_PANOPTIC", "/tmp/panoptic")
PANOPTIC_URL = "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip"

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
# The grid crates/flex-app/src/depth.rs runs the model at.
PATCH, GW, GH = 14, 37, 21
WIDTH, HEIGHT = GW * PATCH, GH * PATCH
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

CLASSES = ["other", "person", "wall", "sky"]
OTHER, PERSON, WALL, SKY = 0, 1, 2, 3
IGNORE = 255
# ADE20K's label image values (0 is unlabelled).
ADE_WALL, ADE_SKY, ADE_PERSON = 1, 3, 13
CLASS_WEIGHTS = torch.tensor([0.5, 1.0, 1.0, 1.0])
COLOURS = np.array([[0, 0, 0], [255, 60, 60], [60, 110, 255], [60, 230, 230]], dtype=np.float32)
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
    classes = {}
    for c in data["categories"]:
        n = c["name"]
        classes[c["id"]] = PERSON if n == "person" else WALL if n.startswith("wall-") else SKY if n.startswith("sky-") else OTHER
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
ADE_LUT[ADE_WALL], ADE_LUT[ADE_SKY], ADE_LUT[ADE_PERSON] = WALL, SKY, PERSON


def to_classes(label, ids, cls):
    a = np.asarray(label)
    if ids is None:
        return ADE_LUT[a]
    # COCO: each pixel's colour is a segment's id. Id 0 and anything unlisted is unlabelled.
    v = a[..., 0].astype(np.int64) + a[..., 1].astype(np.int64) * 256 + a[..., 2].astype(np.int64) * 65536
    at = np.minimum(np.searchsorted(ids, v), len(ids) - 1)
    out = cls[at].copy()
    out[ids[at] != v] = IGNORE
    return out


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


class Half(Sampler):
    """Forever: a coin flip between the first `split` items and the rest, so the smaller
    dataset counts as much as the larger."""

    def __init__(self, split, total):
        self.split, self.total = split, total

    def __iter__(self):
        r = random.Random(0)
        while True:
            yield r.randrange(self.split) if r.random() < 0.5 else r.randrange(self.split, self.total)

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
        self.fine1 = nn.Conv2d(64, 48, 3, padding=1)
        self.fine2 = nn.Conv2d(48, 48, 3, padding=1)
        self.out = nn.Conv2d(48, len(CLASSES), 1)

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
    return tokens[:, 1:].transpose(1, 2).reshape(b, 384, GH, GW).float()


@torch.no_grad()
def features(depth, rgb):
    x = (rgb.float() / 255 - MEAN.to(device)) / STD.to(device)
    with torch.autocast(device, enabled=device == "cuda"):
        maps = depth.backbone(pixel_values=x).feature_maps
        fused = depth.neck(list(maps), patch_height=GH, patch_width=GW)[-1]
    assert maps[3].shape[1:] == (1 + GW * GH, 384) and fused.shape[1:] == (64, GH * 8, GW * 8), (maps[3].shape, fused.shape)
    return grid(maps[2]), grid(maps[3]), fused.float()


def logits_of(head, feats):
    return F.interpolate(head(*feats), size=(HEIGHT, WIDTH), mode="bilinear", align_corners=True)


def loss_of(logits, y):
    """Cross-entropy keeps the masks' soft edges honest; Dice on the three classes that
    matter keeps small people from being lost in the "other" pixels."""
    ce = F.cross_entropy(logits, y, weight=CLASS_WEIGHTS.to(y.device), ignore_index=IGNORE)
    p = logits.softmax(1)
    known = y != IGNORE
    dice = 0.0
    for c in (PERSON, WALL, SKY):
        want = (y == c) & known
        inter = (p[:, c] * want).sum()
        dice = dice + 1 - (2 * inter + 1) / ((p[:, c] * known).sum() + want.sum() + 1)
    return ce + dice / 3


# ---- Checking --------------------------------------------------------------------------------


def check_labels(name, items):
    """How much of the pictures each class covers: a wrong mapping shows up here, cheaply."""
    counts = np.zeros(len(CLASSES), dtype=np.int64)
    data = Pictures(items[:: max(1, len(items) // 300)], False)
    for _, y in DataLoader(data, batch_size=16, num_workers=os.cpu_count()):
        for c in range(len(CLASSES)):
            counts[c] += int((y == c).sum())
    share = counts / max(1, counts.sum())
    print(name, {c: round(float(s), 3) for c, s in zip(CLASSES, share)}, flush=True)
    return share


@torch.no_grad()
def evaluate(depth, head, items):
    head.eval()
    inter, union = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
    for rgb, y in DataLoader(Pictures(items, False), batch_size=BATCH, num_workers=os.cpu_count()):
        rgb, y = rgb.to(device), y.to(device)
        p = logits_of(head, features(depth, rgb)).argmax(1)
        known = y != IGNORE
        for c in range(len(CLASSES)):
            a, b = (p == c) & known, (y == c) & known
            inter[c] += int((a & b).sum())
            union[c] += int((a | b).sum())
    head.train()
    return {c: round(float(i / max(u, 1)), 4) for c, i, u in zip(CLASSES, inter, union)}


@torch.no_grad()
def save_samples(depth, head, items, path):
    head.eval()
    data = Pictures(items, False)
    rows = []
    for i in range(0, len(items), max(1, len(items) // SAMPLES)):
        rgb, _ = data[i]
        p = logits_of(head, features(depth, rgb[None].to(device))).softmax(1)[0].cpu().numpy()
        picture = rgb.permute(1, 2, 0).numpy().astype(np.float32)
        tint = (COLOURS[:, None, None] * p[:, :, :, None]).sum(0)
        alpha = 1.0 - p[OTHER]
        over = picture * (1 - 0.6 * alpha[:, :, None]) + tint * 0.6
        rows.append(np.concatenate([picture, over], 1))
        if len(rows) == SAMPLES:
            break
    head.train()
    Image.fromarray(np.concatenate(rows, 0).clip(0, 255).astype(np.uint8)).save(path, quality=85)


def main():
    os.makedirs(OUT, exist_ok=True)
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv || true; nproc; df -h /tmp | tail -1")
    get_panoptic()
    coco = find("coco2017")
    ade = find("ADEChallengeData2016")
    coco_train, coco_val = coco_items("train2017", f"{coco}/train2017"), coco_items("val2017", f"{coco}/val2017")
    ade_train, ade_val = ade_items("training", ade), ade_items("validation", ade)
    print(f"COCO {len(coco_train)} + {len(coco_val)}, ADE20K {len(ade_train)} + {len(ade_val)}", flush=True)
    assert coco_train and coco_val and ade_train and ade_val

    check_labels("COCO", coco_train)
    other, person, wall, sky = check_labels("ADE20K", ade_train)
    assert wall > sky > person > 0, "ADE20K's label numbers aren't what this expects"

    depth = AutoModelForDepthEstimation.from_pretrained(MODEL, revision=REVISION).eval().to(device).requires_grad_(False)
    head = Head().to(device)
    params = sum(p.numel() for p in head.parameters())
    print(f"head: {params} weights, {params * 4 / 1e6:.2f} MB", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    data = torch.utils.data.ConcatDataset([Pictures(coco_train, True), Pictures(ade_train, True)])
    loader = DataLoader(
        data, batch_size=BATCH, sampler=Half(len(coco_train), len(data)), num_workers=os.cpu_count(),
        pin_memory=device == "cuda", persistent_workers=True, prefetch_factor=4,
    )

    began = time.time()
    total = MINUTES * 60 - (began - START)
    print(f"setup took {(began - START) / 60:.1f} min, training for {total / 60:.0f} min", flush=True)
    history, best, step, running, shown = [], -1.0, 0, 0.0, time.time()
    next_eval = began + EVAL_MINUTES * 60

    def checkpoint():
        nonlocal best
        scores = {"COCO": evaluate(depth, head, coco_val), "ADE20K": evaluate(depth, head, ade_val)}
        score = float(np.mean([s[c] for s in scores.values() for c in ("person", "wall", "sky")]))
        minutes = (time.time() - began) / 60
        history.append({"step": step, "minutes": round(minutes, 1), "score": round(score, 4), **scores})
        print(f"step {step} ({minutes:.0f} min): score {score:.4f} {scores}", flush=True)
        with open(f"{OUT}/metrics.json", "w") as f:
            json.dump(history, f, indent=1)
        save_samples(depth, head, coco_val, f"{OUT}/samples_{step:07d}_coco.jpg")
        save_samples(depth, head, ade_val, f"{OUT}/samples_{step:07d}_ade.jpg")
        if score > best:
            best = score
            save_file({k: v.cpu().contiguous() for k, v in head.state_dict().items()}, f"{OUT}/head.safetensors", {"classes": ",".join(CLASSES), "model": MODEL, "revision": REVISION})
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
