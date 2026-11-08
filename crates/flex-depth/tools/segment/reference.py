"""Reference output for flex-depth's `masks_match_pytorch` test: runs the person, wall and sky
network from run.py on an image and saves the input and the masks.

    python reference.py <image> <head.safetensors> <out.safetensors>

Needs torch, transformers, safetensors, pillow and numpy.
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file, save_file
from transformers import AutoModelForDepthEstimation

MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
GW, GH = 37, 21
W, H = GW * 14, GH * 14


class Head(nn.Module):
    """As in run.py."""

    def __init__(self):
        super().__init__()
        self.token9 = nn.Conv2d(384, 128, 1)
        self.token12 = nn.Conv2d(384, 128, 1)
        self.context = nn.Conv2d(128, 128, 3, padding=1)
        self.mid = nn.Conv2d(128, 64, 3, padding=1)
        self.fused = nn.Conv2d(64, 64, 1)
        self.fine1 = nn.Conv2d(64, 48, 3, padding=1)
        self.fine2 = nn.Conv2d(48, 48, 3, padding=1)
        self.out = nn.Conv2d(48, 4, 1)

    def forward(self, t9, t12, fused):
        x = F.relu(self.context(F.relu(self.token9(t9) + self.token12(t12))))
        x = F.interpolate(x, size=(GH * 4, GW * 4), mode="bilinear", align_corners=True)
        x = F.relu(self.mid(x))
        x = F.interpolate(x, size=fused.shape[2:], mode="bilinear", align_corners=True)
        x = F.relu(self.fine1(F.relu(x + self.fused(fused))))
        return self.out(F.relu(self.fine2(x)))


def grid(tokens):
    return tokens[:, 1:].transpose(1, 2).reshape(tokens.shape[0], 384, GH, GW)


@torch.no_grad()
def main():
    image, weights, out = sys.argv[1:4]
    depth = AutoModelForDepthEstimation.from_pretrained(MODEL, revision=REVISION).eval()
    head = Head().eval()
    head.load_state_dict(load_file(weights))
    rgb = np.asarray(Image.open(image).convert("RGB").resize((W, H), Image.BICUBIC))
    mean, std = torch.tensor([0.485, 0.456, 0.406])[:, None, None], torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    x = ((torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255 - mean) / std)[None]
    maps = depth.backbone(pixel_values=x).feature_maps
    fused = depth.neck(list(maps), patch_height=GH, patch_width=GW)[-1]
    logits = F.interpolate(head(grid(maps[2]), grid(maps[3]), fused), size=(H, W), mode="bilinear", align_corners=True)
    save_file({"rgb": torch.from_numpy(rgb.copy()), "masks": logits.softmax(1)[0]}, out)


main()
