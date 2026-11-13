"""Shared module for shortcut control (D-bal) generation.

Exports the CycleGAN Generator class and the three filter functions,
matching Method 2's pipeline exactly.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["URBANINTEL_DATA_ROOT"]) / "bdd100k"
CKPT_PATH = _REPO_ROOT / "outputs" / "method_2_resume" / "checkpoints" / "epoch_20.pth"

# --- Frozen Method 2 filter thresholds ---
IMG_H, IMG_W = 288, 512
SEED = 42
CLIP_NIGHT_MIN = 0.50
LUMINANCE_MAX = 0.20
PAVED_SURFACE_TOLERANCE = 0.10
ADE_PAVED_SURFACE_CLASSES = [6, 11]


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 1, 1),
            nn.InstanceNorm2d(c),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(*[ResBlock(128) for _ in range(4)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 7, 1, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.dec(self.res(self.enc(x)))


def load_generator(device: str = "cuda") -> Generator:
    """Load the Method 2 CycleGAN generator from the frozen checkpoint."""
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "G_AB" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'G_AB' key")
    gen = Generator().to(device)
    gen.load_state_dict(ckpt["G_AB"], strict=True)
    gen.eval()
    return gen


def checkpoint_sha256() -> str:
    return hashlib.sha256(CKPT_PATH.read_bytes()).hexdigest()


def compute_clip_night(clip_model, clip_proc, image: Image.Image, device: str) -> float:
    # Frozen Method 2 prompts — do not modify.
    # These are residential-specific. For D-bal city/highway images, the score
    # measures "resemblance to nighttime residential" which still preserves
    # the nighttime-vs-daytime comparison pattern.
    prompts = [
        "a photo of a daytime residential street",
        "a photo of a nighttime residential street",
    ]
    inputs = clip_proc(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.inference_mode():
        out = clip_model(**inputs)
    return float(out.logits_per_image.softmax(dim=1).squeeze(0)[1])


def compute_luminance(image: Image.Image) -> float:
    arr = np.array(image).astype(float)
    return float(
        (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]).mean() / 255.0
    )


def compute_paved_frac(seg_model, seg_proc, image: Image.Image, device: str) -> float:
    inp = seg_proc(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = seg_model(**inp)
    up = torch.nn.functional.interpolate(
        out.logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = up.argmax(dim=1).squeeze(0)
    mask = torch.zeros_like(pred, dtype=torch.bool)
    for cid in ADE_PAVED_SURFACE_CLASSES:
        mask |= pred == cid
    return float(mask.sum().item() / mask.numel())
