"""Method 2: Resume CycleGAN training (FINAL VERSION 8).

Fixes from audit round 8:
  1. Smoke test uses torch.autograd.grad() — no .grad accumulation
     - Also snapshots and restores pre-existing .grad fields (defensive)
  2. Smoke test snapshots and restores dataset.rng AND dl_generator states
     - Critical for first resume from epoch_05 (no saved RNG in that checkpoint)
  3. Explicit CUDA requirement at startup (no silent CPU fallback)
"""

import csv
import gc
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import lpips
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

# ruff: noqa: N803, N806, SIM115, F841

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"

ORIGINAL_PILOT_DIR = Path(r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_2_pilot")
ORIGINAL_CKPT = ORIGINAL_PILOT_DIR / "checkpoints" / "epoch_05.pth"

OUTPUT_DIR = Path(r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_2_resume")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
(OUTPUT_DIR / "comparisons").mkdir(exist_ok=True)

# FIX 3: hard require CUDA
if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA is required for this experiment (target: RTX 4050). Aborting.")

DEVICE = "cuda"
print(f"GPU: {torch.cuda.get_device_name(0)}")

IMG_H, IMG_W = 288, 512
TRAIN_COUNT = 1500
HOLDOUT_COUNT = 200
BATCH_SIZE = 2
TOTAL_EPOCHS = 20
LR_INITIAL = 0.0002
LR_FINAL = 0.00002
LR_DECAY_START = 10
SEED = 42
BUFFER_SIZE = 50

CLIP_PROMPTS = [
    "a photo of a daytime residential street",
    "a photo of a nighttime residential street",
]
ADE_ROAD_CLASSES = [6, 11]

FULL_CONFIG = {
    "img_h": IMG_H,
    "img_w": IMG_W,
    "train_count": TRAIN_COUNT,
    "holdout_count": HOLDOUT_COUNT,
    "batch_size": BATCH_SIZE,
    "total_epochs": TOTAL_EPOCHS,
    "lr_initial": LR_INITIAL,
    "lr_final": LR_FINAL,
    "lr_decay_start": LR_DECAY_START,
    "seed": SEED,
    "buffer_size": BUFFER_SIZE,
    "arch": "small_cyclegan_v1",
}


def build_image_index():
    return {p.name: p for p in IMAGES_DIR.rglob("*.jpg")}


IMAGE_INDEX = build_image_index()


def reconstruct_original_split(day_imgs, night_imgs):
    rng = random.Random(SEED)
    day_s = rng.sample(day_imgs, 500 + 50)
    night_s = rng.sample(night_imgs, 500 + 50)
    return (day_s[:500], night_s[:500], day_s[500:], night_s[500:])


def build_manifest():
    with TRAIN_LABEL.open("r", encoding="utf-8") as f:
        data = json.load(f)
    day_imgs, night_imgs = [], []
    for rec in data:
        attrs = rec.get("attributes", {})
        if attrs.get("scene") != "residential":
            continue
        tod = attrs.get("timeofday")
        path = IMAGE_INDEX.get(rec["name"])
        if not path:
            continue
        if tod == "daytime":
            day_imgs.append(path)
        elif tod == "night":
            night_imgs.append(path)
    day_imgs.sort(key=lambda p: p.name)
    night_imgs.sort(key=lambda p: p.name)

    orig_day_train, orig_night_train, orig_day_holdout, orig_night_holdout = (
        reconstruct_original_split(day_imgs, night_imgs)
    )

    used_day = {p.name for p in orig_day_train + orig_day_holdout}
    used_night = {p.name for p in orig_night_train + orig_night_holdout}
    avail_day = [p for p in day_imgs if p.name not in used_day]
    avail_night = [p for p in night_imgs if p.name not in used_night]

    rng = random.Random(SEED)
    need_day_train = TRAIN_COUNT - 500
    need_day_holdout = HOLDOUT_COUNT - 50
    new_day_s = rng.sample(avail_day, need_day_train + need_day_holdout)
    new_day_train = new_day_s[:need_day_train]
    new_day_holdout = new_day_s[need_day_train:]

    need_night_train = TRAIN_COUNT - 500
    need_night_holdout = HOLDOUT_COUNT - 50
    new_night_s = rng.sample(avail_night, need_night_train + need_night_holdout)
    new_night_train = new_night_s[:need_night_train]
    new_night_holdout = new_night_s[need_night_train:]

    day_train = orig_day_train + new_day_train
    night_train = orig_night_train + new_night_train
    day_ho = orig_day_holdout + new_day_holdout
    night_ho = orig_night_holdout + new_night_holdout

    assert len({p.name for p in day_train} & {p.name for p in day_ho}) == 0
    assert len({p.name for p in night_train} & {p.name for p in night_ho}) == 0

    manifest = {
        "day_train": [p.name for p in day_train],
        "night_train": [p.name for p in night_train],
        "day_holdout": [p.name for p in day_ho],
        "night_holdout": [p.name for p in night_ho],
    }
    return day_train, night_train, day_ho, night_ho, manifest


def write_manifest(manifest):
    (OUTPUT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))


class UnpairedDataset(Dataset):
    def __init__(self, day_paths, night_paths, transform):
        self.day_paths = day_paths
        self.night_paths = night_paths
        self.transform = transform
        self.rng = random.Random(SEED + 1)

    def __len__(self):
        return max(len(self.day_paths), len(self.night_paths))

    def __getitem__(self, idx):
        day_path = self.day_paths[idx % len(self.day_paths)]
        night_path = self.rng.choice(self.night_paths)
        with Image.open(day_path) as img:
            day_img = self.transform(img.convert("RGB"))
        with Image.open(night_path) as img:
            night_img = self.transform(img.convert("RGB"))
        if self.rng.random() > 0.5:
            day_img = torch.flip(day_img, dims=[2])
        if self.rng.random() > 0.5:
            night_img = torch.flip(night_img, dims=[2])
        return day_img, night_img


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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, 4, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


class ImageBuffer:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def query(self, images):
        out = []
        for img in images:
            if len(self.buffer) < self.buffer.maxlen or random.random() > 0.5:
                self.buffer.append(img.detach())
                out.append(img)
            else:
                idx = random.randint(0, len(self.buffer) - 1)
                old = self.buffer[idx]
                self.buffer[idx] = img.detach()
                out.append(old)
        return torch.stack(out)

    def state(self):
        return list(self.buffer)

    def load(self, items, device):
        self.buffer = deque([t.to(device) for t in items], maxlen=self.buffer.maxlen)


def atomic_save(ckpt, path):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp_path)
    tmp_path.replace(path)


def save_checkpoint(
    path,
    epoch,
    G_AB,
    G_BA,
    D_A,
    D_B,
    opt_G,
    opt_D,
    scaler_G,
    scaler_D,
    manifest,
    dataset,
    dl_generator,
    buffer_A,
    buffer_B,
):
    ckpt = {
        "epoch": epoch,
        "G_AB": G_AB.state_dict(),
        "G_BA": G_BA.state_dict(),
        "D_A": D_A.state_dict(),
        "D_B": D_B.state_dict(),
        "opt_G": opt_G.state_dict(),
        "opt_D": opt_D.state_dict(),
        "scaler_G": scaler_G.state_dict(),
        "scaler_D": scaler_D.state_dict(),
        "config": FULL_CONFIG,
        "manifest": manifest,
        "rng_python": random.getstate(),
        "rng_numpy": np.random.get_state(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_dataset": dataset.rng.getstate(),
        "rng_dataloader": dl_generator.get_state(),
        "buffer_fake_A": buffer_A.state(),
        "buffer_fake_B": buffer_B.state(),
    }
    atomic_save(ckpt, path)


def evaluate_and_save(G_AB, day_ho, epoch_num):
    print(f"  Evaluating epoch {epoch_num}...")
    was_training = G_AB.training
    G_AB.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    metrics_file = open(OUTPUT_DIR / "metrics.csv", "a", newline="")
    writer = csv.writer(metrics_file)
    if metrics_file.tell() == 0:
        writer.writerow(
            [
                "epoch",
                "image",
                "lum_orig",
                "lum_fake",
                "clip_night_orig",
                "clip_night_fake",
                "road_pixel_fraction_orig",
                "road_pixel_fraction_fake",
                "ssim",
                "lpips",
            ]
        )

    pairs = []
    with torch.no_grad():
        for i in range(5):
            path = day_ho[i]
            with Image.open(path) as img:
                orig_pil = img.convert("RGB").resize((IMG_W, IMG_H), Image.LANCZOS)
            real_A = transform(orig_pil).unsqueeze(0).to(DEVICE)
            fake_B = G_AB(real_A)
            fake_pil = to_pil_image((fake_B.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1))
            pairs.append((path.name, orig_pil, fake_pil))

    clip_mod = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_scores = []
    with torch.no_grad():
        for _name, orig_pil, fake_pil in pairs:
            inputs = clip_proc(
                text=CLIP_PROMPTS, images=[orig_pil, fake_pil], return_tensors="pt", padding=True
            ).to(DEVICE)
            probs = clip_mod(**inputs).logits_per_image.softmax(dim=1)
            clip_scores.append((float(probs[0, 1]), float(probs[1, 1])))
    del clip_mod, clip_proc
    torch.cuda.empty_cache()
    gc.collect()

    seg_proc = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    seg_mod = (
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        .to(DEVICE)
        .eval()
    )
    road_scores = []
    with torch.no_grad():
        for _name, orig_pil, fake_pil in pairs:
            res = []
            for img in [orig_pil, fake_pil]:
                inputs = seg_proc(images=img, return_tensors="pt").to(DEVICE)
                logits = seg_mod(**inputs).logits
                up = torch.nn.functional.interpolate(
                    logits, size=img.size[::-1], mode="bilinear", align_corners=False
                )
                pred = up.argmax(dim=1).squeeze(0)
                mask = torch.zeros_like(pred, dtype=torch.bool)
                for cid in ADE_ROAD_CLASSES:
                    mask |= pred == cid
                res.append(float(mask.sum().item() / pred.numel()))
            road_scores.append(tuple(res))
    del seg_proc, seg_mod
    torch.cuda.empty_cache()
    gc.collect()

    lpips_mod = lpips.LPIPS(net="alex").to(DEVICE).eval()
    lpips_scores = []
    with torch.no_grad():
        for _name, orig_pil, fake_pil in pairs:
            t_o = transforms.ToTensor()(orig_pil).unsqueeze(0).to(DEVICE) * 2 - 1
            t_f = transforms.ToTensor()(fake_pil).unsqueeze(0).to(DEVICE) * 2 - 1
            lpips_scores.append(float(lpips_mod(t_o, t_f).item()))
    del lpips_mod
    torch.cuda.empty_cache()
    gc.collect()

    for i, (name, orig_pil, fake_pil) in enumerate(pairs):
        arr_o = np.array(orig_pil).astype(float)
        arr_f = np.array(fake_pil).astype(float)
        lum_o = float(
            (0.299 * arr_o[:, :, 0] + 0.587 * arr_o[:, :, 1] + 0.114 * arr_o[:, :, 2]).mean()
            / 255.0
        )
        lum_f = float(
            (0.299 * arr_f[:, :, 0] + 0.587 * arr_f[:, :, 1] + 0.114 * arr_f[:, :, 2]).mean()
            / 255.0
        )
        s = float(ssim(arr_o, arr_f, channel_axis=2, data_range=255))
        c_o, c_f = clip_scores[i]
        r_o, r_f = road_scores[i]
        lp = lpips_scores[i]

        writer.writerow(
            [
                epoch_num,
                name,
                f"{lum_o:.3f}",
                f"{lum_f:.3f}",
                f"{c_o:.3f}",
                f"{c_f:.3f}",
                f"{r_o:.3f}",
                f"{r_f:.3f}",
                f"{s:.3f}",
                f"{lp:.3f}",
            ]
        )

        canvas = Image.new("RGB", (IMG_W * 2, IMG_H), (30, 30, 30))
        canvas.paste(orig_pil, (0, 0))
        canvas.paste(fake_pil, (IMG_W, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 10), "REAL DAY", fill=(255, 255, 255))
        draw.text((IMG_W + 10, 10), f"FAKE NIGHT (Epoch {epoch_num})", fill=(255, 255, 255))
        canvas.save(OUTPUT_DIR / "comparisons" / f"epoch{epoch_num:02d}_holdout_{i + 1}.jpg")

    metrics_file.close()
    if was_training:
        G_AB.train()


def find_latest_checkpoint():
    ckpts = sorted((OUTPUT_DIR / "checkpoints").glob("epoch_*.pth"))
    return ckpts[-1] if ckpts else ORIGINAL_CKPT


def validate_config(ckpt, is_original):
    if is_original:
        print("  Original epoch_05 checkpoint — skipping config validation (legacy)")
        return
    ckpt_config = ckpt.get("config", {})
    keys_to_check = [
        "img_h",
        "img_w",
        "batch_size",
        "lr_initial",
        "lr_final",
        "lr_decay_start",
        "total_epochs",
        "train_count",
        "holdout_count",
        "seed",
        "buffer_size",
        "arch",
    ]
    for key in keys_to_check:
        if key not in ckpt_config:
            sys.exit(f"ERROR: checkpoint config missing key '{key}'")
        if ckpt_config[key] != FULL_CONFIG[key]:
            sys.exit(
                f"ERROR: config mismatch on '{key}': checkpoint={ckpt_config[key]}, script={FULL_CONFIG[key]}"
            )
    print("  Config validated (all 12 keys match)")


def validate_manifest(ckpt, new_manifest, is_original):
    if is_original:
        print("  Original epoch_05 checkpoint — no prior manifest to validate (legacy)")
        return
    ckpt_manifest = ckpt.get("manifest")
    if ckpt_manifest != new_manifest:
        sys.exit("ERROR: dataset manifest changed since checkpoint. Aborting to protect integrity.")
    print("  Manifest validated — identical")


def smoke_test(dataloader, G_AB, G_BA, D_A, D_B, dataset, dl_generator):
    """FIX 1 & 2: NON-MUTATING smoke test.

    - Uses torch.autograd.grad() — does not accumulate into .grad
    - Snapshots and restores ALL RNG (Python/NumPy/Torch/CUDA/dataset/dataloader)
    - Snapshots and restores any pre-existing .grad fields (defensive)
    - Restores D requires_grad in finally
    """
    print("\n--- NON-MUTATING SMOKE TEST ---")
    rng_snapshot = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
        "dataset": dataset.rng.getstate(),
        "dataloader": dl_generator.get_state(),
    }
    existing_grads = {
        "G_AB": [p.grad.clone() if p.grad is not None else None for p in G_AB.parameters()],
        "G_BA": [p.grad.clone() if p.grad is not None else None for p in G_BA.parameters()],
        "D_A": [p.grad.clone() if p.grad is not None else None for p in D_A.parameters()],
        "D_B": [p.grad.clone() if p.grad is not None else None for p in D_B.parameters()],
    }
    prev_grads_A = [p.requires_grad for p in D_A.parameters()]
    prev_grads_B = [p.requires_grad for p in D_B.parameters()]

    for p in D_A.parameters():
        p.requires_grad_(False)
    for p in D_B.parameters():
        p.requires_grad_(False)

    try:
        real_A, real_B = next(iter(dataloader))
        real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)
        torch.cuda.reset_peak_memory_stats()

        with torch.amp.autocast("cuda"):
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            loss = nn.MSELoss()(D_B(fake_B), torch.ones_like(D_B(fake_B))) + nn.MSELoss()(
                D_A(fake_A), torch.ones_like(D_A(fake_A))
            )

        # FIX 1: autograd.grad returns gradients WITHOUT writing to .grad fields
        params = list(G_AB.parameters()) + list(G_BA.parameters())
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        n_total = len(grads)
        n_none = sum(1 for g in grads if g is None)
        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  ✅ Forward+Backward OK — Peak VRAM: {peak:.2f} GB")
        print(
            f"     Gradients: {n_total - n_none}/{n_total} non-None (autograd.grad — no .grad mutation)"
        )
    except torch.cuda.OutOfMemoryError:
        sys.exit("❌ OOM during smoke test")
    finally:
        # FIX 1: restore any pre-existing .grad fields (defensive)
        for model_name, model in [("G_AB", G_AB), ("G_BA", G_BA), ("D_A", D_A), ("D_B", D_B)]:
            for param, prev_grad in zip(
                model.parameters(), existing_grads[model_name], strict=True
            ):
                param.grad = prev_grad
        # Restore D requires_grad
        for p, was in zip(D_A.parameters(), prev_grads_A, strict=True):
            p.requires_grad_(was)
        for p, was in zip(D_B.parameters(), prev_grads_B, strict=True):
            p.requires_grad_(was)
        # FIX 2: restore ALL RNG
        random.setstate(rng_snapshot["python"])
        np.random.set_state(rng_snapshot["numpy"])
        torch.set_rng_state(rng_snapshot["torch"])
        torch.cuda.set_rng_state_all(rng_snapshot["cuda"])
        dataset.rng.setstate(rng_snapshot["dataset"])
        dl_generator.set_state(rng_snapshot["dataloader"])


def main():
    # Seed global RNGs before any model construction (for buffers, any stray randomness)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    resume_path = find_latest_checkpoint()
    is_original = resume_path == ORIGINAL_CKPT
    print(f"\nResuming from: {resume_path} (original={is_original})")

    if not resume_path.exists():
        sys.exit(f"ERROR: checkpoint missing at {resume_path}")

    day_tr, night_tr, day_ho, night_ho, new_manifest = build_manifest()
    print(f"Train — day: {len(day_tr)}, night: {len(night_tr)}")
    print(f"Holdout — day: {len(day_ho)}, night: {len(night_ho)}")

    print("\nLoading checkpoint...")
    ckpt = torch.load(resume_path, weights_only=False)
    validate_config(ckpt, is_original)
    validate_manifest(ckpt, new_manifest, is_original)
    write_manifest(new_manifest)

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = UnpairedDataset(day_tr, night_tr, transform)
    dl_generator = torch.Generator()
    dl_generator.manual_seed(SEED + 2)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=dl_generator,
    )

    G_AB = Generator().to(DEVICE)
    G_BA = Generator().to(DEVICE)
    D_A = Discriminator().to(DEVICE)
    D_B = Discriminator().to(DEVICE)

    opt_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=LR_INITIAL, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=LR_INITIAL, betas=(0.5, 0.999)
    )
    scaler_G = torch.amp.GradScaler("cuda")
    scaler_D = torch.amp.GradScaler("cuda")

    buffer_fake_A = ImageBuffer(BUFFER_SIZE)
    buffer_fake_B = ImageBuffer(BUFFER_SIZE)

    G_AB.load_state_dict(ckpt["G_AB"])
    G_BA.load_state_dict(ckpt["G_BA"])
    D_A.load_state_dict(ckpt["D_A"])
    D_B.load_state_dict(ckpt["D_B"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    scaler_G.load_state_dict(ckpt["scaler_G"])
    scaler_D.load_state_dict(ckpt["scaler_D"])
    start_epoch = ckpt["epoch"]

    if "buffer_fake_A" in ckpt:
        buffer_fake_A.load(ckpt["buffer_fake_A"], DEVICE)
        buffer_fake_B.load(ckpt["buffer_fake_B"], DEVICE)
        print("  Restored image buffers to device")

    print(f"  Loaded epoch {start_epoch}")

    smoke_test(dataloader, G_AB, G_BA, D_A, D_B, dataset, dl_generator)

    if "rng_python" in ckpt:
        random.setstate(ckpt["rng_python"])
        np.random.set_state(ckpt["rng_numpy"])
        torch.set_rng_state(ckpt["rng_torch"])
        torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        dataset.rng.setstate(ckpt["rng_dataset"])
        if "rng_dataloader" in ckpt:
            dl_generator.set_state(ckpt["rng_dataloader"])
        print("  RNG state restored from checkpoint")
    else:
        print(
            "  NOTE: original checkpoint has no RNG state — this is a continuation, not exact resume"
        )

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    log_file = open(OUTPUT_DIR / "training_log.csv", "a", newline="")
    log_writer = csv.writer(log_file)
    if log_file.tell() == 0:
        log_writer.writerow(["epoch", "lr", "loss_G", "loss_D"])

    print(f"\nResuming from epoch {start_epoch + 1} to {TOTAL_EPOCHS}...")
    start_time = time.time()

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        actual_epoch = epoch + 1
        if actual_epoch >= LR_DECAY_START:
            progress = (actual_epoch - LR_DECAY_START) / (TOTAL_EPOCHS - LR_DECAY_START)
            progress = min(max(progress, 0.0), 1.0)
            lr = LR_INITIAL - progress * (LR_INITIAL - LR_FINAL)
            for pg in opt_G.param_groups:
                pg["lr"] = lr
            for pg in opt_D.param_groups:
                pg["lr"] = lr
        else:
            lr = LR_INITIAL

        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        for i, (real_A, real_B) in enumerate(dataloader):
            real_A, real_B = real_A.to(DEVICE), real_B.to(DEVICE)

            # Generator
            opt_G.zero_grad()
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)
            with torch.amp.autocast("cuda"):
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                loss_GAN_AB = criterion_GAN(D_B(fake_B), torch.ones_like(D_B(fake_B)))
                loss_GAN_BA = criterion_GAN(D_A(fake_A), torch.ones_like(D_A(fake_A)))
                loss_cycle_A = criterion_cycle(G_BA(fake_B), real_A) * 10.0
                loss_cycle_B = criterion_cycle(G_AB(fake_A), real_B) * 10.0
                loss_idt_A = criterion_identity(G_BA(real_A), real_A) * 5.0
                loss_idt_B = criterion_identity(G_AB(real_B), real_B) * 5.0
                loss_G = (
                    loss_GAN_AB
                    + loss_GAN_BA
                    + loss_cycle_A
                    + loss_cycle_B
                    + loss_idt_A
                    + loss_idt_B
                )
            scaler_G.scale(loss_G).backward()
            scaler_G.step(opt_G)
            scaler_G.update()
            epoch_loss_G += loss_G.item()

            # Discriminator
            opt_D.zero_grad()
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    fake_B = G_AB(real_A)
                    fake_A = G_BA(real_B)
                fake_A_buffered = buffer_fake_A.query(fake_A)
                fake_B_buffered = buffer_fake_B.query(fake_B)
                pred_real_A = D_A(real_A)
                pred_fake_A = D_A(fake_A_buffered)
                loss_D_A = (
                    criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
                    + criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
                ) * 0.5
                pred_real_B = D_B(real_B)
                pred_fake_B = D_B(fake_B_buffered)
                loss_D_B = (
                    criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
                    + criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
                ) * 0.5
                loss_D = loss_D_A + loss_D_B
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
            epoch_loss_D += loss_D.item()

            if i % 100 == 0:
                print(
                    f"[Epoch {actual_epoch}/{TOTAL_EPOCHS}] [Batch {i}/{len(dataloader)}] lr: {lr:.6f} G: {loss_G.item():.4f} D: {loss_D.item():.4f}"
                )

        log_writer.writerow(
            [
                actual_epoch,
                f"{lr:.6f}",
                epoch_loss_G / len(dataloader),
                epoch_loss_D / len(dataloader),
            ]
        )
        log_file.flush()

        if actual_epoch % 2 == 0 or actual_epoch == TOTAL_EPOCHS:
            evaluate_and_save(G_AB, day_ho, actual_epoch)

        save_checkpoint(
            OUTPUT_DIR / "checkpoints" / f"epoch_{actual_epoch:02d}.pth",
            actual_epoch,
            G_AB,
            G_BA,
            D_A,
            D_B,
            opt_G,
            opt_D,
            scaler_G,
            scaler_D,
            new_manifest,
            dataset,
            dl_generator,
            buffer_fake_A,
            buffer_fake_B,
        )
        print(f"  Saved epoch_{actual_epoch:02d}.pth (atomic, post-eval RNG)")

    log_file.close()
    print(f"\nTraining complete in {(time.time() - start_time) / 60:.1f} min.")


if __name__ == "__main__":
    main()
