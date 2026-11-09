"""Method 2 Pilot: Lightweight CycleGAN for Day -> Night.

Fixes from audit round 2:
  - Random pairing: random.choice(night_paths) per __getitem__
  - Freeze discriminators during generator update
  - Combined checkpoint per epoch (models + optimizers + scalers + config)
  - Evaluation uses exact functions from calibrate_controlnet_metrics.py
  - One-batch smoke test gate before full training

Feasibility pilot only. 500 images/domain, 5 epochs.
"""

# ruff: noqa: N803, N806, SIM115, F841
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Reuse exact calibration functions to prevent evaluation drift
from transformers import (
    CLIPModel,
    CLIPProcessor,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

DATA_ROOT = Path(r"C:\Users\Janak Mallik\ml-data\bdd100k")
TRAIN_LABEL = DATA_ROOT / "labels" / "bdd100k_labels_images_train.json"
IMAGES_DIR = DATA_ROOT / "images" / "100k" / "train"
OUTPUT_DIR = Path(r"C:\Users\Janak Mallik\UrbanGen-AI\outputs\method_2_pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "checkpoints").mkdir(exist_ok=True)
(OUTPUT_DIR / "comparisons").mkdir(exist_ok=True)

# Config (identical to previous, for audit comparison)
IMG_H, IMG_W = 288, 512
TRAIN_COUNT = 500
HOLDOUT_COUNT = 50
BATCH_SIZE = 2
EPOCHS = 5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Seeds for reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# These MUST match calibrate_controlnet_metrics.py exactly
ADE_GROUPS = {
    "road": [6, 11],
    "vehicle": [20, 80, 83, 102],
    "sky": [2],
    "building": [1, 25, 48],
    "vegetation": [4, 9, 17, 72],
    "person": [12],
    "other": None,
}
CLIP_PROMPTS = [
    "a photo of a daytime residential street",
    "a photo of a nighttime residential street",
    "a photo of a daytime urban scene",
    "a photo of a nighttime urban scene",
]


def build_image_index():
    return {p.name: p for p in IMAGES_DIR.rglob("*.jpg")}


IMAGE_INDEX = build_image_index()


def get_images():
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
    rng = random.Random(SEED)
    day_s = rng.sample(day_imgs, TRAIN_COUNT + HOLDOUT_COUNT)
    night_s = rng.sample(night_imgs, TRAIN_COUNT + HOLDOUT_COUNT)
    return (
        day_s[:TRAIN_COUNT],
        night_s[:TRAIN_COUNT],
        day_s[TRAIN_COUNT:],
        night_s[TRAIN_COUNT:],
    )


class UnpairedDataset(Dataset):
    def __init__(self, day_paths, night_paths, transform):
        self.day_paths = day_paths
        self.night_paths = night_paths
        self.transform = transform
        # Dedicated RNG so we don't touch global state during eval
        self._rng = random.Random(SEED + 1)

    def __len__(self):
        return max(len(self.day_paths), len(self.night_paths))

    def __getitem__(self, idx):
        day_path = self.day_paths[idx % len(self.day_paths)]
        # FIX 1: random pairing, not index pairing
        night_path = self._rng.choice(self.night_paths)
        with Image.open(day_path) as img:
            day_img = self.transform(img.convert("RGB"))
        with Image.open(night_path) as img:
            night_img = self.transform(img.convert("RGB"))
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


def compute_luminance(image):
    arr = np.array(image.convert("RGB")).astype(float)
    lum = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return float(lum.mean() / 255.0)


def compute_clip_scores(image, clip_processor, clip_model):
    inputs = clip_processor(text=CLIP_PROMPTS, images=image, return_tensors="pt", padding=True).to(
        DEVICE
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1).squeeze(0)
    return float(probs[1])  # nighttime_residential


def compute_road_proportion(image, seg_processor, seg_model):
    inputs = seg_processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = seg_model(**inputs)
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    pred = upsampled.argmax(dim=1).squeeze(0)
    total = pred.numel()
    mask = torch.zeros_like(pred, dtype=torch.bool)
    for cid in ADE_GROUPS["road"]:
        mask |= pred == cid
    return float(mask.sum().item() / total)


def smoke_test(dataloader, G_AB, G_BA, D_A, D_B, opt_G, opt_D, scaler_G, scaler_D):
    """One batch forward+backward to verify VRAM fits."""
    print("\n--- ONE-BATCH SMOKE TEST ---")
    real_A, real_B = next(iter(dataloader))
    real_A = real_A.to(DEVICE)
    real_B = real_B.to(DEVICE)
    torch.cuda.reset_peak_memory_stats()

    try:
        opt_G.zero_grad()
        for p in D_A.parameters():
            p.requires_grad_(False)
        for p in D_B.parameters():
            p.requires_grad_(False)
        with torch.amp.autocast("cuda"):
            fake_B = G_AB(real_A)
            fake_A = G_BA(real_B)
            loss_G = nn.MSELoss()(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        scaler_G.scale(loss_G).backward()
        scaler_G.step(opt_G)
        scaler_G.update()

        opt_D.zero_grad()
        for p in D_A.parameters():
            p.requires_grad_(True)
        for p in D_B.parameters():
            p.requires_grad_(True)
        with torch.amp.autocast("cuda"):
            loss_D = nn.MSELoss()(D_A(real_A), torch.ones_like(D_A(real_A)))
        scaler_D.scale(loss_D).backward()
        scaler_D.step(opt_D)
        scaler_D.update()

        peak = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  ✅ Smoke test OK — peak VRAM: {peak:.2f} GB")
        if peak > 5.5:
            print(f"  ⚠️ WARNING: peak {peak:.2f} GB is close to 6 GB limit")
        return True
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ❌ OOM: {e}")
        return False


def main():
    print(f"Device: {DEVICE}")
    print("Loading images...")
    day_tr, night_tr, day_ho, night_ho = get_images()
    print(f"Train — day: {len(day_tr)}, night: {len(night_tr)}")
    print(f"Holdout — day: {len(day_ho)}, night: {len(night_ho)}")

    config = {
        "train_count": TRAIN_COUNT,
        "holdout_count": HOLDOUT_COUNT,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "img_h": IMG_H,
        "img_w": IMG_W,
        "seed": SEED,
    }
    (OUTPUT_DIR / "config.json").write_text(json.dumps(config, indent=2))

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_H, IMG_W)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = UnpairedDataset(day_tr, night_tr, transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    G_AB = Generator().to(DEVICE)
    G_BA = Generator().to(DEVICE)
    D_A = Discriminator().to(DEVICE)
    D_B = Discriminator().to(DEVICE)

    opt_G = optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()), lr=0.0002, betas=(0.5, 0.999)
    )
    opt_D = optim.Adam(
        list(D_A.parameters()) + list(D_B.parameters()), lr=0.0002, betas=(0.5, 0.999)
    )
    scaler_G = torch.amp.GradScaler("cuda")
    scaler_D = torch.amp.GradScaler("cuda")

    # Smoke test gate
    if not smoke_test(dataloader, G_AB, G_BA, D_A, D_B, opt_G, opt_D, scaler_G, scaler_D):
        print("Aborting — smoke test failed.")
        sys.exit(1)

    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    log_file = open(OUTPUT_DIR / "training_log.csv", "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "loss_G", "loss_D"])

    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0

        for i, (real_A, real_B) in enumerate(dataloader):
            real_A = real_A.to(DEVICE)
            real_B = real_B.to(DEVICE)

            # --- Generators (discriminators frozen) ---
            opt_G.zero_grad()
            for p in D_A.parameters():
                p.requires_grad_(False)
            for p in D_B.parameters():
                p.requires_grad_(False)
            with torch.amp.autocast("cuda"):
                fake_B = G_AB(real_A)
                fake_A = G_BA(real_B)
                pred_fake_B = D_B(fake_B)
                pred_fake_A = D_A(fake_A)
                loss_GAN_AB = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))
                loss_GAN_BA = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))
                rec_A = G_BA(fake_B)
                rec_B = G_AB(fake_A)
                loss_cycle_A = criterion_cycle(rec_A, real_A) * 10.0
                loss_cycle_B = criterion_cycle(rec_B, real_B) * 10.0
                idt_A = G_BA(real_A)
                idt_B = G_AB(real_B)
                loss_idt_A = criterion_identity(idt_A, real_A) * 5.0
                loss_idt_B = criterion_identity(idt_B, real_B) * 5.0
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

            # --- Discriminators (frozen during G, now unfrozen) ---
            opt_D.zero_grad()
            for p in D_A.parameters():
                p.requires_grad_(True)
            for p in D_B.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast("cuda"):
                with torch.no_grad():
                    fake_B = G_AB(real_A)
                    fake_A = G_BA(real_B)
                pred_real_A = D_A(real_A)
                pred_fake_A = D_A(fake_A)
                loss_D_A = (
                    criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
                    + criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
                ) * 0.5
                pred_real_B = D_B(real_B)
                pred_fake_B = D_B(fake_B)
                loss_D_B = (
                    criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
                    + criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
                ) * 0.5
                loss_D = loss_D_A + loss_D_B
            scaler_D.scale(loss_D).backward()
            scaler_D.step(opt_D)
            scaler_D.update()
            epoch_loss_D += loss_D.item()

            if i % 20 == 0:
                print(
                    f"[Epoch {epoch + 1}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
                    f"G: {loss_G.item():.4f} D: {loss_D.item():.4f}"
                )

        log_writer.writerow(
            [epoch + 1, epoch_loss_G / len(dataloader), epoch_loss_D / len(dataloader)]
        )

        # FIX 3: combined checkpoint per epoch
        ckpt = {
            "epoch": epoch + 1,
            "G_AB": G_AB.state_dict(),
            "G_BA": G_BA.state_dict(),
            "D_A": D_A.state_dict(),
            "D_B": D_B.state_dict(),
            "opt_G": opt_G.state_dict(),
            "opt_D": opt_D.state_dict(),
            "scaler_G": scaler_G.state_dict(),
            "scaler_D": scaler_D.state_dict(),
            "config": config,
        }
        torch.save(ckpt, OUTPUT_DIR / "checkpoints" / f"epoch_{epoch + 1:02d}.pth")
        print(f"Saved checkpoint epoch_{epoch + 1:02d}.pth")

    log_file.close()
    print(f"\nTraining complete in {(time.time() - start_time) / 60:.1f} min.")

    # --- Held-out evaluation using exact calibration functions ---
    print("\nEvaluating on 5 held-out daytime images...")
    seg_processor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512"
    )
    seg_model = (
        SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512"
        )
        .to(DEVICE)
        .eval()
    )
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    metrics_file = open(OUTPUT_DIR / "metrics.csv", "w", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow(
        [
            "image",
            "lum_orig",
            "lum_fake",
            "clip_night_orig",
            "clip_night_fake",
            "road_orig",
            "road_fake",
        ]
    )

    G_AB.eval()
    with torch.no_grad():
        for i in range(5):
            real_A_path = day_ho[i]
            with Image.open(real_A_path) as img:
                orig_pil = img.convert("RGB").resize((IMG_W, IMG_H), Image.LANCZOS)
            real_A = transform(orig_pil).unsqueeze(0).to(DEVICE)
            fake_B = G_AB(real_A)
            fake_pil = to_pil_image((fake_B.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1))

            lum_o = compute_luminance(orig_pil)
            lum_f = compute_luminance(fake_pil)
            clip_o = compute_clip_scores(orig_pil, clip_processor, clip_model)
            clip_f = compute_clip_scores(fake_pil, clip_processor, clip_model)
            road_o = compute_road_proportion(orig_pil, seg_processor, seg_model)
            road_f = compute_road_proportion(fake_pil, seg_processor, seg_model)

            writer.writerow(
                [
                    real_A_path.name,
                    f"{lum_o:.3f}",
                    f"{lum_f:.3f}",
                    f"{clip_o:.3f}",
                    f"{clip_f:.3f}",
                    f"{road_o:.3f}",
                    f"{road_f:.3f}",
                ]
            )

            canvas = Image.new("RGB", (IMG_W * 2, IMG_H), (30, 30, 30))
            canvas.paste(orig_pil, (0, 0))
            canvas.paste(fake_pil, (IMG_W, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 10), "REAL DAY (held-out)", fill=(255, 255, 255))
            draw.text((IMG_W + 10, 10), "FAKE NIGHT (CycleGAN)", fill=(255, 255, 255))
            canvas.save(OUTPUT_DIR / "comparisons" / f"holdout_{i + 1}.jpg")
            print(f"  Saved holdout_{i + 1}.jpg")

    metrics_file.close()
    print(f"\nDone. Check {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
