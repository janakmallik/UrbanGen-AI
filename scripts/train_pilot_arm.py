"""Train a ResNet-50 for the G1c pilot. One arm per invocation.

Four arms (all frozen):
    A — real only
    B — real + 3,000 cVAE-generated residential
    C — real + conventional augmentation
    D — real + class-weighted loss

All arms:
    - ImageNet-pretrained ResNet-50
    - Same real BDD100K train split (68,969 images from cache)
    - Same seed (42), same optimizer (AdamW 1e-4), same schedule
    - Same epochs (5), batch 8, mixed precision
    - Best-epoch selection by val macro-F1
    - Save checkpoint to checkpoints/pilot_arm_{A,B,C,D}.pt
    - Log to MLflow (urbangen-g1c)

Evaluation is done by scripts/evaluate_pilot.py after all arms train.

Run:
    uv run python scripts/train_pilot_arm.py --arm A
    uv run python scripts/train_pilot_arm.py --arm B
    uv run python scripts/train_pilot_arm.py --arm C
    uv run python scripts/train_pilot_arm.py --arm D
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}",
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402

# --- Frozen pilot configuration ---
SEED = 42
EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
N_SYNTHETIC_RESIDENTIAL = 3000

IMAGE_SIZE = 128
CONTENT_TOP = 28
CONTENT_BOTTOM = 100
CLASS_NAMES = ["city street", "highway", "residential"]
NUM_CLASSES = 3

_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_CVAE_CKPT = _REPO_ROOT / "checkpoints" / "cvae_beta1_seed42.pt"
_PILOT_CKPT_DIR = _REPO_ROOT / "checkpoints"
_SYNTHETIC_CACHE = _REPO_ROOT / "outputs" / "g1c_synthetic_residential.pt"

CVAE_LATENT_DIM = 128
CVAE_CHANNELS = (32, 64, 128, 256)


def content_to_resnet_input(x: torch.Tensor) -> torch.Tensor:
    """Resize content region (28:100 rows) to 224x224, ImageNet norm.

    x: (B, 3, 128, 128) in [0, 1]. Matches the eval transform used for
    the classifier diagnostic. Produces (B, 3, 224, 224) normalized.
    """
    content = x[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
    resized = torch.nn.functional.interpolate(
        content, size=(224, 224), mode="bilinear", align_corners=False
    )
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (resized - mean) / std


class PilotResNetDataset(Dataset):
    """Wraps a BDD100K128Dataset, applying the pilot's per-image transform.

    mode:
        'real'      — content→224 with no augmentation
        'conv_aug'  — RandomResizedCrop(224, 0.75-1.0), HFlip, ColorJitter
        'synthetic' — same as 'real' (synthetic images have no label noise)
    """

    def __init__(self, base_ds: BDD100K128Dataset, indices: list[int], mode: str):
        self.base_ds = base_ds
        self.indices = indices
        self.mode = mode

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x, y = self.base_ds[self.indices[i]]
        # x: (3, 128, 128) in [0, 1]
        content = x[:, CONTENT_TOP:CONTENT_BOTTOM, :].unsqueeze(0)  # (1, 3, 72, 128)
        if self.mode == "conv_aug" and self.base_ds.training_flag:  # noqa: PLC1901
            # Random resized crop from 72x128 content
            from torchvision.transforms import functional as TF  # noqa: N812

            content = TF.resized_crop(
                content,
                top=0,
                left=0,
                height=72,
                width=128,
                size=(224, 224),
                antialias=True,
            )
            if torch.rand(1).item() > 0.5:
                content = TF.hflip(content)
        else:
            content = torch.nn.functional.interpolate(
                content, size=(224, 224), mode="bilinear", align_corners=False
            )
        # ImageNet normalize (applied to (B, 3, 224, 224) or (1, 3, 224, 224))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (content - mean) / std
        return normalized.squeeze(0), y


class SyntheticDataset(Dataset):
    """Wraps pre-generated synthetic residential images, saved as tensor.

    Each item: (image, class_label=2). Images are already 128x128 with
    padding rows zeroed. We apply the same content→224→normalize transform.
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        # images: (N, 3, 128, 128) in [0, 1]; labels: (N,) all = 2
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.images.size(0)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = self.images[i]  # (3, 128, 128)
        y = int(self.labels[i].item())
        content = x[:, CONTENT_TOP:CONTENT_BOTTOM, :].unsqueeze(0)
        content = torch.nn.functional.interpolate(
            content, size=(224, 224), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        normalized = (content - mean) / std
        return normalized.squeeze(0), y


class ConcatDatasetWithFlag(Dataset):
    """Concatenates two datasets. Records whether this is a training set for aug."""

    def __init__(self, ds_list: list[Dataset]):
        self.ds_list = ds_list
        self.lengths = [len(d) for d in ds_list]
        self.cum = [0]
        for n in self.lengths:
            self.cum.append(self.cum[-1] + n)

    def __len__(self) -> int:
        return self.cum[-1]

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        for k, _n in enumerate(self.lengths):
            if i < self.cum[k + 1]:
                return self.ds_list[k][i - self.cum[k]]
        raise IndexError(i)


def set_ds_training_flag(ds: Dataset, flag: bool) -> None:
    """Recursively set training_flag on BDD100K128Dataset wrappers."""
    if hasattr(ds, "base_ds"):
        set_ds_training_flag(ds.base_ds, flag)
    if hasattr(ds, "ds_list"):
        for d in ds.ds_list:
            set_ds_training_flag(d, flag)
    if hasattr(ds, "training_flag"):
        ds.training_flag = flag


def generate_synthetic_residential(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate N_SYNTHETIC_RESIDENTIAL residential samples from cVAE.

    Caches to disk so subsequent arm trainings don't regenerate.
    """
    if _SYNTHETIC_CACHE.exists():
        print(f"Loading cached synthetic residential from {_SYNTHETIC_CACHE}")
        data = torch.load(_SYNTHETIC_CACHE, map_location="cpu", weights_only=False)
        return data["images"], data["labels"]

    print(f"Generating {N_SYNTHETIC_RESIDENTIAL} synthetic residential samples...")
    from urbangen.models.cvae import ConditionalVAE

    ckpt = torch.load(_CVAE_CKPT, map_location=device, weights_only=False)
    cvae = ConditionalVAE(
        image_size=IMAGE_SIZE,
        latent_dim=CVAE_LATENT_DIM,
        channels=CVAE_CHANNELS,
    )
    cvae.load_state_dict(ckpt["model_state_dict"])
    cvae.to(device).eval()

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SEED)
        all_imgs = []
        batch = 64
        for start in range(0, N_SYNTHETIC_RESIDENTIAL, batch):
            n = min(batch, N_SYNTHETIC_RESIDENTIAL - start)
            labels = torch.full((n,), 2, dtype=torch.long, device=device)
            with torch.no_grad():
                samples = cvae.sample(n, labels, device=device)
            all_imgs.append(samples.cpu())
        images = torch.cat(all_imgs, 0)  # (N, 3, 128, 128) in [0, 1]
        labels = torch.full((N_SYNTHETIC_RESIDENTIAL,), 2, dtype=torch.long)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    _SYNTHETIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "labels": labels}, _SYNTHETIC_CACHE)
    print(f"  saved to {_SYNTHETIC_CACHE}")
    return images, labels


def make_train_loader(arm: str, device) -> tuple[DataLoader, int]:
    """Return (loader, n_batches) for the given arm."""
    train_ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    train_ds.training_flag = True
    all_idx = list(range(len(train_ds)))

    if arm == "A":
        # Real only, no augmentation
        ds = PilotResNetDataset(train_ds, all_idx, mode="real")
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        return loader, len(loader)

    if arm == "B":
        # Real + synthetic
        real_ds = PilotResNetDataset(train_ds, all_idx, mode="real")
        synth_imgs, synth_labels = generate_synthetic_residential(device)
        synth_ds = SyntheticDataset(synth_imgs, synth_labels)
        combined = ConcatDatasetWithFlag([real_ds, synth_ds])
        loader = DataLoader(combined, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        return loader, len(loader)

    if arm == "C":
        # Real + conventional augmentation
        ds = PilotResNetDataset(train_ds, all_idx, mode="conv_aug")
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        return loader, len(loader)

    if arm == "D":
        # Real only (class weighting applied in loss)
        ds = PilotResNetDataset(train_ds, all_idx, mode="real")
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        return loader, len(loader)

    raise ValueError(f"unknown arm: {arm}")


def make_val_loader() -> DataLoader:
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    val_ds.training_flag = False
    idx = list(range(len(val_ds)))
    ds = PilotResNetDataset(val_ds, idx, mode="real")
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)


def class_weights_from_train() -> torch.Tensor:
    """Per-class weights from real train distribution. Used only by arm D."""
    train_ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    labels = train_ds.labels
    counts = torch.tensor([(labels == c).sum() for c in range(NUM_CLASSES)], dtype=torch.float32)
    weights = counts.sum() / (NUM_CLASSES * counts)
    return weights


def evaluate(model, loader, device, class_names) -> dict:
    """Return per-class recall, macro-F1, accuracy."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    per_class_recall = {
        class_names[c]: float(recall_score(labels == c, preds == c, zero_division=0))
        for c in range(NUM_CLASSES)
    }
    return {
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_recall": per_class_recall,
    }


def train_one_epoch(model, loader, optimizer, scaler, device, loss_fn) -> float:
    model.train()
    set_ds_training_flag(loader.dataset, True)
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["A", "B", "C", "D"], required=True)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, arm: {args.arm}")

    if not (_CACHE_ROOT / "manifest.json").exists():
        print(f"ERROR: cache not found at {_CACHE_ROOT}")
        return 1
    if args.arm == "B" and not _CVAE_CKPT.exists():
        print(f"ERROR: cVAE checkpoint not found at {_CVAE_CKPT}")
        return 1

    print(f"Building {args.arm} train loader...")
    train_loader, n_train_batches = make_train_loader(args.arm, device)
    val_loader = make_val_loader()
    print(f"  train batches: {n_train_batches}, val batches: {len(val_loader)}")

    model = models.resnet50(num_classes=NUM_CLASSES)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    if args.arm == "D":
        weights = class_weights_from_train().to(device)
        print(f"  class weights (arm D): {weights.tolist()}")
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    ckpt_path = _PILOT_CKPT_DIR / f"pilot_arm_{args.arm}.pt"
    best_macro_f1 = -1.0
    best_state = None

    print(f"\nTraining arm {args.arm} for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        t0 = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, loss_fn)
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, CLASS_NAMES)
        elapsed = time.perf_counter() - t0

        print(
            f"  epoch {epoch + 1}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_acc={val_metrics['accuracy']:.3f}  "
            f"val_macro_f1={val_metrics['macro_f1']:.3f}  "
            f"residential_recall={val_metrics['per_class_recall']['residential']:.3f}  "
            f"time={elapsed:.0f}s"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "arm": args.arm,
                    "epoch": epoch + 1,
                    "best_macro_f1": best_macro_f1,
                    "model_state_dict": best_state,
                    "val_metrics": val_metrics,
                },
                ckpt_path,
            )
            print(f"    -> saved best checkpoint (val_macro_f1={best_macro_f1:.3f})")

    print(f"\nArm {args.arm} complete. Best val macro-F1: {best_macro_f1:.3f}")
    print(f"Checkpoint: {ckpt_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
