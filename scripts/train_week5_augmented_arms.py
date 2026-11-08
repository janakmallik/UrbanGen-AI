"""Train Week 5 ResNet-50 under four augmentation arms, fair comparison.

Arms:
    A: real 3-class data only (baseline)
    B: real + night residential oversampled 3x (rebalancing control)
    C: real + strong photometric augmentation (appearance control)
    D: real + CLIP-filtered Wan synthetic residential frames

All arms: ImageNet ResNet-50, 5 epochs, AdamW 1e-4, batch 8,
class-weighted CE, best-epoch by val macro-F1, seed configurable.
Evaluated on real night residential TEST set only.

Usage:
    uv run python scripts/train_week5_augmented_arms.py --seeds 42
    uv run python scripts/train_week5_augmented_arms.py --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from urbangen.config import data_root, urbanmind_root
from urbangen.data.preprocessing import WEEK5_CLASS_TO_ID, week5_eval_transform

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("URBANMIND_REPO", str(_REPO_ROOT.parent / "UrbanMind-AI"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


# Paths
_URBANMIND = urbanmind_root()
_DATA = data_root()
_TRAIN_LABEL = _DATA / "labels" / "bdd100k_labels_images_train.json"
_VAL_LABEL = _DATA / "labels" / "bdd100k_labels_images_val.json"
_IMAGES = _DATA / "images"
_VAL_CLEAN = _URBANMIND / "data" / "splits" / "resnet_val_clean.json"
_TEST_CLEAN = _URBANMIND / "data" / "splits" / "resnet_test_clean.json"
_SYNTHETIC = _REPO_ROOT / "outputs" / "method_4_wan" / "frames_clip_filtered"
_OUT_DIR = _REPO_ROOT / "outputs" / "week5_arms"

EPOCHS = 5
BATCH_SIZE = 8
LR = 1e-4
N_BOOTSTRAP = 1000
OVERSAMPLE_FACTOR = 3


def index_images() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for p in _IMAGES.rglob("*.jpg"):
        idx.setdefault(p.name, p)
    return idx


def load_records(label_json: Path) -> list[dict]:
    return json.loads(label_json.read_text(encoding="utf-8"))


def select_split(
    records: list[dict],
    names_filter: set[str] | None,
    image_index: dict[str, Path],
) -> list[dict]:
    out = []
    for rec in records:
        scene = rec.get("attributes", {}).get("scene")
        if scene not in WEEK5_CLASS_TO_ID:
            continue
        name = rec["name"]
        if names_filter is not None and name not in names_filter:
            continue
        path = image_index.get(name)
        if path is None:
            continue
        out.append(
            {
                "path": path,
                "label": WEEK5_CLASS_TO_ID[scene],
                "timeofday": rec.get("attributes", {}).get("timeofday", "unknown"),
                "name": name,
            }
        )
    return out


class ClassificationDataset(Dataset):
    def __init__(self, records, transform, extra_synthetic: list[Path] | None = None):
        self.samples = [(r["path"], r["label"]) for r in records]
        if extra_synthetic:
            for p in extra_synthetic:
                self.samples.append((p, WEEK5_CLASS_TO_ID["residential"]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        with Image.open(path) as img:
            x = self.transform(img.convert("RGB"))
        return x, label


def make_photometric_transform():
    """Strong photometric augmentation for arm C."""
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Lambda(
                lambda t: torch.clamp(t * (0.3 + 0.7 * torch.rand(1)), 0.0, 1.0)
            ),  # gamma-ish
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_arm_loader(
    arm: str,
    full_train: list[dict],
    val_clean: list[dict],
    synthetic: list[Path],
    seed: int,
) -> DataLoader:
    """Build the training loader for a given arm."""
    transform_eval = week5_eval_transform()
    transform_photo = make_photometric_transform()

    if arm == "A":
        records = full_train
        ds = ClassificationDataset(records, transform_eval)
    elif arm == "B":
        # Oversample night residential training images 3x
        night_res = [r for r in full_train if r["label"] == 2 and r["timeofday"] == "night"]
        extra = night_res * (OVERSAMPLE_FACTOR - 1)  # add to existing singles
        records = full_train + extra
        ds = ClassificationDataset(records, transform_eval)
    elif arm == "C":
        records = full_train
        ds = ClassificationDataset(records, transform_photo)
    elif arm == "D":
        records = full_train
        if not synthetic:
            raise ValueError(f"Arm D requires synthetic frames in {_SYNTHETIC}")
        ds = ClassificationDataset(records, transform_eval, extra_synthetic=synthetic)
    else:
        raise ValueError(arm)

    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        generator=g,
        persistent_workers=True,
        pin_memory=True,
    )


def build_val_loader(val_clean: list[dict]) -> DataLoader:
    ds = ClassificationDataset(val_clean, week5_eval_transform())
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)


def build_night_test_loader(
    val_label_records: list[dict],
    test_clean_names: set[str],
    image_index: dict[str, Path],
) -> DataLoader:
    """Build the night residential test loader (only residential, night)."""
    records = []
    for rec in val_label_records:
        attrs = rec.get("attributes", {})
        if (
            attrs.get("scene") == "residential"
            and attrs.get("timeofday") == "night"
            and rec["name"] in test_clean_names
        ):
            path = image_index.get(rec["name"])
            if path:
                records.append(
                    {
                        "path": path,
                        "label": 2,
                        "timeofday": "night",
                        "name": rec["name"],
                    }
                )
    ds = ClassificationDataset(records, week5_eval_transform())
    return DataLoader(ds, batch_size=64, shuffle=False, num_workers=0), len(records)


def fresh_model(seed: int, device) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(2048, 3)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).argmax(dim=1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(preds), np.concatenate(labels)


def train_arm(arm, seed, full_train, val_clean, synthetic, night_loader, n_night, device):
    print(f"\n=== Arm {arm} (seed={seed}) ===")
    train_loader = build_arm_loader(arm, full_train, val_clean, synthetic, seed)
    val_loader = build_val_loader(val_clean)

    model = fresh_model(seed, device)

    # Class weights from full_train label distribution
    labels = np.array([r["label"] for r in full_train])
    counts = np.bincount(labels, minlength=3)
    class_weights = torch.tensor(len(labels) / (3.0 * counts), dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_state = None
    best_epoch = -1
    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_preds, val_labels = evaluate(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        print(f"  epoch {epoch + 1}/{EPOCHS}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

    model.load_state_dict(best_state)
    print(f"  Best val macro-F1: {best_val_f1:.4f} at epoch {best_epoch + 1}")

    # Evaluate on night residential test set
    preds, labels = evaluate(model, night_loader, device)
    assert n_night == len(labels), f"expected {n_night} night res test, got {len(labels)}"
    rec = recall_score(labels, preds, pos_label=2, zero_division=0)
    print(f"  Night residential recall: {rec:.4f}")

    return {
        "arm": arm,
        "seed": seed,
        "best_val_macro_f1": float(best_val_f1),
        "best_epoch": int(best_epoch + 1),
        "night_residential_recall": float(rec),
        "night_residential_predictions": preds.tolist(),
        "n_night": int(len(labels)),
    }


def paired_bootstrap_ci(preds_a, preds_b, n_boot=N_BOOTSTRAP, seed=42):
    """Paired bootstrap on night residential recall difference (B - A)."""
    rng = np.random.default_rng(seed)
    n = len(preds_a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        ra = (preds_a[idx] == 2).mean()
        rb = (preds_b[idx] == 2).mean()
        diffs[i] = rb - ra
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading image index...")
    image_index = index_images()
    print(f"  {len(image_index):,} images indexed")

    print("Loading labels...")
    train_records = load_records(_TRAIN_LABEL)
    val_records = load_records(_VAL_LABEL)
    val_clean_names = set(json.loads(_VAL_CLEAN.read_text(encoding="utf-8")))
    test_clean_names = set(json.loads(_TEST_CLEAN.read_text(encoding="utf-8")))

    full_train = select_split(train_records, None, image_index)
    val_clean = select_split(val_records, val_clean_names, image_index)
    print(f"  train: {len(full_train):,}  val: {len(val_clean):,}")

    # Night residential test set
    night_loader, n_night = build_night_test_loader(val_records, test_clean_names, image_index)
    print(f"  night residential test: {n_night} images")

    # Synthetic residential frames (may be empty if Phase 2 skipped)
    synthetic = sorted(_SYNTHETIC.glob("*.png")) if _SYNTHETIC.exists() else []
    print(f"  synthetic residential frames: {len(synthetic)}")

    arms = ["A", "B", "C"] if not synthetic else ["A", "B", "C", "D"]

    all_results = {arm: [] for arm in arms}
    for seed in args.seeds:
        for arm in arms:
            r = train_arm(
                arm, seed, full_train, val_clean, synthetic, night_loader, n_night, device
            )
            all_results[arm].append(r)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY — night residential recall")
    print("=" * 80)
    for arm in arms:
        recs = [r["night_residential_recall"] for r in all_results[arm]]
        print(f"  Arm {arm}: {np.mean(recs):.4f} ± {np.std(recs):.4f}  (seeds={recs})")

    # Paired bootstrap for D vs A, D vs B, D vs C
    if "D" in arms and len(args.seeds) >= 1:
        # Use the first seed's predictions for paired bootstrap
        preds_a = np.array(all_results["A"][0]["night_residential_predictions"])
        for other in ["B", "C"]:
            if other in arms:
                preds_o = np.array(all_results[other][0]["night_residential_predictions"])
                ci = paired_bootstrap_ci(preds_a, preds_o)
                print(
                    f"\n  Bootstrap {other} - A: Δ={ci['mean']:+.4f}  "
                    f"CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
                    f"{'CI excludes zero' if ci['ci_low'] > 0 or ci['ci_high'] < 0 else 'CI includes zero'}"
                )

        preds_d = np.array(all_results["D"][0]["night_residential_predictions"])
        for other in ["A", "B", "C"]:
            preds_o = np.array(all_results[other][0]["night_residential_predictions"])
            ci = paired_bootstrap_ci(preds_o, preds_d)
            print(
                f"  Bootstrap D - {other}: Δ={ci['mean']:+.4f}  "
                f"CI [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  "
                f"{'CI excludes zero' if ci['ci_low'] > 0 or ci['ci_high'] < 0 else 'CI includes zero'}"
            )

    out = {
        "seeds": args.seeds,
        "arms": arms,
        "results": all_results,
        "n_night_test": n_night,
        "n_synthetic_frames": len(synthetic),
    }
    (_OUT_DIR / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {_OUT_DIR / 'results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
