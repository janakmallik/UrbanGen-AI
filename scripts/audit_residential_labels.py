"""Step 0a — residential label audit.

Uses frozen ImageNet ResNet-50 features + 5-fold out-of-fold logistic
regression predictions to check whether the residential label is:
    - noisy (images mislabeled as residential that aren't),
    - ambiguous (residential vs city street boundary is unclear), or
    - genuinely hard (labels are clean but the visual task is difficult).

Also saves:
    - OOF probabilities (outputs/label_audit/oof_probs.npy)
    - Train filename list in cache order (outputs/label_audit/train_filenames.json)
      — used by Step 0b to join weather/timeofday metadata.
    - Two 24-image grids: suspect_residential, confident_residential

Run:
    uv run python scripts/audit_residential_labels.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision import models
from torchvision.utils import save_image

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.config import data_root, urbanmind_root  # noqa: E402
from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402

CLASS_NAMES = ["city street", "highway", "residential"]
SEED = 42
CONTENT_TOP = 28
CONTENT_BOTTOM = 100

_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_URBANMIND = urbanmind_root()
_OUT_DIR = _REPO_ROOT / "outputs" / "label_audit"


def extract_features(ds: BDD100K128Dataset, device) -> np.ndarray:
    """Extract 2048-d ResNet-50 features for every image in the dataset.

    Uses content region only (rows 28:100), resized to 144x256 to
    preserve aspect ratio, then ImageNet-normalized.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    feats = []
    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")),
    ):
        for x, _ in loader:
            x = x[:, :, CONTENT_TOP:CONTENT_BOTTOM, :].to(device)
            x = torch.nn.functional.interpolate(x, size=(144, 256), mode="bilinear")
            feats.append(model((x - mean) / std).float().cpu())
    return torch.cat(feats, 0).numpy()


def oof_predict(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold OOF probabilities on frozen features."""
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probs = cross_val_predict(clf, x, y, cv=skf, method="predict_proba", n_jobs=1)
    return probs


def load_train_filenames() -> list[str]:
    """Reconstruct the train filenames in cache order.

    The cache was built by scripts/precompute_bdd100k_128.py in the order
    of the train label JSON's 3-class entries. Re-derive that order.
    """
    label_json = data_root() / "labels" / "bdd100k_labels_images_train.json"
    data = json.loads(label_json.read_text(encoding="utf-8"))
    target = {"city street", "highway", "residential"}
    return [rec["name"] for rec in data if rec.get("attributes", {}).get("scene") in target]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Loading train dataset ===")
    ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    y = np.asarray(ds.labels).astype(int)
    print(
        f"  {len(y):,} images, class counts: "
        f"{[(CLASS_NAMES[c], int((y == c).sum())) for c in range(3)]}"
    )

    print("\n=== Extracting ResNet-50 features ===")
    feat_path = _OUT_DIR / "train_feats.npy"
    if feat_path.exists():
        print(f"  loading cached features from {feat_path}")
        x = np.load(feat_path)
    else:
        x = extract_features(ds, device)
        np.save(feat_path, x)
        print(f"  saved to {feat_path}")
    print(f"  x shape: {x.shape}")

    print("\n=== 5-fold OOF logistic regression ===")
    probs = oof_predict(x, y)
    np.save(_OUT_DIR / "oof_probs.npy", probs)
    pred = probs.argmax(1)

    print("\n=== OOF per-class recall ===")
    for c in range(3):
        r = recall_score(y == c, pred == c, zero_division=0)
        n = int((y == c).sum())
        print(f"  {CLASS_NAMES[c]:12s}: recall={r:.4f} (n={n:,})")

    print("\n=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    print(f"  {'':14s} {'pred city':>12s} {'pred hwy':>12s} {'pred res':>12s}")
    for i, name in enumerate(CLASS_NAMES):
        row = cm[i]
        print(f"  {name:14s} {row[0]:>12d} {row[1]:>12d} {row[2]:>12d}")

    print("\n=== Suspect detection ===")
    # Principled suspect: model is confident it's NOT the true class
    #   P(true) < 0.5  AND  P(true) < max_other
    # Alternative class = argmax over the wrong classes
    max_other = np.max(np.where(np.arange(3)[None, :] == y[:, None], -1.0, probs), axis=1)
    p_true = probs[np.arange(len(y)), y]
    suspect_mask = (p_true < 0.5) & (p_true < max_other)
    alt_class = np.argmax(np.where(np.arange(3)[None, :] == y[:, None], -1.0, probs), axis=1)

    for c in range(3):
        mask = (y == c) & suspect_mask
        n = int((y == c).sum())
        print(
            f"  {CLASS_NAMES[c]:12s}: {int(mask.sum()):5d} suspects of {n:5d} "
            f"({100 * mask.sum() / n:.1f}%)"
        )

    # Identify residential suspect and confident images
    residential_idx = np.where(y == 2)[0]
    res_suspects = residential_idx[suspect_mask[residential_idx]]
    # Sort by confidence in the alt class (most confident that it's NOT residential)
    res_suspects_sorted = res_suspects[np.argsort(-max_other[res_suspects])][:24]
    res_confident = residential_idx[np.argsort(-p_true[residential_idx])][:24]

    # Save grids
    for name, idx_array in [
        ("suspect_residential", res_suspects_sorted),
        ("confident_residential", res_confident),
    ]:
        if len(idx_array) == 0:
            print(f"  WARNING: no images for {name}")
            continue
        imgs = torch.stack([ds[int(i)][0][:, CONTENT_TOP:CONTENT_BOTTOM, :] for i in idx_array])
        grid_path = _OUT_DIR / f"{name}.png"
        save_image(imgs, grid_path, nrow=6, padding=2)
        print(f"  saved {grid_path} ({len(idx_array)} images)")

    # Save filename list for Step 0b (weather metadata join)
    print("\n=== Saving train filename list ===")
    train_filenames = load_train_filenames()
    if len(train_filenames) != len(y):
        print(
            f"  WARNING: train_filenames ({len(train_filenames)}) != "
            f"dataset size ({len(y)}). Cache and label JSON may be out of sync."
        )
    else:
        print(f"  {len(train_filenames):,} filenames, matches dataset size")
    (_OUT_DIR / "train_filenames.json").write_text(
        json.dumps(train_filenames, indent=2), encoding="utf-8"
    )

    # Save per-image metadata for later analysis
    per_image = {
        "filename": train_filenames if len(train_filenames) == len(y) else [],
        "y_true": y.tolist(),
        "p_true": p_true.tolist(),
        "p_alt": max_other.tolist(),
        "alt_class": alt_class.tolist(),
        "suspect_mask": suspect_mask.tolist(),
    }
    (_OUT_DIR / "per_image_analysis.json").write_text(json.dumps(per_image), encoding="utf-8")

    print("\nDone.")
    print(f"  Outputs saved to {_OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
