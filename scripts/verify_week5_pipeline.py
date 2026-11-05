"""Step 1 — three-condition pipeline verification for Week 5 ResNet-50.

The robustness report reported macro-F1 0.571 and residential recall 0.124
for Week 5. The Week 5 model card reports 0.6984 macro-F1 and ~0.41
residential recall. Same checkpoint, different numbers.

This script isolates the cause by evaluating the same 4,745 clean test
images through three preprocessing pipelines:

    (a) Original JPEG -> Resize(256) -> CenterCrop(224) -> ImageNet norm
        (Week 5's training pipeline — the reference)
    (b) Original JPEG -> resize full frame to (224, 224) squashed
        (keeps full width, loses aspect)
    (c) 128x128 cache content strip -> resize to 224x224
        (what the robustness report used)

Evaluates all three on:
    - Full test split: 4,932 images (to reproduce Week 5's published 0.7800 / 0.6984)
    - Clean test subset: 4,745 images (to compare against the robustness report)

If (a) reproduces published numbers, and (c) doesn't, the robustness report
measured the wrong pipeline. If all three are similar and all fail
residential, the model genuinely fails. If (a) reproduces published
numbers but (b) and (c) don't, the field-of-view / resolution effect
is quantified.

Run:
    uv run python scripts/verify_week5_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("URBANMIND_REPO", str(_REPO_ROOT.parent / "UrbanMind-AI"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.config import data_root, urbanmind_root  # noqa: E402
from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402

SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]
NUM_CLASSES = 3
CONTENT_TOP = 28
CONTENT_BOTTOM = 100
N_BOOTSTRAP = 1000
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

_URBANMIND = urbanmind_root()
_WEEK5_CKPT = _URBANMIND / "bdd_resnet50_best.pt"
_LABEL_DIR = data_root() / "labels"
_IMAGES_ROOT = data_root() / "images"
_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_OUT_DIR = _REPO_ROOT / "outputs" / "pipeline_verification"


# ---------- Data loading ----------
def load_test_metadata() -> tuple[dict[str, int], set[str]]:
    """Return ({filename: class_id}, clean_test_set).

    Class IDs are 0=city, 1=highway, 2=residential, matching the cache.
    """
    target = {"city street": 0, "highway": 1, "residential": 2}
    meta: dict[str, int] = {}
    val_json = _LABEL_DIR / "bdd100k_labels_images_val.json"
    data = json.loads(val_json.read_text(encoding="utf-8"))
    for rec in data:
        attrs = rec.get("attributes", {})
        scene = attrs.get("scene")
        if scene in target:
            meta[rec["name"]] = target[scene]
    clean_path = _URBANMIND / "data" / "splits" / "resnet_test_clean.json"
    clean = set(json.loads(clean_path.read_text(encoding="utf-8")))
    return meta, clean


def build_image_index() -> dict[str, Path]:
    """Filename -> full JPEG path. Warns on duplicates."""
    index: dict[str, Path] = {}
    dupes = 0
    for p in _IMAGES_ROOT.rglob("*.jpg"):
        if p.name in index:
            dupes += 1
        else:
            index[p.name] = p
    if dupes:
        print(f"  WARNING: {dupes} duplicate JPEG filenames; keeping first")
    return index


class TestJPEGDataset(Dataset):
    """Loads original JPEGs and applies one of three preprocessing modes."""

    def __init__(
        self, filenames: list[str], class_ids: list[int], image_index: dict[str, Path], mode: str
    ):
        assert mode in ("a", "b"), f"JPEG dataset mode must be 'a' or 'b', got {mode!r}"
        self.filenames = filenames
        self.class_ids = class_ids
        self.image_index = image_index
        self.mode = mode

        if mode == "a":
            # Week 5 training pipeline
            self.transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
                ]
            )
        else:  # mode "b": squashed full frame
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist()),
                ]
            )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        path = self.image_index.get(name)
        if path is None:
            raise FileNotFoundError(f"{name} not found in image index")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, self.class_ids[i]


class TestCacheDataset(Dataset):
    """Loads 128x128 cache content strip and resizes to 224x224 (mode 'c')."""

    def __init__(
        self,
        filenames: list[str],
        class_ids: list[int],
        cache_ds: BDD100K128Dataset,
        cache_name_to_idx: dict[str, int],
    ):
        self.filenames = filenames
        self.class_ids = class_ids
        self.cache_ds = cache_ds
        self.cache_name_to_idx = cache_name_to_idx

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        name = self.filenames[i]
        idx = self.cache_name_to_idx.get(name)
        if idx is None:
            raise KeyError(f"{name} not in cache")
        x, _ = self.cache_ds[idx]  # (3, 128, 128) in [0, 1]
        content = x[:, CONTENT_TOP:CONTENT_BOTTOM, :].unsqueeze(0)  # (1, 3, 72, 128)
        resized = F.interpolate(content, size=(224, 224), mode="bilinear", align_corners=False)
        mean = IMAGENET_MEAN.view(1, 3, 1, 1)
        std = IMAGENET_STD.view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        return normalized.squeeze(0), self.class_ids[i]


# ---------- Model ----------
def load_week5(device) -> torch.nn.Module:
    if not _WEEK5_CKPT.exists():
        raise FileNotFoundError(f"Week 5 checkpoint not found at {_WEEK5_CKPT}")
    model = models.resnet50(num_classes=NUM_CLASSES)
    ckpt = torch.load(_WEEK5_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


# ---------- Evaluation ----------
@torch.inference_mode()
def predict(model, loader: DataLoader, device) -> tuple[np.ndarray, np.ndarray]:
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(np.asarray(y))
    return np.concatenate(all_preds), np.concatenate(all_labels)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import f1_score, recall_score

    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "per_class_recall": {
            CLASS_NAMES[c]: float(recall_score(y_true == c, y_pred == c, zero_division=0))
            for c in range(NUM_CLASSES)
        },
        "n": int(len(y_true)),
        "class_counts": {CLASS_NAMES[c]: int((y_true == c).sum()) for c in range(NUM_CLASSES)},
    }


def bootstrap_metric(y_true: np.ndarray, y_pred: np.ndarray, metric_fn) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    point = float(metric_fn(y_true, y_pred))
    vals = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, n)
        vals[i] = metric_fn(y_true[idx], y_pred[idx])
    return {
        "mean": point,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }


def evaluate_condition(
    model,
    condition: str,
    filenames: list[str],
    class_ids: list[int],
    image_index: dict[str, Path],
    cache_ds,
    cache_name_to_idx,
    device,
) -> dict:
    print(f"\n=== Condition ({condition}): n={len(filenames):,} ===")
    if condition == "a":
        print("  Original JPEG -> Resize(256) -> CenterCrop(224) -> ImageNet norm")
        ds = TestJPEGDataset(filenames, class_ids, image_index, mode="a")
    elif condition == "b":
        print("  Original JPEG -> resize full frame to (224,224) squashed -> ImageNet norm")
        ds = TestJPEGDataset(filenames, class_ids, image_index, mode="b")
    elif condition == "c":
        print("  128x128 cache content strip -> resize to 224x224 -> ImageNet norm")
        ds = TestCacheDataset(filenames, class_ids, cache_ds, cache_name_to_idx)
    else:
        raise ValueError(condition)

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    preds, labels = predict(model, loader, device)
    result = metrics(labels, preds)

    # Bootstrap CIs on macro-F1 and residential recall
    from sklearn.metrics import f1_score, recall_score

    result["macro_f1_ci"] = bootstrap_metric(
        labels,
        preds,
        lambda yt, yp: f1_score(yt, yp, labels=[0, 1, 2], average="macro", zero_division=0),
    )
    result["residential_recall_ci"] = bootstrap_metric(
        labels,
        preds,
        lambda yt, yp: float(recall_score(yt == 2, yp == 2, zero_division=0)),
    )

    print(f"  accuracy:            {result['accuracy']:.4f}")
    print(
        f"  macro-F1:            {result['macro_f1']:.4f} "
        f"[{result['macro_f1_ci']['ci_low']:.4f}, {result['macro_f1_ci']['ci_high']:.4f}]"
    )
    for c in CLASS_NAMES:
        print(f"  {c:12s} recall:  {result['per_class_recall'][c]:.4f}")
    print(
        f"  residential recall CI: [{result['residential_recall_ci']['ci_low']:.4f}, "
        f"{result['residential_recall_ci']['ci_high']:.4f}]"
    )

    return result


# ---------- Main ----------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading metadata and image index...")
    meta, clean = load_test_metadata()
    image_index = build_image_index()
    print(
        f"  {len(meta):,} labeled images, {len(clean):,} in clean split, "
        f"{len(image_index):,} JPEGs indexed"
    )

    # Build full test set (4,932) and clean subset (4,745) preserving val order.
    # The test set is half of val; we need the val-side filenames. Reconstruct
    # the split via the same 50/50 logic UrbanMind used. But simpler: the
    # resnet_test_clean and resnet_test_excluded manifests + the 9,864-image
    # 3-class val pool partition cleanly into four disjoint sets. So:
    #   full test = clean ∪ test_excluded
    test_excl_path = _URBANMIND / "data" / "splits" / "resnet_test_excluded.json"
    test_excluded = set(json.loads(test_excl_path.read_text(encoding="utf-8")))
    full_test = sorted(clean | test_excluded)
    clean_test = sorted(clean)

    # Drop any filename that's missing from the image index (should be zero)
    full_test = [n for n in full_test if n in image_index]
    clean_test = [n for n in clean_test if n in image_index]
    print(f"  full test (clean ∪ test_excluded): {len(full_test):,}")
    print(f"  clean test: {len(clean_test):,}")

    full_classes = [meta[n] for n in full_test]
    clean_classes = [meta[n] for n in clean_test]

    # Load cache and build name->idx map (cache order is resnet_test_clean order)
    print("\nLoading cache and building name->idx map...")
    cache_ds = BDD100K128Dataset(_CACHE_ROOT, "test", allow_test=True)
    cache_name_to_idx = {}
    for i, n in enumerate(clean_test):
        if i >= len(cache_ds):
            raise RuntimeError(
                f"Cache shorter than clean test: {len(cache_ds)} vs {len(clean_test)}"
            )
        cache_name_to_idx[n] = i
    print(f"  cache: {len(cache_ds):,} images, {len(cache_name_to_idx):,} name mappings")

    print("\nLoading Week 5 ResNet-50...")
    model = load_week5(device)

    results = {}

    # Condition A on full test (should reproduce published 0.7800 / 0.6984)
    print("\n" + "=" * 70)
    print("REFERENCE: Condition (a) on FULL test (4,932 images)")
    print("Expected from published Week 5 card: acc 0.7800, macro-F1 0.6984")
    print("=" * 70)
    results["a_full"] = evaluate_condition(
        model,
        "a",
        full_test,
        full_classes,
        image_index,
        cache_ds,
        cache_name_to_idx,
        device,
    )

    # Condition A on clean subset
    print("\n" + "=" * 70)
    print("REFERENCE: Condition (a) on CLEAN test (4,745 images)")
    print("Compare to robustness report's 0.571 / 0.124")
    print("=" * 70)
    results["a_clean"] = evaluate_condition(
        model,
        "a",
        clean_test,
        clean_classes,
        image_index,
        cache_ds,
        cache_name_to_idx,
        device,
    )

    # Condition B on clean subset
    print("\n" + "=" * 70)
    print("Pipeline effect (b): squashed full frame")
    print("=" * 70)
    results["b_clean"] = evaluate_condition(
        model,
        "b",
        clean_test,
        clean_classes,
        image_index,
        cache_ds,
        cache_name_to_idx,
        device,
    )

    # Condition C on clean subset
    print("\n" + "=" * 70)
    print("Pipeline effect (c): 128x128 cache content strip")
    print("=" * 70)
    results["c_clean"] = evaluate_condition(
        model,
        "c",
        clean_test,
        clean_classes,
        image_index,
        cache_ds,
        cache_name_to_idx,
        device,
    )

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Condition':<40} {'n':>6} {'acc':>8} {'macro-F1':>10} {'res recall':>12}")
    print("-" * 90)
    rows = [
        ("(a) full test  [reference]", results["a_full"]),
        ("(a) clean test", results["a_clean"]),
        ("(b) clean test  [squashed full frame]", results["b_clean"]),
        ("(c) clean test  [128x128 cache]", results["c_clean"]),
    ]
    for name, r in rows:
        print(
            f"{name:<40} {r['n']:>6} {r['accuracy']:>8.4f} "
            f"{r['macro_f1']:>10.4f} {r['per_class_recall']['residential']:>12.4f}"
        )

    # Interpretation
    print("\n" + "=" * 90)
    print("INTERPRETATION")
    print("=" * 90)
    a_full_f1 = results["a_full"]["macro_f1"]
    a_full_res = results["a_full"]["per_class_recall"]["residential"]
    a_clean_f1 = results["a_clean"]["macro_f1"]
    a_clean_res = results["a_clean"]["per_class_recall"]["residential"]
    c_clean_f1 = results["c_clean"]["macro_f1"]
    c_clean_res = results["c_clean"]["per_class_recall"]["residential"]

    # Reference check
    reference_match = abs(a_full_f1 - 0.6984) < 0.02 and abs(a_full_res - 0.41) < 0.10
    print("\n1. Reference check (condition a on full test):")
    print(f"   macro-F1 = {a_full_f1:.4f} (target 0.6984)")
    print(f"   residential recall = {a_full_res:.4f} (target ~0.41)")
    if reference_match:
        print("   -> MATCHES published Week 5. Condition (a) is the correct pipeline.")
    else:
        print(
            "   -> DOES NOT match published. Investigate checkpoint / class mapping / normalization before drawing conclusions."
        )

    # Clean vs full (should be similar if excluded images aren't special)
    print("\n2. Clean subset effect (a_full vs a_clean):")
    print(f"   macro-F1: {a_full_f1:.4f} vs {a_clean_f1:.4f} (delta {a_clean_f1 - a_full_f1:+.4f})")
    print(
        f"   residential: {a_full_res:.4f} vs {a_clean_res:.4f} "
        f"(delta {a_clean_res - a_full_res:+.4f})"
    )

    # Pipeline effect on clean (a_clean vs c_clean — the key comparison)
    print("\n3. Pipeline effect (a_clean vs c_clean) — THE KEY COMPARISON:")
    print(
        f"   macro-F1: {a_clean_f1:.4f} vs {c_clean_f1:.4f} (delta {c_clean_f1 - a_clean_f1:+.4f})"
    )
    print(
        f"   residential: {a_clean_res:.4f} vs {c_clean_res:.4f} "
        f"(delta {c_clean_res - a_clean_res:+.4f})"
    )

    if abs(c_clean_res - a_clean_res) > 0.10:
        print("   -> RESOLUTION/EFFECT IS THE CAUSE. The robustness report's 0.12 was a")
        print("      pipeline artifact. Rerun the robustness report at condition (a).")
    elif abs(c_clean_res - a_clean_res) < 0.03:
        print("   -> PIPELINE EFFECT IS SMALL. The residential weakness is real at all")
        print("      resolutions. Investigate the model or class definition.")
    else:
        print("   -> MIXED. Pipeline contributes to the gap but doesn't fully explain it.")

    # Save
    out = {
        "reference_target": {"macro_f1": 0.6984, "residential_recall_approx": 0.41},
        "reference_matches": bool(reference_match),
        "results": results,
        "summary": {
            "a_full": {
                "macro_f1": a_full_f1,
                "residential_recall": a_full_res,
                "n": results["a_full"]["n"],
            },
            "a_clean": {
                "macro_f1": a_clean_f1,
                "residential_recall": a_clean_res,
                "n": results["a_clean"]["n"],
            },
            "b_clean": {
                "macro_f1": results["b_clean"]["macro_f1"],
                "residential_recall": results["b_clean"]["per_class_recall"]["residential"],
                "n": results["b_clean"]["n"],
            },
            "c_clean": {
                "macro_f1": c_clean_f1,
                "residential_recall": c_clean_res,
                "n": results["c_clean"]["n"],
            },
        },
    }
    out_path = _OUT_DIR / "pipeline_verification.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
