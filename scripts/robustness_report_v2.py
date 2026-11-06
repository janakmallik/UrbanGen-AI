"""Corrected robustness report at condition (a) — native JPEGs.

Supersedes robustness_report.py (which used 128x128 cache content strip;
pipeline verification showed this reduced residential recall 0.41 → 0.12).

Verified against Week 5 training script (UrbanMind-AI/scripts/train_bdd100k_resnet50.py):
    - Class mapping: ["city street", "highway", "residential"] → 0, 1, 2
    - Eval transform: Resize(256) → CenterCrop(224) → ImageNet norm
    - Training transform: RandomResizedCrop(224, scale=0.75-1.0) +
      HorizontalFlip + ColorJitter

Evaluates BOTH:
    - Full test set (4,932 = clean 4,745 + excluded 187) — reference check
    - Clean test set (4,745) — the analysis subset

Reports:
    - Per-class P/R/F1 with class-specific sample counts
    - Per-weather, per-timeofday subgroup metrics
    - Within-class comparisons with unpaired bootstrap
    - Class-specific n for each comparison; small comparisons flagged
    - Exploratory-only note (no multiple-comparison correction)

Run:
    uv run python scripts/robustness_report_v2.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault("URBANMIND_REPO", str(_REPO_ROOT.parent / "UrbanMind-AI"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.config import data_root, urbanmind_root  # noqa: E402

SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]
NUM_CLASSES = 3
N_BOOTSTRAP = 1000
MIN_CLASS_N = 30  # class-specific flag threshold
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Verified against Week 5 training script (train_bdd100k_resnet50.py)
EXPECTED_REFERENCE = {
    "accuracy": 0.7800,
    "macro_f1": 0.6984,
    "residential_recall": 0.4121,
    "n_full_test": 4932,
}
TOLERANCE = 0.02  # reference check tolerance

_URBANMIND = urbanmind_root()
_WEEK5_CKPT = _URBANMIND / "bdd_resnet50_best.pt"
_LABEL_DIR = data_root() / "labels"
_IMAGES_ROOT = data_root() / "images"
_OUT_DIR = _REPO_ROOT / "outputs" / "robustness_v2"
_DOC_PATH = _REPO_ROOT / "docs" / "robustness_report_v2.md"


# ---- Data ----
def load_metadata() -> dict[str, dict]:
    target = {"city street", "highway", "residential"}
    val_json = _LABEL_DIR / "bdd100k_labels_images_val.json"
    data = json.loads(val_json.read_text(encoding="utf-8"))
    meta: dict[str, dict] = {}
    for rec in data:
        attrs = rec.get("attributes", {}) or {}
        if attrs.get("scene") in target:
            meta[rec["name"]] = {
                "scene": attrs["scene"],
                "weather": attrs.get("weather") or "unknown",
                "timeofday": attrs.get("timeofday") or "unknown",
            }
    return meta


def load_split(name: str) -> list[str]:
    p = _URBANMIND / "data" / "splits" / name
    return sorted(json.loads(p.read_text(encoding="utf-8")))


def build_image_index() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    dupes: list[str] = []
    for p in _IMAGES_ROOT.rglob("*.jpg"):
        if p.name in idx:
            dupes.append(p.name)
        else:
            idx[p.name] = p
    if dupes:
        raise RuntimeError(
            f"{len(dupes)} duplicate image basenames found; "
            f"first 5: {dupes[:5]}. Dataset indexing is ambiguous."
        )
    return idx


class TestJPEGDataset(Dataset):
    def __init__(self, filenames, class_ids, image_index):
        self.filenames = filenames
        self.class_ids = class_ids
        self.image_index = image_index
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, i):
        with Image.open(self.image_index[self.filenames[i]]) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, self.class_ids[i]


# ---- Model ----
def load_week5(device):
    model = models.resnet50(num_classes=NUM_CLASSES)
    ckpt = torch.load(_WEEK5_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model


@torch.inference_mode()
def predict_all(model, loader, device):
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_labels.append(np.asarray(y))
    return np.concatenate(all_probs), np.concatenate(all_labels)


# ---- Metrics ----
def subgroup_metrics(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)
        ),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": float(p[c]),
                "recall": float(r[c]),
                "f1": float(f[c]),
                "support": int(s[c]),
            }
            for c in range(NUM_CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
        "n": int(len(y_true)),
        "class_counts": {CLASS_NAMES[c]: int((y_true == c).sum()) for c in range(NUM_CLASSES)},
    }


def unpaired_bootstrap_difference(y_a, p_a, y_b, p_b, metric_fn, n_boot=N_BOOTSTRAP):
    """Unpaired bootstrap: metric(B) - metric(A), with CI."""
    rng = np.random.default_rng(SEED)
    n_a, n_b = len(y_a), len(y_b)
    if n_a < 5 or n_b < 5:
        return {
            "point_a": None,
            "point_b": None,
            "difference": None,
            "ci_low": None,
            "ci_high": None,
            "n_a": n_a,
            "n_b": n_b,
            "excludes_zero": None,
        }
    point_a = metric_fn(y_a, p_a)
    point_b = metric_fn(y_b, p_b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx_a = rng.integers(0, n_a, n_a)
        idx_b = rng.integers(0, n_b, n_b)
        deltas[i] = metric_fn(y_b[idx_b], p_b[idx_b]) - metric_fn(y_a[idx_a], p_a[idx_a])
    ci_low = float(np.percentile(deltas, 2.5))
    ci_high = float(np.percentile(deltas, 97.5))
    return {
        "point_a": float(point_a),
        "point_b": float(point_b),
        "difference": float(point_b - point_a),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_a": n_a,
        "n_b": n_b,
        "excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "class_specific_n_adequate": bool(n_a >= MIN_CLASS_N and n_b >= MIN_CLASS_N),
    }


# ---- Report writer ----
def _row_cells(m):
    return (
        f"{m['accuracy']:.3f} | {m['macro_f1']:.3f} | "
        f"{m['per_class']['city street']['recall']:.3f} | "
        f"{m['per_class']['highway']['recall']:.3f} | "
        f"{m['per_class']['residential']['precision']:.3f} | "
        f"{m['per_class']['residential']['recall']:.3f} | "
        f"{m['per_class']['residential']['f1']:.3f}"
    )


def write_report(report, path):
    lines = []
    lines.append("# Week 5 ResNet-50 — Robustness Report (Corrected)\n")
    lines.append(
        "Corrected report using condition (a): native JPEGs with "
        "`Resize(256) → CenterCrop(224) → ImageNet norm`. "
        "Supersedes `robustness_report.md` which used the 128×128 cache "
        "pipeline (see `outputs/pipeline_verification/`).\n"
    )
    lines.append("## Verification and provenance\n")
    lines.append(
        "- **Pipeline verified** against `UrbanMind-AI/scripts/train_bdd100k_resnet50.py`: "
        "eval transform is `Resize(256) → CenterCrop(224) → ToTensor → "
        "Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])`. "
        "Condition (a) matches this exactly."
    )
    lines.append(
        "- **Class mapping verified**: `VIABLE_CLASSES = "
        '["city street", "highway", "residential"]` → '
        "`LABEL_TO_ID = {name: i for i, name in enumerate(VIABLE_CLASSES)}`, "
        "so 0=city, 1=highway, 2=residential."
    )
    rc = report["reference_check"]
    status = "✅ PASS" if rc["passes"] else "❌ FAIL"
    lines.append(
        f"- **Reference check {status}**: full test (n={rc['n']:,}) "
        f"accuracy={rc['accuracy']:.4f} (expected {EXPECTED_REFERENCE['accuracy']:.4f}), "
        f"macro-F1={rc['macro_f1']:.4f} (expected {EXPECTED_REFERENCE['macro_f1']:.4f}), "
        f"residential recall={rc['residential_recall']:.4f} "
        f"(expected {EXPECTED_REFERENCE['residential_recall']:.4f})."
    )
    lines.append(f"- **Seed**: {report['seed']}")
    lines.append(f"- **Bootstrap resamples**: {report['n_bootstrap']}")
    lines.append("")

    lines.append("## Data integrity\n")
    di = report["data_integrity"]
    lines.append(f"- Clean split manifest: {di['clean_manifest_count']:,} filenames")
    lines.append(f"- Excluded split manifest: {di['excluded_manifest_count']:,} filenames")
    lines.append(f"- Full test (clean ∪ excluded): {di['full_count']:,} filenames")
    lines.append(f"- Missing images: {di['missing_images']}")
    lines.append(f"- Missing metadata: {di['missing_metadata']}")
    lines.append("")

    # Overall — clean
    o = report["overall_clean"]
    lines.append("## Overall (clean test set, 4,745 images)\n")
    lines.append(f"- Accuracy: {o['accuracy']:.4f}")
    lines.append(f"- Balanced accuracy: {o['balanced_accuracy']:.4f}")
    lines.append(f"- Macro-F1: {o['macro_f1']:.4f}")
    for c in CLASS_NAMES:
        m = o["per_class"][c]
        lines.append(
            f"- {c} — P={m['precision']:.4f} R={m['recall']:.4f} "
            f"F1={m['f1']:.4f} (n={m['support']})"
        )
    lines.append("")

    lines.append("## By weather\n")
    lines.append("| Weather | n | acc | macro-F1 | City R | Hwy R | Res P | Res R | Res F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in report["by_weather"]:
        name = g["name"] + (" ⚠️" if g["n"] < MIN_CLASS_N else "")
        lines.append(f"| {name} | {g['n']} | {_row_cells(g['metrics'])} |")
    lines.append("")
    lines.append(
        "⚠️ marks subgroups with total n < 30. "
        "Class-specific n may still be smaller; check the within-class "
        "comparisons below for class-specific sample sizes.\n"
    )

    lines.append("## By time of day\n")
    lines.append("| Time | n | acc | macro-F1 | City R | Hwy R | Res P | Res R | Res F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for g in report["by_timeofday"]:
        name = g["name"] + (" ⚠️" if g["n"] < MIN_CLASS_N else "")
        lines.append(f"| {name} | {g['n']} | {_row_cells(g['metrics'])} |")
    lines.append("")

    lines.append("## Within-class comparisons (unpaired bootstrap)\n")
    lines.append(
        "Direction: each comparison reads **A → B** where the heading "
        "specifies A and B. Δ = metric(B) − metric(A). "
        "Positive Δ = B better; negative = A better.\n"
    )
    lines.append(
        "⭐ = nominal 95% unpaired bootstrap CI excludes zero. "
        "**Comparisons are exploratory and not adjusted for multiple testing.** "
        "(Six comparisons total across two weather/time pairs × three classes.)\n"
    )

    for key, heading in [
        ("clear_vs_rainy", "Clear → rainy (Δ = rainy − clear)"),
        ("daytime_vs_night", "Daytime → night (Δ = night − daytime)"),
    ]:
        d = report["within_class_differences"].get(key)
        if d is None:
            continue
        lines.append(f"### {heading}\n")
        lines.append("| Class | A recall | B recall | Δ | 95% CI | n_A | n_B | Flags |")
        lines.append("|---|---:|---:|---:|---|---:|---:|---|")
        for c in CLASS_NAMES:
            dd = d[c]
            if dd is None or dd["difference"] is None:
                lines.append(f"| {c} | — | — | — | insufficient | — | — | — |")
                continue
            flags = []
            if dd.get("excludes_zero"):
                flags.append("⭐")
            if not dd.get("class_specific_n_adequate", True):
                flags.append("⚠️ small n")
            lines.append(
                f"| {c} | {dd['point_a']:.3f} | {dd['point_b']:.3f} | "
                f"{dd['difference']:+.3f} | [{dd['ci_low']:+.3f}, {dd['ci_high']:+.3f}] | "
                f"{dd['n_a']} | {dd['n_b']} | {' '.join(flags) if flags else '—'} |"
            )
        lines.append("")

    lines.append("## Cross-tabs\n")
    lines.append("### Weather × scene class\n")
    lines.append("| Weather | " + " | ".join(CLASS_NAMES) + " | Total |")
    lines.append("|---" * (NUM_CLASSES + 2) + "|")
    for k, v in report["cross_tab_weather"].items():
        lines.append(
            f"| {k} | "
            + " | ".join(str(v.get(c, 0)) for c in CLASS_NAMES)
            + f" | {sum(v.values())} |"
        )
    lines.append("")
    lines.append("### Time-of-day × scene class\n")
    lines.append("| Time | " + " | ".join(CLASS_NAMES) + " | Total |")
    lines.append("|---" * (NUM_CLASSES + 2) + "|")
    for k, v in report["cross_tab_timeofday"].items():
        lines.append(
            f"| {k} | "
            + " | ".join(str(v.get(c, 0)) for c in CLASS_NAMES)
            + f" | {sum(v.values())} |"
        )
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append("- ⚠️ on a subgroup: total n < 30; treat as qualitative.")
    lines.append("- ⚠️ small n on a comparison: class-specific n < 30 in A or B.")
    lines.append("- ⭐: nominal 95% CI excludes zero (exploratory, unadjusted).")
    lines.append(
        "- Compare **within a class** to isolate weather/time effects from "
        "class-distribution differences."
    )
    lines.append(
        "- Recall alone can be inflated by predicting a class more often; "
        "check precision and F1 alongside recall."
    )

    path.write_text("\n".join(lines), encoding="utf-8")


# ---- Main ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading metadata, image index, split manifests...")
    meta = load_metadata()
    image_index = build_image_index()
    clean = load_split("resnet_test_clean.json")
    excluded = load_split("resnet_test_excluded.json")
    full_test = sorted(set(clean) | set(excluded))
    print(f"  clean: {len(clean):,}")
    print(f"  excluded: {len(excluded):,}")
    print(f"  full test: {len(full_test):,}")

    # Data integrity: fail loudly
    missing_images_clean = [n for n in clean if n not in image_index]
    missing_meta_clean = [n for n in clean if n not in meta]
    missing_images_full = [n for n in full_test if n not in image_index]
    missing_meta_full = [n for n in full_test if n not in meta]
    if missing_images_clean or missing_meta_clean or missing_images_full or missing_meta_full:
        raise RuntimeError(
            f"Data integrity failure. "
            f"Missing images (clean): {len(missing_images_clean)}, "
            f"missing metadata (clean): {len(missing_meta_clean)}, "
            f"missing images (full): {len(missing_images_full)}, "
            f"missing metadata (full): {len(missing_meta_full)}. "
            f"First missing image (full): {missing_images_full[:1]}"
        )

    class_map = {c: i for i, c in enumerate(CLASS_NAMES)}

    print("Loading Week 5 ResNet-50...")
    model = load_week5(device)

    # ---- Full test set: reference check ----
    print(f"\n=== Full test set (n={len(full_test):,}) — reference check ===")
    full_class_ids = [class_map[meta[n]["scene"]] for n in full_test]
    full_ds = TestJPEGDataset(full_test, full_class_ids, image_index)
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False, num_workers=0)
    full_probs, full_labels = predict_all(model, full_loader, device)
    full_preds = full_probs.argmax(axis=1)
    full_metrics = subgroup_metrics(full_labels, full_preds)

    print(
        f"  accuracy: {full_metrics['accuracy']:.4f} "
        f"(expected {EXPECTED_REFERENCE['accuracy']:.4f})"
    )
    print(
        f"  macro-F1: {full_metrics['macro_f1']:.4f} "
        f"(expected {EXPECTED_REFERENCE['macro_f1']:.4f})"
    )
    print(
        f"  residential recall: {full_metrics['per_class']['residential']['recall']:.4f} "
        f"(expected {EXPECTED_REFERENCE['residential_recall']:.4f})"
    )

    reference_passes = (
        abs(full_metrics["accuracy"] - EXPECTED_REFERENCE["accuracy"]) < TOLERANCE
        and abs(full_metrics["macro_f1"] - EXPECTED_REFERENCE["macro_f1"]) < TOLERANCE
        and abs(
            full_metrics["per_class"]["residential"]["recall"]
            - EXPECTED_REFERENCE["residential_recall"]
        )
        < TOLERANCE
    )
    print(f"  Reference: {'✅ PASS' if reference_passes else '❌ FAIL'}")
    if not reference_passes:
        print("  WARNING: reference check failed. Do not trust downstream analysis.")

    reference_check = {
        "n": int(len(full_test)),
        "accuracy": float(full_metrics["accuracy"]),
        "macro_f1": float(full_metrics["macro_f1"]),
        "residential_recall": float(full_metrics["per_class"]["residential"]["recall"]),
        "expected": EXPECTED_REFERENCE,
        "tolerance": TOLERANCE,
        "passes": bool(reference_passes),
    }

    # ---- Clean test set: analysis ----
    print(f"\n=== Clean test set (n={len(clean):,}) — analysis ===")
    clean_class_ids = [class_map[meta[n]["scene"]] for n in clean]
    clean_ds = TestJPEGDataset(clean, clean_class_ids, image_index)
    clean_loader = DataLoader(clean_ds, batch_size=64, shuffle=False, num_workers=0)
    probs, labels = predict_all(model, clean_loader, device)
    preds = probs.argmax(axis=1)

    overall = subgroup_metrics(labels, preds)
    print(
        f"  accuracy={overall['accuracy']:.4f}, "
        f"macro-F1={overall['macro_f1']:.4f}, "
        f"res recall={overall['per_class']['residential']['recall']:.4f}"
    )

    # Group indices
    weather_idx = defaultdict(list)
    tod_idx = defaultdict(list)
    for i, n in enumerate(clean):
        weather_idx[meta[n]["weather"]].append(i)
        tod_idx[meta[n]["timeofday"]].append(i)

    by_weather = []
    for w in sorted(weather_idx.keys()):
        idx = np.array(weather_idx[w])
        m = subgroup_metrics(labels[idx], preds[idx])
        by_weather.append({"name": w, "n": int(len(idx)), "metrics": m})
        flag = " ⚠️" if len(idx) < MIN_CLASS_N else ""
        print(
            f"  weather={w:15s} n={len(idx):4d}  macro-F1={m['macro_f1']:.3f}  "
            f"res_recall={m['per_class']['residential']['recall']:.3f}{flag}"
        )

    by_tod = []
    for t in sorted(tod_idx.keys()):
        idx = np.array(tod_idx[t])
        m = subgroup_metrics(labels[idx], preds[idx])
        by_tod.append({"name": t, "n": int(len(idx)), "metrics": m})
        flag = " ⚠️" if len(idx) < MIN_CLASS_N else ""
        print(
            f"  tod={t:12s} n={len(idx):4d}  macro-F1={m['macro_f1']:.3f}  "
            f"res_recall={m['per_class']['residential']['recall']:.3f}{flag}"
        )

    # Within-class differences
    def within_class(idx_by_group, group_a, group_b):
        result = {}
        for c in range(NUM_CLASSES):
            idx_a = np.array([i for i in idx_by_group.get(group_a, []) if labels[i] == c])
            idx_b = np.array([i for i in idx_by_group.get(group_b, []) if labels[i] == c])
            if len(idx_a) < 5 or len(idx_b) < 5:
                result[CLASS_NAMES[c]] = None
                continue
            result[CLASS_NAMES[c]] = unpaired_bootstrap_difference(
                labels[idx_a],
                preds[idx_a],
                labels[idx_b],
                preds[idx_b],
                lambda yt, yp, c=c: float((yp[yt == c] == c).mean()),
            )
        return result

    clear_rainy = within_class(weather_idx, "clear", "rainy")
    day_night = within_class(tod_idx, "daytime", "night")

    print("\n=== Within-class comparisons (unpaired bootstrap) ===")
    print("Clear → rainy (Δ = rainy − clear):")
    for c, d in clear_rainy.items():
        if d is None or d["difference"] is None:
            print(f"  {c}: insufficient data")
            continue
        flags = []
        if d.get("excludes_zero"):
            flags.append("⭐")
        if not d.get("class_specific_n_adequate", True):
            flags.append("⚠️ small n")
        print(
            f"  {c}: {d['point_a']:.3f} → {d['point_b']:.3f} "
            f"(Δ={d['difference']:+.3f}, CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
            f"n_a={d['n_a']}, n_b={d['n_b']}) {' '.join(flags)}"
        )

    print("\nDaytime → night (Δ = night − daytime):")
    for c, d in day_night.items():
        if d is None or d["difference"] is None:
            print(f"  {c}: insufficient data")
            continue
        flags = []
        if d.get("excludes_zero"):
            flags.append("⭐")
        if not d.get("class_specific_n_adequate", True):
            flags.append("⚠️ small n")
        print(
            f"  {c}: {d['point_a']:.3f} → {d['point_b']:.3f} "
            f"(Δ={d['difference']:+.3f}, CI [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}], "
            f"n_a={d['n_a']}, n_b={d['n_b']}) {' '.join(flags)}"
        )

    # Cross-tabs
    cross_weather = defaultdict(Counter)
    cross_tod = defaultdict(Counter)
    for _i, n in enumerate(clean):
        cross_weather[meta[n]["weather"]][meta[n]["scene"]] += 1
        cross_tod[meta[n]["timeofday"]][meta[n]["scene"]] += 1

    # Save JSON
    out = {
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "data_integrity": {
            "clean_manifest_count": len(clean),
            "excluded_manifest_count": len(excluded),
            "full_count": len(full_test),
            "missing_images": 0,
            "missing_metadata": 0,
        },
        "pipeline": "(a) native JPEG, Resize(256)->CenterCrop(224)->ImageNet norm",
        "reference_check": reference_check,
        "overall_clean": overall,
        "by_weather": by_weather,
        "by_timeofday": by_tod,
        "within_class_differences": {
            "clear_vs_rainy": clear_rainy,
            "daytime_vs_night": day_night,
        },
        "cross_tab_weather": {k: dict(v) for k, v in cross_weather.items()},
        "cross_tab_timeofday": {k: dict(v) for k, v in cross_tod.items()},
    }
    (_OUT_DIR / "report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {_OUT_DIR / 'report.json'}")

    write_report(out, _DOC_PATH)
    print(f"Saved report: {_DOC_PATH}")


if __name__ == "__main__":
    sys.exit(main())
