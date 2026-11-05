"""Step 0 — robustness report card for the Week 5 ResNet-50.

Evaluates the existing Week 5 ResNet-50 on real BDD100K test-split
images, broken down by weather and timeofday attributes. Reports
per-subgroup F1, per-class recall, bootstrap CIs, and cross-tabs to
check confounding between weather and scene class.

This is a diagnostic artifact, not an experiment. It answers:
    - Does adverse weather correlate with degraded classification?
    - Is any weather gap confounded by scene class distribution?
    - Are the sample sizes large enough to support claims?

Uses:
    - Real Week 5 ResNet-50 checkpoint (bdd_resnet50_best.pt)
    - Clean test split (resnet_test_clean.json, 4,745 images)
    - BDD100K label JSONs (weather, timeofday attributes)
    - The existing 128x128 cache

Outputs:
    - outputs/robustness_report/report.json  (machine-readable)
    - docs/robustness_report.md              (human-readable)

Run:
    uv run python scripts/robustness_report.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models

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

_URBANMIND = urbanmind_root()
_WEEK5_CKPT = _URBANMIND / "bdd_resnet50_best.pt"
_LABEL_DIR = data_root() / "labels"
_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_OUT_DIR = _REPO_ROOT / "outputs" / "robustness_report"
_DOC_PATH = _REPO_ROOT / "docs" / "robustness_report.md"


def load_week5_model(device) -> torch.nn.Module:
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


def load_metadata() -> dict[str, dict]:
    """Return {filename: {"weather": ..., "timeofday": ...}} for all 3-class val images."""
    target = {"city street", "highway", "residential"}
    meta: dict[str, dict] = {}
    val_json = _LABEL_DIR / "bdd100k_labels_images_val.json"
    data = json.loads(val_json.read_text(encoding="utf-8"))
    for rec in data:
        attrs = rec.get("attributes", {})
        if attrs.get("scene") in target:
            meta[rec["name"]] = {
                "weather": attrs.get("weather", "unknown"),
                "timeofday": attrs.get("timeofday", "unknown"),
                "scene": attrs["scene"],
            }
    return meta


def load_test_names() -> list[str]:
    """Load clean test split filenames from UrbanMind."""
    path = _URBANMIND / "data" / "splits" / "resnet_test_clean.json"
    return json.loads(path.read_text(encoding="utf-8"))


@torch.inference_mode()
def predict_all(model, test_ds: BDD100K128Dataset, device) -> tuple[list[int], list[int]]:
    """Return (preds, labels) for the full test set in cache order."""
    loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        # Content crop + 224 resize + ImageNet norm (matches eval pipeline)
        content = x[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
        resized = torch.nn.functional.interpolate(
            content, size=(224, 224), mode="bilinear", align_corners=False
        )
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        logits = model(normalized)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(y.tolist())
    return all_preds, all_labels


def build_test_metadata_index(test_names: list[str], meta: dict) -> dict[int, dict]:
    """Map test_ds index -> metadata. Uses cache manifest filenames order.

    The cache was built from resnet_test_clean.json in the same order.
    Verify by count and reconstruct mapping.
    """
    test_ds = BDD100K128Dataset(_CACHE_ROOT, "test", allow_test=True)
    if len(test_ds) != len(test_names):
        raise ValueError(
            f"Test cache has {len(test_ds)} images but test_names has "
            f"{len(test_names)}. Cache and split manifest disagree."
        )
    idx_to_meta = {}
    for i, name in enumerate(test_names):
        if name in meta:
            idx_to_meta[i] = meta[name]
        else:
            idx_to_meta[i] = {"weather": "unknown", "timeofday": "unknown", "scene": "unknown"}
    return idx_to_meta


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_boot: int = N_BOOTSTRAP,
) -> dict:
    """Bootstrap 95% CI on a metric."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    if n == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    point = float(metric_fn(y_true, y_pred))
    if n < 5:
        return {"mean": point, "ci_low": None, "ci_high": None, "n": n}
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = metric_fn(y_true[idx], y_pred[idx])
    return {
        "mean": point,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "n": n,
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the 3 classes present in y_true ∪ y_pred."""
    from sklearn.metrics import f1_score

    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    if not labels:
        return 0.0
    return f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {}
    for c in range(NUM_CLASSES):
        mask = y_true == c
        if mask.sum() == 0:
            out[CLASS_NAMES[c]] = None
        else:
            out[CLASS_NAMES[c]] = float((y_pred[mask] == c).mean())
    return out


def evaluate_subgroup(
    name: str, indices: np.ndarray, preds: np.ndarray, labels: np.ndarray
) -> dict:
    """Compute metrics for a subset of indices."""
    if len(indices) == 0:
        return {"name": name, "n": 0, "metrics": None}
    sub_preds = preds[indices]
    sub_labels = labels[indices]
    return {
        "name": name,
        "n": int(len(indices)),
        "metrics": {
            "macro_f1": bootstrap_ci(sub_labels, sub_preds, macro_f1),
            "per_class_recall": {
                c: bootstrap_ci(
                    sub_labels,
                    sub_preds,
                    lambda yt, yp, c=c: (
                        float((yp[yt == c] == c).mean()) if (yt == c).sum() > 0 else 0.0
                    ),
                )
                for c in range(NUM_CLASSES)
            },
        },
    }


def write_markdown_report(report: dict, path: Path) -> None:
    lines = []
    lines.append("# Week 5 ResNet-50 — Robustness Report Card\n")
    lines.append(
        "Evaluates UrbanMind-AI's Week 5 ResNet-50 on the clean BDD100K "
        "test split, broken down by weather and time-of-day attributes "
        "from the BDD100K label JSONs.\n"
    )
    lines.append(f"- **Total test images:** {report['n_total']}")
    lines.append(f"- **Bootstrap resamples:** {report['n_bootstrap']}")
    lines.append(f"- **Random seed:** {report['seed']}\n")

    lines.append("## Overall metrics\n")
    overall = report["overall"]
    lines.append(
        f"- **Macro-F1:** {overall['macro_f1']['mean']:.4f} "
        f"[{overall['macro_f1']['ci_low']:.4f}, {overall['macro_f1']['ci_high']:.4f}]"
    )
    for c, m in overall["per_class_recall"].items():
        if m["mean"] is not None:
            lines.append(
                f"- **{c} recall:** {m['mean']:.4f} "
                f"[{m['ci_low']:.4f}, {m['ci_high']:.4f}]  (n={m['n']})"
            )
    lines.append("")

    lines.append("## By weather\n")
    lines.append(
        "| Weather | n | Macro-F1 [95% CI] | City recall | Highway recall | Residential recall |"
    )
    lines.append("|---|---|---|---|---|---|")
    for group in report["by_weather"]:
        m = group["metrics"]
        if m is None:
            lines.append(f"| {group['name']} | 0 | — | — | — | — |")
            continue
        f1 = m["macro_f1"]
        f1_str = f"{f1['mean']:.3f} [{f1['ci_low']:.3f}, {f1['ci_high']:.3f}]"
        recalls = []
        for c in range(NUM_CLASSES):
            r = m["per_class_recall"][c]
            recalls.append(f"{r['mean']:.3f} (n={r['n']})" if r["mean"] is not None else "—")
        lines.append(f"| {group['name']} | {group['n']} | {f1_str} | {' | '.join(recalls)} |")
    lines.append("")

    lines.append("## By time of day\n")
    lines.append(
        "| Time | n | Macro-F1 [95% CI] | City recall | Highway recall | Residential recall |"
    )
    lines.append("|---|---|---|---|---|---|")
    for group in report["by_timeofday"]:
        m = group["metrics"]
        if m is None:
            lines.append(f"| {group['name']} | 0 | — | — | — | — |")
            continue
        f1 = m["macro_f1"]
        f1_str = f"{f1['mean']:.3f} [{f1['ci_low']:.3f}, {f1['ci_high']:.3f}]"
        recalls = []
        for c in range(NUM_CLASSES):
            r = m["per_class_recall"][c]
            recalls.append(f"{r['mean']:.3f} (n={r['n']})" if r["mean"] is not None else "—")
        lines.append(f"| {group['name']} | {group['n']} | {f1_str} | {' | '.join(recalls)} |")
    lines.append("")

    lines.append("## Cross-tab: weather × scene class\n")
    lines.append(
        "Sample counts per (weather, scene class) cell. Small cells produce unstable metrics.\n"
    )
    lines.append("| Weather | " + " | ".join(CLASS_NAMES) + " | Total |")
    lines.append("|---" * (len(CLASS_NAMES) + 2) + "|")
    for weather, counts in report["cross_tab_weather_scene"].items():
        row = [str(counts.get(c, 0)) for c in CLASS_NAMES]
        total = sum(counts.values())
        lines.append(f"| {weather} | {' | '.join(row)} | {total} |")
    lines.append("")

    lines.append("## Cross-tab: time-of-day × scene class\n")
    lines.append("| Time | " + " | ".join(CLASS_NAMES) + " | Total |")
    lines.append("|---" * (len(CLASS_NAMES) + 2) + "|")
    for tod, counts in report["cross_tab_timeofday_scene"].items():
        row = [str(counts.get(c, 0)) for c in CLASS_NAMES]
        total = sum(counts.values())
        lines.append(f"| {tod} | {' | '.join(row)} | {total} |")
    lines.append("")

    lines.append("## Interpretation guide\n")
    lines.append(
        "- **Weather or time subgroups with n < 30** should be treated as "
        "qualitative only; the bootstrap CIs will be very wide.\n"
        "- **Confounding:** if a subgroup's scene-class distribution differs "
        "significantly from the overall distribution, per-subgroup F1 "
        "differences partly reflect class mix, not weather. Check the cross-tabs.\n"
        "- **CI overlap:** if adverse-weather F1 CI overlaps the clear-weather "
        "F1 CI, there is no statistically-supported weather gap.\n"
        "- **Within-class analysis:** the per-class recall columns show "
        "whether any gap is uniform across classes or concentrated in one.\n"
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _DOC_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading Week 5 ResNet-50...")
    model = load_week5_model(device)

    print("Loading metadata (weather, timeofday)...")
    meta = load_metadata()
    print(f"  {len(meta):,} images with metadata")

    print("Loading test split names...")
    test_names = load_test_names()
    print(f"  {len(test_names):,} test filenames")

    print("Building test metadata index...")
    idx_to_meta = build_test_metadata_index(test_names, meta)

    print("Loading test set and predicting...")
    test_ds = BDD100K128Dataset(_CACHE_ROOT, "test", allow_test=True)
    preds, labels = predict_all(model, test_ds, device)
    preds = np.array(preds)
    labels = np.array(labels)

    # Overall
    print("\n=== Overall metrics ===")
    overall_macro_f1 = bootstrap_ci(labels, preds, macro_f1)
    overall_per_class = {
        c: bootstrap_ci(
            labels,
            preds,
            lambda yt, yp, c=c: float((yp[yt == c] == c).mean()) if (yt == c).sum() > 0 else 0.0,
        )
        for c in range(NUM_CLASSES)
    }
    overall = {
        "macro_f1": overall_macro_f1,
        "per_class_recall": {CLASS_NAMES[c]: overall_per_class[c] for c in range(3)},
    }
    print(
        f"  macro-F1: {overall_macro_f1['mean']:.4f} "
        f"[{overall_macro_f1['ci_low']:.4f}, {overall_macro_f1['ci_high']:.4f}]"
    )
    for c, m in overall["per_class_recall"].items():
        print(f"  {c} recall: {m['mean']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]")

    # By weather
    print("\n=== By weather ===")
    weather_to_indices = defaultdict(list)
    for i, m in idx_to_meta.items():
        weather_to_indices[m["weather"]].append(i)
    by_weather = []
    for weather in sorted(weather_to_indices.keys()):
        indices = np.array(weather_to_indices[weather])
        sub = evaluate_subgroup(weather, indices, preds, labels)
        by_weather.append(sub)
        if sub["metrics"]:
            mf1 = sub["metrics"]["macro_f1"]
            print(
                f"  {weather:12s}: n={sub['n']:4d}, "
                f"macro-F1={mf1['mean']:.3f} [{mf1['ci_low']:.3f}, {mf1['ci_high']:.3f}]"
            )

    # By timeofday
    print("\n=== By time of day ===")
    tod_to_indices = defaultdict(list)
    for i, m in idx_to_meta.items():
        tod_to_indices[m["timeofday"]].append(i)
    by_tod = []
    for tod in sorted(tod_to_indices.keys()):
        indices = np.array(tod_to_indices[tod])
        sub = evaluate_subgroup(tod, indices, preds, labels)
        by_tod.append(sub)
        if sub["metrics"]:
            mf1 = sub["metrics"]["macro_f1"]
            print(
                f"  {tod:12s}: n={sub['n']:4d}, "
                f"macro-F1={mf1['mean']:.3f} [{mf1['ci_low']:.3f}, {mf1['ci_high']:.3f}]"
            )

    # Cross-tabs
    print("\n=== Cross-tab: weather × scene class ===")
    cross_wx = defaultdict(lambda: Counter())
    for _i, m in idx_to_meta.items():
        cross_wx[m["weather"]][m["scene"]] += 1
    for weather in sorted(cross_wx.keys()):
        counts = cross_wx[weather]
        print(f"  {weather:12s}: {dict(counts)}")

    print("\n=== Cross-tab: timeofday × scene class ===")
    cross_to = defaultdict(lambda: Counter())
    for _i, m in idx_to_meta.items():
        cross_to[m["timeofday"]][m["scene"]] += 1
    for tod in sorted(cross_to.keys()):
        counts = cross_to[tod]
        print(f"  {tod:12s}: {dict(counts)}")

    report = {
        "n_total": int(len(labels)),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "overall": overall,
        "by_weather": by_weather,
        "by_timeofday": by_tod,
        "cross_tab_weather_scene": {k: dict(v) for k, v in cross_wx.items()},
        "cross_tab_timeofday_scene": {k: dict(v) for k, v in cross_to.items()},
    }

    out_path = _OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    print("\nWriting markdown report...")
    write_markdown_report(report, _DOC_PATH)
    print(f"Saved: {_DOC_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
