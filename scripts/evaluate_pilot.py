"""Evaluate the four G1c pilot arms on the clean test set.

Loads the 4 checkpoints, evaluates each on the clean test split (4,745
images, 591 residential), and computes:

    - Per-arm residential recall, residential F1, macro-F1, accuracy
    - Paired bootstrap CI on residential recall difference:
        B - A (vs real-only baseline)
        B - C (vs conventional augmentation)
        B - D (vs class-weighted loss)
    - Holm correction for the two comparisons (B-C, B-D) that decide
      the success criterion

Success criterion (frozen):
    Arm B beats BOTH arm C and arm D on residential recall with a
    paired-bootstrap 95% CI excluding zero, after Holm correction for
    the 2 comparisons, AND macro-F1 does not degrade by more than 0.01
    vs arm A.

Run after all 4 arm training scripts complete:
    uv run python scripts/evaluate_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from train_pilot_arm import (  # noqa: E402
    _CACHE_ROOT,
    _PILOT_CKPT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    PilotResNetDataset,
)
from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402

SEED = 42
N_BOOTSTRAP = 1000
_OUT_DIR = _REPO_ROOT / "outputs" / "g1c_pilot"


def load_model(ckpt_path: Path, device) -> models.ResNet:
    model = models.resnet50(num_classes=NUM_CLASSES)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(
        f"Loaded {ckpt_path.name}: arm={ckpt['arm']}, epoch={ckpt['epoch']}, "
        f"val_macro_f1={ckpt['best_macro_f1']:.3f}"
    )
    return model


@torch.no_grad()
def get_predictions(model, test_loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob_positive_class) — prob is residential class."""
    all_labels, all_res_probs = [], []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 2]  # residential probability
        all_labels.append(y.numpy())
        all_res_probs.append(probs.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_res_probs)


def metrics_at_threshold(y_true, y_prob, threshold=0.5) -> dict:
    """Standard classification metrics at threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    per_class_recall = {}
    for c in range(NUM_CLASSES):
        per_class_recall[CLASS_NAMES[c]] = float(
            recall_score(y_true == c, y_pred == c, zero_division=0)
        )
    # But multi-class recall needs argmax of softmax, not threshold.
    # Here we're only using binary residential probability to compute
    # residential recall. That's a binary evaluation for the pilot.
    return {
        "residential_recall_at_threshold": float(
            recall_score(y_true == 2, y_pred == 1, zero_division=0)
        ),
        "residential_precision_at_threshold": float(
            (y_pred & (y_true == 2)).sum() / max(y_pred.sum(), 1)
        ),
        "n_predicted_positive": int(y_pred.sum()),
        "n_true_positive": int((y_true == 2).sum()),
    }


@torch.no_grad()
def get_multiclass_predictions(model, test_loader, device):
    """Return argmax predictions for multiclass metrics."""
    all_labels, all_preds = [], []
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        all_labels.append(y.numpy())
        all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_preds)


def multiclass_metrics(y_true, y_pred) -> dict:
    per_class_recall = {}
    for c in range(NUM_CLASSES):
        per_class_recall[CLASS_NAMES[c]] = float(
            recall_score(y_true == c, y_pred == c, zero_division=0)
        )
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_recall": per_class_recall,
    }


def paired_bootstrap_recall_diff(
    y_true_a: np.ndarray,
    y_pred_a: np.ndarray,
    y_true_b: np.ndarray,
    y_pred_b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Paired bootstrap on residential recall: arm_b - arm_a.

    y_true are the ground-truth labels (identical across arms),
    y_pred are the binary residential predictions.
    """
    rng = np.random.default_rng(SEED)
    n = len(y_true_a)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        ta, pa = y_true_a[idx], y_pred_a[idx]
        tb, pb = y_true_b[idx], y_pred_b[idx]
        ra = ((ta == 2) & (pa == 2)).sum() / max((ta == 2).sum(), 1)
        rb = ((tb == 2) & (pb == 2)).sum() / max((tb == 2).sum(), 1)
        diffs[i] = rb - ra
    return {
        "mean": float(diffs.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "n_bootstrap": n_bootstrap,
    }


def holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    # Enforce monotonicity
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    return adjusted


def ci_to_p_value_approx(ci_low: float, ci_high: float) -> float:
    """Approximate two-sided p-value from 95% CI.

    If CI excludes zero, p < 0.05. A crude approximation: linear
    interpolation between CI bounds and 0.
    """
    if ci_low > 0 or ci_high < 0:
        # CI excludes zero. Use the closest bound to zero to estimate p.
        width = ci_high - ci_low
        # 95% CI spans 1.96 * SE * 2 = 3.92 * SE
        # Mean - 0 is the effect. p-value depends on effect/SE.
        mean = (ci_low + ci_high) / 2
        if width > 0:
            se = width / 3.92
            z = abs(mean / se) if se > 0 else 0
            # Rough normal CDF approximation
            from math import erf, sqrt

            p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
            return max(p, 1e-6)
        return 0.01
    else:
        # CI includes zero -> p >= 0.05
        return 0.5


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Verify all 4 checkpoints exist
    ckpt_paths = {arm: _PILOT_CKPT_DIR / f"pilot_arm_{arm}.pt" for arm in "ABCD"}
    missing = [a for a, p in ckpt_paths.items() if not p.exists()]
    if missing:
        print(f"ERROR: missing checkpoints for arms {missing}. Run train_pilot_arm.py.")
        return 1

    # Load test data
    print("\nLoading clean test set...")
    test_ds = BDD100K128Dataset(_CACHE_ROOT, "test", allow_test=True)
    test_ds.training_flag = False
    test_idx = list(range(len(test_ds)))
    test_wrapped = PilotResNetDataset(test_ds, test_idx, mode="real")
    test_loader = DataLoader(test_wrapped, batch_size=64, shuffle=False, num_workers=0)
    print(f"  test set: {len(test_ds):,} images")
    y_true_all = np.array([test_ds[i][1] for i in range(len(test_ds))])

    # Evaluate each arm
    results = {}
    for arm in "ABCD":
        print(f"\nEvaluating arm {arm}...")
        model = load_model(ckpt_paths[arm], device)
        y_true, y_pred = get_multiclass_predictions(model, test_loader, device)
        multiclass = multiclass_metrics(y_true, y_pred)
        binary_res = (y_pred == 2).astype(int)
        binary_true = (y_true == 2).astype(int)
        res_recall = float((binary_res & binary_true).sum() / max(binary_true.sum(), 1))
        results[arm] = {
            **multiclass,
            "residential_recall_binary": res_recall,
            "_y_true": y_true,
            "_y_pred": y_pred,
        }
        print(f"  accuracy: {multiclass['accuracy']:.3f}")
        print(f"  macro-F1: {multiclass['macro_f1']:.3f}")
        print(f"  per-class recall: {multiclass['per_class_recall']}")
        print(f"  residential recall: {res_recall:.3f}")

    # Paired bootstrap on residential recall: B-A, B-C, B-D
    print("\n=== Paired bootstrap on residential recall (1000 resamples) ===")
    comparisons = {}
    for other in ["A", "C", "D"]:
        bs = paired_bootstrap_recall_diff(
            results["B"]["_y_true"],
            results["B"]["_y_pred"],
            results[other]["_y_true"],
            results[other]["_y_pred"],
        )
        comparisons[f"B_vs_{other}"] = bs
        print(
            f"  B - {other}: mean={bs['mean']:+.4f}, "
            f"95% CI [{bs['ci_low']:+.4f}, {bs['ci_high']:+.4f}]"
        )

    # Holm correction for B-C and B-D (the success criterion)
    p_bc = ci_to_p_value_approx(comparisons["B_vs_C"]["ci_low"], comparisons["B_vs_C"]["ci_high"])
    p_bd = ci_to_p_value_approx(comparisons["B_vs_D"]["ci_low"], comparisons["B_vs_D"]["ci_high"])
    adjusted = holm_correction([p_bc, p_bd])
    print(f"\n  Unadjusted p (B-C): {p_bc:.4f}, (B-D): {p_bd:.4f}")
    print(f"  Holm-adjusted p (B-C): {adjusted[0]:.4f}, (B-D): {adjusted[1]:.4f}")

    # Success criterion
    b_beats_c = comparisons["B_vs_C"]["ci_low"] > 0 and adjusted[0] < 0.05
    b_beats_d = comparisons["B_vs_D"]["ci_low"] > 0 and adjusted[1] < 0.05
    macro_f1_ok = results["B"]["macro_f1"] >= results["A"]["macro_f1"] - 0.01
    success = b_beats_c and b_beats_d and macro_f1_ok

    print("\n=== SUCCESS CRITERION ===")
    print(f"  B beats C (CI excludes zero, Holm p < 0.05): {b_beats_c}")
    print(f"  B beats D (CI excludes zero, Holm p < 0.05): {b_beats_d}")
    print(f"  Macro-F1 guardrail (B >= A - 0.01): {macro_f1_ok}")
    print(f"  OVERALL: {'PASS' if success else 'FAIL'}")

    # Save results
    out = {
        "test_set_size": int(len(test_ds)),
        "test_set_residential": int((y_true_all == 2).sum()),
        "per_arm": {
            arm: {
                "accuracy": results[arm]["accuracy"],
                "macro_f1": results[arm]["macro_f1"],
                "per_class_recall": results[arm]["per_class_recall"],
                "residential_recall_binary": results[arm]["residential_recall_binary"],
            }
            for arm in "ABCD"
        },
        "comparisons": comparisons,
        "holm_adjusted_p": {"B_vs_C": adjusted[0], "B_vs_D": adjusted[1]},
        "success_criterion": {
            "b_beats_c": bool(b_beats_c),
            "b_beats_d": bool(b_beats_d),
            "macro_f1_guardrail_ok": bool(macro_f1_ok),
            "overall_pass": bool(success),
        },
    }
    out_path = _OUT_DIR / "pilot_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
