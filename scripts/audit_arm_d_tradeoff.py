"""Audit Arm D-M2 daytime regression using paired bootstrap on identical images."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, recall_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
_OUT_DIR = _REPO_ROOT / "outputs" / "week5_arms"


def paired_bootstrap(labels, preds_a, preds_b, metric_fn, n=10000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(labels), len(labels))
        va = metric_fn(labels[idx], preds_a[idx])
        vb = metric_fn(labels[idx], preds_b[idx])
        diffs[i] = vb - va
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def class_recall(y_true, y_pred, cls):
    return recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]


def main():
    c = json.loads((_OUT_DIR / "daytime_preds_C_seed42.json").read_text())
    d = json.loads((_OUT_DIR / "daytime_preds_D_seed42.json").read_text())

    # 1. Verify sample identity
    if c["day"]["names"] != d["day"]["names"]:
        raise RuntimeError("Daytime sample identity mismatch between Arm C and Arm D")

    labels = np.array(c["day"]["labels"])
    preds_c = np.array(c["day"]["predictions"])
    preds_d = np.array(d["day"]["predictions"])

    print(f"Verified {len(labels)} identical daytime images for both arms.\n")

    # 2. Descriptive stats
    print("--- DAYTIME PER-CLASS F1 (Arm D vs Arm C) ---")
    classes = ["city", "highway", "residential"]
    for i, cls in enumerate(classes):
        c_f1 = f1_score(labels, preds_c, labels=[i], average=None, zero_division=0)[0]
        d_f1 = f1_score(labels, preds_d, labels=[i], average=None, zero_division=0)[0]
        print(f"  {cls:14s} | C: {c_f1:.4f} | D: {d_f1:.4f} | Delta: {d_f1 - c_f1:+.4f}")

    # 3. Paired Bootstrap on Daytime Macro-F1
    print("\n--- PAIRED BOOTSTRAP (Daytime Macro-F1) ---")
    mean, lo, hi = paired_bootstrap(labels, preds_c, preds_d, macro_f1)
    print(f"  Delta (D - C): {mean:+.4f}  95% CI: [{lo:+.4f}, {hi:+.4f}]")

    # 4. Paired Bootstrap on Daytime City Recall
    print("\n--- PAIRED BOOTSTRAP (Daytime City Recall) ---")
    mean, lo, hi = paired_bootstrap(labels, preds_c, preds_d, lambda y, p: class_recall(y, p, 0))
    print(f"  Delta (D - C): {mean:+.4f}  95% CI: [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
