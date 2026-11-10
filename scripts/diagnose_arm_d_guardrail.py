"""Diagnose where Arm D's daytime macro-F1 drop comes from."""

import json
from pathlib import Path

import numpy as np

OUT_DIR = Path("outputs/week5_arms")


def load_preds(arm):
    p = OUT_DIR / f"results_{arm}_seed42.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return d


def main():
    c = load_preds("C")
    d = load_preds("D")

    # We need predictions on the DAY subset, not night
    # BUT: the results file only stores night predictions
    # We need to re-evaluate or extract from saved state
    print("Day metrics from results files:")
    print(f"  Arm C: macro_f1={c['day']['macro_f1']:.4f}")
    print(f"  Arm D: macro_f1={d['day']['macro_f1']:.4f}")
    print()
    print("Arm C day per-class:")
    for cls, m in c["day"]["per_class"].items():
        print(
            f"  {cls:14s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("Arm D day per-class:")
    for cls, m in d["day"]["per_class"].items():
        print(
            f"  {cls:14s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}"
        )
    print()
    print("Day per-class F1 delta (D - C):")
    for cls in c["day"]["per_class"]:
        c_f1 = c["day"]["per_class"][cls]["f1"]
        d_f1 = d["day"]["per_class"][cls]["f1"]
        print(f"  {cls:14s}  DeltaF1 = {d_f1 - c_f1:+.4f}")
    print()
    print("Day confusion matrix, Arm C:")
    print(np.array(c["day"]["confusion_matrix"]))
    print()
    print("Day confusion matrix, Arm D:")
    print(np.array(d["day"]["confusion_matrix"]))
    print()
    print("Class order: [city, highway, residential]")


if __name__ == "__main__":
    main()
