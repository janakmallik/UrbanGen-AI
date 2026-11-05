"""Audit the probe result: where does B's improvement come from?

Four checks:
    1. Re-run probe with three class_weight settings to see if B's
       improvement is stable or regularization-driven.
    2. Feature-space statistics: is synthetic residential distinct from
       real residential? Memorized? Diverse?
    3. Feature-space statistics: is synthetic residential actually closer
       to real residential than to real non-residential?
    4. 5-fold CV variance on residential F1.

Run after the probe:
    uv run python scripts/audit_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]
_FEATURES_CACHE = _REPO_ROOT / "outputs" / "g1c_features"
_OUT_DIR = _REPO_ROOT / "outputs" / "g1c_probe"


def load_cached_features(name: str) -> torch.Tensor:
    return torch.load(_FEATURES_CACHE / f"{name}.pt", map_location="cpu", weights_only=False)


def probe_with_config(x_train, y_train, x_val, y_val, class_weight):
    clf = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=SEED, n_jobs=-1)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)
    p, r, f, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1, 2], zero_division=0)
    return {
        "residential": {"precision": float(p[2]), "recall": float(r[2]), "f1": float(f[2])},
        "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
    }


def check_1_class_weight_sensitivity(x_train, y_train, x_val, y_val, x_synth, y_synth):
    """Re-run A and B with three class_weight settings."""
    print("\n=== Check 1: class_weight sensitivity ===")
    x_b = np.concatenate([x_train, x_synth], axis=0)
    y_b = np.concatenate([y_train, y_synth], axis=0)

    configs = {
        "balanced": "balanced",
        "none": None,
        "residential_boost_2": {0: 1.0, 1: 1.0, 2: 2.0},
        "residential_boost_4": {0: 1.0, 1: 1.0, 2: 4.0},
    }
    results = {}
    for name, cw in configs.items():
        print(f"\n  class_weight = {name}")
        a = probe_with_config(x_train, y_train, x_val, y_val, cw)
        b = probe_with_config(x_b, y_b, x_val, y_val, cw)
        delta_f1 = b["residential"]["f1"] - a["residential"]["f1"]
        delta_p = b["residential"]["precision"] - a["residential"]["precision"]
        delta_r = b["residential"]["recall"] - a["residential"]["recall"]
        print(
            f"    A: P={a['residential']['precision']:.4f} "
            f"R={a['residential']['recall']:.4f} F1={a['residential']['f1']:.4f}"
        )
        print(
            f"    B: P={b['residential']['precision']:.4f} "
            f"R={b['residential']['recall']:.4f} F1={b['residential']['f1']:.4f}"
        )
        print(f"    B-A: dP={delta_p:+.4f} dR={delta_r:+.4f} dF1={delta_f1:+.4f}")
        results[name] = {
            "A": a,
            "B": b,
            "delta": {"precision": delta_p, "recall": delta_r, "f1": delta_f1},
        }
    return results


def check_2_feature_space_stats(x_train, y_train, x_val, y_val, x_synth):
    """How distinct is synthetic residential from real residential?"""
    print("\n=== Check 2: synthetic residential feature-space statistics ===")
    # Normalize for cosine distance
    x_real_res = x_train[y_train == 2]
    x_real_non = x_train[y_train != 2]
    x_synth_n = torch.from_numpy(x_synth).float()
    x_real_res_t = torch.from_numpy(x_real_res).float()
    x_real_non_t = torch.from_numpy(x_real_non).float()

    # Normalize to unit length
    x_synth_n = x_synth_n / x_synth_n.norm(dim=1, keepdim=True).clamp(min=1e-8)
    x_real_res_n = x_real_res_t / x_real_res_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    x_real_non_n = x_real_non_t / x_real_non_t.norm(dim=1, keepdim=True).clamp(min=1e-8)

    # Synthetic -> nearest real residential
    sim_to_real = x_synth_n @ x_real_res_n.T  # (N_synth, N_real_res)
    max_to_real = sim_to_real.max(dim=1).values

    # Synthetic -> nearest real non-residential
    sim_to_non = x_synth_n @ x_real_non_n.T
    max_to_non = sim_to_non.max(dim=1).values

    # Synthetic diversity: pairwise cosine sim, sampled subset for speed
    n_sample = 300
    idx = torch.randperm(x_synth_n.size(0))[:n_sample]
    synth_sample = x_synth_n[idx]
    pairwise = synth_sample @ synth_sample.T
    # Exclude diagonal
    off = pairwise[~torch.eye(n_sample, dtype=torch.bool)]
    pairwise_dist = 1.0 - off

    print("  Synthetic -> nearest real residential cosine:")
    print(
        f"    mean={max_to_real.mean():.4f}, median={max_to_real.median():.4f}, "
        f"max={max_to_real.max():.4f}"
    )
    print(f"    >= 0.95: {(max_to_real >= 0.95).sum().item()}/{len(max_to_real)}")
    print(f"    >= 0.90: {(max_to_real >= 0.90).sum().item()}/{len(max_to_real)}")
    print("  Synthetic -> nearest real NON-residential cosine:")
    print(f"    mean={max_to_non.mean():.4f}, median={max_to_non.median():.4f}")
    print(f"  Synthetic diversity (pairwise cosine dist, n={n_sample}):")
    print(f"    mean={pairwise_dist.mean():.4f}, std={pairwise_dist.std():.4f}")

    return {
        "to_real_residential": {
            "mean": float(max_to_real.mean()),
            "median": float(max_to_real.median()),
            "max": float(max_to_real.max()),
            "n_above_0_95": int((max_to_real >= 0.95).sum()),
            "n_above_0_90": int((max_to_real >= 0.90).sum()),
        },
        "to_real_non_residential": {
            "mean": float(max_to_non.mean()),
            "median": float(max_to_non.median()),
        },
        "diversity": {
            "n_sampled": n_sample,
            "mean_pairwise_distance": float(pairwise_dist.mean()),
            "std_pairwise_distance": float(pairwise_dist.std()),
        },
    }


def check_3_5fold_cv(x_train, y_train, x_synth, y_synth):
    """5-fold CV on train to get variance on A vs B deltas."""
    print("\n=== Check 3: 5-fold CV on train (A vs B) ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    a_f1s, b_f1s = [], []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(x_train, y_train)):
        # Arm A: train on fold's train
        clf_a = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
        clf_a.fit(x_train[tr_idx], y_train[tr_idx])
        y_pred_a = clf_a.predict(x_train[va_idx])
        _, _, f_a, _ = precision_recall_fscore_support(
            y_train[va_idx], y_pred_a, labels=[0, 1, 2], zero_division=0
        )
        a_f1s.append(f_a[2])

        # Arm B: same training fold PLUS all synthetic
        x_tr_b = np.concatenate([x_train[tr_idx], x_synth], axis=0)
        y_tr_b = np.concatenate([y_train[tr_idx], y_synth], axis=0)
        clf_b = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED, n_jobs=-1
        )
        clf_b.fit(x_tr_b, y_tr_b)
        y_pred_b = clf_b.predict(x_train[va_idx])
        _, _, f_b, _ = precision_recall_fscore_support(
            y_train[va_idx], y_pred_b, labels=[0, 1, 2], zero_division=0
        )
        b_f1s.append(f_b[2])
        print(
            f"  fold {fold + 1}: A_f1={f_a[2]:.4f}  B_f1={f_b[2]:.4f}  delta={f_b[2] - f_a[2]:+.4f}"
        )

    a_f1s = np.array(a_f1s)
    b_f1s = np.array(b_f1s)
    deltas = b_f1s - a_f1s
    print(f"\n  A: mean={a_f1s.mean():.4f}, std={a_f1s.std():.4f}")
    print(f"  B: mean={b_f1s.mean():.4f}, std={b_f1s.std():.4f}")
    print(
        f"  B-A: mean={deltas.mean():+.4f}, std={deltas.std():.4f}, "
        f"positive folds: {(deltas > 0).sum()}/5"
    )
    return {
        "A_f1_mean": float(a_f1s.mean()),
        "A_f1_std": float(a_f1s.std()),
        "B_f1_mean": float(b_f1s.mean()),
        "B_f1_std": float(b_f1s.std()),
        "delta_mean": float(deltas.mean()),
        "delta_std": float(deltas.std()),
        "delta_per_fold": deltas.tolist(),
        "positive_folds": int((deltas > 0).sum()),
    }


def main():
    print("Loading cached features...")
    x_train = load_cached_features("train_features").numpy()
    x_val = load_cached_features("val_features").numpy()
    x_synth = load_cached_features("synthetic_features").numpy()

    # Need labels — reconstruct from cache
    from urbangen.data.bdd100k_generative import BDD100K128Dataset

    train_ds = BDD100K128Dataset(_REPO_ROOT / "data" / "bdd100k_128", "train")
    val_ds = BDD100K128Dataset(_REPO_ROOT / "data" / "bdd100k_128", "val")
    y_train = np.array([train_ds[i][1] for i in range(len(train_ds))], dtype=np.int64)
    y_val = np.array([val_ds[i][1] for i in range(len(val_ds))], dtype=np.int64)
    y_synth = np.full(len(x_synth), 2, dtype=np.int64)

    results = {}
    results["check_1_class_weight"] = check_1_class_weight_sensitivity(
        x_train, y_train, x_val, y_val, x_synth, y_synth
    )
    results["check_2_feature_space"] = check_2_feature_space_stats(
        x_train, y_train, x_val, y_val, x_synth
    )
    results["check_3_5fold_cv"] = check_3_5fold_cv(x_train, y_train, x_synth, y_synth)

    print("\n=== FINAL VERDICT ===")
    # A: class-weight sensitivity
    deltas = [
        results["check_1_class_weight"][k]["delta"]["f1"]
        for k in ["balanced", "none", "residential_boost_2", "residential_boost_4"]
    ]
    stable_positive = all(d > 0 for d in deltas)
    cv = results["check_3_5fold_cv"]
    cv_stable_positive = cv["delta_mean"] > 0 and cv["positive_folds"] >= 4
    fs = results["check_2_feature_space"]
    not_memorized = fs["to_real_residential"]["n_above_0_95"] < 30  # < 1% of 3000
    diverse = fs["diversity"]["mean_pairwise_distance"] > 0.05

    if stable_positive and cv_stable_positive and not_memorized and diverse:
        verdict = (
            "SIGNAL CONFIRMED: B consistently beats A across class_weight settings, "
            "wins in >=4/5 CV folds, and synthetics are neither memorized nor "
            "collapsed. Run the fine-tune with a recall AND F1 criterion."
        )
    elif not not_memorized or not diverse:
        verdict = (
            "SYNTHETIC QUALITY PROBLEM: synthetics are either memorized "
            f"({fs['to_real_residential']['n_above_0_95']} samples above cos 0.95) or "
            f"collapsed (diversity {fs['diversity']['mean_pairwise_distance']:.4f}). "
            "B's improvement is a class-prior artifact. Close G1c."
        )
    elif not stable_positive:
        verdict = (
            "REGULARIZATION ARTIFACT: B's improvement depends on class_weight. "
            "Not robust to hyperparameter choice. Close G1c."
        )
    else:
        verdict = "WEAK SIGNAL: B beats A but not robustly. Document and close G1c."
    print(f"  {verdict}")

    out = {"results": results, "verdict": verdict}
    (_OUT_DIR / "audit_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {_OUT_DIR / 'audit_results.json'}")


if __name__ == "__main__":
    sys.exit(main())
