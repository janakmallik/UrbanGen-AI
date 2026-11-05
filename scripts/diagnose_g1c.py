"""Final G1c diagnostic: where does the synthetic data fail?

Four checks:
    1. Feature-space centroid distances: how far are synthetics from the
       real residential centroid vs real non-residential centroids?
    2. Nearest-neighbor visual inspection: for 8 random synthetics, save
       a PNG showing the synthetic + its 3 nearest real residential + 3
       nearest real non-residential.
    3. Real-only classifier on synthetics: train logistic regression on
       REAL train only (no synthetic), predict on synthetics, report
       per-class distribution.
    4. Summary: is the failure visual (blurry), representation (feature
       space off), or distributional (wrong mixture)?

Run:
    uv run python scripts/diagnose_g1c.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
from sklearn.linear_model import LogisticRegression

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402

SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]
_FEATURES_CACHE = _REPO_ROOT / "outputs" / "g1c_features"
_SYNTHETIC_CACHE = _REPO_ROOT / "outputs" / "g1c_synthetic_residential.pt"
_OUT_DIR = _REPO_ROOT / "outputs" / "g1c_diagnostic"


def load_features(name: str) -> np.ndarray:
    return torch.load(
        _FEATURES_CACHE / f"{name}.pt", map_location="cpu", weights_only=False
    ).numpy()


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading cached features and synthetic images...")
    x_train = load_features("train_features")
    x_synth = load_features("synthetic_features")
    train_ds = BDD100K128Dataset(_REPO_ROOT / "data" / "bdd100k_128", "train")
    y_train = np.array([train_ds[i][1] for i in range(len(train_ds))], dtype=np.int64)
    synth_imgs = torch.load(_SYNTHETIC_CACHE, map_location="cpu", weights_only=False)["images"]

    results = {}

    # === Check 1: centroid distances ===
    print("\n=== Check 1: feature-space centroid distances ===")
    centroids = {c: x_train[y_train == c].mean(axis=0) for c in range(3)}

    # Normalize centroids and synthetics for cosine
    def unit(v):
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, 1e-8)

    c_n = {c: unit(centroids[c]) for c in range(3)}
    synth_n = unit(x_synth)

    cos_to = {}
    for c in range(3):
        cos_to[CLASS_NAMES[c]] = (synth_n @ c_n[c]).tolist()
    for c in range(3):
        arr = np.array(cos_to[CLASS_NAMES[c]])
        print(
            f"  Synthetic -> {CLASS_NAMES[c]} centroid: "
            f"mean={arr.mean():.4f}, median={np.median(arr):.4f}, "
            f"p10={np.percentile(arr, 10):.4f}, p90={np.percentile(arr, 90):.4f}"
        )
    results["centroid_cosine"] = {
        c: {
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)),
            "p90": float(np.percentile(v, 90)),
        }
        for c, v in cos_to.items()
    }

    # === Check 2: nearest-neighbor visual inspection ===
    print("\n=== Check 2: nearest-neighbor visual grids ===")
    # Real residential and non-residential features
    real_res_idx = np.where(y_train == 2)[0]
    real_non_idx = np.where(y_train != 2)[0]
    x_real_res = x_train[real_res_idx]
    x_real_non = x_train[real_non_idx]
    # Normalize
    x_res_n = unit(x_real_res)
    x_non_n = unit(x_real_non)

    # Sample 8 synthetics
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(x_synth), size=8, replace=False)
    synth_sample_n = synth_n[sample_idx]

    # For each synthetic, find 3 nearest real residential + 3 nearest non-residential
    sim_res = synth_sample_n @ x_res_n.T  # (8, n_real_res)
    sim_non = synth_sample_n @ x_non_n.T
    nn_res_idx = np.argsort(-sim_res, axis=1)[:, :3]
    nn_non_idx = np.argsort(-sim_non, axis=1)[:, :3]

    # Build a grid: 8 rows (synthetics) × 7 cols (synthetic + 3 res + 3 non)
    # Load images
    grid_rows = []
    for i, sidx in enumerate(sample_idx):
        row = [synth_imgs[sidx]]
        for j in nn_res_idx[i]:
            real_idx = real_res_idx[j]
            img, _ = train_ds[int(real_idx)]
            row.append(img)
        for j in nn_non_idx[i]:
            real_idx = real_non_idx[j]
            img, _ = train_ds[int(real_idx)]
            row.append(img)
        grid_rows.append(torch.stack(row))
    grid = torch.stack(grid_rows)  # (8, 7, 3, 128, 128)
    grid = grid.reshape(-1, 3, 128, 128)
    # Zero padding
    grid[:, :, :28, :] = 0.0
    grid[:, :, 100:, :] = 0.0
    out_path = _OUT_DIR / "nn_grid.png"
    vutils.save_image(vutils.make_grid(grid, nrow=7, padding=2), str(out_path))
    print(f"  Saved 8x7 grid: {out_path}")
    print(" Layout: [synthetic | 3 nearest residential | 3 nearest non-residential]")
    print(f"  Synthetic NN cosine to residential: mean={sim_res.max(axis=1).mean():.4f}")
    print(f"  Synthetic NN cosine to non-residential: mean={sim_non.max(axis=1).mean():.4f}")
    results["nn_visual"] = {
        "n_samples": 8,
        "grid_path": str(out_path.relative_to(_REPO_ROOT)),
        "mean_nn_cos_residential": float(sim_res.max(axis=1).mean()),
        "mean_nn_cos_non_residential": float(sim_non.max(axis=1).mean()),
    }

    # === Check 3: real-only classifier on synthetics ===
    print("\n=== Check 3: real-only classifier on synthetics ===")
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )
    clf.fit(x_train, y_train)
    y_pred_synth = clf.predict(x_synth)
    pred_dist = {CLASS_NAMES[c]: int((y_pred_synth == c).sum()) for c in range(3)}
    print(" A classifier trained ONLY on real data, applied to 3000 synthetics:")
    for c_name, count in pred_dist.items():
        print(f"    classified as {c_name:<12}: {count:5d} ({100 * count / len(x_synth):.1f}%)")
    results["real_only_classifier_on_synthetics"] = pred_dist

    # === Check 4: summary ===
    print("\n=== Check 4: summary ===")
    centroid_cos_res = results["centroid_cosine"]["residential"]["mean"]
    centroid_cos_city = results["centroid_cosine"]["city street"]["mean"]
    centroid_cos_highway = results["centroid_cosine"]["highway"]["mean"]
    nn_cos_res = results["nn_visual"]["mean_nn_cos_residential"]
    nn_cos_non = results["nn_visual"]["mean_nn_cos_non_residential"]
    frac_classified_res = pred_dist["residential"] / len(x_synth)

    print(f"  Centroid cosine to residential: {centroid_cos_res:.4f}")
    print(f"  Centroid cosine to city street:  {centroid_cos_city:.4f}")
    print(f"  Centroid cosine to highway:      {centroid_cos_highway:.4f}")
    print(f"  NN cosine to residential:        {nn_cos_res:.4f}")
    print(f"  NN cosine to non-residential:    {nn_cos_non:.4f}")
    print(f"  Fraction classified residential: {frac_classified_res:.4f}")

    if frac_classified_res < 0.10:
        diagnosis = (
            "REPRESENTATION MISMATCH: real-only classifier assigns <10% of "
            "synthetics to residential. The generator produces samples that "
            "do not occupy the residential feature region."
        )
    elif frac_classified_res < 0.30:
        diagnosis = (
            "PARTIAL REPRESENTATION MISMATCH: real-only classifier assigns "
            "10-30% of synthetics to residential (real ~12% base rate). "
            "Synthetics are near-chance for the residential class."
        )
    elif frac_classified_res < 0.50:
        diagnosis = (
            "BOUNDARY CASE: 30-50% of synthetics classify as residential. "
            "The generator reaches the residential boundary but not the "
            "residential center."
        )
    else:
        diagnosis = (
            "GENERATOR LEARNS RESIDENTIAL: >50% of synthetics classify as "
            "residential. The failure must be elsewhere (e.g., blur, "
            "augmentation utility, or fine-tuning dynamics)."
        )
    print(f"\n  DIAGNOSIS: {diagnosis}")
    results["diagnosis"] = diagnosis

    out_path = _OUT_DIR / "diagnostic.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
