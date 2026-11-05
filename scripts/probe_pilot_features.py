"""Fast feature-probe pilot for G1c.

Before spending 2 hours on fine-tuning 4 ResNet-50 arms, run a much
cheaper test: fit logistic regression on FROZEN ImageNet-pretrained
ResNet-50 features, with and without synthetic residential augmentation.

If synthetic data doesn't move residential F1 in a frozen-feature probe,
it almost certainly won't move it in fine-tuning either (where the
signal-to-noise ratio is worse). If it does move, the fine-tune is worth
running with the corrected 4-arm protocol.

Probe design (single seed, frozen features):
    Arm A: real train only
    Arm B: real train + 3,000 synthetic residential features
    Arm C: real train + 3,000 real residential oversampled (control)
    Arm D: real train with class-weighted logistic regression (control)

Evaluate on the real clean val split (4,727 images). Test set is NOT
touched by this probe.

Run:
    uv run python scripts/probe_pilot_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_recall_fscore_support,
)
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402

SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]
NUM_CLASSES = 3
CONTENT_TOP = 28
CONTENT_BOTTOM = 100
CVAE_LATENT_DIM = 128
CVAE_CHANNELS = (32, 64, 128, 256)
IMAGE_SIZE = 128

_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_CVAE_CKPT = _REPO_ROOT / "checkpoints" / "cvae_beta1_seed42.pt"
_SYNTHETIC_CACHE = _REPO_ROOT / "outputs" / "g1c_synthetic_residential.pt"
_FEATURES_CACHE = _REPO_ROOT / "outputs" / "g1c_features"
_OUT_DIR = _REPO_ROOT / "outputs" / "g1c_probe"


def load_resnet50_features(device):
    """Load ImageNet-pretrained ResNet-50 with fc replaced by Identity."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.inference_mode()
def extract_features(model, images: torch.Tensor, device, batch_size: int = 64):
    """Extract 2048-d features for a batch of images.

    images: (N, 3, 128, 128) in [0, 1]. Applies content->224 resize and
    ImageNet norm to match ResNet-50's training preprocessing.
    """
    all_feats = []
    for i in range(0, images.size(0), batch_size):
        batch = images[i : i + batch_size].to(device)
        content = batch[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
        resized = F.interpolate(content, size=(224, 224), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        normalized = (resized - mean) / std
        feats = model(normalized)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, 0)


def load_cached_features(split: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Load or compute features for real train/val split."""
    _FEATURES_CACHE.mkdir(parents=True, exist_ok=True)
    feat_path = _FEATURES_CACHE / f"{split}_features.pt"

    ds = BDD100K128Dataset(_CACHE_ROOT, split)
    labels = torch.tensor([ds[i][1] for i in range(len(ds))], dtype=torch.long)

    if feat_path.exists():
        print(f"  loading cached {split} features from {feat_path}")
        feats = torch.load(feat_path, map_location="cpu", weights_only=False)
        return feats, labels

    print(f"  extracting {split} features for {len(ds):,} images...")
    model = load_resnet50_features(device)

    all_imgs = []
    chunk = 512
    for i in range(0, len(ds), chunk):
        batch = torch.stack([ds[j][0] for j in range(i, min(i + chunk, len(ds)))])
        all_imgs.append(batch)
        if (i + chunk) % 5000 == 0:
            print(f"    loaded {i + chunk}/{len(ds)} images")
    all_imgs = torch.cat(all_imgs, 0)
    print(f"  extracted all images: {all_imgs.shape}")

    feats = extract_features(model, all_imgs, device)
    print(f"  extracted {split} features: {feats.shape}")
    torch.save(feats, feat_path)
    return feats, labels


def generate_synthetic_residential(device) -> torch.Tensor:
    """Return (N, 3, 128, 128) synthetic residential images."""
    if _SYNTHETIC_CACHE.exists():
        print(f"  loading synthetic from {_SYNTHETIC_CACHE}")
        return torch.load(_SYNTHETIC_CACHE, map_location="cpu", weights_only=False)["images"]

    from urbangen.models.cvae import ConditionalVAE

    ckpt = torch.load(_CVAE_CKPT, map_location=device, weights_only=False)
    cvae = ConditionalVAE(image_size=IMAGE_SIZE, latent_dim=CVAE_LATENT_DIM, channels=CVAE_CHANNELS)
    cvae.load_state_dict(ckpt["model_state_dict"])
    cvae.to(device).eval()

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SEED)
        all_imgs = []
        for start in range(0, 3000, 64):
            n = min(64, 3000 - start)
            labels = torch.full((n,), 2, dtype=torch.long, device=device)
            samples = cvae.sample(n, labels, device=device)
            all_imgs.append(samples.cpu())
        images = torch.cat(all_imgs, 0)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    _SYNTHETIC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"images": images, "labels": torch.full((3000,), 2, dtype=torch.long)},
        _SYNTHETIC_CACHE,
    )
    return images


def probe_arm(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Fit logistic regression (balanced class weight) on train, eval on val."""
    print(f"\nProbe arm: {name}")
    print(
        f"  train samples: {len(y_train):,} "
        f"(per-class: {[int((y_train == c).sum()) for c in range(NUM_CLASSES)]})"
    )
    print(f"  val samples:   {len(y_val):,}")

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_val)

    per_class = precision_recall_fscore_support(
        y_val, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0
    )
    precision, recall, f1, support = per_class

    results = {
        "accuracy": float((y_val == y_pred).mean()),
        "macro_f1": float(f1_score(y_val, y_pred, average="macro", zero_division=0)),
        "per_class": {
            CLASS_NAMES[c]: {
                "precision": float(precision[c]),
                "recall": float(recall[c]),
                "f1": float(f1[c]),
                "support": int(support[c]),
            }
            for c in range(NUM_CLASSES)
        },
    }
    print(f"  macro-F1: {results['macro_f1']:.4f}")
    print(
        f"  residential: P={results['per_class']['residential']['precision']:.4f}, "
        f"R={results['per_class']['residential']['recall']:.4f}, "
        f"F1={results['per_class']['residential']['f1']:.4f}"
    )
    return results


def arm_label(arm: str) -> str:
    return {
        "A": "Real only",
        "B": "Real + 3000 cVAE synthetic residential",
        "C": "Real + 3000 real residential oversampled",
        "D": "Real only, class-weighted LR",
    }[arm]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _FEATURES_CACHE.mkdir(parents=True, exist_ok=True)

    if not _CVAE_CKPT.exists():
        print(f"ERROR: cVAE checkpoint not found at {_CVAE_CKPT}")
        return 1

    print("\n=== Loading real features (cached after first run) ===")
    train_feats, train_labels = load_cached_features("train", device)
    val_feats, val_labels = load_cached_features("val", device)
    print(f"  train features: {train_feats.shape}")
    print(f"  val features:   {val_feats.shape}")

    print("\n=== Loading synthetic residential ===")
    synth_imgs = generate_synthetic_residential(device)
    print(f"  synthetic images: {synth_imgs.shape}")

    print("\n=== Extracting synthetic features ===")
    synth_feat_path = _FEATURES_CACHE / "synthetic_features.pt"
    if synth_feat_path.exists():
        print(f"  loading from cache: {synth_feat_path}")
        synth_feats = torch.load(synth_feat_path, map_location="cpu", weights_only=False)
    else:
        model = load_resnet50_features(device)
        synth_feats = extract_features(model, synth_imgs, device)
        torch.save(synth_feats, synth_feat_path)
    print(f"  synthetic features: {synth_feats.shape}")

    # NumPy arrays for sklearn
    x_train = train_feats.numpy()
    y_train = train_labels.numpy()
    x_val = val_feats.numpy()
    y_val = val_labels.numpy()
    x_synth = synth_feats.numpy()
    y_synth = np.full(len(x_synth), 2, dtype=np.int64)

    results = {}

    # Arm A: real train only
    results["A"] = probe_arm("A — real only", x_train, y_train, x_val, y_val)

    # Arm B: real + synthetic residential
    x_b = np.concatenate([x_train, x_synth], axis=0)
    y_b = np.concatenate([y_train, y_synth], axis=0)
    results["B"] = probe_arm("B — real + synthetic residential", x_b, y_b, x_val, y_val)

    # Arm C: real + real residential oversampled (natural control)
    rng = np.random.default_rng(SEED)
    residential_indices = np.where(y_train == 2)[0]
    oversampled_indices = rng.choice(residential_indices, size=3000, replace=True)
    x_oversampled = x_train[oversampled_indices]
    y_oversampled = y_train[oversampled_indices]
    x_c = np.concatenate([x_train, x_oversampled], axis=0)
    y_c = np.concatenate([y_train, y_oversampled], axis=0)
    results["C"] = probe_arm("C — real + 3000 real residential oversampled", x_c, y_c, x_val, y_val)

    # Arm D: real only, class-weighted logistic regression (no added data)
    # Note: LogisticRegression with class_weight="balanced" is what
    # probe_arm already does for arms A/B/C. Arm D in the pilot would be
    # class-weighted loss on a small model — here the frozen-feature probe
    # can only add per-class weights, which the balanced setting already
    # provides. Arm D therefore differs from arm A by nothing in this
    # probe. We report it to be explicit: the probe cannot distinguish
    # arm D from arm A. Both use the same features and the same
    # balanced logistic regression. In the fine-tune, arm D differs from
    # arm A by a class-weighted loss, which the probe cannot replicate.
    results["D"] = probe_arm(
        "D — real only, class-weighted (same as A in frozen-feature probe)",
        x_train,
        y_train,
        x_val,
        y_val,
    )

    # Summary table
    print("\n=== SUMMARY (real val split) ===")
    print(f"{'Arm':<6} {'Description':<44} {'Res P':<8} {'Res R':<8} {'Res F1':<8} {'Macro F1':<8}")
    print("-" * 96)
    for arm in "ABCD":
        r = results[arm]["per_class"]["residential"]
        print(
            f"{arm:<6} {arm_label(arm):<44} {r['precision']:<8.4f} "
            f"{r['recall']:<8.4f} {r['f1']:<8.4f} {results[arm]['macro_f1']:<8.4f}"
        )

    # Deltas on residential F1
    print("\n=== DELTAS on residential F1 ===")
    res_f1_a = results["A"]["per_class"]["residential"]["f1"]
    res_f1_b = results["B"]["per_class"]["residential"]["f1"]
    res_f1_c = results["C"]["per_class"]["residential"]["f1"]
    res_f1_d = results["D"]["per_class"]["residential"]["f1"]

    print(f"  B - A (synthetic vs baseline):       {res_f1_b - res_f1_a:+.4f}")
    print(f"  B - C (synthetic vs oversampling):   {res_f1_b - res_f1_c:+.4f}")
    print(f"  B - D (synthetic vs class-weighted): {res_f1_b - res_f1_d:+.4f}")

    # Verdict
    print("\n=== VERDICT ===")
    if res_f1_b > res_f1_c + 0.02 and res_f1_b > res_f1_d + 0.02:
        verdict = (
            "PROMISING: Arm B beats both non-generative controls by >0.02 "
            "residential F1. The full fine-tune is worth running with the "
            "corrected 4-arm protocol."
        )
    elif res_f1_b > res_f1_a + 0.02:
        verdict = (
            "AMBIGUOUS: Arm B beats baseline but does not clearly beat the "
            "oversampling and class-weight controls. Fine-tune may show signal "
            "but interpretability is limited."
        )
    else:
        verdict = (
            "NO SIGNAL: Arm B does not meaningfully beat the baseline. The "
            "full fine-tune is unlikely to produce a different result. Close "
            "G1c with this evidence."
        )
    print(f"  {verdict}")

    out = {
        "probe_type": "frozen ResNet-50 ImageNet features + logistic regression",
        "probe_seed": SEED,
        "test_set_used": False,
        "note": (
            "Probe uses real train+val only. Test set is NOT touched. "
            "If this probe shows signal, run the full fine-tune (which uses "
            "test set once at the end). If it shows nothing, close G1c."
        ),
        "caveat_arm_D": (
            "In a frozen-feature probe, 'class-weighted logistic regression' "
            "is identical to arm A because both use balanced class weights by "
            "default. Arm D in the full fine-tune applies class-weighted loss "
            "on a small head — a setup the probe cannot replicate. Arm D's "
            "probe result should be treated as a duplicate of A, not an "
            "independent control."
        ),
        "results": results,
        "deltas_residential_f1": {
            "B_minus_A": float(res_f1_b - res_f1_a),
            "B_minus_C": float(res_f1_b - res_f1_c),
            "B_minus_D": float(res_f1_b - res_f1_d),
        },
        "verdict": verdict,
    }
    (_OUT_DIR / "probe_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {_OUT_DIR / 'probe_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
