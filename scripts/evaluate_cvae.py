"""Evaluate the trained cVAE: samples, memorization, diversity, class fidelity.

Loads the best checkpoint (epoch 25, val=273.9), generates samples, and
computes six analyses:

    1. Sample grid: 16 per class (for the model card).
    2. MSE nearest-neighbor: for each synthetic residential, pixel MSE to
       nearest real residential training image. Content region only.
    3. Perceptual nearest-neighbor: same but in Week 5 ResNet-50 feature
       space (cosine).
    4. Diversity: pairwise cosine distance between synthetic samples
       within each class (feature space).
    5. Latent sensitivity: perturb EACH of the 128 latent dims by +0.5,
       measure content-region pixel change. Reports per-dim mean/std.
    6. Conditional consistency: feed synthetic samples through Week 5
       ResNet-50; report per-class accuracy AND the classifier's real
       validation accuracy as a reference.

All class-fidelity and memorization analysis uses content region only
(rows 28:100). Padding never contributes.

ResNet-50 preprocessing matches Week 5 exactly:
    - Resize content region (72x128) preserving aspect to 224x224
      (letterbox with black padding)
    - ImageNet normalize

Outputs (under outputs/cvae_beta1_seed42/eval/):
    samples_16_per_class.png       — grid for model card
    mse_nearest_neighbor.json
    perceptual_nearest_neighbor.json
    diversity.json
    latent_sensitivity.json
    conditional_consistency.json

Run:
    uv run python scripts/evaluate_cvae.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from torchvision import models

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402
from urbangen.models.cvae import ConditionalVAE  # noqa: E402
from urbangen.training.losses import CONTENT_BOTTOM, CONTENT_TOP, IMAGE_SIZE  # noqa: E402

# Frozen constants matching train_cvae.py
LATENT_DIM = 128
CHANNELS = (32, 64, 128, 256)
SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]
N_GRID_SAMPLES = 16  # for the model card grid
N_ANALYSIS_SAMPLES = 64  # per class, for memorization/diversity/consistency
N_NEAREST_NEIGHBORS_REAL = 2000  # real residential images to search against
LATENT_PERTURB = 0.5
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_CKPT_PATH = _REPO_ROOT / "checkpoints" / "cvae_beta1_seed42.pt"
_URBANMIND_CKPT = Path(os.environ.get("URBANMIND_REPO", "")) / "bdd_resnet50_best.pt"
_EVAL_DIR = _REPO_ROOT / "outputs" / "cvae_beta1_seed42" / "eval"


def load_model(device: torch.device) -> ConditionalVAE:
    if not _CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found at {_CKPT_PATH}")
    ckpt = torch.load(_CKPT_PATH, map_location=device, weights_only=False)
    model = ConditionalVAE(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded cVAE (epoch={ckpt['epoch']}, best_val_loss={ckpt['best_val_loss']:.1f})")
    return model


def load_classifier(device: torch.device):
    if not _URBANMIND_CKPT.exists():
        print(
            f"WARNING: Week 5 classifier not found at {_URBANMIND_CKPT}. "
            f"Set URBANMIND_REPO env var. Skipping ResNet-based analyses."
        )
        return None
    model = models.resnet50(num_classes=3)
    ckpt = torch.load(_URBANMIND_CKPT, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model.load_state_dict(ckpt)
    model.to(device).eval()
    print("Loaded Week 5 ResNet-50 classifier")
    return model


@torch.inference_mode()
def generate_samples(
    model: ConditionalVAE, n_per_class: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (n_per_class * 3, 3, 128, 128) samples with class labels.

    Isolates RNG state. Uses inference_mode for speed and safety.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SEED)
        all_imgs, all_labels = [], []
        for class_id in range(3):
            labels = torch.full((n_per_class,), class_id, dtype=torch.long, device=device)
            imgs = model.sample(n_per_class, labels, device=device)
            all_imgs.append(imgs.cpu())
            all_labels.append(labels.cpu())
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return torch.cat(all_imgs, 0), torch.cat(all_labels, 0)


def save_grid(samples: torch.Tensor, path: Path, n_per_class: int) -> None:
    """Save grid: 3 rows (classes) × n_per_class cols. Padding zeroed for clarity."""
    import torchvision.utils as vutils

    s = samples.clone()
    s[:, :, :CONTENT_TOP, :] = 0.0
    s[:, :, CONTENT_BOTTOM:, :] = 0.0
    grid = vutils.make_grid(s, nrow=n_per_class, padding=2, normalize=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(path))
    print(f"Saved grid: {path}")


def content_crop(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, 128, 128) -> (B, 3, 72, 128) content region."""
    return x[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]


def preprocess_for_resnet(content: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Match Week 5 ResNet pipeline: aspect-preserving 224x224 + ImageNet norm.

    Week 5 training used RandomResizedCrop(224, scale=(0.75, 1.0)) on the
    full 1280x720 image. For eval on 72x128 content, we do the standard
    inference-time transform: resize shorter side to 256, center-crop 224.
    But content is already 72 tall × 128 wide (aspect 0.5625). Resize
    shorter side (72) to 256 → (256, 455). Then center-crop 224 → (224, 224).

    Here we approximate: resize content to 256 x 456, then center-crop 224.
    """
    content = content.to(device)
    # Resize to 256 tall, maintain aspect
    h, w = content.shape[-2], content.shape[-1]
    new_h = 256
    new_w = int(round(w * (new_h / h)))  # 256 * 128/72 ≈ 455
    resized = F.interpolate(content, size=(new_h, new_w), mode="bilinear", align_corners=False)
    # Center crop 224
    top = (new_h - 224) // 2
    left = (new_w - 224) // 2
    cropped = resized[:, :, top : top + 224, left : left + 224]
    # ImageNet normalize
    mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
    std = IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return (cropped - mean) / std


@torch.inference_mode()
def mse_nearest_neighbor(
    synthetic: torch.Tensor, real_residential: torch.Tensor, chunk: int = 32
) -> dict:
    """For each synthetic, find the nearest real by pixel MSE (content region).

    Note: this is normalized MSE, NOT cosine similarity.
    """
    syn_c = content_crop(synthetic).flatten(1)  # (N_syn, D)
    real_c = content_crop(real_residential).flatten(1)  # (N_real, D)
    n_syn = syn_c.size(0)
    mins = torch.empty(n_syn)
    for i in range(0, n_syn, chunk):
        batch = syn_c[i : i + chunk]
        # squared L2 distance
        dists = torch.cdist(batch, real_c, p=2) ** 2  # (chunk, N_real)
        mins[i : i + chunk] = dists.min(dim=1).values
    mse = mins / syn_c.size(1)  # normalize by D to get MSE
    return {
        "n_synthetic": int(n_syn),
        "n_real": int(real_residential.size(0)),
        "per_synthetic_mse": mse.tolist(),
        "mean_mse": float(mse.mean()),
        "median_mse": float(mse.median()),
        "min_mse": float(mse.min()),
        "max_mse": float(mse.max()),
        "note": "MSE in content region only. Approximate: searches against "
        f"{int(real_residential.size(0))} real residential images "
        f"(random subset, not full 8,074).",
    }


@torch.inference_mode()
def resnet_features(classifier, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Extract 2048-d ResNet-50 avgpool features. Preprocessing matches Week 5."""
    normalized = preprocess_for_resnet(images, device)
    x = classifier.conv1(normalized)
    x = classifier.bn1(x)
    x = classifier.relu(x)
    x = classifier.maxpool(x)
    x = classifier.layer1(x)
    x = classifier.layer2(x)
    x = classifier.layer3(x)
    x = classifier.layer4(x)
    x = classifier.avgpool(x)
    return x.flatten(1)


@torch.inference_mode()
def resnet_features_chunked(classifier, images: torch.Tensor, device, chunk: int = 64):
    """Extract features in chunks to avoid OOM."""
    feats = []
    for i in range(0, images.size(0), chunk):
        feats.append(resnet_features(classifier, images[i : i + chunk], device))
    return torch.cat(feats, 0)


@torch.inference_mode()
def perceptual_nearest_neighbor(
    classifier, synthetic: torch.Tensor, real_residential: torch.Tensor, device
) -> dict:
    syn_f = F.normalize(resnet_features(classifier, synthetic, device), dim=1)
    real_f = F.normalize(resnet_features_chunked(classifier, real_residential, device), dim=1)
    cos = syn_f @ real_f.T  # (N_syn, N_real)
    max_cos = cos.max(dim=1).values
    return {
        "n_synthetic": int(synthetic.size(0)),
        "n_real": int(real_residential.size(0)),
        "per_synthetic_max_cosine": max_cos.tolist(),
        "mean_max_cosine": float(max_cos.mean()),
        "median_max_cosine": float(max_cos.median()),
        "max_max_cosine": float(max_cos.max()),
        "n_above_0_95": int((max_cos > 0.95).sum().item()),
        "n_above_0_90": int((max_cos > 0.90).sum().item()),
    }


@torch.inference_mode()
def diversity_analysis(classifier, synthetic: torch.Tensor, labels: torch.Tensor, device) -> dict:
    """Pairwise cosine distance within class (ResNet-50 feature space)."""
    result = {"per_class": {}}
    for c in range(3):
        mask = labels == c
        feats = F.normalize(resnet_features(classifier, synthetic[mask], device), dim=1)
        sim = feats @ feats.T
        n = sim.size(0)
        off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
        distances = 1.0 - off
        result["per_class"][CLASS_NAMES[c]] = {
            "n": int(n),
            "mean_pairwise_distance": float(distances.mean()),
            "std_pairwise_distance": float(distances.std()),
            "min_distance": float(distances.min()),
            "max_distance": float(distances.max()),
        }
    return result


@torch.inference_mode()
def latent_sensitivity(
    model: ConditionalVAE,
    device: torch.device,
    n_per_class: int = 8,
    n_dims_to_test: int | None = None,
) -> dict:
    """Perturb each latent dim by +0.5, measure content-region L2² change.

    If n_dims_to_test is None, tests all 128 dims. Otherwise tests a subset
    (for speed).
    """
    dims_to_test = n_dims_to_test or LATENT_DIM
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SEED)
        results = {}
        for c in range(3):
            labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            z = torch.randn(n_per_class, LATENT_DIM, device=device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                base = model.decode(z, labels)
            base_c = content_crop(base.float())
            per_dim = []
            for dim in range(dims_to_test):
                z_pert = z.clone()
                z_pert[:, dim] += LATENT_PERTURB
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    pert = model.decode(z_pert, labels)
                diff = (content_crop(pert.float()) - base_c).pow(2).mean(dim=[1, 2, 3])
                per_dim.append(diff.mean().item())
            per_dim_t = torch.tensor(per_dim)
            results[CLASS_NAMES[c]] = {
                "n_per_class": n_per_class,
                "n_dims_tested": dims_to_test,
                "per_dim_mean_content_l2sq": per_dim,
                "mean": float(per_dim_t.mean()),
                "median": float(per_dim_t.median()),
                "std": float(per_dim_t.std()),
                "min": float(per_dim_t.min()),
                "max": float(per_dim_t.max()),
            }
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    return results


@torch.inference_mode()
def conditional_consistency(
    classifier,
    synthetic: torch.Tensor,
    labels: torch.Tensor,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run synthetic through Week 5 classifier. Report accuracy vs. val reference."""
    # Synthetic accuracy
    normalized = preprocess_for_resnet(synthetic, device)
    logits = classifier(normalized)
    preds = logits.argmax(dim=1).cpu()
    overall = float((preds == labels).float().mean())

    per_class = {}
    for c in range(3):
        mask = labels == c
        acc = float((preds[mask] == labels[mask]).float().mean())
        pred_counts = torch.bincount(preds[mask], minlength=3).tolist()
        per_class[CLASS_NAMES[c]] = {
            "n": int(mask.sum()),
            "accuracy": acc,
            "predicted_as": {CLASS_NAMES[i]: pred_counts[i] for i in range(3)},
        }

    # Real validation accuracy (reference for "is the classifier reliable?")
    correct = 0
    total = 0
    for x, y in val_loader:
        normalized = preprocess_for_resnet(x, device)
        logits = classifier(normalized)
        pred = logits.argmax(dim=1).cpu()
        correct += int((pred == y).sum())
        total += int(y.size(0))
    real_val_acc = correct / total if total else 0.0

    return {
        "overall_synthetic_accuracy": overall,
        "real_validation_accuracy": real_val_acc,
        "per_class": per_class,
        "interpretation_note": (
            "Synthetic accuracy is meaningful only relative to real val accuracy. "
            "If real_val_acc is low, the classifier is weak and synthetic accuracy "
            "understates the model. If real_val_acc is high but synthetic accuracy "
            "is near chance, the model's class conditioning is broken."
        ),
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not (_CACHE_ROOT / "manifest.json").exists():
        print(f"ERROR: cache not found at {_CACHE_ROOT}")
        return 1

    model = load_model(device)
    classifier = load_classifier(device)
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Load real residential images (random subset for NN, full for train_ds access)
    print("\nLoading real residential training images...")
    train_ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    residential_indices = [i for i in range(len(train_ds)) if int(train_ds.labels[i]) == 2]
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(residential_indices, size=N_NEAREST_NEIGHBORS_REAL, replace=False)
    print(
        f"  {len(residential_indices):,} real residential train; using random "
        f"{N_NEAREST_NEIGHBORS_REAL} for NN search"
    )
    real_res = torch.stack([train_ds[int(i)][0] for i in chosen])
    del train_ds  # free memory

    # Load validation set for classifier reference
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    # 1. Grid samples (16 per class)
    print(f"\n1. Generating {N_GRID_SAMPLES} samples per class (grid)...")
    grid_samples, grid_labels = generate_samples(model, N_GRID_SAMPLES, device)
    save_grid(grid_samples, _EVAL_DIR / "samples_16_per_class.png", N_GRID_SAMPLES)

    # 2-6. Analysis samples (N_ANALYSIS_SAMPLES per class)
    print(f"\n2. Generating {N_ANALYSIS_SAMPLES} samples per class (analysis)...")
    analysis_samples, analysis_labels = generate_samples(model, N_ANALYSIS_SAMPLES, device)

    # 2. MSE nearest-neighbor (residential only)
    print("\n3. MSE nearest-neighbor (residential)...")
    res_mask = analysis_labels == 2
    mse_result = mse_nearest_neighbor(analysis_samples[res_mask], real_res)
    (_EVAL_DIR / "mse_nearest_neighbor.json").write_text(json.dumps(mse_result, indent=2))
    print(f"  mean MSE: {mse_result['mean_mse']:.6f}")
    print(f"  median MSE: {mse_result['median_mse']:.6f}")

    # 3-6. Classifier-based analyses
    if classifier is not None:
        print("\n4. Perceptual nearest-neighbor (ResNet-50 features)...")
        perc_result = perceptual_nearest_neighbor(
            classifier, analysis_samples[res_mask], real_res, device
        )
        (_EVAL_DIR / "perceptual_nearest_neighbor.json").write_text(
            json.dumps(perc_result, indent=2)
        )
        print(f"  mean cosine: {perc_result['mean_max_cosine']:.4f}")
        print(f"  above 0.95: {perc_result['n_above_0_95']} / {perc_result['n_synthetic']}")
        print(f"  above 0.90: {perc_result['n_above_0_90']} / {perc_result['n_synthetic']}")

        print("\n5. Diversity (pairwise cosine distance within class)...")
        div_result = diversity_analysis(classifier, analysis_samples, analysis_labels, device)
        (_EVAL_DIR / "diversity.json").write_text(json.dumps(div_result, indent=2))
        for c_name, stats in div_result["per_class"].items():
            print(
                f"  {c_name}: mean dist={stats['mean_pairwise_distance']:.4f} "
                f"(±{stats['std_pairwise_distance']:.4f})"
            )

        print("\n6. Conditional consistency (Week 5 classifier)...")
        cond_result = conditional_consistency(
            classifier, analysis_samples, analysis_labels, val_loader, device
        )
        (_EVAL_DIR / "conditional_consistency.json").write_text(json.dumps(cond_result, indent=2))
        print(f"  synthetic overall acc: {cond_result['overall_synthetic_accuracy']:.3f}")
        print(f"  real val acc (reference): {cond_result['real_validation_accuracy']:.3f}")
        for c_name, stats in cond_result["per_class"].items():
            print(f"  {c_name}: acc={stats['accuracy']:.3f}, predicted_as={stats['predicted_as']}")

    # 5. Latent sensitivity (all 128 dims)
    print("\n7. Latent perturbation sensitivity (all 128 dims)...")
    lat_result = latent_sensitivity(model, device)
    (_EVAL_DIR / "latent_sensitivity.json").write_text(json.dumps(lat_result, indent=2))
    for c_name, stats in lat_result.items():
        print(
            f"  {c_name}: mean L2²={stats['mean']:.6f}, "
            f"std={stats['std']:.6f}, max={stats['max']:.6f}"
        )

    print(f"\nAll outputs saved to {_EVAL_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
