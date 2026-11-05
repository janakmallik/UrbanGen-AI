"""Diagnose cVAE class conditioning before choosing an architectural fix.

Three checks:
    1. Train a small 128x128 evaluator (ResNet-18 or small CNN) on the
       cache's train split. Save it. Report real val accuracy.
    2. Reconstruction ceiling: encode real val residential images,
       decode with correct label, classify reconstructions. Compare
       accuracy to real val images.
    3. Label swap: encode real residential images to z, decode with
       each of the 3 labels, measure pixel-space difference. Report
       per-class diffs.

Decision tree (see docs/experiments.md):
    Check 2 collapse  -> blur limit. No conditioning fix helps.
    Check 3 near-zero -> decoder ignores label. Class-conditional prior.
    Check 3 large    -> prior-sample problem. Class-conditional prior.
    Check 3 large + prior samples classify OK -> old evaluator was
                        the problem. Run G1c pilot.

Run:
    uv run python scripts/diagnose_class_conditioning.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402
from urbangen.models.cvae import ConditionalVAE  # noqa: E402
from urbangen.training.losses import CONTENT_BOTTOM, CONTENT_TOP, IMAGE_SIZE  # noqa: E402

LATENT_DIM = 128
CHANNELS = (32, 64, 128, 256)
SEED = 42
CLASS_NAMES = ["city street", "highway", "residential"]

_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_CVAE_CKPT = _REPO_ROOT / "checkpoints" / "cvae_beta1_seed42.pt"
_OUT_DIR = _REPO_ROOT / "outputs" / "diagnostic_conditioning"


class SmallEvaluator(nn.Module):
    """Small CNN for 128x128 content-region classification.

    Uses content rows only (28:100). Roughly 1M params. Fast to train.
    """

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # -> 36x64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # -> 18x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # -> 9x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # -> 5x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 128, 128) in [0, 1]
        content = x[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
        h = self.features(content)
        return self.fc(h.flatten(1))


@torch.inference_mode()
def check_conditioning_strength(cvae: ConditionalVAE, device, n_z: int = 16) -> dict:
    """How much does the class label change the decoded image for a fixed z?"""
    cvae.eval()
    # Deliberately no autocast: we are measuring ~1e-4 effects that fp16
    # would round away.
    z = torch.randn(n_z, cvae.latent_dim, device=device)
    imgs = {}
    for c in range(3):
        ids = torch.full((n_z,), c, dtype=torch.long, device=device)
        imgs[c] = cvae.decode(z, ids)  # (n_z, 3, 128, 128)
    pairs = [(0, 1), (0, 2), (1, 2)]
    per_pair = {}
    for a, b in pairs:
        d = (imgs[a] - imgs[b]).abs()
        per_pair[f"{a}_vs_{b}"] = {
            "mean_abs_diff": d.mean().item(),
            "per_channel_mean_abs_diff": d.mean(dim=(0, 2, 3)).tolist(),
            "max_abs_diff": d.max().item(),
        }
    # Also check variance of decoded means across classes
    means = {c: imgs[c].mean(dim=(0, 2, 3)).tolist() for c in imgs}

    worst = min(v["mean_abs_diff"] for v in per_pair.values())
    if worst < 1e-3:
        verdict = "DEAD: decoder ignores label (mean_abs_diff near 1e-4 noise floor)."
    elif worst < 1e-2:
        verdict = "WEAK: label has a small but non-noise effect."
    else:
        verdict = "ALIVE: label materially changes output."

    return {
        "per_pair": per_pair,
        "class_mean_rgb": means,
        "worst_pair_mean_abs_diff": worst,
        "verdict": verdict,
    }


def train_evaluator(device, epochs: int = 10) -> tuple[SmallEvaluator, float]:
    """Train small evaluator on cache train split. Return (model, val_acc)."""
    train_ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")

    # Subsample train for speed: 15,000 images is plenty for a small CNN
    rng = np.random.default_rng(SEED)
    n_train = 15000
    train_indices = rng.choice(len(train_ds), size=n_train, replace=False)
    train_sub = Subset(train_ds, train_indices.tolist())

    train_loader = DataLoader(train_sub, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = SmallEvaluator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        # Quick val check
        model.eval()
        correct = 0
        total = 0
        per_class = {c: [0, 0] for c in range(3)}
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += int((pred == y).sum())
                total += int(y.size(0))
                for c in range(3):
                    mask = y == c
                    per_class[c][0] += int((pred[mask] == c).sum())
                    per_class[c][1] += int(mask.sum())
        val_acc = correct / total
        per_class_acc = [per_class[c][0] / max(per_class[c][1], 1) for c in range(3)]
        print(
            f"  epoch {epoch + 1:2d}: train loss={total_loss / n:.4f}, "
            f"val acc={val_acc:.3f}, "
            f"per-class: {per_class_acc[0]:.2f}/{per_class_acc[1]:.2f}/{per_class_acc[2]:.2f}"
        )

    return model, val_acc


@torch.inference_mode()
def check_reconstruction_ceiling(evaluator: SmallEvaluator, cvae: ConditionalVAE, device) -> dict:
    """Encode real val images, decode with correct label, classify reconstructions."""
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)

    real_correct = 0
    recon_correct = 0
    total = 0
    per_class_real = {c: [0, 0] for c in range(3)}
    per_class_recon = {c: [0, 0] for c in range(3)}

    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        # Real image classification
        pred_real = evaluator(x).argmax(dim=1)
        real_correct += int((pred_real == y).sum())

        # Reconstruction: encode-decode with correct label
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            mu, _ = cvae.encode(x, y)
            recon = cvae.decode(mu, y)
        pred_recon = evaluator(recon.float()).argmax(dim=1)
        recon_correct += int((pred_recon == y).sum())

        total += int(y.size(0))
        for c in range(3):
            mask = y == c
            per_class_real[c][0] += int((pred_real[mask] == c).sum())
            per_class_real[c][1] += int(mask.sum())
            per_class_recon[c][0] += int((pred_recon[mask] == c).sum())
            per_class_recon[c][1] += int(mask.sum())

    return {
        "real_accuracy": real_correct / total,
        "reconstruction_accuracy": recon_correct / total,
        "per_class_real": {
            CLASS_NAMES[c]: per_class_real[c][0] / max(per_class_real[c][1], 1) for c in range(3)
        },
        "per_class_reconstruction": {
            CLASS_NAMES[c]: per_class_recon[c][0] / max(per_class_recon[c][1], 1) for c in range(3)
        },
    }


@torch.inference_mode()
def check_label_swap(cvae: ConditionalVAE, device, n: int = 100) -> dict:
    """Encode real residential images, decode with each of 3 labels, measure diff."""
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    # Find residential indices
    residential_indices = [i for i in range(len(val_ds)) if int(val_ds.labels[i]) == 2][:n]
    residential_samples = torch.stack([val_ds[i][0] for i in residential_indices]).to(device)
    residential_labels = torch.full((n,), 2, dtype=torch.long, device=device)

    # Encode residential to z
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        mu, _ = cvae.encode(residential_samples, residential_labels)

    # Decode with each label
    recons = {}
    for c, cname in enumerate(CLASS_NAMES):
        labels_c = torch.full((n,), c, dtype=torch.long, device=device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            recons[cname] = cvae.decode(mu, labels_c)

    # Pixel differences between decodes
    def diff_content(a, b):
        a_c = a[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
        b_c = b[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
        return float(((a_c - b_c) ** 2).mean())

    diff_res_vs_city = diff_content(recons["residential"], recons["city street"])
    diff_res_vs_highway = diff_content(recons["residential"], recons["highway"])
    diff_city_vs_highway = diff_content(recons["city street"], recons["highway"])

    return {
        "diff_residential_vs_city": diff_res_vs_city,
        "diff_residential_vs_highway": diff_res_vs_highway,
        "diff_city_vs_highway": diff_city_vs_highway,
        "interpretation": (
            "Near-zero diffs = decoder ignores label. Substantial diffs = "
            "decoder uses label at reconstruction, and the prior-sample "
            "failure is a class-agnostic prior problem."
        ),
    }


def _generate_samples(
    cvae: ConditionalVAE, n_per_class: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n_per_class prior samples for every class. Mirrors evaluate_cvae."""
    cvae.eval()
    all_imgs, all_labels = [], []
    for c in range(cvae.num_classes):
        ids = torch.full((n_per_class,), c, dtype=torch.long, device=device)
        all_imgs.append(cvae.sample(n_per_class, ids, device=device))
        all_labels.append(ids)
    return torch.cat(all_imgs, 0), torch.cat(all_labels, 0)


@torch.inference_mode()
def check_prior_samples(evaluator, cvae, device, n_per_class: int = 64):
    """Generate prior samples per class, classify them with evaluator."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SEED)
        all_samples, all_labels = [], []
        for c in range(3):
            labels = torch.full((n_per_class,), c, dtype=torch.long, device=device)
            imgs = cvae.sample(n_per_class, labels, device=device)
            all_samples.append(imgs.cpu())
            all_labels.append(labels.cpu())
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    samples = torch.cat(all_samples, 0).to(device)
    labels = torch.cat(all_labels, 0).to(device)

    preds = evaluator(samples).argmax(dim=1)
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

    return {"overall_accuracy": overall, "per_class": per_class}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check 1: Train evaluator
    print("\n=== Check 1: Train 128x128 evaluator ===")
    evaluator, val_acc = train_evaluator(device, epochs=10)
    print(f"  Final real val accuracy: {val_acc:.3f}")
    torch.save(evaluator.state_dict(), _OUT_DIR / "evaluator_128.pt")

    # Load cVAE
    print("\nLoading cVAE from best checkpoint...")
    ckpt = torch.load(_CVAE_CKPT, map_location=device, weights_only=False)
    cvae = ConditionalVAE(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS)
    cvae.load_state_dict(ckpt["model_state_dict"])
    cvae.to(device).eval()
    print(f"  cVAE loaded (epoch={ckpt['epoch']}, val_loss={ckpt['best_val_loss']:.1f})")

    w = cvae.class_embed.weight.detach()
    print(f"  class_embed weight std: {w.std().item():.6f}  (init scale ~{1 / 16**0.5:.6f})")
    d = torch.cdist(w, w)
    d.fill_diagonal_(float("inf"))
    print(f"  class_embed min pairwise row L2: {d.min().item():.6f}")
    print("  class_embed weight:\n", w.cpu())

    # Check 2: Reconstruction ceiling
    print("\n=== Check 2: Reconstruction ceiling ===")
    recon_result = check_reconstruction_ceiling(evaluator, cvae, device)
    print(f"  Real accuracy:          {recon_result['real_accuracy']:.3f}")
    print(f"  Reconstruction accuracy: {recon_result['reconstruction_accuracy']:.3f}")
    print(f"  Per-class real:          {recon_result['per_class_real']}")
    print(f"  Per-class reconstruction: {recon_result['per_class_reconstruction']}")

    # Check 3: Label swap
    print("\n=== Check 3: Label swap ===")
    swap_result = check_label_swap(cvae, device, n=100)
    print(f"  diff_res_vs_city:     {swap_result['diff_residential_vs_city']:.6f}")
    print(f"  diff_res_vs_highway:  {swap_result['diff_residential_vs_highway']:.6f}")
    print(f"  diff_city_vs_highway: {swap_result['diff_city_vs_highway']:.6f}")

    # Check 3b: Conditioning strength (evaluator-free)
    print("\n=== Check 3b: Conditioning strength (evaluator-free) ===")
    cond_result = check_conditioning_strength(cvae, device, n_z=64)
    for pair, stats in cond_result["per_pair"].items():
        print(
            f"  {pair}: mean_abs_diff={stats['mean_abs_diff']:.6f}, "
            f"max_abs_diff={stats['max_abs_diff']:.6f}, "
            f"per_channel={[round(v, 6) for v in stats['per_channel_mean_abs_diff']]}"
        )
    print(f"  class_mean_rgb: {cond_result['class_mean_rgb']}")
    print(f"  VERDICT: {cond_result['verdict']}")

    # Prior samples
    print("\n=== Check 4: Prior samples with new evaluator ===")
    prior_result = check_prior_samples(evaluator, cvae, device, n_per_class=64)
    print(f"  Overall synthetic accuracy: {prior_result['overall_accuracy']:.3f}")
    for c_name, stats in prior_result["per_class"].items():
        print(f"  {c_name}: acc={stats['accuracy']:.3f}, predicted_as={stats['predicted_as']}")

    # Save all results
    out = {
        "evaluator_real_val_accuracy": val_acc,
        "conditioning_strength": cond_result,
        "reconstruction_ceiling": recon_result,
        "label_swap": swap_result,
        "prior_samples": prior_result,
        "decision_tree": {
            "1_recon_collapse": (
                "recon_accuracy << real_accuracy -> blur limit; no conditioning "
                "fix helps. Close G1b, consider G2 (DDPM)."
            ),
            "2_label_swap_near_zero": (
                "diffs < 1e-4 -> decoder ignores label at reconstruction time. "
                "Class-conditional prior (10 lines) is the fix."
            ),
            "3_label_swap_substantial": (
                "diffs >> 1e-4 -> decoder uses label at reconstruction. "
                "Prior-sample failure is a class-agnostic-prior problem. "
                "Class-conditional prior is still the fix."
            ),
            "4_prior_samples_work_with_new_evaluator": (
                "If prior samples classify well with new evaluator, the old "
                "Week 5 evaluator was the problem. Go straight to G1c pilot."
            ),
        },
    }
    (_OUT_DIR / "conditioning_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {_OUT_DIR / 'conditioning_diagnostic.json'}")


if __name__ == "__main__":
    sys.exit(main())
