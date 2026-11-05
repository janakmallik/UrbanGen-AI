"""Blur-aware evaluator diagnostic with a matched sharp control.

Fixes from the first draft:
- Balanced accuracy and macro-F1 (chance ~0.33), not raw accuracy.
- Class-weighted CrossEntropyLoss.
- Matched sharp-image evaluator (same 15k images, same recipe) as control.
- Best-epoch checkpoint by val macro-F1.
- Prior-sample test with matched z (the actual pilot-relevant test).
- Reconstructions stored as float16 (halves memory).
- Explicit train/val provenance note: train recons come from images the
  cVAE saw; val recons from held-out images.

Metric interpretation:
- Sharp baseline: how well a matched evaluator does on sharp real images.
  This is the ceiling.
- Blur-aware recon acc: how well a matched evaluator does on reconstructions.
  Compare to the sharp baseline (ratio), not to an arbitrary threshold.
- Matched-z prior test: same z, three labels, classify each decode. This is
  what the pilot would actually feed to a downstream classifier.

Run:
    uv run python scripts/diagnose_blur_aware_evaluator.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score
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
_OUT_DIR = _REPO_ROOT / "outputs" / "diagnostic_blur_aware"


class SmallEvaluator(nn.Module):
    """Content-region CNN. Same architecture for sharp and blur-aware."""

    def __init__(self, n_classes: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content = x[:, :, CONTENT_TOP:CONTENT_BOTTOM, :]
        h = self.features(content)
        return self.fc(h.flatten(1))


@torch.inference_mode()
def reconstruct_batch(cvae: ConditionalVAE, x: torch.Tensor, y: torch.Tensor, device):
    x = x.to(device)
    y = y.to(device)
    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        mu, _ = cvae.encode(x, y)
        recon = cvae.decode(mu, y)
    return recon.float().cpu().to(torch.float16)


def build_reconstruction_set(cvae, split, device, indices=None, label=""):
    """Return (recons_float16, labels) for the given split / indices."""
    ds = BDD100K128Dataset(_CACHE_ROOT, split)
    idx = indices if indices is not None else list(range(len(ds)))
    loader = DataLoader(Subset(ds, idx), batch_size=64, shuffle=False, num_workers=0)
    all_recons, all_labels = [], []
    for i, (x, y) in enumerate(loader):
        all_recons.append(reconstruct_batch(cvae, x, y, device))
        all_labels.append(y)
        if (i + 1) % 50 == 0:
            print(f"  {label}: {min((i + 1) * 64, len(idx))}/{len(idx)} done")
    return torch.cat(all_recons, 0), torch.cat(all_labels, 0)


def _eval_metrics(model, loader, device):
    """Compute accuracy, balanced accuracy, macro-F1, per-class recall, confusion."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            pred = model(x).argmax(dim=1)
            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    per_class_recall = {}
    for c in range(3):
        mask = labels == c
        if mask.sum() > 0:
            per_class_recall[CLASS_NAMES[c]] = float((preds[mask] == c).mean())
        else:
            per_class_recall[CLASS_NAMES[c]] = None
    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for t, p in zip(labels, preds, strict=True):
        confusion[t, p] += 1
    return {
        "accuracy": float((preds == labels).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion.tolist(),
    }


def train_evaluator(
    train_images_f16,
    train_labels,
    val_images_f16,
    val_labels,
    device,
    label_prefix="",
    epochs=10,
    use_class_weights=True,
):
    """Train evaluator on the given train set. Evaluate on val each epoch."""
    model = SmallEvaluator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    if use_class_weights:
        counts = torch.bincount(train_labels, minlength=3).float()
        weights = (counts.sum() / (3 * counts.clamp(min=1))).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    train_ds = list(zip(train_images_f16, train_labels, strict=True))
    val_ds = list(zip(val_images_f16, val_labels, strict=True))
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    best_macro_f1 = -1.0
    best_state = None
    best_metrics = None
    history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device).float()  # cast back from float16
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        scheduler.step()

        metrics = _eval_metrics(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / n, **metrics})
        print(
            f"  {label_prefix} epoch {epoch + 1:2d}: loss={total_loss / n:.4f}, "
            f"acc={metrics['accuracy']:.3f}, bal_acc={metrics['balanced_accuracy']:.3f}, "
            f"macro_f1={metrics['macro_f1']:.3f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = metrics

    model.load_state_dict(best_state)
    return model, {
        "best": best_metrics,
        "best_epoch": int(np.argmax([h["macro_f1"] for h in history]) + 1),
        "history": history,
    }


@torch.inference_mode()
def prior_sample_test(cvae, blur_aware_model, device, n=64):
    """Matched-z: same z, three labels. This is the pilot-relevant test.

    For each of n random z vectors drawn from N(0, I), decode with each of
    the three class labels, classify all 3n decodes with the blur-aware
    evaluator. If the conditioning works, class-c decodes should be
    classified as c more often than chance.
    """
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(SEED)
        z = torch.randn(n, LATENT_DIM, device=device)
        preds_per_class = {c: [] for c in range(3)}
        for c in range(3):
            y = torch.full((n,), c, dtype=torch.long, device=device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                x_c = cvae.decode(z, y)
            x_c = x_c.float()
            preds = blur_aware_model(x_c).argmax(dim=1).cpu().numpy()
            preds_per_class[c] = preds.tolist()
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    # Compute how often class-c decodes are classified as c
    results = {}
    for c in range(3):
        preds = preds_per_class[c]
        correct = sum(1 for p in preds if p == c)
        results[CLASS_NAMES[c]] = {
            "n": n,
            "classified_as_self": correct,
            "accuracy": correct / n,
            "predicted_as": {CLASS_NAMES[i]: sum(1 for p in preds if p == i) for i in range(3)},
        }
    overall = sum(r["classified_as_self"] for r in results.values()) / (3 * n)
    return {
        "n_per_class": n,
        "overall_accuracy": overall,
        "per_class": results,
        "note": (
            "Same z, three labels. Chance accuracy is 1/3. If class-c "
            "accuracy is well above 1/3, prior samples are class-distinct."
        ),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading cVAE...")
    ckpt = torch.load(_CVAE_CKPT, map_location=device, weights_only=False)
    cvae = ConditionalVAE(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS)
    cvae.load_state_dict(ckpt["model_state_dict"])
    cvae.to(device).eval()

    # --- Data preparation ---
    print("\nBuilding training reconstruction set (15k subsample)...")
    train_ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    rng = np.random.default_rng(SEED)
    train_indices = rng.choice(len(train_ds), size=15000, replace=False).tolist()
    train_recons, train_labels = build_reconstruction_set(
        cvae, "train", device, indices=train_indices, label="train_recon"
    )
    print(f"  train_recons: {train_recons.shape}, dtype={train_recons.dtype}")

    # Sharp version of same 15k images for the matched control
    print("\nLoading same 15k sharp training images...")
    train_sharp = torch.stack([train_ds[i][0] for i in train_indices]).to(torch.float16)
    print(f"  train_sharp: {train_sharp.shape}")

    print("\nBuilding val reconstruction set (full 4,727)...")
    val_recons, val_labels = build_reconstruction_set(cvae, "val", device, label="val_recon")

    print("\nLoading val sharp images...")
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    val_sharp = torch.stack([val_ds[i][0] for i in range(len(val_ds))]).to(torch.float16)
    print(f"  val_sharp: {val_sharp.shape}")

    # --- Sharp evaluator (control) ---
    print("\n=== Training SHARP evaluator (matched control) ===")
    sharp_model, sharp_result = train_evaluator(
        train_sharp,
        train_labels,
        val_sharp,
        val_labels,
        device,
        label_prefix="sharp",
        epochs=10,
    )
    print(
        f"  Sharp best epoch {sharp_result['best_epoch']}: "
        f"acc={sharp_result['best']['accuracy']:.3f}, "
        f"bal_acc={sharp_result['best']['balanced_accuracy']:.3f}, "
        f"macro_f1={sharp_result['best']['macro_f1']:.3f}"
    )

    # --- Blur-aware evaluator ---
    print("\n=== Training BLUR-AWARE evaluator (on reconstructions) ===")
    blur_model, blur_result = train_evaluator(
        train_recons,
        train_labels,
        val_recons,
        val_labels,
        device,
        label_prefix="blur",
        epochs=10,
    )
    print(
        f"  Blur-aware best epoch {blur_result['best_epoch']}: "
        f"acc={blur_result['best']['accuracy']:.3f}, "
        f"bal_acc={blur_result['best']['balanced_accuracy']:.3f}, "
        f"macro_f1={blur_result['best']['macro_f1']:.3f}"
    )

    # --- Cross-domain check: blur-aware on sharp val ---
    print("\n=== Cross-domain: blur-aware evaluator on sharp val ===")
    cross_loader = DataLoader(
        list(zip(val_sharp, val_labels, strict=True)), batch_size=128, shuffle=False, num_workers=0
    )
    cross_metrics = _eval_metrics(blur_model, cross_loader, device)
    print(
        f"  blur-eval on sharp val: acc={cross_metrics['accuracy']:.3f}, "
        f"bal_acc={cross_metrics['balanced_accuracy']:.3f}, "
        f"macro_f1={cross_metrics['macro_f1']:.3f}"
    )

    # --- Ratio-based verdict ---
    sharp_bal = sharp_result["best"]["balanced_accuracy"]
    blur_bal = blur_result["best"]["balanced_accuracy"]
    ratio = blur_bal / sharp_bal if sharp_bal > 0 else 0.0
    print("\n=== Reconstruction signal ratio ===")
    print(f"  Sharp balanced acc: {sharp_bal:.3f}")
    print(f"  Blur-aware balanced acc: {blur_bal:.3f}")
    print(f"  Ratio (blur/sharp): {ratio:.3f}")

    if ratio >= 0.80:
        recon_verdict = (
            "SIGNAL SURVIVES: blur-aware balanced accuracy is >=80% of sharp "
            "baseline. The class signal survives reconstruction. Prior-sample "
            "test below decides whether it reaches the pilot."
        )
    elif ratio >= 0.60:
        recon_verdict = (
            "PARTIAL SIGNAL: blur removes some class signal (ratio 0.60-0.80). "
            "Prior samples may or may not be class-distinct."
        )
    else:
        recon_verdict = (
            "SIGNAL DEGRADED: blur-aware balanced accuracy is <60% of sharp "
            "baseline. Reconstruction removes substantial class signal."
        )
    print(f"  {recon_verdict}")

    # --- Prior-sample test (matched z, three labels) ---
    print("\n=== Prior-sample test (matched z, different labels) ===")
    prior_result = prior_sample_test(cvae, blur_model, device, n=64)
    print(
        f"  Overall prior-sample accuracy: {prior_result['overall_accuracy']:.3f} (chance = 0.333)"
    )
    for c_name, stats in prior_result["per_class"].items():
        print(f"  {c_name}: acc={stats['accuracy']:.3f}, predicted_as={stats['predicted_as']}")

    # --- Save everything ---
    out = {
        "note": (
            "Train reconstructions come from images the cVAE saw during training. "
            "Val reconstructions come from held-out val images. This is intentional: "
            "the evaluator is trained on 'typical reconstructions' and tested on "
            "held-out reconstructions — the same setup as a real classifier."
        ),
        "train_subsample_size": 15000,
        "sharp_control": {
            "best_epoch": sharp_result["best_epoch"],
            "metrics": sharp_result["best"],
            "history": sharp_result["history"],
        },
        "blur_aware": {
            "best_epoch": blur_result["best_epoch"],
            "metrics": blur_result["best"],
            "history": blur_result["history"],
        },
        "cross_domain_blur_on_sharp": cross_metrics,
        "reconstruction_ratio": ratio,
        "reconstruction_verdict": recon_verdict,
        "prior_samples": prior_result,
    }
    (_OUT_DIR / "blur_aware_diagnostic.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {_OUT_DIR / 'blur_aware_diagnostic.json'}")


if __name__ == "__main__":
    sys.exit(main())
