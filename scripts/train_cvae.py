"""Train the conditional VAE on the BDD100K 128x128 cache.

Frozen configuration (docs/experiments.md, G1 scoping + pre-run amendment):
    image_size=128, latent_dim=128, channels=(32, 64, 128, 256)
    batch_size=32, epochs=30, beta=1.0, seed=42
    optimizer=AdamW(lr=1e-3, wd=1e-4), cosine annealing
    AMP FP16, KL in FP32, grad-clip 1.0

Loss (pre-run amendment — see docs/experiments.md):
    recon_sse = sum over (channels × content pixels) per sample, mean over batch
    kl        = sum over latent dims per sample, mean over batch
    total     = recon_sse + beta * kl

Diagnostics:
    Per-epoch logging of recon_sse, kl, kl_active_dims (dimensions with
    std(mu) > 0.01 across the val set), and recon_mse (per-pixel, for
    interpretability only).

Safety:
    - Non-finite loss raises RuntimeError.
    - Gradient clipping at norm 1.0.
    - Sample generation saves/restores RNG state (no train-shuffle corruption).
    - MLflow run ends in finally block.

No early stopping. Full 30 epochs unless NaN or CUDA error.

Run:
    uv run python scripts/train_cvae.py --epochs 30
    uv run python scripts/train_cvae.py --smoke  # ~30 s pipeline test
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
os.environ.setdefault(
    "MLFLOW_TRACKING_URI",
    f"sqlite:///{(_REPO_ROOT / 'mlflow.db').as_posix()}",
)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from urbangen.data.bdd100k_generative import BDD100K128Dataset  # noqa: E402
from urbangen.models.cvae import ConditionalVAE  # noqa: E402
from urbangen.training.losses import (  # noqa: E402
    kl_divergence,
    masked_recon_mse,
    masked_recon_sse,
)

# --- Frozen hyperparameters ---
IMAGE_SIZE = 128
LATENT_DIM = 128
CHANNELS = (32, 64, 128, 256)
BATCH_SIZE = 32
BETA = 1.0
SEED = 42
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
NUM_EPOCHS = 30
NUM_WORKERS = 0
SMOKE_MAX_BATCHES = 5

_CACHE_ROOT = _REPO_ROOT / "data" / "bdd100k_128"
_CKPT_DIR = _REPO_ROOT / "checkpoints"
_SAMPLES_DIR = _REPO_ROOT / "outputs" / "cvae_beta1_seed42" / "samples"


def save_checkpoint(
    path: Path,
    model: ConditionalVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def compute_active_dims(
    model: ConditionalVAE, loader: DataLoader, device: torch.device
) -> tuple[int, int]:
    """Count latent dims with std(mu) > 0.01 across the validation set.

    A dim with near-zero variance across many samples contributes nothing
    to the encoded representation (posterior collapse). Standard diagnostic
    for VAE health.

    Returns (n_active_dims, total_dims).
    """
    model.eval()
    all_mu = []
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mu, _ = model.encode(x, labels)
            all_mu.append(mu.float().cpu())
    model.train()
    mu_cat = torch.cat(all_mu, dim=0)  # (N, latent_dim)
    stds = mu_cat.std(dim=0)  # (latent_dim,)
    return int((stds > 0.01).sum().item()), mu_cat.size(1)


@torch.no_grad()
def validate(model: ConditionalVAE, loader: DataLoader, device: torch.device) -> dict:
    """Deterministic validation: use mu (not sampled z) for val loss."""
    model.eval()
    total_recon_sse = 0.0
    total_recon_mse = 0.0
    total_kl = 0.0
    total_n = 0
    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            mu, logvar = model.encode(x, labels)
        # Deterministic recon: use mean, not sampled z
        with torch.amp.autocast("cuda", enabled=False):
            recon = model.decode(mu, labels)
            recon_sse = masked_recon_sse(recon, x)
            recon_mse = masked_recon_mse(recon, x)
            kl = kl_divergence(mu.float(), logvar.float())
        n = x.size(0)
        total_recon_sse += recon_sse.item() * n
        total_recon_mse += recon_mse.item() * n
        total_kl += kl.item() * n
        total_n += n
    model.train()
    return {
        "recon_sse": total_recon_sse / total_n,
        "recon_mse": total_recon_mse / total_n,
        "kl": total_kl / total_n,
        "total": (total_recon_sse / total_n) + BETA * (total_kl / total_n),
    }


def make_sample_grid(model: ConditionalVAE, n_per_class: int, device: torch.device) -> torch.Tensor:
    """Sample grid. Isolates RNG state so training shuffle is unaffected.

    Returns (3 * n_per_class, 3, 128, 128) tensor.
    """
    model.eval()
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        all_samples = []
        for class_id in range(3):
            class_ids = torch.full((n_per_class,), class_id, dtype=torch.long, device=device)
            samples = model.sample(n_per_class, class_ids, device=device)
            all_samples.append(samples.cpu())
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
    model.train()
    return torch.cat(all_samples, dim=0)


def save_sample_grid(samples: torch.Tensor, path: Path) -> None:
    """Save samples as a PNG grid. Padding rows are zeroed for clarity."""
    import torchvision.utils as vutils

    # Zero the padding rows so generated artifacts aren't shown as content
    samples = samples.clone()
    samples[:, :, :28, :] = 0.0
    samples[:, :, 100:, :] = 0.0

    n = samples.size(0) // 3
    grid = vutils.make_grid(samples, nrow=n, padding=2, normalize=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, str(path))


def train_one_epoch(
    model: ConditionalVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_batches: int | None = None,
    epoch_label: str = "",
) -> dict:
    model.train()
    total_recon_sse = 0.0
    total_recon_mse = 0.0
    total_kl = 0.0
    total_n = 0
    skipped_steps = 0

    it = itertools.islice(loader, max_batches) if max_batches else loader
    total_iter = max_batches or len(loader)
    pbar = tqdm(it, desc=f"  train {epoch_label}", leave=False, total=total_iter)

    for x, labels in pbar:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            recon, mu, logvar = model(x, labels)
        # Loss terms in FP32 outside autocast for numerical stability
        with torch.amp.autocast("cuda", enabled=False):
            recon_sse = masked_recon_sse(recon, x)
            recon_mse = masked_recon_mse(recon, x)
            kl = kl_divergence(mu.float(), logvar.float())
        loss = recon_sse + BETA * kl

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at {epoch_label}: {loss.item()}. "
                f"recon_sse={recon_sse.item()}, kl={kl.item()}. "
                f"Halting training."
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < scale_before:
            skipped_steps += 1

        n = x.size(0)
        total_recon_sse += recon_sse.item() * n
        total_recon_mse += recon_mse.item() * n
        total_kl += kl.item() * n
        total_n += n
        pbar.set_postfix(
            recon_mse=f"{recon_mse.item():.4f}",
            kl=f"{kl.item():.1f}",
        )
    pbar.close()

    return {
        "recon_sse": total_recon_sse / total_n,
        "recon_mse": total_recon_mse / total_n,
        "kl": total_kl / total_n,
        "total": (total_recon_sse / total_n) + BETA * (total_kl / total_n),
        "skipped_steps": skipped_steps,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            f"Smoke test: 1 epoch, {SMOKE_MAX_BATCHES} batches, exercises "
            f"checkpoint save/load/grid but writes to a temp dir"
        ),
    )
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke:
        ckpt_dir = _REPO_ROOT / "tmp" / "smoke_ckpt"
        samples_dir = _REPO_ROOT / "tmp" / "smoke_samples"
        print(f"SMOKE MODE — writing to {ckpt_dir}, {samples_dir}")
    else:
        ckpt_dir = _CKPT_DIR
        samples_dir = _SAMPLES_DIR

    if not (_CACHE_ROOT / "manifest.json").exists():
        print(f"ERROR: cache not found at {_CACHE_ROOT}.")
        return 1

    print("Loading datasets...")
    train_ds = BDD100K128Dataset(_CACHE_ROOT, "train")
    val_ds = BDD100K128Dataset(_CACHE_ROOT, "val")
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    print(f"  train: {len(train_ds):,}  val: {len(val_ds):,}")

    print("Initializing model...")
    model = ConditionalVAE(image_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS).to(
        device
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    config = {
        "image_size": IMAGE_SIZE,
        "latent_dim": LATENT_DIM,
        "channels": list(CHANNELS),
        "batch_size": BATCH_SIZE,
        "beta": BETA,
        "seed": SEED,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "epochs": args.epochs,
        "loss_convention": "recon_sse + beta*kl",
    }

    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"  resumed at epoch {start_epoch}")

    use_mlflow = not args.smoke
    if use_mlflow:
        import mlflow

        mlflow.set_experiment("urbangen-g1b")
        mlflow.start_run()
        mlflow.log_params(config)

    max_batches = SMOKE_MAX_BATCHES if args.smoke else None
    epochs_to_run = 1 if args.smoke else args.epochs

    try:
        print(f"\nStarting training for {epochs_to_run} epochs...")
        for epoch in range(start_epoch, epochs_to_run):
            t0 = time.perf_counter()
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                max_batches=max_batches,
                epoch_label=f"{epoch + 1}/{epochs_to_run}",
            )
        if train_metrics.get("skipped_steps", 0) > 0:
            print(
                f"  WARNING: {train_metrics['skipped_steps']} batches had "
                f"skipped optimizer steps (GradScaler found inf/nan). "
                f"Current scaler scale: {scaler.get_scale():.0f}"
            )

            val_metrics = validate(model, val_loader, device)
            active_dims, total_dims = compute_active_dims(model, val_loader, device)
            scheduler.step()
            elapsed = time.perf_counter() - t0

            print(
                f"epoch {epoch + 1:2d}/{epochs_to_run}  "
                f"train(recon_mse={train_metrics['recon_mse']:.4f}, "
                f"kl={train_metrics['kl']:.1f}, "
                f"total={train_metrics['total']:.1f})  "
                f"val(recon_mse={val_metrics['recon_mse']:.4f}, "
                f"kl={val_metrics['kl']:.1f}, "
                f"total={val_metrics['total']:.1f})  "
                f"active_dims={active_dims}/{total_dims}  "
                f"time={elapsed:.1f}s"
            )

            if use_mlflow:
                import mlflow

                mlflow.log_metrics(
                    {
                        "train_recon_sse": train_metrics["recon_sse"],
                        "train_recon_mse": train_metrics["recon_mse"],
                        "train_kl": train_metrics["kl"],
                        "train_total": train_metrics["total"],
                        "val_recon_sse": val_metrics["recon_sse"],
                        "val_recon_mse": val_metrics["recon_mse"],
                        "val_kl": val_metrics["kl"],
                        "val_total": val_metrics["total"],
                        "active_dims": active_dims,
                        "lr": scheduler.get_last_lr()[0],
                    },
                    step=epoch,
                )

            # Save best
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                save_checkpoint(
                    ckpt_dir / "cvae_beta1_seed42.pt",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    best_val_loss,
                    config,
                )
                print(f"  -> saved best checkpoint (val={best_val_loss:.1f})")

            # Save last
            save_checkpoint(
                ckpt_dir / "cvae_beta1_seed42_last.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_val_loss,
                config,
            )

            # Sample grid every 5 epochs + last (or every epoch in smoke)
            if args.smoke or (epoch + 1) % 5 == 0 or (epoch + 1) == epochs_to_run:
                samples = make_sample_grid(model, n_per_class=4, device=device)
                grid_path = samples_dir / f"epoch_{epoch + 1:03d}.png"
                save_sample_grid(samples, grid_path)
                print(f"  -> saved sample grid {grid_path}")

        # Verify checkpoint round-trip (only in smoke)
        if args.smoke:
            ckpt_path = ckpt_dir / "cvae_beta1_seed42.pt"
            assert ckpt_path.exists(), f"Smoke test failed: {ckpt_path} not created"
            loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
            assert "model_state_dict" in loaded
            assert "optimizer_state_dict" in loaded
            assert "config" in loaded
            print("  -> smoke: checkpoint round-trip verified")

    finally:
        if use_mlflow:
            import mlflow

            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.end_run()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.1f}")
    if args.smoke:
        print("SMOKE PASSED — pipeline verified end-to-end")
    return 0


if __name__ == "__main__":
    sys.exit(main())
