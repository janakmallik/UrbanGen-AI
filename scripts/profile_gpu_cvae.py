"""G1 GPU profiling for the conditional VAE.

Profiles a complete encoder → latent → decoder cVAE training step at
several resolution × batch-size combinations. This is a profiling-only
model — channel widths match the planned cVAE, but it is not the final
implementation.

Tests forward + backward pass at:
    - 64x64, batch 32
    - 64x64, batch 64
    - 64x64, batch 128
    - 128x128, batch 16
    - 128x128, batch 32

Reports peak VRAM (torch.cuda.max_memory_allocated), ms/step, and
compute-only epoch estimate for each configuration. Epoch time excludes
DataLoader overhead (image decoding, augmentation, disk I/O) — treat it
as a floor.

Run:
    uv run python scripts/profile_gpu_cvae.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


class ProfileVAE(nn.Module):
    """Complete encoder → latent → decoder cVAE.

    Channel widths match the planned cVAE. Resolution determines the
    flattened feature-map size, so the linear layers are constructed
    eagerly in __init__ (not lazily on first forward) — this ensures
    optimizer = model.parameters() captures every trainable tensor.
    """

    def __init__(self, resolution: int, latent_dim: int = 128) -> None:
        super().__init__()
        feat = resolution // 16  # 4 stride-2 convs
        flat_dim = 256 * feat * feat

        self.encoder = nn.Sequential(
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
        )
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        self.feat = feat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        flat = torch.flatten(h, 1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        d = self.fc_dec(z).view(-1, 256, self.feat, self.feat)
        recon = self.decoder(d)
        return recon, mu, logvar


def profile_one(resolution: int, batch_size: int, warmup: int = 3, steps: int = 10) -> dict:
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = ProfileVAE(resolution=resolution).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch_size, 3, resolution, resolution, device=device)

    def step() -> torch.Tensor:
        recon, mu, logvar = model(x)
        recon_loss = (recon - x).pow(2).mean()
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 0.001 * kl

    for _ in range(warmup):
        loss = step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for _ in range(steps):
        loss = step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    sec_per_step = (t1 - t0) / steps
    steps_per_epoch = math.ceil(68_969 / batch_size)
    epoch_seconds = steps_per_epoch * sec_per_step

    return {
        "resolution": resolution,
        "batch_size": batch_size,
        "peak_vram_gb": peak_gb,
        "ms_per_step": sec_per_step * 1000,
        "warmup_steps": warmup,
        "measured_steps": steps,
        "est_compute_only_epoch_min": epoch_seconds / 60,
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available. Cannot profile.")
        return 1

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Total VRAM: {total_vram_gb:.2f} GB")
    print()

    configs = [
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 16),
        (128, 32),
    ]

    results: list[dict] = []
    for res, bs in configs:
        print(f"Profiling resolution={res}, batch={bs} ...")
        try:
            r = profile_one(res, bs)
            results.append(r)
            print(
                f"  peak VRAM: {r['peak_vram_gb']:.2f} GB | "
                f"{r['ms_per_step']:.1f} ms/step | "
                f"~{r['est_compute_only_epoch_min']:.1f} min/epoch (compute only)"
            )
        except torch.cuda.OutOfMemoryError:
            print("  OOM — skipping")
            torch.cuda.empty_cache()

    print()
    print("=" * 80)
    print(
        f"{'resolution':>12} | {'batch':>6} | {'VRAM GB':>8} | "
        f"{'ms/step':>9} | {'epoch min (compute)':>20}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['resolution']:>12} | {r['batch_size']:>6} | "
            f"{r['peak_vram_gb']:>8.2f} | {r['ms_per_step']:>9.1f} | "
            f"{r['est_compute_only_epoch_min']:>20.1f}"
        )

    print()
    print("Notes:")
    print("  - 'epoch min (compute)' excludes DataLoader overhead (image")
    print("    decoding, augmentation, disk I/O). Real epoch time will be")
    print("    higher — treat this as a floor, not a target.")
    print("  - Aim for peak VRAM <= ~5.5 GB to leave practical headroom.")
    print("    Other GPU processes and OS overhead may consume ~0.5 GB.")
    print("  - Prefer the smaller resolution unless the larger one offers")
    print("    a meaningful quality benefit worth the extra cost.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
