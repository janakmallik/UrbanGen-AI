"""Tests for train_cvae.py's script-level helpers.

Pure loss functions are tested in test_losses.py. This file tests the
script's utilities: sample grid shape, checkpoint round-trip, RNG isolation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from train_cvae import make_sample_grid  # noqa: E402
from urbangen.models.cvae import ConditionalVAE  # noqa: E402


def test_sample_grid_shape():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(image_size=128).to(device).eval()
    grid = make_sample_grid(model, n_per_class=2, device=device)
    assert grid.shape == (6, 3, 128, 128)
    assert grid.min() >= 0.0
    assert grid.max() <= 1.0


def test_sample_grid_does_not_change_rng():
    """Sample grid generation must not alter RNG state for subsequent training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalVAE(image_size=128).to(device).eval()

    # Set RNG and record next random value
    torch.manual_seed(42)
    expected = torch.rand(5)

    # Reset RNG, generate grid, then draw random
    torch.manual_seed(42)
    make_sample_grid(model, n_per_class=2, device=device)
    actual = torch.rand(5)

    assert torch.equal(expected, actual), "RNG state was corrupted by sample grid"


def test_checkpoint_roundtrip(tmp_path):
    model = ConditionalVAE(image_size=128)
    path = tmp_path / "test.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)

    model2 = ConditionalVAE(image_size=128)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model2.load_state_dict(ckpt["model_state_dict"])

    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.equal(p1, p2)
