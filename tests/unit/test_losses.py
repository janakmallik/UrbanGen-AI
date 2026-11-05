"""Unit tests for training loss functions. Fast, run on CI."""

from __future__ import annotations

import pytest
import torch

from urbangen.training.losses import (
    CONTENT_BOTTOM,
    CONTENT_TOP,
    IMAGE_SIZE,
    content_mask,
    kl_divergence,
    masked_recon_mse,
    masked_recon_sse,
)


def test_content_mask_shape_and_values():
    mask = content_mask(2, torch.device("cpu"))
    assert mask.shape == (2, 1, IMAGE_SIZE, IMAGE_SIZE)
    # Content rows = 1
    assert mask[:, :, CONTENT_TOP:CONTENT_BOTTOM, :].all()
    # Padding rows = 0
    assert not mask[:, :, :CONTENT_TOP, :].any()
    assert not mask[:, :, CONTENT_BOTTOM:, :].any()


def test_recon_sse_zero_on_identical_content():
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert masked_recon_sse(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_recon_mse_zero_on_identical_content():
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert masked_recon_mse(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_recon_sse_ignores_padding():
    """Differences in padding must not affect the loss."""
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = x.clone()
    y[:, :, :CONTENT_TOP, :] = 0.9
    y[:, :, CONTENT_BOTTOM:, :] = 0.9
    # Both should be ~0 since only padding differs
    assert masked_recon_sse(y, x).item() == pytest.approx(0.0, abs=1e-6)
    assert masked_recon_mse(y, x).item() == pytest.approx(0.0, abs=1e-6)


def test_recon_sse_detects_content_differences():
    """Differences in content rows must produce positive loss."""
    x = torch.rand(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = x.clone()
    y[:, :, CONTENT_TOP:CONTENT_BOTTOM, :] += 0.1
    assert masked_recon_sse(y, x).item() > 0.0


def test_recon_sse_scale_is_large():
    """SSE should be in the hundreds/thousands range for non-trivial differences."""
    x = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    y = torch.ones(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    sse = masked_recon_sse(y, x).item()
    # Content pixels: 3 channels × 72 rows × 128 cols = 27,648
    # Each (1 - 0)^2 = 1, so SSE = 27,648
    assert sse == pytest.approx(27648.0, abs=1.0)


def test_kl_zero_at_standard_normal():
    mu = torch.zeros(4, 128)
    logvar = torch.zeros(4, 128)
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_positive_when_shifted():
    mu = torch.ones(4, 128) * 0.5
    logvar = torch.zeros(4, 128)
    assert kl_divergence(mu, logvar).item() > 0.0


def test_kl_aggregation_correct():
    """KL is mean over batch of sum over latent dims."""
    mu = torch.zeros(2, 128)
    logvar = torch.zeros(2, 128)
    # Both rows KL = 0
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.0, abs=1e-6)
    # One row has mu[0]=1: KL for that row = 0.5
    mu[0, 0] = 1.0
    # Batch mean = (0.5 + 0) / 2 = 0.25
    assert kl_divergence(mu, logvar).item() == pytest.approx(0.25, abs=1e-5)


def test_kl_extreme_logvar_no_overflow():
    """logvar=10 -> exp(10) ~ 22k, no overflow."""
    mu = torch.zeros(2, 128)
    logvar = torch.ones(2, 128) * 10.0
    kl = kl_divergence(mu, logvar)
    assert torch.isfinite(kl).all()
