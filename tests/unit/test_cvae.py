"""Unit tests for the ConditionalVAE."""

from __future__ import annotations

import pytest
import torch

from urbangen.models.cvae.model import ConditionalVAE


def _make_batch(batch_size: int = 4, image_size: int = 128):
    x = torch.rand(batch_size, 3, image_size, image_size)
    labels = torch.tensor([0, 1, 2, 0][:batch_size], dtype=torch.long)
    return x, labels


def test_forward_shapes():
    model = ConditionalVAE(image_size=128)
    x, labels = _make_batch(4)
    recon, mu, logvar = model(x, labels)
    assert recon.shape == (4, 3, 128, 128)
    assert mu.shape == (4, 128)
    assert logvar.shape == (4, 128)


def test_recon_in_unit_interval():
    model = ConditionalVAE(image_size=128)
    x, labels = _make_batch(4)
    recon, _, _ = model(x, labels)
    assert recon.min() >= 0.0
    assert recon.max() <= 1.0


def test_conditioning_changes_output():
    """Different class labels must produce different samples at the same z."""
    model = ConditionalVAE(image_size=128).eval()
    z = torch.randn(1, 128)
    class_0 = torch.tensor([0])
    class_1 = torch.tensor([1])
    with torch.no_grad():
        out_0 = model.decode(z, class_0)
        out_1 = model.decode(z, class_1)
    assert not torch.allclose(out_0, out_1), "conditioning has no effect"


def test_gradients_flow():
    model = ConditionalVAE(image_size=128)
    x, labels = _make_batch(2)
    recon, mu, logvar = model(x, labels)
    loss = ((recon - x) ** 2).mean() + mu.pow(2).mean() + logvar.pow(2).mean()
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {name}"


def test_sample_shape():
    model = ConditionalVAE(image_size=128).eval()
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    with torch.no_grad():
        samples = model.sample(n=6, class_ids=labels)
    assert samples.shape == (6, 3, 128, 128)
    assert samples.min() >= 0.0
    assert samples.max() <= 1.0


def test_image_size_not_divisible_by_16_raises():
    with pytest.raises(ValueError, match="divisible by 16"):
        ConditionalVAE(image_size=100)


def test_sample_with_mismatched_labels_raises():
    model = ConditionalVAE(image_size=128).eval()
    with pytest.raises(ValueError, match="class_ids has length"), torch.no_grad():
        model.sample(n=4, class_ids=torch.tensor([0, 1, 2]))
