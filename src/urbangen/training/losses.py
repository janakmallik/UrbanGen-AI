"""Pure loss functions for the cVAE.

Frozen reduction convention (see docs/experiments.md G1 pre-run amendment):

    recon_sse = sum over (channels × content_pixels) per sample, then mean over batch
    kl        = sum over latent_dims per sample, then mean over batch
    total     = recon_sse + beta * kl

Why sum-of-squares for reconstruction and not mean-squared-error:

    Per-pixel MSE for a well-trained VAE at 128×128 is roughly 0.01-0.05.
    The entire reconstruction gain available is therefore also small.
    KL is a sum over 128 latent dimensions, which is typically tens of nats
    even for a partially active posterior. If we optimize per-pixel MSE +
    KL directly, the optimizer's cheapest move is to drive KL to zero
    (posterior collapse) and accept a class-average blurry image.

    The ELBO in its correct form uses the sum of squared error over the
    data dimensions of each sample, which puts recon in the thousands.
    β = 1.0 is then a meaningful trade-off. We report per-pixel MSE for
    interpretability but optimize the summed form.

Content region: rows 28:100 of a 128-row image are real content; rows 0:28
and 100:128 are padding (black). Only the content region contributes to
the reconstruction loss.
"""

from __future__ import annotations

import torch

IMAGE_SIZE = 128
CONTENT_TOP = 28
CONTENT_BOTTOM = 100


def content_mask(batch_size: int, device: torch.device) -> torch.Tensor:
    """Return (B, 1, H, W) mask: 1 on content rows, 0 on padding."""
    mask = torch.zeros(batch_size, 1, IMAGE_SIZE, IMAGE_SIZE, device=device)
    mask[:, :, CONTENT_TOP:CONTENT_BOTTOM, :] = 1.0
    return mask


def masked_recon_sse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Sum of squared error over content pixels, averaged over batch.

    For each sample: sum over (channels × content_pixels).
    Then average over batch.

    This is the correct ELBO-scale reconstruction term. Value is typically
    in the hundreds-to-thousands range for a partially trained model.

    recon, target: (B, 3, 128, 128). Returns scalar.
    """
    mask = content_mask(recon.size(0), recon.device)
    diff_sq = (recon.float() - target.float()) ** 2
    masked = diff_sq * mask
    return masked.sum() / recon.size(0)


def masked_recon_mse(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-pixel mean squared error over content pixels. For logging only.

    Division: (B × channels × content_pixels). Value is in [0, 1] range,
    typically 0.01-0.06. NOT used for optimization.
    """
    mask = content_mask(recon.size(0), recon.device)
    diff_sq = (recon.float() - target.float()) ** 2
    masked = diff_sq * mask
    n_content_per_sample = 3 * (CONTENT_BOTTOM - CONTENT_TOP) * IMAGE_SIZE
    return masked.sum() / (recon.size(0) * n_content_per_sample)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0, I)): mean over batch of sum over latent dims.

    For each sample: KL = -0.5 * sum_j (1 + logvar_j - mu_j^2 - exp(logvar_j))
    Then average over batch.

    mu, logvar: (B, latent_dim). Returns scalar.
    Caller is responsible for casting to FP32 (for numerical stability
    under autocast).
    """
    kl_per_sample = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
    return kl_per_sample.mean()
