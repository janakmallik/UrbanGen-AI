"""Conditional VAE for 128x128 RGB images with class conditioning.

Architecture (frozen in docs/experiments.md G1 scoping entry):
    Encoder: 3->32->64->128->256 channels, 4 stride-2 convs
             -> flatten (256*8*8=16384) -> fc_mu, fc_logvar (both 128)
    Latent:  128 dim, standard reparameterization
    Decoder: fc_dec (128 -> 256*8*8) -> 4 transposed convs
             (256->128->64->32->3), Sigmoid output
    Conditioning: class embedding (dim 16) concatenated with flattened
                  encoder features (before mu/logvar) and with latent z
                  (before fc_dec).

The class embedding at both ends gives the model both class-informed
encoder features and class-informed decoder input.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):
    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 128,
        num_classes: int = 3,
        class_embed_dim: int = 16,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
    ):
        super().__init__()
        if image_size % 16 != 0:
            raise ValueError(f"image_size must be divisible by 16, got {image_size}")

        c1, c2, c3, c4 = channels
        feat = image_size // 16  # 4 stride-2 convs -> /16
        flat_dim = c4 * feat * feat
        self.feat = feat
        self.feat_channels = c4
        self.latent_dim = latent_dim
        self.class_embed_dim = class_embed_dim

        # Class embedding, shared across encoder/decoder conditioning
        self.class_embed = nn.Embedding(num_classes, class_embed_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3, c4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )
        self.fc_mu = nn.Linear(flat_dim + class_embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim + class_embed_dim, latent_dim)

        # Decoder
        self.fc_dec = nn.Linear(latent_dim + class_embed_dim, flat_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, stride=2, padding=1),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c1, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor, class_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        flat = torch.flatten(h, 1)
        c_emb = self.class_embed(class_ids)
        flat_c = torch.cat([flat, c_emb], dim=1)
        mu = self.fc_mu(flat_c)
        logvar = self.fc_logvar(flat_c)
        # Clamp logvar to prevent exp() overflow and stabilize early training.
        # Range [-10, 10] allows std in [e^-5, e^5] which is ample for this
        # task; values outside would indicate a broken forward pass.
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, class_ids: torch.Tensor) -> torch.Tensor:
        c_emb = self.class_embed(class_ids)
        z_c = torch.cat([z, c_emb], dim=1)
        h = self.fc_dec(z_c).view(-1, self.feat_channels, self.feat, self.feat)
        return self.decoder(h)

    def forward(
        self, x: torch.Tensor, class_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, class_ids)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, class_ids)
        return recon, mu, logvar

    @torch.no_grad()
    def sample(
        self, n: int, class_ids: torch.Tensor, device: torch.device | None = None
    ) -> torch.Tensor:
        """Generate n samples conditioned on class_ids (shape (n,))."""
        if len(class_ids) != n:
            raise ValueError(f"n={n} but class_ids has length {len(class_ids)}")
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, class_ids.to(device))
