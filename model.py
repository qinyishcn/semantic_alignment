"""
Encoder-Decoder Neural Network for Semantic Alignment Verification.

Architecture:
  Encoder: 784 -> 256 -> 128 -> latent_dim
  Decoder: latent_dim -> 128 -> 256 -> 10

The bottleneck (latent_dim) is the "semantic interface" between encoder and decoder.
Two independently trained instances will have different latent representations,
demonstrating that encoder-decoder pairing is not interchangeable without alignment.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))


class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10),
        )

    def forward(self, z):
        return self.net(z)


class AutoencoderClassifier(nn.Module):
    """Full encoder-decoder model for digit classification."""

    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


class CrossModel(nn.Module):
    """A model built from encoder of one instance and decoder of another."""

    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
