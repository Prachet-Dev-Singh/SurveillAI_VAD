"""
Shared reconstruction decoder for all architectures.
Takes a latent vector and reconstructs a frame.
"""

import torch
import torch.nn as nn


class ReconstructionDecoder(nn.Module):
    """
    Decoder that reconstructs a frame from latent representation.

    Input: (B, latent_dim)
    Output: (B, 3, 224, 224)
    """

    def __init__(self, latent_dim=256):
        super().__init__()

        self.decoder = nn.Sequential(
            # Expand latent to spatial: (B, latent_dim) -> (B, 512, 7, 7)
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU(inplace=True),

            # Reshape to (B, 512, 7, 7)
            nn.Unflatten(1, (512, 7, 7)),

            # Upsample: 7*7 -> 14*14
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Upsample: 14*14 -> 28*28
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Upsample: 28*28 -> 56*56
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Upsample: 56*56 -> 112*112
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Upsample: 112*112 -> 224*224
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def forward(self, z):
        """
        Args:
            z: Latent vector of shape (B, latent_dim)

        Returns:
            Reconstructed frame of shape (B, 3, 224, 224)
        """
        return self.decoder(z)
