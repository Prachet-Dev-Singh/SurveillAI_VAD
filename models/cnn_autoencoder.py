"""
CNN Autoencoder baseline model for video anomaly detection.
"""

import torch
import torch.nn as nn


class CNNAutoencoder(nn.Module):
    """
    Simple CNN autoencoder for frame reconstruction.

    Encoder: Downsampling with convolutions
    Decoder: Upsampling with transpose convolutions
    """

    def __init__(self, latent_dim=256):
        """
        Args:
            latent_dim: Dimension of the bottleneck
        """
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            # Input: (B, C=3, H=224, W=224)
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 112, 112)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 56, 56)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 28, 28)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (B, 512, 14, 14)
            nn.ReLU(inplace=True),
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, latent_dim, kernel_size=1),  # (B, latent_dim, 14, 14)
            nn.Flatten(),
            nn.Linear(latent_dim * 14 * 14, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 14 * 14),
            nn.Unflatten(1, (latent_dim, 14, 14)),

            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=2, padding=1),  # (B, 512, 28, 28)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, 56, 56)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, 112, 112)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, 224, 224)
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=1),  # (B, 3, 224, 224)
            nn.Sigmoid(),  # Output in [0, 1] range
        )

    def encode(self, x):
        """Encode frame to latent vector."""
        features = self.encoder(x)
        # Reshape for bottleneck
        B, C, H, W = features.shape
        features = features.view(B, C, -1).mean(dim=2)  # Global average pooling
        z = self.bottleneck(features.unsqueeze(-1).unsqueeze(-1))
        return z

    def decode(self, z):
        """Decode latent vector to reconstructed frame."""
        return self.decoder(z)

    def forward(self, x):
        """
        Forward pass: reconstruct input frame.

        Args:
            x: Input tensor of shape (B, C, H, W) where C=3, H=W=224

        Returns:
            Reconstructed tensor of same shape
        """
        features = self.encoder(x)
        # Global average pooling
        z = features.view(features.size(0), -1).mean(dim=1, keepdim=True)
        z = z.expand(-1, 256)  # Expand to latent_dim

        return self.decoder(z)


class SimpleCNNAutoencoder(nn.Module):
    """Simplified CNN autoencoder for faster training."""

    def __init__(self, latent_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 28 * 28),
            nn.Unflatten(1, (128, 28, 28)),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) where C=3, H=W=224

        Returns:
            Reconstructed (B, C, H, W)
        """
        z = self.encoder(x)
        z = self.bottleneck(z)
        x_recon = self.decoder(z)

        # Resize to match input (in case of size mismatch)
        if x_recon.shape != x.shape:
            x_recon = torch.nn.functional.interpolate(
                x_recon, size=x.shape[-2:], mode='bilinear', align_corners=False
            )

        return x_recon
