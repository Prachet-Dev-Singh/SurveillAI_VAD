"""
Temporal Transformer for video anomaly detection.
Captures temporal dependencies across frames using multi-head attention.
"""

import torch
import torch.nn as nn
import math


class TemporalTransformer(nn.Module):
    """
    Lightweight transformer for temporal modeling.

    Input: Sequence of spatial features from N frames
    Output: Single aggregated representation
    """

    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, max_frames=8):
        """
        Args:
            embed_dim: Feature dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            max_frames: Maximum sequence length
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_frames, embed_dim) * 0.02
        )

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1,
            batch_first=True,
            activation='relu'
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):
        """
        Process temporal sequence of features.

        Args:
            x: Sequence of spatial features
               Shape: (B, N_frames, embed_dim)

        Returns:
            Aggregated temporal representation
            Shape: (B, embed_dim)
        """
        B, N, D = x.shape

        # Add positional embeddings
        pos_embed = self.pos_embed[:, :N, :]  # (1, N, embed_dim)
        x = x + pos_embed  # (B, N, embed_dim)

        # Apply transformer
        x = self.transformer_encoder(x)  # (B, N, embed_dim)

        # Aggregate over time (mean pooling)
        output = x.mean(dim=1)  # (B, embed_dim)

        return output


class TemporalAttentionPooling(nn.Module):
    """
    Temporal attention pooling for better temporal aggregation.

    Learn to weight frames based on their importance.
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Args:
            x: (B, N_frames, embed_dim)

        Returns:
            Weighted aggregation (B, embed_dim)
        """
        # Compute attention weights
        weights = self.attention(x)  # (B, N, 1)

        # Weighted sum
        output = (x * weights).sum(dim=1)  # (B, embed_dim)

        return output


class TemporalConvolution(nn.Module):
    """
    Alternative: Use 1D convolutions for temporal modeling.
    Faster than transformer but potentially less expressive.
    """

    def __init__(self, embed_dim=256, kernel_size=3, num_layers=2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            in_channels = embed_dim
            out_channels = embed_dim

            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    bias=True
                )
            )
            layers.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, N_frames, embed_dim)

        Returns:
            (B, embed_dim)
        """
        # Transpose for conv1d: (B, N, D) -> (B, D, N)
        x = x.permute(0, 2, 1)

        # Apply convolution
        x = self.net(x)  # (B, embed_dim, N)

        # Aggregate: (B, embed_dim, N) -> (B, embed_dim)
        output = x.mean(dim=2)

        return output
