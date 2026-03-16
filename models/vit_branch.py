"""
ViT Spatial Encoder for video anomaly detection.
Uses pretrained Vision Transformer to extract spatial features from frames.
"""

import torch
import torch.nn as nn
import timm


class ViTSpatialEncoder(nn.Module):
    """
    Pretrained ViT-S/16 spatial encoder.

    Freezes early layers and fine-tunes later layers.
    Outputs a fixed-size representation per frame.
    """

    def __init__(self, freeze_blocks=8, embed_dim=384, output_dim=256):
        """
        Args:
            freeze_blocks: Number of early transformer blocks to freeze
            embed_dim: ViT embedding dimension (384 for ViT-S/16)
            output_dim: Output feature dimension
        """
        super().__init__()

        # Load pretrained ViT-S/16
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=0,  # Return patch embeddings, not classification logits
        )

        # Freeze early blocks for efficient transfer learning
        for i, block in enumerate(self.vit.blocks):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        # Projection to output dimension
        self.proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """
        Extract spatial features from frame(s).

        Args:
            x: Input tensor of shape (B, 3, 224, 224)

        Returns:
            Spatial features of shape (B, output_dim)
        """
        # Forward through ViT: (B, 3, 224, 224) -> (B, num_patches+1, embed_dim)
        # ViT outputs: [CLS_token, patch_1, patch_2, ..., patch_196]
        features = self.vit.forward_features(x)

        # Remove CLS token and pool over patches
        patch_features = features[:, 1:, :]  # (B, num_patches=196, embed_dim)

        # Global average pooling over patches
        pooled = patch_features.mean(dim=1)  # (B, embed_dim)

        # Project to output dimension
        output = self.proj(pooled)  # (B, output_dim)

        return output

    def get_num_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ViTSpatialEncoderWithPatchFeatures(nn.Module):
    """
    ViT encoder that returns per-patch features for attention visualization.
    Useful for generating heatmaps.
    """

    def __init__(self, freeze_blocks=8, embed_dim=384, output_dim=256):
        super().__init__()

        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=True,
            num_classes=0,
        )

        for i, block in enumerate(self.vit.blocks):
            if i < freeze_blocks:
                for param in block.parameters():
                    param.requires_grad = False

        self.proj = nn.Linear(embed_dim, output_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Returns both pooled features and patch features.

        Args:
            x: Input tensor (B, 3, 224, 224)

        Returns:
            pooled: (B, output_dim)
            patch_features: (B, num_patches, embed_dim)
            attention_weights: Attention maps for visualization
        """
        # Get features
        features = self.vit.forward_features(x)
        patch_features = features[:, 1:, :]

        # Pool
        pooled = patch_features.mean(dim=1)
        output = self.proj(pooled)

        # For attention visualization, we can extract attention maps
        # from the last block (this is a simplified version)
        attention_weights = None

        return output, patch_features, attention_weights
