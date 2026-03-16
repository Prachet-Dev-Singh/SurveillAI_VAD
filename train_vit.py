"""
Advanced training script supporting ViT, Temporal Transformer, and Self-Distillation.
Used for Week 2 experiments and ablation studies.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

from data.dataset import SlidingWindowDataset
from models.vit_branch import ViTSpatialEncoder
from models.temporal_transformer import TemporalTransformer
from models.decoder import ReconstructionDecoder
from models.self_distillation import (
    StudentEncoder, DistillationLoss, TeacherStudentWrapper, KnowledgeDistillationTrainer
)


class ViTWithTemporalModel(nn.Module):
    """ViT spatial encoder + Temporal transformer."""

    def __init__(self, freeze_blocks=8, use_temporal=True, embed_dim=256):
        super().__init__()

        self.spatial_encoder = ViTSpatialEncoder(
            freeze_blocks=freeze_blocks,
            embed_dim=384,
            output_dim=embed_dim
        )

        if use_temporal:
            self.temporal_encoder = TemporalTransformer(
                embed_dim=embed_dim,
                num_heads=4,
                num_layers=2,
                max_frames=8
            )
        else:
            self.temporal_encoder = None

        self.decoder = ReconstructionDecoder(latent_dim=embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: (B, N_frames, 3, 224, 224) or (B, 3, 224, 224)
        """
        # Handle both single frame and multi-frame input
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            # Encode spatially for each frame
            x_flat = x.view(B * N, C, H, W)
            spatial_feats = self.spatial_encoder(x_flat)
            spatial_feats = spatial_feats.view(B, N, self.embed_dim)

            # Temporal encoding
            if self.temporal_encoder:
                latent = self.temporal_encoder(spatial_feats)
            else:
                latent = spatial_feats.mean(dim=1)

        else:
            # Single frame
            latent = self.spatial_encoder(x)

        # Decode
        reconstructed = self.decoder(latent)

        return reconstructed

    def encode(self, x):
        """Get encoder output (for distillation or analysis)."""
        if x.dim() == 5:
            B, N, C, H, W = x.shape
            x_flat = x.view(B * N, C, H, W)
            spatial_feats = self.spatial_encoder(x_flat).view(B, N, self.embed_dim)
            if self.temporal_encoder:
                return self.temporal_encoder(spatial_feats)
            else:
                return spatial_feats.mean(dim=1)
        else:
            return self.spatial_encoder(x)


def load_config(config_path):
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_vit_epoch(model, train_loader, optimizer, device, use_clip=False):
    """Train ViT model for one epoch."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for clips in pbar:
        # clips: (B, N_frames, 3, 224, 224)
        clips = clips.to(device)
        target = clips[:, clips.shape[1] // 2, :, :, :]

        # Forward
        reconstructed = model(clips)

        # Loss
        loss = F.mse_loss(reconstructed, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        if use_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_vit(model, val_loader, device):
    """Validate ViT model."""
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for clips in val_loader:
            clips = clips.to(device)
            target = clips[:, clips.shape[1] // 2, :, :, :]

            reconstructed = model(clips)
            loss = F.mse_loss(reconstructed, target)

            total_loss += loss.item()

    return total_loss / len(val_loader)


def train_vit_model(config, model, train_loader, val_loader, device):
    """
    Train ViT model with temporal transformer.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {'train': [], 'val': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training ViT model for {config['epochs']} epochs...")

    for epoch in range(config['epochs']):
        train_loss = train_vit_epoch(model, train_loader, optimizer, device)
        val_loss = validate_vit(model, val_loader, device)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        print(f"Epoch [{epoch+1}/{config['epochs']}] "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/vit_best.pth')
            print(f"  -> Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, history


def train_with_distillation(config, teacher_model, train_loader, val_loader, device):
    """
    Train student model with knowledge distillation.
    """
    # Create student
    student = StudentEncoder(input_dim=config['latent_dim'], output_dim=config['latent_dim'])
    decoder = ReconstructionDecoder(latent_dim=config['latent_dim'])

    #Wrapper
    wrapper = TeacherStudentWrapper(teacher_model.spatial_encoder, student, decoder, freeze_teacher=True)
    wrapper.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(decoder.parameters()),
        lr=config['lr']
    )

    # Loss
    distill_loss_fn = DistillationLoss(temperature=4.0, alpha=0.5)

    history = {'train': [], 'val': [], 'distill': []}
    best_val_loss = float('inf')
    patience_counter = 0

    print("Training with distillation...")

    for epoch in range(config['epochs']):
        wrapper.train()
        total_loss = 0.0
        total_distill = 0.0

        for clips in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            clips = clips.to(device)
            target = clips[:, clips.shape[1] // 2, :, :, :]

            # Forward
            teacher_feat, student_feat, recon = wrapper(target)

            # Reconstruction loss
            recon_loss = F.mse_loss(recon, target)

            # Distillation loss
            total_loss_val, distill_loss_val, recon_l = distill_loss_fn(
                student_feat, teacher_feat, recon_loss
            )

            optimizer.zero_grad()
            total_loss_val.backward()
            optimizer.step()

            total_loss += total_loss_val.item()
            total_distill += distill_loss_val.item()

        train_loss = total_loss / len(train_loader)
        train_distill = total_distill / len(train_loader)

        # Validate
        wrapper.eval()
        val_loss = 0.0

        with torch.no_grad():
            for clips in val_loader:
                clips = clips.to(device)
                target = clips[:, clips.shape[1] // 2, :, :, :]

                teacher_feat, student_feat, recon = wrapper(target)
                recon_loss = F.mse_loss(recon, target)
                total_loss_val, distill_loss_val, _ = distill_loss_fn(student_feat, teacher_feat, recon_loss)

                val_loss += total_loss_val.item()

        val_loss = val_loss / len(val_loader)

        history['train'].append(train_loss)
        history['val'].append(val_loss)
        history['distill'].append(train_distill)

        print(f"Epoch {epoch+1} | Train: {train_loss:.6f} (distill: {train_distill:.6f}) | Val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            os.makedirs('checkpoints', exist_ok=True)
            torch.save(student.state_dict(), 'checkpoints/vitdistill_student_best.pth')
            torch.save(decoder.state_dict(), 'checkpoints/vitdistill_decoder_best.pth')
            print(f"  -> Best model saved")
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return student, decoder, history


def plot_history(history, title, output_path):
    """Plot training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Train', marker='o')
    plt.plot(history['val'], label='Val', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train ViT-based anomaly detection')
    parser.add_argument('--config', type=str, default='configs/vit.yaml',
                      help='Config file')
    parser.add_argument('--data_dir', type=str, default='data/processed/train',
                      help='Training data directory')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--use_distillation', action='store_true',
                      help='Use self-distillation')

    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device)

    # Load data
    print(f"Loading dataset from {args.data_dir}...")
    dataset = SlidingWindowDataset(
        frame_dir=args.data_dir,
        window_size=config['window_size'],
        stride=config['stride'],
        use_npy=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = ViTWithTemporalModel(
        freeze_blocks=config['freeze_blocks'],
        use_temporal=config.get('use_temporal', True),
        embed_dim=config['latent_dim']
    )
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Train
    if args.use_distillation:
        student, decoder, history = train_with_distillation(config, model, train_loader, val_loader, device)
        plot_history(history, 'ViT + Distillation Training', ' results/vit_distillation_history.png')
    else:
        model, history = train_vit_model(config, model, train_loader, val_loader, device)
        plot_history(history, 'ViT + Temporal Training', 'results/vit_training_history.png')

    print("Training complete!")


if __name__ == '__main__':
    main()
