"""
Main training script for video anomaly detection models.
Supports CNN, ViT, and Mamba architectures.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

from data.dataset import SlidingWindowDataset
from models.cnn_autoencoder import SimpleCNNAutoencoder


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return type('Config', (), config)()


def create_model(model_type, latent_dim=256, device='cpu'):
    """Create model based on type."""
    if model_type == 'cnn':
        model = SimpleCNNAutoencoder(latent_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def train_epoch(model, train_loader, optimizer, device, use_clip=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch_idx, clips in enumerate(pbar):
        # clips shape: (B, N_frames, C, H, W)
        clips = clips.to(device)

        # Get middle frame as target
        target = clips[:, clips.shape[1] // 2, :, :, :]  # (B, C, H, W)

        # For CNN baseline: pass middle frame to model
        # TODO: For ViT/Temporal models, will need to handle the full clip

        reconstructed = model(target)

        # Calculate MSE loss
        loss = F.mse_loss(reconstructed, target)

        optimizer.zero_grad()
        loss.backward()

        if use_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.update(1)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device):
    """Validate model and return average loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for clips in val_loader:
            clips = clips.to(device)
            target = clips[:, clips.shape[1] // 2, :, :, :]

            reconstructed = model(target)
            loss = F.mse_loss(reconstructed, target)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def train_model(config, model, train_loader, val_loader, device):
    """
    Main training loop.

    Returns:
        model: Trained model
        history: Dict with train/val losses
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20

    print(f"Training {config.model} model for {config.epochs} epochs...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history['train_loss'].append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device)
        history['val_loss'].append(val_loss)

        print(f"Epoch [{epoch+1}/{config.epochs}] "
              f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            checkpoint_path = f"checkpoints/{config.model}_best.pth"
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return model, history


def plot_training_history(history, model_name):
    """Plot and save training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'{model_name.upper()} Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_training_history.png', dpi=100, bbox_inches='tight')
    print(f"Saved plot to results/{model_name}_training_history.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train video anomaly detection model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, default='data/processed/train',
                      help='Path to processed training data')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'],
                      help='Device to use for training')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config.device = args.device

    if config.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = 'cpu'

    device = torch.device(config.device)
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")

    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        print("You need to preprocess the UCSD Ped2 dataset first:")
        print("  python data/preprocess.py --dataset ucsd --input data/ucsd/ --output data/processed/")
        sys.exit(1)

    dataset = SlidingWindowDataset(
        frame_dir=args.data_dir,
        window_size=config.window_size,
        stride=config.stride,
        use_npy=True
    )

    print(f"Found {len(dataset)} clips")

    if len(dataset) == 0:
        print("Error: No data found. Make sure frames are stored as .npy files in:")
        print(f"  {args.data_dir}/")
        sys.exit(1)

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Train set: {len(train_dataset)} clips")
    print(f"Val set: {len(val_dataset)} clips")

    # Create model
    model = create_model(config.model, latent_dim=config.latent_dim, device=device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.model} | Parameters: {num_params:,}")

    # Train
    model, history = train_model(config, model, train_loader, val_loader, device)

    # Save final model
    os.makedirs('checkpoints', exist_ok=True)
    final_checkpoint = f"checkpoints/{config.model}_final.pth"
    torch.save(model.state_dict(), final_checkpoint)
    print(f"Saved final model to {final_checkpoint}")

    # Plot history
    plot_training_history(history, config.model)

    print("\nTraining complete!")
    print(f"Best validation loss: {min(history['val_loss']):.6f}")


if __name__ == '__main__':
    main()
