from train_vit import ViTWithTemporalModel
"""
Evaluation script for video anomaly detection models.
Computes AUC-ROC, reconstruction errors, and generates scores.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

from data.dataset import SlidingWindowDatasetWithLabels, SlidingWindowDataset
from models.cnn_autoencoder import SimpleCNNAutoencoder


def load_model(model_type, checkpoint_path, latent_dim=256, device='cpu'):
    """Load trained model from checkpoint."""
    if model_type == 'cnn':
        model = SimpleCNNAutoencoder(latent_dim=latent_dim)
    elif model_type == 'vit':
        model = ViTWithTemporalModel(freeze_blocks=0, use_temporal=True, embed_dim=latent_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def compute_anomaly_scores(model, data_loader, device):
    """
    Compute per-frame reconstruction errors.

    Returns:
        all_scores: Array of reconstruction errors (lower = normal, higher = anomalous)
    """
    all_scores = []

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            # Handle both (clips) and (clips, labels) cases
            if isinstance(data, (list, tuple)):
                clips = data[0]
            else:
                clips = data

            clips = clips.to(device)

            # Get middle frame as target
            target = clips[:, clips.shape[1] // 2, :, :, :]  # (B, C, H, W)

            # Reconstruct
            reconstructed = model(target)

            # Compute per-frame MSE (pixel-wise)
            errors = F.mse_loss(reconstructed, target, reduction='none')

            # Average over spatial dimensions: (B, C, H, W) -> (B,)
            frame_errors = errors.mean(dim=[1, 2, 3]).cpu().numpy()

            all_scores.append(frame_errors)

    return np.concatenate(all_scores, axis=0)


def compute_auc_roc(model, test_loader, gt_labels, device):
    """
    Compute AUC-ROC score.

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        gt_labels: Ground truth labels (0=normal, 1=anomaly)
        device: Device to use

    Returns:
        auc_score: AUC-ROC score (0 to 1)
        scores: Per-sample reconstruction errors
        fpr, tpr, thresholds: ROC curve values
    """
    scores = compute_anomaly_scores(model, test_loader, device)

    # Compute AUC
    auc_score = roc_auc_score(gt_labels, scores)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(gt_labels, scores)

    return auc_score, scores, fpr, tpr, thresholds


def find_threshold(scores_val, percentile=95):
    """
    Find adaptive threshold based on validation set.

    Args:
        scores_val: Validation set scores (should be normal data only)
        percentile: Percentile threshold (default 95th)

    Returns:
        threshold: Anomaly threshold
    """
    threshold = np.percentile(scores_val, percentile)
    return threshold


def evaluate_with_threshold(scores, threshold):
    """
    Evaluate using a fixed threshold.

    Args:
        scores: Array of anomaly scores
        threshold: Threshold value

    Returns:
        predictions: Binary predictions (0=normal, 1=anomaly)
    """
    predictions = (scores > threshold).astype(int)
    return predictions


def plot_roc_curve(fpr, tpr, auc_score, model_name):
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.upper()} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_roc_curve.png', dpi=100, bbox_inches='tight')
    print(f"Saved ROC curve to results/{model_name}_roc_curve.png")
    plt.close()


def plot_error_distribution(scores_normal, scores_anomaly, model_name, threshold=None):
    """Plot distribution of reconstruction errors."""
    plt.figure(figsize=(10, 6))

    plt.hist(scores_normal, bins=50, alpha=0.7, label='Normal', color='blue', edgecolor='black')
    plt.hist(scores_anomaly, bins=50, alpha=0.7, label='Anomaly', color='red', edgecolor='black')

    if threshold is not None:
        plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')

    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Frequency')
    plt.title(f'{model_name.upper()} Error Distribution')
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs('results', exist_ok=True)
    plt.savefig(f'results/{model_name}_error_distribution.png', dpi=100, bbox_inches='tight')
    print(f"Saved error distribution plot to results/{model_name}_error_distribution.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate video anomaly detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/cnn.yaml',
                      help='Path to config file')
    parser.add_argument('--test_dir', type=str, default='data/processed/test',
                      help='Path to test frames directory')
    parser.add_argument('--label_dir', type=str, default='data/test_labels',
                      help='Path to ground truth labels directory')
    parser.add_argument('--device', type=str, default='cpu',
                      choices=['cpu', 'cuda'])

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        config['model'],
        args.checkpoint,
        latent_dim=config['latent_dim'],
        device=device
    )

    # Load test data
    print(f"Loading test data from {args.test_dir}...")

    # Try to load with labels if available
    if os.path.exists(args.label_dir):
        print(f"Loading labels from {args.label_dir}...")
        dataset = SlidingWindowDatasetWithLabels(
            frame_dir=args.test_dir,
            label_dir=args.label_dir,
            window_size=config['window_size'],
            stride=config['stride'],
            use_npy=True
        )

        # Separate used for evaluation
        test_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        # Extract labels
        all_labels = np.array([label for _, label in dataset.clips])

    else:
        print("Warning: No label directory found. Loading without labels...")
        dataset = SlidingWindowDataset(
            frame_dir=args.test_dir,
            window_size=config['window_size'],
            stride=config['stride'],
            use_npy=True
        )

        test_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )

        all_labels = np.zeros(len(dataset))

    print(f"Test set size: {len(dataset)}")

    # Compute anomaly scores
    print("Computing anomaly scores...")
    scores = compute_anomaly_scores(model, test_loader, device)

    # Compute AUC if labels are available
    if len(np.unique(all_labels)) > 1:
        auc_score = roc_auc_score(all_labels, scores)
        print(f"\nAUC-ROC Score: {auc_score:.4f}")

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(all_labels, scores)

        # Plot ROC
        plot_roc_curve(fpr, tpr, auc_score, config['model'])

        # Find threshold based on false positive rate
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer_threshold = thresholds[eer_idx]
        print(f"EER Threshold: {eer_threshold:.6f}")

        # Separate scores into normal and anomaly for visualization
        scores_normal = scores[all_labels == 0]
        scores_anomaly = scores[all_labels == 1]

        # Threshold at 95th percentile of normal
        threshold_95 = np.percentile(scores_normal, 95)
        print(f"95th percentile threshold: {threshold_95:.6f}")

        # Plot error distribution
        plot_error_distribution(scores_normal, scores_anomaly, config['model'], threshold_95)

    else:
        print("No labels available for AUC computation")
        threshold_95 = np.percentile(scores, 95)
        print(f"95th percentile threshold: {threshold_95:.6f}")

    # Save scores
    os.makedirs('results', exist_ok=True)
    np.save(f'results/{config["model"]}_scores.npy', scores)
    print(f"Saved scores to results/{config['model']}_scores.npy")


if __name__ == '__main__':
    main()
