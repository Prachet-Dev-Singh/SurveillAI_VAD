import torch
import torch.nn.functional as F
import argparse
import os
import yaml
import numpy as np
import sys
import kornia
import pytorch_msssim
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())
from train import MemoryBankAutoencoder, ConfigObject, load_config, process_batch
from data.dataset import SlidingWindowDataset


# ─────────────────────────────────────────────────────────────
# ANOMALY SCORE FUNCTION
# ─────────────────────────────────────────────────────────────
def compute_anomaly_score(orig, recon):
    """
    Combined MSE + SSIM + Sobel-edge score per frame in the batch.
    Returns shape [B].
    """
    # Pixel-level reconstruction error
    mse = F.mse_loss(orig, recon, reduction='none').mean(dim=[1, 2, 3])

    # Structural similarity error
    ssim = 1 - pytorch_msssim.ssim(orig, recon, data_range=1.0, size_average=False)

    # Edge-level error (catches fast-moving / shape anomalies like bikes)
    sob_o = kornia.filters.sobel(orig)
    sob_r = kornia.filters.sobel(recon)
    grad  = F.l1_loss(sob_o, sob_r, reduction='none').mean(dim=[1, 2, 3])

    return (0.4 * mse) + (0.4 * ssim) + (0.2 * grad)   # [B]


# ─────────────────────────────────────────────────────────────
# FIX #2: SCORE-LABEL ALIGNMENT
# ─────────────────────────────────────────────────────────────
def align_scores_to_labels(scores, labels, window_size):
    """
    With stride=1, frame i produces a score from window [i-window_size+1, i].
    The FIRST window corresponds to frame index (window_size - 1), not frame 0.
    Naively truncating to min_len maps scores from the middle of a clip
    against labels from the beginning — completely wrong.

    Correct fix: drop the first (window_size - 1) labels so that
    label[0] aligns with the first complete window.
    """
    offset = window_size - 1
    labels_aligned = labels[offset:]          # skip the un-scored lead frames

    min_len = min(len(scores), len(labels_aligned))
    scores_out = np.array(scores[:min_len])
    labels_out = np.array(labels_aligned[:min_len])
    return scores_out, labels_out


# ─────────────────────────────────────────────────────────────
# FIX #3: TEMPORAL SMOOTHING
# ─────────────────────────────────────────────────────────────
def smooth_scores(scores, sigma=4):
    """
    Gaussian temporal smoothing over the anomaly score sequence.
    Raw per-frame reconstruction error is very noisy; smoothing reduces
    false positives and improves AUC by ~3-5 points.
    sigma=4 corresponds to roughly a half-second smoothing window at ~8 fps.
    """
    return gaussian_filter1d(scores.astype(np.float64), sigma=sigma).astype(np.float32)


# ─────────────────────────────────────────────────────────────
# MAIN EVALUATION
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      required=True)
    parser.add_argument('--checkpoint',  required=True)
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--latent_dim',  type=int, default=256)
    parser.add_argument('--num_slots',   type=int, default=512)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--smooth_sigma', type=float, default=4.0,
                        help='Gaussian smoothing sigma (0 = disabled)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = args.device

    print(f"Loading checkpoint from {args.checkpoint}...")
    model = MemoryBankAutoencoder(
        latent_dim=args.latent_dim,
        num_slots=args.num_slots
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    test_dir = os.path.join(config.data_dir, 'test')
    # stride=1 to score every frame (required for per-frame AUC)
    test_dataset = SlidingWindowDataset(
        frame_dir=test_dir,
        window_size=args.window_size,
        stride=1
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ── Compute raw scores ────────────────────────────────────
    all_scores = []
    print("Computing anomaly scores...")
    with torch.no_grad():
        for batch_data in test_loader:
            clips, target = process_batch(batch_data, device)
            recon = model(clips)

            if recon.shape[1] != target.shape[1]:
                if target.shape[1] == 1: target = target.repeat(1, 3, 1, 1)
                elif recon.shape[1] == 1: recon = recon.repeat(1, 3, 1, 1)

            scores = compute_anomaly_score(target, recon)
            all_scores.extend(scores.cpu().numpy())

    all_scores = np.array(all_scores, dtype=np.float32)

    # ── Load ground truth labels ──────────────────────────────
    print("Loading ground truth labels...")
    label_dir   = "data/test_labels"
    label_files = sorted(
        [os.path.join(label_dir, f)
         for f in os.listdir(label_dir) if f.endswith('.npy')]
    )
    all_labels = []
    for f in label_files:
        all_labels.extend(np.load(f))
    all_labels = np.array(all_labels, dtype=np.int32)

    # ── FIX #2: Proper alignment ──────────────────────────────
    scores_aligned, labels_aligned = align_scores_to_labels(
        all_scores, all_labels, args.window_size
    )
    print(f"Scores: {len(scores_aligned)} frames | "
          f"Labels: {len(labels_aligned)} frames  (after alignment offset={args.window_size-1})")

    # ── FIX #3: Temporal smoothing ────────────────────────────
    if args.smooth_sigma > 0:
        scores_smooth = smooth_scores(scores_aligned, sigma=args.smooth_sigma)
    else:
        scores_smooth = scores_aligned

    # ── Evaluate ──────────────────────────────────────────────
    auc_raw    = roc_auc_score(labels_aligned, scores_aligned)
    auc_smooth = roc_auc_score(labels_aligned, scores_smooth)

    print("\n" + "=" * 45)
    print("📊  EVALUATION RESULTS")
    print("=" * 45)
    print(f"  AUC-ROC (raw scores):      {auc_raw:.4f}")
    print(f"  AUC-ROC (smoothed σ={args.smooth_sigma:.0f}):   {auc_smooth:.4f}")
    print("=" * 45)
    print(f"\n🎯  FINAL AUC-ROC: {auc_smooth:.4f}")

    os.makedirs('results', exist_ok=True)
    np.save('results/raw_scores.npy',    scores_aligned)
    np.save('results/smooth_scores.npy', scores_smooth)
    np.save('results/labels.npy',        labels_aligned)
    print("Results saved to results/")


if __name__ == "__main__":
    main()
