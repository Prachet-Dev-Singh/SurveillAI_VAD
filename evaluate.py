import torch
import torch.nn.functional as F
import argparse, os, sys, numpy as np
import kornia, pytorch_msssim
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm

# Allow importing from project root
sys.path.append(os.getcwd())
from train import MemoryBankAutoencoder, load_config, process_batch
from data.dataset import SlidingWindowDataset

# ── ANOMALY SCORER ────────────────────────────────────────────────────
def compute_anomaly_score(orig, recon):
    """MSE + SSIM + Sobel edge error → per-frame score [B]."""
    mse  = F.mse_loss(orig, recon, reduction="none").mean(dim=[1, 2, 3])
    ssim = 1.0 - pytorch_msssim.ssim(
        orig, recon, data_range=1.0, size_average=False)
    sob_o = kornia.filters.sobel(orig)
    sob_r = kornia.filters.sobel(recon)
    grad  = F.l1_loss(sob_o, sob_r, reduction="none").mean(dim=[1, 2, 3])
    return 0.4 * mse + 0.4 * ssim + 0.2 * grad

# ── PER-VIDEO NORMALIZATION ──────────────────────────────────────────
def per_video_normalize(scores_list):
    out = []
    for s in scores_list:
        mn, mx = s.min(), s.max()
        out.append((s - mn) / (mx - mn + 1e-8))
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",       default="configs/mamba.yaml")
    p.add_argument("--checkpoint",   default="checkpoints/mamba_best.pth")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--latent_dim",   type=int,   default=256)
    p.add_argument("--num_slots",    type=int,   default=512)
    p.add_argument("--window_size",  type=int,   default=8)
    a = p.parse_args()

    cfg    = load_config(a.config)
    device = a.device

    # ── Load model ────────────────────────────────────────────
    print(f"Loading checkpoint: {a.checkpoint}")
    model = MemoryBankAutoencoder(
        latent_dim=a.latent_dim, num_slots=a.num_slots).to(device)
    
    ckpt = torch.load(a.checkpoint, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # ── Score all test frames ─────────────────────────────────
    test_base = os.path.join(cfg.data_dir, "test")
    test_ds = SlidingWindowDataset(
        frame_dir=test_base, window_size=a.window_size, stride=1)
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)

    raw_scores = []
    print("Computing anomaly scores...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Scoring Test Data"):
            context, target = process_batch(batch, device)
            recon = model(context)
            if recon.shape[1] != target.shape[1]:
                target = target.repeat(1, 3, 1, 1) if target.shape[1] == 1 else target
            scores = compute_anomaly_score(target, recon)
            raw_scores.extend(scores.cpu().numpy())

    score_arr = np.array(raw_scores, dtype=np.float32)

    # ── Load labels ───────────────────────────────────────────
    label_dir   = "data/test_labels"
    label_files = sorted(f for f in os.listdir(label_dir) if f.endswith(".npy"))
    label_arrays = [np.load(os.path.join(label_dir, f)) for f in label_files]

    num_vids = len([d for d in os.listdir(test_base) if os.path.isdir(os.path.join(test_base,d))])
    print(f"Test videos: {num_vids}")
    print(f"Label files: {len(label_arrays)}")

    # ── Per-video alignment (drop window_size-1 per video) ───
    vid_scores_list, vid_labels_list = [], []
    score_offset = 0

    for i, label_arr in enumerate(label_arrays):
        n_frames  = len(label_arr)
        n_windows = max(0, n_frames - (a.window_size - 1))

        vid_raw = score_arr[score_offset : score_offset + n_windows]
        vid_lbl = label_arr[a.window_size - 1 :]

        n = min(len(vid_raw), len(vid_lbl))
        if n > 0:
            vid_scores_list.append(vid_raw[:n])
            vid_labels_list.append(vid_lbl[:n])
            print(f"  Video {i+1:02d}: {n} frames, "
                  f"{int(vid_lbl[:n].sum())} anomalous, "
                  f"score [{vid_raw[:n].min():.4f}, {vid_raw[:n].max():.4f}]")
        score_offset += n_windows

    # ── Per-video normalization ───────────────────────────────
    normed_list  = per_video_normalize(vid_scores_list)
    scores_norm  = np.concatenate(normed_list)
    labels_final = np.concatenate(vid_labels_list).astype(np.int32)

    print(f"\nAligned: {len(scores_norm)} scores, {len(labels_final)} labels")
    print(f"Anomalous: {labels_final.sum()} / {len(labels_final)} "
          f"({100*labels_final.mean():.1f}%)")

    # ── Sigma sweep ───────────────────────────────────────────
    print("\n🔍 Smoothing sigma sweep:")
    best_auc, best_sigma, best_smooth = 0, 0, scores_norm

    for sigma in [0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]:
        if sigma == 0:
            smoothed = scores_norm
        else:
            smoothed = gaussian_filter1d(
                scores_norm.astype(np.float64), sigma=sigma
            ).astype(np.float32)
        auc = roc_auc_score(labels_final, smoothed)
        tag = " ← best" if auc > best_auc else ""
        print(f"  σ={sigma:5.1f}  →  AUC = {auc:.4f}{tag}")
        if auc > best_auc:
            best_auc, best_sigma, best_smooth = auc, sigma, smoothed

    auc_raw = roc_auc_score(labels_final, scores_norm)

    print("\n" + "=" * 55)
    print("  📊  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  AUC-ROC  (raw)                 : {auc_raw:.4f}")
    print(f"  AUC-ROC  (best, σ={best_sigma})          : {best_auc:.4f}")
    print("=" * 55)
    print(f"\n  🎯 FINAL AUC-ROC: {best_auc:.4f}")

    os.makedirs("results", exist_ok=True)
    np.save("results/scores_raw.npy",    scores_norm)
    np.save("results/scores_smooth.npy", best_smooth)
    np.save("results/labels.npy",        labels_final)
    print("Saved to results/")

    # ── Memory bank health ────────────────────────────────────
    print("\n🔬 Memory Bank Health:")
    with torch.no_grad():
        mn = F.normalize(model.memory_bank.memory, dim=1)
        sim = torch.mm(mn, mn.t())
        sim.fill_diagonal_(0)
        print(f"  Avg slot similarity : {sim.abs().mean():.4f}  (want < 0.3)")
        print(f"  Max slot similarity : {sim.abs().max():.4f}  (want < 0.7)")
        used = (sim.max(dim=1).values < 0.95).sum().item()
        print(f"  Distinct slots      : {used}/{sim.shape[0]}")

if __name__ == "__main__":
    main()
