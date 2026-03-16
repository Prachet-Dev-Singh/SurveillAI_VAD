# SurveillAI-VAD: Spatial-Temporal Anomaly Detection for Surveillance Video

A research-grade implementation of unsupervised video anomaly detection for CCTV surveillance. This project compares three architectural approaches: CNN Autoencoder (baseline), Vision Transformer with Temporal Attention, and VideoMamba with linear-complexity SSMs.

## Project Overview

**Core Idea:** Train models on normal surveillance footage only. At test time, frames the model cannot reconstruct well are flagged as anomalies.

**Why This Approach:**
- Labelled anomaly data is almost never available in production
- Anomaly types are open-ended (can't enumerate them in advance)
- Normal behaviour is much easier to characterise and collect

## Architecture Comparison

| Model | AUC-ROC | Params | Inference (ms/frame) |
|---|---|---|---|
| CNN Autoencoder (baseline) | ~74–78% | ~2M | ~5ms |
| ViT-S + Temporal Attn | ~86–89% | ~22M | ~30ms |
| ViT-S + Temporal + Distillation | ~85–88% | ~16M | ~23ms |
| VideoMamba (MambaVision-T) | ~85–88% | ~8M | ~15ms |

## Installation

### 1. Clone and Setup Environment

```bash
git clone https://github.com/your-username/surveillai-vad
cd surveillai-vad
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Dataset

Download UCSD Ped2:
```bash
bash data/download_ucsd.sh
```

Or manually download from: http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz

## Quick Start

### Step 1: Preprocess Dataset

```bash
python data/preprocess.py --dataset ucsd --input data/ucsd/ --output data/processed/
```

This extracts frames, resizes to 224×224, and normalizes to [0,1] range.

### Step 2: Train CNN Baseline

```bash
python train.py --config configs/cnn.yaml --data_dir data/processed/train
```

Expected results:
- Training time: ~10 minutes on CPU
- Final loss: ~0.01
- Best validation loss saved automatically

### Step 3: Train ViT Model

```bash
python train.py --config configs/vit.yaml --data_dir data/processed/train
```

### Step 4: Evaluate Models

```bash
python evaluate.py --checkpoint checkpoints/cnn_best.pth --config configs/cnn.yaml --test_dir data/processed/test
```

This computes:
- AUC-ROC score on test set
- Reconstruction error distribution
- ROC curve visualization

## Project Structure

```
surveillai-vad/
├── data/
│   ├── preprocess.py          # Frame extraction and normalisation
│   ├── dataset.py             # PyTorch Dataset with sliding windows
│   └── download_ucsd.sh       # Dataset download script
│
├── models/
│   ├── cnn_autoencoder.py     # CNN baseline
│   ├── vit_branch.py          # ViT spatial encoder
│   ├── temporal_transformer.py # Temporal attention mechanism
│   ├── self_distillation.py   # Model compression via distillation
│   ├── mamba_branch.py        # VideoMamba efficient backbone
│   └── decoder.py             # Shared reconstruction decoder
│
├── train.py                   # Main training script
├── evaluate.py                # Evaluation and metrics
├── visualize.py               # Anomaly heatmap generation
│
├── configs/
│   ├── cnn.yaml
│   ├── vit.yaml
│   └── mamba.yaml
│
├── results/                   # Outputs: plots, scores, results tables
├── checkpoints/               # Saved model weights
└── requirements.txt
```

## Key Concepts

### 1. Reconstruction-Based Anomaly Detection

Train an autoencoder on normal frames only. The anomaly score for a frame is its reconstruction error (MSE).

```python
anomaly_score = mean_squared_error(original_frame, reconstructed_frame)
```

### 2. Sliding Window Dataset

Videos are split into overlapping N-frame clips:
```
Frame: 0  1  2  3  4  5  6  7  8 ...
Clip1: [0, 1, 2, 3, 4, 5, 6, 7]
Clip2:       [4, 5, 6, 7, 8, 9, ...]
```

Window size = 8, stride = 4 (overlapping)

### 3. Evaluation Metrics

- **AUC-ROC:** How well anomaly scores separate normal from anomalous frames
- **Threshold:** Set at 95th percentile of validation error distribution
- **Frame-level evaluation:** Each frame classified independently

## Technical Details

### CNN Baseline Architecture

```
Input (3, 224, 224)
    ↓
Conv(3→32) + ReLU + Conv(32→64) + ReLU
    ↓
Conv(64→128) + ReLU + AvgPool
    ↓
Bottleneck: 128 → 256 (latent dim)
    ↓
Linear(256 → 128*28*28)
    ↓
ConvTranspose(128→64) + ConvTranspose(64→32) + ConvTranspose(32→3)
    ↓
Output (3, 224, 224) [Sigmoid]
```

### ViT Branch with Temporal Attention

- **Spatial Encoder:** Pretrained ViT-S/16 (freeze first 8 blocks, fine-tune last 4)
- **Temporal Transformer:** 2-layer Transformer over N frame embeddings
- **Self-Distillation:** Student network mimics teacher ViT features (25% parameter reduction)

### VideoMamba Branch (Efficient Alternative)

- **Backbone:** VideoMamba / MambaVision from HuggingFace
- **O(n) complexity:** Handles long sequences efficiently vs O(n²) for Transformers
- **Inference:** 2-4x faster than ViT at comparable AUC

## Training Tips

1. **GPU vs CPU:** The training script defaults to CPU. To use GPU:
   ```bash
   python train.py --config configs/cnn.yaml --data_dir data/processed/train --device cuda
   ```

2. **Hyperparameters:** Modify `configs/model_name.yaml`
   - `batch_size`: Start with 16, increase to 32 if GPU memory allows
   - `lr`: 1e-3 for CNN, 1e-4 for ViT
   - `epochs`: CNN trains in ~50 epochs, ViT needs ~100

3. **Data Preparation:** Make sure frames are stored as `.npy` files:
   ```bash
   data/processed/train/
       Train_001/
           000000.npy
           000001.npy
           ...
       Train_002/
           ...
   ```

## Results & Output

After training, you'll get:

1. **Checkpoints:** `checkpoints/model_name_best.pth`
2. **Plots:**
   - `results/model_name_training_history.png` — Loss curves
   - `results/model_name_roc_curve.png` — ROC curve
   - `results/model_name_error_distribution.png` — Error histogram

3. **Scores:** `results/model_name_scores.npy` — Per-frame anomaly scores

## License

MIT License - See LICENSE file for details

---

**Author:** Prachet Dev Singh
**Date:** 2025
