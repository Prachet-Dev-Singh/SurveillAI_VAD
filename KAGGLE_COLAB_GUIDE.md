# Running SurveillAI-VAD on Kaggle & Google Colab

This guide explains how to train and evaluate the SurveillAI-VAD models using cloud GPUs since the project requires significant computational resources for optimal results.

## Understanding Expected Results

The performance metrics shown in the documentation (AUC-ROC 74-88%, inference times) are **theoretical targets based on literature and architecture analysis**, not empirical results from a pre-trained model. They represent:

1. **Baseline expectations** from similar architectures in published research
2. **Projected performance** based on training hyperparameters
3. **Inference benchmarks** calculated from model architecture (not actual runtime)

**To get actual results**, you need to:
1. Train the models on real data (UCSD Ped2 dataset)
2. Evaluate on test set to get true AUC-ROC
3. Benchmark inference time on your hardware

**Why use GPU?**
- **CNN Baseline**: 15 min on CPU → 30 sec on GPU
- **ViT Training**: 2+ hours on CPU → 10-15 min on GPU
- **Mamba Training**: 1.5+ hours on CPU → 12-20 min on GPU

---

## Option 1: Google Colab (Recommended for Beginners)

Google Colab provides free GPU access (NVIDIA Tesla K80/T4).

### Setup (5 minutes)

1. **Open Colab Notebook**
   ```
   https://colab.research.google.com
   ```

2. **Mount GitHub Repo**
   ```python
   !git clone https://github.com/Prachet-Dev-Singh/SurveillAI-VAD.git
   %cd SurveillAI-VAD
   ```

3. **Install Dependencies**
   ```python
   !pip install -q -r requirements.txt
   ```

4. **Check GPU**
   ```python
   !nvidia-smi
   ```

### Full Training Pipeline

```python
# 1. DOWNLOAD DATASET
!bash data/download_ucsd.sh

# 2. PREPROCESS FRAMES
!python data/preprocess.py --dataset ucsd --input data/ucsd/ --output data/processed/

# 3. TRAIN CNN BASELINE (5-10 minutes)
!python train.py --config configs/cnn.yaml --data_dir data/processed/train --device cuda

# 4. TRAIN ViT (15-20 minutes)
!python train_vit.py --config configs/vit.yaml --data_dir data/processed/train --device cuda

# 5. TRAIN ViT WITH DISTILLATION (20-25 minutes)
!python train_vit.py --config configs/vit.yaml --data_dir data/processed/train --device cuda --use_distillation

# 6. EVALUATE ALL MODELS
!python evaluate.py --checkpoint checkpoints/cnn_best.pth --config configs/cnn.yaml --test_dir data/processed/test --device cuda
!python evaluate.py --checkpoint checkpoints/vit_best.pth --config configs/vit.yaml --test_dir data/processed/test --device cuda
!python evaluate.py --checkpoint checkpoints/vitdistill_student_best.pth --config configs/vit.yaml --test_dir data/processed/test --device cuda

# 7. VISUALIZE RESULTS
!python visualize.py --checkpoint checkpoints/cnn_best.pth --frame_dir data/processed/test --output_dir results/heatmaps --num_samples 5 --device cuda

# 8. COMPILE AND VIEW RESULTS
!python compile_results.py --example
```

### Download Results

```python
# Compress results
!zip -r results.zip results/ checkpoints/

# Download
from google.colab import files
files.download('results.zip')
```

### Complete Colab Notebook Template

```python
# SurveillAI-VAD Training on Colab

# GPU Check
!nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Clone and Setup
!git clone https://github.com/Prachet-Dev-Singh/SurveillAI-VAD.git
%cd SurveillAI-VAD
!pip install -q -r requirements.txt

# Dataset
print("Downloading UCSD Ped2 dataset...")
!bash data/download_ucsd.sh 2>/dev/null

print("Preprocessing frames...")
!python data/preprocess.py --dataset ucsd --input data/ucsd/ --output data/processed/ 2>&1 | tail -20

# Training CNN
print("\n" + "="*60)
print("TRAINING CNN BASELINE")
print("="*60)
!python train.py --config configs/cnn.yaml --data_dir data/processed/train --device cuda

# Training ViT
print("\n" + "="*60)
print("TRAINING VIT + TEMPORAL")
print("="*60)
!python train_vit.py --config configs/vit.yaml --data_dir data/processed/train --device cuda

# Evaluation
print("\n" + "="*60)
print("EVALUATING MODELS")
print("="*60)
!python evaluate.py --checkpoint checkpoints/cnn_best.pth --config configs/cnn.yaml --test_dir data/processed/test --device cuda
!python evaluate.py --checkpoint checkpoints/vit_best.pth --config configs/vit.yaml --test_dir data/processed/test --device cuda

# Visualization
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)
!python visualize.py --checkpoint checkpoints/cnn_best.pth --frame_dir data/processed/test --output_dir results/heatmaps --num_samples 5 --device cuda

print("\nTraining complete! Check results/ directory for outputs.")
```

---

## Option 2: Kaggle (More Powerful GPU)

Kaggle provides free NVIDIA Tesla P100 GPU (16GB VRAM).

### Setup in Kaggle

1. **Create New Notebook** on kaggle.com
2. **Enable GPU**
   - Settings → Accelerator → GPU

### Kaggle Training Script

```python
import os
import subprocess
import time

# Check GPU
print("GPU Info:")
os.system("nvidia-smi")

# Clone repo
os.system("git clone https://github.com/Prachet-Dev-Singh/SurveillAI-VAD.git")
os.chdir("SurveillAI-VAD")

# Install dependencies
print("\nInstalling dependencies...")
os.system("pip install -q -r requirements.txt")

# Download dataset
print("\nDownloading UCSD Ped2...")
os.system("bash data/download_ucsd.sh > /dev/null 2>&1")

# Preprocess
print("Preprocessing frames...")
start = time.time()
os.system("python data/preprocess.py --dataset ucsd --input data/ucsd/ --output data/processed/")
print(f"Preprocessing complete in {time.time()-start:.1f}s\n")

# Training
models_to_train = [
    ("CNN Baseline", "configs/cnn.yaml", ""),
    ("ViT + Temporal", "configs/vit.yaml", ""),
    ("ViT + Distillation", "configs/vit.yaml", "--use_distillation"),
]

for name, config, extra_args in models_to_train:
    print("="*60)
    print(f"Training {name}")
    print("="*60)

    cmd = f"python train_vit.py --config {config} --data_dir data/processed/train --device cuda {extra_args}"
    if "cnn" in config:
        cmd = f"python train.py --config {config} --data_dir data/processed/train --device cuda"

    start = time.time()
    os.system(cmd)
    elapsed = time.time() - start
    print(f"\n✓ {name} training complete ({elapsed/60:.1f} min)\n")

# Evaluation
print("="*60)
print("Evaluating Models")
print("="*60)

checkpoints = [
    ("cnn_best.pth", "configs/cnn.yaml"),
    ("vit_best.pth", "configs/vit.yaml"),
]

for ckpt, config in checkpoints:
    if os.path.exists(f"checkpoints/{ckpt}"):
        print(f"\nEvaluating {ckpt}...")
        os.system(
            f"python evaluate.py --checkpoint checkpoints/{ckpt} "
            f"--config {config} --test_dir data/processed/test --device cuda"
        )

# Results
print("\n" + "="*60)
print("Results Summary")
print("="*60)
os.system("ls -lh results/*.csv results/*.png 2>/dev/null | awk '{print $NF, $5}'")
```

---

## Option 3: Local Machine with GPU

If you have a local NVIDIA GPU (RTX 3060 or better):

### Install CUDA and cuDNN

```bash
# Windows: Download from NVIDIA (CUDA 11.8, cuDNN 8.x)
# Linux:
# sudo apt-get install -y nvidia-cuda-toolkit nvidia-cudnn

# Verify
nvidia-smi
```

### Run Training

```bash
python train.py --config configs/cnn.yaml --data_dir data/processed/train --device cuda
python train_vit.py --config configs/vit.yaml --data_dir data/processed/train --device cuda
python evaluate.py --checkpoint checkpoints/cnn_best.pth --test_dir data/processed/test --device cuda
```

---

## Monitoring Training

### Real-time GPU/CPU Stats

```python
# In Colab
!watch -n 1 nvidia-smi  # GPU stats
!htop  # CPU/Memory stats
```

### Loss Curves

```python
# After training, view loss curves
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("results/cnn_training_history.png")
plt.figure(figsize=(12, 4))
plt.imshow(img)
plt.axis('off')
plt.show()
```

---

## Expected Training Times

| Model | CPU | GPU (K80/T4) | GPU (P100) | GPU (RTX 3090) |
|---|---|---|---|---|
| **CNN (50 epochs)** | 15 min | 2-3 min | 1.5 min | 1 min |
| **ViT (100 epochs)** | 120 min | 15-20 min | 10 min | 7 min |
| **ViT+Distill (100 epochs)** | 140 min | 20-25 min | 12 min | 8 min |
| **Mamba (100 epochs)** | 100 min | 12-18 min | 9 min | 6 min |
| **Total Pipeline** | ~260 min | ~50-70 min | ~30-35 min | ~20-25 min |

---

## Output Files to Expect

After completion, you'll have:

```
results/
├── cnn_training_history.png         # Loss curves
├── cnn_roc_curve.png               # ROC curve with AUC score
├── cnn_error_distribution.png      # Anomaly score histogram
├── cnn_scores.npy                  # Per-frame scores (2000+ values)
├── vit_training_history.png
├── vit_roc_curve.png
├── vit_scores.npy
└── comparison_table.csv            # Model comparison

checkpoints/
├── cnn_best.pth                    # Best CNN model (~8MB)
├── vit_best.pth                    # Best ViT model (~90MB)
└── vitdistill_student_best.pth     # Distilled ViT (~65MB)
```

---

## Troubleshooting

### Out of Memory (OOM) Error

```python
# Reduce batch size in config YAML
batch_size: 8  # Instead of 16

# Or use gradient accumulation (modify train.py)
```

### Too Slow on CPU

```python
# Must use GPU
python train.py --config configs/cnn.yaml --device cuda
```

### CUDA Not Found

```python
# Install CUDA/cuDNN or use CPU (slower)
python train.py --config configs/cnn.yaml --device cpu
```

### Dataset Download Fails

```python
# Manual download and extract
# UCSD: http://www.svcl.ucsd.edu/projects/anomaly/UCSD_Anomaly_Dataset.tar.gz
# Extract to: data/ucsd/
```

---

## Next Steps After Training

1. **Download results.zip** from Colab/Kaggle
2. **Analyze metrics** in the CSV and PNG files
3. **Use trained checkpoints** for:
   - Inference on new videos
   - Fine-tuning on custom datasets
   - API deployment

---

**Estimated Total Time (Colab):** 1 hour (including data download and preprocessing)
