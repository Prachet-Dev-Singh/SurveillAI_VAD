import os
import torch
import numpy as np
import cv2
import glob
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Imports from the training scripts
from train import SimpleCNNAutoencoder
from train_vit import ViTWithTemporalModel


from student_loader import DistilledStudentInference

def load_model(checkpoint_path, device='cpu'):
    """Detects model type and loads weights correctly."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    
    # CASE 1: 517KB Student Model
    if 'net.0.weight' in state_dict:
        print(f"Detected 517KB Student Model from {checkpoint_path}")
        model = DistilledStudentInference(latent_dim=256)
        
        # Mapping the tiny student weights
        fixed_state_dict = {}
        for k, v in state_dict.items():
            fixed_state_dict[f'student_brain.{k}'] = v
            
        # Loading Teacher 'Eyes' from the working ViT checkpoint
        try:
            vit_dict = torch.load('checkpoints/vit_best.pth', map_location=device)
            for k, v in vit_dict.items():
                if 'spatial_encoder' in k:
                    fixed_state_dict[k.replace('spatial_encoder.', 'teacher_vision.')] = v
        except:
            print("Warning: vit_best.pth not found. Student will have no 'Eyes'.")
            
        # Loading the Decoder weights saved during distillation
        try:
            dec_dict = torch.load('checkpoints/vitdistill_decoder_best.pth', map_location=device)
            for k, v in dec_dict.items():
                fixed_state_dict[f'decoder.{k}'] = v
        except:
            print("Warning: Decoder weights not found.")

        model.load_state_dict(fixed_state_dict, strict=False)
        
    # CASE 2: Standard ViT Model
    elif 'spatial_encoder.vit.cls_token' in state_dict:
        print(f"Detected ViT Model from {checkpoint_path}")
        model = ViTWithTemporalModel(freeze_blocks=0, use_temporal=True, embed_dim=256)
        model.load_state_dict(state_dict)
        
    # CASE 3: CNN Model
    else:
        print(f"Detected CNN Model from {checkpoint_path}")
        model = SimpleCNNAutoencoder(latent_dim=256)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--frame_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.checkpoint, args.device)

    # Search for frames in subfolders
    frame_files = sorted(glob.glob(os.path.join(args.frame_dir, "**", "*.npy"), recursive=True))
    
    if not frame_files:
        print(f"Error: No frames found in {args.frame_dir}")
        return

    print(f"Found {len(frame_files)} frames. Generating {args.num_samples} visualizations...")
    
    # Pick random samples
    indices = np.random.choice(len(frame_files), min(args.num_samples, len(frame_files)), replace=False)
    
    for i, idx in enumerate(indices):
        frame_path = frame_files[idx]
        frame = np.load(frame_path).astype(np.float32)
        
        # Ensure frame is normalized [0, 1]
        if frame.max() > 1.0: frame /= 255.0
        
        # FIX: The model expects [Batch, Channels, Height, Width] (BCHW)
        # 1. Start with HWC: (224, 224, 3)
        # 2. Permute to CHW: (3, 224, 224)
        # 3. Unsqueeze to BCHW: (1, 3, 224, 224)
        input_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(args.device)
        
        with torch.no_grad():
            reconstructed = model(input_tensor)
            
        # Calculate error map (MSE) across the channel dimension
        error_map = torch.mean((input_tensor - reconstructed)**2, dim=1).squeeze().cpu().numpy()
        
        # Normalize error map for visual contrast
        error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-8)
        
        # Prepare the original image for display (Back to HWC and uint8)
        orig_img = (frame * 255).astype(np.uint8)
        
        # Generate Heatmap and fix color space (OpenCV BGR -> RGB)
        heatmap_bgr = cv2.applyColorMap((error_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        
        # Merge original and heatmap
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_rgb, 0.4, 0)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Original: {os.path.basename(os.path.dirname(frame_path))}")
        plt.imshow(orig_img)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Anomaly Heatmap (Red = High Error)")
        plt.imshow(overlay)
        plt.axis('off')
        
        plt.savefig(os.path.join(args.output_dir, f"sample_{idx}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    main()
