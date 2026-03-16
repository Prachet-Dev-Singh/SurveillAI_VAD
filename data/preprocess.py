import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import glob

def extract_and_save_frames_from_dir(folder_path, output_dir, video_name, image_size=224):
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    # Grab all .tif images in the folder
    frames = sorted(glob.glob(os.path.join(folder_path, '*.tif')))
    if not frames:
        return 0

    frame_count = 0
    for frame_path in frames:
        frame = cv2.imread(frame_path)
        if frame is None: continue
        
        # Resize, recolor, and normalize just like the original script
        frame = cv2.resize(frame, (image_size, image_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0
        
        # Save as .npy
        out_path = os.path.join(video_output_dir, f'{frame_count:06d}.npy')
        np.save(out_path, frame)
        frame_count += 1
        
    return frame_count

def preprocess_ucsd_ped2(input_dir, output_dir, image_size=224):
    print("Preprocessing UCSD Ped2 dataset (Image Sequence version)...")
    
    for split in ['Train', 'Test']:
        split_out = 'train' if split == 'Train' else 'test'
        split_output_dir = os.path.join(output_dir, split_out)
        os.makedirs(split_output_dir, exist_ok=True)

        split_dir = os.path.join(input_dir, split)
        if os.path.exists(split_dir):
            # Look for subdirectories instead of .avi files
            video_folders = sorted([f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))])
            print(f"Found {len(video_folders)} {split_out} sequences")

            for folder_name in tqdm(video_folders, desc=f"Processing {split_out}"):
                folder_path = os.path.join(split_dir, folder_name)
                extract_and_save_frames_from_dir(folder_path, split_output_dir, folder_name, image_size)
                
    print("Preprocessing complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ucsd')
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()

    if args.dataset == 'ucsd':
        preprocess_ucsd_ped2(args.input, args.output, args.image_size)

if __name__ == '__main__':
    main()
