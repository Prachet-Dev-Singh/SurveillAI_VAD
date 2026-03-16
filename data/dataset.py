"""
PyTorch Dataset class for video anomaly detection.
Handles sliding window sampling of frames.
"""

import os
import numpy as np
from pathlib import Path
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image


class SlidingWindowDataset(Dataset):
    """
    Dataset that creates sliding windows of consecutive frames.

    Input: Directory of frame files (either .npy or .jpg)
    Output: Tensor of shape (N_frames, C, H, W)
    """

    def __init__(self, frame_dir, window_size=8, stride=4, transform=None, use_npy=True):
        """
        Args:
            frame_dir: Root directory containing subdirectories of frames per video
            window_size: Number of consecutive frames per clip
            stride: Step size for sliding window
            transform: Optional torchvision transforms
            use_npy: If True, load from .npy files; otherwise load from .jpg
        """
        self.frame_dir = frame_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.use_npy = use_npy
        self.clips = []

        self._build_clips()

    def _build_clips(self):
        """Build list of (video_dir, start_frame_idx) tuples."""
        # Find all video subdirectories
        video_dirs = sorted([d for d in glob(os.path.join(self.frame_dir, '*'))
                           if os.path.isdir(d)])

        for video_dir in video_dirs:
            # Find all frames in this video directory
            if self.use_npy:
                frames = sorted(glob(os.path.join(video_dir, '*.npy')))
            else:
                frames = sorted(glob(os.path.join(video_dir, '*.jpg')))

            # Create sliding windows
            num_frames = len(frames)
            if num_frames < self.window_size:
                continue

            for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
                self.clips.append((video_dir, start_idx))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_dir, start_idx = self.clips[idx]

        # Load frame paths
        if self.use_npy:
            frames = sorted(glob(os.path.join(video_dir, '*.npy')))
        else:
            frames = sorted(glob(os.path.join(video_dir, '*.jpg')))

        # Get window of frames
        frame_paths = frames[start_idx:start_idx + self.window_size]

        # Load frames
        clip = []
        for frame_path in frame_paths:
            if self.use_npy:
                frame = np.load(frame_path)  # Already (H, W, C) and normalized
            else:
                frame = Image.open(frame_path).convert('RGB')
                frame = np.array(frame, dtype=np.float32) / 255.0

            # Apply transforms if provided
            if self.transform:
                frame = self.transform(frame)
            else:
                # Convert to torch tensor and rearrange to (C, H, W)
                if isinstance(frame, np.ndarray):
                    frame = torch.from_numpy(frame)
                    if frame.dim() == 3 and frame.shape[2] == 3:  # (H, W, C)
                        frame = frame.permute(2, 0, 1)

            clip.append(frame)

        return torch.stack(clip)  # (N_frames, C, H, W)


class SlidingWindowDatasetWithLabels(Dataset):
    """
    Dataset for test set that includes anomaly labels.

    Assumes label file format:
    test_label_dir/
        video_name.npy  -- array of shape (num_frames,) with 0 (normal) or 1 (anomaly)
    """

    def __init__(self, frame_dir, label_dir, window_size=8, stride=4,
                 transform=None, use_npy=True):
        """
        Args:
            frame_dir: Root directory containing frame subdirectories
            label_dir: Directory containing label .npy files
            window_size: Number of consecutive frames per clip
            stride: Step size for sliding window
            transform: Optional transforms
            use_npy: Whether frames are stored as .npy
        """
        self.frame_dir = frame_dir
        self.label_dir = label_dir
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.use_npy = use_npy
        self.clips = []
        self.labels = {}

        self._load_labels()
        self._build_clips()

    def _load_labels(self):
        """Load anomaly labels from file."""
        label_files = glob(os.path.join(self.label_dir, '*.npy'))
        for label_file in label_files:
            video_name = os.path.basename(label_file).replace('.npy', '')
            self.labels[video_name] = np.load(label_file)

    def _build_clips(self):
        """Build list of clips with their labels."""
        video_dirs = sorted([d for d in glob(os.path.join(self.frame_dir, '*'))
                           if os.path.isdir(d)])

        for video_dir in video_dirs:
            video_name = os.path.basename(video_dir)

            if self.use_npy:
                frames = sorted(glob(os.path.join(video_dir, '*.npy')))
            else:
                frames = sorted(glob(os.path.join(video_dir, '*.jpg')))

            num_frames = len(frames)
            if num_frames < self.window_size:
                continue

            # Get labels for this video
            if video_name in self.labels:
                video_labels = self.labels[video_name]
            else:
                video_labels = np.zeros(num_frames, dtype=np.int64)

            for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                # Use label of middle frame
                middle_frame_idx = start_idx + self.window_size // 2
                if middle_frame_idx < len(video_labels):
                    label = video_labels[middle_frame_idx]
                else:
                    label = 0
                self.clips.append((video_dir, start_idx, label))

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        video_dir, start_idx, label = self.clips[idx]

        if self.use_npy:
            frames = sorted(glob(os.path.join(video_dir, '*.npy')))
        else:
            frames = sorted(glob(os.path.join(video_dir, '*.jpg')))

        frame_paths = frames[start_idx:start_idx + self.window_size]

        clip = []
        for frame_path in frame_paths:
            if self.use_npy:
                frame = np.load(frame_path)
            else:
                frame = Image.open(frame_path).convert('RGB')
                frame = np.array(frame, dtype=np.float32) / 255.0

            if self.transform:
                frame = self.transform(frame)
            else:
                if isinstance(frame, np.ndarray):
                    frame = torch.from_numpy(frame)
                    if frame.dim() == 3 and frame.shape[2] == 3:
                        frame = frame.permute(2, 0, 1)

            clip.append(frame)

        return torch.stack(clip), label  # (N_frames, C, H, W), label
