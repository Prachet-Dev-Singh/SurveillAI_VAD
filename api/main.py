"""
FastAPI inference server for video anomaly detection.
Allows running inference on video clips through HTTP endpoints.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import torch
import numpy as np
import cv2
import io
import tempfile
from pathlib import Path
import yaml
import json
from typing import List, Optional
import uvicorn

from models.cnn_autoencoder import SimpleCNNAutoencoder
from visualize import apply_colormap_to_heatmap

# Initialize FastAPI app
app = FastAPI(
    title="SurveillAI-VAD",
    description="Spatial-Temporal Anomaly Detection for Surveillance Video",
    version="1.0.0"
)

# Global variables
MODEL = None
DEVICE = None
THRESHOLD = None


def load_model(checkpoint_path: str, model_type: str = 'cnn', device: str = 'cpu'):
    """Load trained model."""
    global MODEL, DEVICE

    DEVICE = torch.device(device)

    if model_type == 'cnn':
        MODEL = SimpleCNNAutoencoder()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    MODEL.load_state_dict(state_dict)
    MODEL.to(DEVICE)
    MODEL.eval()

    print(f"Model loaded from {checkpoint_path}")


def frame_to_tensor(frame_np: np.ndarray) -> torch.Tensor:
    """Convert numpy frame to torch tensor."""
    # Ensure float32 and normalize
    if frame_np.dtype != np.float32:
        frame_np = frame_np.astype(np.float32)

    if frame_np.max() > 1.0:
        frame_np = frame_np / 255.0

    # CHW format
    if frame_np.shape[0] != 3:
        frame_np = np.transpose(frame_np, (2, 0, 1))

    tensor = torch.from_numpy(frame_np).unsqueeze(0).to(DEVICE)
    return tensor


def process_frame(frame_np: np.ndarray) -> tuple:
    """
    Process a single frame through the model.

    Returns:
        (anomaly_score, heatmap, reconstructed)
    """
    # Resize if needed
    if frame_np.shape[:2] != (224, 224):
        frame_np = cv2.resize(frame_np, (224, 224))

    # Normalize
    if frame_np.shape[0] != 224:
        frame_np = np.transpose(frame_np, (1, 2, 0))

    frame_normalized = frame_np / 255.0 if frame_np.max() > 1 else frame_np

    # To tensor
    tensor = frame_to_tensor(frame_np)

    # Forward pass
    with torch.no_grad():
        reconstructed_tensor = MODEL(tensor)

    reconstructed = reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Compute anomaly score
    anomaly_score = float(np.mean((frame_normalized - reconstructed) ** 2))

    # Compute heatmap
    error_map = np.mean((frame_normalized - reconstructed) ** 2, axis=2)
    heatmap = error_map / (error_map.max() + 1e-6)

    return anomaly_score, heatmap, reconstructed


def process_video(video_path: str, frame_stride: int = 1) -> dict:
    """
    Process a video file frame by frame.

    Args:
        video_path: Path to video file
        frame_stride: Process every Nth frame

    Returns:
        Dict with frame scores, anomalies, and statistics
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    results = {
        'frame_scores': [],
        'anomaly_frames': [],
        'statistics': {},
        'threshold_used': float(THRESHOLD) if THRESHOLD else None,
    }

    frame_count = 0
    anomaly_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_stride != 0:
            frame_count += 1
            continue

        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            score, heatmap, reconstructed = process_frame(frame_rgb)

            results['frame_scores'].append({
                'frame_number': frame_count,
                'anomaly_score': float(score),
                'is_anomaly': bool(score > THRESHOLD) if THRESHOLD else None
            })

            if THRESHOLD and score > THRESHOLD:
                anomaly_count += 1
                results['anomaly_frames'].append(frame_count)

        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")

        frame_count += 1

    cap.release()

    # Statistics
    scores = [f['anomaly_score'] for f in results['frame_scores']]
    results['statistics'] = {
        'total_frames': frame_count,
        'processed_frames': len(results['frame_scores']),
        'anomaly_count': anomaly_count,
        'anomaly_percentage': 100 * anomaly_count / len(results['frame_scores']) if results['frame_scores'] else 0,
        'mean_score': float(np.mean(scores)) if scores else 0,
        'std_score': float(np.std(scores)) if scores else 0,
        'min_score': float(np.min(scores)) if scores else 0,
        'max_score': float(np.max(scores)) if scores else 0,
    }

    return results


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    print("SurveillAI-VAD API starting up...")
    # Model will be loaded with the /init endpoint
    print("Use POST /init to load a model checkpoint")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "SurveillAI-VAD API",
        "description": "Spatial-Temporal Anomaly Detection for Surveillance Video",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "init": "POST /init",
            "frame": "POST /frame",
            "video": "POST /video",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else None
    }


@app.post("/init")
async def initialize(
    checkpoint_path: str,
    model_type: str = "cnn",
    device: str = "cpu",
    threshold: Optional[float] = None
):
    """
    Initialize the model.

    Args:
        checkpoint_path: Path to model checkpoint
        model_type: Type of model (cnn, vit, mamba)
        device: Device to use (cpu, cuda)
        threshold: Anomaly threshold (optional)

    Returns:
        Status message
    """
    global THRESHOLD

    try:
        load_model(checkpoint_path, model_type, device)
        THRESHOLD = threshold

        model_size = sum(p.numel() for p in MODEL.parameters())

        return {
            "status": "success",
            "message": "Model loaded successfully",
            "model_type": model_type,
            "device": str(DEVICE),
            "parameters": model_size,
            "threshold": THRESHOLD
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/frame")
async def process_frame_endpoint(file: UploadFile = File(...)):
    """
    Process a single frame.

    Args:
        file: Image file (PNG, JPG, etc.)

    Returns:
        Anomaly score and visualization
    """
    if MODEL is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Use POST /init first")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Could not decode image")

        # Process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        score, heatmap, reconstructed = process_frame(frame_rgb)

        # Create visualization
        visualization = apply_colormap_to_heatmap(heatmap, frame_rgb / 255.0, alpha=0.6)

        # Save visualization
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            cv2.imwrite(tmp.name, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            tmp_path = tmp.name

        return {
            "anomaly_score": float(score),
            "is_anomaly": bool(score > THRESHOLD) if THRESHOLD else None,
            "threshold": float(THRESHOLD) if THRESHOLD else None,
            "frame_shape": frame_rgb.shape,
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/video")
async def process_video_endpoint(
    file: UploadFile = File(...),
    frame_stride: int = 1
):
    """
    Process a video file.

    Args:
        file: Video file (MP4, AVI, etc.)
        frame_stride: Process every Nth frame

    Returns:
        Anomaly scores for each frame
    """
    if MODEL is None:
        raise HTTPException(status_code=400, detail="Model not initialized. Use POST /init first")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        # Process video
        results = process_video(tmp_path, frame_stride=frame_stride)

        # Clean up
        Path(tmp_path).unlink()

        return results

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/results")
async def get_results_example():
    """Return example results structure."""
    return {
        "frame_scores": [
            {"frame_number": 0, "anomaly_score": 0.001, "is_anomaly": False},
            {"frame_number": 10, "anomaly_score": 0.015, "is_anomaly": True},
        ],
        "anomaly_frames": [10],
        "statistics": {
            "total_frames": 100,
            "processed_frames": 100,
            "anomaly_count": 1,
            "anomaly_percentage": 1.0,
            "mean_score": 0.002,
            "std_score": 0.005,
            "min_score": 0.0001,
            "max_score": 0.05,
        },
        "threshold_used": 0.01
    }


if __name__ == "__main__":
    # Example usage
    print("Starting SurveillAI-VAD API...")
    print("API will be available at http://localhost:8000")
    print("API Docs: http://localhost:8000/docs")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
